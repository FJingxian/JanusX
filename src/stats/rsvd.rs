//! Randomized SVD kernels for additive genotype matrices on the centered-GRM scale.
//!
//! Let `X ∈ R^{m×n}` denote the marker-by-sample genotype matrix after per-marker
//! mean imputation and major-allele harmonization, with row `j`
//!
//! `x_j = g_j - 2 p_j`.
//!
//! The PCA target used by this module is the centered genomic relationship matrix
//!
//! `K = X^T X / (sum_j 2 p_j (1 - p_j))`.
//!
//! This module provides packed-BED randomized range-finder kernels for the sample-side
//! spectrum of `K` without materializing `X` or `K` densely.
//!
//! Given target rank `k` and oversampled subspace dimension `k' ≥ k`, the
//! approximation follows:
//!
//! 1. Draw a Gaussian test matrix `Ω`.
//! 2. Form block-streamed products such as `X^T Ω`, `X^T (XQ)`, or the projected Gram terms.
//! 3. Apply optional power iterations through repeated `X^T(X·)` products.
//! 4. Orthonormalize the sketch to obtain `Q`.
//! 5. Form the reduced Gram matrix, solve its symmetric eigendecomposition,
//!    and rescale singular values to the `K` eigenvalue scale.
//!
//! The implementation streams packed genotype rows in blocks, decodes each block
//! into a temporary dense centered matrix, and dispatches the linear algebra through BLAS.
//! Row-wise allele-frequency statistics are computed once and then reused across
//! the subsequent randomized projection kernels.

use memmap2::Mmap;
use numpy::ndarray::Array1;
use rayon::ThreadPoolBuilder;
use std::convert::TryFrom;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::mem::align_of;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::UNIX_EPOCH;
use crate::gfcore::{process_snp_row, read_fam, BedSnpIter, HmpSnpIter, TxtSnpIter, VcfSnpIter};
use crate::gload::{load_file_owned_exact, mmap_readonly_file, open_plink_bed_reader_after_header, read_plink_bed_header_info};
use crate::stats_common::process_memory_usage;
use nalgebra::{DMatrix, SymmetricEigen};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::bedmath::{is_identity_indices, packed_byte_lut, SubsetDecodePlan};
use crate::blas::{
    cblas_sgemm_dispatch, BlasThreadGuard, CblasInt, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use crate::decode::{
    decode_prepared_additive_block_packed_f32, prepare_packed_block_centered_mean_scale_f32,
    prepare_packed_block_row_indices,
};
use crate::gfcore::{
    block_rows_from_memory_target_mb, parse_positive_env_f64, parse_positive_env_usize,
};
use crate::he::{row_major_block_mul_mat_f32, row_major_block_t_mul_mat_accum_f32};
use crate::pipeline::run_double_buffer;
use crate::stats_common::{
    admx_madvise_dontneed_bytes, check_admx_memory_limit, check_ctrlc, get_cached_pool,
    map_err_string_to_py, parse_index_vec_i64_result,
};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RsvdKernelTiming {
    pub(crate) decode_s: f64,
    pub(crate) gemm_s: f64,
}

#[inline]
fn decode_plink_2bit(row: &[u8], sample_idx: usize) -> u8 {
    let b = row[sample_idx / 4];
    (b >> ((sample_idx % 4) * 2)) & 0b11
}

#[inline]
fn packed_row_nonmissing_alt_sum_full(row: &[u8], n_samples: usize) -> (usize, f64) {
    let byte_lut = packed_byte_lut();
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut non_missing = 0usize;
    let mut alt_sum = 0.0_f64;
    for &b in row.iter().take(full_bytes) {
        let idx = b as usize;
        non_missing += byte_lut.nonmiss[idx] as usize;
        alt_sum += byte_lut.alt_sum[idx] as f64;
    }
    if rem > 0 {
        let codes = &byte_lut.code4[row[full_bytes] as usize];
        for &code in codes.iter().take(rem) {
            match code {
                0b00 => {
                    non_missing += 1;
                }
                0b10 => {
                    non_missing += 1;
                    alt_sum += 1.0;
                }
                0b11 => {
                    non_missing += 1;
                    alt_sum += 2.0;
                }
                _ => {}
            }
        }
    }
    (non_missing, alt_sum)
}

#[inline]
fn packed_row_nonmissing_alt_sum_selected(row: &[u8], sample_idx: &[usize]) -> (usize, f64) {
    let mut non_missing = 0usize;
    let mut alt_sum = 0.0_f64;
    for &sid in sample_idx {
        let code = decode_plink_2bit(row, sid);
        match code {
            0b00 => {
                non_missing += 1;
            }
            0b10 => {
                non_missing += 1;
                alt_sum += 1.0;
            }
            0b11 => {
                non_missing += 1;
                alt_sum += 2.0;
            }
            _ => {}
        }
    }
    (non_missing, alt_sum)
}

#[inline]
fn align_rsvd_rows(rows: usize, max_rows: usize) -> usize {
    if max_rows == 0 {
        return 1;
    }
    let capped = rows.max(1).min(max_rows);
    if capped < 256 {
        return capped;
    }
    let aligned = (capped / 256) * 256;
    aligned.max(256).min(max_rows)
}

#[inline]
fn env_truthy_local(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(false)
}

#[inline]
fn rsvd_packed_pipeline_enabled() -> bool {
    env_truthy_local("JX_RSVD_PACKED_PIPELINE") || env_truthy_local("JANUSX_RSVD_PACKED_PIPELINE")
}

#[inline]
fn rsvd_decode_threads(total_threads: usize, overlap_enabled: bool) -> usize {
    if !overlap_enabled {
        return total_threads.max(1);
    }
    if let Some(v) =
        parse_positive_env_usize(&["JX_RSVD_DECODE_THREADS", "JANUSX_RSVD_DECODE_THREADS"])
    {
        return v.max(1).min(total_threads.max(1));
    }
    total_threads.max(1)
}

#[inline]
fn rsvd_blas_threads(total_threads: usize, decode_threads: usize, overlap_enabled: bool) -> usize {
    if let Some(v) = parse_positive_env_usize(&["JX_RSVD_BLAS_THREADS", "JANUSX_RSVD_BLAS_THREADS"])
    {
        return v.max(1).min(total_threads.max(1));
    }
    if !overlap_enabled {
        return total_threads.max(1);
    }
    let _ = decode_threads;
    total_threads.max(1)
}

#[inline]
fn rows_from_target_mb(target_mb: usize, n_cols: usize, max_rows: usize) -> usize {
    let raw_rows = block_rows_from_memory_target_mb(
        target_mb as f64,
        n_cols.max(1).saturating_mul(std::mem::size_of::<f32>()),
        max_rows.max(1),
        1,
        1,
        0,
    );
    align_rsvd_rows(raw_rows, max_rows)
}

#[inline]
fn omega_stats_chunk_rows(m: usize, n_cols: usize, target_mb: Option<usize>) -> usize {
    let rows = rsvd_block_rows_with_target_mb(n_cols, m, target_mb)
        .saturating_mul(16)
        .clamp(1024, 16384);
    rows.min(m.max(1))
}

fn packed_subset_row_stats(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    n_samples: usize,
    sample_idx: &[usize],
) -> Result<(Vec<f32>, Vec<bool>, f64), String> {
    if packed_flat.len() != m * bytes_per_snp {
        return Err("packed buffer length mismatch in packed_subset_row_stats".to_string());
    }
    let n = sample_idx.len();
    if n == 0 {
        return Err("sample_idx is empty in packed_subset_row_stats".to_string());
    }
    let mut row_freq = vec![0.0_f32; m];
    let mut row_flip = vec![false; m];
    let mut varsum = 0.0_f64;
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let chunk_rows = omega_stats_chunk_rows(m, n_samples, None);

    for chunk_start in (0..m).step_by(chunk_rows) {
        check_ctrlc()?;
        let chunk_end = (chunk_start + chunk_rows).min(m);
        let chunk_stats: Vec<(f32, bool, f64)> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|i| {
                let row = &packed_flat[i * bytes_per_snp..(i + 1) * bytes_per_snp];
                let (non_missing, alt_sum) = if full_sample_fast {
                    packed_row_nonmissing_alt_sum_full(row, n_samples)
                } else {
                    packed_row_nonmissing_alt_sum_selected(row, sample_idx)
                };
                if non_missing == 0 {
                    return (0.0_f32, false, 0.0_f64);
                }
                let p = alt_sum / (2.0 * non_missing as f64);
                let flip = p > 0.5;
                let p_minor = if flip { 1.0 - p } else { p };
                (
                    p_minor as f32,
                    flip,
                    2.0_f64 * p_minor * (1.0_f64 - p_minor),
                )
            })
            .collect();
        for (off, (freq, flip, var)) in chunk_stats.into_iter().enumerate() {
            row_freq[chunk_start + off] = freq;
            row_flip[chunk_start + off] = flip;
            varsum += var;
        }
    }

    if !(varsum.is_finite() && varsum > 0.0) {
        return Err("invalid scaling denominator in packed subset RSVD".to_string());
    }
    Ok((row_freq, row_flip, varsum))
}

#[derive(Clone, Copy)]
pub(crate) struct PackedRsvdView<'a> {
    pub(crate) packed_flat: &'a [u8],
    pub(crate) bytes_per_snp: usize,
    pub(crate) n_samples: usize,
    pub(crate) row_freq: &'a [f32],
    pub(crate) row_flip: &'a [bool],
    pub(crate) sample_idx: &'a [usize],
    pub(crate) packed_row_indices: Option<&'a [usize]>,
    pub(crate) block_target_mb: Option<usize>,
}

impl<'a> PackedRsvdView<'a> {
    #[inline]
    fn m(self) -> usize {
        self.row_freq.len()
    }

    #[inline]
    fn n(self) -> usize {
        self.sample_idx.len()
    }

    fn validate(self) -> Result<(bool, Option<SubsetDecodePlan>), String> {
        let m = self.m();
        if self.bytes_per_snp == 0 {
            return Err("bytes_per_snp must be > 0 for packed RSVD".to_string());
        }
        if self.row_flip.len() != m {
            return Err("row_flip length mismatch in packed RSVD".to_string());
        }
        if self.sample_idx.is_empty() {
            return Err("sample_idx must not be empty in packed RSVD".to_string());
        }
        if self.packed_flat.len() % self.bytes_per_snp != 0 {
            return Err("packed buffer is not aligned to bytes_per_snp in packed RSVD".to_string());
        }
        if let Some(row_idx) = self.packed_row_indices {
            if row_idx.len() != m {
                return Err("packed_row_indices length mismatch in packed RSVD".to_string());
            }
        } else if self.packed_flat.len() != m * self.bytes_per_snp {
            return Err("packed buffer length mismatch in packed RSVD".to_string());
        }
        let full_sample_fast = is_identity_indices(self.sample_idx, self.n_samples);
        let subset_plan = if full_sample_fast {
            None
        } else {
            Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
                self.sample_idx,
                self.n_samples,
            ))
        };
        Ok((full_sample_fast, subset_plan))
    }
}

#[inline]
pub(crate) fn rsvd_block_rows_with_target_mb(
    n_cols: usize,
    max_rows: usize,
    target_mb: Option<usize>,
) -> usize {
    if let Some(target_mb) = target_mb.filter(|&v| v > 0) {
        return rows_from_target_mb(target_mb, n_cols, max_rows);
    }
    rsvd_block_rows_env(n_cols, max_rows)
}

#[inline]
pub(crate) fn rsvd_block_rows_env(n_cols: usize, max_rows: usize) -> usize {
    if let Some(rows) = parse_positive_env_usize(&["JANUSX_RSVD_BLOCK_ROWS", "JX_RSVD_BLOCK_ROWS"])
    {
        return align_rsvd_rows(rows, max_rows);
    }
    if let Some(target_mb) = parse_positive_env_f64(&[
        "JANUSX_RSVD_BLOCK_MB",
        "JX_RSVD_BLOCK_MB",
        "JX_BED_BLOCK_TARGET_MB",
        "JANUSX_BED_BLOCK_TARGET_MB",
    ]) {
        return rows_from_target_mb(target_mb as usize, n_cols, max_rows);
    }
    let cap_rows = max_rows.min(4096).max(1);
    let mut heuristic = rows_from_target_mb(64, n_cols, max_rows).min(cap_rows);
    if max_rows >= 256 {
        heuristic = heuristic.max(256);
    }
    align_rsvd_rows(heuristic, max_rows)
}

#[inline]
fn checked_cblas_dim(v: usize, label: &str) -> Result<CblasInt, String> {
    if v > CblasInt::MAX as usize {
        return Err(format!("dimension overflow for {label}: {v}"));
    }
    Ok(v as CblasInt)
}

#[derive(Clone)]
struct PackedDecodeBuf {
    row_start: usize,
    rows: usize,
    block: Vec<f32>,
}

#[derive(Clone)]
struct PackedDecodeOmegaBuf {
    row_start: usize,
    rows: usize,
    block: Vec<f32>,
    omega: Vec<f32>,
}

#[inline]
fn rsvd_packed_overlap_enabled(total_threads: usize, m: usize, block_rows: usize) -> bool {
    rsvd_packed_pipeline_enabled() && total_threads > 1 && m > block_rows
}

#[allow(clippy::too_many_arguments)]
fn decode_block_rows_f32(
    view: PackedRsvdView<'_>,
    full_sample_fast: bool,
    subset_plan: Option<&SubsetDecodePlan>,
    row_start: usize,
    rows: usize,
    row_flip_local: &mut [bool],
    packed_row_indices_local: &mut [usize],
    row_mean: &mut [f32],
    row_inv_sd: &mut [f32],
    out_block: &mut [f32],
    decode_pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    debug_assert!(row_flip_local.len() >= rows);
    if view.row_flip.len() < row_start.saturating_add(rows) {
        return Err("decode_block_rows_f32: row_flip slice too small".to_string());
    }
    row_flip_local[..rows].copy_from_slice(&view.row_flip[row_start..row_start + rows]);
    prepare_packed_block_row_indices(
        view.packed_row_indices,
        row_start,
        rows,
        packed_row_indices_local,
    )?;
    prepare_packed_block_centered_mean_scale_f32(
        view.row_freq,
        row_start,
        rows,
        row_mean,
        row_inv_sd,
    )?;
    decode_prepared_additive_block_packed_f32(
        view.packed_flat,
        view.bytes_per_snp,
        view.n_samples,
        &row_flip_local[..rows],
        &row_mean[..rows],
        &row_inv_sd[..rows],
        view.sample_idx,
        full_sample_fast,
        subset_plan,
        Some(&packed_row_indices_local[..rows]),
        0usize,
        rows,
        out_block,
        decode_pool,
    )
}

#[inline]
fn madvise_packed_rows_after_decode(view: PackedRsvdView<'_>, row_start: usize, rows: usize) {
    if rows == 0 || view.bytes_per_snp == 0 {
        return;
    }
    let Some(row_end) = row_start.checked_add(rows) else {
        return;
    };
    let (first_row, last_row) = if let Some(indices) = view.packed_row_indices {
        let Some(slice) = indices.get(row_start..row_end) else {
            return;
        };
        let Some((&first, rest)) = slice.split_first() else {
            return;
        };
        let mut min_row = first;
        let mut max_row = first;
        for &idx in rest {
            min_row = min_row.min(idx);
            max_row = max_row.max(idx);
        }
        (min_row, max_row)
    } else {
        (row_start, row_end - 1)
    };
    let Some(start) = first_row.checked_mul(view.bytes_per_snp) else {
        return;
    };
    let Some(end) = last_row
        .checked_add(1)
        .and_then(|r| r.checked_mul(view.bytes_per_snp))
    else {
        return;
    };
    if end <= start || end > view.packed_flat.len() {
        return;
    }
    let ptr = unsafe { view.packed_flat.as_ptr().add(start) };
    admx_madvise_dontneed_bytes(ptr, end - start);
}

#[inline]
fn accum_gram_lower_f64(
    block: &[f32],
    rows: usize,
    cols: usize,
    gram: &mut [f64],
    gram_block: &mut [f32],
) -> Result<(), String> {
    debug_assert_eq!(block.len(), rows * cols);
    debug_assert_eq!(gram.len(), cols * cols);
    debug_assert_eq!(gram_block.len(), cols * cols);
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    gram_block.fill(0.0);
    let rows_i = checked_cblas_dim(rows, "rows")?;
    let cols_i = checked_cblas_dim(cols, "cols")?;
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            cols_i,
            cols_i,
            rows_i,
            1.0_f32,
            block.as_ptr(),
            cols_i,
            block.as_ptr(),
            cols_i,
            0.0_f32,
            gram_block.as_mut_ptr(),
            cols_i,
        );
    }
    for i in 0..(cols * cols) {
        gram[i] += gram_block[i] as f64;
    }
    Ok(())
}

pub(crate) fn rsvd_right_singular_from_gram(
    gram: &[f64],
    cols: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if cols == 0 {
        return Err("cols must be > 0 for packed RSVD gram eigendecomposition".to_string());
    }
    if gram.len() != cols * cols {
        return Err("gram buffer length mismatch in packed RSVD".to_string());
    }
    let eig = SymmetricEigen::new(DMatrix::from_row_slice(cols, cols, gram));
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));

    let mut s = vec![0.0_f32; cols];
    let mut v = vec![0.0_f32; cols * cols];
    for (new_col, &src_col) in order.iter().enumerate() {
        let eval = eig.eigenvalues[src_col].max(1e-12);
        s[new_col] = (eval as f32).sqrt();
        for r in 0..cols {
            v[r * cols + new_col] = eig.eigenvectors[(r, src_col)] as f32;
        }
    }
    Ok((s, v))
}

pub(crate) fn rsvd_packed_compute_at_random_omega(
    view: PackedRsvdView<'_>,
    kp: usize,
    seed: u64,
) -> Result<Vec<f32>, String> {
    let (full_sample_fast, subset_plan) = view.validate()?;
    let m = view.m();
    let n = view.n();
    let total_threads = rayon::current_num_threads().max(1);
    let block_rows = rsvd_block_rows_with_target_mb(n, m, view.block_target_mb);
    let overlap_enabled = rsvd_packed_overlap_enabled(total_threads, m, block_rows);
    let decode_threads = rsvd_decode_threads(total_threads, overlap_enabled);
    let blas_threads = rsvd_blas_threads(total_threads, decode_threads, overlap_enabled);
    let decode_pool = if overlap_enabled {
        if decode_threads > 1 {
            get_cached_pool(decode_threads).map_err(|e| e.to_string())?
        } else {
            None
        }
    } else {
        get_cached_pool(total_threads).map_err(|e| e.to_string())?
    };
    let mut out = vec![0.0_f32; n * kp];
    let _n_i = checked_cblas_dim(n, "n")?;
    let _kp_i = checked_cblas_dim(kp, "kp")?;
    let _blas_guard = BlasThreadGuard::enter(blas_threads.max(1));

    if !overlap_enabled {
        let mut block = vec![0.0_f32; block_rows * n];
        let mut omega_block = vec![0.0_f32; block_rows * kp];
        let mut row_flip_local = vec![false; block_rows];
        let mut packed_row_indices_local = vec![0usize; block_rows];
        let mut row_mean = vec![0.0_f32; block_rows];
        let mut row_inv_sd = vec![1.0_f32; block_rows];
        let mut rng = StdRng::seed_from_u64(seed);
        for row_start in (0..m).step_by(block_rows) {
            check_ctrlc()?;
            check_admx_memory_limit("rsvd_packed/at_random_omega")?;
            let row_end = (row_start + block_rows).min(m);
            let cur_rows = row_end - row_start;
            let cur_block = &mut block[..cur_rows * n];
            decode_block_rows_f32(
                view,
                full_sample_fast,
                subset_plan.as_ref(),
                row_start,
                cur_rows,
                &mut row_flip_local,
                &mut packed_row_indices_local,
                &mut row_mean,
                &mut row_inv_sd,
                cur_block,
                decode_pool.as_ref(),
            )?;
            madvise_packed_rows_after_decode(view, row_start, cur_rows);
            let cur_omega = &mut omega_block[..cur_rows * kp];
            fill_random_omega_block(&mut rng, cur_omega);
            let _ = checked_cblas_dim(cur_rows, "cur_rows")?;
            row_major_block_t_mul_mat_accum_f32(
                cur_block, cur_rows, n, cur_omega, kp, &mut out, None,
            );
        }
    } else {
        let producer_error = Arc::new(Mutex::new(None::<String>));
        let producer_error_bg = Arc::clone(&producer_error);
        let mut next_row_start = 0usize;
        let mut row_flip_local = vec![false; block_rows];
        let mut packed_row_indices_local = vec![0usize; block_rows];
        let mut row_mean = vec![0.0_f32; block_rows];
        let mut row_inv_sd = vec![1.0_f32; block_rows];
        let mut rng = StdRng::seed_from_u64(seed);

        run_double_buffer(
            2,
            || PackedDecodeOmegaBuf {
                row_start: 0usize,
                rows: 0usize,
                block: vec![0.0_f32; block_rows * n],
                omega: vec![0.0_f32; block_rows * kp],
            },
            |buf| {
                if next_row_start >= m {
                    buf.rows = 0;
                    return false;
                }
                if let Err(e) = check_admx_memory_limit("rsvd_packed/at_random_omega_decode") {
                    if let Ok(mut slot) = producer_error_bg.lock() {
                        *slot = Some(e);
                    }
                    buf.rows = 0;
                    return false;
                }
                let row_start = next_row_start;
                let row_end = (row_start + block_rows).min(m);
                let cur_rows = row_end - row_start;
                let cur_block = &mut buf.block[..cur_rows * n];
                match decode_block_rows_f32(
                    view,
                    full_sample_fast,
                    subset_plan.as_ref(),
                    row_start,
                    cur_rows,
                    &mut row_flip_local,
                    &mut packed_row_indices_local,
                    &mut row_mean,
                    &mut row_inv_sd,
                    cur_block,
                    decode_pool.as_ref(),
                ) {
                    Ok(()) => {
                        madvise_packed_rows_after_decode(view, row_start, cur_rows);
                        fill_random_omega_block(&mut rng, &mut buf.omega[..cur_rows * kp]);
                        buf.row_start = row_start;
                        buf.rows = cur_rows;
                        next_row_start = row_end;
                        next_row_start < m
                    }
                    Err(e) => {
                        if let Ok(mut slot) = producer_error_bg.lock() {
                            *slot = Some(e);
                        }
                        buf.rows = 0;
                        false
                    }
                }
            },
            |buf| {
                if buf.rows == 0 {
                    return Ok::<(), String>(());
                }
                check_ctrlc()?;
                check_admx_memory_limit("rsvd_packed/at_random_omega_gemm")?;
                let _ = checked_cblas_dim(buf.rows, "cur_rows")?;
                row_major_block_t_mul_mat_accum_f32(
                    &buf.block[..buf.rows * n],
                    buf.rows,
                    n,
                    &buf.omega[..buf.rows * kp],
                    kp,
                    &mut out,
                    None,
                );
                Ok::<(), String>(())
            },
        )?;

        let producer_err = producer_error
            .lock()
            .map_err(|_| "packed RSVD A^TΩ producer error lock poisoned".to_string())?
            .take();
        if let Some(err) = producer_err {
            return Err(err);
        }
    }
    Ok(out)
}

pub(crate) fn rsvd_packed_compute_ata_omega(
    view: PackedRsvdView<'_>,
    omega: &[f32],
    kp: usize,
    timing: Option<&mut RsvdKernelTiming>,
) -> Result<Vec<f32>, String> {
    struct PackedDecodeBuf {
        row_start: usize,
        rows: usize,
        block: Vec<f32>,
    }

    let (full_sample_fast, subset_plan) = view.validate()?;
    let m = view.m();
    let n = view.n();
    if omega.len() != n * kp {
        return Err("omega shape mismatch in packed RSVD A^T*(A*Omega)".to_string());
    }
    let total_threads = rayon::current_num_threads().max(1);
    let block_rows = rsvd_block_rows_with_target_mb(n, m, view.block_target_mb);
    let overlap_enabled = rsvd_packed_pipeline_enabled() && total_threads > 1 && m > block_rows;
    let decode_threads = rsvd_decode_threads(total_threads, overlap_enabled);
    let blas_threads = rsvd_blas_threads(total_threads, decode_threads, overlap_enabled);
    let decode_pool = if overlap_enabled {
        if decode_threads > 1 {
            get_cached_pool(decode_threads).map_err(|e| e.to_string())?
        } else {
            None
        }
    } else {
        get_cached_pool(total_threads).map_err(|e| e.to_string())?
    };
    let mut out = vec![0.0_f32; n * kp];
    let _n_i = checked_cblas_dim(n, "n")?;
    let _kp_i = checked_cblas_dim(kp, "kp")?;
    let _blas_guard = BlasThreadGuard::enter(blas_threads.max(1));
    let measure_timing = timing.is_some();
    let mut decode_s = 0.0_f64;
    let mut gemm_s = 0.0_f64;

    if !overlap_enabled {
        let mut block = vec![0.0_f32; block_rows * n];
        let mut g_block = vec![0.0_f32; block_rows * kp];
        let mut row_flip_local = vec![false; block_rows];
        let mut packed_row_indices_local = vec![0usize; block_rows];
        let mut row_mean = vec![0.0_f32; block_rows];
        let mut row_inv_sd = vec![1.0_f32; block_rows];
        for row_start in (0..m).step_by(block_rows) {
            check_ctrlc()?;
            check_admx_memory_limit("rsvd_packed/ata_omega")?;
            let row_end = (row_start + block_rows).min(m);
            let cur_rows = row_end - row_start;
            let cur_block = &mut block[..cur_rows * n];
            let t_decode = if measure_timing {
                Some(Instant::now())
            } else {
                None
            };
            decode_block_rows_f32(
                view,
                full_sample_fast,
                subset_plan.as_ref(),
                row_start,
                cur_rows,
                &mut row_flip_local,
                &mut packed_row_indices_local,
                &mut row_mean,
                &mut row_inv_sd,
                cur_block,
                decode_pool.as_ref(),
            )?;
            madvise_packed_rows_after_decode(view, row_start, cur_rows);
            if let Some(t0) = t_decode {
                decode_s += t0.elapsed().as_secs_f64();
            }
            let cur_rows_i = checked_cblas_dim(cur_rows, "cur_rows")?;
            let cur_g = &mut g_block[..cur_rows * kp];
            let t_gemm = if measure_timing {
                Some(Instant::now())
            } else {
                None
            };
            row_major_block_mul_mat_f32(cur_block, cur_rows, n, omega, kp, cur_g, None);
            let _ = cur_rows_i;
            row_major_block_t_mul_mat_accum_f32(cur_block, cur_rows, n, cur_g, kp, &mut out, None);
            if let Some(t0) = t_gemm {
                gemm_s += t0.elapsed().as_secs_f64();
            }
        }
    } else {
        let producer_error = Arc::new(Mutex::new(None::<String>));
        let decode_acc = Arc::new(Mutex::new(0.0_f64));
        let mut next_row_start = 0usize;
        let mut g_block = vec![0.0_f32; block_rows * kp];
        let mut row_flip_local = vec![false; block_rows];
        let mut packed_row_indices_local = vec![0usize; block_rows];
        let mut row_mean = vec![0.0_f32; block_rows];
        let mut row_inv_sd = vec![1.0_f32; block_rows];
        let producer_error_bg = Arc::clone(&producer_error);
        let decode_acc_bg = Arc::clone(&decode_acc);

        run_double_buffer(
            2,
            || PackedDecodeBuf {
                row_start: 0,
                rows: 0,
                block: vec![0.0_f32; block_rows * n],
            },
            |buf| {
                if next_row_start >= m {
                    buf.rows = 0;
                    return false;
                }
                if let Err(e) = check_admx_memory_limit("rsvd_packed/ata_omega_decode") {
                    if let Ok(mut slot) = producer_error_bg.lock() {
                        *slot = Some(e);
                    }
                    buf.rows = 0;
                    return false;
                }
                let row_start = next_row_start;
                let row_end = (row_start + block_rows).min(m);
                let cur_rows = row_end - row_start;
                let cur_block = &mut buf.block[..cur_rows * n];
                let t_decode = if measure_timing {
                    Some(Instant::now())
                } else {
                    None
                };
                match decode_block_rows_f32(
                    view,
                    full_sample_fast,
                    subset_plan.as_ref(),
                    row_start,
                    cur_rows,
                    &mut row_flip_local,
                    &mut packed_row_indices_local,
                    &mut row_mean,
                    &mut row_inv_sd,
                    cur_block,
                    decode_pool.as_ref(),
                ) {
                    Ok(()) => {
                        madvise_packed_rows_after_decode(view, row_start, cur_rows);
                        if let Some(t0) = t_decode {
                            if let Ok(mut acc) = decode_acc_bg.lock() {
                                *acc += t0.elapsed().as_secs_f64();
                            }
                        }
                        buf.row_start = row_start;
                        buf.rows = cur_rows;
                        next_row_start = row_end;
                        next_row_start < m
                    }
                    Err(e) => {
                        if let Ok(mut slot) = producer_error_bg.lock() {
                            *slot = Some(e);
                        }
                        buf.rows = 0;
                        false
                    }
                }
            },
            |buf| {
                if buf.rows == 0 {
                    return Ok::<(), String>(());
                }
                check_ctrlc()?;
                check_admx_memory_limit("rsvd_packed/ata_omega_gemm")?;
                let cur_rows = buf.rows;
                let cur_rows_i = checked_cblas_dim(cur_rows, "cur_rows")?;
                let cur_block = &buf.block[..cur_rows * n];
                let cur_g = &mut g_block[..cur_rows * kp];
                let t_gemm = if measure_timing {
                    Some(Instant::now())
                } else {
                    None
                };
                row_major_block_mul_mat_f32(cur_block, cur_rows, n, omega, kp, cur_g, None);
                let _ = cur_rows_i;
                row_major_block_t_mul_mat_accum_f32(
                    cur_block, cur_rows, n, cur_g, kp, &mut out, None,
                );
                if let Some(t0) = t_gemm {
                    gemm_s += t0.elapsed().as_secs_f64();
                }
                Ok::<(), String>(())
            },
        )?;

        if let Some(err) = producer_error
            .lock()
            .map_err(|_| "packed RSVD pipeline producer error lock poisoned".to_string())?
            .take()
        {
            return Err(err);
        }
        decode_s += *decode_acc
            .lock()
            .map_err(|_| "packed RSVD pipeline decode timing lock poisoned".to_string())?;
    }
    if let Some(t) = timing {
        t.decode_s += decode_s;
        t.gemm_s += gemm_s;
    }
    Ok(out)
}

pub(crate) fn rsvd_packed_compute_gram_aq(
    view: PackedRsvdView<'_>,
    q: &[f32],
    kp: usize,
) -> Result<Vec<f64>, String> {
    let (full_sample_fast, subset_plan) = view.validate()?;
    let m = view.m();
    let n = view.n();
    if q.len() != n * kp {
        return Err("q shape mismatch in packed RSVD gram accumulation".to_string());
    }
    let total_threads = rayon::current_num_threads().max(1);
    let block_rows = rsvd_block_rows_with_target_mb(n, m, view.block_target_mb);
    let overlap_enabled = rsvd_packed_overlap_enabled(total_threads, m, block_rows);
    let decode_threads = rsvd_decode_threads(total_threads, overlap_enabled);
    let blas_threads = rsvd_blas_threads(total_threads, decode_threads, overlap_enabled);
    let decode_pool = if overlap_enabled {
        if decode_threads > 1 {
            get_cached_pool(decode_threads).map_err(|e| e.to_string())?
        } else {
            None
        }
    } else {
        get_cached_pool(total_threads).map_err(|e| e.to_string())?
    };
    let mut gram = vec![0.0_f64; kp * kp];
    let mut gram_block = vec![0.0_f32; kp * kp];
    let mut g_block = vec![0.0_f32; block_rows * kp];
    let _n_i = checked_cblas_dim(n, "n")?;
    let _kp_i = checked_cblas_dim(kp, "kp")?;
    let _blas_guard = BlasThreadGuard::enter(blas_threads.max(1));

    if !overlap_enabled {
        let mut block = vec![0.0_f32; block_rows * n];
        let mut row_flip_local = vec![false; block_rows];
        let mut packed_row_indices_local = vec![0usize; block_rows];
        let mut row_mean = vec![0.0_f32; block_rows];
        let mut row_inv_sd = vec![1.0_f32; block_rows];
        for row_start in (0..m).step_by(block_rows) {
            check_ctrlc()?;
            check_admx_memory_limit("rsvd_packed/gram_aq")?;
            let row_end = (row_start + block_rows).min(m);
            let cur_rows = row_end - row_start;
            let cur_block = &mut block[..cur_rows * n];
            decode_block_rows_f32(
                view,
                full_sample_fast,
                subset_plan.as_ref(),
                row_start,
                cur_rows,
                &mut row_flip_local,
                &mut packed_row_indices_local,
                &mut row_mean,
                &mut row_inv_sd,
                cur_block,
                decode_pool.as_ref(),
            )?;
            madvise_packed_rows_after_decode(view, row_start, cur_rows);
            let cur_rows_i = checked_cblas_dim(cur_rows, "cur_rows")?;
            let cur_g = &mut g_block[..cur_rows * kp];
            row_major_block_mul_mat_f32(cur_block, cur_rows, n, q, kp, cur_g, None);
            let _ = cur_rows_i;
            accum_gram_lower_f64(cur_g, cur_rows, kp, &mut gram, &mut gram_block)?;
        }
    } else {
        let producer_error = Arc::new(Mutex::new(None::<String>));
        let producer_error_bg = Arc::clone(&producer_error);
        let mut next_row_start = 0usize;
        let mut row_flip_local = vec![false; block_rows];
        let mut packed_row_indices_local = vec![0usize; block_rows];
        let mut row_mean = vec![0.0_f32; block_rows];
        let mut row_inv_sd = vec![1.0_f32; block_rows];

        run_double_buffer(
            2,
            || PackedDecodeBuf {
                row_start: 0usize,
                rows: 0usize,
                block: vec![0.0_f32; block_rows * n],
            },
            |buf| {
                if next_row_start >= m {
                    buf.rows = 0;
                    return false;
                }
                if let Err(e) = check_admx_memory_limit("rsvd_packed/gram_aq_decode") {
                    if let Ok(mut slot) = producer_error_bg.lock() {
                        *slot = Some(e);
                    }
                    buf.rows = 0;
                    return false;
                }
                let row_start = next_row_start;
                let row_end = (row_start + block_rows).min(m);
                let cur_rows = row_end - row_start;
                let cur_block = &mut buf.block[..cur_rows * n];
                match decode_block_rows_f32(
                    view,
                    full_sample_fast,
                    subset_plan.as_ref(),
                    row_start,
                    cur_rows,
                    &mut row_flip_local,
                    &mut packed_row_indices_local,
                    &mut row_mean,
                    &mut row_inv_sd,
                    cur_block,
                    decode_pool.as_ref(),
                ) {
                    Ok(()) => {
                        madvise_packed_rows_after_decode(view, row_start, cur_rows);
                        buf.row_start = row_start;
                        buf.rows = cur_rows;
                        next_row_start = row_end;
                        next_row_start < m
                    }
                    Err(e) => {
                        if let Ok(mut slot) = producer_error_bg.lock() {
                            *slot = Some(e);
                        }
                        buf.rows = 0;
                        false
                    }
                }
            },
            |buf| {
                if buf.rows == 0 {
                    return Ok::<(), String>(());
                }
                check_ctrlc()?;
                check_admx_memory_limit("rsvd_packed/gram_aq_gemm")?;
                let cur_rows_i = checked_cblas_dim(buf.rows, "cur_rows")?;
                let cur_g = &mut g_block[..buf.rows * kp];
                row_major_block_mul_mat_f32(
                    &buf.block[..buf.rows * n],
                    buf.rows,
                    n,
                    q,
                    kp,
                    cur_g,
                    None,
                );
                let _ = cur_rows_i;
                accum_gram_lower_f64(cur_g, buf.rows, kp, &mut gram, &mut gram_block)?;
                Ok::<(), String>(())
            },
        )?;

        let producer_err = producer_error
            .lock()
            .map_err(|_| "packed RSVD gram(AQ) producer error lock poisoned".to_string())?
            .take();
        if let Some(err) = producer_err {
            return Err(err);
        }
    }

    for i in 0..kp {
        for j in 0..i {
            gram[j * kp + i] = gram[i * kp + j];
        }
    }
    Ok(gram)
}

#[inline]
fn fill_random_omega_block(rng: &mut StdRng, omega_block: &mut [f32]) {
    for v in omega_block.iter_mut() {
        *v = rng.sample::<f64, _>(StandardNormal) as f32;
    }
}

pub(crate) fn rsvd_project_sample_eigvecs(
    q: &[f32],
    rows: usize,
    kp: usize,
    v_small: &[f32],
    k_eff: usize,
) -> Result<Vec<f32>, String> {
    if q.len() != rows.saturating_mul(kp) {
        return Err("sample RSVD projection q shape mismatch".to_string());
    }
    if v_small.len() != kp.saturating_mul(kp) {
        return Err("sample RSVD projection v_small shape mismatch".to_string());
    }
    if k_eff > kp {
        return Err("sample RSVD projection k_eff exceeds kp".to_string());
    }
    let mut rhs = vec![0.0_f32; kp * k_eff];
    for r in 0..kp {
        rhs[r * k_eff..(r + 1) * k_eff].copy_from_slice(&v_small[r * kp..r * kp + k_eff]);
    }
    let mut out = vec![0.0_f32; rows * k_eff];
    row_major_block_mul_mat_f32(q, rows, kp, &rhs, k_eff, &mut out, None);
    Ok(out)
}

fn rsvd_tile_cols_env() -> usize {
    match std::env::var("JANUSX_ADMX_RSVD_TILE") {
        Ok(v) => v.trim().parse::<usize>().ok().unwrap_or(1024).max(1),
        Err(_) => 1024,
    }
}

#[allow(dead_code)]
fn thin_svd_from_tall(
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if rows == 0 || cols == 0 {
        return Err("invalid matrix shape for thin SVD".to_string());
    }
    if x.len() != rows * cols {
        return Err(format!(
            "matrix buffer size mismatch in thin SVD: len={}, expected={}",
            x.len(),
            rows * cols
        ));
    }

    let rows_i = checked_cblas_dim(rows, "rows")?;
    let cols_i = checked_cblas_dim(cols, "cols")?;
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let mut gram_f32 = vec![0.0_f32; cols * cols];
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            cols_i,
            cols_i,
            rows_i,
            1.0_f32,
            x.as_ptr(),
            cols_i,
            x.as_ptr(),
            cols_i,
            0.0_f32,
            gram_f32.as_mut_ptr(),
            cols_i,
        );
    }
    let gram: Vec<f64> = gram_f32.into_iter().map(|v| v as f64).collect();

    let eig = SymmetricEigen::new(DMatrix::from_row_slice(cols, cols, &gram));
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));

    let mut s = vec![0.0_f32; cols];
    let mut v = vec![0.0_f32; cols * cols];
    for (new_col, &src_col) in order.iter().enumerate() {
        let eval = eig.eigenvalues[src_col].max(1e-12);
        s[new_col] = (eval as f32).sqrt();
        for r in 0..cols {
            v[r * cols + new_col] = eig.eigenvectors[(r, src_col)] as f32;
        }
    }

    let mut v_scaled = v.clone();
    for c in 0..cols {
        let inv = if s[c] > 1e-12 {
            1.0_f32 / s[c]
        } else {
            0.0_f32
        };
        for r in 0..cols {
            v_scaled[r * cols + c] *= inv;
        }
    }
    let mut u = vec![0.0_f32; rows * cols];
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            rows_i,
            cols_i,
            cols_i,
            1.0_f32,
            x.as_ptr(),
            cols_i,
            v_scaled.as_ptr(),
            cols_i,
            0.0_f32,
            u.as_mut_ptr(),
            cols_i,
        );
    }
    Ok((u, s, v))
}

fn spectral_proxy_from_columns(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut s = vec![0.0_f32; cols];
    for c in 0..cols {
        let mut ss = 0.0_f64;
        for r in 0..rows {
            let v = x[r * cols + c] as f64;
            ss += v * v;
        }
        s[c] = (ss.max(0.0)).sqrt() as f32;
    }
    s.sort_by(|a, b| b.total_cmp(a));
    s
}

fn qr_normalize_mgs(x: &[f32], rows: usize, cols: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    if x.len() != rows * cols {
        return Err("buffer size mismatch in qr_normalize_mgs".to_string());
    }
    let mut q = vec![0.0_f32; rows * cols];
    let mut rdiag = vec![0.0_f32; cols];
    let mut v = vec![0.0_f64; rows];

    for j in 0..cols {
        for r in 0..rows {
            v[r] = x[r * cols + j] as f64;
        }

        for i in 0..j {
            let mut dot = 0.0_f64;
            for r in 0..rows {
                dot += (q[r * cols + i] as f64) * v[r];
            }
            for r in 0..rows {
                v[r] -= dot * (q[r * cols + i] as f64);
            }
        }

        let mut nrm2 = 0.0_f64;
        for r in 0..rows {
            nrm2 += v[r] * v[r];
        }
        let nrm = nrm2.sqrt();
        if !(nrm.is_finite() && nrm > 1e-12) {
            return Err(format!(
                "QR normalization failed: near-dependent column at j={j}"
            ));
        }
        rdiag[j] = nrm as f32;
        let inv = 1.0_f64 / nrm;
        for r in 0..rows {
            q[r * cols + j] = (v[r] * inv) as f32;
        }
    }
    Ok((q, rdiag))
}

fn lu_normalize_with_qr_fallback(
    x: &[f32],
    rows: usize,
    cols: usize,
    lu_eps: f64,
    cond_min_ratio: f64,
) -> Result<(Vec<f32>, bool), String> {
    if x.len() != rows * cols {
        return Err("buffer size mismatch in lu_normalize_with_qr_fallback".to_string());
    }
    if rows < cols {
        let (q_qr, _) = qr_normalize_mgs(x, rows, cols)?;
        return Ok((q_qr, false));
    }

    let mut a = vec![0.0_f64; rows * cols];
    for i in 0..(rows * cols) {
        a[i] = x[i] as f64;
    }
    let mut piv: Vec<usize> = (0..rows).collect();

    let mut dmin = f64::INFINITY;
    let mut dmax = 0.0_f64;
    let mut lu_ok = true;

    for j in 0..cols {
        let mut piv_row = j;
        let mut piv_abs = 0.0_f64;
        for i in j..rows {
            let v = a[i * cols + j].abs();
            if v > piv_abs {
                piv_abs = v;
                piv_row = i;
            }
        }
        if !(piv_abs.is_finite() && piv_abs > lu_eps) {
            lu_ok = false;
            break;
        }

        if piv_row != j {
            for c in 0..cols {
                a.swap(j * cols + c, piv_row * cols + c);
            }
            piv.swap(j, piv_row);
        }

        let ajj = a[j * cols + j];
        let ad = ajj.abs();
        if !(ad.is_finite() && ad > lu_eps) {
            lu_ok = false;
            break;
        }
        if ad < dmin {
            dmin = ad;
        }
        if ad > dmax {
            dmax = ad;
        }

        let inv = 1.0_f64 / ajj;
        for i in (j + 1)..rows {
            let lij = a[i * cols + j] * inv;
            a[i * cols + j] = lij;
            for c in (j + 1)..cols {
                a[i * cols + c] -= lij * a[j * cols + c];
            }
        }
    }

    if lu_ok {
        if !(dmax.is_finite() && dmax > lu_eps) {
            lu_ok = false;
        } else if !(dmin.is_finite() && dmin > lu_eps) {
            lu_ok = false;
        } else {
            let ratio = dmin / dmax;
            if !(ratio.is_finite() && ratio >= cond_min_ratio) {
                lu_ok = false;
            }
        }
    }

    if !lu_ok {
        let (q_qr, _) = qr_normalize_mgs(x, rows, cols)?;
        return Ok((q_qr, false));
    }

    // A now stores compact LU of (P * X). Build Q := P^T * L (same column space as X).
    let mut q = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        let orig_row = piv[i];
        let dst = &mut q[orig_row * cols..(orig_row + 1) * cols];
        for j in 0..cols {
            let val = if i < cols {
                if i == j {
                    1.0_f64
                } else if i > j {
                    a[i * cols + j]
                } else {
                    0.0_f64
                }
            } else {
                a[i * cols + j]
            };
            dst[j] = val as f32;
        }
    }

    // Column scaling keeps numerics bounded while preserving span.
    for j in 0..cols {
        let mut ss = 0.0_f64;
        for i in 0..rows {
            let v = q[i * cols + j] as f64;
            ss += v * v;
        }
        let nrm = ss.sqrt();
        if !(nrm.is_finite() && nrm > lu_eps) {
            let (q_qr, _) = qr_normalize_mgs(x, rows, cols)?;
            return Ok((q_qr, false));
        }
        let inv = 1.0_f64 / nrm;
        for i in 0..rows {
            q[i * cols + j] = ((q[i * cols + j] as f64) * inv) as f32;
        }
    }
    Ok((q, true))
}

pub fn rsvd_packed_subset(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    n_samples: usize,
    sample_idx: &[usize],
    k: usize,
    seed: u64,
    power: usize,
    tol: f32,
    _tile_cols: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>, usize), String> {
    if k == 0 {
        return Err("k must be > 0".to_string());
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err("tol must be positive and finite".to_string());
    }
    let n = sample_idx.len();
    if n == 0 {
        return Err("sample_idx must not be empty".to_string());
    }
    let (row_freq, row_flip, varsum) =
        packed_subset_row_stats(packed_flat, m, bytes_per_snp, n_samples, sample_idx)?;
    let k_eff = k.min(n);
    let kp = (k_eff + 10).max(20).min(m.max(1)).min(n.max(1));
    let packed_view = PackedRsvdView {
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_freq: &row_freq,
        row_flip: &row_flip,
        sample_idx,
        packed_row_indices: None,
        block_target_mb: None,
    };

    let mut y = rsvd_packed_compute_at_random_omega(packed_view, kp, seed)?;
    let (mut q, _) = qr_normalize_mgs(&y, n, kp)?;

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    let lu_eps = 1e-10_f64;
    let lu_cond_ratio = 1e-8_f64;
    let mut q_is_qr = true;
    // Power iteration acceleration:
    // - first (power-1) rounds: LU normalization with automatic QR fallback on ill-conditioning
    // - final round (or early-exit finalization): QR normalization for stability
    for it in 0..power {
        check_ctrlc()?;
        y = rsvd_packed_compute_ata_omega(packed_view, &q, kp, None)?;
        y.par_iter_mut()
            .zip(q.par_iter())
            .for_each(|(yi, qi)| *yi -= alpha * *qi);
        let force_qr = (it + 1) >= power;
        if force_qr {
            let (q_new, _) = qr_normalize_mgs(&y, n, kp)?;
            q = q_new;
            q_is_qr = true;
        } else {
            let (q_new, used_lu) = lu_normalize_with_qr_fallback(&y, n, kp, lu_eps, lu_cond_ratio)?;
            q = q_new;
            q_is_qr = !used_lu;
        }
        let s_y = spectral_proxy_from_columns(&y, n, kp);

        if it > 0 {
            let mut max_rel = 0.0_f32;
            for i in 0..k_eff {
                let sk_now = s_y[i] + alpha;
                let denom = sk_now.max(1e-12);
                let rel = ((sk_now - sk[i]).abs()) / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                sk[i] = sk_now;
            }
            if max_rel < tol {
                if !q_is_qr {
                    let (q_new, _) = qr_normalize_mgs(&q, n, kp)?;
                    q = q_new;
                    q_is_qr = true;
                }
                break;
            }
        } else {
            for i in 0..kp {
                sk[i] = s_y[i] + alpha;
            }
        }

        let tail = s_y[kp - 1];
        if alpha < tail {
            alpha = 0.5 * (alpha + tail);
        }
    }
    if !q_is_qr {
        let (q_new, _) = qr_normalize_mgs(&q, n, kp)?;
        q = q_new;
    }

    let gram = rsvd_packed_compute_gram_aq(packed_view, &q, kp)?;
    let (s_all, v_small) = rsvd_right_singular_from_gram(&gram, kp)?;

    let scale = varsum as f32;
    let mut eigvals = vec![0.0_f32; k_eff];
    for i in 0..k_eff {
        eigvals[i] = (s_all[i] * s_all[i]) / scale;
    }

    let eigvecs_sample = rsvd_project_sample_eigvecs(&q, n, kp, &v_small, k_eff)?;

    Ok((eigvals, eigvecs_sample, row_freq, row_flip, k_eff))
}

#[pyfunction(name = "rsvd_packed_subset")]
#[pyo3(signature = (
    packed,
    n_samples,
    k,
    sample_indices=None,
    seed=42,
    power=5,
    tol=1e-1
))]
pub fn py_rsvd_packed_subset<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    k: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    seed: u64,
    power: usize,
    tol: f32,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<bool>>,
)> {
    if k == 0 {
        return Err(PyRuntimeError::new_err("k must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    if m == 0 {
        return Err(PyRuntimeError::new_err("packed matrix has zero SNP rows"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        parse_index_vec_i64_result(sidx_slice, n_samples, "sample_indices")
            .map_err(PyRuntimeError::new_err)?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }

    let packed_flat: Cow<[u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };

    let tile_cols = rsvd_tile_cols_env();
    let (eigvals, eigvecs_sample, row_freq, row_flip, k_eff) = py
        .detach(|| {
            rsvd_packed_subset(
                &packed_flat,
                m,
                bytes_per_snp,
                n_samples,
                &sample_idx,
                k,
                seed,
                power,
                tol,
                tile_cols,
            )
        })
        .map_err(map_err_string_to_py)?;

    let eval_arr = PyArray1::<f32>::zeros(py, [k_eff], false).into_bound();
    let evec_arr = PyArray2::<f32>::zeros(py, [n, k_eff], false).into_bound();
    let maf_arr = PyArray1::<f32>::zeros(py, [m], false).into_bound();
    let flip_arr = PyArray1::<bool>::zeros(py, [m], false).into_bound();

    let eval_slice = unsafe {
        eval_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("eigvals output not contiguous"))?
    };
    let evec_slice = unsafe {
        evec_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("sample eigvecs output not contiguous"))?
    };
    let maf_slice = unsafe {
        maf_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("row maf output not contiguous"))?
    };
    let flip_slice = unsafe {
        flip_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("row flip output not contiguous"))?
    };

    eval_slice.copy_from_slice(&eigvals);
    evec_slice.copy_from_slice(&eigvecs_sample);
    maf_slice.copy_from_slice(&row_freq);
    flip_slice.copy_from_slice(&row_flip);
    Ok((eval_arr, evec_arr, maf_arr, flip_arr))
}


// ============================================================
// Streaming RSVD & thread pool helpers (migrated from adamixture.rs)
// ============================================================

struct StreamRsvdConfig {
    genotype_path: String,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    delimiter: Option<String>,
    mmap_window_mb: usize,
    memory_mb: usize,
}

enum StreamSnpIter {
    Bed(BedSnpIter),
    Vcf(VcfSnpIter),
    Hmp(HmpSnpIter),
    Txt(TxtSnpIter),
}

impl StreamSnpIter {
    fn n_samples(&self) -> usize {
        match self {
            StreamSnpIter::Bed(it) => it.n_samples(),
            StreamSnpIter::Vcf(it) => it.n_samples(),
            StreamSnpIter::Hmp(it) => it.n_samples(),
            StreamSnpIter::Txt(it) => it.n_samples(),
        }
    }

    fn next_row_raw(&mut self) -> Option<(Vec<f32>, String, String)> {
        match self {
            StreamSnpIter::Bed(it) => it
                .next_snp_raw()
                .map(|(row, site)| (row, site.ref_allele, site.alt_allele)),
            StreamSnpIter::Vcf(it) => it
                .next_snp_raw()
                .map(|(row, site)| (row, site.ref_allele, site.alt_allele)),
            StreamSnpIter::Hmp(it) => it
                .next_snp_raw()
                .map(|(row, site)| (row, site.ref_allele, site.alt_allele)),
            StreamSnpIter::Txt(it) => it
                .next_snp()
                .map(|(row, site)| (row, site.ref_allele, site.alt_allele)),
        }
    }
}

fn is_dna_base(c: char) -> bool {
    matches!(c, 'A' | 'C' | 'G' | 'T')
}

fn is_simple_snp_alleles(ref_allele: &str, alt_allele: &str) -> bool {
    let r = ref_allele.trim().to_ascii_uppercase();
    let a = alt_allele.trim().to_ascii_uppercase();
    if r.len() != 1 || a.len() != 1 {
        return false;
    }
    let rc = r.chars().next().unwrap_or('N');
    let ac = a.chars().next().unwrap_or('N');
    is_dna_base(rc) && is_dna_base(ac)
}

fn normalize_numeric_row(row: &mut [f32]) {
    for g in row.iter_mut() {
        if !g.is_finite() || *g < 0.0 {
            *g = -9.0;
        } else if *g >= 3.0 {
            // Keep parity with internal missing encodings (e.g. 3 in u8 pipelines).
            *g = -9.0;
        }
    }
}

fn detect_bed_prefix(path_or_prefix: &str) -> Option<String> {
    let p = Path::new(path_or_prefix);
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let prefix: PathBuf = if matches!(ext.as_str(), "bed" | "bim" | "fam") {
        p.with_extension("")
    } else {
        p.to_path_buf()
    };
    let prefix_s = prefix.to_string_lossy().to_string();
    let bed = PathBuf::from(format!("{prefix_s}.bed"));
    let bim = PathBuf::from(format!("{prefix_s}.bim"));
    let fam = PathBuf::from(format!("{prefix_s}.fam"));
    if bed.exists() && bim.exists() && fam.exists() {
        Some(prefix_s)
    } else {
        None
    }
}

enum PackedBedStorage {
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl PackedBedStorage {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            PackedBedStorage::Mmap(mmap) => mmap,
            PackedBedStorage::Owned(buf) => buf,
        }
    }

    #[inline]
    fn label(&self) -> &'static str {
        match self {
            PackedBedStorage::Mmap(_) => "mmap",
            PackedBedStorage::Owned(_) => "owned",
        }
    }
}

struct PackedBedRsvd {
    storage: PackedBedStorage,
    payload_offset: usize,
    compact_rows: bool,
    bytes_per_snp: usize,
    n_samples: usize,
    n_snps: usize,
    n_total_sites: usize,
    active_row_idx: Vec<usize>,
    row_freq: Vec<f32>,
    row_flip: Vec<bool>,
    total_variance: f64,
    block_target_mb: Option<usize>,
}

const PACKED_BED_CACHE_MAGIC: &[u8; 8] = b"JXRSVDC1";

const PACKED_BED_CACHE_VERSION: u32 = 2;

#[inline]
fn packed_bed_cache_enabled() -> bool {
    match std::env::var("JANUSX_RSVD_BED_CACHE") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => true,
    }
}

fn packed_bed_cache_path(prefix: &str, cfg: &StreamRsvdConfig) -> PathBuf {
    PathBuf::from(format!(
        "{prefix}.rsvd_bedcache.maf{:08x}.miss{:08x}.snps{}.bin",
        cfg.maf.to_bits(),
        cfg.missing_rate.to_bits(),
        if cfg.snps_only { 1 } else { 0 }
    ))
}

fn packed_bed_compact_cache_path(prefix: &str, cfg: &StreamRsvdConfig) -> PathBuf {
    PathBuf::from(format!(
        "{prefix}.rsvd_bedpack.maf{:08x}.miss{:08x}.snps{}.bin",
        cfg.maf.to_bits(),
        cfg.missing_rate.to_bits(),
        if cfg.snps_only { 1 } else { 0 }
    ))
}

#[inline]
fn packed_bed_compact_cache_enabled() -> bool {
    match std::env::var("JANUSX_ADMX_BED_COMPACT_CACHE") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => true,
    }
}

#[inline]
fn packed_bed_owned_storage_enabled() -> bool {
    match std::env::var("JANUSX_ADMX_BED_OWNED_PACKED") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => true,
    }
}

#[inline]
fn file_signature(path: &Path) -> Option<(u64, u64)> {
    let meta = path.metadata().ok()?;
    let size = meta.len();
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos()
        .min(u64::MAX as u128) as u64;
    Some((size, mtime))
}

fn scan_bim_site_count_and_simple_mask(
    prefix: &str,
    snps_only: bool,
) -> Result<(usize, Option<Vec<bool>>), String> {
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| format!("{bim_path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut n_sites = 0usize;
    let mut simple_mask = if snps_only { Some(Vec::new()) } else { None };

    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{bim_path}:{}: {e}", line_no + 1))?;
        let cols: Vec<&str> = l.split_whitespace().collect();
        if cols.len() < 6 {
            return Err(format!(
                "Malformed BIM line at {bim_path}:{}: {l}",
                line_no + 1
            ));
        }
        if let Some(mask) = simple_mask.as_mut() {
            mask.push(is_simple_snp_alleles(cols[4], cols[5]));
        }
        n_sites += 1;
    }
    Ok((n_sites, simple_mask))
}

#[inline]
fn push_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_u64_le(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_f32_le(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn read_fixed_from_bytes<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Option<[u8; N]> {
    let end = cursor.checked_add(N)?;
    let slice = bytes.get(*cursor..end)?;
    let mut out = [0u8; N];
    out.copy_from_slice(slice);
    *cursor = end;
    Some(out)
}

#[inline]
fn read_u32_le_from_bytes(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
    Some(u32::from_le_bytes(read_fixed_from_bytes::<4>(
        bytes, cursor,
    )?))
}

#[inline]
fn read_u64_le_from_bytes(bytes: &[u8], cursor: &mut usize) -> Option<u64> {
    Some(u64::from_le_bytes(read_fixed_from_bytes::<8>(
        bytes, cursor,
    )?))
}

#[inline]
fn read_f64_le_from_bytes(bytes: &[u8], cursor: &mut usize) -> Option<f64> {
    Some(f64::from_le_bytes(read_fixed_from_bytes::<8>(
        bytes, cursor,
    )?))
}

#[inline]
fn copy_le_f32_bytes(bytes: &[u8], out: &mut [f32]) -> bool {
    if bytes.len() != out.len().saturating_mul(std::mem::size_of::<f32>()) {
        return false;
    }
    if cfg!(target_endian = "little") && (bytes.as_ptr() as usize).is_multiple_of(align_of::<f32>())
    {
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        }
        return true;
    }
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    true
}

#[inline]
fn decode_le_u32_indices(bytes: &[u8], n_total_sites: usize) -> Option<Vec<usize>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    if cfg!(target_endian = "little") && (bytes.as_ptr() as usize).is_multiple_of(align_of::<u32>())
    {
        let src = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, n) };
        for &idx_u32 in src {
            let idx = idx_u32 as usize;
            if idx >= n_total_sites {
                return None;
            }
            out.push(idx);
        }
        return Some(out);
    }
    for chunk in bytes.chunks_exact(4) {
        let idx = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize;
        if idx >= n_total_sites {
            return None;
        }
        out.push(idx);
    }
    Some(out)
}

fn try_load_packed_bed_cache(
    cache_path: &Path,
    bed_path: &Path,
    bim_path: &Path,
    fam_path: &Path,
    n_samples: usize,
    n_total_sites: usize,
    bytes_per_snp: usize,
) -> Option<(Vec<usize>, Vec<f32>, Vec<bool>, f64)> {
    if !packed_bed_cache_enabled() || !cache_path.exists() {
        return None;
    }
    let (bed_size, bed_mtime) = file_signature(bed_path)?;
    let (bim_size, bim_mtime) = file_signature(bim_path)?;
    let (fam_size, fam_mtime) = file_signature(fam_path)?;
    let mmap = mmap_readonly_file(cache_path).ok()?;
    let bytes = mmap.as_ref();
    let mut cursor = 0usize;
    let magic = read_fixed_from_bytes::<8>(bytes, &mut cursor)?;
    if &magic != PACKED_BED_CACHE_MAGIC {
        return None;
    }
    if read_u32_le_from_bytes(bytes, &mut cursor)? != PACKED_BED_CACHE_VERSION {
        return None;
    }
    if read_u64_le_from_bytes(bytes, &mut cursor)? != bed_size
        || read_u64_le_from_bytes(bytes, &mut cursor)? != bed_mtime
        || read_u64_le_from_bytes(bytes, &mut cursor)? != bim_size
        || read_u64_le_from_bytes(bytes, &mut cursor)? != bim_mtime
        || read_u64_le_from_bytes(bytes, &mut cursor)? != fam_size
        || read_u64_le_from_bytes(bytes, &mut cursor)? != fam_mtime
        || read_u64_le_from_bytes(bytes, &mut cursor)? != n_samples as u64
        || read_u64_le_from_bytes(bytes, &mut cursor)? != n_total_sites as u64
        || read_u64_le_from_bytes(bytes, &mut cursor)? != bytes_per_snp as u64
    {
        return None;
    }
    let n_active = usize::try_from(read_u64_le_from_bytes(bytes, &mut cursor)?).ok()?;
    if n_active > n_total_sites {
        return None;
    }
    let total_variance = read_f64_le_from_bytes(bytes, &mut cursor)?;
    if !total_variance.is_finite() || total_variance <= 0.0 {
        return None;
    }
    let mut row_freq = vec![0.0_f32; n_active];
    let row_freq_bytes_len = n_active.checked_mul(4)?;
    let row_freq_end = cursor.checked_add(row_freq_bytes_len)?;
    let row_freq_bytes = bytes.get(cursor..row_freq_end)?;
    if !copy_le_f32_bytes(row_freq_bytes, &mut row_freq) {
        return None;
    }
    cursor = row_freq_end;
    let flip_end = cursor.checked_add(n_active)?;
    let flip_buf = bytes.get(cursor..flip_end)?;
    let row_flip: Vec<bool> = flip_buf.iter().map(|&v| v != 0).collect();
    cursor = flip_end;
    let idx_bytes_len = n_active.checked_mul(4)?;
    let idx_end = cursor.checked_add(idx_bytes_len)?;
    let idx_bytes = bytes.get(cursor..idx_end)?;
    let active_row_idx = decode_le_u32_indices(idx_bytes, n_total_sites)?;
    cursor = idx_end;
    if cursor != bytes.len() {
        return None;
    }
    Some((active_row_idx, row_freq, row_flip, total_variance))
}

fn try_store_packed_bed_cache(
    cache_path: &Path,
    bed_path: &Path,
    bim_path: &Path,
    fam_path: &Path,
    bytes_per_snp: usize,
    n_total_sites: usize,
    data: &PackedBedRsvd,
) -> Result<(), String> {
    if !packed_bed_cache_enabled() {
        return Ok(());
    }
    let (bed_size, bed_mtime) =
        file_signature(bed_path).ok_or_else(|| "missing BED signature".to_string())?;
    let (bim_size, bim_mtime) =
        file_signature(bim_path).ok_or_else(|| "missing BIM signature".to_string())?;
    let (fam_size, fam_mtime) =
        file_signature(fam_path).ok_or_else(|| "missing FAM signature".to_string())?;
    if data
        .active_row_idx
        .iter()
        .any(|&idx| idx > u32::MAX as usize)
    {
        return Err("active_row_idx exceeds u32 range for RSVD cache".to_string());
    }
    let n_active = data.active_row_idx.len();
    let mut buf = Vec::with_capacity(8 + 4 + 10 * 8 + 8 + n_active * (4 + 1 + 4));
    buf.extend_from_slice(PACKED_BED_CACHE_MAGIC);
    push_u32_le(&mut buf, PACKED_BED_CACHE_VERSION);
    push_u64_le(&mut buf, bed_size);
    push_u64_le(&mut buf, bed_mtime);
    push_u64_le(&mut buf, bim_size);
    push_u64_le(&mut buf, bim_mtime);
    push_u64_le(&mut buf, fam_size);
    push_u64_le(&mut buf, fam_mtime);
    push_u64_le(&mut buf, data.n_samples as u64);
    push_u64_le(&mut buf, n_total_sites as u64);
    push_u64_le(&mut buf, bytes_per_snp as u64);
    push_u64_le(&mut buf, n_active as u64);
    buf.extend_from_slice(&data.total_variance.to_le_bytes());
    for &freq in data.row_freq.iter() {
        push_f32_le(&mut buf, freq);
    }
    for &flip in data.row_flip.iter() {
        buf.push(if flip { 1 } else { 0 });
    }
    for &idx in data.active_row_idx.iter() {
        push_u32_le(&mut buf, idx as u32);
    }
    let tmp_path = cache_path.with_extension("tmp");
    fs::write(&tmp_path, &buf).map_err(|e| e.to_string())?;
    fs::rename(&tmp_path, cache_path).map_err(|e| e.to_string())?;
    Ok(())
}

fn try_open_packed_bed_compact_cache_storage(
    compact_path: &Path,
    n_active: usize,
    bytes_per_snp: usize,
) -> Option<PackedBedStorage> {
    if !packed_bed_compact_cache_enabled() || !compact_path.exists() {
        return None;
    }
    let expected_size = n_active.checked_mul(bytes_per_snp)?;
    if expected_size == 0 {
        return None;
    }
    let file = File::open(compact_path).ok()?;
    let file_size = usize::try_from(file.metadata().ok()?.len()).ok()?;
    if file_size != expected_size {
        return None;
    }
    if packed_bed_owned_storage_enabled() {
        let buf = load_file_owned_exact(compact_path, expected_size).ok()?;
        return Some(PackedBedStorage::Owned(buf));
    }
    mmap_readonly_file(compact_path)
        .ok()
        .map(PackedBedStorage::Mmap)
}

fn build_compact_cache_from_active_rows(
    compact_path: &Path,
    bed_path: &Path,
    n_total_sites: usize,
    n_samples: usize,
    bytes_per_snp: usize,
    active_row_idx: &[usize],
    target_mb: Option<usize>,
) -> Result<(), String> {
    if !packed_bed_compact_cache_enabled() {
        return Err("compact BED row cache is disabled".to_string());
    }
    let mut bed_reader = BufReader::new(File::open(bed_path).map_err(|e| e.to_string())?);
    let mut header = [0u8; 3];
    bed_reader
        .read_exact(&mut header)
        .map_err(|e| e.to_string())?;
    if header != [0x6C, 0x1B, 0x01] {
        return Err("Only SNP-major BED supported".to_string());
    }

    let tmp_path = compact_path.with_extension("tmp");
    let mut writer = BufWriter::new(File::create(&tmp_path).map_err(|e| e.to_string())?);
    let chunk_rows = packed_backend_stats_chunk_rows(n_total_sites, n_samples, target_mb);
    let mut chunk_buf = Vec::<u8>::new();
    let mut active_ptr = 0usize;
    for chunk_start in (0..n_total_sites).step_by(chunk_rows) {
        check_admx_memory_limit("bed_backend/compact_cache_build")?;
        let chunk_end = (chunk_start + chunk_rows).min(n_total_sites);
        let row_count = chunk_end - chunk_start;
        chunk_buf.resize(row_count * bytes_per_snp, 0);
        bed_reader
            .read_exact(&mut chunk_buf)
            .map_err(|e| e.to_string())?;
        while active_ptr < active_row_idx.len() {
            let snp_idx = active_row_idx[active_ptr];
            if snp_idx < chunk_start {
                return Err("active_row_idx is not sorted".to_string());
            }
            if snp_idx >= chunk_end {
                break;
            }
            let local = snp_idx - chunk_start;
            let row_start = local * bytes_per_snp;
            writer
                .write_all(&chunk_buf[row_start..row_start + bytes_per_snp])
                .map_err(|e| e.to_string())?;
            active_ptr += 1;
        }
        emit_rsvd_rss_debug(
            "packed_backend/compact_cache_build",
            &format!(
                "rows={}/{} active_written={}/{}",
                chunk_end,
                n_total_sites,
                active_ptr,
                active_row_idx.len()
            ),
        );
    }
    if active_ptr != active_row_idx.len() {
        return Err(format!(
            "compact cache build wrote {} active rows, expected {}",
            active_ptr,
            active_row_idx.len()
        ));
    }
    writer.flush().map_err(|e| e.to_string())?;
    drop(writer);
    fs::rename(&tmp_path, compact_path).map_err(|e| e.to_string())?;
    Ok(())
}

#[inline]
fn env_truthy_stream_rsvd(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => false,
    }
}

#[inline]
fn admx_rsvd_kp(k_eff: usize, max_cols: usize) -> usize {
    let oversample =
        parse_positive_env_usize(&["JANUSX_ADMX_RSVD_OVERSAMPLE", "JX_ADMX_RSVD_OVERSAMPLE"])
            .unwrap_or(6);
    let kp_min =
        parse_positive_env_usize(&["JANUSX_ADMX_RSVD_KP_MIN", "JX_ADMX_RSVD_KP_MIN"]).unwrap_or(12);
    let kp_cap = parse_positive_env_usize(&["JANUSX_ADMX_RSVD_KP_MAX", "JX_ADMX_RSVD_KP_MAX"]);

    let limit = max_cols.max(1);
    let mut kp = k_eff.max(1).saturating_add(oversample).max(kp_min);
    if let Some(cap) = kp_cap {
        kp = kp.min(cap.max(k_eff.max(1)));
    }
    kp.min(limit)
}

#[inline]
fn rsvd_rss_debug_enabled() -> bool {
    env_truthy_stream_rsvd("JX_RSVD_RSS_DEBUG") || env_truthy_stream_rsvd("JX_PCA_RSS_DEBUG")
}

#[inline]
fn rsvd_time_debug_enabled() -> bool {
    env_truthy_stream_rsvd("JX_RSVD_TIME_STAGE")
        || env_truthy_stream_rsvd("JX_RSVD_TIME_DEBUG")
        || env_truthy_stream_rsvd("JX_PCA_TIME_STAGE")
}

#[inline]
fn format_debug_bytes_local(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn emit_rsvd_rss_debug(stage: &str, detail: &str) {
    if !rsvd_rss_debug_enabled() {
        return;
    }
    match process_memory_usage() {
        Some(usage) => {
            let rss = usage
                .rss_bytes
                .map(format_debug_bytes_local)
                .unwrap_or_else(|| "NA".to_string());
            let footprint = usage
                .footprint_bytes
                .map(format_debug_bytes_local)
                .unwrap_or_else(|| "NA".to_string());
            eprintln!(
                "[RSVD-DEBUG] {stage} mem={} metric={} rss={} footprint={} {detail}",
                format_debug_bytes_local(usage.current_bytes),
                usage.metric,
                rss,
                footprint,
            );
        }
        None => {
            eprintln!(
                "[RSVD-DEBUG] {stage} mem=NA metric=unavailable rss=NA footprint=NA {detail}"
            );
        }
    }
    let _ = std::io::stderr().flush();
}

fn emit_rsvd_time_debug(stage: &str, detail: &str) {
    if !rsvd_time_debug_enabled() {
        return;
    }
    println!("[RSVD-TIME] {stage} {detail}");
    let _ = std::io::stdout().flush();
}

#[inline]
fn fmt_stage_secs(v: f64) -> String {
    format!("{v:.3}s")
}

#[inline]
fn format_f32_vec_bytes(len: usize) -> String {
    format_debug_bytes_local((len.saturating_mul(std::mem::size_of::<f32>())) as u64)
}

#[inline]
fn packed_row_nonmissing_alt_sumsq_full(row: &[u8], n_samples: usize) -> (usize, f64, f64) {
    let byte_lut = packed_byte_lut();
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut non_missing = 0usize;
    let mut alt_sum = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    for &b in row.iter().take(full_bytes) {
        let idx = b as usize;
        non_missing += byte_lut.nonmiss[idx] as usize;
        alt_sum += byte_lut.alt_sum[idx] as f64;
        sq_sum += byte_lut.sq_sum[idx] as f64;
    }
    if rem > 0 {
        let codes = &byte_lut.code4[row[full_bytes] as usize];
        for &code in codes.iter().take(rem) {
            match code {
                0b00 => {
                    non_missing += 1;
                }
                0b10 => {
                    non_missing += 1;
                    alt_sum += 1.0;
                    sq_sum += 1.0;
                }
                0b11 => {
                    non_missing += 1;
                    alt_sum += 2.0;
                    sq_sum += 4.0;
                }
                _ => {}
            }
        }
    }
    (non_missing, alt_sum, sq_sum)
}

#[inline]
fn stream_rsvd_target_mb(cfg: &StreamRsvdConfig) -> Option<usize> {
    if cfg.memory_mb > 0 {
        Some(cfg.memory_mb)
    } else {
        None
    }
}

#[inline]
fn stream_rsvd_block_rows(cfg: &StreamRsvdConfig, n_cols: usize, max_rows: usize) -> usize {
    rsvd_block_rows_with_target_mb(n_cols, max_rows, stream_rsvd_target_mb(cfg))
}

#[inline]
fn packed_backend_block_rows(backend: &PackedBedRsvd, max_rows: usize) -> usize {
    rsvd_block_rows_with_target_mb(backend.n_samples, max_rows, backend.block_target_mb)
}

#[inline]
fn packed_backend_stats_chunk_rows(
    n_rows: usize,
    n_samples: usize,
    target_mb: Option<usize>,
) -> usize {
    let rows = rsvd_block_rows_with_target_mb(n_samples, n_rows, target_mb)
        .saturating_mul(16)
        .clamp(1024, 16384);
    rows.min(n_rows.max(1))
}

#[inline]
fn packed_backend_row_indices(backend: &PackedBedRsvd) -> Option<&[usize]> {
    if backend.compact_rows {
        None
    } else {
        Some(&backend.active_row_idx)
    }
}

fn packed_bed_fast_enabled() -> bool {
    match std::env::var("JANUSX_RSVD_BED_PACKED") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => true,
    }
}

fn try_build_packed_bed_backend(cfg: &StreamRsvdConfig) -> Result<Option<PackedBedRsvd>, String> {
    if !packed_bed_fast_enabled() {
        return Ok(None);
    }
    let Some(prefix) = detect_bed_prefix(&cfg.genotype_path) else {
        return Ok(None);
    };
    let t_backend = Instant::now();
    emit_rsvd_rss_debug("packed_backend/open_start", &format!("prefix={prefix}"));
    emit_rsvd_time_debug("packed_backend/start", &format!("prefix={prefix}"));
    let bed_path = PathBuf::from(format!("{prefix}.bed"));
    let bim_path = PathBuf::from(format!("{prefix}.bim"));
    let fam_path = PathBuf::from(format!("{prefix}.fam"));
    let cache_path = packed_bed_cache_path(&prefix, cfg);
    let compact_path = packed_bed_compact_cache_path(&prefix, cfg);

    let t_stage = Instant::now();
    let samples = read_fam(&prefix)?;
    let fam_s = t_stage.elapsed().as_secs_f64();
    let n_samples = samples.len();
    if n_samples == 0 {
        return Err("no samples found in PLINK input".to_string());
    }
    check_admx_memory_limit("bed_backend/after_fam")?;
    emit_rsvd_rss_debug(
        "packed_backend/fam_ready",
        &format!("prefix={} n_samples={}", prefix, n_samples),
    );
    emit_rsvd_time_debug(
        "packed_backend/fam",
        &format!("elapsed={} n_samples={}", fmt_stage_secs(fam_s), n_samples),
    );

    let t_stage = Instant::now();
    let bed_info = read_plink_bed_header_info(&bed_path, n_samples)?;
    let bed_header_s = t_stage.elapsed().as_secs_f64();
    let bytes_per_snp = bed_info.bytes_per_snp;
    let n_snps_total = bed_info.n_snps_total;
    let expected_payload = bed_info.payload_size;
    check_admx_memory_limit("bed_backend/after_bed_header")?;
    emit_rsvd_rss_debug(
        "packed_backend/bed_header",
        &format!(
            "prefix={} n_samples={} n_total_sites={} bytes_per_snp={} mapped_payload={}",
            prefix,
            n_samples,
            n_snps_total,
            bytes_per_snp,
            format_debug_bytes_local(expected_payload as u64),
        ),
    );
    emit_rsvd_time_debug(
        "packed_backend/bed_header",
        &format!(
            "elapsed={} n_samples={} n_total_sites={} bytes_per_snp={} file_size={} payload={}",
            fmt_stage_secs(bed_header_s),
            n_samples,
            n_snps_total,
            bytes_per_snp,
            format_debug_bytes_local(bed_info.file_size as u64),
            format_debug_bytes_local(expected_payload as u64),
        ),
    );

    let t_stage = Instant::now();
    let cached_meta = try_load_packed_bed_cache(
        &cache_path,
        &bed_path,
        &bim_path,
        &fam_path,
        n_samples,
        n_snps_total,
        bytes_per_snp,
    );
    let meta_cache_s = t_stage.elapsed().as_secs_f64();
    if let Some((active_row_idx, row_freq, row_flip, total_variance)) = cached_meta {
        emit_rsvd_rss_debug(
            "packed_backend/cache_hit",
            &format!(
                "prefix={} cache={} active_rows={}",
                prefix,
                cache_path.display(),
                row_freq.len(),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/meta_cache_hit",
            &format!(
                "elapsed={} cache={} active_rows={}",
                fmt_stage_secs(meta_cache_s),
                cache_path.display(),
                row_freq.len(),
            ),
        );
        if row_freq.is_empty() {
            return Err("no SNPs remain after BED filters".to_string());
        }
        let t_stage = Instant::now();
        if let Some(storage) =
            try_open_packed_bed_compact_cache_storage(&compact_path, row_freq.len(), bytes_per_snp)
        {
            let compact_open_s = t_stage.elapsed().as_secs_f64();
            emit_rsvd_rss_debug(
                "packed_backend/compact_cache_hit",
                &format!(
                    "prefix={} cache={} active_rows={} packed_payload={} storage={}",
                    prefix,
                    compact_path.display(),
                    row_freq.len(),
                    format_debug_bytes_local((row_freq.len().saturating_mul(bytes_per_snp)) as u64),
                    storage.label(),
                ),
            );
            emit_rsvd_time_debug(
                "packed_backend/compact_cache_hit",
                &format!(
                    "elapsed={} cache={} active_rows={} storage={}",
                    fmt_stage_secs(compact_open_s),
                    compact_path.display(),
                    row_freq.len(),
                    storage.label(),
                ),
            );
            emit_rsvd_time_debug(
                "packed_backend/summary",
                &format!(
                    "elapsed={} mode=meta_cache+compact_cache active_rows={} total_sites={}",
                    fmt_stage_secs(t_backend.elapsed().as_secs_f64()),
                    row_freq.len(),
                    n_snps_total,
                ),
            );
            return Ok(Some(PackedBedRsvd {
                storage,
                payload_offset: 0,
                compact_rows: true,
                bytes_per_snp,
                n_samples,
                n_snps: row_freq.len(),
                n_total_sites: n_snps_total,
                active_row_idx,
                row_freq,
                row_flip,
                total_variance,
                block_target_mb: stream_rsvd_target_mb(cfg),
            }));
        }
        let compact_open_s = t_stage.elapsed().as_secs_f64();
        emit_rsvd_rss_debug(
            "packed_backend/compact_cache_miss",
            &format!("prefix={} cache={}", prefix, compact_path.display()),
        );
        emit_rsvd_time_debug(
            "packed_backend/compact_cache_miss",
            &format!(
                "elapsed={} cache={} active_rows={}",
                fmt_stage_secs(compact_open_s),
                compact_path.display(),
                row_freq.len(),
            ),
        );
        let t_stage = Instant::now();
        match build_compact_cache_from_active_rows(
            &compact_path,
            &bed_path,
            n_snps_total,
            n_samples,
            bytes_per_snp,
            &active_row_idx,
            stream_rsvd_target_mb(cfg),
        ) {
            Ok(()) => {
                let compact_build_s = t_stage.elapsed().as_secs_f64();
                emit_rsvd_time_debug(
                    "packed_backend/compact_cache_build",
                    &format!(
                        "elapsed={} cache={} active_rows={}",
                        fmt_stage_secs(compact_build_s),
                        compact_path.display(),
                        row_freq.len(),
                    ),
                );
                let t_stage = Instant::now();
                if let Some(storage) = try_open_packed_bed_compact_cache_storage(
                    &compact_path,
                    row_freq.len(),
                    bytes_per_snp,
                ) {
                    let compact_ready_s = t_stage.elapsed().as_secs_f64();
                    emit_rsvd_rss_debug(
                        "packed_backend/compact_cache_built",
                        &format!(
                            "prefix={} cache={} active_rows={} storage={}",
                            prefix,
                            compact_path.display(),
                            row_freq.len(),
                            storage.label(),
                        ),
                    );
                    emit_rsvd_time_debug(
                        "packed_backend/compact_cache_ready",
                        &format!(
                            "elapsed={} cache={} active_rows={} storage={}",
                            fmt_stage_secs(compact_ready_s),
                            compact_path.display(),
                            row_freq.len(),
                            storage.label(),
                        ),
                    );
                    emit_rsvd_time_debug(
                        "packed_backend/summary",
                        &format!(
                            "elapsed={} mode=meta_cache+compact_rebuild active_rows={} total_sites={}",
                            fmt_stage_secs(t_backend.elapsed().as_secs_f64()),
                            row_freq.len(),
                            n_snps_total,
                        ),
                    );
                    return Ok(Some(PackedBedRsvd {
                        storage,
                        payload_offset: 0,
                        compact_rows: true,
                        bytes_per_snp,
                        n_samples,
                        n_snps: row_freq.len(),
                        n_total_sites: n_snps_total,
                        active_row_idx,
                        row_freq,
                        row_flip,
                        total_variance,
                        block_target_mb: stream_rsvd_target_mb(cfg),
                    }));
                }
            }
            Err(e) => {
                let compact_build_s = t_stage.elapsed().as_secs_f64();
                emit_rsvd_rss_debug(
                    "packed_backend/compact_cache_build_failed",
                    &format!("prefix={} error={}", prefix, e),
                );
                emit_rsvd_time_debug(
                    "packed_backend/compact_cache_build_failed",
                    &format!(
                        "elapsed={} cache={} error={}",
                        fmt_stage_secs(compact_build_s),
                        compact_path.display(),
                        e,
                    ),
                );
            }
        }
        let t_stage = Instant::now();
        let mmap = mmap_readonly_file(&bed_path)?;
        let mmap_s = t_stage.elapsed().as_secs_f64();
        emit_rsvd_rss_debug(
            "packed_backend/original_mmap_fallback",
            &format!(
                "prefix={} active_rows={} mapped_payload={}",
                prefix,
                row_freq.len(),
                format_debug_bytes_local(expected_payload as u64),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/original_mmap_fallback",
            &format!(
                "elapsed={} active_rows={} mapped_payload={}",
                fmt_stage_secs(mmap_s),
                row_freq.len(),
                format_debug_bytes_local(expected_payload as u64),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/summary",
            &format!(
                "elapsed={} mode=meta_cache+original_mmap active_rows={} total_sites={}",
                fmt_stage_secs(t_backend.elapsed().as_secs_f64()),
                row_freq.len(),
                n_snps_total,
            ),
        );
        return Ok(Some(PackedBedRsvd {
            storage: PackedBedStorage::Mmap(mmap),
            payload_offset: 3,
            compact_rows: false,
            bytes_per_snp,
            n_samples,
            n_snps: row_freq.len(),
            n_total_sites: n_snps_total,
            active_row_idx,
            row_freq,
            row_flip,
            total_variance,
            block_target_mb: stream_rsvd_target_mb(cfg),
        }));
    }
    emit_rsvd_time_debug(
        "packed_backend/meta_cache_miss",
        &format!(
            "elapsed={} cache={} total_sites={}",
            fmt_stage_secs(meta_cache_s),
            cache_path.display(),
            n_snps_total,
        ),
    );

    let t_stage = Instant::now();
    let (bim_sites, simple_snp_mask) = scan_bim_site_count_and_simple_mask(&prefix, cfg.snps_only)?;
    let bim_scan_s = t_stage.elapsed().as_secs_f64();
    if bim_sites != n_snps_total {
        return Err(format!(
            "BED/BIM site count mismatch: bed={n_snps_total}, bim={bim_sites}"
        ));
    }
    check_admx_memory_limit("bed_backend/after_bim_scan")?;
    emit_rsvd_rss_debug(
        "packed_backend/cache_miss_bim_ready",
        &format!(
            "prefix={} n_total_sites={} snps_only={}",
            prefix, n_snps_total, cfg.snps_only
        ),
    );
    emit_rsvd_time_debug(
        "packed_backend/bim_scan",
        &format!(
            "elapsed={} n_total_sites={} snps_only={}",
            fmt_stage_secs(bim_scan_s),
            n_snps_total,
            cfg.snps_only,
        ),
    );

    let mut active_row_idx: Vec<usize> = Vec::with_capacity(n_snps_total);
    let mut row_freq: Vec<f32> = Vec::with_capacity(n_snps_total);
    let mut row_flip: Vec<bool> = Vec::with_capacity(n_snps_total);
    let mut total_ss = 0.0_f64;
    let mut total_var = 0.0_f64;
    let n_samples_f64 = n_samples as f64;
    let chunk_rows =
        packed_backend_stats_chunk_rows(n_snps_total, n_samples, stream_rsvd_target_mb(cfg));
    let t_stage = Instant::now();
    let mut bed_reader = open_plink_bed_reader_after_header(&bed_path)?;
    let compact_tmp_path = compact_path.with_extension("tmp");
    let mut compact_writer = if packed_bed_compact_cache_enabled() {
        match File::create(&compact_tmp_path) {
            Ok(file) => Some(BufWriter::new(file)),
            Err(e) => {
                emit_rsvd_rss_debug(
                    "packed_backend/compact_cache_create_failed",
                    &format!(
                        "prefix={} cache={} error={}",
                        prefix,
                        compact_path.display(),
                        e
                    ),
                );
                None
            }
        }
    } else {
        None
    };
    let scan_setup_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "packed_backend/scan_setup",
        &format!(
            "elapsed={} chunk_rows={} compact_cache_enabled={}",
            fmt_stage_secs(scan_setup_s),
            chunk_rows,
            packed_bed_compact_cache_enabled(),
        ),
    );
    let mut chunk_buf = Vec::<u8>::new();
    let t_stage = Instant::now();
    for chunk_start in (0..n_snps_total).step_by(chunk_rows) {
        check_admx_memory_limit("bed_backend/filter_scan")?;
        let chunk_end = (chunk_start + chunk_rows).min(n_snps_total);
        let row_count = chunk_end - chunk_start;
        chunk_buf.resize(row_count * bytes_per_snp, 0);
        bed_reader
            .read_exact(&mut chunk_buf)
            .map_err(|e| e.to_string())?;
        let keep_rows: Vec<Option<(usize, usize, f32, bool, f64, f64)>> = (0..row_count)
            .into_par_iter()
            .map(|local_idx| {
                let snp_idx = chunk_start + local_idx;
                if cfg.snps_only
                    && !simple_snp_mask
                        .as_ref()
                        .and_then(|mask| mask.get(snp_idx))
                        .copied()
                        .unwrap_or(false)
                {
                    return None;
                }
                let row_start = local_idx * bytes_per_snp;
                let row = &chunk_buf[row_start..row_start + bytes_per_snp];
                let (non_missing, alt_sum, sq_sum) =
                    packed_row_nonmissing_alt_sumsq_full(row, n_samples);

                let miss_rate = 1.0_f64 - (non_missing as f64 / n_samples_f64);
                if !crate::gfreader::rate_within_threshold(miss_rate, cfg.missing_rate) {
                    return None;
                }
                if non_missing == 0 {
                    if cfg.maf > 0.0 {
                        return None;
                    }
                    return Some((local_idx, snp_idx, 0.0_f32, false, 0.0_f64, 0.0_f64));
                }

                let p = alt_sum / (2.0 * non_missing as f64);
                let flip = p > 0.5;
                let p_minor = if flip { 1.0 - p } else { p };
                if !crate::gfreader::maf_passes_threshold(p_minor, cfg.maf) {
                    return None;
                }
                let row_mean = alt_sum / non_missing as f64;
                let row_ss = (sq_sum - alt_sum * row_mean).max(0.0_f64);
                let row_var = (2.0_f64 * p_minor * (1.0_f64 - p_minor)).max(0.0_f64);
                Some((local_idx, snp_idx, p_minor as f32, flip, row_ss, row_var))
            })
            .collect();
        for item in keep_rows.into_iter().flatten() {
            let (local_idx, snp_idx, freq, flip, row_ss, row_var) = item;
            active_row_idx.push(snp_idx);
            row_freq.push(freq);
            row_flip.push(flip);
            total_ss += row_ss;
            total_var += row_var;
            if let Some(writer) = compact_writer.as_mut() {
                let row_start = local_idx * bytes_per_snp;
                writer
                    .write_all(&chunk_buf[row_start..row_start + bytes_per_snp])
                    .map_err(|e| e.to_string())?;
            }
        }
        if rsvd_rss_debug_enabled()
            && (((chunk_start / chunk_rows) % 64 == 0) || chunk_end == n_snps_total)
        {
            emit_rsvd_rss_debug(
                "packed_backend/filter_scan",
                &format!(
                    "rows={}/{} active_rows={}",
                    chunk_end,
                    n_snps_total,
                    row_freq.len()
                ),
            );
        }
    }
    let filter_scan_s = t_stage.elapsed().as_secs_f64();

    let n_snps = row_freq.len();
    if n_snps == 0 {
        return Err("no SNPs remain after BED filters".to_string());
    }
    if !(total_var.is_finite() && total_var > 0.0) {
        return Err("invalid RSVD packed BED total variance denominator".to_string());
    }
    let total_variance = total_ss / total_var;
    if !(total_variance.is_finite() && total_variance > 0.0) {
        return Err("invalid RSVD packed BED trace(K)".to_string());
    }
    emit_rsvd_rss_debug(
        "packed_backend/filter_done",
        &format!(
            "prefix={} active_rows={} dropped={} row_freq={} row_flip={} traceK={:.6e}",
            prefix,
            n_snps,
            n_snps_total.saturating_sub(n_snps),
            format_debug_bytes_local((row_freq.len() * std::mem::size_of::<f32>()) as u64),
            format_debug_bytes_local((row_flip.len() * std::mem::size_of::<bool>()) as u64),
            total_variance,
        ),
    );
    emit_rsvd_time_debug(
        "packed_backend/filter_scan",
        &format!(
            "elapsed={} active_rows={} dropped={} traceK={:.6e} chunk_rows={}",
            fmt_stage_secs(filter_scan_s),
            n_snps,
            n_snps_total.saturating_sub(n_snps),
            total_variance,
            chunk_rows,
        ),
    );
    let compact_ready = if let Some(mut writer) = compact_writer {
        writer.flush().map_err(|e| e.to_string())?;
        drop(writer);
        fs::rename(&compact_tmp_path, &compact_path).map_err(|e| e.to_string())?;
        true
    } else {
        false
    };
    let storage;
    let payload_offset;
    let compact_rows;
    let expected_compact_payload = n_snps
        .checked_mul(bytes_per_snp)
        .ok_or_else(|| "compact BED payload size overflow".to_string())?;
    let t_stage = Instant::now();
    if compact_ready {
        storage = try_open_packed_bed_compact_cache_storage(&compact_path, n_snps, bytes_per_snp)
            .ok_or_else(|| "failed to open compact BED row cache after build".to_string())?;
        payload_offset = 0;
        compact_rows = true;
        let storage_ready_s = t_stage.elapsed().as_secs_f64();
        emit_rsvd_rss_debug(
            "packed_backend/compact_cache_ready",
            &format!(
                "prefix={} cache={} packed_payload={} storage={}",
                prefix,
                compact_path.display(),
                format_debug_bytes_local(expected_compact_payload as u64),
                storage.label(),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/compact_cache_ready",
            &format!(
                "elapsed={} cache={} packed_payload={} storage={}",
                fmt_stage_secs(storage_ready_s),
                compact_path.display(),
                format_debug_bytes_local(expected_compact_payload as u64),
                storage.label(),
            ),
        );
    } else {
        let mmap = mmap_readonly_file(&bed_path)?;
        storage = PackedBedStorage::Mmap(mmap);
        payload_offset = 3;
        compact_rows = false;
        let storage_ready_s = t_stage.elapsed().as_secs_f64();
        emit_rsvd_rss_debug(
            "packed_backend/original_mmap_ready",
            &format!(
                "prefix={} mapped_payload={}",
                prefix,
                format_debug_bytes_local(expected_payload as u64),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/original_mmap_ready",
            &format!(
                "elapsed={} mapped_payload={}",
                fmt_stage_secs(storage_ready_s),
                format_debug_bytes_local(expected_payload as u64),
            ),
        );
    }
    let backend = PackedBedRsvd {
        storage,
        payload_offset,
        compact_rows,
        bytes_per_snp,
        n_samples,
        n_snps,
        n_total_sites: n_snps_total,
        active_row_idx,
        row_freq,
        row_flip,
        total_variance,
        block_target_mb: stream_rsvd_target_mb(cfg),
    };
    let t_stage = Instant::now();
    let _ = try_store_packed_bed_cache(
        &cache_path,
        &bed_path,
        &bim_path,
        &fam_path,
        bytes_per_snp,
        n_snps_total,
        &backend,
    );
    let cache_store_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "packed_backend/cache_store",
        &format!(
            "elapsed={} cache={} active_rows={}",
            fmt_stage_secs(cache_store_s),
            cache_path.display(),
            backend.n_snps,
        ),
    );
    emit_rsvd_time_debug(
        "packed_backend/summary",
        &format!(
            "elapsed={} mode=build active_rows={} total_sites={} compact_rows={}",
            fmt_stage_secs(t_backend.elapsed().as_secs_f64()),
            backend.n_snps,
            backend.n_total_sites,
            backend.compact_rows,
        ),
    );
    Ok(Some(backend))
}

fn compute_at_random_omega_packed(
    backend: &PackedBedRsvd,
    kp: usize,
    seed: u64,
) -> Result<Vec<f32>, String> {
    let sample_idx: Vec<usize> = (0..backend.n_samples).collect();
    let packed_view = PackedRsvdView {
        packed_flat: &backend.storage.as_slice()[backend.payload_offset..],
        bytes_per_snp: backend.bytes_per_snp,
        n_samples: backend.n_samples,
        row_freq: &backend.row_freq,
        row_flip: &backend.row_flip,
        sample_idx: &sample_idx,
        packed_row_indices: packed_backend_row_indices(backend),
        block_target_mb: backend.block_target_mb,
    };
    rsvd_packed_compute_at_random_omega(packed_view, kp, seed)
}

fn compute_ata_omega_packed(
    backend: &PackedBedRsvd,
    omega: &[f32], // (n, kp)
    kp: usize,
    timing: Option<&mut RsvdKernelTiming>,
) -> Result<Vec<f32>, String> {
    let sample_idx: Vec<usize> = (0..backend.n_samples).collect();
    let packed_view = PackedRsvdView {
        packed_flat: &backend.storage.as_slice()[backend.payload_offset..],
        bytes_per_snp: backend.bytes_per_snp,
        n_samples: backend.n_samples,
        row_freq: &backend.row_freq,
        row_flip: &backend.row_flip,
        sample_idx: &sample_idx,
        packed_row_indices: packed_backend_row_indices(backend),
        block_target_mb: backend.block_target_mb,
    };
    rsvd_packed_compute_ata_omega(packed_view, omega, kp, timing)
}

fn compute_gram_aq_packed(
    backend: &PackedBedRsvd,
    q: &[f32],
    kp: usize,
) -> Result<Vec<f64>, String> {
    let sample_idx: Vec<usize> = (0..backend.n_samples).collect();
    let packed_view = PackedRsvdView {
        packed_flat: &backend.storage.as_slice()[backend.payload_offset..],
        bytes_per_snp: backend.bytes_per_snp,
        n_samples: backend.n_samples,
        row_freq: &backend.row_freq,
        row_flip: &backend.row_flip,
        sample_idx: &sample_idx,
        packed_row_indices: packed_backend_row_indices(backend),
        block_target_mb: backend.block_target_mb,
    };
    rsvd_packed_compute_gram_aq(packed_view, q, kp)
}

fn open_stream_iter(cfg: &StreamRsvdConfig) -> Result<StreamSnpIter, String> {
    let path = cfg.genotype_path.as_str();
    let path_lower = path.to_ascii_lowercase();
    if let Some(prefix) = detect_bed_prefix(path) {
        let it = if cfg.mmap_window_mb > 0 {
            BedSnpIter::new_with_fill_window(
                &prefix,
                0.0,
                1.0,
                false,
                false,
                0.02,
                cfg.mmap_window_mb,
            )?
        } else {
            BedSnpIter::new_with_fill(&prefix, 0.0, 1.0, false, false, 0.02)?
        };
        return Ok(StreamSnpIter::Bed(it));
    }
    if path_lower.ends_with(".vcf") || path_lower.ends_with(".vcf.gz") {
        return Ok(StreamSnpIter::Vcf(VcfSnpIter::new_with_fill(
            path, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    if path_lower.ends_with(".hmp") || path_lower.ends_with(".hmp.gz") {
        return Ok(StreamSnpIter::Hmp(HmpSnpIter::new_with_fill(
            path, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    Ok(StreamSnpIter::Txt(TxtSnpIter::new(
        path,
        cfg.delimiter.as_deref(),
    )?))
}

fn for_each_processed_row<F>(
    cfg: &StreamRsvdConfig,
    mut on_row: F,
) -> Result<(usize, usize), String>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
{
    let mut it = open_stream_iter(cfg)?;
    let n = it.n_samples();
    if n == 0 {
        return Err("no samples found in genotype input".to_string());
    }

    let mut row_idx: usize = 0;
    while let Some((mut row, mut ref_allele, mut alt_allele)) = it.next_row_raw() {
        if row.len() != n {
            return Err(format!(
                "inconsistent sample count while streaming: expected {n}, got {}",
                row.len()
            ));
        }
        normalize_numeric_row(&mut row);

        if cfg.snps_only && !is_simple_snp_alleles(&ref_allele, &alt_allele) {
            continue;
        }

        let keep = process_snp_row(
            &mut row,
            &mut ref_allele,
            &mut alt_allele,
            cfg.maf,
            cfg.missing_rate,
            true,
            false,
            0.02,
        );
        if !keep {
            continue;
        }

        on_row(row_idx, &row)?;
        row_idx += 1;
    }
    Ok((row_idx, n))
}

#[inline]
fn checked_cblas_dim_local(v: usize, label: &str) -> Result<CblasInt, String> {
    if v > CblasInt::MAX as usize {
        return Err(format!("dimension overflow for {label}: {v}"));
    }
    Ok(v as CblasInt)
}

fn for_each_processed_centered_block<F>(
    cfg: &StreamRsvdConfig,
    row_freq: &[f32],
    block_rows: usize,
    timing: Option<&mut RsvdKernelTiming>,
    mut on_block: F,
) -> Result<(usize, usize), String>
where
    F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
{
    let mut it = open_stream_iter(cfg)?;
    let n = it.n_samples();
    if n == 0 {
        return Err("no samples found in genotype input".to_string());
    }
    let block_rows_use = block_rows.max(1);
    let mut block = vec![0.0_f32; block_rows_use * n];
    let mut row_idx: usize = 0;
    let mut filled: usize = 0;
    let measure_timing = timing.is_some();
    let mut decode_s = 0.0_f64;
    let mut gemm_s = 0.0_f64;

    while let Some((mut row, mut ref_allele, mut alt_allele)) = it.next_row_raw() {
        let t_decode = if measure_timing {
            Some(Instant::now())
        } else {
            None
        };
        if row.len() != n {
            return Err(format!(
                "inconsistent sample count while streaming: expected {n}, got {}",
                row.len()
            ));
        }
        normalize_numeric_row(&mut row);

        if cfg.snps_only && !is_simple_snp_alleles(&ref_allele, &alt_allele) {
            continue;
        }

        let keep = process_snp_row(
            &mut row,
            &mut ref_allele,
            &mut alt_allele,
            cfg.maf,
            cfg.missing_rate,
            true,
            false,
            0.02,
        );
        if !keep {
            continue;
        }
        if row_idx >= row_freq.len() {
            return Err("streamed row count exceeded expected row_freq length".to_string());
        }

        let mean_g = 2.0_f32 * row_freq[row_idx];
        let dst = &mut block[filled * n..(filled + 1) * n];
        for col in 0..n {
            dst[col] = row[col] - mean_g;
        }
        if let Some(t0) = t_decode {
            decode_s += t0.elapsed().as_secs_f64();
        }
        filled += 1;
        row_idx += 1;

        if filled == block_rows_use {
            let t_gemm = if measure_timing {
                Some(Instant::now())
            } else {
                None
            };
            on_block(row_idx - filled, filled, &block[..filled * n])?;
            if let Some(t0) = t_gemm {
                gemm_s += t0.elapsed().as_secs_f64();
            }
            filled = 0;
        }
    }

    if filled > 0 {
        let t_gemm = if measure_timing {
            Some(Instant::now())
        } else {
            None
        };
        on_block(row_idx - filled, filled, &block[..filled * n])?;
        if let Some(t0) = t_gemm {
            gemm_s += t0.elapsed().as_secs_f64();
        }
    }
    if let Some(t) = timing {
        t.decode_s += decode_s;
        t.gemm_s += gemm_s;
    }
    Ok((row_idx, n))
}

#[inline]
fn accum_gram_lower_f64_local(
    block: &[f32],
    rows: usize,
    cols: usize,
    gram: &mut [f64],
    gram_block: &mut [f32],
) -> Result<(), String> {
    debug_assert_eq!(block.len(), rows * cols);
    debug_assert_eq!(gram.len(), cols * cols);
    debug_assert_eq!(gram_block.len(), cols * cols);
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    gram_block.fill(0.0);
    let rows_i = checked_cblas_dim_local(rows, "rows")?;
    let cols_i = checked_cblas_dim_local(cols, "cols")?;
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            cols_i,
            cols_i,
            rows_i,
            1.0_f32,
            block.as_ptr(),
            cols_i,
            block.as_ptr(),
            cols_i,
            0.0_f32,
            gram_block.as_mut_ptr(),
            cols_i,
        );
    }
    for i in 0..(cols * cols) {
        gram[i] += gram_block[i] as f64;
    }
    Ok(())
}

#[inline]
fn subtract_scaled_inplace_f32(dst: &mut [f32], alpha: f32, rhs: &[f32]) {
    debug_assert_eq!(dst.len(), rhs.len());
    dst.par_iter_mut()
        .zip(rhs.par_iter())
        .for_each(|(yi, qi)| *yi -= alpha * *qi);
}

fn compute_at_random_omega_stream(
    cfg: &StreamRsvdConfig,
    row_freq: &[f32], // (m,)
    m: usize,
    n: usize,
    kp: usize,
    seed: u64,
) -> Result<Vec<f32>, String> {
    if row_freq.len() != m {
        return Err("row_freq length mismatch in streaming A^T*Omega_random".to_string());
    }

    let mut y = vec![0.0_f32; n * kp]; // (n, kp)
    let block_rows = stream_rsvd_block_rows(cfg, n, m);
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega_block = vec![0.0_f32; block_rows * kp];
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let (seen, seen_n) = for_each_processed_centered_block(
        cfg,
        row_freq,
        block_rows,
        None,
        |_row_start, rows_here, block| {
            let rows_i = checked_cblas_dim_local(rows_here, "rows_here")?;
            let omega_cur = &mut omega_block[..rows_here * kp];
            fill_random_omega_block(&mut rng, omega_cur);
            let _ = rows_i;
            row_major_block_t_mul_mat_accum_f32(block, rows_here, n, omega_cur, kp, &mut y, None);
            Ok(())
        },
    )?;
    if seen_n != n || seen != m {
        return Err(format!(
            "A^T*Omega_random pass row mismatch: seen={seen}, expected={m}, n_seen={seen_n}, n_expected={n}"
        ));
    }
    Ok(y)
}

fn compute_ata_omega_stream(
    cfg: &StreamRsvdConfig,
    omega: &[f32],    // (n, kp)
    row_freq: &[f32], // (m,)
    m: usize,
    n: usize,
    kp: usize,
    timing: Option<&mut RsvdKernelTiming>,
) -> Result<Vec<f32>, String> {
    if omega.len() != n * kp {
        return Err("omega shape mismatch in streaming A^T*(A*Omega)".to_string());
    }
    if row_freq.len() != m {
        return Err("row_freq length mismatch in streaming A^T*(A*Omega)".to_string());
    }

    let mut out = vec![0.0_f32; n * kp];
    let block_rows = stream_rsvd_block_rows(cfg, n, m);
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let mut g_block = vec![0.0_f32; block_rows * kp];
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let (seen, seen_n) = for_each_processed_centered_block(
        cfg,
        row_freq,
        block_rows,
        timing,
        |_row_start, rows_here, block| {
            let rows_i = checked_cblas_dim_local(rows_here, "rows_here")?;
            let cur_g = &mut g_block[..rows_here * kp];
            row_major_block_mul_mat_f32(block, rows_here, n, omega, kp, cur_g, None);
            let _ = rows_i;
            row_major_block_t_mul_mat_accum_f32(block, rows_here, n, cur_g, kp, &mut out, None);
            Ok(())
        },
    )?;
    if seen_n != n || seen != m {
        return Err(format!(
            "A^T*(A*Omega) pass row mismatch: seen={seen}, expected={m}, n_seen={seen_n}, n_expected={n}"
        ));
    }
    Ok(out)
}

fn compute_gram_aq_stream(
    cfg: &StreamRsvdConfig,
    q: &[f32],        // (n, kp)
    row_freq: &[f32], // (m,)
    m: usize,
    n: usize,
    kp: usize,
) -> Result<Vec<f64>, String> {
    if q.len() != n * kp {
        return Err("q shape mismatch in streaming gram(AQ)".to_string());
    }
    if row_freq.len() != m {
        return Err("row_freq length mismatch in streaming gram(AQ)".to_string());
    }

    let block_rows = stream_rsvd_block_rows(cfg, n, m);
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let mut gram = vec![0.0_f64; kp * kp];
    let mut gram_block = vec![0.0_f32; kp * kp];
    let mut g_block = vec![0.0_f32; block_rows * kp];
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let (seen, seen_n) = for_each_processed_centered_block(
        cfg,
        row_freq,
        block_rows,
        None,
        |_row_start, rows_here, block| {
            let rows_i = checked_cblas_dim_local(rows_here, "rows_here")?;
            let cur_g = &mut g_block[..rows_here * kp];
            row_major_block_mul_mat_f32(block, rows_here, n, q, kp, cur_g, None);
            let _ = rows_i;
            accum_gram_lower_f64_local(cur_g, rows_here, kp, &mut gram, &mut gram_block)?;
            Ok(())
        },
    )?;
    if seen_n != n || seen != m {
        return Err(format!(
            "gram(AQ) pass row mismatch: seen={seen}, expected={m}, n_seen={seen_n}, n_expected={n}"
        ));
    }
    for i in 0..kp {
        for j in 0..i {
            gram[j * kp + i] = gram[i * kp + j];
        }
    }
    Ok(gram)
}

#[pyfunction]
#[pyo3(signature = (
    genotype_path,
    k,
    seed=42,
    power=5,
    tol=1e-1,
    snps_only=true,
    maf=0.02,
    missing_rate=0.05,
    delimiter=None,
    mmap_window_mb=0,
    memory_mb=0
))]
pub fn admx_rsvd_stream_sample<'py>(
    py: Python<'py>,
    genotype_path: String,
    k: usize,
    seed: u64,
    power: usize,
    tol: f32,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    delimiter: Option<String>,
    mmap_window_mb: usize,
    memory_mb: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>, f64)> {
    if k == 0 {
        return Err(PyRuntimeError::new_err("k must be > 0"));
    }
    if !(0.0..=0.5).contains(&maf) {
        return Err(PyRuntimeError::new_err("maf must be within [0, 0.5]"));
    }
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(PyRuntimeError::new_err(
            "missing_rate must be within [0, 1]",
        ));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }

    let cfg = StreamRsvdConfig {
        genotype_path,
        snps_only,
        maf,
        missing_rate,
        delimiter,
        mmap_window_mb,
        memory_mb,
    };
    let t_total = Instant::now();
    let mut t_stage = Instant::now();

    let packed_backend = try_build_packed_bed_backend(&cfg).map_err(PyRuntimeError::new_err)?;
    let packed_backend_s = t_stage.elapsed().as_secs_f64();
    if let Some(backend) = packed_backend.as_ref() {
        let (eigvals, eigvecs_sample, total_variance) =
            rsvd_stream_sample_packed_impl(backend, k, seed, power, tol, packed_backend_s)
                .map_err(PyRuntimeError::new_err)?;
        let k_eff = eigvals.len();
        let eval_arr = row_freq_to_pyarray(py, &eigvals);
        let evec_arr = vec_to_pyarray2_f32(py, backend.n_samples, k_eff, &eigvecs_sample)?;
        return Ok((eval_arr, evec_arr, total_variance));
    }

    let mut row_freq: Vec<f32> = Vec::new();
    let mut varsum: f64 = 0.0;
    let mut total_ss: f64 = 0.0;
    let (m, n) = for_each_processed_row(&cfg, |_idx, row| {
        let mut alt_sum = 0.0_f64;
        let mut non_missing = 0usize;
        let mut sq_sum = 0.0_f64;
        for &g in row.iter() {
            if g >= 0.0 && g.is_finite() {
                alt_sum += g as f64;
                sq_sum += (g as f64) * (g as f64);
                non_missing += 1;
            }
        }
        let freq = if non_missing > 0 {
            (alt_sum / (2.0 * non_missing as f64)) as f32
        } else {
            0.0_f32
        };
        varsum += 2.0_f64 * (freq as f64) * (1.0_f64 - freq as f64);
        if non_missing > 0 {
            let row_mean = alt_sum / non_missing as f64;
            total_ss += (sq_sum - alt_sum * row_mean).max(0.0_f64);
        }
        row_freq.push(freq);
        Ok(())
    })
    .map_err(PyRuntimeError::new_err)?;
    let backend_s = t_stage.elapsed().as_secs_f64();

    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "no SNPs passed filtering in streaming RSVD",
        ));
    }
    if n == 0 {
        return Err(PyRuntimeError::new_err(
            "no samples available for streaming RSVD",
        ));
    }
    if !(varsum.is_finite() && varsum > 0.0) {
        return Err(PyRuntimeError::new_err(
            "invalid scaling denominator in streaming RSVD (varsum <= 0)",
        ));
    }
    let total_variance = total_ss / varsum;
    if !(total_variance.is_finite() && total_variance > 0.0) {
        return Err(PyRuntimeError::new_err(
            "invalid total variance in streaming RSVD (trace(K) <= 0)",
        ));
    }

    let k_eff = k.min(n);
    let kp = admx_rsvd_kp(k_eff, m.max(1).min(n.max(1)));
    let block_rows = if let Some(backend) = packed_backend.as_ref() {
        packed_backend_block_rows(backend, m)
    } else {
        stream_rsvd_block_rows(&cfg, n, m)
    };
    let backend_extra = if let Some(backend) = packed_backend.as_ref() {
        format!(
            " total_sites={} active_row_idx={}",
            backend.n_total_sites,
            format_debug_bytes_local(
                (backend.active_row_idx.len() * std::mem::size_of::<usize>()) as u64
            ),
        )
    } else {
        String::new()
    };
    emit_rsvd_rss_debug(
        "stream_sample/backend_ready",
        &format!(
            "mode={} n_samples={} n_snps={} kp={} row_freq={} varsum={:.6e} traceK={:.6e}{}",
            if packed_backend.is_some() {
                "packed_mmap"
            } else {
                "stream"
            },
            n,
            m,
            kp,
            format_f32_vec_bytes(row_freq.len()),
            varsum,
            total_variance,
            backend_extra,
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/backend_ready",
        &format!("elapsed={}", fmt_stage_secs(backend_s)),
    );

    emit_rsvd_rss_debug(
        "stream_sample/omega_ready",
        &format!(
            "omega_mode=random_block rows={} kp={} omega_scratch={}",
            block_rows,
            kp,
            format_f32_vec_bytes(block_rows * kp)
        ),
    );
    t_stage = Instant::now();
    let mut y = if let Some(backend) = packed_backend.as_ref() {
        compute_at_random_omega_packed(backend, kp, seed)
    } else {
        compute_at_random_omega_stream(&cfg, &row_freq, m, n, kp, seed)
    }
    .map_err(PyRuntimeError::new_err)?;
    let init_proj_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/y_ready",
        &format!(
            "y_shape=({}, {}) y={}",
            n,
            kp,
            format_f32_vec_bytes(y.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/init_proj",
        &format!("elapsed={}", fmt_stage_secs(init_proj_s)),
    );
    t_stage = Instant::now();
    let (mut q, _, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;
    let init_q_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/q_ready",
        &format!(
            "q_shape=({}, {}) q={}",
            n,
            kp,
            format_f32_vec_bytes(q.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/init_q",
        &format!("elapsed={}", fmt_stage_secs(init_q_s)),
    );

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    let mut power_mul_s = 0.0_f64;
    let mut power_mul_decode_s = 0.0_f64;
    let mut power_mul_gemm_s = 0.0_f64;
    let mut power_qr_s = 0.0_f64;
    let mut power_iters_done = 0usize;
    let measure_kernel_timing = rsvd_time_debug_enabled();
    for it in 0..power {
        let t_power_mul = Instant::now();
        let mut kernel_timing = RsvdKernelTiming::default();
        y = if let Some(backend) = packed_backend.as_ref() {
            compute_ata_omega_packed(
                backend,
                &q,
                kp,
                if measure_kernel_timing {
                    Some(&mut kernel_timing)
                } else {
                    None
                },
            )
        } else {
            compute_ata_omega_stream(
                &cfg,
                &q,
                &row_freq,
                m,
                n,
                kp,
                if measure_kernel_timing {
                    Some(&mut kernel_timing)
                } else {
                    None
                },
            )
        }
        .map_err(PyRuntimeError::new_err)?;
        power_mul_s += t_power_mul.elapsed().as_secs_f64();
        power_mul_decode_s += kernel_timing.decode_s;
        power_mul_gemm_s += kernel_timing.gemm_s;
        subtract_scaled_inplace_f32(&mut y, alpha, &q);
        let t_power_qr = Instant::now();
        let (q_new, s_y, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;
        power_qr_s += t_power_qr.elapsed().as_secs_f64();
        q = q_new;
        power_iters_done = it + 1;

        if it > 0 {
            let mut max_rel = 0.0_f32;
            for i in 0..k_eff {
                let sk_now = s_y[i] + alpha;
                let denom = sk_now.max(1e-12);
                let rel = ((sk_now - sk[i]).abs()) / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                sk[i] = sk_now;
            }
            if max_rel < tol {
                break;
            }
        } else {
            for i in 0..kp {
                sk[i] = s_y[i] + alpha;
            }
        }

        let tail = s_y[kp - 1];
        if alpha < tail {
            alpha = 0.5 * (alpha + tail);
        }
    }

    t_stage = Instant::now();
    let gram = if let Some(backend) = packed_backend.as_ref() {
        compute_gram_aq_packed(backend, &q, kp).map_err(PyRuntimeError::new_err)?
    } else {
        compute_gram_aq_stream(&cfg, &q, &row_freq, m, n, kp).map_err(PyRuntimeError::new_err)?
    };
    let gram_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/gram_ready",
        &format!(
            "gram_shape=({}, {}) gram={}",
            kp,
            kp,
            format_debug_bytes_local(
                (gram.len().saturating_mul(std::mem::size_of::<f64>())) as u64
            ),
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/gram",
        &format!("elapsed={}", fmt_stage_secs(gram_s)),
    );
    t_stage = Instant::now();
    let (s_all, v_small) =
        rsvd_right_singular_from_gram(&gram, kp).map_err(PyRuntimeError::new_err)?;
    let eig_s = t_stage.elapsed().as_secs_f64();
    let mut eigvals = vec![0.0_f32; k_eff];
    // Align PCA eigenvalue scaling with the GRM pathway in python/janusx/script/pca.py.
    // For sample-side PCA: lambda = sigma^2 / varsum, where
    // varsum = sum_i 2p_i(1-p_i) over retained variants.
    let scale = varsum as f32;
    for i in 0..k_eff {
        eigvals[i] = (s_all[i] * s_all[i]) / scale;
    }

    // Sample-side eigenvectors/right singular vectors: q @ v_small[:, :k_eff]
    t_stage = Instant::now();
    let eigvecs_sample =
        rsvd_project_sample_eigvecs(&q, n, kp, &v_small, k_eff).map_err(PyRuntimeError::new_err)?;
    let final_vec_s = t_stage.elapsed().as_secs_f64();

    let eval_arr = PyArray1::<f32>::zeros(py, [k_eff], false).into_bound();
    let evec_arr = PyArray2::<f32>::zeros(py, [n, k_eff], false).into_bound();
    let eval_slice = unsafe {
        eval_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("eigvals output not contiguous"))?
    };
    let evec_slice = unsafe {
        evec_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("sample eigvecs output not contiguous"))?
    };
    emit_rsvd_rss_debug(
        "stream_sample/final_ready",
        &format!(
            "eigvals={} eigvecs_shape=({}, {}) eigvecs={}",
            format_f32_vec_bytes(eigvals.len()),
            n,
            k_eff,
            format_f32_vec_bytes(eigvecs_sample.len()),
        ),
    );
    let total_s = t_total.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "stream_sample/summary",
        &format!(
            "total={} backend={} init_proj={} init_q={} power_mul={} power_mul_decode={} power_mul_gemm={} power_qr={} gram={} eig={} final_vec={} power_iters={} block_rows={}",
            fmt_stage_secs(total_s),
            fmt_stage_secs(backend_s),
            fmt_stage_secs(init_proj_s),
            fmt_stage_secs(init_q_s),
            fmt_stage_secs(power_mul_s),
            fmt_stage_secs(power_mul_decode_s),
            fmt_stage_secs(power_mul_gemm_s),
            fmt_stage_secs(power_qr_s),
            fmt_stage_secs(gram_s),
            fmt_stage_secs(eig_s),
            fmt_stage_secs(final_vec_s),
            power_iters_done,
            block_rows,
        ),
    );
    eval_slice.copy_from_slice(&eigvals);
    evec_slice.copy_from_slice(&eigvecs_sample);
    Ok((eval_arr, evec_arr, total_variance))
}

fn row_freq_to_pyarray<'py>(py: Python<'py>, row_freq: &[f32]) -> Bound<'py, PyArray1<f32>> {
    #[allow(deprecated)]
    PyArray1::from_owned_array(py, Array1::from_vec(row_freq.to_vec())).into_bound()
}

fn vec_to_pyarray2_f32<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    data: &[f32],
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let out = PyArray2::<f32>::zeros(py, [rows, cols], false).into_bound();
    let out_s = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_s.copy_from_slice(data);
    Ok(out)
}

fn rsvd_stream_sample_packed_impl(
    backend: &PackedBedRsvd,
    k: usize,
    seed: u64,
    power: usize,
    tol: f32,
    backend_elapsed_s: f64,
) -> Result<(Vec<f32>, Vec<f32>, f64), String> {
    if k == 0 {
        return Err("k must be > 0".to_string());
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err("tol must be positive and finite".to_string());
    }

    let t_total = Instant::now();
    let n = backend.n_samples;
    let m = backend.n_snps;
    let row_freq = &backend.row_freq;
    let total_variance = backend.total_variance;
    let mut varsum: f64 = 0.0;
    for &freq in row_freq.iter() {
        varsum += 2.0_f64 * (freq as f64) * (1.0_f64 - freq as f64);
    }

    if m == 0 {
        return Err("no SNPs passed filtering in streaming RSVD".to_string());
    }
    if n == 0 {
        return Err("no samples available for streaming RSVD".to_string());
    }
    if !(varsum.is_finite() && varsum > 0.0) {
        return Err("invalid scaling denominator in streaming RSVD (varsum <= 0)".to_string());
    }
    check_admx_memory_limit("rsvd_stream_sample/start")?;

    let k_eff = k.min(n);
    let kp = admx_rsvd_kp(k_eff, m.max(1).min(n.max(1)));
    let block_rows = packed_backend_block_rows(backend, m);
    emit_rsvd_rss_debug(
        "stream_sample/backend_ready",
        &format!(
            "mode=packed_session n_samples={} n_snps={} kp={} row_freq={} varsum={:.6e} traceK={:.6e} total_sites={} active_row_idx={}",
            n,
            m,
            kp,
            format_f32_vec_bytes(row_freq.len()),
            varsum,
            total_variance,
            backend.n_total_sites,
            format_debug_bytes_local(
                (backend.active_row_idx.len() * std::mem::size_of::<usize>()) as u64
            ),
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/backend_ready",
        &format!("elapsed={}", fmt_stage_secs(backend_elapsed_s)),
    );
    emit_rsvd_rss_debug(
        "stream_sample/omega_ready",
        &format!(
            "omega_mode=random_block rows={} kp={} omega_scratch={}",
            block_rows,
            kp,
            format_f32_vec_bytes(block_rows * kp)
        ),
    );

    let t_stage = Instant::now();
    let mut y = compute_at_random_omega_packed(backend, kp, seed)?;
    check_admx_memory_limit("rsvd_stream_sample/init_proj")?;
    let init_proj_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/y_ready",
        &format!(
            "y_shape=({}, {}) y={}",
            n,
            kp,
            format_f32_vec_bytes(y.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/init_proj",
        &format!("elapsed={}", fmt_stage_secs(init_proj_s)),
    );

    let t_stage = Instant::now();
    let (mut q, _, _) = thin_svd_from_tall(&y, n, kp)?;
    check_admx_memory_limit("rsvd_stream_sample/init_q")?;
    let init_q_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/q_ready",
        &format!(
            "q_shape=({}, {}) q={}",
            n,
            kp,
            format_f32_vec_bytes(q.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/init_q",
        &format!("elapsed={}", fmt_stage_secs(init_q_s)),
    );

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    let mut power_mul_s = 0.0_f64;
    let mut power_mul_decode_s = 0.0_f64;
    let mut power_mul_gemm_s = 0.0_f64;
    let mut power_qr_s = 0.0_f64;
    let mut power_iters_done = 0usize;
    let measure_kernel_timing = rsvd_time_debug_enabled();
    for it in 0..power {
        check_admx_memory_limit("rsvd_stream_sample/power_start")?;
        let t_power_mul = Instant::now();
        let mut kernel_timing = RsvdKernelTiming::default();
        y = compute_ata_omega_packed(
            backend,
            &q,
            kp,
            if measure_kernel_timing {
                Some(&mut kernel_timing)
            } else {
                None
            },
        )?;
        power_mul_s += t_power_mul.elapsed().as_secs_f64();
        power_mul_decode_s += kernel_timing.decode_s;
        power_mul_gemm_s += kernel_timing.gemm_s;
        subtract_scaled_inplace_f32(&mut y, alpha, &q);
        let t_power_qr = Instant::now();
        let (q_new, s_y, _) = thin_svd_from_tall(&y, n, kp)?;
        check_admx_memory_limit("rsvd_stream_sample/power_qr")?;
        power_qr_s += t_power_qr.elapsed().as_secs_f64();
        q = q_new;
        power_iters_done = it + 1;

        if it > 0 {
            let mut max_rel = 0.0_f32;
            for i in 0..k_eff {
                let sk_now = s_y[i] + alpha;
                let denom = sk_now.max(1e-12);
                let rel = ((sk_now - sk[i]).abs()) / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                sk[i] = sk_now;
            }
            if max_rel < tol {
                break;
            }
        } else {
            for i in 0..kp {
                sk[i] = s_y[i] + alpha;
            }
        }

        let tail = s_y[kp - 1];
        if alpha < tail {
            alpha = 0.5 * (alpha + tail);
        }
    }

    let t_stage = Instant::now();
    let gram = compute_gram_aq_packed(backend, &q, kp)?;
    check_admx_memory_limit("rsvd_stream_sample/gram")?;
    let gram_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/gram_ready",
        &format!(
            "gram_shape=({}, {}) gram={}",
            kp,
            kp,
            format_debug_bytes_local(
                (gram.len().saturating_mul(std::mem::size_of::<f64>())) as u64
            ),
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/gram",
        &format!("elapsed={}", fmt_stage_secs(gram_s)),
    );

    let t_stage = Instant::now();
    let (s_all, v_small) = rsvd_right_singular_from_gram(&gram, kp)?;
    let eig_s = t_stage.elapsed().as_secs_f64();
    let mut eigvals = vec![0.0_f32; k_eff];
    let scale = varsum as f32;
    for i in 0..k_eff {
        eigvals[i] = (s_all[i] * s_all[i]) / scale;
    }

    let t_stage = Instant::now();
    let eigvecs_sample = rsvd_project_sample_eigvecs(&q, n, kp, &v_small, k_eff)?;
    let final_vec_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/final_ready",
        &format!(
            "eigvals={} eigvecs_shape=({}, {}) eigvecs={}",
            format_f32_vec_bytes(eigvals.len()),
            n,
            k_eff,
            format_f32_vec_bytes(eigvecs_sample.len()),
        ),
    );
    let total_s = backend_elapsed_s + t_total.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "stream_sample/summary",
        &format!(
            "total={} backend={} init_proj={} init_q={} power_mul={} power_mul_decode={} power_mul_gemm={} power_qr={} gram={} eig={} final_vec={} power_iters={} block_rows={}",
            fmt_stage_secs(total_s),
            fmt_stage_secs(backend_elapsed_s),
            fmt_stage_secs(init_proj_s),
            fmt_stage_secs(init_q_s),
            fmt_stage_secs(power_mul_s),
            fmt_stage_secs(power_mul_decode_s),
            fmt_stage_secs(power_mul_gemm_s),
            fmt_stage_secs(power_qr_s),
            fmt_stage_secs(gram_s),
            fmt_stage_secs(eig_s),
            fmt_stage_secs(final_vec_s),
            power_iters_done,
            block_rows,
        ),
    );
    Ok((eigvals, eigvecs_sample, total_variance))
}

#[pyfunction]
pub fn admx_set_threads(threads: usize) -> PyResult<()> {
    if threads == 0 {
        return Err(PyRuntimeError::new_err(
            "admx_set_threads expects a positive thread count",
        ));
    }
    match ThreadPoolBuilder::new().num_threads(threads).build_global() {
        Ok(()) => Ok(()),
        Err(_) => Ok(()),
    }
}
