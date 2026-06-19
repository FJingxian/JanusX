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
fn omega_stats_chunk_rows(m: usize, n_cols: usize) -> usize {
    let rows = rsvd_block_rows_env(n_cols, m)
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
    let chunk_rows = omega_stats_chunk_rows(m, n_samples);

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

pub(crate) fn rsvd_packed_compute_a_omega(
    view: PackedRsvdView<'_>,
    omega: &[f32],
    kp: usize,
) -> Result<Vec<f32>, String> {
    let (full_sample_fast, subset_plan) = view.validate()?;
    let m = view.m();
    let n = view.n();
    if omega.len() != n * kp {
        return Err("omega shape mismatch in packed RSVD A*Omega".to_string());
    }
    let total_threads = rayon::current_num_threads().max(1);
    let block_rows = rsvd_block_rows_env(n, m);
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
    let mut out = vec![0.0_f32; m * kp];
    let cur_rows_i = checked_cblas_dim(block_rows, "block_rows")?;
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
            check_admx_memory_limit("rsvd_packed/a_omega")?;
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
            let out_block = &mut out[row_start * kp..row_end * kp];
            let _ = if cur_rows == block_rows {
                cur_rows_i
            } else {
                checked_cblas_dim(cur_rows, "cur_rows")?
            };
            row_major_block_mul_mat_f32(cur_block, cur_rows, n, omega, kp, out_block, None);
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
                if let Err(e) = check_admx_memory_limit("rsvd_packed/a_omega_decode") {
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
                check_admx_memory_limit("rsvd_packed/a_omega_gemm")?;
                let row_end = buf.row_start + buf.rows;
                let out_block = &mut out[buf.row_start * kp..row_end * kp];
                let _ = if buf.rows == block_rows {
                    cur_rows_i
                } else {
                    checked_cblas_dim(buf.rows, "cur_rows")?
                };
                row_major_block_mul_mat_f32(
                    &buf.block[..buf.rows * n],
                    buf.rows,
                    n,
                    omega,
                    kp,
                    out_block,
                    None,
                );
                Ok::<(), String>(())
            },
        )?;

        let producer_err = producer_error
            .lock()
            .map_err(|_| "packed RSVD A*Omega producer error lock poisoned".to_string())?
            .take();
        if let Some(err) = producer_err {
            return Err(err);
        }
    }
    Ok(out)
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
    let block_rows = rsvd_block_rows_env(n, m);
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
    let block_rows = rsvd_block_rows_env(n, m);
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
    let block_rows = rsvd_block_rows_env(n, m);
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

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
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
