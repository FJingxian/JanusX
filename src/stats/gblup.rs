//! Streamed additive GBLUP kernels over PLINK BED blocks.
//!
//! Let `W` denote the mean-imputed, centered additive genotype matrix after
//! marker filtering and sample subsetting, with markers in rows and samples in
//! columns. The additive genomic relationship matrix is
//!
//! `K = W' W / sum_j 2 p_j (1 - p_j)`.
//!
//! This module implements the packed/streamed additive GBLUP workflow used by
//! GS:
//!
//! `y = 1 * mu + g + e,  g ~ N(0, sigma_g^2 K),  e ~ N(0, sigma_e^2 I)`.
//!
//! The exact streamed path is:
//!
//! 1. Stream selected BED rows and accumulate the train-train GRM block `K`.
//! 2. Solve the intercept-only REML problem in the eigenspace of `K`, with
//!    `lambda = sigma_e^2 / sigma_g^2`.
//! 3. Recover sample-space coefficients
//!    `alpha = V^{-1} (y - 1 * beta)`, where `V = K + lambda I`.
//! 4. Predict held-out samples from the cross-kernel `K_* alpha + beta`.
//! 5. When marker effects are requested, project the sample-space solution back
//!    to additive marker effects with
//!    `beta_kp = W alpha / sum_j 2 p_j (1 - p_j)`.
//!
//! Decoding is performed from `WindowedBedMatrix` or resident packed BED blocks,
//! and the GRM build path supports decode/compute overlap through a double
//! buffer. This file also keeps the shared packed MTM and FarmCPU
//! GRM/PCA helpers because they reuse the same additive kinship algebra and
//! streaming decode primitives.

use numpy::ndarray::{Array1, Array2};
use numpy::PyArray1;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

use crate::bedmath::{
    adaptive_grm_block_rows, decode_mean_imputed_additive_packed_block_rows_f32,
    decode_plink_bed_hardcall, decode_row_centered_full_lut, decode_row_centered_full_lut_f64,
    decode_subset_row_from_full_scratch, is_identity_indices, packed_byte_lut, SubsetDecodePlan,
};
use crate::blas::OpenBlasThreadGuard;
use crate::brent::brent_minimize;
use crate::decode::decode_standardized_additive_block_from_maf_f32;
use crate::eigh::{
    load_square_matrix_subset_row_major_f64, square_matrix_subset_cross_dot_f64,
    symmetric_eigh_f64_row_major,
};
use crate::fast_math::gblup_marker_fast_packed;
use crate::gload::WindowedBedMatrix;
use crate::grm::{self, grm_rankk_update_raw_mixed_f32_to_f64, GrmAccumMode};
use crate::packed::{
    bed_packed_row_flip_mask, cross_grm_times_alpha_packed_f64, packed_malpha_f64,
};
use crate::stats_common::{
    check_ctrlc, env_truthy, get_cached_pool, map_err_string_to_py, parse_index_vec_i64,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StreamKernelMode {
    Additive,
    StandardizedAdditive,
    Dominance,
}

#[allow(clippy::too_many_arguments)]
fn decode_subset_dom_row_from_full_scratch(
    row: &[u8],
    n_samples: usize,
    sample_idx: &[usize],
    code4_lut: &[[u8; 4]; 256],
    full_row: &mut [f32],
    out_row: &mut [f32],
) -> (f64, f64) {
    debug_assert!(full_row.len() >= n_samples);
    debug_assert_eq!(out_row.len(), sample_idx.len());

    let value_lut: [f32; 4] = [0.0_f32, -1.0_f32, 1.0_f32, 0.0_f32];
    decode_row_centered_full_lut(
        row,
        n_samples,
        code4_lut,
        &value_lut,
        &mut full_row[..n_samples],
    );

    let n = sample_idx.len();
    let mut obs_n: usize = 0;
    let mut obs_sum = 0.0_f64;
    let mut obs_sq = 0.0_f64;
    for (j, &sid) in sample_idx.iter().enumerate() {
        let gv = full_row[sid];
        out_row[j] = gv;
        if gv >= 0.0_f32 {
            let g = gv as f64;
            obs_n += 1;
            obs_sum += g;
            obs_sq += g * g;
        }
    }

    let (mean_g, var_centered) = if obs_n > 0 {
        let mean = obs_sum / obs_n as f64;
        let ss = (obs_sq - (obs_sum * obs_sum) / obs_n as f64).max(0.0);
        let v_center = (ss / n as f64).max(0.0);
        (mean as f32, v_center as f32)
    } else {
        (0.0_f32, 0.0_f32)
    };

    for v in out_row.iter_mut() {
        let gv = if *v >= 0.0_f32 { *v } else { mean_g };
        *v = gv - mean_g;
    }
    let row_sum = (mean_g as f64) * (n as f64);
    (var_centered as f64, row_sum)
}

#[allow(clippy::too_many_arguments)]
fn decode_raw_block_f64(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip_vec: &[bool],
    row_maf_vec: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    code4_lut: &[[u8; 4]; 256],
    row_start: usize,
    row_end: usize,
    n: usize,
    out_block: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    if cur_rows == 0 {
        return Ok(());
    }
    if out_block.len() < cur_rows.saturating_mul(n) {
        return Err("decode_raw_block_f64: out_block too small".to_string());
    }
    let block = &mut out_block[..cur_rows * n];

    let mut decode_run = || {
        if full_sample_fast {
            block
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(off, out_row)| {
                    let idx = row_start + off;
                    let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                    let flip = row_flip_vec[idx];
                    let mean_g = 2.0_f64 * row_maf_vec[idx] as f64;
                    let value_lut: [f64; 4] = if flip {
                        [2.0_f64, -9.0_f64, 1.0_f64, 0.0_f64]
                    } else {
                        [0.0_f64, -9.0_f64, 1.0_f64, 2.0_f64]
                    };
                    decode_row_centered_full_lut_f64(
                        row, n_samples, code4_lut, &value_lut, out_row,
                    );
                    for v in out_row.iter_mut() {
                        if *v < 0.0_f64 {
                            *v = mean_g;
                        }
                    }
                });
        } else {
            block
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(off, out_row)| {
                    let idx = row_start + off;
                    let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                    let flip = row_flip_vec[idx];
                    let mean_g = 2.0_f64 * row_maf_vec[idx] as f64;
                    for (j, &sid) in sample_idx.iter().enumerate() {
                        let b = row[sid >> 2];
                        let code = (b >> ((sid & 3) * 2)) & 0b11;
                        let mut gv = decode_plink_bed_hardcall(code).unwrap_or(mean_g);
                        if flip && code != 0b01 {
                            gv = 2.0_f64 - gv;
                        }
                        out_row[j] = gv;
                    }
                });
        }
    };

    if let Some(tp) = pool {
        tp.install(&mut decode_run);
    } else {
        decode_run();
    }
    Ok(())
}

#[inline]
fn gblup_stream_window_mb(
    block_rows: usize,
    bytes_per_snp: usize,
    mmap_window_mb: Option<usize>,
) -> usize {
    if let Some(v) = mmap_window_mb {
        return v.max(1);
    }
    let need_bytes = block_rows
        .max(1)
        .saturating_mul(bytes_per_snp.max(1))
        .max(1);
    need_bytes.div_ceil(1024 * 1024).max(1)
}

#[inline]
fn grm_scale_and_symmetrize_inplace_f64(grm: &mut [f64], n: usize, inv_scale: f64) {
    for i in 0..n {
        let ii = i * n + i;
        grm[ii] *= inv_scale;
        for j in 0..i {
            let idx_lo = i * n + j;
            let v = grm[idx_lo] * inv_scale;
            grm[idx_lo] = v;
            grm[j * n + i] = v;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_meta_block_f32(
    packed_flat: &[u8],
    packed_row_indices: &[usize],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    subset_plan: Option<&SubsetDecodePlan>,
    mode: StreamKernelMode,
    out_block: &mut [f32],
    out_varsum: &mut [f64],
    out_rowsum: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let rows = row_flip.len();
    if rows == 0 {
        return Ok(());
    }
    if row_maf.len() != rows || packed_row_indices.len() != rows {
        return Err("decode_meta_block_f32: row metadata length mismatch".to_string());
    }
    if sample_idx.is_empty() {
        return Err("decode_meta_block_f32: sample_idx must not be empty".to_string());
    }
    let n_out = sample_idx.len();
    if out_block.len() < rows.saturating_mul(n_out) {
        return Err("decode_meta_block_f32: out_block too small".to_string());
    }
    if out_varsum.len() < rows {
        return Err("decode_meta_block_f32: out_varsum too small".to_string());
    }
    if out_rowsum.len() < rows {
        return Err("decode_meta_block_f32: out_rowsum too small".to_string());
    }

    let block = &mut out_block[..rows * n_out];
    let varsum = &mut out_varsum[..rows];
    let rowsum = &mut out_rowsum[..rows];

    if mode == StreamKernelMode::StandardizedAdditive {
        let mut scratch_row_indices = vec![0usize; rows];
        let mut scratch_mean = vec![0.0_f32; rows];
        let mut scratch_scale = vec![0.0_f32; rows];
        decode_standardized_additive_block_from_maf_f32(
            packed_flat,
            bytes_per_snp,
            n_samples,
            row_flip,
            row_maf,
            sample_idx,
            full_sample_fast,
            subset_plan,
            Some(packed_row_indices),
            0usize,
            rows,
            1e-12_f32,
            scratch_row_indices.as_mut_slice(),
            scratch_mean.as_mut_slice(),
            scratch_scale.as_mut_slice(),
            block,
            pool,
        )?;
        for local_row in 0..rows {
            let p = row_maf[local_row].clamp(0.0_f32, 1.0_f32) as f64;
            varsum[local_row] = 0.0_f64;
            rowsum[local_row] = 2.0_f64 * p * (n_out as f64);
        }
        return Ok(());
    }

    let code4_lut = &packed_byte_lut().code4;

    let mut decode_run = || {
        if full_sample_fast {
            block
                .par_chunks_mut(n_out)
                .zip(varsum.par_iter_mut())
                .zip(rowsum.par_iter_mut())
                .enumerate()
                .for_each(|(local_row, ((dst, row_varsum_dst), row_sum_dst))| {
                    let rel_row = packed_row_indices[local_row];
                    let row = &packed_flat[rel_row * bytes_per_snp..(rel_row + 1) * bytes_per_snp];
                    let p = row_maf[local_row].clamp(0.0_f32, 1.0_f32) as f64;
                    let (mean_g, var, value_lut): (f64, f64, [f32; 4]) = match mode {
                        StreamKernelMode::Additive => {
                            let mean_g = 2.0_f64 * p;
                            let var = 2.0_f64 * p * (1.0_f64 - p);
                            let value_lut = if row_flip[local_row] {
                                [2.0_f32, mean_g as f32, 1.0_f32, 0.0_f32]
                            } else {
                                [0.0_f32, mean_g as f32, 1.0_f32, 2.0_f32]
                            };
                            (mean_g, var, value_lut)
                        }
                        StreamKernelMode::Dominance => {
                            let mean_g = p;
                            let var = p * (1.0_f64 - p);
                            let value_lut = [0.0_f32, mean_g as f32, 1.0_f32, 0.0_f32];
                            (mean_g, var, value_lut)
                        }
                        StreamKernelMode::StandardizedAdditive => unreachable!(),
                    };
                    decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, dst);
                    let mean_g_f32 = mean_g as f32;
                    for v in dst.iter_mut() {
                        *v -= mean_g_f32;
                    }
                    *row_varsum_dst = var;
                    *row_sum_dst = mean_g * (n_out as f64);
                });
        } else {
            block
                .par_chunks_mut(n_out)
                .zip(varsum.par_iter_mut())
                .zip(rowsum.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || vec![0.0_f32; n_samples],
                    |full_row, (local_row, ((dst, row_varsum_dst), row_sum_dst))| {
                        let rel_row = packed_row_indices[local_row];
                        let row =
                            &packed_flat[rel_row * bytes_per_snp..(rel_row + 1) * bytes_per_snp];
                        let (var_centered, row_sum) = match mode {
                            StreamKernelMode::Additive => {
                                let default_mean_g =
                                    2.0_f32 * row_maf[local_row].clamp(0.0_f32, 1.0_f32);
                                decode_subset_row_from_full_scratch(
                                    row,
                                    n_samples,
                                    sample_idx,
                                    row_flip[local_row],
                                    default_mean_g,
                                    1,
                                    1e-12_f32,
                                    code4_lut,
                                    full_row.as_mut_slice(),
                                    dst,
                                )
                            }
                            StreamKernelMode::Dominance => decode_subset_dom_row_from_full_scratch(
                                row,
                                n_samples,
                                sample_idx,
                                code4_lut,
                                full_row.as_mut_slice(),
                                dst,
                            ),
                            StreamKernelMode::StandardizedAdditive => unreachable!(),
                        };
                        *row_varsum_dst = var_centered;
                        *row_sum_dst = row_sum;
                    },
                );
        }
    };

    if let Some(tp) = pool {
        tp.install(&mut decode_run);
    } else {
        decode_run();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_grm_from_meta_stream<P>(
    prefix: &str,
    row_source_indices: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    train_idx: &[usize],
    mode: StreamKernelMode,
    block_rows: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_every: usize,
    mut progress: Option<P>,
) -> Result<(Vec<f64>, Vec<f64>, f64), String>
where
    P: FnMut(usize, usize) -> Result<(), String>,
{
    let eff_m = row_source_indices.len();
    let n_train = train_idx.len();
    if eff_m == 0 || n_train == 0 {
        return Err("build_grm_from_meta_stream: empty marker or sample set".to_string());
    }
    if row_flip.len() != eff_m || row_maf.len() != eff_m {
        return Err("build_grm_from_meta_stream: row metadata length mismatch".to_string());
    }
    let n_samples = crate::gfcore::read_fam(prefix)?.len();
    if n_samples == 0 {
        return Err("build_grm_from_meta_stream: no samples found in BED".to_string());
    }
    let bytes_per_snp = n_samples.div_ceil(4);
    let window_mb = gblup_stream_window_mb(block_rows, bytes_per_snp, mmap_window_mb);
    let mut matrix = WindowedBedMatrix::open(prefix, window_mb)?;
    let pool_owned = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let pool_ref = pool_owned.as_ref();
    let full_sample_fast = is_identity_indices(train_idx, n_samples);
    let subset_plan = if full_sample_fast {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
            train_idx, n_samples,
        ))
    };
    let row_step = adaptive_grm_block_rows(
        block_rows.max(1),
        eff_m.max(1),
        n_train,
        if full_sample_fast { 0 } else { n_samples },
        threads,
    )
    .max(1);
    let notify_step = if progress_every == 0 {
        row_step
    } else {
        progress_every.max(1)
    };

    let _blas_guard = OpenBlasThreadGuard::enter(threads.max(1));
    let mut grm = vec![0.0_f64; n_train * n_train];
    let grm_ptr = grm.as_mut_ptr();
    let mut grm_tmp = vec![0.0_f32; n_train * n_train];
    let mut varsum_acc = 0.0_f64;
    let mut row_sum_all = vec![0.0_f64; eff_m];
    let mut last_notified = 0usize;
    let overlap_enabled = std::env::var("JX_GBLUP_GRM_OVERLAP")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            !matches!(t.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(true);
    let can_pipeline = overlap_enabled && threads > 1 && eff_m > row_step;

    if !can_pipeline {
        let mut block = vec![0.0_f32; row_step * n_train];
        let mut block_varsum = vec![0.0_f64; row_step];
        let mut block_rowsum = vec![0.0_f64; row_step];
        let mut rel_indices = Vec::<usize>::with_capacity(row_step);
        let mut is_first_block = true;

        for row_start in (0..eff_m).step_by(row_step) {
            check_ctrlc()?;
            let row_end = (row_start + row_step).min(eff_m);
            let cur_rows = row_end - row_start;
            let packed_slice = matrix
                .prepare_source_rows(&row_source_indices[row_start..row_end], &mut rel_indices)?;
            decode_meta_block_f32(
                packed_slice,
                rel_indices.as_slice(),
                bytes_per_snp,
                n_samples,
                &row_flip[row_start..row_end],
                &row_maf[row_start..row_end],
                train_idx,
                full_sample_fast,
                subset_plan.as_ref(),
                mode,
                &mut block[..cur_rows * n_train],
                &mut block_varsum[..cur_rows],
                &mut block_rowsum[..cur_rows],
                pool_ref,
            )?;
            grm_rankk_update_raw_mixed_f32_to_f64(
                grm_ptr,
                &block[..cur_rows * n_train],
                cur_rows,
                n_train,
                GrmAccumMode::Syrk,
                false,
                is_first_block,
                false,
                0,
                &mut grm_tmp,
            )?;
            is_first_block = false;
            varsum_acc += block_varsum[..cur_rows].iter().sum::<f64>();
            row_sum_all[row_start..row_end].copy_from_slice(&block_rowsum[..cur_rows]);
            let done = row_end;
            if (done >= last_notified.saturating_add(notify_step)) || (done == eff_m) {
                last_notified = done;
                if let Some(cb) = progress.as_mut() {
                    cb(done, eff_m)?;
                }
            }
        }
    } else {
        struct Chunk {
            data: Vec<f32>,
            varsum: Vec<f64>,
            rowsum: Vec<f64>,
            start: usize,
            rows: usize,
        }

        let producer_err = Arc::new(std::sync::Mutex::new(None::<String>));
        let producer_err_cloned = Arc::clone(&producer_err);
        let decode_pool_owned = pool_owned.clone();
        let mut next_start = 0usize;
        let mut rel_indices = Vec::<usize>::with_capacity(row_step);

        let make_chunk = || Chunk {
            data: vec![0.0_f32; row_step * n_train],
            varsum: vec![0.0_f64; row_step],
            rowsum: vec![0.0_f64; row_step],
            start: 0usize,
            rows: 0usize,
        };
        let producer = |buf: &mut Chunk| -> bool {
            if next_start >= eff_m {
                return false;
            }
            let row_start = next_start;
            let row_end = (row_start + row_step).min(eff_m);
            let cur_rows = row_end - row_start;
            buf.start = row_start;
            buf.rows = cur_rows;
            let decode_res = (|| -> Result<(), String> {
                let packed_slice = matrix.prepare_source_rows(
                    &row_source_indices[row_start..row_end],
                    &mut rel_indices,
                )?;
                decode_meta_block_f32(
                    packed_slice,
                    rel_indices.as_slice(),
                    bytes_per_snp,
                    n_samples,
                    &row_flip[row_start..row_end],
                    &row_maf[row_start..row_end],
                    train_idx,
                    full_sample_fast,
                    subset_plan.as_ref(),
                    mode,
                    &mut buf.data[..cur_rows * n_train],
                    &mut buf.varsum[..cur_rows],
                    &mut buf.rowsum[..cur_rows],
                    decode_pool_owned.as_ref(),
                )
            })();
            match decode_res {
                Ok(()) => {
                    next_start = row_end;
                    next_start < eff_m
                }
                Err(err) => {
                    if let Ok(mut slot) = producer_err_cloned.lock() {
                        *slot = Some(err);
                    }
                    buf.rows = 0usize;
                    next_start = eff_m;
                    false
                }
            }
        };

        let mut is_first_block = true;
        let consumer = |buf: &mut Chunk| -> Result<(), String> {
            check_ctrlc()?;
            if let Ok(mut slot) = producer_err.lock() {
                if let Some(err) = slot.take() {
                    return Err(err);
                }
            }
            if buf.rows == 0 {
                return Ok(());
            }
            let cur_rows = buf.rows;
            grm_rankk_update_raw_mixed_f32_to_f64(
                grm_ptr,
                &buf.data[..cur_rows * n_train],
                cur_rows,
                n_train,
                GrmAccumMode::Syrk,
                false,
                is_first_block,
                false,
                0,
                &mut grm_tmp,
            )?;
            is_first_block = false;
            varsum_acc += buf.varsum[..cur_rows].iter().sum::<f64>();
            let row_end = buf.start + cur_rows;
            row_sum_all[buf.start..row_end].copy_from_slice(&buf.rowsum[..cur_rows]);
            let done = row_end;
            if (done >= last_notified.saturating_add(notify_step)) || (done == eff_m) {
                last_notified = done;
                if let Some(cb) = progress.as_mut() {
                    cb(done, eff_m)?;
                }
            }
            Ok(())
        };

        crate::pipeline::run_double_buffer(2usize, make_chunk, producer, consumer)?;
        let producer_err_tail = producer_err.lock().ok().and_then(|mut slot| slot.take());
        if let Some(err) = producer_err_tail {
            return Err(err);
        }
    }

    let scale = if mode == StreamKernelMode::StandardizedAdditive {
        eff_m as f64
    } else {
        if !(varsum_acc.is_finite() && varsum_acc > 0.0) {
            return Err("build_grm_from_meta_stream: invalid centered GRM denominator".to_string());
        }
        varsum_acc
    };
    grm_scale_and_symmetrize_inplace_f64(&mut grm, n_train, 1.0_f64 / scale);
    Ok((grm, row_sum_all, scale))
}

fn write_npy_f32_matrix_from_f64(
    path: &str,
    data: &[f64],
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    let expected = rows.saturating_mul(cols);
    if data.len() != expected {
        return Err(format!(
            "write_npy_f32_matrix_from_f64: data length mismatch, got {}, expected {}",
            data.len(),
            expected
        ));
    }

    let f = File::create(path).map_err(|e| format!("create npy file failed: {e}"))?;
    let mut w = BufWriter::new(f);
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    while (10 + header.len() + 1) % 16 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let header_len_u16 = u16::try_from(header.len())
        .map_err(|_| "npy header too long for v1.0 format".to_string())?;

    w.write_all(b"\x93NUMPY")
        .map_err(|e| format!("write npy magic failed: {e}"))?;
    w.write_all(&[1_u8, 0_u8])
        .map_err(|e| format!("write npy version failed: {e}"))?;
    w.write_all(&header_len_u16.to_le_bytes())
        .map_err(|e| format!("write npy header length failed: {e}"))?;
    w.write_all(header.as_bytes())
        .map_err(|e| format!("write npy header failed: {e}"))?;

    #[cfg(target_endian = "little")]
    {
        let mut chunk = vec![0.0_f32; 8192];
        for src in data.chunks(chunk.len()) {
            let used = src.len();
            for (dst, &v) in chunk[..used].iter_mut().zip(src.iter()) {
                *dst = v as f32;
            }
            let byte_len = used.saturating_mul(std::mem::size_of::<f32>());
            let bytes =
                unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const u8, byte_len) };
            w.write_all(bytes)
                .map_err(|e| format!("write npy payload failed: {e}"))?;
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &v in data.iter() {
            w.write_all(&(v as f32).to_le_bytes())
                .map_err(|e| format!("write npy payload failed: {e}"))?;
        }
    }

    w.flush()
        .map_err(|e| format!("flush npy file failed: {e}"))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    out_npy_path,
    row_source_indices,
    row_flip,
    row_maf,
    sample_indices=None,
    method=1,
    block_rows=65536,
    threads=0,
    progress_callback=None,
    progress_every=0,
    mmap_window_mb=None
))]
pub fn gblup_grm_from_meta_to_npy<'py>(
    py: Python<'py>,
    prefix: String,
    out_npy_path: String,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_rows: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<(usize, usize)> {
    if method != 1 && method != 2 && method != 3 {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method={method}; expected 1 (centered additive), 2 (standardized additive), or 3 (centered dominance)"
        )));
    }

    let mut bed_prefix = prefix.trim().to_string();
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len().saturating_sub(4));
    }
    let n_samples_full = crate::gfcore::read_fam(&bed_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples_full == 0 {
        return Err(PyRuntimeError::new_err("no samples found in PLINK input"));
    }

    let sample_idx: Vec<usize> = if let Some(sample_indices) = sample_indices {
        parse_index_vec_i64(sample_indices.as_slice()?, n_samples_full, "sample_indices")?
    } else {
        (0..n_samples_full).collect()
    };
    if sample_idx.is_empty() {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }

    let row_idx64 = row_source_indices
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("row_source_indices must be contiguous int64"))?
        .to_vec();
    let row_flip_vec = match row_flip.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip.as_array().iter().copied().collect(),
    };
    let row_maf_vec = match row_maf.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_maf.as_array().iter().copied().collect(),
    };
    if row_idx64.is_empty() {
        return Err(PyRuntimeError::new_err(
            "row_source_indices must not be empty",
        ));
    }
    if row_flip_vec.len() != row_idx64.len() || row_maf_vec.len() != row_idx64.len() {
        return Err(PyRuntimeError::new_err(format!(
            "row meta length mismatch: row_source_indices={}, row_flip={}, row_maf={}",
            row_idx64.len(),
            row_flip_vec.len(),
            row_maf_vec.len(),
        )));
    }

    let out_npy_path_owned = out_npy_path.clone();
    let n_out = sample_idx.len();
    let eff_m = row_idx64.len();
    let (eff_m_out, n_samples_out) = py
        .detach(move || -> Result<(usize, usize), String> {
            let row_idx: Vec<usize> = row_idx64
                .iter()
                .map(|&sid| {
                    if sid < 0 {
                        Err(format!("row index must be non-negative, got {sid}"))
                    } else {
                        Ok(sid as usize)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            let progress = move |done: usize, total: usize| -> Result<(), String> {
                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (done, total))?;
                        Ok(())
                    })
                    .map_err(|e| e.to_string())?;
                } else {
                    Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                }
                Ok(())
            };

            let stream_mode = match method {
                1 => StreamKernelMode::Additive,
                2 => StreamKernelMode::StandardizedAdditive,
                3 => StreamKernelMode::Dominance,
                _ => unreachable!(),
            };
            let (grm, _row_sum, _varsum) = build_grm_from_meta_stream(
                &bed_prefix,
                row_idx.as_slice(),
                row_flip_vec.as_slice(),
                row_maf_vec.as_slice(),
                sample_idx.as_slice(),
                stream_mode,
                block_rows,
                threads,
                mmap_window_mb,
                progress_every,
                Some(progress),
            )?;
            write_npy_f32_matrix_from_f64(&out_npy_path_owned, grm.as_slice(), n_out, n_out)?;
            Ok((eff_m, n_out))
        })
        .map_err(map_err_string_to_py)?;

    Ok((eff_m_out, n_samples_out))
}

#[allow(clippy::too_many_arguments)]
fn compute_malpha_from_meta_stream(
    prefix: &str,
    row_source_indices: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    alpha: &[f64],
    block_rows: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
) -> Result<Vec<f64>, String> {
    let eff_m = row_source_indices.len();
    if row_flip.len() != eff_m || row_maf.len() != eff_m {
        return Err("compute_malpha_from_meta_stream: row metadata length mismatch".to_string());
    }
    if sample_idx.len() != alpha.len() {
        return Err("compute_malpha_from_meta_stream: alpha length mismatch".to_string());
    }
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Err("compute_malpha_from_meta_stream: sample_idx must not be empty".to_string());
    }
    let n_samples = crate::gfcore::read_fam(prefix)?.len();
    let bytes_per_snp = n_samples.div_ceil(4);
    let window_mb = gblup_stream_window_mb(block_rows, bytes_per_snp, mmap_window_mb);
    let mut matrix = WindowedBedMatrix::open(prefix, window_mb)?;
    let pool_owned = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let pool_ref = pool_owned.as_ref();
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let row_step = block_rows.max(1).min(eff_m.max(1));
    let code4_lut = &packed_byte_lut().code4;
    let mut out = vec![0.0_f64; eff_m];
    let mut block = vec![0.0_f32; row_step * n_out];
    let mut rel_indices = Vec::<usize>::with_capacity(row_step);

    for row_start in (0..eff_m).step_by(row_step) {
        check_ctrlc()?;
        let row_end = (row_start + row_step).min(eff_m);
        let cur_rows = row_end - row_start;
        let packed_slice = matrix
            .prepare_source_rows(&row_source_indices[row_start..row_end], &mut rel_indices)?;
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed_slice,
            bytes_per_snp,
            n_samples,
            &row_flip[row_start..row_end],
            &row_maf[row_start..row_end],
            sample_idx,
            full_sample_fast,
            Some(rel_indices.as_slice()),
            0,
            &mut block[..cur_rows * n_out],
            code4_lut,
            pool_ref,
        )?;
        out[row_start..row_end]
            .par_iter_mut()
            .enumerate()
            .for_each(|(local_row, dst)| {
                let row = &block[local_row * n_out..(local_row + 1) * n_out];
                let mut acc = 0.0_f64;
                for j in 0..n_out {
                    acc += (row[j] as f64) * alpha[j];
                }
                *dst = acc;
            });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn compute_effect_beta_from_meta_stream(
    prefix: &str,
    row_source_indices: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    alpha: &[f64],
    mode: StreamKernelMode,
    block_rows: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
) -> Result<Vec<f64>, String> {
    let eff_m = row_source_indices.len();
    if row_flip.len() != eff_m || row_maf.len() != eff_m {
        return Err(
            "compute_effect_beta_from_meta_stream: row metadata length mismatch".to_string(),
        );
    }
    if sample_idx.len() != alpha.len() {
        return Err("compute_effect_beta_from_meta_stream: alpha length mismatch".to_string());
    }
    let n_out = sample_idx.len();
    if eff_m == 0 {
        return Err("compute_effect_beta_from_meta_stream: empty marker set".to_string());
    }
    if n_out == 0 {
        return Err(
            "compute_effect_beta_from_meta_stream: sample_idx must not be empty".to_string(),
        );
    }
    let n_samples = crate::gfcore::read_fam(prefix)?.len();
    let bytes_per_snp = n_samples.div_ceil(4);
    let window_mb = gblup_stream_window_mb(block_rows, bytes_per_snp, mmap_window_mb);
    let mut matrix = WindowedBedMatrix::open(prefix, window_mb)?;
    let pool_owned = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let pool_ref = pool_owned.as_ref();
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let subset_plan = if full_sample_fast {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
            sample_idx, n_samples,
        ))
    };
    let row_step = adaptive_grm_block_rows(
        block_rows.max(1),
        eff_m.max(1),
        n_out,
        if full_sample_fast { 0 } else { n_samples },
        threads,
    )
    .max(1);

    let mut out = vec![0.0_f64; eff_m];
    let mut varsum_acc = 0.0_f64;
    let mut block = vec![0.0_f32; row_step * n_out];
    let mut block_varsum = vec![0.0_f64; row_step];
    let mut block_rowsum = vec![0.0_f64; row_step];
    let mut rel_indices = Vec::<usize>::with_capacity(row_step);

    for row_start in (0..eff_m).step_by(row_step) {
        check_ctrlc()?;
        let row_end = (row_start + row_step).min(eff_m);
        let cur_rows = row_end - row_start;
        let packed_slice = matrix
            .prepare_source_rows(&row_source_indices[row_start..row_end], &mut rel_indices)?;
        decode_meta_block_f32(
            packed_slice,
            rel_indices.as_slice(),
            bytes_per_snp,
            n_samples,
            &row_flip[row_start..row_end],
            &row_maf[row_start..row_end],
            sample_idx,
            full_sample_fast,
            subset_plan.as_ref(),
            mode,
            &mut block[..cur_rows * n_out],
            &mut block_varsum[..cur_rows],
            &mut block_rowsum[..cur_rows],
            pool_ref,
        )?;
        out[row_start..row_end]
            .par_iter_mut()
            .enumerate()
            .for_each(|(off, dst)| {
                let row = &block[off * n_out..(off + 1) * n_out];
                let mut acc = 0.0_f64;
                for (g, a) in row.iter().zip(alpha.iter()) {
                    acc += (*g as f64) * *a;
                }
                *dst = acc;
            });
        varsum_acc += block_varsum[..cur_rows].iter().sum::<f64>();
    }

    if !(varsum_acc.is_finite() && varsum_acc > 0.0) {
        return Err("compute_effect_beta_from_meta_stream: invalid denominator".to_string());
    }
    let inv_var = 1.0_f64 / varsum_acc;
    for v in out.iter_mut() {
        *v *= inv_var;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn predict_from_effect_stream(
    prefix: &str,
    row_source_indices: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    effect_alpha0: f64,
    effect_beta: &[f64],
    block_rows: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
) -> Result<Vec<f64>, String> {
    if sample_idx.is_empty() {
        return Ok(Vec::new());
    }
    let eff_m = row_source_indices.len();
    if row_flip.len() != eff_m || row_maf.len() != eff_m || effect_beta.len() != eff_m {
        return Err("predict_from_effect_stream: row metadata length mismatch".to_string());
    }
    let n_samples = crate::gfcore::read_fam(prefix)?.len();
    let bytes_per_snp = n_samples.div_ceil(4);
    let window_mb = gblup_stream_window_mb(block_rows, bytes_per_snp, mmap_window_mb);
    let mut matrix = WindowedBedMatrix::open(prefix, window_mb)?;
    let pool_owned = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let pool_ref = pool_owned.as_ref();
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let row_step = block_rows.max(1).min(eff_m.max(1));
    let n_out = sample_idx.len();
    let code4_lut = &packed_byte_lut().code4;
    let mut out = vec![effect_alpha0; n_out];
    let mut block = vec![0.0_f32; row_step * n_out];
    let mut rel_indices = Vec::<usize>::with_capacity(row_step);

    for row_start in (0..eff_m).step_by(row_step) {
        check_ctrlc()?;
        let row_end = (row_start + row_step).min(eff_m);
        let cur_rows = row_end - row_start;
        let packed_slice = matrix
            .prepare_source_rows(&row_source_indices[row_start..row_end], &mut rel_indices)?;
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed_slice,
            bytes_per_snp,
            n_samples,
            &row_flip[row_start..row_end],
            &row_maf[row_start..row_end],
            sample_idx,
            full_sample_fast,
            Some(rel_indices.as_slice()),
            0,
            &mut block[..cur_rows * n_out],
            code4_lut,
            pool_ref,
        )?;
        let beta_block = &effect_beta[row_start..row_end];
        out.par_iter_mut()
            .enumerate()
            .for_each(|(sample_pos, dst)| {
                let mut acc = 0.0_f64;
                for local_row in 0..cur_rows {
                    acc += (block[local_row * n_out + sample_pos] as f64) * beta_block[local_row];
                }
                *dst += acc;
            });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn fit_gblup_reml_from_grm_row_major_f64(
    grm_f64: &[f64],
    y_vec: &[f64],
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
    threads: usize,
) -> Result<(Vec<f64>, f64, f64, f64, f64, f64, String, f64, f64, f64), String> {
    let n = y_vec.len();
    if n <= 1 {
        return Err("GBLUP REML requires at least 2 training samples.".to_string());
    }
    if grm_f64.len() != n.saturating_mul(n) {
        return Err(format!(
            "GBLUP GRM shape mismatch: got {} values, expected {}",
            grm_f64.len(),
            n.saturating_mul(n)
        ));
    }

    let _blas_guard = OpenBlasThreadGuard::enter(threads.max(1));
    let y_mean = y_vec.iter().sum::<f64>() / (n as f64);
    let y_center: Vec<f64> = y_vec.iter().map(|v| *v - y_mean).collect();

    let t_evd = Instant::now();
    let (s, u, evd_backend) = symmetric_eigh_f64_row_major(grm_f64, n)?;
    let evd_elapsed = t_evd.elapsed().as_secs_f64();

    let mut x_rot = vec![0.0_f64; n];
    let mut y_rot = vec![0.0_f64; n];
    for k in 0..n {
        let mut xk = 0.0_f64;
        let mut yk = 0.0_f64;
        for r in 0..n {
            let urk = u[r * n + k];
            xk += urk;
            yk += urk * y_center[r];
        }
        x_rot[k] = xk;
        y_rot[k] = yk;
    }

    let v_floor = 1e-12_f64;
    let n_eff = (n - 1) as f64;
    let c_reml = n_eff * (n_eff.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let c_ml = (n as f64) * ((n as f64).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    let eval_reml = |log10_lbd: f64| -> Option<(f64, f64, f64, f64, Vec<f64>, Vec<f64>)> {
        let lbd = 10.0_f64.powf(log10_lbd);
        if !(lbd.is_finite() && lbd > 0.0) {
            return None;
        }
        let mut v_inv = vec![0.0_f64; n];
        let mut log_det_v = 0.0_f64;
        let mut xtvx = 0.0_f64;
        let mut xtvy = 0.0_f64;
        for i in 0..n {
            let vi = (s[i] + lbd).max(v_floor);
            log_det_v += vi.ln();
            let invi = 1.0_f64 / vi;
            v_inv[i] = invi;
            xtvx += invi * x_rot[i] * x_rot[i];
            xtvy += invi * x_rot[i] * y_rot[i];
        }
        if !(xtvx.is_finite() && xtvx > v_floor) {
            return None;
        }
        let beta = xtvy / xtvx;
        let mut r = vec![0.0_f64; n];
        let mut rtv_invr = 0.0_f64;
        for i in 0..n {
            let ri = y_rot[i] - x_rot[i] * beta;
            r[i] = ri;
            rtv_invr += v_inv[i] * ri * ri;
        }
        if !(rtv_invr.is_finite() && rtv_invr > v_floor) {
            return None;
        }
        let log_det_xtv = xtvx.ln();
        let reml = c_reml - 0.5 * (n_eff * rtv_invr.ln() + log_det_v + log_det_xtv);
        let ml = c_ml - 0.5 * (((n as f64) * rtv_invr.ln()) + log_det_v);
        if !(reml.is_finite() && ml.is_finite()) {
            return None;
        }
        Some((reml, ml, beta, rtv_invr, v_inv, r))
    };

    let (best_log10, _best_cost) = brent_minimize(
        |x0| match eval_reml(x0) {
            Some((reml_v, _, _, _, _, _)) => -reml_v,
            None => 1e100,
        },
        low,
        high,
        tol,
        max_iter,
    );
    let (reml_best, ml_best, beta_rot, rtv_invr, v_inv, r) = eval_reml(best_log10)
        .ok_or_else(|| "GBLUP REML optimization failed to produce a valid optimum.".to_string())?;
    let lambda = 10.0_f64.powf(best_log10);

    let mut alpha = vec![0.0_f64; n];
    for row in 0..n {
        let mut acc = 0.0_f64;
        for k in 0..n {
            acc += u[row * n + k] * (v_inv[k] * r[k]);
        }
        alpha[row] = acc;
    }

    let sigma_g2 = rtv_invr / n_eff.max(1.0);
    let sigma_e2 = lambda * sigma_g2;
    let mean_s = s.iter().copied().sum::<f64>() / (n as f64);
    let var_g = sigma_g2 * mean_s.max(0.0);
    let denom = var_g + sigma_e2;
    let pve = if denom.is_finite() && denom > 0.0 {
        var_g / denom
    } else {
        f64::NAN
    };
    let beta0 = y_mean + beta_rot;

    Ok((
        alpha,
        beta0,
        lambda,
        pve,
        ml_best,
        reml_best,
        evd_backend.to_string(),
        evd_elapsed,
        sigma_g2,
        sigma_e2,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    grm_path,
    train_sample_indices,
    y_train,
    test_sample_indices=None,
    train_pred_local_indices=None,
    g_eps=1e-8_f64,
    low=-6.0_f64,
    high=6.0_f64,
    max_iter=50,
    tol=1e-4_f64,
    threads=0,
    return_variance_components=false,
    estimate_only=false
))]
pub fn gblup_reml_npy_grm<'py>(
    py: Python<'py>,
    grm_path: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    y_train: PyReadonlyArray1<'py, f64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    train_pred_local_indices: Option<PyReadonlyArray1<'py, i64>>,
    g_eps: f64,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    return_variance_components: bool,
    estimate_only: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    f64,
    f64,
    f64,
    String,
    f64,
    usize,
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
)> {
    if !(g_eps.is_finite() && g_eps >= 0.0) {
        return Err(PyRuntimeError::new_err("g_eps must be finite and >= 0"));
    }
    if !(low.is_finite() && high.is_finite() && low < high) {
        return Err(PyRuntimeError::new_err(
            "low/high must be finite and low < high",
        ));
    }
    if max_iter == 0 {
        return Err(PyRuntimeError::new_err("max_iter must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be finite and > 0"));
    }

    let train_idx_raw = train_sample_indices.as_slice()?;
    if train_idx_raw.is_empty() {
        return Err(PyRuntimeError::new_err(
            "train_sample_indices must not be empty.",
        ));
    }
    let mut train_idx = Vec::<usize>::with_capacity(train_idx_raw.len());
    for (pos, &idx) in train_idx_raw.iter().enumerate() {
        if idx < 0 {
            return Err(PyRuntimeError::new_err(format!(
                "train_sample_indices index at position {} is negative ({})",
                pos, idx
            )));
        }
        train_idx.push(idx as usize);
    }

    let y_vec: Vec<f64> = match y_train.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y_train.as_array().iter().copied().collect(),
    };
    if y_vec.len() != train_idx.len() {
        return Err(PyRuntimeError::new_err(format!(
            "y_train length mismatch: got {}, expected {}",
            y_vec.len(),
            train_idx.len()
        )));
    }
    if y_vec.iter().any(|v| !v.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "y_train contains non-finite values.",
        ));
    }

    let mut test_idx = Vec::<usize>::new();
    if let Some(sidx) = test_sample_indices {
        for (pos, &idx) in sidx.as_slice()?.iter().enumerate() {
            if idx < 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "test_sample_indices index at position {} is negative ({})",
                    pos, idx
                )));
            }
            test_idx.push(idx as usize);
        }
    }
    let train_pred_pick: Option<Vec<usize>> = if let Some(local_idx) = train_pred_local_indices {
        Some(parse_index_vec_i64(
            local_idx.as_slice()?,
            train_idx.len(),
            "train_pred_local_indices",
        )?)
    } else {
        None
    };

    let grm_path_owned = grm_path.clone();
    let train_idx_owned = train_idx;
    let test_idx_owned = test_idx;
    let y_vec_owned = y_vec;
    let train_pred_pick_owned = train_pred_pick;
    let threads_owned = threads.max(1);

    let (
        pred_train,
        pred_test,
        pve,
        lambda_opt,
        ml,
        reml,
        evd_backend,
        evd_elapsed,
        sigma_g2,
        sigma_e2,
    ) = py
        .detach(move || -> Result<
            (
                Vec<f64>,
                Vec<f64>,
                f64,
                f64,
                f64,
                f64,
                String,
                f64,
                f64,
                f64,
            ),
            String,
        > {
            let (grm_train, n_train) = load_square_matrix_subset_row_major_f64(
                &grm_path_owned,
                train_idx_owned.as_slice(),
                g_eps,
            )?;
            if n_train != train_idx_owned.len() {
                return Err(format!(
                    "train subset size mismatch: got {}, expected {}",
                    n_train,
                    train_idx_owned.len()
                ));
            }

            let (
                alpha,
                beta0,
                lambda,
                pve_fit,
                ml_fit,
                reml_fit,
                evd_backend_fit,
                evd_elapsed_fit,
                sigma_g2_fit,
                sigma_e2_fit,
            ) = fit_gblup_reml_from_grm_row_major_f64(
                grm_train.as_slice(),
                y_vec_owned.as_slice(),
                low,
                high,
                tol,
                max_iter,
                threads_owned,
            )?;

            if estimate_only {
                return Ok((
                    Vec::new(),
                    Vec::new(),
                    pve_fit,
                    lambda,
                    ml_fit,
                    reml_fit,
                    evd_backend_fit,
                    evd_elapsed_fit,
                    sigma_g2_fit,
                    sigma_e2_fit,
                ));
            }

            let train_pred_abs: Vec<usize> = if let Some(local_idx) = &train_pred_pick_owned {
                local_idx.iter().map(|&i| train_idx_owned[i]).collect()
            } else {
                train_idx_owned.clone()
            };
            let mut pred_train = square_matrix_subset_cross_dot_f64(
                &grm_path_owned,
                train_pred_abs.as_slice(),
                train_idx_owned.as_slice(),
                alpha.as_slice(),
            )?;
            pred_train.iter_mut().for_each(|v| *v += beta0);

            let mut pred_test = square_matrix_subset_cross_dot_f64(
                &grm_path_owned,
                test_idx_owned.as_slice(),
                train_idx_owned.as_slice(),
                alpha.as_slice(),
            )?;
            pred_test.iter_mut().for_each(|v| *v += beta0);

            Ok((
                pred_train,
                pred_test,
                pve_fit,
                lambda,
                ml_fit,
                reml_fit,
                evd_backend_fit,
                evd_elapsed_fit,
                sigma_g2_fit,
                sigma_e2_fit,
            ))
        })
        .map_err(map_err_string_to_py)?;

    let sigma_g2_out = if return_variance_components {
        sigma_g2
    } else {
        f64::NAN
    };
    let sigma_e2_out = if return_variance_components {
        sigma_e2
    } else {
        f64::NAN
    };
    let train_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((pred_train.len(), 1), pred_train)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let test_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((pred_test.len(), 1), pred_test)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let effect_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(Vec::<f64>::new())).into_bound();
    Ok((
        train_arr,
        test_arr,
        pve,
        lambda_opt,
        ml,
        reml,
        evd_backend,
        evd_elapsed,
        0usize,
        sigma_g2_out,
        sigma_e2_out,
        effect_arr,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    train_sample_indices,
    y_train,
    test_sample_indices=None,
    train_pred_local_indices=None,
    site_keep=None,
    g_eps=1e-8_f64,
    low=-6.0_f64,
    high=6.0_f64,
    max_iter=50,
    tol=1e-4_f64,
    block_rows=4096,
    threads=0,
    return_variance_components=false,
    estimate_only=false,
    return_effect=false,
    row_source_indices=None,
    row_flip=None,
    row_maf=None,
    mmap_window_mb=None
))]
pub fn gblup_reml_packed_bed<'py>(
    py: Python<'py>,
    prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    y_train: PyReadonlyArray1<'py, f64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    train_pred_local_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    g_eps: f64,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    block_rows: usize,
    threads: usize,
    return_variance_components: bool,
    estimate_only: bool,
    return_effect: bool,
    row_source_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_maf: Option<PyReadonlyArray1<'py, f32>>,
    mmap_window_mb: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    f64,
    f64,
    f64,
    String,
    f64,
    usize,
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
)> {
    if !(g_eps.is_finite() && g_eps >= 0.0) {
        return Err(PyRuntimeError::new_err("g_eps must be finite and >= 0"));
    }
    if !(low.is_finite() && high.is_finite() && low < high) {
        return Err(PyRuntimeError::new_err(
            "low/high must be finite and low < high",
        ));
    }
    if max_iter == 0 {
        return Err(PyRuntimeError::new_err("max_iter must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be finite and > 0"));
    }

    if let (Some(row_source_indices), Some(row_flip), Some(row_maf)) =
        (row_source_indices, row_flip, row_maf)
    {
        let row_source_vec_i64: Vec<i64> = match row_source_indices.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_source_indices.as_array().iter().copied().collect(),
        };
        if row_source_vec_i64.is_empty() {
            return Err(PyRuntimeError::new_err(
                "row_source_indices must not be empty for metadata streaming path.",
            ));
        }
        let mut row_source_vec = Vec::<usize>::with_capacity(row_source_vec_i64.len());
        for &idx in row_source_vec_i64.iter() {
            if idx < 0 {
                return Err(PyRuntimeError::new_err(
                    "row_source_indices must be non-negative.",
                ));
            }
            row_source_vec.push(idx as usize);
        }
        let row_flip_vec: Vec<bool> = match row_flip.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_flip.as_array().iter().copied().collect(),
        };
        let row_maf_vec: Vec<f32> = match row_maf.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_maf.as_array().iter().copied().collect(),
        };
        if row_flip_vec.len() != row_source_vec.len() || row_maf_vec.len() != row_source_vec.len() {
            return Err(PyRuntimeError::new_err(format!(
                "metadata length mismatch: row_source_indices={}, row_flip={}, row_maf={}",
                row_source_vec.len(),
                row_flip_vec.len(),
                row_maf_vec.len(),
            )));
        }

        let n_samples = crate::gfcore::read_fam(&prefix)
            .map_err(map_err_string_to_py)?
            .len();
        if n_samples == 0 {
            return Err(PyRuntimeError::new_err("No samples found in BED input."));
        }
        let train_idx = parse_index_vec_i64(
            train_sample_indices.as_slice()?,
            n_samples,
            "train_sample_indices",
        )?;
        if train_idx.is_empty() {
            return Err(PyRuntimeError::new_err(
                "train_sample_indices must not be empty.",
            ));
        }
        let y_vec: Vec<f64> = match y_train.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => y_train.as_array().iter().copied().collect(),
        };
        if y_vec.len() != train_idx.len() {
            return Err(PyRuntimeError::new_err(format!(
                "y_train length mismatch: got {}, expected {}",
                y_vec.len(),
                train_idx.len()
            )));
        }
        if y_vec.iter().any(|v| !v.is_finite()) {
            return Err(PyRuntimeError::new_err(
                "y_train contains non-finite values.",
            ));
        }
        let test_idx = if let Some(sidx) = test_sample_indices {
            parse_index_vec_i64(sidx.as_slice()?, n_samples, "test_sample_indices")?
        } else {
            Vec::new()
        };
        let train_pred_pick: Option<Vec<usize>> = if let Some(local_idx) = train_pred_local_indices
        {
            Some(parse_index_vec_i64(
                local_idx.as_slice()?,
                train_idx.len(),
                "train_pred_local_indices",
            )?)
        } else {
            None
        };

        let prefix_owned = prefix.clone();
        let row_source_owned = row_source_vec;
        let row_flip_owned = row_flip_vec;
        let row_maf_owned = row_maf_vec;
        let train_idx_owned = train_idx;
        let y_vec_owned = y_vec;
        let test_idx_owned = test_idx;
        let train_pred_pick_owned = train_pred_pick;
        let block_rows_owned = block_rows.max(1);
        let threads_owned = threads.max(1);
        let return_effect_owned = return_effect;
        let estimate_only_owned = estimate_only;
        let mmap_window_mb_owned = mmap_window_mb;

        let (
            pred_train,
            pred_test,
            pve,
            lambda_opt,
            ml,
            reml,
            evd_backend,
            evd_elapsed,
            eff_m,
            sigma_g2,
            sigma_e2,
            effect_beta,
        ) = py
            .detach(move || -> Result<
                (
                    Vec<f64>,
                    Vec<f64>,
                    f64,
                    f64,
                    f64,
                    f64,
                    String,
                    f64,
                    usize,
                    f64,
                    f64,
                    Vec<f64>,
                ),
                String,
            > {
                let n_train = train_idx_owned.len();
                let eff_m = row_source_owned.len();
                let train_pred_abs: Vec<usize> = if let Some(local_idx) = &train_pred_pick_owned {
                    local_idx.iter().map(|&i| train_idx_owned[i]).collect()
                } else {
                    train_idx_owned.clone()
                };

                let (mut grm_f64, row_sum_vec, var_sum) = build_grm_from_meta_stream(
                    &prefix_owned,
                    row_source_owned.as_slice(),
                    row_flip_owned.as_slice(),
                    row_maf_owned.as_slice(),
                    train_idx_owned.as_slice(),
                    StreamKernelMode::Additive,
                    block_rows_owned,
                    threads_owned,
                    mmap_window_mb_owned,
                    0usize,
                    None::<fn(usize, usize) -> Result<(), String>>,
                )?;
                for i in 0..n_train {
                    grm_f64[i * n_train + i] += g_eps;
                }
                let row_mean: Vec<f64> = row_sum_vec
                    .iter()
                    .map(|v| *v / (n_train as f64))
                    .collect();

                let _blas_guard = OpenBlasThreadGuard::enter(threads_owned.max(1));
                let n = n_train;
                if n <= 1 {
                    return Err("GBLUP REML requires at least 2 training samples.".to_string());
                }
                let y_mean = y_vec_owned.iter().sum::<f64>() / (n as f64);
                let y_center: Vec<f64> = y_vec_owned.iter().map(|v| *v - y_mean).collect();

                let t_evd = Instant::now();
                let (s, u, evd_backend) = symmetric_eigh_f64_row_major(&grm_f64, n)?;
                let evd_elapsed = t_evd.elapsed().as_secs_f64();

                let mut x_rot = vec![0.0_f64; n];
                let mut y_rot = vec![0.0_f64; n];
                for k in 0..n {
                    let mut xk = 0.0_f64;
                    let mut yk = 0.0_f64;
                    for r in 0..n {
                        let urk = u[r * n + k];
                        xk += urk;
                        yk += urk * y_center[r];
                    }
                    x_rot[k] = xk;
                    y_rot[k] = yk;
                }

                let v_floor = 1e-12_f64;
                let n_eff = (n - 1) as f64;
                let c_reml = n_eff * (n_eff.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
                let c_ml = (n as f64) * ((n as f64).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

                let eval_reml =
                    |log10_lbd: f64| -> Option<(f64, f64, f64, f64, Vec<f64>, Vec<f64>)> {
                        let lbd = 10.0_f64.powf(log10_lbd);
                        if !(lbd.is_finite() && lbd > 0.0) {
                            return None;
                        }
                        let mut v_inv = vec![0.0_f64; n];
                        let mut log_det_v = 0.0_f64;
                        let mut xtvx = 0.0_f64;
                        let mut xtvy = 0.0_f64;
                        for i in 0..n {
                            let vi = (s[i] + lbd).max(v_floor);
                            log_det_v += vi.ln();
                            let invi = 1.0_f64 / vi;
                            v_inv[i] = invi;
                            xtvx += invi * x_rot[i] * x_rot[i];
                            xtvy += invi * x_rot[i] * y_rot[i];
                        }
                        if !(xtvx.is_finite() && xtvx > v_floor) {
                            return None;
                        }
                        let beta = xtvy / xtvx;
                        let mut r = vec![0.0_f64; n];
                        let mut rtv_invr = 0.0_f64;
                        for i in 0..n {
                            let ri = y_rot[i] - x_rot[i] * beta;
                            r[i] = ri;
                            rtv_invr += v_inv[i] * ri * ri;
                        }
                        if !(rtv_invr.is_finite() && rtv_invr > v_floor) {
                            return None;
                        }
                        let log_det_xtv = xtvx.ln();
                        let reml =
                            c_reml - 0.5 * (n_eff * rtv_invr.ln() + log_det_v + log_det_xtv);
                        let ml = c_ml - 0.5 * (((n as f64) * rtv_invr.ln()) + log_det_v);
                        if !(reml.is_finite() && ml.is_finite()) {
                            return None;
                        }
                        Some((reml, ml, beta, rtv_invr, v_inv, r))
                    };

                let (best_log10, _best_cost) = brent_minimize(
                    |x0| match eval_reml(x0) {
                        Some((reml_v, _, _, _, _, _)) => -reml_v,
                        None => 1e100,
                    },
                    low,
                    high,
                    tol,
                    max_iter,
                );
                let (reml_best, ml_best, beta_rot, rtv_invr, v_inv, r) = eval_reml(best_log10)
                    .ok_or_else(|| {
                        "GBLUP REML optimization failed to produce a valid optimum.".to_string()
                    })?;
                let lambda = 10.0_f64.powf(best_log10);

                let mut alpha = vec![0.0_f64; n];
                for row in 0..n {
                    let mut acc = 0.0_f64;
                    for k in 0..n {
                        acc += u[row * n + k] * (v_inv[k] * r[k]);
                    }
                    alpha[row] = acc;
                }

                let sigma_g2 = rtv_invr / n_eff.max(1.0);
                let sigma_e2 = lambda * sigma_g2;
                let mean_s = s.iter().copied().sum::<f64>() / (n as f64);
                let var_g = sigma_g2 * mean_s.max(0.0);
                let denom = var_g + sigma_e2;
                let pve = if denom.is_finite() && denom > 0.0 {
                    var_g / denom
                } else {
                    f64::NAN
                };
                let beta0 = y_mean + beta_rot;

                let need_projection = (!estimate_only_owned) || return_effect_owned;
                let (pred_train_local, pred_test_local, effect_beta_vec) = if need_projection {
                    let m_alpha = compute_malpha_from_meta_stream(
                        &prefix_owned,
                        row_source_owned.as_slice(),
                        row_flip_owned.as_slice(),
                        row_maf_owned.as_slice(),
                        train_idx_owned.as_slice(),
                        alpha.as_slice(),
                        block_rows_owned,
                        threads_owned,
                        mmap_window_mb_owned,
                    )?;
                    let alpha_sum = alpha.iter().copied().sum::<f64>();
                    let mean_sq = row_mean.iter().map(|v| *v * *v).sum::<f64>();
                    let mean_malpha = row_mean
                        .iter()
                        .zip(m_alpha.iter())
                        .map(|(a, b)| (*a) * (*b))
                        .sum::<f64>();
                    let inv_var = 1.0_f64 / var_sum.max(1e-12_f64);
                    let effect_beta_vec: Vec<f64> = m_alpha
                        .iter()
                        .zip(row_mean.iter())
                        .map(|(ma, mm)| (*ma - (*mm) * alpha_sum) * inv_var)
                        .collect();
                    let const_term = mean_sq * alpha_sum - mean_malpha;
                    let effect_alpha0 = beta0 + const_term * inv_var;
                    if estimate_only_owned {
                        (Vec::new(), Vec::new(), effect_beta_vec)
                    } else {
                        let pred_train = predict_from_effect_stream(
                            &prefix_owned,
                            row_source_owned.as_slice(),
                            row_flip_owned.as_slice(),
                            row_maf_owned.as_slice(),
                            train_pred_abs.as_slice(),
                            effect_alpha0,
                            effect_beta_vec.as_slice(),
                            block_rows_owned,
                            threads_owned,
                            mmap_window_mb_owned,
                        )?;
                        let pred_test = predict_from_effect_stream(
                            &prefix_owned,
                            row_source_owned.as_slice(),
                            row_flip_owned.as_slice(),
                            row_maf_owned.as_slice(),
                            test_idx_owned.as_slice(),
                            effect_alpha0,
                            effect_beta_vec.as_slice(),
                            block_rows_owned,
                            threads_owned,
                            mmap_window_mb_owned,
                        )?;
                        (pred_train, pred_test, effect_beta_vec)
                    }
                } else {
                    (Vec::new(), Vec::new(), Vec::new())
                };

                Ok((
                    pred_train_local,
                    pred_test_local,
                    pve,
                    lambda,
                    ml_best,
                    reml_best,
                    evd_backend.to_string(),
                    evd_elapsed,
                    eff_m,
                    sigma_g2,
                    sigma_e2,
                    if return_effect_owned {
                        effect_beta_vec
                    } else {
                        Vec::new()
                    },
                ))
            })
            .map_err(map_err_string_to_py)?;

        let sigma_g2_out = if return_variance_components {
            sigma_g2
        } else {
            f64::NAN
        };
        let sigma_e2_out = if return_variance_components {
            sigma_e2
        } else {
            f64::NAN
        };
        let train_arr = PyArray2::from_owned_array(
            py,
            Array2::from_shape_vec((pred_train.len(), 1), pred_train)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
        .into_bound();
        let test_arr = PyArray2::from_owned_array(
            py,
            Array2::from_shape_vec((pred_test.len(), 1), pred_test)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
        .into_bound();
        let effect_arr = if return_effect {
            PyArray1::from_owned_array(py, Array1::from_vec(effect_beta)).into_bound()
        } else {
            PyArray1::from_owned_array(py, Array1::from_vec(Vec::<f64>::new())).into_bound()
        };
        return Ok((
            train_arr,
            test_arr,
            pve,
            lambda_opt,
            ml,
            reml,
            evd_backend,
            evd_elapsed,
            eff_m,
            sigma_g2_out,
            sigma_e2_out,
            effect_arr,
        ));
    }

    let site_keep_vec: Option<Vec<bool>> = if let Some(mask) = site_keep {
        Some(match mask.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mask.as_array().iter().copied().collect(),
        })
    } else {
        None
    };
    let subset_requested = site_keep_vec
        .as_ref()
        .map(|mask| mask.iter().any(|&keep| !keep))
        .unwrap_or(false);

    let mut loaded_packed_arr: Option<Bound<'py, PyArray2<u8>>> = None;
    let mut loaded_maf_arr: Option<Bound<'py, PyArray1<f32>>> = None;
    let mut loaded_row_flip_arr: Option<Bound<'py, PyArray1<bool>>> = None;
    let mut subset_packed_keep: Option<Vec<u8>> = None;
    let mut subset_maf_keep: Option<Vec<f32>> = None;
    let mut subset_row_flip_keep: Option<Vec<bool>> = None;

    let n_samples: usize;
    let bytes_per_snp: usize;
    let eff_m: usize;
    let reuse_loaded_arrays_for_grm: bool;

    if subset_requested {
        let subset = py
            .detach({
                let prefix = prefix.clone();
                let mask_vec = site_keep_vec
                    .clone()
                    .expect("subset path requires site_keep vector");
                move || crate::gfreader::load_bed_2bit_packed_subset_owned(&prefix, &mask_vec)
            })
            .map_err(map_err_string_to_py)?;
        if subset.n_samples == 0 {
            return Err(PyRuntimeError::new_err("No samples found in BED input."));
        }
        n_samples = subset.n_samples;
        bytes_per_snp = subset.bytes_per_snp;
        eff_m = subset.maf.len();
        subset_packed_keep = Some(subset.packed);
        subset_maf_keep = Some(subset.maf);
        subset_row_flip_keep = Some(subset.row_flip);
        reuse_loaded_arrays_for_grm = false;
    } else {
        let (packed_arr, _miss_arr, maf_arr, _std_arr, n_samples_loaded) =
            crate::gfreader::load_bed_2bit_packed(py, prefix.clone())?;
        if n_samples_loaded == 0 {
            return Err(PyRuntimeError::new_err("No samples found in BED input."));
        }
        n_samples = n_samples_loaded;
        loaded_packed_arr = Some(packed_arr);
        loaded_maf_arr = Some(maf_arr);

        let packed_ro = loaded_packed_arr
            .as_ref()
            .expect("packed array must exist")
            .readonly();
        let packed_view = packed_ro.as_array();
        if packed_view.ndim() != 2 {
            return Err(PyRuntimeError::new_err(
                "packed BED payload must be 2D (m, bytes_per_snp).",
            ));
        }
        let m_total = packed_view.shape()[0];
        if m_total == 0 {
            return Err(PyRuntimeError::new_err("No SNP rows found in BED input."));
        }
        bytes_per_snp = packed_view.shape()[1];
        let expected_bps = (n_samples + 3) / 4;
        if bytes_per_snp != expected_bps {
            return Err(PyRuntimeError::new_err(format!(
                "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
            )));
        }
        if let Some(mask_vec) = site_keep_vec.as_ref() {
            if mask_vec.len() != m_total {
                return Err(PyRuntimeError::new_err(format!(
                    "site_keep length mismatch: got {}, expected {m_total}",
                    mask_vec.len()
                )));
            }
        }

        let maf_len = loaded_maf_arr
            .as_ref()
            .expect("maf array must exist")
            .readonly()
            .as_array()
            .len();
        if maf_len != m_total {
            return Err(PyRuntimeError::new_err(format!(
                "maf length mismatch: got {}, expected {m_total}",
                maf_len
            )));
        }

        let row_flip_full_arr = bed_packed_row_flip_mask(
            py,
            loaded_packed_arr
                .as_ref()
                .expect("packed array must exist")
                .readonly(),
            n_samples,
        )?;
        loaded_row_flip_arr = Some(row_flip_full_arr);
        let row_flip_len = loaded_row_flip_arr
            .as_ref()
            .expect("row_flip array must exist")
            .readonly()
            .as_array()
            .len();
        if row_flip_len != m_total {
            return Err(PyRuntimeError::new_err(format!(
                "row_flip length mismatch: got {}, expected {m_total}",
                row_flip_len
            )));
        }

        eff_m = m_total;
        reuse_loaded_arrays_for_grm = true;
    }

    let train_idx = parse_index_vec_i64(
        train_sample_indices.as_slice()?,
        n_samples,
        "train_sample_indices",
    )?;
    if train_idx.is_empty() {
        return Err(PyRuntimeError::new_err(
            "train_sample_indices must not be empty.",
        ));
    }
    let y_vec: Vec<f64> = match y_train.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y_train.as_array().iter().copied().collect(),
    };
    if y_vec.len() != train_idx.len() {
        return Err(PyRuntimeError::new_err(format!(
            "y_train length mismatch: got {}, expected {}",
            y_vec.len(),
            train_idx.len()
        )));
    }
    if y_vec.iter().any(|v| !v.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "y_train contains non-finite values.",
        ));
    }

    let test_idx = if let Some(sidx) = test_sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "test_sample_indices")?
    } else {
        Vec::new()
    };
    let train_pred_pick: Option<Vec<usize>> = if let Some(local_idx) = train_pred_local_indices {
        Some(parse_index_vec_i64(
            local_idx.as_slice()?,
            train_idx.len(),
            "train_pred_local_indices",
        )?)
    } else {
        None
    };

    let n_train = train_idx.len();
    let force_grm_eig = env_truthy("JX_GBLUP_PACKED_FORCE_GRM_EIG");
    let use_marker_fast = (n_train > eff_m) && (!force_grm_eig);

    if use_marker_fast {
        let pool_owned = get_cached_pool(threads)?;
        let pool_ref = pool_owned.as_ref();
        let packed_keep: Cow<[u8]> = if reuse_loaded_arrays_for_grm {
            let packed_ro = loaded_packed_arr
                .as_ref()
                .expect("packed array must exist for marker-fast path")
                .readonly();
            let packed_view = packed_ro.as_array();
            Cow::Owned(match packed_ro.as_slice() {
                Ok(s) => s.to_vec(),
                Err(_) => packed_view.iter().copied().collect(),
            })
        } else {
            Cow::Borrowed(
                subset_packed_keep
                    .as_ref()
                    .expect("subset packed payload must exist")
                    .as_slice(),
            )
        };
        let maf_keep: Cow<[f32]> = if reuse_loaded_arrays_for_grm {
            let maf_ro = loaded_maf_arr
                .as_ref()
                .expect("maf array must exist for marker-fast path")
                .readonly();
            Cow::Owned(match maf_ro.as_slice() {
                Ok(s) => s.to_vec(),
                Err(_) => maf_ro.as_array().iter().copied().collect(),
            })
        } else {
            Cow::Borrowed(
                subset_maf_keep
                    .as_ref()
                    .expect("subset maf payload must exist")
                    .as_slice(),
            )
        };
        let row_flip_keep: Cow<[bool]> = if reuse_loaded_arrays_for_grm {
            let row_flip_ro = loaded_row_flip_arr
                .as_ref()
                .expect("row_flip array must exist for marker-fast path")
                .readonly();
            Cow::Owned(match row_flip_ro.as_slice() {
                Ok(s) => s.to_vec(),
                Err(_) => row_flip_ro.as_array().iter().copied().collect(),
            })
        } else {
            Cow::Borrowed(
                subset_row_flip_keep
                    .as_ref()
                    .expect("subset row_flip payload must exist")
                    .as_slice(),
            )
        };
        let train_pred_abs: Vec<usize> = if let Some(local_idx) = &train_pred_pick {
            local_idx.iter().map(|&i| train_idx[i]).collect()
        } else {
            train_idx.clone()
        };
        let sample_chunk = std::env::var("JX_GBLUP_FAST_SAMPLE_CHUNK")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(1024)
            .max(1);

        let (
            pred_train,
            pred_test,
            pve,
            lambda_opt,
            ml,
            reml,
            evd_backend,
            evd_elapsed,
            sigma_g2,
            sigma_e2,
            effect_alpha0,
            effect_beta,
        ) = py
            .detach(
                move || -> Result<
                    (
                        Vec<f64>,
                        Vec<f64>,
                        f64,
                        f64,
                        f64,
                        f64,
                        String,
                        f64,
                        f64,
                        f64,
                        f64,
                        Vec<f64>,
                    ),
                    String,
                > {
                    gblup_marker_fast_packed(
                        threads,
                        eff_m,
                        n_train,
                        &train_idx,
                        &y_vec,
                        &test_idx,
                        &train_pred_abs,
                        sample_chunk,
                        packed_keep.as_ref(),
                        bytes_per_snp,
                        row_flip_keep.as_ref(),
                        maf_keep.as_ref(),
                        g_eps,
                        low,
                        high,
                        tol,
                        max_iter,
                        estimate_only,
                        return_effect,
                        pool_ref,
                    )
                },
            )
            .map_err(map_err_string_to_py)?;
        let sigma_g2_out = if return_variance_components {
            sigma_g2
        } else {
            f64::NAN
        };
        let sigma_e2_out = if return_variance_components {
            sigma_e2
        } else {
            f64::NAN
        };

        let train_arr = PyArray2::from_owned_array(
            py,
            Array2::from_shape_vec((pred_train.len(), 1), pred_train)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
        .into_bound();
        let test_arr = PyArray2::from_owned_array(
            py,
            Array2::from_shape_vec((pred_test.len(), 1), pred_test)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
        .into_bound();
        let effect_arr = if return_effect {
            PyArray1::from_owned_array(py, Array1::from_vec(effect_beta)).into_bound()
        } else {
            PyArray1::from_owned_array(py, Array1::from_vec(Vec::<f64>::new())).into_bound()
        };
        let _effect_alpha0_out = if return_effect {
            effect_alpha0
        } else {
            f64::NAN
        };

        return Ok((
            train_arr,
            test_arr,
            pve,
            lambda_opt,
            ml,
            reml,
            evd_backend,
            evd_elapsed,
            eff_m,
            sigma_g2_out,
            sigma_e2_out,
            effect_arr,
        ));
    }

    let train_idx_i64: Vec<i64> = train_idx.iter().map(|&v| v as i64).collect();
    let train_idx_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(train_idx_i64)).into_bound();

    let mut owned_packed_keep_arr: Option<Bound<'py, PyArray2<u8>>> = None;
    let mut owned_row_flip_keep_arr: Option<Bound<'py, PyArray1<bool>>> = None;
    let mut owned_maf_keep_arr: Option<Bound<'py, PyArray1<f32>>> = None;
    if !reuse_loaded_arrays_for_grm {
        owned_packed_keep_arr = Some(
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec(
                    (eff_m, bytes_per_snp),
                    subset_packed_keep
                        .take()
                        .expect("subset packed payload must exist for GRM path"),
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
            .into_bound(),
        );
        owned_row_flip_keep_arr = Some(
            PyArray1::from_owned_array(
                py,
                Array1::from_vec(
                    subset_row_flip_keep
                        .take()
                        .expect("subset row_flip payload must exist for GRM path"),
                ),
            )
            .into_bound(),
        );
        owned_maf_keep_arr = Some(
            PyArray1::from_owned_array(
                py,
                Array1::from_vec(
                    subset_maf_keep
                        .take()
                        .expect("subset maf payload must exist for GRM path"),
                ),
            )
            .into_bound(),
        );
    }

    let packed_keep_arr_ref = if reuse_loaded_arrays_for_grm {
        loaded_packed_arr
            .as_ref()
            .expect("packed array must exist for direct GRM reuse")
    } else {
        owned_packed_keep_arr
            .as_ref()
            .expect("subset packed array must exist")
    };
    let row_flip_keep_arr_ref = if reuse_loaded_arrays_for_grm {
        loaded_row_flip_arr
            .as_ref()
            .expect("row_flip array must exist for direct GRM reuse")
    } else {
        owned_row_flip_keep_arr
            .as_ref()
            .expect("subset row_flip array must exist")
    };
    let maf_keep_arr_ref = if reuse_loaded_arrays_for_grm {
        loaded_maf_arr
            .as_ref()
            .expect("maf array must exist for direct GRM reuse")
    } else {
        owned_maf_keep_arr
            .as_ref()
            .expect("subset maf array must exist")
    };

    let (grm_arr, row_sum_arr, var_sum_raw) = if reuse_loaded_arrays_for_grm {
        crate::grm::grm_packed_f64_with_stats(
            py,
            packed_keep_arr_ref.readonly(),
            n_samples,
            row_flip_keep_arr_ref.readonly(),
            maf_keep_arr_ref.readonly(),
            Some(train_idx_arr.readonly()),
            1,
            block_rows.max(1),
            threads,
            None,
            0,
        )?
    } else {
        crate::grm::grm_packed_f64_with_stats(
            py,
            packed_keep_arr_ref.readonly(),
            n_samples,
            row_flip_keep_arr_ref.readonly(),
            maf_keep_arr_ref.readonly(),
            Some(train_idx_arr.readonly()),
            1,
            block_rows.max(1),
            threads,
            None,
            0,
        )?
    };
    let n_train = train_idx.len();
    let grm_ro = grm_arr.readonly();
    let grm_slice = grm_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("GRM buffer is not contiguous: {e}")))?;
    if grm_slice.len() != n_train.saturating_mul(n_train) {
        return Err(PyRuntimeError::new_err(format!(
            "GRM shape mismatch: got {} values, expected {}",
            grm_slice.len(),
            n_train.saturating_mul(n_train)
        )));
    }
    let mut grm_f64 = grm_slice.to_vec();
    if g_eps > 0.0 {
        for i in 0..n_train {
            grm_f64[i * n_train + i] += g_eps;
        }
    }

    let row_sum_vec: Vec<f64> = row_sum_arr
        .readonly()
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("row_sum is not contiguous: {e}")))?
        .to_vec();
    if row_sum_vec.len() != eff_m {
        return Err(PyRuntimeError::new_err(format!(
            "row_sum length mismatch: got {}, expected {}",
            row_sum_vec.len(),
            eff_m
        )));
    }
    let mut var_sum = if var_sum_raw.is_finite() {
        var_sum_raw
    } else {
        f64::NAN
    };
    if !(var_sum.is_finite() && var_sum > 0.0) {
        var_sum = 1e-12_f64;
    }
    let m_mean: Vec<f64> = row_sum_vec.iter().map(|v| *v / (n_train as f64)).collect();

    let (
        alpha_vec,
        beta0,
        lambda_opt,
        pve,
        ml,
        reml,
        evd_backend,
        evd_elapsed,
        sigma_g2,
        sigma_e2,
    ) = py
        .detach(
            move || -> Result<
                (
                    Vec<f64>,
                    f64,
                    f64,
                    f64,
                    f64,
                    f64,
                    String,
                    f64,
                    f64,
                    f64,
                ),
                String,
            > {
                // Keep eig stage pinned to requested OpenBLAS threads.
                let _blas_guard = OpenBlasThreadGuard::enter(threads.max(1));
                let n = n_train;
                if n <= 1 {
                    return Err("GBLUP REML requires at least 2 training samples.".to_string());
                }
                let y_mean = y_vec.iter().sum::<f64>() / (n as f64);
                let y_center: Vec<f64> = y_vec.iter().map(|v| *v - y_mean).collect();

                let t_evd = Instant::now();
                let (s, u, evd_backend) = symmetric_eigh_f64_row_major(&grm_f64, n)?;
                let evd_elapsed = t_evd.elapsed().as_secs_f64();

                let mut x_rot = vec![0.0_f64; n];
                let mut y_rot = vec![0.0_f64; n];
                for k in 0..n {
                    let mut xk = 0.0_f64;
                    let mut yk = 0.0_f64;
                    for r in 0..n {
                        let urk = u[r * n + k];
                        xk += urk; // intercept column is all ones in centered space
                        yk += urk * y_center[r];
                    }
                    x_rot[k] = xk;
                    y_rot[k] = yk;
                }

                let v_floor = 1e-12_f64;
                let n_eff = (n - 1) as f64;
                let c_reml = n_eff * (n_eff.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
                let c_ml = (n as f64) * ((n as f64).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

                let eval_reml =
                    |log10_lbd: f64| -> Option<(f64, f64, f64, f64, Vec<f64>, Vec<f64>)> {
                        let lbd = 10.0_f64.powf(log10_lbd);
                        if !(lbd.is_finite() && lbd > 0.0) {
                            return None;
                        }
                        let mut v_inv = vec![0.0_f64; n];
                        let mut log_det_v = 0.0_f64;
                        let mut xtvx = 0.0_f64;
                        let mut xtvy = 0.0_f64;
                        for i in 0..n {
                            let vi = (s[i] + lbd).max(v_floor);
                            log_det_v += vi.ln();
                            let invi = 1.0_f64 / vi;
                            v_inv[i] = invi;
                            xtvx += invi * x_rot[i] * x_rot[i];
                            xtvy += invi * x_rot[i] * y_rot[i];
                        }
                        if !(xtvx.is_finite() && xtvx > v_floor) {
                            return None;
                        }
                        let beta = xtvy / xtvx;
                        let mut r = vec![0.0_f64; n];
                        let mut rtv_invr = 0.0_f64;
                        for i in 0..n {
                            let ri = y_rot[i] - x_rot[i] * beta;
                            r[i] = ri;
                            rtv_invr += v_inv[i] * ri * ri;
                        }
                        if !(rtv_invr.is_finite() && rtv_invr > v_floor) {
                            return None;
                        }
                        let log_det_xtv = xtvx.ln();
                        let reml = c_reml - 0.5 * (n_eff * rtv_invr.ln() + log_det_v + log_det_xtv);
                        let ml = c_ml - 0.5 * (((n as f64) * rtv_invr.ln()) + log_det_v);
                        if !(reml.is_finite() && ml.is_finite()) {
                            return None;
                        }
                        Some((reml, ml, beta, rtv_invr, v_inv, r))
                    };

                let (best_log10, _best_cost) = brent_minimize(
                    |x0| match eval_reml(x0) {
                        Some((reml_v, _, _, _, _, _)) => -reml_v,
                        None => 1e100,
                    },
                    low,
                    high,
                    tol,
                    max_iter,
                );
                let (reml_best, ml_best, beta_rot, rtv_invr, v_inv, r) = eval_reml(best_log10)
                    .ok_or_else(|| {
                        "GBLUP REML optimization failed to produce a valid optimum.".to_string()
                    })?;
                let lambda = 10.0_f64.powf(best_log10);

                let mut alpha = vec![0.0_f64; n];
                for row in 0..n {
                    let mut acc = 0.0_f64;
                    for k in 0..n {
                        acc += u[row * n + k] * (v_inv[k] * r[k]);
                    }
                    alpha[row] = acc;
                }

                let sigma_g2 = rtv_invr / n_eff.max(1.0);
                let sigma_e2 = lambda * sigma_g2;
                let mean_s = s.iter().copied().sum::<f64>() / (n as f64);
                let var_g = sigma_g2 * mean_s.max(0.0);
                let denom = var_g + sigma_e2;
                let pve = if denom.is_finite() && denom > 0.0 {
                    var_g / denom
                } else {
                    f64::NAN
                };
                let beta0 = y_mean + beta_rot;
                Ok((
                    alpha,
                    beta0,
                    lambda,
                    pve,
                    ml_best,
                    reml_best,
                    evd_backend.to_string(),
                    evd_elapsed,
                    sigma_g2,
                    sigma_e2,
                ))
            },
        )
        .map_err(map_err_string_to_py)?;
    let sigma_g2_out = if return_variance_components {
        sigma_g2
    } else {
        f64::NAN
    };
    let sigma_e2_out = if return_variance_components {
        sigma_e2
    } else {
        f64::NAN
    };

    let alpha_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(alpha_vec.clone())).into_bound();
    let m_mean_arr = PyArray1::from_owned_array(py, Array1::from_vec(m_mean.clone())).into_bound();

    let m_alpha_arr = packed_malpha_f64(
        py,
        packed_keep_arr_ref.readonly(),
        n_samples,
        row_flip_keep_arr_ref.readonly(),
        maf_keep_arr_ref.readonly(),
        train_idx_arr.readonly(),
        alpha_arr.readonly(),
        block_rows.max(1),
        threads,
    )?;
    let m_alpha_ro = m_alpha_arr.readonly();
    let m_alpha_slice = m_alpha_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("m_alpha is not contiguous: {e}")))?;
    if m_alpha_slice.len() != eff_m {
        return Err(PyRuntimeError::new_err(format!(
            "m_alpha length mismatch: got {}, expected {}",
            m_alpha_slice.len(),
            eff_m
        )));
    }
    let alpha_sum = alpha_vec.iter().copied().sum::<f64>();
    let mean_sq = m_mean.iter().map(|v| *v * *v).sum::<f64>();
    let mean_malpha = m_mean
        .iter()
        .zip(m_alpha_slice.iter())
        .map(|(a, b)| (*a) * (*b))
        .sum::<f64>();
    let mut effect_alpha0 = f64::NAN;
    let mut effect_beta_vec: Vec<f64> = Vec::new();
    if return_effect {
        let inv_var = 1.0_f64 / var_sum.max(1e-12_f64);
        effect_beta_vec = m_alpha_slice
            .iter()
            .zip(m_mean.iter())
            .map(|(ma, mm)| (*ma - (*mm) * alpha_sum) * inv_var)
            .collect();
        let const_term = mean_sq * alpha_sum - mean_malpha;
        effect_alpha0 = beta0 + const_term * inv_var;
    }

    let (pred_train, pred_test): (Vec<f64>, Vec<f64>) = if estimate_only {
        (Vec::new(), Vec::new())
    } else {
        let train_pred_all_arr = cross_grm_times_alpha_packed_f64(
            py,
            packed_keep_arr_ref.readonly(),
            n_samples,
            row_flip_keep_arr_ref.readonly(),
            maf_keep_arr_ref.readonly(),
            train_idx_arr.readonly(),
            m_alpha_arr.readonly(),
            m_mean_arr.readonly(),
            alpha_sum,
            mean_sq,
            mean_malpha,
            var_sum,
            block_rows.max(1),
            threads,
        )?;
        let train_pred_all_ro = train_pred_all_arr.readonly();
        let train_pred_all_slice = train_pred_all_ro.as_slice().map_err(|e| {
            PyRuntimeError::new_err(format!("train prediction buffer is not contiguous: {e}"))
        })?;
        if train_pred_all_slice.len() != n_train {
            return Err(PyRuntimeError::new_err(format!(
                "train prediction length mismatch: got {}, expected {}",
                train_pred_all_slice.len(),
                n_train
            )));
        }
        let mut pred_train_all: Vec<f64> =
            train_pred_all_slice.iter().map(|v| *v + beta0).collect();
        let pred_train_local: Vec<f64> = if let Some(local_idx) = train_pred_pick {
            local_idx.iter().map(|&i| pred_train_all[i]).collect()
        } else {
            std::mem::take(&mut pred_train_all)
        };

        let pred_test_local: Vec<f64> = if test_idx.is_empty() {
            Vec::new()
        } else {
            let test_idx_i64: Vec<i64> = test_idx.iter().map(|&v| v as i64).collect();
            let test_idx_arr =
                PyArray1::from_owned_array(py, Array1::from_vec(test_idx_i64)).into_bound();
            let test_pred_arr = cross_grm_times_alpha_packed_f64(
                py,
                packed_keep_arr_ref.readonly(),
                n_samples,
                row_flip_keep_arr_ref.readonly(),
                maf_keep_arr_ref.readonly(),
                test_idx_arr.readonly(),
                m_alpha_arr.readonly(),
                m_mean_arr.readonly(),
                alpha_sum,
                mean_sq,
                mean_malpha,
                var_sum,
                block_rows.max(1),
                threads,
            )?;
            let test_pred_ro = test_pred_arr.readonly();
            let test_pred_slice = test_pred_ro.as_slice().map_err(|e| {
                PyRuntimeError::new_err(format!("test prediction buffer is not contiguous: {e}"))
            })?;
            test_pred_slice.iter().map(|v| *v + beta0).collect()
        };
        (pred_train_local, pred_test_local)
    };

    let train_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((pred_train.len(), 1), pred_train)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let test_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((pred_test.len(), 1), pred_test)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let effect_arr = if return_effect {
        PyArray1::from_owned_array(py, Array1::from_vec(effect_beta_vec)).into_bound()
    } else {
        PyArray1::from_owned_array(py, Array1::from_vec(Vec::<f64>::new())).into_bound()
    };
    let _effect_alpha0_out = if return_effect {
        effect_alpha0
    } else {
        f64::NAN
    };
    Ok((
        train_arr,
        test_arr,
        pve,
        lambda_opt,
        ml,
        reml,
        evd_backend,
        evd_elapsed,
        eff_m,
        sigma_g2_out,
        sigma_e2_out,
        effect_arr,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    sample_indices,
    alpha,
    row_source_indices,
    row_flip,
    row_maf,
    mode="a",
    block_rows=4096,
    threads=0,
    mmap_window_mb=None
))]
pub fn gblup_effect_from_meta_stream<'py>(
    py: Python<'py>,
    prefix: String,
    sample_indices: PyReadonlyArray1<'py, i64>,
    alpha: PyReadonlyArray1<'py, f64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    mode: &str,
    block_rows: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mode_norm = mode.trim().to_ascii_lowercase();
    let kernel_mode = match mode_norm.as_str() {
        "a" | "add" | "additive" => StreamKernelMode::Additive,
        "d" | "dom" | "dominance" => StreamKernelMode::Dominance,
        _ => {
            return Err(PyRuntimeError::new_err(
                "mode must be one of {'a','additive','d','dominance'}",
            ))
        }
    };
    let n_samples = crate::gfcore::read_fam(&prefix)
        .map_err(map_err_string_to_py)?
        .len();
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("No samples found in BED input."));
    }
    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    let alpha_vec: Vec<f64> = match alpha.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => alpha.as_array().iter().copied().collect(),
    };
    if sample_idx.len() != alpha_vec.len() {
        return Err(PyRuntimeError::new_err(format!(
            "alpha length mismatch: got {}, expected {}",
            alpha_vec.len(),
            sample_idx.len()
        )));
    }

    let row_source_vec_i64: Vec<i64> = match row_source_indices.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_source_indices.as_array().iter().copied().collect(),
    };
    if row_source_vec_i64.is_empty() {
        return Err(PyRuntimeError::new_err(
            "row_source_indices must not be empty.",
        ));
    }
    let mut row_source_vec = Vec::<usize>::with_capacity(row_source_vec_i64.len());
    for &idx in row_source_vec_i64.iter() {
        if idx < 0 {
            return Err(PyRuntimeError::new_err(
                "row_source_indices must be non-negative.",
            ));
        }
        row_source_vec.push(idx as usize);
    }
    let row_flip_vec: Vec<bool> = match row_flip.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip.as_array().iter().copied().collect(),
    };
    let row_maf_vec: Vec<f32> = match row_maf.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_maf.as_array().iter().copied().collect(),
    };
    if row_flip_vec.len() != row_source_vec.len() || row_maf_vec.len() != row_source_vec.len() {
        return Err(PyRuntimeError::new_err(format!(
            "metadata length mismatch: row_source_indices={}, row_flip={}, row_maf={}",
            row_source_vec.len(),
            row_flip_vec.len(),
            row_maf_vec.len(),
        )));
    }

    let effect_beta = py
        .detach(move || {
            compute_effect_beta_from_meta_stream(
                &prefix,
                row_source_vec.as_slice(),
                row_flip_vec.as_slice(),
                row_maf_vec.as_slice(),
                sample_idx.as_slice(),
                alpha_vec.as_slice(),
                kernel_mode,
                block_rows.max(1),
                threads.max(1),
                mmap_window_mb,
            )
        })
        .map_err(map_err_string_to_py)?;
    Ok(PyArray1::from_owned_array(py, Array1::from_vec(effect_beta)).into_bound())
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    method=1,
    block_cols=2048,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Reuse optimized shared GRM backend in `src/stats/grm.rs`.
    grm::grm_packed_f32(
        py,
        packed,
        n_samples,
        row_flip,
        row_maf,
        sample_indices,
        method,
        block_cols,
        threads,
        progress_callback,
        progress_every,
    )
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    block_rows=2048,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn packed_mtm_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    block_rows: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "packed must contain at least one SNP row",
        ));
    }
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let full_sample_fast = is_identity_indices(&sample_idx, n_samples);

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;
    let row_step = adaptive_grm_block_rows(
        block_rows.max(1),
        m.max(1),
        n,
        if full_sample_fast { 0 } else { n_samples },
        threads,
    );
    let notify_step = if progress_every == 0 {
        row_step
    } else {
        progress_every.max(1)
    };
    let cblas_copy_rhs = std::env::var("JX_GRM_PACKED_CBLAS_COPY_RHS")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("1") || s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let cblas_beta_zero_accum = std::env::var("JX_GRM_PACKED_CBLAS_BETA0")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(true);
    let stage_timing = env_truthy("JX_MLM_PACKED_MTM_STAGE_TIMING");

    let gram_vec = py
        .detach(move || -> Result<Vec<f64>, String> {
            let _blas_guard = OpenBlasThreadGuard::enter(threads.max(1));
            let total_t0 = Instant::now();
            let mut decode_secs = 0.0_f64;
            let mut gemm_secs = 0.0_f64;
            let mut gram = vec![0.0_f64; n * n];
            let byte_lut = packed_byte_lut();
            let mut block = vec![0.0_f64; row_step * n];
            let mut last_notified = 0usize;

            let mut cur_start = 0usize;
            while cur_start < m {
                check_ctrlc()?;
                let cur_end = (cur_start + row_step).min(m);
                let cur_rows = cur_end.saturating_sub(cur_start);
                if cur_rows == 0 {
                    break;
                }

                let decode_t0 = Instant::now();
                decode_raw_block_f64(
                    packed_flat.as_ref(),
                    bytes_per_snp,
                    n_samples,
                    row_flip_vec.as_ref(),
                    row_maf_vec.as_ref(),
                    &sample_idx,
                    full_sample_fast,
                    &byte_lut.code4,
                    cur_start,
                    cur_end,
                    n,
                    &mut block,
                    pool.as_ref(),
                )?;
                decode_secs += decode_t0.elapsed().as_secs_f64();

                let gemm_t0 = Instant::now();
                grm::grm_rankk_update_f64(
                    &mut gram,
                    &block[..cur_rows * n],
                    cur_rows,
                    n,
                    cblas_copy_rhs,
                    cur_start == 0,
                    cblas_beta_zero_accum,
                )?;
                gemm_secs += gemm_t0.elapsed().as_secs_f64();

                if (cur_end >= last_notified.saturating_add(notify_step)) || (cur_end == m) {
                    last_notified = cur_end;
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (cur_end, m))?;
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                    } else {
                        check_ctrlc()?;
                    }
                }

                cur_start = cur_end;
            }

            for i in 0..n {
                for j in 0..i {
                    let idx_lo = i * n + j;
                    let idx_up = j * n + i;
                    let v = gram[idx_lo];
                    gram[idx_lo] = v;
                    gram[idx_up] = v;
                }
            }

            if stage_timing {
                let total_secs = total_t0.elapsed().as_secs_f64();
                let other_secs = (total_secs - decode_secs - gemm_secs).max(0.0_f64);
                let to_pct = |x: f64| -> f64 {
                    if total_secs > 0.0 {
                        x * 100.0 / total_secs
                    } else {
                        0.0
                    }
                };
                eprintln!(
                    "Packed MTM stage timing: decode={:.3}s ({:.1}%), gemm={:.3}s ({:.1}%), \
other={:.3}s ({:.1}%), total={:.3}s, row_step={}, n_samples={}, full_sample={}, threads={}",
                    decode_secs,
                    to_pct(decode_secs),
                    gemm_secs,
                    to_pct(gemm_secs),
                    other_secs,
                    to_pct(other_secs),
                    total_secs,
                    row_step,
                    n,
                    if full_sample_fast { "yes" } else { "no" },
                    if threads > 0 {
                        threads
                    } else {
                        rayon::current_num_threads()
                    },
                );
            }
            Ok(gram)
        })
        .map_err(map_err_string_to_py)?;

    let gram_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n, n), gram_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok(gram_arr)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    qdim,
    sample_indices=None,
    method=1,
    block_cols=2048,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn farmcpu_q_packed_grm_pca_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    qdim: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    String,
    f64,
)> {
    let grm_arr = grm_packed_f32(
        py,
        packed,
        n_samples,
        row_flip,
        row_maf,
        sample_indices,
        method,
        block_cols,
        threads,
        progress_callback,
        progress_every,
    )?;

    let (n, grm_f64) = {
        let grm_ro = grm_arr.readonly();
        let grm_view = grm_ro.as_array();
        if grm_view.ndim() != 2 {
            return Err(PyRuntimeError::new_err(
                "farmcpu_q_packed_grm_pca_f32 internal GRM is not 2D.",
            ));
        }
        let n0 = grm_view.shape()[0];
        let n1 = grm_view.shape()[1];
        if n0 != n1 {
            return Err(PyRuntimeError::new_err(format!(
                "farmcpu_q_packed_grm_pca_f32 internal GRM is not square: ({n0}, {n1})"
            )));
        }
        if qdim >= n0 && qdim != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "qdim out of range: got {qdim}, valid=[0..{}]",
                n0.saturating_sub(1)
            )));
        }
        let grm_slice = grm_ro
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(format!("GRM buffer is not contiguous: {e}")))?;
        let mut tmp = Vec::with_capacity(grm_slice.len());
        tmp.extend(grm_slice.iter().map(|v| *v as f64));
        (n0, tmp)
    };

    if qdim == 0 {
        let q_empty = PyArray2::from_owned_array(
            py,
            Array2::from_shape_vec((n, 0), Vec::<f32>::new())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
        .into_bound();
        return Ok((grm_arr, q_empty, "none".to_string(), 0.0_f64));
    }

    let (q_vec, evd_backend, evd_elapsed) = py
        .detach(move || -> Result<(Vec<f32>, String, f64), String> {
            let _blas_guard = OpenBlasThreadGuard::enter(threads.max(1));
            let t0 = Instant::now();
            let (_eval, evec, backend) = symmetric_eigh_f64_row_major(&grm_f64, n)?;
            let evd_secs = t0.elapsed().as_secs_f64();

            let mut q = vec![0.0_f32; n.saturating_mul(qdim)];
            let src0 = n.saturating_sub(qdim);
            for r in 0..n {
                let src = &evec[r * n + src0..r * n + n];
                let dst = &mut q[r * qdim..(r + 1) * qdim];
                for (j, v) in src.iter().enumerate() {
                    dst[j] = *v as f32;
                }
            }
            Ok((q, backend.to_string(), evd_secs))
        })
        .map_err(map_err_string_to_py)?;

    let q_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n, qdim), q_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok((grm_arr, q_arr, evd_backend, evd_elapsed))
}
