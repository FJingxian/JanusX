//! rrBLUP marker-effect solvers over additive PLINK BED blocks.
//!
//! Let `W` denote the standardized additive genotype matrix after site
//! filtering and sample subsetting, with markers in rows and samples in
//! columns. With phenotype `y`, fixed-effect design `X`, and
//! `lambda = sigma_e^2 / sigma_beta^2`, rrBLUP solves
//!
//! `y = X b + W' beta + e,  beta ~ N(0, sigma_beta^2 I),  e ~ N(0, sigma_e^2 I)`.
//!
//! This module maintains two marker-space backends:
//!
//! 1. PCG over streamed BED blocks for large-marker settings:
//!    `(W W' + lambda I_m) beta_hat = W (y - X b_hat)`.
//! 2. Low-SNP exact spectral rrBLUP for `m << n`, specialized to the current
//!    GS intercept-only workflow (`X = 1_n`). With `M_1 = I - 11'/n`,
//!    `A_* = W M_1 W'`, and `z = W M_1 y`, the exact backend fits
//!    `beta_hat = (A_* + lambda I_m)^{-1} z`
//!    after REML optimization on the non-zero spectrum of `A_*`.
//!
//! The exact path keeps one packed decode pass over sample blocks to build
//! `A_*` and `z`, then reuses the spectral cache to solve marker effects and
//! predict GEBV as `g_hat = W' beta_hat + mean(y)`.
//!
//! The implementation keeps BLAS inside dense linear algebra kernels and uses
//! Rayon for outer decode / reduction work so both backends stay memory-bounded
//! on packed BED input.

use numpy::ndarray::{Array1, Array2};
use numpy::PyArray1;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::bedmath::{
    decode_standardized_packed_block_rows_f32, is_identity_indices, packed_byte_lut,
};
use crate::blas::{
    cblas_dgemm_dispatch, cblas_dsyrk_dispatch, cblas_sgemm_dispatch,
    rust_sgemm_prefers_rayon_rowmajor_f32_kernel, BlasThreadGuard, CblasInt, OpenBlasThreadGuard,
    CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_UPPER,
};
use crate::brent::brent_minimize;
use crate::decode::decode_prepared_additive_block_packed_f32;
use crate::eigh::symmetric_eigh_f64_row_major;
use crate::gload::WindowedBedMatrix;
use crate::packed::bed_packed_row_flip_mask;
use crate::pcg::{pcg_solve, DiagonalPreconditioner, IdentityPreconditioner, PcgOperator};
use crate::pipeline::run_double_buffer;
use crate::stats_common::{
    check_ctrlc, env_truthy, format_bytes, get_cached_pool, map_err_string_to_py,
    parse_index_vec_i64, process_memory_usage,
};

#[inline]
fn prefer_parallel_block_vec(
    rows: usize,
    cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> bool {
    let Some(tp) = pool else {
        return false;
    };
    tp.current_num_threads() > 1 && rows.saturating_mul(cols) >= 1_048_576
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum XtMulKernelStrategy {
    RowReduce,
    ColChunk,
    Blas,
}

#[inline]
fn xtmul_row_chunk(rows: usize) -> usize {
    256usize.min(rows.max(1))
}

#[inline]
fn xtmul_col_chunk(cols: usize, threads: usize) -> usize {
    let target_tasks = threads.saturating_mul(2).max(1);
    let mut chunk = cols.div_ceil(target_tasks).max(8192);
    let align = 256usize;
    let rem = chunk % align;
    if rem != 0 {
        chunk += align - rem;
    }
    chunk.min(cols.max(1))
}

#[inline]
fn xtmul_prefers_col_chunk_shape(rows: usize, cols: usize) -> bool {
    // In GS rrBLUP-PCG, `rows` is the decoded SNP-block height and `cols` is the
    // train-sample width. The memory planner already chooses a near-optimal
    // block height, so auto strategy should track block aspect ratio instead of
    // hard-coded absolute thresholds. Once the row-major block is wide enough,
    // column chunking avoids the extra reduction pass in RowReduce.
    cols >= 32_768 && cols >= rows.saturating_mul(8)
}

#[inline]
fn xtmul_forced_strategy() -> Option<XtMulKernelStrategy> {
    static OVERRIDE: OnceLock<Option<XtMulKernelStrategy>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        let raw = std::env::var("JX_GS_PCG_XTMUL_KERNEL").ok()?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => None,
            "row_reduce" | "legacy" | "rows" => Some(XtMulKernelStrategy::RowReduce),
            "col_chunk" | "cols" | "columns" => Some(XtMulKernelStrategy::ColChunk),
            "blas" | "gemm" => Some(XtMulKernelStrategy::Blas),
            _ => None,
        }
    })
}

#[inline]
fn rrblup_exact_memory_snapshot() -> String {
    match process_memory_usage() {
        Some(usage) => {
            let rss = usage
                .rss_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "NA".to_string());
            let footprint = usage
                .footprint_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "NA".to_string());
            format!(
                "{}={} rss={} footprint={}",
                usage.metric,
                format_bytes(usage.current_bytes),
                rss,
                footprint,
            )
        }
        None => "memory=NA".to_string(),
    }
}

#[inline]
fn choose_xtmul_kernel_strategy(
    rows: usize,
    cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> XtMulKernelStrategy {
    if let Some(force) = xtmul_forced_strategy() {
        return force;
    }
    let prefer_parallel = prefer_parallel_block_vec(rows, cols, pool)
        && rust_sgemm_prefers_rayon_rowmajor_f32_kernel();
    if !prefer_parallel {
        return XtMulKernelStrategy::Blas;
    }
    if xtmul_prefers_col_chunk_shape(rows, cols) {
        return XtMulKernelStrategy::ColChunk;
    }
    XtMulKernelStrategy::RowReduce
}

#[inline]
fn row_major_block_mul_vec_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);
    let prefer_parallel = prefer_parallel_block_vec(rows, cols, pool)
        && rust_sgemm_prefers_rayon_rowmajor_f32_kernel();
    if prefer_parallel {
        let mut run = || {
            out.par_iter_mut().enumerate().for_each(|(r, out_cell)| {
                let row = &block[r * cols..(r + 1) * cols];
                let mut acc = 0.0_f32;
                for c in 0..cols {
                    acc += row[c] * vec[c];
                }
                *out_cell = acc;
            });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            rows as CblasInt,
            1 as CblasInt,
            cols as CblasInt,
            1.0,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            cols as CblasInt,
            0.0,
            out.as_mut_ptr(),
            rows as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for r in 0..rows {
            let row = &block[r * cols..(r + 1) * cols];
            let mut acc = 0.0_f32;
            for c in 0..cols {
                acc += row[c] * vec[c];
            }
            out[r] = acc;
        }
    }
}

#[inline]
fn row_major_block_t_mul_vec_accum_f32_row_reduce(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);
    let row_chunk = xtmul_row_chunk(rows);
    let mut run = || {
        let partial = block
            .par_chunks(row_chunk * cols)
            .zip(vec.par_chunks(row_chunk))
            .fold(
                || vec![0.0_f32; cols],
                |mut acc, (block_chunk, vec_chunk)| {
                    let rows_here = vec_chunk.len();
                    for r in 0..rows_here {
                        let vr = vec_chunk[r];
                        let row = &block_chunk[r * cols..(r + 1) * cols];
                        for c in 0..cols {
                            acc[c] += row[c] * vr;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0_f32; cols],
                |mut left, right| {
                    for c in 0..cols {
                        left[c] += right[c];
                    }
                    left
                },
            );
        for c in 0..cols {
            out[c] += partial[c];
        }
    };
    if let Some(tp) = pool {
        tp.install(run);
    } else {
        run();
    }
}

#[inline]
fn row_major_block_t_mul_vec_accum_f32_col_chunk(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);
    let threads = pool.map(|tp| tp.current_num_threads()).unwrap_or(1).max(1);
    let col_chunk = xtmul_col_chunk(cols, threads);
    let mut run = || {
        out.par_chunks_mut(col_chunk)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * col_chunk;
                let col_end = col_start + out_chunk.len();
                for r in 0..rows {
                    let vr = vec[r];
                    let row = &block[r * cols + col_start..r * cols + col_end];
                    for (dst, &a) in out_chunk.iter_mut().zip(row.iter()) {
                        *dst += a * vr;
                    }
                }
            });
    };
    if let Some(tp) = pool {
        tp.install(run);
    } else {
        run();
    }
}

#[inline]
fn row_major_block_t_mul_vec_accum_f32_blas_or_serial(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            cols as CblasInt,
            1 as CblasInt,
            rows as CblasInt,
            1.0,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            rows as CblasInt,
            1.0,
            out.as_mut_ptr(),
            cols as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for r in 0..rows {
            let vr = vec[r];
            let row = &block[r * cols..(r + 1) * cols];
            for c in 0..cols {
                out[c] += row[c] * vr;
            }
        }
    }
}

#[inline]
fn row_major_block_t_mul_vec_accum_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    match choose_xtmul_kernel_strategy(rows, cols, pool) {
        XtMulKernelStrategy::RowReduce => {
            row_major_block_t_mul_vec_accum_f32_row_reduce(block, rows, cols, vec, out, pool)
        }
        XtMulKernelStrategy::ColChunk => {
            row_major_block_t_mul_vec_accum_f32_col_chunk(block, rows, cols, vec, out, pool)
        }
        XtMulKernelStrategy::Blas => {
            row_major_block_t_mul_vec_accum_f32_blas_or_serial(block, rows, cols, vec, out)
        }
    }
}

#[inline]
fn row_major_row_sumsq_f64(row: &[f32]) -> f64 {
    let mut ss = 0.0_f64;
    for &v in row {
        let vf = v as f64;
        ss += vf * vf;
    }
    ss
}

#[inline]
fn row_major_row_sum_f64(row: &[f32]) -> f64 {
    row.iter().map(|&v| v as f64).sum::<f64>()
}

#[inline]
fn row_major_block_prepare_rhs_diag_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    dot_blk: &[f32],
    lambda_use: f32,
    b_out: &mut [f32],
    diag_inv_out: &mut [f32],
    train_row_mean_out: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> f64 {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert!(dot_blk.len() >= rows);
    debug_assert!(b_out.len() >= rows);
    debug_assert!(diag_inv_out.len() >= rows);
    debug_assert!(train_row_mean_out.len() >= rows);

    if prefer_parallel_block_vec(rows, cols, pool) {
        let mut run = || {
            block
                .par_chunks(cols)
                .zip(dot_blk[..rows].par_iter())
                .zip(
                    b_out[..rows]
                        .par_iter_mut()
                        .zip(diag_inv_out[..rows].par_iter_mut())
                        .zip(train_row_mean_out[..rows].par_iter_mut()),
                )
                .map(|((row, dot), ((b_slot, diag_slot), mean_slot))| {
                    let sum = row_major_row_sum_f64(row);
                    let mean = sum / (cols as f64);
                    let ss_raw = row_major_row_sumsq_f64(row);
                    let ss = (ss_raw - (cols as f64) * mean * mean).max(0.0_f64);
                    *b_slot = *dot;
                    *mean_slot = mean as f32;
                    let d = ((ss as f32) + lambda_use).max(1e-12_f32);
                    *diag_slot = 1.0_f32 / d;
                    ss
                })
                .sum::<f64>()
        };
        if let Some(tp) = pool {
            return tp.install(run);
        }
        return run();
    }

    let mut sum_ss = 0.0_f64;
    for r in 0..rows {
        let row = &block[r * cols..(r + 1) * cols];
        let sum = row_major_row_sum_f64(row);
        let mean = sum / (cols as f64);
        let ss_raw = row_major_row_sumsq_f64(row);
        let ss = (ss_raw - (cols as f64) * mean * mean).max(0.0_f64);
        sum_ss += ss;
        b_out[r] = dot_blk[r];
        train_row_mean_out[r] = mean as f32;
        let d = ((ss as f32) + lambda_use).max(1e-12_f32);
        diag_inv_out[r] = 1.0_f32 / d;
    }
    sum_ss
}

#[derive(Clone, Copy, Debug, Default)]
struct PcgMatvecTiming {
    decode_secs: f64,
    mul_secs: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct PcgStageTimingAccum {
    decode_secs: f64,
    xmul_secs: f64,
    xtmul_secs: f64,
    callback_secs: f64,
}

enum RrblupPcgSource<'a> {
    Resident {
        packed_flat: &'a [u8],
        bytes_per_snp: usize,
    },
    Windowed {
        matrix: WindowedBedMatrix,
        bytes_per_snp: usize,
    },
}

struct DecodedBlockBuffer {
    row_start: usize,
    row_end: usize,
    decode_secs: f64,
    block: Vec<f32>,
}

struct PrepDecodedBlockBuffer {
    row_start: usize,
    row_end: usize,
    decode_secs: f64,
    block: Vec<f32>,
    dot: Vec<f32>,
}

const RRBLUP_PAR_VEC_THRESHOLD: usize = 16_384;

#[inline]
fn rrblup_stream_window_mb(n_samples: usize, block_rows: usize) -> usize {
    let bytes_per_snp = n_samples.div_ceil(4).max(1);
    let target_bytes = block_rows
        .max(1)
        .saturating_mul(bytes_per_snp)
        .saturating_mul(2);
    target_bytes.div_ceil(1024 * 1024).max(1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RrblupScaleMode {
    Standardized,
    Centered,
}

#[inline]
fn rrblup_parallel_row_standardization(
    maf_keep: &[f32],
    std_eps32: f32,
    pool: Option<&Arc<rayon::ThreadPool>>,
    scale_mode: RrblupScaleMode,
) -> (Vec<f32>, Vec<f32>, usize) {
    let m = maf_keep.len();
    let mut row_mean = vec![0.0_f32; m];
    let mut row_inv_sd = vec![0.0_f32; m];
    if m == 0 {
        return (row_mean, row_inv_sd, 0usize);
    }
    let mut run_parallel = || {
        row_mean
            .par_iter_mut()
            .zip(row_inv_sd.par_iter_mut())
            .zip(maf_keep.par_iter())
            .map(|((mean_slot, inv_slot), &maf_raw)| {
                let p = maf_raw.clamp(0.0, 0.5);
                let mean = 2.0_f32 * p;
                let var = (2.0_f32 * p * (1.0_f32 - p)).max(0.0);
                *mean_slot = mean;
                if var > std_eps32 {
                    *inv_slot = match scale_mode {
                        RrblupScaleMode::Standardized => 1.0_f32 / var.sqrt(),
                        RrblupScaleMode::Centered => 1.0_f32,
                    };
                    1usize
                } else {
                    *inv_slot = 0.0_f32;
                    0usize
                }
            })
            .sum::<usize>()
    };
    let m_effective = if m >= RRBLUP_PAR_VEC_THRESHOLD {
        if let Some(tp) = pool {
            tp.install(run_parallel)
        } else {
            run_parallel()
        }
    } else {
        let mut count = 0usize;
        for j in 0..m {
            let p = maf_keep[j].clamp(0.0, 0.5);
            let mean = 2.0_f32 * p;
            let var = (2.0_f32 * p * (1.0_f32 - p)).max(0.0);
            row_mean[j] = mean;
            if var > std_eps32 {
                row_inv_sd[j] = match scale_mode {
                    RrblupScaleMode::Standardized => 1.0_f32 / var.sqrt(),
                    RrblupScaleMode::Centered => 1.0_f32,
                };
                count += 1;
            } else {
                row_inv_sd[j] = 0.0_f32;
            }
        }
        count
    };
    (row_mean, row_inv_sd, m_effective)
}

#[allow(clippy::too_many_arguments)]
fn rrblup_subset_or_validate_stats(
    row_mean_external: Option<Vec<f32>>,
    row_inv_sd_external: Option<Vec<f32>>,
    packed_row_indices: Option<&[usize]>,
    full_len: usize,
    eff_m: usize,
    std_eps32: f32,
    maf_keep: &[f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
    scale_mode: RrblupScaleMode,
) -> Result<(Vec<f32>, Vec<f32>, usize), String> {
    match (row_mean_external, row_inv_sd_external) {
        (Some(row_mean_vec), Some(row_inv_sd_vec)) => {
            let pair = if row_mean_vec.len() == eff_m && row_inv_sd_vec.len() == eff_m {
                (row_mean_vec, row_inv_sd_vec)
            } else if row_mean_vec.len() == full_len && row_inv_sd_vec.len() == full_len {
                let keep_idx = packed_row_indices.ok_or_else(|| {
                    format!(
                        "External row_mean/row_inv_sd length mismatch: got full-length arrays ({full_len}) but no packed_row_indices were provided for active subset size {eff_m}."
                    )
                })?;
                let mut mean_subset = vec![0.0_f32; eff_m];
                let mut inv_subset = vec![0.0_f32; eff_m];
                for (dst, &src) in keep_idx.iter().enumerate() {
                    mean_subset[dst] = row_mean_vec[src];
                    inv_subset[dst] = row_inv_sd_vec[src];
                }
                (mean_subset, inv_subset)
            } else {
                return Err(format!(
                    "External row_mean/row_inv_sd length mismatch: mean={}, inv={}, expected active={eff_m} or full={full_len}.",
                    row_mean_vec.len(),
                    row_inv_sd_vec.len(),
                ));
            };
            let (row_mean, row_inv_sd) = pair;
            let m_effective = row_inv_sd
                .iter()
                .filter(|&&v| v.is_finite() && (v > 0.0_f32))
                .count();
            if m_effective == 0 {
                return Err(
                    "rrBLUP standardization received zero effective markers from external row_inv_sd."
                        .to_string(),
                );
            }
            Ok((row_mean, row_inv_sd, m_effective))
        }
        (None, None) => Ok(rrblup_parallel_row_standardization(
            maf_keep,
            std_eps32,
            pool,
            scale_mode,
        )),
        _ => Err(
            "rrBLUP standardization requires row_mean and row_inv_sd together when overriding row stats."
                .to_string(),
        ),
    }
}

fn rrblupc_subset_or_validate_stats(
    row_mean_external_full: Option<Vec<f32>>,
    row_inv_sd_external_full: Option<Vec<f32>>,
    packed_row_indices: Option<&[usize]>,
    m_total: usize,
    eff_m: usize,
    std_eps32: f32,
    maf_keep: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, usize), String> {
    let row_mean = if let Some(full) = row_mean_external_full {
        if full.len() != m_total {
            return Err(format!(
                "row_mean length mismatch: got {}, expected {m_total}",
                full.len()
            ));
        }
        if let Some(keep_idx) = packed_row_indices {
            keep_idx.iter().map(|&src| full[src]).collect()
        } else {
            full
        }
    } else {
        maf_keep
            .iter()
            .map(|&p| 2.0_f32 * p.clamp(0.0, 0.5))
            .collect()
    };
    if row_mean.len() != eff_m {
        return Err(format!(
            "row_mean active length mismatch: got {}, expected {eff_m}",
            row_mean.len()
        ));
    }

    let row_inv_sd = if let Some(full) = row_inv_sd_external_full {
        if full.len() != m_total {
            return Err(format!(
                "row_inv_sd length mismatch: got {}, expected {m_total}",
                full.len()
            ));
        }
        if let Some(keep_idx) = packed_row_indices {
            keep_idx.iter().map(|&src| full[src]).collect()
        } else {
            full
        }
    } else {
        let mut inv = vec![0.0_f32; eff_m];
        for j in 0..eff_m {
            let p = maf_keep[j].clamp(0.0, 0.5);
            let var = (2.0_f32 * p * (1.0_f32 - p)).max(0.0);
            if var > std_eps32 {
                inv[j] = 1.0_f32;
            }
        }
        inv
    };
    if row_inv_sd.len() != eff_m {
        return Err(format!(
            "row_inv_sd active length mismatch: got {}, expected {eff_m}",
            row_inv_sd.len()
        ));
    }
    let m_effective = row_inv_sd
        .iter()
        .filter(|&&v| v.is_finite() && v > 0.0_f32)
        .count();
    Ok((row_mean, row_inv_sd, m_effective))
}

fn build_rrblup_source<'a>(
    resident_packed_flat: Option<&'a [u8]>,
    bytes_per_snp: usize,
    prefix: &str,
    n_samples: usize,
    block_rows: usize,
) -> Result<RrblupPcgSource<'a>, String> {
    if let Some(packed_flat) = resident_packed_flat {
        Ok(RrblupPcgSource::Resident {
            packed_flat,
            bytes_per_snp,
        })
    } else {
        let window_mb = rrblup_stream_window_mb(n_samples, block_rows);
        let matrix = WindowedBedMatrix::open(prefix, window_mb)?;
        Ok(RrblupPcgSource::Windowed {
            matrix,
            bytes_per_snp,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn rrblup_prepare_rhs_diag(
    source: &mut RrblupPcgSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    block_rows: usize,
    y_center_f32: &[f32],
    lambda_use: f32,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, f64, f64, f64, f64), String> {
    let m = row_flip.len();
    let n_train = sample_idx.len();
    let row_step = block_rows.max(1).min(m.max(1));
    let mut b = vec![0.0_f32; m];
    let mut diag_inv = vec![0.0_f32; m];
    let mut train_row_mean = vec![0.0_f32; m];
    let mut sum_ss_global = 0.0_f64;
    let mut decode_acc = 0.0_f64;
    let mut rhs_acc = 0.0_f64;
    let mut diag_acc = 0.0_f64;
    let use_double_buffer = matches!(source, RrblupPcgSource::Windowed { .. }) && m > row_step;

    if use_double_buffer {
        let pool_cloned = pool.cloned();
        let mut next_row = 0usize;
        let producer_err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let producer_err_bg = Arc::clone(&producer_err);
        run_double_buffer(
            2usize,
            || PrepDecodedBlockBuffer {
                row_start: 0,
                row_end: 0,
                decode_secs: 0.0,
                block: vec![0.0_f32; row_step * n_train],
                dot: vec![0.0_f32; row_step],
            },
            |buf| {
                if next_row >= m {
                    return false;
                }
                let st = next_row;
                let ed = (st + row_step).min(m);
                let cur_rows = ed - st;
                buf.row_start = st;
                buf.row_end = ed;
                let blk_slice = &mut buf.block[..cur_rows * n_train];
                let t_decode = Instant::now();
                if let Err(err) = decode_standardized_block_f32(
                    source,
                    n_samples,
                    row_flip,
                    row_mean,
                    row_inv_sd,
                    sample_idx,
                    full_sample_fast,
                    packed_row_indices,
                    st,
                    blk_slice,
                    code4_lut,
                    pool_cloned.as_ref(),
                ) {
                    if let Ok(mut slot) = producer_err_bg.lock() {
                        *slot = Some(err);
                    }
                    buf.row_end = buf.row_start;
                    next_row = m;
                    return false;
                }
                buf.decode_secs = t_decode.elapsed().as_secs_f64();
                next_row = ed;
                next_row < m
            },
            |buf| {
                if let Ok(slot) = producer_err.lock() {
                    if let Some(err) = slot.as_ref() {
                        return Err(err.clone());
                    }
                }
                let st = buf.row_start;
                let ed = buf.row_end;
                let cur_rows = ed - st;
                decode_acc += buf.decode_secs;
                let blk_slice = &buf.block[..cur_rows * n_train];
                let dot_slice = &mut buf.dot[..cur_rows];
                let t_rhs = Instant::now();
                row_major_block_mul_vec_f32(
                    blk_slice,
                    cur_rows,
                    n_train,
                    y_center_f32,
                    dot_slice,
                    pool,
                );
                rhs_acc += t_rhs.elapsed().as_secs_f64();
                let t_diag = Instant::now();
                sum_ss_global += row_major_block_prepare_rhs_diag_f32(
                    blk_slice,
                    cur_rows,
                    n_train,
                    dot_slice,
                    lambda_use,
                    &mut b[st..ed],
                    &mut diag_inv[st..ed],
                    &mut train_row_mean[st..ed],
                    pool,
                );
                diag_acc += t_diag.elapsed().as_secs_f64();
                check_ctrlc()?;
                Ok::<(), String>(())
            },
        )?;
        let final_err = producer_err.lock().ok().and_then(|slot| slot.clone());
        if let Some(err) = final_err {
            return Err(err);
        }
    } else {
        let mut block = vec![0.0_f32; row_step * n_train];
        let mut dot_blk = vec![0.0_f32; row_step];
        let mut tick = 0usize;
        for st in (0..m).step_by(row_step) {
            let ed = (st + row_step).min(m);
            let cur_rows = ed - st;
            let blk_slice = &mut block[..cur_rows * n_train];
            let t_decode = Instant::now();
            decode_standardized_block_f32(
                source,
                n_samples,
                row_flip,
                row_mean,
                row_inv_sd,
                sample_idx,
                full_sample_fast,
                packed_row_indices,
                st,
                blk_slice,
                code4_lut,
                pool,
            )?;
            decode_acc += t_decode.elapsed().as_secs_f64();
            let t_rhs = Instant::now();
            row_major_block_mul_vec_f32(
                blk_slice,
                cur_rows,
                n_train,
                y_center_f32,
                &mut dot_blk[..cur_rows],
                pool,
            );
            rhs_acc += t_rhs.elapsed().as_secs_f64();
            let t_diag = Instant::now();
            sum_ss_global += row_major_block_prepare_rhs_diag_f32(
                blk_slice,
                cur_rows,
                n_train,
                &dot_blk[..cur_rows],
                lambda_use,
                &mut b[st..ed],
                &mut diag_inv[st..ed],
                &mut train_row_mean[st..ed],
                pool,
            );
            diag_acc += t_diag.elapsed().as_secs_f64();
            tick += cur_rows;
            if tick >= row_step.saturating_mul(64).max(1) {
                check_ctrlc()?;
                tick = 0;
            }
        }
    }

    Ok((
        b,
        diag_inv,
        train_row_mean,
        sum_ss_global,
        decode_acc,
        rhs_acc,
        diag_acc,
    ))
}

struct RrblupPcgOperator<'a> {
    source: RrblupPcgSource<'a>,
    n_samples: usize,
    n_train: usize,
    row_flip: &'a [bool],
    row_mean: &'a [f32],
    row_inv_sd: &'a [f32],
    train_row_mean: &'a [f32],
    sample_idx: &'a [usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&'a [usize]>,
    block_rows: usize,
    code4_lut: &'a [[u8; 4]; 256],
    pool: Option<&'a Arc<rayon::ThreadPool>>,
    lambda_use: f32,
    stage_timing: bool,
    stage_accum: &'a Mutex<PcgStageTimingAccum>,
}

impl PcgOperator<f32> for RrblupPcgOperator<'_> {
    fn apply(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        let mut xmul_t = PcgMatvecTiming::default();
        let xp = pcg_x_mul_samples(
            &mut self.source,
            self.n_samples,
            self.row_flip,
            self.row_mean,
            self.row_inv_sd,
            self.sample_idx,
            self.full_sample_fast,
            self.packed_row_indices,
            self.block_rows,
            input,
            self.code4_lut,
            self.pool,
            if self.stage_timing {
                Some(&mut xmul_t)
            } else {
                None
            },
        )?;
        let mut xtmul_t = PcgMatvecTiming::default();
        let mut ap = pcg_xt_mul_rows(
            &mut self.source,
            self.n_samples,
            self.row_flip,
            self.row_mean,
            self.row_inv_sd,
            self.sample_idx,
            self.full_sample_fast,
            self.packed_row_indices,
            self.block_rows,
            &xp,
            self.code4_lut,
            self.pool,
            if self.stage_timing {
                Some(&mut xtmul_t)
            } else {
                None
            },
        )?;
        let mean_dot = self
            .train_row_mean
            .iter()
            .zip(input.iter())
            .map(|(mu, pj)| (*mu as f64) * (*pj as f64))
            .sum::<f64>() as f32;
        if let Some(tp) = self.pool {
            tp.install(|| {
                ap.par_iter_mut()
                    .zip(input.par_iter())
                    .zip(self.train_row_mean.par_iter())
                    .for_each(|((apj, pj), mu)| {
                        *apj -= (self.n_train as f32) * *mu * mean_dot;
                        *apj += self.lambda_use * *pj;
                    });
            });
        } else {
            for j in 0..ap.len() {
                ap[j] -= (self.n_train as f32) * self.train_row_mean[j] * mean_dot;
                ap[j] += self.lambda_use * input[j];
            }
        }
        output.copy_from_slice(&ap);
        if self.stage_timing {
            if let Ok(mut acc) = self.stage_accum.lock() {
                acc.decode_secs += xmul_t.decode_secs + xtmul_t.decode_secs;
                acc.xmul_secs += xmul_t.mul_secs;
                acc.xtmul_secs += xtmul_t.mul_secs;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RrblupPcgSpace {
    Dual,
    Primal,
}

impl RrblupPcgSpace {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "dual" | "sample" => Ok(Self::Dual),
            "primal" | "marker" | "snp" => Ok(Self::Primal),
            other => Err(format!(
                "Unsupported JX_RRBLUP_PCG_SPACE '{other}'; expected 'dual' or 'primal'."
            )),
        }
    }
}

#[inline]
fn row_centered_sum_squares_f32_to_f64(
    block: &[f32],
    rows: usize,
    cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> f64 {
    if rows == 0 || cols == 0 {
        return 0.0_f64;
    }
    let cols_f64 = cols as f64;
    let row_ss = |row: &[f32]| -> f64 {
        let mean = row_major_row_sum_f64(row) / cols_f64;
        let ss_raw = row_major_row_sumsq_f64(row);
        (ss_raw - cols_f64 * mean * mean).max(0.0_f64)
    };
    if let Some(tp) = pool.filter(|_| rows.saturating_mul(cols) >= 1_048_576) {
        tp.install(|| block.par_chunks_exact(cols).map(row_ss).sum::<f64>())
    } else {
        block.chunks_exact(cols).map(row_ss).sum::<f64>()
    }
}

struct RrblupDualPcgOperator<'a> {
    source: RrblupPcgSource<'a>,
    n_samples: usize,
    n_train: usize,
    m: usize,
    row_flip: &'a [bool],
    row_mean: &'a [f32],
    row_inv_sd: &'a [f32],
    sample_idx: &'a [usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&'a [usize]>,
    block_rows: usize,
    code4_lut: &'a [[u8; 4]; 256],
    pool: Option<&'a Arc<rayon::ThreadPool>>,
    lambda_use: f32,
    stage_timing: bool,
    stage_accum: &'a Mutex<PcgStageTimingAccum>,
    block_buf: Vec<f32>,
    tmp_buf: Vec<f32>,
    q_buf: Vec<f32>,
    sum_ss_global: &'a mut f64,
    sum_ss_captured: &'a mut bool,
}

impl PcgOperator<f32> for RrblupDualPcgOperator<'_> {
    fn apply(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        if input.len() != self.n_train || output.len() != self.n_train {
            return Err(format!(
                "RrblupDualPcgOperator length mismatch: input={}, output={}, n_train={}",
                input.len(),
                output.len(),
                self.n_train
            ));
        }
        let n = self.n_train;
        if n == 0 {
            return Ok(());
        }
        let mean_p = (input.iter().map(|&v| v as f64).sum::<f64>() / (n as f64)) as f32;
        for i in 0..n {
            self.q_buf[i] = input[i] - mean_p;
        }
        output.fill(0.0_f32);

        let capture_ss = !*self.sum_ss_captured;
        let mut ss_acc = 0.0_f64;
        let mut decode_acc = 0.0_f64;
        let mut xmul_acc = 0.0_f64;
        let mut xtmul_acc = 0.0_f64;
        let row_step = self.block_rows.max(1).min(self.m.max(1));
        let use_double_buffer =
            matches!(self.source, RrblupPcgSource::Windowed { .. }) && self.m > row_step;

        if use_double_buffer {
            let source = &mut self.source;
            let n_samples = self.n_samples;
            let m = self.m;
            let row_flip = self.row_flip;
            let row_mean = self.row_mean;
            let row_inv_sd = self.row_inv_sd;
            let sample_idx = self.sample_idx;
            let full_sample_fast = self.full_sample_fast;
            let packed_row_indices = self.packed_row_indices;
            let code4_lut = self.code4_lut;
            let pool = self.pool;
            let stage_timing = self.stage_timing;
            let q_buf = self.q_buf.as_slice();
            let tmp_buf = self.tmp_buf.as_mut_slice();
            let pool_cloned = pool.cloned();
            let mut next_row = 0usize;
            let producer_err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
            let producer_err_bg = Arc::clone(&producer_err);
            run_double_buffer(
                2usize,
                || DecodedBlockBuffer {
                    row_start: 0,
                    row_end: 0,
                    decode_secs: 0.0,
                    block: vec![0.0_f32; row_step * n],
                },
                |buf| {
                    if next_row >= m {
                        return false;
                    }
                    let st = next_row;
                    let ed = (st + row_step).min(m);
                    let cur_rows = ed - st;
                    buf.row_start = st;
                    buf.row_end = ed;
                    let t_decode = if stage_timing {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    let blk_slice = &mut buf.block[..cur_rows * n];
                    if let Err(err) = decode_standardized_block_f32(
                        source,
                        n_samples,
                        row_flip,
                        row_mean,
                        row_inv_sd,
                        sample_idx,
                        full_sample_fast,
                        packed_row_indices,
                        st,
                        blk_slice,
                        code4_lut,
                        pool_cloned.as_ref(),
                    ) {
                        if let Ok(mut slot) = producer_err_bg.lock() {
                            *slot = Some(err);
                        }
                        buf.row_end = buf.row_start;
                        next_row = m;
                        return false;
                    }
                    buf.decode_secs = t_decode.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
                    next_row = ed;
                    next_row < m
                },
                |buf| {
                    if let Ok(slot) = producer_err.lock() {
                        if let Some(err) = slot.as_ref() {
                            return Err(err.clone());
                        }
                    }
                    let cur_rows = buf.row_end.saturating_sub(buf.row_start);
                    if cur_rows == 0 {
                        return Ok(());
                    }
                    let blk_slice = &buf.block[..cur_rows * n];
                    decode_acc += buf.decode_secs;
                    if capture_ss {
                        ss_acc += row_centered_sum_squares_f32_to_f64(blk_slice, cur_rows, n, pool);
                    }
                    let tmp_slice = &mut tmp_buf[..cur_rows];
                    let t1 = if stage_timing {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    row_major_block_mul_vec_f32(blk_slice, cur_rows, n, q_buf, tmp_slice, pool);
                    if let Some(ts) = t1 {
                        xmul_acc += ts.elapsed().as_secs_f64();
                    }
                    let t2 = if stage_timing {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    row_major_block_t_mul_vec_accum_f32(
                        blk_slice, cur_rows, n, tmp_slice, output, pool,
                    );
                    if let Some(ts) = t2 {
                        xtmul_acc += ts.elapsed().as_secs_f64();
                    }
                    Ok::<(), String>(())
                },
            )?;
            let final_err = producer_err.lock().ok().and_then(|slot| slot.clone());
            if let Some(err) = final_err {
                return Err(err);
            }
        } else {
            let need = row_step.saturating_mul(n);
            if self.block_buf.len() < need {
                self.block_buf.resize(need, 0.0_f32);
            }
            for row_start in (0..self.m).step_by(row_step) {
                let cur_rows = (self.m - row_start).min(row_step);
                let blk_slice = &mut self.block_buf[..cur_rows * n];
                let t0 = if self.stage_timing {
                    Some(Instant::now())
                } else {
                    None
                };
                decode_standardized_block_f32(
                    &mut self.source,
                    self.n_samples,
                    self.row_flip,
                    self.row_mean,
                    self.row_inv_sd,
                    self.sample_idx,
                    self.full_sample_fast,
                    self.packed_row_indices,
                    row_start,
                    blk_slice,
                    self.code4_lut,
                    self.pool,
                )?;
                if let Some(ts) = t0 {
                    decode_acc += ts.elapsed().as_secs_f64();
                }
                if capture_ss {
                    ss_acc +=
                        row_centered_sum_squares_f32_to_f64(blk_slice, cur_rows, n, self.pool);
                }
                let tmp_slice = &mut self.tmp_buf[..cur_rows];
                let t1 = if self.stage_timing {
                    Some(Instant::now())
                } else {
                    None
                };
                row_major_block_mul_vec_f32(
                    blk_slice,
                    cur_rows,
                    n,
                    &self.q_buf,
                    tmp_slice,
                    self.pool,
                );
                if let Some(ts) = t1 {
                    xmul_acc += ts.elapsed().as_secs_f64();
                }
                let t2 = if self.stage_timing {
                    Some(Instant::now())
                } else {
                    None
                };
                row_major_block_t_mul_vec_accum_f32(
                    blk_slice, cur_rows, n, tmp_slice, output, self.pool,
                );
                if let Some(ts) = t2 {
                    xtmul_acc += ts.elapsed().as_secs_f64();
                }
            }
        }
        if capture_ss {
            *self.sum_ss_global = ss_acc;
            *self.sum_ss_captured = true;
        }
        let mean_u = (output.iter().map(|&v| v as f64).sum::<f64>() / (n as f64)) as f32;
        for i in 0..n {
            output[i] = (output[i] - mean_u) + self.lambda_use * input[i];
        }
        if self.stage_timing {
            if let Ok(mut acc) = self.stage_accum.lock() {
                acc.decode_secs += decode_acc;
                acc.xmul_secs += xmul_acc;
                acc.xtmul_secs += xtmul_acc;
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn rrblup_dual_recover_beta_and_test(
    source: &mut RrblupPcgSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    train_idx: &[usize],
    full_train_fast: bool,
    test_idx: &[usize],
    full_test_fast: bool,
    packed_row_indices: Option<&[usize]>,
    block_rows: usize,
    alpha_c: &[f32],
    y_mean: f64,
    need_sum_ss: bool,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(Vec<f32>, f32, Vec<f64>, f64), String> {
    let m = row_mean.len();
    let n_train = train_idx.len();
    let n_test = test_idx.len();
    let row_step = block_rows.max(1).min(m.max(1));
    let inv_n_train = if n_train > 0 {
        1.0_f64 / (n_train as f64)
    } else {
        0.0_f64
    };
    let mut beta = vec![0.0_f32; m];
    let mut pred_test_f32 = vec![0.0_f32; n_test];
    let mut block_train = vec![0.0_f32; row_step.saturating_mul(n_train)];
    let mut block_test = if n_test > 0 {
        vec![0.0_f32; row_step.saturating_mul(n_test)]
    } else {
        Vec::new()
    };
    let mut dot_mean_beta = 0.0_f64;
    let mut sum_ss_global = 0.0_f64;

    for row_start in (0..m).step_by(row_step) {
        let cur_rows = (m - row_start).min(row_step);
        let row_end = row_start + cur_rows;
        let blk_train = &mut block_train[..cur_rows * n_train];
        decode_standardized_block_f32(
            source,
            n_samples,
            row_flip,
            row_mean,
            row_inv_sd,
            train_idx,
            full_train_fast,
            packed_row_indices,
            row_start,
            blk_train,
            code4_lut,
            pool,
        )?;
        if need_sum_ss {
            sum_ss_global +=
                row_centered_sum_squares_f32_to_f64(blk_train, cur_rows, n_train, pool);
        }
        let beta_blk = &mut beta[row_start..row_end];
        row_major_block_mul_vec_f32(blk_train, cur_rows, n_train, alpha_c, beta_blk, pool);
        let blk_dot: f64 = if let Some(tp) = pool.filter(|_| cur_rows * n_train >= 1_048_576) {
            tp.install(|| {
                beta_blk
                    .par_iter()
                    .enumerate()
                    .map(|(r, &bj)| {
                        let row = &blk_train[r * n_train..(r + 1) * n_train];
                        let row_mu = row_major_row_sum_f64(row) * inv_n_train;
                        row_mu * (bj as f64)
                    })
                    .sum::<f64>()
            })
        } else {
            beta_blk
                .iter()
                .enumerate()
                .map(|(r, &bj)| {
                    let row = &blk_train[r * n_train..(r + 1) * n_train];
                    let row_mu = row_major_row_sum_f64(row) * inv_n_train;
                    row_mu * (bj as f64)
                })
                .sum::<f64>()
        };
        dot_mean_beta += blk_dot;

        if n_test > 0 {
            let blk_test = &mut block_test[..cur_rows * n_test];
            decode_standardized_block_f32(
                source,
                n_samples,
                row_flip,
                row_mean,
                row_inv_sd,
                test_idx,
                full_test_fast,
                packed_row_indices,
                row_start,
                blk_test,
                code4_lut,
                pool,
            )?;
            row_major_block_t_mul_vec_accum_f32(
                blk_test,
                cur_rows,
                n_test,
                beta_blk,
                &mut pred_test_f32,
                pool,
            );
        }
    }
    let alpha_use = (y_mean - dot_mean_beta) as f32;
    let pred_test_ret: Vec<f64> = pred_test_f32
        .iter()
        .map(|&v| (v as f64) + (alpha_use as f64))
        .collect();
    Ok((beta, alpha_use, pred_test_ret, sum_ss_global))
}

#[allow(clippy::too_many_arguments)]
fn rrblup_dual_finalize_train_predictions(
    resident_packed: Option<&[u8]>,
    bytes_per_snp: usize,
    prefix: &str,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    train_idx: &[usize],
    full_train_fast: bool,
    train_pred_pick: Option<&[usize]>,
    packed_row_indices: Option<&[usize]>,
    row_step: usize,
    y_vec_f64: &[f64],
    alpha_c: &[f32],
    lambda_use: f32,
    beta: &[f32],
    alpha_use: f32,
    converged: bool,
    compute_trainvar: bool,
    code4_lut: &[[u8; 4]; 256],
    pool_ref: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(Vec<f64>, f64), String> {
    let need_train_pred_all = train_pred_pick.is_none();
    let need_train_pred_subset = train_pred_pick.map(|idx| !idx.is_empty()).unwrap_or(false);
    let need_train_pred_any = need_train_pred_all || need_train_pred_subset;
    let need_train_full = compute_trainvar || need_train_pred_all;

    if converged {
        let lam_f64 = lambda_use as f64;
        let pred_train_f64: Vec<f64> = y_vec_f64
            .iter()
            .zip(alpha_c.iter())
            .map(|(&yi, &ai)| yi - lam_f64 * (ai as f64))
            .collect();
        let pred_train_ret: Vec<f64> = if let Some(local_idx) = train_pred_pick {
            local_idx.iter().map(|&i| pred_train_f64[i]).collect()
        } else {
            pred_train_f64.clone()
        };
        let pve_trainvar = if compute_trainvar {
            let resid: Vec<f64> = y_vec_f64
                .iter()
                .zip(pred_train_f64.iter())
                .map(|(&y, &p)| y - p)
                .collect();
            let var_g = sample_var_f64(&pred_train_f64, pool_ref);
            let var_e = sample_var_f64(&resid, pool_ref);
            let denom = var_g + var_e;
            if denom.is_finite() && denom > 0.0 {
                var_g / denom
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        return Ok((pred_train_ret, pve_trainvar));
    }

    if need_train_full {
        let mut pred_train_source =
            build_rrblup_source(resident_packed, bytes_per_snp, prefix, n_samples, row_step)?;
        let mut pred_train_full = pcg_x_mul_samples(
            &mut pred_train_source,
            n_samples,
            row_flip,
            row_mean,
            row_inv_sd,
            train_idx,
            full_train_fast,
            packed_row_indices,
            row_step,
            beta,
            code4_lut,
            pool_ref,
            None,
        )?;
        for v in &mut pred_train_full {
            *v += alpha_use;
        }
        let pred_train_f64: Vec<f64> = pred_train_full.iter().map(|v| *v as f64).collect();
        let pred_train_ret: Vec<f64> = if let Some(local_idx) = train_pred_pick {
            local_idx.iter().map(|&i| pred_train_f64[i]).collect()
        } else {
            pred_train_f64.clone()
        };
        let pve_trainvar = if compute_trainvar {
            let resid: Vec<f64> = y_vec_f64
                .iter()
                .zip(pred_train_f64.iter())
                .map(|(&y, &p)| y - p)
                .collect();
            let var_g = sample_var_f64(&pred_train_f64, pool_ref);
            let var_e = sample_var_f64(&resid, pool_ref);
            let denom = var_g + var_e;
            if denom.is_finite() && denom > 0.0 {
                var_g / denom
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        return Ok((pred_train_ret, pve_trainvar));
    }

    if need_train_pred_any {
        if let Some(local_idx) = train_pred_pick {
            let train_pred_abs: Vec<usize> = local_idx.iter().map(|&i| train_idx[i]).collect();
            let train_pred_fast = is_identity_indices(&train_pred_abs, n_samples);
            let mut pred_train_sub_source =
                build_rrblup_source(resident_packed, bytes_per_snp, prefix, n_samples, row_step)?;
            let mut pred_train_sub = pcg_x_mul_samples(
                &mut pred_train_sub_source,
                n_samples,
                row_flip,
                row_mean,
                row_inv_sd,
                &train_pred_abs,
                train_pred_fast,
                packed_row_indices,
                row_step,
                beta,
                code4_lut,
                pool_ref,
                None,
            )?;
            for v in &mut pred_train_sub {
                *v += alpha_use;
            }
            let pred_train_ret = pred_train_sub.iter().map(|v| *v as f64).collect();
            return Ok((pred_train_ret, f64::NAN));
        }
    }
    Ok((Vec::new(), f64::NAN))
}

#[allow(clippy::too_many_arguments)]
fn decode_standardized_block_f32(
    source: &mut RrblupPcgSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    row_start: usize,
    out: &mut [f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let n_out = sample_idx.len();
    if n_out == 0 || out.is_empty() {
        return Ok(());
    }
    let cur_rows = out.len().saturating_div(n_out.max(1));
    let row_end = row_start.saturating_add(cur_rows);
    match source {
        RrblupPcgSource::Resident {
            packed_flat,
            bytes_per_snp,
        } => decode_standardized_packed_block_rows_f32(
            packed_flat,
            *bytes_per_snp,
            n_samples,
            row_flip,
            row_mean,
            row_inv_sd,
            sample_idx,
            full_sample_fast,
            packed_row_indices,
            row_start,
            out,
            code4_lut,
            pool,
        ),
        RrblupPcgSource::Windowed {
            matrix,
            bytes_per_snp,
        } => {
            let packed_slice = if let Some(indices) = packed_row_indices {
                let source_rows = &indices[row_start..row_end];
                let mut rel_indices = Vec::with_capacity(cur_rows);
                let packed_slice = matrix.prepare_source_rows(source_rows, &mut rel_indices)?;
                return decode_prepared_additive_block_packed_f32(
                    packed_slice,
                    *bytes_per_snp,
                    n_samples,
                    &row_flip[row_start..row_end],
                    &row_mean[row_start..row_end],
                    &row_inv_sd[row_start..row_end],
                    sample_idx,
                    full_sample_fast,
                    None,
                    Some(rel_indices.as_slice()),
                    0usize,
                    cur_rows,
                    out,
                    pool,
                );
            } else {
                matrix.read_source_range(row_start, row_end)?
            };
            decode_prepared_additive_block_packed_f32(
                packed_slice,
                *bytes_per_snp,
                n_samples,
                &row_flip[row_start..row_end],
                &row_mean[row_start..row_end],
                &row_inv_sd[row_start..row_end],
                sample_idx,
                full_sample_fast,
                None,
                None,
                0usize,
                cur_rows,
                out,
                pool,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_standardized_block_rows_chunked_f32(
    source: &mut RrblupPcgSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    row_block: usize,
    out: &mut [f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let n_out = sample_idx.len();
    if n_out == 0 || out.is_empty() {
        return Ok(());
    }
    let total_rows = out.len().saturating_div(n_out.max(1));
    if total_rows == 0 {
        return Ok(());
    }
    let row_step = row_block.max(1).min(total_rows);
    let mut tick = 0usize;
    for row_start in (0..total_rows).step_by(row_step) {
        let row_end = (row_start + row_step).min(total_rows);
        let out_slice = &mut out[row_start * n_out..row_end * n_out];
        decode_standardized_block_f32(
            source,
            n_samples,
            row_flip,
            row_mean,
            row_inv_sd,
            sample_idx,
            full_sample_fast,
            packed_row_indices,
            row_start,
            out_slice,
            code4_lut,
            pool,
        )?;
        tick += row_end - row_start;
        if tick >= row_step.saturating_mul(64).max(1) {
            check_ctrlc()?;
            tick = 0;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn pcg_x_mul_samples(
    source: &mut RrblupPcgSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    block_rows: usize,
    weights: &[f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
    timing: Option<&mut PcgMatvecTiming>,
) -> Result<Vec<f32>, String> {
    let m = weights.len();
    let n_out = sample_idx.len();
    let mut out = vec![0.0_f32; n_out];
    if m == 0 || n_out == 0 {
        return Ok(out);
    }
    let row_step = block_rows.max(1).min(m.max(1));
    let mut decode_acc = 0.0_f64;
    let mut mul_acc = 0.0_f64;
    let use_double_buffer = matches!(source, RrblupPcgSource::Windowed { .. }) && m > row_step;
    if use_double_buffer {
        let pool_cloned = pool.cloned();
        let mut next_row = 0usize;
        let producer_err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let producer_err_bg = Arc::clone(&producer_err);
        run_double_buffer(
            2usize,
            || DecodedBlockBuffer {
                row_start: 0,
                row_end: 0,
                decode_secs: 0.0,
                block: vec![0.0_f32; row_step * n_out],
            },
            |buf| {
                if next_row >= m {
                    return false;
                }
                let st = next_row;
                let ed = (st + row_step).min(m);
                let cur_rows = ed - st;
                buf.row_start = st;
                buf.row_end = ed;
                let t_decode = Instant::now();
                let blk_slice = &mut buf.block[..cur_rows * n_out];
                if let Err(err) = decode_standardized_block_f32(
                    source,
                    n_samples,
                    row_flip,
                    row_mean,
                    row_inv_sd,
                    sample_idx,
                    full_sample_fast,
                    packed_row_indices,
                    st,
                    blk_slice,
                    code4_lut,
                    pool_cloned.as_ref(),
                ) {
                    if let Ok(mut slot) = producer_err_bg.lock() {
                        *slot = Some(err);
                    }
                    buf.row_end = buf.row_start;
                    next_row = m;
                    return false;
                }
                buf.decode_secs = t_decode.elapsed().as_secs_f64();
                next_row = ed;
                next_row < m
            },
            |buf| {
                if let Ok(slot) = producer_err.lock() {
                    if let Some(err) = slot.as_ref() {
                        return Err(err.clone());
                    }
                }
                let st = buf.row_start;
                let ed = buf.row_end;
                let cur_rows = ed - st;
                decode_acc += buf.decode_secs;
                let t_mul = Instant::now();
                row_major_block_t_mul_vec_accum_f32(
                    &buf.block[..cur_rows * n_out],
                    cur_rows,
                    n_out,
                    &weights[st..ed],
                    &mut out,
                    pool,
                );
                mul_acc += t_mul.elapsed().as_secs_f64();
                check_ctrlc()?;
                Ok::<(), String>(())
            },
        )?;
        let final_err = producer_err.lock().ok().and_then(|slot| slot.clone());
        if let Some(err) = final_err {
            return Err(err);
        }
    } else {
        let mut block = vec![0.0_f32; row_step * n_out];
        let mut tick = 0usize;
        for st in (0..m).step_by(row_step) {
            let ed = (st + row_step).min(m);
            let cur_rows = ed - st;
            let blk_slice = &mut block[..cur_rows * n_out];
            let t_decode = Instant::now();
            decode_standardized_block_f32(
                source,
                n_samples,
                row_flip,
                row_mean,
                row_inv_sd,
                sample_idx,
                full_sample_fast,
                packed_row_indices,
                st,
                blk_slice,
                code4_lut,
                pool,
            )?;
            decode_acc += t_decode.elapsed().as_secs_f64();
            let t_mul = Instant::now();
            row_major_block_t_mul_vec_accum_f32(
                blk_slice,
                cur_rows,
                n_out,
                &weights[st..ed],
                &mut out,
                pool,
            );
            mul_acc += t_mul.elapsed().as_secs_f64();
            tick += cur_rows;
            if tick >= row_step.saturating_mul(64).max(1) {
                check_ctrlc()?;
                tick = 0;
            }
        }
    }
    if let Some(t) = timing {
        t.decode_secs += decode_acc;
        t.mul_secs += mul_acc;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn pcg_xt_mul_rows(
    source: &mut RrblupPcgSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    block_rows: usize,
    u: &[f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
    timing: Option<&mut PcgMatvecTiming>,
) -> Result<Vec<f32>, String> {
    let m = row_flip.len();
    let n_out = sample_idx.len();
    if u.len() != n_out {
        return Err(format!(
            "pcg_xt_mul_rows length mismatch: len(u)={} != n_samples_subset={n_out}",
            u.len()
        ));
    }
    let mut out = vec![0.0_f32; m];
    if m == 0 || n_out == 0 {
        return Ok(out);
    }
    let row_step = block_rows.max(1).min(m.max(1));
    let mut decode_acc = 0.0_f64;
    let mut mul_acc = 0.0_f64;
    let use_double_buffer = matches!(source, RrblupPcgSource::Windowed { .. }) && m > row_step;
    if use_double_buffer {
        let pool_cloned = pool.cloned();
        let mut next_row = 0usize;
        let producer_err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let producer_err_bg = Arc::clone(&producer_err);
        run_double_buffer(
            2usize,
            || DecodedBlockBuffer {
                row_start: 0,
                row_end: 0,
                decode_secs: 0.0,
                block: vec![0.0_f32; row_step * n_out],
            },
            |buf| {
                if next_row >= m {
                    return false;
                }
                let st = next_row;
                let ed = (st + row_step).min(m);
                let cur_rows = ed - st;
                buf.row_start = st;
                buf.row_end = ed;
                let t_decode = Instant::now();
                let blk_slice = &mut buf.block[..cur_rows * n_out];
                if let Err(err) = decode_standardized_block_f32(
                    source,
                    n_samples,
                    row_flip,
                    row_mean,
                    row_inv_sd,
                    sample_idx,
                    full_sample_fast,
                    packed_row_indices,
                    st,
                    blk_slice,
                    code4_lut,
                    pool_cloned.as_ref(),
                ) {
                    if let Ok(mut slot) = producer_err_bg.lock() {
                        *slot = Some(err);
                    }
                    buf.row_end = buf.row_start;
                    next_row = m;
                    return false;
                }
                buf.decode_secs = t_decode.elapsed().as_secs_f64();
                next_row = ed;
                next_row < m
            },
            |buf| {
                if let Ok(slot) = producer_err.lock() {
                    if let Some(err) = slot.as_ref() {
                        return Err(err.clone());
                    }
                }
                let st = buf.row_start;
                let ed = buf.row_end;
                let cur_rows = ed - st;
                decode_acc += buf.decode_secs;
                let mut tmp = vec![0.0_f32; cur_rows];
                let t_mul = Instant::now();
                row_major_block_mul_vec_f32(
                    &buf.block[..cur_rows * n_out],
                    cur_rows,
                    n_out,
                    u,
                    &mut tmp,
                    pool,
                );
                mul_acc += t_mul.elapsed().as_secs_f64();
                out[st..ed].copy_from_slice(&tmp);
                check_ctrlc()?;
                Ok::<(), String>(())
            },
        )?;
        let final_err = producer_err.lock().ok().and_then(|slot| slot.clone());
        if let Some(err) = final_err {
            return Err(err);
        }
    } else {
        let mut block = vec![0.0_f32; row_step * n_out];
        let mut tmp = vec![0.0_f32; row_step];
        let mut tick = 0usize;
        for st in (0..m).step_by(row_step) {
            let ed = (st + row_step).min(m);
            let cur_rows = ed - st;
            let blk_slice = &mut block[..cur_rows * n_out];
            let t_decode = Instant::now();
            decode_standardized_block_f32(
                source,
                n_samples,
                row_flip,
                row_mean,
                row_inv_sd,
                sample_idx,
                full_sample_fast,
                packed_row_indices,
                st,
                blk_slice,
                code4_lut,
                pool,
            )?;
            decode_acc += t_decode.elapsed().as_secs_f64();
            let out_blk = &mut tmp[..cur_rows];
            let t_mul = Instant::now();
            row_major_block_mul_vec_f32(blk_slice, cur_rows, n_out, u, out_blk, pool);
            mul_acc += t_mul.elapsed().as_secs_f64();
            out[st..ed].copy_from_slice(out_blk);
            tick += cur_rows;
            if tick >= row_step.saturating_mul(64).max(1) {
                check_ctrlc()?;
                tick = 0;
            }
        }
    }
    if let Some(t) = timing {
        t.decode_secs += decode_acc;
        t.mul_secs += mul_acc;
    }
    Ok(out)
}

#[inline]
fn sample_var_f64(v: &[f64], pool: Option<&Arc<rayon::ThreadPool>>) -> f64 {
    let n = v.len();
    if n <= 1 {
        return 0.0;
    }
    let use_parallel = pool.is_some() && n >= RRBLUP_PAR_VEC_THRESHOLD;
    let mean = if use_parallel {
        let run = || v.par_iter().copied().sum::<f64>() / (n as f64);
        if let Some(tp) = pool {
            tp.install(run)
        } else {
            run()
        }
    } else {
        v.iter().sum::<f64>() / (n as f64)
    };
    let ss = if use_parallel {
        let run = || {
            v.par_iter()
                .map(|&x| {
                    let d = x - mean;
                    d * d
                })
                .sum::<f64>()
        };
        if let Some(tp) = pool {
            tp.install(run)
        } else {
            run()
        }
    } else {
        let mut ss_local = 0.0_f64;
        for &x in v {
            let d = x - mean;
            ss_local += d * d;
        }
        ss_local
    };
    ss / ((n - 1) as f64)
}

#[inline]
fn parse_nonnegative_index_vec_i64(values: &[i64], name: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::with_capacity(values.len());
    for (i, &raw) in values.iter().enumerate() {
        if raw < 0 {
            return Err(format!("{name}[{i}] must be >= 0, got {raw}"));
        }
        out.push(raw as usize);
    }
    Ok(out)
}

#[inline]
fn is_identity_row_indices(indices: &[usize]) -> bool {
    indices
        .iter()
        .enumerate()
        .all(|(dst_row, &src_row)| dst_row == src_row)
}

struct RrblupExactSnpPreparedInner {
    n_samples: usize,
    n_train: usize,
    n_eff: usize,
    m: usize,
    rank: usize,
    m_effective: usize,
    sample_block: usize,
    stream_row_block: usize,
    train_idx: Vec<usize>,
    packed_row_indices: Option<Vec<usize>>,
    row_flip: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    train_row_mean: Vec<f32>,
    eigvals: Vec<f64>,
    evecs: Vec<f64>, // row-major (m, rank)
    eig_backend: String,
}

#[pyclass]
pub struct RrblupExactSnpCache {
    inner: Arc<RrblupExactSnpPreparedInner>,
}

#[pymethods]
impl RrblupExactSnpCache {
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples
    }

    #[getter]
    fn n_train(&self) -> usize {
        self.inner.n_train
    }

    #[getter]
    fn m(&self) -> usize {
        self.inner.m
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }

    #[getter]
    fn m_effective(&self) -> usize {
        self.inner.m_effective
    }

    #[getter]
    fn eig_backend(&self) -> String {
        self.inner.eig_backend.clone()
    }
}

#[pyclass(name = "RrblupCenteredExactSnpCache")]
pub struct RrblupCenteredExactSnpCache {
    inner: Arc<RrblupExactSnpPreparedInner>,
}

#[pymethods]
impl RrblupCenteredExactSnpCache {
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples
    }

    #[getter]
    fn n_train(&self) -> usize {
        self.inner.n_train
    }

    #[getter]
    fn m(&self) -> usize {
        self.inner.m
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }

    #[getter]
    fn m_effective(&self) -> usize {
        self.inner.m_effective
    }

    #[getter]
    fn eig_backend(&self) -> String {
        self.inner.eig_backend.clone()
    }
}

const RRBLUP_EXACT_SUBTRACTION_FALLBACK_DIRECT_MSG: &str =
    "__RRBLUP_EXACT_SUBTRACTION_FALLBACK_DIRECT__";

#[inline]
fn rrblup_source_fingerprint_from_handle<'a>(
    source: &'a RrblupPcgSource<'_>,
    fallback: &'a str,
) -> Result<&'a str, String> {
    match source {
        RrblupPcgSource::Windowed { matrix, .. } => matrix.source_fingerprint().ok_or_else(|| {
            "rrBLUP exact CV subtraction requires a verified genotype file identity fingerprint."
                .to_string()
        }),
        RrblupPcgSource::Resident { .. } => {
            let trimmed = fallback.trim();
            if trimmed.is_empty() {
                Err(
                    "rrBLUP exact CV subtraction requires a non-empty resident genotype source fingerprint."
                        .to_string(),
                )
            } else {
                Ok(trimmed)
            }
        }
    }
}

struct RrblupExactSnpRawFullInner {
    source_fingerprint: String,
    n_samples: usize,
    n_full: usize,
    m: usize,
    m_effective: usize,
    scale_mode: RrblupScaleMode,
    full_train_idx: Vec<usize>,
    packed_row_indices: Option<Vec<usize>>,
    row_flip: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    a_full_raw: Vec<f64>,
    row_sum_full: Vec<f64>,
    raw_full_trace: f64,
}

#[pyclass(name = "RrblupExactSnpRawCache")]
pub struct RrblupExactSnpRawCache {
    inner: Arc<RrblupExactSnpRawFullInner>,
}

#[pymethods]
impl RrblupExactSnpRawCache {
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples
    }

    #[getter]
    fn n_full(&self) -> usize {
        self.inner.n_full
    }

    #[getter]
    fn m(&self) -> usize {
        self.inner.m
    }

    #[getter]
    fn m_effective(&self) -> usize {
        self.inner.m_effective
    }
}

#[inline]
fn row_major_block_row_sum_and_cast_f64(
    block_f32: &[f32],
    rows: usize,
    cols: usize,
    row_sum_accum: &mut [f64],
    row_sum_sign: f64,
    block_f64: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block_f32.len(), rows.saturating_mul(cols));
    debug_assert_eq!(row_sum_accum.len(), rows);
    debug_assert_eq!(block_f64.len(), rows.saturating_mul(cols));
    let prefer_parallel = prefer_parallel_block_vec(rows, cols, pool)
        && rust_sgemm_prefers_rayon_rowmajor_f32_kernel();
    if prefer_parallel {
        let mut row_sum_local = vec![0.0_f64; rows];
        let mut run = || {
            row_sum_local
                .par_iter_mut()
                .zip(block_f64.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(r, (sum_slot, dst_row))| {
                    let src_row = &block_f32[r * cols..(r + 1) * cols];
                    let mut acc = 0.0_f64;
                    for c in 0..cols {
                        let v = src_row[c] as f64;
                        dst_row[c] = v;
                        acc += v;
                    }
                    *sum_slot = acc;
                });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        for r in 0..rows {
            row_sum_accum[r] += row_sum_sign * row_sum_local[r];
        }
        return;
    }
    for r in 0..rows {
        let src_row = &block_f32[r * cols..(r + 1) * cols];
        let dst_row = &mut block_f64[r * cols..(r + 1) * cols];
        let mut acc = 0.0_f64;
        for c in 0..cols {
            let v = src_row[c] as f64;
            dst_row[c] = v;
            acc += v;
        }
        row_sum_accum[r] += row_sum_sign * acc;
    }
}

#[inline]
fn symmetrize_upper_minus_rank1_in_place(
    a_upper: &mut [f64],
    n: usize,
    row_sum: &[f64],
    scale: f64,
) {
    debug_assert_eq!(a_upper.len(), n.saturating_mul(n));
    debug_assert_eq!(row_sum.len(), n);
    for i in 0..n {
        let rs_i = row_sum[i];
        for j in i..n {
            let v = a_upper[i * n + j] - scale * rs_i * row_sum[j];
            a_upper[i * n + j] = v;
            a_upper[j * n + i] = v;
        }
    }
}

#[inline]
fn rrblup_exact_reml_cost_from_spectrum(
    lambda: f64,
    eigvals: &[f64],
    y_proj: &[f64],
    y_resid_ss: f64,
    n_eff: usize,
) -> f64 {
    if !(lambda.is_finite() && lambda > 0.0) {
        return f64::INFINITY;
    }
    let r = eigvals.len();
    if r != y_proj.len() || n_eff == 0 || n_eff < r {
        return f64::INFINITY;
    }
    let mut quad = 0.0_f64;
    let mut log_det = 0.0_f64;
    let mut y_proj_ss = 0.0_f64;
    for k in 0..r {
        let s = eigvals[k];
        let yk = y_proj[k];
        if !(s.is_finite() && s >= 0.0 && yk.is_finite()) {
            return f64::INFINITY;
        }
        let vk = s + lambda;
        if !(vk.is_finite() && vk > 0.0) {
            return f64::INFINITY;
        }
        quad += (yk * yk) / vk;
        log_det += vk.ln();
        y_proj_ss += yk * yk;
    }
    let null_df = n_eff.saturating_sub(r);
    let null_ss = (y_resid_ss - y_proj_ss).max(0.0_f64);
    if null_df > 0 {
        quad += null_ss / lambda;
        log_det += (null_df as f64) * lambda.ln();
    }
    if !(quad.is_finite() && quad > 0.0 && log_det.is_finite()) {
        return f64::INFINITY;
    }
    let n_eff_f = n_eff as f64;
    0.5_f64 * (n_eff_f * quad.ln() + log_det)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_rrblup_exact_gram_from_source(
    source: &mut RrblupPcgSource<'_>,
    n_samples: usize,
    sample_idx: &[usize],
    row_flip_keep: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_block_use: usize,
    stream_row_block: usize,
    a_star: &mut [f64],
    row_sum: &mut [f64],
    dsyrk_alpha: f64,
    row_sum_sign: f64,
    threads: usize,
    blas_threads: usize,
    stage_timing: bool,
) -> Result<(f64, f64, f64), String> {
    let n_subset = sample_idx.len();
    if n_subset == 0 {
        return Ok((0.0_f64, 0.0_f64, 0.0_f64));
    }
    let eff_m = row_mean.len();
    let blk_cap = sample_block_use.max(1).min(n_subset);
    let pool_owned = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let pool_ref = pool_owned.as_ref();
    let code4_lut = packed_byte_lut().code4;
    let mut block_f32 = vec![0.0_f32; eff_m * blk_cap];
    let mut block_f64 = vec![0.0_f64; eff_m * blk_cap];
    let mut build_a_decode_secs = 0.0_f64;
    let mut build_a_cast_secs = 0.0_f64;
    let mut build_a_rankk_secs = 0.0_f64;
    let blas_threads_effective = if blas_threads > 0 {
        blas_threads.max(1)
    } else {
        1usize
    };
    let _blas_guard = OpenBlasThreadGuard::enter(blas_threads_effective);
    let use_chunked_stream_decode =
        matches!(source, RrblupPcgSource::Windowed { .. }) && stream_row_block < eff_m;

    for cst in (0..n_subset).step_by(blk_cap) {
        let ced = (cst + blk_cap).min(n_subset);
        let cur_n = ced - cst;
        let blk_slice_f32 = &mut block_f32[..eff_m * cur_n];
        let t_decode = if stage_timing {
            Some(Instant::now())
        } else {
            None
        };
        if use_chunked_stream_decode {
            decode_standardized_block_rows_chunked_f32(
                source,
                n_samples,
                row_flip_keep,
                row_mean,
                row_inv_sd,
                &sample_idx[cst..ced],
                is_identity_indices(&sample_idx[cst..ced], n_samples),
                packed_row_indices,
                stream_row_block,
                blk_slice_f32,
                &code4_lut,
                pool_ref,
            )?;
        } else {
            decode_standardized_block_f32(
                source,
                n_samples,
                row_flip_keep,
                row_mean,
                row_inv_sd,
                &sample_idx[cst..ced],
                is_identity_indices(&sample_idx[cst..ced], n_samples),
                packed_row_indices,
                0usize,
                blk_slice_f32,
                &code4_lut,
                pool_ref,
            )?;
        }
        if let Some(t0) = t_decode {
            build_a_decode_secs += t0.elapsed().as_secs_f64();
        }
        let blk_slice_f64 = &mut block_f64[..eff_m * cur_n];
        let t_cast = if stage_timing {
            Some(Instant::now())
        } else {
            None
        };
        row_major_block_row_sum_and_cast_f64(
            blk_slice_f32,
            eff_m,
            cur_n,
            row_sum,
            row_sum_sign,
            blk_slice_f64,
            pool_ref,
        );
        if let Some(t0) = t_cast {
            build_a_cast_secs += t0.elapsed().as_secs_f64();
        }
        let t_rankk = if stage_timing {
            Some(Instant::now())
        } else {
            None
        };
        unsafe {
            cblas_dsyrk_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_UPPER,
                CBLAS_NO_TRANS,
                eff_m as CblasInt,
                cur_n as CblasInt,
                dsyrk_alpha,
                blk_slice_f64.as_ptr(),
                cur_n as CblasInt,
                1.0_f64,
                a_star.as_mut_ptr(),
                eff_m as CblasInt,
            );
        }
        if let Some(t0) = t_rankk {
            build_a_rankk_secs += t0.elapsed().as_secs_f64();
        }
        check_ctrlc()?;
    }
    Ok((build_a_decode_secs, build_a_cast_secs, build_a_rankk_secs))
}

#[allow(clippy::too_many_arguments)]
fn finalize_rrblup_exact_snp_cache_from_gram(
    mut a_star: Vec<f64>,
    row_sum: Vec<f64>,
    n_samples: usize,
    train_idx: Vec<usize>,
    row_flip_keep: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    packed_row_indices: Option<Vec<usize>>,
    sample_block_use: usize,
    stream_row_block: usize,
    m_effective: usize,
    blas_threads: usize,
    subtraction_trace_scale: Option<f64>,
    stage_timing: bool,
    prepare_t0: Option<Instant>,
    build_a_decode_secs: f64,
    build_a_cast_secs: f64,
    build_a_rankk_secs: f64,
) -> Result<RrblupExactSnpPreparedInner, String> {
    let n_train = train_idx.len();
    let eff_m = row_mean.len();
    let blas_threads_effective = if blas_threads > 0 {
        blas_threads.max(1)
    } else {
        1usize
    };
    let _blas_guard = OpenBlasThreadGuard::enter(blas_threads_effective);
    if stage_timing {
        eprintln!(
            "rrBLUP-EXACT prepare checkpoint stage=post_build_A {}",
            rrblup_exact_memory_snapshot(),
        );
    }
    let t_center = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let center_scale = 1.0_f64 / (n_train as f64);
    let train_row_mean: Vec<f32> = row_sum
        .iter()
        .map(|v| (*v / (n_train as f64)) as f32)
        .collect();
    symmetrize_upper_minus_rank1_in_place(&mut a_star, eff_m, &row_sum, center_scale);
    drop(row_sum);
    let center_secs = t_center
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    let t_eigh = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let (evals_all, mut evecs, eig_backend) = symmetric_eigh_f64_row_major(&a_star, eff_m)?;
    drop(a_star);
    if stage_timing {
        eprintln!(
            "rrBLUP-EXACT prepare checkpoint stage=post_eigh backend={} {}",
            eig_backend,
            rrblup_exact_memory_snapshot(),
        );
    }
    let eigh_secs = t_eigh
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    let max_eval = evals_all.last().copied().unwrap_or(0.0_f64).max(0.0_f64);
    let tol = f64::EPSILON * max_eval.max(1.0_f64) * (eff_m.max(1) as f64);
    if let Some(trace_scale) = subtraction_trace_scale {
        let cancel_tol = 64.0_f64 * f64::EPSILON * trace_scale.max(1.0_f64) * (eff_m.max(1) as f64);
        if max_eval <= cancel_tol || evals_all.iter().any(|&v| v > tol && v <= cancel_tol) {
            return Err(RRBLUP_EXACT_SUBTRACTION_FALLBACK_DIRECT_MSG.to_string());
        }
    }
    let n_eff = n_train.saturating_sub(1);
    let mut keep_start = evals_all.iter().position(|&v| v > tol).ok_or_else(|| {
        "rrblup_exact_snp_packed found no positive spectrum after centering.".to_string()
    })?;
    let rank_pos = eff_m.saturating_sub(keep_start);
    if rank_pos > n_eff {
        keep_start = eff_m.saturating_sub(n_eff);
    }
    let eigvals = evals_all[keep_start..].to_vec();
    let rank = eigvals.len();
    if n_eff == 0 || n_eff < rank {
        return Err(format!(
            "rrblup_exact_snp_packed invalid effective df: n_eff={n_eff}, rank={rank}"
        ));
    }
    let t_basis = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    if rank < eff_m || keep_start > 0 {
        for i in 0..eff_m {
            let src_start = i * eff_m + keep_start;
            let src_end = i * eff_m + eff_m;
            let dst_start = i * rank;
            evecs.copy_within(src_start..src_end, dst_start);
        }
        evecs.truncate(eff_m * rank);
    }
    let basis_secs = t_basis
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    if let Some(t0) = prepare_t0 {
        let total_secs = t0.elapsed().as_secs_f64().max(1e-12_f64);
        let build_a_secs =
            build_a_decode_secs + build_a_cast_secs + build_a_rankk_secs + center_secs;
        let known_secs = build_a_secs + eigh_secs + basis_secs;
        let other_secs = (total_secs - known_secs).max(0.0_f64);
        let pct = |x: f64| -> f64 { (x * 100.0) / total_secs };
        eprintln!(
            "rrBLUP-EXACT prepare timing n_train={} m={} rank={} build_A={:.3}s ({:.1}%) [decode={:.3}s cast={:.3}s syrk={:.3}s center={:.3}s] eigh={:.3}s ({:.1}%) basis={:.3}s ({:.1}%) other={:.3}s ({:.1}%) total={:.3}s backend={}",
            n_train,
            eff_m,
            rank,
            build_a_secs,
            pct(build_a_secs),
            build_a_decode_secs,
            build_a_cast_secs,
            build_a_rankk_secs,
            center_secs,
            eigh_secs,
            pct(eigh_secs),
            basis_secs,
            pct(basis_secs),
            other_secs,
            pct(other_secs),
            total_secs,
            eig_backend,
        );
    }

    Ok(RrblupExactSnpPreparedInner {
        n_samples,
        n_train,
        n_eff,
        m: eff_m,
        rank,
        m_effective,
        sample_block: sample_block_use,
        stream_row_block,
        train_idx,
        packed_row_indices,
        row_flip: row_flip_keep,
        row_mean,
        row_inv_sd,
        train_row_mean,
        eigvals,
        evecs,
        eig_backend: eig_backend.to_string(),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_rrblup_exact_snp_cache_from_source(
    mut source: RrblupPcgSource<'_>,
    n_samples: usize,
    train_idx: Vec<usize>,
    row_flip_keep: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    packed_row_indices: Option<Vec<usize>>,
    sample_block_use: usize,
    stream_row_block: usize,
    m_effective: usize,
    threads: usize,
    blas_threads: usize,
) -> Result<RrblupExactSnpPreparedInner, String> {
    let stage_timing = env_truthy("JX_GS_EXACT_STAGE_TIMING") || env_truthy("JX_GS_DEBUG_STAGE");
    let prepare_t0 = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let n_train = train_idx.len();
    if n_train <= 1 {
        return Err("rrBLUP exact SNP cache requires at least two training samples.".to_string());
    }
    let eff_m = row_mean.len();
    if eff_m == 0 {
        return Err("rrBLUP exact SNP cache received zero active markers.".to_string());
    }
    let source_backend = if matches!(source, RrblupPcgSource::Resident { .. }) {
        "resident"
    } else {
        "windowed"
    };
    let window_target_bytes = if source_backend == "windowed" {
        stream_row_block
            .max(1)
            .saturating_mul(n_samples.div_ceil(4).max(1))
            .saturating_mul(2)
    } else {
        0usize
    };
    let mut a_star = vec![0.0_f64; eff_m * eff_m];
    let mut row_sum = vec![0.0_f64; eff_m];
    if stage_timing {
        eprintln!(
            "rrBLUP-EXACT prepare workspace n_train={} m={} sample_block={} stream_row_block={} backend={} alloc[a_star={} row_sum={} window_target={}] {}",
            n_train,
            eff_m,
            sample_block_use,
            stream_row_block,
            source_backend,
            format_bytes((a_star.len() * std::mem::size_of::<f64>()) as u64),
            format_bytes((row_sum.len() * std::mem::size_of::<f64>()) as u64),
            format_bytes(window_target_bytes as u64),
            rrblup_exact_memory_snapshot(),
        );
    }
    let (build_a_decode_secs, build_a_cast_secs, build_a_rankk_secs) =
        accumulate_rrblup_exact_gram_from_source(
            &mut source,
            n_samples,
            &train_idx,
            row_flip_keep.as_slice(),
            row_mean.as_slice(),
            row_inv_sd.as_slice(),
            packed_row_indices.as_deref(),
            sample_block_use,
            stream_row_block,
            &mut a_star,
            &mut row_sum,
            1.0_f64,
            1.0_f64,
            threads,
            blas_threads,
            stage_timing,
        )?;
    drop(source);
    finalize_rrblup_exact_snp_cache_from_gram(
        a_star,
        row_sum,
        n_samples,
        train_idx,
        row_flip_keep,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        sample_block_use,
        stream_row_block,
        m_effective,
        blas_threads,
        None,
        stage_timing,
        prepare_t0,
        build_a_decode_secs,
        build_a_cast_secs,
        build_a_rankk_secs,
    )
}

fn compute_val_complement_indices(
    full_idx: &[usize],
    train_idx: &[usize],
) -> Result<Vec<usize>, String> {
    if train_idx.len() > full_idx.len() {
        return Err(format!(
            "rrBLUP exact CV subtraction requires train_idx (len={}) to be a subset of full_train_idx (len={}).",
            train_idx.len(),
            full_idx.len()
        ));
    }
    let mut remaining: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(train_idx.len());
    for &idx in train_idx {
        *remaining.entry(idx).or_insert(0usize) += 1;
    }
    let mut val_idx = Vec::with_capacity(full_idx.len().saturating_sub(train_idx.len()));
    for &idx in full_idx {
        if let Some(cnt) = remaining.get_mut(&idx) {
            if *cnt > 0 {
                *cnt -= 1;
                if *cnt == 0 {
                    remaining.remove(&idx);
                }
                continue;
            }
        }
        val_idx.push(idx);
    }
    if !remaining.is_empty() {
        return Err(
            "rrBLUP exact CV subtraction found train_idx sample(s) absent from full_train_idx."
                .to_string(),
        );
    }
    Ok(val_idx)
}

#[allow(clippy::too_many_arguments)]
fn build_rrblup_exact_snp_raw_full_from_source(
    mut source: RrblupPcgSource<'_>,
    source_fingerprint: String,
    n_samples: usize,
    full_train_idx: Vec<usize>,
    row_flip_keep: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    packed_row_indices: Option<Vec<usize>>,
    sample_block_use: usize,
    stream_row_block: usize,
    m_effective: usize,
    scale_mode: RrblupScaleMode,
    threads: usize,
    blas_threads: usize,
) -> Result<RrblupExactSnpRawFullInner, String> {
    let stage_timing = env_truthy("JX_GS_EXACT_STAGE_TIMING") || env_truthy("JX_GS_DEBUG_STAGE");
    let t0 = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let n_full = full_train_idx.len();
    if n_full <= 1 {
        return Err(
            "rrBLUP exact SNP raw full cache requires at least two training samples.".to_string(),
        );
    }
    let eff_m = row_mean.len();
    if eff_m == 0 {
        return Err("rrBLUP exact SNP raw full cache received zero active markers.".to_string());
    }
    let mut a_full_raw = vec![0.0_f64; eff_m * eff_m];
    let mut row_sum_full = vec![0.0_f64; eff_m];
    let resolved_fingerprint =
        rrblup_source_fingerprint_from_handle(&source, source_fingerprint.as_str())?.to_string();
    let (dec_s, cast_s, syrk_s) = accumulate_rrblup_exact_gram_from_source(
        &mut source,
        n_samples,
        &full_train_idx,
        row_flip_keep.as_slice(),
        row_mean.as_slice(),
        row_inv_sd.as_slice(),
        packed_row_indices.as_deref(),
        sample_block_use,
        stream_row_block,
        &mut a_full_raw,
        &mut row_sum_full,
        1.0_f64,
        1.0_f64,
        threads,
        blas_threads,
        stage_timing,
    )?;
    drop(source);
    let raw_full_trace: f64 = (0..eff_m)
        .map(|i| a_full_raw[i * eff_m + i].max(0.0_f64))
        .sum();
    if let Some(start) = t0 {
        eprintln!(
            "rrBLUP-EXACT raw_full timing n_full={} m={} decode={:.3}s cast={:.3}s syrk={:.3}s total={:.3}s",
            n_full,
            eff_m,
            dec_s,
            cast_s,
            syrk_s,
            start.elapsed().as_secs_f64(),
        );
    }
    Ok(RrblupExactSnpRawFullInner {
        source_fingerprint: resolved_fingerprint,
        n_samples,
        n_full,
        m: eff_m,
        m_effective,
        scale_mode,
        full_train_idx,
        packed_row_indices,
        row_flip: row_flip_keep,
        row_mean,
        row_inv_sd,
        a_full_raw,
        row_sum_full,
        raw_full_trace,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_rrblup_exact_snp_cache_from_raw_full<'a, F>(
    raw_full: &RrblupExactSnpRawFullInner,
    source_fingerprint: &str,
    build_source: F,
    n_samples: usize,
    train_idx: Vec<usize>,
    row_flip_keep: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    packed_row_indices: Option<Vec<usize>>,
    sample_block_use: usize,
    stream_row_block: usize,
    m_effective: usize,
    scale_mode: RrblupScaleMode,
    threads: usize,
    blas_threads: usize,
) -> Result<RrblupExactSnpPreparedInner, String>
where
    F: Fn() -> Result<RrblupPcgSource<'a>, String>,
{
    let stage_timing = env_truthy("JX_GS_EXACT_STAGE_TIMING") || env_truthy("JX_GS_DEBUG_STAGE");
    let prepare_t0 = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let n_train = train_idx.len();
    if n_train <= 1 {
        return Err("rrBLUP exact SNP cache requires at least two training samples.".to_string());
    }
    let eff_m = row_mean.len();
    if eff_m == 0 {
        return Err("rrBLUP exact SNP cache received zero active markers.".to_string());
    }
    if raw_full.scale_mode != scale_mode {
        return Err("rrBLUP exact CV subtraction scale_mode mismatch.".to_string());
    }
    if raw_full.n_samples != n_samples || raw_full.m != eff_m {
        return Err(format!(
            "rrBLUP exact CV subtraction dimension mismatch: raw_full=({}, {}), fold=({}, {}).",
            raw_full.n_samples, raw_full.m, n_samples, eff_m
        ));
    }
    if raw_full.row_flip != row_flip_keep || raw_full.packed_row_indices != packed_row_indices {
        return Err("rrBLUP exact CV subtraction marker metadata mismatch.".to_string());
    }
    if raw_full.row_mean != row_mean || raw_full.row_inv_sd != row_inv_sd {
        return Err(
            "rrBLUP exact CV subtraction requires identical row_mean/row_inv_sd as raw_full."
                .to_string(),
        );
    }
    let val_idx = compute_val_complement_indices(&raw_full.full_train_idx, &train_idx)?;
    let mut source = build_source()?;
    let resolved_fp = rrblup_source_fingerprint_from_handle(&source, source_fingerprint)?;
    if raw_full.source_fingerprint != resolved_fp {
        return Err(format!(
            "rrBLUP exact CV subtraction genotype source mismatch: cached={}, requested={resolved_fp}.",
            raw_full.source_fingerprint
        ));
    }
    let mut a_star = raw_full.a_full_raw.clone();
    let mut row_sum = raw_full.row_sum_full.clone();
    let (build_a_decode_secs, build_a_cast_secs, build_a_rankk_secs) = if val_idx.is_empty() {
        drop(source);
        (0.0_f64, 0.0_f64, 0.0_f64)
    } else {
        let times = accumulate_rrblup_exact_gram_from_source(
            &mut source,
            n_samples,
            &val_idx,
            row_flip_keep.as_slice(),
            row_mean.as_slice(),
            row_inv_sd.as_slice(),
            packed_row_indices.as_deref(),
            sample_block_use,
            stream_row_block,
            &mut a_star,
            &mut row_sum,
            -1.0_f64,
            -1.0_f64,
            threads,
            blas_threads,
            stage_timing,
        )?;
        drop(source);
        times
    };
    if stage_timing {
        eprintln!(
            "rrBLUP-EXACT prepare_from_raw n_full={} n_train={} n_val={} m={}",
            raw_full.n_full,
            n_train,
            val_idx.len(),
            eff_m,
        );
    }
    let subtraction_trace_scale = if val_idx.is_empty() {
        None
    } else {
        Some(raw_full.raw_full_trace)
    };
    match finalize_rrblup_exact_snp_cache_from_gram(
        a_star,
        row_sum,
        n_samples,
        train_idx.clone(),
        row_flip_keep.clone(),
        row_mean.clone(),
        row_inv_sd.clone(),
        packed_row_indices.clone(),
        sample_block_use,
        stream_row_block,
        m_effective,
        blas_threads,
        subtraction_trace_scale,
        stage_timing,
        prepare_t0,
        build_a_decode_secs,
        build_a_cast_secs,
        build_a_rankk_secs,
    ) {
        Ok(cache) => Ok(cache),
        Err(err) if err == RRBLUP_EXACT_SUBTRACTION_FALLBACK_DIRECT_MSG => {
            if stage_timing {
                eprintln!(
                    "rrBLUP-EXACT prepare_from_raw fallback=direct n_train={} m={}",
                    n_train, eff_m,
                );
            }
            let source = build_source()?;
            build_rrblup_exact_snp_cache_from_source(
                source,
                n_samples,
                train_idx,
                row_flip_keep,
                row_mean,
                row_inv_sd,
                packed_row_indices,
                sample_block_use,
                stream_row_block,
                m_effective,
                threads,
                blas_threads,
            )
        }
        Err(err) => Err(err),
    }
}

#[inline]
fn rrblup_exact_predict_block_rows(
    m: usize,
    sample_block: usize,
    n_out: usize,
    stream_row_block: usize,
) -> usize {
    let m_use = m.max(1);
    let n_out_use = n_out.max(1);
    let max_rows = 1024usize.min(stream_row_block.max(1)).min(m_use);
    let budget_cells = m_use.saturating_mul(sample_block.max(1));
    let budget_rows = budget_cells / n_out_use;
    budget_rows.max(1).min(max_rows)
}

#[allow(clippy::too_many_arguments)]
fn fit_rrblup_exact_snp_from_cache_source(
    cache: &RrblupExactSnpPreparedInner,
    resident_packed_flat: Option<&[u8]>,
    prefix: &str,
    bytes_per_snp: usize,
    n_samples: usize,
    y_vec_f64: Vec<f64>,
    test_idx: Vec<usize>,
    train_pred_pick: Option<Vec<usize>>,
    log10_lambda_low: f64,
    log10_lambda_high: f64,
    reml_tol: f64,
    reml_max_iter: usize,
    threads: usize,
    blas_threads: usize,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        Vec<f32>,
        String,
    ),
    String,
> {
    let stage_timing = env_truthy("JX_GS_EXACT_STAGE_TIMING") || env_truthy("JX_GS_DEBUG_STAGE");
    let fit_t0 = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    if n_samples != cache.n_samples {
        return Err(format!(
            "rrBLUP exact SNP cache sample-count mismatch: got {n_samples}, expected {}",
            cache.n_samples
        ));
    }
    if y_vec_f64.len() != cache.n_train {
        return Err(format!(
            "rrBLUP exact SNP cache phenotype length mismatch: got {}, expected {}",
            y_vec_f64.len(),
            cache.n_train
        ));
    }
    if y_vec_f64.iter().any(|v| !v.is_finite()) {
        return Err("y_train contains non-finite values.".to_string());
    }
    let pool_owned = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let pool_ref = pool_owned.as_ref();
    let code4_lut = packed_byte_lut().code4;
    let y_mean = y_vec_f64.iter().sum::<f64>() / (cache.n_train as f64);
    let y_center_f32: Vec<f32> = y_vec_f64.iter().map(|v| (*v - y_mean) as f32).collect();
    let y_center_ss = y_vec_f64
        .iter()
        .map(|v| {
            let d = *v - y_mean;
            d * d
        })
        .sum::<f64>();
    if stage_timing {
        eprintln!(
            "rrBLUP-EXACT fit workspace n_train={} m={} rank={} sample_block={} stream_row_block={} n_test={} train_pred={} backend={} alloc[y_center={} coeff={} y_proj={} beta={} pred_out={}] {}",
            cache.n_train,
            cache.m,
            cache.rank,
            cache.sample_block,
            cache.stream_row_block,
            test_idx.len(),
            train_pred_pick
                .as_ref()
                .map(|idx| idx.len())
                .unwrap_or(cache.train_idx.len()),
            if resident_packed_flat.is_some() {
                "resident"
            } else {
                "windowed"
            },
            format_bytes((y_center_f32.len() * std::mem::size_of::<f32>()) as u64),
            format_bytes((cache.rank * std::mem::size_of::<f64>()) as u64),
            format_bytes((cache.rank * std::mem::size_of::<f64>()) as u64),
            format_bytes((cache.m * std::mem::size_of::<f32>()) as u64),
            format_bytes(
                ((cache.train_idx.len() + test_idx.len()) * std::mem::size_of::<f64>()) as u64
            ),
            rrblup_exact_memory_snapshot(),
        );
    }
    let mut source = build_rrblup_source(
        resident_packed_flat,
        bytes_per_snp,
        prefix,
        n_samples,
        cache.stream_row_block.max(1),
    )?;
    let mut z = vec![0.0_f64; cache.m];
    let row_step = cache.stream_row_block.max(1).min(cache.m.max(1));
    let mut tmp_dot = vec![0.0_f32; row_step];
    let mut block_f32 = vec![0.0_f32; row_step * cache.sample_block.max(1)];
    let mut z_decode_secs = 0.0_f64;
    let mut z_mul_secs = 0.0_f64;
    for cst in (0..cache.n_train).step_by(cache.sample_block.max(1)) {
        let ced = (cst + cache.sample_block).min(cache.n_train);
        let cur_n = ced - cst;
        for rst in (0..cache.m).step_by(row_step) {
            let red = (rst + row_step).min(cache.m);
            let cur_rows = red - rst;
            let blk_slice_f32 = &mut block_f32[..cur_rows * cur_n];
            let t_decode = if stage_timing {
                Some(Instant::now())
            } else {
                None
            };
            decode_standardized_block_f32(
                &mut source,
                n_samples,
                cache.row_flip.as_slice(),
                cache.row_mean.as_slice(),
                cache.row_inv_sd.as_slice(),
                &cache.train_idx[cst..ced],
                is_identity_indices(&cache.train_idx[cst..ced], n_samples),
                cache.packed_row_indices.as_deref(),
                rst,
                blk_slice_f32,
                &code4_lut,
                pool_ref,
            )?;
            if let Some(t0) = t_decode {
                z_decode_secs += t0.elapsed().as_secs_f64();
            }
            let t_mul = if stage_timing {
                Some(Instant::now())
            } else {
                None
            };
            row_major_block_mul_vec_f32(
                blk_slice_f32,
                cur_rows,
                cur_n,
                &y_center_f32[cst..ced],
                &mut tmp_dot[..cur_rows],
                pool_ref,
            );
            if let Some(t0) = t_mul {
                z_mul_secs += t0.elapsed().as_secs_f64();
            }
            for j in 0..cur_rows {
                z[rst + j] += tmp_dot[j] as f64;
            }
        }
        check_ctrlc()?;
    }
    drop(source);
    drop(block_f32);
    drop(tmp_dot);
    drop(y_center_f32);
    if stage_timing {
        eprintln!(
            "rrBLUP-EXACT fit checkpoint stage=post_build_z {}",
            rrblup_exact_memory_snapshot(),
        );
    }

    let blas_threads_effective = if blas_threads > 0 {
        blas_threads.max(1)
    } else {
        1usize
    };
    let _blas_guard = OpenBlasThreadGuard::enter(blas_threads_effective);
    let t_project = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let mut coeff = vec![0.0_f64; cache.rank];
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            cache.rank as CblasInt,
            1 as CblasInt,
            cache.m as CblasInt,
            1.0_f64,
            cache.evecs.as_ptr(),
            cache.rank as CblasInt,
            z.as_ptr(),
            1 as CblasInt,
            0.0_f64,
            coeff.as_mut_ptr(),
            1 as CblasInt,
        );
    }
    drop(z);
    let mut y_proj = vec![0.0_f64; cache.rank];
    for k in 0..cache.rank {
        y_proj[k] = coeff[k] / cache.eigvals[k].sqrt();
    }
    let project_secs = t_project
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    let low = log10_lambda_low.min(log10_lambda_high);
    let high = log10_lambda_low.max(log10_lambda_high);
    let t_reml = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let (best_log10, best_cost) = brent_minimize(
        |x| {
            let lam = 10.0_f64.powf(x);
            rrblup_exact_reml_cost_from_spectrum(
                lam,
                cache.eigvals.as_slice(),
                y_proj.as_slice(),
                y_center_ss,
                cache.n_eff,
            )
        },
        low,
        high,
        reml_tol,
        reml_max_iter,
    );
    let reml_secs = t_reml
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    let lambda_opt = 10.0_f64.powf(best_log10).max(1e-12_f64);
    let t_solve = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let mut quad = 0.0_f64;
    let mut y_proj_ss = 0.0_f64;
    let mut g_center_ss = 0.0_f64;
    let mut w = vec![0.0_f64; cache.rank];
    for k in 0..cache.rank {
        let yk = y_proj[k];
        let denom = cache.eigvals[k] + lambda_opt;
        quad += (yk * yk) / denom;
        y_proj_ss += yk * yk;
        let ck = coeff[k];
        g_center_ss += cache.eigvals[k] * (ck * ck) / (denom * denom);
        w[k] = ck / denom;
    }
    let null_ss = (y_center_ss - y_proj_ss).max(0.0_f64);
    let null_df = cache.n_eff.saturating_sub(cache.rank);
    if null_df > 0 {
        quad += null_ss / lambda_opt;
    }
    let sigma_beta2 = quad / (cache.n_eff as f64);
    let sigma_e2 = lambda_opt * sigma_beta2;

    let mut beta_f64 = vec![0.0_f64; cache.m];
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            cache.m as CblasInt,
            1 as CblasInt,
            cache.rank as CblasInt,
            1.0_f64,
            cache.evecs.as_ptr(),
            cache.rank as CblasInt,
            w.as_ptr(),
            1 as CblasInt,
            0.0_f64,
            beta_f64.as_mut_ptr(),
            1 as CblasInt,
        );
    }
    let solve_secs = t_solve
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    let beta_f32: Vec<f32> = beta_f64.iter().map(|&v| v as f32).collect();
    drop(beta_f64);
    let alpha_use = y_mean
        - cache
            .train_row_mean
            .iter()
            .zip(beta_f32.iter())
            .map(|(mu, beta)| (*mu as f64) * (*beta as f64))
            .sum::<f64>();
    let var_g = if cache.n_train > 1 {
        g_center_ss / ((cache.n_train - 1) as f64)
    } else {
        0.0_f64
    };
    if stage_timing {
        eprintln!(
            "rrBLUP-EXACT fit checkpoint stage=post_solve lambda={:.6e} var_g={:.6e} sigma_e2={:.6e} {}",
            lambda_opt,
            var_g,
            sigma_e2,
            rrblup_exact_memory_snapshot(),
        );
    }
    let denom = var_g + sigma_e2;
    let pve_trainvar = if denom.is_finite() && denom > 0.0_f64 {
        var_g / denom
    } else {
        f64::NAN
    };

    let t_pred_train = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let pred_train_ret = if let Some(local_idx) = &train_pred_pick {
        if local_idx.is_empty() {
            Vec::new()
        } else {
            let train_pred_abs: Vec<usize> =
                local_idx.iter().map(|&i| cache.train_idx[i]).collect();
            let pred_block_rows = rrblup_exact_predict_block_rows(
                cache.m,
                cache.sample_block,
                train_pred_abs.len(),
                cache.stream_row_block,
            );
            let mut pred_source = build_rrblup_source(
                resident_packed_flat,
                bytes_per_snp,
                prefix,
                n_samples,
                pred_block_rows,
            )?;
            let pred_train_subset = pcg_x_mul_samples(
                &mut pred_source,
                n_samples,
                cache.row_flip.as_slice(),
                cache.row_mean.as_slice(),
                cache.row_inv_sd.as_slice(),
                train_pred_abs.as_slice(),
                is_identity_indices(train_pred_abs.as_slice(), n_samples),
                cache.packed_row_indices.as_deref(),
                pred_block_rows,
                beta_f32.as_slice(),
                &code4_lut,
                pool_ref,
                None,
            )?;
            pred_train_subset
                .into_iter()
                .map(|v| (v as f64) + alpha_use)
                .collect()
        }
    } else {
        let pred_block_rows = rrblup_exact_predict_block_rows(
            cache.m,
            cache.sample_block,
            cache.train_idx.len(),
            cache.stream_row_block,
        );
        let mut pred_source = build_rrblup_source(
            resident_packed_flat,
            bytes_per_snp,
            prefix,
            n_samples,
            pred_block_rows,
        )?;
        let pred_train_full = pcg_x_mul_samples(
            &mut pred_source,
            n_samples,
            cache.row_flip.as_slice(),
            cache.row_mean.as_slice(),
            cache.row_inv_sd.as_slice(),
            cache.train_idx.as_slice(),
            is_identity_indices(cache.train_idx.as_slice(), n_samples),
            cache.packed_row_indices.as_deref(),
            pred_block_rows,
            beta_f32.as_slice(),
            &code4_lut,
            pool_ref,
            None,
        )?;
        pred_train_full
            .into_iter()
            .map(|v| (v as f64) + alpha_use)
            .collect()
    };
    let pred_train_secs = t_pred_train
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);

    let mut pred_test_ret = Vec::new();
    let t_pred_test = if stage_timing && !test_idx.is_empty() {
        Some(Instant::now())
    } else {
        None
    };
    if !test_idx.is_empty() {
        let pred_block_rows = rrblup_exact_predict_block_rows(
            cache.m,
            cache.sample_block,
            test_idx.len(),
            cache.stream_row_block,
        );
        let mut pred_test_source = build_rrblup_source(
            resident_packed_flat,
            bytes_per_snp,
            prefix,
            n_samples,
            pred_block_rows,
        )?;
        let pred_test_f32 = pcg_x_mul_samples(
            &mut pred_test_source,
            n_samples,
            cache.row_flip.as_slice(),
            cache.row_mean.as_slice(),
            cache.row_inv_sd.as_slice(),
            test_idx.as_slice(),
            is_identity_indices(test_idx.as_slice(), n_samples),
            cache.packed_row_indices.as_deref(),
            pred_block_rows,
            beta_f32.as_slice(),
            &code4_lut,
            pool_ref,
            None,
        )?;
        pred_test_ret = pred_test_f32
            .iter()
            .map(|v| (*v as f64) + alpha_use)
            .collect();
    }
    let pred_test_secs = t_pred_test
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    if let Some(t0) = fit_t0 {
        let total_secs = t0.elapsed().as_secs_f64().max(1e-12_f64);
        let build_z_secs = z_decode_secs + z_mul_secs;
        let predict_secs = pred_train_secs + pred_test_secs;
        let known_secs = build_z_secs + project_secs + reml_secs + solve_secs + predict_secs;
        let other_secs = (total_secs - known_secs).max(0.0_f64);
        let pct = |x: f64| -> f64 { (x * 100.0) / total_secs };
        eprintln!(
            "rrBLUP-EXACT fit timing n_train={} m={} rank={} build_z={:.3}s ({:.1}%) [decode={:.3}s mul={:.3}s] project={:.3}s ({:.1}%) reml={:.3}s ({:.1}%) solve_beta={:.3}s ({:.1}%) predict={:.3}s ({:.1}%) [train={:.3}s test={:.3}s] other={:.3}s ({:.1}%) total={:.3}s backend={}",
            cache.n_train,
            cache.m,
            cache.rank,
            build_z_secs,
            pct(build_z_secs),
            z_decode_secs,
            z_mul_secs,
            project_secs,
            pct(project_secs),
            reml_secs,
            pct(reml_secs),
            solve_secs,
            pct(solve_secs),
            predict_secs,
            pct(predict_secs),
            pred_train_secs,
            pred_test_secs,
            other_secs,
            pct(other_secs),
            total_secs,
            cache.eig_backend,
        );
    }

    Ok((
        pred_train_ret,
        pred_test_ret,
        pve_trainvar,
        lambda_opt,
        -best_cost,
        var_g,
        sigma_e2,
        alpha_use,
        beta_f32,
        cache.eig_backend.clone(),
    ))
}

struct ParsedRrblupExactBedMetaInputs {
    n_samples: usize,
    eff_m: usize,
    train_idx: Vec<usize>,
    row_flip_keep: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    packed_row_indices: Option<Vec<usize>>,
    m_effective: usize,
    bytes_per_snp: usize,
    sample_block_use: usize,
    stream_row_block_use: usize,
}

#[allow(clippy::too_many_arguments)]
fn parse_rrblup_exact_bed_meta_inputs<'py>(
    source_prefix: &str,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    scale_mode: RrblupScaleMode,
    fn_name: &str,
) -> PyResult<ParsedRrblupExactBedMetaInputs> {
    if str::trim(source_prefix).is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "{fn_name} requires non-empty source_prefix."
        )));
    }
    if !(std_eps.is_finite() && std_eps > 0.0) {
        return Err(PyRuntimeError::new_err(format!(
            "{fn_name} requires finite std_eps > 0."
        )));
    }
    let n_samples = crate::gfcore::read_fam(source_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "{fn_name} found zero samples in source_prefix.fam."
        )));
    }
    let train_idx = parse_index_vec_i64(
        train_sample_indices.as_slice()?,
        n_samples,
        "train_sample_indices",
    )?;
    if train_idx.len() <= 1 {
        return Err(PyRuntimeError::new_err(format!(
            "{fn_name} requires at least two training samples."
        )));
    }
    let row_source_idx =
        parse_nonnegative_index_vec_i64(row_source_indices.as_slice()?, "row_source_indices")
            .map_err(PyRuntimeError::new_err)?;
    let eff_m = row_source_idx.len();
    if eff_m == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "{fn_name} received zero active markers."
        )));
    }
    let maf_keep: Vec<f32> = match crate::stats_common::array_standard_slice(&maf) {
        Ok(s) => s.to_vec(),
        Err(_) => crate::stats_common::copy_array_standard_layout(&maf.as_array()),
    };
    let row_flip_keep: Vec<bool> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => s.to_vec(),
        Err(_) => crate::stats_common::copy_array_standard_layout(&row_flip.as_array()),
    };
    let row_mean_external: Option<Vec<f32>> = if let Some(row_mean_ro) = row_mean {
        Some(
            match crate::stats_common::array_standard_slice(&row_mean_ro) {
                Ok(s) => s.to_vec(),
                Err(_) => crate::stats_common::copy_array_standard_layout(&row_mean_ro.as_array()),
            },
        )
    } else {
        None
    };
    let row_inv_sd_external: Option<Vec<f32>> = if let Some(row_inv_sd_ro) = row_inv_sd {
        Some(
            match crate::stats_common::array_standard_slice(&row_inv_sd_ro) {
                Ok(s) => s.to_vec(),
                Err(_) => {
                    crate::stats_common::copy_array_standard_layout(&row_inv_sd_ro.as_array())
                }
            },
        )
    } else {
        None
    };
    if maf_keep.len() != eff_m {
        return Err(PyRuntimeError::new_err(format!(
            "maf length mismatch: got {}, expected {eff_m}",
            maf_keep.len()
        )));
    }
    if row_flip_keep.len() != eff_m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {eff_m}",
            row_flip_keep.len()
        )));
    }
    let packed_row_indices = if is_identity_row_indices(&row_source_idx) {
        None
    } else {
        Some(row_source_idx)
    };
    let std_eps32 = std_eps.max(1e-12) as f32;
    let (row_mean, row_inv_sd, m_effective) = match scale_mode {
        RrblupScaleMode::Standardized => rrblup_subset_or_validate_stats(
            row_mean_external,
            row_inv_sd_external,
            packed_row_indices.as_deref(),
            eff_m,
            eff_m,
            std_eps32,
            maf_keep.as_slice(),
            None,
            scale_mode,
        ),
        RrblupScaleMode::Centered => rrblupc_subset_or_validate_stats(
            row_mean_external,
            row_inv_sd_external,
            packed_row_indices.as_deref(),
            eff_m,
            eff_m,
            std_eps32,
            maf_keep.as_slice(),
        ),
    }
    .map_err(PyRuntimeError::new_err)?;
    if m_effective == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "{fn_name} found zero effective markers after standardization."
        )));
    }
    let bytes_per_snp = n_samples.div_ceil(4);
    let sample_block_use = sample_block.max(1).min(train_idx.len().max(1));
    let stream_row_block_use = stream_row_block.max(1).min(eff_m.max(1));
    Ok(ParsedRrblupExactBedMetaInputs {
        n_samples,
        eff_m,
        train_idx,
        row_flip_keep,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        m_effective,
        bytes_per_snp,
        sample_block_use,
        stream_row_block_use,
    })
}

#[allow(clippy::too_many_arguments)]
fn rrblup_exact_snp_prepare_bed_from_meta_impl<'py>(
    py: Python<'py>,
    source_prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
    scale_mode: RrblupScaleMode,
) -> PyResult<Arc<RrblupExactSnpPreparedInner>> {
    let parsed = parse_rrblup_exact_bed_meta_inputs(
        source_prefix.as_str(),
        train_sample_indices,
        row_source_indices,
        maf,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_block,
        stream_row_block,
        std_eps,
        scale_mode,
        "rrblup_exact_snp_prepare_bed_from_meta",
    )?;
    let _ = parsed.eff_m;
    let cache_inner = py
        .detach(move || {
            let source = build_rrblup_source(
                None,
                parsed.bytes_per_snp,
                source_prefix.as_str(),
                parsed.n_samples,
                parsed.stream_row_block_use,
            )?;
            build_rrblup_exact_snp_cache_from_source(
                source,
                parsed.n_samples,
                parsed.train_idx,
                parsed.row_flip_keep,
                parsed.row_mean,
                parsed.row_inv_sd,
                parsed.packed_row_indices,
                parsed.sample_block_use,
                parsed.stream_row_block_use,
                parsed.m_effective,
                threads,
                blas_threads,
            )
        })
        .map_err(map_err_string_to_py)?;
    Ok(Arc::new(cache_inner))
}

#[allow(clippy::too_many_arguments)]
fn rrblup_exact_snp_prepare_raw_bed_from_meta_impl<'py>(
    py: Python<'py>,
    source_prefix: String,
    full_train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
    scale_mode: RrblupScaleMode,
) -> PyResult<Arc<RrblupExactSnpRawFullInner>> {
    let parsed = parse_rrblup_exact_bed_meta_inputs(
        source_prefix.as_str(),
        full_train_sample_indices,
        row_source_indices,
        maf,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_block,
        stream_row_block,
        std_eps,
        scale_mode,
        "rrblup_exact_snp_prepare_raw_bed_from_meta",
    )?;
    let _ = parsed.eff_m;
    let raw_inner = py
        .detach(move || {
            let source = build_rrblup_source(
                None,
                parsed.bytes_per_snp,
                source_prefix.as_str(),
                parsed.n_samples,
                parsed.stream_row_block_use,
            )?;
            build_rrblup_exact_snp_raw_full_from_source(
                source,
                String::new(),
                parsed.n_samples,
                parsed.train_idx,
                parsed.row_flip_keep,
                parsed.row_mean,
                parsed.row_inv_sd,
                parsed.packed_row_indices,
                parsed.sample_block_use,
                parsed.stream_row_block_use,
                parsed.m_effective,
                scale_mode,
                threads,
                blas_threads,
            )
        })
        .map_err(map_err_string_to_py)?;
    Ok(Arc::new(raw_inner))
}

#[allow(clippy::too_many_arguments)]
fn rrblup_exact_snp_prepare_bed_from_raw_cache_impl<'py>(
    py: Python<'py>,
    raw_cache: &Arc<RrblupExactSnpRawFullInner>,
    source_prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
    scale_mode: RrblupScaleMode,
) -> PyResult<Arc<RrblupExactSnpPreparedInner>> {
    let parsed = parse_rrblup_exact_bed_meta_inputs(
        source_prefix.as_str(),
        train_sample_indices,
        row_source_indices,
        maf,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_block,
        stream_row_block,
        std_eps,
        scale_mode,
        "rrblup_exact_snp_prepare_bed_from_raw_cache",
    )?;
    let _ = parsed.eff_m;
    let raw_arc = Arc::clone(raw_cache);
    let cache_inner = py
        .detach(move || {
            build_rrblup_exact_snp_cache_from_raw_full(
                raw_arc.as_ref(),
                "",
                || {
                    build_rrblup_source(
                        None,
                        parsed.bytes_per_snp,
                        source_prefix.as_str(),
                        parsed.n_samples,
                        parsed.stream_row_block_use,
                    )
                },
                parsed.n_samples,
                parsed.train_idx,
                parsed.row_flip_keep,
                parsed.row_mean,
                parsed.row_inv_sd,
                parsed.packed_row_indices,
                parsed.sample_block_use,
                parsed.stream_row_block_use,
                parsed.m_effective,
                scale_mode,
                threads,
                blas_threads,
            )
        })
        .map_err(map_err_string_to_py)?;
    Ok(Arc::new(cache_inner))
}

#[pyfunction]
#[pyo3(signature = (
    source_prefix,
    train_sample_indices,
    row_source_indices,
    maf,
    row_flip,
    row_mean=None,
    row_inv_sd=None,
    sample_block=2048,
    stream_row_block=2048,
    std_eps=1e-12_f64,
    threads=0,
    blas_threads=0
))]
pub fn rrblup_exact_snp_prepare_bed_from_meta<'py>(
    py: Python<'py>,
    source_prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
) -> PyResult<RrblupExactSnpCache> {
    let inner = rrblup_exact_snp_prepare_bed_from_meta_impl(
        py,
        source_prefix,
        train_sample_indices,
        row_source_indices,
        maf,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_block,
        stream_row_block,
        std_eps,
        threads,
        blas_threads,
        RrblupScaleMode::Standardized,
    )?;
    Ok(RrblupExactSnpCache { inner })
}

#[pyfunction]
#[pyo3(signature = (
    source_prefix,
    train_sample_indices,
    row_source_indices,
    maf,
    row_flip,
    row_mean=None,
    row_inv_sd=None,
    sample_block=2048,
    stream_row_block=2048,
    std_eps=1e-12_f64,
    threads=0,
    blas_threads=0
))]
pub fn rrblupc_exact_snp_prepare_bed_from_meta<'py>(
    py: Python<'py>,
    source_prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
) -> PyResult<RrblupCenteredExactSnpCache> {
    let inner = rrblup_exact_snp_prepare_bed_from_meta_impl(
        py,
        source_prefix,
        train_sample_indices,
        row_source_indices,
        maf,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_block,
        stream_row_block,
        std_eps,
        threads,
        blas_threads,
        RrblupScaleMode::Centered,
    )?;
    Ok(RrblupCenteredExactSnpCache { inner })
}

#[pyfunction]
#[pyo3(signature = (
    source_prefix,
    full_train_sample_indices,
    row_source_indices,
    maf,
    row_flip,
    row_mean=None,
    row_inv_sd=None,
    sample_block=2048,
    stream_row_block=2048,
    std_eps=1e-12_f64,
    threads=0,
    blas_threads=0
))]
#[allow(clippy::too_many_arguments)]
pub fn rrblup_exact_snp_prepare_raw_bed_from_meta<'py>(
    py: Python<'py>,
    source_prefix: String,
    full_train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
) -> PyResult<RrblupExactSnpRawCache> {
    let inner = rrblup_exact_snp_prepare_raw_bed_from_meta_impl(
        py,
        source_prefix,
        full_train_sample_indices,
        row_source_indices,
        maf,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_block,
        stream_row_block,
        std_eps,
        threads,
        blas_threads,
        RrblupScaleMode::Standardized,
    )?;
    Ok(RrblupExactSnpRawCache { inner })
}

#[pyfunction]
#[pyo3(signature = (
    source_prefix,
    full_train_sample_indices,
    row_source_indices,
    maf,
    row_flip,
    row_mean=None,
    row_inv_sd=None,
    sample_block=2048,
    stream_row_block=2048,
    std_eps=1e-12_f64,
    threads=0,
    blas_threads=0
))]
#[allow(clippy::too_many_arguments)]
pub fn rrblupc_exact_snp_prepare_raw_bed_from_meta<'py>(
    py: Python<'py>,
    source_prefix: String,
    full_train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
) -> PyResult<RrblupExactSnpRawCache> {
    let inner = rrblup_exact_snp_prepare_raw_bed_from_meta_impl(
        py,
        source_prefix,
        full_train_sample_indices,
        row_source_indices,
        maf,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_block,
        stream_row_block,
        std_eps,
        threads,
        blas_threads,
        RrblupScaleMode::Centered,
    )?;
    Ok(RrblupExactSnpRawCache { inner })
}

#[pyfunction]
#[pyo3(signature = (
    raw_cache,
    source_prefix,
    train_sample_indices,
    row_source_indices,
    maf,
    row_flip,
    row_mean=None,
    row_inv_sd=None,
    sample_block=2048,
    stream_row_block=2048,
    std_eps=1e-12_f64,
    threads=0,
    blas_threads=0
))]
#[allow(clippy::too_many_arguments)]
pub fn rrblup_exact_snp_prepare_bed_from_raw_cache<'py>(
    py: Python<'py>,
    raw_cache: &RrblupExactSnpRawCache,
    source_prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
) -> PyResult<RrblupExactSnpCache> {
    let inner = rrblup_exact_snp_prepare_bed_from_raw_cache_impl(
        py,
        &raw_cache.inner,
        source_prefix,
        train_sample_indices,
        row_source_indices,
        maf,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_block,
        stream_row_block,
        std_eps,
        threads,
        blas_threads,
        RrblupScaleMode::Standardized,
    )?;
    Ok(RrblupExactSnpCache { inner })
}

#[pyfunction]
#[pyo3(signature = (
    raw_cache,
    source_prefix,
    train_sample_indices,
    row_source_indices,
    maf,
    row_flip,
    row_mean=None,
    row_inv_sd=None,
    sample_block=2048,
    stream_row_block=2048,
    std_eps=1e-12_f64,
    threads=0,
    blas_threads=0
))]
#[allow(clippy::too_many_arguments)]
pub fn rrblupc_exact_snp_prepare_bed_from_raw_cache<'py>(
    py: Python<'py>,
    raw_cache: &RrblupExactSnpRawCache,
    source_prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    sample_block: usize,
    stream_row_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
) -> PyResult<RrblupCenteredExactSnpCache> {
    let inner = rrblup_exact_snp_prepare_bed_from_raw_cache_impl(
        py,
        &raw_cache.inner,
        source_prefix,
        train_sample_indices,
        row_source_indices,
        maf,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_block,
        stream_row_block,
        std_eps,
        threads,
        blas_threads,
        RrblupScaleMode::Centered,
    )?;
    Ok(RrblupCenteredExactSnpCache { inner })
}

#[allow(clippy::too_many_arguments)]
fn rrblup_exact_snp_fit_prepared_bed_impl<'py>(
    py: Python<'py>,
    cache_inner: &Arc<RrblupExactSnpPreparedInner>,
    source_prefix: String,
    y_train: PyReadonlyArray1<'py, f64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    train_pred_local_indices: Option<PyReadonlyArray1<'py, i64>>,
    log10_lambda_low: f64,
    log10_lambda_high: f64,
    reml_tol: f64,
    reml_max_iter: usize,
    threads: usize,
    blas_threads: usize,
    use_fitted_alpha: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    f64,
    f64,
    (f64, f64),
    usize,
    f64,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    String,
)> {
    if str::trim(source_prefix.as_str()).is_empty() {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_fit_prepared_bed requires non-empty source_prefix.",
        ));
    }
    if !(log10_lambda_low.is_finite() && log10_lambda_high.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_fit_prepared_bed requires finite log10(lambda) bounds.",
        ));
    }
    if !(reml_tol.is_finite() && reml_tol > 0.0) {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_fit_prepared_bed requires finite reml_tol > 0.",
        ));
    }
    if reml_max_iter == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_fit_prepared_bed requires reml_max_iter > 0.",
        ));
    }
    let n_samples = crate::gfcore::read_fam(&source_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_fit_prepared_bed found zero samples in source_prefix.fam.",
        ));
    }
    if n_samples != cache_inner.n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "rrblup exact SNP bed cache sample-count mismatch: got {n_samples}, expected {}",
            cache_inner.n_samples
        )));
    }
    let n_train_cache = cache_inner.n_train;
    let y_vec_f64: Vec<f64> = match crate::stats_common::array_standard_slice(&y_train) {
        Ok(s) => s.to_vec(),
        Err(_) => crate::stats_common::copy_array_standard_layout(&y_train.as_array()),
    };
    let test_idx = if let Some(sidx) = test_sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "test_sample_indices")?
    } else {
        Vec::new()
    };
    let train_pred_pick = if let Some(local_idx) = train_pred_local_indices {
        Some(parse_index_vec_i64(
            local_idx.as_slice()?,
            n_train_cache,
            "train_pred_local_indices",
        )?)
    } else {
        None
    };
    let cache_inner = Arc::clone(cache_inner);
    let row_mean_ret = cache_inner.row_mean.clone();
    let row_inv_sd_ret = cache_inner.row_inv_sd.clone();
    let m_effective = cache_inner.m_effective;
    let bytes_per_snp = n_samples.div_ceil(4);
    let source_prefix_fit = source_prefix.clone();
    let (
        pred_train_ret,
        pred_test_ret,
        pve_trainvar,
        lambda_opt,
        reml_opt,
        var_g,
        sigma_e2,
        alpha_ret,
        beta_ret,
        eig_backend,
    ) = py
        .detach(move || {
            fit_rrblup_exact_snp_from_cache_source(
                cache_inner.as_ref(),
                None,
                source_prefix_fit.as_str(),
                bytes_per_snp,
                n_samples,
                y_vec_f64,
                test_idx,
                train_pred_pick,
                log10_lambda_low,
                log10_lambda_high,
                reml_tol,
                reml_max_iter,
                threads,
                blas_threads,
            )
        })
        .map_err(map_err_string_to_py)?;
    let alpha = if use_fitted_alpha {
        alpha_ret
    } else {
        y_train.as_slice()?.iter().copied().sum::<f64>() / (n_train_cache as f64)
    };
    let train_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((pred_train_ret.len(), 1), pred_train_ret)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let test_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((pred_test_ret.len(), 1), pred_test_ret)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let beta_arr = PyArray1::from_owned_array(py, Array1::from_vec(beta_ret)).into_bound();
    let mean_arr = PyArray1::from_owned_array(py, Array1::from_vec(row_mean_ret)).into_bound();
    let inv_arr = PyArray1::from_owned_array(py, Array1::from_vec(row_inv_sd_ret)).into_bound();
    Ok((
        train_arr,
        test_arr,
        pve_trainvar,
        lambda_opt,
        reml_opt,
        (var_g, sigma_e2),
        m_effective,
        alpha,
        beta_arr,
        mean_arr,
        inv_arr,
        eig_backend,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    cache,
    source_prefix,
    y_train,
    test_sample_indices=None,
    train_pred_local_indices=None,
    log10_lambda_low=-6.0_f64,
    log10_lambda_high=6.0_f64,
    reml_tol=1e-4_f64,
    reml_max_iter=50,
    threads=0,
    blas_threads=0
))]
pub fn rrblup_exact_snp_fit_prepared_bed<'py>(
    py: Python<'py>,
    cache: PyRef<'py, RrblupExactSnpCache>,
    source_prefix: String,
    y_train: PyReadonlyArray1<'py, f64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    train_pred_local_indices: Option<PyReadonlyArray1<'py, i64>>,
    log10_lambda_low: f64,
    log10_lambda_high: f64,
    reml_tol: f64,
    reml_max_iter: usize,
    threads: usize,
    blas_threads: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    f64,
    f64,
    (f64, f64),
    usize,
    f64,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    String,
)> {
    rrblup_exact_snp_fit_prepared_bed_impl(
        py,
        &cache.inner,
        source_prefix,
        y_train,
        test_sample_indices,
        train_pred_local_indices,
        log10_lambda_low,
        log10_lambda_high,
        reml_tol,
        reml_max_iter,
        threads,
        blas_threads,
        false,
    )
}

#[pyfunction]
#[pyo3(signature = (
    cache,
    source_prefix,
    y_train,
    test_sample_indices=None,
    train_pred_local_indices=None,
    log10_lambda_low=-6.0_f64,
    log10_lambda_high=6.0_f64,
    reml_tol=1e-4_f64,
    reml_max_iter=50,
    threads=0,
    blas_threads=0
))]
pub fn rrblupc_exact_snp_fit_prepared_bed<'py>(
    py: Python<'py>,
    cache: PyRef<'py, RrblupCenteredExactSnpCache>,
    source_prefix: String,
    y_train: PyReadonlyArray1<'py, f64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    train_pred_local_indices: Option<PyReadonlyArray1<'py, i64>>,
    log10_lambda_low: f64,
    log10_lambda_high: f64,
    reml_tol: f64,
    reml_max_iter: usize,
    threads: usize,
    blas_threads: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    f64,
    f64,
    (f64, f64),
    usize,
    f64,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    String,
)> {
    rrblup_exact_snp_fit_prepared_bed_impl(
        py,
        &cache.inner,
        source_prefix,
        y_train,
        test_sample_indices,
        train_pred_local_indices,
        log10_lambda_low,
        log10_lambda_high,
        reml_tol,
        reml_max_iter,
        threads,
        blas_threads,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn rrblup_pcg_bed_impl<'py>(
    py: Python<'py>,
    prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    y_train: PyReadonlyArray1<'py, f64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    train_pred_local_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    lambda_value: f64,
    tol: f64,
    max_iter: usize,
    block_rows: usize,
    std_eps: f64,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    compute_trainvar: bool,
    packed: Option<PyReadonlyArray2<'py, u8>>,
    packed_n_samples: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_source_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    blas_threads: usize,
    scale_mode: RrblupScaleMode,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    bool,
    usize,
    f64,
    usize,
    f64,
    f64,
    Bound<'py, PyArray1<f32>>,
)> {
    if max_iter == 0 {
        return Err(PyRuntimeError::new_err("max_iter must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be finite and > 0"));
    }
    if !(std_eps.is_finite() && std_eps > 0.0) {
        return Err(PyRuntimeError::new_err("std_eps must be finite and > 0"));
    }
    if !lambda_value.is_finite() || lambda_value < 0.0 {
        return Err(PyRuntimeError::new_err(
            "lambda_value must be finite and >= 0",
        ));
    }

    let fn_name = match scale_mode {
        RrblupScaleMode::Standardized => "rrblup_pcg_bed",
        RrblupScaleMode::Centered => "rrblupc_pcg_bed",
    };
    if packed.is_some() && row_source_indices.is_some() {
        return Err(PyRuntimeError::new_err(format!(
            "{fn_name}: packed payload and row_source_indices are mutually exclusive."
        )));
    }
    if site_keep.is_some() && row_source_indices.is_some() {
        return Err(PyRuntimeError::new_err(format!(
            "{fn_name}: site_keep and row_source_indices are mutually exclusive."
        )));
    }
    let use_external_packed = packed.is_some();
    let use_external_stats =
        (!use_external_packed) && (maf.is_some() || row_flip.is_some() || packed_n_samples > 0);
    let mut loaded_packed_arr: Option<Bound<'py, PyArray2<u8>>> = None;
    let mut loaded_maf_arr: Option<Bound<'py, PyArray1<f32>>> = None;
    #[allow(unused_assignments)]
    let mut external_packed_ro: Option<PyReadonlyArray2<'py, u8>> = None;
    #[allow(unused_assignments)]
    let mut loaded_packed_ro: Option<PyReadonlyArray2<'py, u8>> = None;
    let mut external_maf_ro: Option<PyReadonlyArray1<'py, f32>> = None;
    let mut external_row_flip_ro: Option<PyReadonlyArray1<'py, bool>> = None;
    let mut external_row_source_indices: Option<Vec<usize>> = None;

    let n_samples: usize;
    let bytes_per_snp: usize;
    let m_total: usize;
    let resident_packed_flat: Option<Cow<[u8]>>;

    if let Some(packed_ro) = packed {
        external_packed_ro = Some(packed_ro);
        external_maf_ro = Some(maf.ok_or_else(|| {
            PyRuntimeError::new_err("rrblup_pcg_bed: packed payload path requires `maf` argument.")
        })?);
        external_row_flip_ro = Some(row_flip.ok_or_else(|| {
            PyRuntimeError::new_err(
                "rrblup_pcg_bed: packed payload path requires `row_flip` argument.",
            )
        })?);
        if packed_n_samples == 0 {
            return Err(PyRuntimeError::new_err(
                "rrblup_pcg_bed: packed payload path requires packed_n_samples > 0.",
            ));
        }
        n_samples = packed_n_samples;
        let external_packed_ro_ref = external_packed_ro
            .as_ref()
            .expect("external packed readonly must exist");
        let packed_view = external_packed_ro_ref.as_array();
        if packed_view.ndim() != 2 {
            return Err(PyRuntimeError::new_err(
                "packed BED payload must be 2D (m, bytes_per_snp).",
            ));
        }
        m_total = packed_view.shape()[0];
        if m_total == 0 {
            return Err(PyRuntimeError::new_err("No SNP rows found in BED input."));
        }
        bytes_per_snp = packed_view.shape()[1];
        let expected_bps = n_samples.div_ceil(4);
        if bytes_per_snp != expected_bps {
            return Err(PyRuntimeError::new_err(format!(
                "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
            )));
        }
        resident_packed_flat = Some(
            match crate::stats_common::array_standard_slice(&external_packed_ro_ref) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                    &packed_view,
                )),
            },
        );
    } else if use_external_stats {
        external_maf_ro = Some(maf.ok_or_else(|| {
            PyRuntimeError::new_err("rrblup_pcg_bed: streaming stats path requires `maf` argument.")
        })?);
        external_row_flip_ro = Some(row_flip.ok_or_else(|| {
            PyRuntimeError::new_err(
                "rrblup_pcg_bed: streaming stats path requires `row_flip` argument.",
            )
        })?);
        if packed_n_samples == 0 {
            return Err(PyRuntimeError::new_err(
                "rrblup_pcg_bed: streaming stats path requires packed_n_samples > 0.",
            ));
        }
        if prefix.trim().is_empty() {
            return Err(PyRuntimeError::new_err(
                "rrblup_pcg_bed: streaming stats path requires non-empty prefix.",
            ));
        }
        n_samples = packed_n_samples;
        bytes_per_snp = n_samples.div_ceil(4);
        m_total = external_maf_ro
            .as_ref()
            .expect("external maf must exist")
            .as_array()
            .len();
        if m_total == 0 {
            return Err(PyRuntimeError::new_err(
                "rrblup_pcg_bed: streaming stats path received zero marker stats.",
            ));
        }
        if let Some(row_source_ro) = row_source_indices {
            let row_source_vec =
                parse_nonnegative_index_vec_i64(row_source_ro.as_slice()?, "row_source_indices")
                    .map_err(PyRuntimeError::new_err)?;
            if row_source_vec.len() != m_total {
                return Err(PyRuntimeError::new_err(format!(
                    "row_source_indices length mismatch: got {}, expected {m_total}",
                    row_source_vec.len()
                )));
            }
            if row_source_vec.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "row_source_indices must not be empty for metadata streaming path.",
                ));
            }
            external_row_source_indices = Some(row_source_vec);
        }
        resident_packed_flat = None;
    } else {
        let (packed_arr, _miss_arr, maf_arr, _std_arr, n_samples_loaded) =
            crate::gfreader::load_bed_2bit_packed(py, prefix.clone())?;
        if n_samples_loaded == 0 {
            return Err(PyRuntimeError::new_err("No samples found in BED input."));
        }
        n_samples = n_samples_loaded;
        loaded_packed_arr = Some(packed_arr);
        loaded_maf_arr = Some(maf_arr);
        loaded_packed_ro = Some(
            loaded_packed_arr
                .as_ref()
                .expect("packed array must exist")
                .readonly(),
        );
        let loaded_packed_ro_ref = loaded_packed_ro
            .as_ref()
            .expect("packed readonly view must exist");
        let packed_view = loaded_packed_ro_ref.as_array();
        if packed_view.ndim() != 2 {
            return Err(PyRuntimeError::new_err(
                "packed BED payload must be 2D (m, bytes_per_snp).",
            ));
        }
        m_total = packed_view.shape()[0];
        if m_total == 0 {
            return Err(PyRuntimeError::new_err("No SNP rows found in BED input."));
        }
        bytes_per_snp = packed_view.shape()[1];
        let expected_bps = n_samples.div_ceil(4);
        if bytes_per_snp != expected_bps {
            return Err(PyRuntimeError::new_err(format!(
                "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
            )));
        }
        resident_packed_flat = Some(
            match crate::stats_common::array_standard_slice(&loaded_packed_ro_ref) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                    &packed_view,
                )),
            },
        );
    }

    let maf_full: Vec<f32> = if use_external_packed || use_external_stats {
        let maf_ro = external_maf_ro
            .as_ref()
            .expect("external maf must exist for external stats path");
        match crate::stats_common::array_standard_slice(&maf_ro) {
            Ok(s) => s.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&maf_ro.as_array()),
        }
    } else {
        let maf_ro = loaded_maf_arr
            .as_ref()
            .expect("maf array must exist")
            .readonly();
        match crate::stats_common::array_standard_slice(&maf_ro) {
            Ok(s) => s.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&maf_ro.as_array()),
        }
    };
    if maf_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "maf length mismatch: got {}, expected {m_total}",
            maf_full.len()
        )));
    }

    let row_flip_full: Vec<bool> = if use_external_packed || use_external_stats {
        let row_flip_ro = external_row_flip_ro
            .as_ref()
            .expect("external row_flip must exist for external stats path");
        match crate::stats_common::array_standard_slice(&row_flip_ro) {
            Ok(s) => s.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&row_flip_ro.as_array()),
        }
    } else {
        let row_flip_full_arr = bed_packed_row_flip_mask(
            py,
            loaded_packed_arr
                .as_ref()
                .expect("packed array must exist")
                .readonly(),
            n_samples,
        )?;
        match row_flip_full_arr.readonly().as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_flip_full_arr
                .readonly()
                .as_array()
                .iter()
                .copied()
                .collect(),
        }
    };
    if row_flip_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m_total}",
            row_flip_full.len()
        )));
    }
    let row_mean_external_full: Option<Vec<f32>> = if let Some(row_mean_ro) = row_mean {
        Some(
            match crate::stats_common::array_standard_slice(&row_mean_ro) {
                Ok(s) => s.to_vec(),
                Err(_) => crate::stats_common::copy_array_standard_layout(&row_mean_ro.as_array()),
            },
        )
    } else {
        None
    };
    let row_inv_sd_external_full: Option<Vec<f32>> = if let Some(row_inv_sd_ro) = row_inv_sd {
        Some(
            match crate::stats_common::array_standard_slice(&row_inv_sd_ro) {
                Ok(s) => s.to_vec(),
                Err(_) => {
                    crate::stats_common::copy_array_standard_layout(&row_inv_sd_ro.as_array())
                }
            },
        )
    } else {
        None
    };

    let mut eff_m = m_total;
    let mut maf_keep: Cow<[f32]> = Cow::Borrowed(maf_full.as_slice());
    let mut row_flip_keep: Cow<[bool]> = Cow::Borrowed(row_flip_full.as_slice());
    let mut packed_row_indices: Option<Vec<usize>> = external_row_source_indices;
    if let Some(mask) = site_keep {
        let mask_vec: Vec<bool> = match crate::stats_common::array_standard_slice(&mask) {
            Ok(s) => s.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&mask.as_array()),
        };
        if mask_vec.len() != m_total {
            return Err(PyRuntimeError::new_err(format!(
                "site_keep length mismatch: got {}, expected {m_total}",
                mask_vec.len()
            )));
        }
        let keep_idx: Vec<usize> = mask_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &k)| if k { Some(i) } else { None })
            .collect();
        if keep_idx.is_empty() {
            return Err(PyRuntimeError::new_err(
                "No SNPs remained after applying site_keep mask.",
            ));
        }
        let keep_is_identity = (keep_idx.len() == m_total)
            && keep_idx
                .iter()
                .enumerate()
                .all(|(dst_row, &src_row)| dst_row == src_row);
        if !keep_is_identity {
            eff_m = keep_idx.len();
            let mut maf_subset = vec![0.0_f32; eff_m];
            let mut row_flip_subset = vec![false; eff_m];
            for (dst_row, &src_row) in keep_idx.iter().enumerate() {
                maf_subset[dst_row] = maf_full[src_row].clamp(0.0, 0.5);
                row_flip_subset[dst_row] = row_flip_full[src_row];
            }
            maf_keep = Cow::Owned(maf_subset);
            row_flip_keep = Cow::Owned(row_flip_subset);
            packed_row_indices = Some(keep_idx);
        }
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
    let y_vec_f64: Vec<f64> = match crate::stats_common::array_standard_slice(&y_train) {
        Ok(s) => s.to_vec(),
        Err(_) => crate::stats_common::copy_array_standard_layout(&y_train.as_array()),
    };
    if y_vec_f64.len() != train_idx.len() {
        return Err(PyRuntimeError::new_err(format!(
            "y_train length mismatch: got {}, expected {}",
            y_vec_f64.len(),
            train_idx.len()
        )));
    }
    if y_vec_f64.iter().any(|v| !v.is_finite()) {
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

    let m = eff_m;
    let n_train = train_idx.len();
    let row_step = block_rows.max(1).min(m.max(1));
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();

    let code4_lut = packed_byte_lut().code4;
    let full_train_fast = is_identity_indices(&train_idx, n_samples);
    let full_test_fast = is_identity_indices(&test_idx, n_samples);

    let lambda_use = lambda_value.max(1e-8) as f32;
    let tol_use = tol.max(1e-12);
    let std_eps32 = std_eps.max(1e-12) as f32;
    let notify_step = if progress_every == 0 {
        1usize
    } else {
        progress_every.max(1)
    };
    let stage_timing = env_truthy("JX_GS_PCG_STAGE_TIMING");
    let stage_log_every = std::env::var("JX_GS_PCG_STAGE_LOG_EVERY")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(100usize);
    let pcg_space_raw = std::env::var("JX_RRBLUP_PCG_SPACE").unwrap_or_else(|_| "dual".to_string());
    let pcg_space = RrblupPcgSpace::parse(&pcg_space_raw).map_err(PyRuntimeError::new_err)?;

    let (row_mean, row_inv_sd, m_effective) = rrblup_subset_or_validate_stats(
        row_mean_external_full,
        row_inv_sd_external_full,
        packed_row_indices.as_deref(),
        m_total,
        eff_m,
        std_eps32,
        maf_keep.as_ref(),
        pool_ref,
        scale_mode,
    )
    .map_err(PyRuntimeError::new_err)?;

    let y_mean = y_vec_f64.iter().sum::<f64>() / (n_train as f64);
    let y_center_f32: Vec<f32> = y_vec_f64.iter().map(|v| (*v - y_mean) as f32).collect();

    let (
        pred_train_ret,
        pred_test_ret,
        pve_trainvar,
        converged,
        iters,
        rel_res,
        pve_lambda_vc,
        k_trace_mean,
        beta_ret,
    ) = py
        .detach(
            move || -> Result<(Vec<f64>, Vec<f64>, f64, bool, usize, f64, f64, f64, Vec<f32>), String> {
                let blas_threads_effective = if blas_threads > 0 {
                    blas_threads.max(1)
                } else {
                    1usize
                };
                let _pcg_blas_guard = OpenBlasThreadGuard::enter(blas_threads_effective);
                let stage_accum = Arc::new(Mutex::new(PcgStageTimingAccum::default()));
                if pcg_space == RrblupPcgSpace::Dual {
                    let mut last_notified = 0usize;
                    let pcg_t0 = Instant::now();
                    let resident_packed = resident_packed_flat.as_ref().map(|cow| cow.as_ref());
                    let mut sum_ss_global = 0.0_f64;
                    let mut sum_ss_captured = false;
                    let dual_op = RrblupDualPcgOperator {
                        source: build_rrblup_source(
                            resident_packed,
                            bytes_per_snp,
                            prefix.as_str(),
                            n_samples,
                            row_step,
                        )?,
                        n_samples,
                        n_train,
                        m,
                        row_flip: row_flip_keep.as_ref(),
                        row_mean: &row_mean,
                        row_inv_sd: &row_inv_sd,
                        sample_idx: &train_idx,
                        full_sample_fast: full_train_fast,
                        packed_row_indices: packed_row_indices.as_deref(),
                        block_rows: row_step,
                        code4_lut: &code4_lut,
                        pool: pool_ref,
                        lambda_use,
                        stage_timing,
                        stage_accum: stage_accum.as_ref(),
                        block_buf: Vec::new(),
                        tmp_buf: vec![0.0_f32; row_step],
                        q_buf: vec![0.0_f32; n_train],
                        sum_ss_global: &mut sum_ss_global,
                        sum_ss_captured: &mut sum_ss_captured,
                    };
                    let pcg_res = pcg_solve(
                        &y_center_f32,
                        None,
                        max_iter,
                        tol_use,
                        1e-20_f64,
                        dual_op,
                        IdentityPreconditioner,
                        |iters_now, iters_max, rel_res_now| {
                            let cb_t0 = Instant::now();
                            if ((iters_now == 0usize) && (last_notified == 0usize))
                                || (iters_now >= last_notified.saturating_add(notify_step))
                                || (iters_now == iters_max)
                            {
                                last_notified = iters_now;
                                if let Some(cb) = progress_callback.as_ref() {
                                    Python::attach(|py2| -> PyResult<()> {
                                        py2.check_signals()?;
                                        match cb.call1(py2, (iters_now, iters_max, rel_res_now)) {
                                            Ok(_) => {}
                                            Err(err) => {
                                                if err.is_instance_of::<PyTypeError>(py2) {
                                                    cb.call1(py2, (iters_now, iters_max))?;
                                                } else {
                                                    return Err(err);
                                                }
                                            }
                                        }
                                        Ok(())
                                    })
                                    .map_err(|e| e.to_string())?;
                                }
                            }
                            if (iters_now & 7) == 0 {
                                check_ctrlc()?;
                            }
                            let cb_elapsed = cb_t0.elapsed().as_secs_f64();
                            if stage_timing {
                                if let Ok(mut acc) = stage_accum.lock() {
                                    acc.callback_secs += cb_elapsed;
                                }
                                if (iters_now % stage_log_every) == 0 || iters_now == iters_max {
                                    if let Ok(acc) = stage_accum.lock() {
                                        let elapsed = pcg_t0.elapsed().as_secs_f64().max(1e-12_f64);
                                        let decode_s = acc.decode_secs.max(0.0);
                                        let xmul_s = acc.xmul_secs.max(0.0);
                                        let xtmul_s = acc.xtmul_secs.max(0.0);
                                        let cb_s = acc.callback_secs.max(0.0);
                                        let known = decode_s + xmul_s + xtmul_s + cb_s;
                                        let update_s = (elapsed - known).max(0.0_f64);
                                        let pct = |x: f64| -> f64 { (x * 100.0) / elapsed };
                                        eprintln!(
                                            "rrBLUP-PCG(dual) stage timing iter={}/{} decode={:.3}s ({:.1}%) xmul={:.3}s ({:.1}%) xtmul={:.3}s ({:.1}%) update={:.3}s ({:.1}%) callback={:.3}s ({:.1}%) total={:.3}s",
                                            iters_now,
                                            iters_max,
                                            decode_s,
                                            pct(decode_s),
                                            xmul_s,
                                            pct(xmul_s),
                                            xtmul_s,
                                            pct(xtmul_s),
                                            update_s,
                                            pct(update_s),
                                            cb_s,
                                            pct(cb_s),
                                            elapsed
                                        );
                                    }
                                }
                            }
                            Ok(())
                        },
                    )?;
                    let converged = pcg_res.converged;
                    let iters_done = pcg_res.iters;
                    let rel_res = pcg_res.rel_res;
                    let alpha_raw = pcg_res.x;
                    let alpha_mean =
                        (alpha_raw.iter().map(|&v| v as f64).sum::<f64>() / (n_train as f64))
                            as f32;
                    let alpha_c: Vec<f32> = alpha_raw.iter().map(|&v| v - alpha_mean).collect();
                    if stage_timing {
                        if let Ok(acc) = stage_accum.lock() {
                            let elapsed = pcg_t0.elapsed().as_secs_f64().max(1e-12_f64);
                            let decode_s = acc.decode_secs.max(0.0);
                            let xmul_s = acc.xmul_secs.max(0.0);
                            let xtmul_s = acc.xtmul_secs.max(0.0);
                            let cb_s = acc.callback_secs.max(0.0);
                            let known = decode_s + xmul_s + xtmul_s + cb_s;
                            let update_s = (elapsed - known).max(0.0_f64);
                            let pct = |x: f64| -> f64 { (x * 100.0) / elapsed };
                            eprintln!(
                                "rrBLUP-PCG(dual) stage timing final iter={}/{} decode={:.3}s ({:.1}%) xmul={:.3}s ({:.1}%) xtmul={:.3}s ({:.1}%) update={:.3}s ({:.1}%) callback={:.3}s ({:.1}%) total={:.3}s",
                                iters_done,
                                max_iter,
                                decode_s,
                                pct(decode_s),
                                xmul_s,
                                pct(xmul_s),
                                xtmul_s,
                                pct(xtmul_s),
                                update_s,
                                pct(update_s),
                                cb_s,
                                pct(cb_s),
                                elapsed
                            );
                        }
                    }
                    let mut recover_source = build_rrblup_source(
                        resident_packed,
                        bytes_per_snp,
                        prefix.as_str(),
                        n_samples,
                        row_step,
                    )?;
                    let (beta, alpha_use, pred_test_ret, sum_ss_rec) =
                        rrblup_dual_recover_beta_and_test(
                            &mut recover_source,
                            n_samples,
                            row_flip_keep.as_ref(),
                            &row_mean,
                            &row_inv_sd,
                            &train_idx,
                            full_train_fast,
                            &test_idx,
                            full_test_fast,
                            packed_row_indices.as_deref(),
                            row_step,
                            &alpha_c,
                            y_mean,
                            !sum_ss_captured,
                            &code4_lut,
                            pool_ref,
                        )?;
                    if !sum_ss_captured {
                        sum_ss_global = sum_ss_rec;
                    }
                    let (pred_train_ret, pve_trainvar) = rrblup_dual_finalize_train_predictions(
                        resident_packed,
                        bytes_per_snp,
                        prefix.as_str(),
                        n_samples,
                        row_flip_keep.as_ref(),
                        &row_mean,
                        &row_inv_sd,
                        &train_idx,
                        full_train_fast,
                        train_pred_pick.as_deref(),
                        packed_row_indices.as_deref(),
                        row_step,
                        &y_vec_f64,
                        &alpha_c,
                        lambda_use,
                        &beta,
                        alpha_use,
                        converged,
                        compute_trainvar,
                        &code4_lut,
                        pool_ref,
                    )?;
                    let mean_k_trace = if n_train > 0 && m_effective > 0 {
                        sum_ss_global / ((m_effective as f64) * (n_train as f64))
                    } else {
                        f64::NAN
                    };
                    let pve_lambda_vc =
                        if mean_k_trace.is_finite() && mean_k_trace > 0.0 && m_effective > 0 {
                            let lambda_k = (lambda_use as f64) / (m_effective as f64);
                            let denom_vc = mean_k_trace + lambda_k;
                            if denom_vc.is_finite() && denom_vc > 0.0 {
                                mean_k_trace / denom_vc
                            } else {
                                f64::NAN
                            }
                        } else {
                            f64::NAN
                        };
                    return Ok((
                        pred_train_ret,
                        pred_test_ret,
                        pve_trainvar,
                        converged,
                        iters_done,
                        rel_res,
                        pve_lambda_vc,
                        mean_k_trace,
                        beta,
                    ));
                }
                let prep_t0 = Instant::now();
                let resident_packed = resident_packed_flat.as_ref().map(|cow| cow.as_ref());
                let mut prep_source = build_rrblup_source(
                    resident_packed,
                    bytes_per_snp,
                    prefix.as_str(),
                    n_samples,
                    row_step,
                )?;
                let (
                    b,
                    diag_inv,
                    train_row_mean,
                    sum_ss_global,
                    prep_decode_secs,
                    prep_rhs_secs,
                    prep_diag_secs,
                ) = rrblup_prepare_rhs_diag(
                    &mut prep_source,
                    n_samples,
                    row_flip_keep.as_ref(),
                    &row_mean,
                    &row_inv_sd,
                    &train_idx,
                    full_train_fast,
                    packed_row_indices.as_deref(),
                    row_step,
                    &y_center_f32,
                    lambda_use,
                    &code4_lut,
                    pool_ref,
                )?;
                if stage_timing {
                    let prep_total = prep_t0.elapsed().as_secs_f64().max(1e-12_f64);
                    let prep_decode_s = prep_decode_secs.max(0.0);
                    let prep_rhs_s = prep_rhs_secs.max(0.0);
                    let prep_diag_s = prep_diag_secs.max(0.0);
                    let prep_known = prep_decode_s + prep_rhs_s + prep_diag_s;
                    let prep_other_s = (prep_total - prep_known).max(0.0_f64);
                    let pct = |x: f64| -> f64 { (x * 100.0) / prep_total };
                    eprintln!(
                        "rrBLUP-PCG pre-pass timing decode={:.3}s ({:.1}%) rhs={:.3}s ({:.1}%) diag={:.3}s ({:.1}%) other={:.3}s ({:.1}%) total={:.3}s",
                        prep_decode_s, pct(prep_decode_s),
                        prep_rhs_s, pct(prep_rhs_s),
                        prep_diag_s, pct(prep_diag_s),
                        prep_other_s, pct(prep_other_s),
                        prep_total
                    );
                }

                let mut last_notified = 0usize;
                let pcg_t0 = Instant::now();
                let resident_packed = resident_packed_flat.as_ref().map(|cow| cow.as_ref());
                let apply_a = RrblupPcgOperator {
                    source: build_rrblup_source(
                        resident_packed,
                        bytes_per_snp,
                        prefix.as_str(),
                        n_samples,
                        row_step,
                    )?,
                    n_samples,
                    n_train,
                    row_flip: row_flip_keep.as_ref(),
                    row_mean: &row_mean,
                    row_inv_sd: &row_inv_sd,
                    train_row_mean: &train_row_mean,
                    sample_idx: &train_idx,
                    full_sample_fast: full_train_fast,
                    packed_row_indices: packed_row_indices.as_deref(),
                    block_rows: row_step,
                    code4_lut: &code4_lut,
                    pool: pool_ref,
                    lambda_use,
                    stage_timing,
                    stage_accum: stage_accum.as_ref(),
                };
                let pcg_res = pcg_solve(
                    &b,
                    None,
                    max_iter,
                    tol_use,
                    1e-20_f64,
                    apply_a,
                    DiagonalPreconditioner::new(&diag_inv),
                    |iters_now, iters_max, rel_res_now| {
                        let cb_t0 = Instant::now();
                        if ((iters_now == 0usize) && (last_notified == 0usize))
                            || (iters_now >= last_notified.saturating_add(notify_step))
                            || (iters_now == iters_max)
                        {
                            last_notified = iters_now;
                            if let Some(cb) = progress_callback.as_ref() {
                                Python::attach(|py2| -> PyResult<()> {
                                    py2.check_signals()?;
                                    match cb.call1(py2, (iters_now, iters_max, rel_res_now)) {
                                        Ok(_) => {}
                                        Err(err) => {
                                            if err.is_instance_of::<PyTypeError>(py2) {
                                                cb.call1(py2, (iters_now, iters_max))?;
                                            } else {
                                                return Err(err);
                                            }
                                        }
                                    }
                                    Ok(())
                                })
                                .map_err(|e| e.to_string())?;
                            }
                        }
                        if (iters_now & 7) == 0 {
                            check_ctrlc()?;
                        }
                        let cb_elapsed = cb_t0.elapsed().as_secs_f64();
                        if stage_timing {
                            if let Ok(mut acc) = stage_accum.lock() {
                                acc.callback_secs += cb_elapsed;
                            }
                            if (iters_now % stage_log_every) == 0 || iters_now == iters_max {
                                if let Ok(acc) = stage_accum.lock() {
                                    let elapsed = pcg_t0.elapsed().as_secs_f64().max(1e-12_f64);
                                    let decode_s = acc.decode_secs.max(0.0);
                                    let xmul_s = acc.xmul_secs.max(0.0);
                                    let xtmul_s = acc.xtmul_secs.max(0.0);
                                    let cb_s = acc.callback_secs.max(0.0);
                                    let known = decode_s + xmul_s + xtmul_s + cb_s;
                                    let update_s = (elapsed - known).max(0.0_f64);
                                    let pct = |x: f64| -> f64 { (x * 100.0) / elapsed };
                                    eprintln!(
                                        "rrBLUP-PCG stage timing iter={}/{} decode={:.3}s ({:.1}%) xmul={:.3}s ({:.1}%) xtmul={:.3}s ({:.1}%) update={:.3}s ({:.1}%) callback={:.3}s ({:.1}%) total={:.3}s",
                                        iters_now,
                                        iters_max,
                                        decode_s, pct(decode_s),
                                        xmul_s, pct(xmul_s),
                                        xtmul_s, pct(xtmul_s),
                                        update_s, pct(update_s),
                                        cb_s, pct(cb_s),
                                        elapsed
                                    );
                                }
                            }
                        }
                        Ok(())
                    },
                )?;
                let beta = pcg_res.x;
                let alpha_use = (y_mean as f32)
                    - train_row_mean
                        .iter()
                        .zip(beta.iter())
                        .map(|(mu, bj)| *mu * *bj)
                        .sum::<f32>();
                let converged = pcg_res.converged;
                let iters_done = pcg_res.iters;
                let rel_res = pcg_res.rel_res;
                if stage_timing {
                    if let Ok(acc) = stage_accum.lock() {
                        let elapsed = pcg_t0.elapsed().as_secs_f64().max(1e-12_f64);
                        let decode_s = acc.decode_secs.max(0.0);
                        let xmul_s = acc.xmul_secs.max(0.0);
                        let xtmul_s = acc.xtmul_secs.max(0.0);
                        let cb_s = acc.callback_secs.max(0.0);
                        let known = decode_s + xmul_s + xtmul_s + cb_s;
                        let update_s = (elapsed - known).max(0.0_f64);
                        let pct = |x: f64| -> f64 { (x * 100.0) / elapsed };
                        eprintln!(
                            "rrBLUP-PCG stage timing final iter={}/{} decode={:.3}s ({:.1}%) xmul={:.3}s ({:.1}%) xtmul={:.3}s ({:.1}%) update={:.3}s ({:.1}%) callback={:.3}s ({:.1}%) total={:.3}s",
                            iters_done,
                            max_iter,
                            decode_s, pct(decode_s),
                            xmul_s, pct(xmul_s),
                            xtmul_s, pct(xtmul_s),
                            update_s, pct(update_s),
                            cb_s, pct(cb_s),
                            elapsed
                        );
                    }
                }

                let need_train_pred_all = train_pred_pick.is_none();
                let need_train_pred_subset = train_pred_pick
                    .as_ref()
                    .map(|idx| !idx.is_empty())
                    .unwrap_or(false);
                let need_train_pred_any = need_train_pred_all || need_train_pred_subset;
                let need_train_full = compute_trainvar || need_train_pred_all;

                let mut pred_train_ret: Vec<f64> = Vec::new();
                let pve_trainvar = if need_train_full {
                    let resident_packed = resident_packed_flat.as_ref().map(|cow| cow.as_ref());
                    let mut pred_train_source = build_rrblup_source(
                        resident_packed,
                        bytes_per_snp,
                        prefix.as_str(),
                        n_samples,
                        row_step,
                    )?;
                    let mut pred_train_full = pcg_x_mul_samples(
                        &mut pred_train_source,
                        n_samples,
                        row_flip_keep.as_ref(),
                        &row_mean,
                        &row_inv_sd,
                        &train_idx,
                        full_train_fast,
                        packed_row_indices.as_deref(),
                        row_step,
                        &beta,
                        &code4_lut,
                        pool_ref,
                        None,
                    )?;
                    for v in pred_train_full.iter_mut() {
                        *v += alpha_use;
                    }
                    let pred_train_f64: Vec<f64> =
                        pred_train_full.iter().map(|v| *v as f64).collect();
                    pred_train_ret = if let Some(local_idx) = &train_pred_pick {
                        local_idx.iter().map(|&i| pred_train_f64[i]).collect()
                    } else {
                        pred_train_f64.clone()
                    };

                    if compute_trainvar {
                        let resid: Vec<f64> =
                            if let Some(tp) = pool_ref.filter(|_| y_vec_f64.len() >= RRBLUP_PAR_VEC_THRESHOLD)
                            {
                                tp.install(|| {
                                    y_vec_f64
                                        .par_iter()
                                        .zip(pred_train_f64.par_iter())
                                        .map(|(y, p)| *y - *p)
                                        .collect()
                                })
                            } else {
                                y_vec_f64
                                    .iter()
                                    .zip(pred_train_f64.iter())
                                    .map(|(y, p)| *y - *p)
                                    .collect()
                            };
                        let var_g = sample_var_f64(&pred_train_f64, pool_ref);
                        let var_e = sample_var_f64(&resid, pool_ref);
                        let denom = var_g + var_e;
                        if denom.is_finite() && denom > 0.0 {
                            var_g / denom
                        } else {
                            f64::NAN
                        }
                    } else {
                        f64::NAN
                    }
                } else {
                    if need_train_pred_any {
                        if let Some(local_idx) = &train_pred_pick {
                            let train_pred_abs: Vec<usize> =
                                local_idx.iter().map(|&i| train_idx[i]).collect();
                            let train_pred_fast = is_identity_indices(&train_pred_abs, n_samples);
                            let resident_packed =
                                resident_packed_flat.as_ref().map(|cow| cow.as_ref());
                            let mut pred_train_sub_source = build_rrblup_source(
                                resident_packed,
                                bytes_per_snp,
                                prefix.as_str(),
                                n_samples,
                                row_step,
                            )?;
                            let mut pred_train_sub = pcg_x_mul_samples(
                                &mut pred_train_sub_source,
                                n_samples,
                                row_flip_keep.as_ref(),
                                &row_mean,
                                &row_inv_sd,
                                &train_pred_abs,
                                train_pred_fast,
                                packed_row_indices.as_deref(),
                                row_step,
                                &beta,
                                &code4_lut,
                                pool_ref,
                                None,
                            )?;
                            for v in &mut pred_train_sub {
                                *v += alpha_use;
                            }
                            pred_train_ret =
                                pred_train_sub.iter().map(|v| *v as f64).collect();
                        }
                    }
                    f64::NAN
                };
                let mean_k_trace = if n_train > 0 && m_effective > 0 {
                    sum_ss_global / ((m_effective as f64) * (n_train as f64))
                } else {
                    f64::NAN
                };
                let pve_lambda_vc =
                    if mean_k_trace.is_finite() && mean_k_trace > 0.0 && m_effective > 0 {
                        let lambda_k = (lambda_use as f64) / (m_effective as f64);
                        let denom_vc = mean_k_trace + lambda_k;
                        if denom_vc.is_finite() && denom_vc > 0.0 {
                            mean_k_trace / denom_vc
                        } else {
                            f64::NAN
                        }
                    } else {
                        f64::NAN
                    };

                let mut pred_test_ret: Vec<f64> = Vec::new();
                if !test_idx.is_empty() {
                    let resident_packed = resident_packed_flat.as_ref().map(|cow| cow.as_ref());
                    let mut pred_test_source = build_rrblup_source(
                        resident_packed,
                        bytes_per_snp,
                        prefix.as_str(),
                        n_samples,
                        row_step,
                    )?;
                    let test_pred_f32 = pcg_x_mul_samples(
                        &mut pred_test_source,
                        n_samples,
                        row_flip_keep.as_ref(),
                        &row_mean,
                        &row_inv_sd,
                        &test_idx,
                        full_test_fast,
                        packed_row_indices.as_deref(),
                        row_step,
                        &beta,
                        &code4_lut,
                        pool_ref,
                        None,
                    )?;
                    pred_test_ret = test_pred_f32
                        .iter()
                        .map(|v| (*v as f64) + (alpha_use as f64))
                        .collect();
                }

                Ok((
                    pred_train_ret,
                    pred_test_ret,
                    pve_trainvar,
                    converged,
                    iters_done,
                    rel_res,
                    pve_lambda_vc,
                    mean_k_trace,
                    beta,
                ))
            },
        )
        .map_err(map_err_string_to_py)?;

    let train_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((pred_train_ret.len(), 1), pred_train_ret)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let test_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((pred_test_ret.len(), 1), pred_test_ret)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let beta_arr = PyArray1::from_owned_array(py, Array1::from_vec(beta_ret)).into_bound();

    Ok((
        train_arr,
        test_arr,
        pve_trainvar,
        converged,
        iters,
        rel_res,
        m_effective,
        pve_lambda_vc,
        k_trace_mean,
        beta_arr,
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
    lambda_value=10000.0_f64,
    tol=1e-4_f64,
    max_iter=100,
    block_rows=4096,
    std_eps=1e-12_f64,
    threads=0,
    progress_callback=None,
    progress_every=0,
    compute_trainvar=false,
    packed=None,
    packed_n_samples=0,
    maf=None,
    row_flip=None,
    row_source_indices=None,
    row_mean=None,
    row_inv_sd=None,
    blas_threads=0
))]
pub fn rrblup_pcg_bed<'py>(
    py: Python<'py>,
    prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    y_train: PyReadonlyArray1<'py, f64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    train_pred_local_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    lambda_value: f64,
    tol: f64,
    max_iter: usize,
    block_rows: usize,
    std_eps: f64,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    compute_trainvar: bool,
    packed: Option<PyReadonlyArray2<'py, u8>>,
    packed_n_samples: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_source_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_mean: Option<PyReadonlyArray1<'py, f32>>,
    row_inv_sd: Option<PyReadonlyArray1<'py, f32>>,
    blas_threads: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    bool,
    usize,
    f64,
    usize,
    f64,
    f64,
    Bound<'py, PyArray1<f32>>,
)> {
    rrblup_pcg_bed_impl(
        py,
        prefix,
        train_sample_indices,
        y_train,
        test_sample_indices,
        train_pred_local_indices,
        site_keep,
        lambda_value,
        tol,
        max_iter,
        block_rows,
        std_eps,
        threads,
        progress_callback,
        progress_every,
        compute_trainvar,
        packed,
        packed_n_samples,
        maf,
        row_flip,
        row_source_indices,
        row_mean,
        row_inv_sd,
        blas_threads,
        RrblupScaleMode::Standardized,
    )
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    train_sample_indices,
    y_train,
    test_sample_indices=None,
    train_pred_local_indices=None,
    site_keep=None,
    lambda_value=10000.0_f64,
    tol=1e-4_f64,
    max_iter=100,
    block_rows=4096,
    std_eps=1e-12_f64,
    threads=0,
    progress_callback=None,
    progress_every=0,
    compute_trainvar=false,
    packed=None,
    packed_n_samples=0,
    maf=None,
    row_flip=None,
    row_source_indices=None,
    blas_threads=0
))]
pub fn rrblupc_pcg_bed<'py>(
    py: Python<'py>,
    prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    y_train: PyReadonlyArray1<'py, f64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    train_pred_local_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    lambda_value: f64,
    tol: f64,
    max_iter: usize,
    block_rows: usize,
    std_eps: f64,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    compute_trainvar: bool,
    packed: Option<PyReadonlyArray2<'py, u8>>,
    packed_n_samples: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_source_indices: Option<PyReadonlyArray1<'py, i64>>,
    blas_threads: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    bool,
    usize,
    f64,
    usize,
    f64,
    f64,
    Bound<'py, PyArray1<f32>>,
)> {
    rrblup_pcg_bed_impl(
        py,
        prefix,
        train_sample_indices,
        y_train,
        test_sample_indices,
        train_pred_local_indices,
        site_keep,
        lambda_value,
        tol,
        max_iter,
        block_rows,
        std_eps,
        threads,
        progress_callback,
        progress_every,
        compute_trainvar,
        packed,
        packed_n_samples,
        maf,
        row_flip,
        row_source_indices,
        None,
        None,
        blas_threads,
        RrblupScaleMode::Centered,
    )
}

/// Build the rrBLUP standardized sample GRM from packed BED data using the
/// same per-marker means/scales as the marker-space operator. The dense
/// accumulation is performed through JanusX's selected Rust BLAS backend
/// (Accelerate on macOS, OpenBLAS where available elsewhere).
#[pyfunction]
#[pyo3(signature = (
    row_indices,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    n_samples,
    prefix=None,
    packed=None,
    block_rows=4096,
    threads=0,
    blas_threads=0
))]
pub fn rrblup_standardized_grm_from_source<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_mean: PyReadonlyArray1<'py, f32>,
    row_inv_sd: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    n_samples: usize,
    prefix: Option<String>,
    packed: Option<PyReadonlyArray2<'py, u8>>,
    block_rows: usize,
    threads: usize,
    blas_threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0."));
    }
    if prefix.as_deref().map(str::trim).unwrap_or("").is_empty() == packed.is_none() {
        return Err(PyRuntimeError::new_err(
            "Provide exactly one of `prefix` or `packed` to build the rrBLUP GRM.",
        ));
    }

    let row_indices_vec = parse_nonnegative_index_vec_i64(row_indices.as_slice()?, "row_indices")
        .map_err(PyRuntimeError::new_err)?;
    if row_indices_vec.is_empty() {
        return Err(PyRuntimeError::new_err("row_indices must not be empty."));
    }
    let to_vec_f32 = |values: &PyReadonlyArray1<'py, f32>| -> Vec<f32> {
        match crate::stats_common::array_standard_slice(values) {
            Ok(slice) => slice.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&values.as_array()),
        }
    };
    let row_flip_vec = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(slice) => slice.to_vec(),
        Err(_) => crate::stats_common::copy_array_standard_layout(&row_flip.as_array()),
    };
    let row_maf_vec = to_vec_f32(&row_maf);
    let row_mean_vec = to_vec_f32(&row_mean);
    let row_inv_sd_vec = to_vec_f32(&row_inv_sd);
    let m = row_indices_vec.len();
    if row_flip_vec.len() != m
        || row_maf_vec.len() != m
        || row_mean_vec.len() != m
        || row_inv_sd_vec.len() != m
    {
        return Err(PyRuntimeError::new_err(format!(
            "rrBLUP GRM row metadata length mismatch: indices={m}, flip={}, maf={}, mean={}, inv_sd={}",
            row_flip_vec.len(),
            row_maf_vec.len(),
            row_mean_vec.len(),
            row_inv_sd_vec.len(),
        )));
    }
    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.is_empty() {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty."));
    }

    let packed_ro = packed;
    let mut packed_bytes_per_row = n_samples.div_ceil(4);
    let resident_packed: Option<Cow<[u8]>> = if let Some(packed_view) = packed_ro.as_ref() {
        let arr = packed_view.as_array();
        if arr.ndim() != 2 {
            return Err(PyRuntimeError::new_err(
                "packed BED payload must be 2D (m, bytes_per_snp).",
            ));
        }
        packed_bytes_per_row = arr.shape()[1];
        if packed_bytes_per_row != n_samples.div_ceil(4) {
            return Err(PyRuntimeError::new_err(format!(
                "packed BED row width mismatch: got {packed_bytes_per_row}, expected {}.",
                n_samples.div_ceil(4),
            )));
        }
        if row_indices_vec.iter().any(|&i| i >= arr.shape()[0]) {
            return Err(PyRuntimeError::new_err(
                "row_indices contain an index outside the packed BED payload.",
            ));
        }
        Some(
            match crate::stats_common::array_standard_slice(packed_view) {
                Ok(slice) => Cow::Borrowed(slice),
                Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&arr)),
            },
        )
    } else {
        None
    };

    let source_prefix = prefix.unwrap_or_default();
    let effective_rows: Vec<usize> = row_inv_sd_vec
        .iter()
        .enumerate()
        .filter_map(|(i, &inv)| {
            if inv.is_finite() && inv > 0.0 {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let m_effective = effective_rows.len();
    if m_effective == 0 {
        return Err(PyRuntimeError::new_err(
            "rrBLUP GRM build found zero effective markers.",
        ));
    }
    let row_step = block_rows.max(1).min(m_effective);
    let n_out = sample_idx.len();
    let blas_threads_use = if blas_threads > 0 {
        blas_threads
    } else if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    };

    let grm_vec = py
        .detach(move || -> Result<Vec<f64>, String> {
            let mut matrix = if resident_packed.is_none() {
                let window_mb = rrblup_stream_window_mb(n_samples, row_step);
                let matrix = WindowedBedMatrix::open(&source_prefix, window_mb)?;
                if matrix.n_samples_full() != n_samples {
                    return Err(format!(
                        "rrBLUP GRM sample count mismatch: BED has {}, metadata says {n_samples}.",
                        matrix.n_samples_full()
                    ));
                }
                if row_indices_vec.iter().any(|&i| i >= matrix.n_source_snps()) {
                    return Err("row_indices contain an index outside the BED input.".to_string());
                }
                Some(matrix)
            } else {
                None
            };
            let pool_owned = get_cached_pool(threads).map_err(|e| e.to_string())?;
            let pool = pool_owned.as_ref();
            let _blas_guard = BlasThreadGuard::enter(blas_threads_use);
            let n = n_out;
            let grm_len = n
                .checked_mul(n)
                .ok_or_else(|| "rrBLUP GRM dimensions overflow address space.".to_string())?;
            let block_len = row_step
                .checked_mul(n)
                .ok_or_else(|| "rrBLUP GRM block dimensions overflow address space.".to_string())?;
            let mut grm = vec![0.0_f64; grm_len];
            let mut block = vec![0.0_f64; block_len];
            let mut source_rows = Vec::with_capacity(row_step);
            let mut relative_rows = Vec::with_capacity(row_step);
            for start in (0..m_effective).step_by(row_step) {
                let end = (start + row_step).min(m_effective);
                let cur_rows = end - start;
                source_rows.clear();
                source_rows.extend(
                    effective_rows[start..end]
                        .iter()
                        .map(|&meta_idx| row_indices_vec[meta_idx]),
                );
                relative_rows.clear();
                let packed_slice = if let Some(resident) = resident_packed.as_ref() {
                    relative_rows.extend_from_slice(source_rows.as_slice());
                    resident.as_ref()
                } else {
                    matrix
                        .as_mut()
                        .expect("streamed BED matrix was initialized")
                        .prepare_source_rows(source_rows.as_slice(), &mut relative_rows)?
                };
                let block_slice = &mut block[..cur_rows * n];
                decode_rrblup_grm_block_f64(
                    packed_slice,
                    packed_bytes_per_row,
                    n_samples,
                    relative_rows.as_slice(),
                    &effective_rows[start..end],
                    row_flip_vec.as_slice(),
                    row_maf_vec.as_slice(),
                    row_mean_vec.as_slice(),
                    row_inv_sd_vec.as_slice(),
                    sample_idx.as_slice(),
                    block_slice,
                    pool,
                )?;
                accumulate_rrblup_grm_syrk(block_slice, cur_rows, n, grm.as_mut_slice());
                check_ctrlc().map_err(|e| e.to_string())?;
            }

            scale_and_mirror_rrblup_grm(&mut grm, n, 1.0_f64 / m_effective as f64);
            Ok(grm)
        })
        .map_err(map_err_string_to_py)?;

    let n = n_out;
    let grm_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n, n), grm_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok(grm_arr)
}

#[allow(clippy::too_many_arguments)]
fn decode_rrblup_grm_block_f64(
    packed: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    relative_rows: &[usize],
    metadata_rows: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let n_out = sample_idx.len();
    let n_rows = relative_rows.len();
    if n_out == 0 || n_rows == 0 {
        return Ok(());
    }
    if metadata_rows.len() != n_rows || out.len() != n_rows.saturating_mul(n_out) {
        return Err("rrBLUP GRM decoded block dimensions are inconsistent.".to_string());
    }
    if bytes_per_snp < n_samples.div_ceil(4)
        || relative_rows
            .iter()
            .any(|&row| row >= packed.len().saturating_div(bytes_per_snp.max(1)))
    {
        return Err("rrBLUP GRM BED row storage is truncated.".to_string());
    }

    let mut decode = || {
        out.par_chunks_mut(n_out)
            .enumerate()
            .for_each(|(local_row, output)| {
                let meta = metadata_rows[local_row];
                let packed_row = relative_rows[local_row];
                let row = &packed[packed_row * bytes_per_snp..(packed_row + 1) * bytes_per_snp];
                let mean = row_mean[meta];
                let inv_sd = row_inv_sd[meta];
                let missing_mean = (2.0_f32 * row_maf[meta]).max(0.0_f32);
                let flip = row_flip[meta];
                let value_lut = if flip {
                    [
                        (2.0_f32 - mean) * inv_sd,
                        (missing_mean - mean) * inv_sd,
                        (1.0_f32 - mean) * inv_sd,
                        (0.0_f32 - mean) * inv_sd,
                    ]
                } else {
                    [
                        (0.0_f32 - mean) * inv_sd,
                        (missing_mean - mean) * inv_sd,
                        (1.0_f32 - mean) * inv_sd,
                        (2.0_f32 - mean) * inv_sd,
                    ]
                };
                for (dst, &sample) in output.iter_mut().zip(sample_idx.iter()) {
                    let byte = row[sample >> 2];
                    let code = ((byte >> ((sample & 3) * 2)) & 0b11) as usize;
                    *dst = value_lut[code] as f64;
                }
            });
    };
    if let Some(tp) = pool {
        tp.install(decode);
    } else {
        decode();
    }
    Ok(())
}

#[inline]
fn accumulate_rrblup_grm_syrk(block: &[f64], block_rows: usize, n_samples: usize, grm: &mut [f64]) {
    unsafe {
        cblas_dsyrk_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_UPPER,
            CBLAS_NO_TRANS,
            n_samples as CblasInt,
            block_rows as CblasInt,
            1.0_f64,
            block.as_ptr(),
            n_samples as CblasInt,
            1.0_f64,
            grm.as_mut_ptr(),
            n_samples as CblasInt,
        );
    }
}

#[inline]
fn scale_and_mirror_rrblup_grm(grm: &mut [f64], n: usize, scale: f64) {
    debug_assert_eq!(grm.len(), n.saturating_mul(n));
    // DSYRK wrote the upper triangle in column-major view, which is the lower
    // triangle of this row-major buffer. Mirror it without a dense temporary.
    for col in 0..n {
        grm[col * n + col] *= scale;
        for row in 0..col {
            let lower_idx = col * n + row;
            let upper_idx = row * n + col;
            let value = grm[lower_idx] * scale;
            grm[lower_idx] = value;
            grm[upper_idx] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_xtmul_inputs(rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let mut block = vec![0.0_f32; rows.saturating_mul(cols)];
        for r in 0..rows {
            let row_off = r * cols;
            let row_bias = (((r * 17 + 11) % 97) as f32 - 48.0_f32) * 0.0035_f32;
            for c in 0..cols {
                let val = (((c * 13 + r * 7 + 19) % 251) as f32 - 125.0_f32) * 0.002_f32;
                block[row_off + c] = val + row_bias;
            }
        }
        let mut weights = vec![0.0_f32; rows];
        for (r, w) in weights.iter_mut().enumerate() {
            *w = (((r * 29 + 5) % 113) as f32 - 56.0_f32) * 0.015_f32;
        }
        (block, weights)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn pcg_prepare_rhs_diag_parallel_matches_serial() {
        let rows = 257usize;
        let cols = 1536usize;
        let lambda_use = 0.37_f32;
        let (block, weights) = make_xtmul_inputs(rows, cols);
        let dot_blk: Vec<f32> = (0..rows)
            .map(|r| {
                let row = &block[r * cols..(r + 1) * cols];
                row.iter()
                    .zip(weights.iter().cycle())
                    .map(|(a, w)| *a * *w)
                    .sum::<f32>()
            })
            .collect();

        let mut b_serial = vec![0.0_f32; rows];
        let mut diag_serial = vec![0.0_f32; rows];
        let mut mean_serial = vec![0.0_f32; rows];
        let ss_serial = row_major_block_prepare_rhs_diag_f32(
            &block,
            rows,
            cols,
            &dot_blk,
            lambda_use,
            &mut b_serial,
            &mut diag_serial,
            &mut mean_serial,
            None,
        );

        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .expect("build rayon pool"),
        );
        let mut b_parallel = vec![0.0_f32; rows];
        let mut diag_parallel = vec![0.0_f32; rows];
        let mut mean_parallel = vec![0.0_f32; rows];
        let ss_parallel = row_major_block_prepare_rhs_diag_f32(
            &block,
            rows,
            cols,
            &dot_blk,
            lambda_use,
            &mut b_parallel,
            &mut diag_parallel,
            &mut mean_parallel,
            Some(&pool),
        );

        assert!((ss_serial - ss_parallel).abs() <= 1e-6_f64 * ss_serial.abs().max(1.0));
        assert!(max_abs_diff(&b_serial, &b_parallel) <= 1e-6_f32);
        assert!(max_abs_diff(&diag_serial, &diag_parallel) <= 1e-6_f32);
        assert!(max_abs_diff(&mean_serial, &mean_parallel) <= 1e-6_f32);
    }

    #[test]
    fn rrblup_grm_block_preserves_flip_missing_and_empirical_stats() {
        // PLINK two-bit order for four samples: 0, missing, 1, 2.
        let packed = [0b11_10_01_00_u8];
        let mut decoded = [0.0_f64; 4];
        decode_rrblup_grm_block_f64(
            &packed,
            1,
            4,
            &[0],
            &[0],
            &[true],
            &[0.25],
            &[0.5],
            &[2.0],
            &[0, 1, 2, 3],
            &mut decoded,
            None,
        )
        .expect("decode standardized row");
        assert_eq!(decoded, [3.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn rrblup_grm_syrk_accumulates_and_mirrors_in_place() {
        // Three marker rows by two samples. The row-major buffer can be viewed
        // as a column-major 2x3 matrix, so DSYRK forms X'X directly.
        let block = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut grm = [0.0_f64; 4];
        accumulate_rrblup_grm_syrk(&block, 3, 2, &mut grm);
        scale_and_mirror_rrblup_grm(&mut grm, 2, 1.0 / 3.0);
        let expected = [35.0 / 3.0, 44.0 / 3.0, 44.0 / 3.0, 56.0 / 3.0];
        for (observed, expected) in grm.iter().zip(expected) {
            assert!((observed - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn rrblup_parallel_row_standardization_supports_standardized_and_centered() {
        let maf = [0.0_f32, 0.25_f32, 0.5_f32];
        let (mean_std, inv_std, eff_std) = rrblup_parallel_row_standardization(
            &maf,
            1e-12_f32,
            None,
            RrblupScaleMode::Standardized,
        );
        let (mean_cen, inv_cen, eff_cen) =
            rrblup_parallel_row_standardization(&maf, 1e-12_f32, None, RrblupScaleMode::Centered);
        assert_eq!(eff_std, 2);
        assert_eq!(eff_cen, 2);
        assert_eq!(mean_std, [0.0_f32, 0.5_f32, 1.0_f32]);
        assert_eq!(mean_cen, mean_std);
        assert_eq!(inv_std[0], 0.0_f32);
        assert!((inv_std[1] - 1.0_f32 / (0.375_f32).sqrt()).abs() < 1e-6_f32);
        assert!((inv_std[2] - 1.0_f32 / (0.5_f32).sqrt()).abs() < 1e-6_f32);
        assert_eq!(inv_cen, [0.0_f32, 1.0_f32, 1.0_f32]);
    }

    #[test]
    fn rrblup_exact_cv_subtraction_matches_direct_build_standardized_and_centered() {
        pyo3::Python::initialize();
        let n_samples = 20usize;
        let eff_m = 6usize;
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut packed = vec![0u8; eff_m * bytes_per_snp];
        for r in 0..eff_m {
            for s in 0..n_samples {
                let code: u8 = match (r * 7 + s * 3) % 4 {
                    0 => 0b00,
                    1 => 0b10,
                    2 => 0b11,
                    _ => 0b01,
                };
                packed[r * bytes_per_snp + (s >> 2)] |= code << ((s & 3) * 2);
            }
        }
        let maf = vec![0.25_f32; eff_m];
        let row_flip: Vec<bool> = (0..eff_m).map(|r| r % 2 == 1).collect();
        let full_train_idx: Vec<usize> = (0..16).collect();
        let fold_train_idx: Vec<usize> = (4..16).collect();
        let test_idx: Vec<usize> = (16..20).collect();
        let y_fold: Vec<f64> = fold_train_idx
            .iter()
            .enumerate()
            .map(|(i, &s)| ((s as f64) * 0.37 - (i as f64) * 0.11).sin())
            .collect();
        let y_full: Vec<f64> = full_train_idx
            .iter()
            .enumerate()
            .map(|(i, &s)| ((s as f64) * 0.37 - (i as f64) * 0.11).sin())
            .collect();

        for mode in [RrblupScaleMode::Standardized, RrblupScaleMode::Centered] {
            let (row_mean, row_inv_sd, m_eff) =
                rrblup_parallel_row_standardization(&maf, 1e-12_f32, None, mode);
            let raw_full = build_rrblup_exact_snp_raw_full_from_source(
                RrblupPcgSource::Resident {
                    packed_flat: &packed,
                    bytes_per_snp,
                },
                "resident:test".to_string(),
                n_samples,
                full_train_idx.clone(),
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                mode,
                1,
                1,
            )
            .expect("build raw_full");

            // 1. Fold subtraction (val_idx = 0..4)
            let sub_cache = build_rrblup_exact_snp_cache_from_raw_full(
                &raw_full,
                "resident:test",
                || {
                    Ok(RrblupPcgSource::Resident {
                        packed_flat: &packed,
                        bytes_per_snp,
                    })
                },
                n_samples,
                fold_train_idx.clone(),
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                mode,
                1,
                1,
            )
            .expect("build sub_cache");
            let direct_cache = build_rrblup_exact_snp_cache_from_source(
                RrblupPcgSource::Resident {
                    packed_flat: &packed,
                    bytes_per_snp,
                },
                n_samples,
                fold_train_idx.clone(),
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                1,
                1,
            )
            .expect("build direct_cache");

            assert_eq!(sub_cache.rank, direct_cache.rank);
            assert_eq!(sub_cache.n_eff, direct_cache.n_eff);
            for (a, b) in sub_cache
                .train_row_mean
                .iter()
                .zip(direct_cache.train_row_mean.iter())
            {
                assert!((a - b).abs() < 1e-6_f32);
            }
            for (a, b) in sub_cache.eigvals.iter().zip(direct_cache.eigvals.iter()) {
                assert!((a - b).abs() < 1e-10_f64);
            }

            let (_, pred_test_sub, pve_sub, lambda_sub, ..) =
                fit_rrblup_exact_snp_from_cache_source(
                    &sub_cache,
                    Some(&packed),
                    "",
                    bytes_per_snp,
                    n_samples,
                    y_fold.clone(),
                    test_idx.clone(),
                    None,
                    -6.0,
                    6.0,
                    1e-6,
                    50,
                    1,
                    1,
                )
                .expect("fit sub_cache");
            let (_, pred_test_dir, pve_dir, lambda_dir, ..) =
                fit_rrblup_exact_snp_from_cache_source(
                    &direct_cache,
                    Some(&packed),
                    "",
                    bytes_per_snp,
                    n_samples,
                    y_fold.clone(),
                    test_idx.clone(),
                    None,
                    -6.0,
                    6.0,
                    1e-6,
                    50,
                    1,
                    1,
                )
                .expect("fit direct_cache");
            assert!((lambda_sub - lambda_dir).abs() < 1e-8_f64);
            assert!((pve_sub - pve_dir).abs() < 1e-10_f64);
            for (a, b) in pred_test_sub.iter().zip(pred_test_dir.iter()) {
                assert!((a - b).abs() < 1e-10_f64);
            }

            // 2. Final full-train fit (val_idx empty)
            let full_sub_cache = build_rrblup_exact_snp_cache_from_raw_full(
                &raw_full,
                "resident:test",
                || {
                    Ok(RrblupPcgSource::Resident {
                        packed_flat: &packed,
                        bytes_per_snp,
                    })
                },
                n_samples,
                full_train_idx.clone(),
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                mode,
                1,
                1,
            )
            .expect("build full_sub_cache");
            let full_dir_cache = build_rrblup_exact_snp_cache_from_source(
                RrblupPcgSource::Resident {
                    packed_flat: &packed,
                    bytes_per_snp,
                },
                n_samples,
                full_train_idx.clone(),
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                1,
                1,
            )
            .expect("build full_dir_cache");
            let (_, pred_full_sub, _, lambda_full_sub, ..) =
                fit_rrblup_exact_snp_from_cache_source(
                    &full_sub_cache,
                    Some(&packed),
                    "",
                    bytes_per_snp,
                    n_samples,
                    y_full.clone(),
                    test_idx.clone(),
                    None,
                    -6.0,
                    6.0,
                    1e-6,
                    50,
                    1,
                    1,
                )
                .expect("fit full_sub");
            let (_, pred_full_dir, _, lambda_full_dir, ..) =
                fit_rrblup_exact_snp_from_cache_source(
                    &full_dir_cache,
                    Some(&packed),
                    "",
                    bytes_per_snp,
                    n_samples,
                    y_full.clone(),
                    test_idx.clone(),
                    None,
                    -6.0,
                    6.0,
                    1e-6,
                    50,
                    1,
                    1,
                )
                .expect("fit full_dir");
            assert!((lambda_full_sub - lambda_full_dir).abs() < 1e-10_f64);
            for (a, b) in pred_full_sub.iter().zip(pred_full_dir.iter()) {
                assert!((a - b).abs() < 1e-10_f64);
            }

            // 3. Reject mismatched genotype source fingerprint
            assert!(build_rrblup_exact_snp_cache_from_raw_full(
                &raw_full,
                "resident:other",
                || {
                    Ok(RrblupPcgSource::Resident {
                        packed_flat: &packed,
                        bytes_per_snp,
                    })
                },
                n_samples,
                fold_train_idx.clone(),
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                mode,
                1,
                1,
            )
            .is_err());
        }

        assert!(compute_val_complement_indices(&full_train_idx, &[0, 1, 18]).is_err());
    }

    #[test]
    fn rrblup_exact_cv_subtraction_falls_back_when_train_is_constant() {
        pyo3::Python::initialize();
        let n_samples = 16usize;
        let eff_m = 8usize;
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut packed = vec![0u8; eff_m * bytes_per_snp];
        // Samples 0..12 are constant 0b00; validation samples 12..16 vary.
        for r in 0..eff_m {
            for s in 12..16 {
                let code: u8 = if (r + s) % 2 == 0 { 0b11 } else { 0b10 };
                packed[r * bytes_per_snp + (s >> 2)] |= code << ((s & 3) * 2);
            }
        }
        let maf = vec![0.2_f32; eff_m];
        let row_flip = vec![false; eff_m];
        let full_train_idx: Vec<usize> = (0..16).collect();
        let fold_train_idx: Vec<usize> = (0..12).collect();
        for mode in [RrblupScaleMode::Standardized, RrblupScaleMode::Centered] {
            let (row_mean, row_inv_sd, m_eff) =
                rrblup_parallel_row_standardization(&maf, 1e-12_f32, None, mode);
            let raw_full = build_rrblup_exact_snp_raw_full_from_source(
                RrblupPcgSource::Resident {
                    packed_flat: &packed,
                    bytes_per_snp,
                },
                "resident:const".to_string(),
                n_samples,
                full_train_idx.clone(),
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                mode,
                1,
                1,
            )
            .expect("build raw_full");
            let sub_res = build_rrblup_exact_snp_cache_from_raw_full(
                &raw_full,
                "resident:const",
                || {
                    Ok(RrblupPcgSource::Resident {
                        packed_flat: &packed,
                        bytes_per_snp,
                    })
                },
                n_samples,
                fold_train_idx.clone(),
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                mode,
                1,
                1,
            );
            assert!(
                sub_res.is_err(),
                "constant train fold must fall back to direct build and reject zero spectrum"
            );
        }
    }

    #[test]
    fn rrblup_exact_cv_subtraction_rejects_missing_or_empty_source_fingerprint() {
        pyo3::Python::initialize();
        let n_samples = 20usize;
        let eff_m = 6usize;
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut packed = vec![0u8; eff_m * bytes_per_snp];
        for r in 0..eff_m {
            for s in 0..n_samples {
                let code: u8 = match (r * 7 + s * 3) % 4 {
                    0 => 0b00,
                    1 => 0b10,
                    2 => 0b11,
                    _ => 0b01,
                };
                packed[r * bytes_per_snp + (s >> 2)] |= code << ((s & 3) * 2);
            }
        }
        let maf = vec![0.25_f32; eff_m];
        let row_flip = vec![false; eff_m];
        let full_train_idx: Vec<usize> = (0..16).collect();
        let fold_train_idx: Vec<usize> = (4..16).collect();
        let mode = RrblupScaleMode::Standardized;
        let (row_mean, row_inv_sd, m_eff) =
            rrblup_parallel_row_standardization(&maf, 1e-12_f32, None, mode);

        // 1. Resident source rejects empty or whitespace-only fallback on build and reuse.
        for empty_fp in ["", "   "] {
            assert!(build_rrblup_exact_snp_raw_full_from_source(
                RrblupPcgSource::Resident {
                    packed_flat: &packed,
                    bytes_per_snp,
                },
                empty_fp.to_string(),
                n_samples,
                full_train_idx.clone(),
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                mode,
                1,
                1,
            )
            .is_err());
        }

        let resident_raw = build_rrblup_exact_snp_raw_full_from_source(
            RrblupPcgSource::Resident {
                packed_flat: &packed,
                bytes_per_snp,
            },
            "resident:valid".to_string(),
            n_samples,
            full_train_idx.clone(),
            row_flip.clone(),
            row_mean.clone(),
            row_inv_sd.clone(),
            None,
            4,
            eff_m,
            m_eff,
            mode,
            1,
            1,
        )
        .expect("build valid resident raw_full");
        for empty_fp in ["", "   "] {
            for target_idx in [fold_train_idx.clone(), full_train_idx.clone()] {
                assert!(build_rrblup_exact_snp_cache_from_raw_full(
                    &resident_raw,
                    empty_fp,
                    || {
                        Ok(RrblupPcgSource::Resident {
                            packed_flat: &packed,
                            bytes_per_snp,
                        })
                    },
                    n_samples,
                    target_idx,
                    row_flip.clone(),
                    row_mean.clone(),
                    row_inv_sd.clone(),
                    None,
                    4,
                    eff_m,
                    m_eff,
                    mode,
                    1,
                    1,
                )
                .is_err());
            }
        }

        // 2. Windowed BED source with source_fingerprint() == None rejects build and reuse.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_rrblup_fp_{stamp}"));
        std::fs::create_dir_all(&dir).expect("create temp BED dir");
        let prefix = dir.join("tiny");
        let mut bed_bytes = Vec::with_capacity(3 + packed.len());
        bed_bytes.extend_from_slice(&[0x6c, 0x1b, 0x01]);
        bed_bytes.extend_from_slice(&packed);
        std::fs::write(prefix.with_extension("bed"), &bed_bytes).expect("write BED");
        let fam_text: String = (0..n_samples)
            .map(|i| format!("F{i} I{i} 0 0 1 -9\n"))
            .collect();
        std::fs::write(prefix.with_extension("fam"), fam_text).expect("write FAM");
        let prefix_str = prefix.to_string_lossy().into_owned();

        let valid_matrix = WindowedBedMatrix::open(&prefix_str, 1).expect("open BED matrix");
        assert!(valid_matrix.source_fingerprint().is_some());
        let windowed_raw = build_rrblup_exact_snp_raw_full_from_source(
            RrblupPcgSource::Windowed {
                matrix: valid_matrix,
                bytes_per_snp,
            },
            String::new(),
            n_samples,
            full_train_idx.clone(),
            row_flip.clone(),
            row_mean.clone(),
            row_inv_sd.clone(),
            None,
            4,
            eff_m,
            m_eff,
            mode,
            1,
            1,
        )
        .expect("build valid windowed raw_full");
        assert!(!windowed_raw.source_fingerprint.is_empty());

        let mut untrusted_matrix =
            WindowedBedMatrix::open(&prefix_str, 1).expect("open BED matrix");
        untrusted_matrix.clear_source_fingerprint_for_test();
        assert_eq!(untrusted_matrix.source_fingerprint(), None);
        assert!(build_rrblup_exact_snp_raw_full_from_source(
            RrblupPcgSource::Windowed {
                matrix: untrusted_matrix,
                bytes_per_snp,
            },
            "fallback_must_not_override".to_string(),
            n_samples,
            full_train_idx.clone(),
            row_flip.clone(),
            row_mean.clone(),
            row_inv_sd.clone(),
            None,
            4,
            eff_m,
            m_eff,
            mode,
            1,
            1,
        )
        .is_err());

        for target_idx in [fold_train_idx.clone(), full_train_idx.clone()] {
            assert!(build_rrblup_exact_snp_cache_from_raw_full(
                &windowed_raw,
                "fallback_must_not_override",
                || {
                    let mut m = WindowedBedMatrix::open(&prefix_str, 1)?;
                    m.clear_source_fingerprint_for_test();
                    Ok(RrblupPcgSource::Windowed {
                        matrix: m,
                        bytes_per_snp,
                    })
                },
                n_samples,
                target_idx,
                row_flip.clone(),
                row_mean.clone(),
                row_inv_sd.clone(),
                None,
                4,
                eff_m,
                m_eff,
                mode,
                1,
                1,
            )
            .is_err());
        }

        std::fs::remove_dir_all(dir).expect("remove temp BED dir");
    }

    #[test]
    fn rrblup_dual_pcg_matches_primal_beta_and_predictions() {
        let n_samples = 40usize;
        let n_train = 32usize;
        let eff_m = 128usize;
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut packed = vec![0u8; eff_m * bytes_per_snp];
        for m in 0..eff_m {
            for b in 0..bytes_per_snp {
                let mut byte = 0u8;
                for k in 0..4 {
                    let s = b * 4 + k;
                    let code = match (m * 19 + s * 11 + (m ^ s)) % 5 {
                        0 => 0b00,
                        1 | 2 => 0b10,
                        3 => 0b11,
                        _ => 0b01,
                    };
                    byte |= code << (2 * k);
                }
                packed[m * bytes_per_snp + b] = byte;
            }
        }
        let maf: Vec<f32> = (0..eff_m)
            .map(|m| 0.1 + 0.35 * (((m * 17) % 100) as f32 / 100.0))
            .collect();
        let row_flip = vec![false; eff_m];
        let (row_mean, row_inv_sd, _m_eff) = rrblup_parallel_row_standardization(
            &maf,
            1e-12_f32,
            None,
            RrblupScaleMode::Standardized,
        );
        let train_idx: Vec<usize> = (0..n_train).collect();
        let test_idx: Vec<usize> = (n_train..n_samples).collect();
        let y_train: Vec<f64> = (0..n_train)
            .map(|i| ((i as f64) * 0.31).sin() + 0.4 * ((i as f64) * 0.17).cos())
            .collect();
        let y_mean = y_train.iter().sum::<f64>() / (n_train as f64);
        let y_center: Vec<f32> = y_train.iter().map(|&v| (v - y_mean) as f32).collect();
        let lambda_use = 15.0_f32;
        let code4_lut = packed_byte_lut().code4;
        let stage_accum = Mutex::new(PcgStageTimingAccum::default());

        let mut prep_src = RrblupPcgSource::Resident {
            packed_flat: &packed,
            bytes_per_snp,
        };
        let (b, diag_inv, train_row_mean, sum_ss_primal, _, _, _) = rrblup_prepare_rhs_diag(
            &mut prep_src,
            n_samples,
            &row_flip,
            &row_mean,
            &row_inv_sd,
            &train_idx,
            false,
            None,
            32,
            &y_center,
            lambda_use,
            &code4_lut,
            None,
        )
        .expect("primal prepare");
        let primal_op = RrblupPcgOperator {
            source: RrblupPcgSource::Resident {
                packed_flat: &packed,
                bytes_per_snp,
            },
            n_samples,
            n_train,
            row_flip: &row_flip,
            row_mean: &row_mean,
            row_inv_sd: &row_inv_sd,
            train_row_mean: &train_row_mean,
            sample_idx: &train_idx,
            full_sample_fast: false,
            packed_row_indices: None,
            block_rows: 32,
            code4_lut: &code4_lut,
            pool: None,
            lambda_use,
            stage_timing: false,
            stage_accum: &stage_accum,
        };
        let primal_res = pcg_solve(
            &b,
            None,
            200,
            1e-6_f64,
            1e-20_f64,
            primal_op,
            DiagonalPreconditioner::new(&diag_inv),
            |_, _, _| Ok(()),
        )
        .expect("primal pcg");
        assert!(primal_res.converged);
        let primal_alpha = (y_mean as f32)
            - train_row_mean
                .iter()
                .zip(primal_res.x.iter())
                .map(|(mu, bj)| *mu * *bj)
                .sum::<f32>();
        let mut pred_src = RrblupPcgSource::Resident {
            packed_flat: &packed,
            bytes_per_snp,
        };
        let primal_pred_train: Vec<f64> = pcg_x_mul_samples(
            &mut pred_src,
            n_samples,
            &row_flip,
            &row_mean,
            &row_inv_sd,
            &train_idx,
            false,
            None,
            32,
            &primal_res.x,
            &code4_lut,
            None,
            None,
        )
        .expect("primal pred train")
        .into_iter()
        .map(|v| (v + primal_alpha) as f64)
        .collect();
        let primal_pred_test: Vec<f64> = pcg_x_mul_samples(
            &mut pred_src,
            n_samples,
            &row_flip,
            &row_mean,
            &row_inv_sd,
            &test_idx,
            false,
            None,
            32,
            &primal_res.x,
            &code4_lut,
            None,
            None,
        )
        .expect("primal pred test")
        .into_iter()
        .map(|v| (v + primal_alpha) as f64)
        .collect();

        let mut sum_ss_dual = 0.0_f64;
        let mut sum_ss_captured = false;
        let dual_op = RrblupDualPcgOperator {
            source: RrblupPcgSource::Resident {
                packed_flat: &packed,
                bytes_per_snp,
            },
            n_samples,
            n_train,
            m: eff_m,
            row_flip: &row_flip,
            row_mean: &row_mean,
            row_inv_sd: &row_inv_sd,
            sample_idx: &train_idx,
            full_sample_fast: false,
            packed_row_indices: None,
            block_rows: 32,
            code4_lut: &code4_lut,
            pool: None,
            lambda_use,
            stage_timing: false,
            stage_accum: &stage_accum,
            block_buf: vec![0.0_f32; 32 * n_train],
            tmp_buf: vec![0.0_f32; 32],
            q_buf: vec![0.0_f32; n_train],
            sum_ss_global: &mut sum_ss_dual,
            sum_ss_captured: &mut sum_ss_captured,
        };
        let dual_res = pcg_solve(
            &y_center,
            None,
            200,
            1e-6_f64,
            1e-20_f64,
            dual_op,
            IdentityPreconditioner,
            |_, _, _| Ok(()),
        )
        .expect("dual pcg");
        assert!(dual_res.converged);
        assert!(sum_ss_captured);
        assert!((sum_ss_dual - sum_ss_primal).abs() < 1e-2);

        let a_mean = (dual_res.x.iter().map(|&v| v as f64).sum::<f64>() / (n_train as f64)) as f32;
        let alpha_c: Vec<f32> = dual_res.x.iter().map(|&v| v - a_mean).collect();
        let dual_pred_train: Vec<f64> = y_train
            .iter()
            .zip(alpha_c.iter())
            .map(|(&yi, &ai)| yi - (lambda_use as f64) * (ai as f64))
            .collect();
        let mut rec_src = RrblupPcgSource::Resident {
            packed_flat: &packed,
            bytes_per_snp,
        };
        let (dual_beta, _dual_alpha, dual_pred_test, _) = rrblup_dual_recover_beta_and_test(
            &mut rec_src,
            n_samples,
            &row_flip,
            &row_mean,
            &row_inv_sd,
            &train_idx,
            false,
            &test_idx,
            false,
            None,
            32,
            &alpha_c,
            y_mean,
            false,
            &code4_lut,
            None,
        )
        .expect("dual recover");

        assert!(max_abs_diff(&dual_beta, &primal_res.x) < 1e-4);
        let max_tr_diff = dual_pred_train
            .iter()
            .zip(primal_pred_train.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_te_diff = dual_pred_test
            .iter()
            .zip(primal_pred_test.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_tr_diff < 1e-4, "train diff={max_tr_diff}");
        assert!(max_te_diff < 1e-4, "test diff={max_te_diff}");
    }

    #[test]
    fn rrblup_dual_pcg_unconverged_max_iter_one_matches_beta_predictions() {
        let n_samples = 32usize;
        let n_train = 26usize;
        let eff_m = 96usize;
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut packed = vec![0u8; eff_m * bytes_per_snp];
        for m in 0..eff_m {
            for b in 0..bytes_per_snp {
                let mut byte = 0u8;
                for k in 0..4 {
                    let s = b * 4 + k;
                    let code = match (m * 29 + s * 13 + (m ^ s)) % 5 {
                        0 => 0b00,
                        1 | 2 => 0b10,
                        3 => 0b11,
                        _ => 0b01,
                    };
                    byte |= code << (2 * k);
                }
                packed[m * bytes_per_snp + b] = byte;
            }
        }
        let maf: Vec<f32> = (0..eff_m)
            .map(|m| 0.12 + 0.32 * (((m * 11) % 100) as f32 / 100.0))
            .collect();
        let row_flip = vec![false; eff_m];
        let (row_mean, row_inv_sd, _m_eff) = rrblup_parallel_row_standardization(
            &maf,
            1e-12_f32,
            None,
            RrblupScaleMode::Standardized,
        );
        let train_idx: Vec<usize> = (0..n_train).collect();
        let y_train: Vec<f64> = (0..n_train)
            .map(|i| ((i as f64) * 0.43).sin() + 0.5 * ((i as f64) * 0.19).cos())
            .collect();
        let y_mean = y_train.iter().sum::<f64>() / (n_train as f64);
        let y_center: Vec<f32> = y_train.iter().map(|&v| (v - y_mean) as f32).collect();
        let lambda_use = 5.0_f32;
        let code4_lut = packed_byte_lut().code4;
        let stage_accum = Mutex::new(PcgStageTimingAccum::default());

        let mut sum_ss = 0.0_f64;
        let mut sum_ss_captured = false;
        let dual_op = RrblupDualPcgOperator {
            source: RrblupPcgSource::Resident {
                packed_flat: &packed,
                bytes_per_snp,
            },
            n_samples,
            n_train,
            m: eff_m,
            row_flip: &row_flip,
            row_mean: &row_mean,
            row_inv_sd: &row_inv_sd,
            sample_idx: &train_idx,
            full_sample_fast: false,
            packed_row_indices: None,
            block_rows: 32,
            code4_lut: &code4_lut,
            pool: None,
            lambda_use,
            stage_timing: false,
            stage_accum: &stage_accum,
            block_buf: Vec::new(),
            tmp_buf: vec![0.0_f32; 32],
            q_buf: vec![0.0_f32; n_train],
            sum_ss_global: &mut sum_ss,
            sum_ss_captured: &mut sum_ss_captured,
        };
        let pcg_res = pcg_solve(
            &y_center,
            None,
            1,
            1e-12_f64,
            1e-20_f64,
            dual_op,
            IdentityPreconditioner,
            |_, _, _| Ok(()),
        )
        .expect("max_iter=1 dual pcg");
        assert!(!pcg_res.converged);

        let a_mean = (pcg_res.x.iter().map(|&v| v as f64).sum::<f64>() / (n_train as f64)) as f32;
        let alpha_c: Vec<f32> = pcg_res.x.iter().map(|&v| v - a_mean).collect();
        let mut rec_src = RrblupPcgSource::Resident {
            packed_flat: &packed,
            bytes_per_snp,
        };
        let (beta, alpha_use, _, _) = rrblup_dual_recover_beta_and_test(
            &mut rec_src,
            n_samples,
            &row_flip,
            &row_mean,
            &row_inv_sd,
            &train_idx,
            false,
            &[],
            false,
            None,
            32,
            &alpha_c,
            y_mean,
            false,
            &code4_lut,
            None,
        )
        .expect("recover beta");

        let (pred_train_fixed, pve_fixed) = rrblup_dual_finalize_train_predictions(
            Some(&packed),
            bytes_per_snp,
            "",
            n_samples,
            &row_flip,
            &row_mean,
            &row_inv_sd,
            &train_idx,
            false,
            None,
            None,
            32,
            &y_train,
            &alpha_c,
            lambda_use,
            &beta,
            alpha_use,
            pcg_res.converged,
            true,
            &code4_lut,
            None,
        )
        .expect("finalize unconverged train predictions");
        let resid_check: Vec<f64> = y_train
            .iter()
            .zip(pred_train_fixed.iter())
            .map(|(&y, &p)| y - p)
            .collect();
        let var_g_check = sample_var_f64(&pred_train_fixed, None);
        let var_e_check = sample_var_f64(&resid_check, None);
        let expected_pve = var_g_check / (var_g_check + var_e_check);
        assert!(pve_fixed.is_finite());
        assert!(
            (pve_fixed - expected_pve).abs() < 1e-12,
            "unconverged pve mismatch: got {pve_fixed}, expected {expected_pve}"
        );

        let mut check_src = RrblupPcgSource::Resident {
            packed_flat: &packed,
            bytes_per_snp,
        };
        let pred_from_beta: Vec<f64> = pcg_x_mul_samples(
            &mut check_src,
            n_samples,
            &row_flip,
            &row_mean,
            &row_inv_sd,
            &train_idx,
            false,
            None,
            32,
            &beta,
            &code4_lut,
            None,
            None,
        )
        .expect("direct train pred from beta")
        .into_iter()
        .map(|v| (v + alpha_use) as f64)
        .collect();
        let max_beta_diff = pred_train_fixed
            .iter()
            .zip(pred_from_beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_beta_diff < 1e-5,
            "unconverged train predictions diverged from W*beta: {max_beta_diff}"
        );
    }

    #[test]
    fn rrblup_dual_pcg_windowed_double_buffer_and_error_propagation() {
        let n_samples = 24usize;
        let eff_m = 96usize;
        let block_rows = 32usize;
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut packed = vec![0u8; eff_m * bytes_per_snp];
        for m in 0..eff_m {
            for b in 0..bytes_per_snp {
                let mut byte = 0u8;
                for k in 0..4 {
                    let s = b * 4 + k;
                    let code = match (m * 17 + s * 9 + (m ^ s)) % 5 {
                        0 => 0b00,
                        1 | 2 => 0b10,
                        3 => 0b11,
                        _ => 0b01,
                    };
                    byte |= code << (2 * k);
                }
                packed[m * bytes_per_snp + b] = byte;
            }
        }
        let maf: Vec<f32> = (0..eff_m)
            .map(|m| 0.1 + 0.35 * (((m * 13) % 100) as f32 / 100.0))
            .collect();
        let row_flip = vec![false; eff_m];
        let (row_mean, row_inv_sd, _m_eff) = rrblup_parallel_row_standardization(
            &maf,
            1e-12_f32,
            None,
            RrblupScaleMode::Standardized,
        );
        let train_idx: Vec<usize> = (0..n_samples).collect();
        let y_train: Vec<f64> = (0..n_samples)
            .map(|i| ((i as f64) * 0.33).sin() - 0.25 * ((i as f64) * 0.21).cos())
            .collect();
        let y_mean = y_train.iter().sum::<f64>() / (n_samples as f64);
        let y_center: Vec<f32> = y_train.iter().map(|&v| (v - y_mean) as f32).collect();
        let lambda_use = 12.0_f32;
        let code4_lut = packed_byte_lut().code4;
        let stage_accum = Mutex::new(PcgStageTimingAccum::default());

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_dual_db_{stamp}"));
        std::fs::create_dir_all(&dir).expect("create temp BED dir");
        let prefix = dir.join("tiny");
        let mut bed_bytes = Vec::with_capacity(3 + packed.len());
        bed_bytes.extend_from_slice(&[0x6c, 0x1b, 0x01]);
        bed_bytes.extend_from_slice(&packed);
        std::fs::write(prefix.with_extension("bed"), &bed_bytes).expect("write BED");
        let fam_text: String = (0..n_samples)
            .map(|i| format!("F{i} I{i} 0 0 1 -9\n"))
            .collect();
        std::fs::write(prefix.with_extension("fam"), fam_text).expect("write FAM");
        let prefix_str = prefix.to_string_lossy().into_owned();

        let mut sum_ss_res = 0.0_f64;
        let mut cap_res = false;
        let res_op = RrblupDualPcgOperator {
            source: RrblupPcgSource::Resident {
                packed_flat: &packed,
                bytes_per_snp,
            },
            n_samples,
            n_train: n_samples,
            m: eff_m,
            row_flip: &row_flip,
            row_mean: &row_mean,
            row_inv_sd: &row_inv_sd,
            sample_idx: &train_idx,
            full_sample_fast: true,
            packed_row_indices: None,
            block_rows,
            code4_lut: &code4_lut,
            pool: None,
            lambda_use,
            stage_timing: false,
            stage_accum: &stage_accum,
            block_buf: Vec::new(),
            tmp_buf: vec![0.0_f32; block_rows],
            q_buf: vec![0.0_f32; n_samples],
            sum_ss_global: &mut sum_ss_res,
            sum_ss_captured: &mut cap_res,
        };
        let res_sol = pcg_solve(
            &y_center,
            None,
            200,
            1e-6_f64,
            1e-20_f64,
            res_op,
            IdentityPreconditioner,
            |_, _, _| Ok(()),
        )
        .expect("resident dual pcg");

        let win_mat = WindowedBedMatrix::open(&prefix_str, 1).expect("open windowed BED");
        let mut sum_ss_win = 0.0_f64;
        let mut cap_win = false;
        let win_op = RrblupDualPcgOperator {
            source: RrblupPcgSource::Windowed {
                matrix: win_mat,
                bytes_per_snp,
            },
            n_samples,
            n_train: n_samples,
            m: eff_m,
            row_flip: &row_flip,
            row_mean: &row_mean,
            row_inv_sd: &row_inv_sd,
            sample_idx: &train_idx,
            full_sample_fast: true,
            packed_row_indices: None,
            block_rows,
            code4_lut: &code4_lut,
            pool: None,
            lambda_use,
            stage_timing: false,
            stage_accum: &stage_accum,
            block_buf: Vec::new(),
            tmp_buf: vec![0.0_f32; block_rows],
            q_buf: vec![0.0_f32; n_samples],
            sum_ss_global: &mut sum_ss_win,
            sum_ss_captured: &mut cap_win,
        };
        let win_sol = pcg_solve(
            &y_center,
            None,
            200,
            1e-6_f64,
            1e-20_f64,
            win_op,
            IdentityPreconditioner,
            |_, _, _| Ok(()),
        )
        .expect("windowed double-buffer dual pcg");
        assert!(win_sol.converged);
        assert!(cap_win);
        assert!((sum_ss_win - sum_ss_res).abs() < 1e-6);
        assert!(max_abs_diff(&win_sol.x, &res_sol.x) < 1e-6);

        // Verify decode error inside the double-buffer producer propagates cleanly.
        let win_mat_err = WindowedBedMatrix::open(&prefix_str, 1).expect("open windowed BED err");
        let mut bad_rows: Vec<usize> = (0..eff_m).collect();
        bad_rows[eff_m - 1] = eff_m + 10;
        let mut sum_ss_err = 0.0_f64;
        let mut cap_err = false;
        let mut err_op = RrblupDualPcgOperator {
            source: RrblupPcgSource::Windowed {
                matrix: win_mat_err,
                bytes_per_snp,
            },
            n_samples,
            n_train: n_samples,
            m: eff_m,
            row_flip: &row_flip,
            row_mean: &row_mean,
            row_inv_sd: &row_inv_sd,
            sample_idx: &train_idx,
            full_sample_fast: true,
            packed_row_indices: Some(&bad_rows),
            block_rows,
            code4_lut: &code4_lut,
            pool: None,
            lambda_use,
            stage_timing: false,
            stage_accum: &stage_accum,
            block_buf: Vec::new(),
            tmp_buf: vec![0.0_f32; block_rows],
            q_buf: vec![0.0_f32; n_samples],
            sum_ss_global: &mut sum_ss_err,
            sum_ss_captured: &mut cap_err,
        };
        let mut out_err = vec![0.0_f32; n_samples];
        let err_msg = err_op
            .apply(&y_center, &mut out_err)
            .expect_err("expected out-of-bounds decode error");
        assert!(
            err_msg.contains("out of bounds"),
            "unexpected double-buffer error message: {err_msg}"
        );
        drop(err_op);

        std::fs::remove_dir_all(dir).expect("remove temp BED dir");
    }
}
