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
    rust_sgemm_prefers_rayon_rowmajor_f32_kernel, CblasInt, OpenBlasThreadGuard, CBLAS_COL_MAJOR,
    CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_UPPER,
};
use crate::brent::brent_minimize;
use crate::decode::decode_prepared_additive_block_packed_f32;
use crate::eigh::symmetric_eigh_f64_row_major;
use crate::gload::WindowedBedMatrix;
use crate::packed::bed_packed_row_flip_mask;
use crate::pcg::{pcg_solve, DiagonalPreconditioner, PcgOperator};
use crate::pipeline::run_double_buffer;
use crate::stats_common::{
    check_ctrlc, env_truthy, get_cached_pool, map_err_string_to_py, parse_index_vec_i64,
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
    // `rows` here is the decoded SNP block height, not the sample count. In the
    // PCG BED path it is driven by `block_rows`/`--memory`, so a "small rows,
    // very wide cols" front-half window can still occur on large-sample folds.
    // Keep a dedicated column-chunk fast path for that shape, and otherwise
    // prefer the explicit Rayon reduction kernel over dropping back to serial
    // BLAS under the default `rayon_parallel_blas_serial` thread policy.
    if cols >= 65_536 && rows <= 4_096 {
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
fn row_major_block_prepare_rhs_diag_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    dot_blk: &[f32],
    lambda_use: f32,
    b_out: &mut [f32],
    diag_inv_out: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> f64 {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert!(dot_blk.len() >= rows);
    debug_assert!(b_out.len() >= rows);
    debug_assert!(diag_inv_out.len() >= rows);

    if prefer_parallel_block_vec(rows, cols, pool) {
        let mut run = || {
            block
                .par_chunks(cols)
                .zip(dot_blk[..rows].par_iter())
                .zip(
                    b_out[..rows]
                        .par_iter_mut()
                        .zip(diag_inv_out[..rows].par_iter_mut()),
                )
                .map(|((row, dot), (b_slot, diag_slot))| {
                    let ss = row_major_row_sumsq_f64(row);
                    *b_slot = *dot;
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
        let ss = row_major_row_sumsq_f64(row);
        sum_ss += ss;
        b_out[r] = dot_blk[r];
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

#[inline]
fn rrblup_parallel_row_standardization(
    maf_keep: &[f32],
    std_eps32: f32,
    pool: Option<&Arc<rayon::ThreadPool>>,
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
                    *inv_slot = 1.0_f32 / var.sqrt();
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
                row_inv_sd[j] = 1.0_f32 / var.sqrt();
                count += 1;
            } else {
                row_inv_sd[j] = 0.0_f32;
            }
        }
        count
    };
    (row_mean, row_inv_sd, m_effective)
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
) -> Result<(Vec<f32>, Vec<f32>, f64, f64, f64, f64), String> {
    let m = row_flip.len();
    let n_train = sample_idx.len();
    let row_step = block_rows.max(1).min(m.max(1));
    let mut b = vec![0.0_f32; m];
    let mut diag_inv = vec![0.0_f32; m];
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

    Ok((b, diag_inv, sum_ss_global, decode_acc, rhs_acc, diag_acc))
}

struct RrblupPcgOperator<'a> {
    source: RrblupPcgSource<'a>,
    n_samples: usize,
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
        if let Some(tp) = self.pool {
            tp.install(|| {
                ap.par_iter_mut()
                    .zip(input.par_iter())
                    .for_each(|(apj, pj)| *apj += self.lambda_use * *pj);
            });
        } else {
            for j in 0..ap.len() {
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
        if let Some(tp) = pool { tp.install(run) } else { run() }
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
        if let Some(tp) = pool { tp.install(run) } else { run() }
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

#[inline]
fn row_major_block_row_sum_and_cast_f64(
    block_f32: &[f32],
    rows: usize,
    cols: usize,
    row_sum_accum: &mut [f64],
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
            row_sum_accum[r] += row_sum_local[r];
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
        row_sum_accum[r] += acc;
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
    let pool_owned = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let pool_ref = pool_owned.as_ref();
    let code4_lut = packed_byte_lut().code4;
    let mut a_star = vec![0.0_f64; eff_m * eff_m];
    let mut row_sum = vec![0.0_f64; eff_m];
    let mut block_f32 = vec![0.0_f32; eff_m * sample_block_use];
    let mut block_f64 = vec![0.0_f64; eff_m * sample_block_use];
    let mut build_a_decode_secs = 0.0_f64;
    let mut build_a_cast_secs = 0.0_f64;
    let mut build_a_rankk_secs = 0.0_f64;
    let blas_threads_effective = if blas_threads > 0 {
        blas_threads.max(1)
    } else {
        1usize
    };
    let _blas_guard = OpenBlasThreadGuard::enter(blas_threads_effective);
    let use_chunked_stream_decode = matches!(source, RrblupPcgSource::Windowed { .. })
        && stream_row_block < eff_m;

    for cst in (0..n_train).step_by(sample_block_use.max(1)) {
        let ced = (cst + sample_block_use).min(n_train);
        let cur_n = ced - cst;
        let blk_slice_f32 = &mut block_f32[..eff_m * cur_n];
        let t_decode = if stage_timing {
            Some(Instant::now())
        } else {
            None
        };
        if use_chunked_stream_decode {
            decode_standardized_block_rows_chunked_f32(
                &mut source,
                n_samples,
                row_flip_keep.as_slice(),
                row_mean.as_slice(),
                row_inv_sd.as_slice(),
                &train_idx[cst..ced],
                is_identity_indices(&train_idx[cst..ced], n_samples),
                packed_row_indices.as_deref(),
                stream_row_block,
                blk_slice_f32,
                &code4_lut,
                pool_ref,
            )?;
        } else {
            decode_standardized_block_f32(
                &mut source,
                n_samples,
                row_flip_keep.as_slice(),
                row_mean.as_slice(),
                row_inv_sd.as_slice(),
                &train_idx[cst..ced],
                is_identity_indices(&train_idx[cst..ced], n_samples),
                packed_row_indices.as_deref(),
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
            row_sum.as_mut_slice(),
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
                1.0_f64,
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

    let t_center = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let center_scale = 1.0_f64 / (n_train as f64);
    symmetrize_upper_minus_rank1_in_place(&mut a_star, eff_m, &row_sum, center_scale);
    let center_secs = t_center
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    let t_eigh = if stage_timing {
        Some(Instant::now())
    } else {
        None
    };
    let (evals_all, evecs_all, eig_backend) = symmetric_eigh_f64_row_major(&a_star, eff_m)?;
    let eigh_secs = t_eigh
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    let max_eval = evals_all.last().copied().unwrap_or(0.0_f64).max(0.0_f64);
    let tol = f64::EPSILON * max_eval.max(1.0_f64) * (eff_m.max(1) as f64);
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
    let mut evecs = vec![0.0_f64; eff_m * rank];
    for i in 0..eff_m {
        let src = &evecs_all[i * eff_m + keep_start..i * eff_m + eff_m];
        let dst = &mut evecs[i * rank..(i + 1) * rank];
        dst.copy_from_slice(src);
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
        eigvals,
        evecs,
        eig_backend: eig_backend.to_string(),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_rrblup_exact_snp_cache_from_packed(
    packed_flat: &[u8],
    bytes_per_snp: usize,
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
    build_rrblup_exact_snp_cache_from_source(
        RrblupPcgSource::Resident {
            packed_flat,
            bytes_per_snp,
        },
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
    let pred_block_rows = 1024usize.min(cache.m.max(1));
    let var_g = if cache.n_train > 1 {
        g_center_ss / ((cache.n_train - 1) as f64)
    } else {
        0.0_f64
    };
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
                .map(|v| (v as f64) + y_mean)
                .collect()
        }
    } else {
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
            .map(|v| (v as f64) + y_mean)
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
        pred_test_ret = pred_test_f32.iter().map(|v| (*v as f64) + y_mean).collect();
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
        beta_f32,
        cache.eig_backend.clone(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn fit_rrblup_exact_snp_from_cache_packed(
    cache: &RrblupExactSnpPreparedInner,
    packed_flat: &[u8],
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
        Vec<f32>,
        String,
    ),
    String,
> {
    fit_rrblup_exact_snp_from_cache_source(
        cache,
        Some(packed_flat),
        "",
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
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    train_sample_indices,
    site_keep=None,
    maf=None,
    row_flip=None,
    sample_block=2048,
    std_eps=1e-12_f64,
    threads=0,
    blas_threads=0
))]
pub fn rrblup_exact_snp_prepare_packed<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    sample_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
) -> PyResult<RrblupExactSnpCache> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(std_eps.is_finite() && std_eps > 0.0) {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_packed requires finite std_eps > 0.",
        ));
    }
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp).",
        ));
    }
    let m_total = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let maf_ro = maf.ok_or_else(|| {
        PyRuntimeError::new_err("rrblup_exact_snp_prepare_packed requires `maf` argument.")
    })?;
    let row_flip_ro = row_flip.ok_or_else(|| {
        PyRuntimeError::new_err("rrblup_exact_snp_prepare_packed requires `row_flip` argument.")
    })?;
    let maf_full: Vec<f32> = match maf_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => maf_ro.as_array().iter().copied().collect(),
    };
    let row_flip_full: Vec<bool> = match row_flip_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip_ro.as_array().iter().copied().collect(),
    };
    if maf_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "maf length mismatch: got {}, expected {m_total}",
            maf_full.len()
        )));
    }
    if row_flip_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m_total}",
            row_flip_full.len()
        )));
    }
    let mut eff_m = m_total;
    let mut maf_keep: Cow<[f32]> = Cow::Borrowed(maf_full.as_slice());
    let mut row_flip_keep: Cow<[bool]> = Cow::Borrowed(row_flip_full.as_slice());
    let mut packed_row_indices: Option<Vec<usize>> = None;
    if let Some(mask) = site_keep {
        let mask_vec: Vec<bool> = match mask.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mask.as_array().iter().copied().collect(),
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
    if eff_m == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_packed received zero active markers.",
        ));
    }
    let train_idx = parse_index_vec_i64(
        train_sample_indices.as_slice()?,
        n_samples,
        "train_sample_indices",
    )?;
    if train_idx.len() <= 1 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_packed requires at least two training samples.",
        ));
    }
    let std_eps32 = std_eps.max(1e-12) as f32;
    let mut row_mean = vec![0.0_f32; eff_m];
    let mut row_inv_sd = vec![0.0_f32; eff_m];
    let mut m_effective = 0usize;
    for j in 0..eff_m {
        let p = maf_keep[j].clamp(0.0, 0.5);
        let mean = 2.0_f32 * p;
        let var = (2.0_f32 * p * (1.0_f32 - p)).max(0.0);
        row_mean[j] = mean;
        if var > std_eps32 {
            row_inv_sd[j] = 1.0_f32 / var.sqrt();
            m_effective += 1;
        } else {
            row_inv_sd[j] = 0.0_f32;
        }
    }
    if m_effective == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_packed found zero effective markers after standardization.",
        ));
    }
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let sample_block_use = sample_block.max(1).min(train_idx.len().max(1));
    let stream_row_block = eff_m.max(1);
    let cache_inner = py
        .detach(move || {
            build_rrblup_exact_snp_cache_from_packed(
                packed_flat.as_ref(),
                bytes_per_snp,
                n_samples,
                train_idx,
                row_flip_keep.into_owned(),
                row_mean,
                row_inv_sd,
                packed_row_indices,
                sample_block_use,
                stream_row_block,
                m_effective,
                threads,
                blas_threads,
            )
        })
        .map_err(map_err_string_to_py)?;
    Ok(RrblupExactSnpCache {
        inner: Arc::new(cache_inner),
    })
}

#[pyfunction]
#[pyo3(signature = (
    cache,
    packed,
    n_samples,
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
pub fn rrblup_exact_snp_fit_prepared<'py>(
    py: Python<'py>,
    cache: PyRef<'py, RrblupExactSnpCache>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
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
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(log10_lambda_low.is_finite() && log10_lambda_high.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_fit_prepared requires finite log10(lambda) bounds.",
        ));
    }
    if !(reml_tol.is_finite() && reml_tol > 0.0) {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_fit_prepared requires finite reml_tol > 0.",
        ));
    }
    if reml_max_iter == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_fit_prepared requires reml_max_iter > 0.",
        ));
    }
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp).",
        ));
    }
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let y_vec_f64: Vec<f64> = match y_train.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y_train.as_array().iter().copied().collect(),
    };
    let test_idx = if let Some(sidx) = test_sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "test_sample_indices")?
    } else {
        Vec::new()
    };
    let train_pred_pick = if let Some(local_idx) = train_pred_local_indices {
        Some(parse_index_vec_i64(
            local_idx.as_slice()?,
            cache.inner.n_train,
            "train_pred_local_indices",
        )?)
    } else {
        None
    };
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let cache_inner = Arc::clone(&cache.inner);
    let row_mean_ret = cache_inner.row_mean.clone();
    let row_inv_sd_ret = cache_inner.row_inv_sd.clone();
    let m_effective = cache_inner.m_effective;
    let (
        pred_train_ret,
        pred_test_ret,
        pve_trainvar,
        lambda_opt,
        reml_opt,
        var_g,
        sigma_e2,
        beta_ret,
        eig_backend,
    ) = py
        .detach(move || {
            fit_rrblup_exact_snp_from_cache_packed(
                cache_inner.as_ref(),
                packed_flat.as_ref(),
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
    let alpha = y_train.as_slice()?.iter().copied().sum::<f64>() / (cache.inner.n_train as f64);
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
    source_prefix,
    train_sample_indices,
    row_source_indices,
    maf,
    row_flip,
    sample_block=2048,
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
    sample_block: usize,
    std_eps: f64,
    threads: usize,
    blas_threads: usize,
) -> PyResult<RrblupExactSnpCache> {
    if str::trim(source_prefix.as_str()).is_empty() {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_bed_from_meta requires non-empty source_prefix.",
        ));
    }
    if !(std_eps.is_finite() && std_eps > 0.0) {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_bed_from_meta requires finite std_eps > 0.",
        ));
    }
    let n_samples = crate::gfcore::read_fam(&source_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_bed_from_meta found zero samples in source_prefix.fam.",
        ));
    }
    let train_idx = parse_index_vec_i64(
        train_sample_indices.as_slice()?,
        n_samples,
        "train_sample_indices",
    )?;
    if train_idx.len() <= 1 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_bed_from_meta requires at least two training samples.",
        ));
    }
    let row_source_idx = parse_nonnegative_index_vec_i64(
        row_source_indices.as_slice()?,
        "row_source_indices",
    )
    .map_err(PyRuntimeError::new_err)?;
    let eff_m = row_source_idx.len();
    if eff_m == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_bed_from_meta received zero active markers.",
        ));
    }
    let maf_keep: Vec<f32> = match maf.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => maf.as_array().iter().copied().collect(),
    };
    let row_flip_keep: Vec<bool> = match row_flip.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip.as_array().iter().copied().collect(),
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
    let mut row_mean = vec![0.0_f32; eff_m];
    let mut row_inv_sd = vec![0.0_f32; eff_m];
    let mut m_effective = 0usize;
    for j in 0..eff_m {
        let p = maf_keep[j].clamp(0.0, 0.5);
        let mean = 2.0_f32 * p;
        let var = (2.0_f32 * p * (1.0_f32 - p)).max(0.0);
        row_mean[j] = mean;
        if var > std_eps32 {
            row_inv_sd[j] = 1.0_f32 / var.sqrt();
            m_effective += 1;
        } else {
            row_inv_sd[j] = 0.0_f32;
        }
    }
    if m_effective == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_prepare_bed_from_meta found zero effective markers after standardization.",
        ));
    }
    let bytes_per_snp = n_samples.div_ceil(4);
    let sample_block_use = sample_block.max(1).min(train_idx.len().max(1));
    let stream_row_block = sample_block_use.max(1).min(eff_m.max(1));
    let cache_inner = py
        .detach(move || {
            let source = build_rrblup_source(
                None,
                bytes_per_snp,
                source_prefix.as_str(),
                n_samples,
                stream_row_block,
            )?;
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
        })
        .map_err(map_err_string_to_py)?;
    Ok(RrblupExactSnpCache {
        inner: Arc::new(cache_inner),
    })
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
    if n_samples != cache.inner.n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "rrblup exact SNP bed cache sample-count mismatch: got {n_samples}, expected {}",
            cache.inner.n_samples
        )));
    }
    let y_vec_f64: Vec<f64> = match y_train.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y_train.as_array().iter().copied().collect(),
    };
    let test_idx = if let Some(sidx) = test_sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "test_sample_indices")?
    } else {
        Vec::new()
    };
    let train_pred_pick = if let Some(local_idx) = train_pred_local_indices {
        Some(parse_index_vec_i64(
            local_idx.as_slice()?,
            cache.inner.n_train,
            "train_pred_local_indices",
        )?)
    } else {
        None
    };
    let cache_inner = Arc::clone(&cache.inner);
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
    let alpha = y_train.as_slice()?.iter().copied().sum::<f64>() / (cache.inner.n_train as f64);
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
    packed,
    n_samples,
    train_sample_indices,
    y_train,
    test_sample_indices=None,
    train_pred_local_indices=None,
    site_keep=None,
    maf=None,
    row_flip=None,
    log10_lambda_low=-6.0_f64,
    log10_lambda_high=6.0_f64,
    reml_tol=1e-4_f64,
    reml_max_iter=50,
    sample_block=2048,
    std_eps=1e-12_f64,
    threads=0,
    blas_threads=0
))]
pub fn rrblup_exact_snp_packed<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    y_train: PyReadonlyArray1<'py, f64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    train_pred_local_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    log10_lambda_low: f64,
    log10_lambda_high: f64,
    reml_tol: f64,
    reml_max_iter: usize,
    sample_block: usize,
    std_eps: f64,
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
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(log10_lambda_low.is_finite() && log10_lambda_high.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_packed requires finite log10(lambda) bounds.",
        ));
    }
    if !(reml_tol.is_finite() && reml_tol > 0.0) {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_packed requires finite reml_tol > 0.",
        ));
    }
    if reml_max_iter == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_packed requires reml_max_iter > 0.",
        ));
    }
    if !(std_eps.is_finite() && std_eps > 0.0) {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_packed requires finite std_eps > 0.",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp).",
        ));
    }
    let m_total = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let maf_ro = maf.ok_or_else(|| {
        PyRuntimeError::new_err("rrblup_exact_snp_packed requires `maf` argument.")
    })?;
    let row_flip_ro = row_flip.ok_or_else(|| {
        PyRuntimeError::new_err("rrblup_exact_snp_packed requires `row_flip` argument.")
    })?;
    let maf_full: Vec<f32> = match maf_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => maf_ro.as_array().iter().copied().collect(),
    };
    let row_flip_full: Vec<bool> = match row_flip_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip_ro.as_array().iter().copied().collect(),
    };
    if maf_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "maf length mismatch: got {}, expected {m_total}",
            maf_full.len()
        )));
    }
    if row_flip_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m_total}",
            row_flip_full.len()
        )));
    }

    let mut eff_m = m_total;
    let mut maf_keep: Cow<[f32]> = Cow::Borrowed(maf_full.as_slice());
    let mut row_flip_keep: Cow<[bool]> = Cow::Borrowed(row_flip_full.as_slice());
    let mut packed_row_indices: Option<Vec<usize>> = None;
    if let Some(mask) = site_keep {
        let mask_vec: Vec<bool> = match mask.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mask.as_array().iter().copied().collect(),
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
    if eff_m == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_packed received zero active markers.",
        ));
    }

    let train_idx = parse_index_vec_i64(
        train_sample_indices.as_slice()?,
        n_samples,
        "train_sample_indices",
    )?;
    let n_train = train_idx.len();
    if n_train <= 1 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_packed requires at least two training samples.",
        ));
    }
    let y_vec_f64: Vec<f64> = match y_train.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y_train.as_array().iter().copied().collect(),
    };
    if y_vec_f64.len() != n_train {
        return Err(PyRuntimeError::new_err(format!(
            "y_train length mismatch: got {}, expected {n_train}",
            y_vec_f64.len()
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

    let std_eps32 = std_eps.max(1e-12) as f32;
    let mut row_mean = vec![0.0_f32; eff_m];
    let mut row_inv_sd = vec![0.0_f32; eff_m];
    let mut m_effective = 0usize;
    for j in 0..eff_m {
        let p = maf_keep[j].clamp(0.0, 0.5);
        let mean = 2.0_f32 * p;
        let var = (2.0_f32 * p * (1.0_f32 - p)).max(0.0);
        row_mean[j] = mean;
        if var > std_eps32 {
            row_inv_sd[j] = 1.0_f32 / var.sqrt();
            m_effective += 1;
        } else {
            row_inv_sd[j] = 0.0_f32;
        }
    }
    if m_effective == 0 {
        return Err(PyRuntimeError::new_err(
            "rrblup_exact_snp_packed found zero effective markers after standardization.",
        ));
    }

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let packed_flat_fit = packed_flat.clone();
    let sample_block_use = sample_block.max(1).min(n_train.max(1));
    let stream_row_block = eff_m.max(1);
    let cache_inner = py
        .detach(move || {
            build_rrblup_exact_snp_cache_from_packed(
                packed_flat.as_ref(),
                bytes_per_snp,
                n_samples,
                train_idx,
                row_flip_keep.into_owned(),
                row_mean,
                row_inv_sd,
                packed_row_indices,
                sample_block_use,
                stream_row_block,
                m_effective,
                threads,
                blas_threads,
            )
        })
        .map_err(map_err_string_to_py)?;
    let row_mean_ret = cache_inner.row_mean.clone();
    let row_inv_sd_ret = cache_inner.row_inv_sd.clone();
    let m_effective = cache_inner.m_effective;
    let y_mean = y_vec_f64.iter().sum::<f64>() / (n_train as f64);
    let (
        pred_train_ret,
        pred_test_ret,
        pve_trainvar,
        lambda_opt,
        reml_opt,
        var_g,
        sigma_e2,
        beta_ret,
        eig_backend,
    ) = py
        .detach(move || {
            fit_rrblup_exact_snp_from_cache_packed(
                &cache_inner,
                packed_flat_fit.as_ref(),
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
        y_mean,
        beta_arr,
        mean_arr,
        inv_arr,
        eig_backend,
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
        resident_packed_flat = Some(match external_packed_ro_ref.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(packed_view.iter().copied().collect()),
        });
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
        resident_packed_flat = Some(match loaded_packed_ro_ref.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(packed_view.iter().copied().collect()),
        });
    }

    let maf_full: Vec<f32> = if use_external_packed || use_external_stats {
        let maf_ro = external_maf_ro
            .as_ref()
            .expect("external maf must exist for external stats path");
        match maf_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => maf_ro.as_array().iter().copied().collect(),
        }
    } else {
        let maf_ro = loaded_maf_arr
            .as_ref()
            .expect("maf array must exist")
            .readonly();
        match maf_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => maf_ro.as_array().iter().copied().collect(),
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
        match row_flip_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_flip_ro.as_array().iter().copied().collect(),
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

    let mut eff_m = m_total;
    let mut maf_keep: Cow<[f32]> = Cow::Borrowed(maf_full.as_slice());
    let mut row_flip_keep: Cow<[bool]> = Cow::Borrowed(row_flip_full.as_slice());
    let mut packed_row_indices: Option<Vec<usize>> = None;
    if let Some(mask) = site_keep {
        let mask_vec: Vec<bool> = match mask.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mask.as_array().iter().copied().collect(),
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
    let y_vec_f64: Vec<f64> = match y_train.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y_train.as_array().iter().copied().collect(),
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

    let (row_mean, row_inv_sd, m_effective) =
        rrblup_parallel_row_standardization(maf_keep.as_ref(), std_eps32, pool_ref);

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
                        *v += y_mean as f32;
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
                                *v += y_mean as f32;
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
                        .map(|v| (*v as f64) + y_mean)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct XtMulBenchCase {
        name: &'static str,
        rows: usize,
        cols: usize,
        threads: usize,
        expect: XtMulKernelStrategy,
    }

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

    fn best_of_n_secs(mut f: impl FnMut()) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            f();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        best
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
        let ss_serial = row_major_block_prepare_rhs_diag_f32(
            &block,
            rows,
            cols,
            &dot_blk,
            lambda_use,
            &mut b_serial,
            &mut diag_serial,
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
        let ss_parallel = row_major_block_prepare_rhs_diag_f32(
            &block,
            rows,
            cols,
            &dot_blk,
            lambda_use,
            &mut b_parallel,
            &mut diag_parallel,
            Some(&pool),
        );

        assert!((ss_serial - ss_parallel).abs() <= 1e-6_f64 * ss_serial.abs().max(1.0));
        assert!(max_abs_diff(&b_serial, &b_parallel) <= 1e-6_f32);
        assert!(max_abs_diff(&diag_serial, &diag_parallel) <= 1e-6_f32);
    }

    #[test]
    #[ignore]
    fn compare_pcg_xtmul_strategies() {
        std::env::set_var("JX_ROWMAJOR_F32_KERNEL", "rayon");

        let cases = [
            XtMulBenchCase {
                name: "front_half_wide",
                rows: 768,
                cols: 262_144,
                threads: 8,
                expect: XtMulKernelStrategy::ColChunk,
            },
            XtMulBenchCase {
                name: "row_reduce_friendly",
                rows: 3_072,
                cols: 16_384,
                threads: 4,
                expect: XtMulKernelStrategy::RowReduce,
            },
            XtMulBenchCase {
                name: "row_reduce_fallback_window",
                rows: 1_024,
                cols: 32_768,
                threads: 16,
                expect: XtMulKernelStrategy::RowReduce,
            },
        ];

        for case in cases {
            let pool = Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(case.threads)
                    .build()
                    .expect("build rayon pool"),
            ));
            let pool_ref = pool.as_ref();
            let picked = choose_xtmul_kernel_strategy(case.rows, case.cols, pool_ref);
            let row_chunk = xtmul_row_chunk(case.rows);
            let row_tasks = case.rows.div_ceil(row_chunk);
            let col_chunk = xtmul_col_chunk(case.cols, case.threads);
            let col_tasks = case.cols.div_ceil(col_chunk);
            assert_eq!(picked, case.expect, "unexpected strategy for {}", case.name);

            let (block, weights) = make_xtmul_inputs(case.rows, case.cols);
            let _blas_guard = OpenBlasThreadGuard::enter(1);

            let mut out_legacy = vec![0.0_f32; case.cols];
            row_major_block_t_mul_vec_accum_f32_row_reduce(
                &block,
                case.rows,
                case.cols,
                &weights,
                &mut out_legacy,
                pool_ref,
            );

            let mut out_adaptive = vec![0.0_f32; case.cols];
            row_major_block_t_mul_vec_accum_f32(
                &block,
                case.rows,
                case.cols,
                &weights,
                &mut out_adaptive,
                pool_ref,
            );

            let diff = max_abs_diff(&out_legacy, &out_adaptive);
            assert!(
                diff <= 5e-3_f32,
                "strategy output drift too large for {}: diff={diff}",
                case.name
            );

            let legacy_secs = best_of_n_secs(|| {
                let mut out = vec![0.0_f32; case.cols];
                row_major_block_t_mul_vec_accum_f32_row_reduce(
                    &block, case.rows, case.cols, &weights, &mut out, pool_ref,
                );
            });
            let adaptive_secs = best_of_n_secs(|| {
                let mut out = vec![0.0_f32; case.cols];
                row_major_block_t_mul_vec_accum_f32(
                    &block, case.rows, case.cols, &weights, &mut out, pool_ref,
                );
            });
            let speedup = legacy_secs / adaptive_secs.max(1e-12_f64);

            eprintln!(
                "pcg_xtmul case={} rows={} cols={} threads={} strategy={:?} row_chunk={} row_tasks={} col_chunk={} col_tasks={} legacy={:.6}s adaptive={:.6}s speedup={:.2}x",
                case.name,
                case.rows,
                case.cols,
                case.threads,
                picked,
                row_chunk,
                row_tasks,
                col_chunk,
                col_tasks,
                legacy_secs,
                adaptive_secs,
                speedup,
            );
        }
    }
}
