//! rrBLUP marker-effect solver over additive PLINK BED blocks.
//!
//! Let `W` denote the standardized additive genotype matrix after site
//! filtering and sample subsetting, with markers in rows and samples in
//! columns, and let `y_c` be the centered phenotype. This module fits the
//! classical ridge-regression BLUP model
//!
//! `y_c = W' beta + e,  beta ~ N(0, sigma_beta^2 I),  e ~ N(0, sigma_e^2 I)`,
//!
//! which yields the normal equations
//!
//! `(W W' + lambda I_m) beta_hat = W y_c`,
//!
//! with `lambda = sigma_e^2 / sigma_beta^2`.
//!
//! The maintained solver is PCG in marker space with streamed BED decoding:
//!
//! 1. A pre-pass computes `W y_c` and the diagonal preconditioner.
//! 2. Each PCG iteration applies `v -> W (W' v) + lambda v` by decoding marker
//!    blocks on demand instead of materializing `W`.
//! 3. Prediction is obtained as `g_hat = W' beta_hat`, and the phenotype mean
//!    is added back only at output time.
//!
//! The implementation keeps BLAS single-threaded inside each kernel and uses
//! Rayon for the outer SNP/block parallelism so that the streamed operator stays
//! memory-bounded and scalable on large marker sets.

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
    cblas_sgemm_dispatch, rust_sgemm_prefers_rayon_rowmajor_f32_kernel, CblasInt,
    OpenBlasThreadGuard, CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
};
use crate::decode::decode_prepared_additive_block_packed_f32;
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
    let threads = pool.map(|tp| tp.current_num_threads()).unwrap_or(1).max(1);
    if cols >= 65_536 && rows <= 4_096 {
        return XtMulKernelStrategy::ColChunk;
    }
    let row_tasks = rows.div_ceil(xtmul_row_chunk(rows));
    if row_tasks >= threads.saturating_mul(2) {
        XtMulKernelStrategy::RowReduce
    } else {
        XtMulKernelStrategy::Blas
    }
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

#[inline]
fn rrblup_stream_window_mb(n_samples: usize, block_rows: usize) -> usize {
    let bytes_per_snp = n_samples.div_ceil(4).max(1);
    let target_bytes = block_rows
        .max(1)
        .saturating_mul(bytes_per_snp)
        .saturating_mul(2);
    target_bytes.div_ceil(1024 * 1024).max(1)
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
fn sample_var_f64(v: &[f64]) -> f64 {
    let n = v.len();
    if n <= 1 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / (n as f64);
    let mut ss = 0.0_f64;
    for &x in v {
        let d = x - mean;
        ss += d * d;
    }
    ss / ((n - 1) as f64)
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

    let mut row_mean = vec![0.0_f32; m];
    let mut row_inv_sd = vec![0.0_f32; m];
    let mut m_effective = 0usize;
    for j in 0..m {
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
                let mut b = vec![0.0_f32; m];
                let mut diag_inv = vec![0.0_f32; m];
                let mut sum_ss_global = 0.0_f64;
                let prep_t0 = Instant::now();
                let mut prep_decode_secs = 0.0_f64;
                let mut prep_rhs_secs = 0.0_f64;
                let mut prep_diag_secs = 0.0_f64;
                {
                    let resident_packed = resident_packed_flat.as_ref().map(|cow| cow.as_ref());
                    let mut prep_source = build_rrblup_source(
                        resident_packed,
                        bytes_per_snp,
                        prefix.as_str(),
                        n_samples,
                        row_step,
                    )?;
                    let mut block = vec![0.0_f32; row_step * n_train];
                    let mut dot_blk = vec![0.0_f32; row_step];
                    let mut tick = 0usize;
                    for st in (0..m).step_by(row_step) {
                        let ed = (st + row_step).min(m);
                        let cur_rows = ed - st;
                        let blk_slice = &mut block[..cur_rows * n_train];
                        let t_decode = Instant::now();
                        decode_standardized_block_f32(
                            &mut prep_source,
                            n_samples,
                            row_flip_keep.as_ref(),
                            &row_mean,
                            &row_inv_sd,
                            &train_idx,
                            full_train_fast,
                            packed_row_indices.as_deref(),
                            st,
                            blk_slice,
                            &code4_lut,
                            pool_ref,
                        )?;
                        prep_decode_secs += t_decode.elapsed().as_secs_f64();
                        let t_rhs = Instant::now();
                        row_major_block_mul_vec_f32(
                            blk_slice,
                            cur_rows,
                            n_train,
                            &y_center_f32,
                            &mut dot_blk[..cur_rows],
                            pool_ref,
                        );
                        prep_rhs_secs += t_rhs.elapsed().as_secs_f64();
                        let t_diag = Instant::now();
                        sum_ss_global += row_major_block_prepare_rhs_diag_f32(
                            blk_slice,
                            cur_rows,
                            n_train,
                            &dot_blk[..cur_rows],
                            lambda_use,
                            &mut b[st..ed],
                            &mut diag_inv[st..ed],
                            pool_ref,
                        );
                        prep_diag_secs += t_diag.elapsed().as_secs_f64();
                        tick += cur_rows;
                        if tick >= row_step.saturating_mul(64).max(1) {
                            check_ctrlc()?;
                            tick = 0;
                        }
                    }
                }
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
                        let mut g_hat = pred_train_f64.clone();
                        for g in &mut g_hat {
                            *g -= y_mean;
                        }
                        let resid: Vec<f64> = y_vec_f64
                            .iter()
                            .zip(pred_train_f64.iter())
                            .map(|(y, p)| *y - *p)
                            .collect();
                        let var_g = sample_var_f64(&g_hat);
                        let var_e = sample_var_f64(&resid);
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
                name: "blas_fallback_window",
                rows: 1_024,
                cols: 32_768,
                threads: 16,
                expect: XtMulKernelStrategy::Blas,
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
