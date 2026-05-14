use numpy::ndarray::{Array1, Array2};
use numpy::PyArray1;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::bedmath::{
    adaptive_grm_block_rows, decode_plink_bed_hardcall, decode_row_centered_full_lut_f64,
    decode_standardized_packed_block_rows_f32, is_identity_indices, packed_byte_lut,
};
use crate::blas::{
    cblas_sgemm_dispatch, rust_sgemm_prefers_rayon_rowmajor_f32_kernel, CblasInt,
    OpenBlasThreadGuard, CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
};
use crate::brent::brent_minimize;
use crate::eigh::symmetric_eigh_f64_row_major;
use crate::grm;
use crate::packed::{
    bed_packed_row_flip_mask, cross_grm_times_alpha_packed_f64, packed_malpha_f64,
};
use crate::pcg::pcg_solve_f32;
use crate::stats_common::{
    check_ctrlc, env_truthy, get_cached_pool, map_err_string_to_py, parse_index_vec_i64,
};

#[path = "../math/FaST.rs"]
mod fast_math;
use fast_math::gblup_marker_fast_packed;

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

// packed decode/hash/prediction kernels were migrated to `stats/packed.rs`
// to keep this file focused on association/model routines.

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

#[allow(clippy::too_many_arguments)]
fn decode_standardized_block_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
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
    decode_standardized_packed_block_rows_f32(
        packed_flat,
        bytes_per_snp,
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
    )
}

#[allow(clippy::too_many_arguments)]
fn pcg_x_mul_samples(
    packed_flat: &[u8],
    bytes_per_snp: usize,
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
    let mut block = vec![0.0_f32; row_step * n_out];
    let mut tick = 0usize;
    let mut decode_acc = 0.0_f64;
    let mut mul_acc = 0.0_f64;

    for st in (0..m).step_by(row_step) {
        let ed = (st + row_step).min(m);
        let cur_rows = ed - st;
        let blk_slice = &mut block[..cur_rows * n_out];
        let t_decode = Instant::now();
        decode_standardized_block_f32(
            packed_flat,
            bytes_per_snp,
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
    if let Some(t) = timing {
        t.decode_secs += decode_acc;
        t.mul_secs += mul_acc;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn pcg_xt_mul_rows(
    packed_flat: &[u8],
    bytes_per_snp: usize,
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
    let mut block = vec![0.0_f32; row_step * n_out];
    let mut tmp = vec![0.0_f32; row_step];
    let mut tick = 0usize;
    let mut decode_acc = 0.0_f64;
    let mut mul_acc = 0.0_f64;

    for st in (0..m).step_by(row_step) {
        let ed = (st + row_step).min(m);
        let cur_rows = ed - st;
        let blk_slice = &mut block[..cur_rows * n_out];
        let t_decode = Instant::now();
        decode_standardized_block_f32(
            packed_flat,
            bytes_per_snp,
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
    g_eps=1e-8_f64,
    low=-6.0_f64,
    high=6.0_f64,
    max_iter=50,
    tol=1e-4_f64,
    block_rows=4096,
    threads=0,
    return_variance_components=false,
    estimate_only=false,
    return_effect=false
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

    let (packed_arr, _miss_arr, maf_arr, _std_arr, n_samples) =
        crate::gfreader::load_bed_2bit_packed(py, prefix)?;
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("No samples found in BED input."));
    }

    let packed_ro = packed_arr.readonly();
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
    let bytes_per_snp = packed_view.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let packed_flat: Cow<[u8]> = match packed_ro.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_view.iter().copied().collect()),
    };

    let maf_ro = maf_arr.readonly();
    let maf_full: Vec<f32> = match maf_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => maf_ro.as_array().iter().copied().collect(),
    };
    if maf_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "maf length mismatch: got {}, expected {m_total}",
            maf_full.len()
        )));
    }

    let row_flip_full_arr = bed_packed_row_flip_mask(py, packed_arr.readonly(), n_samples)?;
    let row_flip_full: Vec<bool> = match row_flip_full_arr.readonly().as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip_full_arr
            .readonly()
            .as_array()
            .iter()
            .copied()
            .collect(),
    };
    if row_flip_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m_total}",
            row_flip_full.len()
        )));
    }

    let mut eff_m = m_total;
    let mut packed_keep: Cow<[u8]> = Cow::Borrowed(packed_flat.as_ref());
    let mut maf_keep: Cow<[f32]> = Cow::Borrowed(maf_full.as_slice());
    let mut row_flip_keep: Cow<[bool]> = Cow::Borrowed(row_flip_full.as_slice());
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
            let mut packed_subset = vec![0_u8; eff_m * bytes_per_snp];
            let mut maf_subset = vec![0.0_f32; eff_m];
            let mut row_flip_subset = vec![false; eff_m];
            for (dst_row, &src_row) in keep_idx.iter().enumerate() {
                let src_off = src_row * bytes_per_snp;
                let dst_off = dst_row * bytes_per_snp;
                packed_subset[dst_off..dst_off + bytes_per_snp]
                    .copy_from_slice(&packed_flat[src_off..src_off + bytes_per_snp]);
                maf_subset[dst_row] = maf_full[src_row].clamp(0.0, 0.5);
                row_flip_subset[dst_row] = row_flip_full[src_row];
            }
            packed_keep = Cow::Owned(packed_subset);
            maf_keep = Cow::Owned(maf_subset);
            row_flip_keep = Cow::Owned(row_flip_subset);
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
                        &packed_keep,
                        bytes_per_snp,
                        &row_flip_keep,
                        &maf_keep,
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

    let packed_keep_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((eff_m, bytes_per_snp), packed_keep.into_owned())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let row_flip_keep_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(row_flip_keep.into_owned())).into_bound();
    let maf_keep_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(maf_keep.into_owned())).into_bound();

    let train_idx_i64: Vec<i64> = train_idx.iter().map(|&v| v as i64).collect();
    let train_idx_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(train_idx_i64)).into_bound();

    let (grm_arr, row_sum_arr, var_sum_raw) = crate::grm::grm_packed_f64_with_stats(
        py,
        packed_keep_arr.readonly(),
        n_samples,
        row_flip_keep_arr.readonly(),
        maf_keep_arr.readonly(),
        Some(train_idx_arr.readonly()),
        1,
        block_rows.max(1),
        threads,
        None,
        0,
    )?;
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
        packed_keep_arr.readonly(),
        n_samples,
        row_flip_keep_arr.readonly(),
        maf_keep_arr.readonly(),
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
            packed_keep_arr.readonly(),
            n_samples,
            row_flip_keep_arr.readonly(),
            maf_keep_arr.readonly(),
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
                packed_keep_arr.readonly(),
                n_samples,
                row_flip_keep_arr.readonly(),
                maf_keep_arr.readonly(),
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

    let use_external_packed =
        packed.is_some() || maf.is_some() || row_flip.is_some() || packed_n_samples > 0;
    let mut loaded_packed_arr: Option<Bound<'py, PyArray2<u8>>> = None;
    let mut loaded_maf_arr: Option<Bound<'py, PyArray1<f32>>> = None;
    let mut external_maf_ro: Option<PyReadonlyArray1<'py, f32>> = None;
    let mut external_row_flip_ro: Option<PyReadonlyArray1<'py, bool>> = None;

    let n_samples: usize;
    let packed_ro: PyReadonlyArray2<'py, u8>;

    if use_external_packed {
        packed_ro = packed.ok_or_else(|| {
            PyRuntimeError::new_err(
                "rrblup_pcg_bed: packed payload path requires `packed` argument.",
            )
        })?;
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
    } else {
        let (packed_arr, _miss_arr, maf_arr, _std_arr, n_samples_loaded) =
            crate::gfreader::load_bed_2bit_packed(py, prefix)?;
        if n_samples_loaded == 0 {
            return Err(PyRuntimeError::new_err("No samples found in BED input."));
        }
        n_samples = n_samples_loaded;
        loaded_packed_arr = Some(packed_arr);
        loaded_maf_arr = Some(maf_arr);
        packed_ro = loaded_packed_arr
            .as_ref()
            .expect("packed array must exist")
            .readonly();
    }

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
    let bytes_per_snp = packed_view.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let packed_flat: Cow<[u8]> = match packed_ro.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_view.iter().copied().collect()),
    };

    let maf_full: Vec<f32> = if use_external_packed {
        let maf_ro = external_maf_ro
            .as_ref()
            .expect("external maf must exist for packed payload path");
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

    let row_flip_full: Vec<bool> = if use_external_packed {
        let row_flip_ro = external_row_flip_ro
            .as_ref()
            .expect("external row_flip must exist for packed payload path");
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
    let packed_keep: Cow<[u8]> = Cow::Borrowed(packed_flat.as_ref());
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
                // Default remains BLAS=1 to preserve the current behavior, but
                // benchmarks can override this to compare split thread plans.
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
                {
                    // This decode workspace can reach multiple GiB for large n_train.
                    // Keep it scoped to the diagonal/preconditioner pass so it is
                    // released before the PCG matvec loop allocates its own block.
                    let mut block = vec![0.0_f32; row_step * n_train];
                    let mut dot_blk = vec![0.0_f32; row_step];
                    let mut tick = 0usize;
                    for st in (0..m).step_by(row_step) {
                        let ed = (st + row_step).min(m);
                        let cur_rows = ed - st;
                        let blk_slice = &mut block[..cur_rows * n_train];
                        decode_standardized_block_f32(
                            packed_keep.as_ref(),
                            bytes_per_snp,
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
                        row_major_block_mul_vec_f32(
                            blk_slice,
                            cur_rows,
                            n_train,
                            &y_center_f32,
                            &mut dot_blk[..cur_rows],
                            pool_ref,
                        );
                        for r in 0..cur_rows {
                            let row = &blk_slice[r * n_train..(r + 1) * n_train];
                            let ss = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
                            sum_ss_global += ss;
                            b[st + r] = dot_blk[r];
                            let d = ((ss as f32) + lambda_use).max(1e-12_f32);
                            diag_inv[st + r] = 1.0_f32 / d;
                        }
                        tick += cur_rows;
                        if tick >= row_step.saturating_mul(64).max(1) {
                            check_ctrlc()?;
                            tick = 0;
                        }
                    }
                }

                let mut last_notified = 0usize;
                let pcg_t0 = Instant::now();
                let pcg_res = pcg_solve_f32(
                    &b,
                    max_iter,
                    tol_use,
                    1e-20_f64,
                    |p| {
                        let mut xmul_t = PcgMatvecTiming::default();
                        let xp = pcg_x_mul_samples(
                            packed_keep.as_ref(),
                            bytes_per_snp,
                            n_samples,
                            row_flip_keep.as_ref(),
                            &row_mean,
                            &row_inv_sd,
                            &train_idx,
                            full_train_fast,
                            packed_row_indices.as_deref(),
                            row_step,
                            p,
                            &code4_lut,
                            pool_ref,
                            if stage_timing { Some(&mut xmul_t) } else { None },
                        )?;
                        let mut xtmul_t = PcgMatvecTiming::default();
                        let mut ap = pcg_xt_mul_rows(
                            packed_keep.as_ref(),
                            bytes_per_snp,
                            n_samples,
                            row_flip_keep.as_ref(),
                            &row_mean,
                            &row_inv_sd,
                            &train_idx,
                            full_train_fast,
                            packed_row_indices.as_deref(),
                            row_step,
                            &xp,
                            &code4_lut,
                            pool_ref,
                            if stage_timing { Some(&mut xtmul_t) } else { None },
                        )?;
                        if let Some(tp) = pool_ref {
                            tp.install(|| {
                                ap.par_iter_mut()
                                    .zip(p.par_iter())
                                    .for_each(|(apj, pj)| *apj += lambda_use * *pj);
                            });
                        } else {
                            for j in 0..m {
                                ap[j] += lambda_use * p[j];
                            }
                        }
                        if stage_timing {
                            if let Ok(mut acc) = stage_accum.lock() {
                                acc.decode_secs += xmul_t.decode_secs + xtmul_t.decode_secs;
                                acc.xmul_secs += xmul_t.mul_secs;
                                acc.xtmul_secs += xtmul_t.mul_secs;
                            }
                        }
                        Ok(ap)
                    },
                    |r, z| {
                        if let Some(tp) = pool_ref {
                            tp.install(|| {
                                z.par_iter_mut()
                                    .zip(r.par_iter())
                                    .zip(diag_inv.par_iter())
                                    .for_each(|((zj, rj), dj)| *zj = *dj * *rj);
                            });
                        } else {
                            for j in 0..m {
                                z[j] = diag_inv[j] * r[j];
                            }
                        }
                    },
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
                    let mut pred_train_full = pcg_x_mul_samples(
                        packed_keep.as_ref(),
                        bytes_per_snp,
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
                    let pred_train_f64: Vec<f64> = pred_train_full.iter().map(|v| *v as f64).collect();
                    pred_train_ret = if let Some(local_idx) = &train_pred_pick {
                        local_idx.iter().map(|&i| pred_train_f64[i]).collect()
                    } else {
                        pred_train_f64.clone()
                    };

                    if compute_trainvar {
                        let mut g_hat = pred_train_f64.clone();
                        for g in g_hat.iter_mut() {
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
                            let mut pred_train_sub = pcg_x_mul_samples(
                                packed_keep.as_ref(),
                                bytes_per_snp,
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
                            for v in pred_train_sub.iter_mut() {
                                *v += y_mean as f32;
                            }
                            pred_train_ret = pred_train_sub.iter().map(|v| *v as f64).collect();
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
                    let test_pred_f32 = pcg_x_mul_samples(
                        packed_keep.as_ref(),
                        bytes_per_snp,
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
                    pred_test_ret = test_pred_f32.iter().map(|v| (*v as f64) + y_mean).collect();
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
