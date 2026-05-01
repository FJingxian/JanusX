use numpy::ndarray::{Array1, Array2};
use numpy::PyArray1;
use numpy::PyUntypedArrayMethods;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use crate::bedmath::{
    adaptive_grm_block_rows, decode_row_centered_full_lut, decode_row_centered_full_lut_f64,
    decode_subset_row_from_full_scratch, decode_subset_row_from_full_scratch_f64,
    is_identity_indices, packed_byte_lut,
};
use crate::blas::{
    cblas_dgemm_dispatch, cblas_sgemm_dispatch, CblasInt, OpenBlasThreadGuard, CBLAS_COL_MAJOR,
    CBLAS_NO_TRANS, CBLAS_TRANS,
};
use crate::brent::brent_minimize;
use crate::eigh::symmetric_eigh_f64_row_major;
use crate::packed::{
    bed_packed_row_flip_mask, cross_grm_times_alpha_packed_f64, packed_malpha_f64,
};
use crate::stats_common::{
    check_ctrlc, env_truthy, get_cached_pool, map_err_string_to_py, parse_index_vec_i64,
};

#[path = "../math/FaST.rs"]
mod fast_math;
use fast_math::gblup_marker_fast_packed;
#[path = "../math/grm.rs"]
mod grm_math;
use grm_math::{grm_packed_f32_with_stats_impl, grm_packed_f64_with_stats_impl};

#[allow(clippy::too_many_arguments)]
fn decode_grm_block(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip_vec: &[bool],
    row_maf_vec: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    method: usize,
    eps: f32,
    code4_lut: &[[u8; 4]; 256],
    row_start: usize,
    row_end: usize,
    n: usize,
    out_block: &mut [f32],
    out_varsum: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    if cur_rows == 0 {
        return Ok(());
    }
    if out_block.len() < cur_rows.saturating_mul(n) {
        return Err("decode_grm_block: out_block too small".to_string());
    }
    if out_varsum.len() < cur_rows {
        return Err("decode_grm_block: out_varsum too small".to_string());
    }
    let block = &mut out_block[..cur_rows * n];
    let varsum = &mut out_varsum[..cur_rows];

    let mut decode_run = || {
        if full_sample_fast {
            block
                .par_chunks_mut(n)
                .zip(varsum.par_iter_mut())
                .enumerate()
                .for_each(|(off, (out_row, row_varsum_dst))| {
                    let idx = row_start + off;
                    let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                    let flip = row_flip_vec[idx];
                    let p = row_maf_vec[idx].clamp(0.0, 0.5);
                    let mean_g = 2.0_f32 * p;
                    let var = 2.0_f32 * p * (1.0_f32 - p);
                    let std_scale = if method == 2 {
                        if var > eps {
                            1.0_f32 / var.sqrt()
                        } else {
                            0.0_f32
                        }
                    } else {
                        1.0_f32
                    };
                    let value_lut: [f32; 4] = if flip {
                        [
                            (2.0_f32 - mean_g) * std_scale,
                            0.0_f32, // missing -> mean imputation => centered 0
                            (1.0_f32 - mean_g) * std_scale,
                            (0.0_f32 - mean_g) * std_scale,
                        ]
                    } else {
                        [
                            (0.0_f32 - mean_g) * std_scale,
                            0.0_f32, // missing -> mean imputation => centered 0
                            (1.0_f32 - mean_g) * std_scale,
                            (2.0_f32 - mean_g) * std_scale,
                        ]
                    };
                    decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, out_row);
                    if method == 1 {
                        *row_varsum_dst = var as f64;
                    }
                });
        } else {
            block
                .par_chunks_mut(n)
                .zip(varsum.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || vec![0.0_f32; n_samples],
                    |full_row, (off, (out_row, row_varsum_dst))| {
                        let idx = row_start + off;
                        let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                        let flip = row_flip_vec[idx];
                        let p = row_maf_vec[idx].clamp(0.0, 0.5);
                        let default_mean_g = 2.0_f32 * p;
                        let (var_centered, _row_sum) = decode_subset_row_from_full_scratch(
                            row,
                            n_samples,
                            sample_idx,
                            flip,
                            default_mean_g,
                            method,
                            eps,
                            code4_lut,
                            full_row.as_mut_slice(),
                            out_row,
                        );
                        if method == 1 {
                            *row_varsum_dst = var_centered;
                        }
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

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn grm_rankk_update_cblas(
    grm: &mut [f32],
    block: &[f32],
    cur_rows: usize,
    n: usize,
    copy_rhs: bool,
    beta_zero_accum: bool,
) {
    let rhs_owned = if copy_rhs { Some(block.to_vec()) } else { None };
    let rhs_ptr = match rhs_owned.as_ref() {
        Some(v) => v.as_ptr(),
        None => block.as_ptr(),
    };
    if beta_zero_accum {
        let mut tmp = vec![0.0_f32; n * n];
        unsafe {
            // `block` is row-major (cur_rows, n), which is layout-equivalent to
            // a column-major matrix with shape (n, cur_rows). Use col-major GEMM
            // to compute: tmp = block^T @ block == A @ A^T.
            cblas_sgemm_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n as CblasInt,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                rhs_ptr,
                n as CblasInt,
                0.0,
                tmp.as_mut_ptr(),
                n as CblasInt,
            );
        }
        for (g, t) in grm.iter_mut().zip(tmp.iter()) {
            *g += *t;
        }
    } else {
        unsafe {
            cblas_sgemm_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n as CblasInt,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                rhs_ptr,
                n as CblasInt,
                1.0,
                grm.as_mut_ptr(),
                n as CblasInt,
            );
        }
    }
}

#[inline]
pub(crate) fn grm_rankk_update(
    grm: &mut [f32],
    block: &[f32],
    cur_rows: usize,
    n: usize,
    cblas_copy_rhs: bool,
    cblas_beta_zero_accum: bool,
) -> Result<(), String> {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        grm_rankk_update_cblas(
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            cblas_beta_zero_accum,
        );
        return Ok(());
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = (
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            cblas_beta_zero_accum,
        );
        Err(
            "Packed GRM requires CBLAS backend on this platform; fallback to streaming GRM."
                .to_string(),
        )
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn grm_rankk_update_cblas_f64(
    grm: &mut [f64],
    block: &[f64],
    cur_rows: usize,
    n: usize,
    copy_rhs: bool,
    beta_zero_accum: bool,
) {
    let rhs_owned = if copy_rhs { Some(block.to_vec()) } else { None };
    let rhs_ptr = match rhs_owned.as_ref() {
        Some(v) => v.as_ptr(),
        None => block.as_ptr(),
    };
    if beta_zero_accum {
        let mut tmp = vec![0.0_f64; n * n];
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n as CblasInt,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                rhs_ptr,
                n as CblasInt,
                0.0,
                tmp.as_mut_ptr(),
                n as CblasInt,
            );
        }
        for (g, t) in grm.iter_mut().zip(tmp.iter()) {
            *g += *t;
        }
    } else {
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n as CblasInt,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                rhs_ptr,
                n as CblasInt,
                1.0,
                grm.as_mut_ptr(),
                n as CblasInt,
            );
        }
    }
}

#[inline]
pub(crate) fn grm_rankk_update_f64(
    grm: &mut [f64],
    block: &[f64],
    cur_rows: usize,
    n: usize,
    cblas_copy_rhs: bool,
    cblas_beta_zero_accum: bool,
) -> Result<(), String> {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        grm_rankk_update_cblas_f64(
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            cblas_beta_zero_accum,
        );
        return Ok(());
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = (
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            cblas_beta_zero_accum,
        );
        Err(
            "Packed GRM requires CBLAS backend on this platform; fallback to streaming GRM."
                .to_string(),
        )
    }
}

// packed decode/hash/prediction kernels were migrated to `stats/packed.rs`
// to keep this file focused on association/model routines.

#[inline]
fn row_major_block_mul_vec_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);
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
fn row_major_block_t_mul_vec_accum_f32(
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
    row_start: usize,
    out: &mut [f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Ok(());
    }
    if out.len() % n_out != 0 {
        return Err("decode_standardized_block_f32: output length mismatch".to_string());
    }
    let cur_rows = out.len() / n_out;
    if cur_rows == 0 {
        return Ok(());
    }

    let decode_one = |local_row: usize, out_row: &mut [f32]| {
        let row_idx = row_start + local_row;
        let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
        let mean_g = row_mean[row_idx];
        let inv_sd = row_inv_sd[row_idx];
        if full_sample_fast {
            let value_lut: [f32; 4] = if row_flip[row_idx] {
                [
                    (2.0_f32 - mean_g) * inv_sd,
                    0.0_f32,
                    (1.0_f32 - mean_g) * inv_sd,
                    (0.0_f32 - mean_g) * inv_sd,
                ]
            } else {
                [
                    (0.0_f32 - mean_g) * inv_sd,
                    0.0_f32,
                    (1.0_f32 - mean_g) * inv_sd,
                    (2.0_f32 - mean_g) * inv_sd,
                ]
            };
            decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, out_row);
            return;
        }

        let flip = row_flip[row_idx];
        for (j, &sid) in sample_idx.iter().enumerate() {
            let b = row[sid >> 2];
            let code = (b >> ((sid & 3) * 2)) & 0b11;
            let mut gv = match code {
                0b00 => 0.0_f32,
                0b10 => 1.0_f32,
                0b11 => 2.0_f32,
                _ => mean_g,
            };
            if flip && code != 0b01 {
                gv = 2.0_f32 - gv;
            }
            out_row[j] = (gv - mean_g) * inv_sd;
        }
    };

    if let Some(tp) = pool {
        tp.install(|| {
            out.par_chunks_mut(n_out)
                .enumerate()
                .for_each(|(local_row, out_row)| decode_one(local_row, out_row));
        });
    } else {
        for (local_row, out_row) in out.chunks_mut(n_out).enumerate() {
            decode_one(local_row, out_row);
        }
    }
    Ok(())
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
    block_rows: usize,
    weights: &[f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
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

    for st in (0..m).step_by(row_step) {
        let ed = (st + row_step).min(m);
        let cur_rows = ed - st;
        let blk_slice = &mut block[..cur_rows * n_out];
        decode_standardized_block_f32(
            packed_flat,
            bytes_per_snp,
            n_samples,
            row_flip,
            row_mean,
            row_inv_sd,
            sample_idx,
            full_sample_fast,
            st,
            blk_slice,
            code4_lut,
            pool,
        )?;
        row_major_block_t_mul_vec_accum_f32(blk_slice, cur_rows, n_out, &weights[st..ed], &mut out);
        tick += cur_rows;
        if tick >= row_step.saturating_mul(64).max(1) {
            check_ctrlc()?;
            tick = 0;
        }
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
    block_rows: usize,
    u: &[f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
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

    for st in (0..m).step_by(row_step) {
        let ed = (st + row_step).min(m);
        let cur_rows = ed - st;
        let blk_slice = &mut block[..cur_rows * n_out];
        decode_standardized_block_f32(
            packed_flat,
            bytes_per_snp,
            n_samples,
            row_flip,
            row_mean,
            row_inv_sd,
            sample_idx,
            full_sample_fast,
            st,
            blk_slice,
            code4_lut,
            pool,
        )?;
        let out_blk = &mut tmp[..cur_rows];
        row_major_block_mul_vec_f32(blk_slice, cur_rows, n_out, u, out_blk);
        out[st..ed].copy_from_slice(out_blk);
        tick += cur_rows;
        if tick >= row_step.saturating_mul(64).max(1) {
            check_ctrlc()?;
            tick = 0;
        }
    }
    Ok(out)
}

#[inline]
fn dot_f32_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
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
    threads=0
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

    let keep_idx: Vec<usize> = if let Some(mask) = site_keep {
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
        mask_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &k)| if k { Some(i) } else { None })
            .collect()
    } else {
        (0..m_total).collect()
    };
    if keep_idx.is_empty() {
        return Err(PyRuntimeError::new_err(
            "No SNPs remained after applying site_keep mask.",
        ));
    }

    let eff_m = keep_idx.len();
    let mut packed_keep = vec![0_u8; eff_m * bytes_per_snp];
    let mut maf_keep = vec![0.0_f32; eff_m];
    let mut row_flip_keep = vec![false; eff_m];
    for (dst_row, &src_row) in keep_idx.iter().enumerate() {
        let src_off = src_row * bytes_per_snp;
        let dst_off = dst_row * bytes_per_snp;
        packed_keep[dst_off..dst_off + bytes_per_snp]
            .copy_from_slice(&packed_flat[src_off..src_off + bytes_per_snp]);
        maf_keep[dst_row] = maf_full[src_row].clamp(0.0, 0.5);
        row_flip_keep[dst_row] = row_flip_full[src_row];
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

        let (pred_train, pred_test, pve, lambda_opt, ml, reml, evd_backend, evd_elapsed) = py
            .detach(
                move || -> Result<(Vec<f64>, Vec<f64>, f64, f64, f64, f64, String, f64), String> {
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
                        pool_ref,
                    )
                },
            )
            .map_err(map_err_string_to_py)?;

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
        ));
    }

    let packed_keep_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((eff_m, bytes_per_snp), packed_keep)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let row_flip_keep_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(row_flip_keep)).into_bound();
    let maf_keep_arr = PyArray1::from_owned_array(py, Array1::from_vec(maf_keep)).into_bound();

    let train_idx_i64: Vec<i64> = train_idx.iter().map(|&v| v as i64).collect();
    let train_idx_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(train_idx_i64)).into_bound();

    let (grm_arr, row_sum_arr, var_sum_raw) = grm_packed_f64_with_stats(
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

    let (alpha_vec, beta0, lambda_opt, pve, ml, reml, evd_backend, evd_elapsed) = py
        .detach(
            move || -> Result<(Vec<f64>, f64, f64, f64, f64, f64, String, f64), String> {
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
                ))
            },
        )
        .map_err(map_err_string_to_py)?;

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
    let mut pred_train_all: Vec<f64> = train_pred_all_slice.iter().map(|v| *v + beta0).collect();
    let pred_train: Vec<f64> = if let Some(local_idx) = train_pred_pick {
        local_idx.iter().map(|&i| pred_train_all[i]).collect()
    } else {
        std::mem::take(&mut pred_train_all)
    };

    let pred_test: Vec<f64> = if test_idx.is_empty() {
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
    progress_every=0
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
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    bool,
    usize,
    f64,
    usize,
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

    let keep_idx: Vec<usize> = if let Some(mask) = site_keep {
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
        mask_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &k)| if k { Some(i) } else { None })
            .collect()
    } else {
        (0..m_total).collect()
    };
    if keep_idx.is_empty() {
        return Err(PyRuntimeError::new_err(
            "No SNPs remained after applying site_keep mask.",
        ));
    }

    let eff_m = keep_idx.len();
    let mut packed_keep = vec![0_u8; eff_m * bytes_per_snp];
    let mut maf_keep = vec![0.0_f32; eff_m];
    let mut row_flip_keep = vec![false; eff_m];
    for (dst_row, &src_row) in keep_idx.iter().enumerate() {
        let src_off = src_row * bytes_per_snp;
        let dst_off = dst_row * bytes_per_snp;
        packed_keep[dst_off..dst_off + bytes_per_snp]
            .copy_from_slice(&packed_flat[src_off..src_off + bytes_per_snp]);
        maf_keep[dst_row] = maf_full[src_row].clamp(0.0, 0.5);
        row_flip_keep[dst_row] = row_flip_full[src_row];
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

    let (pred_train_ret, pred_test_ret, pve_trainvar, converged, iters, rel_res) = py
        .detach(
            move || -> Result<(Vec<f64>, Vec<f64>, f64, bool, usize, f64), String> {
                let mut b = vec![0.0_f32; m];
                let mut diag_inv = vec![0.0_f32; m];
                let mut block = vec![0.0_f32; row_step * n_train];
                let mut dot_blk = vec![0.0_f32; row_step];
                let mut tick = 0usize;
                for st in (0..m).step_by(row_step) {
                    let ed = (st + row_step).min(m);
                    let cur_rows = ed - st;
                    let blk_slice = &mut block[..cur_rows * n_train];
                    decode_standardized_block_f32(
                        &packed_keep,
                        bytes_per_snp,
                        n_samples,
                        &row_flip_keep,
                        &row_mean,
                        &row_inv_sd,
                        &train_idx,
                        full_train_fast,
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
                    );
                    for r in 0..cur_rows {
                        let row = &blk_slice[r * n_train..(r + 1) * n_train];
                        let ss = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() as f32;
                        b[st + r] = dot_blk[r];
                        let d = (ss + lambda_use).max(1e-12_f32);
                        diag_inv[st + r] = 1.0_f32 / d;
                    }
                    tick += cur_rows;
                    if tick >= row_step.saturating_mul(64).max(1) {
                        check_ctrlc()?;
                        tick = 0;
                    }
                }

                let bnorm = dot_f32_f64(&b, &b).sqrt();
                if !(bnorm.is_finite()) {
                    return Err("rrBLUP-PCG invalid RHS norm.".to_string());
                }
                let denom_b = bnorm.max(1e-12);

                let mut beta = vec![0.0_f32; m];
                let mut r = b.clone();
                let mut z = vec![0.0_f32; m];
                for i in 0..m {
                    z[i] = diag_inv[i] * r[i];
                }
                let mut p = z.clone();
                let mut rz_old = dot_f32_f64(&r, &z);
                let mut converged = false;
                let mut rel_res = (dot_f32_f64(&r, &r).sqrt() / denom_b).max(0.0);
                let mut iters_done = 0usize;
                let mut last_notified = 0usize;
                let tiny = 1e-20_f64;

                for it in 0..max_iter {
                    let xp = pcg_x_mul_samples(
                        &packed_keep,
                        bytes_per_snp,
                        n_samples,
                        &row_flip_keep,
                        &row_mean,
                        &row_inv_sd,
                        &train_idx,
                        full_train_fast,
                        row_step,
                        &p,
                        &code4_lut,
                        pool_ref,
                    )?;
                    let mut ap = pcg_xt_mul_rows(
                        &packed_keep,
                        bytes_per_snp,
                        n_samples,
                        &row_flip_keep,
                        &row_mean,
                        &row_inv_sd,
                        &train_idx,
                        full_train_fast,
                        row_step,
                        &xp,
                        &code4_lut,
                        pool_ref,
                    )?;
                    for j in 0..m {
                        ap[j] += lambda_use * p[j];
                    }
                    let denom = dot_f32_f64(&p, &ap);
                    if !(denom.is_finite()) || denom <= tiny {
                        break;
                    }
                    let alpha = rz_old / denom;
                    for j in 0..m {
                        beta[j] += (alpha as f32) * p[j];
                        r[j] -= (alpha as f32) * ap[j];
                    }
                    rel_res = (dot_f32_f64(&r, &r).sqrt() / denom_b).max(0.0);
                    iters_done = it + 1;
                    if (iters_done >= last_notified.saturating_add(notify_step))
                        || (iters_done == max_iter)
                    {
                        last_notified = iters_done;
                        if let Some(cb) = progress_callback.as_ref() {
                            Python::attach(|py2| -> PyResult<()> {
                                py2.check_signals()?;
                                cb.call1(py2, (iters_done, max_iter))?;
                                Ok(())
                            })
                            .map_err(|e| e.to_string())?;
                        }
                    }
                    if rel_res.is_finite() && rel_res <= tol_use {
                        converged = true;
                        break;
                    }

                    for j in 0..m {
                        z[j] = diag_inv[j] * r[j];
                    }
                    let rz_new = dot_f32_f64(&r, &z);
                    if !(rz_new.is_finite()) || rz_new <= tiny {
                        break;
                    }
                    let beta_cg = rz_new / rz_old.max(tiny);
                    for j in 0..m {
                        p[j] = z[j] + (beta_cg as f32) * p[j];
                    }
                    rz_old = rz_new;
                    if (it & 7) == 7 {
                        check_ctrlc()?;
                    }
                }

                let mut pred_train_full = pcg_x_mul_samples(
                    &packed_keep,
                    bytes_per_snp,
                    n_samples,
                    &row_flip_keep,
                    &row_mean,
                    &row_inv_sd,
                    &train_idx,
                    full_train_fast,
                    row_step,
                    &beta,
                    &code4_lut,
                    pool_ref,
                )?;
                for v in pred_train_full.iter_mut() {
                    *v += y_mean as f32;
                }

                let pred_train_f64: Vec<f64> = pred_train_full.iter().map(|v| *v as f64).collect();
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
                let pve_trainvar = if denom.is_finite() && denom > 0.0 {
                    var_g / denom
                } else {
                    f64::NAN
                };

                let pred_train_ret = if let Some(local_idx) = &train_pred_pick {
                    local_idx.iter().map(|&i| pred_train_f64[i]).collect()
                } else {
                    pred_train_f64
                };

                let mut pred_test_ret: Vec<f64> = Vec::new();
                if !test_idx.is_empty() {
                    let test_pred_f32 = pcg_x_mul_samples(
                        &packed_keep,
                        bytes_per_snp,
                        n_samples,
                        &row_flip_keep,
                        &row_mean,
                        &row_inv_sd,
                        &test_idx,
                        full_test_fast,
                        row_step,
                        &beta,
                        &code4_lut,
                        pool_ref,
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

    Ok((
        train_arr,
        test_arr,
        pve_trainvar,
        converged,
        iters,
        rel_res,
        m_effective,
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
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if method != 1 && method != 2 {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method={method}; expected 1 (centered) or 2 (standardized)"
        )));
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

    let varsum_full = if method == 1 && full_sample_fast {
        let mut acc = 0.0_f64;
        for &maf in row_maf_vec.iter() {
            let p = maf as f64;
            let v = 2.0_f64 * p * (1.0_f64 - p);
            if v.is_finite() && v > 0.0 {
                acc += v;
            }
        }
        if !(acc.is_finite() && acc > 0.0) {
            return Err(PyRuntimeError::new_err(
                "invalid centered GRM denominator: sum(2p(1-p)) <= 0",
            ));
        }
        acc
    } else {
        0.0_f64
    };

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;
    let row_step = adaptive_grm_block_rows(
        block_cols.max(1),
        m.max(1),
        n,
        if full_sample_fast { 0 } else { n_samples },
        threads,
    );
    let block_target_mb = std::env::var("JX_GRM_PACKED_BLOCK_TARGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(64.0_f64);
    let stage_timing = env_truthy("JX_GRM_PACKED_STAGE_TIMING");
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
        // Default ON: avoids observed diagonal drift for very large SNP blocks.
        .unwrap_or(true);

    let grm_vec = py
        .detach(move || -> Result<Vec<f32>, String> {
            let total_t0 = Instant::now();
            let mut decode_secs = 0.0_f64;
            let mut gemm_secs = 0.0_f64;
            let mut grm = vec![0.0_f32; n * n];
            let mut varsum_acc = varsum_full;
            let eps = 1e-12_f32;
            let byte_lut = packed_byte_lut();

            let mut block_a = vec![0.0_f32; row_step * n];
            let mut block_b = vec![0.0_f32; row_step * n];
            let mut varsum_a = vec![0.0_f64; row_step];
            let mut varsum_b = vec![0.0_f64; row_step];
            // Default strategy: always use dual-buffer overlap when there is
            // enough work and more than one thread.
            let pipeline_overlap = (threads != 1) && (m > row_step);

            let mut cur_start = 0usize;
            let mut cur_end = row_step.min(m);
            let mut use_a = true;
            let mut last_notified = 0usize;

            let decode_t0 = Instant::now();
            decode_grm_block(
                packed_flat.as_ref(),
                bytes_per_snp,
                n_samples,
                row_flip_vec.as_ref(),
                row_maf_vec.as_ref(),
                &sample_idx,
                full_sample_fast,
                method,
                eps,
                &byte_lut.code4,
                cur_start,
                cur_end,
                n,
                &mut block_a,
                &mut varsum_a,
                pool.as_ref(),
            )?;
            decode_secs += decode_t0.elapsed().as_secs_f64();

            loop {
                let cur_rows = cur_end.saturating_sub(cur_start);
                if cur_rows == 0 {
                    break;
                }
                let next_start = cur_end;
                let next_end = (next_start + row_step).min(m);
                let has_next = next_start < m;

                if pipeline_overlap && has_next {
                    if use_a {
                        let (gemm_dt, decode_dt) =
                            std::thread::scope(|scope| -> Result<(f64, f64), String> {
                                let decode_handle = scope.spawn(|| {
                                    let dt0 = Instant::now();
                                    let r = decode_grm_block(
                                        packed_flat.as_ref(),
                                        bytes_per_snp,
                                        n_samples,
                                        row_flip_vec.as_ref(),
                                        row_maf_vec.as_ref(),
                                        &sample_idx,
                                        full_sample_fast,
                                        method,
                                        eps,
                                        &byte_lut.code4,
                                        next_start,
                                        next_end,
                                        n,
                                        &mut block_b,
                                        &mut varsum_b,
                                        pool.as_ref(),
                                    );
                                    (r, dt0.elapsed().as_secs_f64())
                                });

                                let gt0 = Instant::now();
                                let gemm_res = grm_rankk_update(
                                    &mut grm,
                                    &block_a[..cur_rows * n],
                                    cur_rows,
                                    n,
                                    cblas_copy_rhs,
                                    cblas_beta_zero_accum,
                                );
                                let gemm_dt = gt0.elapsed().as_secs_f64();

                                let (decode_res, decode_dt) = decode_handle
                                    .join()
                                    .map_err(|_| "decode worker thread panicked".to_string())?;
                                gemm_res?;
                                decode_res?;
                                Ok((gemm_dt, decode_dt))
                            })?;
                        gemm_secs += gemm_dt;
                        decode_secs += decode_dt;
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_a[..cur_rows].iter().sum::<f64>();
                        }
                    } else {
                        let (gemm_dt, decode_dt) =
                            std::thread::scope(|scope| -> Result<(f64, f64), String> {
                                let decode_handle = scope.spawn(|| {
                                    let dt0 = Instant::now();
                                    let r = decode_grm_block(
                                        packed_flat.as_ref(),
                                        bytes_per_snp,
                                        n_samples,
                                        row_flip_vec.as_ref(),
                                        row_maf_vec.as_ref(),
                                        &sample_idx,
                                        full_sample_fast,
                                        method,
                                        eps,
                                        &byte_lut.code4,
                                        next_start,
                                        next_end,
                                        n,
                                        &mut block_a,
                                        &mut varsum_a,
                                        pool.as_ref(),
                                    );
                                    (r, dt0.elapsed().as_secs_f64())
                                });

                                let gt0 = Instant::now();
                                let gemm_res = grm_rankk_update(
                                    &mut grm,
                                    &block_b[..cur_rows * n],
                                    cur_rows,
                                    n,
                                    cblas_copy_rhs,
                                    cblas_beta_zero_accum,
                                );
                                let gemm_dt = gt0.elapsed().as_secs_f64();

                                let (decode_res, decode_dt) = decode_handle
                                    .join()
                                    .map_err(|_| "decode worker thread panicked".to_string())?;
                                gemm_res?;
                                decode_res?;
                                Ok((gemm_dt, decode_dt))
                            })?;
                        gemm_secs += gemm_dt;
                        decode_secs += decode_dt;
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_b[..cur_rows].iter().sum::<f64>();
                        }
                    }
                } else {
                    if use_a {
                        let gemm_t0 = Instant::now();
                        grm_rankk_update(
                            &mut grm,
                            &block_a[..cur_rows * n],
                            cur_rows,
                            n,
                            cblas_copy_rhs,
                            cblas_beta_zero_accum,
                        )?;
                        gemm_secs += gemm_t0.elapsed().as_secs_f64();
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_a[..cur_rows].iter().sum::<f64>();
                        }
                        if has_next {
                            let decode_t0 = Instant::now();
                            decode_grm_block(
                                packed_flat.as_ref(),
                                bytes_per_snp,
                                n_samples,
                                row_flip_vec.as_ref(),
                                row_maf_vec.as_ref(),
                                &sample_idx,
                                full_sample_fast,
                                method,
                                eps,
                                &byte_lut.code4,
                                next_start,
                                next_end,
                                n,
                                &mut block_b,
                                &mut varsum_b,
                                pool.as_ref(),
                            )?;
                            decode_secs += decode_t0.elapsed().as_secs_f64();
                        }
                    } else {
                        let gemm_t0 = Instant::now();
                        grm_rankk_update(
                            &mut grm,
                            &block_b[..cur_rows * n],
                            cur_rows,
                            n,
                            cblas_copy_rhs,
                            cblas_beta_zero_accum,
                        )?;
                        gemm_secs += gemm_t0.elapsed().as_secs_f64();
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_b[..cur_rows].iter().sum::<f64>();
                        }
                        if has_next {
                            let decode_t0 = Instant::now();
                            decode_grm_block(
                                packed_flat.as_ref(),
                                bytes_per_snp,
                                n_samples,
                                row_flip_vec.as_ref(),
                                row_maf_vec.as_ref(),
                                &sample_idx,
                                full_sample_fast,
                                method,
                                eps,
                                &byte_lut.code4,
                                next_start,
                                next_end,
                                n,
                                &mut block_a,
                                &mut varsum_a,
                                pool.as_ref(),
                            )?;
                            decode_secs += decode_t0.elapsed().as_secs_f64();
                        }
                    }
                }

                let done = cur_end;
                if (done >= last_notified.saturating_add(notify_step)) || (done == m) {
                    last_notified = done;
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done, m))?;
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                    } else {
                        Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                    }
                }

                if !has_next {
                    break;
                }
                cur_start = next_start;
                cur_end = next_end;
                use_a = !use_a;
            }

            let scale = if method == 1 {
                if !(varsum_acc.is_finite() && varsum_acc > 0.0) {
                    return Err("invalid centered GRM denominator in subset path".to_string());
                }
                varsum_acc as f32
            } else {
                m as f32
            };
            if !(scale.is_finite() && scale > 0.0) {
                return Err("invalid GRM scale factor".to_string());
            }
            let inv_scale = 1.0_f32 / scale;
            for i in 0..n {
                let ii = i * n + i;
                grm[ii] *= inv_scale;
                for j in 0..i {
                    let idx_ij = i * n + j;
                    let idx_ji = j * n + i;
                    let v = 0.5_f32 * (grm[idx_ij] + grm[idx_ji]) * inv_scale;
                    grm[idx_ij] = v;
                    grm[idx_ji] = v;
                }
            }
            if stage_timing {
                let total_secs = total_t0.elapsed().as_secs_f64();
                let stage_sum = decode_secs + gemm_secs;
                let overlap_secs = (stage_sum - total_secs).max(0.0_f64);
                let other_secs = if stage_sum <= total_secs {
                    total_secs - stage_sum
                } else {
                    0.0_f64
                };
                let to_pct = |x: f64| -> f64 {
                    if total_secs > 0.0 {
                        x * 100.0 / total_secs
                    } else {
                        0.0
                    }
                };
                eprintln!(
                    "GRM stage timing: decode={:.3}s ({:.1}%), gemm={:.3}s ({:.1}%), \
other={:.3}s ({:.1}%), overlap={:.3}s ({:.1}%), total={:.3}s, row_step={}, \
block_target_mb={:.1}, n_samples={}, full_sample={}, threads={}",
                    decode_secs,
                    to_pct(decode_secs),
                    gemm_secs,
                    to_pct(gemm_secs),
                    other_secs,
                    to_pct(other_secs),
                    overlap_secs,
                    to_pct(overlap_secs),
                    total_secs,
                    row_step,
                    block_target_mb,
                    n,
                    if full_sample_fast { "yes" } else { "no" },
                    if threads > 0 {
                        threads
                    } else {
                        rayon::current_num_threads()
                    }
                );
            }
            Ok(grm)
        })
        .map_err(map_err_string_to_py)?;

    let grm_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n, n), grm_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok(grm_arr)
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    method=1,
    maf_threshold=0.02,
    max_missing_rate=0.05,
    block_cols=2048,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_bed_f32<'py>(
    py: Python<'py>,
    prefix: String,
    method: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, usize, usize)> {
    let maf_thr = maf_threshold.clamp(0.0, 0.5);
    let miss_thr = max_missing_rate.clamp(0.0, 1.0);
    let (
        packed_arr,
        _miss_arr,
        maf_arr,
        _std_arr,
        row_flip_arr,
        _site_keep_arr,
        n_samples,
        _n_total_sites,
    ) = crate::gfreader::prepare_bed_2bit_packed(py, prefix, maf_thr, miss_thr, 0.0_f32, false)?;
    let packed_ro = packed_arr.readonly();
    let packed_view = packed_ro.as_array();
    if packed_view.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed BED payload must be 2D (m, bytes_per_snp).",
        ));
    }
    let eff_m = packed_view.shape()[0];
    if eff_m == 0 {
        return Err(PyRuntimeError::new_err(
            "No SNPs remained after packed BED filtering; GRM is empty.",
        ));
    }
    let maf_ro = maf_arr.readonly();
    let row_flip_ro = row_flip_arr.readonly();
    if maf_ro.len() != eff_m || row_flip_ro.len() != eff_m {
        return Err(PyRuntimeError::new_err(format!(
            "packed BED metadata length mismatch: packed_rows={eff_m}, maf={}, row_flip={}",
            maf_ro.len(),
            row_flip_ro.len()
        )));
    }

    let grm = grm_packed_f32(
        py,
        packed_ro,
        n_samples,
        row_flip_ro,
        maf_ro,
        None,
        method,
        block_cols,
        threads,
        progress_callback,
        progress_every,
    )?;
    Ok((grm, eff_m, n_samples))
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
pub fn grm_packed_f32_with_stats<'py>(
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
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f64>>, f64)> {
    grm_packed_f32_with_stats_impl(
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
    method=1,
    block_cols=2048,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_f64_with_stats<'py>(
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
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    grm_packed_f64_with_stats_impl(
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
