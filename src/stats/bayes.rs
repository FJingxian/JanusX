use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use pyo3::{prelude::*, BoundObject};
use rand::rngs::{OsRng, StdRng};
use rand::{Rng, SeedableRng, TryRngCore};
use rand_distr::{Beta, ChiSquared, Gamma, StandardNormal};
use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, OnceLock};

use crate::bedmath::{decode_standardized_packed_block_f32, is_identity_indices};
use crate::blas::{
    cblas_daxpy_dispatch, cblas_ddot_dispatch, cblas_dgemm_dispatch, CblasInt, OpenBlasThreadGuard,
    CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
};
use crate::stats_common::{get_cached_pool, parse_index_vec_i64_value_error};

fn array1_to_vec(arr: &PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn array2_to_vec(arr: &PyReadonlyArray2<f64>) -> Vec<f64> {
    let view = arr.as_array();
    let (n, p) = view.dim();
    let mut out = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            out.push(view[[i, j]]);
        }
    }
    out
}

fn parse_optional_index_vec_i64(
    indices: Option<&PyReadonlyArray1<i64>>,
    upper_bound: usize,
    label: &str,
) -> PyResult<Vec<usize>> {
    match indices {
        Some(arr) => parse_index_vec_i64_value_error(arr.as_slice()?, upper_bound, label),
        None => Ok(Vec::new()),
    }
}

#[derive(Debug)]
struct PackedBayesTraceResult {
    beta: Vec<f64>,
    alpha: Vec<f64>,
    vare: f64,
    h2_mean: f64,
    var_h2: f64,
    prob_in_mean: f64,
    n_active_mean: f64,
    iter_trace: Vec<i64>,
    h2_trace: Vec<f64>,
    var_e_trace: Vec<f64>,
    prob_in_trace: Vec<f64>,
    n_active_trace: Vec<f64>,
    full_iter_trace: Vec<i64>,
    full_h2_trace: Vec<f64>,
    full_var_e_trace: Vec<f64>,
    full_prob_in_trace: Vec<f64>,
    full_n_active_trace: Vec<f64>,
    beta_trace_indices: Vec<i64>,
    beta_trace: Vec<f64>,
}

fn posterior_keep_iters(n_iter: usize, burnin: usize, thin: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for it in 0..n_iter {
        if it >= burnin && ((it - burnin) % thin == 0) {
            out.push((it + 1) as i64);
        }
    }
    out
}

fn fill_beta_trace_row(
    beta_trace: &mut [f64],
    row_idx: usize,
    n_trace_snps: usize,
    trace_snp_indices: &[usize],
    beta: &[f64],
    d: Option<&[u8]>,
) {
    if n_trace_snps == 0 {
        return;
    }
    let row_off = row_idx * n_trace_snps;
    for (k, &j) in trace_snp_indices.iter().enumerate() {
        let val = if let Some(mask) = d {
            (mask[j] as f64) * beta[j]
        } else {
            beta[j]
        };
        beta_trace[row_off + k] = val;
    }
}

fn packed_trace_result_to_pydict<'py>(
    py: Python<'py>,
    res: PackedBayesTraceResult,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py).into_bound();
    let beta_py = res.beta.into_pyarray(py);
    let alpha_py = res.alpha.into_pyarray(py);
    let iter_trace_py = res.iter_trace.into_pyarray(py);
    let h2_trace_py = res.h2_trace.into_pyarray(py);
    let var_e_trace_py = res.var_e_trace.into_pyarray(py);
    let prob_in_trace_py = res.prob_in_trace.into_pyarray(py);
    let n_active_trace_py = res.n_active_trace.into_pyarray(py);
    let full_iter_trace_py = res.full_iter_trace.into_pyarray(py);
    let full_h2_trace_py = res.full_h2_trace.into_pyarray(py);
    let full_var_e_trace_py = res.full_var_e_trace.into_pyarray(py);
    let full_prob_in_trace_py = res.full_prob_in_trace.into_pyarray(py);
    let full_n_active_trace_py = res.full_n_active_trace.into_pyarray(py);
    let beta_trace_indices_py = res.beta_trace_indices.into_pyarray(py);
    let beta_trace_arr = Array2::from_shape_vec(
        (iter_trace_py.len(), beta_trace_indices_py.len()),
        res.beta_trace,
    )
    .map_err(|e| PyValueError::new_err(format!("invalid beta trace shape: {e}")))?;
    let beta_trace_py = PyArray2::from_owned_array(py, beta_trace_arr);
    out.set_item("beta", beta_py)?;
    out.set_item("alpha", alpha_py)?;
    out.set_item("vare", res.vare)?;
    out.set_item("h2_mean", res.h2_mean)?;
    out.set_item("var_h2", res.var_h2)?;
    out.set_item("prob_in_mean", res.prob_in_mean)?;
    out.set_item("n_active_mean", res.n_active_mean)?;
    out.set_item("iter_trace", iter_trace_py)?;
    out.set_item("h2_trace", h2_trace_py)?;
    out.set_item("var_e_trace", var_e_trace_py)?;
    out.set_item("prob_in_trace", prob_in_trace_py)?;
    out.set_item("n_active_trace", n_active_trace_py)?;
    out.set_item("full_iter_trace", full_iter_trace_py)?;
    out.set_item("full_h2_trace", full_h2_trace_py)?;
    out.set_item("full_var_e_trace", full_var_e_trace_py)?;
    out.set_item("full_prob_in_trace", full_prob_in_trace_py)?;
    out.set_item("full_n_active_trace", full_n_active_trace_py)?;
    out.set_item("beta_trace_indices", beta_trace_indices_py)?;
    out.set_item("beta_trace", beta_trace_py)?;
    Ok(out)
}

#[inline]
fn row_major_block_mul_vec_f64(
    block: &[f64],
    rows: usize,
    cols: usize,
    vec: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            rows as CblasInt,
            1 as CblasInt,
            cols as CblasInt,
            1.0_f64,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            cols as CblasInt,
            0.0_f64,
            out.as_mut_ptr(),
            rows as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for r in 0..rows {
            let row = &block[r * cols..(r + 1) * cols];
            let mut acc = 0.0_f64;
            for c in 0..cols {
                acc += row[c] * vec[c];
            }
            out[r] = acc;
        }
    }
}

#[inline]
fn row_major_block_t_mul_vec_f64(
    block: &[f64],
    rows: usize,
    cols: usize,
    vec: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            cols as CblasInt,
            1 as CblasInt,
            rows as CblasInt,
            1.0_f64,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            rows as CblasInt,
            0.0_f64,
            out.as_mut_ptr(),
            cols as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for c in 0..cols {
            let mut acc = 0.0_f64;
            for r in 0..rows {
                acc += block[r * cols + c] * vec[r];
            }
            out[c] = acc;
        }
    }
}

#[inline]
fn row_major_xtx_f64(x: &[f64], n: usize, q: usize, xtx_out: &mut [f64]) {
    debug_assert_eq!(x.len(), n.saturating_mul(q));
    debug_assert_eq!(xtx_out.len(), q.saturating_mul(q));
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            q as CblasInt,
            q as CblasInt,
            n as CblasInt,
            1.0_f64,
            x.as_ptr(),
            q as CblasInt,
            x.as_ptr(),
            q as CblasInt,
            0.0_f64,
            xtx_out.as_mut_ptr(),
            q as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for i in 0..q {
            for j in 0..q {
                let mut acc = 0.0_f64;
                for r in 0..n {
                    acc += x[r * q + i] * x[r * q + j];
                }
                xtx_out[i * q + j] = acc;
            }
        }
    }
}

#[inline]
fn ddot_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return 0.0_f64;
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        if x.len() <= (CblasInt::MAX as usize) {
            unsafe {
                return cblas_ddot_dispatch(
                    x.len() as CblasInt,
                    x.as_ptr(),
                    1 as CblasInt,
                    y.as_ptr(),
                    1 as CblasInt,
                );
            }
        }
    }
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (*xi) * (*yi))
        .sum::<f64>()
}

#[inline]
fn daxpy_inplace_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    if alpha == 0.0_f64 || x.is_empty() {
        return;
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        if x.len() <= (CblasInt::MAX as usize) {
            unsafe {
                cblas_daxpy_dispatch(
                    x.len() as CblasInt,
                    alpha,
                    x.as_ptr(),
                    1 as CblasInt,
                    y.as_mut_ptr(),
                    1 as CblasInt,
                );
            }
            return;
        }
    }
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = alpha.mul_add(*xi, *yi);
    }
}

#[allow(clippy::too_many_arguments)]
fn update_alpha_gauss_seidel_blas(
    x: &[f64],
    n: usize,
    q: usize,
    inv_var_e: f64,
    inv_var_b_fixed: f64,
    x2_x: &[f64],
    xtx: &[f64],
    alpha: &mut [f64],
    r: &mut [f64],
    xtr_buf: &mut [f64],
    delta_alpha: &mut [f64],
    tmp_n: &mut [f64],
    rng: &mut StdRng,
) {
    debug_assert_eq!(x.len(), n.saturating_mul(q));
    debug_assert_eq!(x2_x.len(), q);
    debug_assert_eq!(xtx.len(), q.saturating_mul(q));
    debug_assert_eq!(alpha.len(), q);
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(xtr_buf.len(), q);
    debug_assert_eq!(delta_alpha.len(), q);
    debug_assert_eq!(tmp_n.len(), n);

    row_major_block_t_mul_vec_f64(x, n, q, r, xtr_buf);
    delta_alpha.fill(0.0_f64);

    for k in 0..q {
        let rhs = xtr_buf[k].mul_add(inv_var_e, x2_x[k] * alpha[k] * inv_var_e);
        let c = x2_x[k].mul_add(inv_var_e, inv_var_b_fixed);
        let z_alpha: f64 = rng.sample(StandardNormal);
        let new_alpha = rhs / c + (1.0_f64 / c).sqrt() * z_alpha;
        let delta = alpha[k] - new_alpha;
        alpha[k] = new_alpha;
        delta_alpha[k] = delta;
        if delta != 0.0_f64 {
            for t in 0..q {
                xtr_buf[t] += delta * xtx[t * q + k];
            }
        }
    }

    row_major_block_mul_vec_f64(x, n, q, delta_alpha, tmp_n);
    for i in 0..n {
        r[i] += tmp_n[i];
    }
}

struct PackedByteLut {
    code4: [[u8; 4]; 256],
}

fn packed_byte_lut() -> &'static PackedByteLut {
    static LUT: OnceLock<PackedByteLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut code4 = [[0u8; 4]; 256];
        for b in 0u16..=255 {
            let byte = b as u8;
            for lane in 0..4usize {
                code4[byte as usize][lane] = (byte >> (lane * 2)) & 0b11;
            }
        }
        PackedByteLut { code4 }
    })
}

fn parse_env_truthy(raw: &str) -> Option<bool> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    if matches!(s.as_str(), "1" | "true" | "yes" | "y" | "on") {
        return Some(true);
    }
    if matches!(s.as_str(), "0" | "false" | "no" | "n" | "off") {
        return Some(false);
    }
    None
}

#[inline]
fn bayes_packed_block_rows(n: usize, p: usize) -> usize {
    if n == 0 || p == 0 {
        return 1;
    }
    let default_target_bytes: usize = 32 * 1024 * 1024;
    let bytes_per_row = n.saturating_mul(std::mem::size_of::<f64>()).max(1);
    let mut rows = (default_target_bytes / bytes_per_row).max(1);
    rows = rows.clamp(8, 2048);
    if let Ok(raw) = env::var("JX_BAYES_PACKED_ROW_BLOCK") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            if v > 0 {
                rows = v;
            }
        }
    }
    rows.clamp(1, p)
}

#[inline]
fn bayes_packed_predecode_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        if let Ok(raw) = env::var("JX_BAYES_PACKED_PREDECODE") {
            if let Some(v) = parse_env_truthy(&raw) {
                return v;
            }
        }
        true
    })
}

#[inline]
fn bayes_packed_predecode_max_mb() -> usize {
    static MAX_MB: OnceLock<usize> = OnceLock::new();
    *MAX_MB.get_or_init(|| {
        if let Ok(raw) = env::var("JX_BAYES_PACKED_PREDECODE_MAX_MB") {
            if let Ok(v) = raw.trim().parse::<usize>() {
                if v > 0 {
                    return v;
                }
            }
        }
        768usize
    })
}

#[inline]
fn bayes_packed_blas_threads() -> usize {
    // Default policy stays "BLAS single-thread + Rayon parallel decode".
    // Env override for experiments:
    //   JX_BAYES_PACKED_BLAS_THREADS=4  -> force BLAS 4 threads in Bayes packed kernels
    //   JX_BAYES_PACKED_BLAS_THREADS=0  -> do not override BLAS threads
    if let Ok(raw) = env::var("JX_BAYES_PACKED_BLAS_THREADS") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            return v;
        }
    }
    1usize
}

#[inline]
fn bayes_packed_should_predecode_dense(n: usize, p: usize) -> bool {
    if !bayes_packed_predecode_enabled() {
        return false;
    }
    let elem_cnt = n.saturating_mul(p);
    if elem_cnt == 0 {
        return false;
    }
    let dense_bytes = elem_cnt.saturating_mul(std::mem::size_of::<f64>());
    let cap_bytes = bayes_packed_predecode_max_mb().saturating_mul(1024 * 1024);
    if dense_bytes > cap_bytes {
        return false;
    }
    // Keep a lightweight guard on huge dimensions even if memory cap allows it.
    elem_cnt <= 80_000_000usize
}

#[inline]
fn decode_packed_block_standardized_into(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_start: usize,
    row_end: usize,
    sample_idx: &[usize],
    full_sample_fast: bool,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    code4_lut: &[[u8; 4]; 256],
    out_block: &mut [f64],
    n: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
    scratch_f32: &mut Vec<f32>,
) -> Result<(), String> {
    let block_rows = row_end - row_start;
    debug_assert_eq!(out_block.len(), block_rows * n);
    if scratch_f32.len() < block_rows * n {
        scratch_f32.resize(block_rows * n, 0.0_f32);
    }
    let tmp = &mut scratch_f32[..block_rows * n];
    decode_standardized_packed_block_f32(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_idx,
        full_sample_fast,
        row_start,
        tmp,
        code4_lut,
        pool,
    )?;
    for (dst, &src) in out_block.iter_mut().zip(tmp.iter()) {
        *dst = src as f64;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_predecode_packed_dense_f64(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    n: usize,
    p: usize,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<Option<Vec<f64>>, String> {
    if !bayes_packed_should_predecode_dense(n, p) {
        return Ok(None);
    }
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let block_rows = bayes_packed_block_rows(n, p);
    let mut dense_f32 = vec![0.0_f32; p * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_standardized_packed_block_f32(
            packed_flat,
            bytes_per_snp,
            n_samples,
            row_flip,
            row_mean,
            row_inv_sd,
            sample_idx,
            full_sample_fast,
            st,
            &mut dense_f32[st * n..(st + br) * n],
            code4_lut,
            pool,
        )?;
    }
    let dense_f64: Vec<f64> = dense_f32.into_iter().map(|v| v as f64).collect();
    Ok(Some(dense_f64))
}

fn genetic_variance_from_residual(
    y: &[f64],
    r: &[f64],
    x: &[f64],
    alpha: &[f64],
    n: usize,
    q: usize,
) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut mean_g = 0.0;
    let mut m2 = 0.0;
    for i in 0..n {
        let mut xa = 0.0;
        for k in 0..q {
            xa += x[i * q + k] * alpha[k];
        }
        let g = y[i] - r[i] - xa;
        let delta = g - mean_g;
        mean_g += delta / (i as f64 + 1.0);
        let delta2 = g - mean_g;
        m2 += delta * delta2;
    }
    m2 / (n as f64 - 1.0)
}

#[inline]
fn bayesb_rate0(shape0: f64, rate0_opt: Option<f64>, s0_b: f64) -> Result<f64, String> {
    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }
    Ok(rate0)
}

#[inline]
fn bayes_inclusion_prior_shapes(prob_in_init: f64, counts: f64) -> (f64, f64) {
    let counts_in = (counts * prob_in_init).max(f64::MIN_POSITIVE);
    let counts_out = (counts * (1.0 - prob_in_init)).max(f64::MIN_POSITIVE);
    (counts_in, counts_out)
}

#[inline]
fn sample_gamma_with_rate<R: Rng + ?Sized>(
    rng: &mut R,
    shape: f64,
    rate: f64,
) -> Result<f64, String> {
    if !(shape.is_finite() && shape > 0.0) {
        return Err("Gamma shape must be positive and finite".to_string());
    }
    if !(rate.is_finite() && rate > 0.0) {
        return Err("Gamma rate must be positive and finite".to_string());
    }
    let gamma = Gamma::new(shape, 1.0).map_err(|e| e.to_string())?;
    Ok(rng.sample(gamma) / rate)
}

#[inline]
fn bayes_positive_floor(x: f64) -> f64 {
    x.max(1e-300_f64)
}

fn bayesb_core_impl(
    y: &[f64],
    m: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
    };

    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += v * v;
            msum += v;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = bayesb_rate0(shape0, rate0_opt, s0_b)?;

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;
    let mut prob_in = prob_in_init;
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_init, counts);

    let mut alpha = vec![0.0; q];
    let mut x2_x = vec![0.0; q];
    for k in 0..q {
        let mut s2 = 0.0;
        for i in 0..n {
            let v = x[i * q + k];
            s2 += v * v;
        }
        x2_x[k] = s2;
    }

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let b_old = beta[j];
            if b_old != 0.0 {
                // Remove current marker effect first so r represents r_{-j}.
                daxpy_inplace_f64(b_old, m_j, &mut r);
            }

            let xe = ddot_f64(&r, m_j);

            let c = x2[j] * inv_var_e + 1.0 / var_b[j];
            if !(c.is_finite() && c > 0.0) {
                return Err("Non-positive posterior precision in BayesB beta update".to_string());
            }
            let rhs = xe * inv_var_e;
            // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
            let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b[j] * c).ln();
            let log_odds = log_odds_prior + log_bf10;
            let p_in = if log_odds >= 0.0 {
                1.0 / (1.0 + (-log_odds).exp())
            } else {
                let e = log_odds.exp();
                e / (1.0 + e)
            };
            let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
            d[j] = new_d;

            if new_d == 1 {
                let z_beta: f64 = rng.sample(StandardNormal);
                let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;
                daxpy_inplace_f64(-new_beta, m_j, &mut r);
                beta[j] = new_beta;
            } else {
                beta[j] = 0.0;
            }
        }

        let mut n_active = 0usize;
        for j in 0..p {
            if d[j] == 1 {
                let beta2 = beta[j] * beta[j];
                var_b[j] = bayes_positive_floor((s + beta2) / rng.sample(chi_b_active));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err("BayesB var_b became non-finite or non-positive".to_string());
                }
                n_active += 1;
            } else {
                var_b[j] = bayes_positive_floor(s / rng.sample(chi_b_inactive));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err(
                        "BayesB inactive var_b became non-finite or non-positive".to_string()
                    );
                }
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating BayesB S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        s = bayes_positive_floor(sample_gamma_with_rate(&mut rng, tmp_shape, tmp_rate)?);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesB S became non-finite or non-positive".to_string());
        }

        let mrk_in = n_active as f64;
        let a = mrk_in + counts_in;
        let b = (p as f64 - mrk_in) + counts_out;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

        let ss_e = ddot_f64(&r, &r) + s0_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesB var_e became non-finite or non-positive".to_string());
        }

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
    ))
}

fn bayescpi_core_impl(
    y: &[f64],
    m: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64, f64, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
    };

    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += v * v;
            msum += v;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = s0_b;
    let mut prob_in = prob_in_init;

    let counts_in = counts * prob_in_init;
    let counts_out = counts - counts_in;

    let mut alpha = vec![0.0; q];
    let mut x2_x = vec![0.0; q];
    for k in 0..q {
        let mut s2 = 0.0;
        for i in 0..n {
            let v = x[i * q + k];
            s2 += v * v;
        }
        x2_x[k] = s2;
    }

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut varb_sum = 0.0;
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;
    let mut chi_b_cache: HashMap<usize, ChiSquared<f64>> = HashMap::new();

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let b_old = beta[j];
            if d[j] == 1 {
                // Remove current marker effect first so r represents r_{-j}.
                daxpy_inplace_f64(b_old, m_j, &mut r);
            }

            let xe = ddot_f64(&r, m_j);

            let c = x2[j] * inv_var_e + 1.0 / var_b;
            if !(c.is_finite() && c > 0.0) {
                return Err("Non-positive posterior precision in BayesCpi beta update".to_string());
            }
            let rhs = xe * inv_var_e;
            // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
            let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b * c).ln();
            let log_odds = log_odds_prior + log_bf10;
            let p_in = if log_odds >= 0.0 {
                1.0 / (1.0 + (-log_odds).exp())
            } else {
                let e = log_odds.exp();
                e / (1.0 + e)
            };
            let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
            d[j] = new_d;

            if new_d == 1 {
                let z_beta: f64 = rng.sample(StandardNormal);
                let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;
                daxpy_inplace_f64(-new_beta, m_j, &mut r);
                beta[j] = new_beta;
            } else {
                beta[j] = 0.0;
            }
        }

        let mut mrk_in_usize = 0usize;
        let mut ss_b = 0.0;
        for j in 0..p {
            if d[j] == 1 {
                ss_b += beta[j] * beta[j];
                mrk_in_usize += 1;
            }
        }
        ss_b += s0_b;
        let chi_b_eff = if let Some(dist) = chi_b_cache.get(&mrk_in_usize) {
            *dist
        } else {
            let dist = ChiSquared::new(df0_b + mrk_in_usize as f64).map_err(|e| e.to_string())?;
            chi_b_cache.insert(mrk_in_usize, dist);
            dist
        };
        var_b = ss_b / rng.sample(chi_b_eff);
        if !(var_b.is_finite() && var_b > 0.0) {
            return Err("BayesCpi var_b became non-finite or non-positive".to_string());
        }

        let mrk_in = mrk_in_usize as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

        let ss_e = ddot_f64(&r, &r) + s0_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesCpi var_e became non-finite or non-positive".to_string());
        }

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            varb_sum += var_b;
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    varb_sum *= inv_keep;
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
    ))
}

fn bayesa_core_impl(
    y: &[f64],
    m: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    _min_abs_beta: f64,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
    };

    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += v * v;
            msum += v;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0)
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = vec![0.0; q];
    let mut x2_x = vec![0.0; q];
    for k in 0..q {
        let mut s2 = 0.0;
        for i in 0..n {
            let v = x[i * q + k];
            s2 += v * v;
        }
        x2_x[k] = s2;
    }

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let mut rhs = ddot_f64(m_j, &r);
            rhs = rhs * inv_var_e + x2[j] * beta[j] * inv_var_e;
            let c = x2[j] * inv_var_e + 1.0 / var_b[j];
            let z_beta: f64 = rng.sample(StandardNormal);
            let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;

            let delta = beta[j] - new_beta;
            daxpy_inplace_f64(delta, m_j, &mut r);
            beta[j] = new_beta;
        }

        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
            if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                return Err("BayesA var_b became non-finite or non-positive".to_string());
            }
        }

        let mut tmp_rate = 0.0;
        for j in 0..p {
            tmp_rate += 1.0 / var_b[j];
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesA S became non-finite or non-positive".to_string());
        }

        let ss_e = ddot_f64(&r, &r) + s0_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesA var_e became non-finite or non-positive".to_string());
        }

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    Ok((beta_sum, alpha_sum, varb_sum, var_e_sum, h2_mean, var_h2))
}

fn bayesa_packed_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    _min_abs_beta: f64,
    seed: Option<u64>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let code4_lut = &packed_byte_lut().code4;
    if let Some(m_dense) = maybe_predecode_packed_dense_f64(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_idx,
        n,
        p,
        code4_lut,
        pool,
    )? {
        return bayesa_core_impl(
            y, &m_dense, x, n, p, q, n_iter, burnin, thin, r2, df0_b, shape0, rate0_opt, s0_b_opt,
            df0_e, s0_e_opt, 0.0, seed,
        );
    }
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let block_rows = bayes_packed_block_rows(n, p);
    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    let mut m_block = vec![0.0_f64; block_rows * n];
    let mut scratch_decode_f32 = vec![0.0_f32; block_rows * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_packed_block_standardized_into(
            packed_flat,
            bytes_per_snp,
            n_samples,
            st,
            ed,
            sample_idx,
            full_sample_fast,
            row_flip,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut m_block[..br * n],
            n,
            pool,
            &mut scratch_decode_f32,
        )?;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += v * v;
                msum += v;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0)
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        for st in (0..p).step_by(block_rows) {
            let ed = (st + block_rows).min(p);
            let br = ed - st;
            decode_packed_block_standardized_into(
                packed_flat,
                bytes_per_snp,
                n_samples,
                st,
                ed,
                sample_idx,
                full_sample_fast,
                row_flip,
                row_mean,
                row_inv_sd,
                code4_lut,
                &mut m_block[..br * n],
                n,
                pool,
                &mut scratch_decode_f32,
            )?;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let mut rhs = ddot_f64(m_row, &r);
                rhs = rhs * inv_var_e + x2[j] * beta[j] * inv_var_e;
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                let z_beta: f64 = rng.sample(StandardNormal);
                let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;

                let delta = beta[j] - new_beta;
                daxpy_inplace_f64(delta, m_row, &mut r);
                beta[j] = new_beta;
            }
        }

        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
            if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                return Err("BayesA packed var_b became non-finite or non-positive".to_string());
            }
        }

        let mut tmp_rate = 0.0;
        for j in 0..p {
            tmp_rate += 1.0 / var_b[j];
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesA packed S became non-finite or non-positive".to_string());
        }

        let ss_e = ddot_f64(&r, &r) + s0_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesA packed var_e became non-finite or non-positive".to_string());
        }

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    Ok((beta_sum, alpha_sum, varb_sum, var_e_sum, h2_mean, var_h2))
}

fn bayesb_packed_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let code4_lut = &packed_byte_lut().code4;
    if let Some(m_dense) = maybe_predecode_packed_dense_f64(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_idx,
        n,
        p,
        code4_lut,
        pool,
    )? {
        return bayesb_core_impl(
            y,
            &m_dense,
            x,
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            prob_in_init,
            counts,
            df0_e,
            s0_e_opt,
            seed,
        );
    }
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let block_rows = bayes_packed_block_rows(n, p);
    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    let mut m_block = vec![0.0_f64; block_rows * n];
    let mut scratch_decode_f32 = vec![0.0_f32; block_rows * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_packed_block_standardized_into(
            packed_flat,
            bytes_per_snp,
            n_samples,
            st,
            ed,
            sample_idx,
            full_sample_fast,
            row_flip,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut m_block[..br * n],
            n,
            pool,
            &mut scratch_decode_f32,
        )?;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += v * v;
                msum += v;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = bayesb_rate0(shape0, rate0_opt, s0_b)?;

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut prob_in = prob_in_init;
    let mut s = s0_b;
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_init, counts);

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for st in (0..p).step_by(block_rows) {
            let ed = (st + block_rows).min(p);
            let br = ed - st;
            decode_packed_block_standardized_into(
                packed_flat,
                bytes_per_snp,
                n_samples,
                st,
                ed,
                sample_idx,
                full_sample_fast,
                row_flip,
                row_mean,
                row_inv_sd,
                code4_lut,
                &mut m_block[..br * n],
                n,
                pool,
                &mut scratch_decode_f32,
            )?;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                if b_old != 0.0 {
                    // Remove current marker effect first so r represents r_{-j}.
                    daxpy_inplace_f64(b_old, m_row, &mut r);
                }

                let xe = ddot_f64(&r, m_row);
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesB packed beta update".to_string(),
                    );
                }
                let rhs = xe * inv_var_e;
                // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b[j] * c).ln();
                let log_odds = log_odds_prior + log_bf10;
                let p_in = if log_odds >= 0.0 {
                    1.0 / (1.0 + (-log_odds).exp())
                } else {
                    let e = log_odds.exp();
                    e / (1.0 + e)
                };
                let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                d[j] = new_d;

                if new_d == 1 {
                    let z_beta: f64 = rng.sample(StandardNormal);
                    let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;
                    daxpy_inplace_f64(-new_beta, m_row, &mut r);
                    beta[j] = new_beta;
                } else {
                    beta[j] = 0.0;
                }
            }
        }

        let mut n_active = 0usize;
        for j in 0..p {
            if d[j] == 1 {
                let beta2 = beta[j] * beta[j];
                var_b[j] = bayes_positive_floor((s + beta2) / rng.sample(chi_b_active));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err("BayesB packed var_b became non-finite or non-positive".to_string());
                }
                n_active += 1;
            } else {
                var_b[j] = bayes_positive_floor(s / rng.sample(chi_b_inactive));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err(
                        "BayesB packed inactive var_b became non-finite or non-positive"
                            .to_string(),
                    );
                }
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err(
                "Gamma rate became non-positive while updating BayesB packed S".to_string(),
            );
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        s = bayes_positive_floor(sample_gamma_with_rate(&mut rng, tmp_shape, tmp_rate)?);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesB packed S became non-finite or non-positive".to_string());
        }

        let mrk_in = n_active as f64;
        let a = mrk_in + counts_in;
        let b = (p as f64 - mrk_in) + counts_out;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

        let ss_e = ddot_f64(&r, &r) + s0_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesB packed var_e became non-finite or non-positive".to_string());
        }

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
    ))
}

fn bayescpi_packed_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64, f64, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let code4_lut = &packed_byte_lut().code4;
    if let Some(m_dense) = maybe_predecode_packed_dense_f64(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_idx,
        n,
        p,
        code4_lut,
        pool,
    )? {
        return bayescpi_core_impl(
            y,
            &m_dense,
            x,
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b_opt,
            prob_in_init,
            counts,
            df0_e,
            s0_e_opt,
            seed,
        );
    }
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let block_rows = bayes_packed_block_rows(n, p);
    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    let mut m_block = vec![0.0_f64; block_rows * n];
    let mut scratch_decode_f32 = vec![0.0_f32; block_rows * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_packed_block_standardized_into(
            packed_flat,
            bytes_per_snp,
            n_samples,
            st,
            ed,
            sample_idx,
            full_sample_fast,
            row_flip,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut m_block[..br * n],
            n,
            pool,
            &mut scratch_decode_f32,
        )?;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += v * v;
                msum += v;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = s0_b;
    let mut prob_in = prob_in_init;
    let counts_in = counts * prob_in_init;
    let counts_out = counts - counts_in;

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut varb_sum = 0.0;
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;
    let mut chi_b_cache: HashMap<usize, ChiSquared<f64>> = HashMap::new();

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for st in (0..p).step_by(block_rows) {
            let ed = (st + block_rows).min(p);
            let br = ed - st;
            decode_packed_block_standardized_into(
                packed_flat,
                bytes_per_snp,
                n_samples,
                st,
                ed,
                sample_idx,
                full_sample_fast,
                row_flip,
                row_mean,
                row_inv_sd,
                code4_lut,
                &mut m_block[..br * n],
                n,
                pool,
                &mut scratch_decode_f32,
            )?;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                if d[j] == 1 {
                    // Remove current marker effect first so r represents r_{-j}.
                    daxpy_inplace_f64(b_old, m_row, &mut r);
                }

                let xe = ddot_f64(&r, m_row);
                let c = x2[j] * inv_var_e + 1.0 / var_b;
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesCpi packed beta update"
                            .to_string(),
                    );
                }
                let rhs = xe * inv_var_e;
                // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b * c).ln();
                let log_odds = log_odds_prior + log_bf10;
                let p_in = if log_odds >= 0.0 {
                    1.0 / (1.0 + (-log_odds).exp())
                } else {
                    let e = log_odds.exp();
                    e / (1.0 + e)
                };
                let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                d[j] = new_d;

                if new_d == 1 {
                    let z_beta: f64 = rng.sample(StandardNormal);
                    let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;
                    daxpy_inplace_f64(-new_beta, m_row, &mut r);
                    beta[j] = new_beta;
                } else {
                    beta[j] = 0.0;
                }
            }
        }

        let mut mrk_in_usize = 0usize;
        let mut ss_b = 0.0;
        for j in 0..p {
            if d[j] == 1 {
                ss_b += beta[j] * beta[j];
                mrk_in_usize += 1;
            }
        }
        ss_b += s0_b;
        let chi_b_eff = if let Some(dist) = chi_b_cache.get(&mrk_in_usize) {
            *dist
        } else {
            let dist = ChiSquared::new(df0_b + mrk_in_usize as f64).map_err(|e| e.to_string())?;
            chi_b_cache.insert(mrk_in_usize, dist);
            dist
        };
        var_b = ss_b / rng.sample(chi_b_eff);
        if !(var_b.is_finite() && var_b > 0.0) {
            return Err("BayesCpi packed var_b became non-finite or non-positive".to_string());
        }

        let mrk_in = mrk_in_usize as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

        let ss_e = ddot_f64(&r, &r) + s0_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesCpi packed var_e became non-finite or non-positive".to_string());
        }

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            varb_sum += var_b;
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    varb_sum *= inv_keep;
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    s0_e = None,
    min_abs_beta = 1e-9,
    seed = None
))]
pub fn bayesa(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: PyReadonlyArray2<f64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    s0_e: Option<f64>,
    min_abs_beta: f64,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_shape = m.shape();
    if m_shape[1] != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_shape[0];
    let m_vec: Cow<'_, [f64]> = match m.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array2_to_vec(&m)),
    };

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesa_core_impl(
            y_vec.as_ref(),
            m_vec.as_ref(),
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            s0_e,
            min_abs_beta,
            seed,
        )
    });

    match result {
        Ok((beta, alpha, varb, vare, h2_mean, var_h2)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((beta_py, alpha_py, varb_py, vare, h2_mean, var_h2))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    seed = None
))]
pub fn bayesb(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: PyReadonlyArray2<f64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_shape = m.shape();
    if m_shape[1] != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_shape[0];
    let m_vec: Cow<'_, [f64]> = match m.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array2_to_vec(&m)),
    };

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesb_core_impl(
            y_vec.as_ref(),
            m_vec.as_ref(),
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
        )
    });

    match result {
        Ok((beta, alpha, varb, vare, h2_mean, var_h2, prob_in_mean, n_active_mean)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_py,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    seed = None
))]
pub fn bayescpi(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: PyReadonlyArray2<f64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = s0_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_e must be > 0"));
        }
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_shape = m.shape();
    if m_shape[1] != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_shape[0];
    let m_vec: Cow<'_, [f64]> = match m.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array2_to_vec(&m)),
    };

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayescpi_core_impl(
            y_vec.as_ref(),
            m_vec.as_ref(),
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
        )
    });

    match result {
        Ok((beta, alpha, varb_mean, vare, h2_mean, var_h2, prob_in_mean, n_active_mean)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_mean,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    s0_e = None,
    min_abs_beta = 1e-9,
    threads = 0,
    seed = None
))]
pub fn bayesa_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    s0_e: Option<f64>,
    min_abs_beta: f64,
    threads: usize,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();

    let result = py.detach(|| {
        bayesa_packed_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            s0_e,
            min_abs_beta,
            seed,
            pool_ref,
        )
    });

    match result {
        Ok((beta, alpha, varb, vare, h2_mean, var_h2)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((beta_py, alpha_py, varb_py, vare, h2_mean, var_h2))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    threads = 0,
    seed = None
))]
pub fn bayesb_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    threads: usize,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();

    let result = py.detach(|| {
        bayesb_packed_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
            pool_ref,
        )
    });

    match result {
        Ok((beta, alpha, varb, vare, h2_mean, var_h2, prob_in_mean, n_active_mean)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_py,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    threads = 0,
    seed = None
))]
pub fn bayescpi_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    threads: usize,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = s0_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_e must be > 0"));
        }
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();

    let result = py.detach(|| {
        bayescpi_packed_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
            pool_ref,
        )
    });

    match result {
        Ok((beta, alpha, varb_mean, vare, h2_mean, var_h2, prob_in_mean, n_active_mean)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_mean,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesa_packed_trace_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
    trace_snp_indices: &[usize],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<PackedBayesTraceResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let code4_lut = &packed_byte_lut().code4;
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let block_rows = bayes_packed_block_rows(n, p);
    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    let mut m_block = vec![0.0_f64; block_rows * n];
    let mut scratch_decode_f32 = vec![0.0_f32; block_rows * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_packed_block_standardized_into(
            packed_flat,
            bytes_per_snp,
            n_samples,
            st,
            ed,
            sample_idx,
            full_sample_fast,
            row_flip,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut m_block[..br * n],
            n,
            pool,
            &mut scratch_decode_f32,
        )?;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += v * v;
                msum += v;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0)
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let keep_iters = posterior_keep_iters(n_iter, burnin, thin);
    let n_keep_target = keep_iters.len();
    let n_trace_snps = trace_snp_indices.len();
    let mut h2_trace = Vec::with_capacity(n_keep_target);
    let mut var_e_trace = Vec::with_capacity(n_keep_target);
    let prob_in_trace = vec![f64::NAN; n_keep_target];
    let n_active_trace = vec![f64::NAN; n_keep_target];
    let full_iter_trace: Vec<i64> = (1..=n_iter).map(|v| v as i64).collect();
    let mut full_h2_trace = Vec::with_capacity(n_iter);
    let mut full_var_e_trace = Vec::with_capacity(n_iter);
    let full_prob_in_trace = vec![f64::NAN; n_iter];
    let full_n_active_trace = vec![f64::NAN; n_iter];
    let mut beta_trace = vec![0.0_f64; n_keep_target * n_trace_snps];
    let mut keep_row = 0usize;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        for st in (0..p).step_by(block_rows) {
            let ed = (st + block_rows).min(p);
            let br = ed - st;
            decode_packed_block_standardized_into(
                packed_flat,
                bytes_per_snp,
                n_samples,
                st,
                ed,
                sample_idx,
                full_sample_fast,
                row_flip,
                row_mean,
                row_inv_sd,
                code4_lut,
                &mut m_block[..br * n],
                n,
                pool,
                &mut scratch_decode_f32,
            )?;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let mut rhs = ddot_f64(m_row, &r);
                rhs = rhs * inv_var_e + x2[j] * beta[j] * inv_var_e;
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                let z_beta: f64 = rng.sample(StandardNormal);
                let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;

                let delta = beta[j] - new_beta;
                daxpy_inplace_f64(delta, m_row, &mut r);
                beta[j] = new_beta;
            }
        }

        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
            if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                return Err("BayesA packed var_b became non-finite or non-positive".to_string());
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesA packed S became non-finite or non-positive".to_string());
        }

        let ss_e = ddot_f64(&r, &r) + s0_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesA packed var_e became non-finite or non-positive".to_string());
        }

        let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
        let h2 = var_g / (var_g + var_e);
        full_h2_trace.push(h2);
        full_var_e_trace.push(var_e);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            h2_trace.push(h2);
            var_e_trace.push(var_e);
            fill_beta_trace_row(
                &mut beta_trace,
                keep_row,
                n_trace_snps,
                trace_snp_indices,
                &beta,
                None,
            );
            keep_row += 1;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for bj in &mut beta_sum {
        *bj *= inv_keep;
    }
    for ak in &mut alpha_sum {
        *ak *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }

    Ok(PackedBayesTraceResult {
        beta: beta_sum,
        alpha: alpha_sum,
        vare: var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean: f64::NAN,
        n_active_mean: f64::NAN,
        iter_trace: keep_iters,
        h2_trace,
        var_e_trace,
        prob_in_trace,
        n_active_trace,
        full_iter_trace,
        full_h2_trace,
        full_var_e_trace,
        full_prob_in_trace,
        full_n_active_trace,
        beta_trace_indices: trace_snp_indices.iter().map(|&v| v as i64).collect(),
        beta_trace,
    })
}

#[allow(clippy::too_many_arguments)]
fn bayesb_packed_trace_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
    trace_snp_indices: &[usize],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<PackedBayesTraceResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let code4_lut = &packed_byte_lut().code4;
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let block_rows = bayes_packed_block_rows(n, p);
    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    let mut m_block = vec![0.0_f64; block_rows * n];
    let mut scratch_decode_f32 = vec![0.0_f32; block_rows * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_packed_block_standardized_into(
            packed_flat,
            bytes_per_snp,
            n_samples,
            st,
            ed,
            sample_idx,
            full_sample_fast,
            row_flip,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut m_block[..br * n],
            n,
            pool,
            &mut scratch_decode_f32,
        )?;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += v * v;
                msum += v;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = bayesb_rate0(shape0, rate0_opt, s0_b)?;

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut prob_in = prob_in_init;
    let mut s = s0_b;
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_init, counts);

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let keep_iters = posterior_keep_iters(n_iter, burnin, thin);
    let n_keep_target = keep_iters.len();
    let n_trace_snps = trace_snp_indices.len();
    let mut h2_trace = Vec::with_capacity(n_keep_target);
    let mut var_e_trace = Vec::with_capacity(n_keep_target);
    let mut prob_in_trace = Vec::with_capacity(n_keep_target);
    let mut n_active_trace = Vec::with_capacity(n_keep_target);
    let full_iter_trace: Vec<i64> = (1..=n_iter).map(|v| v as i64).collect();
    let mut full_h2_trace = Vec::with_capacity(n_iter);
    let mut full_var_e_trace = Vec::with_capacity(n_iter);
    let mut full_prob_in_trace = Vec::with_capacity(n_iter);
    let mut full_n_active_trace = Vec::with_capacity(n_iter);
    let mut beta_trace = vec![0.0_f64; n_keep_target * n_trace_snps];
    let mut keep_row = 0usize;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for st in (0..p).step_by(block_rows) {
            let ed = (st + block_rows).min(p);
            let br = ed - st;
            decode_packed_block_standardized_into(
                packed_flat,
                bytes_per_snp,
                n_samples,
                st,
                ed,
                sample_idx,
                full_sample_fast,
                row_flip,
                row_mean,
                row_inv_sd,
                code4_lut,
                &mut m_block[..br * n],
                n,
                pool,
                &mut scratch_decode_f32,
            )?;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                if b_old != 0.0 {
                    daxpy_inplace_f64(b_old, m_row, &mut r);
                }

                let xe = ddot_f64(&r, m_row);
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesB packed beta update".to_string(),
                    );
                }
                let rhs = xe * inv_var_e;
                let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b[j] * c).ln();
                let log_odds = log_odds_prior + log_bf10;
                let p_in = if log_odds >= 0.0 {
                    1.0 / (1.0 + (-log_odds).exp())
                } else {
                    let e = log_odds.exp();
                    e / (1.0 + e)
                };
                let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                d[j] = new_d;

                if new_d == 1 {
                    let z_beta: f64 = rng.sample(StandardNormal);
                    let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;
                    daxpy_inplace_f64(-new_beta, m_row, &mut r);
                    beta[j] = new_beta;
                } else {
                    beta[j] = 0.0;
                }
            }
        }

        let mut n_active = 0usize;
        for j in 0..p {
            if d[j] == 1 {
                let beta2 = beta[j] * beta[j];
                var_b[j] = bayes_positive_floor((s + beta2) / rng.sample(chi_b_active));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err("BayesB packed var_b became non-finite or non-positive".to_string());
                }
                n_active += 1;
            } else {
                var_b[j] = bayes_positive_floor(s / rng.sample(chi_b_inactive));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err(
                        "BayesB packed inactive var_b became non-finite or non-positive"
                            .to_string(),
                    );
                }
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err(
                "Gamma rate became non-positive while updating BayesB packed trace S".to_string(),
            );
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        s = bayes_positive_floor(sample_gamma_with_rate(&mut rng, tmp_shape, tmp_rate)?);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesB packed trace S became non-finite or non-positive".to_string());
        }

        let mrk_in = n_active as f64;
        let a = mrk_in + counts_in;
        let b = (p as f64 - mrk_in) + counts_out;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

        let ss_e = ddot_f64(&r, &r) + s0_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesB packed var_e became non-finite or non-positive".to_string());
        }

        let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
        let h2 = var_g / (var_g + var_e);
        full_h2_trace.push(h2);
        full_var_e_trace.push(var_e);
        full_prob_in_trace.push(prob_in);
        full_n_active_trace.push(mrk_in);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += (d[j] as f64) * beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            h2_trace.push(h2);
            var_e_trace.push(var_e);
            prob_in_trace.push(prob_in);
            n_active_trace.push(mrk_in);
            fill_beta_trace_row(
                &mut beta_trace,
                keep_row,
                n_trace_snps,
                trace_snp_indices,
                &beta,
                Some(&d),
            );
            keep_row += 1;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for bj in &mut beta_sum {
        *bj *= inv_keep;
    }
    for ak in &mut alpha_sum {
        *ak *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok(PackedBayesTraceResult {
        beta: beta_sum,
        alpha: alpha_sum,
        vare: var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        iter_trace: keep_iters,
        h2_trace,
        var_e_trace,
        prob_in_trace,
        n_active_trace,
        full_iter_trace,
        full_h2_trace,
        full_var_e_trace,
        full_prob_in_trace,
        full_n_active_trace,
        beta_trace_indices: trace_snp_indices.iter().map(|&v| v as i64).collect(),
        beta_trace,
    })
}

#[allow(clippy::too_many_arguments)]
fn bayescpi_packed_trace_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
    trace_snp_indices: &[usize],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<PackedBayesTraceResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let code4_lut = &packed_byte_lut().code4;
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let block_rows = bayes_packed_block_rows(n, p);
    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    let mut m_block = vec![0.0_f64; block_rows * n];
    let mut scratch_decode_f32 = vec![0.0_f32; block_rows * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_packed_block_standardized_into(
            packed_flat,
            bytes_per_snp,
            n_samples,
            st,
            ed,
            sample_idx,
            full_sample_fast,
            row_flip,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut m_block[..br * n],
            n,
            pool,
            &mut scratch_decode_f32,
        )?;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += v * v;
                msum += v;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = s0_b;
    let mut prob_in = prob_in_init;
    let counts_in = counts * prob_in_init;
    let counts_out = counts - counts_in;

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let keep_iters = posterior_keep_iters(n_iter, burnin, thin);
    let n_keep_target = keep_iters.len();
    let n_trace_snps = trace_snp_indices.len();
    let mut h2_trace = Vec::with_capacity(n_keep_target);
    let mut var_e_trace = Vec::with_capacity(n_keep_target);
    let mut prob_in_trace = Vec::with_capacity(n_keep_target);
    let mut n_active_trace = Vec::with_capacity(n_keep_target);
    let full_iter_trace: Vec<i64> = (1..=n_iter).map(|v| v as i64).collect();
    let mut full_h2_trace = Vec::with_capacity(n_iter);
    let mut full_var_e_trace = Vec::with_capacity(n_iter);
    let mut full_prob_in_trace = Vec::with_capacity(n_iter);
    let mut full_n_active_trace = Vec::with_capacity(n_iter);
    let mut beta_trace = vec![0.0_f64; n_keep_target * n_trace_snps];
    let mut keep_row = 0usize;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;
    let mut chi_b_cache: HashMap<usize, ChiSquared<f64>> = HashMap::new();

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for st in (0..p).step_by(block_rows) {
            let ed = (st + block_rows).min(p);
            let br = ed - st;
            decode_packed_block_standardized_into(
                packed_flat,
                bytes_per_snp,
                n_samples,
                st,
                ed,
                sample_idx,
                full_sample_fast,
                row_flip,
                row_mean,
                row_inv_sd,
                code4_lut,
                &mut m_block[..br * n],
                n,
                pool,
                &mut scratch_decode_f32,
            )?;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                if d[j] == 1 {
                    daxpy_inplace_f64(b_old, m_row, &mut r);
                }

                let xe = ddot_f64(&r, m_row);
                let c = x2[j] * inv_var_e + 1.0 / var_b;
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesCpi packed beta update"
                            .to_string(),
                    );
                }
                let rhs = xe * inv_var_e;
                let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b * c).ln();
                let log_odds = log_odds_prior + log_bf10;
                let p_in = if log_odds >= 0.0 {
                    1.0 / (1.0 + (-log_odds).exp())
                } else {
                    let e = log_odds.exp();
                    e / (1.0 + e)
                };
                let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                d[j] = new_d;

                if new_d == 1 {
                    let z_beta: f64 = rng.sample(StandardNormal);
                    let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;
                    daxpy_inplace_f64(-new_beta, m_row, &mut r);
                    beta[j] = new_beta;
                } else {
                    beta[j] = 0.0;
                }
            }
        }

        let mut mrk_in_usize = 0usize;
        let mut ss_b = 0.0;
        for j in 0..p {
            if d[j] == 1 {
                ss_b += beta[j] * beta[j];
                mrk_in_usize += 1;
            }
        }
        ss_b += s0_b;
        let chi_b_eff = if let Some(dist) = chi_b_cache.get(&mrk_in_usize) {
            *dist
        } else {
            let dist = ChiSquared::new(df0_b + mrk_in_usize as f64).map_err(|e| e.to_string())?;
            chi_b_cache.insert(mrk_in_usize, dist);
            dist
        };
        var_b = ss_b / rng.sample(chi_b_eff);
        if !(var_b.is_finite() && var_b > 0.0) {
            return Err("BayesCpi packed var_b became non-finite or non-positive".to_string());
        }

        let mrk_in = mrk_in_usize as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

        let ss_e = ddot_f64(&r, &r) + s0_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesCpi packed var_e became non-finite or non-positive".to_string());
        }

        let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
        let h2 = var_g / (var_g + var_e);
        full_h2_trace.push(h2);
        full_var_e_trace.push(var_e);
        full_prob_in_trace.push(prob_in);
        full_n_active_trace.push(mrk_in);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += (d[j] as f64) * beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            h2_trace.push(h2);
            var_e_trace.push(var_e);
            prob_in_trace.push(prob_in);
            n_active_trace.push(mrk_in);
            fill_beta_trace_row(
                &mut beta_trace,
                keep_row,
                n_trace_snps,
                trace_snp_indices,
                &beta,
                Some(&d),
            );
            keep_row += 1;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for bj in &mut beta_sum {
        *bj *= inv_keep;
    }
    for ak in &mut alpha_sum {
        *ak *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok(PackedBayesTraceResult {
        beta: beta_sum,
        alpha: alpha_sum,
        vare: var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        iter_trace: keep_iters,
        h2_trace,
        var_e_trace,
        prob_in_trace,
        n_active_trace,
        full_iter_trace,
        full_h2_trace,
        full_var_e_trace,
        full_prob_in_trace,
        full_n_active_trace,
        beta_trace_indices: trace_snp_indices.iter().map(|&v| v as i64).collect(),
        beta_trace,
    })
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    s0_e = None,
    min_abs_beta = 1e-9,
    trace_snp_indices = None,
    threads = 0,
    seed = None
))]
pub fn bayesa_packed_trace<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    s0_e: Option<f64>,
    min_abs_beta: f64,
    trace_snp_indices: Option<PyReadonlyArray1<i64>>,
    threads: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let trace_idx =
        parse_optional_index_vec_i64(trace_snp_indices.as_ref(), p, "trace_snp_indices")?;

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        bayesa_packed_trace_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            s0_e,
            seed,
            &trace_idx,
            pool_ref,
        )
    });
    match result {
        Ok(res) => packed_trace_result_to_pydict(py, res),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    trace_snp_indices = None,
    threads = 0,
    seed = None
))]
pub fn bayesb_packed_trace<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    trace_snp_indices: Option<PyReadonlyArray1<i64>>,
    threads: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let trace_idx =
        parse_optional_index_vec_i64(trace_snp_indices.as_ref(), p, "trace_snp_indices")?;

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        bayesb_packed_trace_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
            &trace_idx,
            pool_ref,
        )
    });
    match result {
        Ok(res) => packed_trace_result_to_pydict(py, res),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    trace_snp_indices = None,
    threads = 0,
    seed = None
))]
pub fn bayescpi_packed_trace<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    trace_snp_indices: Option<PyReadonlyArray1<i64>>,
    threads: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = s0_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_e must be > 0"));
        }
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let trace_idx =
        parse_optional_index_vec_i64(trace_snp_indices.as_ref(), p, "trace_snp_indices")?;

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        bayescpi_packed_trace_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
            &trace_idx,
            pool_ref,
        )
    });
    match result {
        Ok(res) => packed_trace_result_to_pydict(py, res),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}
