use numpy::{Element, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::f64::consts::PI;

use crate::brent::brent_minimize;
use crate::linalg::{chi2_sf_df1, cholesky_inplace, cholesky_logdet, cholesky_solve_into, normal_sf};

fn array1_to_cow<'py, T: Copy + Element>(
    arr: &'py PyReadonlyArray1<'py, T>,
) -> PyResult<Cow<'py, [T]>> {
    if let Ok(slice) = arr.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    Ok(Cow::Owned(arr.as_array().iter().cloned().collect()))
}

fn array2_to_cow<'py, T: Copy + Element>(
    arr: &'py PyReadonlyArray2<'py, T>,
) -> PyResult<(Cow<'py, [T]>, usize, usize)> {
    let view = arr.as_array();
    let (rows, cols) = (view.shape()[0], view.shape()[1]);
    if let Ok(slice) = arr.as_slice() {
        return Ok((Cow::Borrowed(slice), rows, cols));
    }
    let vec: Vec<T> = view.iter().cloned().collect();
    Ok((Cow::Owned(vec), rows, cols))
}

struct FastLmmData<'a> {
    s: &'a [f64],
    u1tx: &'a [f64],
    u2tx: &'a [f64],
    u1ty: &'a [f64],
    u2ty: &'a [f64],
    k: usize,
    n: usize,
    p: usize,
    u2_xtx: &'a [f64],
    u2_xty: &'a [f64],
}

struct FastLmmScratch {
    xtv_inv_x: Vec<f64>,
    xtv_inv_y: Vec<f64>,
    beta: Vec<f64>,
    v1_inv: Vec<f64>,
    rhs: Vec<f64>,
    work: Vec<f64>,
}

impl FastLmmScratch {
    fn new(k: usize, dim: usize) -> Self {
        Self {
            xtv_inv_x: vec![0.0; dim * dim],
            xtv_inv_y: vec![0.0; dim],
            beta: vec![0.0; dim],
            v1_inv: vec![0.0; k],
            rhs: vec![0.0; dim],
            work: vec![0.0; dim],
        }
    }
}

struct ThreadScratch {
    core: FastLmmScratch,
    u2_xtsnp: Vec<f64>,
}

struct SnpPrecomp<'a> {
    u1: &'a [f32],
    u2: &'a [f32],
    u2_xtsnp: &'a [f64],
    u2_snp_snp: f64,
    u2_snp_ty: f64,
}

fn precompute_u2_base(u2tx: &[f64], u2ty: &[f64], n: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut u2_xtx = vec![0.0_f64; p * p];
    let mut u2_xty = vec![0.0_f64; p];

    for i in 0..n {
        let base = i * p;
        let yi = u2ty[i];
        for r in 0..p {
            let xir = u2tx[base + r];
            u2_xty[r] += xir * yi;
            for c in 0..=r {
                u2_xtx[r * p + c] += xir * u2tx[base + c];
            }
        }
    }

    for r in 0..p {
        for c in 0..r {
            let vrc = u2_xtx[r * p + c];
            u2_xtx[c * p + r] = vrc;
        }
    }

    (u2_xtx, u2_xty)
}

fn precompute_u2_snp(
    u2tx: &[f64],
    u2ty: &[f64],
    u2snp: &[f32],
    n: usize,
    p: usize,
    out_u2_xtsnp: &mut [f64],
) -> (f64, f64) {
    out_u2_xtsnp.fill(0.0);
    let mut u2_snp_snp = 0.0_f64;
    let mut u2_snp_ty = 0.0_f64;

    for i in 0..n {
        let gi = u2snp[i] as f64;
        u2_snp_snp += gi * gi;
        u2_snp_ty += gi * u2ty[i];
        let base = i * p;
        for r in 0..p {
            out_u2_xtsnp[r] += u2tx[base + r] * gi;
        }
    }

    (u2_snp_snp, u2_snp_ty)
}

fn fill_xtv(
    log10_lbd: f64,
    data: &FastLmmData,
    snp: Option<&SnpPrecomp>,
    scratch: &mut FastLmmScratch,
) -> Option<(f64, f64)> {
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return None;
    }
    let v2_inv = 1.0 / lbd;

    let dim = data.p + if snp.is_some() { 1 } else { 0 };
    scratch.xtv_inv_x[..dim * dim].fill(0.0);
    scratch.xtv_inv_y[..dim].fill(0.0);

    let mut log_det_v = 0.0_f64;
    for i in 0..data.k {
        let v1 = data.s[i] + lbd;
        if v1 <= 0.0 {
            return None;
        }
        scratch.v1_inv[i] = 1.0 / v1;
        log_det_v += v1.ln();
    }
    log_det_v += ((data.n - data.k) as f64) * lbd.ln();

    for r in 0..data.p {
        let mut sum_y = 0.0_f64;
        for i in 0..data.k {
            sum_y += scratch.v1_inv[i] * data.u1tx[i * data.p + r] * data.u1ty[i];
        }
        sum_y += v2_inv * data.u2_xty[r];
        scratch.xtv_inv_y[r] = sum_y;

        for c in 0..=r {
            let mut sum = 0.0_f64;
            for i in 0..data.k {
                sum += scratch.v1_inv[i]
                    * data.u1tx[i * data.p + r]
                    * data.u1tx[i * data.p + c];
            }
            sum += v2_inv * data.u2_xtx[r * data.p + c];
            scratch.xtv_inv_x[r * dim + c] = sum;
        }
    }

    if let Some(snp) = snp {
        let p = data.p;
        let mut sum_y = 0.0_f64;
        for i in 0..data.k {
            sum_y += scratch.v1_inv[i] * (snp.u1[i] as f64) * data.u1ty[i];
        }
        sum_y += v2_inv * snp.u2_snp_ty;
        scratch.xtv_inv_y[p] = sum_y;

        for r in 0..data.p {
            let mut sum = 0.0_f64;
            for i in 0..data.k {
                sum += scratch.v1_inv[i]
                    * data.u1tx[i * data.p + r]
                    * (snp.u1[i] as f64);
            }
            sum += v2_inv * snp.u2_xtsnp[r];
            scratch.xtv_inv_x[p * dim + r] = sum;
            scratch.xtv_inv_x[r * dim + p] = sum;
        }

        let mut sum = 0.0_f64;
        for i in 0..data.k {
            let gi = snp.u1[i] as f64;
            sum += scratch.v1_inv[i] * gi * gi;
        }
        sum += v2_inv * snp.u2_snp_snp;
        scratch.xtv_inv_x[p * dim + p] = sum;
    }

    let ridge = 1e-6;
    for r in 0..dim {
        scratch.xtv_inv_x[r * dim + r] += ridge;
        for c in 0..r {
            let vrc = scratch.xtv_inv_x[r * dim + c];
            scratch.xtv_inv_x[c * dim + r] = vrc;
        }
    }

    Some((log_det_v, v2_inv))
}

fn fast_reml_cost(
    log10_lbd: f64,
    data: &FastLmmData,
    snp: Option<&SnpPrecomp>,
    scratch: &mut FastLmmScratch,
    n_minus_p: f64,
    c_const: f64,
) -> f64 {
    let dim = data.p + if snp.is_some() { 1 } else { 0 };
    if n_minus_p <= 0.0 {
        return 1e100;
    }

    let (log_det_v, v2_inv) = match fill_xtv(log10_lbd, data, snp, scratch) {
        Some(v) => v,
        None => return 1e100,
    };

    if cholesky_inplace(&mut scratch.xtv_inv_x[..dim * dim], dim).is_none() {
        return 1e100;
    }

    let log_det_xtv = cholesky_logdet(&scratch.xtv_inv_x[..dim * dim], dim);
    cholesky_solve_into(
        &scratch.xtv_inv_x[..dim * dim],
        dim,
        &scratch.xtv_inv_y[..dim],
        &mut scratch.beta[..dim],
    );

    let mut r1_sum = 0.0_f64;
    for i in 0..data.k {
        let mut xb = 0.0_f64;
        let base = i * data.p;
        for r in 0..data.p {
            xb += data.u1tx[base + r] * scratch.beta[r];
        }
        if let Some(snp) = snp {
            xb += (snp.u1[i] as f64) * scratch.beta[data.p];
        }
        let ri = data.u1ty[i] - xb;
        r1_sum += scratch.v1_inv[i] * ri * ri;
    }

    let mut r2_sum = 0.0_f64;
    for i in 0..data.n {
        let mut xb = 0.0_f64;
        let base = i * data.p;
        for r in 0..data.p {
            xb += data.u2tx[base + r] * scratch.beta[r];
        }
        if let Some(snp) = snp {
            xb += (snp.u2[i] as f64) * scratch.beta[data.p];
        }
        let ri = data.u2ty[i] - xb;
        r2_sum += ri * ri;
    }

    let rtv_invr = r1_sum + v2_inv * r2_sum;
    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return 1e100;
    }

    let total_log = n_minus_p * rtv_invr.ln() + log_det_v + log_det_xtv;
    total_log * 0.5 - c_const
}

fn fast_ml_loglike(
    log10_lbd: f64,
    data: &FastLmmData,
    snp: Option<&SnpPrecomp>,
    scratch: &mut FastLmmScratch,
    n_f: f64,
    c_const: f64,
) -> f64 {
    let dim = data.p + if snp.is_some() { 1 } else { 0 };

    let (log_det_v, v2_inv) = match fill_xtv(log10_lbd, data, snp, scratch) {
        Some(v) => v,
        None => return f64::NAN,
    };

    if cholesky_inplace(&mut scratch.xtv_inv_x[..dim * dim], dim).is_none() {
        return f64::NAN;
    }

    cholesky_solve_into(
        &scratch.xtv_inv_x[..dim * dim],
        dim,
        &scratch.xtv_inv_y[..dim],
        &mut scratch.beta[..dim],
    );

    let mut r1_sum = 0.0_f64;
    for i in 0..data.k {
        let mut xb = 0.0_f64;
        let base = i * data.p;
        for r in 0..data.p {
            xb += data.u1tx[base + r] * scratch.beta[r];
        }
        if let Some(snp) = snp {
            xb += (snp.u1[i] as f64) * scratch.beta[data.p];
        }
        let ri = data.u1ty[i] - xb;
        r1_sum += scratch.v1_inv[i] * ri * ri;
    }

    let mut r2_sum = 0.0_f64;
    for i in 0..data.n {
        let mut xb = 0.0_f64;
        let base = i * data.p;
        for r in 0..data.p {
            xb += data.u2tx[base + r] * scratch.beta[r];
        }
        if let Some(snp) = snp {
            xb += (snp.u2[i] as f64) * scratch.beta[data.p];
        }
        let ri = data.u2ty[i] - xb;
        r2_sum += ri * ri;
    }

    let rtv_invr = r1_sum + v2_inv * r2_sum;
    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return f64::NAN;
    }

    let total_log = n_f * rtv_invr.ln() + log_det_v;
    let ml = c_const - 0.5 * total_log;

    if ml.is_finite() { ml } else { f64::NAN }
}

fn fast_reml_beta_se(
    log10_lbd: f64,
    data: &FastLmmData,
    snp: &SnpPrecomp,
    scratch: &mut FastLmmScratch,
    n_minus_p: f64,
) -> (f64, f64) {
    let dim = data.p + 1;
    if n_minus_p <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let ( _log_det_v, v2_inv) = match fill_xtv(log10_lbd, data, Some(snp), scratch) {
        Some(v) => v,
        None => return (f64::NAN, f64::NAN),
    };

    if cholesky_inplace(&mut scratch.xtv_inv_x[..dim * dim], dim).is_none() {
        return (f64::NAN, f64::NAN);
    }

    cholesky_solve_into(
        &scratch.xtv_inv_x[..dim * dim],
        dim,
        &scratch.xtv_inv_y[..dim],
        &mut scratch.beta[..dim],
    );

    let mut r1_sum = 0.0_f64;
    for i in 0..data.k {
        let mut xb = 0.0_f64;
        let base = i * data.p;
        for r in 0..data.p {
            xb += data.u1tx[base + r] * scratch.beta[r];
        }
        xb += (snp.u1[i] as f64) * scratch.beta[data.p];
        let ri = data.u1ty[i] - xb;
        r1_sum += scratch.v1_inv[i] * ri * ri;
    }

    let mut r2_sum = 0.0_f64;
    for i in 0..data.n {
        let mut xb = 0.0_f64;
        let base = i * data.p;
        for r in 0..data.p {
            xb += data.u2tx[base + r] * scratch.beta[r];
        }
        xb += (snp.u2[i] as f64) * scratch.beta[data.p];
        let ri = data.u2ty[i] - xb;
        r2_sum += ri * ri;
    }

    let rtv_invr = r1_sum + v2_inv * r2_sum;
    let sigma2 = rtv_invr / n_minus_p;
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return (scratch.beta[data.p], f64::NAN);
    }

    scratch.rhs[..dim].fill(0.0);
    scratch.rhs[data.p] = 1.0;
    cholesky_solve_into(
        &scratch.xtv_inv_x[..dim * dim],
        dim,
        &scratch.rhs[..dim],
        &mut scratch.work[..dim],
    );

    let var_beta = sigma2 * scratch.work[data.p];
    if !var_beta.is_finite() || var_beta <= 0.0 {
        return (scratch.beta[data.p], f64::NAN);
    }

    (scratch.beta[data.p], var_beta.sqrt())
}

#[pyfunction]
#[pyo3(signature = (s, u1tx, u2tx, u1ty, u2ty, low, high, max_iter=50, tol=1e-2))]
pub fn fastlmm_reml_null_f32<'py>(
    _py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    u1tx: PyReadonlyArray2<'py, f64>,
    u2tx: PyReadonlyArray2<'py, f64>,
    u1ty: PyReadonlyArray1<'py, f64>,
    u2ty: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<(f64, f64, f64)> {
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let s_cow = array1_to_cow(&s)?;
    let u1ty_cow = array1_to_cow(&u1ty)?;
    let u2ty_cow = array1_to_cow(&u2ty)?;

    let (u1tx_cow, k1, p1) = array2_to_cow(&u1tx)?;
    let (u2tx_cow, n2, p2) = array2_to_cow(&u2tx)?;

    let k = s_cow.len();
    if k == 0 {
        return Err(PyRuntimeError::new_err("empty s"));
    }
    if k1 != k {
        return Err(PyRuntimeError::new_err("u1tx rows must equal len(s)"));
    }
    if u1ty_cow.len() != k {
        return Err(PyRuntimeError::new_err("u1ty len must equal len(s)"));
    }
    if p1 == 0 {
        return Err(PyRuntimeError::new_err("u1tx must have at least 1 column"));
    }
    if p1 != p2 {
        return Err(PyRuntimeError::new_err("u1tx and u2tx must have same column count"));
    }

    let n = u2ty_cow.len();
    if n2 != n {
        return Err(PyRuntimeError::new_err("u2tx rows must equal len(u2ty)"));
    }

    let n_minus_p = (n as isize) - (p1 as isize);
    if n_minus_p <= 0 {
        return Err(PyRuntimeError::new_err("n must be > p"));
    }

    let (u2_xtx, u2_xty) = precompute_u2_base(u2tx_cow.as_ref(), u2ty_cow.as_ref(), n, p1);

    let data = FastLmmData {
        s: s_cow.as_ref(),
        u1tx: u1tx_cow.as_ref(),
        u2tx: u2tx_cow.as_ref(),
        u1ty: u1ty_cow.as_ref(),
        u2ty: u2ty_cow.as_ref(),
        k,
        n,
        p: p1,
        u2_xtx: &u2_xtx,
        u2_xty: &u2_xty,
    };

    let n_minus_p_f = n_minus_p as f64;
    let c_const = n_minus_p_f * (n_minus_p_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    let mut scratch = FastLmmScratch::new(k, p1);
    let (best_log10, best_cost) = brent_minimize(
        |log10_lbd| fast_reml_cost(log10_lbd, &data, None, &mut scratch, n_minus_p_f, c_const),
        low,
        high,
        tol,
        max_iter,
    );

    let lbd = 10.0_f64.powf(best_log10);
    let reml = -best_cost;
    let ml = fast_ml_loglike(best_log10, &data, None, &mut scratch, n_f, c_ml);
    Ok((lbd, ml, reml))
}

#[pyfunction]
#[pyo3(signature = (s, u1tx, u2tx, u1ty, u2ty, u1tsnp_chunk, u2tsnp_chunk, low, high, max_iter=50, tol=1e-2, threads=0, nullml=None))]
pub fn fastlmm_reml_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    u1tx: PyReadonlyArray2<'py, f64>,
    u2tx: PyReadonlyArray2<'py, f64>,
    u1ty: PyReadonlyArray1<'py, f64>,
    u2ty: PyReadonlyArray1<'py, f64>,
    u1tsnp_chunk: PyReadonlyArray2<'py, f32>,
    u2tsnp_chunk: PyReadonlyArray2<'py, f32>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let s_cow = array1_to_cow(&s)?;
    let u1ty_cow = array1_to_cow(&u1ty)?;
    let u2ty_cow = array1_to_cow(&u2ty)?;

    let (u1tx_cow, k1, p1) = array2_to_cow(&u1tx)?;
    let (u2tx_cow, n2, p2) = array2_to_cow(&u2tx)?;
    let (u1tsnp_cow, m1, k2) = array2_to_cow(&u1tsnp_chunk)?;
    let (u2tsnp_cow, m2, n3) = array2_to_cow(&u2tsnp_chunk)?;

    let k = s_cow.len();
    if k == 0 {
        return Err(PyRuntimeError::new_err("empty s"));
    }
    if k1 != k {
        return Err(PyRuntimeError::new_err("u1tx rows must equal len(s)"));
    }
    if u1ty_cow.len() != k {
        return Err(PyRuntimeError::new_err("u1ty len must equal len(s)"));
    }
    if p1 == 0 {
        return Err(PyRuntimeError::new_err("u1tx must have at least 1 column"));
    }
    if p1 != p2 {
        return Err(PyRuntimeError::new_err("u1tx and u2tx must have same column count"));
    }

    let n = u2ty_cow.len();
    if n2 != n {
        return Err(PyRuntimeError::new_err("u2tx rows must equal len(u2ty)"));
    }

    if k2 != k {
        return Err(PyRuntimeError::new_err("u1tsnp_chunk must have k columns (m, k)"));
    }
    if n3 != n {
        return Err(PyRuntimeError::new_err("u2tsnp_chunk must have n columns (m, n)"));
    }
    if m1 != m2 {
        return Err(PyRuntimeError::new_err("u1tsnp_chunk and u2tsnp_chunk must have same row count"));
    }

    let n_minus_p = (n as isize) - (p1 as isize) - 1;
    if n_minus_p <= 0 {
        return Err(PyRuntimeError::new_err("n must be > p+1"));
    }

    let m = m1;
    let (u2_xtx, u2_xty) = precompute_u2_base(u2tx_cow.as_ref(), u2ty_cow.as_ref(), n, p1);

    let data = FastLmmData {
        s: s_cow.as_ref(),
        u1tx: u1tx_cow.as_ref(),
        u2tx: u2tx_cow.as_ref(),
        u1ty: u1ty_cow.as_ref(),
        u2ty: u2ty_cow.as_ref(),
        k,
        n,
        p: p1,
        u2_xtx: &u2_xtx,
        u2_xty: &u2_xty,
    };

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let beta_se_p = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();

    let beta_se_p_slice: &mut [f64] = unsafe {
        beta_se_p
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("beta_se_p not contiguous"))?
    };
    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    let n_minus_p_f = n_minus_p as f64;
    let c_const = n_minus_p_f * (n_minus_p_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    py.detach(|| {
        let compute_all = || {
            (0..m)
                .into_par_iter()
                .map_init(
                    || ThreadScratch {
                        core: FastLmmScratch::new(k, p1 + 1),
                        u2_xtsnp: vec![0.0; p1],
                    },
                    |scratch, idx| {
                        let u1_row = &u1tsnp_cow.as_ref()[idx * k..(idx + 1) * k];
                        let u2_row = &u2tsnp_cow.as_ref()[idx * n..(idx + 1) * n];

                        let (u2_snp_snp, u2_snp_ty) = precompute_u2_snp(
                            data.u2tx,
                            data.u2ty,
                            u2_row,
                            n,
                            data.p,
                            &mut scratch.u2_xtsnp,
                        );

                        let snp = SnpPrecomp {
                            u1: u1_row,
                            u2: u2_row,
                            u2_xtsnp: &scratch.u2_xtsnp,
                            u2_snp_snp,
                            u2_snp_ty,
                        };

                        let (best_log10, _best_cost) = brent_minimize(
                            |log10_lbd| {
                                fast_reml_cost(
                                    log10_lbd,
                                    &data,
                                    Some(&snp),
                                    &mut scratch.core,
                                    n_minus_p_f,
                                    c_const,
                                )
                            },
                            low,
                            high,
                            tol,
                            max_iter,
                        );

                        let (beta, se) = fast_reml_beta_se(
                            best_log10,
                            &data,
                            &snp,
                            &mut scratch.core,
                            n_minus_p_f,
                        );
                        let pval = if beta.is_finite() && se.is_finite() && se > 0.0 {
                            let z = beta / se;
                            (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        let plrt = if with_plrt {
                            let ml = fast_ml_loglike(
                                best_log10,
                                &data,
                                Some(&snp),
                                &mut scratch.core,
                                n_f,
                                c_ml,
                            );
                            if ml.is_finite() {
                                let mut stat = 2.0 * (ml - nullml_val);
                                if !stat.is_finite() || stat < 0.0 {
                                    stat = 0.0;
                                }
                                chi2_sf_df1(stat)
                            } else {
                                1.0
                            }
                        } else {
                            0.0
                        };

                        (beta, se, if pval.is_finite() { pval } else { 1.0 }, plrt)
                    },
                )
                .collect::<Vec<(f64, f64, f64, f64)>>()
        };

        let results = if let Some(pool) = &pool {
            pool.install(compute_all)
        } else {
            compute_all()
        };

        for (idx, (beta, se, pval, plrt)) in results.into_iter().enumerate() {
            let out_row = &mut beta_se_p_slice[idx * out_cols..(idx + 1) * out_cols];
            out_row[0] = beta;
            out_row[1] = se;
            out_row[2] = pval;
            if with_plrt {
                out_row[3] = plrt;
            }
        }
    });

    Ok(beta_se_p)
}
