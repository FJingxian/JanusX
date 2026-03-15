use nalgebra::{DMatrix, DVector};
use numpy::PyArray1;
use numpy::PyReadonlyArray3;
use numpy::PyUntypedArrayMethods;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

use crate::brent::brent_minimize;
use crate::linalg::{
    chi2_sf_df1, cholesky_inplace, cholesky_logdet, cholesky_solve_into, normal_sf,
};

// =============================================================================
// Common utilities
// =============================================================================

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

thread_local! {
    static LOCAL_RAYON_POOLS: RefCell<HashMap<usize, Arc<rayon::ThreadPool>>> =
        RefCell::new(HashMap::new());
}

#[inline]
fn get_cached_pool(threads: usize) -> PyResult<Option<Arc<rayon::ThreadPool>>> {
    if threads == 0 {
        return Ok(None);
    }
    LOCAL_RAYON_POOLS.with(|cell| {
        let mut pools = cell.borrow_mut();
        if let Some(tp) = pools.get(&threads) {
            return Ok(Some(Arc::clone(tp)));
        }
        let tp = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        );
        pools.insert(threads, Arc::clone(&tp));
        Ok(Some(tp))
    })
}

/// 用 cholesky(L) 解 A x = b，a 中存的是 L（下三角）
fn cholesky_solve(a: &[f64], dim: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; dim];
    for i in 0..dim {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * dim + k] * y[k];
        }
        y[i] = sum / a[i * dim + i];
    }

    let mut x = vec![0.0_f64; dim];
    for ii in 0..dim {
        let i = dim - 1 - ii;
        let mut sum = y[i];
        for k in (i + 1)..dim {
            sum -= a[k * dim + i] * x[k];
        }
        x[i] = sum / a[i * dim + i];
    }
    x
}

// =============================================================================
// Student-t p-value (for GLM)
// =============================================================================

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let maxit = 200;
    let eps = 3.0e-14;
    let fpmin = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=maxit {
        let m2 = 2.0 * (m as f64);

        let mut aa = (m as f64) * (b - (m as f64)) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + (m as f64)) * (qab + (m as f64)) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

fn betai(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    let ln_beta = libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);

    if x < (a + 1.0) / (a + b + 2.0) {
        let front = ((a * x.ln()) + (b * (1.0 - x).ln()) - ln_beta).exp() / a;
        front * betacf(a, b, x)
    } else {
        let front = ((b * (1.0 - x).ln()) + (a * x.ln()) - ln_beta).exp() / b;
        1.0 - front * betacf(b, a, 1.0 - x)
    }
}

#[inline]
fn student_t_p_two_sided(t: f64, df: i32) -> f64 {
    if df <= 0 {
        return f64::NAN;
    }
    if !t.is_finite() {
        return if t.is_nan() {
            f64::NAN
        } else {
            f64::MIN_POSITIVE
        };
    }

    let v = df as f64;
    let x = v / (v + t * t);
    let a = v / 2.0;
    let b = 0.5;

    let mut p = betai(a, b, x);
    if !p.is_finite() {
        p = 1.0;
    }
    p.clamp(f64::MIN_POSITIVE, 1.0)
}

// =============================================================================
// GLM float32 fast path (glmf32) with thread-local scratch
// =============================================================================

#[inline]
fn xs_t_ixx_into(xs: &[f64], ixx: &[f64], q0: usize, out_b21: &mut [f64]) {
    debug_assert_eq!(out_b21.len(), q0);
    for j in 0..q0 {
        let mut acc = 0.0;
        for k in 0..q0 {
            acc += xs[k] * ixx[k * q0 + j];
        }
        out_b21[j] = acc;
    }
}

#[inline]
fn build_ixxs_into(ixx: &[f64], b21: &[f64], invb22: f64, q0: usize, out_ixxs: &mut [f64]) {
    let dim = q0 + 1;
    debug_assert_eq!(out_ixxs.len(), dim * dim);

    for r in 0..q0 {
        for c in 0..q0 {
            out_ixxs[r * dim + c] = ixx[r * q0 + c] + invb22 * (b21[r] * b21[c]);
        }
    }

    out_ixxs[q0 * dim + q0] = invb22;

    for j in 0..q0 {
        let v = -invb22 * b21[j];
        out_ixxs[q0 * dim + j] = v;
        out_ixxs[j * dim + q0] = v;
    }
}

#[inline]
fn matvec_into(a: &[f64], dim: usize, rhs: &[f64], out: &mut [f64]) {
    debug_assert_eq!(rhs.len(), dim);
    debug_assert_eq!(out.len(), dim);
    for r in 0..dim {
        let row = &a[r * dim..(r + 1) * dim];
        out[r] = row.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum();
    }
}

struct GlmScratch {
    xs: Vec<f64>,   // q0
    b21: Vec<f64>,  // q0
    rhs: Vec<f64>,  // q0+1
    beta: Vec<f64>, // q0+1
    ixxs: Vec<f64>, // (q0+1)^2
}

impl GlmScratch {
    fn new(q0: usize) -> Self {
        let dim = q0 + 1;
        Self {
            xs: vec![0.0; q0],
            b21: vec![0.0; q0],
            rhs: vec![0.0; dim],
            beta: vec![0.0; dim],
            ixxs: vec![0.0; dim * dim],
        }
    }
    #[inline]
    fn reset_xs(&mut self) {
        self.xs.fill(0.0);
    }
}

/// Fast GLM interface:
/// y: (n,) float64
/// X: (n, q0) float64
/// ixx: (q0, q0) float64
/// G: (m, n) float32 (marker rows)
///
/// Returns: (m, q0 + 3) float64
///   col0: beta_snp
///   col1: se_snp
///   col2..: p-values for coefficients (q0 covariates + snp)
#[pyfunction]
#[pyo3(signature = (y, x, ixx, g, step=10000, threads=0))]
pub fn glmf32<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray2<'py, f32>,
    step: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let ixx_arr = ixx.as_array();
    let g_arr = g.as_array();

    let n = y.len();
    let (xn, q0) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
        return Err(PyRuntimeError::new_err("ixx must be (q0,q0)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "G must be shape (m, n) for float32 fast path",
        ));
    }
    if n <= q0 + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n too small: require n > q0+1, got n={n}, q0={q0}"
        )));
    }
    let g_slice_opt = g.as_slice().ok();

    let m = g_arr.shape()[0];
    let row_stride = q0 + 3;
    let dim = q0 + 1;

    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let ixx_flat: Cow<[f64]> = match ixx.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ixx_arr.iter().copied().collect()),
    };

    let mut xy = vec![0.0; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();

    let out = PyArray2::<f64>::zeros(py, [m, row_stride], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    let pool = get_cached_pool(threads)?;

    py.detach(|| {
        let mut runner = || {
            let mut i_marker = 0usize;
            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);
                let block = &mut out_slice[i_marker * row_stride..(i_marker + cnt) * row_stride];

                block.par_chunks_mut(row_stride).enumerate().for_each_init(
                    || GlmScratch::new(q0),
                    |scr, (l, row_out)| {
                        let idx = i_marker + l;
                        scr.reset_xs();

                        let mut sy = 0.0_f64;
                        let mut ss = 0.0_f64;

                        if let Some(gs) = g_slice_opt {
                            let grow = &gs[idx * n..(idx + 1) * n];
                            for k in 0..n {
                                let gv = grow[k] as f64;
                                sy += gv * y[k];
                                ss += gv * gv;

                                let row = &x_flat[k * q0..(k + 1) * q0];
                                for j in 0..q0 {
                                    scr.xs[j] += row[j] * gv;
                                }
                            }
                        } else {
                            for k in 0..n {
                                let gv = g_arr[(idx, k)] as f64; // float32 -> f64
                                sy += gv * y[k];
                                ss += gv * gv;

                                let row = &x_flat[k * q0..(k + 1) * q0];
                                for j in 0..q0 {
                                    scr.xs[j] += row[j] * gv;
                                }
                            }
                        }

                        xs_t_ixx_into(&scr.xs, &ixx_flat, q0, &mut scr.b21);
                        let t2 = dot(&scr.b21, &scr.xs);
                        let b22 = ss - t2;

                        let (invb22, df) = if b22 < 1e-8 {
                            (0.0, (n as i32) - (q0 as i32))
                        } else {
                            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                        };
                        if df <= 0 {
                            row_out.fill(f64::NAN);
                            return;
                        }

                        build_ixxs_into(&ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);

                        scr.rhs[..q0].copy_from_slice(&xy);
                        scr.rhs[q0] = sy;

                        matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);

                        let beta_rhs = dot(&scr.beta, &scr.rhs);
                        let ve = (yy - beta_rhs) / (df as f64);

                        for ff in 0..dim {
                            let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
                            let t = scr.beta[ff] / se;
                            row_out[2 + ff] = student_t_p_two_sided(t, df);
                        }

                        if invb22 == 0.0 {
                            row_out[0] = f64::NAN;
                            row_out[1] = f64::NAN;
                            row_out[2 + q0] = f64::NAN;
                        } else {
                            let beta_snp = scr.beta[q0];
                            let se_snp = (scr.ixxs[q0 * dim + q0] * ve).sqrt();
                            row_out[0] = beta_snp;
                            row_out[1] = se_snp;
                        }
                    },
                );

                i_marker += cnt;
            }
        };

        if let Some(p) = &pool {
            p.install(runner);
        } else {
            runner();
        }
    });

    Ok(out)
}

/// Full GLM interface (returns beta/se/p for all coefficients).
///
/// Returns: (m, 3 * (q0 + 1)) float64
///   For each coefficient j (0..q0 covariates, q0 is SNP):
///     col[3*j+0] = beta_j
///     col[3*j+1] = se_j
///     col[3*j+2] = p_j
#[pyfunction]
#[pyo3(signature = (y, x, ixx, g, step=10000, threads=0))]
pub fn glmf32_full<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray2<'py, f32>,
    step: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let ixx_arr = ixx.as_array();
    let g_arr = g.as_array();

    let n = y.len();
    let (xn, q0) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
        return Err(PyRuntimeError::new_err("ixx must be (q0,q0)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "G must be shape (m, n) for float32 fast path",
        ));
    }
    if n <= q0 + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n too small: require n > q0+1, got n={n}, q0={q0}"
        )));
    }
    let g_slice_opt = g.as_slice().ok();

    let m = g_arr.shape()[0];
    let dim = q0 + 1;
    let row_stride = dim * 3;

    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let ixx_flat: Cow<[f64]> = match ixx.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ixx_arr.iter().copied().collect()),
    };

    let mut xy = vec![0.0; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();

    let out = PyArray2::<f64>::zeros(py, [m, row_stride], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    let pool = get_cached_pool(threads)?;

    py.detach(|| {
        let mut runner = || {
            let mut i_marker = 0usize;
            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);
                let block = &mut out_slice[i_marker * row_stride..(i_marker + cnt) * row_stride];

                block.par_chunks_mut(row_stride).enumerate().for_each_init(
                    || GlmScratch::new(q0),
                    |scr, (l, row_out)| {
                        let idx = i_marker + l;
                        scr.reset_xs();

                        let mut sy = 0.0_f64;
                        let mut ss = 0.0_f64;

                        if let Some(gs) = g_slice_opt {
                            let grow = &gs[idx * n..(idx + 1) * n];
                            for k in 0..n {
                                let gv = grow[k] as f64;
                                sy += gv * y[k];
                                ss += gv * gv;

                                let row = &x_flat[k * q0..(k + 1) * q0];
                                for j in 0..q0 {
                                    scr.xs[j] += row[j] * gv;
                                }
                            }
                        } else {
                            for k in 0..n {
                                let gv = g_arr[(idx, k)] as f64;
                                sy += gv * y[k];
                                ss += gv * gv;

                                let row = &x_flat[k * q0..(k + 1) * q0];
                                for j in 0..q0 {
                                    scr.xs[j] += row[j] * gv;
                                }
                            }
                        }

                        xs_t_ixx_into(&scr.xs, &ixx_flat, q0, &mut scr.b21);
                        let t2 = dot(&scr.b21, &scr.xs);
                        let b22 = ss - t2;

                        let (invb22, df) = if b22 < 1e-8 {
                            (0.0, (n as i32) - (q0 as i32))
                        } else {
                            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                        };
                        if df <= 0 {
                            row_out.fill(f64::NAN);
                            return;
                        }

                        build_ixxs_into(&ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);

                        scr.rhs[..q0].copy_from_slice(&xy);
                        scr.rhs[q0] = sy;

                        matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);

                        let beta_rhs = dot(&scr.beta, &scr.rhs);
                        let ve = (yy - beta_rhs) / (df as f64);

                        for ff in 0..dim {
                            let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
                            let t = scr.beta[ff] / se;
                            let base = 3 * ff;
                            row_out[base] = scr.beta[ff];
                            row_out[base + 1] = se;
                            row_out[base + 2] = student_t_p_two_sided(t, df);
                        }

                        if invb22 == 0.0 {
                            let base = 3 * q0;
                            row_out[base] = f64::NAN;
                            row_out[base + 1] = f64::NAN;
                            row_out[base + 2] = f64::NAN;
                        }
                    },
                );

                i_marker += cnt;
            }
        };

        if let Some(p) = &pool {
            p.install(runner);
        } else {
            runner();
        }
    });

    Ok(out)
}

// =============================================================================
// LMM REML chunk (lmm_reml_chunk_f32)
// =============================================================================

fn reml_loglike(
    log10_lbd: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: Option<&[f64]>,
    n: usize,
    p_cov: usize,
) -> f64 {
    let use_snp = snp.is_some();
    let snp = snp.unwrap_or(&[]);
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return -1e8;
    }

    if use_snp && snp.len() != n {
        return -1e8;
    }
    let p = p_cov + if use_snp { 1 } else { 0 };
    if n <= p {
        return -1e8;
    }

    let mut v = vec![0.0_f64; n];
    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return -1e8;
        }
        v[i] = vv;
        vinv[i] = 1.0 / vv;
    }

    let dim = p;
    let mut xtv_inv_x = vec![0.0_f64; dim * dim];
    let mut xtv_inv_y = vec![0.0_f64; dim];

    for i in 0..n {
        let vi = vinv[i];
        let yi = y[i];
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xtv_inv_y[r] += vi * xir * yi;

            for c in 0..=r {
                let xic = if c < p_cov {
                    xcov[i * p_cov + c]
                } else {
                    snp[i]
                };
                xtv_inv_x[r * dim + c] += vi * xir * xic;
            }
        }
    }

    let ridge = 1e-6;
    for r in 0..dim {
        xtv_inv_x[r * dim + r] += ridge;
        for c in 0..r {
            let vrc = xtv_inv_x[r * dim + c];
            xtv_inv_x[c * dim + r] = vrc;
        }
    }

    if cholesky_inplace(&mut xtv_inv_x, dim).is_none() {
        return -1e8;
    }
    let log_det_xtv_inv_x = cholesky_logdet(&xtv_inv_x, dim);
    let beta = cholesky_solve(&xtv_inv_x, dim, &xtv_inv_y);

    let mut r_vec = vec![0.0_f64; n];
    for i in 0..n {
        let mut xb = 0.0;
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xb += xir * beta[r];
        }
        r_vec[i] = y[i] - xb;
    }

    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        rtv_invr += vinv[i] * r_vec[i] * r_vec[i];
    }

    let log_det_v: f64 = v.iter().map(|vv| vv.ln()).sum();
    let n_f = n as f64;
    let p_f = dim as f64;

    let total_log = (n_f - p_f) * (rtv_invr.ln()) + log_det_v + log_det_xtv_inv_x;
    let c = (n_f - p_f) * ((n_f - p_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let reml = c - 0.5 * total_log;

    if !reml.is_finite() {
        -1e8
    } else {
        reml
    }
}

fn ml_loglike(
    log10_lbd: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: Option<&[f64]>,
    n: usize,
    p_cov: usize,
) -> f64 {
    let use_snp = snp.is_some();
    let snp = snp.unwrap_or(&[]);
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return -1e8;
    }

    if use_snp && snp.len() != n {
        return -1e8;
    }
    let p = p_cov + if use_snp { 1 } else { 0 };
    if n <= p {
        return -1e8;
    }

    let mut v = vec![0.0_f64; n];
    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return -1e8;
        }
        v[i] = vv;
        vinv[i] = 1.0 / vv;
    }

    let dim = p;
    let mut xtv_inv_x = vec![0.0_f64; dim * dim];
    let mut xtv_inv_y = vec![0.0_f64; dim];

    for i in 0..n {
        let vi = vinv[i];
        let yi = y[i];
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xtv_inv_y[r] += vi * xir * yi;

            for c in 0..=r {
                let xic = if c < p_cov {
                    xcov[i * p_cov + c]
                } else {
                    snp[i]
                };
                xtv_inv_x[r * dim + c] += vi * xir * xic;
            }
        }
    }

    let ridge = 1e-6;
    for r in 0..dim {
        xtv_inv_x[r * dim + r] += ridge;
        for c in 0..r {
            let vrc = xtv_inv_x[r * dim + c];
            xtv_inv_x[c * dim + r] = vrc;
        }
    }

    if cholesky_inplace(&mut xtv_inv_x, dim).is_none() {
        return -1e8;
    }
    let beta = cholesky_solve(&xtv_inv_x, dim, &xtv_inv_y);

    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        let mut xb = 0.0;
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xb += xir * beta[r];
        }
        let ri = y[i] - xb;
        rtv_invr += vinv[i] * ri * ri;
    }

    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return -1e8;
    }

    let log_det_v: f64 = v.iter().map(|vv| vv.ln()).sum();
    let n_f = n as f64;

    let total_log = n_f * (rtv_invr.ln()) + log_det_v;
    let c = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let ml = c - 0.5 * total_log;

    if !ml.is_finite() {
        -1e8
    } else {
        ml
    }
}

fn final_beta_se(
    log10_lbd: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: &[f64],
    n: usize,
    p_cov: usize,
) -> (f64, f64, f64) {
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let p = p_cov + 1;
    if n <= p {
        return (f64::NAN, f64::NAN, lbd);
    }

    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return (f64::NAN, f64::NAN, lbd);
        }
        vinv[i] = 1.0 / vv;
    }

    let dim = p;
    let mut xtv_inv_x = vec![0.0_f64; dim * dim];
    let mut xtv_inv_y = vec![0.0_f64; dim];

    for i in 0..n {
        let vi = vinv[i];
        let yi = y[i];
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xtv_inv_y[r] += vi * xir * yi;

            for c in 0..=r {
                let xic = if c < p_cov {
                    xcov[i * p_cov + c]
                } else {
                    snp[i]
                };
                xtv_inv_x[r * dim + c] += vi * xir * xic;
            }
        }
    }

    let ridge = 1e-6;
    for r in 0..dim {
        xtv_inv_x[r * dim + r] += ridge;
        for c in 0..r {
            let vrc = xtv_inv_x[r * dim + c];
            xtv_inv_x[c * dim + r] = vrc;
        }
    }

    if cholesky_inplace(&mut xtv_inv_x, dim).is_none() {
        return (f64::NAN, f64::NAN, lbd);
    }
    let beta = cholesky_solve(&xtv_inv_x, dim, &xtv_inv_y);

    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        let mut xb = 0.0;
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xb += xir * beta[r];
        }
        let ri = y[i] - xb;
        rtv_invr += vinv[i] * ri * ri;
    }

    let n_f = n as f64;
    let p_f = p as f64;
    let sigma2 = rtv_invr / (n_f - p_f);

    let k = dim - 1;
    let mut e = vec![0.0_f64; dim];
    e[k] = 1.0;
    let x = cholesky_solve(&xtv_inv_x, dim, &e);
    let var_beta_k = sigma2 * x[k];
    if var_beta_k <= 0.0 || !var_beta_k.is_finite() {
        return (f64::NAN, f64::NAN, lbd);
    }

    (beta[k], var_beta_k.sqrt(), lbd)
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, max_iter=50, tol=1e-2))]
pub fn lmm_reml_null_f32<'py>(
    _py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<(f64, f64, f64)> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let (best_log10_lbd, best_cost) = brent_minimize(
        |x| -reml_loglike(x, s, &xcov_flat, y, None, n, p_cov),
        low,
        high,
        tol,
        max_iter,
    );
    let reml = -best_cost;
    let lbd = 10.0_f64.powf(best_log10_lbd);
    let ml = ml_loglike(best_log10_lbd, s, &xcov_flat, y, None, n, p_cov);
    Ok((lbd, ml, reml))
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd))]
pub fn ml_loglike_null_f32<'py>(
    _py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
) -> PyResult<f64> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let ml = ml_loglike(log10_lbd, s, &xcov_flat, y, None, n, p_cov);
    Ok(ml)
}

#[inline]
fn apply_p_diag_vec(
    xcov: &[f64],
    n: usize,
    p: usize,
    w: &[f64],
    c_inv: &[f64],
    v: &[f64],
    out: &mut [f64],
    tmp: &mut [f64],
    xtmp: &mut [f64],
    cxtmp: &mut [f64],
) {
    debug_assert_eq!(w.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(tmp.len(), n);
    debug_assert_eq!(xtmp.len(), p);
    debug_assert_eq!(cxtmp.len(), p);

    for i in 0..n {
        tmp[i] = w[i] * v[i];
    }
    for r in 0..p {
        let mut acc = 0.0_f64;
        for i in 0..n {
            acc += xcov[i * p + r] * tmp[i];
        }
        xtmp[r] = acc;
    }
    for r in 0..p {
        let mut acc = 0.0_f64;
        for c in 0..p {
            acc += c_inv[r * p + c] * xtmp[c];
        }
        cxtmp[r] = acc;
    }
    for i in 0..n {
        let mut xcu = 0.0_f64;
        for r in 0..p {
            xcu += xcov[i * p + r] * cxtmp[r];
        }
        out[i] = tmp[i] - w[i] * xcu;
    }
}

#[inline]
fn trace_p_d(
    s: &[f64],
    xcov: &[f64],
    n: usize,
    p: usize,
    w: &[f64],
    c_inv: &[f64],
    use_s_as_d: bool,
) -> f64 {
    let mut tr_wd = 0.0_f64;
    for i in 0..n {
        let d = if use_s_as_d { s[i] } else { 1.0 };
        tr_wd += w[i] * d;
    }

    // M = X^T diag(w^2 * d) X
    let mut m = vec![0.0_f64; p * p];
    for i in 0..n {
        let d = if use_s_as_d { s[i] } else { 1.0 };
        let wi2d = w[i] * w[i] * d;
        let base = i * p;
        for r in 0..p {
            let xir = xcov[base + r];
            for c in 0..=r {
                let xic = xcov[base + c];
                m[r * p + c] += wi2d * xir * xic;
            }
        }
    }
    for r in 0..p {
        for c in 0..r {
            m[c * p + r] = m[r * p + c];
        }
    }

    // tr(C^{-1} M) is not right; we already pass C = (X^T W X)^{-1}.
    // Need tr(C * M).
    let mut tr_cm = 0.0_f64;
    for r in 0..p {
        for c in 0..p {
            tr_cm += c_inv[r * p + c] * m[c * p + r];
        }
    }
    tr_wd - tr_cm
}

fn ai_reml_eval(
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    sigma_g2: f64,
    sigma_e2: f64,
) -> Option<(f64, f64, f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if !sigma_g2.is_finite() || !sigma_e2.is_finite() || sigma_g2 <= 0.0 || sigma_e2 <= 0.0 {
        return None;
    }
    if n <= p {
        return None;
    }

    let mut w = vec![0.0_f64; n];
    let mut log_det_v = 0.0_f64;
    for i in 0..n {
        let vi = sigma_g2 * s[i] + sigma_e2;
        if !vi.is_finite() || vi <= 0.0 {
            return None;
        }
        w[i] = 1.0 / vi;
        log_det_v += vi.ln();
    }

    // A = X^T W X, b = X^T W y
    let mut a = vec![0.0_f64; p * p];
    let mut b = vec![0.0_f64; p];
    for i in 0..n {
        let wi = w[i];
        let yi = y[i];
        let base = i * p;
        for r in 0..p {
            let xir = xcov[base + r];
            b[r] += wi * xir * yi;
            for c in 0..=r {
                let xic = xcov[base + c];
                a[r * p + c] += wi * xir * xic;
            }
        }
    }
    for r in 0..p {
        a[r * p + r] += 1e-8;
        for c in 0..r {
            a[c * p + r] = a[r * p + c];
        }
    }

    let mut l = a.clone();
    cholesky_inplace(&mut l, p)?;
    let log_det_xtv_inv_x = cholesky_logdet(&l, p);

    let beta = cholesky_solve(&l, p, &b);

    // z = V^{-1} r
    let mut z = vec![0.0_f64; n];
    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        let base = i * p;
        let mut xb = 0.0_f64;
        for r in 0..p {
            xb += xcov[base + r] * beta[r];
        }
        let ri = y[i] - xb;
        z[i] = w[i] * ri;
        rtv_invr += ri * z[i];
    }
    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return None;
    }

    // C = (X^T W X)^{-1}
    let mut c_inv = vec![0.0_f64; p * p];
    let mut e = vec![0.0_f64; p];
    let mut x = vec![0.0_f64; p];
    for col in 0..p {
        e.fill(0.0);
        e[col] = 1.0;
        x.fill(0.0);
        cholesky_solve_into(&l, p, &e, &mut x);
        for row in 0..p {
            c_inv[row * p + col] = x[row];
        }
    }

    let n_f = n as f64;
    let p_f = p as f64;
    let reml_total = (n_f - p_f) * rtv_invr.ln() + log_det_v + log_det_xtv_inv_x;
    let reml_c = (n_f - p_f) * ((n_f - p_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let reml = reml_c - 0.5 * reml_total;

    let ml_total = n_f * rtv_invr.ln() + log_det_v;
    let ml_c = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let ml = ml_c - 0.5 * ml_total;

    Some((reml, ml, rtv_invr, w, z, c_inv, beta))
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, max_iter=100, tol=1e-6, min_var=1e-12))]
pub fn ai_reml_null_f64<'py>(
    _py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
    min_var: f64,
) -> PyResult<(f64, f64, f64, f64, f64, usize, bool)> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if n <= p_cov {
        return Err(PyRuntimeError::new_err("n must be > p_cov"));
    }
    let min_var = if min_var.is_finite() && min_var > 0.0 {
        min_var
    } else {
        1e-12
    };
    let tol = if tol.is_finite() && tol > 0.0 {
        tol
    } else {
        1e-6
    };

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };

    let mean_y = y.iter().copied().sum::<f64>() / (n as f64);
    let mut var_y = 0.0_f64;
    for &yi in y {
        let d = yi - mean_y;
        var_y += d * d;
    }
    var_y /= (n.max(2) - 1) as f64;
    if !var_y.is_finite() || var_y <= 0.0 {
        var_y = 1.0;
    }
    let mut sigma_g2 = (0.5 * var_y).max(min_var);
    let mut sigma_e2 = (0.5 * var_y).max(min_var);

    let mut converged = false;
    let mut used_iter = 0usize;

    let mut state = ai_reml_eval(s, &xcov_flat, y, n, p_cov, sigma_g2, sigma_e2)
        .ok_or_else(|| PyRuntimeError::new_err("AIREML init failed"))?;

    for it in 0..max_iter {
        used_iter = it + 1;
        let (reml_curr, _ml_curr, _q_curr, w, z, c_inv, _beta) = &state;

        let tr_g = trace_p_d(s, &xcov_flat, n, p_cov, w, c_inv, true);
        let tr_e = trace_p_d(s, &xcov_flat, n, p_cov, w, c_inv, false);

        let mut q_g = 0.0_f64;
        let mut q_e = 0.0_f64;
        for i in 0..n {
            q_g += s[i] * z[i] * z[i];
            q_e += z[i] * z[i];
        }
        let score_g = -0.5 * (tr_g - q_g);
        let score_e = -0.5 * (tr_e - q_e);

        let mut dz_g = vec![0.0_f64; n];
        let mut dz_e = vec![0.0_f64; n];
        for i in 0..n {
            dz_g[i] = s[i] * z[i];
            dz_e[i] = z[i];
        }

        let mut p_dz_g = vec![0.0_f64; n];
        let mut p_dz_e = vec![0.0_f64; n];
        let mut tmp = vec![0.0_f64; n];
        let mut xtmp = vec![0.0_f64; p_cov];
        let mut cxtmp = vec![0.0_f64; p_cov];

        apply_p_diag_vec(
            &xcov_flat,
            n,
            p_cov,
            w,
            c_inv,
            &dz_g,
            &mut p_dz_g,
            &mut tmp,
            &mut xtmp,
            &mut cxtmp,
        );
        apply_p_diag_vec(
            &xcov_flat,
            n,
            p_cov,
            w,
            c_inv,
            &dz_e,
            &mut p_dz_e,
            &mut tmp,
            &mut xtmp,
            &mut cxtmp,
        );

        let mut ai_gg = 0.0_f64;
        let mut ai_ge = 0.0_f64;
        let mut ai_ee = 0.0_f64;
        for i in 0..n {
            ai_gg += dz_g[i] * p_dz_g[i];
            ai_ge += dz_g[i] * p_dz_e[i];
            ai_ee += dz_e[i] * p_dz_e[i];
        }
        ai_gg *= 0.5;
        ai_ge *= 0.5;
        ai_ee *= 0.5;

        let ridge = 1e-10;
        ai_gg += ridge;
        ai_ee += ridge;
        let det = ai_gg * ai_ee - ai_ge * ai_ge;
        if !det.is_finite() || det.abs() < 1e-18 {
            break;
        }
        let delta_g = (score_g * ai_ee - score_e * ai_ge) / det;
        let delta_e = (ai_gg * score_e - ai_ge * score_g) / det;
        if !delta_g.is_finite() || !delta_e.is_finite() {
            break;
        }

        let mut accepted = false;
        let mut step = 1.0_f64;
        let mut next_state = None;
        let mut next_sg = sigma_g2;
        let mut next_se = sigma_e2;

        for _ in 0..24 {
            let cand_sg = (sigma_g2 + step * delta_g).max(min_var);
            let cand_se = (sigma_e2 + step * delta_e).max(min_var);
            if let Some(st) = ai_reml_eval(s, &xcov_flat, y, n, p_cov, cand_sg, cand_se) {
                if st.0.is_finite() && st.0 >= *reml_curr - 1e-12 {
                    accepted = true;
                    next_state = Some(st);
                    next_sg = cand_sg;
                    next_se = cand_se;
                    break;
                }
            }
            step *= 0.5;
            if step < 1e-8 {
                break;
            }
        }

        if !accepted {
            break;
        }
        let rel_g = (next_sg - sigma_g2).abs() / sigma_g2.max(min_var);
        let rel_e = (next_se - sigma_e2).abs() / sigma_e2.max(min_var);
        sigma_g2 = next_sg;
        sigma_e2 = next_se;
        if let Some(st) = next_state {
            state = st;
        } else {
            break;
        }
        if rel_g.max(rel_e) < tol {
            converged = true;
            break;
        }
    }

    let (_reml, _ml, q, _w, _z, _c_inv, _beta) = &state;
    let n_f = n as f64;
    let p_f = p_cov as f64;
    let sigma_g2_out = (q / (n_f - p_f)).max(min_var);
    let sigma_e2_out = (sigma_e2 / sigma_g2).max(min_var) * sigma_g2_out;
    let lbd = (sigma_e2_out / sigma_g2_out).max(min_var);
    let reml = state.0;
    let ml = state.1;

    Ok((
        lbd,
        ml,
        reml,
        sigma_g2_out,
        sigma_e2_out,
        used_iter,
        converged,
    ))
}

#[inline]
fn trace_ab(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    let (nr, nc) = a.shape();
    debug_assert_eq!(b.shape(), (nr, nc));
    let mut s = 0.0_f64;
    for r in 0..nr {
        for c in 0..nc {
            s += a[(r, c)] * b[(c, r)];
        }
    }
    s
}

#[pyfunction]
#[pyo3(signature = (k_stack, xcov, y, max_iter=100, tol=1e-6, min_var=1e-12, trace_probes=8, trace_seed=42))]
pub fn ai_reml_multi_f64<'py>(
    py: Python<'py>,
    k_stack: PyReadonlyArray3<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
    min_var: f64,
    trace_probes: usize,
    trace_seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64, usize, bool)> {
    let kshape = k_stack.shape();
    if kshape.len() != 3 {
        return Err(PyRuntimeError::new_err("k_stack must be 3D (q, n, n)"));
    }
    let q = kshape[0];
    let n = kshape[1];
    if kshape[2] != n {
        return Err(PyRuntimeError::new_err("k_stack must be (q, n, n)"));
    }
    if q == 0 {
        return Err(PyRuntimeError::new_err(
            "k_stack requires at least one kernel",
        ));
    }

    let y_slice = y.as_slice()?;
    if y_slice.len() != n {
        return Err(PyRuntimeError::new_err("len(y) must equal n in k_stack"));
    }

    let x_arr = xcov.as_array();
    let (xn, p) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err(
            "xcov.n_rows must equal n in k_stack",
        ));
    }
    if p == 0 {
        return Err(PyRuntimeError::new_err(
            "xcov must have at least one column",
        ));
    }
    if n <= p {
        return Err(PyRuntimeError::new_err("n must be > p in xcov"));
    }

    let min_var = if min_var.is_finite() && min_var > 0.0 {
        min_var
    } else {
        1e-12
    };
    let tol = if tol.is_finite() && tol > 0.0 {
        tol
    } else {
        1e-6
    };

    let x_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let x_mat = DMatrix::<f64>::from_row_slice(n, p, &x_flat);
    let x_t = x_mat.transpose();
    let y_vec = DVector::<f64>::from_row_slice(y_slice);

    let mut kmats: Vec<DMatrix<f64>> = Vec::with_capacity(q);
    if let Ok(ks) = k_stack.as_slice() {
        for qi in 0..q {
            let off = qi * n * n;
            let mut km = DMatrix::<f64>::from_row_slice(n, n, &ks[off..off + n * n]);
            km = (&km + km.transpose()) * 0.5;
            kmats.push(km);
        }
    } else {
        let k_arr = k_stack.as_array();
        for qi in 0..q {
            let mut buf = vec![0.0_f64; n * n];
            for r in 0..n {
                for c in 0..n {
                    buf[r * n + c] = k_arr[(qi, r, c)];
                }
            }
            let mut km = DMatrix::<f64>::from_row_slice(n, n, &buf);
            km = (&km + km.transpose()) * 0.5;
            kmats.push(km);
        }
    }
    let eye = DMatrix::<f64>::identity(n, n);
    let use_exact_trace = trace_probes == 0 || n <= 768;
    let probes_n = trace_probes.max(1);
    let mut trace_probes_vec: Vec<DVector<f64>> = Vec::new();
    if !use_exact_trace {
        let mut rng = StdRng::seed_from_u64(trace_seed);
        trace_probes_vec.reserve(probes_n);
        for _ in 0..probes_n {
            let mut z = DVector::<f64>::zeros(n);
            for i in 0..n {
                z[i] = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 };
            }
            trace_probes_vec.push(z);
        }
    }

    let mut var_y = {
        let mean = y_slice.iter().copied().sum::<f64>() / (n as f64);
        let mut s2 = 0.0_f64;
        for &yi in y_slice {
            let d = yi - mean;
            s2 += d * d;
        }
        s2 / ((n.max(2) - 1) as f64)
    };
    if !var_y.is_finite() || var_y <= 0.0 {
        var_y = 1.0;
    }

    let m = q + 1; // q kernels + residual
    let mut theta = DVector::<f64>::from_element(m, (var_y / (m as f64)).max(min_var));
    let mut converged = false;
    let mut used_iter = 0usize;

    let mut last_ml = f64::NAN;
    let mut last_reml = f64::NAN;

    for it in 0..max_iter {
        used_iter = it + 1;

        // Build V = sum_j theta_j K_j + theta_e I
        let mut v = eye.clone() * theta[m - 1];
        for j in 0..q {
            v += &kmats[j] * theta[j];
        }
        let Some(chol_v) = v.clone().cholesky() else {
            break;
        };

        let vinv_x = chol_v.solve(&x_mat);
        let xt_vinv_x = &x_t * &vinv_x;
        let Some(chol_x) = xt_vinv_x.clone().cholesky() else {
            break;
        };
        let c_inv = chol_x.inverse();

        let vinv_y = chol_v.solve(&y_vec);
        let xt_vinv_y = &x_t * &vinv_y;
        let beta = chol_x.solve(&xt_vinv_y);

        let r_vec = &y_vec - &x_mat * beta;
        let vinv_r = chol_v.solve(&r_vec);
        let proj = &vinv_x * (&c_inv * (&x_t * &vinv_r));
        let alpha = &vinv_r - proj; // alpha = P y
        let qval = r_vec.dot(&vinv_r);
        if !qval.is_finite() || qval <= 0.0 {
            break;
        }

        let lv = chol_v.l();
        let mut log_det_v = 0.0_f64;
        for i in 0..n {
            log_det_v += 2.0 * lv[(i, i)].ln();
        }
        let lx = chol_x.l();
        let mut log_det_xt = 0.0_f64;
        for i in 0..p {
            log_det_xt += 2.0 * lx[(i, i)].ln();
        }
        let n_f = n as f64;
        let p_f = p as f64;
        let reml_total = (n_f - p_f) * qval.ln() + log_det_v + log_det_xt;
        let reml_c = (n_f - p_f) * ((n_f - p_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
        let reml_curr = reml_c - 0.5 * reml_total;
        let ml_total = n_f * qval.ln() + log_det_v;
        let ml_c = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
        let ml_curr = ml_c - 0.5 * ml_total;
        if !reml_curr.is_finite() || !ml_curr.is_finite() {
            break;
        }
        last_reml = reml_curr;
        last_ml = ml_curr;

        let mut score = DVector::<f64>::zeros(m);
        let mut ka_list: Vec<DVector<f64>> = Vec::with_capacity(m);
        let mut pka_list: Vec<DVector<f64>> = Vec::with_capacity(m);
        let b_t = vinv_x.transpose();
        let vinv_exact = if use_exact_trace {
            Some(chol_v.inverse())
        } else {
            None
        };
        let mut vinv_probe_vec: Vec<DVector<f64>> = Vec::new();
        if !use_exact_trace {
            vinv_probe_vec.reserve(probes_n);
            for z in &trace_probes_vec {
                vinv_probe_vec.push(chol_v.solve(z));
            }
        }

        for j in 0..m {
            let kj = if j < q { &kmats[j] } else { &eye };

            let ka = kj * &alpha;
            let quad = alpha.dot(&ka);
            ka_list.push(ka.clone());

            let tr1 = if let Some(vinv) = &vinv_exact {
                trace_ab(vinv, kj)
            } else {
                // Hutchinson trace estimate for tr(V^{-1} K_j) using cached V^{-1}z.
                let mut s = 0.0_f64;
                for (z, vinv_z) in trace_probes_vec.iter().zip(vinv_probe_vec.iter()) {
                    let kz = kj * z;
                    s += vinv_z.dot(&kz);
                }
                s / probes_n as f64
            };

            // Exact correction tr((X'V^{-1}X)^{-1} X'V^{-1}K_jV^{-1}X)
            // = tr(C^{-1} B'K_jB), where B = V^{-1}X.
            let kb = kj * &vinv_x;
            let bt_kb = &b_t * &kb;
            let tr2 = trace_ab(&c_inv, &bt_kb);
            let tr_pk = tr1 - tr2;
            score[j] = -0.5 * (tr_pk - quad);

            let vinv_ka = chol_v.solve(&ka);
            let pka = &vinv_ka - &vinv_x * (&c_inv * (&b_t * &ka));
            pka_list.push(pka);
        }

        let mut ai = DMatrix::<f64>::zeros(m, m);
        for a in 0..m {
            for b in 0..=a {
                let v_ab = 0.5 * ka_list[a].dot(&pka_list[b]);
                ai[(a, b)] = v_ab;
                ai[(b, a)] = v_ab;
            }
        }
        for d in 0..m {
            ai[(d, d)] += 1e-10;
        }

        let Some(delta) = ai.lu().solve(&score) else {
            break;
        };
        if !delta.iter().all(|v| v.is_finite()) {
            break;
        }

        let mut accepted = false;
        let mut step = 1.0_f64;
        let mut theta_next = theta.clone();
        let mut reml_next = reml_curr;
        let mut ml_next = ml_curr;

        for _ in 0..24 {
            let mut cand = theta.clone();
            for j in 0..m {
                cand[j] = (theta[j] + step * delta[j]).max(min_var);
            }

            let mut v_c = eye.clone() * cand[m - 1];
            for j in 0..q {
                v_c += &kmats[j] * cand[j];
            }
            let Some(chol_vc) = v_c.clone().cholesky() else {
                step *= 0.5;
                if step < 1e-8 {
                    break;
                }
                continue;
            };
            let vinv_xc = chol_vc.solve(&x_mat);
            let xt_vinv_xc = &x_t * &vinv_xc;
            let Some(chol_xc) = xt_vinv_xc.clone().cholesky() else {
                step *= 0.5;
                if step < 1e-8 {
                    break;
                }
                continue;
            };
            let vinv_yc = chol_vc.solve(&y_vec);
            let xt_vinv_yc = &x_t * &vinv_yc;
            let beta_c = chol_xc.solve(&xt_vinv_yc);
            let r_c = &y_vec - &x_mat * beta_c;
            let vinv_rc = chol_vc.solve(&r_c);
            let q_c = r_c.dot(&vinv_rc);
            if !q_c.is_finite() || q_c <= 0.0 {
                step *= 0.5;
                if step < 1e-8 {
                    break;
                }
                continue;
            }
            let lv_c = chol_vc.l();
            let mut log_det_vc = 0.0_f64;
            for i in 0..n {
                log_det_vc += 2.0 * lv_c[(i, i)].ln();
            }
            let lx_c = chol_xc.l();
            let mut log_det_xc = 0.0_f64;
            for i in 0..p {
                log_det_xc += 2.0 * lx_c[(i, i)].ln();
            }
            let reml_total_c = (n_f - p_f) * q_c.ln() + log_det_vc + log_det_xc;
            let reml_cand = reml_c - 0.5 * reml_total_c;
            let ml_total_c = n_f * q_c.ln() + log_det_vc;
            let ml_cand = ml_c - 0.5 * ml_total_c;
            if reml_cand.is_finite() && reml_cand >= reml_curr - 1e-12 {
                accepted = true;
                theta_next = cand;
                reml_next = reml_cand;
                ml_next = ml_cand;
                break;
            }
            step *= 0.5;
            if step < 1e-8 {
                break;
            }
        }

        if !accepted {
            break;
        }

        let mut rel_max = 0.0_f64;
        for j in 0..m {
            let rel = (theta_next[j] - theta[j]).abs() / theta[j].max(min_var);
            if rel > rel_max {
                rel_max = rel;
            }
        }
        theta = theta_next;
        last_reml = reml_next;
        last_ml = ml_next;

        if rel_max < tol {
            converged = true;
            break;
        }
    }

    let theta_arr = PyArray1::<f64>::from_vec(py, theta.iter().copied().collect()).into_bound();
    Ok((theta_arr, last_ml, last_reml, used_iter, converged))
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, g_rot_chunk, max_iter=50, tol=1e-2, threads=0, nullml=None))]
pub fn lmm_reml_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    g_rot_chunk: PyReadonlyArray2<'py, f32>,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let g_arr = g_rot_chunk.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("g_rot_chunk must be (m_chunk, n)"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let m_chunk = g_arr.shape()[0];
    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let g_slice_opt = g_rot_chunk.as_slice().ok();

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let beta_se_p = PyArray2::<f64>::zeros(py, [m_chunk, out_cols], false).into_bound();

    let beta_se_p_slice: &mut [f64] = unsafe {
        beta_se_p
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("beta_se_p not contiguous"))?
    };
    // For REML chunk scanning, rebuilding a local pool is consistently faster
    // than thread-local pool reuse in CLI LMM benchmarks.
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

    py.detach(|| {
        let mut run = || {
            beta_se_p_slice
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || vec![0.0_f64; n],
                    |snp_vec, (idx, out_row)| {
                        if let Some(gs) = g_slice_opt {
                            let row = &gs[idx * n..(idx + 1) * n];
                            for i in 0..n {
                                snp_vec[i] = row[i] as f64;
                            }
                        } else {
                            let row = g_arr.row(idx);
                            for i in 0..n {
                                snp_vec[i] = row[i] as f64;
                            }
                        }

                        let (best_log10_lbd, _best_cost) = brent_minimize(
                            |x| -reml_loglike(x, s, &xcov_flat, y, Some(&snp_vec[..]), n, p_cov),
                            low,
                            high,
                            tol,
                            max_iter,
                        );

                        let (beta, se, _lbd) =
                            final_beta_se(best_log10_lbd, s, &xcov_flat, y, &snp_vec, n, p_cov);

                        let p = if beta.is_finite() && se.is_finite() && se > 0.0 {
                            let z = beta / se;
                            (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta;
                        out_row[1] = se;
                        out_row[2] = if p.is_finite() { p } else { 1.0 };

                        if with_plrt {
                            let ml = ml_loglike(
                                best_log10_lbd,
                                s,
                                &xcov_flat,
                                y,
                                Some(&snp_vec[..]),
                                n,
                                p_cov,
                            );
                            out_row[3] = if ml.is_finite() {
                                let mut stat = 2.0 * (ml - nullml_val);
                                if !stat.is_finite() || stat < 0.0 {
                                    stat = 0.0;
                                }
                                chi2_sf_df1(stat)
                            } else {
                                1.0
                            };
                        }
                    },
                );
        };

        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    });

    Ok(beta_se_p)
}

// ------------------------------------------------------------
// Helpers: dot loops
// ------------------------------------------------------------

#[inline]
fn dot_loop(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn dot_loop_unrolled(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut i = 0usize;
    while i + 3 < n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = s0 + s1 + s2 + s3;
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[inline]
fn axpy_scaled_unrolled(y: &mut [f64], x: &[f64], alpha: f64) {
    debug_assert_eq!(y.len(), x.len());
    if alpha == 0.0 {
        return;
    }
    let n = y.len();
    let mut i = 0usize;
    while i + 3 < n {
        y[i] += x[i] * alpha;
        y[i + 1] += x[i + 1] * alpha;
        y[i + 2] += x[i + 2] * alpha;
        y[i + 3] += x[i + 3] * alpha;
        i += 4;
    }
    while i < n {
        y[i] += x[i] * alpha;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_loop_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd,
        _mm256_storeu_pd,
    };

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0usize;
    let mut acc: __m256d = _mm256_setzero_pd();
    while i + 4 <= n {
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };
        acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
        i += 4;
    }
    let mut lanes = [0.0_f64; 4];
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    let mut s = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn axpy_scaled_avx2(y: &mut [f64], x: &[f64], alpha: f64) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    debug_assert_eq!(y.len(), x.len());
    let n = y.len();
    let mut i = 0usize;
    let a = _mm256_set1_pd(alpha);
    while i + 4 <= n {
        let vy = unsafe { _mm256_loadu_pd(y.as_ptr().add(i)) };
        let vx = unsafe { _mm256_loadu_pd(x.as_ptr().add(i)) };
        let out = _mm256_add_pd(vy, _mm256_mul_pd(vx, a));
        unsafe { _mm256_storeu_pd(y.as_mut_ptr().add(i), out) };
        i += 4;
    }
    while i < n {
        y[i] += x[i] * alpha;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_loop_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::{
        vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64,
    };

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0usize;
    let mut acc = vdupq_n_f64(0.0);
    while i + 2 <= n {
        let va = unsafe { vld1q_f64(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_f64(b.as_ptr().add(i)) };
        acc = vaddq_f64(acc, vmulq_f64(va, vb));
        i += 2;
    }
    let mut lanes = [0.0_f64; 2];
    unsafe { vst1q_f64(lanes.as_mut_ptr(), acc) };
    let mut s = lanes[0] + lanes[1];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "aarch64")]
unsafe fn axpy_scaled_neon(y: &mut [f64], x: &[f64], alpha: f64) {
    use std::arch::aarch64::{
        vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64,
    };

    debug_assert_eq!(y.len(), x.len());
    let n = y.len();
    let mut i = 0usize;
    let a = vdupq_n_f64(alpha);
    while i + 2 <= n {
        let vy = unsafe { vld1q_f64(y.as_ptr().add(i)) };
        let vx = unsafe { vld1q_f64(x.as_ptr().add(i)) };
        let out = vaddq_f64(vy, vmulq_f64(vx, a));
        unsafe { vst1q_f64(y.as_mut_ptr().add(i), out) };
        i += 2;
    }
    while i < n {
        y[i] += x[i] * alpha;
        i += 1;
    }
}

#[inline]
fn dot_loop_simd(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_loop_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_loop_neon(a, b) };
    }
    #[allow(unreachable_code)]
    dot_loop_unrolled(a, b)
}

#[inline]
fn axpy_scaled_simd(y: &mut [f64], x: &[f64], alpha: f64) {
    if alpha == 0.0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                axpy_scaled_avx2(y, x, alpha);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            axpy_scaled_neon(y, x, alpha);
        }
        return;
    }
    #[allow(unreachable_code)]
    axpy_scaled_unrolled(y, x, alpha)
}

#[inline]
fn auto_precond_rank(n: usize, m: usize, threads: usize) -> usize {
    let mut rank = if n <= 512 {
        16
    } else if n <= 2048 {
        32
    } else if n <= 4096 {
        40
    } else {
        48
    };
    if m <= 1024 {
        rank = rank.min(24);
    }
    if m >= 20000 && n >= 2048 {
        rank = (rank + 8).min(96);
    }
    if threads >= 16 {
        rank = (rank + 8).min(96);
    }
    rank.min(n).max(1)
}

#[inline]
fn auto_block_size(n: usize, m: usize, threads: usize) -> usize {
    // Prefer a conservative block size for lower peak memory / stable throughput.
    let mut bs = if n >= 8000 {
        1
    } else if n >= 4000 {
        2
    } else if n >= 2000 {
        if m >= 1500 {
            8
        } else {
            4
        }
    } else if n >= 1000 {
        8
    } else {
        12
    };
    let t = threads.max(1);
    let target_chunks = t.saturating_mul(6).max(1);
    let cap = ((m + target_chunks - 1) / target_chunks).max(1);
    bs = bs.min(cap);
    bs.clamp(1, 32)
}

#[inline]
fn auto_scan_rhs_cap(
    requested_block_size: usize,
    n: usize,
    precond_rank: usize,
    threads: usize,
) -> usize {
    let requested = requested_block_size.max(1);
    let mut rhs_cap = if n >= 12000 {
        1
    } else if n >= 6000 {
        2
    } else if n >= 3000 {
        4
    } else if n >= 1500 {
        6
    } else {
        8
    };
    if precond_rank >= 64 {
        rhs_cap = rhs_cap.min(4);
    }
    if threads >= 16 {
        rhs_cap = rhs_cap.min(4);
    }

    // Keep per-thread scratch within a conservative budget.
    let budget_bytes: usize = 384 * 1024 * 1024;
    let t = threads.max(1);
    let per_thread_budget = budget_bytes / t;
    let fixed_bytes = 8usize.saturating_mul(n); // centered_row
    let per_rhs_bytes = 8usize.saturating_mul(
        6usize
            .saturating_mul(n)
            .saturating_add(precond_rank)
            .saturating_add(6),
    );
    if per_rhs_bytes > 0 {
        if per_thread_budget > fixed_bytes {
            let mem_cap = (per_thread_budget - fixed_bytes) / per_rhs_bytes;
            rhs_cap = rhs_cap.min(mem_cap.max(1));
        } else {
            rhs_cap = 1;
        }
    }

    rhs_cap.clamp(1, requested)
}

#[inline]
fn score_vr_maf_bin(maf: f64) -> usize {
    if maf < 0.01 {
        0
    } else if maf < 0.05 {
        1
    } else if maf < 0.10 {
        2
    } else if maf < 0.20 {
        3
    } else if maf <= 0.5 {
        4
    } else {
        5
    }
}

fn sample_even_indices(src: &[usize], k: usize) -> Vec<usize> {
    if k == 0 || src.is_empty() {
        return Vec::new();
    }
    if k >= src.len() {
        return src.to_vec();
    }
    let len = src.len();
    let mut out = Vec::<usize>::with_capacity(k);
    for t in 0..k {
        let num = (2 * t + 1) * len;
        let den = 2 * k;
        let pos = (num / den).min(len - 1);
        out.push(src[pos]);
    }
    out
}

fn choose_score_vr_calibration_indices(g_flat: &[f32], m_scan: usize, n: usize, calib: usize) -> Vec<usize> {
    if m_scan == 0 {
        return Vec::new();
    }
    let calib = calib.min(m_scan).max(1);
    let mut bins: [Vec<usize>; 6] = std::array::from_fn(|_| Vec::new());
    for row in 0..m_scan {
        let row_off = row * n;
        let row_slice = &g_flat[row_off..row_off + n];
        let mut sum = 0.0_f64;
        let mut cnt = 0usize;
        for &v in row_slice.iter() {
            if v.is_finite() && v >= 0.0 {
                sum += v as f64;
                cnt += 1;
            }
        }
        if cnt == 0 {
            continue;
        }
        let p = (sum / (cnt as f64)) * 0.5;
        let maf = p.min(1.0 - p).clamp(0.0, 0.5);
        let b = score_vr_maf_bin(maf);
        bins[b].push(row);
    }

    let mut nonempty = Vec::<usize>::new();
    let mut total = 0usize;
    for b in 0..bins.len() {
        if !bins[b].is_empty() {
            nonempty.push(b);
            total += bins[b].len();
        }
    }
    if nonempty.is_empty() {
        return sample_even_indices(&(0..m_scan).collect::<Vec<_>>(), calib);
    }

    let mut quota = vec![0usize; bins.len()];
    let mut quota_sum = 0usize;
    for &b in nonempty.iter() {
        let len_b = bins[b].len();
        let mut q = (calib * len_b + total / 2) / total; // nearest integer
        q = q.max(1).min(len_b);
        quota[b] = q;
        quota_sum += q;
    }

    while quota_sum > calib {
        let mut changed = false;
        for &b in nonempty.iter() {
            if quota_sum <= calib {
                break;
            }
            if quota[b] > 1 {
                quota[b] -= 1;
                quota_sum -= 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    while quota_sum < calib {
        let mut changed = false;
        for &b in nonempty.iter() {
            if quota_sum >= calib {
                break;
            }
            if quota[b] < bins[b].len() {
                quota[b] += 1;
                quota_sum += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut selected = Vec::<usize>::with_capacity(calib);
    let mut used = vec![false; m_scan];
    for &b in nonempty.iter() {
        let picks = sample_even_indices(&bins[b], quota[b]);
        for idx in picks {
            if !used[idx] {
                used[idx] = true;
                selected.push(idx);
            }
        }
    }

    if selected.len() < calib {
        let all = (0..m_scan).collect::<Vec<_>>();
        let extra = sample_even_indices(&all, calib);
        for idx in extra {
            if selected.len() >= calib {
                break;
            }
            if !used[idx] {
                used[idx] = true;
                selected.push(idx);
            }
        }
    }
    selected.truncate(calib);
    selected
}

#[derive(Clone)]
struct PackedGeno2Bit {
    n: usize,
    m: usize,
    bytes_per_snp: usize,
    // SNP-major packed bytes, 2-bit per sample:
    // 00 -> 0, 01 -> 1, 10 -> 2, 11 -> missing
    packed: Vec<u8>,
    row_mean: Vec<f64>,
    diag_k: Vec<f64>,
    mean_diag_k: f64,
}

#[inline]
fn quantize_hardcall(v: f32) -> Option<u8> {
    let r = v.round();
    if (v - r).abs() <= 1e-3 && r >= 0.0 && r <= 2.0 {
        Some(r as u8)
    } else {
        None
    }
}

fn pack_snp_major_genotype_2bit(g_flat: &[f32], m: usize, n: usize) -> PyResult<PackedGeno2Bit> {
    if m == 0 || n == 0 {
        return Err(PyRuntimeError::new_err(
            "kinship_geno must be non-empty (m, n)",
        ));
    }
    if g_flat.len() != m * n {
        return Err(PyRuntimeError::new_err("kinship_geno flat length mismatch"));
    }

    let bytes_per_snp = (n + 3) / 4;
    let mut packed = vec![0_u8; m * bytes_per_snp];
    let mut row_mean = vec![0.0_f64; m];
    let mut diag_acc = vec![0.0_f64; n];

    for snp in 0..m {
        let row = &g_flat[snp * n..(snp + 1) * n];

        let mut sum = 0.0_f64;
        let mut cnt = 0_usize;
        for (i, &v) in row.iter().enumerate() {
            if !v.is_finite() || v < 0.0 {
                continue;
            }
            let code = quantize_hardcall(v).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "kinship_geno must be hard-call 0/1/2 (or missing). invalid value={v} at snp={snp}, sample={i}"
                ))
            })?;
            sum += code as f64;
            cnt += 1;
        }
        let mu = if cnt > 0 { sum / (cnt as f64) } else { 0.0 };
        row_mean[snp] = mu;

        let c0 = -mu;
        let c1 = 1.0 - mu;
        let c2 = 2.0 - mu;

        let row_bytes = &mut packed[snp * bytes_per_snp..(snp + 1) * bytes_per_snp];
        row_bytes.fill(0_u8);
        for i in 0..n {
            let v = row[i];
            let code = if !v.is_finite() || v < 0.0 {
                3_u8
            } else {
                quantize_hardcall(v).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "kinship_geno must be hard-call 0/1/2 (or missing). invalid value={v} at snp={snp}, sample={i}"
                    ))
                })?
            };
            let centered = match code {
                0 => c0,
                1 => c1,
                2 => c2,
                _ => 0.0, // missing => imputed mean => centered 0
            };
            diag_acc[i] += centered * centered;

            let byte_idx = i >> 2;
            let shift = (i & 3) * 2;
            row_bytes[byte_idx] |= code << shift;
        }
    }

    let inv_m = 1.0 / (m as f64);
    let mut diag_k = vec![0.0_f64; n];
    let mut mean_diag_k = 0.0_f64;
    for i in 0..n {
        let d = (diag_acc[i] * inv_m).max(1e-12);
        diag_k[i] = d;
        mean_diag_k += d;
    }
    mean_diag_k /= n as f64;

    Ok(PackedGeno2Bit {
        n,
        m,
        bytes_per_snp,
        packed,
        row_mean,
        diag_k,
        mean_diag_k,
    })
}

#[inline]
fn accumulate_dot_tile(
    centered_row: &[f64],
    p_block: &[f64],
    rhs: usize,
    n: usize,
    base: usize,
    end: usize,
    dots: &mut [f64],
) {
    if base >= end {
        return;
    }
    let len = end - base;
    let x_tile = &centered_row[base..end];
    for c in 0..rhs {
        let off = c * n + base;
        let y_tile = &p_block[off..off + len];
        dots[c] += dot_loop_simd(x_tile, y_tile);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn decode_16bytes_to_codes64_neon(src: *const u8, dst: *mut u8) {
    use std::arch::aarch64::{
        uint8x16x4_t, vandq_u8, vdupq_n_u8, vld1q_u8, vshrq_n_u8, vst4q_u8,
    };

    let b = unsafe { vld1q_u8(src) };
    let mask = vdupq_n_u8(0x03);
    let c0 = vandq_u8(b, mask);
    let c1 = vandq_u8(vshrq_n_u8(b, 2), mask);
    let c2 = vandq_u8(vshrq_n_u8(b, 4), mask);
    let c3 = vshrq_n_u8(b, 6);
    let quad = uint8x16x4_t(c0, c1, c2, c3);
    unsafe { vst4q_u8(dst, quad) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_32bytes_to_code_planes_avx2(
    src: *const u8,
    p0: *mut u8,
    p1: *mut u8,
    p2: *mut u8,
    p3: *mut u8,
) {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_srli_epi16,
        _mm256_storeu_si256,
    };

    let b = unsafe { _mm256_loadu_si256(src as *const __m256i) };
    let mask = _mm256_set1_epi8(0x03);
    let c0 = _mm256_and_si256(b, mask);
    let c1 = _mm256_and_si256(_mm256_srli_epi16(b, 2), mask);
    let c2 = _mm256_and_si256(_mm256_srli_epi16(b, 4), mask);
    let c3 = _mm256_and_si256(_mm256_srli_epi16(b, 6), mask);
    unsafe {
        _mm256_storeu_si256(p0 as *mut __m256i, c0);
        _mm256_storeu_si256(p1 as *mut __m256i, c1);
        _mm256_storeu_si256(p2 as *mut __m256i, c2);
        _mm256_storeu_si256(p3 as *mut __m256i, c3);
    }
}

#[inline]
fn decode_centered_row_and_dot_packed_2bit(
    op: &PackedGeno2Bit,
    snp_idx: usize,
    p_block: &[f64],
    rhs: usize,
    centered_row: &mut [f64],
    dots: &mut [f64],
) {
    let n = op.n;
    const DOT_TILE_N: usize = 256;

    debug_assert!(snp_idx < op.m);
    debug_assert_eq!(centered_row.len(), n);
    debug_assert_eq!(p_block.len(), rhs * n);
    debug_assert!(dots.len() >= rhs);
    if rhs == 0 {
        return;
    }

    dots[..rhs].fill(0.0);

    let mu = op.row_mean[snp_idx];
    let c0 = -mu;
    let c1 = 1.0 - mu;
    let c2 = 2.0 - mu;
    let c3 = 0.0_f64;

    let off = snp_idx * op.bytes_per_snp;
    let row_bytes = &op.packed[off..off + op.bytes_per_snp];
    let mut i = 0usize;
    let mut byte_i = 0usize;
    let mut tile_start = 0usize;
    let lut = [c0, c1, c2, c3];

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            const BYTES_PER_SIMD: usize = 32;
            const CODES_PER_SIMD: usize = BYTES_PER_SIMD * 4;
            let mut code0 = [0_u8; BYTES_PER_SIMD];
            let mut code1 = [0_u8; BYTES_PER_SIMD];
            let mut code2 = [0_u8; BYTES_PER_SIMD];
            let mut code3 = [0_u8; BYTES_PER_SIMD];
            while byte_i + BYTES_PER_SIMD <= row_bytes.len() && i + CODES_PER_SIMD <= n {
                unsafe {
                    decode_32bytes_to_code_planes_avx2(
                        row_bytes.as_ptr().add(byte_i),
                        code0.as_mut_ptr(),
                        code1.as_mut_ptr(),
                        code2.as_mut_ptr(),
                        code3.as_mut_ptr(),
                    );
                }
                for j in 0..BYTES_PER_SIMD {
                    let base = i + (j << 2);
                    centered_row[base] = lut[code0[j] as usize];
                    centered_row[base + 1] = lut[code1[j] as usize];
                    centered_row[base + 2] = lut[code2[j] as usize];
                    centered_row[base + 3] = lut[code3[j] as usize];
                }
                i += CODES_PER_SIMD;
                byte_i += BYTES_PER_SIMD;

                while i >= tile_start + DOT_TILE_N {
                    let end = tile_start + DOT_TILE_N;
                    accumulate_dot_tile(centered_row, p_block, rhs, n, tile_start, end, dots);
                    tile_start = end;
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        const BYTES_PER_SIMD: usize = 16;
        const CODES_PER_SIMD: usize = BYTES_PER_SIMD * 4;
        let mut codes = [0_u8; CODES_PER_SIMD];
        while byte_i + BYTES_PER_SIMD <= row_bytes.len() && i + CODES_PER_SIMD <= n {
            unsafe {
                decode_16bytes_to_codes64_neon(
                    row_bytes.as_ptr().add(byte_i),
                    codes.as_mut_ptr(),
                );
            }
            for j in 0..CODES_PER_SIMD {
                centered_row[i + j] = lut[codes[j] as usize];
            }
            i += CODES_PER_SIMD;
            byte_i += BYTES_PER_SIMD;

            while i >= tile_start + DOT_TILE_N {
                let end = tile_start + DOT_TILE_N;
                accumulate_dot_tile(centered_row, p_block, rhs, n, tile_start, end, dots);
                tile_start = end;
            }
        }
    }

    while byte_i < row_bytes.len() {
        let byte = row_bytes[byte_i];
        byte_i += 1;
        for shift in [0_u8, 2, 4, 6] {
            if i >= n {
                break;
            }
            let code = (byte >> shift) & 0b11;
            centered_row[i] = lut[code as usize];
            i += 1;
        }
        while i >= tile_start + DOT_TILE_N {
            let end = tile_start + DOT_TILE_N;
            accumulate_dot_tile(centered_row, p_block, rhs, n, tile_start, end, dots);
            tile_start = end;
        }
    }

    if tile_start < n {
        accumulate_dot_tile(centered_row, p_block, rhs, n, tile_start, n, dots);
    }
}

#[inline]
fn matmat_k_plus_lambda_packed_2bit(
    op: &PackedGeno2Bit,
    lbd: f64,
    p_block: &[f64], // column-major: rhs blocks of len n
    rhs: usize,
    out_block: &mut [f64], // column-major
    centered_row: &mut [f64],
    dots: &mut [f64],
) {
    let n = op.n;
    debug_assert_eq!(p_block.len(), rhs * n);
    debug_assert_eq!(out_block.len(), rhs * n);
    debug_assert_eq!(centered_row.len(), n);
    debug_assert!(dots.len() >= rhs);
    if rhs == 0 {
        return;
    }

    out_block.fill(0.0);
    for snp_idx in 0..op.m {
        // Decode + dot are fused in one pass to reduce memory traffic.
        decode_centered_row_and_dot_packed_2bit(op, snp_idx, p_block, rhs, centered_row, dots);

        for c in 0..rhs {
            let d = dots[c];
            if d == 0.0 {
                continue;
            }
            let out_col = &mut out_block[c * n..(c + 1) * n];
            axpy_scaled_simd(out_col, centered_row, d);
        }
    }

    let inv_m = 1.0 / (op.m as f64);
    for c in 0..rhs {
        let off = c * n;
        for i in 0..n {
            out_block[off + i] = out_block[off + i] * inv_m + lbd * p_block[off + i];
        }
    }
}

struct RandNysPrecond {
    n: usize,
    rank: usize,
    // Column-major (n, rank)
    u: Vec<f64>,
    // Approximate top eigenvalues of K
    d: Vec<f64>,
}

fn mgs_orthonormalize_columns(mat: &mut [f64], n: usize, cols: usize) -> usize {
    let mut kept = 0usize;
    for j in 0..cols {
        for k in 0..kept {
            let proj = dot_loop(&mat[k * n..(k + 1) * n], &mat[j * n..(j + 1) * n]);
            for i in 0..n {
                mat[j * n + i] -= proj * mat[k * n + i];
            }
        }
        let norm2 = dot_loop(&mat[j * n..(j + 1) * n], &mat[j * n..(j + 1) * n]);
        if !norm2.is_finite() || norm2 <= 1e-18 {
            continue;
        }
        let inv = 1.0 / norm2.sqrt();
        if kept != j {
            for i in 0..n {
                mat[kept * n + i] = mat[j * n + i] * inv;
            }
        } else {
            for i in 0..n {
                mat[j * n + i] *= inv;
            }
        }
        kept += 1;
    }
    kept
}

fn build_rand_nys_precond(
    op: &PackedGeno2Bit,
    rank: usize,
    oversample: usize,
    seed: u64,
) -> Option<RandNysPrecond> {
    let n = op.n;
    if n < 2 || rank == 0 {
        return None;
    }
    let rank = rank.min(n).max(1);
    let q = (rank.saturating_mul(oversample.max(1))).max(rank).min(n);
    if q == 0 {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = vec![0.0_f64; n * q];
    for v in omega.iter_mut() {
        *v = rng.sample(StandardNormal);
    }
    let q_eff = mgs_orthonormalize_columns(&mut omega, n, q);
    if q_eff == 0 {
        return None;
    }
    omega.truncate(n * q_eff);

    let mut y = vec![0.0_f64; n * q_eff];
    let mut centered = vec![0.0_f64; n];
    let mut dots = vec![0.0_f64; q_eff];
    matmat_k_plus_lambda_packed_2bit(op, 0.0, &omega, q_eff, &mut y, &mut centered, &mut dots);

    let norm_y = dot_loop(&y, &y).sqrt();
    let shift = (n as f64).sqrt() * f64::EPSILON * norm_y.max(1.0);
    for i in 0..(n * q_eff) {
        y[i] += shift * omega[i];
    }

    // W = (Omega^T Y + Y^T Omega) / 2, row-major.
    let mut w = vec![0.0_f64; q_eff * q_eff];
    for i in 0..q_eff {
        for j in 0..=i {
            let oi = &omega[i * n..(i + 1) * n];
            let oj = &omega[j * n..(j + 1) * n];
            let yi = &y[i * n..(i + 1) * n];
            let yj = &y[j * n..(j + 1) * n];
            let v = 0.5 * (dot_loop(oi, yj) + dot_loop(oj, yi));
            w[i * q_eff + j] = v;
            w[j * q_eff + i] = v;
        }
    }
    let mut l = w;
    cholesky_inplace(&mut l, q_eff)?;
    drop(omega);

    // Convert Y to B = Y * L^{-T} in-place, stored column-major (n, q_eff).
    let mut tmp = vec![0.0_f64; q_eff];
    let mut sol = vec![0.0_f64; q_eff];
    for row in 0..n {
        for t in 0..q_eff {
            tmp[t] = y[t * n + row];
        }
        for r in 0..q_eff {
            let mut s = tmp[r];
            for k in 0..r {
                s -= l[r * q_eff + k] * sol[k];
            }
            let d = l[r * q_eff + r];
            if !d.is_finite() || d.abs() < 1e-14 {
                return None;
            }
            sol[r] = s / d;
        }
        for r in 0..q_eff {
            y[r * n + row] = sol[r];
        }
    }

    let mut btb = vec![0.0_f64; q_eff * q_eff];
    for i in 0..q_eff {
        for j in 0..=i {
            let v = dot_loop(&y[i * n..(i + 1) * n], &y[j * n..(j + 1) * n]);
            btb[i * q_eff + j] = v;
            btb[j * q_eff + i] = v;
        }
    }
    let eig = nalgebra::SymmetricEigen::new(DMatrix::from_row_slice(q_eff, q_eff, &btb));
    let mut order: Vec<usize> = (0..q_eff).collect();
    order.sort_by(|&a, &bidx| eig.eigenvalues[bidx].total_cmp(&eig.eigenvalues[a]));

    let mut u_cols = vec![0.0_f64; n * rank];
    let mut d_vals = Vec::<f64>::with_capacity(rank);
    let mut kept = 0usize;
    for idx in order {
        if kept >= rank {
            break;
        }
        let s2 = eig.eigenvalues[idx];
        if !s2.is_finite() || s2 <= 1e-12 {
            continue;
        }
        let lam = (s2 - shift).max(0.0);
        if lam <= 1e-12 {
            continue;
        }
        let s = s2.sqrt();
        if s <= 1e-12 || !s.is_finite() {
            continue;
        }

        for i in 0..n {
            let mut acc = 0.0_f64;
            for t in 0..q_eff {
                let v_t = eig.eigenvectors[(t, idx)];
                acc += y[t * n + i] * v_t;
            }
            u_cols[kept * n + i] = acc / s;
        }

        // Re-orthogonalize the produced vector for numerical stability.
        for k in 0..kept {
            let proj = dot_loop(
                &u_cols[k * n..(k + 1) * n],
                &u_cols[kept * n..(kept + 1) * n],
            );
            for i in 0..n {
                u_cols[kept * n + i] -= proj * u_cols[k * n + i];
            }
        }
        let norm2 = dot_loop(
            &u_cols[kept * n..(kept + 1) * n],
            &u_cols[kept * n..(kept + 1) * n],
        );
        if !norm2.is_finite() || norm2 <= 1e-18 {
            continue;
        }
        let inv = 1.0 / norm2.sqrt();
        for i in 0..n {
            u_cols[kept * n + i] *= inv;
        }
        d_vals.push(lam);
        kept += 1;
    }
    if kept == 0 {
        return None;
    }
    u_cols.truncate(n * kept);
    Some(RandNysPrecond {
        n,
        rank: kept,
        u: u_cols,
        d: d_vals,
    })
}

#[inline]
fn apply_rand_nys_precond_vec(
    precond: &RandNysPrecond,
    lbd: f64,
    v: &[f64],
    out: &mut [f64],
    tmp_proj: &mut [f64],
) {
    let n = v.len();
    let rank = precond.rank;
    debug_assert_eq!(precond.n, n);
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(tmp_proj.len(), rank);

    out.copy_from_slice(v);
    for (k, tk) in tmp_proj.iter_mut().enumerate().take(rank) {
        let uk = &precond.u[k * n..(k + 1) * n];
        let dot_uv = dot_loop(uk, v);
        let denom = precond.d[k] + lbd;
        let scale = if denom.is_finite() && denom > 1e-14 {
            precond.d[k] / denom
        } else {
            0.0
        };
        *tk = dot_uv * scale;
    }
    for (k, tk) in tmp_proj.iter().enumerate().take(rank) {
        if *tk == 0.0 {
            continue;
        }
        let uk = &precond.u[k * n..(k + 1) * n];
        for i in 0..n {
            out[i] -= uk[i] * *tk;
        }
    }
}

#[inline]
fn apply_precond_block_packed(
    precond: Option<&RandNysPrecond>,
    lbd: f64,
    m_inv_diag: &[f64],
    v_block: &[f64], // column-major
    rhs: usize,
    out_block: &mut [f64],      // column-major
    tmp_proj_block: &mut [f64], // rhs * rank
) {
    let n = m_inv_diag.len();
    debug_assert_eq!(v_block.len(), rhs * n);
    debug_assert_eq!(out_block.len(), rhs * n);
    if let Some(pre) = precond {
        let rank = pre.rank;
        debug_assert_eq!(tmp_proj_block.len(), rhs * rank);
        for c in 0..rhs {
            let off = c * n;
            let tmp = &mut tmp_proj_block[c * rank..(c + 1) * rank];
            apply_rand_nys_precond_vec(
                pre,
                lbd,
                &v_block[off..off + n],
                &mut out_block[off..off + n],
                tmp,
            );
        }
    } else {
        for c in 0..rhs {
            let off = c * n;
            for i in 0..n {
                out_block[off + i] = m_inv_diag[i] * v_block[off + i];
            }
        }
    }
}

fn pcg_solve_block_k_plus_lambda_packed(
    op: &PackedGeno2Bit,
    lbd: f64,
    m_inv_diag: &[f64],
    precond: Option<&RandNysPrecond>,
    b_block: &[f64], // column-major (rhs, n)
    rhs: usize,
    x_block: &mut [f64],
    r_block: &mut [f64],
    z_block: &mut [f64],
    p_block: &mut [f64],
    ap_block: &mut [f64],
    norm_b: &mut [f64],
    rz_old: &mut [f64],
    tmp_proj_block: &mut [f64], // rhs * rank
    centered_row: &mut [f64],
    dots: &mut [f64],
    tol: f64,
    max_iter: usize,
) -> bool {
    let n = op.n;
    let size = rhs * n;
    debug_assert_eq!(m_inv_diag.len(), n);
    debug_assert_eq!(b_block.len(), size);
    debug_assert_eq!(x_block.len(), size);
    debug_assert_eq!(r_block.len(), size);
    debug_assert_eq!(z_block.len(), size);
    debug_assert_eq!(p_block.len(), size);
    debug_assert_eq!(ap_block.len(), size);
    debug_assert!(norm_b.len() >= rhs);
    debug_assert!(rz_old.len() >= rhs);

    x_block.fill(0.0);
    r_block.copy_from_slice(b_block);
    apply_precond_block_packed(
        precond,
        lbd,
        m_inv_diag,
        r_block,
        rhs,
        z_block,
        tmp_proj_block,
    );
    p_block.copy_from_slice(z_block);

    let tol = if tol.is_finite() && tol > 0.0 {
        tol
    } else {
        1e-6
    };
    let max_iter = max_iter.max(1);

    let norm_b = &mut norm_b[..rhs];
    let rz_old = &mut rz_old[..rhs];
    for c in 0..rhs {
        let bc = &b_block[c * n..(c + 1) * n];
        let rc = &r_block[c * n..(c + 1) * n];
        let zc = &z_block[c * n..(c + 1) * n];
        let nb2 = dot_loop(bc, bc);
        norm_b[c] = if nb2.is_finite() && nb2 > 0.0 {
            nb2.sqrt()
        } else {
            1.0
        };
        rz_old[c] = dot_loop(rc, zc);
    }

    for _ in 0..max_iter {
        matmat_k_plus_lambda_packed_2bit(op, lbd, p_block, rhs, ap_block, centered_row, dots);

        let mut all_done = true;
        for c in 0..rhs {
            let off = c * n;
            let pc = &p_block[off..off + n];
            let apc = &ap_block[off..off + n];
            let denom = dot_loop(pc, apc);
            if !denom.is_finite() || denom.abs() < 1e-30 {
                continue;
            }
            let alpha = rz_old[c] / denom;
            if !alpha.is_finite() {
                continue;
            }
            for i in 0..n {
                x_block[off + i] += alpha * p_block[off + i];
                r_block[off + i] -= alpha * ap_block[off + i];
            }
            let rel = dot_loop(&r_block[off..off + n], &r_block[off..off + n]).sqrt() / norm_b[c];
            if !rel.is_finite() || rel > tol {
                all_done = false;
            }
        }
        if all_done {
            return true;
        }

        apply_precond_block_packed(
            precond,
            lbd,
            m_inv_diag,
            r_block,
            rhs,
            z_block,
            tmp_proj_block,
        );
        for c in 0..rhs {
            let off = c * n;
            let rz_new = dot_loop(&r_block[off..off + n], &z_block[off..off + n]);
            if !rz_new.is_finite() || !rz_old[c].is_finite() || rz_old[c].abs() < 1e-30 {
                continue;
            }
            let beta = rz_new / rz_old[c];
            if !beta.is_finite() {
                continue;
            }
            for i in 0..n {
                p_block[off + i] = z_block[off + i] + beta * p_block[off + i];
            }
            rz_old[c] = rz_new;
        }
    }
    false
}

fn pcg_solve_block_k_plus_lambda_packed_f32_rhs(
    op: &PackedGeno2Bit,
    lbd: f64,
    m_inv_diag: &[f64],
    precond: Option<&RandNysPrecond>,
    b_block: &[f32], // row-major (rhs, n)
    rhs: usize,
    x_block: &mut [f64],
    r_block: &mut [f64],
    z_block: &mut [f64],
    p_block: &mut [f64],
    ap_block: &mut [f64],
    norm_b: &mut [f64],
    rz_old: &mut [f64],
    tmp_proj_block: &mut [f64], // rhs * rank
    centered_row: &mut [f64],
    dots: &mut [f64],
    tol: f64,
    max_iter: usize,
) -> bool {
    let n = op.n;
    let size = rhs * n;
    debug_assert_eq!(m_inv_diag.len(), n);
    debug_assert_eq!(b_block.len(), size);
    debug_assert_eq!(x_block.len(), size);
    debug_assert_eq!(r_block.len(), size);
    debug_assert_eq!(z_block.len(), size);
    debug_assert_eq!(p_block.len(), size);
    debug_assert_eq!(ap_block.len(), size);
    debug_assert!(norm_b.len() >= rhs);
    debug_assert!(rz_old.len() >= rhs);

    x_block.fill(0.0);
    for i in 0..size {
        r_block[i] = b_block[i] as f64;
    }
    apply_precond_block_packed(
        precond,
        lbd,
        m_inv_diag,
        r_block,
        rhs,
        z_block,
        tmp_proj_block,
    );
    p_block.copy_from_slice(z_block);

    let tol = if tol.is_finite() && tol > 0.0 {
        tol
    } else {
        1e-6
    };
    let max_iter = max_iter.max(1);

    let norm_b = &mut norm_b[..rhs];
    let rz_old = &mut rz_old[..rhs];
    for c in 0..rhs {
        let bc = &b_block[c * n..(c + 1) * n];
        let rc = &r_block[c * n..(c + 1) * n];
        let zc = &z_block[c * n..(c + 1) * n];
        let mut nb2 = 0.0_f64;
        for &v in bc.iter() {
            let vf = v as f64;
            nb2 += vf * vf;
        }
        norm_b[c] = if nb2.is_finite() && nb2 > 0.0 {
            nb2.sqrt()
        } else {
            1.0
        };
        rz_old[c] = dot_loop(rc, zc);
    }

    for _ in 0..max_iter {
        matmat_k_plus_lambda_packed_2bit(op, lbd, p_block, rhs, ap_block, centered_row, dots);

        let mut all_done = true;
        for c in 0..rhs {
            let off = c * n;
            let pc = &p_block[off..off + n];
            let apc = &ap_block[off..off + n];
            let denom = dot_loop(pc, apc);
            if !denom.is_finite() || denom.abs() < 1e-30 {
                continue;
            }
            let alpha = rz_old[c] / denom;
            if !alpha.is_finite() {
                continue;
            }
            for i in 0..n {
                x_block[off + i] += alpha * p_block[off + i];
                r_block[off + i] -= alpha * ap_block[off + i];
            }
            let rel = dot_loop(&r_block[off..off + n], &r_block[off..off + n]).sqrt() / norm_b[c];
            if !rel.is_finite() || rel > tol {
                all_done = false;
            }
        }
        if all_done {
            return true;
        }

        apply_precond_block_packed(
            precond,
            lbd,
            m_inv_diag,
            r_block,
            rhs,
            z_block,
            tmp_proj_block,
        );
        for c in 0..rhs {
            let off = c * n;
            let rz_new = dot_loop(&r_block[off..off + n], &z_block[off..off + n]);
            if !rz_new.is_finite() || !rz_old[c].is_finite() || rz_old[c].abs() < 1e-30 {
                continue;
            }
            let beta = rz_new / rz_old[c];
            if !beta.is_finite() {
                continue;
            }
            for i in 0..n {
                p_block[off + i] = z_block[off + i] + beta * p_block[off + i];
            }
            rz_old[c] = rz_new;
        }
    }
    false
}

struct MatrixFreeNullState {
    vinv_y: Vec<f64>,
    py: Vec<f64>,
    a_chol: Vec<f64>, // Cholesky factor L of A
    a_inv_b: Vec<f64>,
    b_aib: f64,
    y_vinv_y: f64,
    sigma2_null: f64,
    df: i32,
}

struct MatrixFreeNullScratch {
    x_blk: Vec<f64>,
    r_blk: Vec<f64>,
    z_blk: Vec<f64>,
    p_blk: Vec<f64>,
    ap_blk: Vec<f64>,
    norm_b: Vec<f64>,
    rz_old: Vec<f64>,
    tmp_proj: Vec<f64>,
    centered_row: Vec<f64>,
    dots: Vec<f64>,
}

impl MatrixFreeNullScratch {
    fn new(n: usize, rhs_cap: usize, precond_rank: usize) -> Self {
        Self {
            x_blk: vec![0.0; n * rhs_cap],
            r_blk: vec![0.0; n * rhs_cap],
            z_blk: vec![0.0; n * rhs_cap],
            p_blk: vec![0.0; n * rhs_cap],
            ap_blk: vec![0.0; n * rhs_cap],
            norm_b: vec![0.0; rhs_cap],
            rz_old: vec![0.0; rhs_cap],
            tmp_proj: vec![0.0; rhs_cap * precond_rank],
            centered_row: vec![0.0; n],
            dots: vec![0.0; rhs_cap],
        }
    }
}

struct MatrixFreeScanScratch {
    c: Vec<f64>,
    a_inv_c: Vec<f64>,
    x_blk: Vec<f64>,
    r_blk: Vec<f64>,
    z_blk: Vec<f64>,
    p_blk: Vec<f64>,
    ap_blk: Vec<f64>,
    norm_b: Vec<f64>,
    rz_old: Vec<f64>,
    tmp_proj: Vec<f64>,
    centered_row: Vec<f64>,
    dots: Vec<f64>,
}

impl MatrixFreeScanScratch {
    fn new(n: usize, p_cov: usize, rhs_cap: usize, precond_rank: usize) -> Self {
        Self {
            c: vec![0.0; p_cov],
            a_inv_c: vec![0.0; p_cov],
            x_blk: vec![0.0; n * rhs_cap],
            r_blk: vec![0.0; n * rhs_cap],
            z_blk: vec![0.0; n * rhs_cap],
            p_blk: vec![0.0; n * rhs_cap],
            ap_blk: vec![0.0; n * rhs_cap],
            norm_b: vec![0.0; rhs_cap],
            rz_old: vec![0.0; rhs_cap],
            tmp_proj: vec![0.0; rhs_cap * precond_rank],
            centered_row: vec![0.0; n],
            dots: vec![0.0; rhs_cap],
        }
    }
}

struct MatrixFreeScoreVrScratch {
    c: Vec<f64>,
    a_inv_c: Vec<f64>,
}

impl MatrixFreeScoreVrScratch {
    fn new(p_cov: usize) -> Self {
        Self {
            c: vec![0.0; p_cov],
            a_inv_c: vec![0.0; p_cov],
        }
    }
}

struct ScoreVrCalibScratch {
    x_blk: Vec<f64>,
    r_blk: Vec<f64>,
    z_blk: Vec<f64>,
    p_blk: Vec<f64>,
    ap_blk: Vec<f64>,
    norm_b: Vec<f64>,
    rz_old: Vec<f64>,
    tmp_proj: Vec<f64>,
    centered_row: Vec<f64>,
    dots: Vec<f64>,
    c: Vec<f64>,
    a_inv_c: Vec<f64>,
}

impl ScoreVrCalibScratch {
    fn new(n: usize, p_cov: usize, precond_rank: usize) -> Self {
        Self {
            x_blk: vec![0.0; n],
            r_blk: vec![0.0; n],
            z_blk: vec![0.0; n],
            p_blk: vec![0.0; n],
            ap_blk: vec![0.0; n],
            norm_b: vec![0.0; 1],
            rz_old: vec![0.0; 1],
            tmp_proj: vec![0.0; precond_rank],
            centered_row: vec![0.0; n],
            dots: vec![0.0; 1],
            c: vec![0.0; p_cov],
            a_inv_c: vec![0.0; p_cov],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatrixFreeScanMode {
    Exact,
    ScoreVr,
}

impl MatrixFreeScanMode {
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "exact" | "pcg_exact" | "pcg" => Some(Self::Exact),
            "score_vr" | "scorevr" | "score-vr" => Some(Self::ScoreVr),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ScoreVr => "score_vr",
        }
    }
}

#[pyclass(name = "RustPCGMatrixFreeState")]
pub struct RustPcgMatrixFreeState {
    op: PackedGeno2Bit,
    xcov: Vec<f64>, // row-major (n, p)
    y: Vec<f64>,    // len n
    n: usize,
    p_cov: usize,
    lbd: f64,
    pcg_tol: f64,
    pcg_max_iter: usize,
    block_size: usize,
    scan_rhs_cap: usize,
    m_inv_diag: Vec<f64>,
    precond: Option<RandNysPrecond>,
    precond_kind: String,
    precond_rank: usize,
    scan_mode: MatrixFreeScanMode,
    scan_mode_name: String,
    score_vr_calib_snps: usize,
    score_vr_ratio: Option<f64>,
    mean_diag_k: f64,
    null_state: Option<MatrixFreeNullState>,
}

impl RustPcgMatrixFreeState {
    fn estimate_score_vr_ratio(&self, g_flat: &[f32], m_scan: usize, ns: &MatrixFreeNullState) -> f64 {
        if m_scan == 0 {
            return 1.0;
        }
        let n = self.n;
        let p_cov = self.p_cov;
        let precond_rank = self.precond_rank;
        let calib_idx = choose_score_vr_calibration_indices(
            g_flat,
            m_scan,
            n,
            self.score_vr_calib_snps,
        );
        if calib_idx.is_empty() {
            return 1.0;
        }

        let mut ratios = calib_idx
            .par_iter()
            .map_init(
                || ScoreVrCalibScratch::new(n, p_cov, precond_rank),
                |scr, &idx| {
                    let row_off = idx * n;
                    let b_col = &g_flat[row_off..row_off + n];
                    let _ = pcg_solve_block_k_plus_lambda_packed_f32_rhs(
                        &self.op,
                        self.lbd,
                        &self.m_inv_diag,
                        self.precond.as_ref(),
                        b_col,
                        1,
                        &mut scr.x_blk,
                        &mut scr.r_blk,
                        &mut scr.z_blk,
                        &mut scr.p_blk,
                        &mut scr.ap_blk,
                        &mut scr.norm_b,
                        &mut scr.rz_old,
                        &mut scr.tmp_proj,
                        &mut scr.centered_row,
                        &mut scr.dots,
                        self.pcg_tol,
                        self.pcg_max_iter,
                    );

                    scr.c.fill(0.0);
                    let mut d_exact = 0.0_f64;
                    let mut d_diag = 0.0_f64;
                    for i in 0..n {
                        let gi = b_col[i] as f64;
                        let vi_g_exact = scr.x_blk[i];
                        let vi_g_diag = self.m_inv_diag[i] * gi;
                        d_exact += gi * vi_g_exact;
                        d_diag += gi * vi_g_diag;
                        let base = i * p_cov;
                        for rj in 0..p_cov {
                            scr.c[rj] += self.xcov[base + rj] * vi_g_exact;
                        }
                    }
                    cholesky_solve_into(&ns.a_chol, p_cov, &scr.c, &mut scr.a_inv_c);
                    let schur_exact = d_exact - dot_loop(&scr.c, &scr.a_inv_c);

                    scr.c.fill(0.0);
                    for i in 0..n {
                        let gi = b_col[i] as f64;
                        let vi_g_diag = self.m_inv_diag[i] * gi;
                        let base = i * p_cov;
                        for rj in 0..p_cov {
                            scr.c[rj] += self.xcov[base + rj] * vi_g_diag;
                        }
                    }
                    cholesky_solve_into(&ns.a_chol, p_cov, &scr.c, &mut scr.a_inv_c);
                    let schur_diag = d_diag - dot_loop(&scr.c, &scr.a_inv_c);

                    if schur_exact.is_finite()
                        && schur_diag.is_finite()
                        && schur_exact > 1e-12
                        && schur_diag > 1e-12
                    {
                        Some(schur_exact / schur_diag)
                    } else {
                        None
                    }
                },
            )
            .filter_map(|v| v)
            .collect::<Vec<f64>>();

        if ratios.is_empty() {
            return 1.0;
        }
        ratios.sort_by(|a, b| a.total_cmp(b));
        let mid = ratios.len() / 2;
        let mut vr = if ratios.len() % 2 == 1 {
            ratios[mid]
        } else {
            0.5 * (ratios[mid - 1] + ratios[mid])
        };
        if !vr.is_finite() || vr <= 0.0 {
            vr = 1.0;
        }
        vr.clamp(0.1, 10.0)
    }

    fn fit_null_internal(&mut self) -> PyResult<()> {
        let n = self.n;
        let p_cov = self.p_cov;
        let rhs0 = p_cov + 1;
        let precond_rank = self.precond.as_ref().map(|p| p.rank).unwrap_or(0);

        let mut b0_used = vec![0.0_f64; rhs0 * n];
        for c in 0..p_cov {
            for i in 0..n {
                b0_used[c * n + i] = self.xcov[i * p_cov + c];
            }
        }
        for i in 0..n {
            b0_used[p_cov * n + i] = self.y[i];
        }

        let x_cols = (0..rhs0)
            .into_par_iter()
            .map(|c| {
                let mut scratch = MatrixFreeNullScratch::new(n, 1, precond_rank);
                let b_col = &b0_used[c * n..(c + 1) * n];
                let x_used = &mut scratch.x_blk[..n];
                let r_used = &mut scratch.r_blk[..n];
                let z_used = &mut scratch.z_blk[..n];
                let p_used = &mut scratch.p_blk[..n];
                let ap_used = &mut scratch.ap_blk[..n];
                let norm_b = &mut scratch.norm_b[..1];
                let rz_old = &mut scratch.rz_old[..1];
                let tmp_proj = &mut scratch.tmp_proj[..precond_rank];
                let dots = &mut scratch.dots[..1];
                let centered = &mut scratch.centered_row[..];

                let _ = pcg_solve_block_k_plus_lambda_packed(
                    &self.op,
                    self.lbd,
                    &self.m_inv_diag,
                    self.precond.as_ref(),
                    b_col,
                    1,
                    x_used,
                    r_used,
                    z_used,
                    p_used,
                    ap_used,
                    norm_b,
                    rz_old,
                    tmp_proj,
                    centered,
                    dots,
                    self.pcg_tol,
                    self.pcg_max_iter,
                );
                x_used.to_vec()
            })
            .collect::<Vec<Vec<f64>>>();

        let mut x0_used = vec![0.0_f64; rhs0 * n];
        for c in 0..rhs0 {
            let dst = &mut x0_used[c * n..(c + 1) * n];
            dst.copy_from_slice(&x_cols[c]);
        }

        let vinv_y = x0_used[p_cov * n..(p_cov + 1) * n].to_vec();
        let mut a = vec![0.0_f64; p_cov * p_cov];
        let mut b0 = vec![0.0_f64; p_cov];
        let mut y_vinv_y = 0.0_f64;
        for i in 0..n {
            let yi = self.y[i];
            let viy = vinv_y[i];
            y_vinv_y += yi * viy;
            let base = i * p_cov;
            for rj in 0..p_cov {
                let xir = self.xcov[base + rj];
                b0[rj] += xir * viy;
                for cj in 0..=rj {
                    let vix = x0_used[cj * n + i];
                    a[rj * p_cov + cj] += xir * vix;
                }
            }
        }
        for rj in 0..p_cov {
            a[rj * p_cov + rj] += 1e-6;
            for cj in 0..rj {
                let v = a[rj * p_cov + cj];
                a[cj * p_cov + rj] = v;
            }
        }
        if cholesky_inplace(&mut a, p_cov).is_none() {
            return Err(PyRuntimeError::new_err(
                "X'V^-1X not SPD in matrix-free fit_null",
            ));
        }

        let mut a_inv_b = vec![0.0_f64; p_cov];
        cholesky_solve_into(&a, p_cov, &b0, &mut a_inv_b);
        let b_aib = dot_loop(&b0, &a_inv_b);
        let df = (n as i32) - (p_cov as i32) - 1;
        if df <= 0 {
            return Err(PyRuntimeError::new_err("df <= 0"));
        }
        let rwr0 = (y_vinv_y - b_aib).max(0.0);
        let sigma2_null = (rwr0 / (df as f64)).max(1e-12);
        let mut py = vec![0.0_f64; n];
        for i in 0..n {
            let mut v = vinv_y[i];
            for rj in 0..p_cov {
                v -= x0_used[rj * n + i] * a_inv_b[rj];
            }
            py[i] = v;
        }

        self.null_state = Some(MatrixFreeNullState {
            vinv_y,
            py,
            a_chol: a,
            a_inv_b,
            b_aib,
            y_vinv_y,
            sigma2_null,
            df,
        });
        self.score_vr_ratio = None;
        Ok(())
    }
}

#[pymethods]
impl RustPcgMatrixFreeState {
    #[new]
    #[pyo3(signature = (
        kinship_geno,
        xcov,
        y,
        log10_lbd,
        pcg_tol=1e-4,
        pcg_max_iter=200,
        block_size=0,
        precond_kind="rand_nys",
        precond_rank=0,
        precond_oversample=2,
        precond_seed=42,
        scan_rhs_cap=0,
        scan_mode="exact",
        score_vr_calib_snps=128
    ))]
    fn new(
        kinship_geno: PyReadonlyArray2<'_, f32>,
        xcov: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        log10_lbd: f64,
        pcg_tol: f64,
        pcg_max_iter: usize,
        block_size: usize,
        precond_kind: &str,
        precond_rank: usize,
        precond_oversample: usize,
        precond_seed: u64,
        scan_rhs_cap: usize,
        scan_mode: &str,
        score_vr_calib_snps: usize,
    ) -> PyResult<Self> {
        let y_slice = y.as_slice()?;
        let n = y_slice.len();
        let y_vec = y_slice.to_vec();

        let x_arr = xcov.as_array();
        let (xn, p_cov) = (x_arr.shape()[0], x_arr.shape()[1]);
        if xn != n {
            return Err(PyRuntimeError::new_err("xcov.n_rows must equal len(y)"));
        }
        if n <= p_cov + 1 {
            return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
        }
        let x_flat: Cow<[f64]> = match xcov.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
        };

        let g_arr = kinship_geno.as_array();
        let (m, gn) = (g_arr.shape()[0], g_arr.shape()[1]);
        if gn != n {
            return Err(PyRuntimeError::new_err(
                "kinship_geno must be (m, n) with n=len(y)",
            ));
        }
        let g_flat: Cow<[f32]> = match kinship_geno.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(g_arr.iter().copied().collect()),
        };

        let lbd = 10.0_f64.powf(log10_lbd);
        if !lbd.is_finite() || lbd <= 0.0 {
            return Err(PyRuntimeError::new_err("invalid log10_lbd"));
        }
        let op = pack_snp_major_genotype_2bit(&g_flat, m, n)?;
        let mut m_inv_diag = vec![0.0_f64; n];
        for i in 0..n {
            m_inv_diag[i] = 1.0 / (op.diag_k[i] + lbd).max(1e-8);
        }

        let eff_threads = rayon::current_num_threads().max(1);
        let mut precond_kind_resolved = "diag".to_string();
        let mut precond_rank_resolved = 0usize;
        let precond = match precond_kind.to_ascii_lowercase().as_str() {
            "diag" => None,
            "rand_nys" | "randnys" | "nys" | "auto" => {
                let rank_req = if precond_rank == 0 {
                    auto_precond_rank(n, m, eff_threads)
                } else {
                    precond_rank.min(n).max(1)
                };
                let oversample = precond_oversample.max(1).min(8);
                let built = build_rand_nys_precond(&op, rank_req, oversample, precond_seed);
                if let Some(ref p) = built {
                    precond_kind_resolved = "rand_nys".to_string();
                    precond_rank_resolved = p.rank;
                }
                built
            }
            _ => {
                return Err(PyRuntimeError::new_err(
                    "precond_kind must be one of: rand_nys, diag, auto",
                ));
            }
        };
        let resolved_block_size = if block_size == 0 {
            auto_block_size(n, m, eff_threads)
        } else {
            block_size.max(1).min(32)
        };
        let resolved_scan_rhs_cap = if scan_rhs_cap == 0 {
            auto_scan_rhs_cap(resolved_block_size, n, precond_rank_resolved, eff_threads)
        } else {
            scan_rhs_cap.max(1).min(resolved_block_size)
        };
        let resolved_scan_mode = MatrixFreeScanMode::parse(scan_mode).ok_or_else(|| {
            PyRuntimeError::new_err("scan_mode must be one of: exact, score_vr")
        })?;
        let resolved_score_vr_calib_snps = score_vr_calib_snps.max(8).min(2048);

        Ok(Self {
            mean_diag_k: op.mean_diag_k,
            op,
            xcov: x_flat.into_owned(),
            y: y_vec,
            n,
            p_cov,
            lbd,
            pcg_tol,
            pcg_max_iter,
            block_size: resolved_block_size,
            scan_rhs_cap: resolved_scan_rhs_cap,
            m_inv_diag,
            precond,
            precond_kind: precond_kind_resolved,
            precond_rank: precond_rank_resolved,
            scan_mode: resolved_scan_mode,
            scan_mode_name: resolved_scan_mode.as_str().to_string(),
            score_vr_calib_snps: resolved_score_vr_calib_snps,
            score_vr_ratio: None,
            null_state: None,
        })
    }

    fn fit_null(&mut self) -> PyResult<()> {
        self.fit_null_internal()
    }

    fn set_scan_mode(&mut self, scan_mode: &str) -> PyResult<()> {
        let mode = MatrixFreeScanMode::parse(scan_mode).ok_or_else(|| {
            PyRuntimeError::new_err("scan_mode must be one of: exact, score_vr")
        })?;
        self.scan_mode = mode;
        self.scan_mode_name = mode.as_str().to_string();
        Ok(())
    }

    #[pyo3(signature = (snp_chunk, threads=0))]
    fn scan_chunk<'py>(
        &mut self,
        py: Python<'py>,
        snp_chunk: PyReadonlyArray2<'py, f32>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.null_state.is_none() {
            self.fit_null_internal()?;
        }
        let ns = self
            .null_state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("null model not initialized"))?;

        let g_arr = snp_chunk.as_array();
        if g_arr.shape()[1] != self.n {
            return Err(PyRuntimeError::new_err("snp_chunk must be (m, n)"));
        }
        let g_flat: Cow<[f32]> = match snp_chunk.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(g_arr.iter().copied().collect()),
        };

        let m_scan = g_arr.shape()[0];
        let out = PyArray2::<f64>::zeros(py, [m_scan, 3], false).into_bound();
        let out_slice: &mut [f64] = unsafe {
            out.as_slice_mut()
                .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
        };

        let pool = get_cached_pool(threads)?;
        let n = self.n;
        let p_cov = self.p_cov;
        let x_flat = &self.xcov;
        let op = &self.op;
        let lbd = self.lbd;
        let m_inv_diag = &self.m_inv_diag;
        let precond = self.precond.as_ref();
        let precond_rank = self.precond_rank;
        let pcg_tol = self.pcg_tol;
        let pcg_max_iter = self.pcg_max_iter;
        let a_chol = &ns.a_chol;
        let a_inv_b = &ns.a_inv_b;
        let vinv_y = &ns.vinv_y;
        let py_vec = &ns.py;
        let b_aib = ns.b_aib;
        let y_vinv_y_const = ns.y_vinv_y;
        let sigma2_null = ns.sigma2_null;
        let df = ns.df;
        let scan_mode = self.scan_mode;

        if scan_mode == MatrixFreeScanMode::ScoreVr && self.score_vr_ratio.is_none() {
            let ratio = py.detach(|| self.estimate_score_vr_ratio(&g_flat, m_scan, ns));
            self.score_vr_ratio = Some(ratio);
        }
        let score_vr_ratio = self.score_vr_ratio.unwrap_or(1.0);

        py.detach(|| {
            let block_size = self.block_size.min(m_scan.max(1));
            let rhs_cap = self.scan_rhs_cap.min(block_size).max(1);
            if scan_mode == MatrixFreeScanMode::ScoreVr {
                let mut run = || {
                    out_slice
                        .par_chunks_mut(block_size * 3)
                        .enumerate()
                        .for_each_init(
                            || MatrixFreeScoreVrScratch::new(p_cov),
                            |scr, (blk_idx, out_blk)| {
                                let start = blk_idx * block_size;
                                let bs = usize::min(block_size, m_scan - start);
                                for local in 0..bs {
                                    let row_idx = start + local;
                                    let b_col = &g_flat[row_idx * n..(row_idx + 1) * n];
                                    scr.c.fill(0.0);
                                    let mut d_diag = 0.0_f64;
                                    let mut u = 0.0_f64;
                                    for i in 0..n {
                                        let gi = b_col[i] as f64;
                                        let vi_g_diag = m_inv_diag[i] * gi;
                                        d_diag += gi * vi_g_diag;
                                        u += gi * py_vec[i];
                                        let base = i * p_cov;
                                        for rj in 0..p_cov {
                                            scr.c[rj] += x_flat[base + rj] * vi_g_diag;
                                        }
                                    }
                                    cholesky_solve_into(a_chol, p_cov, &scr.c, &mut scr.a_inv_c);
                                    let schur_diag = d_diag - dot_loop(&scr.c, &scr.a_inv_c);
                                    let schur = score_vr_ratio * schur_diag;

                                    let out_row = &mut out_blk[local * 3..local * 3 + 3];
                                    if !schur.is_finite() || schur <= 1e-12 || !u.is_finite() {
                                        out_row[0] = f64::NAN;
                                        out_row[1] = f64::NAN;
                                        out_row[2] = f64::NAN;
                                        continue;
                                    }

                                    let beta_g = u / schur;
                                    let se_g = (sigma2_null / schur).sqrt();
                                    let pval =
                                        if beta_g.is_finite() && se_g.is_finite() && se_g > 0.0 {
                                            (2.0 * normal_sf((beta_g / se_g).abs()))
                                                .clamp(f64::MIN_POSITIVE, 1.0)
                                        } else {
                                            1.0
                                        };
                                    out_row[0] = beta_g;
                                    out_row[1] = se_g;
                                    out_row[2] = pval;
                                }
                            },
                        );
                };
                if let Some(tp) = &pool {
                    tp.install(run);
                } else {
                    run();
                }
            } else {
                let mut run = || {
                    out_slice
                        .par_chunks_mut(block_size * 3)
                        .enumerate()
                        .for_each_init(
                            || MatrixFreeScanScratch::new(n, p_cov, rhs_cap, precond_rank),
                            |scr, (blk_idx, out_blk)| {
                                let start = blk_idx * block_size;
                                let bs = usize::min(block_size, m_scan - start);
                                let mut local = 0usize;
                                while local < bs {
                                    let sub_bs = usize::min(rhs_cap, bs - local);
                                    let global_start = start + local;

                                    let x_used = &mut scr.x_blk[..sub_bs * n];
                                    let r_used = &mut scr.r_blk[..sub_bs * n];
                                    let z_used = &mut scr.z_blk[..sub_bs * n];
                                    let p_used = &mut scr.p_blk[..sub_bs * n];
                                    let ap_used = &mut scr.ap_blk[..sub_bs * n];
                                    let norm_b_used = &mut scr.norm_b[..sub_bs];
                                    let rz_old_used = &mut scr.rz_old[..sub_bs];
                                    let tmp_proj_used = &mut scr.tmp_proj[..sub_bs * precond_rank];
                                    let dots_used = &mut scr.dots[..sub_bs];
                                    let row_off = global_start * n;
                                    let b_sub = &g_flat[row_off..row_off + sub_bs * n];

                                    let _ = pcg_solve_block_k_plus_lambda_packed_f32_rhs(
                                        op,
                                        lbd,
                                        m_inv_diag,
                                        precond,
                                        b_sub,
                                        sub_bs,
                                        x_used,
                                        r_used,
                                        z_used,
                                        p_used,
                                        ap_used,
                                        norm_b_used,
                                        rz_old_used,
                                        tmp_proj_used,
                                        &mut scr.centered_row,
                                        dots_used,
                                        pcg_tol,
                                        pcg_max_iter,
                                    );

                                    for c in 0..sub_bs {
                                        let b_col = &b_sub[c * n..(c + 1) * n];
                                        let x_col = &x_used[c * n..(c + 1) * n];
                                        scr.c.fill(0.0);
                                        let mut d = 0.0_f64;
                                        let mut e = 0.0_f64;
                                        for i in 0..n {
                                            let gi = b_col[i] as f64;
                                            let vi_g = x_col[i];
                                            d += gi * vi_g;
                                            e += gi * vinv_y[i];
                                            let base = i * p_cov;
                                            for rj in 0..p_cov {
                                                scr.c[rj] += x_flat[base + rj] * vi_g;
                                            }
                                        }

                                        cholesky_solve_into(a_chol, p_cov, &scr.c, &mut scr.a_inv_c);
                                        let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                                        let schur = d - ct_aic;
                                        let out_row =
                                            &mut out_blk[(local + c) * 3..(local + c) * 3 + 3];
                                        if !schur.is_finite() || schur <= 1e-12 {
                                            out_row[0] = f64::NAN;
                                            out_row[1] = f64::NAN;
                                            out_row[2] = f64::NAN;
                                            continue;
                                        }

                                        let ct_aib = dot_loop(&scr.c, a_inv_b);
                                        let num = e - ct_aib;
                                        let beta_g = num / schur;
                                        let q = b_aib + (num * num) / schur;
                                        let rwr = (y_vinv_y_const - q).max(0.0);
                                        let sigma2 = rwr / (df as f64);
                                        let se_g = (sigma2 / schur).sqrt();
                                        let pval =
                                            if beta_g.is_finite() && se_g.is_finite() && se_g > 0.0 {
                                                (2.0 * normal_sf((beta_g / se_g).abs()))
                                                    .clamp(f64::MIN_POSITIVE, 1.0)
                                            } else {
                                                1.0
                                            };
                                        out_row[0] = beta_g;
                                        out_row[1] = se_g;
                                        out_row[2] = pval;
                                    }
                                    local += sub_bs;
                                }
                            },
                        );
                };
                if let Some(tp) = &pool {
                    tp.install(run);
                } else {
                    run();
                }
            }
        });

        Ok(out)
    }

    #[getter]
    fn mean_diag(&self) -> f64 {
        self.mean_diag_k
    }

    #[getter]
    fn pve(&self) -> f64 {
        if (self.mean_diag_k + self.lbd) > 0.0 {
            self.mean_diag_k / (self.mean_diag_k + self.lbd)
        } else {
            f64::NAN
        }
    }

    #[getter]
    fn lbd(&self) -> f64 {
        self.lbd
    }

    #[getter]
    fn fitted(&self) -> bool {
        self.null_state.is_some()
    }

    #[getter]
    fn block_size(&self) -> usize {
        self.block_size
    }

    #[getter]
    fn scan_rhs_cap(&self) -> usize {
        self.scan_rhs_cap
    }

    #[getter]
    fn precond_kind(&self) -> String {
        self.precond_kind.clone()
    }

    #[getter]
    fn precond_rank(&self) -> usize {
        self.precond_rank
    }

    #[getter]
    fn scan_mode(&self) -> String {
        self.scan_mode_name.clone()
    }

    #[getter]
    fn score_vr_ratio(&self) -> Option<f64> {
        self.score_vr_ratio
    }
}

struct AssocScratch {
    c: Vec<f64>,       // len p
    a_inv_c: Vec<f64>, // len p
}
impl AssocScratch {
    fn new(p: usize) -> Self {
        Self {
            c: vec![0.0; p],
            a_inv_c: vec![0.0; p],
        }
    }
    #[inline]
    fn reset(&mut self) {
        self.c.fill(0.0);
    }
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd, g_rot_chunk, threads=0, nullml=None))]
pub fn lmm_assoc_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    g_rot_chunk: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let g_arr = g_rot_chunk.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("g_rot_chunk must be (m, n)"));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let m = g_arr.shape()[0];
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound(); // beta, se, p, (plrt)
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    // Use contiguous slice directly (no copy)
    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let p = p_cov;

    // Build weights W = V^{-1} = 1/(s + lbd) (store as f32 to reduce bandwidth)
    let mut w = vec![0.0_f32; n];
    let mut log_det_v = 0.0_f64;
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return Err(PyRuntimeError::new_err("non-positive s[i]+lbd"));
        }
        w[i] = (1.0 / vv) as f32;
        log_det_v += vv.ln();
    }

    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    // Precompute A = X'WX, b = X'Wy, yWy
    let mut a = vec![0.0_f64; p * p];
    let mut b = vec![0.0_f64; p];
    let mut ywy = 0.0_f64;

    for i in 0..n {
        let wi = w[i] as f64;
        let yi = y[i];
        ywy += wi * yi * yi;

        let base = i * p;
        for r in 0..p {
            let xir = xcov_slice[base + r];
            b[r] += wi * xir * yi;
            for c in 0..=r {
                let xic = xcov_slice[base + c];
                a[r * p + c] += wi * xir * xic;
            }
        }
    }

    // symmetrize + ridge
    let ridge = 1e-6;
    for r in 0..p {
        a[r * p + r] += ridge;
        for c in 0..r {
            let vrc = a[r * p + c];
            a[c * p + r] = vrc;
        }
    }

    // Cholesky(A) in-place; now a stores L
    if cholesky_inplace(&mut a, p).is_none() {
        return Err(PyRuntimeError::new_err("X'WX not SPD"));
    }

    // Solve A^{-1} b once (no-alloc version into tmp)
    let mut a_inv_b = vec![0.0_f64; p];
    cholesky_solve_into(&a, p, &b, &mut a_inv_b);

    // b'A^{-1}b is constant
    let b_aib = dot_loop(&b, &a_inv_b);

    let df = (n as i32) - (p as i32) - 1;
    if df <= 0 {
        return Err(PyRuntimeError::new_err("df <= 0"));
    }

    // Thread pool
    let pool = get_cached_pool(threads)?;

    py.detach(|| {
        let mut run = || {
            out_slice
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || AssocScratch::new(p),
                    |scr, (idx, out_row)| {
                        scr.reset();

                        // c = X'Wg , d = g'Wg, e = g'Wy
                        let mut d = 0.0_f64;
                        let mut e = 0.0_f64;

                        for i in 0..n {
                            let gi = g_arr[(idx, i)] as f64;
                            let wi = w[i] as f64;
                            let yi = y[i];
                            let base = i * p;

                            let wgi = wi * gi;
                            d += wgi * gi;
                            e += wgi * yi;

                            // c[r] += wgi * xcov[i, r]
                            for r in 0..p {
                                scr.c[r] += wgi * xcov_slice[base + r];
                            }
                        }

                        // a_inv_c = A^{-1} c (no alloc)
                        cholesky_solve_into(&a, p, &scr.c, &mut scr.a_inv_c);

                        // ct_aic = c' A^{-1} c
                        let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                        let schur = d - ct_aic;

                        if schur <= 1e-12 || !schur.is_finite() {
                            out_row[0] = f64::NAN;
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        // ct_aib = c' A^{-1} b  —— 这里按你要求：一次循环（不调用 dot）
                        let mut ct_aib = 0.0_f64;
                        for r in 0..p {
                            ct_aib += scr.c[r] * a_inv_b[r];
                        }

                        let num = e - ct_aib;
                        let beta_g = num / schur;

                        // rWr = yWy - [ b'A^{-1}b + (e - c'A^{-1}b)^2 / schur ]
                        let q = b_aib + (num * num) / schur;
                        let rwr = (ywy - q).max(0.0);
                        let sigma2 = rwr / (df as f64);

                        let se_g = (sigma2 / schur).sqrt();
                        let pval = if se_g.is_finite() && se_g > 0.0 && beta_g.is_finite() {
                            let z = (beta_g / se_g).abs();
                            (2.0 * normal_sf(z)).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta_g;
                        out_row[1] = se_g;
                        out_row[2] = pval;
                        if with_plrt {
                            let ml = if rwr > 0.0 && rwr.is_finite() {
                                let total_log = n_f * rwr.ln() + log_det_v;
                                c_ml - 0.5 * total_log
                            } else {
                                f64::NAN
                            };
                            let mut stat = if ml.is_finite() {
                                2.0 * (ml - nullml_val)
                            } else {
                                0.0
                            };
                            if !stat.is_finite() || stat < 0.0 {
                                stat = 0.0;
                            }
                            out_row[3] = chi2_sf_df1(stat);
                        }
                    },
                );
        };

        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    });

    Ok(out)
}

// =============================================================================
// FaST-LMM fixed-lambda association (fastlmm_assoc_chunk_f32)
// =============================================================================

struct FastlmmAssocScratch {
    xtv_inv_x: Vec<f64>,
    xtv_inv_y: Vec<f64>,
    beta: Vec<f64>,
    rhs: Vec<f64>,
    work: Vec<f64>,
    u1_xtsnp: Vec<f64>,
    u2_xtsnp: Vec<f64>,
}

impl FastlmmAssocScratch {
    fn new(p: usize) -> Self {
        let dim = p + 1;
        Self {
            xtv_inv_x: vec![0.0; dim * dim],
            xtv_inv_y: vec![0.0; dim],
            beta: vec![0.0; dim],
            rhs: vec![0.0; dim],
            work: vec![0.0; dim],
            u1_xtsnp: vec![0.0; p],
            u2_xtsnp: vec![0.0; p],
        }
    }
}

fn precompute_u2_base_fastlmm(
    u2tx: &[f64],
    u2ty: &[f64],
    n: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>) {
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

#[pyfunction]
#[pyo3(signature = (s, u1tx, u2tx, u1ty, u2ty, log10_lbd, u1tsnp_chunk, u2tsnp_chunk, threads=0, nullml=None))]
pub fn fastlmm_assoc_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    u1tx: PyReadonlyArray2<'py, f64>,
    u2tx: PyReadonlyArray2<'py, f64>,
    u1ty: PyReadonlyArray1<'py, f64>,
    u2ty: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    u1tsnp_chunk: PyReadonlyArray2<'py, f32>,
    u2tsnp_chunk: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let u1ty = u1ty.as_slice()?;
    let u2ty = u2ty.as_slice()?;

    let u1tx_arr = u1tx.as_array();
    let u2tx_arr = u2tx.as_array();
    let u1tsnp_arr = u1tsnp_chunk.as_array();
    let u2tsnp_arr = u2tsnp_chunk.as_array();

    let k = s.len();
    if k == 0 {
        return Err(PyRuntimeError::new_err("empty s"));
    }
    let (k1, p) = (u1tx_arr.shape()[0], u1tx_arr.shape()[1]);
    if k1 != k {
        return Err(PyRuntimeError::new_err("u1tx rows must equal len(s)"));
    }
    if u1ty.len() != k {
        return Err(PyRuntimeError::new_err("u1ty len must equal len(s)"));
    }
    let (n, p2) = (u2tx_arr.shape()[0], u2tx_arr.shape()[1]);
    if p2 != p {
        return Err(PyRuntimeError::new_err(
            "u1tx/u2tx must have same column count",
        ));
    }
    if u2ty.len() != n {
        return Err(PyRuntimeError::new_err("u2ty len must equal u2tx rows"));
    }

    let (m1, k2) = (u1tsnp_arr.shape()[0], u1tsnp_arr.shape()[1]);
    let (m2, n2) = (u2tsnp_arr.shape()[0], u2tsnp_arr.shape()[1]);
    if k2 != k {
        return Err(PyRuntimeError::new_err("u1tsnp_chunk must be (m, k)"));
    }
    if n2 != n {
        return Err(PyRuntimeError::new_err("u2tsnp_chunk must be (m, n)"));
    }
    if m1 != m2 {
        return Err(PyRuntimeError::new_err(
            "u1tsnp_chunk and u2tsnp_chunk must have same row count",
        ));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err("n must be > p+1"));
    }

    let u1tx_slice: &[f64] = u1tx
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u1tx must be contiguous (C-order)"))?;
    let u2tx_slice: &[f64] = u2tx
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u2tx must be contiguous (C-order)"))?;
    let u1tsnp_slice: &[f32] = u1tsnp_chunk
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u1tsnp_chunk must be contiguous (C-order)"))?;
    let u2tsnp_slice: &[f32] = u2tsnp_chunk
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u2tsnp_chunk must be contiguous (C-order)"))?;

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }
    let v2_inv = 1.0 / lbd;

    let mut v1_inv = vec![0.0_f64; k];
    for i in 0..k {
        let v1 = s[i] + lbd;
        if v1 <= 0.0 {
            return Err(PyRuntimeError::new_err("non-positive s[i]+lbd"));
        }
        v1_inv[i] = 1.0 / v1;
    }

    let (u2_xtx, u2_xty) = precompute_u2_base_fastlmm(u2tx_slice, u2ty, n, p);

    let mut base_a = vec![0.0_f64; p * p];
    for r in 0..p {
        for c in 0..=r {
            base_a[r * p + c] = v2_inv * u2_xtx[r * p + c];
        }
    }

    for i in 0..k {
        let vi = v1_inv[i];
        let base = i * p;
        for r in 0..p {
            let xir = u1tx_slice[base + r];
            for c in 0..=r {
                base_a[r * p + c] += vi * xir * u1tx_slice[base + c];
            }
        }
    }

    for r in 0..p {
        for c in 0..r {
            let vrc = base_a[r * p + c];
            base_a[c * p + r] = vrc;
        }
    }

    let mut base_b = vec![0.0_f64; p];
    for r in 0..p {
        base_b[r] = v2_inv * u2_xty[r];
    }
    for i in 0..k {
        let vi = v1_inv[i];
        let yi = u1ty[i];
        let base = i * p;
        for r in 0..p {
            base_b[r] += vi * u1tx_slice[base + r] * yi;
        }
    }

    let log_det_v: f64 =
        s.iter().map(|v| (v + lbd).ln()).sum::<f64>() + ((n - k) as f64) * lbd.ln();
    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    let m = m1;
    let out_cols = if with_plrt { 4 } else { 3 };
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let df = (n as isize) - (p as isize) - 1;
    let df_f = df as f64;

    let pool = get_cached_pool(threads)?;

    py.detach(|| {
        let mut run = || {
            out_slice
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || FastlmmAssocScratch::new(p),
                    |scr, (idx, out_row)| {
                        scr.u1_xtsnp.fill(0.0);
                        scr.u2_xtsnp.fill(0.0);

                        let u1_row = &u1tsnp_slice[idx * k..(idx + 1) * k];
                        let u2_row = &u2tsnp_slice[idx * n..(idx + 1) * n];

                        let mut u1_snp_snp = 0.0_f64;
                        let mut u1_snp_ty = 0.0_f64;
                        for i in 0..k {
                            let gi = u1_row[i] as f64;
                            let vi = v1_inv[i];
                            u1_snp_snp += vi * gi * gi;
                            u1_snp_ty += vi * gi * u1ty[i];
                            let base = i * p;
                            for r in 0..p {
                                scr.u1_xtsnp[r] += vi * u1tx_slice[base + r] * gi;
                            }
                        }

                        let mut u2_snp_snp = 0.0_f64;
                        let mut u2_snp_ty = 0.0_f64;
                        for i in 0..n {
                            let gi = u2_row[i] as f64;
                            u2_snp_snp += gi * gi;
                            u2_snp_ty += gi * u2ty[i];
                            let base = i * p;
                            for r in 0..p {
                                scr.u2_xtsnp[r] += u2tx_slice[base + r] * gi;
                            }
                        }

                        let dim = p + 1;
                        scr.xtv_inv_x[..p * p].copy_from_slice(&base_a);
                        scr.xtv_inv_y[..p].copy_from_slice(&base_b);

                        for r in 0..p {
                            let c = scr.u1_xtsnp[r] + v2_inv * scr.u2_xtsnp[r];
                            scr.xtv_inv_x[p * dim + r] = c;
                            scr.xtv_inv_x[r * dim + p] = c;
                        }
                        scr.xtv_inv_x[p * dim + p] = u1_snp_snp + v2_inv * u2_snp_snp;
                        scr.xtv_inv_y[p] = u1_snp_ty + v2_inv * u2_snp_ty;

                        let ridge = 1e-6;
                        for r in 0..dim {
                            scr.xtv_inv_x[r * dim + r] += ridge;
                        }

                        if cholesky_inplace(&mut scr.xtv_inv_x, dim).is_none() {
                            out_row[0] = f64::NAN;
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        cholesky_solve_into(&scr.xtv_inv_x, dim, &scr.xtv_inv_y, &mut scr.beta);

                        let mut r1_sum = 0.0_f64;
                        for i in 0..k {
                            let mut xb = 0.0_f64;
                            let base = i * p;
                            for r in 0..p {
                                xb += u1tx_slice[base + r] * scr.beta[r];
                            }
                            xb += (u1_row[i] as f64) * scr.beta[p];
                            let ri = u1ty[i] - xb;
                            r1_sum += v1_inv[i] * ri * ri;
                        }

                        let mut r2_sum = 0.0_f64;
                        for i in 0..n {
                            let mut xb = 0.0_f64;
                            let base = i * p;
                            for r in 0..p {
                                xb += u2tx_slice[base + r] * scr.beta[r];
                            }
                            xb += (u2_row[i] as f64) * scr.beta[p];
                            let ri = u2ty[i] - xb;
                            r2_sum += ri * ri;
                        }

                        let rtv_invr = r1_sum + v2_inv * r2_sum;
                        if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
                            out_row[0] = scr.beta[p];
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        let sigma2 = rtv_invr / df_f;
                        if !sigma2.is_finite() || sigma2 <= 0.0 {
                            out_row[0] = scr.beta[p];
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        scr.rhs.fill(0.0);
                        scr.rhs[p] = 1.0;
                        cholesky_solve_into(&scr.xtv_inv_x, dim, &scr.rhs, &mut scr.work);
                        let var_beta = sigma2 * scr.work[p];
                        let se = if var_beta.is_finite() && var_beta > 0.0 {
                            var_beta.sqrt()
                        } else {
                            f64::NAN
                        };

                        let beta = scr.beta[p];
                        let pval = if beta.is_finite() && se.is_finite() && se > 0.0 {
                            let z = (beta / se).abs();
                            (2.0 * normal_sf(z)).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta;
                        out_row[1] = se;
                        out_row[2] = pval;
                        if with_plrt {
                            let ml = if rtv_invr.is_finite() && rtv_invr > 0.0 {
                                let total_log = n_f * rtv_invr.ln() + log_det_v;
                                c_ml - 0.5 * total_log
                            } else {
                                f64::NAN
                            };
                            let mut stat = if ml.is_finite() {
                                2.0 * (ml - nullml_val)
                            } else {
                                0.0
                            };
                            if !stat.is_finite() || stat < 0.0 {
                                stat = 0.0;
                            }
                            out_row[3] = chi2_sf_df1(stat);
                        }
                    },
                );
        };

        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    });

    Ok(out)
}
