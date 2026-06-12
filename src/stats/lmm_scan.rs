use matrixmultiply::sgemm;
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
use rayon::prelude::*;
use std::borrow::Cow;
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::thread;
use std::time::Instant;

use crate::aireml::ai_reml_null_from_spectral;
use crate::bedmath::{
    decode_row_centered_full_lut, decode_subset_with_plan, packed_byte_lut,
    packed_row_missing_count_selected, SubsetDecodePlan,
};
use crate::blas::{
    cblas_sgemm_dispatch, rust_sgemm_prefers_rayon_rowmajor_f32_kernel, BlasThreadGuard, CblasInt,
    CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use crate::brent::{brent_minimize, brent_minimize_with_init};
use crate::gfcore;
use crate::gfcore::read_bim_columns;
use crate::gfreader::{
    count_packed_row_counts, count_packed_row_counts_selected_with_excluded, is_simple_snp_allele,
    precompute_excluded_sample_indices, sample_indices_are_identity,
};
use crate::gload::WindowedBedMatrix;
use crate::linalg::{
    chi2_sf_df1, chisq_from_beta_se_and_optional_plrt, cholesky_inplace, cholesky_logdet,
    cholesky_solve_into, format_chisq_value, normal_sf, sanitize_assoc_pvalue,
};
use crate::stats_common::{env_truthy, get_cached_pool, parse_index_vec_i64, AsyncTsvWriter};
use memmap2::Mmap;
use std::fs::File;

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

#[pyfunction]
#[pyo3(signature = (u_t, x, y, threads=0))]
pub fn lmm_rotate_x_y_with_ut_f64<'py>(
    py: Python<'py>,
    u_t: PyReadonlyArray2<'py, f32>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let y_slice = y.as_slice()?;
    let n = y_slice.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y must not be empty"));
    }

    let x_arr = x.as_array();
    if x_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("x must be 2D (n, q)"));
    }
    let (xn, q) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err(format!(
            "x rows must equal len(y): rows={xn}, len(y)={n}"
        )));
    }

    let ut_arr = u_t.as_array();
    if ut_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("u_t must be 2D (n, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be shape (n, n) and row-major U^T",
        ));
    }

    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };

    let pool = get_cached_pool(threads)?;
    let dim = q + 1;
    let mut utxy = vec![0.0_f64; n * dim];
    {
        let mut run = || {
            utxy.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
                let ut_row = &ut_flat[i * n..(i + 1) * n];
                for c in 0..q {
                    let mut acc = 0.0_f64;
                    for j in 0..n {
                        acc += (ut_row[j] as f64) * x_flat[j * q + c];
                    }
                    row[c] = acc;
                }
                let mut accy = 0.0_f64;
                for j in 0..n {
                    accy += (ut_row[j] as f64) * y_slice[j];
                }
                row[q] = accy;
            });
        };
        if let Some(p) = &pool {
            p.install(run);
        } else {
            run();
        }
    }

    let mut utx = vec![0.0_f64; n * q];
    let mut uty = vec![0.0_f64; n];
    for i in 0..n {
        let src = &utxy[i * dim..(i + 1) * dim];
        if q > 0 {
            utx[i * q..(i + 1) * q].copy_from_slice(&src[..q]);
        }
        uty[i] = src[q];
    }

    let utx_arr = numpy::ndarray::Array2::from_shape_vec((n, q), utx)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let uty_arr = numpy::ndarray::Array2::from_shape_vec((n, 1), uty)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        PyArray2::from_owned_array(py, utx_arr).into_bound(),
        PyArray2::from_owned_array(py, uty_arr).into_bound(),
    ))
}

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
    let log_det_xtv = cholesky_logdet(&xtv_inv_x, dim);
    let n_f = n as f64;
    let p_f = p as f64;
    let total_log = (n_f - p_f) * (rtv_invr.ln()) + log_det_v + log_det_xtv;
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
    py: Python<'py>,
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
    let (best_log10_lbd, best_cost, ml) = py.detach(|| {
        let (best_log10_lbd, best_cost) = brent_minimize(
            |x| -reml_loglike(x, s, &xcov_flat, y, None, n, p_cov),
            low,
            high,
            tol,
            max_iter,
        );
        let ml = ml_loglike(best_log10_lbd, s, &xcov_flat, y, None, n, p_cov);
        (best_log10_lbd, best_cost, ml)
    });
    let reml = -best_cost;
    let lbd = 10.0_f64.powf(best_log10_lbd);
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
    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };

    let out = ai_reml_null_from_spectral(s, &xcov_flat, y, n, p_cov, max_iter, tol, min_var)
        .map_err(PyRuntimeError::new_err)?;

    Ok((
        out.lbd,
        out.ml,
        out.reml,
        out.sigma_g2,
        out.sigma_e2,
        out.used_iter,
        out.converged,
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

#[inline]
fn rotate_snp_block_with_ut(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
) {
    if rows == 0 || n == 0 {
        return;
    }
    // C(rows, n) = A(rows, n) * U(n, n)
    // We store U^T in row-major (`u_t`). For matrixmultiply strides, expose U by
    // setting B row stride = 1 and col stride = n, i.e. B(i,j)=u_t[j*n + i].
    unsafe {
        sgemm(
            rows,
            n,
            n,
            1.0,
            snp_block.as_ptr(),
            n as isize,
            1,
            u_t.as_ptr(),
            1,
            n as isize,
            0.0,
            out_block.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
fn env_positive_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
}

#[inline]
fn row_major_mat_mul_f32_blas(
    a: &[f32],
    rows: usize,
    cols: usize,
    rhs: &[f32],
    n_rhs: usize,
    out: &mut [f32],
    threads: usize,
) {
    if rows == 0 || cols == 0 || n_rhs == 0 {
        return;
    }
    debug_assert!(a.len() >= rows.saturating_mul(cols));
    debug_assert!(rhs.len() >= cols.saturating_mul(n_rhs));
    debug_assert!(out.len() >= rows.saturating_mul(n_rhs));

    if n_rhs == 1 || (n_rhs <= 4 && rust_sgemm_prefers_rayon_rowmajor_f32_kernel()) {
        row_major_mat_mul_f32_rhs_small(a, rows, cols, rhs, n_rhs, out);
        return;
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        let _blas_guard = BlasThreadGuard::enter(threads.max(1));
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            rows as CblasInt,
            n_rhs as CblasInt,
            cols as CblasInt,
            1.0_f32,
            a.as_ptr(),
            cols as CblasInt,
            rhs.as_ptr(),
            n_rhs as CblasInt,
            0.0_f32,
            out.as_mut_ptr(),
            n_rhs as CblasInt,
        );
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        matmul_rowmajor_f32(a, rows, cols, rhs, n_rhs, out);
    }
}

#[inline]
fn row_major_mat_mul_f32_rhs_small(
    a: &[f32],
    rows: usize,
    cols: usize,
    rhs: &[f32],
    n_rhs: usize,
    out: &mut [f32],
) {
    if rows == 0 || cols == 0 || n_rhs == 0 {
        return;
    }
    debug_assert!(a.len() >= rows.saturating_mul(cols));
    debug_assert!(rhs.len() >= cols.saturating_mul(n_rhs));
    debug_assert!(out.len() >= rows.saturating_mul(n_rhs));

    let run_row = |row_idx: usize, out_row: &mut [f32]| {
        let a_row = &a[row_idx * cols..(row_idx + 1) * cols];
        if n_rhs == 1 {
            let mut acc = 0.0_f32;
            for k in 0..cols {
                acc += a_row[k] * rhs[k];
            }
            out_row[0] = acc;
            return;
        }
        for rhs_col in 0..n_rhs {
            let mut acc = 0.0_f32;
            for k in 0..cols {
                acc += a_row[k] * rhs[k * n_rhs + rhs_col];
            }
            out_row[rhs_col] = acc;
        }
    };

    if rows >= 64 && rayon::current_num_threads() > 1 {
        out.par_chunks_mut(n_rhs)
            .enumerate()
            .for_each(|(row_idx, out_row)| run_row(row_idx, out_row));
        return;
    }

    for (row_idx, out_row) in out.chunks_mut(n_rhs).enumerate() {
        run_row(row_idx, out_row);
    }
}

#[inline]
fn choose_rotate_row_tile_rows_fvlmm(rows: usize, n: usize, thread_hint: usize) -> usize {
    if let Some(v) = env_positive_usize("JX_GWAS_ROTATE_ROW_TILE") {
        return v.max(1).min(rows.max(1));
    }
    if n < 2048 {
        return choose_rotate_tile_rows(rows, thread_hint);
    }
    let target_mb = env_positive_usize("JX_GWAS_ROTATE_ROW_TILE_MB").unwrap_or_else(|| {
        if n >= 16384 {
            8usize
        } else if n >= 8192 {
            6usize
        } else {
            4usize
        }
    });
    let target_bytes = target_mb.saturating_mul(1024 * 1024);
    let bytes_per_row = n
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    let mut tile_rows = (target_bytes / bytes_per_row).max(16).min(rows.max(1));
    let max_tile = if n >= 16384 {
        256usize
    } else if n >= 8192 {
        192usize
    } else {
        128usize
    };
    tile_rows = tile_rows.clamp(16, max_tile.min(rows.max(1)));
    tile_rows.min(rows.max(1))
}

#[inline]
fn choose_rotate_col_block_cols_fvlmm(n: usize) -> usize {
    if let Some(v) = env_positive_usize("JX_GWAS_ROTATE_COL_BLOCK") {
        return v.max(1).min(n.max(1));
    }
    if n < 2048 {
        return n.max(1);
    }
    let target_mb = env_positive_usize("JX_GWAS_ROTATE_COL_BLOCK_MB").unwrap_or_else(|| {
        if n >= 16384 {
            12usize
        } else if n >= 8192 {
            16usize
        } else {
            8usize
        }
    });
    let target_bytes = target_mb.saturating_mul(1024 * 1024);
    let bytes_per_col = n.saturating_mul(std::mem::size_of::<f32>()).max(1);
    let cols = (target_bytes / bytes_per_col).max(64);
    cols.clamp(64, 1024).min(n.max(1))
}

#[inline]
fn rotate_snp_block_with_ut_parallel_blocked(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    row_tile_rows: usize,
    col_block_cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    if rows == 0 || n == 0 {
        return;
    }
    debug_assert!(snp_block.len() >= rows.saturating_mul(n));
    debug_assert!(u_t.len() >= n.saturating_mul(n));
    debug_assert!(out_block.len() >= rows.saturating_mul(n));
    let tile_rows = row_tile_rows.max(1).min(rows);
    let col_block = col_block_cols.max(1).min(n);

    let mut run_parallel = || {
        out_block
            .par_chunks_mut(tile_rows * n)
            .enumerate()
            .for_each_init(
                || vec![0.0_f32; tile_rows.saturating_mul(col_block)],
                |scratch, (chunk_idx, out_chunk)| {
                    let row_start = chunk_idx * tile_rows;
                    let rows_here = out_chunk.len() / n;
                    let a_start = row_start * n;
                    let a_end = a_start + rows_here * n;
                    let a_block = &snp_block[a_start..a_end];
                    if col_block >= n {
                        rotate_snp_block_with_ut(a_block, rows_here, n, u_t, out_chunk);
                        return;
                    }
                    for col_start in (0..n).step_by(col_block) {
                        let cols_here = (n - col_start).min(col_block);
                        let scratch_sub = &mut scratch[..rows_here * cols_here];
                        matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
                            a_block,
                            rows_here,
                            n,
                            &u_t[col_start * n..(col_start + cols_here) * n],
                            cols_here,
                            scratch_sub,
                        );
                        for r in 0..rows_here {
                            let src = &scratch_sub[r * cols_here..(r + 1) * cols_here];
                            let dst =
                                &mut out_chunk[r * n + col_start..r * n + col_start + cols_here];
                            dst.copy_from_slice(src);
                        }
                    }
                },
            );
    };

    if tile_rows >= rows {
        if col_block >= n {
            rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
            return;
        }
        let mut scratch = vec![0.0_f32; rows.saturating_mul(col_block)];
        for col_start in (0..n).step_by(col_block) {
            let cols_here = (n - col_start).min(col_block);
            let scratch_sub = &mut scratch[..rows * cols_here];
            matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
                snp_block,
                rows,
                n,
                &u_t[col_start * n..(col_start + cols_here) * n],
                cols_here,
                scratch_sub,
            );
            for r in 0..rows {
                let src = &scratch_sub[r * cols_here..(r + 1) * cols_here];
                let dst = &mut out_block[r * n + col_start..r * n + col_start + cols_here];
                dst.copy_from_slice(src);
            }
        }
        return;
    }

    if let Some(tp) = pool {
        tp.install(run_parallel);
    } else {
        run_parallel();
    }
}

#[inline]
fn assoc_rotate_prefers_rayon_rowmajor_f32_kernel() -> bool {
    for env_name in ["JX_LMM_ROTATE_F32_KERNEL", "JX_ROWMAJOR_F32_KERNEL"] {
        if let Ok(raw) = std::env::var(env_name) {
            let norm = raw.trim().to_ascii_lowercase();
            match norm.as_str() {
                "rayon" | "parallel" | "custom" => return true,
                "blas" | "gemm" | "serial" => return false,
                _ => {}
            }
        }
    }
    // For LMM/FvLMM association rotate-proj, default to BLAS-backed GEMM
    // across platforms. This path is large dense projection, unlike some
    // HE/PCG row-major kernels that still prefer custom Rayon tiling.
    false
}

#[inline]
fn rotate_snp_block_with_ut_blas(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    threads: usize,
    proj_pool: Option<&Arc<rayon::ThreadPool>>,
) {
    if rows == 0 || n == 0 {
        return;
    }
    debug_assert!(snp_block.len() >= rows.saturating_mul(n));
    debug_assert!(u_t.len() >= n.saturating_mul(n));
    debug_assert!(out_block.len() >= rows.saturating_mul(n));

    if assoc_rotate_prefers_rayon_rowmajor_f32_kernel() {
        let thread_hint = if threads > 0 {
            threads
        } else {
            rayon::current_num_threads()
        };
        let row_tile = choose_rotate_row_tile_rows_fvlmm(rows, n, thread_hint);
        let col_block = choose_rotate_col_block_cols_fvlmm(n);
        rotate_snp_block_with_ut_parallel_blocked(
            snp_block, rows, n, u_t, out_block, row_tile, col_block, proj_pool,
        );
        return;
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        let _blas_guard = BlasThreadGuard::enter(threads.max(1));
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            rows as CblasInt,
            n as CblasInt,
            n as CblasInt,
            1.0_f32,
            snp_block.as_ptr(),
            n as CblasInt,
            u_t.as_ptr(),
            n as CblasInt,
            0.0_f32,
            out_block.as_mut_ptr(),
            n as CblasInt,
        );
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
    }
}

#[inline]
#[allow(dead_code)]
fn matmul_rowmajor_f32(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    debug_assert!(a.len() >= m.saturating_mul(k));
    debug_assert!(b.len() >= k.saturating_mul(n));
    debug_assert!(out.len() >= m.saturating_mul(n));
    // C(m, n) = A(m, k) * B(k, n), all row-major contiguous.
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
#[allow(dead_code)]
fn matmul_rowmajor_f32_parallel(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
    tile_rows: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    let tile_rows = tile_rows.max(1).min(m);
    if tile_rows >= m {
        matmul_rowmajor_f32(a, m, k, b, n, out);
        return;
    }
    out.par_chunks_mut(tile_rows * n)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * tile_rows;
            let rows_here = out_chunk.len() / n;
            let a_start = row_start * k;
            let a_end = a_start + rows_here * k;
            matmul_rowmajor_f32(&a[a_start..a_end], rows_here, k, b, n, out_chunk);
        });
}

#[inline]
#[allow(dead_code)]
fn matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
    a: &[f32],
    m: usize,
    k: usize,
    b_rowmajor_nk: &[f32],
    n: usize,
    out: &mut [f32],
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    debug_assert!(a.len() >= m.saturating_mul(k));
    debug_assert!(b_rowmajor_nk.len() >= n.saturating_mul(k));
    debug_assert!(out.len() >= m.saturating_mul(n));
    // C(m, n) = A(m, k) * B^T(k, n)
    // `b_rowmajor_nk` stores B(n, k) row-major. Expose B^T by setting
    // B strides to: row_stride=1, col_stride=k.
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b_rowmajor_nk.as_ptr(),
            1,
            k as isize,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
#[allow(dead_code)]
fn matmul_rowmajor_rhs_transposed_from_rowmajor_f32_parallel(
    a: &[f32],
    m: usize,
    k: usize,
    b_rowmajor_nk: &[f32],
    n: usize,
    out: &mut [f32],
    tile_rows: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    let tile_rows = tile_rows.max(1).min(m);
    if tile_rows >= m {
        matmul_rowmajor_rhs_transposed_from_rowmajor_f32(a, m, k, b_rowmajor_nk, n, out);
        return;
    }
    out.par_chunks_mut(tile_rows * n)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * tile_rows;
            let rows_here = out_chunk.len() / n;
            let a_start = row_start * k;
            let a_end = a_start + rows_here * k;
            matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
                &a[a_start..a_end],
                rows_here,
                k,
                b_rowmajor_nk,
                n,
                out_chunk,
            );
        });
}

#[inline]
fn choose_rotate_tile_rows(rows: usize, thread_hint: usize) -> usize {
    if rows <= 1 {
        return 1;
    }
    let threads = thread_hint.max(1);
    // Avoid splitting when there is no real parallelism (or chunk is tiny):
    // one large SGEMM is much faster than many tiny calls in this case.
    if threads <= 1 || rows <= 32 {
        return rows;
    }
    // Keep enough independent tiles to feed the rayon pool, but avoid tiles
    // so small that SGEMM launch overhead dominates.
    let target_tasks = threads.saturating_mul(2).max(1);
    let mut tile_rows = (rows + target_tasks - 1) / target_tasks;
    tile_rows = tile_rows.clamp(16, 128);
    tile_rows.min(rows)
}

#[inline]
fn rotate_snp_block_with_ut_parallel(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    tile_rows: usize,
) {
    if rows == 0 || n == 0 {
        return;
    }

    let tile_rows = tile_rows.max(1).min(rows);
    if tile_rows >= rows {
        rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
        return;
    }

    out_block
        .par_chunks_mut(tile_rows * n)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * tile_rows;
            let rows_here = out_chunk.len() / n;
            let a_start = row_start * n;
            let a_end = a_start + rows_here * n;
            rotate_snp_block_with_ut(&snp_block[a_start..a_end], rows_here, n, u_t, out_chunk);
        });
}

#[inline]
fn use_rotate_finalize_pipeline(default_on: bool) -> bool {
    match std::env::var("JX_GWAS_ROT_PIPELINE") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" | "force" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default_on,
        },
        Err(_) => default_on,
    }
}

#[inline]
fn use_packed_three_stage_pipeline(default_on: bool) -> usize {
    let default = if default_on { 2 } else { 0 };
    match std::env::var("JX_GWAS_PACKED_PIPELINE") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "2" | "double" => 2,
            "1" | "true" | "yes" | "on" | "force" => default,
            "0" | "false" | "no" | "off" => 0,
            _ => default,
        },
        Err(_) => default,
    }
}

#[inline]
fn record_elapsed_nanos(acc: &AtomicU64, t0: Instant) {
    let nanos = t0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    acc.fetch_add(nanos, Ordering::Relaxed);
}

#[inline]
fn elapsed_nanos_to_secs(acc: &AtomicU64) -> f64 {
    (acc.load(Ordering::Relaxed) as f64) * 1e-9
}

#[inline]
fn elapsed_nanos_since(t0: Instant) -> u64 {
    t0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

struct FvlmmChunkProfileStats {
    calls: AtomicU64,
    rows: AtomicU64,
    wall_nanos: AtomicU64,
    prep_nanos: AtomicU64,
    proj_nanos: AtomicU64,
    assoc_nanos: AtomicU64,
}

impl FvlmmChunkProfileStats {
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            wall_nanos: AtomicU64::new(0),
            prep_nanos: AtomicU64::new(0),
            proj_nanos: AtomicU64::new(0),
            assoc_nanos: AtomicU64::new(0),
        }
    }
}

static FVLMM_CHUNK_PROFILE_ROT_CACHE: FvlmmChunkProfileStats = FvlmmChunkProfileStats::new();
static FVLMM_CHUNK_PROFILE_ROT_RAW: FvlmmChunkProfileStats = FvlmmChunkProfileStats::new();
static FVLMM_CHUNK_PROFILE_FROM_SNP_CACHE: FvlmmChunkProfileStats = FvlmmChunkProfileStats::new();
static FVLMM_CHUNK_PROFILE_FROM_SNP_RAW: FvlmmChunkProfileStats = FvlmmChunkProfileStats::new();
static FVLMM_CHUNK_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static FVLMM_CHUNK_PROFILE_EVERY: OnceLock<u64> = OnceLock::new();

#[inline]
fn fvlmm_chunk_profiling_enabled() -> bool {
    *FVLMM_CHUNK_PROFILE_ENABLED.get_or_init(|| env_truthy("JX_FVLMM_CHUNK_PROFILING"))
}

#[inline]
fn fvlmm_chunk_profile_every() -> u64 {
    *FVLMM_CHUNK_PROFILE_EVERY
        .get_or_init(|| env_positive_usize("JX_FVLMM_CHUNK_PROFILE_EVERY").unwrap_or(1) as u64)
}

#[inline]
fn fvlmm_chunk_profile_state(route: &'static str) -> &'static FvlmmChunkProfileStats {
    match route {
        "rot_cache" => &FVLMM_CHUNK_PROFILE_ROT_CACHE,
        "rot_raw" => &FVLMM_CHUNK_PROFILE_ROT_RAW,
        "from_snp_cache" => &FVLMM_CHUNK_PROFILE_FROM_SNP_CACHE,
        "from_snp_raw" => &FVLMM_CHUNK_PROFILE_FROM_SNP_RAW,
        _ => &FVLMM_CHUNK_PROFILE_FROM_SNP_CACHE,
    }
}

fn maybe_emit_fvlmm_chunk_profile(
    route: &'static str,
    rows: usize,
    n: usize,
    p: usize,
    prep_nanos: u64,
    proj_nanos: u64,
    assoc_nanos: u64,
    wall_nanos: u64,
    threads: usize,
    proj_threads: usize,
    assoc_threads: usize,
) {
    if !fvlmm_chunk_profiling_enabled() {
        return;
    }

    let stats = fvlmm_chunk_profile_state(route);
    let call_idx = stats.calls.fetch_add(1, Ordering::Relaxed) + 1;
    stats.rows.fetch_add(rows as u64, Ordering::Relaxed);
    stats.wall_nanos.fetch_add(wall_nanos, Ordering::Relaxed);
    stats.prep_nanos.fetch_add(prep_nanos, Ordering::Relaxed);
    stats.proj_nanos.fetch_add(proj_nanos, Ordering::Relaxed);
    stats.assoc_nanos.fetch_add(assoc_nanos, Ordering::Relaxed);

    let every = fvlmm_chunk_profile_every().max(1);
    if call_idx % every != 0 {
        return;
    }

    let rows_total = stats.rows.load(Ordering::Relaxed);
    let cum_wall = stats.wall_nanos.load(Ordering::Relaxed) as f64 * 1e-9;
    let cum_prep = stats.prep_nanos.load(Ordering::Relaxed) as f64 * 1e-9;
    let cum_proj = stats.proj_nanos.load(Ordering::Relaxed) as f64 * 1e-9;
    let cum_assoc = stats.assoc_nanos.load(Ordering::Relaxed) as f64 * 1e-9;
    let active_total = cum_prep + cum_proj + cum_assoc;
    let pct = |secs: f64| -> f64 {
        if active_total > 0.0 {
            secs * 100.0 / active_total
        } else {
            0.0
        }
    };
    let overlap = if cum_wall > 0.0 {
        active_total / cum_wall
    } else {
        0.0
    };

    eprintln!(
        "FvLMM chunk profiling[{route}]: calls={call_idx}, rows_total={rows_total}, last_rows={rows}, n={n}, p={p}, chunk_wall={:.3}s, chunk_prep={:.3}s, chunk_proj={:.3}s, chunk_assoc={:.3}s, cum_wall={:.3}s, cum_prep={:.3}s ({:.1}%), cum_proj={:.3}s ({:.1}%), cum_assoc={:.3}s ({:.1}%), active_over_wall={:.2}x, threads={}, proj_threads={}, assoc_threads={}",
        (wall_nanos as f64) * 1e-9,
        (prep_nanos as f64) * 1e-9,
        (proj_nanos as f64) * 1e-9,
        (assoc_nanos as f64) * 1e-9,
        cum_wall,
        cum_prep,
        pct(cum_prep),
        cum_proj,
        pct(cum_proj),
        cum_assoc,
        pct(cum_assoc),
        overlap,
        threads,
        proj_threads,
        assoc_threads,
    );
}

fn run_rotate_finalize_double_buffer_f32<FR, FF>(
    total_rows: usize,
    n: usize,
    block_rows: usize,
    enable_pipeline: bool,
    rotate_stage: FR,
    mut finalize_stage: FF,
) where
    FR: Fn(usize, usize, &mut [f32]) + Sync + Send,
    FF: FnMut(usize, usize, &[f32]),
{
    if total_rows == 0 || n == 0 || block_rows == 0 {
        return;
    }
    if !enable_pipeline || total_rows <= block_rows {
        let mut buf = vec![0.0_f32; block_rows * n];
        let mut start = 0usize;
        while start < total_rows {
            let rows = (total_rows - start).min(block_rows);
            rotate_stage(start, rows, &mut buf[..rows * n]);
            finalize_stage(start, rows, &buf[..rows * n]);
            start += rows;
        }
        return;
    }

    let buf_len = block_rows * n;
    let (task_tx, task_rx) = mpsc::sync_channel::<(usize, usize, Vec<f32>)>(2);
    let (done_tx, done_rx) = mpsc::sync_channel::<(usize, usize, Vec<f32>)>(2);

    thread::scope(|scope| {
        scope.spawn(move || {
            while let Ok((start, rows, mut buf)) = task_rx.recv() {
                rotate_stage(start, rows, &mut buf[..rows * n]);
                if done_tx.send((start, rows, buf)).is_err() {
                    break;
                }
            }
        });

        let mut next_start = 0usize;
        let mut inflight = 0usize;

        for _ in 0..2usize {
            if next_start >= total_rows {
                break;
            }
            let rows = (total_rows - next_start).min(block_rows);
            task_tx
                .send((next_start, rows, vec![0.0_f32; buf_len]))
                .expect("rotate pipeline task send");
            next_start += rows;
            inflight += 1;
        }

        while inflight > 0 {
            let (start, rows, buf) = done_rx.recv().expect("rotate pipeline recv");
            inflight -= 1;
            finalize_stage(start, rows, &buf[..rows * n]);
            if next_start < total_rows {
                let next_rows = (total_rows - next_start).min(block_rows);
                task_tx
                    .send((next_start, next_rows, buf))
                    .expect("rotate pipeline task resend");
                next_start += next_rows;
                inflight += 1;
            }
        }

        drop(task_tx);
    });
}

struct PackedStageBufferF32 {
    start: usize,
    rows: usize,
    g_block: Vec<f32>,
    rot_block: Vec<f32>,
    out_block: Vec<f64>,
    miss_block: Vec<usize>,
}

impl PackedStageBufferF32 {
    fn new(block_rows: usize, n: usize, out_cols: usize) -> Self {
        Self {
            start: 0usize,
            rows: 0usize,
            g_block: vec![0.0_f32; block_rows.saturating_mul(n)],
            rot_block: vec![0.0_f32; block_rows.saturating_mul(n)],
            out_block: vec![0.0_f64; block_rows.saturating_mul(out_cols)],
            miss_block: vec![0usize; block_rows],
        }
    }
}

/// Per-SNP filter results from parallel counting (Copy-friendly, no heap strings).
struct SnpCounts {
    flip: bool,
    maf: f32,
    miss_rate: f32,
    missing_count: usize,
}

/// Double-buffer for the streaming BED → TSV pipeline.
struct StreamingChunk {
    // Stage 1 output: count + decode
    g_block: Vec<f32>,
    miss_block: Vec<usize>,
    indices: Vec<usize>,
    flip: Vec<bool>,
    maf: Vec<f32>,
    miss_rate: Vec<f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    a0: Vec<String>,
    a1: Vec<String>,
    // Stage 2 output
    rot_block: Vec<f32>,
    out_block: Vec<f64>,
    // Bookkeeping
    rows: usize,
    scanned_to: usize,
}

impl StreamingChunk {
    fn new(capacity: usize, n: usize, out_cols: usize) -> Self {
        Self {
            g_block: vec![0.0_f32; capacity.saturating_mul(n)],
            miss_block: vec![0usize; capacity],
            indices: Vec::with_capacity(capacity),
            flip: Vec::with_capacity(capacity),
            maf: Vec::with_capacity(capacity),
            miss_rate: Vec::with_capacity(capacity),
            chrom: Vec::with_capacity(capacity),
            pos: Vec::with_capacity(capacity),
            snp: Vec::with_capacity(capacity),
            a0: Vec::with_capacity(capacity),
            a1: Vec::with_capacity(capacity),
            rot_block: vec![0.0_f32; capacity.saturating_mul(n)],
            out_block: vec![0.0_f64; capacity.saturating_mul(out_cols)],
            rows: 0,
            scanned_to: 0,
        }
    }

    fn clear(&mut self) {
        self.indices.clear();
        self.flip.clear();
        self.maf.clear();
        self.miss_rate.clear();
        self.chrom.clear();
        self.pos.clear();
        self.snp.clear();
        self.a0.clear();
        self.a1.clear();
        self.rows = 0;
        self.scanned_to = 0;
    }
}

fn run_packed_decode_rotate_finalize_triple_buffer_f32<FD, FR, FF, E>(
    total_rows: usize,
    n: usize,
    block_rows: usize,
    out_cols: usize,
    pipeline_depth: usize,
    decode_stage: FD,
    rotate_stage: FR,
    mut finalize_stage: FF,
) -> Result<(), E>
where
    FD: Fn(usize, usize, &mut [f32], &mut [usize]) + Sync + Send,
    FR: Fn(usize, usize, &[f32], &mut [f32]) + Sync + Send,
    FF: FnMut(usize, usize, &[f32], &mut [f64], &[usize]) -> Result<(), E>,
{
    if total_rows == 0 || n == 0 || block_rows == 0 {
        return Ok(());
    }
    // Sequential fallback: pipeline_depth < 2 or single block
    if pipeline_depth < 2 || total_rows <= block_rows {
        let mut buf = PackedStageBufferF32::new(block_rows, n, out_cols);
        let mut start = 0usize;
        while start < total_rows {
            let rows = (total_rows - start).min(block_rows);
            decode_stage(
                start,
                rows,
                &mut buf.g_block[..rows * n],
                &mut buf.miss_block[..rows],
            );
            rotate_stage(
                start,
                rows,
                &buf.g_block[..rows * n],
                &mut buf.rot_block[..rows * n],
            );
            finalize_stage(
                start,
                rows,
                &buf.rot_block[..rows * n],
                &mut buf.out_block[..rows * out_cols],
                &buf.miss_block[..rows],
            )?;
            start += rows;
        }
        return Ok(());
    }

    // Double-buffer pipeline: decode || rotate+finalize
    {
        let mut next_start = 0usize;
        crate::pipeline::run_double_buffer(
            2,
            || PackedStageBufferF32::new(block_rows, n, out_cols),
            |buf: &mut PackedStageBufferF32| {
                if next_start >= total_rows {
                    return false;
                }
                let rows = (total_rows - next_start).min(block_rows);
                buf.start = next_start;
                buf.rows = rows;
                decode_stage(
                    next_start,
                    rows,
                    &mut buf.g_block[..rows * n],
                    &mut buf.miss_block[..rows],
                );
                next_start += rows;
                next_start < total_rows
            },
            |buf: &mut PackedStageBufferF32| {
                let rows = buf.rows;
                rotate_stage(
                    buf.start,
                    rows,
                    &buf.g_block[..rows * n],
                    &mut buf.rot_block[..rows * n],
                );
                finalize_stage(
                    buf.start,
                    rows,
                    &buf.rot_block[..rows * n],
                    &mut buf.out_block[..rows * out_cols],
                    &buf.miss_block[..rows],
                )
            },
        )?;
        return Ok(());
    }
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, snp_chunk, u_t, max_iter=50, tol=1e-2, threads=0, nullml=None, rotate_block_rows=256))]
pub fn lmm_reml_chunk_from_snp_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m_chunk, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let m_chunk = snp_arr.shape()[0];
    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let beta_se_p = PyArray2::<f64>::zeros(py, [m_chunk, out_cols], false).into_bound();
    let beta_se_p_slice: &mut [f64] = unsafe {
        beta_se_p
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("beta_se_p not contiguous"))?
    };

    // Keep same behavior as lmm_reml_chunk_f32: dedicated pool can be faster here.
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

    let block_rows = rotate_block_rows.max(1);

    py.detach(|| {
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            out_block
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || vec![0.0_f64; n],
                    |snp_vec, (idx, out_row)| {
                        let row = &g_block[idx * n..(idx + 1) * n];
                        for i in 0..n {
                            snp_vec[i] = row[i] as f64;
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

        let mut start = 0usize;
        while start < m_chunk {
            let rows = (m_chunk - start).min(block_rows);
            let snp_block = &snp_flat[start * n..(start + rows) * n];
            let rot_block = &mut rot_buf[..rows * n];
            let rotate_tile_rows = choose_rotate_tile_rows(
                rows,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                },
            );

            let out_block = &mut beta_se_p_slice[start * out_cols..(start + rows) * out_cols];
            if let Some(tp) = &pool {
                tp.install(|| {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                    run_rows(rot_block, out_block);
                });
            } else {
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    rotate_tile_rows,
                );
                run_rows(rot_block, out_block);
            }
            start += rows;
        }
    });

    Ok(beta_se_p)
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, snp_chunk, u_t, nullml, max_iter=50, tol=1e-2, threads=0, rotate_block_rows=256))]
pub fn lmm_reml_lmm2_chunk_from_snp_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    nullml: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m_chunk, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !nullml.is_finite() {
        return Err(PyRuntimeError::new_err("nullml must be finite"));
    }

    let m_chunk = snp_arr.shape()[0];
    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let out_cols = 6usize;
    let out = PyArray2::<f64>::zeros(py, [m_chunk, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("lmm2 output not contiguous"))?
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

    let block_rows = rotate_block_rows.max(1);

    py.detach(|| {
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            out_block
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || vec![0.0_f64; n],
                    |snp_vec, (idx, out_row)| {
                        let row = &g_block[idx * n..(idx + 1) * n];
                        for i in 0..n {
                            snp_vec[i] = row[i] as f64;
                        }

                        let (best_log10_lbd_reml, _best_reml_cost) = brent_minimize(
                            |x| -reml_loglike(x, s, &xcov_flat, y, Some(&snp_vec[..]), n, p_cov),
                            low,
                            high,
                            tol,
                            max_iter,
                        );
                        let (beta, se, lambda_reml) =
                            final_beta_se(best_log10_lbd_reml, s, &xcov_flat, y, &snp_vec, n, p_cov);
                        let pwald = if beta.is_finite() && se.is_finite() && se > 0.0 {
                            let z = beta / se;
                            (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        let (best_log10_lbd_ml, best_ml_cost) = brent_minimize_with_init(
                            |x| -ml_loglike(x, s, &xcov_flat, y, Some(&snp_vec[..]), n, p_cov),
                            low,
                            high,
                            tol,
                            max_iter,
                            Some(best_log10_lbd_reml),
                        );
                        let mut ml_alt = -best_ml_cost;
                        if !ml_alt.is_finite() {
                            ml_alt = ml_loglike(
                                best_log10_lbd_ml,
                                s,
                                &xcov_flat,
                                y,
                                Some(&snp_vec[..]),
                                n,
                                p_cov,
                            );
                        }
                        let mut stat = if ml_alt.is_finite() {
                            2.0 * (ml_alt - nullml)
                        } else {
                            0.0
                        };
                        if !stat.is_finite() || stat < 0.0 {
                            stat = 0.0;
                        }
                        let plrt = chi2_sf_df1(stat);

                        out_row[0] = beta;
                        out_row[1] = se;
                        out_row[2] = if pwald.is_finite() { pwald } else { 1.0 };
                        out_row[3] = lambda_reml;
                        out_row[4] = ml_alt;
                        out_row[5] = plrt;
                    },
                );
        };

        let mut start = 0usize;
        while start < m_chunk {
            let rows = (m_chunk - start).min(block_rows);
            let snp_block = &snp_flat[start * n..(start + rows) * n];
            let rot_block = &mut rot_buf[..rows * n];
            let rotate_tile_rows = choose_rotate_tile_rows(
                rows,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                },
            );
            let out_block = &mut out_slice[start * out_cols..(start + rows) * out_cols];
            if let Some(tp) = &pool {
                tp.install(|| {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                    run_rows(rot_block, out_block);
                });
            } else {
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    rotate_tile_rows,
                );
                run_rows(rot_block, out_block);
            }
            start += rows;
        }
    });

    Ok(out)
}

// ------------------------------------------------------------
// Helpers: dot loops
// ------------------------------------------------------------

#[inline]
fn dot_loop(a: &[f64], b: &[f64]) -> f64 {
    dot_loop_simd(a, b)
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_loop_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd,
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

#[cfg(target_arch = "aarch64")]
unsafe fn dot_loop_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};

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

struct FixedLambdaAssocCacheF32 {
    w: Vec<f32>,
    py_tilde: Vec<f32>,
    wx_tilde: Vec<f32>, // row-major (n, p)
    a_chol: Vec<f64>,
    ypy: f64,
    log_det_v: f64,
    df: i32,
}

#[pyclass]
pub struct FvLmmAssocCache {
    cache: Arc<FixedLambdaAssocCacheF32>,
    n: usize,
    p: usize,
    lbd: f64,
}

#[pymethods]
impl FvLmmAssocCache {
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[getter]
    fn p(&self) -> usize {
        self.p
    }

    #[getter]
    fn lbd(&self) -> f64 {
        self.lbd
    }
}

#[inline]
fn stage_proj_threads_or(requested_threads: usize) -> usize {
    env_positive_usize("JX_FVLMM_PROJ_THREADS")
        .or_else(|| env_positive_usize("JX_MLM_RUST_THREADS"))
        .unwrap_or(requested_threads)
}

#[inline]
fn stage_assoc_threads_or(requested_threads: usize) -> usize {
    env_positive_usize("JX_FVLMM_ASSOC_THREADS")
        .or_else(|| env_positive_usize("JX_MLM_RUST_THREADS"))
        .unwrap_or(requested_threads)
}

fn build_u_t_sub_from_sample_idx(
    u_slice: &[f32],
    k_full: usize,
    sample_idx: &[usize],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Vec<f32> {
    let n = sample_idx.len();
    let mut u_t_sub = vec![0.0_f32; n * n];
    let mut run = || {
        u_t_sub
            .par_chunks_mut(n.max(1))
            .enumerate()
            .for_each(|(col_idx, dst_row)| {
                for (out_col, &sid) in sample_idx.iter().enumerate() {
                    dst_row[out_col] = u_slice[sid * k_full + col_idx];
                }
            });
    };
    if let Some(tp) = pool {
        tp.install(run);
    } else {
        run();
    }
    u_t_sub
}

fn prepare_fixed_lambda_assoc_cache_f32(
    s: &[f64],
    xcov_slice: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    lbd: f64,
) -> PyResult<FixedLambdaAssocCacheF32> {
    let mut w = vec![0.0_f32; n];
    let mut log_det_v = 0.0_f64;
    for i in 0..n {
        let vv = s[i] + lbd;
        if !(vv.is_finite() && vv > 0.0) {
            return Err(PyRuntimeError::new_err("non-positive s[i]+lbd"));
        }
        w[i] = (1.0 / vv) as f32;
        log_det_v += vv.ln();
    }

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
                a[r * p + c] += wi * xir * xcov_slice[base + c];
            }
        }
    }

    let ridge = 1e-6_f64;
    for r in 0..p {
        a[r * p + r] += ridge;
        for c in 0..r {
            a[c * p + r] = a[r * p + c];
        }
    }
    if cholesky_inplace(&mut a, p).is_none() {
        return Err(PyRuntimeError::new_err("X'WX not SPD"));
    }

    let mut a_inv_b = vec![0.0_f64; p];
    cholesky_solve_into(&a, p, &b, &mut a_inv_b);
    let ypy = (ywy - dot_loop(&b, &a_inv_b)).max(0.0);

    let mut wx_tilde = vec![0.0_f32; n * p];
    let mut py_tilde = vec![0.0_f32; n];
    for i in 0..n {
        let wi = w[i] as f64;
        let base = i * p;
        let mut x_aib = 0.0_f64;
        for r in 0..p {
            let xir = xcov_slice[base + r];
            wx_tilde[base + r] = (wi * xir) as f32;
            x_aib += xir * a_inv_b[r];
        }
        py_tilde[i] = (wi * (y[i] - x_aib)) as f32;
    }

    let df = (n as i32) - (p as i32) - 1;
    if df <= 0 {
        return Err(PyRuntimeError::new_err("df <= 0"));
    }

    Ok(FixedLambdaAssocCacheF32 {
        w,
        py_tilde,
        wx_tilde,
        a_chol: a,
        ypy,
        log_det_v,
        df,
    })
}

#[inline]
fn missing_count_from_rate(v: f32, n_samples: usize) -> usize {
    if !v.is_finite() || v <= 0.0 {
        0usize
    } else {
        ((v as f64) * (n_samples as f64)).round().max(0.0) as usize
    }
}

fn fill_packed_missing_block(
    miss_sub: &mut [usize],
    sample_identity: bool,
    row_missing: &[f32],
    row_idx: Option<&[usize]>,
    start: usize,
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    sample_idx: &[usize],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    if sample_identity {
        if let Some(tp) = pool {
            tp.install(|| {
                miss_sub.par_iter_mut().enumerate().for_each(|(off, dst)| {
                    let idx = start + off;
                    // row_missing is aligned to the active/output row order; only packed
                    // row fetches should use row_idx indirection.
                    *dst = missing_count_from_rate(row_missing[idx], n_samples);
                });
            });
        } else {
            miss_sub.iter_mut().enumerate().for_each(|(off, dst)| {
                let idx = start + off;
                *dst = missing_count_from_rate(row_missing[idx], n_samples);
            });
        }
        return;
    }

    if let Some(tp) = pool {
        tp.install(|| {
            miss_sub.par_iter_mut().enumerate().for_each(|(off, dst)| {
                let idx = start + off;
                let src_row = row_idx.map(|v| v[idx]).unwrap_or(idx);
                let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                *dst = packed_row_missing_count_selected(row, n_samples, sample_idx);
            });
        });
    } else {
        miss_sub.iter_mut().enumerate().for_each(|(off, dst)| {
            let idx = start + off;
            let src_row = row_idx.map(|v| v[idx]).unwrap_or(idx);
            let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
            *dst = packed_row_missing_count_selected(row, n_samples, sample_idx);
        });
    }
}

#[inline]
fn maybe_build_dense_subset_pos(sample_idx: &[usize], n_samples_full: usize) -> Option<Vec<isize>> {
    if sample_idx.is_empty() || sample_idx.len() >= n_samples_full {
        return None;
    }
    if sample_idx.len().saturating_mul(4) < n_samples_full.saturating_mul(3) {
        return None;
    }
    let mut pos = vec![-1isize; n_samples_full];
    for (j, &sid) in sample_idx.iter().enumerate() {
        if pos[sid] >= 0 {
            return None;
        }
        pos[sid] = j as isize;
    }
    Some(pos)
}

#[inline]
fn packed_rotate_block_target_bytes(n: usize) -> usize {
    let mb = std::env::var("JX_PACKED_ROTATE_BLOCK_MB")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or_else(|| {
            std::env::var("JX_GWAS_ROTATE_BLOCK_MB")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .filter(|&v| v > 0)
        })
        .unwrap_or_else(|| {
            if n >= 8_000 {
                256usize
            } else if n >= 4_000 {
                128usize
            } else {
                64usize
            }
        });
    mb.saturating_mul(1024 * 1024)
}

#[inline]
fn packed_rotate_block_rows(max_rows: usize, n: usize, extra_row_bytes: usize) -> usize {
    let bytes_per_row = n
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f32>())
        .saturating_add(extra_row_bytes)
        .max(1);
    max_rows
        .min((packed_rotate_block_target_bytes(n) / bytes_per_row).max(1))
        .max(1)
}

#[inline]
fn packed_progress_should_emit(
    done: usize,
    total: usize,
    progress_block: usize,
    next_emit: &mut usize,
) -> bool {
    if done >= total {
        *next_emit = total.saturating_add(progress_block.max(1));
        return true;
    }
    if done < *next_emit {
        return false;
    }
    let step = progress_block.max(1);
    while *next_emit <= done {
        *next_emit = next_emit.saturating_add(step);
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn assoc_fixed_lambda_rot_block_blas_f32(
    g_rot: &[f32],
    rows: usize,
    n: usize,
    p: usize,
    cache: &FixedLambdaAssocCacheF32,
    out: &mut [f64],
    out_cols: usize,
    threads: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
    nullml: Option<f64>,
) {
    if rows == 0 {
        return;
    }
    debug_assert!(g_rot.len() >= rows.saturating_mul(n));
    debug_assert!(out.len() >= rows.saturating_mul(out_cols));

    let mut num_buf = vec![0.0_f32; rows];
    let mut c_buf = vec![0.0_f32; rows * p];
    row_major_mat_mul_f32_blas(g_rot, rows, n, &cache.py_tilde, 1, &mut num_buf, threads);
    row_major_mat_mul_f32_blas(g_rot, rows, n, &cache.wx_tilde, p, &mut c_buf, threads);

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    let mut run = || {
        out.par_chunks_mut(out_cols).enumerate().for_each_init(
            || AssocScratch::new(p),
            |scr, (idx, out_row)| {
                scr.reset();
                let row = &g_rot[idx * n..(idx + 1) * n];
                let crow = &c_buf[idx * p..(idx + 1) * p];

                let mut d = 0.0_f64;
                for i in 0..n {
                    let gi = row[i] as f64;
                    d += (cache.w[i] as f64) * gi * gi;
                }
                for r in 0..p {
                    scr.c[r] = crow[r] as f64;
                }

                cholesky_solve_into(&cache.a_chol, p, &scr.c, &mut scr.a_inv_c);
                let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                let schur = d - ct_aic;
                if schur <= 1e-12 || !schur.is_finite() {
                    out_row[0] = f64::NAN;
                    out_row[1] = f64::NAN;
                    out_row[2] = f64::NAN;
                    if with_plrt {
                        out_row[3] = 1.0;
                    }
                    return;
                }

                let num = num_buf[idx] as f64;
                let beta_g = num / schur;
                let rwr = (cache.ypy - (num * num) / schur).max(0.0);
                let sigma2 = rwr / (cache.df as f64);
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
                        let total_log = n_f * rwr.ln() + cache.log_det_v;
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

    if let Some(tp) = pool {
        tp.install(run);
    } else {
        run();
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

    let ridge = 1e-6;
    for r in 0..p {
        a[r * p + r] += ridge;
        for c in 0..r {
            let vrc = a[r * p + c];
            a[c * p + r] = vrc;
        }
    }

    if cholesky_inplace(&mut a, p).is_none() {
        return Err(PyRuntimeError::new_err("X'WX not SPD"));
    }

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

                        // ct_aib = c' A^{-1} b, computed in one loop without dot().
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

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd))]
pub fn fvlmm_assoc_prepare_cache_f32<'py>(
    _py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
) -> PyResult<FvLmmAssocCache> {
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
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let p = p_cov;
    let cache = prepare_fixed_lambda_assoc_cache_f32(s, xcov_slice, y, n, p, lbd)?;
    Ok(FvLmmAssocCache {
        cache: Arc::new(cache),
        n,
        p,
        lbd,
    })
}

fn fvlmm_assoc_chunk_with_cache_impl<'py>(
    py: Python<'py>,
    cache: Arc<FixedLambdaAssocCacheF32>,
    n: usize,
    p: usize,
    g_rot_chunk: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
    profile_route: Option<&'static str>,
    prep_nanos: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let g_arr = g_rot_chunk.as_array();
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("g_rot_chunk must be (m, n)"));
    }

    let with_plrt = nullml.is_some();
    let out_cols = if with_plrt { 4 } else { 3 };
    let m = g_arr.shape()[0];
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let g_flat: Cow<[f32]> = match g_rot_chunk.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(g_arr.iter().copied().collect()),
    };
    let assoc_threads = stage_assoc_threads_or(threads);
    let pool = get_cached_pool(assoc_threads)?;

    py.detach(move || {
        let total_t0 = Instant::now();
        let at0 = Instant::now();
        assoc_fixed_lambda_rot_block_blas_f32(
            &g_flat,
            m,
            n,
            p,
            cache.as_ref(),
            out_slice,
            out_cols,
            assoc_threads,
            pool.as_ref(),
            nullml,
        );
        if let Some(route) = profile_route {
            let assoc_nanos = elapsed_nanos_since(at0);
            let wall_nanos = elapsed_nanos_since(total_t0);
            maybe_emit_fvlmm_chunk_profile(
                route,
                m,
                n,
                p,
                prep_nanos,
                0,
                assoc_nanos,
                wall_nanos,
                threads,
                0,
                assoc_threads,
            );
        }
    });

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (cache, g_rot_chunk, threads=0, nullml=None))]
pub fn fvlmm_assoc_chunk_with_cache_f32<'py>(
    py: Python<'py>,
    cache: PyRef<'py, FvLmmAssocCache>,
    g_rot_chunk: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    fvlmm_assoc_chunk_with_cache_impl(
        py,
        Arc::clone(&cache.cache),
        cache.n,
        cache.p,
        g_rot_chunk,
        threads,
        nullml,
        Some("rot_cache"),
        0,
    )
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd, g_rot_chunk, threads=0, nullml=None))]
pub fn fvlmm_assoc_chunk_f32<'py>(
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

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let p = p_cov;
    let prep_t0 = Instant::now();
    let cache = Arc::new(prepare_fixed_lambda_assoc_cache_f32(
        s, xcov_slice, y, n, p, lbd,
    )?);
    let prep_nanos = elapsed_nanos_since(prep_t0);
    fvlmm_assoc_chunk_with_cache_impl(
        py,
        cache,
        n,
        p,
        g_rot_chunk,
        threads,
        nullml,
        Some("rot_raw"),
        prep_nanos,
    )
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd, snp_chunk, u_t, threads=0, nullml=None, rotate_block_rows=512))]
pub fn lmm_assoc_chunk_from_snp_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
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

    let m = snp_arr.shape()[0];
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };
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

    let ridge = 1e-6;
    for r in 0..p {
        a[r * p + r] += ridge;
        for c in 0..r {
            let vrc = a[r * p + c];
            a[c * p + r] = vrc;
        }
    }

    if cholesky_inplace(&mut a, p).is_none() {
        return Err(PyRuntimeError::new_err("X'WX not SPD"));
    }

    let mut a_inv_b = vec![0.0_f64; p];
    cholesky_solve_into(&a, p, &b, &mut a_inv_b);

    // b'A^{-1}b is constant
    let b_aib = dot_loop(&b, &a_inv_b);

    let df = (n as i32) - (p as i32) - 1;
    if df <= 0 {
        return Err(PyRuntimeError::new_err("df <= 0"));
    }

    let pool = get_cached_pool(threads)?;
    let block_rows = rotate_block_rows.max(1);

    py.detach(|| {
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            out_block
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || AssocScratch::new(p),
                    |scr, (idx, out_row)| {
                        scr.reset();
                        let row = &g_block[idx * n..(idx + 1) * n];

                        // c = X'Wg , d = g'Wg, e = g'Wy
                        let mut d = 0.0_f64;
                        let mut e = 0.0_f64;

                        for i in 0..n {
                            let gi = row[i] as f64;
                            let wi = w[i] as f64;
                            let yi = y[i];
                            let base = i * p;

                            let wgi = wi * gi;
                            d += wgi * gi;
                            e += wgi * yi;

                            for r in 0..p {
                                scr.c[r] += wgi * xcov_slice[base + r];
                            }
                        }

                        cholesky_solve_into(&a, p, &scr.c, &mut scr.a_inv_c);
                        let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                        let schur = d - ct_aic;
                        if schur <= 1e-12 || !schur.is_finite() {
                            out_row[0] = f64::NAN;
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        let mut ct_aib = 0.0_f64;
                        for r in 0..p {
                            ct_aib += scr.c[r] * a_inv_b[r];
                        }

                        let num = e - ct_aib;
                        let beta_g = num / schur;

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

        let mut start = 0usize;
        while start < m {
            let rows = (m - start).min(block_rows);
            let snp_block = &snp_flat[start * n..(start + rows) * n];
            let rot_block = &mut rot_buf[..rows * n];
            let rotate_tile_rows = choose_rotate_tile_rows(
                rows,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                },
            );

            let out_block = &mut out_slice[start * out_cols..(start + rows) * out_cols];
            if let Some(tp) = &pool {
                tp.install(|| {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                    run_rows(rot_block, out_block);
                });
            } else {
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    rotate_tile_rows,
                );
                run_rows(rot_block, out_block);
            }
            start += rows;
        }
    });

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (cache, snp_chunk, u_t, threads=0, nullml=None, rotate_block_rows=512))]
pub fn fvlmm_assoc_chunk_from_snp_with_cache_f32<'py>(
    py: Python<'py>,
    cache: PyRef<'py, FvLmmAssocCache>,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = cache.n;
    let p = cache.p;
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }

    let with_plrt = nullml.is_some();
    let out_cols = if with_plrt { 4 } else { 3 };
    let m = snp_arr.shape()[0];
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let proj_threads = stage_proj_threads_or(threads);
    let assoc_threads = stage_assoc_threads_or(threads);
    let proj_pool = get_cached_pool(proj_threads)?;
    let assoc_pool = get_cached_pool(assoc_threads)?;
    let block_rows = rotate_block_rows.max(1);
    let cache_arc = Arc::clone(&cache.cache);
    let use_pipeline = use_rotate_finalize_pipeline(
        !assoc_rotate_prefers_rayon_rowmajor_f32_kernel()
            && block_rows < m
            && assoc_threads > 1
            && proj_threads > 1,
    );

    py.detach(|| {
        let total_t0 = Instant::now();
        let proj_nanos = AtomicU64::new(0);
        let assoc_nanos = AtomicU64::new(0);
        run_rotate_finalize_double_buffer_f32(
            m,
            n,
            block_rows,
            use_pipeline,
            |start, rows, rot_block| {
                let snp_block = &snp_flat[start * n..(start + rows) * n];
                let pt0 = Instant::now();
                rotate_snp_block_with_ut_blas(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    proj_threads,
                    proj_pool.as_ref(),
                );
                record_elapsed_nanos(&proj_nanos, pt0);
            },
            |start, rows, rot_block| {
                let out_block = &mut out_slice[start * out_cols..(start + rows) * out_cols];
                let at0 = Instant::now();
                assoc_fixed_lambda_rot_block_blas_f32(
                    rot_block,
                    rows,
                    n,
                    p,
                    cache_arc.as_ref(),
                    out_block,
                    out_cols,
                    assoc_threads,
                    assoc_pool.as_ref(),
                    nullml,
                );
                record_elapsed_nanos(&assoc_nanos, at0);
            },
        );
        maybe_emit_fvlmm_chunk_profile(
            "from_snp_cache",
            m,
            n,
            p,
            0,
            proj_nanos.load(Ordering::Relaxed),
            assoc_nanos.load(Ordering::Relaxed),
            elapsed_nanos_since(total_t0),
            threads,
            proj_threads,
            assoc_threads,
        );
    });

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd, snp_chunk, u_t, threads=0, nullml=None, rotate_block_rows=512))]
pub fn fvlmm_assoc_chunk_from_snp_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let with_plrt = nullml.is_some();
    let out_cols = if with_plrt { 4 } else { 3 };
    let m = snp_arr.shape()[0];
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };
    let p = p_cov;
    let prep_t0 = Instant::now();
    let cache = prepare_fixed_lambda_assoc_cache_f32(s, xcov_slice, y, n, p, lbd)?;
    let prep_nanos = elapsed_nanos_since(prep_t0);
    let proj_threads = stage_proj_threads_or(threads);
    let assoc_threads = stage_assoc_threads_or(threads);
    let proj_pool = get_cached_pool(proj_threads)?;
    let assoc_pool = get_cached_pool(assoc_threads)?;
    let block_rows = rotate_block_rows.max(1);
    let use_pipeline = use_rotate_finalize_pipeline(
        !assoc_rotate_prefers_rayon_rowmajor_f32_kernel()
            && block_rows < m
            && assoc_threads > 1
            && proj_threads > 1,
    );

    py.detach(|| {
        let total_t0 = Instant::now();
        let proj_nanos = AtomicU64::new(0);
        let assoc_nanos = AtomicU64::new(0);
        run_rotate_finalize_double_buffer_f32(
            m,
            n,
            block_rows,
            use_pipeline,
            |start, rows, rot_block| {
                let snp_block = &snp_flat[start * n..(start + rows) * n];
                let pt0 = Instant::now();
                rotate_snp_block_with_ut_blas(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    proj_threads,
                    proj_pool.as_ref(),
                );
                record_elapsed_nanos(&proj_nanos, pt0);
            },
            |start, rows, rot_block| {
                let out_block = &mut out_slice[start * out_cols..(start + rows) * out_cols];
                let at0 = Instant::now();
                assoc_fixed_lambda_rot_block_blas_f32(
                    rot_block,
                    rows,
                    n,
                    p,
                    &cache,
                    out_block,
                    out_cols,
                    assoc_threads,
                    assoc_pool.as_ref(),
                    nullml,
                );
                record_elapsed_nanos(&assoc_nanos, at0);
            },
        );
        maybe_emit_fvlmm_chunk_profile(
            "from_snp_raw",
            m,
            n,
            p,
            prep_nanos,
            proj_nanos.load(Ordering::Relaxed),
            assoc_nanos.load(Ordering::Relaxed),
            elapsed_nanos_since(total_t0),
            threads,
            proj_threads,
            assoc_threads,
        );
    });

    Ok(out)
}

/// FvLMM association from raw SNP chunk (not pre-rotated) that writes TSV text directly.
///
/// Takes centered genotype `snp_chunk` (m × n) and eigenvector matrix `u_t` (n × n),
/// does rotate+assoc via double-buffer BLAS pipeline, and returns pre-formatted TSV
/// text. This eliminates the numpy output array allocation and Python-side formatting
/// overhead compared to `fvlmm_assoc_chunk_from_snp_f32`.
#[pyfunction]
#[pyo3(signature = (
    s, xcov, y_rot, log10_lbd, snp_chunk, u_t,
    chrom, pos, snp, allele0, allele1, maf, miss,
    threads=0, nullml=None, rotate_block_rows=512,
    progress_callback=None, progress_every=0
))]
pub fn fvlmm_assoc_chunk_from_snp_to_tsv_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    maf: Vec<f32>,
    miss: Vec<f32>,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Vec<Vec<u8>>, usize)> {
    let s_slice = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s_slice.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let m = snp_arr.shape()[0];
    if m == 0 {
        return Ok((Vec::new(), 0));
    }
    if chrom.len() != m
        || pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
        || maf.len() != m
        || miss.len() != m
    {
        return Err(PyRuntimeError::new_err(
            "TSV metadata length mismatch with snp_chunk rows",
        ));
    }

    let with_plrt = nullml.is_some();
    let out_cols = if with_plrt { 4 } else { 3 };

    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };
    let p = p_cov;
    let cache = prepare_fixed_lambda_assoc_cache_f32(s_slice, xcov_slice, y, n, p, lbd)?;
    let proj_threads = stage_proj_threads_or(threads);
    let assoc_threads = stage_assoc_threads_or(threads);
    let proj_pool = get_cached_pool(proj_threads)?;
    let assoc_pool = get_cached_pool(assoc_threads)?;
    let block_rows = rotate_block_rows.max(1);
    let use_pipeline = use_rotate_finalize_pipeline(
        !assoc_rotate_prefers_rayon_rowmajor_f32_kernel()
            && block_rows < m
            && assoc_threads > 1
            && proj_threads > 1,
    );

    let progress_block = if progress_every == 0 {
        m.max(1)
    } else {
        progress_every.max(1)
    };

    // Owned copies for 'static lifetime in py.detach
    let chrom_owned = chrom.clone();
    let pos_owned = pos.clone();
    let snp_owned = snp.clone();
    let allele0_owned = allele0.clone();
    let allele1_owned = allele1.clone();
    let maf_owned = maf.clone();
    let miss_owned = miss.clone();

    py.detach(move || {
        let mut out_block = vec![0.0_f64; block_rows * out_cols];
        let mut text_buf = String::with_capacity(block_rows * 128);
        let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(m.div_ceil(block_rows));
        let mut next_progress_emit = progress_block.min(m).max(1);
        let mut done = 0usize;

        run_rotate_finalize_double_buffer_f32(
            m,
            n,
            block_rows,
            use_pipeline,
            |start, rows, rot_block| {
                let snp_block = &snp_flat[start * n..(start + rows) * n];
                rotate_snp_block_with_ut_blas(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    proj_threads,
                    proj_pool.as_ref(),
                );
            },
            |start, rows, rot_block| {
                let out_sub = &mut out_block[..rows * out_cols];
                assoc_fixed_lambda_rot_block_blas_f32(
                    rot_block,
                    rows,
                    n,
                    p,
                    &cache,
                    out_sub,
                    out_cols,
                    assoc_threads,
                    assoc_pool.as_ref(),
                    nullml,
                );

                text_buf.clear();
                for off in 0..rows {
                    let idx = start + off;
                    let base = off * out_cols;
                    let beta = out_sub[base];
                    let se = out_sub[base + 1];
                    let pwald = sanitize_assoc_pvalue(beta, se, out_sub[base + 2]);
                    let chisq_val = chisq_from_beta_se_and_optional_plrt(
                        beta,
                        se,
                        if with_plrt {
                            Some(out_sub[base + 3])
                        } else {
                            None
                        },
                    );
                    let chisq_txt = format_chisq_value(chisq_val);
                    if with_plrt {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                            chrom_owned[idx],
                            pos_owned[idx],
                            snp_owned[idx],
                            allele0_owned[idx],
                            allele1_owned[idx],
                            maf_owned[idx],
                            (miss_owned[idx] * n as f32) as i64,
                            beta,
                            se,
                            chisq_txt,
                            pwald,
                            out_sub[base + 3],
                        );
                    } else {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\n",
                            chrom_owned[idx],
                            pos_owned[idx],
                            snp_owned[idx],
                            allele0_owned[idx],
                            allele1_owned[idx],
                            maf_owned[idx],
                            (miss_owned[idx] * n as f32) as i64,
                            beta,
                            se,
                            chisq_txt,
                            pwald,
                        );
                    }
                }
                blocks.push(std::mem::take(&mut text_buf).into_bytes());
                done = start + rows;

                if progress_every > 0 && done >= next_progress_emit {
                    if let Some(ref cb) = progress_callback {
                        let _ = Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done.min(m), m))?;
                            Ok(())
                        });
                    }
                    next_progress_emit = (done / progress_block + 1)
                        .saturating_mul(progress_block)
                        .min(m);
                }
            },
        );

        if let Some(ref cb) = progress_callback {
            if done >= m {
                let _ = Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (m, m))?;
                    Ok(())
                });
            }
        }
        Ok((blocks, m))
    })
}

/// FvLMM association directly from a PLINK BED file to TSV.
///
/// Opens the BED file via mmap, scans all SNPs applying MAF/missing/het filters,
/// then runs the full triple-buffer pipeline (decode → rotate → assoc → TSV write)
/// in Rust with zero Python roundtrips per chunk.
///
/// Single-entry Rust BED -> TSV path for LMM REML association with optional
/// windowed mmap. This mirrors the FvLMM unified scan path so default memmap
/// LMM can avoid the Python chunk loop and associated file-backed RSS growth.
#[pyfunction]
#[pyo3(signature = (
    bed_prefix,
    out_tsv,
    s, xcov, y_rot, u_t,
    maf_thr, miss_thr, het_thr,
    genetic_model = "add",
    snps_only = false,
    sample_ids = None,
    low = -5.0,
    high = 5.0,
    max_iter = 30,
    tol = 1e-2,
    threads = 0,
    nullml = None,
    init_log10_lbd = None,
    rotate_block_rows = 512,
    progress_callback = None,
    progress_every = 0,
    mmap_window_mb = None,
))]
pub fn lmm_reml_assoc_bed_to_tsv_f32<'py>(
    py: Python<'py>,
    bed_prefix: &str,
    out_tsv: &str,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    u_t: PyReadonlyArray2<'py, f32>,
    maf_thr: f32,
    miss_thr: f32,
    het_thr: f32,
    genetic_model: &str,
    snps_only: bool,
    sample_ids: Option<Vec<String>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
    init_log10_lbd: Option<f64>,
    rotate_block_rows: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<usize> {
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }

    let s_slice = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let ut_arr = u_t.as_array();

    let n = y.len();
    let p = xcov_arr.shape()[1];
    if xcov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s_slice.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("u_t must be (n, n) row-major U^T"));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err("n must be > p+1"));
    }

    let gm = PackedGeneticModel::parse(genetic_model)?;
    let with_plrt = nullml.is_some();
    let out_cols = if with_plrt { 4usize } else { 3usize };
    let nullml_val = nullml.unwrap_or(0.0);
    let init_log10_lbd = init_log10_lbd
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(low, high));

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let bed_prefix_owned = bed_prefix.to_string();
    let out_tsv_owned = out_tsv.to_string();
    let stage_timing = env_truthy("JX_LMM_UNIFIED_STAGE_TIMING");
    let use_warm_start = !env_truthy("JX_LMM_UNIFIED_NO_WARM_START");

    py.detach(move || -> Result<usize, String> {
        let total_t0 = Instant::now();
        let count_nanos = AtomicU64::new(0);
        let meta_nanos = AtomicU64::new(0);
        let decode_nanos = AtomicU64::new(0);
        let proj_nanos = AtomicU64::new(0);
        let assoc_nanos = AtomicU64::new(0);
        let tsv_nanos = AtomicU64::new(0);

        let full_samples = gfcore::read_fam(&bed_prefix_owned)?;
        let n_samples_full = full_samples.len();
        if n_samples_full == 0 {
            return Err("no samples in PLINK FAM".to_string());
        }

        let (all_chrom, all_pos, all_snp, all_allele0, all_allele1) =
            read_bim_columns(&bed_prefix_owned, None)?;

        let sample_idx: Vec<usize> = match &sample_ids {
            Some(ids) => {
                let id_to_idx: std::collections::HashMap<&str, usize> = full_samples
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.as_str(), i))
                    .collect();
                ids.iter()
                    .map(|sid| {
                        id_to_idx
                            .get(sid.as_str())
                            .copied()
                            .ok_or_else(|| format!("sample '{sid}' not found in PLINK FAM"))
                    })
                    .collect::<Result<Vec<usize>, String>>()?
            }
            None => (0..n_samples_full).collect(),
        };
        if sample_idx.len() != n {
            return Err(format!(
                "sample_ids length {} != len(y_rot) {}",
                sample_idx.len(),
                n
            ));
        }

        let bytes_per_snp = (n_samples_full + 3) / 4;
        if bytes_per_snp == 0 {
            return Err("bytes_per_snp is zero".to_string());
        }
        let bed_path = format!("{bed_prefix_owned}.bed");
        let mut bed_window = if let Some(window_mb) = mmap_window_mb.filter(|&v| v > 0) {
            Some(WindowedBedMatrix::open(&bed_prefix_owned, window_mb)?)
        } else {
            None
        };
        let full_mmap = if bed_window.is_none() {
            let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
            let mmap =
                unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
            if mmap.len() < 3 || mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
                return Err("only SNP-major BED supported".to_string());
            }
            let data_len = mmap.len() - 3;
            if data_len % bytes_per_snp != 0 {
                return Err(format!(
                    "BED payload length {data_len} not a multiple of {bytes_per_snp}"
                ));
            }
            Some(mmap)
        } else {
            None
        };
        let n_snps = if let Some(window) = bed_window.as_ref() {
            window.n_source_snps()
        } else {
            let mmap = full_mmap
                .as_ref()
                .ok_or_else(|| "internal error: missing BED source".to_string())?;
            (mmap.len() - 3) / bytes_per_snp
        };

        if all_chrom.len() != n_snps {
            return Err(format!(
                "BIM site count {} != BED SNP count {n_snps}",
                all_chrom.len()
            ));
        }

        let use_selected =
            !sample_indices_are_identity(&sample_idx) || sample_idx.len() != n_samples_full;
        let sample_identity = !use_selected && sample_idx.len() == n_samples_full;
        let selected_excluded_sample_indices = if use_selected {
            precompute_excluded_sample_indices(n_samples_full, &sample_idx)
        } else {
            None
        };
        let sample_subset_plan: Option<SubsetDecodePlan> = if sample_identity {
            None
        } else {
            Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
                &sample_idx,
                n_samples_full,
            ))
        };
        let dense_subset_pos: Option<Vec<isize>> = if sample_identity {
            None
        } else {
            maybe_build_dense_subset_pos(&sample_idx, n_samples_full)
        };

        let proj_threads = stage_proj_threads_or(threads);
        let assoc_threads = stage_assoc_threads_or(threads);
        let proj_pool = get_cached_pool(proj_threads).map_err(|e| format!("{e}"))?;
        let assoc_pool = get_cached_pool(assoc_threads).map_err(|e| format!("{e}"))?;

        let header: &[u8] = if with_plrt {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n"
        } else {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n"
        };
        let writer = AsyncTsvWriter::with_config(&out_tsv_owned, header, 64 * 1024 * 1024, 16)?;

        let scan_chunk_snps = rotate_block_rows.max(1024).min(65536);
        let mut text_buf = String::with_capacity(scan_chunk_snps * 128);
        let mut total_rows = 0usize;

        let progress_block = if progress_every == 0 {
            0usize
        } else {
            progress_every.max(1)
        };
        let mut next_progress_emit = if progress_block > 0 {
            progress_block.min(n_snps).max(1)
        } else {
            0
        };

        let run_rows = |g_block: &[f32], rows: usize, out_block: &mut [f64]| {
            let mut run = || {
                out_block[..rows * out_cols]
                    .par_chunks_mut(out_cols)
                    .enumerate()
                    .for_each_init(
                        || (vec![0.0_f64; n], init_log10_lbd),
                        |state, (idx, out_row)| {
                            let (snp_vec, last_log10_lbd) = state;
                            let row = &g_block[idx * n..(idx + 1) * n];
                            for i in 0..n {
                                snp_vec[i] = row[i] as f64;
                            }

                            let (best_log10_lbd, _best_cost) = if use_warm_start {
                                let init_guess = (*last_log10_lbd).or(init_log10_lbd);
                                let result = brent_minimize_with_init(
                                    |x0| {
                                        -reml_loglike(
                                            x0,
                                            s_slice,
                                            &xcov_flat,
                                            y,
                                            Some(&snp_vec[..]),
                                            n,
                                            p,
                                        )
                                    },
                                    low,
                                    high,
                                    tol,
                                    max_iter,
                                    init_guess,
                                );
                                *last_log10_lbd = Some(result.0);
                                result
                            } else {
                                brent_minimize(
                                    |x0| {
                                        -reml_loglike(
                                            x0,
                                            s_slice,
                                            &xcov_flat,
                                            y,
                                            Some(&snp_vec[..]),
                                            n,
                                            p,
                                        )
                                    },
                                    low,
                                    high,
                                    tol,
                                    max_iter,
                                )
                            };

                            let (beta, se, _lbd) = final_beta_se(
                                best_log10_lbd,
                                s_slice,
                                &xcov_flat,
                                y,
                                &snp_vec[..],
                                n,
                                p,
                            );
                            let pval = if beta.is_finite() && se.is_finite() && se > 0.0 {
                                let z = beta / se;
                                (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
                            } else {
                                1.0
                            };

                            out_row[0] = beta;
                            out_row[1] = se;
                            out_row[2] = if pval.is_finite() { pval } else { 1.0 };

                            if with_plrt {
                                let ml = ml_loglike(
                                    best_log10_lbd,
                                    s_slice,
                                    &xcov_flat,
                                    y,
                                    Some(&snp_vec[..]),
                                    n,
                                    p,
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

            if let Some(tp) = assoc_pool.as_ref() {
                tp.install(run);
            } else {
                run();
            }
        };

        let mut chunk_start = 0usize;
        let producer_err = Arc::new(OnceLock::<String>::new());
        let producer_err_bg = Arc::clone(&producer_err);
        let producer = |chunk: &mut StreamingChunk| -> bool {
            if chunk_start >= n_snps {
                return false;
            }
            chunk.clear();
            let chunk_end = (chunk_start + scan_chunk_snps).min(n_snps);
            let chunk_packed = match bed_window.as_mut() {
                Some(window) => match window.read_source_range(chunk_start, chunk_end) {
                    Ok(slice) => slice,
                    Err(e) => {
                        let _ = producer_err_bg.set(e);
                        return false;
                    }
                },
                None => {
                    let mmap = match full_mmap.as_ref() {
                        Some(mmap) => mmap,
                        None => {
                            let _ = producer_err_bg
                                .set("internal error: missing full BED mmap".to_string());
                            return false;
                        }
                    };
                    let packed = &mmap[3..];
                    let start_byte = chunk_start * bytes_per_snp;
                    let end_byte = chunk_end * bytes_per_snp;
                    &packed[start_byte..end_byte]
                }
            };

            let count_t0 = Instant::now();
            let counts: Vec<Option<SnpCounts>> = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|snp_idx| {
                    let local_idx = snp_idx - chunk_start;
                    let row =
                        &chunk_packed[local_idx * bytes_per_snp..(local_idx + 1) * bytes_per_snp];
                    let (missing, het, hom_alt) = if use_selected {
                        count_packed_row_counts_selected_with_excluded(
                            row,
                            n_samples_full,
                            &sample_idx,
                            selected_excluded_sample_indices.as_deref(),
                        )
                    } else {
                        count_packed_row_counts(row, n_samples_full)
                    };
                    let non_missing = n.saturating_sub(missing);

                    let miss_rate = if n > 0 {
                        missing as f32 / n as f32
                    } else {
                        1.0_f32
                    };
                    if miss_rate > miss_thr {
                        return None;
                    }

                    if non_missing == 0 {
                        return if maf_thr > 0.0 {
                            None
                        } else {
                            Some(SnpCounts {
                                flip: false,
                                maf: 0.0_f32,
                                miss_rate,
                                missing_count: missing,
                            })
                        };
                    }

                    if het_thr > 0.0 {
                        let het_rate = het as f32 / non_missing as f32;
                        if het_rate > het_thr {
                            return None;
                        }
                    }

                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                    let alt_freq = alt_sum as f32 / (2.0_f32 * non_missing as f32);
                    let maf_v = alt_freq.min(1.0_f32 - alt_freq);
                    if maf_v < maf_thr {
                        return None;
                    }

                    Some(SnpCounts {
                        flip: false,
                        maf: alt_freq,
                        miss_rate,
                        missing_count: missing,
                    })
                })
                .collect();
            record_elapsed_nanos(&count_nanos, count_t0);

            let meta_t0 = Instant::now();
            for (offset, snp_idx) in (chunk_start..chunk_end).enumerate() {
                let cnts = match &counts[offset] {
                    Some(c) => c,
                    None => continue,
                };
                if snps_only
                    && (!is_simple_snp_allele(&all_allele0[snp_idx])
                        || !is_simple_snp_allele(&all_allele1[snp_idx]))
                {
                    continue;
                }
                chunk.indices.push(offset);
                chunk.flip.push(cnts.flip);
                chunk.maf.push(cnts.maf);
                chunk.miss_rate.push(cnts.miss_rate);
                chunk.miss_block[chunk.rows] = cnts.missing_count;
                chunk.chrom.push(all_chrom[snp_idx].clone());
                chunk.pos.push(all_pos[snp_idx] as i64);
                chunk.snp.push(resolve_snp_name(
                    &all_snp[snp_idx],
                    &all_chrom[snp_idx],
                    all_pos[snp_idx],
                ));
                chunk.a0.push(all_allele0[snp_idx].clone());
                chunk.a1.push(all_allele1[snp_idx].clone());
                chunk.rows += 1;
            }
            record_elapsed_nanos(&meta_nanos, meta_t0);

            if chunk.rows > 0 {
                let decode_t0 = Instant::now();
                decode_centered_block_packed_f32(
                    chunk_packed,
                    bytes_per_snp,
                    &chunk.flip,
                    &chunk.maf,
                    Some(&chunk.indices),
                    0,
                    chunk.rows,
                    if sample_identity { n_samples_full } else { n },
                    gm,
                    sample_identity,
                    sample_subset_plan.as_ref(),
                    dense_subset_pos.as_deref(),
                    &mut chunk.g_block[..chunk.rows * n],
                );
                record_elapsed_nanos(&decode_nanos, decode_t0);
            }

            chunk.scanned_to = chunk_end;
            chunk_start = chunk_end;
            chunk_start < n_snps
        };

        let consumer = |chunk: &mut StreamingChunk| {
            let rows = chunk.rows;
            if rows > 0 {
                let proj_t0 = Instant::now();
                rotate_snp_block_with_ut_blas(
                    &chunk.g_block[..rows * n],
                    rows,
                    n,
                    &ut_flat,
                    &mut chunk.rot_block[..rows * n],
                    proj_threads,
                    proj_pool.as_ref(),
                );
                record_elapsed_nanos(&proj_nanos, proj_t0);
                let assoc_t0 = Instant::now();
                run_rows(
                    &chunk.rot_block[..rows * n],
                    rows,
                    &mut chunk.out_block[..rows * out_cols],
                );
                record_elapsed_nanos(&assoc_nanos, assoc_t0);

                let tsv_t0 = Instant::now();
                text_buf.clear();
                if text_buf.capacity() < rows * 128 {
                    text_buf.reserve(rows * 128 - text_buf.capacity());
                }
                for off in 0..rows {
                    let base = off * out_cols;
                    let beta = chunk.out_block[base];
                    let se = chunk.out_block[base + 1];
                    let pwald = sanitize_assoc_pvalue(beta, se, chunk.out_block[base + 2]);
                    let chisq = chisq_from_beta_se_and_optional_plrt(
                        beta,
                        se,
                        if with_plrt {
                            Some(chunk.out_block[base + 3])
                        } else {
                            None
                        },
                    );
                    let chisq_txt = format_chisq_value(chisq);
                    if with_plrt {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                            chunk.chrom[off],
                            chunk.pos[off],
                            chunk.snp[off],
                            chunk.a0[off],
                            chunk.a1[off],
                            chunk.maf[off],
                            chunk.miss_block[off],
                            beta,
                            se,
                            chisq_txt,
                            pwald,
                            chunk.out_block[base + 3],
                        );
                    } else {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\n",
                            chunk.chrom[off],
                            chunk.pos[off],
                            chunk.snp[off],
                            chunk.a0[off],
                            chunk.a1[off],
                            chunk.maf[off],
                            chunk.miss_block[off],
                            beta,
                            se,
                            chisq_txt,
                            pwald,
                        );
                    }
                }
                let payload = std::mem::take(&mut text_buf).into_bytes();
                writer.send(payload)?;
                record_elapsed_nanos(&tsv_nanos, tsv_t0);
            }

            total_rows += rows;
            if progress_block > 0 && chunk.scanned_to >= next_progress_emit {
                if let Some(ref cb) = progress_callback {
                    let _ = Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (chunk.scanned_to.min(n_snps), n_snps))?;
                        Ok(())
                    });
                }
                next_progress_emit = (chunk.scanned_to / progress_block + 1)
                    .saturating_mul(progress_block)
                    .min(n_snps);
            }

            Ok::<(), String>(())
        };

        let pipeline_t0 = Instant::now();
        crate::pipeline::run_double_buffer(
            2,
            || StreamingChunk::new(scan_chunk_snps, n, out_cols),
            producer,
            consumer,
        )?;
        let pipeline_secs = pipeline_t0.elapsed().as_secs_f64();
        if let Some(err) = producer_err.get() {
            return Err(err.clone());
        }

        if let Some(ref cb) = progress_callback {
            let _ = Python::attach(|py2| -> PyResult<()> {
                py2.check_signals()?;
                cb.call1(py2, (n_snps, n_snps))?;
                Ok(())
            });
        }

        let writer_t0 = Instant::now();
        writer.finish()?;
        record_elapsed_nanos(&tsv_nanos, writer_t0);

        if stage_timing {
            let total_secs = total_t0.elapsed().as_secs_f64();
            let count_secs = elapsed_nanos_to_secs(&count_nanos);
            let meta_secs = elapsed_nanos_to_secs(&meta_nanos);
            let decode_secs = elapsed_nanos_to_secs(&decode_nanos);
            let proj_secs = elapsed_nanos_to_secs(&proj_nanos);
            let assoc_secs = elapsed_nanos_to_secs(&assoc_nanos);
            let tsv_secs = elapsed_nanos_to_secs(&tsv_nanos);
            let to_pct = |x: f64| -> f64 {
                if total_secs > 0.0 {
                    x * 100.0 / total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "LMM unified timing: count_qc={:.3}s ({:.1}%), metadata={:.3}s ({:.1}%), decode={:.3}s ({:.1}%), project={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), tsv={:.3}s ({:.1}%), pipeline_wall={:.3}s, total={:.3}s, rows={}, snps={}, n={}, chunk={}, proj_threads={}, assoc_threads={}",
                count_secs,
                to_pct(count_secs),
                meta_secs,
                to_pct(meta_secs),
                decode_secs,
                to_pct(decode_secs),
                proj_secs,
                to_pct(proj_secs),
                assoc_secs,
                to_pct(assoc_secs),
                tsv_secs,
                to_pct(tsv_secs),
                pipeline_secs,
                total_secs,
                total_rows,
                n_snps,
                n,
                scan_chunk_snps,
                proj_threads,
                assoc_threads,
            );
        }
        Ok(total_rows)
    })
    .map_err(PyRuntimeError::new_err)
}

/// This eliminates the Python chunk loop and ThreadPoolExecutor overhead for FvLMM.
#[pyfunction]
#[pyo3(signature = (
    bed_prefix,
    out_tsv,
    s, xcov, y_rot, log10_lbd, u_t,
    maf_thr, miss_thr, het_thr,
    genetic_model = "add",
    snps_only = false,
    sample_ids = None,
    threads = 0,
    nullml = None,
    rotate_block_rows = 512,
    progress_callback = None,
    progress_every = 0,
    mmap_window_mb = None,
))]
pub fn fvlmm_assoc_bed_to_tsv_f32<'py>(
    py: Python<'py>,
    bed_prefix: &str,
    out_tsv: &str,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    u_t: PyReadonlyArray2<'py, f32>,
    maf_thr: f32,
    miss_thr: f32,
    het_thr: f32,
    genetic_model: &str,
    snps_only: bool,
    sample_ids: Option<Vec<String>>,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<(usize, f64, f64)> {
    let s_slice = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let ut_arr = u_t.as_array();

    let n = y.len();
    let p = xcov_arr.shape()[1];
    if xcov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s_slice.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("u_t must be (n, n) row-major U^T"));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err("n must be > p+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let gm = PackedGeneticModel::parse(genetic_model)?;
    let with_plrt = nullml.is_some();
    let out_cols = if with_plrt { 4 } else { 3 };

    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let bed_prefix_owned = bed_prefix.to_string();
    let out_tsv_owned = out_tsv.to_string();

    let (rows_written, pve, log_det_v) = py
        .detach(move || -> Result<(usize, f64, f64), String> {
            // ============================================================
            // Setup: mmap BED, parse FAM/BIM, build sample index mapping
            // ============================================================
            let full_samples = gfcore::read_fam(&bed_prefix_owned)?;
            let n_samples_full = full_samples.len();
            if n_samples_full == 0 {
                return Err("no samples in PLINK FAM".to_string());
            }

            let (all_chrom, all_pos, all_snp, all_allele0, all_allele1) =
                read_bim_columns(&bed_prefix_owned, None)?;

            let sample_idx: Vec<usize> = match &sample_ids {
                Some(ids) => {
                    let id_to_idx: std::collections::HashMap<&str, usize> = full_samples
                        .iter()
                        .enumerate()
                        .map(|(i, name)| (name.as_str(), i))
                        .collect();
                    ids.iter()
                        .map(|sid| {
                            id_to_idx
                                .get(sid.as_str())
                                .copied()
                                .ok_or_else(|| format!("sample '{sid}' not found in PLINK FAM"))
                        })
                        .collect::<Result<Vec<usize>, String>>()?
                }
                None => (0..n_samples_full).collect(),
            };
            if sample_idx.len() != n {
                return Err(format!(
                    "sample_ids length {} != len(y_rot) {}",
                    sample_idx.len(),
                    n
                ));
            }

            let bytes_per_snp = (n_samples_full + 3) / 4;
            if bytes_per_snp == 0 {
                return Err("bytes_per_snp is zero".to_string());
            }
            let bed_path = format!("{bed_prefix_owned}.bed");
            let mut bed_window = if let Some(window_mb) = mmap_window_mb.filter(|&v| v > 0) {
                Some(WindowedBedMatrix::open(&bed_prefix_owned, window_mb)?)
            } else {
                None
            };
            let full_mmap = if bed_window.is_none() {
                let bed_file =
                    File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
                let mmap =
                    unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
                if mmap.len() < 3 || mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
                    return Err("only SNP-major BED supported".to_string());
                }
                let data_len = mmap.len() - 3;
                if data_len % bytes_per_snp != 0 {
                    return Err(format!(
                        "BED payload length {data_len} not a multiple of {bytes_per_snp}"
                    ));
                }
                Some(mmap)
            } else {
                None
            };
            let n_snps = if let Some(window) = bed_window.as_ref() {
                window.n_source_snps()
            } else {
                let mmap = full_mmap
                    .as_ref()
                    .ok_or_else(|| "internal error: missing BED source".to_string())?;
                (mmap.len() - 3) / bytes_per_snp
            };

            if all_chrom.len() != n_snps {
                return Err(format!(
                    "BIM site count {} != BED SNP count {n_snps}",
                    all_chrom.len()
                ));
            }

            let use_selected =
                !sample_indices_are_identity(&sample_idx) || sample_idx.len() != n_samples_full;
            let sample_identity = !use_selected && sample_idx.len() == n_samples_full;
            let selected_excluded_sample_indices = if use_selected {
                precompute_excluded_sample_indices(n_samples_full, &sample_idx)
            } else {
                None
            };

            // ============================================================
            // Build assoc cache and thread pools (once, upfront)
            // ============================================================
            let cache = prepare_fixed_lambda_assoc_cache_f32(s_slice, xcov_slice, y, n, p, lbd)
                .map_err(|e| format!("{e}"))?;
            let y_sq = y.iter().map(|&yi| yi * yi).sum::<f64>();
            let pve_out = if y_sq > 0.0 {
                (1.0 - cache.ypy / y_sq).clamp(0.0, 1.0)
            } else {
                f64::NAN
            };
            let log_det_v_out = cache.log_det_v;

            let proj_threads = stage_proj_threads_or(threads);
            let assoc_threads = stage_assoc_threads_or(threads);
            let proj_pool = get_cached_pool(proj_threads).map_err(|e| format!("{e}"))?;
            let assoc_pool = get_cached_pool(assoc_threads).map_err(|e| format!("{e}"))?;
            let sample_subset_plan: Option<SubsetDecodePlan> = if sample_identity {
                None
            } else {
                Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
                    &sample_idx,
                    n_samples_full,
                ))
            };
            let dense_subset_pos: Option<Vec<isize>> = if sample_identity {
                None
            } else {
                maybe_build_dense_subset_pos(&sample_idx, n_samples_full)
            };

            // ============================================================
            // TSV writer
            // ============================================================
            let header: &[u8] = if with_plrt {
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n"
            } else {
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n"
            };
            let writer = AsyncTsvWriter::with_config(&out_tsv_owned, header, 64 * 1024 * 1024, 16)?;

            // ============================================================
            // Double-buffer pipeline: count+decode (bg) || rotate+assoc+write (main)
            // Uses generic pipeline from src/io/pipeline.rs.
            // ============================================================
            let scan_chunk_snps = rotate_block_rows.max(1024).min(65536);

            let mut text_buf = String::with_capacity(scan_chunk_snps * 128);
            let mut total_rows = 0usize;

            let progress_block = if progress_every == 0 {
                0usize
            } else {
                progress_every.max(1)
            };
            let mut next_progress_emit = if progress_block > 0 {
                progress_block.min(n_snps).max(1)
            } else {
                0
            };

            // --- Producer: count (rayon) + filter + metadata + decode ---
            let mut chunk_start = 0usize;
            let producer_err = Arc::new(OnceLock::<String>::new());
            let producer_err_bg = Arc::clone(&producer_err);
            let producer = |chunk: &mut StreamingChunk| -> bool {
                if chunk_start >= n_snps {
                    return false;
                }
                chunk.clear();
                let chunk_end = (chunk_start + scan_chunk_snps).min(n_snps);
                let chunk_packed = match bed_window.as_mut() {
                    Some(window) => match window.read_source_range(chunk_start, chunk_end) {
                        Ok(slice) => slice,
                        Err(e) => {
                            let _ = producer_err_bg.set(e);
                            return false;
                        }
                    },
                    None => {
                        let mmap = match full_mmap.as_ref() {
                            Some(mmap) => mmap,
                            None => {
                                let _ = producer_err_bg
                                    .set("internal error: missing full BED mmap".to_string());
                                return false;
                            }
                        };
                        let packed = &mmap[3..];
                        let start_byte = chunk_start * bytes_per_snp;
                        let end_byte = chunk_end * bytes_per_snp;
                        &packed[start_byte..end_byte]
                    }
                };

                // ---- rayon parallel: per-SNP counting + numeric filter ----
                let counts: Vec<Option<SnpCounts>> = (chunk_start..chunk_end)
                    .into_par_iter()
                    .map(|snp_idx| {
                        let local_idx = snp_idx - chunk_start;
                        let row = &chunk_packed
                            [local_idx * bytes_per_snp..(local_idx + 1) * bytes_per_snp];
                        let (missing, het, hom_alt) = if use_selected {
                            count_packed_row_counts_selected_with_excluded(
                                row,
                                n_samples_full,
                                &sample_idx,
                                selected_excluded_sample_indices.as_deref(),
                            )
                        } else {
                            count_packed_row_counts(row, n_samples_full)
                        };
                        let non_missing = n.saturating_sub(missing);

                        let miss_rate = if n > 0 {
                            missing as f32 / n as f32
                        } else {
                            1.0_f32
                        };
                        if miss_rate > miss_thr {
                            return None;
                        }

                        if non_missing == 0 {
                            return if maf_thr > 0.0 {
                                None
                            } else {
                                Some(SnpCounts {
                                    flip: false,
                                    maf: 0.0_f32,
                                    miss_rate,
                                    missing_count: missing,
                                })
                            };
                        }

                        if het_thr > 0.0 {
                            let het_rate = het as f32 / non_missing as f32;
                            if het_rate > het_thr {
                                return None;
                            }
                        }

                        let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                        let alt_freq = alt_sum as f32 / (2.0_f32 * non_missing as f32);
                        let maf_v = alt_freq.min(1.0_f32 - alt_freq);

                        if maf_v < maf_thr {
                            return None;
                        }

                        Some(SnpCounts {
                            flip: false,
                            maf: alt_freq,
                            miss_rate,
                            missing_count: missing,
                        })
                    })
                    .collect();

                // ---- Sequential: snps_only filter + metadata collection ----
                for (offset, snp_idx) in (chunk_start..chunk_end).enumerate() {
                    let cnts = match &counts[offset] {
                        Some(c) => c,
                        None => continue,
                    };
                    if snps_only {
                        if !is_simple_snp_allele(&all_allele0[snp_idx])
                            || !is_simple_snp_allele(&all_allele1[snp_idx])
                        {
                            continue;
                        }
                    }
                    chunk.indices.push(offset);
                    chunk.flip.push(cnts.flip);
                    chunk.maf.push(cnts.maf);
                    chunk.miss_rate.push(cnts.miss_rate);
                    chunk.miss_block[chunk.rows] = cnts.missing_count;
                    chunk.chrom.push(all_chrom[snp_idx].clone());
                    chunk.pos.push(all_pos[snp_idx] as i64);
                    chunk.snp.push(resolve_snp_name(
                        &all_snp[snp_idx],
                        &all_chrom[snp_idx],
                        all_pos[snp_idx],
                    ));
                    chunk.a0.push(all_allele0[snp_idx].clone());
                    chunk.a1.push(all_allele1[snp_idx].clone());
                    chunk.rows += 1;
                }

                // ---- Decode ----
                if chunk.rows > 0 {
                    decode_centered_block_packed_f32(
                        chunk_packed,
                        bytes_per_snp,
                        &chunk.flip,
                        &chunk.maf,
                        Some(&chunk.indices),
                        0,
                        chunk.rows,
                        if sample_identity { n_samples_full } else { n },
                        gm,
                        sample_identity,
                        sample_subset_plan.as_ref(),
                        dense_subset_pos.as_deref(),
                        &mut chunk.g_block[..chunk.rows * n],
                    );
                }

                chunk.scanned_to = chunk_end;
                chunk_start = chunk_end;
                chunk_start < n_snps
            };

            // --- Consumer: Rotate + assoc + format TSV + write ---
            let consumer = |chunk: &mut StreamingChunk| {
                let rows = chunk.rows;
                if rows > 0 {
                    // Rotate – large SGEMM
                    rotate_snp_block_with_ut_blas(
                        &chunk.g_block[..rows * n],
                        rows,
                        n,
                        &ut_flat,
                        &mut chunk.rot_block[..rows * n],
                        proj_threads,
                        proj_pool.as_ref(),
                    );

                    // Assoc – large SGEMM
                    assoc_fixed_lambda_rot_block_blas_f32(
                        &chunk.rot_block[..rows * n],
                        rows,
                        n,
                        p,
                        &cache,
                        &mut chunk.out_block[..rows * out_cols],
                        out_cols,
                        assoc_threads,
                        assoc_pool.as_ref(),
                        nullml,
                    );

                    // Format TSV and write
                    text_buf.clear();
                    if text_buf.capacity() < rows * 128 {
                        text_buf.reserve(rows * 128 - text_buf.capacity());
                    }
                    for off in 0..rows {
                        let base = off * out_cols;
                        let beta = chunk.out_block[base];
                        let se = chunk.out_block[base + 1];
                        let pwald = sanitize_assoc_pvalue(beta, se, chunk.out_block[base + 2]);
                        let chisq_val = chisq_from_beta_se_and_optional_plrt(
                            beta,
                            se,
                            if with_plrt {
                                Some(chunk.out_block[base + 3])
                            } else {
                                None
                            },
                        );
                        let chisq_txt = format_chisq_value(chisq_val);
                        if with_plrt {
                            let _ = write!(
                                text_buf,
                                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                                chunk.chrom[off],
                                chunk.pos[off],
                                chunk.snp[off],
                                chunk.a0[off],
                                chunk.a1[off],
                                chunk.maf[off],
                                chunk.miss_block[off],
                                beta,
                                se,
                                chisq_txt,
                                pwald,
                                chunk.out_block[base + 3],
                            );
                        } else {
                            let _ = write!(
                                text_buf,
                                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\n",
                                chunk.chrom[off],
                                chunk.pos[off],
                                chunk.snp[off],
                                chunk.a0[off],
                                chunk.a1[off],
                                chunk.maf[off],
                                chunk.miss_block[off],
                                beta,
                                se,
                                chisq_txt,
                                pwald,
                            );
                        }
                    }
                    let payload = std::mem::take(&mut text_buf).into_bytes();
                    writer.send(payload)?;
                }

                total_rows += rows;

                // Progress callback
                if progress_block > 0 && chunk.scanned_to >= next_progress_emit {
                    if let Some(ref cb) = progress_callback {
                        let _ = Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (chunk.scanned_to.min(n_snps), n_snps))?;
                            Ok(())
                        });
                    }
                    next_progress_emit = (chunk.scanned_to / progress_block + 1)
                        .saturating_mul(progress_block)
                        .min(n_snps);
                }

                Ok::<(), String>(())
            };

            crate::pipeline::run_double_buffer(
                2,
                || StreamingChunk::new(scan_chunk_snps, n, out_cols),
                producer,
                consumer,
            )?;
            if let Some(err) = producer_err.get() {
                return Err(err.clone());
            }

            if let Some(ref cb) = progress_callback {
                let _ = Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (n_snps, n_snps))?;
                    Ok(())
                });
            }

            writer.finish()?;
            Ok((total_rows, pve_out, log_det_v_out))
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok((rows_written, pve, log_det_v))
}

/// Resolve SNP name: use the bim name if non-empty and not ".", else chrom_pos.
fn resolve_snp_name(snp: &str, chrom: &str, pos: i32) -> String {
    if snp.is_empty() || snp == "." {
        format!("{chrom}_{pos}")
    } else {
        snp.to_string()
    }
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

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    s,
    xcov,
    y_rot,
    u_t,
    sample_indices=None,
    row_indices=None,
    low=-5.0,
    high=5.0,
    max_iter=50,
    tol=1e-2,
    threads=0,
    model="add",
    progress_callback=None,
    progress_every=0,
    nullml=None,
    init_log10_lbd=None,
    rotate_block_rows=256
))]
pub fn lmm_reml_assoc_packed_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    u_t: PyReadonlyArray2<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    model: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    nullml: Option<f64>,
    init_log10_lbd: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }
    let gm = PackedGeneticModel::parse(model)?;
    let init_log10_lbd = init_log10_lbd
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(low, high));

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m_packed = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_idx: Option<Vec<usize>> = if let Some(ridx) = row_indices {
        Some(parse_index_vec_i64(
            ridx.as_slice()?,
            m_packed,
            "row_indices",
        )?)
    } else {
        None
    };
    let m = row_idx.as_ref().map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let y = y_rot.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y_rot must not be empty"));
    }
    let s = s.as_slice()?;
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        if sidx_slice.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                sidx_slice.len()
            )));
        }
        parse_index_vec_i64(sidx_slice, n_samples, "sample_indices")?
    } else {
        if n != n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "len(y_rot)={} must equal n_samples={} when sample_indices is not provided",
                n, n_samples
            )));
        }
        (0..n_samples).collect()
    };
    let sample_identity = sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
    let sample_subset_plan: Option<SubsetDecodePlan> = if sample_identity {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
            &sample_idx,
            n_samples,
        ))
    };
    let dense_subset_pos: Option<Vec<isize>> = if sample_identity {
        None
    } else {
        maybe_build_dense_subset_pos(&sample_idx, n_samples)
    };

    let xcov_arr = xcov.as_array();
    if xcov_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("xcov must be 2D (n, p_cov)"));
    }
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("xcov.n_rows must equal len(y_rot)"));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let ut_arr = u_t.as_array();
    if ut_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("u_t must be 2D (n, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    let pool = get_cached_pool(threads)?;
    let block_rows = rotate_block_rows.max(1);
    let progress_block = if progress_every == 0 {
        block_rows.max(1)
    } else {
        progress_every.max(1)
    };
    let stage_timing = env_truthy("JX_LMM_PACKED_STAGE_TIMING");

    py.detach(move || -> PyResult<()> {
        let total_t0 = Instant::now();
        let mut decode_secs = 0.0_f64;
        let mut proj_secs = 0.0_f64;
        let mut assoc_secs = 0.0_f64;

        let mut snp_buf = vec![0.0_f32; block_rows * n];
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            out_block
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || (vec![0.0_f64; n], init_log10_lbd),
                    |state, (idx, out_row)| {
                        let (snp_vec, last_log10_lbd) = state;
                        let row = &g_block[idx * n..(idx + 1) * n];
                        for i in 0..n {
                            snp_vec[i] = row[i] as f64;
                        }

                        let init_guess = (*last_log10_lbd).or(init_log10_lbd);
                        let (best_log10_lbd, _best_cost) = brent_minimize_with_init(
                            |x0| -reml_loglike(x0, s, &xcov_flat, y, Some(&snp_vec[..]), n, p_cov),
                            low,
                            high,
                            tol,
                            max_iter,
                            init_guess,
                        );
                        *last_log10_lbd = Some(best_log10_lbd);

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

        for row_start in (0..m).step_by(progress_block) {
            let row_end = (row_start + progress_block).min(m);
            let mut start = row_start;
            while start < row_end {
                let rows = (row_end - start).min(block_rows);
                let snp_block = &mut snp_buf[..rows * n];
                let rot_block = &mut rot_buf[..rows * n];

                let dt0 = Instant::now();
                decode_centered_block_packed_f32(
                    &packed_flat,
                    bytes_per_snp,
                    row_flip,
                    row_maf,
                    row_idx.as_deref(),
                    start,
                    rows,
                    n,
                    gm,
                    sample_identity,
                    sample_subset_plan.as_ref(),
                    dense_subset_pos.as_deref(),
                    snp_block,
                );
                decode_secs += dt0.elapsed().as_secs_f64();

                let rotate_tile_rows = choose_rotate_tile_rows(
                    rows,
                    if threads > 0 {
                        threads
                    } else {
                        rayon::current_num_threads()
                    },
                );
                let out_block = &mut out_slice[start * out_cols..(start + rows) * out_cols];

                let pt0 = Instant::now();
                if let Some(tp) = &pool {
                    tp.install(|| {
                        rotate_snp_block_with_ut_parallel(
                            snp_block,
                            rows,
                            n,
                            &ut_flat,
                            rot_block,
                            rotate_tile_rows,
                        );
                    });
                } else {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                }
                proj_secs += pt0.elapsed().as_secs_f64();

                let at0 = Instant::now();
                if let Some(tp) = &pool {
                    tp.install(|| run_rows(rot_block, out_block));
                } else {
                    run_rows(rot_block, out_block);
                }
                assoc_secs += at0.elapsed().as_secs_f64();
                start += rows;
            }

            if let Some(cb) = progress_callback.as_ref() {
                let done = row_end;
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (done, m))?;
                    Ok(())
                })?;
            }
        }
        if stage_timing {
            let total_secs = total_t0.elapsed().as_secs_f64();
            let other_secs = (total_secs - decode_secs - proj_secs - assoc_secs).max(0.0_f64);
            let to_pct = |x: f64| -> f64 {
                if total_secs > 0.0 {
                    x * 100.0 / total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "LMM packed timing: decode={:.3}s ({:.1}%), proj={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, threads={}",
                decode_secs,
                to_pct(decode_secs),
                proj_secs,
                to_pct(proj_secs),
                assoc_secs,
                to_pct(assoc_secs),
                other_secs,
                to_pct(other_secs),
                total_secs,
                m,
                n,
                if threads > 0 { threads } else { rayon::current_num_threads() }
            );
        }
        Ok(())
    })?;

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_missing,
    s,
    xcov,
    y_rot,
    u_t,
    chrom,
    pos,
    snp,
    allele0,
    allele1,
    out_tsv,
    sample_indices=None,
    row_indices=None,
    low=-5.0,
    high=5.0,
    max_iter=50,
    tol=1e-2,
    threads=0,
    model="add",
    progress_callback=None,
    progress_every=0,
    nullml=None,
    init_log10_lbd=None,
    rotate_block_rows=256
))]
pub fn lmm_reml_assoc_packed_f32_to_tsv<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_missing: PyReadonlyArray1<'py, f32>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    u_t: PyReadonlyArray2<'py, f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: &str,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    model: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    nullml: Option<f64>,
    init_log10_lbd: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<usize> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }
    let gm = PackedGeneticModel::parse(model)?;
    let init_log10_lbd = init_log10_lbd
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(low, high));

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m_packed = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;
    let row_idx: Option<Vec<usize>> = if let Some(ridx) = row_indices {
        Some(parse_index_vec_i64(
            ridx.as_slice()?,
            m_packed,
            "row_indices",
        )?)
    } else {
        None
    };
    let m = row_idx.as_ref().map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m || row_maf.len() != m || row_missing.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf/row_missing length mismatch with packed rows",
        ));
    }
    if chrom.len() != m
        || pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
    {
        return Err(PyRuntimeError::new_err(format!(
            "TSV metadata length mismatch: rows={m}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        )));
    }

    let y = y_rot.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y_rot must not be empty"));
    }
    let s = s.as_slice()?;
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        if sidx_slice.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                sidx_slice.len()
            )));
        }
        parse_index_vec_i64(sidx_slice, n_samples, "sample_indices")?
    } else {
        if n != n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "len(y_rot)={} must equal n_samples={} when sample_indices is not provided",
                n, n_samples
            )));
        }
        (0..n_samples).collect()
    };
    let sample_identity = sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
    let sample_subset_plan: Option<SubsetDecodePlan> = if sample_identity {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
            &sample_idx,
            n_samples,
        ))
    };
    let dense_subset_pos: Option<Vec<isize>> = if sample_identity {
        None
    } else {
        maybe_build_dense_subset_pos(&sample_idx, n_samples)
    };

    let xcov_arr = xcov.as_array();
    if xcov_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("xcov must be 2D (n, p_cov)"));
    }
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("xcov.n_rows must equal len(y_rot)"));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let ut_arr = u_t.as_array();
    if ut_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("u_t must be 2D (n, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4usize } else { 3usize };

    let pool = get_cached_pool(threads)?;
    let block_rows = rotate_block_rows.max(1);
    let progress_block = if progress_every == 0 {
        block_rows.max(1)
    } else {
        progress_every.max(1)
    };
    let stage_timing = env_truthy("JX_LMM_PACKED_STAGE_TIMING");
    let out_tsv_path = out_tsv.to_string();

    py.detach(move || -> PyResult<()> {
        let total_t0 = Instant::now();
        let mut decode_secs = 0.0_f64;
        let mut proj_secs = 0.0_f64;
        let mut assoc_secs = 0.0_f64;
        let mut tsv_secs = 0.0_f64;

        let writer = AsyncTsvWriter::with_config(
            &out_tsv_path,
            if with_plrt {
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n"
                    .as_slice()
            } else {
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n"
                    .as_slice()
            },
            64 * 1024 * 1024,
            4,
        )
        .map_err(PyRuntimeError::new_err)?;

        let mut snp_buf = vec![0.0_f32; block_rows * n];
        let mut rot_buf = vec![0.0_f32; block_rows * n];
        let mut out_block = vec![0.0_f64; block_rows * out_cols];
        let mut miss_block = vec![0usize; block_rows];
        let mut text_buf = String::with_capacity(block_rows * 128);

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            out_block
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || (vec![0.0_f64; n], init_log10_lbd),
                    |state, (idx, out_row)| {
                        let (snp_vec, last_log10_lbd) = state;
                        let row = &g_block[idx * n..(idx + 1) * n];
                        for i in 0..n {
                            snp_vec[i] = row[i] as f64;
                        }

                        let init_guess = (*last_log10_lbd).or(init_log10_lbd);
                        let (best_log10_lbd, _best_cost) = brent_minimize_with_init(
                            |x0| -reml_loglike(x0, s, &xcov_flat, y, Some(&snp_vec[..]), n, p_cov),
                            low,
                            high,
                            tol,
                            max_iter,
                            init_guess,
                        );
                        *last_log10_lbd = Some(best_log10_lbd);

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

        for row_start in (0..m).step_by(progress_block) {
            let row_end = (row_start + progress_block).min(m);
            let mut start = row_start;
            while start < row_end {
                let rows = (row_end - start).min(block_rows);
                let snp_block = &mut snp_buf[..rows * n];
                let rot_block = &mut rot_buf[..rows * n];

                let dt0 = Instant::now();
                decode_centered_block_packed_f32(
                    &packed_flat,
                    bytes_per_snp,
                    row_flip,
                    row_maf,
                    row_idx.as_deref(),
                    start,
                    rows,
                    n,
                    gm,
                    sample_identity,
                    sample_subset_plan.as_ref(),
                    dense_subset_pos.as_deref(),
                    snp_block,
                );
                decode_secs += dt0.elapsed().as_secs_f64();

                let miss_sub = &mut miss_block[..rows];
                fill_packed_missing_block(
                    miss_sub,
                    sample_identity,
                    row_missing,
                    row_idx.as_deref(),
                    start,
                    &packed_flat,
                    bytes_per_snp,
                    n_samples,
                    &sample_idx,
                    pool.as_ref(),
                );

                let rotate_tile_rows = choose_rotate_tile_rows(
                    rows,
                    if threads > 0 {
                        threads
                    } else {
                        rayon::current_num_threads()
                    },
                );
                let pt0 = Instant::now();
                if let Some(tp) = &pool {
                    tp.install(|| {
                        rotate_snp_block_with_ut_parallel(
                            snp_block,
                            rows,
                            n,
                            &ut_flat,
                            rot_block,
                            rotate_tile_rows,
                        );
                    });
                } else {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                }
                proj_secs += pt0.elapsed().as_secs_f64();

                let out_sub = &mut out_block[..rows * out_cols];
                let at0 = Instant::now();
                if let Some(tp) = &pool {
                    tp.install(|| run_rows(rot_block, out_sub));
                } else {
                    run_rows(rot_block, out_sub);
                }
                assoc_secs += at0.elapsed().as_secs_f64();

                let tt0 = Instant::now();
                text_buf.clear();
                if text_buf.capacity() < rows * 128 {
                    text_buf.reserve(rows * 128 - text_buf.capacity());
                }
                for off in 0..rows {
                    let idx = start + off;
                    let base = off * out_cols;
                    let beta = out_sub[base];
                    let se = out_sub[base + 1];
                    let pwald = sanitize_assoc_pvalue(beta, se, out_sub[base + 2]);
                    let chisq = chisq_from_beta_se_and_optional_plrt(
                        beta,
                        se,
                        if with_plrt { Some(out_sub[base + 3]) } else { None },
                    );
                    let chisq_txt = format_chisq_value(chisq);
                    if with_plrt {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                            chrom[idx],
                            pos[idx],
                            snp[idx],
                            allele0[idx],
                            allele1[idx],
                            row_maf[idx],
                            miss_sub[off],
                            beta,
                            se,
                            chisq_txt,
                            pwald,
                            out_sub[base + 3]
                        );
                    } else {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\n",
                            chrom[idx],
                            pos[idx],
                            snp[idx],
                            allele0[idx],
                            allele1[idx],
                            row_maf[idx],
                            miss_sub[off],
                            beta,
                            se,
                            chisq_txt,
                            out_sub[base + 2]
                        );
                    }
                }
                let payload = std::mem::take(&mut text_buf).into_bytes();
                writer.send(payload).map_err(PyRuntimeError::new_err)?;
                tsv_secs += tt0.elapsed().as_secs_f64();

                start += rows;
            }

            if let Some(cb) = progress_callback.as_ref() {
                let done = row_end;
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (done, m))?;
                    Ok(())
                })?;
            } else {
                Python::attach(|py2| py2.check_signals())?;
            }
        }

        let wt0 = Instant::now();
        let writer_result = writer.finish().map_err(PyRuntimeError::new_err);
        tsv_secs += wt0.elapsed().as_secs_f64();
        writer_result?;

        if stage_timing {
            let total_secs = total_t0.elapsed().as_secs_f64();
            let other_secs =
                (total_secs - decode_secs - proj_secs - assoc_secs - tsv_secs).max(0.0_f64);
            let to_pct = |x: f64| -> f64 {
                if total_secs > 0.0 {
                    x * 100.0 / total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "LMM packed timing: decode={:.3}s ({:.1}%), proj={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), tsv={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, threads={}",
                decode_secs,
                to_pct(decode_secs),
                proj_secs,
                to_pct(proj_secs),
                assoc_secs,
                to_pct(assoc_secs),
                tsv_secs,
                to_pct(tsv_secs),
                other_secs,
                to_pct(other_secs),
                total_secs,
                m,
                n,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                }
            );
        }
        Ok(())
    })?;

    Ok(m)
}

#[derive(Clone, Copy)]
pub(crate) enum PackedGeneticModel {
    Add,
    Dom,
    Rec,
    Het,
}

impl PackedGeneticModel {
    fn parse(text: &str) -> PyResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "add" => Ok(Self::Add),
            "dom" => Ok(Self::Dom),
            "rec" => Ok(Self::Rec),
            "het" => Ok(Self::Het),
            _ => Err(PyRuntimeError::new_err(
                "model must be one of: add, dom, rec, het",
            )),
        }
    }

    #[inline]
    fn apply(self, g: f64) -> f64 {
        match self {
            Self::Add => g,
            Self::Dom => {
                if g > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Rec => {
                if (g - 2.0).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Het => {
                if (g - 1.0).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

#[inline]
fn packed_model_value_lut_f32(gm: PackedGeneticModel, flip: bool, mean_g: f32) -> [f32; 4] {
    let raw = if flip {
        [2.0_f32, mean_g, 1.0_f32, 0.0_f32]
    } else {
        [0.0_f32, mean_g, 1.0_f32, 2.0_f32]
    };
    [
        gm.apply(raw[0] as f64) as f32,
        gm.apply(raw[1] as f64) as f32,
        gm.apply(raw[2] as f64) as f32,
        gm.apply(raw[3] as f64) as f32,
    ]
}

#[inline]
fn center_decoded_row_inplace(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let mean = (row.iter().map(|&v| v as f64).sum::<f64>() / (row.len() as f64)) as f32;
    for v in row.iter_mut() {
        *v -= mean;
    }
}

#[inline]
pub(crate) fn decode_centered_block_packed_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    row_start: usize,
    rows: usize,
    n: usize,
    gm: PackedGeneticModel,
    sample_identity: bool,
    subset_plan: Option<&SubsetDecodePlan>,
    dense_subset_pos: Option<&[isize]>,
    out: &mut [f32],
) {
    if rows == 0 || n == 0 {
        return;
    }
    debug_assert_eq!(out.len(), rows * n);
    let code4_lut = &packed_byte_lut().code4;

    if sample_identity {
        out.par_chunks_mut(n).enumerate().for_each(|(off, dst)| {
            let idx = row_start + off;
            let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
            let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
            let mean_g = (2.0_f64 * (row_maf[idx] as f64)).max(0.0);
            let value_lut = packed_model_value_lut_f32(gm, row_flip[idx], mean_g as f32);
            decode_row_centered_full_lut(row, n, code4_lut, &value_lut, dst);
            center_decoded_row_inplace(dst);
        });
        return;
    }

    if let Some(sample_pos) = dense_subset_pos {
        out.par_chunks_mut(n).enumerate().for_each_init(
            || vec![0.0_f32; sample_pos.len()],
            |full_row, (off, dst)| {
                let idx = row_start + off;
                let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                let mean_g = (2.0_f64 * (row_maf[idx] as f64)).max(0.0);
                let value_lut = packed_model_value_lut_f32(gm, row_flip[idx], mean_g as f32);
                decode_row_centered_full_lut(
                    row,
                    sample_pos.len(),
                    code4_lut,
                    &value_lut,
                    full_row,
                );
                let mut sum_g = 0.0_f64;
                for (full_j, &out_j) in sample_pos.iter().enumerate() {
                    if out_j >= 0 {
                        let gv = full_row[full_j];
                        dst[out_j as usize] = gv;
                        sum_g += gv as f64;
                    }
                }
                let g_mean = (sum_g / (n as f64)) as f32;
                for v in dst.iter_mut() {
                    *v -= g_mean;
                }
            },
        );
        return;
    }

    let plan = subset_plan.expect("subset_plan must exist for non-identity mapping");
    out.par_chunks_mut(n).enumerate().for_each(|(off, dst)| {
        let idx = row_start + off;
        let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
        let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
        let mean_g = (2.0_f64 * (row_maf[idx] as f64)).max(0.0);
        let value_lut = packed_model_value_lut_f32(gm, row_flip[idx], mean_g as f32);
        decode_subset_with_plan(row, plan, &value_lut, dst);
        center_decoded_row_inplace(dst);
    });
}

#[allow(dead_code)]
struct PackedNullEval {
    lbd: f64,
    ml: f64,
    reml: f64,
    log_det_v: f64,
    v1_inv: Vec<f64>,
    base_a: Vec<f64>,
    base_b: Vec<f64>,
}

#[allow(dead_code)]
fn fastlmm_null_eval_packed(
    log10_lbd: f64,
    tau: f64,
    s: &[f64],
    u1tx: &[f64],
    u2tx: &[f64],
    u1ty: &[f64],
    u2ty: &[f64],
    u2_xtx: &[f64],
    u2_xty: &[f64],
    k: usize,
    n: usize,
    p: usize,
    c_reml: f64,
    c_ml: f64,
) -> Option<PackedNullEval> {
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return None;
    }
    let v2_denom = lbd + tau;
    if !v2_denom.is_finite() || v2_denom <= 0.0 {
        return None;
    }
    let v2_inv = 1.0 / v2_denom;

    let mut v1_inv = vec![0.0_f64; k];
    let mut log_det_v = 0.0_f64;
    for i in 0..k {
        let v1 = s[i] + v2_denom;
        if !v1.is_finite() || v1 <= 0.0 {
            return None;
        }
        v1_inv[i] = 1.0 / v1;
        log_det_v += v1.ln();
    }
    log_det_v += ((n - k) as f64) * v2_denom.ln();

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
            let xir = u1tx[base + r];
            for c in 0..=r {
                base_a[r * p + c] += vi * xir * u1tx[base + c];
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
            base_b[r] += vi * u1tx[base + r] * yi;
        }
    }

    let mut chol_a = base_a.clone();
    let ridge = 1e-6_f64;
    for i in 0..p {
        chol_a[i * p + i] += ridge;
    }
    if cholesky_inplace(&mut chol_a, p).is_none() {
        return None;
    }
    let log_det_xtv = cholesky_logdet(&chol_a, p);

    let mut beta = vec![0.0_f64; p];
    cholesky_solve_into(&chol_a, p, &base_b, &mut beta);

    let mut r1_sum = 0.0_f64;
    for i in 0..k {
        let mut xb = 0.0_f64;
        let base = i * p;
        for r in 0..p {
            xb += u1tx[base + r] * beta[r];
        }
        let ri = u1ty[i] - xb;
        r1_sum += v1_inv[i] * ri * ri;
    }

    let mut r2_sum = 0.0_f64;
    for i in 0..n {
        let mut xb = 0.0_f64;
        let base = i * p;
        for r in 0..p {
            xb += u2tx[base + r] * beta[r];
        }
        let ri = u2ty[i] - xb;
        r2_sum += ri * ri;
    }

    let rtv_invr = r1_sum + v2_inv * r2_sum;
    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return None;
    }

    let n_f = n as f64;
    let n_minus_p_f = (n - p) as f64;
    let total_reml = n_minus_p_f * rtv_invr.ln() + log_det_v + log_det_xtv;
    let reml = c_reml - 0.5 * total_reml;

    let total_ml = n_f * rtv_invr.ln() + log_det_v;
    let ml = c_ml - 0.5 * total_ml;

    Some(PackedNullEval {
        lbd,
        ml,
        reml,
        log_det_v,
        v1_inv,
        base_a,
        base_b,
    })
}

#[allow(dead_code)]
struct PackedFastlmmScratch {
    xtv_inv_x: Vec<f64>,
    xtv_inv_y: Vec<f64>,
    beta: Vec<f64>,
    rhs: Vec<f64>,
    work: Vec<f64>,
    u1_xtsnp: Vec<f64>,
    u2_xtsnp: Vec<f64>,
}

#[allow(dead_code)]
impl PackedFastlmmScratch {
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

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    u,
    s,
    y,
    x=None,
    sample_indices=None,
    row_indices=None,
    low=-5.0,
    high=5.0,
    max_iter=50,
    tol=1e-2,
    tau=0.0,
    threads=0,
    model="add",
    progress_callback=None,
    progress_every=0,
    fixed_lbd=None,
    fixed_ml0=None
))]
pub fn fastlmm_assoc_packed_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    u: PyReadonlyArray2<'py, f32>,
    s: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    tau: f64,
    threads: usize,
    model: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    fixed_lbd: Option<f64>,
    fixed_ml0: Option<f64>,
) -> PyResult<(f64, f64, f64, Bound<'py, PyArray2<f64>>)> {
    if fixed_lbd.is_none() && low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }
    if !(tau.is_finite() && tau >= 0.0) {
        return Err(PyRuntimeError::new_err("tau must be finite and >= 0"));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let gm = PackedGeneticModel::parse(model)?;

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m_packed = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_idx: Option<Vec<usize>> = if let Some(ridx) = row_indices {
        Some(parse_index_vec_i64(
            ridx.as_slice()?,
            m_packed,
            "row_indices",
        )?)
    } else {
        None
    };
    let m = row_idx.as_ref().map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip.len()
        )));
    }
    if row_maf.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        )));
    }

    let y = y.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y must not be empty"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        if sidx_slice.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                sidx_slice.len()
            )));
        }
        parse_index_vec_i64(sidx_slice, n_samples, "sample_indices")?
    } else {
        if n != n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "len(y)={} must equal n_samples={} when sample_indices is not provided",
                n, n_samples
            )));
        }
        (0..n).collect()
    };
    let sample_identity = sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
    let sample_subset_plan: Option<SubsetDecodePlan> = if sample_identity {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
            &sample_idx,
            n_samples,
        ))
    };
    let dense_subset_pos: Option<Vec<isize>> = if sample_identity {
        None
    } else {
        maybe_build_dense_subset_pos(&sample_idx, n_samples)
    };

    let u_arr = u.as_array();
    if u_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("u must be 2D (n_samples, k)"));
    }
    let (u_n, k_full) = (u_arr.shape()[0], u_arr.shape()[1]);
    if u_n != n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "u row count mismatch: got {u_n}, expected {n_samples}"
        )));
    }
    if k_full == 0 {
        return Err(PyRuntimeError::new_err("u must have at least 1 component"));
    }

    let s_f32 = s.as_slice()?;
    if s_f32.len() != k_full {
        return Err(PyRuntimeError::new_err(format!(
            "s length mismatch: got {}, expected {k_full}",
            s_f32.len()
        )));
    }
    if k_full < n {
        return Err(PyRuntimeError::new_err(format!(
            "u must provide full-rank eigenvectors for packed fixed-lambda scan: got k={}, expected >= n={}",
            k_full, n
        )));
    }
    let mut s_vec: Vec<f64> = s_f32[..n].iter().map(|&v| v as f64).collect();
    if tau != 0.0 {
        for v in s_vec.iter_mut() {
            *v += tau;
        }
    }
    if s_vec.iter().any(|v| !v.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "invalid s/tau produced non-finite values",
        ));
    }

    let p0 = match &x {
        Some(xarr) => {
            let xa = xarr.as_array();
            if xa.ndim() != 2 {
                return Err(PyRuntimeError::new_err("x must be 2D (n, p0)"));
            }
            let xr = xa.shape()[0];
            let xc = xa.shape()[1];
            if xr != n {
                return Err(PyRuntimeError::new_err(format!(
                    "x row count mismatch: got {xr}, expected {n}"
                )));
            }
            xc
        }
        None => 0usize,
    };
    let p = p0 + 1;
    if n <= p {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p for null model: n={n}, p={p}"
        )));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p+1 for SNP tests: n={n}, p={p}"
        )));
    }

    let u_slice: &[f32] = u
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u must be contiguous (C-order)"))?;
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let x_slice_opt: Option<&[f64]> = match &x {
        Some(xarr) => Some(
            xarr.as_slice()
                .map_err(|_| PyRuntimeError::new_err("x must be contiguous (C-order)"))?,
        ),
        None => None,
    };

    let mut x_full = vec![0.0_f64; n * p];
    for i in 0..n {
        x_full[i * p] = 1.0;
    }
    if let Some(x_slice) = x_slice_opt {
        for i in 0..n {
            let src = &x_slice[i * p0..(i + 1) * p0];
            let dst = &mut x_full[i * p + 1..(i + 1) * p];
            dst.copy_from_slice(src);
        }
    }

    let pool = get_cached_pool(threads)?;
    let u_t_sub = build_u_t_sub_from_sample_idx(u_slice, k_full, &sample_idx, pool.as_ref());
    let mut y_rot = vec![0.0_f64; n];
    let mut x_rot = vec![0.0_f64; n * p];
    {
        let mut run = || {
            y_rot.par_iter_mut().enumerate().for_each(|(i, dst)| {
                let ut_row = &u_t_sub[i * n..(i + 1) * n];
                let mut acc = 0.0_f64;
                for j in 0..n {
                    acc += (ut_row[j] as f64) * y[j];
                }
                *dst = acc;
            });
            x_rot.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
                let ut_row = &u_t_sub[i * n..(i + 1) * n];
                for c in 0..p {
                    let mut acc = 0.0_f64;
                    for j in 0..n {
                        acc += (ut_row[j] as f64) * x_full[j * p + c];
                    }
                    row[c] = acc;
                }
            });
        };
        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    }

    let (lbd, ml0, reml0) = if let Some(lbd_fixed) = fixed_lbd {
        if !(lbd_fixed.is_finite() && lbd_fixed > 0.0) {
            return Err(PyRuntimeError::new_err(
                "fixed_lbd must be finite and > 0 when provided",
            ));
        }
        let log10_lbd = lbd_fixed.log10();
        let ml = if let Some(v) = fixed_ml0 {
            if !v.is_finite() {
                return Err(PyRuntimeError::new_err(
                    "fixed_ml0 must be finite when provided",
                ));
            }
            v
        } else {
            ml_loglike(log10_lbd, &s_vec, &x_rot, &y_rot, None, n, p)
        };
        let reml = reml_loglike(log10_lbd, &s_vec, &x_rot, &y_rot, None, n, p);
        (lbd_fixed, ml, reml)
    } else {
        let (best_log10_lbd, best_cost) = brent_minimize(
            |x0| -reml_loglike(x0, &s_vec, &x_rot, &y_rot, None, n, p),
            low,
            high,
            tol,
            max_iter,
        );
        let lbd = 10.0_f64.powf(best_log10_lbd);
        let ml = if let Some(v) = fixed_ml0 {
            if !v.is_finite() {
                return Err(PyRuntimeError::new_err(
                    "fixed_ml0 must be finite when provided",
                ));
            }
            v
        } else {
            ml_loglike(best_log10_lbd, &s_vec, &x_rot, &y_rot, None, n, p)
        };
        (lbd, ml, -best_cost)
    };
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err(
            "invalid fixed lambda in packed fastlmm",
        ));
    }

    let with_plrt = fixed_ml0.is_some();
    let out_cols = if with_plrt { 4usize } else { 3usize };
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let progress_block = if progress_every == 0 {
        m.max(1)
    } else {
        progress_every.max(1)
    };
    let stage_timing = env_truthy("JX_FASTLMM_PACKED_STAGE_TIMING");

    py.detach(move || -> PyResult<()> {
        let total_t0 = Instant::now();
        let mut decode_secs = 0.0_f64;
        let mut proj_secs = 0.0_f64;
        let mut assoc_secs = 0.0_f64;
        let tsv_secs = 0.0_f64;

        let n_f = n as f64;
        let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
        let mut w = vec![0.0_f32; n];
        let mut log_det_v = 0.0_f64;
        for i in 0..n {
            let vv = s_vec[i] + lbd;
            if !(vv.is_finite() && vv > 0.0) {
                return Err(PyRuntimeError::new_err(
                    "non-positive s[i] + lbd in packed fastlmm",
                ));
            }
            w[i] = (1.0 / vv) as f32;
            log_det_v += vv.ln();
        }

        let mut a = vec![0.0_f64; p * p];
        let mut b = vec![0.0_f64; p];
        let mut ywy = 0.0_f64;
        for i in 0..n {
            let wi = w[i] as f64;
            let yi = y_rot[i];
            ywy += wi * yi * yi;
            let base = i * p;
            for r in 0..p {
                let xir = x_rot[base + r];
                b[r] += wi * xir * yi;
                for c in 0..=r {
                    a[r * p + c] += wi * xir * x_rot[base + c];
                }
            }
        }
        let ridge = 1e-6_f64;
        for r in 0..p {
            a[r * p + r] += ridge;
            for c in 0..r {
                a[c * p + r] = a[r * p + c];
            }
        }
        if cholesky_inplace(&mut a, p).is_none() {
            return Err(PyRuntimeError::new_err("X'WX not SPD in packed fastlmm"));
        }
        let mut a_inv_b = vec![0.0_f64; p];
        cholesky_solve_into(&a, p, &b, &mut a_inv_b);
        let b_aib = dot_loop(&b, &a_inv_b);
        let df = (n as isize) - (p as isize) - 1;
        if df <= 0 {
            return Err(PyRuntimeError::new_err("invalid df <= 0 in packed fastlmm"));
        }

        let rotate_block_rows = packed_rotate_block_rows(m, n, 0usize);
        let mut g_block = vec![0.0_f32; rotate_block_rows * n];
        let mut rot_block = vec![0.0_f32; rotate_block_rows * n];
        let mut next_progress_emit = progress_block.min(m).max(1);

        for row_start in (0..m).step_by(rotate_block_rows) {
            let row_end = (row_start + rotate_block_rows).min(m);
            let mut start = row_start;
            while start < row_end {
                let rows = row_end - start;
                let snp_block = &mut g_block[..rows * n];
                let g_rot = &mut rot_block[..rows * n];

                let dt0 = Instant::now();
                decode_centered_block_packed_f32(
                    &packed_flat,
                    bytes_per_snp,
                    row_flip,
                    row_maf,
                    None,
                    start,
                    rows,
                    n,
                    gm,
                    sample_identity,
                    sample_subset_plan.as_ref(),
                    dense_subset_pos.as_deref(),
                    snp_block,
                );
                decode_secs += dt0.elapsed().as_secs_f64();

                let rotate_tile_rows = choose_rotate_tile_rows(
                    rows,
                    if threads > 0 {
                        threads
                    } else {
                        rayon::current_num_threads()
                    },
                );
                let pt0 = Instant::now();
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &u_t_sub,
                    g_rot,
                    rotate_tile_rows,
                );
                proj_secs += pt0.elapsed().as_secs_f64();

                let out_sub = &mut out_slice[start * out_cols..(start + rows) * out_cols];
                let at0 = Instant::now();
                let mut run_assoc = || {
                    out_sub
                        .par_chunks_mut(out_cols)
                        .enumerate()
                        .for_each_init(
                            || AssocScratch::new(p),
                            |scr, (off, out_row)| {
                                scr.reset();
                                let row = &g_rot[off * n..(off + 1) * n];

                                let mut d = 0.0_f64;
                                let mut e = 0.0_f64;
                                for i in 0..n {
                                    let gi = row[i] as f64;
                                    let wi = w[i] as f64;
                                    let yi = y_rot[i];
                                    let base = i * p;
                                    let wgi = wi * gi;
                                    d += wgi * gi;
                                    e += wgi * yi;
                                    for r in 0..p {
                                        scr.c[r] += wgi * x_rot[base + r];
                                    }
                                }

                                cholesky_solve_into(&a, p, &scr.c, &mut scr.a_inv_c);
                                let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                                let schur = d - ct_aic;
                                if schur <= 1e-12 || !schur.is_finite() {
                                    out_row[0] = f64::NAN;
                                    out_row[1] = f64::NAN;
                                    out_row[2] = f64::NAN;
                                    out_row[3] = 1.0;
                                    return;
                                }

                                let ct_aib = dot_loop(&scr.c, &a_inv_b);
                                let num = e - ct_aib;
                                let beta_g = num / schur;
                                let q = b_aib + (num * num) / schur;
                                let rwr = (ywy - q).max(0.0);
                                let sigma2 = rwr / (df as f64);
                                let se_g = if sigma2.is_finite() && sigma2 > 0.0 {
                                    (sigma2 / schur).sqrt()
                                } else {
                                    f64::NAN
                                };
                                let pval = if se_g.is_finite() && se_g > 0.0 && beta_g.is_finite()
                                {
                                    let z = (beta_g / se_g).abs();
                                    (2.0 * normal_sf(z)).clamp(f64::MIN_POSITIVE, 1.0)
                                } else {
                                    1.0
                                };
                                let ml = if rwr > 0.0 && rwr.is_finite() {
                                    let total_log = n_f * rwr.ln() + log_det_v;
                                    c_ml - 0.5 * total_log
                                } else {
                                    f64::NAN
                                };
                                let mut stat = if ml.is_finite() {
                                    2.0 * (ml - ml0)
                                } else {
                                    0.0
                                };
                                if !stat.is_finite() || stat < 0.0 {
                                    stat = 0.0;
                                }
                                let plrt = chi2_sf_df1(stat);

                                out_row[0] = beta_g;
                                out_row[1] = se_g;
                                out_row[2] = pval;
                                out_row[3] = plrt;
                            },
                        );
                };
                if let Some(tp) = &pool {
                    tp.install(run_assoc);
                } else {
                    run_assoc();
                }
                assoc_secs += at0.elapsed().as_secs_f64();
                start += rows;
            }

            let done = row_end;
            if let Some(cb) = progress_callback.as_ref() {
                if !packed_progress_should_emit(done, m, progress_block, &mut next_progress_emit) {
                    continue;
                }
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (done, m))?;
                    Ok(())
                })?;
            } else {
                Python::attach(|py2| py2.check_signals())?;
            }
        }

        if stage_timing {
            let total_secs = total_t0.elapsed().as_secs_f64();
            let other_secs = (total_secs - decode_secs - proj_secs - assoc_secs).max(0.0_f64);
            let to_pct = |x: f64| -> f64 {
                if total_secs > 0.0 {
                    x * 100.0 / total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "FaST-LMM packed timing: decode={:.3}s ({:.1}%), proj={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), tsv={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, threads={}",
                decode_secs,
                to_pct(decode_secs),
                proj_secs,
                to_pct(proj_secs),
                assoc_secs,
                to_pct(assoc_secs),
                tsv_secs,
                to_pct(tsv_secs),
                other_secs,
                to_pct(other_secs),
                total_secs,
                m,
                n,
                if threads > 0 { threads } else { rayon::current_num_threads() }
            );
        }
        Ok(())
    })?;

    Ok((lbd, ml0, reml0, out))
}

#[pyfunction]
pub fn fastlmm_assoc_packed_f32_to_tsv<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_missing: PyReadonlyArray1<'py, f32>,
    u: PyReadonlyArray2<'py, f32>,
    s: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    tau: f64,
    threads: usize,
    model: &str,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    fixed_lbd: Option<f64>,
    fixed_ml0: Option<f64>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    rotate_block_rows: usize,
) -> PyResult<(f64, f64, f64)> {
    if fixed_lbd.is_none() && low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }
    if !(tau.is_finite() && tau >= 0.0) {
        return Err(PyRuntimeError::new_err("tau must be finite and >= 0"));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let gm = PackedGeneticModel::parse(model)?;

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m_packed = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;
    let row_idx: Option<Vec<usize>> = if let Some(ridx) = row_indices {
        Some(parse_index_vec_i64(
            ridx.as_slice()?,
            m_packed,
            "row_indices",
        )?)
    } else {
        None
    };
    let m = row_idx.as_ref().map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip.len()
        )));
    }
    if row_maf.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        )));
    }
    if row_missing.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_missing length mismatch: got {}, expected {m}",
            row_missing.len()
        )));
    }
    if chrom.len() != m
        || pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
    {
        return Err(PyRuntimeError::new_err(format!(
            "TSV metadata length mismatch: rows={m}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        )));
    }

    let y = y.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y must not be empty"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        if sidx_slice.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                sidx_slice.len()
            )));
        }
        parse_index_vec_i64(sidx_slice, n_samples, "sample_indices")?
    } else {
        if n != n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "len(y)={} must equal n_samples={} when sample_indices is not provided",
                n, n_samples
            )));
        }
        (0..n).collect()
    };
    let sample_identity = sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
    let sample_subset_plan: Option<SubsetDecodePlan> = if sample_identity {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
            &sample_idx,
            n_samples,
        ))
    };
    let dense_subset_pos: Option<Vec<isize>> = if sample_identity {
        None
    } else {
        maybe_build_dense_subset_pos(&sample_idx, n_samples)
    };

    let u_arr = u.as_array();
    if u_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("u must be 2D (n_samples, k)"));
    }
    let (u_n, k_full) = (u_arr.shape()[0], u_arr.shape()[1]);
    if u_n != n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "u row count mismatch: got {u_n}, expected {n_samples}"
        )));
    }
    if k_full == 0 {
        return Err(PyRuntimeError::new_err("u must have at least 1 component"));
    }

    let s_f32 = s.as_slice()?;
    if s_f32.len() != k_full {
        return Err(PyRuntimeError::new_err(format!(
            "s length mismatch: got {}, expected {k_full}",
            s_f32.len()
        )));
    }
    if k_full < n {
        return Err(PyRuntimeError::new_err(format!(
            "u must provide full-rank eigenvectors for packed fixed-lambda scan: got k={}, expected >= n={}",
            k_full, n
        )));
    }
    let mut s_vec: Vec<f64> = s_f32[..n].iter().map(|&v| v as f64).collect();
    if tau != 0.0 {
        for v in s_vec.iter_mut() {
            *v += tau;
        }
    }
    if s_vec.iter().any(|v| !v.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "invalid s/tau produced non-finite values",
        ));
    }

    let p0 = match &x {
        Some(xarr) => {
            let xa = xarr.as_array();
            if xa.ndim() != 2 {
                return Err(PyRuntimeError::new_err("x must be 2D (n, p0)"));
            }
            let xr = xa.shape()[0];
            let xc = xa.shape()[1];
            if xr != n {
                return Err(PyRuntimeError::new_err(format!(
                    "x row count mismatch: got {xr}, expected {n}"
                )));
            }
            xc
        }
        None => 0usize,
    };
    let p = p0 + 1;
    if n <= p {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p for null model: n={n}, p={p}"
        )));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p+1 for SNP tests: n={n}, p={p}"
        )));
    }

    let u_slice: &[f32] = u
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u must be contiguous (C-order)"))?;
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let x_slice_opt: Option<&[f64]> = match &x {
        Some(xarr) => Some(
            xarr.as_slice()
                .map_err(|_| PyRuntimeError::new_err("x must be contiguous (C-order)"))?,
        ),
        None => None,
    };

    let mut x_full = vec![0.0_f64; n * p];
    for i in 0..n {
        x_full[i * p] = 1.0;
    }
    if let Some(x_slice) = x_slice_opt {
        for i in 0..n {
            let src = &x_slice[i * p0..(i + 1) * p0];
            let dst = &mut x_full[i * p + 1..(i + 1) * p];
            dst.copy_from_slice(src);
        }
    }

    let pool = get_cached_pool(threads)?;
    let u_t_sub = build_u_t_sub_from_sample_idx(u_slice, k_full, &sample_idx, pool.as_ref());
    let mut y_rot = vec![0.0_f64; n];
    let mut x_rot = vec![0.0_f64; n * p];
    {
        let mut run = || {
            y_rot.par_iter_mut().enumerate().for_each(|(i, dst)| {
                let ut_row = &u_t_sub[i * n..(i + 1) * n];
                let mut acc = 0.0_f64;
                for j in 0..n {
                    acc += (ut_row[j] as f64) * y[j];
                }
                *dst = acc;
            });
            x_rot.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
                let ut_row = &u_t_sub[i * n..(i + 1) * n];
                for c in 0..p {
                    let mut acc = 0.0_f64;
                    for j in 0..n {
                        acc += (ut_row[j] as f64) * x_full[j * p + c];
                    }
                    row[c] = acc;
                }
            });
        };
        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    }

    let (lbd, ml0, reml0) = if let Some(lbd_fixed) = fixed_lbd {
        if !(lbd_fixed.is_finite() && lbd_fixed > 0.0) {
            return Err(PyRuntimeError::new_err(
                "fixed_lbd must be finite and > 0 when provided",
            ));
        }
        let log10_lbd = lbd_fixed.log10();
        let ml = if let Some(v) = fixed_ml0 {
            if !v.is_finite() {
                return Err(PyRuntimeError::new_err(
                    "fixed_ml0 must be finite when provided",
                ));
            }
            v
        } else {
            ml_loglike(log10_lbd, &s_vec, &x_rot, &y_rot, None, n, p)
        };
        let reml = reml_loglike(log10_lbd, &s_vec, &x_rot, &y_rot, None, n, p);
        (lbd_fixed, ml, reml)
    } else {
        let (best_log10_lbd, best_cost) = brent_minimize(
            |x0| -reml_loglike(x0, &s_vec, &x_rot, &y_rot, None, n, p),
            low,
            high,
            tol,
            max_iter,
        );
        let lbd = 10.0_f64.powf(best_log10_lbd);
        let ml = if let Some(v) = fixed_ml0 {
            if !v.is_finite() {
                return Err(PyRuntimeError::new_err(
                    "fixed_ml0 must be finite when provided",
                ));
            }
            v
        } else {
            ml_loglike(best_log10_lbd, &s_vec, &x_rot, &y_rot, None, n, p)
        };
        (lbd, ml, -best_cost)
    };
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err(
            "invalid fixed lambda in packed fastlmm",
        ));
    }

    let out_tsv_path = out_tsv.to_string();
    let progress_block = if progress_every == 0 {
        m.max(1)
    } else {
        progress_every.max(1)
    };
    let with_plrt = fixed_ml0.is_some();
    let out_cols = if with_plrt { 4usize } else { 3usize };
    let stage_timing = env_truthy("JX_FASTLMM_PACKED_STAGE_TIMING");

    py.detach(move || -> PyResult<()> {
        let total_t0 = Instant::now();
        let mut decode_secs = 0.0_f64;
        let mut proj_secs = 0.0_f64;
        let mut assoc_secs = 0.0_f64;
        let mut tsv_secs = 0.0_f64;

        let header: &[u8] = if with_plrt {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n"
        } else {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n"
        };
        let writer = AsyncTsvWriter::with_config(&out_tsv_path, header, 64 * 1024 * 1024, 4)
        .map_err(PyRuntimeError::new_err)?;

        let n_f = n as f64;
        let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
        let mut w = vec![0.0_f32; n];
        let mut log_det_v = 0.0_f64;
        for i in 0..n {
            let vv = s_vec[i] + lbd;
            if !(vv.is_finite() && vv > 0.0) {
                return Err(PyRuntimeError::new_err(
                    "non-positive s[i] + lbd in packed fastlmm",
                ));
            }
            w[i] = (1.0 / vv) as f32;
            log_det_v += vv.ln();
        }

        let mut a = vec![0.0_f64; p * p];
        let mut b = vec![0.0_f64; p];
        let mut ywy = 0.0_f64;
        for i in 0..n {
            let wi = w[i] as f64;
            let yi = y_rot[i];
            ywy += wi * yi * yi;
            let base = i * p;
            for r in 0..p {
                let xir = x_rot[base + r];
                b[r] += wi * xir * yi;
                for c in 0..=r {
                    a[r * p + c] += wi * xir * x_rot[base + c];
                }
            }
        }
        let ridge = 1e-6_f64;
        for r in 0..p {
            a[r * p + r] += ridge;
            for c in 0..r {
                a[c * p + r] = a[r * p + c];
            }
        }
        if cholesky_inplace(&mut a, p).is_none() {
            return Err(PyRuntimeError::new_err("X'WX not SPD in packed fastlmm"));
        }
        let mut a_inv_b = vec![0.0_f64; p];
        cholesky_solve_into(&a, p, &b, &mut a_inv_b);
        let b_aib = dot_loop(&b, &a_inv_b);
        let df = (n as isize) - (p as isize) - 1;
        if df <= 0 {
            return Err(PyRuntimeError::new_err("invalid df <= 0 in packed fastlmm"));
        }

        let rotate_block_rows = if rotate_block_rows > 0 {
            rotate_block_rows.min(m).max(1)
        } else {
            packed_rotate_block_rows(
                m,
                n,
                out_cols
                    .saturating_mul(std::mem::size_of::<f64>())
                    .saturating_add(std::mem::size_of::<usize>()),
            )
        };

        let mut g_block = vec![0.0_f32; rotate_block_rows * n];
        let mut rot_block = vec![0.0_f32; rotate_block_rows * n];
        let mut out_block = vec![0.0_f64; rotate_block_rows * out_cols];
        let mut miss_block = vec![0usize; rotate_block_rows];
        let mut text_buf = String::with_capacity(rotate_block_rows * 128);
        let mut next_progress_emit = progress_block.min(m).max(1);

        for row_start in (0..m).step_by(rotate_block_rows) {
            let row_end = (row_start + rotate_block_rows).min(m);
            let mut start = row_start;
            while start < row_end {
                let rows = row_end - start;
                let snp_block = &mut g_block[..rows * n];
                let g_rot = &mut rot_block[..rows * n];

                let dt0 = Instant::now();
                decode_centered_block_packed_f32(
                    &packed_flat,
                    bytes_per_snp,
                    row_flip,
                    row_maf,
                    row_idx.as_deref(),
                    start,
                    rows,
                    n,
                    gm,
                    sample_identity,
                    sample_subset_plan.as_ref(),
                    dense_subset_pos.as_deref(),
                    snp_block,
                );
                decode_secs += dt0.elapsed().as_secs_f64();

                let rotate_tile_rows = choose_rotate_tile_rows(
                    rows,
                    if threads > 0 {
                        threads
                    } else {
                        rayon::current_num_threads()
                    },
                );
                let pt0 = Instant::now();
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &u_t_sub,
                    g_rot,
                    rotate_tile_rows,
                );
                proj_secs += pt0.elapsed().as_secs_f64();

                let out_sub = &mut out_block[..rows * out_cols];
                let at0 = Instant::now();
                let mut run_assoc = || {
                    out_sub
                        .par_chunks_mut(out_cols)
                        .enumerate()
                        .for_each_init(
                            || AssocScratch::new(p),
                            |scr, (off, out_row)| {
                                scr.reset();
                                let row = &g_rot[off * n..(off + 1) * n];

                                let mut d = 0.0_f64;
                                let mut e = 0.0_f64;
                                for i in 0..n {
                                    let gi = row[i] as f64;
                                    let wi = w[i] as f64;
                                    let yi = y_rot[i];
                                    let base = i * p;
                                    let wgi = wi * gi;
                                    d += wgi * gi;
                                    e += wgi * yi;
                                    for r in 0..p {
                                        scr.c[r] += wgi * x_rot[base + r];
                                    }
                                }

                                cholesky_solve_into(&a, p, &scr.c, &mut scr.a_inv_c);
                                let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                                let schur = d - ct_aic;
                                if schur <= 1e-12 || !schur.is_finite() {
                                    out_row[0] = f64::NAN;
                                    out_row[1] = f64::NAN;
                                    out_row[2] = f64::NAN;
                                    if with_plrt {
                                        out_row[3] = 1.0;
                                    }
                                    return;
                                }

                                let ct_aib = dot_loop(&scr.c, &a_inv_b);
                                let num = e - ct_aib;
                                let beta_g = num / schur;
                                let q = b_aib + (num * num) / schur;
                                let rwr = (ywy - q).max(0.0);
                                let sigma2 = rwr / (df as f64);
                                let se_g = if sigma2.is_finite() && sigma2 > 0.0 {
                                    (sigma2 / schur).sqrt()
                                } else {
                                    f64::NAN
                                };
                                let pval = if se_g.is_finite() && se_g > 0.0 && beta_g.is_finite()
                                {
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
                                        2.0 * (ml - ml0)
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
                    tp.install(run_assoc);
                } else {
                    run_assoc();
                }
                assoc_secs += at0.elapsed().as_secs_f64();

                let miss_sub = &mut miss_block[..rows];
                fill_packed_missing_block(
                    miss_sub,
                    sample_identity,
                    row_missing,
                    row_idx.as_deref(),
                    start,
                    &packed_flat,
                    bytes_per_snp,
                    n_samples,
                    &sample_idx,
                    pool.as_ref(),
                );

                let tt0 = Instant::now();
                text_buf.clear();
                if text_buf.capacity() < rows * 128 {
                    text_buf.reserve(rows * 128 - text_buf.capacity());
                }
                for off in 0..rows {
                    let idx = start + off;
                    let base = off * out_cols;
                    let beta = out_sub[base];
                    let se = out_sub[base + 1];
                    let pwald = sanitize_assoc_pvalue(beta, se, out_sub[base + 2]);
                    let chisq = chisq_from_beta_se_and_optional_plrt(beta, se, None);
                    let chisq_txt = format_chisq_value(chisq);
                    if with_plrt {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                            chrom[idx],
                            pos[idx],
                            snp[idx],
                            allele0[idx],
                            allele1[idx],
                            row_maf[idx],
                            miss_sub[off],
                            beta,
                            se,
                            chisq_txt,
                            pwald,
                            out_sub[base + 3]
                        );
                    } else {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\n",
                            chrom[idx],
                            pos[idx],
                            snp[idx],
                            allele0[idx],
                            allele1[idx],
                            row_maf[idx],
                            miss_sub[off],
                            beta,
                            se,
                            chisq_txt,
                            pwald
                        );
                    }
                }
                let payload = std::mem::take(&mut text_buf).into_bytes();
                writer.send(payload).map_err(PyRuntimeError::new_err)?;
                tsv_secs += tt0.elapsed().as_secs_f64();
                start += rows;
            }

            let done = row_end;
            if let Some(cb) = progress_callback.as_ref() {
                if !packed_progress_should_emit(done, m, progress_block, &mut next_progress_emit) {
                    continue;
                }
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (done, m))?;
                    Ok(())
                })?;
            } else {
                Python::attach(|py2| py2.check_signals())?;
            }
        }

        let wt0 = Instant::now();
        let writer_result = writer.finish().map_err(PyRuntimeError::new_err);
        tsv_secs += wt0.elapsed().as_secs_f64();
        writer_result?;

        if stage_timing {
            let total_secs = total_t0.elapsed().as_secs_f64();
            let other_secs = (total_secs - decode_secs - proj_secs - assoc_secs - tsv_secs)
                .max(0.0_f64);
            let to_pct = |x: f64| -> f64 {
                if total_secs > 0.0 {
                    x * 100.0 / total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "FaST-LMM packed timing: decode={:.3}s ({:.1}%), proj={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), tsv={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, threads={}",
                decode_secs,
                to_pct(decode_secs),
                proj_secs,
                to_pct(proj_secs),
                assoc_secs,
                to_pct(assoc_secs),
                tsv_secs,
                to_pct(tsv_secs),
                other_secs,
                to_pct(other_secs),
                total_secs,
                m,
                n,
                if threads > 0 { threads } else { rayon::current_num_threads() }
            );
        }
        Ok(())
    })?;

    Ok((lbd, ml0, reml0))
}

#[pyfunction]
pub fn fvlmm_assoc_packed_f32_to_tsv<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_missing: PyReadonlyArray1<'py, f32>,
    u: PyReadonlyArray2<'py, f32>,
    s: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    tau: f64,
    threads: usize,
    model: &str,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    fixed_lbd: Option<f64>,
    fixed_ml0: Option<f64>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    rotate_block_rows: usize,
) -> PyResult<(f64, f64, f64)> {
    if fixed_lbd.is_none() && low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }
    if !(tau.is_finite() && tau >= 0.0) {
        return Err(PyRuntimeError::new_err("tau must be finite and >= 0"));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let gm = PackedGeneticModel::parse(model)?;

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m_packed = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;
    let row_idx: Option<Vec<usize>> = if let Some(ridx) = row_indices {
        Some(parse_index_vec_i64(
            ridx.as_slice()?,
            m_packed,
            "row_indices",
        )?)
    } else {
        None
    };
    let m = row_idx.as_ref().map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip.len()
        )));
    }
    if row_maf.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        )));
    }
    if row_missing.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_missing length mismatch: got {}, expected {m}",
            row_missing.len()
        )));
    }
    if chrom.len() != m
        || pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
    {
        return Err(PyRuntimeError::new_err(format!(
            "TSV metadata length mismatch: rows={m}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        )));
    }

    let y = y.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y must not be empty"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        if sidx_slice.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                sidx_slice.len()
            )));
        }
        parse_index_vec_i64(sidx_slice, n_samples, "sample_indices")?
    } else {
        if n != n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "len(y)={} must equal n_samples={} when sample_indices is not provided",
                n, n_samples
            )));
        }
        (0..n).collect()
    };
    let sample_identity = sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
    let sample_subset_plan: Option<SubsetDecodePlan> = if sample_identity {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
            &sample_idx,
            n_samples,
        ))
    };
    let dense_subset_pos: Option<Vec<isize>> = if sample_identity {
        None
    } else {
        maybe_build_dense_subset_pos(&sample_idx, n_samples)
    };

    let u_arr = u.as_array();
    if u_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("u must be 2D (n_samples, k)"));
    }
    let (u_n, k_full) = (u_arr.shape()[0], u_arr.shape()[1]);
    if u_n != n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "u row count mismatch: got {u_n}, expected {n_samples}"
        )));
    }
    if k_full == 0 {
        return Err(PyRuntimeError::new_err("u must have at least 1 component"));
    }

    let s_f32 = s.as_slice()?;
    if s_f32.len() != k_full {
        return Err(PyRuntimeError::new_err(format!(
            "s length mismatch: got {}, expected {k_full}",
            s_f32.len()
        )));
    }
    if k_full < n {
        return Err(PyRuntimeError::new_err(format!(
            "u must provide full-rank eigenvectors for packed fixed-lambda scan: got k={}, expected >= n={}",
            k_full, n
        )));
    }
    let mut s_vec: Vec<f64> = s_f32[..n].iter().map(|&v| v as f64).collect();
    if tau != 0.0 {
        for v in s_vec.iter_mut() {
            *v += tau;
        }
    }
    if s_vec.iter().any(|v| !v.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "invalid s/tau produced non-finite values",
        ));
    }

    let p0 = match &x {
        Some(xarr) => {
            let xa = xarr.as_array();
            if xa.ndim() != 2 {
                return Err(PyRuntimeError::new_err("x must be 2D (n, p0)"));
            }
            let xr = xa.shape()[0];
            let xc = xa.shape()[1];
            if xr != n {
                return Err(PyRuntimeError::new_err(format!(
                    "x row count mismatch: got {xr}, expected {n}"
                )));
            }
            xc
        }
        None => 0usize,
    };
    let p = p0 + 1;
    if n <= p {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p for null model: n={n}, p={p}"
        )));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p+1 for SNP tests: n={n}, p={p}"
        )));
    }

    let u_slice: &[f32] = u
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u must be contiguous (C-order)"))?;
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let x_slice_opt: Option<&[f64]> = match &x {
        Some(xarr) => Some(
            xarr.as_slice()
                .map_err(|_| PyRuntimeError::new_err("x must be contiguous (C-order)"))?,
        ),
        None => None,
    };

    let mut x_full = vec![0.0_f64; n * p];
    for i in 0..n {
        x_full[i * p] = 1.0;
    }
    if let Some(x_slice) = x_slice_opt {
        for i in 0..n {
            let src = &x_slice[i * p0..(i + 1) * p0];
            let dst = &mut x_full[i * p + 1..(i + 1) * p];
            dst.copy_from_slice(src);
        }
    }

    let pool = get_cached_pool(threads)?;
    let u_t_sub = build_u_t_sub_from_sample_idx(u_slice, k_full, &sample_idx, pool.as_ref());
    let mut y_rot = vec![0.0_f64; n];
    let mut x_rot = vec![0.0_f64; n * p];
    {
        let mut run = || {
            y_rot.par_iter_mut().enumerate().for_each(|(i, dst)| {
                let ut_row = &u_t_sub[i * n..(i + 1) * n];
                let mut acc = 0.0_f64;
                for j in 0..n {
                    acc += (ut_row[j] as f64) * y[j];
                }
                *dst = acc;
            });
            x_rot.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
                let ut_row = &u_t_sub[i * n..(i + 1) * n];
                for c in 0..p {
                    let mut acc = 0.0_f64;
                    for j in 0..n {
                        acc += (ut_row[j] as f64) * x_full[j * p + c];
                    }
                    row[c] = acc;
                }
            });
        };
        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    }

    let (lbd, ml0, reml0) = if let Some(lbd_fixed) = fixed_lbd {
        if !(lbd_fixed.is_finite() && lbd_fixed > 0.0) {
            return Err(PyRuntimeError::new_err(
                "fixed_lbd must be finite and > 0 when provided",
            ));
        }
        let log10_lbd = lbd_fixed.log10();
        let ml = if let Some(v) = fixed_ml0 {
            if !v.is_finite() {
                return Err(PyRuntimeError::new_err(
                    "fixed_ml0 must be finite when provided",
                ));
            }
            v
        } else {
            ml_loglike(log10_lbd, &s_vec, &x_rot, &y_rot, None, n, p)
        };
        let reml = reml_loglike(log10_lbd, &s_vec, &x_rot, &y_rot, None, n, p);
        (lbd_fixed, ml, reml)
    } else {
        let (best_log10_lbd, best_cost) = brent_minimize(
            |x0| -reml_loglike(x0, &s_vec, &x_rot, &y_rot, None, n, p),
            low,
            high,
            tol,
            max_iter,
        );
        let lbd_tmp = 10.0_f64.powf(best_log10_lbd);
        let ml = if let Some(v) = fixed_ml0 {
            if !v.is_finite() {
                return Err(PyRuntimeError::new_err(
                    "fixed_ml0 must be finite when provided",
                ));
            }
            v
        } else {
            ml_loglike(best_log10_lbd, &s_vec, &x_rot, &y_rot, None, n, p)
        };
        (lbd_tmp, ml, -best_cost)
    };
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err(
            "invalid fixed lambda in packed fvlmm",
        ));
    }

    let out_tsv_path = out_tsv.to_string();
    let progress_block = if progress_every == 0 {
        m.max(1)
    } else {
        progress_every.max(1)
    };
    let with_plrt = fixed_ml0.is_some();
    let out_cols = if with_plrt { 4usize } else { 3usize };
    let stage_timing =
        env_truthy("JX_FVLMM_PACKED_STAGE_TIMING") || env_truthy("JX_FASTLMM_PACKED_STAGE_TIMING");

    let cache = prepare_fixed_lambda_assoc_cache_f32(&s_vec, &x_rot, &y_rot, n, p, lbd)?;

    py.detach(move || -> PyResult<()> {
        let total_t0 = Instant::now();
        let decode_nanos = AtomicU64::new(0);
        let proj_nanos = AtomicU64::new(0);
        let assoc_nanos = AtomicU64::new(0);
        let tsv_nanos = AtomicU64::new(0);

        let header: &[u8] = if with_plrt {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n"
        } else {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n"
        };
        let writer = AsyncTsvWriter::with_config(&out_tsv_path, header, 64 * 1024 * 1024, 4)
        .map_err(PyRuntimeError::new_err)?;

        let rotate_block_rows = if rotate_block_rows > 0 {
            rotate_block_rows.min(m).max(1)
        } else {
            packed_rotate_block_rows(
                m,
                n,
                out_cols
                    .saturating_mul(std::mem::size_of::<f64>())
                    .saturating_add(std::mem::size_of::<usize>()),
            )
        };

        let mut text_buf = String::with_capacity(rotate_block_rows * 128);
        let mut next_progress_emit = progress_block.min(m).max(1);
        let proj_threads = stage_proj_threads_or(threads);
        let assoc_threads = stage_assoc_threads_or(threads);
        let proj_pool = get_cached_pool(proj_threads)?;
        let assoc_pool = get_cached_pool(assoc_threads)?;
        let packed_pipeline_default = !assoc_rotate_prefers_rayon_rowmajor_f32_kernel()
            && rotate_block_rows < m
            && proj_threads > 1
            && assoc_threads > 1;
        let pipeline_depth = use_packed_three_stage_pipeline(packed_pipeline_default);
        let missing_pool = if pipeline_depth > 0 {
            None
        } else {
            assoc_pool.as_ref()
        };

        run_packed_decode_rotate_finalize_triple_buffer_f32(
            m,
            n,
            rotate_block_rows,
            out_cols,
            pipeline_depth,
            |start, rows, snp_block, miss_sub| {
                let dt0 = Instant::now();
                decode_centered_block_packed_f32(
                    &packed_flat,
                    bytes_per_snp,
                    row_flip,
                    row_maf,
                    row_idx.as_deref(),
                    start,
                    rows,
                    n,
                    gm,
                    sample_identity,
                    sample_subset_plan.as_ref(),
                    dense_subset_pos.as_deref(),
                    snp_block,
                );
                fill_packed_missing_block(
                    miss_sub,
                    sample_identity,
                    row_missing,
                    row_idx.as_deref(),
                    start,
                    &packed_flat,
                    bytes_per_snp,
                    n_samples,
                    &sample_idx,
                    missing_pool,
                );
                record_elapsed_nanos(&decode_nanos, dt0);
            },
            |start, rows, snp_block, g_rot| {
                let _ = start;
                let pt0 = Instant::now();
                rotate_snp_block_with_ut_blas(
                    snp_block,
                    rows,
                    n,
                    &u_t_sub,
                    g_rot,
                    proj_threads,
                    proj_pool.as_ref(),
                );
                record_elapsed_nanos(&proj_nanos, pt0);
            },
            |start, rows, g_rot, out_sub, miss_sub| -> PyResult<()> {
                let at0 = Instant::now();
                assoc_fixed_lambda_rot_block_blas_f32(
                    g_rot,
                    rows,
                    n,
                    p,
                    &cache,
                    out_sub,
                    out_cols,
                    assoc_threads,
                    assoc_pool.as_ref(),
                    if with_plrt { Some(ml0) } else { None },
                );
                record_elapsed_nanos(&assoc_nanos, at0);

                let tt0 = Instant::now();
                text_buf.clear();
                if text_buf.capacity() < rows * 128 {
                    text_buf.reserve(rows * 128 - text_buf.capacity());
                }
                for off in 0..rows {
                    let idx = start + off;
                    let base = off * out_cols;
                    let beta = out_sub[base];
                    let se = out_sub[base + 1];
                    let pwald = sanitize_assoc_pvalue(beta, se, out_sub[base + 2]);
                    let chisq = chisq_from_beta_se_and_optional_plrt(beta, se, None);
                    let chisq_txt = format_chisq_value(chisq);
                    if with_plrt {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                            chrom[idx],
                            pos[idx],
                            snp[idx],
                            allele0[idx],
                            allele1[idx],
                            row_maf[idx],
                            miss_sub[off],
                            beta,
                            se,
                            chisq_txt,
                            pwald,
                            out_sub[base + 3]
                        );
                    } else {
                        let _ = write!(
                            text_buf,
                            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\n",
                            chrom[idx],
                            pos[idx],
                            snp[idx],
                            allele0[idx],
                            allele1[idx],
                            row_maf[idx],
                            miss_sub[off],
                            beta,
                            se,
                            chisq_txt,
                            pwald
                        );
                    }
                }
                let payload = std::mem::take(&mut text_buf).into_bytes();
                writer.send(payload).map_err(PyRuntimeError::new_err)?;
                let done = start + rows;
                if let Some(cb) = progress_callback.as_ref() {
                    if packed_progress_should_emit(done, m, progress_block, &mut next_progress_emit)
                    {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done, m))?;
                            Ok(())
                        })?;
                    }
                } else {
                    Python::attach(|py2| py2.check_signals())?;
                }
                record_elapsed_nanos(&tsv_nanos, tt0);
                Ok(())
            },
        )?;

        let wt0 = Instant::now();
        let writer_result = writer.finish().map_err(PyRuntimeError::new_err);
        record_elapsed_nanos(&tsv_nanos, wt0);
        writer_result?;

        let decode_secs = elapsed_nanos_to_secs(&decode_nanos);
        let proj_secs = elapsed_nanos_to_secs(&proj_nanos);
        let assoc_secs = elapsed_nanos_to_secs(&assoc_nanos);
        let tsv_secs = elapsed_nanos_to_secs(&tsv_nanos);

        if stage_timing {
            let total_secs = total_t0.elapsed().as_secs_f64();
            let other_secs = (total_secs - decode_secs - proj_secs - assoc_secs - tsv_secs)
                .max(0.0_f64);
            let to_pct = |x: f64| -> f64 {
                if total_secs > 0.0 {
                    x * 100.0 / total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "FvLMM packed timing: decode={:.3}s ({:.1}%), proj={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), tsv={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, threads={}",
                decode_secs,
                to_pct(decode_secs),
                proj_secs,
                to_pct(proj_secs),
                assoc_secs,
                to_pct(assoc_secs),
                tsv_secs,
                to_pct(tsv_secs),
                other_secs,
                to_pct(other_secs),
                total_secs,
                m,
                n,
                if threads > 0 { threads } else { rayon::current_num_threads() }
            );
        }
        Ok(())
    })?;

    Ok((lbd, ml0, reml0))
}
