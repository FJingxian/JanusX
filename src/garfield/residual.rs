use std::borrow::Cow;
use std::f64::consts::PI;

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;
use rayon::prelude::*;

use crate::brent::brent_minimize;
use crate::eigh::symmetric_eigh_f64_row_major;
use crate::linalg::{cholesky_inplace, cholesky_logdet};
use crate::stats_common::get_cached_pool;
use std::time::Instant;

#[derive(Debug)]
struct GarfieldResidualFit {
    beta: Vec<f64>,
    resid_rot: Vec<f64>,
    py_rot: Vec<f64>,
    lbd: f64,
    ml: f64,
    reml: f64,
    sigma_g2: f64,
    sigma_e2: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct GarfieldResidualResult {
    pub beta: Vec<f64>,
    pub py: Vec<f64>,
    pub residualized_y: Vec<f64>,
    pub resid: Vec<f64>,
    pub lbd: f64,
    pub ml: f64,
    pub reml: f64,
    pub sigma_g2: f64,
    pub sigma_e2: f64,
    pub pve: f64,
    pub n_samples: usize,
    pub n_fixed_effects: usize,
    pub eigh_backend: String,
    pub eigh_elapsed: f64,
    pub eigenvalues: Vec<f64>,
    pub eff_m: Option<usize>,
}

#[inline]
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

fn build_design_matrix(
    x_cov: Option<&[f64]>,
    n: usize,
    q_cov: usize,
    add_intercept: bool,
) -> Result<(Vec<f64>, usize), String> {
    let p = q_cov + if add_intercept { 1 } else { 0 };
    if p == 0 {
        return Err(
            "at least one fixed-effect column is required; set add_intercept=true or provide x_cov."
                .to_string(),
        );
    }
    let mut x = vec![0.0_f64; n * p];
    for i in 0..n {
        let base = i * p;
        if add_intercept {
            x[base] = 1.0;
            if let Some(cov) = x_cov {
                let cov_base = i * q_cov;
                for j in 0..q_cov {
                    x[base + 1 + j] = cov[cov_base + j];
                }
            }
        } else if let Some(cov) = x_cov {
            let cov_base = i * q_cov;
            for j in 0..q_cov {
                x[base + j] = cov[cov_base + j];
            }
        }
    }
    Ok((x, p))
}

fn rotate_design_and_response(
    u: &[f64],
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    threads: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let dim = p + 1;
    let mut utxy = vec![0.0_f64; n * dim];
    let mut run = || {
        utxy.par_chunks_mut(dim).enumerate().for_each(|(k, row)| {
            for c in 0..p {
                let mut acc = 0.0_f64;
                for i in 0..n {
                    acc += u[i * n + k] * x[i * p + c];
                }
                row[c] = acc;
            }
            let mut accy = 0.0_f64;
            for i in 0..n {
                accy += u[i * n + k] * y[i];
            }
            row[p] = accy;
        });
    };
    if let Some(p0) = &pool {
        p0.install(run);
    } else {
        run();
    }

    let mut x_rot = vec![0.0_f64; n * p];
    let mut y_rot = vec![0.0_f64; n];
    for k in 0..n {
        let src = &utxy[k * dim..(k + 1) * dim];
        x_rot[k * p..(k + 1) * p].copy_from_slice(&src[..p]);
        y_rot[k] = src[p];
    }
    Ok((x_rot, y_rot))
}

fn project_back_from_eigenvectors(
    u: &[f64],
    vec_rot: &[f64],
    n: usize,
    threads: usize,
) -> Result<Vec<f64>, String> {
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let mut out = vec![0.0_f64; n];
    let mut run = || {
        out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += u[i * n + k] * vec_rot[k];
            }
            *out_i = acc;
        });
    };
    if let Some(p0) = &pool {
        p0.install(run);
    } else {
        run();
    }
    Ok(out)
}

fn exact_lmm_null_fit_from_rotated(
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    p: usize,
    log10_lbd: f64,
) -> Option<GarfieldResidualFit> {
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 || n <= p {
        return None;
    }

    let mut v = vec![0.0_f64; n];
    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if !vv.is_finite() || vv <= 0.0 {
            return None;
        }
        v[i] = vv;
        vinv[i] = 1.0 / vv;
    }

    let mut xtv_inv_x = vec![0.0_f64; p * p];
    let mut xtv_inv_y = vec![0.0_f64; p];
    for i in 0..n {
        let vi = vinv[i];
        let yi = y_rot[i];
        let base = i * p;
        for r in 0..p {
            let xir = x_rot[base + r];
            xtv_inv_y[r] += vi * xir * yi;
            for c in 0..=r {
                xtv_inv_x[r * p + c] += vi * xir * x_rot[base + c];
            }
        }
    }

    let ridge = 1e-6_f64;
    for r in 0..p {
        xtv_inv_x[r * p + r] += ridge;
        for c in 0..r {
            xtv_inv_x[c * p + r] = xtv_inv_x[r * p + c];
        }
    }
    if cholesky_inplace(&mut xtv_inv_x, p).is_none() {
        return None;
    }
    let log_det_xtv_inv_x = cholesky_logdet(&xtv_inv_x, p);
    let beta = cholesky_solve(&xtv_inv_x, p, &xtv_inv_y);

    let mut resid_rot = vec![0.0_f64; n];
    let mut py_rot = vec![0.0_f64; n];
    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        let base = i * p;
        let mut xb = 0.0_f64;
        for r in 0..p {
            xb += x_rot[base + r] * beta[r];
        }
        let ri = y_rot[i] - xb;
        resid_rot[i] = ri;
        py_rot[i] = vinv[i] * ri;
        rtv_invr += vinv[i] * ri * ri;
    }
    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return None;
    }

    let n_f = n as f64;
    let p_f = p as f64;
    let log_det_v: f64 = v.iter().map(|vv| vv.ln()).sum();
    let reml_total = (n_f - p_f) * rtv_invr.ln() + log_det_v + log_det_xtv_inv_x;
    let reml_const = (n_f - p_f) * ((n_f - p_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let reml = reml_const - 0.5 * reml_total;
    let ml_total = n_f * rtv_invr.ln() + log_det_v;
    let ml_const = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let ml = ml_const - 0.5 * ml_total;
    if !reml.is_finite() || !ml.is_finite() {
        return None;
    }

    let sigma_g2 = rtv_invr / (n_f - p_f);
    let sigma_e2 = lbd * sigma_g2;
    if !(sigma_g2.is_finite() && sigma_g2 > 0.0 && sigma_e2.is_finite() && sigma_e2 > 0.0) {
        return None;
    }

    Some(GarfieldResidualFit {
        beta,
        resid_rot,
        py_rot,
        lbd,
        ml,
        reml,
        sigma_g2,
        sigma_e2,
    })
}

fn exact_reml_loglike_from_rotated(
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    p: usize,
    log10_lbd: f64,
) -> f64 {
    exact_lmm_null_fit_from_rotated(s, x_rot, y_rot, n, p, log10_lbd)
        .map(|fit| fit.reml)
        .unwrap_or(-1e100_f64)
}

fn validate_grm(grm: &[f64], n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("GRM must not be empty.".to_string());
    }
    for i in 0..n {
        if !grm[i * n + i].is_finite() {
            return Err(format!(
                "GRM diagonal contains non-finite value at row {i}."
            ));
        }
        for j in 0..=i {
            let a = grm[i * n + j];
            let b = grm[j * n + i];
            if !a.is_finite() || !b.is_finite() {
                return Err(format!("GRM contains non-finite value at ({i}, {j})."));
            }
            if (a - b).abs() > 1e-8_f64 {
                return Err(format!("GRM must be symmetric; mismatch at ({i}, {j})."));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn garfield_residualize_exact_from_grm_rust(
    grm_vec: Vec<f64>,
    n: usize,
    y_vec: Vec<f64>,
    x_cov_vec: Option<Vec<f64>>,
    q_cov: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    eff_m: Option<usize>,
) -> Result<GarfieldResidualResult, String> {
    if n > exact_n_max {
        return Err(format!(
            "garfield exact residualization currently supports n <= {exact_n_max}; got n={n}. Large-sample HE/PCG residualization is not implemented yet."
        ));
    }
    if y_vec.len() != n {
        return Err(format!(
            "y length mismatch: got {}, expected {n}",
            y_vec.len()
        ));
    }
    if y_vec.iter().any(|v| !v.is_finite()) {
        return Err("y contains non-finite values.".to_string());
    }
    if let Some(cov) = x_cov_vec.as_ref() {
        if cov.len() != n * q_cov {
            return Err(format!(
                "x_cov payload length mismatch: got {}, expected {}",
                cov.len(),
                n * q_cov
            ));
        }
        if cov.iter().any(|v| !v.is_finite()) {
            return Err("x_cov contains non-finite values.".to_string());
        }
    }
    validate_grm(&grm_vec, n)?;

    let (x_full, p) = build_design_matrix(x_cov_vec.as_deref(), n, q_cov, add_intercept)?;
    if n <= p {
        return Err(format!(
            "residualization requires n > rank(X); got n={n}, p={p}"
        ));
    }

    let driver_t0 = Instant::now();
    let (threads_before, _threads_in_stage, eigh_run, _threads_after) =
        crate::blas::with_eigh_thread_stage(threads, || symmetric_eigh_f64_row_major(&grm_vec, n));
    let eigh_elapsed = driver_t0.elapsed().as_secs_f64();
    let (evals, evecs, evd_backend) = eigh_run?;
    if require_lapack && evd_backend == "nalgebra" {
        return Err(
            "garfield residualization expected LAPACK backend but fell back to nalgebra"
                .to_string(),
        );
    }
    let _ = threads_before;

    if evals.len() != n || evecs.len() != n * n {
        return Err(
            "eigen-decomposition output shape mismatch during residualization.".to_string(),
        );
    }
    let mut s_vec = Vec::with_capacity(n);
    for (i, &sv) in evals.iter().enumerate() {
        if !sv.is_finite() {
            return Err(format!(
                "non-finite eigenvalue at index {i} during residualization."
            ));
        }
        s_vec.push(sv.max(0.0));
    }
    let u_vec = evecs;

    let (x_rot, y_rot) = rotate_design_and_response(&u_vec, &x_full, &y_vec, n, p, threads)?;
    let (best_log10, _best_cost) = brent_minimize(
        |log10_lbd| -exact_reml_loglike_from_rotated(&s_vec, &x_rot, &y_rot, n, p, log10_lbd),
        low,
        high,
        tol,
        max_iter,
    );
    let fit = exact_lmm_null_fit_from_rotated(&s_vec, &x_rot, &y_rot, n, p, best_log10)
        .ok_or_else(|| {
            "garfield exact residualization failed to evaluate the null model at the REML optimum."
                .to_string()
        })?;
    let resid = project_back_from_eigenvectors(&u_vec, &fit.resid_rot, n, threads)?;
    let py_vec = project_back_from_eigenvectors(&u_vec, &fit.py_rot, n, threads)?;
    let mean_s = s_vec.iter().copied().sum::<f64>() / (n as f64);
    let var_g = fit.sigma_g2 * mean_s.max(0.0);
    let denom = var_g + fit.sigma_e2;
    let pve = if denom.is_finite() && denom > 0.0 {
        var_g / denom
    } else {
        f64::NAN
    };

    Ok(GarfieldResidualResult {
        beta: fit.beta,
        py: py_vec.clone(),
        residualized_y: py_vec,
        resid,
        lbd: fit.lbd,
        ml: fit.ml,
        reml: fit.reml,
        sigma_g2: fit.sigma_g2,
        sigma_e2: fit.sigma_e2,
        pve,
        n_samples: n,
        n_fixed_effects: p,
        eigh_backend: evd_backend.to_string(),
        eigh_elapsed,
        eigenvalues: s_vec,
        eff_m,
    })
}

fn garfield_residualize_exact_from_grm_impl<'py>(
    py: Python<'py>,
    grm_vec: Vec<f64>,
    n: usize,
    y_vec: Vec<f64>,
    x_cov_vec: Option<Vec<f64>>,
    q_cov: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    eff_m: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = garfield_residualize_exact_from_grm_rust(
        grm_vec,
        n,
        y_vec,
        x_cov_vec,
        q_cov,
        threads,
        low,
        high,
        max_iter,
        tol,
        add_intercept,
        exact_n_max,
        require_lapack,
        eff_m,
    )
    .map_err(PyRuntimeError::new_err)?;

    let py_arr = PyArray1::from_owned_array(py, Array1::from_vec(result.py.clone())).into_bound();
    let resid_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(result.resid.clone())).into_bound();
    let beta_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(result.beta.clone())).into_bound();
    let s_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(result.eigenvalues.clone())).into_bound();
    let y_resid_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(result.residualized_y.clone()))
            .into_bound();

    let out = PyDict::new(py);
    out.set_item("py", py_arr)?;
    out.set_item("residualized_y", y_resid_arr)?;
    out.set_item("resid", resid_arr)?;
    out.set_item("beta", beta_arr)?;
    out.set_item("lbd", result.lbd)?;
    out.set_item("ml", result.ml)?;
    out.set_item("reml", result.reml)?;
    out.set_item("sigma_g2", result.sigma_g2)?;
    out.set_item("sigma_e2", result.sigma_e2)?;
    out.set_item("pve", result.pve)?;
    out.set_item("n_samples", result.n_samples)?;
    out.set_item("n_fixed_effects", result.n_fixed_effects)?;
    out.set_item("eigh_backend", result.eigh_backend)?;
    out.set_item("eigh_elapsed", result.eigh_elapsed)?;
    out.set_item("eigenvalues", s_arr)?;
    if let Some(m) = result.eff_m {
        out.set_item("eff_m", m)?;
    }
    Ok(out)
}

#[pyfunction(name = "garfield_residualize_grm")]
#[pyo3(signature = (
    grm,
    y,
    x_cov=None,
    threads=0,
    low=-5.0_f64,
    high=5.0_f64,
    max_iter=50_usize,
    tol=1e-3_f64,
    add_intercept=true,
    exact_n_max=15_000_usize,
    require_lapack=false
))]
pub fn garfield_residualize_grm_py<'py>(
    py: Python<'py>,
    grm: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let grm_arr = grm.as_array();
    if grm_arr.ndim() != 2 || grm_arr.shape()[0] != grm_arr.shape()[1] {
        return Err(PyRuntimeError::new_err(
            "grm must be a square float64 matrix.",
        ));
    }
    let n = grm_arr.shape()[0];
    let grm_vec: Vec<f64> = match grm.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => grm_arr.iter().copied().collect(),
    };
    let y_vec: Vec<f64> = match y.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y.as_array().iter().copied().collect(),
    };
    let (x_cov_vec, q_cov) = if let Some(xc) = x_cov {
        let xc_arr = xc.as_array();
        if xc_arr.ndim() != 2 {
            return Err(PyRuntimeError::new_err("x_cov must be 2D (n, q_cov)."));
        }
        if xc_arr.shape()[0] != n {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov row mismatch: got {}, expected {n}",
                xc_arr.shape()[0]
            )));
        }
        let q_cov = xc_arr.shape()[1];
        let cov_vec = match xc.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => xc_arr.iter().copied().collect(),
        };
        (Some(cov_vec), q_cov)
    } else {
        (None, 0usize)
    };
    garfield_residualize_exact_from_grm_impl(
        py,
        grm_vec,
        n,
        y_vec,
        x_cov_vec,
        q_cov,
        threads,
        low,
        high,
        max_iter,
        tol,
        add_intercept,
        exact_n_max,
        require_lapack,
        None,
    )
}

#[pyfunction(name = "garfield_residualize_bed")]
#[pyo3(signature = (
    prefix,
    y,
    x_cov=None,
    maf_threshold=0.02_f32,
    max_missing_rate=0.05_f32,
    block_cols=65_536_usize,
    threads=0_usize,
    progress_callback=None,
    progress_every=0_usize,
    low=-5.0_f64,
    high=5.0_f64,
    max_iter=50_usize,
    tol=1e-3_f64,
    add_intercept=true,
    exact_n_max=15_000_usize,
    require_lapack=false
))]
pub fn garfield_residualize_bed_py<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    maf_threshold: f32,
    max_missing_rate: f32,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let y_vec: Vec<f64> = match y.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y.as_array().iter().copied().collect(),
    };
    let (x_cov_vec, q_cov) = if let Some(xc) = x_cov {
        let xc_arr = xc.as_array();
        if xc_arr.ndim() != 2 {
            return Err(PyRuntimeError::new_err("x_cov must be 2D (n, q_cov)."));
        }
        let q_cov = xc_arr.shape()[1];
        let cov_vec = match xc.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => xc_arr.iter().copied().collect(),
        };
        (Some(cov_vec), q_cov)
    } else {
        (None, 0usize)
    };

    let (grm_f32_arr, eff_m, n_samples) = crate::grm::grm_packed_bed_f32(
        py,
        prefix,
        1,
        maf_threshold,
        max_missing_rate,
        block_cols,
        threads,
        progress_callback,
        progress_every,
    )?;
    if y_vec.len() != n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "y length mismatch: got {}, expected {n_samples} samples from BED input",
            y_vec.len()
        )));
    }
    if let Some(cov) = x_cov_vec.as_ref() {
        if cov.len() != n_samples * q_cov {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov payload length mismatch: got {}, expected {}",
                cov.len(),
                n_samples * q_cov
            )));
        }
    }
    let grm_f32_ro = grm_f32_arr.readonly();
    let grm_f32: Cow<[f32]> = match grm_f32_ro.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(grm_f32_ro.as_array().iter().copied().collect()),
    };
    let grm_vec: Vec<f64> = grm_f32.iter().map(|&v| v as f64).collect();
    garfield_residualize_exact_from_grm_impl(
        py,
        grm_vec,
        n_samples,
        y_vec,
        x_cov_vec,
        q_cov,
        threads,
        low,
        high,
        max_iter,
        tol,
        add_intercept,
        exact_n_max,
        require_lapack,
        Some(eff_m),
    )
}
