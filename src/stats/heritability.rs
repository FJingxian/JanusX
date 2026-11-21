use crate::cholesky::{
    sparse_cholesky_analyze_jxgrm_csc_onehot, SparseJxgrmCholesky, SparseJxgrmCholeskyAnalysis,
};
use crate::linalg::{cholesky_inplace, cholesky_logdet, cholesky_solve_into};
use crate::spgrm::SparseGrmCsc;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const HERITABILITY_OBJ_PENALTY: f64 = 1e30_f64;

fn req_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "prepare_heritability_trait_workflow missing required key '{key}'"
        ))
    })
}

fn opt_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    d.get_item(key)
}

#[derive(Clone, Debug)]
struct OneHotFactorCache {
    name: String,
    codes: Vec<usize>,
    n_levels: usize,
}

#[derive(Clone, Debug)]
struct SparseOneHotModelCache {
    n_obs: usize,
    p_fixed: usize,
    y: Vec<f64>,
    x_row_major: Vec<f64>,
    xtx: Vec<f64>,
    xty: Vec<f64>,
    ztx: Vec<f64>, // column-major (q_random, p_fixed)
    zty: Vec<f64>,
    yty: f64,
    z_cols: Vec<usize>,
    z_col_term_index: Vec<usize>,
    random_term_names: Vec<String>,
    factors: Vec<OneHotFactorCache>,
    analysis: Arc<SparseJxgrmCholeskyAnalysis>,
}

#[derive(Clone, Debug)]
struct SparseBlupFitState {
    beta: Vec<f64>,
    u_hat: Vec<f64>,
    z_fitted: Vec<f64>,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
    vinvr: Vec<f64>,
    cov_beta: Vec<f64>,
    sigma2: f64,
    log_det_v: f64,
    log_det_xt_vinv_x: f64,
    ypy: f64,
}

struct SparseBlupProfileState {
    factor: SparseJxgrmCholesky,
    chol_xt_vinv_x: Vec<f64>,
    beta: Vec<f64>,
    inv_lbd: f64,
    log_det_v: f64,
    log_det_xt_vinv_x: f64,
    ypy: f64,
}

fn normalize_codes_from_i64(
    name: &str,
    raw: &[i64],
    expected_n: usize,
) -> Result<OneHotFactorCache, String> {
    if raw.len() != expected_n {
        return Err(format!(
            "{name} length mismatch: got {}, expected {expected_n}",
            raw.len()
        ));
    }
    if raw.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    let mut uniq = BTreeSet::new();
    for &value in raw.iter() {
        if value < 0 {
            return Err(format!(
                "{name} contains negative code {value}; use non-negative dense integer coding"
            ));
        }
        uniq.insert(value);
    }
    let mut remap = BTreeMap::new();
    for (dense, value) in uniq.iter().copied().enumerate() {
        remap.insert(value, dense);
    }
    let mut codes = Vec::with_capacity(raw.len());
    for &value in raw.iter() {
        let dense = remap
            .get(&value)
            .copied()
            .ok_or_else(|| format!("{name} internal remap failure for code {value}"))?;
        codes.push(dense);
    }
    Ok(OneHotFactorCache {
        name: name.to_string(),
        codes,
        n_levels: remap.len(),
    })
}

fn extract_i64_array_to_vec(arr: PyReadonlyArray1<'_, i64>) -> PyResult<Vec<i64>> {
    match crate::stats_common::array_standard_slice(&arr) {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => Ok(crate::stats_common::copy_array_standard_layout(
            &arr.as_array(),
        )),
    }
}

fn extract_f64_array1_to_vec(arr: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    match crate::stats_common::array_standard_slice(&arr) {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => Ok(crate::stats_common::copy_array_standard_layout(
            &arr.as_array(),
        )),
    }
}

fn extract_f64_array2_row_major(arr: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<f64>> {
    match crate::stats_common::array_standard_slice(&arr) {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => Ok(crate::stats_common::copy_array_standard_layout(
            &arr.as_array(),
        )),
    }
}

fn extract_required_dense_design_row_major<'py>(
    job: &Bound<'py, PyDict>,
    key: &str,
    expected_n: usize,
) -> PyResult<(Vec<f64>, usize)> {
    let obj = req_item(job, key)?;
    let arr: PyReadonlyArray2<'py, f64> = obj.extract()?;
    let shape = arr.shape();
    if shape.len() != 2 || shape[0] != expected_n {
        return Err(PyValueError::new_err(format!(
            "{key} must have shape ({expected_n}, p), got {:?}",
            shape
        )));
    }
    let n_cols = shape[1];
    let data = extract_f64_array2_row_major(arr)?;
    if data.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{key} contains non-finite values; filter missing observations before Rust dispatch"
        )));
    }
    Ok((data, n_cols))
}

fn extract_onehot_random_factor_caches<'py>(
    job: &Bound<'py, PyDict>,
    key: &str,
    expected_n: usize,
) -> PyResult<Vec<OneHotFactorCache>> {
    let Some(obj) = opt_item(job, key)? else {
        return Ok(Vec::new());
    };
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let list = obj.cast::<PyList>()?;
    let mut out = Vec::with_capacity(list.len());
    for (idx, item) in list.iter().enumerate() {
        let term = item.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!(
                "{key}[{idx}] must be a dict with at least a 'name' and one of {{'codes', 'matrix'}}"
            ))
        })?;
        let name = if let Some(name_obj) = term.get_item("name")? {
            let raw: String = name_obj.extract()?;
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                format!("{key}_{idx}")
            } else {
                trimmed.to_string()
            }
        } else {
            format!("{key}_{idx}")
        };
        if let Some(codes_obj) = term.get_item("codes")? {
            let arr: PyReadonlyArray1<'py, i64> = codes_obj.extract()?;
            let raw = extract_i64_array_to_vec(arr)?;
            out.push(
                normalize_codes_from_i64(&name, &raw, expected_n).map_err(PyValueError::new_err)?,
            );
            continue;
        }
        if term.get_item("matrix")?.is_some() {
            return Err(PyValueError::new_err(format!(
                "{key}[{idx}] uses dense/special random matrix input; Rust broad sparse path currently supports one-hot code terms only"
            )));
        }
        return Err(PyValueError::new_err(format!(
            "{key}[{idx}] must contain either 'codes' or 'matrix'"
        )));
    }
    Ok(out)
}

fn build_sparse_onehot_model_cache_core(
    y: Vec<f64>,
    x_row_major: Vec<f64>,
    p_fixed: usize,
    random_term_names: Vec<String>,
    factors: Vec<OneHotFactorCache>,
) -> Result<SparseOneHotModelCache, String> {
    let n_obs = y.len();
    if n_obs == 0 {
        return Err("y must not be empty".to_string());
    }
    if p_fixed == 0 {
        return Err("Sparse one-hot BLUP requires p_fixed > 0".to_string());
    }
    if x_row_major.len() != n_obs.saturating_mul(p_fixed) {
        return Err(format!(
            "x_row_major length mismatch: got {}, expected {}",
            x_row_major.len(),
            n_obs.saturating_mul(p_fixed)
        ));
    }
    if factors.is_empty() {
        return Err("Sparse one-hot BLUP requires at least one random term".to_string());
    }
    let (ztz_csc, z_cols, z_col_term_index) = build_lower_csc_from_factors(factors.as_slice())?;
    let analysis = Arc::new(sparse_cholesky_analyze_jxgrm_csc_onehot(&ztz_csc)?);
    let xtx = dense_xtx(x_row_major.as_slice(), n_obs, p_fixed);
    let xty = dense_xty(x_row_major.as_slice(), y.as_slice(), n_obs, p_fixed);
    let ztx = ztx_from_factors(factors.as_slice(), x_row_major.as_slice(), n_obs, p_fixed);
    let zty = zty_from_factors(factors.as_slice(), y.as_slice(), n_obs);
    let yty = y.iter().map(|v| v * v).sum::<f64>();
    Ok(SparseOneHotModelCache {
        n_obs,
        p_fixed,
        y,
        x_row_major,
        xtx,
        xty,
        ztx,
        zty,
        yty,
        z_cols,
        z_col_term_index,
        random_term_names,
        factors,
        analysis,
    })
}

fn build_lower_csc_from_factors(
    factors: &[OneHotFactorCache],
) -> Result<(SparseGrmCsc, Vec<usize>, Vec<usize>), String> {
    let n_cols = factors.iter().map(|f| f.n_levels).sum::<usize>();
    if n_cols == 0 {
        return Err("Sparse one-hot BLUP requires at least one random-effect column".to_string());
    }
    let mut z_cols = Vec::with_capacity(factors.len());
    let mut z_col_term_index = Vec::with_capacity(n_cols);
    let mut level_offsets = Vec::with_capacity(factors.len());
    let mut offset = 0usize;
    for (term_idx, factor) in factors.iter().enumerate() {
        z_cols.push(factor.n_levels);
        level_offsets.push(offset);
        for _ in 0..factor.n_levels {
            z_col_term_index.push(term_idx);
        }
        offset = offset.saturating_add(factor.n_levels);
    }

    let n_obs = factors
        .first()
        .map(|f| f.codes.len())
        .ok_or_else(|| "Sparse one-hot BLUP requires at least one factor".to_string())?;
    let mut active_cols = vec![0usize; factors.len()];
    let mut counts: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for obs in 0..n_obs {
        for (term_idx, factor) in factors.iter().enumerate() {
            if factor.codes.len() != n_obs {
                return Err("All one-hot factors must have the same observation length".to_string());
            }
            let code = factor.codes[obs];
            if code >= factor.n_levels {
                return Err(format!(
                    "Random term {} contains out-of-range level code {code} for n_levels={}",
                    factor.name, factor.n_levels
                ));
            }
            active_cols[term_idx] = level_offsets[term_idx] + code;
        }
        for i in 0..active_cols.len() {
            for j in 0..=i {
                let a = active_cols[i];
                let b = active_cols[j];
                let (col, row) = if a <= b { (a, b) } else { (b, a) };
                *counts.entry((col, row)).or_insert(0.0) += 1.0;
            }
        }
    }
    let mut per_col: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n_cols];
    for ((col, row), value) in counts.into_iter() {
        per_col[col].push((row as u32, value));
    }
    let mut col_ptr = vec![0u64; n_cols + 1];
    let mut row_indices = Vec::<u32>::new();
    let mut values = Vec::<f64>::new();
    for col in 0..n_cols {
        per_col[col].sort_by_key(|(row, _)| *row);
        col_ptr[col] = row_indices.len() as u64;
        for (row, value) in per_col[col].iter().copied() {
            row_indices.push(row);
            values.push(value);
        }
    }
    col_ptr[n_cols] = row_indices.len() as u64;
    let nnz = row_indices.len();
    Ok((
        SparseGrmCsc {
            n_samples: n_cols,
            nnz,
            col_ptr,
            row_indices,
            values,
        },
        z_cols,
        z_col_term_index,
    ))
}

fn dense_xtx(x_row_major: &[f64], n_obs: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; p * p];
    for i in 0..n_obs {
        let row = &x_row_major[i * p..(i + 1) * p];
        for r in 0..p {
            let xr = row[r];
            for c in 0..=r {
                out[r * p + c] += xr * row[c];
            }
        }
    }
    for r in 0..p {
        for c in 0..r {
            out[c * p + r] = out[r * p + c];
        }
    }
    out
}

fn dense_xty(x_row_major: &[f64], y: &[f64], n_obs: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; p];
    for i in 0..n_obs {
        let yi = y[i];
        let row = &x_row_major[i * p..(i + 1) * p];
        for c in 0..p {
            out[c] += row[c] * yi;
        }
    }
    out
}

fn ztx_from_factors(
    factors: &[OneHotFactorCache],
    x_row_major: &[f64],
    n_obs: usize,
    p: usize,
) -> Vec<f64> {
    let total_cols = factors.iter().map(|f| f.n_levels).sum::<usize>();
    let mut out = vec![0.0_f64; total_cols * p];
    let mut offset = 0usize;
    for factor in factors.iter() {
        for i in 0..n_obs {
            let code = factor.codes[i];
            let z_col = offset + code;
            let row = &x_row_major[i * p..(i + 1) * p];
            for c in 0..p {
                out[c * total_cols + z_col] += row[c];
            }
        }
        offset += factor.n_levels;
    }
    out
}

fn zty_from_factors(factors: &[OneHotFactorCache], y: &[f64], n_obs: usize) -> Vec<f64> {
    let total_cols = factors.iter().map(|f| f.n_levels).sum::<usize>();
    let mut out = vec![0.0_f64; total_cols];
    let mut offset = 0usize;
    for factor in factors.iter() {
        for i in 0..n_obs {
            out[offset + factor.codes[i]] += y[i];
        }
        offset += factor.n_levels;
    }
    out
}

fn compute_diag_from_theta(
    theta_z: &[f64],
    z_col_term_index: &[usize],
) -> Result<Vec<f64>, String> {
    let mut diag = vec![0.0_f64; z_col_term_index.len()];
    for (col, &term_idx) in z_col_term_index.iter().enumerate() {
        let theta = theta_z.get(term_idx).copied().ok_or_else(|| {
            format!(
                "theta_z term index out of bounds: term_idx={term_idx}, len={}",
                theta_z.len()
            )
        })?;
        if !(theta.is_finite() && theta > 0.0) {
            return Err(format!("theta_z must be finite and > 0, got {theta}"));
        }
        diag[col] = 1.0 / theta;
    }
    Ok(diag)
}

fn xbeta_from_row_major(x_row_major: &[f64], beta: &[f64], n_obs: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_obs];
    for i in 0..n_obs {
        let row = &x_row_major[i * p..(i + 1) * p];
        let mut acc = 0.0_f64;
        for c in 0..p {
            acc += row[c] * beta[c];
        }
        out[i] = acc;
    }
    out
}

fn z_times_u(factors: &[OneHotFactorCache], u_hat: &[f64], n_obs: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_obs];
    let mut offset = 0usize;
    for factor in factors.iter() {
        for i in 0..n_obs {
            out[i] += u_hat[offset + factor.codes[i]];
        }
        offset += factor.n_levels;
    }
    out
}

fn profile_sparse_blup_core(
    cache: &SparseOneHotModelCache,
    theta: &[f64],
) -> Result<SparseBlupProfileState, String> {
    if theta.len() != cache.factors.len() + 1 {
        return Err(format!(
            "theta length mismatch: got {}, expected {}",
            theta.len(),
            cache.factors.len() + 1
        ));
    }
    let theta_z = &theta[..theta.len() - 1];
    let lbd = theta[theta.len() - 1];
    if !(lbd.is_finite() && lbd > 0.0) {
        return Err(format!("lambda must be finite and > 0, got {lbd}"));
    }
    if cache.n_obs <= cache.p_fixed {
        return Err(format!(
            "Sparse BLUP requires n_obs > p_fixed, got n_obs={} and p_fixed={}",
            cache.n_obs, cache.p_fixed
        ));
    }
    let diag = compute_diag_from_theta(theta_z, &cache.z_col_term_index)?;
    let factor = cache
        .analysis
        .factorize_scaled_plus_diag(1.0 / lbd, diag.as_slice())?;
    let log_det_m = factor.logdet();

    let mut tmp_y = cache.zty.clone();
    factor.solve_in_place(&mut tmp_y, 1)?;
    let mut tmp_x = cache.ztx.clone();
    if cache.p_fixed > 0 {
        factor.solve_in_place(&mut tmp_x, cache.p_fixed)?;
    }

    let inv_lbd = 1.0 / lbd;
    let inv_lbd2 = inv_lbd * inv_lbd;
    let p = cache.p_fixed;
    let n = cache.n_obs;
    let q = cache.z_col_term_index.len();

    let mut xt_vinv_x = vec![0.0_f64; p * p];
    for r in 0..p {
        for c in 0..p {
            let mut cross = 0.0_f64;
            for z in 0..q {
                cross += cache.ztx[r * q + z] * tmp_x[c * q + z];
            }
            xt_vinv_x[r * p + c] = cache.xtx[r * p + c] * inv_lbd - cross * inv_lbd2;
        }
    }
    let mut xt_vinv_y = vec![0.0_f64; p];
    for r in 0..p {
        let mut cross = 0.0_f64;
        for z in 0..q {
            cross += cache.ztx[r * q + z] * tmp_y[z];
        }
        xt_vinv_y[r] = cache.xty[r] * inv_lbd - cross * inv_lbd2;
    }
    let zty_tmp_y = cache
        .zty
        .iter()
        .zip(tmp_y.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let y_vinv_y = cache.yty * inv_lbd - zty_tmp_y * inv_lbd2;

    let mut chol = xt_vinv_x.clone();
    if cholesky_inplace(&mut chol, p).is_none() {
        return Err("Sparse BLUP XtVinvX is not SPD".to_string());
    }
    let log_det_xt_vinv_x = cholesky_logdet(&chol, p);
    let mut beta = vec![0.0_f64; p];
    cholesky_solve_into(&chol, p, &xt_vinv_y, &mut beta);
    let ypy = y_vinv_y
        - beta
            .iter()
            .zip(xt_vinv_y.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    if !(ypy.is_finite() && ypy > 0.0) {
        return Err(format!(
            "Sparse BLUP profiled quadratic form is invalid: yPy={ypy}"
        ));
    }
    let log_det_v = (n as f64) * lbd.ln()
        + cache
            .z_col_term_index
            .iter()
            .map(|&term_idx| theta_z[term_idx].ln())
            .sum::<f64>()
        + log_det_m;
    Ok(SparseBlupProfileState {
        factor,
        chol_xt_vinv_x: chol,
        beta,
        inv_lbd,
        log_det_v,
        log_det_xt_vinv_x,
        ypy,
    })
}

fn sparse_blup_objective_value(
    cache: &SparseOneHotModelCache,
    theta: &[f64],
) -> Result<f64, String> {
    let profile = profile_sparse_blup_core(cache, theta)?;
    let df = cache.n_obs.saturating_sub(cache.p_fixed);
    if df == 0 {
        return Err("Sparse BLUP objective has zero residual degrees of freedom".to_string());
    }
    let total_log = (df as f64) * profile.ypy.ln() + profile.log_det_v + profile.log_det_xt_vinv_x;
    let c = (df as f64) * ((df as f64).ln() - 1.0 - (2.0 * std::f64::consts::PI).ln()) / 2.0;
    let out = total_log / 2.0 - c;
    if !out.is_finite() {
        return Err("Sparse BLUP objective is non-finite".to_string());
    }
    Ok(out)
}

fn fit_sparse_blup_core(
    cache: &SparseOneHotModelCache,
    theta: &[f64],
) -> Result<SparseBlupFitState, String> {
    let profile = profile_sparse_blup_core(cache, theta)?;
    let p = cache.p_fixed;
    let n = cache.n_obs;
    let q = cache.z_col_term_index.len();
    let xbeta = xbeta_from_row_major(&cache.x_row_major, &profile.beta, n, p);
    let mut r = vec![0.0_f64; n];
    for i in 0..n {
        r[i] = cache.y[i] - xbeta[i];
    }

    let mut rhs_r = cache.zty.clone();
    for c in 0..p {
        let beta_c = profile.beta[c];
        let col = &cache.ztx[c * q..(c + 1) * q];
        for z in 0..q {
            rhs_r[z] -= col[z] * beta_c;
        }
    }
    let mut tmp_r = rhs_r.clone();
    profile.factor.solve_in_place(&mut tmp_r, 1)?;

    let mut u_hat = vec![0.0_f64; tmp_r.len()];
    for i in 0..tmp_r.len() {
        u_hat[i] = tmp_r[i] * profile.inv_lbd;
    }
    let z_fitted = z_times_u(cache.factors.as_slice(), &u_hat, n);
    let mut fitted = vec![0.0_f64; n];
    let mut residuals = vec![0.0_f64; n];
    let mut vinvr = vec![0.0_f64; n];
    for i in 0..n {
        fitted[i] = xbeta[i] + z_fitted[i];
        residuals[i] = cache.y[i] - fitted[i];
        vinvr[i] = r[i] * profile.inv_lbd - z_fitted[i] * profile.inv_lbd;
    }
    let sigma2 = profile.ypy / (n.saturating_sub(p).max(1) as f64);
    let mut cov_beta = vec![0.0_f64; p * p];
    for col in 0..p {
        let mut rhs = vec![0.0_f64; p];
        rhs[col] = 1.0;
        let mut sol = vec![0.0_f64; p];
        cholesky_solve_into(&profile.chol_xt_vinv_x, p, &rhs, &mut sol);
        for row in 0..p {
            cov_beta[row * p + col] = sol[row] * sigma2;
        }
    }
    Ok(SparseBlupFitState {
        beta: profile.beta,
        u_hat,
        z_fitted,
        fitted,
        residuals,
        vinvr,
        cov_beta,
        sigma2,
        log_det_v: profile.log_det_v,
        log_det_xt_vinv_x: profile.log_det_xt_vinv_x,
        ypy: profile.ypy,
    })
}

#[pyclass]
pub struct SparseOneHotBlupCache {
    cache: Arc<SparseOneHotModelCache>,
}

#[pymethods]
impl SparseOneHotBlupCache {
    #[getter]
    fn n_obs(&self) -> usize {
        self.cache.n_obs
    }

    #[getter]
    fn p_fixed(&self) -> usize {
        self.cache.p_fixed
    }

    #[getter]
    fn random_term_names(&self) -> Vec<String> {
        self.cache.random_term_names.clone()
    }

    #[getter]
    fn z_cols(&self) -> Vec<usize> {
        self.cache.z_cols.clone()
    }

    fn objective(&self, theta: Vec<f64>) -> f64 {
        sparse_blup_objective_value(self.cache.as_ref(), theta.as_slice())
            .unwrap_or(HERITABILITY_OBJ_PENALTY)
    }

    fn fit<'py>(&self, py: Python<'py>, theta: Vec<f64>) -> PyResult<Bound<'py, PyDict>> {
        let state = fit_sparse_blup_core(self.cache.as_ref(), theta.as_slice())
            .map_err(PyRuntimeError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("beta", state.beta)?;
        out.set_item("u_hat", state.u_hat)?;
        out.set_item("z_fitted", state.z_fitted)?;
        out.set_item("fitted", state.fitted)?;
        out.set_item("residuals", state.residuals)?;
        out.set_item("vinvr", state.vinvr)?;
        out.set_item("cov_beta", state.cov_beta)?;
        out.set_item("sigma2", state.sigma2)?;
        out.set_item("log_det_v", state.log_det_v)?;
        out.set_item("log_det_xt_vinv_x", state.log_det_xt_vinv_x)?;
        out.set_item("ypy", state.ypy)?;
        Ok(out)
    }
}

fn build_generic_sparse_onehot_model_cache<'py>(
    job: &Bound<'py, PyDict>,
) -> PyResult<SparseOneHotModelCache> {
    let y_arr: PyReadonlyArray1<'py, f64> = req_item(job, "y")?.extract()?;
    let y = extract_f64_array1_to_vec(y_arr)?;
    let n_obs = y.len();
    if n_obs == 0 {
        return Err(PyValueError::new_err("y must not be empty"));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "y contains non-finite values; filter missing observations before Rust dispatch",
        ));
    }
    let (x_row_major, p_fixed) = extract_required_dense_design_row_major(job, "x", n_obs)?;
    let factors = extract_onehot_random_factor_caches(job, "random_terms", n_obs)?;
    let random_term_names = factors.iter().map(|f| f.name.clone()).collect::<Vec<_>>();
    build_sparse_onehot_model_cache_core(y, x_row_major, p_fixed, random_term_names, factors)
        .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
#[pyo3(signature = (job))]
pub fn prepare_sparse_onehot_blup_cache<'py>(
    py: Python<'py>,
    job: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let cache = build_generic_sparse_onehot_model_cache(job)?;
    Py::new(
        py,
        SparseOneHotBlupCache {
            cache: Arc::new(cache),
        },
    )
    .map(|obj| obj.into_bound(py).into_any())
}
