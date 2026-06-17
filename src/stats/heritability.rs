use crate::cholesky::{
    sparse_cholesky_analyze_jxgrm_csc, SparseJxgrmCholesky, SparseJxgrmCholeskyAnalysis,
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
struct EncodedFactorSummary {
    name: String,
    n_obs: usize,
    n_unique: usize,
    max_code: usize,
    dense_zero_based: bool,
}

impl EncodedFactorSummary {
    fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("name", self.name.as_str())?;
        out.set_item("n_obs", self.n_obs)?;
        out.set_item("n_unique", self.n_unique)?;
        out.set_item("max_code", self.max_code)?;
        out.set_item("dense_zero_based", self.dense_zero_based)?;
        Ok(out)
    }
}

#[derive(Clone, Debug)]
enum WorkflowTermKind {
    Codes(EncodedFactorSummary),
    Matrix { n_rows: usize, n_cols: usize },
}

#[derive(Clone, Debug)]
struct WorkflowTermSummary {
    name: String,
    kind: WorkflowTermKind,
}

impl WorkflowTermSummary {
    fn n_cols(&self) -> usize {
        match &self.kind {
            WorkflowTermKind::Codes(f) => f.n_unique,
            WorkflowTermKind::Matrix { n_cols, .. } => *n_cols,
        }
    }

    fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("name", self.name.as_str())?;
        match &self.kind {
            WorkflowTermKind::Codes(factor) => {
                out.set_item("kind", "codes")?;
                out.set_item("n_cols", factor.n_unique)?;
                out.set_item("factor", factor.to_pydict(py)?)?;
            }
            WorkflowTermKind::Matrix { n_rows, n_cols } => {
                out.set_item("kind", "matrix")?;
                out.set_item("n_rows", *n_rows)?;
                out.set_item("n_cols", *n_cols)?;
            }
        }
        Ok(out)
    }
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

fn combine_observed_pair_factor(
    name: &str,
    left: &OneHotFactorCache,
    right: &OneHotFactorCache,
) -> Result<OneHotFactorCache, String> {
    if left.codes.len() != right.codes.len() {
        return Err(format!(
            "{name} pair factor length mismatch: left={}, right={}",
            left.codes.len(),
            right.codes.len()
        ));
    }
    let mut map = BTreeMap::<(usize, usize), usize>::new();
    let mut codes = Vec::with_capacity(left.codes.len());
    for i in 0..left.codes.len() {
        let key = (left.codes[i], right.codes[i]);
        let dense = if let Some(existing) = map.get(&key).copied() {
            existing
        } else {
            let next = map.len();
            map.insert(key, next);
            next
        };
        codes.push(dense);
    }
    Ok(OneHotFactorCache {
        name: name.to_string(),
        codes,
        n_levels: map.len(),
    })
}

fn extract_i64_array_to_vec(arr: PyReadonlyArray1<'_, i64>) -> PyResult<Vec<i64>> {
    match arr.as_slice() {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => Ok(arr.as_array().iter().copied().collect()),
    }
}

fn extract_f64_array1_to_vec(arr: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    match arr.as_slice() {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => Ok(arr.as_array().iter().copied().collect()),
    }
}

fn extract_f64_array2_row_major(arr: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<f64>> {
    match arr.as_slice() {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => Ok(arr.as_array().iter().copied().collect()),
    }
}

fn extract_optional_dense_design_row_major<'py>(
    job: &Bound<'py, PyDict>,
    key: &str,
    expected_n: usize,
) -> PyResult<(Vec<f64>, usize)> {
    let Some(obj) = opt_item(job, key)? else {
        return Ok((Vec::new(), 0));
    };
    if obj.is_none() {
        return Ok((Vec::new(), 0));
    }
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

fn extract_optional_factor_cache<'py>(
    job: &Bound<'py, PyDict>,
    key: &str,
    name: &str,
    expected_n: usize,
) -> PyResult<Option<OneHotFactorCache>> {
    let Some(obj) = opt_item(job, key)? else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    let arr: PyReadonlyArray1<'py, i64> = obj.extract()?;
    let raw = extract_i64_array_to_vec(arr)?;
    normalize_codes_from_i64(name, &raw, expected_n)
        .map(Some)
        .map_err(PyValueError::new_err)
}

fn extract_required_factor_cache<'py>(
    job: &Bound<'py, PyDict>,
    key: &str,
    name: &str,
    expected_n: usize,
) -> PyResult<OneHotFactorCache> {
    let obj = req_item(job, key)?;
    let arr: PyReadonlyArray1<'py, i64> = obj.extract()?;
    let raw = extract_i64_array_to_vec(arr)?;
    normalize_codes_from_i64(name, &raw, expected_n).map_err(PyValueError::new_err)
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
    let analysis = Arc::new(sparse_cholesky_analyze_jxgrm_csc(&ztz_csc)?);
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

#[derive(Clone, Debug)]
struct StageSummary {
    ready: bool,
    reason: String,
    intercept_cols: usize,
    fixed_cols: usize,
    special_cols: usize,
    random_cols: usize,
}

impl StageSummary {
    fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("ready", self.ready)?;
        out.set_item("reason", self.reason.as_str())?;
        out.set_item("intercept_cols", self.intercept_cols)?;
        out.set_item("fixed_cols", self.fixed_cols)?;
        out.set_item("special_cols", self.special_cols)?;
        out.set_item("random_cols", self.random_cols)?;
        Ok(out)
    }
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

fn build_sparse_onehot_model_cache<'py>(
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
    let (x_broad_fixed, x_fixed_cols) =
        extract_optional_dense_design_row_major(job, "x_broad_fixed", n_obs)?;
    let mut x_row_major = vec![1.0_f64; n_obs * (x_fixed_cols + 1)];
    for i in 0..n_obs {
        let dst = &mut x_row_major[i * (x_fixed_cols + 1)..(i + 1) * (x_fixed_cols + 1)];
        for c in 0..x_fixed_cols {
            dst[c + 1] = x_broad_fixed[i * x_fixed_cols + c];
        }
    }
    let p_fixed = x_fixed_cols + 1;

    let line_factor = extract_required_factor_cache(job, "line_codes", "line_codes", n_obs)?;
    let env_factor = extract_optional_factor_cache(job, "env_codes", "env_codes", n_obs)?;
    let mut factors = vec![line_factor];
    let mut random_term_names = vec!["line_codes".to_string()];

    if let Some(env) = env_factor.as_ref() {
        if env.n_levels > 1 {
            let gxe = combine_observed_pair_factor("line_codes_x_env_codes", &factors[0], env)
                .map_err(PyValueError::new_err)?;
            if gxe.n_levels > 0 {
                random_term_names.push(gxe.name.clone());
                factors.push(gxe);
            }
        }
    }

    let extra_terms = extract_onehot_random_factor_caches(job, "broad_random_terms", n_obs)?;
    for factor in extra_terms.into_iter() {
        random_term_names.push(factor.name.clone());
        factors.push(factor);
    }
    build_sparse_onehot_model_cache_core(y, x_row_major, p_fixed, random_term_names, factors)
        .map_err(PyRuntimeError::new_err)
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
pub fn prepare_heritability_broad_sparse_cache<'py>(
    py: Python<'py>,
    job: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let cache = build_sparse_onehot_model_cache(job)?;
    Py::new(
        py,
        SparseOneHotBlupCache {
            cache: Arc::new(cache),
        },
    )
    .map(|obj| obj.into_bound(py).into_any())
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

fn extract_trait_name(job: &Bound<'_, PyDict>) -> PyResult<String> {
    if let Some(obj) = opt_item(job, "trait_name")? {
        let raw: String = obj.extract()?;
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    Ok("trait".to_string())
}

fn extract_maxiter(job: &Bound<'_, PyDict>) -> PyResult<usize> {
    if let Some(obj) = opt_item(job, "maxiter")? {
        let raw: usize = obj.extract()?;
        if raw == 0 {
            return Err(PyValueError::new_err("maxiter must be > 0"));
        }
        return Ok(raw);
    }
    Ok(100)
}

fn summarize_codes(name: &str, codes: &[i64], expected_n: usize) -> PyResult<EncodedFactorSummary> {
    if codes.len() != expected_n {
        return Err(PyValueError::new_err(format!(
            "{name} length mismatch: got {}, expected {expected_n}",
            codes.len()
        )));
    }
    if codes.is_empty() {
        return Err(PyValueError::new_err(format!("{name} must not be empty")));
    }
    let mut uniq = BTreeSet::new();
    let mut max_code = 0usize;
    for &raw in codes.iter() {
        if raw < 0 {
            return Err(PyValueError::new_err(format!(
                "{name} contains negative code {raw}; use non-negative dense integer coding"
            )));
        }
        let code = raw as usize;
        max_code = max_code.max(code);
        uniq.insert(code);
    }
    let n_unique = uniq.len();
    let dense_zero_based = uniq.iter().copied().enumerate().all(|(i, code)| i == code);
    Ok(EncodedFactorSummary {
        name: name.to_string(),
        n_obs: expected_n,
        n_unique,
        max_code,
        dense_zero_based,
    })
}

fn extract_factor_summary<'py>(
    job: &Bound<'py, PyDict>,
    key: &str,
    name: &str,
    expected_n: usize,
) -> PyResult<Option<EncodedFactorSummary>> {
    let Some(obj) = opt_item(job, key)? else {
        return Ok(None);
    };
    let arr: PyReadonlyArray1<'py, i64> = obj.extract()?;
    let codes = arr.as_slice()?;
    summarize_codes(name, codes, expected_n).map(Some)
}

fn extract_design_cols<'py>(
    job: &Bound<'py, PyDict>,
    key: &str,
    expected_n: usize,
) -> PyResult<usize> {
    let Some(obj) = opt_item(job, key)? else {
        return Ok(0);
    };
    let arr: PyReadonlyArray2<'py, f64> = obj.extract()?;
    let shape = arr.shape();
    if shape.len() != 2 || shape[0] != expected_n {
        return Err(PyValueError::new_err(format!(
            "{key} must have shape ({expected_n}, p), got {:?}",
            shape
        )));
    }
    Ok(shape[1])
}

fn extract_random_terms<'py>(
    job: &Bound<'py, PyDict>,
    key: &str,
    expected_n: usize,
) -> PyResult<Vec<WorkflowTermSummary>> {
    let Some(obj) = opt_item(job, key)? else {
        return Ok(Vec::new());
    };
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
            let factor = summarize_codes(&name, arr.as_slice()?, expected_n)?;
            out.push(WorkflowTermSummary {
                name,
                kind: WorkflowTermKind::Codes(factor),
            });
            continue;
        }
        if let Some(matrix_obj) = term.get_item("matrix")? {
            let arr: PyReadonlyArray2<'py, f64> = matrix_obj.extract()?;
            let shape = arr.shape();
            if shape.len() != 2 || shape[0] != expected_n {
                return Err(PyValueError::new_err(format!(
                    "{key}[{idx}].matrix must have shape ({expected_n}, q), got {:?}",
                    shape
                )));
            }
            out.push(WorkflowTermSummary {
                name,
                kind: WorkflowTermKind::Matrix {
                    n_rows: shape[0],
                    n_cols: shape[1],
                },
            });
            continue;
        }
        return Err(PyValueError::new_err(format!(
            "{key}[{idx}] must contain either 'codes' or 'matrix'"
        )));
    }
    Ok(out)
}

fn extract_grm_dim<'py>(
    job: &Bound<'py, PyDict>,
    expected_lines: usize,
) -> PyResult<Option<usize>> {
    let Some(obj) = opt_item(job, "grm")? else {
        return Ok(None);
    };
    let arr: PyReadonlyArray2<'py, f64> = obj.extract()?;
    let shape = arr.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err(format!(
            "grm must be square, got shape {:?}",
            shape
        )));
    }
    if shape[0] != expected_lines {
        return Err(PyValueError::new_err(format!(
            "grm dimension mismatch: got {}, expected {} (number of unique line codes)",
            shape[0], expected_lines
        )));
    }
    Ok(Some(shape[0]))
}

fn observed_pair_levels(line: &EncodedFactorSummary, env: Option<&EncodedFactorSummary>) -> usize {
    if env.is_none() {
        return 0;
    }
    line.n_unique.saturating_mul(env.unwrap().n_unique)
}

fn total_random_cols(terms: &[WorkflowTermSummary]) -> usize {
    terms.iter().map(WorkflowTermSummary::n_cols).sum()
}

fn build_stage_summary(
    n_obs: usize,
    n_lines: usize,
    fixed_cols: usize,
    special_cols: usize,
    random_cols: usize,
) -> StageSummary {
    let ready = n_obs > 1 && n_lines > 1;
    let reason = if ready {
        "validated_scaffold_ready".to_string()
    } else {
        "insufficient_observations_or_lines".to_string()
    };
    StageSummary {
        ready,
        reason,
        intercept_cols: 1,
        fixed_cols,
        special_cols,
        random_cols,
    }
}

#[pyfunction]
#[pyo3(signature = (job))]
pub fn prepare_heritability_trait_workflow<'py>(
    py: Python<'py>,
    job: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let trait_name = extract_trait_name(job)?;
    let maxiter = extract_maxiter(job)?;

    let y: PyReadonlyArray1<'py, f64> = req_item(job, "y")?.extract()?;
    let y_slice = y.as_slice()?;
    let n_obs = y_slice.len();
    if n_obs == 0 {
        return Err(PyValueError::new_err("y must not be empty"));
    }
    if y_slice.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "y contains non-finite values; filter missing observations before Rust dispatch",
        ));
    }

    let line_codes_arr: PyReadonlyArray1<'py, i64> = req_item(job, "line_codes")?.extract()?;
    let line = summarize_codes("line_codes", line_codes_arr.as_slice()?, n_obs)?;
    let env = extract_factor_summary(job, "env_codes", "env_codes", n_obs)?;

    let x_broad_fixed_cols = extract_design_cols(job, "x_broad_fixed", n_obs)?;
    let x_stage1_fixed_cols = extract_design_cols(job, "x_stage1_fixed", n_obs)?;
    let broad_random_terms = extract_random_terms(job, "broad_random_terms", n_obs)?;
    let stage1_random_terms = extract_random_terms(job, "stage1_random_terms", n_obs)?;
    let grm_dim = extract_grm_dim(job, line.n_unique)?;

    let gxe_cols = observed_pair_levels(&line, env.as_ref());
    let broad_random_cols = line.n_unique + gxe_cols + total_random_cols(&broad_random_terms);
    let stage1_random_cols = gxe_cols + total_random_cols(&stage1_random_terms);

    let env_fixed_cols = env
        .as_ref()
        .map(|e| e.n_unique.saturating_sub(1))
        .unwrap_or(0);
    let line_fixed_cols = line.n_unique.saturating_sub(1);

    let broad = build_stage_summary(
        n_obs,
        line.n_unique,
        x_broad_fixed_cols,
        0,
        broad_random_cols,
    );
    let stage1_blue = build_stage_summary(
        n_obs,
        line.n_unique,
        x_stage1_fixed_cols,
        env_fixed_cols + line_fixed_cols,
        stage1_random_cols,
    );
    let narrow = StageSummary {
        ready: grm_dim.is_some() && line.n_unique > 1,
        reason: if grm_dim.is_some() && line.n_unique > 1 {
            "validated_scaffold_ready".to_string()
        } else {
            "disabled_without_aligned_grm".to_string()
        },
        intercept_cols: 1,
        fixed_cols: x_broad_fixed_cols,
        special_cols: 0,
        random_cols: usize::from(grm_dim.is_some()),
    };

    let random_broad_py = PyList::empty(py);
    for term in broad_random_terms.iter() {
        random_broad_py.append(term.to_pydict(py)?)?;
    }
    let random_stage1_py = PyList::empty(py);
    for term in stage1_random_terms.iter() {
        random_stage1_py.append(term.to_pydict(py)?)?;
    }

    let out = PyDict::new(py);
    out.set_item("trait_name", trait_name.as_str())?;
    out.set_item("workflow_status", "scaffold_ready")?;
    out.set_item(
        "message",
        "Rust heritability workflow interface validated; numerical broad/stage1/narrow kernels are the next migration step.",
    )?;
    out.set_item("maxiter", maxiter)?;
    out.set_item("n_obs", n_obs)?;
    out.set_item("line", line.to_pydict(py)?)?;
    match env {
        Some(ref factor) => out.set_item("env", factor.to_pydict(py)?)?,
        None => out.set_item("env", py.None())?,
    }
    out.set_item("x_broad_fixed_cols", x_broad_fixed_cols)?;
    out.set_item("x_stage1_fixed_cols", x_stage1_fixed_cols)?;
    out.set_item("broad_random_terms", random_broad_py)?;
    out.set_item("stage1_random_terms", random_stage1_py)?;
    match grm_dim {
        Some(dim) => {
            out.set_item("has_grm", true)?;
            out.set_item("grm_dim", dim)?;
        }
        None => {
            out.set_item("has_grm", false)?;
            out.set_item("grm_dim", py.None())?;
        }
    }
    out.set_item("broad_model", broad.to_pydict(py)?)?;
    out.set_item("stage1_blue", stage1_blue.to_pydict(py)?)?;
    out.set_item("narrow_lmm", narrow.to_pydict(py)?)?;
    out.set_item("gxe_observed_cols", gxe_cols)?;
    Ok(out)
}
