use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::algwas::algwas_packed_to_tsv;
use crate::farmcpu::farmcpu_packed_to_tsv;
use crate::fvlmm::{fastlmm_assoc_packed_f32_to_tsv, fvlmm_assoc_packed_f32_to_tsv};
use crate::glm::ixx_from_x_qr;
use crate::glm::lm_block_assoc_packed_to_tsv;
use crate::linalg::{chi2_sf_df1, cholesky_inplace, cholesky_solve_into};
use crate::lmm::lmm_reml_assoc_packed_f32_to_tsv;

fn req_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "gwas_packed_unified_to_tsv job missing required key '{key}'"
        ))
    })
}

fn opt_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    d.get_item(key)
}

fn call_progress(
    py: Python<'_>,
    cb: Option<&Py<PyAny>>,
    event: &str,
    job_done: usize,
    job_total: usize,
    model: &str,
    trait_name: &str,
    detail_done: usize,
    detail_total: usize,
) -> PyResult<()> {
    if let Some(c) = cb {
        c.call1(
            py,
            (
                event,
                job_done,
                job_total,
                model,
                trait_name,
                detail_done,
                detail_total,
            ),
        )?;
    }
    Ok(())
}

fn lm_null_ml_from_y_xcov(y: &[f64], xcov: &[f64], n: usize, p_cov: usize) -> Option<f64> {
    if n == 0 {
        return None;
    }
    let dim = p_cov + 1;
    if n <= dim {
        return None;
    }

    let mut xtx = vec![0.0_f64; dim * dim];
    let mut xty = vec![0.0_f64; dim];
    let mut xrow = vec![0.0_f64; dim];
    xrow[0] = 1.0;

    for i in 0..n {
        let yi = y[i];
        for c in 0..p_cov {
            xrow[c + 1] = xcov[i * p_cov + c];
        }

        for r in 0..dim {
            let xr = xrow[r];
            xty[r] += xr * yi;
            for c in 0..=r {
                xtx[r * dim + c] += xr * xrow[c];
            }
        }
    }
    for r in 0..dim {
        for c in 0..r {
            xtx[c * dim + r] = xtx[r * dim + c];
        }
    }

    let mut chol = xtx.clone();
    if cholesky_inplace(&mut chol, dim).is_none() {
        for i in 0..dim {
            chol[i * dim + i] += 1e-8;
        }
        if cholesky_inplace(&mut chol, dim).is_none() {
            return None;
        }
    }

    let mut beta = vec![0.0_f64; dim];
    cholesky_solve_into(&chol, dim, &xty, &mut beta);

    let mut rss = 0.0_f64;
    for i in 0..n {
        let mut fit = beta[0];
        for c in 0..p_cov {
            fit += beta[c + 1] * xcov[i * p_cov + c];
        }
        let r = y[i] - fit;
        rss += r * r;
    }
    if !(rss.is_finite() && rss > 0.0) {
        return None;
    }

    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * std::f64::consts::PI).ln()) / 2.0;
    Some(c_ml - 0.5 * n_f * rss.ln())
}

#[pyfunction]
#[pyo3(signature = (y, x_cov, lmm_ml0, alpha=0.05, boundary_mixture=true))]
pub fn gwas_lmm_lm_null_lrt_decision<'py>(
    _py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: PyReadonlyArray2<'py, f64>,
    lmm_ml0: f64,
    alpha: f64,
    boundary_mixture: bool,
) -> PyResult<(bool, f64, f64, f64)> {
    if !(lmm_ml0.is_finite()) {
        return Err(PyRuntimeError::new_err("lmm_ml0 must be finite"));
    }
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(PyRuntimeError::new_err("alpha must be in (0,1)"));
    }

    let y_slice = y.as_slice()?;
    let x_arr = x_cov.as_array();
    if x_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("x_cov must be 2D"));
    }
    let n = y_slice.len();
    let p_cov = x_arr.shape()[1];
    if x_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("x_cov rows must equal len(y)"));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err(
            "insufficient samples: require n > p_cov + 1",
        ));
    }
    let x_slice: Vec<f64> = match x_cov.as_slice() {
        Ok(v) => v.to_vec(),
        Err(_) => x_arr.iter().copied().collect(),
    };

    let lm_ml0 = lm_null_ml_from_y_xcov(y_slice, &x_slice, n, p_cov)
        .ok_or_else(|| PyRuntimeError::new_err("failed to compute LM null log-likelihood"))?;

    let mut lrt_stat = 2.0 * (lmm_ml0 - lm_ml0);
    if !lrt_stat.is_finite() || lrt_stat < 0.0 {
        lrt_stat = 0.0;
    }
    let mut pval = chi2_sf_df1(lrt_stat);
    if boundary_mixture {
        pval *= 0.5;
    }
    if !pval.is_finite() {
        pval = 1.0;
    }
    pval = pval.clamp(f64::MIN_POSITIVE, 1.0);

    // H0: Va = 0 (LM). If not significant, prefer LM.
    let switch_to_lm = pval >= alpha;
    Ok((switch_to_lm, lrt_stat, pval, lm_ml0))
}

#[pyfunction]
#[pyo3(signature = (models, traits, include_farmcpu=false))]
pub fn gwas_trait_model_schedule<'py>(
    py: Python<'py>,
    models: Vec<String>,
    traits: Vec<String>,
    include_farmcpu: bool,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    if traits.is_empty() {
        return Ok(out);
    }

    let mut model_list: Vec<String> = models
        .into_iter()
        .map(|m| m.trim().to_ascii_lowercase())
        .filter(|m| !m.is_empty())
        .collect();
    if include_farmcpu {
        model_list.push("farmcpu".to_string());
    }
    if model_list.is_empty() {
        return Ok(out);
    }

    let trait_total = traits.len();
    let model_last = model_list.len() - 1;
    for (trait_idx, trait_name_raw) in traits.iter().enumerate() {
        let trait_name = trait_name_raw.trim().to_string();
        for (model_idx, model_name) in model_list.iter().enumerate() {
            let item = PyDict::new(py);
            item.set_item("model", model_name.as_str())?;
            item.set_item("trait", trait_name.as_str())?;
            item.set_item("trait_idx", trait_idx)?;
            item.set_item("model_idx", model_idx)?;
            item.set_item("emit_trait_header", model_idx == 0)?;
            item.set_item(
                "emit_blank_after",
                model_idx == model_last && (trait_idx + 1) < trait_total,
            )?;
            item.set_item("is_farmcpu", model_name == "farmcpu")?;
            out.append(item)?;
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (models, traits, include_farmcpu=false, use_packed_fastlmm=true))]
pub fn gwas_trait_model_dispatch_v2<'py>(
    py: Python<'py>,
    models: Vec<String>,
    traits: Vec<String>,
    include_farmcpu: bool,
    use_packed_fastlmm: bool,
) -> PyResult<Bound<'py, PyList>> {
    let _ = use_packed_fastlmm;
    let out = PyList::empty(py);
    if traits.is_empty() {
        return Ok(out);
    }

    let mut model_list: Vec<String> = Vec::new();
    for raw in models {
        let mk = raw.trim().to_ascii_lowercase();
        let norm = match mk.as_str() {
            "lm" => Some("lm"),
            "lmm" => Some("lmm"),
            "lmm2" => Some("lmm2"),
            "fastlmm" => Some("fastlmm"),
            "fvlmm" => Some("fvlmm"),
            "algwas" => Some("algwas"),
            "splmm" => Some("splmm"),
            _ => None,
        };
        if let Some(v) = norm {
            model_list.push(v.to_string());
        }
    }
    if include_farmcpu {
        model_list.push("farmcpu".to_string());
    }
    if model_list.is_empty() {
        return Ok(out);
    }

    let trait_total = traits.len();
    let model_last = model_list.len() - 1;
    for (trait_idx, trait_name_raw) in traits.iter().enumerate() {
        let trait_name = trait_name_raw.trim().to_string();
        for (model_idx, model_name) in model_list.iter().enumerate() {
            let route = match model_name.as_str() {
                "lm" => "lm_stream",
                "lmm" => "lmm_stream",
                "lmm2" => "lmm2_stream",
                "fastlmm" => "fastlmm_stream",
                "fvlmm" => "fvlmm_stream",
                "farmcpu" => "farmcpu",
                "algwas" => "algwas",
                "splmm" => "splmm",
                _ => "unknown",
            };
            let item = PyDict::new(py);
            item.set_item("model", model_name.as_str())?;
            item.set_item("route", route)?;
            item.set_item("trait", trait_name.as_str())?;
            item.set_item("trait_idx", trait_idx)?;
            item.set_item("model_idx", model_idx)?;
            item.set_item("emit_trait_header", model_idx == 0)?;
            item.set_item(
                "emit_blank_after",
                model_idx == model_last && (trait_idx + 1) < trait_total,
            )?;
            item.set_item("is_farmcpu", model_name == "farmcpu")?;
            out.append(item)?;
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    jobs,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_missing,
    chrom,
    pos,
    snp,
    allele0,
    allele1,
    step=10000,
    threads=0,
    progress_callback=None,
    progress_every=0,
    row_indices=None,
    bed_prefix=None
))]
pub fn gwas_packed_unified_to_tsv<'py>(
    py: Python<'py>,
    jobs: Bound<'py, PyAny>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_missing: PyReadonlyArray1<'py, f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    step: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    bed_prefix: Option<&str>,
) -> PyResult<Bound<'py, PyList>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let jobs_list = jobs
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("jobs must be a list of dict task objects"))?;
    let total_jobs = jobs_list.len();
    let results = PyList::empty(py);

    for (i, job_any) in jobs_list.iter().enumerate() {
        let job = job_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("jobs[{i}] must be a dict task object")))?;
        let model: String = req_item(&job, "model")?.extract()?;
        let model_lc = model.trim().to_ascii_lowercase();
        let trait_name: String = opt_item(&job, "trait")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_default();
        let out_tsv: String = req_item(&job, "out_tsv")?.extract()?;
        call_progress(
            py,
            progress_callback.as_ref(),
            "start",
            i,
            total_jobs,
            model_lc.as_str(),
            trait_name.as_str(),
            0,
            0,
        )?;

        let step_use: usize = opt_item(&job, "step")?
            .and_then(|v| v.extract().ok())
            .unwrap_or(step)
            .max(1);
        let progress_every_use: usize = opt_item(&job, "progress_every")?
            .and_then(|v| v.extract().ok())
            .unwrap_or(progress_every);

        let result_obj = PyDict::new(py);
        result_obj.set_item("model", model_lc.as_str())?;
        result_obj.set_item("trait", trait_name.as_str())?;
        result_obj.set_item("out_tsv", out_tsv.as_str())?;

        match model_lc.as_str() {
            "lm" => {
                let y: PyReadonlyArray1<'py, f64> = req_item(&job, "y")?.extract()?;
                let x: PyReadonlyArray2<'py, f64> = req_item(&job, "x")?.extract()?;
                let y_slice = y.as_slice()?;
                let x_arr = x.as_array();
                let q0 = x_arr.shape()[1];
                if q0 == 0 {
                    return Err(PyRuntimeError::new_err(
                        "x must contain at least one column",
                    ));
                }
                let ixx: Option<PyReadonlyArray2<'py, f64>> = match opt_item(&job, "ixx")? {
                    Some(v) if !v.is_none() => Some(v.extract()?),
                    _ => None,
                };
                let sample_indices: Option<PyReadonlyArray1<'py, i64>> =
                    match opt_item(&job, "sample_indices")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let scan_progress_callback: Option<Py<PyAny>> =
                    match opt_item(&job, "scan_progress_callback")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let ixx_owned = if let Some(ixx_in) = ixx {
                    let ixx_arr = ixx_in.as_array();
                    if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
                        return Err(PyRuntimeError::new_err("ixx must be (q0,q0)"));
                    }
                    match ixx_in.as_slice() {
                        Ok(s) => s.to_vec(),
                        Err(_) => ixx_arr.iter().copied().collect(),
                    }
                } else {
                    let x_flat: Vec<f64> = match x.as_slice() {
                        Ok(s) => s.to_vec(),
                        Err(_) => x_arr.iter().copied().collect(),
                    };
                    ixx_from_x_qr(x_flat.as_slice(), y_slice.len(), q0)
                        .map_err(PyRuntimeError::new_err)?
                };
                let ixx_rows: Vec<Vec<f64>> =
                    ixx_owned.chunks(q0).map(|row| row.to_vec()).collect();
                let ixx_bound = PyArray2::from_vec2(py, &ixx_rows)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let (n_written, _scanned_rows) = lm_block_assoc_packed_to_tsv(
                    py,
                    y,
                    x,
                    ixx_bound.readonly(),
                    packed.clone(),
                    n_samples,
                    row_flip.clone(),
                    row_maf.clone(),
                    row_missing.clone(),
                    chrom.clone(),
                    pos.clone(),
                    snp.clone(),
                    allele0.clone(),
                    allele1.clone(),
                    out_tsv.as_str(),
                    0.0,
                    1.0,
                    0.0,
                    sample_indices,
                    row_indices.clone(),
                    step_use,
                    threads,
                    scan_progress_callback,
                    progress_every_use,
                    bed_prefix,
                )?;
                result_obj.set_item("written_rows", n_written)?;
            }
            "lmm" => {
                let s: PyReadonlyArray1<'py, f64> = req_item(&job, "s")?.extract()?;
                let xcov: PyReadonlyArray2<'py, f64> = req_item(&job, "xcov")?.extract()?;
                let y_rot: PyReadonlyArray1<'py, f64> = req_item(&job, "y_rot")?.extract()?;
                let u_t: PyReadonlyArray2<'py, f32> = req_item(&job, "u_t")?.extract()?;
                let sample_indices: Option<PyReadonlyArray1<'py, i64>> =
                    match opt_item(&job, "sample_indices")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let scan_progress_callback: Option<Py<PyAny>> =
                    match opt_item(&job, "scan_progress_callback")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let low: f64 = opt_item(&job, "low")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(-5.0);
                let high: f64 = opt_item(&job, "high")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(5.0);
                let max_iter: usize = opt_item(&job, "max_iter")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(50);
                let tol: f64 = opt_item(&job, "tol")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(1e-2);
                let genetic_model: String = opt_item(&job, "genetic_model")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| "add".to_string());
                let nullml: Option<f64> = opt_item(&job, "nullml")?.and_then(|v| v.extract().ok());
                let init_log10_lbd: Option<f64> =
                    opt_item(&job, "init_log10_lbd")?.and_then(|v| v.extract().ok());
                let rotate_block_rows: usize = opt_item(&job, "rotate_block_rows")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(256usize);

                let n_written = lmm_reml_assoc_packed_f32_to_tsv(
                    py,
                    packed.clone(),
                    n_samples,
                    row_flip.clone(),
                    row_maf.clone(),
                    row_missing.clone(),
                    s,
                    xcov,
                    y_rot,
                    u_t,
                    chrom.clone(),
                    pos.clone(),
                    snp.clone(),
                    allele0.clone(),
                    allele1.clone(),
                    out_tsv.as_str(),
                    sample_indices,
                    row_indices.clone(),
                    low,
                    high,
                    max_iter,
                    tol,
                    threads,
                    genetic_model.as_str(),
                    scan_progress_callback,
                    progress_every_use,
                    nullml,
                    init_log10_lbd,
                    rotate_block_rows,
                    bed_prefix,
                )?;
                result_obj.set_item("written_rows", n_written)?;
            }
            "fastlmm" | "fvlmm" => {
                let y: PyReadonlyArray1<'py, f64> = req_item(&job, "y")?.extract()?;
                let u: PyReadonlyArray2<'py, f32> = req_item(&job, "u")?.extract()?;
                let s: PyReadonlyArray1<'py, f32> = req_item(&job, "s")?.extract()?;
                let x: Option<PyReadonlyArray2<'py, f64>> = match opt_item(&job, "x")? {
                    Some(v) if !v.is_none() => Some(v.extract()?),
                    _ => None,
                };
                let sample_indices: Option<PyReadonlyArray1<'py, i64>> =
                    match opt_item(&job, "sample_indices")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let scan_progress_callback: Option<Py<PyAny>> =
                    match opt_item(&job, "scan_progress_callback")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let low: f64 = opt_item(&job, "low")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(-5.0);
                let high: f64 = opt_item(&job, "high")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(5.0);
                let max_iter: usize = opt_item(&job, "max_iter")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(50);
                let tol: f64 = opt_item(&job, "tol")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(1e-2);
                let tau: f64 = opt_item(&job, "tau")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0.0);
                let genetic_model: String = opt_item(&job, "genetic_model")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| "add".to_string());

                let fixed_lbd: Option<f64> = if model_lc == "fastlmm" || model_lc == "fvlmm" {
                    opt_item(&job, "fixed_lbd")?.and_then(|v| v.extract().ok())
                } else {
                    None
                };
                let fixed_ml0: Option<f64> = if model_lc == "fastlmm" || model_lc == "fvlmm" {
                    opt_item(&job, "fixed_ml0")?.and_then(|v| v.extract().ok())
                } else {
                    None
                };
                let rotate_block_rows: usize = opt_item(&job, "rotate_block_rows")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0usize);

                let scan_fn = if model_lc == "fvlmm" {
                    fvlmm_assoc_packed_f32_to_tsv
                } else {
                    fastlmm_assoc_packed_f32_to_tsv
                };
                let (lbd, ml0, reml0) = scan_fn(
                    py,
                    packed.clone(),
                    n_samples,
                    row_flip.clone(),
                    row_maf.clone(),
                    row_missing.clone(),
                    u,
                    s,
                    y,
                    x,
                    sample_indices,
                    low,
                    high,
                    max_iter,
                    tol,
                    tau,
                    threads,
                    genetic_model.as_str(),
                    chrom.clone(),
                    pos.clone(),
                    snp.clone(),
                    allele0.clone(),
                    allele1.clone(),
                    out_tsv.as_str(),
                    scan_progress_callback,
                    progress_every_use,
                    fixed_lbd,
                    fixed_ml0,
                    row_indices.clone(),
                    rotate_block_rows,
                    bed_prefix,
                )?;
                result_obj.set_item("lbd", lbd)?;
                result_obj.set_item("ml0", ml0)?;
                result_obj.set_item("reml0", reml0)?;
            }
            "farmcpu" => {
                let y: PyReadonlyArray1<'py, f64> = req_item(&job, "y")?.extract()?;
                let x_cov: PyReadonlyArray2<'py, f64> = req_item(&job, "x_cov")?.extract()?;
                let sample_indices: Option<PyReadonlyArray1<'py, i64>> =
                    match opt_item(&job, "sample_indices")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let scan_progress_callback: Option<Py<PyAny>> =
                    match opt_item(&job, "scan_progress_callback")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let threshold: f64 = opt_item(&job, "threshold")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0.05);
                let max_iter: usize = opt_item(&job, "max_iter")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(30);
                let qtn_bound: Option<usize> =
                    opt_item(&job, "qtn_bound")?.and_then(|v| v.extract().ok());
                let nbin: usize = opt_item(&job, "nbin")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(5);
                let szbin: Vec<f64> = opt_item(&job, "szbin")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| vec![5e5, 5e6, 5e7]);
                let pseudo_tsv: Option<String> =
                    opt_item(&job, "pseudo_tsv")?.and_then(|v| v.extract().ok());
                let raw: bool = opt_item(&job, "raw")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(false);

                let (qtn_n, pseudo_n, rows_n) = farmcpu_packed_to_tsv(
                    py,
                    y,
                    x_cov,
                    chrom.clone(),
                    pos.clone(),
                    snp.clone(),
                    allele0.clone(),
                    allele1.clone(),
                    packed.clone(),
                    n_samples,
                    row_flip.clone(),
                    row_maf.clone(),
                    row_missing.clone(),
                    out_tsv.as_str(),
                    sample_indices,
                    row_indices.clone(),
                    threshold,
                    max_iter,
                    qtn_bound,
                    nbin,
                    szbin,
                    threads,
                    scan_progress_callback,
                    pseudo_tsv.as_deref(),
                    bed_prefix,
                    raw,
                )?;
                result_obj.set_item("qtn_count", qtn_n)?;
                result_obj.set_item("pseudo_rows", pseudo_n)?;
                result_obj.set_item("written_rows", rows_n)?;
            }
            "algwas" => {
                let y: PyReadonlyArray1<'py, f64> = req_item(&job, "y")?.extract()?;
                let x_cov: PyReadonlyArray2<'py, f64> = req_item(&job, "x_cov")?.extract()?;
                let sample_indices: Option<PyReadonlyArray1<'py, i64>> =
                    match opt_item(&job, "sample_indices")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let row_indices_job: Option<PyReadonlyArray1<'py, i64>> =
                    match opt_item(&job, "row_indices")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let row_indices_use = row_indices_job.or_else(|| row_indices.clone());
                let qtn_bound: Option<usize> =
                    opt_item(&job, "qtn_bound")?.and_then(|v| v.extract().ok());
                let lambda_steps: usize = opt_item(&job, "lambda_steps")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(64);
                let lambda_min_ratio: f32 = opt_item(&job, "lambda_min_ratio")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0.001);
                let scan_step: usize = opt_item(&job, "scan_step")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(10_000);
                let scan_progress_callback: Option<Py<PyAny>> =
                    match opt_item(&job, "scan_progress_callback")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let stage1_progress_callback: Option<Py<PyAny>> =
                    match opt_item(&job, "stage1_progress_callback")? {
                        Some(v) if !v.is_none() => Some(v.extract()?),
                        _ => None,
                    };
                let pseudo_tsv: Option<String> =
                    opt_item(&job, "pseudo_tsv")?.and_then(|v| v.extract().ok());

                let out_tsv: String = req_item(&job, "out_tsv")?.extract()?;
                let (qtn_n, pseudo_n, rows_n) = algwas_packed_to_tsv(
                    py,
                    y,
                    x_cov,
                    chrom.clone(),
                    pos.clone(),
                    snp.clone(),
                    allele0.clone(),
                    allele1.clone(),
                    packed.clone(),
                    n_samples,
                    row_flip.clone(),
                    row_maf.clone(),
                    row_missing.clone(),
                    out_tsv.as_str(),
                    sample_indices,
                    row_indices_use,
                    qtn_bound,
                    lambda_steps,
                    lambda_min_ratio,
                    scan_step,
                    stage1_progress_callback,
                    threads,
                    scan_progress_callback,
                    pseudo_tsv.as_deref(),
                    bed_prefix,
                )?;
                result_obj.set_item("qtn_count", qtn_n)?;
                result_obj.set_item("pseudo_rows", pseudo_n)?;
                result_obj.set_item("written_rows", rows_n)?;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "jobs[{i}] has unsupported model '{}'; expected one of: lm, lmm, fastlmm, fvlmm, farmcpu, algwas",
                    model_lc
                )));
            }
        }

        call_progress(
            py,
            progress_callback.as_ref(),
            "done",
            i + 1,
            total_jobs,
            model_lc.as_str(),
            trait_name.as_str(),
            0,
            0,
        )?;
        results.append(result_obj)?;
    }

    Ok(results)
}
