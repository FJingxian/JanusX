use crate::math_farmcpu::{
    decode_dense_rows_to_sample_major, decode_packed_rows_to_sample_major,
    farmcpu_ll_score_from_sample_major, farmcpu_super_keep_from_sample_major, select_lead_indices,
};
use crate::stats_common::parse_index_vec_i64;
use numpy::PyArray1;
use numpy::PyArrayMethods;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use std::borrow::Cow;

// FarmCPU helper kernels moved to `math/farmcpu.rs`.

#[pyfunction]
#[pyo3(signature = (
    sz,
    n_lead,
    pvalue,
    pos,
    packed,
    n_samples,
    row_flip,
    row_maf,
    y,
    x,
    sample_indices=None,
    delta_exp_start=-5.0,
    delta_exp_end=5.0,
    delta_step=0.1,
    svd_eps=1e-8
))]
pub fn farmcpu_rem_packed(
    py: Python<'_>,
    sz: i64,
    n_lead: usize,
    pvalue: PyReadonlyArray1<'_, f64>,
    pos: PyReadonlyArray1<'_, i64>,
    packed: PyReadonlyArray2<'_, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'_, bool>,
    row_maf: PyReadonlyArray1<'_, f32>,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    sample_indices: Option<PyReadonlyArray1<'_, i64>>,
    delta_exp_start: f64,
    delta_exp_end: f64,
    delta_step: f64,
    svd_eps: f64,
) -> PyResult<(f64, Vec<i64>)> {
    if sz <= 0 {
        return Err(PyRuntimeError::new_err("sz must be > 0"));
    }
    let pvalue = pvalue.as_slice()?;
    let pos = pos.as_slice()?;
    if pvalue.len() != pos.len() {
        return Err(PyRuntimeError::new_err(
            "pvalue and pos length mismatch in farmcpu_rem_packed",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    if m != pvalue.len() {
        return Err(PyRuntimeError::new_err(format!(
            "packed row count mismatch: packed m={m}, pvalue len={}",
            pvalue.len()
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let n = y.len();
    if x_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    let p = x_arr.shape()[1];
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    if sample_idx.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let leadidx = select_lead_indices(sz, n_lead, pvalue, pos);
    let k = leadidx.len();
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let (score, leadidx_i64) = py.detach(|| -> PyResult<(f64, Vec<i64>)> {
        let snp_pool_sample_major = decode_packed_rows_to_sample_major(
            &packed_flat,
            bytes_per_snp,
            &leadidx,
            &sample_idx,
            row_flip,
            row_maf,
        );

        let score = farmcpu_ll_score_from_sample_major(
            y,
            &x_flat,
            n,
            p,
            &snp_pool_sample_major,
            k,
            delta_exp_start,
            delta_exp_end,
            delta_step,
            svd_eps,
        )
        .map_err(PyRuntimeError::new_err)?;

        let leadidx_i64 = leadidx.iter().map(|&v| v as i64).collect::<Vec<_>>();
        Ok((score, leadidx_i64))
    })?;
    Ok((score, leadidx_i64))
}

#[pyfunction]
#[pyo3(signature = (
    row_indices,
    pvalue,
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    thr=0.7
))]
pub fn farmcpu_super_packed<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, i64>,
    pvalue: PyReadonlyArray1<'py, f64>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    thr: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ridx_raw = row_indices.as_slice()?;
    let pval = pvalue.as_slice()?;
    if ridx_raw.len() != pval.len() {
        return Err(PyRuntimeError::new_err(
            "row_indices and pvalue length mismatch in farmcpu_super_packed",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let ridx = parse_index_vec_i64(ridx_raw, m, "row_indices")?;
    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    let k = ridx.len();

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let keep = py.detach(|| {
        let sample_major = decode_packed_rows_to_sample_major(
            &packed_flat,
            bytes_per_snp,
            &ridx,
            &sample_idx,
            row_flip,
            row_maf,
        );
        farmcpu_super_keep_from_sample_major(&sample_major, n, k, pval, thr)
    });

    let out = PyArray1::<bool>::zeros(py, [k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&keep);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    sz,
    n_lead,
    pvalue,
    pos,
    g,
    y,
    x,
    delta_exp_start=-5.0,
    delta_exp_end=5.0,
    delta_step=0.1,
    svd_eps=1e-8
))]
pub fn farmcpu_rem_dense(
    py: Python<'_>,
    sz: i64,
    n_lead: usize,
    pvalue: PyReadonlyArray1<'_, f64>,
    pos: PyReadonlyArray1<'_, i64>,
    g: PyReadonlyArray2<'_, f32>,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    delta_exp_start: f64,
    delta_exp_end: f64,
    delta_step: f64,
    svd_eps: f64,
) -> PyResult<(f64, Vec<i64>)> {
    if sz <= 0 {
        return Err(PyRuntimeError::new_err("sz must be > 0"));
    }
    let pvalue = pvalue.as_slice()?;
    let pos = pos.as_slice()?;
    if pvalue.len() != pos.len() {
        return Err(PyRuntimeError::new_err(
            "pvalue and pos length mismatch in farmcpu_rem_dense",
        ));
    }

    let g_arr = g.as_array();
    if g_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("g must be 2D (m, n)"));
    }
    let m = g_arr.shape()[0];
    let n = g_arr.shape()[1];
    if m != pvalue.len() {
        return Err(PyRuntimeError::new_err(format!(
            "g row count mismatch: g m={m}, pvalue len={}",
            pvalue.len()
        )));
    }

    let y = y.as_slice()?;
    if y.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "y length mismatch: got {}, expected n={n}",
            y.len()
        )));
    }

    let x_arr = x.as_array();
    if x_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    let p = x_arr.shape()[1];
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let g_slice_opt = g.as_slice().ok();

    let leadidx = select_lead_indices(sz, n_lead, pvalue, pos);
    let k = leadidx.len();

    let (score, leadidx_i64) = py.detach(|| -> PyResult<(f64, Vec<i64>)> {
        let snp_pool_sample_major =
            decode_dense_rows_to_sample_major(&g_arr, g_slice_opt, n, &leadidx);
        let score = farmcpu_ll_score_from_sample_major(
            y,
            &x_flat,
            n,
            p,
            &snp_pool_sample_major,
            k,
            delta_exp_start,
            delta_exp_end,
            delta_step,
            svd_eps,
        )
        .map_err(PyRuntimeError::new_err)?;
        let leadidx_i64 = leadidx.iter().map(|&v| v as i64).collect::<Vec<_>>();
        Ok((score, leadidx_i64))
    })?;
    Ok((score, leadidx_i64))
}

#[pyfunction]
#[pyo3(signature = (
    row_indices,
    pvalue,
    g,
    thr=0.7
))]
pub fn farmcpu_super_dense<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, i64>,
    pvalue: PyReadonlyArray1<'py, f64>,
    g: PyReadonlyArray2<'py, f32>,
    thr: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ridx_raw = row_indices.as_slice()?;
    let pval = pvalue.as_slice()?;
    if ridx_raw.len() != pval.len() {
        return Err(PyRuntimeError::new_err(
            "row_indices and pvalue length mismatch in farmcpu_super_dense",
        ));
    }

    let g_arr = g.as_array();
    if g_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("g must be 2D (m, n)"));
    }
    let m = g_arr.shape()[0];
    let n = g_arr.shape()[1];
    let ridx = parse_index_vec_i64(ridx_raw, m, "row_indices")?;
    let k = ridx.len();
    let g_slice_opt = g.as_slice().ok();

    let keep = py.detach(|| {
        let sample_major = decode_dense_rows_to_sample_major(&g_arr, g_slice_opt, n, &ridx);
        farmcpu_super_keep_from_sample_major(&sample_major, n, k, pval, thr)
    });

    let out = PyArray1::<bool>::zeros(py, [k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&keep);
    Ok(out)
}

// =============================================================================
