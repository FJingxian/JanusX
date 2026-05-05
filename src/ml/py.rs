use crate::ml::common::{
    matrix_i8_to_binary_rows, parse_response, readonly_f64_to_vec, topk_indices, validate_xy,
};
use crate::ml::engine::{compute_feature_scores, parse_ml_engine};
use crate::ml::extra_trees::ExtraTreesConfig;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction(name = "garfield_ml_feature_scores")]
#[pyo3(signature = (
    x,
    y,
    response="continuous",
    method="auto",
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=1,
    bootstrap=true,
    seed=0,
    feature_subsample=0.0
))]
pub fn garfield_ml_feature_scores_py(
    x: PyReadonlyArray2<'_, i8>,
    y: PyReadonlyArray1<'_, f64>,
    response: &str,
    method: &str,
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    bootstrap: bool,
    seed: u64,
    feature_subsample: f64,
) -> PyResult<Vec<f64>> {
    let resp = parse_response(response).map_err(PyRuntimeError::new_err)?;
    let engine = parse_ml_engine(method).map_err(PyRuntimeError::new_err)?;
    let (x_rows, _m, _n) = matrix_i8_to_binary_rows(x).map_err(PyRuntimeError::new_err)?;
    let y_vec = readonly_f64_to_vec(&y);
    validate_xy(&x_rows, &y_vec, resp).map_err(PyRuntimeError::new_err)?;

    let cfg = ExtraTreesConfig {
        n_estimators: n_estimators.max(1),
        max_depth: max_depth.max(1),
        min_samples_leaf: min_samples_leaf.max(1),
        min_samples_split: (min_samples_leaf.max(1) * 2).max(2),
        bootstrap,
        feature_subsample,
        seed,
    };
    compute_feature_scores(&x_rows, &y_vec, resp, engine, cfg).map_err(PyRuntimeError::new_err)
}

#[pyfunction(name = "garfield_ml_select_topk")]
#[pyo3(signature = (
    x,
    y,
    topk=20,
    response="continuous",
    method="auto",
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=1,
    bootstrap=true,
    seed=0,
    feature_subsample=0.0
))]
pub fn garfield_ml_select_topk_py(
    x: PyReadonlyArray2<'_, i8>,
    y: PyReadonlyArray1<'_, f64>,
    topk: usize,
    response: &str,
    method: &str,
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    bootstrap: bool,
    seed: u64,
    feature_subsample: f64,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let scores = garfield_ml_feature_scores_py(
        x,
        y,
        response,
        method,
        n_estimators,
        max_depth,
        min_samples_leaf,
        bootstrap,
        seed,
        feature_subsample,
    )?;
    let idx = topk_indices(&scores, topk);
    let picked_scores = idx.iter().map(|&i| scores[i]).collect::<Vec<_>>();
    Ok((idx, picked_scores))
}
