use crate::ml::common::{
    matrix_i8_to_binary_rows, parse_importance, parse_permutation_scoring, parse_response,
    readonly_f64_to_vec, topk_indices, validate_xy, PermutationConfig,
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
    feature_subsample=0.0,
    importance="imp",
    permutation_repeats=5,
    permutation_scoring="auto"
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
    importance: &str,
    permutation_repeats: usize,
    permutation_scoring: &str,
) -> PyResult<Vec<f64>> {
    let resp = parse_response(response).map_err(PyRuntimeError::new_err)?;
    let engine = parse_ml_engine(method).map_err(PyRuntimeError::new_err)?;
    let importance_kind = parse_importance(importance).map_err(PyRuntimeError::new_err)?;
    let perm_scoring =
        parse_permutation_scoring(permutation_scoring).map_err(PyRuntimeError::new_err)?;
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
    let perm_cfg = PermutationConfig {
        n_repeats: permutation_repeats.max(1),
        scoring: perm_scoring,
        seed,
    };
    compute_feature_scores(
        &x_rows,
        &y_vec,
        resp,
        engine,
        cfg,
        importance_kind,
        perm_cfg,
    )
    .map_err(PyRuntimeError::new_err)
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
    feature_subsample=0.0,
    importance="imp",
    permutation_repeats=5,
    permutation_scoring="auto"
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
    importance: &str,
    permutation_repeats: usize,
    permutation_scoring: &str,
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
        importance,
        permutation_repeats,
        permutation_scoring,
    )?;
    let idx = topk_indices(&scores, topk);
    let picked_scores = idx.iter().map(|&i| scores[i]).collect::<Vec<_>>();
    Ok((idx, picked_scores))
}
