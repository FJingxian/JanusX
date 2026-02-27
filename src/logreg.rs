//! AND/NOT-only logic search.
//!
//! Goal: given binary X (0/1) and binary or continuous y,
//! find the best single conjunction (AND of literals) that
//! associates with y. Literals are Xj or !Xj.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseKind {
    Binary,
    Continuous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreKind {
    LogLikelihood,
    Mse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Literal {
    pub idx: usize,
    pub negated: bool,
}

#[derive(Debug, Clone)]
pub struct AndNotFitResult {
    pub literals: Vec<Literal>,
    pub indices: Vec<usize>,
    pub indices_1based: Vec<usize>,
    pub expression: String,
    pub xcombine: Vec<u8>,
    pub score: f64,
}

// ============================================================
// PyO3-facing fit function (simple)
// ============================================================

#[pyfunction(name = "fit_best_and_not")]
#[pyo3(signature = (
    x,
    y,
    response="binary",
    score="loglik",
    max_literals=0,
    allow_empty=true,
    weights=None
))]
pub fn fit_best_and_not_py<'py>(
    py: Python<'py>,
    x: Vec<Vec<u8>>,
    y: Vec<f64>,
    response: &str,
    score: &str,
    max_literals: usize,
    allow_empty: bool,
    weights: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let response_kind = parse_response(response)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid response: {e}")))?;
    let score_kind =
        parse_score(score).map_err(|e| PyRuntimeError::new_err(format!("invalid score: {e}")))?;

    let result = fit_best_and_not(
        &x,
        &y,
        response_kind,
        score_kind,
        max_literals,
        allow_empty,
        weights.as_deref(),
    )
    .map_err(PyRuntimeError::new_err)?;

    let d = PyDict::new(py).into_bound();
    let literals: Vec<(usize, bool)> = result.literals.iter().map(|l| (l.idx, l.negated)).collect();
    d.set_item("literals", literals)?;
    d.set_item("indices", result.indices)?;
    d.set_item("indices_1based", result.indices_1based)?;
    d.set_item("expression", result.expression)?;
    d.set_item("xcombine", result.xcombine)?;
    d.set_item("score", result.score)?;
    Ok(d)
}

fn parse_response(response: &str) -> Result<ResponseKind, String> {
    let r = response.to_lowercase();
    match r.as_str() {
        "binary" | "bin" | "b" => Ok(ResponseKind::Binary),
        "continuous" | "cont" | "c" => Ok(ResponseKind::Continuous),
        _ => Err("must be 'binary' or 'continuous'".to_string()),
    }
}

fn parse_score(score: &str) -> Result<ScoreKind, String> {
    let s = score.to_lowercase();
    match s.as_str() {
        "loglik" | "loglikelihood" | "log_likelihood" | "ll" | "nll" | "negloglik" => {
            Ok(ScoreKind::LogLikelihood)
        }
        "mse" => Ok(ScoreKind::Mse),
        _ => Err("must be 'loglik' or 'mse'".to_string()),
    }
}

/// Fit the best AND/NOT conjunction.
///
/// - `max_literals`: maximum number of literals in the conjunction. If 0, no limit.
/// - `allow_empty`: allow the empty conjunction (always true -> all 1 predictions).
/// - `weights`: optional non-negative weights.
pub fn fit_best_and_not(
    x: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    score: ScoreKind,
    max_literals: usize,
    allow_empty: bool,
    weights: Option<&[f64]>,
) -> Result<AndNotFitResult, String> {
    validate_x_y(x, y, response, weights)?;
    let n = x.len();
    let p = x[0].len();
    let max_lits = if max_literals == 0 || max_literals > p {
        p
    } else {
        max_literals
    };

    let mut best_score = f64::INFINITY;
    let mut best_literals: Vec<Literal> = Vec::new();
    let mut best_pred: Vec<u8> = vec![0u8; n];

    let mut current_literals: Vec<Literal> = Vec::new();
    let base_pred = vec![1u8; n];

    dfs_search(
        0,
        x,
        y,
        response,
        score,
        weights,
        max_lits,
        allow_empty,
        &mut current_literals,
        &base_pred,
        &mut best_score,
        &mut best_literals,
        &mut best_pred,
    );

    if best_literals.is_empty() && !allow_empty {
        return Err("no valid conjunction found".to_string());
    }

    let (indices, indices_1based) = collect_indices(&best_literals);
    let expression = build_and_expression(&best_literals);

    Ok(AndNotFitResult {
        literals: best_literals,
        indices,
        indices_1based,
        expression,
        xcombine: best_pred,
        score: best_score,
    })
}

// Exhaustive DFS over variables with 3 choices: exclude / include / include negated.
fn dfs_search(
    var_idx: usize,
    x: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    score: ScoreKind,
    weights: Option<&[f64]>,
    max_lits: usize,
    allow_empty: bool,
    literals: &mut Vec<Literal>,
    pred: &[u8],
    best_score: &mut f64,
    best_literals: &mut Vec<Literal>,
    best_pred: &mut Vec<u8>,
) {
    // If prediction is already all-zero, adding more literals cannot change it.
    if pred.iter().all(|&v| v == 0) {
        if allow_empty || !literals.is_empty() {
            let score_val = score_prediction(y, pred, weights, response, score);
            update_best(
                score_val,
                literals,
                pred,
                best_score,
                best_literals,
                best_pred,
            );
        }
        return;
    }

    if var_idx == x[0].len() {
        if allow_empty || !literals.is_empty() {
            let score_val = score_prediction(y, pred, weights, response, score);
            update_best(
                score_val,
                literals,
                pred,
                best_score,
                best_literals,
                best_pred,
            );
        }
        return;
    }

    // Option 1: exclude this variable.
    dfs_search(
        var_idx + 1,
        x,
        y,
        response,
        score,
        weights,
        max_lits,
        allow_empty,
        literals,
        pred,
        best_score,
        best_literals,
        best_pred,
    );

    if literals.len() >= max_lits {
        return;
    }

    // Option 2: include positive literal.
    literals.push(Literal {
        idx: var_idx,
        negated: false,
    });
    let pos_pred = apply_literal(pred, x, var_idx, false);
    dfs_search(
        var_idx + 1,
        x,
        y,
        response,
        score,
        weights,
        max_lits,
        allow_empty,
        literals,
        &pos_pred,
        best_score,
        best_literals,
        best_pred,
    );
    literals.pop();

    // Option 3: include negated literal.
    literals.push(Literal {
        idx: var_idx,
        negated: true,
    });
    let neg_pred = apply_literal(pred, x, var_idx, true);
    dfs_search(
        var_idx + 1,
        x,
        y,
        response,
        score,
        weights,
        max_lits,
        allow_empty,
        literals,
        &neg_pred,
        best_score,
        best_literals,
        best_pred,
    );
    literals.pop();
}

fn update_best(
    score_val: f64,
    literals: &[Literal],
    pred: &[u8],
    best_score: &mut f64,
    best_literals: &mut Vec<Literal>,
    best_pred: &mut Vec<u8>,
) {
    // Prefer lower score; if tie, prefer simpler (fewer literals).
    let better = score_val < *best_score - 1e-12
        || ((score_val - *best_score).abs() <= 1e-12 && literals.len() < best_literals.len());
    if better {
        *best_score = score_val;
        *best_literals = literals.to_vec();
        *best_pred = pred.to_vec();
    }
}

fn apply_literal(pred: &[u8], x: &[Vec<u8>], var: usize, negated: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(pred.len());
    for i in 0..pred.len() {
        let v = x[i][var];
        let lit = if negated { 1 - v } else { v };
        out.push(pred[i] & lit);
    }
    out
}

fn validate_x_y(
    x: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    weights: Option<&[f64]>,
) -> Result<(usize, usize), String> {
    if x.is_empty() {
        return Err("x must have at least one row".to_string());
    }
    let n = x.len();
    let p = x[0].len();
    if p == 0 {
        return Err("x must have at least one column".to_string());
    }
    if y.len() != n {
        return Err("y length must match number of rows in x".to_string());
    }
    for row in x {
        if row.len() != p {
            return Err("x must be a rectangular matrix".to_string());
        }
        if row.iter().any(|&v| v > 1) {
            return Err("x must contain only 0/1 values".to_string());
        }
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err("weights length must match number of rows in x".to_string());
        }
        if w.iter().any(|&v| v < 0.0) {
            return Err("weights must be non-negative".to_string());
        }
        let sum_w: f64 = w.iter().sum();
        if sum_w <= 0.0 {
            return Err("sum of weights must be > 0".to_string());
        }
    }
    if matches!(response, ResponseKind::Binary) {
        for &val in y {
            if !(val == 0.0 || val == 1.0) {
                return Err("binary response must be 0/1".to_string());
            }
        }
    }
    Ok((n, p))
}

fn build_and_expression(lits: &[Literal]) -> String {
    if lits.is_empty() {
        return "1".to_string();
    }
    let mut parts = Vec::with_capacity(lits.len());
    for lit in lits {
        let var = format!("X{}", lit.idx + 1);
        if lit.negated {
            parts.push(format!("!{}", var));
        } else {
            parts.push(var);
        }
    }
    parts.join(" & ")
}

fn collect_indices(lits: &[Literal]) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = lits.iter().map(|l| l.idx).collect();
    idx.sort_unstable();
    idx.dedup();
    let idx1: Vec<usize> = idx.iter().map(|v| v + 1).collect();
    (idx, idx1)
}

fn group_stats(y: &[f64], pred: &[u8], weights: Option<&[f64]>) -> (f64, f64, f64, f64, f64) {
    let n = y.len();
    let mut w0 = 0.0;
    let mut w1 = 0.0;
    let mut wy0 = 0.0;
    let mut wy1 = 0.0;
    let mut wtot = 0.0;
    let mut wytot = 0.0;

    for i in 0..n {
        let w = weights.map(|v| v[i]).unwrap_or(1.0);
        let yi = y[i];
        wtot += w;
        wytot += w * yi;
        if pred[i] == 0 {
            w0 += w;
            wy0 += w * yi;
        } else {
            w1 += w;
            wy1 += w * yi;
        }
    }

    let overall = if wtot > 0.0 { wytot / wtot } else { 0.5 };
    let m0 = if w0 > 0.0 { wy0 / w0 } else { overall };
    let m1 = if w1 > 0.0 { wy1 / w1 } else { overall };
    (m0, m1, w0, w1, wtot)
}

fn score_prediction(
    y: &[f64],
    pred: &[u8],
    weights: Option<&[f64]>,
    response: ResponseKind,
    score: ScoreKind,
) -> f64 {
    let (m0, m1, _w0, _w1, wtot) = group_stats(y, pred, weights);
    match response {
        ResponseKind::Binary => {
            let eps = 1e-9;
            let p0 = m0.clamp(eps, 1.0 - eps);
            let p1 = m1.clamp(eps, 1.0 - eps);
            match score {
                ScoreKind::LogLikelihood => {
                    let mut nll = 0.0;
                    for i in 0..y.len() {
                        let w = weights.map(|v| v[i]).unwrap_or(1.0);
                        let p = if pred[i] == 0 { p0 } else { p1 };
                        let yi = y[i];
                        nll -= w * (yi * p.ln() + (1.0 - yi) * (1.0 - p).ln());
                    }
                    nll
                }
                ScoreKind::Mse => {
                    let mut sse = 0.0;
                    for i in 0..y.len() {
                        let w = weights.map(|v| v[i]).unwrap_or(1.0);
                        let p = if pred[i] == 0 { p0 } else { p1 };
                        let diff = y[i] - p;
                        sse += w * diff * diff;
                    }
                    sse / wtot.max(1e-12)
                }
            }
        }
        ResponseKind::Continuous => {
            let mu0 = m0;
            let mu1 = m1;
            let mut sse = 0.0;
            for i in 0..y.len() {
                let w = weights.map(|v| v[i]).unwrap_or(1.0);
                let mu = if pred[i] == 0 { mu0 } else { mu1 };
                let diff = y[i] - mu;
                sse += w * diff * diff;
            }
            match score {
                ScoreKind::LogLikelihood => {
                    let sigma2 = (sse / wtot.max(1e-12)).max(1e-12);
                    0.5 * wtot.max(1e-12) * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0)
                }
                ScoreKind::Mse => sse / wtot.max(1e-12),
            }
        }
    }
}
