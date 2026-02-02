//! Logic regression and DNF search (single-file integration).
//!
//! This file merges the former `lib.rs` and `logicreg_like.rs` so the
//! algorithm is easier to read in one place. The PyO3 bindings remain
//! in a separate file (`py.rs`).
//!
//! High-level pipeline (LogicReg-like tree search):
//! 1) Build an initial forest of empty Boolean trees.
//! 2) Evaluate trees on X to get 0/1 predictions per sample.
//! 3) Fit a small regression on tree outputs (and optional covariates).
//! 4) Score by deviance (binary) or RSS (continuous), with penalties.
//! 5) Propose/accept moves (greedy/fast/anneal) to improve the score.
//! 6) Return the best forest, expression, and predictions.
//!
//! DNF annealing pipeline is similar, but the model is a set of DNF terms
//! (OR of AND literals) instead of tree structures.
#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Literal {
    pub idx: usize,
    pub negated: bool,
}

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

#[derive(Debug, Clone, Copy)]
pub struct MoveWeights {
    pub add_literal: f64,
    pub remove_literal: f64,
    pub flip_literal: f64,
    pub add_term: f64,
    pub remove_term: f64,
}

impl Default for MoveWeights {
    fn default() -> Self {
        Self {
            add_literal: 1.0,
            remove_literal: 1.0,
            flip_literal: 1.0,
            add_term: 1.0,
            remove_term: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnealOptions {
    pub max_terms: usize,
    pub max_literals_per_term: usize,
    pub iters: usize,
    pub start_temp: f64,
    pub end_temp: f64,
    pub seed: u64,
    pub move_weights: MoveWeights,
    pub restarts: usize,
}

impl Default for AnnealOptions {
    fn default() -> Self {
        Self {
            max_terms: 3,
            max_literals_per_term: 3,
            iters: 1000,
            start_temp: 2.0,
            end_temp: 0.01,
            seed: 42,
            move_weights: MoveWeights::default(),
            restarts: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BaggingOptions {
    pub n_models: usize,
    pub replace: bool,
    pub sub_frac: f64,
    pub seed: u64,
}

impl Default for BaggingOptions {
    fn default() -> Self {
        Self {
            n_models: 25,
            replace: true,
            sub_frac: 0.632,
            seed: 123,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FitResult {
    pub terms: Vec<Vec<Literal>>,
    pub indices: Vec<usize>,
    pub indices_1based: Vec<usize>,
    pub expression: String,
    pub xcombine: Vec<u8>,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct BaggingResult {
    pub models: Vec<FitResult>,
    pub combined_pred: Vec<f64>,
    pub combined_pred_binary: Vec<u8>,
}

#[derive(Debug, Clone)]
pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub(crate) fn new(seed: u64) -> Self {
        let init = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: init }
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    pub(crate) fn next_f64(&mut self) -> f64 {
        // Uniform in [0, 1) to avoid generating upper-bound indices.
        self.next_u32() as f64 / (u32::MAX as f64 + 1.0)
    }

    pub(crate) fn gen_range(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_f64() * upper as f64) as usize
        }
    }

    pub(crate) fn gen_bool(&mut self) -> bool {
        (self.next_u32() & 1) == 1
    }
}


fn literal_value(x: u8, negated: bool) -> u8 {
    if negated { 1 - x } else { x }
}

fn build_term_expression(lits: &[Literal]) -> String {
    if lits.is_empty() {
        return "".to_string();
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

fn build_expression_from_terms(terms: &[Vec<Literal>]) -> String {
    if terms.is_empty() {
        return "".to_string();
    }
    let multi = terms.len() > 1;
    let mut parts = Vec::with_capacity(terms.len());
    for term in terms {
        let t = build_term_expression(term);
        if multi && term.len() > 1 {
            parts.push(format!("({})", t));
        } else {
            parts.push(t);
        }
    }
    parts.join(" | ")
}

/// Validate input dimensions, data types, and optional weights.
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

/// Predict a single conjunction (AND of literals) as a 0/1 vector.
fn predict_from_literals(x: &[Vec<u8>], lits: &[Literal]) -> Result<Vec<u8>, String> {
    predict_from_terms(x, &[lits.to_vec()])
}

/// Predict from a DNF model: OR over terms, each term is an AND of literals.
fn predict_from_terms(x: &[Vec<u8>], terms: &[Vec<Literal>]) -> Result<Vec<u8>, String> {
    if terms.is_empty() {
        return Err("terms must contain at least one term".to_string());
    }
    let (n, p) = validate_x_y(x, &vec![0.0; x.len()], ResponseKind::Continuous, None)?;
    for term in terms {
        if term.is_empty() {
            return Err("term must contain at least one literal".to_string());
        }
        for lit in term {
            if lit.idx >= p {
                return Err("literal index out of bounds".to_string());
            }
        }
    }

    let mut out = vec![0u8; n];
    for i in 0..n {
        let mut row_val = 0u8;
        for term in terms {
            let mut term_val = 1u8;
            for lit in term {
                let v = literal_value(x[i][lit.idx], lit.negated);
                term_val &= v;
                if term_val == 0 {
                    break;
                }
            }
            if term_val == 1 {
                row_val = 1;
                break;
            }
        }
        out[i] = row_val;
    }
    Ok(out)
}

fn collect_indices(terms: &[Vec<Literal>]) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = terms.iter().flat_map(|t| t.iter().map(|l| l.idx)).collect();
    idx.sort_unstable();
    idx.dedup();
    let idx1: Vec<usize> = idx.iter().map(|v| v + 1).collect();
    (idx, idx1)
}

fn group_stats(
    y: &[f64],
    pred: &[u8],
    weights: Option<&[f64]>,
) -> (f64, f64, f64, f64, f64) {
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

/// Score a binary/continuous response given 0/1 predictions.
fn score_model(
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

/// Geometric annealing schedule from start to end temperature.
fn temperature(start: f64, end: f64, t: usize, iters: usize) -> f64 {
    if iters <= 1 {
        return end.max(1e-12);
    }
    let ratio = end / start;
    let frac = t as f64 / (iters - 1) as f64;
    let temp = start * ratio.powf(frac);
    temp.max(1e-12)
}

fn choose_move(weights: MoveWeights, rng: &mut SimpleRng) -> u8 {
    let mut items = [
        weights.add_literal,
        weights.remove_literal,
        weights.flip_literal,
        weights.add_term,
        weights.remove_term,
    ];
    let mut total: f64 = items.iter().sum();
    if total <= 0.0 {
        items = [1.0, 1.0, 1.0, 1.0, 1.0];
        total = 5.0;
    }
    let mut r = rng.next_f64() * total;
    for (i, w) in items.iter().enumerate() {
        if r <= *w {
            return i as u8;
        }
        r -= *w;
    }
    0
}

/// Propose a neighboring DNF by random add/remove/flip moves.
fn propose_terms(
    current: &[Vec<Literal>],
    p: usize,
    opts: &AnnealOptions,
    rng: &mut SimpleRng,
) -> Vec<Vec<Literal>> {
    let mut terms = current.to_vec();
    if terms.is_empty() {
        terms.push(vec![Literal {
            idx: rng.gen_range(p),
            negated: rng.gen_bool(),
        }]);
        return terms;
    }

    for _ in 0..30 {
        let move_type = choose_move(opts.move_weights, rng);
        match move_type {
            0 => {
                // add literal
                let t = rng.gen_range(terms.len());
                if terms[t].len() >= opts.max_literals_per_term {
                    continue;
                }
                let mut tries = 0;
                while tries < p {
                    let idx = rng.gen_range(p);
                    if !terms[t].iter().any(|l| l.idx == idx) {
                        terms[t].push(Literal {
                            idx,
                            negated: rng.gen_bool(),
                        });
                        return terms;
                    }
                    tries += 1;
                }
            }
            1 => {
                // remove literal
                let t = rng.gen_range(terms.len());
                if terms[t].len() <= 1 {
                    continue;
                }
                let k = rng.gen_range(terms[t].len());
                terms[t].remove(k);
                return terms;
            }
            2 => {
                // flip literal
                let t = rng.gen_range(terms.len());
                let k = rng.gen_range(terms[t].len());
                terms[t][k].negated = !terms[t][k].negated;
                return terms;
            }
            3 => {
                // add term
                if terms.len() >= opts.max_terms {
                    continue;
                }
                let idx = rng.gen_range(p);
                terms.push(vec![Literal {
                    idx,
                    negated: rng.gen_bool(),
                }]);
                return terms;
            }
            _ => {
                // remove term
                if terms.len() <= 1 {
                    continue;
                }
                let t = rng.gen_range(terms.len());
                terms.remove(t);
                return terms;
            }
        }
    }

    terms
}

/// Evaluate DNF terms on all samples.
fn eval_terms(x: &[Vec<u8>], terms: &[Vec<Literal>]) -> Vec<u8> {
    let n = x.len();
    let mut out = vec![0u8; n];
    for i in 0..n {
        let mut row_val = 0u8;
        for term in terms {
            let mut term_val = 1u8;
            for lit in term {
                let v = literal_value(x[i][lit.idx], lit.negated);
                term_val &= v;
                if term_val == 0 {
                    break;
                }
            }
            if term_val == 1 {
                row_val = 1;
                break;
            }
        }
        out[i] = row_val;
    }
    out
}

fn initial_best_literal(
    x: &[Vec<u8>],
    y: &[f64],
    weights: Option<&[f64]>,
    response: ResponseKind,
    score_kind: ScoreKind,
) -> Result<(Literal, Vec<u8>, f64), String> {
    let n = x.len();
    let p = x[0].len();
    let mut best_score = f64::INFINITY;
    let mut best_lit: Option<Literal> = None;
    let mut best_pred = vec![0u8; n];

    for j in 0..p {
        for &neg in &[false, true] {
            let mut pred = vec![0u8; n];
            for i in 0..n {
                pred[i] = literal_value(x[i][j], neg);
            }
            let score = score_model(y, &pred, weights, response, score_kind);
            if score < best_score {
                best_score = score;
                best_lit = Some(Literal { idx: j, negated: neg });
                best_pred = pred;
            }
        }
    }

    let lit = best_lit.ok_or_else(|| "failed to select initial literal".to_string())?;
    Ok((lit, best_pred, best_score))
}

/// Greedy forward selection for a single conjunction (AND-only model).
fn fit_best_conjunction(
    x: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    score_kind: ScoreKind,
    max_terms: usize,
    min_improvement: f64,
    weights: Option<&[f64]>,
) -> Result<FitResult, String> {
    validate_x_y(x, y, response, weights)?;
    let n = x.len();
    let p = x[0].len();

    let mut used = vec![false; p];
    let mut literals: Vec<Literal> = Vec::new();

    let (first, mut current_pred, mut current_score) =
        initial_best_literal(x, y, weights, response, score_kind)?;

    literals.push(first);
    used[first.idx] = true;

    if max_terms > 1 {
        for _ in 2..=max_terms {
            let mut step_best_score = current_score;
            let mut step_best_lit: Option<Literal> = None;
            let mut step_best_pred = current_pred.clone();

            for j in 0..p {
                if used[j] {
                    continue;
                }
                for &neg in &[false, true] {
                    let mut pred = vec![0u8; n];
                    for i in 0..n {
                        let v = literal_value(x[i][j], neg);
                        pred[i] = current_pred[i] & v;
                    }
                    let score = score_model(y, &pred, weights, response, score_kind);
                    if score < step_best_score {
                        step_best_score = score;
                        step_best_lit = Some(Literal { idx: j, negated: neg });
                        step_best_pred = pred;
                    }
                }
            }

            let improvement = current_score - step_best_score;
            if improvement <= min_improvement {
                break;
            }
            let lit = step_best_lit.ok_or_else(|| "failed to select next literal".to_string())?;
            literals.push(lit);
            used[lit.idx] = true;
            current_pred = step_best_pred;
            current_score = step_best_score;
        }
    }

    let terms = vec![literals];
    let (indices, indices_1based) = collect_indices(&terms);
    let expression = build_expression_from_terms(&terms);

    Ok(FitResult {
        terms,
        indices,
        indices_1based,
        expression,
        xcombine: current_pred,
        score: current_score,
    })
}

/// Simulated annealing over DNF models.
fn fit_dnf_anneal(
    x: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    score_kind: ScoreKind,
    weights: Option<&[f64]>,
    opts: AnnealOptions,
) -> Result<FitResult, String> {
    validate_x_y(x, y, response, weights)?;
    if opts.max_terms == 0 || opts.max_literals_per_term == 0 {
        return Err("max_terms and max_literals_per_term must be > 0".to_string());
    }
    if opts.iters == 0 {
        return Err("iters must be > 0".to_string());
    }
    if opts.start_temp <= 0.0 || opts.end_temp <= 0.0 {
        return Err("start_temp and end_temp must be > 0".to_string());
    }
    if opts.restarts == 0 {
        return Err("restarts must be > 0".to_string());
    }

    let n = x.len();
    let p = x[0].len();

    // Track the best solution across all restarts.
    let mut best_terms = Vec::new();
    let mut best_pred_overall = vec![0u8; n];
    let mut best_score_overall = f64::INFINITY;

    for r in 0..opts.restarts {
        // Start each restart from the single best literal (greedy seed).
        let seed = opts
            .seed
            .wrapping_add((r as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let mut rng = SimpleRng::new(seed);

        let (init_lit, best_pred, best_score) =
            initial_best_literal(x, y, weights, response, score_kind)?;
        let mut current_terms = vec![vec![init_lit]];
        let mut current_score = best_score;

        let mut local_best_terms = current_terms.clone();
        let mut local_best_pred = best_pred.clone();
        let mut local_best_score = best_score;

        for t in 0..opts.iters {
            let temp = temperature(opts.start_temp, opts.end_temp, t, opts.iters);
            // Propose a neighboring DNF and accept via annealing criterion.
            let candidate_terms = propose_terms(&current_terms, p, &opts, &mut rng);
            let cand_pred = eval_terms(x, &candidate_terms);
            let cand_score = score_model(y, &cand_pred, weights, response, score_kind);
            let delta = cand_score - current_score;
            let accept = if delta <= 0.0 {
                true
            } else {
                rng.next_f64() < (-delta / temp).exp()
            };
            if accept {
                current_terms = candidate_terms.clone();
                current_score = cand_score;
            }
            if cand_score < local_best_score {
                local_best_score = cand_score;
                local_best_terms = candidate_terms;
                local_best_pred = cand_pred;
            }
        }

        if local_best_score < best_score_overall {
            best_score_overall = local_best_score;
            best_terms = local_best_terms;
            best_pred_overall = local_best_pred;
        }
    }

    let (indices, indices_1based) = collect_indices(&best_terms);
    let expression = build_expression_from_terms(&best_terms);

    Ok(FitResult {
        terms: best_terms,
        indices,
        indices_1based,
        expression,
        xcombine: best_pred_overall,
        score: best_score_overall,
    })
}

fn sample_indices(
    n: usize,
    replace: bool,
    sub_frac: f64,
    rng: &mut SimpleRng,
    weights: Option<&[f64]>,
) -> Vec<usize> {
    let m = if replace { n } else { ((n as f64) * sub_frac).ceil() as usize };
    if replace {
        if let Some(w) = weights {
            let mut prefix = Vec::with_capacity(n);
            let mut sum = 0.0;
            for &v in w {
                sum += v.max(0.0);
                prefix.push(sum);
            }
            let total = sum.max(1e-12);
            let mut out = Vec::with_capacity(m);
            for _ in 0..m {
                let r = rng.next_f64() * total;
                let idx = match prefix.binary_search_by(|v| v.partial_cmp(&r).unwrap()) {
                    Ok(i) => i,
                    Err(i) => i,
                };
                out.push(idx.min(n - 1));
            }
            out
        } else {
            let mut out = Vec::with_capacity(m);
            for _ in 0..m {
                out.push(rng.gen_range(n));
            }
            out
        }
    } else {
        let mut out = Vec::with_capacity(m);
        let mut pool: Vec<(usize, f64)> = (0..n)
            .map(|i| (i, weights.map(|w| w[i]).unwrap_or(1.0)))
            .collect();
        for _ in 0..m.min(n) {
            let total: f64 = pool.iter().map(|(_, w)| w.max(0.0)).sum();
            let mut r = rng.next_f64() * total.max(1e-12);
            let mut chosen = None;
            for (pos, (_, w)) in pool.iter().enumerate() {
                let wv = w.max(0.0);
                if r <= wv {
                    chosen = Some(pos);
                    break;
                }
                r -= wv;
            }
            let pos = chosen.unwrap_or(0);
            let (idx, _) = pool.remove(pos);
            out.push(idx);
        }
        out
    }
}

fn subset_rows<T: Copy>(data: &[T], idx: &[usize]) -> Vec<T> {
    idx.iter().map(|&i| data[i]).collect()
}

fn subset_matrix(data: &[Vec<u8>], idx: &[usize]) -> Vec<Vec<u8>> {
    idx.iter().map(|&i| data[i].clone()).collect()
}

/// Bagging ensemble of DNF annealing models.
fn fit_bagging_dnf(
    x: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    score_kind: ScoreKind,
    weights: Option<&[f64]>,
    anneal_opts: AnnealOptions,
    bag_opts: BaggingOptions,
) -> Result<BaggingResult, String> {
    validate_x_y(x, y, response, weights)?;
    if bag_opts.n_models == 0 {
        return Err("n_models must be > 0".to_string());
    }
    if !bag_opts.replace && (bag_opts.sub_frac <= 0.0 || bag_opts.sub_frac > 1.0) {
        return Err("sub_frac must be in (0, 1] when replace=false".to_string());
    }

    let n = x.len();
    let mut rng = SimpleRng::new(bag_opts.seed);
    let mut models = Vec::with_capacity(bag_opts.n_models);

    for b in 0..bag_opts.n_models {
        let idx = sample_indices(n, bag_opts.replace, bag_opts.sub_frac, &mut rng, weights);
        let x_sub = subset_matrix(x, &idx);
        let y_sub = subset_rows(y, &idx);
        let w_sub = weights.map(|w| subset_rows(w, &idx));

        let mut opts = anneal_opts.clone();
        opts.seed = opts
            .seed
            .wrapping_add((b as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        let model = fit_dnf_anneal(
            &x_sub,
            &y_sub,
            response,
            score_kind,
            w_sub.as_deref(),
            opts,
        )?;
        models.push(model);
    }

    let mut combined = vec![0.0f64; n];
    for model in &models {
        let pred = eval_terms(x, &model.terms);
        for i in 0..n {
            combined[i] += pred[i] as f64;
        }
    }
    for v in &mut combined {
        *v /= bag_opts.n_models as f64;
    }
    let combined_bin: Vec<u8> = combined.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();

    Ok(BaggingResult {
        models,
        combined_pred: combined,
        combined_pred_binary: combined_bin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_simple_and_binary() {
        // y = X1 & !X2
        let x = vec![
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
        ];
        let y = vec![0.0, 0.0, 1.0, 0.0];
        let res = fit_best_conjunction(
            &x,
            &y,
            ResponseKind::Binary,
            ScoreKind::LogLikelihood,
            2,
            1e-12,
            None,
        )
        .unwrap();
        assert_eq!(res.expression, "X1 & !X2");
        assert_eq!(res.xcombine, vec![0, 0, 1, 0]);
    }

    #[test]
    fn predict_or_expression() {
        let terms = vec![
            vec![
                Literal {
                    idx: 0,
                    negated: false,
                },
                Literal {
                    idx: 1,
                    negated: true,
                },
            ],
            vec![Literal {
                idx: 2,
                negated: false,
            }],
        ];
        let expr = build_expression_from_terms(&terms);
        assert_eq!(expr, "(X1 & !X2) | X3");

        let x = vec![
            vec![0, 0, 0],
            vec![1, 0, 0],
            vec![1, 1, 0],
            vec![0, 1, 1],
        ];
        let pred = predict_from_terms(&x, &terms).unwrap();
        assert_eq!(pred, vec![0, 1, 0, 1]);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectMode {
    Anneal,
    Greedy,
    Fast,
}

#[derive(Debug, Clone, Copy)]
pub struct AnnealControl {
    pub start: f64,
    pub end: f64,
    pub iter: usize,
    pub earlyout: usize,
    pub update: usize,
}

impl Default for AnnealControl {
    fn default() -> Self {
        Self {
            start: 0.0,
            end: 0.0,
            iter: 0,
            earlyout: 0,
            update: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogRegOptions {
    pub response: ResponseKind,
    pub score: ScoreKind,
    pub ntrees: usize,
    pub nleaves: usize,
    pub treesize: usize,
    pub select: SelectMode,
    pub anneal: AnnealControl,
    pub minmass: usize,
    pub penalty: f64,
    pub seed: u64,
    pub allow_or: bool,
}

impl Default for LogRegOptions {
    fn default() -> Self {
        Self {
            response: ResponseKind::Binary,
            score: ScoreKind::LogLikelihood,
            ntrees: 1,
            nleaves: 8,
            treesize: 8,
            select: SelectMode::Anneal,
            anneal: AnnealControl::default(),
            minmass: 0,
            penalty: 0.0,
            seed: 42,
            allow_or: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeNode {
    pub conc: u8,     // 0 empty, 1 AND, 2 OR, 3 LEAF
    pub var: usize,   // 0-based predictor index
    pub neg: bool,    // NOT
    pub pick: bool,   // active flag
}

#[derive(Debug, Clone)]
pub struct LogicTree {
    pub nodes: Vec<TreeNode>,
    pub nvars: usize,
}

#[derive(Debug, Clone)]
pub struct LogicRegModel {
    pub trees: Vec<LogicTree>,
    pub expression: String,
    pub score: f64,
    pub xcombine: Vec<u8>,
    pub betas: Vec<f64>,
    pub trace: Vec<TracePoint>,
}

#[derive(Debug, Clone, Copy)]
pub struct TracePoint {
    pub iter: usize,
    pub temp: f64,
    pub score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveKind {
    AltLeaf,
    AltOp,
    DeleteLeaf,
    SplitLeaf,
    Branch,
    Prune,
    NewTree,
}

fn pow2_floor(v: usize) -> usize {
    if v == 0 { return 1; }
    let mut p = 1usize;
    while p * 2 <= v { p *= 2; }
    p
}

fn full_tree_size(treesize: usize) -> usize {
    let ts = pow2_floor(treesize);
    2 * ts - 1
}

fn empty_tree(nvars: usize, treesize: usize) -> LogicTree {
    let nkn = full_tree_size(treesize);
    LogicTree {
        nodes: vec![TreeNode { conc: 0, var: 0, neg: false, pick: false }; nkn],
        nvars,
    }
}

fn eval_tree(tree: &LogicTree, x: &[Vec<u8>]) -> Vec<u8> {
    let n = x.len();
    let nkn = tree.nodes.len();
    if n == 0 || nkn == 0 {
        return Vec::new();
    }
    let all_zero = tree.nodes.iter().all(|n| n.conc == 0);
    let mut out = vec![0u8; n];
    for row in 0..n {
        // Empty nodes evaluate to 0 to match tree_expression("0").
        let mut vals = vec![0u8; nkn];
        for i in 0..nkn {
            if tree.nodes[i].conc == 3 {
                let v = x[row][tree.nodes[i].var];
                vals[i] = if tree.nodes[i].neg { 1 - v } else { v };
            }
        }
        for i in (0..(nkn / 2)).rev() {
            let conc = tree.nodes[i].conc;
            if conc == 1 || conc == 2 {
                let l = 2 * i + 1;
                let r = 2 * i + 2;
                vals[i] = if conc == 1 { vals[l] & vals[r] } else { vals[l] | vals[r] };
            }
        }
        out[row] = if all_zero { 0 } else { vals[0].min(1) };
    }
    out
}

fn eval_forest(trees: &[LogicTree], x: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for t in trees {
        if t.nodes.first().map(|n| n.conc != 0).unwrap_or(false) {
            out.push(eval_tree(t, x));
        }
    }
    out
}

fn tree_expression(tree: &LogicTree, idx: usize) -> String {
    if idx >= tree.nodes.len() {
        return "0".to_string();
    }
    let node = tree.nodes[idx];
    match node.conc {
        0 => "0".to_string(),
        3 => {
            let name = format!("X{}", node.var + 1);
            if node.neg { format!("!{}", name) } else { name }
        }
        1 | 2 => {
            let l = tree_expression(tree, 2 * idx + 1);
            let r = tree_expression(tree, 2 * idx + 2);
            let op = if node.conc == 1 { " & " } else { " | " };
            format!("({}{}{})", l, op, r)
        }
        _ => "0".to_string(),
    }
}

fn forest_expression(trees: &[LogicTree]) -> String {
    let mut parts = Vec::new();
    for t in trees {
        if t.nodes.iter().any(|n| n.conc != 0) {
            parts.push(tree_expression(t, 0));
        }
    }
    parts.join(" + ")
}

fn count_leaves(tree: &LogicTree) -> usize {
    tree.nodes.iter().filter(|n| n.conc == 3).count()
}

fn count_leaves_forest(trees: &[LogicTree]) -> usize {
    trees.iter().map(count_leaves).sum()
}

fn first_knot(tree: &mut LogicTree, rng: &mut SimpleRng) {
    let letter = rng.gen_range(tree.nvars);
    let neg = rng.gen_bool();
    tree.nodes[0] = TreeNode { conc: 3, var: letter, neg, pick: true };
}

fn is_leaf(node: &TreeNode) -> bool { node.conc == 3 }

fn storing(trees: &[LogicTree]) -> (usize, usize, Vec<[usize; 6]>, Vec<[Vec<isize>; 6]>) {
    let ntr = trees.len();
    let mut ssize = 0usize;
    let mut nop = 0usize;
    let mut npckmv = vec![[0usize; 6]; ntr];
    let mut pickmv: Vec<[Vec<isize>; 6]> = vec![[
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
    ]; ntr];

    for (j, t) in trees.iter().enumerate() {
        let nkn = t.nodes.len();
        for i in 0..nkn {
            if !t.nodes[i].pick { continue; }
            nop = j + 1;
            if t.nodes[i].conc == 3 {
                ssize += 1;
                npckmv[j][0] += 1; // alt leaf
                pickmv[j][0].push(i as isize);

                // delete leaf
                if i == 0 {
                    npckmv[j][2] += 1;
                    pickmv[j][2].push(i as isize);
                } else {
                    let sib = if i % 2 == 0 { i - 1 } else { i + 1 };
                    if sib < nkn && t.nodes[sib].conc == 3 {
                        npckmv[j][2] += 1;
                        pickmv[j][2].push(i as isize);
                    }
                }
                // split leaf
                if 2 * i + 2 < nkn {
                    npckmv[j][3] += 1;
                    pickmv[j][3].push(i as isize);
                }
            } else if t.nodes[i].conc == 1 || t.nodes[i].conc == 2 {
                npckmv[j][1] += 1; // alt op
                pickmv[j][1].push(i as isize);

                if 4 * i + 3 < nkn {
                    let left = 2 * i + 1;
                    let right = 2 * i + 2;
                    if is_leaf(&t.nodes[left]) && is_leaf(&t.nodes[right]) {
                        npckmv[j][4] += 1; // branch
                        pickmv[j][4].push(i as isize);
                    }
                    // prune
                    let left_leaf = is_leaf(&t.nodes[left]);
                    let right_leaf = is_leaf(&t.nodes[right]);
                    if left_leaf {
                        let r1 = 4 * i + 5;
                        let r2 = 4 * i + 6;
                        if r2 < nkn && is_leaf(&t.nodes[r1]) && is_leaf(&t.nodes[r2]) {
                            npckmv[j][5] += 1;
                            pickmv[j][5].push(i as isize);
                        }
                    } else if right_leaf {
                        let l1 = 4 * i + 3;
                        let l2 = 4 * i + 4;
                        if l2 < nkn && is_leaf(&t.nodes[l1]) && is_leaf(&t.nodes[l2]) {
                            npckmv[j][5] += 1;
                            pickmv[j][5].push(-(i as isize));
                        }
                    }
                }
            }
        }
    }

    (ssize, nop, npckmv, pickmv)
}

/// Apply one structural move to a logic tree (used by search).
fn apply_move(tree: &mut LogicTree, kind: MoveKind, node_idx: isize, rng: &mut SimpleRng, allow_or: bool) -> bool {
    let nkn = tree.nodes.len();
    let node = node_idx.unsigned_abs() as usize;
    match kind {
        MoveKind::NewTree => {
            first_knot(tree, rng);
            true
        }
        MoveKind::AltLeaf => {
            if tree.nodes[node].conc != 3 { return false; }
            let letter = rng.gen_range(tree.nvars);
            let neg = rng.gen_bool();
            if node > 0 {
                let sib = if node % 2 == 0 { node - 1 } else { node + 1 };
                if sib < nkn && tree.nodes[sib].conc == 3 && tree.nodes[sib].var == letter {
                    return false;
                }
            }
            tree.nodes[node].var = letter;
            tree.nodes[node].neg = neg;
            true
        }
        MoveKind::AltOp => {
            let conc = tree.nodes[node].conc;
            if conc == 1 && allow_or { tree.nodes[node].conc = 2; true }
            else if conc == 2 { tree.nodes[node].conc = 1; true }
            else { false }
        }
        MoveKind::DeleteLeaf => {
            if node == 0 {
                tree.nodes[0] = TreeNode { conc: 0, var: 0, neg: false, pick: false };
                return true;
            }
            let parent = (node - 1) / 2;
            let sib = if node % 2 == 0 { node - 1 } else { node + 1 };
            if sib >= nkn { return false; }
            let s = tree.nodes[sib];
            tree.nodes[parent] = TreeNode { conc: s.conc, var: s.var, neg: s.neg, pick: s.pick };
            tree.nodes[node] = TreeNode { conc: 0, var: 0, neg: false, pick: false };
            tree.nodes[sib] = TreeNode { conc: 0, var: 0, neg: false, pick: false };
            true
        }
        MoveKind::SplitLeaf => {
            if tree.nodes[node].conc != 3 { return false; }
            let left = 2 * node + 1;
            let right = 2 * node + 2;
            if right >= nkn { return false; }
            if tree.nvars <= 1 { return false; }
            let old = tree.nodes[node];
            let op = if allow_or && rng.gen_bool() { 2 } else { 1 };
            let mut letter = rng.gen_range(tree.nvars);
            let mut tries = 0usize;
            while letter == old.var && tries < tree.nvars * 2 {
                letter = rng.gen_range(tree.nvars);
                tries += 1;
            }
            if letter == old.var { return false; }
            let neg = rng.gen_bool();
            tree.nodes[left] = TreeNode { conc: 3, var: old.var, neg: old.neg, pick: true };
            tree.nodes[node].conc = op;
            tree.nodes[node].var = 0;
            tree.nodes[node].neg = false;
            tree.nodes[node].pick = true;
            tree.nodes[right] = TreeNode { conc: 3, var: letter, neg, pick: true };
            true
        }
        MoveKind::Branch => {
            let left = 2 * node + 1;
            let right = 2 * node + 2;
            let ll = 4 * node + 3;
            let lr = 4 * node + 4;
            let rl = 4 * node + 5;
            if rl >= nkn { return false; }
            tree.nodes[ll] = TreeNode { conc: 3, var: tree.nodes[left].var, neg: tree.nodes[left].neg, pick: true };
            tree.nodes[lr] = TreeNode { conc: 3, var: tree.nodes[right].var, neg: tree.nodes[right].neg, pick: true };
            tree.nodes[left] = TreeNode { conc: tree.nodes[node].conc, var: 0, neg: false, pick: true };
            tree.nodes[right] = TreeNode { conc: 3, var: rng.gen_range(tree.nvars), neg: rng.gen_bool(), pick: true };
            tree.nodes[node].conc = if allow_or && rng.gen_bool() { 2 } else { 1 };
            tree.nodes[node].pick = true;
            true
        }
        MoveKind::Prune => {
            let left = 2 * node + 1;
            let right = 2 * node + 2;
            if left >= nkn || right >= nkn { return false; }
            let (dbl, sng) = if node_idx >= 0 { (right, left) } else { (left, right) };
            let dbl_left = 2 * dbl + 1;
            let dbl_right = 2 * dbl + 2;
            if dbl_right >= nkn { return false; }

            tree.nodes[node].conc = tree.nodes[dbl].conc;
            tree.nodes[node].var = 0;
            tree.nodes[node].neg = false;
            tree.nodes[node].pick = true;

            tree.nodes[sng] = tree.nodes[dbl_left];
            tree.nodes[dbl_left] = TreeNode { conc: 0, var: 0, neg: false, pick: false };

            tree.nodes[dbl] = tree.nodes[dbl_right];
            tree.nodes[dbl_right] = TreeNode { conc: 0, var: 0, neg: false, pick: false };
            true
        }
    }
}

fn base_move_weights(allow_or: bool) -> [f64; 6] {
    let mut w = [10.0, 1.0, 3.0, 3.0, 3.0, 3.0];
    if !allow_or { w[1] = 0.0; }
    w
}

fn pick_move(npckmv: &[usize; 6], pickmv: &[Vec<isize>; 6], weights: [f64; 6], rng: &mut SimpleRng) -> Option<(MoveKind, isize)> {
    let mut total = 0.0;
    let mut w = weights;
    for i in 0..6 {
        if npckmv[i] == 0 { w[i] = 0.0; }
        total += w[i];
    }
    if total <= 0.0 { return None; }
    let mut r = rng.next_f64() * total;
    let mut idx = 0usize;
    for i in 0..6 {
        if r <= w[i] { idx = i; break; }
        r -= w[i];
    }
    let node_list = &pickmv[idx];
    if node_list.is_empty() { return None; }
    let node = node_list[rng.gen_range(node_list.len())];
    let kind = match idx {
        0 => MoveKind::AltLeaf,
        1 => MoveKind::AltOp,
        2 => MoveKind::DeleteLeaf,
        3 => MoveKind::SplitLeaf,
        4 => MoveKind::Branch,
        _ => MoveKind::Prune,
    };
    Some((kind, node))
}

fn design_matrix(
    prtr: &[Vec<u8>],
    seps: Option<&[Vec<f64>]>,
    nop: usize,
    n: usize,
) -> Vec<Vec<f64>> {
    let nsep = seps.map(|s| s.len()).unwrap_or(0);
    let mut x = vec![vec![0.0; 1 + nsep + nop]; n];
    for i in 0..n {
        x[i][0] = 1.0;
        if let Some(sep) = seps {
            for j in 0..nsep {
                x[i][1 + j] = sep[j][i];
            }
        }
        for j in 0..nop {
            x[i][1 + nsep + j] = prtr[j][i] as f64;
        }
    }
    x
}

fn weighted_ols(x: &[Vec<f64>], y: &[f64], w: &[f64]) -> Option<Vec<f64>> {
    let n = x.len();
    if n == 0 { return None; }
    let p = x[0].len();
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];

    for i in 0..n {
        let wi = w[i];
        for a in 0..p {
            xty[a] += wi * x[i][a] * y[i];
            for b in 0..p {
                xtx[a][b] += wi * x[i][a] * x[i][b];
            }
        }
    }

    let mut aug = vec![vec![0.0; p + 1]; p];
    for i in 0..p {
        for j in 0..p { aug[i][j] = xtx[i][j]; }
        aug[i][p] = xty[i];
    }
    for i in 0..p {
        let mut pivot = i;
        for r in i+1..p {
            if aug[r][i].abs() > aug[pivot][i].abs() { pivot = r; }
        }
        if aug[pivot][i].abs() < 1e-12 { return None; }
        aug.swap(i, pivot);
        let div = aug[i][i];
        for j in i..=p { aug[i][j] /= div; }
        for r in 0..p {
            if r == i { continue; }
            let factor = aug[r][i];
            for c in i..=p { aug[r][c] -= factor * aug[i][c]; }
        }
    }
    let mut beta = vec![0.0; p];
    for i in 0..p { beta[i] = aug[i][p]; }
    Some(beta)
}

fn seps_are_binary(seps: &[Vec<f64>]) -> bool {
    for col in seps {
        for &v in col {
            if v > 1.000001 || v < -0.000001 {
                return false;
            }
            if v > 0.000001 && v < 0.999999 {
                return false;
            }
        }
    }
    true
}

fn redater(
    prtr: &[Vec<u8>],
    seps: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
    nop: usize,
) -> Option<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
    let nsep = seps.len();
    let n = y.len();
    if nsep + nop >= 10 {
        return None;
    }
    if !seps_are_binary(seps) {
        return None;
    }
    let k = 1usize << (nsep + nop);
    let mut x = vec![vec![0.0; 1 + nsep + nop]; k];
    for i in 0..k {
        x[i][0] = 1.0;
        for bit in 0..(nsep + nop) {
            x[i][1 + bit] = ((i >> bit) & 1) as f64;
        }
    }
    let mut y_bin = vec![0.0; k];
    let mut n_bin = vec![0.0; k];
    for i in 0..n {
        let mut idx = 0usize;
        let mut mult = 1usize;
        for j in 0..nsep {
            let v = if seps[j][i] > 0.5 { 1usize } else { 0usize };
            idx += mult * v;
            mult <<= 1;
        }
        for j in 0..nop {
            idx += mult * (prtr[j][i] as usize);
            mult <<= 1;
        }
        y_bin[idx] += y[i] * weights[i];
        n_bin[idx] += weights[i];
    }
    let mut x2 = Vec::new();
    let mut y2 = Vec::new();
    let mut n2 = Vec::new();
    for i in 0..k {
        if n_bin[i] > 0.0 {
            x2.push(x[i].clone());
            y2.push(y_bin[i]);
            n2.push(n_bin[i]);
        }
    }
    Some((x2, y2, n2))
}

fn calcbetarss(
    y: &[f64],
    prtr: &[Vec<u8>],
    seps: Option<&[Vec<f64>]>,
    weights: &[f64],
    nop: usize,
) -> Option<Vec<f64>> {
    let x = design_matrix(prtr, seps, nop, y.len());
    weighted_ols(&x, y, weights)
}

fn calcrss(
    y: &[f64],
    prtr: &[Vec<u8>],
    seps: Option<&[Vec<f64>]>,
    weights: &[f64],
    beta: &[f64],
    nop: usize,
) -> Option<f64> {
    let n1 = y.len();
    let nsep = seps.map(|s| s.len()).unwrap_or(0);
    if n1 <= 1 + nsep + nop {
        return None;
    }
    let mut sse = 0.0;
    for i in 0..n1 {
        let mut pred = beta[0];
        if let Some(sep) = seps {
            for j in 0..nsep {
                pred += beta[j + 1] * sep[j][i];
            }
        }
        for j in 0..nop {
            pred += beta[nsep + j + 1] * (prtr[j][i] as f64);
        }
        let diff = pred - y[i];
        sse += weights[i] * diff * diff;
    }
    let denom = (n1 - 1 - nsep - nop) as f64;
    Some((sse / denom).sqrt())
}

/// Logistic deviance via IRLS on aggregated (optionally binary) data.
fn calcdev(
    y: &[f64],
    prtr: &[Vec<u8>],
    seps: Option<&[Vec<f64>]>,
    weights: &[f64],
    nop: usize,
) -> Option<(f64, Vec<f64>)> {
    let nsep = seps.map(|s| s.len()).unwrap_or(0);
    let x: Vec<Vec<f64>>;
    let y_bin: Vec<f64>;
    let n_bin: Vec<f64>;

    let sep_ref: &[Vec<f64>] = seps.unwrap_or(&[]);
    if let Some((x2, y2, n2)) = redater(prtr, sep_ref, y, weights, nop) {
        x = x2;
        y_bin = y2;
        n_bin = n2;
    } else {
        x = design_matrix(prtr, seps, nop, y.len());
        y_bin = y.iter().zip(weights.iter()).map(|(yi, wi)| yi * wi).collect();
        n_bin = weights.to_vec();
    }

    let n1tmp = y_bin.len();
    let mut beta = vec![0.0; 1 + nsep + nop];
    let mut loglik_old = -100000.0;
    let eps = 1e-6;

    for _ in 0..20 {
        let mut eta = vec![0.0; n1tmp];
        let mut mu = vec![0.0; n1tmp];
        for i in 0..n1tmp {
            let mut v = 0.0;
            for j in 0..beta.len() {
                v += x[i][j] * beta[j];
            }
            eta[i] = v;
            mu[i] = 1.0 / (1.0 + (-v).exp());
        }
        let mut w = vec![0.0; n1tmp];
        let mut z = vec![0.0; n1tmp];
        for i in 0..n1tmp {
            let var = (mu[i] * (1.0 - mu[i])).max(1e-9);
            let denom = n_bin[i] * var;
            if denom <= 0.0 {
                w[i] = 0.0;
                z[i] = eta[i];
            } else {
                w[i] = denom;
                z[i] = eta[i] + (y_bin[i] - n_bin[i] * mu[i]) / denom;
            }
        }
        let new_beta = weighted_ols(&x, &z, &w)?;
        let mut conv = true;
        for j in 0..beta.len() {
            if (new_beta[j] - beta[j]).abs() > eps * beta[j].abs().max(1.0) {
                conv = false;
                break;
            }
        }
        let mut loglik = 0.0;
        for i in 0..n1tmp {
            let p = mu[i].clamp(1e-12, 1.0 - 1e-12);
            loglik += y_bin[i] * p.ln() + (n_bin[i] - y_bin[i]) * (1.0 - p).ln();
        }
        if (loglik_old - loglik).abs() > eps {
            conv = false;
        }
        beta = new_beta;
        loglik_old = loglik;
        if conv {
            break;
        }
    }

    let mut dev = 0.0;
    for i in 0..n1tmp {
        let mut v = 0.0;
        for j in 0..beta.len() {
            v += x[i][j] * beta[j];
        }
        let p = 1.0 / (1.0 + (-v).exp());
        if p <= 0.0 || p >= 1.0 {
            return None;
        }
        dev -= 2.0 * (y_bin[i] * p.ln() + (n_bin[i] - y_bin[i]) * (1.0 - p).ln());
    }
    Some((dev, beta))
}

fn singularities(
    prtr: &[Vec<u8>],
    seps: Option<&[Vec<f64>]>,
    mtm: usize,
) -> bool {
    if prtr.is_empty() {
        return true;
    }
    let n1 = prtr[0].len();
    let mut n4 = ((0.05 * n1 as f64) as usize).min(15);
    if mtm > 0 {
        n4 = mtm;
    }
    for j in 0..prtr.len() {
        let sum1 = prtr[j].iter().filter(|&&v| v == 1).count();
        if sum1 < n4 || sum1 > n1 - n4 {
            return false;
        }
    }
    for j in 0..prtr.len() {
        for k in 0..prtr.len() {
            if j == k { continue; }
            let mut same = true;
            let mut compl = true;
            for i in 0..n1 {
                if prtr[j][i] != prtr[k][i] { same = false; }
                if prtr[j][i] != (1 - prtr[k][i]) { compl = false; }
                if !same && !compl { break; }
            }
            if same || compl { return false; }
        }
    }
    if let Some(sep) = seps {
        if seps_are_binary(sep) {
            for s in sep {
                for j in 0..prtr.len() {
                    let mut same = true;
                    let mut compl = true;
                    for i in 0..n1 {
                        let sv = if s[i] > 0.5 { 1u8 } else { 0u8 };
                        if prtr[j][i] != sv { same = false; }
                        if prtr[j][i] != (1 - sv) { compl = false; }
                        if !same && !compl { break; }
                    }
                    if same || compl { return false; }
                }
            }
        }
    }
    true
}

/// Score LogicReg-like forest with optional covariates and penalties.
fn score_model_logicreg(
    y: &[f64],
    prtr: &[Vec<u8>],
    seps: Option<&[Vec<f64>]>,
    response: ResponseKind,
    score: ScoreKind,
    weights: Option<&[f64]>,
    penalty: f64,
    nleaves: usize,
    mtm: usize,
    nop: usize,
) -> Option<(f64, Vec<f64>)> {
    let n = y.len();
    let w_owned: Vec<f64>;
    let w = match weights {
        Some(w) => w,
        None => {
            w_owned = vec![1.0; n];
            &w_owned
        }
    };

    if response == ResponseKind::Binary {
        if !singularities(prtr, seps, mtm) { return None; }
        let (dev, beta) = calcdev(y, prtr, seps, w, nop)?;
        let mut score_val = if score == ScoreKind::LogLikelihood { dev } else { dev };
        score_val += penalty * (nleaves as f64);
        return Some((score_val, beta));
    }

    let beta = calcbetarss(y, prtr, seps, w, nop)?;
    let rss = calcrss(y, prtr, seps, w, &beta, nop)?;
    let mut score_val = if score == ScoreKind::LogLikelihood { rss } else { rss };
    score_val += penalty / (n as f64) * (nleaves as f64);
    Some((score_val, beta))
}

fn maybe_trace(trace: &mut Vec<TracePoint>, opts: &LogRegOptions, iter: usize, temp: f64, score: f64) {
    if opts.anneal.update > 0 && iter % opts.anneal.update == 0 {
        trace.push(TracePoint { iter, temp, score });
    }
}

fn temperature_default_search(
    trees: &mut Vec<LogicTree>,
    x: &[Vec<u8>],
    y: &[f64],
    seps: Option<&[Vec<f64>]>,
    weights: Option<&[f64]>,
    opts: &LogRegOptions,
    current_score: &mut f64,
    best_score: &mut f64,
    best_trees: &mut Vec<LogicTree>,
    best_pred: &mut Vec<u8>,
    best_beta: &mut Vec<f64>,
    trace: &mut Vec<TracePoint>,
    rng: &mut SimpleRng,
) {
    let mut tstr = 3.0;
    let hm = 100;
    let mut tcnt = 0.0;
    let mut temp = 10f64.powf(tstr);
    let mut acr = 1.0;

    while acr > 0.75 {
        temp = 10f64.powf(tstr) * 10f64.powf(-tcnt / 5.0);
        let mut nac = 0;
        let mut nrj = 0;
        for j in 0..hm {
            let old_score = *current_score;
            let acc = annealing_step(
                trees, x, y, seps, weights, opts, temp, current_score, best_score,
                best_trees, best_pred, best_beta, trace, rng, j,
            );
            if acc < 0 { *current_score = old_score; nrj += 1; }
            if acc > 0 { nac += 1; }
        }
        let denom = (hm - nrj).max(1) as f64;
        acr = nac as f64 / denom;
        tcnt += 1.0;
    }

    tstr = temp.log10().floor() + 2.0;
    let npertemp = (50 * x[0].len()).min(5000);
    let mut fcnt = 0;
    let mut tcnt2 = 1.0;
    while fcnt < 5 {
        let temp2 = 10f64.powf(tstr) * 10f64.powf(-tcnt2 / 20.0);
        let mut nac = 0;
        let mut nsame = 0;
        for j in 0..npertemp {
            let old_score = *current_score;
            let acc = annealing_step(
                trees, x, y, seps, weights, opts, temp2, current_score, best_score,
                best_trees, best_pred, best_beta, trace, rng, j,
            );
            if acc > 0 {
                nac += 1;
                if (old_score - *current_score).abs() < 1e-12 { nsame += 1; }
            }
        }
        if (nac - nsame) <= 10 { fcnt += 1; } else { fcnt = 0; }
        tcnt2 += 1.0;
    }
}

fn annealing_step(
    trees: &mut Vec<LogicTree>,
    x: &[Vec<u8>],
    y: &[f64],
    seps: Option<&[Vec<f64>]>,
    weights: Option<&[f64]>,
    opts: &LogRegOptions,
    temp: f64,
    current_score: &mut f64,
    best_score: &mut f64,
    best_trees: &mut Vec<LogicTree>,
    best_pred: &mut Vec<u8>,
    best_beta: &mut Vec<f64>,
    trace: &mut Vec<TracePoint>,
    rng: &mut SimpleRng,
    iter: usize,
) -> i32 {
    let (ssize, _nop, npckmv, pickmv) = storing(trees);
    let at_max_leaves = ssize >= opts.nleaves;

    let ntr = trees.len();
    if ntr == 0 {
        maybe_trace(trace, opts, iter, temp, *current_score);
        return 0;
    }
    let wh = rng.gen_range(ntr);
    if trees[wh].nodes[0].conc == 0 {
        if ssize < opts.nleaves {
            let mut cand = trees.clone();
            apply_move(&mut cand[wh], MoveKind::NewTree, 0, rng, opts.allow_or);
            let prtr = eval_forest(&cand, x);
            let nop = prtr.len();
            let nleaves = count_leaves_forest(&cand).min(opts.nleaves);
            if let Some((score_val, beta)) = score_model_logicreg(y, &prtr, seps, opts.response, opts.score, weights, opts.penalty, nleaves, opts.minmass, nop) {
                if score_val < *best_score {
                    *best_score = score_val;
                    *best_trees = cand.clone();
                    *best_pred = prtr.get(0).cloned().unwrap_or_else(|| vec![0u8; y.len()]);
                    *best_beta = beta.clone();
                }
                let accept = if score_val <= *current_score {
                    true
                } else {
                    rng.next_f64() < (( *current_score - score_val) / temp).exp()
                };
                if accept {
                    *trees = cand;
                    *current_score = score_val;
                    maybe_trace(trace, opts, iter, temp, *current_score);
                    return 1;
                }
            }
        }
    }

    let mut weights_sel = base_move_weights(opts.allow_or);
    if at_max_leaves {
        // Disallow growing moves when leaf limit is reached.
        weights_sel[3] = 0.0; // SplitLeaf
        weights_sel[4] = 0.0; // Branch
    }
    let pick = pick_move(&npckmv[wh], &pickmv[wh], weights_sel, rng);
    let Some((kind, node)) = pick else {
        maybe_trace(trace, opts, iter, temp, *current_score);
        return 0;
    };
    let mut cand = trees.clone();
    if !apply_move(&mut cand[wh], kind, node, rng, opts.allow_or) {
        maybe_trace(trace, opts, iter, temp, *current_score);
        return -1;
    }

    let prtr = eval_forest(&cand, x);
    let nop = prtr.len();
    let nleaves = count_leaves_forest(&cand).min(opts.nleaves);

    if let Some((score_val, beta)) = score_model_logicreg(y, &prtr, seps, opts.response, opts.score, weights, opts.penalty, nleaves, opts.minmass, nop) {
        if score_val < *best_score {
            *best_score = score_val;
            *best_trees = cand.clone();
            *best_pred = prtr.get(0).cloned().unwrap_or_else(|| vec![0u8; y.len()]);
            *best_beta = beta.clone();
        }
        let accept = if score_val <= *current_score {
            true
        } else {
            rng.next_f64() < ((*current_score - score_val) / temp).exp()
        };
        if accept {
            *trees = cand;
            *current_score = score_val;
            maybe_trace(trace, opts, iter, temp, *current_score);
            return 1;
        }
    }
    maybe_trace(trace, opts, iter, temp, *current_score);
    -1
}

/// Fit LogicReg-like forest with optional separating covariates (seps).
fn logreg_fit_with_sep(
    x: &[Vec<u8>],
    y: &[f64],
    seps: Option<&[Vec<f64>]>,
    mut opts: LogRegOptions,
    weights: Option<&[f64]>,
) -> Result<LogicRegModel, String> {
    let (n, p) = validate_x_y(x, y, opts.response, weights)?;
    if opts.ntrees == 0 { opts.ntrees = 1; }
    if opts.nleaves == 0 { opts.nleaves = 1; }
    if let Some(sep) = seps {
        for col in sep {
            if col.len() != n {
                return Err("seps column length mismatch".to_string());
            }
        }
    }

    // Initialize an empty forest (all-zero trees).
    let mut trees = Vec::with_capacity(opts.ntrees);
    for _ in 0..opts.ntrees { trees.push(empty_tree(p, opts.treesize)); }

    let mut rng = SimpleRng::new(opts.seed);
    // Score the empty model to seed the search.
    let prtr_init = eval_forest(&trees, x);
    let nop_init = prtr_init.len();
    let nleaves_init = 0usize;
    let (mut current_score, beta) = score_model_logicreg(
        y, &prtr_init, seps, opts.response, opts.score, weights, opts.penalty, nleaves_init, opts.minmass, nop_init,
    ).ok_or_else(|| "initial model could not be fitted".to_string())?;

    let mut best_score = current_score;
    let mut best_trees = trees.clone();
    let mut best_pred = prtr_init.get(0).cloned().unwrap_or_else(|| vec![0u8; n]);
    let mut best_beta = beta.clone();
    let mut trace = Vec::new();

    match opts.select {
        SelectMode::Greedy | SelectMode::Fast => {
            // Greedy: evaluate all possible moves and take the best one.
            loop {
                let (ssize, _nop, _npckmv, pickmv) = storing(&trees);
                let at_max_leaves = ssize >= opts.nleaves;
                let mut best_move: Option<(usize, MoveKind, isize, Vec<LogicTree>, Vec<Vec<u8>>, Vec<f64>, f64)> = None;
                for (tidx, _t) in trees.iter().enumerate() {
                    let moves = &pickmv[tidx];
                    for (k, nodes) in moves.iter().enumerate() {
                        if at_max_leaves && (k == 3 || k == 4) { continue; }
                        if nodes.is_empty() { continue; }
                        let kind = match k {
                            0 => MoveKind::AltLeaf,
                            1 => MoveKind::AltOp,
                            2 => MoveKind::DeleteLeaf,
                            3 => MoveKind::SplitLeaf,
                            4 => MoveKind::Branch,
                            _ => MoveKind::Prune,
                        };
                        for &node in nodes {
                            let mut cand = trees.clone();
                            if !apply_move(&mut cand[tidx], kind, node, &mut rng, opts.allow_or) { continue; }
                            let prtr = eval_forest(&cand, x);
                            let nop = prtr.len();
                            let nleaves = count_leaves_forest(&cand).min(opts.nleaves);
                            if let Some((score_val, betas)) = score_model_logicreg(y, &prtr, seps, opts.response, opts.score, weights, opts.penalty, nleaves, opts.minmass, nop) {
                                if best_move.as_ref().map(|m| score_val < m.6).unwrap_or(true) {
                                    best_move = Some((tidx, kind, node, cand, prtr, betas, score_val));
                                }
                            }
                        }
                    }
                }
                if let Some((_tidx, _kind, _node, cand, prtr, betas, score_val)) = best_move {
                    if score_val < best_score {
                        best_score = score_val;
                        best_trees = cand.clone();
                        best_pred = prtr.get(0).cloned().unwrap_or_else(|| vec![0u8; n]);
                        best_beta = betas.clone();
                    }
                    trees = cand;
                    let _ = score_val;
                } else {
                    break;
                }
                if matches!(opts.select, SelectMode::Fast) { break; }
            }
        }
        SelectMode::Anneal => {
            // Anneal: accept worse moves with probability to escape local minima.
            if opts.anneal.start == 0.0 && opts.anneal.end == 0.0 {
                temperature_default_search(
                    &mut trees, x, y, seps, weights, &opts, &mut current_score,
                    &mut best_score, &mut best_trees, &mut best_pred, &mut best_beta, &mut trace, &mut rng,
                );
            } else {
                let tstr = opts.anneal.start;
                let tend = opts.anneal.end;
                let tint = opts.anneal.iter.max(1);
                let mut fcnt = 0;
                let mut nac2 = 0;
                let mut ntot = 0;
                for i in 0..=tint {
                    let temp = 10f64.powf(tstr) * (10f64.powf((tend - tstr) / tint as f64)).powi(i as i32);
                    let old_score = current_score;
                    let acc = annealing_step(
                        &mut trees, x, y, seps, weights, &opts, temp, &mut current_score,
                        &mut best_score, &mut best_trees, &mut best_pred, &mut best_beta, &mut trace, &mut rng, i,
                    );
                    if acc > 0 && (old_score - current_score).abs() > 1e-12 { nac2 += 1; }
                    ntot += 1;
                    if opts.anneal.earlyout > 0 && ntot == opts.anneal.earlyout {
                        if nac2 <= 10 { fcnt += 1; } else { fcnt = 0; }
                        nac2 = 0;
                        ntot = 0;
                        if fcnt == 5 { break; }
                    }
                }
            }
        }
    }

    Ok(LogicRegModel {
        expression: forest_expression(&best_trees),
        trees: best_trees,
        score: best_score,
        xcombine: best_pred,
        betas: best_beta,
        trace,
    })
}

/// Fit LogicReg-like forest without separators.
fn logreg_fit(
    x: &[Vec<u8>],
    y: &[f64],
    opts: LogRegOptions,
    weights: Option<&[f64]>,
) -> Result<LogicRegModel, String> {
    logreg_fit_with_sep(x, y, None, opts, weights)
}


#[pyclass]
pub struct PyLogicRegModel {
    #[pyo3(get)]
    expression: String,
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    xcombine: Vec<u8>,
    #[pyo3(get)]
    betas: Vec<f64>,
    #[pyo3(get)]
    trace: Vec<(usize, f64, f64)>,
}

#[pymethods]
impl PyLogicRegModel {
    fn __repr__(&self) -> String {
        format!("PyLogicRegModel(expression='{}', score={:.6})", self.expression, self.score)
    }
}

fn parse_response(s: &str) -> PyResult<ResponseKind> {
    match s.to_ascii_lowercase().as_str() {
        "binary" | "bin" | "b" => Ok(ResponseKind::Binary),
        "continuous" | "cont" | "c" => Ok(ResponseKind::Continuous),
        _ => Err(PyValueError::new_err("response must be 'binary' or 'continuous'")),
    }
}

fn parse_score(s: &str) -> PyResult<ScoreKind> {
    match s.to_ascii_lowercase().as_str() {
        "loglik" | "loglikelihood" | "deviance" | "ll" => Ok(ScoreKind::LogLikelihood),
        "mse" | "rss" => Ok(ScoreKind::Mse),
        _ => Err(PyValueError::new_err("score must be 'loglik' or 'mse'")),
    }
}

fn parse_select(s: &str) -> PyResult<SelectMode> {
    match s.to_ascii_lowercase().as_str() {
        "anneal" | "a" => Ok(SelectMode::Anneal),
        "greedy" | "g" => Ok(SelectMode::Greedy),
        "fast" | "f" => Ok(SelectMode::Fast),
        _ => Err(PyValueError::new_err("select must be 'anneal', 'greedy', or 'fast'")),
    }
}

#[pyfunction(name = "logregfit")]
#[pyo3(
    signature = (
        x,
        y,
        *,
        response="binary",
        score="loglik",
        ntrees=1,
        nleaves=8,
        treesize=8,
        select="anneal",
        anneal_start=0.0,
        anneal_end=0.0,
        anneal_iter=0,
        anneal_earlyout=0,
        anneal_update=0,
        minmass=0,
        penalty=0.0,
        seed=42,
        allow_or=true,
        weights=None,
        seps=None
    )
)]
pub fn logregfit(
    x: Vec<Vec<u8>>,
    y: Vec<f64>,
    response: &str,
    score: &str,
    ntrees: usize,
    nleaves: usize,
    treesize: usize,
    select: &str,
    anneal_start: f64,
    anneal_end: f64,
    anneal_iter: usize,
    anneal_earlyout: usize,
    anneal_update: usize,
    minmass: usize,
    penalty: f64,
    seed: u64,
    allow_or: bool,
    weights: Option<Vec<f64>>,
    seps: Option<Vec<Vec<f64>>>,
) -> PyResult<PyLogicRegModel> {
    let response = parse_response(response)?;
    let score = parse_score(score)?;
    let select = parse_select(select)?;

    let opts = LogRegOptions {
        response,
        score,
        ntrees,
        nleaves,
        treesize,
        select,
        anneal: AnnealControl {
            start: anneal_start,
            end: anneal_end,
            iter: anneal_iter,
            earlyout: anneal_earlyout,
            update: anneal_update,
        },
        minmass,
        penalty,
        seed,
        allow_or,
    };

    let model = logreg_fit_with_sep(
        &x,
        &y,
        seps.as_deref(),
        opts,
        weights.as_deref(),
    )
    .map_err(PyValueError::new_err)?;

    let trace: Vec<(usize, f64, f64)> = model
        .trace
        .iter()
        .map(|TracePoint { iter, temp, score }| (*iter, *temp, *score))
        .collect();

    Ok(PyLogicRegModel {
        expression: model.expression,
        score: model.score,
        xcombine: model.xcombine,
        betas: model.betas,
        trace,
    })
}
