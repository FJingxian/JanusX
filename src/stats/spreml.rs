use std::cell::RefCell;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::brent::brent_minimize_with_init;
use crate::cholesky::{
    sparse_cholesky_analyze_subset_jxgrm_path_cached, sparse_jxgrm_header_n_samples,
    SparseJxgrmCholesky, SparseJxgrmCholeskyAnalysis, SparseJxgrmSolveWorkspace,
};
use crate::linalg::{cholesky_inplace, cholesky_logdet, cholesky_solve_into};
use crate::stats_common::{map_err_string_to_py, parse_index_vec_i64};

const SPREML_TINY: f64 = 1e-30_f64;

#[derive(Clone, Copy, Debug)]
enum SparseRemlObjective {
    Profile,
    FastGwaFixedVp { vp_fixed: f64 },
}

#[derive(Clone, Debug)]
struct SparseRemlEval {
    log10_lambda: f64,
    lambda: f64,
    sigma_g2: f64,
    sigma_e2: f64,
    ml: f64,
    reml: f64,
}

#[derive(Clone, Debug)]
struct SparseRemlSearchResult {
    best: SparseRemlEval,
    grid: Vec<SparseRemlEval>,
}

#[derive(Clone, Debug, Default)]
struct SparseRemlEvalTiming {
    factorize_secs: f64,
    fill_rhs_secs: f64,
    solve_secs: f64,
    dense_secs: f64,
    total_secs: f64,
}

#[derive(Clone, Debug, Default)]
struct SparseRemlPhaseTiming {
    n_evals: usize,
    factorize_secs: f64,
    fill_rhs_secs: f64,
    solve_secs: f64,
    dense_secs: f64,
    total_secs: f64,
}

impl SparseRemlPhaseTiming {
    fn add_eval(&mut self, timing: &SparseRemlEvalTiming) {
        self.n_evals = self.n_evals.saturating_add(1);
        self.factorize_secs += timing.factorize_secs;
        self.fill_rhs_secs += timing.fill_rhs_secs;
        self.solve_secs += timing.solve_secs;
        self.dense_secs += timing.dense_secs;
        self.total_secs += timing.total_secs;
    }
}

#[derive(Clone, Debug, Default)]
struct SparseRemlTimingSummary {
    analysis_load_secs: f64,
    design_secs: f64,
    grid_secs: f64,
    brent_secs: f64,
    final_eval_secs: f64,
    total_secs: f64,
    grid_eval: SparseRemlPhaseTiming,
    brent_eval: SparseRemlPhaseTiming,
    final_eval: SparseRemlPhaseTiming,
}

fn env_truthy_local(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no" || s == "off")
        })
        .unwrap_or(false)
}

fn sparse_reml_timing_enabled() -> bool {
    env_truthy_local("JX_SPARSE_REML_TIMING")
        || env_truthy_local("JX_SPLMM_PREPARE_STAGE_TIMING")
        || env_truthy_local("JX_SPLMM_PACKED_STAGE_TIMING")
}

fn emit_sparse_reml_timing(
    timing: &SparseRemlTimingSummary,
    n: usize,
    p: usize,
    grid_size: usize,
    max_iter: usize,
) {
    let phase_other = |phase_total: f64, phase_eval: &SparseRemlPhaseTiming| -> f64 {
        (phase_total - phase_eval.total_secs).max(0.0_f64)
    };
    let final_other = phase_other(timing.final_eval_secs, &timing.final_eval);
    let total_other = (timing.total_secs
        - timing.analysis_load_secs
        - timing.design_secs
        - timing.grid_secs
        - timing.brent_secs
        - timing.final_eval_secs)
        .max(0.0_f64);
    eprintln!(
        "SPREML timing: analysis_load={:.3}s, design={:.3}s, grid={:.3}s (evals={}, factorize={:.3}s, fill_rhs={:.3}s, solve={:.3}s, dense={:.3}s, other={:.3}s), brent={:.3}s (evals={}, factorize={:.3}s, fill_rhs={:.3}s, solve={:.3}s, dense={:.3}s, other={:.3}s), final_eval={:.3}s (factorize={:.3}s, fill_rhs={:.3}s, solve={:.3}s, dense={:.3}s, other={:.3}s), total={:.3}s, other={:.3}s, n={}, p={}, grid_size={}, max_iter={}",
        timing.analysis_load_secs,
        timing.design_secs,
        timing.grid_secs,
        timing.grid_eval.n_evals,
        timing.grid_eval.factorize_secs,
        timing.grid_eval.fill_rhs_secs,
        timing.grid_eval.solve_secs,
        timing.grid_eval.dense_secs,
        phase_other(timing.grid_secs, &timing.grid_eval),
        timing.brent_secs,
        timing.brent_eval.n_evals,
        timing.brent_eval.factorize_secs,
        timing.brent_eval.fill_rhs_secs,
        timing.brent_eval.solve_secs,
        timing.brent_eval.dense_secs,
        phase_other(timing.brent_secs, &timing.brent_eval),
        timing.final_eval_secs,
        timing.final_eval.factorize_secs,
        timing.final_eval.fill_rhs_secs,
        timing.final_eval.solve_secs,
        timing.final_eval.dense_secs,
        final_other,
        timing.total_secs,
        total_other,
        n,
        p,
        grid_size,
        max_iter,
    );
}

fn refine_monotone_valid_lower_bound<F>(
    mut is_valid: F,
    invalid_log10: f64,
    valid_log10: f64,
    tol: f64,
    max_iter: usize,
) -> f64
where
    F: FnMut(f64) -> bool,
{
    if !(invalid_log10.is_finite() && valid_log10.is_finite() && invalid_log10 < valid_log10) {
        return valid_log10;
    }
    if !is_valid(valid_log10) {
        return valid_log10;
    }
    if is_valid(invalid_log10) {
        return invalid_log10;
    }
    let tol_use = tol.abs().max(1e-6_f64);
    let mut lo = invalid_log10;
    let mut hi = valid_log10;
    for _ in 0..max_iter.max(1) {
        if (hi - lo).abs() <= tol_use {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if is_valid(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

#[inline]
fn sparse_reml_lambda_factorizable(
    analysis: &SparseJxgrmCholeskyAnalysis,
    log10_lambda: f64,
) -> bool {
    let lambda = 10.0_f64.powf(log10_lambda);
    lambda.is_finite() && lambda > 0.0 && analysis.factorize_k_plus_lambda_i(lambda).is_ok()
}

fn sparse_reml_refine_lower_feasible_log10(
    analysis: &SparseJxgrmCholeskyAnalysis,
    invalid_log10: f64,
    valid_log10: f64,
    tol: f64,
    max_iter: usize,
) -> f64 {
    refine_monotone_valid_lower_bound(
        |x| sparse_reml_lambda_factorizable(analysis, x),
        invalid_log10,
        valid_log10,
        tol,
        max_iter,
    )
}

fn sparse_reml_debug_enabled() -> bool {
    env_truthy_local("JX_SPARSE_REML_DEBUG")
}

struct SparseRemlEvaluator {
    n: usize,
    p: usize,
    rhs: Vec<f64>,
    xt_vinv_y: Vec<f64>,
    xt_vinv_x: Vec<f64>,
    beta_hat: Vec<f64>,
    solve_ws: Option<SparseJxgrmSolveWorkspace>,
    // Reusable buffer for factorize_diag_shifted_buffered — avoids a
    // per-iteration clone of base_perm_values during REML search.
    perm_values_buf: Vec<f64>,
}

impl SparseRemlEvaluator {
    fn new(n: usize, p: usize, analysis: &SparseJxgrmCholeskyAnalysis) -> Self {
        let perm_values_buf = analysis.base_perm_values().to_vec();
        Self {
            n,
            p,
            rhs: vec![0.0_f64; n.saturating_mul(p + 1)],
            xt_vinv_y: vec![0.0_f64; p],
            xt_vinv_x: vec![0.0_f64; p.saturating_mul(p)],
            beta_hat: vec![0.0_f64; p],
            solve_ws: None,
            perm_values_buf,
        }
    }

    #[inline]
    fn n_rhs(&self) -> usize {
        self.p + 1
    }

    fn fill_rhs(&mut self, x_design: &[f64], y: &[f64]) {
        self.rhs[..self.n].copy_from_slice(y);
        for c in 0..self.p {
            for i in 0..self.n {
                self.rhs[(c + 1) * self.n + i] = x_design[i * self.p + c];
            }
        }
    }

    fn ensure_workspace<'a>(
        &'a mut self,
        factor: &SparseJxgrmCholesky,
    ) -> Result<&'a mut SparseJxgrmSolveWorkspace, String> {
        let need_rhs = self.n_rhs();
        let needs_realloc = self
            .solve_ws
            .as_ref()
            .map(|ws| ws.n_rhs_capacity() < need_rhs)
            .unwrap_or(true);
        if needs_realloc {
            self.solve_ws = Some(factor.make_solve_workspace(need_rhs)?);
        }
        self.solve_ws
            .as_mut()
            .ok_or_else(|| "Sparse REML solve workspace allocation failed".to_string())
    }

    fn solve_rhs_in_place(&mut self, factor: &SparseJxgrmCholesky) -> Result<(), String> {
        let n_rhs = self.n_rhs();
        if self.rhs.len() != self.n.saturating_mul(n_rhs) {
            return Err(format!(
                "Sparse REML RHS length mismatch: got {}, expected {}",
                self.rhs.len(),
                self.n.saturating_mul(n_rhs)
            ));
        }
        let mut rhs = std::mem::take(&mut self.rhs);
        {
            let solve_ws = self.ensure_workspace(factor)?;
            factor.solve_in_place_with_workspace(&mut rhs, n_rhs, solve_ws)?;
        }
        self.rhs = rhs;
        Ok(())
    }
}

fn build_design_matrix(x_cov: Option<&[f64]>, n: usize, p0: usize) -> Result<Vec<f64>, String> {
    if n == 0 {
        return Err("SPREML requires n > 0".to_string());
    }
    let mut x = vec![0.0_f64; n * (p0 + 1)];
    for i in 0..n {
        x[i * (p0 + 1)] = 1.0;
    }
    if let Some(cov) = x_cov {
        if cov.len() != n.saturating_mul(p0) {
            return Err(format!(
                "SPREML covariate length mismatch: got {}, expected {} for n={} and p0={}",
                cov.len(),
                n.saturating_mul(p0),
                n,
                p0
            ));
        }
        for i in 0..n {
            let src = &cov[i * p0..(i + 1) * p0];
            let dst = &mut x[i * (p0 + 1) + 1..(i + 1) * (p0 + 1)];
            dst.copy_from_slice(src);
        }
    }
    Ok(x)
}

#[inline]
fn spd_cholesky_with_jitter(matrix: &[f64], dim: usize, label: &str) -> Result<Vec<f64>, String> {
    if dim == 0 {
        return Err(format!("{label} requires dim > 0"));
    }
    if matrix.len() != dim * dim {
        return Err(format!(
            "{label} matrix length mismatch: got {}, expected {}",
            matrix.len(),
            dim * dim
        ));
    }
    let mut chol = matrix.to_vec();
    if cholesky_inplace(&mut chol, dim).is_some() {
        return Ok(chol);
    }
    let trace = (0..dim).map(|i| matrix[i * dim + i].abs()).sum::<f64>();
    let base = (trace / (dim.max(1) as f64)).max(1.0) * 1e-10_f64;
    for k in 0..8 {
        chol.copy_from_slice(matrix);
        let jitter = base * 10.0_f64.powi(k);
        for i in 0..dim {
            chol[i * dim + i] += jitter;
        }
        if cholesky_inplace(&mut chol, dim).is_some() {
            return Ok(chol);
        }
    }
    Err(format!("{label} is not SPD even after diagonal jitter"))
}

fn xt_vec_row_major(x_row_major: &[f64], n: usize, p: usize, y: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for i in 0..n {
        let x_row = &x_row_major[i * p..(i + 1) * p];
        let yi = y[i];
        for j in 0..p {
            out[j] += x_row[j] * yi;
        }
    }
}

fn xt_mat_col_major_rhs(
    x_row_major: &[f64],
    rhs_col_major: &[f64],
    n: usize,
    p: usize,
    out: &mut [f64],
) {
    out.fill(0.0);
    for i in 0..n {
        let x_row = &x_row_major[i * p..(i + 1) * p];
        for c in 0..p {
            let rhs_ic = rhs_col_major[c * n + i];
            for r in 0..p {
                out[r * p + c] += x_row[r] * rhs_ic;
            }
        }
    }
}

fn evaluate_sparse_reml_at_lambda(
    analysis: &SparseJxgrmCholeskyAnalysis,
    x_design: &[f64],
    y: &[f64],
    log10_lambda: f64,
    objective: SparseRemlObjective,
    threads: usize,
    evaluator: &mut SparseRemlEvaluator,
) -> Result<(SparseRemlEval, SparseRemlEvalTiming), String> {
    let total_t0 = Instant::now();
    let mut timing = SparseRemlEvalTiming::default();
    let lambda = 10.0_f64.powf(log10_lambda);
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(format!(
            "SPREML lambda is invalid at log10(lambda)={log10_lambda}"
        ));
    }
    let n = y.len();
    let p = x_design.len() / n;
    if p == 0 || n <= p {
        return Err(format!("SPREML requires n > p, got n={n}, p={p}"));
    }
    if evaluator.n != n || evaluator.p != p {
        return Err(format!(
            "SPREML evaluator shape mismatch: evaluator=({}, {}), data=({}, {})",
            evaluator.n, evaluator.p, n, p
        ));
    }

    let factorize_t0 = Instant::now();
    let factor = analysis.factorize_k_plus_lambda_i_buffered_parallel(
        lambda,
        &mut evaluator.perm_values_buf,
        threads.max(1),
    )?;
    timing.factorize_secs = factorize_t0.elapsed().as_secs_f64();
    let fill_rhs_t0 = Instant::now();
    evaluator.fill_rhs(x_design, y);
    timing.fill_rhs_secs = fill_rhs_t0.elapsed().as_secs_f64();
    let solve_t0 = Instant::now();
    evaluator.solve_rhs_in_place(&factor)?;
    timing.solve_secs = solve_t0.elapsed().as_secs_f64();

    let dense_t0 = Instant::now();
    let y_vinv = &evaluator.rhs[..n];
    let x_vinv_col = &evaluator.rhs[n..];
    let y_vinv_y = y.iter().zip(y_vinv.iter()).map(|(a, b)| a * b).sum::<f64>();

    xt_vec_row_major(x_design, n, p, y_vinv, &mut evaluator.xt_vinv_y);

    xt_mat_col_major_rhs(x_design, x_vinv_col, n, p, &mut evaluator.xt_vinv_x);
    let xt_vinv_x_chol = spd_cholesky_with_jitter(&evaluator.xt_vinv_x, p, "SPREML XtVinvX")?;
    cholesky_solve_into(
        &xt_vinv_x_chol,
        p,
        &evaluator.xt_vinv_y,
        &mut evaluator.beta_hat,
    );

    let ypy = y_vinv_y
        - evaluator
            .xt_vinv_y
            .iter()
            .zip(evaluator.beta_hat.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    if !ypy.is_finite() || ypy <= SPREML_TINY {
        return Err(format!(
            "SPREML profiled residual quadratic form is invalid at lambda={lambda}: yPy={ypy}"
        ));
    }

    let df = (n - p) as f64;
    let log_det_m = factor.logdet();
    let log_det_xt_m_inv_x = cholesky_logdet(&xt_vinv_x_chol, p);
    let (sigma_g2, sigma_e2, ml, reml) = match objective {
        SparseRemlObjective::Profile => {
            let sigma_g2 = ypy / df;
            if !sigma_g2.is_finite() || sigma_g2 <= 0.0 {
                return Err(format!(
                    "SPREML sigma_g2 is invalid at lambda={lambda}: sigma_g2={sigma_g2}"
                ));
            }
            let sigma_e2 = lambda * sigma_g2;
            let c_reml = df * (df.ln() - 1.0 - (2.0 * PI).ln()) * 0.5;
            let reml = c_reml - 0.5 * (df * ypy.ln() + log_det_m + log_det_xt_m_inv_x);
            let n_f = n as f64;
            let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) * 0.5;
            let ml = c_ml - 0.5 * (n_f * ypy.ln() + log_det_m);
            (sigma_g2, sigma_e2, ml, reml)
        }
        SparseRemlObjective::FastGwaFixedVp { vp_fixed } => {
            if !vp_fixed.is_finite() || vp_fixed <= 0.0 {
                return Err(format!(
                    "SPREML fastGWA fixed-Vp objective requires finite vp_fixed > 0, got {vp_fixed}"
                ));
            }
            let sigma_g2 = vp_fixed / (1.0 + lambda);
            if !sigma_g2.is_finite() || sigma_g2 <= 0.0 {
                return Err(format!(
                    "SPREML fastGWA sigma_g2 is invalid at lambda={lambda}: sigma_g2={sigma_g2}"
                ));
            }
            let sigma_e2 = lambda * sigma_g2;
            let reml =
                -0.5 * (df * sigma_g2.ln() + log_det_m + log_det_xt_m_inv_x + (ypy / sigma_g2));
            (sigma_g2, sigma_e2, f64::NAN, reml)
        }
    };
    if !reml.is_finite() || !(ml.is_finite() || ml.is_nan()) {
        return Err(format!(
            "SPREML likelihood is invalid at lambda={lambda}: ml={ml}, reml={reml}"
        ));
    }
    timing.dense_secs = dense_t0.elapsed().as_secs_f64();
    timing.total_secs = total_t0.elapsed().as_secs_f64();

    Ok((
        SparseRemlEval {
            log10_lambda,
            lambda,
            sigma_g2,
            sigma_e2,
            ml,
            reml,
        },
        timing,
    ))
}

fn sparse_reml_grid_search_core(
    analysis: &SparseJxgrmCholeskyAnalysis,
    x_design: &[f64],
    y: &[f64],
    low: f64,
    high: f64,
    grid_size: usize,
    objective: SparseRemlObjective,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_offset: usize,
    progress_total: usize,
) -> Result<(SparseRemlSearchResult, SparseRemlPhaseTiming), String> {
    if !low.is_finite() || !high.is_finite() || low >= high {
        return Err(format!(
            "SPREML grid search requires finite low < high, got low={low}, high={high}"
        ));
    }
    let grid_n = grid_size.max(2);
    let n = y.len();
    let p = x_design.len() / n.max(1);
    let mut evaluator = SparseRemlEvaluator::new(n, p, analysis);
    let mut evals = Vec::<SparseRemlEval>::with_capacity(grid_n);
    let mut best = None::<SparseRemlEval>;
    let mut first_err = None::<String>;
    let mut phase_timing = SparseRemlPhaseTiming::default();
    for idx in 0..grid_n {
        let t = if grid_n <= 1 {
            0.0_f64
        } else {
            idx as f64 / (grid_n - 1) as f64
        };
        let log10_lambda = low + (high - low) * t;
        match evaluate_sparse_reml_at_lambda(
            analysis,
            x_design,
            y,
            log10_lambda,
            objective,
            threads,
            &mut evaluator,
        ) {
            Ok((eval, eval_timing)) => {
                phase_timing.add_eval(&eval_timing);
                if best
                    .as_ref()
                    .map(|cur| eval.reml > cur.reml)
                    .unwrap_or(true)
                {
                    best = Some(eval.clone());
                }
                evals.push(eval);
            }
            Err(err) => {
                if sparse_reml_debug_enabled() {
                    eprintln!("SPREML grid eval failed at log10(lambda)={log10_lambda:.6e}: {err}");
                }
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }
        emit_sparse_reml_progress(
            progress_callback,
            progress_offset.saturating_add(idx + 1),
            progress_total,
        )?;
    }
    let best = best.ok_or_else(|| match first_err {
        Some(err) => format!(
            "SPREML sparse grid search found no valid lambda in [{low}, {high}]; first failure: {err}"
        ),
        None => format!("SPREML sparse grid search found no valid lambda in [{low}, {high}]"),
    })?;
    Ok((SparseRemlSearchResult { best, grid: evals }, phase_timing))
}

fn sparse_reml_brent_search_with_progress(
    analysis: &SparseJxgrmCholeskyAnalysis,
    x_design: &[f64],
    y: &[f64],
    low: f64,
    high: f64,
    grid_size: usize,
    tol: f64,
    max_iter: usize,
    objective: SparseRemlObjective,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_offset: usize,
    progress_total: usize,
) -> Result<(SparseRemlSearchResult, SparseRemlTimingSummary), String> {
    if !(tol.is_finite() && tol > 0.0) {
        return Err(format!(
            "SPREML Brent tol must be finite and > 0, got {tol}"
        ));
    }
    if max_iter == 0 {
        return Err("SPREML Brent max_iter must be > 0".to_string());
    }
    let grid_n = grid_size.max(2);
    let total_t0 = Instant::now();
    let mut timing = SparseRemlTimingSummary::default();
    let grid_t0 = Instant::now();
    let (grid, grid_phase_timing) = sparse_reml_grid_search_core(
        analysis,
        x_design,
        y,
        low,
        high,
        grid_size,
        objective,
        threads,
        progress_callback,
        progress_offset,
        progress_total,
    )?;
    timing.grid_secs = grid_t0.elapsed().as_secs_f64();
    timing.grid_eval = grid_phase_timing;
    let init = Some(grid.best.log10_lambda);
    let best_idx = grid
        .grid
        .iter()
        .position(|eval| eval.log10_lambda == grid.best.log10_lambda)
        .unwrap_or(0usize);
    let mut brent_low = if best_idx > 0 {
        grid.grid[best_idx - 1].log10_lambda
    } else {
        low
    };
    let brent_high = if best_idx + 1 < grid.grid.len() {
        grid.grid[best_idx + 1].log10_lambda
    } else {
        high
    };
    if best_idx == 0 && !grid.grid.is_empty() && grid_n > 1 {
        let raw_step = (high - low) / (grid_n - 1) as f64;
        let first_valid = grid.grid[0].log10_lambda;
        let prev_raw = (first_valid - raw_step).max(low);
        if prev_raw < first_valid {
            brent_low = sparse_reml_refine_lower_feasible_log10(
                analysis,
                prev_raw,
                first_valid,
                tol.min(1e-2_f64),
                24,
            );
        }
    }
    let (brent_low, brent_high) =
        if brent_low.is_finite() && brent_high.is_finite() && brent_low < brent_high {
            (brent_low, brent_high)
        } else {
            (low, high)
        };
    let brent_budget = max_iter.max(1);
    let brent_progress_err = RefCell::new(None::<String>);
    let brent_eval_err = RefCell::new(None::<String>);
    let n = y.len();
    let p = x_design.len() / n.max(1);
    let evaluator = RefCell::new(SparseRemlEvaluator::new(n, p, analysis));
    let brent_phase_timing = RefCell::new(SparseRemlPhaseTiming::default());
    let mut brent_done = 0usize;
    let brent_t0 = Instant::now();
    let (best_log10, _cost) = brent_minimize_with_init(
        |x0| {
            if brent_progress_err.borrow().is_some() {
                return 1e300_f64;
            }
            let cost = match evaluate_sparse_reml_at_lambda(
                analysis,
                x_design,
                y,
                x0,
                objective,
                threads,
                &mut evaluator.borrow_mut(),
            ) {
                Ok((eval, eval_timing)) => {
                    brent_phase_timing.borrow_mut().add_eval(&eval_timing);
                    -eval.reml
                }
                Err(err) => {
                    if sparse_reml_debug_enabled() {
                        eprintln!("SPREML Brent eval failed at log10(lambda)={x0:.6e}: {err}");
                    }
                    if brent_eval_err.borrow().is_none() {
                        *brent_eval_err.borrow_mut() = Some(err);
                    }
                    1e300_f64
                }
            };
            brent_done = (brent_done + 1).min(brent_budget);
            if let Err(err) = emit_sparse_reml_progress(
                progress_callback,
                progress_offset
                    .saturating_add(grid_n)
                    .saturating_add(brent_done),
                progress_total,
            ) {
                *brent_progress_err.borrow_mut() = Some(err);
                return 1e300_f64;
            }
            cost
        },
        brent_low,
        brent_high,
        tol,
        max_iter,
        init,
    );
    timing.brent_secs = brent_t0.elapsed().as_secs_f64();
    timing.brent_eval = brent_phase_timing.into_inner();
    if let Some(err) = brent_progress_err.into_inner() {
        return Err(err);
    }
    let final_eval_t0 = Instant::now();
    let (best_eval, final_eval_timing) = evaluate_sparse_reml_at_lambda(
        analysis,
        x_design,
        y,
        best_log10,
        objective,
        threads,
        &mut evaluator.into_inner(),
    )
    .map_err(|err| match brent_eval_err.into_inner() {
        Some(first_err) => format!(
            "SPREML Brent final evaluation failed at log10(lambda)={best_log10:.6e}: {err}; earlier failure: {first_err}"
        ),
        None => err,
    })?;
    timing.final_eval_secs = final_eval_t0.elapsed().as_secs_f64();
    timing.final_eval.add_eval(&final_eval_timing);
    timing.total_secs = total_t0.elapsed().as_secs_f64();
    Ok((
        SparseRemlSearchResult {
            best: best_eval,
            grid: grid.grid,
        },
        timing,
    ))
}

#[inline]
fn emit_sparse_reml_progress(
    cb: Option<&Py<PyAny>>,
    done: usize,
    total: usize,
) -> Result<(), String> {
    let total_use = total.max(1);
    let done_use = done.min(total_use);
    if let Some(cb) = cb {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (done_use, total_use))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn load_subset_analysis_from_path(
    path: &str,
    sample_idx_raw: Option<&[i64]>,
) -> Result<Arc<SparseJxgrmCholeskyAnalysis>, String> {
    let sample_idx_vec = if let Some(raw) = sample_idx_raw {
        let n_samples = sparse_jxgrm_header_n_samples(path)?;
        Some(parse_index_vec_i64(raw, n_samples, "sample_indices").map_err(|e| e.to_string())?)
    } else {
        None
    };
    sparse_cholesky_analyze_subset_jxgrm_path_cached(path, sample_idx_vec.as_deref())
}

fn result_to_tuple(
    result: SparseRemlSearchResult,
) -> (
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let mut grid_log10 = Vec::with_capacity(result.grid.len());
    let mut grid_reml = Vec::with_capacity(result.grid.len());
    let mut grid_sigma_g2 = Vec::with_capacity(result.grid.len());
    let mut grid_sigma_e2 = Vec::with_capacity(result.grid.len());
    for eval in result.grid.iter() {
        grid_log10.push(eval.log10_lambda);
        grid_reml.push(eval.reml);
        grid_sigma_g2.push(eval.sigma_g2);
        grid_sigma_e2.push(eval.sigma_e2);
    }
    (
        result.best.lambda,
        result.best.sigma_g2,
        result.best.sigma_e2,
        result.best.ml,
        result.best.reml,
        result.best.log10_lambda,
        grid_log10,
        grid_reml,
        grid_sigma_g2,
        grid_sigma_e2,
    )
}

#[pyfunction]
#[pyo3(signature = (
    jxgrm_path,
    y,
    x_cov=None,
    sample_indices=None,
    low=-5.0,
    high=5.0,
    grid_size=33,
    threads=1
))]
pub fn spreml_sparse_reml_grid_from_jxgrm<'py>(
    py: Python<'py>,
    jxgrm_path: String,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    grid_size: usize,
    threads: usize,
) -> PyResult<(
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    let y_vec = y
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("y must be contiguous"))?
        .to_vec();
    let n = y_vec.len();
    let x_cov_buf: Option<(Vec<f64>, usize)> = if let Some(x) = x_cov {
        let xa = x.as_array();
        if xa.ndim() != 2 || xa.shape()[0] != n {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov shape mismatch: got {:?}, expected ({n}, p)",
                xa.shape()
            )));
        }
        let p0 = xa.shape()[1];
        let xs = x
            .as_slice()
            .map_err(|_| PyRuntimeError::new_err("x_cov must be contiguous (C-order)"))?
            .to_vec();
        Some((xs, p0))
    } else {
        None
    };
    let sample_idx_raw = if let Some(sidx) = sample_indices {
        Some(sidx.as_slice()?.to_vec())
    } else {
        None
    };

    py.detach(move || {
        let analysis = load_subset_analysis_from_path(&jxgrm_path, sample_idx_raw.as_deref())?;
        if analysis.dim() != y_vec.len() {
            return Err(format!(
                "SPREML subset sample size mismatch: sparse n={}, phenotype n={}",
                analysis.dim(),
                y_vec.len()
            ));
        }
        let (x_cov_slice, p0) = match x_cov_buf.as_ref() {
            Some((buf, p)) => (Some(buf.as_slice()), *p),
            None => (None, 0usize),
        };
        let x_design = build_design_matrix(x_cov_slice, y_vec.len(), p0)?;
        let result = sparse_reml_grid_search_core(
            &analysis,
            &x_design,
            &y_vec,
            low,
            high,
            grid_size,
            SparseRemlObjective::Profile,
            threads.max(1),
            None,
            0,
            1,
        )?
        .0;
        Ok(result_to_tuple(result))
    })
    .map_err(map_err_string_to_py)
}

#[pyfunction]
#[pyo3(signature = (
    jxgrm_path,
    y,
    x_cov=None,
    sample_indices=None,
    low=-5.0,
    high=5.0,
    grid_size=9,
    tol=1e-3,
    max_iter=20,
    threads=1,
    progress_callback=None
))]
pub fn spreml_sparse_reml_brent_from_jxgrm<'py>(
    py: Python<'py>,
    jxgrm_path: String,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    grid_size: usize,
    tol: f64,
    max_iter: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<(
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    let y_vec = y
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("y must be contiguous"))?
        .to_vec();
    let n = y_vec.len();
    let x_cov_buf: Option<(Vec<f64>, usize)> = if let Some(x) = x_cov {
        let xa = x.as_array();
        if xa.ndim() != 2 || xa.shape()[0] != n {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov shape mismatch: got {:?}, expected ({n}, p)",
                xa.shape()
            )));
        }
        let p0 = xa.shape()[1];
        let xs = x
            .as_slice()
            .map_err(|_| PyRuntimeError::new_err("x_cov must be contiguous (C-order)"))?
            .to_vec();
        Some((xs, p0))
    } else {
        None
    };
    let sample_idx_raw = if let Some(sidx) = sample_indices {
        Some(sidx.as_slice()?.to_vec())
    } else {
        None
    };

    py.detach(move || {
        let total_t0 = Instant::now();
        let timing_enabled = sparse_reml_timing_enabled();
        let total_progress = 1usize
            .saturating_add(grid_size.max(2))
            .saturating_add(max_iter.max(1));
        emit_sparse_reml_progress(progress_callback.as_ref(), 0, total_progress)?;
        let analysis_t0 = Instant::now();
        let analysis = load_subset_analysis_from_path(&jxgrm_path, sample_idx_raw.as_deref())?;
        let analysis_load_secs = analysis_t0.elapsed().as_secs_f64();
        emit_sparse_reml_progress(progress_callback.as_ref(), 1, total_progress)?;
        if analysis.dim() != y_vec.len() {
            return Err(format!(
                "SPREML subset sample size mismatch: sparse n={}, phenotype n={}",
                analysis.dim(),
                y_vec.len()
            ));
        }
        let (x_cov_slice, p0) = match x_cov_buf.as_ref() {
            Some((buf, p)) => (Some(buf.as_slice()), *p),
            None => (None, 0usize),
        };
        let design_t0 = Instant::now();
        let x_design = build_design_matrix(x_cov_slice, y_vec.len(), p0)?;
        let design_secs = design_t0.elapsed().as_secs_f64();
        let p = x_design.len() / y_vec.len().max(1);
        let (result, mut timing) = sparse_reml_brent_search_with_progress(
            &analysis,
            &x_design,
            &y_vec,
            low,
            high,
            grid_size,
            tol,
            max_iter,
            SparseRemlObjective::Profile,
            threads.max(1),
            progress_callback.as_ref(),
            1,
            total_progress,
        )?;
        timing.analysis_load_secs = analysis_load_secs;
        timing.design_secs = design_secs;
        timing.total_secs = total_t0.elapsed().as_secs_f64();
        if timing_enabled {
            emit_sparse_reml_timing(&timing, y_vec.len(), p, grid_size.max(2), max_iter.max(1));
        }
        emit_sparse_reml_progress(progress_callback.as_ref(), total_progress, total_progress)?;
        Ok(result_to_tuple(result))
    })
    .map_err(map_err_string_to_py)
}

#[pyfunction]
#[pyo3(signature = (
    jxgrm_path,
    y_resid,
    vp_fixed,
    sample_indices=None,
    low=-5.0,
    high=5.0,
    grid_size=9,
    tol=1e-3,
    max_iter=20,
    threads=1,
    progress_callback=None
))]
pub fn spreml_sparse_fastgwa_fixed_vp_brent_from_jxgrm<'py>(
    py: Python<'py>,
    jxgrm_path: String,
    y_resid: PyReadonlyArray1<'py, f64>,
    vp_fixed: f64,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    grid_size: usize,
    tol: f64,
    max_iter: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<(
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    let y_vec = y_resid
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("y_resid must be contiguous"))?
        .to_vec();
    if y_vec.is_empty() {
        return Err(PyRuntimeError::new_err(
            "SparseLMM fastGWA sparse null fit requires non-empty y_resid",
        ));
    }
    if !(vp_fixed.is_finite() && vp_fixed > 0.0) {
        return Err(PyRuntimeError::new_err(format!(
            "SparseLMM fastGWA sparse null fit requires finite vp_fixed > 0, got {vp_fixed}"
        )));
    }
    let sample_idx_raw = if let Some(sidx) = sample_indices {
        Some(sidx.as_slice()?.to_vec())
    } else {
        None
    };

    py.detach(move || {
        let total_t0 = Instant::now();
        let timing_enabled = sparse_reml_timing_enabled();
        let total_progress = 1usize
            .saturating_add(grid_size.max(2))
            .saturating_add(max_iter.max(1));
        emit_sparse_reml_progress(progress_callback.as_ref(), 0, total_progress)?;
        let analysis_t0 = Instant::now();
        let analysis = load_subset_analysis_from_path(&jxgrm_path, sample_idx_raw.as_deref())?;
        let analysis_load_secs = analysis_t0.elapsed().as_secs_f64();
        emit_sparse_reml_progress(progress_callback.as_ref(), 1, total_progress)?;
        if analysis.dim() != y_vec.len() {
            return Err(format!(
                "SparseLMM fastGWA sparse null fit sample size mismatch: sparse n={}, phenotype n={}",
                analysis.dim(),
                y_vec.len()
            ));
        }
        let design_t0 = Instant::now();
        let x_design = build_design_matrix(None, y_vec.len(), 0usize)?;
        let design_secs = design_t0.elapsed().as_secs_f64();
        let p = x_design.len() / y_vec.len().max(1);
        let (result, mut timing) = sparse_reml_brent_search_with_progress(
            &analysis,
            &x_design,
            &y_vec,
            low,
            high,
            grid_size,
            tol,
            max_iter,
            SparseRemlObjective::FastGwaFixedVp { vp_fixed },
            threads.max(1),
            progress_callback.as_ref(),
            1,
            total_progress,
        )?;
        timing.analysis_load_secs = analysis_load_secs;
        timing.design_secs = design_secs;
        timing.total_secs = total_t0.elapsed().as_secs_f64();
        if timing_enabled {
            emit_sparse_reml_timing(&timing, y_vec.len(), p, grid_size.max(2), max_iter.max(1));
        }
        emit_sparse_reml_progress(progress_callback.as_ref(), total_progress, total_progress)?;
        Ok(result_to_tuple(result))
    })
    .map_err(map_err_string_to_py)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cholesky::sparse_cholesky_analyze_jxgrm_csc;
    use crate::spgrm::SparseGrmCsc;
    use nalgebra::{DMatrix, DVector};

    fn assert_close(lhs: f64, rhs: f64, tol: f64, label: &str) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "{label} mismatch: lhs={lhs}, rhs={rhs}, tol={tol}"
        );
    }

    #[test]
    fn sparse_reml_grid_runs_on_small_csc() {
        let csc = SparseGrmCsc {
            n_samples: 3,
            nnz: 5,
            col_ptr: vec![0, 2, 4, 5],
            row_indices: vec![0, 1, 1, 2, 2],
            values: vec![1.0, 0.2, 1.0, 0.1, 1.0],
        };
        let analysis = sparse_cholesky_analyze_jxgrm_csc(&csc).unwrap();
        let y = vec![1.0, 0.5, -0.3];
        let x = build_design_matrix(None, 3, 0).unwrap();
        let res = sparse_reml_grid_search_core(
            &analysis,
            &x,
            &y,
            -3.0,
            2.0,
            9,
            SparseRemlObjective::Profile,
            1,
            None,
            0,
            1,
        )
        .unwrap()
        .0;
        assert!(res.best.lambda.is_finite() && res.best.lambda > 0.0);
        assert!(res.best.sigma_g2.is_finite() && res.best.sigma_g2 > 0.0);
        assert!(!res.grid.is_empty());
    }

    #[test]
    fn refine_monotone_valid_lower_bound_tracks_feasible_edge() {
        let refined = refine_monotone_valid_lower_bound(
            |x| x >= -0.137_f64,
            -0.625_f64,
            0.0_f64,
            1e-6_f64,
            64,
        );
        assert!((refined + 0.137_f64).abs() <= 1e-4_f64);
    }

    #[test]
    fn sparse_reml_fixed_lambda_matches_dense_profile_on_indefinite_k() {
        // K has eigenvalues 3 and -1, so it is indefinite, but K + lambda I
        // is SPD for lambda > 1. This is the regime SparseLMM exact null-fit
        // uses on thresholded sparse GRMs.
        let csc = SparseGrmCsc {
            n_samples: 2,
            nnz: 3,
            col_ptr: vec![0, 2, 3],
            row_indices: vec![0, 1, 1],
            values: vec![1.0, 2.0, 1.0],
        };
        let analysis = sparse_cholesky_analyze_jxgrm_csc(&csc).unwrap();
        let y = vec![0.5_f64, -1.25_f64];
        let x = build_design_matrix(None, 2, 0).unwrap();
        let mut evaluator = SparseRemlEvaluator::new(y.len(), 1, &analysis);
        let lambda = 1.5_f64;
        let log10_lambda = lambda.log10();

        let (eval, _timing) = evaluate_sparse_reml_at_lambda(
            &analysis,
            &x,
            &y,
            log10_lambda,
            SparseRemlObjective::Profile,
            1,
            &mut evaluator,
        )
        .unwrap();

        let k = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let v = k + DMatrix::<f64>::identity(2, 2) * lambda;
        let chol_v = v.clone().cholesky().unwrap();
        let y_vec = DVector::<f64>::from_row_slice(&y);
        let x_mat = DMatrix::<f64>::from_row_slice(2, 1, &[1.0, 1.0]);
        let vinv_y = chol_v.solve(&y_vec);
        let vinv_x = chol_v.solve(&x_mat);
        let xt_vinv_x = x_mat.transpose() * &vinv_x;
        let chol_x = xt_vinv_x.clone().cholesky().unwrap();
        let xt_vinv_y = x_mat.transpose() * &vinv_y;
        let beta = chol_x.solve(&xt_vinv_y);
        let py = vinv_y - vinv_x * beta;
        let ypy = y_vec.dot(&py);
        let df = 1.0_f64;
        let sigma_g2 = ypy / df;
        let log_det_v = 2.0_f64 * chol_v.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        let log_det_xt = 2.0_f64 * chol_x.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        let c_reml = df * (df.ln() - 1.0_f64 - (2.0_f64 * PI).ln()) * 0.5_f64;
        let reml = c_reml - 0.5_f64 * (df * ypy.ln() + log_det_v + log_det_xt);
        let n_f = 2.0_f64;
        let c_ml = n_f * (n_f.ln() - 1.0_f64 - (2.0_f64 * PI).ln()) * 0.5_f64;
        let ml = c_ml - 0.5_f64 * (n_f * ypy.ln() + log_det_v);

        assert_close(eval.lambda, lambda, 1e-12_f64, "lambda");
        assert_close(eval.sigma_g2, sigma_g2, 1e-12_f64, "sigma_g2");
        assert_close(eval.sigma_e2, lambda * sigma_g2, 1e-12_f64, "sigma_e2");
        assert_close(eval.reml, reml, 1e-12_f64, "reml");
        assert_close(eval.ml, ml, 1e-12_f64, "ml");
    }

    #[test]
    fn sparse_fastgwa_fixed_vp_matches_dense_fixed_vp_objective() {
        let csc = SparseGrmCsc {
            n_samples: 3,
            nnz: 5,
            col_ptr: vec![0, 2, 4, 5],
            row_indices: vec![0, 1, 1, 2, 2],
            values: vec![1.0, 0.2, 1.0, 0.1, 1.0],
        };
        let analysis = sparse_cholesky_analyze_jxgrm_csc(&csc).unwrap();
        let y = vec![0.75_f64, -0.10_f64, -0.65_f64];
        let x = build_design_matrix(None, 3, 0).unwrap();
        let mut evaluator = SparseRemlEvaluator::new(y.len(), 1, &analysis);
        let lambda = 1.25_f64;
        let log10_lambda = lambda.log10();
        let vp_fixed = 1.2_f64;

        let (eval, _timing) = evaluate_sparse_reml_at_lambda(
            &analysis,
            &x,
            &y,
            log10_lambda,
            SparseRemlObjective::FastGwaFixedVp { vp_fixed },
            1,
            &mut evaluator,
        )
        .unwrap();

        let sigma_g2 = vp_fixed / (1.0 + lambda);
        let sigma_e2 = lambda * sigma_g2;
        let k = DMatrix::<f64>::from_row_slice(
            3,
            3,
            &[
                1.0, 0.2, 0.0, //
                0.2, 1.0, 0.1, //
                0.0, 0.1, 1.0,
            ],
        );
        let v = k.scale(sigma_g2) + DMatrix::<f64>::identity(3, 3).scale(sigma_e2);
        let chol_v = v.clone().cholesky().unwrap();
        let y_vec = DVector::<f64>::from_row_slice(&y);
        let x_mat = DMatrix::<f64>::from_row_slice(3, 1, &[1.0, 1.0, 1.0]);
        let vinv_y = chol_v.solve(&y_vec);
        let vinv_x = chol_v.solve(&x_mat);
        let xt_vinv_x = x_mat.transpose() * &vinv_x;
        let chol_x = xt_vinv_x.clone().cholesky().unwrap();
        let xt_vinv_y = x_mat.transpose() * &vinv_y;
        let beta = chol_x.solve(&xt_vinv_y);
        let py = vinv_y - vinv_x * beta;
        let ypy = y_vec.dot(&py);
        let log_det_v = 2.0_f64 * chol_v.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        let log_det_xt = 2.0_f64 * chol_x.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        let reml = -0.5_f64 * (log_det_v + log_det_xt + ypy);

        assert_close(eval.lambda, lambda, 1e-12_f64, "fastgwa lambda");
        assert_close(eval.sigma_g2, sigma_g2, 1e-12_f64, "fastgwa sigma_g2");
        assert_close(eval.sigma_e2, sigma_e2, 1e-12_f64, "fastgwa sigma_e2");
        assert_close(eval.reml, reml, 1e-12_f64, "fastgwa reml");
        assert!(eval.ml.is_nan(), "fastgwa ml should remain NaN");
    }
}
