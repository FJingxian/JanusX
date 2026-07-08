// SparseLMM approximate scan on the residualized / GRAMMAR-gamma scale:
// M_X = I - X(X'X)^{-1}X'
// y_tilde = M_X y, g_tilde = M_X g
// fit lambda from y_tilde under V_lambda = K_sparse + lambda I
// a = V_lambda^{-1} y_tilde
// gamma is tuned on sampled markers using fastGWA-style null-SNP filtering:
// keep sampled markers with chi^2 = (g_tilde'a)^2 / (g_tilde'V_lambda^{-1}g_tilde) < 5
// then average gamma_j = (g_tilde'V_lambda^{-1}g_tilde) / (g_tilde'g_tilde)
//
// For diagnostics we also track the exact Schur-form quantity
// g'P_lambda g / (g_tilde'g_tilde), where
// P_lambda = V_lambda^{-1}
//          - V_lambda^{-1}X(X'V_lambda^{-1}X)^{-1}X'V_lambda^{-1}
// beta_hat ~= (g_tilde' a) / (gamma * g_tilde'g_tilde)
// se(beta_hat) ~= 1 / sqrt(gamma * g_tilde'g_tilde)
//
// This path keeps the null fit and scan on the same residualized scale and does not
// reuse the exact SparseLMM sigma2 chain.

use crate::brent::brent_minimize_with_init;
use crate::cholesky::{SparseJxgrmCholesky, SparseJxgrmCholeskyAnalysis};
use crate::decode::PackedGeneticModel;
use crate::gfcore::read_fam;
use crate::linalg::{chi2_sf_df1, cholesky_inplace, cholesky_solve_into};
use crate::splmm::{
    choose_rhat_rows, decode_rhat_markers_col_major, emit_progress_callback,
    emit_splmm_null_scan_core_timing, n_rhat_progress_total, scan_to_tsv_with_py_and_rhat,
    scan_with_py_and_rhat, splmm_residual_sumsq_is_effectively_zero,
    splmm_top_level_timing_enabled, trivial_pcg_null_info, trivial_pcg_rhat_info,
    SplmmPreparedInput, SplmmScanResult, SplmmScanToTsvResult, SplmmTsvMetaInput,
};
use pyo3::prelude::*;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

const SPLMM_APPROX_TINY: f64 = 1e-30_f64;
const SPLMM_APPROX_FASTGWA_NULL_CHISQ_MAX: f64 = 5.0_f64;
const SPLMM_APPROX_SAMPLE_DEBUG_PATH_ENV: &str = "JX_SPLMM_APPROX_SAMPLE_DEBUG_PATH";

#[derive(Clone, Debug)]
struct ResidualizedApproxEval {
    log10_lambda: f64,
    lambda: f64,
    sigma_g2: f64,
    sigma_e2: f64,
    ml: f64,
    reml: f64,
}

#[derive(Debug)]
pub(crate) struct ResidualizedApproxNullFit {
    pub(crate) y_resid: Vec<f64>,
    pub(crate) a_vec: Vec<f64>,
    pub(crate) lambda: f64,
    pub(crate) sigma_g2: f64,
    pub(crate) sigma_e2: f64,
    pub(crate) ml: f64,
    pub(crate) reml: f64,
    // Keeps sampled-marker gamma on the same V^-1 scale as a_vec when the factor is
    // K + lambda I only.
    gamma_scale_correction: f64,
    factor: SparseJxgrmCholesky,
}

#[derive(Clone, Debug)]
pub(crate) struct ResidualizedApproxScanModel {
    x_design: Vec<f64>,
    xtx_chol: Vec<f64>,
    a_resid: Vec<f64>,
    gamma: f64,
    n: usize,
    p: usize,
}

struct ResidualizedApproxEvaluator {
    n: usize,
    n_eff: usize,
    threads: usize,
    y_resid: Vec<f64>,
    rhs: Vec<f64>,
    perm_values_buf: Vec<f64>,
}

impl ResidualizedApproxEvaluator {
    fn new(
        analysis: &SparseJxgrmCholeskyAnalysis,
        y_resid: Vec<f64>,
        n_eff: usize,
        threads: usize,
    ) -> Result<Self, String> {
        let n = y_resid.len();
        if n == 0 {
            return Err("SparseLMM residualized approx requires n > 0".to_string());
        }
        if n_eff == 0 || n_eff > n {
            return Err(format!(
                "SparseLMM residualized approx requires effective df in 1..=n, got n_eff={n_eff}, n={n}"
            ));
        }
        Ok(Self {
            n,
            n_eff,
            threads: threads.max(1),
            rhs: vec![0.0_f64; n],
            y_resid,
            perm_values_buf: analysis.base_perm_values().to_vec(),
        })
    }

    fn evaluate(
        &mut self,
        analysis: &SparseJxgrmCholeskyAnalysis,
        log10_lambda: f64,
    ) -> Result<(ResidualizedApproxEval, Vec<f64>), String> {
        let lambda = 10.0_f64.powf(log10_lambda);
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(format!(
                "SparseLMM residualized approx lambda must be finite and > 0, got log10(lambda)={log10_lambda}"
            ));
        }
        let factor = analysis.factorize_k_plus_lambda_i_buffered_parallel(
            lambda,
            &mut self.perm_values_buf,
            self.threads,
        )?;
        self.rhs.copy_from_slice(&self.y_resid);
        factor.solve_in_place(&mut self.rhs, 1)?;
        let ypy = dot_f64(self.y_resid.as_slice(), self.rhs.as_slice());
        if !ypy.is_finite() || ypy <= SPLMM_APPROX_TINY {
            return Err(format!(
                "SparseLMM residualized approx produced invalid y'V^-1y at lambda={lambda}: {ypy}"
            ));
        }
        let n_f = self.n_eff as f64;
        let sigma_g2 = ypy / n_f;
        if !sigma_g2.is_finite() || sigma_g2 <= 0.0 {
            return Err(format!(
                "SparseLMM residualized approx produced invalid sigma_g2 at lambda={lambda}: {sigma_g2}"
            ));
        }
        let sigma_e2 = lambda * sigma_g2;
        let log_det_v = factor.logdet();
        let c_const = n_f * (n_f.ln() - 1.0 - (2.0 * std::f64::consts::PI).ln()) * 0.5;
        let reml = c_const - 0.5 * (n_f * ypy.ln() + log_det_v);
        let ml = reml;
        if !reml.is_finite() || !ml.is_finite() {
            return Err(format!(
                "SparseLMM residualized approx produced invalid likelihood at lambda={lambda}: ml={ml}, reml={reml}"
            ));
        }
        Ok((
            ResidualizedApproxEval {
                log10_lambda,
                lambda,
                sigma_g2,
                sigma_e2,
                ml,
                reml,
            },
            self.rhs.clone(),
        ))
    }
}

fn splmm_approx_sample_debug_path() -> Option<String> {
    std::env::var(SPLMM_APPROX_SAMPLE_DEBUG_PATH_ENV)
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty())
}

fn maybe_write_residualized_approx_sample_debug_tsv(
    bed_prefix: Option<&str>,
    scan_sample_idx: &[usize],
    y: &[f64],
    fit: &ResidualizedApproxNullFit,
    model: &ResidualizedApproxScanModel,
) -> Result<(), String> {
    let Some(path) = splmm_approx_sample_debug_path() else {
        return Ok(());
    };
    if scan_sample_idx.len() != y.len() {
        return Err(format!(
            "SparseLMM approx sample debug requires sample_indices/y length match: sample_indices={}, y={}",
            scan_sample_idx.len(),
            y.len()
        ));
    }
    if fit.y_resid.len() != y.len()
        || fit.a_vec.len() != y.len()
        || model.score_vec().len() != y.len()
    {
        return Err(format!(
            "SparseLMM approx sample debug length mismatch: y={}, y_resid={}, a_vec={}, a_resid={}",
            y.len(),
            fit.y_resid.len(),
            fit.a_vec.len(),
            model.score_vec().len()
        ));
    }

    let maybe_ids = if let Some(prefix) = bed_prefix {
        match read_fam(prefix) {
            Ok(ids) => Some(ids),
            Err(err) => {
                eprintln!(
                    "SparseLMM approx sample debug: failed to read FAM IDs from {prefix}.fam ({err}); falling back to row/sample indices."
                );
                None
            }
        }
    } else {
        None
    };

    let out_path = Path::new(&path);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent).map_err(|e| {
                format!(
                    "create approx sample debug parent {}: {e}",
                    parent.display()
                )
            })?;
        }
    }
    let file = File::create(out_path)
        .map_err(|e| format!("create approx sample debug TSV {}: {e}", out_path.display()))?;
    let mut writer = BufWriter::with_capacity(256 * 1024, file);
    writeln!(
        writer,
        "row_order\tsample_idx_full\tiid\ty\ty_resid\ta_vec\ta_resid\ta_adjust\tlambda\tsigma_g2\tsigma_e2\tgamma\tgamma_scale_correction"
    )
    .map_err(|e| format!("write approx sample debug header {}: {e}", out_path.display()))?;
    for (row_order, &sid) in scan_sample_idx.iter().enumerate() {
        let iid = maybe_ids
            .as_ref()
            .and_then(|ids| ids.get(sid))
            .cloned()
            .unwrap_or_else(|| format!("idx_{sid}"));
        let a_resid = model.score_vec()[row_order];
        let a_vec = fit.a_vec[row_order];
        writeln!(
            writer,
            "{}\t{}\t{}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}",
            row_order,
            sid,
            iid,
            y[row_order],
            fit.y_resid[row_order],
            a_vec,
            a_resid,
            a_vec - a_resid,
            fit.lambda,
            fit.sigma_g2,
            fit.sigma_e2,
            model.gamma(),
            fit.gamma_scale_correction,
        )
        .map_err(|e| format!("write approx sample debug row {}: {e}", out_path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("flush approx sample debug TSV {}: {e}", out_path.display()))?;
    eprintln!(
        "SparseLMM approx sample debug: wrote {} rows to {}",
        y.len(),
        out_path.display()
    );
    Ok(())
}

#[inline]
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

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
    let base = (trace / (dim.max(1) as f64)).max(1.0_f64) * 1e-10_f64;
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

fn xtx_chol_from_design(x_design: &[f64], n: usize, p: usize) -> Result<Vec<f64>, String> {
    if x_design.len() != n.saturating_mul(p) {
        return Err(format!(
            "design length mismatch: got {}, expected {}",
            x_design.len(),
            n.saturating_mul(p)
        ));
    }
    let mut xtx = vec![0.0_f64; p * p];
    for i in 0..n {
        let row = &x_design[i * p..(i + 1) * p];
        for a in 0..p {
            for b in 0..p {
                xtx[a * p + b] += row[a] * row[b];
            }
        }
    }
    spd_cholesky_with_jitter(&xtx, p, "SparseLMM approx XtX")
}

fn xt_vec_row_major(x_design: &[f64], n: usize, p: usize, y: &[f64], out: &mut [f64]) {
    out.fill(0.0_f64);
    for i in 0..n {
        let row = &x_design[i * p..(i + 1) * p];
        let yi = y[i];
        for j in 0..p {
            out[j] += row[j] * yi;
        }
    }
}

fn xt_v_inv_x_chol_from_factor(
    factor: &SparseJxgrmCholesky,
    x_design: &[f64],
    n: usize,
    p: usize,
) -> Result<Vec<f64>, String> {
    if x_design.len() != n.saturating_mul(p) {
        return Err(format!(
            "SparseLMM residualized approx design length mismatch: got {}, expected {}",
            x_design.len(),
            n.saturating_mul(p)
        ));
    }
    let mut x_col = vec![0.0_f64; n];
    let mut x_v_inv_cols = vec![0.0_f64; n * p];
    for col in 0..p {
        for i in 0..n {
            x_col[i] = x_design[i * p + col];
        }
        let z = factor.solve_vec(x_col.as_slice())?;
        x_v_inv_cols[col * n..(col + 1) * n].copy_from_slice(z.as_slice());
    }
    let mut xt_v_inv_x = vec![0.0_f64; p * p];
    for a in 0..p {
        for b in 0..p {
            let mut sum = 0.0_f64;
            for i in 0..n {
                sum += x_design[i * p + a] * x_v_inv_cols[b * n + i];
            }
            xt_v_inv_x[a * p + b] = sum;
        }
    }
    spd_cholesky_with_jitter(
        xt_v_inv_x.as_slice(),
        p,
        "SparseLMM residualized approx XtVinvX",
    )
}

fn validate_design_and_response(x_design: &[f64], y: &[f64]) -> Result<(usize, usize), String> {
    let n = y.len();
    if n == 0 {
        return Err("SparseLMM residualized approx requires non-empty response".to_string());
    }
    if x_design.len() % n != 0 {
        return Err(format!(
            "SparseLMM residualized approx design length mismatch: len(x)={} is not divisible by n={n}",
            x_design.len()
        ));
    }
    let p = x_design.len() / n;
    if p == 0 {
        return Err(
            "SparseLMM residualized approx requires at least one design column".to_string(),
        );
    }
    Ok((n, p))
}

fn residualize_vector_with_chol(
    x_design: &[f64],
    xtx_chol: &[f64],
    y: &[f64],
    out: &mut [f64],
    xty: &mut [f64],
    beta: &mut [f64],
) -> Result<(), String> {
    let (n, p) = validate_design_and_response(x_design, y)?;
    if xtx_chol.len() != p * p {
        return Err(format!(
            "SparseLMM residualized approx XtX chol length mismatch: got {}, expected {}",
            xtx_chol.len(),
            p * p
        ));
    }
    if out.len() != n || xty.len() != p || beta.len() != p {
        return Err("SparseLMM residualized approx scratch length mismatch".to_string());
    }
    xt_vec_row_major(x_design, n, p, y, xty);
    cholesky_solve_into(xtx_chol, p, xty, beta);
    out.copy_from_slice(y);
    for i in 0..n {
        let row = &x_design[i * p..(i + 1) * p];
        let fitted = row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum::<f64>();
        out[i] -= fitted;
    }
    Ok(())
}

fn residualized_sumsq_from_xtx_chol(
    xtx_chol: &[f64],
    p: usize,
    xtg: &[f64],
    alpha: &mut [f64],
    g_sq: f64,
) -> f64 {
    cholesky_solve_into(xtx_chol, p, xtg, alpha);
    let quad = xtg
        .iter()
        .zip(alpha.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    (g_sq - quad).max(0.0_f64)
}

pub(crate) fn residualize_response(
    x_design: &[f64],
    y: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (n, p) = validate_design_and_response(x_design, y)?;
    if n <= p {
        return Err(format!(
            "SparseLMM residualized approx requires n > p, got n={n}, p={p}"
        ));
    }
    let xtx_chol = xtx_chol_from_design(x_design, n, p)?;
    let mut y_resid = vec![0.0_f64; n];
    let mut xty = vec![0.0_f64; p];
    let mut beta = vec![0.0_f64; p];
    residualize_vector_with_chol(
        x_design,
        xtx_chol.as_slice(),
        y,
        &mut y_resid,
        &mut xty,
        &mut beta,
    )?;
    Ok((y_resid, xtx_chol))
}

pub(crate) fn fit_sparse_reml_on_residualized_response(
    analysis: &SparseJxgrmCholeskyAnalysis,
    x_design: &[f64],
    y: &[f64],
    low: f64,
    high: f64,
    grid_size: usize,
    tol: f64,
    max_iter: usize,
    threads: usize,
) -> Result<ResidualizedApproxNullFit, String> {
    if !low.is_finite() || !high.is_finite() || low >= high {
        return Err(format!(
            "SparseLMM residualized approx requires finite low < high, got low={low}, high={high}"
        ));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(format!(
            "SparseLMM residualized approx Brent tol must be finite and > 0, got {tol}"
        ));
    }
    if max_iter == 0 {
        return Err("SparseLMM residualized approx Brent max_iter must be > 0".to_string());
    }
    let (n_total, p) = validate_design_and_response(x_design, y)?;
    let n_eff = n_total.saturating_sub(p);
    if n_eff == 0 {
        return Err(format!(
            "SparseLMM residualized approx requires n > p, got n={n_total}, p={p}"
        ));
    }
    let (y_resid, _) = residualize_response(x_design, y)?;
    let n = y_resid.len();
    if analysis.dim() != n {
        return Err(format!(
            "SparseLMM residualized approx sparse n mismatch: K n={}, response n={n}",
            analysis.dim()
        ));
    }
    let mut evaluator =
        ResidualizedApproxEvaluator::new(analysis, y_resid.clone(), n_eff, threads)?;
    let grid_n = grid_size.max(2);
    let mut grid = Vec::<ResidualizedApproxEval>::with_capacity(grid_n);
    let mut best = None::<ResidualizedApproxEval>;
    let mut first_err = None::<String>;
    for idx in 0..grid_n {
        let t = if grid_n <= 1 {
            0.0_f64
        } else {
            idx as f64 / (grid_n - 1) as f64
        };
        let log10_lambda = low + (high - low) * t;
        match evaluator.evaluate(analysis, log10_lambda) {
            Ok((eval, _)) => {
                if best
                    .as_ref()
                    .map(|cur| eval.reml > cur.reml)
                    .unwrap_or(true)
                {
                    best = Some(eval.clone());
                }
                grid.push(eval);
            }
            Err(err) => {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }
    }
    let best_grid = best.ok_or_else(|| match first_err {
        Some(err) => format!(
            "SparseLMM residualized approx found no valid lambda in [{low}, {high}]; first failure: {err}"
        ),
        None => format!(
            "SparseLMM residualized approx found no valid lambda in [{low}, {high}]"
        ),
    })?;
    let best_idx = grid
        .iter()
        .position(|eval| eval.log10_lambda == best_grid.log10_lambda)
        .unwrap_or(0usize);
    let brent_low = if best_idx > 0 {
        grid[best_idx - 1].log10_lambda
    } else {
        low
    };
    let brent_high = if best_idx + 1 < grid.len() {
        grid[best_idx + 1].log10_lambda
    } else {
        high
    };
    let (best_log10, _cost) = brent_minimize_with_init(
        |x0| match evaluator.evaluate(analysis, x0) {
            Ok((eval, _)) => -eval.reml,
            Err(_) => 1e300_f64,
        },
        brent_low,
        brent_high,
        tol,
        max_iter,
        Some(best_grid.log10_lambda),
    );
    let (best_eval, _a_vec_unscaled) = evaluator.evaluate(analysis, best_log10)?;
    let factor =
        analysis.factorize_sigma_g2_k_plus_sigma_e2_i(best_eval.sigma_g2, best_eval.sigma_e2)?;
    let a_vec = factor.solve_vec(y_resid.as_slice())?;
    Ok(ResidualizedApproxNullFit {
        y_resid,
        a_vec,
        lambda: best_eval.lambda,
        sigma_g2: best_eval.sigma_g2,
        sigma_e2: best_eval.sigma_e2,
        ml: best_eval.ml,
        reml: best_eval.reml,
        gamma_scale_correction: 1.0_f64,
        factor,
    })
}

pub(crate) fn build_residualized_approx_null_from_components(
    analysis: &SparseJxgrmCholeskyAnalysis,
    x_design: &[f64],
    y: &[f64],
    sigma_g2: f64,
    sigma_e2: f64,
) -> Result<ResidualizedApproxNullFit, String> {
    if !(sigma_g2.is_finite() && sigma_g2 > 0.0) {
        return Err(format!(
            "SparseLMM residualized approx requires finite sigma_g2 > 0, got {sigma_g2}"
        ));
    }
    if !(sigma_e2.is_finite() && sigma_e2 >= 0.0) {
        return Err(format!(
            "SparseLMM residualized approx requires finite sigma_e2 >= 0, got {sigma_e2}"
        ));
    }
    let (y_resid, _) = residualize_response(x_design, y)?;
    build_residualized_approx_null_from_residualized_response(analysis, y_resid, sigma_g2, sigma_e2)
}

fn residualized_scan_sigma2_from_lambda(
    x_design: &[f64],
    y: &[f64],
    lambda: f64,
) -> Result<(Vec<f64>, f64), String> {
    if !(lambda.is_finite() && lambda >= 0.0) {
        return Err(format!(
            "SparseLMM residualized approx requires finite lambda >= 0, got {lambda}"
        ));
    }
    let (n, p) = validate_design_and_response(x_design, y)?;
    if n <= p {
        return Err(format!(
            "SparseLMM residualized approx requires n > p, got n={n}, p={p}"
        ));
    }
    let df = (n - p) as f64;
    let (y_resid, _) = residualize_response(x_design, y)?;
    let rss_mx = dot_f64(y_resid.as_slice(), y_resid.as_slice());
    if !(rss_mx.is_finite() && rss_mx > SPLMM_APPROX_TINY) {
        return Err(format!(
            "SparseLMM residualized approx produced invalid residualized RSS: {rss_mx}"
        ));
    }
    let sigma2_scan = rss_mx / (df * (1.0_f64 + lambda));
    if !(sigma2_scan.is_finite() && sigma2_scan > 0.0) {
        return Err(format!(
            "SparseLMM residualized approx produced invalid scan sigma2 at lambda={lambda}: {sigma2_scan}"
        ));
    }
    Ok((y_resid, sigma2_scan))
}

fn build_residualized_approx_null_from_residualized_response(
    analysis: &SparseJxgrmCholeskyAnalysis,
    y_resid: Vec<f64>,
    sigma_g2: f64,
    sigma_e2: f64,
) -> Result<ResidualizedApproxNullFit, String> {
    let n = y_resid.len();
    if analysis.dim() != n {
        return Err(format!(
            "SparseLMM residualized approx sparse n mismatch: K n={}, response n={n}",
            analysis.dim()
        ));
    }
    let factor = analysis.factorize_sigma_g2_k_plus_sigma_e2_i(sigma_g2, sigma_e2)?;
    let a_vec = factor.solve_vec(y_resid.as_slice())?;
    Ok(ResidualizedApproxNullFit {
        y_resid,
        a_vec,
        lambda: sigma_e2 / sigma_g2,
        sigma_g2,
        sigma_e2,
        ml: f64::NAN,
        reml: f64::NAN,
        gamma_scale_correction: 1.0_f64,
        factor,
    })
}

pub(crate) fn build_residualized_approx_scan_null_from_lambda_and_factor(
    factor: SparseJxgrmCholesky,
    x_design: &[f64],
    y: &[f64],
    lambda: f64,
) -> Result<ResidualizedApproxNullFit, String> {
    let (y_resid, sigma2_scan) = residualized_scan_sigma2_from_lambda(x_design, y, lambda)?;
    let mut a_vec = factor.solve_vec(y_resid.as_slice())?;
    if sigma2_scan != 1.0_f64 {
        let inv_sigma2 = 1.0_f64 / sigma2_scan;
        for value in a_vec.iter_mut() {
            *value *= inv_sigma2;
        }
    }
    Ok(ResidualizedApproxNullFit {
        y_resid,
        a_vec,
        lambda,
        sigma_g2: sigma2_scan,
        sigma_e2: lambda * sigma2_scan,
        ml: f64::NAN,
        reml: f64::NAN,
        gamma_scale_correction: 1.0_f64 / sigma2_scan,
        factor,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_residualized_approx_scan_sparse(
    factor: SparseJxgrmCholesky,
    scan_prepared: SplmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    scan_sample_idx: Vec<usize>,
    gm: PackedGeneticModel,
    lbd: f64,
    threads: usize,
    block_rows: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    approx_fit_tol: f64,
    approx_fit_max_iter: usize,
    mmap_window_mb: Option<usize>,
) -> Result<SplmmScanResult, String> {
    let _ = (approx_fit_tol, approx_fit_max_iter);
    let stage1_cb = stage1_progress_callback.as_ref();
    let n = y_vec.len();
    let p = x_design.len() / n;
    let factor_nnz = factor.factor_nnz();
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 5, 0, 1)?;
    }
    let fit = build_residualized_approx_scan_null_from_lambda_and_factor(
        factor,
        x_design.as_slice(),
        y_vec.as_slice(),
        lbd,
    )?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 5, 1, 1)?;
    }
    let rhat_rows = choose_rhat_rows(scan_prepared.n_rows(), rhat_markers, rhat_seed);
    let n_rhat = rhat_rows.len();
    let rhat_progress_total = n_rhat_progress_total(scan_prepared.n_rows(), rhat_markers);
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 8, 0, rhat_progress_total)?;
    }
    let sampled_markers = decode_rhat_markers_col_major(
        &scan_prepared,
        scan_sample_idx.as_slice(),
        gm,
        rhat_rows.as_slice(),
        stage1_cb,
        8,
        rhat_progress_total,
        mmap_window_mb,
    )?;
    let (gamma, n_used) = fit.estimate_gamma_from_markers(
        x_design.as_slice(),
        sampled_markers.as_slice(),
        n_rhat,
        rhat_markers,
    )?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 8, rhat_progress_total, rhat_progress_total)?;
    }
    let model = fit.build_scan_model(x_design.as_slice(), gamma)?;
    maybe_write_residualized_approx_sample_debug_tsv(
        scan_prepared.bed_prefix(),
        scan_sample_idx.as_slice(),
        y_vec.as_slice(),
        &fit,
        &model,
    )?;
    let out = scan_with_py_and_rhat(
        &scan_prepared,
        x_design.as_slice(),
        model.score_vec(),
        model.xtx_chol(),
        scan_sample_idx.as_slice(),
        gm,
        model.gamma(),
        threads,
        block_rows,
        scan_progress_callback.as_ref(),
        progress_every,
        9,
        0,
        0,
        mmap_window_mb,
    )?;
    Ok((
        gamma,
        out,
        trivial_pcg_null_info(n, p),
        trivial_pcg_rhat_info(n, n_rhat, n_used),
        factor_nnz,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_residualized_approx_scan_to_tsv_sparse(
    factor: SparseJxgrmCholesky,
    scan_prepared: SplmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    scan_sample_idx: Vec<usize>,
    gm: PackedGeneticModel,
    lbd: f64,
    threads: usize,
    block_rows: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    meta_input: SplmmTsvMetaInput,
    out_tsv: String,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    approx_fit_tol: f64,
    approx_fit_max_iter: usize,
    factor_load_secs: f64,
    mmap_window_mb: Option<usize>,
) -> Result<SplmmScanToTsvResult, String> {
    let _ = (approx_fit_tol, approx_fit_max_iter);
    let stage1_cb = stage1_progress_callback.as_ref();
    let n = y_vec.len();
    let p = x_design.len() / n;
    let prepare_t0 = Instant::now();
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 5, 0, 1)?;
    }
    let fit = build_residualized_approx_scan_null_from_lambda_and_factor(
        factor,
        x_design.as_slice(),
        y_vec.as_slice(),
        lbd,
    )?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 5, 1, 1)?;
    }
    let rhat_rows = choose_rhat_rows(scan_prepared.n_rows(), rhat_markers, rhat_seed);
    let n_rhat = rhat_rows.len();
    let rhat_progress_total = n_rhat_progress_total(scan_prepared.n_rows(), rhat_markers);
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 8, 0, rhat_progress_total)?;
    }
    let sampled_markers = decode_rhat_markers_col_major(
        &scan_prepared,
        scan_sample_idx.as_slice(),
        gm,
        rhat_rows.as_slice(),
        stage1_cb,
        8,
        rhat_progress_total,
        mmap_window_mb,
    )?;
    let (gamma, n_used) = fit.estimate_gamma_from_markers(
        x_design.as_slice(),
        sampled_markers.as_slice(),
        n_rhat,
        rhat_markers,
    )?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 8, rhat_progress_total, rhat_progress_total)?;
    }
    let model = fit.build_scan_model(x_design.as_slice(), gamma)?;
    maybe_write_residualized_approx_sample_debug_tsv(
        scan_prepared.bed_prefix(),
        scan_sample_idx.as_slice(),
        y_vec.as_slice(),
        &fit,
        &model,
    )?;
    let scan_prepare_secs = prepare_t0.elapsed().as_secs_f64();
    let scan_exec_t0 = Instant::now();
    let (written_rows, tsv_timing) = scan_to_tsv_with_py_and_rhat(
        &scan_prepared,
        x_design.as_slice(),
        model.score_vec(),
        model.xtx_chol(),
        scan_sample_idx.as_slice(),
        gm,
        model.gamma(),
        threads,
        block_rows,
        meta_input,
        &out_tsv,
        scan_progress_callback.as_ref(),
        progress_every,
        9,
        mmap_window_mb,
    )?;
    let scan_exec_secs = (scan_exec_t0.elapsed().as_secs_f64() - tsv_timing.finish_secs).max(0.0);
    if splmm_top_level_timing_enabled() {
        emit_splmm_null_scan_core_timing(
            "approx",
            factor_load_secs,
            scan_prepare_secs,
            scan_exec_secs,
            tsv_timing.finish_secs,
            scan_prepared.n_rows(),
            n,
            p,
            threads.max(1),
        );
    }
    Ok(SplmmScanToTsvResult {
        r_hat: gamma,
        written_rows,
        factor_nnz: fit.factor_nnz(),
        null_info: trivial_pcg_null_info(n, p),
        rhat_info: trivial_pcg_rhat_info(n, n_rhat, n_used),
        factor_load_secs,
        scan_prepare_secs,
        scan_exec_secs,
        writer_wait_secs: tsv_timing.finish_secs,
    })
}

impl ResidualizedApproxNullFit {
    #[inline]
    pub(crate) fn factor_nnz(&self) -> usize {
        self.factor.factor_nnz()
    }

    pub(crate) fn estimate_gamma_from_markers(
        &self,
        x_design: &[f64],
        markers_col_major: &[f64],
        n_markers: usize,
        soft_cap_markers: usize,
    ) -> Result<(f64, usize), String> {
        let (n, p) = validate_design_and_response(x_design, self.y_resid.as_slice())?;
        if markers_col_major.len() != n.saturating_mul(n_markers) {
            return Err(format!(
                "SparseLMM residualized approx marker matrix length mismatch: got {}, expected {}",
                markers_col_major.len(),
                n.saturating_mul(n_markers)
            ));
        }
        let xtx_chol = xtx_chol_from_design(x_design, n, p)?;
        let xt_v_inv_x_chol = xt_v_inv_x_chol_from_factor(&self.factor, x_design, n, p)?;
        let gamma_debug = std::env::var("JX_SPLMM_APPROX_GAMMA_DEBUG")
            .map(|v| {
                let s = v.trim().to_ascii_lowercase();
                !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
            })
            .unwrap_or(false);
        let mut xtg = vec![0.0_f64; p];
        let mut xtz = vec![0.0_f64; p];
        let mut alpha = vec![0.0_f64; p];
        let mut g_resid = vec![0.0_f64; n];
        let mut fastgwa_ratio_sum = 0.0_f64;
        let mut residualized_ratio_sum = 0.0_f64;
        let mut schur_sum = 0.0_f64;
        let mut s_ms_sum = 0.0_f64;
        let mut exact_ratio_sum = 0.0_f64;
        let mut residualized_n_used = 0usize;
        let mut n_used = 0usize;
        for marker_idx in 0..n_markers {
            let g = &markers_col_major[marker_idx * n..(marker_idx + 1) * n];
            let g_sq = dot_f64(g, g);
            residualize_vector_with_chol(
                x_design,
                xtx_chol.as_slice(),
                g,
                &mut g_resid,
                &mut xtg,
                &mut alpha,
            )?;
            let s_ms = dot_f64(g_resid.as_slice(), g_resid.as_slice());
            if splmm_residual_sumsq_is_effectively_zero(s_ms, g_sq) {
                continue;
            }
            let residualized_v_inv_g = self.factor.solve_vec(g_resid.as_slice())?;
            let residualized_s_v_s = dot_f64(g_resid.as_slice(), residualized_v_inv_g.as_slice());
            if !residualized_s_v_s.is_finite() || residualized_s_v_s <= SPLMM_APPROX_TINY {
                continue;
            }
            let residualized_ratio = residualized_s_v_s / s_ms;
            if !residualized_ratio.is_finite() || residualized_ratio <= 0.0 {
                continue;
            }
            residualized_ratio_sum += residualized_ratio;
            residualized_n_used += 1;

            let score = dot_f64(g_resid.as_slice(), self.a_vec.as_slice());
            let chisq = (score * score) / residualized_s_v_s;
            if chisq.is_finite() && chisq < SPLMM_APPROX_FASTGWA_NULL_CHISQ_MAX {
                fastgwa_ratio_sum += residualized_ratio;
                n_used += 1;
            }

            if gamma_debug {
                let v_inv_g = self.factor.solve_vec(g)?;
                let s_v_s = dot_f64(g, v_inv_g.as_slice());
                if s_v_s.is_finite() && s_v_s > SPLMM_APPROX_TINY {
                    xt_vec_row_major(x_design, n, p, v_inv_g.as_slice(), &mut xtz);
                    cholesky_solve_into(xt_v_inv_x_chol.as_slice(), p, xtz.as_slice(), &mut alpha);
                    let x_quad = xtz
                        .iter()
                        .zip(alpha.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    let schur = (s_v_s - x_quad).max(0.0_f64);
                    if schur.is_finite() && schur > SPLMM_APPROX_TINY {
                        exact_ratio_sum += schur / s_ms;
                        schur_sum += schur;
                        s_ms_sum += s_ms;
                    }
                }
            }
            if marker_idx + 1 >= soft_cap_markers && n_used >= 100 {
                break;
            }
        }
        if n_used == 0 && residualized_n_used == 0 {
            return Err(
                "SparseLMM residualized approx gamma estimation found no valid sampled markers"
                    .to_string(),
            );
        }
        let use_fastgwa_nulls = n_used >= 100;
        let gamma_mean = if use_fastgwa_nulls {
            fastgwa_ratio_sum / (n_used as f64)
        } else {
            residualized_ratio_sum / (residualized_n_used as f64)
        };
        let gamma_n_used = if use_fastgwa_nulls {
            n_used
        } else {
            residualized_n_used
        };
        if gamma_debug {
            let exact_ratio_of_sums = if s_ms_sum > SPLMM_APPROX_TINY {
                schur_sum / s_ms_sum
            } else {
                f64::NAN
            };
            let exact_mean = if residualized_n_used > 0 {
                exact_ratio_sum / (residualized_n_used as f64)
            } else {
                f64::NAN
            };
            let residualized_mean = if residualized_n_used > 0 {
                residualized_ratio_sum / (residualized_n_used as f64)
            } else {
                f64::NAN
            };
            let fastgwa_mean = if n_used > 0 {
                fastgwa_ratio_sum / (n_used as f64)
            } else {
                f64::NAN
            };
            eprintln!(
                "SparseLMM approx gamma debug: exact_mean={:.9e}, exact_ratio_of_sums={:.9e}, residualized_mean={:.9e}, fastgwa_null_mean={:.9e}, scale_correction={:.9e}, exact_mean_scaled={:.9e}, exact_ratio_of_sums_scaled={:.9e}, residualized_mean_scaled={:.9e}, fastgwa_null_mean_scaled={:.9e}, residualized_used={}, fastgwa_null_used={}, selected={}, selected_used={}",
                exact_mean,
                exact_ratio_of_sums,
                residualized_mean,
                fastgwa_mean,
                self.gamma_scale_correction,
                exact_mean * self.gamma_scale_correction,
                exact_ratio_of_sums * self.gamma_scale_correction,
                residualized_mean * self.gamma_scale_correction,
                fastgwa_mean * self.gamma_scale_correction,
                residualized_n_used,
                n_used,
                if use_fastgwa_nulls { "fastgwa_null" } else { "residualized_all_fallback" },
                gamma_n_used,
            );
        }
        Ok((gamma_mean * self.gamma_scale_correction, gamma_n_used))
    }

    pub(crate) fn build_scan_model(
        &self,
        x_design: &[f64],
        gamma: f64,
    ) -> Result<ResidualizedApproxScanModel, String> {
        ResidualizedApproxScanModel::from_a_vec(x_design, self.a_vec.as_slice(), gamma)
    }
}

impl ResidualizedApproxScanModel {
    pub(crate) fn from_a_vec(x_design: &[f64], a_vec: &[f64], gamma: f64) -> Result<Self, String> {
        let (n, p) = validate_design_and_response(x_design, a_vec)?;
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(format!(
                "SparseLMM residualized approx gamma must be finite and > 0, got {gamma}"
            ));
        }
        let xtx_chol = xtx_chol_from_design(x_design, n, p)?;
        let mut xta = vec![0.0_f64; p];
        let mut score_adjust = vec![0.0_f64; p];
        xt_vec_row_major(x_design, n, p, a_vec, &mut xta);
        cholesky_solve_into(
            xtx_chol.as_slice(),
            p,
            xta.as_slice(),
            score_adjust.as_mut_slice(),
        );
        let mut a_resid = a_vec.to_vec();
        for i in 0..n {
            let row = &x_design[i * p..(i + 1) * p];
            let fitted = row
                .iter()
                .zip(score_adjust.iter())
                .map(|(x, b)| x * b)
                .sum::<f64>();
            a_resid[i] -= fitted;
        }
        Ok(Self {
            x_design: x_design.to_vec(),
            xtx_chol,
            a_resid,
            gamma,
            n,
            p,
        })
    }

    #[inline]
    pub(crate) fn score_vec(&self) -> &[f64] {
        self.a_resid.as_slice()
    }

    #[inline]
    pub(crate) fn xtx_chol(&self) -> &[f64] {
        self.xtx_chol.as_slice()
    }

    #[inline]
    pub(crate) fn gamma(&self) -> f64 {
        self.gamma
    }

    pub(crate) fn assoc_from_marker(&self, g: &[f64]) -> Result<(f64, f64, f64), String> {
        if g.len() != self.n {
            return Err(format!(
                "SparseLMM residualized approx marker length mismatch: got {}, expected {}",
                g.len(),
                self.n
            ));
        }
        let mut xtg = vec![0.0_f64; self.p];
        let mut alpha = vec![0.0_f64; self.p];
        xt_vec_row_major(self.x_design.as_slice(), self.n, self.p, g, &mut xtg);
        let score = dot_f64(g, self.a_resid.as_slice());
        let g_sq = dot_f64(g, g);
        let s_ms = residualized_sumsq_from_xtx_chol(
            self.xtx_chol.as_slice(),
            self.p,
            xtg.as_slice(),
            alpha.as_mut_slice(),
            g_sq,
        );
        if splmm_residual_sumsq_is_effectively_zero(s_ms, g_sq) {
            return Ok((f64::NAN, f64::NAN, 1.0_f64));
        }
        let denom = self.gamma * s_ms;
        if !(denom.is_finite() && denom > SPLMM_APPROX_TINY) {
            return Ok((f64::NAN, f64::NAN, 1.0_f64));
        }
        let beta = score / denom;
        let se = 1.0_f64 / denom.sqrt();
        let chisq = (score * score) / denom;
        if !(beta.is_finite() && se.is_finite() && se > 0.0 && chisq.is_finite() && chisq >= 0.0) {
            return Ok((f64::NAN, f64::NAN, 1.0_f64));
        }
        Ok((beta, se, chi2_sf_df1(chisq)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cholesky::sparse_cholesky_analyze_jxgrm_csc;
    use crate::spgrm::SparseGrmCsc;
    use std::fs;

    fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= tol,
                "mismatch at index {idx}: lhs={a:.12e}, rhs={b:.12e}, diff={diff:.12e}, tol={tol:.12e}"
            );
        }
    }

    fn make_diag_analysis(diag: &[f64]) -> SparseJxgrmCholeskyAnalysis {
        let n = diag.len();
        let csc = SparseGrmCsc {
            n_samples: n,
            nnz: n,
            col_ptr: (0..=n).map(|v| v as u64).collect(),
            row_indices: (0..n).map(|v| v as u32).collect(),
            values: diag.to_vec(),
        };
        sparse_cholesky_analyze_jxgrm_csc(&csc).unwrap()
    }

    #[test]
    fn residualize_response_matches_manual_projection() {
        let x_design = vec![
            1.0_f64, 0.0_f64, //
            1.0_f64, 1.0_f64, //
            1.0_f64, 2.0_f64,
        ];
        let y = vec![2.0_f64, 1.0_f64, 3.0_f64];
        let (y_resid, _xtx_chol) = residualize_response(&x_design, &y).unwrap();
        let expected = vec![1.0_f64 / 6.0_f64, -1.0_f64 / 3.0_f64, 1.0_f64 / 6.0_f64];
        assert_close(&y_resid, &expected, 1e-12_f64);
    }

    #[test]
    fn residualized_scan_model_matches_explicit_centered_marker_math() {
        let x_design = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let a_vec = vec![1.0_f64, -2.0_f64, 1.0_f64];
        let gamma = 0.4_f64;
        let g = vec![0.0_f64, 1.0_f64, 2.0_f64];
        let model = ResidualizedApproxScanModel::from_a_vec(&x_design, &a_vec, gamma).unwrap();
        let (beta, se, pwald) = model.assoc_from_marker(&g).unwrap();

        let g_centered = vec![-1.0_f64, 0.0_f64, 1.0_f64];
        let score = dot_f64(g_centered.as_slice(), a_vec.as_slice());
        let denom = gamma * dot_f64(g_centered.as_slice(), g_centered.as_slice());
        let beta_expected = score / denom;
        let se_expected = 1.0_f64 / denom.sqrt();
        let p_expected = chi2_sf_df1((score * score) / denom);

        assert!((beta - beta_expected).abs() <= 1e-12_f64);
        assert!((se - se_expected).abs() <= 1e-12_f64);
        assert!((pwald - p_expected).abs() <= 1e-12_f64);
    }

    #[test]
    fn residualized_scan_model_marks_constant_marker_as_nan() {
        let x_design = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let a_vec = vec![1.0_f64, -1.0_f64, 0.5_f64, -0.5_f64];
        let gamma = 0.4_f64;
        let g = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let model = ResidualizedApproxScanModel::from_a_vec(&x_design, &a_vec, gamma).unwrap();
        let (beta, se, pwald) = model.assoc_from_marker(&g).unwrap();

        assert!(beta.is_nan());
        assert!(se.is_nan());
        assert_eq!(pwald, 1.0_f64);
    }

    #[test]
    fn residualized_sparse_reml_fit_returns_finite_components() {
        let analysis = make_diag_analysis(&[1.0_f64, 2.0_f64, 4.0_f64]);
        let x_design = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let y = vec![1.0_f64, -0.5_f64, 0.25_f64];
        let fit = fit_sparse_reml_on_residualized_response(
            &analysis, &x_design, &y, -3.0_f64, 3.0_f64, 7, 1e-3_f64, 12, 1,
        )
        .unwrap();
        assert!(fit.lambda.is_finite() && fit.lambda > 0.0);
        assert!(fit.sigma_g2.is_finite() && fit.sigma_g2 > 0.0);
        assert!(fit.sigma_e2.is_finite() && fit.sigma_e2 > 0.0);
        assert_eq!(fit.y_resid.len(), 3);
        assert_eq!(fit.a_vec.len(), 3);
        assert!(fit.reml.is_finite());
        assert!(fit.ml.is_finite());
        let expected_a = fit.factor.solve_vec(fit.y_resid.as_slice()).unwrap();
        assert_close(&fit.a_vec, &expected_a, 1e-12_f64);
    }

    #[test]
    fn residualized_null_from_components_reuses_supplied_variance_components() {
        let analysis = make_diag_analysis(&[1.0_f64, 2.0_f64, 4.0_f64]);
        let x_design = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let y = vec![1.0_f64, -0.5_f64, 0.25_f64];
        let fit = build_residualized_approx_null_from_components(
            &analysis, &x_design, &y, 2.5_f64, 1.25_f64,
        )
        .unwrap();
        assert!((fit.sigma_g2 - 2.5_f64).abs() <= 1e-12_f64);
        assert!((fit.sigma_e2 - 1.25_f64).abs() <= 1e-12_f64);
        assert!((fit.lambda - 0.5_f64).abs() <= 1e-12_f64);
        let expected_a = fit.factor.solve_vec(fit.y_resid.as_slice()).unwrap();
        assert_close(&fit.a_vec, &expected_a, 1e-12_f64);
    }

    #[test]
    fn lambda_only_scan_null_matches_component_scaled_gamma_and_assoc() {
        let analysis = make_diag_analysis(&[1.0_f64, 1.5_f64, 2.0_f64, 3.0_f64]);
        let x_design = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let y = vec![1.0_f64, -0.25_f64, 0.75_f64, 1.5_f64];
        let lambda = 0.8_f64;
        let (_y_resid, sigma2_scan) =
            residualized_scan_sigma2_from_lambda(&x_design, &y, lambda).unwrap();

        let component_fit = build_residualized_approx_null_from_components(
            &analysis,
            &x_design,
            &y,
            sigma2_scan,
            lambda * sigma2_scan,
        )
        .unwrap();

        let lambda_fit = build_residualized_approx_scan_null_from_lambda_and_factor(
            analysis.factorize_k_plus_lambda_i(lambda).unwrap(),
            &x_design,
            &y,
            lambda,
        )
        .unwrap();

        assert_close(
            component_fit.a_vec.as_slice(),
            lambda_fit.a_vec.as_slice(),
            1e-12_f64,
        );

        let sampled_markers = vec![
            0.0_f64, 1.0_f64, 2.0_f64, 1.0_f64, //
            2.0_f64, 0.0_f64, 1.0_f64, 1.0_f64,
        ];
        let (gamma_component, used_component) = component_fit
            .estimate_gamma_from_markers(&x_design, sampled_markers.as_slice(), 2, 2)
            .unwrap();
        let (gamma_lambda, used_lambda) = lambda_fit
            .estimate_gamma_from_markers(&x_design, sampled_markers.as_slice(), 2, 2)
            .unwrap();
        assert_eq!(used_component, used_lambda);
        assert!((gamma_component - gamma_lambda).abs() <= 1e-12_f64);

        let test_marker = vec![1.0_f64, 2.0_f64, 0.0_f64, 1.0_f64];
        let assoc_component = component_fit
            .build_scan_model(&x_design, gamma_component)
            .unwrap()
            .assoc_from_marker(test_marker.as_slice())
            .unwrap();
        let assoc_lambda = lambda_fit
            .build_scan_model(&x_design, gamma_lambda)
            .unwrap()
            .assoc_from_marker(test_marker.as_slice())
            .unwrap();
        assert!((assoc_component.0 - assoc_lambda.0).abs() <= 1e-12_f64);
        assert!((assoc_component.1 - assoc_lambda.1).abs() <= 1e-12_f64);
        assert!((assoc_component.2 - assoc_lambda.2).abs() <= 1e-12_f64);
    }

    #[test]
    fn sampled_gamma_matches_exact_schur_reference_for_intercept_only_design() {
        let k_diag = [1.0_f64, 1.5_f64, 2.0_f64, 3.0_f64];
        let analysis = make_diag_analysis(&k_diag);
        let x_design = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let y = vec![1.0_f64, -0.25_f64, 0.75_f64, 1.5_f64];
        let sigma_g2 = 2.5_f64;
        let sigma_e2 = 1.25_f64;
        let fit = build_residualized_approx_null_from_components(
            &analysis, &x_design, &y, sigma_g2, sigma_e2,
        )
        .unwrap();
        let sampled_markers = vec![
            0.0_f64, 1.0_f64, 2.0_f64, 1.0_f64, //
            2.0_f64, 0.0_f64, 1.0_f64, 1.0_f64,
        ];
        let (gamma, used) = fit
            .estimate_gamma_from_markers(&x_design, sampled_markers.as_slice(), 2, 2)
            .unwrap();
        assert_eq!(used, 2);

        let v_inv = k_diag
            .iter()
            .map(|k| 1.0_f64 / (sigma_g2 * k + sigma_e2))
            .collect::<Vec<_>>();
        let x_v_inv_x = v_inv.iter().sum::<f64>();
        let mut ratio_sum = 0.0_f64;
        for marker_idx in 0..2 {
            let g = &sampled_markers[marker_idx * 4..(marker_idx + 1) * 4];
            let mean = g.iter().sum::<f64>() / 4.0_f64;
            let s_ms = g.iter().map(|gi| (gi - mean) * (gi - mean)).sum::<f64>();
            let s_v_s = g
                .iter()
                .zip(v_inv.iter())
                .map(|(gi, wi)| gi * gi * wi)
                .sum::<f64>();
            let xtz = g
                .iter()
                .zip(v_inv.iter())
                .map(|(gi, wi)| gi * wi)
                .sum::<f64>();
            let schur = s_v_s - (xtz * xtz) / x_v_inv_x;
            ratio_sum += schur / s_ms;
        }
        let gamma_ref = ratio_sum / 2.0_f64;
        assert!((gamma - gamma_ref).abs() <= 1e-12_f64);
    }

    #[test]
    fn approx_sample_debug_tsv_writes_expected_columns() {
        let analysis = make_diag_analysis(&[1.0_f64, 1.5_f64, 2.0_f64]);
        let x_design = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let y = vec![1.0_f64, -0.25_f64, 0.75_f64];
        let fit = build_residualized_approx_scan_null_from_lambda_and_factor(
            analysis.factorize_k_plus_lambda_i(0.8_f64).unwrap(),
            &x_design,
            &y,
            0.8_f64,
        )
        .unwrap();
        let model = fit.build_scan_model(&x_design, 0.5_f64).unwrap();
        let scan_sample_idx = vec![2usize, 0usize, 1usize];
        let out_dir = std::env::temp_dir().join(format!(
            "janusx_splmm_approx_debug_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let out_path = out_dir.join("samples.tsv");
        std::env::set_var(
            SPLMM_APPROX_SAMPLE_DEBUG_PATH_ENV,
            out_path.to_string_lossy().to_string(),
        );
        maybe_write_residualized_approx_sample_debug_tsv(
            None,
            scan_sample_idx.as_slice(),
            y.as_slice(),
            &fit,
            &model,
        )
        .unwrap();
        std::env::remove_var(SPLMM_APPROX_SAMPLE_DEBUG_PATH_ENV);

        let text = fs::read_to_string(&out_path).unwrap();
        let mut lines = text.lines();
        let header = lines.next().unwrap();
        assert_eq!(
            header,
            "row_order\tsample_idx_full\tiid\ty\ty_resid\ta_vec\ta_resid\ta_adjust\tlambda\tsigma_g2\tsigma_e2\tgamma\tgamma_scale_correction"
        );
        let first = lines.next().unwrap();
        assert!(first.starts_with("0\t2\tidx_2\t"));
        assert_eq!(lines.count(), 2);
        fs::remove_file(&out_path).unwrap();
        fs::remove_dir(&out_dir).unwrap();
    }
}
