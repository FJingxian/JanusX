use crate::brent::brent_minimize_with_init;
use crate::cholesky::{SparseJxgrmCholesky, SparseJxgrmCholeskyAnalysis};
use crate::linalg::{chi2_sf_df1, cholesky_inplace, cholesky_solve_into};

const SPLMM_APPROX_TINY: f64 = 1e-30_f64;

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
        factor,
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
        let mut xtg = vec![0.0_f64; p];
        let mut alpha = vec![0.0_f64; p];
        let mut g_resid = vec![0.0_f64; n];
        let mut ratio_sum = 0.0_f64;
        let mut n_used = 0usize;
        for marker_idx in 0..n_markers {
            let g = &markers_col_major[marker_idx * n..(marker_idx + 1) * n];
            residualize_vector_with_chol(
                x_design,
                xtx_chol.as_slice(),
                g,
                &mut g_resid,
                &mut xtg,
                &mut alpha,
            )?;
            let s_ms = dot_f64(g_resid.as_slice(), g_resid.as_slice());
            if !s_ms.is_finite() || s_ms <= SPLMM_APPROX_TINY {
                continue;
            }
            let v_inv_g = self.factor.solve_vec(g_resid.as_slice())?;
            let s_v_s = dot_f64(g_resid.as_slice(), v_inv_g.as_slice());
            if !s_v_s.is_finite() || s_v_s <= SPLMM_APPROX_TINY {
                continue;
            }
            let ratio = s_v_s / s_ms;
            if ratio.is_finite() && ratio > 0.0 {
                ratio_sum += ratio;
                n_used += 1;
            }
        }
        if n_used == 0 {
            return Err(
                "SparseLMM residualized approx gamma estimation found no valid sampled markers"
                    .to_string(),
            );
        }
        Ok((ratio_sum / (n_used as f64), n_used))
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
}
