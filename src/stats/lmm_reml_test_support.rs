//! Feature-gated helpers for checking the production LMM REML scorer.

use crate::brent::brent_minimize_with_init;
use crate::reml::{final_beta_se, reml_loglike, reml_loglike_with_workspace, RemlWorkspace};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct FitSummary {
    pub theta: f64,
    pub loglike: f64,
    pub beta: f64,
    pub se: f64,
    pub p: f64,
}

pub struct ProductionRemlWorkspace(RemlWorkspace);

impl ProductionRemlWorkspace {
    pub fn new(p_cov: usize) -> Self {
        Self(RemlWorkspace::new(p_cov.saturating_add(1)))
    }
}

pub fn reference_reml_loglike(
    log10_lambda: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: Option<&[f64]>,
    n: usize,
    p_cov: usize,
) -> f64 {
    reml_loglike(log10_lambda, s, xcov, y, snp, n, p_cov)
}

pub fn production_reml_loglike(
    log10_lambda: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: &[f64],
    n: usize,
    p_cov: usize,
    workspace: &mut ProductionRemlWorkspace,
) -> f64 {
    reml_loglike_with_workspace(log10_lambda, s, xcov, y, snp, n, p_cov, &mut workspace.0)
}

pub fn reference_brent_batch(
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snps: &[f64],
    n: usize,
    p_cov: usize,
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
    init_log10_lambda: Option<f64>,
) -> Result<Vec<FitSummary>, &'static str> {
    brent_batch::<false>(
        s,
        xcov,
        y,
        snps,
        n,
        p_cov,
        low,
        high,
        tol,
        max_iter,
        init_log10_lambda,
    )
}

pub fn production_workspace_brent_batch(
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snps: &[f64],
    n: usize,
    p_cov: usize,
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
    init_log10_lambda: Option<f64>,
) -> Result<Vec<FitSummary>, &'static str> {
    brent_batch::<true>(
        s,
        xcov,
        y,
        snps,
        n,
        p_cov,
        low,
        high,
        tol,
        max_iter,
        init_log10_lambda,
    )
}

#[allow(clippy::too_many_arguments)]
fn brent_batch<const USE_WORKSPACE: bool>(
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snps: &[f64],
    n: usize,
    p_cov: usize,
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
    init_log10_lambda: Option<f64>,
) -> Result<Vec<FitSummary>, &'static str> {
    let n_snps = validate_inputs(s, xcov, y, snps, n, p_cov, low, high)?;
    let group_rows = 16usize;
    let group_width = n.saturating_mul(group_rows).max(n);
    let groups: Vec<Vec<FitSummary>> = snps
        .par_chunks(group_width)
        .map_init(
            || {
                if USE_WORKSPACE {
                    Some(RemlWorkspace::new(p_cov + 1))
                } else {
                    None
                }
            },
            |workspace, group| {
                let mut warm = init_log10_lambda;
                group
                    .chunks_exact(n)
                    .map(|snp| {
                        let (theta, cost) = brent_minimize_with_init(
                            |theta| {
                                let loglike = if USE_WORKSPACE {
                                    reml_loglike_with_workspace(
                                        theta,
                                        s,
                                        xcov,
                                        y,
                                        snp,
                                        n,
                                        p_cov,
                                        workspace.as_mut().expect("workspace-enabled batch"),
                                    )
                                } else {
                                    reml_loglike(theta, s, xcov, y, Some(snp), n, p_cov)
                                };
                                -loglike
                            },
                            low,
                            high,
                            tol,
                            max_iter,
                            warm,
                        );
                        let fit = summarize_fit(theta, -cost, s, xcov, y, snp, n, p_cov);
                        warm = Some(theta);
                        fit
                    })
                    .collect()
            },
        )
        .collect();
    debug_assert_eq!(groups.iter().map(Vec::len).sum::<usize>(), n_snps);
    Ok(groups.into_iter().flatten().collect())
}

fn validate_inputs(
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snps: &[f64],
    n: usize,
    p_cov: usize,
    low: f64,
    high: f64,
) -> Result<usize, &'static str> {
    if n == 0
        || s.len() != n
        || y.len() != n
        || xcov.len()
            != n.checked_mul(p_cov)
                .ok_or("covariate dimensions overflow")?
        || snps.len() % n != 0
        || n <= p_cov + 1
        || !low.is_finite()
        || !high.is_finite()
        || low >= high
    {
        return Err("invalid LMM fit dimensions or bounds");
    }
    Ok(snps.len() / n)
}

fn summarize_fit(
    theta: f64,
    loglike: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: &[f64],
    n: usize,
    p_cov: usize,
) -> FitSummary {
    let (beta, se, _) = final_beta_se(theta, s, xcov, y, snp, n, p_cov);
    let p = if beta.is_finite() && se.is_finite() && se > 0.0 {
        (crate::linalg::normal_sf((beta / se).abs()) * 2.0).clamp(f64::MIN_POSITIVE, 1.0)
    } else {
        1.0
    };
    FitSummary {
        theta,
        loglike,
        beta,
        se,
        p,
    }
}
