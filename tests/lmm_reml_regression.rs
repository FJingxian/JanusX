#![cfg(feature = "lmm-reml-regression")]

use janusx::lmm_reml_test_support::{
    production_reml_loglike, production_workspace_brent_batch, reference_brent_batch,
    reference_reml_loglike, FitSummary, ProductionRemlWorkspace,
};

fn fixture(n: usize, p_cov: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let s = (0..n)
        .map(|i| 0.1 + 1.9 * (i as f64 / n.saturating_sub(1).max(1) as f64))
        .collect();
    let y = (0..n)
        .map(|i| (i as f64 * 0.31).sin() + 0.35 * (i as f64 * 0.071).cos())
        .collect();
    let snp = (0..n)
        .map(|i| ((i * 17 + i / 3) % 3) as f64 - 1.0)
        .collect();
    let mut xcov = Vec::with_capacity(n * p_cov);
    for i in 0..n {
        let x = i as f64 / n as f64;
        for col in 0..p_cov {
            let value = match col {
                0 => 1.0,
                1 => x - 0.5,
                _ => {
                    let frequency = (col - 1) as f64;
                    (i as f64 * frequency * std::f64::consts::TAU / n as f64).sin()
                }
            };
            xcov.push(value);
        }
    }
    (s, xcov, y, snp)
}

fn genotype_block(n: usize, rows: usize) -> Vec<f64> {
    (0..rows)
        .flat_map(|marker| {
            (0..n).map(move |i| {
                ((i * (17 + marker % 11) + i / (3 + marker % 5) + marker * 13) % 3) as f64 - 1.0
            })
        })
        .collect()
}

#[test]
fn production_workspace_objective_matches_reference_across_covariate_counts() {
    let n = 64;
    for p_cov in [0, 1, 2, 3, 4, 8, 16, 17, 32] {
        let (s, xcov, y, snp) = fixture(n, p_cov);
        let mut workspace = ProductionRemlWorkspace::new(p_cov);
        for theta in [-4.0, -2.0, -0.5, 1.0, 4.0] {
            let expected = reference_reml_loglike(theta, &s, &xcov, &y, Some(&snp), n, p_cov);
            let observed =
                production_reml_loglike(theta, &s, &xcov, &y, &snp, n, p_cov, &mut workspace);
            let tolerance = 2e-10 * expected.abs().max(1.0);
            assert!(
                (expected - observed).abs() <= tolerance,
                "p_cov={p_cov}, theta={theta}, expected={expected:.16e}, observed={observed:.16e}"
            );
        }
    }
}

#[test]
fn production_workspace_fits_match_reference_across_covariate_counts() {
    let n = 128;
    for p_cov in [0, 1, 2, 3, 4, 8, 16, 17, 32] {
        let (s, xcov, y, _) = fixture(n, p_cov);
        let snps = genotype_block(n, 12);
        let reference = reference_brent_batch(
            &s,
            &xcov,
            &y,
            &snps,
            n,
            p_cov,
            -5.0,
            5.0,
            1e-8,
            200,
            Some(0.0),
        )
        .expect("reference Brent fits");
        let observed = production_workspace_brent_batch(
            &s,
            &xcov,
            &y,
            &snps,
            n,
            p_cov,
            -5.0,
            5.0,
            1e-8,
            200,
            Some(0.0),
        )
        .expect("production workspace Brent fits");
        assert_fit_sets_close(&reference, &observed, 1e-4, 1e-6);
    }
}

#[test]
fn workspace_preserves_rank_failure_for_redundant_covariates() {
    let n = 64;
    let (s, _, _, snp) = fixture(n, 0);
    let xcov: Vec<f64> = (0..n).flat_map(|_| [1.0, 1.0]).collect();
    let y: Vec<f64> = snp.iter().map(|g| 1.75 + 0.4 * g).collect();
    let expected = reference_reml_loglike(-1.0, &s, &xcov, &y, Some(&snp), n, 2);
    let mut workspace = ProductionRemlWorkspace::new(2);
    let observed = production_reml_loglike(-1.0, &s, &xcov, &y, &snp, n, 2, &mut workspace);
    assert_eq!(observed, expected);
}

fn assert_fit_sets_close(
    reference: &[FitSummary],
    observed: &[FitSummary],
    theta_tol: f64,
    statistic_tol: f64,
) {
    assert_eq!(reference.len(), observed.len());
    for (index, (expected, actual)) in reference.iter().zip(observed).enumerate() {
        assert!(
            (expected.theta - actual.theta).abs() <= theta_tol,
            "theta SNP {index}: {} vs {}",
            expected.theta,
            actual.theta
        );
        for (name, left, right) in [
            ("loglike", expected.loglike, actual.loglike),
            ("beta", expected.beta, actual.beta),
            ("SE", expected.se, actual.se),
            ("p", expected.p, actual.p),
        ] {
            assert!(
                (left - right).abs() <= statistic_tol,
                "{name} SNP {index}: {left} vs {right}"
            );
        }
    }
}
