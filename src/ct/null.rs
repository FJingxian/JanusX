use crate::ct::conditional::{
    centered_design_columns, orthonormal_basis, project_off, CovarianceMetric,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModelKind {
    Lm,
    FvLmm,
}

#[derive(Clone, Debug)]
pub(crate) struct GlobalNullFit {
    pub(crate) screen_residuals: Vec<f64>,
    pub(crate) fixed_rank: usize,
    pub(crate) sigma_g2: Option<f64>,
    pub(crate) sigma_e2: Option<f64>,
    pub(crate) no_grm_boundary: bool,
    pub(crate) metric: CovarianceMetric,
    pub(crate) screen_covariate_basis: Vec<Vec<f64>>,
}

pub(crate) fn fit_global_null(
    y: &[f64],
    covariates: &[f64],
    q_cov: usize,
    grm: Option<&[f64]>,
    model: ModelKind,
    threads: usize,
) -> Result<GlobalNullFit, String> {
    let n = y.len();
    if n < 3 {
        return Err(format!("global null requires at least 3 samples, got {n}"));
    }
    if covariates.len() != n.saturating_mul(q_cov) {
        return Err(format!(
            "covariate shape mismatch: got {} values, expected {} x {q_cov}",
            covariates.len(),
            n
        ));
    }
    if y.iter().chain(covariates).any(|value| !value.is_finite()) {
        return Err("global null input contains non-finite values".to_string());
    }

    let mut c = Vec::with_capacity(n * (q_cov + 1));
    for row in 0..n {
        c.push(1.0);
        c.extend_from_slice(&covariates[row * q_cov..(row + 1) * q_cov]);
    }
    let q_c = q_cov + 1;

    match model {
        ModelKind::Lm => {
            if grm.is_some() {
                return Err("LM global null does not accept a GRM".to_string());
            }
            let c_basis = orthonormal_basis(centered_design_columns(&c, n, q_c));
            if n <= c_basis.len() + 1 {
                return Err(format!(
                    "LM fixed-effect rank leaves insufficient residual degrees of freedom: n={n}, rank={}",
                    c_basis.len()
                ));
            }
            let mut residuals = project_off(y, &c_basis);
            standardize_screen_residuals(&mut residuals)?;
            Ok(GlobalNullFit {
                screen_residuals: residuals,
                fixed_rank: c_basis.len(),
                sigma_g2: None,
                sigma_e2: None,
                no_grm_boundary: false,
                metric: CovarianceMetric::identity(n),
                screen_covariate_basis: c_basis,
            })
        }
        ModelKind::FvLmm => {
            let grm = grm.ok_or_else(|| "FvLMM global null requires an aligned GRM".to_string())?;
            if grm.len() != n.saturating_mul(n) {
                return Err(format!(
                    "GRM shape mismatch: got {} values, expected {} x {n}",
                    grm.len(),
                    n
                ));
            }
            let covariates_opt = (q_cov > 0).then_some(covariates);
            let fit = crate::garfield::estimate_lmm_lambda_from_grm_rust(
                grm,
                n,
                y,
                covariates_opt,
                q_cov,
                threads.max(1),
                -6.0,
                6.0,
                120,
                1.0e-7,
                true,
                n,
                false,
            )?;
            let sigma_g2 = fit.sigma_g2;
            let sigma_e2 = fit.sigma_e2;
            let no_grm_boundary = fit.no_grm_boundary;
            let metric = CovarianceMetric::from_components(grm, n, sigma_g2, sigma_e2)?;
            let y_w = metric.whiten(y)?;
            let c_w = centered_design_columns(&c, n, q_c)
                .into_iter()
                .map(|column| metric.whiten(&column))
                .collect::<Result<Vec<_>, _>>()?;
            let c_basis = orthonormal_basis(c_w);
            if n <= c_basis.len() {
                return Err(format!(
                    "FvLMM fixed-effect rank leaves no residual degrees of freedom: n={n}, rank={}",
                    c_basis.len()
                ));
            }
            let y_after_c = project_off(&y_w, &c_basis);
            let mut screen_residuals = metric.solve_lower_transpose(&y_after_c)?;
            for value in screen_residuals.iter_mut() {
                *value *= sigma_e2;
            }
            let screen_covariate_basis = orthonormal_basis(centered_design_columns(&c, n, q_c));
            Ok(GlobalNullFit {
                screen_residuals,
                fixed_rank: c_basis.len(),
                sigma_g2: Some(sigma_g2),
                sigma_e2: Some(sigma_e2),
                no_grm_boundary,
                metric,
                screen_covariate_basis,
            })
        }
    }
}

fn standardize_screen_residuals(values: &mut [f64]) -> Result<(), String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("global null produced empty or non-finite screen residuals".to_string());
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    if !(variance.is_finite() && variance > 0.0) {
        return Err("global null residual variance is zero".to_string());
    }
    let standard_deviation = variance.sqrt();
    for value in values {
        *value = (*value - mean) / standard_deviation;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{fit_global_null, ModelKind};
    use crate::ct::conditional::CovarianceMetric;

    #[test]
    fn lm_global_null_adds_one_intercept_and_returns_fixed_n_residuals() {
        let x = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let covariates = x.to_vec();
        let residual = [0.2, -0.1, 0.3, -0.2, 0.0, 0.15, -0.25, -0.1];
        let y = x
            .iter()
            .zip(residual)
            .map(|(&x_i, e_i)| 1.4 + 0.75 * x_i + e_i)
            .collect::<Vec<_>>();
        let fit = fit_global_null(&y, &covariates, 1, None, ModelKind::Lm, 1).unwrap();
        assert_eq!(fit.screen_residuals.len(), y.len());
        assert_eq!(fit.fixed_rank, 2);
        let mean_resid = fit.screen_residuals.iter().sum::<f64>() / y.len() as f64;
        assert!(mean_resid.abs() < 1e-12);
        let slope_score = x
            .iter()
            .zip(&fit.screen_residuals)
            .map(|(x_i, e_i)| x_i * e_i)
            .sum::<f64>();
        assert!(slope_score.abs() < 1e-12);
        assert_eq!(fit.sigma_g2, None);
        assert_eq!(fit.sigma_e2, None);
        assert!(!fit.no_grm_boundary);
    }

    #[test]
    fn lm_null_is_invariant_to_a_large_covariate_offset() {
        let n = 200;
        let mut centered = Vec::with_capacity(n);
        let mut shifted = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for row in 0..n {
            let x = ((row * 17 % 101) as f64 - 50.0) / 37.0;
            let noise = ((row * 13 % 19) as f64 - 9.0) / 80.0;
            centered.push(x);
            shifted.push(x + 1.0e7);
            y.push(10.0 + 4.0 * x + noise);
        }

        let fit_centered = fit_global_null(&y, &centered, 1, None, ModelKind::Lm, 1).unwrap();
        let fit_shifted = fit_global_null(&y, &shifted, 1, None, ModelKind::Lm, 1).unwrap();

        assert_eq!(fit_centered.fixed_rank, 2);
        assert_eq!(fit_shifted.fixed_rank, fit_centered.fixed_rank);
        for (left, right) in fit_centered
            .screen_residuals
            .iter()
            .zip(&fit_shifted.screen_residuals)
        {
            assert!(
                (left - right).abs() < 1.0e-6,
                "residual difference: {}",
                left - right
            );
        }
    }

    #[test]
    fn lm_null_keeps_small_unit_covariates_and_matches_rescaled_design() {
        let n = 48;
        let mut small_unit = Vec::with_capacity(n * 2);
        let mut unit_scale = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for row in 0..n {
            let x = ((row * 11 % 37) as f64 - 18.0) / 13.0;
            let w = ((row * 7 % 29) as f64 - 14.0) / 9.0;
            let noise = ((row * 5 % 17) as f64 - 8.0) / 100.0;
            small_unit.extend([x, 1.0e-15 * w]);
            unit_scale.extend([x, w]);
            y.push(1.2 + 0.7 * x - 0.4 * w + noise);
        }

        let fit_small = fit_global_null(&y, &small_unit, 2, None, ModelKind::Lm, 1).unwrap();
        let fit_unit = fit_global_null(&y, &unit_scale, 2, None, ModelKind::Lm, 1).unwrap();

        assert_eq!(fit_small.fixed_rank, 3);
        assert_eq!(fit_unit.fixed_rank, fit_small.fixed_rank);
        for (left, right) in fit_small
            .screen_residuals
            .iter()
            .zip(&fit_unit.screen_residuals)
        {
            assert!((left - right).abs() < 1.0e-10);
        }
    }

    #[test]
    fn zero_genetic_variance_uses_sigma_e_squared_identity_covariance() {
        let n = 4;
        let grm = [
            1.0, 0.2, 0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.1, 1.0,
        ];
        let metric = CovarianceMetric::from_components(&grm, n, 0.0, 4.0).unwrap();
        let whitened = metric.whiten(&[2.0, -4.0, 6.0, 8.0]).unwrap();
        assert_eq!(whitened, vec![1.0, -2.0, 3.0, 4.0]);
    }

    #[test]
    fn fvlmm_identity_grm_selects_the_zero_genetic_variance_boundary() {
        let y = [0.3, 1.8, -0.4, 0.9, 2.2, -1.1, 0.7, 1.3];
        let n = y.len();
        let mut grm = vec![0.0; n * n];
        for i in 0..n {
            grm[i * n + i] = 1.0;
        }
        let fit = fit_global_null(&y, &[], 0, Some(&grm), ModelKind::FvLmm, 1).unwrap();
        assert_eq!(fit.sigma_g2, Some(0.0));
        assert!(fit.sigma_e2.unwrap().is_finite() && fit.sigma_e2.unwrap() > 0.0);
        assert!(fit.no_grm_boundary);
        assert_eq!(fit.fixed_rank, 1);
        assert!(fit.screen_residuals.iter().all(|value| value.is_finite()));
        let mean = y.iter().sum::<f64>() / n as f64;
        for (&actual, &observed) in fit.screen_residuals.iter().zip(&y) {
            assert!(
                (actual - (observed - mean)).abs() < 1.0e-12,
                "FvLMM screen residual should retain its score-test scale"
            );
        }
    }

    #[test]
    fn fvlmm_null_variance_is_invariant_to_covariate_offset_and_units() {
        let n = 80;
        let mut centered = Vec::with_capacity(n);
        let mut shifted = Vec::with_capacity(n);
        let mut tiny = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for row in 0..n {
            let x = ((row * 17 % 101) as f64 - 50.0) / 37.0;
            let noise = ((row * 13 % 19) as f64 - 9.0) / 80.0;
            centered.push(x);
            shifted.push(x + 1.0e7);
            tiny.push(x * 1.0e-15);
            y.push(10.0 + 4.0 * x + noise);
        }
        let mut grm = vec![0.0; n * n];
        for i in 0..n {
            grm[i * n + i] = 1.0;
        }

        let fit_centered =
            fit_global_null(&y, &centered, 1, Some(&grm), ModelKind::FvLmm, 1).unwrap();
        let fit_shifted =
            fit_global_null(&y, &shifted, 1, Some(&grm), ModelKind::FvLmm, 1).unwrap();
        let fit_tiny = fit_global_null(&y, &tiny, 1, Some(&grm), ModelKind::FvLmm, 1).unwrap();

        assert_eq!(fit_centered.fixed_rank, 2);
        assert_eq!(fit_shifted.fixed_rank, 2);
        assert_eq!(fit_tiny.fixed_rank, 2);
        let expected_sigma_e2 = fit_centered.sigma_e2.unwrap();
        for (name, fit) in [("shifted", fit_shifted), ("tiny", fit_tiny)] {
            let sigma_e2 = fit.sigma_e2.unwrap();
            assert!(
                (expected_sigma_e2 - sigma_e2).abs() < 1.0e-9,
                "sigma_e2 changed for {name} covariate: {expected_sigma_e2} vs {sigma_e2}"
            );
            assert!(fit.no_grm_boundary);
        }
    }

    #[test]
    fn global_null_rejects_sample_and_covariate_shape_mismatches() {
        let y = [1.0, 2.0, 3.0];
        assert!(fit_global_null(&y, &[0.0, 1.0], 1, None, ModelKind::Lm, 1).is_err());
        assert!(fit_global_null(&y, &[0.0, 1.0, 2.0, 3.0], 1, None, ModelKind::Lm, 1).is_err());
    }
}
