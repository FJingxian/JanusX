//! BayesR marker-effect sampler.
//!
//! The residual inverse-chi-square prior accepts `prior_ss_e = nu_0 * S_0^2`
//! (the prior sum-of-squares term), rather than the scale `S_0^2` alone.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAny, PyDict};
use pyo3::{prelude::*, BoundObject};
use rand::rngs::{OsRng, StdRng};
use rand::{Rng, SeedableRng, TryRngCore};
use rand_distr::{ChiSquared, Gamma, StandardNormal};
use std::borrow::Cow;
use std::sync::Arc;

use crate::bayes::{
    aggregate_bayes_h2, array2_to_f32_input, bayes_chain_for_each_mut, bayes_chain_seeds,
    bayes_chain_start_scale, bayes_chain_try_for_each_mut, bayes_chain_try_map_mut,
    bayes_marker_update_f32_result, bayes_packed_blas_threads, bayes_packed_block_rows,
    bayes_predictive_monitor_values, build_bayes_chain_pool, copy_f32_to_f64, copy_f64_to_f32,
    ddot_f64, duplicate_bayes_source, effective_bayes_chains, finite_rhat_max,
    genetic_variance_from_residual, marker_sufficient_stats, rhat_metrics_to_py,
    update_alpha_gauss_seidel_blas, BayesFixedEffectBackend, BayesFixedEffectInit,
    BayesMarkerBackend, BayesMultiChainController, BayesPackedSource, BayesPriorCalibration,
    BayesSamplingController, BayesSamplingOptions, DenseBayesBackend, PackedBayesBackend,
    BAYES_PREDICTIVE_MONITOR_NAMES, BAYES_RHAT_R_NAMES,
};
use crate::blas::OpenBlasThreadGuard;
use crate::stats_common::{get_cached_pool, parse_index_vec_i64_value_error};

const BAYESR_COMPONENTS: usize = 4;

fn validate_bayesr_prior(pi: &[f64], gamma: &[f64]) -> Result<(), String> {
    if pi.len() != BAYESR_COMPONENTS || gamma.len() != BAYESR_COMPONENTS {
        return Err(format!(
            "BayesR requires exactly {BAYESR_COMPONENTS} pi and gamma values"
        ));
    }
    if !pi.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err("BayesR pi values must be finite and > 0".to_string());
    }
    let pi_sum: f64 = pi.iter().sum();
    if !pi_sum.is_finite() || pi_sum <= 0.0 {
        return Err("BayesR pi values must have a positive finite sum".to_string());
    }
    if !gamma[0].is_finite() || gamma[0].abs() > 1e-15 {
        return Err("BayesR gamma[0] must be zero for the spike component".to_string());
    }
    if !gamma[1..].iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err("BayesR non-spike gamma values must be finite and > 0".to_string());
    }
    Ok(())
}

fn bayesr_chain_start_pi(
    pi_base: [f64; BAYESR_COMPONENTS],
    scale: f64,
) -> [f64; BAYESR_COMPONENTS] {
    if scale == 1.0 {
        return pi_base;
    }
    let base_non_spike = 1.0 - pi_base[0];
    let non_spike_odds = (base_non_spike / pi_base[0]) * scale;
    let pi0 = 1.0 / (1.0 + non_spike_odds);
    let non_spike_scale = (1.0 - pi0) / base_non_spike;
    let mut out = pi_base;
    out[0] = pi0;
    for k in 1..BAYESR_COMPONENTS {
        out[k] *= non_spike_scale;
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct BayesRScaleCalibration {
    s0_lambda2: f64,
    initial_sigma_lambda2: f64,
}

/// Resolve the global BayesR marker-variance scale.
///
/// The automatic value targets the marker genetic-variance contribution
/// `Var(y_perp) * r2` under the initial mixture weights.  Because
/// `s0_lambda2` is the scaled-inverse-chi-square scale parameter, its prior
/// mode is `df0_lambda * s0_lambda2 / (df0_lambda + 2)`.  The automatic
/// value is therefore converted so that this mode equals the target marker
/// variance.  Explicit values bypass all automatic calibration.
fn resolve_bayesr_scale(
    prior_calibration: &BayesPriorCalibration,
    msx: f64,
    r2: f64,
    pi_init: &[f64],
    gamma: &[f64],
    df0_lambda: f64,
    explicit_s0_lambda2: Option<f64>,
) -> Result<BayesRScaleCalibration, String> {
    if let Some(value) = explicit_s0_lambda2 {
        if !(value.is_finite() && value > 0.0) {
            return Err("BayesR s0_lambda2 must be finite and > 0".to_string());
        }
        return Ok(BayesRScaleCalibration {
            s0_lambda2: value,
            initial_sigma_lambda2: value,
        });
    }
    if !(msx.is_finite() && msx > 0.0) {
        return Err("BayesR marker mean square must be positive".to_string());
    }
    if !(r2.is_finite() && r2 > 0.0 && r2 < 1.0) {
        return Err("BayesR r2 must be finite and in (0, 1)".to_string());
    }
    if !(df0_lambda.is_finite() && df0_lambda > 0.0) {
        return Err("BayesR df0_lambda must be finite and > 0".to_string());
    }
    validate_bayesr_prior(pi_init, gamma)?;
    let pi_total = pi_init.iter().sum::<f64>();
    let expected_gamma = pi_init
        .iter()
        .zip(gamma.iter())
        .map(|(weight, multiplier)| weight * multiplier)
        .sum::<f64>()
        / pi_total;
    if !(expected_gamma.is_finite() && expected_gamma > 0.0) {
        return Err("BayesR weighted gamma must be finite and > 0".to_string());
    }
    let target_mode = prior_calibration.residualized_var * r2 / (msx * expected_gamma);
    if !(target_mode.is_finite() && target_mode > 0.0) {
        return Err("BayesR automatic marker scale must be finite and > 0".to_string());
    }
    let s0_lambda2 = target_mode * (df0_lambda + 2.0) / df0_lambda;
    if !(s0_lambda2.is_finite() && s0_lambda2 > 0.0) {
        return Err("BayesR automatic s0_lambda2 must be finite and > 0".to_string());
    }
    Ok(BayesRScaleCalibration {
        s0_lambda2,
        initial_sigma_lambda2: target_mode,
    })
}

#[inline]
fn component_log_weights_unchecked(
    log_pi: &[f64; BAYESR_COMPONENTS],
    tau2: &[f64; BAYESR_COMPONENTS],
    u: f64,
    d: f64,
    sigma_e2: f64,
) -> [f64; BAYESR_COMPONENTS] {
    if !(u.is_finite() && d.is_finite() && d >= 0.0) {
        debug_assert!(u.is_finite() && d.is_finite() && d >= 0.0);
    }
    let mut out = [0.0; BAYESR_COMPONENTS];
    out[0] = log_pi[0];
    for k in 1..BAYESR_COMPONENTS {
        let component_tau2 = tau2[k];
        let denom = sigma_e2 + component_tau2 * d;
        out[k] = log_pi[k] - 0.5 * (1.0 + component_tau2 * d / sigma_e2).ln()
            + component_tau2 * u * u / (2.0 * sigma_e2 * denom);
    }
    out
}

#[cfg(test)]
fn component_log_weights(
    pi: &[f64],
    gamma: &[f64],
    u: f64,
    d: f64,
    sigma_lambda2: f64,
    sigma_e2: f64,
) -> Result<[f64; BAYESR_COMPONENTS], String> {
    validate_bayesr_prior(pi, gamma)?;
    if !(u.is_finite() && d.is_finite() && d >= 0.0) {
        return Err("BayesR marker sufficient statistics must be finite and d >= 0".to_string());
    }
    if !(sigma_lambda2.is_finite() && sigma_lambda2 > 0.0) {
        return Err("BayesR sigma_lambda2 must be finite and > 0".to_string());
    }
    if !(sigma_e2.is_finite() && sigma_e2 > 0.0) {
        return Err("BayesR sigma_e2 must be finite and > 0".to_string());
    }
    let mut log_pi = [0.0_f64; BAYESR_COMPONENTS];
    let mut tau2 = [0.0_f64; BAYESR_COMPONENTS];
    for k in 0..BAYESR_COMPONENTS {
        log_pi[k] = pi[k].ln();
        tau2[k] = gamma[k] * sigma_lambda2;
    }
    Ok(component_log_weights_unchecked(
        &log_pi, &tau2, u, d, sigma_e2,
    ))
}

#[inline]
fn normalize_component_log_weights_unchecked(
    log_weights: &[f64; BAYESR_COMPONENTS],
) -> [f64; BAYESR_COMPONENTS] {
    let max_log = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut q = [0.0; BAYESR_COMPONENTS];
    for (dst, value) in q.iter_mut().zip(log_weights.iter()) {
        *dst = (*value - max_log).exp();
    }
    let total: f64 = q.iter().sum();
    for value in &mut q {
        *value /= total;
    }
    q
}

fn normalize_component_log_weights_checked(
    log_weights: &[f64],
) -> Result<[f64; BAYESR_COMPONENTS], String> {
    if log_weights.len() != BAYESR_COMPONENTS || !log_weights.iter().all(|v| v.is_finite()) {
        return Err("BayesR component log weights must contain four finite values".to_string());
    }
    let mut fixed = [0.0_f64; BAYESR_COMPONENTS];
    fixed.copy_from_slice(log_weights);
    let q = normalize_component_log_weights_unchecked(&fixed);
    let total: f64 = q.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return Err("BayesR component weights are not normalizable".to_string());
    }
    if !q.iter().all(|value| value.is_finite() && *value >= 0.0) {
        return Err("BayesR component probabilities are not finite and non-negative".to_string());
    }
    Ok(q)
}

#[inline]
fn sample_component<R: Rng + ?Sized>(q: &[f64; BAYESR_COMPONENTS], rng: &mut R) -> usize {
    let draw = rng.random::<f64>();
    let mut cumulative = 0.0;
    for (k, probability) in q.iter().enumerate() {
        cumulative += *probability;
        if draw < cumulative || k + 1 == BAYESR_COMPONENTS {
            return k;
        }
    }
    BAYESR_COMPONENTS - 1
}

struct BayesRResult {
    beta: Vec<f64>,
    alpha: Vec<f64>,
    varbeta: Vec<f64>,
    vare: f64,
    h2_mean: f64,
    var_h2: f64,
    pip: Vec<f64>,
    component_prob: Vec<f32>,
    pi: Vec<f64>,
    sigma_lambda2: f64,
    rhat_h2: f64,
    rhat_metrics: Vec<f64>,
    actual_iterations: usize,
    convergence_iteration: usize,
    posterior_samples: usize,
}

#[allow(clippy::too_many_arguments)]
fn bayesr_core_impl<B: BayesMarkerBackend>(
    backend: &mut B,
    y: &[f64],
    x: &[f64],
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    pi_init: &[f64],
    gamma: &[f64],
    df0_lambda: f64,
    s0_lambda2_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<BayesRResult, String> {
    let n = backend.n_samples();
    let p = backend.n_markers();
    if n <= 1 {
        return Err("BayesR requires n > 1".to_string());
    }
    if y.len() != n {
        return Err("BayesR phenotype length does not match genotype samples".to_string());
    }
    if x.len() != n.saturating_mul(q) || q == 0 {
        return Err("BayesR covariate dimensions are incompatible".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;
    if n_iter == 0 || thin == 0 {
        return Err("BayesR n_iter must be > 0 and thin must be >= 1".to_string());
    }
    if !(r2.is_finite() && r2 > 0.0 && r2 < 1.0) {
        return Err("BayesR r2 must be finite and in (0, 1)".to_string());
    }
    if !(df0_e.is_finite() && df0_e > 0.0) {
        return Err("BayesR df0_e must be finite and > 0".to_string());
    }
    if !(df0_lambda.is_finite() && df0_lambda > 0.0) {
        return Err("BayesR df0_lambda must be finite and > 0".to_string());
    }
    validate_bayesr_prior(pi_init, gamma)?;

    let mut rng = match seed {
        Some(value) => StdRng::seed_from_u64(value),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(error) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("Failed to generate random seed: {error}; using a fixed fallback");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|row_start, row_end, marker_block| {
        for offset in 0..(row_end - row_start) {
            let row = &marker_block[offset * n..(offset + 1) * n];
            let mut sum_sq = 0.0_f64;
            let mut sum = 0.0_f64;
            for &value in row {
                let value = value as f64;
                sum_sq += value * value;
                sum += value;
            }
            x2[row_start + offset] = sum_sq;
            mean_x[row_start + offset] = sum / n as f64;
        }
        Ok(())
    })?;
    let msx =
        x2.iter().sum::<f64>() / n as f64 - mean_x.iter().map(|value| value * value).sum::<f64>();
    if !(msx.is_finite() && msx > 0.0) {
        return Err("BayesR marker mean square must be positive".to_string());
    }
    let scale_calibration = resolve_bayesr_scale(
        &prior_calibration,
        msx,
        r2,
        pi_init,
        gamma,
        df0_lambda,
        s0_lambda2_opt,
    )?;
    let s0_lambda2 = scale_calibration.s0_lambda2;

    let mut pi = [0.0_f64; BAYESR_COMPONENTS];
    let pi_total = pi_init.iter().sum::<f64>();
    for (dst, value) in pi.iter_mut().zip(pi_init.iter()) {
        *dst = *value / pi_total;
    }
    let gamma_array: [f64; BAYESR_COMPONENTS] = [gamma[0], gamma[1], gamma[2], gamma[3]];
    let mut sigma_lambda2 = scale_calibration.initial_sigma_lambda2;
    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut beta = vec![0.0_f64; p];
    let mut component = vec![0usize; p];
    // Fixed-effect updates and posterior diagnostics retain f64 precision.  A
    // separate f32 marker residual is used only during the O(n * p) Gibbs
    // sweep, cutting the repeatedly streamed residual bandwidth in half while
    // retaining f64 dot-product accumulation.
    let mut residual = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0_f64; p];
    let mut beta_second_sum = vec![0.0_f64; p];
    let mut pip_sum = vec![0.0_f64; p];
    let mut component_sum = vec![0.0_f32; p.saturating_mul(BAYESR_COMPONENTS)];
    let mut alpha_sum = vec![0.0_f64; q];
    let mut var_e_sum = 0.0_f64;
    let mut sigma_lambda2_sum = 0.0_f64;
    let mut pi_sum = [0.0_f64; BAYESR_COMPONENTS];
    let mut h2_sum = 0.0_f64;
    let mut h2_sq_sum = 0.0_f64;
    let mut schedule = BayesSamplingController::new_with_options(
        n_iter,
        sampling.burnin,
        sampling.adaptive_burnin,
        thin,
        BAYES_RHAT_R_NAMES,
        sampling.posterior_target,
    );
    let chi_e = ChiSquared::new(n as f64 + df0_e).map_err(|error| error.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        // These four values are constant for an entire marker sweep.  Moving
        // the logarithms and gamma*sigma multiplication out of the inner
        // SNP loop removes eight scalar operations from every marker update.
        let mut log_pi = [0.0_f64; BAYESR_COMPONENTS];
        let mut tau2 = [0.0_f64; BAYESR_COMPONENTS];
        for k in 0..BAYESR_COMPONENTS {
            log_pi[k] = pi[k].ln();
            tau2[k] = gamma_array[k] * sigma_lambda2;
        }
        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            1.0e-10,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut residual,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&residual, &mut marker_residual);

        backend.for_each_block(|row_start, row_end, marker_block| {
            for offset in 0..(row_end - row_start) {
                let j = row_start + offset;
                let marker = &marker_block[offset * n..(offset + 1) * n];
                let old_beta = beta[j];
                let mut q_component = [0.0_f64; BAYESR_COMPONENTS];
                let (_, new_beta) = bayes_marker_update_f32_result(
                    &mut marker_residual,
                    marker,
                    old_beta,
                    x2[j],
                    |u| {
                        if !u.is_finite() {
                            return Err(
                                "BayesR marker sufficient statistics must be finite".to_string()
                            );
                        }
                        let log_weights =
                            component_log_weights_unchecked(&log_pi, &tau2, u, x2[j], var_e);
                        q_component = normalize_component_log_weights_checked(&log_weights)?;
                        let selected = sample_component(&q_component, &mut rng);
                        component[j] = selected;
                        if selected == 0 {
                            Ok(0.0)
                        } else {
                            let precision = x2[j] * inv_var_e + 1.0 / tau2[selected];
                            let posterior_var = 1.0 / precision;
                            let posterior_mean = posterior_var * u * inv_var_e;
                            let z: f64 = rng.sample(StandardNormal);
                            Ok(posterior_mean + posterior_var.sqrt() * z)
                        }
                    },
                )?;
                beta[j] = new_beta;

                if retain_sample {
                    beta_sum[j] += new_beta;
                    beta_second_sum[j] += new_beta * new_beta;
                    pip_sum[j] += 1.0 - q_component[0];
                    for k in 0..BAYESR_COMPONENTS {
                        component_sum[j * BAYESR_COMPONENTS + k] += q_component[k] as f32;
                    }
                }
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut residual);

        let mut active_count = 0usize;
        let mut scaled_sum = 0.0_f64;
        let mut component_counts = [0usize; BAYESR_COMPONENTS];
        for j in 0..p {
            let selected = component[j];
            component_counts[selected] += 1;
            if selected > 0 {
                active_count += 1;
                scaled_sum += beta[j] * beta[j] / gamma_array[selected];
            }
        }
        let lambda_df = df0_lambda + active_count as f64;
        let lambda_numerator = df0_lambda * s0_lambda2 + scaled_sum;
        sigma_lambda2 =
            lambda_numerator / rng.sample(ChiSquared::new(lambda_df).map_err(|e| e.to_string())?);
        if !(sigma_lambda2.is_finite() && sigma_lambda2 > 0.0) {
            return Err("BayesR sigma_lambda2 became non-finite or non-positive".to_string());
        }

        let mut dirichlet_draw = [0.0_f64; BAYESR_COMPONENTS];
        let mut dirichlet_total = 0.0_f64;
        for k in 0..BAYESR_COMPONENTS {
            let shape = 1.0 + component_counts[k] as f64;
            let draw = rng.sample(Gamma::new(shape, 1.0).map_err(|e| e.to_string())?);
            dirichlet_draw[k] = draw;
            dirichlet_total += draw;
        }
        for k in 0..BAYESR_COMPONENTS {
            pi[k] = dirichlet_draw[k] / dirichlet_total;
        }

        var_e = (ddot_f64(&residual, &residual) + prior_ss_e) / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesR residual variance became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for k in 0..BAYESR_COMPONENTS {
                pi_sum[k] += pi[k];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            sigma_lambda2_sum += sigma_lambda2;
            let var_g = genetic_variance_from_residual(y, &residual, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            let rhat_metrics = [
                h2,
                var_g,
                var_e,
                sigma_lambda2,
                pi[0],
                pi[1],
                pi[2],
                pi[3],
                active_count as f64,
            ];
            if schedule.observe(&rhat_metrics) {
                beta_sum.fill(0.0);
                beta_second_sum.fill(0.0);
                pip_sum.fill(0.0);
                component_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                sigma_lambda2_sum = 0.0;
                pi_sum = [0.0; BAYESR_COMPONENTS];
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep != schedule.posterior_target() {
        return Err(format!(
            "BayesR retained {n_keep} posterior samples; expected {}",
            schedule.posterior_target()
        ));
    }
    let inv_keep = 1.0 / n_keep as f64;
    let mut varbeta = vec![0.0_f64; p];
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        let second = beta_second_sum[j] * inv_keep;
        varbeta[j] = (second - beta_sum[j] * beta_sum[j]).max(0.0);
        pip_sum[j] *= inv_keep;
    }
    for value in &mut component_sum {
        *value *= inv_keep as f32;
    }
    for value in &mut alpha_sum {
        *value *= inv_keep;
    }
    let h2_mean = h2_sum * inv_keep;
    let var_h2 = (h2_sq_sum * inv_keep - h2_mean * h2_mean).max(0.0);
    let pi_mean = pi_sum
        .iter()
        .map(|value| value * inv_keep)
        .collect::<Vec<_>>();
    Ok(BayesRResult {
        beta: beta_sum,
        alpha: alpha_sum,
        varbeta,
        vare: var_e_sum * inv_keep,
        h2_mean,
        var_h2,
        pip: pip_sum,
        component_prob: component_sum,
        pi: pi_mean,
        sigma_lambda2: sigma_lambda2_sum * inv_keep,
        rhat_h2: schedule.rhat_value(),
        rhat_metrics: schedule.rhat_values(),
        actual_iterations: schedule.actual_iterations(),
        convergence_iteration: schedule.convergence_iteration(),
        posterior_samples: n_keep,
    })
}

/// Per-chain state for the native BayesR controller.  The genotype backend is
/// deliberately not part of this structure: all chains consume the same
/// decoded marker block while their Markov states and random streams remain
/// independent.
struct BayesRChainState {
    rng: StdRng,
    beta: Vec<f64>,
    component: Vec<usize>,
    alpha: Vec<f64>,
    residual: Vec<f64>,
    marker_residual: Vec<f32>,
    alpha_xtr: Vec<f64>,
    alpha_delta: Vec<f64>,
    alpha_tmp_n: Vec<f64>,
    pi: [f64; BAYESR_COMPONENTS],
    sigma_lambda2: f64,
    var_e: f64,
    beta_sum: Vec<f64>,
    beta_second_sum: Vec<f64>,
    pip_sum: Vec<f64>,
    component_sum: Vec<f32>,
    alpha_sum: Vec<f64>,
    var_e_sum: f64,
    sigma_lambda2_sum: f64,
    pi_sum: [f64; BAYESR_COMPONENTS],
}

impl BayesRChainState {
    fn new(
        fixed_init: &BayesFixedEffectInit,
        p: usize,
        q: usize,
        pi_base: [f64; BAYESR_COMPONENTS],
        sigma_lambda2: f64,
        var_e: f64,
        start_scale: f64,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => {
                let mut seed_bytes = [0u8; 32];
                if OsRng.try_fill_bytes(&mut seed_bytes).is_err() {
                    seed_bytes = [42u8; 32];
                }
                StdRng::from_seed(seed_bytes)
            }
        };
        Self {
            rng,
            beta: vec![0.0; p],
            component: vec![0; p],
            alpha: fixed_init.alpha.clone(),
            residual: fixed_init.residual.clone(),
            marker_residual: vec![0.0; fixed_init.residual.len()],
            alpha_xtr: vec![0.0; q],
            alpha_delta: vec![0.0; q],
            alpha_tmp_n: vec![0.0; fixed_init.residual.len()],
            pi: bayesr_chain_start_pi(pi_base, start_scale),
            sigma_lambda2: sigma_lambda2 * start_scale,
            var_e,
            beta_sum: vec![0.0; p],
            beta_second_sum: vec![0.0; p],
            pip_sum: vec![0.0; p],
            component_sum: vec![0.0; p.saturating_mul(BAYESR_COMPONENTS)],
            alpha_sum: vec![0.0; q],
            var_e_sum: 0.0,
            sigma_lambda2_sum: 0.0,
            pi_sum: [0.0; BAYESR_COMPONENTS],
        }
    }

    fn clear_posterior(&mut self) {
        self.beta_sum.fill(0.0);
        self.beta_second_sum.fill(0.0);
        self.pip_sum.fill(0.0);
        self.component_sum.fill(0.0);
        self.alpha_sum.fill(0.0);
        self.var_e_sum = 0.0;
        self.sigma_lambda2_sum = 0.0;
        self.pi_sum = [0.0; BAYESR_COMPONENTS];
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesr_lockstep_core_impl<B: BayesMarkerBackend>(
    backend: &mut B,
    y: &[f64],
    x: &[f64],
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    pi_init: &[f64],
    gamma: &[f64],
    df0_lambda: f64,
    s0_lambda2_opt: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    chain_pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<BayesRResult, String> {
    let n = backend.n_samples();
    let p = backend.n_markers();
    if n <= 1 || y.len() != n {
        return Err("BayesR phenotype/genotype dimensions are incompatible".to_string());
    }
    if q == 0 || x.len() != n.saturating_mul(q) {
        return Err("BayesR covariate dimensions are incompatible".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;
    if n_iter == 0 || thin == 0 {
        return Err("BayesR n_iter must be > 0 and thin must be >= 1".to_string());
    }
    if !(r2.is_finite() && r2 > 0.0 && r2 < 1.0) {
        return Err("BayesR r2 must be finite and in (0, 1)".to_string());
    }
    if !(df0_e.is_finite() && df0_e > 0.0) {
        return Err("BayesR df0_e must be finite and > 0".to_string());
    }
    if !(df0_lambda.is_finite() && df0_lambda > 0.0) {
        return Err("BayesR df0_lambda must be finite and > 0".to_string());
    }
    validate_bayesr_prior(pi_init, gamma)?;

    let (x2, _mean_x, msx) = marker_sufficient_stats(backend, n, p)?;
    if !(msx.is_finite() && msx > 0.0) {
        return Err("BayesR marker mean square must be positive".to_string());
    }
    let scale_calibration = resolve_bayesr_scale(
        &prior_calibration,
        msx,
        r2,
        pi_init,
        gamma,
        df0_lambda,
        s0_lambda2_opt,
    )?;
    let s0_lambda2 = scale_calibration.s0_lambda2;
    let pi_total = pi_init.iter().sum::<f64>();
    let pi_base = [
        pi_init[0] / pi_total,
        pi_init[1] / pi_total,
        pi_init[2] / pi_total,
        pi_init[3] / pi_total,
    ];
    let initial_var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();

    let gamma_array = [gamma[0], gamma[1], gamma[2], gamma[3]];
    let adaptive = sampling.adaptive_burnin || sampling.burnin.is_none();
    let seeds = bayes_chain_seeds(seed, chains);
    let mut states = seeds
        .into_iter()
        .enumerate()
        .map(|(chain_index, chain_seed)| {
            let start_scale = bayes_chain_start_scale(chain_index, chains, adaptive);
            BayesRChainState::new(
                &fixed_init,
                p,
                q,
                pi_base,
                scale_calibration.initial_sigma_lambda2,
                initial_var_e,
                start_scale,
                chain_seed,
            )
        })
        .collect::<Vec<_>>();
    let mut controller = BayesMultiChainController::new_with_monitor_options(
        chains,
        n_iter,
        sampling.burnin,
        adaptive,
        thin,
        BAYES_RHAT_R_NAMES,
        BAYES_PREDICTIVE_MONITOR_NAMES,
        sampling.posterior_target,
    )?;
    let chi_e = ChiSquared::new(n as f64 + df0_e).map_err(|error| error.to_string())?;

    while controller.should_run() {
        let retained = controller.begin_iteration();
        let collect = controller.collecting_posterior();
        bayes_chain_for_each_mut(&mut states, chain_pool, |state| {
            let inv_var_e = 1.0 / state.var_e;
            update_alpha_gauss_seidel_blas(
                x,
                n,
                q,
                inv_var_e,
                1.0e-10,
                &x2_x,
                &xtx,
                &mut state.alpha,
                &mut state.residual,
                &mut state.alpha_xtr,
                &mut state.alpha_delta,
                &mut state.alpha_tmp_n,
                &mut state.rng,
            );
            copy_f64_to_f32(&state.residual, &mut state.marker_residual);
        });

        backend.for_each_block(|row_start, row_end, marker_block| {
            bayes_chain_try_for_each_mut(&mut states, chain_pool, |state| {
                let mut log_pi = [0.0_f64; BAYESR_COMPONENTS];
                let mut tau2 = [0.0_f64; BAYESR_COMPONENTS];
                for k in 0..BAYESR_COMPONENTS {
                    log_pi[k] = state.pi[k].ln();
                    tau2[k] = gamma_array[k] * state.sigma_lambda2;
                }
                let inv_var_e = 1.0 / state.var_e;
                for offset in 0..(row_end - row_start) {
                    let j = row_start + offset;
                    let marker = &marker_block[offset * n..(offset + 1) * n];
                    let old_beta = state.beta[j];
                    let mut q_component = [0.0_f64; BAYESR_COMPONENTS];
                    let (_, new_beta) = bayes_marker_update_f32_result(
                        &mut state.marker_residual,
                        marker,
                        old_beta,
                        x2[j],
                        |u| {
                            if !u.is_finite() {
                                return Err("BayesR marker sufficient statistics must be finite"
                                    .to_string());
                            }
                            let log_weights = component_log_weights_unchecked(
                                &log_pi,
                                &tau2,
                                u,
                                x2[j],
                                state.var_e,
                            );
                            q_component = normalize_component_log_weights_checked(&log_weights)?;
                            let selected = sample_component(&q_component, &mut state.rng);
                            state.component[j] = selected;
                            if selected == 0 {
                                Ok(0.0)
                            } else {
                                let precision = x2[j] * inv_var_e + 1.0 / tau2[selected];
                                let posterior_var = 1.0 / precision;
                                let posterior_mean = posterior_var * u * inv_var_e;
                                let z: f64 = state.rng.sample(StandardNormal);
                                Ok(posterior_mean + posterior_var.sqrt() * z)
                            }
                        },
                    )?;
                    state.beta[j] = new_beta;
                    if collect && retained {
                        state.beta_sum[j] += new_beta;
                        state.beta_second_sum[j] += new_beta * new_beta;
                        state.pip_sum[j] += 1.0 - q_component[0];
                        for k in 0..BAYESR_COMPONENTS {
                            state.component_sum[j * BAYESR_COMPONENTS + k] += q_component[k] as f32;
                        }
                    }
                }
                Ok(())
            })
        })?;

        bayes_chain_for_each_mut(&mut states, chain_pool, |state| {
            copy_f32_to_f64(&state.marker_residual, &mut state.residual);
        });

        let chain_summaries = bayes_chain_try_map_mut(&mut states, chain_pool, |state| {
            let mut active_count = 0usize;
            let mut scaled_sum = 0.0_f64;
            let mut component_counts = [0usize; BAYESR_COMPONENTS];
            for j in 0..p {
                let selected = state.component[j];
                component_counts[selected] += 1;
                if selected > 0 {
                    active_count += 1;
                    scaled_sum += state.beta[j] * state.beta[j] / gamma_array[selected];
                }
            }
            let lambda_df = df0_lambda + active_count as f64;
            let lambda_numerator = df0_lambda * s0_lambda2 + scaled_sum;
            state.sigma_lambda2 = lambda_numerator
                / state
                    .rng
                    .sample(ChiSquared::new(lambda_df).map_err(|error| error.to_string())?);
            if !(state.sigma_lambda2.is_finite() && state.sigma_lambda2 > 0.0) {
                return Err("BayesR sigma_lambda2 became non-finite or non-positive".to_string());
            }

            let mut dirichlet_draw = [0.0_f64; BAYESR_COMPONENTS];
            let mut dirichlet_total = 0.0_f64;
            for k in 0..BAYESR_COMPONENTS {
                let shape = 1.0 + component_counts[k] as f64;
                let draw = state
                    .rng
                    .sample(Gamma::new(shape, 1.0).map_err(|error| error.to_string())?);
                dirichlet_draw[k] = draw;
                dirichlet_total += draw;
            }
            for k in 0..BAYESR_COMPONENTS {
                state.pi[k] = dirichlet_draw[k] / dirichlet_total;
            }

            state.var_e =
                (ddot_f64(&state.residual, &state.residual) + prior_ss_e) / state.rng.sample(chi_e);
            if !(state.var_e.is_finite() && state.var_e > 0.0) {
                return Err(
                    "BayesR residual variance became non-finite or non-positive".to_string()
                );
            }
            let var_g = genetic_variance_from_residual(y, &state.residual, x, &state.alpha, n, q);
            let h2_value = var_g / (var_g + state.var_e);

            if collect && retained {
                for k in 0..BAYESR_COMPONENTS {
                    state.pi_sum[k] += state.pi[k];
                }
                for k in 0..q {
                    state.alpha_sum[k] += state.alpha[k];
                }
                state.var_e_sum += state.var_e;
                state.sigma_lambda2_sum += state.sigma_lambda2;
            }
            Ok((
                vec![
                    h2_value,
                    var_g,
                    state.var_e,
                    state.sigma_lambda2,
                    state.pi[0],
                    state.pi[1],
                    state.pi[2],
                    state.pi[3],
                    active_count as f64,
                ],
                bayes_predictive_monitor_values(
                    y,
                    &state.residual,
                    x,
                    &state.alpha,
                    n,
                    q,
                    h2_value,
                    var_g,
                    state.var_e,
                ),
            ))
        })?;
        let (metrics, monitor_metrics): (Vec<_>, Vec<_>) = chain_summaries.into_iter().unzip();
        if controller.observe_with_monitor(&metrics, &monitor_metrics, retained)? {
            bayes_chain_for_each_mut(&mut states, chain_pool, |state: &mut BayesRChainState| {
                state.clear_posterior();
            });
        }
    }

    let n_keep = controller.posterior_samples();
    if n_keep != controller.posterior_target() {
        return Err(format!(
            "BayesR retained {n_keep} posterior samples; expected {}",
            controller.posterior_target()
        ));
    }
    let scale = 1.0 / (n_keep * chains) as f64;
    let mut beta = vec![0.0_f64; p];
    let mut varbeta = vec![0.0_f64; p];
    let mut pip = vec![0.0_f64; p];
    let mut component_prob = vec![0.0_f32; p.saturating_mul(BAYESR_COMPONENTS)];
    for j in 0..p {
        let sum_beta = states.iter().map(|state| state.beta_sum[j]).sum::<f64>();
        let sum_second = states
            .iter()
            .map(|state| state.beta_second_sum[j])
            .sum::<f64>();
        beta[j] = sum_beta * scale;
        varbeta[j] = (sum_second * scale - beta[j] * beta[j]).max(0.0);
        pip[j] = states.iter().map(|state| state.pip_sum[j]).sum::<f64>() * scale;
        for k in 0..BAYESR_COMPONENTS {
            component_prob[j * BAYESR_COMPONENTS + k] = states
                .iter()
                .map(|state| state.component_sum[j * BAYESR_COMPONENTS + k])
                .sum::<f32>()
                * scale as f32;
        }
    }
    let alpha = (0..q)
        .map(|k| states.iter().map(|state| state.alpha_sum[k]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let vare = states.iter().map(|state| state.var_e_sum).sum::<f64>() * scale;
    let sigma_lambda2 = states
        .iter()
        .map(|state| state.sigma_lambda2_sum)
        .sum::<f64>()
        * scale;
    let pi = (0..BAYESR_COMPONENTS)
        .map(|k| states.iter().map(|state| state.pi_sum[k]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let (h2_mean, var_h2, rhat_h2) = controller.posterior_h2_summary();
    let rhat_metrics = controller.posterior_rhat_values();
    Ok(BayesRResult {
        beta,
        alpha,
        varbeta,
        vare,
        h2_mean,
        var_h2,
        pip,
        component_prob,
        pi,
        sigma_lambda2,
        rhat_h2,
        rhat_metrics,
        actual_iterations: controller.actual_iterations(),
        convergence_iteration: controller.convergence_iteration(),
        posterior_samples: n_keep,
    })
}

fn average_bayesr_f64(values: &[Vec<f64>]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut output = values[0].clone();
    for value in values.iter().skip(1) {
        for (dst, src) in output.iter_mut().zip(value.iter()) {
            *dst += *src;
        }
    }
    let scale = 1.0 / values.len() as f64;
    for value in &mut output {
        *value *= scale;
    }
    output
}

fn average_bayesr_f32(values: &[Vec<f32>]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut output = values[0].clone();
    for value in values.iter().skip(1) {
        for (dst, src) in output.iter_mut().zip(value.iter()) {
            *dst += *src;
        }
    }
    let scale = 1.0 / values.len() as f32;
    for value in &mut output {
        *value *= scale;
    }
    output
}

fn aggregate_bayesr_results(results: &[BayesRResult]) -> BayesRResult {
    let h2_means = results
        .iter()
        .map(|value| value.h2_mean)
        .collect::<Vec<_>>();
    let h2_vars = results.iter().map(|value| value.var_h2).collect::<Vec<_>>();
    let rhats = results
        .iter()
        .map(|value| value.rhat_h2)
        .collect::<Vec<_>>();
    let rhat_metrics = average_bayesr_f64(
        &results
            .iter()
            .map(|value| value.rhat_metrics.clone())
            .collect::<Vec<_>>(),
    );
    let n_post = results
        .iter()
        .map(|value| value.posterior_samples)
        .min()
        .unwrap_or(0);
    let (h2_mean, var_h2, rhat_h2) = aggregate_bayes_h2(&h2_means, &h2_vars, &rhats, n_post);
    let convergence_values = results
        .iter()
        .map(|value| value.convergence_iteration)
        .collect::<Vec<_>>();
    BayesRResult {
        beta: average_bayesr_f64(
            &results
                .iter()
                .map(|value| value.beta.clone())
                .collect::<Vec<_>>(),
        ),
        alpha: average_bayesr_f64(
            &results
                .iter()
                .map(|value| value.alpha.clone())
                .collect::<Vec<_>>(),
        ),
        varbeta: average_bayesr_f64(
            &results
                .iter()
                .map(|value| value.varbeta.clone())
                .collect::<Vec<_>>(),
        ),
        vare: results.iter().map(|value| value.vare).sum::<f64>() / results.len() as f64,
        h2_mean,
        var_h2,
        pip: average_bayesr_f64(
            &results
                .iter()
                .map(|value| value.pip.clone())
                .collect::<Vec<_>>(),
        ),
        component_prob: average_bayesr_f32(
            &results
                .iter()
                .map(|value| value.component_prob.clone())
                .collect::<Vec<_>>(),
        ),
        pi: average_bayesr_f64(
            &results
                .iter()
                .map(|value| value.pi.clone())
                .collect::<Vec<_>>(),
        ),
        sigma_lambda2: results.iter().map(|value| value.sigma_lambda2).sum::<f64>()
            / results.len() as f64,
        rhat_h2,
        rhat_metrics,
        actual_iterations: results
            .iter()
            .map(|value| value.actual_iterations)
            .max()
            .unwrap_or(0),
        convergence_iteration: if convergence_values.iter().all(|value| *value > 0) {
            convergence_values.iter().copied().max().unwrap_or(0)
        } else {
            0
        },
        posterior_samples: n_post,
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesr_dense_multi_core_impl(
    m: &[f32],
    y: &[f64],
    x: &[f64],
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi_init: &[f64],
    gamma: &[f64],
    df0_lambda: f64,
    s0_lambda2_opt: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    threads: usize,
) -> Result<BayesRResult, String> {
    let effective = effective_bayes_chains(chains, threads)?;
    if effective > 1 {
        let n = y.len();
        let p = m
            .len()
            .checked_div(n)
            .ok_or_else(|| "BayesR M has incompatible dimensions".to_string())?;
        let mut backend = DenseBayesBackend::new(m, n, p)?;
        let chain_pool = build_bayes_chain_pool(threads)?;
        return bayesr_lockstep_core_impl(
            &mut backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            pi_init,
            gamma,
            df0_lambda,
            s0_lambda2_opt,
            seed,
            effective,
            chain_pool.as_ref(),
        );
    }
    let seeds = bayes_chain_seeds(seed, effective);
    let n = y.len();
    let p = m
        .len()
        .checked_div(n)
        .ok_or_else(|| "BayesR M has incompatible dimensions".to_string())?;
    let mut results = Vec::with_capacity(effective);
    for chain_seed in seeds {
        let mut backend = DenseBayesBackend::new(m, n, p)?;
        results.push(bayesr_core_impl(
            &mut backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            pi_init,
            gamma,
            df0_lambda,
            s0_lambda2_opt,
            chain_seed,
        )?);
    }
    Ok(aggregate_bayesr_results(&results))
}

#[allow(clippy::too_many_arguments)]
fn bayesr_packed_multi_core_impl<'a>(
    source: &mut BayesPackedSource<'a>,
    n_samples_total: usize,
    y: &[f64],
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_indices: &[usize],
    x: &[f64],
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi_init: &[f64],
    gamma: &[f64],
    df0_lambda: f64,
    s0_lambda2_opt: Option<f64>,
    seed: Option<u64>,
    block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
    chains: usize,
    threads: usize,
) -> Result<BayesRResult, String> {
    let effective = effective_bayes_chains(chains, threads)?;
    if effective > 1 {
        let mut backend = PackedBayesBackend::new(
            source,
            n_samples_total,
            row_flip.len(),
            row_flip,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_indices,
            block_rows,
            pool,
        )?;
        if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
            let mut dense_backend =
                DenseBayesBackend::new(&m_dense, sample_indices.len(), row_flip.len())?;
            return bayesr_lockstep_core_impl(
                &mut dense_backend,
                y,
                x,
                q,
                n_iter,
                sampling,
                thin,
                r2,
                df0_e,
                prior_ss_e,
                pi_init,
                gamma,
                df0_lambda,
                s0_lambda2_opt,
                seed,
                effective,
                pool,
            );
        }
        return bayesr_lockstep_core_impl(
            &mut backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            pi_init,
            gamma,
            df0_lambda,
            s0_lambda2_opt,
            seed,
            effective,
            pool,
        );
    }
    let seeds = bayes_chain_seeds(seed, effective);
    let mut results = Vec::with_capacity(effective);
    for chain_seed in seeds {
        let mut chain_source = duplicate_bayes_source(source)?;
        results.push(bayesr_packed_core_impl(
            &mut chain_source,
            n_samples_total,
            y,
            row_flip,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_indices,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            pi_init,
            gamma,
            df0_lambda,
            s0_lambda2_opt,
            chain_seed,
            block_rows,
            pool,
        )?);
    }
    Ok(aggregate_bayesr_results(&results))
}

const BAYESR_DEFAULT_PI: [f64; BAYESR_COMPONENTS] = [0.95, 0.03, 0.01, 0.01];
const BAYESR_DEFAULT_GAMMA: [f64; BAYESR_COMPONENTS] = [0.0, 0.01, 0.1, 1.0];

fn array1_to_vec(arr: &PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn array2_to_vec(arr: &PyReadonlyArray2<f64>) -> Vec<f64> {
    let view = arr.as_array();
    let (n, p) = view.dim();
    let mut out = Vec::with_capacity(n.saturating_mul(p));
    for i in 0..n {
        for j in 0..p {
            out.push(view[[i, j]]);
        }
    }
    out
}

fn parse_covariates(x: Option<&PyReadonlyArray2<f64>>, n: usize) -> PyResult<(Vec<f64>, usize)> {
    match x {
        Some(arr) => {
            let shape = arr.shape();
            if shape.len() != 2 || shape[0] != n || shape[1] == 0 {
                return Err(PyValueError::new_err(
                    "BayesR X must be a non-empty matrix with rows matching len(y)",
                ));
            }
            let values = if let Ok(slice) = crate::stats_common::array_standard_slice(arr) {
                slice.to_vec()
            } else {
                array2_to_vec(arr)
            };
            Ok((values, shape[1]))
        }
        None => Ok((vec![1.0; n], 1usize)),
    }
}

fn parse_bayesr_prior(
    values: Option<&PyReadonlyArray1<f64>>,
    default: &[f64; BAYESR_COMPONENTS],
    label: &str,
) -> PyResult<Vec<f64>> {
    let out = match values {
        Some(arr) => array1_to_vec(arr),
        None => default.to_vec(),
    };
    validate_bayesr_prior(
        if label == "pi" {
            &out
        } else {
            &BAYESR_DEFAULT_PI
        },
        if label == "gamma" {
            &out
        } else {
            &BAYESR_DEFAULT_GAMMA
        },
    )
    .map_err(PyValueError::new_err)?;
    Ok(out)
}

fn bayesr_result_to_pydict<'py>(
    py: Python<'py>,
    result: BayesRResult,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py).into_bound();
    let beta = result.beta.into_pyarray(py);
    let alpha = result.alpha.into_pyarray(py);
    let varbeta = result.varbeta.into_pyarray(py);
    let pip = result.pip.into_pyarray(py);
    let pi = result.pi.into_pyarray(py);
    let component_prob =
        Array2::from_shape_vec((pip.len(), BAYESR_COMPONENTS), result.component_prob).map_err(
            |error| {
                PyValueError::new_err(format!(
                    "invalid BayesR component probability shape: {error}"
                ))
            },
        )?;
    let component_prob = PyArray2::from_owned_array(py, component_prob);
    out.set_item("beta", beta)?;
    out.set_item("alpha", alpha)?;
    out.set_item("varbeta", varbeta)?;
    out.set_item("vare", result.vare)?;
    out.set_item("h2_mean", result.h2_mean)?;
    out.set_item("var_h2", result.var_h2)?;
    out.set_item("pip", pip)?;
    out.set_item("component_prob", component_prob)?;
    out.set_item("pi", pi)?;
    out.set_item("sigma_lambda2", result.sigma_lambda2)?;
    out.set_item("rhat_h2", result.rhat_h2)?;
    let rhat_metrics = rhat_metrics_to_py(py, BAYES_RHAT_R_NAMES, &result.rhat_metrics)?;
    out.set_item("rhat_metrics", rhat_metrics)?;
    out.set_item("rhat_max", finite_rhat_max(&result.rhat_metrics))?;
    out.set_item("actual_iterations", result.actual_iterations)?;
    out.set_item("convergence_iteration", result.convergence_iteration)?;
    out.set_item("posterior_samples", result.posterior_samples)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_e = 5.0,
    prior_ss_e = None,
    pi = None,
    gamma = None,
    df0_lambda = 4.0,
    s0_lambda2 = None,
    seed = None,
    chains = 1,
    threads = 1,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesr<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    m: Bound<'py, PyAny>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi: Option<PyReadonlyArray1<'py, f64>>,
    gamma: Option<PyReadonlyArray1<'py, f64>>,
    df0_lambda: f64,
    s0_lambda2: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    threads: usize,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    let y_vec = if let Ok(slice) = y.as_slice() {
        slice.to_vec()
    } else {
        array1_to_vec(&y)
    };
    let m_input = array2_to_f32_input(&m, "BayesR M")?;
    if m_input.cols() != y_vec.len() {
        return Err(PyValueError::new_err("BayesR M cols must match len(y)"));
    }
    let m_slice = m_input.as_slice();
    let (x_vec, q) = parse_covariates(x.as_ref(), y_vec.len())?;
    let pi_vec = parse_bayesr_prior(pi.as_ref(), &BAYESR_DEFAULT_PI, "pi")?;
    let gamma_vec = parse_bayesr_prior(gamma.as_ref(), &BAYESR_DEFAULT_GAMMA, "gamma")?;
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);
    let result = py.detach(|| {
        bayesr_dense_multi_core_impl(
            m_slice, &y_vec, &x_vec, q, n_iter, sampling, thin, r2, df0_e, prior_ss_e, &pi_vec,
            &gamma_vec, df0_lambda, s0_lambda2, seed, chains, threads,
        )
    });
    match result {
        Ok(result) => bayesr_result_to_pydict(py, result),
        Err(error) => Err(PyValueError::new_err(error)),
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesr_packed_core_impl<'a>(
    source: &'a mut BayesPackedSource<'a>,
    n_samples_total: usize,
    y: &[f64],
    row_flip: &'a [bool],
    row_mean: &'a [f32],
    row_inv_sd: &'a [f32],
    packed_row_indices: Option<&'a [usize]>,
    sample_indices: &'a [usize],
    x: &[f64],
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi_init: &[f64],
    gamma: &[f64],
    df0_lambda: f64,
    s0_lambda2_opt: Option<f64>,
    seed: Option<u64>,
    block_rows: Option<usize>,
    pool: Option<&'a Arc<rayon::ThreadPool>>,
) -> Result<BayesRResult, String> {
    let p = row_flip.len();
    let mut backend = PackedBayesBackend::new(
        source,
        n_samples_total,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        sample_indices,
        block_rows,
        pool,
    )?;
    // Keep the backend policy shared with BayesA/B/C: when the memory budget
    // supplies a block covering every active marker, decode once and run the
    // dense sampler rather than repeatedly decoding the same BED rows for
    // every MCMC iteration.
    if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
        let mut dense_backend = DenseBayesBackend::new(&m_dense, sample_indices.len(), p)?;
        return bayesr_core_impl(
            &mut dense_backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            pi_init,
            gamma,
            df0_lambda,
            s0_lambda2_opt,
            seed,
        );
    }
    bayesr_core_impl(
        &mut backend,
        y,
        x,
        q,
        n_iter,
        sampling,
        thin,
        r2,
        df0_e,
        prior_ss_e,
        pi_init,
        gamma,
        df0_lambda,
        s0_lambda2_opt,
        seed,
    )
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_e = 5.0,
    prior_ss_e = None,
    pi = None,
    gamma = None,
    df0_lambda = 4.0,
    s0_lambda2 = None,
    threads = 0,
    chains = 1,
    seed = None,
    block_rows = None,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesr_packed<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_mean: PyReadonlyArray1<'py, f32>,
    row_inv_sd: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi: Option<PyReadonlyArray1<'py, f64>>,
    gamma: Option<PyReadonlyArray1<'py, f64>>,
    df0_lambda: f64,
    s0_lambda2: Option<f64>,
    threads: usize,
    chains: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err("BayesR n_samples must be > 0"));
    }
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    let packed_arr = packed.as_array();
    let shape = packed_arr.shape();
    if shape.len() != 2 || shape[1] != n_samples.div_ceil(4) {
        return Err(PyValueError::new_err(
            "BayesR packed shape is incompatible with n_samples",
        ));
    }
    let p = shape[0];
    if p == 0 {
        return Err(PyValueError::new_err(
            "BayesR packed genotype has no markers",
        ));
    }
    let row_flip_vec = row_flip.as_array().iter().copied().collect::<Vec<_>>();
    let row_maf_vec = row_maf.as_array().iter().copied().collect::<Vec<_>>();
    let row_mean_vec = row_mean.as_array().iter().copied().collect::<Vec<_>>();
    let row_inv_sd_vec = row_inv_sd.as_array().iter().copied().collect::<Vec<_>>();
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "BayesR packed row metadata must match packed marker count",
        ));
    }
    let y_vec = if let Ok(slice) = y.as_slice() {
        slice.to_vec()
    } else {
        array1_to_vec(&y)
    };
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != y_vec.len() {
        return Err(PyValueError::new_err(
            "BayesR sample_indices length must match len(y)",
        ));
    }
    let (x_vec, q) = parse_covariates(x.as_ref(), y_vec.len())?;
    let pi_vec = parse_bayesr_prior(pi.as_ref(), &BAYESR_DEFAULT_PI, "pi")?;
    let gamma_vec = parse_bayesr_prior(gamma.as_ref(), &BAYESR_DEFAULT_GAMMA, "gamma")?;
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);
    let packed_flat: Cow<'_, [u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool = pool_owned.as_ref();
    let result = py.detach(|| {
        let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());
        let mut source = BayesPackedSource::Resident {
            packed_flat: packed_flat.as_ref(),
            bytes_per_snp: n_samples.div_ceil(4),
        };
        bayesr_packed_multi_core_impl(
            &mut source,
            n_samples,
            &y_vec,
            &row_flip_vec,
            &row_mean_vec,
            &row_inv_sd_vec,
            None,
            &sample_idx,
            &x_vec,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            &pi_vec,
            &gamma_vec,
            df0_lambda,
            s0_lambda2,
            seed,
            block_rows,
            pool,
            chains,
            threads,
        )
    });
    match result {
        Ok(result) => bayesr_result_to_pydict(py, result),
        Err(error) => Err(PyValueError::new_err(error)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    n_samples,
    row_indices,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_e = 5.0,
    prior_ss_e = None,
    pi = None,
    gamma = None,
    df0_lambda = 4.0,
    s0_lambda2 = None,
    threads = 0,
    chains = 1,
    seed = None,
    block_rows = None,
    mmap_window_mb = None,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesr_stream_bed<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_mean: PyReadonlyArray1<'py, f32>,
    row_inv_sd: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi: Option<PyReadonlyArray1<'py, f64>>,
    gamma: Option<PyReadonlyArray1<'py, f64>>,
    df0_lambda: f64,
    s0_lambda2: Option<f64>,
    threads: usize,
    chains: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    mmap_window_mb: Option<usize>,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err("BayesR n_samples must be > 0"));
    }
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    let row_indices_raw = row_indices.as_slice()?.to_vec();
    let p = row_indices_raw.len();
    if p == 0 {
        return Err(PyValueError::new_err(
            "BayesR row_indices must not be empty",
        ));
    }
    let row_flip_vec = row_flip.as_array().iter().copied().collect::<Vec<_>>();
    let row_maf_vec = row_maf.as_array().iter().copied().collect::<Vec<_>>();
    let row_mean_vec = row_mean.as_array().iter().copied().collect::<Vec<_>>();
    let row_inv_sd_vec = row_inv_sd.as_array().iter().copied().collect::<Vec<_>>();
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "BayesR stream row metadata must match row_indices",
        ));
    }
    let y_vec = if let Ok(slice) = y.as_slice() {
        slice.to_vec()
    } else {
        array1_to_vec(&y)
    };
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != y_vec.len() {
        return Err(PyValueError::new_err(
            "BayesR sample_indices length must match len(y)",
        ));
    }
    let (x_vec, q) = parse_covariates(x.as_ref(), y_vec.len())?;
    let pi_vec = parse_bayesr_prior(pi.as_ref(), &BAYESR_DEFAULT_PI, "pi")?;
    let gamma_vec = parse_bayesr_prior(gamma.as_ref(), &BAYESR_DEFAULT_GAMMA, "gamma")?;
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);
    let pool_owned = get_cached_pool(threads)?;
    let pool = pool_owned.as_ref();
    let result = py.detach(|| {
        let rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(y_vec.len(), p))
            .max(1);
        let mut source =
            crate::bayes::build_bayes_source(None, &prefix, n_samples, rows, mmap_window_mb)?;
        let n_source = match &source {
            BayesPackedSource::Resident { .. } => p,
            BayesPackedSource::Windowed(matrix) => matrix.n_source_snps(),
        };
        let packed_row_indices =
            parse_index_vec_i64_value_error(row_indices_raw.as_slice(), n_source, "row_indices")
                .map_err(|error| error.to_string())?;
        let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());
        bayesr_packed_multi_core_impl(
            &mut source,
            n_samples,
            &y_vec,
            &row_flip_vec,
            &row_mean_vec,
            &row_inv_sd_vec,
            Some(packed_row_indices.as_slice()),
            &sample_idx,
            &x_vec,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            &pi_vec,
            &gamma_vec,
            df0_lambda,
            s0_lambda2,
            seed,
            Some(rows),
            pool,
            chains,
            threads,
        )
    });
    match result {
        Ok(result) => bayesr_result_to_pydict(py, result),
        Err(error) => Err(PyValueError::new_err(error)),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn component_log_weights_match_closed_form() {
        let pi: [f64; 4] = [0.95, 0.03, 0.01, 0.01];
        let gamma: [f64; 4] = [0.0, 0.01, 0.1, 1.0];
        let got = super::component_log_weights(&pi, &gamma, 2.5, 3.0, 1.7, 0.8)
            .expect("valid BayesR component parameters");
        let log_pi = pi.map(f64::ln);
        let tau2 = gamma.map(|value| value * 1.7);
        let fast = super::component_log_weights_unchecked(&log_pi, &tau2, 2.5, 3.0, 0.8);

        assert_eq!(got.len(), 4);
        assert!((got[0] - pi[0].ln()).abs() < 1e-12);
        for k in 0..4 {
            assert!((got[k] - fast[k]).abs() < 1e-12);
        }
        for k in 1..4 {
            let tau2 = gamma[k] * 1.7;
            let expected = pi[k].ln() - 0.5 * (1.0 + tau2 * 3.0 / 0.8).ln()
                + tau2 * 2.5_f64.powi(2) / (2.0 * 0.8 * (0.8 + tau2 * 3.0));
            assert!((got[k] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn component_probabilities_are_normalized_and_pip_is_one_minus_spike() {
        let log_weights: [f64; 4] = [0.2_f64.ln(), 0.3_f64.ln(), 0.1_f64.ln(), 0.4_f64.ln()];
        let q = super::normalize_component_log_weights_checked(&log_weights)
            .expect("finite component weights");
        let total: f64 = q.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((1.0 - q[0] - (q[1] + q[2] + q[3])).abs() < 1e-12);
        assert!((q[3] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn component_probabilities_reject_nonfinite_weights_before_sampling() {
        assert!(
            super::normalize_component_log_weights_checked(&[f64::INFINITY, 0.0, 0.0, 0.0,])
                .is_err()
        );
        assert!(
            super::normalize_component_log_weights_checked(&[f64::NAN, 0.0, 0.0, 0.0,]).is_err()
        );
        assert!(super::normalize_component_log_weights_checked(&[
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ])
        .is_err());
    }

    #[test]
    fn bayesr_prior_validation_requires_four_components_and_zero_spike() {
        assert!(
            super::validate_bayesr_prior(&[0.95, 0.03, 0.01, 0.01], &[0.0, 0.01, 0.1, 1.0]).is_ok()
        );
        assert!(super::validate_bayesr_prior(&[0.9, 0.1], &[0.0, 1.0]).is_err());
        assert!(
            super::validate_bayesr_prior(&[0.95, 0.03, 0.01, 0.01], &[0.1, 0.01, 0.1, 1.0])
                .is_err()
        );
    }

    #[test]
    fn bayesr_auto_scale_targets_global_mode_and_scales_quadratically() {
        let pi = [0.95, 0.03, 0.01, 0.01];
        let gamma = [0.0, 0.01, 0.1, 1.0];
        let calibration = super::BayesPriorCalibration {
            residualized_var: 4.0,
            initial_var_e: 2.0,
            prior_ss_e: 8.0,
        };
        let scaled_calibration = super::BayesPriorCalibration {
            residualized_var: 400.0,
            initial_var_e: 200.0,
            prior_ss_e: 800.0,
        };
        let got = super::resolve_bayesr_scale(&calibration, 2.0, 0.5, &pi, &gamma, 1.0, None)
            .expect("auto BayesR scale should be finite");
        let scaled =
            super::resolve_bayesr_scale(&scaled_calibration, 2.0, 0.5, &pi, &gamma, 1.0, None)
                .expect("scaled auto BayesR scale should be finite");
        let expected_gamma = pi
            .iter()
            .zip(gamma.iter())
            .map(|(weight, multiplier)| weight * multiplier)
            .sum::<f64>();
        let target_mode = calibration.residualized_var * 0.5 / (2.0 * expected_gamma);
        assert!((got.s0_lambda2 - target_mode * 3.0).abs() < 1e-12);
        assert!((got.initial_sigma_lambda2 - target_mode).abs() < 1e-12);
        let got_df4 = super::resolve_bayesr_scale(&calibration, 2.0, 0.5, &pi, &gamma, 4.0, None)
            .expect("df=4 auto BayesR scale should be finite");
        assert!((got_df4.s0_lambda2 - target_mode * 1.5).abs() < 1e-12);
        assert!((got_df4.initial_sigma_lambda2 - target_mode).abs() < 1e-12);
        assert!((scaled.s0_lambda2 / got.s0_lambda2 - 100.0).abs() < 1e-12);
        assert!((scaled.initial_sigma_lambda2 / got.initial_sigma_lambda2 - 100.0).abs() < 1e-12);
    }

    #[test]
    fn bayesr_explicit_scale_bypasses_auto_calibration() {
        let pi = [0.95, 0.03, 0.01, 0.01];
        let gamma = [0.0, 0.01, 0.1, 1.0];
        let calibration = super::BayesPriorCalibration {
            residualized_var: 4.0,
            initial_var_e: 2.0,
            prior_ss_e: 8.0,
        };
        let got = super::resolve_bayesr_scale(&calibration, 2.0, 0.5, &pi, &gamma, 1.0, Some(7.5))
            .expect("explicit BayesR scale should be accepted");
        assert_eq!(got.s0_lambda2, 7.5);
        assert_eq!(got.initial_sigma_lambda2, 7.5);
    }

    #[test]
    fn adaptive_chain_pi_preserves_simplex_and_scales_nonspike_mass() {
        let base = [0.95, 0.03, 0.01, 0.01];
        let lower = super::bayesr_chain_start_pi(base, 0.7);
        let upper = super::bayesr_chain_start_pi(base, 2.0);
        assert_eq!(super::bayesr_chain_start_pi(base, 1.0), base);
        assert!((lower.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((upper.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(lower.iter().all(|value| *value > 0.0));
        assert!(upper.iter().all(|value| *value > 0.0));
        assert!(lower[0] > base[0] && upper[0] < base[0]);
        let base_odds = (1.0 - base[0]) / base[0];
        let lower_odds = (1.0 - lower[0]) / lower[0];
        let upper_odds = (1.0 - upper[0]) / upper[0];
        assert!((lower_odds / base_odds - 0.7).abs() < 1e-12);
        assert!((upper_odds / base_odds - 2.0).abs() < 1e-12);
    }

    #[test]
    fn adaptive_chain_start_scale_separates_bayesr_state() {
        let fixed_init = super::BayesFixedEffectInit {
            alpha: vec![0.0],
            residual: vec![1.0, -1.0],
        };
        let base = [0.95, 0.03, 0.01, 0.01];
        let lower = super::BayesRChainState::new(&fixed_init, 2, 1, base, 3.0, 2.0, 0.7, Some(1));
        let upper = super::BayesRChainState::new(&fixed_init, 2, 1, base, 3.0, 2.0, 2.0, Some(2));
        assert!(lower.sigma_lambda2 < upper.sigma_lambda2);
        assert!(lower.pi[0] > base[0] && upper.pi[0] < base[0]);
        assert!((lower.pi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((upper.pi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn native_lockstep_bayesr_keeps_independent_chain_state() {
        let y = [0.2, -0.4, 0.1, 0.7, -0.1, 0.5];
        let m = [
            0.0_f32, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0,
        ];
        let x = [1.0_f64; 6];
        let result = super::bayesr_dense_multi_core_impl(
            &m,
            &y,
            &x,
            1,
            1,
            super::BayesSamplingOptions::new(Some(0), false, 1000),
            1,
            0.5,
            5.0,
            None,
            &[0.95, 0.03, 0.01, 0.01],
            &[0.0, 0.01, 0.1, 1.0],
            1.0,
            Some(1.0),
            Some(17),
            2,
            4,
        )
        .expect("native BayesR lockstep should finish its posterior window");
        assert_eq!(result.posterior_samples, 1000);
        assert_eq!(result.actual_iterations, 1000);
        assert_eq!(result.convergence_iteration, 0);
        for probabilities in result.component_prob.chunks_exact(4) {
            let total = probabilities.iter().map(|value| *value as f64).sum::<f64>();
            assert!((total - 1.0).abs() < 2e-5, "component sum={total}");
        }
    }
}
