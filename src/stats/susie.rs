use crate::blas::{
    cblas_dgemm_dispatch, BlasThreadGuard, CblasInt, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR,
};
use crate::brent::brent_minimize_with_init;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const V_BOUNDARY_TOL: f64 = 16.0 * f64::EPSILON;

#[derive(Debug, Clone)]
struct SingleEffectResult {
    alpha: Vec<f64>,
    mu: Vec<f64>,
    mu2: Vec<f64>,
    logbf: Vec<f64>,
    v: f64,
    lbf: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct SusieRssConfig {
    pub l: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub prior_tol: f64,
    pub threads: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SusieRssFit {
    pub p: usize,
    pub l: usize,
    pub alpha: Vec<f64>,
    pub mu: Vec<f64>,
    pub mu2: Vec<f64>,
    pub b: Vec<f64>,
    pub rb: Vec<f64>,
    pub lbf: Vec<f64>,
    pub lbf_variable: Vec<f64>,
    pub prior_variance: Vec<f64>,
    pub posterior_mean: Vec<f64>,
    pub z_expected: Vec<f64>,
    pub pip: Vec<f64>,
    pub elbo: Vec<f64>,
    pub max_alpha_delta: Vec<f64>,
    pub converged: bool,
    pub n_iter: usize,
}

fn normalize_prior_weights(prior: &[f64], p: usize) -> Result<Vec<f64>, String> {
    if p == 0 {
        return Err("z must contain at least one value".to_string());
    }
    if prior.len() != p {
        return Err(format!(
            "prior_weights length mismatch: expected {p}, got {}",
            prior.len()
        ));
    }
    let mut sum = 0.0_f64;
    for (index, &weight) in prior.iter().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "prior_weights[{index}] must be finite and non-negative"
            ));
        }
        sum += weight;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err("prior_weights must have a positive finite sum".to_string());
    }
    Ok(prior.iter().map(|&weight| weight / sum).collect())
}

#[inline]
fn logbf_at(z2: f64, v: f64) -> f64 {
    -0.5 * v.ln_1p() + 0.5 * (v / (1.0 + v)) * z2
}

fn single_effect_lbf(z_tilde: &[f64], log_prior: &[f64], v: f64) -> f64 {
    let mut max_score = f64::NEG_INFINITY;
    for (&z, &log_weight) in z_tilde.iter().zip(log_prior) {
        max_score = max_score.max(log_weight + logbf_at(z * z, v));
    }
    let normalizer = z_tilde
        .iter()
        .zip(log_prior)
        .map(|(&z, &log_weight)| (log_weight + logbf_at(z * z, v) - max_score).exp())
        .sum::<f64>();
    max_score + normalizer.ln()
}

fn single_effect_regression(
    z_tilde: &[f64],
    prior: &[f64],
    previous_v: f64,
) -> Result<SingleEffectResult, String> {
    let p = z_tilde.len();
    let prior = normalize_prior_weights(prior, p)?;
    let mut max_abs_z = 0.0_f64;
    for (index, &z) in z_tilde.iter().enumerate() {
        if !z.is_finite() {
            return Err(format!("z_tilde[{index}] must be finite"));
        }
        max_abs_z = max_abs_z.max(z.abs());
    }
    if max_abs_z > f64::MAX.sqrt() {
        return Err("squared residual z-score would overflow f64".to_string());
    }

    let log_prior: Vec<f64> = prior
        .iter()
        .map(|&weight| {
            if weight == 0.0 {
                f64::NEG_INFINITY
            } else {
                weight.ln()
            }
        })
        .collect();
    let v_upper = (max_abs_z * max_abs_z - 1.0).max(0.0);
    if v_upper <= V_BOUNDARY_TOL {
        return Ok(SingleEffectResult {
            alpha: prior,
            mu: vec![0.0; p],
            mu2: vec![0.0; p],
            logbf: vec![0.0; p],
            v: 0.0,
            lbf: 0.0,
        });
    }

    let (candidate_v, _) = brent_minimize_with_init(
        |v| -single_effect_lbf(z_tilde, &log_prior, v),
        0.0,
        v_upper,
        1e-12,
        200,
        None,
    );
    let mut best_v = 0.0_f64;
    let mut best_lbf = single_effect_lbf(z_tilde, &log_prior, 0.0);
    let candidate_v = candidate_v.clamp(0.0, v_upper);
    let candidate_lbf = single_effect_lbf(z_tilde, &log_prior, candidate_v);
    if candidate_lbf > best_lbf {
        best_v = candidate_v;
        best_lbf = candidate_lbf;
    }
    if previous_v.is_finite() {
        let previous_v = previous_v.clamp(0.0, v_upper);
        let previous_lbf = single_effect_lbf(z_tilde, &log_prior, previous_v);
        if previous_lbf > best_lbf {
            best_v = previous_v;
            best_lbf = previous_lbf;
        }
    }
    let upper_lbf = single_effect_lbf(z_tilde, &log_prior, v_upper);
    let upper_tie_tol = 64.0 * f64::EPSILON * best_lbf.abs().max(1.0);
    if upper_lbf >= best_lbf - upper_tie_tol {
        best_v = v_upper;
        best_lbf = upper_lbf;
    }

    if best_v <= V_BOUNDARY_TOL {
        return Ok(SingleEffectResult {
            alpha: prior,
            mu: vec![0.0; p],
            mu2: vec![0.0; p],
            logbf: vec![0.0; p],
            v: 0.0,
            lbf: 0.0,
        });
    }

    let logbf: Vec<f64> = z_tilde.iter().map(|&z| logbf_at(z * z, best_v)).collect();
    let scores: Vec<f64> = logbf
        .iter()
        .zip(&log_prior)
        .map(|(&bf, &log_weight)| bf + log_weight)
        .collect();
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let score_sum = scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .sum::<f64>();
    let alpha: Vec<f64> = scores
        .iter()
        .map(|&score| (score - max_score).exp() / score_sum)
        .collect();
    let shrinkage = best_v / (1.0 + best_v);
    let mu: Vec<f64> = z_tilde.iter().map(|&z| shrinkage * z).collect();
    let mu2: Vec<f64> = mu.iter().map(|&mean| shrinkage + mean * mean).collect();

    Ok(SingleEffectResult {
        alpha,
        mu,
        mu2,
        logbf,
        v: best_v,
        lbf: best_lbf,
    })
}

#[cfg(test)]
fn dense_r_times_b_scalar(r: &[f64], p: usize, b: &[f64], out: &mut [f64]) {
    for row in 0..p {
        let row_values = &r[row * p..(row + 1) * p];
        out[row] = row_values
            .iter()
            .zip(b)
            .map(|(&value, &effect)| value * effect)
            .sum();
    }
}

fn dense_r_times_b(
    r: &[f64],
    p: usize,
    b: &[f64],
    out: &mut [f64],
    threads: usize,
) -> Result<(), String> {
    if r.len() != p.saturating_mul(p) || b.len() != p || out.len() != p {
        return Err("dense R times b shape mismatch".to_string());
    }
    let p_blas: CblasInt = p
        .try_into()
        .map_err(|_| format!("matrix dimension {p} exceeds BLAS integer range"))?;
    let _blas_guard = BlasThreadGuard::enter(threads.max(1));
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            p_blas,
            1,
            p_blas,
            1.0,
            r.as_ptr(),
            p_blas,
            b.as_ptr(),
            1,
            0.0,
            out.as_mut_ptr(),
            1,
        );
    }
    Ok(())
}

fn compute_pip(
    alpha: &[f64],
    prior_variance: &[f64],
    l: usize,
    p: usize,
    prior_tol: f64,
) -> Vec<f64> {
    debug_assert_eq!(alpha.len(), l * p);
    debug_assert_eq!(prior_variance.len(), l);
    let mut pip = vec![0.0; p];
    for snp in 0..p {
        let log_not_pip = (0..l)
            .filter(|&effect| prior_variance[effect] > prior_tol)
            .map(|effect| (-alpha[effect * p + snp]).ln_1p())
            .sum::<f64>();
        pip[snp] = (-log_not_pip.exp_m1()).clamp(0.0, 1.0);
    }
    pip
}

#[allow(clippy::too_many_arguments)]
fn compute_elbo(
    z: &[f64],
    r: &[f64],
    p: usize,
    l: usize,
    prior: &[f64],
    alpha: &[f64],
    mu: &[f64],
    mu2: &[f64],
    b: &[f64],
    rb: &[f64],
    prior_variance: &[f64],
    z_expected: &[f64],
) -> Result<f64, String> {
    let effect_len = l
        .checked_mul(p)
        .ok_or_else(|| "ELBO dimensions overflow usize".to_string())?;
    if z.len() != p
        || r.len() != p.saturating_mul(p)
        || prior.len() != p
        || alpha.len() != effect_len
        || mu.len() != effect_len
        || mu2.len() != effect_len
        || b.len() != effect_len
        || rb.len() != effect_len
        || prior_variance.len() != l
        || z_expected.len() != p
    {
        return Err("ELBO input shape mismatch".to_string());
    }

    let mut posterior_mean = vec![0.0; p];
    for effect in 0..l {
        for snp in 0..p {
            posterior_mean[snp] += b[effect * p + snp];
        }
    }
    let z_dot_b = z
        .iter()
        .zip(&posterior_mean)
        .map(|(&score, &mean)| score * mean)
        .sum::<f64>();
    let b_dot_rb = posterior_mean
        .iter()
        .zip(z_expected)
        .map(|(&mean, &fitted)| mean * fitted)
        .sum::<f64>();
    let mut variance_correction = 0.0_f64;
    let mut kl = 0.0_f64;
    for effect in 0..l {
        let offset = effect * p;
        let effect_second_moment = alpha[offset..offset + p]
            .iter()
            .zip(&mu2[offset..offset + p])
            .map(|(&probability, &second_moment)| probability * second_moment)
            .sum::<f64>();
        let effect_mean_quad = b[offset..offset + p]
            .iter()
            .zip(&rb[offset..offset + p])
            .map(|(&mean, &fitted)| mean * fitted)
            .sum::<f64>();
        variance_correction += effect_second_moment - effect_mean_quad;

        let v = prior_variance[effect];
        if v <= V_BOUNDARY_TOL {
            continue;
        }
        let posterior_variance = v / (1.0 + v);
        for snp in 0..p {
            let probability = alpha[offset + snp];
            if probability == 0.0 {
                continue;
            }
            if prior[snp] <= 0.0 {
                return Err("positive posterior mass has zero prior weight".to_string());
            }
            let mean = mu[offset + snp];
            let categorical_kl = (probability / prior[snp]).ln();
            let normal_kl = 0.5
                * ((posterior_variance + mean * mean) / v - 1.0 + (v / posterior_variance).ln());
            kl += probability * (categorical_kl + normal_kl);
        }
    }
    let expected_log_likelihood = z_dot_b - 0.5 * (b_dot_rb + variance_correction);
    let elbo = expected_log_likelihood - kl;
    if !elbo.is_finite() {
        return Err("ELBO is non-finite".to_string());
    }
    Ok(elbo)
}

pub(crate) fn fit_susie_rss_f64(
    z: &[f64],
    r: &[f64],
    p: usize,
    prior_weights: Option<&[f64]>,
    config: &SusieRssConfig,
) -> Result<SusieRssFit, String> {
    if p == 0 || z.len() != p {
        return Err(format!(
            "z length mismatch: expected non-zero length {p}, got {}",
            z.len()
        ));
    }
    let expected_r_len = p
        .checked_mul(p)
        .ok_or_else(|| "R dimensions overflow usize".to_string())?;
    if r.len() != expected_r_len {
        return Err(format!(
            "R size mismatch: expected {expected_r_len}, got {}",
            r.len()
        ));
    }
    for (index, &score) in z.iter().enumerate() {
        if !score.is_finite() {
            return Err(format!("z[{index}] must be finite"));
        }
    }
    for row in 0..p {
        let diagonal = r[row * p + row];
        if !diagonal.is_finite() {
            return Err(format!("R[{row},{row}] must be finite"));
        }
        let diagonal_tol = 1e-10 + 1e-8 * diagonal.abs().max(1.0);
        if (diagonal - 1.0).abs() > diagonal_tol {
            return Err(format!(
                "R diagonal must be one within tolerance; R[{row},{row}]={diagonal:.12e}"
            ));
        }
        for col in row + 1..p {
            let upper = r[row * p + col];
            let lower = r[col * p + row];
            if !upper.is_finite() || !lower.is_finite() {
                return Err(format!("R[{row},{col}] and R[{col},{row}] must be finite"));
            }
            let scale = upper.abs().max(lower.abs());
            let symmetry_tol = 1e-10 + 1e-8 * scale;
            if (upper - lower).abs() > symmetry_tol {
                return Err(format!(
                    "R must be symmetric; R[{row},{col}]={upper:.12e}, R[{col},{row}]={lower:.12e}"
                ));
            }
        }
    }
    if config.l == 0 || config.max_iter == 0 {
        return Err("L and max_iter must both be positive".to_string());
    }
    if !config.tol.is_finite()
        || config.tol < 0.0
        || !config.prior_tol.is_finite()
        || config.prior_tol < 0.0
    {
        return Err("tol and prior_tol must be finite and non-negative".to_string());
    }
    let l = config.l.min(p);
    let uniform_prior;
    let prior = match prior_weights {
        Some(weights) => normalize_prior_weights(weights, p)?,
        None => {
            uniform_prior = vec![1.0 / p as f64; p];
            uniform_prior
        }
    };

    let effect_len = l
        .checked_mul(p)
        .ok_or_else(|| "L by p dimensions overflow usize".to_string())?;
    let mut fit = SusieRssFit {
        p,
        l,
        alpha: vec![0.0; effect_len],
        mu: vec![0.0; effect_len],
        mu2: vec![0.0; effect_len],
        b: vec![0.0; effect_len],
        rb: vec![0.0; effect_len],
        lbf: vec![0.0; l],
        lbf_variable: vec![0.0; effect_len],
        prior_variance: vec![0.0; l],
        posterior_mean: vec![0.0; p],
        z_expected: vec![0.0; p],
        pip: vec![0.0; p],
        // Do not reserve from a user-controlled iteration limit: very large but
        // otherwise valid limits can overflow capacity before the first sweep.
        elbo: Vec::new(),
        max_alpha_delta: Vec::new(),
        converged: false,
        n_iter: 0,
    };
    for effect in 0..l {
        fit.alpha[effect * p..(effect + 1) * p].copy_from_slice(&prior);
    }
    fit.elbo.push(compute_elbo(
        z,
        r,
        p,
        l,
        &prior,
        &fit.alpha,
        &fit.mu,
        &fit.mu2,
        &fit.b,
        &fit.rb,
        &fit.prior_variance,
        &fit.z_expected,
    )?);

    let mut z_tilde = vec![0.0; p];
    let mut b_new = vec![0.0; p];
    let mut rb_new = vec![0.0; p];
    for _ in 0..config.max_iter {
        let mut max_alpha_delta = 0.0_f64;
        for effect in 0..l {
            let offset = effect * p;
            for snp in 0..p {
                z_tilde[snp] = z[snp] - fit.z_expected[snp] + fit.rb[offset + snp];
            }
            let previous_v = if fit.prior_variance[effect] > V_BOUNDARY_TOL {
                fit.prior_variance[effect]
            } else {
                0.2
            };
            let ser = single_effect_regression(&z_tilde, &prior, previous_v)?;
            for snp in 0..p {
                b_new[snp] = ser.alpha[snp] * ser.mu[snp];
            }
            dense_r_times_b(r, p, &b_new, &mut rb_new, config.threads)?;
            for snp in 0..p {
                max_alpha_delta =
                    max_alpha_delta.max((ser.alpha[snp] - fit.alpha[offset + snp]).abs());
                fit.z_expected[snp] += rb_new[snp] - fit.rb[offset + snp];
                fit.alpha[offset + snp] = ser.alpha[snp];
                fit.mu[offset + snp] = ser.mu[snp];
                fit.mu2[offset + snp] = ser.mu2[snp];
                fit.b[offset + snp] = b_new[snp];
                fit.rb[offset + snp] = rb_new[snp];
                fit.lbf_variable[offset + snp] = ser.logbf[snp];
            }
            fit.prior_variance[effect] = ser.v;
            fit.lbf[effect] = ser.lbf;
        }
        fit.z_expected.fill(0.0);
        for effect in 0..l {
            for snp in 0..p {
                fit.z_expected[snp] += fit.rb[effect * p + snp];
            }
        }
        fit.n_iter += 1;
        fit.max_alpha_delta.push(max_alpha_delta);
        let current_elbo = compute_elbo(
            z,
            r,
            p,
            l,
            &prior,
            &fit.alpha,
            &fit.mu,
            &fit.mu2,
            &fit.b,
            &fit.rb,
            &fit.prior_variance,
            &fit.z_expected,
        )?;
        let previous_elbo = *fit.elbo.last().expect("initial ELBO is present");
        let delta = current_elbo - previous_elbo;
        let monotonicity_allowance = 1e-9 * previous_elbo.abs().max(1.0);
        if delta < -monotonicity_allowance {
            return Err(format!(
                "ELBO decreased materially at iteration {}: previous={previous_elbo:.12e}, current={current_elbo:.12e}",
                fit.n_iter
            ));
        }
        fit.elbo.push(current_elbo);
        if delta.abs() < config.tol {
            fit.converged = true;
            break;
        }
    }
    for effect in 0..l {
        for snp in 0..p {
            fit.posterior_mean[snp] += fit.b[effect * p + snp];
        }
    }
    fit.pip = compute_pip(&fit.alpha, &fit.prior_variance, l, p, config.prior_tol);
    Ok(fit)
}

#[pyfunction]
#[pyo3(signature = (
    z, r, l=10, prior_weights=None, max_iter=100,
    tol=1e-4, prior_tol=1e-9, threads=1
))]
pub fn susie_rss_f64<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray2<'py, f64>,
    l: usize,
    prior_weights: Option<PyReadonlyArray1<'py, f64>>,
    max_iter: usize,
    tol: f64,
    prior_tol: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let z_slice_storage = crate::stats_common::array_standard_cow(&z);
    let z_slice = &*z_slice_storage;
    let r_shape = r.shape();
    if r_shape.len() != 2 || r_shape[0] != r_shape[1] {
        return Err(PyValueError::new_err(format!(
            "R must be a square matrix; got shape {:?}",
            r_shape
        )));
    }
    let r_slice_storage = crate::stats_common::array_standard_cow(&r);
    let r_slice = &*r_slice_storage;
    let prior_slice = match prior_weights.as_ref() {
        Some(weights) => Some(weights.as_slice().map_err(|_| {
            PyValueError::new_err("prior_weights must be a C-contiguous float64 vector")
        })?),
        None => None,
    };
    let fit = fit_susie_rss_f64(
        z_slice,
        r_slice,
        r_shape[0],
        prior_slice,
        &SusieRssConfig {
            l,
            max_iter,
            tol,
            prior_tol,
            threads,
        },
    )
    .map_err(PyValueError::new_err)?;
    let p = fit.p;
    let l = fit.l;
    let alpha = Array2::from_shape_vec((l, p), fit.alpha)
        .map_err(|error| PyValueError::new_err(format!("invalid alpha shape: {error}")))?
        .into_pyarray(py);
    let mu = Array2::from_shape_vec((l, p), fit.mu)
        .map_err(|error| PyValueError::new_err(format!("invalid mu shape: {error}")))?
        .into_pyarray(py);
    let mu2 = Array2::from_shape_vec((l, p), fit.mu2)
        .map_err(|error| PyValueError::new_err(format!("invalid mu2 shape: {error}")))?
        .into_pyarray(py);
    let lbf_variable = Array2::from_shape_vec((l, p), fit.lbf_variable)
        .map_err(|error| PyValueError::new_err(format!("invalid lbf_variable shape: {error}")))?
        .into_pyarray(py);

    let out = PyDict::new(py);
    out.set_item("alpha", alpha)?;
    out.set_item("mu", mu)?;
    out.set_item("mu2", mu2)?;
    out.set_item("prior_variance", fit.prior_variance.into_pyarray(py))?;
    out.set_item("lbf", fit.lbf.into_pyarray(py))?;
    out.set_item("lbf_variable", lbf_variable)?;
    out.set_item("pip", fit.pip.into_pyarray(py))?;
    out.set_item("posterior_mean", fit.posterior_mean.into_pyarray(py))?;
    out.set_item("elbo", fit.elbo.into_pyarray(py))?;
    out.set_item("max_alpha_delta", fit.max_alpha_delta.into_pyarray(py))?;
    out.set_item("converged", fit.converged)?;
    out.set_item("n_iter", fit.n_iter)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (index, (&left, &right)) in lhs.iter().zip(rhs).enumerate() {
            assert!(
                (left - right).abs() <= tol,
                "mismatch at {index}: left={left:.16e}, right={right:.16e}, tol={tol:.3e}"
            );
        }
    }

    fn identity(p: usize) -> Vec<f64> {
        let mut matrix = vec![0.0; p * p];
        for i in 0..p {
            matrix[i * p + i] = 1.0;
        }
        matrix
    }

    fn fit_test(
        z: &[f64],
        r: &[f64],
        p: usize,
        l: usize,
        threads: usize,
    ) -> Result<SusieRssFit, String> {
        fit_susie_rss_f64(
            z,
            r,
            p,
            None,
            &SusieRssConfig {
                l,
                max_iter: 5,
                tol: 1e-8,
                prior_tol: 1e-9,
                threads,
            },
        )
    }

    #[test]
    fn ser_one_variant_has_analytic_solution() {
        let fit = single_effect_regression(&[3.0], &[1.0], 0.2).unwrap();
        assert!(
            (fit.v - 8.0).abs() < 1e-8,
            "expected V=8, observed V={:.16e}",
            fit.v
        );
        assert_eq!(fit.alpha, vec![1.0]);
        assert!((fit.mu[0] - 8.0 / 3.0).abs() < 1e-8);
        assert!((fit.mu2[0] - 8.0).abs() < 1e-8);
    }

    #[test]
    fn ser_noise_returns_exact_zero_effect() {
        let fit = single_effect_regression(&[0.2, -0.9], &[0.5, 0.5], 0.2).unwrap();
        assert_eq!(fit.v, 0.0);
        assert_eq!(fit.alpha, vec![0.5, 0.5]);
        assert_eq!(fit.mu, vec![0.0, 0.0]);
        assert_eq!(fit.mu2, vec![0.0, 0.0]);
    }

    #[test]
    fn ser_equal_z_preserves_nonuniform_prior() {
        let fit = single_effect_regression(&[2.0, 2.0], &[0.8, 0.2], 0.2).unwrap();
        assert_vec_close(&fit.alpha, &[0.8, 0.2], 1e-12);
    }

    #[test]
    fn ser_rejects_invalid_scores_and_priors() {
        assert!(single_effect_regression(&[], &[], 0.2)
            .unwrap_err()
            .contains("at least one"));
        assert!(single_effect_regression(&[f64::NAN], &[1.0], 0.2)
            .unwrap_err()
            .contains("finite"));
        assert!(single_effect_regression(&[f64::MAX], &[1.0], 0.2)
            .unwrap_err()
            .contains("overflow"));
        assert!(single_effect_regression(&[2.0], &[-1.0], 0.2)
            .unwrap_err()
            .contains("non-negative"));
        assert!(single_effect_regression(&[2.0, 1.0], &[0.0, 0.0], 0.2)
            .unwrap_err()
            .contains("positive finite sum"));
    }

    #[test]
    fn incremental_z_expected_matches_sum_of_cached_rb() {
        let r = vec![1.0, 0.3, 0.3, 1.0];
        let fit = fit_test(&[4.0, -3.0], &r, 2, 2, 1).unwrap();
        let mut direct = vec![0.0; 2];
        for effect in 0..fit.l {
            for snp in 0..fit.p {
                direct[snp] += fit.rb[effect * fit.p + snp];
            }
        }
        assert_vec_close(&fit.z_expected, &direct, 1e-12);
    }

    #[test]
    fn z_expected_matches_r_times_posterior_mean() {
        let r = vec![1.0, 0.3, 0.3, 1.0];
        let fit = fit_test(&[4.0, -3.0], &r, 2, 2, 1).unwrap();
        let direct = vec![
            r[0] * fit.posterior_mean[0] + r[1] * fit.posterior_mean[1],
            r[2] * fit.posterior_mean[0] + r[3] * fit.posterior_mean[1],
        ];
        assert_vec_close(&fit.z_expected, &direct, 1e-11);
    }

    #[test]
    fn one_effect_identity_fit_matches_ser() {
        let z = [4.0, 0.5, -3.0];
        let prior = vec![1.0 / 3.0; 3];
        let ser = single_effect_regression(&z, &prior, 0.2).unwrap();
        let fit = fit_test(&z, &identity(3), 3, 1, 1).unwrap();
        assert_vec_close(&fit.alpha, &ser.alpha, 1e-12);
        assert_vec_close(&fit.mu, &ser.mu, 1e-12);
        let expected_mean: Vec<f64> = ser
            .alpha
            .iter()
            .zip(&ser.mu)
            .map(|(&alpha, &mu)| alpha * mu)
            .collect();
        assert_vec_close(&fit.posterior_mean, &expected_mean, 1e-12);
    }

    #[test]
    fn dense_matvec_backend_matches_scalar_reference() {
        let r = vec![1.0, 0.2, -0.1, 0.2, 1.0, 0.4, -0.1, 0.4, 1.0];
        let b = vec![0.5, -1.0, 2.0];
        let mut expected = vec![0.0; 3];
        dense_r_times_b_scalar(&r, 3, &b, &mut expected);
        for threads in [1, 2] {
            let mut observed = vec![0.0; 3];
            dense_r_times_b(&r, 3, &b, &mut observed, threads).unwrap();
            assert_vec_close(&observed, &expected, 1e-12);
        }
    }

    #[test]
    fn pip_is_stable_near_and_at_one() {
        let alpha = vec![0.999_999_999_999, 1.0, 1e-12, 0.0];
        let pip = compute_pip(&alpha, &[1.0, 1.0], 2, 2, 1e-9);
        assert!(pip[0] >= 0.999_999_999_999 && pip[0] <= 1.0);
        assert_eq!(pip[1], 1.0);
    }

    #[test]
    fn pip_excludes_zero_variance_effects() {
        let pip = compute_pip(&[0.9, 0.1, 0.2, 0.8], &[1.0, 0.0], 2, 2, 1e-9);
        assert_vec_close(&pip, &[0.9, 0.1], 1e-15);
    }

    #[test]
    fn elbo_matches_direct_single_active_effect_calculation() {
        let z: [f64; 2] = [1.5, -0.7];
        let r: [f64; 4] = [1.0, 0.25, 0.25, 1.0];
        let prior: [f64; 2] = [0.6, 0.4];
        let alpha: [f64; 4] = [0.7, 0.3, 0.6, 0.4];
        let mu: [f64; 4] = [1.0, -0.5, 0.0, 0.0];
        let s2: f64 = 2.0 / 3.0;
        let mu2 = [s2 + 1.0, s2 + 0.25, 0.0, 0.0];
        let b = [0.7, -0.15, 0.0, 0.0];
        let rb = [0.6625, 0.025, 0.0, 0.0];
        let z_expected = [0.6625, 0.025];
        let prior_variance = [2.0, 0.0];

        let expected_log_likelihood =
            z[0] * b[0] + z[1] * b[1] - 0.5 * (alpha[0] * mu2[0] + alpha[1] * mu2[1]);
        let mut expected_kl = 0.0;
        for snp in 0..2 {
            expected_kl += alpha[snp]
                * ((alpha[snp] / prior[snp]).ln()
                    + 0.5 * ((s2 + mu[snp] * mu[snp]) / 2.0 - 1.0 + (2.0 / s2).ln()));
        }
        let expected = expected_log_likelihood - expected_kl;

        let observed = compute_elbo(
            &z,
            &r,
            2,
            2,
            &prior,
            &alpha,
            &mu,
            &mu2,
            &b,
            &rb,
            &prior_variance,
            &z_expected,
        )
        .unwrap();
        assert!((observed - expected).abs() < 1e-12);
    }

    #[test]
    fn strong_two_signal_fit_converges_with_monotone_elbo() {
        let z = [5.0, 0.2, -4.5];
        let r = vec![1.0, 0.1, 0.0, 0.1, 1.0, 0.1, 0.0, 0.1, 1.0];
        let fit = fit_susie_rss_f64(
            &z,
            &r,
            3,
            None,
            &SusieRssConfig {
                l: 2,
                max_iter: 50,
                tol: 1e-8,
                prior_tol: 1e-9,
                threads: 1,
            },
        )
        .unwrap();
        assert!(fit.converged);
        assert!(fit.n_iter < 50);
        assert_eq!(fit.elbo.len(), fit.n_iter + 1);
        assert_eq!(fit.max_alpha_delta.len(), fit.n_iter);
        for window in fit.elbo.windows(2) {
            let allowance = 1e-9 * window[0].abs().max(1.0);
            assert!(window[1] - window[0] >= -allowance);
        }
    }

    #[test]
    fn one_sweep_reports_not_converged() {
        let fit = fit_susie_rss_f64(
            &[4.0, -3.0],
            &[1.0, 0.2, 0.2, 1.0],
            2,
            None,
            &SusieRssConfig {
                l: 2,
                max_iter: 1,
                tol: 1e-8,
                prior_tol: 1e-9,
                threads: 1,
            },
        )
        .unwrap();
        assert!(!fit.converged);
        assert_eq!(fit.n_iter, 1);
        assert_eq!(fit.elbo.len(), 2);
    }

    #[test]
    fn perfectly_correlated_ld_is_supported_without_factorization() {
        let fit = fit_susie_rss_f64(
            &[4.0, 4.0],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            Some(&[0.5, 0.5]),
            &SusieRssConfig {
                l: 1,
                max_iter: 20,
                tol: 1e-8,
                prior_tol: 1e-9,
                threads: 1,
            },
        )
        .unwrap();

        assert!(fit.converged);
        assert_vec_close(&fit.alpha, &[0.5, 0.5], 1e-12);
        assert_vec_close(&fit.pip, &[0.5, 0.5], 1e-12);
        assert!((fit.posterior_mean[0] - fit.posterior_mean[1]).abs() < 1e-12);
    }

    #[test]
    fn near_singular_fit_is_consistent_across_thread_counts() {
        let rho = 0.999_f64;
        let r = vec![1.0, rho, rho * rho, rho, 1.0, rho, rho * rho, rho, 1.0];
        let z = [
            5.0 - 4.0 * rho * rho,
            5.0 * rho - 4.0 * rho,
            5.0 * rho * rho - 4.0,
        ];
        let one_thread = fit_test(&z, &r, 3, 2, 1).unwrap();
        let two_threads = fit_test(&z, &r, 3, 2, 2).unwrap();

        assert_vec_close(&one_thread.alpha, &two_threads.alpha, 1e-11);
        assert_vec_close(
            &one_thread.posterior_mean,
            &two_threads.posterior_mean,
            1e-11,
        );
        assert_vec_close(&one_thread.pip, &two_threads.pip, 1e-11);
        assert_vec_close(&one_thread.elbo, &two_threads.elbo, 1e-10);
    }

    #[test]
    fn fit_rejects_asymmetric_ld() {
        let result = fit_susie_rss_f64(
            &[1.0, 2.0],
            &[1.0, 0.2, 0.1, 1.0],
            2,
            None,
            &SusieRssConfig {
                l: 1,
                max_iter: 2,
                tol: 1e-4,
                prior_tol: 1e-9,
                threads: 1,
            },
        );
        assert!(result.unwrap_err().contains("symmetric"));
    }

    #[test]
    fn fit_rejects_nonunit_ld_diagonal() {
        let result = fit_susie_rss_f64(
            &[1.0, 2.0],
            &[1.0, 0.2, 0.2, 0.9],
            2,
            None,
            &SusieRssConfig {
                l: 1,
                max_iter: 2,
                tol: 1e-4,
                prior_tol: 1e-9,
                threads: 1,
            },
        );
        assert!(result.unwrap_err().contains("diagonal"));
    }

    #[test]
    fn fit_rejects_nonfinite_ld() {
        let result = fit_susie_rss_f64(
            &[1.0, 2.0],
            &[1.0, f64::NAN, f64::NAN, 1.0],
            2,
            None,
            &SusieRssConfig {
                l: 1,
                max_iter: 2,
                tol: 1e-4,
                prior_tol: 1e-9,
                threads: 1,
            },
        );
        assert!(result.unwrap_err().contains("finite"));
    }

    #[test]
    fn fit_rejects_invalid_dimensions_config_and_prior() {
        let config = SusieRssConfig {
            l: 1,
            max_iter: 2,
            tol: 1e-4,
            prior_tol: 1e-9,
            threads: 1,
        };
        assert!(fit_susie_rss_f64(&[], &[], 0, None, &config)
            .unwrap_err()
            .contains("z length mismatch"));
        assert!(fit_susie_rss_f64(&[1.0, 2.0], &[1.0], 2, None, &config)
            .unwrap_err()
            .contains("R size mismatch"));
        assert!(
            fit_susie_rss_f64(&[1.0, f64::INFINITY], &identity(2), 2, None, &config)
                .unwrap_err()
                .contains("finite")
        );
        assert!(
            fit_susie_rss_f64(&[1.0, 2.0], &identity(2), 2, Some(&[1.0]), &config)
                .unwrap_err()
                .contains("length mismatch")
        );

        let mut invalid_config = config.clone();
        invalid_config.l = 0;
        assert!(fit_susie_rss_f64(&[1.0], &[1.0], 1, None, &invalid_config)
            .unwrap_err()
            .contains("positive"));
        invalid_config = config.clone();
        invalid_config.max_iter = 0;
        assert!(fit_susie_rss_f64(&[1.0], &[1.0], 1, None, &invalid_config)
            .unwrap_err()
            .contains("positive"));
        invalid_config = config;
        invalid_config.tol = -1.0;
        assert!(fit_susie_rss_f64(&[1.0], &[1.0], 1, None, &invalid_config)
            .unwrap_err()
            .contains("non-negative"));
    }

    #[test]
    fn huge_max_iter_does_not_overflow_diagnostic_capacity() {
        let fit = fit_susie_rss_f64(
            &[0.2],
            &[1.0],
            1,
            None,
            &SusieRssConfig {
                l: 1,
                max_iter: usize::MAX,
                tol: 1e-4,
                prior_tol: 1e-9,
                threads: 1,
            },
        )
        .unwrap();
        assert!(fit.converged);
        assert_eq!(fit.n_iter, 1);
    }
}
