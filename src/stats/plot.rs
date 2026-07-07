use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{Beta, ContinuousCDF};

const PVALUE_EPS: f64 = f64::from_bits(1);
const QQ_SAMPLE_TRIGGER: usize = 20_000;
const QQ_SAMPLE_KEEP_HEAD: usize = 25_000;
const QQ_SAMPLE_MAX_POINTS_DEFAULT: usize = 50_000;
const QQ_BAND_KEEP_HEAD: usize = 128;
const QQ_BAND_MAX_POINTS_DEFAULT: usize = 512;

fn qq_sample_ranks_desc_with_keep_head(n: usize, limit: usize, keep_head_default: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let limit = limit.max(1).min(n);
    if n <= QQ_SAMPLE_TRIGGER || limit >= n {
        return (1..=n).rev().collect();
    }
    let keep_head = keep_head_default.min(limit).min(n);
    if keep_head >= n {
        return (1..=n).rev().collect();
    }
    let tail_target = limit.saturating_sub(keep_head);
    if tail_target == 0 {
        return (1..=keep_head).rev().collect();
    }

    let start = (keep_head + 1) as f64;
    let end = n as f64;
    let ratio = end / start;
    let denom = (tail_target - 1) as f64;
    let mut tail_ranks: Vec<usize> = Vec::with_capacity(tail_target);
    let mut prev_rank = keep_head;
    for idx in 0..tail_target {
        let t = if denom > 0.0 {
            (idx as f64) / denom
        } else {
            1.0
        };
        let raw = if ratio > 1.0 {
            start * ratio.powf(t)
        } else {
            end
        };
        let min_allowed = prev_rank.saturating_add(1);
        let remaining = tail_target.saturating_sub(idx + 1);
        let max_allowed = n.saturating_sub(remaining);
        let rank = (raw.round() as usize).clamp(min_allowed, max_allowed);
        tail_ranks.push(rank);
        prev_rank = rank;
    }

    let mut ranks: Vec<usize> = Vec::with_capacity(tail_ranks.len() + keep_head);
    for rank in tail_ranks.into_iter().rev() {
        ranks.push(rank);
    }
    for rank in (1..=keep_head).rev() {
        ranks.push(rank);
    }
    ranks
}

fn qq_sample_ranks_desc(n: usize, max_points: Option<usize>) -> Vec<usize> {
    let limit = max_points.unwrap_or(QQ_SAMPLE_MAX_POINTS_DEFAULT);
    qq_sample_ranks_desc_with_keep_head(n, limit, QQ_SAMPLE_KEEP_HEAD)
}

fn qq_band_sample_ranks_desc(
    n: usize,
    max_points: Option<usize>,
    rank_max: Option<usize>,
) -> Vec<usize> {
    let limit = max_points.unwrap_or(QQ_BAND_MAX_POINTS_DEFAULT);
    let rank_cap = rank_max.unwrap_or(n).max(1).min(n);
    qq_sample_ranks_desc_with_keep_head(rank_cap, limit, QQ_BAND_KEEP_HEAD)
}

fn qq_band_beta_logp_exact_impl(
    n: usize,
    ci: f64,
    max_points: Option<usize>,
    rank_max: Option<usize>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if n == 0 {
        return Err("n must be >= 1".to_string());
    }
    let mut ci_frac = ci;
    if !ci_frac.is_finite() {
        return Err("ci must be finite".to_string());
    }
    if ci_frac > 1.0 {
        ci_frac /= 100.0;
    }
    if !(0.0 < ci_frac && ci_frac < 1.0) {
        return Err("ci must be in (0, 1) or (0, 100)".to_string());
    }
    let alpha = 1.0 - ci_frac;
    let q_lo = alpha / 2.0;
    let q_hi = 1.0 - alpha / 2.0;
    let n_f = n as f64;

    let ranks = qq_band_sample_ranks_desc(n, max_points, rank_max);
    let rows: Result<Vec<(f64, f64, f64)>, String> = ranks
        .par_iter()
        .map(|&rank| {
            let a = rank as f64;
            let b = (n - rank + 1) as f64;
            let (p_upper, p_lower) = if a <= b {
                let beta = Beta::new(a, b).map_err(|e| format!("beta({a},{b}) init failed: {e}"))?;
                (
                    beta.inverse_cdf(q_hi).clamp(PVALUE_EPS, 1.0),
                    beta.inverse_cdf(q_lo).clamp(PVALUE_EPS, 1.0),
                )
            } else {
                let beta_flip = Beta::new(b, a)
                    .map_err(|e| format!("beta({b},{a}) init failed via symmetry: {e}"))?;
                (
                    (1.0 - beta_flip.inverse_cdf(q_lo)).clamp(PVALUE_EPS, 1.0),
                    (1.0 - beta_flip.inverse_cdf(q_hi)).clamp(PVALUE_EPS, 1.0),
                )
            };
            let exp_p = (a / (n_f + 1.0)).clamp(PVALUE_EPS, 1.0);
            Ok((-exp_p.log10(), -p_upper.log10(), -p_lower.log10()))
        })
        .collect();
    let rows = rows?;
    let mut expected_logp = Vec::with_capacity(rows.len());
    let mut ci_low_logp = Vec::with_capacity(rows.len());
    let mut ci_high_logp = Vec::with_capacity(rows.len());
    for (exp_v, low_v, high_v) in rows {
        expected_logp.push(exp_v);
        ci_low_logp.push(low_v);
        ci_high_logp.push(high_v);
    }

    Ok((expected_logp, ci_low_logp, ci_high_logp))
}

fn qq_rank_sample_zero_based_impl(n: usize, max_points: Option<usize>) -> Result<Vec<i64>, String> {
    if n == 0 {
        return Err("n must be >= 1".to_string());
    }
    Ok(qq_sample_ranks_desc(n, max_points)
        .into_iter()
        .map(|rank| (rank - 1) as i64)
        .collect())
}

#[pyfunction]
#[pyo3(signature = (n, ci=95.0, max_points=Some(QQ_BAND_MAX_POINTS_DEFAULT), rank_max=None))]
pub fn qq_band_beta_logp_exact<'py>(
    py: Python<'py>,
    n: usize,
    ci: f64,
    max_points: Option<usize>,
    rank_max: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be >= 1"));
    }
    let (expected_logp, ci_low_logp, ci_high_logp) = py
        .detach(|| qq_band_beta_logp_exact_impl(n, ci, max_points, rank_max))
        .map_err(PyRuntimeError::new_err)?;
    Ok((
        expected_logp.into_pyarray(py),
        ci_low_logp.into_pyarray(py),
        ci_high_logp.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (n, max_points=Some(QQ_SAMPLE_MAX_POINTS_DEFAULT)))]
pub fn qq_rank_sample_zero_based<'py>(
    py: Python<'py>,
    n: usize,
    max_points: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be >= 1"));
    }
    let draw_idx = py
        .detach(|| qq_rank_sample_zero_based_impl(n, max_points))
        .map_err(PyRuntimeError::new_err)?;
    Ok(draw_idx.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::{
        qq_band_beta_logp_exact_impl, qq_band_sample_ranks_desc, qq_rank_sample_zero_based_impl,
        qq_sample_ranks_desc, QQ_BAND_KEEP_HEAD, QQ_BAND_MAX_POINTS_DEFAULT, QQ_SAMPLE_KEEP_HEAD,
        QQ_SAMPLE_MAX_POINTS_DEFAULT, QQ_SAMPLE_TRIGGER,
    };

    #[test]
    fn rank_sampler_keeps_endpoints_and_desc_order() {
        let ranks = qq_sample_ranks_desc(1_000_000, Some(50_000));
        assert!(!ranks.is_empty());
        assert_eq!(ranks[0], 1_000_000);
        assert_eq!(ranks[ranks.len() - 1], 1);
        assert!(ranks.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn rank_sampler_keeps_head_then_samples_tail() {
        let n = 8_000_000usize;
        let ranks = qq_sample_ranks_desc(n, Some(QQ_SAMPLE_MAX_POINTS_DEFAULT));
        assert_eq!(ranks[0], n);
        assert_eq!(ranks[ranks.len() - 1], 1);
        let head: Vec<usize> = ranks[ranks.len() - QQ_SAMPLE_KEEP_HEAD..].to_vec();
        assert_eq!(head[0], QQ_SAMPLE_KEEP_HEAD);
        assert_eq!(head[head.len() - 1], 1);
        assert!(ranks.len() <= QQ_SAMPLE_MAX_POINTS_DEFAULT);
    }

    #[test]
    fn rank_sampler_returns_full_when_below_trigger() {
        let n = QQ_SAMPLE_TRIGGER;
        let ranks = qq_sample_ranks_desc(n, Some(QQ_SAMPLE_MAX_POINTS_DEFAULT));
        assert_eq!(ranks.len(), n);
        assert_eq!(ranks[0], n);
        assert_eq!(ranks[ranks.len() - 1], 1);
    }

    #[test]
    fn band_sampler_covers_full_range_with_small_exact_head() {
        let n = 5_000_000usize;
        let ranks = qq_band_sample_ranks_desc(n, Some(QQ_BAND_MAX_POINTS_DEFAULT), None);
        assert_eq!(ranks[0], n);
        assert_eq!(ranks[ranks.len() - 1], 1);
        assert_eq!(ranks.len(), QQ_BAND_MAX_POINTS_DEFAULT);
        let head: Vec<usize> = ranks[ranks.len() - QQ_BAND_KEEP_HEAD..].to_vec();
        assert_eq!(head[0], QQ_BAND_KEEP_HEAD);
        assert_eq!(head[head.len() - 1], 1);
    }

    #[test]
    fn band_sampler_rank_cap_limits_visible_window() {
        let n = 5_000_000usize;
        let ranks = qq_band_sample_ranks_desc(n, Some(QQ_BAND_MAX_POINTS_DEFAULT), Some(1_000_000));
        assert_eq!(ranks[0], 1_000_000);
        assert_eq!(ranks[ranks.len() - 1], 1);
        assert!(ranks.iter().all(|&r| r <= 1_000_000));
    }

    #[test]
    fn qq_band_outputs_are_finite_and_monotone() {
        let (exp, lo, hi) = qq_band_beta_logp_exact_impl(10_000, 95.0, Some(512), None).unwrap();
        assert_eq!(exp.len(), lo.len());
        assert_eq!(exp.len(), hi.len());
        assert!(exp.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(lo.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(hi.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(exp.windows(2).all(|w| w[0] <= w[1]));
        assert!(lo.iter().zip(hi.iter()).all(|(a, b)| a <= b));
    }

    #[test]
    fn zero_based_sampler_matches_rank_count() {
        let idx = qq_rank_sample_zero_based_impl(100_000, Some(50_000)).unwrap();
        assert!(!idx.is_empty());
        assert!(idx.iter().all(|v| *v >= 0));
        assert!(idx.windows(2).all(|w| w[0] > w[1]));
    }
}
