#![allow(dead_code)]

use crate::bitwise::popcount;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuousRuleScore {
    pub score: f64,
    pub raw_score: f64,
    pub mean_hit: f64,
    pub mean_miss: f64,
    pub support_frac: f64,
    pub n_hit: usize,
    pub n_miss: usize,
}

#[inline]
fn require_n_valid(n: usize, y_len: usize, bit_words: usize, ctx: &str) -> Result<(), String> {
    if n == 0 {
        return Err(format!("{ctx}: n_samples must be > 0"));
    }
    if n > y_len {
        return Err(format!("{ctx}: n_samples={} exceeds y length={}", n, y_len));
    }
    let bit_cap = bit_words.saturating_mul(64);
    if n > bit_cap {
        return Err(format!(
            "{ctx}: n_samples={} exceeds bit capacity={} ({} words)",
            n, bit_cap, bit_words
        ));
    }
    Ok(())
}

#[inline]
pub fn validate_continuous_y(y: &[f64], n_samples: usize, ctx: &str) -> Result<(), String> {
    if y.len() < n_samples {
        return Err(format!(
            "{ctx}: n_samples={} exceeds y length={}",
            n_samples,
            y.len()
        ));
    }
    if let Some((idx, bad)) = y
        .iter()
        .take(n_samples)
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(format!(
            "{ctx}: y must be finite for continuous mode; found y[{idx}]={bad}"
        ));
    }
    Ok(())
}

#[inline]
fn y_moments(y: &[f64], n_samples: usize) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut ss = 0.0_f64;
    for &v in y.iter().take(n_samples) {
        sum += v;
        ss += v * v;
    }
    (sum, ss)
}

#[inline]
fn moments_y_where_bit1(bits: &[u64], y: &[f64], n_samples: usize) -> (usize, f64, f64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;

    let mut n1 = 0usize;
    let mut s1 = 0.0_f64;
    let mut ss1 = 0.0_f64;

    for (w_idx, &word) in bits.iter().take(full_words).enumerate() {
        n1 += word.count_ones() as usize;
        let mut w = word;
        let base = w_idx << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            let v = y[base + tz];
            s1 += v;
            ss1 += v * v;
            w &= w - 1;
        }
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let mut w = bits[full_words] & mask;
        n1 += w.count_ones() as usize;
        let base = full_words << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            let v = y[base + tz];
            s1 += v;
            ss1 += v * v;
            w &= w - 1;
        }
    }

    (n1, s1, ss1)
}

#[inline]
pub fn support_size_packed(bits: &[u64], n_samples: usize) -> usize {
    if n_samples == 0 || bits.is_empty() {
        return 0usize;
    }
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let mut n1 = if full_words > 0 {
        popcount(&bits[..full_words]) as usize
    } else {
        0usize
    };
    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        n1 += (bits[full_words] & mask).count_ones() as usize;
    }
    n1
}

pub fn score_cont_weighted_mean_diff_packed(
    y: &[f64],
    bits: &[u64],
    n_samples: usize,
) -> ContinuousRuleScore {
    const CTX: &str = "score_cont_weighted_mean_diff_packed";
    if require_n_valid(n_samples, y.len(), bits.len(), CTX).is_err() {
        return ContinuousRuleScore {
            score: f64::NEG_INFINITY,
            raw_score: f64::NEG_INFINITY,
            mean_hit: f64::NAN,
            mean_miss: f64::NAN,
            support_frac: f64::NAN,
            n_hit: 0,
            n_miss: 0,
        };
    }
    if validate_continuous_y(y, n_samples, CTX).is_err() {
        return ContinuousRuleScore {
            score: f64::NEG_INFINITY,
            raw_score: f64::NEG_INFINITY,
            mean_hit: f64::NAN,
            mean_miss: f64::NAN,
            support_frac: f64::NAN,
            n_hit: 0,
            n_miss: 0,
        };
    }

    let (sum_all, ss_all) = y_moments(y, n_samples);
    let (n_hit, sum_hit, ss_hit) = moments_y_where_bit1(bits, y, n_samples);
    let n_miss = n_samples.saturating_sub(n_hit);
    if n_hit == 0 || n_miss == 0 {
        return ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: f64::NAN,
            mean_miss: f64::NAN,
            support_frac: (n_hit as f64) / (n_samples as f64),
            n_hit,
            n_miss,
        };
    }

    let mean_hit = sum_hit / (n_hit as f64);
    let sum_miss = sum_all - sum_hit;
    let ss_miss = ss_all - ss_hit;
    let mean_miss = sum_miss / (n_miss as f64);
    let p = (n_hit as f64) / (n_samples as f64);
    let beta_abs = (mean_hit - mean_miss).abs();
    let sse_hit = (ss_hit - (sum_hit * sum_hit) / (n_hit as f64)).max(0.0);
    let sse_miss = (ss_miss - (sum_miss * sum_miss) / (n_miss as f64)).max(0.0);
    let sigma2 = if n_samples > 2 {
        (sse_hit + sse_miss) / ((n_samples - 2) as f64)
    } else {
        f64::NAN
    };
    let scale = if sigma2.is_finite() && sigma2 > 0.0 {
        sigma2.sqrt()
    } else {
        1.0
    };
    let raw = beta_abs * (p * (1.0 - p)).sqrt() / scale;
    ContinuousRuleScore {
        score: raw,
        raw_score: raw,
        mean_hit,
        mean_miss,
        support_frac: p,
        n_hit,
        n_miss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_cont_weighted_mean_diff_basic() {
        let y = vec![5.0, 4.0, -1.0, -2.0];
        let bits = vec![0b0011_u64];
        let sc = score_cont_weighted_mean_diff_packed(&y, &bits, y.len());
        assert_eq!(sc.n_hit, 2);
        assert_eq!(sc.n_miss, 2);
        assert!((sc.mean_hit - 4.5).abs() < 1e-12);
        assert!((sc.mean_miss + 1.5).abs() < 1e-12);
        assert!((sc.raw_score - 4.242640687119285).abs() < 1e-12);
        assert!((sc.score - 4.242640687119285).abs() < 1e-12);
    }

    #[test]
    fn test_score_cont_weighted_mean_diff_extreme_support_is_zero() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let bits_all_zero = vec![0u64];
        let bits_all_one = vec![0b1111_u64];
        let sc0 = score_cont_weighted_mean_diff_packed(&y, &bits_all_zero, y.len());
        let sc1 = score_cont_weighted_mean_diff_packed(&y, &bits_all_one, y.len());
        assert_eq!(sc0.score, 0.0);
        assert_eq!(sc1.score, 0.0);
        assert_eq!(sc0.n_hit, 0);
        assert_eq!(sc1.n_miss, 0);
    }

    #[test]
    fn test_support_size_packed_masks_tail() {
        let bits = vec![u64::MAX];
        assert_eq!(support_size_packed(&bits, 5), 5);
        assert_eq!(support_size_packed(&bits, 64), 64);
    }
}
