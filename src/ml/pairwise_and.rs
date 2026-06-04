//! Pairwise interaction screening for feature selection.
//!
//! Two-stage pipeline with packed u64 representation:
//!   1. Marginal pre-filter via centred gain → keep top M' candidates.
//!      Each candidate is packed into u64 words as a side effect.
//!   2. For each pair among candidates, evaluate AND / OR / AND_NOT
//!      using hardware bitwise ops + popcount + ctz-based sum traversal.
//!      One pre-allocated scratch buffer reused across all pairs.
//!      Operations are gated by `PairwiseGate` to match beam-search logic.
//!   3. Final score = (1 - α) × marginal + α × interaction (normalised).
//!
//! Env knobs:
//!   - `JX_GARFIELD_PAIRWISE_AND_MAX_CANDIDATES` (default 100)
//!   - `JX_GARFIELD_PAIRWISE_AND_ALPHA`          (default 0.5)

#![allow(dead_code)]

use crate::ml::common::{topk_indices, ResponseKind};
use crate::ml::extra_trees::ExtraTreesConfig;
use std::env;

const DEFAULT_MAX_CANDIDATES: usize = 100;
const DEFAULT_ALPHA: f64 = 0.5;

#[inline]
fn parse_env_usize_fallback(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(fallback)
}

#[inline]
fn parse_env_f64_fallback(name: &str, fallback: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|&v| v.is_finite() && v >= 0.0 && v <= 1.0)
        .unwrap_or(fallback)
}

fn pairwise_and_max_candidates() -> usize {
    parse_env_usize_fallback(
        "JX_GARFIELD_PAIRWISE_AND_MAX_CANDIDATES",
        DEFAULT_MAX_CANDIDATES,
    )
}

fn pairwise_and_alpha() -> f64 {
    parse_env_f64_fallback("JX_GARFIELD_PAIRWISE_AND_ALPHA", DEFAULT_ALPHA)
}

// ---------------------------------------------------------------------------
// Packed u64 helpers
// ---------------------------------------------------------------------------

#[inline]
fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64).max(1)
}

/// Pack a dense `&[u8]` row into `Vec<u64>`.
#[inline]
fn pack_dense(row: &[u8], n_words: usize) -> Vec<u64> {
    let mut out = vec![0u64; n_words];
    for (i, &v) in row.iter().enumerate() {
        if v != 0 {
            out[i >> 6] |= 1u64 << (i & 63);
        }
    }
    out
}

/// Pre-compute per-word y sums: `y_word_sums[w] = Σ y[w*64 .. w*64+64]`.
#[inline]
fn build_y_word_sums(y: &[f64], n_samples: usize, n_words: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_words];
    for (i, &v) in y.iter().enumerate().take(n_samples) {
        out[i >> 6] += v;
    }
    out
}

/// Popcount of `bits[0..n_words]` respecting tail mask.
#[inline]
fn popcount_packed(bits: &[u64], n_words: usize, tail_mask: u64) -> usize {
    let full = n_words.saturating_sub(1);
    let mut n = 0usize;
    for w in 0..full {
        n += bits[w].count_ones() as usize;
    }
    if n_words > 0 {
        n += (bits[full] & tail_mask).count_ones() as usize;
    }
    n
}

/// Sum of y at bit-1 positions in `bits[0..n_words]`.
///
/// Uses the same "pick the sparser side" strategy as the Metal kernel:
///   - popcount ≤ 32 → traverse 1s via ctz
///   - popcount > 32 → traverse 0s, subtract from `y_word_sums[w]`
#[inline]
fn sum_y_packed(
    bits: &[u64],
    y: &[f64],
    y_word_sums: &[f64],
    n_words: usize,
    tail_mask: u64,
) -> f64 {
    let full = n_words.saturating_sub(1);
    let mut sum = 0.0f64;

    for w in 0..full {
        let word = bits[w];
        if word == 0 {
            continue;
        }
        if word.count_ones() <= 32 {
            let mut ww = word;
            let base = w << 6;
            while ww != 0 {
                let tz = ww.trailing_zeros() as usize;
                sum += y[base + tz];
                ww &= ww - 1;
            }
        } else {
            let mut miss = !word;
            let mut miss_sum = 0.0f64;
            let base = w << 6;
            while miss != 0 {
                let tz = miss.trailing_zeros() as usize;
                miss_sum += y[base + tz];
                miss &= miss - 1;
            }
            sum += y_word_sums[w] - miss_sum;
        }
    }

    if n_words > 0 {
        let word = bits[full] & tail_mask;
        if word != 0 {
            let pop = word.count_ones();
            if pop <= 32 {
                let mut ww = word;
                let base = full << 6;
                while ww != 0 {
                    let tz = ww.trailing_zeros() as usize;
                    sum += y[base + tz];
                    ww &= ww - 1;
                }
            } else {
                let valid_mask = tail_mask;
                let mut miss = (!word) & valid_mask;
                let mut miss_sum = 0.0f64;
                let base = full << 6;
                while miss != 0 {
                    let tz = miss.trailing_zeros() as usize;
                    miss_sum += y[base + tz];
                    miss &= miss - 1;
                }
                sum += y_word_sums[full] - miss_sum;
            }
        }
    }

    sum
}

/// Combined popcount + sum_y for one packed row.
#[inline]
fn count_sum_packed(
    bits: &[u64],
    y: &[f64],
    y_word_sums: &[f64],
    n_words: usize,
    tail_mask: u64,
) -> (usize, f64) {
    let n = popcount_packed(bits, n_words, tail_mask);
    let s = sum_y_packed(bits, y, y_word_sums, n_words, tail_mask);
    (n, s)
}

// ---------------------------------------------------------------------------
// Centered-gain scoring
// ---------------------------------------------------------------------------

#[inline]
fn centered_gain_from_counts(
    total_sum: f64,
    sum_hit: f64,
    n_samples: usize,
    n_hit: usize,
) -> f64 {
    let n_miss = n_samples.saturating_sub(n_hit);
    if n_hit == 0 || n_miss == 0 {
        return 0.0;
    }
    let p = (n_hit as f64) / (n_samples as f64);
    let centered_sum_hit = sum_hit - p * total_sum;
    let raw = (n_samples as f64) * centered_sum_hit * centered_sum_hit
        / ((n_hit as f64) * (n_miss as f64));
    raw.max(0.0)
}

/// All 4 AND-family subgroups for a feature pair:
///
///   `i&j`, `!i&!j`, `i&!j`, `!i&j`
///
/// By centered-gain complement symmetry this also covers the full OR family:
///   `i|j` ≡ `!(!i&!j)`, `i|!j` ≡ `!(!i&j)`, etc.
fn pair_gains_packed(
    pi: &[u64],
    pj: &[u64],
    scratch: &mut [u64],
    y: &[f64],
    y_word_sums: &[f64],
    n_words: usize,
    tail_mask: u64,
    n_samples: usize,
    total_sum: f64,
) -> (f64, f64, f64, f64) {
    // --- i & j  (both carry) ---
    for w in 0..n_words {
        scratch[w] = pi[w] & pj[w];
    }
    let (n, s) = count_sum_packed(scratch, y, y_word_sums, n_words, tail_mask);
    let gain_and = centered_gain_from_counts(total_sum, s, n_samples, n);

    // --- !i & !j  (neither carries)  ≡ i | j by complement ---
    for w in 0..n_words {
        scratch[w] = (!pi[w]) & (!pj[w]);
    }
    let (n, s) = count_sum_packed(scratch, y, y_word_sums, n_words, tail_mask);
    let gain_nij = centered_gain_from_counts(total_sum, s, n_samples, n);

    // --- i & !j  (only i carries) ---
    for w in 0..n_words {
        scratch[w] = pi[w] & !pj[w];
    }
    let (n, s) = count_sum_packed(scratch, y, y_word_sums, n_words, tail_mask);
    let gain_ij = centered_gain_from_counts(total_sum, s, n_samples, n);

    // --- !i & j  (only j carries) ---
    for w in 0..n_words {
        scratch[w] = (!pi[w]) & pj[w];
    }
    let (n, s) = count_sum_packed(scratch, y, y_word_sums, n_words, tail_mask);
    let gain_ji = centered_gain_from_counts(total_sum, s, n_samples, n);

    (gain_and, gain_nij, gain_ij, gain_ji)
}

// ---------------------------------------------------------------------------
// Packed-native entry point (used from Garfield fast path — no dense allocs)
// ---------------------------------------------------------------------------

/// Feature scores directly from packed-u64 flat matrix.
pub fn feature_scores_pairwise_and_packed(
    bits_flat: &[u64],
    row_words: usize,
    n_features: usize,
    y: &[f64],
    n_samples: usize,
) -> Vec<f64> {
    if n_features == 0 {
        return Vec::new();
    }
    let total_sum: f64 = y.iter().sum();
    let n_words = row_words; // caller already computed words_for_samples(n_samples)
    let rem = n_samples & 63;
    let tail_mask = if rem == 0 { u64::MAX } else { (1u64 << rem) - 1 };
    let y_word_sums = build_y_word_sums(y, n_samples, n_words);

    // ---- Stage 1: marginal centred gain for every feature -----------------
    let mut marginal_gain = vec![0.0f64; n_features];
    for i in 0..n_features {
        let row = &bits_flat[i * row_words..(i + 1) * row_words];
        let (n_hit, sum_hit) =
            count_sum_packed(row, y, &y_word_sums, n_words, tail_mask);
        marginal_gain[i] =
            centered_gain_from_counts(total_sum, sum_hit, n_samples, n_hit);
    }

    let max_candidates = pairwise_and_max_candidates();
    let keep_n = max_candidates.min(n_features);

    if keep_n < 2 {
        return marginal_gain;
    }

    let top = topk_indices(&marginal_gain, keep_n);
    let alpha = pairwise_and_alpha();

    // ---- Stage 2: pairwise AND / OR / AND_NOT ----------------------------
    let mut interaction_max = vec![0.0f64; n_features];
    let mut scratch = vec![0u64; n_words];

    for a in 0..keep_n {
        let i = top[a];
        let pi = &bits_flat[i * row_words..(i + 1) * row_words];
        for b in (a + 1)..keep_n {
            let j = top[b];
            let pj = &bits_flat[j * row_words..(j + 1) * row_words];

            let (gain_and, gain_nij, gain_ij, gain_ji) = pair_gains_packed(
                pi, pj, &mut scratch, y, &y_word_sums,
                n_words, tail_mask, n_samples, total_sum,
            );

            let baseline = marginal_gain[i].max(marginal_gain[j]);

            let best = (gain_and - baseline)
                .max(gain_nij - baseline)
                .max(gain_ij - baseline)
                .max(gain_ji - baseline);

            if best > interaction_max[i] {
                interaction_max[i] = best;
            }
            if best > interaction_max[j] {
                interaction_max[j] = best;
            }
        }
    }

    // ---- Stage 3: combine -------------------------------------------------
    let max_marginal = marginal_gain.iter().cloned().fold(0.0f64, f64::max);
    let max_interaction = interaction_max.iter().cloned().fold(0.0f64, f64::max);

    let scale = if max_interaction > 0.0 && max_marginal > 0.0 {
        max_marginal / max_interaction
    } else {
        1.0
    };

    let mut final_scores = vec![0.0f64; n_features];
    for i in 0..n_features {
        let interaction_scaled = interaction_max[i] * scale;
        final_scores[i] =
            (1.0 - alpha) * marginal_gain[i] + alpha * interaction_scaled;
    }

    final_scores
}

// ---------------------------------------------------------------------------
// Flat-dense entry point: single allocation, no per-row Vec
// ---------------------------------------------------------------------------

/// Feature scores from flat row-major dense data.
///
/// `data` is `[row0_s0, row0_s1, ..., row1_s0, ...]` with `row_stride` bytes
/// per row.  Stage 1 reads directly from flat; Stage 2 packs only the top-M'
/// candidates into u64.
pub fn feature_scores_pairwise_and_flat(
    data: &[u8],
    row_stride: usize,
    n_features: usize,
    y: &[f64],
    n_samples: usize,
) -> Vec<f64> {
    if n_features == 0 {
        return Vec::new();
    }
    let total_sum: f64 = y.iter().sum();

    // ---- Stage 1: marginal from flat dense (zero alloc) --------------------
    let mut marginal_gain = vec![0.0f64; n_features];
    for i in 0..n_features {
        let row = &data[i * row_stride..][..n_samples];
        let (n_hit, sum_hit) = count_sum_y_dense(row, y);
        marginal_gain[i] =
            centered_gain_from_counts(total_sum, sum_hit, n_samples, n_hit);
    }

    let max_candidates = pairwise_and_max_candidates();
    let keep_n = max_candidates.min(n_features);

    if keep_n < 2 {
        return marginal_gain;
    }

    let top = topk_indices(&marginal_gain, keep_n);
    let alpha = pairwise_and_alpha();

    // ---- Pack only top-M' for Stage 2 --------------------------------------
    let n_words = words_for_samples(n_samples);
    let rem = n_samples & 63;
    let tail_mask = if rem == 0 { u64::MAX } else { (1u64 << rem) - 1 };
    let y_word_sums = build_y_word_sums(y, n_samples, n_words);

    let mut packed_top: Vec<Vec<u64>> = Vec::with_capacity(keep_n);
    for &fi in top.iter() {
        let row = &data[fi * row_stride..][..n_samples];
        packed_top.push(pack_dense(row, n_words));
    }

    // ---- Stage 2: pairwise via packed u64 ----------------------------------
    let mut interaction_max = vec![0.0f64; n_features];
    let mut scratch = vec![0u64; n_words];

    for a in 0..keep_n {
        let i = top[a];
        let pi = &packed_top[a];
        for b in (a + 1)..keep_n {
            let j = top[b];
            let pj = &packed_top[b];

            let (gain_and, gain_nij, gain_ij, gain_ji) = pair_gains_packed(
                pi, pj, &mut scratch, y, &y_word_sums,
                n_words, tail_mask, n_samples, total_sum,
            );

            let baseline = marginal_gain[i].max(marginal_gain[j]);
            let best = (gain_and - baseline)
                .max(gain_nij - baseline)
                .max(gain_ij - baseline)
                .max(gain_ji - baseline);

            if best > interaction_max[i] { interaction_max[i] = best; }
            if best > interaction_max[j] { interaction_max[j] = best; }
        }
    }

    // ---- Stage 3: combine --------------------------------------------------
    let max_marginal = marginal_gain.iter().cloned().fold(0.0f64, f64::max);
    let max_interaction = interaction_max.iter().cloned().fold(0.0f64, f64::max);
    let scale = if max_interaction > 0.0 && max_marginal > 0.0 {
        max_marginal / max_interaction
    } else {
        1.0
    };

    let mut final_scores = vec![0.0f64; n_features];
    for i in 0..n_features {
        let interaction_scaled = interaction_max[i] * scale;
        final_scores[i] =
            (1.0 - alpha) * marginal_gain[i] + alpha * interaction_scaled;
    }
    final_scores
}

// ---------------------------------------------------------------------------
// Dense entry point (for backward compat / non-Garfield callers)
// ---------------------------------------------------------------------------

/// Dense count+sum (branch per sample).  Faster than pack+count for Stage 1
/// because it is a single sequential scan with simple writes.
#[inline]
fn count_sum_y_dense(row: &[u8], y: &[f64]) -> (usize, f64) {
    let mut n = 0usize;
    let mut s = 0.0f64;
    for (i, &v) in row.iter().enumerate() {
        if v != 0 {
            n += 1;
            s += y[i];
        }
    }
    (n, s)
}

pub fn feature_scores_pairwise_and_grouped(
    x_rows: &[Vec<u8>],
    y: &[f64],
    _response: ResponseKind,
    _cfg: ExtraTreesConfig,
    _feature_group_ids: Option<&[usize]>,
) -> Vec<f64> {
    let n_features = x_rows.len();
    if n_features == 0 {
        return Vec::new();
    }
    let n_samples = y.len();
    let total_sum: f64 = y.iter().sum();

    // ---- Stage 1: marginal via dense scan (faster than pack+count) ---------
    let mut marginal_gain = vec![0.0f64; n_features];
    for i in 0..n_features {
        let (n_hit, sum_hit) = count_sum_y_dense(&x_rows[i], y);
        marginal_gain[i] =
            centered_gain_from_counts(total_sum, sum_hit, n_samples, n_hit);
    }

    let max_candidates = pairwise_and_max_candidates();
    let keep_n = max_candidates.min(n_features);

    if keep_n < 2 {
        return marginal_gain;
    }

    let top = topk_indices(&marginal_gain, keep_n);
    let alpha = pairwise_and_alpha();

    // ---- Pack only the top-M' features for Stage 2 -------------------------
    let n_words = words_for_samples(n_samples);
    let rem = n_samples & 63;
    let tail_mask = if rem == 0 { u64::MAX } else { (1u64 << rem) - 1 };
    let y_word_sums = build_y_word_sums(y, n_samples, n_words);

    let mut packed_top: Vec<Vec<u64>> = Vec::with_capacity(keep_n);
    for &fi in top.iter() {
        packed_top.push(pack_dense(&x_rows[fi], n_words));
    }

    // ---- Stage 2: pairwise AND / OR / AND_NOT via packed u64 ---------------
    let mut interaction_max = vec![0.0f64; n_features];
    let mut scratch = vec![0u64; n_words];

    for a in 0..keep_n {
        let i = top[a];
        let pi = &packed_top[a];
        for b in (a + 1)..keep_n {
            let j = top[b];
            let pj = &packed_top[b];

            let (gain_and, gain_nij, gain_ij, gain_ji) = pair_gains_packed(
                pi, pj, &mut scratch, y, &y_word_sums,
                n_words, tail_mask, n_samples, total_sum,
            );

            let baseline = marginal_gain[i].max(marginal_gain[j]);

            let best = (gain_and - baseline)
                .max(gain_nij - baseline)
                .max(gain_ij - baseline)
                .max(gain_ji - baseline);

            if best > interaction_max[i] {
                interaction_max[i] = best;
            }
            if best > interaction_max[j] {
                interaction_max[j] = best;
            }
        }
    }

    // ---- Stage 3: combine marginal + interaction scores --------------------
    let max_marginal = marginal_gain
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);
    let max_interaction = interaction_max
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);

    let scale = if max_interaction > 0.0 && max_marginal > 0.0 {
        max_marginal / max_interaction
    } else {
        1.0
    };

    let mut final_scores = vec![0.0f64; n_features];
    for i in 0..n_features {
        let interaction_scaled = interaction_max[i] * scale;
        final_scores[i] =
            (1.0 - alpha) * marginal_gain[i] + alpha * interaction_scaled;
    }

    final_scores
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> ExtraTreesConfig {
        ExtraTreesConfig {
            n_estimators: 1,
            max_depth: 3,
            min_samples_leaf: 1,
            min_samples_split: 2,
            bootstrap: false,
            feature_subsample: 0.0,
            seed: 0,
            allow_parallel: false,
        }
    }

    #[test]
    fn test_centered_gain_basic() {
        let total = 6.0;
        let gain = centered_gain_from_counts(total, 9.0, 4, 2);
        assert!((gain - 36.0).abs() < 1e-12);
    }

    #[test]
    fn test_pack_and_count_roundtrip() {
        let row: Vec<u8> = (0..130)
            .map(|i| if i % 3 == 0 { 1u8 } else { 0u8 })
            .collect();
        let y: Vec<f64> = (0..130).map(|i| i as f64).collect();
        let n_samples = 130;
        let n_words = words_for_samples(n_samples);
        let yws = build_y_word_sums(&y, n_samples, n_words);
        let rem = n_samples & 63;
        let tail_mask = if rem == 0 { u64::MAX } else { (1u64 << rem) - 1 };

        let packed = pack_dense(&row, n_words);
        let (n, s) = count_sum_packed(&packed, &y, &yws, n_words, tail_mask);

        // Manual count
        let mut n_exp = 0usize;
        let mut s_exp = 0.0f64;
        for (i, &v) in row.iter().enumerate() {
            if v != 0 {
                n_exp += 1;
                s_exp += y[i];
            }
        }
        assert_eq!(n, n_exp);
        assert!((s - s_exp).abs() < 1e-12);
    }

    #[test]
    fn test_pairwise_packed_matches_dense() {
        // Verify that packed pairwise gives the same gains as the dense approach.
        let y = vec![10.0, -5.0, -5.0, 10.0, -5.0, -5.0, 10.0, -5.0];
        let feat_a = vec![1u8, 0, 0, 1, 0, 0, 1, 0];
        let feat_b = vec![1u8, 1, 0, 0, 0, 0, 1, 1];
        let x_rows = vec![feat_a, feat_b];
        let cfg = test_cfg();
        let scores_packed = feature_scores_pairwise_and_grouped(
            &x_rows, &y, ResponseKind::Continuous, cfg, None,
        );

        assert_eq!(scores_packed.len(), 2);
        assert!(scores_packed[0].is_finite());
        assert!(scores_packed[1].is_finite());
        // At least one should be positive (interaction detected)
        assert!(
            scores_packed[0] > 0.0 || scores_packed[1] > 0.0,
            "expected interaction signal"
        );
    }

    #[test]
    fn test_pairwise_uses_and_or_and_not() {
        let y = vec![3.0, 3.0, -1.0, -1.0, -1.0, -1.0];
        let a = vec![1u8, 1, 0, 0, 0, 0];
        let b = vec![0u8, 1, 0, 0, 0, 0];
        let x_rows = vec![a.clone(), b.clone()];

        let cfg = test_cfg();
        let scores = feature_scores_pairwise_and_grouped(
            &x_rows, &y, ResponseKind::Continuous, cfg, None,
        );

        assert!(scores.len() == 2);
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
    }

    #[test]
    fn test_few_features_returns_marginals() {
        let y = vec![1.0, 2.0, 3.0];
        let x_rows = vec![vec![1u8, 0, 1]];
        let cfg = test_cfg();
        let scores = feature_scores_pairwise_and_grouped(
            &x_rows, &y, ResponseKind::Continuous, cfg, None,
        );
        assert_eq!(scores.len(), 1);
        assert!(scores[0].is_finite());
    }

    #[test]
    fn test_empty_input() {
        let y = vec![1.0, 2.0];
        let cfg = test_cfg();
        let scores = feature_scores_pairwise_and_grouped(
            &[], &y, ResponseKind::Continuous, cfg, None,
        );
        assert!(scores.is_empty());
    }

    #[test]
    fn test_partial_tail_word() {
        // 70 samples = 1 full word (64) + 1 partial (6)
        let n_samples = 70;
        let row: Vec<u8> = (0..n_samples).map(|i| (i % 2) as u8).collect();
        let y: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();
        let n_words = words_for_samples(n_samples);
        let yws = build_y_word_sums(&y, n_samples, n_words);
        let rem = n_samples & 63;
        let tail_mask = if rem == 0 { u64::MAX } else { (1u64 << rem) - 1 };

        let packed = pack_dense(&row, n_words);
        let (n, s) = count_sum_packed(&packed, &y, &yws, n_words, tail_mask);

        let mut n_exp = 0usize;
        let mut s_exp = 0.0f64;
        for (i, &v) in row.iter().enumerate() {
            if v != 0 {
                n_exp += 1;
                s_exp += y[i];
            }
        }
        assert_eq!(n, n_exp);
        assert!((s - s_exp).abs() < 1e-12);
    }
}
