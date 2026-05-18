#![allow(dead_code)]

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StratumCounts {
    low: usize,
    mid: usize,
    high: usize,
}

#[inline]
fn percentile_stratum_sizes(n: usize) -> StratumCounts {
    let edge = n / 10;
    StratumCounts {
        low: edge,
        mid: n.saturating_sub(edge * 2),
        high: edge,
    }
}

#[inline]
fn rounded_test_count(group_n: usize, fold: usize) -> usize {
    if group_n == 0 {
        0
    } else {
        ((group_n as f64) / (fold as f64)).round() as usize
    }
    .min(group_n)
}

#[inline]
fn mark_test_indices(indices: &[usize], n_pick: usize, rng: &mut StdRng, test_mask: &mut [bool]) {
    let mut shuffled = indices.to_vec();
    shuffled.shuffle(rng);
    for &idx in shuffled.iter().take(n_pick.min(shuffled.len())) {
        test_mask[idx] = true;
    }
}

pub fn stratified_test_mask(y: &[f64], fold: usize, seed: u64) -> Result<Vec<bool>, String> {
    if fold < 2 {
        return Err("fold must be >= 2 for stratified train/test splitting.".to_string());
    }
    if y.is_empty() {
        return Err("y must contain at least one sample.".to_string());
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("y contains non-finite value at index {i}."));
        }
    }

    let n = y.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        y[a].partial_cmp(&y[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let sizes = percentile_stratum_sizes(n);
    let low_end = sizes.low;
    let mid_end = low_end + sizes.mid;
    let low_idx = &order[..low_end];
    let mid_idx = &order[low_end..mid_end];
    let high_idx = &order[mid_end..];

    let low_pick = rounded_test_count(low_idx.len(), fold);
    let mid_pick = rounded_test_count(mid_idx.len(), fold);
    let high_pick = rounded_test_count(high_idx.len(), fold);

    let mut test_mask = vec![false; n];
    let mut rng = StdRng::seed_from_u64(seed);
    mark_test_indices(low_idx, low_pick, &mut rng, &mut test_mask);
    mark_test_indices(mid_idx, mid_pick, &mut rng, &mut test_mask);
    mark_test_indices(high_idx, high_pick, &mut rng, &mut test_mask);
    Ok(test_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simulated_y(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let x = i as f64;
                0.02 * x + 0.3 * (x / 11.0).sin()
            })
            .collect()
    }

    fn count_strata_from_mask(y: &[f64], test_mask: &[bool]) -> (usize, usize, usize) {
        let mut order: Vec<usize> = (0..y.len()).collect();
        order.sort_by(|&a, &b| {
            y[a].partial_cmp(&y[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let sizes = percentile_stratum_sizes(y.len());
        let low_end = sizes.low;
        let mid_end = low_end + sizes.mid;
        let low = order[..low_end].iter().filter(|&&i| test_mask[i]).count();
        let mid = order[low_end..mid_end]
            .iter()
            .filter(|&&i| test_mask[i])
            .count();
        let high = order[mid_end..].iter().filter(|&&i| test_mask[i]).count();
        (low, mid, high)
    }

    #[test]
    fn stratified_test_mask_follows_three_tier_sampling() {
        let y = build_simulated_y(1_000);
        let test_mask = stratified_test_mask(&y, 5, 20260517).unwrap();
        let n_test = test_mask.iter().filter(|&&x| x).count();
        let (low, mid, high) = count_strata_from_mask(&y, &test_mask);

        assert_eq!(n_test, 200);
        assert_eq!(low, 20);
        assert_eq!(mid, 160);
        assert_eq!(high, 20);
    }

    #[test]
    fn stratified_test_mask_is_reproducible_by_seed() {
        let y = build_simulated_y(300);
        let a = stratified_test_mask(&y, 5, 123).unwrap();
        let b = stratified_test_mask(&y, 5, 123).unwrap();
        let c = stratified_test_mask(&y, 5, 456).unwrap();

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn stratified_test_mask_rejects_invalid_input() {
        assert!(stratified_test_mask(&[], 5, 0).is_err());
        assert!(stratified_test_mask(&[1.0, f64::NAN], 5, 0).is_err());
        assert!(stratified_test_mask(&[1.0, 2.0], 1, 0).is_err());
    }
}
