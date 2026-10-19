#[cfg(test)]
use std::collections::HashMap;

use super::types::{PairSeed, PairSeedReason, SourceMask};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg(test)]
pub(crate) struct SignedSiteVariant {
    pub(crate) marker_ids: Vec<usize>,
    pub(crate) negated: Vec<bool>,
}

#[cfg(test)]
impl SignedSiteVariant {
    pub(crate) fn lexical_key(&self) -> Vec<(usize, bool)> {
        self.marker_ids
            .iter()
            .copied()
            .zip(self.negated.iter().copied())
            .collect()
    }
}

/// Complete an unsigned site set with every 0/2 Boolean polarity.  Marker
/// IDs are canonicalized before expansion, so a repeated proposal cannot
/// create duplicate signed rules or inflate a downstream quota.
#[cfg(test)]
pub(crate) fn complete_signed_variants(marker_ids: &[usize]) -> Vec<SignedSiteVariant> {
    let mut ids = marker_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() || ids.len() > 5 {
        return Vec::new();
    }
    let n_variants = 1usize << ids.len();
    (0..n_variants)
        .map(|mask| SignedSiteVariant {
            marker_ids: ids.clone(),
            negated: (0..ids.len())
                .map(|bit| (mask & (1usize << bit)) != 0)
                .collect(),
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct CandidateOutputStatus {
    pub(crate) formal_eligible: bool,
    pub(crate) reported: bool,
}

/// Keep statistical eligibility independent from a display/output budget.
/// `ranked_indices` must already be in final-score order; a zero budget means
/// no display cap, matching GARFIELD's existing `top_rules_per_unit=0` mode.
#[cfg(test)]
pub(crate) fn apply_display_budget(
    formal_eligible: &[bool],
    ranked_indices: &[usize],
    budget: usize,
) -> Vec<CandidateOutputStatus> {
    let mut status = formal_eligible
        .iter()
        .copied()
        .map(|formal_eligible| CandidateOutputStatus {
            formal_eligible,
            reported: false,
        })
        .collect::<Vec<_>>();
    let mut reported_count = 0usize;
    for &idx in ranked_indices {
        if idx >= status.len() || !status[idx].formal_eligible {
            continue;
        }
        if budget > 0 && reported_count >= budget {
            break;
        }
        status[idx].reported = true;
        reported_count += 1;
    }
    status
}

/// Enumerate each unordered pair of distinct marker IDs exactly once.
/// Duplicate input IDs are removed before enumeration and the result is
/// lexically ordered so downstream ranks are reproducible across channels.
#[cfg(test)]
pub(crate) fn enumerate_unordered_pairs(marker_ids: &[usize]) -> Vec<[usize; 2]> {
    let mut ids = marker_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let mut pairs = Vec::with_capacity(ids.len().saturating_mul(ids.len().saturating_sub(1)) / 2);
    for lhs_idx in 0..ids.len() {
        for rhs_idx in (lhs_idx + 1)..ids.len() {
            pairs.push([ids[lhs_idx], ids[rhs_idx]]);
        }
    }
    pairs
}

#[inline]
#[cfg(test)]
pub(crate) fn accept_pair_seed(seed: PairSeed) -> bool {
    seed.marker_ids[0] != seed.marker_ids[1]
}

/// Rank exact-scored pair families and retain at most `k2` finite seeds.
/// Scores are sorted descending, with marker IDs as deterministic tie-breaks.
pub(crate) fn rank_exact_pair_seeds(
    pairs: &[[usize; 2]],
    exact_scores: &[f64],
    k2: usize,
    sources: SourceMask,
) -> Result<Vec<PairSeed>, String> {
    if pairs.len() != exact_scores.len() {
        return Err(format!(
            "pair/score length mismatch: {} vs {}",
            pairs.len(),
            exact_scores.len()
        ));
    }
    if pairs.iter().any(|pair| pair[0] == pair[1]) {
        return Err("pair seeds must contain two distinct markers".to_string());
    }

    let mut ranked = pairs
        .iter()
        .copied()
        .zip(exact_scores.iter().copied())
        .filter(|(_, score)| score.is_finite())
        .collect::<Vec<_>>();
    ranked.sort_by(|(lhs_pair, lhs_score), (rhs_pair, rhs_score)| {
        rhs_score
            .total_cmp(lhs_score)
            .then_with(|| lhs_pair.cmp(rhs_pair))
    });

    let limit = k2.min(ranked.len());
    Ok(ranked
        .into_iter()
        .take(limit)
        .enumerate()
        .map(|(rank_offset, (marker_ids, exact_score))| {
            let mut seed = PairSeed::new(
                marker_ids,
                PairSeedReason::ExactRank,
                sources,
                rank_offset + 1,
            );
            seed.exact_score = exact_score;
            seed
        })
        .collect())
}

#[inline]
#[cfg(test)]
pub(crate) fn exact_pair_seeds(
    pairs: &[[usize; 2]],
    exact_scores: &[f64],
    k2: usize,
    sources: SourceMask,
) -> Result<Vec<PairSeed>, String> {
    rank_exact_pair_seeds(pairs, exact_scores, k2, sources)
}

#[cfg(test)]
fn pair_sort_key(seed: &PairSeed) -> (bool, f64, bool, usize, [usize; 2]) {
    let finite = seed.exact_score.is_finite();
    (
        finite,
        if finite { seed.exact_score } else { 0.0 },
        matches!(seed.reason, PairSeedReason::ExactRank),
        seed.source_rank,
        seed.marker_ids,
    )
}

/// Merge exact-rank and proposal-rescue pair seeds, preserving provenance and
/// retaining each unordered pair once.  Exact-ranked seeds take precedence on
/// duplicate pairs, while a finite exact score is retained if available.
#[cfg(test)]
pub(crate) fn merge_pair_seed_sources(
    exact_ranked: &[PairSeed],
    proposal_rescue: &[PairSeed],
    k2: usize,
) -> Vec<PairSeed> {
    let mut merged = HashMap::<[usize; 2], PairSeed>::new();
    for incoming in exact_ranked.iter().chain(proposal_rescue) {
        if incoming.marker_ids[0] == incoming.marker_ids[1] {
            continue;
        }
        let mut seed = incoming.clone();
        if seed.marker_ids[0] > seed.marker_ids[1] {
            seed.marker_ids.swap(0, 1);
        }
        match merged.entry(seed.marker_ids) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(seed);
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                let current = slot.get_mut();
                current.sources = current.sources.union(seed.sources);
                current.source_rank = current.source_rank.min(seed.source_rank);
                if matches!(seed.reason, PairSeedReason::ExactRank) {
                    current.reason = PairSeedReason::ExactRank;
                }
                if seed.exact_score.is_finite()
                    && (!current.exact_score.is_finite() || seed.exact_score > current.exact_score)
                {
                    current.exact_score = seed.exact_score;
                }
            }
        }
    }
    let mut out = merged.into_values().collect::<Vec<_>>();
    out.sort_by(|lhs, rhs| {
        let lhs_key = pair_sort_key(lhs);
        let rhs_key = pair_sort_key(rhs);
        rhs_key
            .0
            .cmp(&lhs_key.0)
            .then_with(|| rhs_key.1.total_cmp(&lhs_key.1))
            .then_with(|| rhs_key.2.cmp(&lhs_key.2))
            .then_with(|| lhs_key.3.cmp(&rhs_key.3))
            .then_with(|| lhs_key.4.cmp(&rhs_key.4))
    });
    out.truncate(k2);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garfield::proposal::types::{PairSeedReason, SourceMask};

    #[test]
    fn exhaustive_pairs_are_unordered_and_lexically_stable() {
        let pairs = enumerate_unordered_pairs(&[11, 3, 17, 3]);
        assert_eq!(pairs, vec![[3, 11], [3, 17], [11, 17]],);
    }

    #[test]
    fn exhaustive_pair_count_for_64_sites_is_2016() {
        let ids = (0..64).collect::<Vec<_>>();
        let pairs = enumerate_unordered_pairs(&ids);
        assert_eq!(pairs.len(), 2016);
        assert!(pairs.windows(2).all(|window| window[0] < window[1]));
    }

    #[test]
    fn exact_pair_ranking_is_descending_with_deterministic_ties() {
        let pairs = enumerate_unordered_pairs(&[11, 3, 17]);
        let seeds = exact_pair_seeds(&pairs, &[0.5, 0.9, 0.9], 2, SourceMask::CORR).unwrap();
        assert_eq!(seeds.len(), 2);
        assert_eq!(seeds[0].marker_ids, [3, 17]);
        assert_eq!(seeds[0].source_rank, 1);
        assert_eq!(seeds[1].marker_ids, [11, 17]);
        assert_eq!(seeds[1].source_rank, 2);
        assert_eq!(seeds[0].reason, PairSeedReason::ExactRank);
    }

    #[test]
    fn proposal_rescue_is_preserved_when_merging_duplicate_pairs() {
        let exact = PairSeed {
            marker_ids: [3, 11],
            reason: PairSeedReason::ExactRank,
            sources: SourceMask::CORR,
            source_rank: 4,
            exact_score: 0.7,
        };
        let rescue = PairSeed::new([11, 3], PairSeedReason::ProposalRescue, SourceMask::RF, 2);
        let merged = merge_pair_seed_sources(&[exact], &[rescue], 10);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].marker_ids, [3, 11]);
        assert_eq!(merged[0].reason, PairSeedReason::ExactRank);
        assert!(merged[0].sources.contains(SourceMask::CORR));
        assert!(merged[0].sources.contains(SourceMask::RF));
        assert_eq!(merged[0].source_rank, 2);
        assert!(accept_pair_seed(merged[0].clone()));
    }

    #[test]
    fn order_three_unsigned_set_generates_eight_signed_variants() {
        let variants = complete_signed_variants(&[29, 3, 17]);
        assert_eq!(variants.len(), 8);
        assert_eq!(
            variants
                .iter()
                .map(SignedSiteVariant::lexical_key)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            8
        );
    }

    #[test]
    fn formal_eligibility_survives_a_display_budget() {
        let status = apply_display_budget(&[true, true, false], &[1, 0, 2], 1);
        assert_eq!(
            status,
            vec![
                CandidateOutputStatus {
                    formal_eligible: true,
                    reported: false,
                },
                CandidateOutputStatus {
                    formal_eligible: true,
                    reported: true,
                },
                CandidateOutputStatus::default(),
            ]
        );
    }
}
