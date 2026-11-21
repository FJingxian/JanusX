//! Independent pair-seeded order-3 GARFIELD oracle.
//!
//! This module is intentionally diagnostic-only.  It starts from a caller
//! supplied pair seed list, canonicalizes every extension to `i < j < k`,
//! evaluates all eight signed AND interpretations exactly once per triple,
//! and keeps a bounded raw-score Top-K.  It does not apply Beam retention,
//! parent-gain pruning, support gates, or formal maxT thresholds.

use crate::bstats::{tail_mask, words_for_samples};
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::score::{
    and3_popcount_sum_y_where_all1_with_lookup, score_cont_centered_gain_from_sum_and_n_hit,
    validate_continuous_y, PackedYSumLookup,
};

/// A canonical pair used as a seed for triple refinement.
///
/// `pair_rank` is supplied by the caller (normally the rank in the AllPair
/// Top-K list) and is carried to the output for search diagnostics.  Pair
/// orientation is canonicalized to `first < second` before scanning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PairTripleSeed {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) pair_rank: usize,
}

/// Best signed interpretation for one unordered triple.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PairTripleCandidate {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) third: usize,
    pub(crate) polarity: u8,
    pub(crate) raw_score: f64,
    pub(crate) support: usize,
    pub(crate) owner_first: usize,
    pub(crate) owner_second: usize,
    pub(crate) owner_pair_rank: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PairTripleScanResult {
    pub(crate) candidates: Vec<PairTripleCandidate>,
    pub(crate) targets: Vec<PairTripleTargetDiagnostic>,
    pub(crate) pair_seeds_used: usize,
    pub(crate) expansion_attempts: usize,
    pub(crate) triples_evaluated: usize,
    pub(crate) polarity_evaluated: usize,
    pub(crate) n_rows: usize,
    pub(crate) n_samples: usize,
}

/// Optional per-rule diagnostics returned without retaining the full scan.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PairTripleTargetDiagnostic {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) third: usize,
    pub(crate) evaluated: bool,
    pub(crate) rank: Option<usize>,
    pub(crate) polarity: Option<u8>,
    pub(crate) raw_score: Option<f64>,
    pub(crate) support: Option<usize>,
    pub(crate) owner_first: Option<usize>,
    pub(crate) owner_second: Option<usize>,
    pub(crate) owner_pair_rank: Option<usize>,
}

#[inline]
fn candidate_cmp(a: &PairTripleCandidate, b: &PairTripleCandidate) -> Ordering {
    a.raw_score
        .total_cmp(&b.raw_score)
        // Lower canonical rule indices are deterministic winners for exact
        // raw-score ties; polarity is the final stable tie breaker.
        .then_with(|| b.first.cmp(&a.first))
        .then_with(|| b.second.cmp(&a.second))
        .then_with(|| b.third.cmp(&a.third))
        .then_with(|| b.polarity.cmp(&a.polarity))
}

#[derive(Clone, Debug)]
struct HeapEntry(PairTripleCandidate);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.first == other.0.first
            && self.0.second == other.0.second
            && self.0.third == other.0.third
            && self.0.polarity == other.0.polarity
            && self.0.support == other.0.support
            && self.0.owner_first == other.0.owner_first
            && self.0.owner_second == other.0.owner_second
            && self.0.owner_pair_rank == other.0.owner_pair_rank
            && self.0.raw_score.to_bits() == other.0.raw_score.to_bits()
    }
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        candidate_cmp(&self.0, &other.0)
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn push_top_k(
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    candidate: PairTripleCandidate,
    top_k: usize,
) {
    if top_k == 0 {
        return;
    }
    let item = Reverse(HeapEntry(candidate));
    if heap.len() < top_k {
        heap.push(item);
        return;
    }
    let replace = heap
        .peek()
        .map(|worst| item.0.cmp(&worst.0) == Ordering::Greater)
        .unwrap_or(true);
    if replace {
        let _ = heap.pop();
        heap.push(item);
    }
}

#[inline]
fn row_slice<'a>(bits_flat: &'a [u64], row_words: usize, row: usize, words: usize) -> &'a [u64] {
    let start = row * row_words;
    &bits_flat[start..start + words]
}

fn validate_scan_inputs(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    y: &[f64],
    n_samples: usize,
) -> Result<usize, String> {
    if n_rows == 0 {
        return Err("garfield_pair_triple_refine: n_rows must be > 0".to_string());
    }
    if n_samples == 0 {
        return Err("garfield_pair_triple_refine: n_samples must be > 0".to_string());
    }
    if y.len() < n_samples {
        return Err(format!(
            "garfield_pair_triple_refine: y length={} smaller than n_samples={n_samples}",
            y.len()
        ));
    }
    validate_continuous_y(y, n_samples, "garfield_pair_triple_refine")?;
    let words = words_for_samples(n_samples).max(1);
    if row_words < words {
        return Err(format!(
            "garfield_pair_triple_refine: row_words={row_words} smaller than required {words}"
        ));
    }
    let required = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| "garfield_pair_triple_refine: packed matrix size overflow".to_string())?;
    if bits_flat.len() < required {
        return Err(format!(
            "garfield_pair_triple_refine: bits length={} smaller than n_rows*row_words={required}",
            bits_flat.len()
        ));
    }
    Ok(words)
}

fn canonicalize_seeds(
    seeds: &[PairTripleSeed],
    n_rows: usize,
) -> Result<Vec<PairTripleSeed>, String> {
    let mut unique = HashMap::<(usize, usize), PairTripleSeed>::with_capacity(seeds.len());
    for &seed in seeds {
        if seed.first >= n_rows || seed.second >= n_rows {
            return Err(format!(
                "garfield_pair_triple_refine: pair ({}, {}) is outside n_rows={n_rows}",
                seed.first, seed.second
            ));
        }
        if seed.first == seed.second {
            return Err(format!(
                "garfield_pair_triple_refine: pair ({}, {}) must contain two distinct rows",
                seed.first, seed.second
            ));
        }
        let (first, second) = if seed.first < seed.second {
            (seed.first, seed.second)
        } else {
            (seed.second, seed.first)
        };
        let canonical = PairTripleSeed {
            first,
            second,
            pair_rank: seed.pair_rank,
        };
        unique
            .entry((first, second))
            .and_modify(|current| {
                if canonical.pair_rank < current.pair_rank {
                    *current = canonical;
                }
            })
            .or_insert(canonical);
    }
    let mut out = unique.into_values().collect::<Vec<_>>();
    out.sort_unstable_by_key(|seed| (seed.first, seed.second));
    Ok(out)
}

fn canonicalize_targets(
    targets: &[(usize, usize, usize)],
    n_rows: usize,
) -> Result<Vec<(usize, usize, usize)>, String> {
    let mut unique = targets
        .iter()
        .map(|&(first, second, third)| {
            if first >= n_rows || second >= n_rows || third >= n_rows {
                return Err(format!(
                    "garfield_pair_triple_refine: target ({first}, {second}, {third}) is outside n_rows={n_rows}"
                ));
            }
            if first == second || first == third || second == third {
                return Err(format!(
                    "garfield_pair_triple_refine: target ({first}, {second}, {third}) must contain three distinct rows"
                ));
            }
            let mut indices = [first, second, third];
            indices.sort_unstable();
            Ok((indices[0], indices[1], indices[2]))
        })
        .collect::<Result<Vec<_>, _>>()?;
    unique.sort_unstable();
    unique.dedup();
    Ok(unique)
}

#[inline]
fn seeded_owner(
    first: usize,
    second: usize,
    third: usize,
    seed_map: &HashMap<(usize, usize), PairTripleSeed>,
) -> Option<PairTripleSeed> {
    [(first, second), (first, third), (second, third)]
        .into_iter()
        .filter_map(|pair| seed_map.get(&pair).copied())
        .min_by_key(|owner| (owner.first, owner.second))
}

fn best_triple_candidate(
    bits_flat: &[u64],
    negated: &[u64],
    row_words: usize,
    needed_words: usize,
    first: usize,
    second: usize,
    third: usize,
    owner: PairTripleSeed,
    y: &[f64],
    n_samples: usize,
    lookup: &PackedYSumLookup,
    total_sum: f64,
) -> PairTripleCandidate {
    let first_pos = row_slice(bits_flat, row_words, first, needed_words);
    let second_pos = row_slice(bits_flat, row_words, second, needed_words);
    let third_pos = row_slice(bits_flat, row_words, third, needed_words);
    let first_neg = row_slice(negated, needed_words, first, needed_words);
    let second_neg = row_slice(negated, needed_words, second, needed_words);
    let third_neg = row_slice(negated, needed_words, third, needed_words);
    let rows = [
        (first_pos, first_neg),
        (second_pos, second_neg),
        (third_pos, third_neg),
    ];

    let mut best: Option<PairTripleCandidate> = None;
    for polarity in 0u8..8 {
        let first_bits = if polarity & 1 != 0 {
            rows[0].1
        } else {
            rows[0].0
        };
        let second_bits = if polarity & 2 != 0 {
            rows[1].1
        } else {
            rows[1].0
        };
        let third_bits = if polarity & 4 != 0 {
            rows[2].1
        } else {
            rows[2].0
        };
        let (n_hit, sum_hit) = and3_popcount_sum_y_where_all1_with_lookup(
            first_bits,
            second_bits,
            third_bits,
            y,
            n_samples,
            lookup,
        );
        let score = score_cont_centered_gain_from_sum_and_n_hit(
            total_sum,
            sum_hit,
            n_samples,
            n_hit as usize,
        );
        let candidate = PairTripleCandidate {
            first,
            second,
            third,
            polarity,
            raw_score: score.raw_score,
            support: n_hit as usize,
            owner_first: owner.first,
            owner_second: owner.second,
            owner_pair_rank: owner.pair_rank,
        };
        if best
            .as_ref()
            .map(|current| candidate_cmp(&candidate, current) == Ordering::Greater)
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }
    best.expect("triple has eight polarity variants")
}

/// Enumerate triples reachable from a pair seed list.
///
/// A triple is owned by the lexicographically smallest of its seeded
/// immediate pairs.  Thus a triple generated from `AB + C`, `AC + B`, and
/// `BC + A` is scored once, while a triple with only one seeded subpair is
/// still scored.  Every scored triple evaluates all eight polarity masks and
/// keeps its best signed interpretation.  No statistical or search pruning
/// is applied.
pub(crate) fn scan_pair_seeded_triples_continuous_packed(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    y: &[f64],
    n_samples: usize,
    seeds: &[PairTripleSeed],
    targets: &[(usize, usize, usize)],
    top_k: usize,
) -> Result<PairTripleScanResult, String> {
    let needed_words = validate_scan_inputs(bits_flat, row_words, n_rows, y, n_samples)?;
    let canonical_seeds = canonicalize_seeds(seeds, n_rows)?;
    let canonical_targets = canonicalize_targets(targets, n_rows)?;
    let seed_map = canonical_seeds
        .iter()
        .map(|seed| ((seed.first, seed.second), *seed))
        .collect::<HashMap<_, _>>();
    let tail = tail_mask(n_samples);

    let total_words = n_rows
        .checked_mul(needed_words)
        .ok_or_else(|| "garfield_pair_triple_refine: complement size overflow".to_string())?;
    let mut negated = vec![0u64; total_words];
    for row in 0..n_rows {
        let src = row_slice(bits_flat, row_words, row, needed_words);
        let dst_start = row * needed_words;
        let dst = &mut negated[dst_start..dst_start + needed_words];
        for (out, &word) in dst.iter_mut().zip(src.iter()) {
            *out = !word;
        }
        if let Some(mask) = tail {
            if let Some(last) = dst.last_mut() {
                *last &= mask;
            }
        }
    }

    let lookup = PackedYSumLookup::build(y, n_samples)?;
    let total_sum = y.iter().take(n_samples).copied().sum::<f64>();
    let target_candidates = canonical_targets
        .iter()
        .map(|&(first, second, third)| {
            seeded_owner(first, second, third, &seed_map).map(|owner| {
                best_triple_candidate(
                    bits_flat,
                    &negated,
                    row_words,
                    needed_words,
                    first,
                    second,
                    third,
                    owner,
                    y,
                    n_samples,
                    &lookup,
                    total_sum,
                )
            })
        })
        .collect::<Vec<_>>();
    let mut target_better_counts = vec![0usize; target_candidates.len()];
    let mut heap = BinaryHeap::<Reverse<HeapEntry>>::with_capacity(top_k.min(1024));
    let mut expansion_attempts = 0usize;
    let mut triples_evaluated = 0usize;
    let mut polarity_evaluated = 0usize;

    for seed in &canonical_seeds {
        for third in 0..n_rows {
            if third == seed.first || third == seed.second {
                continue;
            }
            expansion_attempts = expansion_attempts.checked_add(1).ok_or_else(|| {
                "garfield_pair_triple_refine: expansion counter overflow".to_string()
            })?;
            let mut indices = [seed.first, seed.second, third];
            indices.sort_unstable();
            let [first, second, third] = indices;

            // The lexicographically smallest seeded immediate pair owns this
            // triple.  This avoids a large triple HashSet while guaranteeing
            // one exact scoring pass per reachable unordered triple.
            let owner = seeded_owner(first, second, third, &seed_map);
            let Some(owner) = owner else {
                continue;
            };
            if (owner.first, owner.second) != (seed.first, seed.second) {
                continue;
            }

            let candidate = best_triple_candidate(
                bits_flat,
                &negated,
                row_words,
                needed_words,
                first,
                second,
                third,
                owner,
                y,
                n_samples,
                &lookup,
                total_sum,
            );
            triples_evaluated = triples_evaluated.checked_add(1).ok_or_else(|| {
                "garfield_pair_triple_refine: triple counter overflow".to_string()
            })?;
            polarity_evaluated = polarity_evaluated.checked_add(8).ok_or_else(|| {
                "garfield_pair_triple_refine: polarity counter overflow".to_string()
            })?;
            for (idx, target) in target_candidates.iter().enumerate() {
                if let Some(target) = target {
                    if candidate_cmp(&candidate, target) == Ordering::Greater {
                        target_better_counts[idx] = target_better_counts[idx].saturating_add(1);
                    }
                }
            }
            push_top_k(&mut heap, candidate, top_k);
        }
    }

    let mut candidates = heap
        .into_iter()
        .map(|Reverse(entry)| entry.0)
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|a, b| candidate_cmp(b, a));
    let targets = canonical_targets
        .into_iter()
        .zip(target_candidates)
        .zip(target_better_counts)
        .map(|(((first, second, third), candidate), better)| {
            if let Some(candidate) = candidate {
                PairTripleTargetDiagnostic {
                    first,
                    second,
                    third,
                    evaluated: true,
                    rank: Some(better.saturating_add(1)),
                    polarity: Some(candidate.polarity),
                    raw_score: Some(candidate.raw_score),
                    support: Some(candidate.support),
                    owner_first: Some(candidate.owner_first),
                    owner_second: Some(candidate.owner_second),
                    owner_pair_rank: Some(candidate.owner_pair_rank),
                }
            } else {
                PairTripleTargetDiagnostic {
                    first,
                    second,
                    third,
                    evaluated: false,
                    rank: None,
                    polarity: None,
                    raw_score: None,
                    support: None,
                    owner_first: None,
                    owner_second: None,
                    owner_pair_rank: None,
                }
            }
        })
        .collect();
    Ok(PairTripleScanResult {
        candidates,
        targets,
        pair_seeds_used: canonical_seeds.len(),
        expansion_attempts,
        triples_evaluated,
        polarity_evaluated,
        n_rows,
        n_samples,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        candidate_cmp, row_slice, scan_pair_seeded_triples_continuous_packed, PairTripleCandidate,
        PairTripleSeed,
    };

    fn row(n_samples: usize, ones: &[usize]) -> Vec<u64> {
        let mut out = vec![0u64; n_samples.div_ceil(64).max(1)];
        for &idx in ones {
            out[idx >> 6] |= 1u64 << (idx & 63);
        }
        if let Some(rem) = n_samples.checked_rem(64) {
            if rem != 0 {
                let last = out.len() - 1;
                out[last] &= (1u64 << rem) - 1;
            }
        }
        out
    }

    #[test]
    fn canonical_owner_evaluates_each_seed_reachable_triple_once() {
        let n_samples = 7;
        let bits = [
            row(n_samples, &[0, 1, 4]),
            row(n_samples, &[1, 2, 5]),
            row(n_samples, &[0, 2, 3, 6]),
            row(n_samples, &[3, 4, 6]),
        ]
        .concat();
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let seeds = [
            PairTripleSeed {
                first: 0,
                second: 1,
                pair_rank: 0,
            },
            PairTripleSeed {
                first: 0,
                second: 2,
                pair_rank: 1,
            },
            PairTripleSeed {
                first: 1,
                second: 2,
                pair_rank: 2,
            },
            // Duplicate seed must not create duplicate triple scores.
            PairTripleSeed {
                first: 2,
                second: 0,
                pair_rank: 3,
            },
        ];
        let got =
            scan_pair_seeded_triples_continuous_packed(&bits, 1, 4, &y, n_samples, &seeds, &[], 16)
                .unwrap();
        assert_eq!(got.pair_seeds_used, 3);
        assert_eq!(got.expansion_attempts, 6);
        assert_eq!(got.triples_evaluated, 4);
        assert_eq!(got.polarity_evaluated, 32);
        assert_eq!(got.candidates.len(), 4);
        assert!(got
            .candidates
            .windows(2)
            .all(|pair| pair[0].raw_score >= pair[1].raw_score));
    }

    #[test]
    fn all_pair_seeds_reach_the_canonical_triple_ceiling() {
        let n_samples = 5;
        let bits = [
            row(n_samples, &[0, 1]),
            row(n_samples, &[1, 2]),
            row(n_samples, &[2, 3]),
            row(n_samples, &[0, 3, 4]),
        ]
        .concat();
        let y = [0.0, 1.0, 2.0, 3.0, 4.0];
        let seeds = (0..4)
            .flat_map(|i| ((i + 1)..4).map(move |j| (i, j)))
            .enumerate()
            .map(|(pair_rank, (first, second))| PairTripleSeed {
                first,
                second,
                pair_rank,
            })
            .collect::<Vec<_>>();
        let got =
            scan_pair_seeded_triples_continuous_packed(&bits, 1, 4, &y, n_samples, &seeds, &[], 0)
                .unwrap();
        assert_eq!(got.pair_seeds_used, 6);
        assert_eq!(got.expansion_attempts, 12);
        assert_eq!(got.triples_evaluated, 4);
        assert_eq!(got.polarity_evaluated, 32);
        assert!(got.candidates.is_empty());
    }

    #[test]
    fn duplicate_and_reversed_seed_input_is_canonicalized() {
        let n_samples = 6;
        let bits = [
            row(n_samples, &[0, 1]),
            row(n_samples, &[1, 2]),
            row(n_samples, &[2, 3]),
        ]
        .concat();
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let seeds = [
            PairTripleSeed {
                first: 2,
                second: 0,
                pair_rank: 10,
            },
            PairTripleSeed {
                first: 0,
                second: 2,
                pair_rank: 2,
            },
        ];
        let got =
            scan_pair_seeded_triples_continuous_packed(&bits, 1, 3, &y, n_samples, &seeds, &[], 8)
                .unwrap();
        assert_eq!(got.pair_seeds_used, 1);
        assert_eq!(got.expansion_attempts, 1);
        assert_eq!(got.triples_evaluated, 1);
        assert_eq!(got.candidates.len(), 1);
        assert_eq!(got.candidates[0].owner_pair_rank, 2);
    }

    #[test]
    fn target_diagnostics_report_reachability_and_rank_without_full_retention() {
        let n_samples = 6;
        let bits = [
            row(n_samples, &[0, 1]),
            row(n_samples, &[1, 2]),
            row(n_samples, &[2, 3]),
            row(n_samples, &[3, 4]),
        ]
        .concat();
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let seeds = [PairTripleSeed {
            first: 0,
            second: 1,
            pair_rank: 0,
        }];
        let targets = [(0, 1, 2), (1, 2, 3)];
        let got = scan_pair_seeded_triples_continuous_packed(
            &bits, 1, 4, &y, n_samples, &seeds, &targets, 0,
        )
        .unwrap();
        assert_eq!(got.candidates.len(), 0);
        assert_eq!(got.targets.len(), 2);
        assert!(got.targets[0].evaluated);
        assert!(got.targets[0].rank.is_some());
        assert!(got.targets[0].raw_score.is_some());
        assert!(got.targets[0].polarity.is_some());
        assert!(!got.targets[1].evaluated);
        assert!(got.targets[1].rank.is_none());
    }

    #[test]
    fn rejects_out_of_range_or_self_pair_seed() {
        let bits = row(4, &[0, 1, 2]).repeat(3);
        let y = [0.0, 1.0, 2.0, 3.0];
        let bad = [PairTripleSeed {
            first: 0,
            second: 3,
            pair_rank: 0,
        }];
        assert!(
            scan_pair_seeded_triples_continuous_packed(&bits, 1, 3, &y, 4, &bad, &[], 4).is_err()
        );
        let bad = [PairTripleSeed {
            first: 1,
            second: 1,
            pair_rank: 0,
        }];
        assert!(
            scan_pair_seeded_triples_continuous_packed(&bits, 1, 3, &y, 4, &bad, &[], 4).is_err()
        );
    }

    #[test]
    fn evaluates_all_eight_polarities_and_keeps_reference_best() {
        let n_samples = 9;
        let bits = [
            row(n_samples, &[0, 1, 2, 3, 4]),
            row(n_samples, &[5, 6, 7]),
            row(n_samples, &[0, 5]),
        ]
        .concat();
        let y = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 10.0, 9.0, 3.0];
        let seed = [PairTripleSeed {
            first: 0,
            second: 1,
            pair_rank: 7,
        }];
        let got =
            scan_pair_seeded_triples_continuous_packed(&bits, 1, 3, &y, n_samples, &seed, &[], 1)
                .unwrap();
        assert_eq!(got.triples_evaluated, 1);
        assert_eq!(got.polarity_evaluated, 8);

        let total_sum = y.iter().sum::<f64>();
        let mut expected = None;
        for polarity in 0u8..8 {
            let mut combined = vec![0u64; 1];
            for bit in 0..64 {
                if bit >= n_samples {
                    break;
                }
                let mut hit = true;
                for (row_idx, shift) in [(0usize, 1u8), (1usize, 2u8), (2usize, 4u8)] {
                    let row = row_slice(&bits, 1, row_idx, 1)[0];
                    let selected = if polarity & shift != 0 { !row } else { row };
                    hit &= selected & (1u64 << bit) != 0;
                }
                if hit {
                    combined[0] |= 1u64 << bit;
                }
            }
            let score = super::super::score::score_cont_centered_gain_packed_with_sum(
                &y, &combined, n_samples, total_sum,
            );
            let candidate = PairTripleCandidate {
                first: 0,
                second: 1,
                third: 2,
                polarity,
                raw_score: score.raw_score,
                support: score.n_hit,
                owner_first: 0,
                owner_second: 1,
                owner_pair_rank: 7,
            };
            if expected
                .as_ref()
                .map(|current| candidate_cmp(&candidate, current) == std::cmp::Ordering::Greater)
                .unwrap_or(true)
            {
                expected = Some(candidate);
            }
        }
        assert_eq!(got.candidates.as_slice(), &[expected.unwrap()]);
    }
}
