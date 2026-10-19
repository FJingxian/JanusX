//! Diagnostic-only deterministic sampling and exact null-max scoring for the
//! production Max6 screen. This module never supplies or mutates a production
//! threshold.

use super::logic_residual_screen::{self, FourCellGate, OneCellMarginTable};
use super::score::PackedYSumLookup;
use super::{GarfieldLogicBits, Max6ScreenCalibration, Max6ScreenPreparedUniverse};
use rayon::prelude::*;
use rayon::ThreadPool;
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::time::Instant;

pub(crate) const REQUESTED_SAMPLE_SIZES: [usize; 4] = [100_000, 1_000_000, 10_000_000, 50_000_000];
pub(crate) const PAIR_SAMPLE_ALGORITHM: &str = "janusx_splitmix64_bottom_k_v1";
const PRIORITY_DOMAIN: u64 = 0x4A58_4D36_5052_4931;
const UNIVERSE_SUM_DOMAIN: u64 = 0x4A58_554E_4956_5331;
const UNIVERSE_XOR_DOMAIN: u64 = 0x4A58_554E_4956_5831;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
pub(crate) const DEFAULT_TAIL_PAIR_SAMPLE_SEED: u64 = 0x4A58_5341_4D50_4C45;
pub(crate) const TAIL_AUDIT_TOP_K: usize = 4096;
pub(crate) const TAIL_AUDIT_SAMPLE_SIZES: [usize; 4] = [100_000, 1_000_000, 10_000_000, 50_000_000];
pub(crate) const SUPPORT_AWARE_SAMPLE_SIZES: [usize; 3] = [100_000, 1_000_000, 10_000_000];
pub(crate) const WEIGHTED_EXCEEDANCE_SAMPLE_SIZES: [usize; 3] = [100_000, 1_000_000, 10_000_000];
pub(crate) const POT_SAMPLE_SIZES: [usize; 3] = [100_000, 1_000_000, 10_000_000];
pub(crate) const POT_PILOT_SAMPLE_SIZE: usize = POT_SAMPLE_SIZES[0];
pub(crate) const POT_QUANTILES: [f64; 4] = [0.975, 0.99, 0.995, 0.999];
pub(crate) const POT_PAIR_SAMPLE_SEEDS: [u64; 5] = WEIGHTED_EXCEEDANCE_PAIR_SEEDS;
pub(crate) const POT_RETENTION_QUANTILE: f64 = 0.90;
pub(crate) const WEIGHTED_EXCEEDANCE_SUPPORT_SHARE_PERCENT: usize = 60;
pub(crate) const WEIGHTED_EXCEEDANCE_PAIR_SEEDS: [u64; 5] = [
    0x2F6E_2B1D_6F6E_0A11,
    0x79D2_3B47_4A5C_19E3,
    0xA8C5_90D1_2E73_6B45,
    0xC4E1_8F2A_0B6D_9357,
    0xE75A_13C9_8D42_B60F,
];
pub(crate) const SUPPORT_AWARE_ALLOWED_SHARES: [usize; 3] = [40, 50, 60];

pub(crate) struct TailAuditSample {
    pair_sample_seed: u64,
    sample: BottomKSample,
    actual_sizes: [usize; 4],
    selection_seconds: f64,
}

pub(crate) struct TailAuditSampledMaxima {
    pub(crate) sample: TailAuditSample,
    /// target sample size -> null replicate -> sampled maximum pair/score.
    pub(crate) winners: Vec<Vec<Option<TailPairScore>>>,
    pub(crate) preprocessing_seconds: f64,
    pub(crate) scoring_seconds_by_target: [f64; 4],
}

#[derive(Clone, Debug)]
struct SupportAwareAllocationPlan {
    share_percent: usize,
    actual_sizes: [usize; 3],
    quotas_by_size: [[usize; 5]; 3],
}

pub(crate) struct SupportAwareTailAuditMaxima {
    pub(crate) share_percent: usize,
    pub(crate) winners: Vec<Vec<Option<TailPairScore>>>,
    pub(crate) actual_sizes: [usize; 3],
    quotas_by_size: [[usize; 5]; 3],
    cutoffs_by_size: [[Option<PairPriority>; 5]; 3],
    sample_hashes: [String; 3],
    universe_count: usize,
    universe_hash: String,
    universe_stratum_counts: [usize; 5],
    selection_seconds: f64,
    selection_storage_bytes: usize,
    materialized_sample_storage_bytes: usize,
    preprocessing_seconds: f64,
    scoring_seconds_by_size: [f64; 3],
}

pub(crate) struct TailAuditReportPaths {
    pub(crate) top_pairs_tsv: String,
    pub(crate) sampled_ranks_tsv: String,
    pub(crate) summary_json: String,
    pub(crate) support_aware_maxima_tsv: Option<String>,
    pub(crate) support_aware_strata_tsv: Option<String>,
    pub(crate) support_aware_summary_json: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct WeightedExceedanceReportPaths {
    pub(crate) per_null_tsv: String,
    pub(crate) summary_tsv: String,
    pub(crate) top_pairs_tsv: String,
    pub(crate) manifest_json: String,
}

#[derive(Clone, Debug)]
pub(crate) struct PotAuditReportPaths {
    pub(crate) manifest_json: String,
    pub(crate) sample_scores_manifest_json: String,
}

pub(crate) struct SupportAwareTailReportPaths {
    pub(crate) maxima_tsv: String,
    pub(crate) strata_tsv: String,
    pub(crate) summary_json: String,
}

pub(crate) struct TailAuditFullScan {
    pub(crate) top_pairs: Vec<Vec<TailPairScore>>,
    pub(crate) sampled_full_ranks: Vec<Vec<Option<usize>>>,
    pub(crate) unique_pair_count: usize,
    pub(crate) elapsed_seconds: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct WeightedExceedanceFullScan {
    /// Flattened as [null][support stratum][threshold]. Values are exact full-universe counts.
    pub(crate) full_counts: Vec<u32>,
    /// Flattened as [null][sample slot][support stratum][threshold]. A sample slot is
    /// seed-major, then uniform M values, then support-aware M values.
    pub(crate) sampled_counts: Vec<u32>,
    pub(crate) top_pairs: Vec<Vec<TailPairScore>>,
    pub(crate) unique_pair_count: usize,
    pub(crate) threshold_count: usize,
    pub(crate) sample_slot_count: usize,
    pub(crate) preflight_accumulator_bytes: usize,
    pub(crate) reserved_accumulator_bytes: usize,
    pub(crate) preflight_output_bytes: usize,
    pub(crate) reserved_output_bytes: usize,
    pub(crate) work_metadata_bytes: usize,
    pub(crate) preflight_total_bytes: usize,
    pub(crate) elapsed_seconds: f64,
}

/// One exact Max6 screen result retained by the opt-in full-universe tail audit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TailPairScore {
    pub(crate) pair: (usize, usize),
    pub(crate) score: f64,
    pub(crate) best_channel: FourCellGate,
}

impl PartialEq for TailPairScore {
    fn eq(&self, other: &Self) -> bool {
        full_rank_cmp(self, other).is_eq()
    }
}

/// Total ordering used by all tail-audit ranking and rank-counting paths:
/// score descending, then canonical pair ascending.
#[inline]
pub(crate) fn full_rank_cmp(left: &TailPairScore, right: &TailPairScore) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.pair.cmp(&right.pair))
}

impl Eq for TailPairScore {}

impl Ord for TailPairScore {
    fn cmp(&self, other: &Self) -> Ordering {
        full_rank_cmp(self, other)
    }
}

impl PartialOrd for TailPairScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A fixed-capacity max-heap whose root is the worst retained full-rank item.
/// It never holds more than `limit` canonical pair scores.
pub(crate) struct TailTopK {
    limit: usize,
    heap: BinaryHeap<TailPairScore>,
}

impl TailTopK {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit),
        }
    }

    pub(crate) fn try_new(limit: usize) -> Result<Self, String> {
        let mut heap = BinaryHeap::new();
        heap.try_reserve(limit)
            .map_err(|error| format!("unable to reserve Top-{limit} Max6 tail heap: {error}"))?;
        Ok(Self { limit, heap })
    }

    pub(crate) fn storage_bytes(&self) -> usize {
        self.heap
            .capacity()
            .saturating_mul(size_of::<TailPairScore>())
    }

    pub(crate) fn consider(
        &mut self,
        pair: (usize, usize),
        score: f64,
        best_channel: FourCellGate,
    ) -> Result<(), String> {
        if pair.0 >= pair.1 {
            return Err(format!(
                "tail-audit Max6 pair must be canonical (i < j): ({}, {})",
                pair.0, pair.1
            ));
        }
        if !score.is_finite() {
            return Err(format!("tail-audit Max6 score is not finite: {score}"));
        }
        if self.limit == 0 {
            return Ok(());
        }
        let candidate = TailPairScore {
            pair,
            score,
            best_channel,
        };
        if self.heap.len() < self.limit {
            self.heap.push(candidate);
        } else if self
            .heap
            .peek()
            .is_some_and(|worst| full_rank_cmp(&candidate, worst).is_lt())
        {
            self.heap.pop();
            self.heap.push(candidate);
        }
        Ok(())
    }

    pub(crate) fn merge(&mut self, other: Self) {
        for candidate in other.heap.into_vec() {
            if self.limit == 0 {
                break;
            }
            if self.heap.len() < self.limit {
                self.heap.push(candidate);
            } else if self
                .heap
                .peek()
                .is_some_and(|worst| full_rank_cmp(&candidate, worst).is_lt())
            {
                self.heap.pop();
                self.heap.push(candidate);
            }
        }
    }

    pub(crate) fn into_ranked(self) -> Vec<TailPairScore> {
        let mut ranked = self.heap.into_vec();
        ranked.sort_by(full_rank_cmp);
        ranked
    }
}

/// CDF of the best rank selected by a uniform size-M sample without
/// replacement from N fixed, totally ordered pairs.
pub(crate) fn finite_population_sample_max_cdf(
    universe_size: usize,
    sample_size: usize,
    rank_cutoff: usize,
) -> Result<f64, String> {
    if sample_size == 0 || sample_size > universe_size {
        return Err(format!(
            "finite-population sample size must be in 1..={universe_size}, got {sample_size}"
        ));
    }
    if rank_cutoff == 0 {
        return Ok(0.0);
    }
    if rank_cutoff > universe_size {
        return Err(format!(
            "rank cutoff {rank_cutoff} exceeds universe size {universe_size}"
        ));
    }
    let misses_before_rank = universe_size - sample_size;
    if rank_cutoff > misses_before_rank {
        return Ok(1.0);
    }
    let no_hit_probability = (0..rank_cutoff).fold(1.0, |probability, index| {
        probability * (misses_before_rank - index) as f64 / (universe_size - index) as f64
    });
    Ok((1.0 - no_hit_probability).clamp(0.0, 1.0))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PairPriority {
    pub(crate) priority: u64,
    pub(crate) pair_id: u64,
}

impl Ord for PairPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.priority, self.pair_id).cmp(&(other.priority, other.pair_id))
    }
}

impl PartialOrd for PairPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
struct BottomKSample {
    priority_pairs: Vec<PairPriority>,
    unique_universe_count: usize,
    sample_hash: String,
    unique_universe_hash: String,
    storage_capacity_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct WeightedExceedanceSeedPlan {
    pub(crate) pair_sample_seed: u64,
    pub(crate) uniform_cutoffs: [Option<PairPriority>; 3],
    pub(crate) support_cutoffs: [[Option<PairPriority>; 5]; 3],
    pub(crate) uniform_stratum_sample_sizes: [[usize; 5]; 3],
    pub(crate) support_stratum_sample_sizes: [[usize; 5]; 3],
    pub(crate) uniform_sample_hashes: [String; 3],
    pub(crate) support_sample_hashes: [String; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct WeightedExceedanceSamplingPlan {
    pub(crate) requested_sizes: [usize; 3],
    pub(crate) actual_sizes: [usize; 3],
    pub(crate) unique_universe_count: usize,
    pub(crate) unique_universe_hash: String,
    pub(crate) universe_stratum_counts: [usize; 5],
    pub(crate) seeds: Vec<WeightedExceedanceSeedPlan>,
    pub(crate) selection_seconds: f64,
    pub(crate) preflight_heap_bytes: usize,
    pub(crate) retained_cutoff_bytes: usize,
}

impl WeightedExceedanceSamplingPlan {
    pub(crate) fn sample_slot_count(&self) -> usize {
        self.seeds.len() * 2 * self.requested_sizes.len()
    }

    /// Returns membership in the six nested sets for one pair-sample seed:
    /// uniform M=1e5/1e6/1e7, followed by fixed-60% support-aware M values.
    pub(crate) fn memberships_for_pair(
        &self,
        seed_index: usize,
        pair: (usize, usize),
        stratum: usize,
    ) -> Result<[bool; 6], String> {
        let seed_plan = self
            .seeds
            .get(seed_index)
            .ok_or_else(|| format!("exceedance sample seed index {seed_index} is out of range"))?;
        if stratum >= 5 {
            return Err(format!(
                "exceedance support stratum {stratum} is out of range"
            ));
        }
        let priority = pair_priority_for(seed_plan.pair_sample_seed, pair);
        let mut included = [false; 6];
        for size_index in 0..3 {
            included[size_index] =
                seed_plan.uniform_cutoffs[size_index].is_some_and(|cutoff| priority <= cutoff);
            included[3 + size_index] = seed_plan.support_cutoffs[size_index][stratum]
                .is_some_and(|cutoff| priority <= cutoff);
        }
        Ok(included)
    }
}

impl BottomKSample {
    #[cfg(test)]
    fn decoded_pairs(&self) -> Vec<(usize, usize)> {
        self.priority_pairs
            .iter()
            .map(|entry| decode_pair_id(entry.pair_id))
            .collect()
    }
}

struct BottomKPairSampler {
    limit: usize,
    seen: usize,
    heap: BinaryHeap<PairPriority>,
    universe_sum: u64,
    universe_xor: u64,
}

impl BottomKPairSampler {
    fn new(limit: usize, reserve_hint: usize) -> Self {
        let reserve = limit.min(reserve_hint);
        Self {
            limit,
            seen: 0,
            heap: BinaryHeap::with_capacity(reserve),
            universe_sum: 0,
            universe_xor: 0,
        }
    }

    fn try_reserve(&mut self, additional: usize) -> Result<(), String> {
        self.heap.try_reserve(additional).map_err(|error| {
            format!("unable to reserve {additional} entries for Max6 pair bottom-k heap: {error}")
        })
    }

    fn consider(&mut self, first: usize, second: usize, seed: u64) -> Result<(), String> {
        if first >= second {
            return Err(format!(
                "sampled Max6 pair is not canonical (i < j): ({first}, {second})"
            ));
        }
        let first = u32::try_from(first)
            .map_err(|_| "sampled Max6 pair index exceeds u32 capacity".to_string())?;
        let second = u32::try_from(second)
            .map_err(|_| "sampled Max6 pair index exceeds u32 capacity".to_string())?;
        let pair_id = (u64::from(first) << 32) | u64::from(second);
        self.seen = self
            .seen
            .checked_add(1)
            .ok_or_else(|| "sampled Max6 universe pair count overflow".to_string())?;
        self.universe_sum = self
            .universe_sum
            .wrapping_add(splitmix64(pair_id ^ UNIVERSE_SUM_DOMAIN));
        self.universe_xor ^= splitmix64(pair_id ^ UNIVERSE_XOR_DOMAIN);

        if self.limit == 0 {
            return Ok(());
        }
        let candidate = PairPriority {
            priority: stable_pair_priority(seed, first as usize, second as usize),
            pair_id,
        };
        if self.heap.len() < self.limit {
            self.heap.push(candidate);
        } else if self.heap.peek().is_some_and(|largest| candidate < *largest) {
            self.heap.pop();
            self.heap.push(candidate);
        }
        Ok(())
    }

    fn finish(self) -> Result<BottomKSample, String> {
        let storage_capacity_bytes = self
            .heap
            .capacity()
            .saturating_mul(size_of::<PairPriority>());
        let mut priority_pairs = self.heap.into_vec();
        priority_pairs.sort_unstable();
        if priority_pairs
            .windows(2)
            .any(|window| window[0].pair_id == window[1].pair_id)
        {
            return Err("canonical Max6 pair owner emitted a duplicate eligible pair".to_string());
        }
        let sample_hash = sample_set_hash(priority_pairs.as_slice());
        Ok(BottomKSample {
            priority_pairs,
            unique_universe_count: self.seen,
            sample_hash,
            unique_universe_hash: format!(
                "splitmix64_sum_xor_v1:{:016x}:{:016x}:{}",
                self.universe_sum, self.universe_xor, self.seen
            ),
            storage_capacity_bytes,
        })
    }
}

#[cfg(test)]
fn bottom_k_pairs(
    pairs: impl IntoIterator<Item = (usize, usize)>,
    seed: u64,
    limit: usize,
) -> Result<BottomKSample, String> {
    let mut sampler = BottomKPairSampler::new(limit, 0);
    for (first, second) in pairs {
        sampler.consider(first, second, seed)?;
    }
    sampler.finish()
}

#[derive(Clone, Debug)]
struct SupportAwarePairPools {
    unique_universe_count: usize,
    unique_universe_hash: String,
    universe_stratum_counts: [usize; 5],
    pairs_by_stratum: Vec<Vec<PairPriority>>,
    #[cfg(test)]
    sample_hashes_by_stratum: Vec<String>,
    storage_capacity_bytes: usize,
    selection_seconds: f64,
}

struct StratifiedBottomKPairSampler {
    seed: u64,
    limits: [usize; 5],
    seen_by_stratum: [usize; 5],
    heaps: Vec<BinaryHeap<PairPriority>>,
    universe_sum: u64,
    universe_xor: u64,
}

impl StratifiedBottomKPairSampler {
    fn new(seed: u64, limits: [usize; 5]) -> Self {
        Self {
            seed,
            limits,
            seen_by_stratum: [0; 5],
            heaps: (0..5)
                .map(|index| BinaryHeap::with_capacity(limits[index]))
                .collect(),
            universe_sum: 0,
            universe_xor: 0,
        }
    }

    fn try_new(seed: u64, limits: [usize; 5]) -> Result<Self, String> {
        let mut heaps = Vec::new();
        heaps
            .try_reserve_exact(5)
            .map_err(|error| format!("unable to reserve five support strata: {error}"))?;
        for (stratum, &limit) in limits.iter().enumerate() {
            let mut heap = BinaryHeap::new();
            heap.try_reserve(limit).map_err(|error| {
                format!(
                    "unable to reserve {limit} entries for support-aware stratum {}: {error}",
                    MIN_CELL_STRATA[stratum]
                )
            })?;
            heaps.push(heap);
        }
        Ok(Self {
            seed,
            limits,
            seen_by_stratum: [0; 5],
            heaps,
            universe_sum: 0,
            universe_xor: 0,
        })
    }

    fn consider(&mut self, first: usize, second: usize, stratum: usize) -> Result<(), String> {
        if first >= second {
            return Err(format!(
                "support-aware sampled Max6 pair is not canonical (i < j): ({first}, {second})"
            ));
        }
        if stratum >= self.heaps.len() {
            return Err(format!(
                "support-aware sampled Max6 stratum index {stratum} is out of range"
            ));
        }
        let first_u32 = u32::try_from(first)
            .map_err(|_| "support-aware pair index exceeds u32 capacity".to_string())?;
        let second_u32 = u32::try_from(second)
            .map_err(|_| "support-aware pair index exceeds u32 capacity".to_string())?;
        let pair_id = (u64::from(first_u32) << 32) | u64::from(second_u32);
        self.seen_by_stratum[stratum] = self.seen_by_stratum[stratum]
            .checked_add(1)
            .ok_or_else(|| "support-aware stratum count overflow".to_string())?;
        self.universe_sum = self
            .universe_sum
            .wrapping_add(splitmix64(pair_id ^ UNIVERSE_SUM_DOMAIN));
        self.universe_xor ^= splitmix64(pair_id ^ UNIVERSE_XOR_DOMAIN);

        let limit = self.limits[stratum];
        if limit == 0 {
            return Ok(());
        }
        let candidate = PairPriority {
            priority: stable_pair_priority(self.seed, first, second),
            pair_id,
        };
        let heap = &mut self.heaps[stratum];
        if heap.len() < limit {
            heap.push(candidate);
        } else if heap.peek().is_some_and(|largest| candidate < *largest) {
            heap.pop();
            heap.push(candidate);
        }
        Ok(())
    }

    fn finish(self) -> Result<SupportAwarePairPools, String> {
        let storage_capacity_bytes = self
            .heaps
            .iter()
            .map(|heap| heap.capacity().saturating_mul(size_of::<PairPriority>()))
            .sum();
        let mut pairs_by_stratum = Vec::with_capacity(self.heaps.len());
        #[cfg(test)]
        let mut sample_hashes_by_stratum = Vec::with_capacity(self.heaps.len());
        for heap in self.heaps {
            let mut pairs = heap.into_vec();
            pairs.sort_unstable();
            if pairs
                .windows(2)
                .any(|window| window[0].pair_id == window[1].pair_id)
            {
                return Err("support-aware canonical pair stratum contains a duplicate".to_string());
            }
            #[cfg(test)]
            sample_hashes_by_stratum.push(sample_set_hash(pairs.as_slice()));
            pairs_by_stratum.push(pairs);
        }
        let unique_universe_count = self.seen_by_stratum.iter().sum::<usize>();
        Ok(SupportAwarePairPools {
            unique_universe_count,
            unique_universe_hash: format!(
                "splitmix64_sum_xor_v1:{:016x}:{:016x}:{}",
                self.universe_sum, self.universe_xor, unique_universe_count
            ),
            universe_stratum_counts: self.seen_by_stratum,
            pairs_by_stratum,
            #[cfg(test)]
            sample_hashes_by_stratum,
            storage_capacity_bytes,
            selection_seconds: 0.0,
        })
    }
}

impl SupportAwarePairPools {
    fn materialize_sample(
        &self,
        plan: &SupportAwareAllocationPlan,
        pair_sample_seed: u64,
    ) -> Result<(TailAuditSample, [[Option<PairPriority>; 5]; 3], [String; 3]), String> {
        let max_size = *plan.actual_sizes.last().unwrap_or(&0);
        let mut priority_pairs = Vec::with_capacity(max_size);
        let mut cutoffs_by_size = [[None; 5]; 3];
        let mut sample_hashes = std::array::from_fn(|_| String::new());
        let mut previous_quotas = [0usize; 5];
        for size_index in 0..plan.actual_sizes.len() {
            let quotas = plan.quotas_by_size[size_index];
            for stratum in 0..5 {
                let available = self
                    .pairs_by_stratum
                    .get(stratum)
                    .ok_or_else(|| "support-aware sample lost a support stratum".to_string())?;
                let previous = previous_quotas[stratum];
                let target = quotas[stratum];
                if target < previous || target > available.len() {
                    return Err(format!(
                        "support-aware target quota {target} is invalid for stratum {} with {} retained pairs",
                        MIN_CELL_STRATA[stratum],
                        available.len()
                    ));
                }
                priority_pairs.extend_from_slice(&available[previous..target]);
                cutoffs_by_size[size_index][stratum] = target
                    .checked_sub(1)
                    .and_then(|last| available.get(last).copied());
                previous_quotas[stratum] = target;
            }
            if priority_pairs.len() != plan.actual_sizes[size_index] {
                return Err(format!(
                    "support-aware sample at M={} contains {} pairs",
                    plan.actual_sizes[size_index],
                    priority_pairs.len()
                ));
            }
            sample_hashes[size_index] = sample_set_hash(priority_pairs.as_slice());
        }
        if priority_pairs
            .windows(2)
            .any(|window| window[0].pair_id == window[1].pair_id)
        {
            return Err("support-aware nested sample contains a duplicate pair".to_string());
        }
        let actual_sizes = [
            plan.actual_sizes[0],
            plan.actual_sizes[1],
            plan.actual_sizes[2],
            plan.actual_sizes[2],
        ];
        let sample_storage_capacity_bytes = priority_pairs
            .capacity()
            .saturating_mul(size_of::<PairPriority>());
        let sample = BottomKSample {
            sample_hash: sample_set_hash(priority_pairs.as_slice()),
            priority_pairs,
            unique_universe_count: self.unique_universe_count,
            unique_universe_hash: self.unique_universe_hash.clone(),
            storage_capacity_bytes: sample_storage_capacity_bytes,
        };
        Ok((
            TailAuditSample {
                pair_sample_seed,
                sample,
                actual_sizes,
                selection_seconds: self.selection_seconds,
            },
            cutoffs_by_size,
            sample_hashes,
        ))
    }
}

fn support_aware_allocation_plans(
    universe_stratum_counts: [usize; 5],
    unique_universe_count: usize,
    target_shares: &[usize],
) -> Result<Vec<SupportAwareAllocationPlan>, String> {
    if target_shares.is_empty() {
        return Err("support-aware sampling requires at least one target share".to_string());
    }
    let mut seen_shares = std::collections::HashSet::with_capacity(target_shares.len());
    for &share in target_shares {
        if !SUPPORT_AWARE_ALLOWED_SHARES.contains(&share) {
            return Err(format!(
                "support-aware target share must be one of {:?}, got {share}",
                SUPPORT_AWARE_ALLOWED_SHARES
            ));
        }
        if !seen_shares.insert(share) {
            return Err(format!("duplicate support-aware target share {share}"));
        }
    }
    let mut plans = Vec::with_capacity(target_shares.len());
    for &share_percent in target_shares {
        let mut actual_sizes = [0usize; 3];
        let mut quotas_by_size = [[0usize; 5]; 3];
        for (size_index, requested_size) in SUPPORT_AWARE_SAMPLE_SIZES.iter().enumerate() {
            actual_sizes[size_index] = (*requested_size).min(unique_universe_count);
            quotas_by_size[size_index] = support_aware_stratum_quotas(
                universe_stratum_counts,
                *requested_size,
                share_percent,
            )?;
        }
        for size_index in 1..actual_sizes.len() {
            if actual_sizes[size_index] < actual_sizes[size_index - 1]
                || (0..5).any(|stratum| {
                    quotas_by_size[size_index][stratum] < quotas_by_size[size_index - 1][stratum]
                })
            {
                return Err(format!(
                    "support-aware allocation is not nested for {share_percent}% target share"
                ));
            }
        }
        plans.push(SupportAwareAllocationPlan {
            share_percent,
            actual_sizes,
            quotas_by_size,
        });
    }
    Ok(plans)
}

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn actual_sample_sizes(unique_universe_count: usize) -> [usize; 4] {
    REQUESTED_SAMPLE_SIZES.map(|target| target.min(unique_universe_count))
}

pub(crate) fn parse_pair_sample_seeds(raw: &str, null_seed: u64) -> Result<Vec<u64>, String> {
    let seeds = raw
        .split(',')
        .map(str::trim)
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|error| format!("invalid pair-sample seed '{value}': {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if seeds.is_empty() || (seeds.len() != 1 && !(3..=5).contains(&seeds.len())) {
        return Err(format!(
            "provide one pair-sample seed or 3–5 independent seeds; got {}",
            seeds.len()
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(seeds.len());
    for &seed in &seeds {
        if !seen.insert(seed) {
            return Err(format!("duplicate pair-sample seed {seed}"));
        }
        if seed == null_seed {
            return Err("pair-sample seeds must be distinct from the null stream seed".to_string());
        }
    }
    Ok(seeds)
}

#[inline]
fn stable_pair_priority(seed: u64, first: usize, second: usize) -> u64 {
    let pair_id = ((first as u64) << 32) | second as u64;
    splitmix64(seed ^ PRIORITY_DOMAIN ^ splitmix64(pair_id))
}

#[inline]
fn decode_pair_id(pair_id: u64) -> (usize, usize) {
    (
        (pair_id >> 32) as usize,
        (pair_id & u64::from(u32::MAX)) as usize,
    )
}

fn sample_set_hash(entries: &[PairPriority]) -> String {
    let mut hash = FNV_OFFSET;
    for entry in entries {
        hash = fnv1a64_update(hash, &entry.pair_id.to_le_bytes());
    }
    format!("fnv1a64_pairid_priority_order_v1:{hash:016x}")
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn fnv1a64_field(mut hash: u64, bytes: &[u8]) -> u64 {
    hash = fnv1a64_update(hash, &(bytes.len() as u64).to_le_bytes());
    fnv1a64_update(hash, bytes)
}

fn marker_order_hash(logic_bits: &GarfieldLogicBits) -> String {
    let mut hash = fnv1a64_update(FNV_OFFSET, &(logic_bits.sites.len() as u64).to_le_bytes());
    for site in logic_bits.sites.iter() {
        hash = fnv1a64_field(hash, site.chrom().as_bytes());
        hash = fnv1a64_update(hash, &site.pos.to_le_bytes());
        hash = fnv1a64_field(hash, site.snp().as_bytes());
        hash = fnv1a64_field(hash, site.ref_allele().as_bytes());
        hash = fnv1a64_field(hash, site.alt_allele().as_bytes());
        hash = fnv1a64_field(hash, format!("{:?}", site.mode).as_bytes());
    }
    format!("fnv1a64_garfield_marker_order_v1:{hash:016x}")
}

fn sample_order_hash(logic_bits: &GarfieldLogicBits) -> String {
    let mut hash = fnv1a64_update(
        FNV_OFFSET,
        &(logic_bits.sample_ids.len() as u64).to_le_bytes(),
    );
    for sample_id in logic_bits.sample_ids.iter() {
        hash = fnv1a64_field(hash, sample_id.as_bytes());
    }
    format!("fnv1a64_sample_order_v1:{hash:016x}")
}

#[derive(Clone, Debug, Default)]
struct PairStrataCounts {
    owner_scan_unit: BTreeMap<String, u64>,
    min_maf: BTreeMap<String, u64>,
    pairwise_r2: BTreeMap<String, u64>,
    min_cell: BTreeMap<String, u64>,
}

impl PairStrataCounts {
    fn increment(&mut self, dimension: &str, label: String) {
        let counts = match dimension {
            "owner_scan_unit" => &mut self.owner_scan_unit,
            "min_maf" => &mut self.min_maf,
            "pairwise_r2" => &mut self.pairwise_r2,
            "min_cell" => &mut self.min_cell,
            _ => unreachable!("known sampled-pair stratum"),
        };
        *counts.entry(label).or_default() += 1;
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SampledScreenDiagnosticOutput {
    pub(crate) maxima_tsv: String,
    pub(crate) manifest_json: String,
    pub(crate) strata_tsv: String,
    pub(crate) full_maximizer_geometry_tsv: String,
}

#[derive(Clone, Debug)]
struct SeedTargetSummary {
    target_m: usize,
    actual_m: usize,
    exact_capture_rate: f64,
    paired_gap_median: f64,
    paired_gap_p95: f64,
    paired_gap_p99: f64,
    full_minus_sampled_q95: f64,
    full_minus_sampled_q99: f64,
}

#[derive(Clone, Debug)]
struct SampledScreenSeedOutput {
    pair_sample_seed: u64,
    maxima_tsv: String,
    manifest_json: String,
    strata_tsv: String,
    target_summaries: Vec<SeedTargetSummary>,
    selection_seconds: f64,
    strata_seconds: f64,
    null_preprocessing_seconds: f64,
    scoring_seconds_by_target: [f64; 4],
    sample_storage_capacity_bytes: usize,
    total_seconds: f64,
}

pub(crate) fn parse_cutpoints(raw: &str, name: &str, maximum: f64) -> Result<Vec<f64>, String> {
    let cutpoints = raw
        .split(',')
        .map(str::trim)
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|error| format!("invalid {name} cutpoint '{value}': {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if cutpoints.len() < 2
        || cutpoints.iter().any(|value| !value.is_finite())
        || cutpoints[0] != 0.0
        || *cutpoints.last().unwrap_or(&f64::NAN) != maximum
        || cutpoints.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(format!(
            "{name} cutpoints must be strictly increasing and include 0 and {maximum}: {raw}"
        ));
    }
    Ok(cutpoints)
}

fn cutpoint_bin(value: f64, cutpoints: &[f64]) -> String {
    let value = value.clamp(
        cutpoints[0],
        *cutpoints.last().expect("validated cutpoints"),
    );
    let index = cutpoints
        .windows(2)
        .position(|edge| value < edge[1])
        .unwrap_or(cutpoints.len() - 2);
    format!(
        "[{:.4},{:.4}{}",
        cutpoints[index],
        cutpoints[index + 1],
        if index + 2 == cutpoints.len() {
            "]"
        } else {
            ")"
        }
    )
}

fn min_cell_stratum(value: usize) -> &'static str {
    match value {
        0 => "0",
        1..=4 => "1-4",
        5..=9 => "5-9",
        10..=19 => "10-19",
        _ => ">=20",
    }
}

const MIN_CELL_STRATA: [&str; 5] = ["0", "1-4", "5-9", "10-19", ">=20"];

fn min_cell_stratum_index(value: usize) -> usize {
    match value {
        0 => 0,
        1..=4 => 1,
        5..=9 => 2,
        10..=19 => 3,
        _ => 4,
    }
}

fn proportional_stratum_quotas(total: usize, capacities: [usize; 5]) -> [usize; 5] {
    let available = capacities.iter().map(|&count| count as u128).sum::<u128>();
    if total == 0 || available == 0 {
        return [0; 5];
    }
    let target = (total as u128).min(available);
    let mut quotas = [0usize; 5];
    let mut remainders = Vec::with_capacity(capacities.len());
    let mut assigned = 0usize;
    for (index, &capacity) in capacities.iter().enumerate() {
        let scaled = target * capacity as u128;
        let base = (scaled / available) as usize;
        quotas[index] = base.min(capacity);
        assigned += quotas[index];
        remainders.push((scaled % available, index));
    }
    remainders.sort_unstable_by(
        |(left_remainder, left_index), (right_remainder, right_index)| {
            right_remainder
                .cmp(left_remainder)
                .then_with(|| left_index.cmp(right_index))
        },
    );
    let target = target as usize;
    let mut remaining = target.saturating_sub(assigned);
    while remaining > 0 {
        let mut progressed = false;
        for &(_, index) in remainders.iter() {
            if quotas[index] < capacities[index] {
                quotas[index] += 1;
                remaining -= 1;
                progressed = true;
                if remaining == 0 {
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    quotas
}

fn support_aware_stratum_quotas(
    universe_counts: [usize; 5],
    requested_size: usize,
    target_share_percent: usize,
) -> Result<[usize; 5], String> {
    if target_share_percent > 100 {
        return Err(format!(
            "support-aware target share must be in 0..=100, got {target_share_percent}"
        ));
    }
    let universe_size = universe_counts.iter().sum::<usize>();
    let actual_size = requested_size.min(universe_size);
    let target_capacity = universe_counts[2].saturating_add(universe_counts[3]);
    let other_capacity = universe_counts[0]
        .saturating_add(universe_counts[1])
        .saturating_add(universe_counts[4]);
    let requested_target = actual_size.saturating_mul(target_share_percent) / 100;
    let target_size = requested_target.min(target_capacity);

    let target_quotas = proportional_stratum_quotas(
        target_size,
        [0, 0, universe_counts[2], universe_counts[3], 0],
    );
    let mut quotas = target_quotas;
    let target_assigned = target_quotas.iter().sum::<usize>();
    let other_size = (actual_size - target_assigned).min(other_capacity);
    let other_quotas = proportional_stratum_quotas(
        other_size,
        [
            universe_counts[0],
            universe_counts[1],
            0,
            0,
            universe_counts[4],
        ],
    );
    for index in [0usize, 1, 4] {
        quotas[index] = other_quotas[index];
    }

    let assigned = quotas.iter().sum::<usize>();
    let unassigned = actual_size.saturating_sub(assigned);
    if unassigned > 0 {
        let remaining_capacities =
            std::array::from_fn(|index| universe_counts[index].saturating_sub(quotas[index]));
        let extra = proportional_stratum_quotas(unassigned, remaining_capacities);
        for index in 0..quotas.len() {
            quotas[index] += extra[index];
        }
    }
    if quotas.iter().sum::<usize>() != actual_size
        || quotas
            .iter()
            .zip(universe_counts)
            .any(|(&selected, available)| selected > available)
    {
        return Err("support-aware stratum quotas do not match the available universe".to_string());
    }
    Ok(quotas)
}

pub(crate) fn support_counts_by_active_slot(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
) -> Vec<usize> {
    let words = logic_bits.n_samples.div_ceil(64);
    let tail = logic_bits.n_samples % 64;
    universe
        .active_rows
        .iter()
        .map(|&global_row| {
            let start = global_row * logic_bits.row_words;
            let row = &logic_bits.bits()[start..start + words];
            row.iter()
                .enumerate()
                .map(|(word_index, &word)| {
                    if word_index + 1 == words && tail != 0 {
                        (word & ((1u64 << tail) - 1)).count_ones() as usize
                    } else {
                        word.count_ones() as usize
                    }
                })
                .sum()
        })
        .collect()
}

fn pair_n11(logic_bits: &GarfieldLogicBits, first_global: usize, second_global: usize) -> usize {
    let words = logic_bits.n_samples.div_ceil(64);
    let first_start = first_global * logic_bits.row_words;
    let second_start = second_global * logic_bits.row_words;
    let first = &logic_bits.bits()[first_start..first_start + words];
    let second = &logic_bits.bits()[second_start..second_start + words];
    let tail = logic_bits.n_samples % 64;
    first
        .iter()
        .zip(second)
        .enumerate()
        .map(|(word_index, (&left, &right))| {
            let value = left & right;
            if word_index + 1 == words && tail != 0 {
                (value & ((1u64 << tail) - 1)).count_ones() as usize
            } else {
                value.count_ones() as usize
            }
        })
        .sum()
}

fn canonical_owner_unit(
    universe: &Max6ScreenPreparedUniverse,
    first_slot: usize,
    second_slot: usize,
) -> Result<usize, String> {
    let ownership = &universe.pair_ownership;
    if ownership.membership_units.is_empty() {
        let first_owner = *ownership
            .single_membership_units
            .get(first_slot)
            .ok_or_else(|| "sampled Max6 first marker membership is out of range".to_string())?;
        let second_owner = *ownership
            .single_membership_units
            .get(second_slot)
            .ok_or_else(|| "sampled Max6 second marker membership is out of range".to_string())?;
        if first_owner == u32::MAX || first_owner != second_owner {
            return Err("sampled Max6 pair has no unique canonical owner unit".to_string());
        }
        Ok(first_owner as usize)
    } else {
        ownership
            .pair_owner(first_slot, second_slot)
            .ok_or_else(|| "sampled Max6 pair lost its canonical owner".to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PairBinaryGeometry {
    counts: [usize; 4],
    min_cell: usize,
    maf1: f64,
    maf2: f64,
    pairwise_r2: f64,
    owner_scan_unit: usize,
}

fn pair_binary_geometry(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    supports: &[usize],
    first: usize,
    second: usize,
) -> Result<PairBinaryGeometry, String> {
    if first >= second {
        return Err(format!(
            "Max6 pair geometry requires canonical indices (i < j): ({first}, {second})"
        ));
    }
    let first_slot = universe.pair_ownership.active_slot(first);
    let second_slot = universe.pair_ownership.active_slot(second);
    let first_support = *supports
        .get(first_slot)
        .ok_or_else(|| "Max6 pair geometry first support slot is out of range".to_string())?;
    let second_support = *supports
        .get(second_slot)
        .ok_or_else(|| "Max6 pair geometry second support slot is out of range".to_string())?;
    let n11 = pair_n11(logic_bits, first, second);
    if n11 > first_support || n11 > second_support {
        return Err(format!(
            "Max6 pair geometry intersection {n11} exceeds supports {first_support}/{second_support}"
        ));
    }
    let n10 = first_support - n11;
    let n01 = second_support - n11;
    let n00 = logic_bits
        .n_samples
        .checked_sub(first_support + second_support - n11)
        .ok_or_else(|| "Max6 pair geometry n00 underflow".to_string())?;
    let denominator = first_support as f64
        * (logic_bits.n_samples - first_support) as f64
        * second_support as f64
        * (logic_bits.n_samples - second_support) as f64;
    let pairwise_r2 = if denominator > 0.0 {
        let covariance =
            logic_bits.n_samples as f64 * n11 as f64 - first_support as f64 * second_support as f64;
        (covariance * covariance / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Ok(PairBinaryGeometry {
        counts: [n00, n01, n10, n11],
        min_cell: n00.min(n01).min(n10).min(n11),
        maf1: (first_support as f64 / logic_bits.n_samples as f64)
            .min(1.0 - first_support as f64 / logic_bits.n_samples as f64),
        maf2: (second_support as f64 / logic_bits.n_samples as f64)
            .min(1.0 - second_support as f64 / logic_bits.n_samples as f64),
        pairwise_r2,
        owner_scan_unit: canonical_owner_unit(universe, first_slot, second_slot)?,
    })
}

fn visit_unique_eligible_pairs(
    universe: &Max6ScreenPreparedUniverse,
    mut visit_pair: impl FnMut(usize, usize) -> Result<(), String>,
    mut visit_unit: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<usize, String> {
    let total_units = universe.units.len();
    let mut pair_count = 0usize;
    for (unit_index, unit) in universe.units.iter().enumerate() {
        for first_local in 0..unit.selected_global_rows.len().saturating_sub(1) {
            for second_local in first_local + 1..unit.selected_global_rows.len() {
                let Some(owned) = super::max6_screen_owned_pair(
                    universe,
                    unit_index,
                    unit,
                    first_local,
                    second_local,
                )?
                else {
                    continue;
                };
                let first = owned.first_global_row.min(owned.second_global_row);
                let second = owned.first_global_row.max(owned.second_global_row);
                if first >= second {
                    return Err(format!(
                        "support-aware owner emitted a noncanonical pair ({first}, {second})"
                    ));
                }
                visit_pair(first, second)?;
                pair_count = pair_count.saturating_add(1);
            }
        }
        visit_unit(unit_index + 1, total_units)?;
    }
    Ok(pair_count)
}

pub(crate) fn pair_min_cell_stratum(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    supports: &[usize],
    first: usize,
    second: usize,
) -> Result<usize, String> {
    let first_slot = universe.pair_ownership.active_slot(first);
    let second_slot = universe.pair_ownership.active_slot(second);
    let first_support = *supports
        .get(first_slot)
        .ok_or_else(|| "support-aware first marker slot is out of range".to_string())?;
    let second_support = *supports
        .get(second_slot)
        .ok_or_else(|| "support-aware second marker slot is out of range".to_string())?;
    let n11 = pair_n11(logic_bits, first, second);
    if n11 > first_support || n11 > second_support {
        return Err(format!(
            "support-aware intersection count {n11} exceeds marker supports {first_support}/{second_support}"
        ));
    }
    let n10 = first_support - n11;
    let n01 = second_support - n11;
    let n00 = logic_bits
        .n_samples
        .checked_sub(first_support + second_support - n11)
        .ok_or_else(|| "support-aware n00 cell count underflow".to_string())?;
    Ok(min_cell_stratum_index(n00.min(n01).min(n10).min(n11)))
}

fn support_sample_set_hash(
    pairs_by_stratum: &[Vec<PairPriority>],
    quotas: [usize; 5],
) -> Result<String, String> {
    if pairs_by_stratum.len() != 5 {
        return Err(format!(
            "support sample hash requires five strata, got {}",
            pairs_by_stratum.len()
        ));
    }
    let mut hash = FNV_OFFSET;
    for stratum in 0..5 {
        let pairs = &pairs_by_stratum[stratum];
        let quota = quotas[stratum];
        if quota > pairs.len() {
            return Err(format!(
                "support sample hash quota {quota} exceeds retained {} pairs in stratum {}",
                pairs.len(),
                MIN_CELL_STRATA[stratum]
            ));
        }
        hash = fnv1a64_update(hash, &[stratum as u8]);
        for entry in &pairs[..quota] {
            hash = fnv1a64_update(hash, &entry.pair_id.to_le_bytes());
        }
    }
    Ok(format!("fnv1a64_support_priority_order_v1:{hash:016x}"))
}

/// Build exact, deterministic bottom-k cutoffs for five independent pair
/// priority seeds. The first pass establishes true support-stratum sizes; the
/// second pass fills one global and one support-stratified heap per seed.
pub(crate) fn prepare_weighted_exceedance_sampling_plan(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    pair_sample_seeds: [u64; 5],
    mut notify_progress: impl FnMut(&str, usize, usize, usize) -> Result<(), String>,
) -> Result<WeightedExceedanceSamplingPlan, String> {
    let started = Instant::now();
    let supports = support_counts_by_active_slot(logic_bits, universe);
    let total_units = universe.units.len();
    let mut fingerprint = BottomKPairSampler::new(0, 0);
    let mut universe_stratum_counts = [0usize; 5];
    notify_progress("max6_exceedance_pair_census", 0, total_units, 0)?;
    let census_count = visit_unique_eligible_pairs(
        universe,
        |first, second| {
            let stratum =
                pair_min_cell_stratum(logic_bits, universe, supports.as_slice(), first, second)?;
            universe_stratum_counts[stratum] = universe_stratum_counts[stratum]
                .checked_add(1)
                .ok_or_else(|| "exceedance audit support-stratum count overflow".to_string())?;
            fingerprint.consider(first, second, 0)
        },
        |done, total| {
            if done % 64 == 0 || done == total {
                notify_progress("max6_exceedance_pair_census", done, total, 0)
            } else {
                Ok(())
            }
        },
    )?;
    let fingerprint = fingerprint.finish()?;
    if fingerprint.unique_universe_count != census_count
        || universe_stratum_counts.iter().sum::<usize>() != census_count
    {
        return Err(
            "exceedance audit census disagrees with canonical unique-pair count".to_string(),
        );
    }

    let actual_sizes = WEIGHTED_EXCEEDANCE_SAMPLE_SIZES.map(|size| size.min(census_count));
    let quotas_by_size = WEIGHTED_EXCEEDANCE_SAMPLE_SIZES.map(|size| {
        support_aware_stratum_quotas(
            universe_stratum_counts,
            size,
            WEIGHTED_EXCEEDANCE_SUPPORT_SHARE_PERCENT,
        )
    });
    let quotas_by_size = quotas_by_size.into_iter().collect::<Result<Vec<_>, _>>()?;
    let support_heap_limits = std::array::from_fn(|stratum| {
        quotas_by_size
            .iter()
            .map(|quotas| quotas[stratum])
            .max()
            .unwrap_or(0)
    });
    let support_heap_total = support_heap_limits
        .iter()
        .try_fold(0usize, |sum, value| sum.checked_add(*value))
        .ok_or_else(|| "support-aware exceedance heap capacity overflow".to_string())?;
    let per_seed_heap_entries = actual_sizes[2]
        .checked_add(support_heap_total)
        .ok_or_else(|| "exceedance pair-sample heap capacity overflow".to_string())?;
    let preflight_heap_bytes = pair_sample_seeds
        .len()
        .checked_mul(per_seed_heap_entries)
        .and_then(|entries| entries.checked_mul(size_of::<PairPriority>()))
        .ok_or_else(|| "exceedance pair-sample heap byte count overflow".to_string())?;

    notify_progress(
        "max6_exceedance_pair_sample_preflight",
        1,
        1,
        preflight_heap_bytes,
    )?;

    let mut samplers = Vec::new();
    samplers
        .try_reserve_exact(pair_sample_seeds.len())
        .map_err(|error| format!("unable to reserve seed sampler metadata: {error}"))?;
    for seed in pair_sample_seeds {
        let mut uniform = BottomKPairSampler::new(actual_sizes[2], 0);
        uniform.try_reserve(actual_sizes[2])?;
        let support = StratifiedBottomKPairSampler::try_new(seed, support_heap_limits)?;
        samplers.push((seed, uniform, support));
    }
    notify_progress(
        "max6_exceedance_pair_sample_selection",
        0,
        total_units,
        census_count,
    )?;
    let selected_count = visit_unique_eligible_pairs(
        universe,
        |first, second| {
            let stratum =
                pair_min_cell_stratum(logic_bits, universe, supports.as_slice(), first, second)?;
            for (seed, uniform, support) in samplers.iter_mut() {
                uniform.consider(first, second, *seed)?;
                support.consider(first, second, stratum)?;
            }
            Ok(())
        },
        |done, total| {
            if done % 64 == 0 || done == total {
                notify_progress(
                    "max6_exceedance_pair_sample_selection",
                    done,
                    total,
                    census_count,
                )
            } else {
                Ok(())
            }
        },
    )?;
    if selected_count != census_count {
        return Err(format!(
            "exceedance pair-sample selection saw {selected_count} pairs, census had {census_count}"
        ));
    }

    let mut seed_plans = Vec::with_capacity(pair_sample_seeds.len());
    for (seed, uniform_sampler, support_sampler) in samplers {
        let uniform = uniform_sampler.finish()?;
        let support = support_sampler.finish()?;
        if uniform.unique_universe_count != census_count
            || support.unique_universe_count != census_count
            || uniform.unique_universe_hash != fingerprint.unique_universe_hash
            || support.unique_universe_hash != fingerprint.unique_universe_hash
            || support.universe_stratum_counts != universe_stratum_counts
        {
            return Err(format!(
                "exceedance sampling seed {seed} did not reproduce the canonical pair universe"
            ));
        }

        let mut uniform_cutoffs = [None; 3];
        let mut uniform_stratum_sample_sizes = [[0usize; 5]; 3];
        let mut uniform_sample_hashes: [String; 3] = std::array::from_fn(|_| String::new());
        let mut running_strata = [0usize; 5];
        let mut cursor = 0usize;
        for size_index in 0..3 {
            let target = actual_sizes[size_index];
            if target > uniform.priority_pairs.len() || target < cursor {
                return Err(format!(
                    "uniform seed {seed} bottom-k prefix {target} is not available/nested"
                ));
            }
            for entry in &uniform.priority_pairs[cursor..target] {
                let (first, second) = decode_pair_id(entry.pair_id);
                let stratum = pair_min_cell_stratum(
                    logic_bits,
                    universe,
                    supports.as_slice(),
                    first,
                    second,
                )?;
                running_strata[stratum] += 1;
            }
            cursor = target;
            uniform_cutoffs[size_index] = target
                .checked_sub(1)
                .and_then(|index| uniform.priority_pairs.get(index).copied());
            uniform_stratum_sample_sizes[size_index] = running_strata;
            uniform_sample_hashes[size_index] = sample_set_hash(&uniform.priority_pairs[..target]);
        }

        let mut support_cutoffs = [[None; 5]; 3];
        let mut support_stratum_sample_sizes = [[0usize; 5]; 3];
        let mut support_sample_hashes: [String; 3] = std::array::from_fn(|_| String::new());
        for size_index in 0..3 {
            support_stratum_sample_sizes[size_index] = quotas_by_size[size_index];
            for stratum in 0..5 {
                let quota = quotas_by_size[size_index][stratum];
                support_cutoffs[size_index][stratum] = quota
                    .checked_sub(1)
                    .and_then(|index| support.pairs_by_stratum[stratum].get(index).copied());
            }
            support_sample_hashes[size_index] =
                support_sample_set_hash(&support.pairs_by_stratum, quotas_by_size[size_index])?;
        }
        seed_plans.push(WeightedExceedanceSeedPlan {
            pair_sample_seed: seed,
            uniform_cutoffs,
            support_cutoffs,
            uniform_stratum_sample_sizes,
            support_stratum_sample_sizes,
            uniform_sample_hashes,
            support_sample_hashes,
        });
    }

    let retained_cutoff_bytes = seed_plans
        .len()
        .checked_mul(
            3usize
                .checked_mul(size_of::<Option<PairPriority>>() * 6)
                .and_then(|bytes| bytes.checked_add(size_of::<[usize; 5]>() * 6))
                .ok_or_else(|| "exceedance retained cutoff byte count overflow".to_string())?,
        )
        .ok_or_else(|| "exceedance retained cutoff byte count overflow".to_string())?;
    Ok(WeightedExceedanceSamplingPlan {
        requested_sizes: WEIGHTED_EXCEEDANCE_SAMPLE_SIZES,
        actual_sizes,
        unique_universe_count: census_count,
        unique_universe_hash: fingerprint.unique_universe_hash,
        universe_stratum_counts,
        seeds: seed_plans,
        selection_seconds: started.elapsed().as_secs_f64(),
        preflight_heap_bytes,
        retained_cutoff_bytes,
    })
}

fn prepare_support_aware_pair_pools(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    pair_sample_seed: u64,
    target_shares: &[usize],
    mut notify_progress: impl FnMut(&str, usize, usize, usize) -> Result<(), String>,
) -> Result<SupportAwarePairPools, String> {
    let started = Instant::now();
    let supports = support_counts_by_active_slot(logic_bits, universe);
    let total_units = universe.units.len();
    let mut universe_stratum_counts = [0usize; 5];
    let mut fingerprint = BottomKPairSampler::new(0, 0);
    notify_progress("max6_screen_support_strata_count", 0, total_units, 0)?;
    let unique_count = visit_unique_eligible_pairs(
        universe,
        |first, second| {
            let stratum =
                pair_min_cell_stratum(logic_bits, universe, supports.as_slice(), first, second)?;
            universe_stratum_counts[stratum] = universe_stratum_counts[stratum]
                .checked_add(1)
                .ok_or_else(|| "support-aware universe stratum count overflow".to_string())?;
            fingerprint.consider(first, second, pair_sample_seed)
        },
        |done, total| {
            if done % 64 == 0 || done == total {
                notify_progress("max6_screen_support_strata_count", done, total, 0)
            } else {
                Ok(())
            }
        },
    )?;
    let fingerprint = fingerprint.finish()?;
    if fingerprint.unique_universe_count != unique_count
        || universe_stratum_counts.iter().sum::<usize>() != unique_count
    {
        return Err("support-aware stratum census disagrees with canonical pair count".to_string());
    }

    let plans =
        support_aware_allocation_plans(universe_stratum_counts, unique_count, target_shares)?;
    let mut heap_limits = [0usize; 5];
    for plan in plans.iter() {
        for stratum in 0..5 {
            heap_limits[stratum] = heap_limits[stratum].max(plan.quotas_by_size[2][stratum]);
        }
    }
    notify_progress(
        "max6_screen_support_pair_selection",
        0,
        total_units,
        unique_count,
    )?;
    let mut sampler = StratifiedBottomKPairSampler::new(pair_sample_seed, heap_limits);
    let selected_count = visit_unique_eligible_pairs(
        universe,
        |first, second| {
            let stratum =
                pair_min_cell_stratum(logic_bits, universe, supports.as_slice(), first, second)?;
            sampler.consider(first, second, stratum)
        },
        |done, total| {
            if done % 64 == 0 || done == total {
                notify_progress(
                    "max6_screen_support_pair_selection",
                    done,
                    total,
                    unique_count,
                )
            } else {
                Ok(())
            }
        },
    )?;
    let mut pools = sampler.finish()?;
    if pools.unique_universe_count != unique_count
        || pools.unique_universe_hash != fingerprint.unique_universe_hash
        || pools.universe_stratum_counts != universe_stratum_counts
        || selected_count != unique_count
    {
        return Err(
            "support-aware bottom-k selection did not reproduce the census universe".to_string(),
        );
    }
    pools.selection_seconds = started.elapsed().as_secs_f64();
    Ok(pools)
}

fn write_full_maximizer_pair_geometry(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    calibration: &Max6ScreenCalibration,
    null_seed: u64,
    out_prefix: &str,
) -> Result<String, String> {
    let maximizing_pairs = calibration.maximizing_pairs.as_ref().ok_or_else(|| {
        "sampled Max6 diagnostic requires full-scan maximizing pair identities".to_string()
    })?;
    if maximizing_pairs.len() != calibration.maxima.len() {
        return Err(format!(
            "full-maximizer pair count {} differs from full maxima count {}",
            maximizing_pairs.len(),
            calibration.maxima.len()
        ));
    }
    let supports = support_counts_by_active_slot(logic_bits, universe);
    let mut tsv = String::from(
        "null_index\tnull_seed\tfull_max_score\tfirst_global_row\tsecond_global_row\tsnp1\tchrom1\tpos1\tsnp2\tchrom2\tpos2\tn00\tn01\tn10\tn11\tmin_cell\tmaf1_binary\tmaf2_binary\tmin_maf_binary\tpairwise_r2_binary\tcanonical_owner_scan_unit\n",
    );
    for (null_index, (&full_score, pair)) in
        calibration.maxima.iter().zip(maximizing_pairs).enumerate()
    {
        let Some((first, second)) = *pair else {
            tsv.push_str(&format!(
                "{null_index}\t{null_seed}\t{full_score:.17}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n"
            ));
            continue;
        };
        if first >= second {
            return Err(format!(
                "full-Max6 maximizing pair is not canonical (i < j): ({first}, {second})"
            ));
        }
        let geometry = pair_binary_geometry(logic_bits, universe, &supports, first, second)?;
        let site1 = logic_bits
            .sites
            .get(first)
            .ok_or_else(|| format!("full-maximizer marker row {first} is out of range"))?;
        let site2 = logic_bits
            .sites
            .get(second)
            .ok_or_else(|| format!("full-maximizer marker row {second} is out of range"))?;
        tsv.push_str(&format!(
            "{null_index}\t{null_seed}\t{full_score:.17}\t{first}\t{second}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\tscan_unit_{}\n",
            site1.snp(),
            site1.chrom(),
            site1.pos,
            site2.snp(),
            site2.chrom(),
            site2.pos,
            geometry.counts[0],
            geometry.counts[1],
            geometry.counts[2],
            geometry.counts[3],
            geometry.min_cell,
            geometry.maf1,
            geometry.maf2,
            geometry.maf1.min(geometry.maf2),
            geometry.pairwise_r2,
            geometry.owner_scan_unit,
        ));
    }
    let path = format!("{out_prefix}.max6_grmmax6.full_maximizer_pair_geometry.tsv");
    write_atomic(&path, tsv.as_bytes())?;
    Ok(path)
}

fn sample_strata_snapshots(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    sample: &BottomKSample,
    actual_sizes: [usize; 4],
    maf_cutpoints: &[f64],
    r2_cutpoints: &[f64],
) -> Result<Vec<PairStrataCounts>, String> {
    let supports = support_counts_by_active_slot(logic_bits, universe);
    let mut running = PairStrataCounts::default();
    let mut snapshots = Vec::with_capacity(4);
    let mut target_index = 0usize;
    if actual_sizes[0] == 0 {
        while target_index < 4 && actual_sizes[target_index] == 0 {
            snapshots.push(running.clone());
            target_index += 1;
        }
    }
    for (pair_index, entry) in sample.priority_pairs.iter().enumerate() {
        let (first, second) = decode_pair_id(entry.pair_id);
        let first_slot = universe.pair_ownership.active_slot(first);
        let second_slot = universe.pair_ownership.active_slot(second);
        let owner = canonical_owner_unit(universe, first_slot, second_slot)?;
        let n11 = pair_n11(logic_bits, first, second);
        let n1 = supports[first_slot];
        let n2 = supports[second_slot];
        if n11 > n1 || n11 > n2 {
            return Err(format!(
                "sampled Max6 intersection count {n11} exceeds binary marker supports {n1}/{n2}"
            ));
        }
        let n10 = n1 - n11;
        let n01 = n2 - n11;
        let n00 = logic_bits
            .n_samples
            .checked_sub(n1 + n2 - n11)
            .ok_or_else(|| "sampled Max6 n00 count underflow".to_string())?;
        let min_cell = n00.min(n01).min(n10).min(n11);
        let maf1 = (n1 as f64 / logic_bits.n_samples as f64)
            .min(1.0 - n1 as f64 / logic_bits.n_samples as f64);
        let maf2 = (n2 as f64 / logic_bits.n_samples as f64)
            .min(1.0 - n2 as f64 / logic_bits.n_samples as f64);
        let min_maf = maf1.min(maf2);
        let denom = (n1 as f64)
            * (logic_bits.n_samples - n1) as f64
            * (n2 as f64)
            * (logic_bits.n_samples - n2) as f64;
        let r2 = if denom > 0.0 {
            let covariance_numerator =
                (logic_bits.n_samples as f64) * n11 as f64 - (n1 as f64) * (n2 as f64);
            (covariance_numerator * covariance_numerator / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        running.increment("owner_scan_unit", format!("scan_unit_{owner}"));
        running.increment("min_maf", cutpoint_bin(min_maf, maf_cutpoints));
        running.increment("pairwise_r2", cutpoint_bin(r2, r2_cutpoints));
        running.increment("min_cell", min_cell_stratum(min_cell).to_string());

        let count = pair_index + 1;
        while target_index < 4 && actual_sizes[target_index] == count {
            snapshots.push(running.clone());
            target_index += 1;
        }
    }
    while target_index < 4 {
        if actual_sizes[target_index] != sample.priority_pairs.len() {
            return Err("sampled Max6 strata counts did not reach requested prefix".to_string());
        }
        snapshots.push(running.clone());
        target_index += 1;
    }
    Ok(snapshots)
}

fn score_sampled_pair_chunk(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    entries: &[PairPriority],
    draws: &[logic_residual_screen::Max6ScreenNullDraw<'_>],
) -> Result<Vec<f64>, String> {
    let mut maxima = vec![f64::NEG_INFINITY; draws.len()];
    let mut sum11 = vec![0.0; draws.len()];
    let words = logic_bits.n_samples.div_ceil(64);
    let bits = logic_bits.bits();
    for entry in entries {
        let (first, second) = decode_pair_id(entry.pair_id);
        let first_start = first * logic_bits.row_words;
        let second_start = second * logic_bits.row_words;
        let first_row = &bits[first_start..first_start + words];
        let second_row = &bits[second_start..second_start + words];
        logic_residual_screen::update_max6_one_cell_pair_batch(
            first_row,
            second_row,
            logic_bits.n_samples,
            universe.pair_ownership.active_slot(first),
            universe.pair_ownership.active_slot(second),
            None,
            draws,
            sum11.as_mut_slice(),
            maxima.as_mut_slice(),
            None,
            None,
        )?;
    }
    Ok(maxima)
}

fn score_tail_sample_pair_chunk(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    entries: &[PairPriority],
    draws: &[logic_residual_screen::Max6ScreenNullDraw<'_>],
) -> Result<Vec<Option<TailPairScore>>, String> {
    let mut maxima = vec![f64::NEG_INFINITY; draws.len()];
    let mut sum11 = vec![0.0; draws.len()];
    let mut pair_scores = vec![None; draws.len()];
    let words = logic_bits.n_samples.div_ceil(64);
    let bits = logic_bits.bits();
    let mut winners = vec![None; draws.len()];
    for entry in entries {
        let pair = decode_pair_id(entry.pair_id);
        let first_start = pair.0 * logic_bits.row_words;
        let second_start = pair.1 * logic_bits.row_words;
        logic_residual_screen::update_max6_one_cell_pair_batch(
            &bits[first_start..first_start + words],
            &bits[second_start..second_start + words],
            logic_bits.n_samples,
            universe.pair_ownership.active_slot(pair.0),
            universe.pair_ownership.active_slot(pair.1),
            None,
            draws,
            sum11.as_mut_slice(),
            maxima.as_mut_slice(),
            None,
            Some(pair_scores.as_mut_slice()),
        )?;
        for (draw_index, score) in pair_scores.iter().enumerate() {
            let Some(score) = score else {
                continue;
            };
            let candidate = TailPairScore {
                pair,
                score: score.score,
                best_channel: score.gate,
            };
            if winners[draw_index].is_none_or(|current| full_rank_cmp(&candidate, &current).is_lt())
            {
                winners[draw_index] = Some(candidate);
            }
        }
    }
    Ok(winners)
}

/// Score every pair × null-draw cell in pair-major order with the production
/// exact Max6 screen scorer. An unscorable pair is represented as zero, which
/// preserves its non-exceedance contribution for every positive POT level.
fn score_pot_pair_chunk(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    entries: &[PairPriority],
    draws: &[logic_residual_screen::Max6ScreenNullDraw<'_>],
) -> Result<Vec<f64>, String> {
    if draws.is_empty() {
        return Err("POT sample scoring requires at least one null draw".to_string());
    }
    let output_len = entries
        .len()
        .checked_mul(draws.len())
        .ok_or_else(|| "POT sample score block size overflow".to_string())?;
    let mut scores = vec![0.0; output_len];
    let mut maxima = vec![f64::NEG_INFINITY; draws.len()];
    let mut sum11 = vec![0.0; draws.len()];
    let mut pair_scores = vec![None; draws.len()];
    let words = logic_bits.n_samples.div_ceil(64);
    let bits = logic_bits.bits();
    for (pair_index, entry) in entries.iter().enumerate() {
        let pair = decode_pair_id(entry.pair_id);
        if pair.0 >= pair.1 {
            return Err(format!(
                "POT sample pair is not canonical (i < j): ({}, {})",
                pair.0, pair.1
            ));
        }
        let first_start = pair.0 * logic_bits.row_words;
        let second_start = pair.1 * logic_bits.row_words;
        logic_residual_screen::update_max6_one_cell_pair_batch(
            &bits[first_start..first_start + words],
            &bits[second_start..second_start + words],
            logic_bits.n_samples,
            universe.pair_ownership.active_slot(pair.0),
            universe.pair_ownership.active_slot(pair.1),
            None,
            draws,
            sum11.as_mut_slice(),
            maxima.as_mut_slice(),
            None,
            Some(pair_scores.as_mut_slice()),
        )?;
        let row_start = pair_index * draws.len();
        for (draw_index, score) in pair_scores.iter().enumerate() {
            if let Some(score) = score {
                if !score.score.is_finite() || score.score < 0.0 {
                    return Err("POT exact Max6 score must be finite and nonnegative".to_string());
                }
                scores[row_start + draw_index] = score.score;
            }
        }
    }
    Ok(scores)
}

fn score_pot_pair_block(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    entries: &[PairPriority],
    draws: &[logic_residual_screen::Max6ScreenNullDraw<'_>],
    pool: Option<&ThreadPool>,
) -> Result<Vec<f64>, String> {
    const PARALLEL_PAIR_CHUNK: usize = 64;
    let chunks = if let Some(pool) = pool {
        pool.install(|| {
            entries
                .par_chunks(PARALLEL_PAIR_CHUNK)
                .map(|chunk| score_pot_pair_chunk(logic_bits, universe, chunk, draws))
                .collect::<Result<Vec<_>, _>>()
        })?
    } else {
        vec![score_pot_pair_chunk(logic_bits, universe, entries, draws)?]
    };
    let total_len = entries
        .len()
        .checked_mul(draws.len())
        .ok_or_else(|| "POT pair block score size overflow".to_string())?;
    let mut combined = Vec::new();
    combined
        .try_reserve_exact(total_len)
        .map_err(|error| format!("unable to reserve POT pair block scores: {error}"))?;
    combined.extend(chunks.into_iter().flatten());
    if combined.len() != total_len {
        return Err("parallel POT pair score chunks returned an incomplete block".to_string());
    }
    Ok(combined)
}

fn score_tail_sample_segment(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    entries: &[PairPriority],
    draws: &[logic_residual_screen::Max6ScreenNullDraw<'_>],
    pool: Option<&ThreadPool>,
    mut notify_progress: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<Vec<Option<TailPairScore>>, String> {
    if entries.is_empty() {
        return Ok(vec![None; draws.len()]);
    }
    let mut winners = vec![None; draws.len()];
    const PAIR_PROGRESS_BLOCK: usize = 1_000_000;
    let mut done = 0usize;
    for pair_block in entries.chunks(PAIR_PROGRESS_BLOCK) {
        let chunk_winners = if let Some(pool) = pool {
            pool.install(|| {
                pair_block
                    .par_chunks(4096)
                    .map(|chunk| score_tail_sample_pair_chunk(logic_bits, universe, chunk, draws))
                    .collect::<Result<Vec<_>, _>>()
            })?
        } else {
            vec![score_tail_sample_pair_chunk(
                logic_bits, universe, pair_block, draws,
            )?]
        };
        for chunk in chunk_winners {
            for (index, candidate) in chunk.into_iter().enumerate() {
                let Some(candidate) = candidate else {
                    continue;
                };
                if winners[index].is_none_or(|current| full_rank_cmp(&candidate, &current).is_lt())
                {
                    winners[index] = Some(candidate);
                }
            }
        }
        done += pair_block.len();
        notify_progress(done, entries.len())?;
    }
    Ok(winners)
}

pub(crate) fn prepare_tail_audit_sample(
    universe: &Max6ScreenPreparedUniverse,
    pair_sample_seed: u64,
    mut notify_progress: impl FnMut(usize, usize, usize) -> Result<(), String>,
) -> Result<TailAuditSample, String> {
    let (sample, selection_seconds) =
        select_unique_eligible_pairs(universe, pair_sample_seed, |done, total, seen| {
            notify_progress(done, total, seen)
        })?;
    let actual_sizes =
        TAIL_AUDIT_SAMPLE_SIZES.map(|target| target.min(sample.unique_universe_count));
    if actual_sizes
        .iter()
        .any(|&size| size > sample.priority_pairs.len())
    {
        return Err("tail-audit bottom-k sample is shorter than an available target".to_string());
    }
    Ok(TailAuditSample {
        pair_sample_seed,
        sample,
        actual_sizes,
        selection_seconds,
    })
}

pub(crate) fn score_tail_audit_sampled_maxima(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    sample: TailAuditSample,
    null_draws: &[Vec<f64>],
    pool: Option<&ThreadPool>,
    mut notify_progress: impl FnMut(usize, usize, usize) -> Result<(), String>,
) -> Result<TailAuditSampledMaxima, String> {
    if null_draws.is_empty() {
        return Err("tail-audit sampled scoring requires at least one null draw".to_string());
    }
    if sample.sample.unique_universe_count == 0 {
        return Err("tail-audit sampled scoring requires a non-empty pair universe".to_string());
    }
    let mut winners = (0..TAIL_AUDIT_SAMPLE_SIZES.len())
        .map(|_| vec![None; null_draws.len()])
        .collect::<Vec<_>>();
    let mut preprocessing_seconds = 0.0;
    let mut scoring_seconds_by_target = [0.0; 4];
    let batch_count = null_draws
        .len()
        .div_ceil(super::MAX6_SCREEN_NULL_BATCH_SIZE);
    for (batch_number, null_batch) in null_draws
        .chunks(super::MAX6_SCREEN_NULL_BATCH_SIZE)
        .enumerate()
    {
        let batch_index = batch_number * super::MAX6_SCREEN_NULL_BATCH_SIZE;
        let preprocess_started = Instant::now();
        let lookups = null_batch
            .iter()
            .map(|residual| PackedYSumLookup::build(residual, logic_bits.n_samples))
            .collect::<Result<Vec<_>, _>>()?;
        let margins = null_batch
            .iter()
            .zip(lookups.iter())
            .map(|(residual, lookup)| {
                OneCellMarginTable::build_for_rows_with_pool(
                    logic_bits.bits(),
                    logic_bits.row_words,
                    universe.active_rows.as_slice(),
                    residual,
                    logic_bits.n_samples,
                    lookup,
                    pool,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let draws = null_batch
            .iter()
            .zip(lookups.iter())
            .zip(margins.iter())
            .map(
                |((residual, lookup), margins)| logic_residual_screen::Max6ScreenNullDraw {
                    residual: residual.as_slice(),
                    lookup,
                    margins,
                },
            )
            .collect::<Vec<_>>();
        preprocessing_seconds += preprocess_started.elapsed().as_secs_f64();

        let mut cumulative = vec![None; null_batch.len()];
        let mut previous_size = 0usize;
        for target_index in 0..TAIL_AUDIT_SAMPLE_SIZES.len() {
            let target_size = sample.actual_sizes[target_index];
            if target_size > previous_size {
                let scoring_started = Instant::now();
                let segment = score_tail_sample_segment(
                    logic_bits,
                    universe,
                    &sample.sample.priority_pairs[previous_size..target_size],
                    draws.as_slice(),
                    pool,
                    |segment_done, segment_total| {
                        let work_done = batch_number
                            .saturating_mul(sample.actual_sizes[3])
                            .saturating_add(previous_size)
                            .saturating_add(segment_done);
                        let work_total = batch_count.saturating_mul(sample.actual_sizes[3]);
                        notify_progress(
                            work_done.min(work_total),
                            work_total,
                            sample.sample.unique_universe_count,
                        )
                        .map_err(|error| {
                            format!("{error}; tail sampled segment total={segment_total}")
                        })
                    },
                )?;
                scoring_seconds_by_target[target_index] += scoring_started.elapsed().as_secs_f64();
                for (current, candidate) in cumulative.iter_mut().zip(segment) {
                    if let Some(candidate) = candidate {
                        if current
                            .is_none_or(|previous| full_rank_cmp(&candidate, &previous).is_lt())
                        {
                            *current = Some(candidate);
                        }
                    }
                }
                previous_size = target_size;
            }
            let target_start = batch_index;
            let target_end = target_start + null_batch.len();
            winners[target_index][target_start..target_end].copy_from_slice(cumulative.as_slice());
        }
    }
    if winners
        .iter()
        .any(|target| target.len() != null_draws.len())
    {
        return Err("tail-audit sampled maxima have an invalid null count".to_string());
    }
    Ok(TailAuditSampledMaxima {
        sample,
        winners,
        preprocessing_seconds,
        scoring_seconds_by_target,
    })
}

pub(crate) fn run_support_aware_tail_audit_sampling(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    pair_sample_seed: u64,
    null_draws: &[Vec<f64>],
    pool: Option<&ThreadPool>,
    target_shares: &[usize],
    mut notify_progress: impl FnMut(&str, usize, usize, usize) -> Result<(), String>,
) -> Result<Vec<SupportAwareTailAuditMaxima>, String> {
    if null_draws.is_empty() {
        return Err("support-aware tail audit requires at least one null draw".to_string());
    }
    let pools = prepare_support_aware_pair_pools(
        logic_bits,
        universe,
        pair_sample_seed,
        target_shares,
        |stage, done, total, count| notify_progress(stage, done, total, count),
    )?;
    if pools.unique_universe_count == 0 {
        return Err("support-aware tail audit requires a non-empty pair universe".to_string());
    }
    let plans = support_aware_allocation_plans(
        pools.universe_stratum_counts,
        pools.unique_universe_count,
        target_shares,
    )?;
    let mut results = Vec::with_capacity(plans.len());
    for plan in plans {
        let share_percent = plan.share_percent;
        let (sample, cutoffs_by_size, sample_hashes) =
            pools.materialize_sample(&plan, pair_sample_seed)?;
        let actual_sizes = plan.actual_sizes;
        let quotas_by_size = plan.quotas_by_size;
        let materialized_sample_storage_bytes = sample.sample.storage_capacity_bytes;
        let scored = score_tail_audit_sampled_maxima(
            logic_bits,
            universe,
            sample,
            null_draws,
            pool,
            |done, total, count| {
                notify_progress(
                    &format!("max6_screen_support_sample_scoring_{share_percent}"),
                    done,
                    total,
                    count,
                )
            },
        )?;
        if scored.winners.len() != TAIL_AUDIT_SAMPLE_SIZES.len()
            || scored
                .winners
                .iter()
                .any(|target| target.len() != null_draws.len())
        {
            return Err(format!(
                "support-aware {share_percent}% sample produced an invalid null result shape"
            ));
        }
        let sampled_maxima = SupportAwareTailAuditMaxima {
            share_percent,
            winners: scored.winners.into_iter().take(3).collect(),
            actual_sizes,
            quotas_by_size,
            cutoffs_by_size,
            sample_hashes,
            universe_count: pools.unique_universe_count,
            universe_hash: pools.unique_universe_hash.clone(),
            universe_stratum_counts: pools.universe_stratum_counts,
            selection_seconds: pools.selection_seconds,
            selection_storage_bytes: pools.storage_capacity_bytes,
            materialized_sample_storage_bytes,
            preprocessing_seconds: scored.preprocessing_seconds,
            scoring_seconds_by_size: std::array::from_fn(|index| {
                scored.scoring_seconds_by_target[index]
            }),
        };
        if sampled_maxima
            .winners
            .iter()
            .any(|target| target.len() != null_draws.len())
        {
            return Err(format!(
                "support-aware {share_percent}% sample lost a null replicate"
            ));
        }
        results.push(sampled_maxima);
    }
    Ok(results)
}

fn score_sampled_pair_segment(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    entries: &[PairPriority],
    draws: &[logic_residual_screen::Max6ScreenNullDraw<'_>],
    previous_maxima: &[f64],
    pool: Option<&ThreadPool>,
) -> Result<Vec<f64>, String> {
    if entries.is_empty() {
        return Ok(previous_maxima.to_vec());
    }
    let chunk_maxima = if let Some(pool) = pool {
        pool.install(|| {
            entries
                .par_chunks(4096)
                .map(|chunk| score_sampled_pair_chunk(logic_bits, universe, chunk, draws))
                .collect::<Result<Vec<_>, _>>()
        })?
    } else {
        vec![score_sampled_pair_chunk(
            logic_bits, universe, entries, draws,
        )?]
    };
    let mut maxima = previous_maxima.to_vec();
    for chunk in chunk_maxima {
        for (current, chunk_max) in maxima.iter_mut().zip(chunk) {
            if chunk_max.total_cmp(current).is_gt() {
                *current = chunk_max;
            }
        }
    }
    Ok(maxima)
}

fn score_nested_sample_sizes(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    sample: &BottomKSample,
    actual_sizes: [usize; 4],
    null_draws: &[Vec<f64>],
    pool: Option<&ThreadPool>,
    notify_progress: &mut impl FnMut(&str, usize, usize, usize) -> Result<(), String>,
) -> Result<(Vec<Vec<f64>>, f64, [f64; 4]), String> {
    let mut sampled_maxima = vec![Vec::with_capacity(null_draws.len()); 4];
    let mut preprocessing_seconds = 0.0;
    let mut scoring_seconds = [0.0; 4];
    let mut distinct_targets = Vec::<usize>::new();
    for size in actual_sizes {
        if distinct_targets.last().copied() != Some(size) {
            distinct_targets.push(size);
        }
    }
    let batches = null_draws
        .chunks(super::MAX6_SCREEN_NULL_BATCH_SIZE)
        .collect::<Vec<_>>();
    for (batch_index, null_batch) in batches.iter().enumerate() {
        let preprocess_started = Instant::now();
        let lookups = null_batch
            .iter()
            .map(|residual| PackedYSumLookup::build(residual, logic_bits.n_samples))
            .collect::<Result<Vec<_>, _>>()?;
        let margins = null_batch
            .iter()
            .zip(lookups.iter())
            .map(|(residual, lookup)| {
                OneCellMarginTable::build_for_rows_with_pool(
                    logic_bits.bits(),
                    logic_bits.row_words,
                    universe.active_rows.as_slice(),
                    residual,
                    logic_bits.n_samples,
                    lookup,
                    pool,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let draws = null_batch
            .iter()
            .zip(lookups.iter())
            .zip(margins.iter())
            .map(
                |((residual, lookup), margins)| logic_residual_screen::Max6ScreenNullDraw {
                    residual: residual.as_slice(),
                    lookup,
                    margins,
                },
            )
            .collect::<Vec<_>>();
        preprocessing_seconds += preprocess_started.elapsed().as_secs_f64();

        let mut maxima = vec![f64::NEG_INFINITY; null_batch.len()];
        let mut cursor = 0usize;
        let mut cumulative_scoring_seconds = 0.0;
        for &target in distinct_targets.iter() {
            let started = Instant::now();
            maxima = score_sampled_pair_segment(
                logic_bits,
                universe,
                &sample.priority_pairs[cursor..target],
                draws.as_slice(),
                maxima.as_slice(),
                pool,
            )?;
            let elapsed = started.elapsed().as_secs_f64();
            cumulative_scoring_seconds += elapsed;
            for target_index in 0..4 {
                if actual_sizes[target_index] == target {
                    scoring_seconds[target_index] += cumulative_scoring_seconds;
                }
            }
            cursor = target;
            for target_index in 0..4 {
                if actual_sizes[target_index] == target {
                    sampled_maxima[target_index].extend(maxima.iter().map(|value| {
                        if value.is_finite() {
                            *value
                        } else {
                            0.0
                        }
                    }));
                }
            }
        }
        notify_progress(
            "max6_screen_sample_scoring",
            ((batch_index + 1) * super::MAX6_SCREEN_NULL_BATCH_SIZE).min(null_draws.len()),
            null_draws.len(),
            *distinct_targets.last().unwrap_or(&0),
        )?;
    }
    Ok((sampled_maxima, preprocessing_seconds, scoring_seconds))
}

fn quantile_type7_sorted(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() || !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let location = probability * (sorted.len() - 1) as f64;
    let lower = location.floor() as usize;
    let upper = location.ceil() as usize;
    let fraction = location - lower as f64;
    Some(sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction)
}

fn quantile_type7(values: &[f64], probability: f64) -> Option<f64> {
    let mut sorted = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    quantile_type7_sorted(&sorted, probability)
}

const EXCEEDANCE_FIXED_GRID_MAX: f64 = 50.0;
const EXCEEDANCE_FIXED_GRID_STEP: f64 = 0.25;
const EXCEEDANCE_FIXED_GRID_POINTS: usize = 201;
const WILSON_95_Z: f64 = 1.959_963_984_540_054;
const BOOTSTRAP_95_LOWER: f64 = 0.025;
const BOOTSTRAP_95_UPPER: f64 = 0.975;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExceedanceAuditThreshold {
    pub(crate) score: f64,
    pub(crate) sources: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExceedanceAuditGrid {
    pub(crate) thresholds: Vec<ExceedanceAuditThreshold>,
    pub(crate) fixed_grid_points: usize,
    pub(crate) maxima_above_fixed_grid_count: usize,
    pub(crate) max_maxima_above_fixed_grid: Option<f64>,
    pub(crate) finite_maxima_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AuditConfidenceInterval {
    estimate: f64,
    lower: f64,
    upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct AuditMetricValue {
    value: Option<f64>,
    na_reason: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BootstrapMeanInterval {
    estimate: f64,
    lower: f64,
    upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct BootstrapThetaInterval {
    estimate: AuditMetricValue,
    interval: Option<BootstrapMeanInterval>,
    valid_replicates: usize,
    total_replicates: usize,
    na_reason: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg(test)]
struct PairedThresholdCountSummary {
    null_count: usize,
    full_fwer: AuditConfidenceInterval,
    sample_fwer: AuditConfidenceInterval,
    full_lambda: f64,
    sample_lambda: f64,
    lambda_bias: f64,
    relative_bias: AuditMetricValue,
    count_rmse: f64,
    abs_error_median: f64,
    abs_error_p90: f64,
    abs_error_p95: f64,
    full_lambda_bootstrap: BootstrapMeanInterval,
    sample_lambda_bootstrap: BootstrapMeanInterval,
    full_theta: BootstrapThetaInterval,
    sample_theta: BootstrapThetaInterval,
    zero_exceedance_classification_accuracy: f64,
}

/// Build the preregistered fixed score grid and append type-7 audit anchors from
/// the same calibration-null maxima. The anchors are descriptive only and do
/// not constitute independent validation thresholds.
pub(crate) fn build_exceedance_audit_grid(maxima: &[f64]) -> Result<ExceedanceAuditGrid, String> {
    if maxima.is_empty() {
        return Err("cannot build exceedance audit anchors without finite full maxima".to_string());
    }
    if maxima.iter().any(|score| !score.is_finite()) {
        return Err("full-MaxT maxima for exceedance anchors must all be finite".to_string());
    }
    let finite_maxima = maxima.to_vec();

    let mut thresholds = (0..EXCEEDANCE_FIXED_GRID_POINTS)
        .map(|index| ExceedanceAuditThreshold {
            score: index as f64 * EXCEEDANCE_FIXED_GRID_STEP,
            sources: vec!["fixed_grid".to_string()],
        })
        .collect::<Vec<_>>();

    let mut sorted_maxima = finite_maxima.clone();
    sorted_maxima.sort_by(f64::total_cmp);
    for (label, probability) in [
        ("q80", 0.80),
        ("q90", 0.90),
        ("q95", 0.95),
        ("q97.5", 0.975),
        ("q99", 0.99),
    ] {
        let anchor = quantile_type7_sorted(sorted_maxima.as_slice(), probability)
            .ok_or_else(|| format!("failed to compute in-sample audit anchor {label}"))?;
        let source = format!("in_sample_audit_anchor:{label}");
        match thresholds.binary_search_by(|probe| probe.score.total_cmp(&anchor)) {
            Ok(index) => thresholds[index].sources.push(source),
            Err(index) => thresholds.insert(
                index,
                ExceedanceAuditThreshold {
                    score: anchor,
                    sources: vec![source],
                },
            ),
        }
    }

    let maxima_above_fixed_grid = finite_maxima
        .iter()
        .copied()
        .filter(|score| *score > EXCEEDANCE_FIXED_GRID_MAX)
        .collect::<Vec<_>>();
    Ok(ExceedanceAuditGrid {
        thresholds,
        fixed_grid_points: EXCEEDANCE_FIXED_GRID_POINTS,
        maxima_above_fixed_grid_count: maxima_above_fixed_grid.len(),
        max_maxima_above_fixed_grid: maxima_above_fixed_grid.into_iter().max_by(f64::total_cmp),
        finite_maxima_count: finite_maxima.len(),
    })
}

#[inline]
#[cfg(test)]
fn strict_exceedance_count(scores: &[f64], threshold: f64) -> u64 {
    scores.iter().filter(|score| **score > threshold).count() as u64
}

fn wilson_fwer_interval(
    successes: usize,
    trials: usize,
) -> Result<AuditConfidenceInterval, String> {
    if trials == 0 {
        return Err("Wilson interval requires at least one null replicate".to_string());
    }
    if successes > trials {
        return Err(format!(
            "FWER success count {successes} exceeds null replicate count {trials}"
        ));
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = WILSON_95_Z * WILSON_95_Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = WILSON_95_Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    Ok(AuditConfidenceInterval {
        estimate: p,
        lower: (center - half_width).max(0.0),
        upper: (center + half_width).min(1.0),
    })
}

fn relative_bias_metric(bias: f64, lambda_full: f64) -> AuditMetricValue {
    if !bias.is_finite() || !lambda_full.is_finite() {
        return AuditMetricValue {
            value: None,
            na_reason: Some("nonfinite_input"),
        };
    }
    if lambda_full == 0.0 {
        return AuditMetricValue {
            value: None,
            na_reason: Some("lambda_full_is_zero"),
        };
    }
    if lambda_full < 0.0 {
        return AuditMetricValue {
            value: None,
            na_reason: Some("lambda_full_is_negative"),
        };
    }
    AuditMetricValue {
        value: Some(bias / lambda_full),
        na_reason: None,
    }
}

fn theta_eff_metric(fwer_full: f64, lambda_full: f64) -> AuditMetricValue {
    if !fwer_full.is_finite() || !lambda_full.is_finite() {
        return AuditMetricValue {
            value: None,
            na_reason: Some("nonfinite_input"),
        };
    }
    if !(0.0..=1.0).contains(&fwer_full) {
        return AuditMetricValue {
            value: None,
            na_reason: Some("fwer_full_outside_unit_interval"),
        };
    }
    if lambda_full == 0.0 {
        return AuditMetricValue {
            value: None,
            na_reason: Some("lambda_full_is_zero"),
        };
    }
    if lambda_full < 0.0 {
        return AuditMetricValue {
            value: None,
            na_reason: Some("lambda_full_is_negative"),
        };
    }
    if fwer_full == 1.0 {
        return AuditMetricValue {
            value: None,
            na_reason: Some("fwer_full_is_one"),
        };
    }
    AuditMetricValue {
        value: Some(-(1.0 - fwer_full).ln() / lambda_full),
        na_reason: None,
    }
}

fn bootstrap_mean_interval(
    values: &[f64],
    seed: u64,
    bootstrap_replicates: usize,
) -> Result<BootstrapMeanInterval, String> {
    if values.is_empty() {
        return Err("bootstrap interval requires at least one paired null value".to_string());
    }
    if bootstrap_replicates == 0 {
        return Err("bootstrap interval requires at least one resample".to_string());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("bootstrap interval values must be finite".to_string());
    }
    let estimate = values.iter().sum::<f64>() / values.len() as f64;
    let mut state = seed;
    let mut means = Vec::with_capacity(bootstrap_replicates);
    for _ in 0..bootstrap_replicates {
        let mut total = 0.0;
        for _ in 0..values.len() {
            let index = (splitmix64_next(&mut state) % values.len() as u64) as usize;
            total += values[index];
        }
        means.push(total / values.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    Ok(BootstrapMeanInterval {
        estimate,
        lower: quantile_type7_sorted(means.as_slice(), BOOTSTRAP_95_LOWER).unwrap(),
        upper: quantile_type7_sorted(means.as_slice(), BOOTSTRAP_95_UPPER).unwrap(),
    })
}

fn bootstrap_theta_interval(
    exceedance_counts: &[u64],
    seed: u64,
    bootstrap_replicates: usize,
) -> Result<BootstrapThetaInterval, String> {
    if exceedance_counts.is_empty() {
        return Err("theta bootstrap requires at least one null replicate".to_string());
    }
    if bootstrap_replicates == 0 {
        return Err("theta bootstrap requires at least one resample".to_string());
    }
    let point_lambda = exceedance_counts
        .iter()
        .map(|count| *count as f64)
        .sum::<f64>()
        / exceedance_counts.len() as f64;
    let point_fwer = exceedance_counts.iter().filter(|count| **count > 0).count() as f64
        / exceedance_counts.len() as f64;
    let estimate = theta_eff_metric(point_fwer, point_lambda);
    if let Some(reason) = estimate.na_reason {
        return Ok(BootstrapThetaInterval {
            estimate,
            interval: None,
            valid_replicates: 0,
            total_replicates: bootstrap_replicates,
            na_reason: Some(reason),
        });
    }

    let mut state = seed;
    let mut theta_values = Vec::with_capacity(bootstrap_replicates);
    let mut invalid_domain_draw = false;
    for _ in 0..bootstrap_replicates {
        let mut total = 0u128;
        let mut any_exceedance = 0usize;
        for _ in 0..exceedance_counts.len() {
            let index = (splitmix64_next(&mut state) % exceedance_counts.len() as u64) as usize;
            let count = exceedance_counts[index];
            total = total.saturating_add(count as u128);
            any_exceedance += usize::from(count > 0);
        }
        let lambda = total as f64 / exceedance_counts.len() as f64;
        let fwer = any_exceedance as f64 / exceedance_counts.len() as f64;
        let theta = theta_eff_metric(fwer, lambda);
        if let Some(value) = theta.value {
            theta_values.push(value);
        } else {
            invalid_domain_draw = true;
        }
    }
    if invalid_domain_draw || theta_values.is_empty() {
        return Ok(BootstrapThetaInterval {
            estimate,
            interval: None,
            valid_replicates: theta_values.len(),
            total_replicates: bootstrap_replicates,
            na_reason: Some("bootstrap_resample_outside_theta_domain"),
        });
    }
    theta_values.sort_by(f64::total_cmp);
    let theta_point_estimate = estimate.value.unwrap();
    Ok(BootstrapThetaInterval {
        estimate,
        interval: Some(BootstrapMeanInterval {
            estimate: theta_point_estimate,
            lower: quantile_type7_sorted(theta_values.as_slice(), BOOTSTRAP_95_LOWER).unwrap(),
            upper: quantile_type7_sorted(theta_values.as_slice(), BOOTSTRAP_95_UPPER).unwrap(),
        }),
        valid_replicates: theta_values.len(),
        total_replicates: bootstrap_replicates,
        na_reason: None,
    })
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    splitmix64(*state)
}

fn stratified_ht_estimate(
    universe_counts: &[usize; 5],
    sample_sizes: &[usize; 5],
    sampled_exceedances: &[u64; 5],
) -> Result<AuditMetricValue, String> {
    let mut estimate = 0.0;
    for stratum in 0..5 {
        let universe = universe_counts[stratum];
        let sampled = sample_sizes[stratum];
        let exceedances = sampled_exceedances[stratum] as usize;
        if sampled > universe {
            return Err(format!(
                "sample count {sampled} exceeds true universe count {universe} in stratum {stratum}"
            ));
        }
        if exceedances > sampled {
            return Err(format!(
                "sample exceedance count {exceedances} exceeds sample count {sampled} in stratum {stratum}"
            ));
        }
        if universe > 0 && sampled == 0 {
            return Ok(AuditMetricValue {
                value: None,
                na_reason: Some("nonempty_stratum_has_zero_sample"),
            });
        }
        if sampled > 0 {
            estimate += universe as f64 / sampled as f64 * exceedances as f64;
        }
    }
    Ok(AuditMetricValue {
        value: Some(estimate),
        na_reason: None,
    })
}

fn unconditional_uniform_ht_estimate(
    universe_size: usize,
    sample_size: usize,
    sampled_exceedances: usize,
) -> Result<AuditMetricValue, String> {
    if sample_size > universe_size {
        return Err(format!(
            "uniform sample size {sample_size} exceeds unique universe size {universe_size}"
        ));
    }
    if sampled_exceedances > sample_size {
        return Err(format!(
            "sampled exceedance count {sampled_exceedances} exceeds sample size {sample_size}"
        ));
    }
    if sample_size == 0 {
        return Ok(AuditMetricValue {
            value: None,
            na_reason: Some("uniform_sample_is_empty"),
        });
    }
    Ok(AuditMetricValue {
        value: Some(universe_size as f64 / sample_size as f64 * sampled_exceedances as f64),
        na_reason: None,
    })
}

#[cfg(test)]
fn paired_threshold_count_summary(
    full_counts: &[u64],
    sample_counts: &[u64],
    bootstrap_seed: u64,
    bootstrap_replicates: usize,
) -> Result<PairedThresholdCountSummary, String> {
    if full_counts.is_empty() || full_counts.len() != sample_counts.len() {
        return Err(format!(
            "paired threshold summaries require equally sized, nonempty null vectors (full={}, sample={})",
            full_counts.len(),
            sample_counts.len()
        ));
    }
    let n = full_counts.len();
    let full_successes = full_counts.iter().filter(|count| **count > 0).count();
    let sample_successes = sample_counts.iter().filter(|count| **count > 0).count();
    let full_lambda = full_counts.iter().map(|count| *count as f64).sum::<f64>() / n as f64;
    let sample_lambda = sample_counts.iter().map(|count| *count as f64).sum::<f64>() / n as f64;
    let lambda_bias = sample_lambda - full_lambda;
    let errors = full_counts
        .iter()
        .zip(sample_counts)
        .map(|(&full, &sample)| sample as f64 - full as f64)
        .collect::<Vec<_>>();
    let abs_errors = errors.iter().map(|error| error.abs()).collect::<Vec<_>>();
    let rmse = (errors.iter().map(|error| error * error).sum::<f64>() / n as f64).sqrt();
    let mut sorted_abs_errors = abs_errors.clone();
    sorted_abs_errors.sort_by(f64::total_cmp);
    let zero_matches = full_counts
        .iter()
        .zip(sample_counts)
        .filter(|(full, sample)| (**full == 0) == (**sample == 0))
        .count();
    let full_count_values = full_counts
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let sample_count_values = sample_counts
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    Ok(PairedThresholdCountSummary {
        null_count: n,
        full_fwer: wilson_fwer_interval(full_successes, n)?,
        sample_fwer: wilson_fwer_interval(sample_successes, n)?,
        full_lambda,
        sample_lambda,
        lambda_bias,
        relative_bias: relative_bias_metric(lambda_bias, full_lambda),
        count_rmse: rmse,
        abs_error_median: quantile_type7_sorted(sorted_abs_errors.as_slice(), 0.5).unwrap(),
        abs_error_p90: quantile_type7_sorted(sorted_abs_errors.as_slice(), 0.9).unwrap(),
        abs_error_p95: quantile_type7_sorted(sorted_abs_errors.as_slice(), 0.95).unwrap(),
        full_lambda_bootstrap: bootstrap_mean_interval(
            full_count_values.as_slice(),
            bootstrap_seed,
            bootstrap_replicates,
        )?,
        sample_lambda_bootstrap: bootstrap_mean_interval(
            sample_count_values.as_slice(),
            bootstrap_seed ^ 0xA5A5_5A5A_39C6_E7B1,
            bootstrap_replicates,
        )?,
        full_theta: bootstrap_theta_interval(
            full_counts,
            bootstrap_seed ^ 0xD1B5_4A32_D192_ED03,
            bootstrap_replicates,
        )?,
        sample_theta: bootstrap_theta_interval(
            sample_counts,
            bootstrap_seed ^ 0x94D0_49BB_1331_11EB,
            bootstrap_replicates,
        )?,
        zero_exceedance_classification_accuracy: zero_matches as f64 / n as f64,
    })
}

fn sample_variance(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Some(
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64,
    )
}

fn metric_tsv(value: &AuditMetricValue) -> String {
    value
        .value
        .map(|number| format!("{number:.17}"))
        .unwrap_or_else(|| format!("NA:{}", value.na_reason.unwrap_or("unspecified")))
}

fn bootstrap_seed_for_threshold(seed: u64, threshold_index: usize, design_index: usize) -> u64 {
    splitmix64(
        seed ^ (threshold_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (design_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
    )
}

fn calibration_null_draws_hash(draws: &[Vec<f64>]) -> String {
    let mut hash = fnv1a64_update(FNV_OFFSET, &(draws.len() as u64).to_le_bytes());
    for draw in draws {
        hash = fnv1a64_update(hash, &(draw.len() as u64).to_le_bytes());
        for value in draw {
            hash = fnv1a64_update(hash, &value.to_bits().to_le_bytes());
        }
    }
    format!("fnv1a64_calibration_null_f64_bits_v1:{hash:016x}")
}

/// Write the exact full and sampled exceedance-process audit. Every full and
/// sampled count comes from the same exact pair traversal and calibration-null
/// vectors; calibration maxima are used only for in-sample audit anchors.
pub(crate) fn write_weighted_exceedance_audit(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    calibration_nulls: &[Vec<f64>],
    calibration_maxima: &[f64],
    grid: &ExceedanceAuditGrid,
    sampling_plan: &WeightedExceedanceSamplingPlan,
    scan: &WeightedExceedanceFullScan,
    null_stream_seed: u64,
    bootstrap_seed: u64,
    out_prefix: &str,
) -> Result<WeightedExceedanceReportPaths, String> {
    let report_started = Instant::now();
    let null_count = calibration_maxima.len();
    let threshold_count = grid.thresholds.len();
    let slot_count = sampling_plan.sample_slot_count();
    if null_count == 0
        || calibration_nulls.len() != null_count
        || calibration_maxima.iter().any(|value| !value.is_finite())
        || scan.threshold_count != threshold_count
        || scan.sample_slot_count != slot_count
        || scan.top_pairs.len() != null_count
        || scan.unique_pair_count != sampling_plan.unique_universe_count
    {
        return Err("weighted exceedance audit inputs have inconsistent dimensions".to_string());
    }
    let expected_full = null_count
        .checked_mul(5)
        .and_then(|value| value.checked_mul(threshold_count))
        .ok_or_else(|| "weighted exceedance full output shape overflow".to_string())?;
    let expected_sampled = null_count
        .checked_mul(slot_count)
        .and_then(|value| value.checked_mul(5))
        .and_then(|value| value.checked_mul(threshold_count))
        .ok_or_else(|| "weighted exceedance sampled output shape overflow".to_string())?;
    if scan.full_counts.len() != expected_full || scan.sampled_counts.len() != expected_sampled {
        return Err(format!(
            "weighted exceedance count output has invalid shape: full {} != {}, sampled {} != {}",
            scan.full_counts.len(),
            expected_full,
            scan.sampled_counts.len(),
            expected_sampled
        ));
    }

    let per_null_path = format!("{out_prefix}.max6_grmmax6.exceedance_audit_per_null.tsv");
    let summary_path = format!("{out_prefix}.max6_grmmax6.exceedance_audit_summary.tsv");
    let top_pairs_path = format!("{out_prefix}.max6_grmmax6.exceedance_audit_top4096.tsv");
    let manifest_path = format!("{out_prefix}.max6_grmmax6.exceedance_audit_manifest.json");
    let total_universe = sampling_plan.unique_universe_count;
    let support_counts = support_counts_by_active_slot(logic_bits, universe);

    write_atomic_stream(per_null_path.as_str(), |writer| {
        writeln!(writer, "null_index\tthreshold_index\tthreshold\tthreshold_sources\testimator\tdesign\tpair_sample_seed\tpair_sample_hash\trequested_M\tactual_M\tfull_C\tfull_c0\tfull_c1\tfull_c2\tfull_c3\tfull_c4\tsample_c0\tsample_c1\tsample_c2\tsample_c3\tsample_c4\tsampled_total\tlambda_hat\tglobal_srs_sensitivity\tzero_exceedance_match")
            .map_err(|error| error.to_string())?;
        for threshold_index in 0..threshold_count {
            let threshold = &grid.thresholds[threshold_index];
            let source_labels = threshold.sources.join(";");
            for null_index in 0..null_count {
                let full_strata = std::array::from_fn::<_, 5, _>(|stratum| {
                    scan.full_counts[(null_index * 5 + stratum) * threshold_count + threshold_index]
                        as u64
                });
                let full_total = full_strata.iter().sum::<u64>();
                for seed_index in 0..sampling_plan.seeds.len() {
                    let seed_plan = &sampling_plan.seeds[seed_index];
                    for size_index in 0..3 {
                        let uniform_slot = seed_index * 6 + size_index;
                        let support_slot = seed_index * 6 + 3 + size_index;
                        let uniform_counts = std::array::from_fn::<_, 5, _>(|stratum| {
                            scan.sampled_counts[((null_index * slot_count + uniform_slot) * 5
                                + stratum)
                                * threshold_count
                                + threshold_index] as u64
                        });
                        let support_counts = std::array::from_fn::<_, 5, _>(|stratum| {
                            scan.sampled_counts[((null_index * slot_count + support_slot) * 5
                                + stratum)
                                * threshold_count
                                + threshold_index] as u64
                        });
                        let uniform_total = uniform_counts.iter().sum::<u64>();
                        let support_total = support_counts.iter().sum::<u64>();
                        let full_event = full_total > 0;
                        let uniform_match = full_event == (uniform_total > 0);
                        let support_match = full_event == (support_total > 0);

                        let uniform_primary = stratified_ht_estimate(
                            &sampling_plan.universe_stratum_counts,
                            &seed_plan.uniform_stratum_sample_sizes[size_index],
                            &uniform_counts,
                        )?;
                        let uniform_srs = unconditional_uniform_ht_estimate(
                            total_universe,
                            sampling_plan.actual_sizes[size_index],
                            uniform_total as usize,
                        )?;
                        let support_m = seed_plan.support_stratum_sample_sizes[size_index]
                            .iter()
                            .sum::<usize>();
                        let support_primary = stratified_ht_estimate(
                            &sampling_plan.universe_stratum_counts,
                            &seed_plan.support_stratum_sample_sizes[size_index],
                            &support_counts,
                        )?;
                        let requested = sampling_plan.requested_sizes[size_index];
                        let actual_uniform = sampling_plan.actual_sizes[size_index];
                        for (
                            estimator,
                            design,
                            sample_seed,
                            sample_hash,
                            actual,
                            counts,
                            sampled_total,
                            estimate,
                            srs,
                            matched,
                        ) in [
                            (
                                "uniform_poststratified",
                                "uniform",
                                seed_plan.pair_sample_seed,
                                seed_plan.uniform_sample_hashes[size_index].as_str(),
                                actual_uniform,
                                uniform_counts,
                                uniform_total,
                                uniform_primary,
                                uniform_srs,
                                uniform_match,
                            ),
                            (
                                "support_aware_stratified",
                                "support_aware_60pct_min_cell_5_19",
                                seed_plan.pair_sample_seed,
                                seed_plan.support_sample_hashes[size_index].as_str(),
                                support_m,
                                support_counts,
                                support_total,
                                support_primary,
                                AuditMetricValue {
                                    value: None,
                                    na_reason: Some("not_applicable"),
                                },
                                support_match,
                            ),
                        ] {
                            write!(
                                writer,
                                "{}\t{}\t{:.17}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                                null_index + 1,
                                threshold_index + 1,
                                threshold.score,
                                source_labels,
                                estimator,
                                design,
                                sample_seed,
                                sample_hash,
                                requested,
                                actual,
                                full_total
                            )
                            .map_err(|error| error.to_string())?;
                            for count in full_strata {
                                write!(writer, "\t{count}").map_err(|error| error.to_string())?;
                            }
                            for count in counts {
                                write!(writer, "\t{count}").map_err(|error| error.to_string())?;
                            }
                            writeln!(
                                writer,
                                "\t{sampled_total}\t{}\t{}\t{}",
                                metric_tsv(&estimate),
                                metric_tsv(&srs),
                                u8::from(matched)
                            )
                            .map_err(|error| error.to_string())?;
                        }
                    }
                }
            }
        }
        Ok(())
    })?;

    write_atomic_stream(top_pairs_path.as_str(), |writer| {
        writeln!(
            writer,
            "null_index\tfull_rank\tpair_i\tpair_j\tmax6_score\tbest_channel\tn00\tn01\tn10\tn11\tmin_cell\tmaf1\tmaf2\tmin_maf\tpairwise_r2\tcanonical_owner_scan_unit"
        )
        .map_err(|error| error.to_string())?;
        for (null_index, pairs) in scan.top_pairs.iter().enumerate() {
            for (rank, pair) in pairs.iter().enumerate() {
                let geometry = pair_binary_geometry(
                    logic_bits,
                    universe,
                    support_counts.as_slice(),
                    pair.pair.0,
                    pair.pair.1,
                )?;
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}\t{:.17}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{}",
                    null_index + 1,
                    rank + 1,
                    pair.pair.0,
                    pair.pair.1,
                    pair.score,
                    pair.best_channel,
                    geometry.counts[0],
                    geometry.counts[1],
                    geometry.counts[2],
                    geometry.counts[3],
                    geometry.min_cell,
                    geometry.maf1,
                    geometry.maf2,
                    geometry.maf1.min(geometry.maf2),
                    geometry.pairwise_r2,
                    geometry.owner_scan_unit,
                )
                .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    })?;

    let bootstrap_replicates = 1000usize;
    let summary_result = write_atomic_stream(summary_path.as_str(), |writer| {
        writeln!(writer, "threshold_index\tthreshold\tthreshold_sources\testimator\tdesign\trequested_M\tactual_M\tavailable_pair_seeds\tlambda_full\tlambda_full_ci95_low\tlambda_full_ci95_high\tlambda_hat\tlambda_hat_ci95_low\tlambda_hat_ci95_high\tbias\trelative_bias\trmse\tabs_error_median\tabs_error_p90\tabs_error_p95\tvariance_across_nulls_full\tvariance_across_nulls_estimate\tvariance_across_pair_seed_means\tzero_exceedance_classification_accuracy\tfwer_full\tfwer_full_ci95_low\tfwer_full_ci95_high\tfwer_sampled\tfwer_sampled_cluster_ci95_low\tfwer_sampled_cluster_ci95_high\ttheta_eff\ttheta_ci95_low\ttheta_ci95_high\ttheta_na_reason\tunion_bound_lambda_minus_fwer")
            .map_err(|error| error.to_string())?;

        for threshold_index in 0..threshold_count {
            let threshold = &grid.thresholds[threshold_index];
            let source_labels = threshold.sources.join(";");
            let full_counts = (0..null_count)
                .map(|null_index| {
                    (0..5)
                        .map(|stratum| {
                            scan.full_counts
                                [(null_index * 5 + stratum) * threshold_count + threshold_index]
                                as u64
                        })
                        .sum::<u64>()
                })
                .collect::<Vec<_>>();
            let lambda_full =
                full_counts.iter().map(|count| *count as f64).sum::<f64>() / null_count as f64;
            let full_fwer_successes = full_counts.iter().filter(|count| **count > 0).count();
            let fwer_full = wilson_fwer_interval(full_fwer_successes, null_count)?;
            if lambda_full + 1.0e-12 < fwer_full.estimate {
                return Err(format!(
                    "union-bound check failed at threshold {}: lambda_full={} < FWER_full={}",
                    threshold.score, lambda_full, fwer_full.estimate
                ));
            }
            let theta_bootstrap = bootstrap_theta_interval(
                full_counts.as_slice(),
                bootstrap_seed_for_threshold(bootstrap_seed, threshold_index, 99),
                bootstrap_replicates,
            )?;
            let theta = theta_bootstrap.estimate.value;
            let theta_ci = theta_bootstrap.interval;
            let lambda_full_bootstrap = bootstrap_mean_interval(
                &full_counts
                    .iter()
                    .map(|count| *count as f64)
                    .collect::<Vec<_>>(),
                bootstrap_seed_for_threshold(bootstrap_seed, threshold_index, 98),
                bootstrap_replicates,
            )?;

            for size_index in 0..3 {
                let actual_uniform = sampling_plan.actual_sizes[size_index];
                for estimator_index in 0..3 {
                    let (estimator_name, design_name) = match estimator_index {
                        0 => ("poststratified_ht", "uniform"),
                        1 => ("global_srswor_ht", "uniform_sensitivity"),
                        _ => ("stratum_expansion_ht", "support_aware_60pct_min_cell_5_19"),
                    };
                    let mut per_null_estimates = vec![0.0; null_count];
                    let mut per_null_sample_event_fraction = vec![0.0; null_count];
                    let mut per_null_zero_match_fraction = vec![0.0; null_count];
                    let mut per_seed_means = Vec::with_capacity(sampling_plan.seeds.len());
                    let mut per_seed_estimates: Vec<Vec<Option<f64>>> =
                        Vec::with_capacity(sampling_plan.seeds.len());
                    let mut available_seed_count = 0usize;
                    for (seed_index, seed_plan) in sampling_plan.seeds.iter().enumerate() {
                        let slot = seed_index * 6
                            + match estimator_index {
                                0 | 1 => size_index,
                                _ => 3 + size_index,
                            };
                        let mut seed_estimates = Vec::with_capacity(null_count);
                        let mut seed_events = Vec::with_capacity(null_count);
                        for null_index in 0..null_count {
                            let sample_counts = std::array::from_fn::<_, 5, _>(|stratum| {
                                scan.sampled_counts[((null_index * slot_count + slot) * 5
                                    + stratum)
                                    * threshold_count
                                    + threshold_index] as u64
                            });
                            let estimate = if estimator_index == 1 {
                                unconditional_uniform_ht_estimate(
                                    total_universe,
                                    actual_uniform,
                                    sample_counts.iter().sum::<u64>() as usize,
                                )?
                            } else if estimator_index == 0 {
                                stratified_ht_estimate(
                                    &sampling_plan.universe_stratum_counts,
                                    &seed_plan.uniform_stratum_sample_sizes[size_index],
                                    &sample_counts,
                                )?
                            } else {
                                stratified_ht_estimate(
                                    &sampling_plan.universe_stratum_counts,
                                    &seed_plan.support_stratum_sample_sizes[size_index],
                                    &sample_counts,
                                )?
                            };
                            seed_estimates.push(estimate.value);
                            seed_events.push(if sample_counts.iter().sum::<u64>() > 0 {
                                1.0
                            } else {
                                0.0
                            });
                        }
                        if seed_estimates.iter().all(Option::is_some) {
                            available_seed_count += 1;
                            let values = seed_estimates
                                .iter()
                                .map(|value| value.unwrap())
                                .collect::<Vec<_>>();
                            let mean = values.iter().sum::<f64>() / null_count as f64;
                            per_seed_means.push(mean);
                        }
                        per_seed_estimates.push(seed_estimates);
                        for null_index in 0..null_count {
                            per_null_sample_event_fraction[null_index] +=
                                seed_events[null_index] / sampling_plan.seeds.len() as f64;
                        }
                    }
                    for null_index in 0..null_count {
                        let valid_estimates = per_seed_estimates
                            .iter()
                            .filter_map(|values| values[null_index])
                            .collect::<Vec<_>>();
                        if !valid_estimates.is_empty() {
                            per_null_estimates[null_index] =
                                valid_estimates.iter().sum::<f64>() / valid_estimates.len() as f64;
                        }
                        let full_event = full_counts[null_index] > 0;
                        per_null_zero_match_fraction[null_index] = per_seed_estimates
                            .iter()
                            .enumerate()
                            .map(|(seed_index, values)| {
                                let slot = seed_index * 6
                                    + match estimator_index {
                                        0 | 1 => size_index,
                                        _ => 3 + size_index,
                                    };
                                let sample_event = (0..5).any(|stratum| {
                                    scan.sampled_counts[((null_index * slot_count + slot) * 5
                                        + stratum)
                                        * threshold_count
                                        + threshold_index]
                                        > 0
                                });
                                let _ = values;
                                (if sample_event == full_event { 1.0 } else { 0.0 })
                                    / sampling_plan.seeds.len() as f64
                            })
                            .sum();
                    }
                    let estimates_available = available_seed_count > 0;
                    let lambda_hat = if estimates_available {
                        per_null_estimates.iter().sum::<f64>() / null_count as f64
                    } else {
                        f64::NAN
                    };
                    let bias = estimates_available.then_some(lambda_hat - lambda_full);
                    let rel_bias = bias
                        .map(|value| relative_bias_metric(value, lambda_full))
                        .unwrap_or(AuditMetricValue {
                            value: None,
                            na_reason: Some("sampling_estimate_unavailable"),
                        });
                    let errors = per_null_estimates
                        .iter()
                        .zip(full_counts.iter())
                        .map(|(estimate, full)| estimate - *full as f64)
                        .collect::<Vec<_>>();
                    let rmse = if estimates_available {
                        (errors.iter().map(|error| error * error).sum::<f64>() / null_count as f64)
                            .sqrt()
                    } else {
                        f64::NAN
                    };
                    let mut abs_errors = errors.iter().map(|error| error.abs()).collect::<Vec<_>>();
                    abs_errors.sort_by(f64::total_cmp);
                    let lambda_hat_bootstrap = if estimates_available {
                        Some(bootstrap_mean_interval(
                            per_null_estimates.as_slice(),
                            bootstrap_seed_for_threshold(
                                bootstrap_seed,
                                threshold_index,
                                estimator_index * 10 + size_index,
                            ),
                            bootstrap_replicates,
                        )?)
                    } else {
                        None
                    };
                    let sample_fwer =
                        per_null_sample_event_fraction.iter().sum::<f64>() / null_count as f64;
                    let sample_fwer_ci = bootstrap_mean_interval(
                        per_null_sample_event_fraction.as_slice(),
                        bootstrap_seed_for_threshold(
                            bootstrap_seed,
                            threshold_index,
                            estimator_index * 10 + size_index + 100,
                        ),
                        bootstrap_replicates,
                    )?;
                    let zero_accuracy =
                        per_null_zero_match_fraction.iter().sum::<f64>() / null_count as f64;
                    let variance_null_full = sample_variance(
                        &full_counts
                            .iter()
                            .map(|value| *value as f64)
                            .collect::<Vec<_>>(),
                    );
                    let variance_null_estimate = estimates_available
                        .then(|| sample_variance(per_null_estimates.as_slice()))
                        .flatten();
                    let variance_seed_means = sample_variance(per_seed_means.as_slice());
                    let actual_sample = match estimator_index {
                        0 | 1 => actual_uniform,
                        _ => sampling_plan.seeds[0].support_stratum_sample_sizes[size_index]
                            .iter()
                            .sum::<usize>(),
                    };
                    let union_bound_gap = lambda_full - fwer_full.estimate;
                    let unavailable_reason = if estimates_available {
                        String::new()
                    } else {
                        "NA:no_pair_seed_has_all_nonempty_strata_sampled".to_string()
                    };
                    write!(
                        writer,
                        "{}\t{:.17}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17}\t{:.17}\t{:.17}\t",
                        threshold_index + 1,
                        threshold.score,
                        source_labels,
                        estimator_name,
                        design_name,
                        sampling_plan.requested_sizes[size_index],
                        actual_sample,
                        available_seed_count,
                        lambda_full,
                        lambda_full_bootstrap.lower,
                        lambda_full_bootstrap.upper
                    )
                    .map_err(|error| error.to_string())?;
                    if let Some(interval) = lambda_hat_bootstrap {
                        write!(
                            writer,
                            "{:.17}\t{:.17}\t{:.17}",
                            lambda_hat, interval.lower, interval.upper
                        )
                        .map_err(|error| error.to_string())?;
                    } else {
                        write!(writer, "{unavailable_reason}\tNA\tNA")
                            .map_err(|error| error.to_string())?;
                    }
                    let variance_full_tsv = variance_null_full
                        .map(|value| format!("{value:.17}"))
                        .unwrap_or_else(|| "NA:less_than_two_nulls".to_string());
                    let variance_estimate_tsv = variance_null_estimate
                        .map(|value| format!("{value:.17}"))
                        .unwrap_or_else(|| unavailable_reason.clone());
                    let variance_seed_tsv = variance_seed_means
                        .map(|value| format!("{value:.17}"))
                        .unwrap_or_else(|| "NA:less_than_two_available_pair_seeds".to_string());
                    let theta_value =
                        theta
                            .map(|value| format!("{value:.17}"))
                            .unwrap_or_else(|| {
                                format!(
                                    "NA:{}",
                                    theta_bootstrap.estimate.na_reason.unwrap_or("unspecified")
                                )
                            });
                    let theta_lower = theta_ci
                        .map(|interval| format!("{:.17}", interval.lower))
                        .unwrap_or_else(|| {
                            format!(
                                "NA:{}",
                                theta_bootstrap
                                    .na_reason
                                    .unwrap_or("theta_interval_unavailable")
                            )
                        });
                    let theta_upper = theta_ci
                        .map(|interval| format!("{:.17}", interval.upper))
                        .unwrap_or_else(|| {
                            format!(
                                "NA:{}",
                                theta_bootstrap
                                    .na_reason
                                    .unwrap_or("theta_interval_unavailable")
                            )
                        });
                    let tail_fields = [
                        bias.map(|value| format!("{value:.17}"))
                            .unwrap_or_else(|| unavailable_reason.clone()),
                        metric_tsv(&rel_bias),
                        if rmse.is_finite() {
                            format!("{rmse:.17}")
                        } else {
                            unavailable_reason.clone()
                        },
                        if estimates_available {
                            format!(
                                "{:.17}",
                                quantile_type7_sorted(abs_errors.as_slice(), 0.5).unwrap()
                            )
                        } else {
                            unavailable_reason.clone()
                        },
                        if estimates_available {
                            format!(
                                "{:.17}",
                                quantile_type7_sorted(abs_errors.as_slice(), 0.9).unwrap()
                            )
                        } else {
                            unavailable_reason.clone()
                        },
                        if estimates_available {
                            format!(
                                "{:.17}",
                                quantile_type7_sorted(abs_errors.as_slice(), 0.95).unwrap()
                            )
                        } else {
                            unavailable_reason.clone()
                        },
                        variance_full_tsv,
                        variance_estimate_tsv,
                        variance_seed_tsv,
                        format!("{zero_accuracy:.17}"),
                        format!("{:.17}", fwer_full.estimate),
                        format!("{:.17}", fwer_full.lower),
                        format!("{:.17}", fwer_full.upper),
                        format!("{sample_fwer:.17}"),
                        format!("{:.17}", sample_fwer_ci.lower),
                        format!("{:.17}", sample_fwer_ci.upper),
                        theta_value,
                        theta_lower,
                        theta_upper,
                        theta_bootstrap.na_reason.unwrap_or("").to_string(),
                        format!("{union_bound_gap:.17}"),
                    ];
                    writeln!(writer, "\t{}", tail_fields.join("\t"))
                        .map_err(|error| error.to_string())?;
                }
            }
        }
        Ok(())
    });
    summary_result?;

    let seed_manifest = sampling_plan
        .seeds
        .iter()
        .map(|seed_plan| {
            let uniform = (0..3)
                .map(|size_index| {
                    let stratum = std::array::from_fn::<_, 5, _>(|index| {
                        let population = sampling_plan.universe_stratum_counts[index];
                        let selected = seed_plan.uniform_stratum_sample_sizes[size_index][index];
                        json!({
                            "stratum": MIN_CELL_STRATA[index],
                            "N_h": population,
                            "m_h": selected,
                            "pi_h": (selected > 0).then(|| selected as f64 / population.max(1) as f64),
                            "pi_h_na_reason": if population == 0 {
                                Some("empty_population_stratum")
                            } else if selected == 0 {
                                Some("nonempty_stratum_has_zero_sample")
                            } else {
                                None
                            },
                        })
                    });
                    json!({
                        "requested_M": sampling_plan.requested_sizes[size_index],
                        "actual_M": sampling_plan.actual_sizes[size_index],
                        "sample_hash": seed_plan.uniform_sample_hashes[size_index],
                        "strata": stratum,
                    })
                })
                .collect::<Vec<_>>();
            let support_aware = (0..3)
                .map(|size_index| {
                    let stratum = std::array::from_fn::<_, 5, _>(|index| {
                        let population = sampling_plan.universe_stratum_counts[index];
                        let selected = seed_plan.support_stratum_sample_sizes[size_index][index];
                        json!({
                            "stratum": MIN_CELL_STRATA[index],
                            "N_h": population,
                            "m_h": selected,
                            "pi_h": (selected > 0).then(|| selected as f64 / population.max(1) as f64),
                            "pi_h_na_reason": if population == 0 {
                                Some("empty_population_stratum")
                            } else if selected == 0 {
                                Some("nonempty_stratum_has_zero_sample")
                            } else {
                                None
                            },
                        })
                    });
                    json!({
                        "requested_M": sampling_plan.requested_sizes[size_index],
                        "actual_M": seed_plan.support_stratum_sample_sizes[size_index].iter().sum::<usize>(),
                        "sample_hash": seed_plan.support_sample_hashes[size_index],
                        "strata": stratum,
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "pair_sample_seed": seed_plan.pair_sample_seed,
                "uniform": uniform,
                "support_aware_60pct_min_cell_5_19": support_aware,
            })
        })
        .collect::<Vec<_>>();
    let grid_manifest = json!({
        "fixed_score_grid": {
            "start": 0.0,
            "end": 50.0,
            "step": 0.25,
            "points": grid.fixed_grid_points,
        },
        "fixed_grid_full_maxima_above_50_count": grid.maxima_above_fixed_grid_count,
        "fixed_grid_full_maxima_above_50_max": grid.max_maxima_above_fixed_grid,
        "finite_full_maxima_count": grid.finite_maxima_count,
        "threshold_count_with_anchors": threshold_count,
        "anchors": "type-7 q80/q90/q95/q97.5/q99 from the same full-MaxT maxima; in-sample audit anchors, not independent validation thresholds",
        "exceedance_rule": "strict score > threshold",
    });
    let sampling_manifest = json!({
        "sampling_algorithm": "janusx_splitmix64_bottom_k_v1",
        "support_aware_allocation": "fixed 60% of total sample budget allocated to min_cell strata 5-9 and 10-19; remainder proportional across strata 0,1-4,>=20, with existing quota cap/reallocation rules",
        "support_strata": MIN_CELL_STRATA,
        "requested_sample_sizes": sampling_plan.requested_sizes,
        "actual_sample_sizes": sampling_plan.actual_sizes,
        "unique_universe_count": sampling_plan.unique_universe_count,
        "unique_universe_hash": sampling_plan.unique_universe_hash,
        "universe_stratum_counts": sampling_plan.universe_stratum_counts,
        "sampling_plan_seconds": sampling_plan.selection_seconds,
        "sampling_heap_preflight_bytes": sampling_plan.preflight_heap_bytes,
        "retained_cutoff_metadata_bytes": sampling_plan.retained_cutoff_bytes,
        "pair_sample_seeds": seed_manifest,
    });
    let estimator_manifest = json!({
        "uniform_primary_estimator": "sum_h (N_h/m_h) * c_rh(t); unavailable if any nonempty stratum has m_h=0",
        "uniform_sensitivity_estimator": "(N/M) * sum_h c_rh(t), global SRSWOR inclusion weight",
        "support_aware_estimator": "sum_h (N_h/m_h) * c_rh(t); unavailable if any nonempty stratum has m_h=0",
    });
    let paired_maxima_fingerprint = calibration_maxima.iter().fold(FNV_OFFSET, |hash, value| {
        fnv1a64_update(hash, &value.to_bits().to_le_bytes())
    });
    let scan_manifest = json!({
        "top_k": TAIL_AUDIT_TOP_K,
        "top_k_order": "score descending, canonical pair (i,j) ascending",
        "pair_count": scan.unique_pair_count,
        "per_null_exact_count_layout": "full_counts[null][stratum][threshold] and sampled_counts[null][seed-major sample slot][stratum][threshold]",
        "accumulator_preflight_bytes_peak_batch": scan.preflight_accumulator_bytes,
        "accumulator_reserved_bytes_peak_batch": scan.reserved_accumulator_bytes,
        "retained_exact_count_bytes": scan.full_counts.len() * std::mem::size_of::<u32>()
            + scan.sampled_counts.len() * std::mem::size_of::<u32>(),
        "preflight_output_bytes": scan.preflight_output_bytes,
        "reserved_output_bytes": scan.reserved_output_bytes,
        "work_metadata_bytes": scan.work_metadata_bytes,
        "preflight_total_audit_owned_bytes": scan.preflight_total_bytes,
        "scan_seconds": scan.elapsed_seconds,
    });
    let manifest = json!({
        "audit": "Max6 weighted exceedance-count reconstruction diagnostic",
        "null_count": null_count,
        "null_stream_seed": null_stream_seed,
        "calibration_null_draws_hash": calibration_null_draws_hash(calibration_nulls),
        "paired_full_maxima_hash": format!(
            "fnv1a64_full_maxima_f64_bits_v1:{paired_maxima_fingerprint:016x}"
        ),
        "null_source": "train_fit.null_residualized_y; paired calibration nulls used by full-MaxT",
        "maxima_source": "existing full-MaxT maxima for those null draws",
        "grid": grid_manifest,
        "sampling": sampling_manifest,
        "estimators": estimator_manifest,
        "scan": scan_manifest,
        "report_generation_seconds_before_manifest_write": report_started.elapsed().as_secs_f64(),
        "uncertainty": {
            "fwer": "Wilson 95% interval for exact full-universe null event; sampled event uncertainty is null-cluster bootstrap",
            "lambda": "paired null-resample percentile bootstrap 95% interval",
            "theta_eff": "paired null-resample percentile bootstrap 95% interval when all resamples remain in-domain",
            "bootstrap_replicates": bootstrap_replicates,
        },
        "interpretation": "in-sample sampled reconstruction audit only; no independent FWER validation, production threshold, EVT, or new FWER claim",
        "production_threshold_modified": false,
        "evt_fitted": false,
        "production_behavior_changed": false,
        "outputs": {
            "per_null_tsv": per_null_path,
            "summary_tsv": summary_path,
            "top_pairs_tsv": top_pairs_path,
        },
    });
    let bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed serializing weighted exceedance manifest: {error}"))?;
    write_atomic(manifest_path.as_str(), &bytes)?;
    Ok(WeightedExceedanceReportPaths {
        per_null_tsv: per_null_path,
        summary_tsv: summary_path,
        top_pairs_tsv: top_pairs_path,
        manifest_json: manifest_path,
    })
}

fn sample_maxima_summary(values: &[f64]) -> Value {
    let mut sorted = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return Value::Null;
    }
    json!({
        "n": sorted.len(),
        "min": sorted[0],
        "median_type7": quantile_type7_sorted(&sorted, 0.5),
        "p95_type7": quantile_type7_sorted(&sorted, 0.95),
        "p99_type7": quantile_type7_sorted(&sorted, 0.99),
        "max": sorted[sorted.len() - 1],
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64,
    })
}

fn paired_sample_summary(sampled: &[f64], full: &[f64]) -> Value {
    if sampled.len() != full.len() {
        return Value::Null;
    }
    let gaps = full
        .iter()
        .zip(sampled)
        .map(|(&full_score, &sampled_score)| full_score - sampled_score)
        .collect::<Vec<_>>();
    let exact_capture_count = sampled
        .iter()
        .zip(full)
        .filter(|(sampled_score, full_score)| sampled_score.to_bits() == full_score.to_bits())
        .count();
    let sampled_q95 = quantile_type7(sampled, 0.95);
    let sampled_q99 = quantile_type7(sampled, 0.99);
    let full_q95 = quantile_type7(full, 0.95);
    let full_q99 = quantile_type7(full, 0.99);
    json!({
        "exact_full_max_capture_count": exact_capture_count,
        "exact_full_max_capture_rate": exact_capture_count as f64 / sampled.len().max(1) as f64,
        "full_minus_sampled_paired_gap_type7": sample_maxima_summary(gaps.as_slice()),
        "sampled_q95_type7": sampled_q95,
        "full_q95_type7": full_q95,
        "full_minus_sampled_q95_type7": full_q95.zip(sampled_q95).map(|(f, s)| f - s),
        "sampled_q99_type7": sampled_q99,
        "full_q99_type7": full_q99,
        "full_minus_sampled_q99_type7": full_q99.zip(sampled_q99).map(|(f, s)| f - s),
    })
}

fn strata_value(counts: &PairStrataCounts) -> Value {
    json!({
        "canonical_owner_scan_unit": counts.owner_scan_unit,
        "min_maf": counts.min_maf,
        "pairwise_r2": counts.pairwise_r2,
        "min_cell": counts.min_cell,
    })
}

fn write_atomic(path: &str, bytes: &[u8]) -> Result<(), String> {
    let temp = format!("{path}.tmp.{}", std::process::id());
    let result = (|| {
        let mut file = BufWriter::new(File::create(&temp).map_err(|error| error.to_string())?);
        file.write_all(bytes).map_err(|error| error.to_string())?;
        file.flush().map_err(|error| error.to_string())?;
        file.get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
        fs::rename(&temp, path).map_err(|error| error.to_string())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn write_atomic_stream(
    path: &str,
    write_contents: impl FnOnce(&mut BufWriter<File>) -> Result<(), String>,
) -> Result<(), String> {
    let temp = format!("{path}.tmp.{}", std::process::id());
    let result = (|| {
        let file = File::create(&temp).map_err(|error| error.to_string())?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        write_contents(&mut writer)?;
        writer.flush().map_err(|error| error.to_string())?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
        fs::rename(&temp, path).map_err(|error| error.to_string())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn distribution_summary(values: &[f64]) -> Value {
    let mut sorted = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return Value::Null;
    }
    json!({
        "n": sorted.len(),
        "min": sorted[0],
        "median_type7": quantile_type7_sorted(&sorted, 0.5),
        "p90_type7": quantile_type7_sorted(&sorted, 0.9),
        "p99_type7": quantile_type7_sorted(&sorted, 0.99),
        "max": sorted[sorted.len() - 1],
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64,
    })
}

fn score_spacing_summary(top_pairs: &[Vec<TailPairScore>], prefix_size: usize) -> Value {
    let gaps = top_pairs
        .iter()
        .filter(|pairs| pairs.len() >= prefix_size)
        .flat_map(|pairs| {
            (0..prefix_size.saturating_sub(1)).map(|rank| pairs[rank].score - pairs[rank + 1].score)
        })
        .collect::<Vec<_>>();
    distribution_summary(gaps.as_slice())
}

pub(crate) fn ranked_geometry_summary(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    sampled: &TailAuditSampledMaxima,
    full: &TailAuditFullScan,
    null_seed: u64,
    out_prefix: &str,
) -> Result<TailAuditReportPaths, String> {
    let null_count = full.top_pairs.len();
    if null_count == 0
        || full.sampled_full_ranks.len() != null_count
        || sampled
            .winners
            .iter()
            .any(|target| target.len() != null_count)
    {
        return Err("tail-audit report inputs have inconsistent null dimensions".to_string());
    }
    if full.unique_pair_count != sampled.sample.sample.unique_universe_count {
        return Err(format!(
            "tail-audit sampled universe count {} differs from full scan count {}",
            sampled.sample.sample.unique_universe_count, full.unique_pair_count
        ));
    }
    let sample = &sampled.sample.sample;
    let pair_sample_seed = sampled.sample.pair_sample_seed;
    let actual_sizes = sampled.sample.actual_sizes;
    let sample_hashes = actual_sizes.map(|size| sample_set_hash(&sample.priority_pairs[..size]));
    let top_pairs_path = format!("{out_prefix}.max6_tail_top_pairs.tsv");
    let sampled_ranks_path = format!("{out_prefix}.max6_tail_sampled_max_ranks.tsv");
    let summary_path = format!("{out_prefix}.max6_tail_audit_summary.json");
    let supports = support_counts_by_active_slot(logic_bits, universe);

    write_atomic_stream(&top_pairs_path, |writer| {
        writeln!(writer, "null_index\tfull_rank\tpair_i\tpair_j\tscore\tbest_channel\tn00\tn01\tn10\tn11\tmin_cell\tmin_maf\tpairwise_r2\tcanonical_owner_scan_unit\tsnp1\tchrom1\tpos1\tsnp2\tchrom2\tpos2")
            .map_err(|error| error.to_string())?;
        for (null_index, ranked) in full.top_pairs.iter().enumerate() {
            if ranked.len() > TAIL_AUDIT_TOP_K {
                return Err(format!(
                    "tail-audit null {null_index} retained {} pairs, above K={TAIL_AUDIT_TOP_K}",
                    ranked.len()
                ));
            }
            let mut seen = std::collections::HashSet::with_capacity(ranked.len());
            for (rank_index, score) in ranked.iter().enumerate() {
                if score.pair.0 >= score.pair.1 || !seen.insert(score.pair) {
                    return Err(format!(
                        "tail-audit top list has invalid/duplicate canonical pair {:?} at null {null_index}",
                        score.pair
                    ));
                }
                if rank_index > 0 && full_rank_cmp(&ranked[rank_index - 1], score).is_gt() {
                    return Err(format!(
                        "tail-audit null {null_index} is not in total rank order"
                    ));
                }
                let geometry = pair_binary_geometry(
                    logic_bits,
                    universe,
                    supports.as_slice(),
                    score.pair.0,
                    score.pair.1,
                )?;
                let site1 = logic_bits.sites.get(score.pair.0).ok_or_else(|| {
                    format!("tail-audit pair row {} is out of range", score.pair.0)
                })?;
                let site2 = logic_bits.sites.get(score.pair.1).ok_or_else(|| {
                    format!("tail-audit pair row {} is out of range", score.pair.1)
                })?;
                writeln!(
                    writer,
                    "{null_index}\t{}\t{}\t{}\t{:.17}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10}\t{:.10}\tscan_unit_{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    rank_index + 1,
                    score.pair.0,
                    score.pair.1,
                    score.score,
                    score.best_channel.as_str(),
                    geometry.counts[0],
                    geometry.counts[1],
                    geometry.counts[2],
                    geometry.counts[3],
                    geometry.min_cell,
                    geometry.maf1.min(geometry.maf2),
                    geometry.pairwise_r2,
                    geometry.owner_scan_unit,
                    site1.snp(),
                    site1.chrom(),
                    site1.pos,
                    site2.snp(),
                    site2.chrom(),
                    site2.pos,
                )
                .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    })?;

    write_atomic_stream(&sampled_ranks_path, |writer| {
        writeln!(writer, "null_index\tnull_seed_root\ttarget_M\tactual_M\tpair_sample_seed\tpair_sample_hash\tsampled_max_score\tpair_i\tpair_j\tbest_channel\tfull_rank")
            .map_err(|error| error.to_string())?;
        for null_index in 0..null_count {
            for target_index in 0..TAIL_AUDIT_SAMPLE_SIZES.len() {
                if let Some(winner) = sampled.winners[target_index][null_index] {
                    let full_rank = full.sampled_full_ranks[null_index][target_index]
                        .ok_or_else(|| "sampled maximum is missing its full rank".to_string())?;
                    writeln!(
                        writer,
                        "{null_index}\t{null_seed}\t{}\t{}\t{pair_sample_seed}\t{}\t{:.17}\t{}\t{}\t{}\t{full_rank}",
                        TAIL_AUDIT_SAMPLE_SIZES[target_index],
                        actual_sizes[target_index],
                        sample_hashes[target_index],
                        winner.score,
                        winner.pair.0,
                        winner.pair.1,
                        winner.best_channel.as_str(),
                    )
                    .map_err(|error| error.to_string())?;
                } else {
                    writeln!(
                        writer,
                        "{null_index}\t{null_seed}\t{}\t{}\t{pair_sample_seed}\t{}\tNA\tNA\tNA\tNA\tNA",
                        TAIL_AUDIT_SAMPLE_SIZES[target_index],
                        actual_sizes[target_index],
                        sample_hashes[target_index],
                    )
                    .map_err(|error| error.to_string())?;
                }
            }
        }
        Ok(())
    })?;

    let prefix_sizes = [1usize, 10, 100, 1000];
    let mut geometry = serde_json::Map::new();
    for prefix_size in prefix_sizes {
        let selected = full
            .top_pairs
            .iter()
            .flat_map(|pairs| pairs.iter().take(prefix_size))
            .collect::<Vec<_>>();
        let mut min_cells = Vec::with_capacity(selected.len());
        let mut min_mafs = Vec::with_capacity(selected.len());
        let mut r2_values = Vec::with_capacity(selected.len());
        let mut channel_counts = BTreeMap::<String, usize>::new();
        let mut min_cell_strata = BTreeMap::<String, usize>::new();
        for score in selected {
            let pair_geometry = pair_binary_geometry(
                logic_bits,
                universe,
                supports.as_slice(),
                score.pair.0,
                score.pair.1,
            )?;
            min_cells.push(pair_geometry.min_cell as f64);
            min_mafs.push(pair_geometry.maf1.min(pair_geometry.maf2));
            r2_values.push(pair_geometry.pairwise_r2);
            *channel_counts
                .entry(score.best_channel.as_str().to_string())
                .or_default() += 1;
            *min_cell_strata
                .entry(min_cell_stratum(pair_geometry.min_cell).to_string())
                .or_default() += 1;
        }
        geometry.insert(
            format!("top_{prefix_size}"),
            json!({
                "n_pair_nulls": min_cells.len(),
                "min_cell": distribution_summary(min_cells.as_slice()),
                "min_cell_strata": min_cell_strata,
                "min_maf": distribution_summary(min_mafs.as_slice()),
                "pairwise_r2": distribution_summary(r2_values.as_slice()),
                "best_channel_counts": channel_counts,
            }),
        );
    }

    let mut sample_rank_summaries = Vec::with_capacity(TAIL_AUDIT_SAMPLE_SIZES.len());
    for target_index in 0..TAIL_AUDIT_SAMPLE_SIZES.len() {
        let ranks = full
            .sampled_full_ranks
            .iter()
            .filter_map(|rank| rank[target_index].map(|value| value as f64))
            .collect::<Vec<_>>();
        let mut rank_cdf = Vec::new();
        for cutoff in [1usize, 10, 100, 1000, TAIL_AUDIT_TOP_K] {
            if cutoff > full.unique_pair_count {
                continue;
            }
            let expected = finite_population_sample_max_cdf(
                full.unique_pair_count,
                actual_sizes[target_index],
                cutoff,
            )?;
            let observed = ranks.iter().filter(|&&rank| rank <= cutoff as f64).count() as f64
                / ranks.len().max(1) as f64;
            rank_cdf.push(json!({
                "rank_cutoff": cutoff,
                "observed_fraction": observed,
                "uniform_without_replacement_expected_cdf": expected,
            }));
        }
        sample_rank_summaries.push(json!({
            "target_M": TAIL_AUDIT_SAMPLE_SIZES[target_index],
            "actual_M": actual_sizes[target_index],
            "pair_sample_hash": sample_hashes[target_index],
            "full_rank_distribution": distribution_summary(ranks.as_slice()),
            "rank_cdf_diagnostic_only": rank_cdf,
        }));
    }
    let score_spacing = json!({
        "adjacent_gaps_within_top10": score_spacing_summary(full.top_pairs.as_slice(), 10),
        "adjacent_gaps_within_top100": score_spacing_summary(full.top_pairs.as_slice(), 100),
        "adjacent_gaps_within_top1000": score_spacing_summary(full.top_pairs.as_slice(), 1000),
    });
    let summary = json!({
        "analysis": "opt-in exact full-MaxT tail audit",
        "top_k": TAIL_AUDIT_TOP_K,
        "null_draw_count": null_count,
        "null_seed_root": null_seed,
        "pair_sample_seed": pair_sample_seed,
        "sampling_algorithm": PAIR_SAMPLE_ALGORITHM,
        "sampled_pair_count_requested": TAIL_AUDIT_SAMPLE_SIZES,
        "sampled_pair_count_actual": actual_sizes,
        "sample_set_hashes": sample_hashes,
        "unique_pair_universe_count": full.unique_pair_count,
        "unique_pair_universe_hash": sample.unique_universe_hash,
        "sample_storage_capacity_bytes": sample.storage_capacity_bytes,
        "pair_sample_selection_seconds": sampled.sample.selection_seconds,
        "scan_unit_label": "canonical owner scan unit (operational ownership unit; not asserted to be an LD block)",
        "ld_block_pair_class": null,
        "top_rank_geometry": geometry,
        "score_spacing": score_spacing,
        "sampled_max_full_rank_distribution": sample_rank_summaries,
        "timing_seconds": {
            "sample_preprocessing_total": sampled.preprocessing_seconds,
            "sampled_scoring_by_nested_increment": sampled.scoring_seconds_by_target,
            "full_exact_scan": full.elapsed_seconds,
        },
        "production_threshold_modified": false,
        "gev_fitted": false,
        "m_specific_correction_deployed": false,
        "interpretation": "geometry and uniform-sampling rank diagnostics only; not a production threshold calibration or FWER claim",
        "outputs": {
            "top_pairs_tsv": top_pairs_path,
            "sampled_max_ranks_tsv": sampled_ranks_path,
        }
    });
    let summary_bytes = serde_json::to_vec_pretty(&summary)
        .map_err(|error| format!("failed to serialize Max6 tail-audit summary: {error}"))?;
    write_atomic(&summary_path, &summary_bytes)?;

    Ok(TailAuditReportPaths {
        top_pairs_tsv: top_pairs_path,
        sampled_ranks_tsv: sampled_ranks_path,
        summary_json: summary_path,
        support_aware_maxima_tsv: None,
        support_aware_strata_tsv: None,
        support_aware_summary_json: None,
    })
}

fn pair_priority_for(seed: u64, pair: (usize, usize)) -> PairPriority {
    PairPriority {
        priority: stable_pair_priority(seed, pair.0, pair.1),
        pair_id: ((pair.0 as u64) << 32) | pair.1 as u64,
    }
}

pub(crate) fn sample_contains_pair(
    pair: (usize, usize),
    seed: u64,
    cutoff: Option<PairPriority>,
) -> bool {
    cutoff.is_some_and(|cutoff| pair_priority_for(seed, pair) <= cutoff)
}

pub(crate) fn support_aware_geometry_summary(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    uniform: &TailAuditSampledMaxima,
    support_aware: &[SupportAwareTailAuditMaxima],
    full: &TailAuditFullScan,
    null_seed_root: u64,
    out_prefix: &str,
) -> Result<SupportAwareTailReportPaths, String> {
    let null_count = full.top_pairs.len();
    if null_count == 0 || full.sampled_full_ranks.len() != null_count {
        return Err(
            "support-aware audit report has inconsistent full-scan null dimensions".to_string(),
        );
    }
    let uniform_sample = &uniform.sample.sample;
    if uniform.winners.len() < SUPPORT_AWARE_SAMPLE_SIZES.len()
        || support_aware.is_empty()
        || support_aware.iter().any(|design| {
            design.winners.len() != SUPPORT_AWARE_SAMPLE_SIZES.len()
                || design
                    .winners
                    .iter()
                    .any(|target| target.len() != null_count)
                || design.universe_count != full.unique_pair_count
                || design.universe_hash != uniform_sample.unique_universe_hash
        })
    {
        return Err(
            "support-aware audit report has invalid allocation/null dimensions".to_string(),
        );
    }
    let full_rank_offsets = 4usize;
    let expected_rank_width =
        full_rank_offsets.saturating_add(support_aware.len() * SUPPORT_AWARE_SAMPLE_SIZES.len());
    if full
        .sampled_full_ranks
        .iter()
        .any(|ranks| ranks.len() < expected_rank_width)
    {
        return Err("full scan omitted one or more support-aware sampled maxima ranks".to_string());
    }

    let seed = uniform.sample.pair_sample_seed;
    let actual_sizes = [
        uniform.sample.actual_sizes[0],
        uniform.sample.actual_sizes[1],
        uniform.sample.actual_sizes[2],
    ];
    let uniform_prefix_hashes =
        actual_sizes.map(|size| sample_set_hash(&uniform_sample.priority_pairs[..size]));
    let supports = support_counts_by_active_slot(logic_bits, universe);
    let mut uniform_stratum_counts = [[0usize; 5]; 3];
    for (pair_index, priority_pair) in uniform_sample
        .priority_pairs
        .iter()
        .take(actual_sizes[2])
        .enumerate()
    {
        let pair = decode_pair_id(priority_pair.pair_id);
        let stratum =
            pair_min_cell_stratum(logic_bits, universe, supports.as_slice(), pair.0, pair.1)?;
        for size_index in 0..3 {
            if pair_index < actual_sizes[size_index] {
                uniform_stratum_counts[size_index][stratum] += 1;
            }
        }
    }

    let mut full_tail_stratum_counts = [[0usize; 5]; 4];
    for ranked in &full.top_pairs {
        for (rank, score) in ranked.iter().take(1000).enumerate() {
            let stratum = pair_min_cell_stratum(
                logic_bits,
                universe,
                supports.as_slice(),
                score.pair.0,
                score.pair.1,
            )?;
            for (prefix_index, cutoff) in [1usize, 10, 100, 1000].iter().enumerate() {
                if rank < *cutoff {
                    full_tail_stratum_counts[prefix_index][stratum] += 1;
                }
            }
        }
    }

    let maxima_path = format!("{out_prefix}.max6_tail_support_aware_maxima.tsv");
    let strata_path = format!("{out_prefix}.max6_tail_support_aware_strata.tsv");
    let summary_path = format!("{out_prefix}.max6_tail_support_aware_summary.json");
    let design_names = std::iter::once("uniform".to_string())
        .chain(
            support_aware
                .iter()
                .map(|design| format!("support_aware_{}pct_min_cell_5_19", design.share_percent)),
        )
        .collect::<Vec<_>>();

    write_atomic_stream(&maxima_path, |writer| {
        writeln!(writer, "null_index\tnull_seed_root\tdesign\ttarget_M\tactual_M\tfull_max_score\tsampled_max_score\tfull_minus_sampled_gap\tfull_max_score_captured\tcanonical_full_top1_pair_captured\tsampled_max_full_rank\tfull_top10_pairs_captured\tfull_top10_capture_fraction\tfull_top10_complete\tfull_top100_pairs_captured\tfull_top100_capture_fraction\tfull_top100_complete\tfull_top1000_pairs_captured\tfull_top1000_capture_fraction\tfull_top1000_complete\tpair_sample_hash\tsampled_max_pair_i\tsampled_max_pair_j\tbest_channel")
            .map_err(|error| error.to_string())?;
        for null_index in 0..null_count {
            let full_max = full.top_pairs[null_index]
                .first()
                .ok_or_else(|| format!("full tail list for null {null_index} is empty"))?;
            let mut designs = Vec::with_capacity(support_aware.len() + 1);
            for size_index in 0..3 {
                let cutoff = actual_sizes[size_index]
                    .checked_sub(1)
                    .and_then(|index| uniform_sample.priority_pairs.get(index).copied());
                designs.push((
                    0usize,
                    size_index,
                    uniform.winners[size_index][null_index],
                    cutoff,
                    None,
                    uniform_prefix_hashes[size_index].clone(),
                    actual_sizes[size_index],
                ));
            }
            for (share_index, design) in support_aware.iter().enumerate() {
                for size_index in 0..3 {
                    designs.push((
                        share_index + 1,
                        size_index,
                        design.winners[size_index][null_index],
                        None,
                        Some(design),
                        design.sample_hashes[size_index].clone(),
                        design.actual_sizes[size_index],
                    ));
                }
            }
            for (
                design_index,
                size_index,
                winner,
                uniform_cutoff,
                support,
                sample_hash,
                actual_m,
            ) in designs
            {
                let winner = winner.ok_or_else(|| {
                    format!("sampled maximum missing for null {null_index}, design {}, M-index {size_index}", design_names[design_index])
                })?;
                let captured = |pair: (usize, usize)| -> Result<bool, String> {
                    if let Some(support) = support {
                        let stratum = pair_min_cell_stratum(
                            logic_bits,
                            universe,
                            supports.as_slice(),
                            pair.0,
                            pair.1,
                        )?;
                        Ok(sample_contains_pair(
                            pair,
                            seed,
                            support.cutoffs_by_size[size_index][stratum],
                        ))
                    } else {
                        Ok(sample_contains_pair(pair, seed, uniform_cutoff))
                    }
                };
                let mut top_hits = [0usize; 3];
                let top_pairs = &full.top_pairs[null_index];
                for (rank, score) in top_pairs.iter().take(1000).enumerate() {
                    if captured(score.pair)? {
                        for (k_index, k) in [10usize, 100, 1000].iter().enumerate() {
                            if rank < *k {
                                top_hits[k_index] += 1;
                            }
                        }
                    }
                }
                let ranks_index = if design_index == 0 {
                    size_index
                } else {
                    full_rank_offsets + (design_index - 1) * 3 + size_index
                };
                let full_rank = full.sampled_full_ranks[null_index][ranks_index]
                    .ok_or_else(|| "sampled maximum is missing its exact full rank".to_string())?;
                let full_top1_hit = captured(full_max.pair)?;
                writeln!(
                    writer,
                    "{null_index}\t{null_seed_root}\t{}\t{}\t{actual_m}\t{:.17}\t{:.17}\t{:.17}\t{}\t{}\t{full_rank}\t{}\t{:.10}\t{}\t{}\t{:.10}\t{}\t{}\t{:.10}\t{}\t{}\t{}\t{}\t{}",
                    design_names[design_index],
                    SUPPORT_AWARE_SAMPLE_SIZES[size_index],
                    full_max.score,
                    winner.score,
                    full_max.score - winner.score,
                    (winner.score.to_bits() == full_max.score.to_bits()) as u8,
                    full_top1_hit as u8,
                    top_hits[0],
                    top_hits[0] as f64 / top_pairs.len().min(10).max(1) as f64,
                    (top_hits[0] == top_pairs.len().min(10)) as u8,
                    top_hits[1],
                    top_hits[1] as f64 / top_pairs.len().min(100).max(1) as f64,
                    (top_hits[1] == top_pairs.len().min(100)) as u8,
                    top_hits[2],
                    top_hits[2] as f64 / top_pairs.len().min(1000).max(1) as f64,
                    (top_hits[2] == top_pairs.len().min(1000)) as u8,
                    sample_hash,
                    winner.pair.0,
                    winner.pair.1,
                    winner.best_channel.as_str(),
                )
                .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    })?;

    write_atomic_stream(&strata_path, |writer| {
        writeln!(writer, "design\ttarget_M\tmin_cell_stratum\ttrue_universe_pairs\tselected_pairs\tsampling_fraction\tsample_composition_fraction\tfull_top1_count\tfull_top10_count\tfull_top100_count\tfull_top1000_count\tfull_top1000_composition_fraction")
            .map_err(|error| error.to_string())?;
        for size_index in 0..3 {
            for (stratum, label) in MIN_CELL_STRATA.iter().enumerate() {
                let uniform_count = uniform_stratum_counts[size_index][stratum];
                let uniform_universe = support_aware
                    .first()
                    .map(|design| design.universe_stratum_counts[stratum])
                    .unwrap_or(0);
                writeln!(
                    writer,
                    "uniform\t{}\t{}\t{}\t{}\t{:.12}\t{:.12}\t{}\t{}\t{}\t{}\t{:.12}",
                    SUPPORT_AWARE_SAMPLE_SIZES[size_index],
                    label,
                    uniform_universe,
                    uniform_count,
                    uniform_count as f64 / uniform_universe.max(1) as f64,
                    uniform_count as f64 / actual_sizes[size_index].max(1) as f64,
                    full_tail_stratum_counts[0][stratum],
                    full_tail_stratum_counts[1][stratum],
                    full_tail_stratum_counts[2][stratum],
                    full_tail_stratum_counts[3][stratum],
                    full_tail_stratum_counts[3][stratum] as f64
                        / (null_count * full.unique_pair_count.min(1000)).max(1) as f64,
                )
                .map_err(|error| error.to_string())?;
            }
            for design in support_aware {
                for (stratum, label) in MIN_CELL_STRATA.iter().enumerate() {
                    let selected = design.quotas_by_size[size_index][stratum];
                    let true_count = design.universe_stratum_counts[stratum];
                    writeln!(
                        writer,
                        "support_aware_{}pct_min_cell_5_19\t{}\t{}\t{}\t{}\t{:.12}\t{:.12}\t{}\t{}\t{}\t{}\t{:.12}",
                        design.share_percent,
                        SUPPORT_AWARE_SAMPLE_SIZES[size_index],
                        label,
                        true_count,
                        selected,
                        selected as f64 / true_count.max(1) as f64,
                        selected as f64 / design.actual_sizes[size_index].max(1) as f64,
                        full_tail_stratum_counts[0][stratum],
                        full_tail_stratum_counts[1][stratum],
                        full_tail_stratum_counts[2][stratum],
                        full_tail_stratum_counts[3][stratum],
                        full_tail_stratum_counts[3][stratum] as f64
                            / (null_count * full.unique_pair_count.min(1000)).max(1) as f64,
                    )
                    .map_err(|error| error.to_string())?;
                }
            }
        }
        Ok(())
    })?;

    let mut design_summaries = Vec::with_capacity(support_aware.len() + 1);
    for design_index in 0..=support_aware.len() {
        let name = &design_names[design_index];
        let mut size_summaries = Vec::with_capacity(3);
        for size_index in 0..3 {
            let winner_rows = if design_index == 0 {
                &uniform.winners[size_index]
            } else {
                &support_aware[design_index - 1].winners[size_index]
            };
            let full_scores = full
                .top_pairs
                .iter()
                .map(|pairs| pairs.first().map(|pair| pair.score).unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let sample_scores = winner_rows
                .iter()
                .map(|winner| winner.map(|pair| pair.score).unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let gaps = full_scores
                .iter()
                .zip(sample_scores.iter())
                .map(|(&full_score, &sample_score)| full_score - sample_score)
                .collect::<Vec<_>>();
            let score_capture = sample_scores
                .iter()
                .zip(full_scores.iter())
                .filter(|(sample_score, full_score)| sample_score.to_bits() == full_score.to_bits())
                .count();
            let rank_index = if design_index == 0 {
                size_index
            } else {
                full_rank_offsets + (design_index - 1) * 3 + size_index
            };
            let full_ranks = full
                .sampled_full_ranks
                .iter()
                .filter_map(|ranks| ranks[rank_index].map(|rank| rank as f64))
                .collect::<Vec<_>>();
            let top_coverage = |k: usize| -> Result<Value, String> {
                let mut pair_hits = 0usize;
                let mut complete_nulls = 0usize;
                for null_index in 0..null_count {
                    let (uniform_cutoff, support) = if design_index == 0 {
                        let cutoff = actual_sizes[size_index]
                            .checked_sub(1)
                            .and_then(|index| uniform_sample.priority_pairs.get(index).copied());
                        (cutoff, None)
                    } else {
                        (None, Some(&support_aware[design_index - 1]))
                    };
                    let mut captured = 0usize;
                    for pair_score in full.top_pairs[null_index].iter().take(k) {
                        let present = if let Some(support) = support {
                            let stratum = pair_min_cell_stratum(
                                logic_bits,
                                universe,
                                supports.as_slice(),
                                pair_score.pair.0,
                                pair_score.pair.1,
                            )?;
                            sample_contains_pair(
                                pair_score.pair,
                                seed,
                                support.cutoffs_by_size[size_index][stratum],
                            )
                        } else {
                            sample_contains_pair(pair_score.pair, seed, uniform_cutoff)
                        };
                        captured += present as usize;
                    }
                    let available = full.top_pairs[null_index].len().min(k);
                    pair_hits += captured;
                    complete_nulls += (captured == available) as usize;
                }
                Ok(json!({
                    "k": k,
                    "captured_top_pair_fraction": pair_hits as f64
                        / (null_count * k.min(TAIL_AUDIT_TOP_K)).max(1) as f64,
                    "complete_top_k_capture_null_rate": complete_nulls as f64
                        / null_count.max(1) as f64,
                }))
            };
            size_summaries.push(json!({
                "requested_M": SUPPORT_AWARE_SAMPLE_SIZES[size_index],
                "actual_M": if design_index == 0 {
                    actual_sizes[size_index]
                } else {
                    support_aware[design_index - 1].actual_sizes[size_index]
                },
                "paired_full_minus_sampled_max": distribution_summary(gaps.as_slice()),
                "exact_full_max_score_capture_rate": score_capture as f64 / null_count.max(1) as f64,
                "sampled_max_full_rank": distribution_summary(full_ranks.as_slice()),
                "coverage": [top_coverage(10)?, top_coverage(100)?, top_coverage(1000)?],
                "sampling_fraction": (if design_index == 0 {
                    actual_sizes[size_index]
                } else {
                    support_aware[design_index - 1].actual_sizes[size_index]
                }) as f64 / full.unique_pair_count.max(1) as f64,
                "timing_seconds": if design_index == 0 {
                    json!({
                        "pair_selection": uniform.sample.selection_seconds,
                        "null_preprocessing": uniform.preprocessing_seconds,
                        "scoring_increment": uniform.scoring_seconds_by_target[size_index],
                    })
                } else {
                    let support = &support_aware[design_index - 1];
                    json!({
                        "pair_selection_shared_across_support_allocations": support.selection_seconds,
                        "null_preprocessing": support.preprocessing_seconds,
                        "scoring_increment": support.scoring_seconds_by_size[size_index],
                    })
                },
            }));
        }
        let memory = if design_index == 0 {
            json!({
                "sample_heap_or_vector_capacity_bytes": uniform_sample.storage_capacity_bytes,
                "sample_materialized_capacity_bytes": uniform_sample.storage_capacity_bytes,
                "approx_concurrent_sample_storage_bytes": uniform_sample.storage_capacity_bytes,
            })
        } else {
            let support = &support_aware[design_index - 1];
            json!({
                "uniform_reference_sample_capacity_bytes_retained": uniform_sample.storage_capacity_bytes,
                "per_stratum_bottom_k_capacity_bytes": support.selection_storage_bytes,
                "materialized_nested_sample_capacity_bytes": support.materialized_sample_storage_bytes,
                "approx_concurrent_sample_storage_bytes": uniform_sample.storage_capacity_bytes
                    .saturating_add(support.selection_storage_bytes)
                    .saturating_add(support.materialized_sample_storage_bytes),
            })
        };
        design_summaries.push(json!({
            "design": name,
            "allocation_share_min_cell_5_19_percent": if design_index == 0 { Value::Null } else { json!(support_aware[design_index - 1].share_percent) },
            "sample_memory_components": memory,
            "nested_sample_hashes": if design_index == 0 {
                actual_sizes.map(|size| sample_set_hash(&uniform_sample.priority_pairs[..size])).to_vec()
            } else {
                support_aware[design_index - 1].sample_hashes.to_vec()
            },
            "stratum_quotas_by_M": if design_index == 0 {
                Value::Null
            } else {
                json!(support_aware[design_index - 1].quotas_by_size)
            },
            "results_by_M": size_summaries,
        }));
    }

    let summary = json!({
        "analysis": "diagnostic-only support-aware sampled Max6 tail comparison",
        "null_draw_count": null_count,
        "null_seed_root": null_seed_root,
        "pair_sample_seed": seed,
        "unique_pair_universe_count": full.unique_pair_count,
        "unique_pair_universe_hash": uniform_sample.unique_universe_hash,
        "support_strata": MIN_CELL_STRATA,
        "oversampled_strata": ["5-9", "10-19"],
        "sampling_algorithm": "janusx_splitmix64_bottom_k_with_proportional_stratum_quotas_v1",
        "target_sample_sizes": SUPPORT_AWARE_SAMPLE_SIZES,
        "support_allocation_shares_percent": support_aware.iter().map(|design| design.share_percent).collect::<Vec<_>>(),
        "full_scan_seconds_shared_across_designs": full.elapsed_seconds,
        "full_reference_top_k": TAIL_AUDIT_TOP_K,
        "full_tail_stratum_counts_by_top_k_across_nulls": {
            "top1": full_tail_stratum_counts[0],
            "top10": full_tail_stratum_counts[1],
            "top100": full_tail_stratum_counts[2],
            "top1000": full_tail_stratum_counts[3],
        },
        "designs": design_summaries,
        "production_threshold_use": false,
        "production_q95_q99_modified": false,
        "fwer_claim_changed": false,
        "gev_fitted": false,
        "interpretation": "Oversampled maxima are diagnostic only and are not production q95/q99 or FWER calibration.",
        "outputs": {"maxima_tsv": maxima_path.clone(), "strata_tsv": strata_path.clone()},
    });
    let summary_bytes = serde_json::to_vec_pretty(&summary)
        .map_err(|error| format!("failed to serialize support-aware Max6 tail summary: {error}"))?;
    write_atomic(&summary_path, &summary_bytes)?;
    Ok(SupportAwareTailReportPaths {
        maxima_tsv: maxima_path,
        strata_tsv: strata_path,
        summary_json: summary_path,
    })
}

fn select_unique_eligible_pairs(
    universe: &Max6ScreenPreparedUniverse,
    pair_sample_seed: u64,
    mut notify_progress: impl FnMut(usize, usize, usize) -> Result<(), String>,
) -> Result<(BottomKSample, f64), String> {
    let started = Instant::now();
    let mut sampler = BottomKPairSampler::new(
        *REQUESTED_SAMPLE_SIZES.last().expect("four targets"),
        universe.window_pair_count,
    );
    let total_units = universe.units.len();
    notify_progress(0, total_units, 0)?;
    for (unit_index, unit) in universe.units.iter().enumerate() {
        for first_local in 0..unit.selected_global_rows.len().saturating_sub(1) {
            for second_local in first_local + 1..unit.selected_global_rows.len() {
                let Some(owned) = super::max6_screen_owned_pair(
                    universe,
                    unit_index,
                    unit,
                    first_local,
                    second_local,
                )?
                else {
                    continue;
                };
                sampler.consider(
                    owned.first_global_row.min(owned.second_global_row),
                    owned.first_global_row.max(owned.second_global_row),
                    pair_sample_seed,
                )?;
            }
        }
        if (unit_index + 1) % 64 == 0 || unit_index + 1 == total_units {
            notify_progress(unit_index + 1, total_units, sampler.seen)?;
        }
    }
    let selection_seconds = started.elapsed().as_secs_f64();
    Ok((sampler.finish()?, selection_seconds))
}

struct PotSeedSample {
    pair_sample_seed: u64,
    sample: BottomKSample,
    support_strata: Vec<u8>,
}

struct PotPairSamplingPlan {
    samples: Vec<PotSeedSample>,
    unique_universe_count: usize,
    unique_universe_hash: String,
    universe_stratum_counts: [usize; 5],
    selection_seconds: f64,
    preflight_heap_bytes: usize,
}

fn prepare_pot_pair_samples(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_seed: u64,
    pair_sample_seeds: [u64; 5],
    mut notify_progress: impl FnMut(&str, usize, usize, usize) -> Result<(), String>,
) -> Result<PotPairSamplingPlan, String> {
    if pair_sample_seeds.contains(&null_seed) {
        return Err("POT pair-sample seeds must be separate from the null stream".to_string());
    }
    // Window occurrences are an upper bound on the canonical unique-pair
    // universe. Capping the heap reservation here keeps small-region smoke
    // runs from reserving the full five × 10M production sample budget.
    let target_size = POT_SAMPLE_SIZES[2].min(universe.window_pair_count);
    if target_size == 0 {
        return Err("POT audit found an empty eligible-pair universe".to_string());
    }
    let estimated_heap_bytes = pair_sample_seeds
        .len()
        .checked_mul(target_size)
        .and_then(|count| count.checked_mul(size_of::<PairPriority>()))
        .ok_or_else(|| "POT pair-sample heap estimate overflow".to_string())?;
    notify_progress("max6_screen_pot_pair_preflight", 1, 1, estimated_heap_bytes)?;

    let total_units = universe.units.len();
    let supports = support_counts_by_active_slot(logic_bits, universe);
    let mut universe_stratum_counts = [0usize; 5];
    let mut samplers = Vec::with_capacity(pair_sample_seeds.len());
    for &seed in &pair_sample_seeds {
        let mut sampler = BottomKPairSampler::new(target_size, 0);
        sampler.try_reserve(target_size)?;
        samplers.push((seed, sampler));
    }
    notify_progress("max6_screen_pot_pair_selection", 0, total_units, 0)?;
    let selection_started = Instant::now();
    let unique_universe_count = visit_unique_eligible_pairs(
        universe,
        |first, second| {
            let stratum =
                pair_min_cell_stratum(logic_bits, universe, supports.as_slice(), first, second)?;
            universe_stratum_counts[stratum] = universe_stratum_counts[stratum]
                .checked_add(1)
                .ok_or_else(|| "POT universe support count overflow".to_string())?;
            for (seed, sampler) in samplers.iter_mut() {
                sampler.consider(first, second, *seed)?;
            }
            Ok(())
        },
        |done, total| {
            if done % 64 == 0 || done == total {
                notify_progress("max6_screen_pot_pair_selection", done, total, 0)
            } else {
                Ok(())
            }
        },
    )?;
    let selection_seconds = selection_started.elapsed().as_secs_f64();
    if universe_stratum_counts.iter().sum::<usize>() != unique_universe_count {
        return Err("POT universe support strata do not reconcile to pair count".to_string());
    }
    let mut samples = Vec::with_capacity(samplers.len());
    let mut unique_universe_hash = None;
    for (pair_sample_seed, sampler) in samplers {
        let sample = sampler.finish()?;
        if sample.unique_universe_count != unique_universe_count {
            return Err("POT pair sampler universe count mismatch".to_string());
        }
        if unique_universe_hash
            .as_ref()
            .is_some_and(|hash| hash != &sample.unique_universe_hash)
        {
            return Err("POT pair sampler universe fingerprints disagree".to_string());
        }
        unique_universe_hash.get_or_insert_with(|| sample.unique_universe_hash.clone());
        let actual_sample_size = sample.priority_pairs.len();
        if actual_sample_size != target_size.min(unique_universe_count) {
            return Err(format!(
                "POT bottom-k sample size {actual_sample_size} != expected {}",
                target_size.min(unique_universe_count)
            ));
        }
        let mut support_strata = Vec::new();
        support_strata
            .try_reserve_exact(actual_sample_size)
            .map_err(|error| format!("unable to reserve POT sample support metadata: {error}"))?;
        for entry in sample.priority_pairs.iter() {
            let (first, second) = decode_pair_id(entry.pair_id);
            support_strata.push(pair_min_cell_stratum(
                logic_bits,
                universe,
                supports.as_slice(),
                first,
                second,
            )? as u8);
        }
        samples.push(PotSeedSample {
            pair_sample_seed,
            sample,
            support_strata,
        });
    }
    Ok(PotPairSamplingPlan {
        samples,
        unique_universe_count,
        unique_universe_hash: unique_universe_hash
            .ok_or_else(|| "POT sample plan has no pair seeds".to_string())?,
        universe_stratum_counts,
        selection_seconds,
        preflight_heap_bytes: estimated_heap_bytes,
    })
}

fn write_f64_le_values(writer: &mut impl Write, values: &[f64]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(size_of::<f64>()));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    writer
        .write_all(bytes.as_slice())
        .map_err(|error| error.to_string())
}

fn write_pot_sample_pair_metadata(path: &str, sample: &PotSeedSample) -> Result<(), String> {
    write_atomic_stream(path, |writer| {
        for (entry, &stratum) in sample
            .sample
            .priority_pairs
            .iter()
            .zip(sample.support_strata.iter())
        {
            writer
                .write_all(&entry.pair_id.to_le_bytes())
                .map_err(|error| error.to_string())?;
            writer
                .write_all(&[stratum])
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    })
}

fn score_pot_pilot_seed(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    sample: &PotSeedSample,
    null_draws: &[Vec<f64>],
    pilot_path: &str,
    pool: Option<&ThreadPool>,
    mut notify_progress: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<([Option<f64>; 6], f64), String> {
    let pilot_size = POT_PILOT_SAMPLE_SIZE.min(sample.sample.priority_pairs.len());
    if pilot_size == 0 || null_draws.is_empty() {
        return Err("POT pilot requires nonempty sample and calibration nulls".to_string());
    }
    let mut pilot_values: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
    let pilot_strata_counts =
        sample.support_strata[..pilot_size]
            .iter()
            .fold([0usize; 5], |mut counts, &stratum| {
                counts[stratum as usize] += 1;
                counts
            });
    for (stratum, values) in pilot_values.iter_mut().enumerate() {
        let expected = pilot_strata_counts[stratum]
            .checked_mul(null_draws.len())
            .ok_or_else(|| "POT pilot score count overflow".to_string())?;
        values
            .try_reserve_exact(expected)
            .map_err(|error| format!("unable to reserve POT pilot scores: {error}"))?;
    }

    let started = Instant::now();
    write_atomic_stream(pilot_path, |writer| {
        let null_batch_size = super::MAX6_SCREEN_NULL_BATCH_SIZE;
        let batch_count = null_draws.len().div_ceil(null_batch_size);
        const PAIR_BLOCK: usize = 4096;
        for (batch_number, null_batch) in null_draws.chunks(null_batch_size).enumerate() {
            let lookups = null_batch
                .iter()
                .map(|residual| PackedYSumLookup::build(residual, logic_bits.n_samples))
                .collect::<Result<Vec<_>, _>>()?;
            let margins = null_batch
                .iter()
                .zip(lookups.iter())
                .map(|(residual, lookup)| {
                    OneCellMarginTable::build_for_rows_with_pool(
                        logic_bits.bits(),
                        logic_bits.row_words,
                        universe.active_rows.as_slice(),
                        residual,
                        logic_bits.n_samples,
                        lookup,
                        pool,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let draws = null_batch
                .iter()
                .zip(lookups.iter())
                .zip(margins.iter())
                .map(
                    |((residual, lookup), margins)| logic_residual_screen::Max6ScreenNullDraw {
                        residual,
                        lookup,
                        margins,
                    },
                )
                .collect::<Vec<_>>();
            let mut score_rows_by_draw = vec![vec![0.0; pilot_size]; draws.len()];
            for (pair_start, entries) in sample.sample.priority_pairs[..pilot_size]
                .chunks(PAIR_BLOCK)
                .enumerate()
            {
                let offset = pair_start * PAIR_BLOCK;
                let block_scores =
                    score_pot_pair_block(logic_bits, universe, entries, &draws, pool)?;
                for (local_pair, &stratum) in sample.support_strata[offset..offset + entries.len()]
                    .iter()
                    .enumerate()
                {
                    let row_start = local_pair * draws.len();
                    for draw_index in 0..draws.len() {
                        let score = block_scores[row_start + draw_index];
                        score_rows_by_draw[draw_index][offset + local_pair] = score;
                        pilot_values[stratum as usize].push(score);
                    }
                }
            }
            for scores in score_rows_by_draw.iter() {
                write_f64_le_values(writer, scores.as_slice())?;
            }
            notify_progress(batch_number + 1, batch_count)?;
        }
        Ok(())
    })?;

    let mut pooled = Vec::new();
    pooled
        .try_reserve_exact(pilot_size.saturating_mul(null_draws.len()))
        .map_err(|error| format!("unable to reserve pooled POT pilot scores: {error}"))?;
    for values in pilot_values.iter() {
        pooled.extend_from_slice(values);
    }
    pooled.sort_by(f64::total_cmp);
    let pooled_floor = quantile_type7_sorted(&pooled, POT_RETENTION_QUANTILE);
    let mut stratum_floors = [None; 5];
    for (stratum, values) in pilot_values.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        stratum_floors[stratum] = quantile_type7_sorted(values, POT_RETENTION_QUANTILE);
    }
    let floors = [
        pooled_floor,
        stratum_floors[0],
        stratum_floors[1],
        stratum_floors[2],
        stratum_floors[3],
        stratum_floors[4],
    ];
    if floors
        .iter()
        .any(|value| value.is_some_and(|x| !x.is_finite()))
    {
        return Err("POT pilot retention quantile is nonfinite".to_string());
    }
    let _ = pilot_strata_counts;
    Ok((floors, started.elapsed().as_secs_f64()))
}

fn score_pot_larger_sample_seed(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    sample: &PotSeedSample,
    null_draws: &[Vec<f64>],
    retention_floors: [Option<f64>; 6],
    tail_path: &str,
    pool: Option<&ThreadPool>,
    mut notify_progress: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<(u64, f64), String> {
    let sample_size = sample.sample.priority_pairs.len();
    let pilot_size = POT_PILOT_SAMPLE_SIZE.min(sample_size);
    if sample_size <= pilot_size {
        write_atomic_stream(tail_path, |_| Ok(()))?;
        return Ok((0, 0.0));
    }
    let pooled_floor = retention_floors[0]
        .ok_or_else(|| "POT pooled sample-only retention floor is unavailable".to_string())?;
    let started = Instant::now();
    let mut record_count = 0u64;
    write_atomic_stream(tail_path, |writer| {
        const PAIR_BLOCK: usize = 4096;
        let null_batch_size = super::MAX6_SCREEN_NULL_BATCH_SIZE;
        let batch_count = null_draws.len().div_ceil(null_batch_size);
        for (batch_number, null_batch) in null_draws.chunks(null_batch_size).enumerate() {
            let null_start = batch_number * null_batch_size;
            let lookups = null_batch
                .iter()
                .map(|residual| PackedYSumLookup::build(residual, logic_bits.n_samples))
                .collect::<Result<Vec<_>, _>>()?;
            let margins = null_batch
                .iter()
                .zip(lookups.iter())
                .map(|(residual, lookup)| {
                    OneCellMarginTable::build_for_rows_with_pool(
                        logic_bits.bits(),
                        logic_bits.row_words,
                        universe.active_rows.as_slice(),
                        residual,
                        logic_bits.n_samples,
                        lookup,
                        pool,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let draws = null_batch
                .iter()
                .zip(lookups.iter())
                .zip(margins.iter())
                .map(
                    |((residual, lookup), margins)| logic_residual_screen::Max6ScreenNullDraw {
                        residual,
                        lookup,
                        margins,
                    },
                )
                .collect::<Vec<_>>();
            for (block_index, entries) in sample.sample.priority_pairs[pilot_size..]
                .chunks(PAIR_BLOCK)
                .enumerate()
            {
                let rank_start = pilot_size + block_index * PAIR_BLOCK;
                let block_scores =
                    score_pot_pair_block(logic_bits, universe, entries, &draws, pool)?;
                for (local_pair, &stratum) in sample.support_strata
                    [rank_start..rank_start + entries.len()]
                    .iter()
                    .enumerate()
                {
                    let row_start = local_pair * draws.len();
                    let stratum_floor = retention_floors[stratum as usize + 1];
                    let keep_floor =
                        stratum_floor.map_or(pooled_floor, |floor| floor.min(pooled_floor));
                    for draw_index in 0..draws.len() {
                        let score = block_scores[row_start + draw_index];
                        if score > keep_floor {
                            let rank = u32::try_from(rank_start + local_pair)
                                .map_err(|_| "POT sample rank exceeds u32".to_string())?;
                            let null_index = u16::try_from(null_start + draw_index)
                                .map_err(|_| "POT null index exceeds u16".to_string())?;
                            writer
                                .write_all(&null_index.to_le_bytes())
                                .and_then(|_| writer.write_all(&rank.to_le_bytes()))
                                .and_then(|_| writer.write_all(&score.to_le_bytes()))
                                .map_err(|error| error.to_string())?;
                            record_count = record_count
                                .checked_add(1)
                                .ok_or_else(|| "POT tail record count overflow".to_string())?;
                        }
                    }
                }
            }
            notify_progress(batch_number + 1, batch_count)?;
        }
        Ok(())
    })?;
    Ok((record_count, started.elapsed().as_secs_f64()))
}

pub(crate) fn run_sampled_screen_pot_audit(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_draws: &[Vec<f64>],
    calibration: &Max6ScreenCalibration,
    null_seed: u64,
    out_prefix: &str,
    pool: Option<&ThreadPool>,
    mut notify_progress: impl FnMut(&str, usize, usize, usize) -> Result<(), String>,
) -> Result<PotAuditReportPaths, String> {
    if null_draws.len() != calibration.maxima.len() || null_draws.is_empty() {
        return Err(format!(
            "POT audit calibration null/maxima mismatch: nulls={} maxima={}",
            null_draws.len(),
            calibration.maxima.len()
        ));
    }
    if !matches!(null_draws.len(), 100 | 500) {
        return Err(format!(
            "POT audit supports 100 or 500 calibration null draws, got {}",
            null_draws.len()
        ));
    }
    let pot_manifest_path = format!("{out_prefix}.max6_grmmax6.pot_audit_manifest.json");
    let sample_manifest_path = format!("{out_prefix}.max6_grmmax6.pot_sample_scores_manifest.json");
    let plan = prepare_pot_pair_samples(
        logic_bits,
        universe,
        null_seed,
        POT_PAIR_SAMPLE_SEEDS,
        |stage, done, total, bytes| notify_progress(stage, done, total, bytes),
    )?;
    if plan.unique_universe_count != calibration.pair_universe_size {
        return Err(format!(
            "POT sampled pair universe {} != exact calibration universe {}",
            plan.unique_universe_count, calibration.pair_universe_size
        ));
    }
    let sample_size = POT_SAMPLE_SIZES[2].min(plan.unique_universe_count);
    let pilot_size = POT_PILOT_SAMPLE_SIZE.min(sample_size);
    let n_seed = plan.samples.len();
    let pilot_bytes = n_seed
        .checked_mul(pilot_size)
        .and_then(|value| value.checked_mul(null_draws.len()))
        .and_then(|value| value.checked_mul(size_of::<f64>()))
        .ok_or_else(|| "POT pilot artifact size overflow".to_string())?;
    // At most 20% of scores can exceed the union of pooled and within-stratum
    // q90 retention floors, by the union bound. Use 21% to cover type-7 ties.
    let tail_score_count = n_seed
        .checked_mul(sample_size.saturating_sub(pilot_size))
        .and_then(|value| value.checked_mul(null_draws.len()))
        .ok_or_else(|| "POT tail artifact count overflow".to_string())?;
    let estimated_tail_storage_bytes = (tail_score_count as u64)
        .saturating_mul(21)
        .saturating_mul(14)
        .div_ceil(100);
    let expected_peak_memory_bytes = plan
        .preflight_heap_bytes
        .saturating_add(
            pilot_size
                .saturating_mul(null_draws.len())
                .saturating_mul(size_of::<f64>())
                .saturating_mul(2),
        )
        .saturating_add(pilot_size.saturating_mul(size_of::<f64>()));

    let mut sample_seed_manifest = Vec::with_capacity(plan.samples.len());
    let mut tail_record_total = 0u64;
    for sample in plan.samples.iter() {
        let seed_hex = format!("{:016x}", sample.pair_sample_seed);
        let sample_pairs_path =
            format!("{out_prefix}.max6_grmmax6.pot_sample_pairs_seed_{seed_hex}.bin");
        let pilot_scores_path =
            format!("{out_prefix}.max6_grmmax6.pot_pilot_scores_seed_{seed_hex}.f64");
        let tail_scores_path =
            format!("{out_prefix}.max6_grmmax6.pot_tail_scores_seed_{seed_hex}.bin");
        write_pot_sample_pair_metadata(sample_pairs_path.as_str(), sample)?;
        let (retention_floors, pilot_seconds) = score_pot_pilot_seed(
            logic_bits,
            universe,
            sample,
            null_draws,
            pilot_scores_path.as_str(),
            pool,
            |done, total| {
                let completed = sample_seed_manifest.len() * null_draws.len() + done;
                let total_work = n_seed * null_draws.len();
                notify_progress(
                    "max6_screen_pot_pilot_scoring",
                    completed,
                    total_work,
                    total,
                )
            },
        )?;
        let (tail_record_count, tail_seconds) = score_pot_larger_sample_seed(
            logic_bits,
            universe,
            sample,
            null_draws,
            retention_floors,
            tail_scores_path.as_str(),
            pool,
            |done, total| {
                let completed = sample_seed_manifest.len() * null_draws.len() + done;
                let total_work = n_seed * null_draws.len();
                notify_progress("max6_screen_pot_tail_scoring", completed, total_work, total)
            },
        )?;
        tail_record_total = tail_record_total
            .checked_add(tail_record_count)
            .ok_or_else(|| "POT tail record aggregate overflow".to_string())?;
        let actual_sizes = POT_SAMPLE_SIZES.map(|target| target.min(plan.unique_universe_count));
        let sample_hashes =
            actual_sizes.map(|size| sample_set_hash(&sample.sample.priority_pairs[..size]));
        let sample_strata_counts = actual_sizes.map(|size| {
            let mut counts = [0usize; 5];
            for stratum in &sample.support_strata[..size] {
                counts[*stratum as usize] += 1;
            }
            counts
        });
        sample_seed_manifest.push(json!({
            "pair_sample_seed": sample.pair_sample_seed,
            "sample_algorithm": PAIR_SAMPLE_ALGORITHM,
            "sampled_unique_pair_count": sample.sample.priority_pairs.len(),
            "sample_hash_by_M": sample_hashes,
            "sample_stratum_counts_by_M": sample_strata_counts,
            "pilot_score_layout": "little-endian f64; null-major then sample-priority-rank; shape [n_null, actual_100k]",
            "tail_score_layout": "records: null_index:u16le, sample_rank:u32le, exact_score:f64le; tail only for ranks >= pilot size and score above sample-only q90 retention floor",
            "retention_quantile": POT_RETENTION_QUANTILE,
            "retention_floors_pooled_and_strata": retention_floors,
            "tail_record_count": tail_record_count,
            "pilot_scores_path": pilot_scores_path,
            "sample_pairs_path": sample_pairs_path,
            "tail_scores_path": tail_scores_path,
            "pilot_scoring_seconds": pilot_seconds,
            "larger_sample_scoring_seconds": tail_seconds,
        }));
    }
    if tail_record_total > tail_score_count as u64 {
        return Err("POT tail record count exceeds the scored sample universe".to_string());
    }
    let sample_manifest = json!({
        "artifact": "exact sampled Max6 pair-score data for marginal POT audit",
        "format_version": "max6_pot_sample_scores_v1",
        "score_definition": "exact production Max6 one-cell/margin screen score; unscorable pair is encoded as zero, a non-exceedance for positive u",
        "scorer": "logic_residual_screen::update_max6_one_cell_pair_batch",
        "null_count": null_draws.len(),
        "null_stream_seed": null_seed,
        "calibration_null_draws_hash": calibration_null_draws_hash(null_draws),
        "pair_sample_seeds": sample_seed_manifest,
        "requested_sample_sizes": POT_SAMPLE_SIZES,
        "actual_sample_sizes": POT_SAMPLE_SIZES.map(|size| size.min(plan.unique_universe_count)),
        "unique_universe_count": plan.unique_universe_count,
        "unique_universe_hash": plan.unique_universe_hash,
        "universe_stratum_counts": plan.universe_stratum_counts,
        "sample_selection_seconds": plan.selection_seconds,
        "preflight_heap_bytes": plan.preflight_heap_bytes,
        "pilot_score_bytes": pilot_bytes,
        "preflight_estimated_tail_storage_bytes": estimated_tail_storage_bytes,
        "observed_tail_record_count": tail_record_total,
        "observed_tail_storage_bytes": tail_record_total.saturating_mul(14),
        "expected_peak_working_memory_bytes": expected_peak_memory_bytes,
        "candidate_u_policy": "python postprocessor derives 97.5%, 99%, 99.5%, 99.9% from only each seed's M=100k pilot matrix over calibration nulls; pooled and per existing support stratum",
        "full_reference_used_for_sample_selection_or_scoring": false,
        "production_threshold_modified": false,
    });
    let sample_manifest_bytes = serde_json::to_vec_pretty(&sample_manifest)
        .map_err(|error| format!("failed serializing POT sample manifest: {error}"))?;
    write_atomic(sample_manifest_path.as_str(), &sample_manifest_bytes)?;
    let manifest = json!({
        "audit": "Max6 marginal-tail POT diagnostic; opt-in only",
        "format_version": "max6_pot_audit_v1",
        "null_count": null_draws.len(),
        "null_stream_seed": null_seed,
        "calibration_null_draws_hash": calibration_null_draws_hash(null_draws),
        "pair_universe_count": plan.unique_universe_count,
        "pair_universe_hash": plan.unique_universe_hash,
        "universe_stratum_counts": plan.universe_stratum_counts,
        "marker_order_hash": marker_order_hash(logic_bits),
        "sample_order_hash": sample_order_hash(logic_bits),
        "sample_sizes_requested": POT_SAMPLE_SIZES,
        "sample_sizes_actual": POT_SAMPLE_SIZES.map(|size| size.min(plan.unique_universe_count)),
        "pair_sample_seeds": POT_PAIR_SAMPLE_SEEDS,
        "candidate_u_quantiles": POT_QUANTILES,
        "candidate_u_scope": "sample-only; per pair-sample seed, from M=100k calibration-null scores, separately pooled and by min_cell stratum",
        "minimum_exceedances": {"pooled": 1000, "support_stratum": 300},
        "support_strata": ["0", "1-4", "5-9", "10-19", ">=20"],
        "calibration_maxima_hash": format!("fnv1a64_full_maxima_f64_bits_v1:{:016x}", calibration.maxima.iter().fold(FNV_OFFSET, |hash, value| fnv1a64_update(hash, &value.to_bits().to_le_bytes()))),
        "sample_scores_manifest_json": sample_manifest_path,
        "outputs": {"sample_scores_manifest_json": sample_manifest_path},
        "preflight": {
            "sample_selection_heap_bytes": plan.preflight_heap_bytes,
            "pilot_score_bytes": pilot_bytes,
            "estimated_tail_storage_bytes": estimated_tail_storage_bytes,
            "expected_peak_working_memory_bytes": expected_peak_memory_bytes,
            "observed_tail_record_count": tail_record_total,
        },
        "production_threshold_modified": false,
        "production_behavior_changed": false,
        "evt_fitted": false,
    });
    let bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed serializing POT audit manifest: {error}"))?;
    write_atomic(pot_manifest_path.as_str(), &bytes)?;
    Ok(PotAuditReportPaths {
        manifest_json: pot_manifest_path,
        sample_scores_manifest_json: sample_manifest_path,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_sampled_screen_null_diagnostic_one_seed(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_draws: &[Vec<f64>],
    null_seed: u64,
    pair_sample_seed: u64,
    full_calibration: &Max6ScreenCalibration,
    maf_cutpoints: &[f64],
    r2_cutpoints: &[f64],
    out_prefix: &str,
    pool: Option<&ThreadPool>,
    notify_progress: &mut impl FnMut(&str, usize, usize, usize) -> Result<(), String>,
) -> Result<SampledScreenSeedOutput, String> {
    let total_started = Instant::now();
    if null_draws.len() != 100 {
        return Err(format!(
            "sampled Max6 scaling diagnostic requires 100 existing null draws; got {}",
            null_draws.len()
        ));
    }
    if pair_sample_seed == null_seed {
        return Err("pair_sample_seed must be separate from the null stream root seed".to_string());
    }
    if full_calibration.maxima.len() != null_draws.len() {
        return Err("sampled/full Max6 maxima must use the same null draws".to_string());
    }
    let (sample, selection_seconds) =
        select_unique_eligible_pairs(universe, pair_sample_seed, |done, total, count| {
            notify_progress("max6_screen_sample_selection", done, total, count)
        })?;
    if sample.unique_universe_count != full_calibration.pair_universe_size {
        return Err(format!(
            "sampled selector unique universe count {} differs from full-maxT count {}",
            sample.unique_universe_count, full_calibration.pair_universe_size
        ));
    }
    let actual_sample_sizes = actual_sample_sizes(sample.unique_universe_count);
    if actual_sample_sizes
        .iter()
        .any(|&size| size > sample.priority_pairs.len())
    {
        return Err("bottom-k sample is shorter than an actual requested prefix".to_string());
    }

    let strata_started = Instant::now();
    let strata_snapshots = sample_strata_snapshots(
        logic_bits,
        universe,
        &sample,
        actual_sample_sizes,
        maf_cutpoints,
        r2_cutpoints,
    )?;
    let strata_seconds = strata_started.elapsed().as_secs_f64();

    let (sampled_maxima, null_preprocessing_seconds, scoring_seconds_by_target) =
        score_nested_sample_sizes(
            logic_bits,
            universe,
            &sample,
            actual_sample_sizes,
            null_draws,
            pool,
            notify_progress,
        )?;
    if sampled_maxima
        .iter()
        .any(|maxima| maxima.len() != null_draws.len())
    {
        return Err("sampled Max6 maxima output has a null replicate count mismatch".to_string());
    }
    let mut monotonicity_violations = 0usize;
    for null_index in 0..null_draws.len() {
        for target_index in 1..REQUESTED_SAMPLE_SIZES.len() {
            if sampled_maxima[target_index - 1][null_index]
                > sampled_maxima[target_index][null_index]
            {
                monotonicity_violations += 1;
            }
        }
    }
    if monotonicity_violations != 0 {
        return Err(format!(
            "sampled Max6 nested maxima monotonicity failed for {monotonicity_violations} null draws"
        ));
    }

    let maxima_path = format!("{out_prefix}.max6_grmmax6.sampled_screen_null_maxima.tsv");
    let manifest_path = format!("{out_prefix}.max6_grmmax6.sampled_screen_manifest.json");
    let strata_path = format!("{out_prefix}.max6_grmmax6.sampled_screen_strata.tsv");
    let hashes = actual_sample_sizes.map(|size| {
        let entries = &sample.priority_pairs[..size];
        sample_set_hash(entries)
    });
    let mut maxima_tsv = String::from(
        "null_index\tnull_seed\ttarget_M\tactual_M\tmax_score\tpair_sample_seed\tpair_sample_hash\tunique_universe_count\tfull_max_score\tsampled_minus_full\tfull_minus_sampled\texact_full_max_capture\n",
    );
    for null_index in 0..null_draws.len() {
        for target_index in 0..REQUESTED_SAMPLE_SIZES.len() {
            let sampled = sampled_maxima[target_index][null_index];
            let full = full_calibration.maxima[null_index];
            maxima_tsv.push_str(&format!(
                "{null_index}\t{null_seed}\t{}\t{}\t{sampled:.17}\t{pair_sample_seed}\t{}\t{}\t{full:.17}\t{:.17}\t{:.17}\t{}\n",
                REQUESTED_SAMPLE_SIZES[target_index],
                actual_sample_sizes[target_index],
                hashes[target_index],
                sample.unique_universe_count,
                sampled - full,
                full - sampled,
                sampled.to_bits() == full.to_bits(),
            ));
        }
    }
    write_atomic(&maxima_path, maxima_tsv.as_bytes())?;

    let mut strata_tsv = String::from("target_M\tactual_M\tdimension\tstratum\tpair_count\n");
    for index in 0..strata_snapshots.len() {
        let counts = &strata_snapshots[index];
        for (dimension, map) in [
            ("canonical_owner_scan_unit", &counts.owner_scan_unit),
            ("min_maf", &counts.min_maf),
            ("pairwise_r2", &counts.pairwise_r2),
            ("min_cell", &counts.min_cell),
        ] {
            for (stratum, count) in map {
                strata_tsv.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\n",
                    REQUESTED_SAMPLE_SIZES[index],
                    actual_sample_sizes[index],
                    dimension,
                    stratum,
                    count
                ));
            }
        }
    }
    write_atomic(&strata_path, strata_tsv.as_bytes())?;

    let mut sample_sizes_json = Vec::with_capacity(4);
    let mut target_summaries = Vec::with_capacity(4);
    for index in 0..REQUESTED_SAMPLE_SIZES.len() {
        let paired_comparison = paired_sample_summary(
            sampled_maxima[index].as_slice(),
            full_calibration.maxima.as_slice(),
        );
        let paired_gaps = full_calibration
            .maxima
            .iter()
            .zip(sampled_maxima[index].iter())
            .map(|(&full, &sampled)| full - sampled)
            .collect::<Vec<_>>();
        let exact_capture_rate = paired_comparison["exact_full_max_capture_rate"]
            .as_f64()
            .ok_or_else(|| "paired exact-max capture rate is missing".to_string())?;
        let paired_gap_median = quantile_type7(paired_gaps.as_slice(), 0.5)
            .ok_or_else(|| "paired gap median is undefined".to_string())?;
        let paired_gap_p95 = quantile_type7(paired_gaps.as_slice(), 0.95)
            .ok_or_else(|| "paired gap p95 is undefined".to_string())?;
        let paired_gap_p99 = quantile_type7(paired_gaps.as_slice(), 0.99)
            .ok_or_else(|| "paired gap p99 is undefined".to_string())?;
        let full_q95 = quantile_type7(full_calibration.maxima.as_slice(), 0.95)
            .ok_or_else(|| "full maximum q95 is undefined".to_string())?;
        let full_q99 = quantile_type7(full_calibration.maxima.as_slice(), 0.99)
            .ok_or_else(|| "full maximum q99 is undefined".to_string())?;
        let sampled_q95 = quantile_type7(sampled_maxima[index].as_slice(), 0.95)
            .ok_or_else(|| "sampled maximum q95 is undefined".to_string())?;
        let sampled_q99 = quantile_type7(sampled_maxima[index].as_slice(), 0.99)
            .ok_or_else(|| "sampled maximum q99 is undefined".to_string())?;
        target_summaries.push(SeedTargetSummary {
            target_m: REQUESTED_SAMPLE_SIZES[index],
            actual_m: actual_sample_sizes[index],
            exact_capture_rate,
            paired_gap_median,
            paired_gap_p95,
            paired_gap_p99,
            full_minus_sampled_q95: full_q95 - sampled_q95,
            full_minus_sampled_q99: full_q99 - sampled_q99,
        });
        sample_sizes_json.push(json!({
            "target_M": REQUESTED_SAMPLE_SIZES[index],
            "actual_M": actual_sample_sizes[index],
            "sample_set_hash": hashes[index],
            "maxima_summary_type7": sample_maxima_summary(sampled_maxima[index].as_slice()),
            "paired_full_comparison_type7": paired_comparison,
            "cumulative_exact_score_seconds": scoring_seconds_by_target[index],
            "strata_counts": strata_value(&strata_snapshots[index]),
        }));
    }
    let maf_default = maf_cutpoints == [0.0, 0.05, 0.10, 0.20, 0.30, 0.50];
    let r2_default = r2_cutpoints == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let universe_metadata = json!({
        "n_samples": logic_bits.n_samples,
        "global_marker_row_count": logic_bits.sites.len(),
        "active_marker_row_count": universe.active_rows.len(),
        "prepared_scan_unit_count": universe.units.len(),
        "window_pair_occurrence_count": universe.window_pair_count,
        "ld_tag_r2_threshold": super::GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
        "eligibility_rule": "LD-tag prepared universe; exclude within-group pairs; each pair counted once by earliest eligible prepared-unit canonical owner",
    });
    let timing_seconds = json!({
        "sample_construction": selection_seconds,
        "strata_diagnostics": strata_seconds,
        "null_preprocessing_total": null_preprocessing_seconds,
        "exact_scoring_cumulative_by_target": scoring_seconds_by_target,
        "total_diagnostic": total_started.elapsed().as_secs_f64(),
    });
    let strata_definitions = json!({
        "canonical_owner_scan_unit": "canonical owner scan unit id; this is the current operational window/group owner, not an independently inferred LD block or block-pair class",
        "min_maf": "minimum of the two binary logic-coded support frequencies and their complements; a diagnostic proxy, not dosage MAF",
        "min_maf_cutpoints": maf_cutpoints,
        "min_maf_cutpoints_status": if maf_default { "configurable provisional defaults; no canonical project cutpoints found; scientific defaults remain to be chosen" } else { "user-configured" },
        "pairwise_r2": "binary genotype/cell-count r-squared over the same n_samples",
        "pairwise_r2_cutpoints": r2_cutpoints,
        "pairwise_r2_cutpoints_status": if r2_default { "configurable provisional defaults; no canonical project cutpoints found; scientific defaults remain to be chosen" } else { "user-configured" },
        "min_cell": ["0", "1-4", "5-9", "10-19", ">=20"],
    });
    let manifest = json!({
        "sampling_algorithm": PAIR_SAMPLE_ALGORITHM,
        "algorithm_version": 1,
        "pair_sample_seed": pair_sample_seed,
        "quantile_method": "Hyndman-Fan type 7 linear interpolation; paired gap is full_maximum - sampled_maximum for each identical null draw",
        "requested_sample_sizes": REQUESTED_SAMPLE_SIZES,
        "actual_sample_sizes": actual_sample_sizes,
        "sample_sizes": sample_sizes_json,
        "universe_definition": "unique canonical i<j pairs eligible under the exact Max6 observed/full-maxT path: same QC/LD-tag prepared scan units, same within-group exclusions, and existing canonical-owner rule",
        "universe_metadata": universe_metadata,
        "marker_order_hash": marker_order_hash(logic_bits),
        "sample_order_hash": sample_order_hash(logic_bits),
        "universe_count": sample.unique_universe_count,
        "universe_hash": sample.unique_universe_hash,
        "max_sample_set_hash": sample.sample_hash,
        "universe_hash_algorithm": "commutative_splitmix64_sum_xor_v1_plus_count",
        "pair_priority_tiebreak": "priority ascending, canonical pair id ascending",
        "sample_hash_algorithm": "fnv1a64 over canonical pair ids in priority order; version 1",
        "sampling_with_replacement": false,
        "primary_sampling_uses_strata_weights": false,
        "null_draw_count": null_draws.len(),
        "null_seed": null_seed,
        "null_seed_semantics": "root seed for the existing ordered parametric-null stream; pair_sample_seed is independent; null_index identifies the draw",
        "paired_full_maxT_reference": true,
        "monotonicity_violations": monotonicity_violations,
        "production_threshold_modified": false,
        "production_threshold_use": false,
        "interpretation": "sampled search-size diagnostic only; not full-universe FWER calibration",
        "timing_seconds": timing_seconds,
        "sample_storage_capacity_bytes": sample.storage_capacity_bytes,
        "pair_strata_file": strata_path,
        "strata_definitions": strata_definitions,
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed to serialize sampled Max6 manifest: {error}"))?;
    write_atomic(&manifest_path, &manifest_bytes)?;
    notify_progress(
        "max6_screen_sample_scoring",
        null_draws.len(),
        null_draws.len(),
        sample.unique_universe_count,
    )?;

    Ok(SampledScreenSeedOutput {
        pair_sample_seed,
        maxima_tsv: maxima_path,
        manifest_json: manifest_path,
        strata_tsv: strata_path,
        target_summaries,
        selection_seconds,
        strata_seconds,
        null_preprocessing_seconds,
        scoring_seconds_by_target,
        sample_storage_capacity_bytes: sample.storage_capacity_bytes,
        total_seconds: total_started.elapsed().as_secs_f64(),
    })
}

fn numeric_seed_variability(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sample_sd = if values.len() > 1 {
        (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    json!({
        "n_pair_sample_seeds": values.len(),
        "mean": mean,
        "sample_sd": sample_sd,
        "min": sorted[0],
        "max": sorted[sorted.len() - 1],
        "range": sorted[sorted.len() - 1] - sorted[0],
    })
}

fn sample_seed_variability(outputs: &[SampledScreenSeedOutput]) -> Value {
    let mut by_target = Vec::with_capacity(REQUESTED_SAMPLE_SIZES.len());
    for target_index in 0..REQUESTED_SAMPLE_SIZES.len() {
        let per_seed = outputs
            .iter()
            .map(|output| {
                let summary = &output.target_summaries[target_index];
                json!({
                    "pair_sample_seed": output.pair_sample_seed,
                    "target_M": summary.target_m,
                    "actual_M": summary.actual_m,
                    "exact_full_max_capture_rate": summary.exact_capture_rate,
                    "paired_full_minus_sampled_gap_median_type7": summary.paired_gap_median,
                    "paired_full_minus_sampled_gap_p95_type7": summary.paired_gap_p95,
                    "paired_full_minus_sampled_gap_p99_type7": summary.paired_gap_p99,
                    "full_minus_sampled_q95_type7": summary.full_minus_sampled_q95,
                    "full_minus_sampled_q99_type7": summary.full_minus_sampled_q99,
                })
            })
            .collect::<Vec<_>>();
        let capture_rates = outputs
            .iter()
            .map(|output| output.target_summaries[target_index].exact_capture_rate)
            .collect::<Vec<_>>();
        let median_gaps = outputs
            .iter()
            .map(|output| output.target_summaries[target_index].paired_gap_median)
            .collect::<Vec<_>>();
        let p95_gaps = outputs
            .iter()
            .map(|output| output.target_summaries[target_index].paired_gap_p95)
            .collect::<Vec<_>>();
        let q95_differences = outputs
            .iter()
            .map(|output| output.target_summaries[target_index].full_minus_sampled_q95)
            .collect::<Vec<_>>();
        by_target.push(json!({
            "target_M": REQUESTED_SAMPLE_SIZES[target_index],
            "exact_full_max_capture_rate_across_pair_seeds": numeric_seed_variability(&capture_rates),
            "paired_gap_median_across_pair_seeds": numeric_seed_variability(&median_gaps),
            "paired_gap_p95_across_pair_seeds": numeric_seed_variability(&p95_gaps),
            "full_minus_sampled_q95_across_pair_seeds": numeric_seed_variability(&q95_differences),
            "per_seed": per_seed,
        }));
    }
    Value::Array(by_target)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sampled_screen_null_diagnostics(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_draws: &[Vec<f64>],
    null_seed: u64,
    pair_sample_seeds: &[u64],
    full_calibration: &Max6ScreenCalibration,
    maf_cutpoints: &[f64],
    r2_cutpoints: &[f64],
    out_prefix: &str,
    pool: Option<&ThreadPool>,
    mut notify_progress: impl FnMut(&str, usize, usize, usize) -> Result<(), String>,
) -> Result<SampledScreenDiagnosticOutput, String> {
    let seed_string = pair_sample_seeds
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let parsed_seeds = parse_pair_sample_seeds(&seed_string, null_seed)?;
    if parsed_seeds != pair_sample_seeds {
        return Err("pair-sample seed parser changed seed ordering".to_string());
    }
    if null_draws.len() != 100 || full_calibration.maxima.len() != null_draws.len() {
        return Err(
            "multi-seed diagnostic requires the same 100 full-calibration null draws".to_string(),
        );
    }
    let maximizing_pairs = full_calibration.maximizing_pairs.as_ref().ok_or_else(|| {
        "multi-seed diagnostic requires full-scan maximizing pair identities".to_string()
    })?;
    if maximizing_pairs.len() != null_draws.len() {
        return Err("full maximizing-pair identities do not match the null draws".to_string());
    }

    let total_started = Instant::now();
    let full_maximizer_geometry_tsv = write_full_maximizer_pair_geometry(
        logic_bits,
        universe,
        full_calibration,
        null_seed,
        out_prefix,
    )?;
    let mut outputs = Vec::with_capacity(pair_sample_seeds.len());
    for &pair_sample_seed in pair_sample_seeds {
        let seed_prefix =
            format!("{out_prefix}.max6_grmmax6.sampled_screen_seed_{pair_sample_seed:016x}");
        outputs.push(run_sampled_screen_null_diagnostic_one_seed(
            logic_bits,
            universe,
            null_draws,
            null_seed,
            pair_sample_seed,
            full_calibration,
            maf_cutpoints,
            r2_cutpoints,
            seed_prefix.as_str(),
            pool,
            &mut notify_progress,
        )?);
    }
    let first = outputs
        .first()
        .ok_or_else(|| "multi-seed sampled diagnostic produced no outputs".to_string())?;
    let full_maxima_summary = sample_maxima_summary(full_calibration.maxima.as_slice());
    let seed_outputs = outputs
        .iter()
        .map(|output| {
            json!({
                "pair_sample_seed": output.pair_sample_seed,
                "maxima_tsv": output.maxima_tsv,
                "manifest_json": output.manifest_json,
                "strata_tsv": output.strata_tsv,
                "sample_storage_capacity_bytes": output.sample_storage_capacity_bytes,
                "timing_seconds": {
                    "sample_construction": output.selection_seconds,
                    "strata_diagnostics": output.strata_seconds,
                    "null_preprocessing_total": output.null_preprocessing_seconds,
                    "exact_scoring_cumulative_by_target": output.scoring_seconds_by_target,
                    "total_diagnostic": output.total_seconds,
                },
            })
        })
        .collect::<Vec<_>>();
    let index = json!({
        "sampling_algorithm": PAIR_SAMPLE_ALGORITHM,
        "algorithm_version": 2,
        "pair_sample_seeds": pair_sample_seeds,
        "requested_sample_sizes": REQUESTED_SAMPLE_SIZES,
        "actual_sample_sizes": actual_sample_sizes(full_calibration.pair_universe_size),
        "unique_universe_count": full_calibration.pair_universe_size,
        "full_maxima_summary_type7": full_maxima_summary,
        "full_maximizer_pair_geometry_tsv": full_maximizer_geometry_tsv,
        "paired_null_draw_count": null_draws.len(),
        "null_seed": null_seed,
        "same_full_maxima_reused_for_all_pair_seeds": true,
        "sample_seed_variability": sample_seed_variability(outputs.as_slice()),
        "pair_sample_outputs": seed_outputs,
        "timing_seconds": {
            "all_pair_seed_diagnostics": total_started.elapsed().as_secs_f64(),
        },
        "production_threshold_modified": false,
        "gev_fitted": false,
        "production_threshold_use": false,
        "interpretation": "paired sampled search-size diagnostic only; no sampled or extreme-value threshold is fitted or deployed",
    });
    let index_path = format!("{out_prefix}.max6_grmmax6.sampled_screen_runs.json");
    let index_bytes = serde_json::to_vec_pretty(&index)
        .map_err(|error| format!("failed to serialize sampled Max6 multi-seed index: {error}"))?;
    write_atomic(&index_path, &index_bytes)?;

    Ok(SampledScreenDiagnosticOutput {
        maxima_tsv: first.maxima_tsv.clone(),
        manifest_json: index_path,
        strata_tsv: first.strata_tsv.clone(),
        full_maximizer_geometry_tsv,
    })
}

#[cfg(test)]
mod tests {
    use super::super::four_cell::FourCellGate;
    use super::super::{
        GarfieldLogicSite, GarfieldLogicSiteMode, Max6GrmMax6LdPreparedUnit,
        Max6ScreenPairOwnership, Max6ScreenPreparedUniverse,
    };
    use super::*;
    #[test]
    fn exceedance_grid_has_fixed_endpoints_and_exact_spacing() {
        let grid = build_exceedance_audit_grid(&[1.0, 2.0, 3.0]).unwrap();
        let fixed = grid
            .thresholds
            .iter()
            .filter(|threshold| {
                threshold
                    .sources
                    .iter()
                    .any(|source| source == "fixed_grid")
            })
            .map(|threshold| threshold.score)
            .collect::<Vec<_>>();

        assert_eq!(fixed.len(), 201);
        assert_eq!(fixed.first(), Some(&0.0));
        assert_eq!(fixed.last(), Some(&50.0));
        assert!(fixed
            .windows(2)
            .all(|window| (window[1] - window[0] - 0.25).abs() < 1.0e-12));
    }

    #[test]
    fn exceedance_grid_deduplicates_anchor_and_preserves_source_labels() {
        let maxima = (0..101).map(|value| value as f64 * 0.5).collect::<Vec<_>>();
        let grid = build_exceedance_audit_grid(&maxima).unwrap();
        let q80 = quantile_type7(&maxima, 0.8).unwrap();
        let anchor = grid
            .thresholds
            .iter()
            .find(|threshold| threshold.score == q80)
            .unwrap();

        assert!(anchor
            .sources
            .iter()
            .any(|source| source == "in_sample_audit_anchor:q80"));
        assert_eq!(
            grid.thresholds
                .iter()
                .filter(|threshold| threshold.score == q80)
                .count(),
            1
        );
        assert!(grid
            .thresholds
            .windows(2)
            .all(|pair| pair[0].score < pair[1].score));
        assert!(build_exceedance_audit_grid(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn fixed_grid_coverage_reports_maxima_above_fifty_without_extending_grid() {
        let maxima = [12.0, 50.0, 50.0001, 75.0];
        let grid = build_exceedance_audit_grid(&maxima).unwrap();

        assert_eq!(grid.fixed_grid_points, 201);
        assert_eq!(grid.maxima_above_fixed_grid_count, 2);
        assert_eq!(grid.max_maxima_above_fixed_grid, Some(75.0));
        assert!(grid.thresholds.iter().any(|threshold| {
            threshold.score > 50.0
                && threshold
                    .sources
                    .iter()
                    .any(|source| source.starts_with("in_sample_audit_anchor:"))
        }));
        assert_eq!(
            grid.thresholds
                .iter()
                .filter(|threshold| threshold.sources.iter().any(|s| s == "fixed_grid"))
                .map(|threshold| threshold.score)
                .fold(f64::NEG_INFINITY, f64::max),
            50.0
        );
    }

    #[test]
    fn exceedance_counts_use_strict_greater_than_at_ties() {
        assert_eq!(strict_exceedance_count(&[0.5, 1.0, 1.0, 1.5], 1.0), 1);
    }

    #[test]
    fn audit_metric_boundaries_are_explicit_and_not_silently_zeroed() {
        assert_eq!(
            relative_bias_metric(0.0, 0.0).na_reason,
            Some("lambda_full_is_zero")
        );
        assert_eq!(
            theta_eff_metric(0.0, 0.0).na_reason,
            Some("lambda_full_is_zero")
        );
        assert_eq!(
            theta_eff_metric(1.0, 2.0).na_reason,
            Some("fwer_full_is_one")
        );
        assert_eq!(theta_eff_metric(0.0, 2.0).value, Some(0.0));
        assert_eq!(theta_eff_metric(0.5, 2.0).value, Some(-0.5_f64.ln() / 2.0));
    }

    #[test]
    fn wilson_interval_matches_reference_fixture() {
        let interval = wilson_fwer_interval(5, 100).unwrap();
        assert!((interval.estimate - 0.05).abs() < 1.0e-15);
        assert!((interval.lower - 0.0215).abs() < 0.001);
        assert!((interval.upper - 0.1118).abs() < 0.001);
    }

    #[test]
    fn paired_bootstrap_mean_interval_is_seed_deterministic() {
        let values = [0.0, 1.0, 2.0, 4.0, 8.0];
        let first = bootstrap_mean_interval(&values, 7, 500).unwrap();
        let repeated = bootstrap_mean_interval(&values, 7, 500).unwrap();
        let other_seed = bootstrap_mean_interval(&values, 8, 500).unwrap();

        assert_eq!(first, repeated);
        assert_eq!(first.estimate, 3.0);
        assert!(first.lower <= first.estimate && first.estimate <= first.upper);
        assert!(other_seed.lower <= other_seed.estimate && other_seed.estimate <= other_seed.upper);
        assert!(first.lower.is_finite() && first.upper.is_finite());
    }

    #[test]
    fn paired_threshold_summary_validates_null_pairing_and_reports_metrics() {
        let summary =
            paired_threshold_count_summary(&[0, 1, 2, 0], &[0, 0, 1, 0], 31, 200).unwrap();

        assert_eq!(summary.full_lambda, 0.75);
        assert_eq!(summary.sample_lambda, 0.25);
        assert_eq!(summary.zero_exceedance_classification_accuracy, 0.75);
        assert_eq!(summary.relative_bias.value, Some(-2.0 / 3.0));
        assert!(paired_threshold_count_summary(&[0, 1], &[0], 31, 200).is_err());
    }

    #[test]
    fn post_stratified_ht_is_unavailable_for_an_unsampled_nonempty_stratum() {
        let estimate =
            stratified_ht_estimate(&[100, 50, 25, 10, 5], &[20, 0, 5, 2, 1], &[10, 0, 2, 1, 1])
                .unwrap();
        assert_eq!(estimate.value, None);
        assert_eq!(estimate.na_reason, Some("nonempty_stratum_has_zero_sample"));

        let covered =
            stratified_ht_estimate(&[100, 50, 25, 10, 5], &[20, 10, 5, 2, 1], &[10, 2, 1, 1, 1])
                .unwrap();
        assert_eq!(covered.value, Some(75.0));
    }

    #[test]
    fn tail_top_k_is_bounded_and_orders_score_then_canonical_pair() {
        let mut top = TailTopK::new(3);
        top.consider((0, 3), 7.0, FourCellGate::Xor).unwrap();
        top.consider((1, 2), 8.0, FourCellGate::And00).unwrap();
        top.consider((0, 2), 7.0, FourCellGate::And11).unwrap();
        top.consider((0, 1), 7.0, FourCellGate::DosageInteraction)
            .unwrap();
        top.consider((0, 4), 6.0, FourCellGate::And01).unwrap();

        let ranked = top.into_ranked();

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].pair, (1, 2));
        assert_eq!(ranked[0].score, 8.0);
        assert_eq!(ranked[1].pair, (0, 1));
        assert_eq!(ranked[2].pair, (0, 2));
    }

    #[test]
    fn tail_top_k_merge_matches_one_pass_and_rejects_noncanonical_pairs() {
        let entries = [
            ((0, 2), 4.0, FourCellGate::And00),
            ((0, 3), 9.0, FourCellGate::Xor),
            ((1, 2), 9.0, FourCellGate::DosageInteraction),
            ((1, 3), 2.0, FourCellGate::And10),
            ((2, 3), 6.0, FourCellGate::And11),
        ];
        let mut whole = TailTopK::new(3);
        let mut left = TailTopK::new(3);
        let mut right = TailTopK::new(3);
        for (index, &(pair, score, channel)) in entries.iter().enumerate() {
            whole.consider(pair, score, channel).unwrap();
            if index % 2 == 0 {
                left.consider(pair, score, channel).unwrap();
            } else {
                right.consider(pair, score, channel).unwrap();
            }
        }
        left.merge(right);

        assert_eq!(left.into_ranked(), whole.into_ranked());
        assert!(TailTopK::new(4)
            .consider((2, 1), 3.0, FourCellGate::And00)
            .is_err());
    }

    #[test]
    fn finite_population_uniform_sample_max_cdf_matches_hand_calculation() {
        let cdf = (1..=4)
            .map(|rank| finite_population_sample_max_cdf(5, 2, rank).unwrap())
            .collect::<Vec<_>>();

        for (observed, expected) in cdf.iter().zip([0.4, 0.7, 0.9, 1.0]) {
            assert!((observed - expected).abs() < 1.0e-15);
        }
        assert!(finite_population_sample_max_cdf(5, 0, 1).is_err());
        assert!(finite_population_sample_max_cdf(5, 6, 1).is_err());
    }

    fn sampled_test_logic_bits(n_rows: usize, n_samples: usize) -> GarfieldLogicBits {
        let row_words = n_samples.div_ceil(64);
        let mut bits_flat = vec![0u64; n_rows * row_words];
        for row in 0..n_rows {
            for sample in 0..n_samples {
                if (sample * 17 + row * 11 + sample * row * 3) % 29 < 9 {
                    bits_flat[row * row_words + (sample >> 6)] |= 1u64 << (sample & 63);
                }
            }
        }
        GarfieldLogicBits {
            bits_flat,
            bits_mmap: None,
            bits_mmap_words: 0,
            bits_hi_flat: None,
            row_words,
            sample_ids: (0..n_samples)
                .map(|index| format!("sample_{index}"))
                .collect(),
            sites: (0..n_rows)
                .map(|index| {
                    GarfieldLogicSite::owned(
                        "1",
                        index as i32 + 1,
                        format!("snp_{index}"),
                        "A",
                        "T",
                        GarfieldLogicSiteMode::Bin,
                    )
                })
                .collect(),
            group_ids: vec![0; n_rows],
            n_samples,
        }
    }

    fn sampled_test_universe(
        n_rows: usize,
        logic_bits: &GarfieldLogicBits,
    ) -> Max6ScreenPreparedUniverse {
        let rows = (0..n_rows).collect::<Vec<_>>();
        sampled_test_universe_with_rows(rows, logic_bits)
    }

    fn sampled_test_universe_with_rows(
        rows: Vec<usize>,
        logic_bits: &GarfieldLogicBits,
    ) -> Max6ScreenPreparedUniverse {
        let n_rows = rows.len();
        let units = vec![Max6GrmMax6LdPreparedUnit {
            selected_global_rows: rows,
            pair_groups: None,
            ld_compression: Vec::new(),
        }];
        let active_rows = (0..n_rows).collect::<Vec<_>>();
        let pair_ownership = Max6ScreenPairOwnership::build(
            units.as_slice(),
            active_rows.as_slice(),
            logic_bits.sites.len(),
        )
        .unwrap();
        Max6ScreenPreparedUniverse {
            original_unit_indices: vec![0],
            units,
            active_rows,
            pair_ownership,
            window_pair_count: n_rows * n_rows.saturating_sub(1) / 2,
        }
    }

    #[test]
    fn weighted_exceedance_scan_matches_full_exact_scores_and_is_thread_deterministic() {
        let logic_bits = sampled_test_logic_bits(36, 96);
        let universe = sampled_test_universe(logic_bits.sites.len(), &logic_bits);
        let nulls = (0..3usize)
            .map(|draw| {
                (0..logic_bits.n_samples)
                    .map(|sample| {
                        ((sample * 19 + draw * 23 + sample * draw * 7) % 43) as f64 - 21.0
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let (calibration_maxima, _, pair_count) =
            super::super::scan_max6_screen_null_maxima_with_progress(
                &logic_bits,
                &universe,
                nulls.as_slice(),
                None,
                false,
                |_, _, _| Ok(()),
            )
            .unwrap();
        assert_eq!(pair_count, 36 * 35 / 2);
        let grid = build_exceedance_audit_grid(calibration_maxima.as_slice()).unwrap();
        let sample_plan = prepare_weighted_exceedance_sampling_plan(
            &logic_bits,
            &universe,
            WEIGHTED_EXCEEDANCE_PAIR_SEEDS,
            |_, _, _, _| Ok(()),
        )
        .unwrap();
        let repeated_plan = prepare_weighted_exceedance_sampling_plan(
            &logic_bits,
            &universe,
            WEIGHTED_EXCEEDANCE_PAIR_SEEDS,
            |_, _, _, _| Ok(()),
        )
        .unwrap();
        assert_eq!(
            sample_plan.unique_universe_hash,
            repeated_plan.unique_universe_hash
        );
        assert_eq!(
            sample_plan.universe_stratum_counts.iter().sum::<usize>(),
            pair_count
        );
        assert_eq!(sample_plan.seeds.len(), 5);
        for (left, right) in sample_plan.seeds.iter().zip(repeated_plan.seeds.iter()) {
            assert_eq!(left.pair_sample_seed, right.pair_sample_seed);
            assert_eq!(left.uniform_sample_hashes, right.uniform_sample_hashes);
            assert_eq!(left.support_sample_hashes, right.support_sample_hashes);
        }

        let supports = support_counts_by_active_slot(&logic_bits, &universe);
        let mut selected_counts = vec![[0usize; 6]; sample_plan.seeds.len()];
        for first in 0..logic_bits.sites.len() - 1 {
            for second in first + 1..logic_bits.sites.len() {
                let stratum = pair_min_cell_stratum(
                    &logic_bits,
                    &universe,
                    supports.as_slice(),
                    first,
                    second,
                )
                .unwrap();
                for seed_index in 0..sample_plan.seeds.len() {
                    let memberships = sample_plan
                        .memberships_for_pair(seed_index, (first, second), stratum)
                        .unwrap();
                    for slot in 0..6 {
                        selected_counts[seed_index][slot] += usize::from(memberships[slot]);
                    }
                }
            }
        }
        for seed_counts in &selected_counts {
            assert_eq!(seed_counts, &[pair_count; 6]);
        }

        let serial = super::super::scan_max6_screen_exceedance_with_progress(
            &logic_bits,
            &universe,
            nulls.as_slice(),
            calibration_maxima.as_slice(),
            grid.thresholds.as_slice(),
            &sample_plan,
            None,
            |_, _, _, _| Ok(()),
        )
        .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let parallel = super::super::scan_max6_screen_exceedance_with_progress(
            &logic_bits,
            &universe,
            nulls.as_slice(),
            calibration_maxima.as_slice(),
            grid.thresholds.as_slice(),
            &sample_plan,
            Some(&pool),
            |_, _, _, _| Ok(()),
        )
        .unwrap();
        assert_eq!(serial.full_counts, parallel.full_counts);
        assert_eq!(serial.sampled_counts, parallel.sampled_counts);
        assert_eq!(serial.top_pairs, parallel.top_pairs);
        assert_eq!(serial.unique_pair_count, pair_count);
        assert!(serial.preflight_accumulator_bytes > 0);
        assert!(serial.reserved_accumulator_bytes > 0);
        assert!(serial.preflight_output_bytes > 0);
        assert!(serial.reserved_output_bytes > 0);
        assert!(serial.preflight_total_bytes >= serial.preflight_accumulator_bytes);

        for draw_index in 0..nulls.len() {
            let null_y = &nulls[draw_index];
            let lookup = PackedYSumLookup::build(null_y, logic_bits.n_samples).unwrap();
            let margins = OneCellMarginTable::build_for_rows(
                logic_bits.bits(),
                logic_bits.row_words,
                universe.active_rows.as_slice(),
                null_y,
                logic_bits.n_samples,
                &lookup,
            )
            .unwrap();
            let draw = logic_residual_screen::Max6ScreenNullDraw {
                residual: null_y,
                lookup: &lookup,
                margins: &margins,
            };
            let mut exact_pairs = Vec::new();
            let mut expected_by_stratum = vec![0u32; 5 * grid.thresholds.len()];
            let mut max_scratch = [f64::NEG_INFINITY];
            let mut sum11 = [0.0];
            let mut pair_scores = [None];
            for first in 0..logic_bits.sites.len() - 1 {
                for second in first + 1..logic_bits.sites.len() {
                    let first_start = first * logic_bits.row_words;
                    let second_start = second * logic_bits.row_words;
                    logic_residual_screen::update_max6_one_cell_pair_batch(
                        &logic_bits.bits()[first_start..first_start + logic_bits.row_words],
                        &logic_bits.bits()[second_start..second_start + logic_bits.row_words],
                        logic_bits.n_samples,
                        universe.pair_ownership.active_slot(first),
                        universe.pair_ownership.active_slot(second),
                        None,
                        std::slice::from_ref(&draw),
                        &mut sum11,
                        &mut max_scratch,
                        None,
                        Some(&mut pair_scores),
                    )
                    .unwrap();
                    let Some(score) = pair_scores[0] else {
                        continue;
                    };
                    let candidate = TailPairScore {
                        pair: (first, second),
                        score: score.score,
                        best_channel: score.gate,
                    };
                    exact_pairs.push(candidate);
                    let stratum = pair_min_cell_stratum(
                        &logic_bits,
                        &universe,
                        supports.as_slice(),
                        first,
                        second,
                    )
                    .unwrap();
                    for (threshold_index, threshold) in grid.thresholds.iter().enumerate() {
                        if candidate.score > threshold.score {
                            expected_by_stratum
                                [stratum * grid.thresholds.len() + threshold_index] += 1;
                        }
                    }
                }
            }
            exact_pairs.sort_by(full_rank_cmp);
            assert_eq!(
                exact_pairs.first().map(|pair| pair.score.to_bits()),
                Some(calibration_maxima[draw_index].to_bits())
            );
            for (rank, (observed, expected)) in serial.top_pairs[draw_index]
                .iter()
                .zip(exact_pairs.iter())
                .enumerate()
            {
                assert_eq!(
                    observed.pair, expected.pair,
                    "draw={draw_index} rank={rank}"
                );
                assert_eq!(observed.score.to_bits(), expected.score.to_bits());
                assert_eq!(observed.best_channel, expected.best_channel);
            }
            for stratum in 0..5 {
                for threshold_index in 0..grid.thresholds.len() {
                    let full = serial.full_counts
                        [(draw_index * 5 + stratum) * grid.thresholds.len() + threshold_index];
                    assert_eq!(
                        full,
                        expected_by_stratum[stratum * grid.thresholds.len() + threshold_index]
                    );
                    for slot in 0..sample_plan.sample_slot_count() {
                        let sampled = serial.sampled_counts[((draw_index
                            * sample_plan.sample_slot_count()
                            + slot)
                            * 5
                            + stratum)
                            * grid.thresholds.len()
                            + threshold_index];
                        assert_eq!(sampled, full);
                    }
                }
            }
            for threshold_index in 0..grid.thresholds.len() {
                let full_total = (0..5)
                    .map(|stratum| {
                        serial.full_counts
                            [(draw_index * 5 + stratum) * grid.thresholds.len() + threshold_index]
                            as f64
                    })
                    .sum::<f64>();
                let any = if full_total > 0.0 { 1.0 } else { 0.0 };
                assert!(full_total + 1.0e-12 >= any);
            }
        }

        let temp_dir = std::env::temp_dir().join(format!(
            "janusx-weighted-exceedance-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let prefix = temp_dir.join("tiny").to_string_lossy().into_owned();
        let paths = write_weighted_exceedance_audit(
            &logic_bits,
            &universe,
            nulls.as_slice(),
            calibration_maxima.as_slice(),
            &grid,
            &sample_plan,
            &serial,
            0x19A7_6C2D_4E80_B531,
            0x19A7_6C2D_4E80_B531,
            prefix.as_str(),
        )
        .unwrap();
        let manifest_bytes = std::fs::read(&paths.manifest_json).unwrap();
        let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest["production_threshold_modified"], false);
        assert_eq!(manifest["evt_fitted"], false);
        assert_eq!(manifest["null_count"], nulls.len());
        assert!(manifest["calibration_null_draws_hash"]
            .as_str()
            .unwrap()
            .starts_with("fnv1a64_calibration_null_f64_bits_v1:"));
        assert_eq!(
            manifest["grid"]["fixed_grid_full_maxima_above_50_count"],
            grid.maxima_above_fixed_grid_count
        );
        let summary = std::fs::read_to_string(&paths.summary_tsv).unwrap();
        assert!(summary.contains("theta_na_reason"));
        assert!(summary.contains("poststratified_ht"));
        let per_null = std::fs::read_to_string(&paths.per_null_tsv).unwrap();
        assert!(per_null.contains("full_c4"));
        assert!(per_null.contains("pair_sample_hash"));
        std::fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn full_maximizer_binary_geometry_reports_cells_maf_ld_and_owner() {
        let n_rows = 3usize;
        let n_samples = 96usize;
        let mut logic_bits = sampled_test_logic_bits(n_rows, n_samples);
        logic_bits.bits_flat.fill(0);
        for row in 0..n_rows {
            for sample in (0..n_samples).step_by(3) {
                logic_bits.bits_flat[row * logic_bits.row_words + (sample >> 6)] |=
                    1u64 << (sample & 63);
            }
        }
        let universe = sampled_test_universe(n_rows, &logic_bits);
        let supports = support_counts_by_active_slot(&logic_bits, &universe);

        let geometry = pair_binary_geometry(&logic_bits, &universe, &supports, 0, 1).unwrap();

        assert_eq!(geometry.counts, [64, 0, 0, 32]);
        assert_eq!(geometry.min_cell, 0);
        assert!((geometry.maf1 - 1.0 / 3.0).abs() < 1.0e-12);
        assert_eq!(geometry.maf1, geometry.maf2);
        assert!((geometry.pairwise_r2 - 1.0).abs() < 1.0e-12);
        assert_eq!(geometry.owner_scan_unit, 0);
    }

    #[test]
    fn bottom_k_is_canonical_unique_nested_and_reproducible() {
        let universe = (0..40usize)
            .flat_map(|i| ((i + 1)..45usize).map(move |j| (i, j)))
            .collect::<Vec<_>>();
        let small = bottom_k_pairs(universe.iter().copied(), 99, 17).unwrap();
        let medium = bottom_k_pairs(universe.iter().copied(), 99, 101).unwrap();
        let large = bottom_k_pairs(universe.iter().copied(), 99, usize::MAX).unwrap();
        let repeat = bottom_k_pairs(universe.iter().copied(), 99, 101).unwrap();

        let small_pairs = small.decoded_pairs();
        let medium_pairs = medium.decoded_pairs();
        let large_pairs = large.decoded_pairs();
        let repeat_pairs = repeat.decoded_pairs();
        assert_eq!(small_pairs, medium_pairs[..17]);
        assert_eq!(medium_pairs, large_pairs[..101]);
        assert_eq!(medium_pairs, repeat_pairs);
        assert!(large_pairs.iter().all(|&(i, j)| i < j));
        assert_eq!(
            large_pairs
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            large_pairs.len()
        );
    }

    #[test]
    fn bottom_k_is_independent_of_rayon_thread_count_and_input_order() {
        let mut universe = (0..36usize)
            .flat_map(|i| ((i + 1)..40usize).map(move |j| (i, j)))
            .collect::<Vec<_>>();
        let serial = bottom_k_pairs(universe.iter().copied(), 7, 333).unwrap();
        universe.reverse();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel = pool
            .install(|| bottom_k_pairs(universe.iter().copied(), 7, 333))
            .unwrap();
        assert_eq!(serial.decoded_pairs(), parallel.decoded_pairs());
        assert_eq!(serial.sample_hash, parallel.sample_hash);
        assert_eq!(serial.unique_universe_hash, parallel.unique_universe_hash);
    }

    #[test]
    fn bottom_k_reports_actual_size_and_rejects_noncanonical_pairs() {
        let sample = bottom_k_pairs([(0, 2), (1, 2), (0, 1)], 5, 100).unwrap();
        assert_eq!(sample.unique_universe_count, 3);
        assert_eq!(sample.decoded_pairs().len(), 3);
        assert_eq!(
            actual_sample_sizes(sample.unique_universe_count),
            [3, 3, 3, 3]
        );
        assert_eq!(
            actual_sample_sizes(200_000),
            [100_000, 200_000, 200_000, 200_000]
        );
        assert!(bottom_k_pairs([(2, 1)], 5, 1).is_err());
    }

    #[test]
    fn scaling_targets_include_50m_and_report_actual_nested_sizes() {
        assert_eq!(
            REQUESTED_SAMPLE_SIZES.as_slice(),
            &[100_000, 1_000_000, 10_000_000, 50_000_000]
        );
        assert_eq!(
            actual_sample_sizes(80_000_000).as_slice(),
            &[100_000, 1_000_000, 10_000_000, 50_000_000]
        );
        assert_eq!(
            actual_sample_sizes(7_000_000).as_slice(),
            &[100_000, 1_000_000, 7_000_000, 7_000_000]
        );
    }

    #[test]
    fn sampled_pair_seed_list_is_unique_and_disjoint_from_null_seed() {
        assert_eq!(
            parse_pair_sample_seeds("101,202,303,404,505", 42).unwrap(),
            vec![101, 202, 303, 404, 505]
        );
        assert!(parse_pair_sample_seeds("101,202,101", 42).is_err());
        assert!(parse_pair_sample_seeds("101,42,303", 42).is_err());
        assert!(parse_pair_sample_seeds("101,202", 42).is_err());
    }

    #[test]
    fn primary_bottom_k_selector_has_no_strata_weight_input() {
        let pairs = (0..12usize)
            .flat_map(|i| ((i + 1)..15usize).map(move |j| (i, j)))
            .collect::<Vec<_>>();
        let selected = bottom_k_pairs(pairs.iter().copied(), 314159, 25).unwrap();
        let mut expected = pairs
            .iter()
            .copied()
            .map(|pair| (stable_pair_priority(314159, pair.0, pair.1), pair))
            .collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(
            selected.decoded_pairs(),
            expected[..25].iter().map(|x| x.1).collect::<Vec<_>>()
        );
    }

    #[test]
    fn sampled_pair_exact_score_matches_existing_one_cell_screen_score() {
        let n_rows = 7usize;
        let n_samples = 137usize;
        let logic_bits = sampled_test_logic_bits(n_rows, n_samples);
        let selected_rows = vec![4usize, 1, 6, 0, 2, 5, 3];
        let universe = sampled_test_universe_with_rows(selected_rows.clone(), &logic_bits);
        let null_y = (0..n_samples)
            .map(|sample| ((sample * 19 + 5) % 43) as f64 - 21.0)
            .collect::<Vec<_>>();
        let lookup = PackedYSumLookup::build(null_y.as_slice(), n_samples).unwrap();
        let margins = OneCellMarginTable::build_for_rows(
            logic_bits.bits(),
            logic_bits.row_words,
            universe.active_rows.as_slice(),
            null_y.as_slice(),
            n_samples,
            &lookup,
        )
        .unwrap();
        let mut packed = Vec::with_capacity(n_rows * logic_bits.row_words);
        for &global_row in selected_rows.iter() {
            let start = global_row * logic_bits.row_words;
            packed.extend_from_slice(&logic_bits.bits()[start..start + logic_bits.row_words]);
        }
        let selected_margins = margins.select(selected_rows.as_slice()).unwrap();
        let reference = logic_residual_screen::scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups(
            packed.as_slice(),
            None,
            logic_bits.row_words,
            n_rows,
            null_y.as_slice(),
            n_samples,
            n_rows * (n_rows - 1) / 2,
            true,
            super::super::ResidualProposalMode::Max6,
            super::super::ResidualScreenKernel::OneCell,
            &lookup,
            Some(selected_margins),
            None,
            false,
        )
        .unwrap();
        let draw = logic_residual_screen::Max6ScreenNullDraw {
            residual: null_y.as_slice(),
            lookup: &lookup,
            margins: &margins,
        };
        for candidate in reference.candidates.iter() {
            let first_global = selected_rows[candidate.first];
            let second_global = selected_rows[candidate.second];
            let pair = PairPriority {
                priority: 0,
                pair_id: ((first_global.min(second_global) as u64) << 32)
                    | first_global.max(second_global) as u64,
            };
            let exact = score_sampled_pair_chunk(
                &logic_bits,
                &universe,
                &[pair],
                std::slice::from_ref(&draw),
            )
            .unwrap()[0];
            let tail = score_tail_sample_pair_chunk(
                &logic_bits,
                &universe,
                &[pair],
                std::slice::from_ref(&draw),
            )
            .unwrap()[0]
                .unwrap();
            assert_eq!(
                exact.to_bits(),
                candidate.proposal_score.to_bits(),
                "pair ({}, {}) differs between sampled and existing exact screen kernels",
                candidate.first,
                candidate.second,
            );
            assert_eq!(
                tail.pair,
                (
                    first_global.min(second_global),
                    first_global.max(second_global)
                )
            );
            assert_eq!(tail.score.to_bits(), candidate.best_gate.score.to_bits());
            let reference_channel = if first_global <= second_global {
                candidate.best_gate.gate
            } else {
                match candidate.best_gate.gate {
                    FourCellGate::And01 => FourCellGate::And10,
                    FourCellGate::And10 => FourCellGate::And01,
                    channel => channel,
                }
            };
            assert_eq!(tail.best_channel, reference_channel);
        }
    }

    #[test]
    fn pot_pair_chunk_scores_match_exact_max6_screen_score() {
        let n_rows = 8usize;
        let n_samples = 137usize;
        let logic_bits = sampled_test_logic_bits(n_rows, n_samples);
        let universe = sampled_test_universe(n_rows, &logic_bits);
        let nulls = (0..3usize)
            .map(|draw| {
                (0..n_samples)
                    .map(|sample| ((sample * 19 + draw * 7) % 41) as f64 - 20.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let lookups = nulls
            .iter()
            .map(|null_y| PackedYSumLookup::build(null_y, n_samples).unwrap())
            .collect::<Vec<_>>();
        let margins = nulls
            .iter()
            .zip(lookups.iter())
            .map(|(null_y, lookup)| {
                OneCellMarginTable::build_for_rows(
                    logic_bits.bits(),
                    logic_bits.row_words,
                    universe.active_rows.as_slice(),
                    null_y,
                    n_samples,
                    lookup,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let draws = nulls
            .iter()
            .zip(lookups.iter())
            .zip(margins.iter())
            .map(
                |((residual, lookup), margins)| logic_residual_screen::Max6ScreenNullDraw {
                    residual,
                    lookup,
                    margins,
                },
            )
            .collect::<Vec<_>>();
        let pairs = [
            PairPriority {
                priority: 3,
                pair_id: (1u64 << 32) | 4,
            },
            PairPriority {
                priority: 7,
                pair_id: (2u64 << 32) | 6,
            },
        ];
        let scores = score_pot_pair_chunk(&logic_bits, &universe, &pairs, &draws).unwrap();
        assert_eq!(scores.len(), pairs.len() * draws.len());
        for (pair_index, entry) in pairs.iter().enumerate() {
            let pair = decode_pair_id(entry.pair_id);
            let first_start = pair.0 * logic_bits.row_words;
            let second_start = pair.1 * logic_bits.row_words;
            let mut sum11 = vec![0.0; draws.len()];
            let mut maxima = vec![f64::NEG_INFINITY; draws.len()];
            let mut pair_scores = vec![None; draws.len()];
            logic_residual_screen::update_max6_one_cell_pair_batch(
                &logic_bits.bits()[first_start..first_start + logic_bits.row_words],
                &logic_bits.bits()[second_start..second_start + logic_bits.row_words],
                n_samples,
                universe.pair_ownership.active_slot(pair.0),
                universe.pair_ownership.active_slot(pair.1),
                None,
                &draws,
                &mut sum11,
                &mut maxima,
                None,
                Some(&mut pair_scores),
            )
            .unwrap();
            for draw_index in 0..draws.len() {
                let expected = pair_scores[draw_index].map_or(0.0, |item| item.score);
                assert_eq!(
                    scores[pair_index * draws.len() + draw_index].to_bits(),
                    expected.to_bits()
                );
            }
        }
    }

    #[test]
    fn sampled_nested_maxima_match_full_reference_and_are_monotone() {
        let n_rows = 9usize;
        let n_samples = 137usize;
        let logic_bits = sampled_test_logic_bits(n_rows, n_samples);
        let universe = sampled_test_universe(n_rows, &logic_bits);
        let sample = bottom_k_pairs(
            (0..n_rows).flat_map(|i| ((i + 1)..n_rows).map(move |j| (i, j))),
            123,
            usize::MAX,
        )
        .unwrap();
        let null_draws = (0..5usize)
            .map(|replicate| {
                (0..n_samples)
                    .map(|sample| ((sample * 13 + replicate * 7) % 37) as f64 - 18.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let actual_sizes = [11usize, 28, 34, n_rows * (n_rows - 1) / 2];
        let (sampled, _, _) = score_nested_sample_sizes(
            &logic_bits,
            &universe,
            &sample,
            actual_sizes,
            null_draws.as_slice(),
            None,
            &mut |_, _, _, _| Ok(()),
        )
        .unwrap();
        for replicate in 0..null_draws.len() {
            assert!(sampled[0][replicate] <= sampled[1][replicate]);
            assert!(sampled[1][replicate] <= sampled[2][replicate]);
            assert!(sampled[2][replicate] <= sampled[3][replicate]);
            let full = super::super::scan_max6_screen_null_replicate_maximum(
                &logic_bits,
                &universe,
                null_draws[replicate].as_slice(),
                replicate,
                None,
            )
            .unwrap();
            assert_eq!(sampled[3][replicate].to_bits(), full.to_bits());
        }
    }

    #[test]
    fn support_aware_quotas_target_min_cell_5_to_19_and_preserve_total_budget() {
        let universe_counts = [10usize, 20, 30, 40, 100];

        assert_eq!(
            support_aware_stratum_quotas(universe_counts, 100, 40).unwrap(),
            [5, 9, 17, 23, 46]
        );
        assert_eq!(
            support_aware_stratum_quotas(universe_counts, 100, 50).unwrap(),
            [4, 8, 21, 29, 38]
        );
        assert_eq!(
            support_aware_stratum_quotas(universe_counts, 100, 60).unwrap(),
            [3, 6, 26, 34, 31]
        );
        for share in [40usize, 50, 60] {
            let quotas = support_aware_stratum_quotas(universe_counts, 100, share).unwrap();
            assert_eq!(quotas.iter().sum::<usize>(), 100);
            assert_eq!(quotas[2] + quotas[3], share);
            assert!(quotas
                .iter()
                .zip(universe_counts)
                .all(|(&n, available)| n <= available));
        }
    }

    #[test]
    fn support_aware_quotas_cap_by_true_stratum_counts_and_report_actual_size() {
        let universe_counts = [0usize, 0, 2, 3, 5];

        assert_eq!(
            support_aware_stratum_quotas(universe_counts, 9, 60).unwrap(),
            [0, 0, 2, 3, 4]
        );
        assert_eq!(
            support_aware_stratum_quotas(universe_counts, 20, 60).unwrap(),
            universe_counts
        );
    }

    #[test]
    fn support_aware_plans_keep_nested_quotas_for_all_three_candidate_shares() {
        let universe_counts = [
            100_000_000usize,
            100_000_000,
            150_000_000,
            100_000_000,
            400_000_000,
        ];
        let universe_size = universe_counts.iter().sum::<usize>();
        let plans =
            support_aware_allocation_plans(universe_counts, universe_size, &[40, 50, 60]).unwrap();
        assert_eq!(plans.len(), 3);
        for plan in &plans {
            assert_eq!(plan.actual_sizes, SUPPORT_AWARE_SAMPLE_SIZES);
            for (size_index, &size) in SUPPORT_AWARE_SAMPLE_SIZES.iter().enumerate() {
                assert_eq!(plan.quotas_by_size[size_index].iter().sum::<usize>(), size);
                assert_eq!(
                    plan.quotas_by_size[size_index][2] + plan.quotas_by_size[size_index][3],
                    size * plan.share_percent / 100
                );
            }
            for size_index in 1..3 {
                for stratum in 0..5 {
                    assert!(
                        plan.quotas_by_size[size_index][stratum]
                            >= plan.quotas_by_size[size_index - 1][stratum]
                    );
                }
            }
        }
        assert!(support_aware_allocation_plans(universe_counts, universe_size, &[40, 40]).is_err());
        assert!(support_aware_allocation_plans(universe_counts, universe_size, &[45]).is_err());
    }

    #[test]
    fn support_aware_materialized_samples_are_nested_and_report_stratum_cutoffs() {
        let seed = 77u64;
        let mut pairs_by_stratum = Vec::new();
        for stratum in 0..5 {
            let mut pairs = (0..2)
                .map(|index| {
                    let first = stratum * 10 + index * 2;
                    PairPriority {
                        priority: stable_pair_priority(seed, first, first + 1),
                        pair_id: ((first as u64) << 32) | (first + 1) as u64,
                    }
                })
                .collect::<Vec<_>>();
            pairs.sort_unstable();
            pairs_by_stratum.push(pairs);
        }
        let pools = SupportAwarePairPools {
            unique_universe_count: 10,
            unique_universe_hash: "test-universe".to_string(),
            universe_stratum_counts: [2; 5],
            pairs_by_stratum,
            sample_hashes_by_stratum: vec![String::new(); 5],
            storage_capacity_bytes: 10 * size_of::<PairPriority>(),
            selection_seconds: 0.0,
        };
        let plan = SupportAwareAllocationPlan {
            share_percent: 50,
            actual_sizes: [2, 5, 10],
            quotas_by_size: [[0, 0, 1, 0, 1], [1, 1, 1, 1, 1], [2; 5]],
        };
        let (sample, cutoffs, hashes) = pools.materialize_sample(&plan, seed).unwrap();
        assert_eq!(sample.actual_sizes, [2, 5, 10, 10]);
        assert!(sample.sample.priority_pairs[..2]
            .iter()
            .all(|pair| matches!((pair.pair_id >> 32) / 10, 2 | 4)));
        assert_eq!(sample.sample.priority_pairs.len(), 10);
        assert_eq!(
            hashes[0],
            sample_set_hash(&sample.sample.priority_pairs[..2])
        );
        assert_eq!(
            hashes[1],
            sample_set_hash(&sample.sample.priority_pairs[..5])
        );
        assert_eq!(hashes[2], sample_set_hash(&sample.sample.priority_pairs));
        assert!(cutoffs[0][2].is_some());
        assert!(cutoffs[0][4].is_some());
        assert!(cutoffs[0][0].is_none());
        for size_index in 0..3 {
            for stratum in 0..5 {
                for (rank, priority_pair) in pools.pairs_by_stratum[stratum].iter().enumerate() {
                    let pair = decode_pair_id(priority_pair.pair_id);
                    assert_eq!(
                        sample_contains_pair(pair, seed, cutoffs[size_index][stratum]),
                        rank < plan.quotas_by_size[size_index][stratum]
                    );
                }
            }
        }
    }

    #[test]
    fn support_aware_bottom_k_is_deterministic_unique_and_stratified() {
        let pairs_by_stratum = [
            vec![(0usize, 1usize), (0, 2), (0, 3)],
            vec![(0usize, 4usize), (0, 5)],
            vec![(0usize, 6usize), (0, 7), (0, 8)],
            vec![(0usize, 9usize)],
            vec![(1usize, 2usize), (1, 3)],
        ];
        let limits = [2usize, 2, 1, 1, 1];
        let seed = 55u64;
        let run = |strata: &[Vec<(usize, usize)>]| {
            let mut sampler = StratifiedBottomKPairSampler::new(seed, limits);
            for (stratum, pairs) in strata.iter().enumerate() {
                for &(first, second) in pairs {
                    sampler.consider(first, second, stratum).unwrap();
                }
            }
            sampler.finish().unwrap()
        };

        let selected = run(&pairs_by_stratum);
        let reversed = run(&pairs_by_stratum
            .iter()
            .map(|pairs| pairs.iter().copied().rev().collect())
            .collect::<Vec<_>>());
        assert_eq!(selected.universe_stratum_counts, [3, 2, 3, 1, 2]);
        assert_eq!(selected.unique_universe_count, 11);
        assert_eq!(
            selected.sample_hashes_by_stratum,
            reversed.sample_hashes_by_stratum
        );
        for stratum in 0..5 {
            let expected = pairs_by_stratum[stratum]
                .iter()
                .copied()
                .map(|pair| PairPriority {
                    priority: stable_pair_priority(seed, pair.0, pair.1),
                    pair_id: ((pair.0 as u64) << 32) | pair.1 as u64,
                })
                .collect::<std::collections::BinaryHeap<_>>()
                .into_sorted_vec();
            assert_eq!(
                selected.pairs_by_stratum[stratum],
                expected[..limits[stratum].min(expected.len())]
            );
        }
        assert!(selected
            .pairs_by_stratum
            .iter()
            .flatten()
            .all(|entry| decode_pair_id(entry.pair_id).0 < decode_pair_id(entry.pair_id).1));
        assert!(StratifiedBottomKPairSampler::new(seed, limits)
            .consider(2, 1, 0)
            .is_err());
    }
}
