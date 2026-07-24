use crate::linalg::student_t_p_two_sided;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cmp::Ordering;

#[cfg(test)]
use super::bs::BeamBinaryOp;
use super::bs::BeamRule;

pub const DEFAULT_RULE_PERMUTATION_REPRESENTATIVE_UNITS: usize = 32;
pub const DEFAULT_RULE_NULL_PHYSICAL_CHUNKS: usize = 150;
pub const DEFAULT_RULE_NULL_MIN_SNPS_PER_CHUNK: usize = 50;
pub const DEFAULT_RULE_NULL_MAX_REPEATS: usize = 20;
pub const DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS: usize = 5;
pub const DEFAULT_RULE_NULL_ADAPTIVE_STABLE_REPEATS: usize = 3;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MIN_REPEATS: usize = 5;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MAX_REPEATS: usize = 30;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_STABLE_REPEATS: usize = 3;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_KL_THRESHOLD: f64 = 0.005;
pub const DEFAULT_RULE_STRUCTURE_DENSITY_TOPK: usize = 10;
const DEFAULT_RULE_NULL_QUANTILE: f64 = 0.99;
const DEFAULT_RULE_NULL_Q99_REL_TOL: f64 = 0.05;
// Minimum samples per exact bucket before falling back to the global null.
const NULL_EXACT_MIN_SAMPLES: usize = 10;
// Top-k per repeat: keep the same number of extreme null scores for every
// rule length so singleton / pair / higher-order penalties are estimated under
// the same per-repeat truncation rule.
const DEFAULT_RULE_NULL_TOPK_ALL: usize = 2;
const DEFAULT_RULE_NULL_BUCKET_MAX_RULE_LEN: usize = 5;
const DEFAULT_RULE_NULL_BASE_COMPLEXITY_BIN_COUNT: usize = 4;
const DEFAULT_RULE_NULL_UNIT_GROUP_BIN_COUNT: usize = 3;
const DEFAULT_RULE_NULL_CONTEXT_BIN_COUNT: usize =
    DEFAULT_RULE_NULL_BASE_COMPLEXITY_BIN_COUNT * DEFAULT_RULE_NULL_UNIT_GROUP_BIN_COUNT;
const DEFAULT_RULE_NULL_LEN_BUCKET_COUNT: usize = 3;

// ---------------------------------------------------------------------------
// Bucket types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleNullBucket {
    pub rule_len: usize,
    pub complexity_bin: u8,
}
const STRUCTURE_PRIOR_LEN_ALPHA: [f64; 5] = [16.0, 8.0, 4.0, 2.0, 1.0];
const STRUCTURE_PRIOR_TARGET_ESS: f64 = 24.0;
const STRUCTURE_PRIOR_LEN_TEMPER: f64 = 0.72;

#[derive(Clone, Debug, Default)]
struct RuleNullScores {
    train: Vec<f64>,
    test: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuleNullGlobalStats {
    pub mean: f64,
    pub sample_std: f64,
    pub n: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuleNullDistributionSummary {
    pub quantile: f64,
    pub penalty: f64,
    pub mean: f64,
    pub variance: f64,
    pub n: usize,
}

#[derive(Clone, Debug)]
pub struct RuleNullCalibrator {
    by_bucket: Vec<RuleNullScores>,
    by_group_len: Vec<RuleNullScores>,
    by_len: Vec<RuleNullScores>,
    max_rule_len: usize,
    global: RuleNullScores,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuleNullPenaltyLookup {
    bucket_train: Vec<Option<f64>>,
    bucket_test: Vec<Option<f64>>,
    group_len_train: Vec<Option<f64>>,
    group_len_test: Vec<Option<f64>>,
    group_len_train_stats: Vec<Option<RuleNullGlobalStats>>,
    group_len_test_stats: Vec<Option<RuleNullGlobalStats>>,
    len_train: Vec<Option<f64>>,
    len_test: Vec<Option<f64>>,
    len_train_stats: Vec<Option<RuleNullGlobalStats>>,
    len_test_stats: Vec<Option<RuleNullGlobalStats>>,
    max_rule_len: usize,
    quantile: f64,
    global_train: Option<f64>,
    global_test: Option<f64>,
    global_train_stats: Option<RuleNullGlobalStats>,
    global_test_stats: Option<RuleNullGlobalStats>,
}

impl Default for RuleNullPenaltyLookup {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuleStructurePrior {
    config: RuleStructurePriorConfig,
    len_probs: [f64; 6],
    strength: f64,
    best_log_mass: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuleStructurePriorConfig {
    len_alpha: [f64; 6],
    target_ess: f64,
    len_temper: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RuleStructurePriorCalibrator {
    len_counts: [f64; 6],
    score_samples: Vec<f64>,
}

impl RuleNullBucket {
    #[inline]
    fn len_index(self) -> usize {
        self.rule_len.saturating_sub(1)
    }

    #[inline]
    fn context_index(self) -> usize {
        usize::from(self.complexity_bin).min(DEFAULT_RULE_NULL_CONTEXT_BIN_COUNT - 1)
    }

    #[inline]
    pub fn unit_group_bin(self) -> u8 {
        (self.context_index() / DEFAULT_RULE_NULL_BASE_COMPLEXITY_BIN_COUNT) as u8
    }

    #[inline]
    fn bucket_index(self, max_rule_len: usize) -> usize {
        self.context_index()
            .saturating_mul(max_rule_len.max(1))
            .saturating_add(self.len_index().min(max_rule_len.saturating_sub(1)))
    }
}

// ---------------------------------------------------------------------------
// Bucket helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn null_topk_per_repeat_for_bucket(_bucket: RuleNullBucket) -> usize {
    DEFAULT_RULE_NULL_TOPK_ALL
}

#[inline]
fn sanitize_rule_null_quantile(quantile: f64) -> f64 {
    if quantile.is_finite() {
        quantile.clamp(0.0, 1.0)
    } else {
        DEFAULT_RULE_NULL_QUANTILE
    }
}

#[allow(dead_code)]
pub fn rule_null_bucket_count_exact() -> usize {
    DEFAULT_RULE_NULL_BUCKET_MAX_RULE_LEN
}
#[allow(dead_code)]
pub fn rule_null_bucket_count_sign_len() -> usize {
    0
}
#[allow(dead_code)]
pub fn rule_null_bucket_count_maf_len() -> usize {
    0
}

#[inline]
#[allow(dead_code)]
pub fn rule_null_bucket_count(max_rule_len: usize) -> usize {
    max_rule_len
        .max(1)
        .saturating_mul(DEFAULT_RULE_NULL_CONTEXT_BIN_COUNT)
}

#[inline]
pub(crate) fn rule_null_len_bucket_count(max_rule_len: usize) -> usize {
    max_rule_len.max(1).min(DEFAULT_RULE_NULL_LEN_BUCKET_COUNT)
}

#[inline]
pub(crate) fn rule_null_len_bucket_index(rule_len: usize, max_rule_len: usize) -> usize {
    let count = rule_null_len_bucket_count(max_rule_len);
    if count <= 1 {
        0
    } else if count == 2 {
        rule_len.saturating_sub(1).min(1)
    } else {
        rule_len.saturating_sub(1).min(2)
    }
}

#[inline]
pub fn rule_null_complexity_bin(n_features: usize) -> u8 {
    match n_features {
        0..=16 => 0,
        17..=32 => 1,
        33..=64 => 2,
        _ => 3,
    }
}

#[inline]
pub fn rule_null_unit_group_bin(unit_group_count: usize) -> u8 {
    match unit_group_count {
        0..=1 => 0,
        2 => 1,
        _ => 2,
    }
}

#[inline]
pub fn rule_null_context_bin(base_complexity_bin: u8, unit_group_bin: u8) -> u8 {
    let base =
        usize::from(base_complexity_bin).min(DEFAULT_RULE_NULL_BASE_COMPLEXITY_BIN_COUNT - 1);
    let group = usize::from(unit_group_bin).min(DEFAULT_RULE_NULL_UNIT_GROUP_BIN_COUNT - 1);
    (group * DEFAULT_RULE_NULL_BASE_COMPLEXITY_BIN_COUNT + base) as u8
}

#[inline]
pub(crate) fn rule_null_unit_group_bin_count() -> usize {
    DEFAULT_RULE_NULL_UNIT_GROUP_BIN_COUNT
}

#[inline]
pub(crate) fn rule_null_group_len_bucket_count(max_rule_len: usize) -> usize {
    rule_null_len_bucket_count(max_rule_len).saturating_mul(DEFAULT_RULE_NULL_UNIT_GROUP_BIN_COUNT)
}

#[inline]
pub(crate) fn rule_null_group_len_bucket_index(
    unit_group_bin: u8,
    rule_len: usize,
    max_rule_len: usize,
) -> usize {
    let len_count = rule_null_len_bucket_count(max_rule_len);
    usize::from(unit_group_bin)
        .min(DEFAULT_RULE_NULL_UNIT_GROUP_BIN_COUNT - 1)
        .saturating_mul(len_count)
        .saturating_add(rule_null_len_bucket_index(rule_len, max_rule_len))
}

impl RuleNullCalibrator {
    pub fn with_max_rule_len(max_rule_len: usize) -> Self {
        let bucket_count = rule_null_bucket_count(max_rule_len);
        Self {
            by_bucket: vec![RuleNullScores::default(); bucket_count],
            by_group_len: vec![
                RuleNullScores::default();
                rule_null_group_len_bucket_count(max_rule_len)
            ],
            by_len: vec![RuleNullScores::default(); rule_null_len_bucket_count(max_rule_len)],
            max_rule_len: max_rule_len.max(1),
            global: RuleNullScores::default(),
        }
    }

    pub fn new() -> Self {
        Self::with_max_rule_len(DEFAULT_RULE_NULL_BUCKET_MAX_RULE_LEN)
    }

    /// Push a paired (train, test) null score.  NaN-safe: finite-only values
    /// are forwarded to insert_train / insert_test.
    pub fn insert(&mut self, bucket: RuleNullBucket, train_score: f64, test_score: f64) {
        if train_score.is_finite() {
            self.insert_train(bucket, train_score);
        }
        if test_score.is_finite() {
            self.insert_test(bucket, test_score);
        }
    }

    /// Push a train-only null score without touching the test side.
    pub fn insert_train(&mut self, bucket: RuleNullBucket, score: f64) {
        if let Some(slot) = self
            .by_bucket
            .get_mut(bucket.bucket_index(self.max_rule_len))
        {
            slot.train.push(score);
        }
        if let Some(slot) = self.by_group_len.get_mut(rule_null_group_len_bucket_index(
            bucket.unit_group_bin(),
            bucket.rule_len,
            self.max_rule_len,
        )) {
            slot.train.push(score);
        }
        if let Some(slot) = self.by_len.get_mut(rule_null_len_bucket_index(
            bucket.rule_len,
            self.max_rule_len,
        )) {
            slot.train.push(score);
        }
        self.global.train.push(score);
    }

    /// Push a test-only null score without touching the train side.
    pub fn insert_test(&mut self, bucket: RuleNullBucket, score: f64) {
        if let Some(slot) = self
            .by_bucket
            .get_mut(bucket.bucket_index(self.max_rule_len))
        {
            slot.test.push(score);
        }
        if let Some(slot) = self.by_group_len.get_mut(rule_null_group_len_bucket_index(
            bucket.unit_group_bin(),
            bucket.rule_len,
            self.max_rule_len,
        )) {
            slot.test.push(score);
        }
        if let Some(slot) = self.by_len.get_mut(rule_null_len_bucket_index(
            bucket.rule_len,
            self.max_rule_len,
        )) {
            slot.test.push(score);
        }
        self.global.test.push(score);
    }

    pub fn finalize_with_quantile(&self, quantile: f64) -> RuleNullPenaltyLookup {
        let q = sanitize_rule_null_quantile(quantile);
        let mut out = RuleNullPenaltyLookup::with_max_rule_len(self.max_rule_len);
        out.quantile = q;
        for (idx, scores) in self.by_bucket.iter().enumerate() {
            out.bucket_train[idx] = sample_min_safe(scores.train.as_slice(), q);
            out.bucket_test[idx] = sample_min_safe(scores.test.as_slice(), q);
        }
        for (idx, scores) in self.by_group_len.iter().enumerate() {
            out.group_len_train[idx] = sample_min_safe(scores.train.as_slice(), q);
            out.group_len_test[idx] = sample_min_safe(scores.test.as_slice(), q);
            out.group_len_train_stats[idx] = summarize_scores(scores.train.as_slice());
            out.group_len_test_stats[idx] = summarize_scores(scores.test.as_slice());
        }
        for (idx, scores) in self.by_len.iter().enumerate() {
            out.len_train[idx] = sample_min_safe(scores.train.as_slice(), q);
            out.len_test[idx] = sample_min_safe(scores.test.as_slice(), q);
            out.len_train_stats[idx] = summarize_scores(scores.train.as_slice());
            out.len_test_stats[idx] = summarize_scores(scores.test.as_slice());
        }
        out.global_train = sample_min_safe(self.global.train.as_slice(), q);
        out.global_test = sample_min_safe(self.global.test.as_slice(), q);
        out.global_train_stats = summarize_scores(self.global.train.as_slice());
        out.global_test_stats = summarize_scores(self.global.test.as_slice());
        out
    }

    #[cfg(test)]
    pub fn finalize(&self) -> RuleNullPenaltyLookup {
        self.finalize_with_quantile(DEFAULT_RULE_NULL_QUANTILE)
    }
}

#[inline]
fn sample_min_safe(scores: &[f64], q: f64) -> Option<f64> {
    if scores.len() < NULL_EXACT_MIN_SAMPLES {
        return None;
    }
    quantile_nearest_rank(scores, q)
}

#[inline]
fn summarize_scores(scores: &[f64]) -> Option<RuleNullGlobalStats> {
    let mut n = 0usize;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    for value in scores.iter().copied().filter(|x| x.is_finite()) {
        n += 1;
        let delta = value - mean;
        mean += delta / (n as f64);
        let delta2 = value - mean;
        m2 += delta * delta2;
    }
    if n == 0 {
        return None;
    }
    let sample_std = if n >= 2 {
        (m2 / ((n - 1) as f64)).sqrt()
    } else {
        0.0
    };
    Some(RuleNullGlobalStats {
        mean,
        sample_std,
        n,
    })
}

#[inline]
fn one_sided_t_pvalue_greater(
    observed_score: f64,
    null_mean: f64,
    null_sample_std: f64,
    n: usize,
) -> Option<f64> {
    if !(observed_score.is_finite() && null_mean.is_finite() && null_sample_std.is_finite()) {
        return None;
    }
    if n == 0 {
        return None;
    }
    if n < 2 || !(null_sample_std > 0.0) {
        return Some(if observed_score > null_mean {
            f64::MIN_POSITIVE
        } else {
            1.0
        });
    }
    let se = null_sample_std / (n as f64).sqrt();
    if !(se > 0.0 && se.is_finite()) {
        return Some(if observed_score > null_mean {
            f64::MIN_POSITIVE
        } else {
            1.0
        });
    }
    let t = (observed_score - null_mean) / se;
    let two_sided = student_t_p_two_sided(t, (n - 1) as i32);
    if !two_sided.is_finite() {
        return None;
    }
    let p: f64 = if t >= 0.0 {
        0.5 * two_sided
    } else {
        1.0 - 0.5 * two_sided
    };
    Some(p.clamp(f64::MIN_POSITIVE, 1.0))
}

impl Default for RuleNullCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleNullPenaltyLookup {
    pub fn with_max_rule_len(max_rule_len: usize) -> Self {
        let bucket_count = rule_null_bucket_count(max_rule_len);
        Self {
            bucket_train: vec![None; bucket_count],
            bucket_test: vec![None; bucket_count],
            group_len_train: vec![None; rule_null_group_len_bucket_count(max_rule_len)],
            group_len_test: vec![None; rule_null_group_len_bucket_count(max_rule_len)],
            group_len_train_stats: vec![None; rule_null_group_len_bucket_count(max_rule_len)],
            group_len_test_stats: vec![None; rule_null_group_len_bucket_count(max_rule_len)],
            len_train: vec![None; rule_null_len_bucket_count(max_rule_len)],
            len_test: vec![None; rule_null_len_bucket_count(max_rule_len)],
            len_train_stats: vec![None; rule_null_len_bucket_count(max_rule_len)],
            len_test_stats: vec![None; rule_null_len_bucket_count(max_rule_len)],
            max_rule_len: max_rule_len.max(1),
            quantile: DEFAULT_RULE_NULL_QUANTILE,
            global_train: None,
            global_test: None,
            global_train_stats: None,
            global_test_stats: None,
        }
    }

    pub fn new() -> Self {
        Self::with_max_rule_len(DEFAULT_RULE_NULL_BUCKET_MAX_RULE_LEN)
    }

    #[cfg(test)]
    pub fn quantile(&self) -> f64 {
        self.quantile
    }

    fn penalty_with_fallback(&self, bucket: RuleNullBucket, is_train: bool) -> Option<f64> {
        let exact = if is_train {
            self.bucket_train
                .get(bucket.bucket_index(self.max_rule_len))
                .copied()
                .flatten()
        } else {
            self.bucket_test
                .get(bucket.bucket_index(self.max_rule_len))
                .copied()
                .flatten()
        };
        if exact.is_some() {
            return exact;
        }
        let group_len_idx = rule_null_group_len_bucket_index(
            bucket.unit_group_bin(),
            bucket.rule_len,
            self.max_rule_len,
        );
        let by_group_len = if is_train {
            self.group_len_train.get(group_len_idx).copied().flatten()
        } else {
            self.group_len_test.get(group_len_idx).copied().flatten()
        };
        if by_group_len.is_some() {
            return by_group_len;
        }
        let len_idx = rule_null_len_bucket_index(bucket.rule_len, self.max_rule_len);
        let by_len = if is_train {
            self.len_train.get(len_idx).copied().flatten()
        } else {
            self.len_test.get(len_idx).copied().flatten()
        };
        if by_len.is_some() {
            return by_len;
        }
        if is_train {
            self.global_train
        } else {
            self.global_test
        }
    }

    pub fn q99_converged_against(&self, prev: &Self) -> bool {
        let saw_signal = prev.global_train.is_some()
            || self.global_train.is_some()
            || prev.global_test.is_some()
            || self.global_test.is_some()
            || prev.bucket_train.iter().any(|x| x.is_some())
            || self.bucket_train.iter().any(|x| x.is_some())
            || prev.bucket_test.iter().any(|x| x.is_some())
            || self.bucket_test.iter().any(|x| x.is_some())
            || prev.group_len_train.iter().any(|x| x.is_some())
            || self.group_len_train.iter().any(|x| x.is_some())
            || prev.group_len_test.iter().any(|x| x.is_some())
            || self.group_len_test.iter().any(|x| x.is_some())
            || prev.len_train.iter().any(|x| x.is_some())
            || self.len_train.iter().any(|x| x.is_some())
            || prev.len_test.iter().any(|x| x.is_some())
            || self.len_test.iter().any(|x| x.is_some());
        let bucket_train_converged = self
            .bucket_train
            .iter()
            .zip(prev.bucket_train.iter())
            .all(|(curr, old)| penalty_value_converged(*old, *curr));
        let bucket_test_converged = self
            .bucket_test
            .iter()
            .zip(prev.bucket_test.iter())
            .all(|(curr, old)| penalty_value_converged(*old, *curr));
        let group_len_train_converged = self
            .group_len_train
            .iter()
            .zip(prev.group_len_train.iter())
            .all(|(curr, old)| penalty_value_converged(*old, *curr));
        let group_len_test_converged = self
            .group_len_test
            .iter()
            .zip(prev.group_len_test.iter())
            .all(|(curr, old)| penalty_value_converged(*old, *curr));
        let len_train_converged = self
            .len_train
            .iter()
            .zip(prev.len_train.iter())
            .all(|(curr, old)| penalty_value_converged(*old, *curr));
        let len_test_converged = self
            .len_test
            .iter()
            .zip(prev.len_test.iter())
            .all(|(curr, old)| penalty_value_converged(*old, *curr));
        saw_signal
            && bucket_train_converged
            && bucket_test_converged
            && group_len_train_converged
            && group_len_test_converged
            && len_train_converged
            && len_test_converged
            && penalty_value_converged(prev.global_train, self.global_train)
            && penalty_value_converged(prev.global_test, self.global_test)
    }

    pub fn has_signal(&self) -> bool {
        self.bucket_train.iter().any(|x| x.is_some())
            || self.bucket_test.iter().any(|x| x.is_some())
            || self.group_len_train.iter().any(|x| x.is_some())
            || self.group_len_test.iter().any(|x| x.is_some())
            || self.global_train.is_some()
            || self.global_test.is_some()
            || self.len_train.iter().any(|x| x.is_some())
            || self.len_test.iter().any(|x| x.is_some())
    }

    pub fn train_penalty(&self, bucket: RuleNullBucket) -> Option<f64> {
        self.penalty_with_fallback(bucket, true)
    }
    pub fn test_penalty(&self, bucket: RuleNullBucket) -> Option<f64> {
        self.penalty_with_fallback(bucket, false)
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub fn train_score_pvalue_greater(&self, observed_score: f64) -> Option<f64> {
        self.score_pvalue_greater(observed_score, true)
    }

    pub fn test_score_pvalue_greater(&self, observed_score: f64) -> Option<f64> {
        self.score_pvalue_greater(observed_score, false)
    }

    #[cfg(test)]
    pub fn summary(&self, is_train: bool) -> Option<RuleNullDistributionSummary> {
        let stats = if is_train {
            self.global_train_stats
        } else {
            self.global_test_stats
        }?;
        let penalty = if is_train {
            self.global_train
        } else {
            self.global_test
        }?;
        Some(RuleNullDistributionSummary {
            quantile: self.quantile,
            penalty,
            mean: stats.mean,
            variance: stats.sample_std * stats.sample_std,
            n: stats.n,
        })
    }

    pub fn len_bucket_count(&self) -> usize {
        self.len_train.len().max(self.len_test.len())
    }

    pub fn unit_group_bin_count(&self) -> usize {
        DEFAULT_RULE_NULL_UNIT_GROUP_BIN_COUNT
    }

    pub fn len_bucket_summary_by_index(
        &self,
        idx: usize,
        is_train: bool,
    ) -> Option<RuleNullDistributionSummary> {
        let (penalty, stats) = if is_train {
            (
                self.len_train.get(idx).copied().flatten(),
                self.len_train_stats.get(idx).copied().flatten(),
            )
        } else {
            (
                self.len_test.get(idx).copied().flatten(),
                self.len_test_stats.get(idx).copied().flatten(),
            )
        };
        let penalty = penalty?;
        let stats = stats?;
        Some(RuleNullDistributionSummary {
            quantile: self.quantile,
            penalty,
            mean: stats.mean,
            variance: stats.sample_std * stats.sample_std,
            n: stats.n,
        })
    }

    pub fn group_len_bucket_summary_by_index(
        &self,
        unit_group_bin: u8,
        len_idx: usize,
        is_train: bool,
    ) -> Option<RuleNullDistributionSummary> {
        let idx = rule_null_group_len_bucket_index(
            unit_group_bin,
            len_idx.saturating_add(1),
            self.max_rule_len,
        );
        let (penalty, stats) = if is_train {
            (
                self.group_len_train.get(idx).copied().flatten(),
                self.group_len_train_stats.get(idx).copied().flatten(),
            )
        } else {
            (
                self.group_len_test.get(idx).copied().flatten(),
                self.group_len_test_stats.get(idx).copied().flatten(),
            )
        };
        let penalty = penalty?;
        let stats = stats?;
        Some(RuleNullDistributionSummary {
            quantile: self.quantile,
            penalty,
            mean: stats.mean,
            variance: stats.sample_std * stats.sample_std,
            n: stats.n,
        })
    }

    fn score_pvalue_greater(&self, observed_score: f64, is_train: bool) -> Option<f64> {
        let stats = if is_train {
            self.global_train_stats
        } else {
            self.global_test_stats
        }?;
        let penalty = if is_train {
            self.global_train.unwrap_or(0.0)
        } else {
            self.global_test.unwrap_or(0.0)
        };
        one_sided_t_pvalue_greater(
            observed_score,
            stats.mean - penalty,
            stats.sample_std,
            stats.n,
        )
    }
}

#[cfg(test)]
pub fn bucket_from_rule(rule: &BeamRule, _maf: f64) -> RuleNullBucket {
    RuleNullBucket {
        rule_len: rule.len().max(1),
        complexity_bin: rule_null_context_bin(0, 0),
    }
}

#[cfg(test)]
pub fn bucket_from_expr(_expr: &str, rule_len: usize, _maf: f64) -> RuleNullBucket {
    RuleNullBucket {
        rule_len: rule_len.max(1),
        complexity_bin: rule_null_context_bin(0, 0),
    }
}

pub fn bucket_from_rule_with_complexity(
    rule: &BeamRule,
    _maf: f64,
    complexity_bin: u8,
) -> RuleNullBucket {
    RuleNullBucket {
        rule_len: rule.len().max(1),
        complexity_bin: complexity_bin.min((DEFAULT_RULE_NULL_CONTEXT_BIN_COUNT - 1) as u8),
    }
}

impl RuleStructurePriorCalibrator {
    pub fn observed_total(&self) -> f64 {
        self.len_counts.iter().skip(1).sum::<f64>()
    }

    pub fn observed_len_probs_preview(&self) -> Option<[f64; 6]> {
        let observed_total = self.observed_total();
        if !(observed_total.is_finite() && observed_total > 0.0) {
            return None;
        }
        let mut len_probs = [0.0_f64; 6];
        for len in 1..=5usize {
            len_probs[len] = self.len_counts[len] / observed_total.max(1e-12);
        }
        Some(len_probs)
    }

    fn effective_len_counts(&self, cfg: &RuleStructurePriorConfig) -> [f64; 6] {
        let observed_total = self.observed_total();
        let ess_scale = if observed_total > cfg.target_ess {
            cfg.target_ess / observed_total.max(1e-12)
        } else {
            1.0
        };
        let mut eff_len_counts = [0.0_f64; 6];
        for len in 1..=5usize {
            eff_len_counts[len] = self.len_counts[len] * ess_scale;
        }
        eff_len_counts
    }

    pub fn merge_from(&mut self, other: &Self) {
        for len in 1..=5usize {
            self.len_counts[len] += other.len_counts[len];
        }
        self.score_samples.extend(
            other
                .score_samples
                .iter()
                .copied()
                .filter(|x| x.is_finite()),
        );
    }

    pub fn insert(&mut self, rule_len: usize, _n_not: usize, score: f64, weight: f64) {
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }
        let len_bin = rule_len.clamp(1, 5);
        self.len_counts[len_bin] += weight;
        if score.is_finite() && score > 0.0 {
            self.score_samples.push(score);
        }
    }

    pub fn finalize(&self, cfg: &RuleStructurePriorConfig) -> RuleStructurePrior {
        let observed_total = self.observed_total();
        let len_probs = self.posterior_len_probs_preview(cfg);
        let score_scale = quantile_nearest_rank(self.score_samples.as_slice(), 0.75)
            .or_else(|| quantile_nearest_rank(self.score_samples.as_slice(), 0.50))
            .unwrap_or(0.02)
            .clamp(0.005, 0.05);
        let confidence = (observed_total.min(cfg.target_ess) / cfg.target_ess).sqrt();
        let strength = ((score_scale / 5.5) * (0.5 + 0.7 * confidence)).clamp(0.0015, 0.02);
        let mut best_log_mass = f64::NEG_INFINITY;
        for rule_len in 1..=5usize {
            let log_mass = structure_log_mass(len_probs.as_slice(), rule_len);
            if log_mass.is_finite() && log_mass > best_log_mass {
                best_log_mass = log_mass;
            }
        }
        RuleStructurePrior {
            config: cfg.clone(),
            len_probs,
            strength,
            best_log_mass,
        }
    }

    pub fn posterior_len_probs_preview(&self, cfg: &RuleStructurePriorConfig) -> [f64; 6] {
        let eff_len_counts = self.effective_len_counts(cfg);
        let mut len_probs = [0.0_f64; 6];
        let mut denom = 0.0_f64;
        for len in 1..=5usize {
            denom += cfg.len_alpha(len) + eff_len_counts[len];
        }
        for len in 1..=5usize {
            len_probs[len] = (cfg.len_alpha(len) + eff_len_counts[len]) / denom.max(1e-12);
        }
        let mut tempered_sum = 0.0_f64;
        for len in 1..=5usize {
            len_probs[len] = len_probs[len].powf(cfg.len_temper);
            tempered_sum += len_probs[len];
        }
        for len in 1..=5usize {
            len_probs[len] /= tempered_sum.max(1e-12);
        }
        len_probs
    }

    pub fn posterior_len_alpha_preview(&self, cfg: &RuleStructurePriorConfig) -> [f64; 6] {
        let eff_len_counts = self.effective_len_counts(cfg);
        let mut out = [0.0_f64; 6];
        for len in 1..=5usize {
            out[len] = cfg.len_alpha(len) + eff_len_counts[len];
        }
        out
    }
}

impl Default for RuleStructurePriorConfig {
    fn default() -> Self {
        Self::from_len_alpha_values(None)
    }
}

impl RuleStructurePriorConfig {
    pub fn from_len_alpha_values(values: Option<&[f64]>) -> Self {
        let mut len_alpha = [0.0_f64; 6];
        for (idx0, base_alpha) in STRUCTURE_PRIOR_LEN_ALPHA.iter().enumerate() {
            len_alpha[idx0 + 1] = (*base_alpha).max(1e-6);
        }
        if let Some(vs) = values {
            for (idx0, &value) in vs.iter().take(5).enumerate() {
                if value.is_finite() && value > 0.0 {
                    len_alpha[idx0 + 1] = value.max(1e-6);
                }
            }
        }
        Self {
            len_alpha,
            target_ess: STRUCTURE_PRIOR_TARGET_ESS,
            len_temper: STRUCTURE_PRIOR_LEN_TEMPER,
        }
    }

    pub fn len_alpha(&self, rule_len: usize) -> f64 {
        self.len_alpha[rule_len.clamp(1, 5)]
    }

    pub fn len_alpha_array(&self) -> [f64; 6] {
        self.len_alpha
    }

    pub fn target_ess(&self) -> f64 {
        self.target_ess
    }

    pub fn len_temper(&self) -> f64 {
        self.len_temper
    }
}

#[inline]
fn structure_log_mass(len_probs: &[f64], rule_len: usize) -> f64 {
    let len_bin = rule_len.clamp(1, 5);
    len_probs
        .get(len_bin)
        .copied()
        .unwrap_or(1e-12)
        .max(1e-12)
        .ln()
}

impl RuleStructurePrior {
    pub fn penalty(&self, rule_len: usize, _n_not: usize) -> f64 {
        let log_mass = structure_log_mass(self.len_probs.as_slice(), rule_len);
        ((self.best_log_mass - log_mass).max(0.0)) * self.strength
    }

    pub fn len_probs(&self) -> [f64; 6] {
        self.len_probs
    }

    pub fn config(&self) -> &RuleStructurePriorConfig {
        &self.config
    }

    pub fn len_prob(&self, rule_len: usize) -> f64 {
        self.len_probs[rule_len.clamp(1, 5)]
    }

    pub fn strength(&self) -> f64 {
        self.strength
    }

    pub fn best_log_mass(&self) -> f64 {
        self.best_log_mass
    }

    pub fn log_mass(&self, rule_len: usize) -> f64 {
        structure_log_mass(self.len_probs.as_slice(), rule_len)
    }

    pub fn mass(&self, rule_len: usize) -> f64 {
        self.log_mass(rule_len).exp()
    }
}

#[inline]
pub fn structure_prior_penalty(
    prior: Option<&RuleStructurePrior>,
    rule_len: usize,
    n_not: usize,
) -> f64 {
    prior.map(|p| p.penalty(rule_len, n_not)).unwrap_or(0.0)
}

fn quantile_nearest_rank(values: &[f64], quantile: f64) -> Option<f64> {
    let mut v = values
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect::<Vec<_>>();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let qq = quantile.clamp(0.0, 1.0);
    let idx = ((v.len() as f64) * qq).ceil() as usize;
    let idx = idx.saturating_sub(1).min(v.len() - 1);
    Some(v[idx])
}

#[inline]
fn penalty_value_converged(prev: Option<f64>, curr: Option<f64>) -> bool {
    match (prev, curr) {
        (None, None) => true,
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {
            let scale = a.abs().max(b.abs()).max(1.0);
            (a - b).abs() <= (DEFAULT_RULE_NULL_Q99_REL_TOL * scale)
        }
        _ => false,
    }
}

pub fn choose_representative_indices(region_sizes: &[usize], target_count: usize) -> Vec<usize> {
    if region_sizes.is_empty() || target_count == 0 {
        return Vec::new();
    }
    if target_count >= region_sizes.len() {
        return (0..region_sizes.len()).collect();
    }

    let mut order: Vec<usize> = (0..region_sizes.len()).collect();
    order.sort_by(|&a, &b| {
        region_sizes[a]
            .cmp(&region_sizes[b])
            .then_with(|| a.cmp(&b))
    });
    let mut picked = Vec::<usize>::with_capacity(target_count);
    for i in 0..target_count {
        let pos =
            ((((i as f64) + 0.5) * (order.len() as f64)) / (target_count as f64)).floor() as usize;
        picked.push(order[pos.min(order.len() - 1)]);
    }
    picked.sort_unstable();
    picked.dedup();
    picked
}

pub fn shuffled_copy_f64(values: &[f64], seed: u64) -> Vec<f64> {
    let mut out = values.to_vec();
    let mut rng = StdRng::seed_from_u64(seed);
    out.shuffle(&mut rng);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(rule_len: usize) -> RuleNullBucket {
        RuleNullBucket {
            rule_len,
            complexity_bin: 0,
        }
    }

    #[test]
    fn test_bucket_from_rule_new() {
        let rule = BeamRule {
            first: super::super::bs::BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![
                (
                    BeamBinaryOp::And,
                    super::super::bs::BeamLiteral {
                        row_index: 1,
                        group_id: 1,
                        negated: true,
                    },
                ),
                (
                    BeamBinaryOp::And,
                    super::super::bs::BeamLiteral {
                        row_index: 2,
                        group_id: 2,
                        negated: false,
                    },
                ),
            ],
        };
        let bk = bucket_from_rule(&rule, 0.25);
        assert_eq!(bk.rule_len, 3);
        let bk2 = bucket_from_rule(&rule, 0.01);
        assert_eq!(bk2.rule_len, 3);
    }

    #[test]
    fn test_bucket_from_expr_new() {
        let bk = bucket_from_expr("BIN(1) AND NOT BIN(2) AND NOT BIN(3)", 3, 0.01);
        assert_eq!(bk.rule_len, 3);

        let bk2 = bucket_from_expr("NOT BIN(1) AND NOT BIN(2)", 2, 0.20);
        assert_eq!(bk2.rule_len, 2);
    }

    #[test]
    fn test_bucket_singleton_and_pair() {
        let s = BeamRule {
            first: super::super::bs::BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![],
        };
        let sb = bucket_from_rule(&s, 0.10);
        assert_eq!(sb.rule_len, 1);
    }

    #[test]
    fn test_q99_nearest_rank() {
        let mut cal = RuleNullCalibrator::new();
        let bk = b(2);
        // Need >= NULL_EXACT_MIN_SAMPLES (10) for exact bucket to be used.
        for v in 1..=20 {
            cal.insert(bk, v as f64, v as f64);
        }
        cal.insert(bk, 1000.0, 1000.0);
        let lookup = cal.finalize();
        assert_eq!(lookup.train_penalty(bk).unwrap(), 1000.0);
    }

    #[test]
    fn test_penalty_uses_length_buckets_with_len3plus_collapsed() {
        let mut cal = RuleNullCalibrator::new();
        let len1 = b(1);
        let len2 = b(2);
        let len3 = b(3);
        let len4 = b(4);
        for v in 1..=20 {
            cal.insert(len1, v as f64, v as f64);
            cal.insert(len2, (100 + v) as f64, (100 + v) as f64);
            cal.insert(len3, (200 + v) as f64, (200 + v) as f64);
            cal.insert(len4, (300 + v) as f64, (300 + v) as f64);
        }
        let lookup = cal.finalize();
        assert_eq!(lookup.train_penalty(len1).unwrap(), 20.0);
        assert_eq!(lookup.train_penalty(len2).unwrap(), 120.0);
        assert_eq!(lookup.train_penalty(len3).unwrap(), 320.0);
        assert_eq!(lookup.train_penalty(len4).unwrap(), 320.0);
    }

    #[test]
    fn test_finalize_with_custom_quantile_changes_penalty() {
        let mut cal = RuleNullCalibrator::new();
        let bk = b(2);
        for v in 1..=20 {
            cal.insert(bk, v as f64, v as f64);
        }
        let q50 = cal.finalize_with_quantile(0.5);
        let q99 = cal.finalize_with_quantile(0.99);
        assert_eq!(q50.quantile(), 0.5);
        assert_eq!(q50.train_penalty(bk).unwrap(), 10.0);
        assert_eq!(q99.train_penalty(bk).unwrap(), 20.0);
    }

    #[test]
    fn test_test_score_pvalue_greater_is_monotonic() {
        let mut cal = RuleNullCalibrator::new();
        let bk = b(1);
        for v in 1..=20 {
            cal.insert(bk, v as f64, v as f64);
        }
        let lookup = cal.finalize_with_quantile(0.5);
        let p_low = lookup.test_score_pvalue_greater(8.0).unwrap();
        let p_mid = lookup.test_score_pvalue_greater(12.0).unwrap();
        let p_high = lookup.test_score_pvalue_greater(16.0).unwrap();
        assert!(p_low >= p_mid);
        assert!(p_mid >= p_high);
    }

    #[test]
    fn test_summary_returns_penalty_mean_variance_and_n() {
        let mut cal = RuleNullCalibrator::new();
        let bk = b(2);
        for v in 1..=20 {
            cal.insert(bk, v as f64, v as f64);
        }
        let summary = cal.finalize_with_quantile(0.5).summary(true).unwrap();
        assert_eq!(summary.quantile, 0.5);
        assert_eq!(summary.penalty, 10.0);
        assert!((summary.mean - 10.5).abs() < 1e-12);
        assert!((summary.variance - 35.0).abs() < 1e-12);
        assert_eq!(summary.n, 20);
    }

    #[test]
    fn test_len_bucket_summary_returns_penalty_mean_and_n() {
        let mut cal = RuleNullCalibrator::new();
        let len3 = b(3);
        let len4 = b(4);
        for v in 1..=20 {
            cal.insert(len3, (200 + v) as f64, (200 + v) as f64);
            cal.insert(len4, (300 + v) as f64, (300 + v) as f64);
        }
        let summary = cal.finalize().len_bucket_summary_by_index(2, true).unwrap();
        assert_eq!(summary.penalty, 320.0);
        assert!((summary.mean - 260.5).abs() < 1e-12);
        assert_eq!(summary.n, 40);
    }

    #[test]
    fn test_topk_values() {
        assert_eq!(null_topk_per_repeat_for_bucket(b(2)), 2);
        assert_eq!(null_topk_per_repeat_for_bucket(b(3)), 2);
        assert_eq!(null_topk_per_repeat_for_bucket(b(4)), 2);
        assert_eq!(null_topk_per_repeat_for_bucket(b(1)), 2);
    }

    #[test]
    fn test_fallback_to_global() {
        let mut lookup = RuleNullPenaltyLookup::new();
        let bk = b(2);
        lookup.global_train = Some(30.0);
        assert_eq!(lookup.train_penalty(bk).unwrap(), 30.0);
    }

    #[test]
    fn test_convergence_checks_global_only() {
        let mut cur = RuleNullPenaltyLookup::new();
        let mut prev = RuleNullPenaltyLookup::new();
        // Both empty → no signal → NOT converged
        assert!(!cur.q99_converged_against(&prev));
        // Set matching values → converged
        cur.global_train = Some(10.0);
        prev.global_train = Some(10.0);
        assert!(cur.q99_converged_against(&prev));
        // global mismatch → not converged
        cur.global_train = Some(100.0);
        assert!(!cur.q99_converged_against(&prev));
    }

    #[test]
    fn test_choose_representative_indices_spreads() {
        let sizes = vec![10, 20, 30, 40, 50, 60];
        let picked = choose_representative_indices(&sizes, 3);
        assert!(picked.len() >= 3 || picked.len() == sizes.len());
    }

    #[test]
    fn test_rule_null_bucket_count() {
        assert_eq!(
            rule_null_bucket_count(5),
            5 * DEFAULT_RULE_NULL_CONTEXT_BIN_COUNT
        );
    }

    #[test]
    fn test_group_len_penalty_stays_stratified_by_unit_group_bin() {
        let mut cal = RuleNullCalibrator::new();
        let w1_len2 = RuleNullBucket {
            rule_len: 2,
            complexity_bin: rule_null_context_bin(0, 0),
        };
        let w2_len2 = RuleNullBucket {
            rule_len: 2,
            complexity_bin: rule_null_context_bin(0, 1),
        };
        for v in 1..=20 {
            cal.insert(w1_len2, v as f64, v as f64);
            cal.insert(w2_len2, (100 + v) as f64, (100 + v) as f64);
        }
        let lookup = cal.finalize();
        assert_eq!(lookup.train_penalty(w1_len2).unwrap(), 20.0);
        assert_eq!(lookup.train_penalty(w2_len2).unwrap(), 120.0);
        let w2_summary = lookup
            .group_len_bucket_summary_by_index(1, 1, true)
            .unwrap();
        assert_eq!(w2_summary.penalty, 120.0);
        assert_eq!(w2_summary.n, 20);
    }

    #[test]
    fn test_structure_observed_len_probs_preview_uses_raw_counts() {
        let mut cal = RuleStructurePriorCalibrator::default();
        cal.len_counts[1] = 2.0;
        cal.len_counts[3] = 1.0;
        let probs = cal.observed_len_probs_preview().unwrap();
        assert!((probs[1] - (2.0 / 3.0)).abs() < 1e-12);
        assert!((probs[3] - (1.0 / 3.0)).abs() < 1e-12);
        assert_eq!(probs[2], 0.0);
    }
}
