use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cmp::Ordering;

use super::bs::BeamRule;
#[cfg(test)]
use super::bs::BeamBinaryOp;

pub const DEFAULT_RULE_PERMUTATION_REPRESENTATIVE_UNITS: usize = 32;
pub const DEFAULT_RULE_NULL_PHYSICAL_CHUNKS: usize = 150;
pub const DEFAULT_RULE_NULL_MIN_SNPS_PER_CHUNK: usize = 50;
pub const DEFAULT_RULE_NULL_MAX_REPEATS: usize = 20;
pub const DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS: usize = 5;
pub const DEFAULT_RULE_NULL_ADAPTIVE_STABLE_REPEATS: usize = 2;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MIN_REPEATS: usize = 5;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MAX_REPEATS: usize = 30;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_STABLE_REPEATS: usize = 3;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_KL_THRESHOLD: f64 = 0.005;
pub const DEFAULT_RULE_STRUCTURE_DENSITY_TOPK: usize = 10;
const DEFAULT_RULE_NULL_QUANTILE: f64 = 0.99;
const DEFAULT_RULE_NULL_Q99_REL_TOL: f64 = 0.05;
// Minimum samples per exact bucket before falling back to collapsed.
const NULL_EXACT_MIN_SAMPLES: usize = 10;
// Top-k per repeat: how many extreme null scores to keep per bucket.
const DEFAULT_RULE_NULL_TOPK_POS_L2: usize = 3;
const DEFAULT_RULE_NULL_TOPK_POS_L3P: usize = 2;
const DEFAULT_RULE_NULL_TOPK_MIXED: usize = 2;
const DEFAULT_RULE_NULL_TOPK_NEG: usize = 2;
const DEFAULT_RULE_NULL_TOPK_DEFAULT: usize = 1;
// Shrinkage weights: (exact, collapse_1, collapse_2, global).
const DEFAULT_RULE_NULL_SHRINK_POS_L2: (f64, f64, f64, f64) = (0.20, 0.35, 0.30, 0.15);
const DEFAULT_RULE_NULL_SHRINK_POS_L3P: (f64, f64, f64, f64) = (0.20, 0.30, 0.35, 0.15);
const DEFAULT_RULE_NULL_SHRINK_MIXED: (f64, f64, f64, f64) = (0.15, 0.35, 0.35, 0.15);
const DEFAULT_RULE_NULL_SHRINK_NEG: (f64, f64, f64, f64) = (0.15, 0.35, 0.35, 0.15);
const PSEUDO_SNP_MAF_BOUNDARY: f64 = 0.05;

// ---------------------------------------------------------------------------
// Bucket types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MafBucket {
    Low,
    High,
}

/// Sign class: whether NOT literals appear in the rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CombMode {
    PosOnly, // not_count == 0
    Mixed,   // 0 < not_count < rule_len
    NegOnly, // not_count == rule_len (every literal negated)
}

/// Rule-length bin (collapsed to keep bucket count bounded).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LenBin {
    L1,  // rule_len == 1
    L2,  // rule_len == 2
    L3p, // rule_len >= 3
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleNullBucket {
    pub comb: CombMode,
    pub maf: MafBucket,
    pub len_bin: LenBin,
}

// Total: 3 (comb) × 2 (maf) × 3 (len_bin) = 18 slots.
//   rule_len=1 + Mixed is impossible → 2 empty slots, 16 effective.
const NULL_BUCKET_EXACT: usize = 18;
// collapse_sign_len: comb × len_bin
const NULL_BUCKET_SIGN_LEN: usize = 9;
// collapse_maf_len: maf × len_bin
const NULL_BUCKET_MAF_LEN: usize = 6;
const STRUCTURE_PRIOR_LEN_ALPHA: [f64; 5] = [16.0, 8.0, 4.0, 2.0, 1.0];
const STRUCTURE_PRIOR_TARGET_ESS: f64 = 24.0;
const STRUCTURE_PRIOR_LEN_TEMPER: f64 = 0.72;


#[derive(Clone, Debug, Default)]
struct RuleNullScores {
    train: Vec<f64>,
    test: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct RuleNullCalibrator {
    exact: Vec<RuleNullScores>,
    collapse_sign_len: Vec<RuleNullScores>,
    collapse_maf_len: Vec<RuleNullScores>,
    global: RuleNullScores,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuleNullPenaltyLookup {
    exact_train: Vec<Option<f64>>,
    exact_test: Vec<Option<f64>>,
    cs_train: Vec<Option<f64>>,
    cs_test: Vec<Option<f64>>,
    cm_train: Vec<Option<f64>>,
    cm_test: Vec<Option<f64>>,
    global_train: Option<f64>,
    global_test: Option<f64>,
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
    fn exact_index(self) -> usize {
        let comb = match self.comb {
            CombMode::PosOnly => 0usize,
            CombMode::Mixed => 1usize,
            CombMode::NegOnly => 2usize,
        };
        let maf = match self.maf {
            MafBucket::Low => 0usize,
            MafBucket::High => 1usize,
        };
        let lb = match self.len_bin {
            LenBin::L1 => 0usize,
            LenBin::L2 => 1usize,
            LenBin::L3p => 2usize,
        };
        (comb * 2 + maf) * 3 + lb
    }

    /// collapse_sign_len: comb × len_bin → 9 slots.
    #[inline]
    fn collapse_sign_len_index(self) -> usize {
        let comb = match self.comb {
            CombMode::PosOnly => 0usize,
            CombMode::Mixed => 1usize,
            CombMode::NegOnly => 2usize,
        };
        let lb = match self.len_bin {
            LenBin::L1 => 0usize,
            LenBin::L2 => 1usize,
            LenBin::L3p => 2usize,
        };
        comb * 3 + lb
    }

    /// collapse_maf_len: maf × len_bin → 6 slots.
    #[inline]
    fn collapse_maf_len_index(self) -> usize {
        let maf = match self.maf {
            MafBucket::Low => 0usize,
            MafBucket::High => 1usize,
        };
        let lb = match self.len_bin {
            LenBin::L1 => 0usize,
            LenBin::L2 => 1usize,
            LenBin::L3p => 2usize,
        };
        maf * 3 + lb
    }

    #[inline]
    fn from_exact_index(idx: usize) -> Self {
        let lb = match idx % 3 {
            0 => LenBin::L1,
            1 => LenBin::L2,
            _ => LenBin::L3p,
        };
        let rest = idx / 3;
        let maf = if (rest & 1) == 0 {
            MafBucket::Low
        } else {
            MafBucket::High
        };
        let comb = match rest / 2 {
            0 => CombMode::PosOnly,
            1 => CombMode::Mixed,
            _ => CombMode::NegOnly,
        };
        Self { comb, maf, len_bin: lb }
    }

    #[inline]
    fn from_collapse_sign_len_index(idx: usize) -> Self {
        let lb = match idx % 3 {
            0 => LenBin::L1,
            1 => LenBin::L2,
            _ => LenBin::L3p,
        };
        let comb = match idx / 3 {
            0 => CombMode::PosOnly,
            1 => CombMode::Mixed,
            _ => CombMode::NegOnly,
        };
        Self {
            comb,
            maf: MafBucket::Low,
            len_bin: lb,
        }
    }

    #[inline]
    fn from_collapse_maf_len_index(idx: usize) -> Self {
        let lb = match idx % 3 {
            0 => LenBin::L1,
            1 => LenBin::L2,
            _ => LenBin::L3p,
        };
        let maf = if idx / 3 == 0 {
            MafBucket::Low
        } else {
            MafBucket::High
        };
        Self {
            comb: CombMode::PosOnly,
            maf,
            len_bin: lb,
        }
    }
}

// ---------------------------------------------------------------------------
// Bucket helpers
// ---------------------------------------------------------------------------

#[inline]
fn len_bin_from_rule_len(rule_len: usize) -> LenBin {
    if rule_len <= 1 {
        LenBin::L1
    } else if rule_len == 2 {
        LenBin::L2
    } else {
        LenBin::L3p
    }
}

#[inline]
fn comb_mode_from_not_count(not_count: usize, rule_len: usize) -> CombMode {
    if not_count == 0 {
        CombMode::PosOnly
    } else if not_count >= rule_len {
        CombMode::NegOnly
    } else {
        CombMode::Mixed
    }
}

#[inline]
pub fn null_topk_per_repeat_for_bucket(bucket: RuleNullBucket) -> usize {
    match (bucket.comb, bucket.len_bin) {
        (CombMode::PosOnly, LenBin::L2) => DEFAULT_RULE_NULL_TOPK_POS_L2,
        (CombMode::PosOnly, LenBin::L3p) => DEFAULT_RULE_NULL_TOPK_POS_L3P,
        (CombMode::Mixed, _) => DEFAULT_RULE_NULL_TOPK_MIXED,
        (CombMode::NegOnly, _) => DEFAULT_RULE_NULL_TOPK_NEG,
        _ => DEFAULT_RULE_NULL_TOPK_DEFAULT,
    }
}

/// 4-component shrinkage weights: (exact, collapse_sign_len, collapse_maf_len, global).
#[inline]
fn null_shrinkage_weights(bucket: RuleNullBucket) -> Option<(f64, f64, f64, f64)> {
    if matches!(bucket.len_bin, LenBin::L1) {
        return None; // length-1 rules don't shrink
    }
    match bucket.comb {
        CombMode::PosOnly => {
            if matches!(bucket.len_bin, LenBin::L2) {
                Some(DEFAULT_RULE_NULL_SHRINK_POS_L2)
            } else {
                Some(DEFAULT_RULE_NULL_SHRINK_POS_L3P)
            }
        }
        CombMode::Mixed => Some(DEFAULT_RULE_NULL_SHRINK_MIXED),
        CombMode::NegOnly => Some(DEFAULT_RULE_NULL_SHRINK_NEG),
    }
}

#[inline]
fn null_quantile_for_bucket(bucket: RuleNullBucket) -> f64 {
    let _ = bucket;
    DEFAULT_RULE_NULL_QUANTILE
}

/// 4-component blended penalty.
#[inline]
fn blended_penalty_4(
    exact: Option<f64>,
    c1: Option<f64>,
    c2: Option<f64>,
    global: Option<f64>,
    weights: (f64, f64, f64, f64),
) -> Option<f64> {
    let mut numer = 0.0_f64;
    let mut denom = 0.0_f64;
    for (val, w) in &[
        (exact, weights.0),
        (c1, weights.1),
        (c2, weights.2),
        (global, weights.3),
    ] {
        if let Some(v) = val {
            if v.is_finite() {
                numer += v * w;
                denom += w;
            }
        }
    }
    if denom > 0.0 {
        Some(numer / denom)
    } else {
        None
    }
}

#[allow(dead_code)]
pub fn rule_null_bucket_count_exact() -> usize {
    NULL_BUCKET_EXACT
}
#[allow(dead_code)]
pub fn rule_null_bucket_count_sign_len() -> usize {
    NULL_BUCKET_SIGN_LEN
}
#[allow(dead_code)]
pub fn rule_null_bucket_count_maf_len() -> usize {
    NULL_BUCKET_MAF_LEN
}

/// Legacy wrapper — used by callers that still reference the old signature.
#[inline]
#[allow(dead_code)]
pub fn rule_null_bucket_count(_max_rule_len: usize) -> usize {
    NULL_BUCKET_EXACT
}

impl RuleNullCalibrator {
    pub fn with_max_rule_len(max_rule_len: usize) -> Self {
        let _mrl = max_rule_len.max(1);
        Self {
            exact: vec![RuleNullScores::default(); NULL_BUCKET_EXACT],
            collapse_sign_len: vec![RuleNullScores::default(); NULL_BUCKET_SIGN_LEN],
            collapse_maf_len: vec![RuleNullScores::default(); NULL_BUCKET_MAF_LEN],
            global: RuleNullScores::default(),
        }
    }

    pub fn new() -> Self {
        Self::with_max_rule_len(5)
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
        let ei = bucket.exact_index();
        self.exact[ei].train.push(score);
        let si = bucket.collapse_sign_len_index();
        self.collapse_sign_len[si].train.push(score);
        let mi = bucket.collapse_maf_len_index();
        self.collapse_maf_len[mi].train.push(score);
        self.global.train.push(score);
    }

    /// Push a test-only null score without touching the train side.
    pub fn insert_test(&mut self, bucket: RuleNullBucket, score: f64) {
        let ei = bucket.exact_index();
        self.exact[ei].test.push(score);
        let si = bucket.collapse_sign_len_index();
        self.collapse_sign_len[si].test.push(score);
        let mi = bucket.collapse_maf_len_index();
        self.collapse_maf_len[mi].test.push(score);
        self.global.test.push(score);
    }

    pub fn finalize(&self) -> RuleNullPenaltyLookup {
        let mut out = RuleNullPenaltyLookup::new();
        for (idx, scores) in self.exact.iter().enumerate() {
            let bucket = RuleNullBucket::from_exact_index(idx);
            let q = null_quantile_for_bucket(bucket);
            out.exact_train[idx] = sample_min_safe(scores.train.as_slice(), q);
            out.exact_test[idx] = sample_min_safe(scores.test.as_slice(), q);
        }
        for (idx, scores) in self.collapse_sign_len.iter().enumerate() {
            let bucket = RuleNullBucket::from_collapse_sign_len_index(idx);
            let q = null_quantile_for_bucket(bucket);
            out.cs_train[idx] = sample_min_safe(scores.train.as_slice(), q);
            out.cs_test[idx] = sample_min_safe(scores.test.as_slice(), q);
        }
        for (idx, scores) in self.collapse_maf_len.iter().enumerate() {
            let bucket = RuleNullBucket::from_collapse_maf_len_index(idx);
            let q = null_quantile_for_bucket(bucket);
            out.cm_train[idx] = sample_min_safe(scores.train.as_slice(), q);
            out.cm_test[idx] = sample_min_safe(scores.test.as_slice(), q);
        }
        out.global_train =
            sample_min_safe(self.global.train.as_slice(), DEFAULT_RULE_NULL_QUANTILE);
        out.global_test =
            sample_min_safe(self.global.test.as_slice(), DEFAULT_RULE_NULL_QUANTILE);
        out
    }
}

#[inline]
fn sample_min_safe(scores: &[f64], q: f64) -> Option<f64> {
    if scores.len() < NULL_EXACT_MIN_SAMPLES {
        return None;
    }
    quantile_nearest_rank(scores, q)
}

impl Default for RuleNullCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleNullPenaltyLookup {
    pub fn with_max_rule_len(_max_rule_len: usize) -> Self {
        Self {
            exact_train: vec![None; NULL_BUCKET_EXACT],
            exact_test: vec![None; NULL_BUCKET_EXACT],
            cs_train: vec![None; NULL_BUCKET_SIGN_LEN],
            cs_test: vec![None; NULL_BUCKET_SIGN_LEN],
            cm_train: vec![None; NULL_BUCKET_MAF_LEN],
            cm_test: vec![None; NULL_BUCKET_MAF_LEN],
            global_train: None,
            global_test: None,
        }
    }

    pub fn new() -> Self { Self::with_max_rule_len(5) }

    fn penalty_with_fallback(&self, bucket: RuleNullBucket, is_train: bool) -> Option<f64> {
        let exact = if is_train {
            self.exact_train.get(bucket.exact_index()).copied().flatten()
        } else {
            self.exact_test.get(bucket.exact_index()).copied().flatten()
        };
        let cs = if is_train {
            self.cs_train.get(bucket.collapse_sign_len_index()).copied().flatten()
        } else {
            self.cs_test.get(bucket.collapse_sign_len_index()).copied().flatten()
        };
        let cm = if is_train {
            self.cm_train.get(bucket.collapse_maf_len_index()).copied().flatten()
        } else {
            self.cm_test.get(bucket.collapse_maf_len_index()).copied().flatten()
        };
        let global = if is_train { self.global_train } else { self.global_test };
        if let Some(weights) = null_shrinkage_weights(bucket) {
            if let Some(ep) = exact {
                let shrunk = blended_penalty_4(Some(ep), cs, cm, global, weights)
                    .unwrap_or(ep);
                return Some(ep.min(shrunk));
            }
        }
        exact.or(cs).or(cm).or(global)
    }

    pub fn q99_converged_against(&self, prev: &Self) -> bool {
        let mut saw_signal = false;
        // collapse_sign_len (9 slots)
        for idx in 0..NULL_BUCKET_SIGN_LEN {
            let pt = prev.cs_train.get(idx).copied().flatten();
            let ct = self.cs_train.get(idx).copied().flatten();
            saw_signal |= pt.is_some() || ct.is_some();
            if !penalty_value_converged(pt, ct) { return false; }
            let pte = prev.cs_test.get(idx).copied().flatten();
            let cte = self.cs_test.get(idx).copied().flatten();
            saw_signal |= pte.is_some() || cte.is_some();
            if !penalty_value_converged(pte, cte) { return false; }
        }
        // collapse_maf_len (6 slots) — previously unchecked, causing
        // early-stop when maf×len was still fluctuating.
        for idx in 0..NULL_BUCKET_MAF_LEN {
            let pt = prev.cm_train.get(idx).copied().flatten();
            let ct = self.cm_train.get(idx).copied().flatten();
            saw_signal |= pt.is_some() || ct.is_some();
            if !penalty_value_converged(pt, ct) { return false; }
            let pte = prev.cm_test.get(idx).copied().flatten();
            let cte = self.cm_test.get(idx).copied().flatten();
            saw_signal |= pte.is_some() || cte.is_some();
            if !penalty_value_converged(pte, cte) { return false; }
        }
        saw_signal |= prev.global_train.is_some() || self.global_train.is_some()
            || prev.global_test.is_some() || self.global_test.is_some();
        saw_signal
            && penalty_value_converged(prev.global_train, self.global_train)
            && penalty_value_converged(prev.global_test, self.global_test)
    }

    pub fn train_penalty(&self, bucket: RuleNullBucket) -> Option<f64> {
        self.penalty_with_fallback(bucket, true)
    }
    pub fn test_penalty(&self, bucket: RuleNullBucket) -> Option<f64> {
        self.penalty_with_fallback(bucket, false)
    }
}

pub fn bucket_from_rule(rule: &BeamRule, support_frac: f64) -> RuleNullBucket {
    RuleNullBucket {
        comb: comb_mode_from_not_count(rule.not_count(), rule.len()),
        maf: maf_bucket_from_frac(support_frac),
        len_bin: len_bin_from_rule_len(rule.len()),
    }
}

pub fn bucket_from_expr(expr: &str, rule_len: usize, support_frac: f64) -> RuleNullBucket {
    // Count "NOT " followed by a word char — catches BIN, MBIN, and any
    // future label that literal_expr() may emit.  Clamp to rule_len so a
    // fully-negated rule is correctly classified as NegOnly.
    let upper = expr.to_ascii_uppercase();
    let nc = upper.matches("NOT ").count().min(rule_len);
    RuleNullBucket {
        comb: comb_mode_from_not_count(nc, rule_len),
        maf: maf_bucket_from_frac(support_frac),
        len_bin: len_bin_from_rule_len(rule_len),
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
#[inline]
fn support_minor_frac(support_frac: f64) -> f64 {
    if !support_frac.is_finite() {
        return 0.0;
    }
    let p = support_frac.clamp(0.0, 1.0);
    p.min(1.0 - p)
}

pub fn maf_bucket_from_frac(support_frac: f64) -> MafBucket {
    let minor = support_minor_frac(support_frac);
    if minor <= PSEUDO_SNP_MAF_BOUNDARY {
        MafBucket::Low
    } else {
        MafBucket::High
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

    fn b(comb: CombMode, lb: LenBin, maf: MafBucket) -> RuleNullBucket {
        RuleNullBucket { comb, len_bin: lb, maf }
    }

    #[test]
    fn test_bucket_from_rule_new() {
        let rule = BeamRule {
            first: super::super::bs::BeamLiteral { row_index: 0, group_id: 0, negated: false },
            rest: vec![
                (BeamBinaryOp::And, super::super::bs::BeamLiteral { row_index: 1, group_id: 1, negated: true }),
                (BeamBinaryOp::And, super::super::bs::BeamLiteral { row_index: 2, group_id: 2, negated: false }),
            ],
        };
        let bk = bucket_from_rule(&rule, 0.25);
        assert_eq!(bk.comb, CombMode::Mixed);
        assert_eq!(bk.len_bin, LenBin::L3p);
        assert_eq!(bk.maf, MafBucket::High);

        let bk2 = bucket_from_rule(&rule, 0.01);
        assert_eq!(bk2.maf, MafBucket::Low);
    }

    #[test]
    fn test_bucket_from_expr_new() {
        let bk = bucket_from_expr("BIN(1) AND NOT BIN(2) AND NOT BIN(3)", 3, 0.01);
        assert_eq!(bk.comb, CombMode::Mixed);
        assert_eq!(bk.len_bin, LenBin::L3p);
        assert_eq!(bk.maf, MafBucket::Low);

        let bk2 = bucket_from_expr("NOT BIN(1) AND NOT BIN(2)", 2, 0.20);
        assert_eq!(bk2.comb, CombMode::NegOnly);
        assert_eq!(bk2.len_bin, LenBin::L2);
    }

    #[test]
    fn test_bucket_singleton_and_pair() {
        let s = BeamRule {
            first: super::super::bs::BeamLiteral { row_index: 0, group_id: 0, negated: false },
            rest: vec![],
        };
        let sb = bucket_from_rule(&s, 0.10);
        assert_eq!(sb.comb, CombMode::PosOnly);
        assert_eq!(sb.len_bin, LenBin::L1);
    }

    #[test]
    fn test_q99_nearest_rank() {
        let mut cal = RuleNullCalibrator::new();
        let bk = b(CombMode::PosOnly, LenBin::L2, MafBucket::High);
        // Need >= NULL_EXACT_MIN_SAMPLES (10) for exact bucket to be used.
        for v in 1..=20 { cal.insert(bk, v as f64, v as f64); }
        cal.insert(bk, 1000.0, 1000.0);
        let lookup = cal.finalize();
        assert_eq!(lookup.train_penalty(bk).unwrap(), 1000.0);
    }

    #[test]
    fn test_pos_l2_and_neg_buckets() {
        let mut cal = RuleNullCalibrator::new();
        let pos = b(CombMode::PosOnly, LenBin::L2, MafBucket::High);
        let neg = b(CombMode::NegOnly, LenBin::L2, MafBucket::High);
        for v in 1..=40 {
            cal.insert(pos, v as f64, v as f64);
            cal.insert(neg, v as f64, v as f64);
        }
        let lookup = cal.finalize();
        assert_eq!(lookup.train_penalty(pos).unwrap(), 40.0);
        assert_eq!(lookup.train_penalty(neg).unwrap(), 40.0);
    }

    #[test]
    fn test_topk_values() {
        assert_eq!(null_topk_per_repeat_for_bucket(b(CombMode::PosOnly, LenBin::L2, MafBucket::High)), 3);
        assert_eq!(null_topk_per_repeat_for_bucket(b(CombMode::PosOnly, LenBin::L3p, MafBucket::High)), 2);
        assert_eq!(null_topk_per_repeat_for_bucket(b(CombMode::Mixed, LenBin::L2, MafBucket::High)), 2);
        assert_eq!(null_topk_per_repeat_for_bucket(b(CombMode::NegOnly, LenBin::L2, MafBucket::High)), 2);
        assert_eq!(null_topk_per_repeat_for_bucket(b(CombMode::PosOnly, LenBin::L1, MafBucket::High)), 1);
    }

    #[test]
    fn test_penalty_shrinks_toward_cs() {
        let mut lookup = RuleNullPenaltyLookup::new();
        let bk = b(CombMode::PosOnly, LenBin::L2, MafBucket::High);
        lookup.exact_train[bk.exact_index()] = Some(100.0);
        lookup.cs_train[bk.collapse_sign_len_index()] = Some(60.0);
        lookup.cm_train[bk.collapse_maf_len_index()] = Some(50.0);
        lookup.global_train = Some(20.0);
        let p = lookup.train_penalty(bk).unwrap();
        // 4-component blend: 0.20*100 + 0.35*60 + 0.30*50 + 0.15*20 = 59
        // min(100, 59) = 59
        assert!((p - 59.0).abs() < 1e-9, "got {p}");
    }

    #[test]
    fn test_neg_penalty_shrinks_to_blended() {
        let mut lookup = RuleNullPenaltyLookup::new();
        let bk = b(CombMode::NegOnly, LenBin::L2, MafBucket::High);
        lookup.exact_train[bk.exact_index()] = Some(100.0);
        lookup.cs_train[bk.collapse_sign_len_index()] = Some(60.0);
        lookup.cm_train[bk.collapse_maf_len_index()] = Some(50.0);
        lookup.global_train = Some(20.0);
        // NegOnly: 0.15*100 + 0.35*60 + 0.35*50 + 0.15*20 = 15+21+17.5+3 = 56.5
        // min(100, 56.5) = 56.5
        let p = lookup.train_penalty(bk).unwrap();
        assert!((p - 56.5).abs() < 1e-9, "got {p}");
    }

    #[test]
    fn test_fallback_to_cs_then_cm() {
        let mut lookup = RuleNullPenaltyLookup::new();
        let bk = b(CombMode::PosOnly, LenBin::L2, MafBucket::High);
        // exact is None, cs is set, cm is set → fallback to cs
        lookup.cs_train[bk.collapse_sign_len_index()] = Some(70.0);
        lookup.cm_train[bk.collapse_maf_len_index()] = Some(60.0);
        lookup.global_train = Some(30.0);
        assert_eq!(lookup.train_penalty(bk).unwrap(), 70.0);
    }

    #[test]
    fn test_convergence_checks_all_levels() {
        let mut cur = RuleNullPenaltyLookup::new();
        let mut prev = RuleNullPenaltyLookup::new();
        // Both empty → no signal → NOT converged
        assert!(!cur.q99_converged_against(&prev));
        // Set matching values → converged
        cur.cs_train[0] = Some(50.0);
        prev.cs_train[0] = Some(50.0);
        cur.global_train = Some(10.0);
        prev.global_train = Some(10.0);
        assert!(cur.q99_converged_against(&prev));
        // cm mismatches → not converged
        cur.cm_train[0] = Some(100.0);
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
        assert_eq!(rule_null_bucket_count(5), NULL_BUCKET_EXACT);
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
