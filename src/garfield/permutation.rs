use super::bs::{BeamBinaryOp, BeamLogicGateMode, BeamRule};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cmp::Ordering;

pub const DEFAULT_RULE_PERMUTATION_REPRESENTATIVE_UNITS: usize = 32;
pub const DEFAULT_RULE_NULL_PHYSICAL_CHUNKS: usize = 150;
pub const DEFAULT_RULE_NULL_MIN_SNPS_PER_CHUNK: usize = 50;
pub const DEFAULT_RULE_PERMUTATION_REPEATS: usize = 20;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MIN_REPEATS: usize = 5;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MAX_REPEATS: usize = 30;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_STABLE_REPEATS: usize = 3;
pub const DEFAULT_RULE_STRUCTURE_BOOTSTRAP_KL_THRESHOLD: f64 = 0.005;
pub const DEFAULT_RULE_STRUCTURE_DENSITY_TOPK: usize = 10;
const DEFAULT_RULE_NULL_QUANTILE: f64 = 0.99;
const PSEUDO_SNP_MAF_BOUNDARY: f64 = 0.05;
const STRUCTURE_PRIOR_LEN_ALPHA: [f64; 5] = [16.0, 8.0, 4.0, 2.0, 1.0];
const STRUCTURE_PRIOR_TARGET_ESS: f64 = 24.0;
const STRUCTURE_PRIOR_LEN_TEMPER: f64 = 0.72;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GateBucket {
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MafBucket {
    Low,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleNullBucket {
    pub gate: GateBucket,
    pub rule_len: usize,
    pub has_not: bool,
    pub maf: MafBucket,
}

#[derive(Clone, Debug, Default)]
struct RuleNullScores {
    train: Vec<f64>,
    test: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct RuleNullCalibrator {
    use_gate_dim: bool,
    exact: Vec<RuleNullScores>,
    collapsed_gate: Vec<RuleNullScores>,
    global: RuleNullScores,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuleNullPenaltyLookup {
    use_gate_dim: bool,
    exact_train: Vec<Option<f64>>,
    exact_test: Vec<Option<f64>>,
    collapsed_gate_train: Vec<Option<f64>>,
    collapsed_gate_test: Vec<Option<f64>>,
    global_train: Option<f64>,
    global_test: Option<f64>,
}

impl Default for RuleNullPenaltyLookup {
    fn default() -> Self {
        Self::new(false)
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

impl RuleNullScores {
    fn push(&mut self, train_score: f64, test_score: f64) {
        if train_score.is_finite() {
            self.train.push(train_score);
        }
        if test_score.is_finite() {
            self.test.push(test_score);
        }
    }
}

impl RuleNullBucket {
    #[inline]
    fn gate_bin(self) -> usize {
        match self.gate {
            GateBucket::And => 0,
            GateBucket::Or => 1,
        }
    }

    #[inline]
    fn maf_bin(self) -> usize {
        match self.maf {
            MafBucket::Low => 0,
            MafBucket::High => 1,
        }
    }

    #[inline]
    fn collapsed_gate_index(self) -> usize {
        ((((self.rule_len.clamp(1, 5) - 1) * 2) + usize::from(self.has_not)) * 2) + self.maf_bin()
    }

    #[inline]
    fn exact_index(self, use_gate_dim: bool) -> usize {
        let base = self.collapsed_gate_index();
        if use_gate_dim {
            (base * 2) + self.gate_bin()
        } else {
            base
        }
    }
}

#[inline]
pub fn rule_null_bucket_count(use_gate_dim: bool) -> usize {
    5usize * 2usize * 2usize * if use_gate_dim { 2usize } else { 1usize }
}

impl RuleNullCalibrator {
    pub fn new(use_gate_dim: bool) -> Self {
        Self {
            use_gate_dim,
            exact: vec![RuleNullScores::default(); rule_null_bucket_count(use_gate_dim)],
            collapsed_gate: vec![RuleNullScores::default(); rule_null_bucket_count(false)],
            global: RuleNullScores::default(),
        }
    }
}

impl Default for RuleNullCalibrator {
    fn default() -> Self {
        Self::new(false)
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

fn gate_bucket_from_ops(has_or: bool) -> GateBucket {
    if has_or {
        GateBucket::Or
    } else {
        GateBucket::And
    }
}

fn gate_bucket_for_mode(mode: BeamLogicGateMode, actual_gate: GateBucket) -> GateBucket {
    match mode {
        BeamLogicGateMode::AndOnly => GateBucket::And,
        BeamLogicGateMode::OrOnly => GateBucket::Or,
        BeamLogicGateMode::AndOr => actual_gate,
    }
}

fn gate_bucket_from_rule(rule: &BeamRule, logic_gate_mode: BeamLogicGateMode) -> GateBucket {
    let mut has_or = false;
    for (op, _) in rule.rest.iter() {
        match op {
            BeamBinaryOp::Or => has_or = true,
            BeamBinaryOp::And => {}
        }
    }
    gate_bucket_for_mode(logic_gate_mode, gate_bucket_from_ops(has_or))
}

fn gate_bucket_from_expr(
    expr: &str,
    _rule_len: usize,
    logic_gate_mode: BeamLogicGateMode,
) -> GateBucket {
    let upper = expr.to_ascii_uppercase();
    let has_or = upper.contains(" OR ");
    gate_bucket_for_mode(logic_gate_mode, gate_bucket_from_ops(has_or))
}

fn has_not_expr(expr: &str) -> bool {
    expr.to_ascii_uppercase().contains("NOT BIN(")
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

impl RuleNullCalibrator {
    pub fn insert(&mut self, bucket: RuleNullBucket, train_score: f64, test_score: f64) {
        self.exact[bucket.exact_index(self.use_gate_dim)].push(train_score, test_score);
        self.collapsed_gate[bucket.collapsed_gate_index()].push(train_score, test_score);
        self.global.push(train_score, test_score);
    }

    pub fn finalize(&self) -> RuleNullPenaltyLookup {
        let mut out = RuleNullPenaltyLookup::new(self.use_gate_dim);
        for (idx, scores) in self.exact.iter().enumerate() {
            out.exact_train[idx] =
                quantile_nearest_rank(scores.train.as_slice(), DEFAULT_RULE_NULL_QUANTILE);
            out.exact_test[idx] =
                quantile_nearest_rank(scores.test.as_slice(), DEFAULT_RULE_NULL_QUANTILE);
        }
        for (idx, scores) in self.collapsed_gate.iter().enumerate() {
            out.collapsed_gate_train[idx] =
                quantile_nearest_rank(scores.train.as_slice(), DEFAULT_RULE_NULL_QUANTILE);
            out.collapsed_gate_test[idx] =
                quantile_nearest_rank(scores.test.as_slice(), DEFAULT_RULE_NULL_QUANTILE);
        }
        out.global_train =
            quantile_nearest_rank(self.global.train.as_slice(), DEFAULT_RULE_NULL_QUANTILE);
        out.global_test =
            quantile_nearest_rank(self.global.test.as_slice(), DEFAULT_RULE_NULL_QUANTILE);
        out
    }
}

impl RuleNullPenaltyLookup {
    pub fn new(use_gate_dim: bool) -> Self {
        Self {
            use_gate_dim,
            exact_train: vec![None; rule_null_bucket_count(use_gate_dim)],
            exact_test: vec![None; rule_null_bucket_count(use_gate_dim)],
            collapsed_gate_train: vec![None; rule_null_bucket_count(false)],
            collapsed_gate_test: vec![None; rule_null_bucket_count(false)],
            global_train: None,
            global_test: None,
        }
    }

    fn penalty_with_fallback(&self, bucket: RuleNullBucket, is_train: bool) -> Option<f64> {
        let exact_idx = bucket.exact_index(self.use_gate_dim);
        let collapsed_idx = bucket.collapsed_gate_index();
        let exact = if is_train {
            self.exact_train.get(exact_idx).copied().flatten()
        } else {
            self.exact_test.get(exact_idx).copied().flatten()
        };
        if exact.is_some() {
            return exact;
        }
        let collapsed = if is_train {
            self.collapsed_gate_train
                .get(collapsed_idx)
                .copied()
                .flatten()
        } else {
            self.collapsed_gate_test
                .get(collapsed_idx)
                .copied()
                .flatten()
        };
        collapsed.or(if is_train {
            self.global_train
        } else {
            self.global_test
        })
    }

    pub fn train_penalty(&self, bucket: RuleNullBucket) -> Option<f64> {
        self.penalty_with_fallback(bucket, true)
    }

    pub fn test_penalty(&self, bucket: RuleNullBucket) -> Option<f64> {
        self.penalty_with_fallback(bucket, false)
    }
}

pub fn bucket_from_rule(
    rule: &BeamRule,
    support_frac: f64,
    logic_gate_mode: BeamLogicGateMode,
) -> RuleNullBucket {
    RuleNullBucket {
        gate: gate_bucket_from_rule(rule, logic_gate_mode),
        rule_len: rule.len().clamp(1, 5),
        has_not: rule.not_count() > 0,
        maf: maf_bucket_from_frac(support_frac),
    }
}

pub fn bucket_from_expr(
    expr: &str,
    rule_len: usize,
    support_frac: f64,
    logic_gate_mode: BeamLogicGateMode,
) -> RuleNullBucket {
    RuleNullBucket {
        gate: gate_bucket_from_expr(expr, rule_len, logic_gate_mode),
        rule_len: rule_len.clamp(1, 5),
        has_not: has_not_expr(expr),
        maf: maf_bucket_from_frac(support_frac),
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

    #[test]
    fn test_bucket_from_rule_counts_logic_and_not() {
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
                    BeamBinaryOp::Or,
                    super::super::bs::BeamLiteral {
                        row_index: 2,
                        group_id: 2,
                        negated: false,
                    },
                ),
            ],
        };
        let bucket = bucket_from_rule(&rule, 0.25, BeamLogicGateMode::AndOr);
        assert_eq!(bucket.gate, GateBucket::Or);
        assert_eq!(bucket.rule_len, 3);
        assert!(bucket.has_not);
        assert_eq!(bucket.maf, MafBucket::High);
    }

    #[test]
    fn test_bucket_from_expr_counts_not() {
        let bucket = bucket_from_expr(
            "BIN(1) AND NOT BIN(2) AND NOT BIN(3)",
            3,
            0.01,
            BeamLogicGateMode::AndOr,
        );
        assert_eq!(bucket.gate, GateBucket::And);
        assert_eq!(bucket.rule_len, 3);
        assert!(bucket.has_not);
        assert_eq!(bucket.maf, MafBucket::Low);
    }

    #[test]
    fn test_bucket_collapses_gate_when_logic_mode_is_or_only() {
        let singleton = BeamRule {
            first: super::super::bs::BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: Vec::new(),
        };
        let pair = BeamRule {
            first: super::super::bs::BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::Or,
                super::super::bs::BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        };
        let singleton_bucket = bucket_from_rule(&singleton, 0.10, BeamLogicGateMode::OrOnly);
        let pair_bucket = bucket_from_rule(&pair, 0.10, BeamLogicGateMode::OrOnly);
        assert_eq!(singleton_bucket.gate, GateBucket::Or);
        assert_eq!(pair_bucket.gate, GateBucket::Or);
    }

    #[test]
    fn test_choose_representative_indices_spreads_across_sizes() {
        let sizes = vec![10, 20, 30, 40, 50, 60];
        let picked = choose_representative_indices(&sizes, 3);
        assert!(picked.len() >= 3 || picked.len() == sizes.len());
    }

    #[test]
    fn test_q99_uses_nearest_rank() {
        let mut cal = RuleNullCalibrator::new(false);
        let bucket = RuleNullBucket {
            gate: GateBucket::And,
            rule_len: 2,
            has_not: false,
            maf: MafBucket::High,
        };
        for v in [1.0, 2.0, 3.0, 4.0, 100.0] {
            cal.insert(bucket, v, v);
        }
        let lookup = cal.finalize();
        assert_eq!(lookup.train_penalty(bucket).unwrap(), 100.0);
    }

    #[test]
    fn test_penalty_fallback_collapses_gate_dimension() {
        let mut cal = RuleNullCalibrator::new(true);
        let and_bucket = RuleNullBucket {
            gate: GateBucket::And,
            rule_len: 4,
            has_not: true,
            maf: MafBucket::Low,
        };
        let or_bucket = RuleNullBucket {
            gate: GateBucket::Or,
            rule_len: 4,
            has_not: true,
            maf: MafBucket::Low,
        };
        cal.insert(or_bucket, 10.0, 10.0);
        let lookup = cal.finalize();
        assert_eq!(lookup.train_penalty(and_bucket).unwrap(), 10.0);
    }

    #[test]
    fn test_rule_null_bucket_count_matches_20_or_40_layout() {
        assert_eq!(rule_null_bucket_count(false), 20);
        assert_eq!(rule_null_bucket_count(true), 40);
    }

    #[test]
    fn test_structure_observed_len_probs_preview_uses_raw_counts() {
        let mut cal = RuleStructurePriorCalibrator::default();
        cal.insert(1, 0, 1.0, 2.0);
        cal.insert(3, 0, 1.0, 1.0);
        let probs = cal.observed_len_probs_preview().unwrap();
        assert!((probs[1] - (2.0 / 3.0)).abs() < 1e-12);
        assert!((probs[3] - (1.0 / 3.0)).abs() < 1e-12);
        assert_eq!(probs[2], 0.0);
    }
}
