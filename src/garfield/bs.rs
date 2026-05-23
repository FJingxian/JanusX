#![allow(dead_code)]

use super::permutation::{
    bucket_from_rule, structure_prior_penalty, GateBucket, RuleNullBucket, RuleNullPenaltyLookup,
    RuleStructurePrior,
};
use super::score::{
    score_cont_centered_gain_from_sum_and_n_hit, score_cont_centered_gain_packed_with_n_hit,
    score_cont_weighted_mean_diff_packed, sum_y_where_both1, support_size_packed,
    validate_continuous_y, ContinuousRuleScore,
};
use crate::bitwise::{and_popcount, bitand_assign, bitnot_masked, bitor_into};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

const GARFIELD_BEAM_PAR_MIN_TOTAL_CANDS: usize = 1_024;
const GARFIELD_BEAM_PAR_CHUNK_CANDS: usize = 128;
const GARFIELD_INITIAL_SINGLETON_NEGATIONS: [bool; 1] = [false];
const GARFIELD_AND_NOT_SHORTER_SUBRULE_GAIN_MAX: f64 = 0.08;
const GARFIELD_AND_NOT_SHORTER_SUBRULE_HAMMING_FRAC_MAX: f64 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BeamBinaryOp {
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BeamLogicGateMode {
    AndOnly,
    OrOnly,
    AndOr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BeamRankMode {
    Raw,
    InteractionGain,
    ExhaustiveThenGain,
    GainFromLayer(usize),
}

impl BeamLogicGateMode {
    #[inline]
    fn allowed_ops(self) -> &'static [BeamBinaryOp] {
        match self {
            BeamLogicGateMode::AndOnly => &[BeamBinaryOp::And],
            BeamLogicGateMode::OrOnly => &[BeamBinaryOp::Or],
            BeamLogicGateMode::AndOr => &[BeamBinaryOp::And, BeamBinaryOp::Or],
        }
    }

    #[inline]
    fn op_count(self) -> usize {
        self.allowed_ops().len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BeamLiteral {
    pub row_index: usize,
    pub group_id: usize,
    pub negated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeamRule {
    pub first: BeamLiteral,
    pub rest: Vec<(BeamBinaryOp, BeamLiteral)>,
}

impl BeamRule {
    #[inline]
    pub fn len(&self) -> usize {
        1 + self.rest.len()
    }

    #[inline]
    pub fn not_count(&self) -> usize {
        usize::from(self.first.negated)
            + self
                .rest
                .iter()
                .map(|(_, lit)| usize::from(lit.negated))
                .sum::<usize>()
    }

    #[inline]
    pub fn last_row_index(&self) -> usize {
        self.rest
            .last()
            .map(|(_, lit)| lit.row_index)
            .unwrap_or(self.first.row_index)
    }

    #[inline]
    pub fn uses_group(&self, group_id: usize) -> bool {
        if self.first.group_id == group_id {
            return true;
        }
        self.rest.iter().any(|(_, lit)| lit.group_id == group_id)
    }

    #[inline]
    fn lexical_key(&self) -> Vec<(usize, bool, u8)> {
        let mut out = Vec::with_capacity(self.len());
        out.push((self.first.row_index, self.first.negated, 0u8));
        for (op, lit) in self.rest.iter() {
            let op_code = match op {
                BeamBinaryOp::And => 1u8,
                BeamBinaryOp::Or => 2u8,
            };
            out.push((lit.row_index, lit.negated, op_code));
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BeamSearchParams {
    pub max_pick: usize,
    pub beam_width: usize,
    pub min_gain: f64,
    pub min_parent_abs_gain: f64,
    pub surrogate_test_gain_max: f64,
    pub surrogate_hamming_frac_max: f64,
    pub enable_diversity_pruning: bool,
    pub maf_threshold: f64,
    pub lambda_len: f64,
    pub lambda_not: f64,
    pub exhaustive_depth: usize,
    pub logic_gate_mode: BeamLogicGateMode,
    pub rank_mode: BeamRankMode,
    pub null_penalties: Option<Arc<RuleNullPenaltyLookup>>,
    pub structure_prior: Option<Arc<RuleStructurePrior>>,
    pub disable_parent_delta: bool,
    pub allow_parallel: bool,
}

impl Default for BeamSearchParams {
    fn default() -> Self {
        Self {
            max_pick: 3,
            beam_width: 5,
            min_gain: 0.0,
            min_parent_abs_gain: 0.0,
            surrogate_test_gain_max: 0.0,
            surrogate_hamming_frac_max: 0.0,
            enable_diversity_pruning: false,
            maf_threshold: 0.0,
            lambda_len: 0.0,
            lambda_not: 0.0,
            exhaustive_depth: 1,
            logic_gate_mode: BeamLogicGateMode::AndOr,
            rank_mode: BeamRankMode::InteractionGain,
            null_penalties: None,
            structure_prior: None,
            disable_parent_delta: false,
            allow_parallel: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BeamRuleCandidate {
    pub rule: BeamRule,
    pub train_score: f64,
    pub test_score: f64,
    pub train: ContinuousRuleScore,
    pub test: ContinuousRuleScore,
}

#[derive(Clone, Debug)]
struct BeamState {
    rule: BeamRule,
    combined_train: Vec<u64>,
    train: ContinuousRuleScore,
    train_abs_score: f64,
    train_score: f64,
    max_singleton_train_raw: f64,
    max_singleton_test_raw: f64,
}

#[derive(Clone, Copy, Debug)]
struct LiteralSingletonScore {
    train: ContinuousRuleScore,
    test: ContinuousRuleScore,
}

type RuleLexKey = Vec<(usize, bool, u8)>;
type RuleRawScoreCache = HashMap<RuleLexKey, f64>;
type RuleBitsCache = Vec<(BeamRule, Vec<u64>)>;

#[inline]
fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64).max(1)
}

#[inline]
fn tail_mask(n_samples: usize) -> Option<u64> {
    let rem = n_samples & 63;
    if rem == 0 {
        None
    } else {
        Some((1u64 << rem) - 1u64)
    }
}

#[inline]
fn apply_tail_mask(bits: &mut [u64], mask: Option<u64>) {
    if let Some(m) = mask {
        if let Some(last) = bits.last_mut() {
            *last &= m;
        }
    }
}

#[inline]
fn row_prefix<'a>(
    bits_flat: &'a [u64],
    row_words: usize,
    row_idx: usize,
    needed_words: usize,
) -> &'a [u64] {
    let st = row_idx * row_words;
    &bits_flat[st..st + needed_words]
}

#[inline]
fn score_key(s: f64) -> f64 {
    if s.is_nan() {
        f64::NEG_INFINITY
    } else {
        s
    }
}

#[inline]
fn penalty_for_rule(rule: &BeamRule, params: &BeamSearchParams) -> f64 {
    let len_pen = if rule.len() > 1 {
        params.lambda_len * ((rule.len() - 1) as f64)
    } else {
        0.0
    };
    let not_pen = params.lambda_not * (rule.not_count() as f64);
    len_pen + not_pen
}

#[inline]
fn rank_mode_uses_gain(rule_len: usize, params: &BeamSearchParams) -> bool {
    match params.rank_mode {
        BeamRankMode::Raw => false,
        BeamRankMode::InteractionGain => rule_len >= 2,
        BeamRankMode::ExhaustiveThenGain => rule_len > params.exhaustive_depth.max(1),
        BeamRankMode::GainFromLayer(start_layer) => rule_len >= start_layer.max(1),
    }
}

#[inline]
fn rule_is_pure_or(rule: &BeamRule) -> bool {
    !rule.rest.is_empty()
        && rule
            .rest
            .iter()
            .all(|(op, _)| matches!(op, BeamBinaryOp::Or))
}

#[inline]
fn rule_is_pure_and(rule: &BeamRule) -> bool {
    !rule.rest.is_empty()
        && rule
            .rest
            .iter()
            .all(|(op, _)| matches!(op, BeamBinaryOp::And))
}

#[inline]
fn rank_rule_score_components_with_gate(
    _gate: GateBucket,
    rule_len: usize,
    not_count: usize,
    raw_score: f64,
    direct_parent_raw: f64,
    params: &BeamSearchParams,
) -> f64 {
    let use_gain = rank_mode_uses_gain(rule_len, params);
    let base = if use_gain {
        raw_score - direct_parent_raw
    } else {
        raw_score
    };
    let len_pen = if rule_len > 1 {
        params.lambda_len * ((rule_len - 1) as f64)
    } else {
        0.0
    };
    let not_pen = params.lambda_not * (not_count as f64);
    let structure_pen =
        structure_prior_penalty(params.structure_prior.as_deref(), rule_len, not_count);
    base - len_pen - not_pen - structure_pen
}

#[inline]
pub fn rank_rule_score_components(
    rule_len: usize,
    not_count: usize,
    raw_score: f64,
    direct_parent_raw: f64,
    params: &BeamSearchParams,
) -> f64 {
    rank_rule_score_components_with_gate(
        GateBucket::And,
        rule_len,
        not_count,
        raw_score,
        direct_parent_raw,
        params,
    )
}

#[inline]
fn null_penalty_for_bucket(
    bucket: RuleNullBucket,
    params: &BeamSearchParams,
    is_train: bool,
) -> f64 {
    let Some(lookup) = params.null_penalties.as_ref() else {
        return 0.0;
    };
    if is_train {
        lookup.train_penalty(bucket).unwrap_or(0.0)
    } else {
        lookup.test_penalty(bucket).unwrap_or(0.0)
    }
}

#[inline]
pub fn rank_rule_score_components_with_bucket(
    bucket: RuleNullBucket,
    rule_len: usize,
    not_count: usize,
    raw_score: f64,
    direct_parent_raw: f64,
    params: &BeamSearchParams,
    is_train: bool,
) -> f64 {
    rank_rule_score_components_with_gate(
        bucket.gate,
        rule_len,
        not_count,
        raw_score,
        direct_parent_raw,
        params,
    ) - null_penalty_for_bucket(bucket, params, is_train)
}

#[inline]
fn use_parent_delta(rule_len: usize, params: &BeamSearchParams) -> bool {
    rank_mode_uses_gain(rule_len, params)
}

#[inline]
fn train_scores_for_rule(
    rule: &BeamRule,
    train_raw: ContinuousRuleScore,
    direct_parent_raw: f64,
    _parent_abs_score: Option<f64>,
    _parent_raw_score: Option<f64>,
    params: &BeamSearchParams,
) -> (f64, f64) {
    let bucket = bucket_from_rule(rule, train_raw.support_frac, params.logic_gate_mode);
    let gate = if rule_is_pure_or(rule) {
        GateBucket::Or
    } else {
        GateBucket::And
    };
    let abs_score = rank_rule_score_components_with_gate(
        gate,
        rule.len(),
        rule.not_count(),
        train_raw.raw_score,
        direct_parent_raw,
        params,
    );
    let threshold = null_penalty_for_bucket(bucket, params, true);
    let rank_score = abs_score - threshold;
    (abs_score, rank_score)
}

#[inline]
fn support_balance(sc: &ContinuousRuleScore) -> usize {
    sc.n_hit.min(sc.n_miss)
}

#[inline]
fn beam_support_threshold(rule_len: usize, n_samples: usize, params: &BeamSearchParams) -> usize {
    let _ = rule_len;
    if !(params.maf_threshold.is_finite() && params.maf_threshold > 0.0) || n_samples == 0 {
        0
    } else {
        (params.maf_threshold * (n_samples as f64)).floor() as usize
    }
}

#[inline]
fn support_count_for_beam_rule(
    rule: &BeamRule,
    sc: &ContinuousRuleScore,
    params: &BeamSearchParams,
) -> usize {
    match params.logic_gate_mode {
        BeamLogicGateMode::AndOnly => sc.n_hit,
        BeamLogicGateMode::OrOnly => sc.n_miss,
        BeamLogicGateMode::AndOr => {
            let _ = rule;
            support_balance(sc)
        }
    }
}

#[inline]
fn keep_rule_after_support_pruning(
    rule: &BeamRule,
    sc: &ContinuousRuleScore,
    n_samples: usize,
    params: &BeamSearchParams,
) -> bool {
    keep_rule_after_support_counts(sc.n_hit, sc.n_miss, rule.len(), n_samples, params)
}

#[inline]
fn keep_rule_after_support_counts(
    n_hit: usize,
    n_miss: usize,
    rule_len: usize,
    n_samples: usize,
    params: &BeamSearchParams,
) -> bool {
    if n_hit == 0 || n_miss == 0 {
        return false;
    }
    let threshold = beam_support_threshold(rule_len, n_samples, params);
    match params.logic_gate_mode {
        BeamLogicGateMode::AndOnly => n_hit > threshold,
        BeamLogicGateMode::OrOnly => n_miss > threshold,
        BeamLogicGateMode::AndOr => n_hit > threshold && n_miss > threshold,
    }
}

#[inline]
fn cmp_rule_lex(a: &BeamRule, b: &BeamRule) -> std::cmp::Ordering {
    a.lexical_key().cmp(&b.lexical_key())
}

#[inline]
fn child_rule_uses_blind_scan(parent_rule_len: usize) -> bool {
    parent_rule_len.saturating_add(1) >= 3
}

#[inline]
fn expansion_row_bounds(rule: &BeamRule, n_rows: usize) -> (usize, usize) {
    if child_rule_uses_blind_scan(rule.len()) {
        (0, n_rows)
    } else {
        (rule.last_row_index().saturating_add(1), n_rows)
    }
}

fn canonical_commutative_child_rule(
    parent: &BeamRule,
    op: BeamBinaryOp,
    literal: BeamLiteral,
) -> Option<BeamRule> {
    let canonical_op = if let Some((first_op, _)) = parent.rest.first() {
        if *first_op != op || !parent.rest.iter().all(|(rest_op, _)| *rest_op == *first_op) {
            return None;
        }
        *first_op
    } else {
        op
    };

    let mut lits = Vec::<BeamLiteral>::with_capacity(parent.len().saturating_add(1));
    lits.push(parent.first);
    lits.extend(parent.rest.iter().map(|(_, lit)| *lit));
    lits.push(literal);
    lits.sort_unstable();

    let first = *lits.first()?;
    let rest = lits
        .into_iter()
        .skip(1)
        .map(|lit| (canonical_op, lit))
        .collect::<Vec<_>>();
    Some(BeamRule { first, rest })
}

#[inline]
fn literal_score_index(row_index: usize, negated: bool) -> usize {
    row_index
        .saturating_mul(2)
        .saturating_add(usize::from(negated))
}

#[inline]
fn cmp_state(a: &BeamState, b: &BeamState) -> std::cmp::Ordering {
    let sa = score_key(a.train_score);
    let sb = score_key(b.train_score);
    match sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal) {
        std::cmp::Ordering::Equal => match a.rule.len().cmp(&b.rule.len()) {
            std::cmp::Ordering::Equal => {
                match support_balance(&b.train).cmp(&support_balance(&a.train)) {
                    std::cmp::Ordering::Equal => {
                        match a.rule.not_count().cmp(&b.rule.not_count()) {
                            std::cmp::Ordering::Equal => cmp_rule_lex(&a.rule, &b.rule),
                            other => other,
                        }
                    }
                    other => other,
                }
            }
            other => other,
        },
        other => other,
    }
}

#[inline]
fn cmp_candidate(a: &BeamRuleCandidate, b: &BeamRuleCandidate) -> std::cmp::Ordering {
    let sa = score_key(a.test_score);
    let sb = score_key(b.test_score);
    match sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal) {
        std::cmp::Ordering::Equal => match a.rule.len().cmp(&b.rule.len()) {
            std::cmp::Ordering::Equal => {
                match support_balance(&b.test).cmp(&support_balance(&a.test)) {
                    std::cmp::Ordering::Equal => {
                        let ta = score_key(a.train_score);
                        let tb = score_key(b.train_score);
                        match tb.partial_cmp(&ta).unwrap_or(std::cmp::Ordering::Equal) {
                            std::cmp::Ordering::Equal => {
                                match a.rule.not_count().cmp(&b.rule.not_count()) {
                                    std::cmp::Ordering::Equal => cmp_rule_lex(&a.rule, &b.rule),
                                    other => other,
                                }
                            }
                            other => other,
                        }
                    }
                    other => other,
                }
            }
            other => other,
        },
        other => other,
    }
}

#[inline]
fn push_top_k_states(nodes: &mut Vec<BeamState>, cand: BeamState, k: usize) {
    if k == 0 {
        return;
    }
    if nodes.len() < k {
        nodes.push(cand);
        return;
    }
    let mut worst_idx = 0usize;
    for i in 1..nodes.len() {
        if cmp_state(&nodes[i], &nodes[worst_idx]) == std::cmp::Ordering::Greater {
            worst_idx = i;
        }
    }
    if cmp_state(&cand, &nodes[worst_idx]) == std::cmp::Ordering::Less {
        nodes[worst_idx] = cand;
    }
}

#[inline]
fn state_scores_tied(a: f64, b: f64) -> bool {
    let aa = score_key(a);
    let bb = score_key(b);
    if aa == bb {
        return true;
    }
    if !aa.is_finite() || !bb.is_finite() {
        return false;
    }
    let scale = aa.abs().max(bb.abs()).max(1.0);
    (aa - bb).abs() <= 1e-12 * scale
}

#[inline]
fn score_strictly_improves(child_score: f64, parent_score: f64) -> bool {
    let child = score_key(child_score);
    let parent = score_key(parent_score);
    if !child.is_finite() {
        return false;
    }
    if !parent.is_finite() {
        return true;
    }
    child > parent && !state_scores_tied(child, parent)
}

#[inline]
fn score_sum_hit(sc: &ContinuousRuleScore) -> f64 {
    if sc.n_hit == 0 || !sc.mean_hit.is_finite() {
        0.0
    } else {
        sc.mean_hit * (sc.n_hit as f64)
    }
}

#[inline]
fn child_n_hit_from_intersection(
    parent_n_hit: usize,
    row_n_hit: usize,
    inter_n_hit: usize,
    n_samples: usize,
    op: BeamBinaryOp,
    negated: bool,
) -> usize {
    match (op, negated) {
        (BeamBinaryOp::And, false) => inter_n_hit,
        (BeamBinaryOp::And, true) => parent_n_hit.saturating_sub(inter_n_hit),
        (BeamBinaryOp::Or, false) => parent_n_hit + row_n_hit - inter_n_hit,
        (BeamBinaryOp::Or, true) => n_samples - row_n_hit + inter_n_hit,
    }
}

#[inline]
fn child_sum_hit_from_intersection(
    parent_sum_hit: f64,
    row_sum_hit: f64,
    inter_sum_hit: f64,
    total_sum: f64,
    op: BeamBinaryOp,
    negated: bool,
) -> f64 {
    match (op, negated) {
        (BeamBinaryOp::And, false) => inter_sum_hit,
        (BeamBinaryOp::And, true) => parent_sum_hit - inter_sum_hit,
        (BeamBinaryOp::Or, false) => parent_sum_hit + row_sum_hit - inter_sum_hit,
        (BeamBinaryOp::Or, true) => total_sum - row_sum_hit + inter_sum_hit,
    }
}

#[inline]
fn evaluate_child_train_from_parent_virtual(
    parent_bits: &[u64],
    parent_train: &ContinuousRuleScore,
    row: &[u64],
    row_train_pos: &ContinuousRuleScore,
    y_train: &[f64],
    sum_y_train: f64,
    n_train: usize,
    child_rule_len: usize,
    op: BeamBinaryOp,
    negated: bool,
    params: &BeamSearchParams,
) -> Option<ContinuousRuleScore> {
    let inter_n_hit = and_popcount(parent_bits, row) as usize;
    let child_n_hit = child_n_hit_from_intersection(
        parent_train.n_hit,
        row_train_pos.n_hit,
        inter_n_hit,
        n_train,
        op,
        negated,
    );
    let child_n_miss = n_train.saturating_sub(child_n_hit);
    if !keep_rule_after_support_counts(child_n_hit, child_n_miss, child_rule_len, n_train, params) {
        return None;
    }
    let inter_sum_hit = sum_y_where_both1(parent_bits, row, y_train, n_train);
    let child_sum_hit = child_sum_hit_from_intersection(
        score_sum_hit(parent_train),
        score_sum_hit(row_train_pos),
        inter_sum_hit,
        sum_y_train,
        op,
        negated,
    );
    Some(score_cont_centered_gain_from_sum_and_n_hit(
        sum_y_train,
        child_sum_hit,
        n_train,
        child_n_hit,
    ))
}

#[inline]
fn keep_child_after_parent_gain_pruning(
    child_rule: &BeamRule,
    child_rank_score: f64,
    params: &BeamSearchParams,
) -> bool {
    if !gain_threshold_applies(child_rule.len(), params) {
        return true;
    }
    score_strictly_improves(child_rank_score, 0.0)
}

#[inline]
fn keep_child_after_parent_abs_improvement_pruning(
    _parent_abs_score: f64,
    child_rule_len: usize,
    _child_abs_score: f64,
    _params: &BeamSearchParams,
) -> bool {
    let _ = child_rule_len;
    true
}

#[inline]
fn gain_threshold_applies(rule_len: usize, params: &BeamSearchParams) -> bool {
    rank_mode_uses_gain(rule_len, params)
}

#[inline]
fn keep_state_after_min_gain_pruning(
    rule_len: usize,
    train_score: f64,
    params: &BeamSearchParams,
) -> bool {
    if !(params.min_gain.is_finite() && params.min_gain > 0.0) {
        return true;
    }
    if !gain_threshold_applies(rule_len, params) {
        return true;
    }
    let gain = score_key(train_score);
    gain.is_finite() && gain > params.min_gain
}

#[inline]
fn rule_parent(rule: &BeamRule) -> Option<BeamRule> {
    if rule.rest.is_empty() {
        return None;
    }
    let mut parent = rule.clone();
    parent.rest.pop();
    Some(parent)
}

#[inline]
fn rule_without_literal(rule: &BeamRule, remove_idx: usize) -> Option<BeamRule> {
    if rule.len() <= 1 || remove_idx >= rule.len() {
        return None;
    }
    if remove_idx == 0 {
        let (_, first) = *rule.rest.first()?;
        return Some(BeamRule {
            first,
            rest: rule.rest.iter().skip(1).copied().collect(),
        });
    }
    let mut out = BeamRule {
        first: rule.first,
        rest: Vec::with_capacity(rule.rest.len().saturating_sub(1)),
    };
    for (rest_idx, &(op, lit)) in rule.rest.iter().enumerate() {
        if rest_idx + 1 == remove_idx {
            continue;
        }
        out.rest.push((op, lit));
    }
    Some(out)
}

#[inline]
fn rule_max_singleton_raw(
    rule: &BeamRule,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
) -> f64 {
    let mut best = f64::NEG_INFINITY;
    let first_idx = literal_score_index(rule.first.row_index, rule.first.negated);
    let first_score = if is_train {
        literal_scores[first_idx].train.raw_score
    } else {
        literal_scores[first_idx].test.raw_score
    };
    best = best.max(first_score);
    for (_, lit) in rule.rest.iter() {
        let idx = literal_score_index(lit.row_index, lit.negated);
        let score = if is_train {
            literal_scores[idx].train.raw_score
        } else {
            literal_scores[idx].test.raw_score
        };
        best = best.max(score);
    }
    best
}

#[inline]
fn collect_known_rule_raw_scores(states: &[BeamState]) -> RuleRawScoreCache {
    let mut out = RuleRawScoreCache::with_capacity(states.len());
    for state in states.iter() {
        out.insert(state.rule.lexical_key(), state.train.raw_score);
    }
    out
}

#[inline]
fn cache_rule_raw_score(cache: &mut RuleRawScoreCache, rule: &BeamRule, raw_score: f64) {
    cache.insert(rule.lexical_key(), raw_score);
}

#[inline]
fn ensure_rule_bits_cached(
    rule: &BeamRule,
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    local_cache: &mut RuleBitsCache,
) -> Result<(), String> {
    if local_cache.iter().any(|(cached_rule, _)| cached_rule == rule) {
        return Ok(());
    }
    let combined = materialize_rule_bits(rule, bits_flat, row_words, n_rows, n_samples)?;
    local_cache.push((rule.clone(), combined));
    Ok(())
}

#[inline]
fn cached_rule_bits<'a>(rule: &BeamRule, local_cache: &'a RuleBitsCache) -> Option<&'a [u64]> {
    local_cache
        .iter()
        .find(|(cached_rule, _)| cached_rule == rule)
        .map(|(_, bits)| bits.as_slice())
}

fn evaluate_rule_continuous_cached(
    rule: &BeamRule,
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    lambda_len: f64,
    lambda_not: f64,
    local_cache: &mut RuleBitsCache,
) -> Result<ContinuousRuleScore, String> {
    let ctx = "garfield::evaluate_rule_continuous_cached";
    validate_continuous_y(y, n_samples, ctx)?;
    ensure_rule_bits_cached(rule, bits_flat, row_words, n_rows, n_samples, local_cache)?;
    let combined = cached_rule_bits(rule, local_cache)
        .ok_or_else(|| format!("{ctx}: cached combined bits missing after materialization"))?;
    Ok(score_rule_continuous_from_bits(
        rule,
        y,
        combined,
        n_samples,
        lambda_len,
        lambda_not,
    ))
}

fn lookup_rule_raw_score_cached(
    rule: &BeamRule,
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    base_cache: Option<&RuleRawScoreCache>,
    local_cache: &mut RuleRawScoreCache,
) -> Result<f64, String> {
    let key = rule.lexical_key();
    if let Some(score) = local_cache.get(&key) {
        return Ok(*score);
    }
    if let Some(base) = base_cache {
        if let Some(score) = base.get(&key) {
            local_cache.insert(key, *score);
            return Ok(*score);
        }
    }
    let raw_score =
        evaluate_rule_continuous(rule, y, bits_flat, row_words, n_rows, n_samples, 0.0, 0.0)?
            .raw_score;
    local_cache.insert(key, raw_score);
    Ok(raw_score)
}

fn best_direct_parent_raw_baseline_cached(
    rule: &BeamRule,
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
    base_cache: Option<&RuleRawScoreCache>,
    local_cache: &mut RuleRawScoreCache,
) -> Result<f64, String> {
    if rule.len() <= 1 {
        return Ok(0.0);
    }
    if rule.len() == 2 {
        return Ok(rule_max_singleton_raw(rule, literal_scores, is_train));
    }
    let mut best = f64::NEG_INFINITY;
    for remove_idx in 0..rule.len() {
        let Some(parent_rule) = rule_without_literal(rule, remove_idx) else {
            continue;
        };
        let parent_raw = lookup_rule_raw_score_cached(
            &parent_rule,
            y,
            bits_flat,
            row_words,
            n_rows,
            n_samples,
            base_cache,
            local_cache,
        )?;
        best = best.max(parent_raw);
    }
    if best.is_finite() {
        Ok(best)
    } else {
        Ok(0.0)
    }
}

fn best_direct_parent_raw_baseline(
    rule: &BeamRule,
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
) -> Result<f64, String> {
    let mut local_cache = RuleRawScoreCache::new();
    best_direct_parent_raw_baseline_cached(
        rule,
        y,
        bits_flat,
        row_words,
        n_rows,
        n_samples,
        literal_scores,
        is_train,
        None,
        &mut local_cache,
    )
}

#[inline]
fn sort_truncate_states(mut nodes: Vec<BeamState>, k: usize) -> Vec<BeamState> {
    nodes.sort_by(cmp_state);
    if nodes.len() > k {
        let mut keep = k;
        let cutoff = nodes[keep - 1].train_score;
        while keep < nodes.len() && state_scores_tied(nodes[keep].train_score, cutoff) {
            keep += 1;
        }
        nodes.truncate(keep);
    }
    nodes
}

#[inline]
fn diversity_parent_key(rule: &BeamRule) -> Option<Vec<(usize, bool, u8)>> {
    if rule.rest.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(rule.len().saturating_sub(1));
    out.push((rule.first.row_index, rule.first.negated, 0u8));
    for (op, lit) in rule.rest.iter().take(rule.rest.len().saturating_sub(1)) {
        let op_code = match op {
            BeamBinaryOp::And => 1u8,
            BeamBinaryOp::Or => 2u8,
        };
        out.push((lit.row_index, lit.negated, op_code));
    }
    Some(out)
}

fn filter_beam_candidates(
    mut candidates: Vec<BeamState>,
    width: usize,
    params: &BeamSearchParams,
) -> Vec<BeamState> {
    if candidates.is_empty() {
        return candidates;
    }
    if params.min_gain.is_finite() && params.min_gain > 0.0 {
        candidates.retain(|state| {
            keep_state_after_min_gain_pruning(state.rule.len(), state.train_score, params)
        });
        if candidates.is_empty() {
            return candidates;
        }
    }
    if params.enable_diversity_pruning {
        candidates.sort_by(cmp_state);
        let mut seen = HashSet::<Vec<(usize, bool, u8)>>::with_capacity(candidates.len());
        let mut diversified = Vec::<BeamState>::with_capacity(candidates.len());
        for state in candidates.into_iter() {
            let Some(parent_key) = diversity_parent_key(&state.rule) else {
                diversified.push(state);
                continue;
            };
            if seen.insert(parent_key) {
                diversified.push(state);
            }
        }
        candidates = diversified;
        if candidates.is_empty() {
            return candidates;
        }
    }
    sort_truncate_states(candidates, width.max(1))
}

#[inline]
fn should_parallel(total_cands: usize, allow_parallel: bool) -> bool {
    allow_parallel
        && rayon::current_num_threads() > 1
        && total_cands >= GARFIELD_BEAM_PAR_MIN_TOTAL_CANDS
}

fn validate_bit_matrix(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    ctx: &str,
) -> Result<usize, String> {
    if n_rows == 0 {
        return Err(format!("{ctx}: n_rows must be > 0"));
    }
    if n_samples == 0 {
        return Err(format!("{ctx}: n_samples must be > 0"));
    }
    if row_words == 0 {
        return Err(format!("{ctx}: row_words must be > 0"));
    }
    let needed_words = words_for_samples(n_samples);
    if row_words < needed_words {
        return Err(format!(
            "{ctx}: row_words={} smaller than required {}",
            row_words, needed_words
        ));
    }
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows*row_words overflow"))?;
    if bits_flat.len() < total_words {
        return Err(format!(
            "{ctx}: bits length={} smaller than required {}",
            bits_flat.len(),
            total_words
        ));
    }
    Ok(needed_words)
}

fn validate_search_inputs(
    y_train: &[f64],
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    n_train: usize,
    y_test: &[f64],
    bits_test: &[u64],
    row_words_test: usize,
    n_test: usize,
    group_ids: &[usize],
    params: &BeamSearchParams,
) -> Result<(usize, usize), String> {
    let ctx = "garfield::beam_search_train_test_continuous";
    validate_continuous_y(y_train, n_train, ctx)?;
    validate_continuous_y(y_test, n_test, ctx)?;
    let need_train = validate_bit_matrix(bits_train, row_words_train, n_rows, n_train, ctx)?;
    let need_test = validate_bit_matrix(bits_test, row_words_test, n_rows, n_test, ctx)?;
    if group_ids.len() != n_rows {
        return Err(format!(
            "{ctx}: group_ids length mismatch: {} vs n_rows={}",
            group_ids.len(),
            n_rows
        ));
    }
    if params.max_pick == 0 {
        return Err(format!("{ctx}: max_pick must be > 0"));
    }
    if params.beam_width == 0 {
        return Err(format!("{ctx}: beam_width must be > 0"));
    }
    if !params.lambda_len.is_finite() || !params.lambda_not.is_finite() {
        return Err(format!("{ctx}: penalty parameters must be finite"));
    }
    Ok((need_train, need_test))
}

#[inline]
fn apply_first_literal(
    row: &[u64],
    needed_words: usize,
    n_samples: usize,
    negated: bool,
) -> Vec<u64> {
    let mut out = row[..needed_words].to_vec();
    if negated {
        bitnot_masked(&mut out, n_samples);
    } else {
        apply_tail_mask(&mut out, tail_mask(n_samples));
    }
    out
}

fn precompute_literal_singleton_scores(
    y_train: &[f64],
    sum_y_train: f64,
    bits_train: &[u64],
    row_words_train: usize,
    needed_words_train: usize,
    n_train: usize,
    y_test: &[f64],
    sum_y_test: f64,
    bits_test: &[u64],
    row_words_test: usize,
    needed_words_test: usize,
    n_test: usize,
    n_rows: usize,
) -> Vec<LiteralSingletonScore> {
    let mut out = Vec::<LiteralSingletonScore>::with_capacity(n_rows.saturating_mul(2));
    for row_idx in 0..n_rows {
        let row_train = row_prefix(bits_train, row_words_train, row_idx, needed_words_train);
        let row_test = row_prefix(bits_test, row_words_test, row_idx, needed_words_test);
        for &negated in &[false, true] {
            let train_bits = apply_first_literal(row_train, needed_words_train, n_train, negated);
            let test_bits = apply_first_literal(row_test, needed_words_test, n_test, negated);
            let train_n_hit = support_size_packed(&train_bits, n_train);
            let test_n_hit = support_size_packed(&test_bits, n_test);
            out.push(LiteralSingletonScore {
                train: score_cont_centered_gain_packed_with_n_hit(
                    y_train,
                    &train_bits,
                    n_train,
                    sum_y_train,
                    train_n_hit,
                ),
                test: score_cont_centered_gain_packed_with_n_hit(
                    y_test, &test_bits, n_test, sum_y_test, test_n_hit,
                ),
            });
        }
    }
    out
}

#[inline]
fn bitand_not_assign_masked(dst: &mut [u64], rhs: &[u64], n_valid_bits: usize) {
    debug_assert_eq!(dst.len(), rhs.len());
    let needed_words = words_for_samples(n_valid_bits);
    if needed_words == 0 {
        return;
    }
    let full_words = n_valid_bits >> 6;
    let rem = n_valid_bits & 63;
    for i in 0..full_words {
        dst[i] &= !rhs[i];
    }
    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        dst[full_words] &= (!rhs[full_words]) & mask;
    } else if full_words < needed_words {
        dst[full_words] &= !rhs[full_words];
    }
    if needed_words < dst.len() {
        for v in dst[needed_words..].iter_mut() {
            *v = 0u64;
        }
    }
}

#[inline]
fn bitor_not_into_masked(dst: &mut [u64], rhs: &[u64], n_valid_bits: usize) {
    debug_assert_eq!(dst.len(), rhs.len());
    let needed_words = words_for_samples(n_valid_bits);
    if needed_words == 0 {
        return;
    }
    let full_words = n_valid_bits >> 6;
    let rem = n_valid_bits & 63;
    for i in 0..full_words {
        dst[i] |= !rhs[i];
    }
    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        dst[full_words] |= (!rhs[full_words]) & mask;
    } else if full_words < needed_words {
        dst[full_words] |= !rhs[full_words];
    }
    apply_tail_mask(dst, tail_mask(n_valid_bits));
}

#[inline]
fn apply_literal_inplace(
    dst: &mut [u64],
    row: &[u64],
    op: BeamBinaryOp,
    negated: bool,
    n_samples: usize,
) {
    match (op, negated) {
        (BeamBinaryOp::And, false) => {
            bitand_assign(dst, row);
            apply_tail_mask(dst, tail_mask(n_samples));
        }
        (BeamBinaryOp::And, true) => bitand_not_assign_masked(dst, row, n_samples),
        (BeamBinaryOp::Or, false) => {
            bitor_into(dst, row);
            apply_tail_mask(dst, tail_mask(n_samples));
        }
        (BeamBinaryOp::Or, true) => bitor_not_into_masked(dst, row, n_samples),
    }
}

pub fn materialize_rule_bits(
    rule: &BeamRule,
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> Result<Vec<u64>, String> {
    let ctx = "garfield::materialize_rule_bits";
    let needed_words = validate_bit_matrix(bits_flat, row_words, n_rows, n_samples, ctx)?;
    if rule.first.row_index >= n_rows {
        return Err(format!(
            "{ctx}: first literal row index {} out of range for n_rows={}",
            rule.first.row_index, n_rows
        ));
    }
    let mut combined = apply_first_literal(
        row_prefix(bits_flat, row_words, rule.first.row_index, needed_words),
        needed_words,
        n_samples,
        rule.first.negated,
    );
    for (op, lit) in rule.rest.iter() {
        if lit.row_index >= n_rows {
            return Err(format!(
                "{ctx}: literal row index {} out of range for n_rows={}",
                lit.row_index, n_rows
            ));
        }
        let row = row_prefix(bits_flat, row_words, lit.row_index, needed_words);
        apply_literal_inplace(&mut combined, row, *op, lit.negated, n_samples);
    }
    Ok(combined)
}

#[inline]
fn score_rule_continuous_from_bits(
    rule: &BeamRule,
    y: &[f64],
    combined: &[u64],
    n_samples: usize,
    lambda_len: f64,
    lambda_not: f64,
) -> ContinuousRuleScore {
    let mut sc = score_cont_weighted_mean_diff_packed(y, combined, n_samples);
    let penalty = if rule.len() > 1 {
        lambda_len * ((rule.len() - 1) as f64)
    } else {
        0.0
    } + lambda_not * (rule.not_count() as f64);
    sc.score = sc.raw_score - penalty;
    sc
}

pub fn evaluate_rule_continuous(
    rule: &BeamRule,
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    lambda_len: f64,
    lambda_not: f64,
) -> Result<ContinuousRuleScore, String> {
    let ctx = "garfield::evaluate_rule_continuous";
    validate_continuous_y(y, n_samples, ctx)?;
    let combined = materialize_rule_bits(rule, bits_flat, row_words, n_rows, n_samples)?;
    Ok(score_rule_continuous_from_bits(
        rule,
        y,
        combined.as_slice(),
        n_samples,
        lambda_len,
        lambda_not,
    ))
}

fn build_initial_beam(
    _y_train: &[f64],
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Result<Vec<BeamState>, String> {
    let layer_cap = params.beam_width.min(n_rows);
    let total_cands = n_rows;
    let beam = if should_parallel(total_cands, params.allow_parallel) {
        let mut work = Vec::<(usize, usize)>::new();
        let chunk = GARFIELD_BEAM_PAR_CHUNK_CANDS.max(1);
        let mut start = 0usize;
        while start < n_rows {
            let end = (start + chunk).min(n_rows);
            work.push((start, end));
            start = end;
        }
        let local_tops: Vec<Vec<BeamState>> = work
            .into_par_iter()
            .map(|(start, end)| {
                let mut local = Vec::<BeamState>::with_capacity(layer_cap);
                for row_idx in start..end {
                    let row = row_prefix(bits_train, row_words_train, row_idx, needed_words_train);
                    // Layer 1 only seeds raw literals. `NOT SNP` is redundant for singleton
                    // search, but later layers may still append negated literals.
                    for &negated in GARFIELD_INITIAL_SINGLETON_NEGATIONS.iter() {
                        let literal = BeamLiteral {
                            row_index: row_idx,
                            group_id: group_ids[row_idx],
                            negated,
                        };
                        let rule = BeamRule {
                            first: literal,
                            rest: Vec::new(),
                        };
                        let combined =
                            apply_first_literal(row, needed_words_train, n_train, negated);
                        let single = literal_scores[literal_score_index(row_idx, negated)];
                        let train = single.train;
                        let (train_abs_score, train_score) = train_scores_for_rule(
                            &rule,
                            train,
                            train.raw_score,
                            None,
                            None,
                            params,
                        );
                        if !keep_rule_after_support_pruning(&rule, &train, n_train, params) {
                            continue;
                        }
                        if !keep_state_after_min_gain_pruning(rule.len(), train_score, params) {
                            continue;
                        }
                        push_top_k_states(
                            &mut local,
                            BeamState {
                                rule,
                                combined_train: combined,
                                train,
                                train_abs_score,
                                train_score,
                                max_singleton_train_raw: single.train.raw_score,
                                max_singleton_test_raw: single.test.raw_score,
                            },
                            layer_cap,
                        );
                    }
                }
                local
            })
            .collect();
        let mut merged = Vec::<BeamState>::with_capacity(layer_cap);
        for local in local_tops {
            for cand in local {
                push_top_k_states(&mut merged, cand, layer_cap);
            }
        }
        merged
    } else {
        let mut seq = Vec::<BeamState>::with_capacity(layer_cap);
        for row_idx in 0..n_rows {
            let row = row_prefix(bits_train, row_words_train, row_idx, needed_words_train);
            for &negated in GARFIELD_INITIAL_SINGLETON_NEGATIONS.iter() {
                let literal = BeamLiteral {
                    row_index: row_idx,
                    group_id: group_ids[row_idx],
                    negated,
                };
                let rule = BeamRule {
                    first: literal,
                    rest: Vec::new(),
                };
                let combined = apply_first_literal(row, needed_words_train, n_train, negated);
                let single = literal_scores[literal_score_index(row_idx, negated)];
                let train = single.train;
                let (train_abs_score, train_score) =
                    train_scores_for_rule(&rule, train, train.raw_score, None, None, params);
                if !keep_rule_after_support_pruning(&rule, &train, n_train, params) {
                    continue;
                }
                if !keep_state_after_min_gain_pruning(rule.len(), train_score, params) {
                    continue;
                }
                push_top_k_states(
                    &mut seq,
                    BeamState {
                        rule,
                        combined_train: combined,
                        train,
                        train_abs_score,
                        train_score,
                        max_singleton_train_raw: single.train.raw_score,
                        max_singleton_test_raw: single.test.raw_score,
                    },
                    layer_cap,
                );
            }
        }
        seq
    };
    if beam.is_empty() {
        return Err("garfield::build_initial_beam: no valid initial literals".to_string());
    }
    Ok(filter_beam_candidates(beam, layer_cap, params))
}

fn build_initial_states_exhaustive(
    _y_train: &[f64],
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Result<Vec<BeamState>, String> {
    let mut all = Vec::<BeamState>::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let row = row_prefix(bits_train, row_words_train, row_idx, needed_words_train);
        for &negated in GARFIELD_INITIAL_SINGLETON_NEGATIONS.iter() {
            let literal = BeamLiteral {
                row_index: row_idx,
                group_id: group_ids[row_idx],
                negated,
            };
            let rule = BeamRule {
                first: literal,
                rest: Vec::new(),
            };
            let combined = apply_first_literal(row, needed_words_train, n_train, negated);
            let single = literal_scores[literal_score_index(row_idx, negated)];
            let train = single.train;
            let (train_abs_score, train_score) =
                train_scores_for_rule(&rule, train, train.raw_score, None, None, params);
            if !keep_rule_after_support_pruning(&rule, &train, n_train, params) {
                continue;
            }
            if !keep_state_after_min_gain_pruning(rule.len(), train_score, params) {
                continue;
            }
            all.push(BeamState {
                rule,
                combined_train: combined,
                train,
                train_abs_score,
                train_score,
                max_singleton_train_raw: single.train.raw_score,
                max_singleton_test_raw: single.test.raw_score,
            });
        }
    }
    let out = dedup_states_by_train_bits(all);
    if out.is_empty() {
        return Err(
            "garfield::build_initial_states_exhaustive: no valid initial literals".to_string(),
        );
    }
    Ok(out)
}

fn expand_beam_once(
    beam: &[BeamState],
    y_train: &[f64],
    sum_y_train: f64,
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Result<Vec<BeamState>, String> {
    let next_cap = params.beam_width.min(n_rows.saturating_mul(4).max(1));
    let op_count = params.logic_gate_mode.op_count();
    let base_rule_raws = Arc::new(collect_known_rule_raw_scores(beam));
    let total_expand = beam
        .iter()
        .map(|node| {
            let (start, end) = expansion_row_bounds(&node.rule, n_rows);
            end.saturating_sub(start)
                .saturating_mul(2usize.saturating_mul(op_count))
        })
        .sum::<usize>();

    let next = if should_parallel(total_expand, params.allow_parallel) {
        let seen_commutative_children =
            Arc::new(Mutex::new(HashSet::<Vec<(usize, bool, u8)>>::new()));
        let mut work = Vec::<(usize, usize, usize)>::new();
        let chunk = GARFIELD_BEAM_PAR_CHUNK_CANDS.max(1);
        for (bi, node) in beam.iter().enumerate() {
            let (mut start, end_limit) = expansion_row_bounds(&node.rule, n_rows);
            while start < end_limit {
                let end = (start + chunk).min(end_limit);
                work.push((bi, start, end));
                start = end;
            }
        }
        let local_tops = work
            .into_par_iter()
            .map(|(bi, start, end)| {
                let node = &beam[bi];
                let mut local = Vec::<BeamState>::with_capacity(next_cap);
                let mut parent_raw_cache = RuleRawScoreCache::new();
                let blind_scan = child_rule_uses_blind_scan(node.rule.len());
                for cand in start..end {
                    let gid = group_ids[cand];
                    if node.rule.uses_group(gid) {
                        continue;
                    }
                    let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
                    let row_train_pos = literal_scores[literal_score_index(cand, false)].train;
                    for &op in params.logic_gate_mode.allowed_ops() {
                        for &negated in &[false, true] {
                            let literal = BeamLiteral {
                                row_index: cand,
                                group_id: gid,
                                negated,
                            };
                            let canonical_rule =
                                canonical_commutative_child_rule(&node.rule, op, literal);
                            if blind_scan {
                                if let Some(rule) = canonical_rule.as_ref() {
                                    let key = rule.lexical_key();
                                    let mut seen = seen_commutative_children.lock().map_err(|_| {
                                        "garfield::expand_beam_once: commutative dedup pool poisoned"
                                            .to_string()
                                    })?;
                                    if !seen.insert(key) {
                                        continue;
                                    }
                                }
                            }
                            let Some(train) = evaluate_child_train_from_parent_virtual(
                                &node.combined_train,
                                &node.train,
                                row,
                                &row_train_pos,
                                y_train,
                                sum_y_train,
                                n_train,
                                node.rule.len() + 1,
                                op,
                                negated,
                                params,
                            ) else {
                                continue;
                            };
                            let rule = if let Some(rule) = canonical_rule {
                                rule
                            } else {
                                let mut rule = node.rule.clone();
                                rule.rest.push((op, literal));
                                rule
                            };
                            let single = literal_scores[literal_score_index(cand, negated)];
                            let max_singleton_train_raw =
                                node.max_singleton_train_raw.max(single.train.raw_score);
                            let max_singleton_test_raw =
                                node.max_singleton_test_raw.max(single.test.raw_score);
                            let direct_parent_train_raw = if rule.len() == 2 {
                                node.train.raw_score.max(single.train.raw_score)
                            } else {
                                best_direct_parent_raw_baseline_cached(
                                    &rule,
                                    y_train,
                                    bits_train,
                                    row_words_train,
                                    n_rows,
                                    n_train,
                                    literal_scores,
                                    true,
                                    Some(base_rule_raws.as_ref()),
                                    &mut parent_raw_cache,
                                )?
                            };
                            let (train_abs_score, train_score) = train_scores_for_rule(
                                &rule,
                                train,
                                direct_parent_train_raw,
                                None,
                                None,
                                params,
                            );
                            if !keep_child_after_parent_abs_improvement_pruning(
                                node.train_abs_score,
                                rule.len(),
                                train_abs_score,
                                params,
                            ) {
                                continue;
                            }
                            if !keep_state_after_min_gain_pruning(rule.len(), train_score, params) {
                                continue;
                            }
                            if !keep_child_after_parent_gain_pruning(&rule, train_score, params) {
                                continue;
                            }
                            let mut combined = node.combined_train.clone();
                            apply_literal_inplace(&mut combined, row, op, negated, n_train);
                            push_top_k_states(
                                &mut local,
                                BeamState {
                                    rule,
                                    combined_train: combined,
                                    train,
                                    train_abs_score,
                                    train_score,
                                    max_singleton_train_raw,
                                    max_singleton_test_raw,
                                },
                                next_cap,
                            );
                        }
                    }
                }
                Ok(local)
            })
            .collect::<Vec<Result<Vec<BeamState>, String>>>();
        let mut merged = Vec::<BeamState>::with_capacity(next_cap);
        for local in local_tops {
            let local = local?;
            for cand in local {
                push_top_k_states(&mut merged, cand, next_cap);
            }
        }
        merged
    } else {
        let mut seen_commutative_children = HashSet::<Vec<(usize, bool, u8)>>::new();
        let mut seq = Vec::<BeamState>::with_capacity(next_cap);
        let mut parent_raw_cache = RuleRawScoreCache::new();
        for node in beam.iter() {
            let (start, end) = expansion_row_bounds(&node.rule, n_rows);
            let blind_scan = child_rule_uses_blind_scan(node.rule.len());
            for cand in start..end {
                let gid = group_ids[cand];
                if node.rule.uses_group(gid) {
                    continue;
                }
                let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
                let row_train_pos = literal_scores[literal_score_index(cand, false)].train;
                for &op in params.logic_gate_mode.allowed_ops() {
                    for &negated in &[false, true] {
                        let literal = BeamLiteral {
                            row_index: cand,
                            group_id: gid,
                            negated,
                        };
                        let canonical_rule =
                            canonical_commutative_child_rule(&node.rule, op, literal);
                        if blind_scan {
                            if let Some(rule) = canonical_rule.as_ref() {
                                if !seen_commutative_children.insert(rule.lexical_key()) {
                                    continue;
                                }
                            }
                        }
                        let Some(train) = evaluate_child_train_from_parent_virtual(
                            &node.combined_train,
                            &node.train,
                            row,
                            &row_train_pos,
                            y_train,
                            sum_y_train,
                            n_train,
                            node.rule.len() + 1,
                            op,
                            negated,
                            params,
                        ) else {
                            continue;
                        };
                        let rule = if let Some(rule) = canonical_rule {
                            rule
                        } else {
                            let mut rule = node.rule.clone();
                            rule.rest.push((op, literal));
                            rule
                        };
                        let single = literal_scores[literal_score_index(cand, negated)];
                        let max_singleton_train_raw =
                            node.max_singleton_train_raw.max(single.train.raw_score);
                        let max_singleton_test_raw =
                            node.max_singleton_test_raw.max(single.test.raw_score);
                        let direct_parent_train_raw = if rule.len() == 2 {
                            node.train.raw_score.max(single.train.raw_score)
                        } else {
                            best_direct_parent_raw_baseline_cached(
                                &rule,
                                y_train,
                                bits_train,
                                row_words_train,
                                n_rows,
                                n_train,
                                literal_scores,
                                true,
                                Some(base_rule_raws.as_ref()),
                                &mut parent_raw_cache,
                            )?
                        };
                        let (train_abs_score, train_score) = train_scores_for_rule(
                            &rule,
                            train,
                            direct_parent_train_raw,
                            None,
                            None,
                            params,
                        );
                        if !keep_child_after_parent_abs_improvement_pruning(
                            node.train_abs_score,
                            rule.len(),
                            train_abs_score,
                            params,
                        ) {
                            continue;
                        }
                        if !keep_state_after_min_gain_pruning(rule.len(), train_score, params) {
                            continue;
                        }
                        if !keep_child_after_parent_gain_pruning(&rule, train_score, params) {
                            continue;
                        }
                        let mut combined = node.combined_train.clone();
                        apply_literal_inplace(&mut combined, row, op, negated, n_train);
                        push_top_k_states(
                            &mut seq,
                            BeamState {
                                rule,
                                combined_train: combined,
                                train,
                                train_abs_score,
                                train_score,
                                max_singleton_train_raw,
                                max_singleton_test_raw,
                            },
                            next_cap,
                        );
                    }
                }
            }
        }
        seq
    };
    Ok(filter_beam_candidates(next, next_cap, params))
}

fn dedup_states_by_train_bits(states: Vec<BeamState>) -> Vec<BeamState> {
    let mut best = HashMap::<Vec<u64>, BeamState>::with_capacity(states.len());
    for state in states.into_iter() {
        match best.entry(state.combined_train.clone()) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(state);
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if cmp_state(&state, slot.get()) == std::cmp::Ordering::Less {
                    slot.insert(state);
                }
            }
        }
    }
    let mut out = best.into_values().collect::<Vec<_>>();
    out.sort_by(cmp_state);
    out
}

#[inline]
fn same_rule(a: &BeamRule, b: &BeamRule) -> bool {
    a == b
}

fn expand_states_exhaustive(
    frontier: &[BeamState],
    y_train: &[f64],
    sum_y_train: f64,
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Result<Vec<BeamState>, String> {
    let mut best = HashMap::<Vec<u64>, BeamState>::new();
    let base_rule_raws = collect_known_rule_raw_scores(frontier);
    let mut parent_raw_cache = RuleRawScoreCache::new();
    for node in frontier.iter() {
        for cand in (node.rule.last_row_index() + 1)..n_rows {
            let gid = group_ids[cand];
            if node.rule.uses_group(gid) {
                continue;
            }
            let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
            let row_train_pos = literal_scores[literal_score_index(cand, false)].train;
            for &op in params.logic_gate_mode.allowed_ops() {
                for &negated in &[false, true] {
                    let Some(train) = evaluate_child_train_from_parent_virtual(
                        &node.combined_train,
                        &node.train,
                        row,
                        &row_train_pos,
                        y_train,
                        sum_y_train,
                        n_train,
                        node.rule.len() + 1,
                        op,
                        negated,
                        params,
                    ) else {
                        continue;
                    };
                    let literal = BeamLiteral {
                        row_index: cand,
                        group_id: gid,
                        negated,
                    };
                    let mut rule = node.rule.clone();
                    rule.rest.push((op, literal));
                    let single = literal_scores[literal_score_index(cand, negated)];
                    let max_singleton_train_raw =
                        node.max_singleton_train_raw.max(single.train.raw_score);
                    let max_singleton_test_raw =
                        node.max_singleton_test_raw.max(single.test.raw_score);
                    let direct_parent_train_raw = if rule.len() == 2 {
                        node.train.raw_score.max(single.train.raw_score)
                    } else {
                        best_direct_parent_raw_baseline_cached(
                            &rule,
                            y_train,
                            bits_train,
                            row_words_train,
                            n_rows,
                            n_train,
                            literal_scores,
                            true,
                            Some(&base_rule_raws),
                            &mut parent_raw_cache,
                        )?
                    };
                    let (train_abs_score, train_score) = train_scores_for_rule(
                        &rule,
                        train,
                        direct_parent_train_raw,
                        None,
                        None,
                        params,
                    );
                    if !keep_child_after_parent_abs_improvement_pruning(
                        node.train_abs_score,
                        rule.len(),
                        train_abs_score,
                        params,
                    ) {
                        continue;
                    }
                    if !keep_state_after_min_gain_pruning(rule.len(), train_score, params) {
                        continue;
                    }
                    if !keep_child_after_parent_gain_pruning(&rule, train_score, params) {
                        continue;
                    }
                    let mut combined = node.combined_train.clone();
                    apply_literal_inplace(&mut combined, row, op, negated, n_train);
                    let state = BeamState {
                        rule,
                        combined_train: combined,
                        train,
                        train_abs_score,
                        train_score,
                        max_singleton_train_raw,
                        max_singleton_test_raw,
                    };
                    match best.entry(state.combined_train.clone()) {
                        std::collections::hash_map::Entry::Vacant(slot) => {
                            slot.insert(state);
                        }
                        std::collections::hash_map::Entry::Occupied(mut slot) => {
                            if cmp_state(&state, slot.get()) == std::cmp::Ordering::Less {
                                slot.insert(state);
                            }
                        }
                    }
                }
            }
        }
    }
    let out = best.into_values().collect::<Vec<_>>();
    let width = out.len().max(1);
    Ok(filter_beam_candidates(out, width, params))
}

#[allow(clippy::too_many_arguments)]
fn final_test_score_for_state(
    state: &BeamState,
    test: &ContinuousRuleScore,
    y_test: &[f64],
    bits_test: &[u64],
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Result<f64, String> {
    let child_bucket = bucket_from_rule(&state.rule, test.support_frac, params.logic_gate_mode);
    let child_gate = if rule_is_pure_or(&state.rule) {
        GateBucket::Or
    } else {
        GateBucket::And
    };
    let direct_parent_test_raw = best_direct_parent_raw_baseline(
        &state.rule,
        y_test,
        bits_test,
        row_words_test,
        n_rows,
        n_test,
        literal_scores,
        false,
    )?;
    let child_abs_score = rank_rule_score_components_with_gate(
        child_gate,
        state.rule.len(),
        state.rule.not_count(),
        test.raw_score,
        direct_parent_test_raw,
        params,
    );
    let threshold = null_penalty_for_bucket(child_bucket, params, false);
    Ok(child_abs_score - threshold)
}

#[inline]
fn rule_abs_score_for_eval(
    rule: &BeamRule,
    raw: &ContinuousRuleScore,
    y: &[f64],
    bits: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
    params: &BeamSearchParams,
) -> Result<f64, String> {
    let direct_parent_raw = best_direct_parent_raw_baseline(
        rule,
        y,
        bits,
        row_words,
        n_rows,
        n_samples,
        literal_scores,
        is_train,
    )?;
    let gate = if rule_is_pure_or(rule) {
        GateBucket::Or
    } else {
        GateBucket::And
    };
    Ok(rank_rule_score_components_with_gate(
        gate,
        rule.len(),
        rule.not_count(),
        raw.raw_score,
        direct_parent_raw,
        params,
    ))
}

#[inline]
fn rule_abs_score_for_eval_cached(
    rule: &BeamRule,
    raw: &ContinuousRuleScore,
    y: &[f64],
    bits: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
    params: &BeamSearchParams,
    base_cache: Option<&RuleRawScoreCache>,
    local_cache: &mut RuleRawScoreCache,
) -> Result<f64, String> {
    cache_rule_raw_score(local_cache, rule, raw.raw_score);
    let direct_parent_raw = best_direct_parent_raw_baseline_cached(
        rule,
        y,
        bits,
        row_words,
        n_rows,
        n_samples,
        literal_scores,
        is_train,
        base_cache,
        local_cache,
    )?;
    let gate = if rule_is_pure_or(rule) {
        GateBucket::Or
    } else {
        GateBucket::And
    };
    Ok(rank_rule_score_components_with_gate(
        gate,
        rule.len(),
        rule.not_count(),
        raw.raw_score,
        direct_parent_raw,
        params,
    ))
}

#[allow(clippy::too_many_arguments)]
fn final_rule_score_for_eval(
    rule: &BeamRule,
    raw: &ContinuousRuleScore,
    y: &[f64],
    bits: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
    is_train: bool,
) -> Result<f64, String> {
    let bucket = bucket_from_rule(rule, raw.support_frac, params.logic_gate_mode);
    let abs_score = rule_abs_score_for_eval(
        rule,
        raw,
        y,
        bits,
        row_words,
        n_rows,
        n_samples,
        literal_scores,
        is_train,
        params,
    )?;
    let threshold = null_penalty_for_bucket(bucket, params, is_train);
    Ok(abs_score - threshold)
}

#[allow(clippy::too_many_arguments)]
fn final_rule_score_for_eval_cached(
    rule: &BeamRule,
    raw: &ContinuousRuleScore,
    y: &[f64],
    bits: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
    is_train: bool,
    base_cache: Option<&RuleRawScoreCache>,
    local_cache: &mut RuleRawScoreCache,
) -> Result<f64, String> {
    let bucket = bucket_from_rule(rule, raw.support_frac, params.logic_gate_mode);
    let abs_score = rule_abs_score_for_eval_cached(
        rule,
        raw,
        y,
        bits,
        row_words,
        n_rows,
        n_samples,
        literal_scores,
        is_train,
        params,
        base_cache,
        local_cache,
    )?;
    let threshold = null_penalty_for_bucket(bucket, params, is_train);
    Ok(abs_score - threshold)
}

#[inline]
fn surrogate_collapse_enabled(params: &BeamSearchParams) -> bool {
    params.surrogate_test_gain_max.is_finite()
        && params.surrogate_test_gain_max > 0.0
        && params.surrogate_hamming_frac_max.is_finite()
        && params.surrogate_hamming_frac_max > 0.0
}

#[inline]
fn bit_hamming_fraction(a: &[u64], b: &[u64], n_bits: usize) -> f64 {
    if n_bits == 0 || a.len() != b.len() {
        return 1.0;
    }
    let diffs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as usize)
        .sum::<usize>();
    (diffs as f64) / (n_bits as f64)
}

#[inline]
fn surrogate_delta_small_enough(child_score: f64, parent_score: f64, max_gain: f64) -> bool {
    let child = score_key(child_score);
    let parent = score_key(parent_score);
    child.is_finite() && parent.is_finite() && (child - parent) <= max_gain
}

fn singleton_literal_map(rule: &BeamRule) -> Vec<(usize, usize)> {
    let mut out = Vec::<(usize, usize)>::with_capacity(rule.len());
    out.push((rule.first.row_index, rule.first.group_id));
    for (_, lit) in rule.rest.iter() {
        if !out.iter().any(|(row_idx, _)| *row_idx == lit.row_index) {
            out.push((lit.row_index, lit.group_id));
        }
    }
    out
}

fn pure_rule_literals(rule: &BeamRule) -> Option<(BeamBinaryOp, Vec<BeamLiteral>)> {
    let op = rule.rest.first().map(|(op, _)| *op)?;
    if !rule.rest.iter().all(|(rest_op, _)| *rest_op == op) {
        return None;
    }
    let mut lits = Vec::<BeamLiteral>::with_capacity(rule.len());
    lits.push(rule.first);
    lits.extend(rule.rest.iter().map(|(_, lit)| *lit));
    Some((op, lits))
}

fn drop_literal_from_pure_rule(rule: &BeamRule, drop_idx: usize) -> Option<BeamRule> {
    let (op, mut lits) = pure_rule_literals(rule)?;
    if lits.len() <= 1 || drop_idx >= lits.len() {
        return None;
    }
    lits.remove(drop_idx);
    let first = *lits.first()?;
    let rest = lits
        .into_iter()
        .skip(1)
        .map(|lit| (op, lit))
        .collect::<Vec<_>>();
    Some(BeamRule { first, rest })
}

#[inline]
fn shorter_subrule_surrogate_limits(
    current_rule: &BeamRule,
    subrule: &BeamRule,
    params: &BeamSearchParams,
) -> (f64, f64) {
    let mut max_gain = params.surrogate_test_gain_max;
    let mut max_hamming = params.surrogate_hamming_frac_max;
    if rule_is_pure_and(current_rule)
        && rule_is_pure_and(subrule)
        && current_rule.not_count() > subrule.not_count()
    {
        max_gain = max_gain.max(GARFIELD_AND_NOT_SHORTER_SUBRULE_GAIN_MAX);
        max_hamming = max_hamming.max(GARFIELD_AND_NOT_SHORTER_SUBRULE_HAMMING_FRAC_MAX);
    }
    (max_gain, max_hamming)
}

#[allow(clippy::too_many_arguments)]
fn collapse_surrogate_candidate(
    state: &BeamState,
    y_train: &[f64],
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    n_train: usize,
    y_test: &[f64],
    bits_test: &[u64],
    row_words_test: usize,
    n_test: usize,
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Result<BeamRuleCandidate, String> {
    let mut current_rule = state.rule.clone();
    let mut current_train = state.train;
    let mut current_train_score = state.train_score;
    let cache_capacity = current_rule.len().saturating_mul(16).max(16);
    let mut train_raw_cache = RuleRawScoreCache::with_capacity(cache_capacity);
    let mut test_raw_cache = RuleRawScoreCache::with_capacity(cache_capacity);
    let mut train_bits_cache = RuleBitsCache::with_capacity(cache_capacity);
    let mut test_bits_cache = RuleBitsCache::with_capacity(cache_capacity);
    cache_rule_raw_score(&mut train_raw_cache, &current_rule, current_train.raw_score);
    let mut current_test = evaluate_rule_continuous_cached(
        &current_rule,
        y_test,
        bits_test,
        row_words_test,
        n_rows,
        n_test,
        params.lambda_len,
        params.lambda_not,
        &mut test_bits_cache,
    )?;
    let mut current_test_score = final_rule_score_for_eval_cached(
        &current_rule,
        &current_test,
        y_test,
        bits_test,
        row_words_test,
        n_rows,
        n_test,
        literal_scores,
        params,
        false,
        None,
        &mut test_raw_cache,
    )?;

    if surrogate_collapse_enabled(params) && current_rule.len() > 1 {
        loop {
            let Some(parent_rule) = rule_parent(&current_rule) else {
                break;
            };
            ensure_rule_bits_cached(
                &current_rule,
                bits_test,
                row_words_test,
                n_rows,
                n_test,
                &mut test_bits_cache,
            )?;
            ensure_rule_bits_cached(
                &parent_rule,
                bits_test,
                row_words_test,
                n_rows,
                n_test,
                &mut test_bits_cache,
            )?;
            let current_bits_test = cached_rule_bits(&current_rule, &test_bits_cache)
                .ok_or_else(|| "current rule test bits cache miss".to_string())?;
            let parent_bits_test = cached_rule_bits(&parent_rule, &test_bits_cache)
                .ok_or_else(|| "parent rule test bits cache miss".to_string())?;
            let diff_frac = bit_hamming_fraction(
                current_bits_test,
                parent_bits_test,
                n_test,
            );
            if diff_frac > params.surrogate_hamming_frac_max {
                break;
            }
            let parent_test = evaluate_rule_continuous_cached(
                &parent_rule,
                y_test,
                bits_test,
                row_words_test,
                n_rows,
                n_test,
                params.lambda_len,
                params.lambda_not,
                &mut test_bits_cache,
            )?;
            let parent_test_score = final_rule_score_for_eval_cached(
                &parent_rule,
                &parent_test,
                y_test,
                bits_test,
                row_words_test,
                n_rows,
                n_test,
                literal_scores,
                params,
                false,
                None,
                &mut test_raw_cache,
            )?;
            if !surrogate_delta_small_enough(
                current_test_score,
                parent_test_score,
                params.surrogate_test_gain_max,
            ) {
                break;
            }
            let parent_train = evaluate_rule_continuous_cached(
                &parent_rule,
                y_train,
                bits_train,
                row_words_train,
                n_rows,
                n_train,
                params.lambda_len,
                params.lambda_not,
                &mut train_bits_cache,
            )?;
            let parent_train_score = final_rule_score_for_eval_cached(
                &parent_rule,
                &parent_train,
                y_train,
                bits_train,
                row_words_train,
                n_rows,
                n_train,
                literal_scores,
                params,
                true,
                None,
                &mut train_raw_cache,
            )?;
            current_rule = parent_rule;
            current_train = parent_train;
            current_train_score = parent_train_score;
            current_test = parent_test;
            current_test_score = parent_test_score;
            if current_rule.len() <= 1 {
                break;
            }
        }
        while current_rule.len() > 2
            && rule_is_pure_and(&current_rule)
            && current_rule.not_count() > 0
        {
            let mut best_subrule: Option<(BeamRuleCandidate, f64)> = None;
            for drop_idx in 0..current_rule.len() {
                let Some(subrule) = drop_literal_from_pure_rule(&current_rule, drop_idx) else {
                    continue;
                };
                let (gain_limit, hamming_limit) =
                    shorter_subrule_surrogate_limits(&current_rule, &subrule, params);
                if !(gain_limit.is_finite() && gain_limit > 0.0) {
                    continue;
                }
                if !(hamming_limit.is_finite() && hamming_limit > 0.0) {
                    continue;
                }
                ensure_rule_bits_cached(
                    &current_rule,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    &mut test_bits_cache,
                )?;
                ensure_rule_bits_cached(
                    &subrule,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    &mut test_bits_cache,
                )?;
                let current_bits_test = cached_rule_bits(&current_rule, &test_bits_cache)
                    .ok_or_else(|| "current rule test bits cache miss".to_string())?;
                let subrule_bits_test = cached_rule_bits(&subrule, &test_bits_cache)
                    .ok_or_else(|| "subrule test bits cache miss".to_string())?;
                let diff_frac = bit_hamming_fraction(
                    current_bits_test,
                    subrule_bits_test,
                    n_test,
                );
                if diff_frac > hamming_limit {
                    continue;
                }
                let subrule_test = evaluate_rule_continuous_cached(
                    &subrule,
                    y_test,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    params.lambda_len,
                    params.lambda_not,
                    &mut test_bits_cache,
                )?;
                let subrule_test_score = final_rule_score_for_eval_cached(
                    &subrule,
                    &subrule_test,
                    y_test,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    literal_scores,
                    params,
                    false,
                    None,
                    &mut test_raw_cache,
                )?;
                if !surrogate_delta_small_enough(current_test_score, subrule_test_score, gain_limit)
                {
                    continue;
                }
                let subrule_train = evaluate_rule_continuous_cached(
                    &subrule,
                    y_train,
                    bits_train,
                    row_words_train,
                    n_rows,
                    n_train,
                    params.lambda_len,
                    params.lambda_not,
                    &mut train_bits_cache,
                )?;
                let subrule_train_score = final_rule_score_for_eval_cached(
                    &subrule,
                    &subrule_train,
                    y_train,
                    bits_train,
                    row_words_train,
                    n_rows,
                    n_train,
                    literal_scores,
                    params,
                    true,
                    None,
                    &mut train_raw_cache,
                )?;
                let subrule_cand = BeamRuleCandidate {
                    rule: subrule,
                    train_score: subrule_train_score,
                    test_score: subrule_test_score,
                    train: subrule_train,
                    test: subrule_test,
                };
                match best_subrule.as_mut() {
                    Some((best_cand, best_diff)) => {
                        if cmp_candidate(&subrule_cand, best_cand) == std::cmp::Ordering::Less
                            || (cmp_candidate(&subrule_cand, best_cand)
                                == std::cmp::Ordering::Equal
                                && diff_frac < *best_diff)
                        {
                            *best_cand = subrule_cand;
                            *best_diff = diff_frac;
                        }
                    }
                    None => {
                        best_subrule = Some((subrule_cand, diff_frac));
                    }
                }
            }
            let Some((subrule_cand, _)) = best_subrule else {
                break;
            };
            current_rule = subrule_cand.rule;
            current_train = subrule_cand.train;
            current_train_score = subrule_cand.train_score;
            current_test = subrule_cand.test;
            current_test_score = subrule_cand.test_score;
        }
        if current_rule.len() > 1 {
            let mut best_singleton: Option<(BeamRuleCandidate, f64)> = None;
            for (row_index, group_id) in singleton_literal_map(&current_rule).into_iter() {
                let singleton_rule = BeamRule {
                    first: BeamLiteral {
                        row_index,
                        group_id,
                        negated: false,
                    },
                    rest: Vec::new(),
                };
                ensure_rule_bits_cached(
                    &current_rule,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    &mut test_bits_cache,
                )?;
                ensure_rule_bits_cached(
                    &singleton_rule,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    &mut test_bits_cache,
                )?;
                let current_bits_test = cached_rule_bits(&current_rule, &test_bits_cache)
                    .ok_or_else(|| "current rule test bits cache miss".to_string())?;
                let singleton_bits_test = cached_rule_bits(&singleton_rule, &test_bits_cache)
                    .ok_or_else(|| "singleton test bits cache miss".to_string())?;
                let diff_frac = bit_hamming_fraction(
                    current_bits_test,
                    singleton_bits_test,
                    n_test,
                );
                let orient_diff = diff_frac.min(1.0 - diff_frac);
                if orient_diff > params.surrogate_hamming_frac_max {
                    continue;
                }
                let singleton_test = evaluate_rule_continuous_cached(
                    &singleton_rule,
                    y_test,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    params.lambda_len,
                    params.lambda_not,
                    &mut test_bits_cache,
                )?;
                let singleton_test_score = final_rule_score_for_eval_cached(
                    &singleton_rule,
                    &singleton_test,
                    y_test,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    literal_scores,
                    params,
                    false,
                    None,
                    &mut test_raw_cache,
                )?;
                if !surrogate_delta_small_enough(
                    current_test_score,
                    singleton_test_score,
                    params.surrogate_test_gain_max,
                ) {
                    continue;
                }
                let singleton_train = evaluate_rule_continuous_cached(
                    &singleton_rule,
                    y_train,
                    bits_train,
                    row_words_train,
                    n_rows,
                    n_train,
                    params.lambda_len,
                    params.lambda_not,
                    &mut train_bits_cache,
                )?;
                let singleton_train_score = final_rule_score_for_eval_cached(
                    &singleton_rule,
                    &singleton_train,
                    y_train,
                    bits_train,
                    row_words_train,
                    n_rows,
                    n_train,
                    literal_scores,
                    params,
                    true,
                    None,
                    &mut train_raw_cache,
                )?;
                let singleton_cand = BeamRuleCandidate {
                    rule: singleton_rule,
                    train_score: singleton_train_score,
                    test_score: singleton_test_score,
                    train: singleton_train,
                    test: singleton_test,
                };
                match best_singleton.as_mut() {
                    Some((best_cand, best_diff)) => {
                        if cmp_candidate(&singleton_cand, best_cand) == std::cmp::Ordering::Less
                            || (cmp_candidate(&singleton_cand, best_cand)
                                == std::cmp::Ordering::Equal
                                && orient_diff < *best_diff)
                        {
                            *best_cand = singleton_cand;
                            *best_diff = orient_diff;
                        }
                    }
                    None => {
                        best_singleton = Some((singleton_cand, orient_diff));
                    }
                }
            }
            if let Some((singleton_cand, _)) = best_singleton {
                return Ok(singleton_cand);
            }
        }
    }

    Ok(BeamRuleCandidate {
        rule: current_rule,
        train_score: current_train_score,
        test_score: current_test_score,
        train: current_train,
        test: current_test,
    })
}

pub fn beam_search_train_test_continuous(
    y_train: &[f64],
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    n_train: usize,
    y_test: &[f64],
    bits_test: &[u64],
    row_words_test: usize,
    n_test: usize,
    group_ids: &[usize],
    params: BeamSearchParams,
) -> Result<Vec<BeamRuleCandidate>, String> {
    let (needed_words_train, needed_words_test) = validate_search_inputs(
        y_train,
        bits_train,
        row_words_train,
        n_rows,
        n_train,
        y_test,
        bits_test,
        row_words_test,
        n_test,
        group_ids,
        &params,
    )?;
    let sum_y_train = y_train.iter().take(n_train).copied().sum::<f64>();
    let sum_y_test = y_test.iter().take(n_test).copied().sum::<f64>();

    let max_depth = params.max_pick.min(n_rows);
    let exhaustive_depth = params.exhaustive_depth.max(1).min(max_depth);
    let literal_scores = precompute_literal_singleton_scores(
        y_train,
        sum_y_train,
        bits_train,
        row_words_train,
        needed_words_train,
        n_train,
        y_test,
        sum_y_test,
        bits_test,
        row_words_test,
        needed_words_test,
        n_test,
        n_rows,
    );

    let mut kept_all = Vec::<BeamState>::new();
    let mut beam = if exhaustive_depth > 1 {
        let exhaustive_initial = build_initial_states_exhaustive(
            y_train,
            bits_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
            group_ids,
            literal_scores.as_slice(),
            &params,
        )?;
        kept_all.extend(exhaustive_initial.iter().cloned());
        let mut frontier = exhaustive_initial;
        for _depth in 2..=exhaustive_depth {
            let next = expand_states_exhaustive(
                frontier.as_slice(),
                y_train,
                sum_y_train,
                bits_train,
                row_words_train,
                n_rows,
                needed_words_train,
                n_train,
                group_ids,
                literal_scores.as_slice(),
                &params,
            )?;
            if next.is_empty() {
                frontier = Vec::new();
                break;
            }
            kept_all.extend(next.iter().cloned());
            frontier = next;
        }
        sort_truncate_states(frontier, params.beam_width.max(1))
    } else {
        let beam = build_initial_beam(
            y_train,
            bits_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
            group_ids,
            literal_scores.as_slice(),
            &params,
        )?;
        kept_all.extend(beam.iter().cloned());
        beam
    };

    for _depth in (exhaustive_depth + 1)..=max_depth {
        let next = expand_beam_once(
            &beam,
            y_train,
            sum_y_train,
            bits_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
            group_ids,
            literal_scores.as_slice(),
            &params,
        )?;
        if next.is_empty() {
            break;
        }
        kept_all.extend(next.iter().cloned());
        beam = next;
    }

    let retained = dedup_states_by_train_bits(kept_all);

    let mut best_by_rule =
        HashMap::<Vec<(usize, bool, u8)>, BeamRuleCandidate>::with_capacity(retained.len());
    for state in retained.into_iter() {
        let cand = collapse_surrogate_candidate(
            &state,
            y_train,
            bits_train,
            row_words_train,
            n_rows,
            n_train,
            y_test,
            bits_test,
            row_words_test,
            n_test,
            literal_scores.as_slice(),
            &params,
        )?;
        if !keep_rule_after_support_pruning(&cand.rule, &cand.test, n_test, &params) {
            continue;
        }
        let key = cand.rule.lexical_key();
        match best_by_rule.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(cand);
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if cmp_candidate(&cand, slot.get()) == std::cmp::Ordering::Less {
                    slot.insert(cand);
                }
            }
        }
    }
    let mut out = best_by_rule.into_values().collect::<Vec<_>>();
    out.sort_by(cmp_candidate);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_rows(rows: &[Vec<u8>], n_samples: usize) -> (Vec<u64>, usize) {
        let row_words = words_for_samples(n_samples);
        let mut out = vec![0u64; rows.len() * row_words];
        for (ri, row) in rows.iter().enumerate() {
            for (i, &v) in row.iter().take(n_samples).enumerate() {
                if v != 0 {
                    out[ri * row_words + (i >> 6)] |= 1u64 << (i & 63);
                }
            }
        }
        (out, row_words)
    }

    fn literal_scores_for_test(
        y: &[f64],
        bits: &[u64],
        row_words: usize,
        n_rows: usize,
    ) -> Vec<LiteralSingletonScore> {
        let sum_y = y.iter().copied().sum::<f64>();
        let needed_words = words_for_samples(y.len());
        precompute_literal_singleton_scores(
            y,
            sum_y,
            bits,
            row_words,
            needed_words,
            y.len(),
            y,
            sum_y,
            bits,
            row_words,
            needed_words,
            y.len(),
            n_rows,
        )
    }

    fn beam_state_from_rule_for_test(
        rule: BeamRule,
        y: &[f64],
        bits: &[u64],
        row_words: usize,
        n_rows: usize,
        literal_scores: &[LiteralSingletonScore],
    ) -> BeamState {
        let combined = materialize_rule_bits(&rule, bits, row_words, n_rows, y.len()).unwrap();
        let sum_y = y.iter().copied().sum::<f64>();
        let n_hit = support_size_packed(&combined, y.len());
        let train = score_cont_centered_gain_packed_with_n_hit(y, &combined, y.len(), sum_y, n_hit);
        BeamState {
            rule: rule.clone(),
            combined_train: combined,
            train,
            train_abs_score: train.raw_score,
            train_score: train.raw_score,
            max_singleton_train_raw: rule_max_singleton_raw(&rule, literal_scores, true),
            max_singleton_test_raw: rule_max_singleton_raw(&rule, literal_scores, false),
        }
    }

    #[test]
    fn test_materialize_rule_bits_or_and_not() {
        let rows = vec![
            vec![1, 1, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 1, 0, 1, 1, 0],
        ];
        let (bits, row_words) = pack_rows(&rows, 6);
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![
                (
                    BeamBinaryOp::Or,
                    BeamLiteral {
                        row_index: 1,
                        group_id: 1,
                        negated: false,
                    },
                ),
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 2,
                        group_id: 2,
                        negated: true,
                    },
                ),
            ],
        };
        let combined = materialize_rule_bits(&rule, &bits, row_words, rows.len(), 6).unwrap();
        let got = (0..6)
            .map(|i| usize::from(((combined[i >> 6] >> (i & 63)) & 1u64) != 0))
            .collect::<Vec<_>>();
        assert_eq!(got, vec![1, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn test_beam_search_finds_or_rule() {
        let rows = vec![
            vec![1, 1, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 1, 1],
        ];
        let y = vec![4.0, 4.2, 3.8, 4.1, -1.0, -1.2, -1.1, -0.9];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![0usize, 1, 2, 3];
        let out = beam_search_train_test_continuous(
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            &group_ids,
            BeamSearchParams {
                max_pick: 2,
                beam_width: 8,
                rank_mode: BeamRankMode::Raw,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|cand| {
            cand.rule.len() == 2
                && cand.rule.first.row_index == 0
                && cand.rule.rest.len() == 1
                && cand.rule.rest[0].0 == BeamBinaryOp::Or
                && cand.rule.rest[0].1.row_index == 1
        }));
    }

    #[test]
    fn test_beam_search_respects_group_exclusion() {
        let rows = vec![
            vec![1, 0, 1, 0, 1, 0, 1, 0],
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            vec![1, 1, 0, 0, 1, 1, 0, 0],
        ];
        let y = vec![3.0, -2.0, 3.1, -2.1, 3.2, -2.2, 3.0, -2.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![5usize, 5usize, 9usize];
        let out = beam_search_train_test_continuous(
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            &group_ids,
            BeamSearchParams {
                max_pick: 3,
                beam_width: 12,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        for cand in out.iter() {
            let mut used = HashSet::new();
            assert!(used.insert(cand.rule.first.group_id));
            for (_, lit) in cand.rule.rest.iter() {
                assert!(used.insert(lit.group_id));
            }
        }
    }

    #[test]
    fn test_beam_search_logic_gate_and_only() {
        let rows = vec![
            vec![1, 1, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 0, 0, 0, 0],
        ];
        let y = vec![4.0, 4.0, 4.0, 4.0, -1.0, -1.0, -1.0, -1.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![0usize, 1, 2];
        let out = beam_search_train_test_continuous(
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            &group_ids,
            BeamSearchParams {
                max_pick: 2,
                beam_width: 8,
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|cand| {
            cand.rule
                .rest
                .iter()
                .all(|(op, _)| *op == BeamBinaryOp::And)
        }));
    }

    #[test]
    fn test_beam_search_logic_gate_or_only() {
        let rows = vec![
            vec![1, 1, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 1, 1],
        ];
        let y = vec![4.0, 4.2, 3.8, 4.1, -1.0, -1.2, -1.1, -0.9];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![0usize, 1, 2, 3];
        let out = beam_search_train_test_continuous(
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            &group_ids,
            BeamSearchParams {
                max_pick: 2,
                beam_width: 8,
                logic_gate_mode: BeamLogicGateMode::OrOnly,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(out
            .iter()
            .all(|cand| { cand.rule.rest.iter().all(|(op, _)| *op == BeamBinaryOp::Or) }));
    }

    #[test]
    fn test_exhaustive_pair_depth_recovers_weak_single_strong_pair() {
        let rows = vec![
            vec![1, 1, 0, 0, 0, 0, 0, 0],
            vec![1, 0, 1, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 1, 0, 0, 0, 0],
        ];
        let y = vec![2.0, -0.2, -0.2, 0.1, -0.1, -0.1, 0.0, 0.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![0usize, 1, 2];

        let out_beam = beam_search_train_test_continuous(
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            &group_ids,
            BeamSearchParams {
                max_pick: 2,
                beam_width: 1,
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                exhaustive_depth: 1,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(out_beam.iter().all(|cand| {
            !(cand.rule.first.row_index == 1
                && cand.rule.rest.len() == 1
                && cand.rule.rest[0].1.row_index == 2)
        }));

        let out_exh = beam_search_train_test_continuous(
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            &group_ids,
            BeamSearchParams {
                max_pick: 2,
                beam_width: 1,
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                exhaustive_depth: 2,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(out_exh.iter().any(|cand| {
            cand.rule.first.row_index == 1
                && cand.rule.rest.len() == 1
                && cand.rule.rest[0].1.row_index == 2
                && cand.rule.rest[0].0 == BeamBinaryOp::And
        }));
    }

    #[test]
    fn test_interaction_gain_scoring_tempers_and_pair_singleton_baseline() {
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::And,
                BeamLiteral {
                    row_index: 2,
                    group_id: 2,
                    negated: false,
                },
            )],
        };
        let train = ContinuousRuleScore {
            score: 0.8,
            raw_score: 0.8,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.25,
            n_hit: 2,
            n_miss: 6,
        };
        let (_, raw_score) = train_scores_for_rule(
            &rule,
            train,
            0.6,
            None,
            None,
            &BeamSearchParams {
                rank_mode: BeamRankMode::Raw,
                ..BeamSearchParams::default()
            },
        );
        let (_, gain_score) = train_scores_for_rule(
            &rule,
            train,
            0.6,
            None,
            None,
            &BeamSearchParams {
                rank_mode: BeamRankMode::InteractionGain,
                ..BeamSearchParams::default()
            },
        );
        assert!((raw_score - 0.8).abs() < 1e-12);
        assert!((gain_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_support_pruning_uses_hit_count_for_and_only_maf_threshold() {
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: Vec::new(),
        };
        let params = BeamSearchParams {
            logic_gate_mode: BeamLogicGateMode::AndOnly,
            maf_threshold: 0.02,
            ..BeamSearchParams::default()
        };
        let borderline = ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.02,
            n_hit: 20,
            n_miss: 980,
        };
        let passing = ContinuousRuleScore {
            n_hit: 21,
            n_miss: 979,
            support_frac: 0.021,
            ..borderline
        };
        assert!(!keep_rule_after_support_pruning(
            &rule,
            &borderline,
            1000,
            &params
        ));
        assert!(keep_rule_after_support_pruning(
            &rule, &passing, 1000, &params
        ));
    }

    #[test]
    fn test_support_pruning_uses_miss_count_for_or_only_maf_threshold() {
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::Or,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        };
        let params = BeamSearchParams {
            logic_gate_mode: BeamLogicGateMode::OrOnly,
            maf_threshold: 0.02,
            ..BeamSearchParams::default()
        };
        let borderline = ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.98,
            n_hit: 980,
            n_miss: 20,
        };
        let passing = ContinuousRuleScore {
            n_hit: 979,
            n_miss: 21,
            support_frac: 0.979,
            ..borderline
        };
        assert!(!keep_rule_after_support_pruning(
            &rule,
            &borderline,
            1000,
            &params
        ));
        assert!(keep_rule_after_support_pruning(
            &rule, &passing, 1000, &params
        ));
    }

    #[test]
    fn test_support_pruning_uses_two_sample_guardrail_for_and_or() {
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::Or,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        };
        let params = BeamSearchParams {
            logic_gate_mode: BeamLogicGateMode::AndOr,
            maf_threshold: 0.20,
            ..BeamSearchParams::default()
        };
        let singleton_minor = ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.90,
            n_hit: 9,
            n_miss: 1,
        };
        let two_minor = ContinuousRuleScore {
            n_hit: 8,
            n_miss: 2,
            support_frac: 0.80,
            ..singleton_minor
        };
        assert!(!keep_rule_after_support_pruning(
            &rule,
            &singleton_minor,
            10,
            &params
        ));
        assert!(keep_rule_after_support_pruning(
            &rule, &two_minor, 10, &params
        ));
    }

    #[test]
    fn test_interaction_gain_scoring_uses_direct_parent_baseline_with_null_penalty() {
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::And,
                BeamLiteral {
                    row_index: 2,
                    group_id: 2,
                    negated: false,
                },
            )],
        };
        let train = ContinuousRuleScore {
            score: 0.8,
            raw_score: 0.8,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.25,
            n_hit: 2,
            n_miss: 6,
        };
        let mut cal = super::super::permutation::RuleNullCalibrator::new(false);
        let bucket = bucket_from_rule(&rule, train.support_frac, BeamLogicGateMode::AndOnly);
        cal.insert(bucket, 0.0, 0.0);
        let lookup = cal.finalize();
        let (_, gain_score) = train_scores_for_rule(
            &rule,
            train,
            0.6,
            None,
            None,
            &BeamSearchParams {
                rank_mode: BeamRankMode::InteractionGain,
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                null_penalties: Some(Arc::new(lookup)),
                ..BeamSearchParams::default()
            },
        );
        assert!((gain_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_and_only_not_rules_use_same_incremental_baseline() {
        let params = BeamSearchParams {
            logic_gate_mode: BeamLogicGateMode::AndOnly,
            rank_mode: BeamRankMode::InteractionGain,
            null_penalties: Some(Arc::new(
                super::super::permutation::RuleNullPenaltyLookup::default(),
            )),
            ..BeamSearchParams::default()
        };
        let no_not = rank_rule_score_components_with_gate(GateBucket::And, 2, 0, 0.8, 0.6, &params);
        let with_not =
            rank_rule_score_components_with_gate(GateBucket::And, 2, 1, 0.8, 0.6, &params);
        assert!((no_not - 0.2).abs() < 1e-12);
        assert!((with_not - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_interaction_gain_scoring_uses_direct_parent_baseline_for_triple() {
        let params = BeamSearchParams {
            rank_mode: BeamRankMode::InteractionGain,
            logic_gate_mode: BeamLogicGateMode::AndOnly,
            null_penalties: Some(Arc::new(
                super::super::permutation::RuleNullPenaltyLookup::default(),
            )),
            ..BeamSearchParams::default()
        };
        let triple_score =
            rank_rule_score_components_with_gate(GateBucket::And, 3, 0, 0.8, 0.6, &params);
        assert!((triple_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_pure_and_triple_with_null_penalty_uses_direct_parent_baseline() {
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 1,
                        group_id: 1,
                        negated: false,
                    },
                ),
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 2,
                        group_id: 2,
                        negated: false,
                    },
                ),
            ],
        };
        let train = ContinuousRuleScore {
            score: 0.8,
            raw_score: 0.8,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.15,
            n_hit: 2,
            n_miss: 10,
        };
        let mut cal = super::super::permutation::RuleNullCalibrator::new(false);
        let bucket = bucket_from_rule(&rule, train.support_frac, BeamLogicGateMode::AndOnly);
        cal.insert(bucket, 0.0, 0.0);
        let lookup = cal.finalize();
        let parent_raw_score = 0.72;
        let parent_abs_score = 0.52;
        let (_, gain_score) = train_scores_for_rule(
            &rule,
            train,
            0.79,
            Some(parent_abs_score),
            Some(parent_raw_score),
            &BeamSearchParams {
                rank_mode: BeamRankMode::InteractionGain,
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                null_penalties: Some(Arc::new(lookup)),
                ..BeamSearchParams::default()
            },
        );
        assert!((gain_score - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_or_rule_uses_direct_parent_baseline_under_gain_mode() {
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::Or,
                BeamLiteral {
                    row_index: 2,
                    group_id: 2,
                    negated: false,
                },
            )],
        };
        let train = ContinuousRuleScore {
            score: 0.8,
            raw_score: 0.8,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.40,
            n_hit: 3,
            n_miss: 5,
        };
        let (_, gain_score) = train_scores_for_rule(
            &rule,
            train,
            0.6,
            None,
            None,
            &BeamSearchParams {
                rank_mode: BeamRankMode::InteractionGain,
                logic_gate_mode: BeamLogicGateMode::OrOnly,
                ..BeamSearchParams::default()
            },
        );
        assert!((gain_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_exhaustive_then_gain_delays_gain_until_after_exhaustive_depth() {
        let params = BeamSearchParams {
            rank_mode: BeamRankMode::ExhaustiveThenGain,
            exhaustive_depth: 2,
            ..BeamSearchParams::default()
        };
        let single_score = rank_rule_score_components(1, 0, 0.8, 0.6, &params);
        let pair_score = rank_rule_score_components(2, 0, 0.8, 0.6, &params);
        let triple_score = rank_rule_score_components(3, 0, 0.8, 0.6, &params);
        assert!((single_score - 0.8).abs() < 1e-12);
        assert!((pair_score - 0.8).abs() < 1e-12);
        assert!((triple_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_gain_from_layer_starts_gain_at_requested_depth() {
        let params = BeamSearchParams {
            rank_mode: BeamRankMode::GainFromLayer(3),
            ..BeamSearchParams::default()
        };
        let single_score = rank_rule_score_components(1, 0, 0.8, 0.6, &params);
        let pair_score = rank_rule_score_components(2, 0, 0.8, 0.6, &params);
        let triple_score = rank_rule_score_components(3, 0, 0.8, 0.6, &params);
        assert!((single_score - 0.8).abs() < 1e-12);
        assert!((pair_score - 0.8).abs() < 1e-12);
        assert!((triple_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_higher_order_and_gain_uses_direct_parent_baseline() {
        let params = BeamSearchParams {
            rank_mode: BeamRankMode::InteractionGain,
            ..BeamSearchParams::default()
        };
        let triple_score =
            rank_rule_score_components_with_gate(GateBucket::And, 3, 0, 0.8, 0.6, &params);
        assert!((triple_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_layer1_does_not_seed_negated_singletons() {
        let rows = vec![vec![1, 1, 0, 0, 1, 0, 1, 0], vec![0, 1, 0, 1, 0, 1, 0, 1]];
        let y = vec![2.0, 2.1, -1.0, -1.1, 1.9, -0.9, 2.2, -1.2];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![0usize, 1usize];
        let out = beam_search_train_test_continuous(
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            &group_ids,
            BeamSearchParams {
                max_pick: 1,
                beam_width: 8,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|cand| !cand.rule.first.negated));
    }

    #[test]
    fn test_expand_beam_once_layer3_blind_scan_can_add_smaller_index() {
        let rows = vec![vec![1, 1, 0, 0], vec![1, 0, 1, 0], vec![1, 1, 1, 0]];
        let y = vec![2.0, 1.0, -1.0, -2.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let literal_scores = literal_scores_for_test(&y, &bits, row_words, rows.len());
        let group_ids = vec![0usize, 1usize, 2usize];
        let pair_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::And,
                BeamLiteral {
                    row_index: 2,
                    group_id: 2,
                    negated: false,
                },
            )],
        };
        let state = beam_state_from_rule_for_test(
            pair_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            literal_scores.as_slice(),
        );
        let next = expand_beam_once(
            &[state],
            &y,
            y.iter().copied().sum::<f64>(),
            &bits,
            row_words,
            rows.len(),
            words_for_samples(y.len()),
            y.len(),
            &group_ids,
            literal_scores.as_slice(),
            &BeamSearchParams {
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                rank_mode: BeamRankMode::Raw,
                beam_width: 16,
                allow_parallel: false,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        let expected = vec![
            (0usize, false, 0u8),
            (1usize, false, 1u8),
            (2usize, false, 1u8),
        ];
        assert!(next
            .iter()
            .any(|state| state.rule.lexical_key() == expected));
    }

    #[test]
    fn test_expand_beam_once_layer3_blind_scan_dedups_commutative_triple() {
        let rows = vec![vec![1, 1, 0, 0], vec![1, 0, 1, 0], vec![1, 1, 1, 0]];
        let y = vec![2.0, 1.0, -1.0, -2.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let literal_scores = literal_scores_for_test(&y, &bits, row_words, rows.len());
        let group_ids = vec![0usize, 1usize, 2usize];
        let pair_01 = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 0,
                    group_id: 0,
                    negated: false,
                },
                rest: vec![(
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 1,
                        group_id: 1,
                        negated: false,
                    },
                )],
            },
            &y,
            &bits,
            row_words,
            rows.len(),
            literal_scores.as_slice(),
        );
        let pair_02 = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 0,
                    group_id: 0,
                    negated: false,
                },
                rest: vec![(
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 2,
                        group_id: 2,
                        negated: false,
                    },
                )],
            },
            &y,
            &bits,
            row_words,
            rows.len(),
            literal_scores.as_slice(),
        );
        let next = expand_beam_once(
            &[pair_01, pair_02],
            &y,
            y.iter().copied().sum::<f64>(),
            &bits,
            row_words,
            rows.len(),
            words_for_samples(y.len()),
            y.len(),
            &group_ids,
            literal_scores.as_slice(),
            &BeamSearchParams {
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                rank_mode: BeamRankMode::Raw,
                beam_width: 16,
                allow_parallel: false,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        let expected = vec![
            (0usize, false, 0u8),
            (1usize, false, 1u8),
            (2usize, false, 1u8),
        ];
        assert_eq!(
            next.iter()
                .filter(|state| state.rule.lexical_key() == expected)
                .count(),
            1
        );
    }

    #[test]
    fn test_parent_gain_pruning_requires_triple_to_beat_pair() {
        let pair_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::And,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        };
        let triple_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 1,
                        group_id: 1,
                        negated: false,
                    },
                ),
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 2,
                        group_id: 2,
                        negated: false,
                    },
                ),
            ],
        };
        let quad_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 1,
                        group_id: 1,
                        negated: false,
                    },
                ),
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 2,
                        group_id: 2,
                        negated: false,
                    },
                ),
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 3,
                        group_id: 3,
                        negated: false,
                    },
                ),
            ],
        };
        let params_default = BeamSearchParams {
            exhaustive_depth: 2,
            ..BeamSearchParams::default()
        };
        assert!(!keep_child_after_parent_gain_pruning(
            &triple_rule,
            0.0,
            &params_default
        ));
        assert!(!keep_child_after_parent_gain_pruning(
            &triple_rule,
            -0.001,
            &params_default
        ));
        assert!(keep_child_after_parent_gain_pruning(
            &triple_rule,
            0.01,
            &params_default
        ));
        assert!(!keep_child_after_parent_gain_pruning(
            &pair_rule,
            0.0,
            &params_default
        ));
        assert!(!keep_child_after_parent_gain_pruning(
            &quad_rule,
            0.0,
            &params_default
        ));
        assert!(!keep_child_after_parent_gain_pruning(
            &quad_rule,
            -0.001,
            &params_default
        ));
        assert!(keep_child_after_parent_gain_pruning(
            &quad_rule,
            0.01,
            &params_default
        ));

        let params_perm = BeamSearchParams {
            exhaustive_depth: 2,
            logic_gate_mode: BeamLogicGateMode::AndOnly,
            null_penalties: Some(Arc::new(
                super::super::permutation::RuleNullPenaltyLookup::default(),
            )),
            ..BeamSearchParams::default()
        };
        assert!(!keep_child_after_parent_gain_pruning(
            &triple_rule,
            0.0,
            &params_perm
        ));
        assert!(!keep_child_after_parent_gain_pruning(
            &triple_rule,
            -0.001,
            &params_perm
        ));
        assert!(keep_child_after_parent_gain_pruning(
            &triple_rule,
            0.01,
            &params_perm
        ));
    }

    #[test]
    fn test_parent_abs_improvement_pruning_is_noop_under_incremental_scoring() {
        let params = BeamSearchParams::default();
        assert!(keep_child_after_parent_abs_improvement_pruning(
            1.0, 2, 1.0, &params
        ));
        assert!(keep_child_after_parent_abs_improvement_pruning(
            1.0, 2, 0.99, &params
        ));
        assert!(keep_child_after_parent_abs_improvement_pruning(
            1.0, 2, 1.011, &params
        ));
    }

    #[test]
    fn test_exhaustive_seed_still_allows_singleton_to_win() {
        let rows = vec![
            vec![1, 1, 1, 1, 0, 0, 0, 0],
            vec![1, 0, 1, 0, 1, 0, 1, 0],
            vec![0, 1, 0, 1, 0, 1, 0, 1],
        ];
        let y = vec![3.0, 3.1, 2.9, 3.2, -3.0, -2.9, -3.1, -3.2];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![0usize, 1, 2];
        let out = beam_search_train_test_continuous(
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            &group_ids,
            BeamSearchParams {
                max_pick: 3,
                beam_width: 16,
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                exhaustive_depth: 2,
                rank_mode: BeamRankMode::InteractionGain,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert_eq!(out[0].rule.len(), 1);
        assert_eq!(out[0].rule.first.row_index, 0);
        assert!(!out[0].rule.first.negated);
    }

    #[test]
    fn test_best_singleton_is_retained_with_fixed_width_beam() {
        let rows = vec![
            vec![1, 1, 1, 1, 0, 0, 0, 0],
            vec![1, 0, 1, 0, 1, 0, 1, 0],
            vec![0, 1, 0, 1, 0, 1, 0, 1],
        ];
        let y = vec![3.0, 3.1, 2.9, 3.2, -3.0, -2.9, -3.1, -3.2];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![0usize, 1, 2];
        let out = beam_search_train_test_continuous(
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            &group_ids,
            BeamSearchParams {
                max_pick: 3,
                beam_width: 16,
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                exhaustive_depth: 2,
                rank_mode: BeamRankMode::InteractionGain,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(out.iter().any(|cand| {
            cand.rule.len() == 1 && cand.rule.first.row_index == 0 && !cand.rule.first.negated
        }));
    }

    #[test]
    fn test_surrogate_or_rule_collapses_back_to_singleton() {
        let rows = vec![vec![1, 1, 1, 1, 0, 0, 0, 0], vec![0, 0, 0, 0, 1, 0, 0, 0]];
        let y = vec![3.0, 3.1, 2.9, 3.2, 2.7, -3.0, -3.1, -2.9];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let params = BeamSearchParams {
            logic_gate_mode: BeamLogicGateMode::OrOnly,
            rank_mode: BeamRankMode::Raw,
            surrogate_test_gain_max: 0.10,
            surrogate_hamming_frac_max: 0.20,
            ..BeamSearchParams::default()
        };
        let y_sum = y.iter().copied().sum::<f64>();
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            rows.len(),
        );
        let parent_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: Vec::new(),
        };
        let child_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::Or,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        };
        let _parent_train = evaluate_rule_continuous(
            &parent_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            0.0,
            0.0,
        )
        .unwrap();
        let child_train = evaluate_rule_continuous(
            &child_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            0.0,
            0.0,
        )
        .unwrap();
        let max_singleton_train_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), true);
        let max_singleton_test_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), false);
        let child_bits =
            materialize_rule_bits(&child_rule, &bits, row_words, rows.len(), y.len()).unwrap();
        let (child_abs, child_score) = train_scores_for_rule(
            &child_rule,
            child_train,
            max_singleton_train_raw,
            None,
            None,
            &params,
        );
        let state = BeamState {
            rule: child_rule,
            combined_train: child_bits,
            train: child_train,
            train_abs_score: child_abs,
            train_score: child_score,
            max_singleton_train_raw,
            max_singleton_test_raw,
        };
        let collapsed = collapse_surrogate_candidate(
            &state,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            literal_scores.as_slice(),
            &params,
        )
        .unwrap();
        assert_eq!(collapsed.rule.len(), 1);
        assert_eq!(collapsed.rule.first.row_index, 0);
    }

    #[test]
    fn test_surrogate_collapse_does_not_trigger_for_large_support_change() {
        let rows = vec![vec![1, 1, 1, 1, 0, 0, 0, 0], vec![0, 0, 0, 0, 1, 1, 1, 1]];
        let y = vec![3.0, 3.1, 2.9, 3.2, 2.7, 2.8, 2.6, 2.9];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let params = BeamSearchParams {
            logic_gate_mode: BeamLogicGateMode::OrOnly,
            rank_mode: BeamRankMode::Raw,
            surrogate_test_gain_max: 0.10,
            surrogate_hamming_frac_max: 0.20,
            ..BeamSearchParams::default()
        };
        let y_sum = y.iter().copied().sum::<f64>();
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            rows.len(),
        );
        let parent_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: Vec::new(),
        };
        let child_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::Or,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        };
        let _parent_train = evaluate_rule_continuous(
            &parent_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            0.0,
            0.0,
        )
        .unwrap();
        let child_train = evaluate_rule_continuous(
            &child_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            0.0,
            0.0,
        )
        .unwrap();
        let max_singleton_train_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), true);
        let max_singleton_test_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), false);
        let child_bits =
            materialize_rule_bits(&child_rule, &bits, row_words, rows.len(), y.len()).unwrap();
        let (child_abs, child_score) = train_scores_for_rule(
            &child_rule,
            child_train,
            max_singleton_train_raw,
            None,
            None,
            &params,
        );
        let state = BeamState {
            rule: child_rule,
            combined_train: child_bits,
            train: child_train,
            train_abs_score: child_abs,
            train_score: child_score,
            max_singleton_train_raw,
            max_singleton_test_raw,
        };
        let collapsed = collapse_surrogate_candidate(
            &state,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            literal_scores.as_slice(),
            &params,
        )
        .unwrap();
        assert_eq!(collapsed.rule.len(), 2);
    }

    #[test]
    fn test_surrogate_or_not_proxy_collapses_to_positive_singleton() {
        let rows = vec![vec![0, 0, 0, 0, 1, 0, 0, 0], vec![1, 1, 1, 1, 0, 0, 0, 0]];
        let y = vec![-3.0, -3.1, -2.9, -3.2, 2.8, 2.9, 3.1, 3.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let params = BeamSearchParams {
            logic_gate_mode: BeamLogicGateMode::OrOnly,
            rank_mode: BeamRankMode::Raw,
            surrogate_test_gain_max: 0.10,
            surrogate_hamming_frac_max: 0.20,
            ..BeamSearchParams::default()
        };
        let y_sum = y.iter().copied().sum::<f64>();
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            rows.len(),
        );
        let child_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::Or,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: true,
                },
            )],
        };
        let child_train = evaluate_rule_continuous(
            &child_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            0.0,
            0.0,
        )
        .unwrap();
        let max_singleton_train_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), true);
        let max_singleton_test_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), false);
        let child_bits =
            materialize_rule_bits(&child_rule, &bits, row_words, rows.len(), y.len()).unwrap();
        let (child_abs, child_score) = train_scores_for_rule(
            &child_rule,
            child_train,
            max_singleton_train_raw,
            None,
            None,
            &params,
        );
        let state = BeamState {
            rule: child_rule,
            combined_train: child_bits,
            train: child_train,
            train_abs_score: child_abs,
            train_score: child_score,
            max_singleton_train_raw,
            max_singleton_test_raw,
        };
        let collapsed = collapse_surrogate_candidate(
            &state,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            literal_scores.as_slice(),
            &params,
        )
        .unwrap();
        assert_eq!(collapsed.rule.len(), 1);
        assert_eq!(collapsed.rule.first.row_index, 1);
        assert!(!collapsed.rule.first.negated);
    }

    #[test]
    fn test_surrogate_and_not_proxy_collapses_to_shorter_and_subrule() {
        let rows = vec![
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ];
        let y = vec![
            3.0, 3.1, 2.9, 3.2, 2.8, 3.0, 3.1, 2.9, 3.0, 3.2, -3.0, -3.1, -2.9, -3.2, -2.8, -3.0,
            -3.1, -2.9, -3.0, -3.2,
        ];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let params = BeamSearchParams {
            logic_gate_mode: BeamLogicGateMode::AndOnly,
            rank_mode: BeamRankMode::InteractionGain,
            surrogate_test_gain_max: 0.02,
            surrogate_hamming_frac_max: 0.02,
            ..BeamSearchParams::default()
        };
        let y_sum = y.iter().copied().sum::<f64>();
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            rows.len(),
        );
        let child_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 1,
                        group_id: 1,
                        negated: false,
                    },
                ),
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 2,
                        group_id: 2,
                        negated: true,
                    },
                ),
            ],
        };
        let child_train = evaluate_rule_continuous(
            &child_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            0.0,
            0.0,
        )
        .unwrap();
        let max_singleton_train_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), true);
        let max_singleton_test_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), false);
        let direct_parent_train_raw = best_direct_parent_raw_baseline(
            &child_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            literal_scores.as_slice(),
            true,
        )
        .unwrap();
        let child_bits =
            materialize_rule_bits(&child_rule, &bits, row_words, rows.len(), y.len()).unwrap();
        let (child_abs, child_score) = train_scores_for_rule(
            &child_rule,
            child_train,
            direct_parent_train_raw,
            None,
            None,
            &params,
        );
        let state = BeamState {
            rule: child_rule,
            combined_train: child_bits,
            train: child_train,
            train_abs_score: child_abs,
            train_score: child_score,
            max_singleton_train_raw,
            max_singleton_test_raw,
        };
        let collapsed = collapse_surrogate_candidate(
            &state,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            literal_scores.as_slice(),
            &params,
        )
        .unwrap();
        assert_eq!(collapsed.rule.len(), 2);
        assert_eq!(collapsed.rule.first.row_index, 0);
        assert_eq!(collapsed.rule.rest[0].1.row_index, 1);
        assert!(!collapsed.rule.rest[0].1.negated);
    }

    #[test]
    fn test_surrogate_pure_and_triple_does_not_collapse_to_pair() {
        let mut row_a = vec![0u8; 20];
        let mut row_b = vec![0u8; 20];
        let mut row_c = vec![0u8; 20];
        for idx in 0..10 {
            row_a[idx] = 1;
            row_b[idx] = 1;
            row_c[idx] = 1;
        }
        row_c[0] = 0;
        let rows = vec![row_a, row_b, row_c];
        let y = vec![
            3.0, 3.1, 2.9, 3.2, 2.8, 3.0, 3.1, 2.9, 3.0, 3.2, -3.0, -3.1, -2.9, -3.2, -2.8, -3.0,
            -3.1, -2.9, -3.0, -3.2,
        ];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let params = BeamSearchParams {
            logic_gate_mode: BeamLogicGateMode::AndOnly,
            rank_mode: BeamRankMode::InteractionGain,
            surrogate_test_gain_max: 0.20,
            surrogate_hamming_frac_max: 0.20,
            ..BeamSearchParams::default()
        };
        let y_sum = y.iter().copied().sum::<f64>();
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
            y_sum,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            rows.len(),
        );
        let child_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 1,
                        group_id: 1,
                        negated: false,
                    },
                ),
                (
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 2,
                        group_id: 2,
                        negated: false,
                    },
                ),
            ],
        };
        let child_train = evaluate_rule_continuous(
            &child_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            0.0,
            0.0,
        )
        .unwrap();
        let max_singleton_train_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), true);
        let max_singleton_test_raw =
            rule_max_singleton_raw(&child_rule, literal_scores.as_slice(), false);
        let direct_parent_train_raw = best_direct_parent_raw_baseline(
            &child_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            literal_scores.as_slice(),
            true,
        )
        .unwrap();
        let child_bits =
            materialize_rule_bits(&child_rule, &bits, row_words, rows.len(), y.len()).unwrap();
        let (child_abs, child_score) = train_scores_for_rule(
            &child_rule,
            child_train,
            direct_parent_train_raw,
            None,
            None,
            &params,
        );
        let state = BeamState {
            rule: child_rule,
            combined_train: child_bits,
            train: child_train,
            train_abs_score: child_abs,
            train_score: child_score,
            max_singleton_train_raw,
            max_singleton_test_raw,
        };
        let collapsed = collapse_surrogate_candidate(
            &state,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &bits,
            row_words,
            y.len(),
            literal_scores.as_slice(),
            &params,
        )
        .unwrap();
        assert_eq!(collapsed.rule.len(), 3);
    }
}
