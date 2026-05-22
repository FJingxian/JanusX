#![allow(dead_code)]

use super::permutation::{
    bucket_from_rule, structure_prior_penalty, GateBucket, RuleNullBucket, RuleNullPenaltyLookup,
    RuleStructurePrior,
};
use super::score::{
    score_cont_weighted_mean_diff_packed, validate_continuous_y, ContinuousRuleScore,
};
use crate::bitwise::{bitand_assign, bitnot_masked, bitor_into};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const GARFIELD_BEAM_PAR_MIN_TOTAL_CANDS: usize = 1_024;
const GARFIELD_BEAM_PAR_CHUNK_CANDS: usize = 128;
const GARFIELD_INITIAL_SINGLETON_NEGATIONS: [bool; 1] = [false];
const GARFIELD_AND_PAIR_GAIN_SINGLETON_WEIGHT: f64 = 0.70;
const GARFIELD_AND_PAIR_GAIN_SINGLETON_WEIGHT_WITH_NULL: f64 = 0.35;
const GARFIELD_AND_TRIPLE_GAIN_SINGLETON_WEIGHT_WITH_NULL: f64 = 0.35;
const GARFIELD_AND_ONLY_NOT_EXTRA_PENALTY_WITH_NULL: f64 = 0.15;
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
    pub candidate_keep_ratio: f64,
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
            candidate_keep_ratio: 0.10,
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
fn gate_prefers_raw_score(gate: GateBucket, rule_len: usize, params: &BeamSearchParams) -> bool {
    rule_len >= 2 && matches!(gate, GateBucket::Or) && rank_mode_uses_gain(rule_len, params)
}

#[inline]
fn gain_singleton_baseline_weight(
    gate: GateBucket,
    rule_len: usize,
    params: &BeamSearchParams,
) -> f64 {
    if matches!(gate, GateBucket::And) {
        if rule_len == 2 {
            if params.null_penalties.is_some() {
                return GARFIELD_AND_PAIR_GAIN_SINGLETON_WEIGHT_WITH_NULL;
            }
            return GARFIELD_AND_PAIR_GAIN_SINGLETON_WEIGHT;
        }
        if rule_len == 3 && params.null_penalties.is_some() {
            return GARFIELD_AND_TRIPLE_GAIN_SINGLETON_WEIGHT_WITH_NULL;
        }
    }
    1.0
}

#[inline]
fn use_parent_delta_with_gate(
    gate: GateBucket,
    rule_len: usize,
    not_count: usize,
    params: &BeamSearchParams,
) -> bool {
    if params.disable_parent_delta || params.structure_prior.is_some() {
        return false;
    }
    // Under permutation, pure positive 3-site AND rules should be judged by how much
    // they improve on their best 2-site parent, not by a second singleton-style gap.
    if matches!(gate, GateBucket::And)
        && rule_len == 3
        && not_count == 0
        && params.null_penalties.is_some()
    {
        return true;
    }
    let delta_start_len = params.exhaustive_depth.max(1).saturating_add(1).max(3);
    rule_len > delta_start_len
}

#[inline]
fn implicit_gate_penalty(gate: GateBucket, not_count: usize, params: &BeamSearchParams) -> f64 {
    if matches!(gate, GateBucket::And)
        && not_count > 0
        && params.null_penalties.is_some()
        && matches!(params.logic_gate_mode, BeamLogicGateMode::AndOnly)
    {
        GARFIELD_AND_ONLY_NOT_EXTRA_PENALTY_WITH_NULL * (not_count as f64)
    } else {
        0.0
    }
}

#[inline]
fn rank_rule_score_components_with_gate(
    gate: GateBucket,
    rule_len: usize,
    not_count: usize,
    raw_score: f64,
    max_singleton_raw: f64,
    params: &BeamSearchParams,
) -> f64 {
    let use_gain =
        rank_mode_uses_gain(rule_len, params) && !gate_prefers_raw_score(gate, rule_len, params);
    let base = if use_gain {
        raw_score - gain_singleton_baseline_weight(gate, rule_len, params) * max_singleton_raw
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
    let implicit_pen = implicit_gate_penalty(gate, not_count, params);
    base - len_pen - not_pen - structure_pen - implicit_pen
}

#[inline]
pub fn rank_rule_score_components(
    rule_len: usize,
    not_count: usize,
    raw_score: f64,
    max_singleton_raw: f64,
    params: &BeamSearchParams,
) -> f64 {
    rank_rule_score_components_with_gate(
        GateBucket::And,
        rule_len,
        not_count,
        raw_score,
        max_singleton_raw,
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
    max_singleton_raw: f64,
    params: &BeamSearchParams,
    is_train: bool,
) -> f64 {
    rank_rule_score_components_with_gate(
        bucket.gate,
        rule_len,
        not_count,
        raw_score,
        max_singleton_raw,
        params,
    ) - null_penalty_for_bucket(bucket, params, is_train)
}

#[inline]
fn use_parent_delta(rule_len: usize, params: &BeamSearchParams) -> bool {
    use_parent_delta_with_gate(GateBucket::And, rule_len, 0, params)
}

#[inline]
fn rule_rank_score_from_abs(
    abs_score: f64,
    parent_abs_score: Option<f64>,
    use_parent_delta: bool,
) -> f64 {
    if use_parent_delta {
        abs_score - parent_abs_score.unwrap_or(0.0)
    } else {
        abs_score
    }
}

#[inline]
fn train_scores_for_rule(
    rule: &BeamRule,
    train_raw: ContinuousRuleScore,
    max_singleton_raw: f64,
    parent_abs_score: Option<f64>,
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
        max_singleton_raw,
        params,
    );
    let threshold = null_penalty_for_bucket(bucket, params, true);
    let rank_score = if use_parent_delta_with_gate(gate, rule.len(), rule.not_count(), params) {
        abs_score - parent_abs_score.unwrap_or(0.0) - threshold
    } else {
        abs_score - threshold
    };
    (abs_score, rank_score)
}

#[inline]
fn support_balance(sc: &ContinuousRuleScore) -> usize {
    sc.n_hit.min(sc.n_miss)
}

#[inline]
fn cmp_rule_lex(a: &BeamRule, b: &BeamRule) -> std::cmp::Ordering {
    a.lexical_key().cmp(&b.lexical_key())
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
fn keep_child_after_parent_gain_pruning(
    child_rule: &BeamRule,
    child_rank_score: f64,
    params: &BeamSearchParams,
) -> bool {
    let gate = if rule_is_pure_or(child_rule) {
        GateBucket::Or
    } else {
        GateBucket::And
    };
    if !use_parent_delta_with_gate(gate, child_rule.len(), child_rule.not_count(), params) {
        return true;
    }
    score_strictly_improves(child_rank_score, 0.0)
}

#[inline]
fn keep_child_after_parent_abs_improvement_pruning(
    parent_abs_score: f64,
    child_rule_len: usize,
    child_abs_score: f64,
    params: &BeamSearchParams,
) -> bool {
    if child_rule_len != 2 {
        return true;
    }
    if !(params.min_parent_abs_gain.is_finite() && params.min_parent_abs_gain > 0.0) {
        return true;
    }
    let child = score_key(child_abs_score);
    let parent = score_key(parent_abs_score);
    child.is_finite()
        && parent.is_finite()
        && (child - parent) > params.min_parent_abs_gain
        && !state_scores_tied(child, parent)
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
    if !(0.0 < params.candidate_keep_ratio && params.candidate_keep_ratio <= 1.0) {
        return Err(format!(
            "{ctx}: candidate_keep_ratio must be within (0, 1], got {}",
            params.candidate_keep_ratio
        ));
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
    bits_train: &[u64],
    row_words_train: usize,
    needed_words_train: usize,
    n_train: usize,
    y_test: &[f64],
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
            out.push(LiteralSingletonScore {
                train: score_cont_weighted_mean_diff_packed(y_train, &train_bits, n_train),
                test: score_cont_weighted_mean_diff_packed(y_test, &test_bits, n_test),
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
    let mut sc = score_cont_weighted_mean_diff_packed(y, &combined, n_samples);
    let penalty = if rule.len() > 1 {
        lambda_len * ((rule.len() - 1) as f64)
    } else {
        0.0
    } + lambda_not * (rule.not_count() as f64);
    sc.score = sc.raw_score - penalty;
    Ok(sc)
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
                        if support_balance(&train) == 0 {
                            continue;
                        }
                        let (train_abs_score, train_score) =
                            train_scores_for_rule(&rule, train, train.raw_score, None, params);
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
                if support_balance(&train) == 0 {
                    continue;
                }
                let (train_abs_score, train_score) =
                    train_scores_for_rule(&rule, train, train.raw_score, None, params);
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
            if support_balance(&train) == 0 {
                continue;
            }
            let (train_abs_score, train_score) =
                train_scores_for_rule(&rule, train, train.raw_score, None, params);
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
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Vec<BeamState> {
    let next_cap = params.beam_width.min(n_rows.saturating_mul(4).max(1));
    let op_count = params.logic_gate_mode.op_count();
    let total_expand = beam
        .iter()
        .map(|node| {
            n_rows
                .saturating_sub(node.rule.last_row_index().saturating_add(1))
                .saturating_mul(2usize.saturating_mul(op_count))
        })
        .sum::<usize>();

    let next = if should_parallel(total_expand, params.allow_parallel) {
        let mut work = Vec::<(usize, usize, usize)>::new();
        let chunk = GARFIELD_BEAM_PAR_CHUNK_CANDS.max(1);
        for (bi, node) in beam.iter().enumerate() {
            let mut start = node.rule.last_row_index() + 1;
            while start < n_rows {
                let end = (start + chunk).min(n_rows);
                work.push((bi, start, end));
                start = end;
            }
        }
        let local_tops: Vec<Vec<BeamState>> = work
            .into_par_iter()
            .map(|(bi, start, end)| {
                let node = &beam[bi];
                let mut local = Vec::<BeamState>::with_capacity(next_cap);
                for cand in start..end {
                    let gid = group_ids[cand];
                    if node.rule.uses_group(gid) {
                        continue;
                    }
                    let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
                    for &op in params.logic_gate_mode.allowed_ops() {
                        for &negated in &[false, true] {
                            let mut combined = node.combined_train.clone();
                            apply_literal_inplace(&mut combined, row, op, negated, n_train);
                            let train =
                                score_cont_weighted_mean_diff_packed(y_train, &combined, n_train);
                            if support_balance(&train) == 0 {
                                continue;
                            }
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
                            let (train_abs_score, train_score) = train_scores_for_rule(
                                &rule,
                                train,
                                max_singleton_train_raw,
                                Some(node.train_abs_score),
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
                local
            })
            .collect();
        let mut merged = Vec::<BeamState>::with_capacity(next_cap);
        for local in local_tops {
            for cand in local {
                push_top_k_states(&mut merged, cand, next_cap);
            }
        }
        merged
    } else {
        let mut seq = Vec::<BeamState>::with_capacity(next_cap);
        for node in beam.iter() {
            for cand in (node.rule.last_row_index() + 1)..n_rows {
                let gid = group_ids[cand];
                if node.rule.uses_group(gid) {
                    continue;
                }
                let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
                for &op in params.logic_gate_mode.allowed_ops() {
                    for &negated in &[false, true] {
                        let mut combined = node.combined_train.clone();
                        apply_literal_inplace(&mut combined, row, op, negated, n_train);
                        let train =
                            score_cont_weighted_mean_diff_packed(y_train, &combined, n_train);
                        if support_balance(&train) == 0 {
                            continue;
                        }
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
                        let (train_abs_score, train_score) = train_scores_for_rule(
                            &rule,
                            train,
                            max_singleton_train_raw,
                            Some(node.train_abs_score),
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
    filter_beam_candidates(next, next_cap, params)
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
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Vec<BeamState> {
    let mut best = HashMap::<Vec<u64>, BeamState>::new();
    for node in frontier.iter() {
        for cand in (node.rule.last_row_index() + 1)..n_rows {
            let gid = group_ids[cand];
            if node.rule.uses_group(gid) {
                continue;
            }
            let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
            for &op in params.logic_gate_mode.allowed_ops() {
                for &negated in &[false, true] {
                    let mut combined = node.combined_train.clone();
                    apply_literal_inplace(&mut combined, row, op, negated, n_train);
                    let train = score_cont_weighted_mean_diff_packed(y_train, &combined, n_train);
                    if support_balance(&train) == 0 {
                        continue;
                    }
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
                    let (train_abs_score, train_score) = train_scores_for_rule(
                        &rule,
                        train,
                        max_singleton_train_raw,
                        Some(node.train_abs_score),
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
    filter_beam_candidates(out, width, params)
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
    let child_abs_score = rank_rule_score_components_with_gate(
        child_gate,
        state.rule.len(),
        state.rule.not_count(),
        test.raw_score,
        state.max_singleton_test_raw,
        params,
    );
    let threshold = null_penalty_for_bucket(child_bucket, params, false);
    let use_parent_delta =
        use_parent_delta_with_gate(child_gate, state.rule.len(), state.rule.not_count(), params);
    if !use_parent_delta {
        return Ok(child_abs_score - threshold);
    }
    let Some(parent_rule) = rule_parent(&state.rule) else {
        return Ok(child_abs_score - threshold);
    };
    let parent_test = evaluate_rule_continuous(
        &parent_rule,
        y_test,
        bits_test,
        row_words_test,
        n_rows,
        n_test,
        params.lambda_len,
        params.lambda_not,
    )?;
    let parent_max_singleton_test_raw = rule_max_singleton_raw(&parent_rule, literal_scores, false);
    let parent_gate = if rule_is_pure_or(&parent_rule) {
        GateBucket::Or
    } else {
        GateBucket::And
    };
    let parent_abs_score = rank_rule_score_components_with_gate(
        parent_gate,
        parent_rule.len(),
        parent_rule.not_count(),
        parent_test.raw_score,
        parent_max_singleton_test_raw,
        params,
    );
    Ok(
        rule_rank_score_from_abs(child_abs_score, Some(parent_abs_score), use_parent_delta)
            - threshold,
    )
}

#[inline]
fn rule_abs_score_for_eval(
    rule: &BeamRule,
    raw: &ContinuousRuleScore,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
    params: &BeamSearchParams,
) -> f64 {
    let max_singleton_raw = rule_max_singleton_raw(rule, literal_scores, is_train);
    let gate = if rule_is_pure_or(rule) {
        GateBucket::Or
    } else {
        GateBucket::And
    };
    rank_rule_score_components_with_gate(
        gate,
        rule.len(),
        rule.not_count(),
        raw.raw_score,
        max_singleton_raw,
        params,
    )
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
    let abs_score = rule_abs_score_for_eval(rule, raw, literal_scores, is_train, params);
    let threshold = null_penalty_for_bucket(bucket, params, is_train);
    let gate = if rule_is_pure_or(rule) {
        GateBucket::Or
    } else {
        GateBucket::And
    };
    let use_parent_delta = use_parent_delta_with_gate(gate, rule.len(), rule.not_count(), params);
    if !use_parent_delta {
        return Ok(abs_score - threshold);
    }
    let Some(parent_rule) = rule_parent(rule) else {
        return Ok(abs_score - threshold);
    };
    let parent_raw = evaluate_rule_continuous(
        &parent_rule,
        y,
        bits,
        row_words,
        n_rows,
        n_samples,
        params.lambda_len,
        params.lambda_not,
    )?;
    let parent_abs_score =
        rule_abs_score_for_eval(&parent_rule, &parent_raw, literal_scores, is_train, params);
    Ok(rule_rank_score_from_abs(abs_score, Some(parent_abs_score), use_parent_delta) - threshold)
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
    let mut current_test = evaluate_rule_continuous(
        &current_rule,
        y_test,
        bits_test,
        row_words_test,
        n_rows,
        n_test,
        params.lambda_len,
        params.lambda_not,
    )?;
    let mut current_test_score = final_rule_score_for_eval(
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
    )?;

    if surrogate_collapse_enabled(params) && current_rule.len() > 1 {
        let mut current_bits_test =
            materialize_rule_bits(&current_rule, bits_test, row_words_test, n_rows, n_test)?;
        loop {
            let Some(parent_rule) = rule_parent(&current_rule) else {
                break;
            };
            let parent_bits_test =
                materialize_rule_bits(&parent_rule, bits_test, row_words_test, n_rows, n_test)?;
            let diff_frac = bit_hamming_fraction(
                current_bits_test.as_slice(),
                parent_bits_test.as_slice(),
                n_test,
            );
            if diff_frac > params.surrogate_hamming_frac_max {
                break;
            }
            let parent_test = evaluate_rule_continuous(
                &parent_rule,
                y_test,
                bits_test,
                row_words_test,
                n_rows,
                n_test,
                params.lambda_len,
                params.lambda_not,
            )?;
            let parent_test_score = final_rule_score_for_eval(
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
            )?;
            if !surrogate_delta_small_enough(
                current_test_score,
                parent_test_score,
                params.surrogate_test_gain_max,
            ) {
                break;
            }
            let parent_train = evaluate_rule_continuous(
                &parent_rule,
                y_train,
                bits_train,
                row_words_train,
                n_rows,
                n_train,
                params.lambda_len,
                params.lambda_not,
            )?;
            let parent_train_score = final_rule_score_for_eval(
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
            )?;
            current_rule = parent_rule;
            current_train = parent_train;
            current_train_score = parent_train_score;
            current_test = parent_test;
            current_test_score = parent_test_score;
            current_bits_test = parent_bits_test;
            if current_rule.len() <= 1 {
                break;
            }
        }
        while current_rule.len() > 2
            && rule_is_pure_and(&current_rule)
            && current_rule.not_count() > 0
        {
            let mut best_subrule: Option<(BeamRuleCandidate, Vec<u64>, f64)> = None;
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
                let subrule_bits_test =
                    materialize_rule_bits(&subrule, bits_test, row_words_test, n_rows, n_test)?;
                let diff_frac = bit_hamming_fraction(
                    current_bits_test.as_slice(),
                    subrule_bits_test.as_slice(),
                    n_test,
                );
                if diff_frac > hamming_limit {
                    continue;
                }
                let subrule_test = evaluate_rule_continuous(
                    &subrule,
                    y_test,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    params.lambda_len,
                    params.lambda_not,
                )?;
                let subrule_test_score = final_rule_score_for_eval(
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
                )?;
                if !surrogate_delta_small_enough(current_test_score, subrule_test_score, gain_limit)
                {
                    continue;
                }
                let subrule_train = evaluate_rule_continuous(
                    &subrule,
                    y_train,
                    bits_train,
                    row_words_train,
                    n_rows,
                    n_train,
                    params.lambda_len,
                    params.lambda_not,
                )?;
                let subrule_train_score = final_rule_score_for_eval(
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
                )?;
                let subrule_cand = BeamRuleCandidate {
                    rule: subrule,
                    train_score: subrule_train_score,
                    test_score: subrule_test_score,
                    train: subrule_train,
                    test: subrule_test,
                };
                match best_subrule.as_mut() {
                    Some((best_cand, best_bits, best_diff)) => {
                        if cmp_candidate(&subrule_cand, best_cand) == std::cmp::Ordering::Less
                            || (cmp_candidate(&subrule_cand, best_cand)
                                == std::cmp::Ordering::Equal
                                && diff_frac < *best_diff)
                        {
                            *best_cand = subrule_cand;
                            *best_bits = subrule_bits_test;
                            *best_diff = diff_frac;
                        }
                    }
                    None => {
                        best_subrule = Some((subrule_cand, subrule_bits_test, diff_frac));
                    }
                }
            }
            let Some((subrule_cand, subrule_bits_test, _)) = best_subrule else {
                break;
            };
            current_rule = subrule_cand.rule;
            current_train = subrule_cand.train;
            current_train_score = subrule_cand.train_score;
            current_test = subrule_cand.test;
            current_test_score = subrule_cand.test_score;
            current_bits_test = subrule_bits_test;
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
                let singleton_bits_test = materialize_rule_bits(
                    &singleton_rule,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                )?;
                let diff_frac = bit_hamming_fraction(
                    current_bits_test.as_slice(),
                    singleton_bits_test.as_slice(),
                    n_test,
                );
                let orient_diff = diff_frac.min(1.0 - diff_frac);
                if orient_diff > params.surrogate_hamming_frac_max {
                    continue;
                }
                let singleton_test = evaluate_rule_continuous(
                    &singleton_rule,
                    y_test,
                    bits_test,
                    row_words_test,
                    n_rows,
                    n_test,
                    params.lambda_len,
                    params.lambda_not,
                )?;
                let singleton_test_score = final_rule_score_for_eval(
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
                )?;
                if !surrogate_delta_small_enough(
                    current_test_score,
                    singleton_test_score,
                    params.surrogate_test_gain_max,
                ) {
                    continue;
                }
                let singleton_train = evaluate_rule_continuous(
                    &singleton_rule,
                    y_train,
                    bits_train,
                    row_words_train,
                    n_rows,
                    n_train,
                    params.lambda_len,
                    params.lambda_not,
                )?;
                let singleton_train_score = final_rule_score_for_eval(
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

    let max_depth = params.max_pick.min(n_rows);
    let exhaustive_depth = params.exhaustive_depth.max(1).min(max_depth);
    let literal_scores = precompute_literal_singleton_scores(
        y_train,
        bits_train,
        row_words_train,
        needed_words_train,
        n_train,
        y_test,
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
                bits_train,
                row_words_train,
                n_rows,
                needed_words_train,
                n_train,
                group_ids,
                literal_scores.as_slice(),
                &params,
            );
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
            bits_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
            group_ids,
            literal_scores.as_slice(),
            &params,
        );
        if next.is_empty() {
            break;
        }
        kept_all.extend(next.iter().cloned());
        beam = next;
    }

    let pooled = dedup_states_by_train_bits(kept_all);
    let best_singleton = pooled
        .iter()
        .filter(|state| state.rule.len() == 1)
        .min_by(|a, b| cmp_state(a, b))
        .cloned();
    let keep_total = ((pooled.len()) as f64 * params.candidate_keep_ratio)
        .ceil()
        .max(1.0) as usize;
    let mut retained = sort_truncate_states(pooled, keep_total);
    if let Some(single) = best_singleton {
        if !retained
            .iter()
            .any(|state| same_rule(&state.rule, &single.rule))
        {
            retained.push(single);
            retained.sort_by(cmp_state);
        }
    }

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
                candidate_keep_ratio: 1.0,
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
                candidate_keep_ratio: 0.5,
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
                candidate_keep_ratio: 0.5,
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
                candidate_keep_ratio: 0.5,
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
                candidate_keep_ratio: 1.0,
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
                candidate_keep_ratio: 1.0,
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
            &BeamSearchParams {
                rank_mode: BeamRankMode::InteractionGain,
                ..BeamSearchParams::default()
            },
        );
        assert!((raw_score - 0.8).abs() < 1e-12);
        assert!((gain_score - 0.38).abs() < 1e-12);
    }

    #[test]
    fn test_interaction_gain_scoring_tempers_and_pair_more_with_null_penalty() {
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
            &BeamSearchParams {
                rank_mode: BeamRankMode::InteractionGain,
                logic_gate_mode: BeamLogicGateMode::AndOnly,
                null_penalties: Some(Arc::new(lookup)),
                ..BeamSearchParams::default()
            },
        );
        assert!((gain_score - 0.59).abs() < 1e-12);
    }

    #[test]
    fn test_and_only_not_rules_get_extra_penalty_with_null_penalty_enabled() {
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
        assert!((no_not - 0.59).abs() < 1e-12);
        assert!((with_not - 0.44).abs() < 1e-12);
    }

    #[test]
    fn test_interaction_gain_scoring_tempers_and_triple_with_null_penalty() {
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
        assert!((triple_score - 0.59).abs() < 1e-12);
    }

    #[test]
    fn test_or_rule_keeps_raw_score_under_gain_mode() {
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
            &BeamSearchParams {
                rank_mode: BeamRankMode::InteractionGain,
                logic_gate_mode: BeamLogicGateMode::OrOnly,
                ..BeamSearchParams::default()
            },
        );
        assert!((gain_score - 0.8).abs() < 1e-12);
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
    fn test_higher_order_and_gain_still_uses_full_singleton_baseline() {
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
                candidate_keep_ratio: 1.0,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|cand| !cand.rule.first.negated));
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
        assert!(keep_child_after_parent_gain_pruning(
            &triple_rule,
            0.0,
            &params_default
        ));
        assert!(keep_child_after_parent_gain_pruning(
            &triple_rule,
            -0.001,
            &params_default
        ));
        assert!(keep_child_after_parent_gain_pruning(
            &triple_rule,
            0.01,
            &params_default
        ));
        assert!(keep_child_after_parent_gain_pruning(
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
    fn test_parent_abs_improvement_pruning_is_disabled_by_default() {
        let params = BeamSearchParams::default();
        assert!(keep_child_after_parent_abs_improvement_pruning(
            1.0, 2, 1.0, &params
        ));
        assert!(keep_child_after_parent_abs_improvement_pruning(
            1.0, 2, 0.99, &params
        ));
    }

    #[test]
    fn test_parent_abs_improvement_pruning_requires_margin_when_enabled() {
        let params = BeamSearchParams {
            min_parent_abs_gain: 0.01,
            ..BeamSearchParams::default()
        };
        assert!(!keep_child_after_parent_abs_improvement_pruning(
            1.0, 2, 1.005, &params
        ));
        assert!(!keep_child_after_parent_abs_improvement_pruning(
            1.0, 2, 1.01, &params
        ));
        assert!(keep_child_after_parent_abs_improvement_pruning(
            1.0, 2, 1.011, &params
        ));
        assert!(keep_child_after_parent_abs_improvement_pruning(
            1.0, 3, 1.001, &params
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
                candidate_keep_ratio: 1.0,
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
    fn test_best_singleton_is_retained_even_with_small_keep_ratio() {
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
                candidate_keep_ratio: 0.05,
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
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
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
        let parent_train = evaluate_rule_continuous(
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
        let parent_abs = rule_abs_score_for_eval(
            &parent_rule,
            &parent_train,
            literal_scores.as_slice(),
            true,
            &params,
        );
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
            Some(parent_abs),
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
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
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
        let parent_train = evaluate_rule_continuous(
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
        let parent_abs = rule_abs_score_for_eval(
            &parent_rule,
            &parent_train,
            literal_scores.as_slice(),
            true,
            &params,
        );
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
            Some(parent_abs),
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
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
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
            Some(0.0),
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
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
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
        let child_bits =
            materialize_rule_bits(&child_rule, &bits, row_words, rows.len(), y.len()).unwrap();
        let (child_abs, child_score) = train_scores_for_rule(
            &child_rule,
            child_train,
            max_singleton_train_raw,
            Some(0.0),
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
        let literal_scores = precompute_literal_singleton_scores(
            &y,
            &bits,
            row_words,
            words_for_samples(y.len()),
            y.len(),
            &y,
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
        let child_bits =
            materialize_rule_bits(&child_rule, &bits, row_words, rows.len(), y.len()).unwrap();
        let (child_abs, child_score) = train_scores_for_rule(
            &child_rule,
            child_train,
            max_singleton_train_raw,
            Some(0.0),
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
