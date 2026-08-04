#![allow(dead_code)]
// Shared beam-search kernels used by GARFIELD.
//
// Active GARFIELD continuous search is intentionally restricted to:
// - fuzzy dosage rules backed by dual bitplanes (`g >= 1`, `g >= 2`)
// - AND/XOR beam expansion plus negation at the literal level
// - ternary XOR on literal dosages: same homozygotes -> 0, opposite
//   homozygotes -> 2, and any heterozygote-involving mismatch -> 1
// - active search/output scoring only exposes bucket null penalties; inactive
//   compatibility penalty hooks stay pinned off in the current runtime path
//
// Legacy packed-0/1 continuous helpers and OR-family compatibility code are
// retained in this file only for backward-compatible rule parsing/evaluation
// and tests. They are not part of the active `-bfile/-g/-w/-wg` search path.

use super::permutation::{
    bucket_from_rule_with_complexity, structure_prior_penalty, RuleNullBucket,
    RuleNullPenaltyLookup, RuleStructurePrior,
};
use super::score::{
    and_popcount_sum_y_where_both1_with_lookup, binary_maf_from_n_hit, dosage_maf_from_dual_counts,
    dual_packed_summary, score_cont_centered_gain_dual_from_summary,
    score_cont_centered_gain_dual_packed_with_sum, score_cont_centered_gain_from_sum_and_n_hit,
    score_cont_centered_gain_packed_with_n_hit, score_cont_centered_gain_packed_with_sum,
    score_cont_corr_packed, sum_y_where_both1, sum_y_where_both1_four,
    sum_y_where_both1_with_lookup, support_size_packed, validate_continuous_y, ContinuousRuleScore,
    PackedYSumLookup,
};
use super::score_gpu::{
    centered_gain_backend_mode_is_auto, parse_centered_gain_backend_mode_from_env,
    score_cont_centered_gain_singletons_packed_cpu_impl,
    score_cont_centered_gain_singletons_packed_legacy_impl,
    score_cont_centered_gain_singletons_packed_with_backend,
};
use crate::bitwise::{and_popcount, bitand_assign, bitnot_masked, popcount};
use crate::stats_common::{
    check_ctrlc, format_bytes, interrupt_requested, process_memory_usage, INTERRUPTED_MSG,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

const GARFIELD_BEAM_PAR_MIN_TOTAL_CANDS: usize = 1_024;
const GARFIELD_BEAM_PAR_CHUNK_CANDS: usize = 128;
const GARFIELD_EXHAUSTIVE_PAR_MIN_TOTAL_CANDS: usize = 64;
const GARFIELD_EXHAUSTIVE_PAR_CHUNK_CANDS: usize = 32;
/// Layer-1 singletons: evaluate both positive and negated.
/// Negated is NOT redundant — `!i & !j` is an AND-family subgroup
/// identified by pairwise feature selection, and beam search can only
/// reach it if negated singletons enter the beam at layer 1.
const GARFIELD_INITIAL_SINGLETON_NEGATIONS: [bool; 2] = [false, true];
const GARFIELD_INITIAL_SINGLETON_NEGATIONS_POS_ONLY: [bool; 1] = [false];
const GARFIELD_AND_NOT_SHORTER_SUBRULE_GAIN_MAX: f64 = 0.08;
const GARFIELD_AND_NOT_SHORTER_SUBRULE_HAMMING_FRAC_MAX: f64 = 0.05;
const GARFIELD_LITERAL_BATCH_MAX_ROWS_DEFAULT: usize = 16_384;
const GARFIELD_LITERAL_BATCH_MAX_WORK_WORDS_DEFAULT: usize = 1_048_576;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GarfieldBeamProfileSnapshot {
    pub calls: usize,
    pub total_s: f64,
    pub literal_precompute_s: f64,
    pub clone_bits_s: f64,
    pub sum_y_both1_s: f64,
    pub parent_baseline_s: f64,
}

static GARFIELD_BEAM_PROFILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static GARFIELD_BEAM_PROFILE_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_BEAM_PROFILE_LITERAL_PRECOMPUTE_NS: AtomicU64 = AtomicU64::new(0);

// Fine-grained profile atomics for hotspot analysis.
static GARFIELD_PROFILE_CLONE_BITS_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_PROFILE_SUM_Y_BOTH1_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_PROFILE_PARENT_BASELINE_NS: AtomicU64 = AtomicU64::new(0);

#[inline]
fn elapsed_ns_saturating(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

#[inline]
fn beam_detail_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_bool("JX_GARFIELD_BEAM_PROFILE_DETAIL"))
}

#[inline]
fn beam_detail_profile_start() -> Option<Instant> {
    beam_detail_profile_enabled().then(Instant::now)
}

#[inline]
fn beam_detail_profile_end(start: Option<Instant>, counter: &AtomicU64) {
    if let Some(t0) = start {
        counter.fetch_add(elapsed_ns_saturating(t0), Ordering::Relaxed);
    }
}

#[inline]
fn check_interrupt_fast() -> Result<(), String> {
    if interrupt_requested() {
        Err(INTERRUPTED_MSG.to_string())
    } else {
        Ok(())
    }
}

pub(crate) fn reset_garfield_beam_profile() {
    GARFIELD_BEAM_PROFILE_CALLS.store(0, Ordering::Relaxed);
    GARFIELD_BEAM_PROFILE_TOTAL_NS.store(0, Ordering::Relaxed);
    GARFIELD_BEAM_PROFILE_LITERAL_PRECOMPUTE_NS.store(0, Ordering::Relaxed);
    GARFIELD_PROFILE_CLONE_BITS_NS.store(0, Ordering::Relaxed);
    GARFIELD_PROFILE_SUM_Y_BOTH1_NS.store(0, Ordering::Relaxed);
    GARFIELD_PROFILE_PARENT_BASELINE_NS.store(0, Ordering::Relaxed);
}

pub(crate) fn snapshot_garfield_beam_profile() -> GarfieldBeamProfileSnapshot {
    GarfieldBeamProfileSnapshot {
        calls: GARFIELD_BEAM_PROFILE_CALLS.load(Ordering::Relaxed),
        total_s: (GARFIELD_BEAM_PROFILE_TOTAL_NS.load(Ordering::Relaxed) as f64) * 1e-9,
        literal_precompute_s: (GARFIELD_BEAM_PROFILE_LITERAL_PRECOMPUTE_NS.load(Ordering::Relaxed)
            as f64)
            * 1e-9,
        clone_bits_s: (GARFIELD_PROFILE_CLONE_BITS_NS.load(Ordering::Relaxed) as f64) * 1e-9,
        sum_y_both1_s: (GARFIELD_PROFILE_SUM_Y_BOTH1_NS.load(Ordering::Relaxed) as f64) * 1e-9,
        parent_baseline_s: (GARFIELD_PROFILE_PARENT_BASELINE_NS.load(Ordering::Relaxed) as f64)
            * 1e-9,
    }
}

pub(crate) fn add_garfield_beam_profile_literal_precompute_ns(delta_ns: u64) {
    GARFIELD_BEAM_PROFILE_LITERAL_PRECOMPUTE_NS.fetch_add(delta_ns, Ordering::Relaxed);
    GARFIELD_BEAM_PROFILE_TOTAL_NS.fetch_add(delta_ns, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BeamBinaryOp {
    And,
    Or,
    Xor,
}

const BEAM_EXPAND_OPS_AND: [BeamBinaryOp; 1] = [BeamBinaryOp::And];
const BEAM_EXPAND_OPS_AND_XOR: [BeamBinaryOp; 2] = [BeamBinaryOp::And, BeamBinaryOp::Xor];
const BEAM_CHILD_NEGATIONS_AND: [bool; 2] = [false, true];
const BEAM_CHILD_NEGATIONS_XOR: [bool; 1] = [false];

#[inline]
fn beam_binary_op_code(op: BeamBinaryOp) -> u8 {
    match op {
        BeamBinaryOp::And => 1u8,
        BeamBinaryOp::Or => 2u8,
        BeamBinaryOp::Xor => 3u8,
    }
}

#[inline]
fn beam_binary_ops_for_rule(rule: &BeamRule) -> &'static [BeamBinaryOp] {
    if rule_contains_xor(rule) {
        &BEAM_EXPAND_OPS_AND
    } else if rule.len() == 1 && rule.first.negated {
        &BEAM_EXPAND_OPS_AND
    } else {
        &BEAM_EXPAND_OPS_AND_XOR
    }
}

#[inline]
fn child_literal_negations_for_op(op: BeamBinaryOp) -> &'static [bool] {
    match op {
        BeamBinaryOp::And | BeamBinaryOp::Or => &BEAM_CHILD_NEGATIONS_AND,
        BeamBinaryOp::Xor => &BEAM_CHILD_NEGATIONS_XOR,
    }
}

#[inline]
fn beam_child_branch_count_for_rule(rule: &BeamRule) -> usize {
    beam_binary_ops_for_rule(rule)
        .iter()
        .map(|op| child_literal_negations_for_op(*op).len())
        .sum::<usize>()
        .max(1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BeamRankMode {
    Raw,
    InteractionGain,
    ExhaustiveThenGain,
    GainFromLayer(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BeamGroupConstraintMode {
    AlwaysExclude,
    ExcludeUntilDistinctGroups(usize),
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
            out.push((lit.row_index, lit.negated, beam_binary_op_code(*op)));
        }
        out
    }
}

#[inline]
fn rule_contains_xor(rule: &BeamRule) -> bool {
    rule.rest
        .iter()
        .any(|(op, _)| matches!(op, BeamBinaryOp::Xor))
}

#[inline]
fn rule_distinct_group_count(rule: &BeamRule) -> usize {
    let mut seen = [usize::MAX; 5];
    let mut n_seen = 0usize;
    let mut push_group = |group_id: usize| {
        if seen[..n_seen].contains(&group_id) {
            return;
        }
        if n_seen < seen.len() {
            seen[n_seen] = group_id;
            n_seen += 1;
        }
    };
    push_group(rule.first.group_id);
    for (_, lit) in rule.rest.iter() {
        push_group(lit.group_id);
    }
    n_seen
}

#[inline]
fn candidate_group_is_excluded(
    rule: &BeamRule,
    candidate_group_id: usize,
    params: &BeamSearchParams,
) -> bool {
    match params.group_constraint {
        BeamGroupConstraintMode::AlwaysExclude => rule.uses_group(candidate_group_id),
        BeamGroupConstraintMode::ExcludeUntilDistinctGroups(required_groups) => {
            let required_groups = required_groups.max(1);
            rule_distinct_group_count(rule) < required_groups && rule.uses_group(candidate_group_id)
        }
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
    pub maf_threshold: f64,
    pub lambda_len: f64,
    pub lambda_not: f64,
    pub exhaustive_depth: usize,
    pub rank_mode: BeamRankMode,
    pub null_penalties: Option<Arc<RuleNullPenaltyLookup>>,
    pub structure_prior: Option<Arc<RuleStructurePrior>>,
    /// Trait-level lookup shared by scan units to accelerate dense support sums.
    pub y_sum_lookup: Option<Arc<PackedYSumLookup>>,
    pub disable_parent_delta: bool,
    pub null_complexity_bin: u8,
    pub group_constraint: BeamGroupConstraintMode,
    pub allow_parallel: bool,
    pub whole_genome_dev_mode: bool,
    pub filter_xor_substates: bool,
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
            maf_threshold: 0.0,
            lambda_len: 0.0,
            lambda_not: 0.0,
            exhaustive_depth: 1,
            rank_mode: BeamRankMode::InteractionGain,
            null_penalties: None,
            structure_prior: None,
            y_sum_lookup: None,
            disable_parent_delta: false,
            null_complexity_bin: 0,
            group_constraint: BeamGroupConstraintMode::AlwaysExclude,
            allow_parallel: true,
            whole_genome_dev_mode: false,
            filter_xor_substates: true,
        }
    }
}

#[inline]
fn sum_y_where_both1_for_params(
    lhs: &[u64],
    rhs: &[u64],
    y: &[f64],
    n_samples: usize,
    params: &BeamSearchParams,
) -> f64 {
    if let Some(lookup) = params.y_sum_lookup.as_deref() {
        sum_y_where_both1_with_lookup(lhs, rhs, y, n_samples, lookup)
    } else {
        sum_y_where_both1(lhs, rhs, y, n_samples)
    }
}

#[inline]
fn initial_singleton_negations(params: &BeamSearchParams) -> &'static [bool] {
    if params.whole_genome_dev_mode {
        &GARFIELD_INITIAL_SINGLETON_NEGATIONS_POS_ONLY
    } else {
        &GARFIELD_INITIAL_SINGLETON_NEGATIONS
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

#[derive(Clone, Debug)]
struct BeamStateLite {
    rule: BeamRule,
    train: ContinuousRuleScore,
    train_abs_score: f64,
    train_score: f64,
    max_singleton_train_raw: f64,
    max_singleton_test_raw: f64,
}

#[derive(Clone, Debug)]
struct FuzzyBeamState {
    rule: BeamRule,
    combined_train_ge1: Vec<u64>,
    combined_train_ge2: Vec<u64>,
    train: ContinuousRuleScore,
    train_n_ge2: usize,
    train_sum_ge1: f64,
    train_sum_ge2: f64,
    train_abs_score: f64,
    train_score: f64,
    max_singleton_train_raw: f64,
    max_singleton_test_raw: f64,
}

#[derive(Clone, Debug)]
struct FuzzyBeamStateLite {
    rule: BeamRule,
    train: ContinuousRuleScore,
    train_n_ge2: usize,
    train_sum_ge1: f64,
    train_sum_ge2: f64,
    train_abs_score: f64,
    train_score: f64,
    max_singleton_train_raw: f64,
    max_singleton_test_raw: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FuzzyBeamSupportSignature {
    train_ge1: Vec<u64>,
    train_ge2: Vec<u64>,
    test_ge1: Option<Vec<u64>>,
    test_ge2: Option<Vec<u64>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LiteralSingletonScore {
    pub(crate) train: ContinuousRuleScore,
    pub(crate) test: ContinuousRuleScore,
}

#[derive(Clone, Copy, Debug, Default)]
struct DualLiteralSummary {
    pos_n_ge1: usize,
    pos_n_ge2: usize,
    pos_sum_ge1: f64,
    pos_sum_ge2: f64,
}

pub(crate) struct LiteralScoreBatchRequest<'a> {
    pub bits_train: &'a [u64],
    pub row_words_train: usize,
    pub bits_test: &'a [u64],
    pub row_words_test: usize,
    pub n_rows: usize,
}

type RuleLexKey = Vec<(usize, bool, u8)>;
type RuleRawScoreCache = HashMap<RuleLexKey, f64>;
type RuleAncestorBaselineCache = HashMap<RuleLexKey, f64>;
type RuleBitsCache = HashMap<RuleLexKey, Vec<u64>>;

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

const BEAM_PAR_MIN_TOTAL_CANDS: usize = 100_000;
const BEAM_SIMD_MIN_WORDS: usize = 16;
const BEAM_BNB_MIN_ROWS: usize = 128;
const BEAM_BNB_DENSITY_MAX: f64 = 0.08;
const GARFIELD_LAYER_DEBUG_MAX_LAYERS: usize = 64;
const GARFIELD_LAYER_DEBUG_FAMILY_COUNT: usize = 3;
const GARFIELD_LAYER_DEBUG_METRIC_COUNT: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldLayerDebugFamily {
    Singleton = 0,
    And = 1,
    Xor = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldLayerDebugMetric {
    Considered = 0,
    TrainOk = 1,
    AbsOk = 2,
    GainOk = 3,
    ParentOk = 4,
    Kept = 5,
    Retained = 6,
}

static GARFIELD_LAYER_DEBUG_COUNTS: [AtomicU64;
    GARFIELD_LAYER_DEBUG_MAX_LAYERS
        * GARFIELD_LAYER_DEBUG_FAMILY_COUNT
        * GARFIELD_LAYER_DEBUG_METRIC_COUNT] = [const { AtomicU64::new(0) };
    GARFIELD_LAYER_DEBUG_MAX_LAYERS
        * GARFIELD_LAYER_DEBUG_FAMILY_COUNT
        * GARFIELD_LAYER_DEBUG_METRIC_COUNT];

#[derive(Clone, Debug)]
pub struct BeamAndResult {
    pub selected_indices: Vec<usize>,
    pub score: f64,
    pub combined_bits: Vec<u64>,
}

#[derive(Clone, Debug)]
struct BeamNode {
    selected: Vec<usize>,
    combined: Vec<u64>,
    score: f64,
    last_index: usize,
}

#[derive(Clone, Debug)]
struct BinaryBeamNode {
    parent_slot: Option<usize>,
    last_index: usize,
    depth: usize,
    combined: Vec<u64>,
    tp: u64,
    score: f64,
}

#[derive(Clone, Copy, Debug)]
struct BinaryBeamRuntimeOptions {
    use_simd_fast_path: bool,
    use_upper_bound_prune: bool,
}

#[inline]
fn parse_env_bool(name: &str) -> bool {
    std::env::var(name).map_or(false, |v| {
        let t = v.trim().to_ascii_lowercase();
        matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
    })
}

#[inline]
fn parse_env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

#[inline]
fn garfield_layer_rss_debug_enabled() -> bool {
    parse_env_bool("JX_GARFIELD_LAYER_RSS_DEBUG")
}

#[inline]
fn garfield_layer_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_bool("JX_GARFIELD_LAYER_DEBUG"))
}

#[inline]
fn garfield_layer_debug_index(
    layer: usize,
    family: GarfieldLayerDebugFamily,
    metric: GarfieldLayerDebugMetric,
) -> Option<usize> {
    if layer == 0 || layer > GARFIELD_LAYER_DEBUG_MAX_LAYERS {
        return None;
    }
    Some(
        (((layer - 1) * GARFIELD_LAYER_DEBUG_FAMILY_COUNT) + (family as usize))
            * GARFIELD_LAYER_DEBUG_METRIC_COUNT
            + (metric as usize),
    )
}

fn garfield_layer_debug_reset() {
    if !garfield_layer_debug_enabled() {
        return;
    }
    for counter in GARFIELD_LAYER_DEBUG_COUNTS.iter() {
        counter.store(0, Ordering::Relaxed);
    }
}

#[inline]
fn garfield_layer_debug_add(
    layer: usize,
    family: GarfieldLayerDebugFamily,
    metric: GarfieldLayerDebugMetric,
    delta: u64,
) {
    if !garfield_layer_debug_enabled() {
        return;
    }
    if let Some(idx) = garfield_layer_debug_index(layer, family, metric) {
        GARFIELD_LAYER_DEBUG_COUNTS[idx].fetch_add(delta, Ordering::Relaxed);
    }
}

#[inline]
fn garfield_layer_debug_family_name(family: GarfieldLayerDebugFamily) -> &'static str {
    match family {
        GarfieldLayerDebugFamily::Singleton => "singleton",
        GarfieldLayerDebugFamily::And => "and",
        GarfieldLayerDebugFamily::Xor => "xor",
    }
}

#[inline]
fn garfield_layer_debug_rule_family(rule: &BeamRule) -> GarfieldLayerDebugFamily {
    if rule.len() == 1 {
        GarfieldLayerDebugFamily::Singleton
    } else if rule_contains_xor(rule) {
        GarfieldLayerDebugFamily::Xor
    } else {
        GarfieldLayerDebugFamily::And
    }
}

#[inline]
fn garfield_layer_debug_op_family(op: BeamBinaryOp) -> GarfieldLayerDebugFamily {
    match op {
        BeamBinaryOp::Xor => GarfieldLayerDebugFamily::Xor,
        BeamBinaryOp::And | BeamBinaryOp::Or => GarfieldLayerDebugFamily::And,
    }
}

fn garfield_layer_debug_record_fuzzy_states(
    layer: usize,
    metric: GarfieldLayerDebugMetric,
    states: &[FuzzyBeamState],
) {
    if !garfield_layer_debug_enabled() {
        return;
    }
    let mut counts = [0u64; GARFIELD_LAYER_DEBUG_FAMILY_COUNT];
    for state in states.iter() {
        counts[garfield_layer_debug_rule_family(&state.rule) as usize] += 1;
    }
    for (family_idx, count) in counts.iter().copied().enumerate() {
        if count == 0 {
            continue;
        }
        let family = match family_idx {
            0 => GarfieldLayerDebugFamily::Singleton,
            1 => GarfieldLayerDebugFamily::And,
            _ => GarfieldLayerDebugFamily::Xor,
        };
        garfield_layer_debug_add(layer, family, metric, count);
    }
}

fn garfield_layer_debug_dump(mode: &str, max_layer: usize) {
    if !garfield_layer_debug_enabled() {
        return;
    }
    let families = [
        GarfieldLayerDebugFamily::Singleton,
        GarfieldLayerDebugFamily::And,
        GarfieldLayerDebugFamily::Xor,
    ];
    for layer in 1..=max_layer.min(GARFIELD_LAYER_DEBUG_MAX_LAYERS) {
        for family in families.iter().copied() {
            let load = |metric| {
                garfield_layer_debug_index(layer, family, metric)
                    .map(|idx| GARFIELD_LAYER_DEBUG_COUNTS[idx].load(Ordering::Relaxed))
                    .unwrap_or(0)
            };
            let considered = load(GarfieldLayerDebugMetric::Considered);
            let train_ok = load(GarfieldLayerDebugMetric::TrainOk);
            let abs_ok = load(GarfieldLayerDebugMetric::AbsOk);
            let gain_ok = load(GarfieldLayerDebugMetric::GainOk);
            let parent_ok = load(GarfieldLayerDebugMetric::ParentOk);
            let kept = load(GarfieldLayerDebugMetric::Kept);
            let retained = load(GarfieldLayerDebugMetric::Retained);
            if considered == 0
                && train_ok == 0
                && abs_ok == 0
                && gain_ok == 0
                && parent_ok == 0
                && kept == 0
                && retained == 0
            {
                continue;
            }
            eprintln!(
                "[GARFIELD-LAYER-DEBUG] mode={mode} layer={layer} family={} considered={} train_ok={} abs_ok={} gain_ok={} parent_ok={} kept={} retained={}",
                garfield_layer_debug_family_name(family),
                considered,
                train_ok,
                abs_ok,
                gain_ok,
                parent_ok,
                kept,
                retained,
            );
        }
    }
}

#[inline]
fn garfield_layer_rss_limit_bytes() -> Option<u64> {
    parse_env_f64("JX_GARFIELD_LAYER_RSS_LIMIT_GB")
        .map(|gb| (gb * 1024.0_f64 * 1024.0_f64 * 1024.0_f64) as u64)
}

fn garfield_layer_rss_breakpoint(
    mode: &str,
    layer: usize,
    phase: &str,
    frontier_len: usize,
    retained_len: usize,
) -> Result<(), String> {
    let debug_enabled = garfield_layer_rss_debug_enabled();
    let limit_bytes = garfield_layer_rss_limit_bytes();
    if !debug_enabled && limit_bytes.is_none() {
        return Ok(());
    }
    let Some(usage) = process_memory_usage() else {
        return Ok(());
    };
    let rss_txt = usage
        .rss_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "NA".to_string());
    let footprint_txt = usage
        .footprint_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "NA".to_string());
    if debug_enabled {
        eprintln!(
            "[GARFIELD-LAYER-RSS] mode={mode} layer={layer} phase={phase} frontier={frontier_len} retained={retained_len} metric={} current={} rss={} footprint={}",
            usage.metric,
            format_bytes(usage.current_bytes),
            rss_txt,
            footprint_txt,
        );
    }
    if let Some(limit) = limit_bytes {
        if usage.current_bytes > limit {
            return Err(format!(
                "GARFIELD whole-genome memory limit exceeded at layer {layer} ({phase}): {}={} (rss={}, footprint={}, limit={}, frontier={}, retained={}, mode={mode})",
                usage.metric,
                format_bytes(usage.current_bytes),
                rss_txt,
                footprint_txt,
                format_bytes(limit),
                frontier_len,
                retained_len,
            ));
        }
    }
    Ok(())
}

#[inline]
fn beam_force_scalar_runtime() -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    *FORCE_SCALAR.get_or_init(|| parse_env_bool("JANUSX_BEAM_FORCE_SCALAR"))
}

#[inline]
fn beam_disable_bnb_runtime() -> bool {
    static DISABLE_BNB: OnceLock<bool> = OnceLock::new();
    *DISABLE_BNB.get_or_init(|| parse_env_bool("JANUSX_BEAM_DISABLE_BNB"))
}

#[inline]
fn beam_force_bnb_runtime() -> bool {
    static FORCE_BNB: OnceLock<bool> = OnceLock::new();
    *FORCE_BNB.get_or_init(|| parse_env_bool("JANUSX_BEAM_FORCE_BNB"))
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn beam_avx2_runtime_available() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn beam_neon_runtime_available() -> bool {
    static NEON: OnceLock<bool> = OnceLock::new();
    *NEON.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

#[inline]
fn binary_beam_runtime_options() -> BinaryBeamRuntimeOptions {
    BinaryBeamRuntimeOptions {
        use_simd_fast_path: !beam_force_scalar_runtime(),
        use_upper_bound_prune: !beam_disable_bnb_runtime(),
    }
}

#[inline]
fn cmp_nodes(a: &BeamNode, b: &BeamNode) -> std::cmp::Ordering {
    let sa = score_key(a.score);
    let sb = score_key(b.score);
    match sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal) {
        std::cmp::Ordering::Equal => match a.selected.len().cmp(&b.selected.len()) {
            std::cmp::Ordering::Equal => a.selected.cmp(&b.selected),
            other => other,
        },
        other => other,
    }
}

#[inline]
fn push_top_k_streaming(nodes: &mut Vec<BeamNode>, cand: BeamNode, k: usize) {
    if k == 0 {
        return;
    }
    if nodes.len() < k {
        nodes.push(cand);
        return;
    }
    let mut worst_idx = 0usize;
    for i in 1..nodes.len() {
        if cmp_nodes(&nodes[i], &nodes[worst_idx]) == std::cmp::Ordering::Greater {
            worst_idx = i;
        }
    }
    if cmp_nodes(&cand, &nodes[worst_idx]) == std::cmp::Ordering::Less {
        nodes[worst_idx] = cand;
    }
}

#[inline]
fn cmp_binary_nodes(a: &BinaryBeamNode, b: &BinaryBeamNode) -> std::cmp::Ordering {
    let sa = score_key(a.score);
    let sb = score_key(b.score);
    match sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal) {
        std::cmp::Ordering::Equal => match a.depth.cmp(&b.depth) {
            std::cmp::Ordering::Equal => match a.parent_slot.cmp(&b.parent_slot) {
                std::cmp::Ordering::Equal => a.last_index.cmp(&b.last_index),
                other => other,
            },
            other => other,
        },
        other => other,
    }
}

#[inline]
fn push_top_k_streaming_binary(
    nodes: &mut Vec<BinaryBeamNode>,
    cand: BinaryBeamNode,
    k: usize,
) -> bool {
    if k == 0 {
        return false;
    }
    if nodes.len() < k {
        nodes.push(cand);
        return true;
    }
    let mut worst_idx = 0usize;
    for i in 1..nodes.len() {
        if cmp_binary_nodes(&nodes[i], &nodes[worst_idx]) == std::cmp::Ordering::Greater {
            worst_idx = i;
        }
    }
    if cmp_binary_nodes(&cand, &nodes[worst_idx]) == std::cmp::Ordering::Less {
        nodes[worst_idx] = cand;
        return true;
    }
    false
}

#[inline]
fn worst_score_key_in_binary_topk(nodes: &[BinaryBeamNode], k: usize) -> Option<f64> {
    if nodes.len() < k || nodes.is_empty() {
        return None;
    }
    let mut worst = f64::INFINITY;
    for n in nodes {
        let s = score_key(n.score);
        if s < worst {
            worst = s;
        }
    }
    Some(worst)
}

#[inline]
fn mcc_upper_bound_from_tp(tp: u64, y_pos: u64, n_samples: usize) -> f64 {
    let fnv = y_pos.saturating_sub(tp);
    let tn = (n_samples as u64).saturating_sub(y_pos);
    mcc_from_confusion(tp, tn, 0, fnv)
}

#[inline]
fn should_parallel_expand(beam_len: usize, n_rows: usize) -> bool {
    if rayon::current_num_threads() <= 1 || beam_len <= 1 {
        return false;
    }
    beam_len.saturating_mul(n_rows) >= BEAM_PAR_MIN_TOTAL_CANDS
}

#[inline]
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
    if row_words == 0 {
        return Err(format!("{ctx}: row_words must be > 0"));
    }
    if n_samples == 0 {
        return Err(format!("{ctx}: n_samples must be > 0"));
    }
    let needed_words = words_for_samples(n_samples);
    if row_words < needed_words {
        return Err(format!(
            "{ctx}: row_words={} is smaller than required {} for n_samples={}",
            row_words, needed_words, n_samples
        ));
    }
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows * row_words overflow"))?;
    if bits_flat.len() < total_words {
        return Err(format!(
            "{ctx}: bits length={} smaller than n_rows*row_words={}",
            bits_flat.len(),
            total_words
        ));
    }
    Ok(needed_words)
}

#[inline]
fn validate_binary_y(y: &[u8], n_samples: usize, ctx: &str) -> Result<(), String> {
    if y.len() < n_samples {
        return Err(format!(
            "{ctx}: y length={} smaller than n_samples={}",
            y.len(),
            n_samples
        ));
    }
    if let Some((idx, bad)) = y.iter().take(n_samples).enumerate().find(|(_, v)| **v > 1) {
        return Err(format!("{ctx}: y must be binary 0/1; found y[{idx}]={bad}"));
    }
    Ok(())
}

#[inline]
fn pack_binary_y_to_bits(y: &[u8], n_samples: usize) -> (Vec<u64>, u64) {
    let words = words_for_samples(n_samples);
    let mut out = vec![0u64; words];
    let mut y_pos = 0u64;
    for (i, &yv) in y.iter().take(n_samples).enumerate() {
        if yv != 0 {
            out[i >> 6] |= 1u64 << (i & 63);
            y_pos += 1;
        }
    }
    (out, y_pos)
}

#[inline]
fn masked_xpos_tp(bits: &[u64], y_bits: &[u64], n_samples: usize) -> (u64, u64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let mut x_pos = 0u64;
    let mut tp = 0u64;
    if full_words > 0 {
        x_pos += popcount(&bits[..full_words]);
        tp += and_popcount(&bits[..full_words], &y_bits[..full_words]);
    }
    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let xb = bits[full_words] & mask;
        let yb = y_bits[full_words] & mask;
        x_pos += xb.count_ones() as u64;
        tp += (xb & yb).count_ones() as u64;
    }
    (x_pos, tp)
}

#[inline]
fn mcc_from_confusion(tp: u64, tn: u64, fp: u64, fnv: u64) -> f64 {
    let num = (tp as f64) * (tn as f64) - (fp as f64) * (fnv as f64);
    let a = (tp + fp) as f64;
    let b = (tp + fnv) as f64;
    let c = (tn + fp) as f64;
    let d = (tn + fnv) as f64;
    let den = (a * b * c * d).sqrt();
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

#[inline]
fn mcc_from_xpos_tp(x_pos: u64, tp: u64, y_pos: u64, n_samples: usize) -> f64 {
    let fp = x_pos.saturating_sub(tp);
    let fnv = y_pos.saturating_sub(tp);
    let tn = (n_samples as u64).saturating_sub(tp + fp + fnv);
    mcc_from_confusion(tp, tn, fp, fnv)
}

#[inline]
fn abs_corr_from_x_bits(y: &[f64], bits: &[u64], n_samples: usize) -> f64 {
    score_cont_corr_packed(y, bits, n_samples).abs()
}

#[inline]
fn and_assign_xpos_tp_full_words_scalar(
    dst: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
) -> (u64, u64) {
    debug_assert_eq!(dst.len(), rhs.len());
    debug_assert_eq!(dst.len(), y_bits.len());
    let mut x_pos = 0u64;
    let mut tp = 0u64;
    let mut i = 0usize;
    let n = dst.len();
    while i + 4 <= n {
        let v0 = dst[i] & rhs[i];
        let v1 = dst[i + 1] & rhs[i + 1];
        let v2 = dst[i + 2] & rhs[i + 2];
        let v3 = dst[i + 3] & rhs[i + 3];
        dst[i] = v0;
        dst[i + 1] = v1;
        dst[i + 2] = v2;
        dst[i + 3] = v3;
        x_pos += v0.count_ones() as u64
            + v1.count_ones() as u64
            + v2.count_ones() as u64
            + v3.count_ones() as u64;
        tp += (v0 & y_bits[i]).count_ones() as u64
            + (v1 & y_bits[i + 1]).count_ones() as u64
            + (v2 & y_bits[i + 2]).count_ones() as u64
            + (v3 & y_bits[i + 3]).count_ones() as u64;
        i += 4;
    }
    while i < n {
        let v = dst[i] & rhs[i];
        dst[i] = v;
        x_pos += v.count_ones() as u64;
        tp += (v & y_bits[i]).count_ones() as u64;
        i += 1;
    }
    (x_pos, tp)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcount_u8x32_avx2(
    v: core::arch::x86_64::__m256i,
    lut4: core::arch::x86_64::__m256i,
    low_mask: core::arch::x86_64::__m256i,
    zero: core::arch::x86_64::__m256i,
) -> u64 {
    use core::arch::x86_64::*;
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    let cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut4, lo), _mm256_shuffle_epi8(lut4, hi));
    let sum64 = _mm256_sad_epu8(cnt, zero);
    (_mm256_extract_epi64(sum64, 0) as u64)
        + (_mm256_extract_epi64(sum64, 1) as u64)
        + (_mm256_extract_epi64(sum64, 2) as u64)
        + (_mm256_extract_epi64(sum64, 3) as u64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_assign_xpos_tp_full_words_avx2(
    dst: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
) -> (u64, u64) {
    use core::arch::x86_64::*;
    let lut4 = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f_i8);
    let zero = _mm256_setzero_si256();
    let mut x_pos = 0u64;
    let mut tp = 0u64;
    let mut i = 0usize;
    let n = dst.len();
    while i + 4 <= n {
        let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
        let r = _mm256_loadu_si256(rhs.as_ptr().add(i) as *const __m256i);
        let v = _mm256_and_si256(d, r);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, v);
        x_pos += popcount_u8x32_avx2(v, lut4, low_mask, zero);
        let yv = _mm256_loadu_si256(y_bits.as_ptr().add(i) as *const __m256i);
        tp += popcount_u8x32_avx2(_mm256_and_si256(v, yv), lut4, low_mask, zero);
        i += 4;
    }
    let (tx, ttp) = and_assign_xpos_tp_full_words_scalar(&mut dst[i..], &rhs[i..], &y_bits[i..]);
    (x_pos + tx, tp + ttp)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn popcount_u64x2_neon(v: core::arch::aarch64::uint64x2_t) -> u64 {
    use core::arch::aarch64::*;
    let cnt8 = vcntq_u8(vreinterpretq_u8_u64(v));
    let sum16 = vpaddlq_u8(cnt8);
    let sum32 = vpaddlq_u16(sum16);
    let sum64 = vpaddlq_u32(sum32);
    vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1)
}

#[cfg(target_arch = "aarch64")]
unsafe fn and_assign_xpos_tp_full_words_neon(
    dst: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
) -> (u64, u64) {
    use core::arch::aarch64::*;
    let mut x_pos = 0u64;
    let mut tp = 0u64;
    let mut i = 0usize;
    let n = dst.len();
    while i + 2 <= n {
        let d = vld1q_u64(dst.as_ptr().add(i));
        let r = vld1q_u64(rhs.as_ptr().add(i));
        let v = vandq_u64(d, r);
        vst1q_u64(dst.as_mut_ptr().add(i), v);
        x_pos += popcount_u64x2_neon(v);
        let yv = vld1q_u64(y_bits.as_ptr().add(i));
        tp += popcount_u64x2_neon(vandq_u64(v, yv));
        i += 2;
    }
    let (tx, ttp) = and_assign_xpos_tp_full_words_scalar(&mut dst[i..], &rhs[i..], &y_bits[i..]);
    (x_pos + tx, tp + ttp)
}

#[inline]
fn and_assign_xpos_tp_full_words_dispatch(
    dst: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
    use_simd_fast_path: bool,
) -> (u64, u64) {
    if use_simd_fast_path && dst.len() >= BEAM_SIMD_MIN_WORDS {
        #[cfg(target_arch = "x86_64")]
        {
            if beam_avx2_runtime_available() {
                return unsafe { and_assign_xpos_tp_full_words_avx2(dst, rhs, y_bits) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if beam_neon_runtime_available() {
                return unsafe { and_assign_xpos_tp_full_words_neon(dst, rhs, y_bits) };
            }
        }
    }
    and_assign_xpos_tp_full_words_scalar(dst, rhs, y_bits)
}

#[inline]
fn and_assign_xpos_tp_inplace(
    combined: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
    n_samples: usize,
    use_simd_fast_path: bool,
) -> (u64, u64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let mut x_pos = 0u64;
    let mut tp = 0u64;
    if full_words > 0 {
        let (xx, tt) = and_assign_xpos_tp_full_words_dispatch(
            &mut combined[..full_words],
            &rhs[..full_words],
            &y_bits[..full_words],
            use_simd_fast_path,
        );
        x_pos += xx;
        tp += tt;
    }
    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let v = (combined[full_words] & rhs[full_words]) & mask;
        combined[full_words] = v;
        x_pos += v.count_ones() as u64;
        tp += (v & y_bits[full_words]).count_ones() as u64;
    }
    (x_pos, tp)
}

#[inline]
fn estimate_density_for_bnb(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    needed_words: usize,
    n_samples: usize,
) -> f64 {
    let sample_rows = n_rows.min(128);
    if sample_rows == 0 {
        return 1.0;
    }
    let step = (n_rows / sample_rows).max(1);
    let mut ones = 0u64;
    let mut seen = 0usize;
    let mut r = 0usize;
    while r < n_rows && seen < sample_rows {
        let row = row_prefix(bits_flat, row_words, r, needed_words);
        for &w in row {
            ones += w.count_ones() as u64;
        }
        seen += 1;
        r = r.saturating_add(step);
    }
    let denom = (seen as f64) * (n_samples as f64);
    if denom > 0.0 {
        (ones as f64) / denom
    } else {
        1.0
    }
}

#[inline]
fn should_enable_upper_bound_prune(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    needed_words: usize,
    n_samples: usize,
    max_depth: usize,
) -> bool {
    if beam_force_bnb_runtime() {
        return true;
    }
    if max_depth <= 1 || n_rows < BEAM_BNB_MIN_ROWS {
        return false;
    }
    estimate_density_for_bnb(bits_flat, row_words, n_rows, needed_words, n_samples)
        <= BEAM_BNB_DENSITY_MAX
}

#[inline]
fn reconstruct_selected_from_layers(
    layers: &[Vec<(Option<usize>, usize)>],
    best_depth: usize,
    best_slot: usize,
) -> Vec<usize> {
    let mut out_rev = Vec::<usize>::with_capacity(best_depth);
    let mut slot = best_slot;
    for d in (1..=best_depth).rev() {
        let (parent, last) = layers[d - 1][slot];
        out_rev.push(last);
        slot = parent.unwrap_or(0);
    }
    out_rev.reverse();
    out_rev
}

fn beam_search_and_with_score<F>(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    score_fn: F,
) -> Result<BeamAndResult, String>
where
    F: Fn(&[u64]) -> f64 + Sync,
{
    let ctx = "beam_search_and_with_score";
    let needed_words = validate_bit_matrix(bits_flat, row_words, n_rows, n_samples, ctx)?;
    if max_pick == 0 {
        return Err(format!("{ctx}: max_pick must be > 0"));
    }
    if beam_width == 0 {
        return Err(format!("{ctx}: beam_width must be > 0"));
    }
    let max_depth = max_pick.min(n_rows);
    let mask = tail_mask(n_samples);
    let layer_cap = beam_width.min(n_rows);
    let mut beam = Vec::<BeamNode>::with_capacity(layer_cap);
    for i in 0..n_rows {
        let mut combined = row_prefix(bits_flat, row_words, i, needed_words).to_vec();
        apply_tail_mask(&mut combined, mask);
        let score = score_fn(&combined);
        push_top_k_streaming(
            &mut beam,
            BeamNode {
                selected: vec![i],
                combined,
                score,
                last_index: i,
            },
            layer_cap,
        );
    }
    if beam.is_empty() {
        return Err(format!("{ctx}: no candidates"));
    }
    beam.sort_by(cmp_nodes);
    let mut best = beam[0].clone();
    for _depth in 2..=max_depth {
        let next_cap = beam_width.min(n_rows);
        let mut next = if should_parallel_expand(beam.len(), n_rows) {
            let local_tops: Vec<Vec<BeamNode>> = (0..beam.len())
                .into_par_iter()
                .map(|bi| {
                    let node = &beam[bi];
                    let mut local = Vec::<BeamNode>::with_capacity(next_cap);
                    for cand in (node.last_index + 1)..n_rows {
                        let row = row_prefix(bits_flat, row_words, cand, needed_words);
                        let mut combined = node.combined.clone();
                        bitand_assign(&mut combined, row);
                        let score = score_fn(&combined);
                        let mut selected = node.selected.clone();
                        selected.push(cand);
                        push_top_k_streaming(
                            &mut local,
                            BeamNode {
                                selected,
                                combined,
                                score,
                                last_index: cand,
                            },
                            beam_width,
                        );
                    }
                    local
                })
                .collect();
            let mut merged = Vec::<BeamNode>::with_capacity(next_cap);
            for local in local_tops {
                for cand in local {
                    push_top_k_streaming(&mut merged, cand, beam_width);
                }
            }
            merged
        } else {
            let mut seq = Vec::<BeamNode>::with_capacity(next_cap);
            for node in &beam {
                for cand in (node.last_index + 1)..n_rows {
                    let row = row_prefix(bits_flat, row_words, cand, needed_words);
                    let mut combined = node.combined.clone();
                    bitand_assign(&mut combined, row);
                    let score = score_fn(&combined);
                    let mut selected = node.selected.clone();
                    selected.push(cand);
                    push_top_k_streaming(
                        &mut seq,
                        BeamNode {
                            selected,
                            combined,
                            score,
                            last_index: cand,
                        },
                        beam_width,
                    );
                }
            }
            seq
        };
        if next.is_empty() {
            break;
        }
        next.sort_by(cmp_nodes);
        beam = next;
        if cmp_nodes(&beam[0], &best) == std::cmp::Ordering::Less {
            best = beam[0].clone();
        }
    }
    Ok(BeamAndResult {
        selected_indices: best.selected,
        score: best.score,
        combined_bits: best.combined,
    })
}

fn beam_search_and_binary_mcc_with_options(
    y: &[u8],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    options: BinaryBeamRuntimeOptions,
) -> Result<BeamAndResult, String> {
    let ctx = "beam_search_and_binary_mcc";
    validate_binary_y(y, n_samples, ctx)?;
    let needed_words = validate_bit_matrix(bits_flat, row_words, n_rows, n_samples, ctx)?;
    if max_pick == 0 {
        return Err(format!("{ctx}: max_pick must be > 0"));
    }
    if beam_width == 0 {
        return Err(format!("{ctx}: beam_width must be > 0"));
    }
    let max_depth = max_pick.min(n_rows);
    let use_upper_bound_prune = options.use_upper_bound_prune
        && should_enable_upper_bound_prune(
            bits_flat,
            row_words,
            n_rows,
            needed_words,
            n_samples,
            max_depth,
        );
    let (y_bits, y_pos) = pack_binary_y_to_bits(y, n_samples);
    let mask = tail_mask(n_samples);
    let layer_cap = beam_width.min(n_rows);
    let mut beam = Vec::<BinaryBeamNode>::with_capacity(layer_cap);
    for i in 0..n_rows {
        let mut combined = row_prefix(bits_flat, row_words, i, needed_words).to_vec();
        apply_tail_mask(&mut combined, mask);
        let (x_pos, tp) = masked_xpos_tp(&combined, &y_bits, n_samples);
        let score = mcc_from_xpos_tp(x_pos, tp, y_pos, n_samples);
        push_top_k_streaming_binary(
            &mut beam,
            BinaryBeamNode {
                parent_slot: None,
                last_index: i,
                depth: 1,
                combined,
                tp,
                score,
            },
            layer_cap,
        );
    }
    if beam.is_empty() {
        return Err(format!("{ctx}: no candidates"));
    }
    beam.sort_by(cmp_binary_nodes);
    let mut layers = Vec::<Vec<(Option<usize>, usize)>>::with_capacity(max_depth);
    layers.push(
        beam.iter()
            .map(|n| (n.parent_slot, n.last_index))
            .collect::<Vec<_>>(),
    );
    let mut best_depth = 1usize;
    let mut best_slot = 0usize;
    let mut best_score = beam[0].score;
    let mut best_combined = beam[0].combined.clone();
    for depth in 2..=max_depth {
        let next_cap = beam_width.min(n_rows);
        let can_descend_more = depth < max_depth;
        let best_score_cut = score_key(best_score);
        let mut next = if should_parallel_expand(beam.len(), n_rows) {
            let local_tops: Vec<Vec<BinaryBeamNode>> = (0..beam.len())
                .into_par_iter()
                .map(|bi| {
                    let node = &beam[bi];
                    if use_upper_bound_prune {
                        let parent_ub =
                            score_key(mcc_upper_bound_from_tp(node.tp, y_pos, n_samples));
                        if parent_ub <= best_score_cut {
                            return Vec::new();
                        }
                    }
                    let mut local = Vec::<BinaryBeamNode>::with_capacity(next_cap);
                    for cand in (node.last_index + 1)..n_rows {
                        let row = row_prefix(bits_flat, row_words, cand, needed_words);
                        let mut combined = node.combined.clone();
                        let (x_pos, tp) = and_assign_xpos_tp_inplace(
                            &mut combined,
                            row,
                            &y_bits,
                            n_samples,
                            options.use_simd_fast_path,
                        );
                        if use_upper_bound_prune && can_descend_more {
                            let child_ub = score_key(mcc_upper_bound_from_tp(tp, y_pos, n_samples));
                            if child_ub <= best_score_cut {
                                continue;
                            }
                        }
                        let score = mcc_from_xpos_tp(x_pos, tp, y_pos, n_samples);
                        push_top_k_streaming_binary(
                            &mut local,
                            BinaryBeamNode {
                                parent_slot: Some(bi),
                                last_index: cand,
                                depth,
                                combined,
                                tp,
                                score,
                            },
                            beam_width,
                        );
                    }
                    local
                })
                .collect();
            let mut merged = Vec::<BinaryBeamNode>::with_capacity(next_cap);
            for local in local_tops {
                for cand in local {
                    push_top_k_streaming_binary(&mut merged, cand, beam_width);
                }
            }
            merged
        } else {
            let mut seq = Vec::<BinaryBeamNode>::with_capacity(next_cap);
            let mut layer_score_cut = worst_score_key_in_binary_topk(&seq, beam_width);
            for (bi, node) in beam.iter().enumerate() {
                if use_upper_bound_prune {
                    let parent_ub = score_key(mcc_upper_bound_from_tp(node.tp, y_pos, n_samples));
                    if parent_ub <= best_score_cut {
                        continue;
                    }
                    if let Some(cut) = layer_score_cut {
                        if parent_ub < cut {
                            continue;
                        }
                    }
                }
                for cand in (node.last_index + 1)..n_rows {
                    let row = row_prefix(bits_flat, row_words, cand, needed_words);
                    let mut combined = node.combined.clone();
                    let (x_pos, tp) = and_assign_xpos_tp_inplace(
                        &mut combined,
                        row,
                        &y_bits,
                        n_samples,
                        options.use_simd_fast_path,
                    );
                    if use_upper_bound_prune && can_descend_more {
                        let child_ub = score_key(mcc_upper_bound_from_tp(tp, y_pos, n_samples));
                        if child_ub <= best_score_cut {
                            continue;
                        }
                        if let Some(cut) = layer_score_cut {
                            if child_ub < cut {
                                continue;
                            }
                        }
                    }
                    let score = mcc_from_xpos_tp(x_pos, tp, y_pos, n_samples);
                    let inserted = push_top_k_streaming_binary(
                        &mut seq,
                        BinaryBeamNode {
                            parent_slot: Some(bi),
                            last_index: cand,
                            depth,
                            combined,
                            tp,
                            score,
                        },
                        beam_width,
                    );
                    if use_upper_bound_prune && inserted {
                        layer_score_cut = worst_score_key_in_binary_topk(&seq, beam_width);
                    }
                }
            }
            seq
        };
        if next.is_empty() {
            break;
        }
        next.sort_by(cmp_binary_nodes);
        layers.push(
            next.iter()
                .map(|n| (n.parent_slot, n.last_index))
                .collect::<Vec<_>>(),
        );
        let top = &next[0];
        let top_score = score_key(top.score);
        let best_score_key = score_key(best_score);
        if top_score > best_score_key || (top_score == best_score_key && depth < best_depth) {
            best_depth = depth;
            best_slot = 0;
            best_score = top.score;
            best_combined = top.combined.clone();
        }
        beam = next;
    }
    let selected = reconstruct_selected_from_layers(&layers, best_depth, best_slot);
    Ok(BeamAndResult {
        selected_indices: selected,
        score: best_score,
        combined_bits: best_combined,
    })
}

pub fn beam_search_and_binary_mcc(
    y: &[u8],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
) -> Result<BeamAndResult, String> {
    beam_search_and_binary_mcc_with_options(
        y,
        bits_flat,
        row_words,
        n_rows,
        n_samples,
        max_pick,
        beam_width,
        binary_beam_runtime_options(),
    )
}

pub fn beam_search_and_continuous_abs_corr(
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
) -> Result<BeamAndResult, String> {
    let ctx = "beam_search_and_continuous_abs_corr";
    validate_continuous_y(y, n_samples, ctx)?;
    beam_search_and_with_score(
        bits_flat,
        row_words,
        n_rows,
        n_samples,
        max_pick,
        beam_width,
        |combined| abs_corr_from_x_bits(y, combined, n_samples),
    )
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
    !rule.rest.is_empty() && rule.rest.iter().all(|(op, _)| *op == BeamBinaryOp::Or)
}
#[inline]
fn rule_is_pure_and(rule: &BeamRule) -> bool {
    !rule.rest.is_empty() && rule.rest.iter().all(|(op, _)| *op == BeamBinaryOp::And)
}

#[inline]
fn rank_rule_score_components_base(
    rule_len: usize,
    not_count: usize,
    raw_score: f64,
    direct_parent_raw: f64,
    params: &BeamSearchParams,
) -> f64 {
    let use_gain = rank_mode_uses_gain(rule_len, params);
    // A singleton has no interaction parent.  Under the unified gain
    // schedule its gain is therefore defined as its own score; interaction
    // gain starts when the second literal is added.
    let base = if use_gain && rule_len > 1 {
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
    rank_rule_score_components_base(rule_len, not_count, raw_score, direct_parent_raw, params)
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
    rank_rule_score_components_base(rule_len, not_count, raw_score, direct_parent_raw, params)
        - null_penalty_for_bucket(bucket, params, is_train)
}

#[inline]
fn use_parent_delta(rule_len: usize, params: &BeamSearchParams) -> bool {
    if params.disable_parent_delta {
        return false;
    }
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
    let bucket =
        bucket_from_rule_with_complexity(rule, train_raw.dosage_maf, params.null_complexity_bin);
    let abs_score = rank_rule_score_components_base(
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
fn fuzzy_rule_has_dosage_variation(n_ge1: usize, n_ge2: usize, n_samples: usize) -> bool {
    let n0 = n_samples.saturating_sub(n_ge1);
    let n1 = n_ge1.saturating_sub(n_ge2);
    let n2 = n_ge2;
    usize::from(n0 > 0) + usize::from(n1 > 0) + usize::from(n2 > 0) >= 2
}

#[inline]
fn keep_rule_after_dosage_maf_counts(
    n_ge1: usize,
    n_ge2: usize,
    n_samples: usize,
    params: &BeamSearchParams,
) -> bool {
    if !fuzzy_rule_has_dosage_variation(n_ge1, n_ge2, n_samples) {
        return false;
    }
    if !(params.maf_threshold.is_finite() && params.maf_threshold > 0.0) {
        return true;
    }
    dosage_maf_from_dual_counts(n_samples, n_ge1, n_ge2) >= params.maf_threshold
}

#[inline]
fn keep_rule_after_dosage_maf_pruning(sc: &ContinuousRuleScore, params: &BeamSearchParams) -> bool {
    let n_samples = sc.n_hit.saturating_add(sc.n_miss);
    if !fuzzy_rule_has_dosage_variation(sc.n_hit, sc.n_ge2, n_samples) {
        return false;
    }
    if !sc.dosage_maf.is_finite() {
        return false;
    }
    if !(params.maf_threshold.is_finite() && params.maf_threshold > 0.0) {
        return true;
    }
    sc.dosage_maf >= params.maf_threshold
}

#[inline]
fn keep_binary_lmaf_count(n_hit: usize, n_samples: usize, params: &BeamSearchParams) -> bool {
    if n_hit == 0 || n_hit >= n_samples {
        return false;
    }
    if !(params.maf_threshold.is_finite() && params.maf_threshold > 0.0) {
        return true;
    }
    binary_maf_from_n_hit(n_samples, n_hit) >= params.maf_threshold
}

#[inline]
fn keep_xor_substates_binary(
    parent_n_hit: usize,
    row_n_hit: usize,
    intersection_n: usize,
    n_samples: usize,
    negated: bool,
    params: &BeamSearchParams,
) -> bool {
    let (effective_row_n, effective_intersection_n) = if negated {
        (
            n_samples.saturating_sub(row_n_hit),
            parent_n_hit.saturating_sub(intersection_n),
        )
    } else {
        (row_n_hit, intersection_n)
    };
    let parent_and_not_row = parent_n_hit.saturating_sub(effective_intersection_n);
    let not_parent_and_row = effective_row_n.saturating_sub(effective_intersection_n);
    keep_binary_lmaf_count(parent_and_not_row, n_samples, params)
        && keep_binary_lmaf_count(not_parent_and_row, n_samples, params)
}

#[inline]
fn keep_initial_literal_after_seed_pruning(sc: &ContinuousRuleScore) -> bool {
    let n_samples = sc.n_hit.saturating_add(sc.n_miss);
    if !fuzzy_rule_has_dosage_variation(sc.n_hit, sc.n_ge2, n_samples) {
        return false;
    }
    sc.dosage_maf.is_finite()
}

#[derive(Clone, Debug, Default)]
struct FuzzyInitialLiteralStats {
    n_rows: usize,
    n_literals: usize,
    n_variable: usize,
    n_pass_seed_basic: usize,
    n_pass_gain: usize,
    max_dosage_maf: Option<f64>,
}

#[inline]
fn update_fuzzy_initial_literal_stats(
    stats: &mut FuzzyInitialLiteralStats,
    sc: &ContinuousRuleScore,
    pass_seed_basic: bool,
    pass_gain: bool,
) {
    stats.n_literals = stats.n_literals.saturating_add(1);
    let n_samples = sc.n_hit.saturating_add(sc.n_miss);
    if fuzzy_rule_has_dosage_variation(sc.n_hit, sc.n_ge2, n_samples) {
        stats.n_variable = stats.n_variable.saturating_add(1);
    }
    if sc.dosage_maf.is_finite() {
        stats.max_dosage_maf = Some(
            stats
                .max_dosage_maf
                .map(|v| v.max(sc.dosage_maf))
                .unwrap_or(sc.dosage_maf),
        );
    }
    if pass_seed_basic {
        stats.n_pass_seed_basic = stats.n_pass_seed_basic.saturating_add(1);
    }
    if pass_gain {
        stats.n_pass_gain = stats.n_pass_gain.saturating_add(1);
    }
}

#[inline]
fn format_no_valid_initial_literals_fuzzy(
    ctx: &str,
    stats: &FuzzyInitialLiteralStats,
    params: &BeamSearchParams,
) -> String {
    let max_dosage_maf_txt = stats
        .max_dosage_maf
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "NA".to_string());
    format!(
        "{ctx}: no valid initial literals (rows={}, literals={}, variable={}, pass_seed_basic={}, pass_gain={}, combo_lmaf={:.4}, max_singleton_dosage_maf={})",
        stats.n_rows,
        stats.n_literals,
        stats.n_variable,
        stats.n_pass_seed_basic,
        stats.n_pass_gain,
        params.maf_threshold,
        max_dosage_maf_txt
    )
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
    _op: BeamBinaryOp,
    literal: BeamLiteral,
) -> Option<BeamRule> {
    let canonical_op = if let Some((first_op, _)) = parent.rest.first() {
        if *first_op != _op || !parent.rest.iter().all(|(rest_op, _)| *rest_op == *first_op) {
            return None;
        }
        *first_op
    } else {
        _op
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
fn cmp_state_lite(a: &BeamStateLite, b: &BeamStateLite) -> std::cmp::Ordering {
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
pub(crate) fn cmp_candidate(a: &BeamRuleCandidate, b: &BeamRuleCandidate) -> std::cmp::Ordering {
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
fn canonicalize_singleton_output_candidate(
    cand: BeamRuleCandidate,
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> BeamRuleCandidate {
    if cand.rule.len() != 1 || !cand.rule.first.negated {
        return cand;
    }
    let literal = BeamLiteral {
        negated: false,
        ..cand.rule.first
    };
    let rule = BeamRule {
        first: literal,
        rest: Vec::new(),
    };
    let single = literal_scores[literal_score_index(literal.row_index, false)];
    let train = single.train;
    let test = single.test;
    let (_, train_score) = train_scores_for_rule(&rule, train, train.raw_score, None, None, params);
    let bucket =
        bucket_from_rule_with_complexity(&rule, test.dosage_maf, params.null_complexity_bin);
    let test_score = rank_rule_score_components_with_bucket(
        bucket,
        rule.len(),
        rule.not_count(),
        test.raw_score,
        test.raw_score,
        params,
        false,
    );
    BeamRuleCandidate {
        rule,
        train_score,
        test_score,
        train,
        test,
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
fn binary_pair_intersection(
    parent_bits: &[u64],
    row: &[u64],
    y_train: &[f64],
    n_train: usize,
) -> BinaryPairIntersection {
    binary_pair_intersection_with_lookup(parent_bits, row, y_train, n_train, None)
}

#[inline]
fn binary_pair_intersection_with_lookup(
    parent_bits: &[u64],
    row: &[u64],
    y_train: &[f64],
    n_train: usize,
    y_sum_lookup: Option<&PackedYSumLookup>,
) -> BinaryPairIntersection {
    let t_sum = beam_detail_profile_start();
    let (n, sum) = if let Some(lookup) = y_sum_lookup {
        let (n, sum) =
            and_popcount_sum_y_where_both1_with_lookup(parent_bits, row, y_train, n_train, lookup);
        (n as usize, sum)
    } else {
        (
            and_popcount(parent_bits, row) as usize,
            sum_y_where_both1(parent_bits, row, y_train, n_train),
        )
    };
    beam_detail_profile_end(t_sum, &GARFIELD_PROFILE_SUM_Y_BOTH1_NS);
    BinaryPairIntersection { n, sum }
}

#[derive(Clone, Copy, Debug, Default)]
struct BinaryPairIntersection {
    n: usize,
    sum: f64,
}

#[inline]
fn evaluate_child_train_from_parent_virtual_with_intersection(
    parent_train: &ContinuousRuleScore,
    row_train: &ContinuousRuleScore,
    intersection: BinaryPairIntersection,
    sum_y_train: f64,
    n_train: usize,
    _child_rule_len: usize,
    op: BeamBinaryOp,
    negated: bool,
    params: &BeamSearchParams,
) -> Option<ContinuousRuleScore> {
    let parent_n_hit = parent_train.n_hit;
    let parent_sum_hit = score_sum_hit(parent_train);
    let row_n_hit = row_train.n_hit;
    let row_sum_hit = score_sum_hit(row_train);
    let (inter_n_hit, inter_sum_hit) = if negated {
        (
            parent_n_hit.saturating_sub(intersection.n),
            parent_sum_hit - intersection.sum,
        )
    } else {
        (intersection.n, intersection.sum)
    };
    if matches!(op, BeamBinaryOp::Xor)
        && params.filter_xor_substates
        && !keep_xor_substates_binary(
            parent_n_hit,
            row_n_hit,
            intersection.n,
            n_train,
            negated,
            params,
        )
    {
        return None;
    }
    let (child_n_hit, child_sum_hit) = match op {
        BeamBinaryOp::And => (inter_n_hit, inter_sum_hit),
        BeamBinaryOp::Or => (
            parent_n_hit
                .saturating_add(row_n_hit)
                .saturating_sub(inter_n_hit),
            parent_sum_hit + row_sum_hit - inter_sum_hit,
        ),
        BeamBinaryOp::Xor => (
            parent_n_hit
                .saturating_add(row_n_hit)
                .saturating_sub(inter_n_hit.saturating_mul(2)),
            parent_sum_hit + row_sum_hit - (2.0 * inter_sum_hit),
        ),
    };
    let child = score_cont_centered_gain_from_sum_and_n_hit(
        sum_y_train,
        child_sum_hit,
        n_train,
        child_n_hit,
    );
    if !keep_rule_after_dosage_maf_pruning(&child, params) {
        return None;
    }
    Some(child)
}

#[inline]
fn evaluate_child_train_from_parent_virtual(
    parent_bits: &[u64],
    parent_train: &ContinuousRuleScore,
    row: &[u64],
    row_train: &ContinuousRuleScore,
    y_train: &[f64],
    sum_y_train: f64,
    n_train: usize,
    child_rule_len: usize,
    op: BeamBinaryOp,
    negated: bool,
    params: &BeamSearchParams,
) -> Option<ContinuousRuleScore> {
    let intersection = binary_pair_intersection(parent_bits, row, y_train, n_train);
    evaluate_child_train_from_parent_virtual_with_intersection(
        parent_train,
        row_train,
        intersection,
        sum_y_train,
        n_train,
        child_rule_len,
        op,
        negated,
        params,
    )
}

#[inline]
fn keep_child_after_parent_gain_pruning(
    child_rule: &BeamRule,
    child_rank_score: f64,
    params: &BeamSearchParams,
) -> bool {
    keep_state_after_min_gain_pruning(child_rule.len(), child_rank_score, params)
}

/// A child rule must improve on its parent by at least `min_parent_abs_gain`
/// in absolute (unpenalized) score.  This prunes candidates that add a feature
/// with negligible marginal improvement, which both speeds up beam expansion
/// and improves rule quality by filtering noisy extensions.
#[inline]
fn keep_child_after_parent_abs_improvement_pruning(
    parent_abs_score: f64,
    _child_rule_len: usize,
    child_abs_score: f64,
    params: &BeamSearchParams,
) -> bool {
    if !(params.min_parent_abs_gain > 0.0) {
        return true;
    }
    child_abs_score > parent_abs_score + params.min_parent_abs_gain
}

#[inline]
fn gain_threshold_applies(rule_len: usize, params: &BeamSearchParams) -> bool {
    // Layer 1 keeps every valid singleton seed.  With GainFromLayer(1), its
    // rank score is still defined as gain (= the singleton score), but the
    // min-gain filter is only meaningful for interaction extensions and must
    // not discard seeds after the bucket null penalty is applied.
    rule_len > 1 && rank_mode_uses_gain(rule_len, params)
}

#[inline]
fn keep_state_after_min_gain_pruning(
    rule_len: usize,
    train_score: f64,
    params: &BeamSearchParams,
) -> bool {
    if !gain_threshold_applies(rule_len, params) {
        return true;
    }
    let min_gain = if params.min_gain.is_finite() {
        params.min_gain.max(0.0)
    } else {
        0.0
    };
    score_strictly_improves(train_score, min_gain)
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
    let key = rule.lexical_key();
    if local_cache.contains_key(&key) {
        return Ok(());
    }
    let combined = materialize_rule_bits(rule, bits_flat, row_words, n_rows, n_samples)?;
    local_cache.insert(key, combined);
    Ok(())
}

#[inline]
fn cached_rule_bits<'a>(rule: &BeamRule, local_cache: &'a RuleBitsCache) -> Option<&'a [u64]> {
    local_cache
        .get(&rule.lexical_key())
        .map(|bits| bits.as_slice())
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
        rule, y, combined, n_samples, lambda_len, lambda_not,
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

fn best_ancestor_raw_baseline_cached(
    rule: &BeamRule,
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
    base_cache: Option<&RuleRawScoreCache>,
    raw_cache: &mut RuleRawScoreCache,
    ancestor_cache: &mut RuleAncestorBaselineCache,
    disable_parent_delta: bool,
) -> Result<f64, String> {
    let t_profile = beam_detail_profile_start();
    let key = rule.lexical_key();
    if let Some(score) = ancestor_cache.get(&key) {
        return Ok(*score);
    }
    let result: Result<f64, String> = if disable_parent_delta || rule.len() <= 1 {
        Ok(0.0)
    } else if rule.len() == 2 {
        Ok(rule_max_singleton_raw(rule, literal_scores, is_train))
    } else {
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
                raw_cache,
            )?;
            let parent_ancestor = best_ancestor_raw_baseline_cached(
                &parent_rule,
                y,
                bits_flat,
                row_words,
                n_rows,
                n_samples,
                literal_scores,
                is_train,
                base_cache,
                raw_cache,
                ancestor_cache,
                disable_parent_delta,
            )?;
            best = best.max(parent_raw.max(parent_ancestor));
        }
        if best.is_finite() {
            Ok(best)
        } else {
            Ok(0.0)
        }
    };
    let ret = result?;
    ancestor_cache.insert(key, ret);
    beam_detail_profile_end(t_profile, &GARFIELD_PROFILE_PARENT_BASELINE_NS);
    Ok(ret)
}

fn best_ancestor_raw_baseline(
    rule: &BeamRule,
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
    disable_parent_delta: bool,
) -> Result<f64, String> {
    let mut raw_cache = RuleRawScoreCache::new();
    let mut ancestor_cache = RuleAncestorBaselineCache::new();
    best_ancestor_raw_baseline_cached(
        rule,
        y,
        bits_flat,
        row_words,
        n_rows,
        n_samples,
        literal_scores,
        is_train,
        None,
        &mut raw_cache,
        &mut ancestor_cache,
        disable_parent_delta,
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

fn filter_beam_candidates(
    candidates: Vec<BeamState>,
    width: usize,
    params: &BeamSearchParams,
) -> Vec<BeamState> {
    if candidates.is_empty() {
        return candidates;
    }
    let _ = params;
    sort_truncate_states(candidates, width.max(1))
}

#[inline]
fn should_parallel(total_cands: usize, allow_parallel: bool) -> bool {
    allow_parallel
        && rayon::current_num_threads() > 1
        && total_cands >= GARFIELD_BEAM_PAR_MIN_TOTAL_CANDS
}

#[inline]
fn should_parallel_exhaustive(total_cands: usize, allow_parallel: bool) -> bool {
    allow_parallel
        && rayon::current_num_threads() > 1
        && total_cands >= GARFIELD_EXHAUSTIVE_PAR_MIN_TOTAL_CANDS
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

fn materialize_beam_state_lite(
    cand: BeamStateLite,
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    n_train: usize,
) -> Result<BeamState, String> {
    let combined_train =
        materialize_rule_bits(&cand.rule, bits_train, row_words_train, n_rows, n_train)?;
    Ok(beam_state_from_lite_and_bits(combined_train, cand))
}

#[inline]
fn beam_state_into_lite_and_bits(state: BeamState) -> (Vec<u64>, BeamStateLite) {
    let BeamState {
        rule,
        combined_train,
        train,
        train_abs_score,
        train_score,
        max_singleton_train_raw,
        max_singleton_test_raw,
    } = state;
    (
        combined_train,
        BeamStateLite {
            rule,
            train,
            train_abs_score,
            train_score,
            max_singleton_train_raw,
            max_singleton_test_raw,
        },
    )
}

#[inline]
fn beam_state_from_lite_and_bits(combined_train: Vec<u64>, cand: BeamStateLite) -> BeamState {
    BeamState {
        rule: cand.rule,
        combined_train,
        train: cand.train,
        train_abs_score: cand.train_abs_score,
        train_score: cand.train_score,
        max_singleton_train_raw: cand.max_singleton_train_raw,
        max_singleton_test_raw: cand.max_singleton_test_raw,
    }
}

#[inline]
fn parse_env_usize(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
}

#[inline]
fn literal_batch_max_rows() -> usize {
    parse_env_usize("JX_GARFIELD_LITERAL_BATCH_MAX_ROWS")
        .unwrap_or(GARFIELD_LITERAL_BATCH_MAX_ROWS_DEFAULT)
}

#[inline]
fn literal_batch_max_work_words() -> usize {
    parse_env_usize("JX_GARFIELD_LITERAL_BATCH_MAX_WORK_WORDS")
        .unwrap_or(GARFIELD_LITERAL_BATCH_MAX_WORK_WORDS_DEFAULT)
}

#[inline]
fn literal_inputs_are_shared(
    y_train: &[f64],
    n_train: usize,
    y_test: &[f64],
    n_test: usize,
    bits_train: &[u64],
    row_words_train: usize,
    bits_test: &[u64],
    row_words_test: usize,
    n_rows: usize,
) -> bool {
    n_train == n_test
        && row_words_train == row_words_test
        && y_train[..n_train] == y_test[..n_test]
        && bits_train[..n_rows.saturating_mul(row_words_train)]
            == bits_test[..n_rows.saturating_mul(row_words_test)]
}

#[inline]
fn precompute_dual_literal_summaries(
    y: &[f64],
    ge1_flat: &[u64],
    ge2_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    needed_words: usize,
    n_samples: usize,
) -> Vec<DualLiteralSummary> {
    let mut out = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let row_ge1 = row_prefix(ge1_flat, row_words, row_idx, needed_words);
        let row_ge2 = row_prefix(ge2_flat, row_words, row_idx, needed_words);
        let (pos_n_ge1, pos_n_ge2, pos_sum_ge1, pos_sum_ge2) =
            dual_packed_summary(row_ge1, row_ge2, y, n_samples);
        out.push(DualLiteralSummary {
            pos_n_ge1,
            pos_n_ge2,
            pos_sum_ge1,
            pos_sum_ge2,
        });
    }
    out
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
) -> Result<Vec<LiteralSingletonScore>, String> {
    let score_t0 = Instant::now();
    let out = (|| {
        let mode = parse_centered_gain_backend_mode_from_env()?;
        let shared_inputs = needed_words_train == needed_words_test
            && literal_inputs_are_shared(
                y_train,
                n_train,
                y_test,
                n_test,
                bits_train,
                row_words_train,
                bits_test,
                row_words_test,
                n_rows,
            );
        if shared_inputs {
            let shared_scores = precompute_literal_singleton_backend_scores(
                mode,
                y_train,
                sum_y_train,
                bits_train,
                row_words_train,
                needed_words_train,
                n_train,
                n_rows,
            )?;
            return Ok(shared_scores
                .into_iter()
                .map(|score| LiteralSingletonScore {
                    train: score,
                    test: score,
                })
                .collect());
        }
        let train_scores = precompute_literal_singleton_backend_scores(
            mode,
            y_train,
            sum_y_train,
            bits_train,
            row_words_train,
            needed_words_train,
            n_train,
            n_rows,
        )?;
        let test_scores = precompute_literal_singleton_backend_scores(
            mode,
            y_test,
            sum_y_test,
            bits_test,
            row_words_test,
            needed_words_test,
            n_test,
            n_rows,
        )?;

        Ok(train_scores
            .into_iter()
            .zip(test_scores)
            .map(|(train, test)| LiteralSingletonScore { train, test })
            .collect())
    })();
    GARFIELD_BEAM_PROFILE_LITERAL_PRECOMPUTE_NS
        .fetch_add(elapsed_ns_saturating(score_t0), Ordering::Relaxed);
    out
}

pub(crate) fn precompute_literal_singleton_scores_batched(
    y_train: &[f64],
    n_train: usize,
    y_test: &[f64],
    n_test: usize,
    requests: &[LiteralScoreBatchRequest<'_>],
) -> Result<Vec<Vec<LiteralSingletonScore>>, String> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let score_t0 = Instant::now();
    let out = (|| {
        let ctx = "garfield::precompute_literal_singleton_scores_batched";
        validate_continuous_y(y_train, n_train, ctx)?;
        validate_continuous_y(y_test, n_test, ctx)?;
        let mode = parse_centered_gain_backend_mode_from_env()?;
        let sum_y_train = y_train.iter().take(n_train).copied().sum::<f64>();
        let sum_y_test = y_test.iter().take(n_test).copied().sum::<f64>();
        let max_rows = literal_batch_max_rows();
        let max_work_words = literal_batch_max_work_words();
        let mut out = vec![Vec::<LiteralSingletonScore>::new(); requests.len()];
        let mut start = 0usize;

        while start < requests.len() {
            if requests[start].n_rows == 0 {
                start += 1;
                continue;
            }
            let row_words_train = requests[start].row_words_train;
            let row_words_test = requests[start].row_words_test;
            let needed_words_train = validate_bit_matrix(
                requests[start].bits_train,
                row_words_train,
                requests[start].n_rows,
                n_train,
                ctx,
            )?;
            let needed_words_test = validate_bit_matrix(
                requests[start].bits_test,
                row_words_test,
                requests[start].n_rows,
                n_test,
                ctx,
            )?;
            let mut end = start;
            let mut batch_rows = 0usize;
            let mut batch_train_words = 0usize;
            let mut batch_test_words = 0usize;
            while end < requests.len() {
                let req = &requests[end];
                if req.n_rows == 0 {
                    end += 1;
                    continue;
                }
                if req.row_words_train != row_words_train || req.row_words_test != row_words_test {
                    break;
                }
                let req_train_words = req.n_rows.saturating_mul(req.row_words_train);
                let req_test_words = req.n_rows.saturating_mul(req.row_words_test);
                let next_rows = batch_rows.saturating_add(req.n_rows);
                let next_work_words = batch_train_words
                    .saturating_add(req_train_words)
                    .max(batch_test_words.saturating_add(req_test_words));
                if end > start && (next_rows > max_rows || next_work_words > max_work_words) {
                    break;
                }
                validate_bit_matrix(
                    req.bits_train,
                    req.row_words_train,
                    req.n_rows,
                    n_train,
                    ctx,
                )?;
                validate_bit_matrix(req.bits_test, req.row_words_test, req.n_rows, n_test, ctx)?;
                batch_rows = next_rows;
                batch_train_words = batch_train_words.saturating_add(req_train_words);
                batch_test_words = batch_test_words.saturating_add(req_test_words);
                end += 1;
            }
            let batch = &requests[start..end];
            let batch_shared_inputs = needed_words_train == needed_words_test
                && batch.iter().all(|req| {
                    literal_inputs_are_shared(
                        y_train,
                        n_train,
                        y_test,
                        n_test,
                        req.bits_train,
                        req.row_words_train,
                        req.bits_test,
                        req.row_words_test,
                        req.n_rows,
                    )
                });

            let mut merged_train = Vec::<u64>::with_capacity(batch_train_words);
            let mut merged_test = if batch_shared_inputs {
                Vec::<u64>::new()
            } else {
                Vec::<u64>::with_capacity(batch_test_words)
            };
            for req in batch.iter() {
                merged_train.extend_from_slice(
                    &req.bits_train[..req.n_rows.saturating_mul(req.row_words_train)],
                );
                if !batch_shared_inputs {
                    merged_test.extend_from_slice(
                        &req.bits_test[..req.n_rows.saturating_mul(req.row_words_test)],
                    );
                }
            }

            if batch_shared_inputs {
                let shared_scores = precompute_literal_singleton_backend_scores(
                    mode,
                    y_train,
                    sum_y_train,
                    merged_train.as_slice(),
                    row_words_train,
                    needed_words_train,
                    n_train,
                    batch_rows,
                )?;
                let mut row_offset = 0usize;
                for (req_idx, req) in batch.iter().enumerate() {
                    let score_start = row_offset.saturating_mul(2);
                    let score_end = score_start.saturating_add(req.n_rows.saturating_mul(2));
                    out[start + req_idx] = shared_scores[score_start..score_end]
                        .iter()
                        .copied()
                        .map(|score| LiteralSingletonScore {
                            train: score,
                            test: score,
                        })
                        .collect();
                    row_offset = row_offset.saturating_add(req.n_rows);
                }
            } else {
                let train_scores = precompute_literal_singleton_backend_scores(
                    mode,
                    y_train,
                    sum_y_train,
                    merged_train.as_slice(),
                    row_words_train,
                    needed_words_train,
                    n_train,
                    batch_rows,
                )?;
                let test_scores = precompute_literal_singleton_backend_scores(
                    mode,
                    y_test,
                    sum_y_test,
                    merged_test.as_slice(),
                    row_words_test,
                    needed_words_test,
                    n_test,
                    batch_rows,
                )?;
                let mut row_offset = 0usize;
                for (req_idx, req) in batch.iter().enumerate() {
                    let score_start = row_offset.saturating_mul(2);
                    let score_end = score_start.saturating_add(req.n_rows.saturating_mul(2));
                    out[start + req_idx] = train_scores[score_start..score_end]
                        .iter()
                        .copied()
                        .zip(test_scores[score_start..score_end].iter().copied())
                        .map(|(train, test)| LiteralSingletonScore { train, test })
                        .collect();
                    row_offset = row_offset.saturating_add(req.n_rows);
                }
            }
            start = end;
        }
        Ok(out)
    })();
    add_garfield_beam_profile_literal_precompute_ns(elapsed_ns_saturating(score_t0));
    out
}

fn precompute_literal_singleton_backend_scores(
    mode: super::score_gpu::GarfieldCenteredGainBackendMode,
    y: &[f64],
    sum_y: f64,
    bits: &[u64],
    row_words: usize,
    needed_words: usize,
    n_samples: usize,
    n_rows: usize,
) -> Result<Vec<ContinuousRuleScore>, String> {
    let strict = score_cont_centered_gain_singletons_packed_with_backend(
        y, bits, row_words, n_rows, n_samples,
    )
    .map(|v| v.0);
    if !centered_gain_backend_mode_is_auto(mode) {
        return strict;
    }
    strict
        .or_else(|_| {
            score_cont_centered_gain_singletons_packed_cpu_impl(
                y, bits, row_words, n_rows, n_samples,
            )
        })
        .or_else(|_| {
            score_cont_centered_gain_singletons_packed_legacy_impl(
                y, bits, row_words, n_rows, n_samples,
            )
        })
        .or_else(|_| {
            let mut fallback = Vec::with_capacity(n_rows.saturating_mul(2));
            for row_idx in 0..n_rows {
                let row = row_prefix(bits, row_words, row_idx, needed_words);
                for &negated in &[false, true] {
                    let literal_bits = apply_first_literal(row, needed_words, n_samples, negated);
                    let n_hit = support_size_packed(&literal_bits, n_samples);
                    fallback.push(score_cont_centered_gain_packed_with_n_hit(
                        y,
                        &literal_bits,
                        n_samples,
                        sum_y,
                        n_hit,
                    ));
                }
            }
            Ok(fallback)
        })
}

#[inline]
fn bitor_assign(dst: &mut [u64], rhs: &[u64]) {
    debug_assert_eq!(dst.len(), rhs.len());
    for (left, &right) in dst.iter_mut().zip(rhs.iter()) {
        *left |= right;
    }
}

#[inline]
fn bitxor_assign_masked(dst: &mut [u64], rhs: &[u64], n_valid_bits: usize) {
    debug_assert_eq!(dst.len(), rhs.len());
    let needed_words = words_for_samples(n_valid_bits);
    if needed_words == 0 {
        return;
    }
    let full_words = n_valid_bits >> 6;
    let rem = n_valid_bits & 63;
    for i in 0..full_words {
        dst[i] ^= rhs[i];
    }
    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        dst[full_words] ^= rhs[full_words] & mask;
    } else if full_words < needed_words {
        dst[full_words] ^= rhs[full_words];
    }
    apply_tail_mask(dst, tail_mask(n_valid_bits));
}

#[inline]
fn bitxor_not_assign_masked(dst: &mut [u64], rhs: &[u64], n_valid_bits: usize) {
    debug_assert_eq!(dst.len(), rhs.len());
    let needed_words = words_for_samples(n_valid_bits);
    if needed_words == 0 {
        return;
    }
    let full_words = n_valid_bits >> 6;
    let rem = n_valid_bits & 63;
    for i in 0..full_words {
        dst[i] ^= !rhs[i];
    }
    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        dst[full_words] ^= (!rhs[full_words]) & mask;
    } else if full_words < needed_words {
        dst[full_words] ^= !rhs[full_words];
    }
    apply_tail_mask(dst, tail_mask(n_valid_bits));
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
            bitor_assign(dst, row);
            apply_tail_mask(dst, tail_mask(n_samples));
        }
        (BeamBinaryOp::Or, true) => bitor_not_into_masked(dst, row, n_samples),
        (BeamBinaryOp::Xor, false) => bitxor_assign_masked(dst, row, n_samples),
        (BeamBinaryOp::Xor, true) => bitxor_not_assign_masked(dst, row, n_samples),
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
    let mut sc = score_cont_centered_gain_packed_with_sum(
        y,
        combined,
        n_samples,
        y.iter().take(n_samples).copied().sum::<f64>(),
    );
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
    check_interrupt_fast()?;
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
        let local_tops: Vec<Result<Vec<BeamState>, String>> = work
            .into_par_iter()
            .map(|(start, end)| {
                let mut local = Vec::<BeamState>::with_capacity(layer_cap);
                for row_idx in start..end {
                    if ((row_idx - start) & 255) == 0 {
                        check_interrupt_fast()?;
                    }
                    let row = row_prefix(bits_train, row_words_train, row_idx, needed_words_train);
                    // Layer 1 seeds both positive and negated singletons so that
                    // AND-only expansion can still recover former OR hypotheses
                    // via complement forms such as !i & !j.
                    for &negated in initial_singleton_negations(params).iter() {
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
                        if !keep_initial_literal_after_seed_pruning(&train) {
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
                Ok(local)
            })
            .collect();
        let mut merged = Vec::<BeamState>::with_capacity(layer_cap);
        for local in local_tops {
            for cand in local? {
                push_top_k_states(&mut merged, cand, layer_cap);
            }
        }
        merged
    } else {
        let mut seq = Vec::<BeamState>::with_capacity(layer_cap);
        for row_idx in 0..n_rows {
            if (row_idx & 255) == 0 {
                check_interrupt_fast()?;
            }
            let row = row_prefix(bits_train, row_words_train, row_idx, needed_words_train);
            for &negated in initial_singleton_negations(params).iter() {
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
                if !keep_initial_literal_after_seed_pruning(&train) {
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
    check_interrupt_fast()?;
    let total_cands = n_rows.saturating_mul(initial_singleton_negations(params).len());
    let all = if should_parallel_exhaustive(total_cands, params.allow_parallel) {
        let mut work = Vec::<(usize, usize)>::new();
        let chunk = GARFIELD_EXHAUSTIVE_PAR_CHUNK_CANDS.max(1);
        let mut start = 0usize;
        while start < n_rows {
            let end = (start + chunk).min(n_rows);
            work.push((start, end));
            start = end;
        }
        let locals = work
            .into_par_iter()
            .map(|(start, end)| -> Result<Vec<BeamState>, String> {
                let mut local = Vec::<BeamState>::with_capacity(
                    (end - start).saturating_mul(initial_singleton_negations(params).len()),
                );
                for row_idx in start..end {
                    if ((row_idx - start) & 255) == 0 {
                        check_interrupt_fast()?;
                    }
                    let row = row_prefix(bits_train, row_words_train, row_idx, needed_words_train);
                    for &negated in initial_singleton_negations(params).iter() {
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
                        if !keep_initial_literal_after_seed_pruning(&train) {
                            continue;
                        }
                        local.push(BeamState {
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
                Ok(local)
            })
            .collect::<Vec<Result<Vec<BeamState>, String>>>();
        let mut merged = Vec::<BeamState>::with_capacity(n_rows.saturating_mul(2));
        for local in locals {
            merged.extend(local?);
        }
        merged
    } else {
        let mut seq = Vec::<BeamState>::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            if (row_idx & 255) == 0 {
                check_interrupt_fast()?;
            }
            let row = row_prefix(bits_train, row_words_train, row_idx, needed_words_train);
            for &negated in initial_singleton_negations(params).iter() {
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
                if !keep_initial_literal_after_seed_pruning(&train) {
                    continue;
                }
                seq.push(BeamState {
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
        seq
    };
    let out = dedup_states_by_rule_key(all);
    if out.is_empty() {
        return Err(
            "garfield::build_initial_states_exhaustive: no valid initial literals".to_string(),
        );
    }
    Ok(out)
}

fn whole_genome_layer2_parent_variants(
    node: &BeamState,
    bits_train: &[u64],
    row_words_train: usize,
    needed_words_train: usize,
    n_train: usize,
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Vec<BeamState> {
    let mut out = Vec::<BeamState>::with_capacity(2);
    out.push(node.clone());
    if node.rule.len() != 1 || node.rule.first.negated {
        return out;
    }
    let row_idx = node.rule.first.row_index;
    let row = row_prefix(bits_train, row_words_train, row_idx, needed_words_train);
    let literal = BeamLiteral {
        negated: true,
        ..node.rule.first
    };
    let rule = BeamRule {
        first: literal,
        rest: Vec::new(),
    };
    let combined = apply_first_literal(row, needed_words_train, n_train, true);
    let single = literal_scores[literal_score_index(row_idx, true)];
    let train = single.train;
    let (train_abs_score, train_score) =
        train_scores_for_rule(&rule, train, train.raw_score, None, None, params);
    if keep_initial_literal_after_seed_pruning(&train)
        && keep_state_after_min_gain_pruning(rule.len(), train_score, params)
    {
        out.push(BeamState {
            rule,
            combined_train: combined,
            train,
            train_abs_score,
            train_score,
            max_singleton_train_raw: single.train.raw_score,
            max_singleton_test_raw: single.test.raw_score,
        });
    }
    out
}

#[inline]
fn whole_genome_target_work_ranges(n_rows: usize) -> Vec<(usize, usize)> {
    if n_rows == 0 {
        return Vec::new();
    }
    let n_workers = rayon::current_num_threads().max(1).min(n_rows.max(1));
    let chunk = n_rows.div_ceil(n_workers).max(1);
    let mut out = Vec::<(usize, usize)>::with_capacity(n_workers);
    let mut start = 0usize;
    while start < n_rows {
        let end = (start + chunk).min(n_rows);
        out.push((start, end));
        start = end;
    }
    out
}

#[inline]
fn prune_best_state_map(best: &mut HashMap<RuleLexKey, BeamStateLite>, keep: usize) {
    if keep == 0 || best.len() <= keep {
        return;
    }
    let mut states = best.drain().map(|(_, state)| state).collect::<Vec<_>>();
    states.sort_unstable_by(cmp_state_lite);
    states.truncate(keep);
    for state in states.into_iter() {
        best.insert(state.rule.lexical_key(), state);
    }
}

fn merge_best_state_maps(
    worker_maps: Vec<Result<HashMap<RuleLexKey, BeamStateLite>, String>>,
    next_cap: usize,
) -> Result<Vec<BeamStateLite>, String> {
    let mut global_best: HashMap<RuleLexKey, BeamStateLite> =
        HashMap::with_capacity(next_cap.saturating_mul(2));
    for worker_map in worker_maps {
        for (key, state) in worker_map? {
            match global_best.entry(key) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(state);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    if cmp_state_lite(&state, slot.get()) == std::cmp::Ordering::Less {
                        slot.insert(state);
                    }
                }
            }
        }
        if global_best.len() > next_cap.saturating_mul(4).max(next_cap.saturating_add(1)) {
            prune_best_state_map(&mut global_best, next_cap);
        }
    }
    let mut out = global_best.into_values().collect::<Vec<_>>();
    out.sort_unstable_by(cmp_state_lite);
    if out.len() > next_cap {
        out.truncate(next_cap);
    }
    Ok(out)
}

fn expand_beam_once_whole_genome_target_range(
    parents: &[BeamState],
    start: usize,
    end: usize,
    y_train: &[f64],
    sum_y_train: f64,
    bits_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    base_rule_raws: &RuleRawScoreCache,
    next_cap: usize,
    params: &BeamSearchParams,
) -> Result<HashMap<RuleLexKey, BeamStateLite>, String> {
    let mut local_best = HashMap::<RuleLexKey, BeamStateLite>::with_capacity(next_cap.max(1));
    // Whole-genome target scans touch too many unique rules to keep a
    // worker-lifetime ancestor cache. Reuse the maps per target SNP only.
    let mut parent_raw_cache =
        RuleRawScoreCache::with_capacity(parents.len().saturating_mul(8).max(32));
    let mut ancestor_raw_cache =
        RuleAncestorBaselineCache::with_capacity(parents.len().saturating_mul(8).max(32));
    let prune_trigger = next_cap.saturating_mul(4).max(next_cap.saturating_add(1));
    for cand in start..end {
        if ((cand - start) & 127) == 0 {
            check_interrupt_fast()?;
        }
        parent_raw_cache.clear();
        ancestor_raw_cache.clear();
        let gid = group_ids[cand];
        let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
        for parent in parents.iter() {
            if candidate_group_is_excluded(&parent.rule, gid, params) {
                continue;
            }
            let intersection = binary_pair_intersection_with_lookup(
                parent.combined_train.as_slice(),
                row,
                y_train,
                n_train,
                params.y_sum_lookup.as_deref(),
            );
            for &op in beam_binary_ops_for_rule(&parent.rule).iter() {
                for &negated in child_literal_negations_for_op(op).iter() {
                    let literal = BeamLiteral {
                        row_index: cand,
                        group_id: gid,
                        negated,
                    };
                    let single = literal_scores[literal_score_index(cand, negated)];
                    let Some(train) = evaluate_child_train_from_parent_virtual_with_intersection(
                        &parent.train,
                        &single.train,
                        intersection,
                        sum_y_train,
                        n_train,
                        parent.rule.len() + 1,
                        op,
                        negated,
                        params,
                    ) else {
                        continue;
                    };
                    let canonical_rule =
                        canonical_commutative_child_rule(&parent.rule, op, literal);
                    let rule = if let Some(rule) = canonical_rule {
                        rule
                    } else {
                        let mut rule = parent.rule.clone();
                        rule.rest.push((op, literal));
                        rule
                    };
                    let max_singleton_train_raw =
                        parent.max_singleton_train_raw.max(single.train.raw_score);
                    let max_singleton_test_raw =
                        parent.max_singleton_test_raw.max(single.test.raw_score);
                    let direct_parent_train_raw = if rule.len() == 2 {
                        parent.train.raw_score.max(single.train.raw_score)
                    } else {
                        best_ancestor_raw_baseline_cached(
                            &rule,
                            y_train,
                            bits_train,
                            row_words_train,
                            n_rows,
                            n_train,
                            literal_scores,
                            true,
                            Some(base_rule_raws),
                            &mut parent_raw_cache,
                            &mut ancestor_raw_cache,
                            params.disable_parent_delta,
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
                        parent.train_abs_score,
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
                    let state = BeamStateLite {
                        rule,
                        train,
                        train_abs_score,
                        train_score,
                        max_singleton_train_raw,
                        max_singleton_test_raw,
                    };
                    let key = state.rule.lexical_key();
                    match local_best.entry(key) {
                        std::collections::hash_map::Entry::Vacant(slot) => {
                            slot.insert(state);
                        }
                        std::collections::hash_map::Entry::Occupied(mut slot) => {
                            if cmp_state_lite(&state, slot.get()) == std::cmp::Ordering::Less {
                                slot.insert(state);
                            }
                        }
                    }
                }
            }
        }
        if local_best.len() > prune_trigger {
            prune_best_state_map(&mut local_best, next_cap);
        }
    }
    prune_best_state_map(&mut local_best, next_cap);
    Ok(local_best)
}

fn expand_beam_once_whole_genome_target_parallel(
    parents: &[BeamState],
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
    check_interrupt_fast()?;
    let next_cap = params.beam_width.min(n_rows.saturating_mul(4).max(1));
    if parents.is_empty() {
        return Ok(Vec::new());
    }
    let base_rule_raws = Arc::new(collect_known_rule_raw_scores(parents));
    let total_expand = parents
        .iter()
        .map(|parent| n_rows.saturating_mul(beam_child_branch_count_for_rule(&parent.rule)))
        .sum::<usize>();
    let next = if should_parallel(total_expand, params.allow_parallel) {
        let work = whole_genome_target_work_ranges(n_rows);
        let worker_maps = work
            .into_par_iter()
            .map(|(start, end)| {
                expand_beam_once_whole_genome_target_range(
                    parents,
                    start,
                    end,
                    y_train,
                    sum_y_train,
                    bits_train,
                    row_words_train,
                    n_rows,
                    needed_words_train,
                    n_train,
                    group_ids,
                    literal_scores,
                    base_rule_raws.as_ref(),
                    next_cap,
                    params,
                )
            })
            .collect::<Vec<Result<HashMap<RuleLexKey, BeamStateLite>, String>>>();
        merge_best_state_maps(worker_maps, next_cap)?
    } else {
        expand_beam_once_whole_genome_target_range(
            parents,
            0,
            n_rows,
            y_train,
            sum_y_train,
            bits_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
            group_ids,
            literal_scores,
            base_rule_raws.as_ref(),
            next_cap,
            params,
        )?
        .into_values()
        .collect::<Vec<_>>()
    };
    let mut materialized = Vec::<BeamState>::with_capacity(next.len());
    for cand in next.into_iter() {
        materialized.push(materialize_beam_state_lite(
            cand,
            bits_train,
            row_words_train,
            n_rows,
            n_train,
        )?);
    }
    Ok(filter_beam_candidates(materialized, next_cap, params))
}

fn expand_beam_once_whole_genome_layer2(
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
    check_interrupt_fast()?;
    let mut parents = Vec::<BeamState>::with_capacity(beam.len().saturating_mul(2));
    for node in beam.iter() {
        parents.extend(whole_genome_layer2_parent_variants(
            node,
            bits_train,
            row_words_train,
            needed_words_train,
            n_train,
            literal_scores,
            params,
        ));
    }
    expand_beam_once_whole_genome_target_parallel(
        parents.as_slice(),
        y_train,
        sum_y_train,
        bits_train,
        row_words_train,
        n_rows,
        needed_words_train,
        n_train,
        group_ids,
        literal_scores,
        params,
    )
}

fn expand_beam_once_parallel_deferred(
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
    check_interrupt_fast()?;
    let next_cap = params.beam_width.min(n_rows.saturating_mul(4).max(1));
    let base_rule_raws = Arc::new(collect_known_rule_raw_scores(beam));
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

    // Keep candidate scoring and pruning unchanged, but defer bitset
    // materialization until after global rule deduplication and top-k.
    let worker_maps = work
        .into_par_iter()
        .map(|(bi, start, end)| {
            let node = &beam[bi];
            let mut local_best: HashMap<RuleLexKey, BeamStateLite> =
                HashMap::with_capacity(next_cap);
            let mut parent_raw_cache = RuleRawScoreCache::new();
            let mut ancestor_raw_cache = RuleAncestorBaselineCache::new();
            for cand in start..end {
                if ((cand - start) & 127) == 0 {
                    check_interrupt_fast()?;
                }
                let gid = group_ids[cand];
                if candidate_group_is_excluded(&node.rule, gid, params) {
                    continue;
                }
                let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
                let intersection = binary_pair_intersection_with_lookup(
                    &node.combined_train,
                    row,
                    y_train,
                    n_train,
                    params.y_sum_lookup.as_deref(),
                );
                for &op in beam_binary_ops_for_rule(&node.rule).iter() {
                    for &negated in child_literal_negations_for_op(op).iter() {
                        let literal = BeamLiteral {
                            row_index: cand,
                            group_id: gid,
                            negated,
                        };
                        let single = literal_scores[literal_score_index(cand, negated)];
                        let canonical_rule =
                            canonical_commutative_child_rule(&node.rule, op, literal);
                        let Some(train) =
                            evaluate_child_train_from_parent_virtual_with_intersection(
                                &node.train,
                                &single.train,
                                intersection,
                                sum_y_train,
                                n_train,
                                node.rule.len() + 1,
                                op,
                                negated,
                                params,
                            )
                        else {
                            continue;
                        };
                        let rule = if let Some(rule) = canonical_rule {
                            rule
                        } else {
                            let mut rule = node.rule.clone();
                            rule.rest.push((op, literal));
                            rule
                        };
                        let max_singleton_train_raw =
                            node.max_singleton_train_raw.max(single.train.raw_score);
                        let max_singleton_test_raw =
                            node.max_singleton_test_raw.max(single.test.raw_score);
                        let direct_parent_train_raw = if rule.len() == 2 {
                            node.train.raw_score.max(single.train.raw_score)
                        } else {
                            best_ancestor_raw_baseline_cached(
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
                                &mut ancestor_raw_cache,
                                params.disable_parent_delta,
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
                        let state = BeamStateLite {
                            rule,
                            train,
                            train_abs_score,
                            train_score,
                            max_singleton_train_raw,
                            max_singleton_test_raw,
                        };
                        let key = state.rule.lexical_key();
                        match local_best.entry(key) {
                            std::collections::hash_map::Entry::Vacant(slot) => {
                                slot.insert(state);
                            }
                            std::collections::hash_map::Entry::Occupied(mut slot) => {
                                if cmp_state_lite(&state, slot.get()) == std::cmp::Ordering::Less {
                                    slot.insert(state);
                                }
                            }
                        }
                    }
                }
            }
            Ok(local_best)
        })
        .collect::<Vec<Result<HashMap<RuleLexKey, BeamStateLite>, String>>>();

    let mut global_best =
        HashMap::<RuleLexKey, BeamStateLite>::with_capacity(next_cap.saturating_mul(2));
    for worker_map in worker_maps {
        for (key, state) in worker_map? {
            match global_best.entry(key) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(state);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    if cmp_state_lite(&state, slot.get()) == std::cmp::Ordering::Less {
                        slot.insert(state);
                    }
                }
            }
        }
    }

    let mut retained = global_best.into_values().collect::<Vec<_>>();
    retained.sort_unstable_by(cmp_state_lite);
    if retained.len() > next_cap {
        retained.truncate(next_cap);
    }
    let mut materialized = Vec::<BeamState>::with_capacity(retained.len());
    for state in retained {
        materialized.push(materialize_beam_state_lite(
            state,
            bits_train,
            row_words_train,
            n_rows,
            n_train,
        )?);
    }
    Ok(filter_beam_candidates(materialized, next_cap, params))
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
    check_interrupt_fast()?;
    let next_cap = params.beam_width.min(n_rows.saturating_mul(4).max(1));
    let base_rule_raws = Arc::new(collect_known_rule_raw_scores(beam));
    let total_expand = beam
        .iter()
        .map(|node| {
            let (start, end) = expansion_row_bounds(&node.rule, n_rows);
            end.saturating_sub(start)
                .saturating_mul(beam_child_branch_count_for_rule(&node.rule))
        })
        .sum::<usize>();

    if should_parallel(total_expand, params.allow_parallel) {
        return expand_beam_once_parallel_deferred(
            beam,
            y_train,
            sum_y_train,
            bits_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
            group_ids,
            literal_scores,
            params,
        );
    }

    let next = {
        let mut seen_commutative_children = HashSet::<Vec<(usize, bool, u8)>>::new();
        let mut seq = Vec::<BeamState>::with_capacity(next_cap);
        let mut parent_raw_cache = RuleRawScoreCache::new();
        let mut ancestor_raw_cache = RuleAncestorBaselineCache::new();
        for node in beam.iter() {
            let (start, end) = expansion_row_bounds(&node.rule, n_rows);
            let blind_scan = child_rule_uses_blind_scan(node.rule.len());
            for cand in start..end {
                if ((cand - start) & 127) == 0 {
                    check_interrupt_fast()?;
                }
                let gid = group_ids[cand];
                if candidate_group_is_excluded(&node.rule, gid, params) {
                    continue;
                }
                let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
                let intersection = binary_pair_intersection_with_lookup(
                    &node.combined_train,
                    row,
                    y_train,
                    n_train,
                    params.y_sum_lookup.as_deref(),
                );
                for &op in beam_binary_ops_for_rule(&node.rule).iter() {
                    for &negated in child_literal_negations_for_op(op).iter() {
                        let literal = BeamLiteral {
                            row_index: cand,
                            group_id: gid,
                            negated,
                        };
                        let single = literal_scores[literal_score_index(cand, negated)];
                        let canonical_rule =
                            canonical_commutative_child_rule(&node.rule, op, literal);
                        if blind_scan {
                            if let Some(rule) = canonical_rule.as_ref() {
                                if !seen_commutative_children.insert(rule.lexical_key()) {
                                    continue;
                                }
                            }
                        }
                        let Some(train) =
                            evaluate_child_train_from_parent_virtual_with_intersection(
                                &node.train,
                                &single.train,
                                intersection,
                                sum_y_train,
                                n_train,
                                node.rule.len() + 1,
                                op,
                                negated,
                                params,
                            )
                        else {
                            continue;
                        };
                        let rule = if let Some(rule) = canonical_rule {
                            rule
                        } else {
                            let mut rule = node.rule.clone();
                            rule.rest.push((op, literal));
                            rule
                        };
                        let max_singleton_train_raw =
                            node.max_singleton_train_raw.max(single.train.raw_score);
                        let max_singleton_test_raw =
                            node.max_singleton_test_raw.max(single.test.raw_score);
                        let direct_parent_train_raw = if rule.len() == 2 {
                            node.train.raw_score.max(single.train.raw_score)
                        } else {
                            best_ancestor_raw_baseline_cached(
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
                                &mut ancestor_raw_cache,
                                params.disable_parent_delta,
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

/// Dedup by canonical rule key.  Safer than train-bits dedup for the
/// exhaustive frontier because two rules with identical support sets may
/// still have different expansion possibilities (different last_row_index,
/// group membership, etc.).  Only used inside the frontier loop; the final
/// output contraction still uses bitset dedup.
fn dedup_states_by_rule_key(states: Vec<BeamState>) -> Vec<BeamState> {
    let mut best = HashMap::<RuleLexKey, BeamState>::with_capacity(states.len());
    for state in states.into_iter() {
        let key = state.rule.lexical_key();
        match best.entry(key) {
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

fn dedup_states_by_train_bits(states: Vec<BeamState>) -> Vec<BeamState> {
    let mut best = HashMap::<Vec<u64>, BeamStateLite>::with_capacity(states.len());
    for state in states.into_iter() {
        let (train_bits, state) = beam_state_into_lite_and_bits(state);
        match best.entry(train_bits) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(state);
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if cmp_state_lite(&state, slot.get()) == std::cmp::Ordering::Less {
                    slot.insert(state);
                }
            }
        }
    }
    let mut out = best
        .into_iter()
        .map(|(train_bits, state)| beam_state_from_lite_and_bits(train_bits, state))
        .collect::<Vec<_>>();
    out.sort_by(cmp_state);
    out
}

fn dedup_states_by_support_signature(
    states: Vec<BeamState>,
    bits_test: &[u64],
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    train_test_shared: bool,
) -> Result<Vec<BeamState>, String> {
    if train_test_shared {
        return Ok(dedup_states_by_train_bits(states));
    }

    let mut groups = HashMap::<Vec<u64>, Vec<BeamStateLite>>::with_capacity(states.len());
    for state in states.into_iter() {
        let (train_bits, state) = beam_state_into_lite_and_bits(state);
        groups.entry(train_bits).or_default().push(state);
    }

    let mut out = Vec::<BeamState>::with_capacity(groups.len());
    for (train_bits, grouped_states) in groups.into_iter() {
        if grouped_states.len() == 1 {
            let state = grouped_states.into_iter().next().unwrap();
            out.push(beam_state_from_lite_and_bits(train_bits, state));
            continue;
        }

        let mut best = HashMap::<Vec<u64>, BeamStateLite>::with_capacity(grouped_states.len());
        for state in grouped_states.into_iter() {
            let test_bits =
                materialize_rule_bits(&state.rule, bits_test, row_words_test, n_rows, n_test)?;
            match best.entry(test_bits) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(state);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    if cmp_state_lite(&state, slot.get()) == std::cmp::Ordering::Less {
                        slot.insert(state);
                    }
                }
            }
        }

        let retained_len = best.len();
        let mut owned_train_bits = Some(train_bits);
        for (idx, (_, state)) in best.into_iter().enumerate() {
            let combined_train = if idx + 1 == retained_len {
                owned_train_bits.take().unwrap()
            } else {
                owned_train_bits.as_ref().unwrap().clone()
            };
            out.push(beam_state_from_lite_and_bits(combined_train, state));
        }
    }
    out.sort_by(cmp_state);
    Ok(out)
}

#[inline]
fn same_rule(a: &BeamRule, b: &BeamRule) -> bool {
    a == b
}

fn expand_states_exhaustive_parallel_deferred(
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
    check_interrupt_fast()?;
    let base_rule_raws = Arc::new(collect_known_rule_raw_scores(frontier));
    let mut work = Vec::<(usize, usize, usize)>::new();
    let chunk = GARFIELD_EXHAUSTIVE_PAR_CHUNK_CANDS.max(1);
    for (bi, node) in frontier.iter().enumerate() {
        let mut start = node.rule.last_row_index() + 1;
        while start < n_rows {
            let end = (start + chunk).min(n_rows);
            work.push((bi, start, end));
            start = end;
        }
    }

    // Exhaustive seed layers must retain every valid rule. Keep only the
    // score-bearing state in workers and materialize support bits once after
    // global rule-key deduplication.
    let worker_states = work
        .into_par_iter()
        .map(|(bi, start, end)| -> Result<Vec<BeamStateLite>, String> {
            let node = &frontier[bi];
            let mut local_states = Vec::<BeamStateLite>::new();
            let mut parent_raw_cache = RuleRawScoreCache::new();
            let mut ancestor_raw_cache = RuleAncestorBaselineCache::new();
            for cand in start..end {
                if ((cand - start) & 127) == 0 {
                    check_interrupt_fast()?;
                }
                let gid = group_ids[cand];
                if candidate_group_is_excluded(&node.rule, gid, params) {
                    continue;
                }
                let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
                let intersection = binary_pair_intersection_with_lookup(
                    &node.combined_train,
                    row,
                    y_train,
                    n_train,
                    params.y_sum_lookup.as_deref(),
                );
                for &op in beam_binary_ops_for_rule(&node.rule).iter() {
                    for &negated in child_literal_negations_for_op(op).iter() {
                        let single = literal_scores[literal_score_index(cand, negated)];
                        let Some(train) =
                            evaluate_child_train_from_parent_virtual_with_intersection(
                                &node.train,
                                &single.train,
                                intersection,
                                sum_y_train,
                                n_train,
                                node.rule.len() + 1,
                                op,
                                negated,
                                params,
                            )
                        else {
                            continue;
                        };
                        let literal = BeamLiteral {
                            row_index: cand,
                            group_id: gid,
                            negated,
                        };
                        let mut rule = node.rule.clone();
                        rule.rest.push((op, literal));
                        let max_singleton_train_raw =
                            node.max_singleton_train_raw.max(single.train.raw_score);
                        let max_singleton_test_raw =
                            node.max_singleton_test_raw.max(single.test.raw_score);
                        let direct_parent_train_raw = if rule.len() == 2 {
                            node.train.raw_score.max(single.train.raw_score)
                        } else {
                            best_ancestor_raw_baseline_cached(
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
                                &mut ancestor_raw_cache,
                                params.disable_parent_delta,
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
                        let state = BeamStateLite {
                            rule,
                            train,
                            train_abs_score,
                            train_score,
                            max_singleton_train_raw,
                            max_singleton_test_raw,
                        };
                        // Exhaustive expansion appends a strictly larger row index to a
                        // unique parent rule. The parent, candidate, operation, and
                        // negation therefore identify a unique rule; defer sorting until
                        // after all workers finish instead of allocating a lexical-key map
                        // for every candidate.
                        local_states.push(state);
                    }
                }
            }
            Ok(local_states)
        })
        .collect::<Vec<Result<Vec<BeamStateLite>, String>>>();

    let state_count = worker_states
        .iter()
        .filter_map(|states| states.as_ref().ok())
        .map(Vec::len)
        .sum::<usize>();
    let mut out = Vec::<BeamState>::with_capacity(state_count);
    for worker_state in worker_states {
        for state in worker_state? {
            out.push(materialize_beam_state_lite(
                state,
                bits_train,
                row_words_train,
                n_rows,
                n_train,
            )?);
        }
    }
    out.sort_by(cmp_state);
    Ok(out)
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
    check_interrupt_fast()?;
    let total_expand = frontier
        .iter()
        .map(|node| {
            let cand_start = node.rule.last_row_index() + 1;
            n_rows
                .saturating_sub(cand_start)
                .saturating_mul(beam_child_branch_count_for_rule(&node.rule))
        })
        .sum::<usize>();
    let base_rule_raws = Arc::new(collect_known_rule_raw_scores(frontier));
    if should_parallel_exhaustive(total_expand, params.allow_parallel) {
        return expand_states_exhaustive_parallel_deferred(
            frontier,
            y_train,
            sum_y_train,
            bits_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
            group_ids,
            literal_scores,
            params,
        );
    }
    let out = if should_parallel_exhaustive(total_expand, params.allow_parallel) {
        let mut work = Vec::<(usize, usize, usize)>::new();
        let chunk = GARFIELD_EXHAUSTIVE_PAR_CHUNK_CANDS.max(1);
        for (bi, node) in frontier.iter().enumerate() {
            let mut start = node.rule.last_row_index() + 1;
            while start < n_rows {
                let end = (start + chunk).min(n_rows);
                work.push((bi, start, end));
                start = end;
            }
        }
        let worker_maps = work
            .into_par_iter()
            .map(
                |(bi, start, end)| -> Result<HashMap<RuleLexKey, BeamState>, String> {
                    let node = &frontier[bi];
                    let mut local_best = HashMap::<RuleLexKey, BeamState>::new();
                    let mut parent_raw_cache = RuleRawScoreCache::new();
                    let mut ancestor_raw_cache = RuleAncestorBaselineCache::new();
                    for cand in start..end {
                        if ((cand - start) & 127) == 0 {
                            check_interrupt_fast()?;
                        }
                        let gid = group_ids[cand];
                        if candidate_group_is_excluded(&node.rule, gid, params) {
                            continue;
                        }
                        let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
                        let intersection = binary_pair_intersection_with_lookup(
                            &node.combined_train,
                            row,
                            y_train,
                            n_train,
                            params.y_sum_lookup.as_deref(),
                        );
                        for &op in beam_binary_ops_for_rule(&node.rule).iter() {
                            for &negated in child_literal_negations_for_op(op).iter() {
                                let single = literal_scores[literal_score_index(cand, negated)];
                                let Some(train) =
                                    evaluate_child_train_from_parent_virtual_with_intersection(
                                        &node.train,
                                        &single.train,
                                        intersection,
                                        sum_y_train,
                                        n_train,
                                        node.rule.len() + 1,
                                        op,
                                        negated,
                                        params,
                                    )
                                else {
                                    continue;
                                };
                                let literal = BeamLiteral {
                                    row_index: cand,
                                    group_id: gid,
                                    negated,
                                };
                                let mut rule = node.rule.clone();
                                rule.rest.push((op, literal));
                                let max_singleton_train_raw =
                                    node.max_singleton_train_raw.max(single.train.raw_score);
                                let max_singleton_test_raw =
                                    node.max_singleton_test_raw.max(single.test.raw_score);
                                let direct_parent_train_raw = if rule.len() == 2 {
                                    node.train.raw_score.max(single.train.raw_score)
                                } else {
                                    best_ancestor_raw_baseline_cached(
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
                                        &mut ancestor_raw_cache,
                                        params.disable_parent_delta,
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
                                if !keep_state_after_min_gain_pruning(
                                    rule.len(),
                                    train_score,
                                    params,
                                ) {
                                    continue;
                                }
                                if !keep_child_after_parent_gain_pruning(&rule, train_score, params)
                                {
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
                                match local_best.entry(state.rule.lexical_key()) {
                                    std::collections::hash_map::Entry::Vacant(slot) => {
                                        slot.insert(state);
                                    }
                                    std::collections::hash_map::Entry::Occupied(mut slot) => {
                                        if cmp_state(&state, slot.get()) == std::cmp::Ordering::Less
                                        {
                                            slot.insert(state);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(local_best)
                },
            )
            .collect::<Vec<Result<HashMap<RuleLexKey, BeamState>, String>>>();
        let mut best = HashMap::<RuleLexKey, BeamState>::new();
        for wm in worker_maps {
            for (key, state) in wm? {
                match best.entry(key) {
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
        best.into_values().collect::<Vec<_>>()
    } else {
        let mut best = HashMap::<RuleLexKey, BeamState>::new();
        let mut parent_raw_cache = RuleRawScoreCache::new();
        let mut ancestor_raw_cache = RuleAncestorBaselineCache::new();
        for node in frontier.iter() {
            let cand_start = node.rule.last_row_index() + 1;
            for cand in cand_start..n_rows {
                if ((cand - cand_start) & 127) == 0 {
                    check_interrupt_fast()?;
                }
                let gid = group_ids[cand];
                if candidate_group_is_excluded(&node.rule, gid, params) {
                    continue;
                }
                let row = row_prefix(bits_train, row_words_train, cand, needed_words_train);
                let intersection = binary_pair_intersection_with_lookup(
                    &node.combined_train,
                    row,
                    y_train,
                    n_train,
                    params.y_sum_lookup.as_deref(),
                );
                for &op in beam_binary_ops_for_rule(&node.rule).iter() {
                    for &negated in child_literal_negations_for_op(op).iter() {
                        let single = literal_scores[literal_score_index(cand, negated)];
                        let Some(train) =
                            evaluate_child_train_from_parent_virtual_with_intersection(
                                &node.train,
                                &single.train,
                                intersection,
                                sum_y_train,
                                n_train,
                                node.rule.len() + 1,
                                op,
                                negated,
                                params,
                            )
                        else {
                            continue;
                        };
                        let literal = BeamLiteral {
                            row_index: cand,
                            group_id: gid,
                            negated,
                        };
                        let mut rule = node.rule.clone();
                        rule.rest.push((op, literal));
                        let max_singleton_train_raw =
                            node.max_singleton_train_raw.max(single.train.raw_score);
                        let max_singleton_test_raw =
                            node.max_singleton_test_raw.max(single.test.raw_score);
                        let direct_parent_train_raw = if rule.len() == 2 {
                            node.train.raw_score.max(single.train.raw_score)
                        } else {
                            best_ancestor_raw_baseline_cached(
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
                                &mut ancestor_raw_cache,
                                params.disable_parent_delta,
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
                        match best.entry(state.rule.lexical_key()) {
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
        best.into_values().collect::<Vec<_>>()
    };
    let mut out = out;
    out.sort_by(cmp_state);
    Ok(out)
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
    let child_bucket =
        bucket_from_rule_with_complexity(&state.rule, test.dosage_maf, params.null_complexity_bin);
    let direct_parent_test_raw = best_ancestor_raw_baseline(
        &state.rule,
        y_test,
        bits_test,
        row_words_test,
        n_rows,
        n_test,
        literal_scores,
        false,
        params.disable_parent_delta,
    )?;
    let child_abs_score = rank_rule_score_components_base(
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
    let direct_parent_raw = best_ancestor_raw_baseline(
        rule,
        y,
        bits,
        row_words,
        n_rows,
        n_samples,
        literal_scores,
        is_train,
        params.disable_parent_delta,
    )?;
    Ok(rank_rule_score_components_base(
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
    let mut ancestor_cache = RuleAncestorBaselineCache::new();
    let direct_parent_raw = best_ancestor_raw_baseline_cached(
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
        &mut ancestor_cache,
        params.disable_parent_delta,
    )?;
    Ok(rank_rule_score_components_base(
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
    let bucket = bucket_from_rule_with_complexity(rule, raw.dosage_maf, params.null_complexity_bin);
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
    let bucket = bucket_from_rule_with_complexity(rule, raw.dosage_maf, params.null_complexity_bin);
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
    _y_train: &[f64],
    _bits_train: &[u64],
    _row_words_train: usize,
    n_rows: usize,
    _n_train: usize,
    y_test: &[f64],
    bits_test: &[u64],
    row_words_test: usize,
    n_test: usize,
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Result<BeamRuleCandidate, String> {
    let current_rule = state.rule.clone();
    let current_train = state.train;
    let current_train_score = state.train_score;
    let cache_capacity = current_rule.len().saturating_mul(16).max(16);
    let mut train_raw_cache = RuleRawScoreCache::with_capacity(cache_capacity);
    let mut test_raw_cache = RuleRawScoreCache::with_capacity(cache_capacity);
    let _train_bits_cache = RuleBitsCache::with_capacity(cache_capacity);
    let mut test_bits_cache = RuleBitsCache::with_capacity(cache_capacity);
    cache_rule_raw_score(&mut train_raw_cache, &current_rule, current_train.raw_score);
    let current_test = evaluate_rule_continuous_cached(
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
    let current_test_score = final_rule_score_for_eval_cached(
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

    /*
    Surrogate collapse is intentionally disabled for both the legacy binary
    search path and its dedicated tests. Keep the old implementation here for
    possible future recovery.
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
            let diff_frac = bit_hamming_fraction(current_bits_test, parent_bits_test, n_test);
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
                let diff_frac = bit_hamming_fraction(current_bits_test, subrule_bits_test, n_test);
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
                let diff_frac =
                    bit_hamming_fraction(current_bits_test, singleton_bits_test, n_test);
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
    */

    Ok(BeamRuleCandidate {
        rule: current_rule,
        train_score: current_train_score,
        test_score: current_test_score,
        train: current_train,
        test: current_test,
    })
}

fn beam_search_train_test_continuous_impl(
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
    literal_scores_override: Option<&[LiteralSingletonScore]>,
) -> Result<Vec<BeamRuleCandidate>, String> {
    let beam_t0 = Instant::now();
    let out = (|| {
        check_ctrlc()?;
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
        let mode_name = if params.whole_genome_dev_mode {
            "wholegenome"
        } else {
            "standard"
        };
        let literal_scores_storage;
        let literal_scores = if let Some(scores) = literal_scores_override {
            if scores.len() != n_rows.saturating_mul(2) {
                return Err(format!(
                    "garfield::beam_search_train_test_continuous: literal score length mismatch: {} vs expected {}",
                    scores.len(),
                    n_rows.saturating_mul(2)
                ));
            }
            scores
        } else {
            literal_scores_storage = precompute_literal_singleton_scores(
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
            )?;
            literal_scores_storage.as_slice()
        };
        garfield_layer_rss_breakpoint(mode_name, 0, "literal_ready", 0, 0)?;

        let mut kept_all = Vec::<BeamState>::new();
        garfield_layer_rss_breakpoint(mode_name, 1, "start", 0, kept_all.len())?;
        let mut beam = if exhaustive_depth > 1 {
            let exhaustive_initial = build_initial_states_exhaustive(
                y_train,
                bits_train,
                row_words_train,
                n_rows,
                needed_words_train,
                n_train,
                group_ids,
                literal_scores,
                &params,
            )?;
            kept_all.extend(exhaustive_initial.iter().cloned());
            let mut frontier = exhaustive_initial;
            for _depth in 2..=exhaustive_depth {
                check_ctrlc()?;
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
                    literal_scores,
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
                literal_scores,
                &params,
            )?;
            kept_all.extend(beam.iter().cloned());
            beam
        };
        garfield_layer_rss_breakpoint(mode_name, 1, "end", beam.len(), kept_all.len())?;

        let mut depth_start = exhaustive_depth + 1;
        if params.whole_genome_dev_mode && exhaustive_depth == 1 && max_depth >= 2 {
            check_ctrlc()?;
            garfield_layer_rss_breakpoint(mode_name, 2, "start", beam.len(), kept_all.len())?;
            let next = expand_beam_once_whole_genome_layer2(
                &beam,
                y_train,
                sum_y_train,
                bits_train,
                row_words_train,
                n_rows,
                needed_words_train,
                n_train,
                group_ids,
                literal_scores,
                &params,
            )?;
            if !next.is_empty() {
                kept_all.extend(next.iter().cloned());
                beam = next;
            }
            garfield_layer_rss_breakpoint(mode_name, 2, "end", beam.len(), kept_all.len())?;
            depth_start = 3;
        }

        for depth in depth_start..=max_depth {
            check_ctrlc()?;
            garfield_layer_rss_breakpoint(mode_name, depth, "start", beam.len(), kept_all.len())?;
            let next = if params.whole_genome_dev_mode {
                expand_beam_once_whole_genome_target_parallel(
                    beam.as_slice(),
                    y_train,
                    sum_y_train,
                    bits_train,
                    row_words_train,
                    n_rows,
                    needed_words_train,
                    n_train,
                    group_ids,
                    literal_scores,
                    &params,
                )?
            } else {
                expand_beam_once(
                    &beam,
                    y_train,
                    sum_y_train,
                    bits_train,
                    row_words_train,
                    n_rows,
                    needed_words_train,
                    n_train,
                    group_ids,
                    literal_scores,
                    &params,
                )?
            };
            if next.is_empty() {
                garfield_layer_rss_breakpoint(
                    mode_name,
                    depth,
                    "empty",
                    beam.len(),
                    kept_all.len(),
                )?;
                break;
            }
            kept_all.extend(next.iter().cloned());
            beam = next;
            garfield_layer_rss_breakpoint(mode_name, depth, "end", beam.len(), kept_all.len())?;
        }

        let retained = dedup_states_by_support_signature(
            kept_all,
            bits_test,
            row_words_test,
            n_rows,
            n_test,
            literal_inputs_are_shared(
                y_train,
                n_train,
                y_test,
                n_test,
                bits_train,
                row_words_train,
                bits_test,
                row_words_test,
                n_rows,
            ),
        )?;

        let mut best_by_rule =
            HashMap::<Vec<(usize, bool, u8)>, BeamRuleCandidate>::with_capacity(retained.len());
        for state in retained.into_iter() {
            check_interrupt_fast()?;
            let cand = canonicalize_singleton_output_candidate(
                collapse_surrogate_candidate(
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
                    literal_scores,
                    &params,
                )?,
                literal_scores,
                &params,
            );
            if !keep_rule_after_dosage_maf_pruning(&cand.test, &params) {
                continue;
            }
            if !keep_child_after_parent_gain_pruning(&cand.rule, cand.test_score, &params) {
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
    })();
    GARFIELD_BEAM_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
    GARFIELD_BEAM_PROFILE_TOTAL_NS.fetch_add(elapsed_ns_saturating(beam_t0), Ordering::Relaxed);
    out
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
    beam_search_train_test_continuous_impl(
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
        params,
        None,
    )
}

pub(crate) fn beam_search_train_test_continuous_with_literal_scores(
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
    literal_scores: &[LiteralSingletonScore],
) -> Result<Vec<BeamRuleCandidate>, String> {
    beam_search_train_test_continuous_impl(
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
        params,
        Some(literal_scores),
    )
}

#[inline]
fn masked_not_vec(bits: &[u64], n_samples: usize) -> Vec<u64> {
    let mut out = bits.to_vec();
    bitnot_masked(out.as_mut_slice(), n_samples);
    out
}

#[inline]
fn complement_dual_bits_in_place_local(
    ge1_bits: &mut [u64],
    ge2_bits: &mut [u64],
    n_samples: usize,
) {
    debug_assert_eq!(ge1_bits.len(), ge2_bits.len());
    for i in 0..ge1_bits.len() {
        let a1 = ge1_bits[i];
        let a2 = ge2_bits[i];
        ge1_bits[i] = a2;
        ge2_bits[i] = a1;
    }
    bitnot_masked(ge1_bits, n_samples);
    bitnot_masked(ge2_bits, n_samples);
}

#[inline]
fn apply_first_literal_dual(
    row_ge1: &[u64],
    row_ge2: &[u64],
    n_samples: usize,
    negated: bool,
) -> (Vec<u64>, Vec<u64>) {
    if negated {
        (
            masked_not_vec(row_ge2, n_samples),
            masked_not_vec(row_ge1, n_samples),
        )
    } else {
        (row_ge1.to_vec(), row_ge2.to_vec())
    }
}

#[inline]
fn apply_literal_inplace_dual(
    dst_ge1: &mut [u64],
    dst_ge2: &mut [u64],
    row_ge1: &[u64],
    row_ge2: &[u64],
    op: BeamBinaryOp,
    negated: bool,
    n_samples: usize,
) {
    match (op, negated) {
        (BeamBinaryOp::And, true) => {
            bitand_not_assign_masked(dst_ge1, row_ge2, n_samples);
            bitand_not_assign_masked(dst_ge2, row_ge1, n_samples);
        }
        (BeamBinaryOp::And, false) => {
            bitand_assign(dst_ge1, row_ge1);
            bitand_assign(dst_ge2, row_ge2);
            apply_tail_mask(dst_ge1, tail_mask(n_samples));
            apply_tail_mask(dst_ge2, tail_mask(n_samples));
        }
        (BeamBinaryOp::Or, true) => {
            bitor_not_into_masked(dst_ge1, row_ge2, n_samples);
            bitor_not_into_masked(dst_ge2, row_ge1, n_samples);
        }
        (BeamBinaryOp::Or, false) => {
            bitor_assign(dst_ge1, row_ge1);
            bitor_assign(dst_ge2, row_ge2);
            apply_tail_mask(dst_ge1, tail_mask(n_samples));
            apply_tail_mask(dst_ge2, tail_mask(n_samples));
        }
        (BeamBinaryOp::Xor, negated) => {
            debug_assert_eq!(dst_ge1.len(), dst_ge2.len());
            debug_assert_eq!(dst_ge1.len(), row_ge1.len());
            debug_assert_eq!(dst_ge2.len(), row_ge2.len());
            let needed_words = words_for_samples(n_samples);
            if needed_words == 0 {
                return;
            }
            let full_words = n_samples >> 6;
            let rem = n_samples & 63;
            for i in 0..needed_words {
                let mask = if i < full_words || rem == 0 {
                    u64::MAX
                } else {
                    (1u64 << rem) - 1u64
                };
                let a1 = dst_ge1[i];
                let a2 = dst_ge2[i];
                let (b1, b2) = if negated {
                    ((!row_ge2[i]) & mask, (!row_ge1[i]) & mask)
                } else {
                    (row_ge1[i] & mask, row_ge2[i] & mask)
                };
                dst_ge1[i] = ((a1 & !b2) | ((!a2) & b1)) & mask;
                dst_ge2[i] = (((!a1) & b2) | (a2 & !b1)) & mask;
            }
            apply_tail_mask(dst_ge1, tail_mask(n_samples));
            apply_tail_mask(dst_ge2, tail_mask(n_samples));
        }
    }
}

pub fn materialize_rule_bits_dual(
    rule: &BeamRule,
    ge1_flat: &[u64],
    ge2_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let ctx = "garfield::materialize_rule_bits_dual";
    let needed_words_ge1 = validate_bit_matrix(ge1_flat, row_words, n_rows, n_samples, ctx)?;
    let needed_words_ge2 = validate_bit_matrix(ge2_flat, row_words, n_rows, n_samples, ctx)?;
    let needed_words = needed_words_ge1.min(needed_words_ge2);
    if rule.first.row_index >= n_rows {
        return Err(format!(
            "{ctx}: first literal row index {} out of range for n_rows={}",
            rule.first.row_index, n_rows
        ));
    }
    let row_ge1 = row_prefix(ge1_flat, row_words, rule.first.row_index, needed_words);
    let row_ge2 = row_prefix(ge2_flat, row_words, rule.first.row_index, needed_words);
    let (mut combined_ge1, mut combined_ge2) =
        apply_first_literal_dual(row_ge1, row_ge2, n_samples, rule.first.negated);
    for (op, lit) in rule.rest.iter() {
        if lit.row_index >= n_rows {
            return Err(format!(
                "{ctx}: literal row index {} out of range for n_rows={}",
                lit.row_index, n_rows
            ));
        }
        let row_ge1 = row_prefix(ge1_flat, row_words, lit.row_index, needed_words);
        let row_ge2 = row_prefix(ge2_flat, row_words, lit.row_index, needed_words);
        apply_literal_inplace_dual(
            combined_ge1.as_mut_slice(),
            combined_ge2.as_mut_slice(),
            row_ge1,
            row_ge2,
            *op,
            lit.negated,
            n_samples,
        );
    }
    Ok((combined_ge1, combined_ge2))
}

fn score_rule_continuous_from_dual(
    rule: &BeamRule,
    y: &[f64],
    combined_ge1: &[u64],
    combined_ge2: &[u64],
    n_samples: usize,
    total_sum_y: f64,
    lambda_len: f64,
    lambda_not: f64,
) -> ContinuousRuleScore {
    let mut sc = score_cont_centered_gain_dual_packed_with_sum(
        y,
        combined_ge1,
        combined_ge2,
        n_samples,
        total_sum_y,
    );
    let penalty = if rule.len() > 1 {
        lambda_len * ((rule.len() - 1) as f64)
    } else {
        0.0
    } + lambda_not * (rule.not_count() as f64);
    sc.score = sc.raw_score - penalty;
    sc
}

#[inline]
fn complement_dual_summary(
    total_sum_y: f64,
    n_samples: usize,
    pos_n_ge1: usize,
    pos_n_ge2: usize,
    pos_sum_ge1: f64,
    pos_sum_ge2: f64,
) -> (usize, usize, f64, f64) {
    (
        n_samples.saturating_sub(pos_n_ge2),
        n_samples.saturating_sub(pos_n_ge1),
        total_sum_y - pos_sum_ge2,
        total_sum_y - pos_sum_ge1,
    )
}

#[inline]
fn literal_dual_summary_with_negation(
    total_sum_y: f64,
    n_samples: usize,
    summary: DualLiteralSummary,
    negated: bool,
) -> (usize, usize, f64, f64) {
    if negated {
        complement_dual_summary(
            total_sum_y,
            n_samples,
            summary.pos_n_ge1,
            summary.pos_n_ge2,
            summary.pos_sum_ge1,
            summary.pos_sum_ge2,
        )
    } else {
        (
            summary.pos_n_ge1,
            summary.pos_n_ge2,
            summary.pos_sum_ge1,
            summary.pos_sum_ge2,
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DualPairIntersections {
    p1_r1_n: usize,
    p2_r2_n: usize,
    p1_r2_n: usize,
    p2_r1_n: usize,
    p1_r1_sum: f64,
    p2_r2_sum: f64,
    p1_r2_sum: f64,
    p2_r1_sum: f64,
}

#[inline]
fn dual_pair_intersections_for_params(
    parent_ge1: &[u64],
    parent_ge2: &[u64],
    row_ge1: &[u64],
    row_ge2: &[u64],
    y_train: &[f64],
    n_train: usize,
    params: &BeamSearchParams,
) -> DualPairIntersections {
    let p1_r1_n = and_popcount(parent_ge1, row_ge1) as usize;
    let p2_r2_n = and_popcount(parent_ge2, row_ge2) as usize;
    let p1_r2_n = and_popcount(parent_ge1, row_ge2) as usize;
    let p2_r1_n = and_popcount(parent_ge2, row_ge1) as usize;
    let t_sum = beam_detail_profile_start();
    let sums = sum_y_where_both1_four(
        [
            (parent_ge1, row_ge1),
            (parent_ge2, row_ge2),
            (parent_ge1, row_ge2),
            (parent_ge2, row_ge1),
        ],
        y_train,
        n_train,
        params.y_sum_lookup.as_deref(),
    );
    beam_detail_profile_end(t_sum, &GARFIELD_PROFILE_SUM_Y_BOTH1_NS);
    DualPairIntersections {
        p1_r1_n,
        p2_r2_n,
        p1_r2_n,
        p2_r1_n,
        p1_r1_sum: sums[0],
        p2_r2_sum: sums[1],
        p1_r2_sum: sums[2],
        p2_r1_sum: sums[3],
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DualEffectiveIntersections {
    inter_n_ge1: usize,
    inter_n_ge2: usize,
    cross_n_p1_r2: usize,
    cross_n_p2_r1: usize,
    inter_sum_ge1: f64,
    inter_sum_ge2: f64,
    cross_sum_p1_r2: f64,
    cross_sum_p2_r1: f64,
}

#[inline]
fn effective_dual_intersections(
    parent: &FuzzyBeamState,
    intersections: DualPairIntersections,
    negated: bool,
) -> DualEffectiveIntersections {
    if !negated {
        DualEffectiveIntersections {
            inter_n_ge1: intersections.p1_r1_n,
            inter_n_ge2: intersections.p2_r2_n,
            cross_n_p1_r2: intersections.p1_r2_n,
            cross_n_p2_r1: intersections.p2_r1_n,
            inter_sum_ge1: intersections.p1_r1_sum,
            inter_sum_ge2: intersections.p2_r2_sum,
            cross_sum_p1_r2: intersections.p1_r2_sum,
            cross_sum_p2_r1: intersections.p2_r1_sum,
        }
    } else {
        DualEffectiveIntersections {
            inter_n_ge1: parent.train.n_hit.saturating_sub(intersections.p1_r2_n),
            inter_n_ge2: parent.train_n_ge2.saturating_sub(intersections.p2_r1_n),
            cross_n_p1_r2: parent.train.n_hit.saturating_sub(intersections.p1_r1_n),
            cross_n_p2_r1: parent.train_n_ge2.saturating_sub(intersections.p2_r2_n),
            inter_sum_ge1: parent.train_sum_ge1 - intersections.p1_r2_sum,
            inter_sum_ge2: parent.train_sum_ge2 - intersections.p2_r1_sum,
            cross_sum_p1_r2: parent.train_sum_ge1 - intersections.p1_r1_sum,
            cross_sum_p2_r1: parent.train_sum_ge2 - intersections.p2_r2_sum,
        }
    }
}

#[inline]
fn keep_xor_substates_dual(
    parent: &FuzzyBeamState,
    row_n_ge1: usize,
    row_n_ge2: usize,
    intersections: DualEffectiveIntersections,
    n_samples: usize,
    params: &BeamSearchParams,
) -> bool {
    let parent_and_not_row_ge1 = parent
        .train
        .n_hit
        .saturating_sub(intersections.cross_n_p1_r2);
    let parent_and_not_row_ge2 = parent
        .train_n_ge2
        .saturating_sub(intersections.cross_n_p2_r1);
    let not_parent_and_row_ge1 = row_n_ge1.saturating_sub(intersections.cross_n_p2_r1);
    let not_parent_and_row_ge2 = row_n_ge2.saturating_sub(intersections.cross_n_p1_r2);
    keep_rule_after_dosage_maf_counts(
        parent_and_not_row_ge1,
        parent_and_not_row_ge2,
        n_samples,
        params,
    ) && keep_rule_after_dosage_maf_counts(
        not_parent_and_row_ge1,
        not_parent_and_row_ge2,
        n_samples,
        params,
    )
}

#[inline]
fn evaluate_child_train_from_parent_virtual_fuzzy_with_intersections(
    parent: &FuzzyBeamState,
    row_summary: DualLiteralSummary,
    sum_y_train: f64,
    n_train: usize,
    child_rule_len: usize,
    op: BeamBinaryOp,
    negated: bool,
    intersections: DualPairIntersections,
    params: &BeamSearchParams,
) -> Option<(ContinuousRuleScore, usize, f64, f64)> {
    let (row_n_ge1, row_n_ge2, row_sum_ge1, row_sum_ge2) =
        literal_dual_summary_with_negation(sum_y_train, n_train, row_summary, negated);
    let xor_intersections = if matches!(op, BeamBinaryOp::Xor) {
        Some(effective_dual_intersections(parent, intersections, negated))
    } else {
        None
    };
    if params.filter_xor_substates {
        if let Some(effective) = xor_intersections {
            if !keep_xor_substates_dual(parent, row_n_ge1, row_n_ge2, effective, n_train, params) {
                return None;
            }
        }
    }
    let (child_n_ge1, child_n_ge2, child_sum_ge1, child_sum_ge2) = match op {
        BeamBinaryOp::And => {
            let (inter_n_ge1, inter_n_ge2, inter_sum_ge1, inter_sum_ge2) = if negated {
                (
                    parent.train.n_hit.saturating_sub(intersections.p1_r2_n),
                    parent.train_n_ge2.saturating_sub(intersections.p2_r1_n),
                    parent.train_sum_ge1 - intersections.p1_r2_sum,
                    parent.train_sum_ge2 - intersections.p2_r1_sum,
                )
            } else {
                (
                    intersections.p1_r1_n,
                    intersections.p2_r2_n,
                    intersections.p1_r1_sum,
                    intersections.p2_r2_sum,
                )
            };
            (inter_n_ge1, inter_n_ge2, inter_sum_ge1, inter_sum_ge2)
        }
        BeamBinaryOp::Or => (
            {
                let inter_n_ge1 = if negated {
                    parent.train.n_hit.saturating_sub(intersections.p1_r2_n)
                } else {
                    intersections.p1_r1_n
                };
                inter_n_ge1
            },
            {
                let inter_n_ge2 = if negated {
                    parent.train_n_ge2.saturating_sub(intersections.p2_r1_n)
                } else {
                    intersections.p2_r2_n
                };
                inter_n_ge2
            },
            {
                let value = if negated {
                    parent.train_sum_ge1 - intersections.p1_r2_sum
                } else {
                    intersections.p1_r1_sum
                };
                value
            },
            {
                let value = if negated {
                    parent.train_sum_ge2 - intersections.p2_r1_sum
                } else {
                    intersections.p2_r2_sum
                };
                value
            },
        ),
        BeamBinaryOp::Xor => {
            let inter = xor_intersections.expect("XOR intersections must be initialized");
            (
                parent
                    .train
                    .n_hit
                    .saturating_add(row_n_ge1)
                    .saturating_sub(inter.inter_n_ge1)
                    .saturating_sub(inter.inter_n_ge2),
                row_n_ge2
                    .saturating_sub(inter.cross_n_p1_r2)
                    .saturating_add(parent.train_n_ge2.saturating_sub(inter.cross_n_p2_r1)),
                parent.train_sum_ge1 + row_sum_ge1 - inter.inter_sum_ge1 - inter.inter_sum_ge2,
                row_sum_ge2 - inter.cross_sum_p1_r2 + parent.train_sum_ge2 - inter.cross_sum_p2_r1,
            )
        }
    };
    let _ = child_rule_len;
    if !keep_rule_after_dosage_maf_counts(child_n_ge1, child_n_ge2, n_train, params) {
        return None;
    }
    let train = score_cont_centered_gain_dual_from_summary(
        sum_y_train,
        n_train,
        child_n_ge1,
        child_n_ge2,
        child_sum_ge1,
        child_sum_ge2,
    );
    Some((train, child_n_ge2, child_sum_ge1, child_sum_ge2))
}

#[inline]
fn evaluate_child_train_from_parent_virtual_fuzzy(
    parent_ge1: &[u64],
    parent_ge2: &[u64],
    parent: &FuzzyBeamState,
    row_ge1: &[u64],
    row_ge2: &[u64],
    row_summary: DualLiteralSummary,
    y_train: &[f64],
    sum_y_train: f64,
    n_train: usize,
    child_rule_len: usize,
    op: BeamBinaryOp,
    negated: bool,
    params: &BeamSearchParams,
) -> Option<(ContinuousRuleScore, usize, f64, f64)> {
    let intersections = dual_pair_intersections_for_params(
        parent_ge1, parent_ge2, row_ge1, row_ge2, y_train, n_train, params,
    );
    evaluate_child_train_from_parent_virtual_fuzzy_with_intersections(
        parent,
        row_summary,
        sum_y_train,
        n_train,
        child_rule_len,
        op,
        negated,
        intersections,
        params,
    )
}

#[inline]
fn evaluate_rule_continuous_dual_with_sum(
    rule: &BeamRule,
    y: &[f64],
    total_sum_y: f64,
    ge1_flat: &[u64],
    ge2_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    lambda_len: f64,
    lambda_not: f64,
) -> Result<ContinuousRuleScore, String> {
    let ctx = "garfield::evaluate_rule_continuous_dual";
    validate_continuous_y(y, n_samples, ctx)?;
    let (combined_ge1, combined_ge2) =
        materialize_rule_bits_dual(rule, ge1_flat, ge2_flat, row_words, n_rows, n_samples)?;
    Ok(score_rule_continuous_from_dual(
        rule,
        y,
        combined_ge1.as_slice(),
        combined_ge2.as_slice(),
        n_samples,
        total_sum_y,
        lambda_len,
        lambda_not,
    ))
}

pub fn evaluate_rule_continuous_dual(
    rule: &BeamRule,
    y: &[f64],
    ge1_flat: &[u64],
    ge2_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    lambda_len: f64,
    lambda_not: f64,
) -> Result<ContinuousRuleScore, String> {
    evaluate_rule_continuous_dual_with_sum(
        rule,
        y,
        y.iter().take(n_samples).copied().sum::<f64>(),
        ge1_flat,
        ge2_flat,
        row_words,
        n_rows,
        n_samples,
        lambda_len,
        lambda_not,
    )
}

fn validate_search_inputs_fuzzy(
    y_train: &[f64],
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    n_train: usize,
    y_test: &[f64],
    ge1_test: &[u64],
    ge2_test: &[u64],
    row_words_test: usize,
    n_test: usize,
    group_ids: &[usize],
    params: &BeamSearchParams,
) -> Result<(usize, usize), String> {
    let ctx = "garfield::beam_search_train_test_continuous_fuzzy";
    validate_continuous_y(y_train, n_train, ctx)?;
    validate_continuous_y(y_test, n_test, ctx)?;
    let need_train_ge1 = validate_bit_matrix(ge1_train, row_words_train, n_rows, n_train, ctx)?;
    let need_train_ge2 = validate_bit_matrix(ge2_train, row_words_train, n_rows, n_train, ctx)?;
    let need_test_ge1 = validate_bit_matrix(ge1_test, row_words_test, n_rows, n_test, ctx)?;
    let need_test_ge2 = validate_bit_matrix(ge2_test, row_words_test, n_rows, n_test, ctx)?;
    if need_train_ge1 != need_train_ge2 || need_test_ge1 != need_test_ge2 {
        return Err(format!("{ctx}: fuzzy bitplane word counts do not match"));
    }
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
    Ok((need_train_ge1, need_test_ge1))
}

#[inline]
fn cmp_fuzzy_state(a: &FuzzyBeamState, b: &FuzzyBeamState) -> std::cmp::Ordering {
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
fn cmp_fuzzy_state_lite(a: &FuzzyBeamStateLite, b: &FuzzyBeamStateLite) -> std::cmp::Ordering {
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
fn push_top_k_fuzzy_states(nodes: &mut Vec<FuzzyBeamState>, cand: FuzzyBeamState, k: usize) {
    if k == 0 {
        return;
    }
    if nodes.len() < k {
        nodes.push(cand);
        return;
    }
    let mut worst_idx = 0usize;
    for i in 1..nodes.len() {
        if cmp_fuzzy_state(&nodes[i], &nodes[worst_idx]) == std::cmp::Ordering::Greater {
            worst_idx = i;
        }
    }
    if cmp_fuzzy_state(&cand, &nodes[worst_idx]) == std::cmp::Ordering::Less {
        nodes[worst_idx] = cand;
    }
}

fn materialize_fuzzy_beam_state_lite(
    cand: FuzzyBeamStateLite,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    n_train: usize,
) -> Result<FuzzyBeamState, String> {
    let (combined_train_ge1, combined_train_ge2) = materialize_rule_bits_dual(
        &cand.rule,
        ge1_train,
        ge2_train,
        row_words_train,
        n_rows,
        n_train,
    )?;
    Ok(FuzzyBeamState {
        rule: cand.rule,
        combined_train_ge1,
        combined_train_ge2,
        train: cand.train,
        train_n_ge2: cand.train_n_ge2,
        train_sum_ge1: cand.train_sum_ge1,
        train_sum_ge2: cand.train_sum_ge2,
        train_abs_score: cand.train_abs_score,
        train_score: cand.train_score,
        max_singleton_train_raw: cand.max_singleton_train_raw,
        max_singleton_test_raw: cand.max_singleton_test_raw,
    })
}

#[inline]
fn sort_truncate_fuzzy_states(mut nodes: Vec<FuzzyBeamState>, k: usize) -> Vec<FuzzyBeamState> {
    nodes.sort_by(cmp_fuzzy_state);
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

fn filter_fuzzy_beam_candidates(
    candidates: Vec<FuzzyBeamState>,
    width: usize,
    params: &BeamSearchParams,
) -> Vec<FuzzyBeamState> {
    if candidates.is_empty() {
        return candidates;
    }
    let _ = params;
    sort_truncate_fuzzy_states(candidates, width.max(1))
}

fn precompute_literal_singleton_scores_fuzzy(
    y_train: &[f64],
    n_train: usize,
    y_test: &[f64],
    n_test: usize,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    ge1_test: &[u64],
    ge2_test: &[u64],
    row_words_test: usize,
    n_rows: usize,
) -> Result<Vec<LiteralSingletonScore>, String> {
    let score_t0 = Instant::now();
    let out = (|| {
        let ctx = "garfield::precompute_literal_singleton_scores_fuzzy";
        validate_continuous_y(y_train, n_train, ctx)?;
        validate_continuous_y(y_test, n_test, ctx)?;
        let needed_words_train =
            validate_bit_matrix(ge1_train, row_words_train, n_rows, n_train, ctx)?;
        let needed_words_test = validate_bit_matrix(ge1_test, row_words_test, n_rows, n_test, ctx)?;
        validate_bit_matrix(ge2_train, row_words_train, n_rows, n_train, ctx)?;
        validate_bit_matrix(ge2_test, row_words_test, n_rows, n_test, ctx)?;
        let sum_y_train = y_train.iter().take(n_train).copied().sum::<f64>();
        let sum_y_test = y_test.iter().take(n_test).copied().sum::<f64>();
        let mut out = vec![
            LiteralSingletonScore {
                train: ContinuousRuleScore {
                    score: f64::NEG_INFINITY,
                    raw_score: f64::NEG_INFINITY,
                    mean_hit: f64::NAN,
                    mean_miss: f64::NAN,
                    support_frac: f64::NAN,
                    dosage_maf: f64::NAN,
                    n_hit: 0,
                    n_ge2: 0,
                    n_miss: 0,
                },
                test: ContinuousRuleScore {
                    score: f64::NEG_INFINITY,
                    raw_score: f64::NEG_INFINITY,
                    mean_hit: f64::NAN,
                    mean_miss: f64::NAN,
                    support_frac: f64::NAN,
                    dosage_maf: f64::NAN,
                    n_hit: 0,
                    n_ge2: 0,
                    n_miss: 0,
                },
            };
            n_rows.saturating_mul(2)
        ];
        for row_idx in 0..n_rows {
            let row_train_ge1 = row_prefix(ge1_train, row_words_train, row_idx, needed_words_train);
            let row_train_ge2 = row_prefix(ge2_train, row_words_train, row_idx, needed_words_train);
            let row_test_ge1 = row_prefix(ge1_test, row_words_test, row_idx, needed_words_test);
            let row_test_ge2 = row_prefix(ge2_test, row_words_test, row_idx, needed_words_test);
            let (train_n_ge1, train_n_ge2, train_sum_ge1, train_sum_ge2) =
                dual_packed_summary(row_train_ge1, row_train_ge2, y_train, n_train);
            let (test_n_ge1, test_n_ge2, test_sum_ge1, test_sum_ge2) =
                dual_packed_summary(row_test_ge1, row_test_ge2, y_test, n_test);
            let pos_train = score_cont_centered_gain_dual_from_summary(
                sum_y_train,
                n_train,
                train_n_ge1,
                train_n_ge2,
                train_sum_ge1,
                train_sum_ge2,
            );
            let pos_test = score_cont_centered_gain_dual_from_summary(
                sum_y_test,
                n_test,
                test_n_ge1,
                test_n_ge2,
                test_sum_ge1,
                test_sum_ge2,
            );
            out[literal_score_index(row_idx, false)] = LiteralSingletonScore {
                train: pos_train,
                test: pos_test,
            };
            let neg_train = score_cont_centered_gain_dual_from_summary(
                sum_y_train,
                n_train,
                n_train.saturating_sub(train_n_ge2),
                n_train.saturating_sub(train_n_ge1),
                sum_y_train - train_sum_ge2,
                sum_y_train - train_sum_ge1,
            );
            let neg_test = score_cont_centered_gain_dual_from_summary(
                sum_y_test,
                n_test,
                n_test.saturating_sub(test_n_ge2),
                n_test.saturating_sub(test_n_ge1),
                sum_y_test - test_sum_ge2,
                sum_y_test - test_sum_ge1,
            );
            out[literal_score_index(row_idx, true)] = LiteralSingletonScore {
                train: neg_train,
                test: neg_test,
            };
        }
        Ok(out)
    })();
    add_garfield_beam_profile_literal_precompute_ns(elapsed_ns_saturating(score_t0));
    out
}

fn lookup_rule_raw_score_fuzzy_cached(
    rule: &BeamRule,
    y: &[f64],
    total_sum_y: f64,
    ge1_flat: &[u64],
    ge2_flat: &[u64],
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
    let raw_score = evaluate_rule_continuous_dual_with_sum(
        rule,
        y,
        total_sum_y,
        ge1_flat,
        ge2_flat,
        row_words,
        n_rows,
        n_samples,
        0.0,
        0.0,
    )?
    .raw_score;
    local_cache.insert(key, raw_score);
    Ok(raw_score)
}

fn best_ancestor_raw_baseline_fuzzy_cached(
    rule: &BeamRule,
    y: &[f64],
    total_sum_y: f64,
    ge1_flat: &[u64],
    ge2_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
    base_cache: Option<&RuleRawScoreCache>,
    raw_cache: &mut RuleRawScoreCache,
    ancestor_cache: &mut RuleAncestorBaselineCache,
    disable_parent_delta: bool,
) -> Result<f64, String> {
    let t_profile = beam_detail_profile_start();
    let key = rule.lexical_key();
    if let Some(score) = ancestor_cache.get(&key) {
        return Ok(*score);
    }
    let result: Result<f64, String> = if disable_parent_delta || rule.len() <= 1 {
        Ok(0.0)
    } else if rule.len() == 2 {
        Ok(rule_max_singleton_raw(rule, literal_scores, is_train))
    } else {
        let mut best = f64::NEG_INFINITY;
        for remove_idx in 0..rule.len() {
            let Some(parent_rule) = rule_without_literal(rule, remove_idx) else {
                continue;
            };
            let parent_raw = lookup_rule_raw_score_fuzzy_cached(
                &parent_rule,
                y,
                total_sum_y,
                ge1_flat,
                ge2_flat,
                row_words,
                n_rows,
                n_samples,
                base_cache,
                raw_cache,
            )?;
            let parent_ancestor = best_ancestor_raw_baseline_fuzzy_cached(
                &parent_rule,
                y,
                total_sum_y,
                ge1_flat,
                ge2_flat,
                row_words,
                n_rows,
                n_samples,
                literal_scores,
                is_train,
                base_cache,
                raw_cache,
                ancestor_cache,
                disable_parent_delta,
            )?;
            best = best.max(parent_raw.max(parent_ancestor));
        }
        if best.is_finite() {
            Ok(best)
        } else {
            Ok(0.0)
        }
    };
    let ret = result?;
    ancestor_cache.insert(key, ret);
    beam_detail_profile_end(t_profile, &GARFIELD_PROFILE_PARENT_BASELINE_NS);
    Ok(ret)
}

fn best_ancestor_raw_baseline_fuzzy(
    rule: &BeamRule,
    y: &[f64],
    total_sum_y: f64,
    ge1_flat: &[u64],
    ge2_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    literal_scores: &[LiteralSingletonScore],
    is_train: bool,
    disable_parent_delta: bool,
) -> Result<f64, String> {
    let mut raw_cache = RuleRawScoreCache::new();
    let mut ancestor_cache = RuleAncestorBaselineCache::new();
    best_ancestor_raw_baseline_fuzzy_cached(
        rule,
        y,
        total_sum_y,
        ge1_flat,
        ge2_flat,
        row_words,
        n_rows,
        n_samples,
        literal_scores,
        is_train,
        None,
        &mut raw_cache,
        &mut ancestor_cache,
        disable_parent_delta,
    )
}

fn dedup_fuzzy_states_by_rule_key(states: Vec<FuzzyBeamState>) -> Vec<FuzzyBeamState> {
    let mut best = HashMap::<RuleLexKey, FuzzyBeamState>::with_capacity(states.len());
    for state in states.into_iter() {
        let key = state.rule.lexical_key();
        match best.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(state);
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if cmp_fuzzy_state(&state, slot.get()) == std::cmp::Ordering::Less {
                    slot.insert(state);
                }
            }
        }
    }
    let mut out = best.into_values().collect::<Vec<_>>();
    out.sort_by(cmp_fuzzy_state);
    out
}

fn dedup_fuzzy_states_by_support_signature(
    states: Vec<FuzzyBeamState>,
    ge1_test: &[u64],
    ge2_test: &[u64],
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    train_test_shared: bool,
) -> Result<Vec<FuzzyBeamState>, String> {
    if train_test_shared {
        let mut best = HashMap::<(Vec<u64>, Vec<u64>), FuzzyBeamState>::with_capacity(states.len());
        for state in states.into_iter() {
            let key = (
                state.combined_train_ge1.clone(),
                state.combined_train_ge2.clone(),
            );
            match best.entry(key) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(state);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    if cmp_fuzzy_state(&state, slot.get()) == std::cmp::Ordering::Less {
                        slot.insert(state);
                    }
                }
            }
        }
        let mut out = best.into_values().collect::<Vec<_>>();
        out.sort_by(cmp_fuzzy_state);
        return Ok(out);
    }

    let mut best =
        HashMap::<FuzzyBeamSupportSignature, FuzzyBeamState>::with_capacity(states.len());
    for state in states.into_iter() {
        let (test_ge1_bits, test_ge2_bits) = materialize_rule_bits_dual(
            &state.rule,
            ge1_test,
            ge2_test,
            row_words_test,
            n_rows,
            n_test,
        )?;
        let key = FuzzyBeamSupportSignature {
            train_ge1: state.combined_train_ge1.clone(),
            train_ge2: state.combined_train_ge2.clone(),
            test_ge1: Some(test_ge1_bits),
            test_ge2: Some(test_ge2_bits),
        };
        match best.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(state);
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if cmp_fuzzy_state(&state, slot.get()) == std::cmp::Ordering::Less {
                    slot.insert(state);
                }
            }
        }
    }
    let mut out = best.into_values().collect::<Vec<_>>();
    out.sort_by(cmp_fuzzy_state);
    Ok(out)
}

fn build_initial_fuzzy_beam(
    sum_y_train: f64,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    literal_summaries: &[DualLiteralSummary],
    params: &BeamSearchParams,
) -> Result<Vec<FuzzyBeamState>, String> {
    check_interrupt_fast()?;
    let n_rows = literal_summaries.len();
    let layer_cap = params.beam_width.min(n_rows);
    let mut seq = Vec::<FuzzyBeamState>::with_capacity(layer_cap);
    let mut diag = FuzzyInitialLiteralStats {
        n_rows,
        ..FuzzyInitialLiteralStats::default()
    };
    for row_idx in 0..n_rows {
        if (row_idx & 255) == 0 {
            check_interrupt_fast()?;
        }
        let row_ge1 = row_prefix(ge1_train, row_words_train, row_idx, needed_words_train);
        let row_ge2 = row_prefix(ge2_train, row_words_train, row_idx, needed_words_train);
        let summary = literal_summaries[row_idx];
        for &negated in initial_singleton_negations(params).iter() {
            let (_, train_n_ge2, train_sum_ge1, train_sum_ge2) =
                literal_dual_summary_with_negation(sum_y_train, n_train, summary, negated);
            let literal = BeamLiteral {
                row_index: row_idx,
                group_id: group_ids[row_idx],
                negated,
            };
            garfield_layer_debug_add(
                1,
                GarfieldLayerDebugFamily::Singleton,
                GarfieldLayerDebugMetric::Considered,
                1,
            );
            let rule = BeamRule {
                first: literal,
                rest: Vec::new(),
            };
            let (combined_ge1, combined_ge2) =
                apply_first_literal_dual(row_ge1, row_ge2, n_train, negated);
            let single = literal_scores[literal_score_index(row_idx, negated)];
            let train = single.train;
            let (train_abs_score, train_score) =
                train_scores_for_rule(&rule, train, train.raw_score, None, None, params);
            let pass_seed_basic = keep_initial_literal_after_seed_pruning(&train);
            let pass_gain = pass_seed_basic
                && keep_state_after_min_gain_pruning(rule.len(), train_score, params);
            update_fuzzy_initial_literal_stats(&mut diag, &train, pass_seed_basic, pass_gain);
            if !pass_seed_basic {
                continue;
            }
            garfield_layer_debug_add(
                1,
                GarfieldLayerDebugFamily::Singleton,
                GarfieldLayerDebugMetric::TrainOk,
                1,
            );
            if !pass_gain {
                continue;
            }
            garfield_layer_debug_add(
                1,
                GarfieldLayerDebugFamily::Singleton,
                GarfieldLayerDebugMetric::GainOk,
                1,
            );
            garfield_layer_debug_add(
                1,
                GarfieldLayerDebugFamily::Singleton,
                GarfieldLayerDebugMetric::Kept,
                1,
            );
            push_top_k_fuzzy_states(
                &mut seq,
                FuzzyBeamState {
                    rule,
                    combined_train_ge1: combined_ge1,
                    combined_train_ge2: combined_ge2,
                    train,
                    train_n_ge2,
                    train_sum_ge1,
                    train_sum_ge2,
                    train_abs_score,
                    train_score,
                    max_singleton_train_raw: single.train.raw_score,
                    max_singleton_test_raw: single.test.raw_score,
                },
                layer_cap,
            );
        }
    }
    if seq.is_empty() {
        return Err(format_no_valid_initial_literals_fuzzy(
            "garfield::build_initial_fuzzy_beam",
            &diag,
            params,
        ));
    }
    let out = filter_fuzzy_beam_candidates(seq, layer_cap, params);
    garfield_layer_debug_record_fuzzy_states(1, GarfieldLayerDebugMetric::Retained, out.as_slice());
    Ok(out)
}

fn build_initial_fuzzy_states_exhaustive(
    sum_y_train: f64,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    literal_summaries: &[DualLiteralSummary],
    params: &BeamSearchParams,
) -> Result<Vec<FuzzyBeamState>, String> {
    check_interrupt_fast()?;
    let n_rows = literal_summaries.len();
    let mut all = Vec::<FuzzyBeamState>::with_capacity(n_rows);
    let mut diag = FuzzyInitialLiteralStats {
        n_rows,
        ..FuzzyInitialLiteralStats::default()
    };
    for row_idx in 0..n_rows {
        if (row_idx & 255) == 0 {
            check_interrupt_fast()?;
        }
        let row_ge1 = row_prefix(ge1_train, row_words_train, row_idx, needed_words_train);
        let row_ge2 = row_prefix(ge2_train, row_words_train, row_idx, needed_words_train);
        let summary = literal_summaries[row_idx];
        for &negated in initial_singleton_negations(params).iter() {
            let (_, train_n_ge2, train_sum_ge1, train_sum_ge2) =
                literal_dual_summary_with_negation(sum_y_train, n_train, summary, negated);
            let literal = BeamLiteral {
                row_index: row_idx,
                group_id: group_ids[row_idx],
                negated,
            };
            garfield_layer_debug_add(
                1,
                GarfieldLayerDebugFamily::Singleton,
                GarfieldLayerDebugMetric::Considered,
                1,
            );
            let rule = BeamRule {
                first: literal,
                rest: Vec::new(),
            };
            let (combined_ge1, combined_ge2) =
                apply_first_literal_dual(row_ge1, row_ge2, n_train, negated);
            let single = literal_scores[literal_score_index(row_idx, negated)];
            let train = single.train;
            let (train_abs_score, train_score) =
                train_scores_for_rule(&rule, train, train.raw_score, None, None, params);
            let pass_seed_basic = keep_initial_literal_after_seed_pruning(&train);
            update_fuzzy_initial_literal_stats(&mut diag, &train, pass_seed_basic, pass_seed_basic);
            if !pass_seed_basic {
                continue;
            }
            garfield_layer_debug_add(
                1,
                GarfieldLayerDebugFamily::Singleton,
                GarfieldLayerDebugMetric::TrainOk,
                1,
            );
            garfield_layer_debug_add(
                1,
                GarfieldLayerDebugFamily::Singleton,
                GarfieldLayerDebugMetric::GainOk,
                1,
            );
            garfield_layer_debug_add(
                1,
                GarfieldLayerDebugFamily::Singleton,
                GarfieldLayerDebugMetric::Kept,
                1,
            );
            all.push(FuzzyBeamState {
                rule,
                combined_train_ge1: combined_ge1,
                combined_train_ge2: combined_ge2,
                train,
                train_n_ge2,
                train_sum_ge1,
                train_sum_ge2,
                train_abs_score,
                train_score,
                max_singleton_train_raw: single.train.raw_score,
                max_singleton_test_raw: single.test.raw_score,
            });
        }
    }
    let out = dedup_fuzzy_states_by_rule_key(all);
    if out.is_empty() {
        return Err(format_no_valid_initial_literals_fuzzy(
            "garfield::build_initial_fuzzy_states_exhaustive",
            &diag,
            params,
        ));
    }
    garfield_layer_debug_record_fuzzy_states(1, GarfieldLayerDebugMetric::Retained, out.as_slice());
    Ok(out)
}

fn whole_genome_layer2_parent_variants_fuzzy(
    node: &FuzzyBeamState,
    sum_y_train: f64,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    needed_words_train: usize,
    n_train: usize,
    literal_scores: &[LiteralSingletonScore],
    literal_summaries: &[DualLiteralSummary],
    params: &BeamSearchParams,
) -> Vec<FuzzyBeamState> {
    let mut out = Vec::<FuzzyBeamState>::with_capacity(2);
    out.push(node.clone());
    if node.rule.len() != 1 || node.rule.first.negated {
        return out;
    }
    let row_idx = node.rule.first.row_index;
    let row_ge1 = row_prefix(ge1_train, row_words_train, row_idx, needed_words_train);
    let row_ge2 = row_prefix(ge2_train, row_words_train, row_idx, needed_words_train);
    let (_, train_n_ge2, train_sum_ge1, train_sum_ge2) =
        literal_dual_summary_with_negation(sum_y_train, n_train, literal_summaries[row_idx], true);
    let literal = BeamLiteral {
        negated: true,
        ..node.rule.first
    };
    let rule = BeamRule {
        first: literal,
        rest: Vec::new(),
    };
    let (combined_ge1, combined_ge2) = apply_first_literal_dual(row_ge1, row_ge2, n_train, true);
    let single = literal_scores[literal_score_index(row_idx, true)];
    let train = single.train;
    let (train_abs_score, train_score) =
        train_scores_for_rule(&rule, train, train.raw_score, None, None, params);
    if keep_initial_literal_after_seed_pruning(&train)
        && keep_state_after_min_gain_pruning(rule.len(), train_score, params)
    {
        out.push(FuzzyBeamState {
            rule,
            combined_train_ge1: combined_ge1,
            combined_train_ge2: combined_ge2,
            train,
            train_n_ge2,
            train_sum_ge1,
            train_sum_ge2,
            train_abs_score,
            train_score,
            max_singleton_train_raw: single.train.raw_score,
            max_singleton_test_raw: single.test.raw_score,
        });
    }
    out
}

#[inline]
fn prune_best_fuzzy_state_map(best: &mut HashMap<RuleLexKey, FuzzyBeamStateLite>, keep: usize) {
    if keep == 0 || best.len() <= keep {
        return;
    }
    let mut states = best.drain().map(|(_, state)| state).collect::<Vec<_>>();
    states.sort_unstable_by(cmp_fuzzy_state_lite);
    states.truncate(keep);
    for state in states.into_iter() {
        best.insert(state.rule.lexical_key(), state);
    }
}

fn merge_best_fuzzy_state_maps(
    worker_maps: Vec<Result<HashMap<RuleLexKey, FuzzyBeamStateLite>, String>>,
    next_cap: usize,
) -> Result<Vec<FuzzyBeamStateLite>, String> {
    let mut global_best: HashMap<RuleLexKey, FuzzyBeamStateLite> =
        HashMap::with_capacity(next_cap.saturating_mul(2));
    for worker_map in worker_maps {
        for (key, state) in worker_map? {
            match global_best.entry(key) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(state);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    if cmp_fuzzy_state_lite(&state, slot.get()) == std::cmp::Ordering::Less {
                        slot.insert(state);
                    }
                }
            }
        }
        if global_best.len() > next_cap.saturating_mul(4).max(next_cap.saturating_add(1)) {
            prune_best_fuzzy_state_map(&mut global_best, next_cap);
        }
    }
    let mut out = global_best.into_values().collect::<Vec<_>>();
    out.sort_unstable_by(cmp_fuzzy_state_lite);
    if out.len() > next_cap {
        out.truncate(next_cap);
    }
    Ok(out)
}

fn expand_fuzzy_beam_once_whole_genome_target_range(
    parents: &[FuzzyBeamState],
    start: usize,
    end: usize,
    y_train: &[f64],
    sum_y_train: f64,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    literal_summaries: &[DualLiteralSummary],
    base_rule_raws: &RuleRawScoreCache,
    next_cap: usize,
    params: &BeamSearchParams,
) -> Result<HashMap<RuleLexKey, FuzzyBeamStateLite>, String> {
    let layer = parents
        .first()
        .map(|p| p.rule.len().saturating_add(1))
        .unwrap_or(0);
    let mut local_best = HashMap::<RuleLexKey, FuzzyBeamStateLite>::with_capacity(next_cap.max(1));
    // Whole-genome target scans touch too many unique rules to keep a
    // worker-lifetime ancestor cache. Reuse the maps per target SNP only.
    let mut parent_raw_cache =
        RuleRawScoreCache::with_capacity(parents.len().saturating_mul(8).max(32));
    let mut ancestor_raw_cache =
        RuleAncestorBaselineCache::with_capacity(parents.len().saturating_mul(8).max(32));
    let prune_trigger = next_cap.saturating_mul(4).max(next_cap.saturating_add(1));
    for cand in start..end {
        if ((cand - start) & 127) == 0 {
            check_interrupt_fast()?;
        }
        parent_raw_cache.clear();
        ancestor_raw_cache.clear();
        let gid = group_ids[cand];
        let row_ge1 = row_prefix(ge1_train, row_words_train, cand, needed_words_train);
        let row_ge2 = row_prefix(ge2_train, row_words_train, cand, needed_words_train);
        let row_summary = literal_summaries[cand];
        for parent in parents.iter() {
            if candidate_group_is_excluded(&parent.rule, gid, params) {
                continue;
            }
            let intersections = dual_pair_intersections_for_params(
                parent.combined_train_ge1.as_slice(),
                parent.combined_train_ge2.as_slice(),
                row_ge1,
                row_ge2,
                y_train,
                n_train,
                params,
            );
            for &op in beam_binary_ops_for_rule(&parent.rule).iter() {
                for &negated in child_literal_negations_for_op(op).iter() {
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::Considered,
                        1,
                    );
                    let Some((train, train_n_ge2, train_sum_ge1, train_sum_ge2)) =
                        evaluate_child_train_from_parent_virtual_fuzzy_with_intersections(
                            parent,
                            row_summary,
                            sum_y_train,
                            n_train,
                            parent.rule.len() + 1,
                            op,
                            negated,
                            intersections,
                            params,
                        )
                    else {
                        continue;
                    };
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::TrainOk,
                        1,
                    );
                    let literal = BeamLiteral {
                        row_index: cand,
                        group_id: gid,
                        negated,
                    };
                    let canonical_rule =
                        canonical_commutative_child_rule(&parent.rule, op, literal);
                    let rule = if let Some(rule) = canonical_rule {
                        rule
                    } else {
                        let mut rule = parent.rule.clone();
                        rule.rest.push((op, literal));
                        rule
                    };
                    let single = literal_scores[literal_score_index(cand, negated)];
                    let max_singleton_train_raw =
                        parent.max_singleton_train_raw.max(single.train.raw_score);
                    let max_singleton_test_raw =
                        parent.max_singleton_test_raw.max(single.test.raw_score);
                    let direct_parent_train_raw = if rule.len() == 2 {
                        parent.train.raw_score.max(single.train.raw_score)
                    } else {
                        best_ancestor_raw_baseline_fuzzy_cached(
                            &rule,
                            y_train,
                            sum_y_train,
                            ge1_train,
                            ge2_train,
                            row_words_train,
                            n_rows,
                            n_train,
                            literal_scores,
                            true,
                            Some(base_rule_raws),
                            &mut parent_raw_cache,
                            &mut ancestor_raw_cache,
                            params.disable_parent_delta,
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
                        parent.train_abs_score,
                        rule.len(),
                        train_abs_score,
                        params,
                    ) {
                        continue;
                    }
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::AbsOk,
                        1,
                    );
                    if !keep_state_after_min_gain_pruning(rule.len(), train_score, params) {
                        continue;
                    }
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::GainOk,
                        1,
                    );
                    if !keep_child_after_parent_gain_pruning(&rule, train_score, params) {
                        continue;
                    }
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::ParentOk,
                        1,
                    );
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::Kept,
                        1,
                    );
                    let state = FuzzyBeamStateLite {
                        rule,
                        train,
                        train_n_ge2,
                        train_sum_ge1,
                        train_sum_ge2,
                        train_abs_score,
                        train_score,
                        max_singleton_train_raw,
                        max_singleton_test_raw,
                    };
                    let key = state.rule.lexical_key();
                    match local_best.entry(key) {
                        std::collections::hash_map::Entry::Vacant(slot) => {
                            slot.insert(state);
                        }
                        std::collections::hash_map::Entry::Occupied(mut slot) => {
                            if cmp_fuzzy_state_lite(&state, slot.get()) == std::cmp::Ordering::Less
                            {
                                slot.insert(state);
                            }
                        }
                    }
                }
            }
        }
        if local_best.len() > prune_trigger {
            prune_best_fuzzy_state_map(&mut local_best, next_cap);
        }
    }
    prune_best_fuzzy_state_map(&mut local_best, next_cap);
    Ok(local_best)
}

fn expand_fuzzy_beam_once_whole_genome_target_parallel(
    parents: &[FuzzyBeamState],
    y_train: &[f64],
    sum_y_train: f64,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    literal_summaries: &[DualLiteralSummary],
    params: &BeamSearchParams,
) -> Result<Vec<FuzzyBeamState>, String> {
    check_interrupt_fast()?;
    let next_cap = params.beam_width.min(n_rows.saturating_mul(4).max(1));
    if parents.is_empty() {
        return Ok(Vec::new());
    }
    let base_rule_raws = Arc::new(collect_known_rule_raw_scores_fuzzy(parents));
    let total_expand = parents
        .iter()
        .map(|parent| n_rows.saturating_mul(beam_child_branch_count_for_rule(&parent.rule)))
        .sum::<usize>();
    let next = if should_parallel(total_expand, params.allow_parallel) {
        let work = whole_genome_target_work_ranges(n_rows);
        let worker_maps = work
            .into_par_iter()
            .map(|(start, end)| {
                expand_fuzzy_beam_once_whole_genome_target_range(
                    parents,
                    start,
                    end,
                    y_train,
                    sum_y_train,
                    ge1_train,
                    ge2_train,
                    row_words_train,
                    n_rows,
                    needed_words_train,
                    n_train,
                    group_ids,
                    literal_scores,
                    literal_summaries,
                    base_rule_raws.as_ref(),
                    next_cap,
                    params,
                )
            })
            .collect::<Vec<Result<HashMap<RuleLexKey, FuzzyBeamStateLite>, String>>>();
        merge_best_fuzzy_state_maps(worker_maps, next_cap)?
    } else {
        expand_fuzzy_beam_once_whole_genome_target_range(
            parents,
            0,
            n_rows,
            y_train,
            sum_y_train,
            ge1_train,
            ge2_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
            group_ids,
            literal_scores,
            literal_summaries,
            base_rule_raws.as_ref(),
            next_cap,
            params,
        )?
        .into_values()
        .collect::<Vec<_>>()
    };
    let mut materialized = Vec::<FuzzyBeamState>::with_capacity(next.len());
    for cand in next.into_iter() {
        materialized.push(materialize_fuzzy_beam_state_lite(
            cand,
            ge1_train,
            ge2_train,
            row_words_train,
            n_rows,
            n_train,
        )?);
    }
    let layer = parents
        .first()
        .map(|p| p.rule.len().saturating_add(1))
        .unwrap_or(0);
    let out = filter_fuzzy_beam_candidates(materialized, next_cap, params);
    garfield_layer_debug_record_fuzzy_states(
        layer,
        GarfieldLayerDebugMetric::Retained,
        out.as_slice(),
    );
    Ok(out)
}

fn expand_fuzzy_beam_once_whole_genome_layer2(
    beam: &[FuzzyBeamState],
    y_train: &[f64],
    sum_y_train: f64,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    literal_summaries: &[DualLiteralSummary],
    params: &BeamSearchParams,
) -> Result<Vec<FuzzyBeamState>, String> {
    check_interrupt_fast()?;
    let mut parents = Vec::<FuzzyBeamState>::with_capacity(beam.len().saturating_mul(2));
    for node in beam.iter() {
        parents.extend(whole_genome_layer2_parent_variants_fuzzy(
            node,
            sum_y_train,
            ge1_train,
            ge2_train,
            row_words_train,
            needed_words_train,
            n_train,
            literal_scores,
            literal_summaries,
            params,
        ));
    }
    expand_fuzzy_beam_once_whole_genome_target_parallel(
        parents.as_slice(),
        y_train,
        sum_y_train,
        ge1_train,
        ge2_train,
        row_words_train,
        n_rows,
        needed_words_train,
        n_train,
        group_ids,
        literal_scores,
        literal_summaries,
        params,
    )
}

fn expand_fuzzy_beam_once(
    beam: &[FuzzyBeamState],
    y_train: &[f64],
    sum_y_train: f64,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    literal_summaries: &[DualLiteralSummary],
    params: &BeamSearchParams,
) -> Result<Vec<FuzzyBeamState>, String> {
    check_interrupt_fast()?;
    let next_cap = params.beam_width.min(n_rows.saturating_mul(4).max(1));
    if beam.is_empty() {
        return Ok(Vec::new());
    }
    let layer = beam
        .first()
        .map(|p| p.rule.len().saturating_add(1))
        .unwrap_or(0);
    let base_rule_raws = Arc::new(collect_known_rule_raw_scores_fuzzy(beam));
    let mut seq = Vec::<FuzzyBeamState>::with_capacity(next_cap);
    let mut parent_raw_cache = RuleRawScoreCache::new();
    let mut ancestor_raw_cache = RuleAncestorBaselineCache::new();
    let mut seen_commutative_children = HashSet::<Vec<(usize, bool, u8)>>::new();
    for node in beam.iter() {
        let (start, end) = expansion_row_bounds(&node.rule, n_rows);
        let blind_scan = child_rule_uses_blind_scan(node.rule.len());
        for cand in start..end {
            if ((cand - start) & 127) == 0 {
                check_interrupt_fast()?;
            }
            let gid = group_ids[cand];
            if candidate_group_is_excluded(&node.rule, gid, params) {
                continue;
            }
            let row_ge1 = row_prefix(ge1_train, row_words_train, cand, needed_words_train);
            let row_ge2 = row_prefix(ge2_train, row_words_train, cand, needed_words_train);
            let row_summary = literal_summaries[cand];
            let intersections = dual_pair_intersections_for_params(
                node.combined_train_ge1.as_slice(),
                node.combined_train_ge2.as_slice(),
                row_ge1,
                row_ge2,
                y_train,
                n_train,
                params,
            );
            for &op in beam_binary_ops_for_rule(&node.rule).iter() {
                for &negated in child_literal_negations_for_op(op).iter() {
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::Considered,
                        1,
                    );
                    let Some((train, train_n_ge2, train_sum_ge1, train_sum_ge2)) =
                        evaluate_child_train_from_parent_virtual_fuzzy_with_intersections(
                            node,
                            row_summary,
                            sum_y_train,
                            n_train,
                            node.rule.len() + 1,
                            op,
                            negated,
                            intersections,
                            params,
                        )
                    else {
                        continue;
                    };
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::TrainOk,
                        1,
                    );
                    let literal = BeamLiteral {
                        row_index: cand,
                        group_id: gid,
                        negated,
                    };
                    let canonical_rule = canonical_commutative_child_rule(&node.rule, op, literal);
                    if blind_scan {
                        if let Some(rule) = canonical_rule.as_ref() {
                            if !seen_commutative_children.insert(rule.lexical_key()) {
                                continue;
                            }
                        }
                    }
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
                        best_ancestor_raw_baseline_fuzzy_cached(
                            &rule,
                            y_train,
                            sum_y_train,
                            ge1_train,
                            ge2_train,
                            row_words_train,
                            n_rows,
                            n_train,
                            literal_scores,
                            true,
                            Some(base_rule_raws.as_ref()),
                            &mut parent_raw_cache,
                            &mut ancestor_raw_cache,
                            params.disable_parent_delta,
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
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::AbsOk,
                        1,
                    );
                    // Exhaustive seed depths enumerate all QC-valid children and leave
                    // gain-based culling to later beam-only layers / final output reranking.
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::GainOk,
                        1,
                    );
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::ParentOk,
                        1,
                    );
                    let t_clone = beam_detail_profile_start();
                    let mut child_ge1 = node.combined_train_ge1.clone();
                    let mut child_ge2 = node.combined_train_ge2.clone();
                    beam_detail_profile_end(t_clone, &GARFIELD_PROFILE_CLONE_BITS_NS);
                    apply_literal_inplace_dual(
                        child_ge1.as_mut_slice(),
                        child_ge2.as_mut_slice(),
                        row_ge1,
                        row_ge2,
                        op,
                        negated,
                        n_train,
                    );
                    push_top_k_fuzzy_states(
                        &mut seq,
                        FuzzyBeamState {
                            rule,
                            combined_train_ge1: child_ge1,
                            combined_train_ge2: child_ge2,
                            train,
                            train_n_ge2,
                            train_sum_ge1,
                            train_sum_ge2,
                            train_abs_score,
                            train_score,
                            max_singleton_train_raw,
                            max_singleton_test_raw,
                        },
                        next_cap,
                    );
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::Kept,
                        1,
                    );
                }
            }
        }
    }
    let out = filter_fuzzy_beam_candidates(seq, next_cap, params);
    garfield_layer_debug_record_fuzzy_states(
        layer,
        GarfieldLayerDebugMetric::Retained,
        out.as_slice(),
    );
    Ok(out)
}

fn expand_fuzzy_beam_once_parallel(
    beam: &[FuzzyBeamState],
    y_train: &[f64],
    sum_y_train: f64,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    literal_summaries: &[DualLiteralSummary],
    params: &BeamSearchParams,
) -> Result<Vec<FuzzyBeamState>, String> {
    check_interrupt_fast()?;
    if beam.is_empty() {
        return Ok(Vec::new());
    }
    let next_cap = params.beam_width.min(n_rows.saturating_mul(4).max(1));
    let total_expand = beam
        .iter()
        .map(|node| {
            let (start, end) = expansion_row_bounds(&node.rule, n_rows);
            end.saturating_sub(start)
                .saturating_mul(beam_child_branch_count_for_rule(&node.rule))
        })
        .sum::<usize>();
    if !should_parallel(total_expand, params.allow_parallel) {
        return expand_fuzzy_beam_once(
            beam,
            y_train,
            sum_y_train,
            ge1_train,
            ge2_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
            group_ids,
            literal_scores,
            literal_summaries,
            params,
        );
    }

    let layer = beam
        .first()
        .map(|p| p.rule.len().saturating_add(1))
        .unwrap_or(0);
    let base_rule_raws = Arc::new(collect_known_rule_raw_scores_fuzzy(beam));
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

    // Keep the same virtual scoring and pruning as the serial standard-fuzzy
    // path. Workers only defer bitplane materialization until after merging.
    let worker_maps = work
        .into_par_iter()
        .map(|(bi, start, end)| {
            let node = &beam[bi];
            let mut local_best =
                HashMap::<RuleLexKey, FuzzyBeamStateLite>::with_capacity(next_cap.max(1));
            let mut parent_raw_cache = RuleRawScoreCache::new();
            let mut ancestor_raw_cache = RuleAncestorBaselineCache::new();
            let mut seen_commutative_children = HashSet::<Vec<(usize, bool, u8)>>::new();
            let blind_scan = child_rule_uses_blind_scan(node.rule.len());

            for cand in start..end {
                if ((cand - start) & 127) == 0 {
                    check_interrupt_fast()?;
                }
                let gid = group_ids[cand];
                if candidate_group_is_excluded(&node.rule, gid, params) {
                    continue;
                }
                let row_ge1 = row_prefix(ge1_train, row_words_train, cand, needed_words_train);
                let row_ge2 = row_prefix(ge2_train, row_words_train, cand, needed_words_train);
                let row_summary = literal_summaries[cand];
                let intersections = dual_pair_intersections_for_params(
                    node.combined_train_ge1.as_slice(),
                    node.combined_train_ge2.as_slice(),
                    row_ge1,
                    row_ge2,
                    y_train,
                    n_train,
                    params,
                );
                for &op in beam_binary_ops_for_rule(&node.rule).iter() {
                    for &negated in child_literal_negations_for_op(op).iter() {
                        garfield_layer_debug_add(
                            layer,
                            garfield_layer_debug_op_family(op),
                            GarfieldLayerDebugMetric::Considered,
                            1,
                        );
                        let Some((train, train_n_ge2, train_sum_ge1, train_sum_ge2)) =
                            evaluate_child_train_from_parent_virtual_fuzzy_with_intersections(
                                node,
                                row_summary,
                                sum_y_train,
                                n_train,
                                node.rule.len() + 1,
                                op,
                                negated,
                                intersections,
                                params,
                            )
                        else {
                            continue;
                        };
                        garfield_layer_debug_add(
                            layer,
                            garfield_layer_debug_op_family(op),
                            GarfieldLayerDebugMetric::TrainOk,
                            1,
                        );
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
                            best_ancestor_raw_baseline_fuzzy_cached(
                                &rule,
                                y_train,
                                sum_y_train,
                                ge1_train,
                                ge2_train,
                                row_words_train,
                                n_rows,
                                n_train,
                                literal_scores,
                                true,
                                Some(base_rule_raws.as_ref()),
                                &mut parent_raw_cache,
                                &mut ancestor_raw_cache,
                                params.disable_parent_delta,
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
                        // Standard fuzzy exhaustive seed layers intentionally
                        // retain all QC-valid children; later layers/output
                        // apply the configured ranking and pruning.
                        garfield_layer_debug_add(
                            layer,
                            garfield_layer_debug_op_family(op),
                            GarfieldLayerDebugMetric::AbsOk,
                            1,
                        );
                        garfield_layer_debug_add(
                            layer,
                            garfield_layer_debug_op_family(op),
                            GarfieldLayerDebugMetric::GainOk,
                            1,
                        );
                        garfield_layer_debug_add(
                            layer,
                            garfield_layer_debug_op_family(op),
                            GarfieldLayerDebugMetric::ParentOk,
                            1,
                        );
                        let state = FuzzyBeamStateLite {
                            rule,
                            train,
                            train_n_ge2,
                            train_sum_ge1,
                            train_sum_ge2,
                            train_abs_score,
                            train_score,
                            max_singleton_train_raw,
                            max_singleton_test_raw,
                        };
                        let key = state.rule.lexical_key();
                        match local_best.entry(key) {
                            std::collections::hash_map::Entry::Vacant(slot) => {
                                slot.insert(state);
                            }
                            std::collections::hash_map::Entry::Occupied(mut slot) => {
                                if cmp_fuzzy_state_lite(&state, slot.get())
                                    == std::cmp::Ordering::Less
                                {
                                    slot.insert(state);
                                }
                            }
                        }
                    }
                }
            }
            prune_best_fuzzy_state_map(&mut local_best, next_cap);
            Ok(local_best)
        })
        .collect::<Vec<Result<HashMap<RuleLexKey, FuzzyBeamStateLite>, String>>>();

    let next = merge_best_fuzzy_state_maps(worker_maps, next_cap)?;
    let mut materialized = Vec::<FuzzyBeamState>::with_capacity(next.len());
    for cand in next.into_iter() {
        materialized.push(materialize_fuzzy_beam_state_lite(
            cand,
            ge1_train,
            ge2_train,
            row_words_train,
            n_rows,
            n_train,
        )?);
    }
    let out = filter_fuzzy_beam_candidates(materialized, next_cap, params);
    garfield_layer_debug_record_fuzzy_states(
        layer,
        GarfieldLayerDebugMetric::Retained,
        out.as_slice(),
    );
    Ok(out)
}

fn expand_fuzzy_states_exhaustive(
    frontier: &[FuzzyBeamState],
    y_train: &[f64],
    sum_y_train: f64,
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    needed_words_train: usize,
    n_train: usize,
    group_ids: &[usize],
    literal_scores: &[LiteralSingletonScore],
    literal_summaries: &[DualLiteralSummary],
    params: &BeamSearchParams,
) -> Result<Vec<FuzzyBeamState>, String> {
    check_interrupt_fast()?;
    let layer = frontier
        .first()
        .map(|p| p.rule.len().saturating_add(1))
        .unwrap_or(0);
    let mut best = HashMap::<RuleLexKey, FuzzyBeamState>::new();
    let base_rule_raws = collect_known_rule_raw_scores_fuzzy(frontier);
    let mut parent_raw_cache = RuleRawScoreCache::new();
    let mut ancestor_raw_cache = RuleAncestorBaselineCache::new();
    for node in frontier.iter() {
        let cand_start = node.rule.last_row_index() + 1;
        for cand in cand_start..n_rows {
            if ((cand - cand_start) & 127) == 0 {
                check_interrupt_fast()?;
            }
            let gid = group_ids[cand];
            if candidate_group_is_excluded(&node.rule, gid, params) {
                continue;
            }
            let row_ge1 = row_prefix(ge1_train, row_words_train, cand, needed_words_train);
            let row_ge2 = row_prefix(ge2_train, row_words_train, cand, needed_words_train);
            let row_summary = literal_summaries[cand];
            let intersections = dual_pair_intersections_for_params(
                node.combined_train_ge1.as_slice(),
                node.combined_train_ge2.as_slice(),
                row_ge1,
                row_ge2,
                y_train,
                n_train,
                params,
            );
            for &op in beam_binary_ops_for_rule(&node.rule).iter() {
                for &negated in child_literal_negations_for_op(op).iter() {
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::Considered,
                        1,
                    );
                    let Some((train, train_n_ge2, train_sum_ge1, train_sum_ge2)) =
                        evaluate_child_train_from_parent_virtual_fuzzy_with_intersections(
                            node,
                            row_summary,
                            sum_y_train,
                            n_train,
                            node.rule.len() + 1,
                            op,
                            negated,
                            intersections,
                            params,
                        )
                    else {
                        continue;
                    };
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::TrainOk,
                        1,
                    );
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
                        best_ancestor_raw_baseline_fuzzy_cached(
                            &rule,
                            y_train,
                            sum_y_train,
                            ge1_train,
                            ge2_train,
                            row_words_train,
                            n_rows,
                            n_train,
                            literal_scores,
                            true,
                            Some(&base_rule_raws),
                            &mut parent_raw_cache,
                            &mut ancestor_raw_cache,
                            params.disable_parent_delta,
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
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::AbsOk,
                        1,
                    );
                    if !keep_state_after_min_gain_pruning(rule.len(), train_score, params) {
                        continue;
                    }
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::GainOk,
                        1,
                    );
                    if !keep_child_after_parent_gain_pruning(&rule, train_score, params) {
                        continue;
                    }
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::ParentOk,
                        1,
                    );
                    let t_clone = beam_detail_profile_start();
                    let mut child_ge1 = node.combined_train_ge1.clone();
                    let mut child_ge2 = node.combined_train_ge2.clone();
                    beam_detail_profile_end(t_clone, &GARFIELD_PROFILE_CLONE_BITS_NS);
                    apply_literal_inplace_dual(
                        child_ge1.as_mut_slice(),
                        child_ge2.as_mut_slice(),
                        row_ge1,
                        row_ge2,
                        op,
                        negated,
                        n_train,
                    );
                    let state = FuzzyBeamState {
                        rule,
                        combined_train_ge1: child_ge1,
                        combined_train_ge2: child_ge2,
                        train,
                        train_n_ge2,
                        train_sum_ge1,
                        train_sum_ge2,
                        train_abs_score,
                        train_score,
                        max_singleton_train_raw,
                        max_singleton_test_raw,
                    };
                    garfield_layer_debug_add(
                        layer,
                        garfield_layer_debug_op_family(op),
                        GarfieldLayerDebugMetric::Kept,
                        1,
                    );
                    match best.entry(state.rule.lexical_key()) {
                        std::collections::hash_map::Entry::Vacant(slot) => {
                            slot.insert(state);
                        }
                        std::collections::hash_map::Entry::Occupied(mut slot) => {
                            if cmp_fuzzy_state(&state, slot.get()) == std::cmp::Ordering::Less {
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
    let out = filter_fuzzy_beam_candidates(out, width, params);
    garfield_layer_debug_record_fuzzy_states(
        layer,
        GarfieldLayerDebugMetric::Retained,
        out.as_slice(),
    );
    Ok(out)
}

fn collect_known_rule_raw_scores_fuzzy(states: &[FuzzyBeamState]) -> RuleRawScoreCache {
    let mut out = RuleRawScoreCache::with_capacity(states.len());
    for state in states.iter() {
        cache_rule_raw_score(&mut out, &state.rule, state.train.raw_score);
    }
    out
}

fn beam_search_train_test_continuous_fuzzy_impl(
    y_train: &[f64],
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    n_train: usize,
    y_test: &[f64],
    ge1_test: &[u64],
    ge2_test: &[u64],
    row_words_test: usize,
    n_test: usize,
    group_ids: &[usize],
    params: BeamSearchParams,
    literal_scores_override: Option<&[LiteralSingletonScore]>,
) -> Result<Vec<BeamRuleCandidate>, String> {
    let beam_t0 = Instant::now();
    garfield_layer_debug_reset();
    let out = (|| {
        let (needed_words_train, _needed_words_test) = validate_search_inputs_fuzzy(
            y_train,
            ge1_train,
            ge2_train,
            row_words_train,
            n_rows,
            n_train,
            y_test,
            ge1_test,
            ge2_test,
            row_words_test,
            n_test,
            group_ids,
            &params,
        )?;
        let sum_y_train = y_train.iter().take(n_train).copied().sum::<f64>();
        let sum_y_test = y_test.iter().take(n_test).copied().sum::<f64>();
        let literal_scores_storage;
        let literal_scores = if let Some(scores) = literal_scores_override {
            let expected = n_rows.saturating_mul(2);
            if scores.len() != expected {
                return Err(format!(
                    "garfield::beam_search_train_test_continuous_fuzzy literal-score override length mismatch: got {}, expected {}",
                    scores.len(),
                    expected,
                ));
            }
            scores
        } else {
            literal_scores_storage = precompute_literal_singleton_scores_fuzzy(
                y_train,
                n_train,
                y_test,
                n_test,
                ge1_train,
                ge2_train,
                row_words_train,
                ge1_test,
                ge2_test,
                row_words_test,
                n_rows,
            )?;
            literal_scores_storage.as_slice()
        };
        let literal_summaries = precompute_dual_literal_summaries(
            y_train,
            ge1_train,
            ge2_train,
            row_words_train,
            n_rows,
            needed_words_train,
            n_train,
        );
        let max_depth = params.max_pick.min(n_rows);
        let exhaustive_depth = params.exhaustive_depth.max(1).min(max_depth);
        let mode_name = if params.whole_genome_dev_mode {
            "wholegenome_fuzzy"
        } else {
            "standard_fuzzy"
        };
        let mut kept_all = Vec::<FuzzyBeamState>::new();
        garfield_layer_rss_breakpoint(mode_name, 0, "literal_ready", 0, 0)?;
        garfield_layer_rss_breakpoint(mode_name, 0, "summary_ready", 0, 0)?;
        garfield_layer_rss_breakpoint(mode_name, 1, "start", 0, kept_all.len())?;
        let mut beam = if exhaustive_depth > 1 {
            let exhaustive_initial = build_initial_fuzzy_states_exhaustive(
                sum_y_train,
                ge1_train,
                ge2_train,
                row_words_train,
                needed_words_train,
                n_train,
                group_ids,
                literal_scores,
                literal_summaries.as_slice(),
                &params,
            )?;
            kept_all.extend(exhaustive_initial.iter().cloned());
            let mut frontier = exhaustive_initial;
            for _depth in 2..=exhaustive_depth {
                let next = expand_fuzzy_states_exhaustive(
                    frontier.as_slice(),
                    y_train,
                    sum_y_train,
                    ge1_train,
                    ge2_train,
                    row_words_train,
                    n_rows,
                    needed_words_train,
                    n_train,
                    group_ids,
                    literal_scores,
                    literal_summaries.as_slice(),
                    &params,
                )?;
                if next.is_empty() {
                    frontier = Vec::new();
                    break;
                }
                kept_all.extend(next.iter().cloned());
                frontier = next;
            }
            sort_truncate_fuzzy_states(frontier, params.beam_width.max(1))
        } else {
            let beam = build_initial_fuzzy_beam(
                sum_y_train,
                ge1_train,
                ge2_train,
                row_words_train,
                needed_words_train,
                n_train,
                group_ids,
                literal_scores,
                literal_summaries.as_slice(),
                &params,
            )?;
            kept_all.extend(beam.iter().cloned());
            beam
        };
        garfield_layer_rss_breakpoint(mode_name, 1, "end", beam.len(), kept_all.len())?;
        let mut depth_start = exhaustive_depth + 1;
        if params.whole_genome_dev_mode && exhaustive_depth == 1 && max_depth >= 2 {
            garfield_layer_rss_breakpoint(mode_name, 2, "start", beam.len(), kept_all.len())?;
            let next = expand_fuzzy_beam_once_whole_genome_layer2(
                beam.as_slice(),
                y_train,
                sum_y_train,
                ge1_train,
                ge2_train,
                row_words_train,
                n_rows,
                needed_words_train,
                n_train,
                group_ids,
                literal_scores,
                literal_summaries.as_slice(),
                &params,
            )?;
            if !next.is_empty() {
                kept_all.extend(next.iter().cloned());
                beam = next;
            }
            garfield_layer_rss_breakpoint(mode_name, 2, "end", beam.len(), kept_all.len())?;
            depth_start = 3;
        }
        for depth in depth_start..=max_depth {
            garfield_layer_rss_breakpoint(mode_name, depth, "start", beam.len(), kept_all.len())?;
            let next = if params.whole_genome_dev_mode {
                expand_fuzzy_beam_once_whole_genome_target_parallel(
                    beam.as_slice(),
                    y_train,
                    sum_y_train,
                    ge1_train,
                    ge2_train,
                    row_words_train,
                    n_rows,
                    needed_words_train,
                    n_train,
                    group_ids,
                    literal_scores,
                    literal_summaries.as_slice(),
                    &params,
                )?
            } else {
                expand_fuzzy_beam_once_parallel(
                    beam.as_slice(),
                    y_train,
                    sum_y_train,
                    ge1_train,
                    ge2_train,
                    row_words_train,
                    n_rows,
                    needed_words_train,
                    n_train,
                    group_ids,
                    literal_scores,
                    literal_summaries.as_slice(),
                    &params,
                )?
            };
            if next.is_empty() {
                garfield_layer_rss_breakpoint(
                    mode_name,
                    depth,
                    "empty",
                    beam.len(),
                    kept_all.len(),
                )?;
                break;
            }
            kept_all.extend(next.iter().cloned());
            beam = next;
            garfield_layer_rss_breakpoint(mode_name, depth, "end", beam.len(), kept_all.len())?;
        }
        let shared_inputs = literal_inputs_are_shared(
            y_train,
            n_train,
            y_test,
            n_test,
            ge1_train,
            row_words_train,
            ge1_test,
            row_words_test,
            n_rows,
        ) && ge2_train == ge2_test;
        let retained = dedup_fuzzy_states_by_support_signature(
            kept_all,
            ge1_test,
            ge2_test,
            row_words_test,
            n_rows,
            n_test,
            shared_inputs,
        )?;
        let mut best_by_rule =
            HashMap::<Vec<(usize, bool, u8)>, BeamRuleCandidate>::with_capacity(retained.len());
        for state in retained.into_iter() {
            let (test_ge1_bits, test_ge2_bits) = if shared_inputs {
                let t_clone = beam_detail_profile_start();
                let bits = (
                    state.combined_train_ge1.clone(),
                    state.combined_train_ge2.clone(),
                );
                beam_detail_profile_end(t_clone, &GARFIELD_PROFILE_CLONE_BITS_NS);
                bits
            } else {
                materialize_rule_bits_dual(
                    &state.rule,
                    ge1_test,
                    ge2_test,
                    row_words_test,
                    n_rows,
                    n_test,
                )?
            };
            let test = score_rule_continuous_from_dual(
                &state.rule,
                y_test,
                test_ge1_bits.as_slice(),
                test_ge2_bits.as_slice(),
                n_test,
                sum_y_test,
                0.0,
                0.0,
            );
            if !keep_rule_after_dosage_maf_pruning(&test, &params) {
                continue;
            }
            let test_score = final_test_score_for_rule_fuzzy(
                &state.rule,
                &test,
                y_test,
                sum_y_test,
                ge1_test,
                ge2_test,
                row_words_test,
                n_rows,
                n_test,
                literal_scores,
                &params,
            )?;
            let train_score = {
                let direct_parent_train_raw = best_ancestor_raw_baseline_fuzzy(
                    &state.rule,
                    y_train,
                    sum_y_train,
                    ge1_train,
                    ge2_train,
                    row_words_train,
                    n_rows,
                    n_train,
                    literal_scores,
                    true,
                    params.disable_parent_delta,
                )?;
                let bucket = bucket_from_rule_with_complexity(
                    &state.rule,
                    state.train.dosage_maf,
                    params.null_complexity_bin,
                );
                rank_rule_score_components_with_bucket(
                    bucket,
                    state.rule.len(),
                    state.rule.not_count(),
                    state.train.raw_score,
                    direct_parent_train_raw,
                    &params,
                    true,
                )
            };
            let cand = canonicalize_singleton_output_candidate(
                BeamRuleCandidate {
                    rule: state.rule,
                    train_score,
                    test_score,
                    train: state.train,
                    test,
                },
                literal_scores,
                &params,
            );
            if cand.rule.len() > exhaustive_depth
                && !keep_child_after_parent_gain_pruning(&cand.rule, cand.test_score, &params)
            {
                continue;
            }
            match best_by_rule.entry(cand.rule.lexical_key()) {
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
        garfield_layer_debug_dump(mode_name, max_depth);
        Ok(out)
    })();
    GARFIELD_BEAM_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
    GARFIELD_BEAM_PROFILE_TOTAL_NS.fetch_add(elapsed_ns_saturating(beam_t0), Ordering::Relaxed);
    out
}

pub fn beam_search_train_test_continuous_fuzzy(
    y_train: &[f64],
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    n_train: usize,
    y_test: &[f64],
    ge1_test: &[u64],
    ge2_test: &[u64],
    row_words_test: usize,
    n_test: usize,
    group_ids: &[usize],
    params: BeamSearchParams,
) -> Result<Vec<BeamRuleCandidate>, String> {
    beam_search_train_test_continuous_fuzzy_impl(
        y_train,
        ge1_train,
        ge2_train,
        row_words_train,
        n_rows,
        n_train,
        y_test,
        ge1_test,
        ge2_test,
        row_words_test,
        n_test,
        group_ids,
        params,
        None,
    )
}

pub(crate) fn beam_search_train_test_continuous_fuzzy_with_literal_scores(
    y_train: &[f64],
    ge1_train: &[u64],
    ge2_train: &[u64],
    row_words_train: usize,
    n_rows: usize,
    n_train: usize,
    y_test: &[f64],
    ge1_test: &[u64],
    ge2_test: &[u64],
    row_words_test: usize,
    n_test: usize,
    group_ids: &[usize],
    params: BeamSearchParams,
    literal_scores: &[LiteralSingletonScore],
) -> Result<Vec<BeamRuleCandidate>, String> {
    beam_search_train_test_continuous_fuzzy_impl(
        y_train,
        ge1_train,
        ge2_train,
        row_words_train,
        n_rows,
        n_train,
        y_test,
        ge1_test,
        ge2_test,
        row_words_test,
        n_test,
        group_ids,
        params,
        Some(literal_scores),
    )
}

fn final_test_score_for_rule_fuzzy(
    rule: &BeamRule,
    test: &ContinuousRuleScore,
    y_test: &[f64],
    sum_y_test: f64,
    ge1_test: &[u64],
    ge2_test: &[u64],
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    literal_scores: &[LiteralSingletonScore],
    params: &BeamSearchParams,
) -> Result<f64, String> {
    let child_bucket =
        bucket_from_rule_with_complexity(rule, test.dosage_maf, params.null_complexity_bin);
    let direct_parent_test_raw = best_ancestor_raw_baseline_fuzzy(
        rule,
        y_test,
        sum_y_test,
        ge1_test,
        ge2_test,
        row_words_test,
        n_rows,
        n_test,
        literal_scores,
        false,
        params.disable_parent_delta,
    )?;
    let child_abs_score = rank_rule_score_components_base(
        rule.len(),
        rule.not_count(),
        test.raw_score,
        direct_parent_test_raw,
        params,
    );
    let threshold = null_penalty_for_bucket(child_bucket, params, false);
    Ok(child_abs_score - threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_xor_lmaf_requires_both_directional_substates() {
        let params = BeamSearchParams {
            maf_threshold: 0.2,
            filter_xor_substates: true,
            ..BeamSearchParams::default()
        };
        let parent = score_cont_centered_gain_from_sum_and_n_hit(0.0, 0.0, 100, 30);
        let rare_row = score_cont_centered_gain_from_sum_and_n_hit(0.0, 0.0, 100, 5);
        let common_row = score_cont_centered_gain_from_sum_and_n_hit(0.0, 0.0, 100, 30);
        let no_overlap = BinaryPairIntersection { n: 0, sum: 0.0 };

        assert!(evaluate_child_train_from_parent_virtual_with_intersection(
            &parent,
            &common_row,
            no_overlap,
            0.0,
            100,
            2,
            BeamBinaryOp::Xor,
            false,
            &params,
        )
        .is_some());
        assert!(evaluate_child_train_from_parent_virtual_with_intersection(
            &parent,
            &rare_row,
            no_overlap,
            0.0,
            100,
            2,
            BeamBinaryOp::Xor,
            false,
            &params,
        )
        .is_none());
    }

    #[test]
    fn test_binary_xor_lmaf_filter_can_be_disabled() {
        let params = BeamSearchParams {
            maf_threshold: 0.2,
            filter_xor_substates: false,
            ..BeamSearchParams::default()
        };
        let parent = score_cont_centered_gain_from_sum_and_n_hit(0.0, 0.0, 100, 30);
        let rare_row = score_cont_centered_gain_from_sum_and_n_hit(0.0, 0.0, 100, 5);

        assert!(evaluate_child_train_from_parent_virtual_with_intersection(
            &parent,
            &rare_row,
            BinaryPairIntersection { n: 0, sum: 0.0 },
            0.0,
            100,
            2,
            BeamBinaryOp::Xor,
            false,
            &params,
        )
        .is_some());
    }

    fn test_fuzzy_state(n_ge1: usize, n_ge2: usize) -> FuzzyBeamState {
        FuzzyBeamState {
            rule: BeamRule {
                first: BeamLiteral {
                    row_index: 0,
                    group_id: 0,
                    negated: false,
                },
                rest: Vec::new(),
            },
            combined_train_ge1: Vec::new(),
            combined_train_ge2: Vec::new(),
            train: score_cont_centered_gain_dual_from_summary(0.0, 100, n_ge1, n_ge2, 0.0, 0.0),
            train_n_ge2: n_ge2,
            train_sum_ge1: 0.0,
            train_sum_ge2: 0.0,
            train_abs_score: 0.0,
            train_score: 0.0,
            max_singleton_train_raw: 0.0,
            max_singleton_test_raw: 0.0,
        }
    }

    #[test]
    fn test_dual_xor_lmaf_requires_both_directional_substates() {
        let params = BeamSearchParams {
            maf_threshold: 0.1,
            filter_xor_substates: true,
            ..BeamSearchParams::default()
        };
        let parent = test_fuzzy_state(30, 0);
        let rare_row = DualLiteralSummary {
            pos_n_ge1: 5,
            pos_n_ge2: 0,
            pos_sum_ge1: 0.0,
            pos_sum_ge2: 0.0,
        };
        let no_overlap = DualPairIntersections::default();

        assert!(
            evaluate_child_train_from_parent_virtual_fuzzy_with_intersections(
                &parent,
                rare_row,
                0.0,
                100,
                2,
                BeamBinaryOp::Xor,
                false,
                no_overlap,
                &params,
            )
            .is_none()
        );
    }

    #[test]
    fn test_dual_xor_lmaf_filter_can_be_disabled() {
        let params = BeamSearchParams {
            maf_threshold: 0.1,
            filter_xor_substates: false,
            ..BeamSearchParams::default()
        };
        let parent = test_fuzzy_state(30, 0);
        let rare_row = DualLiteralSummary {
            pos_n_ge1: 5,
            pos_n_ge2: 0,
            pos_sum_ge1: 0.0,
            pos_sum_ge2: 0.0,
        };

        assert!(
            evaluate_child_train_from_parent_virtual_fuzzy_with_intersections(
                &parent,
                rare_row,
                0.0,
                100,
                2,
                BeamBinaryOp::Xor,
                false,
                DualPairIntersections::default(),
                &params,
            )
            .is_some()
        );
    }

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

    fn pack_dual_rows(rows: &[Vec<u8>], n_samples: usize) -> (Vec<u64>, Vec<u64>, usize) {
        let row_words = words_for_samples(n_samples);
        let mut ge1 = vec![0u64; rows.len() * row_words];
        let mut ge2 = vec![0u64; rows.len() * row_words];
        for (ri, row) in rows.iter().enumerate() {
            for (i, &g) in row.iter().take(n_samples).enumerate() {
                if g >= 1 {
                    ge1[ri * row_words + (i >> 6)] |= 1u64 << (i & 63);
                }
                if g >= 2 {
                    ge2[ri * row_words + (i >> 6)] |= 1u64 << (i & 63);
                }
            }
        }
        (ge1, ge2, row_words)
    }

    fn unpack_dual_row(ge1: &[u64], ge2: &[u64], n_samples: usize) -> Vec<u8> {
        (0..n_samples)
            .map(|i| {
                (((ge1[i >> 6] >> (i & 63)) & 1u64) + ((ge2[i >> 6] >> (i & 63)) & 1u64)) as u8
            })
            .collect()
    }

    fn init_python_for_tests() {
        pyo3::Python::initialize();
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
        .unwrap()
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

    fn fuzzy_beam_state_from_rule_for_test(
        rule: BeamRule,
        y: &[f64],
        ge1: &[u64],
        ge2: &[u64],
        row_words: usize,
        n_rows: usize,
        literal_scores: &[LiteralSingletonScore],
    ) -> FuzzyBeamState {
        let (combined_train_ge1, combined_train_ge2) =
            materialize_rule_bits_dual(&rule, ge1, ge2, row_words, n_rows, y.len()).unwrap();
        let sum_y = y.iter().copied().sum::<f64>();
        let (n_hit, n_ge2, sum_ge1, sum_ge2) =
            dual_packed_summary(&combined_train_ge1, &combined_train_ge2, y, y.len());
        let train = score_cont_centered_gain_dual_from_summary(
            sum_y,
            y.len(),
            n_hit,
            n_ge2,
            sum_ge1,
            sum_ge2,
        );
        FuzzyBeamState {
            rule: rule.clone(),
            combined_train_ge1,
            combined_train_ge2,
            train,
            train_n_ge2: n_ge2,
            train_sum_ge1: sum_ge1,
            train_sum_ge2: sum_ge2,
            train_abs_score: train.raw_score,
            train_score: train.raw_score,
            max_singleton_train_raw: rule_max_singleton_raw(&rule, literal_scores, true),
            max_singleton_test_raw: rule_max_singleton_raw(&rule, literal_scores, false),
        }
    }

    fn assert_same_beam_states(actual: &[BeamState], expected: &[BeamState]) {
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(expected.iter()) {
            assert_eq!(got.rule.lexical_key(), want.rule.lexical_key());
            assert_eq!(got.combined_train, want.combined_train);
            assert!((got.train.raw_score - want.train.raw_score).abs() < 1e-12);
            assert!((got.train_abs_score - want.train_abs_score).abs() < 1e-12);
            assert!((got.train_score - want.train_score).abs() < 1e-12);
        }
    }

    fn assert_same_fuzzy_beam_states(actual: &[FuzzyBeamState], expected: &[FuzzyBeamState]) {
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(expected.iter()) {
            assert_eq!(got.rule.lexical_key(), want.rule.lexical_key());
            assert_eq!(got.combined_train_ge1, want.combined_train_ge1);
            assert_eq!(got.combined_train_ge2, want.combined_train_ge2);
            assert_eq!(got.train_n_ge2, want.train_n_ge2);
            assert!((got.train_sum_ge1 - want.train_sum_ge1).abs() < 1e-12);
            assert!((got.train_sum_ge2 - want.train_sum_ge2).abs() < 1e-12);
            assert!((got.train.raw_score - want.train.raw_score).abs() < 1e-12);
            assert!((got.train_score - want.train_score).abs() < 1e-12);
        }
    }

    #[test]
    fn test_batched_literal_singleton_precompute_matches_per_unit() {
        let y = vec![0.2, 1.1, -0.4, 0.8, 1.6, -0.3];
        let rows_a = vec![vec![1, 0, 1, 0, 0, 1], vec![0, 1, 1, 0, 1, 0]];
        let rows_b = vec![vec![1, 1, 0, 0, 1, 0], vec![0, 0, 1, 1, 0, 1]];
        let (bits_a, row_words_a) = pack_rows(&rows_a, y.len());
        let (bits_b, row_words_b) = pack_rows(&rows_b, y.len());
        let expected_a = literal_scores_for_test(&y, bits_a.as_slice(), row_words_a, rows_a.len());
        let expected_b = literal_scores_for_test(&y, bits_b.as_slice(), row_words_b, rows_b.len());
        let actual = precompute_literal_singleton_scores_batched(
            y.as_slice(),
            y.len(),
            y.as_slice(),
            y.len(),
            &[
                LiteralScoreBatchRequest {
                    bits_train: bits_a.as_slice(),
                    row_words_train: row_words_a,
                    bits_test: bits_a.as_slice(),
                    row_words_test: row_words_a,
                    n_rows: rows_a.len(),
                },
                LiteralScoreBatchRequest {
                    bits_train: bits_b.as_slice(),
                    row_words_train: row_words_b,
                    bits_test: bits_b.as_slice(),
                    row_words_test: row_words_b,
                    n_rows: rows_b.len(),
                },
            ],
        )
        .unwrap();
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0], expected_a);
        assert_eq!(actual[1], expected_b);
    }

    #[test]
    fn test_train_bit_dedup_keeps_best_state_without_changing_bits() {
        init_python_for_tests();
        let y = vec![0.2, 1.1, -0.4, 0.8];
        let train_rows = vec![vec![1, 0, 0, 1], vec![1, 0, 0, 1], vec![0, 1, 1, 0]];
        let (train_bits, row_words_train) = pack_rows(&train_rows, y.len());
        let literal_scores =
            literal_scores_for_test(&y, train_bits.as_slice(), row_words_train, train_rows.len());
        let state_a = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 0,
                    group_id: 0,
                    negated: false,
                },
                rest: Vec::new(),
            },
            &y,
            train_bits.as_slice(),
            row_words_train,
            train_rows.len(),
            literal_scores.as_slice(),
        );
        let state_b = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
                rest: Vec::new(),
            },
            &y,
            train_bits.as_slice(),
            row_words_train,
            train_rows.len(),
            literal_scores.as_slice(),
        );
        let state_c = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 2,
                    group_id: 2,
                    negated: false,
                },
                rest: Vec::new(),
            },
            &y,
            train_bits.as_slice(),
            row_words_train,
            train_rows.len(),
            literal_scores.as_slice(),
        );

        let deduped =
            dedup_states_by_train_bits(vec![state_c.clone(), state_b.clone(), state_a.clone()]);
        let mut expected = vec![state_a, state_c];
        expected.sort_by(cmp_state);
        assert_same_beam_states(deduped.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_support_signature_dedup_keeps_distinct_test_support() {
        init_python_for_tests();
        let y = vec![0.2, 1.1, -0.4, 0.8];
        let train_rows = vec![vec![1, 0, 0, 1], vec![1, 0, 0, 1]];
        let test_rows = vec![vec![1, 0, 1, 0], vec![0, 1, 1, 0]];
        let (train_bits, row_words_train) = pack_rows(&train_rows, y.len());
        let (test_bits, row_words_test) = pack_rows(&test_rows, y.len());
        let literal_scores =
            literal_scores_for_test(&y, train_bits.as_slice(), row_words_train, train_rows.len());
        let state_a = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 0,
                    group_id: 0,
                    negated: false,
                },
                rest: Vec::new(),
            },
            &y,
            train_bits.as_slice(),
            row_words_train,
            train_rows.len(),
            literal_scores.as_slice(),
        );
        let state_b = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
                rest: Vec::new(),
            },
            &y,
            train_bits.as_slice(),
            row_words_train,
            train_rows.len(),
            literal_scores.as_slice(),
        );

        let deduped = dedup_states_by_support_signature(
            vec![state_a.clone(), state_b.clone()],
            test_bits.as_slice(),
            row_words_test,
            train_rows.len(),
            y.len(),
            false,
        )
        .unwrap();
        let mut expected_distinct = vec![state_a.clone(), state_b.clone()];
        expected_distinct.sort_by(cmp_state);
        assert_same_beam_states(deduped.as_slice(), expected_distinct.as_slice());

        let shared = dedup_states_by_support_signature(
            vec![state_a, state_b],
            train_bits.as_slice(),
            row_words_train,
            train_rows.len(),
            y.len(),
            true,
        )
        .unwrap();
        let expected_shared = vec![beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 0,
                    group_id: 0,
                    negated: false,
                },
                rest: Vec::new(),
            },
            &y,
            train_bits.as_slice(),
            row_words_train,
            train_rows.len(),
            literal_scores.as_slice(),
        )];
        assert_same_beam_states(shared.as_slice(), expected_shared.as_slice());
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
    fn test_materialize_rule_bits_dual_respects_min_and_not() {
        let rows = vec![vec![0u8, 1, 2, 1], vec![2u8, 1, 0, 2]];
        let (ge1, ge2, row_words) = pack_dual_rows(&rows, 4);
        let min_rule = BeamRule {
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
        let (min_ge1, min_ge2) =
            materialize_rule_bits_dual(&min_rule, &ge1, &ge2, row_words, rows.len(), 4).unwrap();
        assert_eq!(unpack_dual_row(&min_ge1, &min_ge2, 4), vec![0u8, 1, 0, 1]);

        let max_rule = BeamRule {
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
        let (max_ge1, max_ge2) =
            materialize_rule_bits_dual(&max_rule, &ge1, &ge2, row_words, rows.len(), 4).unwrap();
        assert_eq!(unpack_dual_row(&max_ge1, &max_ge2, 4), vec![2u8, 1, 2, 2]);

        let not_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: true,
            },
            rest: Vec::new(),
        };
        let (not_ge1, not_ge2) =
            materialize_rule_bits_dual(&not_rule, &ge1, &ge2, row_words, rows.len(), 4).unwrap();
        assert_eq!(unpack_dual_row(&not_ge1, &not_ge2, 4), vec![2u8, 1, 0, 1]);
    }

    #[test]
    fn test_materialize_rule_bits_dual_xor_truth_table() {
        let rows = vec![vec![0u8, 0, 1, 1, 2, 2], vec![0u8, 2, 0, 1, 1, 2]];
        let (ge1, ge2, row_words) = pack_dual_rows(&rows, 6);
        let xor_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::Xor,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        };
        let (xor_ge1, xor_ge2) =
            materialize_rule_bits_dual(&xor_rule, &ge1, &ge2, row_words, rows.len(), 6).unwrap();
        assert_eq!(
            unpack_dual_row(&xor_ge1, &xor_ge2, 6),
            vec![0u8, 2, 1, 1, 1, 0]
        );
    }

    #[test]
    fn test_virtual_fuzzy_xor_matches_materialized_rule_score() {
        let rows = vec![vec![0u8, 0, 1, 1, 2, 2], vec![0u8, 2, 0, 1, 1, 2]];
        let y = vec![-2.0, 2.0, -0.5, 0.5, 1.5, -1.5];
        let (ge1, ge2, row_words) = pack_dual_rows(&rows, y.len());
        let n_rows = rows.len();
        let n = y.len();
        let sum_y = y.iter().copied().sum::<f64>();
        let params = BeamSearchParams {
            allow_parallel: false,
            ..BeamSearchParams::default()
        };
        let literal_scores = precompute_literal_singleton_scores_fuzzy(
            &y, n, &y, n, &ge1, &ge2, row_words, &ge1, &ge2, row_words, n_rows,
        )
        .unwrap();
        let literal_summaries =
            precompute_dual_literal_summaries(&y, &ge1, &ge2, row_words, n_rows, row_words, n);
        let parent_summary = literal_summaries[0];
        let (train_n_ge1, train_n_ge2, train_sum_ge1, train_sum_ge2) =
            literal_dual_summary_with_negation(sum_y, n, parent_summary, false);
        let parent_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: Vec::new(),
        };
        let parent = FuzzyBeamState {
            rule: parent_rule,
            combined_train_ge1: row_prefix(&ge1, row_words, 0, row_words).to_vec(),
            combined_train_ge2: row_prefix(&ge2, row_words, 0, row_words).to_vec(),
            train: literal_scores[literal_score_index(0, false)].train,
            train_n_ge2,
            train_sum_ge1,
            train_sum_ge2,
            train_abs_score: literal_scores[literal_score_index(0, false)]
                .train
                .raw_score,
            train_score: literal_scores[literal_score_index(0, false)]
                .train
                .raw_score,
            max_singleton_train_raw: literal_scores[literal_score_index(0, false)]
                .train
                .raw_score,
            max_singleton_test_raw: literal_scores[literal_score_index(0, false)].test.raw_score,
        };
        let row1_ge1 = row_prefix(&ge1, row_words, 1, row_words);
        let row1_ge2 = row_prefix(&ge2, row_words, 1, row_words);
        let (virtual_train, virtual_n_ge2, virtual_sum_ge1, virtual_sum_ge2) =
            evaluate_child_train_from_parent_virtual_fuzzy(
                parent.combined_train_ge1.as_slice(),
                parent.combined_train_ge2.as_slice(),
                &parent,
                row1_ge1,
                row1_ge2,
                literal_summaries[1],
                &y,
                sum_y,
                n,
                2,
                BeamBinaryOp::Xor,
                false,
                &params,
            )
            .unwrap();
        let xor_rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::Xor,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        };
        let materialized = evaluate_rule_continuous_dual_with_sum(
            &xor_rule, &y, sum_y, &ge1, &ge2, row_words, n_rows, n, 0.0, 0.0,
        )
        .unwrap();
        assert_eq!(virtual_train.n_hit, materialized.n_hit);
        assert_eq!(virtual_n_ge2, materialized.n_ge2);
        assert!(
            (virtual_sum_ge1 - (materialized.mean_hit * materialized.n_hit as f64)).abs() < 1e-12
        );
        let materialized_sum_ge2 = {
            let (_, ge2_bits) =
                materialize_rule_bits_dual(&xor_rule, &ge1, &ge2, row_words, n_rows, n).unwrap();
            sum_y_where_both1(&ge2_bits, &vec![u64::MAX; ge2_bits.len()], &y, n)
        };
        assert!((virtual_sum_ge2 - materialized_sum_ge2).abs() < 1e-12);
        assert!((virtual_train.raw_score - materialized.raw_score).abs() < 1e-12);
    }

    #[test]
    fn test_standard_fuzzy_parallel_expansion_matches_serial() {
        init_python_for_tests();
        let n_rows = 40usize;
        let n_samples = 16usize;
        let rows = (0..n_rows)
            .map(|row_idx| {
                (0..n_samples)
                    .map(|sample_idx| ((row_idx + sample_idx * 2) % 3) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let y = (0..n_samples)
            .map(|idx| idx as f64 - 7.5)
            .collect::<Vec<_>>();
        let (ge1, ge2, row_words) = pack_dual_rows(rows.as_slice(), n_samples);
        let literal_scores = precompute_literal_singleton_scores_fuzzy(
            y.as_slice(),
            n_samples,
            y.as_slice(),
            n_samples,
            ge1.as_slice(),
            ge2.as_slice(),
            row_words,
            ge1.as_slice(),
            ge2.as_slice(),
            row_words,
            n_rows,
        )
        .unwrap();
        let literal_summaries = precompute_dual_literal_summaries(
            y.as_slice(),
            ge1.as_slice(),
            ge2.as_slice(),
            row_words,
            n_rows,
            row_words,
            n_samples,
        );
        let group_ids = (0..n_rows).collect::<Vec<_>>();
        let mut parent_rules = (0..30usize)
            .map(|row_idx| BeamRule {
                first: BeamLiteral {
                    row_index: row_idx,
                    group_id: row_idx,
                    negated: false,
                },
                rest: Vec::new(),
            })
            .collect::<Vec<_>>();
        parent_rules.extend((0..10usize).map(|row_idx| BeamRule {
            first: BeamLiteral {
                row_index: row_idx,
                group_id: row_idx,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::And,
                BeamLiteral {
                    row_index: row_idx + 10,
                    group_id: row_idx + 10,
                    negated: row_idx % 2 == 1,
                },
            )],
        }));
        let parents = parent_rules
            .into_iter()
            .map(|rule| {
                fuzzy_beam_state_from_rule_for_test(
                    rule,
                    y.as_slice(),
                    ge1.as_slice(),
                    ge2.as_slice(),
                    row_words,
                    n_rows,
                    literal_scores.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        let params_serial = BeamSearchParams {
            rank_mode: BeamRankMode::Raw,
            beam_width: 32,
            allow_parallel: false,
            ..BeamSearchParams::default()
        };
        let serial = expand_fuzzy_beam_once(
            parents.as_slice(),
            y.as_slice(),
            y.iter().copied().sum::<f64>(),
            ge1.as_slice(),
            ge2.as_slice(),
            row_words,
            n_rows,
            row_words,
            n_samples,
            group_ids.as_slice(),
            literal_scores.as_slice(),
            literal_summaries.as_slice(),
            &params_serial,
        )
        .unwrap();
        let params_parallel = BeamSearchParams {
            allow_parallel: true,
            ..params_serial.clone()
        };
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                expand_fuzzy_beam_once_parallel(
                    parents.as_slice(),
                    y.as_slice(),
                    y.iter().copied().sum::<f64>(),
                    ge1.as_slice(),
                    ge2.as_slice(),
                    row_words,
                    n_rows,
                    row_words,
                    n_samples,
                    group_ids.as_slice(),
                    literal_scores.as_slice(),
                    literal_summaries.as_slice(),
                    &params_parallel,
                )
            })
            .unwrap();
        assert_same_fuzzy_beam_states(parallel.as_slice(), serial.as_slice());
    }

    #[test]
    fn test_standard_bin_parallel_expansion_matches_serial_after_deferred_materialization() {
        init_python_for_tests();
        let n_rows = 40usize;
        let n_samples = 16usize;
        let rows = (0..n_rows)
            .map(|row_idx| {
                (0..n_samples)
                    .map(|sample_idx| ((row_idx + sample_idx * 3) % 2) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let y = (0..n_samples)
            .map(|idx| idx as f64 - 7.5)
            .collect::<Vec<_>>();
        let (bits, row_words) = pack_rows(rows.as_slice(), n_samples);
        let literal_scores =
            literal_scores_for_test(y.as_slice(), bits.as_slice(), row_words, n_rows);
        let group_ids = (0..n_rows).collect::<Vec<_>>();
        let parents = (0..10usize)
            .map(|row_idx| {
                beam_state_from_rule_for_test(
                    BeamRule {
                        first: BeamLiteral {
                            row_index: row_idx,
                            group_id: row_idx,
                            negated: false,
                        },
                        rest: Vec::new(),
                    },
                    y.as_slice(),
                    bits.as_slice(),
                    row_words,
                    n_rows,
                    literal_scores.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        let params_serial = BeamSearchParams {
            rank_mode: BeamRankMode::Raw,
            beam_width: 32,
            allow_parallel: false,
            ..BeamSearchParams::default()
        };
        let serial = expand_beam_once(
            parents.as_slice(),
            y.as_slice(),
            y.iter().copied().sum::<f64>(),
            bits.as_slice(),
            row_words,
            n_rows,
            row_words,
            n_samples,
            group_ids.as_slice(),
            literal_scores.as_slice(),
            &params_serial,
        )
        .unwrap();
        let params_parallel = BeamSearchParams {
            allow_parallel: true,
            ..params_serial.clone()
        };
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                expand_beam_once_parallel_deferred(
                    parents.as_slice(),
                    y.as_slice(),
                    y.iter().copied().sum::<f64>(),
                    bits.as_slice(),
                    row_words,
                    n_rows,
                    row_words,
                    n_samples,
                    group_ids.as_slice(),
                    literal_scores.as_slice(),
                    &params_parallel,
                )
            })
            .unwrap();
        assert_same_beam_states(parallel.as_slice(), serial.as_slice());
    }

    #[test]
    fn test_binary_pair_intersection_lookup_matches_direct_sum() {
        let n_samples = 130usize;
        let y = (0..n_samples)
            .map(|idx| (idx as f64 * 0.25) - 13.0)
            .collect::<Vec<_>>();
        let row_words = words_for_samples(n_samples);
        let mut lhs = vec![u64::MAX; row_words];
        let mut rhs = vec![u64::MAX; row_words];
        let tail_mask = (1u64 << (n_samples & 63)) - 1;
        lhs[row_words - 1] = tail_mask;
        rhs[row_words - 1] = tail_mask;
        let lookup = PackedYSumLookup::build(y.as_slice(), n_samples).unwrap();
        let direct = binary_pair_intersection(&lhs, &rhs, y.as_slice(), n_samples);
        let (fused_n, fused_sum) =
            crate::garfield::score::and_popcount_sum_y_where_both1_with_lookup(
                &lhs,
                &rhs,
                y.as_slice(),
                n_samples,
                &lookup,
            );
        let cached = binary_pair_intersection_with_lookup(
            &lhs,
            &rhs,
            y.as_slice(),
            n_samples,
            Some(&lookup),
        );

        assert_eq!(direct.n, cached.n);
        assert_eq!(direct.sum.to_bits(), cached.sum.to_bits());
        assert_eq!(fused_n as usize, cached.n);
        assert_eq!(fused_sum.to_bits(), cached.sum.to_bits());
    }

    #[test]
    fn test_exhaustive_bin_parallel_expansion_matches_serial_after_deferred_materialization() {
        init_python_for_tests();
        let n_rows = 40usize;
        let n_samples = 16usize;
        let rows = (0..n_rows)
            .map(|row_idx| {
                (0..n_samples)
                    .map(|sample_idx| ((row_idx + sample_idx * 3) % 2) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let y = (0..n_samples)
            .map(|idx| idx as f64 - 7.5)
            .collect::<Vec<_>>();
        let (bits, row_words) = pack_rows(rows.as_slice(), n_samples);
        let literal_scores =
            literal_scores_for_test(y.as_slice(), bits.as_slice(), row_words, n_rows);
        let group_ids = (0..n_rows).collect::<Vec<_>>();
        let parents = (0..10usize)
            .map(|row_idx| {
                beam_state_from_rule_for_test(
                    BeamRule {
                        first: BeamLiteral {
                            row_index: row_idx,
                            group_id: row_idx,
                            negated: false,
                        },
                        rest: Vec::new(),
                    },
                    y.as_slice(),
                    bits.as_slice(),
                    row_words,
                    n_rows,
                    literal_scores.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        let params_serial = BeamSearchParams {
            rank_mode: BeamRankMode::Raw,
            beam_width: 32,
            allow_parallel: false,
            ..BeamSearchParams::default()
        };
        let serial = expand_states_exhaustive(
            parents.as_slice(),
            y.as_slice(),
            y.iter().copied().sum::<f64>(),
            bits.as_slice(),
            row_words,
            n_rows,
            row_words,
            n_samples,
            group_ids.as_slice(),
            literal_scores.as_slice(),
            &params_serial,
        )
        .unwrap();
        let params_parallel = BeamSearchParams {
            allow_parallel: true,
            ..params_serial.clone()
        };
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                expand_states_exhaustive_parallel_deferred(
                    parents.as_slice(),
                    y.as_slice(),
                    y.iter().copied().sum::<f64>(),
                    bits.as_slice(),
                    row_words,
                    n_rows,
                    row_words,
                    n_samples,
                    group_ids.as_slice(),
                    literal_scores.as_slice(),
                    &params_parallel,
                )
            })
            .unwrap();
        assert_same_beam_states(parallel.as_slice(), serial.as_slice());
    }

    #[test]
    fn test_beam_binary_ops_for_rule_enforces_single_xor_policy() {
        let singleton = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: Vec::new(),
        };
        assert_eq!(
            beam_binary_ops_for_rule(&singleton),
            &[BeamBinaryOp::And, BeamBinaryOp::Xor]
        );

        let neg_singleton = BeamRule {
            first: BeamLiteral {
                negated: true,
                ..singleton.first
            },
            rest: Vec::new(),
        };
        assert_eq!(
            beam_binary_ops_for_rule(&neg_singleton),
            &[BeamBinaryOp::And]
        );

        let xor_rule = BeamRule {
            first: singleton.first,
            rest: vec![(
                BeamBinaryOp::Xor,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        };
        assert_eq!(beam_binary_ops_for_rule(&xor_rule), &[BeamBinaryOp::And]);
    }

    #[test]
    fn test_beam_search_train_test_continuous_fuzzy_prefers_additive_singleton() {
        init_python_for_tests();
        let rows = vec![vec![0u8, 1, 2, 2, 1, 0]];
        let y = vec![0.0, 1.0, 2.0, 2.1, 1.0, 0.0];
        let (ge1, ge2, row_words) = pack_dual_rows(&rows, y.len());
        let hits = beam_search_train_test_continuous_fuzzy(
            &y,
            &ge1,
            &ge2,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &ge1,
            &ge2,
            row_words,
            y.len(),
            &[0usize],
            BeamSearchParams {
                max_pick: 1,
                beam_width: 4,
                allow_parallel: false,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].rule.first.row_index, 0);
        assert!(!hits[0].rule.first.negated);
        assert!(hits[0].train.raw_score > 0.0);
    }

    #[test]
    fn test_fuzzy_exhaustive_depth_keeps_negative_gain_pair_for_final_rerank() {
        init_python_for_tests();
        let rows = vec![
            vec![1u8, 1, 1, 1, 0, 0, 0, 0],
            vec![1u8, 1, 0, 0, 1, 1, 0, 0],
        ];
        let y = vec![3.0, 3.0, 3.0, 3.0, -1.0, -1.0, -1.0, -1.0];
        let (ge1, ge2, row_words) = pack_dual_rows(&rows, y.len());
        let out = beam_search_train_test_continuous_fuzzy(
            &y,
            &ge1,
            &ge2,
            row_words,
            rows.len(),
            y.len(),
            &y,
            &ge1,
            &ge2,
            row_words,
            y.len(),
            &[0usize, 1usize],
            BeamSearchParams {
                max_pick: 2,
                beam_width: 8,
                exhaustive_depth: 2,
                rank_mode: BeamRankMode::InteractionGain,
                allow_parallel: false,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(out.iter().any(|cand| {
            cand.rule.len() == 2
                && !cand.rule.first.negated
                && cand.rule.first.row_index == 0
                && cand.rule.rest.len() == 1
                && cand.rule.rest[0].0 == BeamBinaryOp::And
                && !cand.rule.rest[0].1.negated
                && cand.rule.rest[0].1.row_index == 1
                && cand.test_score <= 0.0
        }));
    }

    #[test]
    fn test_beam_search_recovers_or_signal_with_and_only_rule() {
        init_python_for_tests();
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
                && cand.rule.rest.len() == 1
                && cand.rule.rest[0].0 == BeamBinaryOp::And
                && {
                    let combined =
                        materialize_rule_bits(&cand.rule, &bits, row_words, rows.len(), y.len())
                            .unwrap();
                    let got = (0..y.len())
                        .map(|i| usize::from(((combined[i >> 6] >> (i & 63)) & 1u64) != 0))
                        .collect::<Vec<_>>();
                    got == vec![1, 1, 1, 1, 0, 0, 0, 0] || got == vec![0, 0, 0, 0, 1, 1, 1, 1]
                }
        }));
        assert!(out.iter().all(|cand| {
            cand.rule.rest.is_empty()
                || cand
                    .rule
                    .rest
                    .iter()
                    .all(|(op, _)| *op == BeamBinaryOp::And)
        }));
    }

    #[test]
    fn test_beam_search_respects_group_exclusion() {
        init_python_for_tests();
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
    fn test_geneset_stage_mode_releases_group_exclusion_after_full_coverage() {
        init_python_for_tests();
        let rows = vec![vec![1, 1, 0, 0], vec![1, 0, 1, 0], vec![1, 1, 1, 0]];
        let y = vec![2.0, 1.0, -1.0, -2.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let literal_scores = literal_scores_for_test(&y, &bits, row_words, rows.len());
        let group_ids = vec![0usize, 1usize, 0usize];
        let pair_rule = beam_state_from_rule_for_test(
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
        let expected = vec![
            (0usize, false, 0u8),
            (1usize, false, 1u8),
            (2usize, false, 1u8),
        ];

        let strict = expand_beam_once(
            &[pair_rule.clone()],
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
                rank_mode: BeamRankMode::Raw,
                beam_width: 16,
                allow_parallel: false,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!strict
            .iter()
            .any(|state| state.rule.lexical_key() == expected));

        let staged = expand_beam_once(
            &[pair_rule],
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
                rank_mode: BeamRankMode::Raw,
                beam_width: 16,
                allow_parallel: false,
                group_constraint: BeamGroupConstraintMode::ExcludeUntilDistinctGroups(2),
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(staged
            .iter()
            .any(|state| state.rule.lexical_key() == expected));
    }

    #[test]
    fn test_beam_search_finds_and_rule() {
        init_python_for_tests();
        let rows = vec![
            vec![1, 1, 1, 1, 0, 0, 0, 0],
            vec![1, 1, 0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 1, 1, 1, 1],
        ];
        let y = vec![4.0, 4.2, -1.0, -1.2, -1.1, -1.0, -1.2, -1.1];
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
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|cand| {
            cand.rule.len() == 2
                && cand.rule.first.row_index == 0
                && cand.rule.rest.len() == 1
                && cand.rule.rest[0].0 == BeamBinaryOp::And
                && cand.rule.rest[0].1.row_index == 1
        }));
    }

    #[test]
    fn test_search_null_filters_high_order_candidates_from_final_hits() {
        init_python_for_tests();
        let rows = vec![
            vec![1, 1, 1, 1, 0, 0, 0, 0],
            vec![1, 1, 0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 1, 1, 1, 1],
        ];
        let y = vec![4.0, 4.2, -1.0, -1.2, -1.1, -1.0, -1.2, -1.1];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![0usize, 1, 2];
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
        let pair_bucket = super::super::permutation::bucket_from_rule(&pair_rule, 0.25);
        let mut cal = super::super::permutation::RuleNullCalibrator::new();
        for _ in 0..10 {
            cal.insert(pair_bucket, 1000.0, 1000.0);
        }
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
                rank_mode: BeamRankMode::InteractionGain,
                null_penalties: Some(Arc::new(cal.finalize())),
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(out.iter().all(|cand| cand.rule.len() == 1));
        assert!(out.iter().any(|cand| cand.rule.first.row_index == 0));
    }

    #[test]
    fn test_beam_search_outputs_and_only_rules() {
        init_python_for_tests();
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
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|cand| {
            cand.rule.rest.is_empty()
                || cand
                    .rule
                    .rest
                    .iter()
                    .all(|(op, _)| *op == BeamBinaryOp::And)
        }));
    }

    #[test]
    fn test_exhaustive_pair_depth_recovers_weak_single_strong_pair() {
        init_python_for_tests();
        let rows = vec![
            vec![1, 1, 0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 1, 0, 1, 0],
            vec![0, 0, 1, 1, 0, 1, 0, 1],
        ];
        let y = vec![5.0, 5.0, 3.0, 3.0, -2.0, -2.0, -2.0, -2.0];
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
            dosage_maf: 0.25,
            n_hit: 2,
            n_ge2: 0,
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
    fn test_continuous_lmaf_pruning_uses_binary_minor_allele_frequency() {
        let params = BeamSearchParams {
            maf_threshold: 0.02,
            ..BeamSearchParams::default()
        };
        let borderline = ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.02,
            dosage_maf: 0.02,
            n_hit: 20,
            n_ge2: 0,
            n_miss: 980,
        };
        let passing = ContinuousRuleScore {
            support_frac: 0.98,
            dosage_maf: 0.02,
            n_hit: 980,
            n_miss: 20,
            ..borderline
        };
        assert!(keep_rule_after_dosage_maf_pruning(&borderline, &params));
        assert!(keep_rule_after_dosage_maf_pruning(&passing, &params));
    }

    #[test]
    fn test_continuous_lmaf_pruning_rejects_binary_minor_allele_below_threshold() {
        let params = BeamSearchParams {
            maf_threshold: 0.02,
            ..BeamSearchParams::default()
        };
        let borderline = ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.981,
            dosage_maf: 0.019,
            n_hit: 981,
            n_ge2: 0,
            n_miss: 19,
        };
        assert!(!keep_rule_after_dosage_maf_pruning(&borderline, &params));
    }

    #[test]
    fn test_continuous_lmaf_pruning_rejects_non_variable_binary_rule() {
        let params = BeamSearchParams {
            maf_threshold: 0.20,
            ..BeamSearchParams::default()
        };
        let singleton_minor = ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 1.0,
            dosage_maf: 0.0,
            n_hit: 10,
            n_ge2: 0,
            n_miss: 0,
        };
        assert!(!keep_rule_after_dosage_maf_pruning(
            &singleton_minor,
            &params
        ));
    }

    #[test]
    fn test_initial_singleton_seed_pruning_ignores_lmaf_but_requires_variation() {
        let params = BeamSearchParams {
            maf_threshold: 0.20,
            ..BeamSearchParams::default()
        };
        let rare_but_variable = ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: 1.0,
            mean_miss: 0.0,
            support_frac: 0.05,
            dosage_maf: 0.05,
            n_hit: 5,
            n_ge2: 0,
            n_miss: 95,
        };
        let non_variable = ContinuousRuleScore {
            support_frac: 1.0,
            dosage_maf: 0.0,
            n_hit: 100,
            n_ge2: 0,
            n_miss: 0,
            ..rare_but_variable
        };
        assert!(keep_initial_literal_after_seed_pruning(&rare_but_variable));
        assert!(!keep_rule_after_dosage_maf_pruning(
            &rare_but_variable,
            &params
        ));
        assert!(!keep_initial_literal_after_seed_pruning(&non_variable));
    }

    #[test]
    fn test_fuzzy_support_pruning_uses_dosage_maf_not_support_frac() {
        let params = BeamSearchParams {
            maf_threshold: 0.03,
            ..BeamSearchParams::default()
        };
        let borderline = ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: 0.0,
            mean_miss: 0.0,
            support_frac: 0.04,
            dosage_maf: 0.02,
            n_hit: 4,
            n_ge2: 0,
            n_miss: 96,
        };
        let passing = ContinuousRuleScore {
            dosage_maf: 0.03,
            n_ge2: 2,
            ..borderline
        };
        assert!(!keep_rule_after_dosage_maf_pruning(&borderline, &params));
        assert!(keep_rule_after_dosage_maf_pruning(&passing, &params));
    }

    #[test]
    fn test_interaction_gain_scoring_uses_ancestor_baseline_with_null_penalty() {
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
            dosage_maf: 0.25,
            n_hit: 2,
            n_ge2: 0,
            n_miss: 6,
        };
        let mut cal = super::super::permutation::RuleNullCalibrator::new();
        let bucket = super::super::permutation::bucket_from_rule(&rule, train.dosage_maf);
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
                null_penalties: Some(Arc::new(lookup)),
                ..BeamSearchParams::default()
            },
        );
        assert!((gain_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_ancestor_baseline_prefers_stronger_grandparent_over_direct_parent() {
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
        let literal_scores = vec![
            LiteralSingletonScore {
                train: ContinuousRuleScore {
                    score: 0.90,
                    raw_score: 0.90,
                    mean_hit: 0.0,
                    mean_miss: 0.0,
                    support_frac: 0.5,
                    dosage_maf: 0.5,
                    n_hit: 2,
                    n_ge2: 0,
                    n_miss: 2,
                },
                test: ContinuousRuleScore {
                    score: 0.90,
                    raw_score: 0.90,
                    mean_hit: 0.0,
                    mean_miss: 0.0,
                    support_frac: 0.5,
                    dosage_maf: 0.5,
                    n_hit: 2,
                    n_ge2: 0,
                    n_miss: 2,
                },
            },
            LiteralSingletonScore {
                train: ContinuousRuleScore {
                    score: 0.30,
                    raw_score: 0.30,
                    mean_hit: 0.0,
                    mean_miss: 0.0,
                    support_frac: 0.5,
                    dosage_maf: 0.5,
                    n_hit: 2,
                    n_ge2: 0,
                    n_miss: 2,
                },
                test: ContinuousRuleScore {
                    score: 0.30,
                    raw_score: 0.30,
                    mean_hit: 0.0,
                    mean_miss: 0.0,
                    support_frac: 0.5,
                    dosage_maf: 0.5,
                    n_hit: 2,
                    n_ge2: 0,
                    n_miss: 2,
                },
            },
            LiteralSingletonScore {
                train: ContinuousRuleScore {
                    score: 0.20,
                    raw_score: 0.20,
                    mean_hit: 0.0,
                    mean_miss: 0.0,
                    support_frac: 0.5,
                    dosage_maf: 0.5,
                    n_hit: 2,
                    n_ge2: 0,
                    n_miss: 2,
                },
                test: ContinuousRuleScore {
                    score: 0.20,
                    raw_score: 0.20,
                    mean_hit: 0.0,
                    mean_miss: 0.0,
                    support_frac: 0.5,
                    dosage_maf: 0.5,
                    n_hit: 2,
                    n_ge2: 0,
                    n_miss: 2,
                },
            },
            LiteralSingletonScore {
                train: ContinuousRuleScore {
                    score: f64::NEG_INFINITY,
                    raw_score: f64::NEG_INFINITY,
                    mean_hit: f64::NAN,
                    mean_miss: f64::NAN,
                    support_frac: f64::NAN,
                    dosage_maf: f64::NAN,
                    n_hit: 0,
                    n_ge2: 0,
                    n_miss: 0,
                },
                test: ContinuousRuleScore {
                    score: f64::NEG_INFINITY,
                    raw_score: f64::NEG_INFINITY,
                    mean_hit: f64::NAN,
                    mean_miss: f64::NAN,
                    support_frac: f64::NAN,
                    dosage_maf: f64::NAN,
                    n_hit: 0,
                    n_ge2: 0,
                    n_miss: 0,
                },
            },
            LiteralSingletonScore {
                train: ContinuousRuleScore {
                    score: f64::NEG_INFINITY,
                    raw_score: f64::NEG_INFINITY,
                    mean_hit: f64::NAN,
                    mean_miss: f64::NAN,
                    support_frac: f64::NAN,
                    dosage_maf: f64::NAN,
                    n_hit: 0,
                    n_ge2: 0,
                    n_miss: 0,
                },
                test: ContinuousRuleScore {
                    score: f64::NEG_INFINITY,
                    raw_score: f64::NEG_INFINITY,
                    mean_hit: f64::NAN,
                    mean_miss: f64::NAN,
                    support_frac: f64::NAN,
                    dosage_maf: f64::NAN,
                    n_hit: 0,
                    n_ge2: 0,
                    n_miss: 0,
                },
            },
            LiteralSingletonScore {
                train: ContinuousRuleScore {
                    score: f64::NEG_INFINITY,
                    raw_score: f64::NEG_INFINITY,
                    mean_hit: f64::NAN,
                    mean_miss: f64::NAN,
                    support_frac: f64::NAN,
                    dosage_maf: f64::NAN,
                    n_hit: 0,
                    n_ge2: 0,
                    n_miss: 0,
                },
                test: ContinuousRuleScore {
                    score: f64::NEG_INFINITY,
                    raw_score: f64::NEG_INFINITY,
                    mean_hit: f64::NAN,
                    mean_miss: f64::NAN,
                    support_frac: f64::NAN,
                    dosage_maf: f64::NAN,
                    n_hit: 0,
                    n_ge2: 0,
                    n_miss: 0,
                },
            },
        ];
        let mut base_cache = RuleRawScoreCache::new();
        cache_rule_raw_score(
            &mut base_cache,
            &BeamRule {
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
            0.55,
        );
        cache_rule_raw_score(
            &mut base_cache,
            &BeamRule {
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
            0.50,
        );
        cache_rule_raw_score(
            &mut base_cache,
            &BeamRule {
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
            },
            0.60,
        );
        let mut raw_cache = RuleRawScoreCache::new();
        let mut ancestor_cache = RuleAncestorBaselineCache::new();
        let best = best_ancestor_raw_baseline_cached(
            &rule,
            &[],
            &[],
            0,
            3,
            0,
            literal_scores.as_slice(),
            true,
            Some(&base_cache),
            &mut raw_cache,
            &mut ancestor_cache,
            false,
        )
        .unwrap();
        assert!((best - 0.90).abs() < 1e-12);
    }

    #[test]
    fn test_and_only_not_rules_use_same_incremental_baseline() {
        let params = BeamSearchParams {
            rank_mode: BeamRankMode::InteractionGain,
            null_penalties: Some(Arc::new(
                super::super::permutation::RuleNullPenaltyLookup::default(),
            )),
            ..BeamSearchParams::default()
        };
        let no_not = rank_rule_score_components_base(2, 0, 0.8, 0.6, &params);
        let with_not = rank_rule_score_components_base(2, 1, 0.8, 0.6, &params);
        assert!((no_not - 0.2).abs() < 1e-12);
        assert!((with_not - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_interaction_gain_scoring_uses_ancestor_baseline_for_triple() {
        let params = BeamSearchParams {
            rank_mode: BeamRankMode::InteractionGain,
            null_penalties: Some(Arc::new(
                super::super::permutation::RuleNullPenaltyLookup::default(),
            )),
            ..BeamSearchParams::default()
        };
        let triple_score = rank_rule_score_components_base(3, 0, 0.8, 0.6, &params);
        assert!((triple_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_pure_and_triple_with_null_penalty_uses_ancestor_baseline() {
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
            dosage_maf: 0.15,
            n_hit: 2,
            n_ge2: 0,
            n_miss: 10,
        };
        let mut cal = super::super::permutation::RuleNullCalibrator::new();
        let bucket = super::super::permutation::bucket_from_rule(&rule, train.dosage_maf);
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
                null_penalties: Some(Arc::new(lookup)),
                ..BeamSearchParams::default()
            },
        );
        assert!((gain_score - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_or_rule_uses_ancestor_baseline_under_gain_mode() {
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
            dosage_maf: 0.40,
            n_hit: 3,
            n_ge2: 0,
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
    fn test_gain_from_layer_one_uses_singleton_score_as_gain() {
        let params = BeamSearchParams {
            rank_mode: BeamRankMode::GainFromLayer(1),
            ..BeamSearchParams::default()
        };
        let single_score = rank_rule_score_components(1, 0, 0.8, 0.6, &params);
        let pair_score = rank_rule_score_components(2, 0, 0.8, 0.6, &params);
        assert!((single_score - 0.8).abs() < 1e-12);
        assert!((pair_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_gain_from_layer_one_does_not_filter_singleton_seeds() {
        let params = BeamSearchParams {
            min_gain: 1e-6,
            rank_mode: BeamRankMode::GainFromLayer(1),
            ..BeamSearchParams::default()
        };
        assert!(keep_state_after_min_gain_pruning(1, -10.0, &params));
        assert!(!keep_state_after_min_gain_pruning(2, 1e-7, &params));
        assert!(keep_state_after_min_gain_pruning(2, 1e-4, &params));
    }

    #[test]
    fn test_higher_order_and_gain_uses_ancestor_baseline() {
        let params = BeamSearchParams {
            rank_mode: BeamRankMode::InteractionGain,
            ..BeamSearchParams::default()
        };
        let triple_score = rank_rule_score_components_base(3, 0, 0.8, 0.6, &params);
        assert!((triple_score - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_layer1_seeds_both_pos_and_neg_singletons() {
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
        // After enabling !SNP in layer 1, both positive and negated singletons
        // can appear (they have the same centered_gain by complement symmetry).
        let has_pos = out.iter().any(|c| !c.rule.first.negated);
        let has_neg = out.iter().any(|c| c.rule.first.negated);
        assert!(has_pos || has_neg, "expected at least some singletons");
    }

    #[test]
    fn test_whole_genome_layer1_keeps_positive_singletons_only() {
        init_python_for_tests();
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
                whole_genome_dev_mode: true,
                rank_mode: BeamRankMode::Raw,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(out
            .iter()
            .all(|cand| cand.rule.rest.is_empty() && !cand.rule.first.negated));
    }

    #[test]
    fn test_whole_genome_layer2_can_pair_with_smaller_index() {
        init_python_for_tests();
        let rows = vec![vec![1, 1, 0, 0], vec![1, 0, 1, 0], vec![1, 1, 1, 0]];
        let y = vec![2.0, 1.0, -1.0, -2.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let literal_scores = literal_scores_for_test(&y, &bits, row_words, rows.len());
        let group_ids = vec![0usize, 1usize, 2usize];
        let parent = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 2,
                    group_id: 2,
                    negated: false,
                },
                rest: Vec::new(),
            },
            &y,
            &bits,
            row_words,
            rows.len(),
            literal_scores.as_slice(),
        );
        let next = expand_beam_once_whole_genome_layer2(
            &[parent],
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
                rank_mode: BeamRankMode::Raw,
                beam_width: 16,
                allow_parallel: false,
                whole_genome_dev_mode: true,
                ..BeamSearchParams::default()
            },
        )
        .unwrap();
        let expected = vec![(0usize, false, 0u8), (2usize, false, 1u8)];
        assert!(next
            .iter()
            .any(|state| state.rule.lexical_key() == expected));
    }

    #[test]
    fn test_whole_genome_layer2_parallel_matches_sequential() {
        init_python_for_tests();
        let rows = vec![
            vec![1, 1, 0, 0, 1, 0],
            vec![1, 0, 1, 0, 1, 0],
            vec![0, 1, 1, 0, 0, 1],
            vec![1, 1, 1, 0, 0, 0],
        ];
        let y = vec![3.0, 2.0, 1.0, -1.0, -2.0, -3.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let literal_scores = literal_scores_for_test(&y, &bits, row_words, rows.len());
        let group_ids = vec![0usize, 1usize, 2usize, 3usize];
        let parent = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 3,
                    group_id: 3,
                    negated: false,
                },
                rest: Vec::new(),
            },
            &y,
            &bits,
            row_words,
            rows.len(),
            literal_scores.as_slice(),
        );
        let params_seq = BeamSearchParams {
            rank_mode: BeamRankMode::Raw,
            beam_width: 16,
            allow_parallel: false,
            whole_genome_dev_mode: true,
            ..BeamSearchParams::default()
        };
        let seq = expand_beam_once_whole_genome_layer2(
            &[parent.clone()],
            &y,
            y.iter().copied().sum::<f64>(),
            &bits,
            row_words,
            rows.len(),
            words_for_samples(y.len()),
            y.len(),
            &group_ids,
            literal_scores.as_slice(),
            &params_seq,
        )
        .unwrap();
        let params_par = BeamSearchParams {
            allow_parallel: true,
            ..params_seq.clone()
        };
        let par = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                expand_beam_once_whole_genome_layer2(
                    &[parent],
                    &y,
                    y.iter().copied().sum::<f64>(),
                    &bits,
                    row_words,
                    rows.len(),
                    words_for_samples(y.len()),
                    y.len(),
                    &group_ids,
                    literal_scores.as_slice(),
                    &params_par,
                )
            })
            .unwrap();
        assert_same_beam_states(par.as_slice(), seq.as_slice());
    }

    #[test]
    fn test_whole_genome_layer3_target_parallel_matches_general_blind_scan() {
        init_python_for_tests();
        let rows = vec![
            vec![1, 1, 0, 0, 1, 0],
            vec![1, 0, 1, 0, 1, 0],
            vec![0, 1, 1, 0, 0, 1],
            vec![1, 1, 1, 0, 0, 0],
        ];
        let y = vec![3.0, 2.0, 1.0, -1.0, -2.0, -3.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let literal_scores = literal_scores_for_test(&y, &bits, row_words, rows.len());
        let group_ids = vec![0usize, 1usize, 2usize, 3usize];
        let parent = beam_state_from_rule_for_test(
            BeamRule {
                first: BeamLiteral {
                    row_index: 0,
                    group_id: 0,
                    negated: false,
                },
                rest: vec![(
                    BeamBinaryOp::And,
                    BeamLiteral {
                        row_index: 3,
                        group_id: 3,
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
        let params = BeamSearchParams {
            rank_mode: BeamRankMode::Raw,
            beam_width: 16,
            allow_parallel: false,
            whole_genome_dev_mode: true,
            ..BeamSearchParams::default()
        };
        let general = expand_beam_once(
            &[parent.clone()],
            &y,
            y.iter().copied().sum::<f64>(),
            &bits,
            row_words,
            rows.len(),
            words_for_samples(y.len()),
            y.len(),
            &group_ids,
            literal_scores.as_slice(),
            &params,
        )
        .unwrap();
        let wg = expand_beam_once_whole_genome_target_parallel(
            &[parent],
            &y,
            y.iter().copied().sum::<f64>(),
            &bits,
            row_words,
            rows.len(),
            words_for_samples(y.len()),
            y.len(),
            &group_ids,
            literal_scores.as_slice(),
            &params,
        )
        .unwrap();
        assert_same_beam_states(wg.as_slice(), general.as_slice());
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
    fn test_parent_abs_improvement_pruning_is_noop_with_zero_threshold() {
        let params = BeamSearchParams::default();
        assert_eq!(params.min_parent_abs_gain, 0.0);
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
    fn test_parent_abs_improvement_pruning_actually_prunes() {
        let mut params = BeamSearchParams::default();
        params.min_parent_abs_gain = 0.01;
        // Child barely below parent + threshold → pruned
        assert!(!keep_child_after_parent_abs_improvement_pruning(
            0.5, 2, 0.5099, &params
        ));
        // Child equals parent → pruned
        assert!(!keep_child_after_parent_abs_improvement_pruning(
            0.5, 2, 0.5, &params
        ));
        // Clear improvement → kept
        assert!(keep_child_after_parent_abs_improvement_pruning(
            0.5, 2, 0.52, &params
        ));
    }

    #[test]
    fn test_min_gain_pruning_uses_gain_minus_null_penalty_scale() {
        let mut params = BeamSearchParams {
            exhaustive_depth: 2,
            min_gain: 1e-6,
            ..BeamSearchParams::default()
        };
        assert!(keep_state_after_min_gain_pruning(1, -10.0, &params));
        assert!(!keep_state_after_min_gain_pruning(2, 1e-7, &params));
        assert!(!keep_state_after_min_gain_pruning(2, 1e-6, &params));
        assert!(keep_state_after_min_gain_pruning(2, 1e-4, &params));

        let pair_rule = BeamRule {
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
        assert!(!keep_child_after_parent_gain_pruning(
            &pair_rule, 1e-7, &params
        ));
        assert!(!keep_child_after_parent_gain_pruning(
            &pair_rule, 1e-6, &params
        ));
        assert!(keep_child_after_parent_gain_pruning(
            &pair_rule, 1e-4, &params
        ));

        params.min_gain = 0.0;
        assert!(!keep_state_after_min_gain_pruning(2, 0.0, &params));
        assert!(keep_state_after_min_gain_pruning(2, 1e-7, &params));
    }

    #[test]
    fn test_exhaustive_seed_still_allows_singleton_to_win() {
        init_python_for_tests();
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
    fn test_beam_search_keeps_negated_literals_inside_and_rules() {
        init_python_for_tests();
        let rows = vec![
            vec![1, 1, 1, 1, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 1, 1, 1, 1],
        ];
        let y = vec![-1.0, -1.0, -1.0, -1.0, 5.0, 5.0, -1.0, -1.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let group_ids = vec![0usize, 1usize, 2usize];
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
        assert!(out.iter().any(|cand| {
            cand.rule.len() == 2 && cand.rule.rest.iter().any(|(_, lit)| lit.negated)
                || (cand.rule.len() == 2 && cand.rule.first.negated)
        }));
    }

    #[test]
    fn test_best_singleton_is_retained_with_fixed_width_beam() {
        init_python_for_tests();
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

    /*
    #[test]
    #[ignore]
    fn test_surrogate_or_rule_collapses_back_to_singleton() {
        let rows = vec![vec![1, 1, 1, 1, 0, 0, 0, 0], vec![0, 0, 0, 0, 1, 0, 0, 0]];
        let y = vec![3.0, 3.1, 2.9, 3.2, 2.7, -3.0, -3.1, -2.9];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let params = BeamSearchParams {
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
        )
        .unwrap();
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
                BeamBinaryOp::And,
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
        )
        .unwrap();
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
                BeamBinaryOp::And,
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
    #[ignore]
    fn test_surrogate_or_not_proxy_collapses_to_positive_singleton() {
        let rows = vec![vec![0, 0, 0, 0, 1, 0, 0, 0], vec![1, 1, 1, 1, 0, 0, 0, 0]];
        let y = vec![-3.0, -3.1, -2.9, -3.2, 2.8, 2.9, 3.1, 3.0];
        let (bits, row_words) = pack_rows(&rows, y.len());
        let params = BeamSearchParams {
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
        )
        .unwrap();
        let child_rule = BeamRule {
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
    #[ignore]
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
        )
        .unwrap();
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
        let direct_parent_train_raw = best_ancestor_raw_baseline(
            &child_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            literal_scores.as_slice(),
            true,
            false,
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
    #[ignore]
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
        )
        .unwrap();
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
        let direct_parent_train_raw = best_ancestor_raw_baseline(
            &child_rule,
            &y,
            &bits,
            row_words,
            rows.len(),
            y.len(),
            literal_scores.as_slice(),
            true,
            false,
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
    */

    #[test]
    fn test_dual_pair_intersections_match_direct_intersections() {
        let y = vec![0.2, 1.1, -0.4, 0.8, 1.6, -0.3, 0.7, -1.2, 0.5, 2.0];
        let parent_ge1 = vec![0b0011_0110_01u64];
        let parent_ge2 = vec![0b0001_0010_01u64];
        let row_ge1 = vec![0b0110_1100_10u64];
        let row_ge2 = vec![0b0010_0100_00u64];
        let params = BeamSearchParams::default();

        let got = dual_pair_intersections_for_params(
            parent_ge1.as_slice(),
            parent_ge2.as_slice(),
            row_ge1.as_slice(),
            row_ge2.as_slice(),
            y.as_slice(),
            y.len(),
            &params,
        );

        assert_eq!(
            got.p1_r1_n,
            and_popcount(parent_ge1.as_slice(), row_ge1.as_slice()) as usize
        );
        assert_eq!(
            got.p2_r2_n,
            and_popcount(parent_ge2.as_slice(), row_ge2.as_slice()) as usize
        );
        assert_eq!(
            got.p1_r2_n,
            and_popcount(parent_ge1.as_slice(), row_ge2.as_slice()) as usize
        );
        assert_eq!(
            got.p2_r1_n,
            and_popcount(parent_ge2.as_slice(), row_ge1.as_slice()) as usize
        );
        assert_eq!(
            got.p1_r1_sum,
            sum_y_where_both1_for_params(
                parent_ge1.as_slice(),
                row_ge1.as_slice(),
                y.as_slice(),
                y.len(),
                &params,
            )
        );
        assert_eq!(
            got.p2_r2_sum,
            sum_y_where_both1_for_params(
                parent_ge2.as_slice(),
                row_ge2.as_slice(),
                y.as_slice(),
                y.len(),
                &params,
            )
        );
        assert_eq!(
            got.p1_r2_sum,
            sum_y_where_both1_for_params(
                parent_ge1.as_slice(),
                row_ge2.as_slice(),
                y.as_slice(),
                y.len(),
                &params,
            )
        );
        assert_eq!(
            got.p2_r1_sum,
            sum_y_where_both1_for_params(
                parent_ge2.as_slice(),
                row_ge1.as_slice(),
                y.as_slice(),
                y.len(),
                &params,
            )
        );
    }
}
