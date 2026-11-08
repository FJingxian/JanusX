mod all_pair;
mod api;
pub(crate) mod bs;
mod config;
mod engine;
mod four_cell;
mod grm_max6;
mod input;
mod logic_candidates;
mod logic_classifier;
mod logic_genotype;
mod logic_output;
mod logic_prepare;
mod logic_residual_screen;
mod logic_scan;
mod max6_cross_window;
mod max6_genotype;
mod max6_null;
mod max6_pipeline;
mod max6_qc;
mod max6_support;
mod max6_universe;
mod max6_window_stream;
mod min_cell;
mod output;
mod pair_triple;
mod pcg_residual;
mod permutation;
mod proposal;
mod residual;
mod sampled_screen;
mod sampling;
mod score;
mod simbench;
mod types;
mod universe;

pub(crate) use self::residual::estimate_lmm_lambda_from_grm_rust;
pub use api::*;
use config::*;
use engine::garfield_logic_search_bed_owned;
use input::*;
use logic_candidates::*;
use logic_genotype::*;
use logic_output::*;
use logic_prepare::*;
use logic_scan::*;
use max6_cross_window::*;
use max6_genotype::*;
use max6_null::*;
use max6_pipeline::run_max6_grmmax6_pipeline;
use max6_qc::*;
use max6_support::*;
use max6_universe::*;
use max6_window_stream::*;
use output::*;
use simbench::*;
use types::*;
use universe::*;

// GARFIELD runtime note (2026-08):
// - Active BIN search uses packed 0/1 homozygote bits only:
//   0 -> 0, 2 -> 1, and 1/NA are treated as missing then mode-imputed.
// - DOM/REC/HET feature expansion and the retired MBIN cache route are removed.
// - Active beam expansion is AND/XOR-only; negation remains supported.

use self::all_pair::{
    scan_all_pairs_continuous_packed_with_parallel_and_xor, AllPairCandidate, AllPairGate,
};
use self::bs::{beam_search_and_binary_mcc, beam_search_and_continuous_abs_corr, BeamAndResult};
use self::bs::{
    beam_search_train_test_continuous_fuzzy,
    beam_search_train_test_continuous_fuzzy_with_literal_scores,
    beam_search_train_test_continuous_with_literal_scores, evaluate_rule_continuous_dual,
    materialize_rule_bits_dual, precompute_literal_singleton_scores_batched,
    LiteralScoreBatchRequest, LiteralSingletonScore,
};
use self::bs::{reset_garfield_beam_profile, snapshot_garfield_beam_profile};
use self::pair_triple::{scan_pair_seeded_triples_continuous_packed, PairTripleSeed};
use self::pcg_residual::{garfield_residualize_pcg_from_bed, PcgResidualConfig};
use self::permutation::{
    bucket_from_rule_with_complexity, choose_representative_indices, gumbel_penalty_from_maxima,
    null_topk_per_repeat_for_bucket, rule_null_complexity_bin, rule_null_context_bin,
    rule_null_group_len_bucket_count, rule_null_group_len_bucket_index, rule_null_unit_group_bin,
    shuffled_copy_f64, RuleNullBucket, RuleNullCalibrator, RuleNullDistributionSummary,
    RuleNullPenaltyLookup, RuleNullPenaltyMethod, RuleStructurePrior, RuleStructurePriorCalibrator,
    RuleStructurePriorConfig, DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS,
    DEFAULT_RULE_NULL_ADAPTIVE_STABLE_REPEATS, DEFAULT_RULE_NULL_MAX_REPEATS,
    DEFAULT_RULE_NULL_MIN_SNPS_PER_CHUNK, DEFAULT_RULE_NULL_PHYSICAL_CHUNKS,
    DEFAULT_RULE_PERMUTATION_REPRESENTATIVE_UNITS, DEFAULT_RULE_STRUCTURE_BOOTSTRAP_KL_THRESHOLD,
    DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MAX_REPEATS, DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MIN_REPEATS,
    DEFAULT_RULE_STRUCTURE_BOOTSTRAP_STABLE_REPEATS, DEFAULT_RULE_STRUCTURE_DENSITY_TOPK,
};
use self::proposal::{
    build_proposal_site_pool, merge_and_select_site_pool, resolve_proposal_config, ChannelQuota,
    CorrChannel, ProposalChannelId, ProposalConfig, ProposalFamily, ProposalMode,
    SharedMaxTAccumulator,
};
use crate::bedmath::packed_byte_lut;
use crate::bincore::{
    append_suffix, bin01_header_bytes, bin01_id_sidecar_path, bin_prefix, build_windows_from_sites,
    discover_id_sidecar, discover_site_sidecar, parse_bin01_header, resolve_bin01_path, BinWindow,
};
use crate::binwriter::{Bin01SiteMode, Bin01SiteRecordRef, Bin01Writer};
use crate::bitwise::{
    and_popcount, and_popcount_bounded_blocks_with_backend, bitand_assign, bitnot_masked,
    resolve_reduce_backend, ReduceBackend,
};
use crate::blas::BlasThreadGuard;
use crate::breader::{
    gather_rows_by_indices, gather_rows_by_range, load_bin01_as_u64_words,
    load_bin01_packed_payload_owned, load_bin01_selected_rows_as_u64_words,
};
use crate::bstats::{apply_tail_mask, tail_mask, words_for_samples};
use crate::eigh::load_square_matrix_f64_from_file;
use crate::gblup::{build_grm_from_meta_stream, StreamKernelMode};
use crate::gfcore::{
    read_bim, read_fam, BedSnpIter, BimChunkReader, HmpSnpIter, SiteInfo, TxtSnpIter, VcfSnpIter,
};
use crate::gfreader::{
    build_sample_selection, compute_bed_row_meta_owned_for_source_rows,
    format_zero_sites_pure_line_error, is_simple_snp_allele,
    prepare_bed_logic_meta_owned_for_cached_compact_source_rows_pure_line,
    prepare_bed_logic_meta_owned_for_precomputed_site_keep,
    prepare_bed_logic_meta_owned_for_stats_samples_pure_line_intervals,
    prepare_bed_logic_meta_owned_for_stats_samples_pure_line_with_mmap_window,
    prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window, read_compact_max6_bim_rows,
    sample_indices_are_identity, CompactMax6SiteMetadata,
};
use crate::gload::load_file_owned;
use crate::ml::common::{
    parse_importance, parse_permutation_scoring, topk_indices, ImportanceKind, PermutationConfig,
    ResponseKind,
};
use crate::ml::engine::{compute_feature_scores_grouped, parse_ml_engine, MlEngine};
use crate::ml::extra_trees::ExtraTreesConfig;
use crate::ml::pairwise_and::{
    feature_scores_pairwise_and_packed_dual_with_stage1, reset_pairwise_profile,
    snapshot_pairwise_profile,
};
use crate::stats_common::{
    arm_interrupt_trap, check_ctrlc, env_truthy, format_bytes, interrupt_requested,
    map_err_string_to_py, process_memory_usage, INTERRUPTED_MSG,
};
#[cfg(unix)]
use memmap2::Advice;
use memmap2::{MmapMut, MmapOptions};

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::seq::index::sample as sample_indices_without_replacement;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use statrs::distribution::{Beta, ContinuousCDF};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, UNIX_EPOCH};

use self::bs::cmp_candidate;
use self::bs::{take_garfield_frontier_trace, BeamFrontierTraceRecord};
use self::logic_residual_screen::{OneCellMarginTable, ResidualProposalMode, ResidualScreenKernel};
pub use all_pair::{
    garfield_all_pair_scan_bin_py, garfield_all_pair_scan_bin_with_cells_py,
    garfield_all_pair_scan_bin_xor_py,
};
#[allow(unused_imports)]
pub use bs::{
    beam_search_train_test_continuous, evaluate_rule_continuous, materialize_rule_bits,
    rank_rule_score_components, rank_rule_score_components_with_bucket, BeamBinaryOp,
    BeamGroupConstraintMode, BeamLiteral, BeamRankMode, BeamRule, BeamRuleCandidate,
    BeamSearchParams, GarfieldSearchBackend,
};
#[cfg(test)]
use grm_max6::candidate_rerank_exact;
use grm_max6::{
    candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled,
    candidate_rerank_exact_packed_with_py_trusted_scaled,
    candidate_rerank_exact_with_py_trusted_scaled, project_phenotype_for_candidate_rerank,
    CandidateRerankResult, CandidateRerankTiming, DosageChannel, GrmIntScoreDiagnostics,
};
pub use grm_max6::{
    garfield_grm_max6_full_window_py, garfield_grm_max6_rerank_candidates_py,
    garfield_weighted_c_sample_row_sparse_py,
};
pub use logic_classifier::garfield_logic_pair_cell_summary_py;
pub use logic_residual_screen::{
    garfield_logic_residual_rerank_candidates_bin_py, garfield_logic_residual_screen_bin_py,
};
pub use pair_triple::garfield_pair_triple_refine_bin_py;
use residual::{
    build_design_matrix, build_exact_projection_from_grm_rust, build_no_grm_covariate_projector,
    NoGrmCovariateProjector,
};
use residual::{
    build_exact_projection_from_eigensystem_rust, garfield_residualize_exact_from_grm_rust,
    garfield_residualize_exact_with_eigenvectors_rust, garfield_residualize_lm_rust,
    GarfieldResidualResult,
};
pub use residual::{garfield_residualize_bed_py, garfield_residualize_grm_py};
#[allow(unused_imports)]
pub use sampling::stratified_test_mask;
#[allow(unused_imports)]
pub use score::{
    dual_packed_summary, dual_packed_summary_with_lookup,
    garfield_score_cont_centered_gain_batch_packed_cpu_py, score_binary_ba_mcc_batch_py,
    score_binary_ba_py, score_binary_mcc_packed, score_binary_mcc_py,
    score_cont_centered_gain_packed_with_sum, score_cont_corr_packed, score_cont_corr_py,
    score_cont_mean_diff_corr_batch_py, score_cont_mean_diff_py,
    score_cont_weighted_mean_diff_packed, support_size_packed, ContinuousRuleScore,
    PackedYSumLookup,
};
const GARFIELD_CONSTRAINED_BEAM_PAR_MIN_TOTAL_CANDS: usize = 1_024;
const GARFIELD_CONSTRAINED_BEAM_PAR_CHUNK_CANDS: usize = 256;
const GARFIELD_NULL_ML_TOP_FRAC: f64 = 0.80;
const GARFIELD_SCAN_MIN_GAIN_EPS: f64 = 1e-6;
const GARFIELD_SCAN_SURROGATE_TEST_GAIN_MAX: f64 = 0.02;
const GARFIELD_SCAN_SURROGATE_HAMMING_FRAC_MAX: f64 = 0.02;
// Active runtime keeps only bucket null penalties in user-visible ranking/output.
const GARFIELD_DISABLE_STRUCTURE_PRIOR: bool = true;
const GARFIELD_LD_CLUMP_R2_DEFAULT: f64 = 0.80;
/// Production Max6 defaults to identical encoded genotype vectors only.
/// Values below one explicitly request lossy marginal-LD compression.
const GARFIELD_MAX6_LD_TAG_R2_DEFAULT: f64 = 1.0;
const GARFIELD_GENESET_CORR_SERIAL_MAX_ROWS: usize = 256;
const GARFIELD_GENESET_CORR_PRESCREEN_SLACK_MIN: usize = 8;
const GARFIELD_GENESET_CORR_PRESCREEN_SLACK_MAX: usize = 32;
const GARFIELD_SINGLE_WINDOW_CANDIDATE_MULTIPLIER: usize = 4;
const GARFIELD_SINGLE_WINDOW_CANDIDATE_ADD: usize = 32;
const GARFIELD_LD_BLOCK_WORDS_DEFAULT: usize = 8;
/// Fixed seed budget for the experimental pair-to-triple backend.  This is
/// intentionally not a normal user parameter: changing it changes the
/// candidate space and therefore invalidates any matched null calibration.
const GARFIELD_PAIR_TRIPLE_K2: usize = 50;
// Poll the native interrupt flag in the LD candidate loop instead of
// reacquiring the Python GIL for every candidate. Set the interval to 1 to
// reproduce the previous full check_ctrlc behavior for A/B benchmarks.
const GARFIELD_LD_INTERRUPT_CHECK_INTERVAL_DEFAULT: usize = 64;
static GARFIELD_ML_SELECT_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_DENSE_DOSAGE_DECODE_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_LD_CLUMP_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_LD_CLUMP_EXACT_PAIRS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_LD_CLUMP_ROWS_TOTAL: AtomicU64 = AtomicU64::new(0);
static GARFIELD_LD_CLUMP_ROWS_KEPT: AtomicU64 = AtomicU64::new(0);
static GARFIELD_LD_CLUMP_ROWS_COMPRESSED: AtomicU64 = AtomicU64::new(0);
static GARFIELD_LD_CLUMP_UNITS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_MATERIALIZE_BITS_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_CORR_STAGE1_SUMMARY_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_CORR_STAGE1_SCORE_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_CORR_STAGE1_POOL_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_CORR_STAGE1_POOL_PRIORITY_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_CORR_STAGE1_POOL_LD_NS: AtomicU64 = AtomicU64::new(0);

const GARFIELD_PROPOSAL_ROW_CACHE_DEFAULT_ROWS: usize = 4_096;
const GARFIELD_PROPOSAL_ROW_CACHE_MAX_ROWS: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProposalRowCacheKey {
    source_ptr: usize,
    source_len: usize,
    source_hi_ptr: usize,
    source_hi_len: usize,
    row_idx: usize,
    sample_ptr: usize,
    sample_len: usize,
}

#[derive(Default)]
struct ProposalRowCache {
    rows: HashMap<ProposalRowCacheKey, Vec<u8>>,
    order: VecDeque<ProposalRowCacheKey>,
}

thread_local! {
    static GARFIELD_PROPOSAL_ROW_CACHE: RefCell<ProposalRowCache> =
        RefCell::new(ProposalRowCache::default());
}

#[inline]
fn garfield_proposal_row_cache_capacity() -> usize {
    parse_env_usize_allow_zero("JX_GARFIELD_PROPOSAL_ROW_CACHE_ROWS")
        .unwrap_or(GARFIELD_PROPOSAL_ROW_CACHE_DEFAULT_ROWS)
        .min(GARFIELD_PROPOSAL_ROW_CACHE_MAX_ROWS)
}
type GarfieldLdSupportCacheRegistry =
    Mutex<HashMap<(usize, u64), Arc<OnceLock<Arc<GarfieldLdSupportCacheEntry>>>>>;

static GARFIELD_LD_SUPPORT_CACHE: OnceLock<GarfieldLdSupportCacheRegistry> = OnceLock::new();
static GARFIELD_LD_UNIT_STATS: OnceLock<Mutex<GarfieldLdUnitStatsCollector>> = OnceLock::new();

const GARFIELD_STRUCTURE_TASK_COALESCE_MAX_UNITS_DEFAULT: usize = 32;
// Keep enough scan chunks to balance heterogeneous window sizes without
// returning to one Rayon task per window.
const GARFIELD_SCAN_TASK_COALESCE_MAX_UNITS_DEFAULT: usize = 512;
const GARFIELD_PERM_SLOT_TASK_MAX_SLOTS_DEFAULT: usize = 1;
const GARFIELD_PERM_TASK_COALESCE_MAX_TASKS_DEFAULT: usize = 64;
const GARFIELD_PERM_BITS_CACHE_MAX_MB_DEFAULT: usize = 256;
const GARFIELD_LOGIC_MMAP_WINDOW_MB_DEFAULT: usize = 128;
const GARFIELD_DENSE_DECODE_PAR_MIN_ROWS: usize = 64;
const GARFIELD_DENSE_DECODE_PAR_MIN_SAMPLES: usize = 256;

#[inline]
fn strict_maf_support_count(n_samples: usize, maf_threshold: f32) -> usize {
    if n_samples == 0 {
        return 0;
    }
    if !maf_threshold.is_finite() || maf_threshold <= 0.0 {
        return 1;
    }
    ((f64::from(maf_threshold) * (n_samples as f64)).floor() as usize).saturating_add(1)
}

#[inline]
fn timing_share_pct(part_s: f64, whole_s: f64) -> f64 {
    if part_s.is_finite() && whole_s.is_finite() && whole_s > 0.0 {
        (part_s / whole_s) * 100.0
    } else {
        0.0
    }
}

#[inline]
fn garfield_stage_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("JX_GARFIELD_STAGE_PROFILE"))
}

#[inline]
fn garfield_pair_scan_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("JX_GARFIELD_PAIR_SCAN_PROFILE"))
}

#[inline]
fn garfield_stage_profile_start() -> Option<Instant> {
    garfield_stage_profile_enabled().then(Instant::now)
}

#[inline]
fn garfield_stage_profile_end(start: Option<Instant>, counter: &AtomicU64) {
    if let Some(t0) = start {
        counter.fetch_add(
            t0.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct GarfieldSchedulerTaskSample {
    elapsed_s: f64,
    work_units: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct GarfieldSchedulerTaskSummary {
    task_count: usize,
    total_work_units: usize,
    total_elapsed_s: f64,
    min_elapsed_s: f64,
    p50_elapsed_s: f64,
    p95_elapsed_s: f64,
    max_elapsed_s: f64,
}

type GarfieldSchedulerTaskCollector = Arc<Mutex<Vec<GarfieldSchedulerTaskSample>>>;

#[inline]
fn garfield_scheduler_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("JX_GARFIELD_SCHEDULER_PROFILE"))
}

#[inline]
fn new_garfield_scheduler_task_collector() -> Option<GarfieldSchedulerTaskCollector> {
    garfield_scheduler_profile_enabled()
        .then(|| Arc::new(Mutex::new(Vec::<GarfieldSchedulerTaskSample>::new())))
}

#[inline]
fn record_garfield_scheduler_task(
    collector: Option<&GarfieldSchedulerTaskCollector>,
    elapsed_s: f64,
    work_units: usize,
) {
    let Some(collector) = collector else {
        return;
    };
    if !elapsed_s.is_finite() {
        return;
    }
    if let Ok(mut samples) = collector.lock() {
        samples.push(GarfieldSchedulerTaskSample {
            elapsed_s,
            work_units,
        });
    }
}

#[inline]
fn scheduler_percentile(sorted_values: &[f64], quantile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let quantile = quantile.clamp(0.0, 1.0);
    let rank = ((quantile * sorted_values.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted_values.len().saturating_sub(1));
    sorted_values[rank]
}

fn summarize_scheduler_task_samples(
    samples: &[GarfieldSchedulerTaskSample],
) -> GarfieldSchedulerTaskSummary {
    if samples.is_empty() {
        return GarfieldSchedulerTaskSummary::default();
    }
    let mut elapsed = samples
        .iter()
        .map(|sample| sample.elapsed_s)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if elapsed.is_empty() {
        return GarfieldSchedulerTaskSummary::default();
    }
    elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    GarfieldSchedulerTaskSummary {
        task_count: elapsed.len(),
        total_work_units: samples.iter().map(|sample| sample.work_units).sum(),
        total_elapsed_s: elapsed.iter().sum(),
        min_elapsed_s: elapsed[0],
        p50_elapsed_s: scheduler_percentile(elapsed.as_slice(), 0.50),
        p95_elapsed_s: scheduler_percentile(elapsed.as_slice(), 0.95),
        max_elapsed_s: *elapsed.last().unwrap_or(&0.0),
    }
}

fn summarize_garfield_scheduler_task_collector(
    collector: Option<&GarfieldSchedulerTaskCollector>,
) -> Option<GarfieldSchedulerTaskSummary> {
    let collector = collector?;
    let samples = collector.lock().ok()?;
    Some(summarize_scheduler_task_samples(samples.as_slice()))
}

#[derive(Default, Clone, Debug)]
struct GarfieldLdUnitStatsCollector {
    units_eligible: u64,
    units_pruned: u64,
    rows_pruned: u64,
    rows_total: Vec<u32>,
    rows_kept: Vec<u32>,
    rows_pruned_dist: Vec<u32>,
    exact_pairs: Vec<u32>,
}

#[derive(Default, Clone, Debug)]
struct GarfieldLdUnitStatsSnapshot {
    ld_units_eligible: u64,
    ld_units_pruned: u64,
    ld_rows_pruned: u64,
    ld_unit_rows_total_median: u64,
    ld_unit_rows_total_p95: u64,
    ld_unit_rows_total_max: u64,
    ld_unit_rows_kept_median: u64,
    ld_unit_rows_kept_p95: u64,
    ld_unit_rows_kept_max: u64,
    ld_unit_rows_pruned_median: u64,
    ld_unit_rows_pruned_p95: u64,
    ld_unit_rows_pruned_max: u64,
    ld_unit_exact_pairs_median: u64,
    ld_unit_exact_pairs_p95: u64,
    ld_unit_exact_pairs_max: u64,
}

const GARFIELD_LD_NO_BOUND: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
struct GarfieldLdExactBounds {
    low_max: u32,
    high_min: u32,
}

impl Default for GarfieldLdExactBounds {
    fn default() -> Self {
        Self {
            low_max: GARFIELD_LD_NO_BOUND,
            high_min: GARFIELD_LD_NO_BOUND,
        }
    }
}

impl GarfieldLdExactBounds {
    #[cfg(test)]
    #[inline]
    fn matches(self, both_one: usize) -> bool {
        let both_one = u32::try_from(both_one).unwrap_or(u32::MAX);
        (self.low_max != GARFIELD_LD_NO_BOUND && both_one <= self.low_max)
            || (self.high_min != GARFIELD_LD_NO_BOUND && both_one >= self.high_min)
    }
}

#[derive(Clone, Debug)]
struct GarfieldLdSupportCacheEntry {
    conflict_supports: Vec<Vec<usize>>,
    exact_bounds: Vec<Vec<GarfieldLdExactBounds>>,
}

fn summarize_u32_distribution(values: &[u32]) -> (u64, u64, u64) {
    if values.is_empty() {
        return (0, 0, 0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let idx_p50 = (n - 1) / 2;
    let idx_p95 = ((95usize.saturating_mul(n)).saturating_sub(1) / 100).min(n - 1);
    (
        u64::from(sorted[idx_p50]),
        u64::from(sorted[idx_p95]),
        u64::from(*sorted.last().unwrap_or(&0u32)),
    )
}

fn garfield_ld_unit_stats_reset() {
    let store =
        GARFIELD_LD_UNIT_STATS.get_or_init(|| Mutex::new(GarfieldLdUnitStatsCollector::default()));
    if let Ok(mut guard) = store.lock() {
        *guard = GarfieldLdUnitStatsCollector::default();
    }
}

fn garfield_ld_unit_stats_record(rows_total: usize, rows_kept: usize, exact_pairs: u64) {
    let rows_pruned = rows_total.saturating_sub(rows_kept);
    let store =
        GARFIELD_LD_UNIT_STATS.get_or_init(|| Mutex::new(GarfieldLdUnitStatsCollector::default()));
    if let Ok(mut guard) = store.lock() {
        guard.units_eligible = guard.units_eligible.saturating_add(1);
        if rows_pruned > 0 {
            guard.units_pruned = guard.units_pruned.saturating_add(1);
        }
        guard.rows_pruned = guard
            .rows_pruned
            .saturating_add(u64::try_from(rows_pruned).unwrap_or(u64::MAX));
        guard
            .rows_total
            .push(u32::try_from(rows_total).unwrap_or(u32::MAX));
        guard
            .rows_kept
            .push(u32::try_from(rows_kept).unwrap_or(u32::MAX));
        guard
            .rows_pruned_dist
            .push(u32::try_from(rows_pruned).unwrap_or(u32::MAX));
        guard
            .exact_pairs
            .push(u32::try_from(exact_pairs).unwrap_or(u32::MAX));
    }
}

fn garfield_ld_unit_stats_snapshot() -> GarfieldLdUnitStatsSnapshot {
    let store =
        GARFIELD_LD_UNIT_STATS.get_or_init(|| Mutex::new(GarfieldLdUnitStatsCollector::default()));
    let guard = match store.lock() {
        Ok(v) => v,
        Err(_) => return GarfieldLdUnitStatsSnapshot::default(),
    };
    let (rows_total_median, rows_total_p95, rows_total_max) =
        summarize_u32_distribution(guard.rows_total.as_slice());
    let (rows_kept_median, rows_kept_p95, rows_kept_max) =
        summarize_u32_distribution(guard.rows_kept.as_slice());
    let (rows_pruned_median, rows_pruned_p95, rows_pruned_max) =
        summarize_u32_distribution(guard.rows_pruned_dist.as_slice());
    let (exact_pairs_median, exact_pairs_p95, exact_pairs_max) =
        summarize_u32_distribution(guard.exact_pairs.as_slice());
    GarfieldLdUnitStatsSnapshot {
        ld_units_eligible: guard.units_eligible,
        ld_units_pruned: guard.units_pruned,
        ld_rows_pruned: guard.rows_pruned,
        ld_unit_rows_total_median: rows_total_median,
        ld_unit_rows_total_p95: rows_total_p95,
        ld_unit_rows_total_max: rows_total_max,
        ld_unit_rows_kept_median: rows_kept_median,
        ld_unit_rows_kept_p95: rows_kept_p95,
        ld_unit_rows_kept_max: rows_kept_max,
        ld_unit_rows_pruned_median: rows_pruned_median,
        ld_unit_rows_pruned_p95: rows_pruned_p95,
        ld_unit_rows_pruned_max: rows_pruned_max,
        ld_unit_exact_pairs_median: exact_pairs_median,
        ld_unit_exact_pairs_p95: exact_pairs_p95,
        ld_unit_exact_pairs_max: exact_pairs_max,
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
fn parse_env_mb_usize(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.ceil() as usize)
        .filter(|&v| v > 0)
}

#[inline]
fn parse_env_f64(name: &str) -> Option<f64> {
    env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
}

#[inline]
fn garfield_structure_task_coalesce_max_units() -> usize {
    parse_env_usize("JX_GARFIELD_STRUCTURE_TASK_COALESCE_MAX_UNITS")
        .unwrap_or(GARFIELD_STRUCTURE_TASK_COALESCE_MAX_UNITS_DEFAULT)
        .max(1)
}

#[inline]
fn garfield_scan_task_coalesce_max_units() -> usize {
    parse_env_usize("JX_GARFIELD_SCAN_TASK_COALESCE_MAX_UNITS")
        .unwrap_or(GARFIELD_SCAN_TASK_COALESCE_MAX_UNITS_DEFAULT)
        .max(1)
}

#[inline]
fn garfield_perm_slot_task_max_slots() -> usize {
    parse_env_usize_allow_zero("JX_GARFIELD_PERM_SLOT_TASK_MAX_SLOTS")
        .unwrap_or(GARFIELD_PERM_SLOT_TASK_MAX_SLOTS_DEFAULT)
}

#[inline]
fn garfield_perm_task_coalesce_max_tasks() -> usize {
    parse_env_usize("JX_GARFIELD_PERM_TASK_COALESCE_MAX_TASKS")
        .unwrap_or(GARFIELD_PERM_TASK_COALESCE_MAX_TASKS_DEFAULT)
        .max(1)
}

#[inline]
fn garfield_scheduler_optimization_enabled() -> bool {
    !env_truthy("JX_GARFIELD_DISABLE_SCHEDULER_OPT")
}

#[inline]
fn garfield_perm_bits_cache_max_bytes() -> usize {
    parse_env_mb_usize("JX_GARFIELD_PERM_BITS_CACHE_MAX_MB")
        .unwrap_or(GARFIELD_PERM_BITS_CACHE_MAX_MB_DEFAULT)
        .saturating_mul(1024 * 1024)
}

#[inline]
fn estimate_permutation_bits_cache_bytes(
    null_prepared: &[(GarfieldPermutationNullSource, GarfieldUnitPrepared)],
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
) -> usize {
    let train_words = if sample_indices_are_full_identity(train_idx_local, logic_bits.n_samples) {
        logic_bits.row_words
    } else {
        words_for_samples(train_idx_local.len())
    };
    let test_words = if train_idx_local == test_idx_local {
        0
    } else if sample_indices_are_full_identity(test_idx_local, logic_bits.n_samples) {
        logic_bits.row_words
    } else {
        words_for_samples(test_idx_local.len())
    };
    let planes = if logic_bits.bits_hi_flat.is_some() {
        2usize
    } else {
        1usize
    };
    null_prepared.iter().fold(0usize, |total, (_, prepared)| {
        let words = train_words.saturating_add(test_words);
        total.saturating_add(
            prepared
                .selected_global_rows
                .len()
                .saturating_mul(words)
                .saturating_mul(planes)
                .saturating_mul(std::mem::size_of::<u64>()),
        )
    })
}

#[inline]
fn garfield_task_chunk_units(total_units: usize, threads: usize, max_units: usize) -> usize {
    let total_units = total_units.max(1);
    let max_units = max_units.max(1);
    if threads <= 1 {
        return total_units.min(max_units);
    }
    let target_tasks = threads.saturating_mul(4).max(1);
    total_units.div_ceil(target_tasks).clamp(1, max_units)
}

/// Build contiguous scan chunks with approximately equal SNP work instead of
/// equal unit counts. Unit order is preserved so result ordering remains
/// deterministic, while heterogeneous windows no longer leave the last Rayon
/// workers with most of the expensive units.
fn garfield_cost_balanced_unit_chunks(
    unit_indices: &[usize],
    units: &[GarfieldLogicUnit],
    prepared_units: Option<&[Max6GrmMax6LdPreparedUnit]>,
    threads: usize,
    max_units: usize,
) -> Vec<Vec<usize>> {
    if unit_indices.is_empty() {
        return Vec::new();
    }
    if threads <= 1 {
        return vec![unit_indices.to_vec()];
    }
    let unit_work = |ui: usize| {
        prepared_units
            .and_then(|prepared| prepared.get(ui))
            .map(|unit| unit.selected_global_rows.len().max(1))
            .or_else(|| units.get(ui).map(|unit| unit.indices.len().max(1)))
            .unwrap_or(1)
    };
    let target_tasks = threads.saturating_mul(4).max(1).min(unit_indices.len());
    let total_work = unit_indices
        .iter()
        .fold(0usize, |total, &ui| total.saturating_add(unit_work(ui)));
    let target_work = total_work.div_ceil(target_tasks).max(1);
    let max_units = max_units.max(1);
    let mut chunks = Vec::<Vec<usize>>::with_capacity(target_tasks);
    let mut current = Vec::<usize>::new();
    let mut current_work = 0usize;

    for &ui in unit_indices.iter() {
        let work = unit_work(ui);
        let should_split = !current.is_empty()
            && (current.len() >= max_units
                || (current_work >= target_work && chunks.len() + 1 < target_tasks));
        if should_split {
            chunks.push(std::mem::take(&mut current));
            current_work = 0;
        }
        current.push(ui);
        current_work = current_work.saturating_add(work);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn garfield_scan_task_chunks(
    unit_indices: &[usize],
    units: &[GarfieldLogicUnit],
    prepared_units: Option<&[Max6GrmMax6LdPreparedUnit]>,
    threads: usize,
    max_units: usize,
) -> Vec<Vec<usize>> {
    if garfield_scheduler_optimization_enabled() {
        return garfield_cost_balanced_unit_chunks(
            unit_indices,
            units,
            prepared_units,
            threads,
            max_units,
        );
    }
    let chunk_size = garfield_task_chunk_units(unit_indices.len(), threads, max_units);
    unit_indices
        .chunks(chunk_size.max(1))
        .map(|chunk| chunk.to_vec())
        .collect()
}

#[inline]
fn garfield_logic_mmap_window_mb() -> usize {
    parse_env_mb_usize("JX_GARFIELD_LOGIC_MMAP_WINDOW_MB")
        .or_else(|| parse_env_mb_usize("JX_BED_BLOCK_TARGET_MB"))
        .unwrap_or(GARFIELD_LOGIC_MMAP_WINDOW_MB_DEFAULT)
        .max(1)
}

#[inline]
fn garfield_ld_clump_r2() -> f64 {
    parse_env_f64("JX_GARFIELD_LD_CLUMP_R2")
        .or_else(|| parse_env_f64("JX_GARFIELD_GENESET_LD_PRUNE_R2"))
        .filter(|v| *v > 0.0 && *v <= 1.0)
        .unwrap_or(GARFIELD_LD_CLUMP_R2_DEFAULT)
}

#[inline]
fn garfield_ld_block_words() -> usize {
    match parse_env_usize("JX_GARFIELD_LD_BLOCK_WORDS") {
        Some(8) => 8,
        _ => GARFIELD_LD_BLOCK_WORDS_DEFAULT,
    }
}

#[inline]
fn garfield_ld_interrupt_check_interval() -> usize {
    parse_env_usize("JX_GARFIELD_LD_INTERRUPT_CHECK_INTERVAL")
        .unwrap_or(GARFIELD_LD_INTERRUPT_CHECK_INTERVAL_DEFAULT)
        .max(1)
}

#[inline]
fn check_garfield_ld_interrupt(candidate_offset: usize, interval: usize) -> Result<(), String> {
    if interval == 1 {
        return check_ctrlc();
    }
    if candidate_offset % interval == 0 && interrupt_requested() {
        return Err(INTERRUPTED_MSG.to_string());
    }
    Ok(())
}

#[inline]
fn garfield_ld_support_cache_enabled() -> bool {
    !env_truthy("JX_GARFIELD_DISABLE_LD_SUPPORT_CACHE")
}

#[inline]
fn parse_env_usize_allow_zero(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
}

#[inline]
fn garfield_single_window_candidate_override() -> Option<(usize, usize)> {
    let mul = parse_env_usize_allow_zero("JX_GARFIELD_SINGLE_WINDOW_CANDIDATE_MULTIPLIER");
    let add = parse_env_usize_allow_zero("JX_GARFIELD_SINGLE_WINDOW_CANDIDATE_ADD");
    if mul.is_none() && add.is_none() {
        return None;
    }
    Some((mul.unwrap_or(1).max(1), add.unwrap_or(0)))
}

#[inline]
fn resolve_dynamic_single_window_candidate_pool_k(beam_k: usize, n_region: usize) -> usize {
    if beam_k == 0 || n_region <= beam_k {
        return beam_k.min(n_region);
    }
    let hard_cap = beam_k
        .saturating_mul(GARFIELD_SINGLE_WINDOW_CANDIDATE_MULTIPLIER)
        .max(beam_k.saturating_add(GARFIELD_SINGLE_WINDOW_CANDIDATE_ADD))
        .min(n_region)
        .max(beam_k);
    let ratio = (n_region as f64) / (beam_k as f64);
    let raw_extra = if ratio <= 2.0 {
        n_region.saturating_sub(beam_k)
    } else if ratio <= 4.0 {
        beam_k / 2
    } else if ratio <= 8.0 {
        beam_k.saturating_mul(2) / 3
    } else {
        beam_k.saturating_mul(3) / 4
    };
    let extra = raw_extra
        .max(16)
        .min(96)
        .min(n_region.saturating_sub(beam_k));
    beam_k.saturating_add(extra).min(hard_cap).max(beam_k)
}

#[inline]
fn garfield_rss_debug_enabled() -> bool {
    env_truthy("JX_GARFIELD_RSS_DEBUG")
}

#[inline]
fn garfield_layer_rss_debug_enabled() -> bool {
    env_truthy("JX_GARFIELD_LAYER_RSS_DEBUG")
}

#[inline]
fn garfield_prepare_rss_debug_enabled() -> bool {
    env_truthy("JX_GARFIELD_PREP_RSS_DEBUG")
}

#[inline]
fn garfield_prepare_rss_limit_bytes() -> Option<u64> {
    parse_env_f64("JX_GARFIELD_PREP_RSS_LIMIT_GB")
        .filter(|v| *v > 0.0)
        .map(|gb| (gb * 1024.0_f64 * 1024.0_f64 * 1024.0_f64) as u64)
}

#[inline]
fn garfield_layer_rss_limit_bytes() -> Option<u64> {
    parse_env_f64("JX_GARFIELD_LAYER_RSS_LIMIT_GB")
        .filter(|v| *v > 0.0)
        .map(|gb| (gb * 1024.0_f64 * 1024.0_f64 * 1024.0_f64) as u64)
}

#[inline]
fn garfield_whole_genome_unit_debug_enabled() -> bool {
    env_truthy("JX_GARFIELD_WG_UNIT_DEBUG") || garfield_layer_rss_debug_enabled()
}

#[inline]
fn garfield_whole_genome_unit_breakpoint(
    unit_name: &str,
    phase: &str,
    ml_feature_count: usize,
    elapsed_s: f64,
) -> Result<(), String> {
    let debug_enabled = garfield_whole_genome_unit_debug_enabled();
    let limit_bytes = garfield_layer_rss_limit_bytes();
    if !debug_enabled && limit_bytes.is_none() {
        return Ok(());
    }
    let Some(sample) = garfield_memory_sample_now() else {
        return Ok(());
    };
    let rss_txt = sample
        .rss_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "NA".to_string());
    let footprint_txt = sample
        .footprint_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "NA".to_string());
    if debug_enabled {
        eprintln!(
            "[GARFIELD-WG-UNIT] phase={phase} unit={unit_name} ml_features={ml_feature_count} elapsed={elapsed_s:.1}s metric={} current={} rss={} footprint={}",
            sample.metric,
            format_bytes(sample.current_bytes),
            rss_txt,
            footprint_txt,
        );
    }
    if let Some(limit) = limit_bytes {
        if sample.current_bytes > limit {
            return Err(format!(
                "GARFIELD whole-genome pre-beam memory limit exceeded at phase {phase}: {}={} (rss={}, footprint={}, limit={}, unit={}, ml_features={})",
                sample.metric,
                format_bytes(sample.current_bytes),
                rss_txt,
                footprint_txt,
                format_bytes(limit),
                unit_name,
                ml_feature_count,
            ));
        }
    }
    Ok(())
}

#[inline]
fn garfield_prepare_breakpoint(phase: &str, detail: Option<&str>) -> Result<(), String> {
    let debug_enabled = garfield_prepare_rss_debug_enabled();
    let limit_bytes = garfield_prepare_rss_limit_bytes();
    if !debug_enabled && limit_bytes.is_none() {
        return Ok(());
    }
    let Some(sample) = garfield_memory_sample_now() else {
        return Ok(());
    };
    let rss_now = sample.rss_bytes.unwrap_or(sample.current_bytes);
    let rss_txt = sample
        .rss_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "NA".to_string());
    let footprint_txt = sample
        .footprint_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "NA".to_string());
    let detail_txt = detail.unwrap_or("");
    if debug_enabled {
        if detail_txt.is_empty() {
            eprintln!(
                "[GARFIELD-PREP] phase={phase} metric={} current={} rss={} footprint={}",
                sample.metric,
                format_bytes(sample.current_bytes),
                rss_txt,
                footprint_txt,
            );
        } else {
            eprintln!(
                "[GARFIELD-PREP] phase={phase} detail={detail_txt} metric={} current={} rss={} footprint={}",
                sample.metric,
                format_bytes(sample.current_bytes),
                rss_txt,
                footprint_txt,
            );
        }
    }
    if let Some(limit) = limit_bytes {
        if rss_now > limit {
            return Err(format!(
                "GARFIELD prepare memory limit exceeded at phase {phase}: rss={} (metric {}={}, footprint={}, limit={}, detail={})",
                format_bytes(rss_now),
                sample.metric,
                format_bytes(sample.current_bytes),
                footprint_txt,
                format_bytes(limit),
                if detail_txt.is_empty() { "NA" } else { detail_txt },
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct GarfieldMemorySample {
    current_bytes: u64,
    rss_bytes: Option<u64>,
    footprint_bytes: Option<u64>,
    metric: &'static str,
}

#[inline]
fn garfield_memory_sample_now() -> Option<GarfieldMemorySample> {
    let usage = process_memory_usage()?;
    Some(GarfieldMemorySample {
        current_bytes: usage.current_bytes,
        rss_bytes: usage.rss_bytes,
        footprint_bytes: usage.footprint_bytes,
        metric: usage.metric,
    })
}

#[inline]
fn garfield_nonzero_u64(v: u64) -> Option<u64> {
    if v > 0 {
        Some(v)
    } else {
        None
    }
}

#[derive(Clone, Debug, Default)]
struct GarfieldStageMemoryDebug {
    metric: Option<String>,
    samples: usize,
    start_current_bytes: Option<u64>,
    start_rss_bytes: Option<u64>,
    start_footprint_bytes: Option<u64>,
    end_current_bytes: Option<u64>,
    end_rss_bytes: Option<u64>,
    end_footprint_bytes: Option<u64>,
    observed_peak_current_bytes: Option<u64>,
    observed_peak_rss_bytes: Option<u64>,
    observed_peak_footprint_bytes: Option<u64>,
}

#[derive(Debug, Default)]
struct GarfieldStageMemoryTracker {
    enabled: bool,
    peak_current_bytes: AtomicU64,
    peak_rss_bytes: AtomicU64,
    peak_footprint_bytes: AtomicU64,
    samples: AtomicUsize,
}

impl GarfieldStageMemoryTracker {
    #[inline]
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            peak_current_bytes: AtomicU64::new(0),
            peak_rss_bytes: AtomicU64::new(0),
            peak_footprint_bytes: AtomicU64::new(0),
            samples: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn observe_sample(&self, sample: GarfieldMemorySample) {
        if !self.enabled {
            return;
        }
        self.samples.fetch_add(1, Ordering::Relaxed);
        self.peak_current_bytes
            .fetch_max(sample.current_bytes, Ordering::Relaxed);
        if let Some(rss) = sample.rss_bytes {
            self.peak_rss_bytes.fetch_max(rss, Ordering::Relaxed);
        }
        if let Some(footprint) = sample.footprint_bytes {
            self.peak_footprint_bytes
                .fetch_max(footprint, Ordering::Relaxed);
        }
    }

    #[inline]
    fn sample_now(&self) {
        if !self.enabled {
            return;
        }
        if let Some(sample) = garfield_memory_sample_now() {
            self.observe_sample(sample);
        }
    }

    #[inline]
    fn start_stage(&self) -> Option<GarfieldMemorySample> {
        if !self.enabled {
            return None;
        }
        let sample = garfield_memory_sample_now()?;
        self.observe_sample(sample);
        Some(sample)
    }

    #[inline]
    fn finish_stage(&self, start: Option<GarfieldMemorySample>) -> GarfieldStageMemoryDebug {
        if !self.enabled {
            return GarfieldStageMemoryDebug::default();
        }
        let end = garfield_memory_sample_now();
        if let Some(sample) = end {
            self.observe_sample(sample);
        }
        let metric = end.or(start).map(|sample| sample.metric.to_string());
        GarfieldStageMemoryDebug {
            metric,
            samples: self.samples.load(Ordering::Relaxed),
            start_current_bytes: start.map(|sample| sample.current_bytes),
            start_rss_bytes: start.and_then(|sample| sample.rss_bytes),
            start_footprint_bytes: start.and_then(|sample| sample.footprint_bytes),
            end_current_bytes: end.map(|sample| sample.current_bytes),
            end_rss_bytes: end.and_then(|sample| sample.rss_bytes),
            end_footprint_bytes: end.and_then(|sample| sample.footprint_bytes),
            observed_peak_current_bytes: garfield_nonzero_u64(
                self.peak_current_bytes.load(Ordering::Relaxed),
            ),
            observed_peak_rss_bytes: garfield_nonzero_u64(
                self.peak_rss_bytes.load(Ordering::Relaxed),
            ),
            observed_peak_footprint_bytes: garfield_nonzero_u64(
                self.peak_footprint_bytes.load(Ordering::Relaxed),
            ),
        }
    }
}

#[derive(Debug)]
struct Max6GenotypePrepProfiler {
    enabled: bool,
    started: Instant,
}

impl Max6GenotypePrepProfiler {
    #[inline]
    fn from_env() -> Self {
        Self {
            enabled: env_truthy("JX_GARFIELD_MAX6_PREP_PROFILE"),
            started: Instant::now(),
        }
    }

    #[inline]
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn start_phase(&self) -> Max6GenotypePrepProfilePhase {
        let tracker = Arc::new(GarfieldStageMemoryTracker::new(self.enabled));
        let memory_start = tracker.start_stage();
        let sampler = if self.enabled {
            let stop = Arc::new(AtomicBool::new(false));
            let thread_stop = Arc::clone(&stop);
            let thread_tracker = Arc::clone(&tracker);
            let thread = std::thread::spawn(move || {
                while !thread_stop.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    if !thread_stop.load(Ordering::Relaxed) {
                        thread_tracker.sample_now();
                    }
                }
            });
            Some((stop, thread))
        } else {
            None
        };
        Max6GenotypePrepProfilePhase {
            started: Instant::now(),
            tracker,
            memory_start,
            sampler,
        }
    }

    fn finish_phase(&self, mut phase: Max6GenotypePrepProfilePhase, name: &str, detail: &str) {
        if !self.enabled {
            phase.stop_sampler();
            return;
        }
        let wall_secs = phase.started.elapsed().as_secs_f64();
        let total_secs = self.started.elapsed().as_secs_f64();
        phase.stop_sampler();
        let memory = phase.tracker.finish_stage(phase.memory_start);
        let rss_start = memory.start_rss_bytes.or(memory.start_current_bytes);
        let rss_end = memory.end_rss_bytes.or(memory.end_current_bytes);
        let rss_peak = memory
            .observed_peak_rss_bytes
            .or(memory.observed_peak_current_bytes);
        let footprint_start = memory.start_footprint_bytes;
        let footprint_end = memory.end_footprint_bytes;
        let footprint_peak = memory.observed_peak_footprint_bytes;
        let detail = if detail.is_empty() {
            format!("rss_samples={}", memory.samples)
        } else {
            format!("{detail} rss_samples={}", memory.samples)
        };
        eprintln!(
            "{}",
            format_max6_genotype_prep_profile_line(
                name,
                wall_secs,
                total_secs,
                rss_start,
                rss_end,
                rss_peak,
                footprint_start,
                footprint_end,
                footprint_peak,
                detail.as_str(),
            )
        );
    }
}

#[derive(Debug)]
struct Max6GenotypePrepProfilePhase {
    started: Instant,
    tracker: Arc<GarfieldStageMemoryTracker>,
    memory_start: Option<GarfieldMemorySample>,
    sampler: Option<(Arc<AtomicBool>, std::thread::JoinHandle<()>)>,
}

impl Max6GenotypePrepProfilePhase {
    #[inline]
    fn tracker(&self) -> &GarfieldStageMemoryTracker {
        self.tracker.as_ref()
    }

    #[inline]
    fn sample_now(&self) {
        self.tracker.sample_now();
    }

    fn stop_sampler(&mut self) {
        if let Some((stop, thread)) = self.sampler.take() {
            stop.store(true, Ordering::Relaxed);
            let _ = thread.join();
        }
    }
}

impl Drop for Max6GenotypePrepProfilePhase {
    fn drop(&mut self) {
        self.stop_sampler();
    }
}

#[inline]
fn format_max6_prep_memory_gib(bytes: Option<u64>) -> String {
    bytes
        .map(|value| format!("{:.2}", value as f64 / (1024.0 * 1024.0 * 1024.0)))
        .unwrap_or_else(|| "n/a".to_string())
}

#[allow(clippy::too_many_arguments)]
fn format_max6_genotype_prep_profile_line(
    phase: &str,
    wall_secs: f64,
    total_secs: f64,
    rss_start: Option<u64>,
    rss_end: Option<u64>,
    rss_peak: Option<u64>,
    footprint_start: Option<u64>,
    footprint_end: Option<u64>,
    footprint_peak: Option<u64>,
    detail: &str,
) -> String {
    format!(
        "[MAX6-PREP] phase={phase} wall={wall_secs:.3}s total={total_secs:.3}s rss={}->{}GiB (sampled_peak={}GiB) footprint={}->{}GiB (sampled_peak={}GiB) {detail}",
        format_max6_prep_memory_gib(rss_start),
        format_max6_prep_memory_gib(rss_end),
        format_max6_prep_memory_gib(rss_peak),
        format_max6_prep_memory_gib(footprint_start),
        format_max6_prep_memory_gib(footprint_end),
        format_max6_prep_memory_gib(footprint_peak),
    )
}

#[derive(Clone, Debug, Default)]
struct GarfieldMemoryDebugSummary {
    global_bits_loaded: GarfieldStageMemoryDebug,
    scan: GarfieldStageMemoryDebug,
    null_penalty: GarfieldStageMemoryDebug,
    structure_prior: GarfieldStageMemoryDebug,
}

#[derive(Clone)]
struct EncodedRow {
    site: SiteInfo,
    bits: Vec<u8>,
}

fn write_encoded_rows_bin(writer: &mut Bin01Writer, rows: &[EncodedRow]) -> Result<(), String> {
    for row in rows.iter() {
        writer.write_bitrow(
            row.bits.as_slice(),
            Some(Bin01SiteRecordRef::from(&row.site)),
        )?;
    }
    Ok(())
}

enum GarfieldInputReader {
    Bed(BedSnpIter),
    Vcf(VcfSnpIter),
    Hmp(HmpSnpIter),
    Txt(TxtSnpIter),
}

impl GarfieldInputReader {
    fn sample_ids(&self) -> &[String] {
        match self {
            GarfieldInputReader::Bed(it) => &it.samples,
            GarfieldInputReader::Vcf(it) => &it.samples,
            GarfieldInputReader::Hmp(it) => &it.samples,
            GarfieldInputReader::Txt(it) => &it.samples,
        }
    }
}

type GarfieldWindow = BinWindow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GarfieldNullChunk {
    chrom_code: u32,
    window_index: u32,
    window_id: u64,
    start_index: usize,
    end_index: usize,
    bp_start: i32,
    bp_end: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GarfieldNullChunkSpan {
    start_chunk_idx: usize,
    end_chunk_idx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldPermutationNullSource {
    Chunk(GarfieldNullChunk),
    SyntheticGeneset(usize),
}

#[derive(Clone, Copy, Debug)]
struct GarfieldNullReselectionConfig {
    row_mul: usize,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
}

impl GarfieldPermutationNullSource {
    #[inline]
    fn seed_id(self) -> u64 {
        match self {
            Self::Chunk(chunk) => chunk.window_id,
            Self::SyntheticGeneset(idx) => 0xA11C_E1D5_0000_0000u64.wrapping_add(idx as u64),
        }
    }
}

#[derive(Clone, Debug)]
struct ConstrainedBeamNode {
    selected: Vec<usize>,
    combined: Vec<u64>,
    score: f64,
    last_index: usize,
}

#[inline]
fn format_structure_alpha_meta(
    calibrator: &RuleStructurePriorCalibrator,
    cfg: &RuleStructurePriorConfig,
    display_len: usize,
) -> String {
    let alpha = calibrator.posterior_len_alpha_preview(cfg);
    let mut fields = Vec::<String>::with_capacity(display_len.clamp(1, 5));
    for rule_len in 1..=display_len.clamp(1, 5) {
        fields.push(format!("{:.2}", alpha[rule_len]));
    }
    format!("alpha=[{}]", fields.join(","))
}

#[inline]
fn format_structure_alpha_placeholder(display_len: usize) -> String {
    let fields = vec!["--"; display_len.clamp(1, 5)].join(",");
    format!("alpha=[{}]", fields)
}

#[inline]
fn garfield_progress_notify(
    progress_callback: Option<&Py<PyAny>>,
    done: usize,
    total: usize,
) -> PyResult<()> {
    if let Some(cb) = progress_callback {
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (done, total))?;
            Ok(())
        })?;
    }
    Ok(())
}

#[inline]
fn garfield_stage_progress_notify(
    progress_callback: Option<&Py<PyAny>>,
    stage: &str,
    done: usize,
    total: usize,
    meta: Option<&str>,
) -> PyResult<()> {
    if let Some(cb) = progress_callback {
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (stage, done, total, meta))?;
            Ok(())
        })?;
    } else {
        check_ctrlc().map_err(map_err_string_to_py)?;
    }
    Ok(())
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
fn cmp_constrained_nodes(a: &ConstrainedBeamNode, b: &ConstrainedBeamNode) -> std::cmp::Ordering {
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
fn constrained_node_better(a: &ConstrainedBeamNode, b: &ConstrainedBeamNode) -> bool {
    cmp_constrained_nodes(a, b) == std::cmp::Ordering::Less
}

#[inline]
fn push_top_k_constrained_streaming(
    nodes: &mut Vec<ConstrainedBeamNode>,
    cand: ConstrainedBeamNode,
    k: usize,
) {
    if k == 0 {
        return;
    }
    if nodes.len() < k {
        nodes.push(cand);
        return;
    }

    let mut worst_idx = 0usize;
    for i in 1..nodes.len() {
        if cmp_constrained_nodes(&nodes[i], &nodes[worst_idx]) == std::cmp::Ordering::Greater {
            worst_idx = i;
        }
    }

    if cmp_constrained_nodes(&cand, &nodes[worst_idx]) == std::cmp::Ordering::Less {
        nodes[worst_idx] = cand;
    }
}

#[inline]
fn garfield_should_parallel_constrained(total_cands: usize) -> bool {
    rayon::current_num_threads() > 1 && total_cands >= GARFIELD_CONSTRAINED_BEAM_PAR_MIN_TOTAL_CANDS
}

#[inline]
fn sort_truncate_nodes(mut nodes: Vec<ConstrainedBeamNode>, k: usize) -> Vec<ConstrainedBeamNode> {
    nodes.sort_by(cmp_constrained_nodes);
    if nodes.len() > k {
        nodes.truncate(k);
    }
    nodes
}

#[inline]
fn validate_constrained_inputs(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    group_ids: &[usize],
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
    if max_pick == 0 {
        return Err(format!("{ctx}: max_pick must be > 0"));
    }
    if beam_width == 0 {
        return Err(format!("{ctx}: beam_width must be > 0"));
    }
    if group_ids.len() != n_rows {
        return Err(format!(
            "{ctx}: group_ids length mismatch: {} vs n_rows={}",
            group_ids.len(),
            n_rows
        ));
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

fn beam_search_and_with_group_exclusion<F>(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    group_ids: &[usize],
    score_fn: F,
) -> Result<BeamAndResult, String>
where
    F: Fn(&[u64]) -> f64 + Sync,
{
    let ctx = "garfield::beam_search_and_with_group_exclusion";
    let needed_words = validate_constrained_inputs(
        bits_flat, row_words, n_rows, n_samples, max_pick, beam_width, group_ids, ctx,
    )?;

    let mask = tail_mask(n_samples);
    let max_depth = max_pick.min(n_rows);
    let layer_cap = beam_width.min(n_rows);

    let mut beam: Vec<ConstrainedBeamNode> = if garfield_should_parallel_constrained(n_rows) {
        let mut work = Vec::<(usize, usize)>::new();
        let chunk = GARFIELD_CONSTRAINED_BEAM_PAR_CHUNK_CANDS.max(1);
        let mut start = 0usize;
        while start < n_rows {
            let end = (start + chunk).min(n_rows);
            work.push((start, end));
            start = end;
        }
        let local_tops: Vec<Vec<ConstrainedBeamNode>> = work
            .into_par_iter()
            .map(|(start, end)| {
                let mut local: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
                for i in start..end {
                    let mut combined = row_prefix(bits_flat, row_words, i, needed_words).to_vec();
                    apply_tail_mask(&mut combined, mask);
                    let score = score_fn(&combined);
                    push_top_k_constrained_streaming(
                        &mut local,
                        ConstrainedBeamNode {
                            selected: vec![i],
                            combined,
                            score,
                            last_index: i,
                        },
                        layer_cap,
                    );
                }
                local
            })
            .collect();
        let mut merged: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
        for local in local_tops {
            for cand in local {
                push_top_k_constrained_streaming(&mut merged, cand, layer_cap);
            }
        }
        merged
    } else {
        let mut seq: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
        for i in 0..n_rows {
            let mut combined = row_prefix(bits_flat, row_words, i, needed_words).to_vec();
            apply_tail_mask(&mut combined, mask);
            let score = score_fn(&combined);
            push_top_k_constrained_streaming(
                &mut seq,
                ConstrainedBeamNode {
                    selected: vec![i],
                    combined,
                    score,
                    last_index: i,
                },
                layer_cap,
            );
        }
        seq
    };
    beam = sort_truncate_nodes(beam, layer_cap);
    if beam.is_empty() {
        return Err(format!("{ctx}: no candidates"));
    }

    let mut best = beam[0].clone();
    for _depth in 2..=max_depth {
        let total_expand = beam
            .iter()
            .map(|node| n_rows.saturating_sub(node.last_index.saturating_add(1)))
            .sum::<usize>();
        let next: Vec<ConstrainedBeamNode> = if garfield_should_parallel_constrained(total_expand) {
            let mut work = Vec::<(usize, usize, usize)>::new();
            let chunk = GARFIELD_CONSTRAINED_BEAM_PAR_CHUNK_CANDS.max(1);
            for (bi, node) in beam.iter().enumerate() {
                let mut start = node.last_index + 1;
                while start < n_rows {
                    let end = (start + chunk).min(n_rows);
                    work.push((bi, start, end));
                    start = end;
                }
            }
            let local_tops: Vec<Vec<ConstrainedBeamNode>> = work
                .into_par_iter()
                .map(|(bi, start, end)| {
                    let node = &beam[bi];
                    let mut local: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
                    for cand in start..end {
                        let cg = group_ids[cand];
                        if node.selected.iter().any(|&s| group_ids[s] == cg) {
                            continue;
                        }
                        let row = row_prefix(bits_flat, row_words, cand, needed_words);
                        let mut combined = node.combined.clone();
                        bitand_assign(&mut combined, row);
                        apply_tail_mask(&mut combined, mask);
                        let score = score_fn(&combined);
                        let mut selected = node.selected.clone();
                        selected.push(cand);
                        push_top_k_constrained_streaming(
                            &mut local,
                            ConstrainedBeamNode {
                                selected,
                                combined,
                                score,
                                last_index: cand,
                            },
                            layer_cap,
                        );
                    }
                    local
                })
                .collect();
            let mut merged: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
            for local in local_tops {
                for cand in local {
                    push_top_k_constrained_streaming(&mut merged, cand, layer_cap);
                }
            }
            merged
        } else {
            let mut seq: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
            for node in beam.iter() {
                for cand in (node.last_index + 1)..n_rows {
                    let cg = group_ids[cand];
                    if node.selected.iter().any(|&s| group_ids[s] == cg) {
                        continue;
                    }
                    let row = row_prefix(bits_flat, row_words, cand, needed_words);
                    let mut combined = node.combined.clone();
                    bitand_assign(&mut combined, row);
                    apply_tail_mask(&mut combined, mask);
                    let score = score_fn(&combined);
                    let mut selected = node.selected.clone();
                    selected.push(cand);
                    push_top_k_constrained_streaming(
                        &mut seq,
                        ConstrainedBeamNode {
                            selected,
                            combined,
                            score,
                            last_index: cand,
                        },
                        layer_cap,
                    );
                }
            }
            seq
        };
        if next.is_empty() {
            break;
        }
        beam = sort_truncate_nodes(next, layer_cap);
        if constrained_node_better(&beam[0], &best) {
            best = beam[0].clone();
        }
    }

    Ok(BeamAndResult {
        selected_indices: best.selected,
        score: best.score,
        combined_bits: best.combined,
    })
}

fn run_beam_with_feature_exclusion(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    response: GarfieldResponse,
    y_train_f64: &[f64],
    y_train_bin: Option<&[u8]>,
    max_pick: usize,
    beam_width: usize,
    local_group_ids: Option<&[usize]>,
) -> Result<BeamAndResult, String> {
    if let Some(groups) = local_group_ids {
        return match response {
            GarfieldResponse::Binary => {
                let yb = y_train_bin.ok_or_else(|| "binary y is not prepared".to_string())?;
                beam_search_and_with_group_exclusion(
                    bits_flat,
                    row_words,
                    n_rows,
                    n_samples,
                    max_pick,
                    beam_width,
                    groups,
                    |combined| score_binary_mcc_packed(yb, combined, n_samples),
                )
            }
            GarfieldResponse::Continuous => beam_search_and_with_group_exclusion(
                bits_flat,
                row_words,
                n_rows,
                n_samples,
                max_pick,
                beam_width,
                groups,
                |combined| score_cont_corr_packed(y_train_f64, combined, n_samples).abs(),
            ),
        };
    }

    match response {
        GarfieldResponse::Binary => {
            let yb = y_train_bin.ok_or_else(|| "binary y is not prepared".to_string())?;
            beam_search_and_binary_mcc(
                yb, bits_flat, row_words, n_rows, n_samples, max_pick, beam_width,
            )
        }
        GarfieldResponse::Continuous => beam_search_and_continuous_abs_corr(
            y_train_f64,
            bits_flat,
            row_words,
            n_rows,
            n_samples,
            max_pick,
            beam_width,
        ),
    }
}

fn read_y_f64(arr: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    match crate::stats_common::array_standard_slice(&arr) {
        Ok(s) => s.to_vec(),
        Err(_) => crate::stats_common::copy_array_standard_layout(&arr.as_array()),
    }
}

fn validate_binary_y(y: &[f64]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(y.len());
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("binary y contains non-finite value at index {}", i));
        }
        if v == 0.0 {
            out.push(0u8);
        } else if v == 1.0 {
            out.push(1u8);
        } else {
            return Err(format!(
                "binary response requires y in {{0,1}}; got y[{}]={}",
                i, v
            ));
        }
    }
    Ok(out)
}

fn validate_continuous_y(y: &[f64]) -> Result<(), String> {
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "continuous y contains non-finite value at index {}",
                i
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum GarfieldRuleSupportBits {
    Binary(Box<[u64]>),
    Dual { ge1: Box<[u64]>, ge2: Box<[u64]> },
}

#[derive(Clone, Debug)]
struct GarfieldLogicRuleRecord {
    unit_name: String,
    unit_kind: String,
    unit_index: usize,
    region_size: usize,
    ml_feature_count: usize,
    ml_rank: String,
    selected_row_indices: Vec<usize>,
    display_ops: Vec<BeamBinaryOp>,
    display_negated: Vec<bool>,
    snp_name: String,
    expr: String,
    chrom_field: String,
    bim_snp_name: String,
    bim_allele0: String,
    bim_allele1: String,
    allele1_maf: f64,
    pos: i32,
    score: f64,
    delta_score: String,
    perm_pvalue: Option<f64>,
    perm_fdr: Option<f64>,
    support_bits: Option<GarfieldRuleSupportBits>,
}

#[derive(Clone, Copy, Debug)]
struct GarfieldPermutationNullScores {
    bucket: RuleNullBucket,
    search_train_score: f64,
    search_test_score: f64,
    output_train_raw_score: f64,
    output_test_raw_score: f64,
    delta_train_score: f64,
    delta_test_score: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldRuleDisplayPolarity {
    Original,
    #[allow(dead_code)]
    Complement,
}

#[derive(Clone, Debug)]
struct GarfieldRenderedRule {
    polarity: GarfieldRuleDisplayPolarity,
    display_ops: Vec<BeamBinaryOp>,
    display_negated: Vec<bool>,
    snp_name: String,
    expr: String,
    chrom_field: String,
    bim_snp_name: String,
    bim_allele0: String,
    bim_allele1: String,
    pos: i32,
}

#[derive(Clone, Debug, Default)]
struct GarfieldUnitEvaluationOutput {
    records: Vec<GarfieldLogicRuleRecord>,
    diagnostics: Vec<GarfieldRecallDiagnosticRecord>,
    frontier_diagnostics: Vec<GarfieldFrontierDiagnosticRecord>,
}

#[derive(Clone, Debug)]
struct GarfieldRecallDiagnosticRecord {
    unit_name: String,
    unit_index: usize,
    region_size: usize,
    stage: &'static str,
    rank: usize,
    n_rows: usize,
    global_rows: Vec<usize>,
    snp_name: String,
    expr: String,
    raw_score: Option<f64>,
    output_score: Option<f64>,
    best_parent_raw: Option<f64>,
    best_parent_delta: Option<f64>,
    formal_eligible: bool,
    reported: bool,
}

#[derive(Clone, Debug)]
struct GarfieldFrontierDiagnosticRecord {
    unit_name: String,
    unit_index: usize,
    region_size: usize,
    depth: usize,
    frontier_rank: usize,
    parent_rule: String,
    child_rule: String,
    raw_score: f64,
    search_score: f64,
    beam_cutoff_score: f64,
    frontier_rank_before: usize,
    frontier_rank_after: usize,
    retained: bool,
    support: bool,
    support_current_parent: bool,
    support_any_parent: bool,
    n_feasible_parents: usize,
    rescued_by_alt_parent: bool,
    best_feasible_parent: String,
    best_parent_delta: f64,
    global_rows: Vec<usize>,
}

fn diagnostic_site_names(logic_bits: &GarfieldLogicBits, rows: &[usize]) -> String {
    rows.iter()
        .filter_map(|&idx| logic_bits.sites.get(idx))
        .map(GarfieldLogicSite::snp)
        .collect::<Vec<_>>()
        .join(";")
}

fn diagnostic_global_row_ids(rows: &[usize]) -> String {
    rows.iter()
        .map(|idx| idx.to_string())
        .collect::<Vec<_>>()
        .join(";")
}

fn append_ld_compression_diagnostics(
    out: &mut Vec<GarfieldRecallDiagnosticRecord>,
    unit: &GarfieldLogicUnit,
    unit_index: usize,
    logic_bits: &GarfieldLogicBits,
    records: &[GarfieldLdCompressionRecord],
) {
    for (rank, record) in records.iter().enumerate() {
        let rows = [record.kept_global_row, record.dropped_global_row];
        out.push(GarfieldRecallDiagnosticRecord {
            unit_name: unit.label.clone(),
            unit_index,
            region_size: unit.indices.len(),
            stage: "ld_clump",
            rank: rank + 1,
            n_rows: rows.len(),
            global_rows: rows.to_vec(),
            snp_name: diagnostic_site_names(logic_bits, &rows),
            expr: format!("r2={:.8}", record.r2),
            raw_score: None,
            output_score: None,
            best_parent_raw: None,
            best_parent_delta: None,
            formal_eligible: false,
            reported: false,
        });
    }
}

#[derive(Clone, Debug)]
struct GarfieldBeamDebugProbe {
    unit_name: String,
    out_tsv: String,
}

#[derive(Clone, Debug, Default)]
struct GarfieldLogicPipelineResult {
    pseudo_prefix: Option<String>,
    rules_tsv: Option<String>,
    diagnostics_tsv: Option<String>,
    frontier_diagnostics_tsv: Option<String>,
    // Kept as a nullable compatibility field for callers that deserialize the
    // historical result shape; the old per-unit comparison TSV is no longer
    // computed or written.
    rules_compare_tsv: Option<String>,
    posterior_json: Option<String>,
    memory_debug: Option<GarfieldMemoryDebugSummary>,
    bg_noise_summary: Option<GarfieldBgNoiseSummary>,
    scheduler_scan: Option<GarfieldSchedulerTaskSummary>,
    scheduler_permutation: Option<GarfieldSchedulerTaskSummary>,
    rule_permutation_active: bool,
    null_chunk_bp: usize,
    null_chunk_min_snps: usize,
    null_chunk_target: usize,
    null_chunk_valid_total: usize,
    null_chunk_selected: usize,
    representative_units_target: usize,
    representative_units_used: usize,
    permutation_null_repeats: usize,
    permutation_bootstrap_repeats: usize,
    records: Vec<GarfieldLogicRuleRecord>,
    simbench_count: usize,
    n_samples: usize,
    logic_bits_backend: String,
    logic_bits_bytes: usize,
    units_total: usize,
    units_scanned: usize,
    timing_total_wall_s: f64,
    timing_scan_wall_s: f64,
    max6_observed_scan_timing: Max6ObservedScanTiming,
    timing_max6_screen_calibration_wall_s: f64,
    timing_max6_screen_validation_wall_s: f64,
    timing_scan_beam_wall_s: f64,
    timing_scan_literal_score_wall_s: f64,
    timing_scan_ml_select_wall_s: f64,
    timing_corr_stage1_summary_s: f64,
    timing_corr_stage1_score_s: f64,
    timing_corr_stage1_pool_s: f64,
    timing_corr_stage1_pool_priority_s: f64,
    timing_corr_stage1_pool_ld_s: f64,
    timing_ld_support_cache_s: f64,
    ld_support_cache_rows: usize,
    ld_support_cache_payload_bytes: usize,
    timing_scan_beam_calls: usize,
    timing_clone_bits_s: f64,
    timing_sum_y_both1_s: f64,
    timing_parent_baseline_s: f64,
    timing_pw_marginal_s: f64,
    timing_pw_pack_s: f64,
    timing_pw_kernel_s: f64,
    timing_pw_combine_s: f64,
    timing_dense_extract_s: f64,
    timing_dense_decode_s: f64,
    timing_geneset_ld_prune_s: f64,
    ld_exact_pairs: u64,
    ld_rows_total: u64,
    ld_rows_kept: u64,
    ld_clump_rows_compressed: u64,
    ld_clump_units: u64,
    ld_units_eligible: u64,
    ld_units_pruned: u64,
    ld_rows_pruned: u64,
    ld_unit_rows_total_median: u64,
    ld_unit_rows_total_p95: u64,
    ld_unit_rows_total_max: u64,
    ld_unit_rows_kept_median: u64,
    ld_unit_rows_kept_p95: u64,
    ld_unit_rows_kept_max: u64,
    ld_unit_rows_pruned_median: u64,
    ld_unit_rows_pruned_p95: u64,
    ld_unit_rows_pruned_max: u64,
    ld_unit_exact_pairs_median: u64,
    ld_unit_exact_pairs_p95: u64,
    ld_unit_exact_pairs_max: u64,
    timing_materialize_bits_s: f64,
    timing_literal_score_share_of_total_pct: f64,
    timing_literal_score_share_of_scan_pct: f64,
    timing_literal_score_share_of_beam_pct: f64,
    timing_beam_share_of_total_pct: f64,
    timing_beam_share_of_scan_pct: f64,
    skipped_units: Vec<GarfieldSkippedUnitInfo>,
    skipped_messages: Vec<String>,
    train_fit: Arc<GarfieldResidualResult>,
    max6_pairs_tsv: Option<String>,
    max6_pairs_provenance_tsv: Option<String>,
    max6_ld_tags_tsv: Option<String>,
    max6_threshold_tsv: Option<String>,
    max6_screen_maxima_tsv: Option<String>,
    max6_screen_validation_maxima_tsv: Option<String>,
    max6_screen_sampled_maxima_tsv: Option<String>,
    max6_screen_sample_manifest_json: Option<String>,
    max6_screen_sample_strata_tsv: Option<String>,
    max6_screen_full_maximizer_geometry_tsv: Option<String>,
    timing_max6_screen_sampled_diagnostic_wall_s: f64,
    max6_screen_tail_top_pairs_tsv: Option<String>,
    max6_screen_tail_sampled_ranks_tsv: Option<String>,
    max6_screen_tail_summary_json: Option<String>,
    max6_screen_tail_support_aware_maxima_tsv: Option<String>,
    max6_screen_tail_support_aware_strata_tsv: Option<String>,
    max6_screen_tail_support_aware_summary_json: Option<String>,
    max6_screen_exceedance_audit_per_null_tsv: Option<String>,
    max6_screen_exceedance_audit_summary_tsv: Option<String>,
    max6_screen_exceedance_audit_top_pairs_tsv: Option<String>,
    timing_max6_screen_tail_audit_wall_s: f64,
    max6_screen_exceedance_audit_manifest_json: Option<String>,
    timing_max6_screen_exceedance_audit_wall_s: f64,
    max6_screen_pot_audit_manifest_json: Option<String>,
    max6_screen_pot_sample_scores_manifest_json: Option<String>,
    timing_max6_screen_pot_audit_wall_s: f64,
    max6_permutation_maxima_tsv: Option<String>,
    max6_pair_pve_total_quadratic: Option<f64>,
    max6_pair_universe_size: usize,
    max6_bonferroni_q_cut: Option<f64>,
    max6_bonferroni_p_cut: Option<f64>,
    max6_score_threshold: Option<f64>,
    max6_gumbel_score_threshold: Option<f64>,
    max6_screen_threshold: Option<f64>,
    max6_screen_threshold_method: Option<String>,
    max6_screen_null_repeats: usize,
    max6_screen_validation_repeats: usize,
    max6_screen_pair_universe_size: usize,
    max6_screen_window_pair_occurrences: usize,
    max6_screen_candidate_count: usize,
    max6_screen_validation_rejections: usize,
    max6_screen_validation_fwer: Option<f64>,
    max6_screen_validation_ci_lower: Option<f64>,
    max6_screen_validation_ci_upper: Option<f64>,
    max6_final_screen_subset_audit_passed: Option<bool>,
    max6_final_screen_subset_audit_pairs_checked: usize,
    max6_final_screen_subset_audit_violations: usize,
    max6_score_scale: Option<f64>,
    max6_permutation_score_count: usize,
    max6_permutation_pair_pool_size: usize,
    max6_permutation_pairs_per_repeat: usize,
    max6_permutation_pair_draw_count: usize,
    max6_permutation_timing: Max6PermutationTiming,
    max6_pair_count: usize,
}

#[derive(Clone, Debug)]
struct GarfieldBgNoiseSummary {
    dataset: String,
    buckets: Vec<GarfieldBgNoiseBucketSummary>,
    // Diagnostic-only thresholds copied from the exact lookup used by the
    // final output gate.  Keeping these in the manifest makes oracle replay
    // possible without changing the search or re-running null calibration.
    output_delta_penalties_train: Vec<Option<f64>>,
    output_delta_penalties_test: Vec<Option<f64>>,
    output_family_4plus_train: Option<f64>,
    output_family_4plus_test: Option<f64>,
    output_delta_family_4plus_train: Option<f64>,
    output_delta_family_4plus_test: Option<f64>,
}

#[derive(Clone, Debug)]
struct GarfieldBgNoiseBucketSummary {
    label: String,
    search: Option<RuleNullDistributionSummary>,
    output: Option<RuleNullDistributionSummary>,
}

fn garfield_stage_memory_debug_to_pydict<'py>(
    py: Python<'py>,
    stage: &GarfieldStageMemoryDebug,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("metric", stage.metric.clone())?;
    out.set_item("samples", stage.samples)?;
    out.set_item("start_current_bytes", stage.start_current_bytes)?;
    out.set_item("start_rss_bytes", stage.start_rss_bytes)?;
    out.set_item("start_footprint_bytes", stage.start_footprint_bytes)?;
    out.set_item("end_current_bytes", stage.end_current_bytes)?;
    out.set_item("end_rss_bytes", stage.end_rss_bytes)?;
    out.set_item("end_footprint_bytes", stage.end_footprint_bytes)?;
    out.set_item(
        "observed_peak_current_bytes",
        stage.observed_peak_current_bytes,
    )?;
    out.set_item("observed_peak_rss_bytes", stage.observed_peak_rss_bytes)?;
    out.set_item(
        "observed_peak_footprint_bytes",
        stage.observed_peak_footprint_bytes,
    )?;
    Ok(out)
}

fn garfield_skipped_unit_to_pydict<'py>(
    py: Python<'py>,
    skipped: &GarfieldSkippedUnitInfo,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("phase", skipped.phase.clone())?;
    out.set_item("unit_name", skipped.unit_name.clone())?;
    out.set_item("max_singleton_dosage_maf", skipped.max_singleton_dosage_maf)?;
    Ok(out)
}

fn garfield_memory_debug_to_pydict<'py>(
    py: Python<'py>,
    debug: &GarfieldMemoryDebugSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "global_bits_loaded",
        garfield_stage_memory_debug_to_pydict(py, &debug.global_bits_loaded)?,
    )?;
    out.set_item(
        "scan",
        garfield_stage_memory_debug_to_pydict(py, &debug.scan)?,
    )?;
    out.set_item(
        "null_penalty",
        garfield_stage_memory_debug_to_pydict(py, &debug.null_penalty)?,
    )?;
    out.set_item(
        "structure_prior",
        garfield_stage_memory_debug_to_pydict(py, &debug.structure_prior)?,
    )?;
    Ok(out)
}

fn garfield_scheduler_task_summary_to_pydict<'py>(
    py: Python<'py>,
    summary: &GarfieldSchedulerTaskSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("task_count", summary.task_count)?;
    out.set_item("total_work_units", summary.total_work_units)?;
    out.set_item("total_elapsed_s", summary.total_elapsed_s)?;
    out.set_item("min_elapsed_s", summary.min_elapsed_s)?;
    out.set_item("p50_elapsed_s", summary.p50_elapsed_s)?;
    out.set_item("p95_elapsed_s", summary.p95_elapsed_s)?;
    out.set_item("max_elapsed_s", summary.max_elapsed_s)?;
    Ok(out)
}

fn rule_null_distribution_summary_to_pydict<'py>(
    py: Python<'py>,
    summary: &RuleNullDistributionSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("method", summary.method)?;
    out.set_item("quantile", summary.quantile)?;
    out.set_item("penalty", summary.penalty)?;
    out.set_item("mean", summary.mean)?;
    out.set_item("variance", summary.variance)?;
    out.set_item("n", summary.n)?;
    out.set_item("min", summary.min)?;
    out.set_item("q25", summary.q25)?;
    out.set_item("median", summary.median)?;
    out.set_item("q75", summary.q75)?;
    out.set_item("max", summary.max)?;
    Ok(out)
}

fn garfield_bg_noise_summary_to_pydict<'py>(
    py: Python<'py>,
    summary: &GarfieldBgNoiseSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("dataset", summary.dataset.clone())?;
    let buckets = PyList::empty(py);
    for bucket in summary.buckets.iter() {
        let bucket_out = PyDict::new(py);
        bucket_out.set_item("label", bucket.label.clone())?;
        if let Some(search) = bucket.search.as_ref() {
            bucket_out.set_item(
                "search",
                rule_null_distribution_summary_to_pydict(py, search)?,
            )?;
        } else {
            bucket_out.set_item("search", py.None())?;
        }
        if let Some(output) = bucket.output.as_ref() {
            bucket_out.set_item(
                "output",
                rule_null_distribution_summary_to_pydict(py, output)?,
            )?;
        } else {
            bucket_out.set_item("output", py.None())?;
        }
        buckets.append(bucket_out)?;
    }
    out.set_item("buckets", buckets)?;
    out.set_item(
        "output_delta_penalties_train",
        summary.output_delta_penalties_train.clone(),
    )?;
    out.set_item(
        "output_delta_penalties_test",
        summary.output_delta_penalties_test.clone(),
    )?;
    out.set_item(
        "output_family_4plus_train",
        summary.output_family_4plus_train,
    )?;
    out.set_item("output_family_4plus_test", summary.output_family_4plus_test)?;
    out.set_item(
        "output_delta_family_4plus_train",
        summary.output_delta_family_4plus_train,
    )?;
    out.set_item(
        "output_delta_family_4plus_test",
        summary.output_delta_family_4plus_test,
    )?;
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GarfieldScanBimRange {
    chrom: String,
    bp_start: i32,
    bp_end: i32,
}

#[inline]
fn is_no_valid_initial_literals_error(err: &str) -> bool {
    err.contains("no valid initial literals")
}

#[inline]
fn extract_skip_named_f64(err: &str, key: &str) -> Option<f64> {
    let needle = format!("{key}=");
    let start = err.find(&needle)?.saturating_add(needle.len());
    let tail = err.get(start..)?;
    let end = tail
        .find(|c: char| c == ',' || c == ')' || c.is_whitespace())
        .unwrap_or(tail.len());
    tail.get(..end)?
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
}

#[derive(Clone, Debug)]
struct GarfieldSkippedUnitInfo {
    phase: String,
    unit_name: String,
    max_singleton_dosage_maf: Option<f64>,
}

#[inline]
fn make_skipped_unit_info(phase: &str, unit_name: &str, reason: &str) -> GarfieldSkippedUnitInfo {
    GarfieldSkippedUnitInfo {
        phase: phase.to_string(),
        unit_name: unit_name.to_string(),
        max_singleton_dosage_maf: extract_skip_named_f64(reason, "max_singleton_dosage_maf"),
    }
}

#[inline]
fn format_skipped_unit_message(info: &GarfieldSkippedUnitInfo) -> String {
    if let Some(max_lmaf) = info.max_singleton_dosage_maf {
        format!(
            "GARFIELD skipped unit [{}] {}: max_singleton_dosage_maf={:.4}",
            info.phase, info.unit_name, max_lmaf
        )
    } else {
        format!("GARFIELD skipped unit [{}] {}", info.phase, info.unit_name)
    }
}

#[inline]
fn cmp_logic_rule_records(
    a: &GarfieldLogicRuleRecord,
    b: &GarfieldLogicRuleRecord,
) -> std::cmp::Ordering {
    b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            a.selected_row_indices
                .len()
                .cmp(&b.selected_row_indices.len())
        })
        .then_with(|| a.snp_name.cmp(&b.snp_name))
        .then_with(|| a.unit_name.cmp(&b.unit_name))
}

#[inline]
fn logic_rule_not_count(rec: &GarfieldLogicRuleRecord) -> usize {
    rec.display_negated.iter().filter(|&&neg| neg).count()
}

#[inline]
fn logic_rule_display_family_rank(rec: &GarfieldLogicRuleRecord) -> u8 {
    if rec.display_ops.is_empty() || rec.display_ops.iter().all(|op| *op == BeamBinaryOp::And) {
        0
    } else if rec.display_ops.iter().all(|op| *op == BeamBinaryOp::Or) {
        2
    } else {
        1
    }
}

#[inline]
fn cmp_logic_rule_records_same_support(
    a: &GarfieldLogicRuleRecord,
    b: &GarfieldLogicRuleRecord,
) -> std::cmp::Ordering {
    logic_rule_display_family_rank(a)
        .cmp(&logic_rule_display_family_rank(b))
        .then_with(|| {
            a.selected_row_indices
                .len()
                .cmp(&b.selected_row_indices.len())
        })
        .then_with(|| logic_rule_not_count(a).cmp(&logic_rule_not_count(b)))
        .then_with(|| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| a.snp_name.cmp(&b.snp_name))
        .then_with(|| a.unit_name.cmp(&b.unit_name))
}

#[inline]
fn logic_rule_output_kind_rank(unit_kind: &str) -> u8 {
    match unit_kind {
        "window" => 0,
        "wholegenome" => 1,
        "gene" => 2,
        "geneset" => 3,
        "simbench" => 255,
        _ => 200,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum GarfieldLogicRuleDedupKey {
    Global(GarfieldRuleSupportBits),
    PerUnit {
        unit_kind: String,
        unit_index: usize,
        support: GarfieldRuleSupportBits,
    },
}

#[inline]
fn logic_rule_uses_per_unit_dedup(unit_kind: &str) -> bool {
    matches!(unit_kind, "gene" | "geneset")
}

#[inline]
fn logic_rule_dedup_key(
    rec: &GarfieldLogicRuleRecord,
    support: GarfieldRuleSupportBits,
) -> GarfieldLogicRuleDedupKey {
    if logic_rule_uses_per_unit_dedup(rec.unit_kind.as_str()) {
        GarfieldLogicRuleDedupKey::PerUnit {
            unit_kind: rec.unit_kind.clone(),
            unit_index: rec.unit_index,
            support,
        }
    } else {
        GarfieldLogicRuleDedupKey::Global(support)
    }
}

#[inline]
fn cmp_logic_rule_records_output(
    a: &GarfieldLogicRuleRecord,
    b: &GarfieldLogicRuleRecord,
) -> std::cmp::Ordering {
    logic_rule_output_kind_rank(a.unit_kind.as_str())
        .cmp(&logic_rule_output_kind_rank(b.unit_kind.as_str()))
        .then_with(|| a.unit_kind.cmp(&b.unit_kind))
        .then_with(|| a.unit_index.cmp(&b.unit_index))
        .then_with(|| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            a.selected_row_indices
                .len()
                .cmp(&b.selected_row_indices.len())
        })
        .then_with(|| a.snp_name.cmp(&b.snp_name))
        .then_with(|| a.unit_name.cmp(&b.unit_name))
}

fn dedup_logic_rule_records(
    records: Vec<GarfieldLogicRuleRecord>,
    logic_bits: &GarfieldLogicBits,
) -> Result<Vec<GarfieldLogicRuleRecord>, String> {
    let mut best_by_bits = HashMap::<GarfieldLogicRuleDedupKey, GarfieldLogicRuleRecord>::new();
    for rec in records.into_iter() {
        let support = materialize_logic_rule_record_support_bits(&rec, logic_bits)?;
        let key = logic_rule_dedup_key(&rec, support);
        match best_by_bits.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(rec);
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if cmp_logic_rule_records_same_support(&rec, slot.get()) == std::cmp::Ordering::Less
                {
                    slot.insert(rec);
                }
            }
        }
    }
    let mut out = best_by_bits.into_values().collect::<Vec<_>>();
    out.sort_by(cmp_logic_rule_records);
    Ok(out)
}

#[inline]
fn chrom_sort_key(chrom: &str) -> (u8, i32, String) {
    let norm = normalize_chrom(chrom);
    let bare = norm
        .strip_prefix("chr")
        .or_else(|| norm.strip_prefix("CHR"))
        .unwrap_or(norm.as_str());
    if let Ok(v) = bare.parse::<i32>() {
        return (0u8, v, norm);
    }
    let upper = bare.to_ascii_uppercase();
    match upper.as_str() {
        "X" => (1u8, 23, norm),
        "Y" => (1u8, 24, norm),
        "M" | "MT" => (1u8, 25, norm),
        _ => (2u8, i32::MAX, norm),
    }
}

#[inline]
fn strip_logic_display_target_suffix(token: &str) -> &str {
    let trimmed = token.trim().trim_start_matches('!').trim();
    if let Some((base, suffix)) = trimmed.rsplit_once('(') {
        if suffix.ends_with(')') && !base.trim().is_empty() {
            return base.trim();
        }
    }
    trimmed
}

#[inline]
fn parse_logic_record_primary_site(rec: &GarfieldLogicRuleRecord) -> Option<(String, i32)> {
    let first = strip_logic_display_target_suffix(rec.snp_name.split(['&', '|', '*', '^']).next()?);
    let (chrom, pos_txt) = first.rsplit_once('_')?;
    let pos = pos_txt.parse::<i32>().ok()?;
    Some((normalize_chrom(chrom), pos))
}

#[inline]
fn cmp_logic_rule_records_plink_order(
    a: &GarfieldLogicRuleRecord,
    b: &GarfieldLogicRuleRecord,
) -> std::cmp::Ordering {
    let (a_chrom, a_pos) = parse_logic_record_primary_site(a)
        .unwrap_or_else(|| (normalize_chrom(&a.chrom_field), a.pos));
    let (b_chrom, b_pos) = parse_logic_record_primary_site(b)
        .unwrap_or_else(|| (normalize_chrom(&b.chrom_field), b.pos));
    chrom_sort_key(&a_chrom)
        .cmp(&chrom_sort_key(&b_chrom))
        .then_with(|| a_pos.cmp(&b_pos))
        .then_with(|| a.snp_name.cmp(&b.snp_name))
        .then_with(|| a.unit_name.cmp(&b.unit_name))
}

#[inline]
fn effective_threads_local(threads: usize) -> usize {
    if threads > 0 {
        threads.max(1)
    } else {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .max(1)
    }
}

fn subset_vec_f64(values: &[f64], indices: &[usize]) -> Result<Vec<f64>, String> {
    let mut out = Vec::<f64>::with_capacity(indices.len());
    for &idx in indices.iter() {
        let v = *values
            .get(idx)
            .ok_or_else(|| format!("sample index out of range while subsetting y: {idx}"))?;
        out.push(v);
    }
    Ok(out)
}

#[inline]
fn bootstrap_sample_indices(n_samples: usize, seed: u64) -> Vec<usize> {
    if n_samples == 0 {
        return Vec::new();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_samples)
        .map(|_| rng.random_range(0..n_samples))
        .collect()
}

fn subset_cov_f64(
    x_cov: Option<&[f64]>,
    n_samples: usize,
    q_cov: usize,
    indices: &[usize],
) -> Result<Option<Vec<f64>>, String> {
    let Some(cov) = x_cov else {
        return Ok(None);
    };
    if cov.len() != n_samples.saturating_mul(q_cov) {
        return Err(format!(
            "x_cov payload length mismatch: got {}, expected {}",
            cov.len(),
            n_samples.saturating_mul(q_cov)
        ));
    }
    let mut out = vec![0.0_f64; indices.len().saturating_mul(q_cov)];
    for (dst_r, &src_r) in indices.iter().enumerate() {
        if src_r >= n_samples {
            return Err(format!(
                "sample index out of range while subsetting x_cov: {src_r}"
            ));
        }
        let src_st = src_r * q_cov;
        let dst_st = dst_r * q_cov;
        out[dst_st..dst_st + q_cov].copy_from_slice(&cov[src_st..src_st + q_cov]);
    }
    Ok(Some(out))
}

#[inline]
fn site_base_label<S: GarfieldChromPosSite>(site: &S) -> String {
    let snp_name = site.garfield_snp_name().trim();
    if garfield_snp_name_missing(snp_name) {
        format!("{}_{}", site.garfield_chrom(), site.garfield_pos())
    } else {
        snp_name.to_string()
    }
}

#[inline]
fn site_coord_label<S: GarfieldChromPosSite>(site: &S) -> String {
    format!("{}_{}", site.garfield_chrom(), site.garfield_pos())
}

#[inline]
fn site_model_label<S: GarfieldDisplaySite>(site: &S) -> &'static str {
    site.garfield_model_label()
}

#[inline]
fn literal_expr<S: GarfieldDisplaySite>(site: &S, negated: bool) -> String {
    if negated {
        format!("NOT {}({})", site_model_label(site), site_base_label(site))
    } else {
        format!("{}({})", site_model_label(site), site_base_label(site))
    }
}

#[inline]
fn literal_target_allele<S: GarfieldDisplaySite>(site: &S, negated: bool) -> String {
    literal_bim_alleles(site, negated).1
}

#[inline]
fn literal_target_name<S: GarfieldChromPosSite + GarfieldDisplaySite>(
    site: &S,
    negated: bool,
) -> String {
    format!(
        "{}[{}]",
        site_coord_label(site),
        literal_target_allele(site, negated)
    )
}

#[inline]
fn clean_logic_allele_label(allele: &str) -> String {
    allele
        .split('|')
        .next()
        .unwrap_or(allele)
        .trim()
        .to_string()
}

#[inline]
fn literal_bim_alleles<S: GarfieldDisplaySite>(site: &S, negated: bool) -> (String, String) {
    let zero = clean_logic_allele_label(site.garfield_ref_allele());
    let one = clean_logic_allele_label(site.garfield_alt_allele());
    if negated {
        (one, zero)
    } else {
        (zero, one)
    }
}

#[inline]
fn literal_ml_rank_name(row_index: usize, negated: bool) -> String {
    let rank = row_index.saturating_add(1);
    if negated {
        format!("!{rank}")
    } else {
        rank.to_string()
    }
}

fn rule_selected_global_rows(
    rule: &BeamRule,
    selected_global_rows: &[usize],
) -> Result<Vec<usize>, String> {
    let mut out = Vec::<usize>::with_capacity(rule.len());
    let first = *selected_global_rows
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    out.push(first);
    for (_, lit) in rule.rest.iter() {
        let idx = *selected_global_rows
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        out.push(idx);
    }
    Ok(out)
}

fn logic_rule_record_local_rule(rec: &GarfieldLogicRuleRecord) -> Result<BeamRule, String> {
    let expected = rec.selected_row_indices.len();
    if expected == 0 {
        return Err(format!(
            "GARFIELD stored rule '{}' has no selected row indices",
            rec.snp_name
        ));
    }
    if rec.display_negated.len() != expected {
        return Err(format!(
            "GARFIELD stored rule '{}' negation length mismatch: rows={}, negated={}",
            rec.snp_name,
            expected,
            rec.display_negated.len()
        ));
    }
    if rec.display_ops.len() + 1 != expected {
        return Err(format!(
            "GARFIELD stored rule '{}' operator length mismatch: rows={}, ops={}",
            rec.snp_name,
            expected,
            rec.display_ops.len()
        ));
    }
    let first = BeamLiteral {
        row_index: 0,
        group_id: 0,
        negated: rec.display_negated[0],
    };
    let mut rest = Vec::<(BeamBinaryOp, BeamLiteral)>::with_capacity(expected.saturating_sub(1));
    for idx in 1..expected {
        rest.push((
            rec.display_ops[idx - 1],
            BeamLiteral {
                row_index: idx,
                group_id: idx,
                negated: rec.display_negated[idx],
            },
        ));
    }
    Ok(BeamRule { first, rest })
}

fn materialize_rule_support_bits_from_selected_rows_full(
    rule: &BeamRule,
    selected_row_indices: &[usize],
    logic_bits: &GarfieldLogicBits,
    label: &str,
) -> Result<GarfieldRuleSupportBits, String> {
    let n_rows = selected_row_indices.len();
    let row_words = logic_bits.row_words;
    let mut bits_flat = Vec::<u64>::with_capacity(n_rows.saturating_mul(row_words));
    for &global_row_idx in selected_row_indices.iter() {
        let row_start = global_row_idx.saturating_mul(row_words);
        let row_end = row_start.saturating_add(row_words);
        let row = logic_bits.bits().get(row_start..row_end).ok_or_else(|| {
            format!(
                "GARFIELD stored rule '{}' row slice out of range: row={} row_words={}",
                label, global_row_idx, row_words
            )
        })?;
        bits_flat.extend_from_slice(row);
    }
    if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
        let mut bits_hi = Vec::<u64>::with_capacity(n_rows.saturating_mul(row_words));
        for &global_row_idx in selected_row_indices.iter() {
            let row_start = global_row_idx.saturating_mul(row_words);
            let row_end = row_start.saturating_add(row_words);
            let row = bits_hi_flat.get(row_start..row_end).ok_or_else(|| {
                format!(
                    "GARFIELD stored rule '{}' high-bit row slice out of range: row={} row_words={}",
                    label, global_row_idx, row_words
                )
            })?;
            bits_hi.extend_from_slice(row);
        }
        let (ge1, ge2) = materialize_rule_bits_dual(
            rule,
            bits_flat.as_slice(),
            bits_hi.as_slice(),
            row_words,
            n_rows,
            logic_bits.n_samples,
        )?;
        Ok(GarfieldRuleSupportBits::Dual {
            ge1: ge1.into_boxed_slice(),
            ge2: ge2.into_boxed_slice(),
        })
    } else {
        let bits = materialize_rule_bits(
            rule,
            bits_flat.as_slice(),
            row_words,
            n_rows,
            logic_bits.n_samples,
        )?;
        Ok(GarfieldRuleSupportBits::Binary(bits.into_boxed_slice()))
    }
}

fn materialize_logic_rule_record_support_bits(
    rec: &GarfieldLogicRuleRecord,
    logic_bits: &GarfieldLogicBits,
) -> Result<GarfieldRuleSupportBits, String> {
    if let Some(bits) = rec.support_bits.as_ref() {
        return Ok(bits.clone());
    }
    let local_rule = logic_rule_record_local_rule(rec)?;
    materialize_rule_support_bits_from_selected_rows_full(
        &local_rule,
        rec.selected_row_indices.as_slice(),
        logic_bits,
        rec.snp_name.as_str(),
    )
}

fn build_logic_units_from_groups<S: GarfieldChromPosSite>(
    sites: &[S],
    groups: &[Vec<(String, i32, i32)>],
    group_names: Option<&[String]>,
) -> Vec<GarfieldLogicUnit> {
    let chrom_idx = build_chrom_index(sites, sites.len());
    let mut out = Vec::<GarfieldLogicUnit>::new();
    for (gi, group) in groups.iter().enumerate() {
        let mut idx_all: Vec<usize> = Vec::new();
        for (chrom, start, end) in group.iter() {
            let mut iv = interval_indices(&chrom_idx, chrom, *start, *end);
            idx_all.append(&mut iv);
        }
        if idx_all.is_empty() {
            continue;
        }
        idx_all.sort_unstable();
        idx_all.dedup();
        let label = group_names
            .and_then(|v| v.get(gi).cloned())
            .unwrap_or_else(|| format!("group_{}", gi + 1));
        let spans = group
            .iter()
            .map(|(chrom, start, end)| {
                let (bp_start, bp_end) = normalize_interval_bounds(*start, *end);
                GarfieldUnitSpan {
                    chrom: normalize_chrom(chrom),
                    bp_start,
                    bp_end,
                }
            })
            .collect::<Vec<_>>();
        out.push(GarfieldLogicUnit {
            label,
            indices: idx_all,
            spans,
        });
    }
    out
}

fn select_logic_group_interval_defs_for_scan(
    groups: &[Vec<(String, i32, i32)>],
    group_names: Option<&[String]>,
    scan_bimranges: &[GarfieldScanBimRange],
    unit_kind_lc: &str,
) -> Result<
    (
        Vec<Vec<(String, i32, i32)>>,
        Vec<String>,
        Vec<GarfieldLogicUnit>,
    ),
    String,
> {
    let span_defs = build_logic_unit_span_defs_from_groups(groups, group_names);
    let mut selected_groups = Vec::<Vec<(String, i32, i32)>>::new();
    let mut selected_names = Vec::<String>::new();
    let mut selected_defs = Vec::<GarfieldLogicUnit>::new();
    for (gi, unit_def) in span_defs.into_iter().enumerate() {
        if scan_bimranges.is_empty() || unit_overlaps_scan_bimranges(&unit_def, scan_bimranges) {
            selected_groups.push(groups[gi].clone());
            selected_names.push(unit_def.label.clone());
            selected_defs.push(unit_def);
        }
    }
    if selected_groups.is_empty() {
        let joined = scan_bimranges
            .iter()
            .map(|r| format!("{}:{}-{}", r.chrom, r.bp_start, r.bp_end))
            .collect::<Vec<_>>()
            .join(",");
        return Err(format!(
            "--bimrange matched no scan units under unit_kind '{}': {}",
            unit_kind_lc, joined
        ));
    }
    Ok((selected_groups, selected_names, selected_defs))
}

fn build_logic_windows_from_sites<S: GarfieldChromPosSite>(
    sites: &[S],
    n_rows: usize,
    extension: usize,
    step: usize,
) -> Result<Vec<BinWindow>, String> {
    if n_rows == 0 {
        return Ok(Vec::new());
    }
    if extension == 0 || step == 0 || sites.is_empty() {
        return Ok(vec![BinWindow {
            chrom: "ALL".to_string(),
            bp_start: 0,
            bp_end: 0,
            indices: (0..n_rows).collect(),
        }]);
    }

    let mut groups: HashMap<String, Vec<(i32, usize)>> = HashMap::new();
    let mut chrom_order: Vec<String> = Vec::new();
    for (idx, site) in sites.iter().enumerate().take(n_rows) {
        let chrom = normalize_chrom(site.garfield_chrom());
        if !groups.contains_key(&chrom) {
            groups.insert(chrom.clone(), Vec::new());
            chrom_order.push(chrom.clone());
        }
        if let Some(v) = groups.get_mut(&chrom) {
            v.push((site.garfield_pos(), idx));
        }
    }
    if groups.is_empty() {
        return Ok(vec![BinWindow {
            chrom: "ALL".to_string(),
            bp_start: 0,
            bp_end: 0,
            indices: (0..n_rows).collect(),
        }]);
    }

    let mut windows = Vec::<BinWindow>::new();
    for chrom in chrom_order {
        check_ctrlc()?;
        let Some(mut pairs) = groups.remove(&chrom) else {
            continue;
        };
        if pairs.is_empty() {
            continue;
        }
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let n = pairs.len();
        let bps: Vec<i32> = pairs.iter().map(|(bp, _)| *bp).collect();
        let idxs: Vec<usize> = pairs.iter().map(|(_, i)| *i).collect();
        let min_bp = bps[0];
        let max_bp = bps[n - 1];
        let mut l = 0usize;
        let mut r = 0usize;
        let mut center = min_bp;
        let mut prev_sig = None::<(usize, usize, usize)>;
        let mut poll_ctr = 0usize;

        loop {
            if (poll_ctr & 4095) == 0 {
                check_ctrlc()?;
            }
            poll_ctr = poll_ctr.saturating_add(1);
            let left_i64 = (center as i64)
                .saturating_sub(extension as i64)
                .max(min_bp as i64);
            let right_i64 = (center as i64)
                .saturating_add(extension as i64)
                .min(max_bp as i64);

            while l < n && (bps[l] as i64) < left_i64 {
                l += 1;
            }
            if r < l {
                r = l;
            }
            while r < n && (bps[r] as i64) <= right_i64 {
                r += 1;
            }

            if r > l {
                let chunk = idxs[l..r].to_vec();
                let sig = (chunk[0], chunk[chunk.len() - 1], chunk.len());
                if prev_sig != Some(sig) {
                    let bp_start = if left_i64 > i32::MAX as i64 {
                        i32::MAX
                    } else if left_i64 < i32::MIN as i64 {
                        i32::MIN
                    } else {
                        left_i64 as i32
                    };
                    let bp_end = if right_i64 > i32::MAX as i64 {
                        i32::MAX
                    } else if right_i64 < i32::MIN as i64 {
                        i32::MIN
                    } else {
                        right_i64 as i32
                    };
                    windows.push(BinWindow {
                        chrom: chrom.clone(),
                        bp_start,
                        bp_end,
                        indices: chunk,
                    });
                    prev_sig = Some(sig);
                }
            }

            if center >= max_bp {
                break;
            }
            let next = (center as i64).saturating_add(step as i64);
            if next > i32::MAX as i64 {
                break;
            }
            center = next as i32;
        }

        let has_chrom_window = windows.last().map(|w| w.chrom == chrom).unwrap_or(false);
        if !has_chrom_window {
            windows.push(BinWindow {
                chrom: chrom.clone(),
                bp_start: min_bp,
                bp_end: max_bp,
                indices: idxs.clone(),
            });
        }
    }
    Ok(windows)
}

fn build_logic_units<S: GarfieldChromPosSite>(
    sites: &[S],
    unit_kind: &str,
    groups: Option<&[Vec<(String, i32, i32)>]>,
    group_names: Option<&[String]>,
    extension: usize,
    step: Option<usize>,
) -> Result<Vec<GarfieldLogicUnit>, String> {
    fn build_whole_genome_unit<S: GarfieldChromPosSite>(
        sites: &[S],
    ) -> Result<Vec<GarfieldLogicUnit>, String> {
        if sites.is_empty() {
            return Err("cannot build whole-genome GARFIELD unit from zero sites".to_string());
        }
        let mut spans = Vec::<GarfieldUnitSpan>::new();
        let mut current_chrom = normalize_chrom(sites[0].garfield_chrom());
        let mut current_start = sites[0].garfield_pos();
        let mut current_end = sites[0].garfield_pos();
        for site in sites.iter().skip(1) {
            let chrom = normalize_chrom(site.garfield_chrom());
            let pos = site.garfield_pos();
            if chrom == current_chrom {
                current_end = pos;
            } else {
                spans.push(GarfieldUnitSpan {
                    chrom: current_chrom,
                    bp_start: current_start,
                    bp_end: current_end,
                });
                current_chrom = chrom;
                current_start = pos;
                current_end = pos;
            }
        }
        spans.push(GarfieldUnitSpan {
            chrom: current_chrom,
            bp_start: current_start,
            bp_end: current_end,
        });
        Ok(vec![GarfieldLogicUnit {
            label: "whole_genome".to_string(),
            indices: (0..sites.len()).collect(),
            spans,
        }])
    }

    let kind = unit_kind.trim().to_ascii_lowercase();
    match kind.as_str() {
        "" | "window" => {
            let ext = extension.max(1);
            let step_eff = step.unwrap_or((ext / 2).max(1)).max(1);
            let windows = build_logic_windows_from_sites(sites, sites.len(), ext, step_eff)?;
            Ok(windows
                .into_iter()
                .map(|w| {
                    let GarfieldWindow {
                        chrom,
                        bp_start,
                        bp_end,
                        indices,
                    } = w;
                    GarfieldLogicUnit {
                        label: format!("{}:{}-{}", chrom, bp_start, bp_end),
                        indices,
                        spans: vec![GarfieldUnitSpan {
                            chrom,
                            bp_start,
                            bp_end,
                        }],
                    }
                })
                .collect())
        }
        "wholegenome" => build_whole_genome_unit(sites),
        "gene" | "geneset" | "group" => {
            let g = groups.ok_or_else(|| {
                "groups must be provided for unit_kind in {gene, geneset, group}".to_string()
            })?;
            Ok(build_logic_units_from_groups(sites, g, group_names))
        }
        other => Err(format!(
            "unit_kind must be one of: window, wholegenome, gene, geneset, group; got {other}"
        )),
    }
}

/// Build only windows that overlap the requested BIM ranges. The full
/// chromosome QC positions still define the window anchors and preserve the
/// same selected window contents, while unselected windows are never
/// materialized as large row-index vectors.
fn build_window_units_overlapping_bimranges<S: GarfieldChromPosSite>(
    sites: &[S],
    extension: usize,
    step: usize,
    scan_bimranges: &[GarfieldScanBimRange],
) -> Result<(Vec<GarfieldLogicUnit>, Vec<usize>), String> {
    if sites.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let ext = extension.max(1);
    let step = step.max(1);
    let mut groups = HashMap::<String, Vec<(i32, usize)>>::new();
    let mut chrom_order = Vec::<String>::new();
    for (index, site) in sites.iter().enumerate() {
        let chrom = normalize_chrom(site.garfield_chrom());
        if !groups.contains_key(&chrom) {
            groups.insert(chrom.clone(), Vec::new());
            chrom_order.push(chrom.clone());
        }
        groups
            .get_mut(&chrom)
            .expect("window chromosome group was inserted")
            .push((site.garfield_pos(), index));
    }

    let mut selected_units = Vec::<GarfieldLogicUnit>::new();
    let mut original_indices = Vec::<usize>::new();
    let mut original_index = 0usize;
    for chrom in chrom_order {
        check_ctrlc()?;
        let Some(mut pairs) = groups.remove(&chrom) else {
            continue;
        };
        if pairs.is_empty() {
            continue;
        }
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let positions = pairs.iter().map(|(bp, _)| *bp).collect::<Vec<_>>();
        let row_indices = pairs.iter().map(|(_, index)| *index).collect::<Vec<_>>();
        let min_bp = positions[0];
        let max_bp = *positions.last().unwrap_or(&min_bp);
        let mut left = 0usize;
        let mut right = 0usize;
        let mut center = min_bp;
        let mut previous_signature = None::<(usize, usize, usize)>;
        let mut has_any_window = false;
        let mut poll_counter = 0usize;

        loop {
            if (poll_counter & 4095) == 0 {
                check_ctrlc()?;
            }
            poll_counter = poll_counter.saturating_add(1);
            let left_bp = (center as i64)
                .saturating_sub(ext as i64)
                .max(min_bp as i64);
            let right_bp = (center as i64)
                .saturating_add(ext as i64)
                .min(max_bp as i64);
            while left < positions.len() && (positions[left] as i64) < left_bp {
                left += 1;
            }
            if right < left {
                right = left;
            }
            while right < positions.len() && (positions[right] as i64) <= right_bp {
                right += 1;
            }

            if right > left {
                let indices = row_indices[left..right].to_vec();
                let signature = (
                    indices[0],
                    *indices.last().unwrap_or(&indices[0]),
                    indices.len(),
                );
                if previous_signature != Some(signature) {
                    let unit = GarfieldLogicUnit {
                        label: format!("{}:{}-{}", chrom, left_bp, right_bp),
                        indices,
                        spans: vec![GarfieldUnitSpan {
                            chrom: chrom.clone(),
                            bp_start: clamp_i64_to_i32(left_bp),
                            bp_end: clamp_i64_to_i32(right_bp),
                        }],
                    };
                    if unit_overlaps_scan_bimranges(&unit, scan_bimranges) {
                        selected_units.push(unit);
                        original_indices.push(original_index);
                    }
                    original_index = original_index.saturating_add(1);
                    previous_signature = Some(signature);
                    has_any_window = true;
                }
            }

            if center >= max_bp {
                break;
            }
            let next = (center as i64).saturating_add(step as i64);
            if next > i32::MAX as i64 {
                break;
            }
            center = next as i32;
        }

        if !has_any_window {
            let unit = GarfieldLogicUnit {
                label: format!("{}:{}-{}", chrom, min_bp, max_bp),
                indices: row_indices,
                spans: vec![GarfieldUnitSpan {
                    chrom,
                    bp_start: min_bp,
                    bp_end: max_bp,
                }],
            };
            if unit_overlaps_scan_bimranges(&unit, scan_bimranges) {
                selected_units.push(unit);
                original_indices.push(original_index);
            }
            original_index = original_index.saturating_add(1);
        }
    }
    Ok((selected_units, original_indices))
}

#[inline]
fn parse_optional_ml_engine(method: &str) -> Result<Option<MlEngine>, String> {
    let norm = method.trim().to_ascii_lowercase();
    match norm.as_str() {
        // Default: univariate correlation screening; robust when a scan unit
        // contains many loci because cost stays linear in feature count.
        "" | "auto" => Ok(Some(MlEngine::Corr)),
        "none" | "skip" | "direct" => Ok(None),
        _ => parse_ml_engine(norm.as_str()).map(Some),
    }
}

#[inline]
fn normalized_rank_score(score: f64) -> f64 {
    if score.is_nan() {
        f64::NEG_INFINITY
    } else {
        score
    }
}

#[inline]
fn parse_rule_null_penalty_method(
    method: &str,
    quantile: f64,
) -> Result<RuleNullPenaltyMethod, String> {
    let norm = method.trim().to_ascii_lowercase();
    match norm.as_str() {
        "" | "auto" | "gev" | "gumbel" => Ok(RuleNullPenaltyMethod::GevGumbel {
            fwer_alpha: (1.0 - quantile).clamp(f64::EPSILON, 1.0 - f64::EPSILON),
        }),
        "bayes" | "bayesian" | "hbayes" | "hierarchical" => {
            Ok(RuleNullPenaltyMethod::BayesHierarchical {
                fwer_alpha: (1.0 - quantile).clamp(f64::EPSILON, 1.0 - f64::EPSILON),
            })
        }
        "quantile" | "empirical" | "q" => Ok(RuleNullPenaltyMethod::quantile(quantile)),
        other => Err(format!(
            "unsupported GARFIELD null-penalty method '{other}'; expected one of: gev, gumbel, bayes, quantile"
        )),
    }
}

#[inline]
fn rule_null_penalty_allows_adaptive_early_stop(method: RuleNullPenaltyMethod) -> bool {
    // GEV/Gumbel and hierarchical Bayesian tails both extrapolate an extreme
    // quantile. A stable location/scale fit after 50 repeats does not establish
    // that the 99th-percentile tail is stable, so both paths consume the full
    // repeat budget.
    !method.uses_tail_model()
}

#[inline]
fn null_prepare_engine_for_scan(unit_kind_lc: &str, engine: Option<MlEngine>) -> Option<MlEngine> {
    // Window null calibration must not reuse phenotype-selected ML rows from
    // the observed scan.  Keep geneset grouping semantics unchanged for now;
    // its synthetic units require the group-aware candidate layout.
    if unit_kind_lc == "geneset" {
        engine
    } else {
        None
    }
}

fn collect_topk_rule_null_metric_by_group_len_bucket(
    scored_hits: &[(RuleNullBucket, f64)],
    max_rule_len: usize,
    keep_topk: usize,
    use_rank_score_order: bool,
) -> Vec<(RuleNullBucket, f64)> {
    if keep_topk == 0 || scored_hits.is_empty() {
        return Vec::new();
    }
    let unit_group_bin_count = scored_hits
        .iter()
        .map(|(bucket, _)| usize::from(bucket.unit_group_bin()).saturating_add(1))
        .max()
        .unwrap_or(1);
    let group_len_bucket_count =
        rule_null_group_len_bucket_count(max_rule_len, unit_group_bin_count);
    let mut by_group_len_bucket = vec![Vec::<(RuleNullBucket, f64)>::new(); group_len_bucket_count];
    for &(bucket, score) in scored_hits.iter() {
        if !score.is_finite() {
            continue;
        }
        let group_len_bucket = rule_null_group_len_bucket_index(
            bucket.unit_group_bin(),
            bucket.rule_len,
            max_rule_len,
            unit_group_bin_count,
        );
        by_group_len_bucket[group_len_bucket].push((bucket, score));
    }
    let mut out = Vec::<(RuleNullBucket, f64)>::new();
    for bucket_scores in by_group_len_bucket.iter_mut() {
        if use_rank_score_order {
            bucket_scores.sort_by(|a, b| {
                normalized_rank_score(b.1)
                    .partial_cmp(&normalized_rank_score(a.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            bucket_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        out.extend(bucket_scores.iter().take(keep_topk).copied());
    }
    out
}

/// Collapse one permutation repeat to the maximum output score per
/// (unit-group, rule length) bucket.  The output penalty is applied to a
/// genome-wide scan, so its null distribution must use one global maximum per
/// repeat rather than one maximum per window.  Keeping the group bin in the
/// key preserves the separate calibration used by grouped/geneset scans.
fn collect_global_max_rule_null_metric_by_group_len_bucket(
    scores: &[GarfieldPermutationNullScores],
    use_train: bool,
) -> Vec<(RuleNullBucket, f64)> {
    let mut maxima = BTreeMap::<(u8, usize), (RuleNullBucket, f64)>::new();
    for score in scores.iter() {
        let value = if use_train {
            score.output_train_raw_score
        } else {
            score.output_test_raw_score
        };
        if !value.is_finite() {
            continue;
        }
        let bucket = score.bucket;
        let key = (bucket.unit_group_bin(), bucket.rule_len);
        match maxima.get_mut(&key) {
            Some((_, current)) if value > *current => *current = value,
            Some(_) => {}
            None => {
                maxima.insert(key, (bucket, value));
            }
        }
    }
    maxima.into_values().collect()
}

fn collect_global_max_rule_null_delta_by_group_len_bucket(
    scores: &[GarfieldPermutationNullScores],
    use_train: bool,
) -> Vec<(RuleNullBucket, f64)> {
    let mut maxima = BTreeMap::<(u8, usize), (RuleNullBucket, f64)>::new();
    for score in scores.iter() {
        let value = if use_train {
            score.delta_train_score
        } else {
            score.delta_test_score
        };
        if !value.is_finite() {
            continue;
        }
        let bucket = score.bucket;
        let key = (bucket.unit_group_bin(), bucket.rule_len);
        match maxima.get_mut(&key) {
            Some((_, current)) if value > *current => *current = value,
            Some(_) => {}
            None => {
                maxima.insert(key, (bucket, value));
            }
        }
    }
    maxima.into_values().collect()
}

fn collect_global_max_rule_null_family_4plus(
    scores: &[GarfieldPermutationNullScores],
    use_train: bool,
) -> Option<f64> {
    scores
        .iter()
        .filter(|score| score.bucket.rule_len >= 4)
        .map(|score| {
            if use_train {
                score.output_train_raw_score
            } else {
                score.output_test_raw_score
            }
        })
        .filter(|value| value.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn collect_global_max_rule_null_delta_family_4plus(
    scores: &[GarfieldPermutationNullScores],
    use_train: bool,
) -> Option<f64> {
    scores
        .iter()
        .filter(|score| score.bucket.rule_len >= 4)
        .map(|score| {
            if use_train {
                score.delta_train_score
            } else {
                score.delta_test_score
            }
        })
        .filter(|value| value.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

#[inline]
fn rule_null_layer_label(idx: usize) -> &'static str {
    // Keep the report aligned with the exact length buckets used by the
    // calibrator; do not collapse lengths 3..5 into a single `layer3+` row.
    match idx {
        0 => "layer1",
        1 => "layer2",
        2 => "layer3",
        3 => "layer4",
        4 => "layer5",
        _ => "layer5+",
    }
}

#[inline]
fn rule_null_unit_group_label(unit_group_bin: usize) -> String {
    format!("w{}", unit_group_bin.saturating_add(1))
}

#[inline]
fn scores_tied_for_keep(a: f64, b: f64) -> bool {
    let aa = normalized_rank_score(a);
    let bb = normalized_rank_score(b);
    if aa == bb {
        return true;
    }
    if !aa.is_finite() || !bb.is_finite() {
        return false;
    }
    let scale = aa.abs().max(bb.abs()).max(1.0);
    (aa - bb).abs() <= 1e-12 * scale
}

fn extend_keep_with_score_ties<T, F>(items: &[T], keep_n: usize, score_of: F) -> usize
where
    F: Fn(&T) -> f64,
{
    if keep_n == 0 || items.is_empty() {
        return 0;
    }
    let mut keep = keep_n.min(items.len());
    if keep >= items.len() {
        return items.len();
    }
    let cutoff = score_of(&items[keep - 1]);
    while keep < items.len() && scores_tied_for_keep(score_of(&items[keep]), cutoff) {
        keep += 1;
    }
    keep
}

fn apply_logic_rule_output_limit(
    records: &mut Vec<GarfieldLogicRuleRecord>,
    max_output_rules: usize,
    max_output_ratio: f64,
) -> Result<(), String> {
    if max_output_rules > 0 {
        if records.len() > max_output_rules {
            let keep_n =
                extend_keep_with_score_ties(records.as_slice(), max_output_rules, |rec| rec.score);
            records.truncate(keep_n);
        }
        return Ok(());
    }
    if max_output_ratio == 0.0 {
        return Ok(());
    }
    if !max_output_ratio.is_finite() || max_output_ratio <= 0.0 || max_output_ratio > 1.0 {
        return Err(format!(
            "max_output_ratio must be within (0, 1], got {}",
            max_output_ratio
        ));
    }
    let keep_n = ((records.len() as f64) * max_output_ratio).ceil().max(1.0) as usize;
    if records.len() > keep_n {
        let keep_n = extend_keep_with_score_ties(records.as_slice(), keep_n, |rec| rec.score);
        records.truncate(keep_n);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Max6GrmMax6PairOutput {
    unit_index: usize,
    candidate_rank: usize,
    first_global_row: usize,
    second_global_row: usize,
    four_cell_counts: Option<[usize; 4]>,
    dosage: f64,
    max6: f64,
    best_channel: &'static str,
    pair_pve: f64,
}

#[derive(Clone, Debug)]
struct Max6GrmMax6PermutationPoolMeta {
    unit_index: usize,
    byte_offset: u64,
    n_rows: usize,
}

struct Max6GrmMax6PermutationPool {
    unit_index: usize,
    selected_global_rows: Vec<usize>,
}

fn dense_binary_rows_from_packed(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_rows: usize,
) -> Result<Vec<f64>, String> {
    let needed = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| "packed Max6 rows overflow dense allocation".to_string())?;
    if bits.len() < needed {
        return Err(format!(
            "packed Max6 rows length={} smaller than required {needed}",
            bits.len()
        ));
    }
    let mut row_major = vec![0.0; n_samples.saturating_mul(n_rows)];
    for row in 0..n_rows {
        for sample in 0..n_samples {
            let word = bits[row * row_words + sample / 64];
            row_major[sample * n_rows + row] = ((word >> (sample % 64)) & 1) as f64;
        }
    }
    Ok(row_major)
}

fn dense_candidate_endpoint_matrix_global(
    bits_flat: &[u64],
    row_words: usize,
    n_samples: usize,
    candidate_pairs: &[(usize, usize)],
) -> Result<(Vec<f64>, Vec<usize>, Vec<usize>, Vec<usize>), String> {
    let mut endpoint_rows = Vec::<usize>::with_capacity(candidate_pairs.len().saturating_mul(2));
    for &(left_global, right_global) in candidate_pairs {
        if left_global == right_global {
            return Err("permutation candidate pair must contain two distinct markers".to_string());
        }
        endpoint_rows.push(left_global);
        endpoint_rows.push(right_global);
    }
    endpoint_rows.sort_unstable();
    endpoint_rows.dedup();

    let endpoint_local = endpoint_rows
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect::<HashMap<usize, usize>>();
    let mut pair_i = Vec::with_capacity(candidate_pairs.len());
    let mut pair_j = Vec::with_capacity(candidate_pairs.len());
    for &(left_global, right_global) in candidate_pairs {
        pair_i.push(
            *endpoint_local
                .get(&left_global)
                .ok_or_else(|| "permutation endpoint mapping is incomplete".to_string())?,
        );
        pair_j.push(
            *endpoint_local
                .get(&right_global)
                .ok_or_else(|| "permutation endpoint mapping is incomplete".to_string())?,
        );
    }

    let packed = gather_rows_by_indices(
        bits_flat,
        row_words,
        endpoint_rows.as_slice(),
        "max6 permutation endpoint rows",
    )?;
    let x = dense_binary_rows_from_packed(
        packed.as_slice(),
        row_words,
        n_samples,
        endpoint_rows.len(),
    )?;
    Ok((x, pair_i, pair_j, endpoint_rows))
}

struct Max6GrmMax6UnitScan {
    outputs: Vec<Max6GrmMax6PairOutput>,
    permutation_pool: Option<Max6GrmMax6PermutationPool>,
    deferred_screen: Option<Max6DeferredWindowScreen>,
    screen_pairs_passed: usize,
    final_screen_subset_audit_pairs_checked: usize,
    candidate_pair_keys: Vec<u64>,
    candidate_score_observations: Vec<Max6CandidateScoreObservation>,
    window_unique_candidate_endpoints: usize,
    timing: Max6ObservedUnitTiming,
    skipped_message: Option<String>,
}

struct Max6DeferredWindowScreen {
    selected_global_rows: Vec<usize>,
    candidates: Vec<logic_residual_screen::ResidualScreenCandidate>,
}

struct Max6PrecomputedScreen {
    screen: logic_residual_screen::ResidualScreenResult,
    kernel_seconds: f64,
    shared_grm_int: Option<grm_max6::GrmIntScoreBatch>,
}

#[derive(Debug, PartialEq, Eq)]
struct Max6SharedCandidatePairPlan {
    global_pairs: Vec<(usize, usize)>,
    unit_pair_indices: Vec<Vec<usize>>,
}

fn max6_shared_grm_int_diagnostic_recipient(unit_pair_indices: &[Vec<usize>]) -> Option<usize> {
    unit_pair_indices
        .iter()
        .position(|pair_indices| !pair_indices.is_empty())
        .or_else(|| (!unit_pair_indices.is_empty()).then_some(0))
}

/// Deduplicate window-local screen candidates by their global marker pair,
/// keeping an index map that restores each window's original candidate order.
fn max6_shared_candidate_pair_plan(
    selected_global_rows: &[&[usize]],
    local_pairs_by_unit: &[Vec<(usize, usize)>],
) -> Result<Max6SharedCandidatePairPlan, String> {
    if selected_global_rows.len() != local_pairs_by_unit.len() {
        return Err(format!(
            "Max6 shared candidate plan has {} row lists for {} unit pair lists",
            selected_global_rows.len(),
            local_pairs_by_unit.len()
        ));
    }

    let mut pair_keys_by_unit = Vec::with_capacity(local_pairs_by_unit.len());
    let mut unique_pair_keys = Vec::new();
    for (unit_index, (rows, local_pairs)) in selected_global_rows
        .iter()
        .zip(local_pairs_by_unit)
        .enumerate()
    {
        let mut pair_keys = Vec::with_capacity(local_pairs.len());
        for &(left, right) in local_pairs {
            let global_left = *rows.get(left).ok_or_else(|| {
                format!(
                    "Max6 shared candidate pair left index {left} is out of range for unit {unit_index}"
                )
            })?;
            let global_right = *rows.get(right).ok_or_else(|| {
                format!(
                    "Max6 shared candidate pair right index {right} is out of range for unit {unit_index}"
                )
            })?;
            let key = max6_diagnostic_pair_key(global_left, global_right)?;
            pair_keys.push(key);
            unique_pair_keys.push(key);
        }
        pair_keys_by_unit.push(pair_keys);
    }

    unique_pair_keys.sort_unstable();
    unique_pair_keys.dedup();
    let global_pairs = unique_pair_keys
        .iter()
        .map(|&key| (((key >> 32) as u32) as usize, (key as u32) as usize))
        .collect::<Vec<_>>();
    let pair_index_by_key = unique_pair_keys
        .iter()
        .enumerate()
        .map(|(index, &key)| (key, index))
        .collect::<HashMap<_, _>>();
    let unit_pair_indices = pair_keys_by_unit
        .iter()
        .map(|pair_keys| {
            pair_keys
                .iter()
                .map(|key| {
                    pair_index_by_key.get(key).copied().ok_or_else(|| {
                        "Max6 shared candidate pair index map is incomplete".to_string()
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Max6SharedCandidatePairPlan {
        global_pairs,
        unit_pair_indices,
    })
}

#[inline]
fn max6_order_by_score_and_tie(scores: &[f64], tie_keys: &[usize]) -> Result<Vec<usize>, String> {
    if scores.len() != tie_keys.len() {
        return Err(format!(
            "Max6 score/tie-key lengths differ: {} vs {}",
            scores.len(),
            tie_keys.len()
        ));
    }
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| tie_keys[left].cmp(&tie_keys[right]))
    });
    Ok(order)
}

#[derive(Clone, Copy, Debug, Default)]
struct Max6ObservedScreenProfile {
    units: usize,
    sample_pairs: usize,
    total_pairs: usize,
    sample_words: usize,
    nonzero_intersections: usize,
    intersection_popcount_sum: usize,
    lookup_calls: usize,
    lut8_word_calls: usize,
    setbit_calls: usize,
    bitset_seconds: f64,
    weighted_y_lookup_seconds: f64,
    margin_recovery_seconds: f64,
    score_seconds: f64,
    heap_seconds: f64,
    final_sort_seconds: f64,
    replay_seconds: f64,
    heap_pushes: usize,
    heap_replacements: usize,
}

impl Max6ObservedScreenProfile {
    fn accumulate(&mut self, other: Self) {
        self.units = self.units.saturating_add(other.units);
        self.sample_pairs = self.sample_pairs.saturating_add(other.sample_pairs);
        self.total_pairs = self.total_pairs.saturating_add(other.total_pairs);
        self.sample_words = self.sample_words.saturating_add(other.sample_words);
        self.nonzero_intersections = self
            .nonzero_intersections
            .saturating_add(other.nonzero_intersections);
        self.intersection_popcount_sum = self
            .intersection_popcount_sum
            .saturating_add(other.intersection_popcount_sum);
        self.lookup_calls = self.lookup_calls.saturating_add(other.lookup_calls);
        self.lut8_word_calls = self.lut8_word_calls.saturating_add(other.lut8_word_calls);
        self.setbit_calls = self.setbit_calls.saturating_add(other.setbit_calls);
        self.bitset_seconds += other.bitset_seconds;
        self.weighted_y_lookup_seconds += other.weighted_y_lookup_seconds;
        self.margin_recovery_seconds += other.margin_recovery_seconds;
        self.score_seconds += other.score_seconds;
        self.heap_seconds += other.heap_seconds;
        self.final_sort_seconds += other.final_sort_seconds;
        self.replay_seconds += other.replay_seconds;
        self.heap_pushes = self.heap_pushes.saturating_add(other.heap_pushes);
        self.heap_replacements = self
            .heap_replacements
            .saturating_add(other.heap_replacements);
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Max6ObservedUnitTiming {
    screen_input_prepare_seconds: f64,
    screen_kernel_seconds: f64,
    candidate_prepare_seconds: f64,
    candidate_key_collection_seconds: f64,
    rerank: CandidateRerankTiming,
    grm_int: GrmIntScoreDiagnostics,
    output_finalize_seconds: f64,
    screen_pairs_evaluated: usize,
    screen_candidates: usize,
    exact_candidates: usize,
    grm_int_candidates: usize,
    exact_input_duplicates: usize,
    exact_input_self_pairs: usize,
    screen_profile: Max6ObservedScreenProfile,
}

#[derive(Clone, Copy, Debug, Default)]
struct Max6BatchScanProfile {
    scan_merge_wall_seconds: f64,
    fold_task_count: usize,
    fold_task_wall_sum_seconds: f64,
    fold_task_wall_max_seconds: f64,
    fold_anchor_count_min: usize,
    fold_anchor_count_max: usize,
    fold_active_workers: usize,
    worker_task_wall_min_seconds: f64,
    worker_task_wall_median_seconds: f64,
    worker_task_wall_max_seconds: f64,
    worker_finish_skew_seconds: f64,
    post_last_fold_tail_seconds: f64,
    topk_merge_task_seconds: f64,
    topk_merge_max_task_seconds: f64,
    topk_merge_task_count: usize,
    topk_merge_critical_path_seconds: f64,
    scan_reduce_critical_path_seconds: f64,
    pair_profile_sampled_anchors: usize,
    pair_profile_sampled_pairs: usize,
    pair_profile_sampled_scored_pairs: usize,
    pair_profile_sampled_score_offers: usize,
    pair_profile_cursor_merge_seconds: f64,
    pair_profile_min_cell_gate_seconds: f64,
    pair_profile_sufficient_stats_seconds: f64,
    pair_profile_topk_offer_seconds: f64,
}

#[derive(Clone, Debug, Default)]
struct Max6ObservedScanTiming {
    units_timed: usize,
    screen_input_prepare_task_seconds: f64,
    screen_kernel_task_seconds: f64,
    screen_batch_count: usize,
    screen_batch_pairs_scored_unique: usize,
    screen_batch_score_offers: usize,
    screen_batch_direct_fallbacks: usize,
    screen_batch_prepare_wall_seconds: f64,
    screen_membership_prepare_wall_seconds: f64,
    screen_pair_scan_merge_wall_seconds: f64,
    screen_heap_merge_task_seconds: f64,
    screen_finalist_wall_seconds: f64,
    grm_int_batch_wall_seconds: f64,
    scan_result_drain_wall_seconds: f64,
    screen_parallel_batches: usize,
    screen_serial_batches: usize,
    screen_source_rows_sum: usize,
    screen_source_rows_max: usize,
    screen_memberships_sum: usize,
    screen_batch_profiles: Vec<Max6BatchScanProfile>,
    candidate_prepare_task_seconds: f64,
    rerank_endpoint_prepare_task_seconds: f64,
    rerank_kernel_task_seconds: f64,
    rerank_pcx_task_seconds: f64,
    rerank_pcx_sparse_support_units: usize,
    rerank_geometry_task_seconds: f64,
    rerank_geometry_marker_moments_task_seconds: f64,
    rerank_geometry_pairwise_task_seconds: f64,
    rerank_geometry_au_validation_task_seconds: f64,
    rerank_geometry_au_allocation_task_seconds: f64,
    rerank_geometry_au_materialization_task_seconds: f64,
    rerank_geometry_au_gemm_task_seconds: f64,
    rerank_geometry_au_candidate_lookup_task_seconds: f64,
    rerank_geometry_au_fused_py_task_seconds: f64,
    rerank_geometry_au_copy_task_seconds: f64,
    rerank_geometry_au_max_workspace_bytes: usize,
    rerank_geometry_lookup_task_seconds: f64,
    rerank_geometry_interaction_py_task_seconds: f64,
    rerank_geometry_unclassified_task_seconds: f64,
    rerank_q_task_seconds: f64,
    rerank_q_validation_task_seconds: f64,
    rerank_q_bitset_task_seconds: f64,
    rerank_q_cell_task_seconds: f64,
    rerank_q_enumerate_task_seconds: f64,
    rerank_q_quadratic_task_seconds: f64,
    rerank_q_reconstruct_task_seconds: f64,
    rerank_score_task_seconds: f64,
    rerank_topk_task_seconds: f64,
    rerank_other_task_seconds: f64,
    output_finalize_task_seconds: f64,
    screen_candidate_pairs: usize,
    screen_pairs_evaluated: usize,
    exact_candidate_pairs: usize,
    scored_candidate_pairs: usize,
    grm_int_pair_occurrences: usize,
    grm_int_singular_endpoint_design_pairs: usize,
    grm_int_negligible_interaction_information_pairs: usize,
    grm_int_nonfinite_pairs: usize,
    grm_int_batches: usize,
    grm_int_max_workspace_bytes: usize,
    grm_int_task_seconds: f64,
    exact_input_duplicate_pairs: usize,
    exact_input_self_pairs: usize,
    candidate_reuse_diagnostic_enabled: bool,
    candidate_score_reuse_audit_enabled: bool,
    candidate_score_reuse_summary: Option<Max6CandidateScoreReuseSummary>,
    candidate_score_audit_finalize_wall_seconds: f64,
    candidate_score_audit_buffer_bytes: usize,
    candidate_unique_pairs_global: usize,
    candidate_duplicate_occurrences: usize,
    candidate_duplicate_fraction: f64,
    candidate_multiplicity_p50: usize,
    candidate_multiplicity_p90: usize,
    candidate_multiplicity_p99: usize,
    candidate_multiplicity_max: usize,
    candidate_unique_endpoints_window_sum: usize,
    candidate_unique_endpoints_global: usize,
    candidate_endpoint_reuse_factor: f64,
    candidate_key_collection_task_seconds: f64,
    candidate_key_finalize_wall_seconds: f64,
    candidate_key_buffer_bytes: usize,
    candidate_hash_window_count: usize,
    candidate_set_hash_by_window_order: u64,
    exact_topk_set_hash_by_window_order: u64,
    exact_topk_order_hash_by_window_order: u64,
    candidate_window_pairs_min: usize,
    candidate_window_pairs_p50: usize,
    candidate_window_pairs_p90: usize,
    candidate_window_pairs_p99: usize,
    candidate_window_pairs_max: usize,
    candidate_window_screen_pairs_evaluated_min: usize,
    candidate_window_screen_pairs_evaluated_p50: usize,
    candidate_window_screen_pairs_evaluated_p90: usize,
    candidate_window_screen_pairs_evaluated_p99: usize,
    candidate_window_screen_pairs_evaluated_max: usize,
    candidate_window_endpoints_min: usize,
    candidate_window_endpoints_p50: usize,
    candidate_window_endpoints_p90: usize,
    candidate_window_endpoints_p99: usize,
    candidate_window_endpoints_max: usize,
    screen_profile: Max6ObservedScreenProfile,
}

impl Max6ObservedScanTiming {
    fn accumulate_unit(&mut self, unit: Max6ObservedUnitTiming) {
        self.units_timed = self.units_timed.saturating_add(1);
        self.screen_input_prepare_task_seconds += unit.screen_input_prepare_seconds;
        self.screen_kernel_task_seconds += unit.screen_kernel_seconds;
        self.candidate_prepare_task_seconds += unit.candidate_prepare_seconds;
        self.candidate_key_collection_task_seconds += unit.candidate_key_collection_seconds;
        self.rerank_endpoint_prepare_task_seconds += unit.rerank.endpoint_prepare_seconds;
        self.rerank_kernel_task_seconds += unit.rerank.kernel_seconds;
        self.rerank_pcx_task_seconds += unit.rerank.pcx_u_seconds;
        self.rerank_pcx_sparse_support_units += unit.rerank.pcx_sparse_support as usize;
        self.rerank_geometry_task_seconds += unit.rerank.candidate_geometry_seconds;
        self.rerank_geometry_marker_moments_task_seconds +=
            unit.rerank.geometry_marker_moments_seconds;
        self.rerank_geometry_pairwise_task_seconds += unit.rerank.geometry_pairwise_seconds;
        self.rerank_geometry_au_validation_task_seconds +=
            unit.rerank.geometry_au_validation_seconds;
        self.rerank_geometry_au_allocation_task_seconds +=
            unit.rerank.geometry_au_allocation_seconds;
        self.rerank_geometry_au_materialization_task_seconds +=
            unit.rerank.geometry_au_materialization_seconds;
        self.rerank_geometry_au_gemm_task_seconds += unit.rerank.geometry_au_gemm_seconds;
        self.rerank_geometry_au_candidate_lookup_task_seconds +=
            unit.rerank.geometry_au_candidate_lookup_seconds;
        self.rerank_geometry_au_fused_py_task_seconds += unit.rerank.geometry_au_fused_py_seconds;
        self.rerank_geometry_au_copy_task_seconds += unit.rerank.geometry_au_copy_seconds;
        self.rerank_geometry_au_max_workspace_bytes = self
            .rerank_geometry_au_max_workspace_bytes
            .max(unit.rerank.geometry_au_workspace_bytes);
        self.rerank_geometry_lookup_task_seconds += unit.rerank.geometry_lookup_seconds;
        self.rerank_geometry_interaction_py_task_seconds +=
            unit.rerank.geometry_interaction_py_seconds;
        let measured_geometry_components = unit.rerank.geometry_marker_moments_seconds
            + unit.rerank.geometry_pairwise_seconds
            + unit.rerank.geometry_lookup_seconds
            + unit.rerank.geometry_interaction_py_seconds;
        self.rerank_geometry_unclassified_task_seconds +=
            (unit.rerank.candidate_geometry_seconds - measured_geometry_components).max(0.0);
        self.rerank_q_task_seconds += unit.rerank.q_seconds;
        self.rerank_q_validation_task_seconds += unit.rerank.q_validation_seconds;
        self.rerank_q_bitset_task_seconds += unit.rerank.q_bitset_seconds;
        self.rerank_q_cell_task_seconds += unit.rerank.q_cell_seconds;
        self.rerank_q_enumerate_task_seconds += unit.rerank.q_enumerate_seconds;
        self.rerank_q_quadratic_task_seconds += unit.rerank.q_quadratic_seconds;
        self.rerank_q_reconstruct_task_seconds += unit.rerank.q_reconstruct_seconds;
        self.rerank_score_task_seconds += unit.rerank.score_seconds;
        self.rerank_topk_task_seconds += unit.rerank.topk_seconds;
        let measured_rerank_components = unit.rerank.endpoint_prepare_seconds
            + unit.rerank.py_seconds
            + unit.rerank.pcx_u_seconds
            + unit.rerank.candidate_geometry_seconds
            + unit.rerank.q_seconds
            + unit.rerank.score_seconds
            + unit.rerank.topk_seconds;
        self.rerank_other_task_seconds +=
            (unit.rerank.kernel_seconds - measured_rerank_components).max(0.0);
        self.output_finalize_task_seconds += unit.output_finalize_seconds;
        self.screen_candidate_pairs = self
            .screen_candidate_pairs
            .saturating_add(unit.screen_candidates);
        self.screen_pairs_evaluated = self
            .screen_pairs_evaluated
            .saturating_add(unit.screen_pairs_evaluated);
        self.exact_candidate_pairs = self
            .exact_candidate_pairs
            .saturating_add(unit.exact_candidates);
        self.scored_candidate_pairs = self
            .scored_candidate_pairs
            .saturating_add(unit.exact_candidates)
            .saturating_add(unit.grm_int_candidates);
        self.grm_int_pair_occurrences = self
            .grm_int_pair_occurrences
            .saturating_add(unit.grm_int_candidates);
        self.grm_int_singular_endpoint_design_pairs = self
            .grm_int_singular_endpoint_design_pairs
            .saturating_add(unit.grm_int.singular_endpoint_design);
        self.grm_int_negligible_interaction_information_pairs = self
            .grm_int_negligible_interaction_information_pairs
            .saturating_add(unit.grm_int.negligible_residual_interaction_information);
        self.grm_int_nonfinite_pairs = self
            .grm_int_nonfinite_pairs
            .saturating_add(unit.grm_int.nonfinite);
        self.grm_int_batches = self.grm_int_batches.saturating_add(unit.grm_int.batches);
        self.grm_int_max_workspace_bytes = self
            .grm_int_max_workspace_bytes
            .max(unit.grm_int.max_workspace_bytes);
        self.grm_int_task_seconds += unit.grm_int.elapsed_seconds;
        self.exact_input_duplicate_pairs = self
            .exact_input_duplicate_pairs
            .saturating_add(unit.exact_input_duplicates);
        self.exact_input_self_pairs = self
            .exact_input_self_pairs
            .saturating_add(unit.exact_input_self_pairs);
        self.screen_profile.accumulate(unit.screen_profile);
    }
}

#[inline]
fn max6_usize_quantile_sorted(values: &[usize], quantile: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let q = quantile.clamp(0.0, 1.0);
    let rank = ((q * values.len() as f64).ceil() as usize).clamp(1, values.len());
    values[rank - 1]
}

fn max6_candidate_multiplicity_quantile(
    histogram: &[usize],
    total_unique_pairs: usize,
    quantile: f64,
) -> usize {
    if total_unique_pairs == 0 {
        return 0;
    }
    let q = quantile.clamp(0.0, 1.0);
    let rank = ((q * total_unique_pairs as f64).ceil() as usize).clamp(1, total_unique_pairs);
    let mut cumulative = 0usize;
    for (multiplicity, &count) in histogram.iter().enumerate().skip(1) {
        cumulative = cumulative.saturating_add(count);
        if cumulative >= rank {
            return multiplicity;
        }
    }
    histogram.len().saturating_sub(1)
}

const MAX6_DIAGNOSTIC_FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const MAX6_DIAGNOSTIC_FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug)]
struct Max6CandidateScoreObservation {
    pair_key: u64,
    window_index: u32,
    tie_key: u32,
    score: f64,
    cached_score: f64,
    channel_id: u8,
}

const MAX6_SCORE_REUSE_AUDIT_TOP_K: [usize; 5] = [10, 100, 500, 2_000, 5_000];

#[derive(Clone, Copy, Debug, Default)]
struct Max6CandidateScoreTopKStability {
    top_k: usize,
    windows_compared: usize,
    windows_with_membership_change: usize,
    total_membership_symmetric_difference: usize,
    max_membership_symmetric_difference: usize,
    windows_with_order_change: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct Max6CandidateScoreReuseSummary {
    observation_count: usize,
    unique_pair_count: usize,
    repeated_pair_count: usize,
    duplicate_occurrence_count: usize,
    bitwise_equal_repeated_pair_count: usize,
    score_mismatch_pair_count: usize,
    best_channel_mismatch_pair_count: usize,
    nonfinite_score_occurrence_count: usize,
    max_abs_score_difference: f64,
    topk_stability: [Max6CandidateScoreTopKStability; 5],
}

fn summarize_max6_candidate_score_reuse_observations(
    observations: &mut [Max6CandidateScoreObservation],
) -> Max6CandidateScoreReuseSummary {
    observations
        .sort_unstable_by_key(|observation| (observation.pair_key, observation.window_index));
    let mut summary = Max6CandidateScoreReuseSummary {
        observation_count: observations.len(),
        ..Default::default()
    };
    let mut index = 0usize;
    while index < observations.len() {
        let pair_key = observations[index].pair_key;
        let mut end = index + 1;
        while end < observations.len() && observations[end].pair_key == pair_key {
            end += 1;
        }
        summary.unique_pair_count += 1;
        summary.duplicate_occurrence_count += end - index - 1;
        let first = observations[index].score;
        let first_channel = observations[index].channel_id;
        for observation in &mut observations[index..end] {
            observation.cached_score = first;
        }
        if !first.is_finite() {
            summary.nonfinite_score_occurrence_count += 1;
        }
        let mut bitwise_equal = true;
        let mut channel_equal = true;
        let mut min_score = first;
        let mut max_score = first;
        for observation in &observations[(index + 1)..end] {
            if !observation.score.is_finite() {
                summary.nonfinite_score_occurrence_count += 1;
            }
            bitwise_equal &= observation.score.to_bits() == first.to_bits();
            channel_equal &= observation.channel_id == first_channel;
            if observation.score.is_finite() && first.is_finite() {
                if observation.score < min_score {
                    min_score = observation.score;
                }
                if observation.score > max_score {
                    max_score = observation.score;
                }
            }
        }
        if end - index > 1 {
            summary.repeated_pair_count += 1;
            summary.best_channel_mismatch_pair_count += usize::from(!channel_equal);
            if bitwise_equal {
                summary.bitwise_equal_repeated_pair_count += 1;
            } else {
                summary.score_mismatch_pair_count += 1;
                if first.is_finite() && min_score.is_finite() && max_score.is_finite() {
                    summary.max_abs_score_difference =
                        summary.max_abs_score_difference.max(max_score - min_score);
                }
            }
        }
        index = end;
    }

    summary.topk_stability =
        MAX6_SCORE_REUSE_AUDIT_TOP_K.map(|top_k| Max6CandidateScoreTopKStability {
            top_k,
            ..Default::default()
        });
    observations
        .sort_unstable_by_key(|observation| (observation.window_index, observation.tie_key));
    let mut window_start = 0usize;
    while window_start < observations.len() {
        let window_index = observations[window_start].window_index;
        let mut window_end = window_start + 1;
        while window_end < observations.len()
            && observations[window_end].window_index == window_index
        {
            window_end += 1;
        }
        let window_observations = &observations[window_start..window_end];
        let mut actual_order = (0..window_observations.len()).collect::<Vec<_>>();
        actual_order.sort_unstable_by(|&left, &right| {
            window_observations[right]
                .score
                .total_cmp(&window_observations[left].score)
                .then_with(|| {
                    window_observations[left]
                        .tie_key
                        .cmp(&window_observations[right].tie_key)
                })
        });
        let mut cached_order = (0..window_observations.len()).collect::<Vec<_>>();
        cached_order.sort_unstable_by(|&left, &right| {
            window_observations[right]
                .cached_score
                .total_cmp(&window_observations[left].cached_score)
                .then_with(|| {
                    window_observations[left]
                        .tie_key
                        .cmp(&window_observations[right].tie_key)
                })
        });
        let mut actual_rank = vec![0usize; window_observations.len()];
        let mut cached_rank = vec![0usize; window_observations.len()];
        for (rank, &observation_index) in actual_order.iter().enumerate() {
            actual_rank[observation_index] = rank;
        }
        for (rank, &observation_index) in cached_order.iter().enumerate() {
            cached_rank[observation_index] = rank;
        }
        for (stability_index, top_k) in MAX6_SCORE_REUSE_AUDIT_TOP_K.iter().copied().enumerate() {
            let compared_top_k = top_k.min(window_observations.len());
            let mut symmetric_difference = 0usize;
            for observation_index in 0..window_observations.len() {
                symmetric_difference += usize::from(
                    (actual_rank[observation_index] < compared_top_k)
                        != (cached_rank[observation_index] < compared_top_k),
                );
            }
            let order_changed = actual_order[..compared_top_k] != cached_order[..compared_top_k];
            let stability = &mut summary.topk_stability[stability_index];
            stability.windows_compared += 1;
            stability.windows_with_membership_change += usize::from(symmetric_difference > 0);
            stability.total_membership_symmetric_difference += symmetric_difference;
            stability.max_membership_symmetric_difference = stability
                .max_membership_symmetric_difference
                .max(symmetric_difference);
            stability.windows_with_order_change += usize::from(order_changed);
        }
        window_start = window_end;
    }
    summary
}

fn max6_channel_audit_id(channel: &str) -> Result<u8, String> {
    match channel {
        "AND00" => Ok(0),
        "AND01" => Ok(1),
        "AND10" => Ok(2),
        "AND11" => Ok(3),
        "XOR" => Ok(4),
        "DOSAGE_PRODUCT" => Ok(5),
        "NONE" => Ok(6),
        _ => Err(format!(
            "unknown Max6 channel in score reuse audit: {channel}"
        )),
    }
}

fn max6_diagnostic_hash_u64(mut state: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        state ^= byte as u64;
        state = state.wrapping_mul(MAX6_DIAGNOSTIC_FNV_PRIME);
    }
    state
}

fn max6_diagnostic_pair_key(left: usize, right: usize) -> Result<u64, String> {
    if left == right {
        return Err("Max6 diagnostic pair hash received a self pair".to_string());
    }
    let (left, right) = if left < right {
        (left, right)
    } else {
        (right, left)
    };
    let left = u32::try_from(left)
        .map_err(|_| format!("Max6 diagnostic marker index {left} exceeds u32 range"))?;
    let right = u32::try_from(right)
        .map_err(|_| format!("Max6 diagnostic marker index {right} exceeds u32 range"))?;
    Ok(((left as u64) << 32) | right as u64)
}

fn finalize_max6_candidate_reuse_diagnostic(
    timing: &mut Max6ObservedScanTiming,
    pair_keys: &mut Vec<u64>,
    window_pair_counts: &mut [usize],
    window_screen_pairs_evaluated: &mut [usize],
    window_endpoint_counts: &mut [usize],
    marker_count: usize,
) -> Result<(), String> {
    let finalize_started = Instant::now();
    if pair_keys.len() != timing.scored_candidate_pairs {
        return Err(format!(
            "Max6 candidate reuse diagnostic collected {} pair keys but Max6/GRM-INT scoring processed {} pairs",
            pair_keys.len(),
            timing.scored_candidate_pairs
        ));
    }
    if window_pair_counts.len() != timing.units_timed
        || window_screen_pairs_evaluated.len() != timing.units_timed
        || window_endpoint_counts.len() != timing.units_timed
    {
        return Err(
            "Max6 candidate reuse diagnostic window-count reconciliation failed".to_string(),
        );
    }

    pair_keys.sort_unstable();
    let mut multiplicity_histogram = vec![0usize; timing.units_timed.saturating_add(1).max(2)];
    let mut endpoint_bits = vec![0u64; marker_count.div_ceil(64)];
    let mut unique_pairs = 0usize;
    let mut index = 0usize;
    while index < pair_keys.len() {
        let key = pair_keys[index];
        let mut end = index + 1;
        while end < pair_keys.len() && pair_keys[end] == key {
            end += 1;
        }
        let multiplicity = end - index;
        if multiplicity >= multiplicity_histogram.len() {
            return Err(format!(
                "Max6 candidate pair multiplicity {multiplicity} exceeds scanned window count {}",
                timing.units_timed
            ));
        }
        multiplicity_histogram[multiplicity] =
            multiplicity_histogram[multiplicity].saturating_add(1);
        unique_pairs = unique_pairs.saturating_add(1);

        let left = (key >> 32) as usize;
        let right = (key & u32::MAX as u64) as usize;
        if left >= marker_count || right >= marker_count || left >= right {
            return Err(format!(
                "Max6 candidate reuse diagnostic found invalid canonical pair ({left}, {right})"
            ));
        }
        endpoint_bits[left >> 6] |= 1u64 << (left & 63);
        endpoint_bits[right >> 6] |= 1u64 << (right & 63);
        index = end;
    }

    window_pair_counts.sort_unstable();
    window_screen_pairs_evaluated.sort_unstable();
    window_endpoint_counts.sort_unstable();
    let duplicate_occurrences = pair_keys.len().saturating_sub(unique_pairs);
    let unique_endpoints = endpoint_bits
        .iter()
        .map(|word| word.count_ones() as usize)
        .sum::<usize>();
    timing.candidate_reuse_diagnostic_enabled = true;
    timing.candidate_unique_pairs_global = unique_pairs;
    timing.candidate_duplicate_occurrences = duplicate_occurrences;
    timing.candidate_duplicate_fraction = if pair_keys.is_empty() {
        0.0
    } else {
        duplicate_occurrences as f64 / pair_keys.len() as f64
    };
    timing.candidate_multiplicity_p50 =
        max6_candidate_multiplicity_quantile(&multiplicity_histogram, unique_pairs, 0.50);
    timing.candidate_multiplicity_p90 =
        max6_candidate_multiplicity_quantile(&multiplicity_histogram, unique_pairs, 0.90);
    timing.candidate_multiplicity_p99 =
        max6_candidate_multiplicity_quantile(&multiplicity_histogram, unique_pairs, 0.99);
    timing.candidate_multiplicity_max = multiplicity_histogram
        .iter()
        .rposition(|&count| count > 0)
        .unwrap_or(0);
    timing.candidate_unique_endpoints_window_sum =
        window_endpoint_counts.iter().copied().sum::<usize>();
    timing.candidate_unique_endpoints_global = unique_endpoints;
    timing.candidate_endpoint_reuse_factor = if unique_endpoints == 0 {
        0.0
    } else {
        timing.candidate_unique_endpoints_window_sum as f64 / unique_endpoints as f64
    };
    timing.candidate_key_buffer_bytes = pair_keys.capacity() * std::mem::size_of::<u64>();
    timing.candidate_window_pairs_min = window_pair_counts.first().copied().unwrap_or(0);
    timing.candidate_window_pairs_p50 = max6_usize_quantile_sorted(window_pair_counts, 0.50);
    timing.candidate_window_pairs_p90 = max6_usize_quantile_sorted(window_pair_counts, 0.90);
    timing.candidate_window_pairs_p99 = max6_usize_quantile_sorted(window_pair_counts, 0.99);
    timing.candidate_window_pairs_max = window_pair_counts.last().copied().unwrap_or(0);
    timing.candidate_window_screen_pairs_evaluated_min =
        window_screen_pairs_evaluated.first().copied().unwrap_or(0);
    timing.candidate_window_screen_pairs_evaluated_p50 =
        max6_usize_quantile_sorted(window_screen_pairs_evaluated, 0.50);
    timing.candidate_window_screen_pairs_evaluated_p90 =
        max6_usize_quantile_sorted(window_screen_pairs_evaluated, 0.90);
    timing.candidate_window_screen_pairs_evaluated_p99 =
        max6_usize_quantile_sorted(window_screen_pairs_evaluated, 0.99);
    timing.candidate_window_screen_pairs_evaluated_max =
        window_screen_pairs_evaluated.last().copied().unwrap_or(0);
    timing.candidate_window_endpoints_min = window_endpoint_counts.first().copied().unwrap_or(0);
    timing.candidate_window_endpoints_p50 =
        max6_usize_quantile_sorted(window_endpoint_counts, 0.50);
    timing.candidate_window_endpoints_p90 =
        max6_usize_quantile_sorted(window_endpoint_counts, 0.90);
    timing.candidate_window_endpoints_p99 =
        max6_usize_quantile_sorted(window_endpoint_counts, 0.99);
    timing.candidate_window_endpoints_max = window_endpoint_counts.last().copied().unwrap_or(0);
    timing.candidate_key_finalize_wall_seconds = finalize_started.elapsed().as_secs_f64();
    Ok(())
}

#[derive(Clone, Debug)]
struct Max6ScreenCalibration {
    maxima: Vec<f64>,
    maximizing_pairs: Option<Vec<Option<(usize, usize)>>>,
    empirical_threshold: f64,
    gumbel_threshold: Option<f64>,
    pair_universe_size: usize,
}

fn build_max6_active_margin_rows<'a>(
    prepared_active_rows: Option<&'a [usize]>,
    scan_unit_indices: &[usize],
    units: &[GarfieldLogicUnit],
) -> Cow<'a, [usize]> {
    if let Some(rows) = prepared_active_rows {
        return Cow::Borrowed(rows);
    }

    let mut rows = scan_unit_indices
        .iter()
        .filter_map(|&unit_index| units.get(unit_index))
        .flat_map(|unit| unit.indices.iter().copied())
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    Cow::Owned(rows)
}

#[allow(clippy::too_many_arguments)]
fn slice_square_matrix_by_indices(
    matrix: &[f64],
    n: usize,
    indices: &[usize],
    ctx: &str,
) -> Result<Vec<f64>, String> {
    if matrix.len() != n.saturating_mul(n) {
        return Err(format!(
            "{ctx}: square-matrix payload length mismatch: got {}, expected {}",
            matrix.len(),
            n.saturating_mul(n)
        ));
    }
    let k = indices.len();
    let mut out = vec![0.0_f64; k.saturating_mul(k)];
    for (i_out, &i_src) in indices.iter().enumerate() {
        if i_src >= n {
            return Err(format!(
                "{ctx}: row index {} out of range for n={}",
                i_src, n
            ));
        }
        for (j_out, &j_src) in indices.iter().enumerate() {
            if j_src >= n {
                return Err(format!(
                    "{ctx}: col index {} out of range for n={}",
                    j_src, n
                ));
            }
            out[i_out * k + j_out] = matrix[i_src * n + j_src];
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn validate_max6_audit_output_prefix_is_external(prefix: &str) -> Result<(), String> {
    let output_path = Path::new(prefix);
    let output_parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(output_parent).map_err(|error| {
        format!(
            "Max6 audit output parent '{}' must exist and be resolvable: {error}",
            output_parent.display()
        )
    })?;
    let project_root = fs::canonicalize(env!("CARGO_MANIFEST_DIR")).map_err(|error| {
        format!("unable to resolve JanusX project root for audit-output guard: {error}")
    })?;
    if canonical_parent.starts_with(project_root.as_path()) {
        return Err(format!(
            "Max6 audit output must be outside the JanusX project; prefix '{}' resolves under '{}'",
            prefix,
            project_root.display()
        ));
    }
    Ok(())
}

fn read_single_trait_pheno_aligned(
    pheno_path: &str,
    ordered_iids: &[String],
) -> Result<Vec<f64>, String> {
    let file = File::open(pheno_path).map_err(|e| format!("open phenotype failed: {e}"))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .ok_or_else(|| "phenotype file is empty".to_string())?
        .map_err(|e| format!("read phenotype header failed: {e}"))?;
    let cols = header.split_whitespace().collect::<Vec<_>>();
    if cols.len() < 2 {
        return Err("phenotype file must contain at least IID and one trait column".to_string());
    }
    let iid_idx = cols
        .iter()
        .position(|c| c.eq_ignore_ascii_case("IID"))
        .unwrap_or(0usize);
    let trait_idx = (0..cols.len())
        .find(|&idx| idx != iid_idx)
        .ok_or_else(|| "phenotype file has no trait column".to_string())?;
    let mut by_iid = HashMap::<String, f64>::with_capacity(ordered_iids.len());
    for (line_no, line_res) in lines.enumerate() {
        let line = line_res.map_err(|e| format!("read phenotype line failed: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields = trimmed.split_whitespace().collect::<Vec<_>>();
        let need = iid_idx.max(trait_idx);
        if fields.len() <= need {
            return Err(format!(
                "{pheno_path}:{}: expected at least {} columns, got {}",
                line_no + 2,
                need + 1,
                fields.len()
            ));
        }
        let iid = fields[iid_idx].to_string();
        let value = fields[trait_idx].parse::<f64>().map_err(|e| {
            format!(
                "{pheno_path}:{}: invalid phenotype value '{}': {e}",
                line_no + 2,
                fields[trait_idx]
            )
        })?;
        if !value.is_finite() {
            return Err(format!(
                "{pheno_path}:{}: phenotype value for IID '{}' is not finite",
                line_no + 2,
                iid
            ));
        }
        by_iid.insert(iid, value);
    }
    ordered_iids
        .iter()
        .map(|iid| {
            by_iid
                .get(iid)
                .copied()
                .ok_or_else(|| format!("phenotype missing IID '{iid}' from FAM order"))
        })
        .collect()
}

#[cfg(test)]
fn dense_candidate_endpoint_matrix(
    bits_flat: &[u64],
    row_words: usize,
    n_samples: usize,
    selected_global_rows: &[usize],
    candidate_pairs: &[(usize, usize)],
) -> Result<(Vec<f64>, Vec<usize>, Vec<usize>, Vec<usize>), String> {
    let global_pairs = candidate_pairs
        .iter()
        .map(|&(left, right)| {
            let left_global = selected_global_rows
                .get(left)
                .copied()
                .ok_or_else(|| format!("candidate endpoint index out of range: {left}"))?;
            let right_global = selected_global_rows
                .get(right)
                .copied()
                .ok_or_else(|| format!("candidate endpoint index out of range: {right}"))?;
            Ok((left_global, right_global))
        })
        .collect::<Result<Vec<_>, String>>()?;
    dense_candidate_endpoint_matrix_global(bits_flat, row_words, n_samples, &global_pairs)
}

#[cfg(test)]
#[path = "garfield_tests.rs"]
mod tests;
