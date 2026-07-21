pub(crate) mod bs;
mod permutation;
mod residual;
mod sampling;
mod score;
mod score_gpu;

use self::bs::{beam_search_and_binary_mcc, beam_search_and_continuous_abs_corr, BeamAndResult};
use self::bs::{
    beam_search_train_test_continuous_fuzzy, beam_search_train_test_continuous_with_literal_scores,
    evaluate_rule_continuous_dual, materialize_rule_bits_dual,
    precompute_literal_singleton_scores_batched, LiteralScoreBatchRequest, LiteralSingletonScore,
};
use self::bs::{reset_garfield_beam_profile, snapshot_garfield_beam_profile};
use self::permutation::{
    bucket_from_rule, choose_representative_indices, null_topk_per_repeat_for_bucket,
    shuffled_copy_f64, RuleNullBucket, RuleNullCalibrator, RuleNullPenaltyLookup,
    RuleStructurePrior, RuleStructurePriorCalibrator, RuleStructurePriorConfig,
    DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS, DEFAULT_RULE_NULL_ADAPTIVE_STABLE_REPEATS,
    DEFAULT_RULE_NULL_MAX_REPEATS, DEFAULT_RULE_NULL_MIN_SNPS_PER_CHUNK,
    DEFAULT_RULE_NULL_PHYSICAL_CHUNKS, DEFAULT_RULE_PERMUTATION_REPRESENTATIVE_UNITS,
    DEFAULT_RULE_STRUCTURE_BOOTSTRAP_KL_THRESHOLD, DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MAX_REPEATS,
    DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MIN_REPEATS, DEFAULT_RULE_STRUCTURE_BOOTSTRAP_STABLE_REPEATS,
    DEFAULT_RULE_STRUCTURE_DENSITY_TOPK,
};
use crate::bincore::{
    append_suffix, bin01_header_bytes, bin01_id_sidecar_path, bin_prefix, build_windows_from_sites,
    discover_id_sidecar, discover_site_sidecar, parse_bin01_header, resolve_bin01_path, BinWindow,
};
use crate::binwriter::{Bin01SiteMode, Bin01SiteRecordRef, Bin01Writer};
use crate::bitwise::{and_popcount, bitand_assign, bitnot_masked, popcount};
use crate::breader::{
    gather_rows_by_indices, gather_rows_by_range, load_bin01_as_u64_words,
    load_bin01_packed_payload_owned, load_bin01_selected_rows_as_u64_words,
};
use crate::bstats::{apply_tail_mask, tail_mask, words_for_samples};
use crate::gblup::{build_grm_from_meta_stream, StreamKernelMode};
use crate::gfcore::{
    process_snp_row, read_fam, BedSnpIter, HmpSnpIter, SiteInfo, TxtSnpIter, VcfSnpIter,
};
use crate::gfreader::{
    build_sample_selection, compute_bed_row_meta_owned_for_source_rows,
    prepare_bed_logic_meta_owned_for_stats_samples_pure_line,
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
use crate::ml::univariate::{
    feature_scores_abs_corr_stage1, feature_scores_abs_corr_stage1_with_parallel,
};
use crate::stats_common::{
    arm_interrupt_trap, check_ctrlc, env_truthy, map_err_string_to_py, process_memory_usage,
};

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
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

#[allow(unused_imports)]
pub use bs::{
    beam_search_train_test_continuous, evaluate_rule_continuous, materialize_rule_bits,
    rank_rule_score_components, rank_rule_score_components_with_bucket, BeamBinaryOp,
    BeamGroupConstraintMode, BeamLiteral, BeamRankMode, BeamRule, BeamRuleCandidate,
    BeamSearchParams,
};
pub use residual::{garfield_residualize_bed_py, garfield_residualize_grm_py};
use residual::{garfield_residualize_exact_from_grm_rust, GarfieldResidualResult};
#[allow(unused_imports)]
pub use sampling::stratified_test_mask;
#[allow(unused_imports)]
pub use score::{
    dual_packed_summary, score_binary_ba_mcc_batch_py, score_binary_ba_py, score_binary_mcc_packed,
    score_binary_mcc_py, score_cont_centered_gain_packed_with_sum, score_cont_corr_packed,
    score_cont_corr_py, score_cont_mean_diff_corr_batch_py, score_cont_mean_diff_py,
    score_cont_weighted_mean_diff_packed, support_size_packed, ContinuousRuleScore,
};
pub use score_gpu::{
    garfield_compare_score_cont_centered_gain_batch_metal_vs_cpu_py,
    garfield_compare_score_cont_centered_gain_singleton_backends_py,
    garfield_metal_runtime_status_py, garfield_score_cont_centered_gain_batch_packed_cpu_py,
    garfield_score_cont_centered_gain_batch_packed_metal_py,
};

const GARFIELD_CONSTRAINED_BEAM_PAR_MIN_TOTAL_CANDS: usize = 1_024;
const GARFIELD_CONSTRAINED_BEAM_PAR_CHUNK_CANDS: usize = 256;
const GARFIELD_NULL_ML_TOP_FRAC: f64 = 0.80;
const GARFIELD_SCAN_PAIR_PARENT_ABS_GAIN_MIN: f64 = 1e-6;
const GARFIELD_SCAN_SURROGATE_TEST_GAIN_MAX: f64 = 0.02;
const GARFIELD_SCAN_SURROGATE_HAMMING_FRAC_MAX: f64 = 0.02;
const GARFIELD_DISABLE_STRUCTURE_PRIOR: bool = true;
const GARFIELD_GENESET_LD_PRUNE_R2_DEFAULT: f64 = 0.80;
const GARFIELD_GENESET_CORR_SERIAL_MAX_ROWS: usize = 256;
const GARFIELD_GENESET_CORR_PRESCREEN_SLACK_MIN: usize = 8;
const GARFIELD_GENESET_CORR_PRESCREEN_SLACK_MAX: usize = 32;
static GARFIELD_ML_SELECT_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_DENSE_DOSAGE_DECODE_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_GENESET_LD_PRUNE_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_GENESET_LD_EXACT_PAIRS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_GENESET_LD_ROWS_TOTAL: AtomicU64 = AtomicU64::new(0);
static GARFIELD_GENESET_LD_ROWS_KEPT: AtomicU64 = AtomicU64::new(0);
static GARFIELD_MATERIALIZE_BITS_NS: AtomicU64 = AtomicU64::new(0);
static GARFIELD_GENESET_LD_SUPPORT_CACHE: OnceLock<
    Mutex<HashMap<(usize, u64), Arc<GarfieldGenesetLdSupportCacheEntry>>>,
> = OnceLock::new();
static GARFIELD_GENESET_LD_UNIT_STATS: OnceLock<Mutex<GarfieldGenesetLdUnitStatsCollector>> =
    OnceLock::new();

const GARFIELD_STRUCTURE_TASK_COALESCE_MAX_UNITS_DEFAULT: usize = 32;
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

#[derive(Default, Clone, Debug)]
struct GarfieldGenesetLdUnitStatsCollector {
    units_eligible: u64,
    units_pruned: u64,
    rows_pruned: u64,
    rows_total: Vec<u32>,
    rows_kept: Vec<u32>,
    rows_pruned_dist: Vec<u32>,
    exact_pairs: Vec<u32>,
}

#[derive(Default, Clone, Debug)]
struct GarfieldGenesetLdUnitStatsSnapshot {
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

const GARFIELD_GENESET_LD_NO_BOUND: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
struct GarfieldGenesetLdExactBounds {
    low_max: u32,
    high_min: u32,
}

impl Default for GarfieldGenesetLdExactBounds {
    fn default() -> Self {
        Self {
            low_max: GARFIELD_GENESET_LD_NO_BOUND,
            high_min: GARFIELD_GENESET_LD_NO_BOUND,
        }
    }
}

impl GarfieldGenesetLdExactBounds {
    #[inline]
    fn matches(self, both_one: usize) -> bool {
        let both_one = u32::try_from(both_one).unwrap_or(u32::MAX);
        (self.low_max != GARFIELD_GENESET_LD_NO_BOUND && both_one <= self.low_max)
            || (self.high_min != GARFIELD_GENESET_LD_NO_BOUND && both_one >= self.high_min)
    }
}

#[derive(Clone, Debug)]
struct GarfieldGenesetLdSupportCacheEntry {
    conflict_supports: Vec<Vec<usize>>,
    exact_bounds: Vec<Vec<GarfieldGenesetLdExactBounds>>,
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

fn garfield_geneset_ld_unit_stats_reset() {
    let store = GARFIELD_GENESET_LD_UNIT_STATS
        .get_or_init(|| Mutex::new(GarfieldGenesetLdUnitStatsCollector::default()));
    if let Ok(mut guard) = store.lock() {
        *guard = GarfieldGenesetLdUnitStatsCollector::default();
    }
}

fn garfield_geneset_ld_unit_stats_record(rows_total: usize, rows_kept: usize, exact_pairs: u64) {
    let rows_pruned = rows_total.saturating_sub(rows_kept);
    let store = GARFIELD_GENESET_LD_UNIT_STATS
        .get_or_init(|| Mutex::new(GarfieldGenesetLdUnitStatsCollector::default()));
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

fn garfield_geneset_ld_unit_stats_snapshot() -> GarfieldGenesetLdUnitStatsSnapshot {
    let store = GARFIELD_GENESET_LD_UNIT_STATS
        .get_or_init(|| Mutex::new(GarfieldGenesetLdUnitStatsCollector::default()));
    let guard = match store.lock() {
        Ok(v) => v,
        Err(_) => return GarfieldGenesetLdUnitStatsSnapshot::default(),
    };
    let (rows_total_median, rows_total_p95, rows_total_max) =
        summarize_u32_distribution(guard.rows_total.as_slice());
    let (rows_kept_median, rows_kept_p95, rows_kept_max) =
        summarize_u32_distribution(guard.rows_kept.as_slice());
    let (rows_pruned_median, rows_pruned_p95, rows_pruned_max) =
        summarize_u32_distribution(guard.rows_pruned_dist.as_slice());
    let (exact_pairs_median, exact_pairs_p95, exact_pairs_max) =
        summarize_u32_distribution(guard.exact_pairs.as_slice());
    GarfieldGenesetLdUnitStatsSnapshot {
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
fn garfield_task_chunk_units(total_units: usize, threads: usize, max_units: usize) -> usize {
    let total_units = total_units.max(1);
    let max_units = max_units.max(1);
    if threads <= 1 {
        return total_units.min(max_units);
    }
    let target_tasks = threads.saturating_mul(4).max(1);
    total_units.div_ceil(target_tasks).clamp(1, max_units)
}

#[inline]
fn garfield_logic_mmap_window_mb() -> usize {
    parse_env_mb_usize("JX_GARFIELD_LOGIC_MMAP_WINDOW_MB")
        .or_else(|| parse_env_mb_usize("JX_BED_BLOCK_TARGET_MB"))
        .unwrap_or(GARFIELD_LOGIC_MMAP_WINDOW_MB_DEFAULT)
        .max(1)
}

#[inline]
fn garfield_geneset_ld_prune_r2() -> f64 {
    parse_env_f64("JX_GARFIELD_GENESET_LD_PRUNE_R2")
        .filter(|v| *v > 0.0 && *v <= 1.0)
        .unwrap_or(GARFIELD_GENESET_LD_PRUNE_R2_DEFAULT)
}

#[inline]
fn garfield_rss_debug_enabled() -> bool {
    env_truthy("JX_GARFIELD_RSS_DEBUG")
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

#[derive(Clone, Debug, Default)]
struct GarfieldMemoryDebugSummary {
    global_bits_loaded: GarfieldStageMemoryDebug,
    scan: GarfieldStageMemoryDebug,
    null_penalty: GarfieldStageMemoryDebug,
    structure_prior: GarfieldStageMemoryDebug,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldInputKind {
    Auto,
    Bfile,
    Vcf,
    Hmp,
    File,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldBinMode {
    Bin,
    Mbin,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GarfieldMlKeepPolicy {
    TopK,
    TopKPlusRandom { top_frac: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldResponse {
    Binary,
    Continuous,
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

#[derive(Clone, Debug)]
struct ConstrainedBeamNode {
    selected: Vec<usize>,
    combined: Vec<u64>,
    score: f64,
    last_index: usize,
}

#[inline]
fn parse_response(response: &str) -> Result<GarfieldResponse, String> {
    let t = response.trim().to_ascii_lowercase();
    match t.as_str() {
        "binary" | "bin" | "b" => Ok(GarfieldResponse::Binary),
        "continuous" | "cont" | "c" => Ok(GarfieldResponse::Continuous),
        _ => Err("response must be 'binary' or 'continuous'".to_string()),
    }
}

#[inline]
fn parse_input_kind(input_kind: &str) -> Result<GarfieldInputKind, String> {
    let t = input_kind.trim().to_ascii_lowercase();
    match t.as_str() {
        "" | "auto" => Ok(GarfieldInputKind::Auto),
        "bfile" | "plink" | "bed" => Ok(GarfieldInputKind::Bfile),
        "vcf" => Ok(GarfieldInputKind::Vcf),
        "hmp" | "hapmap" => Ok(GarfieldInputKind::Hmp),
        "file" | "txt" | "npy" | "bin" => Ok(GarfieldInputKind::File),
        _ => Err("input_kind must be one of: auto, bfile, vcf, hmp, file".to_string()),
    }
}

#[inline]
fn parse_bin_mode(mode: &str) -> Result<GarfieldBinMode, String> {
    let t = mode.trim().to_ascii_lowercase();
    match t.as_str() {
        "bin" | "bin02" => Ok(GarfieldBinMode::Bin),
        "mbin" => Ok(GarfieldBinMode::Mbin),
        _ => Err("mode must be one of: bin, mbin".to_string()),
    }
}

#[inline]
fn parse_beam_rank_mode(mode: &str) -> Result<BeamRankMode, String> {
    let t = mode.trim().to_ascii_lowercase();
    if let Some(rest) = t.strip_prefix("gain_from_layer:") {
        let start = rest
            .parse::<usize>()
            .map_err(|_| "rank_score gain_from_layer:<N> requires integer N >= 1".to_string())?;
        if start < 1 {
            return Err("rank_score gain_from_layer:<N> requires integer N >= 1".to_string());
        }
        return Ok(BeamRankMode::GainFromLayer(start));
    }
    match t.as_str() {
        "" | "interaction_gain" | "gain" | "interaction-gain" | "interactiongain" => {
            Ok(BeamRankMode::InteractionGain)
        }
        "exhaustive_then_gain"
        | "exhaustive-then-gain"
        | "exhaustivethengain"
        | "staged_gain"
        | "staged-gain"
        | "stagedgain"
        | "beam_gain"
        | "beam-gain"
        | "beamgain" => Ok(BeamRankMode::ExhaustiveThenGain),
        "raw" | "score" => Ok(BeamRankMode::Raw),
        _ => Err(
            "rank_score must be one of: raw, interaction_gain, exhaustive_then_gain, gain_from_layer:<N>"
                .to_string(),
        ),
    }
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

fn load_bin01_sites(path: &str, n_rows: usize) -> Result<Option<Vec<SiteInfo>>, String> {
    let Some(site_path) = discover_site_sidecar(path) else {
        return Ok(None);
    };
    let sites = crate::gfcore::read_site_file(&site_path)?;
    if sites.len() < n_rows {
        return Err(format!(
            "BIN sidecar site count mismatch: {} < n_rows={}",
            sites.len(),
            n_rows
        ));
    }
    Ok(Some(sites))
}

fn compute_bin01_group_ids(path: &str, n_rows: usize) -> Result<Vec<u64>, String> {
    let Some(sites) = load_bin01_sites(path, n_rows)? else {
        return Ok((0..n_rows).map(|i| i as u64).collect());
    };
    Ok(build_feature_group_ids(&sites, n_rows)
        .into_iter()
        .map(|g| g as u64)
        .collect())
}

fn load_bin01_packed_owned(
    path: &str,
    require_grouped_rows: bool,
) -> Result<(Vec<u8>, Vec<u64>, usize, usize, usize), String> {
    let ctx = if require_grouped_rows {
        "garfield::load_mbin_packed"
    } else {
        "garfield::load_bin01_packed"
    };
    let resolved = resolve_bin01_path(path)?;
    let resolved_str = resolved.to_string_lossy().to_string();
    let (packed, n_rows, n_samples, row_bytes) =
        load_bin01_packed_payload_owned(&resolved_str, ctx)?;
    let group_ids = compute_bin01_group_ids(&resolved_str, n_rows)?;
    if require_grouped_rows {
        let mut seen: HashSet<u64> = HashSet::with_capacity(group_ids.len());
        let mut has_duplicate_group = false;
        for &gid in group_ids.iter() {
            if !seen.insert(gid) {
                has_duplicate_group = true;
                break;
            }
        }
        if !has_duplicate_group {
            return Err(format!(
                "{ctx}: no repeated feature groups detected; this looks like a BIN cache, not MBIN"
            ));
        }
    }
    Ok((packed, group_ids, n_samples, n_rows, row_bytes))
}

fn copy_site_sidecar(src_bin: &str, dst_bin: &str) -> Result<(), String> {
    let Some(src) = discover_site_sidecar(src_bin) else {
        return Ok(());
    };
    let src_name = src
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid sidecar filename".to_string())?;
    let src_prefix = bin_prefix(src_bin);
    let dst_prefix = bin_prefix(dst_bin);
    let src_prefix_name = src_prefix
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid src prefix".to_string())?;

    let suffix = src_name
        .strip_prefix(src_prefix_name)
        .unwrap_or("")
        .to_string();
    if suffix.is_empty() {
        return Ok(());
    }
    let dst = append_suffix(&dst_prefix, &suffix);
    fs::copy(&src, &dst)
        .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;
    Ok(())
}

fn copy_id_sidecar(src_bin: &str, dst_bin: &str) -> Result<(), String> {
    let Some(src) = discover_id_sidecar(src_bin) else {
        return Ok(());
    };
    let src_name = src
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid id-sidecar filename".to_string())?;
    let src_prefix = bin_prefix(src_bin);
    let dst_prefix = bin_prefix(dst_bin);
    let src_prefix_name = src_prefix
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid src prefix".to_string())?;

    let suffix = src_name
        .strip_prefix(src_prefix_name)
        .unwrap_or("")
        .to_string();
    if suffix.is_empty() {
        return Ok(());
    }
    let dst = append_suffix(&dst_prefix, &suffix);
    fs::copy(&src, &dst)
        .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;
    Ok(())
}

fn read_sample_ids_from_sidecar(path: &Path) -> Result<Vec<String>, String> {
    let fr = BufReader::new(File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
    let is_fam = path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.eq_ignore_ascii_case("fam"))
        .unwrap_or(false);
    let mut out: Vec<String> = Vec::new();
    for (ln, line) in fr.lines().enumerate() {
        let raw = line.map_err(|e| format!("read {}:{}: {e}", path.display(), ln + 1))?;
        let s = raw.trim();
        if s.is_empty() {
            continue;
        }
        if is_fam {
            let toks: Vec<&str> = s.split_whitespace().collect();
            if toks.len() < 2 {
                return Err(format!(
                    "{}:{} malformed FAM row (need >=2 columns)",
                    path.display(),
                    ln + 1
                ));
            }
            out.push(toks[1].to_string());
        } else {
            let toks: Vec<&str> = s.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            out.push(toks[0].to_string());
        }
    }
    Ok(out)
}

fn write_subset_id_sidecar(
    src_bin: &str,
    dst_bin: &str,
    sample_indices: &[usize],
) -> Result<bool, String> {
    let Some(src_id_path) = discover_id_sidecar(src_bin) else {
        return Ok(false);
    };
    let src_ids = read_sample_ids_from_sidecar(&src_id_path)?;
    let dst_id_path = bin01_id_sidecar_path(dst_bin);
    if let Some(parent) = dst_id_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let mut fw = BufWriter::new(
        File::create(&dst_id_path).map_err(|e| format!("create {}: {e}", dst_id_path.display()))?,
    );
    for &si in sample_indices.iter() {
        if si >= src_ids.len() {
            return Err(format!(
                "sample index out of range while subsetting IDs: {} >= {}",
                si,
                src_ids.len()
            ));
        }
        fw.write_all(src_ids[si].as_bytes())
            .map_err(|e| format!("write {}: {e}", dst_id_path.display()))?;
        fw.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", dst_id_path.display()))?;
    }
    fw.flush()
        .map_err(|e| format!("flush {}: {e}", dst_id_path.display()))?;
    Ok(true)
}

#[inline]
fn normalize_plink_prefix(path_or_prefix: &str) -> String {
    let s = path_or_prefix.trim();
    let low = s.to_ascii_lowercase();
    if low.ends_with(".bed") || low.ends_with(".bim") || low.ends_with(".fam") {
        s[..s.len() - 4].to_string()
    } else {
        s.to_string()
    }
}

fn make_input_reader(
    input_path: &str,
    input_kind: GarfieldInputKind,
) -> Result<GarfieldInputReader, String> {
    let p = input_path.trim();
    if p.is_empty() {
        return Err("input_path must not be empty".to_string());
    }
    let low = p.to_ascii_lowercase();
    match input_kind {
        GarfieldInputKind::Bfile => {
            let prefix = normalize_plink_prefix(p);
            let it = BedSnpIter::new_with_fill(&prefix, 0.0, 1.0, false, false, 0.02)?;
            Ok(GarfieldInputReader::Bed(it))
        }
        GarfieldInputKind::Vcf => {
            let it = VcfSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
            Ok(GarfieldInputReader::Vcf(it))
        }
        GarfieldInputKind::Hmp => {
            let it = HmpSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
            Ok(GarfieldInputReader::Hmp(it))
        }
        GarfieldInputKind::File => {
            let it = TxtSnpIter::new(p, None)?;
            Ok(GarfieldInputReader::Txt(it))
        }
        GarfieldInputKind::Auto => {
            if low.ends_with(".vcf") || low.ends_with(".vcf.gz") {
                let it = VcfSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
                return Ok(GarfieldInputReader::Vcf(it));
            }
            if low.ends_with(".hmp") || low.ends_with(".hmp.gz") {
                let it = HmpSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
                return Ok(GarfieldInputReader::Hmp(it));
            }
            let prefix = normalize_plink_prefix(p);
            let bed = format!("{prefix}.bed");
            let bim = format!("{prefix}.bim");
            let fam = format!("{prefix}.fam");
            if Path::new(&bed).exists() && Path::new(&bim).exists() && Path::new(&fam).exists() {
                let it = BedSnpIter::new_with_fill(&prefix, 0.0, 1.0, false, false, 0.02)?;
                return Ok(GarfieldInputReader::Bed(it));
            }
            let it = TxtSnpIter::new(p, None)?;
            Ok(GarfieldInputReader::Txt(it))
        }
    }
}

#[inline]
fn row_bytes_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(8).max(1)
}

#[inline]
fn normalize_genotype3(v: f32) -> Option<u8> {
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    let r = v.round();
    if r <= 0.0 {
        Some(0)
    } else if r >= 2.0 {
        Some(2)
    } else {
        Some(1)
    }
}

fn write_sample_id_sidecar(out_bin_path: &str, sample_ids: &[String]) -> Result<(), String> {
    let id_path = bin01_id_sidecar_path(out_bin_path);
    if let Some(parent) = id_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
    }
    let mut fw = BufWriter::new(
        File::create(&id_path).map_err(|e| format!("create {}: {e}", id_path.display()))?,
    );
    for sid in sample_ids.iter() {
        fw.write_all(sid.as_bytes())
            .map_err(|e| format!("write {}: {e}", id_path.display()))?;
        fw.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", id_path.display()))?;
    }
    fw.flush()
        .map_err(|e| format!("flush {}: {e}", id_path.display()))?;
    Ok(())
}

fn row_select_by_indices(row: Vec<f32>, sample_indices: &[usize]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(sample_indices.len());
    for &idx in sample_indices.iter() {
        if idx >= row.len() {
            return Err(format!(
                "sample index out of range while slicing row: {} >= {}",
                idx,
                row.len()
            ));
        }
        out.push(row[idx]);
    }
    Ok(out)
}

fn mbin_site_variants(site: &SiteInfo) -> [SiteInfo; 3] {
    let dom = SiteInfo {
        chrom: site.chrom.clone(),
        pos: site.pos,
        snp: format!("{}|DOM", site.snp),
        ref_allele: site.ref_allele.clone(),
        alt_allele: format!("{}|DOM", site.alt_allele),
    };
    let rec = SiteInfo {
        chrom: site.chrom.clone(),
        pos: site.pos,
        snp: format!("{}|REC", site.snp),
        ref_allele: site.ref_allele.clone(),
        alt_allele: format!("{}|REC", site.alt_allele),
    };
    let het = SiteInfo {
        chrom: site.chrom.clone(),
        pos: site.pos,
        snp: format!("{}|HET", site.snp),
        ref_allele: site.ref_allele.clone(),
        alt_allele: format!("{}|HET", site.alt_allele),
    };
    [dom, rec, het]
}

fn encode_row_to_bits(
    mut row: Vec<f32>,
    mut site: SiteInfo,
    mode: GarfieldBinMode,
    row_bytes: usize,
    maf: f32,
    geno: f32,
    impute: bool,
    het: f32,
) -> Option<Vec<EncodedRow>> {
    let keep = process_snp_row(
        &mut row,
        &mut site.ref_allele,
        &mut site.alt_allele,
        maf,
        geno,
        impute,
        false,
        het,
    );
    if !keep {
        return None;
    }
    if matches!(mode, GarfieldBinMode::Bin) {
        let mut non_missing = 0usize;
        let mut het_count = 0usize;
        for &v in row.iter() {
            if let Some(g) = normalize_genotype3(v) {
                non_missing += 1;
                if g == 1 {
                    het_count += 1;
                }
            }
        }
        if non_missing == 0 {
            return None;
        }
        let het_rate = het_count as f32 / non_missing as f32;
        if het_rate > het {
            return None;
        }
    }

    match mode {
        GarfieldBinMode::Bin => {
            let mut c0 = 0usize;
            let mut c2 = 0usize;
            for &v in row.iter() {
                match normalize_genotype3(v) {
                    Some(0) => c0 += 1,
                    Some(2) => c2 += 1,
                    _ => {}
                }
            }
            let mode02 = if c2 > c0 { 2u8 } else { 0u8 };
            let mut bits = vec![0u8; row_bytes];
            for (i, &v) in row.iter().enumerate() {
                let g = normalize_genotype3(v).unwrap_or(mode02);
                let is_one = if g == 0 {
                    false
                } else if g == 2 {
                    true
                } else {
                    mode02 == 2
                };
                if is_one {
                    bits[i >> 3] |= 1u8 << (i & 7);
                }
            }
            site.alt_allele = format!("{}|BIN", site.alt_allele);
            Some(vec![EncodedRow { site, bits }])
        }
        GarfieldBinMode::Mbin => {
            let mut dom = vec![0u8; row_bytes];
            let mut rec = vec![0u8; row_bytes];
            let mut het_bits = vec![0u8; row_bytes];
            for (i, &v) in row.iter().enumerate() {
                let Some(g) = normalize_genotype3(v) else {
                    continue;
                };
                let mask = 1u8 << (i & 7);
                if g > 0 {
                    dom[i >> 3] |= mask;
                }
                if g == 2 {
                    rec[i >> 3] |= mask;
                }
                if g == 1 {
                    het_bits[i >> 3] |= mask;
                }
            }
            let [s_dom, s_rec, s_het] = mbin_site_variants(&site);
            Some(vec![
                EncodedRow {
                    site: s_dom,
                    bits: dom,
                },
                EncodedRow {
                    site: s_rec,
                    bits: rec,
                },
                EncodedRow {
                    site: s_het,
                    bits: het_bits,
                },
            ])
        }
    }
}

fn encode_batch_rows(
    batch: Vec<(Vec<f32>, SiteInfo)>,
    mode: GarfieldBinMode,
    row_bytes: usize,
    maf: f32,
    geno: f32,
    impute: bool,
    het: f32,
    pool: Option<&rayon::ThreadPool>,
) -> Vec<Option<Vec<EncodedRow>>> {
    if let Some(p) = pool {
        p.install(|| {
            batch
                .into_par_iter()
                .map(|(row, site)| {
                    encode_row_to_bits(row, site, mode, row_bytes, maf, geno, impute, het)
                })
                .collect::<Vec<_>>()
        })
    } else {
        batch
            .into_iter()
            .map(|(row, site)| {
                encode_row_to_bits(row, site, mode, row_bytes, maf, geno, impute, het)
            })
            .collect::<Vec<_>>()
    }
}

#[inline]
fn normalize_chrom(chrom: &str) -> String {
    let s = chrom.trim();
    if s.len() > 2 && (s.ends_with("_1") || s.ends_with("_2")) {
        return s[..s.len() - 2].to_string();
    }
    if s.len() > 1 && (s.ends_with('-') || s.ends_with('+')) {
        return s[..s.len() - 1].to_string();
    }
    s.to_string()
}

trait GarfieldChromPosSite {
    fn garfield_chrom(&self) -> &str;
    fn garfield_pos(&self) -> i32;
    fn garfield_snp_name(&self) -> &str;
}

trait GarfieldDisplaySite: GarfieldChromPosSite {
    fn garfield_model_label(&self) -> &'static str;
    fn garfield_ref_allele(&self) -> &str;
    fn garfield_alt_allele(&self) -> &str;
}

#[inline]
fn chrom_base_view(chrom: &str) -> &str {
    let s = chrom.trim();
    let s = if s.len() > 2 && (s.ends_with("_1") || s.ends_with("_2")) {
        &s[..s.len() - 2]
    } else {
        s
    };
    let s = if s.len() > 1 && (s.ends_with('-') || s.ends_with('+')) {
        &s[..s.len() - 1]
    } else {
        s
    };
    if s.len() >= 3 && s[..3].eq_ignore_ascii_case("chr") {
        &s[3..]
    } else {
        s
    }
}

#[inline]
fn same_normalized_chrom(a: &str, b: &str) -> bool {
    chrom_base_view(a).eq_ignore_ascii_case(chrom_base_view(b))
}

#[inline]
fn normalize_interval_bounds(start: i32, end: i32) -> (i32, i32) {
    if start <= end {
        (start, end)
    } else {
        (end, start)
    }
}

fn parse_scan_bimrange_item(text: &str) -> Result<GarfieldScanBimRange, String> {
    let raw = text.trim();
    if raw.is_empty() {
        return Err("empty --bimrange item".to_string());
    }
    let (chrom_txt, coords_txt) = raw.split_once(':').ok_or_else(|| {
        format!("Invalid --bimrange format: {raw}. Use chr:start-end (or chr:start:end) in bp.")
    })?;
    let (start_txt, end_txt) = if let Some((a, b)) = coords_txt.split_once('-') {
        (a.trim(), b.trim())
    } else if let Some((a, b)) = coords_txt.split_once(':') {
        (a.trim(), b.trim())
    } else {
        return Err(format!(
            "Invalid --bimrange format: {raw}. Use chr:start-end (or chr:start:end) in bp."
        ));
    };
    let parse_bp = |label: &str, value: &str| -> Result<i32, String> {
        let parsed = value.parse::<i64>().map_err(|_| {
            format!("Invalid --bimrange {label}: '{value}' is not an integer bp coordinate.")
        })?;
        if !(0..=i64::from(i32::MAX)).contains(&parsed) {
            return Err(format!(
                "Invalid --bimrange {label}: '{value}' is outside the supported bp range [0, {}].",
                i32::MAX
            ));
        }
        Ok(parsed as i32)
    };
    let start = parse_bp("start", start_txt)?;
    let end = parse_bp("end", end_txt)?;
    let (bp_start, bp_end) = normalize_interval_bounds(start, end);
    Ok(GarfieldScanBimRange {
        chrom: normalize_chrom(chrom_txt),
        bp_start,
        bp_end,
    })
}

fn parse_scan_bimranges(items: &[String]) -> Result<Vec<GarfieldScanBimRange>, String> {
    let mut out = Vec::<GarfieldScanBimRange>::new();
    for raw in items.iter() {
        for item in raw.split(',') {
            let trimmed = item.trim();
            if trimmed.is_empty() {
                continue;
            }
            out.push(parse_scan_bimrange_item(trimmed)?);
        }
    }
    if out.is_empty() {
        return Err("--bimrange was provided but no valid interval was parsed.".to_string());
    }
    Ok(out)
}

#[inline]
fn spans_overlap_bp(a_start: i32, a_end: i32, b_start: i32, b_end: i32) -> bool {
    let (a_lo, a_hi) = normalize_interval_bounds(a_start, a_end);
    let (b_lo, b_hi) = normalize_interval_bounds(b_start, b_end);
    a_lo <= b_hi && b_lo <= a_hi
}

fn unit_overlaps_scan_bimranges(
    unit: &GarfieldLogicUnit,
    bimranges: &[GarfieldScanBimRange],
) -> bool {
    unit.spans.iter().any(|span| {
        bimranges.iter().any(|range| {
            same_normalized_chrom(&span.chrom, &range.chrom)
                && spans_overlap_bp(span.bp_start, span.bp_end, range.bp_start, range.bp_end)
        })
    })
}

#[inline]
fn chrom_code_for_chunk_id(chrom: &str, fallback_seq_code: u32) -> u32 {
    let base = chrom_base_view(chrom);
    if let Ok(v) = base.parse::<u32>() {
        return v;
    }
    if base.eq_ignore_ascii_case("X") {
        return 23;
    }
    if base.eq_ignore_ascii_case("Y") {
        return 24;
    }
    if base.eq_ignore_ascii_case("XY") {
        return 25;
    }
    if base.eq_ignore_ascii_case("M") || base.eq_ignore_ascii_case("MT") {
        return 26;
    }
    fallback_seq_code.max(1)
}

fn build_chrom_index<S: GarfieldChromPosSite>(
    sites: &[S],
    n_rows: usize,
) -> HashMap<String, Vec<(i32, usize)>> {
    let mut idx: HashMap<String, Vec<(i32, usize)>> = HashMap::new();
    for (i, s) in sites.iter().take(n_rows).enumerate() {
        idx.entry(normalize_chrom(s.garfield_chrom()))
            .or_default()
            .push((s.garfield_pos(), i));
    }
    for v in idx.values_mut() {
        v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    }
    idx
}

fn interval_indices(
    chrom_idx: &HashMap<String, Vec<(i32, usize)>>,
    chrom: &str,
    start: i32,
    end: i32,
) -> Vec<usize> {
    let key = normalize_chrom(chrom);
    let Some(v) = chrom_idx.get(&key) else {
        return Vec::new();
    };
    if v.is_empty() {
        return Vec::new();
    }
    let (lo, hi) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let mut out: Vec<usize> = Vec::new();
    for (pos, idx) in v.iter() {
        if *pos < lo {
            continue;
        }
        if *pos > hi {
            break;
        }
        out.push(*idx);
    }
    out
}

fn build_feature_group_ids<S: GarfieldChromPosSite>(sites: &[S], n_rows: usize) -> Vec<usize> {
    let mut map: HashMap<(String, i32), usize> = HashMap::new();
    let mut out = vec![0usize; n_rows];
    for (i, site) in sites.iter().take(n_rows).enumerate() {
        let key = (normalize_chrom(site.garfield_chrom()), site.garfield_pos());
        let gid = if let Some(g) = map.get(&key) {
            *g
        } else {
            let g = map.len();
            map.insert(key, g);
            g
        };
        out[i] = gid;
    }
    out
}

#[inline]
fn build_logic_mode_group_ids(n_sites: usize, mode: GarfieldBinMode) -> Vec<usize> {
    let row_mul = match mode {
        GarfieldBinMode::Bin => 1usize,
        GarfieldBinMode::Mbin => 3usize,
    };
    let mut out = Vec::<usize>::with_capacity(n_sites.saturating_mul(row_mul));
    for site_idx in 0..n_sites {
        for _ in 0..row_mul {
            out.push(site_idx);
        }
    }
    out
}

#[inline]
fn pos_to_physical_window_index(pos: i32, chunk_bp: u64) -> u32 {
    if chunk_bp == 0 {
        return 0;
    }
    let pos_u64 = u64::try_from(pos.max(0)).unwrap_or(0);
    u32::try_from(pos_u64 / chunk_bp).unwrap_or(u32::MAX)
}

#[inline]
fn physical_window_bounds(window_index: u32, chunk_bp: u64) -> (i32, i32) {
    let start = u64::from(window_index).saturating_mul(chunk_bp);
    let end = start.saturating_add(chunk_bp);
    (
        i32::try_from(start).unwrap_or(i32::MAX),
        i32::try_from(end).unwrap_or(i32::MAX),
    )
}

fn build_valid_null_chunks<S: GarfieldChromPosSite>(
    sites: &[S],
    chunk_bp: usize,
    min_snp_count: usize,
) -> Result<(Vec<GarfieldNullChunk>, Vec<GarfieldNullChunkSpan>), String> {
    if sites.is_empty() || chunk_bp == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let chunk_bp_u64 = u64::try_from(chunk_bp).unwrap_or(u64::MAX).max(1);
    let mut out = Vec::<GarfieldNullChunk>::new();
    let mut spans = Vec::<GarfieldNullChunkSpan>::new();
    let mut chrom_seq_code = 1u32;
    let mut i = 0usize;

    while i < sites.len() {
        check_ctrlc()?;
        let chrom_raw = sites[i].garfield_chrom();
        let chrom_code = chrom_code_for_chunk_id(chrom_raw, chrom_seq_code);
        let chunk_span_start = out.len();
        let mut j = i;
        let mut current_window_index =
            pos_to_physical_window_index(sites[i].garfield_pos(), chunk_bp_u64);
        let mut window_start_index = i;
        let mut snp_count = 0usize;

        while j < sites.len() && same_normalized_chrom(sites[j].garfield_chrom(), chrom_raw) {
            let window_index = pos_to_physical_window_index(sites[j].garfield_pos(), chunk_bp_u64);
            if window_index != current_window_index {
                if snp_count >= min_snp_count {
                    let (bp_start, bp_end) =
                        physical_window_bounds(current_window_index, chunk_bp_u64);
                    out.push(GarfieldNullChunk {
                        chrom_code,
                        window_index: current_window_index,
                        window_id: (u64::from(chrom_code) << 32) | u64::from(current_window_index),
                        start_index: window_start_index,
                        end_index: j,
                        bp_start,
                        bp_end,
                    });
                }
                current_window_index = window_index;
                window_start_index = j;
                snp_count = 0;
            }
            snp_count = snp_count.saturating_add(1);
            j = j.saturating_add(1);
        }

        if snp_count >= min_snp_count {
            let (bp_start, bp_end) = physical_window_bounds(current_window_index, chunk_bp_u64);
            out.push(GarfieldNullChunk {
                chrom_code,
                window_index: current_window_index,
                window_id: (u64::from(chrom_code) << 32) | u64::from(current_window_index),
                start_index: window_start_index,
                end_index: j,
                bp_start,
                bp_end,
            });
        }

        if out.len() > chunk_span_start {
            spans.push(GarfieldNullChunkSpan {
                start_chunk_idx: chunk_span_start,
                end_chunk_idx: out.len(),
            });
        }
        i = j;
        chrom_seq_code = chrom_seq_code.saturating_add(1);
    }

    Ok((out, spans))
}

fn allocate_stratified_chunk_quotas(
    spans: &[GarfieldNullChunkSpan],
    total_valid_chunks: usize,
    target_chunks: usize,
) -> Vec<usize> {
    let chrom_counts = spans
        .iter()
        .map(|span| span.end_chunk_idx.saturating_sub(span.start_chunk_idx))
        .collect::<Vec<_>>();
    if total_valid_chunks == 0 || target_chunks == 0 {
        return vec![0usize; chrom_counts.len()];
    }
    if total_valid_chunks <= target_chunks {
        return chrom_counts;
    }

    let target = target_chunks.min(total_valid_chunks);
    let total_u128 = total_valid_chunks as u128;
    let target_u128 = target as u128;
    let mut quotas = vec![0usize; chrom_counts.len()];
    let mut remainders = Vec::<(u128, usize, usize)>::with_capacity(chrom_counts.len());
    let mut assigned = 0usize;
    for (idx, &count) in chrom_counts.iter().enumerate() {
        let weight = (count as u128).saturating_mul(target_u128);
        let floor_q = usize::try_from(weight / total_u128)
            .unwrap_or(usize::MAX)
            .min(count);
        quotas[idx] = floor_q;
        assigned = assigned.saturating_add(floor_q);
        remainders.push((weight % total_u128, count, idx));
    }
    let mut remain = target.saturating_sub(assigned);
    remainders.sort_by(|a, b| b.cmp(a));
    for &(_, count, idx) in remainders.iter() {
        if remain == 0 {
            break;
        }
        if quotas[idx] < count {
            quotas[idx] += 1;
            remain -= 1;
        }
    }
    quotas
}

fn sample_null_chunks_stratified(
    sites: &[SiteInfo],
    extension: usize,
    target_chunks: usize,
    min_snp_count: usize,
    seed: u64,
) -> Result<(Vec<GarfieldNullChunk>, usize), String> {
    let chunk_bp = extension.saturating_mul(2).max(1);
    let (valid_chunks, chrom_spans) = build_valid_null_chunks(sites, chunk_bp, min_snp_count)?;
    let total_valid = valid_chunks.len();
    if total_valid == 0 || target_chunks == 0 {
        return Ok((Vec::new(), total_valid));
    }
    if total_valid <= target_chunks {
        return Ok((valid_chunks, total_valid));
    }

    let quotas = allocate_stratified_chunk_quotas(&chrom_spans, total_valid, target_chunks);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut picked = Vec::<GarfieldNullChunk>::with_capacity(target_chunks.min(total_valid));
    for (span, &quota) in chrom_spans.iter().zip(quotas.iter()) {
        check_ctrlc()?;
        if quota == 0 {
            continue;
        }
        let count = span.end_chunk_idx.saturating_sub(span.start_chunk_idx);
        if quota >= count {
            picked.extend_from_slice(&valid_chunks[span.start_chunk_idx..span.end_chunk_idx]);
            continue;
        }
        let sampled = sample_indices_without_replacement(&mut rng, count, quota);
        for local_idx in sampled.into_vec().into_iter() {
            picked.push(valid_chunks[span.start_chunk_idx + local_idx]);
        }
    }
    picked.sort_by(|a, b| {
        a.chrom_code
            .cmp(&b.chrom_code)
            .then_with(|| a.window_index.cmp(&b.window_index))
            .then_with(|| a.start_index.cmp(&b.start_index))
    });
    if picked.len() > target_chunks {
        picked.truncate(target_chunks);
    }
    Ok((picked, total_valid))
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
    match arr.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => arr.as_array().iter().copied().collect(),
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

#[derive(Clone, Debug)]
struct GarfieldUnitSpan {
    chrom: String,
    bp_start: i32,
    bp_end: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldLogicSiteMode {
    Bin,
    Dom,
    Rec,
    Het,
}

impl GarfieldLogicSiteMode {
    #[inline]
    fn label(self) -> &'static str {
        match self {
            Self::Bin => "BIN",
            Self::Dom => "DOM",
            Self::Rec => "REC",
            Self::Het => "HET",
        }
    }
}

#[derive(Clone, Debug)]
struct GarfieldLogicSite {
    chrom: Arc<str>,
    pos: i32,
    snp: Arc<str>,
    ref_allele: Arc<str>,
    alt_allele: Arc<str>,
    mode: GarfieldLogicSiteMode,
}

impl GarfieldChromPosSite for SiteInfo {
    #[inline]
    fn garfield_chrom(&self) -> &str {
        self.chrom.as_str()
    }

    #[inline]
    fn garfield_pos(&self) -> i32 {
        self.pos
    }

    #[inline]
    fn garfield_snp_name(&self) -> &str {
        self.snp.as_str()
    }
}

impl GarfieldDisplaySite for SiteInfo {
    #[inline]
    fn garfield_model_label(&self) -> &'static str {
        let alt = self.alt_allele.to_ascii_uppercase();
        if alt.contains("|DOM") {
            "DOM"
        } else if alt.contains("|REC") {
            "REC"
        } else if alt.contains("|HET") {
            "HET"
        } else {
            "BIN"
        }
    }

    #[inline]
    fn garfield_ref_allele(&self) -> &str {
        self.ref_allele
            .split('|')
            .next()
            .unwrap_or(self.ref_allele.as_str())
            .trim()
    }

    #[inline]
    fn garfield_alt_allele(&self) -> &str {
        self.alt_allele
            .split('|')
            .next()
            .unwrap_or(self.alt_allele.as_str())
            .trim()
    }
}

impl GarfieldChromPosSite for GarfieldLogicSite {
    #[inline]
    fn garfield_chrom(&self) -> &str {
        self.chrom.as_ref()
    }

    #[inline]
    fn garfield_pos(&self) -> i32 {
        self.pos
    }

    #[inline]
    fn garfield_snp_name(&self) -> &str {
        self.snp.as_ref()
    }
}

impl GarfieldDisplaySite for GarfieldLogicSite {
    #[inline]
    fn garfield_model_label(&self) -> &'static str {
        self.mode.label()
    }

    #[inline]
    fn garfield_ref_allele(&self) -> &str {
        self.ref_allele.as_ref()
    }

    #[inline]
    fn garfield_alt_allele(&self) -> &str {
        self.alt_allele.as_ref()
    }
}

#[derive(Clone, Debug)]
struct GarfieldLogicUnit {
    label: String,
    indices: Vec<usize>,
    spans: Vec<GarfieldUnitSpan>,
}

#[derive(Clone, Debug)]
struct GarfieldLogicBits {
    bits_flat: Vec<u64>,
    bits_hi_flat: Option<Vec<u64>>,
    row_words: usize,
    sample_ids: Vec<String>,
    sites: Vec<GarfieldLogicSite>,
    group_ids: Vec<usize>,
    n_samples: usize,
}

#[derive(Clone, Debug)]
struct GarfieldUnitPrepared {
    selected_global_rows: Vec<usize>,
    local_groups: Vec<usize>,
    geneset_stage_group_target: Option<usize>,
}

#[derive(Clone, Debug)]
struct GarfieldUnitBitMatrices {
    train_bits: Vec<u64>,
    train_bits_hi: Option<Vec<u64>>,
    row_words_train: usize,
    test_bits: Option<Vec<u64>>,
    test_bits_hi: Option<Vec<u64>>,
    row_words_test: usize,
    selected_bits_full: Option<Vec<u64>>,
    selected_bits_full_hi: Option<Vec<u64>>,
    selected_bits_full_alias_train: bool,
    selected_bits_full_hi_alias_train: bool,
}

impl GarfieldUnitBitMatrices {
    #[inline]
    fn test_bits(&self) -> &[u64] {
        self.test_bits
            .as_deref()
            .unwrap_or(self.train_bits.as_slice())
    }

    #[inline]
    fn test_bits_hi(&self) -> Option<&[u64]> {
        self.test_bits_hi
            .as_deref()
            .or(self.train_bits_hi.as_deref())
    }

    #[inline]
    fn has_fuzzy_bin(&self) -> bool {
        self.train_bits_hi.is_some()
    }

    #[inline]
    fn selected_bits_full(&self) -> Option<&[u64]> {
        self.selected_bits_full.as_deref().or_else(|| {
            if self.selected_bits_full_alias_train {
                Some(self.train_bits.as_slice())
            } else {
                None
            }
        })
    }

    #[inline]
    fn selected_bits_full_hi(&self) -> Option<&[u64]> {
        self.selected_bits_full_hi.as_deref().or_else(|| {
            if self.selected_bits_full_hi_alias_train {
                self.train_bits_hi.as_deref()
            } else {
                None
            }
        })
    }
}

fn beam_search_train_test_continuous_dispatch(
    y_train: &[f64],
    prepared_bits: &GarfieldUnitBitMatrices,
    n_rows: usize,
    y_test: &[f64],
    group_ids: &[usize],
    params: BeamSearchParams,
) -> Result<Vec<BeamRuleCandidate>, String> {
    if let Some(train_hi) = prepared_bits.train_bits_hi.as_deref() {
        let test_hi = prepared_bits.test_bits_hi().ok_or_else(|| {
            "internal error: fuzzy GARFIELD prepared bits are missing test high bitplane"
                .to_string()
        })?;
        beam_search_train_test_continuous_fuzzy(
            y_train,
            prepared_bits.train_bits.as_slice(),
            train_hi,
            prepared_bits.row_words_train,
            n_rows,
            y_train.len(),
            y_test,
            prepared_bits.test_bits(),
            test_hi,
            prepared_bits.row_words_test,
            y_test.len(),
            group_ids,
            params,
        )
    } else {
        beam_search_train_test_continuous(
            y_train,
            prepared_bits.train_bits.as_slice(),
            prepared_bits.row_words_train,
            n_rows,
            y_train.len(),
            y_test,
            prepared_bits.test_bits(),
            prepared_bits.row_words_test,
            y_test.len(),
            group_ids,
            params,
        )
    }
}

#[inline]
fn selected_rows_contiguous_range(selected_global_rows: &[usize]) -> Option<(usize, usize)> {
    let start = *selected_global_rows.first()?;
    for (offset, &row_idx) in selected_global_rows.iter().enumerate() {
        if row_idx != start.saturating_add(offset) {
            return None;
        }
    }
    Some((start, start.saturating_add(selected_global_rows.len())))
}

fn local_sites_from_selected_rows(
    selected_global_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
) -> Result<Vec<GarfieldLogicSite>, String> {
    let mut out = Vec::<GarfieldLogicSite>::with_capacity(selected_global_rows.len());
    for &idx in selected_global_rows.iter() {
        let site = logic_bits.sites.get(idx).ok_or_else(|| {
            format!("logic site index out of range while building local sites: {idx}")
        })?;
        out.push(site.clone());
    }
    Ok(out)
}

#[inline]
fn intern_logic_site_text(pool: &mut HashMap<String, Arc<str>>, text: &str) -> Arc<str> {
    if let Some(existing) = pool.get(text) {
        return existing.clone();
    }
    let shared: Arc<str> = Arc::from(text);
    pool.insert(text.to_string(), shared.clone());
    shared
}

fn build_logic_sites_from_metadata(
    sites: &[SiteInfo],
    mode: GarfieldBinMode,
) -> Vec<GarfieldLogicSite> {
    let row_mul = match mode {
        GarfieldBinMode::Bin => 1usize,
        GarfieldBinMode::Mbin => 3usize,
    };
    let mut out = Vec::<GarfieldLogicSite>::with_capacity(sites.len().saturating_mul(row_mul));
    let mut chrom_pool = HashMap::<String, Arc<str>>::new();
    let mut snp_pool = HashMap::<String, Arc<str>>::new();
    let mut allele_pool = HashMap::<String, Arc<str>>::new();
    let use_inherited_snp_names = garfield_sites_have_distinct_snp_names(sites);
    for site in sites.iter() {
        let chrom = intern_logic_site_text(&mut chrom_pool, site.chrom.as_str());
        let snp_name = if use_inherited_snp_names {
            intern_logic_site_text(&mut snp_pool, site.snp.trim())
        } else {
            intern_logic_site_text(
                &mut snp_pool,
                format!("{}_{}", site.chrom, site.pos).as_str(),
            )
        };
        let ref_allele = intern_logic_site_text(&mut allele_pool, site.ref_allele.as_str());
        let alt_allele = intern_logic_site_text(&mut allele_pool, site.alt_allele.as_str());
        let push_mode = |out: &mut Vec<GarfieldLogicSite>, mode_one: GarfieldLogicSiteMode| {
            out.push(GarfieldLogicSite {
                chrom: chrom.clone(),
                pos: site.pos,
                snp: snp_name.clone(),
                ref_allele: ref_allele.clone(),
                alt_allele: alt_allele.clone(),
                mode: mode_one,
            });
        };
        match mode {
            GarfieldBinMode::Bin => push_mode(&mut out, GarfieldLogicSiteMode::Bin),
            GarfieldBinMode::Mbin => {
                push_mode(&mut out, GarfieldLogicSiteMode::Dom);
                push_mode(&mut out, GarfieldLogicSiteMode::Rec);
                push_mode(&mut out, GarfieldLogicSiteMode::Het);
            }
        }
    }
    out
}

#[inline]
fn garfield_snp_name_missing(raw_name: &str) -> bool {
    let text = raw_name.trim();
    text.is_empty() || text == "." || text.eq_ignore_ascii_case("nan")
}

fn garfield_sites_have_distinct_snp_names(sites: &[SiteInfo]) -> bool {
    if sites.is_empty() {
        return false;
    }
    let mut seen = HashSet::<String>::with_capacity(sites.len());
    for site in sites.iter() {
        let snp_name = site.snp.trim();
        if garfield_snp_name_missing(snp_name) {
            return false;
        }
        if !seen.insert(snp_name.to_string()) {
            return false;
        }
    }
    true
}

fn materialize_prepared_bit_matrices(
    prepared: &GarfieldUnitPrepared,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    include_full_bits: bool,
) -> Result<GarfieldUnitBitMatrices, String> {
    let t0 = Instant::now();
    let contiguous = selected_rows_contiguous_range(prepared.selected_global_rows.as_slice());
    let train_is_full_sample =
        sample_indices_are_full_identity(train_idx_local, logic_bits.n_samples);
    let (train_bits, row_words_train) = if let Some((row_start, row_end)) = contiguous {
        packed_rows_subset_from_full_bits_range(
            logic_bits.bits_flat.as_slice(),
            logic_bits.row_words,
            row_start,
            row_end,
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?
    } else {
        packed_rows_subset_from_full_bits(
            logic_bits.bits_flat.as_slice(),
            logic_bits.row_words,
            prepared.selected_global_rows.as_slice(),
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?
    };
    let train_bits_hi = if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
        Some(if let Some((row_start, row_end)) = contiguous {
            packed_rows_subset_from_full_bits_range(
                bits_hi_flat.as_slice(),
                logic_bits.row_words,
                row_start,
                row_end,
                train_idx_local,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?
            .0
        } else {
            packed_rows_subset_from_full_bits(
                bits_hi_flat.as_slice(),
                logic_bits.row_words,
                prepared.selected_global_rows.as_slice(),
                train_idx_local,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?
            .0
        })
    } else {
        None
    };
    let (test_bits, row_words_test) = if train_idx_local == test_idx_local {
        (None, row_words_train)
    } else if let Some((row_start, row_end)) = contiguous {
        let (bits, row_words_test) = packed_rows_subset_from_full_bits_range(
            logic_bits.bits_flat.as_slice(),
            logic_bits.row_words,
            row_start,
            row_end,
            test_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        (Some(bits), row_words_test)
    } else {
        let (bits, row_words_test) = packed_rows_subset_from_full_bits(
            logic_bits.bits_flat.as_slice(),
            logic_bits.row_words,
            prepared.selected_global_rows.as_slice(),
            test_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        (Some(bits), row_words_test)
    };
    let test_bits_hi = if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
        if train_idx_local == test_idx_local {
            None
        } else if let Some((row_start, row_end)) = contiguous {
            let (bits, _) = packed_rows_subset_from_full_bits_range(
                bits_hi_flat.as_slice(),
                logic_bits.row_words,
                row_start,
                row_end,
                test_idx_local,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            Some(bits)
        } else {
            let (bits, _) = packed_rows_subset_from_full_bits(
                bits_hi_flat.as_slice(),
                logic_bits.row_words,
                prepared.selected_global_rows.as_slice(),
                test_idx_local,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            Some(bits)
        }
    } else {
        None
    };
    let alias_full_to_train =
        include_full_bits && train_is_full_sample && row_words_train == logic_bits.row_words;
    let selected_bits_full = if include_full_bits && !alias_full_to_train {
        Some(if let Some((row_start, row_end)) = contiguous {
            gather_rows_by_range(
                logic_bits.bits_flat.as_slice(),
                logic_bits.row_words,
                row_start,
                row_end,
                "garfield::unit_bits_range",
            )?
        } else {
            gather_rows_by_indices(
                logic_bits.bits_flat.as_slice(),
                logic_bits.row_words,
                prepared.selected_global_rows.as_slice(),
                "garfield::unit_bits_indices",
            )?
        })
    } else {
        None
    };
    let selected_bits_full_hi = if include_full_bits && !alias_full_to_train {
        if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
            Some(if let Some((row_start, row_end)) = contiguous {
                gather_rows_by_range(
                    bits_hi_flat.as_slice(),
                    logic_bits.row_words,
                    row_start,
                    row_end,
                    "garfield::unit_bits_hi_range",
                )?
            } else {
                gather_rows_by_indices(
                    bits_hi_flat.as_slice(),
                    logic_bits.row_words,
                    prepared.selected_global_rows.as_slice(),
                    "garfield::unit_bits_hi_indices",
                )?
            })
        } else {
            None
        }
    } else {
        None
    };
    let out = GarfieldUnitBitMatrices {
        train_bits,
        train_bits_hi,
        row_words_train,
        test_bits,
        test_bits_hi,
        row_words_test,
        selected_bits_full,
        selected_bits_full_hi,
        selected_bits_full_alias_train: alias_full_to_train,
        selected_bits_full_hi_alias_train: alias_full_to_train && logic_bits.bits_hi_flat.is_some(),
    };
    GARFIELD_MATERIALIZE_BITS_NS.fetch_add(elapsed_ns_saturating(t0), Ordering::Relaxed);
    Ok(out)
}

#[derive(Clone, Debug)]
struct GarfieldUnitMlContext {
    unit_index: usize,
    selected_global_rows: Vec<usize>,
    ranked_global_rows: Vec<usize>,
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
    pos: i32,
    score: f64,
    delta_score: String,
    support_bits: Option<GarfieldRuleSupportBits>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldRuleDisplayPolarity {
    Original,
    Complement,
}

#[derive(Clone, Debug)]
struct GarfieldLogicPipelineResult {
    pseudo_prefix: Option<String>,
    rules_tsv: Option<String>,
    posterior_json: Option<String>,
    memory_debug: Option<GarfieldMemoryDebugSummary>,
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
    split_applied: bool,
    n_train: usize,
    n_test: usize,
    n_samples: usize,
    units_total: usize,
    units_scanned: usize,
    timing_total_wall_s: f64,
    timing_scan_wall_s: f64,
    timing_scan_beam_wall_s: f64,
    timing_scan_literal_score_wall_s: f64,
    timing_scan_ml_select_wall_s: f64,
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
    train_fit: GarfieldResidualResult,
    test_fit: GarfieldResidualResult,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct GarfieldScanBimRange {
    chrom: String,
    bp_start: i32,
    bp_end: i32,
}

#[inline]
fn singleton_rule_from_literal(lit: BeamLiteral) -> BeamRule {
    BeamRule {
        first: lit,
        rest: Vec::new(),
    }
}

#[inline]
fn format_delta_metric_value(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    format!("{value:.4}")
}

#[inline]
fn literal_metric_token(value: f64, negated: bool) -> String {
    let text = format_delta_metric_value(value);
    if negated {
        format!("!{text}")
    } else {
        text
    }
}

#[inline]
fn logic_symbol_for_display(
    op: BeamBinaryOp,
    polarity: GarfieldRuleDisplayPolarity,
) -> &'static str {
    match (op, polarity) {
        (BeamBinaryOp::And, GarfieldRuleDisplayPolarity::Original)
        | (BeamBinaryOp::Or, GarfieldRuleDisplayPolarity::Complement) => "&",
        (BeamBinaryOp::Or, GarfieldRuleDisplayPolarity::Original)
        | (BeamBinaryOp::And, GarfieldRuleDisplayPolarity::Complement) => "|",
    }
}

#[inline]
fn expr_symbol_for_display(
    op: BeamBinaryOp,
    polarity: GarfieldRuleDisplayPolarity,
) -> &'static str {
    match (op, polarity) {
        (BeamBinaryOp::And, GarfieldRuleDisplayPolarity::Original)
        | (BeamBinaryOp::Or, GarfieldRuleDisplayPolarity::Complement) => "AND",
        (BeamBinaryOp::Or, GarfieldRuleDisplayPolarity::Original)
        | (BeamBinaryOp::And, GarfieldRuleDisplayPolarity::Complement) => "OR",
    }
}

#[inline]
fn display_literal_negated(literal_negated: bool, polarity: GarfieldRuleDisplayPolarity) -> bool {
    match polarity {
        GarfieldRuleDisplayPolarity::Original => literal_negated,
        GarfieldRuleDisplayPolarity::Complement => !literal_negated,
    }
}

#[inline]
fn display_binary_op(op: BeamBinaryOp, polarity: GarfieldRuleDisplayPolarity) -> BeamBinaryOp {
    match polarity {
        GarfieldRuleDisplayPolarity::Original => op,
        GarfieldRuleDisplayPolarity::Complement => match op {
            BeamBinaryOp::And => BeamBinaryOp::Or,
            BeamBinaryOp::Or => BeamBinaryOp::And,
        },
    }
}

fn rule_display_negated_with_polarity(
    rule: &BeamRule,
    polarity: GarfieldRuleDisplayPolarity,
) -> Vec<bool> {
    let mut out = Vec::<bool>::with_capacity(rule.len());
    out.push(display_literal_negated(rule.first.negated, polarity));
    out.extend(
        rule.rest
            .iter()
            .map(|(_, lit)| display_literal_negated(lit.negated, polarity)),
    );
    out
}

fn rule_display_ops_with_polarity(
    rule: &BeamRule,
    polarity: GarfieldRuleDisplayPolarity,
) -> Vec<BeamBinaryOp> {
    rule.rest
        .iter()
        .map(|(op, _)| display_binary_op(*op, polarity))
        .collect()
}

#[inline]
fn complement_bits_in_place(bits: &mut [u64], n_samples: usize) {
    bitnot_masked(bits, n_samples);
}

#[inline]
fn complement_dual_bits_in_place(ge1_bits: &mut [u64], ge2_bits: &mut [u64], n_samples: usize) {
    let orig_ge1 = ge1_bits.to_vec();
    let orig_ge2 = ge2_bits.to_vec();
    for (dst, &src) in ge1_bits.iter_mut().zip(orig_ge2.iter()) {
        *dst = src;
    }
    for (dst, &src) in ge2_bits.iter_mut().zip(orig_ge1.iter()) {
        *dst = src;
    }
    bitnot_masked(ge1_bits, n_samples);
    bitnot_masked(ge2_bits, n_samples);
}

#[inline]
fn preferred_display_polarity(
    rule: &BeamRule,
    full_bits: &[u64],
    n_samples: usize,
) -> GarfieldRuleDisplayPolarity {
    let _ = (rule, full_bits, n_samples);
    // Search is AND-only and output should preserve the discovered rule family
    // directly instead of rewriting complements into OR-form displays.
    GarfieldRuleDisplayPolarity::Original
}

fn rule_expr_with_polarity<S: GarfieldDisplaySite>(
    rule: &BeamRule,
    local_sites: &[S],
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<String, String> {
    let first = local_sites
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    let mut out = literal_expr(first, display_literal_negated(rule.first.negated, polarity));
    for (op, lit) in rule.rest.iter() {
        let site = local_sites
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        out.push(' ');
        out.push_str(expr_symbol_for_display(*op, polarity));
        out.push(' ');
        out.push_str(&literal_expr(
            site,
            display_literal_negated(lit.negated, polarity),
        ));
    }
    Ok(out)
}

fn rule_snp_name_with_polarity<S: GarfieldChromPosSite>(
    rule: &BeamRule,
    local_sites: &[S],
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<String, String> {
    let first = local_sites
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    let mut out = literal_name(first, display_literal_negated(rule.first.negated, polarity));
    for (op, lit) in rule.rest.iter() {
        let site = local_sites
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        out.push_str(logic_symbol_for_display(*op, polarity));
        out.push_str(&literal_name(
            site,
            display_literal_negated(lit.negated, polarity),
        ));
    }
    Ok(out)
}

fn rule_bim_name_with_polarity<S: GarfieldChromPosSite>(
    rule: &BeamRule,
    local_sites: &[S],
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<String, String> {
    rule_snp_name_with_polarity(rule, local_sites, polarity)
}

fn rule_bim_alleles_with_polarity<S: GarfieldDisplaySite>(
    rule: &BeamRule,
    local_sites: &[S],
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<(String, String), String> {
    let mut allele0 = Vec::<String>::with_capacity(rule.len());
    let mut allele1 = Vec::<String>::with_capacity(rule.len());
    let first = local_sites
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    let (a0, a1) =
        literal_bim_alleles(first, display_literal_negated(rule.first.negated, polarity));
    allele0.push(a0);
    allele1.push(a1);
    for (_, lit) in rule.rest.iter() {
        let site = local_sites
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        let (a0, a1) = literal_bim_alleles(site, display_literal_negated(lit.negated, polarity));
        allele0.push(a0);
        allele1.push(a1);
    }
    Ok((allele0.join(","), allele1.join(",")))
}

fn rule_ml_rank_name_with_polarity(
    rule: &BeamRule,
    polarity: GarfieldRuleDisplayPolarity,
) -> String {
    let mut out = literal_ml_rank_name(
        rule.first.row_index,
        display_literal_negated(rule.first.negated, polarity),
    );
    for (op, lit) in rule.rest.iter() {
        out.push_str(logic_symbol_for_display(*op, polarity));
        out.push_str(&literal_ml_rank_name(
            lit.row_index,
            display_literal_negated(lit.negated, polarity),
        ));
    }
    out
}

fn build_rule_delta_score_annotation(
    rule: &BeamRule,
    child_score: f64,
    y_test: &[f64],
    bits_test: &[u64],
    bits_test_hi: Option<&[u64]>,
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    params: &BeamSearchParams,
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<String, String> {
    let mut score_txt = String::new();

    let mut append_literal = |op: Option<BeamBinaryOp>, lit: BeamLiteral| -> Result<(), String> {
        if let Some(op_use) = op {
            let sym = logic_symbol_for_display(op_use, polarity);
            score_txt.push_str(sym);
        }
        let display_lit = BeamLiteral {
            negated: display_literal_negated(lit.negated, polarity),
            ..lit
        };
        let singleton = singleton_rule_from_literal(display_lit);
        let test_sc = if let Some(bits_test_hi_use) = bits_test_hi {
            evaluate_rule_continuous_dual(
                &singleton,
                y_test,
                bits_test,
                bits_test_hi_use,
                row_words_test,
                n_rows,
                n_test,
                params.lambda_len,
                params.lambda_not,
            )?
        } else {
            evaluate_rule_continuous(
                &singleton,
                y_test,
                bits_test,
                row_words_test,
                n_rows,
                n_test,
                params.lambda_len,
                params.lambda_not,
            )?
        };
        let bucket = bucket_from_rule(&singleton, test_sc.dosage_maf);
        let single_score = rank_rule_score_components_with_bucket(
            bucket,
            singleton.len(),
            singleton.not_count(),
            test_sc.raw_score,
            test_sc.raw_score,
            params,
            false,
        );
        score_txt.push_str(&literal_metric_token(single_score, display_lit.negated));
        Ok(())
    };

    append_literal(None, rule.first)?;
    for (op, lit) in rule.rest.iter() {
        append_literal(Some(*op), *lit)?;
    }

    score_txt.push_str("->");
    score_txt.push_str(&format_delta_metric_value(child_score));
    Ok(score_txt)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimBenchLogic {
    Single,
    And,
    Or,
}

#[derive(Clone, Debug)]
struct SimBenchTerm {
    term_id: usize,
    kind: String,
    logic: SimBenchLogic,
    logic_text: String,
    sites_text: String,
    sites: Vec<(String, i32)>,
    label: String,
}

fn parse_simbench_logic(raw: &str) -> Result<SimBenchLogic, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "single" => Ok(SimBenchLogic::Single),
        "and" => Ok(SimBenchLogic::And),
        "or" => Ok(SimBenchLogic::Or),
        other => Err(format!(
            "simbench logic must be one of: single, and, or; got {other}"
        )),
    }
}

fn parse_simbench_sites(raw: &str) -> Result<Vec<(String, i32)>, String> {
    let mut out = Vec::<(String, i32)>::new();
    for token in raw.split(';') {
        let site_txt = token.trim();
        if site_txt.is_empty() {
            continue;
        }
        let Some((chrom, pos_txt)) = site_txt.split_once(':') else {
            return Err(format!(
                "invalid simbench site token '{site_txt}'; expected chrom:pos"
            ));
        };
        let pos = pos_txt.trim().parse::<i32>().map_err(|e| {
            format!("invalid simbench site position '{pos_txt}' in '{site_txt}': {e}")
        })?;
        out.push((chrom.trim().to_string(), pos));
    }
    if out.is_empty() {
        return Err("simbench sites column is empty".to_string());
    }
    Ok(out)
}

fn parse_simbench_terms(path: &str) -> Result<Vec<SimBenchTerm>, String> {
    let file = File::open(path).map_err(|e| format!("open simbench file {path}: {e}"))?;
    let mut rdr = BufReader::new(file);
    let mut header = String::new();
    let n_read = rdr
        .read_line(&mut header)
        .map_err(|e| format!("read simbench header {path}: {e}"))?;
    if n_read == 0 {
        return Err(format!("simbench file is empty: {path}"));
    }
    let cols = header
        .trim_end_matches(&['\r', '\n'][..])
        .split('\t')
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    let mut col_idx = HashMap::<String, usize>::new();
    for (i, col) in cols.iter().enumerate() {
        col_idx.insert(col.to_ascii_lowercase(), i);
    }
    let term_idx = col_idx
        .get("term_id")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: term_id"))?;
    let kind_idx = col_idx
        .get("kind")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: kind"))?;
    let logic_idx = col_idx
        .get("logic")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: logic"))?;
    let sites_idx = col_idx
        .get("sites")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: sites"))?;
    let label_idx = col_idx
        .get("label")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: label"))?;

    let mut out = Vec::<SimBenchTerm>::new();
    for (line_no0, line_res) in rdr.lines().enumerate() {
        let line_no = line_no0 + 2;
        let line = line_res.map_err(|e| format!("read {path}:{line_no}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').map(|s| s.trim()).collect::<Vec<_>>();
        let get_field = |idx: usize| -> &str {
            if idx < fields.len() {
                fields[idx]
            } else {
                ""
            }
        };
        let term_id = get_field(term_idx)
            .parse::<usize>()
            .unwrap_or(out.len() + 1);
        let kind = get_field(kind_idx).to_string();
        let logic_text = get_field(logic_idx).to_string();
        let logic =
            parse_simbench_logic(&logic_text).map_err(|e| format!("{path}:{line_no}: {e}"))?;
        let sites_text = get_field(sites_idx).to_string();
        let sites =
            parse_simbench_sites(&sites_text).map_err(|e| format!("{path}:{line_no}: {e}"))?;
        let label = get_field(label_idx).to_string();
        out.push(SimBenchTerm {
            term_id,
            kind,
            logic,
            logic_text,
            sites_text,
            sites,
            label,
        });
    }
    Ok(out)
}

fn normalize_simbench_raw_row(row: &mut [f32], site: &mut SiteInfo) -> Result<(), String> {
    if row.is_empty() {
        return Err("empty genotype row".to_string());
    }
    let keep = process_snp_row(
        row,
        &mut site.ref_allele,
        &mut site.alt_allele,
        0.0,
        1.0,
        false,
        false,
        1.0,
    );
    if !keep {
        return Err("failed to normalize genotype row".to_string());
    }
    Ok(())
}

fn pack_simbench_raw_row_dual_words(row: &[f32]) -> Result<(Vec<u64>, Vec<u64>), String> {
    if row.is_empty() {
        return Err("empty genotype row".to_string());
    }
    let row_words = words_for_samples(row.len());
    let mut ge1 = vec![0u64; row_words];
    let mut ge2 = vec![0u64; row_words];
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("non-finite genotype value at sample index {i}"));
        }
        let r = v.round();
        let g = if r <= 0.0 {
            0u8
        } else if r >= 2.0 {
            2u8
        } else {
            1u8
        };
        if g >= 1 {
            ge1[i >> 6] |= 1u64 << (i & 63);
        }
        if g >= 2 {
            ge2[i >> 6] |= 1u64 << (i & 63);
        }
    }
    apply_tail_mask(&mut ge1, tail_mask(row.len()));
    apply_tail_mask(&mut ge2, tail_mask(row.len()));
    Ok((ge1, ge2))
}

fn simbench_rule_from_site_count(logic: SimBenchLogic, n_sites: usize) -> Result<BeamRule, String> {
    if n_sites == 0 {
        return Err("simbench term has no sites".to_string());
    }
    if matches!(logic, SimBenchLogic::Single) && n_sites != 1 {
        return Err(format!(
            "simbench logic 'single' expects exactly 1 site, got {n_sites}"
        ));
    }
    let first = BeamLiteral {
        row_index: 0,
        group_id: 0,
        negated: false,
    };
    let mut rest = Vec::<(BeamBinaryOp, BeamLiteral)>::with_capacity(n_sites.saturating_sub(1));
    if let Some(op) = simbench_logic_to_beam_op(logic) {
        for row_index in 1..n_sites {
            rest.push((
                op,
                BeamLiteral {
                    row_index,
                    group_id: row_index,
                    negated: false,
                },
            ));
        }
    }
    Ok(BeamRule { first, rest })
}

fn simbench_rule_without_literal(rule: &BeamRule, remove_idx: usize) -> Option<BeamRule> {
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

#[allow(clippy::too_many_arguments)]
fn simbench_best_ancestor_raw_baseline_dual(
    rule: &BeamRule,
    y: &[f64],
    ge1_flat: &[u64],
    ge2_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    lambda_len: f64,
    lambda_not: f64,
    disable_parent_delta: bool,
) -> Result<f64, String> {
    if disable_parent_delta || rule.len() <= 1 {
        return Ok(0.0);
    }
    if rule.len() == 2 {
        let mut best = f64::NEG_INFINITY;
        for lit in std::iter::once(rule.first).chain(rule.rest.iter().map(|(_, lit)| *lit)) {
            let singleton = singleton_rule_from_literal(lit);
            let sc = evaluate_rule_continuous_dual(
                &singleton, y, ge1_flat, ge2_flat, row_words, n_rows, n_samples, lambda_len,
                lambda_not,
            )?;
            best = best.max(sc.raw_score);
        }
        return Ok(if best.is_finite() { best } else { 0.0 });
    }
    let mut best = f64::NEG_INFINITY;
    for remove_idx in 0..rule.len() {
        let Some(parent_rule) = simbench_rule_without_literal(rule, remove_idx) else {
            continue;
        };
        let parent_raw = evaluate_rule_continuous_dual(
            &parent_rule,
            y,
            ge1_flat,
            ge2_flat,
            row_words,
            n_rows,
            n_samples,
            lambda_len,
            lambda_not,
        )?
        .raw_score;
        let parent_ancestor = simbench_best_ancestor_raw_baseline_dual(
            &parent_rule,
            y,
            ge1_flat,
            ge2_flat,
            row_words,
            n_rows,
            n_samples,
            lambda_len,
            lambda_not,
            disable_parent_delta,
        )?;
        best = best.max(parent_raw.max(parent_ancestor));
    }
    Ok(if best.is_finite() { best } else { 0.0 })
}

fn simbench_logic_symbol(logic: SimBenchLogic) -> &'static str {
    match logic {
        SimBenchLogic::Single => "",
        SimBenchLogic::And => " & ",
        SimBenchLogic::Or => " | ",
    }
}

fn simbench_rule_name<S: GarfieldChromPosSite>(
    logic: SimBenchLogic,
    sites: &[S],
    label: &str,
) -> String {
    let trimmed = label.trim();
    if !trimmed.is_empty() {
        trimmed.to_string()
    } else {
        sites
            .iter()
            .map(site_base_label)
            .collect::<Vec<_>>()
            .join(simbench_logic_symbol(logic))
    }
}

fn simbench_rule_bim_name<S: GarfieldDisplaySite>(
    logic: SimBenchLogic,
    sites: &[S],
    label: &str,
) -> String {
    let trimmed = label.trim();
    if !trimmed.is_empty() {
        trimmed
            .chars()
            .filter(|c| !c.is_ascii_whitespace())
            .collect()
    } else {
        sites
            .iter()
            .map(|site| literal_name(site, false))
            .collect::<Vec<_>>()
            .join(simbench_logic_symbol_compact(logic))
    }
}

fn simbench_rule_bim_alleles<S: GarfieldDisplaySite>(sites: &[S]) -> (String, String) {
    let mut allele0 = Vec::<String>::with_capacity(sites.len());
    let mut allele1 = Vec::<String>::with_capacity(sites.len());
    for site in sites.iter() {
        let (a0, a1) = literal_bim_alleles(site, false);
        allele0.push(a0);
        allele1.push(a1);
    }
    (allele0.join(","), allele1.join(","))
}

fn simbench_rule_expr<S: GarfieldDisplaySite>(
    logic: SimBenchLogic,
    sites: &[S],
) -> Result<String, String> {
    let mut it = sites.iter();
    let Some(first) = it.next() else {
        return Err("simbench term has no sites".to_string());
    };
    let mut out = literal_expr(first, false);
    let op_txt = match logic {
        SimBenchLogic::Single => "",
        SimBenchLogic::And => "AND",
        SimBenchLogic::Or => "OR",
    };
    if logic != SimBenchLogic::Single {
        for site in it {
            out.push(' ');
            out.push_str(op_txt);
            out.push(' ');
            out.push_str(&literal_expr(site, false));
        }
    }
    Ok(out)
}

#[inline]
fn unit_span_contains_site(span: &GarfieldUnitSpan, chrom: &str, pos: i32) -> bool {
    same_normalized_chrom(&span.chrom, chrom)
        && pos >= span.bp_start.min(span.bp_end)
        && pos <= span.bp_start.max(span.bp_end)
}

fn nearest_unit_span_index(unit: &GarfieldLogicUnit, chrom: &str, pos: i32) -> Option<usize> {
    let mut best = None::<(i64, i64, usize)>;
    for (idx, span) in unit.spans.iter().enumerate() {
        if !unit_span_contains_site(span, chrom, pos) {
            continue;
        }
        let lo = i64::from(span.bp_start.min(span.bp_end));
        let hi = i64::from(span.bp_start.max(span.bp_end));
        let center = (lo + hi) / 2;
        let dist = (i64::from(pos) - center).abs();
        let span_len = (hi - lo).abs();
        let cand = (dist, span_len, idx);
        if best.map(|cur| cand < cur).unwrap_or(true) {
            best = Some(cand);
        }
    }
    best.map(|(_, _, idx)| idx)
}

fn build_unit_window_group_ids<S: GarfieldChromPosSite>(
    unit: &GarfieldLogicUnit,
    selected_global_rows: &[usize],
    sites: &[S],
    unit_kind_lc: &str,
) -> Option<Vec<usize>> {
    if unit_kind_lc != "geneset" || unit.spans.len() <= 1 || selected_global_rows.is_empty() {
        return None;
    }
    let mut out = Vec::<usize>::with_capacity(selected_global_rows.len());
    let mut fallback_gid = unit.spans.len();
    for &global_idx in selected_global_rows.iter() {
        let site = sites.get(global_idx)?;
        if let Some(gid) = nearest_unit_span_index(unit, site.garfield_chrom(), site.garfield_pos())
        {
            out.push(gid);
        } else {
            out.push(fallback_gid);
            fallback_gid = fallback_gid.saturating_add(1);
        }
    }
    Some(out)
}

#[inline]
fn distinct_group_count(group_ids: &[usize]) -> usize {
    if group_ids.len() <= 1 {
        return group_ids.len();
    }
    group_ids.iter().copied().collect::<HashSet<_>>().len()
}

#[inline]
fn geneset_stage_group_target(unit_kind_lc: &str, local_groups: &[usize]) -> Option<usize> {
    if unit_kind_lc != "geneset" || local_groups.is_empty() {
        return None;
    }
    Some(distinct_group_count(local_groups).max(1))
}

#[inline]
fn beam_params_for_prepared(
    prepared: &GarfieldUnitPrepared,
    beam_params: BeamSearchParams,
) -> BeamSearchParams {
    if let Some(required_groups) = prepared.geneset_stage_group_target {
        BeamSearchParams {
            group_constraint: BeamGroupConstraintMode::ExcludeUntilDistinctGroups(required_groups),
            ..beam_params
        }
    } else {
        beam_params
    }
}

fn unit_contains_any_simbench_site(unit: &GarfieldLogicUnit, terms: &[SimBenchTerm]) -> bool {
    unit.spans.iter().any(|span| {
        terms.iter().any(|term| {
            term.sites
                .iter()
                .any(|(chrom, pos)| unit_span_contains_site(span, chrom, *pos))
        })
    })
}

fn build_simbench_ml_contexts(
    terms: &[SimBenchTerm],
    units: &[GarfieldLogicUnit],
    scan_unit_indices: &[usize],
    unit_kind_lc: &str,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    y_train: &[f64],
    allow_parallel: bool,
) -> Result<Vec<GarfieldUnitMlContext>, String> {
    if terms.is_empty() || scan_unit_indices.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::<GarfieldUnitMlContext>::new();
    for &ui in scan_unit_indices.iter() {
        let unit = &units[ui];
        if !unit_contains_any_simbench_site(unit, terms) {
            continue;
        }
        if unit.indices.is_empty() {
            continue;
        }
        let dense_train = dense_dosage_rows_from_full_bits(
            logic_bits.bits_flat.as_slice(),
            logic_bits.bits_hi_flat.as_deref(),
            logic_bits.row_words,
            unit.indices.as_slice(),
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        if dense_train.is_empty() {
            continue;
        }
        let n_region = dense_train.len();
        let ml_group_ids = build_unit_window_group_ids(
            unit,
            unit.indices.as_slice(),
            logic_bits.sites.as_slice(),
            unit_kind_lc,
        );
        let (selected_global_rows, ranked_global_rows) = if let Some(engine_one) = engine {
            let keep_k = resolve_ml_keep_k(n_region, ml_top_k, ml_top_frac);
            let top_local = select_ml_top_local_indices(
                dense_train.as_slice(),
                y_train,
                response,
                engine_one,
                ExtraTreesConfig {
                    allow_parallel,
                    ..tree_cfg
                },
                importance,
                perm_cfg,
                keep_k,
                GarfieldMlKeepPolicy::TopK,
                tree_cfg.seed ^ 0xB6D5_0C11_8E91_3F27,
                ml_group_ids.as_deref(),
            )?;
            if top_local.is_empty() {
                continue;
            }
            let ranked_local = select_ml_top_only_local_indices(
                dense_train.as_slice(),
                y_train,
                response,
                engine_one,
                ExtraTreesConfig {
                    allow_parallel,
                    ..tree_cfg
                },
                importance,
                perm_cfg,
                n_region,
                ml_group_ids.as_deref(),
            )?;
            (
                top_local
                    .iter()
                    .map(|&idx| unit.indices[idx])
                    .collect::<Vec<_>>(),
                ranked_local
                    .iter()
                    .map(|&idx| unit.indices[idx])
                    .collect::<Vec<_>>(),
            )
        } else {
            (unit.indices.clone(), unit.indices.clone())
        };
        out.push(GarfieldUnitMlContext {
            unit_index: ui,
            selected_global_rows,
            ranked_global_rows,
        });
    }
    Ok(out)
}

fn format_simbench_ml_rank(logic: SimBenchLogic, ranks: &[Option<usize>]) -> String {
    if ranks.is_empty() {
        return ".".to_string();
    }
    let joiner = match logic {
        SimBenchLogic::Single => "",
        SimBenchLogic::And => " & ",
        SimBenchLogic::Or => " | ",
    };
    ranks
        .iter()
        .map(|rank| match rank {
            Some(v) => v.to_string(),
            None => ".".to_string(),
        })
        .collect::<Vec<_>>()
        .join(joiner)
}

fn resolve_simbench_ml_rank(
    logic: SimBenchLogic,
    site_row_candidates: &[Vec<usize>],
    contexts: &[GarfieldUnitMlContext],
) -> (String, usize) {
    if site_row_candidates.is_empty() {
        return (".".to_string(), 0);
    }
    let mut best_selected = None::<(usize, usize, usize, usize, String, usize)>;
    let mut best_ranked = None::<(usize, usize, usize, usize, String, usize)>;
    for ctx in contexts.iter() {
        let selected_rank_map = ctx
            .selected_global_rows
            .iter()
            .enumerate()
            .map(|(i, &row)| (row, i + 1))
            .collect::<HashMap<usize, usize>>();
        let selected_ranks = site_row_candidates
            .iter()
            .map(|cands| {
                cands
                    .iter()
                    .filter_map(|idx| selected_rank_map.get(idx).copied())
                    .min()
            })
            .collect::<Vec<_>>();
        let selected_matched = selected_ranks.iter().filter(|v| v.is_some()).count();
        if selected_matched > 0 {
            let sum_rank = selected_ranks.iter().flatten().copied().sum::<usize>();
            let max_rank = selected_ranks
                .iter()
                .flatten()
                .copied()
                .max()
                .unwrap_or(usize::MAX);
            let ml_rank = format_simbench_ml_rank(logic, selected_ranks.as_slice());
            let candidate = (
                usize::MAX - selected_matched,
                max_rank,
                sum_rank,
                ctx.unit_index,
                ml_rank,
                ctx.selected_global_rows.len(),
            );
            if best_selected
                .as_ref()
                .map(|best| candidate < *best)
                .unwrap_or(true)
            {
                best_selected = Some(candidate);
            }
        }

        let ranked_rank_map = ctx
            .ranked_global_rows
            .iter()
            .enumerate()
            .map(|(i, &row)| (row, i + 1))
            .collect::<HashMap<usize, usize>>();
        let ranked_ranks = site_row_candidates
            .iter()
            .map(|cands| {
                cands
                    .iter()
                    .filter_map(|idx| ranked_rank_map.get(idx).copied())
                    .min()
            })
            .collect::<Vec<_>>();
        let ranked_matched = ranked_ranks.iter().filter(|v| v.is_some()).count();
        if ranked_matched == 0 {
            continue;
        }
        let sum_rank = ranked_ranks.iter().flatten().copied().sum::<usize>();
        let max_rank = ranked_ranks
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(usize::MAX);
        let ml_rank = format_simbench_ml_rank(logic, ranked_ranks.as_slice());
        let candidate = (
            usize::MAX - ranked_matched,
            max_rank,
            sum_rank,
            ctx.unit_index,
            ml_rank,
            ctx.ranked_global_rows.len(),
        );
        if best_ranked
            .as_ref()
            .map(|best| candidate < *best)
            .unwrap_or(true)
        {
            best_ranked = Some(candidate);
        }
    }
    if let Some((_, _, _, _, ml_rank, ml_feature_count)) = best_selected {
        (ml_rank, ml_feature_count)
    } else if let Some((_, _, _, _, ml_rank, ml_feature_count)) = best_ranked {
        (ml_rank, ml_feature_count)
    } else {
        (
            format_simbench_ml_rank(logic, &vec![None; site_row_candidates.len()]),
            0,
        )
    }
}

fn evaluate_simbench_terms(
    prefix: &str,
    terms: &[SimBenchTerm],
    logic_site_lookup: &HashMap<(String, i32), Vec<usize>>,
    selected_sample_indices: &[usize],
    train_idx_local: &[usize],
    assoc_sample_indices: &[usize],
    y_train: &[f64],
    y_assoc: &[f64],
    beam_params: BeamSearchParams,
    ml_contexts: &[GarfieldUnitMlContext],
) -> Result<Vec<GarfieldLogicRuleRecord>, String> {
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let bed = BedSnpIter::new_with_fill(prefix, 0.0, 1.0, false, false, 1.0)?;
    let mut site_lookup = HashMap::<(String, i32), usize>::with_capacity(bed.sites.len());
    for (idx, site) in bed.sites.iter().enumerate() {
        site_lookup
            .entry((normalize_chrom(&site.chrom), site.pos))
            .or_insert(idx);
    }

    let n_selected = selected_sample_indices.len();
    let row_words_full = words_for_samples(n_selected);
    let mut out = Vec::<GarfieldLogicRuleRecord>::with_capacity(terms.len());
    for term in terms.iter() {
        let mut member_ge1_full = Vec::<u64>::with_capacity(term.sites.len() * row_words_full);
        let mut member_ge2_full = Vec::<u64>::with_capacity(term.sites.len() * row_words_full);
        let mut bench_sites = Vec::<SiteInfo>::with_capacity(term.sites.len());
        let mut selected_row_indices = Vec::<usize>::with_capacity(term.sites.len());
        let mut logic_row_candidates = Vec::<Vec<usize>>::with_capacity(term.sites.len());
        for (chrom, pos) in term.sites.iter() {
            let key = (normalize_chrom(chrom), *pos);
            let src_idx = *site_lookup.get(&key).ok_or_else(|| {
                format!(
                    "simbench term {} site not found in BED/BIM: {}:{}",
                    term.term_id, chrom, pos
                )
            })?;
            let (mut row, mut site) = bed
                .get_snp_row_selected_raw(src_idx, selected_sample_indices)
                .ok_or_else(|| {
                    format!(
                        "simbench term {} failed to decode site {}:{} from BED",
                        term.term_id, chrom, pos
                    )
                })?;
            normalize_simbench_raw_row(&mut row, &mut site).map_err(|e| {
                format!(
                    "simbench term {} failed to normalize site {}:{} from BED: {e}",
                    term.term_id, chrom, pos
                )
            })?;
            let (row_ge1, row_ge2) =
                pack_simbench_raw_row_dual_words(row.as_slice()).map_err(|e| {
                    format!(
                        "simbench term {} failed to pack site {}:{} into dual bits: {e}",
                        term.term_id, chrom, pos
                    )
                })?;
            member_ge1_full.extend(row_ge1);
            member_ge2_full.extend(row_ge2);
            let row_candidates = logic_site_lookup.get(&key).cloned().unwrap_or_default();
            bench_sites.push(site);
            selected_row_indices.push(src_idx);
            logic_row_candidates.push(row_candidates);
        }
        let rule = simbench_rule_from_site_count(term.logic, bench_sites.len()).map_err(|e| {
            format!(
                "simbench term {} ('{}' {} {}) failed to build rule: {e}",
                term.term_id, term.kind, term.logic_text, term.sites_text
            )
        })?;
        let n_rule_rows = bench_sites.len();
        let (train_ge1, row_words_train) = packed_rows_subset_from_full_bits_range(
            member_ge1_full.as_slice(),
            row_words_full,
            0,
            n_rule_rows,
            train_idx_local,
            n_rule_rows,
            n_selected,
        )?;
        let (train_ge2, row_words_train_hi) = packed_rows_subset_from_full_bits_range(
            member_ge2_full.as_slice(),
            row_words_full,
            0,
            n_rule_rows,
            train_idx_local,
            n_rule_rows,
            n_selected,
        )?;
        if row_words_train != row_words_train_hi {
            return Err("simbench train dual row-word mismatch".to_string());
        }
        let (assoc_ge1, row_words_assoc) = packed_rows_subset_from_full_bits_range(
            member_ge1_full.as_slice(),
            row_words_full,
            0,
            n_rule_rows,
            assoc_sample_indices,
            n_rule_rows,
            n_selected,
        )?;
        let (assoc_ge2, row_words_assoc_hi) = packed_rows_subset_from_full_bits_range(
            member_ge2_full.as_slice(),
            row_words_full,
            0,
            n_rule_rows,
            assoc_sample_indices,
            n_rule_rows,
            n_selected,
        )?;
        if row_words_assoc != row_words_assoc_hi {
            return Err("simbench assoc dual row-word mismatch".to_string());
        }
        let train_sc = evaluate_rule_continuous_dual(
            &rule,
            y_train,
            train_ge1.as_slice(),
            train_ge2.as_slice(),
            row_words_train,
            n_rule_rows,
            train_idx_local.len(),
            beam_params.lambda_len,
            beam_params.lambda_not,
        )
        .map_err(|e| {
            format!(
                "simbench term {} failed to score train rule: {e}",
                term.term_id
            )
        })?;
        let assoc_sc = evaluate_rule_continuous_dual(
            &rule,
            y_assoc,
            assoc_ge1.as_slice(),
            assoc_ge2.as_slice(),
            row_words_assoc,
            n_rule_rows,
            assoc_sample_indices.len(),
            beam_params.lambda_len,
            beam_params.lambda_not,
        )
        .map_err(|e| {
            format!(
                "simbench term {} failed to score assoc rule: {e}",
                term.term_id
            )
        })?;
        let direct_parent_train_raw = simbench_best_ancestor_raw_baseline_dual(
            &rule,
            y_train,
            train_ge1.as_slice(),
            train_ge2.as_slice(),
            row_words_train,
            n_rule_rows,
            train_idx_local.len(),
            beam_params.lambda_len,
            beam_params.lambda_not,
            beam_params.disable_parent_delta,
        )
        .map_err(|e| {
            format!(
                "simbench term {} failed to build parent baseline on train set: {e}",
                term.term_id
            )
        })?;
        let direct_parent_assoc_raw = simbench_best_ancestor_raw_baseline_dual(
            &rule,
            y_assoc,
            assoc_ge1.as_slice(),
            assoc_ge2.as_slice(),
            row_words_assoc,
            n_rule_rows,
            assoc_sample_indices.len(),
            beam_params.lambda_len,
            beam_params.lambda_not,
            beam_params.disable_parent_delta,
        )
        .map_err(|e| {
            format!(
                "simbench term {} failed to build parent baseline on assoc set: {e}",
                term.term_id
            )
        })?;
        let sim_expr_txt = simbench_rule_expr(term.logic, bench_sites.as_slice())?;
        let train_bucket = bucket_from_rule(&rule, train_sc.dosage_maf);
        let _train_score = rank_rule_score_components_with_bucket(
            train_bucket,
            rule.len(),
            rule.not_count(),
            train_sc.raw_score,
            direct_parent_train_raw,
            &beam_params,
            true,
        );
        let test_bucket = bucket_from_rule(&rule, assoc_sc.dosage_maf);
        let test_score = rank_rule_score_components_with_bucket(
            test_bucket,
            rule.len(),
            rule.not_count(),
            assoc_sc.raw_score,
            direct_parent_assoc_raw,
            &beam_params,
            false,
        );
        let (full_ge1, full_ge2) = materialize_rule_bits_dual(
            &rule,
            member_ge1_full.as_slice(),
            member_ge2_full.as_slice(),
            row_words_full,
            n_rule_rows,
            n_selected,
        )
        .map_err(|e| {
            format!(
                "simbench term {} failed to materialize full support bits: {e}",
                term.term_id
            )
        })?;
        let first_site = bench_sites
            .first()
            .ok_or_else(|| format!("simbench term {} has no sites", term.term_id))?;
        let (ml_rank, ml_feature_count) =
            resolve_simbench_ml_rank(term.logic, logic_row_candidates.as_slice(), ml_contexts);
        let (bim_allele0, bim_allele1) = simbench_rule_bim_alleles(bench_sites.as_slice());
        out.push(GarfieldLogicRuleRecord {
            unit_name: simbench_rule_name(term.logic, bench_sites.as_slice(), &term.label),
            unit_kind: "simbench".to_string(),
            unit_index: term.term_id,
            region_size: bench_sites.len(),
            ml_feature_count,
            ml_rank,
            selected_row_indices,
            display_ops: simbench_logic_to_beam_op(term.logic)
                .map(|op| vec![op; bench_sites.len().saturating_sub(1)])
                .unwrap_or_default(),
            display_negated: vec![false; bench_sites.len()],
            snp_name: simbench_rule_name(term.logic, bench_sites.as_slice(), &term.label),
            expr: sim_expr_txt,
            chrom_field: first_site.chrom.clone(),
            bim_snp_name: simbench_rule_bim_name(term.logic, bench_sites.as_slice(), &term.label),
            bim_allele0,
            bim_allele1,
            pos: first_site.pos,
            score: test_score,
            delta_score: format!(
                "{}->{}",
                format_delta_metric_value(test_score),
                format_delta_metric_value(test_score)
            ),
            support_bits: Some(GarfieldRuleSupportBits::Dual {
                ge1: full_ge1.into_boxed_slice(),
                ge2: full_ge2.into_boxed_slice(),
            }),
        });
    }
    Ok(out)
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
        "gene" => 1,
        "geneset" => 2,
        "simbench" => 255,
        _ => 200,
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
    let mut best_by_bits = HashMap::<GarfieldRuleSupportBits, GarfieldLogicRuleRecord>::new();
    for rec in records.into_iter() {
        let support = materialize_logic_rule_record_support_bits(&rec, logic_bits)?;
        match best_by_bits.entry(support) {
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
fn parse_logic_record_primary_site(rec: &GarfieldLogicRuleRecord) -> Option<(String, i32)> {
    let first = rec.snp_name.split('&').next()?.trim();
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

#[inline]
fn mask_to_indices(mask: &[bool], choose_value: bool) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &v)| if v == choose_value { Some(i) } else { None })
        .collect()
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

fn literal_name<S: GarfieldChromPosSite>(site: &S, negated: bool) -> String {
    let base = site_base_label(site);
    if negated {
        format!("!{base}")
    } else {
        base
    }
}

#[inline]
fn simbench_logic_symbol_compact(logic: SimBenchLogic) -> &'static str {
    match logic {
        SimBenchLogic::Single => "",
        SimBenchLogic::And => "&",
        SimBenchLogic::Or => "|",
    }
}

#[inline]
fn simbench_logic_to_beam_op(logic: SimBenchLogic) -> Option<BeamBinaryOp> {
    match logic {
        SimBenchLogic::Single => None,
        SimBenchLogic::And => Some(BeamBinaryOp::And),
        SimBenchLogic::Or => Some(BeamBinaryOp::Or),
    }
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

fn materialize_logic_rule_record_support_bits(
    rec: &GarfieldLogicRuleRecord,
    logic_bits: &GarfieldLogicBits,
) -> Result<GarfieldRuleSupportBits, String> {
    if let Some(bits) = rec.support_bits.as_ref() {
        return Ok(bits.clone());
    }
    let local_rule = logic_rule_record_local_rule(rec)?;
    let n_rows = rec.selected_row_indices.len();
    let row_words = logic_bits.row_words;
    let mut bits_flat = Vec::<u64>::with_capacity(n_rows.saturating_mul(row_words));
    for &global_row_idx in rec.selected_row_indices.iter() {
        let row_start = global_row_idx.saturating_mul(row_words);
        let row_end = row_start.saturating_add(row_words);
        let row = logic_bits
            .bits_flat
            .get(row_start..row_end)
            .ok_or_else(|| {
                format!(
                    "GARFIELD stored rule '{}' row slice out of range: row={} row_words={}",
                    rec.snp_name, global_row_idx, row_words
                )
            })?;
        bits_flat.extend_from_slice(row);
    }
    if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
        let mut bits_hi = Vec::<u64>::with_capacity(n_rows.saturating_mul(row_words));
        for &global_row_idx in rec.selected_row_indices.iter() {
            let row_start = global_row_idx.saturating_mul(row_words);
            let row_end = row_start.saturating_add(row_words);
            let row = bits_hi_flat.get(row_start..row_end).ok_or_else(|| {
                format!(
                    "GARFIELD stored rule '{}' high-bit row slice out of range: row={} row_words={}",
                    rec.snp_name, global_row_idx, row_words
                )
            })?;
            bits_hi.extend_from_slice(row);
        }
        let (ge1, ge2) = materialize_rule_bits_dual(
            &local_rule,
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
            &local_rule,
            bits_flat.as_slice(),
            row_words,
            n_rows,
            logic_bits.n_samples,
        )?;
        Ok(GarfieldRuleSupportBits::Binary(bits.into_boxed_slice()))
    }
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
        "gene" | "geneset" | "group" => {
            let g = groups.ok_or_else(|| {
                "groups must be provided for unit_kind in {gene, geneset, group}".to_string()
            })?;
            Ok(build_logic_units_from_groups(sites, g, group_names))
        }
        other => Err(format!(
            "unit_kind must be one of: window, gene, geneset, group; got {other}"
        )),
    }
}

struct LogicSamplePlanEntry {
    src_byte_idx: usize,
    src_shifts: [u8; 4],
    dst_word_idx: [usize; 4],
    dst_bit: [u64; 4],
    n_lanes: u8,
}

fn build_logic_sample_plan(sample_indices: &[usize]) -> Vec<LogicSamplePlanEntry> {
    let mut lanes = sample_indices
        .iter()
        .enumerate()
        .map(|(dst_idx, &src_idx)| {
            (
                src_idx >> 2,
                ((src_idx & 3) * 2) as u8,
                dst_idx >> 6,
                1u64 << (dst_idx & 63),
            )
        })
        .collect::<Vec<_>>();
    lanes.sort_unstable_by_key(|&(src_byte_idx, _, _, _)| src_byte_idx);

    let mut plan = Vec::<LogicSamplePlanEntry>::with_capacity(lanes.len().div_ceil(4));
    let mut cur_src_byte_idx = usize::MAX;
    let mut cur = LogicSamplePlanEntry {
        src_byte_idx: 0,
        src_shifts: [0; 4],
        dst_word_idx: [0; 4],
        dst_bit: [0; 4],
        n_lanes: 0,
    };

    for (src_byte_idx, src_shift, dst_word_idx, dst_bit) in lanes {
        if src_byte_idx != cur_src_byte_idx {
            if cur_src_byte_idx != usize::MAX {
                plan.push(cur);
            }
            cur_src_byte_idx = src_byte_idx;
            cur = LogicSamplePlanEntry {
                src_byte_idx,
                src_shifts: [0; 4],
                dst_word_idx: [0; 4],
                dst_bit: [0; 4],
                n_lanes: 0,
            };
        }
        let lane = cur.n_lanes as usize;
        cur.src_shifts[lane] = src_shift;
        cur.dst_word_idx[lane] = dst_word_idx;
        cur.dst_bit[lane] = dst_bit;
        cur.n_lanes += 1;
    }
    if cur_src_byte_idx != usize::MAX {
        plan.push(cur);
    }
    plan
}

#[inline]
fn decode_logic_bin_genotype(code: u8, flip: bool) -> Option<u8> {
    match code {
        0b00 => Some(if flip { 2 } else { 0 }),
        0b10 => Some(1),
        0b11 => Some(if flip { 0 } else { 2 }),
        _ => None,
    }
}

#[inline]
#[allow(dead_code)]
fn fill_bin_logic_row_bits(
    row: &[u8],
    flip: bool,
    sample_plan: &[LogicSamplePlanEntry],
    dst_words: &mut [u64],
) {
    let mut c0 = 0usize;
    let mut c2 = 0usize;
    for entry in sample_plan.iter() {
        let byte = row[entry.src_byte_idx];
        for lane in 0..entry.n_lanes as usize {
            match decode_logic_bin_genotype((byte >> entry.src_shifts[lane]) & 0b11, flip) {
                Some(0) => c0 += 1,
                Some(2) => c2 += 1,
                _ => {}
            }
        }
    }
    let mode02 = if c2 > c0 { 2u8 } else { 0u8 };
    for entry in sample_plan.iter() {
        let byte = row[entry.src_byte_idx];
        for lane in 0..entry.n_lanes as usize {
            let is_one =
                match decode_logic_bin_genotype((byte >> entry.src_shifts[lane]) & 0b11, flip) {
                    Some(0) => false,
                    Some(2) => true,
                    _ => mode02 == 2,
                };
            if is_one {
                dst_words[entry.dst_word_idx[lane]] |= entry.dst_bit[lane];
            }
        }
    }
}

#[inline]
#[allow(dead_code)]
fn fill_bin_logic_row_bits_pure_line(
    row: &[u8],
    flip: bool,
    sample_plan: &[LogicSamplePlanEntry],
    dst_words: &mut [u64],
) {
    for entry in sample_plan.iter() {
        let byte = row[entry.src_byte_idx];
        for lane in 0..entry.n_lanes as usize {
            let code = (byte >> entry.src_shifts[lane]) & 0b11;
            let is_one = if flip { code == 0b00 } else { code == 0b11 };
            if is_one {
                dst_words[entry.dst_word_idx[lane]] |= entry.dst_bit[lane];
            }
        }
    }
}

#[inline]
fn fill_bin_logic_row_bits_fuzzy(
    row: &[u8],
    flip: bool,
    sample_plan: &[LogicSamplePlanEntry],
    ge1_words: &mut [u64],
    ge2_words: &mut [u64],
) {
    let mut c0 = 0usize;
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    for entry in sample_plan.iter() {
        let byte = row[entry.src_byte_idx];
        for lane in 0..entry.n_lanes as usize {
            match decode_logic_bin_genotype((byte >> entry.src_shifts[lane]) & 0b11, flip) {
                Some(0) => c0 += 1,
                Some(1) => c1 += 1,
                Some(2) => c2 += 1,
                _ => {}
            }
        }
    }
    let mode3 = if c2 >= c1 && c2 >= c0 {
        2u8
    } else if c1 >= c0 {
        1u8
    } else {
        0u8
    };
    for entry in sample_plan.iter() {
        let byte = row[entry.src_byte_idx];
        for lane in 0..entry.n_lanes as usize {
            let g = decode_logic_bin_genotype((byte >> entry.src_shifts[lane]) & 0b11, flip)
                .unwrap_or(mode3);
            if g >= 1 {
                ge1_words[entry.dst_word_idx[lane]] |= entry.dst_bit[lane];
            }
            if g >= 2 {
                ge2_words[entry.dst_word_idx[lane]] |= entry.dst_bit[lane];
            }
        }
    }
}

#[inline]
fn fill_mbin_logic_row_bits(
    row: &[u8],
    flip: bool,
    sample_plan: &[LogicSamplePlanEntry],
    row_words: usize,
    dst_words: &mut [u64],
) {
    let mut c0 = 0usize;
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    for entry in sample_plan.iter() {
        let byte = row[entry.src_byte_idx];
        for lane in 0..entry.n_lanes as usize {
            match decode_logic_bin_genotype((byte >> entry.src_shifts[lane]) & 0b11, flip) {
                Some(0) => c0 += 1,
                Some(1) => c1 += 1,
                Some(2) => c2 += 1,
                _ => {}
            }
        }
    }
    let mode3 = if c2 >= c1 && c2 >= c0 {
        2u8
    } else if c1 >= c0 {
        1u8
    } else {
        0u8
    };
    let (dom_bits, rem) = dst_words.split_at_mut(row_words);
    let (rec_bits, het_bits) = rem.split_at_mut(row_words);
    for entry in sample_plan.iter() {
        let byte = row[entry.src_byte_idx];
        for lane in 0..entry.n_lanes as usize {
            let g = match decode_logic_bin_genotype((byte >> entry.src_shifts[lane]) & 0b11, flip) {
                Some(g) => g,
                _ => mode3,
            };
            if g > 0 {
                dom_bits[entry.dst_word_idx[lane]] |= entry.dst_bit[lane];
            }
            if g == 2 {
                rec_bits[entry.dst_word_idx[lane]] |= entry.dst_bit[lane];
            }
            if g == 1 {
                het_bits[entry.dst_word_idx[lane]] |= entry.dst_bit[lane];
            }
        }
    }
}

#[allow(dead_code)]
fn convert_prepared_bed_to_logic_bits(
    packed: &[u8],
    bytes_per_snp: usize,
    n_samples_total: usize,
    row_flip: &[bool],
    sites: &[SiteInfo],
    sample_indices: &[usize],
    sample_ids: &[String],
    mode: GarfieldBinMode,
) -> Result<GarfieldLogicBits, String> {
    if bytes_per_snp == 0 {
        return Err("bytes_per_snp must be > 0".to_string());
    }
    if packed.len() != sites.len().saturating_mul(bytes_per_snp) {
        return Err(format!(
            "packed/site length mismatch: packed={}, sites={}, bytes_per_snp={bytes_per_snp}",
            packed.len(),
            sites.len()
        ));
    }
    if row_flip.len() != sites.len() {
        return Err(format!(
            "row_flip/site length mismatch: row_flip={}, sites={}",
            row_flip.len(),
            sites.len()
        ));
    }
    if sample_ids.len() != sample_indices.len() {
        return Err(format!(
            "sample id count mismatch: got {}, expected {}",
            sample_ids.len(),
            sample_indices.len()
        ));
    }
    for &idx in sample_indices.iter() {
        if idx >= n_samples_total {
            return Err(format!(
                "sample index out of range while converting packed BED: {idx} >= {n_samples_total}"
            ));
        }
    }

    let n_samples = sample_indices.len();
    let row_words = words_for_samples(n_samples);
    let sample_plan = build_logic_sample_plan(sample_indices);
    let row_mul = match mode {
        GarfieldBinMode::Bin => 1usize,
        GarfieldBinMode::Mbin => 3usize,
    };
    let group_ids = build_logic_mode_group_ids(sites.len(), mode);
    let out_sites = build_logic_sites_from_metadata(sites, mode);
    let n_rows = out_sites.len();
    let mut bits_flat = vec![0u64; n_rows.saturating_mul(row_words)];
    let mut bits_hi_flat = match mode {
        GarfieldBinMode::Bin => Some(vec![0u64; n_rows.saturating_mul(row_words)]),
        GarfieldBinMode::Mbin => None,
    };
    match bits_hi_flat.as_mut() {
        Some(bits_hi) => {
            bits_flat
                .par_chunks_mut(row_words)
                .zip(bits_hi.par_chunks_mut(row_words))
                .zip(row_flip.par_iter().copied())
                .zip(packed.par_chunks(bytes_per_snp))
                .for_each(|(((dst_rows, dst_hi_rows), flip), row)| {
                    fill_bin_logic_row_bits_fuzzy(
                        row,
                        flip,
                        sample_plan.as_slice(),
                        dst_rows,
                        dst_hi_rows,
                    )
                });
        }
        None => {
            bits_flat
                .par_chunks_mut(row_mul * row_words)
                .zip(row_flip.par_iter().copied())
                .zip(packed.par_chunks(bytes_per_snp))
                .for_each(|((dst_rows, flip), row)| {
                    fill_mbin_logic_row_bits(row, flip, sample_plan.as_slice(), row_words, dst_rows)
                });
        }
    }

    Ok(GarfieldLogicBits {
        bits_flat,
        bits_hi_flat,
        row_words,
        sample_ids: sample_ids.to_vec(),
        sites: out_sites,
        group_ids,
        n_samples,
    })
}

fn convert_bed_prefix_to_logic_bits(
    prefix: &str,
    bytes_per_snp: usize,
    n_samples_total: usize,
    row_source_indices: &[usize],
    row_flip: &[bool],
    sites: Vec<SiteInfo>,
    sample_indices: &[usize],
    sample_ids: &[String],
    mode: GarfieldBinMode,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<GarfieldLogicBits, String> {
    if bytes_per_snp == 0 {
        return Err("bytes_per_snp must be > 0".to_string());
    }
    if sample_ids.len() != sample_indices.len() {
        return Err(format!(
            "sample id count mismatch: got {}, expected {}",
            sample_ids.len(),
            sample_indices.len()
        ));
    }
    for &idx in sample_indices.iter() {
        if idx >= n_samples_total {
            return Err(format!(
                "sample index out of range while converting BED to logic bits: {idx} >= {n_samples_total}"
            ));
        }
    }
    if row_source_indices.len() != sites.len() {
        return Err(format!(
            "row_source_indices/site metadata mismatch: rows={}, filtered sites={}",
            row_source_indices.len(),
            sites.len()
        ));
    }
    if row_flip.len() != sites.len() {
        return Err(format!(
            "row_flip/site metadata mismatch: row_flip={}, filtered sites={}",
            row_flip.len(),
            sites.len()
        ));
    }
    if row_source_indices.is_empty() {
        return Err("row_source_indices keeps zero SNPs".to_string());
    }

    let n_samples = sample_indices.len();
    let row_words = words_for_samples(n_samples);
    let sample_plan = build_logic_sample_plan(sample_indices);
    let row_mul = match mode {
        GarfieldBinMode::Bin => 1usize,
        GarfieldBinMode::Mbin => 3usize,
    };
    let group_ids = build_logic_mode_group_ids(sites.len(), mode);
    let out_sites = build_logic_sites_from_metadata(sites.as_slice(), mode);
    drop(sites);
    let n_rows = out_sites.len();
    let mut bits_flat = vec![0u64; n_rows.saturating_mul(row_words)];
    let mut bits_hi_flat = match mode {
        GarfieldBinMode::Bin => Some(vec![0u64; n_rows.saturating_mul(row_words)]),
        GarfieldBinMode::Mbin => None,
    };
    if let Some(tracker) = mem_tracker {
        tracker.sample_now();
    }

    let bed_prefix = normalize_plink_prefix(prefix);
    let mut bed_iter =
        BedSnpIter::new_for_grm_window(bed_prefix.as_str(), garfield_logic_mmap_window_mb())?;
    if bed_iter.n_samples() != n_samples_total {
        return Err(format!(
            "BED sample count mismatch while converting to logic bits: bed={}, expected={}",
            bed_iter.n_samples(),
            n_samples_total
        ));
    }
    if bed_iter.n_snps() == 0 {
        return Err("no SNP sites found in PLINK BED input".to_string());
    }
    if bed_iter.bytes_per_snp() != bytes_per_snp {
        return Err(format!(
            "BED bytes_per_snp mismatch while converting to logic bits: bed={}, expected={}",
            bed_iter.bytes_per_snp(),
            bytes_per_snp
        ));
    }
    if let Some(&max_row) = row_source_indices.last() {
        if max_row >= bed_iter.n_snps() {
            return Err(format!(
                "row_source_indices out of range while converting BED to logic bits: max_row={} >= n_snps={}",
                max_row,
                bed_iter.n_snps()
            ));
        }
    }

    let mut kept_start = 0usize;
    while kept_start < row_source_indices.len() {
        check_ctrlc()?;
        let src_row_start = row_source_indices[kept_start];
        bed_iter.ensure_window_for_snp(src_row_start)?;
        let mapped = bed_iter.mapped_contiguous_snps_from(src_row_start);
        if mapped == 0 {
            return Err(format!(
                "windowed BED logic conversion reached empty window at source row {src_row_start}"
            ));
        }
        let src_row_end = src_row_start.saturating_add(mapped);
        let mut kept_end = kept_start;
        while kept_end < row_source_indices.len() && row_source_indices[kept_end] < src_row_end {
            kept_end += 1;
        }
        if kept_end == kept_start {
            return Err(format!(
                "windowed BED logic conversion made no progress at source row {src_row_start}"
            ));
        }

        let bed_ref: &BedSnpIter = &bed_iter;
        let dst_lo =
            &mut bits_flat[kept_start * row_mul * row_words..kept_end * row_mul * row_words];
        let dst_hi_opt = bits_hi_flat
            .as_mut()
            .map(|v| &mut v[kept_start * row_mul * row_words..kept_end * row_mul * row_words]);
        if let Some(dst_hi) = dst_hi_opt {
            dst_lo
                .par_chunks_mut(row_mul * row_words)
                .zip(dst_hi.par_chunks_mut(row_words))
                .enumerate()
                .for_each(|(off, (dst_rows, dst_hi_rows))| {
                    let kept_idx = kept_start + off;
                    let src_row = row_source_indices[kept_idx];
                    let flip = row_flip[kept_idx];
                    let row = bed_ref
                        .packed_snp_bytes_at(src_row)
                        .expect("windowed BED logic conversion row should be mapped");
                    fill_bin_logic_row_bits_fuzzy(
                        row,
                        flip,
                        sample_plan.as_slice(),
                        dst_rows,
                        dst_hi_rows,
                    );
                });
        } else {
            dst_lo
                .par_chunks_mut(row_mul * row_words)
                .enumerate()
                .for_each(|(off, dst_rows)| {
                    let kept_idx = kept_start + off;
                    let src_row = row_source_indices[kept_idx];
                    let flip = row_flip[kept_idx];
                    let row = bed_ref
                        .packed_snp_bytes_at(src_row)
                        .expect("windowed BED logic conversion row should be mapped");
                    fill_mbin_logic_row_bits(
                        row,
                        flip,
                        sample_plan.as_slice(),
                        row_words,
                        dst_rows,
                    );
                });
        }
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        kept_start = kept_end;
    }

    Ok(GarfieldLogicBits {
        bits_flat,
        bits_hi_flat,
        row_words,
        sample_ids: sample_ids.to_vec(),
        sites: out_sites,
        group_ids,
        n_samples,
    })
}

fn dense_dosage_rows_from_full_bits_range(
    bits_flat: &[u64],
    bits_hi_flat: Option<&[u64]>,
    row_words: usize,
    row_start: usize,
    row_end: usize,
    sample_indices: &[usize],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<Vec<Vec<u8>>, String> {
    let t0 = Instant::now();
    if row_end <= row_start {
        return Ok(Vec::new());
    }
    if row_end > n_rows_all {
        return Err(format!(
            "row range out of bounds while densifying: [{row_start}, {row_end}) vs n_rows={n_rows_all}"
        ));
    }
    if row_words != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words) {
        return Err("full bit matrix length mismatch".to_string());
    }
    if let Some(bits_hi_flat) = bits_hi_flat {
        if bits_hi_flat.len() != n_rows_all.saturating_mul(row_words) {
            return Err("full high-bit matrix length mismatch".to_string());
        }
    }
    if let Some(&sid) = sample_indices.iter().find(|&&sid| sid >= n_samples_all) {
        return Err(format!("sample index out of range while densifying: {sid}"));
    }
    let n_rows = row_end.saturating_sub(row_start);
    let out = if should_parallel_dense_decode(n_rows, sample_indices.len()) {
        (row_start..row_end)
            .into_par_iter()
            .map(|row_idx| {
                let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                let row_ge2 =
                    bits_hi_flat.map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
                dense_dosage_row_from_words(row_ge1, row_ge2, sample_indices)
            })
            .collect()
    } else {
        let mut out = Vec::<Vec<u8>>::with_capacity(n_rows);
        for row_idx in row_start..row_end {
            let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
            let row_ge2 =
                bits_hi_flat.map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
            out.push(dense_dosage_row_from_words(
                row_ge1,
                row_ge2,
                sample_indices,
            ));
        }
        out
    };
    GARFIELD_DENSE_DOSAGE_DECODE_NS.fetch_add(elapsed_ns_saturating(t0), Ordering::Relaxed);
    Ok(out)
}

fn dense_dosage_rows_from_full_bits(
    bits_flat: &[u64],
    bits_hi_flat: Option<&[u64]>,
    row_words: usize,
    row_indices: &[usize],
    sample_indices: &[usize],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<Vec<Vec<u8>>, String> {
    let t0 = Instant::now();
    if row_words != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words) {
        return Err("full bit matrix length mismatch".to_string());
    }
    if let Some(bits_hi_flat) = bits_hi_flat {
        if bits_hi_flat.len() != n_rows_all.saturating_mul(row_words) {
            return Err("full high-bit matrix length mismatch".to_string());
        }
    }
    if let Some(&row_idx) = row_indices.iter().find(|&&row_idx| row_idx >= n_rows_all) {
        return Err(format!(
            "row index out of range while densifying: {row_idx}"
        ));
    }
    if let Some(&sid) = sample_indices.iter().find(|&&sid| sid >= n_samples_all) {
        return Err(format!("sample index out of range while densifying: {sid}"));
    }
    let out = if should_parallel_dense_decode(row_indices.len(), sample_indices.len()) {
        row_indices
            .par_iter()
            .map(|&row_idx| {
                let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                let row_ge2 =
                    bits_hi_flat.map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
                dense_dosage_row_from_words(row_ge1, row_ge2, sample_indices)
            })
            .collect()
    } else {
        let mut out = Vec::<Vec<u8>>::with_capacity(row_indices.len());
        for &row_idx in row_indices.iter() {
            let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
            let row_ge2 =
                bits_hi_flat.map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
            out.push(dense_dosage_row_from_words(
                row_ge1,
                row_ge2,
                sample_indices,
            ));
        }
        out
    };
    GARFIELD_DENSE_DOSAGE_DECODE_NS.fetch_add(elapsed_ns_saturating(t0), Ordering::Relaxed);
    Ok(out)
}

#[inline]
fn should_parallel_dense_decode(n_rows: usize, n_samples: usize) -> bool {
    rayon::current_num_threads() > 1
        && n_rows >= GARFIELD_DENSE_DECODE_PAR_MIN_ROWS
        && n_samples >= GARFIELD_DENSE_DECODE_PAR_MIN_SAMPLES
}

#[inline]
fn dense_dosage_row_from_words(
    row_ge1: &[u64],
    row_ge2: Option<&[u64]>,
    sample_indices: &[usize],
) -> Vec<u8> {
    let mut dense = vec![0u8; sample_indices.len()];
    for (j, &sid) in sample_indices.iter().enumerate() {
        let ge1 = ((row_ge1[sid >> 6] >> (sid & 63)) & 1u64) as u8;
        let ge2 = row_ge2
            .map(|row| ((row[sid >> 6] >> (sid & 63)) & 1u64) as u8)
            .unwrap_or(0);
        dense[j] = ge1.saturating_add(ge2);
    }
    dense
}

#[inline]
fn sample_indices_are_full_identity(sample_indices: &[usize], n_samples_all: usize) -> bool {
    sample_indices.len() == n_samples_all
        && sample_indices.iter().enumerate().all(|(i, &sid)| sid == i)
}

#[inline]
fn dosage_stage1_stats_for_row_words(
    row_ge1: &[u64],
    row_ge2: Option<&[u64]>,
    sample_indices: &[usize],
    y: &[f64],
) -> (f64, f64, f64) {
    let mut sum_x = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_xy = 0.0f64;
    for (dst_s, &src_s) in sample_indices.iter().enumerate() {
        let ge1 = ((row_ge1[src_s >> 6] >> (src_s & 63)) & 1u64) as u8;
        let ge2 = row_ge2
            .map(|row| ((row[src_s >> 6] >> (src_s & 63)) & 1u64) as u8)
            .unwrap_or(0);
        let dosage = f64::from(ge1.saturating_add(ge2));
        sum_x += dosage;
        sum_x2 += dosage * dosage;
        sum_xy += dosage * y[dst_s];
    }
    (sum_x, sum_x2, sum_xy)
}

fn dosage_stage1_stats_from_full_bits(
    bits_flat: &[u64],
    bits_hi_flat: Option<&[u64]>,
    row_words: usize,
    row_indices: &[usize],
    sample_indices: &[usize],
    y: &[f64],
    n_rows_all: usize,
    n_samples_all: usize,
    allow_parallel: bool,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if row_words != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words) {
        return Err("full bit matrix length mismatch".to_string());
    }
    if let Some(bits_hi_flat) = bits_hi_flat {
        if bits_hi_flat.len() != n_rows_all.saturating_mul(row_words) {
            return Err("full high-bit matrix length mismatch".to_string());
        }
    }
    if sample_indices.len() != y.len() {
        return Err(format!(
            "sample_indices length ({}) != y length ({}) while computing dosage stage-1 stats",
            sample_indices.len(),
            y.len()
        ));
    }
    if let Some(&row_idx) = row_indices.iter().find(|&&row_idx| row_idx >= n_rows_all) {
        return Err(format!(
            "row index out of range while computing dosage stage-1 stats: {row_idx}"
        ));
    }
    if let Some(&sid) = sample_indices.iter().find(|&&sid| sid >= n_samples_all) {
        return Err(format!(
            "sample index out of range while computing dosage stage-1 stats: {sid}"
        ));
    }
    let stats: Vec<(f64, f64, f64)> =
        if sample_indices_are_full_identity(sample_indices, n_samples_all) {
            let zero_hi = if bits_hi_flat.is_none() {
                Some(vec![0u64; row_words])
            } else {
                None
            };
            if allow_parallel
                && should_parallel_dense_decode(row_indices.len(), sample_indices.len())
            {
                row_indices
                    .par_iter()
                    .map(|&row_idx| {
                        let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                        let row_ge2 = bits_hi_flat
                            .map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words])
                            .unwrap_or_else(|| zero_hi.as_deref().expect("zero hi row must exist"));
                        let (n_ge1, n_ge2, sum_ge1, sum_ge2) =
                            dual_packed_summary(row_ge1, row_ge2, y, n_samples_all);
                        (
                            (n_ge1 + n_ge2) as f64,
                            (n_ge1 + 3 * n_ge2) as f64,
                            sum_ge1 + sum_ge2,
                        )
                    })
                    .collect()
            } else {
                row_indices
                    .iter()
                    .map(|&row_idx| {
                        let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                        let row_ge2 = bits_hi_flat
                            .map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words])
                            .unwrap_or_else(|| zero_hi.as_deref().expect("zero hi row must exist"));
                        let (n_ge1, n_ge2, sum_ge1, sum_ge2) =
                            dual_packed_summary(row_ge1, row_ge2, y, n_samples_all);
                        (
                            (n_ge1 + n_ge2) as f64,
                            (n_ge1 + 3 * n_ge2) as f64,
                            sum_ge1 + sum_ge2,
                        )
                    })
                    .collect()
            }
        } else if allow_parallel
            && should_parallel_dense_decode(row_indices.len(), sample_indices.len())
        {
            row_indices
                .par_iter()
                .map(|&row_idx| {
                    let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                    let row_ge2 = bits_hi_flat
                        .map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
                    dosage_stage1_stats_for_row_words(row_ge1, row_ge2, sample_indices, y)
                })
                .collect()
        } else {
            row_indices
                .iter()
                .map(|&row_idx| {
                    let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                    let row_ge2 = bits_hi_flat
                        .map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
                    dosage_stage1_stats_for_row_words(row_ge1, row_ge2, sample_indices, y)
                })
                .collect()
        };
    let mut feat_sum_x = Vec::<f64>::with_capacity(stats.len());
    let mut feat_sum_x2 = Vec::<f64>::with_capacity(stats.len());
    let mut feat_sum_xy = Vec::<f64>::with_capacity(stats.len());
    for (sum_x, sum_x2, sum_xy) in stats.into_iter() {
        feat_sum_x.push(sum_x);
        feat_sum_x2.push(sum_x2);
        feat_sum_xy.push(sum_xy);
    }
    Ok((feat_sum_x, feat_sum_x2, feat_sum_xy))
}

fn dosage_stage1_stats_from_full_bits_range(
    bits_flat: &[u64],
    bits_hi_flat: Option<&[u64]>,
    row_words: usize,
    row_start: usize,
    row_end: usize,
    sample_indices: &[usize],
    y: &[f64],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if row_end <= row_start {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    if row_end > n_rows_all {
        return Err(format!(
            "row range out of bounds while computing dosage stage-1 stats: [{row_start}, {row_end}) vs n_rows={n_rows_all}"
        ));
    }
    if row_words != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words) {
        return Err("full bit matrix length mismatch".to_string());
    }
    if let Some(bits_hi_flat) = bits_hi_flat {
        if bits_hi_flat.len() != n_rows_all.saturating_mul(row_words) {
            return Err("full high-bit matrix length mismatch".to_string());
        }
    }
    if sample_indices.len() != y.len() {
        return Err(format!(
            "sample_indices length ({}) != y length ({}) while computing dosage stage-1 stats from range",
            sample_indices.len(),
            y.len()
        ));
    }
    if let Some(&sid) = sample_indices.iter().find(|&&sid| sid >= n_samples_all) {
        return Err(format!(
            "sample index out of range while computing dosage stage-1 stats: {sid}"
        ));
    }
    let n_rows = row_end.saturating_sub(row_start);
    let stats: Vec<(f64, f64, f64)> =
        if sample_indices_are_full_identity(sample_indices, n_samples_all) {
            let zero_hi = if bits_hi_flat.is_none() {
                Some(vec![0u64; row_words])
            } else {
                None
            };
            if should_parallel_dense_decode(n_rows, sample_indices.len()) {
                (row_start..row_end)
                    .into_par_iter()
                    .map(|row_idx| {
                        let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                        let row_ge2 = bits_hi_flat
                            .map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words])
                            .unwrap_or_else(|| zero_hi.as_deref().expect("zero hi row must exist"));
                        let (n_ge1, n_ge2, sum_ge1, sum_ge2) =
                            dual_packed_summary(row_ge1, row_ge2, y, n_samples_all);
                        (
                            (n_ge1 + n_ge2) as f64,
                            (n_ge1 + 3 * n_ge2) as f64,
                            sum_ge1 + sum_ge2,
                        )
                    })
                    .collect()
            } else {
                (row_start..row_end)
                    .map(|row_idx| {
                        let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                        let row_ge2 = bits_hi_flat
                            .map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words])
                            .unwrap_or_else(|| zero_hi.as_deref().expect("zero hi row must exist"));
                        let (n_ge1, n_ge2, sum_ge1, sum_ge2) =
                            dual_packed_summary(row_ge1, row_ge2, y, n_samples_all);
                        (
                            (n_ge1 + n_ge2) as f64,
                            (n_ge1 + 3 * n_ge2) as f64,
                            sum_ge1 + sum_ge2,
                        )
                    })
                    .collect()
            }
        } else if should_parallel_dense_decode(n_rows, sample_indices.len()) {
            (row_start..row_end)
                .into_par_iter()
                .map(|row_idx| {
                    let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                    let row_ge2 = bits_hi_flat
                        .map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
                    dosage_stage1_stats_for_row_words(row_ge1, row_ge2, sample_indices, y)
                })
                .collect()
        } else {
            (row_start..row_end)
                .map(|row_idx| {
                    let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                    let row_ge2 = bits_hi_flat
                        .map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
                    dosage_stage1_stats_for_row_words(row_ge1, row_ge2, sample_indices, y)
                })
                .collect()
        };
    let mut feat_sum_x = Vec::<f64>::with_capacity(stats.len());
    let mut feat_sum_x2 = Vec::<f64>::with_capacity(stats.len());
    let mut feat_sum_xy = Vec::<f64>::with_capacity(stats.len());
    for (sum_x, sum_x2, sum_xy) in stats.into_iter() {
        feat_sum_x.push(sum_x);
        feat_sum_x2.push(sum_x2);
        feat_sum_xy.push(sum_xy);
    }
    Ok((feat_sum_x, feat_sum_x2, feat_sum_xy))
}

static PACKED_EXTRACT_FLAT_NS: AtomicU64 = AtomicU64::new(0);

#[inline]
fn elapsed_ns_saturating(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn packed_dosage_rows_subset_from_full_bits_with_stage1(
    bits_flat: &[u64],
    bits_hi_flat: Option<&[u64]>,
    row_words_full: usize,
    row_indices: &[usize],
    sample_indices: &[usize],
    y: &[f64],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<(Vec<u64>, Vec<u64>, usize, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let _t = Instant::now();
    if row_words_full != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words_full}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words_full) {
        return Err("full bit matrix length mismatch".to_string());
    }
    if let Some(bits_hi_flat) = bits_hi_flat {
        if bits_hi_flat.len() != n_rows_all.saturating_mul(row_words_full) {
            return Err("full high-bit matrix length mismatch".to_string());
        }
    }
    if sample_indices.len() != y.len() {
        return Err(format!(
            "sample_indices length ({}) != y length ({}) while subsetting packed rows",
            sample_indices.len(),
            y.len()
        ));
    }
    let row_words_sub = words_for_samples(sample_indices.len());
    let mut out_ge1 = vec![0u64; row_indices.len().saturating_mul(row_words_sub)];
    let mut out_ge2 = vec![0u64; row_indices.len().saturating_mul(row_words_sub)];
    let mut feat_sum_x = vec![0.0f64; row_indices.len()];
    let mut feat_sum_x2 = vec![0.0f64; row_indices.len()];
    let mut feat_sum_xy = vec![0.0f64; row_indices.len()];
    for (dst_r, &src_r) in row_indices.iter().enumerate() {
        if src_r >= n_rows_all {
            return Err(format!("row index out of range while subsetting: {src_r}"));
        }
        let src_ge1 = &bits_flat[src_r * row_words_full..(src_r + 1) * row_words_full];
        let src_ge2 =
            bits_hi_flat.map(|bits| &bits[src_r * row_words_full..(src_r + 1) * row_words_full]);
        let dst_base = dst_r * row_words_sub;
        let mut sum_x = 0.0f64;
        let mut sum_x2 = 0.0f64;
        let mut sum_xy = 0.0f64;
        for (dst_s, &src_s) in sample_indices.iter().enumerate() {
            if src_s >= n_samples_all {
                return Err(format!(
                    "sample index out of range while subsetting: {src_s}"
                ));
            }
            let ge1 = ((src_ge1[src_s >> 6] >> (src_s & 63)) & 1u64) as u8;
            let ge2 = src_ge2
                .map(|row| ((row[src_s >> 6] >> (src_s & 63)) & 1u64) as u8)
                .unwrap_or(0);
            if ge1 != 0 {
                out_ge1[dst_base + (dst_s >> 6)] |= 1u64 << (dst_s & 63);
            }
            if ge2 != 0 {
                out_ge2[dst_base + (dst_s >> 6)] |= 1u64 << (dst_s & 63);
            }
            let dosage = f64::from(ge1.saturating_add(ge2));
            sum_x += dosage;
            sum_x2 += dosage * dosage;
            sum_xy += dosage * y[dst_s];
        }
        feat_sum_x[dst_r] = sum_x;
        feat_sum_x2[dst_r] = sum_x2;
        feat_sum_xy[dst_r] = sum_xy;
    }
    PACKED_EXTRACT_FLAT_NS.fetch_add(elapsed_ns_saturating(_t), Ordering::Relaxed);
    Ok((
        out_ge1,
        out_ge2,
        row_words_sub,
        feat_sum_x,
        feat_sum_x2,
        feat_sum_xy,
    ))
}

fn packed_dosage_rows_subset_from_full_bits_range_with_stage1(
    bits_flat: &[u64],
    bits_hi_flat: Option<&[u64]>,
    row_words_full: usize,
    row_start: usize,
    row_end: usize,
    sample_indices: &[usize],
    y: &[f64],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<(Vec<u64>, Vec<u64>, usize, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let _t = Instant::now();
    if row_end <= row_start {
        return Ok((
            Vec::new(),
            Vec::new(),
            words_for_samples(sample_indices.len()),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
    }
    if row_end > n_rows_all {
        return Err(format!(
            "row range out of bounds while subsetting: [{row_start}, {row_end}) vs n_rows={n_rows_all}"
        ));
    }
    if row_words_full != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words_full}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words_full) {
        return Err("full bit matrix length mismatch".to_string());
    }
    if let Some(bits_hi_flat) = bits_hi_flat {
        if bits_hi_flat.len() != n_rows_all.saturating_mul(row_words_full) {
            return Err("full high-bit matrix length mismatch".to_string());
        }
    }
    if sample_indices.len() != y.len() {
        return Err(format!(
            "sample_indices length ({}) != y length ({}) while subsetting packed range",
            sample_indices.len(),
            y.len()
        ));
    }
    let row_words_sub = words_for_samples(sample_indices.len());
    let n_rows_sub = row_end.saturating_sub(row_start);
    let mut out_ge1 = vec![0u64; n_rows_sub.saturating_mul(row_words_sub)];
    let mut out_ge2 = vec![0u64; n_rows_sub.saturating_mul(row_words_sub)];
    let mut feat_sum_x = vec![0.0f64; n_rows_sub];
    let mut feat_sum_x2 = vec![0.0f64; n_rows_sub];
    let mut feat_sum_xy = vec![0.0f64; n_rows_sub];
    for (dst_r, src_r) in (row_start..row_end).enumerate() {
        let src_ge1 = &bits_flat[src_r * row_words_full..(src_r + 1) * row_words_full];
        let src_ge2 =
            bits_hi_flat.map(|bits| &bits[src_r * row_words_full..(src_r + 1) * row_words_full]);
        let dst_base = dst_r * row_words_sub;
        let mut sum_x = 0.0f64;
        let mut sum_x2 = 0.0f64;
        let mut sum_xy = 0.0f64;
        for (dst_s, &src_s) in sample_indices.iter().enumerate() {
            if src_s >= n_samples_all {
                return Err(format!(
                    "sample index out of range while subsetting: {src_s}"
                ));
            }
            let ge1 = ((src_ge1[src_s >> 6] >> (src_s & 63)) & 1u64) as u8;
            let ge2 = src_ge2
                .map(|row| ((row[src_s >> 6] >> (src_s & 63)) & 1u64) as u8)
                .unwrap_or(0);
            if ge1 != 0 {
                out_ge1[dst_base + (dst_s >> 6)] |= 1u64 << (dst_s & 63);
            }
            if ge2 != 0 {
                out_ge2[dst_base + (dst_s >> 6)] |= 1u64 << (dst_s & 63);
            }
            let dosage = f64::from(ge1.saturating_add(ge2));
            sum_x += dosage;
            sum_x2 += dosage * dosage;
            sum_xy += dosage * y[dst_s];
        }
        feat_sum_x[dst_r] = sum_x;
        feat_sum_x2[dst_r] = sum_x2;
        feat_sum_xy[dst_r] = sum_xy;
    }
    PACKED_EXTRACT_FLAT_NS.fetch_add(elapsed_ns_saturating(_t), Ordering::Relaxed);
    Ok((
        out_ge1,
        out_ge2,
        row_words_sub,
        feat_sum_x,
        feat_sum_x2,
        feat_sum_xy,
    ))
}

fn packed_rows_subset_from_full_bits_range(
    bits_flat: &[u64],
    row_words_full: usize,
    row_start: usize,
    row_end: usize,
    sample_indices: &[usize],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<(Vec<u64>, usize), String> {
    if row_end <= row_start {
        return Ok((Vec::new(), words_for_samples(sample_indices.len())));
    }
    if row_end > n_rows_all {
        return Err(format!(
            "row range out of bounds while subsetting: [{row_start}, {row_end}) vs n_rows={n_rows_all}"
        ));
    }
    if row_words_full != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words_full}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words_full) {
        return Err("full bit matrix length mismatch".to_string());
    }
    if sample_indices_are_full_identity(sample_indices, n_samples_all) {
        let out = gather_rows_by_range(
            bits_flat,
            row_words_full,
            row_start,
            row_end,
            "garfield::packed_rows_subset_range_full_samples",
        )?;
        return Ok((out, row_words_full));
    }
    let row_words_sub = words_for_samples(sample_indices.len());
    let n_rows_sub = row_end.saturating_sub(row_start);
    let mut out = vec![0u64; n_rows_sub.saturating_mul(row_words_sub)];
    for (dst_r, src_r) in (row_start..row_end).enumerate() {
        let src = &bits_flat[src_r * row_words_full..(src_r + 1) * row_words_full];
        for (dst_s, &src_s) in sample_indices.iter().enumerate() {
            if src_s >= n_samples_all {
                return Err(format!(
                    "sample index out of range while subsetting: {src_s}"
                ));
            }
            let bit = (src[src_s >> 6] >> (src_s & 63)) & 1u64;
            if bit != 0 {
                out[dst_r * row_words_sub + (dst_s >> 6)] |= 1u64 << (dst_s & 63);
            }
        }
    }
    Ok((out, row_words_sub))
}

fn packed_rows_subset_from_full_bits(
    bits_flat: &[u64],
    row_words_full: usize,
    row_indices: &[usize],
    sample_indices: &[usize],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<(Vec<u64>, usize), String> {
    if row_words_full != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words_full}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words_full) {
        return Err("full bit matrix length mismatch".to_string());
    }
    if sample_indices_are_full_identity(sample_indices, n_samples_all) {
        let out = gather_rows_by_indices(
            bits_flat,
            row_words_full,
            row_indices,
            "garfield::packed_rows_subset_indices_full_samples",
        )?;
        return Ok((out, row_words_full));
    }
    let row_words_sub = words_for_samples(sample_indices.len());
    let mut out = vec![0u64; row_indices.len().saturating_mul(row_words_sub)];
    for (dst_r, &src_r) in row_indices.iter().enumerate() {
        if src_r >= n_rows_all {
            return Err(format!("row index out of range while subsetting: {src_r}"));
        }
        let src = &bits_flat[src_r * row_words_full..(src_r + 1) * row_words_full];
        for (dst_s, &src_s) in sample_indices.iter().enumerate() {
            if src_s >= n_samples_all {
                return Err(format!(
                    "sample index out of range while subsetting: {src_s}"
                ));
            }
            let bit = (src[src_s >> 6] >> (src_s & 63)) & 1u64;
            if bit != 0 {
                out[dst_r * row_words_sub + (dst_s >> 6)] |= 1u64 << (dst_s & 63);
            }
        }
    }
    Ok((out, row_words_sub))
}

#[inline]
fn binary_row_var_from_support(support: usize, n_samples: usize) -> f64 {
    if n_samples == 0 {
        return 0.0;
    }
    let p = (support as f64) / (n_samples as f64);
    p * (1.0 - p)
}

#[inline]
fn binary_pair_r2_from_bit_rows(
    row_i: &[u64],
    support_i: usize,
    row_j: &[u64],
    support_j: usize,
    n_samples: usize,
) -> f64 {
    if n_samples == 0 {
        return 0.0;
    }
    let ss_i = (n_samples as f64) * binary_row_var_from_support(support_i, n_samples);
    let ss_j = (n_samples as f64) * binary_row_var_from_support(support_j, n_samples);
    if !(ss_i > 0.0 && ss_j > 0.0) {
        return 0.0;
    }
    let both_one = and_popcount(row_i, row_j) as f64;
    let p_i = (support_i as f64) / (n_samples as f64);
    let p_j = (support_j as f64) / (n_samples as f64);
    let cov = both_one - (n_samples as f64) * p_i * p_j;
    let denom = (ss_i * ss_j).sqrt();
    if !(denom > 0.0) || !cov.is_finite() {
        return 0.0;
    }
    let corr = cov / denom;
    (corr * corr).clamp(0.0, 1.0)
}

#[inline]
fn binary_pair_high_ld_from_support_and_both_one(
    support_i: usize,
    support_j: usize,
    both_one: usize,
    n_samples: usize,
    r2_threshold: f64,
) -> bool {
    if n_samples == 0 {
        return false;
    }
    let ss_i = (n_samples as f64) * binary_row_var_from_support(support_i, n_samples);
    let ss_j = (n_samples as f64) * binary_row_var_from_support(support_j, n_samples);
    if !(ss_i > 0.0 && ss_j > 0.0) {
        return false;
    }
    let expected = ((support_i as f64) * (support_j as f64)) / (n_samples as f64);
    let cov = (both_one as f64) - expected;
    let denom = r2_threshold * ss_i * ss_j;
    cov.is_finite() && denom.is_finite() && (cov * cov) >= denom
}

#[inline]
fn binary_pair_abs_r2_upper_bound_from_support(
    support_i: usize,
    support_j: usize,
    n_samples: usize,
) -> f64 {
    if n_samples == 0
        || support_i == 0
        || support_j == 0
        || support_i >= n_samples
        || support_j >= n_samples
    {
        return 0.0;
    }
    let p_i = (support_i as f64) / (n_samples as f64);
    let p_j = (support_j as f64) / (n_samples as f64);
    let var_i = p_i * (1.0 - p_i);
    let var_j = p_j * (1.0 - p_j);
    if !(var_i > 0.0 && var_j > 0.0) {
        return 0.0;
    }
    let c_min = (p_i + p_j - 1.0).max(0.0);
    let c_max = p_i.min(p_j);
    let expected = p_i * p_j;
    let cov_abs = (c_max - expected).abs().max((c_min - expected).abs());
    let denom = (var_i * var_j).sqrt();
    if !(denom > 0.0) || !cov_abs.is_finite() {
        return 0.0;
    }
    let corr = cov_abs / denom;
    (corr * corr).clamp(0.0, 1.0)
}

fn geneset_ld_exact_bounds_from_support(
    support_i: usize,
    support_j: usize,
    n_samples: usize,
    r2_threshold: f64,
) -> GarfieldGenesetLdExactBounds {
    if n_samples == 0
        || support_i == 0
        || support_j == 0
        || support_i >= n_samples
        || support_j >= n_samples
    {
        return GarfieldGenesetLdExactBounds::default();
    }
    let c_min = support_i
        .saturating_add(support_j)
        .saturating_sub(n_samples);
    let c_max = support_i.min(support_j);
    if c_min > c_max {
        return GarfieldGenesetLdExactBounds::default();
    }
    let ss_i = (n_samples as f64) * binary_row_var_from_support(support_i, n_samples);
    let ss_j = (n_samples as f64) * binary_row_var_from_support(support_j, n_samples);
    if !(ss_i > 0.0 && ss_j > 0.0) {
        return GarfieldGenesetLdExactBounds::default();
    }

    let expected = ((support_i as f64) * (support_j as f64)) / (n_samples as f64);
    let delta = (r2_threshold * ss_i * ss_j).sqrt();
    let c_min_i = isize::try_from(c_min).unwrap_or(isize::MAX);
    let c_max_i = isize::try_from(c_max).unwrap_or(isize::MAX);

    let mut low_max = (expected - delta).floor() as isize;
    if low_max < c_min_i {
        low_max = c_min_i - 1;
    } else if low_max > c_max_i {
        low_max = c_max_i;
    }
    while low_max >= c_min_i
        && !binary_pair_high_ld_from_support_and_both_one(
            support_i,
            support_j,
            low_max as usize,
            n_samples,
            r2_threshold,
        )
    {
        low_max -= 1;
    }
    while low_max + 1 <= c_max_i
        && ((low_max + 1) as f64) <= expected
        && binary_pair_high_ld_from_support_and_both_one(
            support_i,
            support_j,
            (low_max + 1) as usize,
            n_samples,
            r2_threshold,
        )
    {
        low_max += 1;
    }

    let mut high_min = (expected + delta).ceil() as isize;
    if high_min > c_max_i {
        high_min = c_max_i + 1;
    } else if high_min < c_min_i {
        high_min = c_min_i;
    }
    while high_min <= c_max_i
        && !binary_pair_high_ld_from_support_and_both_one(
            support_i,
            support_j,
            high_min as usize,
            n_samples,
            r2_threshold,
        )
    {
        high_min += 1;
    }
    while high_min > c_min_i
        && ((high_min - 1) as f64) >= expected
        && binary_pair_high_ld_from_support_and_both_one(
            support_i,
            support_j,
            (high_min - 1) as usize,
            n_samples,
            r2_threshold,
        )
    {
        high_min -= 1;
    }

    GarfieldGenesetLdExactBounds {
        low_max: if low_max >= c_min_i {
            u32::try_from(low_max).unwrap_or(u32::MAX)
        } else {
            GARFIELD_GENESET_LD_NO_BOUND
        },
        high_min: if high_min <= c_max_i {
            u32::try_from(high_min).unwrap_or(u32::MAX)
        } else {
            GARFIELD_GENESET_LD_NO_BOUND
        },
    }
}

fn geneset_ld_support_cache(
    n_samples: usize,
    r2_threshold: f64,
) -> Arc<GarfieldGenesetLdSupportCacheEntry> {
    let key = (n_samples, r2_threshold.to_bits());
    let cache = GARFIELD_GENESET_LD_SUPPORT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(hit) = guard.get(&key) {
            return Arc::clone(hit);
        }
    }

    let mut buckets = vec![Vec::<usize>::new(); n_samples + 1];
    let mut exact_bounds =
        vec![vec![GarfieldGenesetLdExactBounds::default(); n_samples + 1]; n_samples + 1];
    for (support_i, bucket) in buckets.iter_mut().enumerate().take(n_samples).skip(1) {
        let mut allowed = Vec::<usize>::new();
        for support_j in 1..n_samples {
            if binary_pair_abs_r2_upper_bound_from_support(support_i, support_j, n_samples)
                >= r2_threshold
            {
                allowed.push(support_j);
                exact_bounds[support_i][support_j] = geneset_ld_exact_bounds_from_support(
                    support_i,
                    support_j,
                    n_samples,
                    r2_threshold,
                );
            }
        }
        *bucket = allowed;
    }

    let built = Arc::new(GarfieldGenesetLdSupportCacheEntry {
        conflict_supports: buckets,
        exact_bounds,
    });
    if let Ok(mut guard) = cache.lock() {
        if let Some(hit) = guard.get(&key) {
            return Arc::clone(hit);
        }
        guard.insert(key, Arc::clone(&built));
    }
    built
}

fn maybe_prune_geneset_unit_rows_by_ld(
    unit: &GarfieldLogicUnit,
    candidate_global_rows: &[usize],
    unit_kind_lc: &str,
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
) -> Result<Vec<usize>, String> {
    let t0 = Instant::now();
    let r2_threshold = garfield_geneset_ld_prune_r2();
    if unit_kind_lc != "geneset"
        || unit.spans.len() <= 1
        || candidate_global_rows.len() <= 1
        || !(r2_threshold.is_finite() && r2_threshold > 0.0 && r2_threshold <= 1.0)
    {
        return Ok(candidate_global_rows.to_vec());
    }
    let n_samples = sample_indices.len();
    if n_samples <= 1 {
        return Ok(candidate_global_rows.to_vec());
    }
    GARFIELD_GENESET_LD_ROWS_TOTAL.fetch_add(
        u64::try_from(candidate_global_rows.len()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    let use_full_identity = sample_indices_are_full_identity(sample_indices, logic_bits.n_samples);
    let mut packed_rows = Vec::<u64>::new();
    let row_words = if use_full_identity {
        logic_bits.row_words
    } else {
        let (packed_rows_out, row_words_out) = packed_rows_subset_from_full_bits(
            logic_bits.bits_flat.as_slice(),
            logic_bits.row_words,
            candidate_global_rows,
            sample_indices,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        packed_rows = packed_rows_out;
        row_words_out
    };
    let row_slice = |local_idx: usize| -> &[u64] {
        if use_full_identity {
            let global_idx = candidate_global_rows[local_idx];
            &logic_bits.bits_flat[global_idx * row_words..(global_idx + 1) * row_words]
        } else {
            &packed_rows[local_idx * row_words..(local_idx + 1) * row_words]
        }
    };
    let mut support = vec![0usize; candidate_global_rows.len()];
    let mut var = vec![0.0_f64; candidate_global_rows.len()];
    let mut variable_local = Vec::<usize>::with_capacity(candidate_global_rows.len());
    for local_idx in 0..candidate_global_rows.len() {
        let row = row_slice(local_idx);
        let cnt = popcount(row) as usize;
        support[local_idx] = cnt;
        let row_var = binary_row_var_from_support(cnt, n_samples);
        var[local_idx] = row_var;
        if row_var > 0.0 {
            variable_local.push(local_idx);
        }
    }
    if variable_local.is_empty() {
        GARFIELD_GENESET_LD_ROWS_KEPT.fetch_add(0, Ordering::Relaxed);
        garfield_geneset_ld_unit_stats_record(candidate_global_rows.len(), 0, 0);
        GARFIELD_GENESET_LD_PRUNE_NS.fetch_add(elapsed_ns_saturating(t0), Ordering::Relaxed);
        return Ok(Vec::new());
    }

    let support_cache = geneset_ld_support_cache(n_samples, r2_threshold);
    let support_conflict_buckets = support_cache.conflict_supports.as_slice();
    let exact_bounds = support_cache.exact_bounds.as_slice();

    variable_local.sort_by(|&a, &b| {
        let ga = candidate_global_rows[a];
        let gb = candidate_global_rows[b];
        var[b].total_cmp(&var[a]).then_with(|| {
            let sa = &logic_bits.sites[ga];
            let sb = &logic_bits.sites[gb];
            chrom_sort_key(sa.garfield_chrom())
                .cmp(&chrom_sort_key(sb.garfield_chrom()))
                .then_with(|| sa.garfield_pos().cmp(&sb.garfield_pos()))
                .then_with(|| ga.cmp(&gb))
        })
    });

    let mut kept_local = Vec::<usize>::with_capacity(variable_local.len());
    let mut kept_by_support = vec![Vec::<usize>::new(); n_samples + 1];
    let mut exact_pairs = 0u64;
    for &local_idx in variable_local.iter() {
        check_ctrlc()?;
        let row_i = row_slice(local_idx);
        let mut has_conflict = false;
        let support_i = support[local_idx];
        let exact_bounds_i = &exact_bounds[support_i];
        for &support_j in support_conflict_buckets[support_i].iter() {
            let bounds = exact_bounds_i[support_j];
            for &kept_idx in kept_by_support[support_j].iter() {
                exact_pairs = exact_pairs.saturating_add(1);
                let row_j = row_slice(kept_idx);
                if bounds.matches(and_popcount(row_i, row_j) as usize) {
                    has_conflict = true;
                    break;
                }
            }
            if has_conflict {
                break;
            }
        }
        if !has_conflict {
            kept_local.push(local_idx);
            kept_by_support[support_i].push(local_idx);
        }
    }

    kept_local.sort_unstable();
    let kept = kept_local
        .into_iter()
        .map(|local_idx| candidate_global_rows[local_idx])
        .collect::<Vec<_>>();
    GARFIELD_GENESET_LD_EXACT_PAIRS.fetch_add(exact_pairs, Ordering::Relaxed);
    GARFIELD_GENESET_LD_ROWS_KEPT.fetch_add(
        u64::try_from(kept.len()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    garfield_geneset_ld_unit_stats_record(candidate_global_rows.len(), kept.len(), exact_pairs);
    GARFIELD_GENESET_LD_PRUNE_NS.fetch_add(elapsed_ns_saturating(t0), Ordering::Relaxed);
    Ok(kept)
}

#[inline]
fn resolve_ml_keep_k(n_region: usize, ml_top_k: usize, ml_top_frac: f64) -> usize {
    if n_region == 0 {
        return 0;
    }
    let keep_k = if ml_top_k > 0 {
        ml_top_k.min(n_region)
    } else {
        ((ml_top_frac.clamp(0.0, 1.0) * (n_region as f64)).ceil() as usize)
            .max(1)
            .min(n_region)
    };
    keep_k.max(1).min(n_region)
}

#[inline]
fn corr_geneset_ld_prescreen_k(n_region: usize, keep_k: usize) -> usize {
    if n_region == 0 {
        return 0;
    }
    let keep_k = keep_k.max(1).min(n_region);
    let slack = (keep_k / 4).clamp(
        GARFIELD_GENESET_CORR_PRESCREEN_SLACK_MIN,
        GARFIELD_GENESET_CORR_PRESCREEN_SLACK_MAX,
    );
    n_region.min(keep_k.saturating_add(slack)).max(keep_k)
}

#[inline]
fn geneset_corr_stage1_allow_parallel(
    unit_kind_lc: &str,
    n_region: usize,
    allow_parallel: bool,
) -> bool {
    allow_parallel
        && !(unit_kind_lc == "geneset" && n_region <= GARFIELD_GENESET_CORR_SERIAL_MAX_ROWS)
}

#[inline]
fn resolve_ml_top_random_counts(keep_k: usize, top_frac: f64) -> (usize, usize) {
    if keep_k == 0 {
        return (0, 0);
    }
    if keep_k == 1 {
        return (1, 0);
    }
    let frac = top_frac.clamp(0.0, 1.0);
    if frac >= 1.0 {
        return (keep_k, 0);
    }
    if frac <= 0.0 {
        return (1, keep_k - 1);
    }
    let top_keep = ((keep_k as f64) * frac).floor() as usize;
    let top_keep = top_keep.clamp(1, keep_k - 1);
    (top_keep, keep_k - top_keep)
}

fn select_ml_top_only_local_indices(
    dense_train: &[Vec<u8>],
    y_train: &[f64],
    response: ResponseKind,
    engine: MlEngine,
    tree_cfg: ExtraTreesConfig,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    keep_k: usize,
    feature_group_ids: Option<&[usize]>,
) -> Result<Vec<usize>, String> {
    if keep_k == 0 || dense_train.is_empty() {
        return Ok(Vec::new());
    }
    let n_region = dense_train.len();
    let keep_k = keep_k.min(n_region).max(1);
    match engine {
        MlEngine::Gbdt2 => {
            let coarse_k = n_region.min(
                keep_k
                    .saturating_mul(4)
                    .max(keep_k.saturating_add(8))
                    .max(32),
            );
            let coarse_scores = compute_feature_scores_grouped(
                dense_train,
                y_train,
                response,
                MlEngine::Gbdt,
                tree_cfg,
                ImportanceKind::Imp,
                perm_cfg,
                feature_group_ids,
            )?;
            let coarse_local = topk_indices(coarse_scores.as_slice(), coarse_k);
            if coarse_local.is_empty() {
                return Ok(Vec::new());
            }
            if coarse_local.len() <= keep_k {
                return Ok(coarse_local);
            }
            let refined_rows = coarse_local
                .iter()
                .map(|&idx| dense_train[idx].clone())
                .collect::<Vec<_>>();
            let refined_group_ids = feature_group_ids.map(|groups| {
                coarse_local
                    .iter()
                    .map(|&idx| groups[idx])
                    .collect::<Vec<_>>()
            });
            let refined_scores = compute_feature_scores_grouped(
                refined_rows.as_slice(),
                y_train,
                response,
                MlEngine::Gbdt,
                tree_cfg,
                importance,
                perm_cfg,
                refined_group_ids.as_deref(),
            )?;
            let refined_local = topk_indices(refined_scores.as_slice(), keep_k);
            Ok(refined_local
                .into_iter()
                .map(|idx| coarse_local[idx])
                .collect())
        }
        _ => {
            let scores = compute_feature_scores_grouped(
                dense_train,
                y_train,
                response,
                engine,
                tree_cfg,
                importance,
                perm_cfg,
                feature_group_ids,
            )?;
            Ok(topk_indices(scores.as_slice(), keep_k))
        }
    }
}

#[inline]
fn insert_topk_density_hit(
    bucket: &mut Vec<(usize, f64)>,
    n_not: usize,
    score: f64,
    keep_k: usize,
) {
    bucket.push((n_not, score));
    bucket.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    if bucket.len() > keep_k {
        bucket.truncate(keep_k);
    }
}

fn select_ml_top_local_indices(
    dense_train: &[Vec<u8>],
    y_train: &[f64],
    response: ResponseKind,
    engine: MlEngine,
    tree_cfg: ExtraTreesConfig,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    keep_k: usize,
    keep_policy: GarfieldMlKeepPolicy,
    selection_seed: u64,
    feature_group_ids: Option<&[usize]>,
) -> Result<Vec<usize>, String> {
    if keep_k == 0 || dense_train.is_empty() {
        return Ok(Vec::new());
    }
    let n_region = dense_train.len();
    let keep_k = keep_k.min(n_region).max(1);
    match keep_policy {
        GarfieldMlKeepPolicy::TopK => select_ml_top_only_local_indices(
            dense_train,
            y_train,
            response,
            engine,
            tree_cfg,
            importance,
            perm_cfg,
            keep_k,
            feature_group_ids,
        ),
        GarfieldMlKeepPolicy::TopKPlusRandom { top_frac } => {
            let (top_keep, rand_keep) = resolve_ml_top_random_counts(keep_k, top_frac);
            let mut out = select_ml_top_only_local_indices(
                dense_train,
                y_train,
                response,
                engine,
                tree_cfg,
                importance,
                perm_cfg,
                top_keep,
                feature_group_ids,
            )?;
            if rand_keep == 0 || out.len() >= n_region {
                return Ok(out);
            }
            let mut picked = vec![false; n_region];
            for &idx in out.iter() {
                picked[idx] = true;
            }
            let remaining = (0..n_region)
                .filter(|&idx| !picked[idx])
                .collect::<Vec<_>>();
            if remaining.is_empty() {
                return Ok(out);
            }
            let sample_n = rand_keep.min(remaining.len());
            let mut rng = StdRng::seed_from_u64(selection_seed);
            let sampled = sample_indices_without_replacement(&mut rng, remaining.len(), sample_n);
            let mut sampled_indices = sampled
                .into_vec()
                .into_iter()
                .map(|idx| remaining[idx])
                .collect::<Vec<_>>();
            sampled_indices.sort_unstable();
            out.extend(sampled_indices);
            Ok(out)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn select_logic_unit_global_rows(
    unit: &GarfieldLogicUnit,
    unit_kind_lc: &str,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    y_train: &[f64],
    allow_parallel: bool,
) -> Result<Option<Vec<usize>>, String> {
    check_ctrlc()?;
    if unit.indices.is_empty() {
        return Ok(None);
    }
    let defer_corr_geneset_ld_prune = matches!(engine, Some(MlEngine::Corr))
        && unit_kind_lc == "geneset"
        && unit.spans.len() > 1
        && unit.indices.len() > 1;
    let candidate_global_rows = if defer_corr_geneset_ld_prune {
        unit.indices.clone()
    } else {
        maybe_prune_geneset_unit_rows_by_ld(
            unit,
            unit.indices.as_slice(),
            unit_kind_lc,
            logic_bits,
            train_idx_local,
        )?
    };
    if candidate_global_rows.is_empty() {
        return Ok(None);
    }
    let selected_global_rows = if let Some(engine_one) = engine {
        let n_region = candidate_global_rows.len();
        let keep_k = resolve_ml_keep_k(n_region, ml_top_k, ml_top_frac);

        // ---- PairwiseAnd fast path: packed subset + cached stage-1 stats ----
        if engine_one == MlEngine::PairwiseAnd {
            let t0 = Instant::now();
            let (
                packed_bits_ge1,
                packed_bits_ge2,
                row_words_sub,
                feat_sum_x,
                feat_sum_x2,
                feat_sum_xy,
            ) = packed_dosage_rows_subset_from_full_bits_with_stage1(
                logic_bits.bits_flat.as_slice(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                candidate_global_rows.as_slice(),
                train_idx_local,
                y_train,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            let scores = feature_scores_pairwise_and_packed_dual_with_stage1(
                packed_bits_ge1.as_slice(),
                packed_bits_ge2.as_slice(),
                row_words_sub,
                n_region,
                y_train,
                train_idx_local.len(),
                feat_sum_x.as_slice(),
                feat_sum_x2.as_slice(),
                feat_sum_xy.as_slice(),
            );
            let top_local = topk_indices(&scores, keep_k);
            GARFIELD_ML_SELECT_NS.fetch_add(
                t0.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            if top_local.is_empty() {
                return Ok(None);
            }
            return Ok(Some(
                top_local
                    .iter()
                    .map(|&idx| candidate_global_rows[idx])
                    .collect::<Vec<_>>(),
            ));
        }

        if engine_one == MlEngine::Corr {
            let t0 = Instant::now();
            let corr_allow_parallel =
                geneset_corr_stage1_allow_parallel(unit_kind_lc, n_region, allow_parallel);
            let (feat_sum_x, feat_sum_x2, feat_sum_xy) = dosage_stage1_stats_from_full_bits(
                logic_bits.bits_flat.as_slice(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                candidate_global_rows.as_slice(),
                train_idx_local,
                y_train,
                logic_bits.sites.len(),
                logic_bits.n_samples,
                corr_allow_parallel,
            )?;
            let scores = feature_scores_abs_corr_stage1_with_parallel(
                feat_sum_x.as_slice(),
                feat_sum_x2.as_slice(),
                feat_sum_xy.as_slice(),
                y_train,
                corr_allow_parallel,
            );
            let pruned_global_rows = if defer_corr_geneset_ld_prune {
                if keep_k < candidate_global_rows.len() {
                    let mut prescreen_k =
                        corr_geneset_ld_prescreen_k(candidate_global_rows.len(), keep_k);
                    loop {
                        let top_local = topk_indices(&scores, prescreen_k);
                        let prescreen_rows = top_local
                            .iter()
                            .map(|&idx| candidate_global_rows[idx])
                            .collect::<Vec<_>>();
                        let pruned = maybe_prune_geneset_unit_rows_by_ld(
                            unit,
                            prescreen_rows.as_slice(),
                            unit_kind_lc,
                            logic_bits,
                            train_idx_local,
                        )?;
                        if pruned.len() >= keep_k || prescreen_k >= candidate_global_rows.len() {
                            break pruned;
                        }
                        let next_k = corr_geneset_ld_prescreen_k(
                            candidate_global_rows.len(),
                            prescreen_k.saturating_mul(2),
                        );
                        if next_k <= prescreen_k {
                            break pruned;
                        }
                        prescreen_k = next_k;
                    }
                } else {
                    maybe_prune_geneset_unit_rows_by_ld(
                        unit,
                        candidate_global_rows.as_slice(),
                        unit_kind_lc,
                        logic_bits,
                        train_idx_local,
                    )?
                }
            } else {
                candidate_global_rows.clone()
            };
            let top_local = if pruned_global_rows.len() <= keep_k {
                (0..pruned_global_rows.len()).collect::<Vec<_>>()
            } else {
                let score_index = candidate_global_rows
                    .iter()
                    .enumerate()
                    .map(|(i, &global_idx)| (global_idx, i))
                    .collect::<HashMap<usize, usize>>();
                let pruned_scores = pruned_global_rows
                    .iter()
                    .map(|global_idx| {
                        let score_idx = score_index.get(global_idx).copied().ok_or_else(|| {
                            format!(
                                "GARFIELD Corr geneset prescreen lost score index for row {}",
                                global_idx
                            )
                        })?;
                        Ok(scores[score_idx])
                    })
                    .collect::<Result<Vec<f64>, String>>()?;
                topk_indices(&pruned_scores, keep_k)
            };
            GARFIELD_ML_SELECT_NS.fetch_add(
                t0.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            if top_local.is_empty() {
                return Ok(None);
            }
            return Ok(Some(
                top_local
                    .iter()
                    .map(|&idx| pruned_global_rows[idx])
                    .collect::<Vec<_>>(),
            ));
        }

        // ---- General ML path (Vec<Vec<u8>>) ----
        let dense_train = dense_dosage_rows_from_full_bits(
            logic_bits.bits_flat.as_slice(),
            logic_bits.bits_hi_flat.as_deref(),
            logic_bits.row_words,
            candidate_global_rows.as_slice(),
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        if dense_train.is_empty() {
            return Ok(None);
        }
        let ml_group_ids = build_unit_window_group_ids(
            unit,
            candidate_global_rows.as_slice(),
            logic_bits.sites.as_slice(),
            unit_kind_lc,
        );
        let top_local = select_ml_top_local_indices(
            dense_train.as_slice(),
            y_train,
            response,
            engine_one,
            ExtraTreesConfig {
                allow_parallel,
                ..tree_cfg
            },
            importance,
            perm_cfg,
            keep_k,
            GarfieldMlKeepPolicy::TopK,
            tree_cfg.seed ^ 0xB6D5_0C11_8E91_3F27,
            ml_group_ids.as_deref(),
        )?;
        if top_local.is_empty() {
            return Ok(None);
        }
        top_local
            .iter()
            .map(|&idx| candidate_global_rows[idx])
            .collect::<Vec<_>>()
    } else {
        candidate_global_rows
    };
    if selected_global_rows.is_empty() {
        return Ok(None);
    }
    Ok(Some(selected_global_rows))
}

#[allow(clippy::too_many_arguments)]
fn prepare_logic_unit_continuous(
    unit: &GarfieldLogicUnit,
    unit_kind_lc: &str,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    _test_idx_local: &[usize],
    y_train: &[f64],
    beam_params: BeamSearchParams,
) -> Result<Option<GarfieldUnitPrepared>, String> {
    check_ctrlc()?;
    let Some(selected_global_rows) = select_logic_unit_global_rows(
        unit,
        unit_kind_lc,
        response,
        engine,
        importance,
        perm_cfg,
        ml_top_k,
        ml_top_frac,
        tree_cfg,
        logic_bits,
        train_idx_local,
        y_train,
        beam_params.allow_parallel,
    )?
    else {
        return Ok(None);
    };

    let local_groups = build_unit_window_group_ids(
        unit,
        selected_global_rows.as_slice(),
        logic_bits.sites.as_slice(),
        unit_kind_lc,
    )
    .unwrap_or_else(|| {
        selected_global_rows
            .iter()
            .map(|&idx| logic_bits.group_ids[idx])
            .collect::<Vec<_>>()
    });

    let geneset_stage_group_target =
        geneset_stage_group_target(unit_kind_lc, local_groups.as_slice());
    Ok(Some(GarfieldUnitPrepared {
        selected_global_rows,
        local_groups,
        geneset_stage_group_target,
    }))
}

#[allow(clippy::too_many_arguments)]
fn prepare_logic_chunk_continuous(
    chunk: GarfieldNullChunk,
    row_mul: usize,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    _test_idx_local: &[usize],
    y_train: &[f64],
    beam_params: BeamSearchParams,
) -> Result<Option<GarfieldUnitPrepared>, String> {
    check_ctrlc()?;
    if chunk.end_index <= chunk.start_index || row_mul == 0 {
        return Ok(None);
    }
    let row_start = chunk.start_index.saturating_mul(row_mul);
    let row_end = chunk.end_index.saturating_mul(row_mul);
    if row_end > logic_bits.sites.len() {
        return Ok(None);
    }

    let n_region = row_end.saturating_sub(row_start);
    let keep_k = resolve_ml_keep_k(n_region, ml_top_k, ml_top_frac);
    let keep_policy = GarfieldMlKeepPolicy::TopKPlusRandom {
        top_frac: GARFIELD_NULL_ML_TOP_FRAC,
    };
    let selection_seed = perm_cfg.seed ^ chunk.window_id ^ 0x94D0_49BB_1331_11EB;

    let selected_global_rows = if let Some(engine_one) = engine {
        // ---- PairwiseAnd fast path: packed subset + cached stage-1 stats ----
        if engine_one == MlEngine::PairwiseAnd {
            let (
                packed_bits_ge1,
                packed_bits_ge2,
                row_words_sub,
                feat_sum_x,
                feat_sum_x2,
                feat_sum_xy,
            ) = packed_dosage_rows_subset_from_full_bits_range_with_stage1(
                logic_bits.bits_flat.as_slice(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                row_start,
                row_end,
                train_idx_local,
                y_train,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            let scores = feature_scores_pairwise_and_packed_dual_with_stage1(
                packed_bits_ge1.as_slice(),
                packed_bits_ge2.as_slice(),
                row_words_sub,
                n_region,
                y_train,
                train_idx_local.len(),
                feat_sum_x.as_slice(),
                feat_sum_x2.as_slice(),
                feat_sum_xy.as_slice(),
            );
            let (top_keep, rand_keep) =
                resolve_ml_top_random_counts(keep_k, GARFIELD_NULL_ML_TOP_FRAC);
            let mut top_local = topk_indices(&scores, top_keep.max(1));
            if rand_keep > 0 && top_local.len() < n_region {
                let mut picked = vec![false; n_region];
                for &idx in top_local.iter() {
                    if idx < n_region {
                        picked[idx] = true;
                    }
                }
                let remaining: Vec<usize> = (0..n_region).filter(|&i| !picked[i]).collect();
                if !remaining.is_empty() {
                    let sample_n = rand_keep.min(remaining.len());
                    let mut rng = StdRng::seed_from_u64(selection_seed);
                    let sampled =
                        sample_indices_without_replacement(&mut rng, remaining.len(), sample_n);
                    let mut extra: Vec<usize> = sampled
                        .into_vec()
                        .into_iter()
                        .map(|i| remaining[i])
                        .collect();
                    extra.sort_unstable();
                    top_local.extend(extra);
                }
            }
            if top_local.is_empty() {
                return Ok(None);
            }
            // Fall through to common path: convert local → global indices
            top_local
                .into_iter()
                .map(|idx| row_start + idx)
                .collect::<Vec<_>>()
        } else if engine_one == MlEngine::Corr {
            let t0 = Instant::now();
            let (feat_sum_x, feat_sum_x2, feat_sum_xy) = dosage_stage1_stats_from_full_bits_range(
                logic_bits.bits_flat.as_slice(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                row_start,
                row_end,
                train_idx_local,
                y_train,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            let scores = feature_scores_abs_corr_stage1(
                feat_sum_x.as_slice(),
                feat_sum_x2.as_slice(),
                feat_sum_xy.as_slice(),
                y_train,
            );
            let (top_keep, rand_keep) =
                resolve_ml_top_random_counts(keep_k, GARFIELD_NULL_ML_TOP_FRAC);
            let mut top_local = topk_indices(&scores, top_keep.max(1));
            if rand_keep > 0 && top_local.len() < n_region {
                let mut picked = vec![false; n_region];
                for &idx in top_local.iter() {
                    if idx < n_region {
                        picked[idx] = true;
                    }
                }
                let remaining: Vec<usize> = (0..n_region).filter(|&i| !picked[i]).collect();
                if !remaining.is_empty() {
                    let sample_n = rand_keep.min(remaining.len());
                    let mut rng = StdRng::seed_from_u64(selection_seed);
                    let sampled =
                        sample_indices_without_replacement(&mut rng, remaining.len(), sample_n);
                    let mut extra: Vec<usize> = sampled
                        .into_vec()
                        .into_iter()
                        .map(|i| remaining[i])
                        .collect();
                    extra.sort_unstable();
                    top_local.extend(extra);
                }
            }
            GARFIELD_ML_SELECT_NS.fetch_add(
                t0.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            if top_local.is_empty() {
                return Ok(None);
            }
            top_local
                .into_iter()
                .map(|idx| row_start + idx)
                .collect::<Vec<_>>()
        } else {
            // ---- General ML path ----
            let dense_train = dense_dosage_rows_from_full_bits_range(
                logic_bits.bits_flat.as_slice(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                row_start,
                row_end,
                train_idx_local,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            if dense_train.is_empty() {
                return Ok(None);
            }
            let top_local = select_ml_top_local_indices(
                dense_train.as_slice(),
                y_train,
                response,
                engine_one,
                ExtraTreesConfig {
                    allow_parallel: beam_params.allow_parallel,
                    ..tree_cfg
                },
                importance,
                perm_cfg,
                keep_k,
                keep_policy,
                selection_seed,
                None,
            )?;
            if top_local.is_empty() {
                return Ok(None);
            }
            top_local
                .into_iter()
                .map(|idx| row_start + idx)
                .collect::<Vec<_>>()
        }
    } else {
        (row_start..row_end).collect::<Vec<_>>()
    };
    if selected_global_rows.is_empty() {
        return Ok(None);
    }

    let local_groups = if engine.is_none() {
        logic_bits.group_ids[row_start..row_end].to_vec()
    } else {
        selected_global_rows
            .iter()
            .map(|&idx| logic_bits.group_ids[idx])
            .collect::<Vec<_>>()
    };

    Ok(Some(GarfieldUnitPrepared {
        selected_global_rows,
        local_groups,
        geneset_stage_group_target: None,
    }))
}

fn collect_rule_permutation_nulls_for_repeat(
    prepared: &GarfieldUnitPrepared,
    prepared_bits: &GarfieldUnitBitMatrices,
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    beam_params: BeamSearchParams,
    rep_seed: u64,
) -> Result<Vec<(RuleNullBucket, f64, f64)>, String> {
    if prepared.selected_global_rows.is_empty() {
        return Ok(Vec::new());
    }
    let beam_params = beam_params_for_prepared(prepared, beam_params);

    let perm_train = shuffled_copy_f64(y_train, rep_seed);
    let perm_test = if split_applied {
        shuffled_copy_f64(y_test, rep_seed ^ 0xD1B5_4A32_D192_ED03)
    } else {
        perm_train.clone()
    };
    let perm_hits = beam_search_train_test_continuous_dispatch(
        perm_train.as_slice(),
        prepared_bits,
        prepared.selected_global_rows.len(),
        perm_test.as_slice(),
        prepared.local_groups.as_slice(),
        beam_params,
    )?;
    // Bucket train and test scores independently by their OWN pseudo-SNP MAF,
    // avoiding cross-boundary errors when train/test frequencies differ.
    let mut train_by_bucket = HashMap::<RuleNullBucket, Vec<usize>>::new();
    let mut test_by_bucket = HashMap::<RuleNullBucket, Vec<usize>>::new();
    for (cand_idx, cand) in perm_hits.iter().enumerate() {
        if cand.train_score.is_finite() {
            let b = bucket_from_rule(&cand.rule, cand.train.dosage_maf);
            train_by_bucket.entry(b).or_default().push(cand_idx);
        }
        if cand.test_score.is_finite() {
            let b = bucket_from_rule(&cand.rule, cand.test.dosage_maf);
            test_by_bucket.entry(b).or_default().push(cand_idx);
        }
    }
    let mut out = Vec::<(RuleNullBucket, f64, f64)>::new();
    // Collect all unique buckets
    let all_buckets: HashSet<RuleNullBucket> = train_by_bucket
        .keys()
        .chain(test_by_bucket.keys())
        .copied()
        .collect();
    for bucket in all_buckets {
        let keep_topk = null_topk_per_repeat_for_bucket(bucket).max(1);
        let mut train_chosen = HashSet::<usize>::new();
        let mut test_chosen = HashSet::<usize>::new();

        if let Some(idxs) = train_by_bucket.get(&bucket) {
            let mut sorted = idxs.clone();
            sorted.sort_by(|&a, &b| {
                normalized_rank_score(perm_hits[b].train_score)
                    .partial_cmp(&normalized_rank_score(perm_hits[a].train_score))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &idx in sorted.iter().take(keep_topk) {
                train_chosen.insert(idx);
            }
        }
        if let Some(idxs) = test_by_bucket.get(&bucket) {
            let mut sorted = idxs.clone();
            sorted.sort_by(|&a, &b| {
                normalized_rank_score(perm_hits[b].test_score)
                    .partial_cmp(&normalized_rank_score(perm_hits[a].test_score))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &idx in sorted.iter().take(keep_topk) {
                test_chosen.insert(idx);
            }
        }

        // Push train scores and test scores to their OWN buckets independently.
        // Use NaN for the "other" score so RuleNullScores::push silently
        // drops it — each bucket only sees its own dimension.
        for idx in train_chosen {
            let cand = &perm_hits[idx];
            out.push((bucket, cand.train_score, f64::NAN));
        }
        for idx in test_chosen {
            let cand = &perm_hits[idx];
            out.push((bucket, f64::NAN, cand.test_score));
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Default)]
struct AdaptiveStructurePriorUnitOutcome {
    calibrator: RuleStructurePriorCalibrator,
    repeats_used: usize,
}

#[inline]
fn logsumexp_scaled_scores(scores: &[f64], temperature: f64) -> Option<f64> {
    if scores.is_empty() {
        return None;
    }
    let inv_temp = 1.0 / temperature.max(1e-12);
    let max_scaled = scores
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .map(|score| score * inv_temp)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_scaled.is_finite() {
        return None;
    }
    let tail = scores
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .map(|score| ((score * inv_temp) - max_scaled).exp())
        .sum::<f64>()
        .max(1e-12);
    Some(max_scaled + tail.ln())
}

#[inline]
fn len_prob_rank_signature(probs: &[f64; 6]) -> [usize; 5] {
    let mut order = [1usize, 2usize, 3usize, 4usize, 5usize];
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    order
}

#[inline]
fn kl_divergence_len_probs(p: &[f64; 6], q: &[f64; 6]) -> f64 {
    let mut out = 0.0_f64;
    for len in 1..=5usize {
        let pp = p[len].max(1e-12);
        let qq = q[len].max(1e-12);
        out += pp * (pp / qq).ln();
    }
    out
}

#[inline]
fn symmetric_kl_len_probs(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    0.5 * (kl_divergence_len_probs(a, b) + kl_divergence_len_probs(b, a))
}

fn collect_rule_structure_posterior_for_repeat(
    prepared: &GarfieldUnitPrepared,
    prepared_bits: &GarfieldUnitBitMatrices,
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    beam_params: BeamSearchParams,
    structure_prior_cfg: &RuleStructurePriorConfig,
    rep_seed: u64,
) -> Result<Vec<(usize, usize, f64, f64)>, String> {
    if prepared.selected_global_rows.is_empty() || y_train.is_empty() {
        return Ok(Vec::new());
    }
    let beam_params = beam_params_for_prepared(prepared, beam_params);

    let boot_train_idx = bootstrap_sample_indices(y_train.len(), rep_seed);
    let boot_y_train = subset_vec_f64(y_train, boot_train_idx.as_slice())?;
    let boot_train_bits = {
        let row_indices = (0..prepared.selected_global_rows.len()).collect::<Vec<_>>();
        let (bits, _) = packed_rows_subset_from_full_bits(
            prepared_bits.train_bits.as_slice(),
            prepared_bits.row_words_train,
            row_indices.as_slice(),
            boot_train_idx.as_slice(),
            prepared.selected_global_rows.len(),
            y_train.len(),
        )?;
        bits
    };
    let boot_train_bits_hi = if let Some(train_bits_hi) = prepared_bits.train_bits_hi.as_deref() {
        let row_indices = (0..prepared.selected_global_rows.len()).collect::<Vec<_>>();
        let (bits, _) = packed_rows_subset_from_full_bits(
            train_bits_hi,
            prepared_bits.row_words_train,
            row_indices.as_slice(),
            boot_train_idx.as_slice(),
            prepared.selected_global_rows.len(),
            y_train.len(),
        )?;
        Some(bits)
    } else {
        None
    };
    let boot_row_words_train = words_for_samples(boot_y_train.len());

    let (boot_y_test, boot_test_bits, boot_test_bits_hi, boot_row_words_test) = if split_applied {
        if y_test.is_empty() {
            return Ok(Vec::new());
        }
        let boot_test_idx =
            bootstrap_sample_indices(y_test.len(), rep_seed ^ 0xD1B5_4A32_D192_ED03);
        let boot_y_test = subset_vec_f64(y_test, boot_test_idx.as_slice())?;
        let row_indices = (0..prepared.selected_global_rows.len()).collect::<Vec<_>>();
        let (bits, _) = packed_rows_subset_from_full_bits(
            prepared_bits.test_bits(),
            prepared_bits.row_words_test,
            row_indices.as_slice(),
            boot_test_idx.as_slice(),
            prepared.selected_global_rows.len(),
            y_test.len(),
        )?;
        let bits_hi = if let Some(test_bits_hi) = prepared_bits.test_bits_hi() {
            let (bits_hi, _) = packed_rows_subset_from_full_bits(
                test_bits_hi,
                prepared_bits.row_words_test,
                row_indices.as_slice(),
                boot_test_idx.as_slice(),
                prepared.selected_global_rows.len(),
                y_test.len(),
            )?;
            Some(bits_hi)
        } else {
            None
        };
        let boot_row_words_test = words_for_samples(boot_y_test.len());
        (boot_y_test, bits, bits_hi, boot_row_words_test)
    } else {
        (
            boot_y_train.clone(),
            boot_train_bits.clone(),
            boot_train_bits_hi.clone(),
            boot_row_words_train,
        )
    };

    let perm_hits = if let Some(boot_train_hi) = boot_train_bits_hi.as_ref() {
        let boot_test_hi = boot_test_bits_hi
            .as_ref()
            .ok_or_else(|| "internal error: fuzzy bootstrap test bitplane missing".to_string())?;
        beam_search_train_test_continuous_fuzzy(
            boot_y_train.as_slice(),
            boot_train_bits.as_slice(),
            boot_train_hi.as_slice(),
            boot_row_words_train,
            prepared.selected_global_rows.len(),
            boot_y_train.len(),
            boot_y_test.as_slice(),
            boot_test_bits.as_slice(),
            boot_test_hi.as_slice(),
            boot_row_words_test,
            boot_y_test.len(),
            prepared.local_groups.as_slice(),
            beam_params,
        )?
    } else {
        beam_search_train_test_continuous(
            boot_y_train.as_slice(),
            boot_train_bits.as_slice(),
            boot_row_words_train,
            prepared.selected_global_rows.len(),
            boot_y_train.len(),
            boot_y_test.as_slice(),
            boot_test_bits.as_slice(),
            boot_row_words_test,
            boot_y_test.len(),
            prepared.local_groups.as_slice(),
            beam_params,
        )?
    };

    if perm_hits.is_empty() {
        return Ok(Vec::new());
    }

    let mut topk_by_len = vec![Vec::<(usize, f64)>::new(); 6];
    for cand in perm_hits.iter() {
        let rule_len = cand.rule.len().clamp(1, 5);
        let score = if split_applied {
            cand.test_score
        } else {
            cand.train_score
        };
        if !score.is_finite() || score <= 0.0 {
            continue;
        }
        let bucket = &mut topk_by_len[rule_len];
        insert_topk_density_hit(
            bucket,
            cand.rule.not_count(),
            score,
            DEFAULT_RULE_STRUCTURE_DENSITY_TOPK,
        );
    }
    let pooled_scores = topk_by_len
        .iter()
        .skip(1)
        .flat_map(|hits| hits.iter().map(|(_, score)| *score))
        .collect::<Vec<_>>();
    if pooled_scores.is_empty() {
        return Ok(Vec::new());
    }

    let score_scale = pooled_scores.iter().sum::<f64>() / (pooled_scores.len() as f64);
    let temperature = (score_scale * 1.1).clamp(0.02, 0.08);
    let observed = (1..=5usize)
        .filter_map(|rule_len| {
            let hits = &topk_by_len[rule_len];
            if hits.is_empty() {
                return None;
            }
            let best_n_not = hits[0].0;
            let best_score = hits[0].1;
            let len_scores = hits.iter().map(|(_, score)| *score).collect::<Vec<_>>();
            let log_density = logsumexp_scaled_scores(len_scores.as_slice(), temperature)?;
            Some((rule_len, best_n_not, best_score, log_density))
        })
        .collect::<Vec<_>>();
    if observed.is_empty() {
        return Ok(Vec::new());
    }
    let max_logw = observed
        .iter()
        .map(|(rule_len, _, _, log_density)| {
            structure_prior_cfg.len_alpha(*rule_len).ln() + *log_density
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let mut weights = Vec::<f64>::with_capacity(observed.len());
    for (rule_len, _, _, log_density) in observed.iter() {
        let logw = structure_prior_cfg.len_alpha(*rule_len).ln() + *log_density;
        weights.push((logw - max_logw).exp());
    }
    let total = weights.iter().sum::<f64>().max(1e-12);
    let mut out = Vec::<(usize, usize, f64, f64)>::with_capacity(observed.len());
    for ((rule_len, n_not, score, _log_density), weight_raw) in
        observed.into_iter().zip(weights.into_iter())
    {
        out.push((rule_len, n_not, score, weight_raw / total));
    }
    Ok(out)
}

fn collect_rule_structure_posterior_for_unit_adaptive(
    prepared: &GarfieldUnitPrepared,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    beam_params: BeamSearchParams,
    structure_prior_cfg: &RuleStructurePriorConfig,
    unit_seed: u64,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<AdaptiveStructurePriorUnitOutcome, String> {
    if prepared.selected_global_rows.is_empty() || y_train.is_empty() {
        return Ok(AdaptiveStructurePriorUnitOutcome::default());
    }
    let prepared_bits = materialize_prepared_bit_matrices(
        prepared,
        logic_bits,
        train_idx_local,
        test_idx_local,
        false,
    )?;
    if let Some(tracker) = mem_tracker {
        tracker.sample_now();
    }
    let mut local = RuleStructurePriorCalibrator::default();
    let mut informative_repeats = 0usize;
    let mut prev_probs = None::<[f64; 6]>;
    let mut stable_streak = 0usize;
    let mut repeats_used = 0usize;

    for rep in 0..DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MAX_REPEATS {
        if (rep & 7) == 0 {
            check_ctrlc()?;
        }
        let rep_seed =
            unit_seed ^ ((rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) ^ 0xA24B_AED4_963E_E407;
        let repeat_out = collect_rule_structure_posterior_for_repeat(
            prepared,
            &prepared_bits,
            y_train,
            y_test,
            split_applied,
            beam_params.clone(),
            structure_prior_cfg,
            rep_seed,
        )?;
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        if repeat_out.is_empty() {
            repeats_used += 1;
            continue;
        }
        for (rule_len, n_not, score, weight) in repeat_out.into_iter() {
            local.insert(rule_len, n_not, score, weight);
        }
        repeats_used += 1;
        informative_repeats += 1;
        if informative_repeats < DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MIN_REPEATS {
            continue;
        }
        let Some(cur_probs) = local.observed_len_probs_preview() else {
            continue;
        };
        if let Some(prev) = prev_probs {
            let same_rank = len_prob_rank_signature(&prev) == len_prob_rank_signature(&cur_probs);
            let kl = symmetric_kl_len_probs(&prev, &cur_probs);
            if same_rank && kl <= DEFAULT_RULE_STRUCTURE_BOOTSTRAP_KL_THRESHOLD {
                stable_streak += 1;
            } else {
                stable_streak = 0;
            }
            if stable_streak >= DEFAULT_RULE_STRUCTURE_BOOTSTRAP_STABLE_REPEATS {
                break;
            }
        }
        prev_probs = Some(cur_probs);
    }

    Ok(AdaptiveStructurePriorUnitOutcome {
        calibrator: local,
        repeats_used,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_logic_unit_prepared_continuous(
    ui: usize,
    unit: &GarfieldLogicUnit,
    prepared: &GarfieldUnitPrepared,
    prepared_bits: &GarfieldUnitBitMatrices,
    logic_bits: &GarfieldLogicBits,
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    beam_params: BeamSearchParams,
    top_rules_per_unit: usize,
    unit_kind_lc: &str,
    literal_scores: Option<&[LiteralSingletonScore]>,
) -> Result<Vec<GarfieldLogicRuleRecord>, String> {
    let beam_params_search = beam_params_for_prepared(prepared, beam_params.clone());
    let beam_hits = if prepared_bits.has_fuzzy_bin() {
        beam_search_train_test_continuous_dispatch(
            y_train,
            prepared_bits,
            prepared.selected_global_rows.len(),
            y_test,
            prepared.local_groups.as_slice(),
            beam_params_search.clone(),
        )?
    } else if let Some(scores) = literal_scores {
        beam_search_train_test_continuous_with_literal_scores(
            y_train,
            prepared_bits.train_bits.as_slice(),
            prepared_bits.row_words_train,
            prepared.selected_global_rows.len(),
            y_train.len(),
            y_test,
            prepared_bits.test_bits(),
            prepared_bits.row_words_test,
            test_idx_local.len(),
            prepared.local_groups.as_slice(),
            beam_params_search.clone(),
            scores,
        )?
    } else {
        beam_search_train_test_continuous(
            y_train,
            prepared_bits.train_bits.as_slice(),
            prepared_bits.row_words_train,
            prepared.selected_global_rows.len(),
            y_train.len(),
            y_test,
            prepared_bits.test_bits(),
            prepared_bits.row_words_test,
            test_idx_local.len(),
            prepared.local_groups.as_slice(),
            beam_params_search.clone(),
        )?
    };
    if beam_hits.is_empty() {
        return Ok(Vec::new());
    }
    let local_sites =
        local_sites_from_selected_rows(prepared.selected_global_rows.as_slice(), logic_bits)?;
    let selected_bits_full = prepared_bits.selected_bits_full().ok_or_else(|| {
        "internal error: selected_bits_full missing for GARFIELD evaluation".to_string()
    })?;
    let selected_bits_full_hi = prepared_bits.selected_bits_full_hi();

    let keep_rules = if top_rules_per_unit == 0 {
        beam_hits.len()
    } else {
        extend_keep_with_score_ties(
            beam_hits.as_slice(),
            top_rules_per_unit.min(beam_hits.len()),
            |cand| cand.test_score,
        )
    };
    let mut out = Vec::<GarfieldLogicRuleRecord>::with_capacity(keep_rules.max(1));
    for (cand_idx, cand) in beam_hits.iter().enumerate() {
        let keep = if top_rules_per_unit == 0 {
            true
        } else {
            cand_idx < keep_rules
        };
        if !keep {
            continue;
        }
        let first_site = local_sites
            .get(cand.rule.first.row_index)
            .ok_or_else(|| "beam result first row index out of range".to_string())?;
        let (mut full_bits, mut full_ge2_bits) = if let Some(bits_hi) = selected_bits_full_hi {
            let (ge1, ge2) = materialize_rule_bits_dual(
                &cand.rule,
                selected_bits_full,
                bits_hi,
                logic_bits.row_words,
                prepared.selected_global_rows.len(),
                logic_bits.n_samples,
            )?;
            (ge1, Some(ge2))
        } else {
            (
                materialize_rule_bits(
                    &cand.rule,
                    selected_bits_full,
                    logic_bits.row_words,
                    prepared.selected_global_rows.len(),
                    logic_bits.n_samples,
                )?,
                None,
            )
        };
        let polarity =
            preferred_display_polarity(&cand.rule, full_bits.as_slice(), logic_bits.n_samples);
        if matches!(polarity, GarfieldRuleDisplayPolarity::Complement) {
            if let Some(ge2_bits) = full_ge2_bits.as_mut() {
                complement_dual_bits_in_place(
                    full_bits.as_mut_slice(),
                    ge2_bits.as_mut_slice(),
                    logic_bits.n_samples,
                );
            } else {
                complement_bits_in_place(full_bits.as_mut_slice(), logic_bits.n_samples);
            }
        }
        let expr = rule_expr_with_polarity(&cand.rule, local_sites.as_slice(), polarity)?;
        let snp_name = rule_snp_name_with_polarity(&cand.rule, local_sites.as_slice(), polarity)?;
        let bim_snp_name =
            rule_bim_name_with_polarity(&cand.rule, local_sites.as_slice(), polarity)?;
        let (bim_allele0, bim_allele1) =
            rule_bim_alleles_with_polarity(&cand.rule, local_sites.as_slice(), polarity)?;
        let display_ops = rule_display_ops_with_polarity(&cand.rule, polarity);
        let display_negated = rule_display_negated_with_polarity(&cand.rule, polarity);
        let delta_score = build_rule_delta_score_annotation(
            &cand.rule,
            cand.test_score,
            y_test,
            prepared_bits.test_bits(),
            prepared_bits.test_bits_hi(),
            prepared_bits.row_words_test,
            prepared.selected_global_rows.len(),
            test_idx_local.len(),
            &beam_params,
            polarity,
        )?;
        let selected_row_indices =
            rule_selected_global_rows(&cand.rule, prepared.selected_global_rows.as_slice())?;
        out.push(GarfieldLogicRuleRecord {
            unit_name: unit.label.clone(),
            unit_kind: unit_kind_lc.to_string(),
            unit_index: ui + 1,
            region_size: unit.indices.len(),
            ml_feature_count: prepared.selected_global_rows.len(),
            ml_rank: rule_ml_rank_name_with_polarity(&cand.rule, polarity),
            selected_row_indices,
            display_ops,
            display_negated,
            snp_name,
            expr,
            chrom_field: first_site.chrom.as_ref().to_string(),
            bim_snp_name,
            bim_allele0,
            bim_allele1,
            pos: first_site.pos,
            score: cand.test_score,
            delta_score,
            support_bits: None,
        });
    }
    Ok(out)
}

fn precompute_literal_singleton_scores_for_unit(
    prepared: &GarfieldUnitPrepared,
    prepared_bits: &GarfieldUnitBitMatrices,
    y_train: &[f64],
    y_test: &[f64],
) -> Result<Vec<LiteralSingletonScore>, String> {
    let requests = [LiteralScoreBatchRequest {
        bits_train: prepared_bits.train_bits.as_slice(),
        row_words_train: prepared_bits.row_words_train,
        bits_test: prepared_bits.test_bits(),
        row_words_test: prepared_bits.row_words_test,
        n_rows: prepared.selected_global_rows.len(),
    }];
    let mut out = precompute_literal_singleton_scores_batched(
        y_train,
        y_train.len(),
        y_test,
        y_test.len(),
        requests.as_slice(),
    )?;
    if out.len() != 1 {
        return Err(format!(
            "GARFIELD unit literal score batch size mismatch: got {}, expected 1",
            out.len()
        ));
    }
    Ok(out.swap_remove(0))
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn evaluate_logic_unit_continuous(
    ui: usize,
    unit: &GarfieldLogicUnit,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    beam_params: BeamSearchParams,
    top_rules_per_unit: usize,
    unit_kind_lc: &str,
) -> Result<Vec<GarfieldLogicRuleRecord>, String> {
    let Some(prepared) = prepare_logic_unit_continuous(
        unit,
        unit_kind_lc,
        response,
        engine,
        importance,
        perm_cfg,
        ml_top_k,
        ml_top_frac,
        tree_cfg,
        logic_bits,
        train_idx_local,
        test_idx_local,
        y_train,
        beam_params.clone(),
    )?
    else {
        return Ok(Vec::new());
    };
    let prepared_bits = materialize_prepared_bit_matrices(
        &prepared,
        logic_bits,
        train_idx_local,
        test_idx_local,
        true,
    )?;
    evaluate_logic_unit_prepared_continuous(
        ui,
        unit,
        &prepared,
        &prepared_bits,
        logic_bits,
        test_idx_local,
        y_train,
        y_test,
        beam_params,
        top_rules_per_unit,
        unit_kind_lc,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn process_scan_unit_continuous(
    ui: usize,
    units: &[GarfieldLogicUnit],
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    beam_params: BeamSearchParams,
    top_rules_per_unit: usize,
    unit_kind_lc: &str,
    progress_callback: Option<&Py<PyAny>>,
    scan_progress_done: &AtomicUsize,
    scan_notify_step: usize,
    scanned_units: usize,
    skipped_units: &Arc<Mutex<Vec<GarfieldSkippedUnitInfo>>>,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<Vec<GarfieldLogicRuleRecord>, String> {
    check_ctrlc()?;
    let unit = &units[ui];
    let mut out = Vec::<GarfieldLogicRuleRecord>::new();
    if let Some(prepared) = prepare_logic_unit_continuous(
        unit,
        unit_kind_lc,
        response,
        engine,
        importance,
        perm_cfg,
        ml_top_k,
        ml_top_frac,
        tree_cfg,
        logic_bits,
        train_idx_local,
        test_idx_local,
        y_train,
        beam_params.clone(),
    )? {
        let prepared_bits = materialize_prepared_bit_matrices(
            &prepared,
            logic_bits,
            train_idx_local,
            test_idx_local,
            true,
        )?;
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        let literal_scores = if prepared_bits.has_fuzzy_bin() {
            None
        } else {
            Some(precompute_literal_singleton_scores_for_unit(
                &prepared,
                &prepared_bits,
                y_train,
                y_test,
            )?)
        };
        out = match evaluate_logic_unit_prepared_continuous(
            ui,
            unit,
            &prepared,
            &prepared_bits,
            logic_bits,
            test_idx_local,
            y_train,
            y_test,
            beam_params.clone(),
            top_rules_per_unit,
            unit_kind_lc,
            literal_scores.as_ref().map(|v| v.as_slice()),
        ) {
            Ok(v) => v,
            Err(err) if is_no_valid_initial_literals_error(&err) => {
                skipped_units
                    .lock()
                    .map_err(|_| "GARFIELD skipped-units mutex poisoned".to_string())?
                    .push(make_skipped_unit_info("scan", &unit.label, &err));
                Vec::new()
            }
            Err(err) => return Err(err),
        };
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
    }
    let done = scan_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
    if done % scan_notify_step == 0 || done == scanned_units {
        garfield_stage_progress_notify(progress_callback, "scan", done, scanned_units.max(1), None)
            .map_err(|e| e.to_string())?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn process_rule_permutation_task_chunk(
    slot_indices: &[usize],
    rep_start: usize,
    rep_end: usize,
    null_chunk_prepared: &[(GarfieldNullChunk, GarfieldUnitPrepared)],
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    perm_beam_params: BeamSearchParams,
    seed: u64,
    progress_callback: Option<&Py<PyAny>>,
    null_progress_done: &AtomicUsize,
    null_notify_step: usize,
    permutation_task_total: usize,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<Vec<(usize, Vec<(RuleNullBucket, f64, f64)>)>, String> {
    let mut out = Vec::<(usize, Vec<(RuleNullBucket, f64, f64)>)>::with_capacity(
        slot_indices.len() * (rep_end - rep_start),
    );
    for &slot in slot_indices.iter() {
        check_ctrlc()?;
        let (chunk, prepared) = &null_chunk_prepared[slot];
        let prepared_bits = materialize_prepared_bit_matrices(
            prepared,
            logic_bits,
            train_idx_local,
            test_idx_local,
            false,
        )?;
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        for rep in rep_start..rep_end {
            if ((rep - rep_start) & 7) == 0 {
                check_ctrlc()?;
            }
            let rep_seed = seed
                ^ (chunk.window_id.wrapping_mul(0x94D0_49BB_1331_11EB))
                ^ ((rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                ^ ((slot as u64).wrapping_mul(0xA24B_AED4_963E_E407));
            let vals = collect_rule_permutation_nulls_for_repeat(
                prepared,
                &prepared_bits,
                y_train,
                y_test,
                split_applied,
                perm_beam_params.clone(),
                rep_seed,
            )?;
            if let Some(tracker) = mem_tracker {
                tracker.sample_now();
            }
            let done = null_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % null_notify_step == 0 || done == permutation_task_total {
                garfield_stage_progress_notify(
                    progress_callback,
                    "null_penalty",
                    done,
                    permutation_task_total.max(1),
                    None,
                )
                .map_err(|e| e.to_string())?;
            }
            out.push((rep, vals));
        }
    }
    Ok(out)
}

/// Like `process_rule_permutation_task_chunk` but takes a flat `&[(slot, rep)]`
/// list so callers can balance work across workers regardless of the
/// slot/repeat ratio.
#[allow(dead_code)]
fn process_rule_permutation_task_chunk_flat(
    task_chunk: &[(usize, usize)],
    null_chunk_prepared: &[(GarfieldNullChunk, GarfieldUnitPrepared)],
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    perm_beam_params: BeamSearchParams,
    seed: u64,
    progress_callback: Option<&Py<PyAny>>,
    null_progress_done: &AtomicUsize,
    null_notify_step: usize,
    permutation_task_total: usize,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<Vec<(usize, Vec<(RuleNullBucket, f64, f64)>)>, String> {
    let mut out = Vec::<(usize, Vec<(RuleNullBucket, f64, f64)>)>::with_capacity(task_chunk.len());
    for &(slot, rep) in task_chunk.iter() {
        check_ctrlc()?;
        let (chunk, prepared) = &null_chunk_prepared[slot];
        let prepared_bits = materialize_prepared_bit_matrices(
            prepared,
            logic_bits,
            train_idx_local,
            test_idx_local,
            false,
        )?;
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        let rep_seed = seed
            ^ (chunk.window_id.wrapping_mul(0x94D0_49BB_1331_11EB))
            ^ ((rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ ((slot as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let vals = collect_rule_permutation_nulls_for_repeat(
            prepared,
            &prepared_bits,
            y_train,
            y_test,
            split_applied,
            perm_beam_params.clone(),
            rep_seed,
        )?;
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        let done = null_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done % null_notify_step == 0 || done == permutation_task_total {
            garfield_stage_progress_notify(
                progress_callback,
                "null_penalty",
                done,
                permutation_task_total.max(1),
                None,
            )
            .map_err(|e| e.to_string())?;
        }
        out.push((rep, vals));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn process_structure_prior_unit_chunk(
    slot_start: usize,
    chunk: &[(usize, GarfieldUnitPrepared)],
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    posterior_beam_params: BeamSearchParams,
    structure_prior_cfg: &RuleStructurePriorConfig,
    seed: u64,
    structure_scores: &Arc<Mutex<RuleStructurePriorCalibrator>>,
    structure_repeats_used: &AtomicUsize,
    structure_progress_done: &AtomicUsize,
    structure_notify_step: usize,
    structure_task_total: usize,
    structure_prior_display_len: usize,
    progress_callback: Option<&Py<PyAny>>,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<(), String> {
    for (local_idx, (ui, prepared)) in chunk.iter().enumerate() {
        check_ctrlc()?;
        let slot = slot_start + local_idx;
        let unit_seed = seed
            ^ ((*ui as u64).wrapping_mul(0xD134_2543_DE82_EF95))
            ^ ((slot as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let out = collect_rule_structure_posterior_for_unit_adaptive(
            prepared,
            logic_bits,
            train_idx_local,
            test_idx_local,
            y_train,
            y_test,
            split_applied,
            posterior_beam_params.clone(),
            structure_prior_cfg,
            unit_seed,
            mem_tracker,
        )?;
        structure_repeats_used.fetch_add(out.repeats_used, Ordering::Relaxed);
        let alpha_meta = {
            let mut guard = structure_scores
                .lock()
                .map_err(|_| "GARFIELD structure prior mutex poisoned".to_string())?;
            guard.merge_from(&out.calibrator);
            format_structure_alpha_meta(&guard, structure_prior_cfg, structure_prior_display_len)
        };
        let done = structure_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done % structure_notify_step == 0 || done == structure_task_total {
            garfield_stage_progress_notify(
                progress_callback,
                "structure_prior",
                done,
                structure_task_total.max(1),
                Some(alpha_meta.as_str()),
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
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

#[inline]
fn plink2bits_from_logic_g(g: i8) -> u8 {
    match g {
        0 => 0b00,
        2 => 0b11,
        _ => 0b01,
    }
}

#[derive(Clone, Debug)]
struct GarfieldPseudoLiteralExport {
    chrom: String,
    pos: i32,
    snp_name: String,
    allele0: String,
    allele1: String,
    selected_row_index: usize,
    display_negated: bool,
}

fn collect_logic_pseudo_literal_exports(
    rec: &GarfieldLogicRuleRecord,
    logic_bits: &GarfieldLogicBits,
) -> Result<Vec<GarfieldPseudoLiteralExport>, String> {
    let expected = rec.selected_row_indices.len();
    if rec.display_negated.len() != expected {
        return Err(format!(
            "GARFIELD pseudo export negation mismatch for {}: rows={}, negated={}",
            rec.snp_name,
            expected,
            rec.display_negated.len()
        ));
    }
    let mut out = Vec::<GarfieldPseudoLiteralExport>::with_capacity(expected);
    for idx in 0..expected {
        let row_idx = rec.selected_row_indices[idx];
        let site = logic_bits
            .sites
            .get(row_idx)
            .ok_or_else(|| format!("GARFIELD pseudo export row index out of range: {row_idx}"))?;
        let negated = rec.display_negated[idx];
        let snp_name = literal_name(site, negated);
        let (allele0, allele1) = literal_bim_alleles(site, negated);
        out.push(GarfieldPseudoLiteralExport {
            chrom: site.chrom.as_ref().to_string(),
            pos: site.pos,
            snp_name,
            allele0,
            allele1,
            selected_row_index: row_idx,
            display_negated: negated,
        });
    }
    Ok(out)
}

#[inline]
fn write_logic_bits_as_plink_row<W: Write>(
    w: &mut W,
    row_buf: &mut [u8],
    bits: &[u64],
    n_samples: usize,
    complement: bool,
) -> Result<(), String> {
    row_buf.fill(0u8);
    for s in 0..n_samples {
        let mut hit = ((bits[s >> 6] >> (s & 63)) & 1u64) != 0;
        if complement {
            hit = !hit;
        }
        let g = if hit { 2i8 } else { 0i8 };
        row_buf[s >> 2] |= plink2bits_from_logic_g(g) << ((s & 3) * 2);
    }
    w.write_all(row_buf).map_err(|e| e.to_string())
}

#[inline]
fn write_logic_dual_bits_as_plink_row<W: Write>(
    w: &mut W,
    row_buf: &mut [u8],
    ge1_bits: &[u64],
    ge2_bits: &[u64],
    n_samples: usize,
    complement: bool,
) -> Result<(), String> {
    row_buf.fill(0u8);
    for s in 0..n_samples {
        let ge1 = ((ge1_bits[s >> 6] >> (s & 63)) & 1u64) as i8;
        let ge2 = ((ge2_bits[s >> 6] >> (s & 63)) & 1u64) as i8;
        let mut g = ge1 + ge2;
        if complement {
            g = 2 - g;
        }
        row_buf[s >> 2] |= plink2bits_from_logic_g(g) << ((s & 3) * 2);
    }
    w.write_all(row_buf).map_err(|e| e.to_string())
}

fn write_logic_pseudo_plink(
    prefix: &str,
    sample_ids: &[String],
    records: &[GarfieldLogicRuleRecord],
    logic_bits: &GarfieldLogicBits,
) -> Result<(), String> {
    let bed_path = format!("{prefix}.bed");
    let bim_path = format!("{prefix}.bim");
    let fam_path = format!("{prefix}.fam");
    let mut fam = BufWriter::new(File::create(&fam_path).map_err(|e| e.to_string())?);
    for sid in sample_ids.iter() {
        writeln!(fam, "{0}\t{0}\t0\t0\t1\t-9", sid).map_err(|e| e.to_string())?;
    }
    fam.flush().map_err(|e| e.to_string())?;

    let mut bed = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&bed_path).map_err(|e| e.to_string())?,
    );
    bed.write_all(&[0x6C, 0x1B, 0x01])
        .map_err(|e| e.to_string())?;
    let mut bim = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&bim_path).map_err(|e| e.to_string())?,
    );

    let bytes_per_snp = sample_ids.len().div_ceil(4);
    let mut row_buf = vec![0u8; bytes_per_snp];
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| cmp_logic_rule_records_plink_order(a, b));
    let mut emitted_singletons = HashSet::<(String, i32, String)>::new();
    for rec in ordered.into_iter() {
        let support = materialize_logic_rule_record_support_bits(rec, logic_bits)?;
        let literals = collect_logic_pseudo_literal_exports(rec, logic_bits)?;
        if literals.len() <= 1 {
            let singleton_key = (rec.chrom_field.clone(), rec.pos, rec.bim_snp_name.clone());
            if emitted_singletons.insert(singleton_key) {
                writeln!(
                    bim,
                    "{}\t{}\t0\t{}\t{}\t{}",
                    rec.chrom_field, rec.bim_snp_name, rec.pos, rec.bim_allele0, rec.bim_allele1
                )
                .map_err(|e| e.to_string())?;
                match &support {
                    GarfieldRuleSupportBits::Binary(bits) => {
                        write_logic_bits_as_plink_row(
                            &mut bed,
                            row_buf.as_mut_slice(),
                            bits.as_ref(),
                            sample_ids.len(),
                            false,
                        )?;
                    }
                    GarfieldRuleSupportBits::Dual { ge1, ge2 } => {
                        write_logic_dual_bits_as_plink_row(
                            &mut bed,
                            row_buf.as_mut_slice(),
                            ge1.as_ref(),
                            ge2.as_ref(),
                            sample_ids.len(),
                            false,
                        )?;
                    }
                }
            }
            continue;
        }

        for lit in literals.iter() {
            let singleton_key = (lit.chrom.clone(), lit.pos, lit.snp_name.clone());
            if emitted_singletons.insert(singleton_key) {
                writeln!(
                    bim,
                    "{}\t{}\t0\t{}\t{}\t{}",
                    lit.chrom, lit.snp_name, lit.pos, lit.allele0, lit.allele1
                )
                .map_err(|e| e.to_string())?;
                let row_start = lit.selected_row_index.saturating_mul(logic_bits.row_words);
                let row_end = row_start.saturating_add(logic_bits.row_words);
                let bits = logic_bits.bits_flat.get(row_start..row_end).ok_or_else(|| {
                    format!(
                        "GARFIELD pseudo export literal row slice out of range: row={} row_words={}",
                        lit.selected_row_index, logic_bits.row_words
                    )
                })?;
                if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
                    let bits_hi = bits_hi_flat.get(row_start..row_end).ok_or_else(|| {
                        format!(
                            "GARFIELD pseudo export literal high-bit row slice out of range: row={} row_words={}",
                            lit.selected_row_index, logic_bits.row_words
                        )
                    })?;
                    write_logic_dual_bits_as_plink_row(
                        &mut bed,
                        row_buf.as_mut_slice(),
                        bits,
                        bits_hi,
                        sample_ids.len(),
                        lit.display_negated,
                    )?;
                } else {
                    write_logic_bits_as_plink_row(
                        &mut bed,
                        row_buf.as_mut_slice(),
                        bits,
                        sample_ids.len(),
                        lit.display_negated,
                    )?;
                }
            }
        }

        for lit in literals.iter() {
            writeln!(
                bim,
                "{}\t{}\t0\t{}\t{}\t{}",
                lit.chrom, rec.bim_snp_name, lit.pos, rec.bim_allele0, rec.bim_allele1
            )
            .map_err(|e| e.to_string())?;
            match &support {
                GarfieldRuleSupportBits::Binary(bits) => {
                    write_logic_bits_as_plink_row(
                        &mut bed,
                        row_buf.as_mut_slice(),
                        bits.as_ref(),
                        sample_ids.len(),
                        false,
                    )?;
                }
                GarfieldRuleSupportBits::Dual { ge1, ge2 } => {
                    write_logic_dual_bits_as_plink_row(
                        &mut bed,
                        row_buf.as_mut_slice(),
                        ge1.as_ref(),
                        ge2.as_ref(),
                        sample_ids.len(),
                        false,
                    )?;
                }
            }
        }
    }
    bim.flush().map_err(|e| e.to_string())?;
    bed.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_logic_rules_tsv(path: &str, records: &[GarfieldLogicRuleRecord]) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| cmp_logic_rule_records_output(a, b));
    writeln!(
        w,
        "unit_kind\tunit_index\tunit_name\tregion_size\tml_feature_count\tMLrank\tsnp_name\tscore\tdelta_score"
    )
    .map_err(|e| e.to_string())?;
    for rec in ordered.into_iter() {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}",
            rec.unit_kind,
            rec.unit_index,
            rec.unit_name,
            rec.region_size,
            rec.ml_feature_count,
            rec.ml_rank,
            rec.snp_name,
            rec.score,
            rec.delta_score,
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

fn write_rule_structure_prior_json(path: &str, prior: &RuleStructurePrior) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    let len_probs = prior.len_probs();
    let prior_cfg = prior.config();
    let prior_len_alpha = prior_cfg.len_alpha_array();
    let expected_len = (1..=5usize)
        .map(|rule_len| (rule_len as f64) * len_probs[rule_len])
        .sum::<f64>();
    writeln!(w, "{{").map_err(|e| e.to_string())?;
    writeln!(w, "  \"not_control\": \"null_penalty_only\",").map_err(|e| e.to_string())?;
    writeln!(w, "  \"prior\": {{").map_err(|e| e.to_string())?;
    writeln!(w, "    \"target_ess\": {:.10},", prior_cfg.target_ess())
        .map_err(|e| e.to_string())?;
    writeln!(w, "    \"len_temper\": {:.10},", prior_cfg.len_temper())
        .map_err(|e| e.to_string())?;
    writeln!(w, "    \"len_alpha\": {{").map_err(|e| e.to_string())?;
    for rule_len in 1..=5usize {
        let suffix = if rule_len < 5 { "," } else { "" };
        writeln!(
            w,
            "      \"{}\": {:.10}{}",
            rule_len, prior_len_alpha[rule_len], suffix
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(w, "    }}").map_err(|e| e.to_string())?;
    writeln!(w, "  }},").map_err(|e| e.to_string())?;
    writeln!(w, "  \"strength\": {:.10},", prior.strength()).map_err(|e| e.to_string())?;
    writeln!(w, "  \"best_log_mass\": {:.10},", prior.best_log_mass())
        .map_err(|e| e.to_string())?;
    writeln!(w, "  \"expected_rule_len\": {:.10},", expected_len).map_err(|e| e.to_string())?;
    writeln!(w, "  \"len_probs\": {{").map_err(|e| e.to_string())?;
    for rule_len in 1..=5usize {
        let suffix = if rule_len < 5 { "," } else { "" };
        writeln!(
            w,
            "    \"{}\": {:.10}{}",
            rule_len,
            prior.len_prob(rule_len),
            suffix
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(w, "  }},").map_err(|e| e.to_string())?;
    writeln!(w, "  \"rows\": [").map_err(|e| e.to_string())?;
    let total_rows = 5usize;
    let mut row_idx = 0usize;
    for rule_len in 1..=5usize {
        row_idx += 1;
        let suffix = if row_idx < total_rows { "," } else { "" };
        writeln!(
            w,
            "    {{\"rule_len\": {}, \"prior_len_alpha\": {:.10}, \"len_prob\": {:.10}, \"prior_mass\": {:.10}, \"prior_log_mass\": {:.10}, \"penalty\": {:.10}, \"expected_rule_len\": {:.10}, \"strength\": {:.10}, \"best_log_mass\": {:.10}}}{}",
            rule_len,
            prior_len_alpha[rule_len],
            prior.len_prob(rule_len),
            prior.mass(rule_len),
            prior.log_mass(rule_len),
            prior.penalty(rule_len, 0),
            expected_len,
            prior.strength(),
            prior.best_log_mass(),
            suffix,
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(w, "  ]").map_err(|e| e.to_string())?;
    writeln!(w, "}}").map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())
}

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
fn garfield_logic_search_bed_owned(
    prefix: String,
    y: Vec<f64>,
    grm: Option<Vec<f64>>,
    x_cov: Option<Vec<f64>>,
    q_cov: usize,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
    site_keep_precomputed: Option<Vec<bool>>,
    unit_kind: String,
    groups: Option<Vec<Vec<(String, i32, i32)>>>,
    group_names: Option<Vec<String>>,
    extension: usize,
    step: Option<usize>,
    scan_bimranges: Option<Vec<String>>,
    bin_mode: String,
    ml_method: String,
    ml_importance: String,
    ml_top_k: usize,
    ml_top_frac: f64,
    permutation_repeats: usize,
    permutation_scoring: String,
    tree_cfg: ExtraTreesConfig,
    fold: usize,
    seed: u64,
    max_pick: usize,
    exhaustive_depth: usize,
    beam_width: usize,
    rank_score: String,
    maf_threshold: f32,
    logic_maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    block_cols: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    out_prefix: Option<String>,
    simbench_path: Option<String>,
    top_rules_per_unit: usize,
    max_output_rules: usize,
    max_output_ratio: f64,
    rule_permutation: bool,
    prior_len: Option<Vec<f64>>,
    no_clean: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> Result<GarfieldLogicPipelineResult, String> {
    let total_wall_t0 = Instant::now();
    let rss_debug_enabled = garfield_rss_debug_enabled();
    let mut memory_debug = if rss_debug_enabled {
        Some(GarfieldMemoryDebugSummary::default())
    } else {
        None
    };
    validate_continuous_y(&y)?;
    let using_external_grm = grm.is_some();
    let fam_sample_ids = read_fam(normalize_plink_prefix(&prefix).as_str())?;
    let (selected_sample_indices, selected_sample_ids) =
        build_sample_selection(fam_sample_ids.as_slice(), sample_ids, sample_indices)?;
    let (
        grm_row_source_indices,
        logic_row_flip_external,
        filtered_sites,
        n_samples_total,
        bytes_per_snp,
    ) = if let Some(site_keep_precomputed) = site_keep_precomputed {
        let _ = site_keep_precomputed;
        let meta = prepare_bed_logic_meta_owned_for_stats_samples_pure_line(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            Some(selected_sample_indices.as_slice()),
        )?;
        let logic_row_flip = if using_external_grm {
            Some(meta.row_flip.clone())
        } else {
            None
        };
        (
            meta.row_source_indices,
            logic_row_flip,
            meta.sites,
            meta.n_samples,
            meta.bytes_per_snp,
        )
    } else {
        let meta = prepare_bed_logic_meta_owned_for_stats_samples_pure_line(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            Some(selected_sample_indices.as_slice()),
        )?;
        let logic_row_flip = if using_external_grm {
            Some(meta.row_flip.clone())
        } else {
            None
        };
        (
            meta.row_source_indices,
            logic_row_flip,
            meta.sites,
            meta.n_samples,
            meta.bytes_per_snp,
        )
    };
    let n_selected = selected_sample_indices.len();
    if y.len() != n_selected {
        return Err(format!(
            "y length mismatch: got {}, expected {} selected samples from BED input",
            y.len(),
            n_selected
        ));
    }
    if q_cov > 0 {
        let cov = x_cov
            .as_ref()
            .ok_or_else(|| "q_cov > 0 but x_cov is missing".to_string())?;
        if cov.len() != n_selected.saturating_mul(q_cov) {
            return Err(format!(
                "x_cov payload length mismatch: got {}, expected {}",
                cov.len(),
                n_selected.saturating_mul(q_cov)
            ));
        }
    }

    let split_applied = fold >= 2;
    let (train_idx_local, test_idx_local) = if split_applied {
        let test_mask = stratified_test_mask(&y, fold, seed)?;
        let train_idx_local = mask_to_indices(&test_mask, false);
        let test_idx_local = mask_to_indices(&test_mask, true);
        if train_idx_local.is_empty() || test_idx_local.is_empty() {
            return Err("train/test split produced an empty partition".to_string());
        }
        (train_idx_local, test_idx_local)
    } else {
        let full = (0..n_selected).collect::<Vec<_>>();
        (full.clone(), full)
    };
    let train_idx = train_idx_local
        .iter()
        .map(|&i| selected_sample_indices[i])
        .collect::<Vec<_>>();
    let test_idx = test_idx_local
        .iter()
        .map(|&i| selected_sample_indices[i])
        .collect::<Vec<_>>();

    let y_train = subset_vec_f64(&y, &train_idx_local)?;
    let y_test = subset_vec_f64(&y, &test_idx_local)?;
    let x_train = subset_cov_f64(x_cov.as_deref(), n_selected, q_cov, &train_idx_local)?;
    let x_test = subset_cov_f64(x_cov.as_deref(), n_selected, q_cov, &test_idx_local)?;
    let ml_min_samples_leaf =
        strict_maf_support_count(train_idx_local.len(), logic_maf_threshold).max(1);
    let tree_cfg = ExtraTreesConfig {
        min_samples_leaf: tree_cfg.min_samples_leaf.max(ml_min_samples_leaf),
        min_samples_split: tree_cfg
            .min_samples_split
            .max(ml_min_samples_leaf.saturating_mul(2))
            .max(2),
        ..tree_cfg
    };

    let threads_eff = effective_threads_local(threads);
    let mut auto_grm_full: Option<Vec<f64>> = None;
    let mut auto_eff_m_full: Option<usize> = None;
    let mut logic_row_flip_auto: Option<Vec<bool>> = None;
    if !using_external_grm {
        let row_meta = compute_bed_row_meta_owned_for_source_rows(
            &prefix,
            grm_row_source_indices.as_slice(),
            Some(selected_sample_indices.as_slice()),
        )?;
        if row_meta.n_samples != n_samples_total {
            return Err(format!(
                "internal error: additive row-meta sample count {} != expected {}",
                row_meta.n_samples, n_samples_total
            ));
        }
        if row_meta.bytes_per_snp != bytes_per_snp {
            return Err(format!(
                "internal error: additive row-meta bytes_per_snp {} != expected {}",
                row_meta.bytes_per_snp, bytes_per_snp
            ));
        }
        let (grm_full_auto, _row_sum, varsum_full) = build_grm_from_meta_stream(
            &prefix,
            grm_row_source_indices.as_slice(),
            row_meta.row_flip.as_slice(),
            row_meta.maf.as_slice(),
            selected_sample_indices.as_slice(),
            StreamKernelMode::Additive,
            block_cols,
            threads_eff,
            None,
            0usize,
            None::<fn(usize, usize) -> Result<(), String>>,
        )?;
        auto_eff_m_full = Some(varsum_full.round().max(0.0) as usize);
        logic_row_flip_auto = Some(row_meta.row_flip);
        auto_grm_full = Some(grm_full_auto);
    }
    let (train_fit, test_fit) = if split_applied {
        let (grm_train, eff_m_train) = if let Some(full_grm) = grm.as_ref() {
            (
                slice_square_matrix_by_indices(
                    full_grm.as_slice(),
                    n_selected,
                    train_idx_local.as_slice(),
                    "garfield_logic_search_bed: train GRM slice",
                )?,
                None,
            )
        } else {
            (
                slice_square_matrix_by_indices(
                    auto_grm_full
                        .as_ref()
                        .ok_or_else(|| {
                            "internal error: auto GRM missing for train slice".to_string()
                        })?
                        .as_slice(),
                    n_selected,
                    train_idx_local.as_slice(),
                    "garfield_logic_search_bed: auto train GRM slice",
                )?,
                auto_eff_m_full,
            )
        };
        let train_fit = garfield_residualize_exact_from_grm_rust(
            grm_train,
            train_idx.len(),
            y_train,
            x_train,
            q_cov,
            threads_eff,
            low,
            high,
            max_iter,
            tol,
            add_intercept,
            exact_n_max,
            require_lapack,
            eff_m_train,
        )?;

        let (grm_test, eff_m_test) = if let Some(full_grm) = grm.as_ref() {
            (
                slice_square_matrix_by_indices(
                    full_grm.as_slice(),
                    n_selected,
                    test_idx_local.as_slice(),
                    "garfield_logic_search_bed: test GRM slice",
                )?,
                None,
            )
        } else {
            (
                slice_square_matrix_by_indices(
                    auto_grm_full
                        .as_ref()
                        .ok_or_else(|| {
                            "internal error: auto GRM missing for test slice".to_string()
                        })?
                        .as_slice(),
                    n_selected,
                    test_idx_local.as_slice(),
                    "garfield_logic_search_bed: auto test GRM slice",
                )?,
                auto_eff_m_full,
            )
        };
        let test_fit = garfield_residualize_exact_from_grm_rust(
            grm_test,
            test_idx.len(),
            y_test,
            x_test,
            q_cov,
            threads_eff,
            low,
            high,
            max_iter,
            tol,
            add_intercept,
            exact_n_max,
            require_lapack,
            eff_m_test,
        )?;
        (train_fit, test_fit)
    } else {
        let (grm_full, eff_m_full) = if let Some(full_grm) = grm.as_ref() {
            if full_grm.len() != n_selected.saturating_mul(n_selected) {
                return Err(format!(
                    "garfield_logic_search_bed: provided GRM length mismatch: got {}, expected {}",
                    full_grm.len(),
                    n_selected.saturating_mul(n_selected)
                ));
            }
            (full_grm.clone(), None)
        } else {
            (
                auto_grm_full
                    .take()
                    .ok_or_else(|| "internal error: auto GRM missing for full fit".to_string())?,
                auto_eff_m_full,
            )
        };
        let fit = garfield_residualize_exact_from_grm_rust(
            grm_full,
            train_idx.len(),
            y_train,
            x_train,
            q_cov,
            threads_eff,
            low,
            high,
            max_iter,
            tol,
            add_intercept,
            exact_n_max,
            require_lapack,
            eff_m_full,
        )?;
        (fit.clone(), fit)
    };

    let mode = parse_bin_mode(&bin_mode)?;
    let logic_row_mul = match mode {
        GarfieldBinMode::Bin => 1usize,
        GarfieldBinMode::Mbin => 3usize,
    };
    let logic_row_flip = if using_external_grm {
        logic_row_flip_external.as_ref().ok_or_else(|| {
            "internal error: logic row_flip missing for BED conversion".to_string()
        })?
    } else {
        logic_row_flip_auto.as_ref().ok_or_else(|| {
            "internal error: auto logic row_flip missing for BED conversion".to_string()
        })?
    };
    let null_chunk_bp = extension.max(1).saturating_mul(2);
    let null_chunk_target = if rule_permutation {
        DEFAULT_RULE_NULL_PHYSICAL_CHUNKS
    } else {
        0
    };
    let (null_chunks, null_chunk_valid_total) = if rule_permutation {
        sample_null_chunks_stratified(
            filtered_sites.as_slice(),
            extension.max(1),
            DEFAULT_RULE_NULL_PHYSICAL_CHUNKS,
            DEFAULT_RULE_NULL_MIN_SNPS_PER_CHUNK,
            seed,
        )?
    } else {
        (Vec::new(), 0usize)
    };
    let global_bits_mem_tracker = GarfieldStageMemoryTracker::new(rss_debug_enabled);
    let global_bits_mem_start = global_bits_mem_tracker.start_stage();
    let logic_bits = convert_bed_prefix_to_logic_bits(
        &prefix,
        bytes_per_snp,
        n_samples_total,
        grm_row_source_indices.as_slice(),
        logic_row_flip.as_slice(),
        filtered_sites,
        selected_sample_indices.as_slice(),
        selected_sample_ids.as_slice(),
        mode,
        Some(&global_bits_mem_tracker),
    )?;
    if let Some(debug) = memory_debug.as_mut() {
        debug.global_bits_loaded = global_bits_mem_tracker.finish_stage(global_bits_mem_start);
    }
    let units = build_logic_units(
        logic_bits.sites.as_slice(),
        &unit_kind,
        groups.as_deref(),
        group_names.as_deref(),
        extension,
        step,
    )?;
    if units.is_empty() {
        return Err("no scan units were built from the provided input".to_string());
    }
    let scan_bimranges = scan_bimranges
        .as_deref()
        .map(parse_scan_bimranges)
        .transpose()?
        .unwrap_or_default();

    let response = ResponseKind::Continuous;
    let engine = parse_optional_ml_engine(&ml_method)?;
    let importance = parse_importance(&ml_importance)?;
    let rank_mode = parse_beam_rank_mode(&rank_score)?;
    let perm_cfg = PermutationConfig {
        n_repeats: permutation_repeats
            .max(DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS)
            .min(DEFAULT_RULE_NULL_MAX_REPEATS),
        scoring: parse_permutation_scoring(&permutation_scoring)?,
        seed,
    };
    let structure_prior_cfg = RuleStructurePriorConfig::from_len_alpha_values(prior_len.as_deref());
    let structure_prior_display_len = max_pick.clamp(1, 5);
    let beam_params = BeamSearchParams {
        max_pick: max_pick.max(1),
        beam_width: beam_width.max(1),
        min_gain: 0.0,
        min_parent_abs_gain: 0.0,
        surrogate_test_gain_max: 0.0,
        surrogate_hamming_frac_max: 0.0,
        maf_threshold: logic_maf_threshold.clamp(0.0, 0.5) as f64,
        lambda_len: 0.0,
        lambda_not: 0.0,
        exhaustive_depth: exhaustive_depth.max(1),
        rank_mode,
        null_penalties: None,
        structure_prior: None,
        disable_parent_delta: false,
        group_constraint: BeamGroupConstraintMode::AlwaysExclude,
        allow_parallel: true,
    };
    let unit_kind_lc = unit_kind.trim().to_ascii_lowercase();
    let total_units = units.len();
    let scan_unit_indices = if scan_bimranges.is_empty() {
        (0..total_units).collect::<Vec<_>>()
    } else {
        units
            .iter()
            .enumerate()
            .filter_map(|(ui, unit)| {
                if unit_overlaps_scan_bimranges(unit, scan_bimranges.as_slice()) {
                    Some(ui)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };
    if scan_unit_indices.is_empty() {
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
    let scanned_units = scan_unit_indices.len();
    let progress_callback_parallel = progress_callback.as_ref();
    let mut null_chunk_prepared = Vec::<(GarfieldNullChunk, GarfieldUnitPrepared)>::new();
    if !null_chunks.is_empty() {
        let prep_beam_params = BeamSearchParams {
            allow_parallel: false,
            ..beam_params.clone()
        };
        let prep_notify_step = if progress_every == 0 {
            (null_chunks.len().max(1) / 200).max(1)
        } else {
            progress_every.max(1)
        };
        let prep_progress_done = AtomicUsize::new(0);
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "null_prep",
            0,
            null_chunks.len().max(1),
            None,
        )
        .map_err(|e| e.to_string())?;
        let prep_threads = threads_eff.min(null_chunks.len().max(1));
        let prep_results = if prep_threads > 1 {
            let pool = ThreadPoolBuilder::new()
                .num_threads(prep_threads)
                .build()
                .map_err(|e| format!("build GARFIELD null-prep thread pool: {e}"))?;
            pool.install(|| {
                null_chunks
                    .par_iter()
                    .copied()
                    .map(|chunk| {
                        let prepared = prepare_logic_chunk_continuous(
                            chunk,
                            logic_row_mul,
                            response,
                            engine,
                            importance,
                            perm_cfg,
                            ml_top_k,
                            ml_top_frac,
                            tree_cfg,
                            &logic_bits,
                            train_idx_local.as_slice(),
                            test_idx_local.as_slice(),
                            train_fit.residualized_y.as_slice(),
                            prep_beam_params.clone(),
                        );
                        let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % prep_notify_step == 0 || done == null_chunks.len() {
                            garfield_stage_progress_notify(
                                progress_callback_parallel,
                                "null_prep",
                                done,
                                null_chunks.len().max(1),
                                None,
                            )
                            .map_err(|e| e.to_string())?;
                        }
                        prepared.map(|opt| opt.map(|p| (chunk, p)))
                    })
                    .collect::<Vec<Result<Option<(GarfieldNullChunk, GarfieldUnitPrepared)>, String>>>()
            })
        } else {
            let mut out = Vec::<Result<Option<(GarfieldNullChunk, GarfieldUnitPrepared)>, String>>::with_capacity(
                null_chunks.len(),
            );
            for chunk in null_chunks.iter().copied() {
                let prepared = prepare_logic_chunk_continuous(
                    chunk,
                    logic_row_mul,
                    response,
                    engine,
                    importance,
                    perm_cfg,
                    ml_top_k,
                    ml_top_frac,
                    tree_cfg,
                    &logic_bits,
                    train_idx_local.as_slice(),
                    test_idx_local.as_slice(),
                    train_fit.residualized_y.as_slice(),
                    prep_beam_params.clone(),
                );
                let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % prep_notify_step == 0 || done == null_chunks.len() {
                    garfield_stage_progress_notify(
                        progress_callback.as_ref(),
                        "null_prep",
                        done,
                        null_chunks.len().max(1),
                        None,
                    )
                    .map_err(|e| e.to_string())?;
                }
                out.push(prepared.map(|opt| opt.map(|p| (chunk, p))));
            }
            out
        };
        for prep_out in prep_results.into_iter() {
            if let Some((chunk, prepared)) = prep_out? {
                null_chunk_prepared.push((chunk, prepared));
            }
        }
    }
    let null_chunk_selected = null_chunk_prepared.len();
    let null_permutation_active = rule_permutation && !null_chunk_prepared.is_empty();
    let representative_units_target = if rule_permutation && !GARFIELD_DISABLE_STRUCTURE_PRIOR {
        DEFAULT_RULE_PERMUTATION_REPRESENTATIVE_UNITS.min(units.len())
    } else {
        0
    };

    let representative_units = if rule_permutation && !GARFIELD_DISABLE_STRUCTURE_PRIOR {
        choose_representative_indices(
            units
                .iter()
                .map(|unit| unit.indices.len())
                .collect::<Vec<_>>()
                .as_slice(),
            representative_units_target,
        )
    } else {
        Vec::new()
    };
    let mut representative_prepared = Vec::<(usize, GarfieldUnitPrepared)>::new();
    if !representative_units.is_empty() {
        let prep_beam_params = BeamSearchParams {
            allow_parallel: false,
            ..beam_params.clone()
        };
        let prep_notify_step = if progress_every == 0 {
            (representative_units.len().max(1) / 200).max(1)
        } else {
            progress_every.max(1)
        };
        let prep_progress_done = AtomicUsize::new(0);
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "structure_prep",
            0,
            representative_units.len().max(1),
            None,
        )
        .map_err(|e| e.to_string())?;
        let prep_threads = threads_eff.min(representative_units.len().max(1));
        let prep_results = if prep_threads > 1 {
            let pool = ThreadPoolBuilder::new()
                .num_threads(prep_threads)
                .build()
                .map_err(|e| format!("build GARFIELD prior-prep thread pool: {e}"))?;
            pool.install(|| {
                representative_units
                    .par_iter()
                    .copied()
                    .map(|ui| {
                        let prepared = prepare_logic_unit_continuous(
                            &units[ui],
                            &unit_kind_lc,
                            response,
                            engine,
                            importance,
                            perm_cfg,
                            ml_top_k,
                            ml_top_frac,
                            tree_cfg,
                            &logic_bits,
                            train_idx_local.as_slice(),
                            test_idx_local.as_slice(),
                            train_fit.residualized_y.as_slice(),
                            prep_beam_params.clone(),
                        );
                        let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % prep_notify_step == 0 || done == representative_units.len() {
                            garfield_stage_progress_notify(
                                progress_callback_parallel,
                                "structure_prep",
                                done,
                                representative_units.len().max(1),
                                None,
                            )
                            .map_err(|e| e.to_string())?;
                        }
                        prepared.map(|opt| opt.map(|p| (ui, p)))
                    })
                    .collect::<Vec<Result<Option<(usize, GarfieldUnitPrepared)>, String>>>()
            })
        } else {
            let mut out =
                Vec::<Result<Option<(usize, GarfieldUnitPrepared)>, String>>::with_capacity(
                    representative_units.len(),
                );
            for &ui in representative_units.iter() {
                let prepared = prepare_logic_unit_continuous(
                    &units[ui],
                    &unit_kind_lc,
                    response,
                    engine,
                    importance,
                    perm_cfg,
                    ml_top_k,
                    ml_top_frac,
                    tree_cfg,
                    &logic_bits,
                    train_idx_local.as_slice(),
                    test_idx_local.as_slice(),
                    train_fit.residualized_y.as_slice(),
                    prep_beam_params.clone(),
                );
                let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % prep_notify_step == 0 || done == representative_units.len() {
                    garfield_stage_progress_notify(
                        progress_callback.as_ref(),
                        "structure_prep",
                        done,
                        representative_units.len().max(1),
                        None,
                    )
                    .map_err(|e| e.to_string())?;
                }
                out.push(prepared.map(|opt| opt.map(|p| (ui, p))));
            }
            out
        };
        for prep_out in prep_results.into_iter() {
            if let Some((ui, prepared)) = prep_out? {
                representative_prepared.push((ui, prepared));
            }
        }
    }
    let representative_units_used = representative_prepared.len();
    let soft_structure_mode = rule_permutation
        && !GARFIELD_DISABLE_STRUCTURE_PRIOR
        && !representative_prepared.is_empty();
    let permutation_task_total = if null_permutation_active {
        null_chunk_prepared
            .len()
            .saturating_mul(perm_cfg.n_repeats.max(1))
    } else {
        0
    };
    let structure_task_total = if soft_structure_mode {
        representative_prepared.len()
    } else {
        0
    };

    let mut permutation_null_repeats_used = 0usize;
    let rule_null_lookup: Option<Arc<RuleNullPenaltyLookup>> = if null_permutation_active {
        let null_mem_tracker = GarfieldStageMemoryTracker::new(rss_debug_enabled);
        let null_mem_start = null_mem_tracker.start_stage();
        let null_notify_step = if progress_every == 0 {
            (permutation_task_total.max(1) / 200).max(1)
        } else {
            progress_every.max(1)
        };
        let null_progress_done = AtomicUsize::new(0);
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "null_penalty",
            0,
            permutation_task_total.max(1),
            None,
        )
        .map_err(|e| e.to_string())?;
        let perm_beam_params = BeamSearchParams {
            allow_parallel: false,
            ..beam_params.clone()
        };
        let mut bucket_scores = RuleNullCalibrator::with_max_rule_len(beam_params.max_pick.max(1));
        let min_perm_repeats =
            DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS.min(perm_cfg.n_repeats.max(1));
        let mut stable_rounds = 0usize;
        let mut prev_lookup: Option<RuleNullPenaltyLookup> = None;
        let perm_threads = threads_eff.min(
            null_chunk_prepared
                .len()
                .saturating_mul(perm_cfg.n_repeats.max(1))
                .max(1),
        );
        let repeats_per_batch = if null_chunk_prepared.is_empty() {
            1usize
        } else {
            threads_eff
                .div_ceil(null_chunk_prepared.len().max(1))
                .clamp(1, 4)
        };
        let perm_pool = if perm_threads > 1 {
            Some(
                ThreadPoolBuilder::new()
                    .num_threads(perm_threads)
                    .build()
                    .map_err(|e| format!("build GARFIELD permutation thread pool: {e}"))?,
            )
        } else {
            None
        };
        let max_perm_repeats = perm_cfg.n_repeats.max(1);
        let mut rep_start = 0usize;
        'perm_batches: while rep_start < max_perm_repeats {
            let rep_end = (rep_start + repeats_per_batch).min(max_perm_repeats);
            let n_slots = null_chunk_prepared.len();
            let slot_indices = (0..n_slots).collect::<Vec<_>>();
            let slot_grain = if perm_threads > 1 {
                n_slots.div_ceil(perm_threads.saturating_mul(4)).max(1)
            } else {
                n_slots.max(1)
            };
            let batch_results: Vec<Result<Vec<(usize, Vec<(RuleNullBucket, f64, f64)>)>, String>> =
                if let Some(pool) = perm_pool.as_ref() {
                    pool.install(|| {
                        slot_indices
                            .par_chunks(slot_grain)
                            .map(|slot_chunk| {
                                process_rule_permutation_task_chunk(
                                    slot_chunk,
                                    rep_start,
                                    rep_end,
                                    null_chunk_prepared.as_slice(),
                                    &logic_bits,
                                    train_idx_local.as_slice(),
                                    test_idx_local.as_slice(),
                                    train_fit.residualized_y.as_slice(),
                                    test_fit.residualized_y.as_slice(),
                                    split_applied,
                                    perm_beam_params.clone(),
                                    seed,
                                    progress_callback_parallel,
                                    &null_progress_done,
                                    null_notify_step,
                                    permutation_task_total,
                                    Some(&null_mem_tracker),
                                )
                            })
                            .collect()
                    })
                } else {
                    slot_indices
                        .chunks(slot_grain)
                        .map(|slot_chunk| {
                            process_rule_permutation_task_chunk(
                                slot_chunk,
                                rep_start,
                                rep_end,
                                null_chunk_prepared.as_slice(),
                                &logic_bits,
                                train_idx_local.as_slice(),
                                test_idx_local.as_slice(),
                                train_fit.residualized_y.as_slice(),
                                test_fit.residualized_y.as_slice(),
                                split_applied,
                                perm_beam_params.clone(),
                                seed,
                                progress_callback.as_ref(),
                                &null_progress_done,
                                null_notify_step,
                                permutation_task_total,
                                Some(&null_mem_tracker),
                            )
                        })
                        .collect()
                };
            let mut by_rep = vec![Vec::<(RuleNullBucket, f64, f64)>::new(); rep_end - rep_start];
            for chunk_out in batch_results.into_iter() {
                for (rep, vals) in chunk_out? {
                    by_rep[rep - rep_start].extend(vals);
                }
            }
            for (rep_offset, rep_vals) in by_rep.into_iter().enumerate() {
                for (bucket, train_score, test_score) in rep_vals.into_iter() {
                    bucket_scores.insert(bucket, train_score, test_score);
                }
                permutation_null_repeats_used = rep_start + rep_offset + 1;
                if permutation_null_repeats_used < min_perm_repeats {
                    continue;
                }
                let current_lookup = bucket_scores.finalize();
                if let Some(prev) = prev_lookup.as_ref() {
                    if current_lookup.q99_converged_against(prev) {
                        stable_rounds += 1;
                    } else {
                        stable_rounds = 0;
                    }
                }
                prev_lookup = Some(current_lookup);
                if stable_rounds >= DEFAULT_RULE_NULL_ADAPTIVE_STABLE_REPEATS {
                    break 'perm_batches;
                }
            }
            rep_start = rep_end;
        }
        if null_progress_done.load(Ordering::Relaxed) < permutation_task_total {
            garfield_stage_progress_notify(
                progress_callback.as_ref(),
                "null_penalty",
                permutation_task_total.max(1),
                permutation_task_total.max(1),
                None,
            )
            .map_err(|e| e.to_string())?;
        }
        if let Some(debug) = memory_debug.as_mut() {
            debug.null_penalty = null_mem_tracker.finish_stage(null_mem_start);
        }
        let lookup = prev_lookup.unwrap_or_else(|| bucket_scores.finalize());
        Some(Arc::new(lookup))
    } else {
        None
    };
    let mut structure_bootstrap_repeats_used_total = 0usize;
    let rule_structure_prior: Option<Arc<RuleStructurePrior>> = if soft_structure_mode {
        let structure_mem_tracker = GarfieldStageMemoryTracker::new(rss_debug_enabled);
        let structure_mem_start = structure_mem_tracker.start_stage();
        let structure_notify_step = if progress_every == 0 {
            (structure_task_total.max(1) / 200).max(1)
        } else {
            progress_every.max(1)
        };
        let structure_progress_done = AtomicUsize::new(0);
        let structure_scores = Arc::new(Mutex::new(RuleStructurePriorCalibrator::default()));
        let structure_repeats_used = AtomicUsize::new(0);
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "structure_prior",
            0,
            structure_task_total.max(1),
            Some(format_structure_alpha_placeholder(structure_prior_display_len).as_str()),
        )
        .map_err(|e| e.to_string())?;
        let posterior_beam_params = BeamSearchParams {
            allow_parallel: false,
            null_penalties: rule_null_lookup.clone(),
            structure_prior: None,
            ..beam_params.clone()
        };
        let bootstrap_threads = threads_eff.min(representative_prepared.len().max(1));
        let structure_chunk_units = garfield_task_chunk_units(
            representative_prepared.len(),
            bootstrap_threads,
            garfield_structure_task_coalesce_max_units(),
        );
        if bootstrap_threads > 1 {
            let pool = ThreadPoolBuilder::new()
                .num_threads(bootstrap_threads)
                .build()
                .map_err(|e| format!("build GARFIELD bootstrap thread pool: {e}"))?;
            pool.install(|| {
                representative_prepared
                    .par_chunks(structure_chunk_units)
                    .enumerate()
                    .map(|(chunk_idx, chunk)| {
                        process_structure_prior_unit_chunk(
                            chunk_idx.saturating_mul(structure_chunk_units),
                            chunk,
                            &logic_bits,
                            train_idx_local.as_slice(),
                            test_idx_local.as_slice(),
                            train_fit.residualized_y.as_slice(),
                            test_fit.residualized_y.as_slice(),
                            split_applied,
                            posterior_beam_params.clone(),
                            &structure_prior_cfg,
                            seed,
                            &structure_scores,
                            &structure_repeats_used,
                            &structure_progress_done,
                            structure_notify_step,
                            structure_task_total,
                            structure_prior_display_len,
                            progress_callback_parallel,
                            Some(&structure_mem_tracker),
                        )
                    })
                    .collect::<Vec<Result<(), String>>>()
            })
        } else {
            for (chunk_idx, chunk) in representative_prepared
                .chunks(structure_chunk_units)
                .enumerate()
            {
                process_structure_prior_unit_chunk(
                    chunk_idx.saturating_mul(structure_chunk_units),
                    chunk,
                    &logic_bits,
                    train_idx_local.as_slice(),
                    test_idx_local.as_slice(),
                    train_fit.residualized_y.as_slice(),
                    test_fit.residualized_y.as_slice(),
                    split_applied,
                    posterior_beam_params.clone(),
                    &structure_prior_cfg,
                    seed,
                    &structure_scores,
                    &structure_repeats_used,
                    &structure_progress_done,
                    structure_notify_step,
                    structure_task_total,
                    structure_prior_display_len,
                    progress_callback.as_ref(),
                    Some(&structure_mem_tracker),
                )?;
            }
            Vec::<Result<(), String>>::new()
        }
        .into_iter()
        .try_for_each(|res| res)?;
        if let Some(debug) = memory_debug.as_mut() {
            debug.structure_prior = structure_mem_tracker.finish_stage(structure_mem_start);
        }
        structure_bootstrap_repeats_used_total = structure_repeats_used.load(Ordering::Relaxed);
        let final_structure_scores = structure_scores
            .lock()
            .map_err(|_| "GARFIELD structure prior mutex poisoned".to_string())?
            .clone();
        Some(Arc::new(
            final_structure_scores.finalize(&structure_prior_cfg),
        ))
    } else {
        None
    };
    let beam_params = BeamSearchParams {
        null_penalties: rule_null_lookup,
        structure_prior: rule_structure_prior.clone(),
        disable_parent_delta: soft_structure_mode,
        min_gain: 0.0,
        min_parent_abs_gain: if no_clean {
            0.0
        } else {
            GARFIELD_SCAN_PAIR_PARENT_ABS_GAIN_MIN
        },
        surrogate_test_gain_max: if no_clean {
            0.0
        } else {
            GARFIELD_SCAN_SURROGATE_TEST_GAIN_MAX
        },
        surrogate_hamming_frac_max: if no_clean {
            0.0
        } else {
            GARFIELD_SCAN_SURROGATE_HAMMING_FRAC_MAX
        },
        ..beam_params
    };

    let scan_notify_step = if progress_every == 0 {
        (scanned_units.max(1) / 200).max(1)
    } else {
        progress_every.max(1)
    };
    reset_garfield_beam_profile();
    reset_pairwise_profile();
    PACKED_EXTRACT_FLAT_NS.store(0, Ordering::Relaxed);
    GARFIELD_DENSE_DOSAGE_DECODE_NS.store(0, Ordering::Relaxed);
    GARFIELD_GENESET_LD_PRUNE_NS.store(0, Ordering::Relaxed);
    GARFIELD_GENESET_LD_EXACT_PAIRS.store(0, Ordering::Relaxed);
    GARFIELD_GENESET_LD_ROWS_TOTAL.store(0, Ordering::Relaxed);
    GARFIELD_GENESET_LD_ROWS_KEPT.store(0, Ordering::Relaxed);
    garfield_geneset_ld_unit_stats_reset();
    GARFIELD_MATERIALIZE_BITS_NS.store(0, Ordering::Relaxed);
    GARFIELD_ML_SELECT_NS.store(0, Ordering::Relaxed);
    let scan_stage_t0 = Instant::now();
    let scan_progress_done = AtomicUsize::new(0);
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "scan",
        0,
        scanned_units.max(1),
        None,
    )
    .map_err(|e| e.to_string())?;

    let (assoc_y, assoc_sample_indices): (&[f64], &[usize]) = if split_applied {
        (
            test_fit.residualized_y.as_slice(),
            test_idx_local.as_slice(),
        )
    } else {
        (
            train_fit.residualized_y.as_slice(),
            train_idx_local.as_slice(),
        )
    };
    let unit_parallel_threads = threads_eff.max(1);
    let scan_beam_params = BeamSearchParams {
        // Keep nested parallelism enabled during scan. Rayon will reuse the same
        // pool and lets a few heavy tail units fan out when outer unit-level
        // parallelism no longer saturates all workers.
        allow_parallel: true,
        ..beam_params.clone()
    };
    let scan_mem_tracker = GarfieldStageMemoryTracker::new(rss_debug_enabled);
    let scan_mem_start = scan_mem_tracker.start_stage();
    let skipped_units = Arc::new(Mutex::new(Vec::<GarfieldSkippedUnitInfo>::new()));
    let unit_results = if unit_parallel_threads > 1 {
        let pool = ThreadPoolBuilder::new()
            .num_threads(unit_parallel_threads)
            .build()
            .map_err(|e| format!("build GARFIELD unit thread pool: {e}"))?;
        let skipped_units_parallel = skipped_units.clone();
        pool.install(|| {
            scan_unit_indices
                .par_iter()
                .map(|&ui| {
                    process_scan_unit_continuous(
                        ui,
                        units.as_slice(),
                        response,
                        engine,
                        importance,
                        perm_cfg,
                        ml_top_k,
                        ml_top_frac,
                        tree_cfg,
                        &logic_bits,
                        train_idx_local.as_slice(),
                        test_idx_local.as_slice(),
                        train_fit.residualized_y.as_slice(),
                        test_fit.residualized_y.as_slice(),
                        scan_beam_params.clone(),
                        top_rules_per_unit,
                        &unit_kind_lc,
                        progress_callback_parallel,
                        &scan_progress_done,
                        scan_notify_step,
                        scanned_units,
                        &skipped_units_parallel,
                        Some(&scan_mem_tracker),
                    )
                })
                .collect::<Vec<Result<Vec<GarfieldLogicRuleRecord>, String>>>()
        })
    } else {
        let mut out = Vec::<Result<Vec<GarfieldLogicRuleRecord>, String>>::with_capacity(
            scan_unit_indices.len(),
        );
        for &ui in scan_unit_indices.iter() {
            let chunk_out = process_scan_unit_continuous(
                ui,
                units.as_slice(),
                response,
                engine,
                importance,
                perm_cfg,
                ml_top_k,
                ml_top_frac,
                tree_cfg,
                &logic_bits,
                train_idx_local.as_slice(),
                test_idx_local.as_slice(),
                train_fit.residualized_y.as_slice(),
                test_fit.residualized_y.as_slice(),
                scan_beam_params.clone(),
                top_rules_per_unit,
                &unit_kind_lc,
                progress_callback.as_ref(),
                &scan_progress_done,
                scan_notify_step,
                scanned_units,
                &skipped_units,
                Some(&scan_mem_tracker),
            );
            out.push(chunk_out);
        }
        out
    };
    let mut records = Vec::<GarfieldLogicRuleRecord>::new();
    for unit_out in unit_results.into_iter() {
        records.extend(unit_out?);
    }
    let skipped_units = Arc::try_unwrap(skipped_units)
        .map_err(|_| "GARFIELD skipped-units still shared".to_string())?
        .into_inner()
        .map_err(|_| "GARFIELD skipped-units mutex poisoned".to_string())?;
    let skipped_messages = skipped_units
        .iter()
        .map(format_skipped_unit_message)
        .collect::<Vec<_>>();
    if let Some(debug) = memory_debug.as_mut() {
        debug.scan = scan_mem_tracker.finish_stage(scan_mem_start);
    }
    let scan_stage_wall_s = scan_stage_t0.elapsed().as_secs_f64();
    let ld_unit_stats = garfield_geneset_ld_unit_stats_snapshot();
    let timing_scan_ml_select_wall_s =
        (GARFIELD_ML_SELECT_NS.load(Ordering::Relaxed) as f64) * 1e-9;
    let scan_beam_profile = snapshot_garfield_beam_profile();
    let (pw_marg, pw_pack, pw_kern, pw_comb) = snapshot_pairwise_profile();

    records = dedup_logic_rule_records(records, &logic_bits)?;
    apply_logic_rule_output_limit(&mut records, max_output_rules, max_output_ratio)?;
    let simbench_count = if let Some(path) = simbench_path.as_ref() {
        let simbench_terms = parse_simbench_terms(path)?;
        let mut logic_site_lookup =
            HashMap::<(String, i32), Vec<usize>>::with_capacity(logic_bits.sites.len());
        for (logic_idx, site) in logic_bits.sites.iter().enumerate() {
            logic_site_lookup
                .entry((normalize_chrom(site.garfield_chrom()), site.garfield_pos()))
                .or_default()
                .push(logic_idx);
        }
        let simbench_ml_contexts = build_simbench_ml_contexts(
            simbench_terms.as_slice(),
            units.as_slice(),
            scan_unit_indices.as_slice(),
            &unit_kind_lc,
            response,
            engine,
            importance,
            perm_cfg,
            ml_top_k,
            ml_top_frac,
            tree_cfg,
            &logic_bits,
            train_idx_local.as_slice(),
            train_fit.residualized_y.as_slice(),
            unit_parallel_threads <= 1,
        )?;
        let simbench_records = evaluate_simbench_terms(
            &prefix,
            simbench_terms.as_slice(),
            &logic_site_lookup,
            selected_sample_indices.as_slice(),
            train_idx_local.as_slice(),
            assoc_sample_indices,
            train_fit.residualized_y.as_slice(),
            assoc_y,
            beam_params,
            simbench_ml_contexts.as_slice(),
        )?;
        let n = simbench_records.len();
        records.extend(simbench_records);
        n
    } else {
        0usize
    };
    let total_wall_s = total_wall_t0.elapsed().as_secs_f64();
    let timing_scan_beam_wall_s = scan_beam_profile.total_s;
    let timing_scan_literal_score_wall_s = scan_beam_profile.literal_precompute_s;
    let timing_literal_score_share_of_total_pct =
        timing_share_pct(timing_scan_literal_score_wall_s, total_wall_s);
    let timing_literal_score_share_of_scan_pct =
        timing_share_pct(timing_scan_literal_score_wall_s, scan_stage_wall_s);
    let timing_literal_score_share_of_beam_pct =
        timing_share_pct(timing_scan_literal_score_wall_s, timing_scan_beam_wall_s);
    let timing_beam_share_of_total_pct = timing_share_pct(timing_scan_beam_wall_s, total_wall_s);
    let timing_beam_share_of_scan_pct =
        timing_share_pct(timing_scan_beam_wall_s, scan_stage_wall_s);

    let structure_prior_for_output = rule_structure_prior.clone();
    let mut pseudo_prefix_out = None;
    let mut rules_tsv_out = None;
    let mut posterior_json_out = None;
    if let Some(prefix_out) = out_prefix.as_ref() {
        write_logic_pseudo_plink(
            prefix_out,
            logic_bits.sample_ids.as_slice(),
            records.as_slice(),
            &logic_bits,
        )?;
        let rules_tsv = format!("{prefix_out}.rules.tsv");
        write_logic_rules_tsv(&rules_tsv, records.as_slice())?;
        if let Some(prior) = structure_prior_for_output.as_deref() {
            let posterior_json = format!("{prefix_out}.posterior.json");
            write_rule_structure_prior_json(&posterior_json, prior)?;
            posterior_json_out = Some(posterior_json);
        }
        pseudo_prefix_out = Some(prefix_out.clone());
        rules_tsv_out = Some(rules_tsv);
    }

    Ok(GarfieldLogicPipelineResult {
        pseudo_prefix: pseudo_prefix_out,
        rules_tsv: rules_tsv_out,
        posterior_json: posterior_json_out,
        memory_debug,
        rule_permutation_active: null_permutation_active || soft_structure_mode,
        null_chunk_bp,
        null_chunk_min_snps: DEFAULT_RULE_NULL_MIN_SNPS_PER_CHUNK,
        null_chunk_target,
        null_chunk_valid_total,
        null_chunk_selected,
        representative_units_target,
        representative_units_used,
        permutation_null_repeats: if null_permutation_active {
            permutation_null_repeats_used
        } else {
            0
        },
        permutation_bootstrap_repeats: if soft_structure_mode {
            structure_bootstrap_repeats_used_total
        } else {
            0
        },
        records,
        simbench_count,
        split_applied,
        n_train: train_idx_local.len(),
        n_test: test_idx_local.len(),
        n_samples: logic_bits.n_samples,
        units_total: total_units,
        units_scanned: scanned_units,
        timing_total_wall_s: total_wall_s,
        timing_scan_wall_s: scan_stage_wall_s,
        timing_scan_ml_select_wall_s,
        timing_scan_beam_wall_s,
        timing_scan_literal_score_wall_s,
        timing_scan_beam_calls: scan_beam_profile.calls,
        timing_clone_bits_s: scan_beam_profile.clone_bits_s,
        timing_sum_y_both1_s: scan_beam_profile.sum_y_both1_s,
        timing_parent_baseline_s: scan_beam_profile.parent_baseline_s,
        timing_pw_marginal_s: pw_marg,
        timing_pw_pack_s: pw_pack,
        timing_pw_kernel_s: pw_kern,
        timing_pw_combine_s: pw_comb,
        timing_dense_extract_s: (PACKED_EXTRACT_FLAT_NS.load(Ordering::Relaxed) as f64) * 1e-9,
        timing_dense_decode_s: (GARFIELD_DENSE_DOSAGE_DECODE_NS.load(Ordering::Relaxed) as f64)
            * 1e-9,
        timing_geneset_ld_prune_s: (GARFIELD_GENESET_LD_PRUNE_NS.load(Ordering::Relaxed) as f64)
            * 1e-9,
        ld_exact_pairs: GARFIELD_GENESET_LD_EXACT_PAIRS.load(Ordering::Relaxed),
        ld_rows_total: GARFIELD_GENESET_LD_ROWS_TOTAL.load(Ordering::Relaxed),
        ld_rows_kept: GARFIELD_GENESET_LD_ROWS_KEPT.load(Ordering::Relaxed),
        ld_units_eligible: ld_unit_stats.ld_units_eligible,
        ld_units_pruned: ld_unit_stats.ld_units_pruned,
        ld_rows_pruned: ld_unit_stats.ld_rows_pruned,
        ld_unit_rows_total_median: ld_unit_stats.ld_unit_rows_total_median,
        ld_unit_rows_total_p95: ld_unit_stats.ld_unit_rows_total_p95,
        ld_unit_rows_total_max: ld_unit_stats.ld_unit_rows_total_max,
        ld_unit_rows_kept_median: ld_unit_stats.ld_unit_rows_kept_median,
        ld_unit_rows_kept_p95: ld_unit_stats.ld_unit_rows_kept_p95,
        ld_unit_rows_kept_max: ld_unit_stats.ld_unit_rows_kept_max,
        ld_unit_rows_pruned_median: ld_unit_stats.ld_unit_rows_pruned_median,
        ld_unit_rows_pruned_p95: ld_unit_stats.ld_unit_rows_pruned_p95,
        ld_unit_rows_pruned_max: ld_unit_stats.ld_unit_rows_pruned_max,
        ld_unit_exact_pairs_median: ld_unit_stats.ld_unit_exact_pairs_median,
        ld_unit_exact_pairs_p95: ld_unit_stats.ld_unit_exact_pairs_p95,
        ld_unit_exact_pairs_max: ld_unit_stats.ld_unit_exact_pairs_max,
        timing_materialize_bits_s: (GARFIELD_MATERIALIZE_BITS_NS.load(Ordering::Relaxed) as f64)
            * 1e-9,
        timing_literal_score_share_of_total_pct,
        timing_literal_score_share_of_scan_pct,
        timing_literal_score_share_of_beam_pct,
        timing_beam_share_of_total_pct,
        timing_beam_share_of_scan_pct,
        skipped_units,
        skipped_messages,
        train_fit,
        test_fit,
    })
}

#[pyfunction(name = "garfield_logic_search_bed")]
#[pyo3(signature = (
    prefix,
    y,
    grm=None,
    x_cov=None,
    sample_ids=None,
    sample_indices=None,
    site_keep=None,
    unit_kind="window",
    groups=None,
    group_names=None,
    extension=100000,
    step=None,
    scan_bimranges=None,
    bin_mode="bin",
    ml_method="corr",
    ml_importance="imp",
    ml_top_k=64,
    ml_top_frac=0.0,
    permutation_repeats=20,
    permutation_scoring="auto",
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=1,
    min_samples_split=2,
    bootstrap=true,
    feature_subsample=0.0,
    fold=0,
    seed=42,
    max_pick=4,
    exhaustive_depth=1,
    beam_width=100,
    rank_score="interaction_gain",
    maf_threshold=0.02,
    logic_maf_threshold=0.02,
    max_missing_rate=0.05,
    het_threshold=1.0,
    snps_only=false,
    block_cols=65536,
    threads=0,
    low=-5.0,
    high=5.0,
    max_iter=50,
    tol=1e-3,
    add_intercept=true,
    exact_n_max=15000,
    require_lapack=false,
    out_prefix=None,
    simbench_path=None,
    top_rules_per_unit=1,
    max_output_rules=0,
    max_output_ratio=0.0,
    rule_permutation=true,
    prior_len=None,
    no_clean=false,
    progress_callback=None,
    progress_every=0
))]
pub fn garfield_logic_search_bed_py<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    grm: Option<PyReadonlyArray2<'py, f64>>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    unit_kind: &str,
    groups: Option<Vec<Vec<(String, i32, i32)>>>,
    group_names: Option<Vec<String>>,
    extension: usize,
    step: Option<usize>,
    scan_bimranges: Option<Vec<String>>,
    bin_mode: &str,
    ml_method: &str,
    ml_importance: &str,
    ml_top_k: usize,
    ml_top_frac: f64,
    permutation_repeats: usize,
    permutation_scoring: &str,
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    bootstrap: bool,
    feature_subsample: f64,
    fold: usize,
    seed: u64,
    max_pick: usize,
    exhaustive_depth: usize,
    beam_width: usize,
    rank_score: &str,
    maf_threshold: f32,
    logic_maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    block_cols: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    out_prefix: Option<String>,
    simbench_path: Option<String>,
    top_rules_per_unit: usize,
    max_output_rules: usize,
    max_output_ratio: f64,
    rule_permutation: bool,
    prior_len: Option<Vec<f64>>,
    no_clean: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Bound<'py, PyDict>> {
    arm_interrupt_trap();
    let y_vec = read_y_f64(&y);
    let grm_vec = if let Some(arr) = grm {
        let mat = arr.as_array();
        if mat.ndim() != 2 || mat.shape()[0] != mat.shape()[1] {
            return Err(PyValueError::new_err(
                "grm must be a square float64 matrix.",
            ));
        }
        Some(match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mat.iter().copied().collect(),
        })
    } else {
        None
    };
    let (x_cov_vec, q_cov) = if let Some(arr) = x_cov {
        let mat = arr.as_array();
        if mat.ndim() != 2 {
            return Err(PyValueError::new_err(
                "x_cov must be 2D (n_samples, q_cov).",
            ));
        }
        let q_cov = mat.shape()[1];
        let cov_vec = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mat.iter().copied().collect(),
        };
        (Some(cov_vec), q_cov)
    } else {
        (None, 0usize)
    };
    let site_keep_vec = if let Some(arr) = site_keep {
        Some(match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        })
    } else {
        None
    };
    let tree_cfg = ExtraTreesConfig {
        n_estimators: n_estimators.max(1),
        max_depth: max_depth.max(1),
        min_samples_leaf: min_samples_leaf.max(1),
        min_samples_split: min_samples_split.max(2),
        bootstrap,
        feature_subsample,
        seed,
        allow_parallel: true,
    };
    let result = py
        .detach(move || {
            garfield_logic_search_bed_owned(
                prefix,
                y_vec,
                grm_vec,
                x_cov_vec,
                q_cov,
                sample_ids,
                sample_indices,
                site_keep_vec,
                unit_kind.to_string(),
                groups,
                group_names,
                extension,
                step,
                scan_bimranges,
                bin_mode.to_string(),
                ml_method.to_string(),
                ml_importance.to_string(),
                ml_top_k,
                ml_top_frac,
                permutation_repeats,
                permutation_scoring.to_string(),
                tree_cfg,
                fold,
                seed,
                max_pick,
                exhaustive_depth,
                beam_width,
                rank_score.to_string(),
                maf_threshold,
                logic_maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                block_cols,
                threads,
                low,
                high,
                max_iter,
                tol,
                add_intercept,
                exact_n_max,
                require_lapack,
                out_prefix,
                simbench_path,
                top_rules_per_unit,
                max_output_rules,
                max_output_ratio,
                rule_permutation,
                prior_len,
                no_clean,
                progress_callback,
                progress_every,
            )
        })
        .map_err(map_err_string_to_py)?;

    let out = PyDict::new(py);
    out.set_item("pseudo_prefix", result.pseudo_prefix)?;
    out.set_item("rules_tsv", result.rules_tsv)?;
    out.set_item("posterior_tsv", py.None())?;
    out.set_item("posterior_json", result.posterior_json)?;
    if let Some(debug) = result.memory_debug.as_ref() {
        out.set_item("memory_debug", garfield_memory_debug_to_pydict(py, debug)?)?;
    } else {
        out.set_item("memory_debug", py.None())?;
    }
    out.set_item("rule_permutation_active", result.rule_permutation_active)?;
    out.set_item("null_chunk_bp", result.null_chunk_bp)?;
    out.set_item("null_chunk_min_snps", result.null_chunk_min_snps)?;
    out.set_item("null_chunk_target", result.null_chunk_target)?;
    out.set_item("null_chunk_valid_total", result.null_chunk_valid_total)?;
    out.set_item("null_chunk_selected", result.null_chunk_selected)?;
    out.set_item(
        "representative_units_target",
        result.representative_units_target,
    )?;
    out.set_item(
        "representative_units_used",
        result.representative_units_used,
    )?;
    out.set_item("permutation_null_repeats", result.permutation_null_repeats)?;
    out.set_item(
        "permutation_bootstrap_repeats",
        result.permutation_bootstrap_repeats,
    )?;
    out.set_item("n_rules", result.records.len())?;
    out.set_item("n_simbench", result.simbench_count)?;
    out.set_item("split_applied", result.split_applied)?;
    out.set_item("n_train", result.n_train)?;
    out.set_item("n_test", result.n_test)?;
    out.set_item("n_samples", result.n_samples)?;
    out.set_item("units_total", result.units_total)?;
    out.set_item("units_scanned", result.units_scanned)?;
    out.set_item("timing_total_wall_s", result.timing_total_wall_s)?;
    out.set_item("timing_scan_wall_s", result.timing_scan_wall_s)?;
    out.set_item(
        "timing_scan_ml_select_wall_s",
        result.timing_scan_ml_select_wall_s,
    )?;
    out.set_item("timing_scan_beam_wall_s", result.timing_scan_beam_wall_s)?;
    out.set_item(
        "timing_scan_literal_score_wall_s",
        result.timing_scan_literal_score_wall_s,
    )?;
    out.set_item("timing_scan_beam_calls", result.timing_scan_beam_calls)?;
    out.set_item("timing_clone_bits_s", result.timing_clone_bits_s)?;
    out.set_item("timing_sum_y_both1_s", result.timing_sum_y_both1_s)?;
    out.set_item("timing_parent_baseline_s", result.timing_parent_baseline_s)?;
    out.set_item("timing_pw_marginal_s", result.timing_pw_marginal_s)?;
    out.set_item("timing_pw_pack_s", result.timing_pw_pack_s)?;
    out.set_item("timing_pw_kernel_s", result.timing_pw_kernel_s)?;
    out.set_item("timing_pw_combine_s", result.timing_pw_combine_s)?;
    out.set_item("timing_dense_extract_s", result.timing_dense_extract_s)?;
    out.set_item("timing_dense_decode_s", result.timing_dense_decode_s)?;
    out.set_item(
        "timing_geneset_ld_prune_s",
        result.timing_geneset_ld_prune_s,
    )?;
    out.set_item("ld_exact_pairs", result.ld_exact_pairs)?;
    out.set_item("ld_rows_total", result.ld_rows_total)?;
    out.set_item("ld_rows_kept", result.ld_rows_kept)?;
    out.set_item("ld_units_eligible", result.ld_units_eligible)?;
    out.set_item("ld_units_pruned", result.ld_units_pruned)?;
    out.set_item("ld_rows_pruned", result.ld_rows_pruned)?;
    out.set_item(
        "ld_unit_rows_total_median",
        result.ld_unit_rows_total_median,
    )?;
    out.set_item("ld_unit_rows_total_p95", result.ld_unit_rows_total_p95)?;
    out.set_item("ld_unit_rows_total_max", result.ld_unit_rows_total_max)?;
    out.set_item("ld_unit_rows_kept_median", result.ld_unit_rows_kept_median)?;
    out.set_item("ld_unit_rows_kept_p95", result.ld_unit_rows_kept_p95)?;
    out.set_item("ld_unit_rows_kept_max", result.ld_unit_rows_kept_max)?;
    out.set_item(
        "ld_unit_rows_pruned_median",
        result.ld_unit_rows_pruned_median,
    )?;
    out.set_item("ld_unit_rows_pruned_p95", result.ld_unit_rows_pruned_p95)?;
    out.set_item("ld_unit_rows_pruned_max", result.ld_unit_rows_pruned_max)?;
    out.set_item(
        "ld_unit_exact_pairs_median",
        result.ld_unit_exact_pairs_median,
    )?;
    out.set_item("ld_unit_exact_pairs_p95", result.ld_unit_exact_pairs_p95)?;
    out.set_item("ld_unit_exact_pairs_max", result.ld_unit_exact_pairs_max)?;
    out.set_item(
        "timing_materialize_bits_s",
        result.timing_materialize_bits_s,
    )?;
    out.set_item(
        "timing_literal_score_share_of_total_pct",
        result.timing_literal_score_share_of_total_pct,
    )?;
    out.set_item(
        "timing_literal_score_share_of_scan_pct",
        result.timing_literal_score_share_of_scan_pct,
    )?;
    out.set_item(
        "timing_literal_score_share_of_beam_pct",
        result.timing_literal_score_share_of_beam_pct,
    )?;
    out.set_item(
        "timing_beam_share_of_total_pct",
        result.timing_beam_share_of_total_pct,
    )?;
    out.set_item(
        "timing_beam_share_of_scan_pct",
        result.timing_beam_share_of_scan_pct,
    )?;
    let skipped_units_py = PyList::empty(py);
    for skipped in result.skipped_units.iter() {
        skipped_units_py.append(garfield_skipped_unit_to_pydict(py, skipped)?)?;
    }
    out.set_item("n_skipped_units", result.skipped_units.len())?;
    out.set_item("skipped_units", skipped_units_py)?;
    out.set_item("skipped_messages", result.skipped_messages.clone())?;
    out.set_item("train_pve", result.train_fit.pve)?;
    out.set_item("test_pve", result.test_fit.pve)?;
    out.set_item("train_sigma_g2", result.train_fit.sigma_g2)?;
    out.set_item("train_sigma_e2", result.train_fit.sigma_e2)?;
    out.set_item("test_sigma_g2", result.test_fit.sigma_g2)?;
    out.set_item("test_sigma_e2", result.test_fit.sigma_e2)?;
    out.set_item("train_eigh_backend", result.train_fit.eigh_backend)?;
    out.set_item("test_eigh_backend", result.test_fit.eigh_backend)?;
    out.set_item(
        "snp_names",
        result
            .records
            .iter()
            .map(|r| r.snp_name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "expressions",
        result
            .records
            .iter()
            .map(|r| r.expr.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "unit_names",
        result
            .records
            .iter()
            .map(|r| r.unit_name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "unit_kinds",
        result
            .records
            .iter()
            .map(|r| r.unit_kind.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "scores",
        result.records.iter().map(|r| r.score).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "train_scores",
        result.records.iter().map(|r| r.score).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "test_scores",
        result.records.iter().map(|r| r.score).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "positions",
        result.records.iter().map(|r| r.pos).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "selected_row_indices",
        result
            .records
            .iter()
            .map(|r| r.selected_row_indices.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "ml_ranks",
        result
            .records
            .iter()
            .map(|r| r.ml_rank.clone())
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

#[pyfunction(name = "load_bin01_packed")]
pub fn load_bin01_packed_py<'py>(
    py: Python<'py>,
    path: String,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<u64>>, usize)> {
    let (packed, group_ids, n_samples, n_rows, row_bytes) = py
        .detach(move || load_bin01_packed_owned(&path, false))
        .map_err(PyRuntimeError::new_err)?;
    let packed_mat = Array2::from_shape_vec((n_rows, row_bytes), packed)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let gid_arr = Array1::from_vec(group_ids);
    #[allow(deprecated)]
    let packed_arr = PyArray2::from_owned_array(py, packed_mat).into_bound();
    #[allow(deprecated)]
    let gid_arr = PyArray1::from_owned_array(py, gid_arr).into_bound();
    Ok((packed_arr, gid_arr, n_samples))
}

#[pyfunction(name = "load_mbin_packed")]
pub fn load_mbin_packed_py<'py>(
    py: Python<'py>,
    path: String,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<u64>>, usize)> {
    let (packed, group_ids, n_samples, n_rows, row_bytes) = py
        .detach(move || load_bin01_packed_owned(&path, true))
        .map_err(PyRuntimeError::new_err)?;
    let packed_mat = Array2::from_shape_vec((n_rows, row_bytes), packed)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let gid_arr = Array1::from_vec(group_ids);
    #[allow(deprecated)]
    let packed_arr = PyArray2::from_owned_array(py, packed_mat).into_bound();
    #[allow(deprecated)]
    let gid_arr = PyArray1::from_owned_array(py, gid_arr).into_bound();
    Ok((packed_arr, gid_arr, n_samples))
}

#[pyfunction(name = "garfield_prepare_input_bin")]
#[pyo3(signature = (
    input_path,
    out_bin_path,
    input_kind="auto",
    mode="bin",
    threads=0,
    maf=0.0,
    geno=1.0,
    impute=false,
    het=1.0,
    sample_ids=None,
    sample_indices=None
))]
pub fn garfield_prepare_input_bin_py(
    input_path: String,
    out_bin_path: String,
    input_kind: &str,
    mode: &str,
    threads: usize,
    maf: f64,
    geno: f64,
    impute: bool,
    het: f64,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
) -> PyResult<(usize, usize, usize, usize)> {
    if !(0.0..=0.5).contains(&maf) {
        return Err(PyValueError::new_err("maf must be within [0, 0.5]"));
    }
    if !(0.0..=1.0).contains(&geno) {
        return Err(PyValueError::new_err("geno must be within [0, 1]"));
    }
    if !(0.0..=1.0).contains(&het) {
        return Err(PyValueError::new_err("het must be within [0, 1]"));
    }

    let input_kind = parse_input_kind(input_kind).map_err(PyValueError::new_err)?;
    let mode = parse_bin_mode(mode).map_err(PyValueError::new_err)?;
    let mut reader = make_input_reader(&input_path, input_kind).map_err(PyRuntimeError::new_err)?;

    let (selected_indices, selected_ids) =
        build_sample_selection(reader.sample_ids(), sample_ids, sample_indices)
            .map_err(PyRuntimeError::new_err)?;
    if selected_indices.is_empty() {
        return Err(PyValueError::new_err("selected sample set is empty"));
    }
    write_sample_id_sidecar(&out_bin_path, &selected_ids).map_err(PyRuntimeError::new_err)?;

    let n_samples = selected_indices.len();
    let identity_selection = selected_indices
        .iter()
        .enumerate()
        .all(|(i, &idx)| i == idx);
    let row_bytes = row_bytes_for_samples(n_samples);
    let mut writer = Bin01Writer::new(&out_bin_path, n_samples, Bin01SiteMode::TextTsv)
        .map_err(PyRuntimeError::new_err)?;

    let available_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1usize);
    let n_threads = if threads == 0 {
        available_threads
    } else {
        threads.min(available_threads).max(1)
    };
    let pool = if n_threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("build thread pool: {e}")))?,
        )
    } else {
        None
    };
    let pool_ref = pool.as_ref();

    let maf_f32 = maf as f32;
    let geno_f32 = geno as f32;
    let het_f32 = het as f32;

    let mut n_sites_seen = 0usize;
    let mut n_sites_written = 0usize;
    let mut n_rows_written = 0usize;

    match &mut reader {
        GarfieldInputReader::Bed(bed) => {
            if n_threads > 1 {
                let n_sites = bed.sites.len();
                let target_bytes = 16usize * 1024 * 1024;
                let mut chunk_size = target_bytes / n_samples.max(1);
                if chunk_size == 0 {
                    chunk_size = 1;
                }
                if chunk_size > 4096 {
                    chunk_size = 4096;
                }
                for start in (0..n_sites).step_by(chunk_size) {
                    let end = (start + chunk_size).min(n_sites);
                    let rows_res: Vec<Result<Option<Vec<EncodedRow>>, String>> = pool_ref
                        .expect("thread pool exists when n_threads>1")
                        .install(|| {
                            (start..end)
                                .into_par_iter()
                                .map(|idx| {
                                    let (row, site) = if identity_selection {
                                        bed.get_snp_row_raw(idx).ok_or_else(|| {
                                            format!("BED decode failed at row {}", idx)
                                        })?
                                    } else {
                                        bed.get_snp_row_selected_raw(idx, &selected_indices)
                                            .ok_or_else(|| {
                                                format!("BED decode with sample selection failed at row {}", idx)
                                            })?
                                    };
                                    Ok(encode_row_to_bits(
                                        row,
                                        site,
                                        mode,
                                        row_bytes,
                                        maf_f32,
                                        geno_f32,
                                        impute,
                                        het_f32,
                                    ))
                                })
                                .collect()
                        });

                    for item in rows_res.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        let maybe_rows = item.map_err(PyRuntimeError::new_err)?;
                        if let Some(rows) = maybe_rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows.len());
                            write_encoded_rows_bin(&mut writer, &rows)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            } else {
                while let Some((row, site)) = if identity_selection {
                    bed.next_snp_raw()
                } else {
                    bed.next_snp_selected_raw(&selected_indices)
                } {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows) = encode_row_to_bits(
                        row, site, mode, row_bytes, maf_f32, geno_f32, impute, het_f32,
                    ) {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows.len());
                        write_encoded_rows_bin(&mut writer, &rows)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Vcf(vcf) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            loop {
                let Some((row_raw, site)) = vcf.next_snp_raw() else {
                    break;
                };
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            write_encoded_rows_bin(&mut writer, &rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        write_encoded_rows_bin(&mut writer, &rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Hmp(hmp) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            loop {
                let Some((row_raw, site)) = hmp.next_snp_raw() else {
                    break;
                };
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            write_encoded_rows_bin(&mut writer, &rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        write_encoded_rows_bin(&mut writer, &rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Txt(txt) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            while let Some((row_raw, site)) = txt.next_snp() {
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            write_encoded_rows_bin(&mut writer, &rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        write_encoded_rows_bin(&mut writer, &rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
    }

    let header_rows = writer.finish().map_err(PyRuntimeError::new_err)?;
    if header_rows != n_rows_written {
        return Err(PyRuntimeError::new_err(format!(
            "internal row count mismatch: header_rows={}, tracked_rows={}",
            header_rows, n_rows_written
        )));
    }

    Ok((n_sites_seen, n_sites_written, n_rows_written, n_samples))
}

#[pyfunction(name = "garfield_subset_bin_samples")]
pub fn garfield_subset_bin_samples_py(
    bin_path: String,
    out_bin_path: String,
    sample_indices: Vec<usize>,
) -> PyResult<()> {
    let ctx = "garfield_subset_bin_samples";
    if sample_indices.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{ctx}: sample_indices is empty"
        )));
    }
    let bytes = load_file_owned(Path::new(&bin_path))
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: {e}")))?;
    let (n_rows, n_samples, row_bytes, data_offset) =
        parse_bin01_header(&bytes, ctx).map_err(PyRuntimeError::new_err)?;
    for &si in sample_indices.iter() {
        if si >= n_samples {
            return Err(PyValueError::new_err(format!(
                "{ctx}: sample index out of range {} >= {}",
                si, n_samples
            )));
        }
    }

    let n_out = sample_indices.len();
    let row_bytes_out = n_out.div_ceil(8);
    let payload_len = n_rows
        .checked_mul(row_bytes_out)
        .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: payload overflow")))?;
    let header = bin01_header_bytes(n_rows as u64, n_out);
    let mut out = Vec::<u8>::with_capacity(header.len() + payload_len);
    out.extend_from_slice(&header);

    for r in 0..n_rows {
        let src_start =
            data_offset
                .checked_add(r.checked_mul(row_bytes).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{ctx}: row offset overflow"))
                })?)
                .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: row offset overflow")))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: row end overflow")))?;
        let src = &bytes[src_start..src_end];

        let mut row = vec![0u8; row_bytes_out];
        for (dst_i, &src_i) in sample_indices.iter().enumerate() {
            let src_b = src[src_i >> 3];
            let bit = (src_b >> (src_i & 7)) & 1u8;
            if bit != 0 {
                row[dst_i >> 3] |= 1u8 << (dst_i & 7);
            }
        }
        out.extend_from_slice(&row);
    }

    let mut fw = File::create(&out_bin_path)
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: create {out_bin_path}: {e}")))?;
    fw.write_all(&out)
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: write {out_bin_path}: {e}")))?;
    fw.flush()
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: flush {out_bin_path}: {e}")))?;

    copy_site_sidecar(&bin_path, &out_bin_path).map_err(PyRuntimeError::new_err)?;
    let wrote_subset_ids = write_subset_id_sidecar(&bin_path, &out_bin_path, &sample_indices)
        .map_err(PyRuntimeError::new_err)?;
    if !wrote_subset_ids {
        copy_id_sidecar(&bin_path, &out_bin_path).map_err(PyRuntimeError::new_err)?;
    }
    Ok(())
}

#[pyfunction(name = "garfield_scan_groups_bin")]
#[pyo3(signature = (
    bin_path,
    y_train,
    groups,
    response="continuous",
    max_pick=4,
    beam_width=100,
    enforce_feature_exclusion=true,
    progress_callback=None,
    progress_every=0
))]
pub fn garfield_scan_groups_bin_py(
    bin_path: String,
    y_train: PyReadonlyArray1<'_, f64>,
    groups: Vec<Vec<(String, i32, i32)>>,
    response: &str,
    max_pick: usize,
    beam_width: usize,
    enforce_feature_exclusion: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Vec<(usize, usize, f64, Vec<usize>)>> {
    if groups.is_empty() {
        return Ok(Vec::new());
    }
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    let y_vec = read_y_f64(&y_train);
    let y_bin_owned = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };
    let y_bin = y_bin_owned.as_deref();

    let (bits_flat_all, row_words, n_rows_all, n_samples) =
        load_bin01_as_u64_words(&bin_path, "garfield_scan_groups_bin")
            .map_err(PyRuntimeError::new_err)?;
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_scan_groups_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let sites = TxtSnpIter::new(&bin_path, None)
        .map_err(PyRuntimeError::new_err)?
        .sites;
    let chrom_idx = build_chrom_index(&sites, n_rows_all);
    let feature_group_ids_all = if enforce_feature_exclusion {
        Some(build_feature_group_ids(&sites, n_rows_all))
    } else {
        None
    };
    let total_groups = groups.len();
    let notify_step = if progress_every == 0 {
        (total_groups / 200).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    garfield_progress_notify(progress_callback.as_ref(), 0, total_groups)?;

    let mut out: Vec<(usize, usize, f64, Vec<usize>)> = Vec::with_capacity(groups.len());
    for (gi, group) in groups.iter().enumerate() {
        let mut idx_all: Vec<usize> = Vec::new();
        for (chrom, start, end) in group.iter() {
            let mut iv = interval_indices(&chrom_idx, chrom, *start, *end);
            idx_all.append(&mut iv);
        }
        if idx_all.is_empty() {
            out.push((gi, 0usize, f64::NEG_INFINITY, Vec::new()));
            let done = gi + 1;
            if done - last_notified >= notify_step || done == total_groups {
                garfield_progress_notify(progress_callback.as_ref(), done, total_groups)?;
                last_notified = done;
            }
            continue;
        }
        idx_all.sort_unstable();
        idx_all.dedup();

        let n_rows = idx_all.len();
        let mut bits_flat = vec![0u64; n_rows * row_words];
        for (ri, &src_idx) in idx_all.iter().enumerate() {
            let src_start = src_idx * row_words;
            let dst_start = ri * row_words;
            bits_flat[dst_start..dst_start + row_words]
                .copy_from_slice(&bits_flat_all[src_start..src_start + row_words]);
        }

        let local_group_ids = if let Some(global_ids) = feature_group_ids_all.as_ref() {
            Some(
                idx_all
                    .iter()
                    .map(|&gi| global_ids[gi])
                    .collect::<Vec<usize>>(),
            )
        } else {
            None
        };
        let res = run_beam_with_feature_exclusion(
            &bits_flat,
            row_words,
            n_rows,
            n_samples,
            resp,
            y_vec.as_slice(),
            y_bin,
            max_pick,
            beam_width,
            local_group_ids.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;
        let selected_global = res
            .selected_indices
            .iter()
            .map(|&local_idx| idx_all[local_idx])
            .collect::<Vec<_>>();
        out.push((gi, idx_all.len(), res.score, selected_global));
        let done = gi + 1;
        if done - last_notified >= notify_step || done == total_groups {
            garfield_progress_notify(progress_callback.as_ref(), done, total_groups)?;
            last_notified = done;
        }
    }
    Ok(out)
}

#[pyfunction(name = "garfield_scan_windows_bin")]
#[pyo3(signature = (
    bin_path,
    y_train,
    response="continuous",
    max_pick=4,
    beam_width=100,
    extension=50000,
    step=None,
    enforce_feature_exclusion=true,
    progress_callback=None,
    progress_every=0
))]
pub fn garfield_scan_windows_bin_py(
    bin_path: String,
    y_train: PyReadonlyArray1<'_, f64>,
    response: &str,
    max_pick: usize,
    beam_width: usize,
    extension: usize,
    step: Option<usize>,
    enforce_feature_exclusion: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)>> {
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    if extension == 0 {
        return Err(PyValueError::new_err(
            "garfield_scan_windows_bin: extension must be > 0",
        ));
    }
    let step_v = step.unwrap_or((extension / 2).max(1));
    if step_v == 0 {
        return Err(PyValueError::new_err(
            "garfield_scan_windows_bin: step must be > 0",
        ));
    }

    let y_vec = read_y_f64(&y_train);
    let y_bin_owned = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };
    let y_bin = y_bin_owned.as_deref();

    let (bits_flat_all, row_words, n_rows_all, n_samples) =
        load_bin01_as_u64_words(&bin_path, "garfield_scan_windows_bin")
            .map_err(PyRuntimeError::new_err)?;
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_scan_windows_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let sites = TxtSnpIter::new(&bin_path, None)
        .map_err(PyRuntimeError::new_err)?
        .sites;
    let windows = build_windows_from_sites(&sites, n_rows_all, extension, step_v, normalize_chrom);
    if windows.is_empty() {
        return Ok(Vec::new());
    }
    let total_windows = windows.len();
    let notify_step = if progress_every == 0 {
        (total_windows / 200).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    garfield_progress_notify(progress_callback.as_ref(), 0, total_windows)?;
    let feature_group_ids_all = if enforce_feature_exclusion {
        Some(build_feature_group_ids(&sites, n_rows_all))
    } else {
        None
    };

    let mut out: Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)> =
        Vec::with_capacity(windows.len());
    for (wi0, win) in windows.iter().enumerate() {
        if win.indices.is_empty() {
            let done = wi0 + 1;
            if done - last_notified >= notify_step || done == total_windows {
                garfield_progress_notify(progress_callback.as_ref(), done, total_windows)?;
                last_notified = done;
            }
            continue;
        }
        let n_rows = win.indices.len();
        let sub = if n_rows == n_rows_all {
            bits_flat_all.clone()
        } else {
            gather_rows_by_indices(
                &bits_flat_all,
                row_words,
                &win.indices,
                "garfield_scan_windows_bin::window_bits",
            )
            .map_err(PyRuntimeError::new_err)?
        };

        let local_group_ids = if let Some(global_ids) = feature_group_ids_all.as_ref() {
            Some(
                win.indices
                    .iter()
                    .map(|&gi| global_ids[gi])
                    .collect::<Vec<usize>>(),
            )
        } else {
            None
        };

        let res = run_beam_with_feature_exclusion(
            &sub,
            row_words,
            n_rows,
            n_samples,
            resp,
            y_vec.as_slice(),
            y_bin,
            max_pick,
            beam_width,
            local_group_ids.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;

        let selected_global = res
            .selected_indices
            .iter()
            .map(|&local_idx| win.indices[local_idx])
            .collect::<Vec<_>>();
        out.push((
            wi0 + 1,
            win.chrom.clone(),
            win.bp_start,
            win.bp_end,
            n_rows,
            res.score,
            selected_global,
        ));
        let done = wi0 + 1;
        if done - last_notified >= notify_step || done == total_windows {
            garfield_progress_notify(progress_callback.as_ref(), done, total_windows)?;
            last_notified = done;
        }
    }
    Ok(out)
}

#[pyfunction(name = "garfield_eval_rule_bin")]
#[pyo3(signature = (bin_path, y, snp_indices, response="continuous"))]
pub fn garfield_eval_rule_bin_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, f64>,
    snp_indices: Vec<usize>,
    response: &str,
) -> PyResult<(f64, usize, Vec<u8>)> {
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    if snp_indices.is_empty() {
        return Ok((f64::NEG_INFINITY, 0usize, Vec::new()));
    }
    let y_vec = read_y_f64(&y);
    let y_bin = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };

    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_selected_rows_as_u64_words(&bin_path, &snp_indices, "garfield_eval_rule_bin")
            .map_err(PyRuntimeError::new_err)?;
    if n_rows == 0 {
        return Ok((f64::NEG_INFINITY, 0usize, Vec::new()));
    }
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_eval_rule_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let needed_words = words_for_samples(n_samples);
    let mut combined = bits_flat[0..needed_words].to_vec();
    for r in 1..n_rows {
        let st = r * row_words;
        let row = &bits_flat[st..st + needed_words];
        for (a, &b) in combined.iter_mut().zip(row.iter()) {
            *a &= b;
        }
    }
    apply_tail_mask(&mut combined, tail_mask(n_samples));

    let score = match resp {
        GarfieldResponse::Binary => {
            let yb = y_bin.as_ref().expect("binary y prepared");
            score_binary_mcc_packed(yb.as_slice(), &combined, n_samples)
        }
        GarfieldResponse::Continuous => {
            score_cont_corr_packed(y_vec.as_slice(), &combined, n_samples).abs()
        }
    };
    let support = combined
        .iter()
        .map(|w| w.count_ones() as usize)
        .sum::<usize>();
    let mut xcombine = vec![0u8; n_samples];
    for i in 0..n_samples {
        let bit = (combined[i >> 6] >> (i & 63)) & 1u64;
        xcombine[i] = if bit != 0 { 1u8 } else { 0u8 };
    }
    Ok((score, support, xcombine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(prefix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_garfield_{prefix}_{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_row_bits(n_samples: usize, ones: &[usize]) -> Vec<u8> {
        let mut bits = vec![0u8; row_bytes_for_samples(n_samples)];
        for &idx in ones.iter() {
            bits[idx >> 3] |= 1u8 << (idx & 7);
        }
        bits
    }

    fn build_test_bits(n_rows: usize, n_samples: usize) -> (Vec<u64>, usize, Vec<usize>) {
        let row_words = words_for_samples(n_samples);
        let mut bits_flat = vec![0u64; n_rows * row_words];
        for row in 0..n_rows {
            for word in 0..row_words {
                let a = (row as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let b = (word as u64 + 3).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let rot = ((row * 11 + word * 7) % 63 + 1) as u32;
                let dense = (!0u64).rotate_right(((row * 5 + word * 13) % 64) as u32);
                bits_flat[row * row_words + word] = (a ^ b).rotate_left(rot) | dense;
            }
        }
        let group_ids = (0..n_rows).map(|i| i % 96).collect::<Vec<_>>();
        (bits_flat, row_words, group_ids)
    }

    fn pack_test_binary_rows(rows: &[Vec<u8>]) -> (Vec<u64>, usize) {
        let n_samples = rows.first().map(|row| row.len()).unwrap_or(0);
        let row_words = words_for_samples(n_samples);
        let mut bits_flat = vec![0u64; rows.len().saturating_mul(row_words)];
        for (row_idx, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), n_samples);
            for (sample_idx, &v) in row.iter().enumerate() {
                if v != 0 {
                    bits_flat[row_idx * row_words + (sample_idx >> 6)] |= 1u64 << (sample_idx & 63);
                }
            }
        }
        (bits_flat, row_words)
    }

    #[test]
    fn test_group_exclusion_parallel_matches_serial() {
        let n_rows = 1024usize;
        let n_samples = 256usize;
        let max_pick = 3usize;
        let beam_width = 8usize;
        let (bits_flat, row_words, group_ids) = build_test_bits(n_rows, n_samples);
        let score_fn =
            |combined: &[u64]| combined.iter().map(|w| w.count_ones() as f64).sum::<f64>();

        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build serial pool")
            .install(|| {
                beam_search_and_with_group_exclusion(
                    &bits_flat, row_words, n_rows, n_samples, max_pick, beam_width, &group_ids,
                    score_fn,
                )
                .expect("serial constrained beam")
            });

        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build parallel pool")
            .install(|| {
                beam_search_and_with_group_exclusion(
                    &bits_flat, row_words, n_rows, n_samples, max_pick, beam_width, &group_ids,
                    score_fn,
                )
                .expect("parallel constrained beam")
            });

        assert_eq!(serial.selected_indices, parallel.selected_indices);
        assert_eq!(serial.combined_bits, parallel.combined_bits);
        assert_eq!(serial.score, parallel.score);
    }

    #[test]
    fn test_load_bin01_packed_roundtrip() {
        let dir = make_temp_dir("bin01_load");
        let bin_path = dir.join("toy.bin");
        let bin_str = bin_path.to_str().unwrap();
        let n_samples = 10usize;
        let sample_ids = (0..n_samples).map(|i| format!("s{i}")).collect::<Vec<_>>();
        write_sample_id_sidecar(bin_str, &sample_ids).unwrap();
        let mut writer = Bin01Writer::new(bin_str, n_samples, Bin01SiteMode::TextTsv).unwrap();
        write_encoded_rows_bin(
            &mut writer,
            &[
                EncodedRow {
                    site: SiteInfo {
                        chrom: "1".to_string(),
                        pos: 10,
                        snp: "bin_row_10".to_string(),
                        ref_allele: "A".to_string(),
                        alt_allele: "T|BIN".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[1, 3, 8]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "1".to_string(),
                        pos: 20,
                        snp: "bin_row_20".to_string(),
                        ref_allele: "A".to_string(),
                        alt_allele: "C|BIN".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[0, 4, 9]),
                },
            ],
        )
        .unwrap();
        writer.finish().unwrap();

        let (packed, group_ids, got_n_samples, n_rows, row_bytes) =
            load_bin01_packed_owned(bin_str, false).unwrap();
        assert_eq!(got_n_samples, n_samples);
        assert_eq!(n_rows, 2);
        assert_eq!(row_bytes, row_bytes_for_samples(n_samples));
        assert_eq!(group_ids, vec![0u64, 1u64]);
        assert_eq!(
            packed,
            [
                make_row_bits(n_samples, &[1, 3, 8]),
                make_row_bits(n_samples, &[0, 4, 9]),
            ]
            .concat()
        );
    }

    #[test]
    fn test_load_mbin_packed_groups_triplets() {
        let dir = make_temp_dir("mbin_load");
        let bin_path = dir.join("toy.mbin.bin");
        let bin_str = bin_path.to_str().unwrap();
        let n_samples = 9usize;
        let sample_ids = (0..n_samples).map(|i| format!("s{i}")).collect::<Vec<_>>();
        write_sample_id_sidecar(bin_str, &sample_ids).unwrap();
        let mut writer = Bin01Writer::new(bin_str, n_samples, Bin01SiteMode::TextTsv).unwrap();
        write_encoded_rows_bin(
            &mut writer,
            &[
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 100,
                        snp: "mbin_100_dom".to_string(),
                        ref_allele: "A".to_string(),
                        alt_allele: "G|DOM".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[0, 1, 4, 8]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 100,
                        snp: "mbin_100_rec".to_string(),
                        ref_allele: "A".to_string(),
                        alt_allele: "G|REC".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[1, 8]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 100,
                        snp: "mbin_100_het".to_string(),
                        ref_allele: "A".to_string(),
                        alt_allele: "G|HET".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[0, 4]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 150,
                        snp: "mbin_150_dom".to_string(),
                        ref_allele: "C".to_string(),
                        alt_allele: "T|DOM".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[2, 3, 5]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 150,
                        snp: "mbin_150_rec".to_string(),
                        ref_allele: "C".to_string(),
                        alt_allele: "T|REC".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[5]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 150,
                        snp: "mbin_150_het".to_string(),
                        ref_allele: "C".to_string(),
                        alt_allele: "T|HET".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[2, 3]),
                },
            ],
        )
        .unwrap();
        writer.finish().unwrap();

        let (packed, group_ids, got_n_samples, n_rows, row_bytes) =
            load_bin01_packed_owned(bin_str, true).unwrap();
        assert_eq!(got_n_samples, n_samples);
        assert_eq!(n_rows, 6);
        assert_eq!(row_bytes, row_bytes_for_samples(n_samples));
        assert_eq!(group_ids, vec![0u64, 0, 0, 1, 1, 1]);
        assert_eq!(packed.len(), n_rows * row_bytes);
    }

    #[test]
    fn test_apply_logic_rule_output_limit_ratio() {
        let mut records = (0..10usize)
            .map(|i| GarfieldLogicRuleRecord {
                unit_name: format!("u{i}"),
                unit_kind: "window".to_string(),
                unit_index: i + 1,
                region_size: 8,
                ml_feature_count: 4,
                ml_rank: ".".to_string(),
                selected_row_indices: vec![i],
                display_ops: Vec::new(),
                display_negated: vec![false],
                snp_name: format!("snp{i}"),
                expr: format!("expr{i}"),
                chrom_field: "1".to_string(),
                bim_snp_name: format!("snp{i}"),
                bim_allele0: "A".to_string(),
                bim_allele1: "T".to_string(),
                pos: i as i32,
                score: 20.0 - i as f64,
                delta_score: format!("{}->{}", 20.0 - i as f64, 20.0 - i as f64),
                support_bits: None,
            })
            .collect::<Vec<_>>();
        apply_logic_rule_output_limit(&mut records, 0, 0.3).unwrap();
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn test_extend_keep_with_score_ties_keeps_full_tie_block() {
        let scores = vec![10.0_f64, 9.0, 9.0, 8.0];
        let keep = extend_keep_with_score_ties(scores.as_slice(), 2, |v| *v);
        assert_eq!(keep, 3);
    }

    #[test]
    fn test_apply_logic_rule_output_limit_count_keeps_ties() {
        let mut records = vec![
            GarfieldLogicRuleRecord {
                unit_name: "u1".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 1,
                region_size: 8,
                ml_feature_count: 4,
                ml_rank: ".".to_string(),
                selected_row_indices: vec![1],
                display_ops: Vec::new(),
                display_negated: vec![false],
                snp_name: "snp1".to_string(),
                expr: "expr1".to_string(),
                chrom_field: "1".to_string(),
                bim_snp_name: "snp1".to_string(),
                bim_allele0: "A".to_string(),
                bim_allele1: "T".to_string(),
                pos: 1,
                score: 10.0,
                delta_score: "10->10".to_string(),
                support_bits: None,
            },
            GarfieldLogicRuleRecord {
                unit_name: "u2".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 2,
                region_size: 8,
                ml_feature_count: 4,
                ml_rank: ".".to_string(),
                selected_row_indices: vec![2],
                display_ops: Vec::new(),
                display_negated: vec![false],
                snp_name: "snp2".to_string(),
                expr: "expr2".to_string(),
                chrom_field: "1".to_string(),
                bim_snp_name: "snp2".to_string(),
                bim_allele0: "A".to_string(),
                bim_allele1: "T".to_string(),
                pos: 2,
                score: 9.0,
                delta_score: "9->9".to_string(),
                support_bits: None,
            },
            GarfieldLogicRuleRecord {
                unit_name: "u3".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 3,
                region_size: 8,
                ml_feature_count: 4,
                ml_rank: ".".to_string(),
                selected_row_indices: vec![3],
                display_ops: Vec::new(),
                display_negated: vec![false],
                snp_name: "snp3".to_string(),
                expr: "expr3".to_string(),
                chrom_field: "1".to_string(),
                bim_snp_name: "snp3".to_string(),
                bim_allele0: "A".to_string(),
                bim_allele1: "T".to_string(),
                pos: 3,
                score: 9.0,
                delta_score: "9->9".to_string(),
                support_bits: None,
            },
            GarfieldLogicRuleRecord {
                unit_name: "u4".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 4,
                region_size: 8,
                ml_feature_count: 4,
                ml_rank: ".".to_string(),
                selected_row_indices: vec![4],
                display_ops: Vec::new(),
                display_negated: vec![false],
                snp_name: "snp4".to_string(),
                expr: "expr4".to_string(),
                chrom_field: "1".to_string(),
                bim_snp_name: "snp4".to_string(),
                bim_allele0: "A".to_string(),
                bim_allele1: "T".to_string(),
                pos: 4,
                score: 8.0,
                delta_score: "8->8".to_string(),
                support_bits: None,
            },
        ];
        apply_logic_rule_output_limit(&mut records, 2, 0.0).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[1].score, records[2].score);
    }

    #[test]
    fn test_preferred_display_polarity_keeps_original_for_all_negated_rule() {
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: true,
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
        let full_bits = vec![0b0011u64];
        let polarity = preferred_display_polarity(&rule, full_bits.as_slice(), 4);
        assert_eq!(polarity, GarfieldRuleDisplayPolarity::Original);
    }

    #[test]
    fn test_rule_display_polarity_keeps_negated_and_form() {
        let sites = vec![test_site("1", 100), test_site("1", 200)];
        let rule = BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: true,
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
        let polarity = GarfieldRuleDisplayPolarity::Original;
        let expr = rule_expr_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
        let snp_name = rule_snp_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
        let bim_name = rule_bim_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
        let ml_rank = rule_ml_rank_name_with_polarity(&rule, polarity);
        let (a0, a1) = rule_bim_alleles_with_polarity(&rule, sites.as_slice(), polarity).unwrap();

        assert_eq!(expr, "NOT BIN(1_100) AND NOT BIN(1_200)");
        assert_eq!(snp_name, "!1_100&!1_200");
        assert_eq!(bim_name, "!1_100&!1_200");
        assert_eq!(ml_rank, "!1&!2");
        assert_eq!(a0, "G,G");
        assert_eq!(a1, "A,A");
    }

    #[test]
    fn test_rule_display_polarity_preserves_original_or_rule() {
        let sites = vec![test_site("1", 100), test_site("1", 200)];
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
        let polarity = GarfieldRuleDisplayPolarity::Original;
        let expr = rule_expr_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
        let snp_name = rule_snp_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
        let bim_name = rule_bim_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
        let ml_rank = rule_ml_rank_name_with_polarity(&rule, polarity);

        assert_eq!(expr, "BIN(1_100) OR BIN(1_200)");
        assert_eq!(snp_name, "1_100|1_200");
        assert_eq!(bim_name, "1_100|1_200");
        assert_eq!(ml_rank, "1|2");
    }

    #[test]
    fn test_logic_sites_prefer_inherited_unique_snp_names() {
        let logic_sites = build_logic_sites_from_metadata(
            &vec![
                test_named_site("1", 100, "rsA"),
                test_named_site("1", 200, "rsB"),
            ],
            GarfieldBinMode::Bin,
        );
        let rule = BeamRule {
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
        let polarity = GarfieldRuleDisplayPolarity::Original;
        let expr = rule_expr_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();
        let snp_name =
            rule_snp_name_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();
        let bim_name =
            rule_bim_name_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();

        assert_eq!(expr, "BIN(rsA) AND BIN(rsB)");
        assert_eq!(snp_name, "rsA&rsB");
        assert_eq!(bim_name, "rsA&rsB");
    }

    #[test]
    fn test_logic_sites_fall_back_to_coordinates_when_snp_names_duplicate() {
        let logic_sites = build_logic_sites_from_metadata(
            &vec![
                test_named_site("1", 100, "dup"),
                test_named_site("1", 200, "dup"),
            ],
            GarfieldBinMode::Bin,
        );
        let rule = BeamRule {
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
        let polarity = GarfieldRuleDisplayPolarity::Original;
        let expr = rule_expr_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();
        let snp_name =
            rule_snp_name_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();

        assert_eq!(expr, "BIN(1_100) AND BIN(1_200)");
        assert_eq!(snp_name, "1_100&1_200");
    }

    #[test]
    fn test_dedup_logic_rule_records_prefers_and_form_same_support() {
        let full_bits = vec![0b1010u64];
        let records = vec![
            GarfieldLogicRuleRecord {
                unit_name: "u_not".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 1,
                region_size: 2,
                ml_feature_count: 2,
                ml_rank: "!1&!2".to_string(),
                selected_row_indices: vec![0, 1],
                display_ops: vec![BeamBinaryOp::And],
                display_negated: vec![true, true],
                snp_name: "!1_100&!1_200".to_string(),
                expr: "NOT BIN(1_100) AND NOT BIN(1_200)".to_string(),
                chrom_field: "1".to_string(),
                bim_snp_name: "!1_100&!1_200".to_string(),
                bim_allele0: "G,G".to_string(),
                bim_allele1: "A,A".to_string(),
                pos: 100,
                score: 5.0,
                delta_score: "0.1&0.2->5.0".to_string(),
                support_bits: Some(GarfieldRuleSupportBits::Binary(
                    full_bits.clone().into_boxed_slice(),
                )),
            },
            GarfieldLogicRuleRecord {
                unit_name: "u_pos".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 1,
                region_size: 2,
                ml_feature_count: 2,
                ml_rank: "1|2".to_string(),
                selected_row_indices: vec![0, 1],
                display_ops: vec![BeamBinaryOp::Or],
                display_negated: vec![false, false],
                snp_name: "1_100|1_200".to_string(),
                expr: "BIN(1_100) OR BIN(1_200)".to_string(),
                chrom_field: "1".to_string(),
                bim_snp_name: "1_100|1_200".to_string(),
                bim_allele0: "A,A".to_string(),
                bim_allele1: "G,G".to_string(),
                pos: 100,
                score: 5.0,
                delta_score: "0.1|0.2->5.0".to_string(),
                support_bits: Some(GarfieldRuleSupportBits::Binary(
                    full_bits.into_boxed_slice(),
                )),
            },
        ];
        let empty_logic_bits = GarfieldLogicBits {
            bits_flat: Vec::new(),
            bits_hi_flat: None,
            row_words: 0,
            sample_ids: Vec::new(),
            sites: Vec::new(),
            group_ids: Vec::new(),
            n_samples: 0,
        };
        let deduped = dedup_logic_rule_records(records, &empty_logic_bits).unwrap();
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].snp_name, "!1_100&!1_200");
        assert_eq!(deduped[0].expr, "NOT BIN(1_100) AND NOT BIN(1_200)");
    }

    #[test]
    fn test_write_logic_pseudo_plink_expands_singletons_and_multi_pos_children() {
        let dir = make_temp_dir("pseudo_export_expand");
        let prefix = dir.join("pseudo");
        let prefix_str = prefix.to_string_lossy().to_string();
        let sample_ids = vec![
            "S1".to_string(),
            "S2".to_string(),
            "S3".to_string(),
            "S4".to_string(),
        ];
        let sites = build_logic_sites_from_metadata(
            &vec![test_site("1", 100), test_site("1", 200)],
            GarfieldBinMode::Bin,
        );
        let logic_bits = GarfieldLogicBits {
            bits_flat: vec![0b0101u64, 0b0110u64],
            bits_hi_flat: None,
            row_words: 1,
            sample_ids: sample_ids.clone(),
            sites,
            group_ids: vec![0, 1],
            n_samples: 4,
        };
        let records = vec![GarfieldLogicRuleRecord {
            unit_name: "u1".to_string(),
            unit_kind: "window".to_string(),
            unit_index: 1,
            region_size: 2,
            ml_feature_count: 2,
            ml_rank: "1&!2".to_string(),
            selected_row_indices: vec![0, 1],
            display_ops: vec![BeamBinaryOp::And],
            display_negated: vec![false, true],
            snp_name: "1_100&!1_200".to_string(),
            expr: "BIN(1_100) AND NOT BIN(1_200)".to_string(),
            chrom_field: "1".to_string(),
            bim_snp_name: "1_100&!1_200".to_string(),
            bim_allele0: "A,C".to_string(),
            bim_allele1: "G,T".to_string(),
            pos: 100,
            score: 10.0,
            delta_score: "0.1&!0.2->10".to_string(),
            support_bits: None,
        }];

        write_logic_pseudo_plink(
            prefix_str.as_str(),
            sample_ids.as_slice(),
            records.as_slice(),
            &logic_bits,
        )
        .unwrap();

        let bim_txt = fs::read_to_string(format!("{prefix_str}.bim")).unwrap();
        assert_eq!(
            bim_txt.lines().collect::<Vec<_>>(),
            vec![
                "1\t1_100\t0\t100\tA\tG",
                "1\t!1_200\t0\t200\tG\tA",
                "1\t1_100&!1_200\t0\t100\tA,C\tG,T",
                "1\t1_100&!1_200\t0\t200\tA,C\tG,T",
            ]
        );

        let bed_bytes = fs::read(format!("{prefix_str}.bed")).unwrap();
        assert_eq!(bed_bytes, vec![0x6C, 0x1B, 0x01, 0x33, 0xC3, 0x03, 0x03]);
    }

    #[test]
    fn test_write_logic_dual_bits_as_plink_row_preserves_heterozygotes() {
        let mut row_buf = vec![0u8; 1];
        let mut out = Vec::<u8>::new();
        write_logic_dual_bits_as_plink_row(
            &mut out,
            row_buf.as_mut_slice(),
            &[0b1110u64],
            &[0b0100u64],
            4,
            false,
        )
        .unwrap();
        assert_eq!(out, vec![0x74u8]);

        out.clear();
        write_logic_dual_bits_as_plink_row(
            &mut out,
            row_buf.as_mut_slice(),
            &[0b1110u64],
            &[0b0100u64],
            4,
            true,
        )
        .unwrap();
        assert_eq!(out, vec![0x47u8]);
    }

    fn test_site(chrom: &str, pos: i32) -> SiteInfo {
        SiteInfo {
            chrom: chrom.to_string(),
            pos,
            snp: format!("{chrom}_{pos}"),
            ref_allele: "A".to_string(),
            alt_allele: "G".to_string(),
        }
    }

    fn test_named_site(chrom: &str, pos: i32, snp: &str) -> SiteInfo {
        SiteInfo {
            chrom: chrom.to_string(),
            pos,
            snp: snp.to_string(),
            ref_allele: "A".to_string(),
            alt_allele: "G".to_string(),
        }
    }

    #[test]
    fn test_build_valid_null_chunks_uses_fixed_non_overlapping_windows() {
        let mut sites = Vec::<SiteInfo>::new();
        for pos in 0..55 {
            sites.push(test_site("1", pos * 1_000));
        }
        for pos in 200..260 {
            sites.push(test_site("1", pos * 1_000));
        }
        for pos in 500..520 {
            sites.push(test_site("1", pos * 1_000));
        }
        let (chunks, spans) = build_valid_null_chunks(&sites, 200_000, 50).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start_index, 0);
        assert_eq!(chunks[0].end_index, 55);
        assert_eq!(chunks[0].window_index, 0);
        assert_eq!(chunks[1].start_index, 55);
        assert_eq!(chunks[1].end_index, 115);
        assert_eq!(chunks[1].window_index, 1);
    }

    #[test]
    fn test_sample_null_chunks_stratified_preserves_chromosome_proportions() {
        let mut sites = Vec::<SiteInfo>::new();
        for win in 0..8 {
            let start = win * 200_000;
            for offset in 0..60 {
                sites.push(test_site("1", start + offset));
            }
        }
        for win in 0..2 {
            let start = win * 200_000;
            for offset in 0..60 {
                sites.push(test_site("2", start + offset));
            }
        }
        let (picked, total_valid) =
            sample_null_chunks_stratified(&sites, 100_000, 5, 50, 7).unwrap();
        assert_eq!(total_valid, 10);
        assert_eq!(picked.len(), 5);
        let chr1 = picked.iter().filter(|c| c.chrom_code == 1).count();
        let chr2 = picked.iter().filter(|c| c.chrom_code == 2).count();
        assert_eq!(chr1, 4);
        assert_eq!(chr2, 1);
    }

    #[test]
    fn test_resolve_ml_top_random_counts_keeps_random_slice_for_small_keep_k() {
        assert_eq!(resolve_ml_top_random_counts(2, 0.80), (1, 1));
        assert_eq!(resolve_ml_top_random_counts(3, 0.80), (2, 1));
        assert_eq!(resolve_ml_top_random_counts(10, 0.80), (8, 2));
    }

    #[test]
    fn test_parse_scan_bimrange_item_accepts_bp_interval() {
        let parsed = parse_scan_bimrange_item("chr10:111000000-111200000").unwrap();
        assert_eq!(parsed.chrom, "chr10");
        assert_eq!(parsed.bp_start, 111_000_000);
        assert_eq!(parsed.bp_end, 111_200_000);
    }

    #[test]
    fn test_unit_overlaps_scan_bimranges_uses_any_span_overlap() {
        let unit = GarfieldLogicUnit {
            label: "u1".to_string(),
            indices: vec![1, 2, 3],
            spans: vec![
                GarfieldUnitSpan {
                    chrom: "10".to_string(),
                    bp_start: 111_000_000,
                    bp_end: 111_100_000,
                },
                GarfieldUnitSpan {
                    chrom: "11".to_string(),
                    bp_start: 222_000_000,
                    bp_end: 222_100_000,
                },
            ],
        };
        assert!(unit_overlaps_scan_bimranges(
            &unit,
            &[GarfieldScanBimRange {
                chrom: "chr10".to_string(),
                bp_start: 111_050_000,
                bp_end: 111_250_000,
            }]
        ));
        assert!(!unit_overlaps_scan_bimranges(
            &unit,
            &[GarfieldScanBimRange {
                chrom: "12".to_string(),
                bp_start: 111_050_000,
                bp_end: 111_250_000,
            }]
        ));
    }

    #[test]
    fn test_geneset_ld_prune_drops_cross_chrom_high_ld_rows() {
        let (bits_flat, row_words) =
            pack_test_binary_rows(&[vec![1u8, 1, 0, 0], vec![1u8, 1, 0, 0], vec![1u8, 0, 1, 0]]);
        let logic_bits = GarfieldLogicBits {
            bits_flat,
            bits_hi_flat: None,
            row_words,
            sample_ids: vec![
                "s1".to_string(),
                "s2".to_string(),
                "s3".to_string(),
                "s4".to_string(),
            ],
            sites: vec![
                GarfieldLogicSite {
                    chrom: "1".into(),
                    pos: 100,
                    snp: "chr1.s_100".into(),
                    ref_allele: "A".into(),
                    alt_allele: "G".into(),
                    mode: GarfieldLogicSiteMode::Bin,
                },
                GarfieldLogicSite {
                    chrom: "2".into(),
                    pos: 200,
                    snp: "chr2.s_200".into(),
                    ref_allele: "A".into(),
                    alt_allele: "G".into(),
                    mode: GarfieldLogicSiteMode::Bin,
                },
                GarfieldLogicSite {
                    chrom: "3".into(),
                    pos: 300,
                    snp: "chr3.s_300".into(),
                    ref_allele: "A".into(),
                    alt_allele: "G".into(),
                    mode: GarfieldLogicSiteMode::Bin,
                },
            ],
            group_ids: vec![0, 1, 2],
            n_samples: 4,
        };
        let unit = GarfieldLogicUnit {
            label: "triad".to_string(),
            indices: vec![0, 1, 2],
            spans: vec![
                GarfieldUnitSpan {
                    chrom: "1".to_string(),
                    bp_start: 90,
                    bp_end: 110,
                },
                GarfieldUnitSpan {
                    chrom: "2".to_string(),
                    bp_start: 190,
                    bp_end: 210,
                },
                GarfieldUnitSpan {
                    chrom: "3".to_string(),
                    bp_start: 290,
                    bp_end: 310,
                },
            ],
        };
        let kept = maybe_prune_geneset_unit_rows_by_ld(
            &unit,
            unit.indices.as_slice(),
            "geneset",
            &logic_bits,
            &[0, 1, 2, 3],
        )
        .unwrap();
        assert_eq!(kept, vec![0usize, 2usize]);
    }

    #[test]
    fn test_geneset_ld_exact_bounds_match_direct_r2_grid() {
        for n_samples in 4usize..=16usize {
            for &r2_threshold in &[0.2_f64, 0.5_f64, 0.8_f64] {
                for support_i in 1usize..n_samples {
                    for support_j in 1usize..n_samples {
                        if binary_pair_abs_r2_upper_bound_from_support(
                            support_i, support_j, n_samples,
                        ) < r2_threshold
                        {
                            continue;
                        }
                        let bounds = geneset_ld_exact_bounds_from_support(
                            support_i,
                            support_j,
                            n_samples,
                            r2_threshold,
                        );
                        let both_min = support_i
                            .saturating_add(support_j)
                            .saturating_sub(n_samples);
                        let both_max = support_i.min(support_j);
                        for both_one in both_min..=both_max {
                            let direct = binary_pair_high_ld_from_support_and_both_one(
                                support_i,
                                support_j,
                                both_one,
                                n_samples,
                                r2_threshold,
                            );
                            let fast = bounds.matches(both_one);
                            assert_eq!(
                                fast, direct,
                                "n_samples={n_samples} r2={r2_threshold} support_i={support_i} support_j={support_j} both_one={both_one} bounds={bounds:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_select_logic_unit_global_rows_applies_geneset_ld_prune_without_ml() {
        let (bits_flat, row_words) =
            pack_test_binary_rows(&[vec![1u8, 1, 0, 0], vec![1u8, 1, 0, 0], vec![1u8, 0, 1, 0]]);
        let logic_bits = GarfieldLogicBits {
            bits_flat,
            bits_hi_flat: None,
            row_words,
            sample_ids: vec![
                "s1".to_string(),
                "s2".to_string(),
                "s3".to_string(),
                "s4".to_string(),
            ],
            sites: vec![
                GarfieldLogicSite {
                    chrom: "1".into(),
                    pos: 100,
                    snp: "chr1.s_100".into(),
                    ref_allele: "A".into(),
                    alt_allele: "G".into(),
                    mode: GarfieldLogicSiteMode::Bin,
                },
                GarfieldLogicSite {
                    chrom: "2".into(),
                    pos: 200,
                    snp: "chr2.s_200".into(),
                    ref_allele: "A".into(),
                    alt_allele: "G".into(),
                    mode: GarfieldLogicSiteMode::Bin,
                },
                GarfieldLogicSite {
                    chrom: "3".into(),
                    pos: 300,
                    snp: "chr3.s_300".into(),
                    ref_allele: "A".into(),
                    alt_allele: "G".into(),
                    mode: GarfieldLogicSiteMode::Bin,
                },
            ],
            group_ids: vec![0, 1, 2],
            n_samples: 4,
        };
        let unit = GarfieldLogicUnit {
            label: "triad".to_string(),
            indices: vec![0, 1, 2],
            spans: vec![
                GarfieldUnitSpan {
                    chrom: "1".to_string(),
                    bp_start: 90,
                    bp_end: 110,
                },
                GarfieldUnitSpan {
                    chrom: "2".to_string(),
                    bp_start: 190,
                    bp_end: 210,
                },
                GarfieldUnitSpan {
                    chrom: "3".to_string(),
                    bp_start: 290,
                    bp_end: 310,
                },
            ],
        };
        let selected = select_logic_unit_global_rows(
            &unit,
            "geneset",
            ResponseKind::Continuous,
            None,
            ImportanceKind::Imp,
            PermutationConfig {
                n_repeats: 1,
                scoring: crate::ml::common::PermutationScoring::Auto,
                seed: 1,
            },
            64,
            0.0,
            ExtraTreesConfig {
                n_estimators: 1,
                max_depth: 1,
                min_samples_leaf: 1,
                min_samples_split: 2,
                bootstrap: false,
                feature_subsample: 0.0,
                seed: 1,
                allow_parallel: false,
            },
            &logic_bits,
            &[0, 1, 2, 3],
            &[0.1, 0.2, 0.3, 0.4],
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(selected, vec![0usize, 2usize]);
    }

    #[test]
    fn test_resolve_simbench_ml_rank_preserves_site_order_and_missing_marks() {
        let contexts = vec![GarfieldUnitMlContext {
            unit_index: 7,
            selected_global_rows: vec![100, 300, 200],
            ranked_global_rows: vec![100, 300, 200],
        }];
        let site_row_candidates = vec![vec![300, 301], vec![999], vec![200, 201]];
        let (rank_txt, ml_feature_count) = resolve_simbench_ml_rank(
            SimBenchLogic::And,
            site_row_candidates.as_slice(),
            contexts.as_slice(),
        );
        assert_eq!(rank_txt, "2 & . & 3");
        assert_eq!(ml_feature_count, 3);
    }

    #[test]
    fn test_resolve_simbench_ml_rank_falls_back_to_full_ranking_when_not_selected() {
        let contexts = vec![GarfieldUnitMlContext {
            unit_index: 9,
            selected_global_rows: vec![100, 200],
            ranked_global_rows: vec![400, 300, 200, 100],
        }];
        let site_row_candidates = vec![vec![300], vec![400]];
        let (rank_txt, ml_feature_count) = resolve_simbench_ml_rank(
            SimBenchLogic::And,
            site_row_candidates.as_slice(),
            contexts.as_slice(),
        );
        assert_eq!(rank_txt, "2 & 1");
        assert_eq!(ml_feature_count, 4);
    }

    #[test]
    fn test_simbench_raw_normalization_is_allele_orientation_invariant() {
        let mut row_a = vec![0.0_f32, 0.0, 0.0, 1.0, 2.0];
        let mut site_a = SiteInfo {
            chrom: "10".to_string(),
            pos: 111_023_852,
            snp: "site_a".to_string(),
            ref_allele: "C".to_string(),
            alt_allele: "T".to_string(),
        };
        let mut row_b = vec![2.0_f32, 2.0, 2.0, 1.0, 0.0];
        let mut site_b = SiteInfo {
            chrom: "10".to_string(),
            pos: 111_023_852,
            snp: "site_b".to_string(),
            ref_allele: "T".to_string(),
            alt_allele: "C".to_string(),
        };

        normalize_simbench_raw_row(&mut row_a, &mut site_a).unwrap();
        normalize_simbench_raw_row(&mut row_b, &mut site_b).unwrap();

        let (ge1_a, ge2_a) = pack_simbench_raw_row_dual_words(&row_a).unwrap();
        let (ge1_b, ge2_b) = pack_simbench_raw_row_dual_words(&row_b).unwrap();
        assert_eq!(site_a.ref_allele, site_b.ref_allele);
        assert_eq!(site_a.alt_allele, site_b.alt_allele);
        assert_eq!(ge1_a, ge1_b);
        assert_eq!(ge2_a, ge2_b);
    }

    #[test]
    fn test_fill_bin_logic_row_bits_fuzzy_respects_row_flip() {
        fn pack_raw_hardcalls(geno: &[u8]) -> Vec<u8> {
            let mut out = vec![0u8; geno.len().div_ceil(4)];
            for (i, &g) in geno.iter().enumerate() {
                let code = match g {
                    0 => 0b00u8,
                    1 => 0b10u8,
                    2 => 0b11u8,
                    _ => 0b01u8,
                };
                out[i >> 2] |= code << ((i & 3) * 2);
            }
            out
        }

        fn unpack_dual_bits(ge1: &[u64], ge2: &[u64], n: usize) -> Vec<u8> {
            (0..n)
                .map(|i| {
                    (((ge1[i >> 6] >> (i & 63)) & 1u64) + ((ge2[i >> 6] >> (i & 63)) & 1u64)) as u8
                })
                .collect()
        }

        let raw_row = vec![2u8, 2, 2, 1, 0];
        let packed_row = pack_raw_hardcalls(raw_row.as_slice());
        let sample_indices = (0usize..raw_row.len()).collect::<Vec<_>>();
        let sample_plan = build_logic_sample_plan(sample_indices.as_slice());
        let mut dst_ge1 = vec![0u64; words_for_samples(raw_row.len())];
        let mut dst_ge2 = vec![0u64; words_for_samples(raw_row.len())];
        fill_bin_logic_row_bits_fuzzy(
            packed_row.as_slice(),
            true,
            sample_plan.as_slice(),
            dst_ge1.as_mut_slice(),
            dst_ge2.as_mut_slice(),
        );

        let mut row = raw_row.iter().map(|&g| g as f32).collect::<Vec<_>>();
        let mut site = SiteInfo {
            chrom: "10".to_string(),
            pos: 111_023_852,
            snp: "fill_flip_site".to_string(),
            ref_allele: "T".to_string(),
            alt_allele: "C".to_string(),
        };
        normalize_simbench_raw_row(&mut row, &mut site).unwrap();
        let (exp_ge1, exp_ge2) = pack_simbench_raw_row_dual_words(&row).unwrap();

        assert_eq!(
            unpack_dual_bits(dst_ge1.as_slice(), dst_ge2.as_slice(), raw_row.len()),
            unpack_dual_bits(exp_ge1.as_slice(), exp_ge2.as_slice(), raw_row.len())
        );
    }

    #[test]
    fn test_insert_topk_density_hit_keeps_highest_scores() {
        let mut bucket = Vec::<(usize, f64)>::new();
        insert_topk_density_hit(&mut bucket, 2, 5.0, 3);
        insert_topk_density_hit(&mut bucket, 1, 9.0, 3);
        insert_topk_density_hit(&mut bucket, 4, 7.0, 3);
        insert_topk_density_hit(&mut bucket, 0, 8.0, 3);
        assert_eq!(bucket.len(), 3);
        assert_eq!(bucket[0], (1, 9.0));
        assert_eq!(bucket[1], (0, 8.0));
        assert_eq!(bucket[2], (4, 7.0));
    }

    #[test]
    fn test_build_unit_window_group_ids_assigns_geneset_spans() {
        let unit = GarfieldLogicUnit {
            label: "triad".to_string(),
            indices: vec![0, 1, 2],
            spans: vec![
                GarfieldUnitSpan {
                    chrom: "1".to_string(),
                    bp_start: 100,
                    bp_end: 200,
                },
                GarfieldUnitSpan {
                    chrom: "1".to_string(),
                    bp_start: 300,
                    bp_end: 400,
                },
                GarfieldUnitSpan {
                    chrom: "2".to_string(),
                    bp_start: 500,
                    bp_end: 600,
                },
            ],
        };
        let sites = vec![
            SiteInfo {
                chrom: "1".to_string(),
                pos: 150,
                snp: "triad_1".to_string(),
                ref_allele: "A".to_string(),
                alt_allele: "G".to_string(),
            },
            SiteInfo {
                chrom: "1".to_string(),
                pos: 350,
                snp: "triad_2".to_string(),
                ref_allele: "A".to_string(),
                alt_allele: "G".to_string(),
            },
            SiteInfo {
                chrom: "2".to_string(),
                pos: 580,
                snp: "triad_3".to_string(),
                ref_allele: "A".to_string(),
                alt_allele: "G".to_string(),
            },
        ];
        let gids =
            build_unit_window_group_ids(&unit, &[0, 1, 2], sites.as_slice(), "geneset").unwrap();
        assert_eq!(gids, vec![0usize, 1usize, 2usize]);
    }

    #[test]
    fn test_build_unit_window_group_ids_prefers_nearest_overlap_span() {
        let unit = GarfieldLogicUnit {
            label: "overlap".to_string(),
            indices: vec![0],
            spans: vec![
                GarfieldUnitSpan {
                    chrom: "1".to_string(),
                    bp_start: 100,
                    bp_end: 300,
                },
                GarfieldUnitSpan {
                    chrom: "1".to_string(),
                    bp_start: 220,
                    bp_end: 420,
                },
            ],
        };
        let sites = vec![SiteInfo {
            chrom: "1".to_string(),
            pos: 260,
            snp: "overlap_1".to_string(),
            ref_allele: "A".to_string(),
            alt_allele: "G".to_string(),
        }];
        let gids = build_unit_window_group_ids(&unit, &[0], sites.as_slice(), "geneset").unwrap();
        assert_eq!(gids, vec![1usize]);
    }
}
