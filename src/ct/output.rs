use crate::ct::conditional::{
    build_background, fit_conditional_prepared, fit_conditional_prepared_profiled,
    fit_conditional_prepared_whitened, fit_conditional_prepared_whitened_profiled,
    prepare_conditional_fit, whiten_pair_design_batch_profiled, ConditionMode,
    ConditionalFitTimings, CovarianceMetric, FitStatus, InferenceKind, PairDesign,
    PreparedConditionalFit, WhiteningStageTimings,
};
use crate::ct::hmp::HmpSource;
use crate::ct::input::SiteSource;
use crate::ct::null::{fit_global_null, ModelKind};
use crate::ct::packed::{FilledPair, PairQcThresholds};
use crate::ct::parents::{
    parent_sample_ids, read_parent_map, ParentBedListSource, ParentBedSource, ParentDerivedSource,
};
use crate::ct::screen::{CoreCandidates, ScreenCandidate};
use crate::ct::vcf::VcfSource;
use crate::ct::window::{
    scan_source_dxd_fvlmm, scan_source_dxd_lm, scan_source_fvlmm, scan_source_lm, FvLmmScanContext,
    ScanConfig, ScanStats,
};
use csv::WriterBuilder;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const OUTPUT_HEADER: &[&str] = &[
    "snp1", "snp2", "chrom1", "pos1", "chrom2", "pos2", "n_cis", "n_trans", "beta_a1", "beta_d1",
    "beta_a2", "beta_d2", "beta_c", "se_c", "beta_t", "se_t", "pwald1", "pwald2", "beta_dd",
    "se_dd", "pwald_dd",
];
const DXD_OUTPUT_HEADER: &[&str] = &[
    "snp1",
    "snp2",
    "chrom1",
    "pos1",
    "chrom2",
    "pos2",
    "n_double_het",
    "beta_dd",
    "se_dd",
    "pwald_dd",
];

// Keep enough right-hand sides per solve to reuse the covariance factor, then
// run independent pair batches concurrently. Cap each work matrix so large
// cohorts do not create unbounded transient allocations.
const MAX_PAIR_WHITENING_RHS: usize = 336;
const MAX_PAIR_WHITENING_MATRIX_BYTES: usize = 16 * 1024 * 1024;
const MAX_PENDING_CANDIDATE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Default)]
struct CandidateProfileStats {
    candidates: usize,
    candidate_work_thread: Duration,
    design_build_thread: Duration,
    conditional_fit_thread: ConditionalFitTimings,
    whitening_thread: WhiteningStageTimings,
    row_format_thread: Duration,
    callback_wall: Duration,
    parallel_fit_wall: Duration,
    sort_wall: Duration,
    tsv_write_wall: Duration,
    core_flush_wall: Duration,
}

struct CandidateTaskTimings {
    elapsed: Duration,
    design_build: Duration,
    conditional_fit: ConditionalFitTimings,
    whitening: WhiteningStageTimings,
    row_format: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FilledCategoryCounts {
    pub(crate) cis: usize,
    pub(crate) trans: usize,
    pub(crate) other: usize,
    pub(crate) n_imputed: usize,
}

pub(crate) fn output_header() -> &'static [&'static str] {
    OUTPUT_HEADER
}

pub(crate) fn dxd_output_header() -> &'static [&'static str] {
    DXD_OUTPUT_HEADER
}

pub(crate) fn validate_sample_ids(target: &[String], observed: &[String]) -> Result<(), String> {
    if target.is_empty() {
        return Err("CT requires at least one selected sample ID".to_string());
    }
    let mut target_seen = std::collections::HashSet::with_capacity(target.len());
    for (index, id) in target.iter().enumerate() {
        if id.trim().is_empty() {
            return Err(format!("selected sample ID at position {index} is empty"));
        }
        if !target_seen.insert(id) {
            return Err(format!("duplicate selected sample ID '{id}'"));
        }
    }
    let mut observed_seen = std::collections::HashSet::with_capacity(observed.len());
    for (index, id) in observed.iter().enumerate() {
        if id.trim().is_empty() || !observed_seen.insert(id) {
            return Err(format!(
                "GRM/source sample IDs are empty or duplicated at position {index}"
            ));
        }
    }
    if target != observed {
        return Err("GRM/source sample IDs do not exactly match the selected CT order".to_string());
    }
    Ok(())
}

pub(crate) fn filled_category_counts(filled: &FilledPair) -> Result<FilledCategoryCounts, String> {
    let n = filled.z.len();
    if n == 0 || filled.imputed.len() != n || filled.g_i.len() != n || filled.g_j.len() != n {
        return Err("filled pair vectors have inconsistent lengths".to_string());
    }
    let mut counts = FilledCategoryCounts {
        cis: 0,
        trans: 0,
        other: 0,
        n_imputed: 0,
    };
    for (index, (&z, &imputed)) in filled.z.iter().zip(&filled.imputed).enumerate() {
        match z {
            1 => counts.cis += 1,
            -1 => counts.trans += 1,
            0 => counts.other += 1,
            _ => return Err(format!("invalid filled phase code {z} at sample {index}")),
        }
        counts.n_imputed += usize::from(imputed);
    }
    if counts.n_imputed != filled.n_imputed || counts.cis + counts.trans + counts.other != n {
        return Err("filled pair count invariants do not hold".to_string());
    }
    Ok(counts)
}

enum OpenedSource {
    Vcf {
        source: VcfSource,
        path: String,
    },
    Hmp {
        source: HmpSource,
        path: String,
    },
    ParentVcf {
        source: ParentDerivedSource<VcfSource>,
        path: String,
        cross_path: String,
    },
    ParentHmp {
        source: ParentDerivedSource<HmpSource>,
        path: String,
        cross_path: String,
    },
    Parents {
        source: ParentBedSource,
        father_prefix: String,
        mother_prefix: String,
        cross_path: String,
    },
    ParentBfileList {
        source: ParentBedListSource,
        manifest_path: String,
        cross_path: String,
    },
}

impl OpenedSource {
    fn open(
        source_kind: &str,
        sample_ids: &[String],
        vcf_path: Option<&str>,
        hmp_path: Option<&str>,
        father_prefix: Option<&str>,
        mother_prefix: Option<&str>,
        cross_path: Option<&str>,
    ) -> Result<Self, String> {
        let opened = match source_kind.trim().to_ascii_lowercase().as_str() {
            "vcf" => {
                let path = required_path(vcf_path, "VCF path")?;
                let source = VcfSource::open(path, sample_ids)?;
                Self::Vcf {
                    source,
                    path: path.to_string(),
                }
            }
            "hmp" => {
                let path = required_path(hmp_path, "HMP path")?;
                let source = HmpSource::open(path, sample_ids)?;
                Self::Hmp {
                    source,
                    path: path.to_string(),
                }
            }
            "parent-vcf" => {
                let path = required_path(vcf_path, "VCF path")?;
                let cross = required_path(cross_path, "parent map path")?;
                let links = read_parent_map(cross, sample_ids)?;
                let parent_ids = parent_sample_ids(&links);
                let source = VcfSource::open(path, &parent_ids)?;
                let source = ParentDerivedSource::new(source, links)?;
                Self::ParentVcf {
                    source,
                    path: path.to_string(),
                    cross_path: cross.to_string(),
                }
            }
            "parent-hmp" => {
                let path = required_path(hmp_path, "HMP path")?;
                let cross = required_path(cross_path, "parent map path")?;
                let links = read_parent_map(cross, sample_ids)?;
                let parent_ids = parent_sample_ids(&links);
                let source = HmpSource::open(path, &parent_ids)?;
                let source = ParentDerivedSource::new(source, links)?;
                Self::ParentHmp {
                    source,
                    path: path.to_string(),
                    cross_path: cross.to_string(),
                }
            }
            "bfile" | "parents" => {
                let father = required_path(father_prefix, "father BED prefix")?;
                let mother = required_path(mother_prefix, "mother BED prefix")?;
                let cross = required_path(cross_path, "cross table path")?;
                let source = ParentBedSource::open(father, mother, cross, sample_ids)?;
                Self::Parents {
                    source,
                    father_prefix: father.to_string(),
                    mother_prefix: mother.to_string(),
                    cross_path: cross.to_string(),
                }
            }
            "bfile-list" | "bfile_list" => {
                let manifest = required_path(father_prefix, "parent BFILE manifest")?;
                let cross = required_path(cross_path, "parent map path")?;
                let source = ParentBedListSource::open(manifest, cross, sample_ids)?;
                Self::ParentBfileList {
                    source,
                    manifest_path: manifest.to_string(),
                    cross_path: cross.to_string(),
                }
            }
            other => return Err(format!("unsupported CT source kind '{other}'")),
        };
        validate_sample_ids(sample_ids, opened.sample_ids())?;
        Ok(opened)
    }

    fn sample_ids(&self) -> &[String] {
        match self {
            Self::Vcf { source, .. } => source.sample_ids(),
            Self::Hmp { source, .. } => source.sample_ids(),
            Self::ParentVcf { source, .. } => source.sample_ids(),
            Self::ParentHmp { source, .. } => source.sample_ids(),
            Self::Parents { source, .. } => source.sample_ids(),
            Self::ParentBfileList { source, .. } => source.sample_ids(),
        }
    }

    fn summary(&self) -> Value {
        match self {
            Self::Vcf { source, path } => {
                let stats = source.stats();
                json!({
                    "kind": "vcf",
                    "path": path,
                    "records_seen": stats.records_seen,
                    "sites_emitted": stats.sites_emitted,
                    "skipped_non_snp": stats.skipped_non_snp,
                    "skipped_unsupported_gt": stats.skipped_unsupported_gt,
                })
            }
            Self::Hmp { source, path } => {
                json!({
                    "kind": "hmp",
                    "path": path,
                    "n_samples": source.sample_ids().len(),
                })
            }
            Self::ParentVcf {
                source,
                path,
                cross_path,
            } => {
                json!({
                    "kind": "parent_vcf",
                    "path": path,
                    "parent_map": cross_path,
                    "n_children": source.sample_ids().len(),
                })
            }
            Self::ParentHmp {
                source,
                path,
                cross_path,
            } => {
                json!({
                    "kind": "parent_hmp",
                    "path": path,
                    "parent_map": cross_path,
                    "n_children": source.sample_ids().len(),
                })
            }
            Self::Parents {
                source,
                father_prefix,
                mother_prefix,
                cross_path,
            } => {
                let stats = source.stats();
                json!({
                    "kind": if father_prefix == mother_prefix { "single_bfile_parent_map" } else { "dual_parent_bed" },
                    "father_prefix": father_prefix,
                    "mother_prefix": mother_prefix,
                    "cross_path": cross_path,
                    "father_markers": stats.father_markers,
                    "mother_markers": stats.mother_markers,
                    "matched_markers": stats.matched_markers,
                    "unmatched_father_markers": stats.unmatched_father_markers,
                    "unmatched_mother_markers": stats.unmatched_mother_markers,
                    "allele_conflicts": stats.allele_conflicts,
                    "non_simple_markers": stats.non_simple_markers,
                    "child_calls_from_heterozygous_parent": stats.child_calls_from_heterozygous_parent,
                    "child_calls_from_missing_parent": stats.child_calls_from_missing_parent,
                })
            }
            Self::ParentBfileList {
                source,
                manifest_path,
                cross_path,
            } => {
                let stats = source.stats();
                let chromosomes = source
                    .source_prefixes()
                    .into_iter()
                    .map(|(father_prefix, mother_prefix)| {
                        json!({
                            "father_prefix": father_prefix,
                            "mother_prefix": mother_prefix,
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "kind": "dual_parent_bed_list",
                    "manifest_path": manifest_path,
                    "cross_path": cross_path,
                    "chromosomes": chromosomes,
                    "father_markers": stats.father_markers,
                    "mother_markers": stats.mother_markers,
                    "matched_markers": stats.matched_markers,
                    "unmatched_father_markers": stats.unmatched_father_markers,
                    "unmatched_mother_markers": stats.unmatched_mother_markers,
                    "allele_conflicts": stats.allele_conflicts,
                    "non_simple_markers": stats.non_simple_markers,
                    "child_calls_from_heterozygous_parent": stats.child_calls_from_heterozygous_parent,
                    "child_calls_from_missing_parent": stats.child_calls_from_missing_parent,
                })
            }
        }
    }
}

impl SiteSource for OpenedSource {
    fn sample_ids(&self) -> &[String] {
        match self {
            Self::Vcf { source, .. } => source.sample_ids(),
            Self::Hmp { source, .. } => source.sample_ids(),
            Self::ParentVcf { source, .. } => source.sample_ids(),
            Self::ParentHmp { source, .. } => source.sample_ids(),
            Self::Parents { source, .. } => source.sample_ids(),
            Self::ParentBfileList { source, .. } => source.sample_ids(),
        }
    }

    fn prefetch_site_bytes_hint(&self) -> usize {
        match self {
            Self::Vcf { source, .. } => source.prefetch_site_bytes_hint(),
            Self::Hmp { source, .. } => source.prefetch_site_bytes_hint(),
            Self::ParentVcf { source, .. } => source.prefetch_site_bytes_hint(),
            Self::ParentHmp { source, .. } => source.prefetch_site_bytes_hint(),
            Self::Parents { source, .. } => source.prefetch_site_bytes_hint(),
            Self::ParentBfileList { source, .. } => source.prefetch_site_bytes_hint(),
        }
    }

    fn next_site(&mut self) -> Result<Option<crate::ct::input::Site>, String> {
        match self {
            Self::Vcf { source, .. } => source.next_site(),
            Self::Hmp { source, .. } => source.next_site(),
            Self::ParentVcf { source, .. } => source.next_site(),
            Self::ParentHmp { source, .. } => source.next_site(),
            Self::Parents { source, .. } => source.next_site(),
            Self::ParentBfileList { source, .. } => source.next_site(),
        }
    }

    fn next_scanned_site(&mut self) -> Result<Option<crate::ct::input::ScannedSite>, String> {
        match self {
            Self::Vcf { source, .. } => source.next_scanned_site(),
            Self::Hmp { source, .. } => source.next_scanned_site(),
            Self::ParentVcf { source, .. } => source.next_scanned_site(),
            Self::ParentHmp { source, .. } => source.next_scanned_site(),
            Self::Parents { source, .. } => source.next_scanned_site(),
            Self::ParentBfileList { source, .. } => source.next_scanned_site(),
        }
    }
}

fn required_path<'a>(path: Option<&'a str>, label: &str) -> Result<&'a str, String> {
    path.map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} is required for this CT source"))
}

fn output_paths(prefix: &str, dxd: bool) -> Result<(PathBuf, PathBuf), String> {
    let prefix = prefix.trim();
    if prefix.is_empty() {
        return Err("CT output prefix must not be empty".to_string());
    }
    let base = PathBuf::from(prefix);
    let parent = base.parent().filter(|path| !path.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent).map_err(|e| format!("create output directory: {e}"))?;
    }
    let suffix = if dxd { "dxd" } else { "ct" };
    Ok((
        PathBuf::from(format!("{prefix}.{suffix}.tsv")),
        PathBuf::from(format!("{prefix}.{suffix}.json")),
    ))
}

fn write_tsv_header<W: Write>(writer: &mut csv::Writer<W>) -> Result<(), String> {
    writer
        .write_record(output_header())
        .map_err(|e| format!("write CT TSV header: {e}"))
}

fn write_dxd_tsv_header<W: Write>(writer: &mut csv::Writer<W>) -> Result<(), String> {
    writer
        .write_record(dxd_output_header())
        .map_err(|e| format!("write CT D×D TSV header: {e}"))
}

fn row_for_candidate(
    candidate: &ScreenCandidate,
    fit: &crate::ct::conditional::ConditionalFit,
) -> Result<Vec<String>, String> {
    let _counts = filled_category_counts(&candidate.filled)?;
    let fields = vec![
        candidate.left.id.clone(),
        candidate.right.id.clone(),
        candidate.left.chrom.clone(),
        candidate.left.pos.to_string(),
        candidate.right.chrom.clone(),
        candidate.right.pos.to_string(),
        candidate.raw_counts.cis_observed().to_string(),
        candidate.raw_counts.trans_observed().to_string(),
        format_effect(fit.beta_a1),
        format_effect(fit.beta_d1),
        format_effect(fit.beta_a2),
        format_effect(fit.beta_d2),
        format_effect(fit.beta_cis),
        format_effect(fit.se_cis),
        format_effect(fit.beta_trans),
        format_effect(fit.se_trans),
        format_pvalue(fit.pwald1),
        format_pvalue(fit.pwald2),
        format_effect(fit.beta_dd),
        format_effect(fit.se_dd),
        format_pvalue(fit.pwald_dd),
    ];
    if fields.len() != output_header().len() {
        return Err(format!(
            "internal CT TSV field count mismatch: got {}, expected {}",
            fields.len(),
            output_header().len()
        ));
    }
    Ok(fields)
}

fn row_for_dxd_candidate(
    candidate: &ScreenCandidate,
    fit: &crate::ct::conditional::ConditionalFit,
) -> Result<Vec<String>, String> {
    if candidate.filled.z.iter().any(|&value| value != 0) {
        return Err("D×D candidate unexpectedly carries a phase label".to_string());
    }
    let n_double_het = candidate.raw_counts.phase_unknown;
    let fields = vec![
        candidate.left.id.clone(),
        candidate.right.id.clone(),
        candidate.left.chrom.clone(),
        candidate.left.pos.to_string(),
        candidate.right.chrom.clone(),
        candidate.right.pos.to_string(),
        n_double_het.to_string(),
        format_effect(fit.beta_dd),
        format_effect(fit.se_dd),
        format_pvalue(fit.pwald_dd),
    ];
    if fields.len() != dxd_output_header().len() {
        return Err(format!(
            "internal CT D×D TSV field count mismatch: got {}, expected {}",
            fields.len(),
            dxd_output_header().len()
        ));
    }
    Ok(fields)
}

fn finish_candidate_fit(
    candidate: &ScreenCandidate,
    fit: crate::ct::conditional::ConditionalFit,
    elapsed_before_format: Duration,
    design_build: Duration,
    conditional_fit: ConditionalFitTimings,
    whitening: WhiteningStageTimings,
    profile_timings: bool,
    dxd: bool,
) -> Result<
    (
        (f64, usize, Vec<String>, bool, bool, bool),
        CandidateTaskTimings,
    ),
    String,
> {
    let row_format_started = profile_timings.then(Instant::now);
    let fields = if dxd {
        row_for_dxd_candidate(candidate, &fit)?
    } else {
        row_for_candidate(candidate, &fit)?
    };
    let row_format_elapsed = row_format_started
        .map(|started| started.elapsed())
        .unwrap_or_default();
    let sort_stat = if dxd {
        if fit.beta_dd.is_finite() && fit.se_dd.is_finite() && fit.se_dd > 0.0 {
            (fit.beta_dd / fit.se_dd).powi(2)
        } else {
            f64::NEG_INFINITY
        }
    } else if fit.status == FitStatus::Testable && fit.phase_wald_statistic.is_finite() {
        fit.phase_wald_statistic
    } else {
        f64::NEG_INFINITY
    };
    Ok((
        (
            sort_stat,
            candidate.screen_rank,
            fields,
            !dxd && fit.pwald1.is_finite(),
            !dxd && fit.pwald2.is_finite(),
            fit.pwald_dd.is_finite(),
        ),
        CandidateTaskTimings {
            elapsed: elapsed_before_format + row_format_elapsed,
            design_build,
            conditional_fit,
            whitening,
            row_format: row_format_elapsed,
        },
    ))
}

type FittedCandidate = (
    (f64, usize, Vec<String>, bool, bool, bool),
    CandidateTaskTimings,
);

fn whitening_pairs_per_batch(n_samples: usize, condition_mode: ConditionMode) -> usize {
    let columns_per_pair = match condition_mode {
        ConditionMode::AddDom => 6,
        ConditionMode::Genotype => 9,
    };
    let bytes_per_rhs = n_samples.saturating_mul(std::mem::size_of::<f64>()).max(1);
    let rhs_by_workspace = MAX_PAIR_WHITENING_MATRIX_BYTES / bytes_per_rhs;
    let rhs_budget = MAX_PAIR_WHITENING_RHS
        .min(rhs_by_workspace.max(columns_per_pair))
        .max(columns_per_pair);
    (rhs_budget / columns_per_pair).max(1)
}

fn conditional_batch_target(
    n_samples: usize,
    threads: usize,
    condition_mode: ConditionMode,
) -> usize {
    let work_target =
        whitening_pairs_per_batch(n_samples, condition_mode).saturating_mul(threads.max(1));
    // FilledPair owns four sample-sized byte vectors. Limit how many cores
    // wait for fitting as well as the per-worker whitening workspace.
    let bytes_per_candidate = n_samples
        .saturating_mul(4)
        .saturating_add(std::mem::size_of::<ScreenCandidate>());
    let memory_target = MAX_PENDING_CANDIDATE_BYTES / bytes_per_candidate.max(1);
    work_target.min(memory_target.max(1)).max(1)
}

#[allow(clippy::too_many_arguments)]
fn fit_candidate_cores(
    cores: &[CoreCandidates],
    prepared: &PreparedConditionalFit,
    condition_mode: ConditionMode,
    dxd: bool,
    metric: &CovarianceMetric,
    inference: InferenceKind,
    profile_timings: bool,
    fit_pool: &rayon::ThreadPool,
) -> Result<Vec<Vec<FittedCandidate>>, String> {
    // Keep each core's original solve dimensions; fit cores concurrently and
    // preserve their order, including empty cores.
    fit_pool.install(|| {
        cores
            .par_iter()
            .map(|core| {
                fit_core_candidates(
                    core,
                    prepared,
                    condition_mode,
                    dxd,
                    metric,
                    inference,
                    profile_timings,
                    fit_pool,
                )
            })
            .collect()
    })
}

fn fit_core_candidates(
    core: &CoreCandidates,
    prepared: &PreparedConditionalFit,
    condition_mode: ConditionMode,
    dxd: bool,
    metric: &CovarianceMetric,
    inference: InferenceKind,
    profile_timings: bool,
    fit_pool: &rayon::ThreadPool,
) -> Result<Vec<FittedCandidate>, String> {
    if !metric.has_cholesky_factor() {
        return fit_pool.install(|| {
            core.candidates
                .par_iter()
                .map(|candidate| {
                    let task_started = profile_timings.then(Instant::now);
                    let design_started = profile_timings.then(Instant::now);
                    let (background, q_b) = build_background(&candidate.filled, condition_mode)?;
                    let phase = candidate
                        .filled
                        .z
                        .iter()
                        .map(|&value| value as f64)
                        .collect::<Vec<_>>();
                    let design_build = design_started
                        .map(|started| started.elapsed())
                        .unwrap_or_default();
                    let mut conditional_timings = ConditionalFitTimings::default();
                    let fit_result = if profile_timings {
                        fit_conditional_prepared_profiled(
                            prepared,
                            &background,
                            q_b,
                            &phase,
                            metric,
                            inference,
                            Some(q_b - 1),
                            &mut conditional_timings,
                        )
                    } else {
                        fit_conditional_prepared(
                            prepared,
                            &background,
                            q_b,
                            &phase,
                            metric,
                            inference,
                            Some(q_b - 1),
                        )
                    };
                    let fit = fit_result.map_err(|error| {
                        format!(
                            "fit pair {}:{}-{}:{}: {error}",
                            candidate.left.chrom,
                            candidate.left.pos,
                            candidate.right.chrom,
                            candidate.right.pos
                        )
                    })?;
                    let elapsed = task_started
                        .map(|started| started.elapsed())
                        .unwrap_or_default();
                    finish_candidate_fit(
                        candidate,
                        fit,
                        elapsed,
                        design_build,
                        conditional_timings,
                        WhiteningStageTimings::default(),
                        profile_timings,
                        dxd,
                    )
                })
                .collect::<Result<Vec<_>, String>>()
        });
    }

    if core.candidates.is_empty() {
        return Ok(Vec::new());
    }
    let q_b: usize = match condition_mode {
        ConditionMode::AddDom => 5,
        ConditionMode::Genotype => 8,
    };
    let max_pairs_per_batch = whitening_pairs_per_batch(metric.n(), condition_mode);
    let mut fitted_rows = Vec::with_capacity(core.candidates.len());
    let batch_rows = fit_pool.install(|| {
        core.candidates
            .par_chunks(max_pairs_per_batch)
            .map(|candidates| {
                let mut designs = Vec::with_capacity(candidates.len());
                let mut design_build_times = Vec::with_capacity(candidates.len());
                for candidate in candidates {
                    let design_started = profile_timings.then(Instant::now);
                    let (background, candidate_q_b) =
                        build_background(&candidate.filled, condition_mode)?;
                    if candidate_q_b != q_b {
                        return Err(
                            "candidate core has inconsistent background design widths".to_string()
                        );
                    }
                    let phase = candidate
                        .filled
                        .z
                        .iter()
                        .map(|&value| value as f64)
                        .collect::<Vec<_>>();
                    designs.push(PairDesign { background, phase });
                    design_build_times.push(
                        design_started
                            .map(|started| started.elapsed())
                            .unwrap_or_default(),
                    );
                }

                let (whitened, batch_whitening_timings) =
                    whiten_pair_design_batch_profiled(&designs, q_b, metric, profile_timings)?;
                let whitening_share = batch_whitening_timings.share(candidates.len());
                let whiten_share = whitening_share.total;
                candidates
                    .par_iter()
                    .zip(whitened.into_par_iter())
                    .zip(design_build_times.into_par_iter())
                    .enumerate()
                    .map(|(candidate_index, ((candidate, design), design_build))| {
                        let fit_started = profile_timings.then(Instant::now);
                        let mut conditional_timings = ConditionalFitTimings::default();
                        let fit_result = if profile_timings {
                            fit_conditional_prepared_whitened_profiled(
                                prepared,
                                design,
                                q_b,
                                inference,
                                Some(q_b - 1),
                                &mut conditional_timings,
                            )
                        } else {
                            fit_conditional_prepared_whitened(
                                prepared,
                                design,
                                q_b,
                                inference,
                                Some(q_b - 1),
                            )
                        };
                        let fit = fit_result.map_err(|error| {
                            format!(
                                "fit pair {}:{}-{}:{}: {error}",
                                candidate.left.chrom,
                                candidate.left.pos,
                                candidate.right.chrom,
                                candidate.right.pos
                            )
                        })?;
                        let fit_elapsed = fit_started
                            .map(|started| started.elapsed())
                            .unwrap_or_default();
                        conditional_timings.joint_design_whiten += whiten_share;
                        let mut candidate_whitening = whitening_share;
                        if candidate_index == 0 {
                            candidate_whitening.rhs_columns_total =
                                batch_whitening_timings.rhs_columns_total;
                            candidate_whitening.rhs_columns_unique =
                                batch_whitening_timings.rhs_columns_unique;
                        }
                        finish_candidate_fit(
                            candidate,
                            fit,
                            design_build + whiten_share + fit_elapsed,
                            design_build,
                            conditional_timings,
                            candidate_whitening,
                            profile_timings,
                            dxd,
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()
    })?;
    fitted_rows.extend(batch_rows.into_iter().flatten());
    Ok(fitted_rows)
}

fn format_effect(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.4}")
    } else {
        "NA".to_string()
    }
}

fn format_pvalue(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.4e}")
    } else {
        "NA".to_string()
    }
}

fn sample_order_fingerprint(sample_ids: &[String]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for sample_id in sample_ids {
        for byte in sample_id.as_bytes().iter().copied().chain([0xff]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn build_summary(
    status: &str,
    error: Option<&str>,
    model: ModelKind,
    condition: ConditionMode,
    dxd: bool,
    phenotype: &str,
    source: Value,
    sample_ids: &[String],
    window_bp: u64,
    core_bp: u64,
    topk: usize,
    snp_maf: f64,
    snp_geno: f64,
    pair_qc: PairQcThresholds,
    alpha: f64,
    threads: usize,
    global_null: &crate::ct::null::GlobalNullFit,
    scan: Option<&ScanStats>,
    n_rows: usize,
    n_pwald1_testable: usize,
    n_pwald2_testable: usize,
    n_pwald_dd_testable: usize,
    elapsed_seconds: f64,
    timings: Option<Value>,
) -> Value {
    let m = scan.map(|stats| stats.eligible_hypotheses);
    let bonferroni = if status == "complete" {
        m.filter(|count| *count > 0)
            .map(|count| alpha / count as f64)
    } else {
        None
    };
    let p1_p2_bonferroni = bonferroni.map(|threshold| threshold / 2.0);
    let all_three_bonferroni = bonferroni.map(|threshold| threshold / 3.0);
    let mut summary = json!({
        "schema_version": 4,
        "status": status,
        "error": error,
        "model": match model { ModelKind::Lm => "lm", ModelKind::FvLmm => "fvlmm" },
        "condition": match condition { ConditionMode::AddDom => "add-dom", ConditionMode::Genotype => "genotype" },
        "phenotype": phenotype,
        "source": source,
        "n_samples": sample_ids.len(),
        "n_used": sample_ids.len(),
        "sample_order_fingerprint": sample_order_fingerprint(sample_ids),
        "parameters": {
            "window_bp": window_bp,
            "core_bp": core_bp,
            "topk": topk,
            "snp_maf": snp_maf,
            "snp_geno": snp_geno,
            "phase_minor_freq": pair_qc.min_phase_minor_freq,
            "pair_geno": pair_qc.max_unknown_rate,
            "min_cis": pair_qc.min_cis,
            "min_trans": pair_qc.min_trans,
            "alpha": alpha,
            "threads": threads,
            "screen_threads": threads,
            "conditional_fit_threads": threads
        },
        "global_null": {
            "fixed_effect_rank": global_null.fixed_rank,
            "sigma_g2": global_null.sigma_g2,
            "sigma_e2": global_null.sigma_e2,
            "no_grm_boundary": global_null.no_grm_boundary
        },
        "phase_model": {
            "version": 4,
            "parameterization": "eta*H + theta*Z",
            "H": "1 for cis or trans (double heterozygotes), 0 otherwise",
            "Z": "cis=1, trans=-1, other=0",
            "observed_phase_support_rule": "require n_cis > 0 and n_trans > 0 before imputation; otherwise skip as unidentifiable, with no one-degree fallback",
            "background": match condition {
                ConditionMode::AddDom => "g_i,g_j,h_i,h_j,H",
                ConditionMode::Genotype => "g_i,g_j,h_i,h_j,g_i*g_j,g_i*h_j,h_i*g_j,H"
            },
            "null_hypothesis_pwald1": "beta_cis = beta_trans (theta = 0), 1 degree of freedom",
            "pwald1": match model {
                ModelKind::Lm => "1-df Wald t test with residual degrees of freedom",
                ModelKind::FvLmm => "1-df Wald chi-square test under fixed global covariance",
            },
            "null_hypothesis_pwald2": "beta_cis = beta_trans = 0 (eta = theta = 0), 2 degrees of freedom",
            "pwald2": match model {
                ModelKind::Lm => "joint 2-df Wald F test with residual degrees of freedom",
                ModelKind::FvLmm => "joint 2-df Wald chi-square test under fixed global covariance",
            },
            "null_hypothesis_pwald_dd": "beta_dd = 0 in the reduced model with the configured non-H background and H, but no Z, 1 degree of freedom",
            "pwald_dd": match model {
                ModelKind::Lm => "1-df Wald t test in the reduced no-phase D×D model, with reduced-model residual degrees of freedom",
                ModelKind::FvLmm => "1-df Wald chi-square test under the fixed global covariance in the reduced no-phase D×D model",
            },
            "beta_cis": "eta + theta",
            "beta_trans": "eta - theta",
            "beta_dd": "coefficient of H in the reduced model with the configured non-H background and H, but no phase term Z",
            "main_effects": "beta_a1/beta_d1 and beta_a2/beta_d2 are conditional additive/dominance coefficients from the full pair model",
            "effect_reference": "deviations from the remaining background terms, not absolute group means",
            "unidentifiable_shared_effect": "if H is collinear with other fixed effects, beta_c, beta_t, and pwald2 are NA because the joint two-degree phase test is not identifiable; pwald1 can remain estimable if Z is identifiable"
        },
        "screening": {
            "method": match model {
                ModelKind::Lm => "joint_h_z_score",
                ModelKind::FvLmm => "ct_joint_h_z_rank"
            },
            "score_note": match model {
                ModelKind::Lm => "the fast screen ranks the joint H/Z score after global fixed-effect projection; it is a Top-K ranking score, not a calibrated p-value",
                ModelKind::FvLmm => "the fast screen ranks an uncalibrated joint H/Z score after approximate global fixed-effect projection; final conditional fits use the full fixed covariance"
            }
        },
        "scan": scan.map(|stats| json!({
            "sites_seen": stats.sites_seen,
            "snp_qc_pass": stats.snp_qc_pass,
            "snp_qc_rejected": stats.snp_qc_rejected,
            "pairs_enumerated": stats.pairs_enumerated,
            "pair_qc_rejected": stats.pair_qc_rejected,
            "no_compatible_donor": stats.no_compatible_donor,
            "eligible_hypotheses_M": stats.eligible_hypotheses,
            "nonfinite_screen_scores": stats.nonfinite_screen_scores,
            "cores_flushed": stats.cores_flushed,
            "rayon_workers_used": stats.screen_workers_used,
        })),
        "results": {
            "rows_written": n_rows,
            "testable_rows": n_pwald2_testable,
            "pwald1_testable_rows": n_pwald1_testable,
            "pwald2_testable_rows": n_pwald2_testable,
            "pwald_dd_testable_rows": n_pwald_dd_testable,
            "non_testable_rows": n_rows.saturating_sub(n_pwald2_testable)
        },
        "bonferroni_reference_threshold": bonferroni,
        "bonferroni_reference_thresholds": {
            "pwald1_per_test_family": bonferroni,
            "pwald2_per_test_family": bonferroni,
            "pwald_dd_per_test_family": bonferroni,
            "pwald1_pwald2_combined": p1_p2_bonferroni,
            "all_three_test_families_combined": all_three_bonferroni
        },
        "elapsed_seconds": elapsed_seconds,
        "calibration_note": "Screen scores are used only for Top-K ranking, so nominal conditional p-values are not calibrated within the selected set. Bonferroni references use the full eligible-pair count M (not the number of returned Top-K rows) and control family-wise error across all eligible pairs when the per-pair tests are calibrated. Use alpha/(2M) for pwald1+pwald2 jointly, or alpha/(3M) if pwald_dd is included as a third discovery family."
    });
    if dxd {
        summary["schema_version"] = json!(1);
        summary["scan_mode"] = json!("dxd");
        summary["phase_model"] = json!({
            "version": 1,
            "parameterization": "beta_dd * H",
            "H": "1 when both endpoint dosages are 1, regardless of phase; 0 otherwise",
            "background": "g_i,g_j,h_i,h_j",
            "n_double_het": "raw observed (1,1) count before missing-value imputation",
            "null_hypothesis": "beta_dd = 0, 1 degree of freedom",
            "pwald_dd": match model {
                ModelKind::Lm => "1-df Wald t test with residual degrees of freedom",
                ModelKind::FvLmm => "1-df Wald chi-square test under the fixed global covariance",
            },
            "pair_imputation": "modal unphased dosage pair across the 9 genotype categories; cis/trans labels are collapsed into (1,1)"
        });
        summary["screening"] = json!({
            "method": match model {
                ModelKind::Lm => "double_heterozygote_h_only_score",
                ModelKind::FvLmm => "fvlmm_double_heterozygote_h_only_rank",
            },
            "score_note": "Ranks H after global fixed-effect projection; Top-K ranking only, not a calibrated p-value",
            "phase_support_required": false
        });
        summary["parameters"]["phase_minor_freq"] = Value::Null;
        summary["parameters"]["min_cis"] = Value::Null;
        summary["parameters"]["min_trans"] = Value::Null;
        summary["results"]["testable_rows"] = json!(n_pwald_dd_testable);
        summary["results"]["non_testable_rows"] = json!(n_rows.saturating_sub(n_pwald_dd_testable));
        summary["bonferroni_reference_thresholds"]["pwald1_pwald2_combined"] = Value::Null;
        summary["bonferroni_reference_thresholds"]["all_three_test_families_combined"] =
            Value::Null;
        summary["calibration_note"] = json!("The H-only screen and per-core Top-K use the same phenotype, so returned pwald_dd values are subject to selection and are not calibrated discovery p-values. The Bonferroni reference uses all eligible pairs M, not only returned Top-K rows, and applies only if per-pair tests are calibrated.");
    }
    if let Some(timings) = timings {
        summary["timings"] = timings;
    }
    summary
}

#[allow(clippy::too_many_arguments)]
fn make_profile_timings(
    enabled: bool,
    total_before_json: Duration,
    input_prepare: Duration,
    global_null_fit: Duration,
    scan_with_callbacks: Duration,
    candidate_profile: &CandidateProfileStats,
    final_tsv_flush: Duration,
    scan: Option<&ScanStats>,
) -> Option<Value> {
    if !enabled {
        return None;
    }

    let quick_scan = scan_with_callbacks.saturating_sub(candidate_profile.callback_wall);
    let (
        scoring_context_prepare,
        source_next_scanned_site,
        snp_qc,
        packed_site_prepare,
        pair_prepare,
        screen_score,
        topk_update,
        pair_enumeration_bookkeeping,
        pair_screen_wall,
        candidate_materialize,
        candidates_materialized,
    ) = scan
        .map(|stats| {
            (
                stats.timings.scoring_context_prepare,
                stats.timings.source_next_scanned_site,
                stats.timings.snp_qc,
                stats.timings.packed_site_prepare,
                stats.timings.pair_prepare_qc_impute,
                stats.timings.screen_score,
                stats.timings.topk_update,
                stats.timings.pair_enumeration_bookkeeping,
                stats.timings.pair_screen_wall,
                stats.timings.candidate_materialize,
                stats.candidates_materialized,
            )
        })
        .unwrap_or_default();
    let measured_scan_wall_stages = scoring_context_prepare
        + source_next_scanned_site
        + snp_qc
        + packed_site_prepare
        + pair_screen_wall
        + candidate_materialize;
    let site_window_bookkeeping = quick_scan.saturating_sub(measured_scan_wall_stages);
    let other_scan_engine_wall = pair_screen_wall + site_window_bookkeeping;
    let whitening_rhs_total = candidate_profile.whitening_thread.rhs_columns_total;
    let whitening_rhs_unique = candidate_profile.whitening_thread.rhs_columns_unique;
    let whitening_rhs_duplicate_fraction = if whitening_rhs_total == 0 {
        0.0
    } else {
        1.0 - whitening_rhs_unique as f64 / whitening_rhs_total as f64
    };

    Some(json!({
        "profile_enabled_by": "JX_CT_PROFILE",
        "total_before_json_seconds": total_before_json.as_secs_f64(),
        "input_and_source_prepare_seconds": input_prepare.as_secs_f64(),
        "global_null_fit_seconds": global_null_fit.as_secs_f64(),
        "scan_with_candidate_callbacks_seconds": scan_with_callbacks.as_secs_f64(),
        "quick_scan_excluding_candidate_callbacks_seconds": quick_scan.as_secs_f64(),
        "candidates_materialized_after_topk": candidates_materialized,
        "candidate_count_profiled": candidate_profile.candidates,
        "candidate_callback_wall_seconds": candidate_profile.callback_wall.as_secs_f64(),
        "candidate_parallel_fit_wall_seconds": candidate_profile.parallel_fit_wall.as_secs_f64(),
        "candidate_sort_wall_seconds": candidate_profile.sort_wall.as_secs_f64(),
        "candidate_tsv_write_wall_seconds": candidate_profile.tsv_write_wall.as_secs_f64(),
        "candidate_core_flush_wall_seconds": candidate_profile.core_flush_wall.as_secs_f64(),
        "candidate_thread_stages": {
            "total_candidate_work_seconds": candidate_profile.candidate_work_thread.as_secs_f64(),
            "design_build_background_and_z_seconds": candidate_profile.design_build_thread.as_secs_f64(),
            "conditional_joint_design_whitening_seconds": candidate_profile.conditional_fit_thread.joint_design_whiten.as_secs_f64(),
            "conditional_joint_design_whitening_stages_seconds": {
                "rhs_columns_total": whitening_rhs_total,
                "rhs_columns_unique": whitening_rhs_unique,
                "rhs_duplicate_fraction": whitening_rhs_duplicate_fraction,
                "matrix_assembly": candidate_profile.whitening_thread.matrix_assembly.as_secs_f64(),
                "column_deduplication": candidate_profile.whitening_thread.column_deduplication.as_secs_f64(),
                "row_matrix_copy": candidate_profile.whitening_thread.row_matrix_copy.as_secs_f64(),
                "solve_or_scale": candidate_profile.whitening_thread.solve_or_scale.as_secs_f64(),
                "column_unpack": candidate_profile.whitening_thread.column_unpack.as_secs_f64(),
                "duplicate_restore_and_pair_pack": candidate_profile.whitening_thread.duplicate_restore_and_pair_pack.as_secs_f64(),
                "unattributed": candidate_profile.whitening_thread.total.saturating_sub(
                    candidate_profile.whitening_thread.matrix_assembly
                        + candidate_profile.whitening_thread.column_deduplication
                        + candidate_profile.whitening_thread.row_matrix_copy
                        + candidate_profile.whitening_thread.solve_or_scale
                        + candidate_profile.whitening_thread.column_unpack
                        + candidate_profile.whitening_thread.duplicate_restore_and_pair_pack
                ).as_secs_f64()
            },
            "conditional_nuisance_projection_and_basis_seconds": candidate_profile.conditional_fit_thread.nuisance_projection.as_secs_f64(),
            "conditional_phase_test_seconds": candidate_profile.conditional_fit_thread.phase_test.as_secs_f64(),
            "conditional_main_effect_estimation_seconds": candidate_profile.conditional_fit_thread.main_effects.as_secs_f64(),
            "conditional_phase_effects_and_wald_seconds": candidate_profile.conditional_fit_thread.phase_effects.as_secs_f64(),
            "row_format_seconds": candidate_profile.row_format_thread.as_secs_f64()
        },
        "final_tsv_flush_seconds": final_tsv_flush.as_secs_f64(),
        "quick_scan_stages": {
            "scoring_context_prepare_seconds": scoring_context_prepare.as_secs_f64(),
            "source_next_scanned_site_seconds": source_next_scanned_site.as_secs_f64(),
            "snp_qc_seconds": snp_qc.as_secs_f64(),
            "packed_site_prepare_seconds": packed_site_prepare.as_secs_f64(),
            "pair_prepare_qc_impute_seconds": pair_prepare.as_secs_f64(),
            "screen_score_seconds": screen_score.as_secs_f64(),
            "topk_update_seconds": topk_update.as_secs_f64(),
            "candidate_materialize_seconds": candidate_materialize.as_secs_f64(),
            "pair_enumeration_bookkeeping_seconds": pair_enumeration_bookkeeping.as_secs_f64(),
            "pair_screen_wall_seconds": pair_screen_wall.as_secs_f64(),
            "site_window_bookkeeping_seconds": site_window_bookkeeping.as_secs_f64(),
            "other_scan_engine_wall_seconds": other_scan_engine_wall.as_secs_f64(),
            "note": "source_next_scanned_site measures the source API call (for dual-parent BED it includes parent BED decoding and offspring bitset construction); pair preparation, scoring, TopK, and pair enumeration timings are summed worker time and can exceed wall time with multiple screen threads; pair_screen_wall measures elapsed time across all pair batches; site_window_bookkeeping is residual scan wall time outside the measured stages"
        },
        "candidate_timing_note": "candidate_thread_stages sum elapsed time across candidate tasks run in parallel and can exceed wall time; candidate_*_wall_seconds are independently measured wall-clock scopes and are not derived by subtracting thread time"
    }))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("create CT JSON {}: {e}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|e| format!("serialize CT JSON {}: {e}", path.display()))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .map_err(|e| format!("write CT JSON {}: {e}", path.display()))
}

fn matrix_values(array: &PyReadonlyArray2<'_, f64>) -> Vec<f64> {
    let view = array.as_array();
    let rows = view.shape()[0];
    let cols = view.shape()[1];
    let mut out = Vec::with_capacity(rows.saturating_mul(cols));
    for row in 0..rows {
        for col in 0..cols {
            out.push(view[[row, col]]);
        }
    }
    out
}

fn make_fixed_design(covariates: &[f64], n: usize, q_cov: usize) -> Vec<f64> {
    let mut design = Vec::with_capacity(n * (q_cov + 1));
    for row in 0..n {
        design.push(1.0);
        design.extend_from_slice(&covariates[row * q_cov..(row + 1) * q_cov]);
    }
    design
}

fn rust_error(error: String) -> PyErr {
    PyRuntimeError::new_err(error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    source_kind, sample_ids, y, covariates, output_prefix,
    vcf_path=None, hmp_path=None, father_prefix=None, mother_prefix=None, cross_path=None,
    grm=None, grm_ids=None, model="lm", phenotype="trait",
    window_bp=50_000, core_bp=50_000, topk=100, snp_maf=0.0, snp_geno=0.05,
    phase_minor_freq=0.0, pair_geno=0.05, min_cis=0, min_trans=0, alpha=0.05,
    threads=1, scan_mode="phase"
))]
pub(crate) fn ct_run<'py>(
    py: Python<'py>,
    source_kind: &str,
    sample_ids: Vec<String>,
    y: PyReadonlyArray1<'py, f64>,
    covariates: PyReadonlyArray2<'py, f64>,
    output_prefix: &str,
    vcf_path: Option<&str>,
    hmp_path: Option<&str>,
    father_prefix: Option<&str>,
    mother_prefix: Option<&str>,
    cross_path: Option<&str>,
    grm: Option<PyReadonlyArray2<'py, f64>>,
    grm_ids: Option<Vec<String>>,
    model: &str,
    phenotype: &str,
    window_bp: u64,
    core_bp: u64,
    topk: usize,
    snp_maf: f64,
    snp_geno: f64,
    phase_minor_freq: f64,
    pair_geno: f64,
    min_cis: usize,
    min_trans: usize,
    alpha: f64,
    threads: usize,
    scan_mode: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let started = Instant::now();
    let profile_timings = std::env::var_os("JX_CT_PROFILE").is_some();
    let model_kind = match model.trim().to_ascii_lowercase().as_str() {
        "lm" => ModelKind::Lm,
        "fvlmm" => ModelKind::FvLmm,
        _ => return Err(PyValueError::new_err("model must be 'lm' or 'fvlmm'")),
    };
    let dxd_mode = match scan_mode.trim().to_ascii_lowercase().as_str() {
        "phase" => false,
        "dxd" => true,
        _ => return Err(PyValueError::new_err("scan_mode must be 'phase' or 'dxd'")),
    };
    let condition_mode = ConditionMode::AddDom;
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(PyValueError::new_err(
            "alpha must be finite and between 0 and 1",
        ));
    }
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be positive"));
    }
    for (name, value, upper) in [
        ("phase_minor_freq", phase_minor_freq, 0.5),
        ("pair_geno", pair_geno, 1.0),
    ] {
        if !value.is_finite() || !(0.0..=upper).contains(&value) {
            return Err(PyValueError::new_err(format!(
                "{name} must be finite and between 0 and {upper}"
            )));
        }
    }
    validate_sample_ids(&sample_ids, &sample_ids).map_err(PyValueError::new_err)?;
    let y_values = y.as_array().iter().copied().collect::<Vec<_>>();
    let n = sample_ids.len();
    if y_values.len() != n {
        return Err(PyValueError::new_err(format!(
            "phenotype length mismatch: got {}, expected {n}",
            y_values.len()
        )));
    }
    let cov_view = covariates.as_array();
    let (cov_n, q_cov) = (cov_view.shape()[0], cov_view.shape()[1]);
    if cov_n != n {
        return Err(PyValueError::new_err(format!(
            "covariate row count mismatch: got {cov_n}, expected {n}"
        )));
    }
    let cov_values = matrix_values(&covariates);
    if cov_values
        .iter()
        .chain(&y_values)
        .any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "phenotype or covariates contain non-finite values; filter samples before CT",
        ));
    }
    let mut grm_values = None;
    match (model_kind, grm) {
        (ModelKind::Lm, Some(_)) => {
            return Err(PyValueError::new_err("LM does not accept a GRM"));
        }
        (ModelKind::Lm, None) => {
            if grm_ids.is_some() {
                return Err(PyValueError::new_err("GRM IDs were provided without a GRM"));
            }
        }
        (ModelKind::FvLmm, None) => {
            return Err(PyValueError::new_err(
                "FvLMM requires an aligned GRM matrix",
            ));
        }
        (ModelKind::FvLmm, Some(grm_array)) => {
            let ids = grm_ids
                .as_deref()
                .ok_or_else(|| PyValueError::new_err("FvLMM requires GRM sample IDs"))?;
            validate_sample_ids(&sample_ids, ids).map_err(PyValueError::new_err)?;
            let grm_view = grm_array.as_array();
            if grm_view.shape() != [n, n] {
                return Err(PyValueError::new_err(format!(
                    "GRM shape mismatch: got {:?}, expected ({n}, {n})",
                    grm_view.shape()
                )));
            }
            grm_values = Some(matrix_values(&grm_array));
        }
    }
    let mut source = OpenedSource::open(
        source_kind,
        &sample_ids,
        vcf_path,
        hmp_path,
        father_prefix,
        mother_prefix,
        cross_path,
    )
    .map_err(PyValueError::new_err)?;
    if source.sample_ids() != sample_ids {
        return Err(PyValueError::new_err(
            "genotype source sample IDs do not exactly match CT aligned sample order",
        ));
    }
    let input_prepare_duration = started.elapsed();
    let global_null_started = Instant::now();
    let global_null = fit_global_null(
        &y_values,
        &cov_values,
        q_cov,
        grm_values.as_deref(),
        model_kind,
        threads,
    )
    .map_err(rust_error)?;
    let global_null_duration = global_null_started.elapsed();
    let inference = match model_kind {
        ModelKind::Lm => InferenceKind::LinearModel,
        ModelKind::FvLmm => InferenceKind::FixedCovariance,
    };
    let design_c = make_fixed_design(&cov_values, n, q_cov);
    let q_c = q_cov + 1;
    let prepared_fit = prepare_conditional_fit(&y_values, &design_c, q_c, &global_null.metric)
        .map_err(rust_error)?;
    let fit_pool = ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .thread_name(|index| format!("ct-fit-{index}"))
        .build()
        .map_err(|error| rust_error(format!("create CT fit thread pool: {error}")))?;
    let scan_config = ScanConfig {
        window_bp,
        core_bp,
        topk,
        threads,
        snp_maf,
        snp_geno,
        profile_timings,
        pair_qc: PairQcThresholds {
            min_phase_minor_freq: phase_minor_freq,
            max_unknown_rate: pair_geno,
            min_cis,
            min_trans,
        },
    };
    let (tsv_path, json_path) = output_paths(output_prefix, dxd_mode).map_err(rust_error)?;
    let tsv_file = File::create(&tsv_path).map_err(|e| {
        PyRuntimeError::new_err(format!("create CT TSV {}: {e}", tsv_path.display()))
    })?;
    let mut writer = WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(BufWriter::new(tsv_file));
    if dxd_mode {
        write_dxd_tsv_header(&mut writer).map_err(rust_error)?;
    } else {
        write_tsv_header(&mut writer).map_err(rust_error)?;
    }

    let mut n_rows = 0usize;
    let mut n_pwald1_testable = 0usize;
    let mut n_pwald2_testable = 0usize;
    let mut n_pwald_dd_testable = 0usize;
    let mut candidate_profile = CandidateProfileStats::default();
    let scan_started = Instant::now();
    let scan_result = {
        let mut process_batch = |cores: &[CoreCandidates]| {
            let callback_started = profile_timings.then(Instant::now);
            let callback_result = (|| -> Result<(), String> {
                let parallel_fit_started = profile_timings.then(Instant::now);
                let fitted_cores_result = fit_candidate_cores(
                    cores,
                    &prepared_fit,
                    condition_mode,
                    dxd_mode,
                    &global_null.metric,
                    inference,
                    profile_timings,
                    &fit_pool,
                );
                if let Some(started) = parallel_fit_started {
                    candidate_profile.parallel_fit_wall += started.elapsed();
                }
                for fitted_rows in fitted_cores_result? {
                    for (_, timing) in &fitted_rows {
                        candidate_profile.candidates += 1;
                        candidate_profile.candidate_work_thread += timing.elapsed;
                        candidate_profile.design_build_thread += timing.design_build;
                        candidate_profile.conditional_fit_thread.joint_design_whiten +=
                            timing.conditional_fit.joint_design_whiten;
                        candidate_profile.conditional_fit_thread.nuisance_projection +=
                            timing.conditional_fit.nuisance_projection;
                        candidate_profile.conditional_fit_thread.phase_test +=
                            timing.conditional_fit.phase_test;
                        candidate_profile.conditional_fit_thread.main_effects +=
                            timing.conditional_fit.main_effects;
                        candidate_profile.conditional_fit_thread.phase_effects +=
                            timing.conditional_fit.phase_effects;
                        candidate_profile
                            .whitening_thread
                            .add_assign(timing.whitening);
                        candidate_profile.row_format_thread += timing.row_format;
                    }
                    let sort_started = profile_timings.then(Instant::now);
                    let mut rows = fitted_rows
                        .into_iter()
                        .map(|(row, _)| row)
                        .collect::<Vec<_>>();
                    rows.sort_by(|left, right| {
                        right
                            .0
                            .total_cmp(&left.0)
                            .then_with(|| left.1.cmp(&right.1))
                    });
                    if let Some(started) = sort_started {
                        candidate_profile.sort_wall += started.elapsed();
                    }
                    let write_started = profile_timings.then(Instant::now);
                    for (_, _, fields, contrast_testable, joint_testable, dd_testable) in rows {
                        writer
                            .write_record(&fields)
                            .map_err(|error| format!("write CT TSV row: {error}"))?;
                        n_rows += 1;
                        n_pwald1_testable += usize::from(contrast_testable);
                        n_pwald2_testable += usize::from(joint_testable);
                        n_pwald_dd_testable += usize::from(dd_testable);
                    }
                    if let Some(started) = write_started {
                        candidate_profile.tsv_write_wall += started.elapsed();
                    }
                }
                let core_flush_started = profile_timings.then(Instant::now);
                writer
                    .flush()
                    .map_err(|error| format!("flush CT TSV core: {error}"))?;
                if let Some(started) = core_flush_started {
                    candidate_profile.core_flush_wall += started.elapsed();
                }
                Ok(())
            })();
            if let Some(callback_started) = callback_started {
                candidate_profile.callback_wall += callback_started.elapsed();
            }
            callback_result
        };
        // A single Top50 core provides only two 192-RHS whitening jobs in
        // add-dom mode. Accumulate consecutive cores so the same large solves
        // can occupy the fit pool, without changing core ownership or TopK.
        let batch_target = if global_null.metric.has_cholesky_factor() {
            conditional_batch_target(n, threads, condition_mode)
        } else {
            1
        };
        let max_pending_cores = threads.saturating_mul(2).max(1);
        let mut pending_cores = Vec::new();
        let mut pending_candidates = 0usize;
        let mut process_core = |core: CoreCandidates| {
            pending_candidates = pending_candidates.saturating_add(core.candidates.len());
            pending_cores.push(core);
            if pending_candidates >= batch_target || pending_cores.len() >= max_pending_cores {
                process_batch(&pending_cores)?;
                pending_cores.clear();
                pending_candidates = 0;
            }
            Ok(())
        };
        let result = match (model_kind, dxd_mode) {
            (ModelKind::Lm, false) => scan_source_lm(
                &mut source,
                scan_config,
                &global_null.screen_residuals,
                &global_null.screen_covariate_basis,
                &mut process_core,
            ),
            (ModelKind::Lm, true) => scan_source_dxd_lm(
                &mut source,
                scan_config,
                &global_null.screen_residuals,
                &global_null.screen_covariate_basis,
                &mut process_core,
            ),
            (ModelKind::FvLmm, false) => {
                let context = FvLmmScanContext {
                    fixed_effect_basis: &global_null.screen_covariate_basis,
                    sigma_e2: global_null
                        .sigma_e2
                        .expect("FvLMM global null returns sigma_e2"),
                };
                scan_source_fvlmm(
                    &mut source,
                    scan_config,
                    &global_null.screen_residuals,
                    &context,
                    &mut process_core,
                )
            }
            (ModelKind::FvLmm, true) => {
                let context = FvLmmScanContext {
                    fixed_effect_basis: &global_null.screen_covariate_basis,
                    sigma_e2: global_null
                        .sigma_e2
                        .expect("FvLMM global null returns sigma_e2"),
                };
                scan_source_dxd_fvlmm(
                    &mut source,
                    scan_config,
                    &global_null.screen_residuals,
                    &context,
                    &mut process_core,
                )
            }
        };
        drop(process_core);
        result.and_then(|stats| {
            if !pending_cores.is_empty() {
                process_batch(&pending_cores)?;
            }
            Ok(stats)
        })
    };
    let scan_duration = scan_started.elapsed();
    let final_flush_started = profile_timings.then(Instant::now);
    let scan_result = match writer.flush() {
        Ok(()) => scan_result,
        Err(error) => Err(format!("flush CT TSV: {error}")),
    };
    let final_tsv_flush_duration = final_flush_started
        .map(|flush_started| flush_started.elapsed())
        .unwrap_or_default();
    let source_summary = source.summary();
    match scan_result {
        Ok(scan_stats) => {
            let timings = make_profile_timings(
                profile_timings,
                started.elapsed(),
                input_prepare_duration,
                global_null_duration,
                scan_duration,
                &candidate_profile,
                final_tsv_flush_duration,
                Some(&scan_stats),
            );
            let summary = build_summary(
                "complete",
                None,
                model_kind,
                condition_mode,
                dxd_mode,
                phenotype,
                source_summary,
                &sample_ids,
                window_bp,
                core_bp,
                topk,
                snp_maf,
                snp_geno,
                scan_config.pair_qc,
                alpha,
                threads,
                &global_null,
                Some(&scan_stats),
                n_rows,
                n_pwald1_testable,
                n_pwald2_testable,
                n_pwald_dd_testable,
                started.elapsed().as_secs_f64(),
                timings,
            );
            write_json(&json_path, &summary).map_err(rust_error)?;
            let result = PyDict::new(py);
            result.set_item("status", "complete")?;
            result.set_item("tsv_path", tsv_path.to_string_lossy().as_ref())?;
            result.set_item("json_path", json_path.to_string_lossy().as_ref())?;
            result.set_item("M", scan_stats.eligible_hypotheses)?;
            result.set_item("rows_written", n_rows)?;
            result.set_item(
                "testable_rows",
                if dxd_mode {
                    n_pwald_dd_testable
                } else {
                    n_pwald2_testable
                },
            )?;
            result.set_item("pwald1_testable_rows", n_pwald1_testable)?;
            result.set_item("pwald2_testable_rows", n_pwald2_testable)?;
            result.set_item("pwald_dd_testable_rows", n_pwald_dd_testable)?;
            Ok(result)
        }
        Err(error) => {
            let timings = make_profile_timings(
                profile_timings,
                started.elapsed(),
                input_prepare_duration,
                global_null_duration,
                scan_duration,
                &candidate_profile,
                final_tsv_flush_duration,
                None,
            );
            let failed = build_summary(
                "failed",
                Some(&error),
                model_kind,
                condition_mode,
                dxd_mode,
                phenotype,
                source_summary,
                &sample_ids,
                window_bp,
                core_bp,
                topk,
                snp_maf,
                snp_geno,
                scan_config.pair_qc,
                alpha,
                threads,
                &global_null,
                None,
                n_rows,
                n_pwald1_testable,
                n_pwald2_testable,
                n_pwald_dd_testable,
                started.elapsed().as_secs_f64(),
                timings,
            );
            write_json(&json_path, &failed).map_err(rust_error)?;
            Err(PyRuntimeError::new_err(format!(
                "CT scan failed; partial TSV and failed JSON written: {error}"
            )))
        }
    }
}

pub(crate) fn register_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ct_run, module)?)
}

#[cfg(test)]
mod tests {
    use super::{
        build_summary, dxd_output_header, filled_category_counts, output_header,
        validate_sample_ids,
    };
    use crate::ct::conditional::CovarianceMetric;
    use crate::ct::null::{GlobalNullFit, ModelKind};
    use crate::ct::packed::FilledPair;

    #[test]
    fn conditional_batch_target_supplies_multiple_workers_with_bounded_sample_storage() {
        for condition in [super::ConditionMode::AddDom, super::ConditionMode::Genotype] {
            let pairs = super::whitening_pairs_per_batch(5831, condition);
            assert_eq!(
                super::conditional_batch_target(5831, 8, condition),
                pairs * 8
            );
            let target = super::conditional_batch_target(1_000_000, 64, condition);
            assert!(target > 0);
            assert!(
                target * (4_000_000 + std::mem::size_of::<super::ScreenCandidate>())
                    <= super::MAX_PENDING_CANDIDATE_BYTES
            );
        }
    }

    #[test]
    fn cross_core_fits_preserve_rows_empty_cores_and_partial_whitening_batches() {
        use crate::ct::conditional::{prepare_conditional_fit, ConditionMode, InferenceKind};
        use crate::ct::packed::PairCounts;
        use crate::ct::screen::{CoreCandidates, PairQc, ScreenCandidate, VariantMeta};

        let n = 80;
        let gi = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2];
        let gj = [0, 1, 2, 0, 1, 1, 2, 0, 1, 2];
        let phase = [0, 0, 0, 0, 1, -1, 0, 0, 0, 0];
        let y = (0..n)
            .map(|row| {
                4.0 + 0.2 * gi[row % 10] as f64
                    + 0.7 * phase[row % 10] as f64
                    + (row as f64 * 0.71).sin()
            })
            .collect::<Vec<_>>();
        let mut cores = Vec::new();
        let mut index = 0;
        for (core_index, count) in [4, 0, 36, 7].into_iter().enumerate() {
            let chrom = (core_index + 1).to_string();
            let mut candidates = Vec::new();
            for rank in 0..count {
                let left = VariantMeta {
                    index,
                    chrom: chrom.clone(),
                    pos: index as i64 + 1,
                    id: format!("left_{index}"),
                    allele0: "A".into(),
                    allele1: "C".into(),
                    snp_maf: 0.3,
                    snp_geno: 0.0,
                };
                let mut right = left.clone();
                right.pos += 100;
                right.id = format!("right_{index}");
                let raw_counts = PairCounts {
                    category_counts: [8; 10],
                    genotype_missing: 0,
                    phase_unknown: 0,
                    n_total: n,
                };
                candidates.push(ScreenCandidate {
                    left,
                    right,
                    distance_bp: 100,
                    pair_qc: PairQc::from_counts(&raw_counts),
                    raw_counts,
                    filled: FilledPair {
                        g_i: (0..n).map(|row| gi[(row + index) % 10]).collect(),
                        g_j: (0..n).map(|row| gj[(row + index) % 10]).collect(),
                        z: (0..n).map(|row| phase[(row + index) % 10]).collect(),
                        imputed: vec![false; n],
                        n_imputed: 0,
                    },
                    screen_score: rank as f64,
                    screen_rank: rank + 1,
                });
                index += 1;
            }
            cores.push(CoreCandidates {
                chrom,
                start: 1,
                end: 1000,
                candidates,
            });
        }
        let mut covariance = vec![0.0; n * n];
        for row in 0..n {
            covariance[row * n + row] = 1.2;
            if row > 0 {
                covariance[row * n + row - 1] = 0.1;
                covariance[(row - 1) * n + row] = 0.1;
            }
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (metric, inference) in [
            (CovarianceMetric::identity(n), InferenceKind::LinearModel),
            (
                CovarianceMetric::from_dense(&covariance, n).unwrap(),
                InferenceKind::FixedCovariance,
            ),
        ] {
            let prepared = prepare_conditional_fit(&y, &vec![1.0; n], 1, &metric).unwrap();
            for condition in [ConditionMode::AddDom, ConditionMode::Genotype] {
                let batched = super::fit_candidate_cores(
                    &cores, &prepared, condition, false, &metric, inference, false, &pool,
                )
                .unwrap();
                assert_eq!(
                    batched.iter().map(Vec::len).collect::<Vec<_>>(),
                    vec![4, 0, 36, 7]
                );
                for (core, rows) in cores.iter().zip(batched) {
                    let reference = super::fit_candidate_cores(
                        std::slice::from_ref(core),
                        &prepared,
                        condition,
                        false,
                        &metric,
                        inference,
                        false,
                        &pool,
                    )
                    .unwrap()
                    .pop()
                    .unwrap();
                    for (actual, expected) in rows.iter().zip(reference) {
                        assert_eq!(actual.0 .2, expected.0 .2);
                        assert_eq!(actual.0 .1, expected.0 .1);
                        assert!((actual.0 .0 - expected.0 .0).abs() < 1.0e-9);
                    }
                }
            }
        }
        assert!(super::fit_candidate_cores(
            &[],
            &prepare_conditional_fit(&y, &vec![1.0; n], 1, &CovarianceMetric::identity(n)).unwrap(),
            ConditionMode::AddDom,
            false,
            &CovarianceMetric::identity(n),
            InferenceKind::LinearModel,
            false,
            &pool,
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn output_header_contains_requested_minimal_model_statistics() {
        let header = output_header();
        for required in [
            "snp1", "snp2", "chrom1", "pos1", "chrom2", "pos2", "n_cis", "n_trans", "beta_a1",
            "beta_d1", "beta_a2", "beta_d2", "beta_c", "se_c", "beta_t", "se_t", "pwald1",
            "pwald2", "beta_dd", "se_dd", "pwald_dd",
        ] {
            assert!(
                header.contains(&required),
                "missing output column {required}"
            );
        }
        assert!(
            !header.contains(&"pwald"),
            "the retired pwald column must not be present"
        );
    }

    #[test]
    fn dxd_header_contains_only_unphased_pair_test_columns() {
        assert_eq!(
            dxd_output_header(),
            [
                "snp1",
                "snp2",
                "chrom1",
                "pos1",
                "chrom2",
                "pos2",
                "n_double_het",
                "beta_dd",
                "se_dd",
                "pwald_dd",
            ]
        );
    }

    #[test]
    fn empty_result_file_contains_a_tsv_header_and_no_rows() {
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(Vec::<u8>::new());
        super::write_tsv_header(&mut writer).unwrap();
        let bytes = writer.into_inner().unwrap();
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(bytes.as_slice());
        let header = reader.headers().unwrap().clone();
        assert_eq!(header.len(), output_header().len());
        assert_eq!(reader.records().count(), 0);
    }

    #[test]
    fn filled_counts_keep_other_distinct_from_cis_trans_and_imputed() {
        let filled = FilledPair {
            z: vec![1, -1, 0, 0, 1, 0],
            g_i: vec![1, 1, 0, 0, 1, 2],
            g_j: vec![1, 1, 2, 0, 1, 1],
            imputed: vec![false, false, false, true, true, true],
            n_imputed: 3,
        };
        let counts = filled_category_counts(&filled).unwrap();
        assert_eq!((counts.cis, counts.trans, counts.other), (2, 1, 3));
        assert_eq!(counts.n_imputed, 3);
        assert_eq!(counts.cis + counts.trans + counts.other, filled.z.len());
    }

    #[test]
    fn sample_ids_must_be_unique_nonempty_and_match_grm_order_exactly() {
        let target = vec!["c1".to_string(), "c2".to_string()];
        assert!(validate_sample_ids(&target, &target).is_ok());
        assert!(validate_sample_ids(&target, &["c2".into(), "c1".into()]).is_err());
        assert!(validate_sample_ids(&target, &["c1".into(), "c1".into()]).is_err());
        assert!(validate_sample_ids(&["".into()], &["".into()]).is_err());
        assert!(validate_sample_ids(&target, &[]).is_err());
    }

    #[test]
    fn summary_reports_the_requested_scan_and_fit_thread_counts() {
        let sample_ids = vec!["s0".to_string(), "s1".to_string(), "s2".to_string()];
        let global_null = GlobalNullFit {
            screen_residuals: vec![0.0; sample_ids.len()],
            fixed_rank: 1,
            sigma_g2: Some(0.25),
            sigma_e2: Some(0.75),
            no_grm_boundary: false,
            metric: CovarianceMetric::identity(sample_ids.len()),
            screen_covariate_basis: Vec::new(),
        };
        let summary = build_summary(
            "complete",
            None,
            ModelKind::FvLmm,
            super::ConditionMode::AddDom,
            false,
            "trait",
            serde_json::json!({}),
            &sample_ids,
            50_000,
            50_000,
            8,
            0.02,
            0.05,
            crate::ct::packed::PairQcThresholds {
                min_phase_minor_freq: 0.02,
                max_unknown_rate: 0.05,
                min_cis: 10,
                min_trans: 10,
            },
            0.05,
            4,
            &global_null,
            None,
            0,
            0,
            0,
            0,
            1.0,
            None,
        );

        assert_eq!(summary["schema_version"], 4);
        assert_eq!(
            summary["phase_model"]["null_hypothesis_pwald_dd"],
            "beta_dd = 0 in the reduced model with the configured non-H background and H, but no Z, 1 degree of freedom"
        );
        assert_eq!(
            summary["bonferroni_reference_thresholds"]["all_three_test_families_combined"],
            serde_json::Value::Null
        );
        assert_eq!(summary["parameters"]["screen_threads"], 4);
        assert_eq!(summary["parameters"]["conditional_fit_threads"], 4);
        assert_eq!(summary["results"]["pwald_dd_testable_rows"], 0);

        let dxd_summary = build_summary(
            "complete",
            None,
            ModelKind::FvLmm,
            super::ConditionMode::AddDom,
            true,
            "trait",
            serde_json::json!({}),
            &sample_ids,
            50_000,
            50_000,
            50,
            0.02,
            0.05,
            crate::ct::packed::PairQcThresholds {
                min_phase_minor_freq: 0.02,
                max_unknown_rate: 0.05,
                min_cis: 10,
                min_trans: 10,
            },
            0.05,
            12,
            &global_null,
            None,
            3,
            0,
            0,
            2,
            1.0,
            None,
        );
        assert_eq!(dxd_summary["schema_version"], 1);
        assert_eq!(dxd_summary["scan_mode"], "dxd");
        assert_eq!(dxd_summary["screening"]["phase_support_required"], false);
        assert_eq!(
            dxd_summary["parameters"]["phase_minor_freq"],
            serde_json::Value::Null
        );
        assert_eq!(dxd_summary["results"]["testable_rows"], 2);
    }
}
