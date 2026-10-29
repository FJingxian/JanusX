use crate::ct::input::{GenotypeCall, Site, SiteSource};
use crate::ct::packed::{
    materialize_encoded_pair_dxd_optional, materialize_encoded_pair_optional,
    prepare_pair_for_dxd_screen_with_scratch_optional,
    prepare_pair_for_screen_with_scratch_optional, PackedGenotypes, PairCounts,
    PairDxdScreenPreparation, PairQcThresholds, PairScreenPreparation, PairScreenScratch,
};
use crate::ct::screen::{
    double_dominance_rank_score_mask, joint_rank_score_phase_masks, passes_snp_qc,
    summarize_snp_packed, CoreCandidates, PairQc, PairRankContext, RankedCandidate,
    SnpQcThresholds, VariantMeta,
};
use rayon::prelude::*;
use rayon::ThreadPool;
use std::collections::{HashSet, VecDeque};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;
use std::time::{Duration, Instant};

const SOURCE_PREFETCH_MAX_BYTES: usize = 64 << 20;
const SOURCE_PREFETCH_MAX_SITES: usize = 16_384;
const LEFT_MARKERS_PER_SCAN_SHARD: usize = 16;
const MAX_SCAN_SHARDS_PER_CORE: usize = 4;

pub(crate) struct FvLmmScanContext<'a> {
    pub(crate) fixed_effect_basis: &'a [Vec<f64>],
    pub(crate) sigma_e2: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanConfig {
    pub(crate) window_bp: u64,
    pub(crate) core_bp: u64,
    pub(crate) topk: usize,
    pub(crate) threads: usize,
    pub(crate) snp_maf: f64,
    pub(crate) snp_geno: f64,
    pub(crate) pair_qc: PairQcThresholds,
    pub(crate) profile_timings: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ScanStageTimings {
    pub(crate) scoring_context_prepare: Duration,
    pub(crate) source_next_scanned_site: Duration,
    pub(crate) snp_qc: Duration,
    pub(crate) packed_site_prepare: Duration,
    pub(crate) pair_prepare_qc_impute: Duration,
    pub(crate) screen_score: Duration,
    pub(crate) topk_update: Duration,
    pub(crate) pair_enumeration_bookkeeping: Duration,
    pub(crate) pair_screen_wall: Duration,
    pub(crate) candidate_materialize: Duration,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ScanStats {
    pub(crate) sites_seen: usize,
    pub(crate) snp_qc_pass: usize,
    pub(crate) snp_qc_rejected: usize,
    pub(crate) pairs_enumerated: usize,
    pub(crate) pair_qc_rejected: usize,
    pub(crate) no_compatible_donor: usize,
    pub(crate) eligible_hypotheses: usize,
    pub(crate) nonfinite_screen_scores: usize,
    pub(crate) cores_flushed: usize,
    pub(crate) screen_workers_used: usize,
    pub(crate) candidates_materialized: usize,
    pub(crate) timings: ScanStageTimings,
}

struct BufferedSite {
    meta: VariantMeta,
    calls: Vec<GenotypeCall>,
    packed_calls: PackedGenotypes,
    core_start: i64,
}

struct CoreState {
    chrom: String,
    start: i64,
    end: i64,
    end_plus_window: u64,
    candidates: Vec<RankedCandidate>,
}

enum ScanScoring {
    Joint(PairRankContext),
    DoubleDominance(PairRankContext),
}

#[derive(Clone, Copy)]
enum ScanMode {
    Phase,
    DoubleDominance,
}

enum PairScorePreparation {
    QcRejected,
    NoCompatibleDonor,
    Ready {
        counts: PairCounts,
        score: Option<f64>,
    },
}

#[cfg(test)]
pub(crate) fn scan_source<S, F>(
    source: &mut S,
    config: ScanConfig,
    residuals: &[f64],
    on_core: F,
) -> Result<ScanStats, String>
where
    S: SiteSource,
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    scan_source_impl(
        source,
        config,
        residuals,
        &[],
        1.0,
        ScanMode::Phase,
        on_core,
    )
}

pub(crate) fn scan_source_lm<S, F>(
    source: &mut S,
    config: ScanConfig,
    residuals: &[f64],
    fixed_effect_basis: &[Vec<f64>],
    on_core: F,
) -> Result<ScanStats, String>
where
    S: SiteSource,
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    scan_source_impl(
        source,
        config,
        residuals,
        fixed_effect_basis,
        1.0,
        ScanMode::Phase,
        on_core,
    )
}

pub(crate) fn scan_source_dxd_lm<S, F>(
    source: &mut S,
    config: ScanConfig,
    residuals: &[f64],
    fixed_effect_basis: &[Vec<f64>],
    on_core: F,
) -> Result<ScanStats, String>
where
    S: SiteSource,
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    scan_source_impl(
        source,
        config,
        residuals,
        fixed_effect_basis,
        1.0,
        ScanMode::DoubleDominance,
        on_core,
    )
}

pub(crate) fn scan_source_fvlmm<S, F>(
    source: &mut S,
    config: ScanConfig,
    residuals: &[f64],
    context: &FvLmmScanContext<'_>,
    on_core: F,
) -> Result<ScanStats, String>
where
    S: SiteSource,
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    if !(context.sigma_e2.is_finite() && context.sigma_e2 > 0.0) {
        return Err("FvLMM rank scan requires positive finite sigma_e2".to_string());
    }
    scan_source_impl(
        source,
        config,
        residuals,
        context.fixed_effect_basis,
        context.sigma_e2,
        ScanMode::Phase,
        on_core,
    )
}

pub(crate) fn scan_source_dxd_fvlmm<S, F>(
    source: &mut S,
    config: ScanConfig,
    residuals: &[f64],
    context: &FvLmmScanContext<'_>,
    on_core: F,
) -> Result<ScanStats, String>
where
    S: SiteSource,
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    if !(context.sigma_e2.is_finite() && context.sigma_e2 > 0.0) {
        return Err("FvLMM D×D rank scan requires positive finite sigma_e2".to_string());
    }
    scan_source_impl(
        source,
        config,
        residuals,
        context.fixed_effect_basis,
        context.sigma_e2,
        ScanMode::DoubleDominance,
        on_core,
    )
}

fn scan_source_impl<S, F>(
    source: &mut S,
    config: ScanConfig,
    residuals: &[f64],
    fixed_effect_basis: &[Vec<f64>],
    sigma_e2: f64,
    mode: ScanMode,
    mut on_core: F,
) -> Result<ScanStats, String>
where
    S: SiteSource,
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    validate_config(config)?;
    let n_samples = source.sample_ids().len();
    if n_samples == 0 || residuals.len() != n_samples {
        return Err(format!(
            "screen residual/sample count mismatch: {} sample IDs, {} residuals",
            n_samples,
            residuals.len()
        ));
    }
    if residuals.iter().any(|value| !value.is_finite()) {
        return Err("screen residuals contain nonfinite values".to_string());
    }
    let snp_thresholds = SnpQcThresholds {
        min_maf: config.snp_maf,
        max_geno: config.snp_geno,
    };
    let mut stats = ScanStats::default();
    let scoring_context_started = config.profile_timings.then(Instant::now);
    let rank_context = PairRankContext::new(residuals, fixed_effect_basis, sigma_e2)?;
    let scoring = match mode {
        ScanMode::Phase => ScanScoring::Joint(rank_context),
        ScanMode::DoubleDominance => ScanScoring::DoubleDominance(rank_context),
    };
    if let Some(started) = scoring_context_started {
        stats.timings.scoring_context_prepare += started.elapsed();
    }

    if config.threads > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .thread_name(|index| format!("ct-screen-{index}"))
            .build()
            .map_err(|error| format!("create CT screening thread pool: {error}"))?;
        return scan_source_parallel(
            source,
            config,
            n_samples,
            snp_thresholds,
            scoring,
            pool,
            stats,
            on_core,
        );
    }

    let mut active_sites = VecDeque::<BufferedSite>::new();
    let mut cores = VecDeque::<CoreState>::new();
    let mut previous: Option<(String, i64)> = None;
    let mut seen_chromosomes = HashSet::new();
    let mut pair_scratch = PairScreenScratch::default();

    loop {
        let source_started = config.profile_timings.then(Instant::now);
        let scanned_site_result = source.next_scanned_site();
        if let Some(started) = source_started {
            stats.timings.source_next_scanned_site += started.elapsed();
        }
        let Some(scanned_site) = scanned_site_result? else {
            break;
        };
        let mut packed_calls = scanned_site.packed_calls;
        if packed_calls
            .as_ref()
            .is_some_and(|packed| packed.n_samples() != n_samples)
        {
            return Err("packed site sample count does not match source sample IDs".to_string());
        }
        let mut site = scanned_site.site;
        stats.sites_seen += 1;
        if site.pos <= 0 || site.chrom.trim().is_empty() {
            return Err(format!(
                "invalid site coordinate for '{}' at {}:{}",
                site.id, site.chrom, site.pos
            ));
        }
        if !site.calls.is_empty() && site.calls.len() != n_samples {
            return Err(format!(
                "site '{}' has {} calls but source has {n_samples} samples",
                site.id,
                site.calls.len()
            ));
        }
        if site.calls.is_empty() && packed_calls.is_none() {
            return Err(format!(
                "site '{}' has no calls or packed genotypes",
                site.id
            ));
        }
        let chromosome_changed = validate_site_order(&site, &mut previous, &mut seen_chromosomes)?;
        if chromosome_changed {
            flush_all(
                &mut cores,
                config.topk,
                config.profile_timings,
                &mut on_core,
                &mut stats,
            )?;
            active_sites.clear();
        }

        while active_sites
            .front()
            .is_some_and(|left| site.pos as u64 - left.meta.pos as u64 > config.window_bp)
        {
            active_sites.pop_front();
        }

        let qc_started = config.profile_timings.then(Instant::now);
        let qc = if site.calls.is_empty() {
            summarize_snp_packed(
                packed_calls
                    .as_ref()
                    .expect("packed-only site was checked above"),
            )?
        } else {
            super::screen::summarize_snp(&site.calls)?
        };
        if let Some(started) = qc_started {
            stats.timings.snp_qc += started.elapsed();
        }
        if !passes_snp_qc(&qc, snp_thresholds) {
            stats.snp_qc_rejected += 1;
            flush_ready(
                &mut cores,
                &site.chrom,
                site.pos as u64,
                config.topk,
                config.profile_timings,
                &mut on_core,
                &mut stats,
            )?;
            continue;
        }
        stats.snp_qc_pass += 1;
        if packed_calls.is_none() {
            let pack_started = config.profile_timings.then(Instant::now);
            packed_calls = Some(PackedGenotypes::from_calls(&site.calls));
            if let Some(started) = pack_started {
                stats.timings.packed_site_prepare += started.elapsed();
            }
        }
        let packed_calls = packed_calls.expect("site packing completed after SNP QC");
        let (core_start, core_end) = core_bounds(site.pos, config.core_bp)?;
        let core_end_plus_window = (core_end as u64).saturating_add(config.window_bp);
        ensure_core(
            &mut cores,
            &site.chrom,
            core_start,
            core_end,
            core_end_plus_window,
        );

        let current_meta = VariantMeta {
            index: site.index,
            chrom: site.chrom.clone(),
            pos: site.pos,
            id: site.id.clone(),
            allele0: site.allele0.clone(),
            allele1: site.allele1.clone(),
            snp_maf: qc.maf,
            snp_geno: qc.geno,
        };

        let pair_loop_started = config.profile_timings.then(Instant::now);
        let pair_prepare_before = stats.timings.pair_prepare_qc_impute;
        let screen_score_before = stats.timings.screen_score;
        let topk_update_before = stats.timings.topk_update;
        for left in &active_sites {
            if left.meta.chrom != site.chrom || left.meta.pos == site.pos {
                continue;
            }
            if left.meta.pos > site.pos {
                return Err(format!(
                    "site order moved backwards from {}:{} to {}:{}",
                    left.meta.chrom, left.meta.pos, site.chrom, site.pos
                ));
            }
            let distance = (site.pos - left.meta.pos) as u64;
            if distance > config.window_bp {
                continue;
            }
            stats.pairs_enumerated += 1;
            let prepare_started = config.profile_timings.then(Instant::now);
            let mut score_elapsed = Duration::default();
            let left_calls = (!left.calls.is_empty()).then_some(left.calls.as_slice());
            let right_calls = (!site.calls.is_empty()).then_some(site.calls.as_slice());
            let pair_preparation = prepare_and_score_pair(
                left_calls,
                right_calls,
                &left.packed_calls,
                &packed_calls,
                config.pair_qc,
                &scoring,
                config.profile_timings,
                &mut score_elapsed,
                &mut pair_scratch,
            )?;
            if let Some(started) = prepare_started {
                stats.timings.pair_prepare_qc_impute +=
                    started.elapsed().saturating_sub(score_elapsed);
            }
            stats.timings.screen_score += score_elapsed;
            match pair_preparation {
                PairScorePreparation::QcRejected => stats.pair_qc_rejected += 1,
                PairScorePreparation::NoCompatibleDonor => stats.no_compatible_donor += 1,
                PairScorePreparation::Ready { counts, score } => {
                    stats.eligible_hypotheses += 1;
                    let Some(screen_score) = score else {
                        stats.nonfinite_screen_scores += 1;
                        continue;
                    };
                    let core = cores
                        .iter_mut()
                        .find(|core| core.chrom == left.meta.chrom && core.start == left.core_start)
                        .ok_or_else(|| {
                            format!(
                                "internal scan error: core {}:{} was released before its window closed",
                                left.meta.chrom, left.core_start
                            )
                        })?;
                    let topk_started = config.profile_timings.then(Instant::now);
                    if let Some(insertion) = topk_insertion_index(
                        &core.candidates,
                        screen_score,
                        left.meta.index,
                        current_meta.index,
                        config.topk,
                    ) {
                        let pair = materialize_scan_pair(
                            left_calls,
                            right_calls,
                            &left.packed_calls,
                            &packed_calls,
                            &counts,
                            &scoring,
                        )
                        .ok_or_else(|| {
                            format!(
                                "screened pair {}:{}-{}:{} lost a compatible imputation donor",
                                left.meta.chrom,
                                left.meta.pos,
                                current_meta.chrom,
                                current_meta.pos
                            )
                        })?;
                        let candidate = RankedCandidate {
                            left: left.meta.clone(),
                            right: current_meta.clone(),
                            distance_bp: site.pos - left.meta.pos,
                            pair_qc: PairQc::from_counts(&counts),
                            raw_counts: counts,
                            encoded: pair,
                            screen_score,
                            screen_rank: 0,
                        };
                        core.candidates.insert(insertion, candidate);
                        core.candidates.truncate(config.topk);
                    }
                    if let Some(started) = topk_started {
                        stats.timings.topk_update += started.elapsed();
                    }
                }
            }
        }
        if let Some(started) = pair_loop_started {
            let pair_loop_elapsed = started.elapsed();
            stats.timings.pair_screen_wall += pair_loop_elapsed;
            let measured_pair_work = stats
                .timings
                .pair_prepare_qc_impute
                .saturating_sub(pair_prepare_before)
                + stats
                    .timings
                    .screen_score
                    .saturating_sub(screen_score_before)
                + stats.timings.topk_update.saturating_sub(topk_update_before);
            stats.timings.pair_enumeration_bookkeeping +=
                pair_loop_elapsed.saturating_sub(measured_pair_work);
        }

        active_sites.push_back(BufferedSite {
            meta: current_meta,
            calls: std::mem::take(&mut site.calls),
            packed_calls,
            core_start,
        });
        flush_ready(
            &mut cores,
            &site.chrom,
            site.pos as u64,
            config.topk,
            config.profile_timings,
            &mut on_core,
            &mut stats,
        )?;
    }
    flush_all(
        &mut cores,
        config.topk,
        config.profile_timings,
        &mut on_core,
        &mut stats,
    )?;
    Ok(stats)
}

fn scan_source_parallel<S, F>(
    source: &mut S,
    config: ScanConfig,
    n_samples: usize,
    snp_thresholds: SnpQcThresholds,
    scoring: ScanScoring,
    pool: ThreadPool,
    stats: ScanStats,
    on_core: F,
) -> Result<ScanStats, String>
where
    S: SiteSource,
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    let queue_capacity = source_prefetch_capacity(source.prefetch_site_bytes_hint());
    let (sender, receiver) = sync_channel(queue_capacity);
    thread::scope(|scope| {
        let producer = scope.spawn(move || loop {
            let started = Instant::now();
            let result = source.next_scanned_site();
            let decode_elapsed = started.elapsed();
            let finished = !matches!(&result, Ok(Some(_)));
            if sender
                .send(PrefetchedSite {
                    result,
                    decode_elapsed,
                })
                .is_err()
                || finished
            {
                break;
            }
        });
        let scan_result = scan_prefetched_source_parallel(
            &receiver,
            config,
            n_samples,
            snp_thresholds,
            scoring,
            pool,
            stats,
            on_core,
        );
        drop(receiver);
        if producer.join().is_err() {
            return Err("CT source prefetch worker panicked".to_string());
        }
        scan_result
    })
}

fn source_prefetch_capacity(site_bytes_hint: usize) -> usize {
    (SOURCE_PREFETCH_MAX_BYTES / site_bytes_hint.max(1)).clamp(1, SOURCE_PREFETCH_MAX_SITES)
}

struct PrefetchedSite {
    result: Result<Option<crate::ct::input::ScannedSite>, String>,
    decode_elapsed: Duration,
}

fn scan_prefetched_source_parallel<F>(
    receiver: &Receiver<PrefetchedSite>,
    config: ScanConfig,
    n_samples: usize,
    snp_thresholds: SnpQcThresholds,
    scoring: ScanScoring,
    pool: ThreadPool,
    mut stats: ScanStats,
    mut on_core: F,
) -> Result<ScanStats, String>
where
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    let mut pending_sites = Vec::<BufferedSite>::new();
    let mut pending_core_starts = Vec::<i64>::new();
    let mut previous: Option<(String, i64)> = None;
    let mut seen_chromosomes = HashSet::new();
    let mut workers_used = HashSet::new();

    loop {
        let received = receiver
            .recv()
            .map_err(|_| "CT source prefetch worker stopped unexpectedly".to_string())?;
        if config.profile_timings {
            stats.timings.source_next_scanned_site += received.decode_elapsed;
        }
        let scanned_site_result = received.result;
        let Some(scanned_site) = scanned_site_result? else {
            break;
        };
        let mut packed_calls = scanned_site.packed_calls;
        if packed_calls
            .as_ref()
            .is_some_and(|packed| packed.n_samples() != n_samples)
        {
            return Err("packed site sample count does not match source sample IDs".to_string());
        }
        let mut site = scanned_site.site;
        stats.sites_seen += 1;
        if site.pos <= 0 || site.chrom.trim().is_empty() {
            return Err(format!(
                "invalid site coordinate for '{}' at {}:{}",
                site.id, site.chrom, site.pos
            ));
        }
        if !site.calls.is_empty() && site.calls.len() != n_samples {
            return Err(format!(
                "site '{}' has {} calls but source has {n_samples} samples",
                site.id,
                site.calls.len()
            ));
        }
        if site.calls.is_empty() && packed_calls.is_none() {
            return Err(format!(
                "site '{}' has no calls or packed genotypes",
                site.id
            ));
        }
        let chromosome_changed = validate_site_order(&site, &mut previous, &mut seen_chromosomes)?;
        if chromosome_changed {
            flush_parallel_batches(
                &mut pending_sites,
                &mut pending_core_starts,
                true,
                0,
                config,
                &scoring,
                &pool,
                &mut workers_used,
                &mut on_core,
                &mut stats,
            )?;
        }

        let qc_started = config.profile_timings.then(Instant::now);
        let qc = if site.calls.is_empty() {
            summarize_snp_packed(
                packed_calls
                    .as_ref()
                    .expect("packed-only site was checked above"),
            )?
        } else {
            super::screen::summarize_snp(&site.calls)?
        };
        if let Some(started) = qc_started {
            stats.timings.snp_qc += started.elapsed();
        }
        if !passes_snp_qc(&qc, snp_thresholds) {
            stats.snp_qc_rejected += 1;
        } else {
            stats.snp_qc_pass += 1;
            if packed_calls.is_none() {
                let pack_started = config.profile_timings.then(Instant::now);
                packed_calls = Some(PackedGenotypes::from_calls(&site.calls));
                if let Some(started) = pack_started {
                    stats.timings.packed_site_prepare += started.elapsed();
                }
            }
            let packed_calls = packed_calls.expect("site packing completed after SNP QC");
            let (core_start, _) = core_bounds(site.pos, config.core_bp)?;
            let current_meta = VariantMeta {
                index: site.index,
                chrom: site.chrom.clone(),
                pos: site.pos,
                id: site.id.clone(),
                allele0: site.allele0.clone(),
                allele1: site.allele1.clone(),
                snp_maf: qc.maf,
                snp_geno: qc.geno,
            };
            if pending_core_starts.last().copied() != Some(core_start) {
                pending_core_starts.push(core_start);
            }
            pending_sites.push(BufferedSite {
                meta: current_meta,
                calls: std::mem::take(&mut site.calls),
                packed_calls,
                core_start,
            });
        }

        flush_parallel_batches(
            &mut pending_sites,
            &mut pending_core_starts,
            false,
            site.pos as u64,
            config,
            &scoring,
            &pool,
            &mut workers_used,
            &mut on_core,
            &mut stats,
        )?;
    }

    flush_parallel_batches(
        &mut pending_sites,
        &mut pending_core_starts,
        true,
        0,
        config,
        &scoring,
        &pool,
        &mut workers_used,
        &mut on_core,
        &mut stats,
    )?;
    stats.screen_workers_used = workers_used.len();
    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
fn flush_parallel_batches<F>(
    pending_sites: &mut Vec<BufferedSite>,
    pending_core_starts: &mut Vec<i64>,
    drain_all: bool,
    current_pos: u64,
    config: ScanConfig,
    scoring: &ScanScoring,
    pool: &ThreadPool,
    workers_used: &mut HashSet<usize>,
    on_core: &mut F,
    stats: &mut ScanStats,
) -> Result<(), String>
where
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    // Keep four core waves available so uneven marker density does not leave
    // Rayon workers idle while a batch waits for its slowest core.
    let batch_core_limit = config.threads.saturating_mul(4).max(1);
    loop {
        let target_count = if drain_all {
            pending_core_starts.len().min(batch_core_limit)
        } else if pending_core_starts.len() > batch_core_limit {
            batch_core_limit
        } else {
            break;
        };
        if target_count == 0 {
            break;
        }
        let last_target_start = pending_core_starts[target_count - 1];
        let (_, last_target_end) = core_bounds(last_target_start, config.core_bp)?;
        let end_plus_window = (last_target_end as u64).saturating_add(config.window_bp);
        // All right-hand sites for these left cores must be read before scoring.
        if !drain_all && current_pos <= end_plus_window {
            break;
        }

        let target_starts = pending_core_starts[..target_count].to_vec();
        let scan_shards = target_starts
            .iter()
            .map(|&start| core_scan_shards(start, pending_sites, config.threads))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let pair_batch_started = config.profile_timings.then(Instant::now);
        let mut results = pool.install(|| {
            scan_shards
                .par_iter()
                .map_init(PairScreenScratch::default, |scratch, &shard| {
                    scan_one_core_shard(shard, pending_sites, config, scoring, scratch)
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        if let Some(started) = pair_batch_started {
            stats.timings.pair_screen_wall += started.elapsed();
        }
        results.sort_by(|left, right| {
            left.core
                .chrom
                .cmp(&right.core.chrom)
                .then_with(|| left.core.start.cmp(&right.core.start))
                .then_with(|| left.left_range_start.cmp(&right.left_range_start))
        });

        let mut merged_results = Vec::<CoreScanResult>::new();
        for mut result in results {
            if let Some(worker_index) = result.worker_index {
                workers_used.insert(worker_index);
            }
            if let Some(previous) = merged_results.last_mut().filter(|previous| {
                previous.core.chrom == result.core.chrom && previous.core.start == result.core.start
            }) {
                previous.core.candidates.append(&mut result.core.candidates);
                previous
                    .scored_candidates
                    .append(&mut result.scored_candidates);
                add_pair_stats(&mut previous.stats, &result.stats);
            } else {
                merged_results.push(result);
            }
        }

        for mut result in merged_results {
            result.scored_candidates.sort_by(scored_candidate_order);
            result.scored_candidates.truncate(config.topk);
            let encode_started = config.profile_timings.then(Instant::now);
            for (rank, scored) in result.scored_candidates.drain(..).enumerate() {
                let left = &pending_sites[scored.left_site];
                let right = &pending_sites[scored.right_site];
                let left_calls = (!left.calls.is_empty()).then_some(left.calls.as_slice());
                let right_calls = (!right.calls.is_empty()).then_some(right.calls.as_slice());
                let encoded = materialize_scan_pair(
                    left_calls,
                    right_calls,
                    &left.packed_calls,
                    &right.packed_calls,
                    &scored.counts,
                    scoring,
                )
                .ok_or_else(|| {
                    format!(
                        "screened pair {}:{}-{}:{} lost a compatible imputation donor",
                        left.meta.chrom, left.meta.pos, right.meta.chrom, right.meta.pos
                    )
                })?;
                result.core.candidates.push(RankedCandidate {
                    left: left.meta.clone(),
                    right: right.meta.clone(),
                    distance_bp: right.meta.pos - left.meta.pos,
                    pair_qc: PairQc::from_counts(&scored.counts),
                    raw_counts: scored.counts,
                    encoded,
                    screen_score: scored.screen_score,
                    screen_rank: rank + 1,
                });
            }
            if let Some(started) = encode_started {
                result.stats.timings.candidate_materialize += started.elapsed();
            }
            add_pair_stats(stats, &result.stats);
            let mut cores = VecDeque::from([result.core]);
            flush_one(
                &mut cores,
                config.topk,
                config.profile_timings,
                on_core,
                stats,
            )?;
        }

        // Pair ownership follows the left core; later-core sites are retained as the next batch's halo.
        let first_future_site =
            pending_sites.partition_point(|site| site.core_start <= last_target_start);
        pending_sites.drain(..first_future_site);
        pending_core_starts.drain(..target_count);
    }
    stats.screen_workers_used = workers_used.len();
    Ok(())
}

struct CoreScanResult {
    core: CoreState,
    scored_candidates: Vec<ScoredCandidate>,
    stats: ScanStats,
    worker_index: Option<usize>,
    left_range_start: usize,
}

struct ScoredCandidate {
    left_site: usize,
    right_site: usize,
    left_index: usize,
    right_index: usize,
    counts: PairCounts,
    screen_score: f64,
}

#[derive(Clone, Copy)]
struct CoreScanShard {
    core_start: i64,
    left_start: usize,
    left_end: usize,
}

fn core_scan_shards(
    core_start: i64,
    sites: &[BufferedSite],
    thread_count: usize,
) -> Result<Vec<CoreScanShard>, String> {
    let first_left = sites.partition_point(|site| site.core_start < core_start);
    let left_end = sites.partition_point(|site| site.core_start <= core_start);
    if first_left >= left_end {
        return Err(format!("internal scan error: empty core {core_start}"));
    }
    let n_left = left_end - first_left;
    let max_shards = thread_count.min(MAX_SCAN_SHARDS_PER_CORE).max(1);
    let n_shards = n_left
        .div_ceil(LEFT_MARKERS_PER_SCAN_SHARD)
        .clamp(1, max_shards);
    let chunk_size = n_left.div_ceil(n_shards);
    let mut shards = Vec::with_capacity(n_shards);
    let mut start = first_left;
    while start < left_end {
        let end = (start + chunk_size).min(left_end);
        shards.push(CoreScanShard {
            core_start,
            left_start: start,
            left_end: end,
        });
        start = end;
    }
    Ok(shards)
}

fn scan_one_core_shard(
    shard: CoreScanShard,
    sites: &[BufferedSite],
    config: ScanConfig,
    scoring: &ScanScoring,
    pair_scratch: &mut PairScreenScratch,
) -> Result<CoreScanResult, String> {
    let worker_index = rayon::current_thread_index();
    let first_left = shard.left_start;
    let left_end = shard.left_end;
    let core_start = shard.core_start;
    let chrom = sites[first_left].meta.chrom.clone();
    let (start, end) = core_bounds(core_start, config.core_bp)?;
    let core = CoreState {
        chrom,
        start,
        end,
        end_plus_window: (end as u64).saturating_add(config.window_bp),
        candidates: Vec::new(),
    };
    let mut scored_candidates = Vec::<ScoredCandidate>::new();
    let mut stats = ScanStats::default();
    let pair_loop_started = config.profile_timings.then(Instant::now);
    for left_index in first_left..left_end {
        let left = &sites[left_index];
        for (right_index, right) in sites.iter().enumerate().skip(left_index + 1) {
            if right.meta.pos as u64 - left.meta.pos as u64 > config.window_bp {
                break;
            }
            if right.meta.pos == left.meta.pos {
                continue;
            }
            if right.meta.chrom != left.meta.chrom {
                continue;
            }
            stats.pairs_enumerated += 1;
            let prepare_started = config.profile_timings.then(Instant::now);
            let mut score_elapsed = Duration::default();
            let left_calls = (!left.calls.is_empty()).then_some(left.calls.as_slice());
            let right_calls = (!right.calls.is_empty()).then_some(right.calls.as_slice());
            let pair_preparation = prepare_and_score_pair(
                left_calls,
                right_calls,
                &left.packed_calls,
                &right.packed_calls,
                config.pair_qc,
                scoring,
                config.profile_timings,
                &mut score_elapsed,
                pair_scratch,
            )?;
            if let Some(started) = prepare_started {
                stats.timings.pair_prepare_qc_impute +=
                    started.elapsed().saturating_sub(score_elapsed);
            }
            stats.timings.screen_score += score_elapsed;
            match pair_preparation {
                PairScorePreparation::QcRejected => stats.pair_qc_rejected += 1,
                PairScorePreparation::NoCompatibleDonor => stats.no_compatible_donor += 1,
                PairScorePreparation::Ready { counts, score } => {
                    stats.eligible_hypotheses += 1;
                    let Some(screen_score) = score else {
                        stats.nonfinite_screen_scores += 1;
                        continue;
                    };
                    let topk_started = config.profile_timings.then(Instant::now);
                    if let Some(insertion) = scored_topk_insertion_index(
                        &scored_candidates,
                        screen_score,
                        left.meta.index,
                        right.meta.index,
                        config.topk,
                    ) {
                        scored_candidates.insert(
                            insertion,
                            ScoredCandidate {
                                left_site: left_index,
                                right_site: right_index,
                                left_index: left.meta.index,
                                right_index: right.meta.index,
                                counts,
                                screen_score,
                            },
                        );
                        scored_candidates.truncate(config.topk);
                    }
                    if let Some(started) = topk_started {
                        stats.timings.topk_update += started.elapsed();
                    }
                }
            }
        }
    }
    if let Some(started) = pair_loop_started {
        stats.timings.pair_enumeration_bookkeeping += started.elapsed().saturating_sub(
            stats.timings.pair_prepare_qc_impute
                + stats.timings.screen_score
                + stats.timings.topk_update,
        );
    }
    Ok(CoreScanResult {
        core,
        scored_candidates,
        stats,
        worker_index,
        left_range_start: shard.left_start,
    })
}

fn add_pair_stats(total: &mut ScanStats, source: &ScanStats) {
    total.pairs_enumerated += source.pairs_enumerated;
    total.pair_qc_rejected += source.pair_qc_rejected;
    total.no_compatible_donor += source.no_compatible_donor;
    total.eligible_hypotheses += source.eligible_hypotheses;
    total.nonfinite_screen_scores += source.nonfinite_screen_scores;
    total.timings.pair_prepare_qc_impute += source.timings.pair_prepare_qc_impute;
    total.timings.screen_score += source.timings.screen_score;
    total.timings.topk_update += source.timings.topk_update;
    total.timings.pair_enumeration_bookkeeping += source.timings.pair_enumeration_bookkeeping;
    total.timings.candidate_materialize += source.timings.candidate_materialize;
}

fn prepare_and_score_pair(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    thresholds: PairQcThresholds,
    scoring: &ScanScoring,
    profile_timings: bool,
    score_elapsed: &mut Duration,
    scratch: &mut PairScreenScratch,
) -> Result<PairScorePreparation, String> {
    match scoring {
        ScanScoring::Joint(context) => {
            let prepared = prepare_pair_for_screen_with_scratch_optional(
                left,
                right,
                left_packed,
                right_packed,
                thresholds,
                scratch,
            )?;
            match prepared {
                PairScreenPreparation::QcRejected => Ok(PairScorePreparation::QcRejected),
                PairScreenPreparation::NoCompatibleDonor => {
                    Ok(PairScorePreparation::NoCompatibleDonor)
                }
                PairScreenPreparation::Ready {
                    counts,
                    phase_masks,
                } => {
                    let started = profile_timings.then(Instant::now);
                    let score = joint_rank_score_phase_masks(&phase_masks, context);
                    if let Some(started) = started {
                        *score_elapsed += started.elapsed();
                    }
                    scratch.recycle_phase_masks(phase_masks);
                    Ok(PairScorePreparation::Ready { counts, score })
                }
            }
        }
        ScanScoring::DoubleDominance(context) => {
            let prepared = prepare_pair_for_dxd_screen_with_scratch_optional(
                left,
                right,
                left_packed,
                right_packed,
                thresholds,
                scratch,
            )?;
            match prepared {
                PairDxdScreenPreparation::QcRejected => Ok(PairScorePreparation::QcRejected),
                PairDxdScreenPreparation::NoCompatibleDonor => {
                    Ok(PairScorePreparation::NoCompatibleDonor)
                }
                PairDxdScreenPreparation::Ready { counts, masks } => {
                    let started = profile_timings.then(Instant::now);
                    let score = double_dominance_rank_score_mask(
                        left_packed.n_samples(),
                        masks.double_het_mask(),
                        masks.n_double_het(),
                        context,
                    );
                    if let Some(started) = started {
                        *score_elapsed += started.elapsed();
                    }
                    scratch.recycle_dxd_masks(masks);
                    Ok(PairScorePreparation::Ready { counts, score })
                }
            }
        }
    }
}

fn materialize_scan_pair(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    counts: &PairCounts,
    scoring: &ScanScoring,
) -> Option<crate::ct::packed::EncodedPair> {
    match scoring {
        ScanScoring::Joint(_) => {
            materialize_encoded_pair_optional(left, right, left_packed, right_packed, counts)
        }
        ScanScoring::DoubleDominance(_) => {
            materialize_encoded_pair_dxd_optional(left, right, left_packed, right_packed, counts)
        }
    }
}

fn validate_config(config: ScanConfig) -> Result<(), String> {
    if config.window_bp == 0 || config.core_bp == 0 || config.topk == 0 || config.threads == 0 {
        return Err("window-bp, core-bp, topk, and threads must be positive".to_string());
    }
    if !(config.snp_maf.is_finite() && (0.0..=0.5).contains(&config.snp_maf))
        || !(config.snp_geno.is_finite() && (0.0..=1.0).contains(&config.snp_geno))
        || !(config.pair_qc.min_phase_minor_freq.is_finite()
            && (0.0..=0.5).contains(&config.pair_qc.min_phase_minor_freq))
        || !(config.pair_qc.max_unknown_rate.is_finite()
            && (0.0..=1.0).contains(&config.pair_qc.max_unknown_rate))
    {
        return Err("CT QC thresholds are outside their valid ranges".to_string());
    }
    Ok(())
}

fn validate_site_order(
    site: &Site,
    previous: &mut Option<(String, i64)>,
    seen_chromosomes: &mut HashSet<String>,
) -> Result<bool, String> {
    let Some((previous_chrom, previous_pos)) = previous.as_ref() else {
        seen_chromosomes.insert(site.chrom.clone());
        *previous = Some((site.chrom.clone(), site.pos));
        return Ok(false);
    };
    let changed = previous_chrom != &site.chrom;
    if changed {
        if !seen_chromosomes.insert(site.chrom.clone()) {
            return Err(format!(
                "chromosome '{}' occurs in more than one source block",
                site.chrom
            ));
        }
    } else if site.pos < *previous_pos {
        return Err(format!(
            "positions must be nondecreasing within chromosome {}: {} after {}",
            site.chrom, site.pos, previous_pos
        ));
    }
    *previous = Some((site.chrom.clone(), site.pos));
    Ok(changed)
}

fn core_bounds(pos: i64, core_bp: u64) -> Result<(i64, i64), String> {
    let pos_u = u64::try_from(pos).map_err(|_| format!("invalid site position {pos}"))?;
    let start_u = ((pos_u - 1) / core_bp)
        .checked_mul(core_bp)
        .and_then(|x| x.checked_add(1))
        .ok_or_else(|| "core coordinate overflow".to_string())?;
    let end_u = start_u.saturating_add(core_bp - 1);
    let start = i64::try_from(start_u).map_err(|_| "core start exceeds i64".to_string())?;
    let end = i64::try_from(end_u.min(i64::MAX as u64)).unwrap_or(i64::MAX);
    Ok((start, end))
}

fn ensure_core(
    cores: &mut VecDeque<CoreState>,
    chrom: &str,
    start: i64,
    end: i64,
    end_plus_window: u64,
) {
    if cores
        .back()
        .is_some_and(|core| core.chrom == chrom && core.start == start)
    {
        return;
    }
    cores.push_back(CoreState {
        chrom: chrom.to_string(),
        start,
        end,
        end_plus_window,
        candidates: Vec::new(),
    });
}

fn topk_insertion_index(
    candidates: &[RankedCandidate],
    score: f64,
    left_index: usize,
    right_index: usize,
    topk: usize,
) -> Option<usize> {
    if topk == 0 {
        return None;
    }
    let insertion = candidates
        .iter()
        .position(|candidate| {
            score.total_cmp(&candidate.screen_score).is_gt()
                || (score.total_cmp(&candidate.screen_score).is_eq()
                    && (left_index, right_index) < (candidate.left.index, candidate.right.index))
        })
        .unwrap_or(candidates.len());
    if insertion >= topk && candidates.len() >= topk {
        None
    } else {
        Some(insertion)
    }
}

fn candidate_order(left: &RankedCandidate, right: &RankedCandidate) -> std::cmp::Ordering {
    right
        .screen_score
        .total_cmp(&left.screen_score)
        .then_with(|| left.left.index.cmp(&right.left.index))
        .then_with(|| left.right.index.cmp(&right.right.index))
}

fn scored_topk_insertion_index(
    candidates: &[ScoredCandidate],
    score: f64,
    left_index: usize,
    right_index: usize,
    topk: usize,
) -> Option<usize> {
    if topk == 0 {
        return None;
    }
    let insertion = candidates
        .iter()
        .position(|candidate| {
            score.total_cmp(&candidate.screen_score).is_gt()
                || (score.total_cmp(&candidate.screen_score).is_eq()
                    && (left_index, right_index) < (candidate.left_index, candidate.right_index))
        })
        .unwrap_or(candidates.len());
    if insertion >= topk && candidates.len() >= topk {
        None
    } else {
        Some(insertion)
    }
}

fn scored_candidate_order(left: &ScoredCandidate, right: &ScoredCandidate) -> std::cmp::Ordering {
    right
        .screen_score
        .total_cmp(&left.screen_score)
        .then_with(|| left.left_index.cmp(&right.left_index))
        .then_with(|| left.right_index.cmp(&right.right_index))
}

fn flush_ready<F>(
    cores: &mut VecDeque<CoreState>,
    chrom: &str,
    current_pos: u64,
    topk: usize,
    profile_timings: bool,
    on_core: &mut F,
    stats: &mut ScanStats,
) -> Result<(), String>
where
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    while cores
        .front()
        .is_some_and(|core| core.chrom != chrom || current_pos > core.end_plus_window)
    {
        flush_one(cores, topk, profile_timings, on_core, stats)?;
    }
    Ok(())
}

fn flush_all<F>(
    cores: &mut VecDeque<CoreState>,
    topk: usize,
    profile_timings: bool,
    on_core: &mut F,
    stats: &mut ScanStats,
) -> Result<(), String>
where
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    while !cores.is_empty() {
        flush_one(cores, topk, profile_timings, on_core, stats)?;
    }
    Ok(())
}

fn flush_one<F>(
    cores: &mut VecDeque<CoreState>,
    topk: usize,
    profile_timings: bool,
    on_core: &mut F,
    stats: &mut ScanStats,
) -> Result<(), String>
where
    F: FnMut(CoreCandidates) -> Result<(), String>,
{
    let mut core = cores
        .pop_front()
        .ok_or_else(|| "internal scan error: attempted to flush an empty core queue".to_string())?;
    core.candidates.sort_by(candidate_order);
    core.candidates.truncate(topk);
    for (rank, candidate) in core.candidates.iter_mut().enumerate() {
        candidate.screen_rank = rank + 1;
    }
    let materialize_started = profile_timings.then(Instant::now);
    let candidates: Vec<_> = core
        .candidates
        .into_iter()
        .map(RankedCandidate::materialize)
        .collect();
    if let Some(started) = materialize_started {
        stats.timings.candidate_materialize += started.elapsed();
    }
    stats.candidates_materialized += candidates.len();
    on_core(CoreCandidates {
        chrom: core.chrom,
        start: core.start,
        end: core.end,
        candidates,
    })?;
    stats.cores_flushed += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::scan_source_dxd_lm;
    use super::{scan_source, scan_source_fvlmm, FvLmmScanContext, ScanConfig};
    use crate::ct::input::{GenotypeCall, PhaseSet, Site, SiteSource};
    use crate::ct::packed::PairQcThresholds;
    use crate::ct::screen::CoreCandidates;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[derive(Clone)]
    struct VecSource {
        ids: Vec<String>,
        sites: Vec<Site>,
        cursor: usize,
    }

    impl SiteSource for VecSource {
        fn sample_ids(&self) -> &[String] {
            &self.ids
        }

        fn next_site(&mut self) -> Result<Option<Site>, String> {
            let Some(site) = self.sites.get(self.cursor).cloned() else {
                return Ok(None);
            };
            self.cursor += 1;
            Ok(Some(site))
        }
    }

    struct SlowVecSource {
        ids: Vec<String>,
        sites: Vec<Site>,
        cursor: usize,
        sites_returned: Arc<AtomicUsize>,
    }

    impl SiteSource for SlowVecSource {
        fn sample_ids(&self) -> &[String] {
            &self.ids
        }

        fn next_site(&mut self) -> Result<Option<Site>, String> {
            thread::sleep(Duration::from_millis(1));
            let Some(site) = self.sites.get(self.cursor).cloned() else {
                return Ok(None);
            };
            self.cursor += 1;
            self.sites_returned.fetch_add(1, Ordering::SeqCst);
            Ok(Some(site))
        }
    }

    fn site(index: usize, chrom: &str, pos: i64, phase: [bool; 4]) -> Site {
        Site {
            index,
            chrom: chrom.into(),
            pos,
            id: format!("s{index}"),
            allele0: "A".into(),
            allele1: "G".into(),
            calls: phase
                .into_iter()
                .map(|alt_on_hap0| GenotypeCall {
                    dosage: Some(1),
                    alt_on_hap0: Some(alt_on_hap0),
                    phase_set: PhaseSet::Implicit,
                })
                .collect(),
        }
    }

    fn genotype_site(
        index: usize,
        chrom: &str,
        pos: i64,
        heterozygous: [bool; 4],
        alt_on_hap0: [bool; 4],
    ) -> Site {
        Site {
            index,
            chrom: chrom.into(),
            pos,
            id: format!("s{index}"),
            allele0: "A".into(),
            allele1: "G".into(),
            calls: (0..4)
                .map(|sample| GenotypeCall {
                    dosage: Some(u8::from(heterozygous[sample])),
                    alt_on_hap0: heterozygous[sample].then_some(alt_on_hap0[sample]),
                    phase_set: if heterozygous[sample] {
                        PhaseSet::Implicit
                    } else {
                        PhaseSet::Unknown
                    },
                })
                .collect(),
        }
    }

    fn run(
        sites: Vec<Site>,
        config: ScanConfig,
        residuals: &[f64],
    ) -> (super::ScanStats, Vec<CoreCandidates>) {
        let mut source = VecSource {
            ids: (0..4).map(|i| format!("s{i}")).collect(),
            sites,
            cursor: 0,
        };
        let mut cores = Vec::new();
        let stats = scan_source(&mut source, config, residuals, |core| {
            cores.push(core);
            Ok(())
        })
        .unwrap();
        (stats, cores)
    }

    fn run_dxd(
        sites: Vec<Site>,
        config: ScanConfig,
        residuals: &[f64],
    ) -> (super::ScanStats, Vec<CoreCandidates>) {
        let mut source = VecSource {
            ids: (0..4).map(|i| format!("s{i}")).collect(),
            sites,
            cursor: 0,
        };
        let mut cores = Vec::new();
        let stats = scan_source_dxd_lm(&mut source, config, residuals, &[], |core| {
            cores.push(core);
            Ok(())
        })
        .unwrap();
        (stats, cores)
    }

    fn config(topk: usize, snp_maf: f64) -> ScanConfig {
        ScanConfig {
            window_bp: 3,
            core_bp: 2,
            topk,
            threads: 1,
            snp_maf,
            snp_geno: 0.0,
            pair_qc: PairQcThresholds {
                min_phase_minor_freq: 0.25,
                max_unknown_rate: 0.0,
                min_cis: 1,
                min_trans: 1,
            },
            profile_timings: false,
        }
    }

    #[test]
    fn dxd_screen_ranks_double_het_mask_without_phase_labels() {
        let sites = vec![
            genotype_site(0, "1", 1, [true, true, false, false], [true; 4]),
            genotype_site(
                1,
                "1",
                2,
                [true, true, false, false],
                [true, false, false, false],
            ),
            genotype_site(2, "1", 3, [false, false, true, true], [true; 4]),
            genotype_site(
                3,
                "1",
                4,
                [false, false, true, true],
                [false, false, true, false],
            ),
        ];
        let y = [1.0, 1.0, 2.0, -2.0];
        let mut scan_config = config(1, 0.0);
        scan_config.window_bp = 10;
        scan_config.core_bp = 10;

        let (_, ct_cores) = run(sites.clone(), scan_config, &y);
        let (_, dxd_cores) = run_dxd(sites, scan_config, &y);
        let ct_top = &ct_cores[0].candidates[0];
        let dxd_top = &dxd_cores[0].candidates[0];

        assert_eq!((ct_top.left.index, ct_top.right.index), (2, 3));
        assert_eq!((dxd_top.left.index, dxd_top.right.index), (0, 1));
        assert_eq!(dxd_top.pair_qc.n_cis_observed, 0);
        assert_eq!(dxd_top.pair_qc.n_trans_observed, 0);
        assert_eq!(dxd_top.pair_qc.phase_unknown, 2);
        assert!(dxd_top.filled.z.iter().all(|&value| value == 0));
    }

    #[test]
    fn dxd_parallel_scan_matches_serial_topk_across_core_windows() {
        let masks = [
            [true, true, false, false],
            [false, false, true, true],
            [true, false, true, false],
            [false, true, false, true],
        ];
        let sites = (0..160)
            .map(|index| {
                let mask = masks[index % masks.len()];
                let phase = [
                    index % 2 == 0,
                    index % 3 == 0,
                    index % 5 == 0,
                    index % 7 == 0,
                ];
                genotype_site(index, "1", index as i64 + 1, mask, phase)
            })
            .collect::<Vec<_>>();
        let residuals = [1.0, -1.0, 0.5, -0.5];
        let mut serial_config = config(5, 0.0);
        serial_config.core_bp = 20;
        serial_config.window_bp = 5;
        let (serial_stats, serial_cores) = run_dxd(sites.clone(), serial_config, &residuals);
        let mut parallel_config = serial_config;
        parallel_config.threads = 4;
        let (parallel_stats, parallel_cores) = run_dxd(sites, parallel_config, &residuals);

        assert_eq!(parallel_cores, serial_cores);
        assert_eq!(
            parallel_stats.pairs_enumerated,
            serial_stats.pairs_enumerated
        );
        assert_eq!(
            parallel_stats.eligible_hypotheses,
            serial_stats.eligible_hypotheses
        );
        assert_eq!(
            parallel_stats.nonfinite_screen_scores,
            serial_stats.nonfinite_screen_scores
        );
        assert_eq!(
            parallel_stats.candidates_materialized,
            serial_stats.candidates_materialized
        );
        assert!(parallel_stats.screen_workers_used > 0);
    }

    #[test]
    fn parallel_scan_matches_serial_across_batches_and_chromosomes() {
        let mut sites = Vec::new();
        let mut index = 0;
        for chrom in ["1", "2"] {
            for pos in 1..=160 {
                let phase = if pos % 2 == 0 {
                    [true, false, true, false]
                } else {
                    [true, true, false, false]
                };
                sites.push(site(index, chrom, pos, phase));
                index += 1;
                if pos % 13 == 0 {
                    sites.push(site(index, chrom, pos, [true, false, true, false]));
                    index += 1;
                }
                if pos % 17 == 0 {
                    let mut monomorphic = site(index, chrom, pos, [false; 4]);
                    for call in &mut monomorphic.calls {
                        call.dosage = Some(0);
                        call.alt_on_hap0 = None;
                        call.phase_set = PhaseSet::Unknown;
                    }
                    sites.push(monomorphic);
                    index += 1;
                }
            }
        }
        let residuals = [1.0, -1.0, 0.5, -0.5];
        let mut serial_config = config(3, 0.0);
        serial_config.core_bp = 20;
        let (serial_stats, serial_cores) = run(sites.clone(), serial_config, &residuals);
        let mut parallel_config = serial_config;
        parallel_config.threads = 4;
        let (parallel_stats, parallel_cores) = run(sites, parallel_config, &residuals);

        assert_eq!(parallel_cores, serial_cores);
        assert_eq!(parallel_stats.sites_seen, serial_stats.sites_seen);
        assert_eq!(parallel_stats.snp_qc_pass, serial_stats.snp_qc_pass);
        assert_eq!(parallel_stats.snp_qc_rejected, serial_stats.snp_qc_rejected);
        assert_eq!(
            parallel_stats.pairs_enumerated,
            serial_stats.pairs_enumerated
        );
        assert_eq!(
            parallel_stats.pair_qc_rejected,
            serial_stats.pair_qc_rejected
        );
        assert_eq!(
            parallel_stats.no_compatible_donor,
            serial_stats.no_compatible_donor
        );
        assert_eq!(
            parallel_stats.eligible_hypotheses,
            serial_stats.eligible_hypotheses
        );
        assert_eq!(
            parallel_stats.nonfinite_screen_scores,
            serial_stats.nonfinite_screen_scores
        );
        assert_eq!(parallel_stats.cores_flushed, serial_stats.cores_flushed);
        assert_eq!(
            parallel_stats.candidates_materialized,
            serial_stats.candidates_materialized
        );
        assert!(parallel_stats.cores_flushed > 8);
    }

    #[test]
    fn parallel_scan_splits_dense_core_and_preserves_exact_topk() {
        let sites = (0..128)
            .map(|index| {
                let phase = match index % 4 {
                    0 => [true, false, true, false],
                    1 => [false, true, false, true],
                    2 => [true, true, false, false],
                    _ => [false, false, true, true],
                };
                site(index, "1", index as i64 + 1, phase)
            })
            .collect::<Vec<_>>();
        let residuals = [1.0, -1.0, 0.5, -0.5];
        let mut serial_config = config(7, 0.0);
        serial_config.core_bp = 256;
        serial_config.window_bp = 256;
        let (serial_stats, serial_cores) = run(sites.clone(), serial_config, &residuals);
        let mut parallel_config = serial_config;
        parallel_config.threads = 4;
        let (parallel_stats, parallel_cores) = run(sites, parallel_config, &residuals);

        assert_eq!(parallel_cores, serial_cores);
        assert_eq!(
            parallel_stats.pairs_enumerated,
            serial_stats.pairs_enumerated
        );
        assert_eq!(
            parallel_stats.eligible_hypotheses,
            serial_stats.eligible_hypotheses
        );
        assert_eq!(
            parallel_stats.candidates_materialized,
            serial_stats.candidates_materialized
        );
        assert!(parallel_stats.screen_workers_used > 1);
    }

    #[test]
    fn parallel_scan_prefetches_sites_while_a_core_callback_runs() {
        let sites = (0..5_000)
            .map(|index| site(index, "1", index as i64 + 1, [true, false, true, false]))
            .collect();
        let sites_returned = Arc::new(AtomicUsize::new(0));
        let mut source = SlowVecSource {
            ids: (0..4).map(|i| format!("s{i}")).collect(),
            sites,
            cursor: 0,
            sites_returned: Arc::clone(&sites_returned),
        };
        let mut scan_config = config(1, 0.0);
        scan_config.core_bp = 1;
        scan_config.window_bp = 1;
        scan_config.threads = 2;
        let mut checked_overlap = false;
        scan_source(&mut source, scan_config, &[1.0, -1.0, 0.5, -0.5], |_| {
            if !checked_overlap {
                checked_overlap = true;
                let before = sites_returned.load(Ordering::SeqCst);
                thread::sleep(Duration::from_millis(200));
                let after = sites_returned.load(Ordering::SeqCst);
                assert!(
                    after > before + 80,
                    "the bounded source queue advanced from {before} to {after}, not enough to overlap the core batch"
                );
            }
            Ok(())
        })
        .unwrap();
        assert!(checked_overlap);
    }

    #[test]
    fn enumerates_window_pairs_once_by_left_core_and_keeps_exact_top_candidates() {
        let a = [true, true, false, false];
        let b = [true, false, true, false];
        let sites = vec![
            site(0, "1", 1, a),
            site(1, "1", 2, b),
            site(2, "1", 4, b),
            site(3, "1", 7, a),
            site(4, "2", 1, b),
        ];
        let (stats, cores) = run(sites.clone(), config(10, 0.0), &[1.0, 1.0, -1.0, -1.0]);
        assert_eq!(stats.sites_seen, 5);
        assert_eq!(stats.snp_qc_pass, 5);
        assert_eq!(stats.pairs_enumerated, 4);
        assert_eq!(stats.pair_qc_rejected, 1);
        assert_eq!(stats.eligible_hypotheses, 3);
        assert_eq!(stats.cores_flushed, 4);
        assert_eq!(
            cores
                .iter()
                .map(|core| (core.chrom.as_str(), core.start, core.end))
                .collect::<Vec<_>>(),
            vec![("1", 1, 2), ("1", 3, 4), ("1", 7, 8), ("2", 1, 2)]
        );
        let retained = cores
            .iter()
            .flat_map(|core| core.candidates.iter())
            .map(|candidate| (candidate.left.index, candidate.right.index))
            .collect::<Vec<_>>();
        // Independent nested-loop result: (0,1), (0,2), and (2,3) have both
        // cis and trans support; (1,2) is all cis and fails raw pair QC.
        assert_eq!(retained, vec![(0, 1), (0, 2), (2, 3)]);
        assert!(cores[0]
            .candidates
            .iter()
            .all(|candidate| candidate.screen_score == 0.0));
        assert_eq!(cores[0].candidates[0].screen_rank, 1);
        assert_eq!(cores[0].candidates[1].screen_rank, 2);
        assert_eq!(cores[0].candidates[0].filled.z.len(), 4);

        let (_, top_one) = run(sites.clone(), config(1, 0.0), &[1.0, 1.0, -1.0, -1.0]);
        assert_eq!(top_one[0].candidates.len(), 1);
        assert_eq!(
            (
                top_one[0].candidates[0].left.index,
                top_one[0].candidates[0].right.index
            ),
            (0, 1)
        );
        assert_eq!(
            top_one[0].candidates[0].filled,
            cores[0].candidates[0].filled
        );

        let (repeat_stats, repeat_cores) = run(sites, config(10, 0.0), &[1.0, 1.0, -1.0, -1.0]);
        assert_eq!(stats, repeat_stats);
        assert_eq!(
            retained,
            repeat_cores
                .iter()
                .flat_map(|core| core.candidates.iter())
                .map(|candidate| (candidate.left.index, candidate.right.index))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn same_position_and_cross_chromosome_sites_never_form_pairs() {
        let sites = vec![
            site(0, "1", 5, [true, true, false, false]),
            site(1, "1", 5, [true, false, true, false]),
            site(2, "2", 5, [true, false, true, false]),
        ];
        let (stats, cores) = run(sites, config(10, 0.0), &[0.0; 4]);
        assert_eq!(stats.pairs_enumerated, 0);
        assert_eq!(stats.eligible_hypotheses, 0);
        assert!(cores.iter().all(|core| core.candidates.is_empty()));
    }

    #[test]
    fn source_snp_qc_runs_before_any_pair_is_formed() {
        let mut low_maf = site(0, "1", 1, [false; 4]);
        for call in &mut low_maf.calls {
            call.dosage = Some(0);
            call.alt_on_hap0 = None;
            call.phase_set = PhaseSet::Unknown;
        }
        let high_maf = site(1, "1", 2, [true, false, true, false]);
        let (stats, _) = run(vec![low_maf, high_maf], config(10, 0.1), &[0.0; 4]);
        assert_eq!(stats.snp_qc_rejected, 1);
        assert_eq!(stats.snp_qc_pass, 1);
        assert_eq!(stats.pairs_enumerated, 0);
        assert_eq!(stats.eligible_hypotheses, 0);
    }

    #[test]
    fn fvlmm_screen_skips_phase_vectors_explained_by_fixed_effects() {
        let mut sites = Vec::new();
        for pair in 0..3 {
            let right_phase = if pair == 2 {
                [true, false, false, true]
            } else {
                [true, true, false, false]
            };
            let chrom = format!("chr{pair}");
            sites.push(site(pair * 2, &chrom, 1, [true; 4]));
            sites.push(site(pair * 2 + 1, &chrom, 2, right_phase));
        }
        let mut source = VecSource {
            ids: (0..4).map(|i| format!("s{i}")).collect(),
            sites: sites.clone(),
            cursor: 0,
        };
        let fixed_effect_basis = vec![vec![0.5; 4], vec![0.5, 0.5, -0.5, -0.5]];
        let context = FvLmmScanContext {
            fixed_effect_basis: &fixed_effect_basis,
            sigma_e2: 1.0,
        };

        let mut candidates = Vec::new();
        let stats = scan_source_fvlmm(&mut source, config(10, 0.0), &[0.0; 4], &context, |core| {
            candidates.extend(core.candidates);
            Ok(())
        })
        .unwrap();

        assert_eq!(stats.eligible_hypotheses, 3);
        assert_eq!(stats.nonfinite_screen_scores, 2);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].left.index, 4);

        let mut parallel_source = VecSource {
            ids: (0..4).map(|i| format!("s{i}")).collect(),
            sites,
            cursor: 0,
        };
        let mut parallel_config = config(10, 0.0);
        parallel_config.threads = 2;
        let mut parallel_candidates = Vec::new();
        let parallel_stats = scan_source_fvlmm(
            &mut parallel_source,
            parallel_config,
            &[0.0; 4],
            &context,
            |core| {
                parallel_candidates.extend(core.candidates);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(parallel_candidates, candidates);
        assert_eq!(
            parallel_stats.eligible_hypotheses,
            stats.eligible_hypotheses
        );
        assert_eq!(
            parallel_stats.nonfinite_screen_scores,
            stats.nonfinite_screen_scores
        );
        assert!(parallel_stats.screen_workers_used > 0);
    }

    #[test]
    fn topk_ties_break_by_source_index() {
        let a = [true, true, false, false];
        let b = [true, false, true, false];
        let (stats, cores) = run(
            vec![site(0, "1", 1, a), site(1, "1", 2, b), site(2, "1", 3, b)],
            config(1, 0.0),
            &[0.0; 4],
        );
        assert_eq!(stats.eligible_hypotheses, 2);
        let first = cores
            .iter()
            .find(|core| core.chrom == "1" && core.start == 1)
            .unwrap();
        assert_eq!(first.candidates.len(), 1);
        assert_eq!(
            (
                first.candidates[0].left.index,
                first.candidates[0].right.index
            ),
            (0, 1)
        );
    }
}
