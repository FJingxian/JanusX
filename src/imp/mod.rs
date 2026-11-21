//! Native Beagle 5.5-style phasing and imputation pipeline.
//!
//! This module is deliberately separate from JanusX's dosage-oriented
//! genotype readers. The `imp` path must preserve diploid allele order,
//! missingness, and whether each genotype was phased.

mod cluster_codes;
pub(crate) mod composite;
pub(crate) mod direct;
pub(crate) mod hmm;
pub(crate) mod ibd2;
mod ibs_mosaic;
pub(crate) mod impute;
pub(crate) mod memory;
pub(crate) mod pbwt;
pub(crate) mod phase;
pub(crate) mod phase_baum;
mod posterior_store;
mod projection;
mod reference_groups;
mod reference_store;
mod reference_window;
pub(crate) mod types;
pub(crate) mod window;

pub use types::{ImpConfig, ImpConversionSummary, ImpProgress, ImpSummary};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

struct ReferenceWindowStream {
    reader: crate::imp_reader::ImpReader,
    map: window::GeneticMap,
    samples: usize,
    pending: Option<(reference_store::CompactReferenceRow, f64, Option<u64>)>,
    carry: Vec<(types::PhasedRow, f64, Option<u64>)>,
    sequence_groups: reference_groups::ReferenceSequenceGroups,
    last_chromosome: Option<String>,
    last_pos: u64,
    last_cm: f64,
    seen_chromosomes: HashMap<String, ()>,
    last_emitted_target_pos: Option<u64>,
    last_emitted_chromosome: Option<String>,
    pending_carry_start_cm: Option<f64>,
    profile_decode_elapsed: Duration,
    profile_validate_elapsed: Duration,
    profile_map_elapsed: Duration,
}

enum ImpOutputWriter {
    Vcf(crate::imp_vcf_writer::ImpVcfWriter),
    Pgen(crate::pgen_writer::PgenWriter),
}

impl ImpOutputWriter {
    fn create(
        format: types::OutputFormat,
        path: &Path,
        samples: &[String],
        fields: types::VcfOutputFields,
        requested_workers: usize,
    ) -> Result<Self, String> {
        match format {
            types::OutputFormat::Vcf => Ok(Self::Vcf(
                crate::imp_vcf_writer::ImpVcfWriter::create_with_threads(
                    path,
                    samples,
                    fields,
                    requested_workers,
                )?,
            )),
            types::OutputFormat::Pgen => Ok(Self::Pgen(crate::pgen_writer::PgenWriter::create(
                path, samples,
            )?)),
        }
    }

    fn write_rows(
        &mut self,
        rows: &[&types::PhasedRow],
        worker_pool: &rayon::ThreadPool,
    ) -> Result<(), String> {
        match self {
            Self::Vcf(writer) => writer.write_rows(rows, worker_pool),
            Self::Pgen(writer) => {
                for row in rows {
                    writer.write_row(&row.meta, &row.calls, None)?;
                }
                Ok(())
            }
        }
    }

    fn finish(self, worker_pool: Option<&rayon::ThreadPool>) -> Result<(), String> {
        match self {
            Self::Vcf(writer) => writer.finish(worker_pool),
            Self::Pgen(writer) => writer.finish(),
        }
    }
}

enum PostPhaseJob {
    Impute {
        window: types::PhaseWindow,
        phased: types::PhasedWindow,
        reference_window: reference_store::CompactReferenceWindow,
        impute_config: types::ImputeConfig,
        fitted_hmm_config: types::HMMConfig,
        imputation_error: f64,
        seed: u64,
        previous_emitted_pos: Option<u64>,
        reference_emit_end_pos: Option<u64>,
        orientation_flips: Vec<bool>,
        region: Option<types::RegionSpec>,
        worker_pool: Arc<rayon::ThreadPool>,
    },
    Write {
        rows: Vec<types::PhasedRow>,
        indices: Vec<usize>,
        region: Option<types::RegionSpec>,
        worker_pool: Arc<rayon::ThreadPool>,
    },
}

struct PostPhaseRequest {
    job: PostPhaseJob,
    done: Option<mpsc::SyncSender<()>>,
}

struct PostPhasePipeline {
    path: PathBuf,
    sender: Option<mpsc::SyncSender<PostPhaseRequest>>,
    handle: Option<JoinHandle<Result<(), String>>>,
}

impl PostPhasePipeline {
    fn start(
        writer: ImpOutputWriter,
        path: &Path,
        completed_rows: Arc<AtomicU64>,
    ) -> Result<Self, String> {
        let path = path.to_path_buf();
        let (sender, receiver) = mpsc::sync_channel::<PostPhaseRequest>(1);
        let handle = std::thread::Builder::new()
            .name("imp-postphase".to_owned())
            .spawn(move || {
                let mut writer = writer;
                let mut last_pool = None;
                let profile = std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some()
                    || std::env::var_os("JANUSX_IMP_POSTPHASE_PROFILE").is_some()
                    || std::env::var_os("JANUSX_IMP_VCF_PROFILE").is_some();
                for request in receiver {
                    let PostPhaseRequest { job, done } = request;
                    let job_kind = match &job {
                        PostPhaseJob::Impute { .. } => "impute",
                        PostPhaseJob::Write { .. } => "write",
                    };
                    if let PostPhaseJob::Impute { worker_pool, .. } = &job {
                        last_pool = Some(Arc::clone(worker_pool));
                    } else if let PostPhaseJob::Write { worker_pool, .. } = &job {
                        last_pool = Some(Arc::clone(worker_pool));
                    }
                    let started = profile.then(Instant::now);
                    process_post_phase_job(&mut writer, job, &completed_rows)?;
                    if let Some(done) = done {
                        let _ = done.send(());
                    }
                    if let Some(started) = started {
                        eprintln!(
                            "JANUSX_IMP_POSTPHASE_PROFILE job={job_kind} elapsed_s={:.6}",
                            started.elapsed().as_secs_f64()
                        );
                    }
                }
                writer.finish(last_pool.as_deref())
            })
            .map_err(|err| format!("spawn IMP post-phase thread for {}: {err}", path.display()))?;
        Ok(Self {
            path,
            sender: Some(sender),
            handle: Some(handle),
        })
    }

    fn submit(&self, job: PostPhaseJob) -> Result<(), String> {
        self.sender
            .as_ref()
            .ok_or_else(|| format!("IMP post-phase pipeline closed for {}", self.path.display()))?
            .send(PostPhaseRequest { job, done: None })
            .map_err(|_| {
                format!(
                    "IMP post-phase pipeline stopped for {}",
                    self.path.display()
                )
            })
    }

    /// Submit a job and wait until the post-phase worker finishes and drops
    /// its owned window buffers, preventing two dense windows from coexisting
    /// in resident memory.
    fn submit_sync(&self, job: PostPhaseJob) -> Result<(), String> {
        let (done_tx, done_rx) = mpsc::sync_channel::<()>(0);
        self.sender
            .as_ref()
            .ok_or_else(|| format!("IMP post-phase pipeline closed for {}", self.path.display()))?
            .send(PostPhaseRequest {
                job,
                done: Some(done_tx),
            })
            .map_err(|_| {
                format!(
                    "IMP post-phase pipeline stopped for {}",
                    self.path.display()
                )
            })?;
        done_rx.recv().map_err(|_| {
            format!(
                "IMP post-phase pipeline stopped for {}",
                self.path.display()
            )
        })
    }

    fn finish(mut self) -> Result<(), String> {
        self.sender.take();
        join_post_phase(&self.path, self.handle.take())
    }
}

impl Drop for PostPhasePipeline {
    fn drop(&mut self) {
        self.sender.take();
        let _ = join_post_phase(&self.path, self.handle.take());
    }
}

fn join_post_phase(
    path: &Path,
    handle: Option<JoinHandle<Result<(), String>>>,
) -> Result<(), String> {
    match handle {
        Some(handle) => handle
            .join()
            .map_err(|_| format!("IMP post-phase thread panicked for {}", path.display()))?,
        None => Ok(()),
    }
}

fn process_post_phase_job(
    writer: &mut ImpOutputWriter,
    job: PostPhaseJob,
    completed_rows: &AtomicU64,
) -> Result<(), String> {
    match job {
        PostPhaseJob::Impute {
            window,
            phased,
            reference_window,
            impute_config,
            fitted_hmm_config,
            imputation_error,
            seed,
            previous_emitted_pos,
            reference_emit_end_pos,
            orientation_flips,
            region,
            worker_pool,
        } => {
            let mut imputation_hmm_config = fitted_hmm_config;
            imputation_hmm_config.err = imputation_error;
            let profile = std::env::var_os("JANUSX_IMP_VCF_PROFILE").is_some();
            let profile_window = format!("{}:{}", window.chrom, window.index);
            let job_started = profile.then(Instant::now);
            let impute_started = profile.then(Instant::now);
            let mut selection_elapsed = Duration::ZERO;
            let mut write_elapsed = Duration::ZERO;
            let mut emitted_rows = 0usize;
            impute::impute_window_stream_compact(
                &window,
                &phased,
                reference_window,
                &impute_config,
                &imputation_hmm_config,
                seed,
                previous_emitted_pos,
                reference_emit_end_pos,
                &orientation_flips,
                worker_pool.as_ref(),
                &mut |rows, indices| {
                    let started = profile.then(Instant::now);
                    let selected = indices
                        .iter()
                        .filter(|&&index| {
                            region_marker_is_emittable(
                                region.as_ref(),
                                &rows[index].meta.chrom,
                                rows[index].meta.pos,
                            )
                        })
                        .map(|&index| &rows[index])
                        .collect::<Vec<_>>();
                    selection_elapsed += started.map_or(Duration::ZERO, |s| s.elapsed());
                    let started = profile.then(Instant::now);
                    writer.write_rows(&selected, worker_pool.as_ref())?;
                    write_elapsed += started.map_or(Duration::ZERO, |s| s.elapsed());
                    emitted_rows += selected.len();
                    completed_rows.fetch_add(selected.len() as u64, Ordering::Relaxed);
                    Ok(())
                },
            )?;
            let impute_elapsed = impute_started
                .map_or(Duration::ZERO, |started| started.elapsed())
                .saturating_sub(selection_elapsed)
                .saturating_sub(write_elapsed);
            if let Some(started) = job_started {
                eprintln!(
                    "JANUSX_IMP_POSTPHASE_DETAIL_PROFILE job=impute window={} rows={} impute_s={:.6} selection_s={:.6} write_s={:.6} total_s={:.6}",
                    profile_window,
                    emitted_rows,
                    impute_elapsed.as_secs_f64(),
                    selection_elapsed.as_secs_f64(),
                    write_elapsed.as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                );
            }
            Ok(())
        }
        PostPhaseJob::Write {
            rows,
            indices,
            region,
            worker_pool,
        } => {
            let profile = std::env::var_os("JANUSX_IMP_VCF_PROFILE").is_some();
            let job_started = profile.then(Instant::now);
            let mut emitted = Vec::with_capacity(indices.len());
            for index in indices {
                let row = rows.get(index).ok_or_else(|| {
                    format!("output row index {index} exceeds {} rows", rows.len())
                })?;
                if region_marker_is_emittable(region.as_ref(), &row.meta.chrom, row.meta.pos) {
                    emitted.push(row);
                }
            }
            let selection_elapsed = job_started.map_or(Duration::ZERO, |started| started.elapsed());
            let write_started = profile.then(Instant::now);
            writer.write_rows(&emitted, worker_pool.as_ref())?;
            let write_elapsed = write_started.map_or(Duration::ZERO, |started| started.elapsed());
            completed_rows.fetch_add(emitted.len() as u64, Ordering::Relaxed);
            if let Some(started) = job_started {
                eprintln!(
                    "JANUSX_IMP_POSTPHASE_DETAIL_PROFILE job=write rows={} selection_s={:.6} write_s={:.6} total_s={:.6}",
                    emitted.len(),
                    selection_elapsed.as_secs_f64(),
                    write_elapsed.as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                );
            }
            Ok(())
        }
    }
}

impl ReferenceWindowStream {
    fn open(path: &Path, map: window::GeneticMap, workers: usize) -> Result<Self, String> {
        let lower = path.to_string_lossy().to_ascii_lowercase();
        let is_vcf = lower.ends_with(".vcf") || lower.ends_with(".vcf.gz");
        let is_pgen = crate::pgen::path_uses_pgen(&path.to_string_lossy());
        if !is_vcf && !is_pgen {
            return Err(format!(
                "reference input must be a phased VCF/VCF.GZ or PGEN/PVAR/PSAM dataset: {}",
                path.display()
            ));
        }
        let mut reader = crate::imp_reader::ImpReader::open(path)?;
        reader.set_vcf_parse_threads(workers)?;
        let samples = reader.sample_ids().len();
        if samples == 0 {
            return Err("reference input contains no samples".to_owned());
        }
        Ok(Self {
            reader,
            map,
            samples,
            pending: None,
            carry: Vec::new(),
            sequence_groups: reference_groups::ReferenceSequenceGroups::new(samples),
            last_chromosome: None,
            last_pos: 0,
            last_cm: f64::NEG_INFINITY,
            seen_chromosomes: HashMap::new(),
            last_emitted_target_pos: None,
            last_emitted_chromosome: None,
            pending_carry_start_cm: None,
            profile_decode_elapsed: Duration::ZERO,
            profile_validate_elapsed: Duration::ZERO,
            profile_map_elapsed: Duration::ZERO,
        })
    }

    fn window_for(
        &mut self,
        target: &types::PhaseWindow,
        max_markers: usize,
        config: &ImpConfig,
        terminal_target_window: bool,
    ) -> Result<(types::PhasedWindow, Vec<f64>, Vec<Option<u64>>), String> {
        let profile_stream = std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some();
        let profile_decode_start = self.profile_decode_elapsed;
        let profile_validate_start = self.profile_validate_elapsed;
        let profile_map_start = self.profile_map_elapsed;
        if self.last_emitted_chromosome.as_deref() != Some(target.chrom.as_str()) {
            self.last_emitted_target_pos = None;
        }
        let target_start_cm = *target
            .cm_positions
            .first()
            .ok_or_else(|| "target window has no genetic positions".to_owned())?;
        let target_end_cm = *target.cm_positions.last().unwrap();
        let first_target_pos = target.rows.first().unwrap().meta.pos;
        let last_target_pos = target.rows.last().unwrap().meta.pos;
        let region = config
            .region
            .as_ref()
            .filter(|region| region.contains_chromosome(&target.chrom));
        let first_target_window = self.last_emitted_target_pos.is_none();
        let include_chromosome_start = config.impute
            && first_target_window
            && region.map_or(true, |region| region.start.is_none());
        let include_chromosome_end = config.impute
            && terminal_target_window
            && region.map_or(true, |region| region.end.is_none());
        let output_region_start = if config.impute && first_target_window {
            region
                .and_then(|region| region.start)
                .filter(|&start| start < first_target_pos)
        } else {
            None
        };
        let output_region_end = if config.impute && terminal_target_window {
            region
                .and_then(|region| region.end)
                .filter(|&end| end > last_target_pos)
        } else {
            None
        };
        let start_cm = if include_chromosome_start {
            f64::NEG_INFINITY
        } else {
            match output_region_start {
                Some(start) => target_start_cm.min(self.map.cm_at(&target.chrom, start)?),
                None => target_start_cm,
            }
        };
        let end_cm = if include_chromosome_end {
            f64::INFINITY
        } else {
            match output_region_end {
                Some(end) => target_end_cm.max(self.map.cm_at(&target.chrom, end)?),
                None => target_end_cm,
            }
        };
        let previous_emitted_pos = self
            .last_emitted_chromosome
            .as_deref()
            .filter(|chrom| *chrom == target.chrom.as_str())
            .and(self.last_emitted_target_pos);
        if self
            .carry
            .first()
            .is_some_and(|(row, _, _)| row.meta.chrom != target.chrom)
        {
            self.carry.clear();
        }
        self.carry.retain(|(row, _, _)| {
            row.meta.chrom == target.chrom
                && previous_emitted_pos.map_or(true, |previous| row.meta.pos > previous)
        });
        let filter_to_target = !config.impute;
        let mut rows = std::mem::take(&mut self.carry);
        let mut saw_reference_in_window = !rows.is_empty();
        let mut fallback_row: Option<(reference_store::CompactReferenceRow, f64, Option<u64>)> =
            None;
        let mut merged_marker_count = target.rows.len();
        for (row, _, _) in &rows {
            if !target_contains_position(target, row.meta.pos) {
                merged_marker_count = merged_marker_count
                    .checked_add(1)
                    .ok_or_else(|| "merged marker memory estimate overflow".to_owned())?;
            }
        }
        if rows.len() > max_markers {
            return Err(format!(
                "reference window exceeds --window-markers={max_markers}; lower --window or use a less dense reference interval"
            ));
        }
        if merged_marker_count > max_markers {
            return Err(format!(
                "merged target/reference window exceeds --window-markers={max_markers}; reduce window size or split the reference input"
            ));
        }
        loop {
            let Some((row, cm, group)) = self.read_next()? else {
                break;
            };
            if row.meta.chrom != target.chrom {
                if !saw_reference_in_window {
                    // Reference-only contigs and contigs absent from the
                    // current target are consumed until the current contig is found.
                    continue;
                }
                self.pending = Some((row, cm, group));
                break;
            }
            if output_region_start.is_some_and(|start| row.meta.pos < start) {
                continue;
            }
            if output_region_end.is_some_and(|end| row.meta.pos > end) {
                self.pending = Some((row, cm, group));
                break;
            }
            if cm < start_cm {
                if previous_emitted_pos.is_some_and(|previous| row.meta.pos > previous) {
                    saw_reference_in_window = true;
                    let matches_target = target_contains_position(target, row.meta.pos);
                    if filter_to_target && !matches_target {
                        if fallback_row.is_none() && rows.is_empty() {
                            fallback_row = Some((row, cm, group));
                        }
                        continue;
                    }
                    if !matches_target {
                        merged_marker_count = merged_marker_count
                            .checked_add(1)
                            .ok_or_else(|| "merged marker memory estimate overflow".to_owned())?;
                    }
                    rows.push((row.to_phased_row(), cm, group));
                    if rows.len() > max_markers {
                        return Err(format!(
                            "reference window exceeds --window-markers={max_markers}; lower --window or use a less dense reference interval"
                        ));
                    }
                    if merged_marker_count > max_markers {
                        return Err(format!(
                            "merged target/reference window exceeds --window-markers={max_markers}; reduce window size or split the reference input"
                        ));
                    }
                }
                continue;
            }
            if cm > end_cm {
                self.pending = Some((row, cm, group));
                break;
            }
            saw_reference_in_window = true;
            let matches_target = target_contains_position(target, row.meta.pos);
            if filter_to_target && !matches_target {
                if fallback_row.is_none() && rows.is_empty() {
                    fallback_row = Some((row, cm, group));
                }
                continue;
            }
            if !matches_target {
                merged_marker_count = merged_marker_count
                    .checked_add(1)
                    .ok_or_else(|| "merged marker memory estimate overflow".to_owned())?;
            }
            rows.push((row.to_phased_row(), cm, group));
            if rows.len() > max_markers {
                return Err(format!(
                    "reference window exceeds --window-markers={max_markers}; lower --window or use a less dense reference interval"
                ));
            }
            if merged_marker_count > max_markers {
                return Err(format!(
                    "merged target/reference window exceeds --window-markers={max_markers}; reduce window size or split the reference input"
                ));
            }
        }
        if !saw_reference_in_window {
            return Err(format!(
                "reference input has no variants in target window {}:{}-{}; verify chromosome names and genetic map",
                target.chrom,
                target.rows.first().map_or(0, |row| row.meta.pos),
                target.rows.last().map_or(0, |row| row.meta.pos)
            ));
        }
        if rows.is_empty() {
            if let Some((row, cm, group)) = fallback_row {
                rows.push((row.to_phased_row(), cm, group));
            }
        }
        let finalize_started = profile_stream.then(Instant::now);
        let mut phased_rows = Vec::with_capacity(rows.len());
        let mut cm_positions = Vec::with_capacity(rows.len());
        let mut reference_groups = Vec::with_capacity(rows.len());
        for (row, cm, group) in rows {
            phased_rows.push(row);
            cm_positions.push(cm);
            reference_groups.push(group);
        }
        self.pending_carry_start_cm = target.cm_positions.get(target.emit_end).copied();
        let phased = types::PhasedWindow {
            chrom: target.chrom.clone(),
            index: target.index,
            emit_start: 0,
            emit_end: phased_rows.len(),
            rows: phased_rows,
        };
        if let Some(started) = finalize_started {
            eprintln!(
                "JANUSX_IMP_REFERENCE_STREAM_PROFILE window={}:{} decode_s={:.6} validate_s={:.6} map_s={:.6} finalize_s={:.6}",
                target.chrom,
                target.index,
                (self.profile_decode_elapsed - profile_decode_start).as_secs_f64(),
                (self.profile_validate_elapsed - profile_validate_start).as_secs_f64(),
                (self.profile_map_elapsed - profile_map_start).as_secs_f64(),
                started.elapsed().as_secs_f64(),
            );
        }
        // `read_next` validates every newly decoded row; rows carried across
        // windows were validated when first read, so a whole-window rescan
        // here would repeat the same genotype checks for each overlapping
        // window.
        Ok((phased, cm_positions, reference_groups))
    }

    fn cache_carry(
        &mut self,
        reference_window: &types::PhasedWindow,
        cm_positions: &[f64],
        reference_groups: &[Option<u64>],
    ) -> Result<(), String> {
        if reference_groups.len() != reference_window.rows.len() {
            return Err(
                "reference carry rows and sequence groups have different lengths".to_owned(),
            );
        }
        let carry_start = cm_positions
            .partition_point(|&cm| self.pending_carry_start_cm.map_or(true, |start| cm < start));
        self.carry = window::reference_carry_rows(
            reference_window,
            cm_positions,
            self.pending_carry_start_cm.take(),
        )?
        .into_iter()
        .zip(&reference_groups[carry_start..])
        .map(|((row, cm), &group)| (row, cm, group))
        .collect();
        Ok(())
    }

    fn read_next(
        &mut self,
    ) -> Result<Option<(reference_store::CompactReferenceRow, f64, Option<u64>)>, String> {
        if let Some(pending) = self.pending.take() {
            return Ok(Some(pending));
        }
        let profile_stream = std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some();
        let decode_started = profile_stream.then(Instant::now);
        let row = self.reader.next_biallelic_reference_row()?;
        if let Some(started) = decode_started {
            self.profile_decode_elapsed += started.elapsed();
        }
        let Some(row) = row else {
            return Ok(None);
        };
        if row.meta.alt_alleles.len() != 1 {
            return Err(format!(
                "reference input must be biallelic for imputation; {}:{} has {} ALT alleles",
                row.meta.chrom,
                row.meta.pos,
                row.meta.alt_alleles.len()
            ));
        }
        let validate_started = profile_stream.then(Instant::now);
        types::validate_biallelic_reference_calls(&row)?;
        if let Some(started) = validate_started {
            self.profile_validate_elapsed += started.elapsed();
        }
        let chrom = row.meta.chrom.clone();
        if let Some(previous_chrom) = self.last_chromosome.as_ref() {
            if chrom == *previous_chrom {
                if row.meta.pos <= self.last_pos {
                    return Err(format!(
                        "reference positions are not strictly increasing: {}:{} follows {}:{}",
                        chrom, row.meta.pos, previous_chrom, self.last_pos
                    ));
                }
            } else {
                self.seen_chromosomes.insert(previous_chrom.clone(), ());
                if self.seen_chromosomes.contains_key(&chrom) {
                    return Err(format!(
                        "reference chromosome {chrom} reappears after another chromosome"
                    ));
                }
                self.last_cm = f64::NEG_INFINITY;
            }
        }
        let map_started = profile_stream.then(Instant::now);
        let cm = self.map.cm_at(&chrom, row.meta.pos)?;
        if let Some(started) = map_started {
            self.profile_map_elapsed += started.elapsed();
        }
        if cm + f64::EPSILON < self.last_cm {
            return Err(format!(
                "reference genetic positions decrease at {}:{}",
                chrom, row.meta.pos
            ));
        }
        let group = self.sequence_groups.push(&chrom, &row.calls)?;
        let compact = reference_store::CompactReferenceRow::from_validated_variant_row(row)?;
        self.last_chromosome = Some(chrom);
        self.last_pos = compact.meta.pos;
        self.last_cm = cm;
        Ok(Some((compact, cm, group)))
    }
}

/// Run the phasing/imputation pipeline.
///
/// Stream target and optional reference genotypes through bounded windows.
pub fn run_imp(config: ImpConfig) -> Result<ImpSummary, String> {
    run_imp_with_progress(config, |_| Ok(()))
}

pub fn convert_to_pgen_with_progress<F>(
    input: &Path,
    output_prefix: &Path,
    mut progress: F,
) -> Result<ImpConversionSummary, String>
where
    F: FnMut(&ImpProgress) -> Result<(), String>,
{
    let mut reader = crate::imp_reader::ImpReader::open(input)?;
    let samples = reader.sample_ids().to_vec();
    let mut writer = crate::pgen_writer::PgenWriter::create(output_prefix, &samples)?;
    let mut variants = 0usize;
    let mut last_chrom = String::new();
    report_progress(&mut progress, "cache", "", 0, 0, 0, samples.len())?;
    while let Some(row) = reader.next_row()? {
        last_chrom.clone_from(&row.meta.chrom);
        writer.write_row(&row.meta, &row.calls, None)?;
        variants += 1;
        if variants % 10_000 == 0 {
            report_progress(
                &mut progress,
                "cache",
                &last_chrom,
                0,
                variants,
                variants as u64,
                samples.len(),
            )?;
        }
    }
    writer.finish()?;
    report_progress(
        &mut progress,
        "cache_done",
        &last_chrom,
        0,
        variants,
        variants as u64,
        samples.len(),
    )?;
    Ok(ImpConversionSummary {
        samples: samples.len(),
        variants,
    })
}

pub fn run_imp_with_progress<F>(config: ImpConfig, mut progress: F) -> Result<ImpSummary, String>
where
    F: FnMut(&ImpProgress) -> Result<(), String>,
{
    config.validate()?;
    validate_output_paths(&config)?;

    let mut reader = crate::imp_reader::ImpReader::open(&config.target)?;
    reader.set_vcf_parse_threads(config.threads)?;
    let samples = reader.sample_ids().to_vec();
    progress(&ImpProgress {
        stage: "prepare".to_owned(),
        chrom: String::new(),
        window_index: 0,
        markers: 0,
        variants_written: 0,
        samples: samples.len(),
    })?;
    let diagnostic_site: Option<types::ResolvedDiagnosticSite> = config
        .diagnostic_site
        .as_ref()
        .map(|site| {
            let sample_index = samples
                .iter()
                .position(|sample| sample == &site.sample)
                .ok_or_else(|| format!("trace-site sample {:?} was not found", site.sample))?;
            Ok::<types::ResolvedDiagnosticSite, String>(types::ResolvedDiagnosticSite {
                chrom: site.chrom.clone(),
                position: site.position,
                sample: site.sample.clone(),
                sample_index,
            })
        })
        .transpose()?;
    let map = window::map_from_config(&config)?;
    let mut reference = config
        .reference
        .as_deref()
        .map(|path| ReferenceWindowStream::open(path, map.clone(), config.threads))
        .transpose()?;
    let reference_samples = reference.as_ref().map_or(0, |stream| stream.samples);
    let phase_haplotypes = samples
        .len()
        .checked_add(reference_samples)
        .and_then(|samples| samples.checked_mul(2))
        .ok_or_else(|| "phase haplotype count overflow".to_owned())?;
    let phase_config = config.phase_config(phase_haplotypes)?;
    let temp_output = if config.format == types::OutputFormat::Vcf {
        Some(temporary_vcf_path(&config.output)?)
    } else {
        None
    };
    let mut temp_guard = temp_output.as_ref().map(|path| TempOutputGuard {
        path: path.clone(),
        armed: true,
    });
    let output_path = temp_output.as_deref().unwrap_or(&config.output);
    let phase_provenance = reader.phase_provenance();
    let result = run_to_output(
        &config,
        &mut reader,
        &samples,
        &phase_config,
        phase_provenance,
        diagnostic_site.as_ref(),
        &mut reference,
        reference_samples,
        config.threads,
        output_path,
        &mut progress,
    );
    match result {
        Ok(summary) => {
            if let Some(temp_output) = temp_output.as_ref() {
                crate::pgen_writer::publish_without_overwrite(temp_output, &config.output)
                    .map_err(|err| {
                        format!(
                            "publish VCF output {} without replacing an existing file: {err}",
                            config.output.display(),
                        )
                    })?;
                if std::fs::remove_file(temp_output).is_ok() {
                    if let Some(guard) = temp_guard.as_mut() {
                        guard.armed = false;
                    }
                }
            }
            Ok(summary)
        }
        Err(err) => Err(err),
    }
}

/// Estimate the peak resident working set for one active genotype window.
///
/// Windows are processed serially, so the process estimate is the startup
/// baseline plus the largest single-window allocation rather than a sum over
/// worker threads. The marker bound is supplied by the metadata scanner and is
/// capped by `window_markers` inside the estimator.
pub fn estimate_imp_window_bytes(
    config: &ImpConfig,
    target_samples: usize,
    reference_samples: usize,
    max_target_window_markers: usize,
    max_reference_window_markers: usize,
    max_union_window_markers: usize,
    max_initial_pbwt_chunk_markers: usize,
    max_window_cm_span: f64,
) -> Result<u64, String> {
    if target_samples == 0 {
        return Err("memory estimate requires at least one target sample".to_owned());
    }
    if config.threads == 0 {
        return Err("memory estimate requires at least one worker".to_owned());
    }
    let target_marker_count = max_target_window_markers.min(config.window_markers);
    let reference_marker_count = max_reference_window_markers.min(config.window_markers);
    let union_marker_count = max_union_window_markers
        .min(config.window_markers)
        .max(target_marker_count);
    if target_marker_count == 0 || union_marker_count == 0 {
        return Err("memory estimate requires at least one marker".to_owned());
    }

    // Metadata-only planning cannot observe multiallelic records. Most inputs
    // are biallelic, whose allele index needs one bit; PhaseTrack additionally
    // stores a second missingness bit.
    let bits_per_allele = 1;
    let phase_samples = target_samples
        .checked_add(reference_samples)
        .ok_or_else(|| "memory estimate phase sample count overflow".to_owned())?;
    let imp_states = if config.reference.is_some() && config.impute {
        config.imp_states
    } else {
        config.phase_states
    };
    let mut estimate = memory::estimate_window(
        target_samples,
        reference_samples,
        target_marker_count,
        union_marker_count,
        config.phase_states,
        imp_states,
        bits_per_allele,
        config.threads,
    )?;
    memory::include_phase_pbwt_donors(
        &mut estimate,
        phase_samples,
        max_phase_pbwt_steps(config, target_marker_count),
    )?;
    memory::include_initial_phase_pbwt_workspace_exact(
        &mut estimate,
        target_samples,
        reference_samples,
        target_marker_count,
        max_initial_pbwt_chunk_markers.clamp(1, target_marker_count),
        bits_per_allele,
        config.threads,
    )?;
    if config.reference.is_some() {
        if config.impute {
            let ref_allele_bytes = reference_samples
                .saturating_mul(2)
                .div_ceil(64)
                .saturating_mul(std::mem::size_of::<u64>())
                .saturating_mul(reference_marker_count);
            let will_spill =
                reference_store::ReferenceSpillPolicy::from_env().should_spill(ref_allele_bytes);
            memory::include_compact_reference_window(
                &mut estimate,
                reference_samples,
                reference_marker_count,
                will_spill,
            )?;
        } else {
            memory::include_reference_window(
                &mut estimate,
                reference_samples,
                target_marker_count.min(reference_marker_count),
            )?;
        }
        if config.impute {
            let step_count =
                estimated_imp_step_count(config, max_window_cm_span, union_marker_count);
            memory::set_compact_target_mosaic_storage(
                &mut estimate,
                union_marker_count,
                target_marker_count,
                imp_states,
                reference_samples
                    .checked_mul(2)
                    .ok_or_else(|| "reference haplotype count overflow".to_owned())?,
                step_count,
            )?;
            memory::include_imp_ibs_reference_index(
                &mut estimate,
                reference_samples
                    .checked_mul(2)
                    .ok_or_else(|| "reference haplotype count overflow".to_owned())?,
                union_marker_count,
                step_count,
                config.threads,
            )?;
            memory::include_streamed_imputation_window(
                &mut estimate,
                target_samples,
                union_marker_count,
                reference_samples
                    .checked_mul(2)
                    .ok_or_else(|| "reference haplotype count overflow".to_owned())?,
                config.threads,
            )?;
        }
    }
    if is_bgzf_vcf_output(config) {
        memory::include_bgzf_writer_workspace(&mut estimate)?;
    }

    // Rayon worker stacks and small reader/writer structures are outside the
    // active-window accounting. Two MiB per worker plus 16 MiB of headroom is
    // a conservative allowance for those allocations.
    let fixed_headroom = config
        .threads
        .checked_mul(2 * 1024 * 1024)
        .and_then(|bytes| bytes.checked_add(16 * 1024 * 1024))
        .ok_or_else(|| "memory estimate headroom overflow".to_owned())?;
    let total_bytes = estimate
        .total_bytes
        .checked_add(fixed_headroom)
        .ok_or_else(|| "total memory estimate overflow".to_owned())?;
    u64::try_from(total_bytes).map_err(|_| "memory estimate exceeds u64".to_owned())
}

fn validate_output_paths(config: &ImpConfig) -> Result<(), String> {
    let mut inputs = genotype_input_paths(&config.target);
    if config.impute {
        if let Some(reference) = config.reference.as_deref() {
            inputs.extend(genotype_input_paths(reference));
        }
    }
    if let Some(map) = config.map.as_deref() {
        inputs.push(map.to_path_buf());
    }
    let outputs = match config.format {
        types::OutputFormat::Vcf => vec![config.output.clone()],
        types::OutputFormat::Pgen => {
            let prefix = strip_pgen_suffix(&config.output);
            [".pgen", ".pvar", ".psam"]
                .into_iter()
                .map(|suffix| append_suffix(&prefix, suffix))
                .collect()
        }
    };
    for output in outputs {
        if output.exists() {
            return Err(format!("output path {} already exists", output.display()));
        }
        let canonical_output = canonicalish_path(&output);
        if inputs
            .iter()
            .any(|input| canonical_output == canonicalish_path(input))
        {
            return Err(format!(
                "output path {} would overwrite an input file",
                output.display()
            ));
        }
    }
    Ok(())
}

fn genotype_input_paths(path: &Path) -> Vec<PathBuf> {
    let mut paths = vec![path.to_path_buf()];
    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower.ends_with(".vcf")
        || lower.ends_with(".vcf.gz")
        || lower.ends_with(".hmp")
        || lower.ends_with(".hmp.gz")
        || lower.ends_with(".hmp.txt")
        || lower.ends_with(".hmp.txt.gz")
    {
        return paths;
    }
    if crate::pgen::path_uses_pgen(&path.to_string_lossy()) {
        let prefix = PathBuf::from(crate::pgen::normalize_pgen_prefix(&path.to_string_lossy()));
        paths.push(append_suffix(&prefix, ".pgen"));
        paths.push(crate::pgen::pvar_path_for_prefix(
            prefix.to_string_lossy().as_ref(),
        ));
        paths.push(append_suffix(&prefix, ".psam"));
        return dedup_paths(paths);
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let plink_prefix = if matches!(extension.as_str(), "bed" | "bim" | "fam")
        || append_suffix(path, ".bed").is_file()
    {
        Some(if matches!(extension.as_str(), "bed" | "bim" | "fam") {
            path.with_extension("")
        } else {
            path.to_path_buf()
        })
    } else {
        None
    };
    if let Some(prefix) = plink_prefix {
        for suffix in [".bed", ".bim", ".fam"] {
            paths.push(append_suffix(&prefix, suffix));
        }
        return dedup_paths(paths);
    }

    let prefix = if matches!(extension.as_str(), "txt" | "tsv" | "csv" | "npy" | "bin") {
        path.with_extension("")
    } else {
        path.to_path_buf()
    };
    let cached_prefix = cache_prefix_path(&prefix);
    for stem in [&prefix, &cached_prefix] {
        for suffix in [
            ".txt",
            ".tsv",
            ".csv",
            ".npy",
            ".bin",
            ".id",
            ".bin.id",
            ".fam",
            ".site",
            ".site.tsv",
            ".site.txt",
            ".site.csv",
            ".sites.tsv",
            ".sites.txt",
            ".sites.csv",
            ".bsite",
            ".bim",
        ] {
            let candidate = append_suffix(stem, suffix);
            if candidate.exists() {
                paths.push(candidate);
            }
        }
    }
    dedup_paths(paths)
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique = Vec::with_capacity(paths.len());
    for path in paths {
        if !unique.contains(&path) {
            unique.push(path);
        }
    }
    unique
}

fn cache_prefix_path(prefix: &Path) -> PathBuf {
    let name = prefix
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if name.starts_with('~') {
        return prefix.to_path_buf();
    }
    let mut cached = prefix.to_path_buf();
    cached.set_file_name(format!("~{name}"));
    cached
}

fn strip_pgen_suffix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    for suffix in [".pgen", ".pvar.zst", ".pvar", ".psam"] {
        if text.to_ascii_lowercase().ends_with(suffix) {
            return PathBuf::from(&text[..text.len() - suffix.len()]);
        }
    }
    path.to_path_buf()
}

fn append_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut output = prefix.as_os_str().to_os_string();
    output.push(suffix);
    PathBuf::from(output)
}

fn canonicalish_path(path: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    let parent = parent.unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
    match path.file_name() {
        Some(file_name) => parent.join(file_name),
        None => parent,
    }
}

struct TempOutputGuard {
    path: PathBuf,
    armed: bool,
}

impl Drop for TempOutputGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn report_progress<F>(
    progress: &mut F,
    stage: &str,
    chrom: &str,
    window_index: u64,
    markers: usize,
    variants_written: u64,
    samples: usize,
) -> Result<(), String>
where
    F: FnMut(&ImpProgress) -> Result<(), String>,
{
    progress(&ImpProgress {
        stage: stage.to_owned(),
        chrom: chrom.to_owned(),
        window_index,
        markers,
        variants_written,
        samples,
    })
}

fn run_to_output<F>(
    config: &ImpConfig,
    reader: &mut crate::imp_reader::ImpReader,
    samples: &[String],
    phase_config: &types::PhaseConfig,
    phase_provenance: crate::imp_reader::PhaseProvenance,
    diagnostic_site: Option<&types::ResolvedDiagnosticSite>,
    reference: &mut Option<ReferenceWindowStream>,
    reference_samples: usize,
    requested_workers: usize,
    output_path: &std::path::Path,
    progress: &mut F,
) -> Result<ImpSummary, String>
where
    F: FnMut(&ImpProgress) -> Result<(), String>,
{
    let imputation_error = f64::from(phase_config.hmm.err as f32);
    let profile_windows = std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some();
    let force_normal_phasing = std::env::var("JX_IMP_DISABLE_DIRECT_IMPUTE")
        .ok()
        .is_some_and(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        });
    let rows = std::iter::from_fn(|| reader.next_row().transpose());
    let map = window::map_from_config(config)?;
    let mut windows = window::WindowCursor::new(rows, map, config);
    let output_fields = if reference.is_some() && config.impute {
        config.impute_config().output
    } else {
        types::VcfOutputFields {
            dosage: false,
            allele_probabilities: false,
            genotype_probabilities: false,
        }
    };
    let completed_rows = Arc::new(AtomicU64::new(0));
    let writer = PostPhasePipeline::start(
        ImpOutputWriter::create(
            config.format,
            output_path,
            samples,
            output_fields,
            requested_workers,
        )?,
        output_path,
        Arc::clone(&completed_rows),
    )?;
    let mut summary = ImpSummary {
        variants_written: 0,
        samples_written: samples.len(),
        peak_window_bytes: 0,
        region: config.region.clone(),
        total_target_windows: 0,
        direct_imputation_windows: 0,
        normal_phasing_windows: 0,
        reference_phasing_windows: 0,
        imputation_windows: 0,
        direct_fallback_reasons: std::collections::BTreeMap::new(),
        direct_bypass_fraction: 0.0,
        parameter_trace: Vec::new(),
        hmm_trace: Vec::new(),
    };
    let mut previous_phase_carry = Vec::new();
    let mut reference_schedule = reference_window::ReferenceWindowSchedule::default();
    let mut pool: Option<(usize, Arc<rayon::ThreadPool>)> = None;
    loop {
        let window_started = profile_windows.then(Instant::now);
        let mut staged_reference_data = None;
        let mut scheduled_bounds = None;
        let next_window = {
            let mut admission = |_marker_count: usize, _row: &types::VariantRow| Ok(());
            if config.impute {
                if let Some(reference_stream) = reference.as_mut() {
                    match reference_schedule.next_window(
                        &mut windows,
                        reference_stream,
                        config,
                        &mut admission,
                    )? {
                        Some(scheduled) => {
                            staged_reference_data = Some(scheduled.reference);
                            scheduled_bounds = Some((
                                scheduled.terminal,
                                scheduled.outgoing_phase_carry_len,
                                scheduled.previous_emitted_pos,
                                scheduled.reference_emit_end_pos,
                            ));
                            Some(scheduled.target)
                        }
                        None => None,
                    }
                } else {
                    windows.next_window_with_admission(&mut admission)?
                }
            } else {
                windows.next_window_with_admission(&mut admission)?
            }
        };
        let Some(mut window) = next_window else {
            break;
        };
        let terminal_target_window = scheduled_bounds.map_or_else(
            || windows.is_terminal_window(&window.chrom),
            |bounds| bounds.0,
        );
        let reference_emit_end_pos = if let Some(bounds) = scheduled_bounds {
            Some(bounds.3)
        } else if config.impute && terminal_target_window {
            Some(
                config
                    .region
                    .as_ref()
                    .filter(|region| region.contains_chromosome(&window.chrom))
                    .and_then(|region| region.end)
                    .unwrap_or(u64::MAX),
            )
        } else {
            None
        };
        report_progress(
            progress,
            "read",
            &window.chrom,
            window.index,
            window.rows.len(),
            summary.variants_written,
            samples.len(),
        )?;
        // Beagle's rare-carrier classification reads the unspliced Window GT.
        // Only clone when an incoming overlap prefix will be overwritten.
        let carrier_window_storage = (window.incoming_overlap_len > 0).then(|| window.clone());
        phase::splice_phase_overlap(&mut window, &previous_phase_carry)?;
        summary.total_target_windows = summary.total_target_windows.saturating_add(1);
        let direct_eligibility = if force_normal_phasing {
            direct::DirectEligibility::Fallback(direct::DirectFallbackReason::ForcedNormal)
        } else {
            direct::classify(&window, phase_provenance, config.assume_global_phase)
        };
        let direct_allowed = matches!(direct_eligibility, direct::DirectEligibility::Eligible);
        let phase_bypass_allowed = direct_allowed;
        if let direct::DirectEligibility::Fallback(reason) = direct_eligibility {
            *summary
                .direct_fallback_reasons
                .entry(reason.as_str().to_owned())
                .or_default() += 1;
        }
        let target_storage_excess =
            memory::variant_storage_excess_sum(&window.rows, samples.len())?;
        let target_marginal_excess = memory::variant_marginal_excess_sum(&window.rows)?;
        let target_phase_expansion_bytes =
            memory::phase_expansion_window_bytes(&window.rows, samples.len())?
                .checked_add(memory::packed_state_major_biallelic_bytes(
                    reference_samples
                        .checked_mul(2)
                        .ok_or_else(|| "reference haplotype count overflow".to_owned())?,
                    window.rows.len(),
                )?)
                .ok_or_else(|| "reference phase memory estimate overflow".to_owned())?;
        let target_bits = max_allele_bits(&window.rows, &[]);
        let target_base_estimate = target_window_estimate(
            config,
            samples.len(),
            reference_samples,
            window.rows.len(),
            target_bits,
            target_storage_excess,
            target_marginal_excess,
            target_phase_expansion_bytes,
            1,
        )?;
        let mut target_base_estimate = target_base_estimate;
        if !phase_bypass_allowed {
            memory::include_initial_phase_pbwt_workspace_exact(
                &mut target_base_estimate,
                samples.len(),
                reference_samples,
                window.rows.len(),
                pbwt::initial_pbwt_max_chunk_markers(&window.cm_positions, requested_workers)?,
                target_bits,
                requested_workers,
            )?;
        }
        let mut ibd2_elapsed = Duration::ZERO;
        let mut ibd2_intervals = None;
        let previous_emitted_pos = scheduled_bounds.map(|bounds| bounds.2).unwrap_or_else(|| {
            reference
                .as_ref()
                .filter(|_| config.impute)
                .filter(|stream| {
                    stream.last_emitted_chromosome.as_deref() == Some(window.chrom.as_str())
                })
                .and_then(|stream| stream.last_emitted_target_pos)
        });
        enum ActiveReferenceWindow {
            Compact(reference_store::CompactReferenceWindow),
            Phased(types::PhasedWindow, Vec<f64>, Vec<Option<u64>>),
        }
        let reference_started = profile_windows.then(Instant::now);
        let mut reference_window_data = if let Some(compact) = staged_reference_data.take() {
            Some(ActiveReferenceWindow::Compact(compact))
        } else if let Some(reference_stream) = reference.as_mut() {
            let (phased, cm, groups) = reference_stream.window_for(
                &window,
                config.window_markers,
                config,
                terminal_target_window,
            )?;
            Some(ActiveReferenceWindow::Phased(phased, cm, groups))
        } else {
            None
        };

        let reference_elapsed =
            reference_started.map_or(Duration::ZERO, |started| started.elapsed());
        report_progress(
            progress,
            "reference",
            &window.chrom,
            window.index,
            window.rows.len(),
            summary.variants_written,
            samples.len(),
        )?;
        let merged_markers = match reference_window_data.as_ref() {
            Some(ActiveReferenceWindow::Compact(compact_ref)) => {
                impute::merged_marker_count_from_metas(
                    &window.rows,
                    &compact_ref.metas,
                    Some(config.window_markers),
                )?
            }
            Some(ActiveReferenceWindow::Phased(reference_window, _, _)) => {
                impute::merged_marker_count(
                    &window.rows,
                    &reference_window.rows,
                    Some(config.window_markers),
                )?
            }
            None => window.rows.len(),
        };
        let reference_marker_count = match reference_window_data.as_ref() {
            Some(ActiveReferenceWindow::Compact(compact_ref)) => compact_ref.n_markers(),
            Some(ActiveReferenceWindow::Phased(reference_window, _, _)) => {
                reference_window.rows.len()
            }
            None => window.rows.len(),
        };
        let perform_imputation = config.impute && reference_window_data.is_some();
        if perform_imputation {
            summary.imputation_windows = summary.imputation_windows.saturating_add(1);
        }
        let imp_states = if perform_imputation {
            if config.adaptive_states {
                impute::adaptive_imp_states(
                    config.imp_states,
                    reference_samples
                        .checked_mul(2)
                        .ok_or_else(|| "reference haplotype count overflow".to_owned())?,
                    reference_marker_count,
                    window.rows.len(),
                )
            } else {
                config.imp_states
            }
        } else {
            phase_config.hmm.phase_states
        };
        let bits_per_allele = match reference_window_data.as_ref() {
            Some(ActiveReferenceWindow::Phased(reference_window, _, _)) => {
                max_allele_bits(&window.rows, reference_window.rows.as_slice())
            }
            _ => max_allele_bits(&window.rows, &[]),
        };
        let reference_marker_alignment = match reference_window_data.as_ref() {
            Some(ActiveReferenceWindow::Compact(compact_ref)) => Some(
                impute::ReferenceMarkerAlignment::new_from_metas(&window.rows, &compact_ref.metas)?,
            ),
            Some(ActiveReferenceWindow::Phased(reference_window, _, _)) => Some(
                impute::ReferenceMarkerAlignment::new(&window.rows, &reference_window.rows)?,
            ),
            None => None,
        };
        let mut low_frequency_data = if phase_bypass_allowed {
            None
        } else {
            let phase_ref = match reference_window_data.as_ref() {
                Some(ActiveReferenceWindow::Compact(compact_ref)) => {
                    Some(reference_store::PhaseReference::Compact(compact_ref))
                }
                Some(ActiveReferenceWindow::Phased(reference_window, _, _)) => {
                    Some(reference_store::PhaseReference::Phased(reference_window))
                }
                None => None,
            };
            Some(impute::low_frequency_marker_data_with_alignment_ref(
                carrier_window_storage.as_ref().unwrap_or(&window),
                phase_ref,
                reference_marker_alignment.as_ref(),
            )?)
        };
        let use_two_stage_phase = low_frequency_data
            .as_ref()
            .is_some_and(|(mask, _)| impute::use_two_stage_phase_from_mask(mask));
        let mut phase_cm_positions = if phase_bypass_allowed {
            window.cm_positions.clone()
        } else {
            phase::beagle_phase_marker_map(&window)?
        };
        let stage1_low_frequency_mask =
            use_two_stage_phase.then(|| low_frequency_data.as_ref().unwrap().0.as_slice());
        let stage1_marker_indices = stage1_low_frequency_mask
            .as_deref()
            .map(|mask| {
                mask.iter()
                    .enumerate()
                    .filter_map(|(marker, &is_low_frequency)| (!is_low_frequency).then_some(marker))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| (0..window.rows.len()).collect());
        let stage1_cm_positions = stage1_marker_indices
            .iter()
            .map(|&marker| phase_cm_positions[marker])
            .collect::<Vec<_>>();
        let stage1_marker_count = stage1_cm_positions.len();
        let mut active_estimate = memory::estimate_window(
            samples.len(),
            reference_samples,
            window.rows.len(),
            merged_markers,
            phase_config.hmm.phase_states,
            imp_states,
            bits_per_allele,
            requested_workers,
        )?;
        if !phase_bypass_allowed {
            memory::include_initial_phase_pbwt_workspace_exact(
                &mut active_estimate,
                samples.len(),
                reference_samples,
                window.rows.len(),
                pbwt::initial_pbwt_max_chunk_markers(&phase_cm_positions, requested_workers)?,
                bits_per_allele,
                requested_workers,
            )?;
        }
        if is_bgzf_vcf_output(config) {
            memory::include_bgzf_writer_workspace(&mut active_estimate)?;
        }
        memory::include_variant_storage(&mut active_estimate, &window.rows, samples.len(), 2)?;
        if let Some(mask) = stage1_low_frequency_mask.as_deref() {
            let stage1_storage_excess = window
                .rows
                .iter()
                .zip(mask)
                .filter(|(_, &is_low_frequency)| !is_low_frequency)
                .try_fold(0usize, |sum, (row, _)| {
                    sum.checked_add(memory::row_storage_excess(row, samples.len())?)
                        .ok_or_else(|| "stage-1 projection storage estimate overflow".to_owned())
                })?;
            memory::include_row_storage_excess(&mut active_estimate, stage1_storage_excess, 2)?;
        }
        memory::include_phase_expansion(&mut active_estimate, target_phase_expansion_bytes)?;
        let mut marginal_excess = target_marginal_excess;
        if !phase_bypass_allowed {
            if stage1_low_frequency_mask.is_some() {
                memory::include_low_frequency_phase_pbwt_donors(
                    &mut active_estimate,
                    samples.len().saturating_add(reference_samples),
                    phase_pbwt_step_count(&config, &stage1_cm_positions, stage1_marker_count),
                )?;
            } else {
                memory::include_phase_pbwt_donors(
                    &mut active_estimate,
                    samples.len().saturating_add(reference_samples),
                    phase_pbwt_step_count(&config, &stage1_cm_positions, stage1_marker_count),
                )?;
            }
        }
        if perform_imputation {
            let (ref_markers, reference_cm) = match reference_window_data
                .as_ref()
                .expect("imputation requires a reference window")
            {
                ActiveReferenceWindow::Compact(compact_ref) => (
                    compact_ref.n_markers(),
                    compact_ref.cm_positions.as_slice(),
                ),
                ActiveReferenceWindow::Phased(reference_window, reference_cm, _) => (
                    reference_window.rows.len(),
                    reference_cm.as_slice(),
                ),
            };
            memory::set_compact_target_mosaic_storage(
                &mut active_estimate,
                merged_markers,
                window.rows.len(),
                imp_states,
                reference_samples
                    .checked_mul(2)
                    .ok_or_else(|| "reference haplotype count overflow".to_owned())?,
                estimated_imp_step_count(
                    config,
                    merged_cm_span(&window.cm_positions, reference_cm)?,
                    merged_markers,
                ),
            )?;
            match reference_window_data.as_ref() {
                Some(ActiveReferenceWindow::Compact(compact_ref)) => {
                    let will_spill = !phase_bypass_allowed
                        && reference_store::ReferenceSpillPolicy::from_env()
                            .should_spill(compact_ref.allele_bytes());
                    memory::include_compact_reference_window(
                        &mut active_estimate,
                        reference_samples,
                        ref_markers,
                        will_spill,
                    )?;
                }
                _ => {
                    memory::include_reference_window(
                        &mut active_estimate,
                        reference_samples,
                        ref_markers,
                    )?;
                }
            }
            memory::include_imp_ibs_reference_index(
                &mut active_estimate,
                reference_samples
                    .checked_mul(2)
                    .ok_or_else(|| "reference haplotype count overflow".to_owned())?,
                merged_markers,
                estimated_imp_step_count(
                    config,
                    merged_cm_span(&window.cm_positions, reference_cm)?,
                    merged_markers,
                ),
                requested_workers,
            )?;
            if let Some(ActiveReferenceWindow::Phased(reference_window, _, _)) =
                reference_window_data.as_ref()
            {
                memory::include_variant_storage(
                    &mut active_estimate,
                    &reference_window.rows,
                    reference_samples,
                    2,
                )?;
                marginal_excess = marginal_excess
                    .checked_add(memory::variant_marginal_excess_sum(&reference_window.rows)?)
                    .ok_or_else(|| "multiallelic posterior memory estimate overflow".to_owned())?;
            }
            memory::include_streamed_imputation_window(
                &mut active_estimate,
                samples.len(),
                merged_markers,
                reference_samples
                    .checked_mul(2)
                    .ok_or_else(|| "reference haplotype count overflow".to_owned())?,
                requested_workers,
            )?;
        } else if let Some(ActiveReferenceWindow::Phased(reference_window, _, _)) =
            reference_window_data.as_ref()
        {
            memory::include_reference_window(
                &mut active_estimate,
                reference_samples,
                reference_window.rows.len(),
            )?;
            memory::include_variant_storage(
                &mut active_estimate,
                &reference_window.rows,
                reference_samples,
                2,
            )?;
        }
        let stage1_ibd2_intervals = if !phase_bypass_allowed {
            // Beagle constructs a fresh Ibs2 object from Stage1's projected
            // target GT and effective marker map. The map enforces Beagle's
            // minimum physical-to-genetic distance independently of the raw
            // map positions used to define sliding windows.
            drop(ibd2_intervals.take());
            let ibd2_started = profile_windows.then(Instant::now);
            let phase_ref = match reference_window_data.as_ref() {
                Some(ActiveReferenceWindow::Compact(compact_ref)) => {
                    Some(reference_store::PhaseReference::Compact(compact_ref))
                }
                Some(ActiveReferenceWindow::Phased(reference_window, _, _)) => {
                    Some(reference_store::PhaseReference::Phased(reference_window))
                }
                None => None,
            };
            let intervals = Some(ibd2::detect_projected_intervals_with_phase_reference(
                &window,
                &stage1_marker_indices,
                &phase_cm_positions,
                phase_ref,
                reference_marker_alignment.as_ref(),
                Some(config.master_seed()),
                None,
                None,
            )?);
            ibd2_elapsed += ibd2_started.map_or(Duration::ZERO, |started| started.elapsed());
            intervals
        } else {
            None
        };
        let prepacked_compact_reference = if !phase_bypass_allowed {
            match (
                reference_window_data.as_mut(),
                reference_marker_alignment.as_ref(),
            ) {
                (Some(ActiveReferenceWindow::Compact(compact_ref)), Some(alignment)) => {
                    // Spill before allocating state-major `packed` so the
                    // marker-major 1-bit reference buffer and `packed` never
                    // coexist in RAM when P1-1b is active.
                    let _spilled = compact_ref.maybe_spill_to_disk()?;
                    let (packed, target_samples, reference_present) =
                        phase::pack_compact_reference_haplotypes_aligned(
                            &window,
                            compact_ref,
                            alignment,
                        )?;
                    let lf_data = low_frequency_data
                        .take()
                        .expect("low-frequency data is computed whenever phasing runs");
                    Some(phase::PrepackedPhaseReference {
                        packed,
                        target_samples,
                        reference_present,
                        low_frequency_data: lf_data,
                    })
                }
                _ => None,
            }
        } else {
            None
        };
        let no_ibd2_intervals = Vec::new();
        let phase_ibd2_intervals = stage1_ibd2_intervals
            .as_ref()
            .or(ibd2_intervals.as_ref())
            .unwrap_or(&no_ibd2_intervals);
        memory::include_ibd2_intervals(&mut active_estimate, phase_ibd2_intervals)?;
        memory::include_multiallelic_marginal_storage(&mut active_estimate, marginal_excess)?;
        let workers = requested_workers;
        memory::set_worker_count(&mut active_estimate, workers)?;
        summary.peak_window_bytes = summary
            .peak_window_bytes
            .max(active_estimate.total_bytes as u64);
        ensure_thread_pool(&mut pool, workers)?;
        let worker_pool = Arc::clone(&pool.as_ref().unwrap().1);
        let phase_started = profile_windows.then(Instant::now);
        let (mut phased, fitted_hmm_config, parameter_trace, hmm_trace) = if direct_allowed {
            report_progress(
                progress,
                "direct",
                &window.chrom,
                window.index,
                window.rows.len(),
                summary.variants_written,
                samples.len(),
            )?;
            summary.direct_imputation_windows = summary.direct_imputation_windows.saturating_add(1);
            (
                direct::phased_window_from_calls(&window)?,
                phase_config.hmm,
                Vec::new(),
                Vec::new(),
            )
        } else {
            report_progress(
                progress,
                "phase",
                &window.chrom,
                window.index,
                window.rows.len(),
                summary.variants_written,
                samples.len(),
            )?;
            summary.normal_phasing_windows = summary.normal_phasing_windows.saturating_add(1);
            std::mem::swap(&mut window.cm_positions, &mut phase_cm_positions);
            let phase_result = {
                let carrier_window = carrier_window_storage.as_ref().unwrap_or(&window);
                if let Some(prepacked) = prepacked_compact_reference {
                    phase::phase_window_prepared_with_prepacked(
                        &window,
                        carrier_window,
                        phase_config,
                        pbwt::beagle_window_seed(config.master_seed(), window.index),
                        phase_ibd2_intervals,
                        worker_pool.as_ref(),
                        Some(prepacked),
                        None,
                        diagnostic_site,
                    )
                } else {
                    let phased_ref = match reference_window_data.as_ref() {
                        Some(ActiveReferenceWindow::Phased(reference_window, _, _)) => {
                            Some(reference_window)
                        }
                        _ => None,
                    };
                    phase::phase_window_prepared(
                        &window,
                        carrier_window,
                        phase_config,
                        pbwt::beagle_window_seed(config.master_seed(), window.index),
                        phase_ibd2_intervals,
                        worker_pool.as_ref(),
                        phased_ref,
                        reference_marker_alignment.as_ref(),
                        low_frequency_data.as_ref().map(|(mask, _)| mask.as_slice()),
                        diagnostic_site,
                    )
                }
            };
            std::mem::swap(&mut window.cm_positions, &mut phase_cm_positions);
            phase_result?
        };
        drop(carrier_window_storage);
        drop(reference_marker_alignment);
        drop(low_frequency_data);
        let expected_reference_haplotypes = samples
            .len()
            .checked_add(reference_samples)
            .and_then(|samples| samples.checked_mul(2));
        if !phase_bypass_allowed
            && reference_window_data.is_some()
            && expected_reference_haplotypes == Some(fitted_hmm_config.haplotype_count)
        {
            summary.reference_phasing_windows = summary.reference_phasing_windows.saturating_add(1);
        }
        let phase_elapsed = phase_started.map_or(Duration::ZERO, |started| started.elapsed());
        summary.parameter_trace.extend(parameter_trace);
        summary.hmm_trace.extend(hmm_trace);
        let orientation_flips = align_overlap_phases(
            &mut phased,
            &previous_phase_carry,
            window.incoming_overlap_len,
        )?;
        let outgoing_phase_carry_len =
            scheduled_bounds.map_or_else(|| windows.outgoing_phase_carry_len(), |bounds| bounds.1);
        if outgoing_phase_carry_len > phased.emit_end {
            return Err(format!(
                "outgoing phase carry has {outgoing_phase_carry_len} rows before splice {}",
                phased.emit_end
            ));
        }
        let outgoing_phase_carry_start = phased.emit_end - outgoing_phase_carry_len;
        previous_phase_carry = phased.rows[outgoing_phase_carry_start..phased.emit_end].to_vec();
        let output_chrom = window.chrom.clone();
        let output_window_index = window.index;
        let output_markers = window.rows.len();
        let impute_started = profile_windows.then(Instant::now);
        report_progress(
            progress,
            "impute",
            &window.chrom,
            window.index,
            window.rows.len(),
            summary.variants_written,
            samples.len(),
        )?;
        match reference_window_data {
            Some(ActiveReferenceWindow::Compact(reference_window)) => {
                let mut impute_config = config.impute_config();
                impute_config.states = imp_states;
                // ImpIbs samples donor candidates from the command-line seed,
                // independently of the per-window phasing random stream.
                let seed = config.master_seed();
                let write_started = profile_windows.then(Instant::now);
                report_progress(
                    progress,
                    "write",
                    &output_chrom,
                    output_window_index,
                    output_markers,
                    summary.variants_written,
                    samples.len(),
                )?;
                if let Some(reference_stream) = reference.as_mut() {
                    if let Some(last) = phased.rows[phased.emit_start..phased.emit_end]
                        .iter()
                        .rev()
                        .find(|row| {
                            region_marker_is_emittable(
                                config.region.as_ref(),
                                &row.meta.chrom,
                                row.meta.pos,
                            )
                        })
                    {
                        reference_stream.last_emitted_target_pos = Some(last.meta.pos);
                        reference_stream.last_emitted_chromosome = Some(last.meta.chrom.clone());
                    }
                }
                let sync_impute = merged_markers >= 50_000
                    || reference_window
                        .n_markers()
                        .saturating_mul(reference_samples)
                        >= 8_000_000;
                let impute_job = PostPhaseJob::Impute {
                    window,
                    phased,
                    reference_window,
                    impute_config,
                    fitted_hmm_config,
                    imputation_error,
                    seed,
                    previous_emitted_pos,
                    reference_emit_end_pos,
                    orientation_flips,
                    region: config.region.clone(),
                    worker_pool: Arc::clone(&worker_pool),
                };
                if sync_impute {
                    writer.submit_sync(impute_job)?;
                } else {
                    writer.submit(impute_job)?;
                }
                summary.variants_written = completed_rows.load(Ordering::Relaxed);
                let write_elapsed =
                    write_started.map_or(Duration::ZERO, |started| started.elapsed());
                let impute_elapsed = Duration::ZERO;
                if let Some(started) = window_started {
                    eprintln!(
                    "JANUSX_IMP_WINDOW_PROFILE window={}:{} markers={} imp_states={} phase_s={:.6} impute_s={:.6} ibd2_s={:.6} reference_s={:.6} write_s={:.6} total_s={:.6}",
                    output_chrom,
                    output_window_index,
                    output_markers,
                    imp_states,
                    phase_elapsed.as_secs_f64(),
                    impute_elapsed.as_secs_f64(),
                    ibd2_elapsed.as_secs_f64(),
                    reference_elapsed.as_secs_f64(),
                    write_elapsed.as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                );
                }
            }
            Some(ActiveReferenceWindow::Phased(
                reference_window,
                reference_cm,
                reference_groups,
            )) => {
                let impute_elapsed =
                    impute_started.map_or(Duration::ZERO, |started| started.elapsed());
                let write_started = profile_windows.then(Instant::now);
                report_progress(
                    progress,
                    "write",
                    &output_chrom,
                    output_window_index,
                    output_markers,
                    summary.variants_written,
                    samples.len(),
                )?;
                if let Some(reference_stream) = reference.as_mut() {
                    reference_stream.cache_carry(
                        &reference_window,
                        &reference_cm,
                        &reference_groups,
                    )?;
                    if let Some(last) = phased.rows[phased.emit_start..phased.emit_end]
                        .iter()
                        .rev()
                        .find(|row| {
                            region_marker_is_emittable(
                                config.region.as_ref(),
                                &row.meta.chrom,
                                row.meta.pos,
                            )
                        })
                    {
                        reference_stream.last_emitted_target_pos = Some(last.meta.pos);
                        reference_stream.last_emitted_chromosome = Some(last.meta.chrom.clone());
                    }
                }
                writer.submit(PostPhaseJob::Write {
                    indices: (phased.emit_start..phased.emit_end).collect(),
                    rows: phased.rows,
                    region: config.region.clone(),
                    worker_pool: Arc::clone(&worker_pool),
                })?;
                summary.variants_written = completed_rows.load(Ordering::Relaxed);
                let write_elapsed =
                    write_started.map_or(Duration::ZERO, |started| started.elapsed());
                if let Some(started) = window_started {
                    eprintln!(
                        "JANUSX_IMP_WINDOW_PROFILE window={}:{} markers={} imp_states={} phase_s={:.6} impute_s={:.6} ibd2_s={:.6} reference_s={:.6} write_s={:.6} total_s={:.6}",
                        output_chrom,
                        output_window_index,
                        output_markers,
                        imp_states,
                        phase_elapsed.as_secs_f64(),
                        impute_elapsed.as_secs_f64(),
                        ibd2_elapsed.as_secs_f64(),
                        reference_elapsed.as_secs_f64(),
                        write_elapsed.as_secs_f64(),
                        started.elapsed().as_secs_f64(),
                    );
                }
            }
            None => {
            let impute_elapsed = impute_started.map_or(Duration::ZERO, |started| started.elapsed());
            let write_started = profile_windows.then(Instant::now);
            report_progress(
                progress,
                "write",
                &output_chrom,
                output_window_index,
                output_markers,
                summary.variants_written,
                samples.len(),
            )?;
            writer.submit(PostPhaseJob::Write {
                indices: (phased.emit_start..phased.emit_end).collect(),
                rows: phased.rows,
                region: config.region.clone(),
                worker_pool: Arc::clone(&worker_pool),
            })?;
            summary.variants_written = completed_rows.load(Ordering::Relaxed);
            let write_elapsed = write_started.map_or(Duration::ZERO, |started| started.elapsed());
            if let Some(started) = window_started {
                eprintln!(
                    "JANUSX_IMP_WINDOW_PROFILE window={}:{} markers={} imp_states={} phase_s={:.6} impute_s={:.6} ibd2_s={:.6} reference_s={:.6} write_s={:.6} total_s={:.6}",
                    output_chrom,
                    output_window_index,
                    output_markers,
                    imp_states,
                    phase_elapsed.as_secs_f64(),
                    impute_elapsed.as_secs_f64(),
                    ibd2_elapsed.as_secs_f64(),
                    reference_elapsed.as_secs_f64(),
                    write_elapsed.as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                );
                }
            }
        }
        report_progress(
            progress,
            "window_done",
            &output_chrom,
            output_window_index,
            output_markers,
            summary.variants_written,
            samples.len(),
        )?;
        summary.direct_bypass_fraction = if summary.total_target_windows == 0 {
            0.0
        } else {
            summary.direct_imputation_windows as f64 / summary.total_target_windows as f64
        };
    }
    writer.finish()?;
    summary.variants_written = completed_rows.load(Ordering::Relaxed);
    if summary.variants_written == 0 {
        return Err(
            "no target or reference markers remain inside the requested -bimrange".to_owned(),
        );
    }
    Ok(summary)
}

fn region_marker_is_emittable(region: Option<&types::RegionSpec>, chrom: &str, pos: u64) -> bool {
    region.is_none_or(|region| region.contains(chrom, pos))
}

#[cfg(test)]
fn low_frequency_phase_candidate(
    phase_bypass_allowed: bool,
    carrier_window: &types::PhaseWindow,
    reference_window: Option<&types::PhasedWindow>,
    reference_alignment: Option<&impute::ReferenceMarkerAlignment>,
) -> Result<Option<Vec<bool>>, String> {
    if phase_bypass_allowed {
        return Ok(None);
    }
    impute::low_frequency_markers_with_alignment(
        carrier_window,
        reference_window,
        reference_alignment,
    )
    .map(Some)
}

fn ensure_thread_pool(
    pool: &mut Option<(usize, Arc<rayon::ThreadPool>)>,
    workers: usize,
) -> Result<(), String> {
    if pool
        .as_ref()
        .is_some_and(|(current, _)| *current == workers)
    {
        return Ok(());
    }
    let replacement = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|err| format!("create imp thread pool with {workers} workers: {err}"))?;
    *pool = Some((workers, Arc::new(replacement)));
    Ok(())
}

fn target_window_estimate(
    config: &ImpConfig,
    target_samples: usize,
    reference_samples: usize,
    marker_count: usize,
    bits_per_allele: u8,
    target_storage_excess: usize,
    target_marginal_excess: usize,
    target_phase_expansion_bytes: usize,
    workers: usize,
) -> Result<types::MemoryEstimate, String> {
    let has_reference = config.reference.is_some();
    let has_imputation = has_reference && config.impute;
    let imp_states = if has_imputation {
        config.imp_states
    } else {
        config.phase_states
    };
    let phase_samples = target_samples
        .checked_add(reference_samples)
        .ok_or_else(|| "phase sample count overflow in memory estimate".to_owned())?;
    let mut estimate = memory::estimate_window(
        target_samples,
        reference_samples,
        marker_count,
        marker_count,
        config.phase_states,
        imp_states,
        bits_per_allele,
        workers,
    )?;
    memory::include_row_storage_excess(&mut estimate, target_storage_excess, 2)?;
    memory::include_phase_expansion(&mut estimate, target_phase_expansion_bytes)?;
    memory::include_multiallelic_marginal_storage(&mut estimate, target_marginal_excess)?;
    memory::include_phase_pbwt_donors(
        &mut estimate,
        phase_samples,
        max_phase_pbwt_steps(config, marker_count),
    )?;
    if has_imputation {
        memory::include_streamed_imputation_window(
            &mut estimate,
            target_samples,
            marker_count,
            reference_samples
                .checked_mul(2)
                .ok_or_else(|| "reference haplotype count overflow".to_owned())?,
            workers,
        )?;
    }
    if is_bgzf_vcf_output(config) {
        memory::include_bgzf_writer_workspace(&mut estimate)?;
    }
    Ok(estimate)
}

fn is_bgzf_vcf_output(config: &ImpConfig) -> bool {
    config.format == types::OutputFormat::Vcf
        && config
            .output
            .to_string_lossy()
            .to_ascii_lowercase()
            .ends_with(".vcf.gz")
}

fn max_phase_pbwt_steps(config: &ImpConfig, max_markers: usize) -> usize {
    if max_markers == 0 {
        return 0;
    }
    let span_bound = config.window + 2.0 * config.overlap;
    let count = if span_bound.is_finite() && span_bound >= 0.0 {
        ((span_bound / 0.05).ceil() as usize).saturating_add(1)
    } else {
        max_markers
    };
    count.min(max_markers).max(1)
}

fn estimated_imp_step_count(config: &ImpConfig, genetic_span_cm: f64, max_markers: usize) -> usize {
    let step_cm = f64::from(config.imp_step as f32);
    if max_markers == 0 || !step_cm.is_finite() || step_cm <= 0.0 {
        return 0;
    }
    let span = genetic_span_cm.max(0.0);
    ((span / step_cm).ceil() as usize)
        .saturating_add(1)
        .min(max_markers)
        .max(1)
}

fn phase_pbwt_step_count(config: &ImpConfig, cm_positions: &[f64], max_markers: usize) -> usize {
    if max_markers == 0 || cm_positions.is_empty() {
        return 0;
    }
    let step_cm = phase::phase_ibs_step_cm(cm_positions);
    phase_pbwt_step_count_with_step_cm(config, cm_positions, step_cm, max_markers)
}

fn phase_pbwt_step_count_with_step_cm(
    config: &ImpConfig,
    cm_positions: &[f64],
    step_cm: f64,
    max_markers: usize,
) -> usize {
    if max_markers == 0 || cm_positions.is_empty() {
        return 0;
    }
    let span =
        cm_positions.last().copied().unwrap_or(0.0) - cm_positions.first().copied().unwrap_or(0.0);
    let estimate = if span.is_finite() && span >= 0.0 && step_cm.is_finite() && step_cm > 0.0 {
        ((span / step_cm).ceil() as usize).saturating_add(1)
    } else {
        max_phase_pbwt_steps(config, max_markers)
    };
    estimate.min(max_markers).max(1)
}

fn merged_cm_span(
    target_cm_positions: &[f64],
    reference_cm_positions: &[f64],
) -> Result<f64, String> {
    let Some(target_start) = target_cm_positions.first().copied() else {
        return Err("cannot estimate genetic span for an empty target map".to_owned());
    };
    let target_end = *target_cm_positions.last().unwrap();
    let start_cm = reference_cm_positions
        .first()
        .copied()
        .map_or(target_start, |reference_start| {
            target_start.min(reference_start)
        });
    let end_cm = reference_cm_positions
        .last()
        .copied()
        .map_or(target_end, |reference_end| target_end.max(reference_end));
    if !start_cm.is_finite() || !end_cm.is_finite() || end_cm < start_cm {
        return Err("cannot estimate genetic span from invalid positions".to_owned());
    }
    Ok(end_cm - start_cm)
}

fn max_allele_bits(target: &[types::VariantRow], reference: &[types::PhasedRow]) -> u8 {
    let max_allele = target
        .iter()
        .map(|row| row.meta.allele_count().saturating_sub(1))
        .chain(
            reference
                .iter()
                .map(|row| row.meta.allele_count().saturating_sub(1)),
        )
        .max()
        .unwrap_or(1);
    allele_bits_from_index(max_allele)
}

fn target_contains_position(target: &types::PhaseWindow, position: u64) -> bool {
    target
        .rows
        .binary_search_by_key(&position, |row| row.meta.pos)
        .is_ok()
}

fn allele_bits_from_index(max_allele: usize) -> u8 {
    match max_allele {
        0..=1 => 1,
        2..=3 => 2,
        4..=15 => 4,
        16..=255 => 8,
        _ => 16,
    }
}

fn align_overlap_phases(
    current: &mut types::PhasedWindow,
    previous_carry: &[types::PhasedRow],
    overlap: usize,
) -> Result<Vec<bool>, String> {
    let n_samples = current.rows.first().map_or(0, |row| row.calls.len());
    if overlap == 0 {
        return Ok(vec![false; n_samples]);
    }
    if overlap > previous_carry.len() || overlap > current.rows.len() {
        return Err(format!(
            "phase overlap has {overlap} rows, but previous/current windows contain {}/{} rows",
            previous_carry.len(),
            current.rows.len()
        ));
    }
    let previous_start = previous_carry.len() - overlap;
    if previous_carry
        .iter()
        .skip(previous_start)
        .take(overlap)
        .any(|row| row.calls.len() != n_samples)
    {
        return Err("phase overlap sample counts differ between adjacent windows".to_owned());
    }
    let mut same = vec![0usize; n_samples];
    let mut reverse = vec![0usize; n_samples];
    for marker in 0..overlap {
        let previous = &previous_carry[previous_start + marker];
        let next = &current.rows[marker];
        if previous.meta.chrom != next.meta.chrom || previous.meta.pos != next.meta.pos {
            return Err(format!(
                "phase overlap variants do not match at {}:{} versus {}:{}",
                previous.meta.chrom, previous.meta.pos, next.meta.chrom, next.meta.pos
            ));
        }
        for sample in 0..n_samples {
            let left = previous.calls.get(sample).unwrap().alleles;
            let right = next.calls.get(sample).unwrap().alleles;
            if !matches!(left, [Some(a), Some(b)] if a != b)
                || !matches!(right, [Some(a), Some(b)] if a != b)
            {
                continue;
            }
            if left == right {
                same[sample] += 1;
            } else if left[0] == right[1] && left[1] == right[0] {
                reverse[sample] += 1;
            }
        }
    }
    let flip = reverse
        .iter()
        .zip(&same)
        .map(|(&reverse, &same)| reverse > same)
        .collect::<Vec<_>>();
    if flip.iter().any(|&value| value) {
        for row in &mut current.rows {
            row.calls = types::DiploidCalls::from_iter(row.calls.iter().enumerate().map(
                |(sample, mut call)| {
                    if flip[sample] {
                        call.alleles.swap(0, 1);
                    }
                    call
                },
            ));
        }
    }
    Ok(flip)
}

fn temporary_vcf_path(output: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("imp-output.vcf");
    let lowercase_name = name.to_ascii_lowercase();
    let (prefix, suffix) = if lowercase_name.ends_with(".vcf.gz") {
        (&name[..name.len() - ".vcf.gz".len()], ".vcf.gz")
    } else if lowercase_name.ends_with(".vcf") {
        (&name[..name.len() - ".vcf".len()], ".vcf")
    } else {
        (name, ".vcf")
    };
    let parent = output.parent().unwrap_or_else(|| std::path::Path::new("."));
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| format!("system clock cannot name temporary output: {err}"))?
        .as_nanos();
    for attempt in 0..1000u16 {
        let candidate = parent.join(format!(
            "{prefix}.tmp.{}.{}.{attempt}{suffix}",
            std::process::id(),
            nonce
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                drop(file);
                return Ok(candidate);
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(format!(
                    "create temporary VCF {}: {err}",
                    candidate.display()
                ));
            }
        }
    }
    Err(format!(
        "could not allocate a unique temporary VCF path next to {}",
        output.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp::types::{
        DiploidCall, DiploidCalls, PhaseWindow, PhasedRow, VariantMeta, VariantRow,
    };
    use flate2::read::MultiGzDecoder;
    use std::io::Read;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn stage1_low_frequency_mask_uses_original_overlap_genotypes() {
        let n_samples = 1_000;
        let meta = VariantMeta {
            chrom: "1".to_owned(),
            pos: 1,
            id: "v1".to_owned(),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let original_calls = DiploidCalls::from_iter((0..n_samples).map(|sample| {
            let allele = u16::from(sample < 3);
            DiploidCall {
                alleles: [Some(allele), Some(allele)],
                phased: true,
            }
        }));
        let mut window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 1,
            rows: vec![VariantRow {
                meta: meta.clone(),
                calls: original_calls,
            }],
            cm_positions: vec![0.1],
            incoming_overlap_len: 1,
            emit_start: 0,
            emit_end: 1,
        };
        let carrier_window = window.clone();
        let previous_carry = vec![PhasedRow {
            meta,
            calls: DiploidCalls::from_iter((0..n_samples).map(|sample| {
                let allele = u16::from(sample < 100);
                DiploidCall {
                    alleles: [Some(allele), Some(allele)],
                    phased: true,
                }
            })),
            dosage: None,
            ap1: None,
            ap2: None,
            gp: None,
            info_af: None,
            info_dr2: None,
            imputed: false,
        }];

        phase::splice_phase_overlap(&mut window, &previous_carry).unwrap();
        assert_eq!(
            impute::low_frequency_markers_with_alignment(&carrier_window, None, None).unwrap(),
            vec![true]
        );
        assert_eq!(
            impute::low_frequency_markers_with_alignment(&window, None, None).unwrap(),
            vec![false]
        );
        assert_eq!(
            low_frequency_phase_candidate(false, &carrier_window, None, None).unwrap(),
            Some(vec![true])
        );
    }

    #[test]
    fn cached_phase_pbwt_step_scale_preserves_marker_caps() {
        let config = ImpConfig::default();
        let cm_positions = [0.0, 0.1, 0.3, 0.5];
        let step_cm = phase::phase_ibs_step_cm(&cm_positions);
        assert_eq!(
            phase_pbwt_step_count_with_step_cm(&config, &cm_positions, step_cm, 1),
            1
        );
        assert_eq!(
            phase_pbwt_step_count_with_step_cm(&config, &cm_positions, step_cm, 100),
            phase_pbwt_step_count(&config, &cm_positions, 100)
        );
        assert_eq!(
            phase_pbwt_step_count_with_step_cm(&config, &cm_positions, step_cm, 0),
            0
        );
    }

    #[test]
    fn region_with_no_target_or_reference_output_is_an_error() {
        let region = types::RegionSpec {
            chrom: "1".to_owned(),
            start: Some(100),
            end: Some(200),
        };
        assert!(!region_marker_is_emittable(Some(&region), "1", 99));
        assert!(region_marker_is_emittable(Some(&region), "1", 100));
        assert!(region_marker_is_emittable(Some(&region), "chr1", 200));
        assert!(!region_marker_is_emittable(Some(&region), "1", 201));
        assert!(region_marker_is_emittable(None, "1", 1));
    }

    #[test]
    fn post_phase_pipeline_preserves_window_write_order() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let output = std::env::temp_dir().join(format!(
            "janusx-imp-postphase-output-{}-{unique}.vcf.gz",
            std::process::id()
        ));
        let samples = vec!["sample".to_owned()];
        let writer = ImpOutputWriter::create(
            types::OutputFormat::Vcf,
            &output,
            &samples,
            types::VcfOutputFields {
                dosage: false,
                allele_probabilities: false,
                genotype_probabilities: false,
            },
            2,
        )
        .unwrap();
        let completed_rows = Arc::new(AtomicU64::new(0));
        let writer =
            PostPhasePipeline::start(writer, &output, Arc::clone(&completed_rows)).unwrap();
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );
        let row = |position| types::PhasedRow {
            meta: types::VariantMeta {
                chrom: "1".to_owned(),
                pos: position,
                id: format!("v{position}"),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: vec![types::DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            }]
            .into(),
            dosage: None,
            ap1: None,
            ap2: None,
            gp: None,
            info_af: None,
            info_dr2: None,
            imputed: false,
        };
        writer
            .submit(PostPhaseJob::Write {
                rows: vec![row(1), row(2)],
                indices: vec![1],
                region: None,
                worker_pool: Arc::clone(&pool),
            })
            .unwrap();
        writer
            .submit(PostPhaseJob::Write {
                rows: vec![row(3)],
                indices: vec![0],
                region: None,
                worker_pool: Arc::clone(&pool),
            })
            .unwrap();
        writer.finish().unwrap();
        assert_eq!(completed_rows.load(Ordering::Relaxed), 2);

        let mut text = String::new();
        MultiGzDecoder::new(std::fs::File::open(&output).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        let _ = std::fs::remove_file(output);
        let positions = text
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| line.split('\t').nth(1).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(positions, ["2", "3"]);
    }

    #[test]
    fn raw_phase_only_with_reference_emits_only_target_markers() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "janusx-imp-reference-phase-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("target.vcf");
        let reference = root.join("reference.vcf");
        let output = root.join("output.vcf");
        let header = concat!(
            "##fileformat=VCFv4.3\n",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4\n"
        );
        std::fs::write(
            &target,
            format!(
                "{header}\
                 1\t100\tt1\tA\tG\t.\tPASS\t.\tGT\t0/1\t0/0\t1/1\t0/1\n\
                 1\t200\tt2\tA\tG\t.\tPASS\t.\tGT\t1/1\t0/1\t0/0\t1/0\n"
            ),
        )
        .unwrap();
        std::fs::write(
            &reference,
            format!(
                "{header}\
                 1\t100\tr1\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0\t1|1\t0|1\n\
                 1\t200\tr2\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|1\t0|0\t1|0\n\
                 1\t300\tr3\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\t0|1\t1|0\n"
            ),
        )
        .unwrap();
        let config = ImpConfig {
            target,
            reference: Some(reference),
            output: output.clone(),
            format: types::OutputFormat::Vcf,
            threads: 1,
            window_markers: 20,
            phase_states: 32,
            imp_states: 8,
            burnin: 1,
            iterations: 1,
            em: false,
            impute: false,
            ..ImpConfig::default()
        };

        let summary = run_imp(config).unwrap();
        let text = std::fs::read_to_string(&output).unwrap();
        let positions = text
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| line.split('\t').nth(1).unwrap().parse::<u64>().unwrap())
            .collect::<Vec<_>>();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(positions, vec![100, 200]);
        assert_eq!(summary.variants_written, 2);
        assert_eq!(summary.normal_phasing_windows, 1);
        assert_eq!(summary.reference_phasing_windows, 1);
        assert_eq!(summary.imputation_windows, 0);
    }

    #[test]
    fn fully_phased_vcf_reports_actual_direct_bypass() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "janusx-imp-direct-count-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("target.vcf");
        let output = root.join("output.vcf");
        std::fs::write(
            &target,
            concat!(
                "##fileformat=VCFv4.3\n",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n",
                "1\t100\tv1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n",
                "1\t200\tv2\tA\tG\t.\tPASS\t.\tGT\t1|0\t0|1\n"
            ),
        )
        .unwrap();
        let config = ImpConfig {
            target,
            output,
            threads: 1,
            window_markers: 20,
            phase_states: 32,
            burnin: 1,
            iterations: 1,
            em: false,
            impute: false,
            ..ImpConfig::default()
        };
        let summary = run_imp(config).unwrap();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(summary.total_target_windows, 1);
        assert_eq!(summary.direct_imputation_windows, 1);
        assert_eq!(summary.normal_phasing_windows, 0);
        assert_eq!(summary.direct_bypass_fraction, 1.0);
        assert!(summary.direct_fallback_reasons.is_empty());
    }

    #[test]
    fn unphased_vcf_reports_zero_bypass_and_fallback_reason() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "janusx-imp-normal-count-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("target.vcf");
        let output = root.join("output.vcf");
        std::fs::write(
            &target,
            concat!(
                "##fileformat=VCFv4.3\n",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n",
                "1\t100\tv1\tA\tG\t.\tPASS\t.\tGT\t0/1\t1/0\n"
            ),
        )
        .unwrap();
        let config = ImpConfig {
            target,
            output,
            threads: 1,
            window_markers: 20,
            phase_states: 32,
            burnin: 1,
            iterations: 1,
            em: false,
            impute: false,
            ..ImpConfig::default()
        };
        let summary = run_imp(config).unwrap();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(summary.total_target_windows, 1);
        assert_eq!(summary.direct_imputation_windows, 0);
        assert_eq!(summary.normal_phasing_windows, 1);
        assert_eq!(summary.direct_bypass_fraction, 0.0);
        assert_eq!(
            summary.direct_fallback_reasons.get("unphased-het").copied(),
            Some(1)
        );
    }

    #[test]
    fn direct_fully_phased_target_still_runs_reference_imputation() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "janusx-imp-direct-impute-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("target.vcf");
        let reference = root.join("reference.vcf");
        let output = root.join("output.vcf");
        let header = concat!(
            "##fileformat=VCFv4.3\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4\n"
        );
        std::fs::write(
            &target,
            format!(
                "{header}\
                 1\t100\tt1\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0\t1|1\t0|1\n\
                 1\t1000\tt2\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|1\t0|0\t1|0\n"
            ),
        )
        .unwrap();
        std::fs::write(
            &reference,
            format!(
                "{header}\
                 1\t100\tr1\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0\t1|1\t0|1\n\
                 1\t500\tr2\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\t0|1\t1|0\n\
                 1\t600\tr3\tA\tG\t.\tPASS\t.\tGT\t1|0\t0|1\t1|0\t0|1\n\
                 1\t1000\tr4\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|1\t0|0\t1|0\n"
            ),
        )
        .unwrap();
        let config = ImpConfig {
            target,
            reference: Some(reference),
            output: output.clone(),
            threads: 1,
            window_markers: 20,
            phase_states: 32,
            imp_states: 8,
            burnin: 1,
            iterations: 1,
            em: false,
            impute: true,
            ..ImpConfig::default()
        };

        let summary = run_imp(config).unwrap();
        let text = std::fs::read_to_string(&output).unwrap();
        let positions = text
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| line.split('\t').nth(1).unwrap().parse::<u64>().unwrap())
            .collect::<Vec<_>>();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(summary.direct_imputation_windows, 1);
        assert_eq!(summary.reference_phasing_windows, 0);
        assert_eq!(summary.direct_bypass_fraction, 1.0);
        assert_eq!(positions, vec![100, 500, 600, 1000]);
    }

    #[test]
    fn reference_imputation_emits_reference_rows_to_bimrange_edges() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "janusx-imp-bimrange-edges-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("target.vcf");
        let reference = root.join("reference.vcf");
        let output = root.join("output.vcf");
        let header = concat!(
            "##fileformat=VCFv4.3\n",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        );
        std::fs::write(
            &target,
            format!("{header}1\t200\tt1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n"),
        )
        .unwrap();
        std::fs::write(
            &reference,
            format!(
                "{header}\
                 1\t100\tr1\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|1\n\
                 1\t200\tr2\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n\
                 1\t300\tr3\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|0\n"
            ),
        )
        .unwrap();
        let config = ImpConfig {
            target,
            reference: Some(reference),
            output: output.clone(),
            threads: 1,
            window_markers: 20,
            phase_states: 4,
            imp_states: 4,
            burnin: 1,
            iterations: 1,
            em: false,
            impute: true,
            region: Some(types::RegionSpec {
                chrom: "1".to_owned(),
                start: Some(100),
                end: Some(300),
            }),
            ..ImpConfig::default()
        };

        let summary = run_imp(config).unwrap();
        let text = std::fs::read_to_string(&output).unwrap();
        let positions = text
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| line.split('\t').nth(1).unwrap().parse::<u64>().unwrap())
            .collect::<Vec<_>>();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(positions, vec![100, 200, 300]);
        assert_eq!(summary.variants_written, 3);
    }

    #[test]
    fn homozygous_missing_vcf_without_reference_uses_target_haplotype_imputation() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "janusx-imp-no-reference-missing-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("target.vcf");
        let output = root.join("output.vcf");
        std::fs::write(
            &target,
            concat!(
                "##fileformat=VCFv4.3\n",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4\ts5\ts6\n",
                "1\t100\tv1\tA\tG\t.\tPASS\t.\tGT\t0/0\t./.\t1/1\t./.\t0/0\t1/1\n",
                "1\t200\tv2\tC\tT\t.\tPASS\t.\tGT\t./.\t1/1\t0/0\t./.\t1/1\t0/0\n",
                "1\t300\tv3\tG\tA\t.\tPASS\t.\tGT\t0/0\t./.\t1/1\t1/1\t./.\t0/0\n",
                "1\t400\tv4\tT\tC\t.\tPASS\t.\tGT\t1/1\t0/0\t./.\t0/0\t1/1\t./.\n",
                "1\t500\tv5\tA\tC\t.\tPASS\t.\tGT\t0/0\t1/1\t0/0\t./.\t./.\t1/1\n",
                "1\t600\tv6\tG\tT\t.\tPASS\t.\tGT\t1/1\t./.\t1/1\t0/0\t0/0\t./.\n"
            ),
        )
        .unwrap();
        let config = ImpConfig {
            target,
            output: output.clone(),
            threads: 1,
            window_markers: 20,
            phase_states: 8,
            imp_states: 8,
            burnin: 1,
            iterations: 1,
            em: false,
            impute: true,
            ..ImpConfig::default()
        };

        let summary = run_imp(config).unwrap();
        let text = std::fs::read_to_string(&output).unwrap();
        let data_rows = text
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>();
        let positions = data_rows
            .iter()
            .map(|line| line.split('\t').nth(1).unwrap().parse::<u64>().unwrap())
            .collect::<Vec<_>>();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(summary.total_target_windows, 1);
        assert_eq!(summary.direct_imputation_windows, 0);
        assert_eq!(summary.normal_phasing_windows, 1);
        assert_eq!(positions, vec![100, 200, 300, 400, 500, 600]);
        assert!(data_rows.iter().all(|line| {
            line.split('\t')
                .skip(9)
                .all(|sample| !sample.split(':').next().unwrap().contains('.'))
        }));
    }

    #[test]
    fn homozygous_missing_vcf_uses_reference_haplotypes_during_phasing() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "janusx-imp-missing-scaffold-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("target.vcf");
        let reference = root.join("reference.vcf");
        let output = root.join("output.vcf");
        let header = concat!(
            "##fileformat=VCFv4.3\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4\n"
        );
        std::fs::write(
            &target,
            format!(
                "{header}\
                 1\t100\tt1\tA\tG\t.\tPASS\t.\tGT\t0/0\t./.\t1/1\t./.\n\
                 1\t600\tt2\tA\tG\t.\tPASS\t.\tGT\t./.\t1/1\t0/0\t./.\n\
                 1\t1000\tt3\tA\tG\t.\tPASS\t.\tGT\t0/0\t./.\t1/1\t./.\n"
            ),
        )
        .unwrap();
        std::fs::write(
            &reference,
            format!(
                "{header}\
                 1\t100\tr1\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1\t1|0\n\
                 1\t500\tr2\tA\tG\t.\tPASS\t.\tGT\t0|0\t1|0\t1|1\t0|1\n\
                 1\t600\tr3\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|1\t0|0\t0|1\n\
                 1\t700\tr4\tA\tG\t.\tPASS\t.\tGT\t1|0\t0|1\t1|0\t0|0\n\
                 1\t1000\tr5\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1\t1|0\n"
            ),
        )
        .unwrap();
        let config = ImpConfig {
            target,
            reference: Some(reference),
            output: output.clone(),
            threads: 1,
            window_markers: 20,
            phase_states: 32,
            imp_states: 8,
            burnin: 1,
            iterations: 1,
            em: false,
            impute: true,
            ..ImpConfig::default()
        };

        let summary = run_imp(config).unwrap();
        let text = std::fs::read_to_string(&output).unwrap();
        let data_rows = text
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>();
        let positions = data_rows
            .iter()
            .map(|line| line.split('\t').nth(1).unwrap().parse::<u64>().unwrap())
            .collect::<Vec<_>>();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(summary.total_target_windows, 1);
        assert_eq!(summary.direct_imputation_windows, 0);
        assert_eq!(summary.normal_phasing_windows, 1);
        assert_eq!(summary.direct_bypass_fraction, 0.0);
        assert_eq!(positions, vec![100, 500, 600, 700, 1000]);
        assert!(data_rows.iter().all(|line| {
            line.split('\t')
                .skip(9)
                .all(|sample| !sample.split(':').next().unwrap().contains('.'))
        }));
    }
}
