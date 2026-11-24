//! Isolated whole-chromosome PBWT IBD segmentation & multi-parent ancestry
//! painting module (`--only-ibd`).
//!
//! Deliberately isolated from `src/imp` so imputation memory/runtime
//! optimizations remain untouched. Produces compact Rust-native sparse binary
//! archives (`JHIBD01`) designed for downstream `src/hgwas` computation.

pub mod ancestry;
pub mod ibd;
pub mod pbwt;
pub mod phasing;
pub mod post;
pub mod sparse_io;
pub mod types;

use crate::imp::types::{PhaseSetId, VariantRow};
use crate::imp_reader::ImpReader;
use ancestry::{paint_chromosome_ancestry_cancellable, ChromosomeAncestryBinsSparse};
use ibd::{build_sparse_ibd_kinship_csr, detect_chromosome_ibd_cancellable};
use phasing::{phase_unphased_diploids_one_pass, DisjointPhaseBlock, SampleHaplotypeLayout};
use sparse_io::{
    write_segments_tsv_gz_parallel, write_sparse_archive_with_threads, HaploidSparseArchive,
    HaploidSparsePayload,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use rayon::prelude::*;
use types::{
    AncestrySegment, BitHaplotypeMatrix, GeneticMap, HaploidIbdConfig, HaploidIbdMode,
    HaploidIbdProgress, HaploidIbdSummary, IbdSegment, PloidyMode,
};

#[inline]
fn is_cancelled(cancelled: Option<&AtomicBool>) -> bool {
    cancelled
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

/// Main entry point for `jx ibd` without a progress callback.
pub fn run_haploid_ibd(config: HaploidIbdConfig) -> Result<HaploidIbdSummary, String> {
    run_haploid_ibd_with_progress(config, |_| Ok(()))
}

/// Main entry point for `jx ibd` with stage-level progress callbacks.
pub fn run_haploid_ibd_with_progress<F>(
    config: HaploidIbdConfig,
    on_progress: F,
) -> Result<HaploidIbdSummary, String>
where
    F: FnMut(&HaploidIbdProgress) -> Result<(), String>,
{
    run_haploid_ibd_with_progress_and_cancel(config, on_progress, None)
}

/// Main entry point for `jx ibd` with stage-level progress callbacks and
/// cooperative cancellation (`AtomicBool`) checked inside long-running loops:
/// streams one chromosome at a time, filters to biallelic markers,
/// auto-classifies inbred/DH vs diploid samples, runs 1-pass PBWT phasing on
/// unphased diploids (and across disjoint VCF PS phase blocks), and dispatches
/// to either all-vs-all pairwise IBD (no `--ref`) or multi-parent ancestry
/// painting (with `--ref`).
pub fn run_haploid_ibd_with_progress_and_cancel<F>(
    config: HaploidIbdConfig,
    mut on_progress: F,
    cancelled: Option<&AtomicBool>,
) -> Result<HaploidIbdSummary, String>
where
    F: FnMut(&HaploidIbdProgress) -> Result<(), String>,
{
    let started = Instant::now();
    if config.target.as_os_str().is_empty() {
        return Err("--only-ibd requires a target genotype input".to_owned());
    }
    if config.output.as_os_str().is_empty() {
        return Err("jx ibd requires an output path (--out)".to_owned());
    }
    if config.ancestry_top_donors == 0 || config.ancestry_top_donors >= u32::MAX as usize {
        return Err(format!(
            "ancestry_top_donors must be in 1..{}, got {}",
            u32::MAX,
            config.ancestry_top_donors
        ));
    }
    if !config.max_mismatch_rate.is_finite()
        || config.max_mismatch_rate < 0.0
        || config.max_mismatch_rate >= 0.5
    {
        return Err(format!(
            "max_mismatch_rate must be in [0, 0.5), got {}",
            config.max_mismatch_rate
        ));
    }
    let threads = config.threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|err| format!("build haploid thread pool ({threads} threads): {err}"))?;

    let map = match &config.map {
        Some(path) => GeneticMap::from_plink_file(path, config.cm_per_mb)?,
        None => GeneticMap::constant(config.cm_per_mb),
    };

    let mut target_reader = ImpReader::open(&config.target)?;
    target_reader.set_vcf_parse_threads(threads)?;
    let target_sample_ids = target_reader.sample_ids().to_vec();
    if target_sample_ids.is_empty() {
        return Err("target genotype file contains no samples".to_owned());
    }

    let mode = if config.reference.is_some() {
        HaploidIbdMode::MultiParentAncestry
    } else {
        HaploidIbdMode::PairwiseIbd
    };

    let mut ref_reader_opt = if let Some(ref_path) = &config.reference {
        let mut r = ImpReader::open(ref_path)?;
        r.set_vcf_parse_threads(threads)?;
        Some(r)
    } else {
        None
    };
    let reference_sample_ids = ref_reader_opt
        .as_ref()
        .map(|r| r.sample_ids().to_vec())
        .unwrap_or_default();

    // Binary PLINK (BED/PGEN) encodings only represent biallelic calls, so they
    // can safely use `next_biallelic_reference_row()` (O(1) shape validation).
    // Text formats (VCF/HapMap) must use `next_row()` so out-of-range alleles
    // such as `0|2` on a biallelic site still fail validation.
    let target_fast_binary_read = is_binary_plink_input(&config.target);
    let mut target_pending: Option<VariantRow> = pool.install(|| {
        if target_fast_binary_read {
            target_reader.next_biallelic_reference_row()
        } else {
            target_reader.next_row()
        }
    })?;
    let ref_fast_binary_read = config
        .reference
        .as_deref()
        .map(is_binary_plink_input)
        .unwrap_or(false);
    let mut ref_pending: Option<VariantRow> = match ref_reader_opt.as_mut() {
        Some(r) => pool.install(|| {
            if ref_fast_binary_read {
                r.next_biallelic_reference_row()
            } else {
                r.next_row()
            }
        })?,
        None => None,
    };
    // Bounded-memory disk-backed reference chromosome spool: zero I/O when
    // chromosomes match order, and strictly O(1 chromosome) RAM when target
    // and reference chromosome orders differ or target has missing chroms.
    let mut ref_chrom_spool = RefChromSpool::new();

    let mut chrom_names: Vec<String> = Vec::new();
    let mut raw_target_markers_scanned: u64 = 0;
    let mut markers_processed: u64 = 0;
    let mut total_genome_cm: f64 = 0.0;
    let mut last_inbred_count: usize = 0;
    let mut last_diploid_count: usize = 0;

    let mut all_ibd_segments: Vec<IbdSegment> = Vec::new();
    let mut all_ancestry_segments: Vec<AncestrySegment> = Vec::new();
    let mut all_chrom_bins: Vec<ChromosomeAncestryBinsSparse> = Vec::new();

    let mut emit_progress = |stage: &str,
                             chrom: &str,
                             chrom_index: u64,
                             chrom_markers: u64,
                             raw_target_markers_done: u64,
                             markers_done: u64,
                             segments_found: u64| {
        if is_cancelled(cancelled) {
            return Err("Interrupted".to_owned());
        }
        on_progress(&HaploidIbdProgress {
            stage: stage.to_owned(),
            chrom: chrom.to_owned(),
            chrom_index,
            chrom_markers,
            raw_target_markers_done,
            markers_done,
            segments_found,
        })
    };

    emit_progress("prepare", "", 0, 0, 0, 0, 0)?;

    while let Some(first_row) = target_pending.take() {
        if is_cancelled(cancelled) {
            return Err("Interrupted".to_owned());
        }
        let chrom = first_row.meta.chrom.clone();
        let next_chrom_index = chrom_names.len() as u64 + 1;
        let cur_segs = match mode {
            HaploidIbdMode::PairwiseIbd => all_ibd_segments.len() as u64,
            HaploidIbdMode::MultiParentAncestry => all_ancestry_segments.len() as u64,
        };
        emit_progress(
            "load",
            &chrom,
            next_chrom_index,
            0,
            raw_target_markers_scanned,
            markers_processed,
            cur_segs,
        )?;

        let mut target_rows = Vec::new();
        let mut chrom_raw_target_rows: u64 = 1;
        if first_row.meta.alt_alleles.len() == 1 {
            target_rows.push(first_row);
        }
        pool.install(|| -> Result<(), String> {
            while let Some(next_row) = if target_fast_binary_read {
                target_reader.next_biallelic_reference_row()?
            } else {
                target_reader.next_row()?
            } {
                if (chrom_raw_target_rows & 4095) == 0 && is_cancelled(cancelled) {
                    return Err("Interrupted".to_owned());
                }
                if next_row.meta.chrom == chrom {
                    chrom_raw_target_rows += 1;
                    if next_row.meta.alt_alleles.len() == 1 {
                        target_rows.push(next_row);
                    }
                } else {
                    target_pending = Some(next_row);
                    break;
                }
            }
            Ok(())
        })?;
        if target_rows.is_empty() {
            raw_target_markers_scanned += chrom_raw_target_rows;
            continue;
        }

        match mode {
            HaploidIbdMode::PairwiseIbd => {
                let chrom_idx = chrom_names.len() as u16;
                let chrom_index = chrom_idx as u64 + 1;
                let chrom_markers = target_rows.len() as u64;
                chrom_names.push(chrom.clone());

                emit_progress(
                    "encode",
                    &chrom,
                    chrom_index,
                    chrom_markers,
                    raw_target_markers_scanned,
                    markers_processed,
                    all_ibd_segments.len() as u64,
                )?;
                let (is_inbred, unphased_hets, disjoint_ps_blocks) = pool.install(|| {
                    classify_and_collect_hets(
                        &target_rows,
                        target_sample_ids.len(),
                        config.ploidy_mode,
                        config.inbred_het_threshold,
                    )
                });
                last_inbred_count = is_inbred.iter().filter(|&&b| b).count();
                last_diploid_count = target_sample_ids.len().saturating_sub(last_inbred_count);
                let layout = SampleHaplotypeLayout::from_inbred_flags(is_inbred);
                let (mut matrix, positions_bp, cm_positions) = pool.install(|| {
                    encode_rows_to_matrix(&target_rows, &layout, &map)
                });
                drop(target_rows);

                if cm_positions.len() >= 2 {
                    total_genome_cm +=
                        (cm_positions[cm_positions.len() - 1] - cm_positions[0]).max(0.0);
                }

                if unphased_hets.iter().any(|v| !v.is_empty())
                    || disjoint_ps_blocks
                        .iter()
                        .any(|b| b.iter().any(|x| !x.all_phased_markers.is_empty()))
                {
                    emit_progress(
                        "phase",
                        &chrom,
                        chrom_index,
                        chrom_markers,
                        raw_target_markers_scanned,
                        markers_processed,
                        all_ibd_segments.len() as u64,
                    )?;
                    phase_unphased_diploids_one_pass(
                        &mut matrix,
                        &layout,
                        &unphased_hets,
                        &disjoint_ps_blocks,
                    );
                }

                emit_progress(
                    "ibd",
                    &chrom,
                    chrom_index,
                    chrom_markers,
                    raw_target_markers_scanned,
                    markers_processed,
                    all_ibd_segments.len() as u64,
                )?;
                let segs = pool.install(|| {
                    detect_chromosome_ibd_cancellable(
                        &matrix,
                        &layout,
                        chrom_idx,
                        &positions_bp,
                        &cm_positions,
                        &config,
                        cancelled,
                    )
                })?;
                all_ibd_segments.extend(segs);
                raw_target_markers_scanned += chrom_raw_target_rows;
                markers_processed += matrix.n_markers as u64;
                emit_progress(
                    "chrom_done",
                    &chrom,
                    chrom_index,
                    chrom_markers,
                    raw_target_markers_scanned,
                    markers_processed,
                    all_ibd_segments.len() as u64,
                )?;
            }
            HaploidIbdMode::MultiParentAncestry => {
                let ref_reader = ref_reader_opt
                    .as_mut()
                    .expect("reference reader present in ancestry mode");
                let ref_rows = pool.install(|| {
                    take_reference_chromosome_rows(
                        &chrom,
                        ref_reader,
                        &mut ref_pending,
                        &mut ref_chrom_spool,
                        ref_fast_binary_read,
                        cancelled,
                    )
                })?;
                if ref_rows.is_empty() {
                    raw_target_markers_scanned += chrom_raw_target_rows;
                    continue;
                }

                let aligned_pairs = pool.install(|| align_chromosome_row_indices(&ref_rows, &target_rows));
                if aligned_pairs.is_empty() {
                    drop(ref_rows);
                    drop(target_rows);
                    raw_target_markers_scanned += chrom_raw_target_rows;
                    continue;
                }

                let chrom_idx = chrom_names.len() as u16;
                let chrom_index = chrom_idx as u64 + 1;
                let chrom_markers = aligned_pairs.len() as u64;
                chrom_names.push(chrom.clone());

                emit_progress(
                    "encode",
                    &chrom,
                    chrom_index,
                    chrom_markers,
                    raw_target_markers_scanned,
                    markers_processed,
                    all_ancestry_segments.len() as u64,
                )?;
                let (ref_inbred, ref_unphased_hets, ref_ps_blocks) = pool.install(|| {
                    classify_and_collect_hets_with(
                        aligned_pairs.len(),
                        reference_sample_ids.len(),
                        config.ploidy_mode,
                        config.inbred_het_threshold,
                        |m| &ref_rows[aligned_pairs[m].0],
                    )
                });
                let (tgt_inbred, tgt_unphased_hets, tgt_ps_blocks) = pool.install(|| {
                    classify_and_collect_hets_with(
                        aligned_pairs.len(),
                        target_sample_ids.len(),
                        config.ploidy_mode,
                        config.inbred_het_threshold,
                        |m| &target_rows[aligned_pairs[m].1],
                    )
                });
                last_inbred_count = tgt_inbred.iter().filter(|&&b| b).count();
                last_diploid_count = target_sample_ids.len().saturating_sub(last_inbred_count);

                let ref_layout = SampleHaplotypeLayout::from_inbred_flags(ref_inbred);
                let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(tgt_inbred);
                let mut joint_inbred = ref_layout.is_inbred.clone();
                joint_inbred.extend_from_slice(&tgt_layout.is_inbred);
                let joint_layout = SampleHaplotypeLayout::from_inbred_flags(joint_inbred);

                let mut joint_unphased_hets = ref_unphased_hets;
                joint_unphased_hets.extend(tgt_unphased_hets);
                let mut joint_ps_blocks = ref_ps_blocks;
                joint_ps_blocks.extend(tgt_ps_blocks);

                let (mut joint_matrix, positions_bp, cm_positions) = pool.install(|| {
                    encode_joint_rows_to_matrix_indexed(
                        &ref_rows,
                        &target_rows,
                        &aligned_pairs,
                        &ref_layout,
                        &tgt_layout,
                        &map,
                    )
                });
                drop(ref_rows);
                drop(target_rows);

                if cm_positions.len() >= 2 {
                    total_genome_cm +=
                        (cm_positions[cm_positions.len() - 1] - cm_positions[0]).max(0.0);
                }

                if joint_unphased_hets.iter().any(|v| !v.is_empty())
                    || joint_ps_blocks
                        .iter()
                        .any(|b| b.iter().any(|x| !x.all_phased_markers.is_empty()))
                {
                    emit_progress(
                        "phase",
                        &chrom,
                        chrom_index,
                        chrom_markers,
                        raw_target_markers_scanned,
                        markers_processed,
                        all_ancestry_segments.len() as u64,
                    )?;
                    phase_unphased_diploids_one_pass(
                        &mut joint_matrix,
                        &joint_layout,
                        &joint_unphased_hets,
                        &joint_ps_blocks,
                    );
                }

                emit_progress(
                    "ancestry",
                    &chrom,
                    chrom_index,
                    chrom_markers,
                    raw_target_markers_scanned,
                    markers_processed,
                    all_ancestry_segments.len() as u64,
                )?;
                let (segs, bins) = pool.install(|| {
                    paint_chromosome_ancestry_cancellable(
                        &joint_matrix,
                        &ref_layout,
                        &tgt_layout,
                        chrom_idx,
                        &positions_bp,
                        &cm_positions,
                        &config,
                        cancelled,
                    )
                })?;
                all_ancestry_segments.extend(segs);
                all_chrom_bins.push(bins);
                raw_target_markers_scanned += chrom_raw_target_rows;
                markers_processed += joint_matrix.n_markers as u64;
                emit_progress(
                    "chrom_done",
                    &chrom,
                    chrom_index,
                    chrom_markers,
                    raw_target_markers_scanned,
                    markers_processed,
                    all_ancestry_segments.len() as u64,
                )?;
            }
        }
    }

    if mode == HaploidIbdMode::MultiParentAncestry && markers_processed == 0 {
        return Err(
            "no overlapping biallelic markers found between target and reference panel".to_owned(),
        );
    }

    let (segments_written, sparse_bins_written, sparse_nnz, payload) = match mode {
        HaploidIbdMode::PairwiseIbd => {
            emit_progress(
                "kinship",
                "",
                chrom_names.len() as u64,
                0,
                raw_target_markers_scanned,
                markers_processed,
                all_ibd_segments.len() as u64,
            )?;
            let kinship_csr = pool.install(|| {
                build_sparse_ibd_kinship_csr(
                    target_sample_ids.len(),
                    total_genome_cm,
                    &all_ibd_segments,
                )
            });
            let segs_n = all_ibd_segments.len() as u64;
            let nnz = kinship_csr.col_idx.len() as u64;
            (
                segs_n,
                0u64,
                nnz,
                HaploidSparsePayload::PairwiseIbd {
                    segments: all_ibd_segments,
                    kinship_csr,
                },
            )
        }
        HaploidIbdMode::MultiParentAncestry => {
            let segs_n = all_ancestry_segments.len() as u64;
            let bins_n: u64 = all_chrom_bins
                .iter()
                .map(|b| b.bin_starts_bp.len() as u64)
                .sum();
            let nnz: u64 = all_chrom_bins.iter().map(|b| b.dosage.len() as u64).sum();
            (
                segs_n,
                bins_n,
                nnz,
                HaploidSparsePayload::MultiParentAncestry {
                    segments: all_ancestry_segments,
                    chrom_bins: all_chrom_bins,
                },
            )
        }
    };

    let archive = HaploidSparseArchive {
        version: 1,
        mode,
        target_sample_ids: target_sample_ids.clone(),
        reference_sample_ids: reference_sample_ids.clone(),
        chrom_names: chrom_names.clone(),
        payload,
    };

    emit_progress(
        "write",
        "",
        chrom_names.len() as u64,
        0,
        raw_target_markers_scanned,
        markers_processed,
        segments_written,
    )?;
    let output_path = normalize_jhibd_output_path(&config.output);
    if is_cancelled(cancelled) {
        return Err("Interrupted".to_owned());
    }
    pool.install(|| write_sparse_archive_with_threads(&output_path, &archive, threads))?;

    let tsv_path = if config.write_tsv {
        let mut p = output_path.clone();
        let file_name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("haploid_ibd");
        p.set_file_name(format!("{file_name}.tsv.gz"));
        pool.install(|| write_segments_tsv_gz_parallel(&p, &archive, threads, cancelled))?;
        Some(p.to_string_lossy().into_owned())
    } else {
        None
    };

    emit_progress(
        "complete",
        "",
        chrom_names.len() as u64,
        0,
        raw_target_markers_scanned,
        markers_processed,
        segments_written,
    )?;

    Ok(HaploidIbdSummary {
        mode,
        chromosomes: chrom_names.len(),
        markers_processed,
        target_samples: target_sample_ids.len(),
        reference_samples: reference_sample_ids.len(),
        inbred_samples: last_inbred_count,
        diploid_samples: last_diploid_count,
        segments_written,
        sparse_bins_written,
        sparse_nnz,
        output_path: output_path.to_string_lossy().into_owned(),
        tsv_path,
        elapsed_s: started.elapsed().as_secs_f64(),
    })
}

static SPOOL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Disk-backed per-chromosome reference spool that keeps peak RAM strictly
/// bounded to a single chromosome even when target and reference files have
/// different chromosome orders or target contains chromosomes absent from ref.
///
/// Uses explicit `Option<VariantRow>` framing (`None` end-of-stream sentinel)
/// plus an expected row count so truncated or corrupted spool files always fail
/// with an explicit error rather than silently returning partial rows.
struct RefChromSpool {
    files: HashMap<String, (PathBuf, u64)>,
}

impl RefChromSpool {
    fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    fn take(&mut self, chrom: &str) -> Result<Option<Vec<VariantRow>>, String> {
        let Some((path, expected_count)) = self.files.remove(chrom) else {
            return Ok(None);
        };
        let file = File::open(&path)
            .map_err(|err| format!("open reference spool {}: {err}", path.display()))?;
        let mut decoder = zstd::stream::Decoder::new(BufReader::new(file))
            .map_err(|err| format!("init reference spool decoder {}: {err}", path.display()))?;
        let mut rows = Vec::with_capacity(expected_count as usize);
        loop {
            match bincode::deserialize_from::<_, Option<VariantRow>>(&mut decoder) {
                Ok(Some(row)) => rows.push(row),
                Ok(None) => break,
                Err(err) => {
                    let _ = std::fs::remove_file(&path);
                    return Err(format!(
                        "truncated or corrupted reference spool {}: {err}",
                        path.display()
                    ));
                }
            }
        }
        let _ = std::fs::remove_file(&path);
        if rows.len() as u64 != expected_count {
            return Err(format!(
                "reference spool {} row count mismatch: expected {expected_count}, read {}",
                path.display(),
                rows.len()
            ));
        }
        Ok(Some(rows))
    }

    fn create_writer(
        &mut self,
        chrom: &str,
    ) -> Result<(PathBuf, zstd::stream::Encoder<'static, BufWriter<File>>), String> {
        let id = SPOOL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "janusx_haploid_ref_spool_{}_{}.zst",
            std::process::id(),
            id
        ));
        let file = File::create(&path)
            .map_err(|err| format!("create reference spool {}: {err}", path.display()))?;
        let encoder = zstd::stream::Encoder::new(BufWriter::new(file), 1)
            .map_err(|err| format!("init reference spool encoder {}: {err}", path.display()))?;
        self.files.insert(chrom.to_owned(), (path.clone(), 0));
        Ok((path, encoder))
    }

    fn record_count(&mut self, chrom: &str, count: u64) {
        if let Some(entry) = self.files.get_mut(chrom) {
            entry.1 = count;
        }
    }
}

impl Drop for RefChromSpool {
    fn drop(&mut self) {
        for (path, _) in self.files.values() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Fetch all biallelic reference rows for `target_chrom` without assuming
/// lexicographical chromosome ordering between target and reference files.
/// Non-matching chromosomes are streamed directly to a compressed temp spool
/// row-by-row so peak RAM never exceeds a single chromosome.
fn take_reference_chromosome_rows(
    target_chrom: &str,
    ref_reader: &mut ImpReader,
    ref_pending: &mut Option<VariantRow>,
    spool: &mut RefChromSpool,
    fast_binary_read: bool,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<VariantRow>, String> {
    if is_cancelled(cancelled) {
        return Err("Interrupted".to_owned());
    }
    if let Some(rows) = spool.take(target_chrom)? {
        return Ok(rows);
    }
    while let Some(first_row) = ref_pending.take() {
        if is_cancelled(cancelled) {
            return Err("Interrupted".to_owned());
        }
        let cur_chrom = first_row.meta.chrom.clone();
        if cur_chrom == target_chrom {
            let mut rows = Vec::new();
            let mut seen = 1u64;
            if first_row.meta.alt_alleles.len() == 1 {
                rows.push(first_row);
            }
            while let Some(next_row) = if fast_binary_read {
                ref_reader.next_biallelic_reference_row()?
            } else {
                ref_reader.next_row()?
            } {
                seen += 1;
                if (seen & 4095) == 0 && is_cancelled(cancelled) {
                    return Err("Interrupted".to_owned());
                }
                if next_row.meta.chrom == cur_chrom {
                    if next_row.meta.alt_alleles.len() == 1 {
                        rows.push(next_row);
                    }
                } else {
                    *ref_pending = Some(next_row);
                    break;
                }
            }
            return Ok(rows);
        }
        let (path, mut encoder) = spool.create_writer(&cur_chrom)?;
        let mut count = 0u64;
        let mut seen = 1u64;
        if first_row.meta.alt_alleles.len() == 1 {
            bincode::serialize_into(&mut encoder, &Some(&first_row))
                .map_err(|err| format!("write reference spool {}: {err}", path.display()))?;
            count += 1;
        }
        drop(first_row);
        while let Some(next_row) = if fast_binary_read {
            ref_reader.next_biallelic_reference_row()?
        } else {
            ref_reader.next_row()?
        } {
            seen += 1;
            if (seen & 4095) == 0 && is_cancelled(cancelled) {
                return Err("Interrupted".to_owned());
            }
            if next_row.meta.chrom == cur_chrom {
                if next_row.meta.alt_alleles.len() == 1 {
                    bincode::serialize_into(&mut encoder, &Some(&next_row)).map_err(|err| {
                        format!("write reference spool {}: {err}", path.display())
                    })?;
                    count += 1;
                }
            } else {
                *ref_pending = Some(next_row);
                break;
            }
        }
        bincode::serialize_into(&mut encoder, &None::<VariantRow>)
            .map_err(|err| format!("write reference spool sentinel {}: {err}", path.display()))?;
        let mut writer = encoder
            .finish()
            .map_err(|err| format!("finish reference spool {}: {err}", path.display()))?;
        writer
            .flush()
            .map_err(|err| format!("flush reference spool {}: {err}", path.display()))?;
        spool.record_count(&cur_chrom, count);
    }
    Ok(Vec::new())
}

fn is_binary_plink_input(path: &std::path::Path) -> bool {
    let path_str = path.to_string_lossy();
    let lower = path_str.to_ascii_lowercase();
    if lower.ends_with(".vcf")
        || lower.ends_with(".vcf.gz")
        || lower.ends_with(".hmp")
        || lower.ends_with(".hmp.gz")
        || lower.ends_with(".hmp.txt")
        || lower.ends_with(".hmp.txt.gz")
    {
        return false;
    }
    crate::pgen::path_uses_pgen(&path_str)
        || lower.ends_with(".bed")
        || lower.ends_with(".bim")
        || lower.ends_with(".fam")
}

fn normalize_jhibd_output_path(path: &std::path::Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s.ends_with(".jhibd") {
        path.to_path_buf()
    } else {
        PathBuf::from(format!("{s}.jhibd"))
    }
}

/// Classify samples as inbred/DH vs diploid and collect diploid heterozygotes
/// that need 1-pass PBWT orientation:
/// - only biallelic calls (`0` or `1`) are counted (alleles `> 1` are ignored);
/// - unphased heterozygotes (`call.phased == false`) are queued individually;
/// - phased calls belonging to a subsequent VCF `PS` phase block (including both
///   complete heterozygotes and partial-missing phased calls like `0|.`) are
///   grouped by `PS` ID into `DisjointPhaseBlock`s so the entire block flips
///   atomically around its first complete heterozygote (preserving intra-block phase).
fn classify_and_collect_hets(
    rows: &[VariantRow],
    n_samples: usize,
    ploidy_mode: PloidyMode,
    inbred_het_threshold: f64,
) -> (Vec<bool>, Vec<Vec<u32>>, Vec<Vec<DisjointPhaseBlock>>) {
    classify_and_collect_hets_with(
        rows.len(),
        n_samples,
        ploidy_mode,
        inbred_het_threshold,
        |m| &rows[m],
    )
}

fn classify_and_collect_hets_with<'a, F>(
    n_markers: usize,
    n_samples: usize,
    ploidy_mode: PloidyMode,
    inbred_het_threshold: f64,
    row_at: F,
) -> (Vec<bool>, Vec<Vec<u32>>, Vec<Vec<DisjointPhaseBlock>>)
where
    F: Fn(usize) -> &'a VariantRow + Sync,
{
    const CHUNK_MARKERS: usize = 2048;
    let n_chunks = n_markers.div_ceil(CHUNK_MARKERS).max(1);
    let chunk_results: Vec<(
        Vec<u64>,
        Vec<u64>,
        Vec<Vec<u32>>,
        Vec<Vec<(u32, PhaseSetId, bool)>>,
    )> = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let m_start = chunk_idx * CHUNK_MARKERS;
            let m_end = (m_start + CHUNK_MARKERS).min(n_markers);
            let mut local_called = vec![0u64; n_samples];
            let mut local_het = vec![0u64; n_samples];
            let mut local_unphased = vec![Vec::<u32>::new(); n_samples];
            let mut local_ps_events = vec![Vec::<(u32, PhaseSetId, bool)>::new(); n_samples];

            for marker in m_start..m_end {
                let row = row_at(marker);
                let has_ps = row.calls.has_phase_set_track();
                let marker_u32 = marker as u32;
                for s in 0..n_samples {
                    if let Some(call) = row.calls.get(s) {
                        let is_complete_biallelic = matches!(
                            call.alleles,
                            [Some(a0), Some(a1)] if a0 <= 1 && a1 <= 1
                        );
                        let is_complete_het = matches!(
                            call.alleles,
                            [Some(a0), Some(a1)] if a0 <= 1 && a1 <= 1 && a0 != a1
                        );
                        if is_complete_biallelic {
                            local_called[s] += 1;
                            if is_complete_het {
                                local_het[s] += 1;
                                if !call.phased {
                                    local_unphased[s].push(marker_u32);
                                }
                            }
                        }
                        if has_ps && call.phased {
                            let is_partial_phased = matches!(
                                call.alleles,
                                [Some(a), None] | [None, Some(a)] if a <= 1
                            );
                            if is_complete_het || is_partial_phased {
                                if let Some(ps_id) = row.calls.phase_set(s) {
                                    local_ps_events[s].push((marker_u32, ps_id, is_complete_het));
                                }
                            }
                        }
                    }
                }
            }
            (local_called, local_het, local_unphased, local_ps_events)
        })
        .collect();

    let mut called = vec![0u64; n_samples];
    let mut het = vec![0u64; n_samples];
    let mut unphased_hets = vec![Vec::new(); n_samples];
    // (optional first complete het anchor, all phased markers in block)
    let mut raw_ps_blocks: Vec<Vec<(Option<u32>, Vec<u32>)>> = vec![Vec::new(); n_samples];
    let mut first_ps: Vec<Option<PhaseSetId>> = vec![None; n_samples];
    let mut ps_block_map: Vec<HashMap<PhaseSetId, usize>> = vec![HashMap::new(); n_samples];

    for (local_called, local_het, local_unphased, local_ps_events) in chunk_results {
        for s in 0..n_samples {
            called[s] += local_called[s];
            het[s] += local_het[s];
            if !local_unphased[s].is_empty() {
                unphased_hets[s].extend_from_slice(&local_unphased[s]);
            }
            for &(marker_u32, ps_id, is_complete_het) in &local_ps_events[s] {
                match first_ps[s] {
                    None => {
                        first_ps[s] = Some(ps_id);
                    }
                    Some(first) if first == ps_id => {}
                    Some(_) => {
                        let block_idx = *ps_block_map[s].entry(ps_id).or_insert_with(|| {
                            let idx = raw_ps_blocks[s].len();
                            raw_ps_blocks[s].push((None, Vec::new()));
                            idx
                        });
                        let entry = &mut raw_ps_blocks[s][block_idx];
                        if is_complete_het && entry.0.is_none() {
                            entry.0 = Some(marker_u32);
                        }
                        entry.1.push(marker_u32);
                    }
                }
            }
        }
    }

    let is_inbred: Vec<bool> = (0..n_samples)
        .map(|s| match ploidy_mode {
            PloidyMode::Inbred => true,
            PloidyMode::Diploid => false,
            PloidyMode::Auto => {
                if called[s] == 0 {
                    true
                } else {
                    (het[s] as f64 / called[s] as f64) <= inbred_het_threshold
                }
            }
        })
        .collect();

    let mut disjoint_ps_blocks: Vec<Vec<DisjointPhaseBlock>> = raw_ps_blocks
        .into_iter()
        .map(|blocks| {
            let mut out: Vec<DisjointPhaseBlock> = blocks
                .into_iter()
                .filter_map(|(first_het_marker, markers)| {
                    // Always anchor on the earliest phased marker in the block so
                    // PBWT never advances past any block marker before the block
                    // orientation decision is applied.
                    let anchor_marker = markers.first().copied()?;
                    Some(DisjointPhaseBlock {
                        anchor_marker,
                        first_het_marker,
                        all_phased_markers: markers,
                    })
                })
                .collect();
            out.sort_by_key(|b| b.anchor_marker);
            out
        })
        .collect();

    for (s, &inbred) in is_inbred.iter().enumerate() {
        if inbred {
            unphased_hets[s].clear();
            disjoint_ps_blocks[s].clear();
        }
    }
    (is_inbred, unphased_hets, disjoint_ps_blocks)
}

#[inline]
fn biallelic_code(allele: Option<u16>) -> Option<u8> {
    match allele {
        Some(0) => Some(0),
        Some(1) => Some(1),
        _ => None,
    }
}

#[inline]
fn set_bit_in_row(allele_row: &mut [u64], missing_row: &mut [u64], hap: usize, val: Option<u8>) {
    let word = hap >> 6;
    let mask = 1u64 << (hap & 63);
    match val {
        Some(0) => {}
        Some(_) => allele_row[word] |= mask,
        None => missing_row[word] |= mask,
    }
}

fn encode_rows_to_matrix(
    rows: &[VariantRow],
    layout: &SampleHaplotypeLayout,
    map: &GeneticMap,
) -> (BitHaplotypeMatrix, Vec<u64>, Vec<f64>) {
    let n_markers = rows.len();
    let mut matrix = BitHaplotypeMatrix::new(layout.total_haps, n_markers);
    let positions_bp: Vec<u64> = rows.par_iter().map(|row| row.meta.pos).collect();
    let cm_positions: Vec<f64> = rows
        .par_iter()
        .map(|row| map.interpolate_cm(&row.meta.chrom, row.meta.pos))
        .collect();
    let stride = matrix.words_per_marker;
    if stride > 0 {
        matrix
            .allele_bits
            .par_chunks_mut(stride)
            .zip(matrix.missing_bits.par_chunks_mut(stride))
            .zip(rows.par_iter())
            .for_each(|((allele_row, missing_row), row)| {
                for s in 0..layout.n_samples {
                    let off = layout.hap_offset[s];
                    let call = row.calls.get(s);
                    if layout.is_inbred[s] {
                        let val = call.and_then(|c| match c.alleles {
                            [Some(a0), Some(a1)] if a0 == a1 && a0 <= 1 => Some(a0 as u8),
                            _ => None,
                        });
                        set_bit_in_row(allele_row, missing_row, off, val);
                    } else {
                        match call {
                            Some(c) => {
                                set_bit_in_row(
                                    allele_row,
                                    missing_row,
                                    off,
                                    biallelic_code(c.alleles[0]),
                                );
                                set_bit_in_row(
                                    allele_row,
                                    missing_row,
                                    off + 1,
                                    biallelic_code(c.alleles[1]),
                                );
                            }
                            None => {
                                set_bit_in_row(allele_row, missing_row, off, None);
                                set_bit_in_row(allele_row, missing_row, off + 1, None);
                            }
                        }
                    }
                }
            });
    }
    (matrix, positions_bp, cm_positions)
}

fn encode_joint_rows_to_matrix_indexed(
    ref_rows: &[VariantRow],
    tgt_rows: &[VariantRow],
    aligned_pairs: &[(usize, usize)],
    ref_layout: &SampleHaplotypeLayout,
    tgt_layout: &SampleHaplotypeLayout,
    map: &GeneticMap,
) -> (BitHaplotypeMatrix, Vec<u64>, Vec<f64>) {
    let total_haps = ref_layout.total_haps + tgt_layout.total_haps;
    let n_markers = aligned_pairs.len();
    let mut matrix = BitHaplotypeMatrix::new(total_haps, n_markers);
    let positions_bp: Vec<u64> = aligned_pairs
        .par_iter()
        .map(|&(_, tidx)| tgt_rows[tidx].meta.pos)
        .collect();
    let cm_positions: Vec<f64> = aligned_pairs
        .par_iter()
        .map(|&(_, tidx)| {
            let row = &tgt_rows[tidx];
            map.interpolate_cm(&row.meta.chrom, row.meta.pos)
        })
        .collect();
    let stride = matrix.words_per_marker;
    let base = ref_layout.total_haps;
    if stride > 0 {
        matrix
            .allele_bits
            .par_chunks_mut(stride)
            .zip(matrix.missing_bits.par_chunks_mut(stride))
            .zip(aligned_pairs.par_iter())
            .for_each(|((allele_row, missing_row), &(ridx, tidx))| {
                let rrow = &ref_rows[ridx];
                let trow = &tgt_rows[tidx];
                for s in 0..ref_layout.n_samples {
                    let off = ref_layout.hap_offset[s];
                    let call = rrow.calls.get(s);
                    if ref_layout.is_inbred[s] {
                        let val = call.and_then(|c| match c.alleles {
                            [Some(a0), Some(a1)] if a0 == a1 && a0 <= 1 => Some(a0 as u8),
                            _ => None,
                        });
                        set_bit_in_row(allele_row, missing_row, off, val);
                    } else {
                        match call {
                            Some(c) => {
                                set_bit_in_row(
                                    allele_row,
                                    missing_row,
                                    off,
                                    biallelic_code(c.alleles[0]),
                                );
                                set_bit_in_row(
                                    allele_row,
                                    missing_row,
                                    off + 1,
                                    biallelic_code(c.alleles[1]),
                                );
                            }
                            None => {
                                set_bit_in_row(allele_row, missing_row, off, None);
                                set_bit_in_row(allele_row, missing_row, off + 1, None);
                            }
                        }
                    }
                }
                for s in 0..tgt_layout.n_samples {
                    let off = base + tgt_layout.hap_offset[s];
                    let call = trow.calls.get(s);
                    if tgt_layout.is_inbred[s] {
                        let val = call.and_then(|c| match c.alleles {
                            [Some(a0), Some(a1)] if a0 == a1 && a0 <= 1 => Some(a0 as u8),
                            _ => None,
                        });
                        set_bit_in_row(allele_row, missing_row, off, val);
                    } else {
                        match call {
                            Some(c) => {
                                set_bit_in_row(
                                    allele_row,
                                    missing_row,
                                    off,
                                    biallelic_code(c.alleles[0]),
                                );
                                set_bit_in_row(
                                    allele_row,
                                    missing_row,
                                    off + 1,
                                    biallelic_code(c.alleles[1]),
                                );
                            }
                            None => {
                                set_bit_in_row(allele_row, missing_row, off, None);
                                set_bit_in_row(allele_row, missing_row, off + 1, None);
                            }
                        }
                    }
                }
            });
    }
    (matrix, positions_bp, cm_positions)
}

fn align_chromosome_row_indices(
    ref_rows: &[VariantRow],
    tgt_rows: &[VariantRow],
) -> Vec<(usize, usize)> {
    if ref_rows.is_empty() || tgt_rows.is_empty() {
        return Vec::new();
    }
    let ref_sorted = ref_rows.par_windows(2).all(|w| w[0].meta.pos <= w[1].meta.pos);
    let tgt_sorted = tgt_rows.par_windows(2).all(|w| w[0].meta.pos <= w[1].meta.pos);
    if ref_sorted && tgt_sorted {
        let mut aligned = Vec::with_capacity(ref_rows.len().min(tgt_rows.len()));
        let mut ri = 0usize;
        let mut ti = 0usize;
        while ri < ref_rows.len() && ti < tgt_rows.len() {
            let rpos = ref_rows[ri].meta.pos;
            let tpos = tgt_rows[ti].meta.pos;
            if rpos < tpos {
                ri += 1;
            } else if rpos > tpos {
                ti += 1;
            } else {
                let mut r_end = ri + 1;
                while r_end < ref_rows.len() && ref_rows[r_end].meta.pos == rpos {
                    r_end += 1;
                }
                let mut t_end = ti + 1;
                while t_end < tgt_rows.len() && tgt_rows[t_end].meta.pos == tpos {
                    t_end += 1;
                }
                if r_end == ri + 1 && t_end == ti + 1 {
                    let rrow = &ref_rows[ri];
                    let trow = &tgt_rows[ti];
                    let r_alt0 = rrow.meta.alt_alleles.first().map(String::as_str).unwrap_or("");
                    let t_alt0 = trow.meta.alt_alleles.first().map(String::as_str).unwrap_or("");
                    if rrow.meta.ref_allele == trow.meta.ref_allele && r_alt0 == t_alt0 {
                        aligned.push((ri, ti));
                    }
                } else if (r_end - ri).saturating_mul(t_end - ti) > 64 {
                    let mut local_ref: HashMap<(&str, &str), usize> =
                        HashMap::with_capacity(r_end - ri);
                    for ridx in ri..r_end {
                        let rrow = &ref_rows[ridx];
                        let r_alt0 = rrow.meta.alt_alleles.first().map(String::as_str).unwrap_or("");
                        local_ref.insert((rrow.meta.ref_allele.as_str(), r_alt0), ridx);
                    }
                    for tidx in ti..t_end {
                        let trow = &tgt_rows[tidx];
                        let t_alt0 = trow.meta.alt_alleles.first().map(String::as_str).unwrap_or("");
                        if let Some(&ridx) = local_ref.get(&(trow.meta.ref_allele.as_str(), t_alt0)) {
                            aligned.push((ridx, tidx));
                        }
                    }
                } else {
                    for tidx in ti..t_end {
                        let trow = &tgt_rows[tidx];
                        let t_alt0 = trow.meta.alt_alleles.first().map(String::as_str).unwrap_or("");
                        // Match HashMap::insert semantics: last matching reference row wins.
                        for ridx in (ri..r_end).rev() {
                            let rrow = &ref_rows[ridx];
                            let r_alt0 =
                                rrow.meta.alt_alleles.first().map(String::as_str).unwrap_or("");
                            if rrow.meta.ref_allele == trow.meta.ref_allele && r_alt0 == t_alt0 {
                                aligned.push((ridx, tidx));
                                break;
                            }
                        }
                    }
                }
                ri = r_end;
                ti = t_end;
            }
        }
        return aligned;
    }

    let mut ref_index: HashMap<(u64, &str, &str), usize> = HashMap::with_capacity(ref_rows.len());
    for (idx, row) in ref_rows.iter().enumerate() {
        let alt0 = row.meta.alt_alleles.first().map(String::as_str).unwrap_or("");
        ref_index.insert((row.meta.pos, row.meta.ref_allele.as_str(), alt0), idx);
    }
    tgt_rows
        .par_iter()
        .enumerate()
        .filter_map(|(tidx, trow)| {
            let alt0 = trow.meta.alt_alleles.first().map(String::as_str).unwrap_or("");
            ref_index
                .get(&(trow.meta.pos, trow.meta.ref_allele.as_str(), alt0))
                .map(|&ridx| (ridx, tidx))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{classify_and_collect_hets, DisjointPhaseBlock, PloidyMode, RefChromSpool};
    use crate::imp::types::{DiploidCall, DiploidCalls, PhaseSetId, VariantMeta, VariantRow};
    use std::io::Write;

    #[test]
    fn multiallelic_1_2_is_ignored_and_ps_blocks_include_partial_missing_phased_calls() {
        let meta = VariantMeta {
            chrom: "chr1".to_owned(),
            pos: 100,
            id: "v1".to_owned(),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let mut calls0 = DiploidCalls::from_calls(vec![DiploidCall {
            alleles: [Some(0), Some(1)],
            phased: true,
        }]);
        calls0.set_phase_set(0, Some(PhaseSetId::new(1))).unwrap();

        let mut calls1 = DiploidCalls::from_calls(vec![DiploidCall {
            alleles: [Some(0), None],
            phased: true,
        }]);
        calls1.set_phase_set(0, Some(PhaseSetId::new(2))).unwrap();

        let mut calls2 = DiploidCalls::from_calls(vec![DiploidCall {
            alleles: [Some(0), Some(1)],
            phased: true,
        }]);
        calls2.set_phase_set(0, Some(PhaseSetId::new(2))).unwrap();

        let calls3 = DiploidCalls::from_calls(vec![DiploidCall {
            alleles: [Some(1), Some(2)],
            phased: false,
        }]);

        let rows = vec![
            VariantRow {
                meta: meta.clone(),
                calls: calls0,
            },
            VariantRow {
                meta: meta.clone(),
                calls: calls1,
            },
            VariantRow {
                meta: meta.clone(),
                calls: calls2,
            },
            VariantRow {
                meta,
                calls: calls3,
            },
        ];

        let (is_inbred, unphased, ps_blocks) =
            classify_and_collect_hets(&rows, 1, PloidyMode::Diploid, 0.02);
        assert_eq!(is_inbred, vec![false]);
        assert!(unphased[0].is_empty());
        // PS=2 anchors on earliest phased marker 1 (with first_het_marker=Some(2))
        // so PBWT never advances past marker 1 before the block orientation is decided.
        assert_eq!(
            ps_blocks[0],
            vec![DisjointPhaseBlock {
                anchor_marker: 1,
                first_het_marker: Some(2),
                all_phased_markers: vec![1, 2],
            }]
        );
    }

    #[test]
    fn truncated_reference_spool_returns_error() {
        let mut spool = RefChromSpool::new();
        let (path, mut encoder) = spool.create_writer("chr2").unwrap();
        for i in 0..100u64 {
            let row = VariantRow {
                meta: VariantMeta {
                    chrom: "chr2".to_owned(),
                    pos: (i + 1) * 100,
                    id: format!("v{i}"),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: true,
                }]),
            };
            bincode::serialize_into(&mut encoder, &Some(&row)).unwrap();
        }
        bincode::serialize_into(&mut encoder, &None::<VariantRow>).unwrap();
        let mut writer = encoder.finish().unwrap();
        writer.flush().unwrap();
        spool.record_count("chr2", 100);

        // Truncate the compressed spool file and verify `take` returns an explicit Err.
        let orig_len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(orig_len.saturating_sub(20)).unwrap();
        drop(f);

        let res = spool.take("chr2");
        assert!(
            res.is_err(),
            "truncated reference spool must fail instead of returning partial rows"
        );
    }

    #[test]
    fn vcf_out_of_range_allele_on_biallelic_site_returns_error() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let vcf_path = dir.join(format!("janusx_haploid_bad_gt_{}_{unique}.vcf", std::process::id()));
        let out_path = dir.join(format!("janusx_haploid_bad_gt_{}_{unique}", std::process::id()));
        std::fs::write(
            &vcf_path,
            concat!(
                "##fileformat=VCFv4.2\n",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n",
                "1\t100\tv1\tA\tG\t.\tPASS\t.\tGT\t0|2\t0|0\n"
            ),
        )
        .unwrap();

        for ploidy_mode in [PloidyMode::Inbred, PloidyMode::Diploid] {
            let cfg = super::HaploidIbdConfig {
                target: vcf_path.clone(),
                output: out_path.clone(),
                ploidy_mode,
                ..super::HaploidIbdConfig::default()
            };
            let res = super::run_haploid_ibd(cfg);
            assert!(
                res.is_err(),
                "biallelic VCF site with GT=0|2 must error in {ploidy_mode:?} mode"
            );
        }
        let _ = std::fs::remove_file(&vcf_path);
        let _ = std::fs::remove_file(format!("{}.jhibd", out_path.display()));
    }

    #[test]
    fn ref_vcf_out_of_range_allele_on_biallelic_site_returns_error() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let tgt_path = dir.join(format!("janusx_haploid_tgt_ok_{}_{unique}.vcf", std::process::id()));
        let ref_path = dir.join(format!("janusx_haploid_ref_bad_{}_{unique}.vcf", std::process::id()));
        let out_path = dir.join(format!("janusx_haploid_ref_bad_{}_{unique}", std::process::id()));
        std::fs::write(
            &tgt_path,
            concat!(
                "##fileformat=VCFv4.2\n",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tt1\n",
                "1\t100\tv1\tA\tG\t.\tPASS\t.\tGT\t0|0\n"
            ),
        )
        .unwrap();
        std::fs::write(
            &ref_path,
            concat!(
                "##fileformat=VCFv4.2\n",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tr1\tr2\n",
                "1\t100\tv1\tA\tG\t.\tPASS\t.\tGT\t0|2\t0|0\n"
            ),
        )
        .unwrap();

        for ploidy_mode in [PloidyMode::Inbred, PloidyMode::Diploid] {
            let cfg = super::HaploidIbdConfig {
                target: tgt_path.clone(),
                reference: Some(ref_path.clone()),
                output: out_path.clone(),
                ploidy_mode,
                ..super::HaploidIbdConfig::default()
            };
            let res = super::run_haploid_ibd(cfg);
            assert!(
                res.is_err(),
                "biallelic reference VCF site with GT=0|2 must error in {ploidy_mode:?} mode"
            );
        }
        let _ = std::fs::remove_file(&tgt_path);
        let _ = std::fs::remove_file(&ref_path);
        let _ = std::fs::remove_file(format!("{}.jhibd", out_path.display()));
    }
}
