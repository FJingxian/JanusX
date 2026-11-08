use super::*;

pub(super) const MAX6_WINDOW_STREAM_UNSUPPORTED_ORDER: &str =
    "__MAX6_WINDOW_STREAM_UNSUPPORTED_ORDER__";

pub(super) struct Max6WindowStreamPrep {
    pub(super) global_site_keep: Vec<bool>,
    pub(super) packed_source_rows: Vec<usize>,
    pub(super) logic_bits: GarfieldLogicBits,
    pub(super) units: Vec<GarfieldLogicUnit>,
    pub(super) source_unit_indices: Vec<usize>,
}

pub(super) struct Max6StreamWindowRow {
    pub(super) qc_index: usize,
    pub(super) source_row: usize,
    pub(super) site: SiteInfo,
    pub(super) row_flip: bool,
    pub(super) pos: i32,
    pub(super) logic_words: Option<Vec<u64>>,
    pub(super) support: Option<u32>,
    pub(super) block_support: Option<Vec<u16>>,
}

pub(super) struct Max6StreamWindowState {
    pub(super) chrom: String,
    pub(super) min_bp: i32,
    pub(super) last_bp: i32,
    pub(super) next_center: i64,
    pub(super) rows: VecDeque<Max6StreamWindowRow>,
    pub(super) previous_signature: Option<(usize, usize, usize)>,
    pub(super) done: bool,
}

pub(super) struct Max6StreamWindowTask {
    pub(super) original_unit_index: usize,
    pub(super) unit: GarfieldLogicUnit,
    pub(super) source_rows: Vec<usize>,
    pub(super) sites: Vec<SiteInfo>,
    pub(super) row_flip: Vec<bool>,
    pub(super) logic_words: Vec<u64>,
    pub(super) support: Vec<u32>,
    pub(super) block_support: Vec<u16>,
}

pub(super) struct Max6WindowStreamLdPrep {
    pub(super) global_site_keep: Vec<bool>,
    pub(super) units: Vec<Max6PreparedUnitMetadata>,
    pub(super) screen_units: Vec<Max6GrmMax6LdPreparedUnit>,
    pub(super) original_unit_indices: Vec<usize>,
    pub(super) active_source_rows: Vec<usize>,
    pub(super) source_marker_count: usize,
    pub(super) window_pair_count: usize,
    pub(super) qc_row_count: usize,
    pub(super) distinct_window_count: usize,
    pub(super) selected_window_count: usize,
    pub(super) max_active_rows: usize,
    pub(super) max_task_batch_bytes: usize,
}

pub(super) struct Max6WindowPairUniverseAccumulator {
    pub(super) current_chrom: Option<String>,
    pub(super) units: Vec<Max6GrmMax6LdPreparedUnit>,
    pub(super) active_rows: Vec<usize>,
    pub(super) window_pair_count: usize,
    pub(super) total_pair_count: usize,
    pub(super) total_active_rows: usize,
    pub(super) total_window_pair_count: usize,
}

impl Default for Max6WindowPairUniverseAccumulator {
    fn default() -> Self {
        Self {
            current_chrom: None,
            units: Vec::new(),
            active_rows: Vec::new(),
            window_pair_count: 0,
            total_pair_count: 0,
            total_active_rows: 0,
            total_window_pair_count: 0,
        }
    }
}

impl Max6WindowPairUniverseAccumulator {
    pub(super) fn push_window(
        &mut self,
        chrom: &str,
        unit: &Max6GrmMax6LdPreparedUnit,
    ) -> Result<(), String> {
        if self
            .current_chrom
            .as_deref()
            .is_some_and(|current| current != chrom)
        {
            self.finish_chromosome()?;
        }
        if self.current_chrom.is_none() {
            self.current_chrom = Some(chrom.to_string());
        }
        let rows = unit.selected_global_rows.as_slice();
        self.window_pair_count = self
            .window_pair_count
            .checked_add(rows.len().saturating_mul(rows.len().saturating_sub(1)) / 2)
            .ok_or_else(|| "Max6 chromosome pair-occurrence count overflow".to_string())?;
        if !rows.is_empty() {
            self.active_rows.extend_from_slice(rows);
            self.units.push(Max6GrmMax6LdPreparedUnit {
                selected_global_rows: rows.to_vec(),
                pair_groups: unit.pair_groups.clone(),
                ld_compression: Vec::new(),
            });
        }
        Ok(())
    }

    pub(super) fn finish_chromosome(&mut self) -> Result<(), String> {
        if !self.active_rows.is_empty() {
            self.active_rows.sort_unstable();
            self.active_rows.dedup();
            let n_global_rows = self
                .active_rows
                .last()
                .copied()
                .and_then(|last| last.checked_add(1))
                .ok_or_else(|| "Max6 chromosome active-row bound overflow".to_string())?;
            let ownership = Max6ScreenPairOwnership::build(
                self.units.as_slice(),
                self.active_rows.as_slice(),
                n_global_rows,
            )?;
            let pair_count = count_max6_pairs_by_membership(
                self.active_rows.as_slice(),
                self.window_pair_count,
                &ownership,
            )?;
            self.total_pair_count = self
                .total_pair_count
                .checked_add(pair_count)
                .ok_or_else(|| "Max6 genome-wide unique-pair count overflow".to_string())?;
            self.total_active_rows = self
                .total_active_rows
                .checked_add(self.active_rows.len())
                .ok_or_else(|| "Max6 genome-wide active-marker count overflow".to_string())?;
            self.total_window_pair_count = self
                .total_window_pair_count
                .checked_add(self.window_pair_count)
                .ok_or_else(|| "Max6 genome-wide pair-occurrence count overflow".to_string())?;
        }
        self.current_chrom = None;
        self.units.clear();
        self.active_rows.clear();
        self.window_pair_count = 0;
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<(usize, usize, usize), String> {
        self.finish_chromosome()?;
        Ok((
            self.total_pair_count,
            self.total_active_rows,
            self.total_window_pair_count,
        ))
    }
}

pub(super) struct Max6StreamWindowSelection {
    pub(super) metadata: Max6PreparedUnitMetadata,
    pub(super) unit: GarfieldLogicUnit,
    pub(super) candidate_source_rows: Vec<usize>,
    pub(super) selected_source_rows: Vec<usize>,
    pub(super) bits: GarfieldLogicBits,
    pub(super) prepared_local: Max6GrmMax6LdPreparedUnit,
    pub(super) prepared_source: Max6GrmMax6LdPreparedUnit,
}

pub(super) fn select_max6_stream_window_task(
    task: Max6StreamWindowTask,
    sample_indices: &[usize],
    row_words: usize,
    n_samples: usize,
    ld_tag_r2: f64,
) -> Result<Max6StreamWindowSelection, String> {
    let Max6StreamWindowTask {
        original_unit_index,
        unit,
        source_rows,
        sites,
        row_flip,
        mut logic_words,
        mut support,
        mut block_support,
    } = task;
    let n_rows = source_rows.len();
    if logic_words.len() != n_rows.saturating_mul(row_words)
        || sites.len() != n_rows
        || row_flip.len() != n_rows
    {
        return Err(format!(
            "streamed Max6 window '{}' has inconsistent genotype or metadata rows ({} words, {} source rows, {} sites, {} flips, {} words per row)",
            unit.label,
            logic_words.len(),
            n_rows,
            sites.len(),
            row_flip.len(),
            row_words,
        ));
    }
    let logic_sites = build_logic_sites_from_metadata_owned_direct(
        sites,
        row_flip.as_slice(),
        GarfieldBinMode::Bin,
    );
    let logic_bits = GarfieldLogicBits {
        bits_flat: std::mem::take(&mut logic_words),
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words,
        sample_ids: Vec::new(),
        sites: logic_sites,
        group_ids: build_logic_mode_group_ids(n_rows, GarfieldBinMode::Bin),
        n_samples,
    };
    let support_cache = if ld_tag_r2 < 1.0 {
        let block_words = garfield_ld_block_words();
        let block_count = row_words.div_ceil(block_words);
        if support.len() != n_rows || block_support.len() != n_rows.saturating_mul(block_count) {
            return Err(format!(
                "streamed Max6 window '{}' has an incomplete LD support cache",
                unit.label
            ));
        }
        Some(GarfieldLdSupportCache {
            n_rows,
            row_words,
            block_words,
            block_count,
            support: std::mem::take(&mut support),
            block_support: std::mem::take(&mut block_support),
            sparse_global_to_local: None,
        })
    } else {
        None
    };
    let candidates = (0..n_rows).collect::<Vec<_>>();
    let unit = GarfieldLogicUnit {
        label: unit.label.clone(),
        indices: candidates,
        spans: unit.spans.clone(),
    };
    let selected = prepare_max6_grmmax6_ld_unit_with_support_cache(
        &unit,
        "window",
        &logic_bits,
        sample_indices,
        support_cache.as_ref(),
        ld_tag_r2,
    )?;
    let map_row = |row: usize| {
        source_rows
            .get(row)
            .copied()
            .ok_or_else(|| format!("streamed Max6 LD row {row} is outside its window"))
    };
    let selected_local_rows = selected.selected_global_rows;
    let selected_source_rows = selected_local_rows
        .iter()
        .map(|&row| map_row(row))
        .collect::<Result<Vec<_>, _>>()?;
    let prepared_source = Max6GrmMax6LdPreparedUnit {
        selected_global_rows: selected_source_rows.clone(),
        pair_groups: None,
        ld_compression: selected
            .ld_compression
            .iter()
            .map(|record| {
                Ok(GarfieldLdCompressionRecord {
                    kept_global_row: map_row(record.kept_global_row)?,
                    dropped_global_row: map_row(record.dropped_global_row)?,
                    r2: record.r2,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    let prepared_local = Max6GrmMax6LdPreparedUnit {
        selected_global_rows: selected_local_rows,
        pair_groups: None,
        ld_compression: Vec::new(),
    };
    let metadata = Max6PreparedUnitMetadata {
        original_unit_index,
        label: unit.label.clone(),
        spans: unit.spans.clone(),
    };
    let local_unit = GarfieldLogicUnit {
        label: unit.label,
        indices: (0..n_rows).collect(),
        spans: unit.spans,
    };
    Ok(Max6StreamWindowSelection {
        metadata,
        unit: local_unit,
        candidate_source_rows: source_rows,
        selected_source_rows,
        bits: logic_bits,
        prepared_local,
        prepared_source,
    })
}

pub(super) fn process_max6_stream_window_batch(
    tasks: &mut Vec<Max6StreamWindowTask>,
    sample_indices: &[usize],
    row_words: usize,
    n_samples: usize,
    ld_tag_r2: f64,
    pool: Option<&rayon::ThreadPool>,
    units: &mut Vec<Max6PreparedUnitMetadata>,
    screen_units: &mut Vec<Max6GrmMax6LdPreparedUnit>,
    original_unit_indices: &mut Vec<usize>,
    active_source_rows: &mut Vec<usize>,
    window_pair_count: &mut usize,
    any_eligible_unit: &mut bool,
    retain_prepared_metadata: bool,
) -> Result<Vec<Max6StreamWindowSelection>, String> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }
    let batch = std::mem::take(tasks);
    let results = if let Some(pool) = pool {
        pool.install(|| {
            batch
                .into_par_iter()
                .map(|task| {
                    select_max6_stream_window_task(
                        task,
                        sample_indices,
                        row_words,
                        n_samples,
                        ld_tag_r2,
                    )
                })
                .collect::<Result<Vec<_>, String>>()
        })?
    } else {
        batch
            .into_iter()
            .map(|task| {
                select_max6_stream_window_task(
                    task,
                    sample_indices,
                    row_words,
                    n_samples,
                    ld_tag_r2,
                )
            })
            .collect::<Result<Vec<_>, String>>()?
    };
    for selection in results.iter() {
        *any_eligible_unit |= selection.selected_source_rows.len() >= 2;
        *window_pair_count = window_pair_count.saturating_add(
            selection
                .selected_source_rows
                .len()
                .saturating_mul(selection.selected_source_rows.len().saturating_sub(1))
                / 2,
        );
        if retain_prepared_metadata {
            active_source_rows.extend(selection.selected_source_rows.iter().copied());
            original_unit_indices.push(selection.metadata.original_unit_index);
            units.push(selection.metadata.clone());
            screen_units.push(selection.prepared_source.clone());
        }
    }
    Ok(results)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_max6_stream_windows(
    state: &mut Max6StreamWindowState,
    max_bp: i32,
    at_chromosome_end: bool,
    extension: usize,
    step: usize,
    scan_bimranges: &[GarfieldScanBimRange],
    next_original_unit_index: &mut usize,
    distinct_window_count: &mut usize,
    selected_window_count: &mut usize,
    tasks: &mut Vec<Max6StreamWindowTask>,
    task_batch_bytes: &mut usize,
    max_task_batch_bytes: &mut usize,
) -> Result<(), String> {
    if state.done {
        return Ok(());
    }
    let extension_i64 = extension.max(1) as i64;
    let step_i64 = step.max(1) as i64;
    loop {
        let right_unclipped = state.next_center.saturating_add(extension_i64);
        if !at_chromosome_end && i64::from(max_bp) <= right_unclipped {
            break;
        }
        let left = state
            .next_center
            .saturating_sub(extension_i64)
            .max(i64::from(state.min_bp));
        let right = if at_chromosome_end {
            right_unclipped.min(i64::from(max_bp))
        } else {
            right_unclipped
        };
        let rows = state
            .rows
            .iter()
            .filter(|row| i64::from(row.pos) >= left && i64::from(row.pos) <= right)
            .collect::<Vec<_>>();
        if let (Some(first), Some(last)) = (rows.first(), rows.last()) {
            let signature = (first.qc_index, last.qc_index, rows.len());
            if state.previous_signature != Some(signature) {
                let bp_start = clamp_i64_to_i32(left);
                let bp_end = clamp_i64_to_i32(right);
                let unit = GarfieldLogicUnit {
                    label: format!("{}:{}-{}", state.chrom, bp_start, bp_end),
                    indices: Vec::new(),
                    spans: vec![GarfieldUnitSpan {
                        chrom: state.chrom.clone(),
                        bp_start,
                        bp_end,
                    }],
                };
                *distinct_window_count = distinct_window_count.saturating_add(1);
                if scan_bimranges.is_empty() || unit_overlaps_scan_bimranges(&unit, scan_bimranges)
                {
                    let row_words = rows
                        .first()
                        .and_then(|row| row.logic_words.as_ref())
                        .map_or(0, Vec::len);
                    if row_words == 0 {
                        return Err(format!(
                            "selected streamed window '{}' has no packed genotype rows",
                            unit.label
                        ));
                    }
                    let mut source_rows = Vec::with_capacity(rows.len());
                    let mut sites = Vec::with_capacity(rows.len());
                    let mut row_flip = Vec::with_capacity(rows.len());
                    let mut logic_words = Vec::with_capacity(rows.len().saturating_mul(row_words));
                    let mut support = Vec::with_capacity(rows.len());
                    let block_count = row_words.div_ceil(garfield_ld_block_words());
                    let mut block_support =
                        Vec::with_capacity(rows.len().saturating_mul(block_count));
                    for row in rows.iter() {
                        let words = row.logic_words.as_ref().ok_or_else(|| {
                            format!(
                                "selected streamed window '{}' includes unpacked source row {}",
                                unit.label, row.source_row
                            )
                        })?;
                        if words.len() != row_words {
                            return Err(format!(
                                "streamed Max6 window '{}' contains inconsistent row widths",
                                unit.label
                            ));
                        }
                        source_rows.push(row.source_row);
                        sites.push(row.site.clone());
                        row_flip.push(row.row_flip);
                        logic_words.extend_from_slice(words);
                        if let Some(row_support) = row.support {
                            support.push(row_support);
                        }
                        if let Some(row_blocks) = row.block_support.as_ref() {
                            block_support.extend_from_slice(row_blocks);
                        }
                    }
                    let task_bytes = logic_words
                        .len()
                        .saturating_mul(std::mem::size_of::<u64>())
                        .saturating_add(
                            block_support
                                .len()
                                .saturating_mul(std::mem::size_of::<u16>()),
                        )
                        .saturating_add(support.len().saturating_mul(std::mem::size_of::<u32>()));
                    *task_batch_bytes = task_batch_bytes.saturating_add(task_bytes);
                    *max_task_batch_bytes = (*max_task_batch_bytes).max(*task_batch_bytes);
                    tasks.push(Max6StreamWindowTask {
                        original_unit_index: *next_original_unit_index,
                        unit,
                        source_rows,
                        sites,
                        row_flip,
                        logic_words,
                        support,
                        block_support,
                    });
                    *selected_window_count = selected_window_count.saturating_add(1);
                }
                *next_original_unit_index = next_original_unit_index.saturating_add(1);
                state.previous_signature = Some(signature);
            }
        }

        if at_chromosome_end && i64::from(max_bp) <= state.next_center {
            state.done = true;
            break;
        }
        let next_center = state.next_center.saturating_add(step_i64);
        if next_center > i64::from(i32::MAX) {
            state.done = true;
            break;
        }
        state.next_center = next_center;
        let keep_from = state
            .next_center
            .saturating_sub(extension_i64)
            .max(i64::from(state.min_bp));
        while state
            .rows
            .front()
            .is_some_and(|row| i64::from(row.pos) < keep_from)
        {
            state.rows.pop_front();
        }
    }
    if at_chromosome_end {
        state.rows.clear();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_max6_stream_chromosome(
    state: &mut Max6StreamWindowState,
    restrict_genotypes_to_scope: bool,
    tail_scope_complete: bool,
    scan_bimranges: &[GarfieldScanBimRange],
    extension: usize,
    step: usize,
    next_original_unit_index: &mut usize,
    distinct_window_count: &mut usize,
    selected_window_count: &mut usize,
    tasks: &mut Vec<Max6StreamWindowTask>,
    task_batch_bytes: &mut usize,
    max_task_batch_bytes: &mut usize,
) -> Result<(), String> {
    let (max_bp, at_chromosome_end) = if !restrict_genotypes_to_scope || tail_scope_complete {
        (state.last_bp, true)
    } else if let Some(scope_end) = max6_window_stream_scope_finish_bp(
        state.chrom.as_str(),
        scan_bimranges,
        (extension.max(1) as i64).saturating_mul(2),
    ) {
        (scope_end, false)
    } else {
        (state.last_bp, true)
    };
    emit_max6_stream_windows(
        state,
        max_bp,
        at_chromosome_end,
        extension,
        step,
        scan_bimranges,
        next_original_unit_index,
        distinct_window_count,
        selected_window_count,
        tasks,
        task_batch_bytes,
        max_task_batch_bytes,
    )?;
    if !at_chromosome_end {
        state.done = true;
        state.rows.clear();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_max6_window_streamed_ld(
    prefix: &str,
    sample_ids: &[String],
    sample_indices: &[usize],
    extension: usize,
    step: Option<usize>,
    scan_bimranges: &[GarfieldScanBimRange],
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    threads: usize,
    ld_tag_r2: f64,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<Max6WindowStreamLdPrep, String> {
    prepare_max6_window_streamed_ld_with_callback(
        prefix,
        sample_ids,
        sample_indices,
        extension,
        step,
        scan_bimranges,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        threads,
        0,
        ld_tag_r2,
        mem_tracker,
        false,
        None,
        None,
    )
}

pub(super) type Max6WindowFusedBatchCallback<'a> =
    dyn FnMut(&[Max6StreamWindowSelection], Option<&rayon::ThreadPool>) -> Result<(), String> + 'a;
pub(super) type Max6WindowStreamProgressCallback<'a> =
    dyn FnMut(usize, usize) -> Result<(), String> + 'a;

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_max6_window_streamed_ld_with_callback(
    prefix: &str,
    sample_ids: &[String],
    sample_indices: &[usize],
    extension: usize,
    step: Option<usize>,
    scan_bimranges: &[GarfieldScanBimRange],
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    threads: usize,
    logic_mem_budget_bytes: u64,
    ld_tag_r2: f64,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
    restrict_to_bimrange_scope: bool,
    mut on_batch: Option<&mut Max6WindowFusedBatchCallback<'_>>,
    mut on_progress: Option<&mut Max6WindowStreamProgressCallback<'_>>,
) -> Result<Max6WindowStreamLdPrep, String> {
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err("maf_threshold must be within [0, 0.5]".to_string());
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err("max_missing_rate must be within [0, 1.0]".to_string());
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err("het_threshold must be within [0, 1.0]".to_string());
    }
    if sample_ids.is_empty() || sample_ids.len() != sample_indices.len() {
        return Err(format!(
            "Max6 window stream requires matching nonempty sample IDs/indices; got {}/{}",
            sample_ids.len(),
            sample_indices.len()
        ));
    }
    validate_max6_ld_tag_r2(ld_tag_r2)?;

    let restrict_genotypes_to_scope =
        restrict_to_bimrange_scope && on_batch.is_some() && !scan_bimranges.is_empty();
    let anchor_ready_chroms = Arc::new(Mutex::new(HashSet::<String>::new()));
    let tail_scope_complete_chroms = Arc::new(Mutex::new(HashSet::<String>::new()));
    let anchor_ready_for_consumer = Arc::clone(&anchor_ready_chroms);
    let tail_scope_complete_for_consumer = Arc::clone(&tail_scope_complete_chroms);
    let mut source = Max6WindowRangeSource::new(
        prefix,
        scan_bimranges,
        extension,
        restrict_genotypes_to_scope,
        anchor_ready_chroms,
        tail_scope_complete_chroms,
    )?;
    let source_marker_count = source.n_source_rows;
    if let Some(callback) = on_progress.as_deref_mut() {
        callback(0, source_marker_count)?;
    }
    let n_samples_total = source.bed.n_samples();
    if sample_indices.iter().any(|&index| index >= n_samples_total) {
        return Err("selected sample index out of range for Max6 window stream".to_string());
    }
    let n_samples = sample_indices.len();
    let row_words = words_for_samples(n_samples);
    let bytes_per_variant = source.bed.bytes_per_snp();
    let default_chunk_rows = source.max_chunk_rows.max(1);
    let (chunk_rows, task_batch_byte_limit) = if logic_mem_budget_bytes == 0 {
        (default_chunk_rows, 64usize * 1024 * 1024)
    } else {
        let budget = usize::try_from(logic_mem_budget_bytes).unwrap_or(usize::MAX);
        let estimated_row_bytes = bytes_per_variant
            .saturating_add(row_words.saturating_mul(std::mem::size_of::<u64>()))
            .saturating_add(std::mem::size_of::<Max6FusedRowQc>())
            .saturating_add(std::mem::size_of::<SiteInfo>())
            .saturating_add(std::mem::size_of::<usize>());
        // Reserve roughly half of the requested budget for the two decode/QC
        // buffers and one quarter for pending window tasks. The remaining
        // quarter covers worker scratch, the current window state, and output
        // buffers. A single oversized window remains an unavoidable floor.
        let chunk_rows = (budget / 4 / estimated_row_bytes.max(1))
            .max(1)
            .min(default_chunk_rows);
        let task_limit = (budget / 4).min(64 * 1024 * 1024).max(1);
        (chunk_rows, task_limit)
    };
    source.max_chunk_rows = chunk_rows;
    let sample_identity =
        n_samples == n_samples_total && sample_indices_are_identity(sample_indices);
    let sample_plan = build_logic_sample_plan(sample_indices);
    let prep_threads = effective_threads_local(threads).max(1);
    let pool = if prep_threads > 1 {
        crate::stats_common::get_cached_pool(prep_threads).map_err(|error| error.to_string())?
    } else {
        None
    };

    let mut global_site_keep = if on_batch.is_some() {
        Vec::new()
    } else {
        vec![false; source_marker_count]
    };
    let retain_prepared_metadata = on_batch.is_none();
    let mut fail_missing = 0usize;
    let mut fail_het = 0usize;
    let mut fail_no_non_missing = 0usize;
    let mut fail_maf = 0usize;
    let mut fail_non_simple_snp = 0usize;
    let mut het_sites = 0usize;
    let mut raw_missing_sites = 0usize;
    let mut qc_row_count = 0usize;
    let mut distinct_window_count = 0usize;
    let mut selected_window_count = 0usize;
    let mut max_active_rows = 0usize;
    let mut max_task_batch_bytes = 0usize;
    let extension_bp = (extension.max(1) as i64).saturating_mul(2);
    let in_pack_scope =
        |site: &SiteInfo| max6_window_site_in_pack_scope(site, scan_bimranges, extension_bp);
    let extension_eff = extension.max(1);
    let step_eff = step.unwrap_or((extension_eff / 2).max(1)).max(1);
    let worker_count = pool
        .as_ref()
        .map_or(1, |pool| pool.current_num_threads())
        .max(1);
    // Keep enough adjacent windows together for the Max6 batch scanner to
    // merge overlapping pair work. The independent byte cap below remains the
    // memory bound when dense windows exceed this count-based target.
    let task_batch_limit = worker_count.saturating_mul(8).max(1);
    let support_cache_required = ld_tag_r2 < 1.0;
    let support_block_words = garfield_ld_block_words();
    let support_block_count = row_words.div_ceil(support_block_words);
    let mut tasks = Vec::<Max6StreamWindowTask>::new();
    let mut task_batch_bytes = 0usize;
    let train_indices = (0..n_samples).collect::<Vec<_>>();
    let mut units = Vec::<Max6PreparedUnitMetadata>::new();
    let mut screen_units = Vec::<Max6GrmMax6LdPreparedUnit>::new();
    let mut original_unit_indices = Vec::<usize>::new();
    let mut active_source_rows = Vec::<usize>::new();
    let mut window_pair_count = 0usize;
    let mut any_eligible_unit = false;
    let mut next_original_unit_index = 0usize;
    let mut next_qc_index = 0usize;
    let mut window_state: Option<Max6StreamWindowState> = None;
    let mut seen_qc_chromosomes = HashSet::<String>::new();
    let mut anchor_ready_marked = HashSet::<String>::new();
    let mut chrom_pool = HashMap::<String, Arc<str>>::new();
    let mut unsupported_order = false;

    crate::pipeline::run_double_buffer(
        2,
        || Max6WindowStreamChunk::new(chunk_rows, bytes_per_variant, row_words),
        |chunk| match source.fill_chunk(chunk) {
            Ok(more) => more,
            Err(error) => {
                chunk.clear();
                chunk.error = Some(error);
                false
            }
        },
        |chunk| {
            if let Some(error) = chunk.error.take() {
                return Err(error);
            }
            let rows = chunk.sites.len();
            if rows == 0 {
                return Ok(());
            }
            let required_words = rows
                .checked_mul(row_words)
                .ok_or_else(|| "Max6 window logic chunk size overflow".to_string())?;
            chunk.logic_bits.resize(required_words, 0);
            chunk.logic_bits.fill(0);
            chunk.qc.resize(rows, Max6FusedRowQc::default());
            let qc_batch = &mut chunk.qc;
            let row_bits = &mut chunk.logic_bits;
            let sites = &chunk.sites;
            let packed_genotypes = &chunk.packed_genotypes;
            let mut process_rows = || {
                row_bits
                    .par_chunks_mut(row_words)
                    .zip(qc_batch.par_iter_mut())
                    .enumerate()
                    .for_each_init(
                        || vec![0u64; row_words],
                        |missing_words, (offset, (dst_words, qc_dst))| {
                            let site = &sites[offset];
                            let row_start = offset * bytes_per_variant;
                            let row = &packed_genotypes[row_start..row_start + bytes_per_variant];
                            let simple_snp_alleles = !snps_only
                                || (is_simple_snp_allele(&site.ref_allele)
                                    && is_simple_snp_allele(&site.alt_allele));
                            *qc_dst = scan_max6_row_qc_and_pack_logic(
                                row,
                                n_samples,
                                sample_identity,
                                sample_plan.as_slice(),
                                maf_threshold,
                                max_missing_rate,
                                simple_snp_alleles,
                                dst_words,
                                missing_words.as_mut_slice(),
                            );
                        },
                    );
            };
            if let Some(pool) = pool.as_ref() {
                pool.install(process_rows);
            } else {
                process_rows();
            }
            check_ctrlc()?;

            for offset in 0..rows {
                let qc = chunk.qc[offset];
                let source_row = chunk.source_rows[offset];
                if qc.keep {
                    if !global_site_keep.is_empty() {
                        global_site_keep[source_row] = true;
                    }
                    let site = &chunk.sites[offset];
                    if restrict_genotypes_to_scope {
                        let anchor_chrom =
                            chrom_base_view(site.chrom.as_str()).to_ascii_lowercase();
                        if anchor_ready_marked.insert(anchor_chrom.clone()) {
                            anchor_ready_for_consumer
                                .lock()
                                .map_err(|_| {
                                    "Max6 window range anchor state was poisoned".to_string()
                                })?
                                .insert(anchor_chrom);
                        }
                    }
                    let chrom_key = normalize_chrom(site.chrom.as_str());
                    let chrom = chrom_pool
                        .entry(chrom_key.clone())
                        .or_insert_with(|| Arc::<str>::from(chrom_key.as_str()))
                        .clone();
                    let is_in_scope = in_pack_scope(site);
                    let source_words =
                        &chunk.logic_bits[offset * row_words..(offset + 1) * row_words];
                    let (logic_words, support, block_support) = if is_in_scope {
                        let words = source_words.to_vec();
                        if support_cache_required {
                            let mut blocks = vec![0u16; support_block_count];
                            let support = fill_ld_block_supports(
                                words.as_slice(),
                                support_block_words,
                                blocks.as_mut_slice(),
                            );
                            (
                                Some(words),
                                Some(u32::try_from(support).unwrap_or(u32::MAX)),
                                Some(blocks),
                            )
                        } else {
                            (Some(words), None, None)
                        }
                    } else {
                        (None, None, None)
                    };
                    let row = Max6StreamWindowRow {
                        qc_index: next_qc_index,
                        source_row,
                        site: site.clone(),
                        row_flip: qc.row_flip,
                        pos: site.pos,
                        logic_words,
                        support,
                        block_support,
                    };
                    next_qc_index = next_qc_index.saturating_add(1);
                    qc_row_count = qc_row_count.saturating_add(1);
                    match window_state.as_mut() {
                        Some(state) if state.chrom == chrom.as_ref() => {
                            if site.pos < state.last_bp {
                                unsupported_order = true;
                                return Err(MAX6_WINDOW_STREAM_UNSUPPORTED_ORDER.to_string());
                            }
                            state.last_bp = site.pos;
                            state.rows.push_back(row);
                        }
                        Some(state) => {
                            if seen_qc_chromosomes.contains(chrom.as_ref()) {
                                unsupported_order = true;
                                return Err(MAX6_WINDOW_STREAM_UNSUPPORTED_ORDER.to_string());
                            }
                            let tail_scope_complete = tail_scope_complete_for_consumer
                                .lock()
                                .map_err(|_| {
                                    "Max6 window range tail state was poisoned".to_string()
                                })?
                                .contains(state.chrom.as_str());
                            finish_max6_stream_chromosome(
                                state,
                                restrict_genotypes_to_scope,
                                tail_scope_complete,
                                scan_bimranges,
                                extension_eff,
                                step_eff,
                                &mut next_original_unit_index,
                                &mut distinct_window_count,
                                &mut selected_window_count,
                                &mut tasks,
                                &mut task_batch_bytes,
                                &mut max_task_batch_bytes,
                            )?;
                            seen_qc_chromosomes.insert(state.chrom.clone());
                            window_state = Some(Max6StreamWindowState {
                                chrom: chrom.to_string(),
                                min_bp: site.pos,
                                last_bp: site.pos,
                                next_center: i64::from(site.pos),
                                rows: VecDeque::new(),
                                previous_signature: None,
                                done: false,
                            });
                            window_state
                                .as_mut()
                                .expect("new Max6 window state was inserted")
                                .rows
                                .push_back(row);
                        }
                        None => {
                            if seen_qc_chromosomes.contains(chrom.as_ref()) {
                                unsupported_order = true;
                                return Err(MAX6_WINDOW_STREAM_UNSUPPORTED_ORDER.to_string());
                            }
                            window_state = Some(Max6StreamWindowState {
                                chrom: chrom.to_string(),
                                min_bp: site.pos,
                                last_bp: site.pos,
                                next_center: i64::from(site.pos),
                                rows: VecDeque::new(),
                                previous_signature: None,
                                done: false,
                            });
                            window_state
                                .as_mut()
                                .expect("new Max6 window state was inserted")
                                .rows
                                .push_back(row);
                        }
                    }
                    if let Some(state) = window_state.as_mut() {
                        max_active_rows = max_active_rows.max(state.rows.len());
                        emit_max6_stream_windows(
                            state,
                            site.pos,
                            false,
                            extension_eff,
                            step_eff,
                            scan_bimranges,
                            &mut next_original_unit_index,
                            &mut distinct_window_count,
                            &mut selected_window_count,
                            &mut tasks,
                            &mut task_batch_bytes,
                            &mut max_task_batch_bytes,
                        )?;
                    }
                    let should_flush = tasks.len() >= task_batch_limit
                        || task_batch_bytes >= task_batch_byte_limit;
                    if should_flush {
                        let selections = process_max6_stream_window_batch(
                            &mut tasks,
                            train_indices.as_slice(),
                            row_words,
                            n_samples,
                            ld_tag_r2,
                            pool.as_deref(),
                            &mut units,
                            &mut screen_units,
                            &mut original_unit_indices,
                            &mut active_source_rows,
                            &mut window_pair_count,
                            &mut any_eligible_unit,
                            retain_prepared_metadata,
                        )?;
                        if let Some(callback) = on_batch.as_deref_mut() {
                            callback(&selections, pool.as_deref())?;
                        }
                        task_batch_bytes = 0;
                    }
                } else {
                    match qc.reason {
                        MAX6_FUSED_QC_FAIL_MISSING => fail_missing += 1,
                        MAX6_FUSED_QC_FAIL_HET => fail_het += 1,
                        MAX6_FUSED_QC_FAIL_NO_HOMOZYGOUS => fail_no_non_missing += 1,
                        MAX6_FUSED_QC_FAIL_MAF => fail_maf += 1,
                        MAX6_FUSED_QC_FAIL_NON_SIMPLE_SNP => fail_non_simple_snp += 1,
                        _ => {}
                    }
                }
                if qc.has_het {
                    het_sites += 1;
                }
                if qc.has_raw_missing {
                    raw_missing_sites += 1;
                }
            }
            if let Some(tracker) = mem_tracker {
                tracker.sample_now();
            }
            if let Some(callback) = on_progress.as_deref_mut() {
                let done = chunk
                    .source_rows
                    .last()
                    .map_or(0, |source_row| source_row.saturating_add(1));
                let progress_total = source_marker_count.max(1);
                let progress_done = if progress_total > 1 {
                    done.min(progress_total - 1)
                } else {
                    0
                };
                callback(progress_done, progress_total)?;
            }
            Ok(())
        },
    )
    .map_err(|error| {
        if unsupported_order {
            MAX6_WINDOW_STREAM_UNSUPPORTED_ORDER.to_string()
        } else {
            error
        }
    })?;

    if let Some(state) = window_state.as_mut() {
        let tail_scope_complete = tail_scope_complete_for_consumer
            .lock()
            .map_err(|_| "Max6 window range tail state was poisoned".to_string())?
            .contains(state.chrom.as_str());
        finish_max6_stream_chromosome(
            state,
            restrict_genotypes_to_scope,
            tail_scope_complete,
            scan_bimranges,
            extension_eff,
            step_eff,
            &mut next_original_unit_index,
            &mut distinct_window_count,
            &mut selected_window_count,
            &mut tasks,
            &mut task_batch_bytes,
            &mut max_task_batch_bytes,
        )?;
    }
    let selections = process_max6_stream_window_batch(
        &mut tasks,
        train_indices.as_slice(),
        row_words,
        n_samples,
        ld_tag_r2,
        pool.as_deref(),
        &mut units,
        &mut screen_units,
        &mut original_unit_indices,
        &mut active_source_rows,
        &mut window_pair_count,
        &mut any_eligible_unit,
        retain_prepared_metadata,
    )?;
    if let Some(callback) = on_batch.as_deref_mut() {
        callback(&selections, pool.as_deref())?;
    }

    if qc_row_count == 0 {
        if !scan_bimranges.is_empty() && source.seen_selected_chroms.is_empty() {
            return Err("--bimrange matched no BIM/PVAR chromosomes".to_string());
        }
        return Err(format_zero_sites_pure_line_error(
            n_samples,
            n_samples_total,
            source_marker_count,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            fail_missing,
            fail_het,
            fail_no_non_missing,
            fail_maf,
            fail_non_simple_snp,
            het_sites,
            raw_missing_sites,
        ));
    }
    if selected_window_count == 0 {
        return Err(if scan_bimranges.is_empty() {
            "no scan units were built from the globally QC-passing markers".to_string()
        } else {
            "--bimrange matched no Max6 scan units".to_string()
        });
    }
    if !any_eligible_unit {
        return Err("Max6 scan found no eligible LD-tag unit with at least two markers".into());
    }
    if retain_prepared_metadata {
        active_source_rows.sort_unstable();
        active_source_rows.dedup();
        if active_source_rows.len() < 2 {
            return Err("Max6 streamed LD selection produced fewer than two active markers".into());
        }
        if units.len() != screen_units.len() || units.len() != original_unit_indices.len() {
            return Err("streamed Max6 window unit metadata lengths disagree".to_string());
        }
    }
    Ok(Max6WindowStreamLdPrep {
        global_site_keep,
        units,
        screen_units,
        original_unit_indices,
        active_source_rows,
        source_marker_count,
        window_pair_count,
        qc_row_count,
        distinct_window_count,
        selected_window_count,
        max_active_rows,
        max_task_batch_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_max6_window_fused_grm_int_scan(
    prefix: &str,
    sample_ids: &[String],
    sample_indices: &[usize],
    extension: usize,
    step: Option<usize>,
    scan_bimranges: &[GarfieldScanBimRange],
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    threads: usize,
    logic_mem_budget_bytes: u64,
    ld_tag_r2: f64,
    y: &[f64],
    train_fit: Arc<GarfieldResidualResult>,
    p_c: Option<&[f64]>,
    no_grm_projector: Option<&NoGrmCovariateProjector>,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    beam_params: &BeamSearchParams,
    final_top_k: usize,
    min_cell: usize,
    min_cell_fraction: f64,
    seed: u64,
    out_prefix: Option<&str>,
    progress_callback: Option<&Py<PyAny>>,
    write_provenance: bool,
) -> Result<GarfieldLogicPipelineResult, String> {
    let total_wall_t0 = Instant::now();
    let n_samples = sample_ids.len();
    if n_samples == 0 || n_samples != sample_indices.len() || y.len() != n_samples {
        return Err(format!(
            "fused Max6 window scan sample lengths disagree: ids={}, indices={}, y={}",
            n_samples,
            sample_indices.len(),
            y.len()
        ));
    }
    let min_cell = min_cell::resolve_min_cell(min_cell, min_cell_fraction, n_samples)?;
    let score_scale = if train_fit.no_grm_boundary {
        if !(train_fit.sigma_e2.is_finite() && train_fit.sigma_e2 > 0.0) {
            return Err(format!(
                "Max6 -> GRM-INT no-GRM boundary requires positive finite sigma_e2, got {}",
                train_fit.sigma_e2
            ));
        }
        1.0 / train_fit.sigma_e2
    } else {
        if !(train_fit.sigma_g2.is_finite() && train_fit.sigma_g2 > 0.0) {
            return Err(format!(
                "Max6 -> GRM-INT requires positive finite sigma_g2, got {}",
                train_fit.sigma_g2
            ));
        }
        1.0 / train_fit.sigma_g2
    };
    let projected_y = if let Some(p_c) = p_c {
        validate_grm_int_dense_projection(n_samples, p_c)?;
        project_phenotype_for_candidate_rerank(p_c, n_samples, y)
    } else if let Some(projector) = no_grm_projector {
        if projector.n_samples() != n_samples {
            return Err(format!(
                "fused Max6 no-GRM projector sample count {} != {n_samples}",
                projector.n_samples()
            ));
        }
        projector.project(y)?
    } else {
        return Err("fused Max6 scan requires P_C or a no-GRM projector".to_string());
    };
    let scaled_total_quadratic = score_scale
        * y.iter()
            .zip(projected_y.iter())
            .map(|(&value, &projected)| value * projected)
            .sum::<f64>();
    if !score_scale.is_finite()
        || score_scale <= 0.0
        || !scaled_total_quadratic.is_finite()
        || scaled_total_quadratic <= 0.0
    {
        return Err(format!(
            "fused Max6 score scale/quadratic is invalid: scale={score_scale}, quadratic={scaled_total_quadratic}"
        ));
    }
    let shared_y_lookup = PackedYSumLookup::build(train_fit.residualized_y.as_slice(), n_samples)?;
    let scan_started = Instant::now();

    let pair_path = out_prefix.map(max6_grmmax6_pairs_output_path);
    let pair_tmp_path = pair_path.as_ref().map(|path| {
        std::path::PathBuf::from(format!(
            "{path}.tmp.{}.{}",
            std::process::id(),
            GARFIELD_BITS_MMAP_SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    });
    let _pair_tmp_owner = pair_tmp_path
        .as_ref()
        .map(|path| Max6BitsTempFileOwner { path: path.clone() });
    let pair_writer = if let Some(path) = pair_tmp_path.as_ref() {
        let header = format!("{MAX6_GRM_INT_PAIRS_HEADER}\n");
        Some(crate::stats_common::AsyncTsvWriter::with_config(
            path.to_string_lossy().as_ref(),
            header.as_bytes(),
            8 * 1024 * 1024,
            2,
        )?)
    } else {
        None
    };
    let provenance_path = if let Some(prefix) = out_prefix {
        if write_provenance {
            Some(max6_grm_int_provenance_output_path(prefix))
        } else {
            remove_stale_max6_grm_int_provenance(prefix, false)?;
            None
        }
    } else {
        None
    };
    let provenance_writer = if let Some(path) = provenance_path.as_ref() {
        Some(crate::stats_common::AsyncTsvWriter::with_config(
            path,
            b"output_row_index\tunit_index_zero_based\tunit_label\n",
            8 * 1024 * 1024,
            2,
        )?)
    } else {
        None
    };

    let mut output_count = 0usize;
    let mut output_row_index = 0usize;
    let mut screen_candidate_count = 0usize;
    let mut final_screen_subset_audit_pairs_checked = 0usize;
    let mut observed_scan_timing = Max6ObservedScanTiming::default();
    let mut pair_universe_accumulator = Max6WindowPairUniverseAccumulator::default();
    let mut skipped_messages = Vec::<String>::new();
    let shared_grm_int_score_pairs = parse_env_usize("JX_GARFIELD_GRM_INT_SCORE_PAIRS")
        .unwrap_or(if cfg!(windows) { 32 } else { 64 });
    let mut on_batch = |selections: &[Max6StreamWindowSelection],
                        pool: Option<&rayon::ThreadPool>|
     -> Result<(), String> {
        if selections.is_empty() {
            return Ok(());
        }
        let _blas_guard = (selections.len() > 1).then(|| BlasThreadGuard::enter(1));
        let batch_scan_started = Instant::now();
        let row_words = selections[0].bits.row_words;
        if selections
            .iter()
            .any(|selection| selection.bits.row_words != row_words)
        {
            return Err("fused Max6 window batch has inconsistent row widths".to_string());
        }

        // Keep only LD-selected rows for the current bounded window batch.
        // Sorting by source row gives every unit a sorted compact membership
        // list, while repeated markers across overlapping windows share one
        // packed row and one GRM-INT score.
        let mut source_row_origin = HashMap::<usize, (usize, usize)>::new();
        let mut unit_source_rows = Vec::<Vec<usize>>::with_capacity(selections.len());
        for (selection_index, selection) in selections.iter().enumerate() {
            let mut rows = Vec::with_capacity(selection.prepared_local.selected_global_rows.len());
            for &local_row in selection.prepared_local.selected_global_rows.iter() {
                let source_row = *selection.candidate_source_rows.get(local_row).ok_or_else(|| {
                    format!(
                        "fused Max6 selected row {local_row} is outside candidate metadata for '{}'",
                        selection.unit.label
                    )
                })?;
                source_row_origin
                    .entry(source_row)
                    .or_insert((selection_index, local_row));
                rows.push(source_row);
            }
            unit_source_rows.push(rows);
        }
        let mut batch_source_rows = source_row_origin.keys().copied().collect::<Vec<_>>();
        batch_source_rows.sort_unstable();
        let source_to_batch = batch_source_rows
            .iter()
            .enumerate()
            .map(|(index, &source_row)| (source_row, index))
            .collect::<HashMap<_, _>>();
        let mut batch_bits =
            Vec::<u64>::with_capacity(batch_source_rows.len().saturating_mul(row_words));
        for &source_row in batch_source_rows.iter() {
            let (selection_index, local_row) = *source_row_origin
                .get(&source_row)
                .ok_or_else(|| format!("fused Max6 source row {source_row} lost its packed row"))?;
            let selection = &selections[selection_index];
            let start = local_row.saturating_mul(row_words);
            let end = start.saturating_add(row_words);
            let row = selection.bits.bits().get(start..end).ok_or_else(|| {
                format!(
                    "fused Max6 packed row {local_row} is outside window '{}'",
                    selection.unit.label
                )
            })?;
            batch_bits.extend_from_slice(row);
        }
        let unit_selected_rows = unit_source_rows
            .iter()
            .map(|source_rows| {
                source_rows
                    .iter()
                    .map(|source_row| {
                        source_to_batch.get(source_row).copied().ok_or_else(|| {
                            format!("fused Max6 source row {source_row} has no batch index")
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?;
        let scan_unit_indices = unit_selected_rows
            .iter()
            .enumerate()
            .filter_map(|(index, rows)| (rows.len() >= 2).then_some(index))
            .collect::<Vec<_>>();
        let mut scans = (0..selections.len())
            .map(|index| {
                if unit_selected_rows[index].len() >= 2 {
                    None
                } else {
                    Some(Max6GrmMax6UnitScan {
                        outputs: Vec::new(),
                        permutation_pool: None,
                        deferred_screen: None,
                        screen_pairs_passed: 0,
                        final_screen_subset_audit_pairs_checked: 0,
                        candidate_pair_keys: Vec::new(),
                        candidate_score_observations: Vec::new(),
                        window_unique_candidate_endpoints: 0,
                        timing: Max6ObservedUnitTiming::default(),
                        skipped_message: Some(format!(
                            "unit '{}' has fewer than two SNPs after Max6 marker selection",
                            selections[index].unit.label
                        )),
                    })
                }
            })
            .collect::<Vec<_>>();

        if !scan_unit_indices.is_empty() {
            let batch_units = scan_unit_indices
                .iter()
                .map(|&index| logic_residual_screen::ResidualScreenBatchUnit {
                    selected_global_rows: unit_selected_rows[index].as_slice(),
                    pair_groups: None,
                })
                .collect::<Vec<_>>();
            let all_batch_rows = (0..batch_source_rows.len()).collect::<Vec<_>>();
            let batch_margins = OneCellMarginTable::build_for_rows(
                batch_bits.as_slice(),
                row_words,
                all_batch_rows.as_slice(),
                train_fit.residualized_y.as_slice(),
                n_samples,
                &shared_y_lookup,
            )?;
            let screen_started = Instant::now();
            let batch_screen = logic_residual_screen::scan_residual_pairs_packed_max6_batch(
                batch_bits.as_slice(),
                row_words,
                batch_units.as_slice(),
                train_fit.residualized_y.as_slice(),
                n_samples,
                max6_grmmax6_screen_top_k(true, final_top_k),
                min_cell,
                &shared_y_lookup,
                &batch_margins,
                true,
            )?;
            let screen_kernel_seconds = screen_started.elapsed().as_secs_f64();
            let batch_timing = batch_screen.timing;
            observed_scan_timing.screen_batch_count =
                observed_scan_timing.screen_batch_count.saturating_add(1);
            observed_scan_timing.screen_batch_prepare_wall_seconds += screen_started
                .duration_since(batch_scan_started)
                .as_secs_f64();
            observed_scan_timing.screen_membership_prepare_wall_seconds +=
                batch_timing.prepare_wall_seconds;
            observed_scan_timing.screen_pair_scan_merge_wall_seconds +=
                batch_timing.scan_merge_wall_seconds;
            observed_scan_timing.screen_heap_merge_task_seconds += batch_timing.merge_task_seconds;
            observed_scan_timing.screen_finalist_wall_seconds += batch_timing.finalize_wall_seconds;
            observed_scan_timing.screen_parallel_batches = observed_scan_timing
                .screen_parallel_batches
                .saturating_add(usize::from(batch_timing.parallel));
            observed_scan_timing.screen_serial_batches = observed_scan_timing
                .screen_serial_batches
                .saturating_add(usize::from(!batch_timing.parallel));
            observed_scan_timing.screen_source_rows_sum = observed_scan_timing
                .screen_source_rows_sum
                .saturating_add(batch_timing.source_rows);
            observed_scan_timing.screen_source_rows_max = observed_scan_timing
                .screen_source_rows_max
                .max(batch_timing.source_rows);
            observed_scan_timing.screen_memberships_sum = observed_scan_timing
                .screen_memberships_sum
                .saturating_add(batch_timing.memberships);
            if batch_screen.used_direct_fallback {
                observed_scan_timing.screen_batch_direct_fallbacks = observed_scan_timing
                    .screen_batch_direct_fallbacks
                    .saturating_add(1);
            } else {
                observed_scan_timing.screen_batch_pairs_scored_unique = observed_scan_timing
                    .screen_batch_pairs_scored_unique
                    .saturating_add(batch_screen.pairs_scored_unique);
                observed_scan_timing.screen_batch_score_offers = observed_scan_timing
                    .screen_batch_score_offers
                    .saturating_add(batch_screen.score_offers);
            }
            if batch_screen.units.len() != scan_unit_indices.len() {
                return Err(format!(
                    "fused Max6 batch scanner returned {} unit screens for {} requested units",
                    batch_screen.units.len(),
                    scan_unit_indices.len()
                ));
            }

            let selected_rows_by_unit = scan_unit_indices
                .iter()
                .map(|&index| unit_selected_rows[index].as_slice())
                .collect::<Vec<_>>();
            let local_pairs_by_unit = batch_screen
                .units
                .iter()
                .map(|screen| {
                    screen
                        .candidates
                        .iter()
                        .map(|candidate| (candidate.first, candidate.second))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let pair_plan = max6_shared_candidate_pair_plan(
                selected_rows_by_unit.as_slice(),
                local_pairs_by_unit.as_slice(),
            )?;
            let grm_int_started = Instant::now();
            let shared_grm_int_by_unit = if pair_plan.global_pairs.is_empty() {
                vec![None; scan_unit_indices.len()]
            } else {
                let score_chunk = |candidate_pairs: &[(usize, usize)]| {
                    grm_max6::grm_int_interaction_scores_packed(
                        batch_bits.as_slice(),
                        row_words,
                        n_samples,
                        batch_source_rows.len(),
                        candidate_pairs,
                        p_c,
                        no_grm_projector,
                        projected_y.as_slice(),
                        score_scale,
                    )
                };
                let scored_chunks = if let Some(pool) = pool {
                    pool.install(|| {
                        pair_plan
                            .global_pairs
                            .par_chunks(shared_grm_int_score_pairs.max(1))
                            .map(score_chunk)
                            .collect::<Result<Vec<_>, String>>()
                    })?
                } else {
                    pair_plan
                        .global_pairs
                        .chunks(shared_grm_int_score_pairs.max(1))
                        .map(score_chunk)
                        .collect::<Result<Vec<_>, String>>()?
                };
                let mut unique_chisq = Vec::with_capacity(pair_plan.global_pairs.len());
                let mut unique_invalid = Vec::with_capacity(pair_plan.global_pairs.len());
                let mut shared_diagnostics = grm_max6::GrmIntScoreDiagnostics::default();
                for batch in scored_chunks {
                    if batch.chisq.len() != batch.invalid.len() {
                        return Err(
                            "fused Max6 shared scores/statuses have different lengths".to_string()
                        );
                    }
                    unique_chisq.extend(batch.chisq);
                    unique_invalid.extend(batch.invalid);
                    shared_diagnostics.batches = shared_diagnostics
                        .batches
                        .saturating_add(batch.diagnostics.batches);
                    shared_diagnostics.max_workspace_bytes = shared_diagnostics
                        .max_workspace_bytes
                        .max(batch.diagnostics.max_workspace_bytes);
                    shared_diagnostics.elapsed_seconds += batch.diagnostics.elapsed_seconds;
                }
                if unique_chisq.len() != pair_plan.global_pairs.len()
                    || unique_invalid.len() != pair_plan.global_pairs.len()
                {
                    return Err(format!(
                        "fused Max6 scorer returned {} scores / {} statuses for {} unique pairs",
                        unique_chisq.len(),
                        unique_invalid.len(),
                        pair_plan.global_pairs.len()
                    ));
                }
                let diagnostic_recipient =
                    max6_shared_grm_int_diagnostic_recipient(&pair_plan.unit_pair_indices);
                pair_plan
                    .unit_pair_indices
                    .iter()
                    .enumerate()
                    .map(|(unit_index, pair_indices)| {
                        let mut chisq = Vec::with_capacity(pair_indices.len());
                        let mut invalid = Vec::with_capacity(pair_indices.len());
                        let mut diagnostics = grm_max6::GrmIntScoreDiagnostics::default();
                        for &pair_index in pair_indices {
                            chisq.push(*unique_chisq.get(pair_index).ok_or_else(|| {
                                format!("fused Max6 pair score index {pair_index} is out of range")
                            })?);
                            let status = *unique_invalid.get(pair_index).ok_or_else(|| {
                                format!("fused Max6 pair status index {pair_index} is out of range")
                            })?;
                            invalid.push(status);
                            match status {
                                Some(grm_max6::GrmIntInvalidScore::SingularEndpointDesign) => {
                                    diagnostics.singular_endpoint_design =
                                        diagnostics.singular_endpoint_design.saturating_add(1);
                                }
                                Some(
                                    grm_max6::GrmIntInvalidScore::NegligibleInteractionInformation,
                                ) => {
                                    diagnostics.negligible_residual_interaction_information =
                                        diagnostics
                                            .negligible_residual_interaction_information
                                            .saturating_add(1);
                                }
                                Some(grm_max6::GrmIntInvalidScore::Nonfinite) => {
                                    diagnostics.nonfinite = diagnostics.nonfinite.saturating_add(1);
                                }
                                None => {}
                            }
                        }
                        if diagnostic_recipient == Some(unit_index) {
                            diagnostics.batches = shared_diagnostics.batches;
                            diagnostics.max_workspace_bytes =
                                shared_diagnostics.max_workspace_bytes;
                            diagnostics.elapsed_seconds = shared_diagnostics.elapsed_seconds;
                        }
                        Ok(Some(grm_max6::GrmIntScoreBatch {
                            chisq,
                            invalid,
                            diagnostics,
                        }))
                    })
                    .collect::<Result<Vec<_>, String>>()?
            };
            observed_scan_timing.grm_int_batch_wall_seconds +=
                grm_int_started.elapsed().as_secs_f64();

            for (batch_index, &selection_index) in scan_unit_indices.iter().enumerate() {
                let selection = &selections[selection_index];
                let selected = selection.prepared_local.selected_global_rows.as_slice();
                let margins = OneCellMarginTable::build_for_rows(
                    selection.bits.bits(),
                    selection.bits.row_words,
                    selected,
                    train_fit.residualized_y.as_slice(),
                    n_samples,
                    &shared_y_lookup,
                )?;
                let precomputed = Max6PrecomputedScreen {
                    screen: batch_screen.units[batch_index].clone(),
                    kernel_seconds: screen_kernel_seconds / scan_unit_indices.len().max(1) as f64,
                    shared_grm_int: shared_grm_int_by_unit[batch_index].clone(),
                };
                let mut unit_beam_params = beam_params.clone();
                unit_beam_params.allow_parallel &= pool.is_none();
                let unit_result = scan_max6_grmmax6_unit(
                    &selection.unit,
                    selection.metadata.original_unit_index,
                    "window",
                    response,
                    engine,
                    importance,
                    perm_cfg,
                    ml_top_k,
                    ml_top_frac,
                    tree_cfg,
                    &selection.bits,
                    train_idx_local,
                    test_idx_local,
                    train_fit.residualized_y.as_slice(),
                    &shared_y_lookup,
                    &margins,
                    p_c,
                    no_grm_projector,
                    projected_y.as_slice(),
                    y,
                    score_scale,
                    scaled_total_quadratic,
                    &unit_beam_params,
                    final_top_k,
                    min_cell,
                    true,
                    false,
                    true,
                    Some(&selection.prepared_local),
                    None,
                    false,
                    false,
                    None,
                    false,
                    Some(precomputed),
                    None,
                )?;
                scans[selection_index] = Some(unit_result);
            }
        }
        let scans = scans
            .into_iter()
            .map(|scan| scan.ok_or_else(|| "fused Max6 batch lost a scan result".to_string()))
            .collect::<Result<Vec<_>, String>>()?;
        if scans.len() != selections.len() {
            return Err("fused Max6 window batch result count mismatch".to_string());
        }
        let _batch_elapsed = batch_scan_started.elapsed().as_secs_f64();
        let mut pair_text = String::with_capacity(1024 * 1024);
        let mut provenance_text = String::with_capacity(256 * 1024);
        for (selection, unit_result) in selections.iter().zip(scans.into_iter()) {
            let chrom = selection
                .metadata
                .spans
                .first()
                .map(|span| span.chrom.as_str())
                .ok_or_else(|| {
                    format!(
                        "fused Max6 window '{}' has no chromosome span",
                        selection.unit.label
                    )
                })?;
            pair_universe_accumulator.push_window(chrom, &selection.prepared_source)?;
            observed_scan_timing.accumulate_unit(unit_result.timing);
            screen_candidate_count =
                screen_candidate_count.saturating_add(unit_result.screen_pairs_passed);
            final_screen_subset_audit_pairs_checked = final_screen_subset_audit_pairs_checked
                .saturating_add(unit_result.final_screen_subset_audit_pairs_checked);
            if let Some(message) = unit_result.skipped_message {
                skipped_messages.push(message);
            }
            for record in unit_result.outputs.iter() {
                output_count = output_count.saturating_add(1);
                if pair_writer.is_some() {
                    let first_site = selection
                        .bits
                        .sites
                        .get(record.first_global_row)
                        .ok_or_else(|| {
                            format!(
                                "fused Max6 first site index out of range: {}",
                                record.first_global_row
                            )
                        })?;
                    let second_site = selection
                        .bits
                        .sites
                        .get(record.second_global_row)
                        .ok_or_else(|| {
                            format!(
                                "fused Max6 second site index out of range: {}",
                                record.second_global_row
                            )
                        })?;
                    let four_cell_counts = match record.four_cell_counts {
                        Some(counts) => counts,
                        None => max6_pair_four_cell_counts(
                            &selection.bits,
                            record.first_global_row,
                            record.second_global_row,
                            train_idx_local,
                        )?,
                    };
                    let best_channel =
                        max6_output_channel_label(record.best_channel, first_site, second_site);
                    let four_cell =
                        format_max6_four_cell_counts(four_cell_counts, first_site, second_site);
                    let min_count = four_cell_counts.iter().min().copied().unwrap_or(0);
                    use std::fmt::Write as _;
                    if garfield_snp_name_missing(first_site.snp()) {
                        write!(&mut pair_text, "{}_{}", first_site.chrom(), first_site.pos)
                            .map_err(|error| error.to_string())?;
                    } else {
                        pair_text.push_str(first_site.snp().trim());
                    }
                    pair_text.push('\t');
                    if garfield_snp_name_missing(second_site.snp()) {
                        write!(
                            &mut pair_text,
                            "{}_{}",
                            second_site.chrom(),
                            second_site.pos
                        )
                        .map_err(|error| error.to_string())?;
                    } else {
                        pair_text.push_str(second_site.snp().trim());
                    }
                    writeln!(
                        &mut pair_text,
                        "\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        first_site.chrom(),
                        first_site.pos,
                        second_site.chrom(),
                        second_site.pos,
                        record.candidate_rank,
                        format_max6_score_value(record.max6),
                        best_channel,
                        four_cell,
                        min_count,
                        format_max6_score_value(record.dosage),
                        format_max6_pvalue(max6_chi2_reference_pvalue(record.dosage)),
                    )
                    .map_err(|error| error.to_string())?;
                }
                if provenance_writer.is_some() {
                    output_row_index = output_row_index.saturating_add(1);
                    use std::fmt::Write as _;
                    writeln!(
                        &mut provenance_text,
                        "{}\t{}\t{}",
                        output_row_index, record.unit_index, selection.unit.label
                    )
                    .map_err(|error| error.to_string())?;
                }
            }
        }
        if let Some(writer) = pair_writer.as_ref() {
            writer.send_text(&mut pair_text)?;
        }
        if let Some(writer) = provenance_writer.as_ref() {
            writer.send_text(&mut provenance_text)?;
        }
        Ok(())
    };

    garfield_stage_progress_notify(
        progress_callback,
        "garfield_window_fused_scan",
        0,
        1,
        Some("starting Wscan"),
    )
    .map_err(|error| error.to_string())?;
    let mut notify_stream_progress = |done: usize, total: usize| {
        garfield_stage_progress_notify(
            progress_callback,
            "garfield_window_fused_scan",
            done,
            total,
            None,
        )
        .map_err(|error| error.to_string())
    };
    let on_stream_progress = if progress_callback.is_some() {
        Some(&mut notify_stream_progress as &mut Max6WindowStreamProgressCallback<'_>)
    } else {
        None
    };
    let stream = prepare_max6_window_streamed_ld_with_callback(
        prefix,
        sample_ids,
        sample_indices,
        extension,
        step,
        scan_bimranges,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        threads,
        logic_mem_budget_bytes,
        ld_tag_r2,
        None,
        !write_provenance,
        Some(&mut on_batch),
        on_stream_progress,
    )?;
    drop(on_batch);
    if let Some(writer) = pair_writer {
        writer.finish()?;
    }
    if let Some(writer) = provenance_writer {
        writer.finish()?;
    }

    let (pair_universe_size, active_tag_count, pair_occurrence_count) =
        pair_universe_accumulator.finish()?;
    if pair_occurrence_count != stream.window_pair_count {
        return Err(format!(
            "fused Max6 streamed pair occurrence count mismatch: chromosome accumulator={pair_occurrence_count}, scan={}",
            stream.window_pair_count
        ));
    }
    let (bonferroni_q_cut, bonferroni_p_cut) = max6_bonferroni_cutoff(pair_universe_size)?;
    if let (Some(tmp_path), Some(final_path)) = (pair_tmp_path.as_ref(), pair_path.as_ref()) {
        fs::rename(tmp_path, final_path).map_err(|error| {
            format!(
                "publish fused Max6 pair output '{}' from '{}': {error}",
                final_path,
                tmp_path.display()
            )
        })?;
    }
    let scan_wall_seconds = scan_started.elapsed().as_secs_f64();
    let total_wall_seconds = total_wall_t0.elapsed().as_secs_f64();
    let mut result = GarfieldLogicPipelineResult::default();
    result.n_samples = n_samples;
    result.logic_bits_backend = "streamed".to_string();
    result.logic_bits_bytes = stream.max_task_batch_bytes;
    result.units_total = stream.distinct_window_count;
    result.units_scanned = stream.selected_window_count;
    result.timing_total_wall_s = total_wall_seconds;
    result.timing_scan_wall_s = scan_wall_seconds;
    result.max6_observed_scan_timing = observed_scan_timing;
    result.skipped_messages = skipped_messages;
    result.train_fit = train_fit;
    result.max6_pairs_tsv = pair_path;
    result.max6_pairs_provenance_tsv = provenance_path;
    result.max6_pair_universe_size = pair_universe_size;
    result.max6_bonferroni_q_cut = Some(bonferroni_q_cut);
    result.max6_bonferroni_p_cut = Some(bonferroni_p_cut);
    result.max6_screen_pair_universe_size = pair_universe_size;
    result.max6_screen_window_pair_occurrences = stream.window_pair_count;
    result.max6_screen_candidate_count = screen_candidate_count;
    result.max6_final_screen_subset_audit_passed = Some(true);
    result.max6_final_screen_subset_audit_pairs_checked = final_screen_subset_audit_pairs_checked;
    result.max6_score_scale = Some(score_scale);
    result.max6_pair_count = output_count;
    garfield_stage_progress_notify(
        progress_callback,
        "garfield_window_fused_scan",
        stream.source_marker_count,
        stream.source_marker_count.max(1),
        Some(&format!(
            "{} windows; {} LD tags; {} pair results",
            stream.selected_window_count, active_tag_count, output_count
        )),
    )
    .map_err(|error| error.to_string())?;
    let _ = seed;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_max6_window_stream(
    prefix: &str,
    sample_ids: &[String],
    sample_indices: &[usize],
    extension: usize,
    step: Option<usize>,
    scan_bimranges: &[GarfieldScanBimRange],
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    threads: usize,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<Max6WindowStreamPrep, String> {
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err("maf_threshold must be within [0, 0.5]".to_string());
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err("max_missing_rate must be within [0, 1.0]".to_string());
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err("het_threshold must be within [0, 1.0]".to_string());
    }
    if sample_ids.is_empty() || sample_ids.len() != sample_indices.len() {
        return Err(format!(
            "Max6 window stream requires matching nonempty sample IDs/indices; got {}/{}",
            sample_ids.len(),
            sample_indices.len()
        ));
    }

    let mut source = Max6WindowRangeSource::new(
        prefix,
        scan_bimranges,
        extension,
        false,
        Arc::new(Mutex::new(HashSet::new())),
        Arc::new(Mutex::new(HashSet::new())),
    )?;
    let source_marker_count = source.n_source_rows;
    let n_samples_total = source.bed.n_samples();
    if sample_indices.iter().any(|&index| index >= n_samples_total) {
        return Err("selected sample index out of range for Max6 window stream".to_string());
    }
    let n_samples = sample_indices.len();
    let row_words = words_for_samples(n_samples);
    let bytes_per_variant = source.bed.bytes_per_snp();
    let sample_identity =
        n_samples == n_samples_total && sample_indices_are_identity(sample_indices);
    let sample_plan = build_logic_sample_plan(sample_indices);
    let prep_threads = effective_threads_local(threads).max(1);
    let pool = if prep_threads > 1 {
        crate::stats_common::get_cached_pool(prep_threads).map_err(|error| error.to_string())?
    } else {
        None
    };

    let mut global_site_keep = vec![false; source_marker_count];
    let mut window_sites = Vec::<Max6WindowPositionSite>::new();
    let mut qc_to_packed = Vec::<usize>::new();
    let mut packed_source_rows = Vec::<usize>::new();
    let mut row_flip_keep = Vec::<bool>::new();
    let mut chrom_pool = HashMap::<String, Arc<str>>::new();
    let mut fail_missing = 0usize;
    let mut fail_het = 0usize;
    let mut fail_no_non_missing = 0usize;
    let mut fail_maf = 0usize;
    let mut fail_non_simple_snp = 0usize;
    let mut het_sites = 0usize;
    let mut raw_missing_sites = 0usize;

    let extension_bp = (extension.max(1) as i64).saturating_mul(2);
    let in_pack_scope = |site: &SiteInfo| {
        scan_bimranges.is_empty()
            || scan_bimranges.iter().any(|range| {
                if !same_normalized_chrom(site.chrom.as_str(), range.chrom.as_str()) {
                    return false;
                }
                let start = i64::from(range.bp_start)
                    .saturating_sub(extension_bp)
                    .max(0);
                let end = i64::from(range.bp_end)
                    .saturating_add(extension_bp)
                    .min(i64::from(i32::MAX));
                (i64::from(site.pos) >= start) && (i64::from(site.pos) <= end)
            })
    };

    let max_writer_block_rows = 8192usize;
    let mut bits_writer = Max6StreamingLogicBitsAccumulator::new(row_words, max_writer_block_rows)?;
    crate::pipeline::run_double_buffer(
        2,
        || Max6WindowStreamChunk::new(MAX6_WINDOW_STREAM_CHUNK_ROWS, bytes_per_variant, row_words),
        |chunk| match source.fill_chunk(chunk) {
            Ok(more) => more,
            Err(error) => {
                chunk.clear();
                chunk.error = Some(error);
                false
            }
        },
        |chunk| {
            if let Some(error) = chunk.error.take() {
                return Err(error);
            }
            let rows = chunk.sites.len();
            if rows == 0 {
                return Ok(());
            }
            let required_words = rows
                .checked_mul(row_words)
                .ok_or_else(|| "Max6 window logic chunk size overflow".to_string())?;
            chunk.logic_bits.resize(required_words, 0);
            chunk.logic_bits.fill(0);
            chunk.qc.resize(rows, Max6FusedRowQc::default());
            let qc_batch = &mut chunk.qc;
            let row_bits = &mut chunk.logic_bits;
            let sites = &chunk.sites;
            let packed_genotypes = &chunk.packed_genotypes;
            let mut process_rows = || {
                row_bits
                    .par_chunks_mut(row_words)
                    .zip(qc_batch.par_iter_mut())
                    .enumerate()
                    .for_each_init(
                        || vec![0u64; row_words],
                        |missing_words, (offset, (dst_words, qc_dst))| {
                            let site = &sites[offset];
                            let row_start = offset * bytes_per_variant;
                            let row = &packed_genotypes[row_start..row_start + bytes_per_variant];
                            let simple_snp_alleles = !snps_only
                                || (is_simple_snp_allele(&site.ref_allele)
                                    && is_simple_snp_allele(&site.alt_allele));
                            *qc_dst = scan_max6_row_qc_and_pack_logic(
                                row,
                                n_samples,
                                sample_identity,
                                sample_plan.as_slice(),
                                maf_threshold,
                                max_missing_rate,
                                simple_snp_alleles,
                                dst_words,
                                missing_words.as_mut_slice(),
                            );
                        },
                    );
            };
            if let Some(pool) = pool.as_ref() {
                pool.install(process_rows);
            } else {
                process_rows();
            }
            check_ctrlc()?;

            let mut packed_rows_in_chunk = 0usize;
            for offset in 0..rows {
                let qc = chunk.qc[offset];
                let source_row = chunk.source_rows[offset];
                if qc.keep {
                    global_site_keep[source_row] = true;
                    let site = &chunk.sites[offset];
                    let chrom_key = normalize_chrom(site.chrom.as_str());
                    let chrom = chrom_pool
                        .entry(chrom_key.clone())
                        .or_insert_with(|| Arc::<str>::from(chrom_key.as_str()))
                        .clone();
                    window_sites.push(Max6WindowPositionSite {
                        chrom,
                        pos: site.pos,
                    });
                    if in_pack_scope(site) {
                        let src_start = offset * row_words;
                        let dst_start = packed_rows_in_chunk * row_words;
                        if src_start != dst_start {
                            chunk
                                .logic_bits
                                .copy_within(src_start..src_start + row_words, dst_start);
                        }
                        qc_to_packed.push(packed_source_rows.len());
                        packed_source_rows.push(source_row);
                        row_flip_keep.push(qc.row_flip);
                        packed_rows_in_chunk += 1;
                    } else {
                        qc_to_packed.push(usize::MAX);
                    }
                } else {
                    match qc.reason {
                        MAX6_FUSED_QC_FAIL_MISSING => fail_missing += 1,
                        MAX6_FUSED_QC_FAIL_HET => fail_het += 1,
                        MAX6_FUSED_QC_FAIL_NO_HOMOZYGOUS => fail_no_non_missing += 1,
                        MAX6_FUSED_QC_FAIL_MAF => fail_maf += 1,
                        MAX6_FUSED_QC_FAIL_NON_SIMPLE_SNP => fail_non_simple_snp += 1,
                        _ => {}
                    }
                }
                if qc.has_het {
                    het_sites += 1;
                }
                if qc.has_raw_missing {
                    raw_missing_sites += 1;
                }
            }
            chunk.logic_bits.truncate(packed_rows_in_chunk * row_words);
            bits_writer.append_rows(chunk.logic_bits.as_slice())?;
            if let Some(tracker) = mem_tracker {
                tracker.sample_now();
            }
            Ok(())
        },
    )?;

    if window_sites.is_empty() {
        if !scan_bimranges.is_empty() && source.seen_selected_chroms.is_empty() {
            return Err("--bimrange matched no BIM/PVAR chromosomes".to_string());
        }
        return Err(format_zero_sites_pure_line_error(
            n_samples,
            n_samples_total,
            source_marker_count,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            fail_missing,
            fail_het,
            fail_no_non_missing,
            fail_maf,
            fail_non_simple_snp,
            het_sites,
            raw_missing_sites,
        ));
    }
    if packed_source_rows.is_empty() {
        return Err("--bimrange matched no QC-passing markers in the selected window span".into());
    }

    let (units, source_unit_indices) = if scan_bimranges.is_empty() {
        let units = build_logic_units(
            window_sites.as_slice(),
            "window",
            None,
            None,
            extension,
            step,
        )?;
        let source_indices = (0..units.len()).collect();
        (units, source_indices)
    } else {
        let step_eff = step.unwrap_or((extension.max(1) / 2).max(1)).max(1);
        let (mut units, source_indices) = build_window_units_overlapping_bimranges(
            window_sites.as_slice(),
            extension,
            step_eff,
            scan_bimranges,
        )?;
        for unit in units.iter_mut() {
            for row in unit.indices.iter_mut() {
                let Some(&packed_row) = qc_to_packed.get(*row) else {
                    return Err(format!(
                        "selected window row {} is outside the range QC map",
                        *row
                    ));
                };
                if packed_row == usize::MAX {
                    return Err(format!(
                        "selected window row {} was not packed inside the range span",
                        *row
                    ));
                }
                *row = packed_row;
            }
        }
        (units, source_indices)
    };
    if units.is_empty() {
        return Err(if scan_bimranges.is_empty() {
            "no scan units were built from the globally QC-passing markers".to_string()
        } else {
            "--bimrange matched no Max6 scan units".to_string()
        });
    }

    let (bits_flat, bits_mmap, bits_mmap_words, bits_mmap_owner) = bits_writer.finish()?;
    let empty_text: Arc<str> = Arc::from("");
    let logic_sites = window_sites
        .iter()
        .zip(qc_to_packed.iter())
        .filter_map(|(site, &packed_row)| {
            (packed_row != usize::MAX).then(|| {
                GarfieldLogicSite::owned(
                    site.chrom.clone(),
                    site.pos,
                    empty_text.clone(),
                    empty_text.clone(),
                    empty_text.clone(),
                    GarfieldLogicSiteMode::Bin,
                )
            })
        })
        .collect::<Vec<_>>();
    if logic_sites.len() != packed_source_rows.len() {
        return Err(format!(
            "streamed Max6 logic metadata mismatch: sites={}, rows={}",
            logic_sites.len(),
            packed_source_rows.len()
        ));
    }
    let logic_bits = GarfieldLogicBits {
        bits_flat,
        bits_mmap,
        bits_mmap_words,
        bits_mmap_owner: Some(bits_mmap_owner),
        bits_hi_flat: None,
        row_words,
        sample_ids: sample_ids.to_vec(),
        sites: logic_sites,
        group_ids: build_logic_mode_group_ids(packed_source_rows.len(), GarfieldBinMode::Bin),
        n_samples,
    };
    Ok(Max6WindowStreamPrep {
        global_site_keep,
        packed_source_rows,
        logic_bits,
        units,
        source_unit_indices,
    })
}
