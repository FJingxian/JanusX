use super::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub(super) const MAX6_GENE_STREAM_UNSUPPORTED_ORDER: &str =
    "__MAX6_GENE_STREAM_UNSUPPORTED_ORDER__";

struct Max6GeneOutputSpoolEntry {
    offset: u64,
    length: usize,
    output_rows: usize,
    unit_index: usize,
    label: String,
}

pub(super) struct Max6GeneOutputSpool {
    path: std::path::PathBuf,
    writer: std::io::BufWriter<std::fs::File>,
    next_offset: u64,
    entries: Vec<Option<Max6GeneOutputSpoolEntry>>,
}

impl Max6GeneOutputSpool {
    pub(super) fn new(n_units: usize) -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "janusx_garfield_gene_order_{}_{}.tmp",
            std::process::id(),
            GARFIELD_BITS_MMAP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                format!(
                    "create Max6 gene output-order spool '{}': {error}",
                    path.display()
                )
            })?;
        Ok(Self {
            path,
            writer: std::io::BufWriter::new(file),
            next_offset: 0,
            entries: (0..n_units).map(|_| None).collect(),
        })
    }

    pub(super) fn append(
        &mut self,
        unit_index: usize,
        label: &str,
        pair_text: &str,
        output_rows: usize,
    ) -> Result<(), String> {
        let n_units = self.entries.len();
        let slot = self.entries.get_mut(unit_index).ok_or_else(|| {
            format!(
                "Max6 gene output unit index {unit_index} is outside {} input units",
                n_units
            )
        })?;
        if slot.is_some() {
            return Err(format!(
                "Max6 gene output unit index {unit_index} was emitted more than once"
            ));
        }
        let offset = self.next_offset;
        self.writer
            .write_all(pair_text.as_bytes())
            .map_err(|error| format!("write Max6 gene output spool: {error}"))?;
        self.next_offset = self.next_offset.saturating_add(pair_text.len() as u64);
        *slot = Some(Max6GeneOutputSpoolEntry {
            offset,
            length: pair_text.len(),
            output_rows,
            unit_index,
            label: label.to_string(),
        });
        Ok(())
    }

    pub(super) fn replay(
        mut self,
        pair_writer: Option<&crate::stats_common::AsyncTsvWriter>,
        provenance_writer: Option<&crate::stats_common::AsyncTsvWriter>,
    ) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|error| format!("flush Max6 gene output spool: {error}"))?;
        let mut reader = std::fs::File::open(&self.path).map_err(|error| {
            format!(
                "open Max6 gene output spool '{}': {error}",
                self.path.display()
            )
        })?;
        let mut output_row_index = 0usize;
        for entry in self.entries.iter().flatten() {
            if let Some(writer) = pair_writer {
                reader
                    .seek(std::io::SeekFrom::Start(entry.offset))
                    .map_err(|error| format!("seek Max6 gene output spool: {error}"))?;
                let mut bytes = vec![0u8; entry.length];
                reader
                    .read_exact(bytes.as_mut_slice())
                    .map_err(|error| format!("read Max6 gene output spool: {error}"))?;
                let mut text = String::from_utf8(bytes)
                    .map_err(|error| format!("decode Max6 gene output spool: {error}"))?;
                writer.send_text(&mut text)?;
            }
            if let Some(writer) = provenance_writer {
                let mut text = String::with_capacity(entry.output_rows.saturating_mul(64));
                for _ in 0..entry.output_rows {
                    output_row_index = output_row_index.saturating_add(1);
                    use std::fmt::Write as _;
                    writeln!(
                        &mut text,
                        "{}\t{}\t{}",
                        output_row_index, entry.unit_index, entry.label
                    )
                    .map_err(|error| error.to_string())?;
                }
                writer.send_text(&mut text)?;
            }
        }
        drop(reader);
        fs::remove_file(&self.path).map_err(|error| {
            format!(
                "remove Max6 gene output spool '{}': {error}",
                self.path.display()
            )
        })?;
        Ok(())
    }
}

impl Drop for Max6GeneOutputSpool {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Clone, Debug)]
pub(super) struct Max6GeneInterval {
    pub(super) unit_index: usize,
    pub(super) span_index: usize,
    pub(super) label: String,
    pub(super) chrom: String,
    pub(super) start: i32,
    pub(super) end: i32,
}

#[derive(Clone, Debug)]
pub(super) struct Max6GeneIntervalCompletion<T> {
    pub(super) unit_index: usize,
    pub(super) span_index: usize,
    pub(super) label: String,
    pub(super) chrom: String,
    pub(super) start: i32,
    pub(super) end: i32,
    pub(super) rows: Vec<T>,
}

fn max6_gene_interval_completion<T>(
    interval: Max6GeneInterval,
    rows: Vec<T>,
) -> Max6GeneIntervalCompletion<T> {
    Max6GeneIntervalCompletion {
        unit_index: interval.unit_index,
        span_index: interval.span_index,
        label: interval.label,
        chrom: interval.chrom,
        start: interval.start,
        end: interval.end,
        rows,
    }
}

struct Max6ActiveGeneInterval {
    interval: Max6GeneInterval,
}

/// Routes ordered variants to active gene or geneset spans without
/// retaining rows after their last containing interval has closed.
pub(super) struct Max6GeneIntervalSweep<T> {
    intervals_by_chrom: HashMap<String, VecDeque<Max6GeneInterval>>,
    current_chrom: Option<String>,
    current_pos: Option<i32>,
    seen_chroms: HashSet<String>,
    pending: VecDeque<Max6GeneInterval>,
    active: Vec<Max6ActiveGeneInterval>,
    retained_rows: VecDeque<(i32, T)>,
}

impl<T: Clone> Max6GeneIntervalSweep<T> {
    pub(super) fn new(intervals: Vec<Max6GeneInterval>) -> Result<Self, String> {
        let mut intervals_by_chrom = HashMap::<String, Vec<Max6GeneInterval>>::new();
        for mut interval in intervals {
            interval.chrom = normalize_chrom(interval.chrom.as_str());
            if interval.chrom.is_empty() {
                return Err(format!(
                    "Max6 gene interval '{}' has an empty chromosome",
                    interval.label
                ));
            }
            (interval.start, interval.end) =
                normalize_interval_bounds(interval.start, interval.end);
            intervals_by_chrom
                .entry(interval.chrom.clone())
                .or_default()
                .push(interval);
        }
        let intervals_by_chrom = intervals_by_chrom
            .into_iter()
            .map(|(chrom, mut intervals)| {
                intervals.sort_by(|left, right| {
                    left.start
                        .cmp(&right.start)
                        .then_with(|| left.end.cmp(&right.end))
                        .then_with(|| left.unit_index.cmp(&right.unit_index))
                        .then_with(|| left.span_index.cmp(&right.span_index))
                });
                (chrom, VecDeque::from(intervals))
            })
            .collect();
        Ok(Self {
            intervals_by_chrom,
            current_chrom: None,
            current_pos: None,
            seen_chroms: HashSet::new(),
            pending: VecDeque::new(),
            active: Vec::new(),
            retained_rows: VecDeque::new(),
        })
    }

    #[cfg(test)]
    pub(super) fn push_variant(
        &mut self,
        chrom: &str,
        pos: i32,
        row: Option<T>,
    ) -> Result<Vec<Max6GeneIntervalCompletion<T>>, String> {
        let mut completed = self.advance_to(chrom, pos)?;
        if let Some(row) = row {
            self.push_current_row(row);
        }
        completed.sort_by_key(|unit| (unit.unit_index, unit.span_index));
        Ok(completed)
    }

    pub(super) fn advance_to(
        &mut self,
        chrom: &str,
        pos: i32,
    ) -> Result<Vec<Max6GeneIntervalCompletion<T>>, String> {
        let chrom = normalize_chrom(chrom);
        if chrom.is_empty() {
            return Err("Max6 gene stream encountered an empty chromosome".to_string());
        }
        let mut completed = Vec::new();
        if self.current_chrom.as_deref() != Some(chrom.as_str()) {
            if self.seen_chroms.contains(chrom.as_str()) {
                return Err(format!(
                    "{MAX6_GENE_STREAM_UNSUPPORTED_ORDER}: chromosome '{chrom}' reappeared"
                ));
            }
            completed.extend(self.finish_current_chromosome());
            if let Some(previous) = self.current_chrom.replace(chrom.clone()) {
                self.seen_chroms.insert(previous);
            }
            self.current_pos = None;
            self.pending = self
                .intervals_by_chrom
                .remove(chrom.as_str())
                .unwrap_or_default();
        }
        if self.current_pos.is_some_and(|previous| pos < previous) {
            return Err(format!(
                "{MAX6_GENE_STREAM_UNSUPPORTED_ORDER}: position {pos} follows {} on chromosome '{chrom}'",
                self.current_pos.unwrap_or(pos)
            ));
        }
        self.current_pos = Some(pos);

        let active_groups = std::mem::take(&mut self.active);
        let mut retained = Vec::with_capacity(active_groups.len());
        for active in active_groups {
            if active.interval.end < pos {
                let rows = self.rows_for_interval(&active.interval);
                completed.push(max6_gene_interval_completion(active.interval, rows));
            } else {
                retained.push(active);
            }
        }
        self.active = retained;

        while self
            .pending
            .front()
            .is_some_and(|interval| interval.start <= pos)
        {
            let interval = self
                .pending
                .pop_front()
                .expect("front interval was checked");
            if interval.end >= pos {
                self.active.push(Max6ActiveGeneInterval { interval });
            } else {
                completed.push(max6_gene_interval_completion(interval, Vec::new()));
            }
        }
        self.prune_retained_rows();
        completed.sort_by_key(|unit| (unit.unit_index, unit.span_index));
        Ok(completed)
    }

    pub(super) fn has_active_intervals(&self) -> bool {
        !self.active.is_empty()
    }

    pub(super) fn push_current_row(&mut self, row: T) {
        if self.active.is_empty() {
            return;
        }
        if let Some(pos) = self.current_pos {
            self.retained_rows.push_back((pos, row));
        }
    }

    pub(super) fn finish(mut self) -> Result<Vec<Max6GeneIntervalCompletion<T>>, String> {
        let mut completed = self.finish_current_chromosome();
        for intervals in self
            .intervals_by_chrom
            .drain()
            .map(|(_, intervals)| intervals)
        {
            completed.extend(
                intervals
                    .into_iter()
                    .map(|interval| max6_gene_interval_completion(interval, Vec::new())),
            );
        }
        completed.sort_by_key(|unit| (unit.unit_index, unit.span_index));
        Ok(completed)
    }

    fn finish_current_chromosome(&mut self) -> Vec<Max6GeneIntervalCompletion<T>> {
        self.current_pos = None;
        let active_groups = std::mem::take(&mut self.active);
        let mut completed = active_groups
            .into_iter()
            .map(|active| {
                let rows = self.rows_for_interval(&active.interval);
                max6_gene_interval_completion(active.interval, rows)
            })
            .collect::<Vec<_>>();
        completed.extend(
            self.pending
                .drain(..)
                .map(|interval| max6_gene_interval_completion(interval, Vec::new())),
        );
        self.retained_rows.clear();
        completed.sort_by_key(|unit| (unit.unit_index, unit.span_index));
        completed
    }

    fn rows_for_interval(&self, interval: &Max6GeneInterval) -> Vec<T> {
        self.retained_rows
            .iter()
            .filter(|(pos, _)| interval.start <= *pos && *pos <= interval.end)
            .map(|(_, row)| row.clone())
            .collect()
    }

    fn prune_retained_rows(&mut self) {
        let Some(earliest_active_start) =
            self.active.iter().map(|active| active.interval.start).min()
        else {
            self.retained_rows.clear();
            return;
        };
        while self
            .retained_rows
            .front()
            .is_some_and(|(pos, _)| *pos < earliest_active_start)
        {
            self.retained_rows.pop_front();
        }
    }
}

fn max6_interval_unit_task(
    unit_index: usize,
    label: String,
    unit_kind: &str,
    spans: Vec<GarfieldUnitSpan>,
    rows: Vec<Arc<Max6StreamWindowRow>>,
    ld_tags_preselected: bool,
) -> Max6StreamWindowTask {
    let unit = GarfieldLogicUnit {
        label,
        // The task needs only the label/span until the shared rows are
        // materialized for LD selection.
        indices: Vec::new(),
        spans,
    };
    Max6StreamWindowTask {
        original_unit_index: unit_index,
        unit_kind: unit_kind.to_string(),
        ld_tags_preselected,
        unit,
        source_rows: Vec::new(),
        sites: Vec::new(),
        row_flip: Vec::new(),
        logic_words: Vec::new(),
        support: Vec::new(),
        block_support: Vec::new(),
        shared_gene_rows: Some(rows),
    }
}

struct Max6GeneSetRowAccumulator {
    labels: Vec<String>,
    spans: Vec<Vec<GarfieldUnitSpan>>,
    completed_spans: Vec<Vec<Option<Vec<Arc<Max6StreamWindowRow>>>>>,
    completed_counts: Vec<usize>,
}

impl Max6GeneSetRowAccumulator {
    fn new(groups: &[Vec<(String, i32, i32)>], labels: &[String]) -> Result<Self, String> {
        if groups.len() != labels.len() {
            return Err(format!(
                "Max6 geneset stream received {} groups and {} labels",
                groups.len(),
                labels.len()
            ));
        }
        if groups.iter().any(Vec::is_empty) {
            return Err("Max6 geneset stream requires at least one interval per unit".to_string());
        }
        let spans = groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|(chrom, bp_start, bp_end)| GarfieldUnitSpan {
                        chrom: chrom.clone(),
                        bp_start: *bp_start,
                        bp_end: *bp_end,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let completed_spans = spans
            .iter()
            .map(|unit_spans| vec![None; unit_spans.len()])
            .collect::<Vec<_>>();
        Ok(Self {
            labels: labels.to_vec(),
            spans,
            completed_spans,
            completed_counts: vec![0; groups.len()],
        })
    }

    fn push(
        &mut self,
        unit_index: usize,
        span_index: usize,
        rows: Vec<Arc<Max6StreamWindowRow>>,
    ) -> Result<Option<Max6StreamWindowTask>, String> {
        let label = self
            .labels
            .get(unit_index)
            .ok_or_else(|| format!("Max6 geneset unit index {unit_index} is out of range"))?;
        let span_rows = self
            .completed_spans
            .get_mut(unit_index)
            .and_then(|spans| spans.get_mut(span_index))
            .ok_or_else(|| {
                format!("Max6 geneset span {span_index} is out of range for '{label}'")
            })?;
        if span_rows.is_some() {
            return Err(format!(
                "Max6 geneset span {span_index} for '{label}' completed more than once"
            ));
        }
        *span_rows = Some(rows);
        self.completed_counts[unit_index] += 1;
        if self.completed_counts[unit_index] != self.spans[unit_index].len() {
            return Ok(None);
        }

        let mut rows = self.completed_spans[unit_index]
            .iter_mut()
            .flat_map(|span_rows| span_rows.take().unwrap_or_default())
            .collect::<Vec<_>>();
        rows.sort_unstable_by_key(|row| row.source_row);
        rows.dedup_by_key(|row| row.source_row);
        Ok(Some(max6_interval_unit_task(
            unit_index,
            label.clone(),
            "geneset",
            self.spans[unit_index].clone(),
            rows,
            true,
        )))
    }
}

fn prune_max6_geneset_span_rows(
    mut rows: Vec<Arc<Max6StreamWindowRow>>,
    spans: &[GarfieldUnitSpan],
    span_index: usize,
    row_words: usize,
    n_samples: usize,
    ld_tag_r2: f64,
) -> Result<Vec<Arc<Max6StreamWindowRow>>, String> {
    let span_unit = GarfieldLogicUnit {
        label: "geneset span".to_string(),
        indices: Vec::new(),
        spans: spans.to_vec(),
    };
    rows.retain(|row| {
        nearest_unit_span_index(
            &span_unit,
            row.site.garfield_chrom(),
            row.site.garfield_pos(),
        ) == Some(span_index)
    });
    if rows.len() < 2 {
        return Ok(rows);
    }

    let mut bits = Vec::with_capacity(rows.len().saturating_mul(row_words));
    let mut sites = Vec::with_capacity(rows.len());
    let mut row_flip = Vec::with_capacity(rows.len());
    let support_cache_required = ld_tag_r2 < 1.0;
    let block_words = garfield_ld_block_words();
    let block_count = row_words.div_ceil(block_words);
    let mut support = Vec::with_capacity(if support_cache_required {
        rows.len()
    } else {
        0
    });
    let mut block_support = Vec::with_capacity(if support_cache_required {
        rows.len().saturating_mul(block_count)
    } else {
        0
    });
    for row in rows.iter() {
        let words = row.logic_words.as_ref().ok_or_else(|| {
            format!(
                "streamed Max6 geneset span includes unpacked source row {}",
                row.source_row
            )
        })?;
        if words.len() != row_words {
            return Err(
                "streamed Max6 geneset span has an inconsistent packed row width".to_string(),
            );
        }
        bits.extend_from_slice(words);
        sites.push(row.site.clone());
        row_flip.push(row.row_flip);
        if support_cache_required {
            support.push(row.support.ok_or_else(|| {
                "streamed Max6 geneset span is missing its LD support cache".to_string()
            })?);
            let values = row.block_support.as_ref().ok_or_else(|| {
                "streamed Max6 geneset span is missing its LD block support cache".to_string()
            })?;
            block_support.extend_from_slice(values);
        }
    }
    let n_rows = rows.len();
    let logic_bits = GarfieldLogicBits {
        bits_flat: bits,
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words,
        sample_ids: Vec::new(),
        sites: build_logic_sites_from_metadata_owned_direct(
            sites,
            row_flip.as_slice(),
            GarfieldBinMode::Bin,
        ),
        group_ids: build_logic_mode_group_ids(n_rows, GarfieldBinMode::Bin),
        n_samples,
    };
    let support_cache = support_cache_required.then_some(GarfieldLdSupportCache {
        n_rows,
        row_words,
        block_words,
        block_count,
        support,
        block_support,
        sparse_global_to_local: None,
    });
    let candidates = (0..n_rows).collect::<Vec<_>>();
    // The packed stream rows already contain only selected samples, in local
    // order. LD selection must use that packed coordinate system rather than
    // the source PGEN sample indices used while decoding.
    let local_sample_indices = (0..n_samples).collect::<Vec<_>>();
    let selected = select_max6_ld_tag_rows_impl(
        candidates.as_slice(),
        &logic_bits,
        local_sample_indices.as_slice(),
        None,
        ld_tag_r2,
        false,
        support_cache.as_ref(),
    )?;
    let mut selected_rows = selected
        .selected_global_rows
        .into_iter()
        .map(|row| rows[row].clone())
        .collect::<Vec<_>>();
    selected_rows.sort_unstable_by_key(|row| row.source_row);
    rows.clear();
    Ok(selected_rows)
}

#[allow(clippy::too_many_arguments)]
fn flush_max6_gene_stream_tasks(
    tasks: &mut Vec<Max6StreamWindowTask>,
    sample_indices: &[usize],
    row_words: usize,
    n_samples: usize,
    ld_tag_r2: f64,
    pool: Option<&rayon::ThreadPool>,
    window_pair_count: &mut usize,
    any_eligible_unit: &mut bool,
    max_task_batch_bytes: &mut usize,
    task_batch_bytes: &mut usize,
    on_batch: &mut Option<&mut Max6WindowFusedBatchCallback<'_>>,
) -> Result<(), String> {
    if tasks.is_empty() {
        return Ok(());
    }
    let batch_bytes = tasks.iter().fold(0usize, |total, task| {
        total.saturating_add(max6_stream_window_task_materialized_bytes(task, row_words))
    });
    *max_task_batch_bytes = (*max_task_batch_bytes).max(batch_bytes);
    let mut unused_units = Vec::new();
    let mut unused_screen_units = Vec::new();
    let mut unused_original_unit_indices = Vec::new();
    let mut unused_active_source_rows = Vec::new();
    let selections = process_max6_stream_window_batch(
        tasks,
        sample_indices,
        row_words,
        n_samples,
        ld_tag_r2,
        pool,
        &mut unused_units,
        &mut unused_screen_units,
        &mut unused_original_unit_indices,
        &mut unused_active_source_rows,
        window_pair_count,
        any_eligible_unit,
        false,
    )?;
    if let Some(callback) = on_batch.as_deref_mut() {
        callback(&selections, pool)?;
    }
    *task_batch_bytes = 0;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_max6_interval_streamed_ld_with_callback(
    unit_kind: &str,
    prefix: &str,
    sample_ids: &[String],
    sample_indices: &[usize],
    groups: &[Vec<(String, i32, i32)>],
    group_names: &[String],
    scan_bimranges: &[GarfieldScanBimRange],
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    threads: usize,
    logic_mem_budget_bytes: u64,
    ld_tag_r2: f64,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
    mut on_batch: Option<&mut Max6WindowFusedBatchCallback<'_>>,
    mut on_progress: Option<&mut Max6WindowStreamProgressCallback<'_>>,
) -> Result<Max6WindowStreamLdPrep, String> {
    if !matches!(unit_kind, "gene" | "geneset") {
        return Err(format!(
            "unsupported Max6 interval stream unit kind '{unit_kind}'"
        ));
    }
    if groups.len() != group_names.len() {
        return Err(format!(
            "Max6 interval stream received {} groups and {} labels",
            groups.len(),
            group_names.len()
        ));
    }
    if groups.iter().any(Vec::is_empty) {
        return Err("Max6 interval stream requires at least one interval per unit".to_string());
    }
    if unit_kind == "gene" && groups.iter().any(|group| group.len() != 1) {
        return Err("fused Max6 gene scan requires one interval per gene".to_string());
    }
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
            "Max6 gene stream requires matching nonempty sample IDs/indices; got {}/{}",
            sample_ids.len(),
            sample_indices.len()
        ));
    }
    validate_max6_ld_tag_r2(ld_tag_r2)?;

    let intervals = groups
        .iter()
        .zip(group_names.iter())
        .enumerate()
        .flat_map(|(unit_index, (group, label))| {
            group
                .iter()
                .enumerate()
                .map(move |(span_index, (chrom, start, end))| Max6GeneInterval {
                    unit_index,
                    span_index,
                    label: label.clone(),
                    chrom: chrom.clone(),
                    start: *start,
                    end: *end,
                })
        })
        .collect::<Vec<_>>();
    let mut sweep = Max6GeneIntervalSweep::<Arc<Max6StreamWindowRow>>::new(intervals)?;
    let gene_ranges = groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|(chrom, bp_start, bp_end)| GarfieldScanBimRange {
                    chrom: chrom.clone(),
                    bp_start: *bp_start,
                    bp_end: *bp_end,
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();
    let anchor_ready = Arc::new(Mutex::new(
        gene_ranges
            .iter()
            .map(|range| chrom_base_view(range.chrom.as_str()).to_ascii_lowercase())
            .collect::<HashSet<_>>(),
    ));
    let tail_scope_complete = Arc::new(Mutex::new(HashSet::<String>::new()));
    let mut source = Max6WindowRangeSource::new(
        prefix,
        gene_ranges.as_slice(),
        0,
        true,
        anchor_ready,
        tail_scope_complete,
    )?;
    let source_marker_count = source.n_source_rows;
    if let Some(callback) = on_progress.as_deref_mut() {
        callback(0, source_marker_count)?;
    }
    let n_samples_total = source.bed.n_samples();
    if sample_indices.iter().any(|&index| index >= n_samples_total) {
        return Err("selected sample index out of range for Max6 gene stream".to_string());
    }
    let n_samples = sample_ids.len();
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
        let chunk_rows = (budget / 4 / estimated_row_bytes.max(1))
            .max(1)
            .min(default_chunk_rows);
        let task_limit = (budget / 4).min(64 * 1024 * 1024).max(1);
        (chunk_rows, task_limit)
    };
    source.max_chunk_rows = chunk_rows;
    let n_samples_source = n_samples_total;
    let sample_identity =
        n_samples == n_samples_source && sample_indices_are_identity(sample_indices);
    let sample_plan = build_logic_sample_plan(sample_indices);
    let prep_threads = effective_threads_local(threads).max(1);
    let pool = if prep_threads > 1 {
        crate::stats_common::get_cached_pool(prep_threads).map_err(|error| error.to_string())?
    } else {
        None
    };
    let support_cache_required = ld_tag_r2 < 1.0;
    let support_block_words = garfield_ld_block_words();
    let support_block_count = row_words.div_ceil(support_block_words);
    let train_indices = (0..n_samples).collect::<Vec<_>>();
    let worker_count = pool
        .as_ref()
        .map_or(1, |pool| pool.current_num_threads())
        .max(1);
    let task_batch_limit = worker_count.saturating_mul(8).max(1);
    let mut tasks = Vec::<Max6StreamWindowTask>::new();
    let mut task_batch_bytes = 0usize;
    let mut max_task_batch_bytes = 0usize;
    let mut pair_occurrence_count = 0usize;
    let mut any_eligible_unit = false;
    let mut qc_row_count = 0usize;
    let mut distinct_gene_count = 0usize;
    let mut fail_missing = 0usize;
    let mut fail_het = 0usize;
    let mut fail_no_non_missing = 0usize;
    let mut fail_maf = 0usize;
    let mut fail_non_simple_snp = 0usize;
    let mut het_sites = 0usize;
    let mut raw_missing_sites = 0usize;
    let mut next_qc_index = 0usize;
    let mut unsupported_order = false;
    let mut geneset_accumulator = if unit_kind == "geneset" {
        Some(Max6GeneSetRowAccumulator::new(groups, group_names)?)
    } else {
        None
    };
    let mut queue_completions =
        |completions: Vec<Max6GeneIntervalCompletion<Arc<Max6StreamWindowRow>>>| {
            for completion in completions {
                let task = if unit_kind == "geneset" {
                    let accumulator = geneset_accumulator
                        .as_mut()
                        .expect("geneset mode has a row accumulator");
                    let spans = accumulator
                        .spans
                        .get(completion.unit_index)
                        .ok_or_else(|| {
                            format!(
                                "Max6 geneset unit index {} is out of range",
                                completion.unit_index
                            )
                        })?
                        .clone();
                    let rows = prune_max6_geneset_span_rows(
                        completion.rows,
                        spans.as_slice(),
                        completion.span_index,
                        row_words,
                        n_samples,
                        ld_tag_r2,
                    )?;
                    accumulator.push(completion.unit_index, completion.span_index, rows)?
                } else {
                    let span = GarfieldUnitSpan {
                        chrom: completion.chrom,
                        bp_start: completion.start,
                        bp_end: completion.end,
                    };
                    Some(max6_interval_unit_task(
                        completion.unit_index,
                        completion.label,
                        unit_kind,
                        vec![span],
                        completion.rows,
                        false,
                    ))
                };
                let Some(task) = task else {
                    continue;
                };
                if task.shared_gene_rows.as_ref().is_some_and(Vec::is_empty) {
                    continue;
                }
                distinct_gene_count = distinct_gene_count.saturating_add(1);
                task_batch_bytes = task_batch_bytes
                    .saturating_add(max6_stream_window_task_materialized_bytes(&task, row_words));
                tasks.push(task);
                if tasks.len() >= task_batch_limit || task_batch_bytes >= task_batch_byte_limit {
                    flush_max6_gene_stream_tasks(
                        &mut tasks,
                        train_indices.as_slice(),
                        row_words,
                        n_samples,
                        ld_tag_r2,
                        pool.as_deref(),
                        &mut pair_occurrence_count,
                        &mut any_eligible_unit,
                        &mut max_task_batch_bytes,
                        &mut task_batch_bytes,
                        &mut on_batch,
                    )?;
                }
            }
            Ok::<(), String>(())
        };

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
                .ok_or_else(|| "Max6 gene logic chunk size overflow".to_string())?;
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
                let site = &chunk.sites[offset];
                let completed = match sweep.advance_to(site.chrom.as_str(), site.pos) {
                    Ok(completed) => completed,
                    Err(error) => {
                        unsupported_order = error.contains(MAX6_GENE_STREAM_UNSUPPORTED_ORDER);
                        return Err(error);
                    }
                };
                queue_completions(completed)?;
                if qc.keep {
                    qc_row_count = qc_row_count.saturating_add(1);
                    if sweep.has_active_intervals() {
                        let source_words =
                            &chunk.logic_bits[offset * row_words..(offset + 1) * row_words];
                        let words = source_words.to_vec();
                        let (support, block_support) = if support_cache_required {
                            let mut blocks = vec![0u16; support_block_count];
                            let support = fill_ld_block_supports(
                                words.as_slice(),
                                support_block_words,
                                blocks.as_mut_slice(),
                            );
                            (
                                Some(u32::try_from(support).unwrap_or(u32::MAX)),
                                Some(blocks),
                            )
                        } else {
                            (None, None)
                        };
                        sweep.push_current_row(Arc::new(Max6StreamWindowRow {
                            qc_index: next_qc_index,
                            source_row,
                            site: site.clone(),
                            row_flip: qc.row_flip,
                            pos: site.pos,
                            logic_words: Some(words),
                            support,
                            block_support,
                        }));
                    }
                    next_qc_index = next_qc_index.saturating_add(1);
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
                callback(done.min(progress_total), progress_total)?;
            }
            Ok(())
        },
    )
    .map_err(|error| {
        if unsupported_order {
            MAX6_GENE_STREAM_UNSUPPORTED_ORDER.to_string()
        } else {
            error
        }
    })?;

    queue_completions(sweep.finish()?)?;
    drop(queue_completions);
    flush_max6_gene_stream_tasks(
        &mut tasks,
        train_indices.as_slice(),
        row_words,
        n_samples,
        ld_tag_r2,
        pool.as_deref(),
        &mut pair_occurrence_count,
        &mut any_eligible_unit,
        &mut max_task_batch_bytes,
        &mut task_batch_bytes,
        &mut on_batch,
    )?;
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
    if distinct_gene_count == 0 {
        return Err("no scan units were built from the globally QC-passing markers".to_string());
    }
    if !any_eligible_unit {
        return Err(format!(
            "Max6 scan found no eligible LD-tag {unit_kind} with at least two markers"
        ));
    }
    Ok(Max6WindowStreamLdPrep {
        global_site_keep: Vec::new(),
        units: Vec::new(),
        screen_units: Vec::new(),
        original_unit_indices: Vec::new(),
        active_source_rows: Vec::new(),
        source_marker_count,
        window_pair_count: pair_occurrence_count,
        qc_row_count,
        distinct_window_count: distinct_gene_count,
        selected_window_count: distinct_gene_count,
        max_active_rows: 0,
        max_task_batch_bytes,
    })
}
