use super::*;

#[inline]
pub(super) fn unit_span_contains_site(span: &GarfieldUnitSpan, chrom: &str, pos: i32) -> bool {
    same_normalized_chrom(&span.chrom, chrom)
        && pos >= span.bp_start.min(span.bp_end)
        && pos <= span.bp_start.max(span.bp_end)
}

pub(super) fn nearest_unit_span_index(
    unit: &GarfieldLogicUnit,
    chrom: &str,
    pos: i32,
) -> Option<usize> {
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
        // When a site lies in an exactly symmetric overlap, prefer the later
        // (right-most) span.  This keeps overlap assignment deterministic and
        // avoids always starving the later geneset member.
        let tie_break = usize::MAX.saturating_sub(idx);
        let cand = (dist, span_len, tie_break);
        if best.map(|cur| cand < cur).unwrap_or(true) {
            best = Some(cand);
        }
    }
    best.map(|(_, _, tie_break)| usize::MAX.saturating_sub(tie_break))
}

pub(super) fn build_unit_window_group_ids<S: GarfieldChromPosSite>(
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
pub(super) fn geneset_min_keep_k(unit_kind_lc: &str, unit: &GarfieldLogicUnit) -> usize {
    if unit_kind_lc == "geneset" {
        unit.spans.len().max(1)
    } else {
        1
    }
}

pub(super) fn geneset_priority_order_from_scores(scores: &[f64]) -> Vec<usize> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        let sa = if scores[a].is_finite() {
            scores[a]
        } else {
            f64::NEG_INFINITY
        };
        let sb = if scores[b].is_finite() {
            scores[b]
        } else {
            f64::NEG_INFINITY
        };
        sb.total_cmp(&sa).then_with(|| a.cmp(&b))
    });
    order
}

pub(super) fn geneset_priority_order_from_scores_subset(
    scores: &[f64],
    local_rows: &[usize],
) -> Vec<usize> {
    let mut order = local_rows.to_vec();
    order.sort_by(|&a, &b| {
        let sa = if scores
            .get(a)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
            .is_finite()
        {
            scores[a]
        } else {
            f64::NEG_INFINITY
        };
        let sb = if scores
            .get(b)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
            .is_finite()
        {
            scores[b]
        } else {
            f64::NEG_INFINITY
        };
        sb.total_cmp(&sa).then_with(|| a.cmp(&b))
    });
    order
}

pub(super) fn take_prefix_per_group_rows(
    group_rows: &[Vec<usize>],
    per_group_limit: usize,
) -> Vec<usize> {
    if per_group_limit == 0 || group_rows.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::<usize>::new();
    for rows in group_rows.iter() {
        out.extend(rows.iter().take(per_group_limit).copied());
    }
    out
}

#[derive(Debug)]
pub(super) struct GarfieldLdSupportCache {
    pub(super) n_rows: usize,
    pub(super) row_words: usize,
    pub(super) block_words: usize,
    pub(super) block_count: usize,
    pub(super) support: Vec<u32>,
    pub(super) block_support: Vec<u16>,
    pub(super) sparse_global_to_local: Option<HashMap<usize, usize>>,
}

impl GarfieldLdSupportCache {
    #[inline]
    pub(super) fn local_index(&self, global_idx: usize) -> Option<usize> {
        if let Some(map) = self.sparse_global_to_local.as_ref() {
            map.get(&global_idx).copied()
        } else if global_idx < self.n_rows {
            Some(global_idx)
        } else {
            None
        }
    }

    #[inline]
    pub(super) fn block_support_for_local(&self, local_idx: usize) -> &[u16] {
        let start = local_idx.saturating_mul(self.block_count);
        &self.block_support[start..start + self.block_count]
    }

    #[inline]
    pub(super) fn compatible_with(&self, logic_bits: &GarfieldLogicBits) -> bool {
        self.n_rows == logic_bits.sites.len()
            && self.row_words == logic_bits.row_words
            && self.block_words == garfield_ld_block_words()
            && self.block_count == logic_bits.row_words.div_ceil(self.block_words)
    }

    #[inline]
    pub(super) fn payload_bytes(&self) -> usize {
        self.support
            .len()
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(
                self.block_support
                    .len()
                    .saturating_mul(std::mem::size_of::<u16>()),
            )
    }
}

pub(super) fn build_garfield_ld_support_cache(
    logic_bits: &GarfieldLogicBits,
    requested_global_rows: &[usize],
    sample_indices: &[usize],
) -> Result<GarfieldLdSupportCache, String> {
    if !sample_indices_are_full_identity(sample_indices, logic_bits.n_samples) {
        return Err("GARFIELD LD support cache requires full-identity sample indices".to_string());
    }
    let n_rows = logic_bits.sites.len();
    if requested_global_rows.iter().any(|&idx| idx >= n_rows) {
        return Err("GARFIELD LD support cache row index out of range".to_string());
    }
    let mut rows = requested_global_rows.to_vec();
    rows.sort_unstable();
    rows.dedup();
    if rows.is_empty() {
        return Ok(GarfieldLdSupportCache {
            n_rows,
            row_words: logic_bits.row_words,
            block_words: garfield_ld_block_words(),
            block_count: logic_bits.row_words.div_ceil(garfield_ld_block_words()),
            support: Vec::new(),
            block_support: Vec::new(),
            sparse_global_to_local: Some(HashMap::new()),
        });
    }

    let dense = rows.len() == n_rows && rows.iter().enumerate().all(|(idx, &row)| idx == row);
    let local_rows = if dense { n_rows } else { rows.len() };
    let block_words = garfield_ld_block_words();
    let block_count = logic_bits.row_words.div_ceil(block_words);
    let mut support = vec![0u32; local_rows];
    let mut block_support = vec![0u16; local_rows.saturating_mul(block_count)];
    let bits_flat = logic_bits.bits();
    let row_words = logic_bits.row_words;

    support
        .par_iter_mut()
        .zip(block_support.par_chunks_mut(block_count))
        .enumerate()
        .for_each(|(local_idx, (support_out, block_out))| {
            let global_idx = if dense { local_idx } else { rows[local_idx] };
            let row = &bits_flat[global_idx * row_words..(global_idx + 1) * row_words];
            let count = fill_ld_block_supports(row, block_words, block_out);
            *support_out = u32::try_from(count).unwrap_or(u32::MAX);
        });

    let sparse_global_to_local = if dense {
        None
    } else {
        Some(
            rows.into_iter()
                .enumerate()
                .map(|(local_idx, global_idx)| (global_idx, local_idx))
                .collect::<HashMap<usize, usize>>(),
        )
    };
    Ok(GarfieldLdSupportCache {
        n_rows,
        row_words,
        block_words,
        block_count,
        support,
        block_support,
        sparse_global_to_local,
    })
}

pub(super) fn collect_scan_ld_cache_rows(
    units: &[GarfieldLogicUnit],
    scan_unit_indices: &[usize],
    n_rows: usize,
    unit_kind_lc: &str,
) -> Vec<usize> {
    if unit_kind_lc == "window" && scan_unit_indices.len() == units.len() {
        return (0..n_rows).collect();
    }
    let mut seen = vec![false; n_rows];
    for &ui in scan_unit_indices.iter() {
        let Some(unit) = units.get(ui) else {
            continue;
        };
        for &row_idx in unit.indices.iter() {
            if row_idx < n_rows {
                seen[row_idx] = true;
            }
        }
    }
    seen.into_iter()
        .enumerate()
        .filter_map(|(row_idx, keep)| keep.then_some(row_idx))
        .collect()
}

#[inline]
pub(super) fn fill_ld_block_supports(row: &[u64], block_words: usize, out: &mut [u16]) -> usize {
    debug_assert!(block_words > 0);
    debug_assert_eq!(out.len(), row.len().div_ceil(block_words));
    let mut total = 0usize;
    for (block_idx, block) in row.chunks(block_words).enumerate() {
        let block_support = block
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        out[block_idx] = u16::try_from(block_support).unwrap_or(u16::MAX);
        total = total.saturating_add(block_support);
    }
    total
}

#[inline(always)]
pub(super) fn ld_bounds_match_blocks(
    row_i: &[u64],
    row_j: &[u64],
    support_i_blocks: &[u16],
    support_j_blocks: &[u16],
    support_i: u64,
    support_j: u64,
    bounds: GarfieldLdExactBounds,
    block_words: usize,
    backend: ReduceBackend,
) -> bool {
    match block_words {
        8 => and_popcount_bounded_blocks_with_backend::<8>(
            row_i,
            row_j,
            support_i_blocks,
            support_j_blocks,
            support_i,
            support_j,
            bounds.low_max,
            bounds.high_min,
            backend,
        ),
        _ => and_popcount_bounded_blocks_with_backend::<4>(
            row_i,
            row_j,
            support_i_blocks,
            support_j_blocks,
            support_i,
            support_j,
            bounds.low_max,
            bounds.high_min,
            backend,
        ),
    }
}

#[cfg(test)]
pub(super) fn prune_candidate_rows_by_ld_priority(
    candidate_global_rows: &[usize],
    priority_local_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
) -> Result<Vec<usize>, String> {
    prune_candidate_rows_by_ld_priority_with_cache(
        candidate_global_rows,
        priority_local_rows,
        logic_bits,
        sample_indices,
        None,
    )
}

#[cfg(test)]
pub(super) fn prune_candidate_rows_by_ld_priority_with_cache(
    candidate_global_rows: &[usize],
    priority_local_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
    support_cache: Option<&GarfieldLdSupportCache>,
) -> Result<Vec<usize>, String> {
    prune_candidate_rows_by_ld_priority_with_cache_limit_trace(
        candidate_global_rows,
        priority_local_rows,
        logic_bits,
        sample_indices,
        support_cache,
        None,
    )
    .map(|selection| selection.selected_global_rows)
}

pub(super) fn prune_candidate_rows_by_ld_priority_with_cache_limit_trace(
    candidate_global_rows: &[usize],
    priority_local_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
    support_cache: Option<&GarfieldLdSupportCache>,
    max_kept: Option<usize>,
) -> Result<GarfieldLdSelection, String> {
    if max_kept == Some(0) {
        return Ok(GarfieldLdSelection::default());
    }
    // This is score-prioritized greedy LD clumping: priority_local_rows must
    // already be ordered by descending single-site score.
    let t0 = Instant::now();
    let profile_t0 = garfield_stage_profile_start();
    let result = prune_candidate_rows_by_ld_priority_impl_trace(
        candidate_global_rows,
        priority_local_rows,
        logic_bits,
        sample_indices,
        support_cache,
        max_kept,
    );
    GARFIELD_LD_CLUMP_NS.fetch_add(elapsed_ns_saturating(t0), Ordering::Relaxed);
    garfield_stage_profile_end(profile_t0, &GARFIELD_CORR_STAGE1_POOL_LD_NS);
    result
}

pub(super) fn prune_candidate_rows_by_ld_priority_with_cache_limit_trace_and_threshold(
    candidate_global_rows: &[usize],
    priority_local_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
    support_cache: Option<&GarfieldLdSupportCache>,
    max_kept: Option<usize>,
    r2_threshold: f64,
    record_compression: bool,
) -> Result<GarfieldLdSelection, String> {
    if max_kept == Some(0) {
        return Ok(GarfieldLdSelection::default());
    }
    let t0 = Instant::now();
    let profile_t0 = garfield_stage_profile_start();
    let result = prune_candidate_rows_by_ld_priority_impl_trace_with_threshold(
        candidate_global_rows,
        priority_local_rows,
        logic_bits,
        sample_indices,
        support_cache,
        max_kept,
        r2_threshold,
        record_compression,
    );
    GARFIELD_LD_CLUMP_NS.fetch_add(elapsed_ns_saturating(t0), Ordering::Relaxed);
    garfield_stage_profile_end(profile_t0, &GARFIELD_CORR_STAGE1_POOL_LD_NS);
    result
}

pub(super) fn binary_row_support_for_samples(
    logic_bits: &GarfieldLogicBits,
    global_row: usize,
    sample_indices: &[usize],
) -> Result<usize, String> {
    if global_row >= logic_bits.sites.len() {
        return Err(format!(
            "Max6 LD tag row index {global_row} is outside {} logic rows",
            logic_bits.sites.len()
        ));
    }
    if sample_indices_are_full_identity(sample_indices, logic_bits.n_samples) {
        let row = &logic_bits.bits()
            [global_row * logic_bits.row_words..(global_row + 1) * logic_bits.row_words];
        let needed_words = words_for_samples(logic_bits.n_samples);
        let mut support = row
            .iter()
            .take(needed_words)
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        if let Some(mask) = tail_mask(logic_bits.n_samples) {
            if let Some(last) = row.get(needed_words.saturating_sub(1)) {
                support = support.saturating_sub((last & !mask).count_ones() as usize);
            }
        }
        return Ok(support);
    }
    let mut support = 0usize;
    let row_start = global_row * logic_bits.row_words;
    for &sample in sample_indices {
        if sample >= logic_bits.n_samples {
            return Err(format!(
                "Max6 LD tag sample index {sample} is outside {} samples",
                logic_bits.n_samples
            ));
        }
        let word = logic_bits.bits()[row_start + sample / 64];
        support += usize::from(((word >> (sample % 64)) & 1) != 0);
    }
    Ok(support)
}

pub(super) fn validate_max6_ld_tag_r2(value: f64) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err("ld_tag_r2 must be finite and in (0, 1]".to_string());
    }
    Ok(())
}

pub(super) fn select_max6_identical_tag_rows(
    candidates: &[usize],
    logic_bits: &GarfieldLogicBits,
    samples: &[usize],
    record_compression: bool,
) -> Result<GarfieldLdSelection, String> {
    let mut candidates = candidates.to_vec();
    candidates.sort_unstable();
    candidates.dedup();
    let full_identity = sample_indices_are_full_identity(samples, logic_bits.n_samples);
    let words = words_for_samples(samples.len());
    let mut representatives = HashMap::<Vec<u64>, usize>::new();
    let mut selected_global_rows = Vec::new();
    let mut compressed = Vec::new();
    let interrupt_interval = garfield_ld_interrupt_check_interval();
    for (offset, row_index) in candidates.into_iter().enumerate() {
        check_garfield_ld_interrupt(offset, interrupt_interval)?;
        if row_index >= logic_bits.sites.len() {
            return Err(format!("Max6 tag row {row_index} is out of range"));
        }
        let row = &logic_bits.bits()
            [row_index * logic_bits.row_words..(row_index + 1) * logic_bits.row_words];
        let mut encoded = if full_identity {
            row[..words].to_vec()
        } else {
            let mut encoded = vec![0u64; words];
            for (local, &sample) in samples.iter().enumerate() {
                if sample >= logic_bits.n_samples {
                    return Err(format!("Max6 tag sample {sample} is out of range"));
                }
                encoded[local / 64] |= ((row[sample / 64] >> (sample % 64)) & 1) << (local % 64);
            }
            encoded
        };
        if let Some(mask) = tail_mask(samples.len()) {
            if let Some(last) = encoded.last_mut() {
                *last &= mask;
            }
        }
        let support = encoded
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>();
        if support == 0 || support == samples.len() {
            continue;
        }
        if let Some(&kept) = representatives.get(&encoded) {
            if record_compression {
                compressed.push(GarfieldLdCompressionRecord {
                    kept_global_row: kept,
                    dropped_global_row: row_index,
                    r2: 1.0,
                });
            }
        } else {
            representatives.insert(encoded, row_index);
            selected_global_rows.push(row_index);
        }
    }
    Ok(GarfieldLdSelection {
        selected_global_rows,
        compressed,
    })
}

/// Select deterministic LD tags for the production Max6 pair universe.
///
/// The input is the complete marker set for one scan unit (or one group
/// inside a multi-window geneset). Exact mode keeps the lowest source index
/// for each identical encoded vector, separately per group. Approximate mode
/// uses descending minor support and the marginal-LD clumping kernel. Neither
/// mode uses phenotype scores. Complement vectors remain distinct in exact
/// mode because replacing a single endpoint can change the Dosage channel.
#[cfg(test)]
pub(super) fn select_max6_ld_tag_rows(
    candidate_global_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
    group_ids: Option<&[usize]>,
    r2_threshold: f64,
) -> Result<GarfieldLdSelection, String> {
    select_max6_ld_tag_rows_impl(
        candidate_global_rows,
        logic_bits,
        sample_indices,
        group_ids,
        r2_threshold,
        true,
        None,
    )
}

pub(super) fn select_max6_ld_tag_rows_impl(
    candidate_global_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
    group_ids: Option<&[usize]>,
    r2_threshold: f64,
    record_compression: bool,
    support_cache: Option<&GarfieldLdSupportCache>,
) -> Result<GarfieldLdSelection, String> {
    if let Some(group_ids) = group_ids {
        if group_ids.len() != candidate_global_rows.len() {
            return Err(format!(
                "Max6 LD tag group length {} != candidate row length {}",
                group_ids.len(),
                candidate_global_rows.len()
            ));
        }
    }
    if candidate_global_rows.is_empty() {
        return Ok(GarfieldLdSelection::default());
    }
    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    if let Some(group_ids) = group_ids {
        for (local_idx, &group_id) in group_ids.iter().enumerate() {
            grouped.entry(group_id).or_default().push(local_idx);
        }
    } else {
        grouped.insert(0, (0..candidate_global_rows.len()).collect());
    }

    let mut selected_global_rows = Vec::<usize>::new();
    let mut compressed = Vec::<GarfieldLdCompressionRecord>::new();
    let support_cache = support_cache.filter(|cache| {
        cache.compatible_with(logic_bits)
            && candidate_global_rows
                .iter()
                .all(|&global_idx| cache.local_index(global_idx).is_some())
    });
    for local_rows in grouped.values() {
        let group_candidates = local_rows
            .iter()
            .map(|&local_idx| candidate_global_rows[local_idx])
            .collect::<Vec<_>>();
        if r2_threshold == 1.0 {
            let selection = select_max6_identical_tag_rows(
                &group_candidates,
                logic_bits,
                sample_indices,
                record_compression,
            )?;
            selected_global_rows.extend(selection.selected_global_rows);
            compressed.extend(selection.compressed);
            continue;
        }
        let mut priority = (0..group_candidates.len()).collect::<Vec<_>>();
        let mut support = Vec::<usize>::with_capacity(group_candidates.len());
        for &global_row in group_candidates.iter() {
            let row_support = if let Some(cache) = support_cache {
                let cache_idx = cache
                    .local_index(global_row)
                    .expect("validated LD support cache contains candidate row");
                cache.support[cache_idx] as usize
            } else {
                binary_row_support_for_samples(logic_bits, global_row, sample_indices)?
            };
            support.push(row_support);
        }
        priority.sort_by(|&left, &right| {
            let right_minor =
                support[right].min(sample_indices.len().saturating_sub(support[right]));
            let left_minor = support[left].min(sample_indices.len().saturating_sub(support[left]));
            right_minor
                .cmp(&left_minor)
                .then_with(|| group_candidates[left].cmp(&group_candidates[right]))
        });
        let selection = prune_candidate_rows_by_ld_priority_with_cache_limit_trace_and_threshold(
            group_candidates.as_slice(),
            priority.as_slice(),
            logic_bits,
            sample_indices,
            support_cache,
            None,
            r2_threshold,
            record_compression,
        )?;
        selected_global_rows.extend(selection.selected_global_rows);
        if record_compression {
            compressed.extend(selection.compressed);
        }
    }
    selected_global_rows.sort_unstable();
    selected_global_rows.dedup();
    Ok(GarfieldLdSelection {
        selected_global_rows,
        compressed,
    })
}

pub(super) fn prune_candidate_rows_by_ld_priority_impl_trace(
    candidate_global_rows: &[usize],
    priority_local_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
    support_cache: Option<&GarfieldLdSupportCache>,
    max_kept: Option<usize>,
) -> Result<GarfieldLdSelection, String> {
    prune_candidate_rows_by_ld_priority_impl_trace_with_threshold(
        candidate_global_rows,
        priority_local_rows,
        logic_bits,
        sample_indices,
        support_cache,
        max_kept,
        garfield_ld_clump_r2(),
        true,
    )
}

pub(super) fn prune_candidate_rows_by_ld_priority_impl_trace_with_threshold(
    candidate_global_rows: &[usize],
    priority_local_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
    support_cache: Option<&GarfieldLdSupportCache>,
    max_kept: Option<usize>,
    r2_threshold: f64,
    record_compression: bool,
) -> Result<GarfieldLdSelection, String> {
    let max_kept = max_kept.unwrap_or(usize::MAX);
    if candidate_global_rows.len() <= 1
        || priority_local_rows.is_empty()
        || !(r2_threshold.is_finite() && r2_threshold > 0.0 && r2_threshold <= 1.0)
    {
        return Ok(GarfieldLdSelection {
            selected_global_rows: priority_local_rows
                .iter()
                .filter_map(|&local_idx| candidate_global_rows.get(local_idx).copied())
                .take(max_kept)
                .collect::<Vec<_>>(),
            compressed: Vec::new(),
        });
    }
    let n_samples = sample_indices.len();
    if n_samples <= 1 {
        return Ok(GarfieldLdSelection {
            selected_global_rows: priority_local_rows
                .iter()
                .filter_map(|&local_idx| candidate_global_rows.get(local_idx).copied())
                .take(max_kept)
                .collect::<Vec<_>>(),
            compressed: Vec::new(),
        });
    }

    // A support cache can only be constructed from full-identity sample rows.
    // Reuse that invariant here instead of rescanning the sample-index vector
    // once per gene/window.
    let use_full_identity = support_cache.is_some()
        || sample_indices_are_full_identity(sample_indices, logic_bits.n_samples);
    let support_cache = support_cache.filter(|cache| {
        use_full_identity
            && cache.compatible_with(logic_bits)
            && candidate_global_rows
                .iter()
                .all(|&global_idx| cache.local_index(global_idx).is_some())
    });
    let mut packed_rows = Vec::<u64>::new();
    let row_words = if use_full_identity {
        logic_bits.row_words
    } else {
        let (packed_rows_out, row_words_out) = packed_rows_subset_from_full_bits(
            logic_bits.bits(),
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
            &logic_bits.bits()[global_idx * row_words..(global_idx + 1) * row_words]
        } else {
            &packed_rows[local_idx * row_words..(local_idx + 1) * row_words]
        }
    };

    let support_bounds_cache = ld_support_cache(n_samples, r2_threshold);
    let support_conflict_buckets = support_bounds_cache.conflict_supports.as_slice();
    let exact_bounds = support_bounds_cache.exact_bounds.as_slice();
    let ld_backend = resolve_reduce_backend();
    let block_words = garfield_ld_block_words();
    let block_count = row_words.div_ceil(block_words);
    let mut block_support = vec![0u16; candidate_global_rows.len() * block_count];
    let mut support = vec![0usize; candidate_global_rows.len()];
    let mut variable = vec![false; candidate_global_rows.len()];
    if let Some(cache) = support_cache {
        for (local_idx, &global_idx) in candidate_global_rows.iter().enumerate() {
            let cache_idx = cache
                .local_index(global_idx)
                .ok_or_else(|| "GARFIELD LD support cache lost candidate row".to_string())?;
            let block_start = local_idx * block_count;
            let cached_blocks = cache.block_support_for_local(cache_idx);
            block_support[block_start..block_start + block_count].copy_from_slice(cached_blocks);
            let cnt = cache.support[cache_idx] as usize;
            support[local_idx] = cnt;
            variable[local_idx] = binary_row_var_from_support(cnt, n_samples) > 0.0;
        }
    } else {
        for local_idx in 0..candidate_global_rows.len() {
            let block_start = local_idx * block_count;
            let cnt = fill_ld_block_supports(
                row_slice(local_idx),
                block_words,
                &mut block_support[block_start..block_start + block_count],
            );
            support[local_idx] = cnt;
            variable[local_idx] = binary_row_var_from_support(cnt, n_samples) > 0.0;
        }
    }

    let mut kept_local = Vec::<usize>::with_capacity(priority_local_rows.len());
    let mut compressed = Vec::<GarfieldLdCompressionRecord>::new();
    let mut kept_by_support = vec![Vec::<usize>::new(); n_samples + 1];
    let mut seen = vec![false; candidate_global_rows.len()];
    let mut exact_pairs = 0u64;
    let interrupt_interval = garfield_ld_interrupt_check_interval();
    for (candidate_offset, &local_idx) in priority_local_rows.iter().enumerate() {
        check_garfield_ld_interrupt(candidate_offset, interrupt_interval)?;
        if local_idx >= candidate_global_rows.len() || seen[local_idx] || !variable[local_idx] {
            continue;
        }
        seen[local_idx] = true;
        let row_i = row_slice(local_idx);
        let support_i = support[local_idx];
        let exact_bounds_i = &exact_bounds[support_i];
        let mut has_conflict = false;
        for &support_j in support_conflict_buckets[support_i].iter() {
            let bounds = exact_bounds_i[support_j];
            if bounds.low_max == GARFIELD_LD_NO_BOUND && bounds.high_min == GARFIELD_LD_NO_BOUND {
                continue;
            }
            for &kept_idx in kept_by_support[support_j].iter() {
                exact_pairs = exact_pairs.saturating_add(1);
                let row_j = row_slice(kept_idx);
                let block_start_i = local_idx * block_count;
                let block_start_j = kept_idx * block_count;
                if ld_bounds_match_blocks(
                    row_i,
                    row_j,
                    &block_support[block_start_i..block_start_i + block_count],
                    &block_support[block_start_j..block_start_j + block_count],
                    support_i as u64,
                    support[kept_idx] as u64,
                    bounds,
                    block_words,
                    ld_backend,
                ) {
                    has_conflict = true;
                    if record_compression {
                        let both = and_popcount(row_i, row_j) as usize;
                        let n = n_samples as f64;
                        let pi = support_i as f64 / n;
                        let pj = support[kept_idx] as f64 / n;
                        let covariance = both as f64 / n - pi * pj;
                        let variance_product = pi * (1.0 - pi) * pj * (1.0 - pj);
                        let r2 = if variance_product > 0.0 {
                            (covariance * covariance / variance_product).clamp(0.0, 1.0)
                        } else {
                            1.0
                        };
                        compressed.push(GarfieldLdCompressionRecord {
                            kept_global_row: candidate_global_rows[kept_idx],
                            dropped_global_row: candidate_global_rows[local_idx],
                            r2,
                        });
                    }
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
            if kept_local.len() >= max_kept {
                break;
            }
        }
    }
    check_garfield_ld_interrupt(priority_local_rows.len(), interrupt_interval)?;
    let selected_global_rows = kept_local
        .into_iter()
        .map(|local_idx| candidate_global_rows[local_idx])
        .collect::<Vec<_>>();
    GARFIELD_LD_CLUMP_EXACT_PAIRS.fetch_add(exact_pairs, Ordering::Relaxed);
    GARFIELD_LD_CLUMP_ROWS_TOTAL.fetch_add(
        u64::try_from(candidate_global_rows.len()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    GARFIELD_LD_CLUMP_ROWS_KEPT.fetch_add(
        u64::try_from(selected_global_rows.len()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    garfield_ld_unit_stats_record(
        candidate_global_rows.len(),
        selected_global_rows.len(),
        exact_pairs,
    );
    Ok(GarfieldLdSelection {
        selected_global_rows,
        compressed,
    })
}

pub(super) fn select_geneset_window_candidate_pool_rows_with_cache_trace(
    unit: &GarfieldLogicUnit,
    candidate_global_rows: &[usize],
    scores: &[f64],
    per_window_keep_k: usize,
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
    prune_with_ld: bool,
    support_cache: Option<&GarfieldLdSupportCache>,
) -> Result<GarfieldLdSelection, String> {
    if candidate_global_rows.is_empty() || per_window_keep_k == 0 || unit.spans.len() <= 1 {
        return Ok(GarfieldLdSelection::default());
    }
    let Some(group_ids) = build_unit_window_group_ids(
        unit,
        candidate_global_rows,
        logic_bits.sites.as_slice(),
        "geneset",
    ) else {
        return Ok(GarfieldLdSelection::default());
    };
    let span_count = unit.spans.len();
    let mut per_group_local = vec![Vec::<usize>::new(); span_count];
    let mut overflow_local = Vec::<usize>::new();
    for (local_idx, &gid) in group_ids.iter().enumerate() {
        if gid < span_count {
            per_group_local[gid].push(local_idx);
        } else {
            overflow_local.push(local_idx);
        }
    }

    let mut per_group_global = Vec::<Vec<usize>>::with_capacity(span_count);
    let mut compressed = Vec::<GarfieldLdCompressionRecord>::new();
    for local_rows in per_group_local.iter() {
        if local_rows.is_empty() {
            per_group_global.push(Vec::new());
            continue;
        }
        let priority_local =
            geneset_priority_order_from_scores_subset(scores, local_rows.as_slice());
        let pruned_global = if prune_with_ld {
            let selection = prune_candidate_rows_by_ld_priority_with_cache_limit_trace(
                candidate_global_rows,
                priority_local.as_slice(),
                logic_bits,
                sample_indices,
                support_cache,
                Some(per_window_keep_k),
            )?;
            compressed.extend(selection.compressed);
            selection.selected_global_rows
        } else {
            priority_local
                .iter()
                .filter_map(|&local_idx| candidate_global_rows.get(local_idx).copied())
                .collect::<Vec<_>>()
        };
        per_group_global.push(pruned_global);
    }

    let mut selected = take_prefix_per_group_rows(per_group_global.as_slice(), per_window_keep_k);
    if selected.is_empty() && !overflow_local.is_empty() {
        let overflow_priority =
            geneset_priority_order_from_scores_subset(scores, overflow_local.as_slice());
        let overflow_rows = if prune_with_ld {
            let selection = prune_candidate_rows_by_ld_priority_with_cache_limit_trace(
                candidate_global_rows,
                overflow_priority.as_slice(),
                logic_bits,
                sample_indices,
                support_cache,
                Some(per_window_keep_k),
            )?;
            compressed.extend(selection.compressed);
            selection.selected_global_rows
        } else {
            overflow_priority
                .iter()
                .filter_map(|&local_idx| candidate_global_rows.get(local_idx).copied())
                .collect::<Vec<_>>()
        };
        selected.extend(overflow_rows.into_iter().take(per_window_keep_k));
    }
    Ok(GarfieldLdSelection {
        selected_global_rows: selected,
        compressed,
    })
}

#[inline]
pub(super) fn geneset_stage_group_target(
    unit_kind_lc: &str,
    unit: &GarfieldLogicUnit,
    local_groups: &[usize],
) -> Option<usize> {
    if unit_kind_lc != "geneset" || unit.spans.is_empty() {
        return None;
    }
    let span_count = unit.spans.len().max(1);
    if span_count <= 1 {
        return Some(1);
    }
    let mut covered = vec![false; span_count];
    let mut distinct = 0usize;
    for &gid in local_groups.iter() {
        if gid < span_count && !covered[gid] {
            covered[gid] = true;
            distinct = distinct.saturating_add(1);
        }
    }
    Some(distinct.max(1))
}

#[inline]
pub(super) fn null_unit_group_bin_for_group_count(unit_kind_lc: &str, group_count: usize) -> u8 {
    if !matches!(unit_kind_lc, "gene" | "geneset") {
        return 0;
    }
    rule_null_unit_group_bin(group_count.max(1))
}

#[inline]
pub(super) fn null_unit_group_count_for_penalty(
    unit_kind_lc: &str,
    unit: &GarfieldLogicUnit,
) -> usize {
    if matches!(unit_kind_lc, "gene" | "geneset") {
        unit.spans.len().max(1)
    } else {
        1
    }
}

#[inline]
pub(super) fn beam_params_for_prepared(
    prepared: &GarfieldUnitPrepared,
    beam_params: BeamSearchParams,
) -> BeamSearchParams {
    let null_complexity_bin = rule_null_context_bin(
        rule_null_complexity_bin(prepared.selected_global_rows.len()),
        prepared.null_unit_group_bin,
    );
    if let Some(required_groups) = prepared.geneset_stage_group_target {
        let effective_max_pick = if required_groups > 1 {
            required_groups.max(1)
        } else {
            beam_params.max_pick.max(1)
        };
        BeamSearchParams {
            max_pick: effective_max_pick,
            null_complexity_bin,
            group_constraint: BeamGroupConstraintMode::ExcludeUntilDistinctGroups(required_groups),
            ..beam_params
        }
    } else {
        BeamSearchParams {
            null_complexity_bin,
            ..beam_params
        }
    }
}

#[inline]
pub(super) fn apply_proposal_search_config(
    params: BeamSearchParams,
    proposal_config: ProposalConfig,
) -> BeamSearchParams {
    if matches!(proposal_config.mode, ProposalMode::Off) {
        return params;
    }
    if matches!(params.search_backend, GarfieldSearchBackend::PairTriple) {
        // PairTriple has its own exhaustive-pair/pair-seeded-triple scanner;
        // do not silently turn it back into the historical Beam by applying
        // the legacy proposal-mode width/rank overrides here.
        return BeamSearchParams {
            max_pick: params.max_pick.max(1).min(3),
            ..params
        };
    }
    let max_pick = params.max_pick.max(proposal_config.max_order).min(5);
    BeamSearchParams {
        // Proposal mode starts with an exhaustive pair layer and then lets the
        // existing Beam expander refine those seeds up to the configured
        // order.  A larger width avoids spending the pair budget on one
        // polarity of the same site family.
        max_pick,
        beam_width: params.beam_width.max(proposal_config.k2),
        exhaustive_depth: proposal_config.max_order.min(max_pick).max(2),
        rank_mode: BeamRankMode::ExhaustiveThenGain,
        ..params
    }
}

pub(super) fn effective_proposal_config(
    search_backend: GarfieldSearchBackend,
    max_pick: usize,
) -> Result<ProposalConfig, String> {
    let mut proposal_config = resolve_proposal_config()?;
    if matches!(search_backend, GarfieldSearchBackend::PairTriple)
        && matches!(proposal_config.mode, ProposalMode::Off)
    {
        proposal_config.mode = ProposalMode::CorrRf;
    }
    if matches!(search_backend, GarfieldSearchBackend::PairTriple) {
        proposal_config.max_order = if max_pick >= 3 { 3 } else { 2 };
    }
    Ok(proposal_config)
}
