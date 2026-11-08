use super::*;

#[inline]
pub(super) fn normalize_chrom(chrom: &str) -> String {
    let s = chrom.trim();
    if s.len() > 2 && (s.ends_with("_1") || s.ends_with("_2")) {
        return s[..s.len() - 2].to_string();
    }
    if s.len() > 1 && (s.ends_with('-') || s.ends_with('+')) {
        return s[..s.len() - 1].to_string();
    }
    s.to_string()
}

pub(super) trait GarfieldChromPosSite {
    fn garfield_chrom(&self) -> &str;
    fn garfield_pos(&self) -> i32;
    fn garfield_snp_name(&self) -> &str;
}

pub(super) trait GarfieldDisplaySite: GarfieldChromPosSite {
    fn garfield_model_label(&self) -> &'static str;
    fn garfield_ref_allele(&self) -> &str;
    fn garfield_alt_allele(&self) -> &str;
}

#[inline]
pub(super) fn chrom_base_view(chrom: &str) -> &str {
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
pub(super) fn same_normalized_chrom(a: &str, b: &str) -> bool {
    chrom_base_view(a).eq_ignore_ascii_case(chrom_base_view(b))
}

#[inline]
pub(super) fn normalize_interval_bounds(start: i32, end: i32) -> (i32, i32) {
    if start <= end {
        (start, end)
    } else {
        (end, start)
    }
}

pub(super) fn parse_scan_bimrange_item(text: &str) -> Result<GarfieldScanBimRange, String> {
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

pub(super) fn parse_scan_bimranges(items: &[String]) -> Result<Vec<GarfieldScanBimRange>, String> {
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
pub(super) fn spans_overlap_bp(a_start: i32, a_end: i32, b_start: i32, b_end: i32) -> bool {
    let (a_lo, a_hi) = normalize_interval_bounds(a_start, a_end);
    let (b_lo, b_hi) = normalize_interval_bounds(b_start, b_end);
    a_lo <= b_hi && b_lo <= a_hi
}

pub(super) fn unit_overlaps_scan_bimranges(
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
pub(super) fn chrom_code_for_chunk_id(chrom: &str, fallback_seq_code: u32) -> u32 {
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

pub(super) fn build_chrom_index<S: GarfieldChromPosSite>(
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

pub(super) fn interval_indices(
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

pub(super) fn build_feature_group_ids<S: GarfieldChromPosSite>(
    sites: &[S],
    n_rows: usize,
) -> Vec<usize> {
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
pub(super) fn build_logic_mode_group_ids(n_sites: usize, mode: GarfieldBinMode) -> Vec<usize> {
    let _ = mode;
    let row_mul = 1usize;
    let mut out = Vec::<usize>::with_capacity(n_sites.saturating_mul(row_mul));
    for site_idx in 0..n_sites {
        for _ in 0..row_mul {
            out.push(site_idx);
        }
    }
    out
}

#[inline]
pub(super) fn pos_to_physical_window_index(pos: i32, chunk_bp: u64) -> u32 {
    if chunk_bp == 0 {
        return 0;
    }
    let pos_u64 = u64::try_from(pos.max(0)).unwrap_or(0);
    u32::try_from(pos_u64 / chunk_bp).unwrap_or(u32::MAX)
}

#[inline]
pub(super) fn physical_window_bounds(window_index: u32, chunk_bp: u64) -> (i32, i32) {
    let start = u64::from(window_index).saturating_mul(chunk_bp);
    let end = start.saturating_add(chunk_bp);
    (
        i32::try_from(start).unwrap_or(i32::MAX),
        i32::try_from(end).unwrap_or(i32::MAX),
    )
}

pub(super) fn build_valid_null_chunks<S: GarfieldChromPosSite>(
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

pub(super) fn allocate_stratified_chunk_quotas(
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

pub(super) fn sample_null_chunks_stratified<S: GarfieldChromPosSite>(
    sites: &[S],
    extension: usize,
    target_chunks: usize,
    min_snp_count: usize,
    seed: u64,
) -> Result<(Vec<GarfieldNullChunk>, usize), String> {
    let chunk_bp = extension.max(1).saturating_mul(2);
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

pub(super) fn allocate_weighted_quotas(counts: &[usize], target_total: usize) -> Vec<usize> {
    let mut quotas = vec![0usize; counts.len()];
    if target_total == 0 || counts.is_empty() {
        return quotas;
    }
    let active = counts
        .iter()
        .enumerate()
        .filter_map(|(idx, &count)| (count > 0).then_some(idx))
        .collect::<Vec<_>>();
    if active.is_empty() {
        return quotas;
    }

    let mut remaining = target_total;
    if target_total >= active.len() {
        for &idx in active.iter() {
            quotas[idx] = 1;
        }
        remaining = remaining.saturating_sub(active.len());
    }
    if remaining == 0 {
        return quotas;
    }

    let total_weight = active
        .iter()
        .map(|&idx| counts[idx] as u128)
        .sum::<u128>()
        .max(1);
    let remaining_u128 = remaining as u128;
    let mut assigned = 0usize;
    let mut remainders = Vec::<(u128, usize)>::with_capacity(active.len());
    for &idx in active.iter() {
        let weight = (counts[idx] as u128).saturating_mul(remaining_u128);
        let floor_q = usize::try_from(weight / total_weight).unwrap_or(usize::MAX);
        quotas[idx] = quotas[idx].saturating_add(floor_q);
        assigned = assigned.saturating_add(floor_q);
        remainders.push((weight % total_weight, idx));
    }
    let mut remain = remaining.saturating_sub(assigned);
    remainders.sort_by(|a, b| b.cmp(a));
    for &(_, idx) in remainders.iter() {
        if remain == 0 {
            break;
        }
        quotas[idx] = quotas[idx].saturating_add(1);
        remain -= 1;
    }
    quotas
}

#[inline]
pub(super) fn clamp_i64_to_i32(v: i64) -> i32 {
    if v > i64::from(i32::MAX) {
        i32::MAX
    } else if v < i64::from(i32::MIN) {
        i32::MIN
    } else {
        v as i32
    }
}

#[inline]
pub(super) fn fixed_atom_window_from_interval(
    chrom: &str,
    start: i32,
    end: i32,
    extension: usize,
) -> (String, i32, i32) {
    let chrom_norm = normalize_chrom(chrom);
    let (bp_start, bp_end) = normalize_interval_bounds(start, end);
    let ext = i64::try_from(extension.max(1)).unwrap_or(i64::MAX);
    let center = (i64::from(bp_start) + i64::from(bp_end)) / 2;
    (
        chrom_norm,
        clamp_i64_to_i32(center.saturating_sub(ext)),
        clamp_i64_to_i32(center.saturating_add(ext)),
    )
}

pub(super) fn build_logic_unit_span_defs_from_groups(
    groups: &[Vec<(String, i32, i32)>],
    group_names: Option<&[String]>,
) -> Vec<GarfieldLogicUnit> {
    let mut out = Vec::<GarfieldLogicUnit>::with_capacity(groups.len());
    for (gi, group) in groups.iter().enumerate() {
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
        if spans.is_empty() {
            continue;
        }
        let label = group_names
            .and_then(|v| v.get(gi).cloned())
            .unwrap_or_else(|| format!("group_{}", gi + 1));
        out.push(GarfieldLogicUnit {
            label,
            indices: Vec::new(),
            spans,
        });
    }
    out
}

pub(super) fn build_synthetic_geneset_null_group_defs(
    scan_units: &[GarfieldLogicUnit],
    null_groups: Option<&[Vec<(String, i32, i32)>]>,
    fallback_groups: Option<&[Vec<(String, i32, i32)>]>,
    target_units: usize,
    extension: usize,
    seed: u64,
) -> (Vec<Vec<(String, i32, i32)>>, usize) {
    if scan_units.is_empty() || target_units == 0 {
        return (Vec::new(), 0);
    }
    let source_groups = null_groups
        .filter(|groups| !groups.is_empty())
        .or_else(|| fallback_groups.filter(|groups| !groups.is_empty()));
    let Some(source_groups) = source_groups else {
        return (Vec::new(), 0);
    };

    let mut singleton_groups = Vec::<Vec<(String, i32, i32)>>::new();
    let mut seen_spans = HashSet::<(String, i32, i32)>::new();
    for group in source_groups.iter() {
        for (chrom, start, end) in group.iter() {
            let key = fixed_atom_window_from_interval(chrom.as_str(), *start, *end, extension);
            if seen_spans.insert(key.clone()) {
                singleton_groups.push(vec![key]);
            }
        }
    }
    let atom_pool_size = singleton_groups.len();
    if atom_pool_size == 0 {
        return (Vec::new(), 0);
    }

    let mut span_count_hist = BTreeMap::<usize, usize>::new();
    for unit in scan_units.iter() {
        let span_count = unit.spans.len().max(1);
        *span_count_hist.entry(span_count).or_insert(0usize) += 1;
    }
    if span_count_hist.is_empty() {
        return (Vec::new(), atom_pool_size);
    }

    let span_counts = span_count_hist.keys().copied().collect::<Vec<_>>();
    let span_weights = span_counts
        .iter()
        .map(|span_count| span_count_hist.get(span_count).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    let quotas = allocate_weighted_quotas(span_weights.as_slice(), target_units);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x6A09_E667_F3BC_C909);
    let mut out = Vec::<Vec<(String, i32, i32)>>::with_capacity(target_units);

    for (bucket_idx, &span_count) in span_counts.iter().enumerate() {
        let quota = quotas.get(bucket_idx).copied().unwrap_or(0);
        if quota == 0 || span_count == 0 || atom_pool_size < span_count {
            continue;
        }
        let mut seen_combos = HashSet::<Vec<usize>>::new();
        let unique_attempt_limit = quota.saturating_mul(64).max(256);
        let mut unique_attempts = 0usize;
        let mut built = 0usize;
        while built < quota {
            let mut chosen =
                sample_indices_without_replacement(&mut rng, atom_pool_size, span_count).into_vec();
            chosen.sort_unstable();
            if unique_attempts < unique_attempt_limit {
                unique_attempts += 1;
                if !seen_combos.insert(chosen.clone()) {
                    continue;
                }
            }
            let mut merged_group = Vec::<(String, i32, i32)>::with_capacity(span_count);
            for atom_idx in chosen.into_iter() {
                merged_group.extend(singleton_groups[atom_idx].iter().cloned());
            }
            if merged_group.is_empty() {
                continue;
            }
            out.push(merged_group);
            built += 1;
        }
    }
    (out, atom_pool_size)
}
