use super::*;

pub(super) struct LogicSamplePlanEntry {
    pub(super) src_byte_idx: usize,
    pub(super) src_shifts: [u8; 4],
    pub(super) dst_word_idx: [usize; 4],
    pub(super) dst_bit: [u64; 4],
    pub(super) src_lane_mask: u8,
    pub(super) dst_start_sample: usize,
    pub(super) dst_contiguous: bool,
    pub(super) n_lanes: u8,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Max6SubsetByteLutEntry {
    pub(super) logic_missing: u8,
    pub(super) hom_alt: u8,
    pub(super) logic_alt_bits: u8,
    pub(super) logic_missing_bits: u8,
    pub(super) flags: u8,
}

pub(super) fn max6_subset_byte_lut() -> &'static [Max6SubsetByteLutEntry; 16 * 256] {
    static LUT: OnceLock<[Max6SubsetByteLutEntry; 16 * 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [Max6SubsetByteLutEntry::default(); 16 * 256];
        for lane_mask in 0u8..16 {
            for byte in 0u8..=u8::MAX {
                let index = lane_mask as usize * 256 + byte as usize;
                let entry = &mut lut[index];
                let mut dst_lane = 0u8;
                for src_lane in 0..4u8 {
                    if lane_mask & (1u8 << src_lane) == 0 {
                        continue;
                    }
                    let code = (byte >> (src_lane * 2)) & 0b11;
                    match code {
                        0b00 => {}
                        0b01 => {
                            entry.logic_missing += 1;
                            entry.logic_missing_bits |= 1u8 << dst_lane;
                            entry.flags |= 0b01;
                        }
                        0b10 => {
                            entry.logic_missing += 1;
                            entry.logic_missing_bits |= 1u8 << dst_lane;
                            entry.flags |= 0b10;
                        }
                        0b11 => {
                            entry.hom_alt += 1;
                            entry.logic_alt_bits |= 1u8 << dst_lane;
                        }
                        _ => unreachable!("PLINK genotype code is two bits"),
                    }
                    dst_lane += 1;
                }
            }
        }
        lut
    })
}

pub(super) fn build_logic_sample_plan(sample_indices: &[usize]) -> Vec<LogicSamplePlanEntry> {
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
    lanes.sort_unstable_by_key(|&(src_byte_idx, src_shift, _, _)| (src_byte_idx, src_shift));

    let mut plan = Vec::<LogicSamplePlanEntry>::with_capacity(lanes.len().div_ceil(4));
    let mut cur_src_byte_idx = usize::MAX;
    let mut cur = LogicSamplePlanEntry {
        src_byte_idx: 0,
        src_shifts: [0; 4],
        dst_word_idx: [0; 4],
        dst_bit: [0; 4],
        src_lane_mask: 0,
        dst_start_sample: 0,
        dst_contiguous: true,
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
                src_lane_mask: 0,
                dst_start_sample: 0,
                dst_contiguous: true,
                n_lanes: 0,
            };
        }
        let lane = cur.n_lanes as usize;
        let dst_sample_idx = dst_word_idx * 64 + dst_bit.trailing_zeros() as usize;
        if lane == 0 {
            cur.dst_start_sample = dst_sample_idx;
        } else if dst_sample_idx != cur.dst_start_sample + lane {
            cur.dst_contiguous = false;
        }
        cur.src_shifts[lane] = src_shift;
        cur.dst_word_idx[lane] = dst_word_idx;
        cur.dst_bit[lane] = dst_bit;
        cur.src_lane_mask |= 1u8 << (src_shift / 2);
        cur.n_lanes += 1;
    }
    if cur_src_byte_idx != usize::MAX {
        plan.push(cur);
    }
    plan
}

#[inline]
pub(super) fn pack_compact_logic_bits(
    words: &mut [u64],
    start_sample: usize,
    n_samples: u8,
    bits: u8,
) {
    if n_samples == 0 {
        return;
    }
    let word_idx = start_sample >> 6;
    let shift = start_sample & 63;
    words[word_idx] |= (bits as u64) << shift;
    if shift + n_samples as usize > 64 {
        words[word_idx + 1] |= (bits as u64) >> (64 - shift);
    }
}

#[inline]
pub(super) fn scan_max6_subset_row_and_pack(
    row: &[u8],
    sample_plan: &[LogicSamplePlanEntry],
    dst_words: &mut [u64],
    missing_words: &mut [u64],
) -> (usize, usize, usize, bool, bool) {
    let lut = max6_subset_byte_lut();
    let mut logic_missing = 0usize;
    let mut hom_alt = 0usize;
    let mut hom_ref = 0usize;
    let mut has_het = false;
    let mut has_raw_missing = false;
    for entry in sample_plan {
        let byte = row[entry.src_byte_idx];
        let lut_index = entry.src_lane_mask as usize * 256 + byte as usize;
        let packed = lut[lut_index];
        logic_missing += packed.logic_missing as usize;
        hom_alt += packed.hom_alt as usize;
        hom_ref += (entry.n_lanes as usize)
            .saturating_sub(packed.logic_missing as usize)
            .saturating_sub(packed.hom_alt as usize);
        has_het |= packed.flags & 0b10 != 0;
        has_raw_missing |= packed.flags & 0b01 != 0;
        if entry.dst_contiguous {
            pack_compact_logic_bits(
                dst_words,
                entry.dst_start_sample,
                entry.n_lanes,
                packed.logic_alt_bits,
            );
            pack_compact_logic_bits(
                missing_words,
                entry.dst_start_sample,
                entry.n_lanes,
                packed.logic_missing_bits,
            );
        } else {
            for lane in 0..entry.n_lanes as usize {
                let code = (byte >> entry.src_shifts[lane]) & 0b11;
                let word_idx = entry.dst_word_idx[lane];
                let bit = entry.dst_bit[lane];
                match code {
                    0b01 | 0b10 => missing_words[word_idx] |= bit,
                    0b11 => dst_words[word_idx] |= bit,
                    _ => {}
                }
            }
        }
    }
    (logic_missing, hom_alt, hom_ref, has_het, has_raw_missing)
}

#[inline]
pub(super) fn decode_logic_bin_genotype(code: u8, flip: bool) -> Option<u8> {
    match code {
        0b00 => Some(if flip { 2 } else { 0 }),
        0b10 => Some(1),
        0b11 => Some(if flip { 0 } else { 2 }),
        _ => None,
    }
}

#[inline]
#[allow(dead_code)]
pub(super) fn fill_bin_logic_row_bits(
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

/// Materialize one binary logic row in a single genotype pass.
///
/// Known 0/2 calls are written immediately. Missing calls are kept in a
/// reusable scratch row and merged only when the modal known state is 2.
/// This preserves the existing mode imputation while avoiding a second full
/// sample-plan decode for every SNP.
#[inline]
pub(super) fn fill_bin_logic_row_bits_one_pass(
    row: &[u8],
    flip: bool,
    sample_plan: &[LogicSamplePlanEntry],
    dst_words: &mut [u64],
    missing_words: &mut [u64],
) {
    debug_assert_eq!(dst_words.len(), missing_words.len());
    missing_words.fill(0);
    let mut c0 = 0usize;
    let mut c2 = 0usize;
    for entry in sample_plan.iter() {
        let byte = row[entry.src_byte_idx];
        for lane in 0..entry.n_lanes as usize {
            let code = (byte >> entry.src_shifts[lane]) & 0b11;
            let word_idx = entry.dst_word_idx[lane];
            let bit = entry.dst_bit[lane];
            match code {
                0b00 => {
                    if flip {
                        c2 += 1;
                        dst_words[word_idx] |= bit;
                    } else {
                        c0 += 1;
                    }
                }
                0b10 => {
                    missing_words[word_idx] |= bit;
                }
                0b11 => {
                    if flip {
                        c0 += 1;
                    } else {
                        c2 += 1;
                        dst_words[word_idx] |= bit;
                    }
                }
                0b01 => {
                    missing_words[word_idx] |= bit;
                }
                _ => unreachable!("PLINK genotype code is two bits"),
            }
        }
    }

    if c2 > c0 {
        for (dst, &missing) in dst_words.iter_mut().zip(missing_words.iter()) {
            *dst |= missing;
        }
    }
}

/// Materialize a binary logic row directly when selected samples are the
/// identity order. Each BED byte contributes one output nibble, avoiding the
/// per-sample gather plan and its temporary missing bit row.
#[inline]
pub(super) fn fill_bin_logic_row_bits_identity(
    row: &[u8],
    n_samples: usize,
    flip: bool,
    dst_words: &mut [u64],
) {
    let active_bytes = n_samples.div_ceil(4);
    debug_assert!(row.len() >= active_bytes);
    if active_bytes == 0 || dst_words.is_empty() {
        return;
    }

    let lut = packed_byte_lut();
    let rem_lanes = n_samples & 3;
    let mut raw_c0 = 0usize;
    let mut raw_c2 = 0usize;
    for byte_idx in 0..active_bytes {
        let active_lanes = if rem_lanes > 0 && byte_idx + 1 == active_bytes {
            rem_lanes
        } else {
            4
        };
        let mut byte = row[byte_idx];
        if active_lanes < 4 {
            byte &= (1u8 << (active_lanes * 2)) - 1u8;
        }
        let idx = byte as usize;
        let missing = lut.logic_missing[idx] as usize;
        let hom_alt = lut.hom_alt[idx] as usize;
        raw_c2 += hom_alt;
        raw_c0 += active_lanes.saturating_sub(missing).saturating_sub(hom_alt);
    }

    // A flip swaps the two homozygous states, while heterozygotes and raw
    // missing calls remain the mode-imputed missing state.
    let mode_is_one = if flip {
        raw_c0 > raw_c2
    } else {
        raw_c2 > raw_c0
    };
    for (word_idx, dst) in dst_words.iter_mut().enumerate() {
        let byte_start = word_idx.saturating_mul(16);
        let byte_end = (byte_start + 16).min(active_bytes);
        let mut word = 0u64;
        let mut missing = 0u64;
        for byte_idx in byte_start..byte_end {
            let active_lanes = if rem_lanes > 0 && byte_idx + 1 == active_bytes {
                rem_lanes
            } else {
                4
            };
            let mut byte = row[byte_idx];
            if active_lanes < 4 {
                byte &= (1u8 << (active_lanes * 2)) - 1u8;
            }
            let idx = byte as usize;
            let shift = ((byte_idx - byte_start) * 4) as u32;
            let one_bits = if flip {
                lut.logic_ref_bits[idx]
            } else {
                lut.logic_alt_bits[idx]
            };
            word |= (one_bits as u64) << shift;
            missing |= (lut.logic_missing_bits[idx] as u64) << shift;
        }
        if mode_is_one {
            word |= missing;
        }
        *dst = word;
    }

    let tail_bits = n_samples & 63;
    if tail_bits > 0 {
        if let Some(last) = dst_words.last_mut() {
            if let Some(mask) = tail_mask(tail_bits) {
                *last &= mask;
            }
        }
    }
}

#[inline]
#[allow(dead_code)]
pub(super) fn fill_bin_logic_row_bits_pure_line(
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

// Retained only for backward-compatible debugging of old fuzzy 0/1/2 BIN
// artifacts. The active BIN runtime now uses `fill_bin_logic_row_bits`.
#[inline]
#[allow(dead_code)]
pub(super) fn fill_bin_logic_row_bits_fuzzy(
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

#[allow(dead_code)]
pub(super) fn convert_prepared_bed_to_logic_bits(
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
    let _ = mode;
    let group_ids = build_logic_mode_group_ids(sites.len(), mode);
    let out_sites = build_logic_sites_from_metadata(sites, row_flip, mode);
    let n_rows = out_sites.len();
    let mut bits_flat = vec![0u64; n_rows.saturating_mul(row_words)];
    let bits_hi_flat = None;
    bits_flat
        .par_chunks_mut(row_words)
        .zip(row_flip.par_iter().copied())
        .zip(packed.par_chunks(bytes_per_snp))
        .for_each(|((dst_rows, flip), row)| {
            fill_bin_logic_row_bits(row, flip, sample_plan.as_slice(), dst_rows)
        });

    Ok(GarfieldLogicBits {
        bits_flat,
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat,
        row_words,
        sample_ids: sample_ids.to_vec(),
        sites: out_sites,
        group_ids,
        n_samples,
    })
}

#[inline]
pub(super) fn should_use_mapped_logic_bits(estimated_bytes: usize, budget_bytes: u64) -> bool {
    budget_bytes > 0 && (estimated_bytes as u64) > budget_bytes
}

pub(super) static GARFIELD_BITS_MMAP_SEQ: AtomicU64 = AtomicU64::new(0);

pub(super) fn convert_bed_prefix_to_logic_bits(
    prefix: &str,
    bytes_per_snp: usize,
    n_samples_total: usize,
    row_source_indices: &[usize],
    row_flip: &[bool],
    sites: Vec<SiteInfo>,
    prebuilt_sites: Option<Vec<GarfieldLogicSite>>,
    sample_indices: &[usize],
    sample_ids: Vec<String>,
    mode: GarfieldBinMode,
    logic_mem_budget_bytes: u64,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<GarfieldLogicBits, String> {
    let total_t0 = Instant::now();
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
    let sites_deferred = sites.is_empty() && prebuilt_sites.is_none();
    if prebuilt_sites.is_some() && !sites.is_empty() {
        return Err("prebuilt logic sites cannot be combined with raw BIM sites".to_string());
    }
    if !sites_deferred && prebuilt_sites.is_none() && row_source_indices.len() != sites.len() {
        return Err(format!(
            "row_source_indices/site metadata mismatch: rows={}, filtered sites={}",
            row_source_indices.len(),
            sites.len()
        ));
    }
    if row_flip.len() != row_source_indices.len() {
        return Err(format!(
            "row_flip/source row metadata mismatch: row_flip={}, source_rows={}",
            row_flip.len(),
            row_source_indices.len()
        ));
    }
    if row_source_indices.is_empty() {
        return Err("row_source_indices keeps zero SNPs".to_string());
    }

    let n_samples = sample_indices.len();
    let row_words = words_for_samples(n_samples);
    let sample_identity =
        sample_indices.len() == n_samples_total && sample_indices_are_identity(sample_indices);
    let sample_plan = build_logic_sample_plan(sample_indices);
    let bed_prefix = normalize_plink_prefix(prefix);
    let bed_open_t0 = Instant::now();
    let mut bed_iter =
        BedSnpIter::new_for_grm_window(bed_prefix.as_str(), garfield_logic_mmap_window_mb())?;
    let bed_open_secs = bed_open_t0.elapsed().as_secs_f64();
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
    let _ = mode;
    let row_mul = 1usize;
    let sites_t0 = Instant::now();
    let out_sites = if let Some(prebuilt) = prebuilt_sites {
        if prebuilt.len() != row_source_indices.len().saturating_mul(row_mul) {
            return Err("prebuilt logic site count does not match BED candidate rows".to_string());
        }
        prebuilt
    } else if sites.is_empty() {
        build_logic_sites_from_bim_owned_direct(
            prefix,
            bed_iter.n_snps(),
            row_source_indices,
            row_flip,
            mode,
        )?
    } else {
        build_logic_sites_from_metadata_owned_direct(sites, row_flip, mode)
    };
    let sites_secs = sites_t0.elapsed().as_secs_f64();
    let group_ids = build_logic_mode_group_ids(row_source_indices.len(), mode);
    let n_rows = out_sites.len();
    let bits_alloc_t0 = Instant::now();
    let total_words = n_rows.saturating_mul(row_mul).saturating_mul(row_words);
    let estimated_bytes = total_words.saturating_mul(std::mem::size_of::<u64>());
    let use_mmap = should_use_mapped_logic_bits(estimated_bytes, logic_mem_budget_bytes);
    let mut bits_flat = if use_mmap {
        Vec::new()
    } else {
        vec![0u64; total_words]
    };
    let mut bits_mmap_mut: Option<MmapMut> = None;
    let mut bits_mmap_path: Option<std::path::PathBuf> = None;
    if use_mmap {
        let path = std::env::temp_dir().join(format!(
            "janusx_garfield_bits_{}_{}.bin",
            std::process::id(),
            GARFIELD_BITS_MMAP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| format!("create GARFIELD mapped bitset '{}': {e}", path.display()))?;
        if let Err(e) = file.set_len(estimated_bytes as u64) {
            let _ = fs::remove_file(&path);
            return Err(format!(
                "size GARFIELD mapped bitset '{}': {e}",
                path.display()
            ));
        }
        let mmap = unsafe {
            MmapOptions::new()
                .len(estimated_bytes)
                .map_mut(&file)
                .map_err(|e| {
                    let _ = fs::remove_file(&path);
                    format!("map GARFIELD bitset '{}': {e}", path.display())
                })?
        };
        bits_mmap_mut = Some(mmap);
        bits_mmap_path = Some(path);
    }
    let bits_alloc_secs = bits_alloc_t0.elapsed().as_secs_f64();
    let bits_hi_flat = None;
    if let Some(tracker) = mem_tracker {
        tracker.sample_now();
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

    let fill_t0 = Instant::now();
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
        let bits_storage_mut: &mut [u64] = if let Some(mmap) = bits_mmap_mut.as_mut() {
            unsafe { std::slice::from_raw_parts_mut(mmap.as_mut_ptr() as *mut u64, total_words) }
        } else {
            bits_flat.as_mut_slice()
        };
        let dst_lo =
            &mut bits_storage_mut[kept_start * row_mul * row_words..kept_end * row_mul * row_words];
        if sample_identity {
            dst_lo
                .par_chunks_mut(row_words)
                .enumerate()
                .for_each(|(off, dst_rows)| {
                    let kept_idx = kept_start + off;
                    let src_row = row_source_indices[kept_idx];
                    let flip = row_flip[kept_idx];
                    let row = bed_ref
                        .packed_snp_bytes_at(src_row)
                        .expect("windowed BED logic conversion row should be mapped");
                    fill_bin_logic_row_bits_identity(&row, n_samples, flip, dst_rows);
                });
        } else {
            dst_lo.par_chunks_mut(row_words).enumerate().for_each_init(
                || vec![0u64; row_words],
                |missing_words, (off, dst_rows)| {
                    let kept_idx = kept_start + off;
                    let src_row = row_source_indices[kept_idx];
                    let flip = row_flip[kept_idx];
                    let row = bed_ref
                        .packed_snp_bytes_at(src_row)
                        .expect("windowed BED logic conversion row should be mapped");
                    fill_bin_logic_row_bits_one_pass(
                        &row,
                        flip,
                        sample_plan.as_slice(),
                        dst_rows,
                        missing_words.as_mut_slice(),
                    );
                },
            );
        }
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        kept_start = kept_end;
    }
    if env_truthy("JX_BED_LOGIC_META_TIMING") {
        eprintln!(
            "GARFIELD logic bits timing: sites={:.3}s, alloc={:.3}s, bed_open={:.3}s, fill={:.3}s, total={:.3}s, mode={:?}, source_rows={}, output_rows={}, samples={}, row_words={}, mmap_window_mb={}",
            sites_secs,
            bits_alloc_secs,
            bed_open_secs,
            fill_t0.elapsed().as_secs_f64(),
            total_t0.elapsed().as_secs_f64(),
            mode,
            row_source_indices.len(),
            n_rows,
            n_samples,
            row_words,
            garfield_logic_mmap_window_mb(),
        );
    }

    let bits_mmap = if let Some(mmap_mut) = bits_mmap_mut {
        let mmap = match mmap_mut.make_read_only() {
            Ok(mmap) => mmap,
            Err(e) => {
                if let Some(path) = bits_mmap_path.as_ref() {
                    let _ = fs::remove_file(path);
                }
                return Err(format!("finalize GARFIELD mapped bitset: {e}"));
            }
        };
        #[cfg(unix)]
        let _ = mmap.advise(Advice::Random);
        if let Some(path) = bits_mmap_path.as_ref() {
            let _ = fs::remove_file(path);
        }
        Some(Arc::new(mmap))
    } else {
        None
    };

    Ok(GarfieldLogicBits {
        bits_flat,
        bits_mmap,
        bits_mmap_words: if use_mmap { total_words } else { 0 },
        bits_mmap_owner: None,
        bits_hi_flat,
        row_words,
        sample_ids,
        sites: out_sites,
        group_ids,
        n_samples,
    })
}

pub(super) fn dense_dosage_rows_from_full_bits_range(
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

pub(super) fn dense_dosage_rows_from_full_bits(
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

/// Decode a small set of proposal rows with a bounded per-worker cache.  Scan
/// windows are generated in genomic order and overlap heavily; a thread-local
/// cache therefore reuses rows shared by neighbouring windows without a
/// cross-thread lock or unbounded resident matrix.  Cache keys include both
/// bitplane allocations and the sample-index allocation so separate scans
/// cannot silently reuse incompatible rows.
pub(super) fn dense_dosage_rows_from_full_bits_cached(
    bits_flat: &[u64],
    bits_hi_flat: Option<&[u64]>,
    row_words: usize,
    row_indices: &[usize],
    sample_indices: &[usize],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<Vec<Vec<u8>>, String> {
    if row_indices.is_empty() {
        return Ok(Vec::new());
    }
    if row_words != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for cached densifying: got {row_words}, expected {}",
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
            "row index out of range while cached densifying: {row_idx}"
        ));
    }
    if let Some(&sid) = sample_indices.iter().find(|&&sid| sid >= n_samples_all) {
        return Err(format!(
            "sample index out of range while cached densifying: {sid}"
        ));
    }
    let cache_capacity = garfield_proposal_row_cache_capacity();
    if cache_capacity == 0 {
        return dense_dosage_rows_from_full_bits(
            bits_flat,
            bits_hi_flat,
            row_words,
            row_indices,
            sample_indices,
            n_rows_all,
            n_samples_all,
        );
    }
    let key_base = (
        bits_flat.as_ptr() as usize,
        bits_flat.len(),
        bits_hi_flat.map_or(0, |bits| bits.as_ptr() as usize),
        bits_hi_flat.map_or(0, <[u64]>::len),
        sample_indices.as_ptr() as usize,
        sample_indices.len(),
    );
    let mut out = vec![None::<Vec<u8>>; row_indices.len()];
    let mut missing = Vec::<usize>::new();
    let mut missing_positions = Vec::<usize>::new();
    GARFIELD_PROPOSAL_ROW_CACHE.with(|cache_cell| {
        let cache = cache_cell.borrow();
        for (position, &row_idx) in row_indices.iter().enumerate() {
            let key = ProposalRowCacheKey {
                source_ptr: key_base.0,
                source_len: key_base.1,
                source_hi_ptr: key_base.2,
                source_hi_len: key_base.3,
                row_idx,
                sample_ptr: key_base.4,
                sample_len: key_base.5,
            };
            if let Some(row) = cache.rows.get(&key) {
                out[position] = Some(row.clone());
            } else {
                missing.push(row_idx);
                missing_positions.push(position);
            }
        }
    });
    if !missing.is_empty() {
        let decoded = dense_dosage_rows_from_full_bits(
            bits_flat,
            bits_hi_flat,
            row_words,
            missing.as_slice(),
            sample_indices,
            n_rows_all,
            n_samples_all,
        )?;
        GARFIELD_PROPOSAL_ROW_CACHE.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();
            for ((&row_idx, &position), row) in missing
                .iter()
                .zip(missing_positions.iter())
                .zip(decoded.into_iter())
            {
                let key = ProposalRowCacheKey {
                    source_ptr: key_base.0,
                    source_len: key_base.1,
                    source_hi_ptr: key_base.2,
                    source_hi_len: key_base.3,
                    row_idx,
                    sample_ptr: key_base.4,
                    sample_len: key_base.5,
                };
                if !cache.rows.contains_key(&key) {
                    cache.order.push_back(key);
                }
                cache.rows.insert(key, row.clone());
                out[position] = Some(row);
                while cache.rows.len() > cache_capacity {
                    let Some(oldest) = cache.order.pop_front() else {
                        break;
                    };
                    cache.rows.remove(&oldest);
                }
            }
        });
    }
    out.into_iter()
        .map(|row| row.ok_or_else(|| "cached proposal row was not materialized".to_string()))
        .collect()
}

#[inline]
pub(super) fn should_parallel_dense_decode(n_rows: usize, n_samples: usize) -> bool {
    rayon::current_num_threads() > 1
        && n_rows >= GARFIELD_DENSE_DECODE_PAR_MIN_ROWS
        && n_samples >= GARFIELD_DENSE_DECODE_PAR_MIN_SAMPLES
}

#[inline]
pub(super) fn dense_dosage_row_from_words(
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
pub(super) fn sample_indices_are_full_identity(
    sample_indices: &[usize],
    n_samples_all: usize,
) -> bool {
    sample_indices.len() == n_samples_all
        && sample_indices.iter().enumerate().all(|(i, &sid)| sid == i)
}

#[inline]
pub(super) fn dosage_stage1_dual_summary_for_row_words(
    row_ge1: &[u64],
    row_ge2: Option<&[u64]>,
    sample_indices: &[usize],
    y: &[f64],
) -> DosageStage1DualSummary {
    let mut n_ge1 = 0usize;
    let mut n_ge2 = 0usize;
    let mut sum_ge1 = 0.0f64;
    let mut sum_ge2 = 0.0f64;
    for (dst_s, &src_s) in sample_indices.iter().enumerate() {
        let ge1 = ((row_ge1[src_s >> 6] >> (src_s & 63)) & 1u64) != 0;
        let ge2 = row_ge2
            .map(|row| ((row[src_s >> 6] >> (src_s & 63)) & 1u64) != 0)
            .unwrap_or(false);
        if ge1 {
            n_ge1 += 1;
            sum_ge1 += y[dst_s];
        }
        if ge2 {
            n_ge2 += 1;
            sum_ge2 += y[dst_s];
        }
    }
    DosageStage1DualSummary {
        n_ge1,
        n_ge2,
        sum_ge1,
        sum_ge2,
    }
}

#[inline]
pub(super) fn dosage_stage1_positive_score_from_summary(
    summary: DosageStage1DualSummary,
    total_sum_y: f64,
    n_samples: usize,
) -> ContinuousRuleScore {
    score::score_cont_centered_gain_dual_from_summary(
        total_sum_y,
        n_samples,
        summary.n_ge1,
        summary.n_ge2,
        summary.sum_ge1,
        summary.sum_ge2,
    )
}

#[inline]
pub(super) fn dosage_stage1_negated_score_from_summary(
    summary: DosageStage1DualSummary,
    total_sum_y: f64,
    n_samples: usize,
) -> ContinuousRuleScore {
    score::score_cont_centered_gain_dual_from_summary(
        total_sum_y,
        n_samples,
        n_samples.saturating_sub(summary.n_ge2),
        n_samples.saturating_sub(summary.n_ge1),
        total_sum_y - summary.sum_ge2,
        total_sum_y - summary.sum_ge1,
    )
}

pub(super) fn dosage_stage1_dual_summaries_from_full_bits(
    bits_flat: &[u64],
    bits_hi_flat: Option<&[u64]>,
    row_words: usize,
    row_indices: &[usize],
    sample_indices: &[usize],
    y: &[f64],
    n_rows_all: usize,
    n_samples_all: usize,
    y_lookup: Option<&PackedYSumLookup>,
    allow_parallel: bool,
) -> Result<Vec<DosageStage1DualSummary>, String> {
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
            "sample_indices length ({}) != y length ({}) while computing dosage stage-1 dual summaries",
            sample_indices.len(),
            y.len()
        ));
    }
    if let Some(&row_idx) = row_indices.iter().find(|&&row_idx| row_idx >= n_rows_all) {
        return Err(format!(
            "row index out of range while computing dosage stage-1 dual summaries: {row_idx}"
        ));
    }
    if let Some(&sid) = sample_indices.iter().find(|&&sid| sid >= n_samples_all) {
        return Err(format!(
            "sample index out of range while computing dosage stage-1 dual summaries: {sid}"
        ));
    }
    let summaries = if sample_indices_are_full_identity(sample_indices, n_samples_all) {
        let zero_hi = if bits_hi_flat.is_none() {
            Some(vec![0u64; row_words])
        } else {
            None
        };
        if allow_parallel && should_parallel_dense_decode(row_indices.len(), sample_indices.len()) {
            row_indices
                .par_iter()
                .map(|&row_idx| {
                    let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                    let row_ge2 = bits_hi_flat
                        .map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words])
                        .unwrap_or_else(|| zero_hi.as_deref().expect("zero hi row must exist"));
                    let (n_ge1, n_ge2, sum_ge1, sum_ge2) = if let Some(lookup) = y_lookup {
                        dual_packed_summary_with_lookup(row_ge1, row_ge2, y, n_samples_all, lookup)
                    } else {
                        dual_packed_summary(row_ge1, row_ge2, y, n_samples_all)
                    };
                    DosageStage1DualSummary {
                        n_ge1,
                        n_ge2,
                        sum_ge1,
                        sum_ge2,
                    }
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
                    let (n_ge1, n_ge2, sum_ge1, sum_ge2) = if let Some(lookup) = y_lookup {
                        dual_packed_summary_with_lookup(row_ge1, row_ge2, y, n_samples_all, lookup)
                    } else {
                        dual_packed_summary(row_ge1, row_ge2, y, n_samples_all)
                    };
                    DosageStage1DualSummary {
                        n_ge1,
                        n_ge2,
                        sum_ge1,
                        sum_ge2,
                    }
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
                let row_ge2 =
                    bits_hi_flat.map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
                dosage_stage1_dual_summary_for_row_words(row_ge1, row_ge2, sample_indices, y)
            })
            .collect()
    } else {
        row_indices
            .iter()
            .map(|&row_idx| {
                let row_ge1 = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
                let row_ge2 =
                    bits_hi_flat.map(|bits| &bits[row_idx * row_words..(row_idx + 1) * row_words]);
                dosage_stage1_dual_summary_for_row_words(row_ge1, row_ge2, sample_indices, y)
            })
            .collect()
    };
    Ok(summaries)
}

pub(super) fn dosage_stage1_dual_summaries_from_full_bits_range(
    bits_flat: &[u64],
    bits_hi_flat: Option<&[u64]>,
    row_words: usize,
    row_start: usize,
    row_end: usize,
    sample_indices: &[usize],
    y: &[f64],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<Vec<DosageStage1DualSummary>, String> {
    if row_end <= row_start {
        return Ok(Vec::new());
    }
    let row_indices = (row_start..row_end).collect::<Vec<_>>();
    dosage_stage1_dual_summaries_from_full_bits(
        bits_flat,
        bits_hi_flat,
        row_words,
        row_indices.as_slice(),
        sample_indices,
        y,
        n_rows_all,
        n_samples_all,
        None,
        true,
    )
}

pub(super) fn dosage_stage1_raw_scores_from_dual_summaries(
    summaries: &[DosageStage1DualSummary],
    total_sum_y: f64,
    n_samples: usize,
    allow_parallel: bool,
) -> Vec<f64> {
    if allow_parallel && should_parallel_dense_decode(summaries.len(), n_samples) {
        summaries
            .par_iter()
            .map(|summary| {
                dosage_stage1_positive_score_from_summary(*summary, total_sum_y, n_samples)
                    .raw_score
            })
            .collect()
    } else {
        summaries
            .iter()
            .map(|summary| {
                dosage_stage1_positive_score_from_summary(*summary, total_sum_y, n_samples)
                    .raw_score
            })
            .collect()
    }
}

/// Compute the dense Corr proposal statistic directly from packed stage-1
/// summaries.  The dense proposal channel first maps every non-zero dosage to
/// a binary carrier indicator, so only the GE1 count/sum are needed here;
/// GE2 remains available to the caller for dosage/MAF bookkeeping.
#[inline]
pub(super) fn corr_score_from_dual_summary(
    summary: DosageStage1DualSummary,
    total_sum_y: f64,
    total_sum_y2: f64,
    n_samples: usize,
) -> f64 {
    if n_samples == 0 {
        return 0.0;
    }
    let n = n_samples as f64;
    let mean_y = total_sum_y / n;
    let var_y = total_sum_y2 / n - mean_y * mean_y;
    if !var_y.is_finite() || var_y <= 0.0 {
        return 0.0;
    }
    let hit = summary.n_ge1 as f64;
    let mean_x = hit / n;
    let var_x = hit / n - mean_x * mean_x;
    if !var_x.is_finite() || var_x <= 0.0 {
        return 0.0;
    }
    let covariance = summary.sum_ge1 / n - mean_x * mean_y;
    (covariance / (var_x * var_y).sqrt()).abs()
}

pub(super) fn corr_scores_from_dual_summaries(
    summaries: &[DosageStage1DualSummary],
    total_sum_y: f64,
    total_sum_y2: f64,
    n_samples: usize,
    allow_parallel: bool,
) -> Vec<f64> {
    if allow_parallel && should_parallel_dense_decode(summaries.len(), n_samples) {
        summaries
            .par_iter()
            .map(|summary| {
                corr_score_from_dual_summary(*summary, total_sum_y, total_sum_y2, n_samples)
            })
            .collect()
    } else {
        summaries
            .iter()
            .map(|summary| {
                corr_score_from_dual_summary(*summary, total_sum_y, total_sum_y2, n_samples)
            })
            .collect()
    }
}

pub(super) fn build_cached_literal_scores_from_selected_dual_summaries(
    candidate_global_rows: &[usize],
    selected_global_rows: &[usize],
    summaries: &[DosageStage1DualSummary],
    total_sum_y: f64,
    n_samples: usize,
    single_plane_binary: bool,
) -> Result<Vec<LiteralSingletonScore>, String> {
    let summary_index = candidate_global_rows
        .iter()
        .enumerate()
        .map(|(i, &global_idx)| (global_idx, i))
        .collect::<HashMap<usize, usize>>();
    let mut out =
        Vec::<LiteralSingletonScore>::with_capacity(selected_global_rows.len().saturating_mul(2));
    for &global_idx in selected_global_rows.iter() {
        let src_idx = summary_index.get(&global_idx).copied().ok_or_else(|| {
            format!(
                "GARFIELD Corr singleton cache lost stage-1 summary for row {}",
                global_idx
            )
        })?;
        let summary = summaries[src_idx];
        let (pos, neg) = if single_plane_binary {
            // Single-plane BIN rows encode only one homozygote class. Their raw
            // centered-gain score is identical to the dual-summary route when
            // n_ge2==0, but dosage_maf must remain the binary minor-class
            // frequency instead of being halved as if the row were 0/1 dosage.
            let pos = score::score_cont_centered_gain_from_sum_and_n_hit(
                total_sum_y,
                summary.sum_ge1,
                n_samples,
                summary.n_ge1,
            );
            let neg = score::score_cont_centered_gain_from_sum_and_n_hit(
                total_sum_y,
                total_sum_y - summary.sum_ge1,
                n_samples,
                n_samples.saturating_sub(summary.n_ge1),
            );
            (pos, neg)
        } else {
            (
                dosage_stage1_positive_score_from_summary(summary, total_sum_y, n_samples),
                dosage_stage1_negated_score_from_summary(summary, total_sum_y, n_samples),
            )
        };
        out.push(LiteralSingletonScore {
            train: pos,
            test: pos,
        });
        out.push(LiteralSingletonScore {
            train: neg,
            test: neg,
        });
    }
    Ok(out)
}

pub(super) static PACKED_EXTRACT_FLAT_NS: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(super) fn elapsed_ns_saturating(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

pub(super) fn packed_dosage_rows_subset_from_full_bits_with_stage1(
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

pub(super) fn packed_dosage_rows_subset_from_full_bits_range_with_stage1(
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

pub(super) fn packed_rows_subset_from_full_bits_range(
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

pub(super) fn borrow_rows_from_full_bits_range<'a>(
    bits_flat: &'a [u64],
    row_words_full: usize,
    row_start: usize,
    row_end: usize,
    n_rows_all: usize,
    n_samples_all: usize,
    ctx: &str,
) -> Result<&'a [u64], String> {
    if row_end <= row_start {
        return Ok(&bits_flat[0..0]);
    }
    if row_end > n_rows_all {
        return Err(format!(
            "{ctx}: row range out of bounds: [{row_start}, {row_end}) vs n_rows={n_rows_all}"
        ));
    }
    if row_words_full != words_for_samples(n_samples_all) {
        return Err(format!(
            "{ctx}: row_words mismatch for full bit matrix: got {row_words_full}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words_full) {
        return Err(format!("{ctx}: full bit matrix length mismatch"));
    }
    let start = row_start.saturating_mul(row_words_full);
    let end = row_end.saturating_mul(row_words_full);
    bits_flat
        .get(start..end)
        .ok_or_else(|| format!("{ctx}: bit slice range out of bounds"))
}

pub(super) fn packed_rows_subset_from_full_bits(
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
pub(super) fn binary_row_var_from_support(support: usize, n_samples: usize) -> f64 {
    if n_samples == 0 {
        return 0.0;
    }
    let p = (support as f64) / (n_samples as f64);
    p * (1.0 - p)
}

#[inline]
pub(super) fn binary_pair_high_ld_from_support_and_both_one(
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
pub(super) fn binary_pair_abs_r2_upper_bound_from_support(
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

pub(super) fn ld_exact_bounds_from_support(
    support_i: usize,
    support_j: usize,
    n_samples: usize,
    r2_threshold: f64,
) -> GarfieldLdExactBounds {
    if n_samples == 0
        || support_i == 0
        || support_j == 0
        || support_i >= n_samples
        || support_j >= n_samples
    {
        return GarfieldLdExactBounds::default();
    }
    let c_min = support_i
        .saturating_add(support_j)
        .saturating_sub(n_samples);
    let c_max = support_i.min(support_j);
    if c_min > c_max {
        return GarfieldLdExactBounds::default();
    }
    let ss_i = (n_samples as f64) * binary_row_var_from_support(support_i, n_samples);
    let ss_j = (n_samples as f64) * binary_row_var_from_support(support_j, n_samples);
    if !(ss_i > 0.0 && ss_j > 0.0) {
        return GarfieldLdExactBounds::default();
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

    GarfieldLdExactBounds {
        low_max: if low_max >= c_min_i {
            u32::try_from(low_max).unwrap_or(u32::MAX)
        } else {
            GARFIELD_LD_NO_BOUND
        },
        high_min: if high_min <= c_max_i {
            u32::try_from(high_min).unwrap_or(u32::MAX)
        } else {
            GARFIELD_LD_NO_BOUND
        },
    }
}

pub(super) fn cached_ld_support_entry<F>(
    registry: &GarfieldLdSupportCacheRegistry,
    key: (usize, u64),
    build: F,
) -> Arc<GarfieldLdSupportCacheEntry>
where
    F: FnOnce() -> GarfieldLdSupportCacheEntry,
{
    let slot = {
        let mut entries = registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Arc::clone(
            entries
                .entry(key)
                .or_insert_with(|| Arc::new(OnceLock::new())),
        )
    };
    Arc::clone(slot.get_or_init(|| Arc::new(build())))
}

pub(super) fn build_ld_support_cache_entry(
    n_samples: usize,
    r2_threshold: f64,
) -> GarfieldLdSupportCacheEntry {
    let mut buckets = vec![Vec::<usize>::new(); n_samples + 1];
    let mut exact_bounds =
        vec![vec![GarfieldLdExactBounds::default(); n_samples + 1]; n_samples + 1];
    for (support_i, bucket) in buckets.iter_mut().enumerate().take(n_samples).skip(1) {
        let mut allowed = Vec::<usize>::new();
        for support_j in 1..n_samples {
            if binary_pair_abs_r2_upper_bound_from_support(support_i, support_j, n_samples)
                >= r2_threshold
            {
                allowed.push(support_j);
                exact_bounds[support_i][support_j] =
                    ld_exact_bounds_from_support(support_i, support_j, n_samples, r2_threshold);
            }
        }
        *bucket = allowed;
    }

    GarfieldLdSupportCacheEntry {
        conflict_supports: buckets,
        exact_bounds,
    }
}

pub(super) fn ld_support_cache(
    n_samples: usize,
    r2_threshold: f64,
) -> Arc<GarfieldLdSupportCacheEntry> {
    let cache = GARFIELD_LD_SUPPORT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    cached_ld_support_entry(cache, (n_samples, r2_threshold.to_bits()), || {
        build_ld_support_cache_entry(n_samples, r2_threshold)
    })
}
