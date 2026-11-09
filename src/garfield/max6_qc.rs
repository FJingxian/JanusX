use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Max6FusedRowQc {
    pub(super) keep: bool,
    pub(super) reason: u8,
    pub(super) row_flip: bool,
    pub(super) has_het: bool,
    pub(super) has_raw_missing: bool,
}

pub(super) const MAX6_FUSED_QC_KEEP: u8 = 0;
pub(super) const MAX6_FUSED_QC_FAIL_MISSING: u8 = 1;
pub(super) const MAX6_FUSED_QC_FAIL_HET: u8 = 2;
pub(super) const MAX6_FUSED_QC_FAIL_NO_HOMOZYGOUS: u8 = 3;
pub(super) const MAX6_FUSED_QC_FAIL_MAF: u8 = 4;
pub(super) const MAX6_FUSED_QC_FAIL_NON_SIMPLE_SNP: u8 = 5;

#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn max6_identity_pack_bmi2_runtime_available() -> bool {
    static BMI2: OnceLock<bool> = OnceLock::new();
    *BMI2.get_or_init(|| std::arch::is_x86_feature_detected!("bmi2"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
pub(super) unsafe fn pack_max6_identity_row_logic_bmi2(
    row: &[u8],
    n_samples: usize,
    dst_words: &mut [u64],
    missing_words: &mut [u64],
) {
    use std::arch::x86_64::_pext_u64;

    const LOW_BITS: u64 = 0x5555_5555_5555_5555;
    let mut sample = 0usize;
    while sample + 32 <= n_samples {
        let byte_start = sample / 4;
        let packed = u64::from_le_bytes(
            row[byte_start..byte_start + 8]
                .try_into()
                .expect("eight packed genotype bytes form 32 samples"),
        );
        let low = packed & LOW_BITS;
        let high = (packed >> 1) & LOW_BITS;
        let alt_bits = _pext_u64(low & high, LOW_BITS) as u32;
        let missing_bits = _pext_u64(low ^ high, LOW_BITS) as u32;
        let word_idx = sample >> 6;
        let shift = (sample & 63) as u32;
        dst_words[word_idx] |= (alt_bits as u64) << shift;
        missing_words[word_idx] |= (missing_bits as u64) << shift;
        sample += 32;
    }

    let tail_samples = n_samples - sample;
    if tail_samples > 0 {
        let byte_start = sample / 4;
        let tail_bytes = tail_samples.div_ceil(4);
        let mut tail = [0u8; 8];
        tail[..tail_bytes].copy_from_slice(&row[byte_start..byte_start + tail_bytes]);
        let active_lanes = tail_samples & 3;
        if active_lanes > 0 {
            tail[tail_bytes - 1] &= (1u8 << (active_lanes * 2)) - 1;
        }
        let packed = u64::from_le_bytes(tail);
        let low = packed & LOW_BITS;
        let high = (packed >> 1) & LOW_BITS;
        let alt_bits = _pext_u64(low & high, LOW_BITS) as u32;
        let missing_bits = _pext_u64(low ^ high, LOW_BITS) as u32;
        let word_idx = sample >> 6;
        let shift = (sample & 63) as u32;
        dst_words[word_idx] |= (alt_bits as u64) << shift;
        missing_words[word_idx] |= (missing_bits as u64) << shift;
    }
}

#[inline]
pub(super) fn compact_max6_even_bits_swar(bits: u64) -> u32 {
    let mut packed = bits & 0x5555_5555_5555_5555;
    packed = (packed | (packed >> 1)) & 0x3333_3333_3333_3333;
    packed = (packed | (packed >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    packed = (packed | (packed >> 4)) & 0x00ff_00ff_00ff_00ff;
    packed = (packed | (packed >> 8)) & 0x0000_ffff_0000_ffff;
    packed = (packed | (packed >> 16)) & 0x0000_0000_ffff_ffff;
    packed as u32
}

#[inline]
pub(super) fn pack_max6_identity_row_logic_swar(
    row: &[u8],
    n_samples: usize,
    dst_words: &mut [u64],
    missing_words: &mut [u64],
) {
    const LOW_BITS: u64 = 0x5555_5555_5555_5555;

    let mut sample = 0usize;
    while sample + 32 <= n_samples {
        let byte_start = sample / 4;
        let packed = u64::from_le_bytes(
            row[byte_start..byte_start + 8]
                .try_into()
                .expect("eight packed genotype bytes form 32 samples"),
        );
        let low = packed & LOW_BITS;
        let high = (packed >> 1) & LOW_BITS;
        let alt_bits = compact_max6_even_bits_swar(low & high);
        let missing_bits = compact_max6_even_bits_swar(low ^ high);
        let word_idx = sample >> 6;
        let shift = (sample & 63) as u32;
        dst_words[word_idx] |= (alt_bits as u64) << shift;
        missing_words[word_idx] |= (missing_bits as u64) << shift;
        sample += 32;
    }

    let tail_samples = n_samples - sample;
    if tail_samples > 0 {
        let byte_start = sample / 4;
        let tail_bytes = tail_samples.div_ceil(4);
        let mut tail = [0u8; 8];
        tail[..tail_bytes].copy_from_slice(&row[byte_start..byte_start + tail_bytes]);
        let active_lanes = tail_samples & 3;
        if active_lanes > 0 {
            tail[tail_bytes - 1] &= (1u8 << (active_lanes * 2)) - 1;
        }
        let packed = u64::from_le_bytes(tail);
        let low = packed & LOW_BITS;
        let high = (packed >> 1) & LOW_BITS;
        let alt_bits = compact_max6_even_bits_swar(low & high);
        let missing_bits = compact_max6_even_bits_swar(low ^ high);
        let word_idx = sample >> 6;
        let shift = (sample & 63) as u32;
        dst_words[word_idx] |= (alt_bits as u64) << shift;
        missing_words[word_idx] |= (missing_bits as u64) << shift;
    }
}

#[inline]
pub(super) fn pack_max6_identity_row_logic_lut(
    row: &[u8],
    n_samples: usize,
    dst_words: &mut [u64],
    missing_words: &mut [u64],
) {
    let lut = packed_byte_lut();
    let active_bytes = n_samples.div_ceil(4);
    let rem_lanes = n_samples & 3;
    for byte_idx in 0..active_bytes {
        let active_lanes = if rem_lanes > 0 && byte_idx + 1 == active_bytes {
            rem_lanes
        } else {
            4
        };
        let mut byte = row[byte_idx];
        if active_lanes < 4 {
            byte &= (1u8 << (active_lanes * 2)) - 1;
        }
        let idx = byte as usize;
        let word_idx = byte_idx >> 4;
        let shift = ((byte_idx & 15) * 4) as u32;
        dst_words[word_idx] |= (lut.logic_alt_bits[idx] as u64) << shift;
        missing_words[word_idx] |= (lut.logic_missing_bits[idx] as u64) << shift;
    }
}

#[inline]
pub(super) fn pack_max6_identity_row_logic(
    row: &[u8],
    n_samples: usize,
    dst_words: &mut [u64],
    missing_words: &mut [u64],
) {
    dst_words.fill(0);
    missing_words.fill(0);

    #[cfg(target_arch = "x86_64")]
    if n_samples >= 32 && max6_identity_pack_bmi2_runtime_available() {
        // SAFETY: the runtime feature check above guarantees BMI2 support.
        unsafe {
            pack_max6_identity_row_logic_bmi2(row, n_samples, dst_words, missing_words);
        }
        return;
    }

    if n_samples >= 32 {
        pack_max6_identity_row_logic_swar(row, n_samples, dst_words, missing_words);
    } else {
        pack_max6_identity_row_logic_lut(row, n_samples, dst_words, missing_words);
    }
}

#[inline]
pub(super) fn scan_max6_identity_row_qc_counts(
    row: &[u8],
    n_samples: usize,
) -> (usize, usize, usize, bool, bool) {
    const LOW_BITS: u64 = 0x5555_5555_5555_5555;

    let full_word_bytes = (n_samples / 32) * 8;
    let mut logic_missing = 0usize;
    let mut hom_alt = 0usize;
    let mut has_het = false;
    let mut has_raw_missing = false;
    for packed_bytes in row[..full_word_bytes].chunks_exact(8) {
        let packed = u64::from_le_bytes(
            packed_bytes
                .try_into()
                .expect("eight packed genotype bytes form one word"),
        );
        let low = packed & LOW_BITS;
        let high = (packed >> 1) & LOW_BITS;
        logic_missing += (low ^ high).count_ones() as usize;
        hom_alt += (low & high).count_ones() as usize;
        has_het |= high & !low != 0;
        has_raw_missing |= low & !high != 0;
    }

    let lut = packed_byte_lut();
    let active_bytes = n_samples.div_ceil(4);
    let rem_lanes = n_samples & 3;
    for byte_idx in full_word_bytes..active_bytes {
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
        logic_missing += lut.logic_missing[idx] as usize;
        hom_alt += lut.hom_alt[idx] as usize;
        has_het |= lut.has_het[idx];
        has_raw_missing |= lut.has_raw_missing[idx];
    }
    let hom_ref = n_samples
        .saturating_sub(logic_missing)
        .saturating_sub(hom_alt);
    (logic_missing, hom_alt, hom_ref, has_het, has_raw_missing)
}

/// Count pure-line QC statistics and build the Max6 BIN row in one traversal
/// of a mapped BED marker. QC and output use the same selected samples, so
/// the homozygous counts collected here also determine the existing majority
/// imputation and allele orientation.
#[inline]
pub(super) fn scan_max6_row_qc_and_pack_logic(
    row: &[u8],
    n_samples: usize,
    sample_identity: bool,
    sample_plan: &[LogicSamplePlanEntry],
    maf_threshold: f32,
    max_missing_rate: f32,
    simple_snp_alleles: bool,
    dst_words: &mut [u64],
    missing_words: &mut [u64],
) -> Max6FusedRowQc {
    let (logic_missing, hom_alt, hom_ref, has_het, has_raw_missing) = if sample_identity {
        scan_max6_identity_row_qc_counts(row, n_samples)
    } else {
        dst_words.fill(0);
        missing_words.fill(0);
        scan_max6_subset_row_and_pack(row, sample_plan, dst_words, missing_words)
    };

    let logic_missing = logic_missing.min(n_samples);
    let missing_rate = if n_samples > 0 {
        logic_missing as f32 / n_samples as f32
    } else {
        0.0
    };
    let usable_homo = n_samples.saturating_sub(logic_missing);
    let alt_freq = if usable_homo > 0 {
        hom_alt as f32 / usable_homo as f32
    } else {
        0.0
    };
    let maf = alt_freq.min(1.0 - alt_freq).max(0.0);

    let reason = if missing_rate > max_missing_rate {
        MAX6_FUSED_QC_FAIL_MISSING
    } else if usable_homo == 0 {
        MAX6_FUSED_QC_FAIL_NO_HOMOZYGOUS
    } else if maf < maf_threshold {
        MAX6_FUSED_QC_FAIL_MAF
    } else if !simple_snp_alleles {
        MAX6_FUSED_QC_FAIL_NON_SIMPLE_SNP
    } else {
        MAX6_FUSED_QC_KEEP
    };
    let row_flip = alt_freq > 0.5_f32;
    let keep = reason == MAX6_FUSED_QC_KEEP;
    if keep {
        if sample_identity {
            // Identity rows arrive in a fresh zeroed batch slice. Delay all
            // output writes until QC passes so rejected rows stay untouched.
            pack_max6_identity_row_logic(row, n_samples, dst_words, missing_words);
        }
        let n_words = dst_words.len();
        let tail_bits = n_samples & 63;
        for word_idx in 0..n_words {
            let missing = missing_words[word_idx];
            let mut word = dst_words[word_idx];
            if row_flip {
                word = !word & !missing;
            }
            let majority_is_one = if row_flip {
                hom_ref > hom_alt
            } else {
                hom_alt > hom_ref
            };
            if majority_is_one {
                word |= missing;
            }
            if tail_bits > 0 && word_idx + 1 == n_words {
                if let Some(mask) = tail_mask(tail_bits) {
                    word &= mask;
                }
            }
            dst_words[word_idx] = word;
        }
    }

    Max6FusedRowQc {
        keep,
        reason,
        row_flip,
        has_het,
        has_raw_missing,
    }
}

#[derive(Debug)]
pub(super) struct Max6BitsTempFileOwner {
    pub(super) path: std::path::PathBuf,
}

impl Drop for Max6BitsTempFileOwner {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub(super) const MAX6_WINDOW_STREAM_CHUNK_ROWS: usize = 4096;
pub(super) const MAX6_WINDOW_STREAM_METADATA_ROWS: usize = 4096;
pub(super) const MAX6_WINDOW_STREAM_PGEN_CHUNK_ROWS: usize = 16_384;
pub(super) const MAX6_WINDOW_STREAM_PGEN_METADATA_ROWS: usize = 16_384;

pub(super) struct Max6WindowStreamChunk {
    pub(super) sites: Vec<SiteInfo>,
    pub(super) source_rows: Vec<usize>,
    pub(super) packed_genotypes: Vec<u8>,
    pub(super) qc: Vec<Max6FusedRowQc>,
    pub(super) logic_bits: Vec<u64>,
    pub(super) error: Option<String>,
}

impl Max6WindowStreamChunk {
    pub(super) fn new(max_rows: usize, bytes_per_variant: usize, row_words: usize) -> Self {
        Self {
            sites: Vec::with_capacity(max_rows),
            source_rows: Vec::with_capacity(max_rows),
            packed_genotypes: Vec::with_capacity(max_rows.saturating_mul(bytes_per_variant)),
            qc: Vec::with_capacity(max_rows),
            logic_bits: Vec::with_capacity(max_rows.saturating_mul(row_words)),
            error: None,
        }
    }

    pub(super) fn clear(&mut self) {
        self.sites.clear();
        self.source_rows.clear();
        self.packed_genotypes.clear();
        self.qc.clear();
        self.logic_bits.clear();
        self.error = None;
    }
}

#[inline]
pub(super) fn max6_window_site_in_pack_scope(
    site: &SiteInfo,
    scan_bimranges: &[GarfieldScanBimRange],
    padding_bp: i64,
) -> bool {
    scan_bimranges.is_empty()
        || scan_bimranges.iter().any(|range| {
            if !same_normalized_chrom(site.chrom.as_str(), range.chrom.as_str()) {
                return false;
            }
            let start = i64::from(range.bp_start).saturating_sub(padding_bp).max(0);
            let end = i64::from(range.bp_end)
                .saturating_add(padding_bp)
                .min(i64::from(i32::MAX));
            (i64::from(site.pos) >= start) && (i64::from(site.pos) <= end)
        })
}

pub(super) fn max6_window_stream_scope_finish_bp(
    chrom: &str,
    scan_bimranges: &[GarfieldScanBimRange],
    padding_bp: i64,
) -> Option<i32> {
    let max_end = scan_bimranges
        .iter()
        .filter(|range| same_normalized_chrom(chrom, range.chrom.as_str()))
        .map(|range| i64::from(range.bp_end).saturating_add(padding_bp))
        .max()?;
    Some(clamp_i64_to_i32(max_end.saturating_add(1)))
}

/// Sequential BIM/PVAR metadata reader that decodes genotypes only for the
/// chromosomes selected by `--bimrange`. The fused window path can also seek
/// over selected chromosomes and decode only rows inside window scope plus the
/// first QC-passing row needed to anchor the chromosome's window grid.
pub(super) struct Max6WindowRangeSource {
    pub(super) bed: BedSnpIter,
    pub(super) bim: BimChunkReader,
    pub(super) max_chunk_rows: usize,
    pub(super) metadata_chunk_rows: usize,
    pub(super) n_source_rows: usize,
    pub(super) next_metadata_row: usize,
    pub(super) selected_chroms: HashSet<String>,
    pub(super) seen_selected_chroms: HashSet<String>,
    pub(super) last_chrom: Option<String>,
    pub(super) pending: VecDeque<(usize, SiteInfo)>,
    pub(super) scan_all: bool,
    pub(super) scan_bimranges: Vec<GarfieldScanBimRange>,
    pub(super) pack_scope_by_chrom: HashMap<String, Vec<(i64, i64)>>,
    pub(super) pack_scope_padding_bp: i64,
    pub(super) restrict_genotypes_to_scope: bool,
    pub(super) anchor_ready_chroms: Arc<Mutex<HashSet<String>>>,
    pub(super) tail_scope_complete_chroms: Arc<Mutex<HashSet<String>>>,
    pub(super) last_chrom_max_pos: Option<i32>,
    pub(super) done: bool,
    pub(super) validated_full_metadata: bool,
}

impl Max6WindowRangeSource {
    pub(super) fn new(
        prefix: &str,
        ranges: &[GarfieldScanBimRange],
        extension: usize,
        restrict_genotypes_to_scope: bool,
        anchor_ready_chroms: Arc<Mutex<HashSet<String>>>,
        tail_scope_complete_chroms: Arc<Mutex<HashSet<String>>>,
    ) -> Result<Self, String> {
        let pgen_source = crate::pgen::path_uses_pgen(prefix);
        let prefix_base = normalize_plink_prefix(prefix);
        let reader_prefix = if pgen_source {
            format!("{prefix_base}.pgen")
        } else {
            prefix_base
        };
        let bed = BedSnpIter::new_for_grm_window_header_count(
            reader_prefix.as_str(),
            garfield_logic_mmap_window_mb(),
        )?;
        let n_source_rows = bed.n_snps();
        if n_source_rows == 0 {
            return Err("no SNP sites found in genotype input".to_string());
        }
        let selected_chroms = ranges
            .iter()
            .map(|range| chrom_base_view(range.chrom.as_str()).to_ascii_lowercase())
            .collect::<HashSet<_>>();
        let pack_scope_padding_bp = (extension.max(1) as i64).saturating_mul(2);
        let mut pack_scope_by_chrom = HashMap::<String, Vec<(i64, i64)>>::new();
        for range in ranges {
            let chrom = chrom_base_view(range.chrom.as_str()).to_ascii_lowercase();
            let start = i64::from(range.bp_start)
                .saturating_sub(pack_scope_padding_bp)
                .max(0);
            let end = i64::from(range.bp_end)
                .saturating_add(pack_scope_padding_bp)
                .min(i64::from(i32::MAX));
            pack_scope_by_chrom
                .entry(chrom)
                .or_default()
                .push((start, end));
        }
        for intervals in pack_scope_by_chrom.values_mut() {
            intervals.sort_unstable();
            let mut merged = Vec::<(i64, i64)>::with_capacity(intervals.len());
            for (start, end) in intervals.drain(..) {
                if let Some(last) = merged.last_mut() {
                    if start <= last.1.saturating_add(1) {
                        last.1 = last.1.max(end);
                        continue;
                    }
                }
                merged.push((start, end));
            }
            *intervals = merged;
        }
        let (max_chunk_rows, metadata_chunk_rows) = if bed.has_pgen_source() {
            (
                MAX6_WINDOW_STREAM_PGEN_CHUNK_ROWS,
                MAX6_WINDOW_STREAM_PGEN_METADATA_ROWS,
            )
        } else {
            (
                MAX6_WINDOW_STREAM_CHUNK_ROWS,
                MAX6_WINDOW_STREAM_METADATA_ROWS,
            )
        };
        let scan_all = selected_chroms.is_empty();
        Ok(Self {
            bed,
            bim: BimChunkReader::open(reader_prefix.as_str())?,
            max_chunk_rows,
            metadata_chunk_rows,
            n_source_rows,
            next_metadata_row: 0,
            selected_chroms,
            seen_selected_chroms: HashSet::new(),
            last_chrom: None,
            pending: VecDeque::new(),
            scan_all,
            scan_bimranges: ranges.to_vec(),
            pack_scope_by_chrom,
            pack_scope_padding_bp,
            restrict_genotypes_to_scope: restrict_genotypes_to_scope && !ranges.is_empty(),
            anchor_ready_chroms,
            tail_scope_complete_chroms,
            last_chrom_max_pos: None,
            done: false,
            validated_full_metadata: false,
        })
    }

    pub(super) fn mark_chromosome_tail_scope_complete(
        &mut self,
        chrom: &str,
        max_pos: i32,
    ) -> Result<(), String> {
        if !self.restrict_genotypes_to_scope
            || max6_window_stream_scope_finish_bp(
                chrom,
                self.scan_bimranges.as_slice(),
                self.pack_scope_padding_bp,
            )
            .is_none_or(|scope_end| scope_end <= max_pos)
        {
            return Ok(());
        }
        self.tail_scope_complete_chroms
            .lock()
            .map_err(|_| "Max6 window range tail state was poisoned".to_string())?
            .insert(normalize_chrom(chrom));
        Ok(())
    }

    pub(super) fn read_metadata_chunk(&mut self) -> Result<(), String> {
        if self.next_metadata_row >= self.n_source_rows {
            if !self.validated_full_metadata {
                if let (Some(chrom), Some(max_pos)) =
                    (self.last_chrom.clone(), self.last_chrom_max_pos)
                {
                    self.mark_chromosome_tail_scope_complete(chrom.as_str(), max_pos)?;
                }
                self.bim.ensure_exhausted(self.n_source_rows)?;
                self.validated_full_metadata = true;
            }
            self.done = true;
            return Ok(());
        }
        let start = self.next_metadata_row;
        let end = start
            .saturating_add(self.metadata_chunk_rows)
            .min(self.n_source_rows);
        let sites = self.bim.read_range(start, end)?;
        if sites.len() != end.saturating_sub(start) {
            return Err(format!(
                "BIM/PVAR chunk length mismatch: read {}, expected {} rows at {start}",
                sites.len(),
                end.saturating_sub(start)
            ));
        }
        self.next_metadata_row = end;
        let anchor_ready_snapshot = if self.restrict_genotypes_to_scope {
            self.anchor_ready_chroms
                .lock()
                .map_err(|_| "Max6 window range anchor state was poisoned".to_string())?
                .clone()
        } else {
            HashSet::new()
        };
        for (offset, site) in sites.into_iter().enumerate() {
            let source_row = start + offset;
            let chrom = chrom_base_view(site.chrom.as_str()).to_ascii_lowercase();
            let changed = self
                .last_chrom
                .as_ref()
                .is_some_and(|previous| previous != &chrom);
            if changed {
                if let (Some(previous), Some(max_pos)) =
                    (self.last_chrom.clone(), self.last_chrom_max_pos)
                {
                    self.mark_chromosome_tail_scope_complete(previous.as_str(), max_pos)?;
                }
                self.last_chrom_max_pos = None;
            }
            if !self.scan_all {
                if changed
                    && self.seen_selected_chroms.len() == self.selected_chroms.len()
                    && self
                        .last_chrom
                        .as_ref()
                        .is_some_and(|previous| self.selected_chroms.contains(previous))
                    && !self.selected_chroms.contains(&chrom)
                {
                    self.done = true;
                    break;
                }
            }
            self.last_chrom = Some(chrom.clone());
            let selected = self.scan_all || self.selected_chroms.contains(&chrom);
            if selected {
                self.last_chrom_max_pos = Some(
                    self.last_chrom_max_pos
                        .map_or(site.pos, |previous| previous.max(site.pos)),
                );
                if !self.scan_all {
                    self.seen_selected_chroms.insert(chrom.clone());
                }
                let in_scope = !self.restrict_genotypes_to_scope
                    || self.site_in_pack_scope(site.chrom.as_str(), site.pos);
                let anchor_not_ready =
                    self.restrict_genotypes_to_scope && !anchor_ready_snapshot.contains(&chrom);
                if in_scope || anchor_not_ready {
                    self.pending.push_back((source_row, site));
                }
            }
        }
        if self.next_metadata_row >= self.n_source_rows && !self.done {
            if let (Some(chrom), Some(max_pos)) = (self.last_chrom.clone(), self.last_chrom_max_pos)
            {
                self.mark_chromosome_tail_scope_complete(chrom.as_str(), max_pos)?;
            }
            self.bim.ensure_exhausted(self.n_source_rows)?;
            self.validated_full_metadata = true;
            self.done = true;
        }
        Ok(())
    }

    fn site_in_pack_scope(&self, chrom: &str, pos: i32) -> bool {
        let key = chrom_base_view(chrom).to_ascii_lowercase();
        let Some(intervals) = self.pack_scope_by_chrom.get(key.as_str()) else {
            return false;
        };
        let pos = i64::from(pos);
        let index = intervals.partition_point(|&(_, end)| end < pos);
        intervals
            .get(index)
            .is_some_and(|&(start, end)| start <= pos && pos <= end)
    }

    pub(super) fn fill_chunk(&mut self, chunk: &mut Max6WindowStreamChunk) -> Result<bool, String> {
        chunk.clear();
        while self.pending.is_empty() && !self.done {
            self.read_metadata_chunk()?;
        }
        let Some(&(first_source_row, _)) = self.pending.front() else {
            return Ok(false);
        };
        self.bed.set_cursor(first_source_row);
        let max_rows = if self.bed.has_pgen_source() {
            self.max_chunk_rows
        } else {
            self.bed.ensure_window_for_snp(first_source_row)?;
            self.max_chunk_rows
                .min(self.bed.mapped_contiguous_snps_from(first_source_row))
        };
        if max_rows == 0 {
            return Err(format!(
                "empty mapped genotype block at source row {first_source_row}"
            ));
        }
        let mut next_source_row = first_source_row;
        while chunk.sites.len() < max_rows {
            let Some((source_row, _)) = self.pending.front() else {
                break;
            };
            if *source_row != next_source_row {
                break;
            }
            let (source_row, site) = self.pending.pop_front().expect("front checked");
            chunk.source_rows.push(source_row);
            chunk.sites.push(site);
            next_source_row = source_row.saturating_add(1);
        }
        if chunk.sites.is_empty() {
            return Err("streaming genotype source failed to advance".to_string());
        }

        let bytes_per_variant = self.bed.bytes_per_snp();
        chunk
            .packed_genotypes
            .resize(chunk.sites.len().saturating_mul(bytes_per_variant), 0);
        self.bed.read_packed_source_rows(
            first_source_row,
            chunk.sites.len(),
            chunk.packed_genotypes.as_mut_slice(),
        )?;
        Ok(true)
    }
}

pub(super) struct Max6StreamingWriteBlock {
    pub(super) data: Vec<u64>,
    pub(super) words: usize,
}

/// Bounded asynchronous writer for the packed Max6 logic rows. Two recycled
/// buffers cap queued write memory while the BED/PGEN producer continues QC
/// and packing in parallel.
pub(super) struct Max6StreamingLogicBitsAccumulator {
    pub(super) row_words: usize,
    pub(super) block_words: usize,
    pub(super) words: usize,
    pub(super) path: Option<std::path::PathBuf>,
    pub(super) tx: Option<std::sync::mpsc::SyncSender<Max6StreamingWriteBlock>>,
    pub(super) free_rx: std::sync::mpsc::Receiver<Max6StreamingWriteBlock>,
    pub(super) handle: Option<std::thread::JoinHandle<Result<(), String>>>,
}

impl Max6StreamingLogicBitsAccumulator {
    pub(super) fn new(row_words: usize, block_rows: usize) -> Result<Self, String> {
        if row_words == 0 || block_rows == 0 {
            return Err("streaming Max6 logic writer requires nonzero row and block sizes".into());
        }
        let block_words = row_words
            .checked_mul(block_rows)
            .ok_or_else(|| "streaming Max6 logic block size overflow".to_string())?;
        let path = std::env::temp_dir().join(format!(
            "janusx_garfield_stream_bits_{}_{}.bin",
            std::process::id(),
            GARFIELD_BITS_MMAP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                format!("create streaming Max6 bitset '{}': {error}", path.display())
            })?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<Max6StreamingWriteBlock>(2);
        let (free_tx, free_rx) = std::sync::mpsc::sync_channel::<Max6StreamingWriteBlock>(2);
        for _ in 0..2 {
            free_tx
                .send(Max6StreamingWriteBlock {
                    data: vec![0u64; block_words],
                    words: 0,
                })
                .map_err(|error| format!("initialize streaming Max6 write buffers: {error}"))?;
        }
        let writer_path = path.clone();
        let handle = match std::thread::Builder::new()
            .name("garfield-max6-bit-writer".to_string())
            .spawn(move || -> Result<(), String> {
                let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
                for mut block in rx {
                    let byte_len = block.words.saturating_mul(std::mem::size_of::<u64>());
                    let bytes = unsafe {
                        std::slice::from_raw_parts(block.data.as_ptr() as *const u8, byte_len)
                    };
                    writer.write_all(bytes).map_err(|error| {
                        format!(
                            "write streaming Max6 bitset '{}': {error}",
                            writer_path.display()
                        )
                    })?;
                    block.words = 0;
                    if free_tx.send(block).is_err() {
                        break;
                    }
                }
                writer.flush().map_err(|error| {
                    format!(
                        "flush streaming Max6 bitset '{}': {error}",
                        writer_path.display()
                    )
                })
            }) {
            Ok(handle) => handle,
            Err(error) => {
                let _ = fs::remove_file(&path);
                return Err(format!(
                    "spawn streaming Max6 bit writer '{}': {error}",
                    path.display()
                ));
            }
        };
        Ok(Self {
            row_words,
            block_words,
            words: 0,
            path: Some(path),
            tx: Some(tx),
            free_rx,
            handle: Some(handle),
        })
    }

    pub(super) fn append_rows(&mut self, rows: &[u64]) -> Result<(), String> {
        if rows.len() % self.row_words != 0 {
            return Err(format!(
                "streaming Max6 logic rows have {} words, expected a multiple of {}",
                rows.len(),
                self.row_words
            ));
        }
        if rows.is_empty() {
            return Ok(());
        }
        for chunk in rows.chunks(self.block_words) {
            let mut block = self.free_rx.recv().map_err(|error| {
                format!("streaming Max6 bit writer stopped before recycling a buffer: {error}")
            })?;
            block.words = chunk.len();
            block.data[..chunk.len()].copy_from_slice(chunk);
            let tx = self
                .tx
                .as_ref()
                .ok_or_else(|| "streaming Max6 bit writer is already closed".to_string())?;
            tx.send(block)
                .map_err(|error| format!("send streaming Max6 bit block to writer: {error}"))?;
            self.words = self
                .words
                .checked_add(chunk.len())
                .ok_or_else(|| "streaming Max6 logic word count overflow".to_string())?;
        }
        Ok(())
    }

    pub(super) fn finish(
        mut self,
    ) -> Result<
        (
            Vec<u64>,
            Option<Arc<memmap2::Mmap>>,
            usize,
            Arc<Max6BitsTempFileOwner>,
        ),
        String,
    > {
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| "streaming Max6 bit writer panicked".to_string())??;
        }
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| "streaming Max6 bitset lost its backing path".to_string())?;
        let file = File::open(&path)
            .map_err(|error| format!("open streaming Max6 bitset '{}': {error}", path.display()))?;
        let expected_bytes = self
            .words
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| "streaming Max6 bitset size overflow".to_string())?;
        let actual_bytes = file
            .metadata()
            .map_err(|error| format!("stat streaming Max6 bitset: {error}"))?
            .len() as usize;
        if actual_bytes != expected_bytes {
            return Err(format!(
                "streaming Max6 bitset length mismatch: wrote {actual_bytes} bytes, expected {expected_bytes}"
            ));
        }
        let mmap = unsafe {
            MmapOptions::new()
                .len(expected_bytes.max(1))
                .map(&file)
                .map_err(|error| format!("map streaming Max6 bitset: {error}"))?
        };
        #[cfg(unix)]
        let _ = mmap.advise(Advice::Random);
        let path = self
            .path
            .take()
            .ok_or_else(|| "streaming Max6 bitset lost its backing path".to_string())?;
        let owner = Arc::new(Max6BitsTempFileOwner { path });
        Ok((Vec::new(), Some(Arc::new(mmap)), self.words, owner))
    }
}

impl Drop for Max6StreamingLogicBitsAccumulator {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}

pub(super) struct Max6LogicBitsAccumulator {
    pub(super) row_words: usize,
    pub(super) words: usize,
    pub(super) flat: Vec<u64>,
    pub(super) mmap_file: Option<File>,
    pub(super) mmap_path: Option<std::path::PathBuf>,
    pub(super) mmap_mut: Option<MmapMut>,
    pub(super) mmap_capacity_words: usize,
    pub(super) mem_budget_bytes: u64,
}

impl Max6LogicBitsAccumulator {
    pub(super) fn new(row_words: usize, mem_budget_bytes: u64) -> Self {
        Self {
            row_words,
            words: 0,
            flat: Vec::new(),
            mmap_file: None,
            mmap_path: None,
            mmap_mut: None,
            mmap_capacity_words: 0,
            mem_budget_bytes,
        }
    }

    pub(super) fn append_rows(&mut self, rows: &[u64]) -> Result<(), String> {
        if self.row_words == 0 || rows.len() % self.row_words != 0 {
            return Err(format!(
                "Max6 fused logic rows have {} words, expected a multiple of {}",
                rows.len(),
                self.row_words
            ));
        }
        if rows.is_empty() {
            return Ok(());
        }
        let new_words = self
            .words
            .checked_add(rows.len())
            .ok_or_else(|| "Max6 fused logic word count overflow".to_string())?;
        let new_bytes = new_words.saturating_mul(std::mem::size_of::<u64>());
        if self.mmap_mut.is_none() && should_use_mapped_logic_bits(new_bytes, self.mem_budget_bytes)
        {
            self.start_mmap(new_words)?;
        } else if self.mmap_mut.is_some() && new_words > self.mmap_capacity_words {
            self.grow_mmap(new_words)?;
        }

        if let Some(mmap) = self.mmap_mut.as_mut() {
            let words = unsafe {
                std::slice::from_raw_parts_mut(
                    mmap.as_mut_ptr() as *mut u64,
                    self.mmap_capacity_words,
                )
            };
            words[self.words..new_words].copy_from_slice(rows);
        } else {
            self.flat.extend_from_slice(rows);
        }
        self.words = new_words;
        Ok(())
    }

    pub(super) fn start_mmap(&mut self, needed_words: usize) -> Result<(), String> {
        let capacity_words = needed_words.saturating_mul(2).max(needed_words).max(1);
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
            .map_err(|error| format!("create Max6 mapped bitset '{}': {error}", path.display()))?;
        let capacity_bytes = capacity_words.saturating_mul(std::mem::size_of::<u64>());
        if let Err(error) = file.set_len(capacity_bytes as u64) {
            let _ = fs::remove_file(&path);
            return Err(format!(
                "size Max6 mapped bitset '{}': {error}",
                path.display()
            ));
        }
        let mut mmap = match unsafe {
            MmapOptions::new()
                .len(capacity_bytes)
                .map_mut(&file)
                .map_err(|error| error.to_string())
        } {
            Ok(mmap) => mmap,
            Err(error) => {
                let _ = fs::remove_file(&path);
                return Err(format!("map Max6 bitset '{}': {error}", path.display()));
            }
        };
        if !self.flat.is_empty() {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(mmap.as_mut_ptr() as *mut u64, capacity_words)
            };
            dst[..self.flat.len()].copy_from_slice(self.flat.as_slice());
        }
        self.flat = Vec::new();
        self.mmap_file = Some(file);
        self.mmap_path = Some(path);
        self.mmap_mut = Some(mmap);
        self.mmap_capacity_words = capacity_words;
        Ok(())
    }

    pub(super) fn grow_mmap(&mut self, needed_words: usize) -> Result<(), String> {
        let capacity_words = needed_words
            .max(self.mmap_capacity_words.saturating_mul(2))
            .max(needed_words);
        self.mmap_mut.take();
        let file = self
            .mmap_file
            .as_ref()
            .ok_or_else(|| "Max6 mapped bitset lost its backing file".to_string())?;
        let capacity_bytes = capacity_words.saturating_mul(std::mem::size_of::<u64>());
        file.set_len(capacity_bytes as u64)
            .map_err(|error| format!("grow Max6 mapped bitset backing file: {error}"))?;
        let mmap = unsafe {
            MmapOptions::new()
                .len(capacity_bytes)
                .map_mut(file)
                .map_err(|error| format!("remap Max6 mapped bitset: {error}"))?
        };
        self.mmap_mut = Some(mmap);
        self.mmap_capacity_words = capacity_words;
        Ok(())
    }

    pub(super) fn finish(
        mut self,
    ) -> Result<(Vec<u64>, Option<Arc<memmap2::Mmap>>, usize), String> {
        if let Some(mmap_mut) = self.mmap_mut.take() {
            let mmap = mmap_mut
                .make_read_only()
                .map_err(|error| format!("finalize Max6 mapped bitset: {error}"))?;
            #[cfg(unix)]
            let _ = mmap.advise(Advice::Random);
            if let Some(path) = self.mmap_path.take() {
                let _ = fs::remove_file(path);
            }
            return Ok((Vec::new(), Some(Arc::new(mmap)), self.words));
        }
        Ok((std::mem::take(&mut self.flat), None, 0))
    }
}

impl Drop for Max6LogicBitsAccumulator {
    fn drop(&mut self) {
        if let Some(path) = self.mmap_path.take() {
            let _ = fs::remove_file(path);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_max6_global_qc_and_logic_bits(
    prefix: &str,
    sample_ids: &[String],
    sample_indices: &[usize],
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    threads: usize,
    logic_mem_budget_bytes: u64,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<(Vec<bool>, Vec<usize>, GarfieldLogicBits), String> {
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
            "Max6 fused BED scan requires matching nonempty sample IDs/indices; got {}/{}",
            sample_ids.len(),
            sample_indices.len()
        ));
    }

    let mut bed_iter = BedSnpIter::new_for_grm_window(prefix, garfield_logic_mmap_window_mb())?;
    let n_samples_total = bed_iter.n_samples();
    let source_marker_count = bed_iter.n_snps();
    if source_marker_count == 0 {
        return Err("no SNP sites found in PLINK BED input".to_string());
    }
    if sample_indices.iter().any(|&index| index >= n_samples_total) {
        return Err("selected sample index out of range for Max6 fused BED scan".to_string());
    }

    let n_samples = sample_indices.len();
    let row_words = words_for_samples(n_samples);
    let sample_identity =
        n_samples == n_samples_total && sample_indices_are_identity(sample_indices);
    let sample_plan = build_logic_sample_plan(sample_indices);
    let sites_all = if snps_only {
        let sites = read_bim(prefix)?;
        if sites.len() != source_marker_count {
            return Err(format!(
                "BED/BIM SNP count mismatch: bed={source_marker_count}, bim={}",
                sites.len()
            ));
        }
        Some(sites)
    } else {
        None
    };

    let prep_threads = effective_threads_local(threads).max(1);
    let pool = if prep_threads > 1 {
        crate::stats_common::get_cached_pool(prep_threads).map_err(|error| error.to_string())?
    } else {
        None
    };
    let mut site_keep = Vec::<bool>::with_capacity(source_marker_count);
    let mut row_flip_keep = Vec::<bool>::new();
    let mut row_source_indices = Vec::<usize>::new();
    let mut fail_missing = 0usize;
    let mut fail_het = 0usize;
    let mut fail_no_non_missing = 0usize;
    let mut fail_maf = 0usize;
    let mut fail_non_simple_snp = 0usize;
    let mut het_sites = 0usize;
    let mut raw_missing_sites = 0usize;
    let max_logic_batch_bytes = 32usize * 1024 * 1024;
    let max_rows_by_logic_bytes = max_logic_batch_bytes
        .checked_div(row_words.saturating_mul(std::mem::size_of::<u64>()).max(1))
        .unwrap_or(1)
        .max(1);
    let max_batch_rows = max_rows_by_logic_bytes.min(1_000_000).max(1);
    let mut bits_accumulator = Max6LogicBitsAccumulator::new(row_words, logic_mem_budget_bytes);

    while bed_iter.cursor() < source_marker_count {
        check_ctrlc()?;
        let base = bed_iter.cursor();
        bed_iter.ensure_window_for_snp(base)?;
        let mapped_rows = bed_iter.mapped_contiguous_snps_from(base);
        if mapped_rows == 0 {
            return Err("Max6 fused genotype scan reached an empty input window".to_string());
        }
        let scan_rows = mapped_rows.min(max_batch_rows);
        let batch_words = scan_rows
            .checked_mul(row_words)
            .ok_or_else(|| "Max6 fused logic batch size overflow".to_string())?;
        let mut qc_batch = vec![Max6FusedRowQc::default(); scan_rows];
        let mut row_bits = vec![0u64; batch_words];
        let sites_all_ref = sites_all.as_ref();
        let process_row =
            |source_row: usize, row: &[u8], dst_words: &mut [u64], missing_words: &mut [u64]| {
                let simple_snp_alleles = sites_all_ref
                    .map(|sites| {
                        let site = &sites[source_row];
                        is_simple_snp_allele(&site.ref_allele)
                            && is_simple_snp_allele(&site.alt_allele)
                    })
                    .unwrap_or(true);
                scan_max6_row_qc_and_pack_logic(
                    row,
                    n_samples,
                    sample_identity,
                    sample_plan.as_slice(),
                    maf_threshold,
                    max_missing_rate,
                    simple_snp_alleles,
                    dst_words,
                    missing_words,
                )
            };

        let it_ref: &BedSnpIter = &bed_iter;
        let mut run_rows = || {
            row_bits
                .par_chunks_mut(row_words)
                .zip(qc_batch.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || vec![0u64; row_words],
                    |missing_words, (off, (dst_words, qc_dst))| {
                        let source_row = base + off;
                        let row = it_ref
                            .packed_snp_bytes_at(source_row)
                            .expect("Max6 fused packed row should be mapped");
                        *qc_dst =
                            process_row(source_row, &row, dst_words, missing_words.as_mut_slice());
                    },
                );
        };
        if let Some(pool) = pool.as_ref() {
            pool.install(&mut run_rows);
        } else {
            run_rows();
        }
        check_ctrlc()?;

        let mut kept_rows = 0usize;
        for off in 0..scan_rows {
            let qc = qc_batch[off];
            let keep = qc.keep;
            site_keep.push(keep);
            if keep {
                row_flip_keep.push(qc.row_flip);
                row_source_indices.push(base + off);
                kept_rows += 1;
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
        if kept_rows > 0 {
            let mut write_row = 0usize;
            for off in 0..scan_rows {
                if qc_batch[off].keep {
                    if write_row != off {
                        let src_start = off * row_words;
                        let dst_start = write_row * row_words;
                        row_bits.copy_within(src_start..src_start + row_words, dst_start);
                    }
                    write_row += 1;
                }
            }
            row_bits.truncate(kept_rows * row_words);
            bits_accumulator.append_rows(row_bits.as_slice())?;
        }
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        bed_iter.set_cursor(base + scan_rows);
    }

    if site_keep.len() != source_marker_count {
        return Err(format!(
            "internal error: Max6 fused QC rows {} != source markers {source_marker_count}",
            site_keep.len()
        ));
    }
    let kept_n = row_source_indices.len();
    if kept_n == 0 {
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

    let sites_keep = if let Some(sites_all) = sites_all {
        sites_all
            .into_iter()
            .enumerate()
            .filter_map(|(source_row, site)| site_keep[source_row].then_some(site))
            .collect::<Vec<_>>()
    } else {
        let mut sites_keep = Vec::<SiteInfo>::with_capacity(kept_n);
        let mut bim = BimChunkReader::open(prefix)?;
        let chunk_rows = 131_072usize;
        let mut base = 0usize;
        while base < source_marker_count {
            let end = (base + chunk_rows).min(source_marker_count);
            sites_keep.extend(bim.read_range_masked(base, end, &site_keep[base..end])?);
            base = end;
        }
        bim.ensure_exhausted(source_marker_count)?;
        sites_keep
    };
    if sites_keep.len() != kept_n || row_flip_keep.len() != kept_n {
        return Err(format!(
            "Max6 fused metadata mismatch: sites={}, flips={}, QC rows={kept_n}",
            sites_keep.len(),
            row_flip_keep.len()
        ));
    }

    let (bits_flat, bits_mmap, bits_mmap_words) = bits_accumulator.finish()?;
    let sites = build_logic_sites_from_metadata_owned_direct(
        sites_keep,
        row_flip_keep.as_slice(),
        GarfieldBinMode::Bin,
    );
    let group_ids = build_logic_mode_group_ids(kept_n, GarfieldBinMode::Bin);
    let logic_bits = GarfieldLogicBits {
        bits_flat,
        bits_mmap,
        bits_mmap_words,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words,
        sample_ids: sample_ids.to_vec(),
        sites,
        group_ids,
        n_samples,
    };
    Ok((site_keep, row_source_indices, logic_bits))
}
