//! Native, streaming reader for the biallelic hard-call subset of PLINK 2 PGEN.
//!
//! PGEN is decoded directly into JanusX's BED-compatible two-bit row layout:
//! `00=REF/REF`, `10=REF/ALT`, `11=ALT/ALT`, `01=missing`. Dosage and phase
//! auxiliary tracks are intentionally ignored to match BED hard-call semantics.

use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const PGEN_BLOCK_VARIANTS: usize = 1 << 16;
const PGEN_PARALLEL_WORK_THRESHOLD: usize = 1 << 20;
const PGEN_PARALLEL_ANCHORS_PER_TASK: usize = 128;

#[derive(Clone, Debug)]
pub(crate) struct PgenSite {
    pub chrom: String,
    pub pos: i32,
    pub id: String,
    pub ref_allele: String,
    pub alt_allele: String,
}

/// Per-variant hard-call counts, shared by packed genotype readers and their
/// GWAS/GS/GARFIELD consumers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GenotypeRowStats {
    pub sample_count: usize,
    pub missing_count: usize,
    pub heterozygote_count: usize,
    pub homozygous_alt_count: usize,
}

impl GenotypeRowStats {
    #[inline]
    pub(crate) fn missing_rate(self) -> Option<f32> {
        (self.sample_count > 0).then(|| self.missing_count as f32 / self.sample_count as f32)
    }

    #[inline]
    pub(crate) fn minor_allele_frequency(self) -> Option<f32> {
        self.alternate_allele_frequency()
            .map(|alt_frequency| alt_frequency.min(1.0_f32 - alt_frequency))
    }

    #[inline]
    pub(crate) fn alternate_allele_frequency(self) -> Option<f32> {
        let called = self.sample_count.saturating_sub(self.missing_count);
        if called == 0 {
            return None;
        }
        let alt_count = self
            .heterozygote_count
            .saturating_add(self.homozygous_alt_count.saturating_mul(2));
        Some(alt_count as f32 / (2.0_f32 * called as f32))
    }

    #[inline]
    pub(crate) fn heterozygosity_rate(self) -> Option<f32> {
        let called = self.sample_count.saturating_sub(self.missing_count);
        (called > 0).then(|| self.heterozygote_count as f32 / called as f32)
    }
}

/// Selects the sample set used for row-level QC. Exclusion is useful when a
/// caller keeps most samples: the packed row can be counted once and only the
/// small excluded set needs per-sample adjustments.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GenotypeSampleSelection<'a> {
    All,
    Include(&'a [usize]),
    Exclude(&'a [usize]),
}

impl<'a> GenotypeSampleSelection<'a> {
    #[inline]
    pub(crate) const fn all() -> Self {
        Self::All
    }

    #[inline]
    pub(crate) const fn include(indices: &'a [usize]) -> Self {
        Self::Include(indices)
    }

    /// Pick the cheaper representation for an already validated selected set
    /// and its complement.
    #[inline]
    pub(crate) fn from_selected_and_excluded(selected: &'a [usize], excluded: &'a [usize]) -> Self {
        if excluded.len() < selected.len() {
            Self::Exclude(excluded)
        } else {
            Self::Include(selected)
        }
    }

    pub(crate) fn validate(self, n_samples: usize) -> Result<(), String> {
        let indices = match self {
            Self::All => return Ok(()),
            Self::Include(indices) | Self::Exclude(indices) => indices,
        };
        if let Some(&bad_index) = indices.iter().find(|&&idx| idx >= n_samples) {
            return Err(format!(
                "QC sample index {bad_index} exceeds sample count {n_samples}"
            ));
        }
        // This is checked once per block, not once per variant. Duplicate
        // indices would otherwise double-count an included sample or subtract
        // an excluded call more than once.
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|window| window[0] == window[1]) {
            return Err("QC sample selection contains duplicate indices".to_string());
        }
        Ok(())
    }
}

pub(crate) trait PgenDosageValue: Copy + Send + Sync {
    fn from_raw_genotype_code(code: u8) -> Self;
}

impl PgenDosageValue for f32 {
    #[inline]
    fn from_raw_genotype_code(code: u8) -> Self {
        match code {
            0 => 0.0,
            1 => 1.0,
            2 => 2.0,
            _ => -9.0,
        }
    }
}

impl PgenDosageValue for f64 {
    #[inline]
    fn from_raw_genotype_code(code: u8) -> Self {
        match code {
            0 => 0.0,
            1 => 1.0,
            2 => 2.0,
            _ => -9.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum StorageMode {
    BedCompat,
    FixedHardcall,
    Variable,
}

#[derive(Default)]
struct BlockMeta {
    record_types: Vec<u8>,
    record_lengths: Vec<usize>,
    record_offsets: Vec<usize>,
    anchor_rows: Vec<usize>,
}

/// PGEN decoder with a compact in-memory index, the active variable-record
/// block, and reusable genotype scratch rows.
pub(crate) struct PgenReader {
    path: PathBuf,
    file: BufReader<File>,
    file_size: u64,
    mode: StorageMode,
    n_samples: usize,
    n_variants: usize,
    bytes_per_variant: usize,
    variant_block_offsets: Vec<u64>,
    block_meta_offsets: Vec<u64>,
    record_type_bits: usize,
    record_length_bytes: usize,
    fixed_data_offset: u64,
    current_block: Option<usize>,
    block_meta: BlockMeta,
    block_records: Vec<u8>,
    block_record_cursor: usize,
    next_variant: usize,
    record_offset: u64,
    base_categories: Vec<u8>,
    current_categories: Vec<u8>,
    record_buf: Vec<u8>,
    scratch_packed: Vec<u8>,
}

impl PgenReader {
    pub(crate) fn open(
        path: &Path,
        n_samples: usize,
        n_variants_from_pvar: usize,
    ) -> Result<Self, String> {
        if n_samples == 0 || n_variants_from_pvar == 0 {
            return Err("PGEN input contains zero samples or variants".to_string());
        }
        let raw_file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let file_size = raw_file
            .metadata()
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        // PGEN's variable records are individually sized. A BufReader avoids
        // issuing one operating-system read for every variant during a scan.
        let mut file = BufReader::with_capacity(1 << 20, raw_file);
        let mut magic = [0u8; 3];
        file.read_exact(&mut magic)
            .map_err(|e| format!("read PGEN header {}: {e}", path.display()))?;
        if magic[0..2] != [0x6c, 0x1b] {
            return Err(format!("{}: invalid PGEN magic bytes", path.display()));
        }
        let bytes_per_variant = n_samples.div_ceil(4);
        let mut variant_block_offsets = Vec::new();
        let mut block_meta_offsets = Vec::new();
        let mut record_type_bits = 0usize;
        let mut record_length_bytes = 0usize;
        let mut fixed_data_offset = 0u64;
        let mode = match magic[2] {
            0x01 => {
                let expected = 3u64
                    .checked_add(
                        (bytes_per_variant as u64).saturating_mul(n_variants_from_pvar as u64),
                    )
                    .ok_or_else(|| format!("{}: PGEN size overflow", path.display()))?;
                if file_size != expected {
                    return Err(format!(
                        "{}: BED-compatible PGEN size mismatch; expected {expected}, got {file_size}",
                        path.display()
                    ));
                }
                fixed_data_offset = 3;
                StorageMode::BedCompat
            }
            0x02 => {
                let (n_variants, n_samples_header, header_format) =
                    read_standard_dimensions(&mut file, path)?;
                validate_dimensions(
                    path,
                    n_samples_header,
                    n_variants,
                    n_samples,
                    n_variants_from_pvar,
                )?;
                if header_format & 0x0f != 0 {
                    return Err(format!(
                        "{}: fixed-width PGEN has a nonzero record-index format",
                        path.display()
                    ));
                }
                let allele_count_mode = (header_format >> 4) & 0x03;
                if allele_count_mode != 0 {
                    return Err(format!(
                        "{}: fixed-width PGEN with allele-count metadata is outside the biallelic hard-call reader",
                        path.display()
                    ));
                }
                let provisional_ref_mode = header_format >> 6;
                if provisional_ref_mode == 3 {
                    fixed_data_offset = 12 + (n_variants.div_ceil(8) as u64);
                } else {
                    fixed_data_offset = 12;
                }
                if fixed_data_offset > file_size {
                    return Err(format!(
                        "{}: truncated fixed-width PGEN header",
                        path.display()
                    ));
                }
                let expected_size = fixed_data_offset
                    .checked_add((bytes_per_variant as u64).saturating_mul(n_variants as u64))
                    .ok_or_else(|| format!("{}: fixed-width PGEN size overflow", path.display()))?;
                if expected_size != file_size {
                    return Err(format!(
                        "{}: fixed-width PGEN size mismatch; expected {expected_size}, got {file_size}",
                        path.display()
                    ));
                }
                StorageMode::FixedHardcall
            }
            0x10 => {
                let (n_variants, n_samples_header, header_format) =
                    read_standard_dimensions(&mut file, path)?;
                validate_dimensions(
                    path,
                    n_samples_header,
                    n_variants,
                    n_samples,
                    n_variants_from_pvar,
                )?;
                let format = (header_format & 0x0f) as usize;
                if format > 7 {
                    return Err(format!(
                        "{}: reserved PGEN record-index format {format}",
                        path.display()
                    ));
                }
                record_type_bits = if format < 4 { 4 } else { 8 };
                record_length_bytes = (format & 3) + 1;
                let allele_count_mode = (header_format >> 4) & 0x03;
                if allele_count_mode != 0 {
                    return Err(format!(
                        "{}: PGEN stores per-variant allele counts; multi-allelic input is not supported",
                        path.display()
                    ));
                }
                let provisional_ref_mode = header_format >> 6;
                let block_count = n_variants.div_ceil(PGEN_BLOCK_VARIANTS);
                let offsets_end = 12u64
                    .checked_add((block_count as u64).saturating_mul(8))
                    .ok_or_else(|| format!("{}: PGEN header size overflow", path.display()))?;
                if offsets_end > file_size {
                    return Err(format!(
                        "{}: truncated PGEN variant-block index",
                        path.display()
                    ));
                }
                file.seek(SeekFrom::Start(12))
                    .map_err(|e| format!("seek {} header: {e}", path.display()))?;
                variant_block_offsets.reserve(block_count);
                for _ in 0..block_count {
                    let mut bytes = [0u8; 8];
                    file.read_exact(&mut bytes)
                        .map_err(|e| format!("read {} block offsets: {e}", path.display()))?;
                    variant_block_offsets.push(u64::from_le_bytes(bytes));
                }
                let mut meta_pos = offsets_end;
                block_meta_offsets.reserve(block_count);
                for block_idx in 0..block_count {
                    let rows =
                        (n_variants - block_idx * PGEN_BLOCK_VARIANTS).min(PGEN_BLOCK_VARIANTS);
                    block_meta_offsets.push(meta_pos);
                    let type_bytes = rows.saturating_mul(record_type_bits).div_ceil(8);
                    let length_bytes = rows.checked_mul(record_length_bytes).ok_or_else(|| {
                        format!("{}: PGEN record-index size overflow", path.display())
                    })?;
                    let allele_bytes = 0usize;
                    let ref_bytes = if provisional_ref_mode == 3 {
                        rows.div_ceil(8)
                    } else {
                        0
                    };
                    meta_pos = meta_pos
                        .checked_add(type_bytes as u64)
                        .and_then(|v| v.checked_add(length_bytes as u64))
                        .and_then(|v| v.checked_add(allele_bytes as u64))
                        .and_then(|v| v.checked_add(ref_bytes as u64))
                        .ok_or_else(|| {
                            format!("{}: PGEN header offset overflow", path.display())
                        })?;
                }
                if block_meta_offsets.first().copied().unwrap_or(0) >= file_size
                    || variant_block_offsets
                        .iter()
                        .any(|&offset| offset > file_size)
                {
                    return Err(format!(
                        "{}: PGEN header index points outside the file",
                        path.display()
                    ));
                }
                if meta_pos > variant_block_offsets[0]
                    || variant_block_offsets
                        .windows(2)
                        .any(|pair| pair[0] > pair[1])
                {
                    return Err(format!(
                        "{}: PGEN variant block index overlaps the header or is not increasing",
                        path.display()
                    ));
                }
                StorageMode::Variable
            }
            other => {
                let description = match other {
                    0x03 | 0x04 => "fixed-width dosage-only storage",
                    0x11 => "PGEN with optional extensions",
                    0x20 | 0x21 => "PGEN using a separate index file",
                    _ => "unknown or unsupported storage mode",
                };
                return Err(format!(
                    "{}: {description} (storage mode 0x{other:02x}) is not supported by the biallelic hard-call reader",
                    path.display()
                ));
            }
        };
        let mut out = Self {
            path: path.to_path_buf(),
            file,
            file_size,
            mode,
            n_samples,
            n_variants: n_variants_from_pvar,
            bytes_per_variant,
            variant_block_offsets,
            block_meta_offsets,
            record_type_bits,
            record_length_bytes,
            fixed_data_offset,
            current_block: None,
            block_meta: BlockMeta::default(),
            block_records: Vec::new(),
            block_record_cursor: 0,
            next_variant: 0,
            record_offset: fixed_data_offset,
            base_categories: vec![0u8; bytes_per_variant],
            current_categories: vec![0u8; bytes_per_variant],
            record_buf: Vec::new(),
            scratch_packed: vec![0u8; bytes_per_variant],
        };
        match out.mode {
            StorageMode::BedCompat => {}
            StorageMode::FixedHardcall => {
                out.file
                    .seek(SeekFrom::Start(out.fixed_data_offset))
                    .map_err(|e| format!("seek {} genotype data: {e}", out.path.display()))?;
            }
            StorageMode::Variable => out.load_block(0)?,
        }
        Ok(out)
    }

    pub(crate) fn read_source_block(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [u8],
    ) -> Result<usize, String> {
        if source_start > self.n_variants {
            return Err(format!(
                "PGEN source row {source_start} exceeds variant count {}",
                self.n_variants
            ));
        }
        let rows = requested_rows.min(self.n_variants - source_start);
        let byte_len = rows
            .checked_mul(self.bytes_per_variant)
            .ok_or_else(|| "PGEN output block size overflow".to_string())?;
        if out.len() < byte_len {
            return Err(format!(
                "packed PGEN destination is too small: got {}, need {byte_len}",
                out.len()
            ));
        }
        if rows == 0 {
            return Ok(0);
        }
        if matches!(self.mode, StorageMode::Variable) {
            self.read_variable_rows(source_start, rows, &mut out[..byte_len])?;
            self.next_variant = source_start + rows;
            return Ok(rows);
        }
        if source_start != self.next_variant {
            self.reset_to(source_start)?;
        }
        for row in 0..rows {
            let range = row * self.bytes_per_variant..(row + 1) * self.bytes_per_variant;
            self.read_variant_into(&mut out[range])?;
        }
        Ok(rows)
    }

    /// Decode hard calls directly into row-major f32 dosages. Missing values
    /// use the same `-9.0` sentinel as the genotype streaming iterators.
    pub(crate) fn read_source_block_f32(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f32],
    ) -> Result<usize, String> {
        self.read_source_block_values(source_start, requested_rows, out)
    }

    /// f64 counterpart to `read_source_block_f32`, for consumers whose
    /// downstream association calculations operate in double precision.
    #[allow(dead_code)] // Shared native streaming API; current Python scans use f32.
    pub(crate) fn read_source_block_f64(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f64],
    ) -> Result<usize, String> {
        self.read_source_block_values(source_start, requested_rows, out)
    }

    fn read_source_block_values<T: PgenDosageValue>(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [T],
    ) -> Result<usize, String> {
        if source_start > self.n_variants {
            return Err(format!(
                "PGEN source row {source_start} exceeds variant count {}",
                self.n_variants
            ));
        }
        let rows = requested_rows.min(self.n_variants - source_start);
        let value_len = rows
            .checked_mul(self.n_samples)
            .ok_or_else(|| "PGEN dosage block size overflow".to_string())?;
        if out.len() < value_len {
            return Err(format!(
                "PGEN dosage destination is too small: got {}, need {value_len}",
                out.len()
            ));
        }
        if rows == 0 {
            return Ok(0);
        }
        if matches!(self.mode, StorageMode::Variable) {
            self.read_variable_rows_to_values(source_start, rows, &mut out[..value_len])?;
            self.next_variant = source_start + rows;
            return Ok(rows);
        }

        let packed_len = rows
            .checked_mul(self.bytes_per_variant)
            .ok_or_else(|| "PGEN packed dosage scratch size overflow".to_string())?;
        let mut packed = vec![0u8; packed_len];
        self.read_source_block(source_start, rows, &mut packed)?;
        for (row_idx, packed_row) in packed.chunks_exact(self.bytes_per_variant).enumerate() {
            decode_bed_row_to_values(
                packed_row,
                self.n_samples,
                &mut out[row_idx * self.n_samples..(row_idx + 1) * self.n_samples],
            );
        }
        Ok(rows)
    }

    /// Decodes a source block, computes selected-sample QC counts in the same
    /// PGEN pass, and converts only rows accepted by `keep_variant` to BED
    /// packed codes. Rejected packed rows are zeroed; callers must use
    /// `keep_out` before consuming a row.
    pub(crate) fn read_source_block_with_qc<F>(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        sample_selection: GenotypeSampleSelection<'_>,
        out_packed: &mut [u8],
        stats_out: &mut [GenotypeRowStats],
        keep_out: &mut [bool],
        keep_variant: &F,
    ) -> Result<usize, String>
    where
        F: Fn(usize, GenotypeRowStats) -> bool + Sync,
    {
        self.read_source_block_qc_impl(
            source_start,
            requested_rows,
            sample_selection,
            Some(out_packed),
            stats_out,
            keep_out,
            keep_variant,
        )
    }

    /// Computes QC and the keep mask without materializing BED-packed rows.
    /// Use this for metadata scans whose downstream consumer only needs counts.
    pub(crate) fn read_source_block_qc_only<F>(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        sample_selection: GenotypeSampleSelection<'_>,
        stats_out: &mut [GenotypeRowStats],
        keep_out: &mut [bool],
        keep_variant: &F,
    ) -> Result<usize, String>
    where
        F: Fn(usize, GenotypeRowStats) -> bool + Sync,
    {
        self.read_source_block_qc_impl(
            source_start,
            requested_rows,
            sample_selection,
            None,
            stats_out,
            keep_out,
            keep_variant,
        )
    }

    fn read_source_block_qc_impl<F>(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        sample_selection: GenotypeSampleSelection<'_>,
        mut out_packed: Option<&mut [u8]>,
        stats_out: &mut [GenotypeRowStats],
        keep_out: &mut [bool],
        keep_variant: &F,
    ) -> Result<usize, String>
    where
        F: Fn(usize, GenotypeRowStats) -> bool + Sync,
    {
        if source_start > self.n_variants {
            return Err(format!(
                "PGEN source row {source_start} exceeds variant count {}",
                self.n_variants
            ));
        }
        sample_selection.validate(self.n_samples)?;
        let rows = requested_rows.min(self.n_variants - source_start);
        let byte_len = rows
            .checked_mul(self.bytes_per_variant)
            .ok_or_else(|| "PGEN QC packed block size overflow".to_string())?;
        if let Some(out_packed) = out_packed.as_deref() {
            if out_packed.len() < byte_len {
                return Err(format!(
                    "packed PGEN QC destination is too small: got {}, need {byte_len}",
                    out_packed.len()
                ));
            }
        }
        if stats_out.len() < rows || keep_out.len() < rows {
            return Err(format!(
                "PGEN QC metadata destination too small: stats={}, keep={}, need {rows}",
                stats_out.len(),
                keep_out.len()
            ));
        }
        if rows == 0 {
            return Ok(0);
        }
        if matches!(self.mode, StorageMode::Variable) {
            let packed_out = out_packed.as_deref_mut().map(|out| &mut out[..byte_len]);
            self.read_variable_rows_with_qc(
                source_start,
                rows,
                sample_selection,
                keep_variant,
                packed_out,
                &mut stats_out[..rows],
                &mut keep_out[..rows],
            )?;
            self.next_variant = source_start + rows;
            return Ok(rows);
        }

        if let Some(out_packed) = out_packed {
            self.read_source_block(source_start, rows, &mut out_packed[..byte_len])?;
            for row_idx in 0..rows {
                let row_start = row_idx * self.bytes_per_variant;
                let row_end = row_start + self.bytes_per_variant;
                let stats = count_bed_genotype_row(
                    &out_packed[row_start..row_end],
                    self.n_samples,
                    sample_selection,
                );
                let keep = keep_variant(source_start + row_idx, stats);
                stats_out[row_idx] = stats;
                keep_out[row_idx] = keep;
                if !keep {
                    out_packed[row_start..row_end].fill(0);
                }
            }
        } else {
            if source_start != self.next_variant {
                self.reset_to(source_start)?;
            }
            let mut scratch = vec![0u8; self.bytes_per_variant];
            for row_idx in 0..rows {
                self.read_variant_into(&mut scratch)?;
                let stats = count_bed_genotype_row(&scratch, self.n_samples, sample_selection);
                stats_out[row_idx] = stats;
                keep_out[row_idx] = keep_variant(source_start + row_idx, stats);
            }
        }
        Ok(rows)
    }

    fn reset_to(&mut self, target: usize) -> Result<(), String> {
        self.next_variant = 0;
        self.record_offset = self.fixed_data_offset;
        self.current_block = None;
        self.base_categories.fill(0xff);
        self.current_categories.fill(0xff);
        if matches!(self.mode, StorageMode::Variable) {
            let block_idx = target / PGEN_BLOCK_VARIANTS;
            self.load_block(block_idx)?;
            self.next_variant = block_idx * PGEN_BLOCK_VARIANTS;
        } else {
            self.file
                .seek(SeekFrom::Start(self.fixed_data_offset))
                .map_err(|e| format!("seek {}: {e}", self.path.display()))?;
        }
        while self.next_variant < target {
            let mut discard = std::mem::take(&mut self.scratch_packed);
            self.read_variant_into(&mut discard)?;
            self.scratch_packed = discard;
        }
        Ok(())
    }

    fn load_block(&mut self, block_idx: usize) -> Result<(), String> {
        if self.current_block == Some(block_idx) {
            return Ok(());
        }
        let block_start = block_idx.saturating_mul(PGEN_BLOCK_VARIANTS);
        if block_start >= self.n_variants {
            return Err(format!("PGEN block index out of range: {block_idx}"));
        }
        let rows = (self.n_variants - block_start).min(PGEN_BLOCK_VARIANTS);
        let block_data_offset = *self.variant_block_offsets.get(block_idx).ok_or_else(|| {
            format!(
                "{}: missing variant block offset {block_idx}",
                self.path.display()
            )
        })?;
        let next_block_offset = self
            .variant_block_offsets
            .get(block_idx + 1)
            .copied()
            .unwrap_or(self.file_size);
        if block_data_offset > next_block_offset || next_block_offset > self.file_size {
            return Err(format!(
                "{}: invalid PGEN block offsets",
                self.path.display()
            ));
        }
        let meta_start = *self.block_meta_offsets.get(block_idx).ok_or_else(|| {
            format!(
                "{}: missing block metadata offset {block_idx}",
                self.path.display()
            )
        })?;
        let type_bytes_len = rows.saturating_mul(self.record_type_bits).div_ceil(8);
        let length_bytes_len = rows
            .checked_mul(self.record_length_bytes)
            .ok_or_else(|| "PGEN block length table size overflow".to_string())?;
        let mut header_reader = self
            .file
            .get_ref()
            .try_clone()
            .map_err(|e| format!("clone {} header reader: {e}", self.path.display()))?;
        header_reader
            .seek(SeekFrom::Start(meta_start))
            .map_err(|e| format!("seek {} block metadata: {e}", self.path.display()))?;
        let mut packed_types = vec![0u8; type_bytes_len];
        header_reader
            .read_exact(&mut packed_types)
            .map_err(|e| format!("read {} record types: {e}", self.path.display()))?;
        let mut packed_lengths = vec![0u8; length_bytes_len];
        header_reader
            .read_exact(&mut packed_lengths)
            .map_err(|e| format!("read {} record lengths: {e}", self.path.display()))?;
        let mut record_types = Vec::with_capacity(rows);
        let mut record_lengths = Vec::with_capacity(rows);
        for idx in 0..rows {
            let record_type = if self.record_type_bits == 4 {
                let byte = packed_types[idx >> 1];
                if idx & 1 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                }
            } else {
                packed_types[idx]
            };
            record_types.push(record_type);
            let start = idx * self.record_length_bytes;
            let mut length = 0usize;
            for (byte_idx, &byte) in packed_lengths[start..start + self.record_length_bytes]
                .iter()
                .enumerate()
            {
                length |= (byte as usize) << (byte_idx * 8);
            }
            let record_type = record_types[idx];
            if length == 0 && record_type & 0x07 != 5 {
                return Err(format!(
                    "{}: variant {} has an empty PGEN record",
                    self.path.display(),
                    block_start + idx
                ));
            }
            record_lengths.push(length);
        }
        if record_types
            .first()
            .is_some_and(|record_type| matches!(record_type & 0x07, 2 | 3))
        {
            return Err(format!(
                "{}: first variant in PGEN block {} uses LD compression",
                self.path.display(),
                block_idx
            ));
        }
        let total_record_bytes = record_lengths
            .iter()
            .try_fold(0u64, |acc, &len| acc.checked_add(len as u64))
            .ok_or_else(|| "PGEN variant-block record-size overflow".to_string())?;
        if block_data_offset
            .checked_add(total_record_bytes)
            .unwrap_or(u64::MAX)
            > next_block_offset
        {
            return Err(format!(
                "{}: PGEN record lengths exceed their block boundary",
                self.path.display()
            ));
        }
        let block_bytes = usize::try_from(total_record_bytes).map_err(|_| {
            format!(
                "{}: PGEN variant block is too large for this platform",
                self.path.display()
            )
        })?;
        self.file
            .seek(SeekFrom::Start(block_data_offset))
            .map_err(|e| format!("seek {} variant block: {e}", self.path.display()))?;
        self.block_records.resize(block_bytes, 0);
        self.file
            .read_exact(&mut self.block_records)
            .map_err(|e| format!("read {} variant block payload: {e}", self.path.display()))?;
        self.block_record_cursor = 0;
        self.record_offset = block_data_offset;
        let mut record_offsets = Vec::with_capacity(rows + 1);
        let mut record_cursor = 0usize;
        let mut anchor_rows = Vec::with_capacity(rows);
        for (row, (&record_type, &record_len)) in
            record_types.iter().zip(&record_lengths).enumerate()
        {
            record_offsets.push(record_cursor);
            record_cursor = record_cursor
                .checked_add(record_len)
                .ok_or_else(|| "PGEN block record offsets overflow".to_string())?;
            if !matches!(record_type & 0x07, 2 | 3) {
                anchor_rows.push(row);
            }
        }
        record_offsets.push(record_cursor);
        if record_cursor != block_bytes {
            return Err(format!(
                "{}: PGEN block record lengths do not match loaded payload",
                self.path.display()
            ));
        }
        self.block_meta = BlockMeta {
            record_types,
            record_lengths,
            record_offsets,
            anchor_rows,
        };
        self.current_block = Some(block_idx);
        Ok(())
    }

    fn read_variable_rows(
        &mut self,
        source_start: usize,
        rows: usize,
        out: &mut [u8],
    ) -> Result<(), String> {
        let mut rows_read = 0usize;
        while rows_read < rows {
            let source_row = source_start + rows_read;
            let block_idx = source_row / PGEN_BLOCK_VARIANTS;
            self.load_block(block_idx)?;
            let block_start = block_idx * PGEN_BLOCK_VARIANTS;
            let local_start = source_row - block_start;
            let block_rows = self.block_meta.record_types.len();
            let local_rows = (rows - rows_read).min(block_rows - local_start);
            let output_start = rows_read * self.bytes_per_variant;
            let output_end = output_start + local_rows * self.bytes_per_variant;
            self.read_variable_block_range(
                local_start,
                local_rows,
                &mut out[output_start..output_end],
            )?;
            rows_read += local_rows;
        }
        Ok(())
    }

    fn read_variable_block_range(
        &self,
        local_start: usize,
        rows: usize,
        out: &mut [u8],
    ) -> Result<(), String> {
        let local_end = local_start + rows;
        let block_rows = self.block_meta.record_types.len();
        if local_end > block_rows || out.len() != rows * self.bytes_per_variant {
            return Err("invalid PGEN block output range".to_string());
        }

        #[derive(Clone, Copy)]
        struct DecodeTask {
            variant_start: usize,
            variant_end: usize,
            output_start: usize,
        }

        let anchors = &self.block_meta.anchor_rows;
        let first_anchor = anchors
            .partition_point(|&anchor| anchor <= local_start)
            .saturating_sub(1);
        let mut tasks = Vec::new();
        let mut anchor_idx = first_anchor;
        while anchor_idx < anchors.len() && anchors[anchor_idx] < local_end {
            let task_anchor_end = (anchor_idx + PGEN_PARALLEL_ANCHORS_PER_TASK).min(anchors.len());
            let variant_start = anchors[anchor_idx];
            let variant_end = if task_anchor_end < anchors.len() {
                anchors[task_anchor_end]
            } else {
                block_rows
            }
            .min(local_end);
            tasks.push(DecodeTask {
                variant_start,
                variant_end,
                output_start: variant_start.max(local_start),
            });
            if variant_end == local_end {
                break;
            }
            anchor_idx = task_anchor_end;
        }
        if tasks.is_empty() {
            return Err("PGEN block range has no preceding anchor variant".to_string());
        }

        let decode_task_into = |task: &DecodeTask, task_output: &mut [u8]| -> Result<(), String> {
            let output_rows = task.variant_end - task.output_start;
            let expected_len = output_rows
                .checked_mul(self.bytes_per_variant)
                .ok_or_else(|| "PGEN task output size overflow".to_string())?;
            if task_output.len() != expected_len {
                return Err("invalid PGEN task output slice".to_string());
            }
            let block_start = self
                .current_block
                .unwrap_or_default()
                .saturating_mul(PGEN_BLOCK_VARIANTS);
            let mut current = vec![0u8; self.bytes_per_variant];
            let mut base = vec![0u8; self.bytes_per_variant];
            for variant in task.variant_start..task.variant_end {
                let record_type = self.block_meta.record_types[variant];
                let record_start = self.block_meta.record_offsets[variant];
                let record_end = self.block_meta.record_offsets[variant + 1];
                decode_packed_record(
                    record_type,
                    &self.block_records[record_start..record_end],
                    self.n_samples,
                    block_start + variant,
                    &self.path,
                    &mut current,
                    &mut base,
                )?;
                if variant >= task.output_start {
                    let row_offset = (variant - task.output_start) * self.bytes_per_variant;
                    for (dst, &category_byte) in task_output
                        [row_offset..row_offset + self.bytes_per_variant]
                        .iter_mut()
                        .zip(current.iter())
                    {
                        *dst = pgen_category_byte_to_row_byte(category_byte);
                    }
                }
            }
            Ok(())
        };

        let expected_output_len = tasks.iter().try_fold(0usize, |total, task| {
            (task.variant_end - task.output_start)
                .checked_mul(self.bytes_per_variant)
                .and_then(|task_len| total.checked_add(task_len))
        });
        if expected_output_len != Some(out.len()) {
            return Err(format!(
                "PGEN task ranges cover {} bytes, output has {}",
                expected_output_len.unwrap_or(usize::MAX),
                out.len()
            ));
        }
        let mut task_outputs = Vec::with_capacity(tasks.len());
        let mut output_remainder = out;
        for task in &tasks {
            let task_len = (task.variant_end - task.output_start) * self.bytes_per_variant;
            let (task_output, remainder) = output_remainder.split_at_mut(task_len);
            task_outputs.push(task_output);
            output_remainder = remainder;
        }

        let work = rows.saturating_mul(self.n_samples);
        if work >= PGEN_PARALLEL_WORK_THRESHOLD && tasks.len() > 1 {
            tasks
                .par_iter()
                .zip(task_outputs.into_par_iter())
                .try_for_each(|(task, task_output)| decode_task_into(task, task_output))?;
        } else {
            tasks
                .iter()
                .zip(task_outputs.into_iter())
                .try_for_each(|(task, task_output)| decode_task_into(task, task_output))?;
        }
        Ok(())
    }

    fn read_variable_rows_with_qc<F>(
        &mut self,
        source_start: usize,
        rows: usize,
        sample_selection: GenotypeSampleSelection<'_>,
        keep_variant: &F,
        mut out_packed: Option<&mut [u8]>,
        stats_out: &mut [GenotypeRowStats],
        keep_out: &mut [bool],
    ) -> Result<(), String>
    where
        F: Fn(usize, GenotypeRowStats) -> bool + Sync,
    {
        let mut rows_read = 0usize;
        while rows_read < rows {
            let source_row = source_start + rows_read;
            let block_idx = source_row / PGEN_BLOCK_VARIANTS;
            self.load_block(block_idx)?;
            let block_start = block_idx * PGEN_BLOCK_VARIANTS;
            let local_start = source_row - block_start;
            let block_rows = self.block_meta.record_types.len();
            let local_rows = (rows - rows_read).min(block_rows - local_start);
            let packed_start = rows_read * self.bytes_per_variant;
            let packed_end = packed_start + local_rows * self.bytes_per_variant;
            let packed_out = out_packed
                .as_deref_mut()
                .map(|out| &mut out[packed_start..packed_end]);
            self.read_variable_block_range_with_qc(
                local_start,
                local_rows,
                sample_selection,
                keep_variant,
                packed_out,
                &mut stats_out[rows_read..rows_read + local_rows],
                &mut keep_out[rows_read..rows_read + local_rows],
            )?;
            rows_read += local_rows;
        }
        Ok(())
    }

    fn read_variable_rows_to_values<T: PgenDosageValue>(
        &mut self,
        source_start: usize,
        rows: usize,
        out: &mut [T],
    ) -> Result<(), String> {
        let expected_len = rows
            .checked_mul(self.n_samples)
            .ok_or_else(|| "PGEN dosage block size overflow".to_string())?;
        if out.len() != expected_len {
            return Err(format!(
                "invalid PGEN dosage destination: got {}, need {expected_len}",
                out.len()
            ));
        }
        let mut rows_read = 0usize;
        while rows_read < rows {
            let source_row = source_start + rows_read;
            let block_idx = source_row / PGEN_BLOCK_VARIANTS;
            self.load_block(block_idx)?;
            let block_start = block_idx * PGEN_BLOCK_VARIANTS;
            let local_start = source_row - block_start;
            let block_rows = self.block_meta.record_types.len();
            let local_rows = (rows - rows_read).min(block_rows - local_start);
            let value_start = rows_read * self.n_samples;
            let value_end = value_start + local_rows * self.n_samples;
            self.read_variable_block_range_to_values(
                local_start,
                local_rows,
                &mut out[value_start..value_end],
            )?;
            rows_read += local_rows;
        }
        Ok(())
    }

    fn read_variable_block_range_to_values<T: PgenDosageValue>(
        &self,
        local_start: usize,
        rows: usize,
        out: &mut [T],
    ) -> Result<(), String> {
        let local_end = local_start + rows;
        let block_rows = self.block_meta.record_types.len();
        if local_end > block_rows || out.len() != rows * self.n_samples {
            return Err("invalid PGEN direct-dosage output range".to_string());
        }

        #[derive(Clone, Copy)]
        struct DecodeTask {
            variant_start: usize,
            variant_end: usize,
            output_start: usize,
        }

        let anchors = &self.block_meta.anchor_rows;
        let first_anchor = anchors
            .partition_point(|&anchor| anchor <= local_start)
            .saturating_sub(1);
        let mut tasks = Vec::new();
        let mut anchor_idx = first_anchor;
        while anchor_idx < anchors.len() && anchors[anchor_idx] < local_end {
            let task_anchor_end = (anchor_idx + PGEN_PARALLEL_ANCHORS_PER_TASK).min(anchors.len());
            let variant_start = anchors[anchor_idx];
            let variant_end = if task_anchor_end < anchors.len() {
                anchors[task_anchor_end]
            } else {
                block_rows
            }
            .min(local_end);
            tasks.push(DecodeTask {
                variant_start,
                variant_end,
                output_start: variant_start.max(local_start),
            });
            if variant_end == local_end {
                break;
            }
            anchor_idx = task_anchor_end;
        }
        if tasks.is_empty() {
            return Err("PGEN dosage range has no preceding anchor variant".to_string());
        }

        let block_start = self
            .current_block
            .unwrap_or_default()
            .saturating_mul(PGEN_BLOCK_VARIANTS);
        let decode_task_into = |task: &DecodeTask, task_output: &mut [T]| {
            let output_rows = task.variant_end - task.output_start;
            let expected_len = output_rows * self.n_samples;
            if task_output.len() != expected_len {
                return Err("invalid PGEN direct-dosage task output slice".to_string());
            }
            let mut current = vec![0u8; self.bytes_per_variant];
            let mut base = vec![0u8; self.bytes_per_variant];
            for variant in task.variant_start..task.variant_end {
                let record_type = self.block_meta.record_types[variant];
                let record_start = self.block_meta.record_offsets[variant];
                let record_end = self.block_meta.record_offsets[variant + 1];
                decode_packed_record(
                    record_type,
                    &self.block_records[record_start..record_end],
                    self.n_samples,
                    block_start + variant,
                    &self.path,
                    &mut current,
                    &mut base,
                )?;
                if variant >= task.output_start {
                    let row_idx = variant - task.output_start;
                    let row_start = row_idx * self.n_samples;
                    let row_end = row_start + self.n_samples;
                    decode_pgen_category_row_to_values(
                        &current,
                        self.n_samples,
                        &mut task_output[row_start..row_end],
                    );
                }
            }
            Ok(())
        };

        let mut task_outputs = Vec::with_capacity(tasks.len());
        let mut output_remainder = out;
        for task in &tasks {
            let task_len = (task.variant_end - task.output_start) * self.n_samples;
            let (task_output, remainder) = output_remainder.split_at_mut(task_len);
            task_outputs.push(task_output);
            output_remainder = remainder;
        }
        let work = rows.saturating_mul(self.n_samples);
        if work >= PGEN_PARALLEL_WORK_THRESHOLD && tasks.len() > 1 {
            tasks
                .par_iter()
                .zip(task_outputs.into_par_iter())
                .try_for_each(|(task, task_output)| decode_task_into(task, task_output))?;
        } else {
            tasks
                .iter()
                .zip(task_outputs.into_iter())
                .try_for_each(|(task, task_output)| decode_task_into(task, task_output))?;
        }
        Ok(())
    }

    fn read_variable_block_range_with_qc<F>(
        &self,
        local_start: usize,
        rows: usize,
        sample_selection: GenotypeSampleSelection<'_>,
        keep_variant: &F,
        out: Option<&mut [u8]>,
        stats_out: &mut [GenotypeRowStats],
        keep_out: &mut [bool],
    ) -> Result<(), String>
    where
        F: Fn(usize, GenotypeRowStats) -> bool + Sync,
    {
        let local_end = local_start + rows;
        let block_rows = self.block_meta.record_types.len();
        if local_end > block_rows
            || out
                .as_ref()
                .is_some_and(|packed| packed.len() != rows * self.bytes_per_variant)
            || stats_out.len() != rows
            || keep_out.len() != rows
        {
            return Err("invalid PGEN fused-QC output range".to_string());
        }

        #[derive(Clone, Copy)]
        struct DecodeTask {
            variant_start: usize,
            variant_end: usize,
            output_start: usize,
        }

        let anchors = &self.block_meta.anchor_rows;
        let first_anchor = anchors
            .partition_point(|&anchor| anchor <= local_start)
            .saturating_sub(1);
        let mut tasks = Vec::new();
        let mut anchor_idx = first_anchor;
        while anchor_idx < anchors.len() && anchors[anchor_idx] < local_end {
            let task_anchor_end = (anchor_idx + PGEN_PARALLEL_ANCHORS_PER_TASK).min(anchors.len());
            let variant_start = anchors[anchor_idx];
            let variant_end = if task_anchor_end < anchors.len() {
                anchors[task_anchor_end]
            } else {
                block_rows
            }
            .min(local_end);
            tasks.push(DecodeTask {
                variant_start,
                variant_end,
                output_start: variant_start.max(local_start),
            });
            if variant_end == local_end {
                break;
            }
            anchor_idx = task_anchor_end;
        }
        if tasks.is_empty() {
            return Err("PGEN fused-QC range has no preceding anchor variant".to_string());
        }

        let block_start = self
            .current_block
            .unwrap_or_default()
            .saturating_mul(PGEN_BLOCK_VARIANTS);
        let decode_task_into = |task: &DecodeTask,
                                mut task_output: Option<&mut [u8]>,
                                task_stats: &mut [GenotypeRowStats],
                                task_keep: &mut [bool]| {
            let output_rows = task.variant_end - task.output_start;
            let expected_packed = output_rows * self.bytes_per_variant;
            if task_output
                .as_deref()
                .is_some_and(|packed| packed.len() != expected_packed)
                || task_stats.len() != output_rows
                || task_keep.len() != output_rows
            {
                return Err("invalid PGEN fused-QC task output slice".to_string());
            }
            let mut current = vec![0u8; self.bytes_per_variant];
            let mut base = vec![0u8; self.bytes_per_variant];
            for variant in task.variant_start..task.variant_end {
                let record_type = self.block_meta.record_types[variant];
                let record_start = self.block_meta.record_offsets[variant];
                let record_end = self.block_meta.record_offsets[variant + 1];
                decode_packed_record(
                    record_type,
                    &self.block_records[record_start..record_end],
                    self.n_samples,
                    block_start + variant,
                    &self.path,
                    &mut current,
                    &mut base,
                )?;
                if variant >= task.output_start {
                    let row_idx = variant - task.output_start;
                    let row_start = row_idx * self.bytes_per_variant;
                    let row_end = row_start + self.bytes_per_variant;
                    let stats = count_pgen_category_row(&current, self.n_samples, sample_selection);
                    let keep = keep_variant(block_start + variant, stats);
                    task_stats[row_idx] = stats;
                    task_keep[row_idx] = keep;
                    if let Some(task_output) = task_output.as_deref_mut() {
                        if keep {
                            for (dst, &category_byte) in
                                task_output[row_start..row_end].iter_mut().zip(&current)
                            {
                                *dst = pgen_category_byte_to_row_byte(category_byte);
                            }
                            clear_unused_output_lanes(
                                &mut task_output[row_start..row_end],
                                self.n_samples,
                            );
                        } else {
                            task_output[row_start..row_end].fill(0);
                        }
                    }
                }
            }
            Ok(())
        };

        let mut task_packed_outputs: Vec<Option<&mut [u8]>> = Vec::with_capacity(tasks.len());
        if let Some(out) = out {
            let mut packed_remainder = out;
            for task in &tasks {
                let task_rows = task.variant_end - task.output_start;
                let packed_len = task_rows * self.bytes_per_variant;
                let (packed_task, packed_next) = packed_remainder.split_at_mut(packed_len);
                task_packed_outputs.push(Some(packed_task));
                packed_remainder = packed_next;
            }
        } else {
            task_packed_outputs.extend(tasks.iter().map(|_| None));
        }
        let mut task_stats_outputs = Vec::with_capacity(tasks.len());
        let mut stats_remainder = stats_out;
        let mut task_keep_outputs = Vec::with_capacity(tasks.len());
        let mut keep_remainder = keep_out;
        for task in &tasks {
            let task_rows = task.variant_end - task.output_start;
            let (stats_task, stats_next) = stats_remainder.split_at_mut(task_rows);
            task_stats_outputs.push(stats_task);
            stats_remainder = stats_next;
            let (keep_task, keep_next) = keep_remainder.split_at_mut(task_rows);
            task_keep_outputs.push(keep_task);
            keep_remainder = keep_next;
        }

        let work = rows.saturating_mul(self.n_samples);
        if work >= PGEN_PARALLEL_WORK_THRESHOLD && tasks.len() > 1 {
            tasks
                .par_iter()
                .zip(task_packed_outputs.into_par_iter())
                .zip(task_stats_outputs.into_par_iter())
                .zip(task_keep_outputs.into_par_iter())
                .try_for_each(|(((task, packed), stats), keep)| {
                    decode_task_into(task, packed, stats, keep)
                })?;
        } else {
            tasks
                .iter()
                .zip(task_packed_outputs.into_iter())
                .zip(task_stats_outputs.into_iter())
                .zip(task_keep_outputs.into_iter())
                .try_for_each(|(((task, packed), stats), keep)| {
                    decode_task_into(task, packed, stats, keep)
                })?;
        }
        Ok(())
    }

    fn read_variant_into(&mut self, out: &mut [u8]) -> Result<(), String> {
        if self.next_variant >= self.n_variants {
            return Err("PGEN read past end of variants".to_string());
        }
        if out.len() < self.bytes_per_variant {
            return Err("PGEN packed row destination is too small".to_string());
        }
        match self.mode {
            StorageMode::BedCompat => {
                self.file
                    .read_exact(&mut out[..self.bytes_per_variant])
                    .map_err(|e| {
                        format!("read {} BED-compatible PGEN row: {e}", self.path.display())
                    })?;
                self.record_offset = self
                    .record_offset
                    .saturating_add(self.bytes_per_variant as u64);
                self.next_variant += 1;
                return Ok(());
            }
            StorageMode::FixedHardcall => {
                self.record_buf.resize(self.bytes_per_variant, 0);
                self.file.read_exact(&mut self.record_buf).map_err(|e| {
                    format!("read {} fixed-width PGEN row: {e}", self.path.display())
                })?;
                let record = std::mem::take(&mut self.record_buf);
                let decoded = self.decode_uncompressed(&record);
                self.record_buf = record;
                decoded?;
                self.next_variant += 1;
            }
            StorageMode::Variable => {
                let block_idx = self.next_variant / PGEN_BLOCK_VARIANTS;
                if self.current_block != Some(block_idx) {
                    self.load_block(block_idx)?;
                }
                let local_idx = self.next_variant - block_idx * PGEN_BLOCK_VARIANTS;
                let record_type =
                    *self.block_meta.record_types.get(local_idx).ok_or_else(|| {
                        format!(
                            "{}: missing record type for variant {}",
                            self.path.display(),
                            self.next_variant
                        )
                    })?;
                let record_len =
                    *self
                        .block_meta
                        .record_lengths
                        .get(local_idx)
                        .ok_or_else(|| {
                            format!(
                                "{}: missing record length for variant {}",
                                self.path.display(),
                                self.next_variant
                            )
                        })?;
                let record_end = self
                    .block_record_cursor
                    .checked_add(record_len)
                    .ok_or_else(|| "PGEN block record cursor overflow".to_string())?;
                if record_end > self.block_records.len() {
                    return Err(format!(
                        "{}: variant {} record exceeds the loaded PGEN block payload",
                        self.path.display(),
                        self.next_variant
                    ));
                }
                let record_start = self.block_record_cursor;
                self.block_record_cursor = record_end;
                let decoded = decode_packed_record(
                    record_type,
                    &self.block_records[record_start..record_end],
                    self.n_samples,
                    self.next_variant,
                    &self.path,
                    &mut self.current_categories,
                    &mut self.base_categories,
                );
                decoded?;
                self.record_offset = self.record_offset.saturating_add(record_len as u64);
                self.next_variant += 1;
            }
        }
        for (dst, &src) in out[..self.bytes_per_variant]
            .iter_mut()
            .zip(&self.current_categories)
        {
            *dst = pgen_category_byte_to_row_byte(src);
        }
        clear_unused_output_lanes(&mut out[..self.bytes_per_variant], self.n_samples);
        Ok(())
    }

    fn decode_uncompressed(&mut self, record: &[u8]) -> Result<(), String> {
        decode_packed_record(
            0,
            record,
            self.n_samples,
            self.next_variant,
            &self.path,
            &mut self.current_categories,
            &mut self.base_categories,
        )
    }
}

fn decode_packed_record(
    record_type: u8,
    record: &[u8],
    n_samples: usize,
    variant_idx: usize,
    path: &Path,
    current: &mut [u8],
    base: &mut [u8],
) -> Result<(), String> {
    let packed_len = n_samples.div_ceil(4);
    if n_samples == 0 || current.len() != packed_len || base.len() != packed_len {
        return Err(format!(
            "{}: invalid packed category state for {} samples",
            path.display(),
            n_samples
        ));
    }
    if record_type & 0x08 != 0 {
        return Err(format!(
            "{}: multi-allelic hard calls are not supported (variant {variant_idx})",
            path.display()
        ));
    }

    let compression = record_type & 0x07;
    match compression {
        0 => {
            if record.len() < packed_len {
                return Err(format!(
                    "{}: truncated uncompressed PGEN hard-call track at variant {variant_idx}",
                    path.display()
                ));
            }
            current.copy_from_slice(&record[..packed_len]);
            clear_unused_output_lanes(current, n_samples);
            base.copy_from_slice(current);
        }
        1 => {
            decode_one_bit_packed(record, n_samples, path, variant_idx, current)?;
            clear_unused_output_lanes(current, n_samples);
            base.copy_from_slice(current);
        }
        2 | 3 => {
            current.copy_from_slice(base);
            let mut pos = 0usize;
            apply_difflist(record, &mut pos, n_samples, |sample, category| {
                set_packed_category(current, sample, category);
            })?;
            if compression == 3 {
                invert_packed_categories(current);
            }
            clear_unused_output_lanes(current, n_samples);
        }
        4 | 6 | 7 => {
            let common = match compression {
                4 => 0,
                6 => 2,
                _ => 3,
            };
            current.fill(common * 0x55);
            let mut pos = 0usize;
            apply_difflist(record, &mut pos, n_samples, |sample, category| {
                set_packed_category(current, sample, category);
            })?;
            clear_unused_output_lanes(current, n_samples);
            base.copy_from_slice(current);
        }
        5 => {
            current.fill(0);
            base.copy_from_slice(current);
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn decode_one_bit_packed(
    record: &[u8],
    n_samples: usize,
    path: &Path,
    variant_idx: usize,
    out: &mut [u8],
) -> Result<(), String> {
    let pair_code = *record
        .first()
        .ok_or_else(|| format!("{}: truncated one-bit PGEN record", path.display()))?;
    let pair_idx = one_bit_pair_index(pair_code).ok_or_else(|| {
        format!(
            "{}: invalid PGEN one-bit category pair {pair_code} at variant {variant_idx}",
            path.display()
        )
    })?;
    let bit_bytes = n_samples.div_ceil(8);
    let bit_end = 1usize
        .checked_add(bit_bytes)
        .ok_or_else(|| "PGEN one-bit array size overflow".to_string())?;
    if record.len() < bit_end {
        return Err(format!(
            "{}: truncated PGEN one-bit array at variant {variant_idx}",
            path.display()
        ));
    }
    let bits = &record[1..bit_end];
    for (packed_idx, packed_byte) in out.iter_mut().enumerate() {
        let first_sample = packed_idx * 4;
        let bit_byte = bits[first_sample >> 3];
        let selector = (bit_byte >> (first_sample & 7)) & 0x0f;
        *packed_byte = ONE_BIT_NIBBLE_LUT[pair_idx][selector as usize];
    }
    let mut pos = bit_end;
    apply_difflist(record, &mut pos, n_samples, |sample, category| {
        set_packed_category(out, sample, category);
    })?;
    Ok(())
}

#[inline]
fn clear_unused_output_lanes(packed: &mut [u8], n_samples: usize) {
    let used_in_last_byte = n_samples & 3;
    if used_in_last_byte != 0 {
        if let Some(last) = packed.last_mut() {
            let mask = (1u16 << (used_in_last_byte * 2)) - 1;
            *last &= mask as u8;
        }
    }
}

fn read_standard_dimensions(
    file: &mut BufReader<File>,
    path: &Path,
) -> Result<(usize, usize, u8), String> {
    file.seek(SeekFrom::Start(3))
        .map_err(|e| format!("seek {} header: {e}", path.display()))?;
    let mut dims = [0u8; 9];
    file.read_exact(&mut dims)
        .map_err(|e| format!("read {} dimensions: {e}", path.display()))?;
    let n_variants = u32::from_le_bytes(dims[0..4].try_into().unwrap()) as usize;
    let n_samples = u32::from_le_bytes(dims[4..8].try_into().unwrap()) as usize;
    if n_variants == 0 || n_samples == 0 {
        return Err(format!(
            "{}: PGEN header reports zero samples or variants",
            path.display()
        ));
    }
    Ok((n_variants, n_samples, dims[8]))
}

fn validate_dimensions(
    path: &Path,
    header_samples: usize,
    header_variants: usize,
    psam_samples: usize,
    pvar_variants: usize,
) -> Result<(), String> {
    if header_samples != psam_samples || header_variants != pvar_variants {
        return Err(format!(
            "{}: PGEN/PVAR/PSAM dimension mismatch (pgen={header_samples} samples x {header_variants} variants, psam={psam_samples}, pvar={pvar_variants})",
            path.display()
        ));
    }
    Ok(())
}

const fn build_pgen_category_byte_to_row_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    let mut byte = 0usize;
    while byte < lut.len() {
        let mut lane = 0usize;
        while lane < 4 {
            let category = ((byte >> (lane * 2)) & 0x03) as u8;
            let row_code = match category {
                0 => 0b00,
                1 => 0b10,
                2 => 0b11,
                _ => 0b01,
            };
            lut[byte] |= row_code << (lane * 2);
            lane += 1;
        }
        byte += 1;
    }
    lut
}

const PGEN_CATEGORY_BYTE_TO_ROW_LUT: [u8; 256] = build_pgen_category_byte_to_row_lut();

const fn build_pgen_category_byte_to_dosage_lut() -> [[u8; 4]; 256] {
    let mut lut = [[0u8; 4]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut lane = 0usize;
        while lane < 4 {
            let category = ((byte >> (lane * 2)) & 0x03) as u8;
            lut[byte][lane] = match category {
                0 => 0,
                1 => 1,
                2 => 2,
                _ => 3,
            };
            lane += 1;
        }
        byte += 1;
    }
    lut
}

const fn build_bed_byte_to_dosage_lut() -> [[u8; 4]; 256] {
    let mut lut = [[0u8; 4]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut lane = 0usize;
        while lane < 4 {
            let code = ((byte >> (lane * 2)) & 0x03) as u8;
            lut[byte][lane] = match code {
                0 => 0,
                1 => 3,
                2 => 1,
                _ => 2,
            };
            lane += 1;
        }
        byte += 1;
    }
    lut
}

const PGEN_CATEGORY_BYTE_TO_DOSAGE_LUT: [[u8; 4]; 256] = build_pgen_category_byte_to_dosage_lut();
const BED_BYTE_TO_DOSAGE_LUT: [[u8; 4]; 256] = build_bed_byte_to_dosage_lut();

#[inline]
pub(crate) fn pgen_category_byte_to_row_byte(byte: u8) -> u8 {
    PGEN_CATEGORY_BYTE_TO_ROW_LUT[byte as usize]
}

fn decode_pgen_category_row_to_values<T: PgenDosageValue>(
    row: &[u8],
    n_samples: usize,
    out: &mut [T],
) {
    debug_assert!(out.len() >= n_samples);
    for (byte_idx, &byte) in row.iter().enumerate() {
        let sample_start = byte_idx * 4;
        if sample_start >= n_samples {
            break;
        }
        let lane_count = (n_samples - sample_start).min(4);
        let codes = PGEN_CATEGORY_BYTE_TO_DOSAGE_LUT[byte as usize];
        for lane in 0..lane_count {
            out[sample_start + lane] = T::from_raw_genotype_code(codes[lane]);
        }
    }
}

pub(crate) fn decode_bed_row_to_values<T: PgenDosageValue>(
    row: &[u8],
    n_samples: usize,
    out: &mut [T],
) {
    debug_assert!(out.len() >= n_samples);
    for (byte_idx, &byte) in row.iter().enumerate() {
        let sample_start = byte_idx * 4;
        if sample_start >= n_samples {
            break;
        }
        let lane_count = (n_samples - sample_start).min(4);
        let codes = BED_BYTE_TO_DOSAGE_LUT[byte as usize];
        for lane in 0..lane_count {
            out[sample_start + lane] = T::from_raw_genotype_code(codes[lane]);
        }
    }
}

pub(crate) fn count_bed_genotype_row(
    row: &[u8],
    n_samples: usize,
    sample_selection: GenotypeSampleSelection<'_>,
) -> GenotypeRowStats {
    count_two_bit_genotypes(row, n_samples, sample_selection, false)
}

fn count_pgen_category_row(
    row: &[u8],
    n_samples: usize,
    sample_selection: GenotypeSampleSelection<'_>,
) -> GenotypeRowStats {
    count_two_bit_genotypes(row, n_samples, sample_selection, true)
}

fn count_two_bit_genotypes(
    row: &[u8],
    n_samples: usize,
    sample_selection: GenotypeSampleSelection<'_>,
    pgen_categories: bool,
) -> GenotypeRowStats {
    if let GenotypeSampleSelection::Include(indices) = sample_selection {
        let mut counts = GenotypeRowStats {
            sample_count: indices.len(),
            ..GenotypeRowStats::default()
        };
        for &sample_idx in indices {
            let code = (row[sample_idx / 4] >> ((sample_idx % 4) * 2)) & 0x03;
            match (pgen_categories, code) {
                (_, 0) => {}
                (true, 1) | (false, 2) => counts.heterozygote_count += 1,
                (true, 2) | (false, 3) => counts.homozygous_alt_count += 1,
                (true, 3) | (false, 1) => counts.missing_count += 1,
                _ => unreachable!("two-bit genotype code is in range"),
            }
        }
        return counts;
    }

    let mut counts = GenotypeRowStats {
        sample_count: n_samples,
        ..GenotypeRowStats::default()
    };
    let full_bytes = n_samples / 4;
    for &byte in row.iter().take(full_bytes) {
        let low = byte & 0x55;
        let high = (byte >> 1) & 0x55;
        if pgen_categories {
            counts.missing_count += (low & high).count_ones() as usize;
            counts.heterozygote_count += (low & !high).count_ones() as usize;
            counts.homozygous_alt_count += (!low & high & 0x55).count_ones() as usize;
        } else {
            counts.missing_count += (low & !high).count_ones() as usize;
            counts.heterozygote_count += (!low & high & 0x55).count_ones() as usize;
            counts.homozygous_alt_count += (low & high).count_ones() as usize;
        }
    }
    let remainder = n_samples & 3;
    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let code = (byte >> (lane * 2)) & 0x03;
            match (pgen_categories, code) {
                (_, 0) => {}
                (true, 1) | (false, 2) => counts.heterozygote_count += 1,
                (true, 2) | (false, 3) => counts.homozygous_alt_count += 1,
                (true, 3) | (false, 1) => counts.missing_count += 1,
                _ => unreachable!("two-bit genotype code is in range"),
            }
        }
    }
    if let GenotypeSampleSelection::Exclude(indices) = sample_selection {
        counts.sample_count = n_samples.saturating_sub(indices.len());
        for &sample_idx in indices {
            let code = (row[sample_idx / 4] >> ((sample_idx % 4) * 2)) & 0x03;
            match (pgen_categories, code) {
                (_, 0) => {}
                (true, 1) | (false, 2) => counts.heterozygote_count -= 1,
                (true, 2) | (false, 3) => counts.homozygous_alt_count -= 1,
                (true, 3) | (false, 1) => counts.missing_count -= 1,
                _ => unreachable!("two-bit genotype code is in range"),
            }
        }
    }
    counts
}

const ONE_BIT_PAIR_CATEGORIES: [(u8, u8); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

const fn build_one_bit_nibble_lut() -> [[u8; 16]; 6] {
    let mut lut = [[0u8; 16]; 6];
    let mut pair_idx = 0usize;
    while pair_idx < ONE_BIT_PAIR_CATEGORIES.len() {
        let (low, high) = ONE_BIT_PAIR_CATEGORIES[pair_idx];
        let mut selector = 0usize;
        while selector < 16 {
            let mut lane = 0usize;
            while lane < 4 {
                let category = if (selector >> lane) & 1 == 0 {
                    low
                } else {
                    high
                };
                lut[pair_idx][selector] |= category << (lane * 2);
                lane += 1;
            }
            selector += 1;
        }
        pair_idx += 1;
    }
    lut
}

const ONE_BIT_NIBBLE_LUT: [[u8; 16]; 6] = build_one_bit_nibble_lut();

#[inline]
fn one_bit_pair_index(pair_code: u8) -> Option<usize> {
    match pair_code {
        1 => Some(0),
        2 => Some(1),
        3 => Some(2),
        5 => Some(3),
        6 => Some(4),
        9 => Some(5),
        _ => None,
    }
}

#[inline]
#[cfg(test)]
fn one_bit_nibble_to_packed(pair_code: u8, selector_nibble: u8) -> Result<u8, String> {
    let pair_idx = one_bit_pair_index(pair_code)
        .ok_or_else(|| format!("invalid PGEN one-bit category pair {pair_code}"))?;
    Ok(ONE_BIT_NIBBLE_LUT[pair_idx][(selector_nibble & 0x0f) as usize])
}

#[inline]
fn set_packed_category(packed: &mut [u8], sample: usize, category: u8) {
    let byte = &mut packed[sample >> 2];
    let shift = (sample & 3) * 2;
    let mask = 0x03 << shift;
    *byte = (*byte & !mask) | ((category & 0x03) << shift);
}

#[inline]
fn invert_packed_categories(packed: &mut [u8]) {
    for byte in packed {
        let low_bits = *byte & 0x55;
        let swap_high_bits = (!low_bits) & 0x55;
        *byte ^= swap_high_bits << 1;
    }
}

fn read_varint(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    let mut value = 0u32;
    for byte_idx in 0..5u32 {
        let byte = *data
            .get(*pos)
            .ok_or_else(|| "truncated PGEN varint".to_string())?;
        *pos += 1;
        if byte_idx == 4 && byte & 0xf0 != 0 {
            return Err("PGEN varint overflow".to_string());
        }
        value |= ((byte & 0x7f) as u32) << (byte_idx * 7);
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("PGEN varint overflow".to_string())
}

fn apply_difflist(
    data: &[u8],
    pos: &mut usize,
    n_samples: usize,
    mut apply: impl FnMut(usize, u8),
) -> Result<(), String> {
    let count = read_varint(data, pos)? as usize;
    if count == 0 {
        return Ok(());
    }
    if count > n_samples {
        return Err(format!(
            "PGEN difflist length {count} exceeds sample count {n_samples}"
        ));
    }
    let groups = count.div_ceil(64);
    let id_width = difflist_sample_id_width(n_samples);
    let ids_start = *pos;
    let ids_end = pos
        .checked_add(groups.saturating_mul(id_width))
        .ok_or_else(|| "PGEN difflist sample-ID array overflow".to_string())?;
    let group_sizes_end = ids_end
        .checked_add(groups.saturating_sub(1))
        .ok_or_else(|| "PGEN difflist group-size array overflow".to_string())?;
    if group_sizes_end > data.len() {
        return Err("truncated PGEN difflist group index".to_string());
    }
    *pos = group_sizes_end;
    let values_len = count.div_ceil(4);
    let values_end = pos
        .checked_add(values_len)
        .ok_or_else(|| "PGEN difflist genotype array overflow".to_string())?;
    let values = data
        .get(*pos..values_end)
        .ok_or_else(|| "truncated PGEN difflist genotype array".to_string())?;
    *pos = values_end;
    let mut sample_id = 0usize;
    for idx in 0..count {
        if idx % 64 == 0 {
            let group_start = ids_start + (idx / 64) * id_width;
            sample_id = 0;
            for byte_idx in 0..id_width {
                sample_id |= (data[group_start + byte_idx] as usize) << (byte_idx * 8);
            }
        } else {
            sample_id = sample_id
                .checked_add(read_varint(data, pos)? as usize)
                .ok_or_else(|| "PGEN difflist sample-ID overflow".to_string())?;
        }
        if sample_id >= n_samples {
            return Err(format!(
                "PGEN difflist sample index {sample_id} is out of range"
            ));
        }
        let category = (values[idx >> 2] >> ((idx & 3) * 2)) & 0x03;
        apply(sample_id, category);
    }
    Ok(())
}

#[inline]
fn difflist_sample_id_width(n_samples: usize) -> usize {
    if n_samples < (1 << 8) {
        1
    } else if n_samples < (1 << 16) {
        2
    } else if n_samples < (1 << 24) {
        3
    } else {
        4
    }
}

pub(crate) fn normalize_pgen_prefix(path_or_prefix: &str) -> String {
    let path = path_or_prefix.trim();
    let lower = path.to_ascii_lowercase();
    for suffix in [".pgen", ".pvar", ".psam"] {
        if lower.ends_with(suffix) {
            return path[..path.len() - suffix.len()].to_string();
        }
    }
    path.to_string()
}

pub(crate) fn path_uses_pgen(path_or_prefix: &str) -> bool {
    let lower = path_or_prefix.trim().to_ascii_lowercase();
    if [".pgen", ".pvar", ".psam"]
        .iter()
        .any(|suffix| lower.ends_with(suffix))
    {
        return true;
    }
    let prefix = normalize_pgen_prefix(path_or_prefix);
    Path::new(&format!("{prefix}.pgen")).is_file() && !Path::new(&format!("{prefix}.bed")).is_file()
}

pub(crate) fn read_psam_iids(path_or_prefix: &str) -> Result<Vec<String>, String> {
    let prefix = normalize_pgen_prefix(path_or_prefix);
    let path = PathBuf::from(format!("{prefix}.psam"));
    let file = std::fs::File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let reader = BufReader::new(file);
    let mut header: Option<Vec<String>> = None;
    let mut sample_ids = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read {}:{}: {e}", path.display(), line_idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("##") {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            let cols = fields
                .iter()
                .map(|field| field.trim_start_matches('#').to_ascii_uppercase())
                .collect::<Vec<_>>();
            if !cols.iter().any(|field| field == "IID") {
                return Err(format!(
                    "{}:{}: PSAM header has no IID column",
                    path.display(),
                    line_idx + 1
                ));
            }
            header = Some(cols);
            continue;
        }
        let iid_idx = if let Some(cols) = header.as_ref() {
            cols.iter()
                .position(|field| field == "IID")
                .unwrap_or(usize::MAX)
        } else if fields.len() >= 6 {
            1
        } else {
            0
        };
        let iid = fields
            .get(iid_idx)
            .ok_or_else(|| format!("{}:{}: malformed PSAM row", path.display(), line_idx + 1))?;
        sample_ids.push((*iid).to_string());
    }
    if sample_ids.is_empty() {
        return Err(format!("{} contains zero samples", path.display()));
    }
    Ok(sample_ids)
}

fn count_non_comment_rows(path: &Path) -> Result<usize, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut line = Vec::with_capacity(256);
    let mut count = 0usize;
    let mut line_idx = 0usize;
    loop {
        line.clear();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| format!("read {}:{}: {e}", path.display(), line_idx + 1))?;
        if bytes_read == 0 {
            break;
        }
        line_idx += 1;
        let first_non_whitespace = line
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_whitespace());
        if first_non_whitespace.is_some_and(|byte| byte != b'#') {
            count = count.saturating_add(1);
        }
    }
    Ok(count)
}

pub(crate) fn count_psam_samples(path_or_prefix: &str) -> Result<usize, String> {
    let prefix = normalize_pgen_prefix(path_or_prefix);
    let path = PathBuf::from(format!("{prefix}.psam"));
    let count = count_non_comment_rows(&path)?;
    if count == 0 {
        return Err(format!("{} contains zero samples", path.display()));
    }
    Ok(count)
}

pub(crate) fn read_pvar_sites(path_or_prefix: &str) -> Result<Vec<PgenSite>, String> {
    let prefix = normalize_pgen_prefix(path_or_prefix);
    let path = PathBuf::from(format!("{prefix}.pvar"));
    let file = std::fs::File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let reader = BufReader::new(file);
    let mut header: Option<Vec<String>> = None;
    let mut sites = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read {}:{}: {e}", path.display(), line_idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("##") {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            let cols = fields
                .iter()
                .map(|field| field.trim_start_matches('#').to_ascii_uppercase())
                .collect::<Vec<_>>();
            for required in ["CHROM", "POS", "ID", "REF", "ALT"] {
                if !cols.iter().any(|field| field == required) {
                    return Err(format!(
                        "{}:{}: PVAR header is missing {required}",
                        path.display(),
                        line_idx + 1
                    ));
                }
            }
            header = Some(cols);
            continue;
        }
        let cols = header.as_ref().ok_or_else(|| {
            format!(
                "{}:{}: PVAR must include a #CHROM header",
                path.display(),
                line_idx + 1
            )
        })?;
        let field = |name: &str| -> Result<&str, String> {
            let idx = cols
                .iter()
                .position(|col| col == name)
                .ok_or_else(|| format!("{}: PVAR header is missing {name}", path.display()))?;
            fields.get(idx).copied().ok_or_else(|| {
                format!(
                    "{}:{}: PVAR row is missing {name}",
                    path.display(),
                    line_idx + 1
                )
            })
        };
        let alt = field("ALT")?;
        if alt == "." || alt.contains(',') {
            return Err(format!(
                "{}:{}: only biallelic PGEN variants are supported; ALT={alt}",
                path.display(),
                line_idx + 1
            ));
        }
        let pos = field("POS")?.parse::<i32>().map_err(|e| {
            format!(
                "{}:{}: invalid POS value: {e}",
                path.display(),
                line_idx + 1
            )
        })?;
        sites.push(PgenSite {
            chrom: field("CHROM")?.to_string(),
            pos,
            id: field("ID")?.to_string(),
            ref_allele: field("REF")?.to_string(),
            alt_allele: alt.to_string(),
        });
    }
    if sites.is_empty() {
        return Err(format!("{} contains zero variants", path.display()));
    }
    Ok(sites)
}

pub(crate) fn count_pvar_variants(path_or_prefix: &str) -> Result<usize, String> {
    let prefix = normalize_pgen_prefix(path_or_prefix);
    let path = PathBuf::from(format!("{prefix}.pvar"));
    let count = count_non_comment_rows(&path)?;
    if count == 0 {
        return Err(format!("{} contains zero variants", path.display()));
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::{
        decode_packed_record, invert_packed_categories, one_bit_nibble_to_packed,
        pgen_category_byte_to_row_byte, set_packed_category, GenotypeRowStats,
        GenotypeSampleSelection, PgenReader, PGEN_BLOCK_VARIANTS,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEMP_FILE_ID: AtomicUsize = AtomicUsize::new(0);

    struct TempPgen(PathBuf);

    impl TempPgen {
        fn new(bytes: &[u8]) -> Self {
            let id = TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("janusx-pgen-unit-{}-{id}.pgen", std::process::id()));
            fs::write(&path, bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for TempPgen {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn variable_pgen(n_samples: usize, records: &[(u8, Vec<u8>)]) -> Vec<u8> {
        let block_count = records.len().div_ceil(PGEN_BLOCK_VARIANTS);
        let mut block_metas = Vec::with_capacity(block_count);
        for block in records.chunks(PGEN_BLOCK_VARIANTS) {
            let mut meta = Vec::with_capacity(block.len().div_ceil(2) + block.len());
            for pair in block.chunks(2) {
                let low = pair[0].0;
                let high = pair.get(1).map(|entry| entry.0).unwrap_or(0);
                meta.push(low | (high << 4));
            }
            for (_, record) in block {
                assert!(record.len() < 256);
                meta.push(record.len() as u8);
            }
            block_metas.push(meta);
        }

        let mut record_offsets = Vec::with_capacity(block_count);
        let mut data_offset =
            12 + block_count * 8 + block_metas.iter().map(Vec::len).sum::<usize>();
        for block in records.chunks(PGEN_BLOCK_VARIANTS) {
            record_offsets.push(data_offset as u64);
            data_offset += block.iter().map(|(_, record)| record.len()).sum::<usize>();
        }

        let mut bytes = Vec::with_capacity(data_offset);
        bytes.extend_from_slice(&[0x6c, 0x1b, 0x10]);
        bytes.extend_from_slice(&(records.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(n_samples as u32).to_le_bytes());
        bytes.push(0); // 4-bit record type and 1-byte record length.
        for offset in record_offsets {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for meta in block_metas {
            bytes.extend_from_slice(&meta);
        }
        for (_, record) in records {
            bytes.extend_from_slice(record);
        }
        bytes
    }

    fn decode_record(
        record_type: u8,
        record: &[u8],
        n_samples: usize,
        current: &mut [u8],
        base: &mut [u8],
    ) -> Result<(), String> {
        decode_packed_record(
            record_type,
            record,
            n_samples,
            0,
            std::path::Path::new("unit.pgen"),
            current,
            base,
        )
    }

    #[test]
    fn packed_category_byte_maps_to_existing_row_codes() {
        assert_eq!(pgen_category_byte_to_row_byte(0xe4), 0x78);
    }

    #[test]
    fn one_bit_selector_nibble_builds_expected_categories() {
        for (pair_code, expected) in [
            (1, 0x44),
            (2, 0x88),
            (3, 0xcc),
            (5, 0x99),
            (6, 0xdd),
            (9, 0xee),
        ] {
            assert_eq!(
                one_bit_nibble_to_packed(pair_code, 0x0a).unwrap(),
                expected,
                "pair code {pair_code}"
            );
        }
    }

    #[test]
    fn one_bit_selector_nibble_rejects_invalid_pair_code() {
        assert!(one_bit_nibble_to_packed(0x04, 0x0a).is_err());
    }

    #[test]
    fn packed_difflist_sample_ids_use_one_more_byte_at_power_of_256_boundaries() {
        for (n_samples, record) in [
            (256usize, vec![1, 0xff, 0x00, 0x02]),
            (65_536, vec![1, 0xff, 0xff, 0x00, 0x02]),
            (16_777_216, vec![1, 0xff, 0xff, 0xff, 0x00, 0x02]),
        ] {
            let mut current = vec![0u8; n_samples.div_ceil(4)];
            let mut base = vec![0u8; n_samples.div_ceil(4)];
            decode_record(4, &record, n_samples, &mut current, &mut base).unwrap();
            assert_eq!(current[n_samples.div_ceil(4) - 1], 0x80, "n={n_samples}");
            assert_eq!(base[n_samples.div_ceil(4) - 1], 0x80, "n={n_samples}");
        }
    }

    #[test]
    fn packed_category_update_preserves_neighbor_lanes() {
        let mut packed = [0x1b, 0b1011_0100];
        set_packed_category(&mut packed, 4, 2);
        assert_eq!(packed, [0x1b, 0b1011_0110]);
    }

    #[test]
    fn packed_category_inversion_swaps_only_homozygotes() {
        let mut packed = [0xe4]; // packed categories [0, 1, 2, 3]
        invert_packed_categories(&mut packed);
        assert_eq!(packed, [0xc6]); // packed categories [2, 1, 0, 3]
    }

    #[test]
    fn packed_uncompressed_record_copies_and_clears_padding() {
        for (n_samples, record, expected) in [
            (1usize, vec![0xff], vec![0x03]),
            (3usize, vec![0xe4], vec![0x24]),
            (5usize, vec![0xe4, 0b1111_1101], vec![0xe4, 0x01]),
        ] {
            let mut current = vec![0u8; n_samples.div_ceil(4)];
            let mut base = vec![0u8; n_samples.div_ceil(4)];
            decode_record(0, &record, n_samples, &mut current, &mut base).unwrap();
            assert_eq!(current, expected);
            assert_eq!(base, expected);
        }
    }

    #[test]
    fn packed_one_bit_record_applies_sparse_override() {
        let mut current = [0u8; 2];
        let mut base = [0u8; 2];
        // Pair (0,2), selector bits for samples 1 and 3, then sample 4 -> category 1.
        decode_record(1, &[2, 0x0a, 1, 4, 1], 5, &mut current, &mut base).unwrap();
        assert_eq!(current, [0x88, 0x01]);
        assert_eq!(base, [0x88, 0x01]);
    }

    #[test]
    fn packed_common_category_difflist_patches_only_selected_lane() {
        let mut current = [0xff];
        let mut base = [0xff];
        // Type 4: common category 0; sample 1 differs with category 2.
        decode_record(4, &[1, 1, 2], 4, &mut current, &mut base).unwrap();
        assert_eq!(current, [0x08]);
        assert_eq!(base, [0x08]);

        // Type 6: common category 2; sample 0 differs with category 0.
        decode_record(6, &[1, 0, 0], 4, &mut current, &mut base).unwrap();
        assert_eq!(current, [0xa8]);
        assert_eq!(base, [0xa8]);

        // Type 7: common category 3; sample 2 differs with category 1.
        decode_record(7, &[1, 2, 1], 4, &mut current, &mut base).unwrap();
        assert_eq!(current, [0xdf]);
        assert_eq!(base, [0xdf]);
    }

    #[test]
    fn packed_difflist_decodes_across_64_sample_groups() {
        let mut record = vec![65, 0, 64, 64];
        record.extend([0x55; 17]);
        record.extend([1; 63]);
        let mut current = [0u8; 17];
        let mut base = [0u8; 17];

        decode_record(4, &record, 65, &mut current, &mut base).unwrap();

        let mut expected = [0x55; 17];
        expected[16] = 0x01;
        assert_eq!(current, expected);
        assert_eq!(base, expected);
    }

    #[test]
    fn packed_ld_difflist_keeps_base_and_inverts_only_for_type_three() {
        let mut current = [0xe4];
        let mut base = [0xe4];
        decode_record(2, &[1, 1, 3], 4, &mut current, &mut base).unwrap();
        assert_eq!(current, [0xec]);
        assert_eq!(base, [0xe4]);

        decode_record(3, &[1, 2, 0], 4, &mut current, &mut base).unwrap();
        assert_eq!(current, [0xe6]);
        assert_eq!(base, [0xe4]);
    }

    #[test]
    fn packed_decoder_rejects_multiallelic_records() {
        let mut current = [0u8; 1];
        let mut base = [0u8; 1];
        assert!(decode_record(8, &[0], 4, &mut current, &mut base).is_err());
    }

    #[test]
    fn packed_type_five_record_decodes_an_all_reference_row() {
        let mut current = [0xff];
        let mut base = [0xff];
        decode_record(5, &[], 4, &mut current, &mut base).unwrap();
        assert_eq!(current, [0x00]);
        assert_eq!(base, [0x00]);
    }

    #[test]
    fn packed_reader_keeps_bed_compatible_mode_as_passthrough() {
        let file = TempPgen::new(&[0x6c, 0x1b, 0x01, 0x78]);
        let mut reader = PgenReader::open(&file.0, 4, 1).unwrap();
        let mut output = [0u8; 1];
        assert_eq!(reader.read_source_block(0, 1, &mut output).unwrap(), 1);
        assert_eq!(output, [0x78]);

        let mut qc_reader = PgenReader::open(&file.0, 4, 1).unwrap();
        let mut stats = [GenotypeRowStats::default(); 1];
        let mut keep = [false; 1];
        qc_reader
            .read_source_block_qc_only(
                0,
                1,
                GenotypeSampleSelection::all(),
                &mut stats,
                &mut keep,
                &|_, _| true,
            )
            .unwrap();
        assert_eq!(stats[0].missing_count, 1);
        assert_eq!(stats[0].heterozygote_count, 1);
        assert_eq!(stats[0].homozygous_alt_count, 1);
        assert_eq!(keep, [true]);
    }

    #[test]
    fn packed_reader_fixed_hardcall_clears_unused_output_lanes() {
        let mut bytes = vec![0x6c, 0x1b, 0x02];
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&[0xe4, 0b1111_1101]);
        let file = TempPgen::new(&bytes);
        let mut reader = PgenReader::open(&file.0, 5, 1).unwrap();
        let mut output = [0xff; 2];
        assert_eq!(reader.read_source_block(0, 1, &mut output).unwrap(), 1);
        assert_eq!(output, [0x78, 0x02]);

        let mut qc_reader = PgenReader::open(&file.0, 5, 1).unwrap();
        let mut stats = [GenotypeRowStats::default(); 1];
        let mut keep = [false; 1];
        qc_reader
            .read_source_block_qc_only(
                0,
                1,
                GenotypeSampleSelection::all(),
                &mut stats,
                &mut keep,
                &|_, _| true,
            )
            .unwrap();
        assert_eq!(stats[0].missing_count, 1);
        assert_eq!(stats[0].heterozygote_count, 2);
        assert_eq!(stats[0].homozygous_alt_count, 1);
        assert_eq!(keep, [true]);
    }

    #[test]
    fn packed_reader_seeks_past_fixed_provisional_ref_flags_before_first_row() {
        let mut bytes = vec![0x6c, 0x1b, 0x02];
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.push(0xc0); // provisional-ref mode 3 stores one flag byte.
        bytes.push(0x01); // ref flags, not genotype data.
        bytes.push(0xe4); // packed PGEN categories [0, 1, 2, 3].
        let file = TempPgen::new(&bytes);
        let mut reader = PgenReader::open(&file.0, 4, 1).unwrap();
        let mut output = [0u8; 1];
        assert_eq!(reader.read_source_block(0, 1, &mut output).unwrap(), 1);
        assert_eq!(output, [0x78]);
    }

    #[test]
    fn packed_reader_accepts_zero_length_type_five_all_reference_variant() {
        let file = TempPgen::new(&variable_pgen(4, &[(5, vec![])]));
        let mut reader = PgenReader::open(&file.0, 4, 1).unwrap();
        let mut output = [0xff];
        assert_eq!(reader.read_source_block(0, 1, &mut output).unwrap(), 1);
        assert_eq!(output, [0x00]);
    }

    #[test]
    fn packed_reader_parallel_anchor_groups_match_sequential_decode() {
        let n_samples = 1024;
        let records = (0..1024)
            .map(|idx| match idx % 4 {
                0 => (4, vec![1, 0, 0, 2]),
                1 => (2, vec![1, 1, 0, 1]),
                2 => (3, vec![1, 2, 0, 3]),
                _ => (2, vec![0]),
            })
            .collect::<Vec<_>>();
        let file = TempPgen::new(&variable_pgen(n_samples, &records));
        let mut reader = PgenReader::open(&file.0, n_samples, records.len()).unwrap();
        let bytes_per_variant = n_samples.div_ceil(4);
        let mut actual = vec![0u8; records.len() * bytes_per_variant];

        reader
            .read_source_block(0, records.len(), &mut actual)
            .unwrap();

        let mut expected = vec![0u8; actual.len()];
        let mut current = vec![0u8; bytes_per_variant];
        let mut base = vec![0u8; bytes_per_variant];
        for (variant, (record_type, record)) in records.iter().enumerate() {
            decode_packed_record(
                *record_type,
                record,
                n_samples,
                variant,
                &file.0,
                &mut current,
                &mut base,
            )
            .unwrap();
            let row_start = variant * bytes_per_variant;
            for (dst, &category_byte) in expected[row_start..row_start + bytes_per_variant]
                .iter_mut()
                .zip(&current)
            {
                *dst = pgen_category_byte_to_row_byte(category_byte);
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn fused_pgen_qc_uses_sample_subset_and_skips_bed_encoding_for_rejected_rows() {
        let records = vec![
            (0u8, vec![0xc4, 0x01]), // categories [0, 1, 2, 3, 1] for five samples.
            (4u8, vec![1, 1, 2]),    // common category 0; sample 1 is category 2.
        ];
        let file = TempPgen::new(&variable_pgen(5, &records));
        let mut reader = PgenReader::open(&file.0, 5, records.len()).unwrap();
        let sample_indices = [1usize, 3, 4];
        let mut packed = [0xaau8; 4];
        let mut stats = [GenotypeRowStats::default(); 2];
        let mut keep = [true; 2];

        assert_eq!(
            reader
                .read_source_block_with_qc(
                    0,
                    2,
                    GenotypeSampleSelection::include(&sample_indices),
                    &mut packed,
                    &mut stats,
                    &mut keep,
                    &|source_idx, row_stats| source_idx == 1 && row_stats.missing_count == 0,
                )
                .unwrap(),
            2
        );

        assert_eq!(stats[0].sample_count, 3);
        assert_eq!(stats[0].missing_count, 1);
        assert_eq!(stats[0].heterozygote_count, 2);
        assert_eq!(stats[0].homozygous_alt_count, 0);
        assert_eq!(stats[1].sample_count, 3);
        assert_eq!(stats[1].missing_count, 0);
        assert_eq!(stats[1].heterozygote_count, 0);
        assert_eq!(stats[1].homozygous_alt_count, 1);
        assert_eq!(keep, [false, true]);
        assert_eq!(packed, [0, 0, 0x0c, 0]);
    }

    #[test]
    fn pgen_qc_only_matches_packed_qc_without_emitting_rows() {
        let records = vec![(0u8, vec![0xc4, 0x01]), (4u8, vec![1, 1, 2])];
        let file = TempPgen::new(&variable_pgen(5, &records));
        let mut packed_reader = PgenReader::open(&file.0, 5, records.len()).unwrap();
        let mut stats_only_reader = PgenReader::open(&file.0, 5, records.len()).unwrap();
        let mut packed = [0xaau8; 4];
        let mut packed_stats = [GenotypeRowStats::default(); 2];
        let mut packed_keep = [false; 2];
        let mut stats_only = [GenotypeRowStats::default(); 2];
        let mut stats_only_keep = [false; 2];
        let filter =
            |source_idx, stats: GenotypeRowStats| source_idx == 1 && stats.missing_count == 0;

        packed_reader
            .read_source_block_with_qc(
                0,
                2,
                GenotypeSampleSelection::all(),
                &mut packed,
                &mut packed_stats,
                &mut packed_keep,
                &filter,
            )
            .unwrap();
        stats_only_reader
            .read_source_block_qc_only(
                0,
                2,
                GenotypeSampleSelection::all(),
                &mut stats_only,
                &mut stats_only_keep,
                &filter,
            )
            .unwrap();

        assert_eq!(stats_only, packed_stats);
        assert_eq!(stats_only_keep, packed_keep);
        assert_eq!(packed, [0, 0, 0x0c, 0]);
    }

    #[test]
    fn pgen_streaming_dosage_decode_writes_f32_and_f64_directly() {
        let records = vec![(0u8, vec![0xc4, 0x01]), (4u8, vec![1, 1, 2])];
        let file = TempPgen::new(&variable_pgen(5, &records));
        let mut f32_reader = PgenReader::open(&file.0, 5, records.len()).unwrap();
        let mut f64_reader = PgenReader::open(&file.0, 5, records.len()).unwrap();
        let mut values_f32 = [0.0_f32; 10];
        let mut values_f64 = [0.0_f64; 10];

        assert_eq!(
            f32_reader
                .read_source_block_f32(0, 2, &mut values_f32)
                .unwrap(),
            2
        );
        assert_eq!(
            f64_reader
                .read_source_block_f64(0, 2, &mut values_f64)
                .unwrap(),
            2
        );
        assert_eq!(
            values_f32,
            [0.0, 1.0, 0.0, -9.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            values_f64,
            [0.0, 1.0, 0.0, -9.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn metadata_row_counter_skips_headers_comments_and_blank_lines() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "janusx-pgen-row-count-{}-{stamp}.pvar",
            std::process::id()
        ));
        fs::write(
            &path,
            b"##meta=value\r\n#CHROM\tPOS\tID\tREF\tALT\r\n\r\n  # comment\n1\t100\trs1\tA\tG\n2\t200\trs2\tC\tT\n",
        )
        .unwrap();
        assert_eq!(super::count_non_comment_rows(&path).unwrap(), 2);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fused_pgen_qc_exclusion_selection_matches_inclusion_selection() {
        let records = vec![(0u8, vec![0xc4, 0x01])];
        let file = TempPgen::new(&variable_pgen(5, &records));
        let mut include_reader = PgenReader::open(&file.0, 5, 1).unwrap();
        let mut exclude_reader = PgenReader::open(&file.0, 5, 1).unwrap();
        let include = [1usize, 3, 4];
        let exclude = [0usize, 2];
        let mut include_packed = [0u8; 2];
        let mut exclude_packed = [0u8; 2];
        let mut include_stats = [GenotypeRowStats::default(); 1];
        let mut exclude_stats = [GenotypeRowStats::default(); 1];
        let mut include_keep = [false; 1];
        let mut exclude_keep = [false; 1];

        include_reader
            .read_source_block_with_qc(
                0,
                1,
                GenotypeSampleSelection::include(&include),
                &mut include_packed,
                &mut include_stats,
                &mut include_keep,
                &|_, _| true,
            )
            .unwrap();
        exclude_reader
            .read_source_block_with_qc(
                0,
                1,
                GenotypeSampleSelection::Exclude(&exclude),
                &mut exclude_packed,
                &mut exclude_stats,
                &mut exclude_keep,
                &|_, _| true,
            )
            .unwrap();

        assert_eq!(exclude_stats, include_stats);
        assert_eq!(exclude_keep, include_keep);
        assert_eq!(exclude_packed, include_packed);
    }

    #[test]
    fn packed_reader_zero_row_read_at_exact_block_boundary_eof_is_empty() {
        let records = vec![(4u8, vec![0]); PGEN_BLOCK_VARIANTS];
        let file = TempPgen::new(&variable_pgen(4, &records));
        let mut reader = PgenReader::open(&file.0, 4, records.len()).unwrap();
        let mut first = [0u8; 1];
        assert_eq!(reader.read_source_block(0, 1, &mut first).unwrap(), 1);
        assert_eq!(
            reader.read_source_block(records.len(), 0, &mut []).unwrap(),
            0
        );
    }

    #[test]
    fn packed_reader_resets_and_crosses_pgen_variant_block_boundary() {
        let n_samples = 4;
        // An all-reference block has zero-length type-5 records, so its data
        // offset is shared with the following block's data offset.
        let mut records = vec![(5u8, vec![]); PGEN_BLOCK_VARIANTS + 2];
        records[PGEN_BLOCK_VARIANTS] = (6, vec![0]); // New block base: all category 2.
        records[PGEN_BLOCK_VARIANTS + 1] = (2, vec![1, 0, 0]); // LD patch sample 0 to category 0.
        let file = TempPgen::new(&variable_pgen(n_samples, &records));
        let mut reader = PgenReader::open(&file.0, n_samples, records.len()).unwrap();

        let mut late = [0u8; 1];
        assert_eq!(
            reader
                .read_source_block(PGEN_BLOCK_VARIANTS + 1, 1, &mut late)
                .unwrap(),
            1
        );
        assert_eq!(late, [0xfc]);

        let mut all_rows = vec![0u8; records.len()];
        assert_eq!(
            reader
                .read_source_block(0, records.len(), &mut all_rows)
                .unwrap(),
            records.len()
        );
        assert_eq!(all_rows[PGEN_BLOCK_VARIANTS], 0xff);
        assert_eq!(all_rows[PGEN_BLOCK_VARIANTS + 1], 0xfc);
    }
}
