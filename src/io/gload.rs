//! Unified genotype-loading abstraction.
//!
//! Two back-ends implement [`GenotypeMatrix`]:
//! * [`BedMmapMatrix`] — the BED file is memory-mapped; genotype rows are
//!   zero-copy slices into the mmap, indexed by source SNP position.
//! * [`PackedBedMatrix`] — the filtered BED payload is owned in-memory
//!   (`Arc<[u8]>`), packed contiguously across kept SNPs.
//!
//! [`GlobalStats`] carries per-marker statistics (MAF / miss / flip) computed
//! once at load time.  Together with a [`GenotypeMatrix`] they form a
//! self-contained [`UnifiedInput`] that can be passed to downstream scanners.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;

use crate::bedmath::{decode_mean_imputed_additive_packed_block_rows_f32, packed_byte_lut};
use crate::decode::{decode_centered_block_packed_model_f32, PackedGeneticModel};
use crate::gfcore;
use crate::gstats::{compute_bed_filter_meta, BedFilterMetaMode};

const BED_HEADER_LEN: usize = 3;

#[inline]
fn gload_force_sequential_owned() -> bool {
    match std::env::var("JX_GLOAD_FORCE_SEQUENTIAL") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => false,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PlinkBedHeaderInfo {
    pub(crate) bytes_per_snp: usize,
    pub(crate) file_size: usize,
    pub(crate) payload_size: usize,
    pub(crate) n_snps_total: usize,
}

#[inline]
fn validate_plink_bed_header(header: [u8; BED_HEADER_LEN]) -> Result<(), String> {
    if header != [0x6C, 0x1B, 0x01] {
        return Err("Only SNP-major BED supported".to_string());
    }
    Ok(())
}

pub(crate) fn read_plink_bed_header_info(
    bed_path: &Path,
    n_samples_full: usize,
) -> Result<PlinkBedHeaderInfo, String> {
    if n_samples_full == 0 {
        return Err("no samples found in BED".to_string());
    }
    let mut file = File::open(bed_path).map_err(|e| format!("open {}: {e}", bed_path.display()))?;
    let mut header = [0u8; BED_HEADER_LEN];
    file.read_exact(&mut header)
        .map_err(|e| format!("read {} header: {e}", bed_path.display()))?;
    validate_plink_bed_header(header)?;
    let bytes_per_snp = n_samples_full.div_ceil(4);
    let file_size = usize::try_from(
        file.metadata()
            .map_err(|e| format!("stat {}: {e}", bed_path.display()))?
            .len(),
    )
    .map_err(|_| format!("BED file size overflow: {}", bed_path.display()))?;
    if file_size < BED_HEADER_LEN {
        return Err(format!("BED file is too small: {}", bed_path.display()));
    }
    let payload_size = file_size - BED_HEADER_LEN;
    if bytes_per_snp == 0 || payload_size % bytes_per_snp != 0 {
        return Err(format!(
            "BED payload size mismatch: payload={payload_size}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    Ok(PlinkBedHeaderInfo {
        bytes_per_snp,
        file_size,
        payload_size,
        n_snps_total: payload_size / bytes_per_snp,
    })
}

pub(crate) fn open_plink_bed_reader_after_header(
    bed_path: &Path,
) -> Result<BufReader<File>, String> {
    let file = File::open(bed_path).map_err(|e| format!("open {}: {e}", bed_path.display()))?;
    let mut reader = BufReader::new(file);
    let mut header = [0u8; BED_HEADER_LEN];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("read {} header: {e}", bed_path.display()))?;
    validate_plink_bed_header(header)?;
    Ok(reader)
}

pub(crate) fn mmap_readonly_file(path: &Path) -> Result<Mmap, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", path.display()))
}

#[inline]
fn file_size_usize(file: &File, path: &Path) -> Result<usize, String> {
    usize::try_from(
        file.metadata()
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len(),
    )
    .map_err(|_| format!("file size overflow: {}", path.display()))
}

fn load_file_owned_range_from_file(
    file: &File,
    path: &Path,
    offset: usize,
    expected_size: usize,
) -> Result<Vec<u8>, String> {
    let file_size = file_size_usize(file, path)?;
    let end = offset
        .checked_add(expected_size)
        .ok_or_else(|| format!("file range overflow for {}", path.display()))?;
    if end > file_size {
        return Err(format!(
            "file range out of bounds for {}: offset={} size={} file_size={}",
            path.display(),
            offset,
            expected_size,
            file_size
        ));
    }
    if expected_size == 0 {
        return Ok(Vec::new());
    }

    let threads = if gload_force_sequential_owned() {
        1usize
    } else {
        rayon::current_num_threads().max(1)
    };
    let parallel_threshold = 64usize << 20;
    let min_chunk = 8usize << 20;
    let mut buf = vec![0u8; expected_size];
    if threads <= 1 || expected_size < parallel_threshold {
        #[cfg(any(unix, windows))]
        {
            read_file_exact_at(file, &mut buf, offset as u64)?;
            return Ok(buf);
        }
        #[cfg(not(any(unix, windows)))]
        {
            use std::io::{Seek, SeekFrom};
            let mut reader_file = file
                .try_clone()
                .map_err(|e| format!("clone {} for sequential read: {e}", path.display()))?;
            reader_file
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| format!("seek {}: {e}", path.display()))?;
            let mut reader = BufReader::new(reader_file);
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            return Ok(buf);
        }
    }

    let nominal = expected_size.div_ceil(threads);
    let chunk_size = nominal.max(min_chunk);
    buf.par_chunks_mut(chunk_size)
        .enumerate()
        .try_for_each(|(chunk_idx, chunk)| {
            let chunk_offset = chunk_idx
                .checked_mul(chunk_size)
                .ok_or_else(|| format!("chunk offset overflow: {}", path.display()))?;
            let file_offset = offset
                .checked_add(chunk_offset)
                .ok_or_else(|| format!("file offset overflow: {}", path.display()))?
                as u64;
            read_file_exact_at(file, chunk, file_offset)
        })?;
    Ok(buf)
}

pub(crate) fn load_file_owned(path: &Path) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let file_size = file_size_usize(&file, path)?;
    load_file_owned_range_from_file(&file, path, 0, file_size)
}

pub(crate) fn load_file_owned_exact(path: &Path, expected_size: usize) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let file_size = file_size_usize(&file, path)?;
    if file_size != expected_size {
        return Err(format!(
            "file size mismatch for {}: expected {}, got {}",
            path.display(),
            expected_size,
            file_size
        ));
    }
    load_file_owned_range_from_file(&file, path, 0, expected_size)
}

pub(crate) fn load_file_owned_range_exact(
    path: &Path,
    offset: usize,
    expected_size: usize,
) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    load_file_owned_range_from_file(&file, path, offset, expected_size)
}

#[inline]
fn system_page_size() -> usize {
    #[cfg(unix)]
    {
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if ps > 0 {
            ps as usize
        } else {
            4096
        }
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

#[cfg(unix)]
pub(crate) fn read_file_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
        .map_err(|e| format!("BED positioned read: {e}"))
}

#[cfg(windows)]
pub(crate) fn read_file_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::windows::fs::FileExt;

    let mut done = 0usize;
    while done < buf.len() {
        let n = file
            .seek_read(&mut buf[done..], offset + done as u64)
            .map_err(|e| format!("BED positioned read: {e}"))?;
        if n == 0 {
            return Err("BED positioned read: unexpected EOF".to_string());
        }
        done += n;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn read_file_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = file
        .try_clone()
        .map_err(|e| format!("BED file clone for positioned read: {e}"))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("BED seek: {e}"))?;
    file.read_exact(buf).map_err(|e| format!("BED read: {e}"))
}

/// Per-marker statistics computed once at load time.
///
/// All vectors are in **kept-marker order** (i.e. after MAF/missing/het
/// filtering).  `site_keep` is the only field indexed in the **original**
/// SNP order.
#[derive(Clone)]
#[allow(dead_code)]
pub struct GlobalStats {
    pub maf: Vec<f32>,
    pub miss: Vec<f32>,
    pub row_flip: Vec<bool>,
    pub row_source_indices: Vec<usize>,
    pub site_keep: Vec<bool>,
    pub n_samples_full: usize,
    pub n_markers_total: usize,
    pub bytes_per_snp: usize,
}

impl GlobalStats {
    #[inline]
    pub fn n_markers(&self) -> usize {
        if self.maf.is_empty() {
            self.n_markers_total
        } else {
            self.maf.len()
        }
    }

    /// Row-source slice for a block of kept markers, suitable for
    /// `decode_mean_imputed_additive_packed_block_rows_f32`.
    #[inline]
    pub fn row_source_slice(&self, row_start: usize, rows_here: usize) -> &[usize] {
        &self.row_source_indices[row_start..][..rows_here]
    }

    /// Row-flip slice for a block of kept markers.
    #[inline]
    pub fn row_flip_slice(&self, row_start: usize, rows_here: usize) -> &[bool] {
        &self.row_flip[row_start..][..rows_here]
    }

    /// MAF slice for a block of kept markers.
    #[inline]
    pub fn row_maf_slice(&self, row_start: usize, rows_here: usize) -> &[f32] {
        &self.maf[row_start..][..rows_here]
    }
}

// ---------------------------------------------------------------------------
// GenotypeMatrix trait
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub trait GenotypeMatrix: Send + Sync {
    fn n_samples_full(&self) -> usize;
    fn bytes_per_snp(&self) -> usize;

    /// Full packed payload.  For mmap-backed matrices this covers the
    /// **entire** BED (all original SNPs); callers must index through
    /// row_source_indices to reach kept rows.
    fn packed_flat(&self) -> &[u8];

    /// Raw packed bytes at a specific source SNP index (original BIM row).
    /// This is the primitive accessor for mmap matrices.
    fn source_row_bytes(&self, source_idx: usize) -> &[u8];

    /// Ensure the packed source rows for a kept-marker block are available to
    /// `source_row_bytes`. Resident backends need no preparation; streaming
    /// sources may fill a bounded block cache here.
    fn prepare_source_rows(
        &mut self,
        _stats: &GlobalStats,
        _row_start: usize,
        _rows_here: usize,
    ) -> Result<(), String> {
        Ok(())
    }

    /// MAF values produced by a streaming decoder for the most recent block.
    /// Resident BED backends return `None` because their metadata is already
    /// available in `GlobalStats`; k-file adapters use this bounded slice for
    /// direct TSV output without allocating one vector per marker.
    fn block_maf(&self, _rows_here: usize) -> Option<&[f32]> {
        None
    }

    /// Decode a block of kept markers via `decode_mean_imputed_additive_packed_block_rows_f32`.
    /// Default impl uses `packed_flat()` with absolute source offsets.
    /// Streaming backends override to pre-read file windows.
    fn decode_additive_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let cols = if sample_identity {
            stats.n_samples_full
        } else {
            sample_idx.len()
        };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 {
            return Ok(());
        }
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            self.packed_flat(),
            stats.bytes_per_snp,
            stats.n_samples_full,
            stats.row_flip_slice(row_start, rows_here),
            stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(stats.row_source_slice(row_start, rows_here)),
            0,
            out,
            code4_lut,
            pool,
        )
    }
}

// ---------------------------------------------------------------------------
/// Direct source-order BED reader with a caller-owned packed destination.
/// It validates the file header and row shape once, then reads only requested
/// source rows into the supplied buffer.
pub(crate) struct PackedBedSourceReader {
    source: PackedBedSource,
    n_samples_full: usize,
    bytes_per_snp: usize,
    n_snps: usize,
    qc_scratch: Vec<u8>,
}

enum PackedBedSource {
    Bed(File),
    Pgen(crate::pgen::PgenReader),
}

impl PackedBedSourceReader {
    pub(crate) fn open(prefix: &str, n_samples_full: usize) -> Result<Self, String> {
        let pgen_source = crate::pgen::path_uses_pgen(prefix);
        let bed_prefix = normalize_plink_prefix(prefix);
        if pgen_source {
            let pgen_prefix = crate::pgen::normalize_pgen_prefix(prefix);
            let psam_samples = crate::pgen::count_psam_samples(&pgen_prefix)?;
            if psam_samples != n_samples_full {
                return Err(format!(
                    "PGEN/PSAM sample count mismatch: reader was given {n_samples_full}, PSAM has {psam_samples}"
                ));
            }
            let n_snps = crate::pgen::count_pvar_variants(&pgen_prefix)?;
            let pgen_path = Path::new(&format!("{pgen_prefix}.pgen")).to_path_buf();
            let reader = crate::pgen::PgenReader::open(&pgen_path, n_samples_full, n_snps)?;
            return Ok(Self {
                source: PackedBedSource::Pgen(reader),
                n_samples_full,
                bytes_per_snp: n_samples_full.div_ceil(4),
                n_snps,
                qc_scratch: Vec::new(),
            });
        }
        let bed_path = format!("{bed_prefix}.bed");
        let info = read_plink_bed_header_info(Path::new(&bed_path), n_samples_full)?;
        if info.n_snps_total == 0 {
            return Err("no SNP sites found in PLINK BED input".to_string());
        }
        let file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        Ok(Self {
            source: PackedBedSource::Bed(file),
            n_samples_full,
            bytes_per_snp: info.bytes_per_snp,
            n_snps: info.n_snps_total,
            qc_scratch: Vec::new(),
        })
    }

    #[inline]
    pub(crate) fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }

    #[inline]
    pub(crate) fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }

    #[inline]
    pub(crate) fn n_snps(&self) -> usize {
        self.n_snps
    }

    pub(crate) fn read_source_block(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [u8],
    ) -> Result<usize, String> {
        if source_start > self.n_snps {
            return Err(format!(
                "BED source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        let rows = requested_rows.min(self.n_snps - source_start);
        let byte_len = rows.saturating_mul(self.bytes_per_snp);
        if out.len() < byte_len {
            return Err(format!(
                "packed BED destination is too small: got {}, need {byte_len}",
                out.len()
            ));
        }
        if rows > 0 {
            match &mut self.source {
                PackedBedSource::Bed(file) => {
                    let offset = (BED_HEADER_LEN as u64)
                        .saturating_add((source_start.saturating_mul(self.bytes_per_snp)) as u64);
                    read_file_exact_at(file, &mut out[..byte_len], offset)
                        .map_err(|e| format!("read BED source rows [{source_start}..): {e}"))?;
                }
                PackedBedSource::Pgen(reader) => {
                    let rows_read =
                        reader.read_source_block(source_start, rows, &mut out[..byte_len])?;
                    if rows_read != rows {
                        return Err(format!(
                            "PGEN reader returned {rows_read} rows; requested {rows} starting at {source_start}"
                        ));
                    }
                }
            }
        }
        Ok(rows)
    }

    /// Reads a row-major f32 hard-call matrix, decoding PGEN category bytes
    /// directly to dosages instead of passing through BED-packed rows.
    #[allow(dead_code)] // Shared Rust streaming API for future genotype consumers.
    pub(crate) fn read_source_block_f32(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f32],
    ) -> Result<usize, String> {
        self.read_source_block_values_f32(source_start, requested_rows, out)
    }

    /// f64 counterpart for streaming consumers that need double-precision
    /// genotype rows.
    #[allow(dead_code)] // Shared Rust streaming API for future genotype consumers.
    pub(crate) fn read_source_block_f64(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f64],
    ) -> Result<usize, String> {
        self.read_source_block_values_f64(source_start, requested_rows, out)
    }

    #[allow(dead_code)]
    fn read_source_block_values_f32(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f32],
    ) -> Result<usize, String> {
        if source_start > self.n_snps {
            return Err(format!(
                "genotype source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        match &mut self.source {
            PackedBedSource::Pgen(reader) => {
                reader.read_source_block_f32(source_start, requested_rows, out)
            }
            PackedBedSource::Bed(file) => read_bed_source_block_values(
                file,
                self.n_samples_full,
                self.bytes_per_snp,
                self.n_snps,
                &mut self.qc_scratch,
                source_start,
                requested_rows,
                out,
            ),
        }
    }

    #[allow(dead_code)]
    fn read_source_block_values_f64(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f64],
    ) -> Result<usize, String> {
        if source_start > self.n_snps {
            return Err(format!(
                "genotype source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        match &mut self.source {
            PackedBedSource::Pgen(reader) => {
                reader.read_source_block_f64(source_start, requested_rows, out)
            }
            PackedBedSource::Bed(file) => read_bed_source_block_values(
                file,
                self.n_samples_full,
                self.bytes_per_snp,
                self.n_snps,
                &mut self.qc_scratch,
                source_start,
                requested_rows,
                out,
            ),
        }
    }

    /// Shared QC-aware read contract for BED and PGEN consumers. `stats_out`
    /// and `keep_out` are indexed by source row; rejected packed rows are
    /// zeroed and must be ignored by callers. The predicate is intentionally
    /// caller-defined so GWAS, GS, and GARFIELD can share decode/QC mechanics
    /// while retaining their own filtering rules.
    pub(crate) fn read_source_block_with_qc<F>(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        sample_selection: crate::pgen::GenotypeSampleSelection<'_>,
        out_packed: &mut [u8],
        stats_out: &mut [crate::pgen::GenotypeRowStats],
        keep_out: &mut [bool],
        keep_variant: &F,
    ) -> Result<usize, String>
    where
        F: Fn(usize, crate::pgen::GenotypeRowStats) -> bool + Sync,
    {
        if source_start > self.n_snps {
            return Err(format!(
                "BED source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        let rows = requested_rows.min(self.n_snps - source_start);
        let byte_len = rows
            .checked_mul(self.bytes_per_snp)
            .ok_or_else(|| "packed genotype QC block size overflow".to_string())?;
        if out_packed.len() < byte_len {
            return Err(format!(
                "packed genotype QC destination is too small: got {}, need {byte_len}",
                out_packed.len()
            ));
        }
        if stats_out.len() < rows || keep_out.len() < rows {
            return Err(format!(
                "genotype QC metadata destination too small: stats={}, keep={}, need {rows}",
                stats_out.len(),
                keep_out.len()
            ));
        }
        if rows == 0 {
            return Ok(0);
        }

        match &mut self.source {
            PackedBedSource::Pgen(reader) => reader.read_source_block_with_qc(
                source_start,
                rows,
                sample_selection,
                &mut out_packed[..byte_len],
                &mut stats_out[..rows],
                &mut keep_out[..rows],
                keep_variant,
            ),
            PackedBedSource::Bed(file) => {
                sample_selection.validate(self.n_samples_full)?;
                let offset = (BED_HEADER_LEN as u64)
                    .saturating_add((source_start.saturating_mul(self.bytes_per_snp)) as u64);
                read_file_exact_at(file, &mut out_packed[..byte_len], offset)
                    .map_err(|e| format!("read BED source rows [{source_start}..): {e}"))?;
                for row_idx in 0..rows {
                    let row_start = row_idx * self.bytes_per_snp;
                    let row_end = row_start + self.bytes_per_snp;
                    let stats = crate::pgen::count_bed_genotype_row(
                        &out_packed[row_start..row_end],
                        self.n_samples_full,
                        sample_selection,
                    );
                    let keep = keep_variant(source_start + row_idx, stats);
                    stats_out[row_idx] = stats;
                    keep_out[row_idx] = keep;
                    if !keep {
                        out_packed[row_start..row_end].fill(0);
                    }
                }
                Ok(rows)
            }
        }
    }

    /// Shared QC-only read contract. PGEN rows are counted while decoding and
    /// are not converted to packed BED; BED input is scanned through a reusable
    /// packed scratch block. Consumers receive per-source-row statistics and a
    /// keep mask without allocating a packed output matrix.
    pub(crate) fn read_source_block_qc_only<F>(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        sample_selection: crate::pgen::GenotypeSampleSelection<'_>,
        stats_out: &mut [crate::pgen::GenotypeRowStats],
        keep_out: &mut [bool],
        keep_variant: &F,
    ) -> Result<usize, String>
    where
        F: Fn(usize, crate::pgen::GenotypeRowStats) -> bool + Sync,
    {
        if source_start > self.n_snps {
            return Err(format!(
                "BED source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        let rows = requested_rows.min(self.n_snps - source_start);
        if stats_out.len() < rows || keep_out.len() < rows {
            return Err(format!(
                "genotype QC metadata destination too small: stats={}, keep={}, need {rows}",
                stats_out.len(),
                keep_out.len()
            ));
        }
        if rows == 0 {
            return Ok(0);
        }

        match &mut self.source {
            PackedBedSource::Pgen(reader) => reader.read_source_block_qc_only(
                source_start,
                rows,
                sample_selection,
                &mut stats_out[..rows],
                &mut keep_out[..rows],
                keep_variant,
            ),
            PackedBedSource::Bed(file) => {
                sample_selection.validate(self.n_samples_full)?;
                let byte_len = rows
                    .checked_mul(self.bytes_per_snp)
                    .ok_or_else(|| "packed genotype QC scratch size overflow".to_string())?;
                self.qc_scratch.resize(byte_len, 0);
                let offset = (BED_HEADER_LEN as u64)
                    .saturating_add((source_start.saturating_mul(self.bytes_per_snp)) as u64);
                read_file_exact_at(file, &mut self.qc_scratch, offset)
                    .map_err(|e| format!("read BED source rows [{source_start}..): {e}"))?;
                for row_idx in 0..rows {
                    let row_start = row_idx * self.bytes_per_snp;
                    let row_end = row_start + self.bytes_per_snp;
                    let stats = crate::pgen::count_bed_genotype_row(
                        &self.qc_scratch[row_start..row_end],
                        self.n_samples_full,
                        sample_selection,
                    );
                    stats_out[row_idx] = stats;
                    keep_out[row_idx] = keep_variant(source_start + row_idx, stats);
                }
                Ok(rows)
            }
        }
    }
}

#[allow(dead_code)]
fn read_bed_source_block_values<T: crate::pgen::PgenDosageValue>(
    file: &File,
    n_samples: usize,
    bytes_per_snp: usize,
    n_snps: usize,
    scratch: &mut Vec<u8>,
    source_start: usize,
    requested_rows: usize,
    out: &mut [T],
) -> Result<usize, String> {
    let rows = requested_rows.min(n_snps - source_start);
    let value_len = rows
        .checked_mul(n_samples)
        .ok_or_else(|| "genotype dosage block size overflow".to_string())?;
    if out.len() < value_len {
        return Err(format!(
            "genotype dosage destination is too small: got {}, need {value_len}",
            out.len()
        ));
    }
    if rows == 0 {
        return Ok(0);
    }
    let byte_len = rows
        .checked_mul(bytes_per_snp)
        .ok_or_else(|| "packed genotype dosage scratch size overflow".to_string())?;
    scratch.resize(byte_len, 0);
    let offset =
        (BED_HEADER_LEN as u64).saturating_add((source_start.saturating_mul(bytes_per_snp)) as u64);
    read_file_exact_at(file, scratch, offset)
        .map_err(|e| format!("read BED source rows [{source_start}..): {e}"))?;
    for row_idx in 0..rows {
        let packed_start = row_idx * bytes_per_snp;
        let packed_end = packed_start + bytes_per_snp;
        let values_start = row_idx * n_samples;
        let values_end = values_start + n_samples;
        crate::pgen::decode_bed_row_to_values(
            &scratch[packed_start..packed_end],
            n_samples,
            &mut out[values_start..values_end],
        );
    }
    Ok(rows)
}

// StreamingBedMatrix — chunked file I/O, no mmap
// ---------------------------------------------------------------------------

/// Reads BED rows on-demand via `pread` (or `seek`+`read`) instead of
/// mmap-ing the whole file.  Dramatically reduces RSS for large BEDs
/// when only sequential access is needed (e.g. GWAS scan).
#[allow(dead_code)]
pub struct StreamingBedMatrix {
    file: std::fs::File,
    n_samples_full: usize,
    bytes_per_snp: usize,
    /// Reusable read buffer sized for one block's worth of packed rows.
    buf: Vec<u8>,
    /// Byte range currently held in `buf` (start, end) in source-index units.
    buf_range: (usize, usize),
}

#[allow(dead_code)]
impl StreamingBedMatrix {
    /// Open a BED prefix for streaming reads.
    pub fn open(prefix: &str, block_rows: usize) -> Result<Self, String> {
        let bed_prefix = normalize_plink_prefix(prefix);
        let n_samples_full = crate::gfcore::read_fam(&bed_prefix)
            .map_err(|e| e.to_string())?
            .len();
        if n_samples_full == 0 {
            return Err("no samples found in BED".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let bed_path = format!("{bed_prefix}.bed");
        let file = std::fs::File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        // 3-byte BED header is skipped via offset in read_rows.
        let buf = vec![0u8; block_rows.saturating_mul(bytes_per_snp).max(1)];
        Ok(Self {
            file,
            n_samples_full,
            bytes_per_snp,
            buf,
            buf_range: (usize::MAX, 0),
        })
    }

    /// Ensure `buf` contains the packed bytes for source rows `[start .. end)`.
    /// Returns a slice covering the requested range.
    pub fn read_source_range(&mut self, start: usize, end: usize) -> Result<&[u8], String> {
        let bps = self.bytes_per_snp;
        let need = end.saturating_sub(start).saturating_mul(bps);
        if need > self.buf.len() {
            self.buf.resize(need, 0);
        }
        if start == self.buf_range.0 && end == self.buf_range.1 {
            return Ok(&self.buf[..need]);
        }
        let offset = 3u64.saturating_add((start.saturating_mul(bps)) as u64);
        read_file_exact_at(&self.file, &mut self.buf[..need], offset)
            .map_err(|e| format!("read BED rows [{start}..{end}): {e}"))?;
        self.buf_range = (start, end);
        Ok(&self.buf[..need])
    }
}

#[allow(dead_code)]
impl StreamingBedMatrix {
    /// Pre-fetch the byte range for a block of kept markers and return a
    /// contiguous packed slice + relative row-source indices ready for
    /// `decode_mean_imputed_additive_packed_block_rows_f32`.
    pub fn prepare_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        rows_here: usize,
        rel_indices_out: &mut Vec<usize>,
    ) -> Result<&[u8], String> {
        let src_start = stats.row_source_indices[row_start];
        let src_end = stats.row_source_indices[row_start + rows_here - 1] + 1;
        let window = self.read_source_range(src_start, src_end)?;
        rel_indices_out.clear();
        rel_indices_out.extend(
            (row_start..row_start + rows_here)
                .map(|i| stats.row_source_indices[i].saturating_sub(src_start)),
        );
        if rel_indices_out.len() != rows_here {
            return Err("prepare_block: rel_indices mismatch".to_string());
        }
        Ok(window)
    }
}

impl GenotypeMatrix for StreamingBedMatrix {
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }
    fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }
    fn packed_flat(&self) -> &[u8] {
        &self.buf[..(self
            .buf_range
            .1
            .saturating_sub(self.buf_range.0)
            .saturating_mul(self.bytes_per_snp))]
    }
    fn source_row_bytes(&self, _source_idx: usize) -> &[u8] {
        &[]
    }

    fn decode_additive_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let cols = if sample_identity {
            stats.n_samples_full
        } else {
            sample_idx.len()
        };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 {
            return Ok(());
        }
        let mut rel_indices = Vec::with_capacity(rows_here);
        let packed_slice = self.prepare_block(stats, row_start, rows_here, &mut rel_indices)?;
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed_slice,
            stats.bytes_per_snp,
            stats.n_samples_full,
            stats.row_flip_slice(row_start, rows_here),
            stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(&rel_indices),
            0,
            out,
            code4_lut,
            pool,
        )
    }
}

#[cfg(test)]
mod packed_source_tests {
    use super::{choose_lm_lmm_bed_pread, PackedBedSourceReader, WindowedBedMatrix};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn windowed_bed_source_block_preserves_raw_bim_order() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_packed_source_{stamp}"));
        fs::create_dir_all(&dir).expect("create temporary BED directory");
        let prefix = dir.join("tiny");
        fs::write(
            prefix.with_extension("bed"),
            [0x6c, 0x1b, 0x01, 0x31, 0x72, 0xB4, 0xA5],
        )
        .expect("write BED fixture");

        let mut reader = PackedBedSourceReader::open(prefix.to_str().unwrap(), 4)
            .expect("open direct packed BED source");
        assert_eq!(reader.n_snps(), 4);
        let mut out = vec![0u8; 3];
        assert_eq!(reader.read_source_block(1, 3, &mut out).unwrap(), 3);
        assert_eq!(out, [0x72, 0xB4, 0xA5]);

        let mut short = vec![0u8; 3];
        assert_eq!(reader.read_source_block(3, 3, &mut short).unwrap(), 1);
        assert_eq!(short[0], 0xA5);
        assert_eq!(reader.read_source_block(4, 1, &mut short).unwrap(), 0);
        assert!(reader.read_source_block(5, 1, &mut short).is_err());

        fs::remove_dir_all(dir).expect("remove temporary BED directory");
    }

    #[test]
    fn shared_qc_reader_applies_subset_stats_and_returns_keep_mask_for_bed() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_qc_source_{stamp}"));
        fs::create_dir_all(&dir).expect("create temporary BED directory");
        let prefix = dir.join("tiny");
        fs::write(
            prefix.with_extension("bed"),
            [0x6c, 0x1b, 0x01, 0x78, 0x02, 0x00, 0x00],
        )
        .expect("write BED fixture");

        let mut reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("open packed source");
        let sample_indices = [1usize, 2, 4];
        let mut packed = [0xaau8; 4];
        let mut stats = [crate::pgen::GenotypeRowStats::default(); 2];
        let mut keep = [true; 2];
        assert_eq!(
            reader
                .read_source_block_with_qc(
                    0,
                    2,
                    crate::pgen::GenotypeSampleSelection::include(&sample_indices),
                    &mut packed,
                    &mut stats,
                    &mut keep,
                    &|source_idx, _| source_idx == 1,
                )
                .unwrap(),
            2
        );
        assert_eq!(stats[0].sample_count, 3);
        assert_eq!(stats[0].missing_count, 0);
        assert_eq!(stats[0].heterozygote_count, 2);
        assert_eq!(stats[0].homozygous_alt_count, 1);
        assert_eq!(
            stats[1],
            crate::pgen::GenotypeRowStats {
                sample_count: 3,
                ..crate::pgen::GenotypeRowStats::default()
            }
        );
        assert_eq!(keep, [false, true]);
        assert_eq!(packed, [0, 0, 0, 0]);

        let mut exclude_reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("reopen packed source");
        let excluded = [0usize, 3];
        let mut exclude_packed = [0xaau8; 4];
        let mut exclude_stats = [crate::pgen::GenotypeRowStats::default(); 2];
        let mut exclude_keep = [true; 2];
        exclude_reader
            .read_source_block_with_qc(
                0,
                2,
                crate::pgen::GenotypeSampleSelection::Exclude(&excluded),
                &mut exclude_packed,
                &mut exclude_stats,
                &mut exclude_keep,
                &|source_idx, _| source_idx == 1,
            )
            .unwrap();
        assert_eq!(exclude_stats, stats);
        assert_eq!(exclude_keep, keep);
        assert_eq!(exclude_packed, packed);

        let mut stats_reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("reopen packed source");
        let mut stats_only = [crate::pgen::GenotypeRowStats::default(); 2];
        let mut stats_only_keep = [false; 2];
        stats_reader
            .read_source_block_qc_only(
                0,
                2,
                crate::pgen::GenotypeSampleSelection::include(&sample_indices),
                &mut stats_only,
                &mut stats_only_keep,
                &|source_idx, _| source_idx == 1,
            )
            .unwrap();
        assert_eq!(stats_only, stats);
        assert_eq!(stats_only_keep, keep);

        let mut float_reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("reopen BED source");
        let mut values_f32 = [0.0_f32; 10];
        assert_eq!(
            float_reader
                .read_source_block_f32(0, 2, &mut values_f32)
                .unwrap(),
            2
        );
        assert_eq!(
            values_f32,
            [0.0, 1.0, 2.0, -9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        let mut double_reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("reopen BED source");
        let mut values_f64 = [0.0_f64; 10];
        assert_eq!(
            double_reader
                .read_source_block_f64(0, 2, &mut values_f64)
                .unwrap(),
            2
        );
        assert_eq!(
            values_f64,
            [0.0, 1.0, 2.0, -9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        fs::remove_dir_all(dir).expect("remove temporary BED directory");
    }

    #[test]
    fn lm_lmm_pread_switch_defaults_on_and_accepts_explicit_disable() {
        assert!(choose_lm_lmm_bed_pread(None));
        assert!(choose_lm_lmm_bed_pread(Some("1")));
        assert!(choose_lm_lmm_bed_pread(Some(" true ")));
        assert!(!choose_lm_lmm_bed_pread(Some("0")));
        assert!(!choose_lm_lmm_bed_pread(Some(" off ")));
    }

    #[test]
    fn pread_window_returns_the_same_raw_rows_as_mmap_window() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_window_pread_{stamp}"));
        fs::create_dir_all(&dir).expect("create temporary BED directory");
        let prefix = dir.join("tiny");
        fs::write(
            prefix.with_extension("bed"),
            [0x6c, 0x1b, 0x01, 0x31, 0x72, 0xB4, 0xA5],
        )
        .expect("write BED fixture");
        fs::write(
            prefix.with_extension("fam"),
            "1 1 0 0 1 -9\n2 2 0 0 1 -9\n3 3 0 0 1 -9\n4 4 0 0 1 -9\n",
        )
        .expect("write FAM fixture");

        let mut mmap =
            WindowedBedMatrix::open(prefix.to_str().unwrap(), 1).expect("open mmap BED window");
        let mut pread = WindowedBedMatrix::open_pread(prefix.to_str().unwrap(), 1)
            .expect("open positioned-read BED window");
        assert_eq!(
            mmap.read_source_range(1, 4).unwrap(),
            pread.read_source_range(1, 4).unwrap()
        );
        let mut mmap_rows = Vec::new();
        let mut pread_rows = Vec::new();
        let mmap_block = mmap
            .prepare_source_rows(&[0, 2, 3], &mut mmap_rows)
            .unwrap()
            .to_vec();
        let pread_block = pread
            .prepare_source_rows(&[0, 2, 3], &mut pread_rows)
            .unwrap()
            .to_vec();
        assert_eq!(pread_rows, mmap_rows);
        assert_eq!(pread_block, mmap_block);

        fs::remove_dir_all(dir).expect("remove temporary BED directory");
    }
}

// ---------------------------------------------------------------------------
// WindowedBedMatrix — bounded mmap or positioned-read blocks for BED scans
// ---------------------------------------------------------------------------

/// Use positioned reads for sequential LM/LMM BED scans by default.
/// Set `JX_LM_LMM_BED_PREAD=0` to retain the windowed mmap reader.
pub(crate) fn choose_lm_lmm_bed_pread(value: Option<&str>) -> bool {
    !matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0" | "false" | "no" | "off")
    )
}

#[allow(dead_code)]
pub struct WindowedBedMatrix {
    file: std::fs::File,
    n_samples_full: usize,
    bytes_per_snp: usize,
    bed_len: usize,
    mmap: Option<Mmap>,
    mmap_offset: usize,
    window_start_snp: usize,
    window_len_snps: usize,
    target_window_snps: usize,
    pread: bool,
    scratch: Vec<u8>,
    pread_buffer: Vec<u8>,
}

#[allow(dead_code)]
impl WindowedBedMatrix {
    pub fn open(prefix: &str, mmap_window_mb: usize) -> Result<Self, String> {
        Self::open_inner(prefix, mmap_window_mb, false)
    }

    pub fn open_pread(prefix: &str, window_mb: usize) -> Result<Self, String> {
        Self::open_inner(prefix, window_mb, true)
    }

    pub fn open_lm_lmm(prefix: &str, window_mb: usize) -> Result<Self, String> {
        if choose_lm_lmm_bed_pread(std::env::var("JX_LM_LMM_BED_PREAD").ok().as_deref()) {
            Self::open_pread(prefix, window_mb)
        } else {
            Self::open(prefix, window_mb)
        }
    }

    fn open_inner(prefix: &str, mmap_window_mb: usize, pread: bool) -> Result<Self, String> {
        if mmap_window_mb == 0 {
            return Err("mmap_window_mb must be > 0".to_string());
        }
        let bed_prefix = normalize_plink_prefix(prefix);
        let n_samples_full = gfcore::read_fam(&bed_prefix)
            .map_err(|e| e.to_string())?
            .len();
        if n_samples_full == 0 {
            return Err("no samples found in BED".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let bed_path = format!("{bed_prefix}.bed");
        let file = std::fs::File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        let mut header = [0u8; BED_HEADER_LEN];
        read_file_exact_at(&file, &mut header, 0)
            .map_err(|e| format!("read BED header {bed_path}: {e}"))?;
        if header != [0x6C, 0x1B, 0x01] {
            return Err("only SNP-major BED supported".to_string());
        }
        let bed_len = file
            .metadata()
            .map_err(|e| format!("metadata {bed_path}: {e}"))?
            .len() as usize;
        if bed_len < BED_HEADER_LEN {
            return Err("BED too small".to_string());
        }
        let payload_len = bed_len - BED_HEADER_LEN;
        if bytes_per_snp == 0 || payload_len % bytes_per_snp != 0 {
            return Err(format!(
                "invalid BED payload length: data_len={payload_len}, bytes_per_snp={bytes_per_snp}"
            ));
        }
        let window_bytes = mmap_window_mb.saturating_mul(1024 * 1024);
        let target_window_snps = std::cmp::max(1, window_bytes / bytes_per_snp);
        let (mmap, mmap_offset, window_len_snps) = if pread {
            (None, 0, 0)
        } else {
            let (mmap, mmap_offset, window_len_snps) =
                Self::map_window(&file, bed_len, 0, target_window_snps, bytes_per_snp)?;
            (Some(mmap), mmap_offset, window_len_snps)
        };
        Ok(Self {
            file,
            n_samples_full,
            bytes_per_snp,
            bed_len,
            mmap,
            mmap_offset,
            window_start_snp: 0,
            window_len_snps,
            target_window_snps,
            pread,
            scratch: Vec::new(),
            pread_buffer: Vec::new(),
        })
    }

    /// Create an independent mmap/window cursor over the same BED payload.
    ///
    /// The duplicate is used by streamed Bayesian double buffering: one
    /// cursor belongs to the decode worker while the caller consumes the
    /// previously decoded block.  The file and mmap are independent, but the
    /// underlying BED pages remain shared by the operating system.
    pub fn duplicate(&self) -> Result<Self, String> {
        let file = self.file.try_clone().map_err(|e| e.to_string())?;
        if self.pread {
            return Ok(Self {
                file,
                n_samples_full: self.n_samples_full,
                bytes_per_snp: self.bytes_per_snp,
                bed_len: self.bed_len,
                mmap: None,
                mmap_offset: 0,
                window_start_snp: 0,
                window_len_snps: 0,
                target_window_snps: self.target_window_snps,
                pread: true,
                scratch: Vec::new(),
                pread_buffer: Vec::new(),
            });
        }
        // Every Bayesian scan starts at marker zero.  Do not inherit the
        // caller's last window (which is usually near the end of the BED
        // file), otherwise the worker would immediately remap before its
        // first block.
        let (mmap, mmap_offset, window_len_snps) = Self::map_window(
            &file,
            self.bed_len,
            0,
            self.target_window_snps,
            self.bytes_per_snp,
        )?;
        Ok(Self {
            file,
            n_samples_full: self.n_samples_full,
            bytes_per_snp: self.bytes_per_snp,
            bed_len: self.bed_len,
            mmap: Some(mmap),
            mmap_offset,
            window_start_snp: 0,
            window_len_snps,
            target_window_snps: self.target_window_snps,
            pread: false,
            scratch: Vec::new(),
            pread_buffer: Vec::new(),
        })
    }

    #[inline]
    pub fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }

    #[inline]
    pub fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }

    #[inline]
    pub fn n_source_snps(&self) -> usize {
        self.bed_len
            .saturating_sub(BED_HEADER_LEN)
            .saturating_div(self.bytes_per_snp.max(1))
    }

    fn map_window(
        file: &std::fs::File,
        bed_len: usize,
        start_snp: usize,
        window_snps: usize,
        bytes_per_snp: usize,
    ) -> Result<(Mmap, usize, usize), String> {
        let data_len = bed_len
            .checked_sub(BED_HEADER_LEN)
            .ok_or_else(|| "BED too small".to_string())?;
        let max_snps = data_len / bytes_per_snp;
        if start_snp >= max_snps {
            return Err("window start SNP out of range".to_string());
        }
        let remaining_snps = max_snps - start_snp;
        let map_snps = remaining_snps.min(window_snps.max(1));
        let desired_offset = BED_HEADER_LEN + start_snp.saturating_mul(bytes_per_snp);
        let page_size = system_page_size();
        let aligned_offset = desired_offset / page_size * page_size;
        let leading = desired_offset - aligned_offset;
        let desired_len = leading + map_snps.saturating_mul(bytes_per_snp);
        let max_len = bed_len - aligned_offset;
        let map_len = desired_len.min(max_len);
        let mmap = unsafe {
            MmapOptions::new()
                .offset(aligned_offset as u64)
                .len(map_len)
                .map(file)
                .map_err(|e| e.to_string())?
        };
        Ok((mmap, aligned_offset, map_snps))
    }

    fn remap_window_for_range(&mut self, start_snp: usize, end_snp: usize) -> Result<(), String> {
        if end_snp <= start_snp {
            return Err("invalid BED window range".to_string());
        }
        let span_snps = end_snp - start_snp;
        let window_snps = self.target_window_snps.max(span_snps);
        let (mmap, mmap_offset, window_len_snps) = Self::map_window(
            &self.file,
            self.bed_len,
            start_snp,
            window_snps,
            self.bytes_per_snp,
        )?;
        self.mmap = Some(mmap);
        self.mmap_offset = mmap_offset;
        self.window_start_snp = start_snp;
        self.window_len_snps = window_len_snps;
        Ok(())
    }

    pub fn read_source_range(&mut self, start: usize, end: usize) -> Result<&[u8], String> {
        if end <= start {
            return Ok(&[]);
        }
        if self.pread {
            if end > self.n_source_snps() {
                return Err(format!(
                    "BED positioned-read range out of bounds: start={start}, end={end}, n_snps={}",
                    self.n_source_snps()
                ));
            }
            let len = (end - start).saturating_mul(self.bytes_per_snp);
            self.pread_buffer.resize(len, 0);
            let offset =
                BED_HEADER_LEN.saturating_add(start.saturating_mul(self.bytes_per_snp)) as u64;
            read_file_exact_at(&self.file, &mut self.pread_buffer, offset)?;
            return Ok(&self.pread_buffer);
        }
        let window_end = self.window_start_snp + self.window_len_snps;
        if start < self.window_start_snp || end > window_end {
            self.remap_window_for_range(start, end)?;
        }
        let desired_offset = BED_HEADER_LEN + start.saturating_mul(self.bytes_per_snp);
        let local_offset = desired_offset
            .checked_sub(self.mmap_offset)
            .ok_or_else(|| "BED window local offset underflow".to_string())?;
        let len = (end - start).saturating_mul(self.bytes_per_snp);
        let end_offset = local_offset.saturating_add(len);
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| "internal error: mmap BED window is unavailable".to_string())?;
        if end_offset > mmap.len() {
            return Err(format!(
                "BED window slice out of bounds: local_offset={local_offset}, len={len}, mmap_len={}",
                mmap.len()
            ));
        }
        Ok(&mmap[local_offset..end_offset])
    }

    fn decode_sparse_rows_to_scratch(
        &mut self,
        source_indices: &[usize],
        rel_indices_out: &mut Vec<usize>,
    ) -> Result<&[u8], String> {
        let rows_here = source_indices.len();
        let bps = self.bytes_per_snp;
        let need = rows_here.saturating_mul(self.bytes_per_snp);
        if self.scratch.len() < need {
            self.scratch.resize(need, 0);
        }
        rel_indices_out.clear();
        for (local_idx, &src_idx) in source_indices.iter().enumerate() {
            let row_bytes = self.read_source_range(src_idx, src_idx + 1)?.to_vec();
            let dst = &mut self.scratch[local_idx * bps..(local_idx + 1) * bps];
            dst.copy_from_slice(row_bytes.as_slice());
            rel_indices_out.push(local_idx);
        }
        Ok(&self.scratch[..need])
    }

    pub fn prepare_source_rows(
        &mut self,
        source_indices: &[usize],
        rel_indices_out: &mut Vec<usize>,
    ) -> Result<&[u8], String> {
        if source_indices.is_empty() {
            rel_indices_out.clear();
            return Ok(&[]);
        }
        let src_start = source_indices[0];
        let src_end = source_indices[source_indices.len() - 1] + 1;
        let source_span = src_end - src_start;
        let zero_copy_ok = source_span <= source_indices.len().saturating_mul(8).max(1);
        if zero_copy_ok {
            rel_indices_out.clear();
            rel_indices_out.extend(source_indices.iter().map(|&src_idx| src_idx - src_start));
            self.read_source_range(src_start, src_end)
        } else {
            self.decode_sparse_rows_to_scratch(source_indices, rel_indices_out)
        }
    }

    pub fn prepare_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        rows_here: usize,
        rel_indices_out: &mut Vec<usize>,
    ) -> Result<&[u8], String> {
        let source_indices = stats.row_source_slice(row_start, rows_here);
        self.prepare_source_rows(source_indices, rel_indices_out)
    }
}

impl GenotypeMatrix for WindowedBedMatrix {
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }
    fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }
    fn packed_flat(&self) -> &[u8] {
        &[]
    }
    fn source_row_bytes(&self, _source_idx: usize) -> &[u8] {
        &[]
    }

    fn decode_additive_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let cols = if sample_identity {
            stats.n_samples_full
        } else {
            sample_idx.len()
        };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 {
            return Ok(());
        }
        let mut rel_indices = Vec::with_capacity(rows_here);
        let packed_slice = self.prepare_block(stats, row_start, rows_here, &mut rel_indices)?;
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed_slice,
            stats.bytes_per_snp,
            stats.n_samples_full,
            stats.row_flip_slice(row_start, rows_here),
            stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(rel_indices.as_slice()),
            0,
            out,
            code4_lut,
            pool,
        )
    }
}

// ---------------------------------------------------------------------------
// BedMmapMatrix
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct BedMmapMatrix {
    mmap: Arc<Mmap>,
    n_samples_full: usize,
    bytes_per_snp: usize,
    bed_prefix: String,
}

#[allow(dead_code)]
impl BedMmapMatrix {
    pub fn open(prefix: &str) -> Result<Self, String> {
        let bed_prefix = normalize_plink_prefix(prefix);
        let n_samples_full = gfcore::read_fam(&bed_prefix)
            .map_err(|e| e.to_string())?
            .len();
        if n_samples_full == 0 {
            return Err("no samples found in BED".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let bed_path = format!("{bed_prefix}.bed");
        let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
        if mmap.len() < 3 || mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err("invalid or non-SNP-major BED".to_string());
        }
        Ok(Self {
            mmap: Arc::new(mmap),
            n_samples_full,
            bytes_per_snp,
            bed_prefix,
        })
    }

    pub fn bed_prefix(&self) -> &str {
        &self.bed_prefix
    }

    /// Number of total SNP rows in the mmap'd BED.
    pub fn n_snps_total(&self) -> usize {
        let payload = self.mmap.len().saturating_sub(3);
        payload / self.bytes_per_snp
    }
}

impl GenotypeMatrix for BedMmapMatrix {
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }
    fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }

    fn packed_flat(&self) -> &[u8] {
        &self.mmap[3..]
    }

    fn source_row_bytes(&self, source_idx: usize) -> &[u8] {
        let offset = 3usize.saturating_add(source_idx.saturating_mul(self.bytes_per_snp));
        &self.mmap[offset..][..self.bytes_per_snp]
    }
}

// ---------------------------------------------------------------------------
// PackedBedMatrix
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct PackedBedMatrix {
    payload: Arc<[u8]>,
    n_samples_full: usize,
    bytes_per_snp: usize,
}

#[allow(dead_code)]
impl PackedBedMatrix {
    pub fn from_packed(payload: Arc<[u8]>, n_samples_full: usize) -> Result<Self, String> {
        let bytes_per_snp = n_samples_full.div_ceil(4);
        if payload.len() % bytes_per_snp != 0 {
            return Err("packed payload length is not a multiple of bytes_per_snp".to_string());
        }
        Ok(Self {
            payload,
            n_samples_full,
            bytes_per_snp,
        })
    }

    pub fn n_markers_contiguous(&self) -> usize {
        self.payload.len() / self.bytes_per_snp
    }
}

impl GenotypeMatrix for PackedBedMatrix {
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }
    fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }

    fn packed_flat(&self) -> &[u8] {
        &self.payload
    }

    /// For PackedBedMatrix, the payload IS the contiguous kept-marker data,
    /// so local_idx directly indexes into the payload.
    fn source_row_bytes(&self, local_idx: usize) -> &[u8] {
        let start = local_idx.saturating_mul(self.bytes_per_snp);
        &self.payload[start..start.saturating_add(self.bytes_per_snp)]
    }
}

// ---------------------------------------------------------------------------
// UnifiedInput
// ---------------------------------------------------------------------------

pub struct UnifiedInput<G: GenotypeMatrix> {
    pub matrix: G,
    pub stats: GlobalStats,
}

#[allow(dead_code)]
impl<G: GenotypeMatrix> UnifiedInput<G> {
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.stats.n_samples_full
    }

    #[inline]
    pub fn n_markers(&self) -> usize {
        self.stats.n_markers()
    }

    /// Block-decode a range of kept markers (additive, mean-imputed) into
    /// `out` which must be `rows_here × n_samples` f32 contiguous.
    ///
    /// `sample_idx` / `sample_identity` control per-sample byte-index
    /// mapping.  When all kept samples are included in identity order, pass
    /// `(&[], true)`.
    pub fn decode_additive_block(
        &self,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let cols = if sample_identity {
            self.n_samples()
        } else {
            sample_idx.len()
        };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 {
            return Ok(());
        }
        let code4_lut = &packed_byte_lut().code4;
        let row_source_indices = self.stats.row_source_slice(row_start, rows_here);
        decode_mean_imputed_additive_packed_block_rows_f32(
            self.matrix.packed_flat(),
            self.stats.bytes_per_snp,
            self.stats.n_samples_full,
            self.stats.row_flip_slice(row_start, rows_here),
            self.stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(row_source_indices),
            0,
            out,
            code4_lut,
            pool,
        )
    }

    /// Decode from a pre-read packed slice (used by streaming backends).
    /// `packed_slice` covers the source range for this block; `rel_indices`
    /// are source-index offsets relative to the start of `packed_slice`.
    #[inline]
    pub fn decode_additive_block_from_slice(
        &self,
        packed_slice: &[u8],
        rel_indices: &[usize],
        row_start: usize,
        rows_here: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed_slice,
            self.stats.bytes_per_snp,
            self.stats.n_samples_full,
            self.stats.row_flip_slice(row_start, rows_here),
            self.stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(rel_indices),
            0,
            out,
            code4_lut,
            pool,
        )
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub fn compute_global_stats_mmap(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    sample_indices: Option<&[usize]>,
    threads: usize,
) -> Result<GlobalStats, String> {
    let meta = compute_bed_filter_meta(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        sample_indices,
        BedFilterMetaMode::FullDecodeMeta,
        threads,
    )?;
    Ok(GlobalStats {
        maf: meta.maf,
        miss: meta.miss,
        row_flip: meta.row_flip,
        row_source_indices: meta.row_source_indices,
        site_keep: meta.site_keep,
        n_samples_full: meta.n_samples_full,
        n_markers_total: meta.n_markers_total,
        bytes_per_snp: meta.bytes_per_snp,
    })
}

#[allow(dead_code)]
pub fn open_bed_mmap_unified(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    sample_indices: Option<&[usize]>,
    threads: usize,
) -> Result<UnifiedInput<BedMmapMatrix>, String> {
    let stats = compute_global_stats_mmap(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        sample_indices,
        threads,
    )?;
    let matrix = BedMmapMatrix::open(prefix)?;
    Ok(UnifiedInput { matrix, stats })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
#[allow(dead_code)]
fn normalize_plink_prefix(prefix: &str) -> String {
    let mut out = prefix.trim().to_string();
    let lower = out.to_ascii_lowercase();
    for suffix in [".bed", ".bim", ".fam", ".pgen", ".pvar", ".psam"] {
        if lower.ends_with(suffix) {
            out.truncate(out.len().saturating_sub(suffix.len()));
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Centered decode (zero-mean, used by LMM / FastLMM / FvLMM)
// ---------------------------------------------------------------------------

/// Centered block decode for any GenotypeMatrix backend.
/// Uses the shared `crate::decode` packed centered-decode path internally.
#[allow(dead_code)]
pub fn decode_centered_block_unified<G: GenotypeMatrix>(
    matrix: &G,
    stats: &GlobalStats,
    row_start: usize,
    gm: PackedGeneticModel,
    out: &mut [f32],
    sample_idx: &[usize],
    sample_identity: bool,
) -> Result<(), String> {
    let n = if sample_identity {
        stats.n_samples_full
    } else {
        sample_idx.len()
    };
    let rows_here = out.len().saturating_div(n.max(1));
    if rows_here == 0 {
        return Ok(());
    }
    decode_centered_block_packed_model_f32(
        matrix.packed_flat(),
        stats.bytes_per_snp,
        stats.row_flip_slice(row_start, rows_here),
        stats.row_maf_slice(row_start, rows_here),
        Some(stats.row_source_slice(row_start, rows_here)),
        row_start,
        rows_here,
        n,
        gm,
        sample_identity,
        None,
        None,
        out,
    );
    Ok(())
}
