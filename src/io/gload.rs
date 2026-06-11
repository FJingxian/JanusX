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
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};

use crate::bedmath::{decode_mean_imputed_additive_packed_block_rows_f32, packed_byte_lut};
use crate::gfcore;
use crate::gfreader::{
    count_packed_row_counts, count_packed_row_counts_selected_with_excluded,
    evaluate_packed_row_keep_and_flip, packed_row_stats_from_counts,
    precompute_excluded_sample_indices,
};
use crate::stats_common::get_cached_pool;

const BED_HEADER_LEN: usize = 3;

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
fn read_file_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
        .map_err(|e| format!("BED positioned read: {e}"))
}

#[cfg(windows)]
fn read_file_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
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
fn read_file_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
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
        self.maf.len()
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

// ---------------------------------------------------------------------------
// WindowedBedMatrix — remapped mmap windows for sequential BED scans
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct WindowedBedMatrix {
    file: std::fs::File,
    n_samples_full: usize,
    bytes_per_snp: usize,
    bed_len: usize,
    mmap: Mmap,
    mmap_offset: usize,
    window_start_snp: usize,
    window_len_snps: usize,
    target_window_snps: usize,
    scratch: Vec<u8>,
}

#[allow(dead_code)]
impl WindowedBedMatrix {
    pub fn open(prefix: &str, mmap_window_mb: usize) -> Result<Self, String> {
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
        let (mmap, mmap_offset, window_len_snps) =
            Self::map_window(&file, bed_len, 0, target_window_snps, bytes_per_snp)?;
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
            scratch: Vec::new(),
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
        self.mmap = mmap;
        self.mmap_offset = mmap_offset;
        self.window_start_snp = start_snp;
        self.window_len_snps = window_len_snps;
        Ok(())
    }

    pub fn read_source_range(&mut self, start: usize, end: usize) -> Result<&[u8], String> {
        if end <= start {
            return Ok(&[]);
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
        if end_offset > self.mmap.len() {
            return Err(format!(
                "BED window slice out of bounds: local_offset={local_offset}, len={len}, mmap_len={}",
                self.mmap.len()
            ));
        }
        Ok(&self.mmap[local_offset..end_offset])
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

    pub fn prepare_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        rows_here: usize,
        rel_indices_out: &mut Vec<usize>,
    ) -> Result<&[u8], String> {
        let source_indices = stats.row_source_slice(row_start, rows_here);
        let src_start = source_indices[0];
        let src_end = source_indices[rows_here - 1] + 1;
        let source_span = src_end - src_start;
        let zero_copy_ok = source_span <= rows_here.saturating_mul(8).max(1);
        if zero_copy_ok {
            rel_indices_out.clear();
            rel_indices_out.extend(source_indices.iter().map(|&src_idx| src_idx - src_start));
            self.read_source_range(src_start, src_end)
        } else {
            self.decode_sparse_rows_to_scratch(source_indices, rel_indices_out)
        }
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
    let bed_prefix = normalize_plink_prefix(prefix);
    let n_samples_full = gfcore::read_fam(&bed_prefix)
        .map_err(|e| e.to_string())?
        .len();
    let bytes_per_snp = n_samples_full.div_ceil(4);

    let bed_path = format!("{bed_prefix}.bed");
    let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
    let mmap = unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
    if mmap.len() < 3 || mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
        return Err("invalid or non-SNP-major BED".to_string());
    }
    let packed_full = &mmap[3..];
    let n_markers_total = packed_full.len() / bytes_per_snp;
    if n_markers_total == 0 {
        return Err("BED contains no SNPs".to_string());
    }

    let sites_all = if snps_only {
        Some(gfcore::read_bim(&bed_prefix).map_err(|e| e.to_string())?)
    } else {
        None
    };

    let stats_identity = sample_indices
        .map(|s| {
            s.is_empty()
                || (s.len() == n_samples_full && s.iter().enumerate().all(|(i, &sid)| sid == i))
        })
        .unwrap_or(true);
    let stats_n = if stats_identity {
        n_samples_full
    } else {
        sample_indices.unwrap().len()
    };
    let stats_excluded_sample_indices = if stats_identity {
        None
    } else {
        precompute_excluded_sample_indices(n_samples_full, sample_indices.unwrap())
    };
    let apply_het = het_threshold > 0.0;
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;

    let keep_flip_stats: Vec<(bool, bool, f32, f32)> = {
        let si_ref = sample_indices;
        let run = || -> Vec<(bool, bool, f32, f32)> {
            use rayon::prelude::*;
            packed_full
                .par_chunks(bytes_per_snp)
                .enumerate()
                .map(|(i, row)| {
                    let (missing, het, hom_alt) = if stats_identity {
                        count_packed_row_counts(row, n_samples_full)
                    } else {
                        count_packed_row_counts_selected_with_excluded(
                            row,
                            n_samples_full,
                            si_ref.unwrap(),
                            stats_excluded_sample_indices.as_deref(),
                        )
                    };
                    let non_missing = stats_n.saturating_sub(missing);
                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                    let het_count = if apply_het { het } else { 0 };
                    let (miss, maf, _std) =
                        packed_row_stats_from_counts(stats_n, non_missing, alt_sum);
                    let (pass_num, flip) = evaluate_packed_row_keep_and_flip(
                        stats_n,
                        non_missing,
                        alt_sum,
                        het_count,
                        maf_threshold,
                        max_missing_rate,
                        apply_het,
                        het_threshold,
                    );
                    let pass_snp = if let Some(ref sites) = sites_all {
                        is_simple_snp_allele(&sites[i].ref_allele)
                            && is_simple_snp_allele(&sites[i].alt_allele)
                    } else {
                        true
                    };
                    (
                        pass_num && pass_snp,
                        (pass_num && pass_snp) && flip,
                        miss,
                        maf,
                    )
                })
                .collect()
        };
        if let Some(ref tp) = pool {
            tp.install(run)
        } else {
            run()
        }
    };

    let site_keep: Vec<bool> = keep_flip_stats.iter().map(|(k, _, _, _)| *k).collect();
    let kept_n = site_keep.iter().filter(|&&x| x).count();
    if kept_n == 0 {
        return Err("No SNPs left after filtering".to_string());
    }

    let mut maf_keep = Vec::with_capacity(kept_n);
    let mut miss_keep = Vec::with_capacity(kept_n);
    let mut row_flip_keep = Vec::with_capacity(kept_n);
    let mut row_source_indices = Vec::with_capacity(kept_n);
    for i in 0..n_markers_total {
        if site_keep[i] {
            let (_, flip, miss, maf) = keep_flip_stats[i];
            maf_keep.push(maf);
            miss_keep.push(miss);
            row_flip_keep.push(flip);
            row_source_indices.push(i);
        }
    }

    Ok(GlobalStats {
        maf: maf_keep,
        miss: miss_keep,
        row_flip: row_flip_keep,
        row_source_indices,
        site_keep,
        n_samples_full,
        n_markers_total,
        bytes_per_snp,
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
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        out.truncate(out.len().saturating_sub(4));
    }
    out
}

#[inline]
#[allow(dead_code)]
fn is_simple_snp_allele(allele: &str) -> bool {
    matches!(allele, "A" | "C" | "G" | "T" | "a" | "c" | "g" | "t")
}

// ---------------------------------------------------------------------------
// PackedGeneticModel — shared model enum for centered decode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum PackedGeneticModel {
    Add,
    Dom,
    Rec,
    Het,
}

// ---------------------------------------------------------------------------
// Centered decode (zero-mean, used by LMM / FastLMM / FvLMM)
// ---------------------------------------------------------------------------

/// Centered block decode for any GenotypeMatrix backend.
/// Uses `crate::lmm_scan::decode_centered_block_packed_f32` internally.
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
    use crate::lmm_scan::PackedGeneticModel as LmmModel;
    let n = if sample_identity {
        stats.n_samples_full
    } else {
        sample_idx.len()
    };
    let rows_here = out.len().saturating_div(n.max(1));
    if rows_here == 0 {
        return Ok(());
    }
    let gm_lmm = match gm {
        PackedGeneticModel::Add => LmmModel::Add,
        PackedGeneticModel::Dom => LmmModel::Dom,
        PackedGeneticModel::Rec => LmmModel::Rec,
        PackedGeneticModel::Het => LmmModel::Het,
    };
    crate::lmm_scan::decode_centered_block_packed_f32(
        matrix.packed_flat(),
        stats.bytes_per_snp,
        stats.row_flip_slice(row_start, rows_here),
        stats.row_maf_slice(row_start, rows_here),
        Some(stats.row_source_slice(row_start, rows_here)),
        row_start,
        rows_here,
        n,
        gm_lmm,
        sample_identity,
        None,
        None,
        out,
    );
    Ok(())
}
