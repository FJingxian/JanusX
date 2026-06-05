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

use memmap2::Mmap;

use crate::bedmath::{decode_mean_imputed_additive_packed_block_rows_f32, packed_byte_lut};
use crate::gfcore;
use crate::gfreader::{
    count_packed_row_counts, count_packed_row_counts_selected,
    evaluate_packed_row_keep_and_flip, packed_row_stats_from_counts,
};
use crate::stats_common::get_cached_pool;

/// Per-marker statistics computed once at load time.
///
/// All vectors are in **kept-marker order** (i.e. after MAF/missing/het
/// filtering).  `site_keep` is the only field indexed in the **original**
/// SNP order.
#[derive(Clone)]
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
        let cols = if sample_identity { stats.n_samples_full } else { sample_idx.len() };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 { return Ok(()); }
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            self.packed_flat(), stats.bytes_per_snp, stats.n_samples_full,
            stats.row_flip_slice(row_start, rows_here),
            stats.row_maf_slice(row_start, rows_here),
            sample_idx, sample_identity,
            Some(stats.row_source_slice(row_start, rows_here)),
            0, out, code4_lut, pool,
        )
    }
}

// ---------------------------------------------------------------------------
// StreamingBedMatrix — chunked file I/O, no mmap
// ---------------------------------------------------------------------------

/// Reads BED rows on-demand via `pread` (or `seek`+`read`) instead of
/// mmap-ing the whole file.  Dramatically reduces RSS for large BEDs
/// when only sequential access is needed (e.g. GWAS scan).
pub struct StreamingBedMatrix {
    file: std::fs::File,
    n_samples_full: usize,
    bytes_per_snp: usize,
    /// Reusable read buffer sized for one block's worth of packed rows.
    buf: Vec<u8>,
    /// Byte range currently held in `buf` (start, end) in source-index units.
    buf_range: (usize, usize),
}

impl StreamingBedMatrix {
    /// Open a BED prefix for streaming reads.
    pub fn open(prefix: &str, block_rows: usize) -> Result<Self, String> {
        let bed_prefix = normalize_plink_prefix(prefix);
        let n_samples_full = crate::gfcore::read_fam(&bed_prefix)
            .map_err(|e| e.to_string())?.len();
        if n_samples_full == 0 {
            return Err("no samples found in BED".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let bed_path = format!("{bed_prefix}.bed");
        let file = std::fs::File::open(&bed_path)
            .map_err(|e| format!("open {bed_path}: {e}"))?;
        // 3-byte BED header is skipped via offset in read_rows.
        let buf = vec![0u8; block_rows.saturating_mul(bytes_per_snp).max(1)];
        Ok(Self { file, n_samples_full, bytes_per_snp, buf, buf_range: (usize::MAX, 0) })
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
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(&mut self.buf[..need], offset)
                .map_err(|e| format!("read BED rows [{start}..{end}): {e}"))?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            self.file.seek(SeekFrom::Start(offset))
                .map_err(|e| format!("BED seek: {e}"))?;
            self.file.read_exact(&mut self.buf[..need])
                .map_err(|e| format!("BED read: {e}"))?;
        }
        self.buf_range = (start, end);
        Ok(&self.buf[..need])
    }
}

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
    fn n_samples_full(&self) -> usize { self.n_samples_full }
    fn bytes_per_snp(&self) -> usize { self.bytes_per_snp }
    fn packed_flat(&self) -> &[u8] {
        &self.buf[..(self.buf_range.1.saturating_sub(self.buf_range.0).saturating_mul(self.bytes_per_snp))]
    }
    fn source_row_bytes(&self, _source_idx: usize) -> &[u8] { &[] }

    fn decode_additive_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let cols = if sample_identity { stats.n_samples_full } else { sample_idx.len() };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 { return Ok(()); }
        let mut rel_indices = Vec::with_capacity(rows_here);
        let packed_slice = self.prepare_block(stats, row_start, rows_here, &mut rel_indices)?;
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed_slice, stats.bytes_per_snp, stats.n_samples_full,
            stats.row_flip_slice(row_start, rows_here),
            stats.row_maf_slice(row_start, rows_here),
            sample_idx, sample_identity, Some(&rel_indices),
            0, out, code4_lut, pool,
        )
    }
}

// ---------------------------------------------------------------------------
// BedMmapMatrix
// ---------------------------------------------------------------------------

pub struct BedMmapMatrix {
    mmap: Arc<Mmap>,
    n_samples_full: usize,
    bytes_per_snp: usize,
    bed_prefix: String,
}

impl BedMmapMatrix {
    pub fn open(prefix: &str) -> Result<Self, String> {
        let bed_prefix = normalize_plink_prefix(prefix);
        let n_samples_full =
            gfcore::read_fam(&bed_prefix).map_err(|e| e.to_string())?.len();
        if n_samples_full == 0 {
            return Err("no samples found in BED".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let bed_path = format!("{bed_prefix}.bed");
        let bed_file =
            File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&bed_file) }
            .map_err(|e| format!("mmap {bed_path}: {e}"))?;
        if mmap.len() < 3 || mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err("invalid or non-SNP-major BED".to_string());
        }
        Ok(Self { mmap: Arc::new(mmap), n_samples_full, bytes_per_snp, bed_prefix })
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
    fn n_samples_full(&self) -> usize { self.n_samples_full }
    fn bytes_per_snp(&self) -> usize { self.bytes_per_snp }

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

pub struct PackedBedMatrix {
    payload: Arc<[u8]>,
    n_samples_full: usize,
    bytes_per_snp: usize,
}

impl PackedBedMatrix {
    pub fn from_packed(
        payload: Arc<[u8]>,
        n_samples_full: usize,
    ) -> Result<Self, String> {
        let bytes_per_snp = n_samples_full.div_ceil(4);
        if payload.len() % bytes_per_snp != 0 {
            return Err("packed payload length is not a multiple of bytes_per_snp".to_string());
        }
        Ok(Self { payload, n_samples_full, bytes_per_snp })
    }

    pub fn n_markers_contiguous(&self) -> usize {
        self.payload.len() / self.bytes_per_snp
    }
}

impl GenotypeMatrix for PackedBedMatrix {
    fn n_samples_full(&self) -> usize { self.n_samples_full }
    fn bytes_per_snp(&self) -> usize { self.bytes_per_snp }

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
    let n_samples_full = gfcore::read_fam(&bed_prefix).map_err(|e| e.to_string())?.len();
    let bytes_per_snp = n_samples_full.div_ceil(4);

    let bed_path = format!("{bed_prefix}.bed");
    let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
    let mmap =
        unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
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
        .map(|s| s.is_empty() || (s.len() == n_samples_full && s.iter().enumerate().all(|(i, &sid)| sid == i)))
        .unwrap_or(true);
    let stats_n = if stats_identity { n_samples_full } else { sample_indices.unwrap().len() };
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
                        count_packed_row_counts_selected(row, n_samples_full, si_ref.unwrap())
                    };
                    let non_missing = stats_n.saturating_sub(missing);
                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                    let het_count = if apply_het { het } else { 0 };
                    let (miss, maf, _std) =
                        packed_row_stats_from_counts(stats_n, non_missing, alt_sum);
                    let (pass_num, flip) = evaluate_packed_row_keep_and_flip(
                        stats_n, non_missing, alt_sum, het_count,
                        maf_threshold, max_missing_rate, apply_het, het_threshold,
                    );
                    let pass_snp = if let Some(ref sites) = sites_all {
                        is_simple_snp_allele(&sites[i].ref_allele)
                            && is_simple_snp_allele(&sites[i].alt_allele)
                    } else {
                        true
                    };
                    (pass_num && pass_snp, (pass_num && pass_snp) && flip, miss, maf)
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
        prefix, maf_threshold, max_missing_rate, het_threshold, snps_only,
        sample_indices, threads,
    )?;
    let matrix = BedMmapMatrix::open(prefix)?;
    Ok(UnifiedInput { matrix, stats })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn normalize_plink_prefix(prefix: &str) -> String {
    let mut out = prefix.trim().to_string();
    let lower = out.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        out.truncate(out.len().saturating_sub(4));
    }
    out
}

#[inline]
fn is_simple_snp_allele(allele: &str) -> bool {
    matches!(allele, "A" | "C" | "G" | "T" | "a" | "c" | "g" | "t")
}

// ---------------------------------------------------------------------------
// PackedGeneticModel — shared model enum for centered decode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum PackedGeneticModel { Add, Dom, Rec, Het }

// ---------------------------------------------------------------------------
// Centered decode (zero-mean, used by LMM / FastLMM / FvLMM)
// ---------------------------------------------------------------------------

/// Centered block decode for any GenotypeMatrix backend.
/// Uses `crate::lmm_scan::decode_centered_block_packed_f32` internally.
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
    let n = if sample_identity { stats.n_samples_full } else { sample_idx.len() };
    let rows_here = out.len().saturating_div(n.max(1));
    if rows_here == 0 { return Ok(()); }
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
