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
