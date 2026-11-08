#[cfg(unix)]
use memmap2::Advice;
use memmap2::Mmap;
use numpy::ndarray::Array1;
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use crate::bedmath::{packed_byte_lut, packed_pair_lut};
use crate::gfcore;
use crate::gfreader::{
    count_packed_row_counts as count_packed_row_counts_simd,
    count_packed_row_counts_selected_with_excluded, evaluate_packed_row_keep_and_flip,
    packed_row_stats_from_counts, SampleSubsetPlan,
};
use crate::gload::PackedBedSourceReader;
use crate::math_ld::{
    build_bitplanes_u64, compute_packed_row_stats, dot_nomiss_pair_bitplanes,
    dot_nomiss_pair_from_packed, r2_pairwise_complete_bitplanes, r2_pairwise_complete_from_packed,
    PackedRowStats,
};
use crate::stats_common::{get_cached_pool, map_err_string_to_py};

const LDSC_BITPLANE_MAX_MB_DEFAULT: u64 = 1024;
static PGEN_LDSC_TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug)]
enum LdscWindow {
    Variants(usize),
    Bp(i64),
    Cm(f64),
}

#[derive(Clone, Copy, Debug, Default)]
struct GstatsRequest {
    site_maf: bool,
    site_miss: bool,
    site_het: bool,
    individual_miss: bool,
    individual_het: bool,
}

impl GstatsRequest {
    #[inline]
    fn needs_site(self) -> bool {
        self.site_maf || self.site_miss || self.site_het
    }

    #[inline]
    fn needs_individual(self) -> bool {
        self.individual_miss || self.individual_het
    }
}

struct GstatsCombinedOutput {
    site_maf: Option<Vec<f32>>,
    site_miss: Option<Vec<f32>>,
    site_het: Option<Vec<f32>>,
    individual_miss: Option<Vec<f32>>,
    individual_het: Option<Vec<f32>>,
}

struct GstatsBlockResult {
    site_maf: Option<Vec<f32>>,
    site_miss: Option<Vec<f32>>,
    site_het: Option<Vec<f32>>,
    miss_ct: Option<Vec<u64>>,
    nonmiss_ct: Option<Vec<u64>>,
    het_ct: Option<Vec<u64>>,
}

/// Marker-level metadata reused by downstream GWAS loaders after
/// bitwise filtering on the raw BED rows.
#[derive(Clone, Debug)]
pub(crate) struct BedFilterMeta {
    pub maf: Vec<f32>,
    pub miss: Vec<f32>,
    pub het: Vec<f32>,
    pub row_flip: Vec<bool>,
    pub row_source_indices: Vec<usize>,
    pub site_keep: Vec<bool>,
    pub n_samples_full: usize,
    pub n_markers_total: usize,
    pub bytes_per_snp: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BedFilterMetaMode {
    FullDecodeMeta,
    SiteStatsOnly,
}

#[inline]
fn normalize_plink_prefix(p: &str) -> String {
    let s = p.trim();
    let low = s.to_ascii_lowercase();
    if low.ends_with(".bed") || low.ends_with(".bim") || low.ends_with(".fam") {
        return s[..s.len() - 4].to_string();
    }
    s.to_string()
}

#[inline]
fn normalize_chr_token(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 3 && t[..3].eq_ignore_ascii_case("chr") {
        t[3..].to_string()
    } else {
        t.to_string()
    }
}

#[inline]
fn is_simple_snp_allele(allele: &str) -> bool {
    matches!(allele, "A" | "C" | "G" | "T" | "a" | "c" | "g" | "t")
}

fn open_bed_mmap(prefix: &str) -> Result<(Mmap, usize, usize, usize), String> {
    let samples = gfcore::read_fam(prefix)?;
    let n_samples = samples.len();
    if n_samples == 0 {
        return Err("no samples found in PLINK input".to_string());
    }

    let bed_path = format!("{prefix}.bed");
    let bed_file = File::open(&bed_path).map_err(|e| format!("{bed_path}: {e}"))?;
    let mmap = unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("{bed_path}: {e}"))?;
    #[cfg(unix)]
    let _ = mmap.advise(Advice::Sequential);
    if mmap.len() < 3 {
        return Err(format!("{bed_path}: BED too small"));
    }
    if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
        return Err(format!(
            "{bed_path}: unsupported BED header (expect SNP-major 0x6C 0x1B 0x01)"
        ));
    }

    let bytes_per_snp = (n_samples + 3) / 4;
    let data_len = mmap.len() - 3;
    if bytes_per_snp == 0 || data_len % bytes_per_snp != 0 {
        return Err(format!(
            "{bed_path}: invalid payload length data_len={data_len}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    let n_snps = data_len / bytes_per_snp;
    if n_snps == 0 {
        return Err(format!("{bed_path}: no variant rows found"));
    }
    Ok((mmap, n_samples, n_snps, bytes_per_snp))
}

#[inline]
fn packed_site_rates(
    n_samples: usize,
    missing: usize,
    het_count: usize,
    hom_alt: usize,
) -> (f32, f32, f32) {
    let non_missing = n_samples.saturating_sub(missing);
    let miss_rate = (missing as f32) / (n_samples as f32);
    if non_missing == 0 {
        return (0.0_f32, miss_rate, 0.0_f32);
    }
    let alt_sum = het_count.saturating_add(hom_alt.saturating_mul(2));
    let p_alt = (alt_sum as f32) / (2.0_f32 * non_missing as f32);
    (
        p_alt.min(1.0_f32 - p_alt),
        miss_rate,
        (het_count as f32) / (non_missing as f32),
    )
}

#[inline]
fn accumulate_individual_row_counts(
    row: &[u8],
    code4_lut: &[[u8; 4]; 256],
    full_bytes: usize,
    rem: usize,
    miss_ct: &mut [u64],
    nonmiss_ct: &mut [u64],
    het_ct: &mut [u64],
) {
    let mut sample_idx = 0usize;
    for &b in row.iter().take(full_bytes) {
        let codes = &code4_lut[b as usize];
        for &code in codes.iter() {
            match code {
                0b01 => miss_ct[sample_idx] += 1,
                0b10 => {
                    nonmiss_ct[sample_idx] += 1;
                    het_ct[sample_idx] += 1;
                }
                0b00 | 0b11 => nonmiss_ct[sample_idx] += 1,
                _ => {}
            }
            sample_idx += 1;
        }
    }
    if rem > 0 {
        let codes = &code4_lut[row[full_bytes] as usize];
        for &code in codes.iter().take(rem) {
            match code {
                0b01 => miss_ct[sample_idx] += 1,
                0b10 => {
                    nonmiss_ct[sample_idx] += 1;
                    het_ct[sample_idx] += 1;
                }
                0b00 | 0b11 => nonmiss_ct[sample_idx] += 1,
                _ => {}
            }
            sample_idx += 1;
        }
    }
}

fn finalize_individual_rates(
    miss_ct: Vec<u64>,
    nonmiss_ct: Vec<u64>,
    het_ct: Vec<u64>,
    n_snps: usize,
    request: GstatsRequest,
) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut miss_rate = request
        .individual_miss
        .then(|| vec![0.0_f32; miss_ct.len()]);
    let mut het_rate = request.individual_het.then(|| vec![0.0_f32; miss_ct.len()]);
    let n_snps_f = n_snps as f32;
    for i in 0..miss_ct.len() {
        if let Some(dst) = miss_rate.as_mut() {
            dst[i] = (miss_ct[i] as f32) / n_snps_f;
        }
        if let Some(dst) = het_rate.as_mut() {
            if nonmiss_ct[i] > 0 {
                dst[i] = (het_ct[i] as f32) / (nonmiss_ct[i] as f32);
            }
        }
    }
    (miss_rate, het_rate)
}

fn compute_site_stats_core(
    packed_src: &[u8],
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    threads: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let mut maf = vec![0.0_f32; n_snps];
    let mut miss = vec![0.0_f32; n_snps];
    let mut het = vec![0.0_f32; n_snps];

    let mut run = || {
        maf.par_iter_mut()
            .zip(miss.par_iter_mut())
            .zip(het.par_iter_mut())
            .enumerate()
            .for_each(|(row_idx, ((maf_dst, miss_dst), het_dst))| {
                let row = &packed_src[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                let (missing, het_count, hom_alt) = count_packed_row_counts_simd(row, n_samples);
                let (maf_v, miss_v, het_v) =
                    packed_site_rates(n_samples, missing, het_count, hom_alt);
                *maf_dst = maf_v;
                *miss_dst = miss_v;
                *het_dst = het_v;
            });
    };
    if let Some(tp) = &pool {
        tp.install(&mut run);
    } else {
        run();
    }
    Ok((maf, miss, het))
}

fn compute_bed_filter_meta_core(
    packed_src: &[u8],
    n_samples_full: usize,
    n_markers_total: usize,
    bytes_per_snp: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    sites_all: Option<&[gfcore::SiteInfo]>,
    sample_indices: Option<&[usize]>,
    mode: BedFilterMetaMode,
    threads: usize,
) -> Result<BedFilterMeta, String> {
    let subset_plan = SampleSubsetPlan::from_optional_indices(n_samples_full, sample_indices);
    let stats_identity = subset_plan.is_identity();
    let stats_n = subset_plan.n_selected();
    let apply_het = het_threshold > 0.0;
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;

    if matches!(mode, BedFilterMetaMode::SiteStatsOnly)
        && !snps_only_for_light_mode(sites_all)
        && maf_threshold <= 0.0
        && max_missing_rate >= 1.0
        && !apply_het
    {
        let mut maf = vec![0.0_f32; n_markers_total];
        let mut miss = vec![0.0_f32; n_markers_total];
        let mut het = vec![0.0_f32; n_markers_total];
        let mut run = || {
            maf.par_iter_mut()
                .zip(miss.par_iter_mut())
                .zip(het.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((maf_dst, miss_dst), het_dst))| {
                    let row = &packed_src[i * bytes_per_snp..(i + 1) * bytes_per_snp];
                    let (missing, het_count, hom_alt) = if stats_identity {
                        count_packed_row_counts_simd(row, n_samples_full)
                    } else {
                        count_packed_row_counts_selected_with_excluded(
                            row,
                            n_samples_full,
                            subset_plan.selected().unwrap(),
                            subset_plan.excluded(),
                        )
                    };
                    let (maf_v, miss_v, het_v) =
                        packed_site_rates(stats_n, missing, het_count, hom_alt);
                    *maf_dst = maf_v;
                    *miss_dst = miss_v;
                    *het_dst = het_v;
                });
        };
        if let Some(ref tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return Ok(BedFilterMeta {
            maf,
            miss,
            het,
            row_flip: Vec::new(),
            row_source_indices: Vec::new(),
            site_keep: Vec::new(),
            n_samples_full,
            n_markers_total,
            bytes_per_snp,
        });
    }

    let keep_flip_stats: Vec<(bool, bool, f32, f32, f32)> = {
        let run = || -> Vec<(bool, bool, f32, f32, f32)> {
            packed_src
                .par_chunks(bytes_per_snp)
                .enumerate()
                .map(|(i, row)| {
                    let (missing, het, hom_alt) = if stats_identity {
                        count_packed_row_counts_simd(row, n_samples_full)
                    } else {
                        count_packed_row_counts_selected_with_excluded(
                            row,
                            n_samples_full,
                            subset_plan.selected().unwrap(),
                            subset_plan.excluded(),
                        )
                    };
                    let non_missing = stats_n.saturating_sub(missing);
                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                    let het_count = if apply_het { het } else { 0 };
                    let (miss, maf, _std) =
                        packed_row_stats_from_counts(stats_n, non_missing, alt_sum);
                    let (_, _, het_v) = packed_site_rates(stats_n, missing, het, hom_alt);
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
                    let pass_snp = if let Some(sites) = sites_all {
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
                        het_v,
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

    let site_keep: Vec<bool> = keep_flip_stats
        .iter()
        .map(|(keep, _, _, _, _)| *keep)
        .collect();
    let kept_n = site_keep.iter().filter(|&&keep| keep).count();
    if kept_n == 0 {
        return Err("No SNPs left after filtering".to_string());
    }

    let mut maf = Vec::with_capacity(kept_n);
    let mut miss = Vec::with_capacity(kept_n);
    let mut het = Vec::with_capacity(kept_n);
    let mut row_flip = Vec::with_capacity(kept_n);
    let mut row_source_indices = Vec::with_capacity(kept_n);
    for (row_idx, &(keep, flip, miss_v, maf_v, het_v)) in keep_flip_stats.iter().enumerate() {
        if keep {
            maf.push(maf_v);
            miss.push(miss_v);
            het.push(het_v);
            row_flip.push(flip);
            row_source_indices.push(row_idx);
        }
    }

    Ok(BedFilterMeta {
        maf,
        miss,
        het,
        row_flip,
        row_source_indices,
        site_keep,
        n_samples_full,
        n_markers_total,
        bytes_per_snp,
    })
}

pub(crate) fn compute_bed_filter_meta(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    sample_indices: Option<&[usize]>,
    mode: BedFilterMetaMode,
    threads: usize,
) -> Result<BedFilterMeta, String> {
    let bed_prefix = normalize_plink_prefix(prefix);
    let (mmap, n_samples_full, n_markers_total, bytes_per_snp) = open_bed_mmap(&bed_prefix)?;
    let packed_src = &mmap[3..];
    let sites_all = if snps_only {
        Some(gfcore::read_bim(&bed_prefix).map_err(|e| e.to_string())?)
    } else {
        None
    };
    compute_bed_filter_meta_core(
        packed_src,
        n_samples_full,
        n_markers_total,
        bytes_per_snp,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        sites_all.as_deref(),
        sample_indices,
        mode,
        threads,
    )
}

#[inline]
fn snps_only_for_light_mode(sites_all: Option<&[gfcore::SiteInfo]>) -> bool {
    sites_all.is_some()
}

fn compute_site_stats_legacy_from_prefix(
    prefix: &str,
    threads: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize), String> {
    let bed_prefix = normalize_plink_prefix(prefix);
    let (mmap, n_samples, n_snps, bytes_per_snp) = open_bed_mmap(&bed_prefix)?;
    let packed_src = &mmap[3..];
    let (maf, miss, het) =
        compute_site_stats_core(packed_src, n_samples, n_snps, bytes_per_snp, threads)?;
    Ok((maf, miss, het, n_samples, n_snps))
}

fn compute_site_stats_unified_from_prefix(
    prefix: &str,
    threads: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize), String> {
    let meta = compute_bed_filter_meta(
        prefix,
        0.0,
        1.0,
        0.0,
        false,
        None,
        BedFilterMetaMode::SiteStatsOnly,
        threads,
    )?;
    Ok((
        meta.maf,
        meta.miss,
        meta.het,
        meta.n_samples_full,
        meta.n_markers_total,
    ))
}

fn compute_joint_site_only_via_meta(
    prefix: &str,
    request: GstatsRequest,
    threads: usize,
) -> Result<(GstatsCombinedOutput, usize, usize), String> {
    let meta = compute_bed_filter_meta(
        prefix,
        0.0,
        1.0,
        0.0,
        false,
        None,
        BedFilterMetaMode::SiteStatsOnly,
        threads,
    )?;
    Ok((
        GstatsCombinedOutput {
            site_maf: request.site_maf.then_some(meta.maf),
            site_miss: request.site_miss.then_some(meta.miss),
            site_het: request.site_het.then_some(meta.het),
            individual_miss: None,
            individual_het: None,
        },
        meta.n_samples_full,
        meta.n_markers_total,
    ))
}

#[inline]
fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn compute_individual_stats_core(
    packed_src: &[u8],
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    threads: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let (miss_ct, nonmiss_ct, het_ct) = compute_individual_counts_core(
        packed_src,
        n_samples,
        n_snps,
        bytes_per_snp,
        None,
        threads,
    )?;
    let request = GstatsRequest {
        individual_miss: true,
        individual_het: true,
        ..GstatsRequest::default()
    };
    let (miss_rate, het_rate) =
        finalize_individual_rates(miss_ct, nonmiss_ct, het_ct, n_snps, request);
    let miss_rate = miss_rate.unwrap_or_else(|| vec![0.0_f32; n_samples]);
    let het_rate = het_rate.unwrap_or_else(|| vec![0.0_f32; n_samples]);
    Ok((miss_rate, het_rate))
}

fn compute_individual_counts_core(
    packed_src: &[u8],
    n_samples_full: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    sample_indices: Option<&[usize]>,
    threads: usize,
) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), String> {
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let code4_lut = &packed_byte_lut().code4;
    let full_bytes = n_samples_full / 4;
    let rem = n_samples_full % 4;
    let n_output_samples = sample_indices.map_or(n_samples_full, <[usize]>::len);
    let block_rows = 2048usize;
    let n_blocks = n_snps.div_ceil(block_rows);

    let run = || {
        (0..n_blocks)
            .into_par_iter()
            .fold(
                || {
                    (
                        vec![0u64; n_output_samples],
                        vec![0u64; n_output_samples],
                        vec![0u64; n_output_samples],
                    )
                },
                |mut acc, block_idx| {
                    let (ref mut miss_ct, ref mut nonmiss_ct, ref mut het_ct) = acc;
                    let row_start = block_idx * block_rows;
                    let row_end = std::cmp::min(n_snps, row_start + block_rows);
                    for row_idx in row_start..row_end {
                        let row =
                            &packed_src[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                        if let Some(indices) = sample_indices {
                            accumulate_individual_row_counts_selected(
                                row, indices, miss_ct, nonmiss_ct, het_ct,
                            );
                        } else {
                            accumulate_individual_row_counts(
                                row, code4_lut, full_bytes, rem, miss_ct, nonmiss_ct, het_ct,
                            );
                        }
                    }
                    acc
                },
            )
            .reduce(
                || {
                    (
                        vec![0u64; n_output_samples],
                        vec![0u64; n_output_samples],
                        vec![0u64; n_output_samples],
                    )
                },
                |mut left, right| {
                    let (ref mut miss_l, ref mut nonmiss_l, ref mut het_l) = left;
                    let (miss_r, nonmiss_r, het_r) = right;
                    for i in 0..n_output_samples {
                        miss_l[i] += miss_r[i];
                        nonmiss_l[i] += nonmiss_r[i];
                        het_l[i] += het_r[i];
                    }
                    left
                },
            )
    };

    let (miss_ct, nonmiss_ct, het_ct) = if let Some(tp) = &pool {
        tp.install(run)
    } else {
        run()
    };

    Ok((miss_ct, nonmiss_ct, het_ct))
}

fn compute_joint_stats_core(
    packed_src: &[u8],
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    request: GstatsRequest,
    threads: usize,
) -> Result<GstatsCombinedOutput, String> {
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let code4_lut = &packed_byte_lut().code4;
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let block_rows = 2048usize;
    let n_blocks = n_snps.div_ceil(block_rows);

    let run = || {
        (0..n_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let row_start = block_idx * block_rows;
                let row_end = std::cmp::min(n_snps, row_start + block_rows);
                let rows_in_block = row_end.saturating_sub(row_start);
                let mut site_maf = request.site_maf.then(|| vec![0.0_f32; rows_in_block]);
                let mut site_miss = request.site_miss.then(|| vec![0.0_f32; rows_in_block]);
                let mut site_het = request.site_het.then(|| vec![0.0_f32; rows_in_block]);
                let mut miss_ct = request.needs_individual().then(|| vec![0u64; n_samples]);
                let mut nonmiss_ct = request.needs_individual().then(|| vec![0u64; n_samples]);
                let mut het_ct = request.needs_individual().then(|| vec![0u64; n_samples]);

                for (local_row, row_idx) in (row_start..row_end).enumerate() {
                    let row = &packed_src[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                    let (missing, het_count, hom_alt) =
                        count_packed_row_counts_simd(row, n_samples);
                    if request.needs_site() {
                        let (maf_v, miss_v, het_v) =
                            packed_site_rates(n_samples, missing, het_count, hom_alt);
                        if let Some(dst) = site_maf.as_mut() {
                            dst[local_row] = maf_v;
                        }
                        if let Some(dst) = site_miss.as_mut() {
                            dst[local_row] = miss_v;
                        }
                        if let Some(dst) = site_het.as_mut() {
                            dst[local_row] = het_v;
                        }
                    }
                    if let (Some(miss_dst), Some(nonmiss_dst), Some(het_dst)) =
                        (miss_ct.as_mut(), nonmiss_ct.as_mut(), het_ct.as_mut())
                    {
                        accumulate_individual_row_counts(
                            row,
                            code4_lut,
                            full_bytes,
                            rem,
                            miss_dst,
                            nonmiss_dst,
                            het_dst,
                        );
                    }
                }

                GstatsBlockResult {
                    site_maf,
                    site_miss,
                    site_het,
                    miss_ct,
                    nonmiss_ct,
                    het_ct,
                }
            })
            .collect::<Vec<GstatsBlockResult>>()
    };

    let block_results = if let Some(tp) = &pool {
        tp.install(run)
    } else {
        run()
    };

    let mut site_maf = request.site_maf.then(|| vec![0.0_f32; n_snps]);
    let mut site_miss = request.site_miss.then(|| vec![0.0_f32; n_snps]);
    let mut site_het = request.site_het.then(|| vec![0.0_f32; n_snps]);
    let mut total_miss_ct = request.needs_individual().then(|| vec![0u64; n_samples]);
    let mut total_nonmiss_ct = request.needs_individual().then(|| vec![0u64; n_samples]);
    let mut total_het_ct = request.needs_individual().then(|| vec![0u64; n_samples]);

    let mut row_start = 0usize;
    for block in block_results.into_iter() {
        if let Some(local) = block.site_maf {
            let row_end = row_start + local.len();
            if let Some(dst) = site_maf.as_mut() {
                dst[row_start..row_end].copy_from_slice(local.as_slice());
            }
        }
        if let Some(local) = block.site_miss {
            let row_end = row_start + local.len();
            if let Some(dst) = site_miss.as_mut() {
                dst[row_start..row_end].copy_from_slice(local.as_slice());
            }
        }
        if let Some(local) = block.site_het {
            let row_end = row_start + local.len();
            if let Some(dst) = site_het.as_mut() {
                dst[row_start..row_end].copy_from_slice(local.as_slice());
                row_start = row_end;
            }
        } else if let Some(dst) = site_maf.as_ref() {
            row_start += std::cmp::min(block_rows, dst.len().saturating_sub(row_start));
        } else if let Some(dst) = site_miss.as_ref() {
            row_start += std::cmp::min(block_rows, dst.len().saturating_sub(row_start));
        } else {
            row_start += std::cmp::min(block_rows, n_snps.saturating_sub(row_start));
        }

        if let Some(local) = block.miss_ct {
            if let Some(dst) = total_miss_ct.as_mut() {
                for (x, y) in dst.iter_mut().zip(local.into_iter()) {
                    *x += y;
                }
            }
        }
        if let Some(local) = block.nonmiss_ct {
            if let Some(dst) = total_nonmiss_ct.as_mut() {
                for (x, y) in dst.iter_mut().zip(local.into_iter()) {
                    *x += y;
                }
            }
        }
        if let Some(local) = block.het_ct {
            if let Some(dst) = total_het_ct.as_mut() {
                for (x, y) in dst.iter_mut().zip(local.into_iter()) {
                    *x += y;
                }
            }
        }
    }

    let (individual_miss, individual_het) = if let (Some(miss_ct), Some(nonmiss_ct), Some(het_ct)) =
        (total_miss_ct, total_nonmiss_ct, total_het_ct)
    {
        finalize_individual_rates(miss_ct, nonmiss_ct, het_ct, n_snps, request)
    } else {
        (None, None)
    };

    Ok(GstatsCombinedOutput {
        site_maf,
        site_miss,
        site_het,
        individual_miss,
        individual_het,
    })
}

fn validate_selection_indices(
    indices: Option<&[usize]>,
    upper_bound: usize,
    label: &str,
) -> Result<Option<Vec<usize>>, String> {
    let Some(indices) = indices else {
        return Ok(None);
    };
    if indices.is_empty() {
        return Err(format!("{label}_indices must not be empty"));
    }
    let mut previous = None;
    for &index in indices {
        if index >= upper_bound {
            return Err(format!(
                "{label}_indices contains out-of-range index {index} (size={upper_bound})"
            ));
        }
        if previous.is_some_and(|value| index <= value) {
            return Err(format!(
                "{label}_indices must be strictly increasing and unique"
            ));
        }
        previous = Some(index);
    }
    Ok(Some(indices.to_vec()))
}

#[inline]
fn accumulate_individual_row_counts_selected(
    row: &[u8],
    sample_indices: &[usize],
    miss_ct: &mut [u64],
    nonmiss_ct: &mut [u64],
    het_ct: &mut [u64],
) {
    for (output_idx, &sample_idx) in sample_indices.iter().enumerate() {
        let code = (row[sample_idx >> 2] >> ((sample_idx & 3) << 1)) & 0b11;
        match code {
            0b01 => miss_ct[output_idx] += 1,
            0b10 => {
                nonmiss_ct[output_idx] += 1;
                het_ct[output_idx] += 1;
            }
            0b00 | 0b11 => nonmiss_ct[output_idx] += 1,
            _ => unreachable!(),
        }
    }
}

fn compute_joint_stats_selected_core(
    packed_src: &[u8],
    n_samples_full: usize,
    n_snps_full: usize,
    bytes_per_snp: usize,
    request: GstatsRequest,
    sample_indices: Option<&[usize]>,
    site_indices: Option<&[usize]>,
    threads: usize,
) -> Result<GstatsCombinedOutput, String> {
    let sample_indices = validate_selection_indices(sample_indices, n_samples_full, "sample")?;
    let site_indices = validate_selection_indices(site_indices, n_snps_full, "site")?;
    if sample_indices.is_none() && site_indices.is_none() {
        return compute_joint_stats_core(
            packed_src,
            n_samples_full,
            n_snps_full,
            bytes_per_snp,
            request,
            threads,
        );
    }

    let n_samples = sample_indices
        .as_ref()
        .map_or(n_samples_full, |indices| indices.len());
    let n_snps = site_indices
        .as_ref()
        .map_or(n_snps_full, |indices| indices.len());
    let subset_plan = SampleSubsetPlan::from_optional_indices(
        n_samples_full,
        sample_indices.as_ref().map(Vec::as_slice),
    );
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let code4_lut = &packed_byte_lut().code4;
    let full_bytes = n_samples_full / 4;
    let rem = n_samples_full % 4;
    let block_rows = 2048usize;
    let n_blocks = n_snps.div_ceil(block_rows);

    let run = || {
        (0..n_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let output_start = block_idx * block_rows;
                let output_end = std::cmp::min(n_snps, output_start + block_rows);
                let rows_in_block = output_end - output_start;
                let mut site_maf = request.site_maf.then(|| vec![0.0_f32; rows_in_block]);
                let mut site_miss = request.site_miss.then(|| vec![0.0_f32; rows_in_block]);
                let mut site_het = request.site_het.then(|| vec![0.0_f32; rows_in_block]);
                let mut miss_ct = request.needs_individual().then(|| vec![0u64; n_samples]);
                let mut nonmiss_ct = request.needs_individual().then(|| vec![0u64; n_samples]);
                let mut het_ct = request.needs_individual().then(|| vec![0u64; n_samples]);

                for output_idx in output_start..output_end {
                    let source_idx = site_indices
                        .as_ref()
                        .map_or(output_idx, |indices| indices[output_idx]);
                    let row =
                        &packed_src[source_idx * bytes_per_snp..(source_idx + 1) * bytes_per_snp];
                    if request.needs_site() {
                        let (missing, het_count, hom_alt) = if subset_plan.is_identity() {
                            count_packed_row_counts_simd(row, n_samples_full)
                        } else {
                            count_packed_row_counts_selected_with_excluded(
                                row,
                                n_samples_full,
                                subset_plan.selected().unwrap(),
                                subset_plan.excluded(),
                            )
                        };
                        let (maf_value, miss_value, het_value) =
                            packed_site_rates(n_samples, missing, het_count, hom_alt);
                        let local_idx = output_idx - output_start;
                        if let Some(values) = site_maf.as_mut() {
                            values[local_idx] = maf_value;
                        }
                        if let Some(values) = site_miss.as_mut() {
                            values[local_idx] = miss_value;
                        }
                        if let Some(values) = site_het.as_mut() {
                            values[local_idx] = het_value;
                        }
                    }
                    if let (Some(miss), Some(nonmiss), Some(het)) =
                        (miss_ct.as_mut(), nonmiss_ct.as_mut(), het_ct.as_mut())
                    {
                        if let Some(indices) = sample_indices.as_ref() {
                            accumulate_individual_row_counts_selected(
                                row, indices, miss, nonmiss, het,
                            );
                        } else {
                            accumulate_individual_row_counts(
                                row, code4_lut, full_bytes, rem, miss, nonmiss, het,
                            );
                        }
                    }
                }
                (
                    output_start,
                    GstatsBlockResult {
                        site_maf,
                        site_miss,
                        site_het,
                        miss_ct,
                        nonmiss_ct,
                        het_ct,
                    },
                )
            })
            .collect::<Vec<_>>()
    };
    let blocks = if let Some(pool) = &pool {
        pool.install(run)
    } else {
        run()
    };

    let mut site_maf = request.site_maf.then(|| vec![0.0_f32; n_snps]);
    let mut site_miss = request.site_miss.then(|| vec![0.0_f32; n_snps]);
    let mut site_het = request.site_het.then(|| vec![0.0_f32; n_snps]);
    let mut total_miss = request.needs_individual().then(|| vec![0u64; n_samples]);
    let mut total_nonmiss = request.needs_individual().then(|| vec![0u64; n_samples]);
    let mut total_het = request.needs_individual().then(|| vec![0u64; n_samples]);
    for (output_start, block) in blocks {
        if let (Some(dst), Some(src)) = (site_maf.as_mut(), block.site_maf) {
            dst[output_start..output_start + src.len()].copy_from_slice(&src);
        }
        if let (Some(dst), Some(src)) = (site_miss.as_mut(), block.site_miss) {
            dst[output_start..output_start + src.len()].copy_from_slice(&src);
        }
        if let (Some(dst), Some(src)) = (site_het.as_mut(), block.site_het) {
            dst[output_start..output_start + src.len()].copy_from_slice(&src);
        }
        if let (Some(dst), Some(src)) = (total_miss.as_mut(), block.miss_ct) {
            for (left, right) in dst.iter_mut().zip(src) {
                *left += right;
            }
        }
        if let (Some(dst), Some(src)) = (total_nonmiss.as_mut(), block.nonmiss_ct) {
            for (left, right) in dst.iter_mut().zip(src) {
                *left += right;
            }
        }
        if let (Some(dst), Some(src)) = (total_het.as_mut(), block.het_ct) {
            for (left, right) in dst.iter_mut().zip(src) {
                *left += right;
            }
        }
    }
    let (individual_miss, individual_het) = match (total_miss, total_nonmiss, total_het) {
        (Some(miss), Some(nonmiss), Some(het)) => {
            finalize_individual_rates(miss, nonmiss, het, n_snps, request)
        }
        _ => (None, None),
    };
    Ok(GstatsCombinedOutput {
        site_maf,
        site_miss,
        site_het,
        individual_miss,
        individual_het,
    })
}

fn compute_pgen_stats_core(
    prefix: &str,
    request: GstatsRequest,
    sample_indices: Option<&[usize]>,
    site_indices: Option<&[usize]>,
    threads: usize,
) -> Result<(GstatsCombinedOutput, usize, usize), String> {
    let sample_count = gfcore::read_fam(prefix)?.len();
    if sample_count == 0 {
        return Err("no samples found in PGEN input".to_string());
    }
    let mut reader = PackedBedSourceReader::open(prefix, sample_count)?;
    let n_snps_full = reader.n_snps();
    let sample_indices = validate_selection_indices(sample_indices, sample_count, "sample")?;
    let site_indices = validate_selection_indices(site_indices, n_snps_full, "site")?;
    let n_samples = sample_indices.as_ref().map_or(sample_count, Vec::len);
    let n_snps = site_indices.as_ref().map_or(n_snps_full, Vec::len);
    let bytes_per_snp = reader.bytes_per_snp();
    let block_rows = ((16usize * 1024 * 1024) / bytes_per_snp.max(1))
        .clamp(1, 65_536)
        .min(n_snps_full.max(1));
    let mut packed_block = vec![0u8; block_rows.saturating_mul(bytes_per_snp)];
    let mut site_maf = request.site_maf.then(|| Vec::with_capacity(n_snps));
    let mut site_miss = request.site_miss.then(|| Vec::with_capacity(n_snps));
    let mut site_het = request.site_het.then(|| Vec::with_capacity(n_snps));
    let mut total_miss = request.needs_individual().then(|| vec![0u64; n_samples]);
    let mut total_nonmiss = request.needs_individual().then(|| vec![0u64; n_samples]);
    let mut total_het = request.needs_individual().then(|| vec![0u64; n_samples]);
    let mut site_cursor = 0usize;

    for source_start in (0..n_snps_full).step_by(block_rows) {
        let rows_here = block_rows.min(n_snps_full - source_start);
        let byte_len = rows_here.saturating_mul(bytes_per_snp);
        let got =
            reader.read_source_block(source_start, rows_here, &mut packed_block[..byte_len])?;
        if got != rows_here {
            return Err(format!(
                "PGEN source block length mismatch at row {source_start}: got {got}, expected {rows_here}"
            ));
        }
        let source_end = source_start + rows_here;
        let selected_begin = site_cursor;
        if let Some(indices) = site_indices.as_ref() {
            while site_cursor < indices.len() && indices[site_cursor] < source_end {
                site_cursor += 1;
            }
        } else {
            site_cursor = source_end;
        }
        let selected_here = site_cursor - selected_begin;
        let local_site_indices = site_indices.as_ref().map(|indices| {
            indices[selected_begin..site_cursor]
                .iter()
                .map(|&index| index - source_start)
                .collect::<Vec<_>>()
        });

        if request.needs_site() && (site_indices.is_none() || selected_here > 0) {
            let local_request = GstatsRequest {
                individual_miss: false,
                individual_het: false,
                ..request
            };
            let out = compute_joint_stats_selected_core(
                &packed_block[..byte_len],
                sample_count,
                rows_here,
                bytes_per_snp,
                local_request,
                sample_indices.as_deref(),
                local_site_indices.as_deref(),
                threads,
            )?;
            if let (Some(dst), Some(src)) = (site_maf.as_mut(), out.site_maf) {
                dst.extend(src);
            }
            if let (Some(dst), Some(src)) = (site_miss.as_mut(), out.site_miss) {
                dst.extend(src);
            }
            if let (Some(dst), Some(src)) = (site_het.as_mut(), out.site_het) {
                dst.extend(src);
            }
        }

        if request.needs_individual() && (site_indices.is_none() || selected_here > 0) {
            let individual_rows = if let Some(local_indices) = local_site_indices.as_deref() {
                let mut selected_packed = Vec::with_capacity(selected_here * bytes_per_snp);
                for &local_idx in local_indices {
                    let start = local_idx * bytes_per_snp;
                    selected_packed.extend_from_slice(&packed_block[start..start + bytes_per_snp]);
                }
                selected_packed
            } else {
                packed_block[..byte_len].to_vec()
            };
            let selected_rows = if site_indices.is_some() {
                selected_here
            } else {
                rows_here
            };
            let (miss, nonmiss, het) = compute_individual_counts_core(
                &individual_rows,
                sample_count,
                selected_rows,
                bytes_per_snp,
                sample_indices.as_deref(),
                threads,
            )?;
            for (dst, src) in total_miss.as_mut().unwrap().iter_mut().zip(miss) {
                *dst += src;
            }
            for (dst, src) in total_nonmiss.as_mut().unwrap().iter_mut().zip(nonmiss) {
                *dst += src;
            }
            for (dst, src) in total_het.as_mut().unwrap().iter_mut().zip(het) {
                *dst += src;
            }
        }
    }

    if site_cursor != n_snps {
        return Err(format!(
            "PGEN selected site count mismatch: processed {site_cursor}, expected {n_snps}"
        ));
    }
    let (individual_miss, individual_het) = match (total_miss, total_nonmiss, total_het) {
        (Some(miss), Some(nonmiss), Some(het)) => {
            finalize_individual_rates(miss, nonmiss, het, n_snps, request)
        }
        _ => (None, None),
    };
    Ok((
        GstatsCombinedOutput {
            site_maf,
            site_miss,
            site_het,
            individual_miss,
            individual_het,
        },
        n_samples,
        n_snps,
    ))
}

fn contiguous_site_range(indices: &[usize]) -> Option<(usize, usize)> {
    let (&start, rest) = indices.split_first()?;
    if rest
        .iter()
        .enumerate()
        .all(|(offset, &value)| value == start + offset + 1)
    {
        Some((start, start + indices.len()))
    } else {
        None
    }
}

fn compact_bed_selection(
    packed_src: &[u8],
    bytes_per_snp_full: usize,
    n_snps_full: usize,
    site_indices: Option<&[usize]>,
    sample_indices: Option<&[usize]>,
) -> Vec<u8> {
    let n_sites = site_indices.map_or(n_snps_full, <[usize]>::len);
    if sample_indices.is_none() {
        let mut output = Vec::with_capacity(n_sites * bytes_per_snp_full);
        for output_idx in 0..n_sites {
            let source_idx = site_indices.map_or(output_idx, |indices| indices[output_idx]);
            output.extend_from_slice(
                &packed_src[source_idx * bytes_per_snp_full..(source_idx + 1) * bytes_per_snp_full],
            );
        }
        return output;
    }

    let samples = sample_indices.unwrap();
    let bytes_per_snp = samples.len().div_ceil(4);
    let mut output = vec![0_u8; n_sites * bytes_per_snp];
    for output_site in 0..n_sites {
        let source_site = site_indices.map_or(output_site, |indices| indices[output_site]);
        let source_row =
            &packed_src[source_site * bytes_per_snp_full..(source_site + 1) * bytes_per_snp_full];
        let output_row =
            &mut output[output_site * bytes_per_snp..(output_site + 1) * bytes_per_snp];
        for (output_sample, &source_sample) in samples.iter().enumerate() {
            let code = (source_row[source_sample >> 2] >> ((source_sample & 3) << 1)) & 0b11;
            output_row[output_sample >> 2] |= code << ((output_sample & 3) << 1);
        }
    }
    output
}

#[cfg(test)]
fn compact_selected_bed(
    packed_src: &[u8],
    bytes_per_snp_full: usize,
    site_indices: &[usize],
    sample_indices: Option<&[usize]>,
) -> Vec<u8> {
    compact_bed_selection(
        packed_src,
        bytes_per_snp_full,
        site_indices.len(),
        Some(site_indices),
        sample_indices,
    )
}

fn readonly_indices_to_usize(
    values: Option<PyReadonlyArray1<'_, i64>>,
    label: &str,
) -> PyResult<Option<Vec<usize>>> {
    let Some(values) = values else {
        return Ok(None);
    };
    let mut output = Vec::with_capacity(values.as_array().len());
    for &value in values.as_array().iter() {
        if value < 0 {
            return Err(map_err_string_to_py(format!(
                "{label}_indices contains negative index {value}"
            )));
        }
        output.push(value as usize);
    }
    Ok(Some(output))
}

fn parse_bim_ldsc_meta(prefix: &str) -> Result<(Vec<i32>, Vec<i64>, Vec<f64>), String> {
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| format!("{bim_path}: {e}"))?;
    let reader = BufReader::new(file);

    let mut chrom_codes = Vec::<i32>::new();
    let mut positions = Vec::<i64>::new();
    let mut cm_positions = Vec::<f64>::new();
    let mut chrom_dict = HashMap::<String, i32>::new();

    for (line_no0, line) in reader.lines().enumerate() {
        let line_no = line_no0 + 1;
        let l = line.map_err(|e| format!("{bim_path}:{line_no}: {e}"))?;
        let toks: Vec<&str> = l.split_whitespace().collect();
        if toks.len() < 4 {
            return Err(format!(
                "{bim_path}:{line_no}: malformed BIM row, expect at least 4 columns"
            ));
        }
        let chrom = normalize_chr_token(toks[0]);
        let cm = toks[2]
            .parse::<f64>()
            .map_err(|e| format!("{bim_path}:{line_no}: invalid cM value '{}': {e}", toks[2]))?;
        let bp = toks[3]
            .parse::<i64>()
            .map_err(|e| format!("{bim_path}:{line_no}: invalid BP value '{}': {e}", toks[3]))?;
        let next_code = chrom_dict.len() as i32;
        let chrom_code = *chrom_dict.entry(chrom).or_insert(next_code);
        chrom_codes.push(chrom_code);
        positions.push(bp);
        cm_positions.push(cm);
    }

    if chrom_codes.is_empty() {
        return Err(format!("{bim_path}: no variant rows found"));
    }
    Ok((chrom_codes, positions, cm_positions))
}

fn parse_pvar_ldsc_meta(
    prefix: &str,
    require_cm: bool,
) -> Result<(Vec<i32>, Vec<i64>, Vec<f64>), String> {
    let pvar_path = format!("{}.pvar", crate::pgen::normalize_pgen_prefix(prefix));
    let file = File::open(&pvar_path).map_err(|e| format!("{pvar_path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut header: Option<Vec<String>> = None;
    let mut chrom_codes = Vec::<i32>::new();
    let mut positions = Vec::<i64>::new();
    let mut cm_positions = Vec::<f64>::new();
    let mut chrom_dict = HashMap::<String, i32>::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = line.map_err(|e| format!("{pvar_path}:{line_no}: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("##") {
            continue;
        }
        if trimmed.starts_with('#') {
            let columns = trimmed
                .split_whitespace()
                .map(|field| field.trim_start_matches('#').to_ascii_uppercase())
                .collect::<Vec<_>>();
            for name in ["CHROM", "POS"] {
                if !columns.iter().any(|column| column == name) {
                    return Err(format!(
                        "{pvar_path}:{line_no}: PVAR header is missing {name}"
                    ));
                }
            }
            header = Some(columns);
            continue;
        }
        let columns = header.as_ref().ok_or_else(|| {
            format!("{pvar_path}:{line_no}: PVAR data appears before #CHROM header")
        })?;
        let fields = trimmed.split_whitespace().collect::<Vec<_>>();
        let column = |name: &str| -> Result<&str, String> {
            let idx = columns
                .iter()
                .position(|field| field == name)
                .ok_or_else(|| format!("{pvar_path}: PVAR header is missing {name}"))?;
            fields
                .get(idx)
                .copied()
                .ok_or_else(|| format!("{pvar_path}:{line_no}: PVAR row is missing {name}"))
        };
        let chrom = normalize_chr_token(column("CHROM")?);
        let pos = column("POS")?
            .parse::<i64>()
            .map_err(|e| format!("{pvar_path}:{line_no}: invalid POS value: {e}"))?;
        let cm = if columns.iter().any(|field| field == "CM") {
            let raw = column("CM")?;
            if raw == "." {
                if require_cm {
                    return Err(format!(
                        "{pvar_path}:{line_no}: genetic-distance LD score requires numeric CM values"
                    ));
                }
                0.0
            } else {
                raw.parse::<f64>()
                    .map_err(|e| format!("{pvar_path}:{line_no}: invalid CM value '{raw}': {e}"))?
            }
        } else if require_cm {
            return Err(format!(
                "{pvar_path}: genetic-distance LD score requires a CM column in the PVAR header"
            ));
        } else {
            0.0
        };
        let next_code = chrom_dict.len() as i32;
        let chrom_code = *chrom_dict.entry(chrom).or_insert(next_code);
        chrom_codes.push(chrom_code);
        positions.push(pos);
        cm_positions.push(cm);
    }
    if chrom_codes.is_empty() {
        return Err(format!("{pvar_path}: no variant rows found"));
    }
    Ok((chrom_codes, positions, cm_positions))
}

struct TemporaryPackedBed(PathBuf);

impl Drop for TemporaryPackedBed {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn compute_pgen_ldscore(
    prefix: &str,
    window: LdscWindow,
    sample_indices: Option<&[usize]>,
    site_indices: Option<&[usize]>,
    threads: usize,
) -> Result<(Vec<i64>, Vec<f64>, usize), String> {
    let n_samples_full = gfcore::read_fam(prefix)?.len();
    if n_samples_full == 0 {
        return Err("no samples found in PGEN input".to_string());
    }
    let mut source = PackedBedSourceReader::open(prefix, n_samples_full)?;
    let n_snps_full = source.n_snps();
    let bytes_per_snp_full = source.bytes_per_snp();
    let sample_indices = validate_selection_indices(sample_indices, n_samples_full, "sample")?;
    let site_indices = validate_selection_indices(site_indices, n_snps_full, "site")?;
    let (chrom_codes_full, positions_full, cm_positions_full) =
        parse_pvar_ldsc_meta(prefix, matches!(window, LdscWindow::Cm(_)))?;
    if chrom_codes_full.len() != n_snps_full {
        return Err(format!(
            "PGEN/PVAR row mismatch: pgen={n_snps_full}, pvar={}",
            chrom_codes_full.len()
        ));
    }

    let unique = PGEN_LDSC_TEMP_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let temp_path = std::env::temp_dir().join(format!(
        "janusx-gstats-ldsc-{}-{unique}.bed",
        std::process::id()
    ));
    let temp_guard = TemporaryPackedBed(temp_path.clone());
    let file = File::create(&temp_path).map_err(|e| {
        format!(
            "create temporary PGEN LD-score payload {}: {e}",
            temp_path.display()
        )
    })?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    writer
        .write_all(&[0x6c, 0x1b, 0x01])
        .map_err(|e| format!("write temporary PGEN LD-score header: {e}"))?;
    let block_rows = ((16usize * 1024 * 1024) / bytes_per_snp_full.max(1))
        .clamp(1, 65_536)
        .min(n_snps_full.max(1));
    let mut packed = vec![0u8; block_rows.saturating_mul(bytes_per_snp_full)];
    for row_start in (0..n_snps_full).step_by(block_rows) {
        let rows = block_rows.min(n_snps_full - row_start);
        let byte_len = rows.saturating_mul(bytes_per_snp_full);
        let got = source.read_source_block(row_start, rows, &mut packed[..byte_len])?;
        if got != rows {
            return Err(format!(
                "PGEN source block length mismatch at row {row_start}: got {got}, expected {rows}"
            ));
        }
        writer
            .write_all(&packed[..byte_len])
            .map_err(|e| format!("write temporary PGEN LD-score payload: {e}"))?;
    }
    writer
        .flush()
        .map_err(|e| format!("flush temporary PGEN LD-score payload: {e}"))?;
    drop(writer);

    let payload_file = File::open(&temp_path).map_err(|e| {
        format!(
            "open temporary PGEN LD-score payload {}: {e}",
            temp_path.display()
        )
    })?;
    let mmap = unsafe { Mmap::map(&payload_file) }
        .map_err(|e| format!("mmap temporary PGEN LD-score payload: {e}"))?;
    let packed_src = &mmap[3..];
    let n_samples = sample_indices.as_ref().map_or(n_samples_full, Vec::len);
    let n_snps = site_indices.as_ref().map_or(n_snps_full, Vec::len);
    let select_i64 = |values: &[i64]| -> Vec<i64> {
        site_indices.as_ref().map_or_else(
            || values.to_vec(),
            |indices| indices.iter().map(|&index| values[index]).collect(),
        )
    };
    let chrom_i64 = chrom_codes_full
        .iter()
        .map(|&value| i64::from(value))
        .collect::<Vec<_>>();
    let chrom_codes = select_i64(&chrom_i64)
        .into_iter()
        .map(|value| value as i32)
        .collect::<Vec<_>>();
    let positions = select_i64(&positions_full);
    let cm_positions = site_indices.as_ref().map_or_else(
        || cm_positions_full.clone(),
        |indices| {
            indices
                .iter()
                .map(|&index| cm_positions_full[index])
                .collect()
        },
    );
    let run_core = |data: &[u8], row_bytes: usize| {
        compute_ldscore_core(
            data,
            n_samples,
            n_snps,
            row_bytes,
            &chrom_codes,
            &positions,
            &cm_positions,
            window,
            threads,
        )
    };
    let (m_counts, ld_scores) = if sample_indices.is_none() {
        if let Some(indices) = site_indices.as_ref() {
            if let Some((start, end)) = contiguous_site_range(indices) {
                run_core(
                    &packed_src[start * bytes_per_snp_full..end * bytes_per_snp_full],
                    bytes_per_snp_full,
                )?
            } else {
                let compact = compact_bed_selection(
                    packed_src,
                    bytes_per_snp_full,
                    n_snps_full,
                    Some(indices),
                    None,
                );
                run_core(&compact, bytes_per_snp_full)?
            }
        } else {
            run_core(packed_src, bytes_per_snp_full)?
        }
    } else {
        let compact = compact_bed_selection(
            packed_src,
            bytes_per_snp_full,
            n_snps_full,
            site_indices.as_deref(),
            sample_indices.as_deref(),
        );
        run_core(&compact, n_samples.div_ceil(4))?
    };
    drop(mmap);
    drop(payload_file);
    drop(temp_guard);
    Ok((m_counts, ld_scores, n_samples))
}

fn parse_ldsc_window(kind: &str, value: f64) -> Result<LdscWindow, String> {
    if !(value.is_finite() && value > 0.0_f64) {
        return Err(format!("window_value must be finite and > 0, got {value}"));
    }
    let key = kind.trim().to_ascii_lowercase();
    match key.as_str() {
        "variant" | "variants" | "snp" | "snps" => {
            let rounded = value.round();
            if (rounded - value).abs() > 1e-9 {
                return Err(format!(
                    "variant-count LD-score window must be an integer, got {value}"
                ));
            }
            let w = rounded as i64;
            if w <= 0 {
                return Err(format!(
                    "variant-count LD-score window must be > 0, got {w}"
                ));
            }
            Ok(LdscWindow::Variants(w as usize))
        }
        "bp" | "b" | "kb" | "mb" => {
            let rounded = value.round();
            if (rounded - value).abs() > 1e-6 {
                return Err(format!(
                    "bp LD-score window must resolve to an integer, got {value}"
                ));
            }
            let w = rounded as i64;
            if w <= 0 {
                return Err(format!("bp LD-score window must be > 0, got {w}"));
            }
            Ok(LdscWindow::Bp(w))
        }
        "cm" | "genetic" => Ok(LdscWindow::Cm(value)),
        _ => Err(format!(
            "window_kind must be one of: variants, bp, cm; got '{kind}'"
        )),
    }
}

fn ldsc_bitplane_max_bytes() -> u64 {
    let raw = std::env::var("JANUSX_GSTATS_LDSC_BITPLANE_MAX_MB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(LDSC_BITPLANE_MAX_MB_DEFAULT);
    raw.saturating_mul(1024_u64 * 1024_u64)
}

fn build_sorted_chrom_groups(
    chrom_codes: &[i32],
    positions: &[i64],
    cm_positions: &[f64],
    window: LdscWindow,
) -> Vec<Vec<usize>> {
    let mut by_chr = HashMap::<i32, Vec<usize>>::new();
    for (idx, &chrom_code) in chrom_codes.iter().enumerate() {
        by_chr.entry(chrom_code).or_default().push(idx);
    }
    let mut groups: Vec<Vec<usize>> = by_chr.into_values().collect();
    for group in groups.iter_mut() {
        match window {
            LdscWindow::Cm(_) => group.sort_by(|a, b| {
                cm_positions[*a]
                    .partial_cmp(&cm_positions[*b])
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| positions[*a].cmp(&positions[*b]))
            }),
            _ => group.sort_by_key(|&idx| positions[idx]),
        }
    }
    groups
}

fn compute_window_bounds(
    group: &[usize],
    positions: &[i64],
    cm_positions: &[f64],
    window: LdscWindow,
) -> (Vec<usize>, Vec<usize>) {
    let n = group.len();
    let mut starts = vec![0usize; n];
    let mut ends = vec![0usize; n];
    match window {
        LdscWindow::Variants(w) => {
            for i in 0..n {
                starts[i] = i.saturating_sub(w);
                ends[i] = std::cmp::min(n, i.saturating_add(w).saturating_add(1));
            }
        }
        LdscWindow::Bp(w) => {
            let mut left = 0usize;
            let mut right = 0usize;
            for i in 0..n {
                let pos_i = positions[group[i]];
                while left < i && pos_i.saturating_sub(positions[group[left]]) > w {
                    left += 1;
                }
                if right < i {
                    right = i;
                }
                while right + 1 < n && positions[group[right + 1]].saturating_sub(pos_i) <= w {
                    right += 1;
                }
                starts[i] = left;
                ends[i] = right + 1;
            }
        }
        LdscWindow::Cm(w) => {
            let eps = 1e-12_f64;
            let mut left = 0usize;
            let mut right = 0usize;
            for i in 0..n {
                let cm_i = cm_positions[group[i]];
                while left < i && (cm_i - cm_positions[group[left]]) > w + eps {
                    left += 1;
                }
                if right < i {
                    right = i;
                }
                while right + 1 < n && (cm_positions[group[right + 1]] - cm_i) <= w + eps {
                    right += 1;
                }
                starts[i] = left;
                ends[i] = right + 1;
            }
        }
    }
    (starts, ends)
}

#[inline]
fn nomiss_r2_from_stats_and_bitplanes(
    i: usize,
    j: usize,
    n_samples: usize,
    stats: &[PackedRowStats],
    h_bits: &[u64],
    l_bits: &[u64],
    bitplane_words: usize,
    word_masks: &[u64],
) -> f64 {
    let st_i = stats[i];
    let st_j = stats[j];
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let dot = dot_nomiss_pair_bitplanes(i, j, h_bits, l_bits, bitplane_words, word_masks);
    let cov = dot - (n_samples as f64) * st_i.mean * st_j.mean;
    let denom_corr = denom * st_i.std * st_j.std;
    if denom_corr > 0.0_f64 && cov.is_finite() {
        let corr = cov / denom_corr;
        (corr * corr).clamp(0.0_f64, 1.0_f64)
    } else {
        0.0_f64
    }
}

#[inline]
fn nomiss_r2_from_stats_and_packed(
    row_i: &[u8],
    row_j: &[u8],
    n_samples: usize,
    stats_i: PackedRowStats,
    stats_j: PackedRowStats,
) -> f64 {
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let byte_lut = packed_byte_lut();
    let dot =
        dot_nomiss_pair_from_packed(row_i, row_j, n_samples, packed_pair_lut(), &byte_lut.code4);
    let cov = dot - (n_samples as f64) * stats_i.mean * stats_j.mean;
    let denom_corr = denom * stats_i.std * stats_j.std;
    if denom_corr > 0.0_f64 && cov.is_finite() {
        let corr = cov / denom_corr;
        (corr * corr).clamp(0.0_f64, 1.0_f64)
    } else {
        0.0_f64
    }
}

fn compute_ldscore_core(
    packed_src: &[u8],
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    chrom_codes: &[i32],
    positions: &[i64],
    cm_positions: &[f64],
    window: LdscWindow,
    threads: usize,
) -> Result<(Vec<i64>, Vec<f64>), String> {
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let byte_lut = packed_byte_lut();
    let mut row_stats = vec![PackedRowStats::default(); n_snps];

    let mut stats_run = || {
        row_stats
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, dst)| {
                let row = &packed_src[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                *dst = compute_packed_row_stats(row, n_samples, byte_lut);
            });
    };
    if let Some(tp) = &pool {
        tp.install(&mut stats_run);
    } else {
        stats_run();
    }

    let chrom_groups = build_sorted_chrom_groups(chrom_codes, positions, cm_positions, window);
    let words = n_samples.div_ceil(64);
    let bitplane_bytes = 3_u64
        .saturating_mul(n_snps as u64)
        .saturating_mul(words as u64)
        .saturating_mul(std::mem::size_of::<u64>() as u64);
    let use_bitplanes = words > 0 && bitplane_bytes <= ldsc_bitplane_max_bytes();

    let mut m_counts = vec![0_i64; n_snps];
    let mut ld_scores = vec![0.0_f64; n_snps];

    if use_bitplanes {
        let (h_bits, l_bits, m_bits, bitplane_words, word_masks) =
            build_bitplanes_u64(packed_src, n_snps, bytes_per_snp, n_samples, pool.as_ref());
        for group in chrom_groups.iter() {
            let (starts, ends) = compute_window_bounds(group, positions, cm_positions, window);
            let run = || {
                group
                    .par_iter()
                    .enumerate()
                    .map(|(local_i, &global_i)| {
                        let st_i = row_stats[global_i];
                        let mut sum = if st_i.non_missing > 1 && st_i.maf > 0.0_f64 {
                            1.0_f64
                        } else {
                            0.0_f64
                        };
                        let off_i = global_i * bitplane_words;
                        let hi_i = &h_bits[off_i..off_i + bitplane_words];
                        let li_i = &l_bits[off_i..off_i + bitplane_words];
                        let mi_i = &m_bits[off_i..off_i + bitplane_words];
                        for local_j in starts[local_i]..ends[local_i] {
                            if local_j == local_i {
                                continue;
                            }
                            let global_j = group[local_j];
                            let st_j = row_stats[global_j];
                            let r2 = if !st_i.has_missing && !st_j.has_missing {
                                nomiss_r2_from_stats_and_bitplanes(
                                    global_i,
                                    global_j,
                                    n_samples,
                                    &row_stats,
                                    &h_bits,
                                    &l_bits,
                                    bitplane_words,
                                    &word_masks,
                                )
                            } else {
                                let off_j = global_j * bitplane_words;
                                let hi_j = &h_bits[off_j..off_j + bitplane_words];
                                let li_j = &l_bits[off_j..off_j + bitplane_words];
                                let mi_j = &m_bits[off_j..off_j + bitplane_words];
                                r2_pairwise_complete_bitplanes(
                                    hi_i,
                                    li_i,
                                    mi_i,
                                    hi_j,
                                    li_j,
                                    mi_j,
                                    &word_masks,
                                )
                                .unwrap_or(0.0_f64)
                                .clamp(0.0_f64, 1.0_f64)
                            };
                            if r2.is_finite() {
                                sum += r2;
                            }
                        }
                        ((ends[local_i] - starts[local_i]) as i64, sum)
                    })
                    .collect::<Vec<(i64, f64)>>()
            };
            let local_out = if let Some(tp) = &pool {
                tp.install(run)
            } else {
                run()
            };
            for (local_i, &global_i) in group.iter().enumerate() {
                m_counts[global_i] = local_out[local_i].0;
                ld_scores[global_i] = local_out[local_i].1;
            }
        }
    } else {
        let code4_lut = &byte_lut.code4;
        let pair_lut = packed_pair_lut();
        for group in chrom_groups.iter() {
            let (starts, ends) = compute_window_bounds(group, positions, cm_positions, window);
            let run = || {
                group
                    .par_iter()
                    .enumerate()
                    .map(|(local_i, &global_i)| {
                        let st_i = row_stats[global_i];
                        let row_i =
                            &packed_src[global_i * bytes_per_snp..(global_i + 1) * bytes_per_snp];
                        let mut sum = if st_i.non_missing > 1 && st_i.maf > 0.0_f64 {
                            1.0_f64
                        } else {
                            0.0_f64
                        };
                        for local_j in starts[local_i]..ends[local_i] {
                            if local_j == local_i {
                                continue;
                            }
                            let global_j = group[local_j];
                            let st_j = row_stats[global_j];
                            let row_j = &packed_src
                                [global_j * bytes_per_snp..(global_j + 1) * bytes_per_snp];
                            let r2 = if !st_i.has_missing && !st_j.has_missing {
                                nomiss_r2_from_stats_and_packed(row_i, row_j, n_samples, st_i, st_j)
                            } else {
                                r2_pairwise_complete_from_packed(
                                    row_i, row_j, n_samples, pair_lut, code4_lut,
                                )
                                .unwrap_or(0.0_f64)
                                .clamp(0.0_f64, 1.0_f64)
                            };
                            if r2.is_finite() {
                                sum += r2;
                            }
                        }
                        ((ends[local_i] - starts[local_i]) as i64, sum)
                    })
                    .collect::<Vec<(i64, f64)>>()
            };
            let local_out = if let Some(tp) = &pool {
                tp.install(run)
            } else {
                run()
            };
            for (local_i, &global_i) in group.iter().enumerate() {
                m_counts[global_i] = local_out[local_i].0;
                ld_scores[global_i] = local_out[local_i].1;
            }
        }
    }

    Ok((m_counts, ld_scores))
}

#[pyfunction]
#[pyo3(signature = (prefix, threads=0, sample_indices=None, site_indices=None))]
pub fn gstats_bed_site_stats<'py>(
    py: Python<'py>,
    prefix: String,
    threads: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    usize,
)> {
    let sample_indices = readonly_indices_to_usize(sample_indices, "sample")?;
    let site_indices = readonly_indices_to_usize(site_indices, "site")?;
    let (maf, miss, het, n_samples) = py
        .detach(
            move || -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize), String> {
                if crate::pgen::path_uses_pgen(&prefix) {
                    let request = GstatsRequest {
                        site_maf: true,
                        site_miss: true,
                        site_het: true,
                        ..GstatsRequest::default()
                    };
                    let (out, n_samples, _n_snps) = compute_pgen_stats_core(
                        &prefix,
                        request,
                        sample_indices.as_deref(),
                        site_indices.as_deref(),
                        threads,
                    )?;
                    return Ok((
                        out.site_maf.unwrap(),
                        out.site_miss.unwrap(),
                        out.site_het.unwrap(),
                        n_samples,
                    ));
                }
                if sample_indices.is_none() && site_indices.is_none() {
                    let (maf, miss, het, n_samples, _n_snps) =
                        compute_site_stats_unified_from_prefix(&prefix, threads)?;
                    return Ok((maf, miss, het, n_samples));
                }
                let bed_prefix = normalize_plink_prefix(&prefix);
                let (mmap, n_samples_full, n_snps_full, bytes_per_snp) =
                    open_bed_mmap(&bed_prefix)?;
                let request = GstatsRequest {
                    site_maf: true,
                    site_miss: true,
                    site_het: true,
                    ..GstatsRequest::default()
                };
                let out = compute_joint_stats_selected_core(
                    &mmap[3..],
                    n_samples_full,
                    n_snps_full,
                    bytes_per_snp,
                    request,
                    sample_indices.as_deref(),
                    site_indices.as_deref(),
                    threads,
                )?;
                let n_samples = sample_indices.as_ref().map_or(n_samples_full, Vec::len);
                Ok((
                    out.site_maf.unwrap(),
                    out.site_miss.unwrap(),
                    out.site_het.unwrap(),
                    n_samples,
                ))
            },
        )
        .map_err(map_err_string_to_py)?;

    let maf_arr = PyArray1::from_owned_array(py, Array1::from_vec(maf)).into_bound();
    let miss_arr = PyArray1::from_owned_array(py, Array1::from_vec(miss)).into_bound();
    let het_arr = PyArray1::from_owned_array(py, Array1::from_vec(het)).into_bound();
    Ok((maf_arr, miss_arr, het_arr, n_samples))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    site_maf=true,
    site_miss=true,
    site_het=true,
    individual_miss=true,
    individual_het=true,
    threads=0,
    sample_indices=None,
    site_indices=None
))]
pub fn gstats_bed_joint_stats<'py>(
    py: Python<'py>,
    prefix: String,
    site_maf: bool,
    site_miss: bool,
    site_het: bool,
    individual_miss: bool,
    individual_het: bool,
    threads: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(
    Option<Bound<'py, PyArray1<f32>>>,
    Option<Bound<'py, PyArray1<f32>>>,
    Option<Bound<'py, PyArray1<f32>>>,
    Option<Bound<'py, PyArray1<f32>>>,
    Option<Bound<'py, PyArray1<f32>>>,
    usize,
    usize,
)> {
    let sample_indices = readonly_indices_to_usize(sample_indices, "sample")?;
    let site_indices = readonly_indices_to_usize(site_indices, "site")?;
    let bed_prefix = normalize_plink_prefix(&prefix);
    let request = GstatsRequest {
        site_maf,
        site_miss,
        site_het,
        individual_miss,
        individual_het,
    };
    let out = py
        .detach(
            move || -> Result<(GstatsCombinedOutput, usize, usize), String> {
                if crate::pgen::path_uses_pgen(&prefix) {
                    return compute_pgen_stats_core(
                        &prefix,
                        request,
                        sample_indices.as_deref(),
                        site_indices.as_deref(),
                        threads,
                    );
                }
                if sample_indices.is_none() && site_indices.is_none() && !request.needs_individual()
                {
                    compute_joint_site_only_via_meta(&bed_prefix, request, threads)
                } else {
                    let (mmap, n_samples_full, n_snps_full, bytes_per_snp) =
                        open_bed_mmap(&bed_prefix)?;
                    let packed_src = &mmap[3..];
                    let out = compute_joint_stats_selected_core(
                        packed_src,
                        n_samples_full,
                        n_snps_full,
                        bytes_per_snp,
                        request,
                        sample_indices.as_deref(),
                        site_indices.as_deref(),
                        threads,
                    )?;
                    let n_samples = sample_indices.as_ref().map_or(n_samples_full, Vec::len);
                    let n_snps = site_indices.as_ref().map_or(n_snps_full, Vec::len);
                    Ok((out, n_samples, n_snps))
                }
            },
        )
        .map_err(map_err_string_to_py)?;

    let (out, n_samples, n_snps) = out;
    let maf_arr = out
        .site_maf
        .map(|v| PyArray1::from_owned_array(py, Array1::from_vec(v)).into_bound());
    let miss_arr = out
        .site_miss
        .map(|v| PyArray1::from_owned_array(py, Array1::from_vec(v)).into_bound());
    let het_arr = out
        .site_het
        .map(|v| PyArray1::from_owned_array(py, Array1::from_vec(v)).into_bound());
    let imiss_arr = out
        .individual_miss
        .map(|v| PyArray1::from_owned_array(py, Array1::from_vec(v)).into_bound());
    let ihet_arr = out
        .individual_het
        .map(|v| PyArray1::from_owned_array(py, Array1::from_vec(v)).into_bound());
    Ok((
        maf_arr, miss_arr, het_arr, imiss_arr, ihet_arr, n_samples, n_snps,
    ))
}

#[pyfunction]
#[pyo3(signature = (prefix, threads=0, repeats=3))]
pub fn gstats_bed_site_stats_compare<'py>(
    py: Python<'py>,
    prefix: String,
    threads: usize,
    repeats: usize,
) -> PyResult<(f64, f64, f32, f32, f32, usize, usize)> {
    let reps = repeats.max(1);
    py.detach(move || -> Result<(f64, f64, f32, f32, f32, usize, usize), String> {
        let (maf_legacy, miss_legacy, het_legacy, n_samples_legacy, n_snps_legacy) =
            compute_site_stats_legacy_from_prefix(&prefix, threads)?;
        let (maf_unified, miss_unified, het_unified, n_samples_unified, n_snps_unified) =
            compute_site_stats_unified_from_prefix(&prefix, threads)?;
        if n_samples_legacy != n_samples_unified || n_snps_legacy != n_snps_unified {
            return Err(format!(
                "site-stats shape mismatch: legacy=({n_samples_legacy},{n_snps_legacy}) unified=({n_samples_unified},{n_snps_unified})"
            ));
        }
        let maf_diff = max_abs_diff_f32(&maf_legacy, &maf_unified);
        let miss_diff = max_abs_diff_f32(&miss_legacy, &miss_unified);
        let het_diff = max_abs_diff_f32(&het_legacy, &het_unified);

        let mut legacy_secs = 0.0_f64;
        let mut unified_secs = 0.0_f64;
        for _ in 0..reps {
            let t0 = Instant::now();
            let (maf, miss, het, _n_samples, _n_snps) =
                compute_site_stats_legacy_from_prefix(&prefix, threads)?;
            let _ = (maf.len(), miss.len(), het.len());
            legacy_secs += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            let (maf, miss, het, _n_samples, _n_snps) =
                compute_site_stats_unified_from_prefix(&prefix, threads)?;
            let _ = (maf.len(), miss.len(), het.len());
            unified_secs += t0.elapsed().as_secs_f64();
        }
        Ok((
            legacy_secs / reps as f64,
            unified_secs / reps as f64,
            maf_diff,
            miss_diff,
            het_diff,
            n_samples_legacy,
            n_snps_legacy,
        ))
    })
    .map_err(map_err_string_to_py)
}

#[pyfunction]
#[pyo3(signature = (prefix, threads=0, sample_indices=None, site_indices=None))]
pub fn gstats_bed_individual_stats<'py>(
    py: Python<'py>,
    prefix: String,
    threads: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, usize)> {
    let sample_indices = readonly_indices_to_usize(sample_indices, "sample")?;
    let site_indices = readonly_indices_to_usize(site_indices, "site")?;
    let bed_prefix = normalize_plink_prefix(&prefix);
    let (miss_rate, het_rate, n_snps) = py
        .detach(move || -> Result<(Vec<f32>, Vec<f32>, usize), String> {
            if crate::pgen::path_uses_pgen(&prefix) {
                let request = GstatsRequest {
                    individual_miss: true,
                    individual_het: true,
                    ..GstatsRequest::default()
                };
                let (out, _n_samples, n_snps) = compute_pgen_stats_core(
                    &prefix,
                    request,
                    sample_indices.as_deref(),
                    site_indices.as_deref(),
                    threads,
                )?;
                return Ok((
                    out.individual_miss.unwrap(),
                    out.individual_het.unwrap(),
                    n_snps,
                ));
            }
            let (mmap, n_samples, n_snps, bytes_per_snp) = open_bed_mmap(&bed_prefix)?;
            let packed_src = &mmap[3..];
            if sample_indices.is_none() && site_indices.is_none() {
                let (miss_rate, het_rate) = compute_individual_stats_core(
                    packed_src,
                    n_samples,
                    n_snps,
                    bytes_per_snp,
                    threads,
                )?;
                return Ok((miss_rate, het_rate, n_snps));
            }
            let request = GstatsRequest {
                individual_miss: true,
                individual_het: true,
                ..GstatsRequest::default()
            };
            let out = compute_joint_stats_selected_core(
                packed_src,
                n_samples,
                n_snps,
                bytes_per_snp,
                request,
                sample_indices.as_deref(),
                site_indices.as_deref(),
                threads,
            )?;
            let selected_n_snps = site_indices.as_ref().map_or(n_snps, Vec::len);
            Ok((
                out.individual_miss.unwrap(),
                out.individual_het.unwrap(),
                selected_n_snps,
            ))
        })
        .map_err(map_err_string_to_py)?;

    let miss_arr = PyArray1::from_owned_array(py, Array1::from_vec(miss_rate)).into_bound();
    let het_arr = PyArray1::from_owned_array(py, Array1::from_vec(het_rate)).into_bound();
    Ok((miss_arr, het_arr, n_snps))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    window_kind,
    window_value,
    threads=0,
    sample_indices=None,
    site_indices=None
))]
pub fn gstats_bed_ldscore<'py>(
    py: Python<'py>,
    prefix: String,
    window_kind: String,
    window_value: f64,
    threads: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>, usize)> {
    let sample_indices = readonly_indices_to_usize(sample_indices, "sample")?;
    let site_indices = readonly_indices_to_usize(site_indices, "site")?;
    let window = parse_ldsc_window(&window_kind, window_value).map_err(map_err_string_to_py)?;
    let (m_counts, ld_scores, n_samples) = py
        .detach(move || -> Result<(Vec<i64>, Vec<f64>, usize), String> {
            if crate::pgen::path_uses_pgen(&prefix) {
                return compute_pgen_ldscore(
                    &prefix,
                    window,
                    sample_indices.as_deref(),
                    site_indices.as_deref(),
                    threads,
                );
            }
            let bed_prefix = normalize_plink_prefix(&prefix);
            let (mmap, n_samples_full, n_snps_full, bytes_per_snp_full) =
                open_bed_mmap(&bed_prefix)?;
            let (chrom_codes_full, positions_full, cm_positions_full) =
                parse_bim_ldsc_meta(&bed_prefix)?;
            if chrom_codes_full.len() != n_snps_full
                || positions_full.len() != n_snps_full
                || cm_positions_full.len() != n_snps_full
            {
                return Err(format!(
                    "BED/BIM row mismatch: bed={n_snps_full}, bim={}",
                    chrom_codes_full.len()
                ));
            }
            if sample_indices.is_none() && site_indices.is_none() {
                let (m_counts, ld_scores) = compute_ldscore_core(
                    &mmap[3..],
                    n_samples_full,
                    n_snps_full,
                    bytes_per_snp_full,
                    &chrom_codes_full,
                    &positions_full,
                    &cm_positions_full,
                    window,
                    threads,
                )?;
                return Ok((m_counts, ld_scores, n_samples_full));
            }
            let sample_indices =
                validate_selection_indices(sample_indices.as_deref(), n_samples_full, "sample")?;
            let site_indices =
                validate_selection_indices(site_indices.as_deref(), n_snps_full, "site")?;
            let n_samples = sample_indices.as_ref().map_or(n_samples_full, Vec::len);
            let n_snps = site_indices.as_ref().map_or(n_snps_full, Vec::len);
            let select_meta = |values: &[i64]| -> Vec<i64> {
                site_indices.as_ref().map_or_else(
                    || values.to_vec(),
                    |indices| indices.iter().map(|&index| values[index]).collect(),
                )
            };
            let chrom_as_i64: Vec<i64> = chrom_codes_full.iter().map(|&v| i64::from(v)).collect();
            let chrom_codes: Vec<i32> = select_meta(&chrom_as_i64)
                .into_iter()
                .map(|v| v as i32)
                .collect();
            let positions = select_meta(&positions_full);
            let cm_positions: Vec<f64> = site_indices.as_ref().map_or_else(
                || cm_positions_full.clone(),
                |indices| {
                    indices
                        .iter()
                        .map(|&index| cm_positions_full[index])
                        .collect()
                },
            );
            let packed_src = &mmap[3..];
            let run_core = |data: &[u8], bytes_per_snp: usize| {
                compute_ldscore_core(
                    data,
                    n_samples,
                    n_snps,
                    bytes_per_snp,
                    &chrom_codes,
                    &positions,
                    &cm_positions,
                    window,
                    threads,
                )
            };
            let (m_counts, ld_scores) = if sample_indices.is_none() {
                if let Some(indices) = site_indices.as_ref() {
                    if let Some((start, end)) = contiguous_site_range(indices) {
                        run_core(
                            &packed_src[start * bytes_per_snp_full..end * bytes_per_snp_full],
                            bytes_per_snp_full,
                        )?
                    } else {
                        let compact = compact_bed_selection(
                            packed_src,
                            bytes_per_snp_full,
                            n_snps_full,
                            Some(indices),
                            None,
                        );
                        run_core(&compact, bytes_per_snp_full)?
                    }
                } else {
                    run_core(packed_src, bytes_per_snp_full)?
                }
            } else {
                let compact = compact_bed_selection(
                    packed_src,
                    bytes_per_snp_full,
                    n_snps_full,
                    site_indices.as_deref(),
                    sample_indices.as_deref(),
                );
                run_core(&compact, n_samples.div_ceil(4))?
            };
            Ok((m_counts, ld_scores, n_samples))
        })
        .map_err(map_err_string_to_py)?;

    let m_arr = PyArray1::from_owned_array(py, Array1::from_vec(m_counts)).into_bound();
    let ld_arr = PyArray1::from_owned_array(py, Array1::from_vec(ld_scores)).into_bound();
    Ok((m_arr, ld_arr, n_samples))
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    struct TempPgenDataset {
        dir: PathBuf,
        prefix: PathBuf,
        n_samples: usize,
        n_variants: usize,
    }

    impl TempPgenDataset {
        fn new(bed_rows: Vec<u8>, n_samples: usize, n_variants: usize) -> Self {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "janusx_gstats_pgen_{}_{}",
                std::process::id(),
                stamp
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let prefix = dir.join("toy");
            let bytes_per_variant = n_samples.div_ceil(4);
            assert_eq!(bed_rows.len(), n_variants * bytes_per_variant);

            let mut pgen = vec![0x6c, 0x1b, 0x02];
            pgen.extend_from_slice(&(n_variants as u32).to_le_bytes());
            pgen.extend_from_slice(&(n_samples as u32).to_le_bytes());
            pgen.push(0); // fixed hard-call storage
            for row in bed_rows.chunks_exact(bytes_per_variant) {
                let mut category_row = vec![0u8; bytes_per_variant];
                for sample_idx in 0..n_samples {
                    let bed_code = (row[sample_idx >> 2] >> ((sample_idx & 3) * 2)) & 3;
                    let category = match bed_code {
                        0b00 => 0b00,
                        0b10 => 0b01,
                        0b11 => 0b10,
                        0b01 => 0b11,
                        _ => unreachable!(),
                    };
                    category_row[sample_idx >> 2] |= category << ((sample_idx & 3) * 2);
                }
                pgen.extend_from_slice(&category_row);
            }
            std::fs::write(prefix.with_extension("pgen"), pgen).unwrap();
            std::fs::write(
                prefix.with_extension("psam"),
                format!(
                    "#FID IID\n{}",
                    (0..n_samples)
                        .map(|i| format!("F{i} I{i}\n"))
                        .collect::<String>()
                ),
            )
            .unwrap();
            std::fs::write(
                prefix.with_extension("pvar"),
                format!(
                    "#CHROM POS ID REF ALT CM\n{}",
                    (0..n_variants)
                        .map(|i| format!(
                            "{} {} v{} A C {}\n",
                            if i == 2 { 2 } else { 1 },
                            (i + 1) * 100,
                            i + 1,
                            (i + 1) as f64 / 10.0
                        ))
                        .collect::<String>()
                ),
            )
            .unwrap();

            Self {
                dir,
                prefix,
                n_samples,
                n_variants,
            }
        }

        fn pgen_path(&self) -> String {
            format!("{}.pgen", self.prefix.to_string_lossy())
        }
    }

    impl Drop for TempPgenDataset {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn selected_joint_stats_use_selected_samples_and_sites() {
        // Four samples per row, PLINK codes packed low-to-high in one byte.
        let packed = [120_u8, 47_u8, 233_u8];
        let request = GstatsRequest {
            site_maf: true,
            site_miss: true,
            site_het: true,
            individual_miss: true,
            individual_het: true,
        };
        let out = compute_joint_stats_selected_core(
            &packed,
            4,
            3,
            1,
            request,
            Some(&[1, 3]),
            Some(&[0, 2]),
            1,
        )
        .unwrap();

        assert_eq!(out.site_maf.unwrap(), vec![0.5, 0.25]);
        assert_eq!(out.site_miss.unwrap(), vec![0.5, 0.0]);
        assert_eq!(out.site_het.unwrap(), vec![1.0, 0.5]);
        assert_eq!(out.individual_miss.unwrap(), vec![0.0, 0.5]);
        assert_eq!(out.individual_het.unwrap(), vec![1.0, 0.0]);
    }

    #[test]
    fn selection_indices_must_be_strictly_increasing() {
        assert!(validate_selection_indices(Some(&[1, 1]), 4, "sample").is_err());
        assert!(validate_selection_indices(Some(&[2, 1]), 4, "sample").is_err());
        assert!(validate_selection_indices(Some(&[4]), 4, "sample").is_err());
        assert!(validate_selection_indices(Some(&[]), 4, "sample").is_err());
        assert_eq!(
            validate_selection_indices(Some(&[0, 2]), 4, "sample").unwrap(),
            Some(vec![0, 2])
        );
    }

    #[test]
    fn selected_bed_repacking_preserves_plink_codes() {
        let packed = [120_u8, 47_u8, 233_u8];
        let compact = compact_selected_bed(&packed, 1, &[0, 2], Some(&[1, 3]));
        assert_eq!(compact, vec![6_u8, 14_u8]);
        assert_eq!(contiguous_site_range(&[1, 2]), Some((1, 3)));
        assert_eq!(contiguous_site_range(&[0, 2]), None);
    }

    #[test]
    fn pgen_joint_stats_match_bed_for_selected_samples_and_sites() {
        let bed_rows = vec![0x78, 0x02, 0xe4, 0x01, 0x1b, 0x02, 0x2c, 0x00];
        let dataset = TempPgenDataset::new(bed_rows.clone(), 5, 4);
        let request = GstatsRequest {
            site_maf: true,
            site_miss: true,
            site_het: true,
            individual_miss: true,
            individual_het: true,
        };
        let sample_indices = [0usize, 2, 4];
        let site_indices = [0usize, 2, 3];
        let (pgen, n_samples, n_sites) = compute_pgen_stats_core(
            &dataset.pgen_path(),
            request,
            Some(&sample_indices),
            Some(&site_indices),
            2,
        )
        .unwrap();
        let bed = compute_joint_stats_selected_core(
            &bed_rows,
            dataset.n_samples,
            dataset.n_variants,
            dataset.n_samples.div_ceil(4),
            request,
            Some(&sample_indices),
            Some(&site_indices),
            2,
        )
        .unwrap();

        assert_eq!(
            (n_samples, n_sites),
            (sample_indices.len(), site_indices.len())
        );
        for (pgen, bed) in [
            (pgen.site_maf.unwrap(), bed.site_maf.unwrap()),
            (pgen.site_miss.unwrap(), bed.site_miss.unwrap()),
            (pgen.site_het.unwrap(), bed.site_het.unwrap()),
            (pgen.individual_miss.unwrap(), bed.individual_miss.unwrap()),
            (pgen.individual_het.unwrap(), bed.individual_het.unwrap()),
        ] {
            assert_eq!(pgen.len(), bed.len());
            for (lhs, rhs) in pgen.iter().zip(bed.iter()) {
                assert!((lhs - rhs).abs() <= 1e-7, "PGEN={lhs}, BED={rhs}");
            }
        }
    }

    #[test]
    fn pgen_ldscore_matches_bed_for_selected_samples_and_sites() {
        let bed_rows = vec![0x78, 0x02, 0xe4, 0x01, 0x1b, 0x02, 0x2c, 0x00];
        let dataset = TempPgenDataset::new(bed_rows.clone(), 5, 4);
        let sample_indices = [0usize, 2, 4];
        let site_indices = [0usize, 2, 3];
        let (pgen_m, pgen_ld, n_samples) = compute_pgen_ldscore(
            &dataset.pgen_path(),
            LdscWindow::Bp(250),
            Some(&sample_indices),
            Some(&site_indices),
            2,
        )
        .unwrap();
        let compact = compact_bed_selection(
            &bed_rows,
            dataset.n_samples.div_ceil(4),
            dataset.n_variants,
            Some(&site_indices),
            Some(&sample_indices),
        );
        let (expected_m, expected_ld) = compute_ldscore_core(
            &compact,
            sample_indices.len(),
            site_indices.len(),
            sample_indices.len().div_ceil(4),
            &[0, 1, 0],
            &[100, 300, 400],
            &[0.1, 0.3, 0.4],
            LdscWindow::Bp(250),
            2,
        )
        .unwrap();

        assert_eq!(n_samples, sample_indices.len());
        assert_eq!(pgen_m, expected_m);
        assert_eq!(pgen_ld.len(), expected_ld.len());
        for (lhs, rhs) in pgen_ld.iter().zip(expected_ld.iter()) {
            assert!((lhs - rhs).abs() <= 1e-10, "PGEN={lhs}, BED={rhs}");
        }
    }
}
