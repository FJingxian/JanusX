#[cfg(unix)]
use memmap2::Advice;
use memmap2::Mmap;
use numpy::ndarray::Array1;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::bedmath::{packed_byte_lut, packed_pair_lut};
use crate::gfcore;
use crate::gfreader::count_packed_row_counts as count_packed_row_counts_simd;
use crate::math_ld::{
    build_bitplanes_u64, compute_packed_row_stats, dot_nomiss_pair_bitplanes,
    dot_nomiss_pair_from_packed, r2_pairwise_complete_bitplanes, r2_pairwise_complete_from_packed,
    PackedRowStats,
};
use crate::stats_common::{get_cached_pool, map_err_string_to_py};

const LDSC_BITPLANE_MAX_MB_DEFAULT: u64 = 1024;

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

fn compute_individual_stats_core(
    packed_src: &[u8],
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    threads: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let code4_lut = &packed_byte_lut().code4;
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let block_rows = 2048usize;
    let n_blocks = n_snps.div_ceil(block_rows);

    let run = || {
        (0..n_blocks)
            .into_par_iter()
            .fold(
                || {
                    (
                        vec![0u64; n_samples],
                        vec![0u64; n_samples],
                        vec![0u64; n_samples],
                    )
                },
                |mut acc, block_idx| {
                    let (ref mut miss_ct, ref mut nonmiss_ct, ref mut het_ct) = acc;
                    let row_start = block_idx * block_rows;
                    let row_end = std::cmp::min(n_snps, row_start + block_rows);
                    for row_idx in row_start..row_end {
                        let row =
                            &packed_src[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                        accumulate_individual_row_counts(
                            row, code4_lut, full_bytes, rem, miss_ct, nonmiss_ct, het_ct,
                        );
                    }
                    acc
                },
            )
            .reduce(
                || {
                    (
                        vec![0u64; n_samples],
                        vec![0u64; n_samples],
                        vec![0u64; n_samples],
                    )
                },
                |mut left, right| {
                    let (ref mut miss_l, ref mut nonmiss_l, ref mut het_l) = left;
                    let (miss_r, nonmiss_r, het_r) = right;
                    for i in 0..n_samples {
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
#[pyo3(signature = (prefix, threads=0))]
pub fn gstats_bed_site_stats<'py>(
    py: Python<'py>,
    prefix: String,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    usize,
)> {
    let bed_prefix = normalize_plink_prefix(&prefix);
    let (maf, miss, het, n_samples) = py
        .detach(
            move || -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize), String> {
                let (mmap, n_samples, n_snps, bytes_per_snp) = open_bed_mmap(&bed_prefix)?;
                let packed_src = &mmap[3..];
                let (maf, miss, het) =
                    compute_site_stats_core(packed_src, n_samples, n_snps, bytes_per_snp, threads)?;
                Ok((maf, miss, het, n_samples))
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
    threads=0
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
) -> PyResult<(
    Option<Bound<'py, PyArray1<f32>>>,
    Option<Bound<'py, PyArray1<f32>>>,
    Option<Bound<'py, PyArray1<f32>>>,
    Option<Bound<'py, PyArray1<f32>>>,
    Option<Bound<'py, PyArray1<f32>>>,
    usize,
    usize,
)> {
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
                let (mmap, n_samples, n_snps, bytes_per_snp) = open_bed_mmap(&bed_prefix)?;
                let packed_src = &mmap[3..];
                let out = compute_joint_stats_core(
                    packed_src,
                    n_samples,
                    n_snps,
                    bytes_per_snp,
                    request,
                    threads,
                )?;
                Ok((out, n_samples, n_snps))
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
#[pyo3(signature = (prefix, threads=0))]
pub fn gstats_bed_individual_stats<'py>(
    py: Python<'py>,
    prefix: String,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, usize)> {
    let bed_prefix = normalize_plink_prefix(&prefix);
    let (miss_rate, het_rate, n_snps) = py
        .detach(move || -> Result<(Vec<f32>, Vec<f32>, usize), String> {
            let (mmap, n_samples, n_snps, bytes_per_snp) = open_bed_mmap(&bed_prefix)?;
            let packed_src = &mmap[3..];
            let (miss_rate, het_rate) = compute_individual_stats_core(
                packed_src,
                n_samples,
                n_snps,
                bytes_per_snp,
                threads,
            )?;
            Ok((miss_rate, het_rate, n_snps))
        })
        .map_err(map_err_string_to_py)?;

    let miss_arr = PyArray1::from_owned_array(py, Array1::from_vec(miss_rate)).into_bound();
    let het_arr = PyArray1::from_owned_array(py, Array1::from_vec(het_rate)).into_bound();
    Ok((miss_arr, het_arr, n_snps))
}

#[pyfunction]
#[pyo3(signature = (prefix, window_kind, window_value, threads=0))]
pub fn gstats_bed_ldscore<'py>(
    py: Python<'py>,
    prefix: String,
    window_kind: String,
    window_value: f64,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>, usize)> {
    let bed_prefix = normalize_plink_prefix(&prefix);
    let window = parse_ldsc_window(&window_kind, window_value).map_err(map_err_string_to_py)?;
    let (m_counts, ld_scores, n_samples) = py
        .detach(move || -> Result<(Vec<i64>, Vec<f64>, usize), String> {
            let (mmap, n_samples, n_snps, bytes_per_snp) = open_bed_mmap(&bed_prefix)?;
            let (chrom_codes, positions, cm_positions) = parse_bim_ldsc_meta(&bed_prefix)?;
            if chrom_codes.len() != n_snps
                || positions.len() != n_snps
                || cm_positions.len() != n_snps
            {
                return Err(format!(
                    "BED/BIM row mismatch: bed={n_snps}, bim={}",
                    chrom_codes.len()
                ));
            }
            let packed_src = &mmap[3..];
            let (m_counts, ld_scores) = compute_ldscore_core(
                packed_src,
                n_samples,
                n_snps,
                bytes_per_snp,
                chrom_codes.as_slice(),
                positions.as_slice(),
                cm_positions.as_slice(),
                window,
                threads,
            )?;
            Ok((m_counts, ld_scores, n_samples))
        })
        .map_err(map_err_string_to_py)?;

    let m_arr = PyArray1::from_owned_array(py, Array1::from_vec(m_counts)).into_bound();
    let ld_arr = PyArray1::from_owned_array(py, Array1::from_vec(ld_scores)).into_bound();
    Ok((m_arr, ld_arr, n_samples))
}
