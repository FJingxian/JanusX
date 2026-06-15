use numpy::ndarray::Array2;
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::bedmath::{decode_row_centered_full_lut, packed_byte_lut, packed_pair_lut};
use crate::blas::{
    cblas_ssyrk_dispatch, CblasInt, OpenBlasThreadGuard, CBLAS_COL_MAJOR, CBLAS_TRANS, CBLAS_UPPER,
};
use crate::math_ld::{
    build_bitplanes_u64, build_row_bitplanes_u64_with_aux, classify_ld_pair_by_maf,
    compute_packed_row_stats, dot_nomiss_pair_bitplanes, dot_nomiss_pair_from_packed,
    dot_nomiss_row_bitplanes, fill_row_bitplanes_hlav_u64,
    packed_prune_kernel_stats as packed_prune_kernel_stats_core, r2_pairwise_complete_bitplanes,
    r2_pairwise_complete_bitplanes_cached_masks, r2_pairwise_complete_from_packed, PackedRowStats,
};
use crate::stats_common::{get_cached_pool, map_err_string_to_py};

#[inline]
fn should_write_output_site(max_output_sites: Option<usize>, written: usize) -> bool {
    max_output_sites.map(|lim| written < lim).unwrap_or(true)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LdPruneMode {
    Fast,
    Strict,
}

#[inline]
fn resolve_ld_prune_mode() -> LdPruneMode {
    let raw = std::env::var("JX_LD_PRUNE_MODE")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "".to_string());
    // Strict semantics are now the default/only supported runtime behavior.
    // Legacy env tokens are accepted for compatibility and map to strict.
    match raw.as_str() {
        "strict" | "plink" | "exact" | "fast" | "" => LdPruneMode::Strict,
        _ => LdPruneMode::Strict,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LdR2Kernel {
    Blas,
    Bitwise,
}

#[inline]
fn resolve_ld_r2_kernel() -> LdR2Kernel {
    let raw = std::env::var("JX_LD_R2_KERNEL")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "".to_string());
    match raw.as_str() {
        "bitwise" | "packed" => LdR2Kernel::Bitwise,
        "blas" | "syrk" | "" => LdR2Kernel::Blas,
        _ => LdR2Kernel::Blas,
    }
}

#[inline]
fn effective_threads(threads: usize) -> usize {
    if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
    }
}

#[pyfunction]
#[pyo3(signature = (reset=false))]
pub fn packed_prune_kernel_stats(
    reset: bool,
) -> (String, bool, bool, bool, u64, u64, u64, u64, f64) {
    packed_prune_kernel_stats_core(reset)
}

fn prune_one_chrom_packed(
    idx_list: &[usize],
    pos_vec: &[i64],
    n_samples: usize,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    stats: &[PackedRowStats],
    bitplane_h: &[u64],
    bitplane_l: &[u64],
    bitplane_m: &[u64],
    bitplane_words: usize,
    bitplane_masks: &[u64],
    enable_intra_chrom_parallel: bool,
    intra_parallel_min_neighbors: usize,
    mode: LdPruneMode,
) -> Vec<bool> {
    let l = idx_list.len();
    let mut dropped = vec![false; l];
    if l <= 1 {
        return dropped;
    }

    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let n_samples_f = n_samples as f64;
    let eps = 1e-12_f64;
    let prune_r2_thresh = r2_threshold * (1.0_f64 + eps);
    let step = step_variants.max(1);
    let use_bp = window_bp.is_some();
    let mut pos_sorted = true;
    for k in 1..l {
        if pos_vec[idx_list[k]] < pos_vec[idx_list[k - 1]] {
            pos_sorted = false;
            break;
        }
    }
    let mut bp_end_ptr = 1usize;

    if mode == LdPruneMode::Fast {
        let mut block_start = 0usize;
        while block_start < l {
            let block_end = (block_start + step).min(l);
            for li in block_start..block_end {
                if dropped[li] {
                    continue;
                }
                let end = if use_bp {
                    let bp = window_bp.unwrap_or(1);
                    if pos_sorted {
                        if bp_end_ptr < li + 1 {
                            bp_end_ptr = li + 1;
                        }
                        let target = pos_vec[idx_list[li]].saturating_add(bp);
                        while bp_end_ptr < l && pos_vec[idx_list[bp_end_ptr]] <= target {
                            bp_end_ptr += 1;
                        }
                        bp_end_ptr
                    } else {
                        let mut e = li + 1;
                        let p0 = pos_vec[idx_list[li]];
                        while e < l {
                            let d = pos_vec[idx_list[e]].saturating_sub(p0);
                            if d <= bp {
                                e += 1;
                            } else if pos_vec[idx_list[e]] > p0 {
                                break;
                            } else {
                                e += 1;
                            }
                        }
                        e
                    }
                } else {
                    (li + window_variants.unwrap_or(1)).min(l)
                };
                if end <= li + 1 {
                    continue;
                }

                let gi = idx_list[li];
                let st_i = stats[gi];
                let off_i = gi * bitplane_words;
                let hi_i = &bitplane_h[off_i..off_i + bitplane_words];
                let li_i = &bitplane_l[off_i..off_i + bitplane_words];
                let mi_i = &bitplane_m[off_i..off_i + bitplane_words];
                let mut drop_i = false;
                let mut drop_low: Vec<usize> = Vec::new();
                let classify_lj = |lj: usize| -> Option<u8> {
                    if dropped[lj] {
                        return None;
                    }
                    let gj = idx_list[lj];
                    let st_j = stats[gj];
                    let off_j = gj * bitplane_words;
                    let hi_j = &bitplane_h[off_j..off_j + bitplane_words];
                    let li_j = &bitplane_l[off_j..off_j + bitplane_words];
                    let mi_j = &bitplane_m[off_j..off_j + bitplane_words];
                    let r2 = if !st_i.has_missing && !st_j.has_missing {
                        let dot_imp = dot_nomiss_pair_bitplanes(
                            gi,
                            gj,
                            bitplane_h,
                            bitplane_l,
                            bitplane_words,
                            bitplane_masks,
                        );
                        let cov = dot_imp - n_samples_f * st_i.mean * st_j.mean;
                        let denom_corr = denom * st_i.std * st_j.std;
                        let corr = if denom_corr > 0.0_f64 {
                            cov / denom_corr
                        } else {
                            0.0_f64
                        };
                        corr * corr
                    } else {
                        r2_pairwise_complete_bitplanes(
                            hi_i,
                            li_i,
                            mi_i,
                            hi_j,
                            li_j,
                            mi_j,
                            bitplane_masks,
                        )
                        .unwrap_or(f64::NAN)
                    };
                    if !(r2.is_finite() && r2 > prune_r2_thresh) {
                        return None;
                    }
                    Some(classify_ld_pair_by_maf(st_i.maf, st_j.maf, eps))
                };

                let n_neighbors = end.saturating_sub(li + 1);
                if enable_intra_chrom_parallel && n_neighbors >= intra_parallel_min_neighbors {
                    let decisions: Vec<(usize, u8)> = ((li + 1)..end)
                        .into_par_iter()
                        .filter_map(|lj| classify_lj(lj).map(|tag| (lj, tag)))
                        .collect();
                    if decisions.iter().any(|(_, tag)| *tag == 1u8) {
                        drop_i = true;
                    } else {
                        drop_low.extend(decisions.into_iter().filter_map(|(lj, tag)| {
                            if tag == 2u8 {
                                Some(lj)
                            } else {
                                None
                            }
                        }));
                    }
                } else {
                    for lj in (li + 1)..end {
                        if let Some(tag) = classify_lj(lj) {
                            if tag == 1u8 {
                                drop_i = true;
                                break;
                            }
                            drop_low.push(lj);
                        }
                    }
                }

                if drop_i {
                    dropped[li] = true;
                } else if !drop_low.is_empty() {
                    for j in drop_low {
                        dropped[j] = true;
                    }
                }
            }
            block_start = block_start.saturating_add(step);
        }
        return dropped;
    }

    let mut first_unchecked = vec![0usize; l];
    for (li, slot) in first_unchecked.iter_mut().enumerate() {
        *slot = li.saturating_add(1);
    }

    let mut block_start = 0usize;
    while block_start < l {
        let end = if use_bp {
            let bp = window_bp.unwrap_or(1);
            if pos_sorted {
                if bp_end_ptr < block_start + 1 {
                    bp_end_ptr = block_start + 1;
                }
                let target = pos_vec[idx_list[block_start]].saturating_add(bp);
                while bp_end_ptr < l && pos_vec[idx_list[bp_end_ptr]] <= target {
                    bp_end_ptr += 1;
                }
                bp_end_ptr
            } else {
                let mut e = block_start + 1;
                let p0 = pos_vec[idx_list[block_start]];
                while e < l {
                    let d = pos_vec[idx_list[e]].saturating_sub(p0);
                    if d <= bp {
                        e += 1;
                    } else if pos_vec[idx_list[e]] > p0 {
                        break;
                    } else {
                        e += 1;
                    }
                }
                e
            }
        } else {
            (block_start + window_variants.unwrap_or(1)).min(l)
        };

        if end > block_start + 1 {
            loop {
                let mut at_least_one_prune = false;
                for li in block_start..end.saturating_sub(1) {
                    if dropped[li] {
                        continue;
                    }
                    let scan_min = first_unchecked[li].max(block_start.saturating_add(1));
                    if scan_min >= end {
                        first_unchecked[li] = end;
                        continue;
                    }

                    let gi = idx_list[li];
                    let st_i = stats[gi];
                    let off_i = gi * bitplane_words;
                    let hi_i = &bitplane_h[off_i..off_i + bitplane_words];
                    let li_i = &bitplane_l[off_i..off_i + bitplane_words];
                    let mi_i = &bitplane_m[off_i..off_i + bitplane_words];
                    let mut pruned_this_round = false;
                    let mut lj = scan_min;
                    while lj < end {
                        if dropped[lj] {
                            lj = lj.saturating_add(1);
                            continue;
                        }
                        let gj = idx_list[lj];
                        let st_j = stats[gj];
                        let off_j = gj * bitplane_words;
                        let hi_j = &bitplane_h[off_j..off_j + bitplane_words];
                        let li_j = &bitplane_l[off_j..off_j + bitplane_words];
                        let mi_j = &bitplane_m[off_j..off_j + bitplane_words];
                        let r2 = if !st_i.has_missing && !st_j.has_missing {
                            let dot_imp = dot_nomiss_pair_bitplanes(
                                gi,
                                gj,
                                bitplane_h,
                                bitplane_l,
                                bitplane_words,
                                bitplane_masks,
                            );
                            let cov = dot_imp - n_samples_f * st_i.mean * st_j.mean;
                            let denom_corr = denom * st_i.std * st_j.std;
                            let corr = if denom_corr > 0.0_f64 {
                                cov / denom_corr
                            } else {
                                0.0_f64
                            };
                            corr * corr
                        } else {
                            r2_pairwise_complete_bitplanes(
                                hi_i,
                                li_i,
                                mi_i,
                                hi_j,
                                li_j,
                                mi_j,
                                bitplane_masks,
                            )
                            .unwrap_or(f64::NAN)
                        };
                        if r2.is_finite() && r2 > prune_r2_thresh {
                            at_least_one_prune = true;
                            pruned_this_round = true;
                            let tag = classify_ld_pair_by_maf(st_i.maf, st_j.maf, eps);
                            if tag == 1u8 {
                                dropped[li] = true;
                            } else {
                                dropped[lj] = true;
                                let mut nxt = lj.saturating_add(1);
                                while nxt < end && dropped[nxt] {
                                    nxt = nxt.saturating_add(1);
                                }
                                first_unchecked[li] = nxt;
                            }
                            break;
                        }
                        lj = lj.saturating_add(1);
                    }

                    if !pruned_this_round && !dropped[li] {
                        first_unchecked[li] = end;
                    }
                }
                if !at_least_one_prune {
                    break;
                }
            }
        }

        if end >= l {
            break;
        }
        block_start = block_start.saturating_add(step);
    }
    dropped
}

fn bed_packed_ld_prune_keep(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    n_samples: usize,
    chrom_vec: &[i32],
    pos_vec: &[i64],
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    threads: usize,
) -> Result<Vec<bool>, String> {
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    if m == 0 {
        return Ok(Vec::new());
    }
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        ));
    }
    if packed_flat.len() != m.saturating_mul(bytes_per_snp) {
        return Err(format!(
            "packed payload length mismatch: got {}, expected {}",
            packed_flat.len(),
            m.saturating_mul(bytes_per_snp)
        ));
    }
    if chrom_vec.len() != m {
        return Err(format!(
            "chrom_codes length mismatch: got {}, expected {m}",
            chrom_vec.len()
        ));
    }
    if pos_vec.len() != m {
        return Err(format!(
            "positions length mismatch: got {}, expected {m}",
            pos_vec.len()
        ));
    }
    if !(r2_threshold.is_finite() && r2_threshold > 0.0 && r2_threshold <= 1.0) {
        return Err("r2_threshold must be finite and in (0, 1]".to_string());
    }
    if step_variants == 0 {
        return Err("step_variants must be > 0".to_string());
    }
    if window_bp.is_none() && window_variants.is_none() {
        return Err("provide one of window_bp or window_variants".to_string());
    }
    if let Some(bp) = window_bp {
        if bp <= 0 {
            return Err("window_bp must be > 0".to_string());
        }
    }
    if let Some(wv) = window_variants {
        if wv == 0 {
            return Err("window_variants must be > 0".to_string());
        }
    }

    let mut stats = vec![PackedRowStats::default(); m];
    let byte_lut = packed_byte_lut();
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    {
        let mut run = || {
            stats.par_iter_mut().enumerate().for_each(|(row_idx, st)| {
                let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                let mut non_missing: usize = 0;
                let mut alt_sum: usize = 0;
                let mut sq_sum: usize = 0;
                for &b in row.iter().take(full_bytes) {
                    let idx = b as usize;
                    non_missing += byte_lut.nonmiss[idx] as usize;
                    alt_sum += byte_lut.alt_sum[idx] as usize;
                    sq_sum += byte_lut.sq_sum[idx] as usize;
                }
                if rem > 0 {
                    let codes = &byte_lut.code4[row[full_bytes] as usize];
                    for &code in codes.iter().take(rem) {
                        match code {
                            0b00 => {
                                non_missing += 1;
                            }
                            0b10 => {
                                non_missing += 1;
                                alt_sum += 1;
                                sq_sum += 1;
                            }
                            0b11 => {
                                non_missing += 1;
                                alt_sum += 2;
                                sq_sum += 4;
                            }
                            _ => {}
                        }
                    }
                }

                if non_missing > 0 {
                    let obs_n = non_missing as f64;
                    let sum_g = alt_sum as f64;
                    let sum_g2 = sq_sum as f64;
                    let p = sum_g / (2.0_f64 * obs_n);
                    let maf = p.min(1.0_f64 - p);
                    let mean = sum_g / obs_n;
                    let ss = (sum_g2 - (sum_g * sum_g / obs_n)).max(0.0_f64);
                    let var = ss / denom;
                    let std = var.max(1e-12_f64).sqrt();
                    *st = PackedRowStats {
                        mean,
                        std,
                        maf,
                        non_missing,
                        has_missing: non_missing < n_samples,
                    };
                } else {
                    *st = PackedRowStats {
                        mean: 0.0_f64,
                        std: 1e-6_f64,
                        maf: 0.0_f64,
                        non_missing: 0usize,
                        has_missing: true,
                    };
                }
            });
        };
        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    }

    let mut by_chr: HashMap<i32, Vec<usize>> = HashMap::new();
    for i in 0..m {
        by_chr.entry(chrom_vec[i]).or_default().push(i);
    }
    let chrom_groups: Vec<Vec<usize>> = by_chr.into_values().collect();

    let (bitplane_h, bitplane_l, bitplane_m, bitplane_words, bitplane_masks) =
        build_bitplanes_u64(packed_flat, m, bytes_per_snp, n_samples, pool.as_ref());
    let mode = resolve_ld_prune_mode();
    let worker_ct = if threads > 0 {
        threads
    } else {
        rayon::current_num_threads()
    };
    let max_chrom_group = chrom_groups.iter().map(|g| g.len()).max().unwrap_or(0usize);
    // Refined intra/outer parallel strategy:
    // - Always enable for single chromosome.
    // - Also enable when chromosome groups are few relative to workers and one
    //   group is large enough to dominate runtime.
    let enable_intra_chrom_parallel = chrom_groups.len() == 1
        || ((chrom_groups.len().saturating_mul(2) <= worker_ct) && (max_chrom_group >= 50_000));
    let intra_parallel_min_neighbors = 64usize;
    let dropped_groups: Vec<Vec<bool>> = {
        let run = || {
            chrom_groups
                .par_iter()
                .map(|idx_list| {
                    prune_one_chrom_packed(
                        idx_list,
                        pos_vec,
                        n_samples,
                        window_bp,
                        window_variants,
                        step_variants,
                        r2_threshold,
                        &stats,
                        &bitplane_h,
                        &bitplane_l,
                        &bitplane_m,
                        bitplane_words,
                        &bitplane_masks,
                        enable_intra_chrom_parallel,
                        intra_parallel_min_neighbors,
                        mode,
                    )
                })
                .collect::<Vec<Vec<bool>>>()
        };
        if let Some(tp) = &pool {
            tp.install(run)
        } else {
            run()
        }
    };

    let mut keep = vec![true; m];
    for (idx_list, dropped) in chrom_groups.iter().zip(dropped_groups.iter()) {
        for (local_idx, &global_idx) in idx_list.iter().enumerate() {
            if dropped[local_idx] {
                keep[global_idx] = false;
            }
        }
    }
    Ok(keep)
}

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

fn build_bimrange_map(
    chrom_ranges: &[String],
    start_bp: &[i64],
    end_bp: &[i64],
) -> Result<HashMap<String, Vec<(i64, i64)>>, String> {
    if chrom_ranges.len() != start_bp.len() || chrom_ranges.len() != end_bp.len() {
        return Err(format!(
            "bimrange length mismatch: chrom_ranges={}, start_bp={}, end_bp={}",
            chrom_ranges.len(),
            start_bp.len(),
            end_bp.len()
        ));
    }
    if chrom_ranges.is_empty() {
        return Err("bimrange list must not be empty".to_string());
    }
    let mut out: HashMap<String, Vec<(i64, i64)>> = HashMap::new();
    for i in 0..chrom_ranges.len() {
        let chrom = normalize_chr_token(&chrom_ranges[i]);
        let mut s = start_bp[i];
        let mut e = end_bp[i];
        if s < 0 || e < 0 {
            return Err(format!(
                "bimrange[{i}] start/end must be >= 0, got ({s}, {e})"
            ));
        }
        if s > e {
            std::mem::swap(&mut s, &mut e);
        }
        out.entry(chrom).or_default().push((s, e));
    }
    Ok(out)
}

#[inline]
fn pos_in_ranges(pos: i64, ranges: &[(i64, i64)]) -> bool {
    for &(s, e) in ranges.iter() {
        if pos >= s && pos <= e {
            return true;
        }
    }
    false
}

fn select_bim_indices_for_ld(
    prefix: &str,
    bimrange_map: &HashMap<String, Vec<(i64, i64)>>,
    selected_sites: Option<&HashSet<(String, i64)>>,
) -> Result<(usize, Vec<usize>, Vec<String>, Vec<i64>), String> {
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| format!("{bim_path}: {e}"))?;
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file);

    let mut total_snps = 0usize;
    let mut selected_idx: Vec<usize> = Vec::new();
    let mut selected_chr: Vec<String> = Vec::new();
    let mut selected_pos: Vec<i64> = Vec::new();

    for (line_no0, line) in reader.lines().enumerate() {
        let line_no = line_no0 + 1;
        let l = line.map_err(|e| format!("{bim_path}:{}: {}", line_no, e))?;
        let (chrom_raw, pos) = parse_bim_line_chrom_pos(&l, line_no, &bim_path)?;
        let chrom_norm = normalize_chr_token(&chrom_raw);
        let mut keep = false;
        if let Some(ranges) = bimrange_map.get(&chrom_norm) {
            keep = pos_in_ranges(pos, ranges);
        }
        if keep {
            if let Some(sel) = selected_sites {
                if !sel.contains(&(chrom_norm.clone(), pos)) {
                    keep = false;
                }
            }
        }
        if keep {
            selected_idx.push(total_snps);
            selected_chr.push(chrom_norm);
            selected_pos.push(pos);
        }
        total_snps = total_snps.saturating_add(1);
    }
    if total_snps == 0 {
        return Err(format!("no variant rows in {bim_path}"));
    }
    Ok((total_snps, selected_idx, selected_chr, selected_pos))
}

fn read_selected_bed_rows(
    prefix: &str,
    n_samples: usize,
    total_snps: usize,
    selected_idx: &[usize],
) -> Result<Vec<u8>, String> {
    if selected_idx.is_empty() {
        return Ok(Vec::new());
    }
    let bed_path = format!("{prefix}.bed");
    let mut bread = BufReader::new(File::open(&bed_path).map_err(|e| format!("{bed_path}: {e}"))?);
    let mut header = [0u8; 3];
    bread
        .read_exact(&mut header)
        .map_err(|e| format!("{bed_path}: {e}"))?;
    if header != [0x6C, 0x1B, 0x01] {
        return Err(format!(
            "{bed_path}: unsupported BED header (expect SNP-major 0x6C 0x1B 0x01)"
        ));
    }
    let bytes_per_snp = (n_samples + 3) / 4;
    let mut out = vec![0u8; selected_idx.len().saturating_mul(bytes_per_snp)];
    let mut row = vec![0u8; bytes_per_snp];
    let mut want_ptr = 0usize;
    for ridx in 0..total_snps {
        bread
            .read_exact(&mut row)
            .map_err(|e| format!("{bed_path}: row {}: {}", ridx + 1, e))?;
        if want_ptr < selected_idx.len() && selected_idx[want_ptr] == ridx {
            let off = want_ptr * bytes_per_snp;
            out[off..off + bytes_per_snp].copy_from_slice(&row);
            want_ptr = want_ptr.saturating_add(1);
            if want_ptr >= selected_idx.len() {
                break;
            }
        }
    }
    if want_ptr != selected_idx.len() {
        return Err(format!(
            "{bed_path}: selected rows truncated: got {}, expected {}",
            want_ptr,
            selected_idx.len()
        ));
    }
    Ok(out)
}

fn ld_r2_matrix_from_packed_rows_bitwise(
    packed_rows: &[u8],
    m: usize,
    bytes_per_snp: usize,
    n_samples: usize,
    threads: usize,
) -> Result<Vec<f32>, String> {
    if m == 0 {
        return Ok(Vec::new());
    }
    if m == 1 {
        return Ok(vec![1.0_f32]);
    }
    if packed_rows.len() != m.saturating_mul(bytes_per_snp) {
        return Err(format!(
            "packed row length mismatch: got {}, expected {}",
            packed_rows.len(),
            m.saturating_mul(bytes_per_snp)
        ));
    }
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let byte_lut = packed_byte_lut();
    let mut stats = vec![PackedRowStats::default(); m];
    {
        let mut run = || {
            stats.par_iter_mut().enumerate().for_each(|(i, st)| {
                let row = &packed_rows[i * bytes_per_snp..(i + 1) * bytes_per_snp];
                *st = compute_packed_row_stats(row, n_samples, byte_lut);
            });
        };
        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    }
    let (bit_h, bit_l, bit_m, bit_words, bit_masks) =
        build_bitplanes_u64(packed_rows, m, bytes_per_snp, n_samples, pool.as_ref());
    let mut bit_a = vec![0u64; m.saturating_mul(bit_words)];
    let mut bit_v = vec![0u64; m.saturating_mul(bit_words)];
    {
        let tail_mask = bit_masks[bit_words - 1];
        let mut run = || {
            bit_a
                .par_chunks_mut(bit_words)
                .zip(bit_v.par_chunks_mut(bit_words))
                .enumerate()
                .for_each(|(i, (arow, vrow))| {
                    let off = i * bit_words;
                    let hrow = &bit_h[off..off + bit_words];
                    let lrow = &bit_l[off..off + bit_words];
                    let mrow = &bit_m[off..off + bit_words];
                    for w in 0..bit_words {
                        arow[w] = hrow[w] & lrow[w];
                        vrow[w] = !mrow[w];
                    }
                    vrow[bit_words - 1] &= tail_mask;
                });
        };
        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    }

    let n_samples_f = n_samples as f64;
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let mut out = vec![0.0_f32; m.saturating_mul(m)];
    {
        let mut run = || {
            out.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
                row[i] = 1.0_f32;
                let st_i = stats[i];
                let off_i = i * bit_words;
                let hi_i = &bit_h[off_i..off_i + bit_words];
                let ai_i = &bit_a[off_i..off_i + bit_words];
                let vi_i = &bit_v[off_i..off_i + bit_words];
                for j in (i + 1)..m {
                    let st_j = stats[j];
                    let off_j = j * bit_words;
                    let hi_j = &bit_h[off_j..off_j + bit_words];
                    let ai_j = &bit_a[off_j..off_j + bit_words];
                    let vi_j = &bit_v[off_j..off_j + bit_words];
                    let r2 = if !st_i.has_missing && !st_j.has_missing {
                        let dot_imp =
                            dot_nomiss_pair_bitplanes(i, j, &bit_h, &bit_l, bit_words, &bit_masks);
                        let cov = dot_imp - n_samples_f * st_i.mean * st_j.mean;
                        let denom_corr = denom * st_i.std * st_j.std;
                        let corr = if denom_corr > 0.0_f64 {
                            cov / denom_corr
                        } else {
                            0.0_f64
                        };
                        corr * corr
                    } else {
                        r2_pairwise_complete_bitplanes_cached_masks(
                            hi_i, ai_i, vi_i, hi_j, ai_j, vi_j,
                        )
                        .unwrap_or(f64::NAN)
                    };
                    let rr = if r2.is_finite() {
                        r2.clamp(0.0_f64, 1.0_f64) as f32
                    } else {
                        0.0_f32
                    };
                    row[j] = rr;
                }
            });
        };
        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    }
    for i in 1..m {
        for j in 0..i {
            let v = out[j * m + i];
            out[i * m + j] = v;
        }
    }
    Ok(out)
}

fn ld_r2_matrix_from_packed_rows_blas(
    packed_rows: &[u8],
    m: usize,
    bytes_per_snp: usize,
    n_samples: usize,
    threads: usize,
) -> Result<Vec<f32>, String> {
    if m == 0 {
        return Ok(Vec::new());
    }
    if m == 1 {
        return Ok(vec![1.0_f32]);
    }
    if packed_rows.len() != m.saturating_mul(bytes_per_snp) {
        return Err(format!(
            "packed row length mismatch: got {}, expected {}",
            packed_rows.len(),
            m.saturating_mul(bytes_per_snp)
        ));
    }
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }

    let total_threads = effective_threads(threads);
    let pool = get_cached_pool(total_threads).map_err(|e| e.to_string())?;
    let byte_lut = packed_byte_lut();
    let code4_lut = &byte_lut.code4;

    // Row-major (m x n_samples) buffer. Under ColMajor + TRANS in SYRK,
    // this maps directly to C = X * X^T without an explicit transpose.
    let mut x_row_major = vec![0.0_f32; m.saturating_mul(n_samples)];
    {
        let mut run = || {
            x_row_major
                .par_chunks_mut(n_samples)
                .enumerate()
                .for_each(|(i, out_row)| {
                    let row = &packed_rows[i * bytes_per_snp..(i + 1) * bytes_per_snp];
                    let st = compute_packed_row_stats(row, n_samples, byte_lut);
                    let mean_g = st.mean as f32;
                    // Missing code (0b01) maps to 0 after centering,
                    // equivalent to mean-impute then center.
                    let value_lut = [
                        0.0_f32 - mean_g,
                        0.0_f32,
                        1.0_f32 - mean_g,
                        2.0_f32 - mean_g,
                    ];
                    decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, out_row);
                });
        };
        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    }

    let mut gram_col_major = vec![0.0_f32; m.saturating_mul(m)];
    let _blas_guard = OpenBlasThreadGuard::enter(total_threads.max(1));
    unsafe {
        cblas_ssyrk_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_UPPER,
            CBLAS_TRANS,
            m as CblasInt,
            n_samples as CblasInt,
            1.0,
            x_row_major.as_ptr(),
            n_samples as CblasInt,
            0.0,
            gram_col_major.as_mut_ptr(),
            m as CblasInt,
        );
    }

    let mut diag = vec![0.0_f32; m];
    for i in 0..m {
        diag[i] = gram_col_major[i + i * m].max(0.0_f32);
    }

    let mut out = vec![0.0_f32; m.saturating_mul(m)];
    let eps = 1e-20_f32;
    for col in 0..m {
        for row in 0..=col {
            let rr = if row == col {
                1.0_f32
            } else {
                let cov = gram_col_major[row + col * m];
                let den = (diag[row] * diag[col]).sqrt();
                let corr = if den > eps { cov / den } else { 0.0_f32 };
                let mut r2 = corr * corr;
                if !r2.is_finite() {
                    r2 = 0.0_f32;
                }
                r2.clamp(0.0_f32, 1.0_f32)
            };
            out[row * m + col] = rr;
            out[col * m + row] = rr;
        }
    }
    Ok(out)
}

fn ld_r2_matrix_from_packed_rows(
    packed_rows: &[u8],
    m: usize,
    bytes_per_snp: usize,
    n_samples: usize,
    threads: usize,
) -> Result<Vec<f32>, String> {
    match resolve_ld_r2_kernel() {
        LdR2Kernel::Bitwise => {
            ld_r2_matrix_from_packed_rows_bitwise(packed_rows, m, bytes_per_snp, n_samples, threads)
        }
        LdR2Kernel::Blas => {
            ld_r2_matrix_from_packed_rows_blas(packed_rows, m, bytes_per_snp, n_samples, threads)
                .or_else(|e_blas| {
                    ld_r2_matrix_from_packed_rows_bitwise(
                        packed_rows,
                        m,
                        bytes_per_snp,
                        n_samples,
                        threads,
                    )
                    .map_err(|e_bw| {
                        format!("LD BLAS path failed: {e_blas}; bitwise fallback failed: {e_bw}")
                    })
                })
        }
    }
}

fn read_bim_lines_chrom_pos(prefix: &str) -> Result<(Vec<String>, Vec<i32>, Vec<i64>), String> {
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| format!("{bim_path}: {e}"))?;
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut lines: Vec<String> = Vec::new();
    let mut chrom_codes: Vec<i32> = Vec::new();
    let mut positions: Vec<i64> = Vec::new();
    let mut chr_map: HashMap<String, i32> = HashMap::new();
    let mut next_code: i32 = 0;
    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{bim_path}:{}: {}", line_no + 1, e))?;
        let t = l.trim();
        if t.is_empty() {
            continue;
        }
        let mut toks = t.split_whitespace();
        let Some(chr_tok) = toks.next() else {
            continue;
        };
        let _id = toks.next();
        let _cm = toks.next();
        let Some(pos_tok) = toks.next() else {
            return Err(format!(
                "{bim_path}:{}: malformed BIM row (need >=4 columns)",
                line_no + 1
            ));
        };
        let chr_raw = chr_tok.to_string();
        let pos = pos_tok.parse::<i64>().map_err(|_| {
            format!(
                "{bim_path}:{}: invalid POS column '{}'",
                line_no + 1,
                pos_tok
            )
        })?;
        let code = if let Some(&c) = chr_map.get(&chr_raw) {
            c
        } else {
            let c = next_code;
            chr_map.insert(chr_raw, c);
            next_code = next_code.saturating_add(1);
            c
        };
        lines.push(t.to_string());
        chrom_codes.push(code);
        positions.push(pos);
    }
    if lines.is_empty() {
        return Err(format!("no variant rows in {bim_path}"));
    }
    Ok((lines, chrom_codes, positions))
}

fn read_bed_payload(prefix: &str, n_samples: usize, n_snps: usize) -> Result<Vec<u8>, String> {
    let bed_path = format!("{prefix}.bed");
    let mut f = File::open(&bed_path).map_err(|e| format!("{bed_path}: {e}"))?;
    let mut bytes = Vec::<u8>::new();
    f.read_to_end(&mut bytes)
        .map_err(|e| format!("{bed_path}: {e}"))?;
    if bytes.len() < 3 {
        return Err(format!("{bed_path}: invalid BED header"));
    }
    if bytes[0] != 0x6C || bytes[1] != 0x1B || bytes[2] != 0x01 {
        return Err(format!(
            "{bed_path}: unsupported BED header (expect SNP-major 0x6C 0x1B 0x01)"
        ));
    }
    let bps = (n_samples + 3) / 4;
    let expect = n_snps.saturating_mul(bps);
    let got = bytes.len().saturating_sub(3);
    if got != expect {
        return Err(format!(
            "{bed_path}: payload size mismatch, got {got}, expected {expect} (n_snps={n_snps}, n_samples={n_samples})"
        ));
    }
    Ok(bytes[3..].to_vec())
}

fn write_pruned_plink(
    src_prefix: &str,
    out_prefix: &str,
    keep: &[bool],
    bed_payload: &[u8],
    bytes_per_snp: usize,
    bim_lines: &[String],
    max_output_sites: Option<usize>,
) -> Result<(usize, usize), String> {
    if keep.len() != bim_lines.len() {
        return Err(format!(
            "keep mask length mismatch: keep={}, bim_rows={}",
            keep.len(),
            bim_lines.len()
        ));
    }
    if bed_payload.len() != keep.len().saturating_mul(bytes_per_snp) {
        return Err(format!(
            "bed payload mismatch: got {}, expected {}",
            bed_payload.len(),
            keep.len().saturating_mul(bytes_per_snp)
        ));
    }
    let out_bed = format!("{out_prefix}.bed");
    let out_bim = format!("{out_prefix}.bim");
    let out_fam = format!("{out_prefix}.fam");
    let src_fam = format!("{src_prefix}.fam");
    if let Some(parent) = Path::new(&out_bed).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    let mut wbed = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&out_bed).map_err(|e| format!("{out_bed}: {e}"))?,
    );
    wbed.write_all(&[0x6C, 0x1B, 0x01])
        .map_err(|e| format!("{out_bed}: {e}"))?;
    let mut wbim = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&out_bim).map_err(|e| format!("{out_bim}: {e}"))?,
    );

    let mut kept = 0usize;
    let mut written = 0usize;
    for i in 0..keep.len() {
        if !keep[i] {
            continue;
        }
        kept = kept.saturating_add(1);
        let write_this = should_write_output_site(max_output_sites, written);
        if !write_this {
            continue;
        }
        let s = i.saturating_mul(bytes_per_snp);
        let e = s.saturating_add(bytes_per_snp);
        wbed.write_all(&bed_payload[s..e])
            .map_err(|e| format!("{out_bed}: {e}"))?;
        wbim.write_all(bim_lines[i].as_bytes())
            .and_then(|_| wbim.write_all(b"\n"))
            .map_err(|e| format!("{out_bim}: {e}"))?;
        written = written.saturating_add(1);
    }
    wbed.flush().map_err(|e| format!("{out_bed}: {e}"))?;
    wbim.flush().map_err(|e| format!("{out_bim}: {e}"))?;

    fs::copy(&src_fam, &out_fam).map_err(|e| format!("{src_fam} -> {out_fam}: {e}"))?;
    Ok((kept, keep.len()))
}

#[inline]
fn validate_ld_prune_args(
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
) -> Result<(), String> {
    if !(r2_threshold.is_finite() && r2_threshold > 0.0 && r2_threshold <= 1.0) {
        return Err("r2_threshold must be finite and in (0, 1]".to_string());
    }
    if step_variants == 0 {
        return Err("step_variants must be > 0".to_string());
    }
    if window_bp.is_none() && window_variants.is_none() {
        return Err("provide one of window_bp or window_variants".to_string());
    }
    if let Some(bp) = window_bp {
        if bp <= 0 {
            return Err("window_bp must be > 0".to_string());
        }
    }
    if let Some(wv) = window_variants {
        if wv == 0 {
            return Err("window_variants must be > 0".to_string());
        }
    }
    Ok(())
}

fn parse_bim_line_chrom_pos(
    line: &str,
    line_no: usize,
    bim_path: &str,
) -> Result<(String, i64), String> {
    let t = line.trim();
    if t.is_empty() {
        return Err(format!("{bim_path}:{}: empty BIM row", line_no));
    }
    let mut toks = t.split_whitespace();
    let Some(chrom) = toks.next() else {
        return Err(format!("{bim_path}:{}: empty BIM row", line_no));
    };
    let _id = toks.next();
    let _cm = toks.next();
    let Some(pos_tok) = toks.next() else {
        return Err(format!(
            "{bim_path}:{}: malformed BIM row (need >=4 columns)",
            line_no
        ));
    };
    let pos = pos_tok
        .parse::<i64>()
        .map_err(|_| format!("{bim_path}:{}: invalid POS column '{}'", line_no, pos_tok))?;
    Ok((chrom.to_string(), pos))
}

#[derive(Clone)]
struct BimChromBlock {
    chrom: String,
    start_idx: usize,
    len: usize,
    bim_start: u64,
    bim_end: u64,
    pos_sorted: bool,
}

struct BimStreamScan {
    total_snps: usize,
    chrom_last_index: HashMap<String, usize>,
    chrom_pos_sorted: bool,
    chrom_contiguous: bool,
    blocks: Vec<BimChromBlock>,
}

fn scan_bim_stream_blocks(prefix: &str) -> Result<BimStreamScan, String> {
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| format!("{bim_path}: {e}"))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut chrom_last_index: HashMap<String, usize> = HashMap::new();
    let mut last_pos_by_chrom: HashMap<String, i64> = HashMap::new();
    let mut closed_chroms: HashSet<String> = HashSet::new();
    let mut blocks: Vec<BimChromBlock> = Vec::new();
    let mut chrom_pos_sorted = true;
    let mut chrom_contiguous = true;
    let mut total = 0usize;
    let mut byte_pos = 0u64;
    let mut line = String::new();
    let mut current_chrom: Option<String> = None;
    let mut block_start_idx = 0usize;
    let mut block_start_byte = 0u64;
    let mut block_len = 0usize;
    let mut block_pos_sorted = true;
    let mut block_last_pos: Option<i64> = None;

    loop {
        line.clear();
        let line_start = byte_pos;
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("{bim_path}:{}: {}", total + 1, e))?;
        if n == 0 {
            break;
        }
        byte_pos = byte_pos.saturating_add(n as u64);
        let (chrom, pos) = parse_bim_line_chrom_pos(&line, total + 1, &bim_path)?;

        let starts_new_block = current_chrom.as_ref().map(|c| c != &chrom).unwrap_or(true);
        if starts_new_block {
            if let Some(prev_chrom) = current_chrom.take() {
                closed_chroms.insert(prev_chrom.clone());
                blocks.push(BimChromBlock {
                    chrom: prev_chrom,
                    start_idx: block_start_idx,
                    len: block_len,
                    bim_start: block_start_byte,
                    bim_end: line_start,
                    pos_sorted: block_pos_sorted,
                });
            }
            if closed_chroms.contains(&chrom) {
                chrom_contiguous = false;
            }
            current_chrom = Some(chrom.clone());
            block_start_idx = total;
            block_start_byte = line_start;
            block_len = 0usize;
            block_pos_sorted = true;
            block_last_pos = None;
        }

        if let Some(prev) = last_pos_by_chrom.insert(chrom.clone(), pos) {
            if pos < prev {
                chrom_pos_sorted = false;
            }
        }
        if let Some(prev) = block_last_pos.replace(pos) {
            if pos < prev {
                block_pos_sorted = false;
            }
        }
        chrom_last_index.insert(chrom, total);
        block_len = block_len.saturating_add(1);
        total = total.saturating_add(1);
    }

    if let Some(prev_chrom) = current_chrom.take() {
        blocks.push(BimChromBlock {
            chrom: prev_chrom,
            start_idx: block_start_idx,
            len: block_len,
            bim_start: block_start_byte,
            bim_end: byte_pos,
            pos_sorted: block_pos_sorted,
        });
    }

    if total == 0 {
        return Err(format!("no variant rows in {bim_path}"));
    }
    Ok(BimStreamScan {
        total_snps: total,
        chrom_last_index,
        chrom_pos_sorted,
        chrom_contiguous,
        blocks,
    })
}

struct StreamingPruneBufferedRow {
    pos: i64,
    chrom_seq: usize,
    first_unchecked_seq: usize,
    packed: Vec<u8>,
    bim_line: String,
    stats: PackedRowStats,
    bit_h: Option<Vec<u64>>,
    bit_l: Option<Vec<u64>>,
    bit_m: Option<Vec<u64>>,
    bit_a: Option<Vec<u64>>,
    bit_v: Option<Vec<u64>>,
    dropped: bool,
    finalized: bool,
}

impl StreamingPruneBufferedRow {
    fn mark_dropped(&mut self) {
        self.dropped = true;
        self.finalized = true;
        self.packed.clear();
        self.bim_line.clear();
        self.bit_h = None;
        self.bit_l = None;
        self.bit_m = None;
        self.bit_a = None;
        self.bit_v = None;
    }
}

#[inline]
fn mark_streaming_row_dropped(rows: &mut [Option<StreamingPruneBufferedRow>], rid: usize) -> bool {
    if let Some(Some(row)) = rows.get_mut(rid) {
        if !row.finalized && !row.dropped {
            row.mark_dropped();
            return true;
        }
    }
    false
}

#[inline]
fn row_expired_from_window(
    row: &StreamingPruneBufferedRow,
    current_pos: i64,
    current_chrom_seq: usize,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
) -> bool {
    if let Some(bp) = window_bp {
        current_pos > row.pos.saturating_add(bp)
    } else {
        current_chrom_seq >= row.chrom_seq.saturating_add(window_variants.unwrap_or(1))
    }
}

fn finalize_expired_streaming_rows(
    active: &mut VecDeque<usize>,
    rows: &mut [Option<StreamingPruneBufferedRow>],
    current_pos: i64,
    current_chrom_seq: usize,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
) {
    loop {
        let Some(&rid) = active.front() else {
            break;
        };
        let should_pop = match rows.get_mut(rid) {
            Some(Some(row)) => {
                if row.finalized {
                    true
                } else if row_expired_from_window(
                    row,
                    current_pos,
                    current_chrom_seq,
                    window_bp,
                    window_variants,
                ) {
                    row.finalized = true;
                    true
                } else {
                    false
                }
            }
            _ => true,
        };
        if should_pop {
            active.pop_front();
        } else {
            break;
        }
    }
}

#[inline]
fn finalize_all_streaming_rows(
    active: &mut VecDeque<usize>,
    rows: &mut [Option<StreamingPruneBufferedRow>],
) {
    while let Some(rid) = active.pop_front() {
        if let Some(Some(row)) = rows.get_mut(rid) {
            row.finalized = true;
        }
    }
}

fn finalize_streaming_rows_before_seq(
    active: &mut VecDeque<usize>,
    rows: &mut [Option<StreamingPruneBufferedRow>],
    min_seq: usize,
) {
    loop {
        let Some(&rid) = active.front() else {
            break;
        };
        let should_pop = match rows.get_mut(rid) {
            Some(Some(row)) => {
                if row.finalized {
                    true
                } else if row.chrom_seq < min_seq {
                    row.finalized = true;
                    true
                } else {
                    false
                }
            }
            _ => true,
        };
        if should_pop {
            active.pop_front();
        } else {
            break;
        }
    }
}

#[inline]
fn lower_bound_window_seq(
    rows: &[Option<StreamingPruneBufferedRow>],
    window_ids: &[usize],
    start_idx: usize,
    min_seq: usize,
) -> usize {
    let mut lo = start_idx.min(window_ids.len());
    let mut hi = window_ids.len();
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        let rid = window_ids[mid];
        let seq = rows
            .get(rid)
            .and_then(|x| x.as_ref())
            .map(|row| row.chrom_seq)
            .unwrap_or(usize::MAX);
        if seq < min_seq {
            lo = mid.saturating_add(1);
        } else {
            hi = mid;
        }
    }
    lo
}

fn classify_streaming_pair_by_maf(
    rows: &[Option<StreamingPruneBufferedRow>],
    i_has_missing: bool,
    i_non_missing: usize,
    i_mean_scaled: f64,
    i_corr_scale: f64,
    i_maf: f64,
    ih: &[u64],
    il: &[u64],
    ia: &[u64],
    iv: &[u64],
    rid_j: usize,
    prune_r2_thresh: f64,
    eps: f64,
    word_masks: &[u64],
) -> Option<u8> {
    let row_j = rows.get(rid_j)?.as_ref()?;
    if row_j.finalized || row_j.dropped {
        return None;
    }

    let (jh, jl, ja, jv) = match (
        row_j.bit_h.as_deref(),
        row_j.bit_l.as_deref(),
        row_j.bit_a.as_deref(),
        row_j.bit_v.as_deref(),
    ) {
        (Some(h), Some(l), Some(a), Some(v)) => (h, l, a, v),
        _ => return None,
    };

    let r2 = if !i_has_missing && !row_j.stats.has_missing {
        let dot_imp = dot_nomiss_row_bitplanes(ih, il, jh, jl, word_masks);
        let cov = dot_imp - i_mean_scaled * row_j.stats.mean;
        let denom_corr = i_corr_scale * row_j.stats.std;
        let corr = if denom_corr > 0.0_f64 {
            cov / denom_corr
        } else {
            0.0_f64
        };
        corr * corr
    } else {
        if i_non_missing <= 1 || row_j.stats.non_missing <= 1 {
            return None;
        }
        r2_pairwise_complete_bitplanes_cached_masks(ih, ia, iv, jh, ja, jv).unwrap_or(f64::NAN)
    };
    if !(r2.is_finite() && r2 > prune_r2_thresh) {
        return None;
    }

    Some(classify_ld_pair_by_maf(i_maf, row_j.stats.maf, eps))
}

fn drain_finalized_streaming_rows(
    rows: &mut Vec<Option<StreamingPruneBufferedRow>>,
    next_write: &mut usize,
    wbed: &mut BufWriter<File>,
    wbim: &mut BufWriter<File>,
    kept: &mut usize,
    written: &mut usize,
    max_output_sites: Option<usize>,
    out_bed: &str,
    out_bim: &str,
) -> Result<(), String> {
    while *next_write < rows.len() {
        let ready = rows[*next_write]
            .as_ref()
            .map(|row| row.finalized)
            .unwrap_or(true);
        if !ready {
            break;
        }
        if let Some(row) = rows[*next_write].take() {
            if !row.dropped {
                *kept = kept.saturating_add(1);
                let write_this = should_write_output_site(max_output_sites, *written);
                if write_this {
                    wbed.write_all(&row.packed)
                        .map_err(|e| format!("{out_bed}: {e}"))?;
                    wbim.write_all(row.bim_line.as_bytes())
                        .and_then(|_| wbim.write_all(b"\n"))
                        .map_err(|e| format!("{out_bim}: {e}"))?;
                    *written = written.saturating_add(1);
                }
            }
        }
        *next_write = (*next_write).saturating_add(1);
    }
    Ok(())
}

struct StreamChromRow {
    first_unchecked_seq: usize,
    pos: i64,
    packed: Vec<u8>,
    bim_line: String,
    stats: PackedRowStats,
    bits: Vec<u64>,
    dropped: bool,
}

struct ChromPruneTempResult {
    block_idx: usize,
    kept: usize,
    temp_bed: Option<String>,
    temp_bim: Option<String>,
}

#[inline]
fn stream_row_alive(rows: &[StreamChromRow], base_seq: usize, seq: usize) -> bool {
    if seq < base_seq {
        return false;
    }
    let idx = seq - base_seq;
    rows.get(idx).map(|r| !r.dropped).unwrap_or(false)
}

#[inline]
fn stream_chrom_bit_slices(bits: &[u64]) -> Option<(&[u64], &[u64], &[u64], &[u64])> {
    let words = bits.len() / 4;
    if words == 0 || words.saturating_mul(4) != bits.len() {
        return None;
    }
    Some((
        &bits[..words],
        &bits[words..words.saturating_mul(2)],
        &bits[words.saturating_mul(2)..words.saturating_mul(3)],
        &bits[words.saturating_mul(3)..],
    ))
}

#[inline]
fn drop_stream_chrom_row(row: &mut StreamChromRow, bit_pool: &mut Vec<Vec<u64>>) {
    row.dropped = true;
    row.packed.clear();
    row.bim_line.clear();
    let mut bits = std::mem::take(&mut row.bits);
    bits.clear();
    if bits.capacity() > 0 {
        bit_pool.push(bits);
    }
}

fn classify_stream_chrom_pair_by_maf(
    row_i: &StreamChromRow,
    row_j: &StreamChromRow,
    n_samples_f: f64,
    denom: f64,
    prune_r2_thresh: f64,
    eps: f64,
    word_masks: &[u64],
) -> Option<u8> {
    let (ih, il, ia, iv) = stream_chrom_bit_slices(&row_i.bits)?;
    let (jh, jl, ja, jv) = stream_chrom_bit_slices(&row_j.bits)?;
    let r2 = if !row_i.stats.has_missing && !row_j.stats.has_missing {
        let dot_imp = dot_nomiss_row_bitplanes(ih, il, jh, jl, word_masks);
        let cov = dot_imp - n_samples_f * row_i.stats.mean * row_j.stats.mean;
        let denom_corr = denom * row_i.stats.std * row_j.stats.std;
        let corr = if denom_corr > 0.0_f64 {
            cov / denom_corr
        } else {
            0.0_f64
        };
        corr * corr
    } else {
        if row_i.stats.non_missing <= 1 || row_j.stats.non_missing <= 1 {
            return None;
        }
        r2_pairwise_complete_bitplanes_cached_masks(ih, ia, iv, jh, ja, jv).unwrap_or(f64::NAN)
    };
    if !(r2.is_finite() && r2 > prune_r2_thresh) {
        return None;
    }
    Some(classify_ld_pair_by_maf(
        row_i.stats.maf,
        row_j.stats.maf,
        eps,
    ))
}

fn flush_stream_chrom_rows(
    rows: &mut Vec<StreamChromRow>,
    base_seq: &mut usize,
    flush_before: usize,
    kept: &mut usize,
    bit_pool: &mut Vec<Vec<u64>>,
    wbed: Option<&mut BufWriter<File>>,
    wbim: Option<&mut BufWriter<File>>,
    out_bed: &str,
    out_bim: &str,
) -> Result<(), String> {
    if flush_before <= *base_seq {
        return Ok(());
    }
    let drain_n = flush_before.saturating_sub(*base_seq).min(rows.len());
    if drain_n == 0 {
        return Ok(());
    }
    let mut wbed_opt = wbed;
    let mut wbim_opt = wbim;
    for mut row in rows.drain(0..drain_n) {
        if row.dropped {
            let mut bits = std::mem::take(&mut row.bits);
            bits.clear();
            if bits.capacity() > 0 {
                bit_pool.push(bits);
            }
            continue;
        }
        *kept = kept.saturating_add(1);
        if let (Some(wbed_ref), Some(wbim_ref)) = (wbed_opt.as_deref_mut(), wbim_opt.as_deref_mut())
        {
            wbed_ref
                .write_all(&row.packed)
                .map_err(|e| format!("{out_bed}: {e}"))?;
            wbim_ref
                .write_all(row.bim_line.as_bytes())
                .and_then(|_| wbim_ref.write_all(b"\n"))
                .map_err(|e| format!("{out_bim}: {e}"))?;
        }
        let mut bits = std::mem::take(&mut row.bits);
        bits.clear();
        if bits.capacity() > 0 {
            bit_pool.push(bits);
        }
    }
    *base_seq = base_seq.saturating_add(drain_n);
    Ok(())
}

fn process_stream_chrom_ready_windows(
    rows: &mut Vec<StreamChromRow>,
    base_seq: &mut usize,
    seen_count: usize,
    chrom_done: bool,
    next_start: &mut usize,
    bp_scan: &mut usize,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step: usize,
    r2_threshold: f64,
    n_samples_f: f64,
    denom: f64,
    word_masks: &[u64],
    kept: &mut usize,
    bit_pool: &mut Vec<Vec<u64>>,
    mut wbed: Option<&mut BufWriter<File>>,
    mut wbim: Option<&mut BufWriter<File>>,
    out_bed: &str,
    out_bim: &str,
) -> Result<(), String> {
    let eps = 1e-12_f64;
    let prune_r2_thresh = r2_threshold * (1.0_f64 + eps);
    loop {
        if *next_start >= seen_count {
            break;
        }
        if *next_start < *base_seq {
            *next_start = *base_seq;
        }
        let (window_end_seq, ready) = if let Some(bp) = window_bp {
            let start_idx = next_start.saturating_sub(*base_seq);
            let Some(start_row) = rows.get(start_idx) else {
                break;
            };
            let target = start_row.pos.saturating_add(bp);
            if *bp_scan < next_start.saturating_add(1) {
                *bp_scan = next_start.saturating_add(1);
            }
            while *bp_scan < seen_count {
                let idx = bp_scan.saturating_sub(*base_seq);
                let Some(row) = rows.get(idx) else {
                    break;
                };
                if row.pos > target {
                    break;
                }
                *bp_scan = bp_scan.saturating_add(1);
            }
            let ready_now = chrom_done || (*bp_scan < seen_count);
            let end_seq = if ready_now { *bp_scan } else { seen_count };
            (end_seq, ready_now)
        } else {
            let want_end = next_start.saturating_add(window_variants.unwrap_or(1));
            let end_seq = want_end.min(seen_count);
            let ready_now = chrom_done || (seen_count >= want_end);
            (end_seq, ready_now)
        };
        if !ready {
            break;
        }

        if window_end_seq > next_start.saturating_add(1) {
            let mut window_seqs: Vec<usize> = ((*next_start)..window_end_seq)
                .filter(|seq| stream_row_alive(rows, *base_seq, *seq))
                .collect();
            loop {
                let mut keep_n = 0usize;
                for read_idx in 0..window_seqs.len() {
                    let seq = window_seqs[read_idx];
                    if stream_row_alive(rows, *base_seq, seq) {
                        if keep_n != read_idx {
                            window_seqs[keep_n] = seq;
                        }
                        keep_n = keep_n.saturating_add(1);
                    }
                }
                window_seqs.truncate(keep_n);
                if window_seqs.len() <= 1 {
                    break;
                }

                let mut at_least_one_prune = false;
                for wi in 0..window_seqs.len().saturating_sub(1) {
                    let seq_i = window_seqs[wi];
                    if !stream_row_alive(rows, *base_seq, seq_i) {
                        continue;
                    }
                    let i_idx = seq_i.saturating_sub(*base_seq);
                    let scan_min_seq = rows[i_idx]
                        .first_unchecked_seq
                        .max(next_start.saturating_add(1));
                    if scan_min_seq >= window_end_seq {
                        rows[i_idx].first_unchecked_seq =
                            rows[i_idx].first_unchecked_seq.max(window_end_seq);
                        continue;
                    }

                    let wj_start = window_seqs.partition_point(|seq| *seq < scan_min_seq);
                    let mut prune_hit: Option<(u8, usize, usize)> = None;
                    for wj in wj_start..window_seqs.len() {
                        let seq_j = window_seqs[wj];
                        if seq_j >= window_end_seq {
                            break;
                        }
                        if !stream_row_alive(rows, *base_seq, seq_j) {
                            continue;
                        }
                        let j_idx = seq_j.saturating_sub(*base_seq);
                        if let Some(tag) = classify_stream_chrom_pair_by_maf(
                            &rows[i_idx],
                            &rows[j_idx],
                            n_samples_f,
                            denom,
                            prune_r2_thresh,
                            eps,
                            word_masks,
                        ) {
                            prune_hit = Some((tag, wj, seq_j));
                            break;
                        }
                    }

                    if let Some((tag, wj, seq_j)) = prune_hit {
                        at_least_one_prune = true;
                        if tag == 1u8 {
                            drop_stream_chrom_row(&mut rows[i_idx], bit_pool);
                        } else {
                            let j_idx = seq_j.saturating_sub(*base_seq);
                            drop_stream_chrom_row(&mut rows[j_idx], bit_pool);
                            let mut next_seq = window_end_seq;
                            for &candidate_seq in window_seqs.iter().skip(wj + 1) {
                                if candidate_seq >= window_end_seq {
                                    break;
                                }
                                if stream_row_alive(rows, *base_seq, candidate_seq) {
                                    next_seq = candidate_seq;
                                    break;
                                }
                            }
                            if stream_row_alive(rows, *base_seq, seq_i) {
                                rows[i_idx].first_unchecked_seq =
                                    rows[i_idx].first_unchecked_seq.max(next_seq);
                            }
                        }
                    } else if stream_row_alive(rows, *base_seq, seq_i) {
                        rows[i_idx].first_unchecked_seq =
                            rows[i_idx].first_unchecked_seq.max(window_end_seq);
                    }
                }
                if !at_least_one_prune {
                    break;
                }
            }
        }

        *next_start = next_start.saturating_add(step);
        let flush_before = (*next_start).min(seen_count);
        flush_stream_chrom_rows(
            rows,
            base_seq,
            flush_before,
            kept,
            bit_pool,
            wbed.as_deref_mut(),
            wbim.as_deref_mut(),
            out_bed,
            out_bim,
        )?;
    }
    Ok(())
}

fn notify_chrom_prune_progress(
    callback: Option<&Arc<Mutex<Py<PyAny>>>>,
    block_idx: usize,
    chrom: &str,
    done: usize,
    total: usize,
    finished: bool,
) -> Result<(), String> {
    let Some(cb_arc) = callback else {
        return Ok(());
    };
    let cb_guard = cb_arc
        .lock()
        .map_err(|_| "chromosome progress callback lock poisoned".to_string())?;
    Python::attach(|py2| -> PyResult<()> {
        py2.check_signals()?;
        cb_guard.call1(py2, (block_idx, chrom, done, total, finished))?;
        Ok(())
    })
    .map_err(|e| e.to_string())
}

fn process_stream_chrom_block(
    src_prefix: &str,
    out_prefix: &str,
    block: &BimChromBlock,
    block_idx: usize,
    n_samples: usize,
    bytes_per_snp: usize,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    word_masks: &[u64],
    write_temp: bool,
    chrom_progress_callback: Option<&Arc<Mutex<Py<PyAny>>>>,
    progress_every: usize,
) -> Result<ChromPruneTempResult, String> {
    let src_bed = format!("{src_prefix}.bed");
    let src_bim = format!("{src_prefix}.bim");
    let pid = std::process::id();
    let temp_bed = if write_temp {
        Some(format!("{out_prefix}.jxprune_chr_{pid}_{block_idx}.bedtmp"))
    } else {
        None
    };
    let temp_bim = if write_temp {
        Some(format!("{out_prefix}.jxprune_chr_{pid}_{block_idx}.bimtmp"))
    } else {
        None
    };

    let mut bed_file = File::open(&src_bed).map_err(|e| format!("{src_bed}: {e}"))?;
    let bed_offset = 3u64
        .checked_add(
            (block.start_idx as u64)
                .checked_mul(bytes_per_snp as u64)
                .ok_or_else(|| "BED row offset overflow".to_string())?,
        )
        .ok_or_else(|| "BED row offset overflow".to_string())?;
    bed_file
        .seek(SeekFrom::Start(bed_offset))
        .map_err(|e| format!("{src_bed}: {e}"))?;
    let mut bed_reader = BufReader::with_capacity(8 * 1024 * 1024, bed_file);

    let mut bim_file = File::open(&src_bim).map_err(|e| format!("{src_bim}: {e}"))?;
    bim_file
        .seek(SeekFrom::Start(block.bim_start))
        .map_err(|e| format!("{src_bim}: {e}"))?;
    let mut bim_reader = BufReader::with_capacity(4 * 1024 * 1024, bim_file);

    if let Some(parent) = Path::new(out_prefix).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let mut wbed = match temp_bed.as_ref() {
        Some(path) => Some(BufWriter::with_capacity(
            8 * 1024 * 1024,
            File::create(path).map_err(|e| format!("{path}: {e}"))?,
        )),
        None => None,
    };
    let mut wbim = match temp_bim.as_ref() {
        Some(path) => Some(BufWriter::with_capacity(
            4 * 1024 * 1024,
            File::create(path).map_err(|e| format!("{path}: {e}"))?,
        )),
        None => None,
    };

    let byte_lut = packed_byte_lut();
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let n_samples_f = n_samples as f64;
    let step = step_variants.max(1);
    let mut rows: Vec<StreamChromRow> = Vec::new();
    let mut bit_pool: Vec<Vec<u64>> = Vec::new();
    let mut base_seq = 0usize;
    let mut seen_count = 0usize;
    let mut next_start = 0usize;
    let mut bp_scan = 0usize;
    let mut kept = 0usize;
    let mut row_buf = vec![0u8; bytes_per_snp];
    let mut line_buf = String::new();
    let mut consumed_bim = block.bim_start;
    let notify_step = if progress_every == 0 {
        (block.len / 100).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;

    notify_chrom_prune_progress(
        chrom_progress_callback,
        block_idx,
        &block.chrom,
        0usize,
        block.len,
        false,
    )?;

    for local_seq in 0..block.len {
        line_buf.clear();
        let n_line = bim_reader
            .read_line(&mut line_buf)
            .map_err(|e| format!("{src_bim}:{}: {}", block.start_idx + local_seq + 1, e))?;
        if n_line == 0 {
            return Err(format!(
                "{src_bim}: unexpected EOF at variant row {}",
                block.start_idx + local_seq + 1
            ));
        }
        consumed_bim = consumed_bim.saturating_add(n_line as u64);
        if consumed_bim > block.bim_end {
            return Err(format!(
                "{src_bim}: chromosome block {} exceeded BIM byte range",
                block.chrom
            ));
        }
        let bim_line_trimmed = line_buf.trim_end_matches(&['\r', '\n'][..]);
        let (_chrom, pos) =
            parse_bim_line_chrom_pos(bim_line_trimmed, block.start_idx + local_seq + 1, &src_bim)?;
        bed_reader
            .read_exact(&mut row_buf)
            .map_err(|e| format!("{src_bed}: row {}: {}", block.start_idx + local_seq + 1, e))?;

        let stats = compute_packed_row_stats(&row_buf, n_samples, byte_lut);
        let mut bits = bit_pool.pop().unwrap_or_default();
        fill_row_bitplanes_hlav_u64(&row_buf, n_samples, &mut bits);
        rows.push(StreamChromRow {
            first_unchecked_seq: seen_count.saturating_add(1),
            pos,
            packed: if write_temp {
                row_buf.to_vec()
            } else {
                Vec::new()
            },
            bim_line: if write_temp {
                bim_line_trimmed.to_string()
            } else {
                String::new()
            },
            stats,
            bits,
            dropped: false,
        });
        seen_count = seen_count.saturating_add(1);
        process_stream_chrom_ready_windows(
            &mut rows,
            &mut base_seq,
            seen_count,
            false,
            &mut next_start,
            &mut bp_scan,
            window_bp,
            window_variants,
            step,
            r2_threshold,
            n_samples_f,
            denom,
            word_masks,
            &mut kept,
            &mut bit_pool,
            wbed.as_mut(),
            wbim.as_mut(),
            temp_bed.as_deref().unwrap_or(""),
            temp_bim.as_deref().unwrap_or(""),
        )?;
        let done = local_seq.saturating_add(1);
        if done >= last_notified.saturating_add(notify_step) || done == block.len {
            last_notified = done;
            notify_chrom_prune_progress(
                chrom_progress_callback,
                block_idx,
                &block.chrom,
                done,
                block.len,
                done == block.len,
            )?;
        }
    }

    process_stream_chrom_ready_windows(
        &mut rows,
        &mut base_seq,
        seen_count,
        true,
        &mut next_start,
        &mut bp_scan,
        window_bp,
        window_variants,
        step,
        r2_threshold,
        n_samples_f,
        denom,
        word_masks,
        &mut kept,
        &mut bit_pool,
        wbed.as_mut(),
        wbim.as_mut(),
        temp_bed.as_deref().unwrap_or(""),
        temp_bim.as_deref().unwrap_or(""),
    )?;
    flush_stream_chrom_rows(
        &mut rows,
        &mut base_seq,
        seen_count,
        &mut kept,
        &mut bit_pool,
        wbed.as_mut(),
        wbim.as_mut(),
        temp_bed.as_deref().unwrap_or(""),
        temp_bim.as_deref().unwrap_or(""),
    )?;
    if !rows.is_empty() {
        return Err(format!(
            "streaming chromosome block {} finished with unflushed rows",
            block.chrom
        ));
    }
    if let Some(w) = wbed.as_mut() {
        w.flush()
            .map_err(|e| format!("{}: {e}", temp_bed.as_deref().unwrap_or("")))?;
    }
    if let Some(w) = wbim.as_mut() {
        w.flush()
            .map_err(|e| format!("{}: {e}", temp_bim.as_deref().unwrap_or("")))?;
    }

    Ok(ChromPruneTempResult {
        block_idx,
        kept,
        temp_bed,
        temp_bim,
    })
}

fn copy_exact_bytes_limited(
    src_path: &str,
    dst: &mut BufWriter<File>,
    n_bytes: usize,
) -> Result<(), String> {
    let mut src = BufReader::with_capacity(
        8 * 1024 * 1024,
        File::open(src_path).map_err(|e| format!("{src_path}: {e}"))?,
    );
    let mut remaining = n_bytes;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    while remaining > 0 {
        let take = remaining.min(buf.len());
        src.read_exact(&mut buf[..take])
            .map_err(|e| format!("{src_path}: {e}"))?;
        dst.write_all(&buf[..take])
            .map_err(|e| format!("{src_path}: {e}"))?;
        remaining -= take;
    }
    Ok(())
}

fn copy_bim_lines_limited(
    src_path: &str,
    dst: &mut BufWriter<File>,
    n_lines: usize,
) -> Result<(), String> {
    let mut src = BufReader::with_capacity(
        4 * 1024 * 1024,
        File::open(src_path).map_err(|e| format!("{src_path}: {e}"))?,
    );
    let mut line = String::new();
    for _ in 0..n_lines {
        line.clear();
        let n = src
            .read_line(&mut line)
            .map_err(|e| format!("{src_path}: {e}"))?;
        if n == 0 {
            return Err(format!("{src_path}: unexpected EOF while copying BIM rows"));
        }
        dst.write_all(line.as_bytes())
            .map_err(|e| format!("{src_path}: {e}"))?;
    }
    Ok(())
}

fn concat_stream_chrom_results_to_plink(
    src_prefix: &str,
    out_prefix: &str,
    results: &[ChromPruneTempResult],
    bytes_per_snp: usize,
    max_output_sites: Option<usize>,
) -> Result<(), String> {
    let out_bed = format!("{out_prefix}.bed");
    let out_bim = format!("{out_prefix}.bim");
    let out_fam = format!("{out_prefix}.fam");
    let src_fam = format!("{src_prefix}.fam");
    if let Some(parent) = Path::new(&out_bed).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let mut wbed = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&out_bed).map_err(|e| format!("{out_bed}: {e}"))?,
    );
    let mut wbim = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&out_bim).map_err(|e| format!("{out_bim}: {e}"))?,
    );
    wbed.write_all(&[0x6C, 0x1B, 0x01])
        .map_err(|e| format!("{out_bed}: {e}"))?;

    let mut written = 0usize;
    let max_rows = max_output_sites.unwrap_or(usize::MAX);
    for result in results {
        if written >= max_rows {
            break;
        }
        let rows_to_copy = result.kept.min(max_rows - written);
        if rows_to_copy == 0 {
            continue;
        }
        let bed_path = result
            .temp_bed
            .as_deref()
            .ok_or_else(|| "missing chromosome temporary BED output".to_string())?;
        let bim_path = result
            .temp_bim
            .as_deref()
            .ok_or_else(|| "missing chromosome temporary BIM output".to_string())?;
        copy_exact_bytes_limited(
            bed_path,
            &mut wbed,
            rows_to_copy
                .checked_mul(bytes_per_snp)
                .ok_or_else(|| "BED output byte count overflow".to_string())?,
        )?;
        copy_bim_lines_limited(bim_path, &mut wbim, rows_to_copy)?;
        written = written.saturating_add(rows_to_copy);
    }
    wbed.flush().map_err(|e| format!("{out_bed}: {e}"))?;
    wbim.flush().map_err(|e| format!("{out_bim}: {e}"))?;
    fs::copy(&src_fam, &out_fam).map_err(|e| format!("{src_fam} -> {out_fam}: {e}"))?;

    for result in results {
        if let Some(path) = result.temp_bed.as_ref() {
            let _ = fs::remove_file(path);
        }
        if let Some(path) = result.temp_bim.as_ref() {
            let _ = fs::remove_file(path);
        }
    }
    Ok(())
}

fn stream_prune_chrom_blocks_to_plink(
    src_prefix: &str,
    out_prefix: &str,
    scan: BimStreamScan,
    n_samples: usize,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    chrom_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    max_output_sites: Option<usize>,
) -> Result<(usize, usize), String> {
    if scan.blocks.is_empty() {
        return Err("no chromosome blocks found in BIM".to_string());
    }
    let bytes_per_snp = (n_samples + 3) / 4;
    let words = (n_samples + 63) / 64;
    let mut word_masks = vec![u64::MAX; words];
    if words > 0 {
        let rem = n_samples % 64;
        if rem != 0 {
            word_masks[words - 1] = (1u64 << rem) - 1u64;
        }
    }
    let write_temp = max_output_sites != Some(0usize);
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let blocks = scan.blocks;
    let chrom_progress = chrom_progress_callback.map(|cb| Arc::new(Mutex::new(cb)));
    let run = || -> Result<Vec<ChromPruneTempResult>, String> {
        blocks
            .par_iter()
            .enumerate()
            .map(|(block_idx, block)| {
                process_stream_chrom_block(
                    src_prefix,
                    out_prefix,
                    block,
                    block_idx,
                    n_samples,
                    bytes_per_snp,
                    window_bp,
                    window_variants,
                    step_variants,
                    r2_threshold,
                    &word_masks,
                    write_temp,
                    chrom_progress.as_ref(),
                    progress_every,
                )
            })
            .collect()
    };
    let mut results = if let Some(tp) = pool.as_ref() {
        tp.install(run)?
    } else {
        run()?
    };
    results.sort_by_key(|r| r.block_idx);
    let kept = results
        .iter()
        .fold(0usize, |acc, r| acc.saturating_add(r.kept));
    concat_stream_chrom_results_to_plink(
        src_prefix,
        out_prefix,
        &results,
        bytes_per_snp,
        max_output_sites,
    )?;
    if let Some(cb) = progress_callback.as_ref() {
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (scan.total_snps, scan.total_snps))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }
    Ok((kept, scan.total_snps))
}

fn stream_prune_plink_to_plink(
    src_prefix: &str,
    out_prefix: &str,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    chrom_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    max_output_sites: Option<usize>,
    pre_scan: Option<BimStreamScan>,
) -> Result<(usize, usize), String> {
    validate_ld_prune_args(window_bp, window_variants, step_variants, r2_threshold)?;
    let mode = resolve_ld_prune_mode();

    let scan = match pre_scan {
        Some(scan) => scan,
        None => scan_bim_stream_blocks(src_prefix)?,
    };
    let total_snps = scan.total_snps;
    let chrom_pos_sorted = scan.chrom_pos_sorted;
    if mode == LdPruneMode::Strict
        && scan.chrom_contiguous
        && (window_bp.is_none() || scan.blocks.iter().all(|b| b.pos_sorted))
    {
        let n_samples = crate::gfcore::read_fam(src_prefix)?.len();
        if n_samples == 0 || total_snps == 0 {
            return Err("empty PLINK input (no samples or no SNPs)".to_string());
        }
        return stream_prune_chrom_blocks_to_plink(
            src_prefix,
            out_prefix,
            scan,
            n_samples,
            window_bp,
            window_variants,
            step_variants,
            r2_threshold,
            threads,
            progress_callback,
            chrom_progress_callback,
            progress_every,
            max_output_sites,
        );
    }
    let chrom_last_index = scan.chrom_last_index;
    if window_bp.is_some() && !chrom_pos_sorted {
        return Err(
            "streaming packed prune requires BIM positions to be nondecreasing within each chromosome"
                .to_string(),
        );
    }

    let n_samples = crate::gfcore::read_fam(src_prefix)?.len();
    if n_samples == 0 || total_snps == 0 {
        return Err("empty PLINK input (no samples or no SNPs)".to_string());
    }

    let src_bed = format!("{src_prefix}.bed");
    let src_bim = format!("{src_prefix}.bim");
    let src_fam = format!("{src_prefix}.fam");
    let out_bed = format!("{out_prefix}.bed");
    let out_bim = format!("{out_prefix}.bim");
    let out_fam = format!("{out_prefix}.fam");
    if let Some(parent) = Path::new(&out_bed).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    let mut bread = BufReader::with_capacity(
        8 * 1024 * 1024,
        File::open(&src_bed).map_err(|e| format!("{src_bed}: {e}"))?,
    );
    let mut header = [0u8; 3];
    bread
        .read_exact(&mut header)
        .map_err(|e| format!("{src_bed}: {e}"))?;
    if header != [0x6C, 0x1B, 0x01] {
        return Err(format!(
            "{src_bed}: unsupported BED header (expect SNP-major 0x6C 0x1B 0x01)"
        ));
    }

    let bim_file = File::open(&src_bim).map_err(|e| format!("{src_bim}: {e}"))?;
    let mut bim_reader = BufReader::with_capacity(8 * 1024 * 1024, bim_file);
    let mut wbed = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&out_bed).map_err(|e| format!("{out_bed}: {e}"))?,
    );
    let mut wbim = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&out_bim).map_err(|e| format!("{out_bim}: {e}"))?,
    );
    wbed.write_all(&header)
        .map_err(|e| format!("{out_bed}: {e}"))?;

    let bytes_per_snp = (n_samples + 3) / 4;
    let byte_lut = packed_byte_lut();
    let pair_lut = packed_pair_lut();
    let code4_lut = &byte_lut.code4;
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let n_samples_f = n_samples as f64;
    let eps = 1e-12_f64;
    let step = step_variants.max(1);
    let notify_step = if progress_every == 0 {
        ((total_snps / 200).max(1)).max(step_variants.max(1))
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    let words = (n_samples + 63) / 64;
    let mut word_masks = vec![u64::MAX; words];
    if words > 0 {
        let rem = n_samples % 64;
        if rem != 0 {
            word_masks[words - 1] = (1u64 << rem) - 1u64;
        }
    }
    if let Some(cb) = progress_callback.as_ref() {
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (0usize, total_snps))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }

    let mut rows: Vec<Option<StreamingPruneBufferedRow>> = Vec::new();
    let mut next_write = 0usize;
    let mut kept = 0usize;
    let mut written = 0usize;
    let mut active_by_chrom: HashMap<String, VecDeque<usize>> = HashMap::new();
    let mut seen_by_chrom: HashMap<String, usize> = HashMap::new();
    let mut chrom_pos_by_seq: HashMap<String, Vec<i64>> = HashMap::new();
    let mut next_window_start_seq_by_chrom: HashMap<String, usize> = HashMap::new();
    let mut bp_end_scan_seq_by_chrom: HashMap<String, usize> = HashMap::new();
    let mut row_buf = vec![0u8; bytes_per_snp];
    let mut line_buf = String::new();
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;

    if mode == LdPruneMode::Fast {
        let mut next_anchor_seq_by_chrom: HashMap<String, usize> = HashMap::new();
        for row_idx in 0..total_snps {
            line_buf.clear();
            let n_line = bim_reader
                .read_line(&mut line_buf)
                .map_err(|e| format!("{src_bim}:{}: {}", row_idx + 1, e))?;
            if n_line == 0 {
                return Err(format!(
                    "{src_bim}: unexpected EOF at variant row {}",
                    row_idx + 1
                ));
            }
            let bim_line_trimmed = line_buf.trim_end_matches(&['\r', '\n'][..]);
            let (chrom, pos) = parse_bim_line_chrom_pos(bim_line_trimmed, row_idx + 1, &src_bim)?;

            let current_chrom_seq = {
                let entry = seen_by_chrom.entry(chrom.clone()).or_insert(0usize);
                let out = *entry;
                *entry = entry.saturating_add(1);
                out
            };
            let run_ld_compare = {
                let next_anchor = next_anchor_seq_by_chrom
                    .entry(chrom.clone())
                    .or_insert(0usize);
                if current_chrom_seq >= *next_anchor {
                    while *next_anchor <= current_chrom_seq {
                        *next_anchor = next_anchor.saturating_add(step);
                    }
                    true
                } else {
                    false
                }
            };

            if let Some(active) = active_by_chrom.get_mut(&chrom) {
                finalize_expired_streaming_rows(
                    active,
                    &mut rows,
                    pos,
                    current_chrom_seq,
                    window_bp,
                    window_variants,
                );
                active.retain(|rid| {
                    rows[*rid]
                        .as_ref()
                        .map(|row| !row.finalized)
                        .unwrap_or(false)
                });
            }

            bread
                .read_exact(&mut row_buf)
                .map_err(|e| format!("{src_bed}: row {}: {}", row_idx + 1, e))?;

            let current_stats = compute_packed_row_stats(&row_buf, n_samples, byte_lut);
            let (h_now, l_now, m_now, a_now, v_now) =
                build_row_bitplanes_u64_with_aux(&row_buf, n_samples);
            let current_bit_h = Some(h_now);
            let current_bit_l = Some(l_now);
            let current_bit_m = Some(m_now);
            let current_bit_a = Some(a_now);
            let current_bit_v = Some(v_now);

            let mut current_dropped = false;
            let mut drop_prior_ids: Vec<usize> = Vec::new();
            if run_ld_compare {
                if let Some(active) = active_by_chrom.get(&chrom) {
                    let classify_prior = |prior: &StreamingPruneBufferedRow| -> Option<u8> {
                        let r2 = match (
                            prior.bit_h.as_deref(),
                            prior.bit_l.as_deref(),
                            prior.bit_m.as_deref(),
                            prior.bit_a.as_deref(),
                            prior.bit_v.as_deref(),
                            current_bit_h.as_deref(),
                            current_bit_l.as_deref(),
                            current_bit_m.as_deref(),
                            current_bit_a.as_deref(),
                            current_bit_v.as_deref(),
                        ) {
                            (
                                Some(ph),
                                Some(pl),
                                Some(pm),
                                Some(pa),
                                Some(pv),
                                Some(ch),
                                Some(cl),
                                Some(cm),
                                Some(ca),
                                Some(cv),
                            ) => {
                                if !prior.stats.has_missing && !current_stats.has_missing {
                                    let dot_imp =
                                        dot_nomiss_row_bitplanes(ph, pl, ch, cl, &word_masks);
                                    let cov = dot_imp
                                        - n_samples_f * prior.stats.mean * current_stats.mean;
                                    let denom_corr = denom * prior.stats.std * current_stats.std;
                                    let corr = if denom_corr > 0.0_f64 {
                                        cov / denom_corr
                                    } else {
                                        0.0_f64
                                    };
                                    corr * corr
                                } else {
                                    if prior.stats.non_missing <= 1
                                        || current_stats.non_missing <= 1
                                    {
                                        f64::NAN
                                    } else {
                                        let _ = pm;
                                        let _ = cm;
                                        r2_pairwise_complete_bitplanes_cached_masks(
                                            ph, pa, pv, ch, ca, cv,
                                        )
                                        .unwrap_or(f64::NAN)
                                    }
                                }
                            }
                            (
                                Some(ph),
                                Some(pl),
                                Some(pm),
                                _,
                                _,
                                Some(ch),
                                Some(cl),
                                Some(cm),
                                _,
                                _,
                            ) => {
                                if !prior.stats.has_missing && !current_stats.has_missing {
                                    let dot_imp =
                                        dot_nomiss_row_bitplanes(ph, pl, ch, cl, &word_masks);
                                    let cov = dot_imp
                                        - n_samples_f * prior.stats.mean * current_stats.mean;
                                    let denom_corr = denom * prior.stats.std * current_stats.std;
                                    let corr = if denom_corr > 0.0_f64 {
                                        cov / denom_corr
                                    } else {
                                        0.0_f64
                                    };
                                    corr * corr
                                } else {
                                    r2_pairwise_complete_bitplanes(
                                        ph,
                                        pl,
                                        pm,
                                        ch,
                                        cl,
                                        cm,
                                        &word_masks,
                                    )
                                    .unwrap_or(f64::NAN)
                                }
                            }
                            _ => {
                                if !prior.stats.has_missing && !current_stats.has_missing {
                                    let dot_imp = dot_nomiss_pair_from_packed(
                                        &prior.packed,
                                        &row_buf,
                                        n_samples,
                                        pair_lut,
                                        code4_lut,
                                    );
                                    let cov = dot_imp
                                        - n_samples_f * prior.stats.mean * current_stats.mean;
                                    let denom_corr = denom * prior.stats.std * current_stats.std;
                                    let corr = if denom_corr > 0.0_f64 {
                                        cov / denom_corr
                                    } else {
                                        0.0_f64
                                    };
                                    corr * corr
                                } else {
                                    r2_pairwise_complete_from_packed(
                                        &prior.packed,
                                        &row_buf,
                                        n_samples,
                                        pair_lut,
                                        code4_lut,
                                    )
                                    .unwrap_or(f64::NAN)
                                }
                            }
                        };
                        let prune_r2_thresh = r2_threshold * (1.0_f64 + eps);
                        if !(r2.is_finite() && r2 > prune_r2_thresh) {
                            return None;
                        }
                        Some(classify_ld_pair_by_maf(
                            prior.stats.maf,
                            current_stats.maf,
                            eps,
                        ))
                    };

                    let active_ids: Vec<usize> = active
                        .iter()
                        .copied()
                        .filter(|rid| {
                            rows.get(*rid)
                                .and_then(|x| x.as_ref())
                                .map(|row| !row.finalized && !row.dropped)
                                .unwrap_or(false)
                        })
                        .collect();
                    let use_parallel_active = pool.is_some() && active_ids.len() >= 128usize;
                    if use_parallel_active {
                        let decisions: Vec<(usize, u8)> = if let Some(tp) = pool.as_ref() {
                            tp.install(|| {
                                active_ids
                                    .par_iter()
                                    .filter_map(|rid| {
                                        let prior = rows.get(*rid).and_then(|x| x.as_ref())?;
                                        let tag = classify_prior(prior)?;
                                        Some((*rid, tag))
                                    })
                                    .collect()
                            })
                        } else {
                            Vec::new()
                        };
                        current_dropped = decisions.iter().any(|(_, tag)| *tag == 2u8);
                        drop_prior_ids.extend(decisions.into_iter().filter_map(|(rid, tag)| {
                            if tag == 1u8 {
                                Some(rid)
                            } else {
                                None
                            }
                        }));
                    } else {
                        for rid in active_ids.into_iter() {
                            let Some(Some(prior)) = rows.get(rid) else {
                                continue;
                            };
                            if let Some(tag) = classify_prior(prior) {
                                if tag == 2u8 {
                                    current_dropped = true;
                                } else {
                                    drop_prior_ids.push(rid);
                                }
                            }
                        }
                    }
                }
            }

            if !drop_prior_ids.is_empty() {
                for rid in drop_prior_ids {
                    if let Some(Some(prior)) = rows.get_mut(rid) {
                        if !prior.finalized {
                            prior.mark_dropped();
                        }
                    }
                }
            }

            let keep_output_payload = should_write_output_site(max_output_sites, written);
            let stored_row = if current_dropped {
                StreamingPruneBufferedRow {
                    pos,
                    chrom_seq: current_chrom_seq,
                    first_unchecked_seq: current_chrom_seq.saturating_add(1),
                    packed: Vec::new(),
                    bim_line: String::new(),
                    stats: current_stats,
                    bit_h: None,
                    bit_l: None,
                    bit_m: None,
                    bit_a: None,
                    bit_v: None,
                    dropped: true,
                    finalized: true,
                }
            } else {
                StreamingPruneBufferedRow {
                    pos,
                    chrom_seq: current_chrom_seq,
                    first_unchecked_seq: current_chrom_seq.saturating_add(1),
                    packed: if keep_output_payload {
                        row_buf.to_vec()
                    } else {
                        Vec::new()
                    },
                    bim_line: if keep_output_payload {
                        bim_line_trimmed.to_string()
                    } else {
                        String::new()
                    },
                    stats: current_stats,
                    bit_h: current_bit_h,
                    bit_l: current_bit_l,
                    bit_m: current_bit_m,
                    bit_a: current_bit_a,
                    bit_v: current_bit_v,
                    dropped: false,
                    finalized: false,
                }
            };
            let row_id = rows.len();
            rows.push(Some(stored_row));

            let active = active_by_chrom.entry(chrom.clone()).or_default();
            active.retain(|rid| {
                rows[*rid]
                    .as_ref()
                    .map(|row| !row.finalized)
                    .unwrap_or(false)
            });
            if !current_dropped {
                active.push_back(row_id);
            }

            if chrom_last_index.get(&chrom).copied() == Some(row_idx) {
                if let Some(active_done) = active_by_chrom.get_mut(&chrom) {
                    finalize_all_streaming_rows(active_done, &mut rows);
                }
                active_by_chrom.remove(&chrom);
                next_anchor_seq_by_chrom.remove(&chrom);
            }

            drain_finalized_streaming_rows(
                &mut rows,
                &mut next_write,
                &mut wbed,
                &mut wbim,
                &mut kept,
                &mut written,
                max_output_sites,
                &out_bed,
                &out_bim,
            )?;
            let done = row_idx.saturating_add(1);
            if done >= last_notified.saturating_add(notify_step) || done == total_snps {
                last_notified = done;
                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (done, total_snps))?;
                        Ok(())
                    })
                    .map_err(|e| e.to_string())?;
                } else {
                    Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                }
            }
        }
    } else {
        for row_idx in 0..total_snps {
            line_buf.clear();
            let n_line = bim_reader
                .read_line(&mut line_buf)
                .map_err(|e| format!("{src_bim}:{}: {}", row_idx + 1, e))?;
            if n_line == 0 {
                return Err(format!(
                    "{src_bim}: unexpected EOF at variant row {}",
                    row_idx + 1
                ));
            }
            let bim_line_trimmed = line_buf.trim_end_matches(&['\r', '\n'][..]);
            let (chrom, pos) = parse_bim_line_chrom_pos(bim_line_trimmed, row_idx + 1, &src_bim)?;

            let current_chrom_seq = {
                let entry = seen_by_chrom.entry(chrom.clone()).or_insert(0usize);
                let out = *entry;
                *entry = entry.saturating_add(1);
                out
            };
            chrom_pos_by_seq.entry(chrom.clone()).or_default().push(pos);

            bread
                .read_exact(&mut row_buf)
                .map_err(|e| format!("{src_bed}: row {}: {}", row_idx + 1, e))?;

            let current_stats = compute_packed_row_stats(&row_buf, n_samples, byte_lut);
            let (h_now, l_now, m_now, a_now, v_now) =
                build_row_bitplanes_u64_with_aux(&row_buf, n_samples);
            let current_bit_h = Some(h_now);
            let current_bit_l = Some(l_now);
            let current_bit_m = Some(m_now);
            let current_bit_a = Some(a_now);
            let current_bit_v = Some(v_now);

            let keep_output_payload = should_write_output_site(max_output_sites, written);
            let stored_row = StreamingPruneBufferedRow {
                pos,
                chrom_seq: current_chrom_seq,
                first_unchecked_seq: current_chrom_seq.saturating_add(1),
                packed: if keep_output_payload {
                    row_buf.to_vec()
                } else {
                    Vec::new()
                },
                bim_line: if keep_output_payload {
                    bim_line_trimmed.to_string()
                } else {
                    String::new()
                },
                stats: current_stats,
                bit_h: current_bit_h,
                bit_l: current_bit_l,
                bit_m: current_bit_m,
                bit_a: current_bit_a,
                bit_v: current_bit_v,
                dropped: false,
                finalized: false,
            };
            let row_id = rows.len();
            rows.push(Some(stored_row));

            {
                let active = active_by_chrom.entry(chrom.clone()).or_default();
                active.push_back(row_id);
            }

            let chrom_done = chrom_last_index.get(&chrom).copied() == Some(row_idx);
            {
                let active = active_by_chrom.entry(chrom.clone()).or_default();
                active.retain(|rid| {
                    rows[*rid]
                        .as_ref()
                        .map(|row| !row.finalized)
                        .unwrap_or(false)
                });

                let chrom_positions = chrom_pos_by_seq.get(&chrom).ok_or_else(|| {
                    "streaming prune internal error: missing chromosome positions".to_string()
                })?;
                let seen_count = chrom_positions.len();
                let mut next_start = *next_window_start_seq_by_chrom
                    .get(&chrom)
                    .unwrap_or(&0usize);
                let mut bp_scan = *bp_end_scan_seq_by_chrom.get(&chrom).unwrap_or(&0usize);

                loop {
                    if next_start >= seen_count {
                        break;
                    }

                    let (window_end_seq, ready) = if let Some(bp) = window_bp {
                        let start_pos = chrom_positions[next_start];
                        let target = start_pos.saturating_add(bp);
                        if bp_scan < next_start.saturating_add(1) {
                            bp_scan = next_start.saturating_add(1);
                        }
                        while bp_scan < seen_count && chrom_positions[bp_scan] <= target {
                            bp_scan = bp_scan.saturating_add(1);
                        }
                        let ready_now = chrom_done || (bp_scan < seen_count);
                        let end_seq = if ready_now { bp_scan } else { seen_count };
                        (end_seq, ready_now)
                    } else {
                        let want_end = next_start.saturating_add(window_variants.unwrap_or(1));
                        let end_seq = want_end.min(seen_count);
                        let ready_now = chrom_done || (seen_count >= want_end);
                        (end_seq, ready_now)
                    };
                    if !ready {
                        break;
                    }
                    if window_end_seq > next_start.saturating_add(1) {
                        // Build window membership once, then keep the state resident
                        // through strict re-scan rounds by in-place compaction.
                        let mut window_ids: Vec<usize> = Vec::with_capacity(active.len());
                        for &rid in active.iter() {
                            let Some(Some(row)) = rows.get(rid) else {
                                continue;
                            };
                            if row.chrom_seq < next_start {
                                continue;
                            }
                            if row.chrom_seq >= window_end_seq {
                                break;
                            }
                            if !row.finalized && !row.dropped {
                                window_ids.push(rid);
                            }
                        }
                        loop {
                            let mut keep_n = 0usize;
                            for read_idx in 0..window_ids.len() {
                                let rid = window_ids[read_idx];
                                let keep = rows
                                    .get(rid)
                                    .and_then(|x| x.as_ref())
                                    .map(|row| !row.finalized && !row.dropped)
                                    .unwrap_or(false);
                                if keep {
                                    if keep_n != read_idx {
                                        window_ids[keep_n] = rid;
                                    }
                                    keep_n = keep_n.saturating_add(1);
                                }
                            }
                            window_ids.truncate(keep_n);
                            if window_ids.len() <= 1 {
                                break;
                            }

                            let mut at_least_one_prune = false;
                            for wi in 0..window_ids.len().saturating_sub(1) {
                                let rid_i = window_ids[wi];
                                let (i_alive, scan_min_seq) = rows
                                    .get(rid_i)
                                    .and_then(|x| x.as_ref())
                                    .map(|row| {
                                        (
                                            !row.finalized && !row.dropped,
                                            row.first_unchecked_seq
                                                .max(next_start.saturating_add(1)),
                                        )
                                    })
                                    .unwrap_or((false, window_end_seq));
                                if !i_alive {
                                    continue;
                                }
                                if scan_min_seq >= window_end_seq {
                                    if let Some(Some(row_i)) = rows.get_mut(rid_i) {
                                        if !row_i.finalized && !row_i.dropped {
                                            row_i.first_unchecked_seq =
                                                row_i.first_unchecked_seq.max(window_end_seq);
                                        }
                                    }
                                    continue;
                                }

                                let mut pruned_this_round = false;
                                let wj_start = lower_bound_window_seq(
                                    &rows,
                                    &window_ids,
                                    wi + 1,
                                    scan_min_seq,
                                );
                                let wj_end = lower_bound_window_seq(
                                    &rows,
                                    &window_ids,
                                    wj_start,
                                    window_end_seq,
                                );
                                let mut prune_hit: Option<(u8, usize, usize)> = None;
                                {
                                    let Some(Some(row_i)) = rows.get(rid_i) else {
                                        continue;
                                    };
                                    let (ih, il, ia, iv) = match (
                                        row_i.bit_h.as_deref(),
                                        row_i.bit_l.as_deref(),
                                        row_i.bit_a.as_deref(),
                                        row_i.bit_v.as_deref(),
                                    ) {
                                        (Some(h), Some(l), Some(a), Some(v)) => (h, l, a, v),
                                        _ => continue,
                                    };
                                    let i_has_missing = row_i.stats.has_missing;
                                    let i_non_missing = row_i.stats.non_missing;
                                    let i_mean = row_i.stats.mean;
                                    let i_std = row_i.stats.std;
                                    let i_maf = row_i.stats.maf;
                                    let i_mean_scaled = n_samples_f * i_mean;
                                    let i_corr_scale = denom * i_std;
                                    let prune_r2_thresh = r2_threshold * (1.0_f64 + eps);

                                    for wj in wj_start..wj_end {
                                        let rid_j = window_ids[wj];
                                        if let Some(tag) = classify_streaming_pair_by_maf(
                                            &rows,
                                            i_has_missing,
                                            i_non_missing,
                                            i_mean_scaled,
                                            i_corr_scale,
                                            i_maf,
                                            ih,
                                            il,
                                            ia,
                                            iv,
                                            rid_j,
                                            prune_r2_thresh,
                                            eps,
                                            &word_masks,
                                        ) {
                                            prune_hit = Some((tag, wj, rid_j));
                                            break;
                                        }
                                    }
                                }
                                if let Some((tag, wj, rid_j)) = prune_hit {
                                    at_least_one_prune = true;
                                    pruned_this_round = true;
                                    if tag == 1u8 {
                                        mark_streaming_row_dropped(&mut rows, rid_i);
                                    } else {
                                        mark_streaming_row_dropped(&mut rows, rid_j);
                                        let mut next_seq = window_end_seq;
                                        for wk in (wj + 1)..wj_end {
                                            let rid_k = window_ids[wk];
                                            let Some(Some(row_k)) = rows.get(rid_k) else {
                                                continue;
                                            };
                                            if row_k.finalized || row_k.dropped {
                                                continue;
                                            }
                                            next_seq = row_k.chrom_seq;
                                            break;
                                        }
                                        if let Some(Some(row_i)) = rows.get_mut(rid_i) {
                                            if !row_i.finalized && !row_i.dropped {
                                                row_i.first_unchecked_seq =
                                                    row_i.first_unchecked_seq.max(next_seq);
                                            }
                                        }
                                    }
                                }
                                if !pruned_this_round {
                                    if let Some(Some(row_i)) = rows.get_mut(rid_i) {
                                        if !row_i.finalized && !row_i.dropped {
                                            row_i.first_unchecked_seq =
                                                row_i.first_unchecked_seq.max(window_end_seq);
                                        }
                                    }
                                }
                            }

                            if !at_least_one_prune {
                                break;
                            }
                        }
                    }

                    next_start = next_start.saturating_add(step);
                    finalize_streaming_rows_before_seq(active, &mut rows, next_start);
                }

                next_window_start_seq_by_chrom.insert(chrom.clone(), next_start);
                bp_end_scan_seq_by_chrom.insert(chrom.clone(), bp_scan);
                active.retain(|rid| {
                    rows[*rid]
                        .as_ref()
                        .map(|row| !row.finalized)
                        .unwrap_or(false)
                });
                if chrom_done {
                    finalize_all_streaming_rows(active, &mut rows);
                }
            }

            if chrom_done {
                active_by_chrom.remove(&chrom);
                next_window_start_seq_by_chrom.remove(&chrom);
                bp_end_scan_seq_by_chrom.remove(&chrom);
                chrom_pos_by_seq.remove(&chrom);
            }

            drain_finalized_streaming_rows(
                &mut rows,
                &mut next_write,
                &mut wbed,
                &mut wbim,
                &mut kept,
                &mut written,
                max_output_sites,
                &out_bed,
                &out_bim,
            )?;
            let done = row_idx.saturating_add(1);
            if done >= last_notified.saturating_add(notify_step) || done == total_snps {
                last_notified = done;
                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (done, total_snps))?;
                        Ok(())
                    })
                    .map_err(|e| e.to_string())?;
                } else {
                    Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                }
            }
        }
    }

    let mut extra = [0u8; 1];
    let extra_n = bread
        .read(&mut extra)
        .map_err(|e| format!("{src_bed}: {e}"))?;
    if extra_n != 0 {
        return Err(format!(
            "{src_bed}: payload size mismatch, found extra bytes after {total_snps} SNP rows"
        ));
    }
    line_buf.clear();
    if bim_reader
        .read_line(&mut line_buf)
        .map_err(|e| format!("{src_bim}: {e}"))?
        != 0
    {
        return Err(format!(
            "{src_bim}: contains more rows than expected ({total_snps})"
        ));
    }

    for active in active_by_chrom.values_mut() {
        finalize_all_streaming_rows(active, &mut rows);
    }
    drain_finalized_streaming_rows(
        &mut rows,
        &mut next_write,
        &mut wbed,
        &mut wbim,
        &mut kept,
        &mut written,
        max_output_sites,
        &out_bed,
        &out_bim,
    )?;
    if next_write != rows.len() {
        return Err("streaming prune finished with unflushed rows".to_string());
    }

    wbed.flush().map_err(|e| format!("{out_bed}: {e}"))?;
    wbim.flush().map_err(|e| format!("{out_bim}: {e}"))?;
    fs::copy(&src_fam, &out_fam).map_err(|e| format!("{src_fam} -> {out_fam}: {e}"))?;
    Ok((kept, total_snps))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    chrom_codes,
    positions,
    window_bp=None,
    window_variants=None,
    step_variants=1,
    r2_threshold=0.2,
    threads=0
))]
pub fn bed_packed_ld_prune_maf_priority<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    chrom_codes: PyReadonlyArray1<'py, i32>,
    positions: PyReadonlyArray1<'py, i64>,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(r2_threshold.is_finite() && r2_threshold > 0.0 && r2_threshold <= 1.0) {
        return Err(PyRuntimeError::new_err(
            "r2_threshold must be finite and in (0, 1]",
        ));
    }
    if step_variants == 0 {
        return Err(PyRuntimeError::new_err("step_variants must be > 0"));
    }
    if window_bp.is_none() && window_variants.is_none() {
        return Err(PyRuntimeError::new_err(
            "provide one of window_bp or window_variants",
        ));
    }
    if let Some(bp) = window_bp {
        if bp <= 0 {
            return Err(PyRuntimeError::new_err("window_bp must be > 0"));
        }
    }
    if let Some(wv) = window_variants {
        if wv == 0 {
            return Err(PyRuntimeError::new_err("window_variants must be > 0"));
        }
    }

    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }
    let chrom_vec: Cow<[i32]> = match chrom_codes.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(chrom_codes.as_array().iter().copied().collect()),
    };
    let pos_vec: Cow<[i64]> = match positions.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(positions.as_array().iter().copied().collect()),
    };
    if chrom_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "chrom_codes length mismatch: got {}, expected {m}",
            chrom_vec.len()
        )));
    }
    if pos_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "positions length mismatch: got {}, expected {m}",
            pos_vec.len()
        )));
    }

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };

    let keep = py
        .detach(|| {
            bed_packed_ld_prune_keep(
                &packed_flat,
                m,
                bytes_per_snp,
                n_samples,
                &chrom_vec,
                &pos_vec,
                window_bp,
                window_variants,
                step_variants,
                r2_threshold,
                threads,
            )
        })
        .map_err(map_err_string_to_py)?;

    let out = PyArray1::<bool>::zeros(py, [m], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&keep);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    src_prefix,
    out_prefix,
    window_bp=None,
    window_variants=None,
    step_variants=1,
    r2_threshold=0.2,
    threads=0,
    progress_callback=None,
    chrom_progress_callback=None,
    progress_every=0,
    max_output_sites=None
))]
pub fn bed_prune_to_plink_rust(
    py: Python<'_>,
    src_prefix: String,
    out_prefix: String,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    chrom_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    max_output_sites: Option<usize>,
) -> PyResult<(usize, usize)> {
    let src = normalize_plink_prefix(&src_prefix);
    let out = normalize_plink_prefix(&out_prefix);
    if src.is_empty() || out.is_empty() {
        return Err(PyRuntimeError::new_err(
            "src_prefix/out_prefix must not be empty",
        ));
    }

    validate_ld_prune_args(window_bp, window_variants, step_variants, r2_threshold)
        .map_err(map_err_string_to_py)?;

    let bim_scan = scan_bim_stream_blocks(&src);
    let use_streaming = match &bim_scan {
        Ok(scan) => {
            if window_bp.is_some() {
                scan.chrom_pos_sorted
            } else {
                true
            }
        }
        Err(_) => false,
    };

    if use_streaming {
        return py
            .detach(|| {
                stream_prune_plink_to_plink(
                    &src,
                    &out,
                    window_bp,
                    window_variants,
                    step_variants,
                    r2_threshold,
                    threads,
                    progress_callback,
                    chrom_progress_callback,
                    progress_every,
                    max_output_sites,
                    bim_scan.ok(),
                )
            })
            .map_err(map_err_string_to_py);
    }

    let (bim_lines, chrom_codes, positions) =
        read_bim_lines_chrom_pos(&src).map_err(map_err_string_to_py)?;
    let n_snps = bim_lines.len();
    let n_samples = crate::gfcore::read_fam(&src)
        .map_err(map_err_string_to_py)?
        .len();
    if n_samples == 0 || n_snps == 0 {
        return Err(PyRuntimeError::new_err(
            "empty PLINK input (no samples or no SNPs)",
        ));
    }

    let bed_payload = read_bed_payload(&src, n_samples, n_snps).map_err(map_err_string_to_py)?;
    let bytes_per_snp = (n_samples + 3) / 4;

    let keep = py
        .detach(|| {
            bed_packed_ld_prune_keep(
                &bed_payload,
                n_snps,
                bytes_per_snp,
                n_samples,
                &chrom_codes,
                &positions,
                window_bp,
                window_variants,
                step_variants,
                r2_threshold,
                threads,
            )
        })
        .map_err(map_err_string_to_py)?;

    let (kept, total) = write_pruned_plink(
        &src,
        &out,
        &keep,
        &bed_payload,
        bytes_per_snp,
        &bim_lines,
        max_output_sites,
    )
    .map_err(map_err_string_to_py)?;
    if let Some(cb) = progress_callback.as_ref() {
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (total, total))?;
            Ok(())
        })?;
    }
    Ok((kept, total))
}

#[pyfunction]
#[pyo3(signature = (
    bfile,
    chrom_ranges,
    start_bp,
    end_bp,
    selected_chrom=None,
    selected_pos=None,
    threads=0
))]
pub fn bed_ldblock_r2_rust<'py>(
    py: Python<'py>,
    bfile: String,
    chrom_ranges: Vec<String>,
    start_bp: Vec<i64>,
    end_bp: Vec<i64>,
    selected_chrom: Option<Vec<String>>,
    selected_pos: Option<Vec<i64>>,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Vec<String>, Vec<i64>)> {
    let prefix = normalize_plink_prefix(&bfile);
    if prefix.is_empty() {
        return Err(PyRuntimeError::new_err("bfile must not be empty"));
    }

    let bimrange_map =
        build_bimrange_map(&chrom_ranges, &start_bp, &end_bp).map_err(map_err_string_to_py)?;
    let selected_sites: Option<HashSet<(String, i64)>> = match (selected_chrom, selected_pos) {
        (None, None) => None,
        (Some(ch), Some(ps)) => {
            if ch.len() != ps.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "selected_chrom/selected_pos length mismatch: {} vs {}",
                    ch.len(),
                    ps.len()
                )));
            }
            let mut set = HashSet::<(String, i64)>::with_capacity(ch.len());
            for i in 0..ch.len() {
                set.insert((normalize_chr_token(&ch[i]), ps[i]));
            }
            Some(set)
        }
        _ => {
            return Err(PyRuntimeError::new_err(
                "selected_chrom and selected_pos must be provided together",
            ))
        }
    };

    let (total_snps, selected_idx, selected_chr, selected_pos) =
        select_bim_indices_for_ld(&prefix, &bimrange_map, selected_sites.as_ref())
            .map_err(map_err_string_to_py)?;

    if selected_idx.is_empty() {
        let out = PyArray2::<f32>::zeros(py, [0, 0], false).into_bound();
        return Ok((out, Vec::new(), Vec::new()));
    }

    let n_samples = crate::gfcore::read_fam(&prefix)
        .map_err(map_err_string_to_py)?
        .len();
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err(
            "empty PLINK input (no samples in .fam)",
        ));
    }
    let bytes_per_snp = (n_samples + 3) / 4;

    let r2_vec = py
        .detach(|| -> Result<Vec<f32>, String> {
            let packed_rows =
                read_selected_bed_rows(&prefix, n_samples, total_snps, &selected_idx)?;
            ld_r2_matrix_from_packed_rows(
                &packed_rows,
                selected_idx.len(),
                bytes_per_snp,
                n_samples,
                threads,
            )
        })
        .map_err(map_err_string_to_py)?;

    let out = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((selected_idx.len(), selected_idx.len()), r2_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok((out, selected_chr, selected_pos))
}
