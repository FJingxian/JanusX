//! Error-tolerant hap-IBD pairwise IBD1/IBD2 detection (Mode 1: no `--ref`)
//! and genome-wide sparse IBD kinship CSR matrix builder for `src/hgwas`.

use super::pbwt::{scan_chromosome_seeds, PbwtSeed};
use super::phasing::SampleHaplotypeLayout;
use super::types::{BitHaplotypeMatrix, HaploidIbdConfig, IbdSegment};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimum fraction of markers in a candidate IBD segment that must be jointly
/// observed (non-missing in both haplotypes) to accept the segment.
const MIN_JOINT_CALL_RATE: f64 = 0.5;

/// Genome-wide sparse IBD kinship matrix in CSR format (`n_samples x n_samples`,
/// upper-triangular off-diagonal pairs with nonzero IBD sharing), designed for
/// direct consumption by `src/hgwas`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparseIbdKinshipCsr {
    pub n_samples: usize,
    pub total_genome_cm: f64,
    pub row_ptr: Vec<u64>,
    pub col_idx: Vec<u32>,
    /// Fraction of genome shared IBD1 (mutually exclusive with IBD2).
    pub ibd1_frac: Vec<f32>,
    /// Fraction of genome shared IBD2 (mutually exclusive with IBD1).
    pub ibd2_frac: Vec<f32>,
    /// Kinship coefficient `0.25 * ibd1_frac + 0.5 * ibd2_frac`.
    pub kinship: Vec<f32>,
}

/// Detect error-tolerant pairwise IBD1/IBD2 segments on a single chromosome.
pub fn detect_chromosome_ibd(
    matrix: &BitHaplotypeMatrix,
    layout: &SampleHaplotypeLayout,
    chrom_idx: u16,
    positions_bp: &[u64],
    cm_positions: &[f64],
    config: &HaploidIbdConfig,
) -> Vec<IbdSegment> {
    let n_markers = matrix.n_markers;
    if n_markers < config.min_markers.max(2) || matrix.n_haps < 2 {
        return Vec::new();
    }

    // Step 1: Whole-chromosome PBWT seed discovery.
    let raw_seeds = scan_chromosome_seeds(
        matrix,
        cm_positions,
        config.min_seed_cm,
        config.min_markers.min(15).max(5),
        None,
    );
    if raw_seeds.is_empty() {
        return Vec::new();
    }

    // Group seeds by haplotype pair `(hap_a, hap_b)` skipping intra-sample pairs.
    let mut grouped: HashMap<(u32, u32), Vec<PbwtSeed>> = HashMap::new();
    for seed in raw_seeds {
        let (s1, _) = layout.hap_to_sample[seed.hap_a as usize];
        let (s2, _) = layout.hap_to_sample[seed.hap_b as usize];
        if s1 == s2 {
            continue;
        }
        grouped
            .entry((seed.hap_a, seed.hap_b))
            .or_default()
            .push(seed);
    }

    let pairs: Vec<((u32, u32), Vec<PbwtSeed>)> = grouped.into_iter().collect();

    // Step 2: Parallel error-tolerant left/right extension & gap bridging per pair.
    let hap_segments: Vec<IbdSegment> = pairs
        .into_par_iter()
        .flat_map(|((hap_a, hap_b), mut seeds)| {
            seeds.sort_by_key(|s| (s.start_marker, s.end_marker));
            let (s1, h1) = layout.hap_to_sample[hap_a as usize];
            let (s2, h2) = layout.hap_to_sample[hap_b as usize];
            let (sample1, hap1, sample2, hap2) = if s1 < s2 {
                (s1, h1, s2, h2)
            } else {
                (s2, h2, s1, h1)
            };
            let both_inbred =
                layout.is_inbred[sample1 as usize] && layout.is_inbred[sample2 as usize];
            let base_ibd_type = if both_inbred { 2u8 } else { 1u8 };

            let mut extended: Vec<(usize, usize)> = Vec::with_capacity(seeds.len());
            for seed in seeds {
                let (ext_l, ext_r) = extend_seed_error_tolerant(
                    matrix,
                    cm_positions,
                    hap_a as usize,
                    hap_b as usize,
                    seed.start_marker as usize,
                    seed.end_marker as usize,
                    config.max_gap_cm,
                    config.max_mismatch_rate,
                );
                if let Some(last) = extended.last_mut() {
                    let gap_cm = if ext_l > last.1 {
                        cm_positions[ext_l] - cm_positions[last.1.saturating_sub(1)]
                    } else {
                        0.0
                    };
                    if ext_l <= last.1 || gap_cm <= config.max_gap_cm {
                        let merged_r = last.1.max(ext_r);
                        let merged_len = merged_r.saturating_sub(last.0);
                        let (comp, mis) = matrix.count_mismatches(
                            hap_a as usize,
                            hap_b as usize,
                            last.0,
                            merged_r,
                        );
                        if comp > 0
                            && (comp as f64 / merged_len.max(1) as f64) >= MIN_JOINT_CALL_RATE
                            && (mis as f64 / comp as f64) <= config.max_mismatch_rate * 2.0
                        {
                            last.1 = merged_r;
                            continue;
                        }
                    }
                }
                extended.push((ext_l, ext_r));
            }

            let mut out = Vec::new();
            for (l, r) in extended {
                if r <= l {
                    continue;
                }
                let len_markers = r - l;
                if len_markers < config.min_markers {
                    continue;
                }
                let start_cm = cm_positions[l];
                let end_cm = cm_positions[r - 1];
                if (end_cm - start_cm) < config.min_output_cm {
                    continue;
                }
                let (comp, mis) = matrix.count_mismatches(hap_a as usize, hap_b as usize, l, r);
                // Require sufficient jointly called markers and call rate; never
                // accept zero-evidence / all-missing regions as matches.
                if (comp as usize) < config.min_markers {
                    continue;
                }
                if (comp as f64 / len_markers as f64) < MIN_JOINT_CALL_RATE {
                    continue;
                }
                let mismatch_rate_f64 = mis as f64 / comp as f64;
                if mismatch_rate_f64 > config.max_mismatch_rate {
                    continue;
                }
                let mismatch_rate = mismatch_rate_f64 as f32;
                out.push(IbdSegment {
                    sample1,
                    hap1,
                    sample2,
                    hap2,
                    track2: None,
                    chrom_idx,
                    start_bp: positions_bp[l],
                    end_bp: positions_bp[r - 1],
                    start_cm: start_cm as f32,
                    end_cm: end_cm as f32,
                    ibd_type: base_ibd_type,
                    n_markers: comp,
                    mismatch_rate,
                });
            }
            out
        })
        .collect();

    // Step 3: Split overlapping diploid haplotype tracks into exact mutually
    // exclusive IBD1 and IBD2 intervals so no region is double-counted.
    resolve_diploid_ibd2_intervals(hap_segments, layout)
}

/// Extend a PBWT seed `[start, end)` left and right across single genotyping
/// errors / short missing gaps as long as the mismatch rate stays within budget
/// and no gap since the last observed match exceeds `max_gap_cm`.
#[allow(clippy::too_many_arguments)]
fn extend_seed_error_tolerant(
    matrix: &BitHaplotypeMatrix,
    cm_positions: &[f64],
    hap_a: usize,
    hap_b: usize,
    seed_start: usize,
    seed_end: usize,
    max_gap_cm: f64,
    max_mismatch_rate: f64,
) -> (usize, usize) {
    let n_markers = matrix.n_markers;
    let max_consecutive_mismatches = 2usize;

    // Extend left
    let mut best_left = seed_start;
    let mut cur = seed_start;
    let mut last_match_cm = cm_positions[seed_start];
    let mut consec_mis = 0usize;
    let mut ext_comp = 0usize;
    let mut ext_mis = 0usize;
    while cur > 0 {
        cur -= 1;
        if (last_match_cm - cm_positions[cur]) > max_gap_cm {
            break;
        }
        match (matrix.allele(cur, hap_a), matrix.allele(cur, hap_b)) {
            (Some(a), Some(b)) => {
                ext_comp += 1;
                if a == b {
                    consec_mis = 0;
                    last_match_cm = cm_positions[cur];
                    if ext_mis as f64 <= (ext_comp as f64 * max_mismatch_rate).max(2.0) {
                        best_left = cur;
                    }
                } else {
                    ext_mis += 1;
                    consec_mis += 1;
                    if consec_mis > max_consecutive_mismatches {
                        break;
                    }
                }
            }
            _ => {
                // Missing allele does not advance `best_left` or `last_match_cm`.
            }
        }
    }

    // Extend right
    let mut best_right = seed_end;
    let mut cur = seed_end;
    let mut last_match_cm = cm_positions[seed_end.saturating_sub(1)];
    consec_mis = 0;
    ext_comp = 0;
    ext_mis = 0;
    while cur < n_markers {
        if (cm_positions[cur] - last_match_cm) > max_gap_cm {
            break;
        }
        match (matrix.allele(cur, hap_a), matrix.allele(cur, hap_b)) {
            (Some(a), Some(b)) => {
                ext_comp += 1;
                if a == b {
                    consec_mis = 0;
                    last_match_cm = cm_positions[cur];
                    if ext_mis as f64 <= (ext_comp as f64 * max_mismatch_rate).max(2.0) {
                        best_right = cur + 1;
                    }
                } else {
                    ext_mis += 1;
                    consec_mis += 1;
                    if consec_mis > max_consecutive_mismatches {
                        break;
                    }
                }
            }
            _ => {}
        }
        cur += 1;
    }

    (best_left, best_right)
}

/// Partition overlapping diploid haplotype-pair segments for each `(sample1, sample2)`
/// into exact mutually exclusive elementary intervals: regions covered by two
/// complementary haplotype pairs `(0, a)` and `(1, 1 - a)` are emitted once as
/// `ibd_type = 2`, while flanking single-track regions remain `ibd_type = 1`.
fn resolve_diploid_ibd2_intervals(
    segments: Vec<IbdSegment>,
    layout: &SampleHaplotypeLayout,
) -> Vec<IbdSegment> {
    if segments.is_empty() {
        return segments;
    }
    let mut out = Vec::with_capacity(segments.len());
    let mut diploid_groups: HashMap<(u16, u32, u32), Vec<IbdSegment>> = HashMap::new();

    for seg in segments {
        if !layout.is_inbred[seg.sample1 as usize] && !layout.is_inbred[seg.sample2 as usize] {
            diploid_groups
                .entry((seg.chrom_idx, seg.sample1, seg.sample2))
                .or_default()
                .push(seg);
        } else {
            out.push(seg);
        }
    }

    for ((chrom_idx, sample1, sample2), group) in diploid_groups {
        if group.len() == 1 {
            out.push(group.into_iter().next().unwrap());
            continue;
        }
        // Collect unique sorted breakpoints `(bp, cm)`.
        let mut pts: Vec<(u64, f32)> = Vec::with_capacity(group.len() * 2);
        for seg in &group {
            pts.push((seg.start_bp, seg.start_cm));
            pts.push((seg.end_bp, seg.end_cm));
        }
        pts.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        pts.dedup_by_key(|p| p.0);

        let mut resolved: Vec<IbdSegment> = Vec::new();
        for w in pts.windows(2) {
            let (bp_l, cm_l) = w[0];
            let (bp_r, cm_r) = w[1];
            if bp_r <= bp_l {
                continue;
            }
            let mut cov_mask = 0u8; // bit (hap1 * 2 + hap2)
            let mut first_hap = (0u8, 0u8);
            let mut has_any = false;
            let mut track_markers = [0u32; 4];
            let mut track_mis = [0.0f32; 4];
            for seg in &group {
                if seg.start_bp <= bp_l && seg.end_bp >= bp_r {
                    let idx = ((seg.hap1 & 1) * 2 + (seg.hap2 & 1)) as usize;
                    cov_mask |= 1u8 << idx;
                    if !has_any {
                        first_hap = (seg.hap1, seg.hap2);
                        has_any = true;
                    }
                    let full_bp = seg.end_bp.saturating_sub(seg.start_bp).max(1) as f64;
                    let frac = (bp_r - bp_l) as f64 / full_bp;
                    let m_est = ((seg.n_markers as f64) * frac).round() as u32;
                    track_markers[idx] = track_markers[idx].max(m_est);
                    track_mis[idx] = track_mis[idx].max(seg.mismatch_rate);
                }
            }
            if !has_any {
                continue;
            }
            // Complementary diploid pairs:
            //   parallel: (0,0)&(1,1) = bits 0b1001 -> hap=(0,0), track2=Some((1,1))
            //   crossed : (0,1)&(1,0) = bits 0b0110 -> hap=(0,1), track2=Some((1,0))
            let (ibd_type, hap1, hap2, track2, markers_sum, mis_rate) = if (cov_mask & 0b1001)
                == 0b1001
            {
                (
                    2u8,
                    0u8,
                    0u8,
                    Some((1u8, 1u8)),
                    track_markers[0].min(track_markers[3]),
                    track_mis[0].max(track_mis[3]),
                )
            } else if (cov_mask & 0b0110) == 0b0110 {
                (
                    2u8,
                    0u8,
                    1u8,
                    Some((1u8, 0u8)),
                    track_markers[1].min(track_markers[2]),
                    track_mis[1].max(track_mis[2]),
                )
            } else {
                let idx = ((first_hap.0 & 1) * 2 + (first_hap.1 & 1)) as usize;
                (
                    1u8,
                    first_hap.0,
                    first_hap.1,
                    None,
                    track_markers[idx],
                    track_mis[idx],
                )
            };

            if let Some(prev) = resolved.last_mut() {
                if prev.end_bp == bp_l
                    && prev.ibd_type == ibd_type
                    && prev.hap1 == hap1
                    && prev.hap2 == hap2
                    && prev.track2 == track2
                {
                    prev.end_bp = bp_r;
                    prev.end_cm = cm_r;
                    prev.n_markers = prev.n_markers.saturating_add(markers_sum);
                    prev.mismatch_rate = prev.mismatch_rate.max(mis_rate);
                    continue;
                }
            }
            resolved.push(IbdSegment {
                sample1,
                hap1,
                sample2,
                hap2,
                track2,
                chrom_idx,
                start_bp: bp_l,
                end_bp: bp_r,
                start_cm: cm_l,
                end_cm: cm_r,
                ibd_type,
                n_markers: markers_sum,
                mismatch_rate: mis_rate,
            });
        }
        out.extend(resolved);
    }

    out.sort_by_key(|s| (s.chrom_idx, s.sample1, s.sample2, s.start_bp, s.hap1, s.hap2));
    out
}

/// Build the genome-wide sparse IBD kinship CSR matrix from all detected
/// pairwise segments across chromosomes.
///
/// Uses an exact per-`(chrom_idx, sample1, sample2)` sweep-line partition in cM
/// coordinates so overlapping haplotype tracks are decomposed into mutually
/// exclusive IBD1 and IBD2 lengths (`kinship = 0.25 * ibd1 + 0.5 * ibd2`,
/// matching the IBDkin definition).
pub fn build_sparse_ibd_kinship_csr(
    n_samples: usize,
    total_genome_cm: f64,
    segments: &[IbdSegment],
) -> SparseIbdKinshipCsr {
    if n_samples == 0 || total_genome_cm <= 0.0 {
        return SparseIbdKinshipCsr {
            n_samples,
            total_genome_cm,
            row_ptr: vec![0u64; n_samples + 1],
            col_idx: Vec::new(),
            ibd1_frac: Vec::new(),
            ibd2_frac: Vec::new(),
            kinship: Vec::new(),
        };
    }

    // Group segments by `(chrom_idx, s1 < s2)`.
    let mut by_chrom_pair: HashMap<(u16, u32, u32), Vec<&IbdSegment>> = HashMap::new();
    for seg in segments {
        if seg.end_cm <= seg.start_cm || seg.sample1 == seg.sample2 {
            continue;
        }
        let (s1, s2) = if seg.sample1 < seg.sample2 {
            (seg.sample1, seg.sample2)
        } else {
            (seg.sample2, seg.sample1)
        };
        by_chrom_pair
            .entry((seg.chrom_idx, s1, s2))
            .or_default()
            .push(seg);
    }

    // Sweep-line partition each `(chrom, s1, s2)` into mutually exclusive IBD1 & IBD2 cM.
    let mut pair_cm: HashMap<(u32, u32), (f64, f64)> = HashMap::new();
    for ((_, s1, s2), group) in by_chrom_pair {
        let mut coords: Vec<f64> = Vec::with_capacity(group.len() * 2);
        for seg in &group {
            coords.push(seg.start_cm as f64);
            coords.push(seg.end_cm as f64);
        }
        coords.sort_by(|a, b| a.total_cmp(b));
        coords.dedup_by(|a, b| (*a - *b).abs() <= 1e-9);

        let mut ibd1_cm = 0.0f64;
        let mut ibd2_cm = 0.0f64;
        for w in coords.windows(2) {
            let l = w[0];
            let r = w[1];
            let len = r - l;
            if len <= 1e-9 {
                continue;
            }
            let mid = 0.5 * (l + r);
            let mut explicit_ibd2 = false;
            let mut cov_mask = 0u8;
            let mut any_cov = false;
            for seg in &group {
                if (seg.start_cm as f64) <= mid && (seg.end_cm as f64) >= mid {
                    any_cov = true;
                    if seg.ibd_type >= 2 {
                        explicit_ibd2 = true;
                    } else {
                        let (h1, h2) = if seg.sample1 == s1 {
                            (seg.hap1 & 1, seg.hap2 & 1)
                        } else {
                            (seg.hap2 & 1, seg.hap1 & 1)
                        };
                        cov_mask |= 1u8 << (h1 * 2 + h2);
                    }
                }
            }
            if !any_cov {
                continue;
            }
            let complementary_ibd2 =
                (cov_mask & 0b1001) == 0b1001 || (cov_mask & 0b0110) == 0b0110;
            if explicit_ibd2 || complementary_ibd2 {
                ibd2_cm += len;
            } else {
                ibd1_cm += len;
            }
        }
        let entry = pair_cm.entry((s1, s2)).or_insert((0.0, 0.0));
        entry.0 += ibd1_cm;
        entry.1 += ibd2_cm;
    }

    let mut rows: Vec<Vec<(u32, f32, f32, f32)>> = vec![Vec::new(); n_samples];
    for ((s1, s2), (cm1, cm2)) in pair_cm {
        let f2 = (cm2 / total_genome_cm).clamp(0.0, 1.0);
        let f1 = (cm1 / total_genome_cm).clamp(0.0, (1.0 - f2).max(0.0));
        let k = 0.25 * f1 + 0.5 * f2;
        if k > 0.0 {
            rows[s1 as usize].push((s2, f1 as f32, f2 as f32, k as f32));
        }
    }

    let mut row_ptr = Vec::with_capacity(n_samples + 1);
    let mut col_idx = Vec::new();
    let mut ibd1_frac = Vec::new();
    let mut ibd2_frac = Vec::new();
    let mut kinship = Vec::new();
    row_ptr.push(0u64);
    for mut row in rows {
        row.sort_by_key(|&(col, _, _, _)| col);
        for (col, f1, f2, k) in row {
            col_idx.push(col);
            ibd1_frac.push(f1);
            ibd2_frac.push(f2);
            kinship.push(k);
        }
        row_ptr.push(col_idx.len() as u64);
    }

    SparseIbdKinshipCsr {
        n_samples,
        total_genome_cm,
        row_ptr,
        col_idx,
        ibd1_frac,
        ibd2_frac,
        kinship,
    }
}

#[cfg(test)]
mod tests {
    use super::{build_sparse_ibd_kinship_csr, detect_chromosome_ibd};
    use crate::haploid::phasing::SampleHaplotypeLayout;
    use crate::haploid::types::{BitHaplotypeMatrix, HaploidIbdConfig, IbdSegment};

    #[test]
    fn error_tolerant_ibd_bridges_single_genotyping_error_in_inbred_lines() {
        let n_markers = 300;
        let layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true, true]);
        let mut matrix = BitHaplotypeMatrix::new(layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect(); // 15 cM

        for m in 0..n_markers {
            let a0 = ((m * 13 + 5) % 2) as u8;
            let a1 = if (30..230).contains(&m) {
                if m == 130 { 1 - a0 } else { a0 }
            } else {
                1 - a0
            };
            let a2 = ((m * 17 + 1) % 3 == 0) as u8;
            matrix.set_allele(m, 0, Some(a0));
            matrix.set_allele(m, 1, Some(a1));
            matrix.set_allele(m, 2, Some(a2));
        }

        let cfg = HaploidIbdConfig {
            min_seed_cm: 1.5,
            min_output_cm: 3.0,
            max_gap_cm: 0.5,
            max_mismatch_rate: 0.02,
            min_markers: 20,
            ..HaploidIbdConfig::default()
        };
        let segs = detect_chromosome_ibd(&matrix, &layout, 0, &pos, &cm, &cfg);
        let pair01: Vec<_> = segs
            .iter()
            .filter(|s| s.sample1 == 0 && s.sample2 == 1)
            .collect();
        assert_eq!(pair01.len(), 1, "should bridge single error into 1 segment");
        assert!(pair01[0].end_cm - pair01[0].start_cm >= 9.0);
        assert_eq!(pair01[0].ibd_type, 2);

        let kinship = build_sparse_ibd_kinship_csr(3, 15.0, &segs);
        assert_eq!(kinship.col_idx.len(), 1);
        assert_eq!(kinship.col_idx[0], 1);
        assert!(kinship.kinship[0] > 0.25);
    }

    #[test]
    fn all_missing_sample_produces_zero_ibd_against_all_zero_sample() {
        let n_markers = 200;
        let layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let mut matrix = BitHaplotypeMatrix::new(layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            matrix.set_allele(m, 0, Some(0));
            matrix.set_allele(m, 1, None);
        }
        let cfg = HaploidIbdConfig {
            min_seed_cm: 1.0,
            min_output_cm: 2.0,
            min_markers: 20,
            ..HaploidIbdConfig::default()
        };
        let segs = detect_chromosome_ibd(&matrix, &layout, 0, &pos, &cm, &cfg);
        assert!(segs.is_empty());
        let kinship = build_sparse_ibd_kinship_csr(2, 10.0, &segs);
        assert!(kinship.col_idx.is_empty());
    }

    #[test]
    fn overlapping_diploid_ibd_splits_mutually_exclusive_ibd1_and_ibd2_kinship() {
        // 100 cM genome: [0, 20] on (hap0, hap0) and [10, 30] on (hap1, hap1).
        // Exact decomposition: [0, 10] IBD1 (10 cM), [10, 20] IBD2 (10 cM), [20, 30] IBD1 (10 cM).
        // Kinship = 0.25 * 0.20 + 0.5 * 0.10 = 0.10.
        let segs = vec![
            IbdSegment {
                sample1: 0,
                hap1: 0,
                sample2: 1,
                hap2: 0,
                track2: None,
                chrom_idx: 0,
                start_bp: 0,
                end_bp: 20_000_000,
                start_cm: 0.0,
                end_cm: 20.0,
                ibd_type: 1,
                n_markers: 200,
                mismatch_rate: 0.0,
            },
            IbdSegment {
                sample1: 0,
                hap1: 1,
                sample2: 1,
                hap2: 1,
                track2: None,
                chrom_idx: 0,
                start_bp: 10_000_000,
                end_bp: 30_000_000,
                start_cm: 10.0,
                end_cm: 30.0,
                ibd_type: 1,
                n_markers: 200,
                mismatch_rate: 0.0,
            },
        ];
        let kinship = build_sparse_ibd_kinship_csr(2, 100.0, &segs);
        assert_eq!(kinship.col_idx, vec![1]);
        assert!((kinship.ibd1_frac[0] - 0.20).abs() < 1e-5, "got {}", kinship.ibd1_frac[0]);
        assert!((kinship.ibd2_frac[0] - 0.10).abs() < 1e-5, "got {}", kinship.ibd2_frac[0]);
        assert!((kinship.kinship[0] - 0.10).abs() < 1e-5, "got {}", kinship.kinship[0]);
    }

    #[test]
    fn crossed_diploid_ibd2_records_exact_supporting_tracks_not_zero_zero() {
        let n_markers = 60;
        let layout = SampleHaplotypeLayout::from_inbred_flags(vec![false, false]);
        let mut matrix = BitHaplotypeMatrix::new(layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 50_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.1).collect();
        for m in 0..n_markers {
            let a = (m % 2) as u8;
            let b = 1 - a;
            matrix.set_allele(m, 0, Some(a));
            matrix.set_allele(m, 1, Some(b));
            matrix.set_allele(m, 2, Some(b));
            matrix.set_allele(m, 3, Some(a));
        }
        let cfg = HaploidIbdConfig {
            min_seed_cm: 1.0,
            min_output_cm: 2.0,
            min_markers: 20,
            ..HaploidIbdConfig::default()
        };
        let segs = detect_chromosome_ibd(&matrix, &layout, 0, &pos, &cm, &cfg);
        assert_eq!(segs.len(), 1);
        let seg = &segs[0];
        assert_eq!(seg.ibd_type, 2);
        assert_eq!((seg.hap1, seg.hap2), (0, 1));
        assert_eq!(seg.track2, Some((1, 0)));
        assert!(seg.mismatch_rate.abs() < 1e-6);
    }

    #[test]
    fn final_ibd_output_enforces_exact_max_mismatch_rate() {
        // 100 markers (5.0 cM) with 1 mismatch -> exact mismatch rate = 0.01.
        // With max_mismatch_rate = 0.005, final output must reject this segment.
        let n_markers = 100;
        let layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let mut matrix = BitHaplotypeMatrix::new(layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 50_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            let a = (m % 2) as u8;
            matrix.set_allele(m, 0, Some(a));
            matrix.set_allele(m, 1, Some(if m == 50 { 1 - a } else { a }));
        }
        let strict_cfg = HaploidIbdConfig {
            min_seed_cm: 1.0,
            min_output_cm: 2.0,
            max_gap_cm: 0.5,
            max_mismatch_rate: 0.005,
            min_markers: 20,
            ..HaploidIbdConfig::default()
        };
        let segs_strict = detect_chromosome_ibd(&matrix, &layout, 0, &pos, &cm, &strict_cfg);
        assert!(
            segs_strict.is_empty(),
            "segment with 0.01 mismatch rate must be rejected when max_mismatch_rate=0.005, got {segs_strict:?}"
        );
    }
}
