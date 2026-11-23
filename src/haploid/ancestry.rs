//! Multi-parent ancestry / mosaic painting (Mode 2: with `--ref`) using
//! whole-chromosome PBWT candidate narrowing + genetic-step Viterbi decoding,
//! plus Union-of-Breakpoints sparse ancestry dosage matrix for `src/hgwas`.

use super::pbwt::scan_chromosome_seeds;
use super::phasing::SampleHaplotypeLayout;
use super::types::{AncestrySegment, BitHaplotypeMatrix, HaploidIbdConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Exact population-level union-of-breakpoints sparse ancestry dosage matrix
/// for one chromosome, stored in bin-major (CSC-like) layout so `src/hgwas`
/// can scan genetic bins in O(1) seek without materializing a dense 10M-SNP matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChromosomeAncestryBinsSparse {
    pub chrom_idx: u16,
    pub n_target_samples: usize,
    pub n_parents: usize,
    pub bin_starts_bp: Vec<u64>,
    pub bin_ends_bp: Vec<u64>,
    pub bin_starts_cm: Vec<f32>,
    pub bin_ends_cm: Vec<f32>,
    /// Length `n_bins + 1`: entries for bin `b` lie in `bin_ptr[b]..bin_ptr[b + 1]`.
    pub bin_ptr: Vec<u64>,
    /// Target sample index for each nonzero `(bin, sample, parent)` entry.
    pub sample_idx: Vec<u32>,
    /// Reference parent sample index for each nonzero entry.
    pub parent_idx: Vec<u32>,
    /// Parent dosage in `{0.5, 1.0}` (1.0 for inbred/DH or homozygous diploid ancestry).
    pub dosage: Vec<f32>,
}

/// Paint multi-parent ancestry mosaic segments for all target haplotypes on one
/// chromosome against a reference parent panel.
///
/// `joint_matrix` lays out reference haplotypes in columns `0..ref_layout.total_haps`
/// and target haplotypes in columns `ref_layout.total_haps..joint_matrix.n_haps`.
pub fn paint_chromosome_ancestry(
    joint_matrix: &BitHaplotypeMatrix,
    ref_layout: &SampleHaplotypeLayout,
    target_layout: &SampleHaplotypeLayout,
    chrom_idx: u16,
    positions_bp: &[u64],
    cm_positions: &[f64],
    config: &HaploidIbdConfig,
) -> (Vec<AncestrySegment>, ChromosomeAncestryBinsSparse) {
    let n_markers = joint_matrix.n_markers;
    let n_ref_haps = ref_layout.total_haps;
    let n_target_haps = target_layout.total_haps;
    if n_markers == 0 || n_ref_haps == 0 || n_target_haps == 0 {
        return (Vec::new(), ChromosomeAncestryBinsSparse::default());
    }

    // Step 1: Partition the chromosome into genetic steps (~0.05 cM) so even
    // 10M-SNP chromosomes decode in ~2,000-4,000 lightweight Viterbi steps.
    let step_ranges = build_genetic_step_ranges(cm_positions, config.ancestry_step_cm);
    let n_steps = step_ranges.len();

    // Step 2: If reference panel is larger than `ancestry_top_donors`, run a
    // single whole-chromosome PBWT sweep (cross-group target->ref only) to
    // select the top candidate reference haplotypes per target haplotype.
    let per_target_candidates: Vec<Vec<usize>> = if n_ref_haps <= config.ancestry_top_donors {
        vec![(0..n_ref_haps).collect(); n_target_haps]
    } else {
        let seeds = scan_chromosome_seeds(
            joint_matrix,
            cm_positions,
            (config.min_seed_cm * 0.5).max(0.2),
            config.min_markers.min(10).max(4),
            Some(n_ref_haps),
        );
        let mut counts: Vec<HashMap<usize, u64>> = vec![HashMap::new(); n_target_haps];
        for seed in seeds {
            let (ref_h, tgt_h) = if (seed.hap_a as usize) < n_ref_haps {
                (seed.hap_a as usize, seed.hap_b as usize - n_ref_haps)
            } else {
                (seed.hap_b as usize, seed.hap_a as usize - n_ref_haps)
            };
            let span = seed.end_marker.saturating_sub(seed.start_marker) as u64;
            *counts[tgt_h].entry(ref_h).or_insert(0) += span;
        }
        counts
            .into_par_iter()
            .enumerate()
            .map(|(tgt_local_h, map)| {
                let mut ranked: Vec<(usize, u64)> = map.into_iter().collect();
                ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let mut cands: Vec<usize> = ranked
                    .into_iter()
                    .take(config.ancestry_top_donors)
                    .map(|(h, _)| h)
                    .collect();
                if cands.is_empty() {
                    // Order-independent evidence fallback when PBWT emitted zero seeds
                    // (e.g. short chromosomes/regions below `min_seed_cm` or high divergence):
                    // score all reference haplotypes by actual allele match evidence instead
                    // of taking the first `ancestry_top_donors` by file order.
                    let tgt_col = n_ref_haps + tgt_local_h;
                    let stride = (n_markers / 4096).max(1);
                    let mut scored: Vec<(usize, bool, f32, u32)> = (0..n_ref_haps)
                        .map(|ref_h| {
                            let (comp, mis) = if stride == 1 {
                                joint_matrix.count_mismatches(tgt_col, ref_h, 0, n_markers)
                            } else {
                                let mut c = 0u32;
                                let mut m_err = 0u32;
                                let mut m = 0usize;
                                while m < n_markers {
                                    if let (Some(a), Some(b)) =
                                        (joint_matrix.allele(m, tgt_col), joint_matrix.allele(m, ref_h))
                                    {
                                        c += 1;
                                        if a != b {
                                            m_err += 1;
                                        }
                                    }
                                    m += stride;
                                }
                                (c, m_err)
                            };
                            let matched = comp.saturating_sub(mis);
                            let score = if comp > 0 {
                                matched as f32 * 0.7 - mis as f32 * 4.5
                            } else {
                                f32::NEG_INFINITY
                            };
                            (ref_h, comp > 0, score, matched)
                        })
                        .collect();
                    scored.sort_by(|a, b| {
                        b.1.cmp(&a.1)
                            .then_with(|| b.2.total_cmp(&a.2))
                            .then_with(|| b.3.cmp(&a.3))
                            .then_with(|| a.0.cmp(&b.0))
                    });
                    cands.extend(
                        scored
                            .into_iter()
                            .take(config.ancestry_top_donors)
                            .map(|(h, _, _, _)| h),
                    );
                }
                cands
            })
            .collect()
    };

    // Step 3: Parallel step-level Viterbi mosaic decoding per target haplotype.
    // Log-likelihood ratio scores relative to uncalled random background (p=0.5, err≈0.005):
    //   match ≈ ln(0.995 / 0.5) ≈ 0.7, mismatch ≈ -ln(0.005 / 0.5) ≈ -4.5.
    let mismatch_penalty: f32 = 4.5;
    let match_reward: f32 = 0.7;
    let switch_penalty = config.ancestry_switch_penalty.max(0.5);

    let segments: Vec<AncestrySegment> = (0..n_target_haps)
        .into_par_iter()
        .flat_map(|tgt_local_h| {
            let tgt_col = n_ref_haps + tgt_local_h;
            let (target_sample, target_hap) = target_layout.hap_to_sample[tgt_local_h];
            let candidates = &per_target_candidates[tgt_local_h];
            let k = candidates.len();
            if k == 0 || n_steps == 0 {
                return Vec::new();
            }

            // Evidence-aware Viterbi DP with `k + 1` states:
            //   - states `0..k`: candidate reference haplotypes (feasible at step `t`
            //     ONLY when `step_comparable[t, c] > 0`; otherwise `-inf`).
            //   - state `k`: explicit `Uncalled` (no-evidence) state with emission `0.0`.
            // Leaving an observed parent pays `switch_penalty`; leaving a parent that
            // has zero evidence at step `t` (forced by missingness) or resuming from
            // `Uncalled` after a step where no parent was observable pays `0.0`.
            // Entering a parent from `Uncalled` after an observable step pays
            // `switch_penalty` so `Uncalled` cannot be used to dodge switch penalties.
            let uncalled_state = k;
            let n_states = k + 1;
            let mut prev_dp = vec![f32::NEG_INFINITY; n_states];
            let mut curr_dp = vec![f32::NEG_INFINITY; n_states];
            // Use u32 backpointers so large `--ancestry-top-donors` (e.g. >= 65536)
            // never wrap `uncalled_state` to 0.
            let mut backptr = vec![uncalled_state as u32; n_steps * n_states];
            let mut step_scores = vec![0.0f32; n_steps * k];
            let mut step_comparable = vec![0u32; n_steps * k];
            let mut prev_had_obs = false;

            for (step_idx, &(m_start, m_end)) in step_ranges.iter().enumerate() {
                let row_off = step_idx * k;
                let bp_off = step_idx * n_states;

                let mut curr_has_obs = false;
                for (c_idx, &ref_col) in candidates.iter().enumerate() {
                    let (comp, mis) =
                        joint_matrix.count_mismatches(tgt_col, ref_col, m_start, m_end);
                    step_comparable[row_off + c_idx] = comp;
                    if comp > 0 {
                        curr_has_obs = true;
                    }
                    let matched = comp.saturating_sub(mis);
                    step_scores[row_off + c_idx] =
                        matched as f32 * match_reward - mis as f32 * mismatch_penalty;
                }

                if step_idx == 0 {
                    for c_idx in 0..k {
                        if step_comparable[row_off + c_idx] > 0 {
                            curr_dp[c_idx] = step_scores[row_off + c_idx];
                            backptr[bp_off + c_idx] = c_idx as u32;
                        } else {
                            curr_dp[c_idx] = f32::NEG_INFINITY;
                            backptr[bp_off + c_idx] = uncalled_state as u32;
                        }
                    }
                    curr_dp[uncalled_state] = 0.0;
                    backptr[bp_off + uncalled_state] = uncalled_state as u32;
                } else {
                    let mut best_prev_obs = f32::NEG_INFINITY;
                    let mut best_prev_obs_idx = uncalled_state;
                    let mut best_prev_forced_parent = f32::NEG_INFINITY;
                    let mut best_prev_forced_parent_idx = uncalled_state;

                    for c_prev in 0..k {
                        let val = prev_dp[c_prev];
                        if !val.is_finite() {
                            continue;
                        }
                        if val > best_prev_obs {
                            best_prev_obs = val;
                            best_prev_obs_idx = c_prev;
                        }
                        if step_comparable[row_off + c_prev] == 0
                            && val > best_prev_forced_parent
                        {
                            best_prev_forced_parent = val;
                            best_prev_forced_parent_idx = c_prev;
                        }
                    }

                    // Best transition into a called parent `c_idx` (other than staying on `c_idx`):
                    let penalized_obs = best_prev_obs - switch_penalty;
                    let from_uncalled_to_parent = if prev_had_obs {
                        prev_dp[uncalled_state] - switch_penalty
                    } else {
                        prev_dp[uncalled_state]
                    };
                    let mut switch_to_parent_val = penalized_obs;
                    let mut switch_to_parent_idx = best_prev_obs_idx;
                    if best_prev_forced_parent >= switch_to_parent_val {
                        switch_to_parent_val = best_prev_forced_parent;
                        switch_to_parent_idx = best_prev_forced_parent_idx;
                    }
                    if from_uncalled_to_parent >= switch_to_parent_val {
                        switch_to_parent_val = from_uncalled_to_parent;
                        switch_to_parent_idx = uncalled_state;
                    }

                    for c_idx in 0..k {
                        if step_comparable[row_off + c_idx] == 0 {
                            curr_dp[c_idx] = f32::NEG_INFINITY;
                            backptr[bp_off + c_idx] = uncalled_state as u32;
                            continue;
                        }
                        let emit = step_scores[row_off + c_idx];
                        let stay = prev_dp[c_idx];
                        if stay >= switch_to_parent_val {
                            curr_dp[c_idx] = stay + emit;
                            backptr[bp_off + c_idx] = c_idx as u32;
                        } else {
                            curr_dp[c_idx] = switch_to_parent_val + emit;
                            backptr[bp_off + c_idx] = switch_to_parent_idx as u32;
                        }
                    }

                    // Best transition into `Uncalled`: staying in `Uncalled` is always free,
                    // forced exit from an unobservable parent is free, voluntary exit from an
                    // observable parent pays `switch_penalty`.
                    let mut to_uncalled_val = prev_dp[uncalled_state];
                    let mut to_uncalled_idx = uncalled_state;
                    if best_prev_forced_parent > to_uncalled_val {
                        to_uncalled_val = best_prev_forced_parent;
                        to_uncalled_idx = best_prev_forced_parent_idx;
                    }
                    if penalized_obs > to_uncalled_val {
                        to_uncalled_val = penalized_obs;
                        to_uncalled_idx = best_prev_obs_idx;
                    }
                    curr_dp[uncalled_state] = to_uncalled_val;
                    backptr[bp_off + uncalled_state] = to_uncalled_idx as u32;
                }
                prev_had_obs = curr_has_obs;
                std::mem::swap(&mut prev_dp, &mut curr_dp);
            }

            // Backtrack optimal path across `0..=k`.
            let mut best_end_idx = uncalled_state;
            let mut best_end_val = f32::NEG_INFINITY;
            for (s_idx, &val) in prev_dp.iter().enumerate() {
                if val > best_end_val {
                    best_end_val = val;
                    best_end_idx = s_idx;
                }
            }
            let mut path = vec![uncalled_state; n_steps];
            let mut cur_idx = best_end_idx;
            for step_idx in (0..n_steps).rev() {
                path[step_idx] = cur_idx;
                cur_idx = backptr[step_idx * n_states + cur_idx] as usize;
            }

            // Map `Uncalled` state (or zero-comparable safety check) to `None`.
            let called_path: Vec<Option<usize>> = (0..n_steps)
                .map(|step_idx| {
                    let st = path[step_idx];
                    if st < k && step_comparable[step_idx * k + st] > 0 {
                        Some(st)
                    } else {
                        None
                    }
                })
                .collect();

            // Collapse consecutive CALLED steps with the same `(parent_sample, parent_hap)`
            // track into contiguous segments so intra-parent haplotype switches split
            // cleanly and every segment's `parent_hap` & `n_markers` match its exact track.
            let mut out = Vec::new();
            let mut seg_start_step = 0usize;
            while seg_start_step < n_steps {
                let Some(c_idx) = called_path[seg_start_step] else {
                    seg_start_step += 1;
                    continue;
                };
                let ref_col = candidates[c_idx];
                let (parent_sample, parent_hap) = ref_layout.hap_to_sample[ref_col];
                let mut seg_end_step = seg_start_step + 1;
                while seg_end_step < n_steps {
                    let Some(next_c_idx) = called_path[seg_end_step] else {
                        break;
                    };
                    let next_ref = candidates[next_c_idx];
                    let (next_parent, next_parent_hap) = ref_layout.hap_to_sample[next_ref];
                    if next_parent != parent_sample || next_parent_hap != parent_hap {
                        break;
                    }
                    seg_end_step += 1;
                }

                let mut seg_comparable: u32 = 0;
                let mut cand_totals = vec![0.0f64; k];
                let mut cand_comp = vec![0u32; k];
                for s_idx in seg_start_step..seg_end_step {
                    let chosen = path[s_idx];
                    let row_off = s_idx * k;
                    seg_comparable =
                        seg_comparable.saturating_add(step_comparable[row_off + chosen]);
                    for ci in 0..k {
                        cand_totals[ci] += step_scores[row_off + ci] as f64;
                        cand_comp[ci] =
                            cand_comp[ci].saturating_add(step_comparable[row_off + ci]);
                    }
                }
                if seg_comparable == 0 {
                    seg_start_step = seg_end_step;
                    continue;
                }

                let best_seg_score = cand_totals[c_idx];
                let mut ambiguous = 0u16;
                let mut exp_sum = 0.0f64;
                for ci in 0..k {
                    if cand_comp[ci] == 0 {
                        continue;
                    }
                    let score = cand_totals[ci];
                    if (best_seg_score - score) <= 0.5 {
                        ambiguous = ambiguous.saturating_add(1);
                    }
                    let delta = (score - best_seg_score).max(-40.0);
                    exp_sum += delta.exp();
                }
                let confidence = if exp_sum > 0.0 {
                    (1.0 / exp_sum).clamp(0.0, 1.0) as f32
                } else {
                    1.0
                };

                let m_start = step_ranges[seg_start_step].0;
                let m_end = step_ranges[seg_end_step - 1].1;
                let start_bp = positions_bp[m_start];
                let start_cm = cm_positions[m_start] as f32;
                // Share the exact recombination boundary with the next segment ONLY
                // when immediately followed by another called segment; otherwise stop
                // at this segment's own last marker so uncalled gaps remain uncalled.
                let followed_by_called =
                    seg_end_step < n_steps && called_path[seg_end_step].is_some();
                let (end_bp, end_cm) = if followed_by_called && m_end < n_markers {
                    (
                        positions_bp[m_end].saturating_sub(1).max(start_bp),
                        cm_positions[m_end] as f32,
                    )
                } else {
                    (positions_bp[m_end - 1], cm_positions[m_end - 1] as f32)
                };

                out.push(AncestrySegment {
                    target_sample,
                    hap: target_hap,
                    chrom_idx,
                    start_bp,
                    end_bp,
                    start_cm,
                    end_cm,
                    parent_sample,
                    parent_hap,
                    confidence,
                    ambiguous_parents: ambiguous.max(1),
                    n_markers: seg_comparable,
                });
                seg_start_step = seg_end_step;
            }
            out
        })
        .collect();

    // Step 4: Build exact Union-of-Breakpoints sparse ancestry dosage matrix.
    let sparse_bins = build_union_breakpoint_sparse_matrix(
        chrom_idx,
        target_layout,
        ref_layout.n_samples,
        &segments,
    );
    (segments, sparse_bins)
}

fn build_genetic_step_ranges(cm_positions: &[f64], step_cm: f64) -> Vec<(usize, usize)> {
    let n = cm_positions.len();
    if n == 0 {
        return Vec::new();
    }
    let step = if step_cm.is_finite() && step_cm > 0.0 {
        step_cm
    } else {
        0.05
    };
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < n {
        let limit_cm = cm_positions[start] + step;
        let mut end = start + 1;
        while end < n && cm_positions[end] < limit_cm {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }
    ranges
}

/// Compress all ancestry segments on a chromosome into the exact population-wide
/// Union-of-Breakpoints sparse dosage matrix (bin-major CSC-like layout).
pub fn build_union_breakpoint_sparse_matrix(
    chrom_idx: u16,
    target_layout: &SampleHaplotypeLayout,
    n_parents: usize,
    segments: &[AncestrySegment],
) -> ChromosomeAncestryBinsSparse {
    if segments.is_empty() || target_layout.n_samples == 0 || n_parents == 0 {
        return ChromosomeAncestryBinsSparse {
            chrom_idx,
            n_target_samples: target_layout.n_samples,
            n_parents,
            ..ChromosomeAncestryBinsSparse::default()
        };
    }

    // Collect unique sorted breakpoint coordinates `(bp, cm)`.
    // Because adjacent segments share `left.end_bp + 1 == right.start_bp`,
    // recombination boundaries deduplicate to a single breakpoint with zero empty bins.
    let mut bp_to_cm: Vec<(u64, f32)> = Vec::with_capacity(segments.len() * 2);
    for seg in segments {
        bp_to_cm.push((seg.start_bp, seg.start_cm));
        bp_to_cm.push((seg.end_bp.saturating_add(1), seg.end_cm));
    }
    bp_to_cm.sort_by_key(|&(bp, _)| bp);
    bp_to_cm.dedup_by_key(|(bp, _)| *bp);
    if bp_to_cm.len() < 2 {
        return ChromosomeAncestryBinsSparse {
            chrom_idx,
            n_target_samples: target_layout.n_samples,
            n_parents,
            ..ChromosomeAncestryBinsSparse::default()
        };
    }

    let n_bins = bp_to_cm.len() - 1;
    let mut bin_starts_bp = Vec::with_capacity(n_bins);
    let mut bin_ends_bp = Vec::with_capacity(n_bins);
    let mut bin_starts_cm = Vec::with_capacity(n_bins);
    let mut bin_ends_cm = Vec::with_capacity(n_bins);
    for b in 0..n_bins {
        bin_starts_bp.push(bp_to_cm[b].0);
        bin_ends_bp.push(bp_to_cm[b + 1].0.saturating_sub(1));
        bin_starts_cm.push(bp_to_cm[b].1);
        bin_ends_cm.push(bp_to_cm[b + 1].1);
    }

    // Map each segment onto covered bins and sum haplotype dosages per `(bin, sample, parent)`.
    let mut bin_entries: Vec<HashMap<(u32, u32), f32>> = vec![HashMap::new(); n_bins];
    for seg in segments {
        let b_start = bp_to_cm
            .binary_search_by_key(&seg.start_bp, |&(bp, _)| bp)
            .unwrap_or(0);
        let b_end = bp_to_cm
            .binary_search_by_key(&seg.end_bp.saturating_add(1), |&(bp, _)| bp)
            .unwrap_or(n_bins);
        let weight = if target_layout.is_inbred[seg.target_sample as usize] {
            1.0f32
        } else {
            0.5f32
        };
        for b in b_start..b_end.min(n_bins) {
            *bin_entries[b]
                .entry((seg.target_sample, seg.parent_sample))
                .or_insert(0.0) += weight;
        }
    }

    let mut out_bin_starts_bp = Vec::with_capacity(n_bins);
    let mut out_bin_ends_bp = Vec::with_capacity(n_bins);
    let mut out_bin_starts_cm = Vec::with_capacity(n_bins);
    let mut out_bin_ends_cm = Vec::with_capacity(n_bins);
    let mut bin_ptr = Vec::with_capacity(n_bins + 1);
    let mut sample_idx = Vec::new();
    let mut parent_idx = Vec::new();
    let mut dosage = Vec::new();
    bin_ptr.push(0u64);
    for (b, map) in bin_entries.into_iter().enumerate() {
        if map.is_empty() {
            continue;
        }
        out_bin_starts_bp.push(bin_starts_bp[b]);
        out_bin_ends_bp.push(bin_ends_bp[b]);
        out_bin_starts_cm.push(bin_starts_cm[b]);
        out_bin_ends_cm.push(bin_ends_cm[b]);
        let mut sorted: Vec<((u32, u32), f32)> = map.into_iter().collect();
        sorted.sort_by_key(|&(k, _)| k);
        for ((s, p), d) in sorted {
            sample_idx.push(s);
            parent_idx.push(p);
            dosage.push(d);
        }
        bin_ptr.push(sample_idx.len() as u64);
    }

    ChromosomeAncestryBinsSparse {
        chrom_idx,
        n_target_samples: target_layout.n_samples,
        n_parents,
        bin_starts_bp: out_bin_starts_bp,
        bin_ends_bp: out_bin_ends_bp,
        bin_starts_cm: out_bin_starts_cm,
        bin_ends_cm: out_bin_ends_cm,
        bin_ptr,
        sample_idx,
        parent_idx,
        dosage,
    }
}

#[cfg(test)]
mod tests {
    use super::paint_chromosome_ancestry;
    use crate::haploid::phasing::SampleHaplotypeLayout;
    use crate::haploid::types::{BitHaplotypeMatrix, HaploidIbdConfig};

    #[test]
    fn multi_parent_ancestry_recovers_dh_recombination_breakpoint_and_sparse_bins() {
        // 4 inbred maize parents + 2 DH progeny:
        //   DH 0: parent 0 on [0, 150), parent 2 on [150, 300)
        //   DH 1: parent 1 on [0, 300)
        let n_markers = 300;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 4]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 2]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let p0 = ((m * 3 + 1) % 2) as u8;
            let p1 = ((m * 5 + 2) % 3 == 0) as u8;
            let p2 = ((m * 7 + 4) % 2) as u8;
            let p3 = ((m * 11 + 1) % 5 < 2) as u8;
            joint.set_allele(m, 0, Some(p0));
            joint.set_allele(m, 1, Some(p1));
            joint.set_allele(m, 2, Some(p2));
            joint.set_allele(m, 3, Some(p3));

            let dh0 = if m < 150 { p0 } else { p2 };
            let dh1 = p1;
            joint.set_allele(m, 4, Some(dh0));
            joint.set_allele(m, 5, Some(dh1));
        }

        let cfg = HaploidIbdConfig {
            ancestry_step_cm: 0.25,
            ancestry_switch_penalty: 8.0,
            ..HaploidIbdConfig::default()
        };
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);

        let dh0_segs: Vec<_> = segs.iter().filter(|s| s.target_sample == 0).collect();
        let dh1_segs: Vec<_> = segs.iter().filter(|s| s.target_sample == 1).collect();
        assert_eq!(dh0_segs.len(), 2, "DH 0 should have 2 ancestry blocks");
        assert_eq!(dh0_segs[0].parent_sample, 0);
        assert_eq!(dh0_segs[1].parent_sample, 2);
        assert_eq!(dh0_segs[0].end_bp + 1, dh0_segs[1].start_bp);
        assert_eq!(dh1_segs.len(), 1, "DH 1 should have 1 ancestry block");
        assert_eq!(dh1_segs[0].parent_sample, 1);

        // Union of breakpoints across DH0 (1 shared boundary) + DH1 (0 breakpoints) = 2 bins.
        assert_eq!(bins.bin_starts_bp.len(), 2);
        assert_eq!(bins.dosage.len(), 4); // 2 samples x 2 bins
        assert!(bins.dosage.iter().all(|&d| (d - 1.0).abs() < 1e-6));
    }

    #[test]
    fn all_missing_target_leaves_ancestry_unassigned_with_zero_dosage() {
        let n_markers = 150;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 3]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 1]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            joint.set_allele(m, 0, Some((m % 2) as u8));
            joint.set_allele(m, 1, Some(((m + 1) % 2) as u8));
            joint.set_allele(m, 2, Some(((m / 2) % 2) as u8));
            joint.set_allele(m, 3, None); // all-missing target
        }
        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert!(
            segs.is_empty(),
            "all-missing target must produce 0 ancestry segments"
        );
        assert!(
            bins.dosage.is_empty(),
            "all-missing target must produce 0 sparse dosages"
        );
    }

    #[test]
    fn zero_evidence_parent_cannot_beat_mismatched_parent_for_ancestry_dosage() {
        // Parent 0 is all-missing (0 comparable sites), Parent 1 mismatches target
        // at all 40 markers. Parent 0 must NEVER be emitted with dosage=1.0.
        let n_markers = 40;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            joint.set_allele(m, 0, None); // all-missing parent 0
            joint.set_allele(m, 1, Some(1)); // parent 1 mismatches target everywhere
            joint.set_allele(m, 2, Some(0)); // target
        }
        let cfg = HaploidIbdConfig {
            ancestry_step_cm: 0.25,
            ..HaploidIbdConfig::default()
        };
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert!(
            segs.is_empty(),
            "zero-evidence parent must not produce ancestry segments, got {segs:?}"
        );
        assert!(bins.dosage.is_empty());
    }

    #[test]
    fn internal_missing_gap_breaks_ancestry_segment_with_zero_gap_dosage() {
        // Target matches Parent 0 on [0..40) and [60..100), with 20 consecutive
        // all-missing markers on [40..60). Must break into 2 segments and leave
        // the missing gap uncalled.
        let n_markers = 100;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            joint.set_allele(m, 0, Some(p0));
            joint.set_allele(m, 1, Some(p1));
            if (40..60).contains(&m) {
                joint.set_allele(m, 2, None);
            } else {
                joint.set_allele(m, 2, Some(p0));
            }
        }
        let cfg = HaploidIbdConfig {
            ancestry_step_cm: 0.25,
            ..HaploidIbdConfig::default()
        };
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(segs.len(), 2, "20-marker missing gap must break segment into 2");
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 0);
        assert_eq!(segs[0].end_bp, pos[39]);
        assert_eq!(segs[1].start_bp, pos[60]);
        assert_eq!(segs[0].n_markers, 40);
        assert_eq!(segs[1].n_markers, 40);
        // Gap [pos[40]..pos[59]] must not appear in called sparse bins.
        assert_eq!(bins.bin_starts_bp.len(), 2);
        assert_eq!(bins.bin_ends_bp[0], pos[39]);
        assert_eq!(bins.bin_starts_bp[1], pos[60]);
    }

    #[test]
    fn missing_parent_cannot_suppress_observed_ancestry_in_second_half() {
        // [0..20): Parent 0 matches 20/20, Parent 1 is all-missing.
        // [20..40): Parent 0 is all-missing, Parent 1 matches 20/20.
        // Default switch_penalty (12.0) exceeds 20 * 0.25 = 5.0, yet BOTH halves
        // must be recovered because a missing parent cannot hold the Viterbi path.
        let n_markers = 40;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            let a = (m % 2) as u8;
            if m < 20 {
                joint.set_allele(m, 0, Some(a));
                joint.set_allele(m, 1, None);
            } else {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, Some(a));
            }
            joint.set_allele(m, 2, Some(a));
        }
        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "both observed halves must be recovered, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].n_markers, 20);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[1].n_markers, 20);
        assert_eq!(bins.bin_starts_bp.len(), 2);
    }

    #[test]
    fn diploid_parent_haplotype_switch_splits_ancestry_segments_by_track() {
        // Single diploid parent (hap0 & hap1 complementary); target matches
        // hap0 on [0..20) and hap1 on [20..40). Must emit 2 segments recording
        // parent_hap=0 (20 markers) and parent_hap=1 (20 markers).
        let n_markers = 40;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![false]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            let h0 = (m % 2) as u8;
            let h1 = 1 - h0;
            joint.set_allele(m, 0, Some(h0));
            joint.set_allele(m, 1, Some(h1));
            joint.set_allele(m, 2, Some(if m < 20 { h0 } else { h1 }));
        }
        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(segs.len(), 2, "intra-parent track switch must split segment");
        assert_eq!(
            (segs[0].parent_sample, segs[0].parent_hap, segs[0].n_markers),
            (0, 0, 20)
        );
        assert_eq!(
            (segs[1].parent_sample, segs[1].parent_hap, segs[1].n_markers),
            (0, 1, 20)
        );
        assert_eq!(segs[0].end_bp + 1, segs[1].start_bp);
        assert_eq!(bins.bin_starts_bp.len(), 2);
        assert!(bins.dosage.iter().all(|&d| (d - 1.0).abs() < 1e-6));
    }

    #[test]
    fn zero_seed_short_region_with_more_than_top_donors_finds_last_parent() {
        // 65 inbred reference parents (> default ancestry_top_donors=64), 40 markers
        // spanning 0.39 cM (< min_seed_cm * 0.5 = 0.5 cM -> 0 PBWT seeds).
        // Only parent #64 (the 65th parent) matches the target; candidate selection
        // must be order-independent and recover parent #64 across all 40 markers.
        let n_parents = 65usize;
        let n_markers = 40usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.01).collect(); // 0.39 cM total

        for m in 0..n_markers {
            let target_allele = (m % 2) as u8;
            for p in 0..(n_parents - 1) {
                joint.set_allele(m, p, Some(1 - target_allele));
            }
            joint.set_allele(m, n_parents - 1, Some(target_allele));
            joint.set_allele(m, n_parents, Some(target_allele));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "65th parent must be recovered despite 0 PBWT seeds, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, (n_parents - 1) as u32);
        assert_eq!(segs[0].n_markers, n_markers as u32);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }
}
