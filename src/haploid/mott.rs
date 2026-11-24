//! Author-compatible CUBIC Mott HMM multi-parent ancestry inference.
//!
//! Mirrors the two-pass forward `P` / backward `Q` recurrence and `Z = (P + Q) / 2`
//! posterior combination from Liu et al. (Genome Biology 2020, `HMM-IBD`), while
//! outputting standard `AncestrySegment` and `ChromosomeAncestryBinsSparse`
//! payloads consumable by `jx postibd`.

use super::ancestry::{
    build_union_breakpoint_sparse_matrix, build_union_breakpoint_sparse_matrix_cancellable,
    ChromosomeAncestryBinsSparse,
};
use super::phasing::SampleHaplotypeLayout;
use super::types::{AncestrySegment, BitHaplotypeMatrix, HaploidIbdConfig};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MottFounderCall {
    Homozygous(u8),
    Ambiguous,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MottCoreConfig {
    pub generations: f64,
    pub parent_match_prob: f64,
    pub map_scale_cm: f64,
    pub unknown_fold: f64,
}

impl MottCoreConfig {
    pub(crate) fn from_ibd_config(config: &HaploidIbdConfig) -> Result<Self, String> {
        let cfg = Self {
            generations: config.mott_generations,
            parent_match_prob: config.mott_parent_match_prob,
            map_scale_cm: config.mott_map_scale_cm,
            unknown_fold: config.mott_unknown_fold,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if !self.generations.is_finite() || self.generations <= 0.0 {
            return Err(format!(
                "mott_generations must be finite and > 0, got {}",
                self.generations
            ));
        }
        if !self.parent_match_prob.is_finite()
            || self.parent_match_prob <= 0.0
            || self.parent_match_prob >= 1.0
        {
            return Err(format!(
                "mott_parent_match_prob must be in (0, 1), got {}",
                self.parent_match_prob
            ));
        }
        if !self.map_scale_cm.is_finite() || self.map_scale_cm <= 0.0 {
            return Err(format!(
                "mott_map_scale_cm must be finite and > 0, got {}",
                self.map_scale_cm
            ));
        }
        if !self.unknown_fold.is_finite() || self.unknown_fold <= 0.0 {
            return Err(format!(
                "mott_unknown_fold must be finite and > 0, got {}",
                self.unknown_fold
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MottPosteriorMatrix {
    pub n_states: usize,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct MottChromosomeOutput {
    pub segments: Vec<AncestrySegment>,
    pub bins: ChromosomeAncestryBinsSparse,
    pub posterior: Option<MottPosteriorMatrix>,
    pub selected_markers: usize,
}

#[inline]
fn is_cancelled(cancelled: Option<&AtomicBool>) -> bool {
    cancelled
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

fn validate_inbred_layout(
    layout: &SampleHaplotypeLayout,
    label: &str,
    n_haps: usize,
) -> Result<(), String> {
    if layout.n_samples == 0
        || layout.is_inbred.len() != layout.n_samples
        || layout.hap_offset.len() != layout.n_samples
        || layout.hap_count.len() != layout.n_samples
        || layout.hap_to_sample.len() != layout.total_haps
    {
        return Err(format!(
            "{label} layout has inconsistent sample/haplotype vectors"
        ));
    }
    if layout.total_haps == 0 || layout.total_haps > n_haps {
        return Err(format!(
            "{label} layout total_haps ({}) exceeds matrix haplotypes ({n_haps})",
            layout.total_haps
        ));
    }
    for sample in 0..layout.n_samples {
        if !layout.is_inbred[sample] || layout.hap_count[sample] != 1 {
            return Err(format!(
                "{label} layout sample {sample} is not a single inbred haplotype"
            ));
        }
        if layout.hap_offset[sample] >= layout.total_haps {
            return Err(format!(
                "{label} layout sample {sample} has out-of-range haplotype offset {}",
                layout.hap_offset[sample]
            ));
        }
    }
    Ok(())
}

/// Build the author `pmatrix.r` emission vector over founders for a single marker.
///
/// - When `target_call` is `None` (missing or heterozygous), returns the uniform
///   `1 / K` vector.
/// - When all founders are homozygous (`0` or `1`), matching founders evenly
///   share `parent_match_prob` and nonmatching founders evenly share
///   `1 - parent_match_prob`.
/// - When one or more founders are ambiguous (heterozygous or missing), each
///   ambiguous founder receives half-credit (`0.5`) to both alleles, matching
///   `pmatrix.r` lines 47–55.
pub(crate) fn founder_emission(
    founder_calls: &[MottFounderCall],
    target_call: Option<u8>,
    parent_match_prob: f64,
) -> Vec<f64> {
    let k = founder_calls.len();
    if k == 0 {
        return Vec::new();
    }
    let author_uniform = (1000.0 / k as f64).round() / 1000.0;
    let Some(target_allele) = target_call else {
        return vec![author_uniform; k];
    };
    let mut n_match = 0usize;
    let mut n_other = 0usize;
    let mut n_amb = 0usize;
    for &c in founder_calls {
        match c {
            MottFounderCall::Homozygous(a) if a == target_allele => n_match += 1,
            MottFounderCall::Homozygous(_) => n_other += 1,
            MottFounderCall::Ambiguous => n_amb += 1,
        }
    }

    if n_amb == 0 {
        if n_match == 0 {
            let other_val = (1.0 - parent_match_prob) / n_other as f64;
            return vec![other_val; k];
        }
        if n_other == 0 {
            let match_val = parent_match_prob / n_match as f64;
            return vec![match_val; k];
        }
        let match_val = parent_match_prob / n_match as f64;
        let other_val = (1.0 - parent_match_prob) / n_other as f64;
        return founder_calls
            .iter()
            .map(|&c| match c {
                MottFounderCall::Homozygous(a) if a == target_allele => match_val,
                _ => other_val,
            })
            .collect();
    }

    let sum_w = n_match as f64 + 0.5 * n_amb as f64;
    if sum_w <= 0.0 {
        return vec![author_uniform; k];
    }
    let scale = parent_match_prob / sum_w;
    let other_val = if n_other > 0 {
        (1.0 - parent_match_prob) / n_other as f64
    } else {
        0.0
    };
    founder_calls
        .iter()
        .map(|&c| match c {
            MottFounderCall::Homozygous(a) if a == target_allele => scale,
            MottFounderCall::Ambiguous => 0.5 * scale,
            MottFounderCall::Homozygous(_) => other_val,
        })
        .collect()
}

/// Compute `lambda` for an adjacent marker interval in cM, matching `omatrix.r`.
#[inline]
pub(crate) fn interval_lambda(interval_cm: f64, cfg: &MottCoreConfig) -> f64 {
    let rho = if interval_cm.is_finite() && interval_cm > 0.0 {
        interval_cm
    } else {
        0.0
    };
    if rho > 1e-6 {
        cfg.generations * rho / cfg.map_scale_cm
    } else {
        cfg.generations * 2e-8
    }
}

#[inline]
fn transition_diag_off(interval_cm: f64, k: usize, cfg: &MottCoreConfig) -> (f64, f64) {
    let lambda = interval_lambda(interval_cm, cfg);
    let stay = (-lambda).exp();
    let off = (1.0 - stay) / k as f64;
    let diag = stay + off;
    (diag, off)
}

/// Author `omatrix.r` single-step recurrence in `O(K)` time:
/// `f[j, k] = r[j, k] * emission[j] / sum_m(r[m, k] * emission[m])`,
/// `next[j] = sum_k previous[k] * f[j, k]`.
#[cfg(test)]
pub(crate) fn author_forward_step(
    previous: &[f64],
    emission: &[f64],
    diagonal: f64,
    off_diagonal: f64,
) -> Vec<f64> {
    let k = previous.len();
    let mut out = vec![0.0f64; k];
    if k == 0 || emission.len() != k {
        return out;
    }
    author_step_into(previous, emission, diagonal, off_diagonal, &mut out);
    out
}

#[inline]
fn author_step_into(
    previous: &[f64],
    emission: &[f64],
    diagonal: f64,
    off_diagonal: f64,
    out: &mut [f64],
) -> bool {
    let k = previous.len();
    let emit_sum: f64 = emission.iter().sum();
    let diff = diagonal - off_diagonal;
    let mut ratio_sum = 0.0f64;
    for j in 0..k {
        let a_j = diagonal * emission[j] + off_diagonal * (emit_sum - emission[j]);
        if !a_j.is_finite() || a_j <= 0.0 {
            out.fill(0.0);
            return false;
        }
        let r = previous[j] / a_j;
        out[j] = r;
        ratio_sum += r;
    }
    for j in 0..k {
        let r_j = out[j];
        out[j] = emission[j] * (diff * r_j + off_diagonal * ratio_sum);
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn posterior_one_sample_precomputed(
    n_markers: usize,
    k: usize,
    emit0: &[f64],
    emit1: &[f64],
    uniform: &[f64],
    target_calls: &[Option<u8>],
    cm_positions: &[f64],
    cfg: &MottCoreConfig,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<f64>, String> {
    if is_cancelled(cancelled) {
        return Err("Interrupted".to_owned());
    }
    let emission_at = |m: usize| -> &[f64] {
        match target_calls[m] {
            Some(0) => &emit0[m * k..(m + 1) * k],
            Some(1) => &emit1[m * k..(m + 1) * k],
            _ => uniform,
        }
    };

    // Forward pass P (stored row-major in `posterior`, then averaged in-place with Q).
    let mut posterior = vec![0.0f64; n_markers * k];
    posterior[0..k].copy_from_slice(emission_at(0));
    for i in 0..n_markers.saturating_sub(1) {
        if (i & 1023) == 0 && is_cancelled(cancelled) {
            return Err("Interrupted".to_owned());
        }
        let (diag, off) = transition_diag_off(cm_positions[i + 1] - cm_positions[i], k, cfg);
        let (left, right) = posterior.split_at_mut((i + 1) * k);
        let prev = &left[i * k..(i + 1) * k];
        let next = &mut right[0..k];
        if !author_step_into(prev, emission_at(i + 1), diag, off, next) {
            return Err(format!(
                "invalid non-positive emission denominator in Mott forward pass at marker {}",
                i + 1
            ));
        }
    }

    // Backward pass Q + in-place combination Z = (P + Q) / 2.
    let mut q_cur = vec![0.0f64; k];
    let mut q_next = vec![0.0f64; k];
    q_cur.copy_from_slice(emission_at(n_markers - 1));
    let last_off = (n_markers - 1) * k;
    for j in 0..k {
        posterior[last_off + j] = 0.5 * (posterior[last_off + j] + q_cur[j]);
    }
    for i in (0..n_markers.saturating_sub(1)).rev() {
        if (i & 1023) == 0 && is_cancelled(cancelled) {
            return Err("Interrupted".to_owned());
        }
        let (diag, off) = transition_diag_off(cm_positions[i + 1] - cm_positions[i], k, cfg);
        if !author_step_into(&q_cur, emission_at(i), diag, off, &mut q_next) {
            return Err(format!(
                "invalid non-positive emission denominator in Mott backward pass at marker {i}"
            ));
        }
        std::mem::swap(&mut q_cur, &mut q_next);
        let row_off = i * k;
        for j in 0..k {
            posterior[row_off + j] = 0.5 * (posterior[row_off + j] + q_cur[j]);
        }
    }

    Ok(posterior)
}

/// Apply the strict author unknown threshold `max_f P(s, f) > unknown_fold / K`.
pub(crate) fn hard_calls(
    posterior: &[f64],
    n_markers: usize,
    n_states: usize,
    unknown_fold: f64,
) -> Result<Vec<Option<usize>>, String> {
    if n_states == 0 {
        return Err("n_states must be > 0".to_owned());
    }
    if posterior.len() != n_markers.saturating_mul(n_states) {
        return Err(format!(
            "posterior length ({}) does not match n_markers ({n_markers}) * n_states ({n_states})",
            posterior.len()
        ));
    }
    if !unknown_fold.is_finite() || unknown_fold <= 0.0 {
        return Err(format!(
            "unknown_fold must be finite and > 0, got {unknown_fold}"
        ));
    }
    let threshold = unknown_fold / n_states as f64;
    let mut calls = Vec::with_capacity(n_markers);
    for m in 0..n_markers {
        let row = &posterior[m * n_states..(m + 1) * n_states];
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (j, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!(
                    "non-finite posterior value at marker {m}, state {j}: {val}"
                ));
            }
            if val > best_val {
                best_val = val;
                best_idx = j;
            }
        }
        if best_val > threshold {
            calls.push(Some(best_idx));
        } else {
            calls.push(None);
        }
    }
    Ok(calls)
}

/// Deterministically select up to `cap` informative marker indices with even
/// rank spacing, always preserving the first and last informative markers.
pub(crate) fn select_even_marker_indices(informative: &[bool], cap: usize) -> Vec<usize> {
    let eligible: Vec<usize> = informative
        .iter()
        .enumerate()
        .filter_map(|(i, &ok)| ok.then_some(i))
        .collect();
    let n = eligible.len();
    if n == 0 || cap == 0 || cap >= n {
        return eligible;
    }
    if cap == 1 {
        return vec![eligible[0]];
    }
    (0..cap)
        .map(|j| {
            let pos = (j * (n - 1)) / (cap - 1);
            eligible[pos]
        })
        .collect()
}

fn compress_calls_to_segments(
    target_sample: u32,
    chrom_idx: u16,
    calls: &[Option<usize>],
    posterior: &[f64],
    n_states: usize,
    positions_bp: &[u64],
    cm_positions: &[f64],
) -> Vec<AncestrySegment> {
    let n_markers = calls.len();
    let mut segments = Vec::new();
    let mut i = 0usize;
    while i < n_markers {
        let Some(parent) = calls[i] else {
            i += 1;
            continue;
        };
        let run_start = i;
        let mut conf_sum = posterior[i * n_states + parent];
        i += 1;
        while i < n_markers && calls[i] == Some(parent) {
            conf_sum += posterior[i * n_states + parent];
            i += 1;
        }
        let run_end = i - 1;
        let run_len = run_end - run_start + 1;
        let followed_by_called = i < n_markers && calls[i].is_some();
        let (end_bp, end_cm) = if followed_by_called {
            (
                positions_bp[i].saturating_sub(1).max(positions_bp[run_start]),
                cm_positions[i] as f32,
            )
        } else {
            (
                positions_bp[run_end],
                cm_positions[run_end] as f32,
            )
        };
        segments.push(AncestrySegment {
            target_sample,
            hap: 0,
            chrom_idx,
            start_bp: positions_bp[run_start],
            end_bp,
            start_cm: cm_positions[run_start] as f32,
            end_cm,
            parent_sample: parent as u32,
            parent_hap: 0,
            confidence: (conf_sum / run_len as f64).clamp(0.0, 1.0) as f32,
            ambiguous_parents: 1,
            n_markers: run_len as u32,
        });
    }
    segments
}

/// Run author-compatible Mott HMM ancestry painting on a single chromosome.
#[allow(clippy::too_many_arguments)]
pub(crate) fn paint_chromosome_mott_cancellable(
    joint_matrix: &BitHaplotypeMatrix,
    ref_layout: &SampleHaplotypeLayout,
    target_layout: &SampleHaplotypeLayout,
    chrom_idx: u16,
    positions_bp: &[u64],
    cm_positions: &[f64],
    config: &HaploidIbdConfig,
    cancelled: Option<&AtomicBool>,
) -> Result<MottChromosomeOutput, String> {
    let core_cfg = MottCoreConfig::from_ibd_config(config)?;
    let n_founders = ref_layout.n_samples;
    let n_targets = target_layout.n_samples;
    let n_total_markers = joint_matrix.n_markers;

    if n_founders == 0 || n_targets == 0 || n_total_markers == 0 {
        let empty_bins =
            build_union_breakpoint_sparse_matrix(chrom_idx, target_layout, n_founders, &[]);
        return Ok(MottChromosomeOutput {
            segments: Vec::new(),
            bins: empty_bins,
            posterior: None,
            selected_markers: 0,
        });
    }
    if positions_bp.len() != n_total_markers || cm_positions.len() != n_total_markers {
        return Err(format!(
            "marker coordinate mismatch: matrix={}, bp={}, cm={}",
            n_total_markers,
            positions_bp.len(),
            cm_positions.len()
        ));
    }
    if ref_layout.is_inbred.iter().any(|&b| !b) || target_layout.is_inbred.iter().any(|&b| !b) {
        return Err(
            "Mott ancestry model requires all reference and target samples to be inbred \
             (1 haplotype track per sample); pass --ploidy inbred"
                .to_owned(),
        );
    }
    validate_inbred_layout(ref_layout, "reference", joint_matrix.n_haps)?;
    validate_inbred_layout(target_layout, "target", joint_matrix.n_haps)?;
    if ref_layout.total_haps + target_layout.total_haps != joint_matrix.n_haps {
        return Err(format!(
            "joint matrix haplotype count ({}) does not equal reference ({}) + target ({})",
            joint_matrix.n_haps, ref_layout.total_haps, target_layout.total_haps
        ));
    }
    if config.mott_write_posterior && n_targets != 1 {
        return Err(format!(
            "--mott-write-posterior requires exactly 1 target sample, got {n_targets}"
        ));
    }

    // Select markers: when `mott_max_markers_per_chrom == 0`, keep ALL aligned
    // markers (for exact author demo parity); when `> 0`, filter to markers with
    // >= 2 distinct founder homozygous states and evenly select up to the cap.
    let selected_indices: Vec<usize> = if config.mott_max_markers_per_chrom == 0 {
        (0..n_total_markers).collect()
    } else {
        const MARKER_SCAN_CHUNK: usize = 32_768;
        let mut informative = Vec::with_capacity(n_total_markers);
        for start in (0..n_total_markers).step_by(MARKER_SCAN_CHUNK) {
            if is_cancelled(cancelled) {
                return Err("Interrupted".to_owned());
            }
            let end = (start + MARKER_SCAN_CHUNK).min(n_total_markers);
            let chunk: Vec<bool> = (start..end)
                .into_par_iter()
                .map(|m| {
                    let mut seen0 = false;
                    let mut seen1 = false;
                    for f in 0..n_founders {
                        let h = ref_layout.hap_offset[f];
                        match joint_matrix.allele(m, h) {
                            Some(0) => seen0 = true,
                            Some(1) => seen1 = true,
                            _ => {}
                        }
                        if seen0 && seen1 {
                            return true;
                        }
                    }
                    false
                })
                .collect();
            informative.extend(chunk);
        }
        select_even_marker_indices(&informative, config.mott_max_markers_per_chrom)
    };

    let n_sel = selected_indices.len();
    if n_sel == 0 {
        let empty_bins =
            build_union_breakpoint_sparse_matrix(chrom_idx, target_layout, n_founders, &[]);
        return Ok(MottChromosomeOutput {
            segments: Vec::new(),
            bins: empty_bins,
            posterior: None,
            selected_markers: 0,
        });
    }

    let sel_bp: Vec<u64> = selected_indices.iter().map(|&m| positions_bp[m]).collect();
    let sel_cm: Vec<f64> = selected_indices.iter().map(|&m| cm_positions[m]).collect();

    // Precompute founder emission vectors for target allele 0 and allele 1 once per marker.
    let mut emit0 = vec![0.0f64; n_sel * n_founders];
    let mut emit1 = vec![0.0f64; n_sel * n_founders];
    for start in (0..n_sel).step_by(4096) {
        if is_cancelled(cancelled) {
            return Err("Interrupted".to_owned());
        }
        let end = (start + 4096).min(n_sel);
        emit0[start * n_founders..end * n_founders]
            .par_chunks_mut(n_founders)
            .zip(emit1[start * n_founders..end * n_founders].par_chunks_mut(n_founders))
            .zip(selected_indices[start..end].par_iter())
            .for_each(|((row0, row1), &m)| {
                let mut fcalls = Vec::with_capacity(n_founders);
                for f in 0..n_founders {
                    let h = ref_layout.hap_offset[f];
                    fcalls.push(match joint_matrix.allele(m, h) {
                        Some(a) => MottFounderCall::Homozygous(a),
                        None => MottFounderCall::Ambiguous,
                    });
                }
                let e0 = founder_emission(&fcalls, Some(0), core_cfg.parent_match_prob);
                let e1 = founder_emission(&fcalls, Some(1), core_cfg.parent_match_prob);
                row0.copy_from_slice(&e0);
                row1.copy_from_slice(&e1);
            });
    }
    let uniform_value = (1000.0 / n_founders as f64).round() / 1000.0;
    let uniform = vec![uniform_value; n_founders];
    let target_base = ref_layout.total_haps;

    // Process targets in bounded chunks of 256 so cancellation is checked regularly
    // and memory never holds more than one chunk's temporary posteriors.
    const TARGET_CHUNK: usize = 256;
    let mut all_segments: Vec<AncestrySegment> = Vec::new();
    let mut exported_posterior: Option<MottPosteriorMatrix> = None;

    for chunk_start in (0..n_targets).step_by(TARGET_CHUNK) {
        if is_cancelled(cancelled) {
            return Err("Interrupted".to_owned());
        }
        let chunk_end = (chunk_start + TARGET_CHUNK).min(n_targets);
        let chunk_res: Result<Vec<(Vec<AncestrySegment>, Option<Vec<f64>>)>, String> = (chunk_start
            ..chunk_end)
            .into_par_iter()
            .map(|s| {
                if is_cancelled(cancelled) {
                    return Err("Interrupted".to_owned());
                }
                let h = target_base + target_layout.hap_offset[s];
                let mut tcalls = Vec::with_capacity(n_sel);
                for &m in &selected_indices {
                    tcalls.push(joint_matrix.allele(m, h));
                }
                let post = posterior_one_sample_precomputed(
                    n_sel,
                    n_founders,
                    &emit0,
                    &emit1,
                    &uniform,
                    &tcalls,
                    &sel_cm,
                    &core_cfg,
                    cancelled,
                )?;
                let calls = hard_calls(&post, n_sel, n_founders, core_cfg.unknown_fold)?;
                let segs = compress_calls_to_segments(
                    s as u32, chrom_idx, &calls, &post, n_founders, &sel_bp, &sel_cm,
                );
                let keep_post = if config.mott_write_posterior {
                    Some(post)
                } else {
                    None
                };
                Ok((segs, keep_post))
            })
            .collect();

        for (segs, post_opt) in chunk_res? {
            all_segments.extend(segs);
            if let Some(values) = post_opt {
                exported_posterior = Some(MottPosteriorMatrix {
                    n_states: n_founders,
                    values,
                });
            }
        }
    }

    if is_cancelled(cancelled) {
        return Err("Interrupted".to_owned());
    }
    let bins = build_union_breakpoint_sparse_matrix_cancellable(
        chrom_idx,
        target_layout,
        n_founders,
        &all_segments,
        cancelled,
    )?;

    Ok(MottChromosomeOutput {
        segments: all_segments,
        bins,
        posterior: exported_posterior,
        selected_markers: n_sel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn founder_emission_matches_author_homozygous_case() {
        let calls = [
            MottFounderCall::Homozygous(0),
            MottFounderCall::Homozygous(0),
            MottFounderCall::Homozygous(1),
        ];
        let got = founder_emission(&calls, Some(0), 0.97);
        assert!((got[0] - 0.485).abs() < 1e-15);
        assert!((got[1] - 0.485).abs() < 1e-15);
        assert!((got[2] - 0.03).abs() < 1e-15);
    }

    #[test]
    fn author_forward_step_matches_r_recurrence() {
        let got = author_forward_step(&[0.75, 0.25], &[0.8, 0.2], 0.75, 0.25);
        let expected = [0.8351648351648351, 0.16483516483516483];
        assert!((got[0] - expected[0]).abs() < 1e-15);
        assert!((got[1] - expected[1]).abs() < 1e-15);
    }

    #[test]
    fn zero_interval_uses_author_floor_lambda() {
        let cfg = MottCoreConfig {
            generations: 9.0,
            parent_match_prob: 0.97,
            map_scale_cm: 50.0,
            unknown_fold: 2.0,
        };
        assert_eq!(interval_lambda(0.0, &cfg), 9.0 * 2e-8);
    }

    #[test]
    fn ambiguous_or_missing_target_uses_uniform_emission() {
        let calls = [
            MottFounderCall::Homozygous(0),
            MottFounderCall::Homozygous(1),
        ];
        assert_eq!(founder_emission(&calls, None, 0.97), vec![0.5, 0.5]);
    }

    #[test]
    fn author_rounded_uniform_and_monomorphic_endpoint_emissions() {
        let missing_target = vec![MottFounderCall::Homozygous(0); 24];
        let uniform = founder_emission(&missing_target, None, 0.97);
        assert!(uniform.iter().all(|&value| value == 0.042));

        let present = founder_emission(&missing_target, Some(0), 0.97);
        assert!(present.iter().all(|&value| (value - 0.97 / 24.0).abs() < 1e-15));

        let absent = founder_emission(&missing_target, Some(1), 0.97);
        assert!(absent.iter().all(|&value| (value - 0.03 / 24.0).abs() < 1e-15));
    }

    #[test]
    fn ambiguous_founder_uses_author_half_credit_fallback() {
        let calls = [MottFounderCall::Homozygous(0), MottFounderCall::Ambiguous];
        let got = founder_emission(&calls, Some(0), 0.97);
        assert!((got[0] - 0.6466666666666666).abs() < 1e-15);
        assert!((got[1] - 0.3233333333333333).abs() < 1e-15);
    }

    #[test]
    fn hard_calls_enforces_strict_unknown_threshold_and_even_selection() {
        assert_eq!(hard_calls(&[0.5, 0.5], 1, 2, 2.0).unwrap(), vec![None]);
        assert_eq!(hard_calls(&[0.51, 0.49], 1, 2, 1.0).unwrap(), vec![Some(0)]);
        assert_eq!(
            hard_calls(&[0.5, 0.25, 0.25, 0.51, 0.245, 0.245], 2, 3, 1.5).unwrap(),
            vec![None, Some(0)]
        );
        assert_eq!(
            select_even_marker_indices(&[true, false, true, true, false, true, true], 3),
            vec![0, 3, 6]
        );
    }

    #[test]
    fn compress_calls_skips_unknown_and_splits_segments() {
        let calls = vec![Some(0), Some(0), Some(1), None, Some(1)];
        let posterior = vec![
            0.9, 0.1, // m0
            0.8, 0.2, // m1
            0.1, 0.9, // m2
            0.5, 0.5, // m3 (unknown)
            0.2, 0.8, // m4
        ];
        let bp = vec![100, 200, 300, 400, 500];
        let cm = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let segs = compress_calls_to_segments(0, 0, &calls, &posterior, 2, &bp, &cm);
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(
            (segs[0].start_bp, segs[0].end_bp, segs[0].n_markers),
            (100, 299, 2)
        );
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(
            (segs[1].start_bp, segs[1].end_bp, segs[1].n_markers),
            (300, 300, 1)
        );
        assert_eq!(segs[2].parent_sample, 1);
        assert_eq!(
            (segs[2].start_bp, segs[2].end_bp, segs[2].n_markers),
            (500, 500, 1)
        );
    }

    #[test]
    fn compress_calls_closes_adjacent_switch_at_recombination_boundary() {
        let calls = vec![Some(0), Some(0), Some(1), Some(1)];
        let posterior = vec![
            0.9, 0.1, // m0
            0.8, 0.2, // m1
            0.1, 0.9, // m2
            0.2, 0.8, // m3
        ];
        let bp = vec![100, 200, 350, 500];
        let cm = vec![0.1, 0.2, 0.3, 0.4];
        let segs = compress_calls_to_segments(0, 0, &calls, &posterior, 2, &bp, &cm);
        assert_eq!(segs.len(), 2);
        assert_eq!((segs[0].start_bp, segs[0].end_bp), (100, 349));
        assert_eq!((segs[1].start_bp, segs[1].end_bp), (350, 500));
        assert_eq!(segs[0].end_bp + 1, segs[1].start_bp);
        assert!((segs[0].end_cm - 0.3).abs() < 1e-6);
    }

    #[test]
    fn paint_chromosome_mott_recovers_switch_and_rejects_diploid() {
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let n_markers = 20;
        let mut matrix = BitHaplotypeMatrix::new(4, n_markers);
        let mut bp = Vec::with_capacity(n_markers);
        let mut cm = Vec::with_capacity(n_markers);
        for m in 0..n_markers {
            bp.push((m as u64 + 1) * 100_000);
            cm.push((m as f64) * 0.5);
            // F0 = 0, F1 = 1, F2 = alternating
            let f0 = 0u8;
            let f1 = 1u8;
            let f2 = (m % 2) as u8;
            let tgt = if m < 10 { f0 } else { f1 };
            matrix.set_allele(m, 0, Some(f0));
            matrix.set_allele(m, 1, Some(f1));
            matrix.set_allele(m, 2, Some(f2));
            matrix.set_allele(m, 3, Some(tgt));
        }
        let cfg = HaploidIbdConfig {
            ancestry_model: super::super::types::AncestryModel::Mott,
            mott_write_posterior: true,
            ..HaploidIbdConfig::default()
        };
        let out = paint_chromosome_mott_cancellable(
            &matrix,
            &ref_layout,
            &tgt_layout,
            0,
            &bp,
            &cm,
            &cfg,
            None,
        )
        .unwrap();
        assert_eq!(out.selected_markers, n_markers);
        assert!(out.posterior.is_some());
        assert!(out.segments.len() >= 2);
        assert_eq!(out.segments.first().unwrap().parent_sample, 0);
        assert_eq!(out.segments.last().unwrap().parent_sample, 1);

        // Diploid target must be rejected in Mott mode.
        let dip_layout = SampleHaplotypeLayout::from_inbred_flags(vec![false]);
        let err = paint_chromosome_mott_cancellable(
            &matrix,
            &ref_layout,
            &dip_layout,
            0,
            &bp,
            &cm,
            &cfg,
            None,
        )
        .expect_err("diploid target must be rejected in Mott mode");
        assert!(err.contains("inbred"));
    }

    #[test]
    fn mott_rejects_malformed_inbred_layout_without_panicking() {
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let malformed_target = SampleHaplotypeLayout {
            n_samples: 1,
            is_inbred: vec![true],
            hap_offset: Vec::new(),
            hap_count: vec![1],
            hap_to_sample: vec![(0, 0)],
            total_haps: 1,
        };
        let matrix = BitHaplotypeMatrix::new(2, 1);
        let cfg = HaploidIbdConfig {
            ancestry_model: super::super::types::AncestryModel::Mott,
            ..HaploidIbdConfig::default()
        };
        let err = paint_chromosome_mott_cancellable(
            &matrix,
            &ref_layout,
            &malformed_target,
            0,
            &[100],
            &[0.0],
            &cfg,
            None,
        )
        .expect_err("malformed target layout must be rejected");
        assert!(err.contains("target layout"));
    }

    #[test]
    fn mott_posterior_checks_cancellation_during_compute() {
        let founder_calls = vec![MottFounderCall::Homozygous(0); 2];
        let emit0 = founder_emission(&founder_calls, Some(0), 0.97);
        let emit1 = founder_emission(&founder_calls, Some(1), 0.97);
        let uniform = founder_emission(&founder_calls, None, 0.97);
        let cfg = MottCoreConfig {
            generations: 9.0,
            parent_match_prob: 0.97,
            map_scale_cm: 50.0,
            unknown_fold: 2.0,
        };
        let cancelled = AtomicBool::new(true);
        let err = posterior_one_sample_precomputed(
            2,
            2,
            &[emit0.clone(), emit0].concat(),
            &[emit1.clone(), emit1].concat(),
            &uniform,
            &[Some(0), Some(1)],
            &[0.0, 0.1],
            &cfg,
            Some(&cancelled),
        )
        .expect_err("pre-cancelled posterior compute must stop");
        assert!(err.contains("Interrupted"));
    }
}
