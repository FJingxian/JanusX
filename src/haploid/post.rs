//! Fast post-processing and aggregation directly from `.jhibd` sparse archives
//! (`HaploidSparseArchive`) for `jx postibd`.
//!
//! Computes:
//! 1. Empirical population recombination rate & cumulative genetic map per
//!    physical window from adjacent ancestry breakpoints, enforcing per-haplotype
//!    numerator/denominator consistency (`>= 50%` window coverage per track) plus
//!    a population-level coverage gate.
//! 2. Genome-wide & per-chromosome parent contribution and windowed descriptive
//!    segregation distortion directly from `ChromosomeAncestryBinsSparse`.
//! 3. Sample-level karyotype mosaic segments and physically-proportioned
//!    multi-chromosome population mosaic matrices using cumulative per-parent
//!    overlap weights.

use super::sparse_io::{
    infer_target_is_inbred_from_segments, read_ancestry_segments_archive, AncestrySegmentsArchive,
    HaploidSparseArchive, HaploidSparsePayload,
};
use super::types::{AncestrySegment, HaploidIbdMode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

fn default_window_bp() -> u64 {
    1_000_000
}
fn default_min_window_coverage() -> f64 {
    0.10
}
fn default_true() -> bool {
    true
}
fn default_max_mosaic_samples() -> usize {
    12
}
fn default_pop_mosaic_bins() -> usize {
    2000
}
fn default_max_pop_samples() -> usize {
    0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PostIbdConfig {
    pub input: PathBuf,
    pub window_bp: u64,
    /// Minimum fraction of `(total_haplotype_tracks * window_len)` that must be
    /// callable in a window to compute an empirical recombination rate and add
    /// to the cumulative genetic map (default: 0.10).
    pub min_window_coverage: f64,
    pub want_recomb: bool,
    pub want_segregation: bool,
    pub want_mosaic: bool,
    pub mosaic_samples: Vec<String>,
    pub max_mosaic_samples: usize,
    pub mosaic_chroms: Vec<String>,
    /// Total horizontal bin budget distributed across selected chromosomes in
    /// proportion to physical length (default: 2000).
    pub pop_mosaic_bins: usize,
    /// Maximum samples included in population mosaic heatmaps (`0` = all samples).
    pub max_pop_samples: usize,
    /// Optional mapping from reference parent ID to ancestry group name
    /// (e.g., `{"YZW1": "A", "YZW2": "B"}`) used to aggregate contributions
    /// and mosaic coloring by group while preserving founder-level breakpoints.
    pub parent_groups: HashMap<String, String>,
}

impl Default for PostIbdConfig {
    fn default() -> Self {
        Self {
            input: PathBuf::new(),
            window_bp: default_window_bp(),
            min_window_coverage: default_min_window_coverage(),
            want_recomb: default_true(),
            want_segregation: default_true(),
            want_mosaic: default_true(),
            mosaic_samples: Vec::new(),
            max_mosaic_samples: default_max_mosaic_samples(),
            mosaic_chroms: Vec::new(),
            pop_mosaic_bins: default_pop_mosaic_bins(),
            max_pop_samples: default_max_pop_samples(),
            parent_groups: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecombWindowRow {
    pub chrom_idx: u16,
    pub start_bp: u64,
    pub end_bp: u64,
    pub mid_bp: u64,
    pub breakpoints: u64,
    pub obs_haps: f64,
    pub window_coverage: f64,
    pub callable: bool,
    pub effective_cm: f64,
    pub effective_cm_per_mb: Option<f64>,
    pub cum_effective_cm: f64,
    pub input_start_cm: f64,
    pub input_end_cm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentContributionChrom {
    pub chrom_idx: u16,
    pub parent_bp: Vec<f64>,
    pub parent_cm: Vec<f64>,
    pub called_bp: f64,
    pub total_bp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegregationWindowRow {
    pub chrom_idx: u16,
    pub start_bp: u64,
    pub end_bp: u64,
    pub mid_bp: u64,
    pub eff_samples: f64,
    pub uncalled_frac: f64,
    pub expected_per_parent: f64,
    /// Descriptive chi-square divergence from uniform `1/K` based on window-averaged
    /// parent dosages (not calibrated to an asymptotic p-value).
    pub chi2: f64,
    pub max_parent_idx: u32,
    pub max_parent_prop: f64,
    pub parent_props: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MosaicSampleInfo {
    pub sample_idx: u32,
    pub sample_id: String,
    pub n_haps: u8,
    pub total_breakpoints: u64,
}

/// Compact segment tuple for Python karyotype rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MosaicSegmentCompact {
    pub chrom_idx: u16,
    pub local_sample_idx: u32,
    pub hap: u8,
    pub start_bp: u64,
    pub end_bp: u64,
    pub parent_sample: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopMosaicChrom {
    pub chrom_idx: u16,
    pub n_samples: usize,
    pub n_bins: usize,
    pub chrom_length_bp: u64,
    pub sample_indices: Vec<u32>,
    /// Row-major `n_samples * n_bins` matrix of dominant parent index (`-1` = uncalled).
    pub parent_matrix: Vec<i16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostIbdResult {
    pub mode: HaploidIbdMode,
    pub n_target_samples: usize,
    pub n_reference_samples: usize,
    pub total_segments: usize,
    pub total_haplotype_tracks: usize,
    pub chrom_names: Vec<String>,
    pub chrom_lengths_bp: Vec<u64>,
    #[serde(default)]
    pub target_sample_ids: Vec<String>,
    pub reference_sample_ids: Vec<String>,
    #[serde(default)]
    pub grouped_parents: bool,
    pub recomb_windows: Vec<RecombWindowRow>,
    pub parent_contrib_chroms: Vec<ParentContributionChrom>,
    pub segregation_windows: Vec<SegregationWindowRow>,
    pub mosaic_samples: Vec<MosaicSampleInfo>,
    pub mosaic_segments: Vec<MosaicSegmentCompact>,
    pub pop_mosaics: Vec<PopMosaicChrom>,
}

/// Load a `.jhibd` archive from disk (`JHIBD01` or `JHIBD02`) and extract all
/// requested post-processing summaries directly from `segments` without
/// materializing multi-GB expanded `chrom_bins`.
pub fn extract_post_ibd(config: &PostIbdConfig) -> Result<PostIbdResult, String> {
    if config.input.as_os_str().is_empty() {
        return Err("postibd requires a .jhibd input file".to_owned());
    }
    if config.window_bp == 0 {
        return Err("window_bp must be > 0".to_owned());
    }
    let archive = read_ancestry_segments_archive(&config.input).map_err(|err| {
        if err.contains("PairwiseIbd") {
            "input .jhibd archive is in pairwise_ibd mode (no --ref); recombination, \
             segregation distortion, and mosaic plots require a multi-parent ancestry \
             archive produced by `jx ibd -ref ...`"
                .to_owned()
        } else {
            err
        }
    })?;
    extract_post_ibd_from_segments(&archive, config)
}

/// Core aggregation operating directly on an in-memory `HaploidSparseArchive`.
pub fn extract_post_ibd_from_archive(
    archive: &HaploidSparseArchive,
    config: &PostIbdConfig,
) -> Result<PostIbdResult, String> {
    let segments = match &archive.payload {
        HaploidSparsePayload::MultiParentAncestry { segments, .. } => segments.clone(),
        HaploidSparsePayload::PairwiseIbd { .. } => {
            return Err(
                "input .jhibd archive is in pairwise_ibd mode (no --ref); recombination, \
                 segregation distortion, and mosaic plots require a multi-parent ancestry \
                 archive produced by `jx ibd -ref ...`"
                    .to_owned(),
            );
        }
    };
    let target_is_inbred =
        infer_target_is_inbred_from_segments(archive.target_sample_ids.len(), &segments);
    let compact = AncestrySegmentsArchive {
        version: archive.version,
        mode: archive.mode,
        target_sample_ids: archive.target_sample_ids.clone(),
        reference_sample_ids: archive.reference_sample_ids.clone(),
        chrom_names: archive.chrom_names.clone(),
        target_is_inbred,
        segments,
        fallback_chrom_bins: Vec::new(),
    };
    extract_post_ibd_from_segments(&compact, config)
}

/// Core aggregation operating directly on `AncestrySegmentsArchive` (segments +
/// per-sample ploidy flags, zero `chrom_bins` required).
pub fn extract_post_ibd_from_segments(
    archive: &AncestrySegmentsArchive,
    config: &PostIbdConfig,
) -> Result<PostIbdResult, String> {
    let segments = archive.segments.as_slice();
    let n_chroms = archive.chrom_names.len();
    let n_samples = archive.target_sample_ids.len();
    let n_parents = archive.reference_sample_ids.len();
    if n_chroms == 0 || n_samples == 0 || n_parents == 0 {
        return Err(
            "archive contains no chromosomes, target samples, or reference parents".to_owned(),
        );
    }

    // Determine chromosome physical lengths and per-sample haplotype count.
    let mut chrom_lengths_bp = vec![0u64; n_chroms];
    let mut sample_n_haps: Vec<u8> = (0..n_samples)
        .map(|s| {
            if archive.target_is_inbred.get(s).copied().unwrap_or(true) {
                1u8
            } else {
                2u8
            }
        })
        .collect();
    for seg in segments {
        let c = seg.chrom_idx as usize;
        if c < n_chroms {
            chrom_lengths_bp[c] = chrom_lengths_bp[c].max(seg.end_bp);
        }
        let s = seg.target_sample as usize;
        if s < n_samples && seg.hap > 0 {
            sample_n_haps[s] = 2;
        }
    }
    let total_haplotype_tracks: usize = sample_n_haps.iter().map(|&h| h as usize).sum();

    // Partition segments by chromosome once so per-chromosome computations run in parallel.
    let mut segs_by_chrom: Vec<Vec<&AncestrySegment>> = vec![Vec::new(); n_chroms];
    for seg in segments {
        let c = seg.chrom_idx as usize;
        if c < n_chroms {
            segs_by_chrom[c].push(seg);
        }
    }

    let min_cov = if config.min_window_coverage.is_finite() {
        config.min_window_coverage.clamp(0.0, 1.0)
    } else {
        default_min_window_coverage()
    };

    // 1. Recombination rate & empirical genetic map.
    let (recomb_windows, sample_breakpoints) = if config.want_recomb || config.want_mosaic {
        compute_recomb_and_breakpoints(
            &segs_by_chrom,
            &chrom_lengths_bp,
            n_samples,
            total_haplotype_tracks,
            config.window_bp,
            min_cov,
            config.want_recomb,
        )
    } else {
        (Vec::new(), vec![0u64; n_samples])
    };

    // Resolve optional parent -> ancestry group mapping (sorted group names).
    let (group_names, parent_to_group, grouped_parents) =
        resolve_parent_groups(&archive.reference_sample_ids, &config.parent_groups)?;
    let n_groups = group_names.len();

    // 2. Parent / group contribution & windowed segregation distortion.
    let (parent_contrib_chroms, segregation_windows) = if config.want_segregation {
        compute_segregation_from_segments(
            &segs_by_chrom,
            &chrom_lengths_bp,
            &sample_n_haps,
            n_samples,
            n_groups,
            &parent_to_group,
            config.window_bp,
        )
    } else {
        (Vec::new(), Vec::new())
    };

    // 3. Mosaic karyotype segments & population mosaic heatmaps.
    let (mosaic_samples, mosaic_segments, pop_mosaics) = if config.want_mosaic {
        compute_mosaic_data(
            archive,
            &segs_by_chrom,
            &chrom_lengths_bp,
            &sample_n_haps,
            &sample_breakpoints,
            n_groups,
            &parent_to_group,
            config,
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };

    Ok(PostIbdResult {
        mode: archive.mode,
        n_target_samples: n_samples,
        n_reference_samples: n_parents,
        total_segments: segments.len(),
        total_haplotype_tracks,
        chrom_names: archive.chrom_names.clone(),
        chrom_lengths_bp,
        target_sample_ids: archive.target_sample_ids.clone(),
        reference_sample_ids: group_names,
        grouped_parents,
        recomb_windows,
        parent_contrib_chroms,
        segregation_windows,
        mosaic_samples,
        mosaic_segments,
        pop_mosaics,
    })
}

fn resolve_parent_groups(
    reference_sample_ids: &[String],
    parent_groups: &HashMap<String, String>,
) -> Result<(Vec<String>, Vec<usize>, bool), String> {
    let n_parents = reference_sample_ids.len();
    if parent_groups.is_empty() {
        return Ok((
            reference_sample_ids.to_vec(),
            (0..n_parents).collect(),
            false,
        ));
    }
    let mut matched_any = false;
    let mut mapped_group_names: HashSet<String> = HashSet::new();
    let mut mapped_flags: Vec<bool> = Vec::with_capacity(n_parents);
    let mut raw_labels: Vec<String> = Vec::with_capacity(n_parents);
    for pid in reference_sample_ids {
        if let Some(g) = parent_groups.get(pid) {
            let trimmed = g.trim();
            if !trimmed.is_empty() {
                raw_labels.push(trimmed.to_owned());
                mapped_group_names.insert(trimmed.to_owned());
                mapped_flags.push(true);
                matched_any = true;
                continue;
            }
        }
        raw_labels.push(pid.clone());
        mapped_flags.push(false);
    }
    if !matched_any {
        return Ok((
            reference_sample_ids.to_vec(),
            (0..n_parents).collect(),
            false,
        ));
    }
    for (pid, &is_mapped) in reference_sample_ids.iter().zip(mapped_flags.iter()) {
        if !is_mapped && mapped_group_names.contains(pid) {
            return Err(format!(
                "ambiguous parent group mapping: unmapped reference parent '{pid}' has the same \
                 name as a mapped ancestry group; rename the group or explicitly map '{pid}'"
            ));
        }
    }
    let mut group_names: Vec<String> = raw_labels.clone();
    group_names.sort();
    group_names.dedup();
    let name_to_idx: HashMap<&str, usize> = group_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let parent_to_group: Vec<usize> = raw_labels
        .iter()
        .map(|lbl| name_to_idx[lbl.as_str()])
        .collect();
    Ok((group_names, parent_to_group, true))
}

fn build_bp_cm_anchors(chrom_segs: &[&AncestrySegment]) -> Vec<(u64, f64)> {
    let mut anchors: Vec<(u64, f64)> = Vec::with_capacity(chrom_segs.len() * 2);
    for seg in chrom_segs {
        anchors.push((seg.start_bp, seg.start_cm as f64));
        anchors.push((seg.end_bp, seg.end_cm as f64));
    }
    anchors.sort_unstable_by_key(|&(bp, _)| bp);
    anchors.dedup_by_key(|(bp, _)| *bp);
    anchors
}

fn interpolate_bp_cm(anchors: &[(u64, f64)], bp: u64) -> f64 {
    let n = anchors.len();
    if n == 0 {
        return 0.0;
    }
    if bp <= anchors[0].0 {
        return anchors[0].1;
    }
    if bp >= anchors[n - 1].0 {
        return anchors[n - 1].1;
    }
    let idx = anchors.partition_point(|&(p, _)| p <= bp).saturating_sub(1);
    let (bp0, cm0) = anchors[idx];
    if bp == bp0 || idx + 1 >= n {
        return cm0;
    }
    let (bp1, cm1) = anchors[idx + 1];
    if bp1 <= bp0 {
        return cm0;
    }
    let frac = (bp - bp0) as f64 / (bp1 - bp0) as f64;
    cm0 + frac * (cm1 - cm0)
}

#[allow(clippy::too_many_arguments)]
fn compute_recomb_and_breakpoints(
    segs_by_chrom: &[Vec<&AncestrySegment>],
    chrom_lengths_bp: &[u64],
    n_samples: usize,
    total_haplotype_tracks: usize,
    window_bp: u64,
    min_window_coverage: f64,
    emit_windows: bool,
) -> (Vec<RecombWindowRow>, Vec<u64>) {
    let total_tracks_f = (total_haplotype_tracks.max(1)) as f64;
    let per_chrom: Vec<(Vec<RecombWindowRow>, Vec<u64>)> = (0..segs_by_chrom.len())
        .into_par_iter()
        .map(|c| {
            let mut local_sample_bp = vec![0u64; n_samples];
            let max_bp = chrom_lengths_bp[c];
            let chrom_segs = &segs_by_chrom[c];
            if max_bp == 0 || chrom_segs.is_empty() {
                return (Vec::new(), local_sample_bp);
            }

            // Merge any trailing tail shorter than `window_bp / 2` into the preceding
            // window so a 1-bp remainder at the end of a chromosome never produces a
            // spurious 100M cM/Mb hotspot.
            let raw_n_win = max_bp.div_ceil(window_bp) as usize;
            let rem = max_bp % window_bp;
            let n_win = if raw_n_win >= 2 && rem > 0 && rem.saturating_mul(2) < window_bp {
                raw_n_win - 1
            } else {
                raw_n_win
            };
            let win_bounds = |w: usize| -> (u64, u64) {
                let wb0 = w as u64 * window_bp + 1;
                let wb1 = if w + 1 == n_win {
                    max_bp
                } else {
                    ((w as u64 + 1) * window_bp).min(max_bp)
                };
                (wb0, wb1)
            };

            let mut win_breakpoints = if emit_windows {
                vec![0u64; n_win]
            } else {
                Vec::new()
            };
            let mut win_obs_haps = if emit_windows {
                vec![0.0f64; n_win]
            } else {
                Vec::new()
            };

            // Group by (target_sample, hap) via sorting a compact index array so
            // each haplotype track is processed as a single contiguous run.
            let mut order: Vec<usize> = (0..chrom_segs.len()).collect();
            order.sort_unstable_by(|&a, &b| {
                let sa = chrom_segs[a];
                let sb = chrom_segs[b];
                sa.target_sample
                    .cmp(&sb.target_sample)
                    .then_with(|| sa.hap.cmp(&sb.hap))
                    .then_with(|| sa.start_bp.cmp(&sb.start_bp))
            });

            // Per-track scratch buffers so numerator (breakpoints) and denominator
            // (obs_haps) are strictly restricted to the SAME callable tracks per
            // window (`track_win_bp[w] >= 50%` of `win_len`).
            let mut track_win_bp = if emit_windows {
                vec![0u64; n_win]
            } else {
                Vec::new()
            };
            let mut track_win_breaks = if emit_windows {
                vec![0u64; n_win]
            } else {
                Vec::new()
            };
            let mut touched_wins: Vec<usize> = Vec::new();

            let flush_track = |touched: &mut Vec<usize>,
                               t_bp: &mut [u64],
                               t_brk: &mut [u64],
                               w_obs: &mut [f64],
                               w_brk: &mut [u64]| {
                for &w in touched.iter() {
                    let (wb0, wb1) = win_bounds(w);
                    let win_len = wb1.saturating_sub(wb0).saturating_add(1);
                    let called_bp = t_bp[w];
                    let brk = t_brk[w];
                    // Include a haplotype track in window `w`'s numerator AND
                    // denominator iff it is called over >= 50% of window `w`.
                    if win_len > 0 && called_bp.saturating_mul(2) >= win_len {
                        w_obs[w] += called_bp as f64 / win_len as f64;
                        w_brk[w] = w_brk[w].saturating_add(brk);
                    }
                    t_bp[w] = 0;
                    t_brk[w] = 0;
                }
                touched.clear();
            };

            let mut prev_seg: Option<&AncestrySegment> = None;
            for &idx in &order {
                let seg = chrom_segs[idx];
                let same_track = prev_seg
                    .map(|p| p.target_sample == seg.target_sample && p.hap == seg.hap)
                    .unwrap_or(false);
                if emit_windows && prev_seg.is_some() && !same_track {
                    flush_track(
                        &mut touched_wins,
                        &mut track_win_bp,
                        &mut track_win_breaks,
                        &mut win_obs_haps,
                        &mut win_breakpoints,
                    );
                }

                if emit_windows {
                    let w_start =
                        ((seg.start_bp.saturating_sub(1) / window_bp) as usize).min(n_win - 1);
                    let w_end =
                        ((seg.end_bp.saturating_sub(1) / window_bp) as usize).min(n_win - 1);
                    for w in w_start..=w_end {
                        let (wb0, wb1) = win_bounds(w);
                        let ov0 = seg.start_bp.max(wb0);
                        let ov1 = seg.end_bp.min(wb1);
                        if ov1 >= ov0 {
                            if track_win_bp[w] == 0 && track_win_breaks[w] == 0 {
                                touched_wins.push(w);
                            }
                            track_win_bp[w] = track_win_bp[w].saturating_add(ov1 - ov0 + 1);
                        }
                    }
                }

                if same_track {
                    if let Some(prev) = prev_seg {
                        let switched = prev.parent_sample != seg.parent_sample
                            || prev.parent_hap != seg.parent_hap;
                        // Contiguous recombination boundary (adjacent called segments share
                        // `prev.end_bp + 1 == seg.start_bp`, avoiding false breaks across
                        // uncalled missing gaps).
                        let contiguous = prev.end_bp.saturating_add(1) >= seg.start_bp;
                        if switched && contiguous {
                            let s = seg.target_sample as usize;
                            if s < n_samples {
                                local_sample_bp[s] = local_sample_bp[s].saturating_add(1);
                            }
                            if emit_windows {
                                let w = ((seg.start_bp.saturating_sub(1) / window_bp) as usize)
                                    .min(n_win - 1);
                                if track_win_bp[w] == 0 && track_win_breaks[w] == 0 {
                                    touched_wins.push(w);
                                }
                                track_win_breaks[w] = track_win_breaks[w].saturating_add(1);
                            }
                        }
                    }
                }
                prev_seg = Some(seg);
            }
            if emit_windows && !touched_wins.is_empty() {
                flush_track(
                    &mut touched_wins,
                    &mut track_win_bp,
                    &mut track_win_breaks,
                    &mut win_obs_haps,
                    &mut win_breakpoints,
                );
            }

            if !emit_windows {
                return (Vec::new(), local_sample_bp);
            }

            let cm_anchors = build_bp_cm_anchors(chrom_segs);
            let mut rows = Vec::with_capacity(n_win);
            let mut cum_cm = 0.0f64;
            for w in 0..n_win {
                let (wb0, wb1) = win_bounds(w);
                let win_len_bp = wb1.saturating_sub(wb0).saturating_add(1);
                let win_mb = win_len_bp as f64 / 1_000_000.0;
                let obs = win_obs_haps[w];
                let bp_cnt = win_breakpoints[w];
                let window_coverage = (obs / total_tracks_f).clamp(0.0, 1.0);
                // Require at least half a nominal window length so standalone
                // ultra-short contigs also never extrapolate to extreme cM/Mb.
                let callable = win_len_bp.saturating_mul(2) >= window_bp
                    && window_coverage >= min_window_coverage
                    && obs > 0.0
                    && win_mb > 0.0;
                let (eff_cm, eff_rate) = if callable {
                    let cm = (bp_cnt as f64 / obs) * 100.0;
                    (cm, Some(cm / win_mb))
                } else {
                    (0.0, None)
                };
                cum_cm += eff_cm;
                let in_s_cm = interpolate_bp_cm(&cm_anchors, wb0);
                let in_e_cm = interpolate_bp_cm(&cm_anchors, wb1);
                rows.push(RecombWindowRow {
                    chrom_idx: c as u16,
                    start_bp: wb0,
                    end_bp: wb1,
                    mid_bp: wb0 + win_len_bp / 2,
                    breakpoints: bp_cnt,
                    obs_haps: obs,
                    window_coverage,
                    callable,
                    effective_cm: eff_cm,
                    effective_cm_per_mb: eff_rate,
                    cum_effective_cm: cum_cm,
                    input_start_cm: in_s_cm,
                    input_end_cm: in_e_cm,
                });
            }
            (rows, local_sample_bp)
        })
        .collect();

    let mut all_rows = Vec::new();
    let mut total_sample_bp = vec![0u64; n_samples];
    for (rows, local_bp) in per_chrom {
        all_rows.extend(rows);
        for (dst, src) in total_sample_bp.iter_mut().zip(local_bp.iter()) {
            *dst = dst.saturating_add(*src);
        }
    }
    (all_rows, total_sample_bp)
}

fn compute_segregation_from_segments(
    segs_by_chrom: &[Vec<&AncestrySegment>],
    chrom_lengths_bp: &[u64],
    sample_n_haps: &[u8],
    n_samples: usize,
    n_parents: usize,
    parent_to_group: &[usize],
    window_bp: u64,
) -> (Vec<ParentContributionChrom>, Vec<SegregationWindowRow>) {
    let per_chrom: Vec<(ParentContributionChrom, Vec<SegregationWindowRow>)> = (0..segs_by_chrom
        .len())
        .into_par_iter()
        .map(|c| {
            let mut parent_bp = vec![0.0f64; n_parents];
            let mut parent_cm = vec![0.0f64; n_parents];
            let mut called_bp = 0.0f64;
            let max_bp = chrom_lengths_bp[c];
            let total_bp = (n_samples as f64) * (max_bp.saturating_add(1) as f64);
            let chrom_segs = &segs_by_chrom[c];

            if chrom_segs.is_empty() || max_bp == 0 {
                return (
                    ParentContributionChrom {
                        chrom_idx: c as u16,
                        parent_bp,
                        parent_cm,
                        called_bp,
                        total_bp,
                    },
                    Vec::new(),
                );
            }

            let n_win = max_bp.div_ceil(window_bp) as usize;
            let mut win_parent_bp = vec![0.0f64; n_win * n_parents];

            for seg in chrom_segs {
                let bs = seg.start_bp;
                let be = seg.end_bp;
                if be < bs {
                    continue;
                }
                let s = seg.target_sample as usize;
                if s >= n_samples {
                    continue;
                }
                let raw_p = seg.parent_sample as usize;
                if raw_p >= parent_to_group.len() {
                    continue;
                }
                let p = parent_to_group[raw_p];
                if p >= n_parents {
                    continue;
                }
                let weight = if sample_n_haps[s] <= 1 {
                    1.0f64
                } else {
                    0.5f64
                };
                let blen_bp = (be - bs + 1) as f64;
                let blen_cm = (seg.end_cm - seg.start_cm).max(0.0) as f64;
                called_bp += weight * blen_bp;
                parent_bp[p] += weight * blen_bp;
                parent_cm[p] += weight * blen_cm;

                let w_start = (bs.saturating_sub(1) / window_bp) as usize;
                let w_end = ((be.saturating_sub(1) / window_bp) as usize).min(n_win - 1);
                for w in w_start..=w_end {
                    let wb0 = w as u64 * window_bp + 1;
                    let wb1 = ((w as u64 + 1) * window_bp).min(max_bp);
                    let ov0 = bs.max(wb0);
                    let ov1 = be.min(wb1);
                    if ov1 >= ov0 {
                        let ov_bp = (ov1 - ov0 + 1) as f64;
                        win_parent_bp[w * n_parents + p] += weight * ov_bp;
                    }
                }
            }

            let mut seg_rows = Vec::with_capacity(n_win);
            let k_f64 = n_parents as f64;
            for w in 0..n_win {
                let wb0 = w as u64 * window_bp + 1;
                let wb1 = ((w as u64 + 1) * window_bp).min(max_bp);
                let win_len_bp = wb1.saturating_sub(wb0).saturating_add(1);
                let win_len_f = win_len_bp as f64;
                let w_off = w * n_parents;

                let mut sum_called_bp = 0.0f64;
                for p in 0..n_parents {
                    sum_called_bp += win_parent_bp[w_off + p];
                }
                let eff_samples = if win_len_f > 0.0 {
                    sum_called_bp / win_len_f
                } else {
                    0.0
                };
                let uncalled_frac = if n_samples > 0 {
                    (1.0 - eff_samples / n_samples as f64).clamp(0.0, 1.0)
                } else {
                    1.0
                };

                let mut props = vec![0.0f64; n_parents];
                let mut chi2 = 0.0f64;
                let mut max_p = 0u32;
                let mut max_prop = 0.0f64;
                let expected = if k_f64 > 0.0 {
                    eff_samples / k_f64
                } else {
                    0.0
                };
                if sum_called_bp > 0.0 && expected > 0.0 {
                    for p in 0..n_parents {
                        let prop = win_parent_bp[w_off + p] / sum_called_bp;
                        props[p] = prop;
                        if prop > max_prop {
                            max_prop = prop;
                            max_p = p as u32;
                        }
                        let obs = win_parent_bp[w_off + p] / win_len_f;
                        let diff = obs - expected;
                        chi2 += (diff * diff) / expected;
                    }
                }

                seg_rows.push(SegregationWindowRow {
                    chrom_idx: c as u16,
                    start_bp: wb0,
                    end_bp: wb1,
                    mid_bp: wb0 + win_len_bp / 2,
                    eff_samples,
                    uncalled_frac,
                    expected_per_parent: expected,
                    chi2,
                    max_parent_idx: max_p,
                    max_parent_prop: max_prop,
                    parent_props: props,
                });
            }

            (
                ParentContributionChrom {
                    chrom_idx: c as u16,
                    parent_bp,
                    parent_cm,
                    called_bp,
                    total_bp,
                },
                seg_rows,
            )
        })
        .collect();

    let mut contrib_all = Vec::with_capacity(per_chrom.len());
    let mut seg_all = Vec::new();
    for (contrib, rows) in per_chrom {
        contrib_all.push(contrib);
        seg_all.extend(rows);
    }
    (contrib_all, seg_all)
}

fn select_evenly_spaced(indices: &[usize], limit: usize) -> Vec<usize> {
    if limit == 0 || indices.len() <= limit {
        return indices.to_vec();
    }
    if limit == 1 {
        return vec![indices[indices.len() / 2]];
    }
    let n = indices.len();
    (0..limit)
        .map(|i| {
            let pos = (i * (n - 1)) / (limit - 1);
            indices[pos]
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn compute_mosaic_data(
    archive: &AncestrySegmentsArchive,
    segs_by_chrom: &[Vec<&AncestrySegment>],
    chrom_lengths_bp: &[u64],
    sample_n_haps: &[u8],
    sample_breakpoints: &[u64],
    n_groups: usize,
    parent_to_group: &[usize],
    config: &PostIbdConfig,
) -> (
    Vec<MosaicSampleInfo>,
    Vec<MosaicSegmentCompact>,
    Vec<PopMosaicChrom>,
) {
    let n_samples = archive.target_sample_ids.len();
    let n_parents = n_groups;

    // Select samples for whole-genome karyotype mosaic.
    let selected_sample_indices: Vec<usize> = if !config.mosaic_samples.is_empty() {
        let id_to_idx: HashMap<&str, usize> = archive
            .target_sample_ids
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let mut chosen = Vec::new();
        let mut seen = HashSet::new();
        for name in &config.mosaic_samples {
            if let Some(&idx) = id_to_idx.get(name.as_str()) {
                if seen.insert(idx) {
                    chosen.push(idx);
                }
            }
        }
        chosen
    } else {
        // Representative selection: sort by total breakpoint count and pick evenly spaced quantiles.
        let mut ranked: Vec<usize> = (0..n_samples).collect();
        ranked.sort_by(|&a, &b| {
            sample_breakpoints[a]
                .cmp(&sample_breakpoints[b])
                .then_with(|| a.cmp(&b))
        });
        select_evenly_spaced(&ranked, config.max_mosaic_samples.max(1))
    };

    let mut sample_to_local: HashMap<u32, u32> =
        HashMap::with_capacity(selected_sample_indices.len());
    let mut mosaic_samples = Vec::with_capacity(selected_sample_indices.len());
    for (local_idx, &s_idx) in selected_sample_indices.iter().enumerate() {
        sample_to_local.insert(s_idx as u32, local_idx as u32);
        mosaic_samples.push(MosaicSampleInfo {
            sample_idx: s_idx as u32,
            sample_id: archive.target_sample_ids[s_idx].clone(),
            n_haps: sample_n_haps[s_idx],
            total_breakpoints: sample_breakpoints[s_idx],
        });
    }

    let target_chrom_set: HashSet<String> = config
        .mosaic_chroms
        .iter()
        .map(|s| normalize_chrom_token(s))
        .collect();

    let mut mosaic_segments = Vec::new();
    for (c, chrom_segs) in segs_by_chrom.iter().enumerate() {
        if !target_chrom_set.is_empty()
            && !target_chrom_set.contains(&normalize_chrom_token(&archive.chrom_names[c]))
        {
            continue;
        }
        for seg in chrom_segs {
            if let Some(&local_idx) = sample_to_local.get(&seg.target_sample) {
                let raw_p = seg.parent_sample as usize;
                let mapped_p = if raw_p < parent_to_group.len() {
                    parent_to_group[raw_p] as u32
                } else {
                    seg.parent_sample
                };
                mosaic_segments.push(MosaicSegmentCompact {
                    chrom_idx: seg.chrom_idx,
                    local_sample_idx: local_idx,
                    hap: seg.hap,
                    start_bp: seg.start_bp,
                    end_bp: seg.end_bp,
                    parent_sample: mapped_p,
                });
            }
        }
    }

    // Build population mosaic heatmaps across selected chromosomes, allocating
    // horizontal bins strictly in proportion to each chromosome's physical length.
    let all_sample_indices: Vec<usize> = (0..n_samples).collect();
    let pop_sample_indices = select_evenly_spaced(&all_sample_indices, config.max_pop_samples);
    let n_pop_samples = pop_sample_indices.len();
    let mut global_to_pop: Vec<Option<usize>> = vec![None; n_samples];
    for (local, &g) in pop_sample_indices.iter().enumerate() {
        global_to_pop[g] = Some(local);
    }
    let pop_sample_u32: Vec<u32> = pop_sample_indices.iter().map(|&i| i as u32).collect();

    let active_chroms: Vec<usize> = (0..archive.chrom_names.len())
        .filter(|&c| {
            let keep_name = target_chrom_set.is_empty()
                || target_chrom_set.contains(&normalize_chrom_token(&archive.chrom_names[c]));
            keep_name && chrom_lengths_bp[c] > 0 && !segs_by_chrom[c].is_empty()
        })
        .collect();
    let total_active_bp: f64 = active_chroms
        .iter()
        .map(|&c| chrom_lengths_bp[c] as f64)
        .sum::<f64>()
        .max(1.0);
    let total_bin_budget = config.pop_mosaic_bins.max(20) as f64;

    let pop_mosaics: Vec<PopMosaicChrom> = active_chroms
        .into_par_iter()
        .filter_map(|c| {
            let max_bp = chrom_lengths_bp[c];
            let chrom_segs = &segs_by_chrom[c];
            if max_bp == 0 || chrom_segs.is_empty() || n_pop_samples == 0 {
                return None;
            }
            let n_bins_out =
                ((total_bin_budget * (max_bp as f64 / total_active_bp)).round() as usize).max(1);
            let step_bp = (max_bp as f64 / n_bins_out as f64).max(1.0);
            let mut parent_matrix = vec![-1i16; n_pop_samples * n_bins_out];

            // Filter & sort segments for selected population samples by `start_bp`
            // so each output heatmap bin sweeps only active overlapping segments.
            let mut pop_segs: Vec<(u64, u64, usize, usize, f32)> = Vec::new();
            for seg in chrom_segs {
                if seg.end_bp < seg.start_bp {
                    continue;
                }
                let s_global = seg.target_sample as usize;
                if s_global >= n_samples {
                    continue;
                }
                let Some(s_local) = global_to_pop[s_global] else {
                    continue;
                };
                let raw_p = seg.parent_sample as usize;
                if raw_p >= parent_to_group.len() {
                    continue;
                }
                let p = parent_to_group[raw_p];
                if p >= n_parents {
                    continue;
                }
                let w = if sample_n_haps[s_global] <= 1 {
                    1.0f32
                } else {
                    0.5f32
                };
                pop_segs.push((seg.start_bp, seg.end_bp, s_local, p, w));
            }
            pop_segs.sort_unstable_by_key(|&(s_bp, _, _, _, _)| s_bp);

            // Accumulate total `dosage * overlap_bp` per `(sample, parent_or_group)` across ALL
            // segments overlapping each output heatmap bin so a parent or ancestry group
            // split into multiple sub-segments reflects its true summed coverage.
            let use_dense = n_pop_samples.saturating_mul(n_parents) <= 2_000_000;
            let mut dense_acc = if use_dense {
                vec![0.0f32; n_pop_samples * n_parents]
            } else {
                Vec::new()
            };
            let mut sparse_acc: HashMap<(usize, usize), f32> = HashMap::new();
            let mut active: Vec<(u64, u64, usize, usize, f32)> = Vec::new();
            let mut seg_cursor = 0usize;

            for ob in 0..n_bins_out {
                let ob_s = (ob as f64 * step_bp).floor() as u64 + 1;
                let ob_e = if ob + 1 == n_bins_out {
                    max_bp
                } else {
                    (((ob + 1) as f64 * step_bp).floor() as u64).min(max_bp)
                };
                if ob_e < ob_s {
                    continue;
                }
                active.retain(|&(_, e_bp, _, _, _)| e_bp >= ob_s);
                while seg_cursor < pop_segs.len() && pop_segs[seg_cursor].0 <= ob_e {
                    let item = pop_segs[seg_cursor];
                    seg_cursor += 1;
                    if item.1 >= ob_s {
                        active.push(item);
                    }
                }
                if active.is_empty() {
                    continue;
                }

                if use_dense {
                    dense_acc.fill(0.0);
                } else {
                    sparse_acc.clear();
                }

                for &(bs, be, s_local, p, dosage_w) in &active {
                    let ov0 = bs.max(ob_s);
                    let ov1 = be.min(ob_e);
                    if ov1 >= ov0 {
                        let w = dosage_w * ((ov1 - ov0 + 1) as f32);
                        if use_dense {
                            dense_acc[s_local * n_parents + p] += w;
                        } else {
                            *sparse_acc.entry((s_local, p)).or_insert(0.0) += w;
                        }
                    }
                }

                if use_dense {
                    for s_local in 0..n_pop_samples {
                        let row_off = s_local * n_parents;
                        let mut best_p = -1i16;
                        let mut best_w = 0.0f32;
                        for p in 0..n_parents {
                            let w = dense_acc[row_off + p];
                            if w > best_w {
                                best_w = w;
                                best_p = p as i16;
                            }
                        }
                        parent_matrix[s_local * n_bins_out + ob] = best_p;
                    }
                } else {
                    let mut best_w = vec![0.0f32; n_pop_samples];
                    for (&(s_local, p), &w) in &sparse_acc {
                        let cell = s_local * n_bins_out + ob;
                        let cur = best_w[s_local];
                        if w > cur || (w == cur && cur > 0.0 && (p as i16) < parent_matrix[cell]) {
                            best_w[s_local] = w;
                            parent_matrix[cell] = p as i16;
                        }
                    }
                }
            }

            Some(PopMosaicChrom {
                chrom_idx: c as u16,
                n_samples: n_pop_samples,
                n_bins: n_bins_out,
                chrom_length_bp: max_bp,
                sample_indices: pop_sample_u32.clone(),
                parent_matrix,
            })
        })
        .collect();

    (mosaic_samples, mosaic_segments, pop_mosaics)
}

fn normalize_chrom_token(s: &str) -> String {
    let trimmed = s.trim().to_ascii_lowercase();
    trimmed.strip_prefix("chr").unwrap_or(&trimmed).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haploid::ancestry::build_union_breakpoint_sparse_matrix;
    use crate::haploid::phasing::SampleHaplotypeLayout;

    fn make_synthetic_archive() -> HaploidSparseArchive {
        // 4 inbred target lines, 4 parents, 1 chromosome of 4 Mb:
        // - Line 0: P0 [1..2_000_000], P1 [2_000_001..4_000_000]  (1 contiguous breakpoint at 2_000_001)
        // - Line 1: P0 [1..4_000_000]                            (0 breakpoints)
        // - Line 2: P0 [1..1_500_000], gap, P2 [2_500_000..4_000_000] (gap -> NOT a contiguous breakpoint)
        // - Line 3: P0 [1..2_000_000], P3 [2_000_001..4_000_000] (1 contiguous breakpoint at 2_000_001)
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 4]);
        let segments = vec![
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1,
                end_bp: 2_000_000,
                start_cm: 0.0,
                end_cm: 2.0,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 100,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 2_000_001,
                end_bp: 4_000_000,
                start_cm: 2.0,
                end_cm: 4.0,
                parent_sample: 1,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 100,
            },
            AncestrySegment {
                target_sample: 1,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1,
                end_bp: 4_000_000,
                start_cm: 0.0,
                end_cm: 4.0,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 200,
            },
            AncestrySegment {
                target_sample: 2,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1,
                end_bp: 1_500_000,
                start_cm: 0.0,
                end_cm: 1.5,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 75,
            },
            AncestrySegment {
                target_sample: 2,
                hap: 0,
                chrom_idx: 0,
                start_bp: 2_500_000,
                end_bp: 4_000_000,
                start_cm: 2.5,
                end_cm: 4.0,
                parent_sample: 2,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 75,
            },
            AncestrySegment {
                target_sample: 3,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1,
                end_bp: 2_000_000,
                start_cm: 0.0,
                end_cm: 2.0,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 100,
            },
            AncestrySegment {
                target_sample: 3,
                hap: 0,
                chrom_idx: 0,
                start_bp: 2_000_001,
                end_bp: 4_000_000,
                start_cm: 2.0,
                end_cm: 4.0,
                parent_sample: 3,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 100,
            },
        ];
        let bins = build_union_breakpoint_sparse_matrix(0, &tgt_layout, 4, &segments);
        HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: vec![
                "L0".to_owned(),
                "L1".to_owned(),
                "L2".to_owned(),
                "L3".to_owned(),
            ],
            reference_sample_ids: vec![
                "P0".to_owned(),
                "P1".to_owned(),
                "P2".to_owned(),
                "P3".to_owned(),
            ],
            chrom_names: vec!["chr1".to_owned()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments,
                chrom_bins: vec![bins],
            },
        }
    }

    #[test]
    fn post_ibd_computes_recomb_segregation_and_mosaic_accurately() {
        let archive = make_synthetic_archive();
        let cfg = PostIbdConfig {
            window_bp: 1_000_000,
            want_recomb: true,
            want_segregation: true,
            want_mosaic: true,
            max_mosaic_samples: 4,
            pop_mosaic_bins: 40,
            max_pop_samples: 4,
            ..PostIbdConfig::default()
        };
        let res = extract_post_ibd_from_archive(&archive, &cfg).unwrap();

        // Breakpoints: only L0 and L3 have contiguous breakpoints at 2_000_001 (window 2).
        // L2 has an uncalled gap [1_500_001..2_499_999] so it must NOT count as a breakpoint.
        let total_bp_events: u64 = res.recomb_windows.iter().map(|r| r.breakpoints).sum();
        assert_eq!(total_bp_events, 2);
        assert_eq!(res.recomb_windows[2].breakpoints, 2);
        assert!(res.recomb_windows[2].callable);
        assert!(res.recomb_windows[2].effective_cm_per_mb.is_some());
        // Cumulative empirical cM must be non-decreasing.
        for pair in res.recomb_windows.windows(2) {
            assert!(pair[1].cum_effective_cm >= pair[0].cum_effective_cm);
        }

        // Segregation in window 0 ([1..1_000_000]): all 4 lines are P0 -> prop[0] == 1.0,
        // descriptive chi2 == 12.0, expected_per_parent == 1.0.
        let w0 = &res.segregation_windows[0];
        assert!((w0.parent_props[0] - 1.0).abs() < 1e-5);
        assert!((w0.chi2 - 12.0).abs() < 1e-3);
        assert!((w0.expected_per_parent - 1.0).abs() < 1e-5);
        assert_eq!(w0.max_parent_idx, 0);

        // Window 1 ([1_000_001..2_000_000]): L2 is uncalled on the second half -> uncalled_frac ~ 0.125.
        let w1 = &res.segregation_windows[1];
        assert!((w1.uncalled_frac - 0.125).abs() < 1e-3);

        // Mosaic: 4 samples selected, pop mosaic has 4 x 40 cells, and L2's gap bin is -1.
        assert_eq!(res.mosaic_samples.len(), 4);
        assert_eq!(res.pop_mosaics.len(), 1);
        let pm = &res.pop_mosaics[0];
        assert_eq!(pm.n_samples, 4);
        assert_eq!(pm.n_bins, 40);
        // L2 is sample index 2; bin 19 (~1.9-2.0 Mb) is inside L2's uncalled gap.
        assert_eq!(pm.parent_matrix[2 * 40 + 19], -1);
        // L0 is sample index 0; bin 5 (~0.5 Mb) is P0, bin 30 (~3.0 Mb) is P1.
        assert_eq!(pm.parent_matrix[0 * 40 + 5], 0);
        assert_eq!(pm.parent_matrix[0 * 40 + 30], 1);
    }

    #[test]
    fn low_coverage_window_breakpoint_does_not_create_spurious_hotspot() {
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let segments = vec![
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1,
                end_bp: 1_000_000,
                start_cm: 0.0,
                end_cm: 1.0,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 50,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1_000_001,
                end_bp: 1_000_002,
                start_cm: 1.0,
                end_cm: 1.0001,
                parent_sample: 1,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 2,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 2_000_001,
                end_bp: 3_000_000,
                start_cm: 2.0,
                end_cm: 3.0,
                parent_sample: 1,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 50,
            },
        ];
        let bins = build_union_breakpoint_sparse_matrix(0, &tgt_layout, 2, &segments);
        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: vec!["L0".to_owned()],
            reference_sample_ids: vec!["P0".to_owned(), "P1".to_owned()],
            chrom_names: vec!["chr1".to_owned()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments,
                chrom_bins: vec![bins],
            },
        };
        let cfg = PostIbdConfig {
            window_bp: 1_000_000,
            want_recomb: true,
            want_segregation: false,
            want_mosaic: false,
            ..PostIbdConfig::default()
        };
        let res = extract_post_ibd_from_archive(&archive, &cfg).unwrap();
        let w1 = &res.recomb_windows[1];
        assert_eq!(w1.breakpoints, 0);
        assert!(
            !w1.callable,
            "2 bp called in 1 Mb window must not be callable"
        );
        assert!(
            w1.effective_cm_per_mb.is_none(),
            "low-coverage window must not produce a 50M cM/Mb hotspot"
        );
        assert_eq!(w1.effective_cm, 0.0);
        assert_eq!(w1.cum_effective_cm, 0.0);
    }

    #[test]
    fn partial_1bp_haplotype_breakpoints_excluded_from_numerator_and_denominator() {
        // 10 inbred haplotypes across 2 Mb (two 1 Mb windows):
        // - L0: fully called P0 across [1..2_000_000] (no breakpoints; gives 10% window coverage)
        // - L1..L8 (8 haplotypes): called [1..1_000_000] P0, then ONLY 1 bp [1_000_001..1_000_001] P1
        //   (each has a contiguous breakpoint at 1_000_001, but only 1 bp called in window 1)
        // - L9: called [1..1_000_000] P0, uncalled in window 1
        // Window 1 has population coverage 10% (from L0), so it is callable, yet its rate MUST
        // be 0.0 cM/Mb (NOT ~800 cM/Mb), because the 8 1-bp tracks are excluded from BOTH
        // numerator and denominator.
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 10]);
        let mut segments = vec![AncestrySegment {
            target_sample: 0,
            hap: 0,
            chrom_idx: 0,
            start_bp: 1,
            end_bp: 2_000_000,
            start_cm: 0.0,
            end_cm: 2.0,
            parent_sample: 0,
            parent_hap: 0,
            confidence: 0.99,
            ambiguous_parents: 1,
            n_markers: 100,
        }];
        for s in 1..=8u32 {
            segments.push(AncestrySegment {
                target_sample: s,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1,
                end_bp: 1_000_000,
                start_cm: 0.0,
                end_cm: 1.0,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 50,
            });
            segments.push(AncestrySegment {
                target_sample: s,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1_000_001,
                end_bp: 1_000_001,
                start_cm: 1.0,
                end_cm: 1.0,
                parent_sample: 1,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 1,
            });
        }
        segments.push(AncestrySegment {
            target_sample: 9,
            hap: 0,
            chrom_idx: 0,
            start_bp: 1,
            end_bp: 1_000_000,
            start_cm: 0.0,
            end_cm: 1.0,
            parent_sample: 0,
            parent_hap: 0,
            confidence: 0.99,
            ambiguous_parents: 1,
            n_markers: 50,
        });

        let bins = build_union_breakpoint_sparse_matrix(0, &tgt_layout, 2, &segments);
        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: (0..10).map(|i| format!("L{i}")).collect(),
            reference_sample_ids: vec!["P0".to_owned(), "P1".to_owned()],
            chrom_names: vec!["chr1".to_owned()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments,
                chrom_bins: vec![bins],
            },
        };
        let cfg = PostIbdConfig {
            window_bp: 1_000_000,
            min_window_coverage: 0.10,
            want_recomb: true,
            want_segregation: false,
            want_mosaic: false,
            ..PostIbdConfig::default()
        };
        let res = extract_post_ibd_from_archive(&archive, &cfg).unwrap();
        let w1 = &res.recomb_windows[1];
        assert!(w1.callable);
        assert_eq!(w1.breakpoints, 0);
        assert!((w1.obs_haps - 1.0).abs() < 1e-6);
        assert_eq!(w1.effective_cm_per_mb, Some(0.0));
        assert_eq!(w1.cum_effective_cm, 0.0);
    }

    #[test]
    fn pop_mosaic_sums_split_parent_overlap_within_same_heatmap_bin() {
        // 1 line, 2 parents, chromosome length 2,000 bp, 20 heatmap bins -> 100 bp/bin.
        // In bin 0 ([1..100]): P0 covers [1..30] (30 bp) + [71..100] (30 bp) = 60 bp total,
        // while P1 covers [31..70] (40 bp). Dominant parent in bin 0 MUST be P0 (60 > 40).
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let segments = vec![
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1,
                end_bp: 30,
                start_cm: 0.0,
                end_cm: 0.01,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 5,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 31,
                end_bp: 70,
                start_cm: 0.01,
                end_cm: 0.02,
                parent_sample: 1,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 5,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 71,
                end_bp: 100,
                start_cm: 0.02,
                end_cm: 0.03,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 5,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 101,
                end_bp: 2_000,
                start_cm: 0.03,
                end_cm: 1.0,
                parent_sample: 1,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 50,
            },
        ];
        let bins = build_union_breakpoint_sparse_matrix(0, &tgt_layout, 2, &segments);
        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: vec!["L0".to_owned()],
            reference_sample_ids: vec!["P0".to_owned(), "P1".to_owned()],
            chrom_names: vec!["chr1".to_owned()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments,
                chrom_bins: vec![bins],
            },
        };
        let cfg = PostIbdConfig {
            window_bp: 1_000,
            want_recomb: false,
            want_segregation: false,
            want_mosaic: true,
            pop_mosaic_bins: 20,
            max_pop_samples: 1,
            ..PostIbdConfig::default()
        };
        let res = extract_post_ibd_from_archive(&archive, &cfg).unwrap();
        let pm = &res.pop_mosaics[0];
        assert_eq!(
            pm.parent_matrix[0], 0,
            "P0 (30 + 30 = 60 bp) must win over P1 (40 bp) in bin 0"
        );
        assert_eq!(pm.parent_matrix[1], 1);
    }

    #[test]
    fn short_1bp_trailing_window_merged_without_spurious_hotspot() {
        // Chromosome length 1_000_001 bp with 1 Mb window: trailing 1 bp is merged
        // into window 0 ([1..1_000_001]), so a breakpoint at 1_000_001 gives
        // ~100 cM/Mb instead of 100,000,000 cM/Mb.
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let segments = vec![
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1,
                end_bp: 1_000_000,
                start_cm: 0.0,
                end_cm: 1.0,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 50,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1_000_001,
                end_bp: 1_000_001,
                start_cm: 1.0,
                end_cm: 1.0,
                parent_sample: 1,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 1,
            },
        ];
        let bins = build_union_breakpoint_sparse_matrix(0, &tgt_layout, 2, &segments);
        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: vec!["L0".to_owned()],
            reference_sample_ids: vec!["P0".to_owned(), "P1".to_owned()],
            chrom_names: vec!["chr1".to_owned()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments,
                chrom_bins: vec![bins],
            },
        };
        let cfg = PostIbdConfig {
            window_bp: 1_000_000,
            want_recomb: true,
            want_segregation: false,
            want_mosaic: false,
            ..PostIbdConfig::default()
        };
        let res = extract_post_ibd_from_archive(&archive, &cfg).unwrap();
        assert_eq!(res.recomb_windows.len(), 1);
        let w0 = &res.recomb_windows[0];
        assert_eq!(w0.start_bp, 1);
        assert_eq!(w0.end_bp, 1_000_001);
        assert!(w0.callable);
        let rate = w0.effective_cm_per_mb.unwrap();
        assert!(rate < 101.0, "expected ~100 cM/Mb, got {rate}");
    }

    #[test]
    fn parent_groups_aggregate_mosaic_bins_and_contributions_accurately() {
        // 1 line, 3 parents (P0->A, P1->A, P2->B), chrom length 2,000 bp, 20 bins (100 bp/bin).
        // Bin 0 ([1..100]): P0 covers [1..30] (30 bp), P2 covers [31..70] (40 bp), P1 covers [71..100] (30 bp).
        // Individually P2 (40 bp) > P0 (30 bp) = P1 (30 bp), but grouped A (30+30=60 bp) > B (40 bp)!
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let segments = vec![
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 1,
                end_bp: 30,
                start_cm: 0.0,
                end_cm: 0.01,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 5,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 31,
                end_bp: 70,
                start_cm: 0.01,
                end_cm: 0.02,
                parent_sample: 2,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 5,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 71,
                end_bp: 100,
                start_cm: 0.02,
                end_cm: 0.03,
                parent_sample: 1,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 5,
            },
            AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: 101,
                end_bp: 2_000,
                start_cm: 0.03,
                end_cm: 1.0,
                parent_sample: 2,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 50,
            },
        ];
        let bins = build_union_breakpoint_sparse_matrix(0, &tgt_layout, 3, &segments);
        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: vec!["L0".to_owned()],
            reference_sample_ids: vec!["P0".to_owned(), "P1".to_owned(), "P2".to_owned()],
            chrom_names: vec!["chr1".to_owned()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments,
                chrom_bins: vec![bins],
            },
        };
        let mut groups = HashMap::new();
        groups.insert("P0".to_owned(), "A".to_owned());
        groups.insert("P1".to_owned(), "A".to_owned());
        groups.insert("P2".to_owned(), "B".to_owned());
        let cfg = PostIbdConfig {
            window_bp: 1_000,
            want_recomb: false,
            want_segregation: true,
            want_mosaic: true,
            pop_mosaic_bins: 20,
            max_pop_samples: 1,
            parent_groups: groups,
            ..PostIbdConfig::default()
        };
        let res = extract_post_ibd_from_archive(&archive, &cfg).unwrap();
        assert!(res.grouped_parents);
        assert_eq!(res.reference_sample_ids, vec!["A", "B"]);
        assert_eq!(res.n_reference_samples, 3);
        let pm = &res.pop_mosaics[0];
        assert_eq!(pm.parent_matrix[0], 0);
        assert_eq!(pm.parent_matrix[1], 1);
        let c0 = &res.parent_contrib_chroms[0];
        assert!((c0.parent_bp[0] - 60.0).abs() < 1e-6);
        assert!((c0.parent_bp[1] - 1940.0).abs() < 1e-6);
    }

    #[test]
    fn parent_groups_rejects_group_name_colliding_with_unmapped_parent() {
        let archive = make_synthetic_archive();
        let mut groups = HashMap::new();
        groups.insert("P0".to_owned(), "P1".to_owned());
        let cfg = PostIbdConfig {
            want_recomb: false,
            want_segregation: true,
            want_mosaic: true,
            parent_groups: groups,
            ..PostIbdConfig::default()
        };
        let err = extract_post_ibd_from_archive(&archive, &cfg)
            .expect_err("mapping P0 -> P1 while P1 is unmapped must be rejected");
        assert!(err.contains("P1"), "unexpected error message: {err}");
    }
}
