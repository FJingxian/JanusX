//! Beagle-style IBS2 segment detection used to exclude close relatives from
//! composite reference panels.

use super::impute::ReferenceMarkerAlignment;
use super::types::{Ibd2Interval, PhaseWindow, PhasedWindow};
use std::collections::{BTreeMap, BTreeSet};

const MAX_MISS_FREQ: f32 = 0.1;
const MIN_MINOR_FREQ: f32 = 0.1;
const MIN_MARKER_COUNT: usize = 50;
const MIN_INTERMARKER_CM: f64 = 0.02_f32 as f64;
const MAX_IBD_GAP_CM: f64 = 4.0;
const MIN_IBS2_CM: f64 = 2.0_f32 as f64;

#[derive(Clone)]
struct SampleCluster {
    samples: Vec<usize>,
    homozygous: bool,
}

#[derive(Clone, Copy, Debug)]
struct PairSegment {
    left: usize,
    right: usize,
    start: usize,
    end: usize,
}

struct SampledIbs2Haplotypes {
    target: Vec<usize>,
    reference: Vec<usize>,
}

pub(crate) fn detect_intervals(
    window: &PhaseWindow,
    cm_positions: &[f64],
    max_intervals: Option<usize>,
    max_scratch_bytes: Option<usize>,
) -> Result<Vec<Ibd2Interval>, String> {
    if window.rows.is_empty() {
        return Ok(Vec::new());
    }
    if cm_positions.len() != window.rows.len() {
        return Err("IBD2 map length differs from phase-window marker count".to_owned());
    }
    if cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("IBD2 genetic positions must be finite and nondecreasing".to_owned());
    }
    let n_samples = window.rows[0].calls.len();
    if n_samples < 2 {
        return Ok(Vec::new());
    }
    if window.rows.iter().any(|row| row.calls.len() != n_samples) {
        return Err("IBD2 phase window has inconsistent sample counts".to_owned());
    }

    let use_marker = ibs2_markers_with_reference(window, None, None, None, None)?;
    detect_intervals_with_marker_flags(
        window,
        cm_positions,
        use_marker,
        max_intervals,
        max_scratch_bytes,
    )
}

fn detect_intervals_with_marker_flags(
    window: &PhaseWindow,
    cm_positions: &[f64],
    mut use_marker: Vec<bool>,
    max_intervals: Option<usize>,
    max_scratch_bytes: Option<usize>,
) -> Result<Vec<Ibd2Interval>, String> {
    if use_marker.len() != window.rows.len() || cm_positions.len() != window.rows.len() {
        return Err("IBD2 marker flags, genetic map, and window differ in length".to_owned());
    }
    let n_samples = window.rows.first().map_or(0, |row| row.calls.len());
    if n_samples < 2 {
        return Ok(Vec::new());
    }
    if window.rows.iter().any(|row| row.calls.len() != n_samples) {
        return Err("IBD2 phase window has inconsistent sample counts".to_owned());
    }
    let steps = ibs2_step_ranges(&mut use_marker, cm_positions);
    let mut segments = Vec::new();
    for (start, end) in steps {
        let marker_indices: Vec<usize> =
            (start..end).filter(|&marker| use_marker[marker]).collect();
        if marker_indices.is_empty() {
            continue;
        }
        let pairs = ibs2_pairs_for_step(
            window,
            &marker_indices,
            n_samples,
            max_scratch_bytes,
            max_intervals,
        )?;
        for (left, right) in pairs {
            let directed_count = segments
                .len()
                .checked_add(1)
                .and_then(|count| count.checked_mul(2))
                .ok_or_else(|| "IBD2 interval count overflow".to_owned())?;
            if max_intervals.is_some_and(|limit| directed_count > limit) {
                return Err(
                    "IBD2 interval storage exceeds --max-memory for the active window; reduce the window size or HMM state counts".to_owned(),
                );
            }
            segments.push(PairSegment {
                left,
                right,
                start,
                end,
            });
        }
    }

    let segments = merge_segments(segments, cm_positions);
    let segments = extend_segments(window, segments);
    let segments = merge_segments(segments, cm_positions);
    let mut intervals = Vec::with_capacity(segments.len().saturating_mul(2));
    for segment in segments {
        let start_cm = cm_positions[segment.start];
        let end_cm = cm_positions[segment.end - 1];
        if end_cm - start_cm < MIN_IBS2_CM {
            continue;
        }
        if max_intervals.is_some_and(|limit| intervals.len().saturating_add(2) > limit) {
            return Err(
                "IBD2 interval storage exceeds --max-memory for the active window; reduce the window size or HMM state counts".to_owned(),
            );
        }
        intervals.push(Ibd2Interval {
            source_sample: segment.left,
            target_sample: segment.right,
            start_marker: segment.start,
            end_marker: segment.end,
        });
        intervals.push(Ibd2Interval {
            source_sample: segment.right,
            target_sample: segment.left,
            start_marker: segment.start,
            end_marker: segment.end,
        });
    }
    intervals.sort_unstable_by_key(|interval| {
        (
            interval.source_sample,
            interval.target_sample,
            interval.start_marker,
        )
    });
    if std::env::var("JANUSX_TRACE_IBD2_TARGET").as_deref() == Ok("35:63") && window.index == 35 {
        eprintln!(
            "JANUSX_IBD2_HEADER\tWINDOW={}\tSAMPLE=63\tNMARKERS={}\tFIRST={}\tLAST={}",
            window.index,
            window.rows.len(),
            window.rows.first().map_or(0, |row| row.meta.pos),
            window.rows.last().map_or(0, |row| row.meta.pos),
        );
        for interval in intervals
            .iter()
            .filter(|interval| interval.source_sample == 63 || interval.target_sample == 63)
        {
            eprintln!(
                "JANUSX_IBD2\tSOURCE={}\tTARGET={}\tSTART={}\tEND={}\tSTARTPOS={}\tENDPOS={}",
                interval.source_sample,
                interval.target_sample,
                interval.start_marker,
                interval.end_marker,
                window.rows[interval.start_marker].meta.pos,
                window.rows[interval.end_marker - 1].meta.pos,
            );
        }
    }
    Ok(intervals)
}

/// Detects IBS2 intervals after projecting a full window to a marker set.
/// The returned marker coordinates are local to `marker_indices`, matching
/// Beagle's Stage1 `Ibs2`, which is constructed from its restricted Stage1 GT.
pub(crate) fn detect_projected_intervals(
    window: &PhaseWindow,
    marker_indices: &[usize],
    max_intervals: Option<usize>,
    max_scratch_bytes: Option<usize>,
) -> Result<Vec<Ibd2Interval>, String> {
    detect_projected_intervals_with_reference(
        window,
        marker_indices,
        None,
        None,
        None,
        max_intervals,
        max_scratch_bytes,
    )
}

pub(crate) fn detect_projected_intervals_with_reference(
    window: &PhaseWindow,
    marker_indices: &[usize],
    reference: Option<&PhasedWindow>,
    reference_alignment: Option<&ReferenceMarkerAlignment>,
    sample_seed: Option<u64>,
    max_intervals: Option<usize>,
    max_scratch_bytes: Option<usize>,
) -> Result<Vec<Ibd2Interval>, String> {
    detect_projected_intervals_with_reference_map(
        window,
        marker_indices,
        &window.cm_positions,
        reference,
        reference_alignment,
        sample_seed,
        max_intervals,
        max_scratch_bytes,
    )
}

pub(crate) fn detect_projected_intervals_with_reference_map(
    window: &PhaseWindow,
    marker_indices: &[usize],
    cm_positions: &[f64],
    reference: Option<&PhasedWindow>,
    reference_alignment: Option<&ReferenceMarkerAlignment>,
    sample_seed: Option<u64>,
    max_intervals: Option<usize>,
    max_scratch_bytes: Option<usize>,
) -> Result<Vec<Ibd2Interval>, String> {
    if marker_indices.is_empty()
        || marker_indices
            .iter()
            .any(|&marker| marker >= window.rows.len())
        || marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
        || cm_positions.len() != window.rows.len()
    {
        return Err(
            "IBD2 marker projection must be nonempty, unique, ordered, and map-aligned".to_owned(),
        );
    }
    if reference.is_some() != reference_alignment.is_some() {
        return Err("IBD2 reference markers require their alignment".to_owned());
    }
    let projected = PhaseWindow {
        chrom: window.chrom.clone(),
        index: window.index,
        rows: marker_indices
            .iter()
            .map(|&marker| window.rows[marker].clone())
            .collect(),
        cm_positions: marker_indices
            .iter()
            .map(|&marker| cm_positions[marker])
            .collect(),
        incoming_overlap_len: marker_indices
            .partition_point(|&marker| marker < window.incoming_overlap_len),
        emit_start: marker_indices.partition_point(|&marker| marker < window.emit_start),
        emit_end: marker_indices.partition_point(|&marker| marker < window.emit_end),
    };
    let use_marker = ibs2_markers_with_reference(
        window,
        reference,
        reference_alignment,
        Some(marker_indices),
        sample_seed,
    )?;
    detect_intervals_with_marker_flags(
        &projected,
        &projected.cm_positions,
        use_marker,
        max_intervals,
        max_scratch_bytes,
    )
}

pub(crate) fn can_contribute(
    source: usize,
    target: usize,
    variant: usize,
    intervals: &[Ibd2Interval],
) -> bool {
    // Beagle's Ibs2.areIbs2() returns true for a sample paired with itself,
    // which prevents its second haplotype from being selected as a donor.
    if source == target {
        return false;
    }
    let index = intervals.partition_point(|interval| {
        (
            interval.source_sample,
            interval.target_sample,
            interval.start_marker,
        ) <= (source, target, variant)
    });
    index
        .checked_sub(1)
        .and_then(|index| intervals.get(index))
        .map_or(true, |interval| {
            interval.source_sample != source
                || interval.target_sample != target
                || variant >= interval.end_marker
        })
}

pub(crate) fn can_contribute_segment(
    source: usize,
    target: usize,
    start_marker: usize,
    end_marker: usize,
    intervals: &[Ibd2Interval],
) -> bool {
    if source == target || start_marker >= end_marker {
        return false;
    }
    let pair = (source, target);
    let mut index = intervals
        .partition_point(|interval| (interval.source_sample, interval.target_sample) < pair);
    while let Some(interval) = intervals.get(index) {
        if (interval.source_sample, interval.target_sample) != pair {
            break;
        }
        if interval.start_marker >= end_marker {
            break;
        }
        if interval.end_marker > start_marker {
            return false;
        }
        index += 1;
    }
    true
}

/// Mirrors Beagle's Ibs2Markers.useMarker(): MAF is computed over observed
/// haplotypes, and a marker is excluded when more than 10% of all haplotypes
/// are missing.
fn ibs2_markers_with_reference(
    window: &PhaseWindow,
    reference: Option<&PhasedWindow>,
    reference_alignment: Option<&ReferenceMarkerAlignment>,
    marker_indices: Option<&[usize]>,
    sample_seed: Option<u64>,
) -> Result<Vec<bool>, String> {
    let n_samples = window.rows.first().map_or(0, |row| row.calls.len());
    let n_haps = n_samples
        .checked_mul(2)
        .ok_or_else(|| "IBD2 haplotype count overflow".to_owned())?;
    let max_missing = (MAX_MISS_FREQ * n_haps as f32).ceil() as usize;
    if reference.is_some() != reference_alignment.is_some() {
        return Err("IBD2 reference markers require their alignment".to_owned());
    }
    if let Some(alignment) = reference_alignment {
        if alignment.len() != window.rows.len() {
            return Err("IBD2 reference alignment differs from the target window".to_owned());
        }
    }
    let indices = marker_indices
        .map(|indices| indices.to_vec())
        .unwrap_or_else(|| (0..window.rows.len()).collect());
    let sampled_haplotypes = sample_seed
        .map(|seed| {
            sampled_ibs2_haplotypes(
                n_haps,
                reference.map_or(0, |reference| {
                    reference.rows.first().map_or(0, |row| row.calls.len()) * 2
                }),
                seed,
            )
        })
        .transpose()?;
    let mut use_marker = Vec::with_capacity(indices.len());
    for marker in indices {
        let row = window
            .rows
            .get(marker)
            .ok_or_else(|| format!("IBD2 marker index {marker} exceeds the target window"))?;
        let mut counts = vec![0usize; row.meta.allele_count()];
        let missing = row
            .calls
            .iter()
            .flat_map(|call| call.alleles.into_iter())
            .filter(|allele| allele.is_none())
            .count();
        if let Some(sampled) = sampled_haplotypes.as_ref() {
            for &haplotype in &sampled.target {
                let call = row.calls.get(haplotype / 2).expect("validated target hap");
                if let Some(allele) = call.alleles[haplotype & 1] {
                    let Some(count) = counts.get_mut(usize::from(allele)) else {
                        return Err(format!(
                            "IBD2 allele index {allele} exceeds the allele count at {}:{}",
                            row.meta.chrom, row.meta.pos
                        ));
                    };
                    *count += 1;
                }
            }
        } else {
            for allele in row.calls.iter().flat_map(|call| call.alleles.into_iter()) {
                if let Some(allele) = allele {
                    let Some(count) = counts.get_mut(usize::from(allele)) else {
                        return Err(format!(
                            "IBD2 allele index {allele} exceeds the allele count at {}:{}",
                            row.meta.chrom, row.meta.pos
                        ));
                    };
                    *count += 1;
                }
            }
        }
        if let (Some(reference), Some(alignment)) = (reference, reference_alignment) {
            if let Some(reference_row) = alignment
                .reference_row(marker)
                .map(|index| &reference.rows[index])
            {
                let swap = alignment.reference_swap(marker);
                if let Some(sampled) = sampled_haplotypes.as_ref() {
                    for &haplotype in &sampled.reference {
                        let call = reference_row
                            .calls
                            .get(haplotype / 2)
                            .expect("validated reference hap");
                        if let Some(allele) = call.alleles[haplotype & 1] {
                            let allele = if swap { u16::from(allele == 0) } else { allele };
                            let Some(count) = counts.get_mut(usize::from(allele)) else {
                                return Err(format!(
                                    "IBD2 reference allele {allele} exceeds the target allele count at {}:{}",
                                    row.meta.chrom, row.meta.pos
                                ));
                            };
                            *count += 1;
                        }
                    }
                } else {
                    for call in reference_row.calls.iter() {
                        for allele in call.alleles.into_iter().flatten() {
                            let allele = if swap { u16::from(allele == 0) } else { allele };
                            let Some(count) = counts.get_mut(usize::from(allele)) else {
                                return Err(format!(
                                    "IBD2 reference allele {allele} exceeds the target allele count at {}:{}",
                                    row.meta.chrom, row.meta.pos
                                ));
                            };
                            *count += 1;
                        }
                    }
                }
            }
        }
        let observed: usize = counts.iter().sum();
        counts.sort_unstable();
        let minor_frequency = if observed == 0 {
            0.0_f32
        } else {
            (counts[counts.len().saturating_sub(2)] as f64 / observed as f64) as f32
        };
        use_marker.push(minor_frequency >= MIN_MINOR_FREQ && missing <= max_missing);
    }
    Ok(use_marker)
}

fn sampled_ibs2_haplotypes(
    target_haplotypes: usize,
    reference_haplotypes: usize,
    seed: u64,
) -> Result<SampledIbs2Haplotypes, String> {
    const MAF_SAMPLE_HAPLOTYPES: usize = 10_000;
    if target_haplotypes == 0 {
        return Err("IBS2 MAF sampling requires target haplotypes".to_owned());
    }
    let mut random = super::pbwt::JavaRandom::new(seed);
    let target_count = target_haplotypes.min(MAF_SAMPLE_HAPLOTYPES);
    let mut target = (0..target_haplotypes).collect::<Vec<_>>();
    shuffle_prefix(&mut target, target_count, &mut random);
    target.truncate(target_count);

    let reference_count =
        reference_haplotypes.min(MAF_SAMPLE_HAPLOTYPES.saturating_sub(target_count));
    let mut reference = (0..reference_haplotypes).collect::<Vec<_>>();
    shuffle_prefix(&mut reference, reference_count, &mut random);
    reference.truncate(reference_count);
    Ok(SampledIbs2Haplotypes { target, reference })
}

fn shuffle_prefix<T>(values: &mut [T], count: usize, random: &mut super::pbwt::JavaRandom) {
    debug_assert!(count <= values.len());
    for index in 0..count {
        let offset = random.next_int(values.len() - index);
        values.swap(index, index + offset);
    }
}

/// Mirrors Beagle's minimum-50-marker, minimum-0.02-cM step construction.
/// The boundary marker is the first marker after the 50th selected marker;
/// it is included in the following step's range, just as in Ibs2Markers.
fn ibs2_step_ranges(use_marker: &mut [bool], cm_positions: &[f64]) -> Vec<(usize, usize)> {
    if use_marker.is_empty() {
        return Vec::new();
    }
    let mut starts = Vec::new();
    let mut last_start = 0usize;
    let mut next_start = next_ibs2_step_start(use_marker, cm_positions, last_start);
    while next_start < use_marker.len() {
        starts.push(last_start);
        last_start = next_start;
        next_start = next_ibs2_step_start(use_marker, cm_positions, next_start);
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, &start)| {
            let end = starts.get(index + 1).copied().unwrap_or(use_marker.len());
            (start, end)
        })
        .collect()
}

fn next_ibs2_step_start(use_marker: &mut [bool], cm_positions: &[f64], start: usize) -> usize {
    let min_intermarker_cm = MIN_INTERMARKER_CM;
    let mut cm_position = cm_positions[start];
    let mut min_cm_position = cm_position + min_intermarker_cm;
    let mut next = start + 1;
    let mut marker_count = 0usize;
    while next < use_marker.len() && marker_count < MIN_MARKER_COUNT {
        if use_marker[next] {
            cm_position = cm_positions[next];
            if cm_position < min_cm_position {
                use_marker[next] = false;
            } else {
                marker_count += 1;
                min_cm_position = cm_position + min_intermarker_cm;
            }
        }
        next += 1;
    }
    next
}

fn ibs2_pairs_for_step(
    window: &PhaseWindow,
    markers: &[usize],
    n_samples: usize,
    max_scratch_bytes: Option<usize>,
    max_intervals: Option<usize>,
) -> Result<Vec<(usize, usize)>, String> {
    let mut missing_counts = vec![0usize; n_samples];
    for &marker in markers {
        for (sample, call) in window.rows[marker].calls.iter().enumerate() {
            if !call.is_complete() {
                missing_counts[sample] += 1;
            }
        }
    }
    let max_step_missing = (MAX_MISS_FREQ * markers.len() as f32).floor() as usize;
    let initial_samples: Vec<usize> = missing_counts
        .iter()
        .enumerate()
        .filter_map(|(sample, &count)| (count <= max_step_missing).then_some(sample))
        .collect();
    if initial_samples.len() < 2 {
        return Ok(Vec::new());
    }
    check_cluster_scratch(1, initial_samples.len(), n_samples, max_scratch_bytes)?;
    let mut clusters = vec![SampleCluster {
        samples: initial_samples,
        homozygous: true,
    }];

    for &marker in markers {
        let mut next_clusters = Vec::new();
        let mut next_members = 0usize;
        let current_members: usize = clusters.iter().map(|cluster| cluster.samples.len()).sum();
        for parent in &clusters {
            let mut missing = Vec::new();
            let mut genotype_groups = BTreeMap::<u32, (Vec<usize>, bool)>::new();
            for &sample in &parent.samples {
                let call = window.rows[marker]
                    .calls
                    .get(sample)
                    .expect("validated sample index");
                match call.alleles {
                    [Some(a), Some(b)] => {
                        let genotype = canonical_genotype(a, b);
                        let group = genotype_groups.entry(genotype).or_default();
                        group.0.push(sample);
                        group.1 = a == b;
                    }
                    _ => missing.push(sample),
                }
            }
            for (_, (mut samples, is_homozygous_genotype)) in genotype_groups {
                samples.extend_from_slice(&missing);
                samples.sort_unstable();
                samples.dedup();
                if samples.len() < 2 {
                    continue;
                }
                next_members = next_members
                    .checked_add(samples.len())
                    .ok_or_else(|| "IBD2 cluster size overflow".to_owned())?;
                check_cluster_scratch(
                    clusters.len().saturating_add(next_clusters.len() + 1),
                    current_members.saturating_add(next_members),
                    n_samples,
                    max_scratch_bytes,
                )?;
                next_clusters.push(SampleCluster {
                    samples,
                    homozygous: parent.homozygous && is_homozygous_genotype,
                });
            }
        }
        clusters = next_clusters;
        if clusters.is_empty() {
            return Ok(Vec::new());
        }
    }

    let mut pairs = BTreeSet::new();
    for cluster in clusters.iter().filter(|cluster| !cluster.homozygous) {
        for left_index in 0..cluster.samples.len() {
            for right_index in left_index + 1..cluster.samples.len() {
                pairs.insert((cluster.samples[left_index], cluster.samples[right_index]));
                if max_intervals.is_some_and(|limit| pairs.len().saturating_mul(2) > limit) {
                    return Err(
                        "IBD2 interval storage exceeds --max-memory for the active window; reduce the window size or HMM state counts".to_owned(),
                    );
                }
            }
        }
    }
    Ok(pairs.into_iter().collect())
}

fn check_cluster_scratch(
    cluster_count: usize,
    member_count: usize,
    n_samples: usize,
    max_scratch_bytes: Option<usize>,
) -> Result<(), String> {
    let estimated = cluster_count
        .checked_mul(std::mem::size_of::<SampleCluster>())
        .and_then(|bytes| {
            member_count
                .checked_mul(std::mem::size_of::<usize>())
                .and_then(|member_bytes| bytes.checked_add(member_bytes))
        })
        .and_then(|bytes| {
            n_samples
                .checked_mul(32)
                .and_then(|sample_bytes| bytes.checked_add(sample_bytes))
        })
        .ok_or_else(|| "IBD2 scratch-memory estimate overflow".to_owned())?;
    if max_scratch_bytes.is_some_and(|limit| estimated > limit) {
        return Err(
            "IBD2 candidate search exceeds --max-memory scratch allowance for the active window; reduce the window size or HMM state counts".to_owned(),
        );
    }
    Ok(())
}

fn merge_segments(mut segments: Vec<PairSegment>, cm_positions: &[f64]) -> Vec<PairSegment> {
    segments
        .sort_unstable_by_key(|segment| (segment.left, segment.right, segment.start, segment.end));
    let mut merged: Vec<PairSegment> = Vec::with_capacity(segments.len());
    for segment in segments {
        if let Some(previous) = merged.last_mut() {
            if previous.left == segment.left && previous.right == segment.right {
                let gap_cm = cm_positions[segment.start] - cm_positions[previous.end - 1];
                if gap_cm <= MAX_IBD_GAP_CM {
                    previous.start = previous.start.min(segment.start);
                    previous.end = previous.end.max(segment.end);
                    continue;
                }
            }
        }
        merged.push(segment);
    }
    merged
}

fn extend_segments(window: &PhaseWindow, segments: Vec<PairSegment>) -> Vec<PairSegment> {
    let n_markers = window.rows.len();
    segments
        .into_iter()
        .map(|mut segment| {
            while segment.start > 0
                && ibs2_at(window, segment.left, segment.right, segment.start - 1)
            {
                segment.start -= 1;
            }
            while segment.end < n_markers
                && ibs2_at(window, segment.left, segment.right, segment.end)
            {
                segment.end += 1;
            }
            segment
        })
        .collect()
}

fn ibs2_at(window: &PhaseWindow, left: usize, right: usize, marker: usize) -> bool {
    let left = window.rows[marker]
        .calls
        .get(left)
        .expect("sample index")
        .alleles;
    let right = window.rows[marker]
        .calls
        .get(right)
        .expect("sample index")
        .alleles;
    phase_consistent(left[0], right[0]) && phase_consistent(left[1], right[1])
        || phase_consistent(left[0], right[1]) && phase_consistent(left[1], right[0])
}

fn phase_consistent(left: Option<u16>, right: Option<u16>) -> bool {
    left.is_none() || right.is_none() || left == right
}

fn canonical_genotype(a: u16, b: u16) -> u32 {
    let (low, high) = if a <= b { (a, b) } else { (b, a) };
    (u32::from(low) << 16) | u32::from(high)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp::types::{
        DiploidCall, DiploidCalls, PhasedRow, PhasedWindow, VariantMeta, VariantRow,
    };

    fn input_with_shared_heterozygotes(n_het_samples: usize) -> (PhaseWindow, Vec<f64>) {
        let n_markers = 102;
        let n_samples = 64;
        let rows = (0..n_markers)
            .map(|marker| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("m{marker}"),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_iter((0..n_samples).map(|sample| {
                    let alleles = if sample < n_het_samples {
                        [Some(0), Some(1)]
                    } else {
                        [Some(0), Some(0)]
                    };
                    DiploidCall {
                        alleles,
                        phased: false,
                    }
                })),
            })
            .collect();
        let cm_positions: Vec<f64> = (0..n_markers).map(|marker| marker as f64 * 0.02).collect();
        (
            PhaseWindow {
                chrom: "1".to_owned(),
                index: 0,
                rows,
                cm_positions: cm_positions.clone(),
                incoming_overlap_len: 0,
                emit_start: 0,
                emit_end: n_markers,
            },
            cm_positions,
        )
    }

    fn window_with_stage1_coordinate_mismatch() -> (PhaseWindow, Vec<usize>) {
        let n_markers = 190;
        let n_samples = 64;
        let rows = (0..n_markers)
            .map(|marker| {
                let calls = (0..n_samples)
                    .map(|sample| {
                        let alleles = if marker < 141 {
                            if sample < 13 {
                                [Some(0), Some(1)]
                            } else {
                                [Some(0), Some(0)]
                            }
                        } else if sample == 1 {
                            [Some(1), Some(1)]
                        } else if sample == 0 || (2..14).contains(&sample) {
                            [Some(0), Some(1)]
                        } else {
                            [Some(0), Some(0)]
                        };
                        DiploidCall {
                            alleles,
                            phased: false,
                        }
                    })
                    .collect::<Vec<_>>();
                VariantRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("m{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(calls),
                }
            })
            .collect();
        let cm_positions: Vec<f64> = (0..n_markers).map(|marker| marker as f64 * 0.02).collect();
        let stage1_markers = (0..110).chain(141..n_markers).collect();
        (
            PhaseWindow {
                chrom: "1".to_owned(),
                index: 0,
                rows,
                cm_positions,
                incoming_overlap_len: 0,
                emit_start: 0,
                emit_end: n_markers,
            },
            stage1_markers,
        )
    }

    #[test]
    fn beagle_maf_threshold_excludes_shared_rare_heterozygous_segments() {
        let (window, cm_positions) = input_with_shared_heterozygotes(7);
        let intervals = detect_intervals(&window, &cm_positions, None, None).unwrap();

        assert!(!intervals
            .iter()
            .any(|interval| { (interval.source_sample, interval.target_sample) == (0, 1) }));
    }

    #[test]
    fn stage1_ibs2_maf_uses_target_and_reference_haplotypes() {
        let (window, _) = input_with_shared_heterozygotes(1);
        let reference = PhasedWindow {
            chrom: window.chrom.clone(),
            index: window.index,
            rows: window
                .rows
                .iter()
                .map(|row| PhasedRow {
                    meta: row.meta.clone(),
                    calls: DiploidCalls::from_iter((0..10).map(|_| DiploidCall {
                        alleles: [Some(1), Some(1)],
                        phased: true,
                    })),
                    dosage: None,
                    ap1: None,
                    ap2: None,
                    gp: None,
                    info_af: None,
                    info_dr2: None,
                    imputed: false,
                })
                .collect(),
            emit_start: 0,
            emit_end: window.rows.len(),
        };
        let alignment = ReferenceMarkerAlignment::new(&window.rows, &reference.rows).unwrap();
        let marker_indices = [0usize];

        assert_eq!(
            ibs2_markers_with_reference(&window, None, None, Some(&marker_indices), None).unwrap(),
            vec![false]
        );
        assert_eq!(
            ibs2_markers_with_reference(
                &window,
                Some(&reference),
                Some(&alignment),
                Some(&marker_indices),
                None,
            )
            .unwrap(),
            vec![true]
        );
    }

    #[test]
    fn stage1_ibs2_maf_samples_at_most_ten_thousand_target_first() {
        let target_first = sampled_ibs2_haplotypes(15_000, 1_000, 20261108).unwrap();
        assert_eq!(target_first.target.len(), 10_000);
        assert!(target_first.reference.is_empty());

        let mixed = sampled_ibs2_haplotypes(100, 12_000, 20261108).unwrap();
        assert_eq!(mixed.target.len(), 100);
        assert_eq!(mixed.reference.len(), 9_900);
        let mut unique = mixed.reference.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), mixed.reference.len());

        let repeated = sampled_ibs2_haplotypes(100, 12_000, 20261108).unwrap();
        assert_eq!(mixed.target, repeated.target);
        assert_eq!(mixed.reference, repeated.reference);
        let different = sampled_ibs2_haplotypes(100, 12_000, 20261109).unwrap();
        assert_ne!(mixed.target, different.target);
    }

    #[test]
    fn beagle_retains_shared_heterozygous_segments_but_drops_homozygous_sets() {
        let (window, cm_positions) = input_with_shared_heterozygotes(13);
        let intervals = detect_intervals(&window, &cm_positions, None, None).unwrap();

        assert!(intervals.iter().any(|interval| {
            interval.source_sample == 0
                && interval.target_sample == 1
                && interval.start_marker == 0
                && interval.end_marker == window.rows.len()
        }));
        assert!(!intervals
            .iter()
            .any(|interval| { (interval.source_sample, interval.target_sample) == (13, 14) }));
    }

    #[test]
    fn partially_missing_calls_are_wildcards_when_extending_ibs2() {
        let (mut window, cm_positions) = input_with_shared_heterozygotes(13);
        window.rows[50]
            .calls
            .set_call(
                1,
                DiploidCall {
                    alleles: [Some(0), None],
                    phased: false,
                },
            )
            .unwrap();
        let intervals = detect_intervals(&window, &cm_positions, None, None).unwrap();

        assert!(intervals.iter().any(|interval| {
            interval.source_sample == 0
                && interval.target_sample == 1
                && interval.start_marker == 0
                && interval.end_marker == window.rows.len()
        }));
    }

    #[test]
    fn beagle_treats_a_sample_as_ibs2_with_itself_even_without_segments() {
        assert!(!can_contribute(4, 4, 12, &[]));
        assert!(!can_contribute_segment(4, 4, 10, 15, &[]));
    }

    #[test]
    fn stage1_local_pbwt_range_does_not_use_full_window_ibs2_coordinates() {
        let (window, stage1_markers) = window_with_stage1_coordinate_mismatch();
        let full_intervals = detect_intervals(&window, &window.cm_positions, None, None).unwrap();
        assert!(full_intervals.iter().any(|interval| {
            interval.source_sample == 0
                && interval.target_sample == 1
                && interval.start_marker == 0
                && interval.end_marker >= 141
        }));

        // Stage1 keeps the first 110 markers and then markers after the full
        // window's IBS2 segment. Its PBWT step [134, 138) is local to this
        // projection and must not be blocked by the full-window [0, 141) hit.
        assert!(!can_contribute_segment(0, 1, 134, 138, &full_intervals));
        let stage1_intervals =
            detect_projected_intervals(&window, &stage1_markers, None, None).unwrap();
        assert!(stage1_intervals.iter().any(|interval| {
            interval.source_sample == 0
                && interval.target_sample == 1
                && interval.start_marker == 0
                && interval.end_marker <= 110
        }));
        assert!(can_contribute_segment(0, 1, 134, 138, &stage1_intervals));
    }
}
