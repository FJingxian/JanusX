//! Reference matching and window posterior construction.

use super::composite;
use super::hmm;
use super::types::{
    validate_phased_reference_calls, CompositePanel, DiploidCall, DiploidCalls, HMMConfig,
    ImputeConfig, ImputedWindow, PackedHaplotypes, PhaseWindow, PhasedRow, PhasedWindow,
    VariantMeta, VariantRow,
};
use rayon::prelude::*;
use rayon::ThreadPool;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

#[inline]
fn hard_call_alt(p_alt: f32) -> u16 {
    // With normalized biallelic probabilities this matches Beagle's strict
    // maxIndex scan: REF wins an exact tie.
    (p_alt > 0.5) as u16
}

/// One sorted-window alignment shared by reference phasing and low-frequency
/// marker classification. Reference row ids and allele swaps are retained so
/// callers do not rebuild the same target/reference merge more than once.
#[derive(Debug, Clone)]
pub(crate) struct ReferenceMarkerAlignment {
    reference_rows: Vec<Option<usize>>,
    reference_swaps: Vec<bool>,
}

impl ReferenceMarkerAlignment {
    pub(crate) fn new(target: &[VariantRow], reference: &[PhasedRow]) -> Result<Self, String> {
        validate_sorted_unique(target.iter().map(|row| &row.meta), "target")?;
        validate_sorted_unique(reference.iter().map(|row| &row.meta), "reference")?;
        Self::build_impl(target, reference.len(), |index| &reference[index].meta)
    }

    pub(crate) fn new_from_metas(
        target: &[VariantRow],
        reference: &[VariantMeta],
    ) -> Result<Self, String> {
        validate_sorted_unique(target.iter().map(|row| &row.meta), "target")?;
        validate_sorted_unique(reference.iter(), "reference")?;
        Self::build_impl(target, reference.len(), |index| &reference[index])
    }

    fn build_impl<'a>(
        target: &[VariantRow],
        reference_len: usize,
        reference_meta: impl Fn(usize) -> &'a VariantMeta,
    ) -> Result<Self, String> {
        let mut reference_index = 0usize;
        let mut reference_rows = Vec::with_capacity(target.len());
        let mut reference_swaps = Vec::with_capacity(target.len());
        for target_row in target {
            while reference_index < reference_len
                && compare_position(reference_meta(reference_index), &target_row.meta)
                    == Ordering::Less
            {
                reference_index += 1;
            }
            let matched_meta = (reference_index < reference_len)
                .then(|| reference_meta(reference_index))
                .filter(|ref_meta| compare_position(ref_meta, &target_row.meta) == Ordering::Equal);
            reference_rows.push(matched_meta.map(|_| reference_index));
            reference_swaps.push(
                matched_meta
                    .map(|ref_meta| harmonization(&target_row.meta, ref_meta))
                    .transpose()?
                    .unwrap_or(false),
            );
        }
        Ok(Self {
            reference_rows,
            reference_swaps,
        })
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.reference_rows.len()
    }

    #[inline(always)]
    pub(crate) fn reference_row(&self, marker: usize) -> Option<usize> {
        self.reference_rows[marker]
    }

    #[inline(always)]
    pub(crate) fn reference_swap(&self, marker: usize) -> bool {
        self.reference_swaps[marker]
    }

    pub(crate) fn reference_present(&self) -> impl Iterator<Item = bool> + '_ {
        self.reference_rows.iter().map(Option::is_some)
    }
}

#[cfg(test)]
pub(crate) fn low_frequency_markers(window: &PhaseWindow) -> Result<Vec<bool>, String> {
    low_frequency_markers_with_reference(window, None)
}

#[cfg(test)]
pub(crate) fn low_frequency_markers_with_reference(
    window: &PhaseWindow,
    reference: Option<&PhasedWindow>,
) -> Result<Vec<bool>, String> {
    let alignment = reference
        .map(|reference| ReferenceMarkerAlignment::new(&window.rows, &reference.rows))
        .transpose()?;
    low_frequency_markers_with_alignment(window, reference, alignment.as_ref())
}

#[cfg(test)]
pub(crate) fn low_frequency_markers_with_alignment(
    window: &PhaseWindow,
    reference: Option<&PhasedWindow>,
    alignment: Option<&ReferenceMarkerAlignment>,
) -> Result<Vec<bool>, String> {
    low_frequency_marker_data_with_alignment(window, reference, alignment)
        .map(|(marker_flags, _)| marker_flags)
}

#[cfg(test)]
pub(crate) fn low_frequency_allele_flags_with_alignment(
    window: &PhaseWindow,
    reference: Option<&PhasedWindow>,
    alignment: Option<&ReferenceMarkerAlignment>,
) -> Result<Vec<Vec<bool>>, String> {
    low_frequency_marker_data_with_alignment(window, reference, alignment)
        .map(|(_, allele_flags)| allele_flags)
}

pub(crate) fn low_frequency_marker_data_with_alignment(
    window: &PhaseWindow,
    reference: Option<&PhasedWindow>,
    alignment: Option<&ReferenceMarkerAlignment>,
) -> Result<(Vec<bool>, Vec<Vec<bool>>), String> {
    low_frequency_marker_data_with_alignment_ref(
        window,
        reference.map(super::reference_store::PhaseReference::Phased),
        alignment,
    )
}

pub(crate) fn low_frequency_marker_data_with_alignment_ref(
    window: &PhaseWindow,
    reference: Option<super::reference_store::PhaseReference<'_>>,
    alignment: Option<&ReferenceMarkerAlignment>,
) -> Result<(Vec<bool>, Vec<Vec<bool>>), String> {
    if reference.is_some() != alignment.is_some() {
        return Err("reference marker alignment must accompany the reference window".to_owned());
    }
    if let Some(alignment) = alignment {
        if alignment.len() != window.rows.len() {
            return Err("reference marker alignment does not match the target window".to_owned());
        }
    }
    let target_sample_count = window.rows.first().map_or(0, |row| row.calls.len());
    let reference_sample_count = reference.map_or(0, |reference| reference.n_samples());
    let max_carriers =
        beagle_rare_carrier_limit(target_sample_count.saturating_add(reference_sample_count));
    let mut low_frequency = Vec::with_capacity(window.rows.len());
    let mut allele_flags = Vec::with_capacity(window.rows.len());
    for (marker, row) in window.rows.iter().enumerate() {
        let mut carrier_counts = vec![0usize; row.meta.allele_count()];
        for call in row.calls.iter() {
            add_sample_carriers(&mut carrier_counts, call.alleles, max_carriers);
        }
        if let (Some(reference), Some(alignment)) = (reference, alignment) {
            if let Some(reference_index) = alignment.reference_row(marker) {
                let swap = alignment.reference_swap(marker);
                match reference {
                    super::reference_store::PhaseReference::Phased(phased_ref) => {
                        let reference_row = &phased_ref.rows[reference_index];
                        for call in reference_row.calls.iter() {
                            let alleles = if swap {
                                call.alleles.map(|allele| allele.map(swap_biallelic))
                            } else {
                                call.alleles
                            };
                            add_sample_carriers(&mut carrier_counts, alleles, max_carriers);
                        }
                    }
                    super::reference_store::PhaseReference::Compact(compact_ref) => {
                        let words = compact_ref.resident_marker_words(reference_index)?;
                        let swap_mask = u16::from(swap);
                        for sample in 0..compact_ref.n_samples {
                            let hap0 = sample * 2;
                            let pair = (words[hap0 >> 6] >> (hap0 & 63)) & 0x3;
                            let a0 = ((pair & 1) as u16) ^ swap_mask;
                            let a1 = (((pair >> 1) & 1) as u16) ^ swap_mask;
                            add_sample_carriers(
                                &mut carrier_counts,
                                [Some(a0), Some(a1)],
                                max_carriers,
                            );
                        }
                    }
                }
            }
        }
        low_frequency.push(
            carrier_counts
                .iter()
                .filter(|&&count| count > max_carriers)
                .count()
                <= 1,
        );
        allele_flags.push(
            carrier_counts
                .into_iter()
                .map(|count| count <= max_carriers)
                .collect(),
        );
    }
    Ok((low_frequency, allele_flags))
}

pub(crate) fn beagle_rare_carrier_limit(sample_count: usize) -> usize {
    ((sample_count as f32 * 0.002_f32).floor() as usize).max(3)
}

#[cfg(test)]
pub(crate) fn use_two_stage_phase(window: &PhaseWindow) -> Result<bool, String> {
    use_two_stage_phase_with_reference(window, None)
}

#[cfg(test)]
pub(crate) fn use_two_stage_phase_with_reference(
    window: &PhaseWindow,
    reference: Option<&PhasedWindow>,
) -> Result<bool, String> {
    let low_frequency = low_frequency_markers_with_reference(window, reference)?;
    Ok(use_two_stage_phase_from_mask(&low_frequency))
}

pub(crate) fn use_two_stage_phase_from_mask(low_frequency: &[bool]) -> bool {
    if low_frequency.is_empty() {
        return false;
    }
    let high_frequency_count = low_frequency.iter().filter(|&&is_low| !is_low).count();
    high_frequency_count >= 2 && high_frequency_count * 4 <= low_frequency.len() * 3
}

pub(crate) fn adaptive_imp_states(
    max_states: usize,
    reference_haplotypes: usize,
    reference_markers: usize,
    target_markers: usize,
) -> usize {
    const MIN_STATES: usize = 400;
    let state_scale = reference_haplotypes.div_ceil(8);
    let density_factor = if target_markers == 0 {
        1
    } else {
        let marker_ratio = reference_markers.div_ceil(target_markers).max(1);
        (marker_ratio as f64).sqrt().floor().max(1.0) as usize
    };
    MIN_STATES
        .max(state_scale)
        .saturating_mul(density_factor)
        .min(max_states)
        .min(reference_haplotypes)
}

fn add_sample_carriers(counts: &mut [usize], alleles: [Option<u16>; 2], cap: usize) {
    let mut seen = [None, None];
    for allele in alleles.into_iter().flatten() {
        if seen.contains(&Some(allele)) {
            continue;
        }
        seen[usize::from(seen[0].is_some())] = Some(allele);
        if let Some(count) = counts.get_mut(usize::from(allele)) {
            *count = count.saturating_add(1).min(cap.saturating_add(1));
        }
    }
}

pub(crate) fn merged_marker_count(
    target: &[super::types::VariantRow],
    reference: &[PhasedRow],
    max_rows: Option<usize>,
) -> Result<usize, String> {
    validate_sorted_unique(target.iter().map(|row| &row.meta), "target")?;
    validate_sorted_unique(reference.iter().map(|row| &row.meta), "reference")?;
    merged_marker_count_impl(
        target,
        reference.len(),
        |index| &reference[index].meta,
        max_rows,
    )
}

pub(crate) fn merged_marker_count_from_metas(
    target: &[super::types::VariantRow],
    reference: &[VariantMeta],
    max_rows: Option<usize>,
) -> Result<usize, String> {
    validate_sorted_unique(target.iter().map(|row| &row.meta), "target")?;
    validate_sorted_unique(reference.iter(), "reference")?;
    merged_marker_count_impl(target, reference.len(), |index| &reference[index], max_rows)
}

fn merged_marker_count_impl<'a>(
    target: &[super::types::VariantRow],
    reference_len: usize,
    reference_meta: impl Fn(usize) -> &'a VariantMeta,
    max_rows: Option<usize>,
) -> Result<usize, String> {
    let (mut target_index, mut reference_index) = (0usize, 0usize);
    let mut count = 0usize;
    while target_index < target.len() || reference_index < reference_len {
        match (
            target.get(target_index),
            (reference_index < reference_len).then(|| reference_meta(reference_index)),
        ) {
            (Some(target_row), Some(ref_meta)) => {
                match compare_position(&target_row.meta, ref_meta) {
                    Ordering::Less => target_index += 1,
                    Ordering::Greater => reference_index += 1,
                    Ordering::Equal => {
                        harmonization(&target_row.meta, ref_meta)?;
                        target_index += 1;
                        reference_index += 1;
                    }
                }
            }
            (Some(_), None) => target_index += 1,
            (None, Some(_)) => reference_index += 1,
            (None, None) => break,
        }
        count = count
            .checked_add(1)
            .ok_or_else(|| "merged target/reference marker count overflow".to_owned())?;
        if max_rows.is_some_and(|limit| count > limit) {
            let limit = max_rows.unwrap();
            return Err(format!(
                "merged target/reference window exceeds --window-markers={limit}; reduce window size or split the reference input"
            ));
        }
    }
    Ok(count)
}

#[derive(Debug)]
struct HmmClusterPlan {
    member_union_indices: Vec<Vec<usize>>,
    member_target_indices: Vec<Vec<usize>>,
    cm_positions: Vec<f64>,
    cluster_end_cm_positions: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ImpIbsStepHashIndex {
    hashes: Vec<u64>,
    donor_haplotypes: Vec<u32>,
}

#[derive(Debug, Clone)]
struct ImpIbsReferenceIndex {
    step_ranges: Vec<(usize, usize)>,
    marker_step_ranges: Vec<(usize, usize)>,
    step_marker_ranges: Vec<(usize, usize)>,
    step_marker_indices: Vec<usize>,
    steps: Vec<ImpIbsStepHashIndex>,
    max_haps_per_step: usize,
    nsteps: usize,
}

impl ImpIbsReferenceIndex {
    fn new(
        reference: &composite::PackedBiallelicReference,
        cluster_plan: &HmmClusterPlan,
        step_cm: f64,
        nsteps: usize,
        segment_cm: f64,
        max_states: usize,
        pool: &ThreadPool,
    ) -> Result<Self, String> {
        // Beagle parses imp-step and imp-segment as float32 values. Keep the
        // same rounding before constructing steps or calculating the per-step
        // state cap.
        let step_cm = f64::from(step_cm as f32);
        let segment_cm = f64::from(segment_cm as f32);
        let n_reference_haplotypes = reference.n_haplotypes();
        if n_reference_haplotypes == 0
            || n_reference_haplotypes > u32::MAX as usize
            || cluster_plan.member_union_indices.is_empty()
            || cluster_plan.member_union_indices.len() != cluster_plan.cm_positions.len()
            || cluster_plan.member_union_indices.iter().any(Vec::is_empty)
            || !segment_cm.is_finite()
            || segment_cm <= 0.0
            || max_states == 0
            || nsteps == 0
        {
            return Err("imputation IBS reference index dimensions are invalid".to_owned());
        }
        let step_ranges = imp_step_ranges(&cluster_plan.cm_positions, step_cm)?;
        let marker_step_ranges = step_ranges
            .iter()
            .enumerate()
            .map(|(step, &(cluster_start, cluster_end))| {
                let start = if step == 0 {
                    0
                } else {
                    cluster_plan.member_union_indices[cluster_start][0]
                };
                let end = cluster_plan
                    .member_union_indices
                    .get(cluster_end)
                    .map_or(reference.n_markers(), |members| members[0]);
                (start, end)
            })
            .collect::<Vec<_>>();
        if marker_step_ranges
            .iter()
            .any(|&(start, end)| start >= end || end > reference.n_markers())
            || marker_step_ranges
                .windows(2)
                .any(|pair| pair[0].1 != pair[1].0)
        {
            return Err("imputation IBS marker steps do not partition the union window".to_owned());
        }
        let mut step_marker_ranges = Vec::with_capacity(step_ranges.len());
        let mut step_marker_indices = Vec::new();
        for &(cluster_start, cluster_end) in &step_ranges {
            let marker_start = step_marker_indices.len();
            for cluster in cluster_start..cluster_end {
                step_marker_indices.extend_from_slice(&cluster_plan.member_union_indices[cluster]);
            }
            step_marker_ranges.push((marker_start, step_marker_indices.len()));
        }
        if step_marker_indices.is_empty() {
            return Err("imputation IBS index has no target markers".to_owned());
        }
        let steps_per_segment = ((segment_cm as f32) / (step_cm as f32)).round() as usize;
        if steps_per_segment == 0 {
            return Err("imputation IBS segment contains no steps".to_owned());
        }
        // Small user state budgets must still retain IBS matches; a zero cap
        // would discard every candidate and force the random fallback.
        let max_haps_per_step = (max_states / steps_per_segment).max(1);
        let steps = pool.install(|| {
            step_marker_ranges
                .par_iter()
                .map(|&(marker_start, marker_end)| {
                    let marker_indices = &step_marker_indices[marker_start..marker_end];
                    let mut entries = (0..n_reference_haplotypes)
                        .map(|donor| {
                            let mut hasher = std::collections::hash_map::DefaultHasher::new();
                            marker_indices.len().hash(&mut hasher);
                            for &marker in marker_indices {
                                reference.allele_unchecked(marker, donor).hash(&mut hasher);
                            }
                            (hasher.finish(), donor as u32)
                        })
                        .collect::<Vec<_>>();
                    entries.sort_unstable();
                    let mut hashes = Vec::with_capacity(entries.len());
                    let mut donor_haplotypes = Vec::with_capacity(entries.len());
                    for (hash, donor) in entries {
                        hashes.push(hash);
                        donor_haplotypes.push(donor);
                    }
                    ImpIbsStepHashIndex {
                        hashes,
                        donor_haplotypes,
                    }
                })
                .collect::<Vec<_>>()
        });
        Ok(Self {
            step_ranges,
            marker_step_ranges,
            step_marker_ranges,
            step_marker_indices,
            steps,
            max_haps_per_step,
            nsteps,
        })
    }

    fn exact_matches_for_step(
        &self,
        reference: &composite::PackedBiallelicReference,
        query_alleles: &[Option<u16>],
        step: usize,
    ) -> Vec<usize> {
        let (marker_start, marker_end) = self.step_marker_ranges[step];
        let marker_indices = &self.step_marker_indices[marker_start..marker_end];
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        marker_indices.len().hash(&mut hasher);
        for &marker in marker_indices {
            query_alleles
                .get(marker)
                .copied()
                .unwrap_or(None)
                .hash(&mut hasher);
        }
        let query_hash = hasher.finish();
        let index = &self.steps[step];
        let start = index.hashes.partition_point(|&hash| hash < query_hash);
        let end = index.hashes.partition_point(|&hash| hash <= query_hash);
        let mut matching_donors = Vec::with_capacity(end.saturating_sub(start));
        for entry in start..end {
            let donor = index.donor_haplotypes[entry] as usize;
            let matches = marker_indices.iter().all(|&marker| {
                query_alleles
                    .get(marker)
                    .is_some_and(|&query| query == reference.allele_unchecked(marker, donor))
            });
            if matches {
                matching_donors.push(donor);
            }
        }
        matching_donors.sort_unstable();
        matching_donors
    }

    fn candidate_sets(
        &self,
        reference: &composite::PackedBiallelicReference,
        query_alleles: &[Option<u16>],
        seed: u64,
    ) -> Vec<Vec<usize>> {
        let exact_matches = (0..self.step_ranges.len())
            .map(|step| self.exact_matches_for_step(reference, query_alleles, step))
            .collect::<Vec<_>>();
        (0..self.step_ranges.len())
            .map(|step| {
                select_imp_ibs_candidates(
                    &exact_matches,
                    step,
                    self.nsteps,
                    self.max_haps_per_step,
                    seed,
                )
            })
            .collect()
    }
}

fn imp_step_ranges(cm_positions: &[f64], step_cm: f64) -> Result<Vec<(usize, usize)>, String> {
    // Beagle's Par.imp_step() is a float promoted to double in CodedSteps.
    let step_cm = f64::from(step_cm as f32);
    if cm_positions.is_empty()
        || !step_cm.is_finite()
        || step_cm <= 0.0
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("imputation IBS step map or step size is invalid".to_owned());
    }

    let mut starts = vec![0usize];
    let mut cursor = 0usize;
    let mut next_position = cm_positions[0] + step_cm * 0.5;
    while cursor < cm_positions.len() {
        cursor += cm_positions[cursor..].partition_point(|&position| position < next_position);
        if cursor >= cm_positions.len() {
            break;
        }
        starts.push(cursor);
        let current_position = cm_positions[cursor];
        next_position = current_position + step_cm;
        if !(next_position > current_position) {
            return Err("imputation IBS step is too small for the genetic map".to_owned());
        }
    }

    Ok(starts
        .iter()
        .copied()
        .zip(
            starts
                .iter()
                .copied()
                .skip(1)
                .chain(std::iter::once(cm_positions.len())),
        )
        .collect())
}

fn select_imp_ibs_candidates(
    exact_matches_by_step: &[Vec<usize>],
    start_step: usize,
    n_steps: usize,
    max_haps_per_step: usize,
    seed: u64,
) -> Vec<usize> {
    let Some(first_matches) = exact_matches_by_step.get(start_step) else {
        return Vec::new();
    };
    if max_haps_per_step == 0 {
        return Vec::new();
    }
    if first_matches.len() <= max_haps_per_step {
        return first_matches.clone();
    }

    let end_step = start_step
        .saturating_add(n_steps.max(1))
        .min(exact_matches_by_step.len());
    let mut parent = first_matches.clone();
    for matches in &exact_matches_by_step[start_step + 1..end_step] {
        let mut child = Vec::with_capacity(parent.len().min(matches.len()));
        let (mut parent_index, mut match_index) = (0usize, 0usize);
        while parent_index < parent.len() && match_index < matches.len() {
            match parent[parent_index].cmp(&matches[match_index]) {
                std::cmp::Ordering::Less => parent_index += 1,
                std::cmp::Ordering::Greater => match_index += 1,
                std::cmp::Ordering::Equal => {
                    child.push(parent[parent_index]);
                    parent_index += 1;
                    match_index += 1;
                }
            }
        }
        if child.len() <= max_haps_per_step {
            let mut siblings = Vec::with_capacity(parent.len().saturating_sub(child.len()));
            let (mut parent_index, mut child_index) = (0usize, 0usize);
            while parent_index < parent.len() {
                if child_index < child.len() && parent[parent_index] == child[child_index] {
                    child_index += 1;
                } else {
                    siblings.push(parent[parent_index]);
                }
                parent_index += 1;
            }
            let sample_count = max_haps_per_step
                .saturating_sub(child.len())
                .min(siblings.len());
            if sample_count > 0 {
                let mut random = super::pbwt::JavaRandom::new(
                    seed.wrapping_add(parent.first().copied().unwrap_or(0) as u64),
                );
                for index in 0..sample_count {
                    let selected = index + random.next_int(siblings.len() - index);
                    siblings.swap(index, selected);
                }
                child.extend_from_slice(&siblings[..sample_count]);
                child.sort_unstable();
            }
            return child;
        }
        parent = child;
    }

    let sample_count = max_haps_per_step.min(parent.len());
    if sample_count < parent.len() {
        let mut random = super::pbwt::JavaRandom::new(
            seed.wrapping_add(parent.first().copied().unwrap_or(0) as u64),
        );
        for index in 0..sample_count {
            let selected = index + random.next_int(parent.len() - index);
            parent.swap(index, selected);
        }
        parent.truncate(sample_count);
        parent.sort_unstable();
    }
    parent
}

#[cfg(test)]
fn build_hmm_cluster_plan(
    target_row_indices: &[Option<usize>],
    target_cm_positions: &[f64],
    target_projection_cm_positions: &[f64],
    cluster_cm: f64,
) -> HmmClusterPlan {
    build_hmm_cluster_plan_with_groups(
        target_row_indices,
        target_cm_positions,
        target_projection_cm_positions,
        cluster_cm,
        &vec![None; target_row_indices.len()],
    )
}

fn build_hmm_cluster_plan_with_groups(
    target_row_indices: &[Option<usize>],
    target_cm_positions: &[f64],
    target_projection_cm_positions: &[f64],
    cluster_cm: f64,
    union_reference_groups: &[Option<u64>],
) -> HmmClusterPlan {
    debug_assert_eq!(target_row_indices.len(), union_reference_groups.len());
    debug_assert_eq!(
        target_cm_positions.len(),
        target_projection_cm_positions.len()
    );
    let target_markers = target_row_indices
        .iter()
        .enumerate()
        .filter_map(|(union_index, target_index)| {
            target_index.map(|target_index| {
                (
                    union_index,
                    target_index,
                    target_cm_positions[target_index],
                    target_projection_cm_positions[target_index],
                )
            })
        })
        .collect::<Vec<_>>();
    // Beagle's cluster parameter is parsed as float32.
    let cluster_cm = f64::from(cluster_cm as f32);
    let cluster_cm = if cluster_cm.is_finite() && cluster_cm > 0.0 {
        cluster_cm
    } else {
        0.0
    };
    let mut member_union_indices = Vec::new();
    let mut member_target_indices = Vec::new();
    let mut cm_positions = Vec::new();
    let mut cluster_end_cm_positions = Vec::new();
    // ImpData.targBlockEnd: only sequence-coded shared target records can
    // change the block. Intervening allele-coded records keep the last group.
    let mut last_group = None;
    let mut block_ends = Vec::new();
    for (typed, &(union_index, _, _, _)) in target_markers.iter().enumerate() {
        if let Some(group) = union_reference_groups[union_index] {
            if last_group.is_some() && last_group != Some(group) {
                block_ends.push(typed);
            }
            last_group = Some(group);
        }
    }
    block_ends.push(target_markers.len());
    let mut start = 0usize;
    let mut block_index = 0usize;
    while start < target_markers.len() {
        while block_ends[block_index] <= start {
            block_index += 1;
        }
        let cluster_start_cm = target_markers[start].2;
        let mut end = start + 1;
        while end < block_ends[block_index]
            && target_markers[end].2 - cluster_start_cm <= cluster_cm
        {
            end += 1;
        }
        let midpoint_cm = (cluster_start_cm + target_markers[end - 1].2) * 0.5;
        member_union_indices.push(
            target_markers[start..end]
                .iter()
                .map(|&(union_index, _, _, _)| union_index)
                .collect(),
        );
        member_target_indices.push(
            target_markers[start..end]
                .iter()
                .map(|&(_, target_index, _, _)| target_index)
                .collect(),
        );
        cm_positions.push(midpoint_cm);
        cluster_end_cm_positions.push(target_markers[end - 1].3);
        start = end;
    }
    HmmClusterPlan {
        member_union_indices,
        member_target_indices,
        cm_positions,
        cluster_end_cm_positions,
    }
}

/// Reproduce Beagle ImpData.cumPos: cumulative map distance with a minimum
/// adjacent-marker increment of 1e-7 cM.
fn beagle_cumulative_positions(genetic_positions: &[f64]) -> Result<Vec<f64>, String> {
    const MIN_CM_DIST: f64 = 1.0e-7;
    if genetic_positions.is_empty() || genetic_positions.iter().any(|cm| !cm.is_finite()) {
        return Err("imputation genetic map positions must be nonempty and finite".to_owned());
    }
    let mut cumulative = Vec::with_capacity(genetic_positions.len());
    cumulative.push(0.0);
    let mut previous = genetic_positions[0];
    for &position in &genetic_positions[1..] {
        let distance = (position - previous).abs().max(MIN_CM_DIST);
        cumulative.push(cumulative.last().copied().unwrap_or(0.0) + distance);
        previous = position;
    }
    Ok(cumulative)
}

/// Map a target-only marker onto Beagle's cumulative reference-map coordinate
/// used to interpolate probabilities between adjacent typed markers.
fn project_genetic_position(
    position: f64,
    reference_genetic_positions: &[f64],
    reference_cumulative_positions: &[f64],
) -> Result<f64, String> {
    if reference_genetic_positions.is_empty()
        || reference_genetic_positions.len() != reference_cumulative_positions.len()
        || !position.is_finite()
        || reference_genetic_positions.iter().any(|cm| !cm.is_finite())
        || reference_genetic_positions
            .windows(2)
            .any(|pair| pair[1] < pair[0])
    {
        return Err("imputation projection map positions are inconsistent".to_owned());
    }
    let upper = reference_genetic_positions.partition_point(|&cm| cm < position);
    if upper == 0 {
        return Ok(reference_cumulative_positions[0] - (reference_genetic_positions[0] - position));
    }
    if upper == reference_genetic_positions.len() {
        let last = upper - 1;
        return Ok(
            reference_cumulative_positions[last] + (position - reference_genetic_positions[last])
        );
    }
    if reference_genetic_positions[upper] == position {
        return Ok(reference_cumulative_positions[upper]);
    }
    let lower = upper - 1;
    let raw_span = reference_genetic_positions[upper] - reference_genetic_positions[lower];
    if raw_span <= 0.0 {
        return Ok(reference_cumulative_positions[lower]);
    }
    let fraction = (position - reference_genetic_positions[lower]) / raw_span;
    Ok(reference_cumulative_positions[lower]
        + fraction
            * (reference_cumulative_positions[upper] - reference_cumulative_positions[lower]))
}

fn phase_estimate_cluster_errors(
    phased_alleles: &[Option<u16>],
    plan: &HmmClusterPlan,
    error_rate: f64,
) -> Vec<f64> {
    plan.member_union_indices
        .iter()
        .map(|members| {
            let observed = members
                .iter()
                .filter(|&&marker| phased_alleles[marker].is_some())
                .count();
            (error_rate * observed as f64).min(0.5)
        })
        .collect()
}

/// Construct the reference-imputation HMM observations and emission error
/// rates from Beagle's phased target haplotypes. Beagle treats its phase
/// estimates at originally missing GTs as the haplotype observations here;
/// the error rate is scaled by the number of available phased alleles in each
/// marker cluster.
#[cfg(test)]
fn build_imputation_hmm_inputs(
    full_panel: &CompositePanel,
    phased_alleles: &[Option<u16>],
    plan: &HmmClusterPlan,
    error_rate: f64,
) -> Result<(PackedHaplotypes, CompositePanel, Vec<f64>), String> {
    let errors = phase_estimate_cluster_errors(phased_alleles, plan, error_rate);
    let (observations, states) = build_cluster_hmm_panels(full_panel, phased_alleles, plan)?;
    Ok((observations, states, errors))
}

#[cfg(test)]
fn build_cluster_hmm_panels(
    full_panel: &CompositePanel,
    phased_alleles: &[Option<u16>],
    plan: &HmmClusterPlan,
) -> Result<(PackedHaplotypes, CompositePanel), String> {
    if plan.member_union_indices.is_empty()
        || plan.member_union_indices.len() != plan.member_target_indices.len()
        || plan.member_union_indices.len() != plan.cm_positions.len()
        || plan.member_union_indices.len() != plan.cluster_end_cm_positions.len()
        || full_panel.haplotypes.n_haplotypes == 0
        || full_panel.haplotypes.n_markers
            < plan
                .member_target_indices
                .iter()
                .flatten()
                .copied()
                .max()
                .map_or(0, |marker| marker.saturating_add(1))
        || plan
            .member_union_indices
            .iter()
            .flatten()
            .any(|&marker| marker >= phased_alleles.len())
    {
        return Err("cluster HMM marker mapping is inconsistent".to_owned());
    }
    let cluster_count = plan.member_union_indices.len();
    let state_count = full_panel.haplotypes.n_haplotypes;
    let mut observations = PackedHaplotypes::empty(1, cluster_count, 1)?;
    let mut states = PackedHaplotypes::empty(state_count, cluster_count, 1)?;
    let word_count = state_count.div_ceil(64);
    let mut mismatch_words = vec![0u64; word_count];
    for cluster in 0..cluster_count {
        mismatch_words.fill(0);
        let mut observed = false;
        let mut can_pack = true;
        for (&union_marker, &target_marker) in plan.member_union_indices[cluster]
            .iter()
            .zip(&plan.member_target_indices[cluster])
        {
            let Some(expected) = phased_alleles[union_marker] else {
                continue;
            };
            observed = true;
            let Some((allele_words, missing_words)) =
                full_panel.haplotypes.biallelic_marker_planes(target_marker)
            else {
                can_pack = false;
                break;
            };
            let expected_word = if expected == 1 { u64::MAX } else { 0 };
            for ((output, &allele), &missing) in mismatch_words
                .iter_mut()
                .zip(allele_words)
                .zip(missing_words)
            {
                *output |= (allele ^ expected_word) & !missing;
            }
        }
        if can_pack {
            states.set_biallelic_marker_words(cluster, &mismatch_words)?;
        } else {
            for state in 0..state_count {
                let mismatches = plan.member_union_indices[cluster]
                    .iter()
                    .zip(&plan.member_target_indices[cluster])
                    .filter_map(|(&union_marker, &target_marker)| {
                        phased_alleles[union_marker].map(|expected| {
                            full_panel
                                .haplotypes
                                .allele(target_marker, state)
                                .is_some_and(|observed| observed != expected)
                        })
                    })
                    .any(|mismatch| mismatch);
                states.set_allele(cluster, state, Some(u16::from(mismatches)))?;
            }
        }
        if observed {
            observations.set_biallelic_marker_with(cluster, |_| Some(0))?;
        }
    }
    Ok((
        observations,
        CompositePanel {
            haplotypes: states,
            source_samples: full_panel.source_samples.clone(),
            marker_indices: (0..cluster_count).collect(),
            allele_counts: vec![2; cluster_count],
        },
    ))
}

/// Generate one mismatch code per cluster/state directly from IBS tracks.
/// Cursors move monotonically; mismatching states skip the rest of a cluster.
#[allow(clippy::too_many_arguments)]
fn fill_cluster_mismatch_words_from_codes<T, F>(
    codes: &[(T, T)],
    tracks: &[Vec<composite::DonorHapSegment>],
    cursors: &mut [usize],
    active_donors: &mut [usize],
    active_ends: &mut [usize],
    members: &[usize],
    query_bits: u64,
    query_mask: u64,
    words: &mut [u64],
    mut allele_at: F,
) -> Result<(), String>
where
    T: Copy + Into<u64>,
    F: FnMut(usize, usize) -> Option<u16>,
{
    if tracks.len() != cursors.len()
        || tracks.len() != active_donors.len()
        || tracks.len() != active_ends.len()
        || words.len() != tracks.len().div_ceil(64)
    {
        return Err("cluster mismatch word dimensions are inconsistent".to_owned());
    }
    let first_marker = *members
        .first()
        .ok_or_else(|| "cluster mismatch members are empty".to_owned())?;
    let last_marker = *members
        .last()
        .ok_or_else(|| "cluster mismatch members are empty".to_owned())?;
    words.fill(0);
    for (state, track) in tracks.iter().enumerate() {
        let cursor = &mut cursors[state];
        let donor = &mut active_donors[state];
        let end = &mut active_ends[state];
        advance_active_track(track, cursor, donor, end, first_marker);
        // A composite state can change donor inside a cluster. Such states
        // keep the original marker-by-marker scan.
        if *end > last_marker {
            let (bits, known) = codes
                .get(*donor)
                .copied()
                .ok_or_else(|| "cluster code donor index is out of bounds".to_owned())?;
            if (bits.into() ^ query_bits) & known.into() & query_mask != 0 {
                words[state >> 6] |= 1u64 << (state & 63);
            }
            continue;
        }
        let mut mismatch = false;
        for (offset, &marker) in members.iter().enumerate() {
            advance_active_track(track, cursor, donor, end, marker);
            let expected = ((query_bits >> offset) & 1) as u16;
            let mask = (query_mask >> offset) & 1;
            if mask != 0 && allele_at(marker, *donor).is_some_and(|allele| allele != expected) {
                mismatch = true;
                break;
            }
        }
        if mismatch {
            words[state >> 6] |= 1u64 << (state & 63);
        }
    }
    Ok(())
}

fn fill_cluster_mismatch_words_scalar<F>(
    tracks: &[Vec<composite::DonorHapSegment>],
    cursors: &mut [usize],
    active_donors: &mut [usize],
    active_ends: &mut [usize],
    members: &[usize],
    query: &[Option<u16>],
    words: &mut [u64],
    mut allele_at: F,
) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Option<u16>,
{
    if tracks.len() != cursors.len()
        || tracks.len() != active_donors.len()
        || tracks.len() != active_ends.len()
        || words.len() != tracks.len().div_ceil(64)
    {
        return Err("cluster mismatch word dimensions are inconsistent".to_owned());
    }
    words.fill(0);
    for (state, track) in tracks.iter().enumerate() {
        let cursor = &mut cursors[state];
        let donor = &mut active_donors[state];
        let end = &mut active_ends[state];
        let mut mismatch = false;
        for &marker in members {
            let Some(expected) = query[marker] else {
                continue;
            };
            advance_active_track(track, cursor, donor, end, marker);
            if allele_at(marker, *donor).is_some_and(|allele| allele != expected) {
                mismatch = true;
                break;
            }
        }
        if mismatch {
            words[state >> 6] |= 1u64 << (state & 63);
        }
    }
    Ok(())
}

#[inline(always)]
fn advance_active_track(
    track: &[composite::DonorHapSegment],
    cursor: &mut usize,
    donor: &mut usize,
    end: &mut usize,
    marker: usize,
) {
    while *end <= marker {
        *cursor += 1;
        let segment = &track[*cursor];
        *donor = segment.donor_haplotype;
        *end = segment.end_marker;
    }
}

fn build_ibs_cluster_hmm_inputs(
    mosaic: &super::ibs_mosaic::IbsTargetMosaic,
    reference: &composite::PackedBiallelicReference,
    query: &[Option<u16>],
    plan: &HmmClusterPlan,
    err: f64,
    sequence_codes: Option<&super::cluster_codes::ClusterSequenceIndex>,
) -> Result<(PackedHaplotypes, CompositePanel, Vec<f64>), String> {
    let clusters = plan.member_union_indices.len();
    let n_states = mosaic.n_states();
    let mut observations = PackedHaplotypes::empty(1, clusters, 1)?;
    let mut states = PackedHaplotypes::empty(n_states, clusters, 1)?;
    for (cluster, members) in plan.member_union_indices.iter().enumerate() {
        if members.iter().any(|&m| query[m].is_some()) {
            observations.set_allele(cluster, 0, Some(0))?;
        }
    }
    let mut cursors = vec![0usize; n_states];
    let mut active_donors = mosaic
        .tracks
        .iter()
        .map(|track| track[0].donor_haplotype)
        .collect::<Vec<_>>();
    let mut active_ends = mosaic
        .tracks
        .iter()
        .map(|track| track[0].end_marker)
        .collect::<Vec<_>>();
    let mut words = vec![0u64; n_states.div_ceil(64)];
    for (cluster, members) in plan.member_union_indices.iter().enumerate() {
        let codes = sequence_codes.and_then(|index| index.cluster(cluster));
        let encoded_query = codes.and_then(|codes| codes.encode_query(members, query));
        match (codes, encoded_query) {
            (Some(super::cluster_codes::ClusterSequenceCodes::U8(codes)), Some(bits)) => {
                fill_cluster_mismatch_words_from_codes(
                    codes,
                    &mosaic.tracks,
                    &mut cursors,
                    &mut active_donors,
                    &mut active_ends,
                    members,
                    bits.0,
                    bits.1,
                    &mut words,
                    |marker, donor| reference.allele_unchecked(marker, donor),
                )?;
            }
            (Some(super::cluster_codes::ClusterSequenceCodes::U16(codes)), Some(bits)) => {
                fill_cluster_mismatch_words_from_codes(
                    codes,
                    &mosaic.tracks,
                    &mut cursors,
                    &mut active_donors,
                    &mut active_ends,
                    members,
                    bits.0,
                    bits.1,
                    &mut words,
                    |marker, donor| reference.allele_unchecked(marker, donor),
                )?;
            }
            (Some(super::cluster_codes::ClusterSequenceCodes::U32(codes)), Some(bits)) => {
                fill_cluster_mismatch_words_from_codes(
                    codes,
                    &mosaic.tracks,
                    &mut cursors,
                    &mut active_donors,
                    &mut active_ends,
                    members,
                    bits.0,
                    bits.1,
                    &mut words,
                    |marker, donor| reference.allele_unchecked(marker, donor),
                )?;
            }
            (Some(super::cluster_codes::ClusterSequenceCodes::U64(codes)), Some(bits)) => {
                fill_cluster_mismatch_words_from_codes(
                    codes,
                    &mosaic.tracks,
                    &mut cursors,
                    &mut active_donors,
                    &mut active_ends,
                    members,
                    bits.0,
                    bits.1,
                    &mut words,
                    |marker, donor| reference.allele_unchecked(marker, donor),
                )?;
            }
            _ => {
                fill_cluster_mismatch_words_scalar(
                    &mosaic.tracks,
                    &mut cursors,
                    &mut active_donors,
                    &mut active_ends,
                    members,
                    query,
                    &mut words,
                    |marker, donor| reference.allele_unchecked(marker, donor),
                )?;
            }
        }
        states.set_biallelic_marker_words(cluster, &words)?;
    }
    Ok((
        observations,
        CompositePanel {
            haplotypes: states,
            source_samples: mosaic
                .tracks
                .iter()
                .map(|t| t[0].donor_haplotype / 2)
                .collect(),
            marker_indices: (0..clusters).collect(),
            allele_counts: vec![2; clusters],
        },
        phase_estimate_cluster_errors(query, plan, err),
    ))
}

const SAMPLE_ASSEMBLY_CHUNK_MARKERS: usize = 256;

#[allow(clippy::too_many_arguments)]
fn assemble_sample_batch(
    output_rows: &mut [PhasedRow],
    posteriors: &[f32],
    posterior_markers: usize,
    batch_start: usize,
    _target: &PhaseWindow,
    phased_target: &PhasedWindow,
    target_row_indices: &[Option<usize>],
    posterior_marker_by_output: &[Option<usize>],
    reference_present: &[bool],
    reference_frequencies: &[Option<f32>],
    target_frequencies: &[Option<f32>],
    _orientation_flips: &[bool],
    collect_dosage_statistics: bool,
    pool: &ThreadPool,
) -> Vec<(f64, f64)> {
    debug_assert!(output_rows.len() <= target_row_indices.len());
    debug_assert_eq!(output_rows.len(), posterior_marker_by_output.len());
    debug_assert_eq!(target_row_indices.len(), reference_present.len());
    debug_assert_eq!(target_row_indices.len(), reference_frequencies.len());
    debug_assert_eq!(target_row_indices.len(), target_frequencies.len());
    debug_assert!(posterior_markers > 0);
    let batch_samples = posteriors.len() / (2 * posterior_markers);
    debug_assert_eq!(posteriors.len() % (2 * posterior_markers), 0);

    let mut statistics = vec![(0.0, 0.0); output_rows.len()];
    pool.install(|| {
        output_rows
            .par_chunks_mut(SAMPLE_ASSEMBLY_CHUNK_MARKERS)
            .zip(statistics.par_chunks_mut(SAMPLE_ASSEMBLY_CHUNK_MARKERS))
            .enumerate()
            .for_each(|(chunk_index, (rows, chunk_statistics))| {
                let marker_base = chunk_index * SAMPLE_ASSEMBLY_CHUNK_MARKERS;
                for (offset, (output_row, statistics)) in
                    rows.iter_mut().zip(chunk_statistics).enumerate()
                {
                    let posterior_marker = marker_base + offset;
                    let marker = posterior_marker;
                    let target_index = target_row_indices[marker];
                    let phased_calls = target_index.map(|index| &phased_target.rows[index].calls);
                    let reference_present = reference_present[marker];
                    let p_alt = reference_frequencies[marker]
                        .or(target_frequencies[marker])
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0);
                    let assemble_dosage = output_row.dosage.is_some();
                    let assemble_ap1 = output_row.ap1.is_some();
                    let assemble_ap2 = output_row.ap2.is_some();
                    let assemble_gp = output_row.gp.is_some();
                    let mut dosage_sum = 0.0f64;
                    let mut dosage_square_sum = 0.0f64;
                    for batch_offset in 0..batch_samples {
                        let sample = batch_start + batch_offset;
                        let posterior_start = batch_offset * 2 * posterior_markers;
                        let posterior0 =
                            &posteriors[posterior_start..posterior_start + posterior_markers];
                        let posterior1 = &posteriors[posterior_start + posterior_markers
                            ..posterior_start + 2 * posterior_markers];
                        let phased = phased_calls.map(|calls| calls.get(sample).unwrap());
                        let (mut p0, mut p1) = if let Some(phased_call) = phased {
                            // Beagle's ImputedVcfWriter sets isImputed only for
                            // reference-only records. Every record present in
                            // the target VCF, including an originally missing
                            // one, is restored from phasedTarg after phasing.
                            (
                                phased_call.alleles[0].unwrap_or(0) as f32,
                                phased_call.alleles[1].unwrap_or(0) as f32,
                            )
                        } else if reference_present {
                            let posterior_marker = posterior_marker_by_output[marker]
                                .expect("reference-only output marker has a posterior marker");
                            (posterior0[posterior_marker], posterior1[posterior_marker])
                        } else {
                            (p_alt, p_alt)
                        };
                        p0 = p0.clamp(0.0, 1.0);
                        p1 = p1.clamp(0.0, 1.0);
                        let call = phased.unwrap_or_else(|| DiploidCall {
                            alleles: [Some(hard_call_alt(p0)), Some(hard_call_alt(p1))],
                            phased: true,
                        });
                        output_row.calls.set_call_unchecked(
                            sample,
                            DiploidCall {
                                alleles: call.alleles,
                                phased: true,
                            },
                        );
                        let dosage = (p0 + p1) as f64;
                        if collect_dosage_statistics {
                            dosage_sum += dosage;
                            dosage_square_sum += dosage * dosage;
                        }
                        if assemble_dosage {
                            output_row.dosage.as_mut().unwrap().push(dosage as f32);
                        }
                        if assemble_ap1 {
                            output_row.ap1.as_mut().unwrap().push(p0);
                        }
                        if assemble_ap2 {
                            output_row.ap2.as_mut().unwrap().push(p1);
                        }
                        if assemble_gp {
                            output_row.gp.as_mut().unwrap().push([
                                (1.0 - p0) * (1.0 - p1),
                                p0 * (1.0 - p1) + (1.0 - p0) * p1,
                                p0 * p1,
                            ]);
                        }
                    }
                    *statistics = (dosage_sum, dosage_square_sum);
                }
            });
    });
    statistics
}

#[allow(clippy::too_many_arguments)]
fn assemble_direct_target_batch(
    output_rows: &mut [PhasedRow],
    sample_start: usize,
    sample_end: usize,
    phased_target: &PhasedWindow,
    target_row_indices: &[Option<usize>],
    collect_dosage_statistics: bool,
    pool: &ThreadPool,
) -> Vec<(f64, f64)> {
    debug_assert_eq!(output_rows.len(), target_row_indices.len());
    let mut statistics = vec![(0.0, 0.0); output_rows.len()];
    if output_rows.is_empty() {
        return statistics;
    }
    let first_target_index = target_row_indices[0]
        .expect("direct target assembly requires a target row at every marker");
    let sample_count = phased_target.rows[first_target_index].calls.len();
    debug_assert!(sample_start < sample_end);
    debug_assert!(sample_end <= sample_count);
    pool.install(|| {
        output_rows
            .par_chunks_mut(SAMPLE_ASSEMBLY_CHUNK_MARKERS)
            .zip(statistics.par_chunks_mut(SAMPLE_ASSEMBLY_CHUNK_MARKERS))
            .enumerate()
            .for_each(|(chunk_index, (rows, chunk_statistics))| {
                let marker_base = chunk_index * SAMPLE_ASSEMBLY_CHUNK_MARKERS;
                for (offset, (output_row, statistics)) in
                    rows.iter_mut().zip(chunk_statistics).enumerate()
                {
                    let target_index = target_row_indices[marker_base + offset]
                        .expect("direct target assembly requires a target row at every marker");
                    let phased_calls = &phased_target.rows[target_index].calls;
                    let mut dosage_sum = 0.0f64;
                    let mut dosage_square_sum = 0.0f64;
                    for sample in sample_start..sample_end {
                        let call = phased_calls
                            .get(sample)
                            .expect("phased target sample is in range");
                        let p0 = call.alleles[0].unwrap_or(0) as f32;
                        let p1 = call.alleles[1].unwrap_or(0) as f32;
                        output_row.calls.set_call_unchecked(
                            sample,
                            DiploidCall {
                                alleles: [Some(p0 as u16), Some(p1 as u16)],
                                phased: true,
                            },
                        );
                        let dosage = (p0 + p1) as f64;
                        if collect_dosage_statistics {
                            dosage_sum += dosage;
                            dosage_square_sum += dosage * dosage;
                        }
                        if let Some(values) = output_row.dosage.as_mut() {
                            values.push((p0 + p1) as f32);
                        }
                        if let Some(values) = output_row.ap1.as_mut() {
                            values.push(p0);
                        }
                        if let Some(values) = output_row.ap2.as_mut() {
                            values.push(p1);
                        }
                        if let Some(values) = output_row.gp.as_mut() {
                            values.push([
                                (1.0 - p0) * (1.0 - p1),
                                p0 * (1.0 - p1) + (1.0 - p0) * p1,
                                p0 * p1,
                            ]);
                        }
                    }
                    *statistics = (dosage_sum, dosage_square_sum);
                }
            });
    });
    statistics
}

fn fill_dosage_info(
    output_rows: &mut [PhasedRow],
    statistics: &[(f64, f64)],
    sample_count: usize,
    collect_dosage_statistics: bool,
) {
    if !collect_dosage_statistics {
        return;
    }
    for (row, &(sum, square_sum)) in output_rows.iter_mut().zip(statistics) {
        let mean_ds = sum / sample_count as f64;
        let variance = (square_sum / sample_count as f64 - mean_ds.powi(2)).max(0.0);
        let af = (mean_ds / 2.0).clamp(0.0, 1.0) as f32;
        let expected = 2.0 * af as f64 * (1.0 - af as f64);
        row.info_af = Some(af);
        row.info_dr2 = Some(if expected > f64::EPSILON {
            (variance / expected).clamp(0.0, 1.0) as f32
        } else {
            0.0
        });
    }
}

fn make_output_row(
    meta: &VariantMeta,
    target_samples: usize,
    output: &super::types::VcfOutputFields,
    imputed: bool,
) -> PhasedRow {
    PhasedRow {
        meta: meta.clone(),
        calls: DiploidCalls::empty(target_samples),
        dosage: output.dosage.then(|| Vec::with_capacity(target_samples)),
        ap1: output
            .allele_probabilities
            .then(|| Vec::with_capacity(target_samples)),
        ap2: output
            .allele_probabilities
            .then(|| Vec::with_capacity(target_samples)),
        gp: output
            .genotype_probabilities
            .then(|| Vec::with_capacity(target_samples)),
        info_af: None,
        info_dr2: None,
        imputed,
    }
}

fn owned_output_indices(
    target: &PhaseWindow,
    target_row_indices: &[Option<usize>],
    previous_emitted_pos: Option<u64>,
    reference_emit_end_pos: Option<u64>,
    metadata: &[VariantMeta],
) -> Vec<usize> {
    debug_assert_eq!(metadata.len(), target_row_indices.len());
    let last_owned_pos = target
        .emit_end
        .checked_sub(1)
        .and_then(|index| target.rows.get(index))
        .map(|row| row.meta.pos);
    let last_reference_owned_pos = last_owned_pos
        .into_iter()
        .chain(reference_emit_end_pos)
        .max();
    let next_unowned_pos = target.rows.get(target.emit_end).map(|row| row.meta.pos);
    let mut owned = Vec::new();
    for (marker, meta) in metadata.iter().enumerate() {
        let target_index = target_row_indices[marker];
        let target_owned =
            target_index.is_some_and(|index| index >= target.emit_start && index < target.emit_end);
        let after_previous = previous_emitted_pos.map_or(true, |previous| meta.pos > previous);
        let before_next_target = match next_unowned_pos {
            Some(next_pos) => {
                meta.pos < next_pos && reference_emit_end_pos.map_or(true, |end| meta.pos <= end)
            }
            None => last_reference_owned_pos.is_some_and(|end| meta.pos <= end),
        };
        let reference_only_owned = target_index.is_none() && after_previous && before_next_target;
        if target_owned || reference_only_owned {
            owned.push(marker);
        }
    }
    owned
}

#[cfg(test)]
pub(crate) fn impute_window_prevalidated(
    target: &PhaseWindow,
    phased_target: &PhasedWindow,
    reference: PhasedWindow,
    reference_cm_positions: &[f64],
    config: &ImputeConfig,
    hmm_config: &HMMConfig,
    seed: u64,
    previous_emitted_pos: Option<u64>,
    reference_emit_end_pos: Option<u64>,
    orientation_flips: &[bool],
    pool: &ThreadPool,
) -> Result<ImputedWindow, String> {
    for row in &reference.rows {
        validate_phased_reference_calls(row)?;
    }
    let mut groups = super::reference_groups::ReferenceSequenceGroups::new(
        reference.rows.first().map_or(0, |row| row.calls.len()),
    );
    let reference_groups = reference
        .rows
        .iter()
        .map(|row| groups.push(&row.meta.chrom, &row.calls))
        .collect::<Result<Vec<_>, _>>()?;
    impute_window_with_output(
        target,
        phased_target,
        ReferenceInput::Phased(reference),
        reference_cm_positions,
        &reference_groups,
        config,
        hmm_config,
        seed,
        previous_emitted_pos,
        reference_emit_end_pos,
        orientation_flips,
        pool,
        None,
    )
}

#[allow(dead_code)]
pub(crate) fn impute_window_stream(
    target: &PhaseWindow,
    phased_target: &PhasedWindow,
    reference: PhasedWindow,
    reference_cm_positions: &[f64],
    reference_groups: &[Option<u64>],
    config: &ImputeConfig,
    hmm_config: &HMMConfig,
    seed: u64,
    previous_emitted_pos: Option<u64>,
    reference_emit_end_pos: Option<u64>,
    orientation_flips: &[bool],
    pool: &ThreadPool,
    emit: &mut dyn FnMut(&[PhasedRow], &[usize]) -> Result<(), String>,
) -> Result<(), String> {
    impute_window_with_output(
        target,
        phased_target,
        ReferenceInput::Phased(reference),
        reference_cm_positions,
        reference_groups,
        config,
        hmm_config,
        seed,
        previous_emitted_pos,
        reference_emit_end_pos,
        orientation_flips,
        pool,
        Some(emit),
    )
    .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn impute_window_stream_compact(
    target: &PhaseWindow,
    phased_target: &PhasedWindow,
    mut reference: super::reference_store::CompactReferenceWindow,
    config: &ImputeConfig,
    hmm_config: &HMMConfig,
    seed: u64,
    previous_emitted_pos: Option<u64>,
    reference_emit_end_pos: Option<u64>,
    orientation_flips: &[bool],
    pool: &ThreadPool,
    emit: &mut dyn FnMut(&[PhasedRow], &[usize]) -> Result<(), String>,
) -> Result<(), String> {
    let reference_cm_positions = std::mem::take(&mut reference.cm_positions);
    let reference_groups = std::mem::take(&mut reference.groups);
    impute_window_with_output(
        target,
        phased_target,
        ReferenceInput::Compact(reference),
        &reference_cm_positions,
        &reference_groups,
        config,
        hmm_config,
        seed,
        previous_emitted_pos,
        reference_emit_end_pos,
        orientation_flips,
        pool,
        Some(emit),
    )
    .map(|_| ())
}

enum ReferenceInput {
    Phased(PhasedWindow),
    Compact(super::reference_store::CompactReferenceWindow),
}

impl ReferenceInput {
    fn chrom(&self) -> &str {
        match self {
            Self::Phased(window) => &window.chrom,
            Self::Compact(window) => &window.chrom,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Phased(window) => window.rows.len(),
            Self::Compact(window) => window.metas.len(),
        }
    }

    fn n_samples(&self) -> usize {
        match self {
            Self::Phased(window) => window.rows.first().map_or(0, |row| row.calls.len()),
            Self::Compact(window) => window.n_samples,
        }
    }

    fn meta(&self, index: usize) -> &VariantMeta {
        match self {
            Self::Phased(window) => &window.rows[index].meta,
            Self::Compact(window) => &window.metas[index],
        }
    }
}

fn impute_window_with_output(
    target: &PhaseWindow,
    phased_target: &PhasedWindow,
    reference: ReferenceInput,
    reference_cm_positions: &[f64],
    reference_groups: &[Option<u64>],
    config: &ImputeConfig,
    hmm_config: &HMMConfig,
    seed: u64,
    previous_emitted_pos: Option<u64>,
    reference_emit_end_pos: Option<u64>,
    orientation_flips: &[bool],
    pool: &ThreadPool,
    mut emit: Option<&mut dyn FnMut(&[PhasedRow], &[usize]) -> Result<(), String>>,
) -> Result<ImputedWindow, String> {
    let profile_impute = std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some()
        || std::env::var_os("JANUSX_IMP_DETAIL_PROFILE").is_some();
    let profile_started = profile_impute.then(Instant::now);
    let reference_index_started = profile_started;
    if target.rows.is_empty() || reference.len() == 0 {
        return Err(
            "reference imputation requires nonempty target and reference windows".to_owned(),
        );
    }
    if let ReferenceInput::Phased(ref phased_ref) = reference {
        for row in &phased_ref.rows {
            validate_phased_reference_calls(row)?;
        }
    }
    if target.rows.len() != target.cm_positions.len() {
        return Err("target window rows and map positions have different lengths".to_owned());
    }
    if target.chrom != reference.chrom() {
        return Err(format!(
            "target and reference windows must be on the same chromosome ({} versus {})",
            target.chrom,
            reference.chrom()
        ));
    }
    if target.rows.len() != phased_target.rows.len()
        || target
            .rows
            .iter()
            .zip(&phased_target.rows)
            .any(|(raw, phased)| {
                raw.meta.chrom != phased.meta.chrom || raw.meta.pos != phased.meta.pos
            })
    {
        return Err("phased target window does not match its input window".to_owned());
    }
    if reference.len() != reference_cm_positions.len() {
        return Err("reference window rows and map positions have different lengths".to_owned());
    }
    if reference.len() != reference_groups.len() {
        return Err("reference window rows and sequence groups have different lengths".to_owned());
    }
    for row in &target.rows {
        if row.meta.alt_alleles.len() != 1 {
            return Err(format!(
                "reference imputation currently requires biallelic variants; {}:{} has {} ALT alleles",
                row.meta.chrom,
                row.meta.pos,
                row.meta.alt_alleles.len()
            ));
        }
    }
    let target_samples = target.rows[0].calls.len();
    let target_accounting = target_marker_accounting(target);
    let reference_samples = reference.n_samples();
    if target_samples == 0 || reference_samples == 0 {
        return Err("target or reference window has zero samples".to_owned());
    }
    let reference_haplotypes = reference_samples
        .checked_mul(2)
        .ok_or_else(|| "imputation reference haplotype count overflow".to_owned())?;
    let imputation_hmm_config = imputation_hmm_config(hmm_config, reference_haplotypes)?;
    if orientation_flips.len() != target_samples {
        return Err(format!(
            "phase-orientation vector has {} entries; expected {target_samples}",
            orientation_flips.len()
        ));
    }
    let merged_indices = match &reference {
        ReferenceInput::Phased(window) => {
            merge_sorted_indices(&target.rows, &window.rows, Some(config.window_markers))?
        }
        ReferenceInput::Compact(window) => merge_sorted_indices_from_metas(
            &target.rows,
            &window.metas,
            Some(config.window_markers),
        )?,
    };
    let mut target_row_indices = Vec::with_capacity(merged_indices.len());
    let mut reference_row_indices = Vec::with_capacity(merged_indices.len());
    let mut reference_swaps = Vec::with_capacity(merged_indices.len());
    let mut merged_cm = Vec::with_capacity(merged_indices.len());
    for (target_index, reference_index) in merged_indices {
        let target_meta = target_index.map(|index| &target.rows[index].meta);
        let reference_meta = reference_index.map(|index| reference.meta(index));
        let swap = match (target_meta, reference_meta) {
            (Some(target_meta), Some(reference_meta)) => {
                harmonization(target_meta, reference_meta)?
            }
            _ => false,
        };
        let meta = target_meta
            .or(reference_meta)
            .ok_or_else(|| "target/reference position has no variant metadata".to_owned())?;
        let cm = target_index
            .map(|index| target.cm_positions[index])
            .or_else(|| reference_index.map(|index| reference_cm_positions[index]))
            .ok_or_else(|| format!("missing map position for {}:{}", meta.chrom, meta.pos))?;
        if merged_cm.last().is_some_and(|previous| cm < *previous) {
            return Err(format!(
                "merged target/reference genetic map decreases at {}:{}",
                meta.chrom, meta.pos
            ));
        }
        target_row_indices.push(target_index);
        reference_row_indices.push(reference_index);
        reference_swaps.push(swap);
        merged_cm.push(cm);
    }

    let posterior_projection_markers = target_row_indices
        .iter()
        .filter(|target_index| target_index.is_none())
        .count();
    let direct_target_markers = target_row_indices.len() - posterior_projection_markers;
    if posterior_projection_markers == 0 {
        drop(reference);
        let metadata = target_row_indices
            .iter()
            .map(|target_index| {
                target_index
                    .map(|index| target.rows[index].meta.clone())
                    .ok_or_else(|| "direct target fast path missing target metadata".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Every output marker is present in the target VCF. Phasing has
        // already recovered its genotype, so reference imputation and
        // posterior projection cannot change any value.
        let output_setup_started = profile_impute.then(Instant::now);
        let collect_dosage_statistics = config.output.dosage;
        let owned = owned_output_indices(
            target,
            &target_row_indices,
            previous_emitted_pos,
            reference_emit_end_pos,
            &metadata,
        );
        let floats_per_sample = 2
            + usize::from(config.output.dosage)
            + 2 * usize::from(config.output.allele_probabilities)
            + 3 * usize::from(config.output.genotype_probabilities);
        let bytes_per_marker = target_samples
            .saturating_mul(floats_per_sample * 4 + 1)
            .max(1);
        let tile_markers = ((32 * 1024 * 1024) / bytes_per_marker).clamp(1, 4096);
        let output_setup_elapsed =
            output_setup_started.map_or(Duration::ZERO, |started| started.elapsed());
        let direct_setup_elapsed = reference_index_started
            .map_or(Duration::ZERO, |started| started.elapsed())
            .saturating_sub(output_setup_elapsed);

        if let Some(emit) = emit.as_mut() {
            let mut assembly_elapsed = Duration::ZERO;
            for start in (0..metadata.len()).step_by(tile_markers) {
                let end = (start + tile_markers).min(metadata.len());
                let mut rows = metadata[start..end]
                    .iter()
                    .map(|meta| make_output_row(meta, target_samples, &config.output, false))
                    .collect::<Vec<_>>();
                let assembly_started = profile_impute.then(Instant::now);
                let statistics = assemble_direct_target_batch(
                    &mut rows,
                    0,
                    target_samples,
                    phased_target,
                    &target_row_indices[start..end],
                    collect_dosage_statistics,
                    pool,
                );
                assembly_elapsed +=
                    assembly_started.map_or(Duration::ZERO, |started| started.elapsed());
                fill_dosage_info(
                    &mut rows,
                    &statistics,
                    target_samples,
                    collect_dosage_statistics,
                );
                let first = owned.partition_point(|&marker| marker < start);
                let last = owned.partition_point(|&marker| marker < end);
                let indices = owned[first..last]
                    .iter()
                    .map(|marker| marker - start)
                    .collect::<Vec<_>>();
                emit(&rows, &indices)?;
            }
            emit_imputation_state_accounting(
                profile_impute,
                target,
                target_accounting,
                metadata.len(),
                posterior_projection_markers,
                target_samples,
                [HaplotypePosteriorStats::default(); 2],
            );
            let finalize_started = profile_impute.then(Instant::now);
            if let (Some(profile_started), Some(finalize_started)) =
                (profile_started, finalize_started)
            {
                eprintln!(
                    "JANUSX_IMP_IMPUTE_PROFILE window={}:{} markers={} posterior_projection_markers={posterior_projection_markers} direct_target_markers={direct_target_markers} target_samples={} batch_size={} reference_index_s={:.6} output_setup_s={:.6} posterior_s=0.000000 assembly_s={:.6} finalize_s={:.6} total_s={:.6} direct_target_fast_path=true streamed=true",
                    target.chrom,
                    target.index,
                    metadata.len(),
                    target_samples,
                    target_samples,
                    direct_setup_elapsed.as_secs_f64(),
                    output_setup_elapsed.as_secs_f64(),
                    assembly_elapsed.as_secs_f64(),
                    finalize_started.elapsed().as_secs_f64(),
                    profile_started.elapsed().as_secs_f64(),
                );
            }
            return Ok(ImputedWindow { rows: Vec::new() });
        }

        let mut output_rows = metadata
            .iter()
            .map(|meta| make_output_row(meta, target_samples, &config.output, false))
            .collect::<Vec<_>>();
        let assembly_started = profile_impute.then(Instant::now);
        let statistics = assemble_direct_target_batch(
            &mut output_rows,
            0,
            target_samples,
            phased_target,
            &target_row_indices,
            collect_dosage_statistics,
            pool,
        );
        let assembly_elapsed = assembly_started.map_or(Duration::ZERO, |started| started.elapsed());
        fill_dosage_info(
            &mut output_rows,
            &statistics,
            target_samples,
            collect_dosage_statistics,
        );
        emit_imputation_state_accounting(
            profile_impute,
            target,
            target_accounting,
            metadata.len(),
            posterior_projection_markers,
            target_samples,
            [HaplotypePosteriorStats::default(); 2],
        );
        let finalize_started = profile_impute.then(Instant::now);
        if let (Some(profile_started), Some(finalize_started)) = (profile_started, finalize_started)
        {
            eprintln!(
                "JANUSX_IMP_IMPUTE_PROFILE window={}:{} markers={} posterior_projection_markers={posterior_projection_markers} direct_target_markers={direct_target_markers} target_samples={} batch_size={} reference_index_s={:.6} output_setup_s={:.6} posterior_s=0.000000 assembly_s={:.6} finalize_s={:.6} total_s={:.6} direct_target_fast_path=true",
                target.chrom,
                target.index,
                metadata.len(),
                target_samples,
                target_samples,
                direct_setup_elapsed.as_secs_f64(),
                output_setup_elapsed.as_secs_f64(),
                assembly_elapsed.as_secs_f64(),
                finalize_started.elapsed().as_secs_f64(),
                profile_started.elapsed().as_secs_f64(),
            );
        }
        return Ok(ImputedWindow { rows: output_rows });
    }

    let (packed_reference, mut reference_metas): (
        composite::PackedBiallelicReference,
        Vec<Option<VariantMeta>>,
    ) = match reference {
        ReferenceInput::Phased(window) => {
            let packed = composite::PackedBiallelicReference::from_reference_rows(
                &window.rows,
                &reference_row_indices,
                &reference_swaps,
                reference_samples,
            )?;
            let metas = window.rows.into_iter().map(|row| Some(row.meta)).collect();
            (packed, metas)
        }
        ReferenceInput::Compact(mut window) => {
            let packed = composite::PackedBiallelicReference::from_compact_reference(
                &mut window,
                &reference_row_indices,
                &reference_swaps,
            )?;
            let metas = window.metas.into_iter().map(Some).collect();
            (packed, metas)
        }
    };
    let mut metadata = Vec::with_capacity(target_row_indices.len());
    for (&target_index, &reference_index) in target_row_indices.iter().zip(&reference_row_indices) {
        let meta = if let Some(index) = target_index {
            target.rows[index].meta.clone()
        } else if let Some(index) = reference_index {
            reference_metas[index]
                .take()
                .ok_or_else(|| "duplicate reference metadata index".to_owned())?
        } else {
            return Err("target/reference position has no variant metadata".to_owned());
        };
        metadata.push(meta);
    }
    drop(reference_metas);

    let dense_posterior_layout = std::env::var_os("JANUSX_IMP_DENSE_POSTERIOR_LAYOUT").is_some();
    let mut next_posterior_marker = 0usize;
    let posterior_marker_by_output = if dense_posterior_layout {
        (0..metadata.len()).map(Some).collect::<Vec<_>>()
    } else {
        target_row_indices
            .iter()
            .map(|target_index| {
                if target_index.is_some() {
                    None
                } else {
                    let marker = next_posterior_marker;
                    next_posterior_marker += 1;
                    Some(marker)
                }
            })
            .collect::<Vec<_>>()
    };
    let posterior_marker_indices = posterior_marker_by_output
        .iter()
        .enumerate()
        .filter_map(|(marker, posterior_marker)| posterior_marker.map(|_| marker))
        .collect::<Vec<_>>();
    let posterior_layout_markers = if dense_posterior_layout {
        metadata.len()
    } else {
        posterior_projection_markers
    };
    debug_assert_eq!(posterior_marker_indices.len(), posterior_layout_markers);
    let target_hmm_cm_positions = beagle_cumulative_positions(&target.cm_positions)?;
    let reference_projection_cm_positions = beagle_cumulative_positions(reference_cm_positions)?;
    let mut merged_projection_cm_positions = Vec::with_capacity(merged_cm.len());
    let mut target_projection_cm_positions = vec![0.0; target.rows.len()];
    for (union_marker, (&genetic_position, &reference_index)) in
        merged_cm.iter().zip(&reference_row_indices).enumerate()
    {
        let projection_position = if let Some(reference_index) = reference_index {
            *reference_projection_cm_positions
                .get(reference_index)
                .ok_or_else(|| "reference projection marker index is out of range".to_owned())?
        } else {
            project_genetic_position(
                genetic_position,
                reference_cm_positions,
                &reference_projection_cm_positions,
            )?
        };
        merged_projection_cm_positions.push(projection_position);
        if let Some(target_index) = target_row_indices[union_marker] {
            target_projection_cm_positions[target_index] = projection_position;
        }
    }
    let (reference_major_alleles, packed_reference_alt_frequencies) =
        packed_reference.allele_summaries();
    let hmm_union_indices = target_row_indices
        .iter()
        .enumerate()
        .filter_map(|(union_index, target_index)| target_index.map(|_| union_index))
        .collect::<Vec<_>>();
    if hmm_union_indices.is_empty() {
        return Err("reference imputation has no target markers for the HMM".to_owned());
    }
    // Target-only variants remain in the output and mosaic coordinate map,
    // but cannot provide reference IBS matches or reference HMM evidence.
    let shared_target_row_indices = target_row_indices
        .iter()
        .zip(&reference_row_indices)
        .map(|(&target, reference)| reference.and(target))
        .collect::<Vec<_>>();
    let union_reference_groups = reference_row_indices
        .iter()
        .map(|row| row.and_then(|index| reference_groups[index]))
        .collect::<Vec<_>>();
    let cluster_plan = build_hmm_cluster_plan_with_groups(
        &shared_target_row_indices,
        &target_hmm_cm_positions,
        &target_projection_cm_positions,
        config.cluster_cm,
        &union_reference_groups,
    );
    if cluster_plan.member_union_indices.is_empty() {
        return Err(
            "reference imputation has no shared target/reference marker clusters".to_owned(),
        );
    }
    if let Some(directory) = std::env::var_os("JANUSX_IMP_CLUSTER_TRACE_DIR") {
        use std::io::Write;
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)
            .map_err(|error| format!("cannot create cluster trace directory: {error}"))?;
        let path = directory.join(format!("window_{}.tsv", target.index));
        let mut trace =
            std::io::BufWriter::new(std::fs::File::create(&path).map_err(|error| {
                format!("cannot create cluster trace {}: {error}", path.display())
            })?);
        writeln!(trace, "cluster\tchrom\ttarget_start\ttarget_end\tref_start\tref_end\tmidpoint_bits\terr_bits\tintensity_bits")
            .map_err(|error| error.to_string())?;
        for cluster in 0..cluster_plan.cm_positions.len() {
            let members = &cluster_plan.member_union_indices[cluster];
            let targets = &cluster_plan.member_target_indices[cluster];
            let err = ((hmm_config.err * members.len() as f64).min(0.5)) as f32;
            writeln!(
                trace,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:016x}\t{:08x}\t{:08x}",
                cluster,
                target.chrom,
                targets[0],
                targets.last().unwrap() + 1,
                reference_row_indices[members[0]].unwrap(),
                reference_row_indices[*members.last().unwrap()].unwrap() + 1,
                cluster_plan.cm_positions[cluster].to_bits(),
                err.to_bits(),
                imputation_hmm_config.recombination_intensity.to_bits()
            )
            .map_err(|error| error.to_string())?;
        }
        trace.flush().map_err(|error| error.to_string())?;
    }
    let ibs_index = ImpIbsReferenceIndex::new(
        &packed_reference,
        &cluster_plan,
        config.step_cm,
        config.nsteps,
        config.segment_cm,
        config.states,
        pool,
    )?;
    // State transitions and emissions are evaluated once per Beagle marker
    let sequence_index = super::projection::ReferenceSequenceIndex::new(&packed_reference, pool);
    let cluster_codes = super::cluster_codes::ClusterSequenceIndex::new(
        &packed_reference,
        &cluster_plan.member_union_indices,
        64 * 1024 * 1024,
        pool,
    );
    // State transitions and emissions are evaluated once per Beagle marker
    // cluster. Each target haplotype gets its own bounded IBS mosaic panel;
    // the full union remains available for output interpolation.

    // The packed reference has already decoded every reference genotype.
    // Re-scan the source diploid calls here would repeat O(markers × samples)
    // work; take AF from the packed bitplane counts instead.
    let reference_frequencies = reference_row_indices
        .iter()
        .enumerate()
        .map(|(marker, reference_index)| {
            reference_index.and(packed_reference_alt_frequencies[marker])
        })
        .collect::<Vec<_>>();
    let target_frequencies = metadata
        .iter()
        .enumerate()
        .map(|(marker, _)| {
            target_row_indices[marker]
                .map(|index| target_alt_frequency(&target.rows[index]))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let reference_present = reference_row_indices
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    let reference_index_elapsed =
        reference_index_started.map_or(Duration::ZERO, |started| started.elapsed());

    let output_setup_started = profile_impute.then(Instant::now);
    let mut output_rows = if emit.is_none() {
        metadata
            .iter()
            .enumerate()
            .map(|(marker, meta)| {
                make_output_row(
                    meta,
                    target_samples,
                    &config.output,
                    target_row_indices[marker].is_none(),
                )
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    // Use the same layout on disk and in the output callback. A batch/tile is
    // contiguous, so reading needs one operation per batch, not per haplotype.
    let floats_per_sample = 2
        + usize::from(config.output.dosage)
        + 2 * usize::from(config.output.allele_probabilities)
        + 3 * usize::from(config.output.genotype_probabilities);
    let bytes_per_marker = target_samples
        .saturating_mul(floats_per_sample * 4 + 1)
        .max(1);
    let tile_markers = ((32 * 1024 * 1024) / bytes_per_marker).clamp(1, 4096);
    let mut posterior_store = emit
        .as_ref()
        .map(|_| {
            super::posterior_store::PosteriorStore::new(
                posterior_layout_markers,
                tile_markers,
                target_samples * 2,
            )
        })
        .transpose()?;
    let owned = owned_output_indices(
        target,
        &target_row_indices,
        previous_emitted_pos,
        reference_emit_end_pos,
        &metadata,
    );
    let collect_dosage_statistics = config.output.dosage;
    let mut dosage_sums = if collect_dosage_statistics {
        vec![0.0f64; metadata.len()]
    } else {
        Vec::new()
    };
    let mut dosage_squares = if collect_dosage_statistics {
        vec![0.0f64; metadata.len()]
    } else {
        Vec::new()
    };
    let batch_multiplier = std::env::var("JX_IMP_SAMPLE_BATCH_MULTIPLIER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(1, 16);
    const DEFAULT_POSTERIOR_BATCH_BUDGET_BYTES: usize = 256 * 1024 * 1024;
    let posterior_batch_budget_bytes = std::env::var("JX_IMP_POSTERIOR_BATCH_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_POSTERIOR_BATCH_BUDGET_BYTES / (1024 * 1024))
        .clamp(16, 512)
        .saturating_mul(1024 * 1024);
    let posterior_bytes_per_sample = posterior_layout_markers
        .checked_mul(2 * std::mem::size_of::<f32>())
        .unwrap_or(usize::MAX)
        .max(1);
    let memory_limited_batch = (posterior_batch_budget_bytes / posterior_bytes_per_sample).max(1);
    let batch_size = pool
        .current_num_threads()
        .max(1)
        .saturating_mul(batch_multiplier)
        .min(memory_limited_batch)
        .min(target_samples);
    let output_setup_elapsed =
        output_setup_started.map_or(Duration::ZERO, |started| started.elapsed());
    let mut batch_start = 0usize;
    let mut posterior_elapsed = Duration::ZERO;
    let mut assembly_elapsed = Duration::ZERO;
    let mut hap_posterior_stats = [HaplotypePosteriorStats::default(); 2];
    while batch_start < target_samples {
        let batch_end = batch_start.saturating_add(batch_size).min(target_samples);
        let batch_haplotypes = (batch_end - batch_start)
            .checked_mul(2)
            .ok_or_else(|| "imputation batch haplotype count overflow".to_owned())?;
        let posterior_count = batch_haplotypes
            .checked_mul(posterior_layout_markers)
            .ok_or_else(|| "imputation posterior batch size overflow".to_owned())?;
        let mut batch_posteriors = vec![0.0f32; posterior_count];
        let posterior_started = profile_impute.then(Instant::now);
        let batch_stats = pool.install(|| {
            batch_posteriors
                .par_chunks_mut(posterior_layout_markers)
                .enumerate()
                .map_init(
                    || {
                        (
                            HaplotypeScratch::new(packed_reference.n_markers()),
                            Vec::with_capacity(packed_reference.n_markers()),
                        )
                    },
                    |(scratch, full_posterior), (batch_haplotype, posterior)| {
                        let task = batch_start * 2 + batch_haplotype;
                        let mut statistics = if dense_posterior_layout {
                            sample_haplotype_posterior(
                                target,
                                phased_target,
                                &target_row_indices,
                                &hmm_union_indices,
                                &packed_reference,
                                &sequence_index,
                                &cluster_codes,
                                &ibs_index,
                                &reference_major_alleles,
                                config.states,
                                &cluster_plan,
                                &merged_projection_cm_positions,
                                &imputation_hmm_config,
                                task / 2,
                                task % 2,
                                seed,
                                scratch,
                                posterior,
                            )?
                        } else {
                            full_posterior.resize(metadata.len(), 0.0);
                            full_posterior.fill(0.0);
                            let statistics = sample_haplotype_posterior(
                                target,
                                phased_target,
                                &target_row_indices,
                                &hmm_union_indices,
                                &packed_reference,
                                &sequence_index,
                                &cluster_codes,
                                &ibs_index,
                                &reference_major_alleles,
                                config.states,
                                &cluster_plan,
                                &merged_projection_cm_positions,
                                &imputation_hmm_config,
                                task / 2,
                                task % 2,
                                seed,
                                scratch,
                                full_posterior,
                            )?;
                            for (destination, &source) in
                                posterior.iter_mut().zip(&posterior_marker_indices)
                            {
                                *destination = full_posterior[source];
                            }
                            statistics
                        };
                        statistics.posterior_markers = posterior.len() as u64;
                        Ok::<_, String>(statistics)
                    },
                )
                .collect::<Result<Vec<_>, _>>()
        })?;
        for (batch_haplotype, stats) in batch_stats.into_iter().enumerate() {
            let haplotype = (batch_start * 2 + batch_haplotype) % 2;
            hap_posterior_stats[haplotype].cluster_state_updates = hap_posterior_stats[haplotype]
                .cluster_state_updates
                .saturating_add(stats.cluster_state_updates);
            hap_posterior_stats[haplotype].posterior_markers = hap_posterior_stats[haplotype]
                .posterior_markers
                .saturating_add(stats.posterior_markers);
        }
        posterior_elapsed += posterior_started.map_or(Duration::ZERO, |started| started.elapsed());
        let assembly_started = profile_impute.then(Instant::now);
        if let Some(store) = posterior_store.as_mut() {
            store.append(&batch_posteriors)?;
            assembly_elapsed += assembly_started.map_or(Duration::ZERO, |s| s.elapsed());
            batch_start = batch_end;
            continue;
        }
        let batch_statistics = assemble_sample_batch(
            &mut output_rows,
            &batch_posteriors,
            posterior_layout_markers,
            batch_start,
            target,
            phased_target,
            &target_row_indices,
            &posterior_marker_by_output,
            &reference_present,
            &reference_frequencies,
            &target_frequencies,
            orientation_flips,
            collect_dosage_statistics,
            pool,
        );
        assembly_elapsed += assembly_started.map_or(Duration::ZERO, |started| started.elapsed());
        if collect_dosage_statistics {
            for (marker, (sum, square_sum)) in batch_statistics.into_iter().enumerate() {
                dosage_sums[marker] += sum;
                dosage_squares[marker] += square_sum;
            }
        }
        batch_start = batch_end;
    }

    let finalize_started = profile_impute.then(Instant::now);
    emit_imputation_state_accounting(
        profile_impute,
        target,
        target_accounting,
        metadata.len(),
        posterior_projection_markers,
        target_samples,
        hap_posterior_stats,
    );
    if let (Some(store), Some(emit)) = (posterior_store.as_mut(), emit.as_mut()) {
        // Bound both the posterior tile and assembled GT/DS/AP/GP storage.
        let mut output_start = 0usize;
        for posterior_start in (0..posterior_layout_markers).step_by(tile_markers) {
            let posterior_end = (posterior_start + tile_markers).min(posterior_layout_markers);
            let posterior_width = posterior_end - posterior_start;
            let output_end = if posterior_end == posterior_layout_markers {
                metadata.len()
            } else {
                posterior_marker_indices[posterior_end]
            };
            let output_width = output_end - output_start;
            let tile_posteriors = store.read_tile(posterior_start, posterior_end)?;
            let tile_posterior_marker_by_output = posterior_marker_by_output
                [output_start..output_end]
                .iter()
                .map(|marker| marker.map(|posterior_marker| posterior_marker - posterior_start))
                .collect::<Vec<_>>();
            let mut rows = metadata[output_start..output_end]
                .iter()
                .enumerate()
                .map(|(m, meta)| {
                    make_output_row(
                        meta,
                        target_samples,
                        &config.output,
                        target_row_indices[output_start + m].is_none(),
                    )
                })
                .collect::<Vec<_>>();
            let mut sums = vec![(0.0, 0.0); output_width];
            // Preserve the original batch summation order for AF and DR2.
            for sample_start in (0..target_samples).step_by(batch_size) {
                let sample_end = (sample_start + batch_size).min(target_samples);
                let stats = assemble_sample_batch(
                    &mut rows,
                    &tile_posteriors
                        [sample_start * 2 * posterior_width..sample_end * 2 * posterior_width],
                    posterior_width,
                    sample_start,
                    target,
                    phased_target,
                    &target_row_indices[output_start..output_end],
                    &tile_posterior_marker_by_output,
                    &reference_present[output_start..output_end],
                    &reference_frequencies[output_start..output_end],
                    &target_frequencies[output_start..output_end],
                    orientation_flips,
                    collect_dosage_statistics,
                    pool,
                );
                for (sum, (x, y)) in sums.iter_mut().zip(stats) {
                    sum.0 += x;
                    sum.1 += y;
                }
            }
            if collect_dosage_statistics {
                for (row, (sum, square_sum)) in rows.iter_mut().zip(sums) {
                    let mean_ds = sum / target_samples as f64;
                    let variance = (square_sum / target_samples as f64 - mean_ds.powi(2)).max(0.0);
                    let af = (mean_ds / 2.0).clamp(0.0, 1.0) as f32;
                    let expected = 2.0 * af as f64 * (1.0 - af as f64);
                    row.info_af = Some(af);
                    row.info_dr2 = Some(if expected > f64::EPSILON {
                        (variance / expected).clamp(0.0, 1.0) as f32
                    } else {
                        0.0
                    });
                }
            }
            let first = owned.partition_point(|&m| m < output_start);
            let last = owned.partition_point(|&m| m < output_end);
            let indices = owned[first..last]
                .iter()
                .map(|m| m - output_start)
                .collect::<Vec<_>>();
            emit(&rows, &indices)?;
            output_start = output_end;
        }
        debug_assert_eq!(output_start, metadata.len());
        if let (Some(profile_started), Some(finalize_started)) = (profile_started, finalize_started)
        {
            eprintln!(
                "JANUSX_IMP_IMPUTE_PROFILE window={}:{} markers={} posterior_projection_markers={posterior_projection_markers} direct_target_markers={direct_target_markers} target_samples={} batch_size={batch_size} reference_index_s={:.6} output_setup_s={:.6} posterior_s={:.6} assembly_s={:.6} finalize_s={:.6} total_s={:.6} streamed=true",
                target.chrom,
                target.index,
                metadata.len(),
                target_samples,
                reference_index_elapsed.as_secs_f64(),
                output_setup_elapsed.as_secs_f64(),
                posterior_elapsed.as_secs_f64(),
                assembly_elapsed.as_secs_f64(),
                finalize_started.elapsed().as_secs_f64(),
                profile_started.elapsed().as_secs_f64(),
            );
        }
        return Ok(ImputedWindow { rows: Vec::new() });
    }
    if collect_dosage_statistics {
        for marker in 0..metadata.len() {
            let mean_ds = dosage_sums[marker] / target_samples as f64;
            let variance =
                (dosage_squares[marker] / target_samples as f64 - mean_ds.powi(2)).max(0.0);
            let estimated_af = (mean_ds / 2.0).clamp(0.0, 1.0) as f32;
            let expected_variance = 2.0 * estimated_af as f64 * (1.0 - estimated_af as f64);
            output_rows[marker].info_af = Some(estimated_af);
            output_rows[marker].info_dr2 = Some(if expected_variance > f64::EPSILON {
                (variance / expected_variance).clamp(0.0, 1.0) as f32
            } else {
                0.0
            });
        }
    }
    if let (Some(profile_started), Some(finalize_started)) = (profile_started, finalize_started) {
        eprintln!(
            "JANUSX_IMP_IMPUTE_PROFILE window={}:{} markers={} posterior_projection_markers={posterior_projection_markers} direct_target_markers={direct_target_markers} target_samples={} batch_size={batch_size} reference_index_s={:.6} output_setup_s={:.6} posterior_s={:.6} assembly_s={:.6} finalize_s={:.6} total_s={:.6}",
            target.chrom,
            target.index,
            metadata.len(),
            target_samples,
            reference_index_elapsed.as_secs_f64(),
            output_setup_elapsed.as_secs_f64(),
            posterior_elapsed.as_secs_f64(),
            assembly_elapsed.as_secs_f64(),
            finalize_started.elapsed().as_secs_f64(),
            profile_started.elapsed().as_secs_f64(),
        );
    }
    Ok(ImputedWindow { rows: output_rows })
}

fn rescale_hmm_config_haplotype_count(
    config: &mut HMMConfig,
    new_haplotype_count: usize,
) -> Result<(), String> {
    if config.haplotype_count == 0 || new_haplotype_count == 0 {
        return Err("HMM haplotype counts must be positive when rescaling".to_owned());
    }
    config.recombination_intensity = (f64::from(config.recombination_intensity)
        * config.haplotype_count as f64
        / new_haplotype_count as f64) as f32;
    config.haplotype_count = new_haplotype_count;
    Ok(())
}

fn imputation_hmm_config(
    phase_config: &HMMConfig,
    reference_haplotypes: usize,
) -> Result<HMMConfig, String> {
    let mut result = *phase_config;
    rescale_hmm_config_haplotype_count(&mut result, reference_haplotypes)?;
    Ok(result)
}

#[cfg(test)]
mod reference_hmm_config_tests {
    use super::rescale_hmm_config_haplotype_count;
    use crate::imp::types::HMMConfig;

    #[test]
    fn reference_haplotype_rescaling_preserves_unrounded_ne_intensity() {
        let mut config = HMMConfig {
            ne: 206,
            recombination_intensity: 0.013_746_13,
            err: 0.01,
            em: false,
            burnin: 3,
            iterations: 12,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 600,
        };
        let expected_rate = f64::from(config.recombination_intensity) * 600.0;

        rescale_hmm_config_haplotype_count(&mut config, 1_600).unwrap();

        let observed_rate = f64::from(config.recombination_intensity) * 1_600.0;
        assert!((observed_rate - expected_rate).abs() < 1e-6);
        assert_eq!(config.haplotype_count, 1_600);
        assert_eq!(config.ne, 206);
    }
}

#[derive(Clone, Copy, Default)]
struct HaplotypePosteriorStats {
    cluster_state_updates: u64,
    posterior_markers: u64,
}

#[derive(Clone, Copy, Default)]
struct TargetMarkerAccounting {
    observed_gt: u64,
    masked_gt: u64,
    partial_gt: u64,
    observed_markers: u64,
    fully_masked_markers: u64,
}

fn target_marker_accounting(target: &PhaseWindow) -> TargetMarkerAccounting {
    let mut accounting = TargetMarkerAccounting::default();
    for row in &target.rows {
        let marker_observed = row
            .calls
            .iter()
            .any(|call| call.alleles.iter().any(Option::is_some));
        if marker_observed {
            accounting.observed_markers += 1;
        } else {
            accounting.fully_masked_markers += 1;
        }
        for call in row.calls.iter() {
            match call.alleles {
                [Some(_), Some(_)] => accounting.observed_gt += 1,
                [None, None] => accounting.masked_gt += 1,
                _ => accounting.partial_gt += 1,
            }
        }
    }
    accounting
}

#[allow(clippy::too_many_arguments)]
fn emit_imputation_state_accounting(
    enabled: bool,
    target: &PhaseWindow,
    accounting: TargetMarkerAccounting,
    output_markers: usize,
    reference_only_markers: usize,
    target_samples: usize,
    haplotype_stats: [HaplotypePosteriorStats; 2],
) {
    if !enabled {
        return;
    }
    let haplotypes = if reference_only_markers == 0 {
        0
    } else {
        target_samples.saturating_mul(2)
    };
    eprintln!(
        "JANUSX_IMP_STATE_ACCOUNTING_PROFILE window={}:{} target_markers={} target_observed_markers={} target_fully_masked_markers={} target_observed_gt={} target_masked_gt={} target_partial_gt={} output_markers={} reference_only_markers={} imputation_haplotypes={} hap0_cluster_state_updates={} hap1_cluster_state_updates={} hap0_posterior_markers={} hap1_posterior_markers={} shared_alignment_passes=1 shared_cluster_plans=1 shared_emission_code_indices=1",
        target.chrom,
        target.index,
        target.rows.len(),
        accounting.observed_markers,
        accounting.fully_masked_markers,
        accounting.observed_gt,
        accounting.masked_gt,
        accounting.partial_gt,
        output_markers,
        reference_only_markers,
        haplotypes,
        haplotype_stats[0].cluster_state_updates,
        haplotype_stats[1].cluster_state_updates,
        haplotype_stats[0].posterior_markers,
        haplotype_stats[1].posterior_markers,
    );
}

struct HaplotypeScratch {
    query: Vec<Option<u16>>,
    cluster_errors: Vec<f64>,
    workspace: super::types::HMMWorkspace,
}

impl HaplotypeScratch {
    fn new(n_reference_markers: usize) -> Self {
        Self {
            query: Vec::with_capacity(n_reference_markers),
            cluster_errors: Vec::new(),
            workspace: Default::default(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_haplotype_posterior(
    target: &PhaseWindow,
    phased_target: &PhasedWindow,
    target_row_indices: &[Option<usize>],
    hmm_union_indices: &[usize],
    packed_reference: &composite::PackedBiallelicReference,
    sequence_index: &super::projection::ReferenceSequenceIndex,
    cluster_codes: &super::cluster_codes::ClusterSequenceIndex,
    ibs_index: &ImpIbsReferenceIndex,
    reference_major_alleles: &[Option<u16>],
    max_states: usize,
    cluster_plan: &HmmClusterPlan,
    union_cm_positions: &[f64],
    config: &HMMConfig,
    sample: usize,
    haplotype: usize,
    seed: u64,
    scratch: &mut HaplotypeScratch,
    posterior: &mut [f32],
) -> Result<HaplotypePosteriorStats, String> {
    debug_assert!(!hmm_union_indices.is_empty());
    debug_assert!(hmm_union_indices
        .iter()
        .all(|&marker| marker < target_row_indices.len()));
    debug_assert!(hmm_union_indices.windows(2).all(|pair| pair[1] > pair[0]));
    debug_assert!(!cluster_plan.member_union_indices.is_empty());
    debug_assert_eq!(
        cluster_plan.member_union_indices.len(),
        cluster_plan.member_target_indices.len()
    );
    debug_assert_eq!(
        cluster_plan.member_union_indices.len(),
        cluster_plan.cm_positions.len()
    );
    debug_assert_eq!(
        cluster_plan.member_union_indices.len(),
        cluster_plan.cluster_end_cm_positions.len()
    );
    debug_assert_eq!(packed_reference.n_markers(), union_cm_positions.len());
    debug_assert_eq!(reference_major_alleles.len(), packed_reference.n_markers());
    debug_assert!(max_states > 0);
    debug_assert!(haplotype <= 1);
    debug_assert_eq!(posterior.len(), union_cm_positions.len());
    let profile_haplotypes = std::env::var_os("JANUSX_IMP_DETAIL_PROFILE").is_some();
    let total_started = profile_haplotypes.then(Instant::now);
    let query_started = total_started;
    target_query_alleles_from_phased_target_into(
        phased_target,
        target_row_indices,
        hmm_union_indices,
        reference_major_alleles,
        sample,
        haplotype,
        &mut scratch.query,
    )?;
    let query_elapsed = query_started.map_or(Duration::ZERO, |started| started.elapsed());
    let mosaic_started = profile_haplotypes.then(Instant::now);
    let ibs_candidates = ibs_index.candidate_sets(packed_reference, &scratch.query, seed);
    let full_mosaic = super::ibs_mosaic::IbsTargetMosaic::new(
        packed_reference,
        &ibs_index.marker_step_ranges,
        &ibs_candidates,
        max_states,
        sample * 2 + haplotype,
        seed,
    )?;
    let state_count = full_mosaic.n_states();
    let cluster_state_updates = u64::try_from(cluster_plan.member_union_indices.len())
        .ok()
        .and_then(|clusters| clusters.checked_mul(state_count as u64))
        .ok_or_else(|| "imputation state-update count overflow".to_owned())?;
    let posterior_markers = u64::try_from(union_cm_positions.len())
        .map_err(|_| "imputation posterior-marker count overflow".to_owned())?;
    let mosaic_elapsed = mosaic_started.map_or(Duration::ZERO, |started| started.elapsed());
    let cluster_started = profile_haplotypes.then(Instant::now);
    let (cluster_observations, cluster_states, cluster_errors) = build_ibs_cluster_hmm_inputs(
        &full_mosaic,
        packed_reference,
        &scratch.query,
        cluster_plan,
        config.err,
        Some(cluster_codes),
    )?;
    scratch.cluster_errors = cluster_errors;
    let cluster_elapsed = cluster_started.map_or(Duration::ZERO, |started| started.elapsed());
    let hmm_started = profile_haplotypes.then(Instant::now);
    interpolated_haplotype_posteriors(
        &cluster_observations,
        &cluster_states,
        &full_mosaic,
        packed_reference,
        cluster_plan,
        &scratch.cluster_errors,
        Some(sequence_index),
        target,
        target_row_indices,
        sample,
        haplotype,
        union_cm_positions,
        config,
        &mut scratch.workspace,
        posterior,
    )?;
    let hmm_elapsed = hmm_started.map_or(Duration::ZERO, |started| started.elapsed());
    if let Some(started) = total_started {
        eprintln!(
            "JANUSX_IMP_HAPLOTYPE_PROFILE window={}:{} sample={sample} haplotype={haplotype} query_s={:.6} mosaic_s={:.6} cluster_s={:.6} hmm_s={:.6} total_s={:.6}",
            target.chrom,
            target.index,
            query_elapsed.as_secs_f64(),
            mosaic_elapsed.as_secs_f64(),
            cluster_elapsed.as_secs_f64(),
            hmm_elapsed.as_secs_f64(),
            started.elapsed().as_secs_f64(),
        );
    }
    Ok(HaplotypePosteriorStats {
        cluster_state_updates,
        posterior_markers,
    })
}

#[cfg(test)]
fn target_query_alleles_from_phased_target(
    phased_target: &PhasedWindow,
    target_row_indices: &[Option<usize>],
    target_marker_indices: &[usize],
    reference_major_alleles: &[Option<u16>],
    sample: usize,
    haplotype: usize,
) -> Result<Vec<Option<u16>>, String> {
    let mut query = Vec::with_capacity(reference_major_alleles.len());
    target_query_alleles_from_phased_target_into(
        phased_target,
        target_row_indices,
        target_marker_indices,
        reference_major_alleles,
        sample,
        haplotype,
        &mut query,
    )?;
    Ok(query)
}

fn target_query_alleles_from_phased_target_into(
    phased_target: &PhasedWindow,
    target_row_indices: &[Option<usize>],
    target_marker_indices: &[usize],
    reference_major_alleles: &[Option<u16>],
    sample: usize,
    haplotype: usize,
    query: &mut Vec<Option<u16>>,
) -> Result<(), String> {
    debug_assert_eq!(target_row_indices.len(), reference_major_alleles.len());
    debug_assert!(!target_marker_indices.is_empty());
    debug_assert!(haplotype <= 1);
    debug_assert!(target_marker_indices
        .iter()
        .all(|&marker| marker < reference_major_alleles.len()));
    debug_assert!(target_marker_indices
        .windows(2)
        .all(|pair| pair[1] > pair[0]));
    query.clear();
    query.extend_from_slice(reference_major_alleles);
    for &union_marker in target_marker_indices {
        let target_index = target_row_indices[union_marker]
            .ok_or_else(|| "phased target query marker has no target row".to_owned())?;
        let call = phased_target
            .rows
            .get(target_index)
            .and_then(|row| row.calls.get(sample))
            .ok_or_else(|| "phased target query sample is out of range".to_owned())?;
        query[union_marker] = call.alleles[haplotype];
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn target_known_haplotype_alleles_into(
    target: &PhaseWindow,
    phased_target: &PhasedWindow,
    target_row_indices: &[Option<usize>],
    target_marker_indices: &[usize],
    sample: usize,
    haplotype: usize,
    orientation_flip: bool,
    observed: &mut Vec<Option<u16>>,
) -> Result<(), String> {
    if haplotype > 1
        || target_marker_indices
            .iter()
            .any(|&marker| marker >= target_row_indices.len())
        || target_marker_indices
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
    {
        return Err("target observed-haplotype mapping is inconsistent".to_owned());
    }
    observed.clear();
    observed.resize(target_row_indices.len(), None);
    for &union_marker in target_marker_indices {
        let target_index = target_row_indices[union_marker]
            .ok_or_else(|| "phased target observation marker has no target row".to_owned())?;
        let raw = target
            .rows
            .get(target_index)
            .and_then(|row| row.calls.get(sample))
            .ok_or_else(|| "target observation sample is out of range".to_owned())?;
        let phased = phased_target
            .rows
            .get(target_index)
            .and_then(|row| row.calls.get(sample))
            .ok_or_else(|| "phased target observation sample is out of range".to_owned())?;
        let known_allele = match raw.alleles {
            [Some(_), Some(_)] => {
                observed[union_marker] = phased.alleles[haplotype];
                continue;
            }
            [Some(allele), None] | [None, Some(allele)] if raw.phased => {
                let known_haplotype = if matches!(raw.alleles, [Some(_), None]) {
                    usize::from(orientation_flip)
                } else {
                    1 ^ usize::from(orientation_flip)
                };
                (haplotype == known_haplotype).then_some(allele)
            }
            [Some(allele), None] | [None, Some(allele)] => {
                (haplotype == usize::from(orientation_flip)).then_some(allele)
            }
            [None, None] => continue,
        };
        observed[union_marker] = known_allele;
    }
    Ok(())
}

fn sparse_active_states(
    current: &[f32],
    right: Option<&[f32]>,
    threshold: f32,
    active: &mut Vec<usize>,
) {
    active.clear();
    debug_assert!(right.is_none_or(|right| right.len() == current.len()));

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 availability was checked above, and the vectorized
        // interior loop stays within both probability slices.
        unsafe {
            sparse_active_states_avx2(current, right, threshold, active);
        }
        return;
    }

    for (state, &probability) in current.iter().enumerate() {
        if probability > threshold
            || right.is_some_and(|right| right.get(state).copied().unwrap_or(0.0) > threshold)
        {
            active.push(state);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sparse_active_states_avx2(
    current: &[f32],
    right: Option<&[f32]>,
    threshold: f32,
    active: &mut Vec<usize>,
) {
    use std::arch::x86_64::*;

    unsafe {
        let threshold = _mm256_set1_ps(threshold);
        let vector_end = current.len() / 8 * 8;
        let mut state = 0usize;
        while state < vector_end {
            let current_values = _mm256_loadu_ps(current.as_ptr().add(state));
            let mut selected = _mm256_cmp_ps::<_CMP_GT_OQ>(current_values, threshold);
            if let Some(right) = right {
                let right_values = _mm256_loadu_ps(right.as_ptr().add(state));
                let right_selected = _mm256_cmp_ps::<_CMP_GT_OQ>(right_values, threshold);
                selected = _mm256_or_ps(selected, right_selected);
            }
            let mut bits = _mm256_movemask_ps(selected) as u32;
            while bits != 0 {
                let offset = bits.trailing_zeros() as usize;
                active.push(state + offset);
                bits &= bits - 1;
            }
            state += 8;
        }
        for state in vector_end..current.len() {
            if current[state] > _mm256_cvtss_f32(threshold)
                || right.is_some_and(|right| right[state] > _mm256_cvtss_f32(threshold))
            {
                active.push(state);
            }
        }
    }
}

#[inline]
fn retained_state_projection_required(active_states: usize, state_count: usize) -> bool {
    active_states <= state_count && state_count > 0
}

fn sparse_panel_allele_probability(
    panel: &CompositePanel,
    marker: usize,
    state_probabilities: &[f32],
    active_states: &[usize],
    active_weights: &[f32],
    total_weight: f32,
    allele: u16,
) -> f32 {
    let mut allele_weight = 0.0f32;
    if let Some((allele_words, missing_words)) = panel.haplotypes.biallelic_marker_planes(marker) {
        for (&state, &weight) in active_states.iter().zip(active_weights) {
            if packed_state_allele(missing_words, state) == 1 {
                allele_weight += weight * 0.5;
            } else if packed_state_allele(allele_words, state) == allele {
                allele_weight += weight;
            }
        }
    } else {
        for (&state, &weight) in active_states.iter().zip(active_weights) {
            match panel.haplotypes.allele(marker, state) {
                Some(observed) if observed == allele => allele_weight += weight,
                Some(_) => {}
                None => allele_weight += weight * 0.5,
            }
        }
    }
    if total_weight > f32::MIN_POSITIVE {
        (allele_weight / total_weight).clamp(0.0, 1.0)
    } else {
        panel_allele_probability(panel, marker, state_probabilities, allele)
    }
}

#[inline]
fn packed_state_allele(words: &[u64], state: usize) -> u16 {
    ((words[state >> 6] >> (state & 63)) & 1) as u16
}

/// Project only retained states, preserving the packed panel's accumulation
/// order and missing-allele contribution. Tiny totals require its dense fallback.
fn sparse_typed_mosaic_probability(
    mosaic: &super::ibs_mosaic::IbsTargetMosaic,
    reference: &composite::PackedBiallelicReference,
    marker: usize,
    active_states: &[usize],
    active_weights: &[f32],
    total_weight: f32,
) -> Option<f32> {
    if !(total_weight > f32::MIN_POSITIVE) {
        return None;
    }
    let mut allele_weight = 0.0f32;
    for (&state, &weight) in active_states.iter().zip(active_weights) {
        match reference.allele_unchecked(marker, mosaic.donor_at(marker, state)) {
            Some(1) => allele_weight += weight,
            None => allele_weight += weight * 0.5,
            _ => {}
        }
    }
    Some((allele_weight / total_weight).clamp(0.0, 1.0))
}

fn sparse_complete_mosaic_probability(
    reference_words: &[u64],
    active_donors: &[usize],
    active_weights: &[f32],
    total_weight: f32,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 availability was checked above, and both slices have
        // matching lengths by construction.
        return unsafe {
            sparse_complete_mosaic_probability_avx2(
                reference_words,
                active_donors,
                active_weights,
                total_weight,
            )
        };
    }

    let mut allele_weight = 0.0f32;
    for (&donor, &weight) in active_donors.iter().zip(active_weights) {
        if packed_state_allele(reference_words, donor) == 1 {
            allele_weight += weight;
        }
    }
    if total_weight > f32::MIN_POSITIVE {
        (allele_weight / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sparse_complete_mosaic_probability_avx2(
    reference_words: &[u64],
    active_donors: &[usize],
    active_weights: &[f32],
    total_weight: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let mut allele_weight = 0.0f32;
    let vector_end = active_donors.len() / 4 * 4;
    unsafe {
        let shift_mask = _mm256_set1_epi64x(63);
        let one = _mm256_set1_epi64x(1);
        let mut state = 0usize;
        while state < vector_end {
            let donor_indices =
                _mm256_loadu_si256(active_donors.as_ptr().add(state).cast::<__m256i>());
            let reference_word_indices = _mm256_srli_epi64(donor_indices, 6);
            let bit_offsets = _mm256_and_si256(donor_indices, shift_mask);
            let reference_values = _mm256_i64gather_epi64::<8>(
                reference_words.as_ptr().cast::<i64>(),
                reference_word_indices,
            );
            let allele_values =
                _mm256_and_si256(_mm256_srlv_epi64(reference_values, bit_offsets), one);
            let allele_mask = _mm256_cmpeq_epi64(allele_values, one);
            let allele_bits = _mm256_movemask_pd(_mm256_castsi256_pd(allele_mask)) as usize;
            for offset in 0..4 {
                if allele_bits & (1 << offset) != 0 {
                    allele_weight += active_weights[state + offset];
                }
            }
            state += 4;
        }
        for state in vector_end..active_donors.len() {
            if packed_state_allele(reference_words, active_donors[state]) == 1 {
                allele_weight += active_weights[state];
            }
        }
    }
    if total_weight > f32::MIN_POSITIVE {
        (allele_weight / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn interpolated_sparse_complete_mosaic_probability(
    reference_words: &[u64],
    active_donors: &[usize],
    left_weights: &[f32],
    right_weights: &[f32],
    alpha: f32,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 availability was checked above, and all slices have
        // matching lengths by construction.
        return unsafe {
            interpolated_sparse_complete_mosaic_probability_avx2(
                reference_words,
                active_donors,
                left_weights,
                right_weights,
                alpha,
            )
        };
    }

    let left_weight = 1.0 - alpha;
    let mut total_weight = 0.0f32;
    let mut allele_weight = 0.0f32;
    for ((&donor, &left), &right) in active_donors.iter().zip(left_weights).zip(right_weights) {
        let weight = left_weight * left + alpha * right;
        total_weight += weight;
        if packed_state_allele(reference_words, donor) == 1 {
            allele_weight += weight;
        }
    }
    if total_weight > f32::MIN_POSITIVE {
        (allele_weight / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interpolated_sparse_complete_mosaic_probability_avx2(
    reference_words: &[u64],
    active_donors: &[usize],
    left_weights: &[f32],
    right_weights: &[f32],
    alpha: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let mut total_weight = 0.0f32;
    let mut allele_weight = 0.0f32;
    let vector_end = active_donors.len() / 4 * 4;
    let left_scale_scalar = 1.0 - alpha;
    let alpha_scalar = alpha;
    unsafe {
        let shift_mask = _mm256_set1_epi64x(63);
        let one = _mm256_set1_epi64x(1);
        let left_scale = _mm256_set1_ps(left_scale_scalar);
        let alpha = _mm256_set1_ps(alpha_scalar);
        let mut state = 0usize;
        while state < vector_end {
            let donor_indices =
                _mm256_loadu_si256(active_donors.as_ptr().add(state).cast::<__m256i>());
            let reference_word_indices = _mm256_srli_epi64(donor_indices, 6);
            let bit_offsets = _mm256_and_si256(donor_indices, shift_mask);
            let reference_values = _mm256_i64gather_epi64::<8>(
                reference_words.as_ptr().cast::<i64>(),
                reference_word_indices,
            );
            let allele_values =
                _mm256_and_si256(_mm256_srlv_epi64(reference_values, bit_offsets), one);
            let allele_mask = _mm256_cmpeq_epi64(allele_values, one);
            let left = _mm_loadu_ps(left_weights.as_ptr().add(state));
            let right = _mm_loadu_ps(right_weights.as_ptr().add(state));
            let weights = _mm_add_ps(
                _mm_mul_ps(left, _mm256_castps256_ps128(left_scale)),
                _mm_mul_ps(right, _mm256_castps256_ps128(alpha)),
            );
            let mut weight_lanes = [0.0f32; 4];
            _mm_storeu_ps(weight_lanes.as_mut_ptr(), weights);
            let allele_bits = _mm256_movemask_pd(_mm256_castsi256_pd(allele_mask)) as usize;
            for lane in 0..4 {
                total_weight += weight_lanes[lane];
                if allele_bits & (1 << lane) != 0 {
                    allele_weight += weight_lanes[lane];
                }
            }
            state += 4;
        }
        for state in vector_end..active_donors.len() {
            let weight =
                left_scale_scalar * left_weights[state] + alpha_scalar * right_weights[state];
            total_weight += weight;
            if packed_state_allele(reference_words, active_donors[state]) == 1 {
                allele_weight += weight;
            }
        }
    }
    if total_weight > f32::MIN_POSITIVE {
        (allele_weight / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sparse_biallelic_words_probability(
    allele_words: &[u64],
    missing_words: &[u64],
    active_states: &[usize],
    active_weights: &[f32],
    total_weight: f32,
) -> f32 {
    let mut allele_weight = 0.0f32;
    for (&state, &weight) in active_states.iter().zip(active_weights) {
        if packed_state_allele(missing_words, state) == 1 {
            allele_weight += weight * 0.5;
        } else if packed_state_allele(allele_words, state) == 1 {
            allele_weight += weight;
        }
    }
    if total_weight > f32::MIN_POSITIVE {
        (allele_weight / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn interpolated_sparse_biallelic_words_probability(
    allele_words: &[u64],
    missing_words: &[u64],
    active_states: &[usize],
    left_weights: &[f32],
    right_weights: &[f32],
    alpha: f32,
) -> f32 {
    let left_weight = 1.0 - alpha;
    let mut total_weight = 0.0f32;
    let mut allele_weight = 0.0f32;
    for ((&state, &left), &right) in active_states.iter().zip(left_weights).zip(right_weights) {
        let weight = left_weight * left + alpha * right;
        total_weight += weight;
        if packed_state_allele(missing_words, state) == 1 {
            allele_weight += weight * 0.5;
        } else if packed_state_allele(allele_words, state) == 1 {
            allele_weight += weight;
        }
    }
    if total_weight > f32::MIN_POSITIVE {
        (allele_weight / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn interpolated_haplotype_posteriors(
    observations: &PackedHaplotypes,
    hmm_states: &CompositePanel,
    mosaic: &super::ibs_mosaic::IbsTargetMosaic,
    packed_reference: &composite::PackedBiallelicReference,
    cluster_plan: &HmmClusterPlan,
    marker_errors: &[f64],
    sequence_index: Option<&super::projection::ReferenceSequenceIndex>,
    target: &PhaseWindow,
    target_row_indices: &[Option<usize>],
    sample: usize,
    haplotype: usize,
    union_cm_positions: &[f64],
    config: &HMMConfig,
    workspace: &mut super::types::HMMWorkspace,
    allele_probabilities: &mut [f32],
) -> Result<(), String> {
    if allele_probabilities.len() != union_cm_positions.len() {
        return Err("target HMM posterior buffer has the wrong marker count".to_owned());
    }
    if cluster_plan.member_union_indices.len() != observations.n_markers
        || cluster_plan.member_union_indices.len() != hmm_states.marker_indices.len()
        || hmm_states.haplotypes.n_markers != observations.n_markers
        || hmm_states.haplotypes.n_haplotypes != mosaic.n_states()
        || cluster_plan.member_union_indices.len() != cluster_plan.member_target_indices.len()
        || cluster_plan.member_union_indices.len() != cluster_plan.cm_positions.len()
        || cluster_plan.member_union_indices.len() != cluster_plan.cluster_end_cm_positions.len()
        || cluster_plan.member_union_indices.len() != marker_errors.len()
        || target_row_indices.len() != union_cm_positions.len()
        || cluster_plan
            .member_union_indices
            .iter()
            .flatten()
            .any(|&marker| marker >= union_cm_positions.len())
        || cluster_plan
            .member_target_indices
            .iter()
            .flatten()
            .any(|&marker| marker >= target.rows.len())
        || cluster_plan
            .member_union_indices
            .iter()
            .any(|members| members.is_empty())
        || cluster_plan
            .member_union_indices
            .windows(2)
            .any(|pair| pair[0].last() >= pair[1].first())
    {
        return Err("target HMM clusters do not map to increasing union markers".to_owned());
    }
    let state_count = mosaic.n_states();
    use std::io::Write;
    let trace_hap = std::env::var("JANUSX_IMP_CLUSTER_TRACE_HAP")
        .ok()
        .map(|value| {
            value.parse::<usize>().map_err(|_| {
                "JANUSX_IMP_CLUSTER_TRACE_HAP must be a nonnegative integer".to_owned()
            })
        })
        .transpose()?;
    let mut model_trace = if trace_hap == Some(sample * 2 + haplotype) {
        let directory = std::env::var_os("JANUSX_IMP_CLUSTER_TRACE_DIR").ok_or_else(|| {
            "JANUSX_IMP_CLUSTER_TRACE_HAP requires JANUSX_IMP_CLUSTER_TRACE_DIR".to_owned()
        })?;
        let path = std::path::PathBuf::from(directory).join(format!(
            "model_{}_hap_{}.tsv",
            target.index,
            sample * 2 + haplotype,
        ));
        let mut writer =
            std::io::BufWriter::new(std::fs::File::create(&path).map_err(|error| {
                format!("cannot create model trace {}: {error}", path.display())
            })?);
        writeln!(writer, "cluster\tstate\tdonor\tmismatch\tposterior_bits")
            .map_err(|error| error.to_string())?;
        Some(writer)
    } else {
        None
    };
    let mut projection_panel = CompositePanel {
        haplotypes: PackedHaplotypes::empty(state_count, 1, 1)?,
        source_samples: Vec::new(),
        marker_indices: vec![0],
        allele_counts: vec![2],
    };
    let mosaic_word_count = state_count.div_ceil(64);
    let mut mosaic_allele_words = vec![0u64; mosaic_word_count];
    let mut mosaic_missing_words = vec![0u64; mosaic_word_count];
    let mut right_hmm_marker: Option<usize> = None;
    let mut right_probabilities = vec![0.0f32; state_count];
    let state_threshold = 0.005f32.min(0.9999f32 / state_count as f32);
    let mut active_states = Vec::with_capacity(state_count);
    let mut active_weights = Vec::with_capacity(state_count);
    let mut right_active_weights = Vec::with_capacity(state_count);
    let mut active_donors = Vec::with_capacity(state_count);
    let all_states = (0..state_count).collect::<Vec<_>>();
    let mut rightmost_endpoint: Option<(Vec<f32>, Vec<usize>, Vec<f32>, f32)> = None;
    let mut sequence_scratch = super::projection::SequenceProjectionScratch::default();
    let mut left_endpoint: Option<(Vec<f32>, Vec<usize>, Vec<f32>, f32)> = None;
    let profile_projection = std::env::var_os("JANUSX_IMP_DETAIL_PROFILE").is_some();
    let mut projection_fused_elapsed = Duration::ZERO;
    let mut projection_observed_elapsed = Duration::ZERO;
    let mut projection_donor_lookup_elapsed = Duration::ZERO;
    let mut projection_pack_elapsed = Duration::ZERO;
    let mut projection_probability_elapsed = Duration::ZERO;
    let mut complete_gap_markers = 0usize;
    let mut incomplete_gap_markers = 0usize;
    let mut skipped_target_markers = 0usize;
    let mut active_state_sum = 0usize;
    let mut retained_projection_cluster_count = 0usize;
    hmm::forward_backward_dense_with_state_callback_and_errors(
        observations,
        hmm_states,
        &cluster_plan.cm_positions,
        config,
        marker_errors,
        workspace,
        |cluster, state_probabilities| {
            let cluster_members = &cluster_plan.member_union_indices[cluster];
            if let Some(trace) = model_trace.as_mut() {
                for (state, &probability) in state_probabilities.iter().enumerate() {
                    writeln!(
                        trace,
                        "{}\t{}\t{}\t{}\t{:08x}",
                        cluster,
                        state,
                        mosaic.donor_at(cluster_members[0], state),
                        hmm_states.haplotypes.allele(cluster, state).unwrap_or(0),
                        probability.to_bits()
                    )
                    .map_err(|error| error.to_string())?;
                }
            }
            sparse_active_states(
                state_probabilities,
                right_hmm_marker.map(|_| right_probabilities.as_slice()),
                state_threshold,
                &mut active_states,
            );
            // Beagle prunes low-probability states before projecting allele
            // marginals. Keep this state set identical regardless of whether
            // the active fraction would make a dense kernel faster.
            let use_retained_projection =
                retained_state_projection_required(active_states.len(), state_count);
            active_weights.clear();
            active_weights.extend(
                active_states
                    .iter()
                    .map(|&state| state_probabilities[state]),
            );
            right_active_weights.clear();
            if right_hmm_marker.is_some() {
                right_active_weights.extend(
                    active_states
                        .iter()
                        .map(|&state| right_probabilities[state]),
                );
            }
            let active_total_weight = if use_retained_projection {
                active_weights.iter().copied().sum::<f32>()
            } else {
                0.0
            };
            if profile_projection {
                active_state_sum += active_states.len();
                retained_projection_cluster_count += usize::from(use_retained_projection);
            }
            let cluster_start = cluster_members[0];
            let cluster_end = cluster_members
                .last()
                .copied()
                .and_then(|marker| marker.checked_add(1))
                .ok_or_else(|| "target HMM cluster end overflows".to_owned())?;
            for &union_marker in cluster_members {
                // Every marker present in the target VCF is restored from the
                // phased target call during assembly, including originally
                // missing calls. Its reference posterior is never consumed.
                if target_row_indices[union_marker].is_some() {
                    skipped_target_markers += 1;
                    continue;
                }
                let started = profile_projection.then(Instant::now);
                if use_retained_projection {
                    if let Some(probability) = sparse_typed_mosaic_probability(
                        mosaic,
                        packed_reference,
                        union_marker,
                        &active_states,
                        &active_weights,
                        active_total_weight,
                    ) {
                        allele_probabilities[union_marker] = probability;
                        projection_observed_elapsed +=
                            started.map_or(Duration::ZERO, |s| s.elapsed());
                        continue;
                    }
                }
                mosaic.pack_marker(
                    packed_reference,
                    union_marker,
                    &mut projection_panel.haplotypes,
                )?;
                allele_probabilities[union_marker] = if use_retained_projection {
                    sparse_panel_allele_probability(
                        &projection_panel,
                        0,
                        state_probabilities,
                        &active_states,
                        &active_weights,
                        active_total_weight,
                        1,
                    )
                } else {
                    panel_allele_probability(&projection_panel, 0, state_probabilities, 1)
                };
                projection_observed_elapsed += started.map_or(Duration::ZERO, |s| s.elapsed());
            }
            let mut internal_marker = cluster_start;
            while internal_marker < cluster_end {
                if cluster_members.binary_search(&internal_marker).is_ok() {
                    internal_marker += 1;
                    continue;
                }
                let started = profile_projection.then(Instant::now);
                let segment_end = mosaic.active_donor_run_at(
                    internal_marker,
                    if use_retained_projection {
                        &active_states
                    } else {
                        &all_states
                    },
                    &mut active_donors,
                )?;
                projection_donor_lookup_elapsed += started.map_or(Duration::ZERO, |s| s.elapsed());
                let marker_end = segment_end.min(cluster_end);
                let donors = active_donors.as_slice();
                let target_owned = target_row_indices[internal_marker..marker_end]
                    .iter()
                    .any(Option::is_some);
                if !target_owned && use_retained_projection && marker_end - internal_marker >= 8 {
                    if sequence_index.is_some_and(|index| {
                        index.project_run(
                            internal_marker,
                            marker_end,
                            &active_donors,
                            &active_weights,
                            None,
                            union_cm_positions,
                            0.0,
                            0.0,
                            allele_probabilities,
                            &mut sequence_scratch,
                        )
                    }) {
                        internal_marker = marker_end;
                        continue;
                    }
                }
                let mut next_reference_words =
                    packed_reference.complete_marker_words(internal_marker);
                for union_index in internal_marker..marker_end {
                    if cluster_members.binary_search(&union_index).is_ok() {
                        next_reference_words =
                            packed_reference.complete_marker_words(union_index + 1);
                        continue;
                    }
                    let following_reference_words = (union_index + 1 < marker_end)
                        .then(|| packed_reference.complete_marker_words(union_index + 1))
                        .flatten();
                    if target_row_indices[union_index].is_some() {
                        skipped_target_markers += 1;
                        next_reference_words = following_reference_words;
                        continue;
                    }
                    allele_probabilities[union_index] =
                        if let Some(reference_words) = next_reference_words.take() {
                            if let Some(following_reference_words) = following_reference_words {
                                prefetch_complete_marker_words(following_reference_words);
                            }
                            complete_gap_markers += 1;
                            if profile_projection {
                                let started = Instant::now();
                                let probability = if use_retained_projection {
                                    sparse_complete_mosaic_probability(
                                        reference_words,
                                        &active_donors,
                                        &active_weights,
                                        active_total_weight,
                                    )
                                } else {
                                    interpolated_complete_mosaic_probability(
                                        reference_words,
                                        donors,
                                        state_probabilities,
                                        state_probabilities,
                                        0.0,
                                        1,
                                    )
                                };
                                projection_fused_elapsed += started.elapsed();
                                probability
                            } else if use_retained_projection {
                                sparse_complete_mosaic_probability(
                                    reference_words,
                                    &active_donors,
                                    &active_weights,
                                    active_total_weight,
                                )
                            } else {
                                interpolated_complete_mosaic_probability(
                                    reference_words,
                                    donors,
                                    state_probabilities,
                                    state_probabilities,
                                    0.0,
                                    1,
                                )
                            }
                        } else {
                            incomplete_gap_markers += 1;
                            mosaic.pack_marker_words(
                                packed_reference,
                                union_index,
                                &mut mosaic_allele_words,
                                &mut mosaic_missing_words,
                            )?;
                            if use_retained_projection {
                                sparse_biallelic_words_probability(
                                    &mosaic_allele_words,
                                    &mosaic_missing_words,
                                    &active_states,
                                    &active_weights,
                                    active_total_weight,
                                )
                            } else {
                                interpolated_biallelic_words_probability(
                                    &mosaic_allele_words,
                                    &mosaic_missing_words,
                                    state_probabilities,
                                    state_probabilities,
                                    0.0,
                                    1,
                                )
                            }
                        };
                    next_reference_words = following_reference_words;
                }
                internal_marker = marker_end;
            }

            if let Some(right_cluster) = right_hmm_marker {
                let right_union_marker = cluster_plan.member_union_indices[right_cluster][0];
                let left_end_cm = cluster_plan.cluster_end_cm_positions[cluster];
                let span_cm = union_cm_positions[right_union_marker] - left_end_cm;
                let mut next_union_marker = cluster_end;
                while next_union_marker < right_union_marker {
                    let started = profile_projection.then(Instant::now);
                    let segment_end = mosaic.active_donor_run_at(
                        next_union_marker,
                        if use_retained_projection {
                            &active_states
                        } else {
                            &all_states
                        },
                        &mut active_donors,
                    )?;
                    projection_donor_lookup_elapsed +=
                        started.map_or(Duration::ZERO, |s| s.elapsed());
                    let marker_end = segment_end.min(right_union_marker);
                    let donors = active_donors.as_slice();
                    let target_owned = target_row_indices[next_union_marker..marker_end]
                        .iter()
                        .any(Option::is_some);
                    if !target_owned
                        && use_retained_projection
                        && marker_end - next_union_marker >= 8
                    {
                        if sequence_index.is_some_and(|index| {
                            index.project_run(
                                next_union_marker,
                                marker_end,
                                &active_donors,
                                &active_weights,
                                Some(&right_active_weights),
                                union_cm_positions,
                                left_end_cm,
                                span_cm,
                                allele_probabilities,
                                &mut sequence_scratch,
                            )
                        }) {
                            next_union_marker = marker_end;
                            continue;
                        }
                    }
                    let mut next_reference_words =
                        packed_reference.complete_marker_words(next_union_marker);
                    for union_index in next_union_marker..marker_end {
                        let alpha = if span_cm > f64::EPSILON {
                            ((union_cm_positions[union_index] - left_end_cm) / span_cm)
                                .clamp(0.0, 1.0) as f32
                        } else {
                            0.0
                        };
                        let following_reference_words = (union_index + 1 < marker_end)
                            .then(|| packed_reference.complete_marker_words(union_index + 1))
                            .flatten();
                        if target_row_indices[union_index].is_some() {
                            skipped_target_markers += 1;
                            next_reference_words = following_reference_words;
                            continue;
                        }
                        allele_probabilities[union_index] =
                            if let Some(reference_words) = next_reference_words.take() {
                                if let Some(following_reference_words) = following_reference_words {
                                    prefetch_complete_marker_words(following_reference_words);
                                }
                                complete_gap_markers += 1;
                                if profile_projection {
                                    let started = Instant::now();
                                    let probability = if use_retained_projection {
                                        interpolated_sparse_complete_mosaic_probability(
                                            reference_words,
                                            &active_donors,
                                            &active_weights,
                                            &right_active_weights,
                                            alpha,
                                        )
                                    } else {
                                        interpolated_complete_mosaic_probability(
                                            reference_words,
                                            donors,
                                            state_probabilities,
                                            &right_probabilities,
                                            alpha,
                                            1,
                                        )
                                    };
                                    projection_fused_elapsed += started.elapsed();
                                    probability
                                } else if use_retained_projection {
                                    interpolated_sparse_complete_mosaic_probability(
                                        reference_words,
                                        &active_donors,
                                        &active_weights,
                                        &right_active_weights,
                                        alpha,
                                    )
                                } else {
                                    interpolated_complete_mosaic_probability(
                                        reference_words,
                                        donors,
                                        state_probabilities,
                                        &right_probabilities,
                                        alpha,
                                        1,
                                    )
                                }
                            } else {
                                incomplete_gap_markers += 1;
                                if profile_projection {
                                    let started = Instant::now();
                                    mosaic.pack_marker_words(
                                        packed_reference,
                                        union_index,
                                        &mut mosaic_allele_words,
                                        &mut mosaic_missing_words,
                                    )?;
                                    projection_pack_elapsed += started.elapsed();
                                    let started = Instant::now();
                                    let probability = if use_retained_projection {
                                        interpolated_sparse_biallelic_words_probability(
                                            &mosaic_allele_words,
                                            &mosaic_missing_words,
                                            &active_states,
                                            &active_weights,
                                            &right_active_weights,
                                            alpha,
                                        )
                                    } else {
                                        interpolated_biallelic_words_probability(
                                            &mosaic_allele_words,
                                            &mosaic_missing_words,
                                            state_probabilities,
                                            &right_probabilities,
                                            alpha,
                                            1,
                                        )
                                    };
                                    projection_probability_elapsed += started.elapsed();
                                    probability
                                } else {
                                    mosaic.pack_marker_words(
                                        packed_reference,
                                        union_index,
                                        &mut mosaic_allele_words,
                                        &mut mosaic_missing_words,
                                    )?;
                                    if use_retained_projection {
                                        interpolated_sparse_biallelic_words_probability(
                                            &mosaic_allele_words,
                                            &mosaic_missing_words,
                                            &active_states,
                                            &active_weights,
                                            &right_active_weights,
                                            alpha,
                                        )
                                    } else {
                                        interpolated_biallelic_words_probability(
                                            &mosaic_allele_words,
                                            &mosaic_missing_words,
                                            state_probabilities,
                                            &right_probabilities,
                                            alpha,
                                            1,
                                        )
                                    }
                                }
                            };
                        next_reference_words = following_reference_words;
                    }
                    next_union_marker = marker_end;
                }
            }
            if cluster == 0 {
                left_endpoint = Some((
                    state_probabilities.to_vec(),
                    active_states.clone(),
                    active_weights.clone(),
                    active_total_weight,
                ));
            }
            if right_hmm_marker.is_none() {
                rightmost_endpoint = Some((
                    state_probabilities.to_vec(),
                    active_states.clone(),
                    active_weights.clone(),
                    active_total_weight,
                ));
            }
            right_probabilities.copy_from_slice(state_probabilities);
            right_hmm_marker = Some(cluster);
            Ok(())
        },
    )?;

    if let Some(trace) = model_trace.as_mut() {
        trace.flush().map_err(|error| error.to_string())?;
    }

    if let Some((probabilities, states, weights, total_weight)) = left_endpoint {
        let first_union_marker = cluster_plan.member_union_indices[0][0];
        for union_index in 0..first_union_marker {
            if target_row_indices[union_index].is_some() {
                skipped_target_markers += 1;
                continue;
            }
            mosaic.pack_marker(
                packed_reference,
                union_index,
                &mut projection_panel.haplotypes,
            )?;
            allele_probabilities[union_index] = sparse_panel_allele_probability(
                &projection_panel,
                0,
                &probabilities,
                &states,
                &weights,
                total_weight,
                1,
            );
        }
    }
    if let Some((probabilities, states, weights, total_weight)) = rightmost_endpoint {
        let last_union_marker = *cluster_plan
            .member_union_indices
            .last()
            .and_then(|members| members.last())
            .ok_or_else(|| "target HMM cluster has no final marker".to_owned())?;
        for union_index in last_union_marker + 1..union_cm_positions.len() {
            if target_row_indices[union_index].is_some() {
                skipped_target_markers += 1;
                continue;
            }
            mosaic.pack_marker(
                packed_reference,
                union_index,
                &mut projection_panel.haplotypes,
            )?;
            allele_probabilities[union_index] = sparse_panel_allele_probability(
                &projection_panel,
                0,
                &probabilities,
                &states,
                &weights,
                total_weight,
                1,
            );
        }
    }
    if profile_projection {
        eprintln!(
            "JANUSX_IMP_PROJECTION_PROFILE complete_markers={complete_gap_markers} incomplete_markers={incomplete_gap_markers} skipped_target_markers={skipped_target_markers} clusters={} retained_projection_clusters={retained_projection_cluster_count} mean_active_states={:.3} observed_s={:.6} donor_lookup_s={:.6} fused_s={:.6} fallback_pack_s={:.6} fallback_probability_s={:.6}",
            observations.n_markers,
            active_state_sum as f64 / observations.n_markers as f64,
            projection_observed_elapsed.as_secs_f64(),
            projection_donor_lookup_elapsed.as_secs_f64(),
            projection_fused_elapsed.as_secs_f64(),
            projection_pack_elapsed.as_secs_f64(),
            projection_probability_elapsed.as_secs_f64(),
        );
    }
    Ok(())
}

fn panel_allele_probability(
    panel: &super::types::CompositePanel,
    marker: usize,
    state_probabilities: &[f32],
    allele: u16,
) -> f32 {
    if state_probabilities.len() == panel.haplotypes.n_haplotypes {
        if let Some((allele_words, missing_words)) =
            panel.haplotypes.biallelic_marker_planes(marker)
        {
            return project_biallelic_row(allele_words, missing_words, allele, |state| {
                state_probabilities[state]
            });
        }
    }
    let mut observed_alleles = 0usize;
    let mut observed_alt = 0usize;
    let mut observed_probability = 0.0f32;
    let mut missing_probability = 0.0f32;
    for (state, &probability) in state_probabilities.iter().enumerate() {
        match panel.haplotypes.allele(marker, state) {
            Some(state_allele) => {
                observed_alleles += 1;
                observed_alt += usize::from(state_allele == allele);
                if state_allele == allele {
                    observed_probability += probability;
                }
            }
            None => missing_probability += probability,
        }
    }
    let fallback_frequency = if observed_alleles == 0 {
        0.5
    } else {
        observed_alt as f32 / observed_alleles as f32
    };
    (observed_probability + missing_probability * fallback_frequency).clamp(0.0, 1.0)
}

/// Interpolate state probabilities and project them to an allele marginal in
/// one pass. The projection is linear, so materializing an intermediate
/// per-state vector is unnecessary.
#[cfg(test)]
fn interpolated_panel_allele_probability(
    panel: &super::types::CompositePanel,
    marker: usize,
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    alpha: f32,
    allele: u16,
) -> f32 {
    debug_assert_eq!(left_probabilities.len(), panel.haplotypes.n_haplotypes);
    debug_assert_eq!(right_probabilities.len(), panel.haplotypes.n_haplotypes);
    if let Some((allele_words, missing_words)) = panel.haplotypes.biallelic_marker_planes(marker) {
        return interpolated_biallelic_words_probability(
            allele_words,
            missing_words,
            left_probabilities,
            right_probabilities,
            alpha,
            allele,
        );
    }
    let left_weight = 1.0 - alpha;
    let mut observed_alleles = 0usize;
    let mut observed_alt = 0usize;
    let mut left_observed_probability = 0.0f32;
    let mut right_observed_probability = 0.0f32;
    let mut left_missing_probability = 0.0f32;
    let mut right_missing_probability = 0.0f32;

    for (state, (&left_probability, &right_probability)) in left_probabilities
        .iter()
        .zip(right_probabilities)
        .enumerate()
    {
        match panel.haplotypes.allele(marker, state) {
            Some(state_allele) => {
                observed_alleles += 1;
                let matches = state_allele == allele;
                observed_alt += usize::from(matches);
                if matches {
                    left_observed_probability += left_probability;
                    right_observed_probability += right_probability;
                }
            }
            None => {
                left_missing_probability += left_probability;
                right_missing_probability += right_probability;
            }
        }
    }
    let fallback_frequency = if observed_alleles == 0 {
        0.5
    } else {
        observed_alt as f32 / observed_alleles as f32
    };
    let left_probability =
        left_observed_probability + left_missing_probability * fallback_frequency;
    let right_probability =
        right_observed_probability + right_missing_probability * fallback_frequency;
    (left_probability * left_weight + right_probability * alpha).clamp(0.0, 1.0)
}

fn interpolated_biallelic_words_probability(
    allele_words: &[u64],
    missing_words: &[u64],
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    alpha: f32,
    allele: u16,
) -> f32 {
    debug_assert_eq!(allele_words.len(), missing_words.len());
    debug_assert_eq!(left_probabilities.len(), right_probabilities.len());
    if missing_words.iter().all(|&word| word == 0) {
        return interpolated_biallelic_complete_words_probability(
            allele_words,
            left_probabilities,
            right_probabilities,
            alpha,
            allele,
        );
    }

    let mut observed_alleles = 0usize;
    let mut observed_alt = 0usize;
    let mut left_observed_probability = 0.0f32;
    let mut right_observed_probability = 0.0f32;
    let mut left_missing_probability = 0.0f32;
    let mut right_missing_probability = 0.0f32;
    for (word_index, (&allele_word, &missing_word)) in
        allele_words.iter().zip(missing_words).enumerate()
    {
        let observed_word = !missing_word;
        let matching_word = match allele {
            0 => !allele_word & observed_word,
            1 => allele_word & observed_word,
            _ => 0,
        };
        observed_alleles += observed_word.count_ones() as usize;
        observed_alt += matching_word.count_ones() as usize;

        let mut matching_word = matching_word;
        while matching_word != 0 {
            let state = word_index * 64 + matching_word.trailing_zeros() as usize;
            left_observed_probability += left_probabilities[state];
            right_observed_probability += right_probabilities[state];
            matching_word &= matching_word - 1;
        }

        let mut missing_word = missing_word;
        while missing_word != 0 {
            let state = word_index * 64 + missing_word.trailing_zeros() as usize;
            left_missing_probability += left_probabilities[state];
            right_missing_probability += right_probabilities[state];
            missing_word &= missing_word - 1;
        }
    }
    let fallback_frequency = if observed_alleles == 0 {
        0.5
    } else {
        observed_alt as f32 / observed_alleles as f32
    };
    let left_probability =
        left_observed_probability + left_missing_probability * fallback_frequency;
    let right_probability =
        right_observed_probability + right_missing_probability * fallback_frequency;
    (left_probability * (1.0 - alpha) + right_probability * alpha).clamp(0.0, 1.0)
}

fn interpolated_biallelic_complete_words_probability(
    allele_words: &[u64],
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    alpha: f32,
    allele: u16,
) -> f32 {
    debug_assert_eq!(left_probabilities.len(), right_probabilities.len());
    debug_assert_eq!(allele_words.len(), left_probabilities.len().div_ceil(64));

    if left_probabilities.len() % 64 == 0 {
        #[cfg(target_arch = "aarch64")]
        {
            // Complete rows need no missingness plane; retain the same SIMD
            // summation order as the missing-aware path.
            return unsafe {
                interpolated_biallelic_projection_neon(
                    allele_words,
                    left_probabilities,
                    right_probabilities,
                    alpha,
                    allele,
                )
            };
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return unsafe {
                interpolated_biallelic_projection_avx2(
                    allele_words,
                    left_probabilities,
                    right_probabilities,
                    alpha,
                    allele,
                )
            };
        }
    }

    let state_count = left_probabilities.len();
    debug_assert_eq!(right_probabilities.len(), state_count);
    debug_assert_eq!(allele_words.len(), state_count.div_ceil(64));
    let mut left_probability = 0.0f32;
    let mut right_probability = 0.0f32;
    for (word_index, &allele_word) in allele_words.iter().enumerate() {
        let first_state = word_index * 64;
        let states_in_word = (state_count - first_state).min(64);
        let valid_mask = if states_in_word == 64 {
            u64::MAX
        } else {
            (1u64 << states_in_word) - 1
        };
        let mut matching_word = match allele {
            0 => !allele_word & valid_mask,
            1 => allele_word & valid_mask,
            _ => 0,
        };
        while matching_word != 0 {
            let state = first_state + matching_word.trailing_zeros() as usize;
            left_probability += left_probabilities[state];
            right_probability += right_probabilities[state];
            matching_word &= matching_word - 1;
        }
    }
    (left_probability * (1.0 - alpha) + right_probability * alpha).clamp(0.0, 1.0)
}

#[inline]
fn prefetch_complete_marker_words(words: &[u64]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

        let bytes = std::mem::size_of_val(words);
        let pointer = words.as_ptr().cast::<i8>();
        let mut offset = 0usize;
        while offset < bytes {
            _mm_prefetch(pointer.add(offset), _MM_HINT_T0);
            offset += 64;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = words;
}

fn interpolated_complete_mosaic_probability(
    reference_words: &[u64],
    donors: &[usize],
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    alpha: f32,
    allele: u16,
) -> f32 {
    debug_assert_eq!(donors.len(), left_probabilities.len());
    debug_assert_eq!(left_probabilities.len(), right_probabilities.len());
    debug_assert!(donors
        .iter()
        .all(|&donor| donor < reference_words.len() * 64));

    if donors.len() % 64 == 0 {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return unsafe {
                interpolated_complete_mosaic_probability_avx2(
                    reference_words,
                    donors,
                    left_probabilities,
                    right_probabilities,
                    alpha,
                    allele,
                )
            };
        }
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe {
                interpolated_complete_mosaic_probability_neon(
                    reference_words,
                    donors,
                    left_probabilities,
                    right_probabilities,
                    alpha,
                    allele,
                )
            };
        }
    }

    let mut left_probability = 0.0f32;
    let mut right_probability = 0.0f32;
    for (state, &donor) in donors.iter().enumerate() {
        let state_allele = (reference_words[donor >> 6] >> (donor & 63)) & 1;
        if state_allele == u64::from(allele) {
            left_probability += left_probabilities[state];
            right_probability += right_probabilities[state];
        }
    }
    (left_probability * (1.0 - alpha) + right_probability * alpha).clamp(0.0, 1.0)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn interpolated_complete_mosaic_probability_neon(
    reference_words: &[u64],
    donors: &[usize],
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    alpha: f32,
    allele: u16,
) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        let mut left_sum = vdupq_n_f32(0.0);
        let mut right_sum = vdupq_n_f32(0.0);
        let mut matching_lanes = [0u32; 4];
        for state in (0..donors.len()).step_by(4) {
            for lane in 0..4 {
                let donor = donors[state + lane];
                let state_allele = (reference_words[donor >> 6] >> (donor & 63)) & 1;
                matching_lanes[lane] = if state_allele == u64::from(allele) {
                    u32::MAX
                } else {
                    0
                };
            }
            let mask = vld1q_u32(matching_lanes.as_ptr());
            let left = vld1q_f32(left_probabilities.as_ptr().add(state));
            let right = vld1q_f32(right_probabilities.as_ptr().add(state));
            left_sum = vaddq_f32(
                left_sum,
                vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(left), mask)),
            );
            right_sum = vaddq_f32(
                right_sum,
                vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(right), mask)),
            );
        }
        let left_probability = vaddvq_f32(left_sum);
        let right_probability = vaddvq_f32(right_sum);
        (left_probability * (1.0 - alpha) + right_probability * alpha).clamp(0.0, 1.0)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interpolated_complete_mosaic_probability_avx2(
    reference_words: &[u64],
    donors: &[usize],
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    alpha: f32,
    allele: u16,
) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let shift_mask = _mm256_set1_epi64x(63);
        let one = _mm256_set1_epi64x(1);
        let target_allele = _mm256_set1_epi64x(i64::from(allele));
        let mut left_sum = _mm256_setzero_ps();
        let mut right_sum = _mm256_setzero_ps();
        for state in (0..donors.len()).step_by(8) {
            let first_donors = _mm256_loadu_si256(donors.as_ptr().add(state).cast());
            let first_word_indices = _mm256_srli_epi64(first_donors, 6);
            let first_bit_offsets = _mm256_and_si256(first_donors, shift_mask);
            let first_reference_words = _mm256_i64gather_epi64::<8>(
                reference_words.as_ptr().cast::<i64>(),
                first_word_indices,
            );
            let first_alleles = _mm256_and_si256(
                _mm256_srlv_epi64(first_reference_words, first_bit_offsets),
                one,
            );
            let first_matches = _mm256_cmpeq_epi64(first_alleles, target_allele);
            let first_pattern = _mm256_movemask_pd(_mm256_castsi256_pd(first_matches)) as u32;

            let second_donors = _mm256_loadu_si256(donors.as_ptr().add(state + 4).cast());
            let second_word_indices = _mm256_srli_epi64(second_donors, 6);
            let second_bit_offsets = _mm256_and_si256(second_donors, shift_mask);
            let second_reference_words = _mm256_i64gather_epi64::<8>(
                reference_words.as_ptr().cast::<i64>(),
                second_word_indices,
            );
            let second_alleles = _mm256_and_si256(
                _mm256_srlv_epi64(second_reference_words, second_bit_offsets),
                one,
            );
            let second_matches = _mm256_cmpeq_epi64(second_alleles, target_allele);
            let second_pattern = _mm256_movemask_pd(_mm256_castsi256_pd(second_matches)) as u32;
            let pattern = (first_pattern | (second_pattern << 4)) as usize;

            let mask = _mm256_loadu_si256(PROJECTION_MASKS8[pattern].as_ptr().cast());
            let left = _mm256_loadu_ps(left_probabilities.as_ptr().add(state));
            let right = _mm256_loadu_ps(right_probabilities.as_ptr().add(state));
            left_sum = _mm256_add_ps(left_sum, _mm256_and_ps(left, _mm256_castsi256_ps(mask)));
            right_sum = _mm256_add_ps(right_sum, _mm256_and_ps(right, _mm256_castsi256_ps(mask)));
        }

        let mut left_lanes = [0.0f32; 8];
        let mut right_lanes = [0.0f32; 8];
        _mm256_storeu_ps(left_lanes.as_mut_ptr(), left_sum);
        _mm256_storeu_ps(right_lanes.as_mut_ptr(), right_sum);
        let left_probability = left_lanes.iter().sum::<f32>();
        let right_probability = right_lanes.iter().sum::<f32>();
        (left_probability * (1.0 - alpha) + right_probability * alpha).clamp(0.0, 1.0)
    }
}

#[cfg(target_arch = "aarch64")]
const fn make_projection_masks4() -> [[u32; 4]; 16] {
    let mut masks = [[0u32; 4]; 16];
    let mut pattern = 0usize;
    while pattern < 16 {
        let mut lane = 0usize;
        while lane < 4 {
            if pattern & (1 << lane) != 0 {
                masks[pattern][lane] = u32::MAX;
            }
            lane += 1;
        }
        pattern += 1;
    }
    masks
}

#[cfg(target_arch = "aarch64")]
const PROJECTION_MASKS4: [[u32; 4]; 16] = make_projection_masks4();

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn interpolated_biallelic_projection_neon(
    allele_words: &[u64],
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    alpha: f32,
    allele: u16,
) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        let mut left_sum = vdupq_n_f32(0.0);
        let mut right_sum = vdupq_n_f32(0.0);
        for (word_index, &allele_word) in allele_words.iter().enumerate() {
            let matching_word = match allele {
                0 => !allele_word,
                1 => allele_word,
                _ => 0,
            };
            for group in 0..16 {
                let pattern = ((matching_word >> (group * 4)) & 0x0f) as usize;
                let mask = vld1q_u32(PROJECTION_MASKS4[pattern].as_ptr());
                let state = word_index * 64 + group * 4;
                let left = vld1q_f32(left_probabilities.as_ptr().add(state));
                let right = vld1q_f32(right_probabilities.as_ptr().add(state));
                left_sum = vaddq_f32(
                    left_sum,
                    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(left), mask)),
                );
                right_sum = vaddq_f32(
                    right_sum,
                    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(right), mask)),
                );
            }
        }
        let left_probability = vaddvq_f32(left_sum);
        let right_probability = vaddvq_f32(right_sum);
        (left_probability * (1.0 - alpha) + right_probability * alpha).clamp(0.0, 1.0)
    }
}

#[cfg(target_arch = "x86_64")]
const fn make_projection_masks8() -> [[u32; 8]; 256] {
    let mut masks = [[0u32; 8]; 256];
    let mut pattern = 0usize;
    while pattern < 256 {
        let mut lane = 0usize;
        while lane < 8 {
            if pattern & (1 << lane) != 0 {
                masks[pattern][lane] = u32::MAX;
            }
            lane += 1;
        }
        pattern += 1;
    }
    masks
}

#[cfg(target_arch = "x86_64")]
const PROJECTION_MASKS8: [[u32; 8]; 256] = make_projection_masks8();

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interpolated_biallelic_projection_avx2(
    allele_words: &[u64],
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    alpha: f32,
    allele: u16,
) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut left_sum = _mm256_setzero_ps();
        let mut right_sum = _mm256_setzero_ps();
        for (word_index, &allele_word) in allele_words.iter().enumerate() {
            let matching_word = match allele {
                0 => !allele_word,
                1 => allele_word,
                _ => 0,
            };
            for group in 0..8 {
                let pattern = ((matching_word >> (group * 8)) & 0xff) as usize;
                let mask = _mm256_loadu_si256(PROJECTION_MASKS8[pattern].as_ptr().cast());
                let state = word_index * 64 + group * 8;
                let left = _mm256_loadu_ps(left_probabilities.as_ptr().add(state));
                let right = _mm256_loadu_ps(right_probabilities.as_ptr().add(state));
                left_sum = _mm256_add_ps(left_sum, _mm256_and_ps(left, _mm256_castsi256_ps(mask)));
                right_sum =
                    _mm256_add_ps(right_sum, _mm256_and_ps(right, _mm256_castsi256_ps(mask)));
            }
        }
        let mut left_lanes = [0.0f32; 8];
        let mut right_lanes = [0.0f32; 8];
        _mm256_storeu_ps(left_lanes.as_mut_ptr(), left_sum);
        _mm256_storeu_ps(right_lanes.as_mut_ptr(), right_sum);
        let left_probability = left_lanes.iter().sum::<f32>();
        let right_probability = right_lanes.iter().sum::<f32>();
        (left_probability * (1.0 - alpha) + right_probability * alpha).clamp(0.0, 1.0)
    }
}

#[inline]
fn project_biallelic_row(
    allele_words: &[u64],
    missing_words: &[u64],
    allele: u16,
    mut probability_at: impl FnMut(usize) -> f32,
) -> f32 {
    debug_assert_eq!(allele_words.len(), missing_words.len());
    let mut observed_alleles = 0usize;
    let mut observed_matching_alleles = 0usize;
    let mut observed_probability = 0.0f32;
    let mut missing_probability = 0.0f32;
    for (word_index, (&allele_word, &missing_word)) in
        allele_words.iter().zip(missing_words).enumerate()
    {
        let observed_word = !missing_word;
        let alternate_word = allele_word & observed_word;
        observed_alleles += observed_word.count_ones() as usize;
        let mut matching_word = match allele {
            0 => !allele_word & observed_word,
            1 => alternate_word,
            _ => 0,
        };
        observed_matching_alleles += matching_word.count_ones() as usize;
        while matching_word != 0 {
            let bit = matching_word.trailing_zeros() as usize;
            observed_probability += probability_at(word_index * 64 + bit);
            matching_word &= matching_word - 1;
        }
        let mut missing_word = missing_word;
        while missing_word != 0 {
            let bit = missing_word.trailing_zeros() as usize;
            missing_probability += probability_at(word_index * 64 + bit);
            missing_word &= missing_word - 1;
        }
    }
    let fallback_frequency = if observed_alleles == 0 {
        0.5
    } else {
        observed_matching_alleles as f32 / observed_alleles as f32
    };
    (observed_probability + missing_probability * fallback_frequency).clamp(0.0, 1.0)
}

fn target_alt_frequency(row: &super::types::VariantRow) -> Result<f32, String> {
    let mut count = 0usize;
    let mut observed = 0usize;
    for call in row.calls.iter() {
        for allele in call.alleles.into_iter().flatten() {
            count += usize::from(allele == 1);
            observed += 1;
        }
    }
    if observed == 0 {
        return Ok(0.5);
    }
    Ok(count as f32 / observed as f32)
}

fn swap_biallelic(allele: u16) -> u16 {
    if allele == 0 {
        1
    } else {
        0
    }
}

fn merge_sorted_indices(
    target: &[super::types::VariantRow],
    reference: &[PhasedRow],
    max_rows: Option<usize>,
) -> Result<Vec<(Option<usize>, Option<usize>)>, String> {
    validate_sorted_unique(target.iter().map(|row| &row.meta), "target")?;
    validate_sorted_unique(reference.iter().map(|row| &row.meta), "reference")?;
    merge_sorted_indices_impl(
        target,
        reference.len(),
        |index| &reference[index].meta,
        max_rows,
    )
}

fn merge_sorted_indices_from_metas(
    target: &[super::types::VariantRow],
    reference: &[VariantMeta],
    max_rows: Option<usize>,
) -> Result<Vec<(Option<usize>, Option<usize>)>, String> {
    validate_sorted_unique(target.iter().map(|row| &row.meta), "target")?;
    validate_sorted_unique(reference.iter(), "reference")?;
    merge_sorted_indices_impl(target, reference.len(), |index| &reference[index], max_rows)
}

fn merge_sorted_indices_impl<'a>(
    target: &[super::types::VariantRow],
    reference_len: usize,
    reference_meta: impl Fn(usize) -> &'a VariantMeta,
    max_rows: Option<usize>,
) -> Result<Vec<(Option<usize>, Option<usize>)>, String> {
    let total = target.len().saturating_add(reference_len);
    let capacity = max_rows
        .map(|limit| limit.saturating_add(1).min(total))
        .unwrap_or(total);
    let mut merged = Vec::with_capacity(capacity);
    let (mut target_index, mut reference_index) = (0, 0);
    while target_index < target.len() || reference_index < reference_len {
        let entry = match (
            target.get(target_index),
            (reference_index < reference_len).then(|| reference_meta(reference_index)),
        ) {
            (Some(target_row), Some(ref_meta)) => {
                match compare_position(&target_row.meta, ref_meta) {
                    Ordering::Less => {
                        target_index += 1;
                        (Some(target_index - 1), None)
                    }
                    Ordering::Greater => {
                        reference_index += 1;
                        (None, Some(reference_index - 1))
                    }
                    Ordering::Equal => {
                        target_index += 1;
                        reference_index += 1;
                        (Some(target_index - 1), Some(reference_index - 1))
                    }
                }
            }
            (Some(_), None) => {
                target_index += 1;
                (Some(target_index - 1), None)
            }
            (None, Some(_)) => {
                reference_index += 1;
                (None, Some(reference_index - 1))
            }
            (None, None) => break,
        };
        if max_rows.is_some_and(|limit| merged.len() >= limit) {
            let limit = max_rows.unwrap();
            return Err(format!(
                "merged target/reference window exceeds --window-markers={limit}; reduce window size or split the reference input"
            ));
        }
        merged.push(entry);
    }
    Ok(merged)
}

fn validate_sorted_unique<'a>(
    metadata: impl Iterator<Item = &'a VariantMeta>,
    label: &str,
) -> Result<(), String> {
    let mut previous: Option<&VariantMeta> = None;
    for meta in metadata {
        if let Some(previous) = previous {
            match compare_position(previous, meta) {
                Ordering::Equal => {
                    return Err(format!(
                        "{label} contains duplicate variant positions at {}:{}; split or remove duplicate records before imputation",
                        meta.chrom, meta.pos
                    ));
                }
                Ordering::Greater => {
                    return Err(format!(
                        "{label} variants are not sorted by chromosome and position at {}:{}",
                        meta.chrom, meta.pos
                    ));
                }
                Ordering::Less => {}
            }
        }
        previous = Some(meta);
    }
    Ok(())
}

fn compare_position(left: &VariantMeta, right: &VariantMeta) -> Ordering {
    left.chrom
        .cmp(&right.chrom)
        .then_with(|| left.pos.cmp(&right.pos))
}

fn harmonization(target: &VariantMeta, reference: &VariantMeta) -> Result<bool, String> {
    if target.alt_alleles.len() != 1 || reference.alt_alleles.len() != 1 {
        if target.ref_allele == reference.ref_allele && target.alt_alleles == reference.alt_alleles
        {
            return Ok(false);
        }
        return Err(format!(
            "cannot harmonize multiallelic variants at {}:{}",
            target.chrom, target.pos
        ));
    }
    if target.ref_allele == reference.ref_allele
        && target.alt_alleles[0] == reference.alt_alleles[0]
    {
        Ok(false)
    } else if target.ref_allele == reference.alt_alleles[0]
        && target.alt_alleles[0] == reference.ref_allele
    {
        Ok(true)
    } else {
        Err(format!(
            "target/reference alleles do not match at {}:{} ({}>{} versus {}>{})",
            target.chrom,
            target.pos,
            target.ref_allele,
            target.alt_alleles.join(","),
            reference.ref_allele,
            reference.alt_alleles.join(",")
        ))
    }
}

#[cfg(test)]
mod low_frequency_tests {
    use super::{
        beagle_rare_carrier_limit, low_frequency_markers, low_frequency_markers_with_reference,
        use_two_stage_phase,
    };
    use crate::imp::types::{
        DiploidCall, DiploidCalls, PhaseWindow, PhasedRow, PhasedWindow, VariantMeta, VariantRow,
    };

    fn window_with_alt_carrier_counts(counts: &[usize], n_samples: usize) -> PhaseWindow {
        let rows = counts
            .iter()
            .enumerate()
            .map(|(marker, &alt_carriers)| {
                let calls = (0..n_samples)
                    .map(|sample| {
                        let allele = u16::from(sample < alt_carriers);
                        DiploidCall {
                            alleles: [Some(allele), Some(allele)],
                            phased: true,
                        }
                    })
                    .collect::<Vec<_>>();
                VariantRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(calls),
                }
            })
            .collect::<Vec<_>>();
        PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            cm_positions: (0..rows.len()).map(|marker| marker as f64 * 0.01).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: rows.len(),
            rows,
        }
    }

    #[test]
    fn low_frequency_boundary_uses_unique_sample_carriers_and_strict_greater_than() {
        // With 2,000 samples, floor(n*0.002)=4 and maxCarriers=4.
        // Four carriers are low frequency; five make both alleles high frequency.
        let window = window_with_alt_carrier_counts(&[4, 5], 2_000);

        assert_eq!(low_frequency_markers(&window).unwrap(), vec![true, false]);
    }

    #[test]
    fn rare_carrier_limit_uses_beagle_float_frequency() {
        assert_eq!(beagle_rare_carrier_limit(0), 3);
        assert_eq!(beagle_rare_carrier_limit(1_000), 3);
        assert_eq!(beagle_rare_carrier_limit(2_499), 4);
        assert_eq!(beagle_rare_carrier_limit(2_501), 5);
    }

    #[test]
    fn two_stage_phase_keeps_exactly_seventy_five_percent_high_frequency_markers() {
        // At n=1,000, maxCarriers=3. Four carriers make a marker eligible;
        // three do not. Exactly three of four eligible markers is allowed.
        let exactly_seventy_five = window_with_alt_carrier_counts(&[4, 4, 4, 3], 1_000);
        let above_seventy_five = window_with_alt_carrier_counts(&[4, 4, 4, 4], 1_000);

        assert_eq!(
            low_frequency_markers(&exactly_seventy_five).unwrap(),
            vec![false, false, false, true]
        );
        assert!(use_two_stage_phase(&exactly_seventy_five).unwrap());
        assert!(!use_two_stage_phase(&above_seventy_five).unwrap());
    }

    #[test]
    fn two_stage_phase_falls_back_when_fewer_than_two_markers_are_eligible() {
        let window = window_with_alt_carrier_counts(&[4, 3, 3, 3], 1_000);
        assert!(!use_two_stage_phase(&window).unwrap());
    }

    #[test]
    fn reference_carriers_are_counted_after_biallelic_allele_harmonization() {
        let target = window_with_alt_carrier_counts(&[3], 1_000);
        assert_eq!(low_frequency_markers(&target).unwrap(), vec![true]);
        let reference = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: 1,
                    id: "v1".to_owned(),
                    ref_allele: "G".to_owned(),
                    alt_alleles: vec!["A".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: true,
                }]),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            }],
            emit_start: 0,
            emit_end: 1,
        };

        assert_eq!(
            low_frequency_markers_with_reference(&target, Some(&reference)).unwrap(),
            vec![false]
        );
    }

    #[test]
    fn stage2_allele_flags_use_original_target_and_reference_carriers() {
        let target = window_with_alt_carrier_counts(&[3], 10);
        let reference = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta: target.rows[0].meta.clone(),
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                ]),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            }],
            emit_start: 0,
            emit_end: 1,
        };
        let alignment =
            super::ReferenceMarkerAlignment::new(&target.rows, &reference.rows).unwrap();

        let flags = super::low_frequency_allele_flags_with_alignment(
            &target,
            Some(&reference),
            Some(&alignment),
        )
        .unwrap();

        assert_eq!(flags, vec![vec![false, true]]);
    }
}

#[cfg(test)]
mod imputation_alignment_tests {
    use super::{
        adaptive_imp_states, assemble_sample_batch, beagle_cumulative_positions,
        build_cluster_hmm_panels, build_hmm_cluster_plan, build_imputation_hmm_inputs,
        hard_call_alt, imp_step_ranges, imputation_hmm_config, impute_window_prevalidated,
        phase_estimate_cluster_errors, project_genetic_position,
        retained_state_projection_required, select_imp_ibs_candidates, sparse_active_states,
        target_known_haplotype_alleles_into, target_query_alleles_from_phased_target,
        ImpIbsReferenceIndex,
    };
    use crate::imp::types::{
        CompositePanel, DiploidCall, DiploidCalls, HMMConfig, ImputeConfig, PackedHaplotypes,
        PhaseWindow, PhasedRow, PhasedWindow, VariantMeta, VariantRow, VcfOutputFields,
    };

    #[test]
    fn retained_states_are_used_even_when_projection_is_not_sparse() {
        assert!(retained_state_projection_required(300, 822));
    }

    #[test]
    fn hard_call_uses_beagle_argmax_tie_rule() {
        assert_eq!(hard_call_alt(0.49), 0);
        assert_eq!(hard_call_alt(0.5), 0);
        assert_eq!(hard_call_alt(0.500_1), 1);
    }

    fn phased_window() -> PhasedWindow {
        let rows = (0..3)
            .map(|marker| PhasedRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: if marker == 1 {
                        [Some(1), Some(0)]
                    } else {
                        [Some(0), Some(0)]
                    },
                    phased: true,
                }]),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
            .collect();
        PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            emit_start: 0,
            emit_end: 3,
        }
    }

    #[test]
    fn phased_target_query_uses_phase_estimate_at_missing_genotype() {
        let phased = phased_window();
        let target_row_indices = [Some(0), Some(1), Some(2)];
        let target_marker_indices = [0, 1, 2];
        let reference_major_alleles = [Some(0), Some(0), Some(0)];

        let query = target_query_alleles_from_phased_target(
            &phased,
            &target_row_indices,
            &target_marker_indices,
            &reference_major_alleles,
            0,
            0,
        )
        .unwrap();

        assert_eq!(query, vec![Some(0), Some(1), Some(0)]);
    }

    #[test]
    fn observed_haplotype_alleles_exclude_phased_missing_calls() {
        let meta = |marker: u64| VariantMeta {
            chrom: "1".to_owned(),
            pos: marker,
            id: format!("v{marker}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let target = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![
                VariantRow {
                    meta: meta(1),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: false,
                    }]),
                },
                VariantRow {
                    meta: meta(2),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [None, None],
                        phased: false,
                    }]),
                },
                VariantRow {
                    meta: meta(3),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(1), None],
                        phased: false,
                    }]),
                },
            ],
            cm_positions: vec![0.0, 0.001, 0.002],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 3,
        };
        let phased_target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![
                PhasedRow {
                    meta: meta(1),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(1), Some(0)],
                        phased: true,
                    }]),
                    dosage: None,
                    ap1: None,
                    ap2: None,
                    gp: None,
                    info_af: None,
                    info_dr2: None,
                    imputed: false,
                },
                PhasedRow {
                    meta: meta(2),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    }]),
                    dosage: None,
                    ap1: None,
                    ap2: None,
                    gp: None,
                    info_af: None,
                    info_dr2: None,
                    imputed: false,
                },
                PhasedRow {
                    meta: meta(3),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(1), Some(0)],
                        phased: true,
                    }]),
                    dosage: None,
                    ap1: None,
                    ap2: None,
                    gp: None,
                    info_af: None,
                    info_dr2: None,
                    imputed: false,
                },
            ],
            emit_start: 0,
            emit_end: 3,
        };
        let target_rows = [Some(0), Some(1), Some(2)];
        let markers = [0, 1, 2];
        let mut observed = Vec::new();

        target_known_haplotype_alleles_into(
            &target,
            &phased_target,
            &target_rows,
            &markers,
            0,
            0,
            false,
            &mut observed,
        )
        .unwrap();
        assert_eq!(observed, vec![Some(1), None, Some(1)]);

        target_known_haplotype_alleles_into(
            &target,
            &phased_target,
            &target_rows,
            &markers,
            0,
            1,
            true,
            &mut observed,
        )
        .unwrap();
        assert_eq!(observed, vec![Some(0), None, Some(1)]);
    }

    #[test]
    fn sample_assembly_restores_phased_target_markers_and_projects_reference_only_markers() {
        let meta = |marker: u64| VariantMeta {
            chrom: "1".to_owned(),
            pos: marker,
            id: format!("v{marker}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let target = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: meta(10),
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some(0), None],
                        phased: false,
                    },
                    DiploidCall {
                        alleles: [None, None],
                        phased: false,
                    },
                ]),
            }],
            cm_positions: vec![0.1],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let phased_target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta: meta(10),
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some(1), Some(0)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(1), Some(1)],
                        phased: true,
                    },
                ]),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            }],
            emit_start: 0,
            emit_end: 1,
        };
        let mut output_rows = vec![
            PhasedRow {
                meta: meta(10),
                calls: DiploidCalls::empty(2),
                dosage: Some(Vec::with_capacity(2)),
                ap1: Some(Vec::with_capacity(2)),
                ap2: Some(Vec::with_capacity(2)),
                gp: Some(Vec::with_capacity(2)),
                info_af: None,
                info_dr2: None,
                imputed: false,
            },
            PhasedRow {
                meta: meta(20),
                calls: DiploidCalls::empty(2),
                dosage: Some(Vec::with_capacity(2)),
                ap1: Some(Vec::with_capacity(2)),
                ap2: Some(Vec::with_capacity(2)),
                gp: Some(Vec::with_capacity(2)),
                info_af: None,
                info_dr2: None,
                imputed: true,
            },
        ];
        let posteriors = vec![0.2, 0.8, 0.4, 0.1, 0.1, 0.3, 0.2, 0.9];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let statistics = assemble_sample_batch(
            &mut output_rows,
            &posteriors,
            2,
            0,
            &target,
            &phased_target,
            &[Some(0), None],
            &[Some(0), Some(1)],
            &[true, true],
            &[Some(0.3), Some(0.7)],
            &[Some(0.4), None],
            &[false, true],
            true,
            &pool,
        );

        assert_eq!(
            output_rows[0].calls.get(0).unwrap().alleles,
            [Some(1), Some(0)]
        );
        assert_eq!(
            output_rows[0].calls.get(1).unwrap().alleles,
            [Some(1), Some(1)]
        );
        assert_eq!(
            output_rows[1].calls.get(0).unwrap().alleles,
            [Some(1), Some(0)]
        );
        assert_eq!(
            output_rows[1].calls.get(1).unwrap().alleles,
            [Some(0), Some(1)]
        );
        let row0_dosage = output_rows[0].dosage.as_deref().unwrap();
        let row1_dosage = output_rows[1].dosage.as_deref().unwrap();
        assert!((row0_dosage[0] - 1.0).abs() < 1.0e-6);
        assert!((row0_dosage[1] - 2.0).abs() < 1.0e-6);
        assert!((row1_dosage[0] - 0.9).abs() < 1.0e-6);
        assert!((row1_dosage[1] - 1.2).abs() < 1.0e-6);
        assert!((statistics[0].0 - 3.0).abs() < 1.0e-6);
        assert!((statistics[1].0 - 2.1).abs() < 1.0e-6);
    }

    #[test]
    fn imputation_hmm_config_preserves_fitted_phase_parameters() {
        let phase_config = HMMConfig {
            ne: 1_210,
            recombination_intensity: 0.000_9,
            err: 0.000_59,
            em: true,
            burnin: 3,
            iterations: 12,
            initial_lr: 100_000.0,
            phase_states: 280,
            haplotype_count: 764,
        };

        let observed = imputation_hmm_config(&phase_config, 3_116).unwrap();

        assert_eq!(observed.ne, 1_210);
        assert_eq!(observed.haplotype_count, 3_116);
        assert!(
            (f64::from(observed.recombination_intensity) - 0.000_9 * 764.0 / 3_116.0).abs()
                < 1.0e-7
        );
        assert_eq!(observed.err, 0.000_59);
    }

    #[test]
    fn adaptive_imp_states_scales_with_reference_panel() {
        assert_eq!(adaptive_imp_states(1_600, 3_116, 202, 202), 400);
        assert_eq!(adaptive_imp_states(1_600, 800, 800, 800), 400);
        assert_eq!(adaptive_imp_states(1_600, 20_000, 200, 200), 1_600);
        assert_eq!(adaptive_imp_states(320, 20_000, 200, 200), 320);
        assert_eq!(adaptive_imp_states(1_600, 64, 64, 64), 64);
        assert_eq!(adaptive_imp_states(1_600, 2_388, 51_059, 2_553), 1_600);
        assert_eq!(adaptive_imp_states(1_600, 2_000, 4_000, 1_000), 800);
    }

    #[test]
    fn imputation_clusters_use_a_fixed_start_and_midpoint_map() {
        let target_rows = [Some(0), Some(1), Some(2), None, Some(3)];
        let cluster_positions = [0.0, 0.004, 0.009, 0.013];

        let plan =
            build_hmm_cluster_plan(&target_rows, &cluster_positions, &cluster_positions, 0.005);

        assert_eq!(plan.member_union_indices, vec![vec![0, 1], vec![2, 4]]);
        assert_eq!(plan.member_target_indices, vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(plan.cm_positions, vec![0.002, 0.011]);
        assert_eq!(plan.cluster_end_cm_positions, vec![0.004, 0.013]);
    }

    #[test]
    fn imputation_map_uses_beagle_minimum_adjacent_marker_distance() {
        let positions = beagle_cumulative_positions(&[0.0, 0.0, 0.000_000_05, 0.25]).unwrap();

        assert_eq!(positions[0], 0.0);
        assert!((positions[1] - 0.000_000_1).abs() < 1e-15);
        assert!((positions[2] - 0.000_000_2).abs() < 1e-15);
        assert!((positions[3] - 0.250_000_15).abs() < 1e-12);
    }

    #[test]
    fn imputation_hmm_and_projection_use_their_own_beagle_map_coordinates() {
        let target_rows = [Some(0), Some(1), Some(2)];
        let target_hmm_positions = [0.0, 0.000_000_1, 0.000_000_2];
        let target_projection_positions = [10.0, 10.25, 10.5];
        let plan = build_hmm_cluster_plan(
            &target_rows,
            &target_hmm_positions,
            &target_projection_positions,
            0.000_000_15,
        );

        assert_eq!(plan.member_union_indices, vec![vec![0, 1], vec![2]]);
        assert!((plan.cm_positions[0] - 0.000_000_05).abs() < 1e-15);
        assert_eq!(plan.cluster_end_cm_positions, vec![10.25, 10.5]);
        assert_eq!(
            project_genetic_position(0.125, &[0.0, 0.25], &[0.0, 0.25]).unwrap(),
            0.125
        );
    }

    #[test]
    fn imp_step_ranges_start_with_half_step_and_use_lower_bound() {
        let ranges = imp_step_ranges(&[0.00, 0.03, 0.06, 0.10, 0.14, 0.21], 0.1).unwrap();

        assert_eq!(ranges, vec![(0, 2), (2, 5), (5, 6)]);
    }

    #[test]
    fn imp_ibs_candidates_do_not_fill_when_first_match_set_fits_cap() {
        let exact_matches = vec![vec![1, 4, 7], vec![1, 4], vec![1]];

        assert_eq!(
            select_imp_ibs_candidates(&exact_matches, 0, 3, 3, 42),
            vec![1, 4, 7]
        );
        assert_eq!(
            select_imp_ibs_candidates(&exact_matches, 0, 3, 2, 42),
            vec![1, 4]
        );
    }

    #[test]
    fn imp_ibs_candidates_refine_large_sets_and_sample_parent_donors_like_beagle() {
        let exact_matches = vec![vec![0, 1, 2, 3, 4, 5, 6, 7], vec![1], vec![1]];

        assert_eq!(
            select_imp_ibs_candidates(&exact_matches, 0, 3, 4, 42),
            vec![1, 2, 5, 6]
        );
    }

    #[test]
    fn imp_ibs_candidates_random_subset_is_stable_for_same_seed() {
        let exact_matches = vec![vec![0, 1, 2, 3, 4, 5, 6, 7], vec![1], vec![1]];

        let first = select_imp_ibs_candidates(&exact_matches, 0, 3, 4, 42);
        let second = select_imp_ibs_candidates(&exact_matches, 0, 3, 4, 42);

        assert_eq!(first, second);
    }

    #[test]
    fn imp_ibs_reference_index_returns_only_exact_step_sequence_matches() {
        let reference = phased_window();
        let packed = crate::imp::composite::PackedBiallelicReference::from_reference_rows(
            &reference.rows,
            &[Some(0), Some(1), Some(2)],
            &[false; 3],
            1,
        )
        .unwrap();
        let positions = [0.0, 0.02, 0.04];
        let plan =
            build_hmm_cluster_plan(&[Some(0), Some(1), Some(2)], &positions, &positions, 0.05);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let index = ImpIbsReferenceIndex::new(&packed, &plan, 0.1, 7, 0.2, 2, &pool).unwrap();

        assert_eq!(index.max_haps_per_step, 1);
        let small_budget =
            ImpIbsReferenceIndex::new(&packed, &plan, 0.1, 7, 6.0, 2, &pool).unwrap();
        assert_eq!(small_budget.max_haps_per_step, 1);
        assert_eq!(
            small_budget.candidate_sets(&packed, &[Some(0), Some(0), Some(0)], 42),
            vec![vec![1]]
        );
        assert_eq!(index.step_ranges, vec![(0, 1)]);
        assert_eq!(
            index.candidate_sets(&packed, &[Some(0), Some(0), Some(0)], 42),
            vec![vec![1]]
        );
    }

    #[test]
    fn imputation_cluster_errors_count_available_phased_alleles() {
        let target_rows = [Some(0), Some(1), None, Some(2)];
        let positions = [0.0, 0.001, 0.010, 0.011];
        let plan = build_hmm_cluster_plan(&target_rows, &positions, &positions, 0.005);
        let phased_alleles = [Some(0), Some(1), None, Some(1)];

        assert_eq!(
            phase_estimate_cluster_errors(&phased_alleles, &plan, 0.01),
            vec![0.02, 0.01]
        );
    }

    #[test]
    fn imputation_cluster_state_matches_only_the_full_allele_sequence() {
        let target_rows = [Some(0), Some(1)];
        let positions = [0.0, 0.001];
        let plan = build_hmm_cluster_plan(&target_rows, &positions, &positions, 0.005);
        let mut haplotypes = PackedHaplotypes::empty(2, 2, 1).unwrap();
        haplotypes.set_allele(0, 0, Some(0)).unwrap();
        haplotypes.set_allele(1, 0, Some(0)).unwrap();
        haplotypes.set_allele(0, 1, Some(0)).unwrap();
        haplotypes.set_allele(1, 1, Some(1)).unwrap();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0, 1],
            marker_indices: vec![0, 1],
            allele_counts: vec![2, 2],
        };
        let mut query = vec![Some(0); 2];
        query[1] = Some(1);

        let (observations, states) = build_cluster_hmm_panels(&panel, &query, &plan).unwrap();

        assert_eq!(observations.allele(0, 0), Some(0));
        assert_eq!(states.haplotypes.allele(0, 0), Some(1));
        assert_eq!(states.haplotypes.allele(0, 1), Some(0));
    }

    #[test]
    fn imputation_cluster_packed_missing_is_neutral_like_scalar() {
        let plan = build_hmm_cluster_plan(&[Some(0)], &[0.0], &[0.0], 0.005);
        let panel = CompositePanel {
            haplotypes: PackedHaplotypes::empty(2, 1, 1).unwrap(),
            source_samples: vec![0, 0],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let (observations, states) = build_cluster_hmm_panels(&panel, &[Some(1)], &plan).unwrap();
        assert_eq!(observations.allele(0, 0), Some(0));
        assert_eq!(states.haplotypes.allele(0, 0), Some(0));
        assert_eq!(states.haplotypes.allele(0, 1), Some(0));
    }

    #[test]
    fn imputation_hmm_uses_phase_estimates_at_missing_target_markers() {
        let target_rows = [Some(0), Some(1)];
        let positions = [0.0, 0.001];
        let plan = build_hmm_cluster_plan(&target_rows, &positions, &positions, 0.005);
        let mut haplotypes = PackedHaplotypes::empty(2, 2, 1).unwrap();
        haplotypes.set_allele(0, 0, Some(0)).unwrap();
        haplotypes.set_allele(1, 0, Some(0)).unwrap();
        haplotypes.set_allele(0, 1, Some(1)).unwrap();
        haplotypes.set_allele(1, 1, Some(1)).unwrap();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0, 1],
            marker_indices: vec![0, 1],
            allele_counts: vec![2, 2],
        };
        let phased_query = [Some(1), Some(1)];

        let (observations, states, errors) =
            build_imputation_hmm_inputs(&panel, &phased_query, &plan, 0.01).unwrap();

        assert_eq!(errors, vec![0.02]);
        assert_eq!(observations.allele(0, 0), Some(0));
        assert_eq!(states.haplotypes.allele(0, 0), Some(1));
        assert_eq!(states.haplotypes.allele(0, 1), Some(0));
    }

    #[test]
    fn imputation_cluster_observation_words_do_not_scale_with_state_count() {
        let mut haplotypes = PackedHaplotypes::empty(65, 1, 1).unwrap();
        for state in 0..65 {
            haplotypes.set_allele(0, state, Some(0)).unwrap();
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0; 65],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let plan = build_hmm_cluster_plan(&[Some(0)], &[0.0], &[0.0], 0.005);

        build_cluster_hmm_panels(&panel, &[Some(0)], &plan).unwrap();
    }

    #[test]
    fn sparse_projection_keeps_states_informative_at_either_cluster_endpoint() {
        let current = [0.0001, 0.2, 0.0001, 0.4];
        let right = [0.0001, 0.0001, 0.2, 0.0];
        let mut active = Vec::new();

        sparse_active_states(&current, Some(&right), 0.001, &mut active);

        assert_eq!(active, vec![1, 2, 3]);
    }

    #[test]
    fn pgen_output_skips_unused_dosage_and_dr2_statistics() {
        let meta = VariantMeta {
            chrom: "1".to_owned(),
            pos: 100,
            id: "v1".to_owned(),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let calls = DiploidCalls::from_calls(vec![
            DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            },
        ]);
        let target = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: meta.clone(),
                calls: calls.clone(),
            }],
            cm_positions: vec![0.1],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let phased_target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta: meta.clone(),
                calls: calls.clone(),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            }],
            emit_start: 0,
            emit_end: 1,
        };
        let reference = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta,
                calls,
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            }],
            emit_start: 0,
            emit_end: 1,
        };
        let config = ImputeConfig {
            states: 2,
            window_markers: 10,
            segment_cm: 6.0,
            step_cm: 0.1,
            nsteps: 7,
            cluster_cm: 0.005,
            output: VcfOutputFields {
                dosage: false,
                allele_probabilities: false,
                genotype_probabilities: false,
            },
        };
        let hmm = HMMConfig {
            ne: 1_000,
            recombination_intensity: 0.000_01,
            err: 0.001,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 4,
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let output = impute_window_prevalidated(
            &target,
            &phased_target,
            reference,
            &[0.1],
            &config,
            &hmm,
            0,
            None,
            None,
            &[false, false],
            &pool,
        )
        .unwrap();
        let row = &output.rows[0];
        assert!(row.dosage.is_none());
        assert!(row.ap1.is_none());
        assert!(row.ap2.is_none());
        assert!(row.gp.is_none());
        assert!(row.info_af.is_none());
        assert!(row.info_dr2.is_none());
    }

    #[test]
    fn reference_missing_call_is_rejected_before_hmm() {
        let meta = VariantMeta {
            chrom: "1".to_owned(),
            pos: 100,
            id: "v1".to_owned(),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let calls = DiploidCalls::from_calls(vec![
            DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            },
        ]);
        let target = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: meta.clone(),
                calls: calls.clone(),
            }],
            cm_positions: vec![0.1],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let phased_target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta: meta.clone(),
                calls,
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            }],
            emit_start: 0,
            emit_end: 1,
        };
        let reference = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta,
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                    DiploidCall::missing(true),
                ]),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            }],
            emit_start: 0,
            emit_end: 1,
        };
        let config = ImputeConfig {
            states: 2,
            window_markers: 10,
            segment_cm: 6.0,
            step_cm: 0.1,
            nsteps: 7,
            cluster_cm: 0.005,
            output: VcfOutputFields {
                dosage: false,
                allele_probabilities: false,
                genotype_probabilities: false,
            },
        };
        let hmm = HMMConfig {
            ne: 1_000,
            recombination_intensity: 0.000_01,
            err: 0.001,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 4,
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let error = impute_window_prevalidated(
            &target,
            &phased_target,
            reference,
            &[0.1],
            &config,
            &hmm,
            0,
            None,
            None,
            &[false, false],
            &pool,
        )
        .unwrap_err();

        assert_eq!(
            error,
            "reference GT must be phased and nonmissing at 1:100 sample 2"
        );
    }
}

#[cfg(test)]
mod interpolation_fast_path_tests {
    use super::{
        interpolated_biallelic_complete_words_probability,
        interpolated_biallelic_words_probability, interpolated_complete_mosaic_probability,
        interpolated_panel_allele_probability, panel_allele_probability,
    };
    use crate::imp::types::{CompositePanel, PackedHaplotypes};

    fn panel() -> CompositePanel {
        let mut haplotypes = PackedHaplotypes::empty(5, 2, 1).unwrap();
        let rows = [
            [Some(0), Some(1), None, Some(1), Some(0)],
            [Some(1), None, Some(0), Some(1), Some(1)],
        ];
        for (marker, row) in rows.iter().enumerate() {
            for (state, &allele) in row.iter().enumerate() {
                haplotypes.set_allele(marker, state, allele).unwrap();
            }
        }
        CompositePanel {
            haplotypes,
            source_samples: vec![0, 1, 2, 3, 4],
            marker_indices: vec![0, 1],
            allele_counts: vec![2, 2],
        }
    }

    fn sparse_aligned_panel() -> CompositePanel {
        let mut haplotypes = PackedHaplotypes::empty(128, 3, 1).unwrap();
        for marker in 0..3 {
            for state in 0..128 {
                let allele = if marker == 2 {
                    None
                } else {
                    match (state + marker * 5) % 37 {
                        0 => None,
                        1 | 2 | 3 => Some(1),
                        _ => Some(0),
                    }
                };
                haplotypes.set_allele(marker, state, allele).unwrap();
            }
        }
        CompositePanel {
            haplotypes,
            source_samples: (0..128).collect(),
            marker_indices: vec![0, 1, 2],
            allele_counts: vec![2; 3],
        }
    }

    fn dense_aligned_panel() -> CompositePanel {
        let mut haplotypes = PackedHaplotypes::empty(1_600, 1, 1).unwrap();
        for state in 0..1_600 {
            let allele = match state % 11 {
                0 => None,
                1..=5 => Some(1),
                _ => Some(0),
            };
            haplotypes.set_allele(0, state, allele).unwrap();
        }
        CompositePanel {
            haplotypes,
            source_samples: (0..1_600).collect(),
            marker_indices: vec![0],
            allele_counts: vec![2],
        }
    }

    fn dense_complete_panel() -> CompositePanel {
        let mut haplotypes = PackedHaplotypes::empty(1_600, 1, 1).unwrap();
        for state in 0..1_600 {
            haplotypes
                .set_allele(0, state, Some(u16::from((state * 17 + 3) % 11 < 5)))
                .unwrap();
        }
        CompositePanel {
            haplotypes,
            source_samples: (0..1_600).collect(),
            marker_indices: vec![0],
            allele_counts: vec![2],
        }
    }

    fn scalar_panel_allele_probability(
        panel: &CompositePanel,
        marker: usize,
        state_probabilities: &[f32],
        allele: u16,
    ) -> f32 {
        let mut observed_alleles = 0usize;
        let mut observed_alt = 0usize;
        let mut observed_probability = 0.0f32;
        let mut missing_probability = 0.0f32;
        for (state, &probability) in state_probabilities.iter().enumerate() {
            match panel.haplotypes.allele(marker, state) {
                Some(state_allele) => {
                    observed_alleles += 1;
                    observed_alt += usize::from(state_allele == allele);
                    if state_allele == allele {
                        observed_probability += probability;
                    }
                }
                None => missing_probability += probability,
            }
        }
        let fallback_frequency = if observed_alleles == 0 {
            0.5
        } else {
            observed_alt as f32 / observed_alleles as f32
        };
        (observed_probability + missing_probability * fallback_frequency).clamp(0.0, 1.0)
    }

    fn normalized(values: impl IntoIterator<Item = f32>) -> Vec<f32> {
        let mut values = values.into_iter().collect::<Vec<_>>();
        let sum = values.iter().sum::<f32>();
        for value in &mut values {
            *value /= sum;
        }
        values
    }

    #[test]
    fn fused_interpolation_matches_materialize_then_project_within_rounding_tolerance() {
        let panel = panel();
        let left = [0.05f32, 0.15, 0.10, 0.30, 0.40];
        let right = [0.40f32, 0.30, 0.10, 0.15, 0.05];
        for marker in 0..panel.haplotypes.n_markers {
            for alpha in [0.0f32, 0.125, 0.35, 0.5, 0.875, 1.0] {
                let interpolated = left
                    .iter()
                    .zip(right)
                    .map(|(&left, right)| left * (1.0 - alpha) + right * alpha)
                    .collect::<Vec<_>>();
                for allele in [0, 1] {
                    let expected =
                        scalar_panel_allele_probability(&panel, marker, &interpolated, allele);
                    let observed = interpolated_panel_allele_probability(
                        &panel, marker, &left, &right, alpha, allele,
                    );
                    assert!(
                        (observed - expected).abs() <= 2.0e-6,
                        "marker={marker} alpha={alpha} allele={allele}: {observed} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_word_projection_matches_scalar_for_alleles_missing_and_interpolation() {
        let panel = sparse_aligned_panel();
        let left = normalized((0..128).map(|state| ((state * 13) % 127 + 1) as f32));
        let right = normalized((0..128).map(|state| ((state * 29) % 113 + 1) as f32));
        for marker in 0..panel.haplotypes.n_markers {
            for alpha in [0.0f32, 0.125, 0.35, 0.5, 0.875, 1.0] {
                let interpolated = left
                    .iter()
                    .zip(&right)
                    .map(|(&left, &right)| left * (1.0 - alpha) + right * alpha)
                    .collect::<Vec<_>>();
                for allele in [0, 1] {
                    let expected =
                        scalar_panel_allele_probability(&panel, marker, &interpolated, allele);
                    let observed = panel_allele_probability(&panel, marker, &interpolated, allele);
                    assert_eq!(observed.to_bits(), expected.to_bits());

                    let interpolated_observed = interpolated_panel_allele_probability(
                        &panel, marker, &left, &right, alpha, allele,
                    );
                    assert!(
                        (interpolated_observed - expected).abs() <= 2.0e-6,
                        "marker={marker} alpha={alpha} allele={allele}: {interpolated_observed} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn interpolated_word_projection_stays_within_tolerance_at_full_donor_count() {
        let panel = dense_aligned_panel();
        let left = normalized((0..1_600).map(|state| ((state * 13) % 127 + 1) as f32));
        let right = normalized((0..1_600).map(|state| ((state * 29) % 113 + 1) as f32));
        let alpha = 0.37f32;
        let interpolated = left
            .iter()
            .zip(&right)
            .map(|(&left, &right)| left * (1.0 - alpha) + right * alpha)
            .collect::<Vec<_>>();

        for allele in [0, 1] {
            let expected = scalar_panel_allele_probability(&panel, 0, &interpolated, allele);
            let observed =
                interpolated_panel_allele_probability(&panel, 0, &left, &right, alpha, allele);
            assert!(
                (observed - expected).abs() <= 2.0e-6,
                "allele={allele}: {observed} != {expected}"
            );
        }
    }

    #[test]
    fn complete_dense_interpolation_projection_stays_within_tolerance() {
        let panel = dense_complete_panel();
        let left = normalized((0..1_600).map(|state| ((state * 23) % 131 + 1) as f32));
        let right = normalized((0..1_600).map(|state| ((state * 31) % 127 + 1) as f32));
        for alpha in [0.0f32, 0.17, 0.37, 0.72, 1.0] {
            let interpolated = left
                .iter()
                .zip(&right)
                .map(|(&left, &right)| left * (1.0 - alpha) + right * alpha)
                .collect::<Vec<_>>();
            let expected = scalar_panel_allele_probability(&panel, 0, &interpolated, 1);
            let observed =
                interpolated_panel_allele_probability(&panel, 0, &left, &right, alpha, 1);
            assert!(
                (observed - expected).abs() <= 2.0e-6,
                "alpha={alpha}: {observed} != {expected}"
            );
        }
    }

    #[test]
    fn packed_word_projection_matches_materialized_panel_projection_exactly() {
        let panel = dense_complete_panel();
        let (allele_words, missing_words) = panel.haplotypes.biallelic_marker_planes(0).unwrap();
        let left = normalized((0..1_600).map(|state| ((state * 19) % 137 + 1) as f32));
        let right = normalized((0..1_600).map(|state| ((state * 23) % 139 + 1) as f32));

        for alpha in [0.0f32, 0.13, 0.37, 0.81, 1.0] {
            for allele in [0, 1] {
                let expected =
                    interpolated_panel_allele_probability(&panel, 0, &left, &right, alpha, allele);
                let observed = interpolated_biallelic_words_probability(
                    allele_words,
                    missing_words,
                    &left,
                    &right,
                    alpha,
                    allele,
                );
                assert_eq!(observed.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn complete_word_projection_matches_missing_aware_simd_path_bit_for_bit() {
        let panel = dense_complete_panel();
        let (allele_words, missing_words) = panel.haplotypes.biallelic_marker_planes(0).unwrap();
        let left = normalized((0..1_600).map(|state| ((state * 31) % 149 + 1) as f32));
        let right = normalized((0..1_600).map(|state| ((state * 43) % 151 + 1) as f32));

        for alpha in [0.0f32, 0.17, 0.37, 0.72, 1.0] {
            for allele in [0, 1] {
                let expected = interpolated_biallelic_words_probability(
                    allele_words,
                    missing_words,
                    &left,
                    &right,
                    alpha,
                    allele,
                );
                let observed = interpolated_biallelic_complete_words_probability(
                    allele_words,
                    &left,
                    &right,
                    alpha,
                    allele,
                );
                assert_eq!(observed.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn fused_mosaic_projection_matches_pack_then_project_bit_for_bit() {
        let reference_words = (0..40usize)
            .map(|word| {
                0x9a5a_c33c_00ff_817eu64.rotate_left((word * 11) as u32)
                    ^ 0x2d2d_4b4b_9696_a5a5u64.rotate_right((word * 7) as u32)
            })
            .collect::<Vec<_>>();
        let left = normalized((0..1_600).map(|state| ((state * 19) % 137 + 1) as f32));
        let right = normalized((0..1_600).map(|state| ((state * 23) % 139 + 1) as f32));

        for state_count in [1usize, 7, 8, 64, 70, 1_600] {
            let donors = (0..state_count)
                .map(|state| (state * 997 + 123) % 2_400)
                .collect::<Vec<_>>();
            let mut allele_words = vec![0u64; state_count.div_ceil(64)];
            for (state, &donor) in donors.iter().enumerate() {
                let allele = reference_words[donor >> 6] >> (donor & 63) & 1;
                allele_words[state >> 6] |= allele << (state & 63);
            }
            for alpha in [0.0f32, 0.17, 0.37, 0.72, 1.0] {
                for allele in [0, 1] {
                    let expected = interpolated_biallelic_complete_words_probability(
                        &allele_words,
                        &left[..state_count],
                        &right[..state_count],
                        alpha,
                        allele,
                    );
                    let observed = interpolated_complete_mosaic_probability(
                        &reference_words,
                        &donors,
                        &left[..state_count],
                        &right[..state_count],
                        alpha,
                        allele,
                    );
                    assert_eq!(
                        observed.to_bits(),
                        expected.to_bits(),
                        "states={state_count} alpha={alpha} allele={allele}"
                    );
                }
            }
        }
    }

    #[test]
    fn complete_word_projection_handles_a_partial_state_word_without_overread() {
        let state_count = 70;
        let allele_words = [0x9a5a_c33c_00ff_817e, 0x2d];
        let left = normalized((0..state_count).map(|state| ((state * 13) % 127 + 1) as f32));
        let right = normalized((0..state_count).map(|state| ((state * 29) % 113 + 1) as f32));
        let alpha = 0.37f32;

        for allele in [0, 1] {
            let mut left_sum = 0.0f32;
            let mut right_sum = 0.0f32;
            for state in 0..state_count {
                let is_alt = allele_words[state / 64] & (1u64 << (state % 64)) != 0;
                if is_alt == (allele == 1) {
                    left_sum += left[state];
                    right_sum += right[state];
                }
            }
            let expected = (left_sum * (1.0 - alpha) + right_sum * alpha).clamp(0.0, 1.0);
            let observed = interpolated_biallelic_complete_words_probability(
                &allele_words,
                &left,
                &right,
                alpha,
                allele,
            );
            assert_eq!(observed.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn complete_word_projection_preserves_scalar_order_for_280_states() {
        let state_count = 280usize;
        let allele_words = (0..state_count.div_ceil(64))
            .map(|word| {
                0x9a5a_c33c_00ff_817eu64.rotate_left((word * 11) as u32)
                    ^ 0x2d2d_4b4b_9696_a5a5u64.rotate_right((word * 7) as u32)
            })
            .collect::<Vec<_>>();
        let left = normalized((0..state_count).map(|state| ((state * 17) % 131 + 1) as f32));
        let right = normalized((0..state_count).map(|state| ((state * 29) % 137 + 1) as f32));

        for alpha in [0.0f32, 0.13, 0.37, 0.81, 1.0] {
            for allele in [0, 1] {
                let mut left_sum = 0.0f32;
                let mut right_sum = 0.0f32;
                for state in 0..state_count {
                    let state_allele = allele_words[state / 64] >> (state % 64) & 1;
                    if state_allele == u64::from(allele) {
                        left_sum += left[state];
                        right_sum += right[state];
                    }
                }
                let expected = (left_sum * (1.0 - alpha) + right_sum * alpha).clamp(0.0, 1.0);
                let observed = interpolated_biallelic_complete_words_probability(
                    &allele_words,
                    &left,
                    &right,
                    alpha,
                    allele,
                );
                assert_eq!(observed.to_bits(), expected.to_bits());
            }
        }
    }
}
