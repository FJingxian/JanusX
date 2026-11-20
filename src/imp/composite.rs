//! Bounded reference-state selection for phase and imputation windows.

use super::pbwt::{LowFrequencyPhasePbwtIndex, PbwtIndex, PhasePbwtIndex, SegmentPbwtIndex};
use super::types::{
    CompositePanel, DiploidCall, Ibd2Interval, PackedHaplotypes, PhaseWindow, PhasedWindow,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

static MOSAIC_SELECT_NANOS: AtomicU64 = AtomicU64::new(0);
static MOSAIC_FILL_NANOS: AtomicU64 = AtomicU64::new(0);
static MOSAIC_SETUP_NANOS: AtomicU64 = AtomicU64::new(0);
static MOSAIC_FALLBACK_NANOS: AtomicU64 = AtomicU64::new(0);
static MOSAIC_SOURCE_NANOS: AtomicU64 = AtomicU64::new(0);
static MOSAIC_SETUP_INPUT_NANOS: AtomicU64 = AtomicU64::new(0);
static MOSAIC_SETUP_PANEL_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_PANEL_PACKED_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_PANEL_SCALAR_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_PANEL_PACKED_CALLS: AtomicU64 = AtomicU64::new(0);
static STAGE2_PANEL_SCALAR_CALLS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn mosaic_detail_profile_nanos() -> [u64; 7] {
    [
        MOSAIC_SELECT_NANOS.load(Ordering::Relaxed),
        MOSAIC_FILL_NANOS.load(Ordering::Relaxed),
        MOSAIC_SETUP_NANOS.load(Ordering::Relaxed),
        MOSAIC_FALLBACK_NANOS.load(Ordering::Relaxed),
        MOSAIC_SOURCE_NANOS.load(Ordering::Relaxed),
        MOSAIC_SETUP_INPUT_NANOS.load(Ordering::Relaxed),
        MOSAIC_SETUP_PANEL_NANOS.load(Ordering::Relaxed),
    ]
}

fn mosaic_detail_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some())
}

pub(crate) fn stage2_panel_profile_totals() -> [u64; 4] {
    [
        STAGE2_PANEL_PACKED_NANOS.load(Ordering::Relaxed),
        STAGE2_PANEL_SCALAR_NANOS.load(Ordering::Relaxed),
        STAGE2_PANEL_PACKED_CALLS.load(Ordering::Relaxed),
        STAGE2_PANEL_SCALAR_CALLS.load(Ordering::Relaxed),
    ]
}

fn stage2_panel_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some()
            || std::env::var_os("JANUSX_IMP_PHASE_PROFILE").is_some()
    })
}

fn stage2_panel_fast_path_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JANUSX_IMP_DISABLE_STAGE2_PANEL_FAST_PATH").is_none())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DonorHapSegment {
    pub(crate) start_marker: usize,
    pub(crate) end_marker: usize,
    pub(crate) donor_haplotype: usize,
}

/// Read-only view over mutable target haplotypes followed by fixed reference
/// haplotypes. Target calls remain in the compact `PhasedWindow`; reference
/// calls are stored once in a packed state-major matrix and are never copied
/// into a combined diploid window.
#[derive(Clone, Copy)]
pub(crate) struct PhaseHaplotypeView<'a> {
    target: &'a PhasedWindow,
    target_packed: Option<&'a PackedHaplotypes>,
    reference_packed: Option<&'a PackedHaplotypes>,
    target_samples: usize,
}

impl<'a> PhaseHaplotypeView<'a> {
    pub(crate) fn new(
        target: &'a PhasedWindow,
        target_packed: Option<&'a PackedHaplotypes>,
        reference_packed: Option<&'a PackedHaplotypes>,
        target_samples: usize,
    ) -> Result<Self, String> {
        let markers = target.rows.len();
        if markers == 0 || target_samples == 0 {
            return Err("phase haplotype view requires positive target dimensions".to_owned());
        }
        if target
            .rows
            .iter()
            .any(|row| row.calls.len() != target_samples)
        {
            return Err("phase haplotype view target sample counts differ".to_owned());
        }
        if let Some(packed) = target_packed {
            if packed.n_markers != markers || packed.n_haplotypes != target_samples * 2 {
                return Err("packed target haplotypes do not match the target view".to_owned());
            }
        }
        if let Some(packed) = reference_packed {
            if packed.n_markers != markers || packed.n_haplotypes % 2 != 0 {
                return Err("packed reference haplotypes do not match the target view".to_owned());
            }
        }
        Ok(Self {
            target,
            target_packed,
            reference_packed,
            target_samples,
        })
    }

    pub(crate) fn target(window: &'a PhasedWindow) -> Self {
        Self {
            target: window,
            target_packed: None,
            reference_packed: None,
            target_samples: window.rows.first().map_or(0, |row| row.calls.len()),
        }
    }

    #[inline(always)]
    pub(crate) fn target_samples(&self) -> usize {
        self.target_samples
    }

    #[inline(always)]
    pub(crate) fn has_reference(&self) -> bool {
        self.reference_packed.is_some()
    }

    #[inline(always)]
    pub(crate) fn n_samples(&self) -> usize {
        self.target_samples
            + self
                .reference_packed
                .map_or(0, |packed| packed.n_haplotypes / 2)
    }

    #[inline(always)]
    pub(crate) fn n_markers(&self) -> usize {
        self.target.rows.len()
    }

    #[inline(always)]
    pub(crate) fn target_window(&self) -> &'a PhasedWindow {
        self.target
    }

    #[inline(always)]
    pub(crate) fn marker(&self, marker: usize) -> &super::types::VariantMeta {
        &self.target.rows[marker].meta
    }

    #[inline(always)]
    pub(crate) fn allele(&self, marker: usize, sample: usize, haplotype: usize) -> Option<u16> {
        if sample < self.target_samples {
            if let Some(packed) = self.target_packed {
                return packed.allele(marker, sample * 2 + haplotype);
            }
            return self.target.rows[marker].calls.get_allele(sample, haplotype);
        }
        let packed = self.reference_packed?;
        packed.allele(marker, (sample - self.target_samples) * 2 + haplotype)
    }

    #[inline(always)]
    pub(crate) fn call(&self, marker: usize, sample: usize) -> DiploidCall {
        DiploidCall {
            alleles: [
                self.allele(marker, sample, 0),
                self.allele(marker, sample, 1),
            ],
            phased: true,
        }
    }

    pub(crate) fn has_complete_state_major_reference(&self) -> bool {
        self.reference_packed.is_some_and(|packed| {
            packed.is_complete()
                && packed.is_state_major()
                && packed.bits_per_allele == 1
                && packed.missing_bits.iter().all(|&word| word == 0)
        })
    }

    #[inline(always)]
    pub(crate) unsafe fn complete_reference_biallelic_pair_unchecked(
        &self,
        marker: usize,
        donor_sample: usize,
        haplotype: usize,
    ) -> [u16; 2] {
        debug_assert!(donor_sample >= self.target_samples);
        debug_assert!(donor_sample < self.n_samples());
        let packed = self
            .reference_packed
            .expect("complete reference pair requires a packed reference");
        let first_state = (donor_sample - self.target_samples) * 2 + haplotype;
        unsafe {
            [
                packed.state_major_biallelic_allele_unchecked(marker, first_state),
                packed.state_major_biallelic_allele_unchecked(marker, first_state ^ 1),
            ]
        }
    }

    #[inline]
    pub(crate) fn copy_biallelic_segment_from(
        &self,
        destination: &mut PackedHaplotypes,
        destination_state: usize,
        start_marker: usize,
        end_marker: usize,
        source_haplotype: usize,
        copy_missing: bool,
    ) -> Result<(), String> {
        if source_haplotype < self.target_samples * 2 && self.target_packed.is_none() {
            let sample = source_haplotype / 2;
            let haplotype = source_haplotype & 1;
            for marker in start_marker..end_marker {
                let allele = self.allele(marker, sample, haplotype);
                destination.set_allele(marker, destination_state, allele)?;
            }
            return Ok(());
        }
        let source = if source_haplotype < self.target_samples * 2 {
            self.target_packed.ok_or_else(|| {
                "packed target haplotypes are required for phase mosaic filling".to_owned()
            })?
        } else {
            self.reference_packed.ok_or_else(|| {
                "packed reference haplotypes are missing from the phase view".to_owned()
            })?
        };
        let local_haplotype = if source_haplotype < self.target_samples * 2 {
            source_haplotype
        } else {
            source_haplotype - self.target_samples * 2
        };
        destination.copy_biallelic_state_segment_from(
            destination_state,
            start_marker,
            end_marker,
            source,
            local_haplotype,
            copy_missing,
        )
    }
}

fn stage2_panel_packed_view_eligible(
    view: &PhaseHaplotypeView<'_>,
    expected_markers: usize,
) -> bool {
    view.n_markers() == expected_markers
        && view.target_packed.is_some_and(|packed| {
            packed.is_state_major() && packed.bits_per_allele == 1 && packed.is_complete()
        })
        && view.reference_packed.map_or(true, |packed| {
            packed.is_state_major() && packed.bits_per_allele == 1 && packed.is_complete()
        })
}

#[derive(Clone, Copy)]
struct ActivePhaseDonor {
    haplotype: usize,
    start_marker: usize,
    last_seen_step: usize,
}

#[derive(Clone, Copy)]
struct DonorQueueEntry {
    slot: usize,
    queued_last_seen_step: usize,
}

const NO_DONOR_SLOT: u32 = u32::MAX;

/// Java `PriorityQueue` semantics for Beagle's `CompHapSegment` comparator.
/// Equal `lastIbsStep` entries keep their heap insertion order; choosing the
/// smallest haplotype ID on ties changes which composite state is evicted.
#[derive(Default)]
struct BeagleDonorQueue {
    heap: Vec<DonorQueueEntry>,
}

impl BeagleDonorQueue {
    #[inline(always)]
    fn len(&self) -> usize {
        self.heap.len()
    }

    #[inline(always)]
    fn peek(&self) -> Option<DonorQueueEntry> {
        self.heap.first().copied()
    }

    #[inline(always)]
    fn push(&mut self, entry: DonorQueueEntry) {
        let mut index = self.heap.len();
        self.heap.push(entry);
        while index > 0 {
            let parent = (index - 1) / 2;
            // SAFETY: a binary heap parent index is below the current length.
            let parent_entry = unsafe { *self.heap.get_unchecked(parent) };
            if entry.queued_last_seen_step >= parent_entry.queued_last_seen_step {
                break;
            }
            // SAFETY: `index` is below the heap length.
            unsafe {
                *self.heap.get_unchecked_mut(index) = parent_entry;
            }
            index = parent;
        }
        // SAFETY: `index` was either the original push position or a parent.
        unsafe {
            *self.heap.get_unchecked_mut(index) = entry;
        }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<DonorQueueEntry> {
        let result = *self.heap.first()?;
        let last = self.heap.pop().unwrap();
        if !self.heap.is_empty() {
            let mut index = 0;
            let half = self.heap.len() / 2;
            while index < half {
                let left = index * 2 + 1;
                let right = left + 1;
                // SAFETY: `left < heap.len()` because `index < heap.len()/2`.
                let left_entry = unsafe { *self.heap.get_unchecked(left) };
                let right_entry =
                    (right < self.heap.len()).then(|| unsafe { *self.heap.get_unchecked(right) });
                let child = if right_entry.is_some_and(|entry| {
                    entry.queued_last_seen_step < left_entry.queued_last_seen_step
                }) {
                    right
                } else {
                    left
                };
                // SAFETY: `child` is a valid existing child.
                let child_entry = unsafe { *self.heap.get_unchecked(child) };
                if last.queued_last_seen_step <= child_entry.queued_last_seen_step {
                    break;
                }
                // SAFETY: both indices are in bounds.
                unsafe {
                    *self.heap.get_unchecked_mut(index) = child_entry;
                }
                index = child;
            }
            // SAFETY: `index` is the original root or a valid child index.
            unsafe {
                *self.heap.get_unchecked_mut(index) = last;
            }
        }
        Some(result)
    }

    /// Refresh stale heap priorities exactly as Beagle's `updateHeadOfQ` does.
    /// Existing donors update the hap-to-last-step map, while their queue node
    /// is lazily updated only when it reaches the queue head.
    #[inline(always)]
    fn update_head(&mut self, active: &[ActivePhaseDonor]) {
        loop {
            let Some(head) = self.peek() else {
                break;
            };
            // SAFETY: every queue entry stores an active slot.
            let current_step = unsafe { active.get_unchecked(head.slot) }.last_seen_step;
            if head.queued_last_seen_step == current_step {
                break;
            }
            let mut refreshed = self.pop().unwrap();
            refreshed.queued_last_seen_step = current_step;
            self.push(refreshed);
        }
    }
}

#[derive(Default)]
pub(crate) struct DonorSegmentWorkspace {
    active: Vec<ActivePhaseDonor>,
    first_haplotypes: Vec<usize>,
    donor_to_slot: Vec<u32>,
    segments: Vec<Vec<DonorHapSegment>>,
    queue: BeagleDonorQueue,
}

impl DonorSegmentWorkspace {
    pub(crate) fn segments(&self, count: usize) -> &[Vec<DonorHapSegment>] {
        &self.segments[..count]
    }
}

pub(crate) fn select_donor_segments(
    step_ranges: &[(usize, usize)],
    proposals: &[[Option<usize>; 2]],
    n_samples: usize,
    target_sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
) -> Result<Vec<Vec<DonorHapSegment>>, String> {
    let mut workspace = DonorSegmentWorkspace::default();
    let count = select_donor_segments_into(
        step_ranges,
        proposals,
        n_samples,
        target_sample,
        max_states,
        min_lifetime_steps,
        &mut workspace,
    )?;
    Ok(workspace.segments(count).to_vec())
}

pub(crate) fn select_donor_segments_into(
    step_ranges: &[(usize, usize)],
    proposals: &[[Option<usize>; 2]],
    n_samples: usize,
    target_sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    workspace: &mut DonorSegmentWorkspace,
) -> Result<usize, String> {
    select_donor_segments_emit::<true, false, _, _>(
        step_ranges,
        proposals.len(),
        |step| proposals[step].map(|donor| donor.map_or(-1, |donor| donor as i32)),
        n_samples,
        target_sample,
        max_states,
        min_lifetime_steps,
        workspace,
        |_, _, _, _| Ok(()),
    )
}

/// Build Beagle-style composite donor tracks from the complete IBS candidate
/// set at every imputation step. State slots are replaced in last-seen-step
/// order, with donor switches placed halfway between the old and new IBS hits.
pub(crate) fn select_ibs_donor_segments(
    step_ranges: &[(usize, usize)],
    candidate_sets: &[Vec<usize>],
    n_reference_haplotypes: usize,
    max_states: usize,
) -> Result<Vec<Vec<DonorHapSegment>>, String> {
    if step_ranges.is_empty()
        || step_ranges.len() != candidate_sets.len()
        || step_ranges[0].0 != 0
        || step_ranges.iter().any(|&(start, end)| start >= end)
        || step_ranges.windows(2).any(|pair| pair[0].1 != pair[1].0)
        || n_reference_haplotypes > u32::MAX as usize
        || max_states == 0
    {
        return Err("imputation IBS donor steps or state limit are invalid".to_owned());
    }
    if candidate_sets.iter().any(|candidates| {
        candidates
            .iter()
            .any(|&donor| donor >= n_reference_haplotypes)
            || candidates.windows(2).any(|pair| pair[0] >= pair[1])
    }) {
        return Err(
            "imputation IBS candidates must be unique, sorted reference haplotypes".to_owned(),
        );
    }

    let mut workspace = DonorSegmentWorkspace::default();
    workspace
        .donor_to_slot
        .resize(n_reference_haplotypes, NO_DONOR_SLOT);
    workspace
        .active
        .reserve(max_states.min(n_reference_haplotypes));
    workspace
        .segments
        .reserve(max_states.min(n_reference_haplotypes));
    workspace
        .queue
        .heap
        .reserve(max_states.min(n_reference_haplotypes));

    for (step, candidates) in candidate_sets.iter().enumerate() {
        for &donor in candidates {
            let mapped_slot = workspace.donor_to_slot[donor];
            if mapped_slot != NO_DONOR_SLOT {
                workspace.active[mapped_slot as usize].last_seen_step = step;
                continue;
            }

            workspace.queue.update_head(&workspace.active);
            if workspace.queue.len() == max_states {
                let old_entry = workspace.queue.pop().unwrap();
                let slot = old_entry.slot;
                let old = workspace.active[slot];
                let midpoint_step = (old.last_seen_step + step) / 2;
                let boundary = step_ranges[midpoint_step].0.max(old.start_marker);
                if boundary > old.start_marker {
                    workspace.segments[slot].push(DonorHapSegment {
                        start_marker: old.start_marker,
                        end_marker: boundary,
                        donor_haplotype: old.haplotype,
                    });
                }
                workspace.donor_to_slot[old.haplotype] = NO_DONOR_SLOT;
                workspace.active[slot] = ActivePhaseDonor {
                    haplotype: donor,
                    start_marker: boundary,
                    last_seen_step: step,
                };
                workspace.donor_to_slot[donor] = slot as u32;
                workspace.queue.push(DonorQueueEntry {
                    slot,
                    queued_last_seen_step: step,
                });
            } else {
                let slot = workspace.queue.len();
                workspace.active.push(ActivePhaseDonor {
                    haplotype: donor,
                    start_marker: 0,
                    last_seen_step: step,
                });
                workspace.segments.push(Vec::new());
                workspace.donor_to_slot[donor] = slot as u32;
                workspace.queue.push(DonorQueueEntry {
                    slot,
                    queued_last_seen_step: step,
                });
            }
        }
    }

    let marker_count = step_ranges.last().unwrap().1;
    for (slot, state) in workspace.active.iter().copied().enumerate() {
        if state.start_marker < marker_count {
            workspace.segments[slot].push(DonorHapSegment {
                start_marker: state.start_marker,
                end_marker: marker_count,
                donor_haplotype: state.haplotype,
            });
        }
    }
    Ok(workspace.segments)
}

pub(crate) fn ibs_fallback_donor_haplotypes(
    n_reference_haplotypes: usize,
    max_states: usize,
    target_haplotype: usize,
    seed: u64,
) -> Vec<usize> {
    let state_count = max_states.min(n_reference_haplotypes);
    // Target and reference haplotype indices belong to different panels.
    // Without an explicit identity map, no reference index is a self match.
    let mut random = super::pbwt::JavaRandom::new(seed.wrapping_add(target_haplotype as u64));
    (0..state_count)
        .map(|_| random.next_int(n_reference_haplotypes))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn select_donor_segments_emit<const RECORD_SEGMENTS: bool, const INPUT_VALIDATED: bool, F, G>(
    step_ranges: &[(usize, usize)],
    proposal_count: usize,
    mut proposal_at: F,
    n_samples: usize,
    target_sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    workspace: &mut DonorSegmentWorkspace,
    mut emit_segment: G,
) -> Result<usize, String>
where
    F: FnMut(usize) -> [i32; 2],
    G: FnMut(usize, usize, usize, usize) -> Result<(), String>,
{
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "phase donor haplotype count overflow".to_owned())?;
    if INPUT_VALIDATED {
        debug_assert!(!step_ranges.is_empty());
        debug_assert_eq!(proposal_count, step_ranges.len());
        debug_assert!(n_samples > 0 && target_sample < n_samples);
        debug_assert!(max_states > 0 && max_states <= u32::MAX as usize);
        debug_assert_eq!(step_ranges[0].0, 0);
        debug_assert!(step_ranges.iter().all(|&(start, end)| start < end));
        debug_assert!(step_ranges.windows(2).all(|pair| pair[0].1 == pair[1].0));
    } else if step_ranges.is_empty()
        || proposal_count != step_ranges.len()
        || n_samples == 0
        || target_sample >= n_samples
        || max_states == 0
        || max_states > u32::MAX as usize
        || step_ranges[0].0 != 0
        || step_ranges.iter().any(|&(start, end)| start >= end)
        || step_ranges.windows(2).any(|pair| pair[0].1 != pair[1].0)
    {
        return Err("phase donor steps, target, or state limit are invalid".to_owned());
    }

    if workspace.donor_to_slot.len() != n_haplotypes {
        workspace.donor_to_slot.clear();
        workspace.donor_to_slot.resize(n_haplotypes, NO_DONOR_SLOT);
    } else {
        for donor in &workspace.active {
            workspace.donor_to_slot[donor.haplotype] = NO_DONOR_SLOT;
        }
    }
    workspace.active.clear();
    workspace.first_haplotypes.clear();
    workspace.queue.heap.clear();
    if RECORD_SEGMENTS {
        for segments in &mut workspace.segments {
            segments.clear();
        }
    }
    let min_lifetime_steps = min_lifetime_steps.max(1);

    for step in 0..proposal_count {
        let step_proposals = proposal_at(step);
        for raw_donor in step_proposals {
            if raw_donor < 0 {
                continue;
            }
            let donor = raw_donor as usize;
            if donor >= n_haplotypes {
                return Err(format!("phase donor haplotype {donor} is out of range"));
            }
            // SAFETY: `donor` was checked against `n_haplotypes`.
            let mapped_slot = unsafe { *workspace.donor_to_slot.get_unchecked(donor) };
            if mapped_slot != NO_DONOR_SLOT {
                let slot = mapped_slot as usize;
                // SAFETY: donor-to-slot mappings refer to active slots.
                unsafe { workspace.active.get_unchecked_mut(slot) }.last_seen_step = step;
                continue;
            }

            workspace.queue.update_head(&workspace.active);
            let oldest = workspace.queue.peek();
            let replace = oldest.is_some_and(|entry| {
                workspace.queue.len() == max_states
                    || step.saturating_sub(
                        // SAFETY: queue entries refer to active slots.
                        unsafe { workspace.active.get_unchecked(entry.slot) }.last_seen_step,
                    ) >= min_lifetime_steps
            });
            if replace {
                let old_entry = workspace.queue.pop().unwrap();
                let slot = old_entry.slot;
                // SAFETY: queue entries refer to active slots.
                let old = unsafe { *workspace.active.get_unchecked(slot) };
                let midpoint_step = (old.last_seen_step + step) / 2;
                // SAFETY: midpoint is between two proposal steps.
                let boundary = unsafe { step_ranges.get_unchecked(midpoint_step) }
                    .0
                    .max(old.start_marker);
                if boundary > old.start_marker {
                    if RECORD_SEGMENTS {
                        workspace.segments[slot].push(DonorHapSegment {
                            start_marker: old.start_marker,
                            end_marker: boundary,
                            donor_haplotype: old.haplotype,
                        });
                    }
                    emit_segment(slot, old.start_marker, boundary, old.haplotype)?;
                }
                // SAFETY: both mappings contain validated haplotype IDs.
                unsafe {
                    *workspace.donor_to_slot.get_unchecked_mut(old.haplotype) = NO_DONOR_SLOT;
                    *workspace.active.get_unchecked_mut(slot) = ActivePhaseDonor {
                        haplotype: donor,
                        start_marker: boundary,
                        last_seen_step: step,
                    };
                    *workspace.donor_to_slot.get_unchecked_mut(donor) = slot as u32;
                }
                workspace.queue.push(DonorQueueEntry {
                    slot,
                    queued_last_seen_step: step,
                });
            } else {
                let slot = workspace.queue.len();
                workspace.active.push(ActivePhaseDonor {
                    haplotype: donor,
                    start_marker: 0,
                    last_seen_step: step,
                });
                if slot == workspace.first_haplotypes.len() {
                    workspace.first_haplotypes.push(donor);
                } else {
                    workspace.first_haplotypes[slot] = donor;
                }
                workspace.donor_to_slot[donor] = slot as u32;
                if RECORD_SEGMENTS && slot == workspace.segments.len() {
                    workspace.segments.push(Vec::new());
                }
                workspace.queue.push(DonorQueueEntry {
                    slot,
                    queued_last_seen_step: step,
                });
            }
        }
    }

    let marker_count = step_ranges.last().unwrap().1;
    for (slot, state) in workspace.active.iter().copied().enumerate() {
        if state.start_marker < marker_count {
            if RECORD_SEGMENTS {
                workspace.segments[slot].push(DonorHapSegment {
                    start_marker: state.start_marker,
                    end_marker: marker_count,
                    donor_haplotype: state.haplotype,
                });
            }
            emit_segment(slot, state.start_marker, marker_count, state.haplotype)?;
        }
    }
    Ok(workspace.active.len())
}

fn trace_phase_composite_target(view: &PhaseHaplotypeView<'_>, target_sample: usize) -> bool {
    std::env::var("JANUSX_TRACE_COMPOSITE_DONORS").is_ok_and(|targets| {
        targets
            .split(',')
            .any(|target| target == format!("{}:{target_sample}", view.target_window().index))
    })
}

/// Build a shared phase donor panel from IBS candidates for both target
/// haplotypes. A state retains its donor until it is evicted by the recency
/// queue; replacement boundaries fall halfway between the old and new IBS
/// steps. This keeps the genotype and two haplotype tracks on one state space.
pub(crate) fn build_joint_phase_mosaic(
    window: &PhasedWindow,
    pbwt_index: &PhasePbwtIndex,
    target_sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
    packed_window: Option<&PackedHaplotypes>,
) -> Result<CompositePanel, String> {
    let view = PhaseHaplotypeView::new(
        window,
        packed_window,
        None,
        window.rows.first().map_or(0, |row| row.calls.len()),
    )?;
    build_joint_phase_mosaic_view(
        &view,
        pbwt_index,
        target_sample,
        max_states,
        min_lifetime_steps,
        seed,
    )
}

pub(crate) fn build_joint_phase_mosaic_view(
    view: &PhaseHaplotypeView<'_>,
    pbwt_index: &PhasePbwtIndex,
    target_sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
) -> Result<CompositePanel, String> {
    if view.n_markers() == 0 || pbwt_index.step_ranges.is_empty() || max_states == 0 {
        return Err("joint phase mosaic requires a nonempty window and donor limit".to_owned());
    }
    let n_samples = view.n_samples();
    if n_samples == 0
        || target_sample >= view.target_samples()
        || pbwt_index.n_haplotypes != n_samples.saturating_mul(2)
    {
        return Err("joint phase mosaic dimensions do not match the target".to_owned());
    }

    let step_ranges = &pbwt_index.step_ranges;
    if step_ranges[0].0 != 0
        || step_ranges.last().map(|range| range.1) != Some(view.n_markers())
        || step_ranges
            .iter()
            .any(|&(start, end)| start >= end || end > view.n_markers())
        || step_ranges.windows(2).any(|pair| pair[0].1 != pair[1].0)
    {
        return Err("phase PBWT steps do not form an ordered window partition".to_owned());
    }
    let mut proposals = Vec::with_capacity(step_ranges.len());
    let all_haplotypes = n_samples * 2;
    for step in 0..step_ranges.len() {
        proposals.push([
            pbwt_index.donor_haplotype(step, 2 * target_sample),
            pbwt_index.donor_haplotype(step, 2 * target_sample + 1),
        ]);
    }
    if trace_phase_composite_target(view, target_sample) {
        eprintln!(
            "JANUSX_COMPOSITE_DONORS\tWINDOW={}:{}\tSAMPLE={target_sample}\tPROPOSALS={proposals:?}",
            view.target_window().chrom,
            view.target_window().index
        );
    }
    let state_segments = select_donor_segments(
        step_ranges,
        &proposals,
        n_samples,
        target_sample,
        max_states.min(all_haplotypes.saturating_sub(1)),
        min_lifetime_steps,
    )?;
    if trace_phase_composite_target(view, target_sample) {
        eprintln!(
            "JANUSX_COMPOSITE_SEGMENTS\tWINDOW={}:{}\tSAMPLE={target_sample}\tSEGMENTS={state_segments:?}",
            view.target_window().chrom,
            view.target_window().index
        );
    }
    if state_segments.is_empty() {
        return build_random_phase_panel_view(view, target_sample, max_states, seed);
    }
    let max_allele = (0..view.n_markers())
        .map(|marker| view.marker(marker).allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let mut haplotypes = PackedHaplotypes::empty_state_major(
        state_segments.len(),
        view.n_markers(),
        allele_width(max_allele),
    )?;
    let source_samples = fill_joint_phase_mosaic_view(view, &state_segments, &mut haplotypes)?;
    Ok(CompositePanel {
        haplotypes,
        source_samples,
        marker_indices: (0..view.n_markers()).collect(),
        allele_counts: (0..view.n_markers())
            .map(|marker| view.marker(marker).allele_count())
            .collect(),
    })
}

/// Fill a reusable state-major phase panel while retaining its bit storage.
pub(crate) fn build_joint_phase_mosaic_reuse(
    window: &PhasedWindow,
    pbwt_index: &PhasePbwtIndex,
    target_sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
    packed_window: Option<&PackedHaplotypes>,
    panel_slot: &mut Option<CompositePanel>,
    donor_workspace: &mut DonorSegmentWorkspace,
) -> Result<(), String> {
    let view = PhaseHaplotypeView::new(
        window,
        packed_window,
        None,
        window.rows.first().map_or(0, |row| row.calls.len()),
    )?;
    build_joint_phase_mosaic_reuse_impl::<false>(
        &view,
        pbwt_index,
        target_sample,
        max_states,
        min_lifetime_steps,
        seed,
        panel_slot,
        donor_workspace,
    )
}

/// Sample-hot variant for callers that have validated the window once and
/// obtained the PBWT index for this exact snapshot.
pub(crate) fn build_joint_phase_mosaic_reuse_validated(
    window: &PhasedWindow,
    pbwt_index: &PhasePbwtIndex,
    target_sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
    packed_window: Option<&PackedHaplotypes>,
    panel_slot: &mut Option<CompositePanel>,
    donor_workspace: &mut DonorSegmentWorkspace,
) -> Result<(), String> {
    let view = PhaseHaplotypeView::new(
        window,
        packed_window,
        None,
        window.rows.first().map_or(0, |row| row.calls.len()),
    )?;
    build_joint_phase_mosaic_reuse_impl::<true>(
        &view,
        pbwt_index,
        target_sample,
        max_states,
        min_lifetime_steps,
        seed,
        panel_slot,
        donor_workspace,
    )
}

pub(crate) fn build_joint_phase_mosaic_reuse_validated_view(
    view: &PhaseHaplotypeView<'_>,
    pbwt_index: &PhasePbwtIndex,
    target_sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
    panel_slot: &mut Option<CompositePanel>,
    donor_workspace: &mut DonorSegmentWorkspace,
) -> Result<(), String> {
    build_joint_phase_mosaic_reuse_impl::<true>(
        view,
        pbwt_index,
        target_sample,
        max_states,
        min_lifetime_steps,
        seed,
        panel_slot,
        donor_workspace,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_joint_phase_mosaic_reuse_impl<const INPUT_VALIDATED: bool>(
    view: &PhaseHaplotypeView<'_>,
    pbwt_index: &PhasePbwtIndex,
    target_sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
    panel_slot: &mut Option<CompositePanel>,
    donor_workspace: &mut DonorSegmentWorkspace,
) -> Result<(), String> {
    let detail_profile = mosaic_detail_profiling_enabled();
    let setup_started = detail_profile.then(Instant::now);
    let Some(target_packed) = view.target_packed.filter(|packed| packed.is_state_major()) else {
        *panel_slot = Some(build_joint_phase_mosaic_view(
            view,
            pbwt_index,
            target_sample,
            max_states,
            min_lifetime_steps,
            seed,
        )?);
        return Ok(());
    };
    if view.n_markers() == 0 || pbwt_index.step_ranges.is_empty() || max_states == 0 {
        return Err("joint phase mosaic requires a nonempty window and donor limit".to_owned());
    }
    let n_samples = view.n_samples();
    let step_ranges = &pbwt_index.step_ranges;
    if INPUT_VALIDATED {
        debug_assert!(n_samples > 0 && target_sample < view.target_samples());
        debug_assert_eq!(pbwt_index.n_haplotypes, n_samples.saturating_mul(2));
        debug_assert_eq!(step_ranges[0].0, 0);
        debug_assert_eq!(
            step_ranges.last().map(|range| range.1),
            Some(view.n_markers())
        );
        debug_assert!(step_ranges
            .iter()
            .all(|&(start, end)| start < end && end <= view.n_markers()));
        debug_assert!(step_ranges.windows(2).all(|pair| pair[0].1 == pair[1].0));
        debug_assert_eq!(target_packed.bits_per_allele, 1);
        debug_assert!(target_packed.is_complete());
    } else if n_samples == 0
        || target_sample >= view.target_samples()
        || pbwt_index.n_haplotypes != n_samples.saturating_mul(2)
    {
        return Err("joint phase mosaic dimensions do not match the target".to_owned());
    }
    if !INPUT_VALIDATED
        && (step_ranges[0].0 != 0
            || step_ranges.last().map(|range| range.1) != Some(view.n_markers())
            || step_ranges
                .iter()
                .any(|&(start, end)| start >= end || end > view.n_markers())
            || step_ranges.windows(2).any(|pair| pair[0].1 != pair[1].0))
    {
        return Err("phase PBWT steps do not form an ordered window partition".to_owned());
    }
    if let Some(started) = setup_started {
        MOSAIC_SETUP_INPUT_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let setup_panel_started = detail_profile.then(Instant::now);
    let all_haplotypes = n_samples.saturating_mul(2);
    let marker_count = view.n_markers();
    let panel = panel_slot.get_or_insert_with(|| CompositePanel {
        haplotypes: PackedHaplotypes::empty_state_major(1, 1, 1)
            .expect("constant phase panel dimensions are valid"),
        source_samples: Vec::new(),
        marker_indices: Vec::new(),
        allele_counts: Vec::new(),
    });
    let refresh_allele_counts = panel.allele_counts.len() != marker_count;
    let max_allele = if INPUT_VALIDATED {
        1
    } else if refresh_allele_counts {
        (0..view.n_markers())
            .map(|marker| view.marker(marker).allele_count().saturating_sub(1) as u16)
            .max()
            .unwrap_or(1)
    } else {
        panel
            .allele_counts
            .iter()
            .map(|&count| count.saturating_sub(1) as u16)
            .max()
            .unwrap_or(1)
    };
    if allele_width(max_allele) != 1 {
        return Err("packed phase donor panel requires biallelic alleles".to_owned());
    }
    let state_capacity = max_states.min(all_haplotypes.saturating_sub(1));
    panel
        .haplotypes
        .resize_state_major_biallelic_for_overwrite(state_capacity, marker_count)?;
    let complete = target_packed.is_complete()
        && view
            .reference_packed
            .map_or(true, |packed| packed.is_complete());
    if let Some(started) = setup_panel_started {
        MOSAIC_SETUP_PANEL_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let Some(started) = setup_started {
        MOSAIC_SETUP_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let select_started = detail_profile.then(Instant::now);
    let state_count = select_donor_segments_emit::<false, INPUT_VALIDATED, _, _>(
        step_ranges,
        step_ranges.len(),
        |step| unsafe { pbwt_index.donor_haplotype_pair_raw_unchecked(step, 2 * target_sample) },
        n_samples,
        target_sample,
        state_capacity,
        min_lifetime_steps,
        donor_workspace,
        |state, start, end, donor_haplotype| {
            view.copy_biallelic_segment_from(
                &mut panel.haplotypes,
                state,
                start,
                end,
                donor_haplotype,
                !complete,
            )
        },
    )?;
    if let Some(started) = select_started {
        MOSAIC_SELECT_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if trace_phase_composite_target(view, target_sample) {
        let first_target_haplotype = 2 * target_sample;
        let proposals = (0..step_ranges.len())
            .map(|step| pbwt_index.donor_haplotype_pair(step, first_target_haplotype))
            .collect::<Vec<_>>();
        eprintln!(
            "JANUSX_COMPOSITE_DONORS\tWINDOW={}:{}\tSAMPLE={target_sample}\tPROPOSALS={proposals:?}",
            view.target_window().chrom,
            view.target_window().index
        );
        let state_segments = select_donor_segments(
            step_ranges,
            &proposals,
            n_samples,
            target_sample,
            state_capacity,
            min_lifetime_steps,
        )?;
        eprintln!(
            "JANUSX_COMPOSITE_SEGMENTS\tWINDOW={}:{}\tSAMPLE={target_sample}\tSEGMENTS={state_segments:?}",
            view.target_window().chrom,
            view.target_window().index
        );
    }
    if state_count == 0 {
        let fallback_started = detail_profile.then(Instant::now);
        *panel_slot = Some(build_random_phase_panel_view(
            view,
            target_sample,
            max_states,
            seed,
        )?);
        if let Some(started) = fallback_started {
            MOSAIC_FALLBACK_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        return Ok(());
    }
    let fill_started = detail_profile.then(Instant::now);
    panel
        .haplotypes
        .truncate_state_major_biallelic_for_overwrite(state_count)?;
    if complete {
        panel.haplotypes.mark_complete();
    }
    let source_started = detail_profile.then(Instant::now);
    panel.source_samples.clear();
    panel.source_samples.extend(
        donor_workspace.first_haplotypes[..state_count]
            .iter()
            .map(|&haplotype| haplotype / 2),
    );
    if let Some(started) = source_started {
        MOSAIC_SOURCE_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if panel.marker_indices.len() != marker_count {
        panel.marker_indices.clear();
        panel.marker_indices.extend(0..marker_count);
    }
    if refresh_allele_counts {
        panel.allele_counts.clear();
        panel
            .allele_counts
            .extend((0..view.n_markers()).map(|marker| view.marker(marker).allele_count()));
    }
    if let Some(started) = fill_started {
        MOSAIC_FILL_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    Ok(())
}

fn fill_joint_phase_mosaic(
    window: &PhasedWindow,
    state_segments: &[Vec<DonorHapSegment>],
    haplotypes: &mut PackedHaplotypes,
) -> Result<Vec<usize>, String> {
    let source_samples = validate_joint_phase_segments(window, state_segments)?;
    let marker_count = window.rows.len();
    if haplotypes.n_markers != marker_count || haplotypes.n_haplotypes != state_segments.len() {
        return Err("phase donor mosaic dimensions do not match the window".to_owned());
    }

    let mut segment_indices = vec![0usize; state_segments.len()];
    for marker in 0..marker_count {
        let calls = &window.rows[marker].calls;
        if haplotypes.bits_per_allele == 1 {
            haplotypes.set_biallelic_marker_with(marker, |state| {
                let segments = &state_segments[state];
                let mut segment_index = segment_indices[state];
                while marker >= segments[segment_index].end_marker {
                    segment_index += 1;
                }
                segment_indices[state] = segment_index;
                let donor_haplotype = segments[segment_index].donor_haplotype;
                calls.get_allele(donor_haplotype / 2, donor_haplotype & 1)
            })?;
        } else {
            for (state, segments) in state_segments.iter().enumerate() {
                let mut segment_index = segment_indices[state];
                while marker >= segments[segment_index].end_marker {
                    segment_index += 1;
                }
                segment_indices[state] = segment_index;
                let donor_haplotype = segments[segment_index].donor_haplotype;
                let allele = calls.get_allele(donor_haplotype / 2, donor_haplotype & 1);
                haplotypes.set_allele(marker, state, allele)?;
            }
        }
    }
    Ok(source_samples)
}

fn fill_joint_phase_mosaic_view(
    view: &PhaseHaplotypeView<'_>,
    state_segments: &[Vec<DonorHapSegment>],
    haplotypes: &mut PackedHaplotypes,
) -> Result<Vec<usize>, String> {
    let source_samples = validate_joint_phase_segments_view(view, state_segments)?;
    if haplotypes.n_markers != view.n_markers()
        || haplotypes.n_haplotypes != state_segments.len()
        || !haplotypes.is_state_major()
        || haplotypes.bits_per_allele != 1
    {
        return Err("phase donor mosaic dimensions do not match the haplotype view".to_owned());
    }
    let complete = view.target_packed.map_or_else(
        || {
            view.target
                .rows
                .iter()
                .all(|row| row.calls.iter().all(|call| call.is_complete()))
        },
        PackedHaplotypes::is_complete,
    ) && view
        .reference_packed
        .map_or(true, PackedHaplotypes::is_complete);
    if complete {
        haplotypes.missing_bits.fill(0);
    }
    for (state, segments) in state_segments.iter().enumerate() {
        for segment in segments {
            view.copy_biallelic_segment_from(
                haplotypes,
                state,
                segment.start_marker,
                segment.end_marker,
                segment.donor_haplotype,
                !complete,
            )?;
        }
    }
    if complete {
        haplotypes.mark_complete();
    }
    Ok(source_samples)
}

fn validate_joint_phase_segments_view(
    view: &PhaseHaplotypeView<'_>,
    state_segments: &[Vec<DonorHapSegment>],
) -> Result<Vec<usize>, String> {
    let marker_count = view.n_markers();
    let all_haplotypes = view
        .n_samples()
        .checked_mul(2)
        .ok_or_else(|| "phase donor haplotype count overflow".to_owned())?;
    let mut source_samples = Vec::with_capacity(state_segments.len());
    for segments in state_segments {
        let Some(first) = segments.first() else {
            return Err("phase donor state has no covered markers".to_owned());
        };
        source_samples.push(first.donor_haplotype / 2);
        let mut covered = 0usize;
        for segment in segments {
            if segment.start_marker != covered
                || segment.start_marker >= segment.end_marker
                || segment.end_marker > marker_count
                || segment.donor_haplotype >= all_haplotypes
            {
                return Err("phase donor segments do not form a valid partition".to_owned());
            }
            covered = segment.end_marker;
        }
        if covered != marker_count {
            return Err("phase donor state does not cover the full window".to_owned());
        }
    }
    Ok(source_samples)
}

fn validate_joint_phase_segments(
    window: &PhasedWindow,
    state_segments: &[Vec<DonorHapSegment>],
) -> Result<Vec<usize>, String> {
    let marker_count = window.rows.len();
    if marker_count == 0
        || window
            .rows
            .iter()
            .any(|row| row.calls.len() != window.rows[0].calls.len())
    {
        return Err("phase donor mosaic dimensions do not match the window".to_owned());
    }
    let all_haplotypes = window.rows[0]
        .calls
        .len()
        .checked_mul(2)
        .ok_or_else(|| "phase donor haplotype count overflow".to_owned())?;
    let mut source_samples = Vec::with_capacity(state_segments.len());
    for segments in state_segments {
        let Some(first) = segments.first() else {
            return Err("phase donor state has no covered markers".to_owned());
        };
        source_samples.push(first.donor_haplotype / 2);
        let mut covered = 0usize;
        for segment in segments {
            if segment.start_marker != covered
                || segment.start_marker >= segment.end_marker
                || segment.end_marker > marker_count
                || segment.donor_haplotype >= all_haplotypes
            {
                return Err("phase donor segments do not form a valid partition".to_owned());
            }
            covered = segment.end_marker;
        }
        if covered != marker_count {
            return Err("phase donor state does not cover the full window".to_owned());
        }
    }
    Ok(source_samples)
}

fn fill_joint_phase_mosaic_from_packed(
    window: &PhasedWindow,
    packed_window: &PackedHaplotypes,
    state_segments: &[Vec<DonorHapSegment>],
    haplotypes: &mut PackedHaplotypes,
) -> Result<Vec<usize>, String> {
    let source_samples = validate_joint_phase_segments(window, state_segments)?;
    let marker_count = window.rows.len();
    let state_count = state_segments.len();
    if !packed_window.is_state_major()
        || packed_window.n_markers != marker_count
        || packed_window.n_haplotypes
            != window.rows[0]
                .calls
                .len()
                .checked_mul(2)
                .ok_or_else(|| "phase donor haplotype count overflow".to_owned())?
        || packed_window.bits_per_allele != 1
        || haplotypes.n_markers != marker_count
        || haplotypes.n_haplotypes != state_count
        || haplotypes.bits_per_allele != 1
        || !haplotypes.is_state_major()
    {
        return Err("packed phase donor mosaic dimensions do not match the window".to_owned());
    }

    let complete = packed_window.is_complete();
    if complete {
        // Complete packed windows have an all-zero missingness plane. Clear it
        // once instead of copying zeroes for every donor segment and state.
        haplotypes.missing_bits.fill(0);
    }
    for (state, segments) in state_segments.iter().enumerate() {
        for segment in segments {
            haplotypes.copy_biallelic_state_segment_from(
                state,
                segment.start_marker,
                segment.end_marker,
                packed_window,
                segment.donor_haplotype,
                !complete,
            )?;
        }
    }
    if complete {
        haplotypes.mark_complete();
    }
    Ok(source_samples)
}

/// Build a Beagle Stage2 composite state panel for one target haplotype.
/// Forward and backward IBS proposals are consumed in step order and the
/// returned marker-major donor map preserves each state's source haplotype
/// for the Stage2 rare-allele genotype projection.
pub(crate) fn build_low_frequency_phase_mosaic(
    window: &PhasedWindow,
    pbwt_index: &LowFrequencyPhasePbwtIndex,
    target_sample: usize,
    target_haplotype_bit: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
) -> Result<(CompositePanel, Vec<Vec<DonorHapSegment>>), String> {
    let view = PhaseHaplotypeView::target(window);
    build_low_frequency_phase_mosaic_view(
        &view,
        pbwt_index,
        target_sample,
        target_haplotype_bit,
        max_states,
        min_lifetime_steps,
        seed,
    )
}

pub(crate) fn build_low_frequency_phase_mosaic_view(
    view: &PhaseHaplotypeView<'_>,
    pbwt_index: &LowFrequencyPhasePbwtIndex,
    target_sample: usize,
    target_haplotype_bit: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
) -> Result<(CompositePanel, Vec<Vec<DonorHapSegment>>), String> {
    build_low_frequency_phase_mosaic_view_selected(
        view,
        None,
        pbwt_index,
        target_sample,
        target_haplotype_bit,
        max_states,
        min_lifetime_steps,
        seed,
        None,
    )
}

pub(crate) fn build_low_frequency_phase_mosaic_view_on_markers(
    view: &PhaseHaplotypeView<'_>,
    pbwt_index: &LowFrequencyPhasePbwtIndex,
    target_sample: usize,
    target_haplotype_bit: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
    marker_indices: &[usize],
) -> Result<(CompositePanel, Vec<Vec<DonorHapSegment>>), String> {
    if marker_indices.is_empty()
        || marker_indices
            .iter()
            .any(|&marker| marker >= view.n_markers())
        || marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("low-frequency phase projection markers are invalid".to_owned());
    }
    build_low_frequency_phase_mosaic_view_selected(
        view,
        None,
        pbwt_index,
        target_sample,
        target_haplotype_bit,
        max_states,
        min_lifetime_steps,
        seed,
        Some(marker_indices),
    )
}

pub(crate) fn build_low_frequency_phase_mosaic_view_on_markers_with_stage1_view(
    view: &PhaseHaplotypeView<'_>,
    packed_stage1_view: &PhaseHaplotypeView<'_>,
    pbwt_index: &LowFrequencyPhasePbwtIndex,
    target_sample: usize,
    target_haplotype_bit: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
    marker_indices: &[usize],
) -> Result<(CompositePanel, Vec<Vec<DonorHapSegment>>), String> {
    if marker_indices.is_empty()
        || marker_indices
            .iter()
            .any(|&marker| marker >= view.n_markers())
        || marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("low-frequency phase projection markers are invalid".to_owned());
    }
    build_low_frequency_phase_mosaic_view_selected(
        view,
        Some(packed_stage1_view),
        pbwt_index,
        target_sample,
        target_haplotype_bit,
        max_states,
        min_lifetime_steps,
        seed,
        Some(marker_indices),
    )
}

fn fill_low_frequency_phase_mosaic_haplotypes(
    view: &PhaseHaplotypeView<'_>,
    packed_stage1_view: Option<&PhaseHaplotypeView<'_>>,
    state_segments: &[Vec<DonorHapSegment>],
    marker_indices: &[usize],
    haplotypes: &mut PackedHaplotypes,
) -> Result<Vec<usize>, String> {
    if state_segments.is_empty()
        || marker_indices.is_empty()
        || haplotypes.n_haplotypes != state_segments.len()
        || haplotypes.n_markers != marker_indices.len()
    {
        return Err("low-frequency mosaic fill dimensions are inconsistent".to_owned());
    }
    if let Some(packed_view) = packed_stage1_view {
        if !haplotypes.is_state_major()
            || haplotypes.bits_per_allele != 1
            || packed_view.n_markers() != marker_indices.len()
            || packed_view.target_samples() != view.target_samples()
            || packed_view.n_samples() != view.n_samples()
            || !stage2_panel_packed_view_eligible(packed_view, marker_indices.len())
        {
            return Err("low-frequency packed projection inputs are incomplete".to_owned());
        }
    }
    let packed_fill = packed_stage1_view.is_some();
    if packed_fill {
        haplotypes.missing_bits.fill(0);
    }
    let mut source_samples = Vec::with_capacity(state_segments.len());
    for (state, segments) in state_segments.iter().enumerate() {
        let Some(first_segment) = segments.first() else {
            return Err("low-frequency donor state has no covered markers".to_owned());
        };
        source_samples.push(first_segment.donor_haplotype / 2);
        let mut covered = 0usize;
        for segment in segments {
            if segment.start_marker != covered
                || segment.start_marker >= segment.end_marker
                || segment.end_marker > view.n_markers()
                || segment.donor_haplotype >= view.n_samples().saturating_mul(2)
            {
                return Err("low-frequency donor segments do not form a partition".to_owned());
            }
            let first = marker_indices.partition_point(|&marker| marker < segment.start_marker);
            let last = marker_indices.partition_point(|&marker| marker < segment.end_marker);
            if let Some(packed_view) = packed_stage1_view {
                packed_view.copy_biallelic_segment_from(
                    haplotypes,
                    state,
                    first,
                    last,
                    segment.donor_haplotype,
                    false,
                )?;
            } else if haplotypes.is_state_major() && haplotypes.bits_per_allele == 1 {
                for (local, &marker) in marker_indices[first..last].iter().enumerate() {
                    unsafe {
                        haplotypes.set_state_major_biallelic_unchecked(
                            first + local,
                            state,
                            view.allele(
                                marker,
                                segment.donor_haplotype / 2,
                                segment.donor_haplotype & 1,
                            ),
                        );
                    }
                }
            } else {
                for (local, &marker) in marker_indices[first..last].iter().enumerate() {
                    haplotypes.set_allele(
                        first + local,
                        state,
                        view.allele(
                            marker,
                            segment.donor_haplotype / 2,
                            segment.donor_haplotype & 1,
                        ),
                    )?;
                }
            }
            covered = segment.end_marker;
        }
        if covered != view.n_markers() {
            return Err("low-frequency donor state does not cover the window".to_owned());
        }
    }
    Ok(source_samples)
}

#[allow(clippy::too_many_arguments)]
fn build_low_frequency_phase_mosaic_view_selected(
    view: &PhaseHaplotypeView<'_>,
    packed_stage1_view: Option<&PhaseHaplotypeView<'_>>,
    pbwt_index: &LowFrequencyPhasePbwtIndex,
    target_sample: usize,
    target_haplotype_bit: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    seed: u64,
    selected_markers: Option<&[usize]>,
) -> Result<(CompositePanel, Vec<Vec<DonorHapSegment>>), String> {
    if view.n_markers() == 0
        || pbwt_index.step_ranges.is_empty()
        || target_haplotype_bit > 1
        || max_states == 0
    {
        return Err(
            "low-frequency phase mosaic requires a window and valid donor limit".to_owned(),
        );
    }
    let n_samples = view.n_samples();
    let n_haplotypes = n_samples.saturating_mul(2);
    let target_haplotype = target_sample
        .checked_mul(2)
        .and_then(|value| value.checked_add(target_haplotype_bit))
        .ok_or_else(|| "low-frequency target haplotype index overflow".to_owned())?;
    if n_samples == 0
        || target_sample >= view.target_samples()
        || pbwt_index.n_haplotypes != n_haplotypes
    {
        return Err("low-frequency phase mosaic dimensions differ from the target".to_owned());
    }
    let step_ranges = &pbwt_index.step_ranges;
    if step_ranges[0].0 != 0
        || step_ranges.last().map(|range| range.1) != Some(view.n_markers())
        || step_ranges
            .iter()
            .any(|&(start, end)| start >= end || end > view.n_markers())
        || step_ranges.windows(2).any(|pair| pair[0].1 != pair[1].0)
    {
        return Err("low-frequency PBWT steps do not partition the phase window".to_owned());
    }
    let proposals = (0..step_ranges.len())
        .map(|step| {
            [
                pbwt_index.donor_haplotype(
                    super::pbwt::PbwtDirection::Forward,
                    step,
                    target_haplotype,
                ),
                pbwt_index.donor_haplotype(
                    super::pbwt::PbwtDirection::Reverse,
                    step,
                    target_haplotype,
                ),
            ]
        })
        .collect::<Vec<_>>();
    let state_segments = select_donor_segments(
        step_ranges,
        &proposals,
        n_samples,
        target_sample,
        max_states.min(n_haplotypes.saturating_sub(2)),
        min_lifetime_steps,
    )?;
    if state_segments.is_empty() {
        return build_random_low_frequency_phase_panel_view(
            view,
            target_sample,
            max_states,
            target_haplotype,
            seed,
            selected_markers,
        );
    }

    let all_marker_indices;
    let marker_indices = if let Some(markers) = selected_markers {
        markers
    } else {
        all_marker_indices = (0..view.n_markers()).collect::<Vec<_>>();
        &all_marker_indices
    };
    let max_allele = marker_indices
        .iter()
        .copied()
        .map(|marker| view.marker(marker).allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let state_count = state_segments.len();
    let marker_count = marker_indices.len();
    let mut haplotypes = if view.target_packed.is_some() {
        PackedHaplotypes::empty_state_major(state_count, marker_count, allele_width(max_allele))?
    } else {
        PackedHaplotypes::empty(state_count, marker_count, allele_width(max_allele))?
    };
    let selected_packed_view = packed_stage1_view.filter(|packed_view| {
        stage2_panel_fast_path_enabled()
            && stage2_panel_packed_view_eligible(packed_view, marker_indices.len())
    });
    let profile_started = stage2_panel_profiling_enabled().then(Instant::now);
    let source_samples = fill_low_frequency_phase_mosaic_haplotypes(
        view,
        selected_packed_view,
        &state_segments,
        marker_indices,
        &mut haplotypes,
    )?;
    if let Some(started) = profile_started {
        let elapsed = started.elapsed().as_nanos() as u64;
        if selected_packed_view.is_some() {
            STAGE2_PANEL_PACKED_NANOS.fetch_add(elapsed, Ordering::Relaxed);
        } else {
            STAGE2_PANEL_SCALAR_NANOS.fetch_add(elapsed, Ordering::Relaxed);
        }
    }
    if selected_packed_view.is_some() {
        STAGE2_PANEL_PACKED_CALLS.fetch_add(1, Ordering::Relaxed);
    } else {
        STAGE2_PANEL_SCALAR_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    Ok((
        CompositePanel {
            haplotypes,
            source_samples,
            marker_indices: (0..marker_count).collect(),
            allele_counts: marker_indices
                .iter()
                .copied()
                .map(|marker| view.marker(marker).allele_count())
                .collect(),
        },
        state_segments,
    ))
}

fn build_random_low_frequency_phase_panel_view(
    view: &PhaseHaplotypeView<'_>,
    target_sample: usize,
    max_states: usize,
    target_haplotype: usize,
    seed: u64,
    selected_markers: Option<&[usize]>,
) -> Result<(CompositePanel, Vec<Vec<DonorHapSegment>>), String> {
    let n_samples = view.n_samples();
    let n_haplotypes = n_samples.saturating_mul(2);
    if target_sample >= view.target_samples() || max_states == 0 || n_haplotypes <= 2 {
        return Err("cannot construct a random low-frequency phase panel".to_owned());
    }
    let state_count = max_states.min(n_haplotypes - 2);
    let all_marker_indices;
    let marker_indices = if let Some(markers) = selected_markers {
        markers
    } else {
        all_marker_indices = (0..view.n_markers()).collect::<Vec<_>>();
        &all_marker_indices
    };
    let marker_count = marker_indices.len();
    let max_allele = marker_indices
        .iter()
        .copied()
        .map(|marker| view.marker(marker).allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let mut haplotypes = if view.target_packed.is_some() {
        PackedHaplotypes::empty_state_major(state_count, marker_count, allele_width(max_allele))?
    } else {
        PackedHaplotypes::empty(state_count, marker_count, allele_width(max_allele))?
    };
    let mut source_samples = Vec::with_capacity(state_count);
    let mut state_segments = Vec::with_capacity(state_count);
    let mut random = super::pbwt::JavaRandom::new(seed.wrapping_add(target_haplotype as u64));
    for state in 0..state_count {
        let donor = loop {
            let candidate = random.next_int(n_haplotypes);
            if candidate / 2 != target_sample {
                break candidate;
            }
        };
        let donor_sample = donor / 2;
        let donor_bit = donor & 1;
        source_samples.push(donor_sample);
        state_segments.push(vec![DonorHapSegment {
            start_marker: 0,
            end_marker: view.n_markers(),
            donor_haplotype: donor,
        }]);
        if haplotypes.is_state_major() && haplotypes.bits_per_allele == 1 {
            for (local, &marker) in marker_indices.iter().enumerate() {
                unsafe {
                    haplotypes.set_state_major_biallelic_unchecked(
                        local,
                        state,
                        view.allele(marker, donor_sample, donor_bit),
                    );
                }
            }
        } else {
            for (local, &marker) in marker_indices.iter().enumerate() {
                haplotypes.set_allele(
                    local,
                    state,
                    view.allele(marker, donor_sample, donor_bit),
                )?;
            }
        }
    }
    Ok((
        CompositePanel {
            haplotypes,
            source_samples,
            marker_indices: (0..marker_count).collect(),
            allele_counts: marker_indices
                .iter()
                .copied()
                .map(|marker| view.marker(marker).allele_count())
                .collect(),
        },
        state_segments,
    ))
}

pub(crate) fn donor_haplotype_at(segments: &[DonorHapSegment], marker: usize) -> Option<usize> {
    let segment = segments.get(segments.partition_point(|segment| segment.end_marker <= marker))?;
    (segment.start_marker <= marker && marker < segment.end_marker)
        .then_some(segment.donor_haplotype)
}

fn build_random_phase_panel(
    window: &PhasedWindow,
    target_sample: usize,
    max_states: usize,
    seed: u64,
) -> Result<CompositePanel, String> {
    let n_samples = window.rows.first().map_or(0, |row| row.calls.len());
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "phase donor haplotype count overflow".to_owned())?;
    if target_sample >= n_samples || max_states == 0 || n_haplotypes <= 2 {
        return Err("cannot construct a random phase panel for this target".to_owned());
    }
    let mut donors = (0..n_haplotypes)
        .filter(|haplotype| haplotype / 2 != target_sample)
        .collect::<Vec<_>>();
    let state_count = max_states.min(donors.len());
    let max_allele = window
        .rows
        .iter()
        .map(|row| row.meta.allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let mut haplotypes = PackedHaplotypes::empty_state_major(
        state_count,
        window.rows.len(),
        allele_width(max_allele),
    )?;
    let mut source_samples = Vec::with_capacity(state_count);
    let mut rng = seed;
    for state in 0..state_count {
        let selected =
            state + (super::pbwt::next_random(&mut rng) as usize % (donors.len() - state));
        donors.swap(state, selected);
        let donor = donors[state];
        let donor_sample = donor / 2;
        let donor_haplotype = donor & 1;
        source_samples.push(donor_sample);
        for marker in 0..window.rows.len() {
            haplotypes.set_allele(
                marker,
                state,
                window.rows[marker].calls.get(donor_sample).unwrap().alleles[donor_haplotype],
            )?;
        }
    }
    Ok(CompositePanel {
        haplotypes,
        source_samples,
        marker_indices: (0..window.rows.len()).collect(),
        allele_counts: window
            .rows
            .iter()
            .map(|row| row.meta.allele_count())
            .collect(),
    })
}

fn build_random_phase_panel_view(
    view: &PhaseHaplotypeView<'_>,
    target_sample: usize,
    max_states: usize,
    seed: u64,
) -> Result<CompositePanel, String> {
    let n_samples = view.n_samples();
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "phase donor haplotype count overflow".to_owned())?;
    if target_sample >= view.target_samples() || max_states == 0 || n_haplotypes <= 2 {
        return Err("cannot construct a random phase panel for this target".to_owned());
    }
    let mut donors = (0..n_haplotypes)
        .filter(|haplotype| haplotype / 2 != target_sample)
        .collect::<Vec<_>>();
    let state_count = max_states.min(donors.len());
    let max_allele = (0..view.n_markers())
        .map(|marker| view.marker(marker).allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let mut haplotypes = PackedHaplotypes::empty_state_major(
        state_count,
        view.n_markers(),
        allele_width(max_allele),
    )?;
    let mut source_samples = Vec::with_capacity(state_count);
    let mut rng = seed;
    for state in 0..state_count {
        let selected =
            state + (super::pbwt::next_random(&mut rng) as usize % (donors.len() - state));
        donors.swap(state, selected);
        let donor = donors[state];
        let donor_sample = donor / 2;
        let donor_haplotype = donor & 1;
        source_samples.push(donor_sample);
        for marker in 0..view.n_markers() {
            haplotypes.set_allele(
                marker,
                state,
                view.allele(marker, donor_sample, donor_haplotype),
            )?;
        }
    }
    Ok(CompositePanel {
        haplotypes,
        source_samples,
        marker_indices: (0..view.n_markers()).collect(),
        allele_counts: (0..view.n_markers())
            .map(|marker| view.marker(marker).allele_count())
            .collect(),
    })
}

pub(crate) fn build_reference(
    window: &PhasedWindow,
    exclude_sample: Option<usize>,
    max_states: usize,
) -> Result<CompositePanel, String> {
    build_reference_with_index(window, exclude_sample, max_states, None)
}

pub(crate) fn reference_major_alleles(window: &PhasedWindow) -> Result<Vec<Option<u16>>, String> {
    let Some(first) = window.rows.first() else {
        return Err("cannot estimate reference alleles from an empty window".to_owned());
    };
    let n_samples = first.calls.len();
    if n_samples == 0 || window.rows.iter().any(|row| row.calls.len() != n_samples) {
        return Err("reference window has an invalid or inconsistent sample count".to_owned());
    }
    Ok(window
        .rows
        .iter()
        .map(|row| {
            let mut counts = vec![0usize; row.meta.allele_count()];
            for allele in row
                .calls
                .iter()
                .flat_map(|call| call.alleles.into_iter().flatten())
            {
                if let Some(count) = counts.get_mut(usize::from(allele)) {
                    *count += 1;
                }
            }
            counts
                .iter()
                .enumerate()
                .max_by_key(|(allele, count)| (**count, std::cmp::Reverse(*allele)))
                .and_then(|(allele, count)| (*count > 0).then_some(allele as u16))
        })
        .collect())
}

/// Dense, complete biallelic references are used for millions of repeated
/// donor projections. Keep their haplotypes in a marker-major bit plane so
/// each target mosaic can gather bits without decoding every packed diploid
/// nibble again.
#[derive(Debug, Clone)]
pub(crate) struct PackedBiallelicReference {
    n_haplotypes: usize,
    n_markers: usize,
    words_per_marker: usize,
    allele_words: Vec<u64>,
    missing_markers: Vec<bool>,
}

#[derive(Debug, Clone)]
struct PackedTargetMosaicSegment {
    start: usize,
    end: usize,
    donors: Vec<usize>,
}

/// A target-specific donor mosaic that stores HMM alleles only at target
/// markers. Dense reference-only markers are projected from the segment's
/// donor list on demand, avoiding a full `imp_states × reference_markers`
/// matrix per target haplotype.
pub(crate) struct PackedTargetMosaic {
    pub(crate) panel: CompositePanel,
    segments: Vec<PackedTargetMosaicSegment>,
}

impl PackedTargetMosaic {
    pub(crate) fn donors_at(&self, marker: usize) -> Option<&[usize]> {
        self.donor_run_at(marker).map(|(_, donors)| donors)
    }

    pub(crate) fn donor_run_at(&self, marker: usize) -> Option<(usize, &[usize])> {
        let segment_index = self
            .segments
            .partition_point(|segment| segment.end <= marker);
        self.segments
            .get(segment_index)
            .filter(|segment| segment.start <= marker && marker < segment.end)
            .map(|segment| (segment.end, segment.donors.as_slice()))
    }

    pub(crate) fn pack_marker(
        &self,
        reference: &PackedBiallelicReference,
        marker: usize,
        output: &mut PackedHaplotypes,
    ) -> Result<(), String> {
        let Some(donors) = self.donors_at(marker) else {
            return Err("target mosaic marker is outside its PBWT segments".to_owned());
        };
        reference.set_mosaic_marker_at(marker, 0, donors, output)
    }

    pub(crate) fn pack_marker_words(
        &self,
        reference: &PackedBiallelicReference,
        marker: usize,
        allele_words: &mut [u64],
        missing_words: &mut [u64],
    ) -> Result<(), String> {
        let Some(donors) = self.donors_at(marker) else {
            return Err("target mosaic marker is outside its PBWT segments".to_owned());
        };
        reference.mosaic_marker_words_at(marker, donors, allele_words, missing_words)
    }

    pub(crate) fn pack_marker_allele_words(
        &self,
        reference: &PackedBiallelicReference,
        marker: usize,
        allele_words: &mut [u64],
    ) -> Result<bool, String> {
        let Some(donors) = self.donors_at(marker) else {
            return Err("target mosaic marker is outside its PBWT segments".to_owned());
        };
        reference.mosaic_marker_allele_words_at(marker, donors, allele_words)
    }

    #[cfg(test)]
    pub(crate) fn allele_probability(
        &self,
        reference: &PackedBiallelicReference,
        marker: usize,
        state_probabilities: &[f32],
        allele: u16,
    ) -> f32 {
        debug_assert_eq!(
            state_probabilities.len(),
            self.panel.haplotypes.n_haplotypes
        );
        let Some(donors) = self.donors_at(marker) else {
            return 0.5;
        };
        let mut observed_alleles = 0usize;
        let mut observed_matches = 0usize;
        let mut observed_probability = 0.0f32;
        let mut missing_probability = 0.0f32;
        for (state, &probability) in state_probabilities.iter().enumerate() {
            match reference.allele_unchecked(marker, donors[state]) {
                Some(state_allele) => {
                    observed_alleles += 1;
                    let matches = state_allele == allele;
                    observed_matches += usize::from(matches);
                    if matches {
                        observed_probability += probability;
                    }
                }
                None => missing_probability += probability,
            }
        }
        let fallback_frequency = if observed_alleles == 0 {
            0.5
        } else {
            observed_matches as f32 / observed_alleles as f32
        };
        (observed_probability + missing_probability * fallback_frequency).clamp(0.0, 1.0)
    }

    #[cfg(test)]
    pub(crate) fn interpolated_allele_probability(
        &self,
        reference: &PackedBiallelicReference,
        marker: usize,
        left_probabilities: &[f32],
        right_probabilities: &[f32],
        alpha: f32,
        allele: u16,
    ) -> f32 {
        debug_assert_eq!(left_probabilities.len(), self.panel.haplotypes.n_haplotypes);
        debug_assert_eq!(
            right_probabilities.len(),
            self.panel.haplotypes.n_haplotypes
        );
        let Some(donors) = self.donors_at(marker) else {
            return 0.5;
        };
        let mut observed_alleles = 0usize;
        let mut observed_matches = 0usize;
        let mut left_observed_probability = 0.0f32;
        let mut right_observed_probability = 0.0f32;
        let mut left_missing_probability = 0.0f32;
        let mut right_missing_probability = 0.0f32;
        for (state, (&left_probability, &right_probability)) in left_probabilities
            .iter()
            .zip(right_probabilities)
            .enumerate()
        {
            match reference.allele_unchecked(marker, donors[state]) {
                Some(state_allele) => {
                    observed_alleles += 1;
                    if state_allele == allele {
                        observed_matches += 1;
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
            observed_matches as f32 / observed_alleles as f32
        };
        let left_probability =
            left_observed_probability + left_missing_probability * fallback_frequency;
        let right_probability =
            right_observed_probability + right_missing_probability * fallback_frequency;
        (left_probability * (1.0 - alpha) + right_probability * alpha).clamp(0.0, 1.0)
    }
}

#[inline]
fn pack_donor_alleles(reference_words: &[u64], donors: &[usize], allele_scratch: &mut [u64]) {
    debug_assert!(!donors.is_empty());
    debug_assert_eq!(allele_scratch.len(), donors.len().div_ceil(64));
    debug_assert!(donors
        .iter()
        .all(|&donor| donor < reference_words.len() * 64));

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // Donor IDs are checked when the immutable mosaic segments are built,
        // so the AVX2 gather indices remain within `reference_words`.
        unsafe { pack_donor_alleles_avx2(reference_words, donors, allele_scratch) };
        return;
    }

    pack_donor_alleles_scalar(reference_words, donors, allele_scratch);
}

#[inline]
fn pack_donor_alleles_scalar(
    reference_words: &[u64],
    donors: &[usize],
    allele_scratch: &mut [u64],
) {
    for (word_index, donor_chunk) in donors.chunks(64).enumerate() {
        let mut packed_word = 0u64;
        for (bit, &donor) in donor_chunk.iter().enumerate() {
            let allele = (reference_words[donor >> 6] >> (donor & 63)) & 1;
            packed_word |= allele << bit;
        }
        allele_scratch[word_index] = packed_word;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_donor_alleles_avx2(
    reference_words: &[u64],
    donors: &[usize],
    allele_scratch: &mut [u64],
) {
    use std::arch::x86_64::*;

    debug_assert!(!donors.is_empty());
    debug_assert_eq!(allele_scratch.len(), donors.len().div_ceil(64));
    debug_assert!(donors
        .iter()
        .all(|&donor| donor < reference_words.len() * 64));

    unsafe {
        let shift_mask = _mm256_set1_epi64x(63);
        let one = _mm256_set1_epi64x(1);
        for (word_index, donor_chunk) in donors.chunks(64).enumerate() {
            let vector_end = donor_chunk.len() & !3;
            let mut packed_word = 0u64;
            for state in (0..vector_end).step_by(4) {
                let donor_indices =
                    _mm256_loadu_si256(donor_chunk.as_ptr().add(state).cast::<__m256i>());
                let reference_word_indices = _mm256_srli_epi64(donor_indices, 6);
                let bit_offsets = _mm256_and_si256(donor_indices, shift_mask);
                let reference_values = _mm256_i64gather_epi64::<8>(
                    reference_words.as_ptr().cast::<i64>(),
                    reference_word_indices,
                );
                let allele_values =
                    _mm256_and_si256(_mm256_srlv_epi64(reference_values, bit_offsets), one);
                let matching_lanes = _mm256_cmpeq_epi64(allele_values, one);
                let lane_bits = _mm256_movemask_pd(_mm256_castsi256_pd(matching_lanes)) as u64;
                packed_word |= lane_bits << state;
            }
            for state in vector_end..donor_chunk.len() {
                let donor = donor_chunk[state];
                let allele = (reference_words[donor >> 6] >> (donor & 63)) & 1;
                packed_word |= allele << state;
            }
            allele_scratch[word_index] = packed_word;
        }
    }
}

impl PackedBiallelicReference {
    /// Pack a merged target/reference marker order directly from the existing
    /// reference window. `None` entries are target-only markers and are kept
    /// missing in the donor panel.
    pub(crate) fn from_reference_rows(
        reference_rows: &[super::types::PhasedRow],
        reference_row_indices: &[Option<usize>],
        reference_swaps: &[bool],
        n_samples: usize,
    ) -> Result<Self, String> {
        if reference_rows.is_empty()
            || reference_row_indices.is_empty()
            || reference_row_indices.len() != reference_swaps.len()
            || n_samples == 0
        {
            return Err("packed reference dimensions are invalid".to_owned());
        }
        if reference_rows
            .iter()
            .any(|row| row.meta.allele_count() != 2 || row.calls.len() != n_samples)
        {
            return Err(
                "packed reference rows must be biallelic with consistent samples".to_owned(),
            );
        }
        let n_haplotypes = n_samples
            .checked_mul(2)
            .ok_or_else(|| "packed reference haplotype count overflow".to_owned())?;
        let words_per_marker = n_haplotypes.div_ceil(64);
        let word_count = words_per_marker
            .checked_mul(reference_row_indices.len())
            .ok_or_else(|| "packed reference dimensions overflow".to_owned())?;
        let mut allele_words = vec![0u64; word_count];
        let mut missing_markers = vec![false; reference_row_indices.len()];
        for (marker, reference_index) in reference_row_indices.iter().copied().enumerate() {
            let Some(reference_index) = reference_index else {
                missing_markers[marker] = true;
                continue;
            };
            let row = reference_rows.get(reference_index).ok_or_else(|| {
                format!("reference marker index {reference_index} exceeds the reference window")
            })?;
            let row_start = marker * words_per_marker;
            let marker_words = &mut allele_words[row_start..row_start + words_per_marker];
            let packed_complete_biallelic =
                marker_words
                    .iter_mut()
                    .enumerate()
                    .all(|(word_index, word)| {
                        let Some(packed_word) = row
                            .calls
                            .biallelic_haplotype_word(word_index, reference_swaps[marker])
                        else {
                            return false;
                        };
                        *word = packed_word;
                        true
                    });
            if !packed_complete_biallelic {
                // Keep the original detailed error checks for rare missing or
                // unsupported calls while the common complete biallelic path
                // expands four haplotype alleles per packed genotype byte.
                marker_words.fill(0);
                for haplotype in 0..n_haplotypes {
                    let Some(allele) = row.calls.get_allele_unchecked(haplotype) else {
                        return Err(format!(
                            "reference GT must be complete at {}:{}",
                            row.meta.chrom, row.meta.pos
                        ));
                    };
                    if allele > 1 {
                        return Err(format!(
                            "reference GT must be biallelic at {}:{}",
                            row.meta.chrom, row.meta.pos
                        ));
                    }
                    let allele = allele ^ u16::from(reference_swaps[marker]);
                    if allele == 1 {
                        marker_words[haplotype >> 6] |= 1u64 << (haplotype & 63);
                    }
                }
            }
        }
        Ok(Self {
            n_haplotypes,
            n_markers: reference_row_indices.len(),
            words_per_marker,
            allele_words,
            missing_markers,
        })
    }

    #[inline]
    pub(crate) fn n_haplotypes(&self) -> usize {
        self.n_haplotypes
    }

    #[inline]
    pub(crate) fn n_markers(&self) -> usize {
        self.n_markers
    }

    #[inline]
    pub(crate) fn allele_unchecked(&self, marker: usize, haplotype: usize) -> Option<u16> {
        debug_assert!(marker < self.n_markers);
        debug_assert!(haplotype < self.n_haplotypes);
        if self.missing_markers[marker] {
            return None;
        }
        let word = self.allele_words[marker * self.words_per_marker + (haplotype >> 6)];
        Some(((word >> (haplotype & 63)) & 1) as u16)
    }

    /// Return the marker's complete allele bit plane for fused donor
    /// projection. Missing or target-only markers use the general path.
    pub(crate) fn complete_marker_words(&self, marker: usize) -> Option<&[u64]> {
        if marker >= self.n_markers || self.missing_markers[marker] {
            return None;
        }
        let row_start = marker * self.words_per_marker;
        Some(&self.allele_words[row_start..row_start + self.words_per_marker])
    }

    /// Count alternate alleles directly from the marker-major bitplane.
    /// The same pass supplies both the major allele used by PBWT queries and
    /// the reference AF used when projecting non-reference observations.
    pub(crate) fn allele_summaries(&self) -> (Vec<Option<u16>>, Vec<Option<f32>>) {
        let mut major_alleles = Vec::with_capacity(self.n_markers);
        let mut alt_frequencies = Vec::with_capacity(self.n_markers);
        for marker in 0..self.n_markers {
            if self.missing_markers[marker] {
                major_alleles.push(None);
                alt_frequencies.push(None);
                continue;
            }
            let row_start = marker * self.words_per_marker;
            let alternate_count = self.allele_words[row_start..row_start + self.words_per_marker]
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum::<usize>();
            major_alleles.push(Some(u16::from(alternate_count > self.n_haplotypes / 2)));
            alt_frequencies.push(Some(alternate_count as f32 / self.n_haplotypes as f32));
        }
        (major_alleles, alt_frequencies)
    }

    pub(crate) fn major_alleles(&self) -> Vec<Option<u16>> {
        self.allele_summaries().0
    }

    pub(crate) fn from_window(
        window: &PhasedWindow,
        reference_present: &[bool],
    ) -> Result<Option<Self>, String> {
        if window.rows.is_empty() || reference_present.len() != window.rows.len() {
            return Err("packed biallelic reference dimensions are invalid".to_owned());
        }
        if window.rows.iter().any(|row| row.meta.allele_count() != 2) {
            return Ok(None);
        }
        let n_samples = window.rows[0].calls.len();
        let Some(n_haplotypes) = n_samples.checked_mul(2) else {
            return Err("packed reference haplotype count overflow".to_owned());
        };
        if n_samples == 0 || window.rows.iter().any(|row| row.calls.len() != n_samples) {
            return Err("packed reference sample dimensions are invalid".to_owned());
        }
        let words_per_marker = n_haplotypes.div_ceil(64);
        let word_count = words_per_marker
            .checked_mul(window.rows.len())
            .ok_or_else(|| "packed reference dimensions overflow".to_owned())?;
        let mut allele_words = vec![0u64; word_count];
        let mut missing_markers = vec![false; window.rows.len()];
        for (marker, row) in window.rows.iter().enumerate() {
            if !reference_present[marker] {
                missing_markers[marker] = true;
                continue;
            }
            let row_start = marker * words_per_marker;
            let marker_words = &mut allele_words[row_start..row_start + words_per_marker];
            let complete_biallelic =
                marker_words
                    .iter_mut()
                    .enumerate()
                    .all(|(word_index, word)| {
                        let Some(packed_word) =
                            row.calls.biallelic_haplotype_word(word_index, false)
                        else {
                            return false;
                        };
                        *word = packed_word;
                        true
                    });
            if !complete_biallelic {
                marker_words.fill(0);
                for haplotype in 0..n_haplotypes {
                    let Some(allele) = row.calls.get_allele_unchecked(haplotype) else {
                        // Partial or fully missing reference rows retain the
                        // general diploid path, which preserves per-haplotype
                        // missingness exactly.
                        return Ok(None);
                    };
                    if allele > 1 {
                        return Ok(None);
                    }
                    if allele == 1 {
                        marker_words[haplotype >> 6] |= 1u64 << (haplotype & 63);
                    }
                }
            }
        }
        Ok(Some(Self {
            n_haplotypes,
            n_markers: window.rows.len(),
            words_per_marker,
            allele_words,
            missing_markers,
        }))
    }

    fn set_mosaic_marker(
        &self,
        marker: usize,
        donors: &[usize],
        output: &mut PackedHaplotypes,
    ) -> Result<(), String> {
        self.set_mosaic_marker_at(marker, marker, donors, output)
    }

    fn set_mosaic_marker_at(
        &self,
        reference_marker: usize,
        output_marker: usize,
        donors: &[usize],
        output: &mut PackedHaplotypes,
    ) -> Result<(), String> {
        if reference_marker >= self.n_markers
            || output_marker >= output.n_markers
            || donors.len() != output.n_haplotypes
        {
            return Err("packed reference mosaic dimensions are invalid".to_owned());
        }
        if self.missing_markers[reference_marker] {
            return output.set_biallelic_marker_with(output_marker, |_| None);
        }
        let row_start = reference_marker * self.words_per_marker;
        let allele_words = &self.allele_words[row_start..row_start + self.words_per_marker];
        output.set_biallelic_marker_with(output_marker, |state| {
            let donor = donors[state];
            debug_assert!(donor < self.n_haplotypes);
            let allele = (allele_words[donor >> 6] >> (donor & 63)) & 1;
            Some(allele as u16)
        })
    }

    fn mosaic_marker_words_at(
        &self,
        reference_marker: usize,
        donors: &[usize],
        allele_scratch: &mut [u64],
        missing_scratch: &mut [u64],
    ) -> Result<(), String> {
        let required_words = donors.len().div_ceil(64);
        if reference_marker >= self.n_markers
            || donors.len() == 0
            || allele_scratch.len() != required_words
            || missing_scratch.len() != required_words
        {
            return Err("packed reference mosaic dimensions are invalid".to_owned());
        }
        missing_scratch.fill(0);
        if self.missing_markers[reference_marker] {
            allele_scratch.fill(0);
            for (word_index, word) in missing_scratch.iter_mut().enumerate() {
                let state_count = (donors.len() - word_index * 64).min(64);
                *word = if state_count == 64 {
                    u64::MAX
                } else {
                    (1u64 << state_count) - 1
                };
            }
            return Ok(());
        }

        let row_start = reference_marker * self.words_per_marker;
        let reference_words = &self.allele_words[row_start..row_start + self.words_per_marker];
        pack_donor_alleles(reference_words, donors, allele_scratch);
        Ok(())
    }

    fn mosaic_marker_allele_words_at(
        &self,
        reference_marker: usize,
        donors: &[usize],
        allele_scratch: &mut [u64],
    ) -> Result<bool, String> {
        let required_words = donors.len().div_ceil(64);
        if reference_marker >= self.n_markers
            || donors.is_empty()
            || allele_scratch.len() != required_words
        {
            return Err("packed reference mosaic dimensions are invalid".to_owned());
        }
        if self.missing_markers[reference_marker] {
            return Ok(false);
        }

        let row_start = reference_marker * self.words_per_marker;
        let reference_words = &self.allele_words[row_start..row_start + self.words_per_marker];
        pack_donor_alleles(reference_words, donors, allele_scratch);
        Ok(true)
    }

    fn set_mosaic_marker_at_with_scratch(
        &self,
        reference_marker: usize,
        output_marker: usize,
        donors: &[usize],
        output: &mut PackedHaplotypes,
        word_scratch: &mut [u64],
    ) -> Result<(), String> {
        if reference_marker >= self.n_markers
            || output_marker >= output.n_markers
            || donors.len() != output.n_haplotypes
            || word_scratch.len() != output.n_haplotypes.div_ceil(64)
        {
            return Err("packed reference mosaic dimensions are invalid".to_owned());
        }
        if self.missing_markers[reference_marker] {
            return output.set_biallelic_marker_with(output_marker, |_| None);
        }

        word_scratch.fill(0);
        let row_start = reference_marker * self.words_per_marker;
        let allele_words = &self.allele_words[row_start..row_start + self.words_per_marker];
        for (state, &donor) in donors.iter().enumerate() {
            debug_assert!(donor < self.n_haplotypes);
            let allele = (allele_words[donor >> 6] >> (donor & 63)) & 1;
            word_scratch[state >> 6] |= allele << (state & 63);
        }
        output.set_biallelic_marker_words(output_marker, word_scratch)
    }
}

/// Reorder a new segment's donor set so donors retained from the preceding
/// segment keep their HMM state indices. New donors fill the remaining slots
/// in PBWT rank order.
fn align_mosaic_donor_states(previous: &[usize], current: Vec<usize>) -> Vec<usize> {
    debug_assert_eq!(previous.len(), current.len());

    let mut previous_slots = previous
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, donor)| (donor, slot))
        .collect::<Vec<_>>();
    previous_slots.sort_unstable_by_key(|&(donor, _)| donor);

    let mut aligned = vec![usize::MAX; previous.len()];
    let mut new_donors = Vec::with_capacity(current.len());
    for donor in current {
        match previous_slots.binary_search_by_key(&donor, |&(previous_donor, _)| previous_donor) {
            Ok(index) => {
                let (_, slot) = previous_slots[index];
                aligned[slot] = donor;
            }
            Err(_) => new_donors.push(donor),
        }
    }

    let mut new_donors = new_donors.into_iter();
    for donor in &mut aligned {
        if *donor == usize::MAX {
            *donor = new_donors
                .next()
                .expect("current donor set has one donor per HMM state");
        }
    }
    debug_assert!(new_donors.next().is_none());
    aligned
}

/// Construct a compact target-specific mosaic. Only target-observed marker
/// rows are materialized for the HMM; donor lists retain the dense reference
/// projection for every marker in each PBWT segment.
pub(crate) fn build_pbwt_target_mosaic_hmm_packed(
    packed_reference: &PackedBiallelicReference,
    pbwt_index: &SegmentPbwtIndex,
    query_alleles: &[Option<u16>],
    hmm_marker_indices: &[usize],
    cm_positions: &[f64],
    segment_cm: f64,
    max_states: usize,
) -> Result<PackedTargetMosaic, String> {
    let n_markers = packed_reference.n_markers();
    let n_haplotypes = packed_reference.n_haplotypes();
    if n_markers == 0
        || query_alleles.len() != n_markers
        || hmm_marker_indices.is_empty()
        || hmm_marker_indices.iter().any(|&marker| marker >= n_markers)
        || hmm_marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
        || cm_positions.len() != n_markers
        || pbwt_index.segments.is_empty()
        || pbwt_index.n_haplotypes != n_haplotypes
        || !segment_cm.is_finite()
        || segment_cm <= 0.0
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
        || max_states == 0
    {
        return Err("PBWT target mosaic dimensions, map, or segment length are invalid".to_owned());
    }
    let state_count = max_states.min(n_haplotypes);
    let mut hmm_haplotypes = PackedHaplotypes::empty(state_count, hmm_marker_indices.len(), 1)?;
    let mut source_samples = vec![0usize; state_count];
    let mut segments: Vec<PackedTargetMosaicSegment> =
        Vec::with_capacity(pbwt_index.segments.len());
    let mut mosaic_word_scratch = vec![0u64; state_count.div_ceil(64)];
    let mut expected_start = 0usize;
    let mut next_hmm_marker = 0usize;
    for (segment_index, segment) in pbwt_index.segments.iter().enumerate() {
        if segment.start != expected_start
            || segment.end <= segment.start
            || segment.end > n_markers
        {
            return Err("PBWT segment index does not cover the reference window".to_owned());
        }
        let donors = super::pbwt::segment_neighbors_from_alleles(
            pbwt_index,
            segment_index,
            n_markers,
            query_alleles,
            state_count,
            |marker, haplotype| packed_reference.allele_unchecked(marker, haplotype),
        )?;
        if donors.len() != state_count || donors.iter().any(|&donor| donor >= n_haplotypes) {
            return Err("PBWT segment returned invalid reference haplotypes".to_owned());
        }
        let donors = match segments.last() {
            Some(previous) => align_mosaic_donor_states(&previous.donors, donors),
            None => donors,
        };
        if segment_index == 0 {
            for (state, &donor) in donors.iter().enumerate() {
                source_samples[state] = donor / 2;
            }
        }
        while let Some(&marker) = hmm_marker_indices.get(next_hmm_marker) {
            if marker < segment.start {
                return Err("HMM marker is outside the PBWT segment partition".to_owned());
            }
            if marker >= segment.end {
                break;
            }
            packed_reference.set_mosaic_marker_at_with_scratch(
                marker,
                next_hmm_marker,
                &donors,
                &mut hmm_haplotypes,
                &mut mosaic_word_scratch,
            )?;
            next_hmm_marker += 1;
        }
        segments.push(PackedTargetMosaicSegment {
            start: segment.start,
            end: segment.end,
            donors,
        });
        expected_start = segment.end;
    }
    if expected_start != n_markers || next_hmm_marker != hmm_marker_indices.len() {
        return Err("PBWT segments do not cover all target HMM markers".to_owned());
    }
    Ok(PackedTargetMosaic {
        panel: CompositePanel {
            haplotypes: hmm_haplotypes,
            source_samples,
            marker_indices: (0..hmm_marker_indices.len()).collect(),
            allele_counts: vec![2; hmm_marker_indices.len()],
        },
        segments,
    })
}

/// Construct a target-specific HMM mosaic from Beagle-style IBS candidates.
/// The number of states is the number of composite tracks that were actually
/// created, which can be smaller than `max_states`.
pub(crate) fn build_ibs_target_mosaic_hmm_packed(
    packed_reference: &PackedBiallelicReference,
    hmm_marker_indices: &[usize],
    step_ranges: &[(usize, usize)],
    candidate_sets: &[Vec<usize>],
    max_states: usize,
    target_haplotype: usize,
    seed: u64,
) -> Result<PackedTargetMosaic, String> {
    let n_markers = packed_reference.n_markers();
    let n_haplotypes = packed_reference.n_haplotypes();
    if n_markers == 0
        || n_haplotypes == 0
        || hmm_marker_indices.is_empty()
        || hmm_marker_indices.iter().any(|&marker| marker >= n_markers)
        || hmm_marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
        || step_ranges.last().is_none_or(|range| range.1 != n_markers)
        || max_states == 0
    {
        return Err("imputation IBS mosaic dimensions are invalid".to_owned());
    }
    let mut state_segments =
        select_ibs_donor_segments(step_ranges, candidate_sets, n_haplotypes, max_states)?;
    if state_segments.is_empty() {
        let fallback =
            ibs_fallback_donor_haplotypes(n_haplotypes, max_states, target_haplotype, seed);
        state_segments.reserve(fallback.len());
        for donor in fallback {
            state_segments.push(vec![DonorHapSegment {
                start_marker: 0,
                end_marker: n_markers,
                donor_haplotype: donor,
            }]);
        }
    }
    let state_count = state_segments.len();
    let mut haplotypes = PackedHaplotypes::empty(state_count, hmm_marker_indices.len(), 1)?;
    let mut source_samples = Vec::with_capacity(state_count);
    let mut boundaries = Vec::with_capacity(state_count.saturating_mul(2).saturating_add(2));
    boundaries.push(0);
    boundaries.push(n_markers);
    for track in &state_segments {
        let mut covered = 0usize;
        for segment in track {
            if segment.start_marker != covered
                || segment.end_marker <= segment.start_marker
                || segment.end_marker > n_markers
                || segment.donor_haplotype >= n_haplotypes
            {
                return Err("imputation IBS donor track is not a full marker partition".to_owned());
            }
            boundaries.push(segment.start_marker);
            boundaries.push(segment.end_marker);
            covered = segment.end_marker;
        }
        if covered != n_markers {
            return Err("imputation IBS donor track does not cover the marker window".to_owned());
        }
        source_samples.push(track[0].donor_haplotype / 2);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut segment_donors = vec![0usize; state_count];
    let mut segment_cursors = vec![0usize; state_count];
    let mut segments = Vec::with_capacity(boundaries.len().saturating_sub(1));
    let mut next_hmm_marker = 0usize;
    let mut word_scratch = vec![0u64; state_count.div_ceil(64)];
    for pair in boundaries.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        if start >= end {
            continue;
        }
        for state in 0..state_count {
            let track = &state_segments[state];
            while track[segment_cursors[state]].end_marker <= start {
                segment_cursors[state] += 1;
            }
            let donor_segment = track[segment_cursors[state]];
            if donor_segment.start_marker > start || donor_segment.end_marker < end {
                return Err("imputation IBS donor tracks have inconsistent boundaries".to_owned());
            }
            segment_donors[state] = donor_segment.donor_haplotype;
        }
        while hmm_marker_indices
            .get(next_hmm_marker)
            .is_some_and(|&marker| marker < end)
        {
            let marker = hmm_marker_indices[next_hmm_marker];
            if marker < start {
                return Err("imputation IBS marker is outside its donor segment".to_owned());
            }
            packed_reference.set_mosaic_marker_at_with_scratch(
                marker,
                next_hmm_marker,
                &segment_donors,
                &mut haplotypes,
                &mut word_scratch,
            )?;
            next_hmm_marker += 1;
        }
        segments.push(PackedTargetMosaicSegment {
            start,
            end,
            donors: segment_donors.clone(),
        });
    }
    if next_hmm_marker != hmm_marker_indices.len() {
        return Err("imputation IBS mosaic does not cover every target marker".to_owned());
    }
    Ok(PackedTargetMosaic {
        panel: CompositePanel {
            haplotypes,
            source_samples,
            marker_indices: (0..hmm_marker_indices.len()).collect(),
            allele_counts: vec![2; hmm_marker_indices.len()],
        },
        segments,
    })
}

/// Build a target-specific donor mosaic from a marker-major reference plane.
/// This path avoids constructing a second full `PhasedWindow` copy.
pub(crate) fn build_pbwt_target_mosaic_packed(
    packed_reference: &PackedBiallelicReference,
    pbwt_index: &SegmentPbwtIndex,
    query_alleles: &[Option<u16>],
    hmm_marker_indices: &[usize],
    cm_positions: &[f64],
    segment_cm: f64,
    max_states: usize,
) -> Result<CompositePanel, String> {
    let n_markers = packed_reference.n_markers();
    let n_haplotypes = packed_reference.n_haplotypes();
    if n_markers == 0
        || query_alleles.len() != n_markers
        || hmm_marker_indices.is_empty()
        || hmm_marker_indices.iter().any(|&marker| marker >= n_markers)
        || hmm_marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
        || cm_positions.len() != n_markers
        || pbwt_index.segments.is_empty()
        || pbwt_index.n_haplotypes != n_haplotypes
        || !segment_cm.is_finite()
        || segment_cm <= 0.0
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
        || max_states == 0
    {
        return Err("PBWT target mosaic dimensions, map, or segment length are invalid".to_owned());
    }
    let state_count = max_states.min(n_haplotypes);
    let mut haplotypes = PackedHaplotypes::empty(state_count, n_markers, 1)?;
    let mut source_samples = vec![0usize; state_count];
    let mut expected_start = 0usize;
    for (segment_index, segment) in pbwt_index.segments.iter().enumerate() {
        if segment.start != expected_start
            || segment.end <= segment.start
            || segment.end > n_markers
        {
            return Err("PBWT segment index does not cover the reference window".to_owned());
        }
        let donors = super::pbwt::segment_neighbors_from_alleles(
            pbwt_index,
            segment_index,
            n_markers,
            query_alleles,
            state_count,
            |marker, haplotype| packed_reference.allele_unchecked(marker, haplotype),
        )?;
        if donors.len() != state_count || donors.iter().any(|&donor| donor >= n_haplotypes) {
            return Err("PBWT segment returned invalid reference haplotypes".to_owned());
        }
        if segment_index == 0 {
            for (state, &donor) in donors.iter().enumerate() {
                source_samples[state] = donor / 2;
            }
        }
        for marker in segment.start..segment.end {
            packed_reference.set_mosaic_marker(marker, &donors, &mut haplotypes)?;
        }
        expected_start = segment.end;
    }
    if expected_start != n_markers {
        return Err("PBWT segments do not span the reference window".to_owned());
    }
    Ok(CompositePanel {
        haplotypes,
        source_samples,
        marker_indices: hmm_marker_indices.to_vec(),
        allele_counts: vec![2; n_markers],
    })
}

/// Construct a target-specific local IBS mosaic directly from the full
/// reference window. Each genetic segment queries its own shared PBWT order,
/// so no full reference haplotype matrix or per-target full scan is needed.
pub(crate) fn build_pbwt_target_mosaic(
    window: &PhasedWindow,
    pbwt_index: &SegmentPbwtIndex,
    query_alleles: &[Option<u16>],
    hmm_marker_indices: &[usize],
    cm_positions: &[f64],
    segment_cm: f64,
    max_states: usize,
    packed_reference: Option<&PackedBiallelicReference>,
) -> Result<CompositePanel, String> {
    if window.rows.is_empty()
        || query_alleles.len() != window.rows.len()
        || hmm_marker_indices.is_empty()
        || hmm_marker_indices
            .iter()
            .any(|&marker| marker >= window.rows.len())
        || hmm_marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
        || cm_positions.len() != window.rows.len()
        || pbwt_index.segments.is_empty()
        || !segment_cm.is_finite()
        || segment_cm <= 0.0
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("PBWT target mosaic dimensions, map, or segment length are invalid".to_owned());
    }
    let n_samples = window.rows[0].calls.len();
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "reference haplotype count overflow".to_owned())?;
    if n_samples == 0
        || window.rows.iter().any(|row| row.calls.len() != n_samples)
        || pbwt_index.n_haplotypes != n_haplotypes
        || max_states == 0
    {
        return Err("PBWT target mosaic reference dimensions are invalid".to_owned());
    }
    let state_count = max_states.min(n_haplotypes);
    if let Some(packed_reference) = packed_reference {
        if packed_reference.n_haplotypes != n_haplotypes
            || packed_reference.n_markers != window.rows.len()
        {
            return Err("packed reference differs from the PBWT target window".to_owned());
        }
    }
    let max_allele = window
        .rows
        .iter()
        .map(|row| row.meta.allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let mut haplotypes =
        PackedHaplotypes::empty(state_count, window.rows.len(), allele_width(max_allele))?;
    let mut source_samples = vec![0usize; state_count];
    let mut expected_start = 0usize;
    for (segment_index, segment) in pbwt_index.segments.iter().enumerate() {
        if segment.start != expected_start
            || segment.end <= segment.start
            || segment.end > window.rows.len()
        {
            return Err("PBWT segment index does not cover the reference window".to_owned());
        }
        let donors = super::pbwt::segment_neighbors(
            pbwt_index,
            segment_index,
            window,
            query_alleles,
            state_count,
        )?;
        if donors.len() != state_count {
            return Err("PBWT segment returned too few reference haplotypes".to_owned());
        }
        if packed_reference
            .is_some_and(|reference| donors.iter().any(|&donor| donor >= reference.n_haplotypes))
        {
            return Err("PBWT donor is outside the packed reference".to_owned());
        }
        if segment_index == 0 {
            for (state, &donor) in donors.iter().enumerate() {
                source_samples[state] = donor / 2;
            }
        }
        let bulk_biallelic = haplotypes.bits_per_allele == 1 && state_count % 64 == 0;
        for marker in segment.start..segment.end {
            if let Some(packed_reference) = packed_reference {
                packed_reference.set_mosaic_marker(marker, &donors, &mut haplotypes)?;
                continue;
            }
            let calls = &window.rows[marker].calls;
            if bulk_biallelic {
                haplotypes.set_biallelic_marker_with(marker, |state| {
                    let donor = donors[state];
                    calls.get_allele_unchecked(donor)
                })?;
            } else {
                for (state, &donor) in donors.iter().enumerate() {
                    haplotypes.set_allele(marker, state, calls.get_allele_unchecked(donor))?;
                }
            }
        }
        expected_start = segment.end;
    }
    if expected_start != window.rows.len() {
        return Err("PBWT segments do not span the reference window".to_owned());
    }
    Ok(CompositePanel {
        haplotypes,
        source_samples,
        marker_indices: hmm_marker_indices.to_vec(),
        allele_counts: window
            .rows
            .iter()
            .map(|row| row.meta.allele_count())
            .collect(),
    })
}

fn build_reference_with_index(
    window: &PhasedWindow,
    exclude_sample: Option<usize>,
    max_states: usize,
    pbwt_index: Option<&PbwtIndex>,
) -> Result<CompositePanel, String> {
    let view = PhaseHaplotypeView::target(window);
    build_reference_with_index_view(&view, exclude_sample, max_states, pbwt_index)
}

fn build_reference_with_index_view(
    view: &PhaseHaplotypeView<'_>,
    exclude_sample: Option<usize>,
    max_states: usize,
    pbwt_index: Option<&PbwtIndex>,
) -> Result<CompositePanel, String> {
    if view.n_markers() == 0 || max_states == 0 {
        return Err("composite panel requires variants and a positive state count".to_owned());
    }
    let n_samples = view.n_samples();
    if n_samples == 0 {
        return Err("composite panel input contains zero samples".to_owned());
    }
    if let Some(index) = exclude_sample {
        if index >= n_samples {
            return Err(format!("excluded sample index {index} is out of range"));
        }
    }
    let candidate_samples = n_samples - if exclude_sample.is_some() { 1 } else { 0 };
    let candidate_count = candidate_samples
        .checked_mul(2)
        .ok_or_else(|| "phase candidate haplotype count overflow".to_owned())?;
    if candidate_count == 0 {
        return Err("cannot build a composite panel after excluding the target".to_owned());
    }

    let max_allele = (0..view.n_markers())
        .map(|marker| view.marker(marker).allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let bits_per_allele = allele_width(max_allele);
    let selected = if let (Some(target), Some(index)) = (exclude_sample, pbwt_index) {
        super::pbwt::neighbors(index, target, n_samples, max_states)?
    } else {
        systematic_indices(candidate_count, max_states)
            .into_iter()
            .map(|candidate| {
                let (sample, hap) = candidate_sample_hap(candidate, n_samples, exclude_sample);
                sample * 2 + hap
            })
            .collect()
    };
    let mut haplotypes =
        PackedHaplotypes::empty(selected.len(), view.n_markers(), bits_per_allele)?;
    let mut source_samples = Vec::with_capacity(selected.len());
    for (state, &candidate) in selected.iter().enumerate() {
        let sample = candidate / 2;
        let hap = candidate & 1;
        source_samples.push(sample);
        for marker in 0..view.n_markers() {
            haplotypes.set_allele(marker, state, view.allele(marker, sample, hap))?;
        }
    }
    Ok(CompositePanel {
        haplotypes,
        source_samples,
        marker_indices: (0..view.n_markers()).collect(),
        allele_counts: (0..view.n_markers())
            .map(|marker| view.marker(marker).allele_count())
            .collect(),
    })
}

/// Build one target-haplotype-specific composite panel from the bounded
/// reference panel. Within each genetic segment, donors are ranked by their
/// match to the observed target haplotype; each output state follows the same
/// rank across segments, creating a local-IBS mosaic without materializing the
/// full reference haplotype matrix.
pub(crate) fn build_target_mosaic(
    reference: &CompositePanel,
    observations: &PackedHaplotypes,
    target_marker_indices: &[usize],
    union_cm_positions: &[f64],
    segment_cm: f64,
) -> Result<CompositePanel, String> {
    let marker_count = reference.haplotypes.n_markers;
    let state_count = reference.haplotypes.n_haplotypes;
    if marker_count == 0
        || state_count == 0
        || reference.allele_counts.len() != marker_count
        || union_cm_positions.len() != marker_count
        || observations.n_haplotypes != 1
        || observations.n_markers != target_marker_indices.len()
        || target_marker_indices.is_empty()
        || !segment_cm.is_finite()
        || segment_cm <= 0.0
        || target_marker_indices
            .iter()
            .any(|&marker| marker >= marker_count)
        || target_marker_indices
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
        || union_cm_positions.iter().any(|cm| !cm.is_finite())
        || union_cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("target mosaic dimensions, map, or segment length are invalid".to_owned());
    }

    let mut haplotypes = PackedHaplotypes::empty(
        state_count,
        marker_count,
        reference.haplotypes.bits_per_allele,
    )?;
    let mut source_samples = vec![0usize; state_count];
    let mut target_marker = 0usize;
    let mut segment_start = 0usize;
    let mut first_segment = true;
    while segment_start < marker_count {
        let segment_cm_start = union_cm_positions[segment_start];
        let mut segment_end = segment_start + 1;
        while segment_end < marker_count
            && union_cm_positions[segment_end] - segment_cm_start < segment_cm
        {
            segment_end += 1;
        }

        let target_start = target_marker;
        while target_marker < target_marker_indices.len()
            && target_marker_indices[target_marker] < segment_end
        {
            target_marker += 1;
        }
        let mut scores = vec![0i64; state_count];
        let mut comparisons = vec![0usize; state_count];
        for target_index in target_start..target_marker {
            let Some(observed) = observations.allele(target_index, 0) else {
                continue;
            };
            let union_index = target_marker_indices[target_index];
            for (state, score) in scores.iter_mut().enumerate() {
                if let Some(allele) = reference.haplotypes.allele(union_index, state) {
                    *score += if allele == observed { 1 } else { -1 };
                    comparisons[state] += 1;
                }
            }
        }
        let mut ranked_states = (0..state_count).collect::<Vec<_>>();
        ranked_states.sort_unstable_by(|left, right| {
            let score_order = match (comparisons[*left], comparisons[*right]) {
                (0, 0) => std::cmp::Ordering::Equal,
                (0, _) => std::cmp::Ordering::Greater,
                (_, 0) => std::cmp::Ordering::Less,
                (left_count, right_count) => {
                    let left_score = scores[*left] as f64 / left_count as f64;
                    let right_score = scores[*right] as f64 / right_count as f64;
                    right_score.total_cmp(&left_score)
                }
            };
            score_order
                .then_with(|| comparisons[*right].cmp(&comparisons[*left]))
                .then_with(|| left.cmp(right))
        });
        for (mosaic_state, &donor_state) in ranked_states.iter().enumerate() {
            if first_segment {
                source_samples[mosaic_state] = reference
                    .source_samples
                    .get(donor_state)
                    .copied()
                    .unwrap_or(0);
            }
            for marker in segment_start..segment_end {
                haplotypes.set_allele(
                    marker,
                    mosaic_state,
                    reference.haplotypes.allele(marker, donor_state),
                )?;
            }
        }
        first_segment = false;
        segment_start = segment_end;
    }

    Ok(CompositePanel {
        haplotypes,
        source_samples,
        marker_indices: target_marker_indices.to_vec(),
        allele_counts: reference.allele_counts.clone(),
    })
}

pub(crate) fn build_reference_with_ibd2(
    window: &PhasedWindow,
    exclude_sample: Option<usize>,
    max_states: usize,
    intervals: &[Ibd2Interval],
    pbwt_index: &PbwtIndex,
) -> Result<CompositePanel, String> {
    let view = PhaseHaplotypeView::target(window);
    build_reference_with_ibd2_view(&view, exclude_sample, max_states, intervals, pbwt_index)
}

fn build_reference_with_ibd2_view(
    view: &PhaseHaplotypeView<'_>,
    exclude_sample: Option<usize>,
    max_states: usize,
    intervals: &[Ibd2Interval],
    pbwt_index: &PbwtIndex,
) -> Result<CompositePanel, String> {
    let mut panel =
        build_reference_with_index_view(view, exclude_sample, max_states, Some(pbwt_index))?;
    let Some(target) = exclude_sample else {
        return Ok(panel);
    };
    for state in 0..panel.haplotypes.n_haplotypes {
        let source = panel.source_samples[state];
        for interval in intervals
            .iter()
            .filter(|interval| interval.source_sample == source && interval.target_sample == target)
        {
            for marker in interval.start_marker..interval.end_marker.min(panel.haplotypes.n_markers)
            {
                panel.haplotypes.set_allele(marker, state, None)?;
            }
        }
    }
    Ok(panel)
}

pub(crate) fn build_phase_reference(
    window: &PhasedWindow,
    input: &PhaseWindow,
    target_sample: usize,
    max_states: usize,
    intervals: &[Ibd2Interval],
    low_frequency_mask: Option<&[bool]>,
    pbwt_index: &PbwtIndex,
) -> Result<CompositePanel, String> {
    let view = PhaseHaplotypeView::target(window);
    build_phase_reference_view(
        &view,
        input,
        target_sample,
        max_states,
        intervals,
        low_frequency_mask,
        pbwt_index,
    )
}

pub(crate) fn build_phase_reference_view(
    view: &PhaseHaplotypeView<'_>,
    input: &PhaseWindow,
    target_sample: usize,
    max_states: usize,
    intervals: &[Ibd2Interval],
    low_frequency_mask: Option<&[bool]>,
    pbwt_index: &PbwtIndex,
) -> Result<CompositePanel, String> {
    let mut panel = build_reference_with_ibd2_view(
        view,
        Some(target_sample),
        max_states,
        intervals,
        pbwt_index,
    )?;
    let Some(mask) = low_frequency_mask else {
        return Ok(panel);
    };
    if mask.len() != input.rows.len() || input.rows.len() != view.n_markers() {
        return Err("low-frequency mask and phase panel have different marker counts".to_owned());
    }
    for (marker, &is_low_frequency) in mask.iter().enumerate() {
        if !is_low_frequency {
            continue;
        }
        let input_row = &input.rows[marker];
        let mut counts = vec![0usize; input_row.meta.allele_count()];
        for call in input_row.calls.iter() {
            for allele in call.alleles.into_iter().flatten() {
                counts[usize::from(allele)] += 1;
            }
        }
        let major_allele = counts
            .iter()
            .enumerate()
            .max_by_key(|(allele, count)| (**count, std::cmp::Reverse(*allele)))
            .map(|(allele, _)| allele as u16)
            .unwrap_or(0);
        let target_call = input_row.calls.get(target_sample).unwrap();
        // Beagle uses this carrier-conditioned projection only to orient a
        // fully observed low-frequency heterozygote. For a missing or
        // partially missing target genotype, Stage2 imputes each haplotype
        // from the selected donor haplotype's actual allele. Replacing both
        // haplotypes of a heterozygous donor with the major allele here would
        // erase exactly the rare-allele evidence needed for that imputation.
        if !target_call.is_heterozygous() {
            continue;
        }
        for state in 0..panel.haplotypes.n_haplotypes {
            let source = panel.source_samples[state];
            if !super::ibd2::can_contribute(source, target_sample, marker, intervals) {
                continue;
            }
            let donor_call = view.call(marker, source);
            if !donor_call.is_heterozygous() {
                continue;
            }
            let target_alleles = target_call
                .alleles
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let donor_rare_alleles = donor_call
                .alleles
                .into_iter()
                .flatten()
                .filter(|&allele| allele != major_allele && target_alleles.contains(&allele))
                .collect::<Vec<_>>();
            let assigned = donor_rare_alleles
                .into_iter()
                .min_by_key(|&allele| counts.get(usize::from(allele)).copied().unwrap_or(0))
                .unwrap_or(major_allele);
            panel.haplotypes.set_allele(marker, state, Some(assigned))?;
        }
    }
    Ok(panel)
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use super::pack_donor_alleles_avx2;
    use super::{
        build_ibs_target_mosaic_hmm_packed, build_low_frequency_phase_mosaic,
        build_pbwt_target_mosaic, build_pbwt_target_mosaic_hmm_packed,
        build_pbwt_target_mosaic_packed, build_phase_reference, build_random_phase_panel,
        donor_haplotype_at, fill_joint_phase_mosaic, fill_joint_phase_mosaic_from_packed,
        fill_joint_phase_mosaic_view, fill_low_frequency_phase_mosaic_haplotypes,
        ibs_fallback_donor_haplotypes, select_donor_segments, select_ibs_donor_segments,
        stage2_panel_packed_view_eligible, CompositePanel, DonorHapSegment,
        PackedBiallelicReference, PackedTargetMosaic, PackedTargetMosaicSegment,
        PhaseHaplotypeView,
    };
    use crate::imp::pbwt::{self, build_low_frequency_phase_donor_index, PbwtDirection};
    use crate::imp::types::{
        DiploidCall, DiploidCalls, PackedHaplotypes, PhaseWindow, PhasedRow, PhasedWindow,
        VariantMeta, VariantRow,
    };

    fn target_panel_allele(marker: usize, haplotype: usize) -> u16 {
        ((marker * 17 + haplotype * 13 + 3) % 5 < 2) as u16
    }

    fn reference_panel_allele(marker: usize, haplotype: usize) -> u16 {
        ((marker * 19 + haplotype * 7 + 1) % 11 < 5) as u16
    }

    fn assert_packed_fill_matches_scalar(
        marker_count: usize,
        marker_indices: &[usize],
        target_samples: usize,
        reference_samples: usize,
        state_segments: &[Vec<DonorHapSegment>],
    ) -> PackedHaplotypes {
        let target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: (0..marker_count)
                .map(|marker| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(
                        (0..target_samples)
                            .map(|sample| DiploidCall {
                                alleles: [
                                    Some(target_panel_allele(marker, sample * 2)),
                                    Some(target_panel_allele(marker, sample * 2 + 1)),
                                ],
                                phased: true,
                            })
                            .collect(),
                    ),
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
            emit_end: marker_count,
        };
        let target_packed = PackedHaplotypes::complete_biallelic_state_major(&target)
            .unwrap()
            .unwrap();
        let reference_packed = if reference_samples == 0 {
            None
        } else {
            let mut packed =
                PackedHaplotypes::empty_state_major(reference_samples * 2, marker_count, 1)
                    .unwrap();
            for marker in 0..marker_count {
                for haplotype in 0..reference_samples * 2 {
                    packed
                        .set_allele(
                            marker,
                            haplotype,
                            Some(reference_panel_allele(marker, haplotype)),
                        )
                        .unwrap();
                }
            }
            packed.mark_complete();
            Some(packed)
        };
        let full_view = PhaseHaplotypeView::new(
            &target,
            Some(&target_packed),
            reference_packed.as_ref(),
            target_samples,
        )
        .unwrap();
        let stage1_target = PhasedWindow {
            chrom: target.chrom.clone(),
            index: target.index,
            rows: marker_indices
                .iter()
                .map(|&marker| target.rows[marker].clone())
                .collect(),
            emit_start: 0,
            emit_end: marker_indices.len(),
        };
        let stage1_target_packed = PackedHaplotypes::complete_biallelic_state_major(&stage1_target)
            .unwrap()
            .unwrap();
        let stage1_reference_packed = reference_packed.as_ref().map(|packed| {
            packed
                .project_state_major_biallelic(marker_indices)
                .unwrap()
        });
        let stage1_view = PhaseHaplotypeView::new(
            &stage1_target,
            Some(&stage1_target_packed),
            stage1_reference_packed.as_ref(),
            target_samples,
        )
        .unwrap();
        let mut scalar =
            PackedHaplotypes::empty_state_major(state_segments.len(), marker_indices.len(), 1)
                .unwrap();
        let mut projected =
            PackedHaplotypes::empty_state_major(state_segments.len(), marker_indices.len(), 1)
                .unwrap();
        let scalar_sources = fill_low_frequency_phase_mosaic_haplotypes(
            &full_view,
            None,
            state_segments,
            marker_indices,
            &mut scalar,
        )
        .unwrap();
        let projected_sources = fill_low_frequency_phase_mosaic_haplotypes(
            &full_view,
            Some(&stage1_view),
            state_segments,
            marker_indices,
            &mut projected,
        )
        .unwrap();
        assert_eq!(scalar_sources, projected_sources);
        assert_eq!(scalar.allele_bits, projected.allele_bits);
        assert_eq!(scalar.missing_bits, projected.missing_bits);
        assert_eq!(scalar.is_complete(), projected.is_complete());
        projected
    }

    #[test]
    fn phase_haplotype_view_joins_target_and_packed_reference_samples() {
        let target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: (0..2)
                .map(|marker| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: row_calls(&[[Some(0), Some(1)], [Some(1), Some(0)]]),
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
            emit_end: 2,
        };
        let target_packed = PackedHaplotypes::complete_biallelic_state_major(&target)
            .unwrap()
            .unwrap();
        let mut reference = PackedHaplotypes::empty_state_major(2, 2, 1).unwrap();
        reference.set_allele(0, 0, Some(1)).unwrap();
        reference.set_allele(0, 1, Some(0)).unwrap();
        reference.set_allele(1, 0, Some(0)).unwrap();
        reference.set_allele(1, 1, Some(1)).unwrap();
        reference.mark_complete();
        let view = PhaseHaplotypeView::new(
            &target,
            Some(&target_packed),
            Some(&reference),
            target.rows[0].calls.len(),
        )
        .unwrap();

        assert_eq!(view.target_samples(), 2);
        assert_eq!(view.n_samples(), 3);
        assert_eq!(view.n_markers(), 2);
        assert_eq!(view.allele(0, 0, 0), Some(0));
        assert_eq!(view.allele(0, 2, 1), Some(0));
        let mut destination = PackedHaplotypes::empty_state_major(1, 1, 1).unwrap();
        destination
            .resize_state_major_biallelic_for_overwrite(1, 2)
            .unwrap();
        view.copy_biallelic_segment_from(&mut destination, 0, 0, 2, 4, false)
            .unwrap();
        assert_eq!(destination.allele(0, 0), Some(1));
        assert_eq!(destination.allele(1, 0), Some(0));
    }

    #[test]
    fn random_phase_fallback_uses_state_major_panel_layout() {
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: (0..4)
                .map(|marker| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: row_calls(&[
                        [Some(0), Some(0)],
                        [Some(0), Some(1)],
                        [Some(1), Some(0)],
                        [Some(1), Some(1)],
                    ]),
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
            emit_end: 4,
        };

        let panel = build_random_phase_panel(&window, 0, 6, 17).unwrap();

        assert!(panel.haplotypes.is_state_major());
    }

    #[test]
    fn pbwt_target_mosaic_bulk_rows_match_scalar_donor_alleles() {
        let marker_count = 4;
        let sample_count = 32;
        let cm_positions = (0..marker_count)
            .map(|marker| marker as f64 * 0.1)
            .collect::<Vec<_>>();
        let rows = (0..marker_count)
            .map(|marker| {
                let calls = (0..sample_count)
                    .map(|sample| {
                        if (marker + sample) % 29 == 0 {
                            [None, None]
                        } else {
                            [
                                Some(u16::from((sample + marker) % 5 < 2)),
                                Some(u16::from((sample * 3 + marker) % 7 < 3)),
                            ]
                        }
                    })
                    .collect::<Vec<_>>();
                PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: row_calls(&calls),
                    dosage: None,
                    ap1: None,
                    ap2: None,
                    gp: None,
                    info_af: None,
                    info_dr2: None,
                    imputed: false,
                }
            })
            .collect::<Vec<_>>();
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            emit_start: 0,
            emit_end: marker_count,
        };
        let query = vec![Some(0); marker_count];
        let marker_indices = vec![0, 1, 2, 3];
        let searchable = vec![true; marker_count];
        let index = pbwt::build_segment_index(
            &window,
            &cm_positions,
            &searchable,
            6.0,
            PbwtDirection::Forward,
        )
        .unwrap();
        let donors = pbwt::segment_neighbors(&index, 0, &window, &query, 64).unwrap();

        let observed = build_pbwt_target_mosaic(
            &window,
            &index,
            &query,
            &marker_indices,
            &cm_positions,
            6.0,
            64,
            None,
        )
        .unwrap();
        let mut expected = PackedHaplotypes::empty(64, marker_count, 1).unwrap();
        for marker in 0..marker_count {
            for (state, &donor) in donors.iter().enumerate() {
                let allele = window.rows[marker].calls.get(donor / 2).unwrap().alleles[donor & 1];
                expected.set_allele(marker, state, allele).unwrap();
            }
        }

        assert_eq!(observed.haplotypes.allele_bits, expected.allele_bits);
        assert_eq!(observed.haplotypes.missing_bits, expected.missing_bits);
        assert_eq!(
            observed.source_samples,
            donors.iter().map(|donor| donor / 2).collect::<Vec<_>>()
        );
        assert_eq!(observed.marker_indices, marker_indices);
        assert_eq!(observed.allele_counts, vec![2; marker_count]);
    }

    #[test]
    fn prepacked_reference_mosaic_matches_diploid_call_projection() {
        let marker_count = 4;
        let sample_count = 32;
        let cm_positions = (0..marker_count)
            .map(|marker| marker as f64 * 0.1)
            .collect::<Vec<_>>();
        let reference_present = [true, false, true, true];
        let rows = (0..marker_count)
            .map(|marker| {
                let calls = (0..sample_count)
                    .map(|sample| {
                        if !reference_present[marker] {
                            [None, None]
                        } else {
                            [
                                Some(u16::from((sample + marker) % 5 < 2)),
                                Some(u16::from((sample * 3 + marker) % 7 < 3)),
                            ]
                        }
                    })
                    .collect::<Vec<_>>();
                PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: row_calls(&calls),
                    dosage: None,
                    ap1: None,
                    ap2: None,
                    gp: None,
                    info_af: None,
                    info_dr2: None,
                    imputed: false,
                }
            })
            .collect::<Vec<_>>();
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            emit_start: 0,
            emit_end: marker_count,
        };
        let query = vec![Some(0), None, Some(1), Some(0)];
        let marker_indices = vec![0, 2, 3];
        let searchable = reference_present.to_vec();
        let index = pbwt::build_segment_index(
            &window,
            &cm_positions,
            &searchable,
            6.0,
            PbwtDirection::Forward,
        )
        .unwrap();
        let donors = pbwt::segment_neighbors(&index, 0, &window, &query, 64).unwrap();
        let packed = PackedBiallelicReference::from_window(&window, &reference_present)
            .unwrap()
            .unwrap();
        let observed = build_pbwt_target_mosaic(
            &window,
            &index,
            &query,
            &marker_indices,
            &cm_positions,
            6.0,
            64,
            Some(&packed),
        )
        .unwrap();

        let mut expected = PackedHaplotypes::empty(64, marker_count, 1).unwrap();
        for marker in 0..marker_count {
            for (state, &donor) in donors.iter().enumerate() {
                let allele = if reference_present[marker] {
                    window.rows[marker].calls.get_allele_unchecked(donor)
                } else {
                    None
                };
                expected.set_allele(marker, state, allele).unwrap();
            }
        }
        assert_eq!(observed.haplotypes.allele_bits, expected.allele_bits);
        assert_eq!(observed.haplotypes.missing_bits, expected.missing_bits);
    }

    #[test]
    fn direct_reference_packing_matches_merged_window_pbwt_and_mosaic() {
        let sample_count = 32;
        let reference_calls = (0..3)
            .map(|marker| {
                (0..sample_count)
                    .map(|sample| {
                        [
                            Some(u16::from((sample + marker) % 5 < 2)),
                            Some(u16::from((sample * 3 + marker) % 7 < 3)),
                        ]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let make_meta = |position: u64| VariantMeta {
            chrom: "1".to_owned(),
            pos: position,
            id: format!("v{position}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let reference_rows = reference_calls
            .iter()
            .enumerate()
            .map(|(marker, calls)| PhasedRow {
                meta: make_meta([1, 3, 4][marker]),
                calls: row_calls(calls),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
            .collect::<Vec<_>>();
        let reference_row_indices = [Some(0), None, Some(1), Some(2)];
        let reference_swaps = [false, false, true, false];
        let reference_present = [true, false, true, true];
        let merged_rows = vec![
            reference_rows[0].clone(),
            PhasedRow {
                meta: make_meta(2),
                calls: row_calls(&vec![[None, None]; sample_count]),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            },
            PhasedRow {
                calls: row_calls(
                    &reference_calls[1]
                        .iter()
                        .map(|call| {
                            [
                                call[0].map(|allele| allele ^ 1),
                                call[1].map(|allele| allele ^ 1),
                            ]
                        })
                        .collect::<Vec<_>>(),
                ),
                ..reference_rows[1].clone()
            },
            reference_rows[2].clone(),
        ];
        let merged_window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: merged_rows,
            emit_start: 0,
            emit_end: 4,
        };
        let packed_direct = PackedBiallelicReference::from_reference_rows(
            &reference_rows,
            &reference_row_indices,
            &reference_swaps,
            sample_count,
        )
        .unwrap();
        let packed_merged =
            PackedBiallelicReference::from_window(&merged_window, &reference_present)
                .unwrap()
                .unwrap();
        assert_eq!(packed_direct.allele_words, packed_merged.allele_words);
        assert_eq!(packed_direct.missing_markers, packed_merged.missing_markers);
        assert_eq!(
            packed_direct.major_alleles(),
            super::reference_major_alleles(&merged_window).unwrap()
        );
        let (packed_major, packed_alt_frequencies) = packed_direct.allele_summaries();
        assert_eq!(packed_major, packed_direct.major_alleles());
        let expected_alt_frequencies = reference_row_indices
            .iter()
            .enumerate()
            .map(|(marker, reference_index)| {
                reference_index.map(|reference_index| {
                    let row = &reference_rows[reference_index];
                    let swapped = reference_swaps[marker];
                    let (alt_count, observed_count) = row
                        .calls
                        .iter()
                        .flat_map(|call| call.alleles.into_iter().flatten())
                        .fold((0usize, 0usize), |(alt, observed), allele| {
                            (
                                alt + usize::from(if swapped { allele == 0 } else { allele == 1 }),
                                observed + 1,
                            )
                        });
                    alt_count as f32 / observed_count as f32
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(packed_alt_frequencies, expected_alt_frequencies);

        let cm_positions = [0.0, 0.02, 0.04, 0.06];
        let searchable_markers = [true, false, true, true];
        let old_index = pbwt::build_segment_index(
            &merged_window,
            &cm_positions,
            &searchable_markers,
            0.05,
            PbwtDirection::Forward,
        )
        .unwrap();
        let direct_index = pbwt::build_segment_index_from_alleles(
            packed_direct.n_haplotypes(),
            packed_direct.n_markers(),
            1,
            &cm_positions,
            &searchable_markers,
            0.05,
            PbwtDirection::Forward,
            |marker, haplotype| packed_direct.allele_unchecked(marker, haplotype),
        )
        .unwrap();
        assert_eq!(old_index.n_haplotypes, direct_index.n_haplotypes);
        assert_eq!(old_index.direction, direct_index.direction);
        assert_eq!(
            old_index
                .segments
                .iter()
                .map(|segment| (segment.start, segment.end, &segment.markers, &segment.order))
                .collect::<Vec<_>>(),
            direct_index
                .segments
                .iter()
                .map(|segment| (segment.start, segment.end, &segment.markers, &segment.order))
                .collect::<Vec<_>>()
        );

        let query = [Some(0), None, Some(1), Some(0)];
        let hmm_markers = [0, 2, 3];
        let old_mosaic = build_pbwt_target_mosaic(
            &merged_window,
            &old_index,
            &query,
            &hmm_markers,
            &cm_positions,
            0.05,
            12,
            Some(&packed_merged),
        )
        .unwrap();
        let direct_mosaic = build_pbwt_target_mosaic_packed(
            &packed_direct,
            &direct_index,
            &query,
            &hmm_markers,
            &cm_positions,
            0.05,
            12,
        )
        .unwrap();
        let compact_mosaic = build_pbwt_target_mosaic_hmm_packed(
            &packed_direct,
            &direct_index,
            &query,
            &hmm_markers,
            &cm_positions,
            0.05,
            12,
        )
        .unwrap();
        assert_eq!(
            old_mosaic.haplotypes.allele_bits,
            direct_mosaic.haplotypes.allele_bits
        );
        assert_eq!(
            old_mosaic.haplotypes.missing_bits,
            direct_mosaic.haplotypes.missing_bits
        );
        assert_eq!(old_mosaic.source_samples, direct_mosaic.source_samples);
        assert_eq!(old_mosaic.marker_indices, direct_mosaic.marker_indices);
        assert_eq!(old_mosaic.allele_counts, direct_mosaic.allele_counts);
        assert_eq!(compact_mosaic.panel.haplotypes.n_markers, hmm_markers.len());
        assert_eq!(
            compact_mosaic.panel.source_samples,
            old_mosaic.source_samples
        );
        for (hmm_marker, &union_marker) in hmm_markers.iter().enumerate() {
            for state in 0..old_mosaic.haplotypes.n_haplotypes {
                assert_eq!(
                    compact_mosaic.panel.haplotypes.allele(hmm_marker, state),
                    old_mosaic.haplotypes.allele(union_marker, state),
                    "HMM marker={union_marker} state={state}"
                );
            }
        }
        let mut mosaic_allele_words = vec![0u64; old_mosaic.haplotypes.n_haplotypes.div_ceil(64)];
        let mut mosaic_missing_words = vec![0u64; old_mosaic.haplotypes.n_haplotypes.div_ceil(64)];
        for marker in 0..merged_window.rows.len() {
            compact_mosaic
                .pack_marker_words(
                    &packed_direct,
                    marker,
                    &mut mosaic_allele_words,
                    &mut mosaic_missing_words,
                )
                .unwrap();
            for state in 0..old_mosaic.haplotypes.n_haplotypes {
                let mask = 1u64 << (state & 63);
                let word = state >> 6;
                let expected = old_mosaic.haplotypes.allele(marker, state);
                assert_eq!(
                    mosaic_allele_words[word] & mask != 0,
                    expected == Some(1),
                    "allele marker={marker} state={state}"
                );
                assert_eq!(
                    mosaic_missing_words[word] & mask != 0,
                    expected.is_none(),
                    "missing marker={marker} state={state}"
                );
            }
        }
        let probabilities = (0..12)
            .map(|state| ((state * 13) % 29 + 1) as f32)
            .collect::<Vec<_>>();
        let mut left = (0..12)
            .map(|state| ((state * 11) % 31 + 1) as f32)
            .collect::<Vec<_>>();
        let mut right = (0..12)
            .map(|state| ((state * 17) % 37 + 1) as f32)
            .collect::<Vec<_>>();
        let left_sum = left.iter().sum::<f32>();
        let right_sum = right.iter().sum::<f32>();
        left.iter_mut().for_each(|value| *value /= left_sum);
        right.iter_mut().for_each(|value| *value /= right_sum);
        let alpha = 0.37f32;
        for marker in 0..merged_window.rows.len() {
            let donors = compact_mosaic.donors_at(marker).unwrap();
            for (state, &donor) in donors.iter().enumerate() {
                assert_eq!(
                    packed_direct.allele_unchecked(marker, donor),
                    old_mosaic.haplotypes.allele(marker, state),
                    "marker={marker} state={state}"
                );
            }
            let mut observed_alleles = 0usize;
            let mut observed_alt = 0usize;
            let mut observed_probability = 0.0f32;
            let mut missing_probability = 0.0f32;
            for (state, &probability) in probabilities.iter().enumerate() {
                match old_mosaic.haplotypes.allele(marker, state) {
                    Some(allele) => {
                        observed_alleles += 1;
                        if allele == 1 {
                            observed_alt += 1;
                            observed_probability += probability;
                        }
                    }
                    None => missing_probability += probability,
                }
            }
            let fallback = if observed_alleles == 0 {
                0.5
            } else {
                observed_alt as f32 / observed_alleles as f32
            };
            let expected = (observed_probability + missing_probability * fallback).clamp(0.0, 1.0);
            let observed =
                compact_mosaic.allele_probability(&packed_direct, marker, &probabilities, 1);
            assert_eq!(observed.to_bits(), expected.to_bits(), "marker={marker}");

            let mut left_observed = 0.0f32;
            let mut right_observed = 0.0f32;
            let mut left_missing = 0.0f32;
            let mut right_missing = 0.0f32;
            for (state, (&left_probability, &right_probability)) in
                left.iter().zip(&right).enumerate()
            {
                match old_mosaic.haplotypes.allele(marker, state) {
                    Some(1) => {
                        left_observed += left_probability;
                        right_observed += right_probability;
                    }
                    Some(_) => {}
                    None => {
                        left_missing += left_probability;
                        right_missing += right_probability;
                    }
                }
            }
            let fallback = if old_mosaic.haplotypes.n_haplotypes == 0 {
                0.5
            } else {
                let observed = (0..old_mosaic.haplotypes.n_haplotypes)
                    .filter(|&state| old_mosaic.haplotypes.allele(marker, state).is_some())
                    .count();
                if observed == 0 {
                    0.5
                } else {
                    (0..old_mosaic.haplotypes.n_haplotypes)
                        .filter(|&state| old_mosaic.haplotypes.allele(marker, state) == Some(1))
                        .count() as f32
                        / observed as f32
                }
            };
            let left_probability = left_observed + left_missing * fallback;
            let right_probability = right_observed + right_missing * fallback;
            let expected_interpolated =
                (left_probability * (1.0 - alpha) + right_probability * alpha).clamp(0.0, 1.0);
            let observed_interpolated = compact_mosaic.interpolated_allele_probability(
                &packed_direct,
                marker,
                &left,
                &right,
                alpha,
                1,
            );
            assert_eq!(
                observed_interpolated.to_bits(),
                expected_interpolated.to_bits(),
                "interpolated marker={marker}"
            );
        }
    }

    #[test]
    fn packed_mosaic_word_gather_handles_full_words_tails_and_missing_rows() {
        let sample_count = 64;
        let calls = (0..sample_count)
            .map(|sample| {
                [
                    Some(u16::from((sample * 7 + 3) % 11 < 5)),
                    Some(u16::from((sample * 13 + 2) % 17 < 8)),
                ]
            })
            .collect::<Vec<_>>();
        let row = PhasedRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: 1,
                id: "v1".to_owned(),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: row_calls(&calls),
            dosage: None,
            ap1: None,
            ap2: None,
            gp: None,
            info_af: None,
            info_dr2: None,
            imputed: false,
        };
        let reference = PackedBiallelicReference::from_reference_rows(
            std::slice::from_ref(&row),
            &[Some(0)],
            &[false],
            sample_count,
        )
        .unwrap();
        let donors = (0..70).map(|state| (state * 37) % 128).collect::<Vec<_>>();
        let mut allele_words = vec![u64::MAX; donors.len().div_ceil(64)];
        let mut missing_words = vec![u64::MAX; donors.len().div_ceil(64)];

        reference
            .mosaic_marker_words_at(0, &donors, &mut allele_words, &mut missing_words)
            .unwrap();
        assert_eq!(missing_words, vec![0, 0]);
        for (state, &donor) in donors.iter().enumerate() {
            let expected = row.calls.get_allele_unchecked(donor).unwrap() == 1;
            assert_eq!(
                allele_words[state >> 6] & (1u64 << (state & 63)) != 0,
                expected,
                "state={state} donor={donor}"
            );
        }
        assert_eq!(allele_words[1] >> 6, 0);

        let missing_reference = PackedBiallelicReference::from_reference_rows(
            std::slice::from_ref(&row),
            &[None],
            &[false],
            sample_count,
        )
        .unwrap();
        missing_reference
            .mosaic_marker_words_at(0, &donors, &mut allele_words, &mut missing_words)
            .unwrap();
        assert_eq!(allele_words, vec![0, 0]);
        assert_eq!(missing_words, vec![u64::MAX, 0b11_1111]);
    }

    #[test]
    fn complete_mosaic_word_fast_path_matches_general_gather_and_preserves_missing_words() {
        let sample_count = 64;
        let calls = (0..sample_count)
            .map(|sample| {
                [
                    Some(u16::from((sample * 7 + 3) % 11 < 5)),
                    Some(u16::from((sample * 13 + 2) % 17 < 8)),
                ]
            })
            .collect::<Vec<_>>();
        let row = PhasedRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: 1,
                id: "v1".to_owned(),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: row_calls(&calls),
            dosage: None,
            ap1: None,
            ap2: None,
            gp: None,
            info_af: None,
            info_dr2: None,
            imputed: false,
        };
        let reference = PackedBiallelicReference::from_reference_rows(
            std::slice::from_ref(&row),
            &[Some(0)],
            &[false],
            sample_count,
        )
        .unwrap();
        let donors = (0..70).map(|state| (state * 37) % 128).collect::<Vec<_>>();
        let mut expected_alleles = vec![0u64; donors.len().div_ceil(64)];
        let mut expected_missing = vec![0u64; donors.len().div_ceil(64)];
        reference
            .mosaic_marker_words_at(0, &donors, &mut expected_alleles, &mut expected_missing)
            .unwrap();
        assert!(reference.complete_marker_words(0).is_some());

        let mosaic = PackedTargetMosaic {
            panel: CompositePanel {
                haplotypes: PackedHaplotypes::empty(donors.len(), 1, 1).unwrap(),
                source_samples: vec![0; donors.len()],
                marker_indices: vec![0],
                allele_counts: vec![2],
            },
            segments: vec![PackedTargetMosaicSegment {
                start: 0,
                end: 1,
                donors: donors.clone(),
            }],
        };
        let mut observed_alleles = vec![u64::MAX; donors.len().div_ceil(64)];
        assert!(mosaic
            .pack_marker_allele_words(&reference, 0, &mut observed_alleles)
            .unwrap());
        assert_eq!(observed_alleles, expected_alleles);

        let missing_reference = PackedBiallelicReference::from_reference_rows(
            std::slice::from_ref(&row),
            &[None],
            &[false],
            sample_count,
        )
        .unwrap();
        assert!(missing_reference.complete_marker_words(0).is_none());
        let mut untouched_words = vec![u64::MAX; donors.len().div_ceil(64)];
        assert!(!mosaic
            .pack_marker_allele_words(&missing_reference, 0, &mut untouched_words)
            .unwrap());
        assert_eq!(untouched_words, vec![u64::MAX; donors.len().div_ceil(64)]);
    }

    #[test]
    fn donor_run_returns_segment_end_across_marker_boundaries() {
        let first = vec![0, 1, 2, 3];
        let second = vec![4, 5, 6, 7];
        let mosaic = PackedTargetMosaic {
            panel: CompositePanel {
                haplotypes: PackedHaplotypes::empty(4, 1, 1).unwrap(),
                source_samples: vec![0; 4],
                marker_indices: vec![0],
                allele_counts: vec![2],
            },
            segments: vec![
                PackedTargetMosaicSegment {
                    start: 0,
                    end: 2,
                    donors: first.clone(),
                },
                PackedTargetMosaicSegment {
                    start: 2,
                    end: 4,
                    donors: second.clone(),
                },
            ],
        };

        assert_eq!(mosaic.donor_run_at(0), Some((2, first.as_slice())));
        assert_eq!(mosaic.donor_run_at(1), Some((2, first.as_slice())));
        assert_eq!(mosaic.donor_run_at(2), Some((4, second.as_slice())));
        assert_eq!(mosaic.donor_run_at(4), None);
    }

    #[test]
    fn hmm_mosaic_keeps_shared_donors_in_the_same_state_across_segments() {
        let calls = [
            [
                [Some(0), Some(0)],
                [Some(0), Some(0)],
                [Some(1), Some(1)],
                [Some(1), Some(1)],
            ],
            [
                [Some(0), Some(0)],
                [Some(0), Some(0)],
                [Some(1), Some(1)],
                [Some(1), Some(1)],
            ],
            [
                [Some(0), Some(1)],
                [Some(0), Some(1)],
                [Some(0), Some(0)],
                [Some(1), Some(1)],
            ],
            [
                [Some(0), Some(1)],
                [Some(0), Some(1)],
                [Some(0), Some(0)],
                [Some(1), Some(1)],
            ],
        ];
        let reference_rows = calls
            .iter()
            .enumerate()
            .map(|(marker, calls)| PhasedRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{marker}"),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: row_calls(calls),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
            .collect::<Vec<_>>();
        let packed_reference = PackedBiallelicReference::from_reference_rows(
            &reference_rows,
            &[Some(0), Some(1), Some(2), Some(3)],
            &[false; 4],
            4,
        )
        .unwrap();
        let cm_positions = [0.0, 0.0, 1.0, 1.0];
        let pbwt_index = pbwt::build_segment_index_from_alleles(
            packed_reference.n_haplotypes(),
            packed_reference.n_markers(),
            1,
            &cm_positions,
            &[true; 4],
            1.0,
            PbwtDirection::Forward,
            |marker, haplotype| packed_reference.allele_unchecked(marker, haplotype),
        )
        .unwrap();
        let mosaic = build_pbwt_target_mosaic_hmm_packed(
            &packed_reference,
            &pbwt_index,
            &[Some(0); 4],
            &[0, 1, 2, 3],
            &cm_positions,
            1.0,
            4,
        )
        .unwrap();

        assert_eq!(mosaic.donors_at(0), Some(&[0, 1, 2, 3][..]));
        assert_eq!(mosaic.donors_at(2), Some(&[0, 4, 2, 5][..]));
        assert_eq!(
            mosaic.donors_at(0).unwrap()[2],
            mosaic.donors_at(2).unwrap()[2]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_donor_gather_matches_scalar_for_vector_tails_and_partial_words() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let reference_words = [
            0xa55a_00ff_3cc3_6996,
            0x1337_cafe_f00d_beef,
            0xdead_beef_8765_4321,
            0x0101_1010_aaaa_5555,
            0xffff_0000_aaaa_5555,
        ];
        for state_count in [1, 3, 4, 5, 63, 64, 65, 127, 130] {
            let donors = (0..state_count)
                .map(|state| (state * 197 + 37) % (reference_words.len() * 64))
                .collect::<Vec<_>>();
            let mut observed = vec![u64::MAX; state_count.div_ceil(64)];
            unsafe {
                pack_donor_alleles_avx2(&reference_words, &donors, &mut observed);
            }

            let mut expected = vec![0u64; state_count.div_ceil(64)];
            for (state, &donor) in donors.iter().enumerate() {
                let allele = (reference_words[donor >> 6] >> (donor & 63)) & 1;
                expected[state >> 6] |= allele << (state & 63);
            }
            assert_eq!(observed, expected, "state_count={state_count}");
        }
    }

    #[test]
    fn phase_composite_may_keep_the_target_samples_other_haplotype() {
        let segments = select_donor_segments(&[(0, 1)], &[[Some(1), None]], 2, 0, 2, 1).unwrap();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].len(), 1);
        assert_eq!(segments[0][0].donor_haplotype, 1);
        assert_eq!(segments[0][0].start_marker, 0);
        assert_eq!(segments[0][0].end_marker, 1);
    }

    #[test]
    fn joint_phase_mosaic_marker_rows_match_scalar_segment_alleles() {
        let marker_calls = [
            [[Some(0), Some(1)], [Some(1), Some(0)], [None, Some(1)]],
            [[Some(1), Some(1)], [Some(0), Some(1)], [Some(0), None]],
            [[Some(0), None], [Some(1), Some(0)], [Some(1), Some(1)]],
            [[Some(1), Some(0)], [None, Some(1)], [Some(0), Some(0)]],
        ];
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: marker_calls
                .iter()
                .enumerate()
                .map(|(marker, calls)| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("p{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: row_calls(calls),
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
            emit_end: marker_calls.len(),
        };
        let state_segments = vec![
            vec![
                DonorHapSegment {
                    start_marker: 0,
                    end_marker: 2,
                    donor_haplotype: 0,
                },
                DonorHapSegment {
                    start_marker: 2,
                    end_marker: 4,
                    donor_haplotype: 5,
                },
            ],
            vec![
                DonorHapSegment {
                    start_marker: 0,
                    end_marker: 1,
                    donor_haplotype: 2,
                },
                DonorHapSegment {
                    start_marker: 1,
                    end_marker: 4,
                    donor_haplotype: 3,
                },
            ],
            vec![
                DonorHapSegment {
                    start_marker: 0,
                    end_marker: 3,
                    donor_haplotype: 4,
                },
                DonorHapSegment {
                    start_marker: 3,
                    end_marker: 4,
                    donor_haplotype: 1,
                },
            ],
        ];
        let mut observed = PackedHaplotypes::empty(3, marker_calls.len(), 1).unwrap();
        fill_joint_phase_mosaic(&window, &state_segments, &mut observed).unwrap();

        let mut expected = PackedHaplotypes::empty(3, marker_calls.len(), 1).unwrap();
        for (state, segments) in state_segments.iter().enumerate() {
            for segment in segments {
                for marker in segment.start_marker..segment.end_marker {
                    let donor = segment.donor_haplotype;
                    let allele =
                        window.rows[marker].calls.get(donor / 2).unwrap().alleles[donor & 1];
                    expected.set_allele(marker, state, allele).unwrap();
                }
            }
        }
        assert_eq!(observed.allele_bits, expected.allele_bits);
        assert_eq!(observed.missing_bits, expected.missing_bits);
    }

    #[test]
    fn complete_packed_joint_phase_mosaic_skips_missing_plane_copy() {
        let marker_calls = [
            [[Some(0), Some(1)], [Some(1), Some(0)], [Some(1), Some(1)]],
            [[Some(1), Some(1)], [Some(0), Some(1)], [Some(0), Some(0)]],
            [[Some(0), Some(0)], [Some(1), Some(0)], [Some(1), Some(1)]],
            [[Some(1), Some(0)], [Some(1), Some(1)], [Some(0), Some(0)]],
        ];
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: marker_calls
                .iter()
                .enumerate()
                .map(|(marker, calls)| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("p{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: row_calls(calls),
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
            emit_end: marker_calls.len(),
        };
        let state_segments = vec![
            vec![DonorHapSegment {
                start_marker: 0,
                end_marker: 4,
                donor_haplotype: 0,
            }],
            vec![
                DonorHapSegment {
                    start_marker: 0,
                    end_marker: 2,
                    donor_haplotype: 3,
                },
                DonorHapSegment {
                    start_marker: 2,
                    end_marker: 4,
                    donor_haplotype: 5,
                },
            ],
        ];
        let packed = PackedHaplotypes::complete_biallelic_state_major(&window)
            .unwrap()
            .unwrap();
        let mut observed = PackedHaplotypes::empty_state_major(2, 4, 1).unwrap();
        observed.missing_bits.fill(u64::MAX);
        fill_joint_phase_mosaic_from_packed(&window, &packed, &state_segments, &mut observed)
            .unwrap();

        let mut expected = PackedHaplotypes::empty_state_major(2, 4, 1).unwrap();
        expected.missing_bits.fill(0);
        for (state, segments) in state_segments.iter().enumerate() {
            for segment in segments {
                for marker in segment.start_marker..segment.end_marker {
                    let donor = segment.donor_haplotype;
                    let allele =
                        window.rows[marker].calls.get(donor / 2).unwrap().alleles[donor & 1];
                    expected.set_allele(marker, state, allele).unwrap();
                }
            }
        }
        assert_eq!(observed.allele_bits, expected.allele_bits);
        assert_eq!(observed.missing_bits, expected.missing_bits);
        assert!(observed.is_complete());
    }

    #[test]
    fn unpacked_complete_target_mosaic_keeps_alleles_observed() {
        let marker_calls = [
            [[Some(0), Some(1)], [Some(1), Some(1)]],
            [[Some(1), Some(1)], [Some(0), Some(1)]],
            [[Some(0), Some(0)], [Some(1), Some(0)]],
            [[Some(1), Some(0)], [Some(1), Some(1)]],
        ];
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: marker_calls
                .iter()
                .enumerate()
                .map(|(marker, calls)| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("p{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: row_calls(calls),
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
            emit_end: marker_calls.len(),
        };
        let view = PhaseHaplotypeView::new(&window, None, None, 2).unwrap();
        let state_segments = vec![vec![DonorHapSegment {
            start_marker: 0,
            end_marker: marker_calls.len(),
            donor_haplotype: 0,
        }]];
        let mut observed = PackedHaplotypes::empty_state_major(1, marker_calls.len(), 1).unwrap();
        assert!(observed.missing_bits.iter().all(|&word| word != 0));
        fill_joint_phase_mosaic_view(&view, &state_segments, &mut observed).unwrap();

        assert!(observed.is_complete());
        assert!(observed.missing_bits.iter().all(|&word| word == 0));
        for marker in 0..marker_calls.len() {
            assert_eq!(
                observed.allele(marker, 0),
                window.rows[marker].calls.get_allele(0, 0)
            );
        }
    }

    #[test]
    fn unpacked_incomplete_target_mosaic_copies_only_donor_missingness() {
        let marker_calls = [
            [[Some(0), Some(1)], [Some(1), Some(1)]],
            [[None, Some(1)], [Some(0), Some(1)]],
            [[Some(0), Some(0)], [Some(1), Some(0)]],
        ];
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: marker_calls
                .iter()
                .enumerate()
                .map(|(marker, calls)| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("p{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: row_calls(calls),
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
            emit_end: marker_calls.len(),
        };
        let view = PhaseHaplotypeView::new(&window, None, None, 2).unwrap();
        let state_segments = vec![vec![DonorHapSegment {
            start_marker: 0,
            end_marker: marker_calls.len(),
            donor_haplotype: 0,
        }]];
        let mut observed = PackedHaplotypes::empty_state_major(1, marker_calls.len(), 1).unwrap();
        fill_joint_phase_mosaic_view(&view, &state_segments, &mut observed).unwrap();

        assert!(!observed.is_complete());
        assert_eq!(observed.allele(0, 0), Some(0));
        assert_eq!(observed.allele(1, 0), None);
        assert_eq!(observed.allele(2, 0), Some(0));
    }

    fn row_calls(calls: &[[Option<u16>; 2]]) -> DiploidCalls {
        DiploidCalls::from_calls(
            calls
                .iter()
                .map(|&alleles| DiploidCall {
                    alleles,
                    phased: true,
                })
                .collect(),
        )
    }

    #[test]
    fn low_frequency_reference_preserves_donor_haplotype_alleles_for_missing_target() {
        let meta = VariantMeta {
            chrom: "1".to_owned(),
            pos: 1,
            id: "rare".to_owned(),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: meta.clone(),
                calls: row_calls(&[[None, None], [Some(0), Some(1)], [Some(0), Some(0)]]),
            }],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta,
                calls: row_calls(&[[Some(0), Some(0)], [Some(0), Some(1)], [Some(0), Some(0)]]),
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
        let pbwt_index = pbwt::build_index(&snapshot).unwrap();
        let panel = build_phase_reference(&snapshot, &input, 0, 4, &[], Some(&[true]), &pbwt_index)
            .unwrap();

        let alleles = (0..panel.haplotypes.n_haplotypes)
            .filter_map(|state| panel.haplotypes.allele(0, state))
            .collect::<Vec<_>>();
        assert_eq!(alleles.iter().filter(|&&allele| allele == 1).count(), 1);
        assert_eq!(alleles.iter().filter(|&&allele| allele == 0).count(), 3);
    }

    #[test]
    fn complete_reference_pair_fast_path_matches_general_allele_view() {
        let target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: (0..3)
                .map(|marker| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: row_calls(&[[Some(0), Some(1)]]),
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
            emit_end: 3,
        };
        let target_packed = PackedHaplotypes::complete_biallelic_state_major(&target)
            .unwrap()
            .unwrap();
        let mut reference_packed = PackedHaplotypes::empty_state_major(2, 3, 1).unwrap();
        for marker in 0..3 {
            for haplotype in 0..2 {
                reference_packed
                    .set_allele(
                        marker,
                        haplotype,
                        Some(((marker * 5 + haplotype) & 1) as u16),
                    )
                    .unwrap();
            }
        }
        reference_packed.mark_complete();
        let view =
            PhaseHaplotypeView::new(&target, Some(&target_packed), Some(&reference_packed), 1)
                .unwrap();
        assert!(view.has_complete_state_major_reference());
        for marker in 0..3 {
            let fast = unsafe { view.complete_reference_biallelic_pair_unchecked(marker, 1, 0) };
            assert_eq!(Some(fast[0]), view.allele(marker, 1, 0));
            assert_eq!(Some(fast[1]), view.allele(marker, 1, 1));
        }

        let mut incomplete_reference = PackedHaplotypes::empty_state_major(2, 3, 1).unwrap();
        for marker in 0..3 {
            for haplotype in 0..2 {
                incomplete_reference
                    .set_allele(
                        marker,
                        haplotype,
                        (marker != 1).then_some(((marker + haplotype) & 1) as u16),
                    )
                    .unwrap();
            }
        }
        let incomplete_view = PhaseHaplotypeView::new(
            &target,
            Some(&target_packed),
            Some(&incomplete_reference),
            1,
        )
        .unwrap();
        assert!(!incomplete_view.has_complete_state_major_reference());

        let mut unmarked_reference = PackedHaplotypes::empty_state_major(2, 3, 1).unwrap();
        for marker in 0..3 {
            for haplotype in 0..2 {
                unmarked_reference
                    .set_allele(marker, haplotype, Some(((marker + haplotype) & 1) as u16))
                    .unwrap();
            }
        }
        assert!(unmarked_reference
            .missing_bits
            .iter()
            .all(|&word| word == 0));
        assert!(!unmarked_reference.is_complete());
        let unmarked_view =
            PhaseHaplotypeView::new(&target, Some(&target_packed), Some(&unmarked_reference), 1)
                .unwrap();
        assert!(!unmarked_view.has_complete_state_major_reference());

        let multiallelic_reference = PackedHaplotypes::empty_state_major(2, 3, 2).unwrap();
        let multiallelic_view = PhaseHaplotypeView::new(
            &target,
            Some(&target_packed),
            Some(&multiallelic_reference),
            1,
        )
        .unwrap();
        assert!(!multiallelic_view.has_complete_state_major_reference());
    }

    #[test]
    fn packed_low_frequency_panel_fill_matches_scalar_projection() {
        let marker_count = 137usize;
        let target_samples = 3usize;
        let reference_samples = 2usize;
        let target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: (0..marker_count)
                .map(|marker| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(
                        (0..target_samples)
                            .map(|sample| DiploidCall {
                                alleles: [
                                    Some(((marker + sample * 3) % 5 < 2) as u16),
                                    Some(((marker * 7 + sample) % 9 < 4) as u16),
                                ],
                                phased: true,
                            })
                            .collect(),
                    ),
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
            emit_end: marker_count,
        };
        let target_packed = PackedHaplotypes::complete_biallelic_state_major(&target)
            .unwrap()
            .unwrap();
        let mut reference_packed =
            PackedHaplotypes::empty_state_major(reference_samples * 2, marker_count, 1).unwrap();
        for marker in 0..marker_count {
            for state in 0..reference_samples * 2 {
                reference_packed
                    .set_allele(
                        marker,
                        state,
                        Some(((marker * 11 + state * 5) % 13 < 6) as u16),
                    )
                    .unwrap();
            }
        }
        reference_packed.mark_complete();
        let view = PhaseHaplotypeView::new(
            &target,
            Some(&target_packed),
            Some(&reference_packed),
            target_samples,
        )
        .unwrap();
        let marker_indices = (0..marker_count)
            .filter(|marker| marker % 4 != 1)
            .collect::<Vec<_>>();
        let stage1_target = PhasedWindow {
            chrom: target.chrom.clone(),
            index: target.index,
            rows: marker_indices
                .iter()
                .map(|&marker| target.rows[marker].clone())
                .collect(),
            emit_start: 0,
            emit_end: marker_indices.len(),
        };
        let stage1_target_packed = PackedHaplotypes::complete_biallelic_state_major(&stage1_target)
            .unwrap()
            .unwrap();
        let stage1_reference_packed = reference_packed
            .project_state_major_biallelic(&marker_indices)
            .unwrap();
        let stage1_view = PhaseHaplotypeView::new(
            &stage1_target,
            Some(&stage1_target_packed),
            Some(&stage1_reference_packed),
            target_samples,
        )
        .unwrap();
        let state_segments = vec![
            vec![
                DonorHapSegment {
                    start_marker: 0,
                    end_marker: 63,
                    donor_haplotype: 0,
                },
                DonorHapSegment {
                    start_marker: 63,
                    end_marker: marker_count,
                    donor_haplotype: 6,
                },
            ],
            vec![DonorHapSegment {
                start_marker: 0,
                end_marker: marker_count,
                donor_haplotype: 5,
            }],
        ];
        let mut scalar = PackedHaplotypes::empty_state_major(2, marker_indices.len(), 1).unwrap();
        let mut projected =
            PackedHaplotypes::empty_state_major(2, marker_indices.len(), 1).unwrap();
        let scalar_sources = fill_low_frequency_phase_mosaic_haplotypes(
            &view,
            None,
            &state_segments,
            &marker_indices,
            &mut scalar,
        )
        .unwrap();
        let projected_sources = fill_low_frequency_phase_mosaic_haplotypes(
            &view,
            Some(&stage1_view),
            &state_segments,
            &marker_indices,
            &mut projected,
        )
        .unwrap();

        assert_eq!(scalar_sources, projected_sources);
        assert_eq!(scalar.allele_bits, projected.allele_bits);
        assert_eq!(scalar.missing_bits, projected.missing_bits);
        assert_eq!(scalar.is_complete(), projected.is_complete());
        for marker in 0..marker_indices.len() {
            for state in 0..2 {
                assert_eq!(
                    scalar.allele(marker, state),
                    projected.allele(marker, state)
                );
            }
        }
    }

    #[test]
    fn packed_low_frequency_panel_covers_state_word_boundaries() {
        for state_count in [1usize, 63, 64, 65, 127, 129, 2_000] {
            let marker_count = 67usize;
            let marker_indices = (0..marker_count).collect::<Vec<_>>();
            let target_samples = state_count.div_ceil(2).max(1);
            let target_haplotypes = target_samples * 2;
            let state_segments = (0..state_count)
                .map(|state| {
                    vec![DonorHapSegment {
                        start_marker: 0,
                        end_marker: marker_count,
                        donor_haplotype: state % target_haplotypes,
                    }]
                })
                .collect::<Vec<_>>();
            let panel = assert_packed_fill_matches_scalar(
                marker_count,
                &marker_indices,
                target_samples,
                0,
                &state_segments,
            );
            assert_eq!(panel.n_haplotypes, state_count);
            assert_eq!(panel.n_markers, marker_count);
        }
    }

    #[test]
    fn packed_low_frequency_panel_uses_right_open_segment_boundaries() {
        let marker_count = 32usize;
        let marker_indices = vec![5usize, 7, 10, 11, 12, 20, 21];
        let target_samples = 4usize;
        let state_segments = vec![vec![
            DonorHapSegment {
                start_marker: 0,
                end_marker: 3,
                donor_haplotype: 0,
            },
            DonorHapSegment {
                start_marker: 3,
                end_marker: 5,
                donor_haplotype: 1,
            },
            DonorHapSegment {
                start_marker: 5,
                end_marker: 7,
                donor_haplotype: 3,
            },
            DonorHapSegment {
                start_marker: 7,
                end_marker: 10,
                donor_haplotype: 4,
            },
            DonorHapSegment {
                start_marker: 10,
                end_marker: 11,
                donor_haplotype: 6,
            },
            DonorHapSegment {
                start_marker: 11,
                end_marker: 12,
                donor_haplotype: 8,
            },
            DonorHapSegment {
                start_marker: 12,
                end_marker: 21,
                donor_haplotype: 10,
            },
            DonorHapSegment {
                start_marker: 21,
                end_marker: 22,
                donor_haplotype: 11,
            },
            DonorHapSegment {
                start_marker: 22,
                end_marker: marker_count,
                donor_haplotype: 0,
            },
        ]];
        let panel = assert_packed_fill_matches_scalar(
            marker_count,
            &marker_indices,
            target_samples,
            2,
            &state_segments,
        );
        let target_haplotypes = target_samples * 2;
        let boundary_donors = [
            (0usize, 3usize, 0usize),
            (3, 5, 1),
            (5, 7, 3),
            (7, 10, 4),
            (10, 11, 6),
            (11, 12, 8),
            (12, 21, 10),
            (21, 22, 11),
            (22, marker_count, 0),
        ];
        for (projected_marker, &source_marker) in marker_indices.iter().enumerate() {
            let donor = boundary_donors
                .iter()
                .find_map(|&(start, end, donor)| {
                    (source_marker >= start && source_marker < end).then_some(donor)
                })
                .unwrap();
            let expected = if donor < target_haplotypes {
                target_panel_allele(source_marker, donor)
            } else {
                reference_panel_allele(source_marker, donor - target_haplotypes)
            };
            assert_eq!(
                panel.allele(projected_marker, 0),
                Some(expected),
                "marker {source_marker} must use donor {donor}"
            );
        }
    }

    #[test]
    fn packed_low_frequency_panel_supports_a_single_stage1_marker() {
        let marker_count = 5usize;
        let marker_indices = vec![2usize];
        let state_segments = vec![
            vec![DonorHapSegment {
                start_marker: 0,
                end_marker: marker_count,
                donor_haplotype: 0,
            }],
            vec![DonorHapSegment {
                start_marker: 0,
                end_marker: marker_count,
                donor_haplotype: 5,
            }],
        ];
        let panel =
            assert_packed_fill_matches_scalar(marker_count, &marker_indices, 2, 1, &state_segments);
        assert_eq!(panel.n_markers, 1);
        assert_eq!(panel.allele(0, 0), Some(target_panel_allele(2, 0)));
        assert_eq!(panel.allele(0, 1), Some(reference_panel_allele(2, 1)));
    }

    #[test]
    fn packed_low_frequency_panel_rejects_incomplete_or_multiallelic_sources() {
        let marker_count = 3usize;
        let target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: (0..marker_count)
                .map(|marker| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [
                            Some(target_panel_allele(marker, 0)),
                            Some(target_panel_allele(marker, 1)),
                        ],
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
                .collect(),
            emit_start: 0,
            emit_end: marker_count,
        };
        let complete = PackedHaplotypes::complete_biallelic_state_major(&target)
            .unwrap()
            .unwrap();
        let complete_view = PhaseHaplotypeView::new(&target, Some(&complete), None, 1).unwrap();
        assert!(complete.missing_bits.iter().all(|&word| word == 0));
        assert!(stage2_panel_packed_view_eligible(
            &complete_view,
            marker_count
        ));
        assert!(!stage2_panel_packed_view_eligible(
            &complete_view,
            marker_count - 1
        ));

        let all_missing = PackedHaplotypes::empty_state_major(2, marker_count, 1).unwrap();
        let all_missing_view =
            PhaseHaplotypeView::new(&target, Some(&all_missing), None, 1).unwrap();
        assert!(all_missing.missing_bits.iter().all(|&word| word != 0));
        assert!(!stage2_panel_packed_view_eligible(
            &all_missing_view,
            marker_count
        ));

        let mut partial_missing = PackedHaplotypes::empty_state_major(2, marker_count, 1).unwrap();
        for marker in 0..marker_count {
            for haplotype in 0..2 {
                partial_missing
                    .set_allele(
                        marker,
                        haplotype,
                        if marker == 1 && haplotype == 0 {
                            None
                        } else {
                            Some(target_panel_allele(marker, haplotype))
                        },
                    )
                    .unwrap();
            }
        }
        let partial_view =
            PhaseHaplotypeView::new(&target, Some(&partial_missing), None, 1).unwrap();
        assert!(partial_missing.missing_bits.iter().any(|&word| word != 0));
        assert!(partial_missing
            .missing_bits
            .iter()
            .any(|&word| word != u64::MAX));
        assert!(!stage2_panel_packed_view_eligible(
            &partial_view,
            marker_count
        ));

        let mut incomplete_reference =
            PackedHaplotypes::empty_state_major(2, marker_count, 1).unwrap();
        for marker in 0..marker_count {
            for haplotype in 0..2 {
                incomplete_reference
                    .set_allele(
                        marker,
                        haplotype,
                        Some(reference_panel_allele(marker, haplotype)),
                    )
                    .unwrap();
            }
        }
        let incomplete_reference_view =
            PhaseHaplotypeView::new(&target, Some(&complete), Some(&incomplete_reference), 1)
                .unwrap();
        assert!(!stage2_panel_packed_view_eligible(
            &incomplete_reference_view,
            marker_count
        ));

        let mut multiallelic = PackedHaplotypes::empty_state_major(2, marker_count, 2).unwrap();
        for marker in 0..marker_count {
            for haplotype in 0..2 {
                multiallelic
                    .set_allele(marker, haplotype, Some(((marker + haplotype) & 1) as u16))
                    .unwrap();
            }
        }
        multiallelic.mark_complete();
        let multiallelic_view =
            PhaseHaplotypeView::new(&target, Some(&multiallelic), None, 1).unwrap();
        assert!(!stage2_panel_packed_view_eligible(
            &multiallelic_view,
            marker_count
        ));

        let wrong_reference_markers =
            PackedHaplotypes::empty_state_major(2, marker_count - 1, 1).unwrap();
        assert!(PhaseHaplotypeView::new(
            &target,
            Some(&complete),
            Some(&wrong_reference_markers),
            1,
        )
        .is_err());
    }

    #[test]
    fn low_frequency_phase_mosaic_keeps_target_hap_specific_donor_ids() {
        let markers = 3;
        let n_samples = 4;
        let haplotypes = [
            [0u16, 0, 0],
            [0u16, 0, 1],
            [0u16, 1, 0],
            [1u16, 1, 1],
            [0u16, 0, 0],
            [0u16, 0, 1],
            [1u16, 0, 1],
            [0u16, 1, 0],
        ];
        let rows = (0..markers)
            .map(|marker| {
                let calls = (0..n_samples)
                    .map(|sample| DiploidCall {
                        alleles: [
                            Some(haplotypes[2 * sample][marker]),
                            Some(haplotypes[2 * sample + 1][marker]),
                        ],
                        phased: true,
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
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: rows.clone(),
            cm_positions: vec![0.0, 0.01, 0.02],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: markers,
        };
        let snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: rows
                .into_iter()
                .map(|row| PhasedRow {
                    meta: row.meta,
                    calls: row.calls,
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
            emit_end: markers,
        };
        let marker_indices = [0usize, 1, 2];
        let pbwt_index = build_low_frequency_phase_donor_index(
            &snapshot,
            &input.cm_positions,
            &input,
            &marker_indices,
            0.01,
            3,
            10,
            30,
            &[],
            7,
        )
        .unwrap();

        let (panel, donor_haplotypes) =
            build_low_frequency_phase_mosaic(&snapshot, &pbwt_index, 0, 0, 4, 200, 11).unwrap();

        assert_eq!(donor_haplotypes.len(), panel.haplotypes.n_haplotypes);
        for marker in 0..markers {
            for state in 0..panel.haplotypes.n_haplotypes {
                let donor = donor_haplotype_at(&donor_haplotypes[state], marker).unwrap();
                assert_ne!(donor / 2, 0);
                let expected =
                    snapshot.rows[marker].calls.get(donor / 2).unwrap().alleles[donor & 1];
                assert_eq!(panel.haplotypes.allele(marker, state), expected);
            }
        }
    }

    #[test]
    fn donor_queue_ties_follow_beagle_priority_queue_insertion_order() {
        let segments = select_donor_segments(
            &[(0, 1), (1, 2), (2, 3)],
            &[[Some(4), Some(2)], [Some(6), None], [None, None]],
            5,
            0,
            2,
            1,
        )
        .unwrap();

        // Beagle's PriorityQueue does not move a newly inserted element above
        // an equal-priority parent. The first step inserts donor 4 before 2,
        // so donor 4 is the one evicted when donor 6 appears at step 1.
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].len(), 1);
        assert_eq!(segments[0][0].donor_haplotype, 6);
        assert_eq!(segments[0][0].start_marker, 0);
        assert_eq!(segments[0][0].end_marker, 3);
        assert_eq!(segments[1].len(), 1);
        assert_eq!(segments[1][0].donor_haplotype, 2);
        assert_eq!(segments[1][0].start_marker, 0);
        assert_eq!(segments[1][0].end_marker, 3);
    }

    #[test]
    fn ibs_composite_tracks_use_midpoint_replacement_and_allow_variable_state_count() {
        let steps = [(0, 1), (1, 2), (2, 3)];
        let candidates = [vec![0], vec![], vec![1]];

        let tracks = select_ibs_donor_segments(&steps, &candidates, 4, 1).unwrap();

        assert_eq!(tracks.len(), 1);
        assert_eq!(
            tracks[0],
            vec![
                DonorHapSegment {
                    start_marker: 0,
                    end_marker: 1,
                    donor_haplotype: 0,
                },
                DonorHapSegment {
                    start_marker: 1,
                    end_marker: 3,
                    donor_haplotype: 1,
                },
            ]
        );

        let stable_candidates = [vec![2], vec![2], vec![2]];
        let stable_tracks = select_ibs_donor_segments(&steps, &stable_candidates, 4, 4).unwrap();
        assert_eq!(stable_tracks.len(), 1);
        assert_eq!(stable_tracks[0].len(), 1);
        assert_eq!(stable_tracks[0][0].donor_haplotype, 2);
        assert_eq!(stable_tracks[0][0].end_marker, 3);
    }

    #[test]
    fn ibs_composite_returns_no_states_for_empty_candidate_sets() {
        let tracks = select_ibs_donor_segments(&[(0, 2), (2, 4)], &[vec![], vec![]], 4, 3).unwrap();

        assert!(tracks.is_empty());
    }

    #[test]
    fn ibs_composite_uses_beagle_queue_tie_order_for_multiple_candidates() {
        let tracks =
            select_ibs_donor_segments(&[(0, 1), (1, 2)], &[vec![2, 4], vec![6]], 8, 2).unwrap();

        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0][0].donor_haplotype, 6);
        assert_eq!(tracks[1][0].donor_haplotype, 4);
    }

    #[test]
    fn ibs_mosaic_projects_composite_donor_tracks_into_the_hmm_panel() {
        let rows = [
            [[Some(1), Some(0)], [Some(0), Some(1)]],
            [[Some(0), Some(1)], [Some(0), Some(0)]],
            [[Some(0), Some(0)], [Some(1), Some(1)]],
        ]
        .iter()
        .enumerate()
        .map(|(marker, calls)| PhasedRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: marker as u64 + 1,
                id: format!("v{}", marker + 1),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: row_calls(calls),
            dosage: None,
            ap1: None,
            ap2: None,
            gp: None,
            info_af: None,
            info_dr2: None,
            imputed: false,
        })
        .collect::<Vec<_>>();
        let reference = PackedBiallelicReference::from_reference_rows(
            &rows,
            &[Some(0), Some(1), Some(2)],
            &[false; 3],
            2,
        )
        .unwrap();

        let mosaic = build_ibs_target_mosaic_hmm_packed(
            &reference,
            &[0, 1, 2],
            &[(0, 1), (1, 2), (2, 3)],
            &[vec![0], vec![], vec![1]],
            1,
            5,
            42,
        )
        .unwrap();

        assert_eq!(mosaic.panel.haplotypes.n_haplotypes, 1);
        assert_eq!(
            (0..3)
                .map(|marker| mosaic.panel.haplotypes.allele(marker, 0))
                .collect::<Vec<_>>(),
            vec![Some(1), Some(1), Some(0)]
        );
        assert_eq!(mosaic.donors_at(0), Some(&[0][..]));
        assert_eq!(mosaic.donors_at(1), Some(&[1][..]));
    }

    #[test]
    fn ibs_mosaic_uses_seeded_fallback_when_all_candidate_sets_are_empty() {
        let rows = (0..2)
            .map(|marker| PhasedRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: row_calls(&[[Some(0), Some(1)], [Some(1), Some(0)]]),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
            .collect::<Vec<_>>();
        let reference = PackedBiallelicReference::from_reference_rows(
            &rows,
            &[Some(0), Some(1)],
            &[false; 2],
            2,
        )
        .unwrap();

        let mosaic =
            build_ibs_target_mosaic_hmm_packed(&reference, &[0, 1], &[(0, 2)], &[vec![]], 2, 0, 42)
                .unwrap();

        assert_eq!(mosaic.panel.haplotypes.n_haplotypes, 2);
        assert_eq!(mosaic.donors_at(0).unwrap().len(), 2);
        assert_eq!(mosaic.donors_at(1).unwrap().len(), 2);
    }

    #[test]
    fn ibs_mosaic_fallback_handles_a_single_reference_haplotype() {
        assert_eq!(ibs_fallback_donor_haplotypes(1, 4, 0, 42), vec![0]);
    }

    #[test]
    fn ibs_fallback_uses_seed_and_does_not_exclude_reference_local_indices() {
        let donors = ibs_fallback_donor_haplotypes(64, 64, 0, 42);
        let mut random = crate::imp::pbwt::JavaRandom::new(42);
        let expected = (0..64).map(|_| random.next_int(64)).collect::<Vec<_>>();
        assert_eq!(donors, expected);
        assert!(donors.contains(&0));
        assert_ne!(donors, ibs_fallback_donor_haplotypes(64, 64, 0, 43));
    }
}

pub(crate) fn build_panel_from_calls(
    rows: &[Vec<DiploidCall>],
    target_sample: usize,
    max_states: usize,
) -> Result<CompositePanel, String> {
    if rows.is_empty() || max_states == 0 {
        return Err("composite panel requires variants and a positive state count".to_owned());
    }
    let n_samples = rows[0].len();
    if target_sample >= n_samples || n_samples < 2 {
        return Err(
            "phase panel requires at least two samples and a valid target index".to_owned(),
        );
    }
    if rows.iter().any(|row| row.len() != n_samples) {
        return Err("phase rows have inconsistent sample counts".to_owned());
    }
    let candidate_count = n_samples
        .saturating_sub(1)
        .checked_mul(2)
        .ok_or_else(|| "phase candidate haplotype count overflow".to_owned())?;
    if candidate_count == 0 {
        return Err("phase composite panel is empty".to_owned());
    }
    let max_allele = rows
        .iter()
        .flat_map(|row| row.iter())
        .flat_map(|call| call.alleles.into_iter().flatten())
        .max()
        .unwrap_or(1);
    let allele_counts = rows
        .iter()
        .map(|row| {
            row.iter()
                .flat_map(|call| call.alleles.into_iter().flatten())
                .max()
                .map_or(1, |allele| usize::from(allele) + 1)
        })
        .collect::<Vec<_>>();
    let selected = systematic_indices(candidate_count, max_states);
    let mut haplotypes =
        PackedHaplotypes::empty(selected.len(), rows.len(), allele_width(max_allele))?;
    let mut source_samples = Vec::with_capacity(selected.len());
    for (state, &candidate) in selected.iter().enumerate() {
        let (sample, hap) = candidate_sample_hap(candidate, n_samples, Some(target_sample));
        source_samples.push(sample);
        for (marker, row) in rows.iter().enumerate() {
            haplotypes.set_allele(marker, state, row[sample].alleles[hap])?;
        }
    }
    Ok(CompositePanel {
        haplotypes,
        source_samples,
        marker_indices: (0..rows.len()).collect(),
        allele_counts,
    })
}

fn systematic_indices(candidate_count: usize, max_states: usize) -> Vec<usize> {
    if candidate_count <= max_states {
        return (0..candidate_count).collect();
    }
    if max_states == 1 {
        return vec![candidate_count / 2];
    }
    let last = candidate_count - 1;
    let denom = max_states - 1;
    (0..max_states)
        .map(|index| ((index as u128 * last as u128) / denom as u128) as usize)
        .collect()
}

fn candidate_sample_hap(
    candidate: usize,
    n_samples: usize,
    excluded_sample: Option<usize>,
) -> (usize, usize) {
    let sample_without_exclusion = candidate / 2;
    let sample = match excluded_sample {
        Some(excluded) if sample_without_exclusion >= excluded => sample_without_exclusion + 1,
        _ => sample_without_exclusion,
    };
    debug_assert!(sample < n_samples);
    (sample, candidate & 1)
}

fn allele_width(max_allele: u16) -> u8 {
    match max_allele {
        0..=1 => 1,
        2..=3 => 2,
        4..=15 => 4,
        16..=255 => 8,
        _ => 16,
    }
}
