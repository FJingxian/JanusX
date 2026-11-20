//! PBWT-neighbour initialization of window haplotypes.

use super::composite::PhaseHaplotypeView;
use super::types::{
    DiploidCall, DiploidCalls, PackedHaplotypes, PhaseWindow, PhasedRow, PhasedWindow,
};
use rayon::prelude::*;
use rayon::ThreadPool;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

static PBWT_SEARCH_THREAD_NANOS: AtomicU64 = AtomicU64::new(0);

fn pbwt_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JANUSX_IMP_PBWT_PROFILE").is_some())
}

/// Final prefix order plus its inverse for one phased window. The index is
/// built once per phase snapshot and shared by all sample workers.
#[derive(Debug, Clone)]
pub(crate) struct PbwtIndex {
    pub(crate) order: Vec<usize>,
    pub(crate) inverse: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentPbwtIndex {
    pub(crate) n_haplotypes: usize,
    pub(crate) direction: PbwtDirection,
    pub(crate) segments: Vec<PbwtSegment>,
}

/// One donor proposal per target haplotype and genetic step. Donor ids are
/// stored in 16 bits when the 65,535-haplotype capacity is sufficient and use
/// 32 bits otherwise.
#[derive(Debug, Clone)]
pub(crate) struct PhasePbwtIndex {
    pub(crate) n_haplotypes: usize,
    pub(crate) n_proposal_haplotypes: usize,
    pub(crate) step_ranges: Vec<(usize, usize)>,
    donor_haplotypes: DonorProposalStorage,
}

/// PBWT categories are assigned from zero in first-haplotype order. Cohorts
/// with at most 65,535 haplotypes can store every category in 16 bits while
/// retaining the 32-bit update kernel by expanding one row at a time.
enum PhaseStepCodes {
    Compact(Vec<u16>),
    Wide(Vec<u32>),
}

trait DonorProposalId: Copy + Send + Sync {
    fn from_haplotype(haplotype: usize) -> Self;
    fn haplotype(self) -> Option<usize>;
}

impl DonorProposalId for u16 {
    fn from_haplotype(haplotype: usize) -> Self {
        debug_assert!(haplotype < u16::MAX as usize);
        haplotype as u16
    }

    fn haplotype(self) -> Option<usize> {
        (self != u16::MAX).then_some(usize::from(self))
    }
}

impl DonorProposalId for i32 {
    fn from_haplotype(haplotype: usize) -> Self {
        i32::try_from(haplotype).expect("phase PBWT haplotype fits i32")
    }

    fn haplotype(self) -> Option<usize> {
        (self >= 0).then_some(self as usize)
    }
}

/// Donor haplotype ids use the same hybrid representation as phase step codes.
/// Widening a row is lossless and preserves all downstream PBWT arithmetic.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DonorProposalStorage {
    Compact(Vec<u16>),
    Wide(Vec<i32>),
}

impl DonorProposalStorage {
    fn new(count: usize, compact: bool) -> Self {
        if compact {
            Self::Compact(vec![u16::MAX; count])
        } else {
            Self::Wide(vec![-1; count])
        }
    }

    fn get(&self, index: usize) -> Option<usize> {
        match self {
            Self::Compact(values) => values.get(index).copied()?.haplotype(),
            Self::Wide(values) => values.get(index).copied()?.haplotype(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Compact(values) => values.len(),
            Self::Wide(values) => values.len(),
        }
    }

    fn raw_pair(&self, index: usize) -> [i32; 2] {
        match self {
            Self::Compact(values) => values
                .get(index..index + 2)
                .map(|pair| {
                    [
                        if pair[0] == u16::MAX {
                            -1
                        } else {
                            i32::from(pair[0])
                        },
                        if pair[1] == u16::MAX {
                            -1
                        } else {
                            i32::from(pair[1])
                        },
                    ]
                })
                .unwrap_or([-1, -1]),
            Self::Wide(values) => values
                .get(index..index + 2)
                .map(|pair| [pair[0], pair[1]])
                .unwrap_or([-1, -1]),
        }
    }
}

/// Beagle Stage2 IBS proposals. Donors are computed independently in both
/// PBWT directions for every target haplotype and Stage1 step; the arrays
/// retain only one 32-bit donor id per query and direction.
#[derive(Debug, Clone)]
pub(crate) struct LowFrequencyPhasePbwtIndex {
    pub(crate) n_haplotypes: usize,
    pub(crate) step_ranges: Vec<(usize, usize)>,
    forward_donors: Vec<i32>,
    backward_donors: Vec<i32>,
}

impl PhasePbwtIndex {
    #[inline]
    pub(crate) fn donor_haplotype(&self, step: usize, target_haplotype: usize) -> Option<usize> {
        if target_haplotype >= self.n_proposal_haplotypes {
            return None;
        }
        let index = step
            .checked_mul(self.n_proposal_haplotypes)?
            .checked_add(target_haplotype)?;
        self.donor_haplotypes.get(index)
    }

    #[inline(always)]
    pub(crate) fn donor_haplotype_pair(
        &self,
        step: usize,
        first_target_haplotype: usize,
    ) -> [Option<usize>; 2] {
        if first_target_haplotype + 1 >= self.n_proposal_haplotypes {
            return [None, None];
        }
        let Some(index) = step
            .checked_mul(self.n_proposal_haplotypes)
            .and_then(|index| index.checked_add(first_target_haplotype))
        else {
            return [None, None];
        };
        let [left, right] = self.donor_haplotypes.raw_pair(index);
        [
            (left >= 0).then_some(left as usize),
            (right >= 0).then_some(right as usize),
        ]
    }

    #[inline(always)]
    pub(crate) unsafe fn donor_haplotype_pair_raw_unchecked(
        &self,
        step: usize,
        first_target_haplotype: usize,
    ) -> [i32; 2] {
        debug_assert!(step < self.step_ranges.len());
        debug_assert!(first_target_haplotype + 1 < self.n_proposal_haplotypes);
        let index = step * self.n_proposal_haplotypes + first_target_haplotype;
        // SAFETY: callers validate the step and target haplotype dimensions.
        self.donor_haplotypes.raw_pair(index)
    }
}

impl LowFrequencyPhasePbwtIndex {
    pub(crate) fn donor_haplotype(
        &self,
        direction: PbwtDirection,
        step: usize,
        target_haplotype: usize,
    ) -> Option<usize> {
        let index = step
            .checked_mul(self.n_haplotypes)?
            .checked_add(target_haplotype)?;
        let proposals = match direction {
            PbwtDirection::Forward => &self.forward_donors,
            PbwtDirection::Reverse => &self.backward_donors,
        };
        let donor = *proposals.get(index)?;
        (donor >= 0).then_some(donor as usize)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PbwtDirection {
    Forward,
    Reverse,
}

const PACKED_STEP_SEQUENCE_WORDS: usize = 32;
const PACKED_STEP_SEQUENCE_MAX_MARKERS: usize = PACKED_STEP_SEQUENCE_WORDS * 32;
#[derive(Default)]
struct StepSequenceScratch {
    packed_biallelic: Vec<[u64; PACKED_STEP_SEQUENCE_WORDS]>,
    packed_sequence_table: Vec<(u64, u32)>,
    packed_sequence_generation: Vec<u32>,
    current_generation: u32,
    first_sequence_by_hash: HashMap<u64, (u32, usize)>,
    collisions: HashMap<u64, Vec<(u32, usize)>>,
}

thread_local! {
    static STEP_SEQUENCE_SCRATCH: RefCell<Option<StepSequenceScratch>> =
        const { RefCell::new(None) };
}

fn with_step_sequence_scratch<R>(action: impl FnOnce(&mut StepSequenceScratch) -> R) -> R {
    STEP_SEQUENCE_SCRATCH.with(|slot| {
        let mut guard = slot.borrow_mut();
        action(guard.get_or_insert_with(StepSequenceScratch::default))
    })
}

impl PbwtDirection {
    pub(crate) fn for_iteration(iteration: usize) -> Self {
        if iteration % 2 == 0 {
            Self::Reverse
        } else {
            Self::Forward
        }
    }
}

/// Build Beagle-style IBS donor proposals using one batch per available worker
/// and 1 cM overlap buffers. Beagle's PBWT partition intentionally depends on
/// the requested thread count.
pub(crate) fn build_phase_donor_index_with_pool(
    window: &PhasedWindow,
    cm_positions: &[f64],
    step_cm: f64,
    direction: PbwtDirection,
    max_candidates: usize,
    ibd2_intervals: &[super::types::Ibd2Interval],
    pool: &ThreadPool,
    seed: u64,
) -> Result<PhasePbwtIndex, String> {
    let view = PhaseHaplotypeView::target(window);
    build_phase_donor_index_with_pool_view(
        &view,
        cm_positions,
        step_cm,
        direction,
        max_candidates,
        ibd2_intervals,
        pool,
        seed,
    )
}

pub(crate) fn build_phase_donor_index_with_pool_view(
    view: &PhaseHaplotypeView<'_>,
    cm_positions: &[f64],
    step_cm: f64,
    direction: PbwtDirection,
    max_candidates: usize,
    ibd2_intervals: &[super::types::Ibd2Interval],
    pool: &ThreadPool,
    seed: u64,
) -> Result<PhasePbwtIndex, String> {
    let profile = pbwt_profile_enabled();
    let profile_started = profile.then(Instant::now);
    if view.n_markers() == 0 {
        return Err("phase PBWT requires a nonempty phased window".to_owned());
    }
    if cm_positions.len() != view.n_markers() {
        return Err("phase PBWT map and window lengths differ".to_owned());
    }
    if !step_cm.is_finite() || step_cm <= 0.0 {
        return Err("phase PBWT requires a positive finite step size".to_owned());
    }
    if pool.current_num_threads() == 0 {
        return Err("phase PBWT requires a positive thread count".to_owned());
    }
    if cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("phase PBWT requires a finite nondecreasing map".to_owned());
    }
    if ibd2_intervals.windows(2).any(|pair| {
        (
            pair[0].source_sample,
            pair[0].target_sample,
            pair[0].start_marker,
        ) > (
            pair[1].source_sample,
            pair[1].target_sample,
            pair[1].start_marker,
        )
    }) {
        return Err("phase PBWT requires ordered IBD2 intervals".to_owned());
    }
    let n_samples = view.n_samples();
    if n_samples == 0 {
        return Err("phase view has no samples".to_owned());
    }
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "PBWT haplotype count overflow".to_owned())?;
    let n_proposal_haplotypes = view
        .target_samples()
        .checked_mul(2)
        .ok_or_else(|| "PBWT target haplotype count overflow".to_owned())?;
    if n_proposal_haplotypes == 0 || n_proposal_haplotypes > n_haplotypes {
        return Err("phase PBWT target haplotype count is invalid".to_owned());
    }
    if n_haplotypes > i32::MAX as usize {
        return Err(
            "phase PBWT has more haplotypes than its 32-bit donor index supports".to_owned(),
        );
    }
    let step_ranges = phase_step_ranges(cm_positions, step_cm)?;
    if i32::try_from(step_ranges.len()).is_err() {
        return Err("phase PBWT has more steps than its 32-bit divergence supports".to_owned());
    }
    let proposal_count = step_ranges
        .len()
        .checked_mul(n_proposal_haplotypes)
        .ok_or_else(|| "phase PBWT donor proposal count overflow".to_owned())?;
    let compact_donors = n_haplotypes <= u16::MAX as usize;
    let mut donor_haplotypes = DonorProposalStorage::new(proposal_count, compact_donors);
    let candidate_limit = max_candidates.min(n_haplotypes);
    let trace_ignore_ibd2 = std::env::var_os("JANUSX_TRACE_IGNORE_IBD2").is_some();
    let search_ibd2_intervals = if trace_ignore_ibd2 {
        &[][..]
    } else {
        ibd2_intervals
    };

    let buffer_ratio = 1.0f32 / step_cm as f32;
    let overlap_steps = if buffer_ratio.is_finite() {
        (buffer_ratio as f64).round_ties_even() as usize
    } else {
        step_ranges.len()
    };
    let batch_count = phase_pbwt_batch_count(pool.current_num_threads(), step_ranges.len());
    let batch_ranges = phase_pbwt_batch_ranges(step_ranges.len(), batch_count, overlap_steps);
    let step_code_count = step_ranges
        .len()
        .checked_mul(n_haplotypes)
        .ok_or_else(|| "phase PBWT step code count overflow".to_owned())?;
    let mut step_codes = if n_haplotypes <= u16::MAX as usize {
        PhaseStepCodes::Compact(vec![0u16; step_code_count])
    } else {
        PhaseStepCodes::Wide(vec![0u32; step_code_count])
    };
    let encode_started = profile.then(Instant::now);
    let step_code_counts = match &mut step_codes {
        PhaseStepCodes::Compact(rows) => pool.install(|| {
            rows.par_chunks_mut(n_haplotypes)
                .enumerate()
                .map(|(step, output)| {
                    with_step_sequence_scratch(|scratch| {
                        let mut codes = Vec::<u32>::with_capacity(n_haplotypes);
                        let (start, end) = step_ranges[step];
                        let count =
                            encode_step_sequences_view(view, start, end, &mut codes, scratch);
                        if count > u16::MAX as usize {
                            return Err(
                                "compact phase PBWT step code count exceeds its u16 capacity"
                                    .to_owned(),
                            );
                        }
                        for (destination, source) in output.iter_mut().zip(codes) {
                            *destination = source as u16;
                        }
                        Ok(count)
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })?,
        PhaseStepCodes::Wide(rows) => pool.install(|| {
            rows.par_chunks_mut(n_haplotypes)
                .enumerate()
                .map(|(step, output)| {
                    with_step_sequence_scratch(|scratch| {
                        let mut codes = Vec::<u32>::with_capacity(n_haplotypes);
                        let (start, end) = step_ranges[step];
                        let count =
                            encode_step_sequences_view(view, start, end, &mut codes, scratch);
                        output.copy_from_slice(&codes);
                        Ok(count)
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })?,
    };
    let encode_elapsed = encode_started.map_or(std::time::Duration::ZERO, |start| start.elapsed());
    let steps_per_batch = step_ranges.len().div_ceil(batch_count).max(1);
    let donor_chunk_size = steps_per_batch
        .checked_mul(n_proposal_haplotypes)
        .ok_or_else(|| "phase PBWT donor batch size overflow".to_owned())?;
    let trace_target = std::env::var("JANUSX_TRACE_PBWT_TARGET")
        .is_ok_and(|target| target == format!("{}:13", view.target_window().index));
    let trace_seed = std::env::var("JANUSX_TRACE_PBWT_SEED")
        .is_ok_and(|value| value.parse::<u64>().ok() == Some(seed));
    let trace_state_step = std::env::var("JANUSX_TRACE_PBWT_STATE_STEP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let trace_state_window = std::env::var("JANUSX_TRACE_PBWT_STATE_WINDOW")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    let search_started = profile.then(Instant::now);
    let search_thread_start = PBWT_SEARCH_THREAD_NANOS.load(Ordering::Relaxed);
    match &mut donor_haplotypes {
        DonorProposalStorage::Compact(rows) => pool.install(|| {
            rows.par_chunks_mut(donor_chunk_size)
                .zip(batch_ranges.par_iter())
                .for_each(|(donor_chunk, &batch_range)| {
                    build_phase_donor_batch::<u16>(
                        view,
                        direction,
                        &step_ranges,
                        &step_codes,
                        &step_code_counts,
                        batch_range,
                        seed,
                        candidate_limit,
                        n_proposal_haplotypes,
                        search_ibd2_intervals,
                        trace_target,
                        trace_seed,
                        trace_state_step,
                        trace_state_window,
                        profile,
                        donor_chunk,
                    );
                });
        }),
        DonorProposalStorage::Wide(rows) => pool.install(|| {
            rows.par_chunks_mut(donor_chunk_size)
                .zip(batch_ranges.par_iter())
                .for_each(|(donor_chunk, &batch_range)| {
                    build_phase_donor_batch::<i32>(
                        view,
                        direction,
                        &step_ranges,
                        &step_codes,
                        &step_code_counts,
                        batch_range,
                        seed,
                        candidate_limit,
                        n_proposal_haplotypes,
                        search_ibd2_intervals,
                        trace_target,
                        trace_seed,
                        trace_state_step,
                        trace_state_window,
                        profile,
                        donor_chunk,
                    );
                });
        }),
    }
    let search_elapsed = search_started.map_or(std::time::Duration::ZERO, |start| start.elapsed());
    let search_thread_elapsed = search_started.map_or(std::time::Duration::ZERO, |_| {
        std::time::Duration::from_nanos(
            PBWT_SEARCH_THREAD_NANOS
                .load(Ordering::Relaxed)
                .saturating_sub(search_thread_start),
        )
    });

    if let Ok(trace_sample) = std::env::var("JANUSX_TRACE_DONOR_SAMPLE") {
        if let Ok(trace_sample) = trace_sample.parse::<usize>() {
            let explicit_target = std::env::var("JANUSX_TRACE_DONOR_TARGET").is_ok_and(|target| {
                target == format!("{}:{trace_sample}", view.target_window().index)
            });
            if (explicit_target
                || (view.target_window().chrom == "9" && view.target_window().index == 35))
                && trace_sample < view.target_samples()
            {
                eprintln!(
                    "JANUSX_DONOR_HEADER\tSEED={seed}\tDIRECTION={direction:?}\tSAMPLE={trace_sample}\tCANDIDATES={candidate_limit}\tIGNORE_IBD2={trace_ignore_ibd2}\tSTEP_CM={step_cm:.12}\tSPAN={:.12}\tNSTEPS={}",
                    cm_positions.last().copied().unwrap_or(0.0)
                        - cm_positions.first().copied().unwrap_or(0.0),
                    step_ranges.len()
                );
                let hap0 = 2 * trace_sample;
                for step in 0..step_ranges.len() {
                    let offset = step * n_proposal_haplotypes;
                    let [first, second] = donor_haplotypes.raw_pair(offset + hap0);
                    eprintln!(
                        "JANUSX_DONOR\tSEED={seed}\tSTEP={step}\tHAP0={}\tHAP1={}",
                        first, second
                    );
                }
            }
        }
    }

    if let Some(started) = profile_started {
        eprintln!(
            "JANUSX_IMP_PBWT_CALL_PROFILE window={}:{} direction={direction:?} steps={} haplotypes={} batches={} ibd2_intervals={} encode_s={:.6} search_s={:.6} search_thread_s={:.6} total_s={:.6}",
            view.target_window().chrom,
            view.target_window().index,
            step_ranges.len(),
            n_haplotypes,
            batch_ranges.len(),
            ibd2_intervals.len(),
            encode_elapsed.as_secs_f64(),
            search_elapsed.as_secs_f64(),
            search_thread_elapsed.as_secs_f64(),
            started.elapsed().as_secs_f64(),
        );
    }
    Ok(PhasePbwtIndex {
        n_haplotypes,
        n_proposal_haplotypes,
        step_ranges,
        donor_haplotypes,
    })
}

#[cfg(test)]
pub(crate) fn build_phase_donor_index(
    window: &PhasedWindow,
    cm_positions: &[f64],
    step_cm: f64,
    direction: PbwtDirection,
    max_candidates: usize,
    ibd2_intervals: &[super::types::Ibd2Interval],
    n_threads: usize,
    seed: u64,
) -> Result<PhasePbwtIndex, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|error| format!("cannot build phase PBWT test pool: {error}"))?;
    build_phase_donor_index_with_pool(
        window,
        cm_positions,
        step_cm,
        direction,
        max_candidates,
        ibd2_intervals,
        &pool,
        seed,
    )
}

/// Build Beagle Stage2's forward and backward IBS proposals. PBWT orders are
/// updated from the projected Stage1 markers; within each step, low-frequency
/// allele carriers add adjacency links that can select donors with a rare
/// allele even when the Stage1 sequence itself is not unique.
pub(crate) fn build_low_frequency_phase_donor_index(
    stage1_window: &PhasedWindow,
    stage1_cm_positions: &[f64],
    input: &PhaseWindow,
    stage1_marker_indices: &[usize],
    step_cm: f64,
    max_carriers: usize,
    n_candidates: usize,
    max_backoff_steps: usize,
    ibd2_intervals: &[super::types::Ibd2Interval],
    seed: u64,
) -> Result<LowFrequencyPhasePbwtIndex, String> {
    if stage1_window.rows.is_empty() || input.rows.is_empty() {
        return Err("low-frequency phase PBWT requires nonempty windows".to_owned());
    }
    let stage1_view = PhaseHaplotypeView::target(stage1_window);
    let full_window = PhasedWindow {
        chrom: input.chrom.clone(),
        index: input.index,
        rows: input
            .rows
            .iter()
            .map(|row| PhasedRow {
                meta: row.meta.clone(),
                calls: row.calls.clone(),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
            .collect(),
        emit_start: input.emit_start,
        emit_end: input.emit_end,
    };
    let full_view = PhaseHaplotypeView::target(&full_window);
    build_low_frequency_phase_donor_index_view(
        &stage1_view,
        &full_view,
        Some(input),
        stage1_cm_positions,
        stage1_marker_indices,
        step_cm,
        max_carriers,
        n_candidates,
        max_backoff_steps,
        ibd2_intervals,
        seed,
    )
}

pub(crate) fn build_low_frequency_phase_donor_index_view(
    stage1_view: &PhaseHaplotypeView<'_>,
    full_view: &PhaseHaplotypeView<'_>,
    carrier_window: Option<&PhaseWindow>,
    stage1_cm_positions: &[f64],
    stage1_marker_indices: &[usize],
    step_cm: f64,
    max_carriers: usize,
    n_candidates: usize,
    max_backoff_steps: usize,
    ibd2_intervals: &[super::types::Ibd2Interval],
    seed: u64,
) -> Result<LowFrequencyPhasePbwtIndex, String> {
    build_low_frequency_phase_donor_index_view_impl(
        stage1_view,
        full_view,
        carrier_window,
        stage1_cm_positions,
        stage1_marker_indices,
        step_cm,
        max_carriers,
        n_candidates,
        max_backoff_steps,
        ibd2_intervals,
        seed,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_low_frequency_phase_donor_index_view_with_pool(
    stage1_view: &PhaseHaplotypeView<'_>,
    full_view: &PhaseHaplotypeView<'_>,
    carrier_window: Option<&PhaseWindow>,
    stage1_cm_positions: &[f64],
    stage1_marker_indices: &[usize],
    step_cm: f64,
    max_carriers: usize,
    n_candidates: usize,
    max_backoff_steps: usize,
    ibd2_intervals: &[super::types::Ibd2Interval],
    seed: u64,
    pool: &ThreadPool,
) -> Result<LowFrequencyPhasePbwtIndex, String> {
    build_low_frequency_phase_donor_index_view_impl(
        stage1_view,
        full_view,
        carrier_window,
        stage1_cm_positions,
        stage1_marker_indices,
        step_cm,
        max_carriers,
        n_candidates,
        max_backoff_steps,
        ibd2_intervals,
        seed,
        Some(pool),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_low_frequency_phase_donor_index_view_impl(
    stage1_view: &PhaseHaplotypeView<'_>,
    full_view: &PhaseHaplotypeView<'_>,
    carrier_window: Option<&PhaseWindow>,
    stage1_cm_positions: &[f64],
    stage1_marker_indices: &[usize],
    step_cm: f64,
    max_carriers: usize,
    n_candidates: usize,
    max_backoff_steps: usize,
    ibd2_intervals: &[super::types::Ibd2Interval],
    seed: u64,
    pool: Option<&ThreadPool>,
) -> Result<LowFrequencyPhasePbwtIndex, String> {
    if stage1_view.n_markers() == 0
        || full_view.n_markers() == 0
        || stage1_view.n_markers() != stage1_marker_indices.len()
        || stage1_view.n_markers() != stage1_cm_positions.len()
        || stage1_view.target_samples() != full_view.target_samples()
        || carrier_window.is_some_and(|window| {
            window.rows.len() != full_view.n_markers()
                || window.rows.first().map_or(0, |row| row.calls.len())
                    != full_view.target_samples()
        })
        || stage1_marker_indices
            .iter()
            .any(|&marker| marker >= full_view.n_markers())
        || stage1_marker_indices
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
        || stage1_cm_positions.iter().any(|cm| !cm.is_finite())
        || stage1_cm_positions.windows(2).any(|pair| pair[1] < pair[0])
        || !step_cm.is_finite()
        || step_cm <= 0.0
        || max_carriers == 0
        || n_candidates == 0
        || ibd2_intervals.windows(2).any(|pair| {
            (
                pair[0].source_sample,
                pair[0].target_sample,
                pair[0].start_marker,
            ) > (
                pair[1].source_sample,
                pair[1].target_sample,
                pair[1].start_marker,
            )
        })
    {
        return Err("low-frequency phase PBWT dimensions or parameters are invalid".to_owned());
    }
    let n_samples = full_view.n_samples();
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "low-frequency phase haplotype count overflow".to_owned())?;
    if n_haplotypes > i32::MAX as usize {
        return Err(
            "low-frequency phase PBWT has more haplotypes than its 32-bit donor index supports"
                .to_owned(),
        );
    }

    let stage1_steps = phase_step_ranges(stage1_cm_positions, step_cm)?;
    if i32::try_from(stage1_steps.len()).is_err() {
        return Err(
            "low-frequency phase PBWT has more steps than its 32-bit divergence supports"
                .to_owned(),
        );
    }
    let mut full_step_ranges = Vec::with_capacity(stage1_steps.len());
    for (step, &(start, end)) in stage1_steps.iter().enumerate() {
        let full_start = if step == 0 {
            0
        } else {
            stage1_marker_indices[start]
        };
        let full_end = if end == stage1_marker_indices.len() {
            full_view.n_markers()
        } else {
            stage1_marker_indices[end]
        };
        if full_start >= full_end {
            return Err("low-frequency phase PBWT produced an empty full-marker step".to_owned());
        }
        full_step_ranges.push((full_start, full_end));
    }
    if full_step_ranges.first().map(|range| range.0) != Some(0)
        || full_step_ranges.last().map(|range| range.1) != Some(full_view.n_markers())
        || full_step_ranges
            .windows(2)
            .any(|pair| pair[0].1 != pair[1].0)
    {
        return Err("low-frequency phase PBWT steps do not partition the input window".to_owned());
    }

    let proposal_count = stage1_steps
        .len()
        .checked_mul(n_haplotypes)
        .ok_or_else(|| "low-frequency phase donor proposal count overflow".to_owned())?;
    let mut forward_donors = vec![-1i32; proposal_count];
    let mut backward_donors = vec![-1i32; proposal_count];
    let mut build_forward = || {
        build_low_frequency_direction_view(
            stage1_view,
            full_view,
            carrier_window,
            &stage1_steps,
            &full_step_ranges,
            max_carriers,
            n_candidates.min(n_haplotypes),
            max_backoff_steps,
            ibd2_intervals,
            seed,
            PbwtDirection::Forward,
            &mut forward_donors,
        )
    };
    let mut build_backward = || {
        build_low_frequency_direction_view(
            stage1_view,
            full_view,
            carrier_window,
            &stage1_steps,
            &full_step_ranges,
            max_carriers,
            n_candidates.min(n_haplotypes),
            max_backoff_steps,
            ibd2_intervals,
            seed,
            PbwtDirection::Reverse,
            &mut backward_donors,
        )
    };
    if let Some(pool) = pool {
        let (forward_result, backward_result) =
            pool.install(|| rayon::join(build_forward, build_backward));
        forward_result?;
        backward_result?;
    } else {
        build_forward()?;
        build_backward()?;
    }

    Ok(LowFrequencyPhasePbwtIndex {
        n_haplotypes,
        step_ranges: full_step_ranges,
        forward_donors,
        backward_donors,
    })
}

fn build_low_frequency_direction_view(
    stage1_view: &PhaseHaplotypeView<'_>,
    full_view: &PhaseHaplotypeView<'_>,
    carrier_window: Option<&PhaseWindow>,
    stage1_steps: &[(usize, usize)],
    full_step_ranges: &[(usize, usize)],
    max_carriers: usize,
    n_candidates: usize,
    max_backoff_steps: usize,
    ibd2_intervals: &[super::types::Ibd2Interval],
    seed: u64,
    direction: PbwtDirection,
    donor_haplotypes: &mut [i32],
) -> Result<(), String> {
    let n_samples = full_view.n_samples();
    let n_haplotypes = n_samples * 2;
    let mut order = (0..n_haplotypes).collect::<Vec<_>>();
    let mut divergence = vec![0i32; n_haplotypes + 1];
    if direction == PbwtDirection::Reverse {
        divergence.fill((stage1_steps.len() - 1) as i32);
    }
    let mut updater = PbwtDivUpdater::new(n_haplotypes);
    let mut codes = Vec::<u32>::with_capacity(n_haplotypes);
    let mut sequence_scratch = StepSequenceScratch::default();
    let mut inverse = vec![0usize; n_haplotypes];
    let mut random = JavaRandom::new(seed);

    let step_iter: Box<dyn Iterator<Item = usize>> = match direction {
        PbwtDirection::Forward => Box::new(0..stage1_steps.len()),
        PbwtDirection::Reverse => Box::new((0..stage1_steps.len()).rev()),
    };
    for step in step_iter {
        let (start, end) = stage1_steps[step];
        let n_codes =
            encode_step_sequences_view(stage1_view, start, end, &mut codes, &mut sequence_scratch);
        match direction {
            PbwtDirection::Forward => {
                updater.forward_update(&codes, n_codes, step, &mut order, &mut divergence);
                divergence[n_haplotypes] = step as i32 + 1;
            }
            PbwtDirection::Reverse => {
                updater.reverse_update(&codes, n_codes, step, &mut order, &mut divergence);
                divergence[n_haplotypes] = step as i32 - 1;
            }
        }
        for (rank, &haplotype) in order.iter().enumerate() {
            inverse[haplotype] = rank;
        }
        let (previous, next) = low_frequency_hap_neighbors_view(
            full_view,
            carrier_window,
            full_step_ranges[step].0,
            full_step_ranges[step].1,
            &inverse,
            max_carriers,
        )?;
        random.reset(seed.wrapping_add(step as u64));
        for (target_rank, &target_haplotype) in order.iter().enumerate() {
            if target_haplotype >= full_view.target_samples() * 2 {
                continue;
            }
            let proposal = best_low_frequency_donor_rank(
                direction,
                step,
                stage1_steps.len(),
                target_rank,
                &order,
                &divergence,
                &previous,
                &next,
                max_backoff_steps,
                target_haplotype / 2,
                stage1_steps[step],
                ibd2_intervals,
            )
            .or_else(|| {
                low_frequency_fallback_interval(
                    direction,
                    step,
                    target_rank,
                    n_candidates,
                    &divergence,
                )
                .and_then(|(left, right)| {
                    let length = right - left;
                    if length <= 1 {
                        return None;
                    }
                    let first = random.next_int(length);
                    (0..length).find_map(|offset| {
                        let donor_rank = left + (first + offset) % length;
                        let donor_haplotype = order[donor_rank];
                        let donor_sample = donor_haplotype / 2;
                        (donor_sample != target_haplotype / 2
                            && super::ibd2::can_contribute_segment(
                                donor_sample,
                                target_haplotype / 2,
                                stage1_steps[step].0,
                                stage1_steps[step].1,
                                ibd2_intervals,
                            ))
                        .then_some(donor_rank)
                    })
                })
            });
            if let Some(donor_rank) = proposal {
                let index = step * n_haplotypes + target_haplotype;
                donor_haplotypes[index] = order[donor_rank] as i32;
            }
        }
    }
    Ok(())
}

fn best_low_frequency_donor_rank(
    direction: PbwtDirection,
    step: usize,
    n_steps: usize,
    target_rank: usize,
    order: &[usize],
    divergence: &[i32],
    previous: &[Option<usize>],
    next: &[Option<usize>],
    max_backoff_steps: usize,
    target_sample: usize,
    marker_range: (usize, usize),
    ibd2_intervals: &[super::types::Ibd2Interval],
) -> Option<usize> {
    let is_allowed = |rank: usize| {
        let Some(&donor_haplotype) = order.get(rank) else {
            return false;
        };
        let source_sample = donor_haplotype / 2;
        source_sample != target_sample
            && super::ibd2::can_contribute_segment(
                source_sample,
                target_sample,
                marker_range.0,
                marker_range.1,
                ibd2_intervals,
            )
    };
    let left_div = *divergence.get(target_rank)?;
    let right_div = *divergence.get(target_rank + 1)?;
    match direction {
        PbwtDirection::Forward => {
            let min_match_start = left_div.min(right_div);
            let d_max = min_match_start
                .saturating_add(i32::try_from(max_backoff_steps).unwrap_or(i32::MAX))
                .min(step as i32);
            let mut best_prev = None;
            let mut prev_match_start = 0i32;
            let mut prev_rank = previous.get(target_rank).copied().flatten();
            while prev_rank.is_some_and(|rank| !is_allowed(rank)) {
                prev_rank = prev_rank.and_then(|rank| previous[rank]);
            }
            if let Some(prev) = prev_rank {
                let mut u = target_rank;
                while u > prev + 1 && divergence[u] <= d_max {
                    prev_match_start = prev_match_start.max(divergence[u]);
                    u -= 1;
                }
                if u == prev + 1 && divergence[u] <= d_max {
                    prev_match_start = prev_match_start.max(divergence[u]);
                    best_prev = Some(prev);
                }
            }

            let mut best_next = None;
            let mut next_match_start = 0i32;
            let mut next_rank = next.get(target_rank).copied().flatten();
            while next_rank.is_some_and(|rank| !is_allowed(rank)) {
                next_rank = next_rank.and_then(|rank| next[rank]);
            }
            if let Some(next_rank) = next_rank {
                let mut v = target_rank;
                while v + 1 < next_rank && divergence[v + 1] <= d_max {
                    v += 1;
                    next_match_start = next_match_start.max(divergence[v]);
                }
                if v + 1 == next_rank && divergence[v + 1] <= d_max {
                    next_match_start = next_match_start.max(divergence[v + 1]);
                    best_next = Some(next_rank);
                }
            }
            if prev_match_start < next_match_start && best_prev.is_some() {
                best_prev
            } else {
                best_next
            }
        }
        PbwtDirection::Reverse => {
            let max_match_start = left_div.max(right_div);
            let d_min = max_match_start
                .saturating_sub(i32::try_from(max_backoff_steps).unwrap_or(i32::MAX))
                .max(step as i32);
            let mut best_prev = None;
            let mut prev_match_end = n_steps.saturating_sub(1) as i32;
            let mut prev_rank = previous.get(target_rank).copied().flatten();
            while prev_rank.is_some_and(|rank| !is_allowed(rank)) {
                prev_rank = prev_rank.and_then(|rank| previous[rank]);
            }
            if let Some(prev) = prev_rank {
                let mut u = target_rank;
                while u > prev + 1 && divergence[u] >= d_min {
                    prev_match_end = prev_match_end.min(divergence[u]);
                    u -= 1;
                }
                if u == prev + 1 && divergence[u] >= d_min {
                    prev_match_end = prev_match_end.min(divergence[u]);
                    best_prev = Some(prev);
                }
            }

            let mut best_next = None;
            let mut next_match_end = n_steps.saturating_sub(1) as i32;
            let mut next_rank = next.get(target_rank).copied().flatten();
            while next_rank.is_some_and(|rank| !is_allowed(rank)) {
                next_rank = next_rank.and_then(|rank| next[rank]);
            }
            if let Some(next_rank) = next_rank {
                let mut v = target_rank;
                while v + 1 < next_rank && divergence[v + 1] >= d_min {
                    v += 1;
                    next_match_end = next_match_end.min(divergence[v]);
                }
                if v + 1 == next_rank && divergence[v + 1] >= d_min {
                    next_match_end = next_match_end.min(divergence[v + 1]);
                    best_next = Some(next_rank);
                }
            }
            if prev_match_end > next_match_end && best_prev.is_some() {
                best_prev
            } else {
                best_next
            }
        }
    }
}

fn low_frequency_fallback_interval(
    direction: PbwtDirection,
    step: usize,
    target_rank: usize,
    n_candidates: usize,
    divergence: &[i32],
) -> Option<(usize, usize)> {
    if n_candidates <= 1 || target_rank + 1 >= divergence.len() {
        return None;
    }
    let mut left = target_rank;
    let mut right = target_rank + 1;
    let mut left_div = divergence[left];
    let mut right_div = divergence[right];
    while right - left < n_candidates {
        let should_expand = match direction {
            PbwtDirection::Forward => left_div <= step as i32 || right_div <= step as i32,
            PbwtDirection::Reverse => step as i32 <= left_div || step as i32 <= right_div,
        };
        if !should_expand {
            break;
        }
        match direction {
            PbwtDirection::Forward if right_div <= left_div && right + 1 < divergence.len() => {
                right += 1;
                right_div = right_div.max(divergence[right]);
            }
            PbwtDirection::Forward if left > 0 => {
                left -= 1;
                left_div = left_div.max(divergence[left]);
            }
            PbwtDirection::Reverse if left_div <= right_div && right + 1 < divergence.len() => {
                right += 1;
                right_div = right_div.min(divergence[right]);
            }
            PbwtDirection::Reverse if left > 0 => {
                left -= 1;
                left_div = left_div.min(divergence[left]);
            }
            _ => break,
        }
    }
    (right - left > 1).then_some((left, right))
}

fn low_frequency_hap_neighbors(
    window: &PhaseWindow,
    start_marker: usize,
    end_marker: usize,
    inverse_order: &[usize],
    max_carriers: usize,
) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>), String> {
    if window.rows.is_empty() {
        return Err("low-frequency carrier search requires markers".to_owned());
    }
    let phased = PhasedWindow {
        chrom: window.chrom.clone(),
        index: window.index,
        rows: window
            .rows
            .iter()
            .map(|row| PhasedRow {
                meta: row.meta.clone(),
                calls: row.calls.clone(),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
            .collect(),
        emit_start: window.emit_start,
        emit_end: window.emit_end,
    };
    let view = PhaseHaplotypeView::target(&phased);
    low_frequency_hap_neighbors_view(
        &view,
        None,
        start_marker,
        end_marker,
        inverse_order,
        max_carriers,
    )
}

fn low_frequency_hap_neighbors_view(
    view: &PhaseHaplotypeView<'_>,
    carrier_window: Option<&PhaseWindow>,
    start_marker: usize,
    end_marker: usize,
    inverse_order: &[usize],
    max_carriers: usize,
) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>), String> {
    let n_samples = view.n_samples();
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "low-frequency carrier haplotype count overflow".to_owned())?;
    if n_samples == 0
        || start_marker >= end_marker
        || end_marker > view.n_markers()
        || inverse_order.len() != n_haplotypes
        || max_carriers == 0
    {
        return Err("low-frequency carrier search dimensions are invalid".to_owned());
    }

    let mut previous = vec![None; n_haplotypes];
    let mut next = vec![None; n_haplotypes];
    let mut carriers_by_allele = Vec::<Vec<usize>>::new();
    for marker in start_marker..end_marker {
        carriers_by_allele.clear();
        carriers_by_allele.resize_with(view.marker(marker).allele_count(), Vec::new);
        for sample in 0..n_samples {
            let call = if sample < view.target_samples() {
                carrier_window.map_or_else(
                    || Some(view.call(marker, sample)),
                    |window| window.rows[marker].calls.get(sample),
                )
            } else {
                Some(view.call(marker, sample))
            };
            let Some(call) = call else {
                return Err("low-frequency carrier sample is out of range".to_owned());
            };
            let [first, second] = call.alleles;
            for allele in [first, second].into_iter().flatten() {
                let Some(carriers) = carriers_by_allele.get_mut(usize::from(allele)) else {
                    return Err(format!(
                        "low-frequency allele {allele} is outside marker {marker}'s allele range"
                    ));
                };
                if carriers.last().copied() == Some(sample) {
                    continue;
                }
                if carriers.len() <= max_carriers {
                    carriers.push(sample);
                }
            }
        }

        for carriers in &carriers_by_allele {
            if !(2..=max_carriers).contains(&carriers.len()) {
                continue;
            }
            let mut ranks = Vec::with_capacity(carriers.len() * 2);
            for &sample in carriers {
                ranks.push(inverse_order[2 * sample]);
                ranks.push(inverse_order[2 * sample + 1]);
            }
            ranks.sort_unstable();
            for adjacent in ranks.windows(2) {
                let left = adjacent[0];
                let right = adjacent[1];
                if previous[right].is_none_or(|current| left > current) {
                    previous[right] = Some(left);
                }
                if next[left].is_none_or(|current| right < current) {
                    next[left] = Some(right);
                }
            }
        }
    }
    Ok((previous, next))
}

/// Partition markers as Beagle's `Steps`: each next step starts at the first
/// marker whose genetic position is at least `step_cm` beyond the current
/// step's first marker.
pub(crate) fn phase_step_ranges(
    cm_positions: &[f64],
    step_cm: f64,
) -> Result<Vec<(usize, usize)>, String> {
    if cm_positions.is_empty()
        || !step_cm.is_finite()
        || step_cm <= 0.0
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err(
            "phase PBWT steps require a nonempty, ordered map and positive step size".to_owned(),
        );
    }
    let min_step = f64::from(step_cm as f32);
    if !min_step.is_finite() || min_step <= 0.0 {
        return Err("phase PBWT step size is outside Beagle's float range".to_owned());
    }
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < cm_positions.len() {
        let min_gen_pos = cm_positions[start] + min_step;
        let mut end = start + 1;
        while end < cm_positions.len() && cm_positions[end] < min_gen_pos {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }
    Ok(ranges)
}

fn phase_pbwt_batch_ranges(
    n_steps: usize,
    batch_count: usize,
    overlap_steps: usize,
) -> Vec<(usize, usize, usize, usize)> {
    if n_steps == 0 || batch_count == 0 {
        return Vec::new();
    }
    let steps_per_batch = n_steps.div_ceil(batch_count).max(1);
    (0..n_steps)
        .step_by(steps_per_batch)
        .map(|batch_start| {
            let batch_end = (batch_start + steps_per_batch).min(n_steps);
            let buffer_start = batch_start.saturating_sub(overlap_steps);
            let buffer_end = batch_end.saturating_add(overlap_steps).min(n_steps);
            (batch_start, batch_end, buffer_start, buffer_end)
        })
        .collect()
}

fn initial_pbwt_partition_count(n_threads: usize) -> usize {
    n_threads.max(1)
}

fn phase_pbwt_batch_count(n_threads: usize, n_steps: usize) -> usize {
    n_threads.min(n_steps).max(1)
}

#[allow(clippy::too_many_arguments)]
fn build_phase_donor_batch<T: DonorProposalId>(
    view: &PhaseHaplotypeView<'_>,
    direction: PbwtDirection,
    step_ranges: &[(usize, usize)],
    step_codes: &PhaseStepCodes,
    step_code_counts: &[usize],
    batch_range: (usize, usize, usize, usize),
    seed: u64,
    candidate_limit: usize,
    n_proposal_haplotypes: usize,
    ibd2_intervals: &[super::types::Ibd2Interval],
    trace_target: bool,
    trace_seed: bool,
    trace_state_step: Option<usize>,
    trace_state_window: Option<u64>,
    profile: bool,
    donor_chunk: &mut [T],
) {
    let batch_started = profile.then(Instant::now);
    let (batch_start, batch_end, buffer_start, buffer_end) = batch_range;
    let n_haplotypes = view.n_samples() * 2;
    with_pbwt_batch_scratch(n_haplotypes, |batch| {
        batch.reset(n_haplotypes, buffer_start);
        let PbwtBatchScratch {
            order,
            divergence,
            updater,
            codes_u32,
        } = batch;
        let mut random = JavaRandom::new(seed);

        match direction {
            PbwtDirection::Forward => {
                for step in buffer_start..batch_end {
                    let (start, end) = step_ranges[step];
                    let codes = match step_codes {
                        PhaseStepCodes::Compact(rows) => {
                            codes_u32.clear();
                            codes_u32.extend(
                                rows[step * n_haplotypes..(step + 1) * n_haplotypes]
                                    .iter()
                                    .map(|&code| u32::from(code)),
                            );
                            codes_u32.as_slice()
                        }
                        PhaseStepCodes::Wide(rows) => {
                            &rows[step * n_haplotypes..(step + 1) * n_haplotypes]
                        }
                    };
                    updater.forward_update(codes, step_code_counts[step], step, order, divergence);
                    let sentinel = step as i32 + 2;
                    divergence[0] = sentinel;
                    divergence[n_haplotypes] = sentinel;
                    if trace_state_window == Some(view.target_window().index)
                        && trace_state_step == Some(step)
                    {
                        eprintln!(
                            "JANUSX_PBWT_STATE\tWINDOW={}\tDIRECTION=Forward\tSTEP={step}\tSEED={seed}\tCODES={codes:?}\tORDER={order:?}\tDIVERGENCE={divergence:?}",
                            view.target_window().index,
                        );
                    }
                    if step >= batch_start {
                        random.reset(seed.wrapping_add(step as u64));
                        write_step_donors::<T, true>(
                            (step - batch_start) * n_proposal_haplotypes,
                            start,
                            end,
                            step as i32,
                            candidate_limit,
                            n_proposal_haplotypes,
                            order,
                            divergence,
                            ibd2_intervals,
                            &mut random,
                            donor_chunk,
                        );
                    }
                }
            }
            PbwtDirection::Reverse => {
                divergence.fill((buffer_end - 1) as i32);
                for step in (buffer_start..buffer_end).rev() {
                    let (start, end) = step_ranges[step];
                    let codes = match step_codes {
                        PhaseStepCodes::Compact(rows) => {
                            codes_u32.clear();
                            codes_u32.extend(
                                rows[step * n_haplotypes..(step + 1) * n_haplotypes]
                                    .iter()
                                    .map(|&code| u32::from(code)),
                            );
                            codes_u32.as_slice()
                        }
                        PhaseStepCodes::Wide(rows) => {
                            &rows[step * n_haplotypes..(step + 1) * n_haplotypes]
                        }
                    };
                    updater.reverse_update(codes, step_code_counts[step], step, order, divergence);
                    let sentinel = step as i32 - 2;
                    divergence[0] = sentinel;
                    divergence[n_haplotypes] = sentinel;
                    if trace_state_window == Some(view.target_window().index)
                        && trace_state_step == Some(step)
                    {
                        eprintln!(
                            "JANUSX_PBWT_STATE\tWINDOW={}\tDIRECTION=Reverse\tSTEP={step}\tSEED={seed}\tCODES={codes:?}\tORDER={order:?}\tDIVERGENCE={divergence:?}",
                            view.target_window().index,
                        );
                    }
                    let in_batch = (batch_start..batch_end).contains(&step);
                    if in_batch && trace_target && trace_seed {
                        eprintln!(
                            "JANUSX_STEP_CODES\tWINDOW={}\tSTEP={step}\tSTART={start}\tEND={end}\tCODES={codes:?}",
                            view.target_window().index,
                        );
                    }
                    if step == 44 && trace_target && trace_seed {
                        let calls = (0..view.n_samples())
                            .map(|sample| {
                                let call = view.call(start, sample);
                                format!(
                                    "{}|{}",
                                    call.alleles[0].map_or(-1, i32::from),
                                    call.alleles[1].map_or(-1, i32::from)
                                )
                            })
                            .collect::<Vec<_>>();
                        eprintln!(
                            "JANUSX_PBWT_RAW_CALLS\tWINDOW={}\tSTEP={step}\tMARKER={start}\tCALLS={calls:?}",
                            view.target_window().index
                        );
                    }
                    if in_batch && trace_target && trace_seed {
                        eprintln!(
                            "JANUSX_PBWT_STEP_STATE\tWINDOW={}\tSEED={seed}\tSTEP={step}\tSTART={start}\tEND={end}\tORDER={order:?}\tDIVERGENCE={divergence:?}",
                            view.target_window().index
                        );
                    }
                    if in_batch {
                        random.reset(seed.wrapping_add(step as u64));
                        write_step_donors::<T, false>(
                            (step - batch_start) * n_proposal_haplotypes,
                            start,
                            end,
                            step as i32,
                            candidate_limit,
                            n_proposal_haplotypes,
                            order,
                            divergence,
                            ibd2_intervals,
                            &mut random,
                            donor_chunk,
                        );
                    }
                }
            }
        }
    });
    if let Some(started) = batch_started {
        PBWT_SEARCH_THREAD_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Encode the exact multi-marker allele sequence in each genetic step as one
/// categorical PBWT symbol. Beagle assigns category IDs in first-haplotype
/// order, which also fixes the stable order of distinct sequence groups.
fn encode_step_sequences(
    window: &PhasedWindow,
    start: usize,
    end: usize,
    codes: &mut Vec<u32>,
    scratch: &mut StepSequenceScratch,
) -> usize {
    if let Some(unique_sequences) =
        encode_biallelic_step_sequences(window, start, end, codes, scratch)
    {
        return unique_sequences;
    }

    let n_samples = window.rows[start].calls.len();
    let n_haplotypes = n_samples * 2;
    codes.clear();
    codes.reserve(n_haplotypes);
    scratch.first_sequence_by_hash.clear();
    scratch.collisions.clear();
    let mut next_code = 0u32;
    for haplotype in 0..n_haplotypes {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for marker in start..end {
            window.rows[marker]
                .calls
                .get(haplotype / 2)
                .unwrap()
                .alleles[haplotype & 1]
                .hash(&mut hasher);
        }
        let sequence_hash = hasher.finish();
        let code = if let Some(&(first_code, first_haplotype)) =
            scratch.first_sequence_by_hash.get(&sequence_hash)
        {
            if same_step_sequence(window, start, end, first_haplotype, haplotype) {
                first_code
            } else {
                let bucket = scratch
                    .collisions
                    .entry(sequence_hash)
                    .or_insert_with(|| vec![(first_code, first_haplotype)]);
                if let Some(&(code, _)) = bucket.iter().find(|&&(_, representative)| {
                    same_step_sequence(window, start, end, representative, haplotype)
                }) {
                    code
                } else {
                    let code = next_code;
                    next_code += 1;
                    bucket.push((code, haplotype));
                    code
                }
            }
        } else {
            scratch
                .first_sequence_by_hash
                .insert(sequence_hash, (next_code, haplotype));
            let code = next_code;
            next_code += 1;
            code
        };
        codes.push(code);
    }
    next_code as usize
}

fn encode_step_sequences_view(
    view: &PhaseHaplotypeView<'_>,
    start: usize,
    end: usize,
    codes: &mut Vec<u32>,
    scratch: &mut StepSequenceScratch,
) -> usize {
    if let Some(unique_sequences) =
        encode_biallelic_step_sequences_view(view, start, end, codes, scratch)
    {
        return unique_sequences;
    }

    let n_haplotypes = view.n_samples() * 2;
    codes.clear();
    codes.reserve(n_haplotypes);
    scratch.first_sequence_by_hash.clear();
    scratch.collisions.clear();
    let mut next_code = 0u32;
    for haplotype in 0..n_haplotypes {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for marker in start..end {
            view.allele(marker, haplotype / 2, haplotype & 1)
                .hash(&mut hasher);
        }
        let sequence_hash = hasher.finish();
        let code = if let Some(&(first_code, first_haplotype)) =
            scratch.first_sequence_by_hash.get(&sequence_hash)
        {
            if same_step_sequence_view(view, start, end, first_haplotype, haplotype) {
                first_code
            } else {
                let bucket = scratch
                    .collisions
                    .entry(sequence_hash)
                    .or_insert_with(|| vec![(first_code, first_haplotype)]);
                if let Some(&(code, _)) = bucket.iter().find(|&&(_, representative)| {
                    same_step_sequence_view(view, start, end, representative, haplotype)
                }) {
                    code
                } else {
                    let code = next_code;
                    next_code += 1;
                    bucket.push((code, haplotype));
                    code
                }
            }
        } else {
            scratch
                .first_sequence_by_hash
                .insert(sequence_hash, (next_code, haplotype));
            let code = next_code;
            next_code += 1;
            code
        };
        codes.push(code);
    }
    next_code as usize
}

fn encode_biallelic_step_sequences_view(
    view: &PhaseHaplotypeView<'_>,
    start: usize,
    end: usize,
    codes: &mut Vec<u32>,
    scratch: &mut StepSequenceScratch,
) -> Option<usize> {
    let marker_count = end.checked_sub(start)?;
    if marker_count == 0 || marker_count > PACKED_STEP_SEQUENCE_MAX_MARKERS {
        return None;
    }
    let n_samples = view.n_samples();
    let n_haplotypes = n_samples.checked_mul(2)?;
    if n_haplotypes == 0 {
        return None;
    }
    scratch
        .packed_biallelic
        .resize(n_haplotypes, [0; PACKED_STEP_SEQUENCE_WORDS]);
    let n_words = marker_count.div_ceil(32);
    for sequence in &mut scratch.packed_biallelic[..n_haplotypes] {
        sequence[..n_words].fill(0);
    }

    for marker_offset in 0..marker_count {
        let marker = start + marker_offset;
        let word = marker_offset / 32;
        let shift = (marker_offset % 32) * 2;
        for sample in 0..n_samples {
            let first = match view.allele(marker, sample, 0) {
                Some(0) => 0u64,
                Some(1) => 1u64,
                None => 2u64,
                Some(_) => return None,
            };
            let second = match view.allele(marker, sample, 1) {
                Some(0) => 0u64,
                Some(1) => 1u64,
                None => 2u64,
                Some(_) => return None,
            };
            let haplotype = sample * 2;
            scratch.packed_biallelic[haplotype][word] |= first << shift;
            scratch.packed_biallelic[haplotype + 1][word] |= second << shift;
        }
    }

    codes.clear();
    codes.reserve(n_haplotypes);
    let table_len = n_haplotypes.checked_mul(2)?.checked_next_power_of_two()?;
    scratch.packed_sequence_table.resize(table_len, (0, 0));
    scratch.packed_sequence_generation.resize(table_len, 0);
    if scratch.current_generation == u32::MAX {
        scratch.packed_sequence_generation.fill(0);
        scratch.current_generation = 0;
    }
    scratch.current_generation += 1;
    let generation = scratch.current_generation;
    let table_mask = table_len - 1;
    let mut next_code = 0u32;
    for haplotype in 0..n_haplotypes {
        let sequence = &scratch.packed_biallelic[haplotype];
        let sequence_hash = hash_packed_biallelic_sequence(sequence, n_words, marker_count);
        let mut slot = sequence_hash as usize & table_mask;
        let code = loop {
            let (stored_hash, representative) = scratch.packed_sequence_table[slot];
            if scratch.packed_sequence_generation[slot] != generation {
                scratch.packed_sequence_table[slot] = (sequence_hash, haplotype as u32);
                scratch.packed_sequence_generation[slot] = generation;
                let code = next_code;
                next_code += 1;
                break code;
            }
            if stored_hash == sequence_hash
                && packed_biallelic_sequences_equal(
                    sequence,
                    &scratch.packed_biallelic[representative as usize],
                    n_words,
                )
            {
                break codes[representative as usize];
            }
            slot = (slot + 1) & table_mask;
        };
        codes.push(code);
    }
    Some(next_code as usize)
}

fn encode_biallelic_step_sequences(
    window: &PhasedWindow,
    start: usize,
    end: usize,
    codes: &mut Vec<u32>,
    scratch: &mut StepSequenceScratch,
) -> Option<usize> {
    let marker_count = end.checked_sub(start)?;
    if marker_count == 0 || marker_count > PACKED_STEP_SEQUENCE_MAX_MARKERS {
        return None;
    }
    let n_samples = window.rows.get(start)?.calls.len();
    let n_haplotypes = n_samples.checked_mul(2)?;
    if n_haplotypes == 0 {
        return None;
    }
    scratch
        .packed_biallelic
        .resize(n_haplotypes, [0; PACKED_STEP_SEQUENCE_WORDS]);
    let n_words = marker_count.div_ceil(32);
    for sequence in &mut scratch.packed_biallelic[..n_haplotypes] {
        sequence[..n_words].fill(0);
    }

    for (marker_offset, row) in window.rows.get(start..end)?.iter().enumerate() {
        let word = marker_offset / 32;
        let shift = (marker_offset % 32) * 2;
        for sample in 0..n_samples {
            let haplotype = sample * 2;
            if let Some(code) = row.calls.common_biallelic_code(sample) {
                let pair = match code {
                    0 => 0u64,
                    1 | 2 => 4u64,
                    3 => 1u64,
                    4 => 5u64,
                    5 => 10u64,
                    _ => unreachable!(),
                };
                scratch.packed_biallelic[haplotype][word] |= (pair & 3) << shift;
                scratch.packed_biallelic[haplotype + 1][word] |= ((pair >> 2) & 3) << shift;
            } else {
                let call = row.calls.get(sample).unwrap();
                let key0 = match call.alleles[0] {
                    Some(0) => 0u64,
                    Some(1) => 1u64,
                    None => 2u64,
                    Some(_) => return None,
                };
                let key1 = match call.alleles[1] {
                    Some(0) => 0u64,
                    Some(1) => 1u64,
                    None => 2u64,
                    Some(_) => return None,
                };
                scratch.packed_biallelic[haplotype][word] |= key0 << shift;
                scratch.packed_biallelic[haplotype + 1][word] |= key1 << shift;
            }
        }
    }

    codes.clear();
    codes.reserve(n_haplotypes);
    let table_len = n_haplotypes.checked_mul(2)?.checked_next_power_of_two()?;
    scratch.packed_sequence_table.resize(table_len, (0, 0));
    scratch.packed_sequence_generation.resize(table_len, 0);
    if scratch.current_generation == u32::MAX {
        scratch.packed_sequence_generation.fill(0);
        scratch.current_generation = 0;
    }
    scratch.current_generation += 1;
    let generation = scratch.current_generation;
    let table_mask = table_len - 1;
    let mut next_code = 0u32;
    for haplotype in 0..n_haplotypes {
        let sequence = &scratch.packed_biallelic[haplotype];
        let sequence_hash = hash_packed_biallelic_sequence(sequence, n_words, marker_count);
        let mut slot = sequence_hash as usize & table_mask;
        let code = loop {
            let (stored_hash, representative) = scratch.packed_sequence_table[slot];
            if scratch.packed_sequence_generation[slot] != generation {
                scratch.packed_sequence_table[slot] = (sequence_hash, haplotype as u32);
                scratch.packed_sequence_generation[slot] = generation;
                let code = next_code;
                next_code += 1;
                break code;
            }
            if stored_hash == sequence_hash
                && packed_biallelic_sequences_equal(
                    sequence,
                    &scratch.packed_biallelic[representative as usize],
                    n_words,
                )
            {
                break codes[representative as usize];
            }
            slot = (slot + 1) & table_mask;
        };
        codes.push(code);
    }
    Some(next_code as usize)
}

#[inline]
fn packed_biallelic_sequences_equal(
    left: &[u64; PACKED_STEP_SEQUENCE_WORDS],
    right: &[u64; PACKED_STEP_SEQUENCE_WORDS],
    n_words: usize,
) -> bool {
    left[..n_words] == right[..n_words]
}

#[inline]
fn hash_packed_biallelic_sequence(
    sequence: &[u64; PACKED_STEP_SEQUENCE_WORDS],
    n_words: usize,
    marker_count: usize,
) -> u64 {
    let mut hash = 0x9e37_79b9_7f4a_7c15u64 ^ marker_count as u64;
    for word in &sequence[..n_words] {
        hash ^= word.wrapping_mul(0x9e37_79b1_85eb_ca87);
        hash = hash
            .rotate_left(27)
            .wrapping_mul(0x3c79_ac49_2ba7_b653)
            .wrapping_add(0x1c69_b3f7_4ac4_ae35);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

fn same_step_sequence(
    window: &PhasedWindow,
    start: usize,
    end: usize,
    left_haplotype: usize,
    right_haplotype: usize,
) -> bool {
    (start..end).all(|marker| {
        let calls = &window.rows[marker].calls;
        calls.get(left_haplotype / 2).unwrap().alleles[left_haplotype & 1]
            == calls.get(right_haplotype / 2).unwrap().alleles[right_haplotype & 1]
    })
}

fn same_step_sequence_view(
    view: &PhaseHaplotypeView<'_>,
    start: usize,
    end: usize,
    left_haplotype: usize,
    right_haplotype: usize,
) -> bool {
    (start..end).all(|marker| {
        view.allele(marker, left_haplotype / 2, left_haplotype & 1)
            == view.allele(marker, right_haplotype / 2, right_haplotype & 1)
    })
}

fn write_step_donors<T: DonorProposalId, const FORWARD: bool>(
    donor_proposal_offset: usize,
    start_marker: usize,
    end_marker: usize,
    divergence_boundary: i32,
    max_candidates: usize,
    n_proposal_haplotypes: usize,
    order: &[usize],
    divergence: &[i32],
    ibd2_intervals: &[super::types::Ibd2Interval],
    random: &mut JavaRandom,
    donor_haplotypes: &mut [T],
) {
    if max_candidates <= 1 {
        return;
    }
    let check_ibs2 = !ibd2_intervals.is_empty();
    for target_rank in 0..order.len() {
        let target_haplotype = order[target_rank];
        if target_haplotype >= n_proposal_haplotypes {
            continue;
        }
        let target_sample = target_haplotype / 2;
        let (mut left, mut right, mut left_divergence, mut right_divergence) = (
            target_rank,
            target_rank + 1,
            divergence[target_rank],
            divergence[target_rank + 1],
        );
        loop {
            let interval_len = right - left;
            let should_expand = interval_len < max_candidates
                && if FORWARD {
                    left_divergence <= divergence_boundary
                        || right_divergence <= divergence_boundary
                } else {
                    divergence_boundary <= left_divergence
                        || divergence_boundary <= right_divergence
                };
            if !should_expand {
                break;
            }
            if FORWARD {
                if right_divergence <= left_divergence {
                    right += 1;
                    right_divergence = right_divergence.max(divergence[right]);
                } else {
                    left -= 1;
                    left_divergence = left_divergence.max(divergence[left]);
                }
            } else if left_divergence <= right_divergence {
                right += 1;
                right_divergence = right_divergence.min(divergence[right]);
            } else {
                left -= 1;
                left_divergence = left_divergence.min(divergence[left]);
            }
        }

        let interval_len = right - left;
        if interval_len <= 1 {
            continue;
        }
        let first = random.next_int(interval_len);
        for offset in 0..interval_len {
            let rank = left + (first + offset) % interval_len;
            if rank == target_rank {
                continue;
            }
            let donor_haplotype = order[rank];
            let donor_sample = donor_haplotype / 2;
            if donor_sample == target_sample
                || (check_ibs2
                    && !super::ibd2::can_contribute_segment(
                        donor_sample,
                        target_sample,
                        start_marker,
                        end_marker,
                        ibd2_intervals,
                    ))
            {
                continue;
            }
            let proposal_index = donor_proposal_offset + target_haplotype;
            donor_haplotypes[proposal_index] = T::from_haplotype(donor_haplotype);
            break;
        }
    }
}

struct PbwtDivUpdater {
    p: Vec<i32>,
    counts: Vec<u32>,
    alleles: Vec<u32>,
    updated_divergence: Vec<i32>,
    next_order: Vec<usize>,
}

impl PbwtDivUpdater {
    fn new(n_haplotypes: usize) -> Self {
        Self {
            p: Vec::new(),
            counts: Vec::new(),
            alleles: vec![0; n_haplotypes],
            updated_divergence: vec![0; n_haplotypes],
            next_order: vec![0; n_haplotypes],
        }
    }

    fn forward_update(
        &mut self,
        alleles: &[u32],
        n_alleles: usize,
        marker: usize,
        order: &mut Vec<usize>,
        divergence: &mut [i32],
    ) {
        self.prepare(n_alleles, order.len());
        self.update_regroup_scalar::<true>(alleles, n_alleles, marker, order, divergence);
        self.commit(n_alleles, order, divergence);
    }

    fn reverse_update(
        &mut self,
        alleles: &[u32],
        n_alleles: usize,
        marker: usize,
        order: &mut Vec<usize>,
        divergence: &mut [i32],
    ) {
        self.prepare(n_alleles, order.len());
        self.update_regroup_scalar::<false>(alleles, n_alleles, marker, order, divergence);
        self.commit(n_alleles, order, divergence);
    }

    fn update_regroup_scalar<const FORWARD: bool>(
        &mut self,
        alleles: &[u32],
        n_alleles: usize,
        marker: usize,
        order: &[usize],
        divergence: &[i32],
    ) {
        self.p[..n_alleles].fill(if FORWARD {
            marker as i32 + 1
        } else {
            marker as i32 - 1
        });
        let probabilities = &mut self.p[..n_alleles];
        for (rank, &haplotype) in order.iter().enumerate() {
            let div = divergence[rank];
            for value in probabilities.iter_mut() {
                *value = if FORWARD {
                    (*value).max(div)
                } else {
                    (*value).min(div)
                };
            }
            let allele = alleles[haplotype] as usize;
            self.alleles[rank] = allele as u32;
            self.updated_divergence[rank] = probabilities[allele];
            probabilities[allele] = if FORWARD { i32::MIN } else { i32::MAX };
        }
    }

    fn prepare(&mut self, n_alleles: usize, n_haplotypes: usize) {
        if self.p.len() < n_alleles {
            self.p.resize(n_alleles, 0);
            self.counts.resize(n_alleles + 1, 0);
        }
        self.alleles.resize(n_haplotypes, 0);
        self.updated_divergence.resize(n_haplotypes, 0);
        self.next_order.resize(n_haplotypes, 0);
    }

    fn commit(&mut self, n_alleles: usize, order: &mut Vec<usize>, divergence: &mut [i32]) {
        self.counts[..=n_alleles].fill(0);
        for &allele in &self.alleles[..order.len()] {
            self.counts[allele as usize + 1] += 1;
        }
        for allele in 1..=n_alleles {
            self.counts[allele] += self.counts[allele - 1];
        }
        let n_haplotypes = order.len();
        for rank in 0..n_haplotypes {
            let allele = self.alleles[rank] as usize;
            let destination = self.counts[allele] as usize;
            self.counts[allele] += 1;
            self.next_order[destination] = order[rank];
            divergence[destination] = self.updated_divergence[rank];
        }
        order.copy_from_slice(&self.next_order[..n_haplotypes]);
    }
}

struct PbwtBatchScratch {
    order: Vec<usize>,
    divergence: Vec<i32>,
    updater: PbwtDivUpdater,
    codes_u32: Vec<u32>,
}

impl PbwtBatchScratch {
    fn new(n_haplotypes: usize) -> Self {
        Self {
            order: (0..n_haplotypes).collect(),
            divergence: vec![0; n_haplotypes + 1],
            updater: PbwtDivUpdater::new(n_haplotypes),
            codes_u32: Vec::with_capacity(n_haplotypes),
        }
    }

    fn reset(&mut self, n_haplotypes: usize, initial_divergence: usize) {
        self.order.clear();
        self.order.extend(0..n_haplotypes);
        self.divergence.clear();
        self.divergence
            .resize(n_haplotypes + 1, initial_divergence as i32);
    }
}

thread_local! {
    static PBWT_BATCH_SCRATCH: RefCell<Option<PbwtBatchScratch>> = const { RefCell::new(None) };
}

fn with_pbwt_batch_scratch<R>(
    n_haplotypes: usize,
    action: impl FnOnce(&mut PbwtBatchScratch) -> R,
) -> R {
    PBWT_BATCH_SCRATCH.with(|slot| {
        let mut guard = slot.borrow_mut();
        action(guard.get_or_insert_with(|| PbwtBatchScratch::new(n_haplotypes)))
    })
}

/// Java's `Random` is used by Beagle to choose a random member of each PBWT
/// neighborhood. Matching its 48-bit generator keeps the proposal sampling
/// stable when the PBWT interval itself is identical.
pub(crate) struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    const MULTIPLIER: u64 = 0x5deece66d;
    const ADDEND: u64 = 0xb;
    const MASK: u64 = (1u64 << 48) - 1;

    pub(crate) fn new(seed: u64) -> Self {
        Self {
            seed: (seed ^ Self::MULTIPLIER) & Self::MASK,
        }
    }

    fn reset(&mut self, seed: u64) {
        self.seed = (seed ^ Self::MULTIPLIER) & Self::MASK;
    }

    fn next_bits(&mut self, bits: u32) -> u32 {
        self.seed = self
            .seed
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::ADDEND)
            & Self::MASK;
        (self.seed >> (48 - bits)) as u32
    }

    pub(crate) fn next_boolean(&mut self) -> bool {
        self.next_bits(1) != 0
    }

    /// Match `java.util.Random.nextLong()` so each Beagle input window gets
    /// the same independent phase seed as the Java driver.
    pub(crate) fn next_long(&mut self) -> u64 {
        let high = i64::from(self.next_bits(32) as i32) << 32;
        let low = i64::from(self.next_bits(32) as i32);
        high.wrapping_add(low) as u64
    }

    pub(crate) fn next_int(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        if bound.is_power_of_two() {
            return ((bound as u64 * self.next_bits(31) as u64) >> 31) as usize;
        }
        if bound <= i32::MAX as usize {
            loop {
                let bits = self.next_bits(31) as usize;
                let value = bits % bound;
                if bits - value + (bound - 1) < (1usize << 31) {
                    return value;
                }
            }
        }
        (self.next_bits(31) as usize) % bound
    }
}

pub(crate) fn beagle_window_seed(master_seed: u64, window_index: u64) -> u64 {
    let mut random = JavaRandom::new(master_seed);
    let mut seed = 0;
    for _ in 0..=window_index {
        seed = random.next_long();
    }
    seed
}

#[derive(Debug, Clone)]
pub(crate) struct PbwtSegment {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) markers: Vec<usize>,
    pub(crate) order: Vec<usize>,
}

pub(crate) fn build_index(window: &PhasedWindow) -> Result<PbwtIndex, String> {
    if window.rows.is_empty() {
        return Err("cannot build a PBWT index for an empty phased window".to_owned());
    }
    let n_samples = window.rows[0].calls.len();
    if n_samples == 0 || window.rows.iter().any(|row| row.calls.len() != n_samples) {
        return Err("phased window has an invalid or inconsistent sample count".to_owned());
    }
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "PBWT haplotype count overflow".to_owned())?;
    let max_allele = window
        .rows
        .iter()
        .map(|row| row.meta.allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let missing_bucket = usize::from(max_allele) + 1;
    let bucket_count = missing_bucket + 1;
    let mut order = (0..n_haplotypes).collect::<Vec<_>>();
    let mut inverse = order.clone();
    let mut reordered = vec![0usize; n_haplotypes];
    let mut counts = vec![0usize; bucket_count];
    let mut cursors = vec![0usize; bucket_count];

    for row in &window.rows {
        counts.fill(0);
        for haplotype in 0..n_haplotypes {
            let allele = row
                .calls
                .get(haplotype / 2)
                .and_then(|call| call.alleles[haplotype & 1]);
            let bucket = allele.map_or(missing_bucket, |allele| {
                usize::from(allele).min(usize::from(max_allele))
            });
            counts[bucket] += 1;
        }
        let mut offset = 0usize;
        for bucket in 0..bucket_count {
            cursors[bucket] = offset;
            offset += counts[bucket];
        }
        for &haplotype in &order {
            let allele = row
                .calls
                .get(haplotype / 2)
                .and_then(|call| call.alleles[haplotype & 1]);
            let bucket = allele.map_or(missing_bucket, |allele| {
                usize::from(allele).min(usize::from(max_allele))
            });
            reordered[cursors[bucket]] = haplotype;
            cursors[bucket] += 1;
        }
        std::mem::swap(&mut order, &mut reordered);
        for (rank, &haplotype) in order.iter().enumerate() {
            inverse[haplotype] = rank;
        }
    }

    Ok(PbwtIndex { order, inverse })
}

/// Select a bounded donor list around both target haplotypes in the final
/// PBWT order, omitting the target sample and duplicate donor haplotypes.
pub(crate) fn neighbors(
    index: &PbwtIndex,
    target_sample: usize,
    n_samples: usize,
    max_states: usize,
) -> Result<Vec<usize>, String> {
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "PBWT haplotype count overflow".to_owned())?;
    if target_sample >= n_samples
        || index.order.len() != n_haplotypes
        || index.inverse.len() != n_haplotypes
    {
        return Err("PBWT index dimensions do not match the target window".to_owned());
    }
    let candidate_count = n_haplotypes.saturating_sub(2);
    let limit = max_states.min(candidate_count);
    if limit == 0 {
        return Ok(Vec::new());
    }
    let centers = [
        index.inverse[2 * target_sample],
        index.inverse[2 * target_sample + 1],
    ];
    let mut selected = Vec::with_capacity(limit);
    let mut seen = HashSet::with_capacity(limit);
    for distance in 1..=n_haplotypes {
        for center in centers {
            for rank in [
                center.checked_sub(distance),
                center
                    .checked_add(distance)
                    .filter(|&rank| rank < n_haplotypes),
            ]
            .into_iter()
            .flatten()
            {
                let haplotype = index.order[rank];
                if haplotype / 2 != target_sample && seen.insert(haplotype) {
                    selected.push(haplotype);
                    if selected.len() == limit {
                        return Ok(selected);
                    }
                }
            }
        }
    }
    Ok(selected)
}

/// Build one local PBWT order per genetic segment. Only the order at the end
/// of each segment is retained; scan buffers are reused between segments.
pub(crate) fn build_segment_index(
    window: &PhasedWindow,
    cm_positions: &[f64],
    searchable_markers: &[bool],
    segment_cm: f64,
    direction: PbwtDirection,
) -> Result<SegmentPbwtIndex, String> {
    if window.rows.is_empty()
        || cm_positions.len() != window.rows.len()
        || searchable_markers.len() != window.rows.len()
        || !segment_cm.is_finite()
        || segment_cm <= 0.0
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err(
            "PBWT segment index requires a valid map and positive segment length".to_owned(),
        );
    }
    let n_samples = window.rows[0].calls.len();
    if n_samples == 0 || window.rows.iter().any(|row| row.calls.len() != n_samples) {
        return Err("phased window has an invalid or inconsistent sample count".to_owned());
    }
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "PBWT haplotype count overflow".to_owned())?;
    let max_allele = window
        .rows
        .iter()
        .map(|row| row.meta.allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let missing_bucket = usize::from(max_allele) + 1;
    let bucket_count = missing_bucket + 1;
    let mut counts = vec![0usize; bucket_count];
    let mut cursors = vec![0usize; bucket_count];
    let mut reordered = vec![0usize; n_haplotypes];
    let mut segment_count = 0usize;
    let mut start = 0usize;
    while start < window.rows.len() {
        let end = segment_end(cm_positions, start, segment_cm);
        segment_count += 1;
        start = end;
    }
    let mut segments = Vec::with_capacity(segment_count);
    let mut start = 0usize;
    while start < window.rows.len() {
        let end = segment_end(cm_positions, start, segment_cm);
        let mut markers = Vec::with_capacity(
            searchable_markers[start..end]
                .iter()
                .filter(|&&searchable| searchable)
                .count(),
        );
        markers.extend((start..end).filter(|&marker| searchable_markers[marker]));
        let mut order = (0..n_haplotypes).collect::<Vec<_>>();
        let mut update_marker = |marker: usize| {
            let row = &window.rows[marker];
            counts.fill(0);
            for haplotype in 0..n_haplotypes {
                let allele = row
                    .calls
                    .get(haplotype / 2)
                    .and_then(|call| call.alleles[haplotype & 1]);
                let bucket = allele.map_or(missing_bucket, |allele| {
                    usize::from(allele).min(usize::from(max_allele))
                });
                counts[bucket] += 1;
            }
            let mut offset = 0usize;
            for bucket in 0..bucket_count {
                cursors[bucket] = offset;
                offset += counts[bucket];
            }
            for &haplotype in &order {
                let allele = row
                    .calls
                    .get(haplotype / 2)
                    .and_then(|call| call.alleles[haplotype & 1]);
                let bucket = allele.map_or(missing_bucket, |allele| {
                    usize::from(allele).min(usize::from(max_allele))
                });
                reordered[cursors[bucket]] = haplotype;
                cursors[bucket] += 1;
            }
            std::mem::swap(&mut order, &mut reordered);
        };
        match direction {
            PbwtDirection::Forward => {
                for &marker in &markers {
                    update_marker(marker);
                }
            }
            PbwtDirection::Reverse => {
                for &marker in markers.iter().rev() {
                    update_marker(marker);
                }
            }
        }
        segments.push(PbwtSegment {
            start,
            end,
            markers,
            order,
        });
        start = end;
    }
    Ok(SegmentPbwtIndex {
        n_haplotypes,
        direction,
        segments,
    })
}

/// Build a segment PBWT directly from an allele accessor. This allows callers
/// with a marker-major packed panel to avoid materializing a second diploid
/// row matrix just to build the donor index.
pub(crate) fn build_segment_index_from_alleles<F>(
    n_haplotypes: usize,
    n_markers: usize,
    max_allele: u16,
    cm_positions: &[f64],
    searchable_markers: &[bool],
    segment_cm: f64,
    direction: PbwtDirection,
    allele_at: F,
) -> Result<SegmentPbwtIndex, String>
where
    F: Fn(usize, usize) -> Option<u16>,
{
    if n_haplotypes == 0
        || n_markers == 0
        || cm_positions.len() != n_markers
        || searchable_markers.len() != n_markers
        || !segment_cm.is_finite()
        || segment_cm <= 0.0
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err(
            "PBWT segment index requires a valid map and positive segment length".to_owned(),
        );
    }
    let missing_bucket = usize::from(max_allele) + 1;
    let bucket_count = missing_bucket + 1;
    let mut counts = vec![0usize; bucket_count];
    let mut cursors = vec![0usize; bucket_count];
    let mut reordered = vec![0usize; n_haplotypes];
    let mut segment_count = 0usize;
    let mut start = 0usize;
    while start < n_markers {
        let end = segment_end(cm_positions, start, segment_cm);
        segment_count += 1;
        start = end;
    }
    let mut segments = Vec::with_capacity(segment_count);
    let mut start = 0usize;
    while start < n_markers {
        let end = segment_end(cm_positions, start, segment_cm);
        let mut markers = Vec::with_capacity(
            searchable_markers[start..end]
                .iter()
                .filter(|&&searchable| searchable)
                .count(),
        );
        markers.extend((start..end).filter(|&marker| searchable_markers[marker]));
        let mut order = (0..n_haplotypes).collect::<Vec<_>>();
        let mut update_marker = |marker: usize| {
            counts.fill(0);
            for haplotype in 0..n_haplotypes {
                let bucket = allele_at(marker, haplotype).map_or(missing_bucket, |allele| {
                    usize::from(allele).min(usize::from(max_allele))
                });
                counts[bucket] += 1;
            }
            let mut offset = 0usize;
            for bucket in 0..bucket_count {
                cursors[bucket] = offset;
                offset += counts[bucket];
            }
            for &haplotype in &order {
                let bucket = allele_at(marker, haplotype).map_or(missing_bucket, |allele| {
                    usize::from(allele).min(usize::from(max_allele))
                });
                reordered[cursors[bucket]] = haplotype;
                cursors[bucket] += 1;
            }
            std::mem::swap(&mut order, &mut reordered);
        };
        match direction {
            PbwtDirection::Forward => {
                for &marker in &markers {
                    update_marker(marker);
                }
            }
            PbwtDirection::Reverse => {
                for &marker in markers.iter().rev() {
                    update_marker(marker);
                }
            }
        }
        segments.push(PbwtSegment {
            start,
            end,
            markers,
            order,
        });
        start = end;
    }
    Ok(SegmentPbwtIndex {
        n_haplotypes,
        direction,
        segments,
    })
}

fn segment_end(cm_positions: &[f64], start: usize, segment_cm: f64) -> usize {
    let segment_cm_start = cm_positions[start];
    let mut end = start + 1;
    while end < cm_positions.len() && cm_positions[end] - segment_cm_start < segment_cm {
        end += 1;
    }
    end
}

fn nearest_segment_haplotypes<T, F>(order: &[T], limit: usize, compare: F) -> Vec<T>
where
    T: Copy,
    F: Fn(T) -> std::cmp::Ordering,
{
    let mut low = 0usize;
    let mut high = order.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if compare(order[middle]).is_lt() {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let exact_start = low;
    high = order.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if compare(order[middle]).is_le() {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let exact_end = low;

    let exact_count = (exact_end - exact_start).min(limit);
    let mut selected = Vec::with_capacity(limit);
    selected.extend_from_slice(&order[exact_start..exact_start + exact_count]);

    let mut left = exact_start;
    let mut right = exact_end;
    while selected.len() < limit && (left > 0 || right < order.len()) {
        if left > 0 {
            left -= 1;
            selected.push(order[left]);
            if selected.len() == limit {
                break;
            }
        }
        if right < order.len() {
            selected.push(order[right]);
            right += 1;
        }
    }
    selected
}

/// Locate a target sequence in a segment's reversed-prefix order and return
/// exact matches first, followed by the nearest haplotypes on both sides.
pub(crate) fn segment_neighbors(
    index: &SegmentPbwtIndex,
    segment_index: usize,
    window: &PhasedWindow,
    query_alleles: &[Option<u16>],
    max_states: usize,
) -> Result<Vec<usize>, String> {
    let segment = index
        .segments
        .get(segment_index)
        .ok_or_else(|| "PBWT segment index is out of range".to_owned())?;
    let n_samples = window.rows.first().map_or(0, |row| row.calls.len());
    if query_alleles.len() != window.rows.len()
        || index.n_haplotypes != n_samples.saturating_mul(2)
        || segment.order.len() != index.n_haplotypes
    {
        return Err("PBWT query dimensions do not match the reference window".to_owned());
    }
    let limit = max_states.min(index.n_haplotypes);
    if limit == 0 {
        return Ok(Vec::new());
    }
    Ok(nearest_segment_haplotypes(
        &segment.order,
        limit,
        |haplotype| {
            compare_haplotype_to_query(window, haplotype, segment, index.direction, query_alleles)
        },
    ))
}

/// Locate a target sequence in a segment PBWT order using a packed-panel
/// allele accessor rather than a materialized `PhasedWindow`.
pub(crate) fn segment_neighbors_from_alleles<F>(
    index: &SegmentPbwtIndex,
    segment_index: usize,
    n_markers: usize,
    query_alleles: &[Option<u16>],
    max_states: usize,
    allele_at: F,
) -> Result<Vec<usize>, String>
where
    F: Fn(usize, usize) -> Option<u16>,
{
    let segment = index
        .segments
        .get(segment_index)
        .ok_or_else(|| "PBWT segment index is out of range".to_owned())?;
    if query_alleles.len() != n_markers
        || index.n_haplotypes == 0
        || segment.order.len() != index.n_haplotypes
    {
        return Err("PBWT query dimensions do not match the reference window".to_owned());
    }
    let limit = max_states.min(index.n_haplotypes);
    if limit == 0 {
        return Ok(Vec::new());
    }
    Ok(nearest_segment_haplotypes(
        &segment.order,
        limit,
        |haplotype| {
            compare_haplotype_to_query_by(
                haplotype,
                segment,
                index.direction,
                query_alleles,
                &allele_at,
            )
        },
    ))
}

/// Return haplotypes that are identical to a query across the complete PBWT
/// step. Beagle's phase IBS proposals require a match that reaches the
/// current step; a lexicographically nearby but mismatching haplotype is not
/// an IBS proposal.
pub(crate) fn segment_exact_neighbors(
    index: &SegmentPbwtIndex,
    segment_index: usize,
    window: &PhasedWindow,
    query_alleles: &[Option<u16>],
    target_sample: usize,
) -> Result<Vec<usize>, String> {
    let segment = index
        .segments
        .get(segment_index)
        .ok_or_else(|| "PBWT segment index is out of range".to_owned())?;
    let n_samples = window.rows.first().map_or(0, |row| row.calls.len());
    if query_alleles.len() != window.rows.len()
        || target_sample >= n_samples
        || index.n_haplotypes != n_samples.saturating_mul(2)
        || segment.order.len() != index.n_haplotypes
    {
        return Err("PBWT query dimensions do not match the reference window".to_owned());
    }
    let mut low = 0usize;
    let mut high = segment.order.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if compare_haplotype_to_query(
            window,
            segment.order[middle],
            segment,
            index.direction,
            query_alleles,
        )
        .is_lt()
        {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let start = low;
    high = segment.order.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if compare_haplotype_to_query(
            window,
            segment.order[middle],
            segment,
            index.direction,
            query_alleles,
        )
        .is_le()
        {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(segment.order[start..low]
        .iter()
        .copied()
        .filter(|haplotype| haplotype / 2 != target_sample)
        .collect())
}

fn compare_haplotype_to_query(
    window: &PhasedWindow,
    haplotype: usize,
    segment: &PbwtSegment,
    direction: PbwtDirection,
    query_alleles: &[Option<u16>],
) -> std::cmp::Ordering {
    compare_haplotype_to_query_by(
        haplotype,
        segment,
        direction,
        query_alleles,
        &|marker, haplotype| {
            window.rows[marker]
                .calls
                .get(haplotype / 2)
                .and_then(|call| call.alleles[haplotype & 1])
        },
    )
}

fn compare_haplotype_to_query_by<F>(
    haplotype: usize,
    segment: &PbwtSegment,
    direction: PbwtDirection,
    query_alleles: &[Option<u16>],
    allele_at: &F,
) -> std::cmp::Ordering
where
    F: Fn(usize, usize) -> Option<u16>,
{
    let compare_marker = |&marker: &usize| {
        let observed = allele_at(marker, haplotype);
        let query = query_alleles[marker];
        let ordering = match (observed, query) {
            (Some(observed), Some(query)) => observed.cmp(&query),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };
        (!ordering.is_eq()).then_some(ordering)
    };
    match direction {
        PbwtDirection::Forward => segment.markers.iter().rev().find_map(compare_marker),
        PbwtDirection::Reverse => segment.markers.iter().find_map(compare_marker),
    }
    .unwrap_or(std::cmp::Ordering::Equal)
}

pub(crate) fn initialize(window: &PhaseWindow, seed: u64) -> Result<PhasedWindow, String> {
    initialize_with_threads(window, seed, 1)
}

pub(crate) fn initialize_with_threads(
    window: &PhaseWindow,
    seed: u64,
    n_threads: usize,
) -> Result<PhasedWindow, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|error| format!("cannot build initial phase Rayon pool: {error}"))?;
    initialize_with_pool(window, seed, &pool)
}

pub(crate) fn initialize_with_pool(
    window: &PhaseWindow,
    seed: u64,
    pool: &ThreadPool,
) -> Result<PhasedWindow, String> {
    initialize_with_reference_pool(window, None, seed, pool)
}

pub(crate) fn initialize_with_reference_pool(
    window: &PhaseWindow,
    reference: Option<&PackedHaplotypes>,
    seed: u64,
    pool: &ThreadPool,
) -> Result<PhasedWindow, String> {
    let n_threads = pool.current_num_threads();
    let profile = pbwt_profile_enabled();
    let profile_started = profile.then(Instant::now);
    let mut phase_elapsed = std::time::Duration::ZERO;
    let mut reverse_elapsed = std::time::Duration::ZERO;
    let mut forward_elapsed = std::time::Duration::ZERO;
    let mut stitch_elapsed = std::time::Duration::ZERO;
    if window.rows.is_empty() {
        return Err("cannot initialize an empty phase window".to_owned());
    }
    let n_samples = window.rows[0].calls.len();
    if n_samples == 0 || window.rows.iter().any(|row| row.calls.len() != n_samples) {
        return Err("phase window has an invalid or inconsistent sample count".to_owned());
    }
    let n_markers = window.rows.len();
    let n_target_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "haplotype count overflow".to_owned())?;
    let n_reference_haplotypes = reference.map_or(0, |panel| panel.n_haplotypes);
    let n_panel_haplotypes = n_target_haplotypes
        .checked_add(n_reference_haplotypes)
        .ok_or_else(|| "initial PBWT panel haplotype count overflow".to_owned())?;
    if let Some(reference) = reference {
        if reference.n_markers != n_markers
            || reference.bits_per_allele != 1
            || !reference.is_complete()
            || window.rows.iter().any(|row| row.meta.allele_count() != 2)
        {
            return Err(
                "reference-assisted initial PBWT requires complete biallelic marker alignment"
                    .to_owned(),
            );
        }
    }
    if n_threads == 0 {
        return Err("initial PBWT phasing requires at least one thread".to_owned());
    }
    let max_allele = window
        .rows
        .iter()
        .map(|row| row.meta.allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let width = allele_width(max_allele);
    let phased_overlap = window.incoming_overlap_len;
    let chunk_ranges = beagle_initial_pbwt_ranges(
        &window.cm_positions,
        initial_pbwt_partition_count(n_threads),
    )?;
    let mut haplotypes = PackedHaplotypes::empty(n_target_haplotypes, n_markers, width)?;
    let rephase_heterozygotes = super::phase_baum::has_initial_unphased_heterozygote(window);
    let first_hets = first_heterozygote_by_sample(window, n_samples);
    let compute_started = profile.then(Instant::now);
    let phase_fragments = if width == 1 {
        pool.install(|| {
            chunk_ranges
                .par_iter()
                .enumerate()
                .map(|(chunk_index, &(start, end))| {
                    initial_phase_fragment(
                        window,
                        seed,
                        chunk_index,
                        start,
                        end,
                        phased_overlap,
                        rephase_heterozygotes,
                        n_samples,
                        n_target_haplotypes,
                        n_panel_haplotypes,
                        reference,
                        width,
                        profile,
                    )
                })
                .collect::<Result<Vec<_>, String>>()
        })?
    } else {
        chunk_ranges
            .iter()
            .enumerate()
            .map(|(chunk_index, &(start, end))| {
                initial_phase_fragment(
                    window,
                    seed,
                    chunk_index,
                    start,
                    end,
                    phased_overlap,
                    rephase_heterozygotes,
                    n_samples,
                    n_target_haplotypes,
                    n_panel_haplotypes,
                    reference,
                    width,
                    profile,
                )
            })
            .collect::<Result<Vec<_>, String>>()?
    };
    if let Some(started) = compute_started {
        phase_elapsed = started.elapsed();
    }
    for fragment in &phase_fragments {
        reverse_elapsed += fragment.reverse_elapsed;
        forward_elapsed += fragment.forward_elapsed;
    }

    let stitch_started = profile.then(Instant::now);
    let mut previous_end = 0usize;
    for fragment in phase_fragments {
        let start = fragment.start;
        let end = fragment.end;
        let forward = fragment.forward;
        let copy_start = (start + previous_end) >> 1;
        let mut swap_samples = vec![false; n_samples];
        if start > 0 {
            for sample in 0..n_samples {
                let Some(align_marker) = initial_alignment_het(
                    window,
                    sample,
                    first_hets[sample],
                    phased_overlap,
                    start,
                    copy_start,
                    previous_end,
                ) else {
                    continue;
                };
                let old0 = haplotypes
                    .allele(align_marker, sample * 2)
                    .map_or(-1, i32::from);
                let old1 = haplotypes
                    .allele(align_marker, sample * 2 + 1)
                    .map_or(-1, i32::from);
                let local_marker = align_marker - start;
                let new0 = forward.allele(local_marker, sample * 2);
                let new1 = forward.allele(local_marker, sample * 2 + 1);
                swap_samples[sample] = old0 == new1 && old1 == new0;
            }
        }
        for marker in copy_start..end {
            let local_marker = marker - start;
            for sample in 0..n_samples {
                let swap = swap_samples[sample];
                for copy in 0..2 {
                    let source_copy = copy ^ usize::from(swap);
                    let haplotype = sample * 2 + copy;
                    let source_haplotype = sample * 2 + source_copy;
                    let allele = forward.allele(local_marker, source_haplotype);
                    haplotypes.set_allele(
                        marker,
                        haplotype,
                        (allele >= 0).then_some(allele as u16),
                    )?;
                }
            }
        }
        previous_end = end;
    }
    stitch_elapsed += stitch_started.map_or(std::time::Duration::ZERO, |start| start.elapsed());

    let rows_started = profile.then(Instant::now);
    let rows = pool.install(|| {
        (0..n_markers)
            .into_par_iter()
            .map(|marker| {
                let row = &window.rows[marker];
                let calls = DiploidCalls::from_iter((0..n_samples).map(|sample| DiploidCall {
                    alleles: [
                        haplotypes.allele(marker, sample * 2),
                        haplotypes.allele(marker, sample * 2 + 1),
                    ],
                    phased: true,
                }));
                PhasedRow {
                    meta: row.meta.clone(),
                    calls,
                    dosage: None,
                    ap1: None,
                    ap2: None,
                    gp: None,
                    info_af: None,
                    info_dr2: None,
                    imputed: false,
                }
            })
            .collect::<Vec<_>>()
    });
    if let Some(started) = rows_started {
        let rows_elapsed = started.elapsed();
        if let Some(total_started) = profile_started {
            eprintln!(
                "JANUSX_IMP_INITIAL_PHASE_PROFILE markers={} samples={} chunks={} threads={} phase_s={:.6} reverse_thread_s={:.6} forward_thread_s={:.6} stitch_s={:.6} rows_s={:.6} total_s={:.6}",
                n_markers,
                n_samples,
                chunk_ranges.len(),
                n_threads,
                phase_elapsed.as_secs_f64(),
                reverse_elapsed.as_secs_f64(),
                forward_elapsed.as_secs_f64(),
                stitch_elapsed.as_secs_f64(),
                rows_elapsed.as_secs_f64(),
                total_started.elapsed().as_secs_f64(),
            );
        }
    }
    if std::env::var("JANUSX_TRACE_INITIAL_PHASE_WINDOW")
        .is_ok_and(|target| target == window.index.to_string())
    {
        for row in &rows {
            let calls = row
                .calls
                .iter()
                .map(|call| call.alleles)
                .collect::<Vec<_>>();
            eprintln!(
                "JANUSX_INIT_GT\tWINDOW={}\tPOS={}\tCALLS={calls:?}",
                window.index, row.meta.pos
            );
        }
    }
    Ok(PhasedWindow {
        chrom: window.chrom.clone(),
        index: window.index,
        rows,
        emit_start: window.emit_start,
        emit_end: window.emit_end,
    })
}

struct InitialPhaseFragment {
    start: usize,
    end: usize,
    forward: PhaseTrack,
    reverse_elapsed: std::time::Duration,
    forward_elapsed: std::time::Duration,
}

#[allow(clippy::too_many_arguments)]
fn initial_phase_fragment(
    window: &PhaseWindow,
    seed: u64,
    chunk_index: usize,
    start: usize,
    end: usize,
    phased_overlap: usize,
    rephase_heterozygotes: bool,
    n_samples: usize,
    n_target_haplotypes: usize,
    n_panel_haplotypes: usize,
    reference: Option<&PackedHaplotypes>,
    width: u8,
    profile: bool,
) -> Result<InitialPhaseFragment, String> {
    let chunk_markers = end - start;
    let mut reverse = PhaseTrack::new(n_target_haplotypes, chunk_markers, width)?;
    let mut reverse_rec = PbwtRecPhaser::new(
        n_samples,
        n_target_haplotypes,
        n_panel_haplotypes,
        rephase_heterozygotes,
        reference,
    );
    let mut reverse_random = JavaRandom::new(seed.wrapping_add(chunk_index as u64));
    let mut previous_marker = None;
    let reverse_started = profile.then(Instant::now);
    for marker in (start..end).rev() {
        let cdf =
            reverse_rec.phase_marker(window, previous_marker, marker, phased_overlap, "reverse")?;
        if marker >= phased_overlap {
            reverse_rec.finish_reverse(&cdf, &mut reverse_random);
        }
        reverse.store_marker(marker - start, &reverse_rec.alleles[..n_target_haplotypes])?;
        previous_marker = Some(marker);
    }
    let reverse_elapsed =
        reverse_started.map_or(std::time::Duration::ZERO, |started| started.elapsed());

    let mut forward = PhaseTrack::new(n_target_haplotypes, chunk_markers, width)?;
    let mut forward_rec = PbwtRecPhaser::new(
        n_samples,
        n_target_haplotypes,
        n_panel_haplotypes,
        rephase_heterozygotes,
        reference,
    );
    let mut last_het = vec![None; n_samples];
    previous_marker = None;
    let forward_started = profile.then(Instant::now);
    for marker in start..end {
        forward_rec.phase_marker(window, previous_marker, marker, phased_overlap, "forward")?;
        if marker >= phased_overlap {
            forward_rec.finish_forward(
                marker - start,
                n_target_haplotypes,
                &reverse,
                &forward,
                &last_het,
            );
        }
        forward.store_marker(marker - start, &forward_rec.alleles[..n_target_haplotypes])?;
        for sample in 0..n_samples {
            let hap0 = sample * 2;
            let call = window.rows[marker].calls.get(sample).unwrap();
            if call.is_complete() && forward_rec.alleles[hap0] != forward_rec.alleles[hap0 + 1] {
                last_het[sample] = Some(marker - start);
            }
        }
        previous_marker = Some(marker);
    }
    let forward_elapsed =
        forward_started.map_or(std::time::Duration::ZERO, |started| started.elapsed());
    Ok(InitialPhaseFragment {
        start,
        end,
        forward,
        reverse_elapsed,
        forward_elapsed,
    })
}

fn beagle_initial_pbwt_ranges(
    genetic_positions: &[f64],
    logical_chunks: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if genetic_positions.is_empty() || logical_chunks == 0 {
        return Err(
            "initial PBWT ranges require markers and at least one logical chunk".to_owned(),
        );
    }
    let marker_count = genetic_positions.len();
    let total_cm = genetic_positions[marker_count - 1] - genetic_positions[0];
    let advance_cm = (total_cm / logical_chunks as f64).max(2.0);
    let mut ranges = Vec::with_capacity(logical_chunks);
    let mut start = 0usize;
    let mut end = initial_pbwt_to(genetic_positions, genetic_positions[0] + advance_cm);
    while end < marker_count {
        ranges.push((start, end));
        start = genetic_positions.partition_point(|&cm| cm < genetic_positions[end] - 0.5);
        let next_end = initial_pbwt_to(genetic_positions, genetic_positions[end] + advance_cm);
        if next_end <= end {
            return Err("initial PBWT window partition did not advance".to_owned());
        }
        end = next_end;
    }
    ranges.push((start, marker_count));
    Ok(ranges)
}

pub(crate) fn initial_pbwt_max_chunk_markers(
    cm_positions: &[f64],
    n_threads: usize,
) -> Result<usize, String> {
    beagle_initial_pbwt_ranges(cm_positions, initial_pbwt_partition_count(n_threads)).map(
        |ranges| {
            ranges
                .iter()
                .map(|(start, end)| end - start)
                .max()
                .unwrap_or(0)
        },
    )
}

fn initial_pbwt_to(genetic_positions: &[f64], position: f64) -> usize {
    genetic_positions.partition_point(|&cm| cm <= position)
}

fn first_heterozygote_by_sample(window: &PhaseWindow, n_samples: usize) -> Vec<Option<usize>> {
    let mut first_hets = vec![None; n_samples];
    for (marker, row) in window.rows.iter().enumerate() {
        for sample in 0..n_samples {
            let call = row.calls.get(sample).unwrap();
            if call.is_heterozygous() && first_hets[sample].is_none() {
                first_hets[sample] = Some(marker);
            }
        }
    }
    first_hets
}

fn initial_alignment_het(
    window: &PhaseWindow,
    sample: usize,
    first_het: Option<usize>,
    phased_overlap: usize,
    start: usize,
    copy_start: usize,
    overlap_end: usize,
) -> Option<usize> {
    let first_het = first_het?;
    let mut previous = None;
    for marker in start.max(phased_overlap)..overlap_end {
        if marker <= first_het
            || !window.rows[marker]
                .calls
                .get(sample)
                .unwrap()
                .is_heterozygous()
        {
            continue;
        }
        if marker >= copy_start {
            return Some(marker);
        }
        previous = Some(marker);
    }
    previous
}

// The PBWT permutation contains target and fixed reference haplotypes; only
// the target sample prefix is eligible for phase or missing-allele updates.
struct PbwtRecPhaser<'a> {
    n_target_samples: usize,
    n_target_haplotypes: usize,
    rephase_heterozygotes: bool,
    order: Vec<usize>,
    inverse_order: Vec<usize>,
    next_order: Vec<usize>,
    allele_counts: Vec<u32>,
    alleles: Vec<i32>,
    unphased_hets: Vec<bool>,
    reference: Option<&'a PackedHaplotypes>,
}

impl<'a> PbwtRecPhaser<'a> {
    fn new(
        n_target_samples: usize,
        n_target_haplotypes: usize,
        n_panel_haplotypes: usize,
        rephase_heterozygotes: bool,
        reference: Option<&'a PackedHaplotypes>,
    ) -> Self {
        let n_haplotypes = n_panel_haplotypes;
        let order = (0..n_haplotypes).collect::<Vec<_>>();
        Self {
            n_target_samples,
            n_target_haplotypes,
            rephase_heterozygotes,
            inverse_order: order.clone(),
            next_order: order.clone(),
            order,
            allele_counts: Vec::new(),
            alleles: vec![-1; n_haplotypes],
            unphased_hets: vec![false; n_target_samples],
            reference,
        }
    }

    fn phase_marker(
        &mut self,
        window: &PhaseWindow,
        current_marker: Option<usize>,
        marker: usize,
        phased_overlap: usize,
        direction: &str,
    ) -> Result<Vec<usize>, String> {
        if let Some(previous) = current_marker {
            self.update_order(
                window.rows[previous].meta.allele_count(),
                previous,
                direction,
            )?;
        }
        let row = &window.rows[marker];
        let n_alleles = row.meta.allele_count();
        let mut allele_counts = vec![0usize; n_alleles];
        for sample in 0..self.n_target_samples {
            let call = row.calls.get(sample).unwrap();
            for copy in 0..2 {
                let haplotype = sample * 2 + copy;
                self.alleles[haplotype] = match call.alleles[copy] {
                    Some(allele) => {
                        let allele_index = usize::from(allele);
                        let Some(count) = allele_counts.get_mut(allele_index) else {
                            return Err(format!(
                                "allele index {allele} exceeds marker allele count"
                            ));
                        };
                        *count += 1;
                        i32::from(allele)
                    }
                    None => -1,
                };
            }
            self.unphased_hets[sample] =
                self.rephase_heterozygotes && marker >= phased_overlap && call.is_heterozygous();
        }
        if let Some(reference) = self.reference {
            for reference_haplotype in 0..reference.n_haplotypes {
                let allele = reference
                    .allele(marker, reference_haplotype)
                    .ok_or_else(|| {
                        format!(
                            "initial PBWT reference allele is missing at marker {} haplotype {}",
                            marker + 1,
                            reference_haplotype + 1
                        )
                    })?;
                let allele_index = usize::from(allele);
                let Some(count) = allele_counts.get_mut(allele_index) else {
                    return Err(format!(
                        "reference allele {allele} exceeds marker allele count"
                    ));
                };
                *count += 1;
                self.alleles[self.n_target_haplotypes + reference_haplotype] = i32::from(allele);
            }
        }
        let mut cumulative = 0usize;
        for count in &mut allele_counts {
            cumulative += *count;
            *count = cumulative;
        }
        for (position, &haplotype) in self.order.iter().enumerate() {
            self.inverse_order[haplotype] = position;
        }
        if marker >= phased_overlap {
            self.phase_current_marker();
        }
        Ok(allele_counts)
    }

    fn update_order(
        &mut self,
        n_alleles: usize,
        marker: usize,
        direction: &str,
    ) -> Result<(), String> {
        if self.allele_counts.len() < n_alleles + 1 {
            self.allele_counts.resize(n_alleles + 1, 0);
        }
        self.allele_counts[..=n_alleles].fill(0);
        for &haplotype in &self.order {
            let allele = self.alleles[haplotype];
            if allele < 0 || allele as usize >= n_alleles {
                return Err(format!(
                    "PBWT {direction} prefix contains allele {allele} for haplotype {haplotype} at marker {} with {n_alleles} alleles",
                    marker + 1
                ));
            }
            self.allele_counts[allele as usize + 1] += 1;
        }
        for allele in 1..=n_alleles {
            self.allele_counts[allele] += self.allele_counts[allele - 1];
        }
        for &haplotype in &self.order {
            let allele = self.alleles[haplotype] as usize;
            let destination = self.allele_counts[allele] as usize;
            self.allele_counts[allele] += 1;
            self.next_order[destination] = haplotype;
        }
        std::mem::swap(&mut self.order, &mut self.next_order);
        for (position, &haplotype) in self.order.iter().enumerate() {
            self.inverse_order[haplotype] = position;
        }
        Ok(())
    }

    fn phase_current_marker(&mut self) {
        let mut threshold = 2i32;
        let mut change_made = true;
        while threshold > 0 || change_made {
            change_made = false;
            for sample in 0..self.n_target_samples {
                if self.unphased_hets[sample] {
                    change_made |= self.phase_heterozygote(sample, threshold);
                } else {
                    for copy in 0..2 {
                        let haplotype = sample * 2 + copy;
                        if self.alleles[haplotype] < 0 {
                            self.alleles[haplotype] = self.impute_from_neighbors(haplotype);
                            change_made |= self.alleles[haplotype] >= 0;
                        }
                    }
                }
            }
            if !change_made {
                threshold -= 1;
            }
        }
    }

    fn phase_heterozygote(&mut self, sample: usize, threshold: i32) -> bool {
        let hap0 = sample * 2;
        let hap1 = hap0 + 1;
        let allele0 = self.alleles[hap0];
        let allele1 = self.alleles[hap1];
        let count =
            self.phase_count(hap0, allele0, allele1) + self.phase_count(hap1, allele1, allele0);
        if count >= threshold {
            self.unphased_hets[sample] = false;
            true
        } else if count <= -threshold {
            self.alleles.swap(hap0, hap1);
            self.unphased_hets[sample] = false;
            true
        } else {
            false
        }
    }

    fn phase_count(&self, haplotype: usize, allele: i32, other_allele: i32) -> i32 {
        let position = self.inverse_order[haplotype];
        let mut count = 0;
        if position > 0 {
            count += self.phase_count_neighbor(self.order[position - 1], allele, other_allele);
        }
        if position + 1 < self.order.len() {
            count += self.phase_count_neighbor(self.order[position + 1], allele, other_allele);
        }
        count
    }

    fn phase_count_neighbor(&self, neighbor: usize, allele: i32, other_allele: i32) -> i32 {
        if self.is_unphased_target_haplotype(neighbor) {
            return 0;
        }
        match self.alleles[neighbor] {
            adjacent if adjacent == allele => 1,
            adjacent if adjacent == other_allele => -1,
            _ => 0,
        }
    }

    fn impute_from_neighbors(&self, haplotype: usize) -> i32 {
        let position = self.inverse_order[haplotype];
        let previous = position
            .checked_sub(1)
            .map(|index| self.order[index])
            .and_then(|neighbor| self.resolved_neighbor_allele(neighbor));
        let next = self
            .order
            .get(position + 1)
            .copied()
            .and_then(|neighbor| self.resolved_neighbor_allele(neighbor));
        match (previous, next) {
            (Some(left), Some(right)) if left == right => left,
            (Some(left), None) => left,
            (None, Some(right)) => right,
            _ => -1,
        }
    }

    fn resolved_neighbor_allele(&self, neighbor: usize) -> Option<i32> {
        (!self.is_unphased_target_haplotype(neighbor))
            .then_some(self.alleles[neighbor])
            .filter(|&allele| allele >= 0)
    }

    #[inline]
    fn is_unphased_target_haplotype(&self, haplotype: usize) -> bool {
        haplotype < self.n_target_haplotypes && self.unphased_hets[haplotype / 2]
    }

    fn finish_reverse(&mut self, allele_cdf: &[usize], random: &mut JavaRandom) {
        for sample in 0..self.n_target_samples {
            let hap0 = sample * 2;
            let hap1 = hap0 + 1;
            if self.unphased_hets[sample] {
                if random.next_boolean() {
                    self.alleles.swap(hap0, hap1);
                }
                self.unphased_hets[sample] = false;
            } else {
                for haplotype in [hap0, hap1] {
                    if self.alleles[haplotype] < 0 {
                        self.alleles[haplotype] = draw_allele_from_cdf(allele_cdf, random);
                    }
                }
            }
        }
    }

    fn finish_forward(
        &mut self,
        marker: usize,
        n_haplotypes: usize,
        reverse: &PhaseTrack,
        forward: &PhaseTrack,
        last_het: &[Option<usize>],
    ) {
        for sample in 0..self.n_target_samples {
            let hap0 = sample * 2;
            let hap1 = hap0 + 1;
            if self.unphased_hets[sample] {
                if let Some(previous_het) = last_het[sample] {
                    let reverse_same = (reverse.allele(previous_het, hap0)
                        < reverse.allele(previous_het, hap1))
                        == (reverse.allele(marker, hap0) < reverse.allele(marker, hap1));
                    let forward_same = (forward.allele(previous_het, hap0)
                        < forward.allele(previous_het, hap1))
                        == (self.alleles[hap0] < self.alleles[hap1]);
                    if reverse_same != forward_same {
                        self.alleles.swap(hap0, hap1);
                    }
                }
                self.unphased_hets[sample] = false;
            } else {
                for haplotype in [hap0, hap1] {
                    if self.alleles[haplotype] < 0 {
                        self.alleles[haplotype] = impute_from_reverse(
                            marker,
                            haplotype,
                            last_het[sample],
                            n_haplotypes,
                            reverse,
                            forward,
                        );
                    }
                }
            }
        }
    }
}

fn draw_allele_from_cdf(cdf: &[usize], random: &mut JavaRandom) -> i32 {
    let total = cdf.last().copied().unwrap_or(0);
    if total == 0 {
        return 0;
    }
    let draw = random.next_int(total);
    cdf.iter()
        .position(|&cumulative| draw < cumulative)
        .unwrap_or(cdf.len().saturating_sub(1)) as i32
}

enum PhaseTrack {
    Packed {
        haplotypes: PackedHaplotypes,
        marker_words: Vec<u64>,
    },
    Scalar {
        values: Vec<u16>,
        n_haplotypes: usize,
    },
}

impl PhaseTrack {
    fn new(n_haplotypes: usize, n_markers: usize, bits_per_allele: u8) -> Result<Self, String> {
        if bits_per_allele == 1 {
            return Ok(Self::Packed {
                haplotypes: PackedHaplotypes::empty(n_haplotypes, n_markers, 1)?,
                marker_words: vec![0; n_haplotypes.div_ceil(64)],
            });
        }
        let values = n_haplotypes
            .checked_mul(n_markers)
            .ok_or_else(|| "initial phase track size overflow".to_owned())?;
        Ok(Self::Scalar {
            values: vec![u16::MAX; values],
            n_haplotypes,
        })
    }

    fn store_marker(&mut self, marker: usize, alleles: &[i32]) -> Result<(), String> {
        match self {
            Self::Packed {
                haplotypes,
                marker_words,
            } => {
                marker_words.fill(0);
                for (haplotype, &allele) in alleles.iter().enumerate() {
                    if allele < 0 {
                        haplotypes.missing_bits[haplotype >> 6] |= 1u64 << (haplotype & 63);
                    } else if allele == 1 {
                        marker_words[haplotype >> 6] |= 1u64 << (haplotype & 63);
                    } else if allele > 1 {
                        return Err(format!(
                            "biallelic phase track contains allele index {allele}"
                        ));
                    }
                }
                haplotypes.set_biallelic_marker_words(marker, marker_words)
            }
            Self::Scalar {
                values,
                n_haplotypes,
            } => {
                if alleles.len() != *n_haplotypes {
                    return Err("initial phase track haplotype count differs".to_owned());
                }
                let start = marker
                    .checked_mul(*n_haplotypes)
                    .ok_or_else(|| "initial phase track offset overflow".to_owned())?;
                let destination = values
                    .get_mut(start..start + *n_haplotypes)
                    .ok_or_else(|| "initial phase track marker is out of range".to_owned())?;
                for (destination, &allele) in destination.iter_mut().zip(alleles) {
                    *destination = if allele < 0 {
                        u16::MAX
                    } else {
                        u16::try_from(allele)
                            .map_err(|_| "initial phase allele exceeds u16".to_owned())?
                    };
                }
                Ok(())
            }
        }
    }

    #[inline]
    fn allele(&self, marker: usize, haplotype: usize) -> i32 {
        match self {
            Self::Packed { haplotypes, .. } => {
                let index = marker * haplotypes.n_haplotypes + haplotype;
                if (haplotypes.missing_bits[index >> 6] >> (index & 63)) & 1 != 0 {
                    -1
                } else {
                    ((haplotypes.allele_bits[index >> 6] >> (index & 63)) & 1) as i32
                }
            }
            Self::Scalar {
                values,
                n_haplotypes,
            } => {
                let allele = values[marker * *n_haplotypes + haplotype];
                if allele == u16::MAX {
                    -1
                } else {
                    i32::from(allele)
                }
            }
        }
    }
}

fn impute_from_reverse(
    marker: usize,
    haplotype: usize,
    previous_het: Option<usize>,
    _n_haplotypes: usize,
    reverse: &PhaseTrack,
    forward: &PhaseTrack,
) -> i32 {
    let Some(previous_het) = previous_het else {
        return reverse.allele(marker, haplotype);
    };
    let complement = haplotype ^ 1;
    let reverse_same =
        reverse.allele(previous_het, haplotype) < reverse.allele(previous_het, complement);
    let forward_same =
        forward.allele(previous_het, haplotype) < forward.allele(previous_het, complement);
    let source_haplotype = if reverse_same == forward_same {
        haplotype
    } else {
        complement
    };
    reverse.allele(marker, source_haplotype)
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

pub(crate) fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{
        beagle_initial_pbwt_ranges, best_low_frequency_donor_rank,
        build_low_frequency_phase_donor_index, build_low_frequency_phase_donor_index_view,
        build_phase_donor_index, build_phase_donor_index_with_pool_view, build_segment_index,
        build_segment_index_from_alleles, encode_biallelic_step_sequences,
        encode_biallelic_step_sequences_view, encode_step_sequences, first_heterozygote_by_sample,
        initial_alignment_het, initialize, initialize_with_reference_pool, initialize_with_threads,
        low_frequency_hap_neighbors, low_frequency_hap_neighbors_view, phase_pbwt_batch_ranges,
        phase_step_ranges, segment_exact_neighbors, segment_neighbors_from_alleles,
        DonorProposalStorage, PbwtDirection, StepSequenceScratch,
    };
    use crate::imp::composite::PhaseHaplotypeView;
    use crate::imp::types::{
        DiploidCall, DiploidCalls, PackedHaplotypes, PhaseWindow, PhasedRow, PhasedWindow,
        VariantMeta, VariantRow,
    };

    fn synthetic_window() -> (PhaseWindow, PhasedWindow) {
        let haplotypes = [
            [0u16, 0, 0], // target sample haplotype 0
            [0u16, 0, 1], // target sample haplotype 1
            [0u16, 0, 0], // exact donor for target haplotype 0
            [0u16, 1, 0],
            [0u16, 0, 0], // duplicate exact donor for stable-order coverage
            [1u16, 0, 0],
            [1u16, 1, 1],
            [0u16, 1, 1],
        ];
        let rows = (0..3)
            .map(|marker| {
                let calls = (0..4)
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
        let phase = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: rows.clone(),
            cm_positions: vec![0.0, 0.01, 0.02],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: rows.len(),
        };
        let phased = PhasedWindow {
            chrom: phase.chrom.clone(),
            index: phase.index,
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
            emit_end: 3,
        };
        (phase, phased)
    }

    fn phased_heterozygote_window() -> PhaseWindow {
        let rows = (0..24)
            .map(|marker| {
                let calls = (0..4)
                    .map(|sample| DiploidCall {
                        alleles: if (sample + marker) % 2 == 0 {
                            [Some(0), Some(1)]
                        } else {
                            [Some(1), Some(0)]
                        },
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
        PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            cm_positions: (0..24).map(|marker| marker as f64).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 24,
        }
    }

    #[test]
    fn low_frequency_carrier_neighbors_follow_pbwt_order_and_join_both_haps() {
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: 1,
                    id: "rare".to_owned(),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: false,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: false,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: false,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: false,
                    },
                ]),
            }],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let order = [2usize, 6, 0, 7, 3, 1, 5, 4];
        let mut inverse = vec![0usize; order.len()];
        for (rank, &haplotype) in order.iter().enumerate() {
            inverse[haplotype] = rank;
        }

        let (previous, next) = low_frequency_hap_neighbors(&input, 0, 1, &inverse, 3).unwrap();

        assert_eq!(
            previous,
            vec![None, Some(0), None, Some(1), Some(3), None, None, None]
        );
        assert_eq!(
            next,
            vec![Some(1), Some(3), None, Some(4), None, None, None, None]
        );
    }

    #[test]
    fn low_frequency_neighbors_use_original_target_carriers() {
        let meta = VariantMeta {
            chrom: "1".to_owned(),
            pos: 1,
            id: "rare".to_owned(),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let original = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: meta.clone(),
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                    DiploidCall::missing(false),
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: true,
                    },
                ]),
            }],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let phased = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta,
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(1), Some(0)],
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
        let view = super::super::composite::PhaseHaplotypeView::target(&phased);

        let (previous, next) = low_frequency_hap_neighbors_view(
            &view,
            Some(&original),
            0,
            1,
            &(0..8).collect::<Vec<_>>(),
            2,
        )
        .unwrap();

        assert_eq!(
            previous,
            vec![None, None, None, None, None, Some(4), Some(5), Some(6)]
        );
        assert_eq!(
            next,
            vec![None, None, None, None, Some(5), Some(6), Some(7), None]
        );
    }

    #[test]
    fn low_frequency_donor_choice_uses_directional_divergence_and_carrier_edges() {
        let order = [0usize, 2, 4, 6, 8, 10];
        let mut previous = vec![None; order.len()];
        let mut next = vec![None; order.len()];
        previous[2] = Some(0);
        next[2] = Some(4);

        let forward_divergence = [0, 1, 0, 1, 2, 0, 4];
        let forward = best_low_frequency_donor_rank(
            PbwtDirection::Forward,
            3,
            5,
            2,
            &order,
            &forward_divergence,
            &previous,
            &next,
            2,
            2,
            (0, 1),
            &[],
        );
        assert_eq!(forward, Some(0));

        let backward_divergence = [4, 3, 4, 3, 2, 2, 1];
        let backward = best_low_frequency_donor_rank(
            PbwtDirection::Reverse,
            2,
            5,
            2,
            &order,
            &backward_divergence,
            &previous,
            &next,
            2,
            2,
            (0, 1),
            &[],
        );
        assert_eq!(backward, Some(0));
    }

    #[test]
    fn low_frequency_donor_index_uses_explicit_map_and_excludes_target_sample() {
        let (input, phased) = synthetic_window();
        let marker_indices = [0usize, 1, 2];
        let index = build_low_frequency_phase_donor_index(
            &phased,
            &input.cm_positions,
            &input,
            &marker_indices,
            0.01,
            3,
            10,
            30,
            &[],
            19,
        )
        .unwrap();

        assert_eq!(index.step_ranges, [(0, 1), (1, 2), (2, 3)]);
        for direction in [PbwtDirection::Forward, PbwtDirection::Reverse] {
            for step in 0..index.step_ranges.len() {
                for target_haplotype in 0..8 {
                    if let Some(donor) = index.donor_haplotype(direction, step, target_haplotype) {
                        assert_ne!(donor / 2, target_haplotype / 2);
                    }
                }
            }
        }
    }

    #[test]
    fn low_frequency_donor_view_includes_packed_reference_haplotypes() {
        let meta = |marker: usize| VariantMeta {
            chrom: "1".to_owned(),
            pos: marker as u64 + 1,
            id: format!("v{marker}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: [Some(0), Some(1)]
                .into_iter()
                .enumerate()
                .map(|(marker, allele)| PhasedRow {
                    meta: meta(marker),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [allele, Some(u16::from(marker == 0))],
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
            emit_end: 2,
        };
        let mut reference = PackedHaplotypes::empty_state_major(2, 2, 1).unwrap();
        reference.set_allele(0, 0, Some(1)).unwrap();
        reference.set_allele(0, 1, Some(1)).unwrap();
        reference.set_allele(1, 0, Some(0)).unwrap();
        reference.set_allele(1, 1, Some(0)).unwrap();
        reference.mark_complete();
        let view =
            super::super::composite::PhaseHaplotypeView::new(&target, None, Some(&reference), 1)
                .unwrap();
        let index = build_low_frequency_phase_donor_index_view(
            &view,
            &view,
            None,
            &[0.0, 0.01],
            &[0, 1],
            0.01,
            3,
            4,
            1,
            &[],
            19,
        )
        .unwrap();

        assert_eq!(index.n_haplotypes, 4);
        let mut donor_count = 0usize;
        for direction in [PbwtDirection::Forward, PbwtDirection::Reverse] {
            for step in 0..index.step_ranges.len() {
                for target_haplotype in 0..2 {
                    if let Some(donor) = index.donor_haplotype(direction, step, target_haplotype) {
                        assert!(donor >= 2);
                        donor_count += 1;
                    }
                }
                assert_eq!(index.donor_haplotype(direction, step, 2), None);
                assert_eq!(index.donor_haplotype(direction, step, 3), None);
            }
        }
        assert!(donor_count > 0);
    }

    #[test]
    fn packed_view_step_codes_match_materialized_target_and_reference() {
        let markers = 70usize;
        let target = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: (0..markers)
                .map(|marker| {
                    let calls = [
                        match marker % 7 {
                            0 => DiploidCall::missing(false),
                            1 => DiploidCall {
                                alleles: [Some(0), Some(1)],
                                phased: false,
                            },
                            2 => DiploidCall {
                                alleles: [Some(1), Some(0)],
                                phased: true,
                            },
                            value => DiploidCall {
                                alleles: [Some((value & 1) as u16); 2],
                                phased: false,
                            },
                        },
                        match marker % 5 {
                            0 => DiploidCall {
                                alleles: [Some(0), Some(1)],
                                phased: true,
                            },
                            value => DiploidCall {
                                alleles: [Some((value & 1) as u16); 2],
                                phased: false,
                            },
                        },
                    ];
                    PhasedRow {
                        meta: VariantMeta {
                            chrom: "1".to_owned(),
                            pos: marker as u64 + 1,
                            id: format!("v{marker}"),
                            ref_allele: "A".to_owned(),
                            alt_alleles: vec!["G".to_owned()],
                        },
                        calls: DiploidCalls::from_calls(calls.to_vec()),
                        dosage: None,
                        ap1: None,
                        ap2: None,
                        gp: None,
                        info_af: None,
                        info_dr2: None,
                        imputed: false,
                    }
                })
                .collect(),
            emit_start: 0,
            emit_end: markers,
        };
        let mut reference = PackedHaplotypes::empty_state_major(4, markers, 1).unwrap();
        for marker in 0..markers {
            for haplotype in 0..4 {
                let allele = Some(((marker + haplotype * 3) & 1) as u16);
                reference.set_allele(marker, haplotype, allele).unwrap();
            }
        }
        reference.mark_complete();
        let view =
            super::super::composite::PhaseHaplotypeView::new(&target, None, Some(&reference), 2)
                .unwrap();
        let materialized = PhasedWindow {
            chrom: target.chrom.clone(),
            index: target.index,
            rows: target
                .rows
                .iter()
                .enumerate()
                .map(|(marker, row)| {
                    let mut calls = row.calls.iter().collect::<Vec<_>>();
                    calls.extend((0..2).map(|sample| DiploidCall {
                        alleles: [
                            reference.allele(marker, sample * 2),
                            reference.allele(marker, sample * 2 + 1),
                        ],
                        phased: true,
                    }));
                    PhasedRow {
                        meta: row.meta.clone(),
                        calls: DiploidCalls::from_calls(calls),
                        dosage: None,
                        ap1: None,
                        ap2: None,
                        gp: None,
                        info_af: None,
                        info_dr2: None,
                        imputed: false,
                    }
                })
                .collect(),
            emit_start: target.emit_start,
            emit_end: target.emit_end,
        };
        let mut scalar_scratch = StepSequenceScratch::default();
        let mut packed_scratch = StepSequenceScratch::default();
        let mut scalar_codes = Vec::new();
        let mut packed_codes = Vec::new();
        let scalar_count = encode_biallelic_step_sequences(
            &materialized,
            0,
            markers,
            &mut scalar_codes,
            &mut scalar_scratch,
        )
        .unwrap();
        let packed_count = encode_biallelic_step_sequences_view(
            &view,
            0,
            markers,
            &mut packed_codes,
            &mut packed_scratch,
        )
        .unwrap();

        assert_eq!(packed_count, scalar_count);
        assert_eq!(packed_codes, scalar_codes);
    }

    #[test]
    fn phase_donor_proposals_are_stored_only_for_target_haplotypes() {
        let (_, target) = synthetic_window();
        let mut reference = PackedHaplotypes::empty_state_major(4, 3, 1).unwrap();
        for marker in 0..3 {
            for haplotype in 0..4 {
                reference
                    .set_allele(marker, haplotype, Some(((marker + haplotype) & 1) as u16))
                    .unwrap();
            }
        }
        reference.mark_complete();
        let view = PhaseHaplotypeView::new(&target, None, Some(&reference), 4).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let index = build_phase_donor_index_with_pool_view(
            &view,
            &[0.0, 1.0, 2.0],
            1.0,
            PbwtDirection::Reverse,
            8,
            &[],
            &pool,
            91,
        )
        .unwrap();

        assert_eq!(index.n_haplotypes, 12);
        assert_eq!(index.donor_haplotypes.len(), index.step_ranges.len() * 8);
        assert!(index.step_ranges.iter().enumerate().all(|(step, _)| {
            (8..12).all(|haplotype| index.donor_haplotype(step, haplotype).is_none())
        }));
    }

    fn phased_window_from_haplotypes(haplotypes: &[[u16; 4]; 8]) -> PhasedWindow {
        let rows = (0..4)
            .map(|marker| {
                let calls = (0..4)
                    .map(|sample| DiploidCall {
                        alleles: [
                            Some(haplotypes[2 * sample][marker]),
                            Some(haplotypes[2 * sample + 1][marker]),
                        ],
                        phased: true,
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
                    calls: DiploidCalls::from_calls(calls),
                    dosage: None,
                    ap1: None,
                    ap2: None,
                    gp: None,
                    info_af: None,
                    info_dr2: None,
                    imputed: false,
                }
            })
            .collect();
        PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            emit_start: 0,
            emit_end: 4,
        }
    }

    fn phased_window_from_sequences(sequences: &[Vec<Option<u16>>]) -> PhasedWindow {
        let marker_count = sequences[0].len();
        let n_samples = sequences.len() / 2;
        let rows = (0..marker_count)
            .map(|marker| {
                let calls = (0..n_samples)
                    .map(|sample| DiploidCall {
                        alleles: [
                            sequences[2 * sample][marker],
                            sequences[2 * sample + 1][marker],
                        ],
                        phased: true,
                    })
                    .collect();
                PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(calls),
                    dosage: None,
                    ap1: None,
                    ap2: None,
                    gp: None,
                    info_af: None,
                    info_dr2: None,
                    imputed: false,
                }
            })
            .collect();
        PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            emit_start: 0,
            emit_end: marker_count,
        }
    }

    fn scalar_step_codes(window: &PhasedWindow, start: usize, end: usize) -> (usize, Vec<u32>) {
        let n_haplotypes = window.rows[start].calls.len() * 2;
        let mut sequences = Vec::<Vec<Option<u16>>>::new();
        let mut codes = Vec::with_capacity(n_haplotypes);
        for haplotype in 0..n_haplotypes {
            let sequence = (start..end)
                .map(|marker| {
                    window.rows[marker]
                        .calls
                        .get(haplotype / 2)
                        .unwrap()
                        .alleles[haplotype & 1]
                })
                .collect::<Vec<_>>();
            let code =
                if let Some(code) = sequences.iter().position(|existing| *existing == sequence) {
                    code as u32
                } else {
                    let code = sequences.len() as u32;
                    sequences.push(sequence);
                    code
                };
            codes.push(code);
        }
        (sequences.len(), codes)
    }

    #[test]
    fn forward_and_reverse_pbwt_return_stable_exact_donors_without_target_haps() {
        let (phase, phased) = synthetic_window();
        let searchable = [true; 3];
        let query = [Some(0), Some(0), Some(0)];

        for direction in [PbwtDirection::Forward, PbwtDirection::Reverse] {
            let index =
                build_segment_index(&phased, &phase.cm_positions, &searchable, 10.0, direction)
                    .unwrap();
            let donors = segment_exact_neighbors(&index, 0, &phased, &query, 0).unwrap();
            let repeated = segment_exact_neighbors(&index, 0, &phased, &query, 0).unwrap();
            assert_eq!(donors, repeated, "{direction:?} order must be stable");
            assert_eq!(
                donors
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>(),
                [2, 4].into()
            );
            assert!(donors.iter().all(|haplotype| haplotype / 2 != 0));
        }
    }

    #[test]
    fn segment_neighbors_prioritize_all_exact_haplotypes_before_mismatches() {
        let alleles = [
            [Some(0), Some(0), Some(0)],
            [Some(0), Some(0), Some(1)],
            [Some(0), Some(0), Some(1)],
            [Some(1), Some(1), Some(1)],
        ];
        let cm_positions = [0.0, 0.01, 0.02];
        let searchable = [true; 3];
        let index = build_segment_index_from_alleles(
            4,
            3,
            1,
            &cm_positions,
            &searchable,
            10.0,
            PbwtDirection::Forward,
            |marker, haplotype| alleles[haplotype][marker],
        )
        .unwrap();

        let donors = segment_neighbors_from_alleles(
            &index,
            0,
            3,
            &[Some(0), Some(0), Some(1)],
            2,
            |marker, haplotype| alleles[haplotype][marker],
        )
        .unwrap();

        assert_eq!(donors.len(), 2);
        assert!(donors
            .iter()
            .all(|&haplotype| alleles[haplotype] == [Some(0), Some(0), Some(1)]));
    }

    #[test]
    fn beagle_iteration_parity_selects_reverse_then_forward_pbwt() {
        assert_eq!(PbwtDirection::for_iteration(0), PbwtDirection::Reverse);
        assert_eq!(PbwtDirection::for_iteration(1), PbwtDirection::Forward);
        assert_eq!(PbwtDirection::for_iteration(2), PbwtDirection::Reverse);
    }

    #[test]
    fn phase_steps_use_beagle_minimum_start_distance() {
        let ranges = phase_step_ranges(&[0.0, 0.1, 0.249, 0.25, 0.499, 0.5], 0.25).unwrap();

        assert_eq!(ranges, [(0, 3), (3, 5), (5, 6)]);
    }

    #[test]
    fn phase_pbwt_batches_match_logical_batch_and_overlap_partition() {
        assert_eq!(
            phase_pbwt_batch_ranges(11, 3, 2),
            [(0, 4, 0, 6), (4, 8, 2, 10), (8, 11, 6, 11)]
        );
        assert_eq!(
            phase_pbwt_batch_ranges(3, 8, 4),
            [(0, 1, 0, 3), (1, 2, 0, 3), (2, 3, 0, 3)]
        );
    }

    #[test]
    fn phase_donor_proposals_follow_beagle_worker_schedule() {
        let (_, phased) = synthetic_window();
        let cm_positions = [0.0, 1.0, 2.0];
        let build = |threads| {
            build_phase_donor_index(
                &phased,
                &cm_positions,
                1.0,
                PbwtDirection::Reverse,
                8,
                &[],
                threads,
                91,
            )
            .unwrap()
        };

        let one = build(1);
        let eight = build(8);

        assert_eq!(one.step_ranges, eight.step_ranges);
        assert_ne!(one.donor_haplotypes, eight.donor_haplotypes);
    }

    #[test]
    fn forward_phase_donors_are_selected_from_the_exact_current_step_group() {
        let haplotypes = [
            [0u16, 0, 0, 0], // target sample 0, haplotype 0
            [1u16, 1, 1, 1], // target sample 0, haplotype 1
            [0u16, 0, 0, 1], // sample 1 matches the prefix but diverges at the last marker
            [1u16, 1, 1, 1],
            [1u16, 0, 0, 0], // sample 2 matches the last marker, not the prefix
            [0u16, 1, 1, 1],
            [0u16, 0, 0, 0], // sample 3 is the only full-prefix donor
            [1u16, 1, 1, 1],
        ];
        let window = phased_window_from_haplotypes(&haplotypes);
        let cm_positions = [0.0, 1.0, 2.0, 3.0];
        let index = build_phase_donor_index(
            &window,
            &cm_positions,
            1.0,
            PbwtDirection::Forward,
            16,
            &[],
            1,
            19,
        )
        .unwrap();

        assert_eq!(index.step_ranges, [(0, 1), (1, 2), (2, 3), (3, 4)]);
        let donor = index.donor_haplotype(3, 0).unwrap();
        assert!([4, 6].contains(&donor));
        assert_eq!(
            window.rows[3].calls.get(donor / 2).unwrap().alleles[donor & 1],
            Some(0)
        );
    }

    #[test]
    fn phase_donors_exclude_the_target_samples_other_haplotype_as_ibs2() {
        let haplotypes = [
            [0u16, 0, 0, 0], // target haplotype 0
            [0u16, 0, 0, 0], // Beagle treats both haplotypes from one sample as IBS2
            [1u16, 1, 1, 1],
            [1u16, 1, 1, 1],
            [1u16, 1, 1, 1],
            [1u16, 1, 1, 1],
            [1u16, 1, 1, 1],
            [1u16, 1, 1, 1],
        ];
        let window = phased_window_from_haplotypes(&haplotypes);
        let index = build_phase_donor_index(
            &window,
            &[0.0, 1.0, 2.0, 3.0],
            1.0,
            PbwtDirection::Forward,
            8,
            &[],
            1,
            11,
        )
        .unwrap();

        assert_eq!(index.donor_haplotype(0, 0), None);
    }

    #[test]
    fn forward_and_reverse_proposals_share_the_context_through_the_current_step() {
        let haplotypes = [
            [0u16, 0, 0, 0],
            [1u16, 1, 1, 1],
            [0u16, 0, 0, 1],
            [1u16, 1, 1, 1],
            [1u16, 0, 0, 0],
            [0u16, 1, 1, 1],
            [0u16, 0, 0, 0],
            [1u16, 1, 1, 1],
        ];
        let window = phased_window_from_haplotypes(&haplotypes);
        let cm_positions = [0.0, 1.0, 2.0, 3.0];

        for direction in [PbwtDirection::Forward, PbwtDirection::Reverse] {
            let index =
                build_phase_donor_index(&window, &cm_positions, 1.0, direction, 8, &[], 1, 101)
                    .unwrap();
            for (step, &(start, end)) in index.step_ranges.iter().enumerate() {
                for target_haplotype in 0..index.n_haplotypes {
                    let Some(donor) = index.donor_haplotype(step, target_haplotype) else {
                        continue;
                    };
                    assert_ne!(donor, target_haplotype);
                    for marker in start..end {
                        assert_eq!(
                            window.rows[marker].calls.get(donor / 2).unwrap().alleles[donor & 1],
                            window.rows[marker]
                                .calls
                                .get(target_haplotype / 2)
                                .unwrap()
                                .alleles[target_haplotype & 1],
                            "{direction:?} step {step} donor {donor} differs at marker {marker}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn donor_sampling_uses_the_java_random_sequence() {
        let mut random = super::JavaRandom::new(0);

        assert_eq!(random.next_int(100), 60);
        assert_eq!(random.next_int(100), 48);
        assert_eq!(random.next_int(100), 29);
    }

    #[test]
    fn window_seeds_match_java_random_next_long_sequence() {
        assert_eq!(super::beagle_window_seed(0, 0) as i64, -4962768465676381896);
        assert_eq!(super::beagle_window_seed(0, 1) as i64, 4437113781045784766);
        assert_eq!(super::beagle_window_seed(0, 2) as i64, -6688467811848818630);
    }

    #[test]
    fn multi_marker_steps_require_an_exact_sequence_in_both_directions() {
        let haplotypes = [
            [0u16, 0, 0, 0],
            [1u16, 1, 1, 1],
            [1u16, 0, 0, 0], // matches only the final marker of forward step 0
            [1u16, 1, 1, 1],
            [0u16, 0, 0, 1], // matches only the first marker of reverse step 1
            [1u16, 1, 1, 1],
            [1u16, 0, 1, 0],
            [1u16, 1, 1, 1],
        ];
        let window = phased_window_from_haplotypes(&haplotypes);
        let cm_positions = [0.0, 0.1, 0.25, 0.35];
        let forward = build_phase_donor_index(
            &window,
            &cm_positions,
            0.25,
            PbwtDirection::Forward,
            8,
            &[],
            1,
            5,
        )
        .unwrap();
        let reverse = build_phase_donor_index(
            &window,
            &cm_positions,
            0.25,
            PbwtDirection::Reverse,
            8,
            &[],
            1,
            5,
        )
        .unwrap();

        assert_eq!(forward.step_ranges, [(0, 2), (2, 4)]);
        assert_eq!(forward.donor_haplotype(0, 0), Some(4));
        assert_eq!(reverse.donor_haplotype(1, 0), Some(2));
    }

    #[test]
    fn step_sequence_categories_follow_first_haplotype_order() {
        // At this step, allele 1 first appears before allele 0. Beagle assigns
        // sequence codes in haplotype order, not in numeric allele order.
        let haplotypes = [
            [1u16, 0, 0, 0],
            [0u16, 1, 1, 1],
            [1u16, 1, 0, 0],
            [0u16, 0, 1, 1],
            [0u16, 0, 0, 0],
            [1u16, 1, 0, 0],
            [0u16, 1, 1, 0],
            [1u16, 0, 0, 1],
        ];
        let window = phased_window_from_haplotypes(&haplotypes);
        let mut codes = Vec::new();
        let mut scratch = StepSequenceScratch::default();
        let unique_sequences =
            encode_biallelic_step_sequences(&window, 0, 1, &mut codes, &mut scratch).unwrap();

        assert_eq!(unique_sequences, 2);
        assert_eq!(codes, [0, 1, 0, 1, 1, 0, 1, 0]);
    }

    #[test]
    fn packed_biallelic_step_codes_match_scalar_groups_across_word_boundaries() {
        for marker_count in [1usize, 31, 32, 33, 511, 512, 513, 1024, 1025] {
            let sequences = (0..12)
                .map(|haplotype| {
                    let group = haplotype % 7;
                    (0..marker_count)
                        .map(|marker| match (group * 5 + marker * 7 + marker / 3) % 3 {
                            0 => Some(0),
                            1 => Some(1),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let window = phased_window_from_sequences(&sequences);
            let (expected_unique, expected_codes) = scalar_step_codes(&window, 0, marker_count);
            let mut codes = Vec::new();
            let mut scratch = StepSequenceScratch::default();
            let observed_unique =
                encode_step_sequences(&window, 0, marker_count, &mut codes, &mut scratch);

            assert_eq!(
                observed_unique, expected_unique,
                "marker_count={marker_count}"
            );
            assert_eq!(codes, expected_codes, "marker_count={marker_count}");
            assert_eq!(
                encode_biallelic_step_sequences(
                    &window,
                    0,
                    marker_count,
                    &mut Vec::new(),
                    &mut scratch,
                )
                .is_some(),
                marker_count <= super::PACKED_STEP_SEQUENCE_MAX_MARKERS,
                "marker_count={marker_count}"
            );
        }
    }

    #[test]
    fn multiallelic_step_sequences_keep_scalar_category_codes() {
        let sequences = vec![
            vec![Some(2), Some(0), Some(1)],
            vec![Some(0), Some(1), Some(2)],
            vec![Some(2), Some(0), Some(1)],
            vec![None, Some(0), Some(1)],
            vec![Some(0), Some(1), Some(2)],
            vec![None, Some(0), Some(1)],
        ];
        let window = phased_window_from_sequences(&sequences);
        let (expected_unique, expected_codes) = scalar_step_codes(&window, 0, 3);
        let mut codes = Vec::new();
        let mut scratch = StepSequenceScratch::default();
        let observed_unique = encode_step_sequences(&window, 0, 3, &mut codes, &mut scratch);

        assert_eq!(observed_unique, expected_unique);
        assert_eq!(codes, expected_codes);
    }

    #[test]
    fn full_candidate_neighborhood_matches_direct_step_sequence_groups() {
        let cm_positions = [0.0, 0.1, 0.25, 0.35];
        for seed in 0..32u16 {
            let mut haplotypes = [[0u16; 4]; 8];
            for (haplotype, sequence) in haplotypes.iter_mut().enumerate() {
                for (marker, allele) in sequence.iter_mut().enumerate() {
                    *allele = ((seed as usize * 11 + haplotype * 7 + marker * 5) % 3) as u16;
                }
            }
            let window = phased_window_from_haplotypes(&haplotypes);
            for direction in [PbwtDirection::Forward, PbwtDirection::Reverse] {
                let index = build_phase_donor_index(
                    &window,
                    &cm_positions,
                    0.25,
                    direction,
                    window.rows[0].calls.len() * 2,
                    &[],
                    1,
                    u64::from(seed),
                )
                .unwrap();
                for (step, &(start, end)) in index.step_ranges.iter().enumerate() {
                    for target_haplotype in 0..index.n_haplotypes {
                        let has_external_exact_match = (0..index.n_haplotypes).any(|donor| {
                            donor != target_haplotype
                                && (start..end).all(|marker| {
                                    window.rows[marker].calls.get(donor / 2).unwrap().alleles
                                        [donor & 1]
                                        == window.rows[marker]
                                            .calls
                                            .get(target_haplotype / 2)
                                            .unwrap()
                                            .alleles[target_haplotype & 1]
                                })
                        });
                        let proposal = index.donor_haplotype(step, target_haplotype);
                        assert_eq!(proposal.is_some(), has_external_exact_match);
                        if let Some(donor) = proposal {
                            assert!((start..end).all(|marker| {
                                window.rows[marker].calls.get(donor / 2).unwrap().alleles[donor & 1]
                                    == window.rows[marker]
                                        .calls
                                        .get(target_haplotype / 2)
                                        .unwrap()
                                        .alleles[target_haplotype & 1]
                            }));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn initialization_preserves_input_phased_heterozygotes() {
        let row = VariantRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: 1,
                id: "v1".to_owned(),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: DiploidCalls::from_calls(vec![
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: true,
                },
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: true,
                },
            ]),
        };
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![row],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };

        for seed in 0..32 {
            let initialized = initialize(&window, seed).unwrap();
            assert_eq!(
                initialized.rows[0].calls.get(0).unwrap().alleles,
                [Some(0), Some(1)],
                "input-phased sample was reoriented at seed {seed}"
            );
        }
    }

    #[test]
    fn initialization_preserves_first_unresolved_heterozygote_orientation() {
        let row = VariantRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: 1,
                id: "v1".to_owned(),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: DiploidCalls::from_calls(vec![
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
            ]),
        };
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![row],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };

        let initialized = initialize(&window, 0).unwrap();

        for sample in 0..4 {
            assert_eq!(
                initialized.rows[0].calls.get(sample).unwrap().alleles,
                [Some(0), Some(1)],
                "Beagle leaves the first unresolved heterozygote in input orientation"
            );
        }
    }

    #[test]
    fn initial_pbwt_windows_match_beagle_cm_partition_and_overlap() {
        let positions = (0..100)
            .map(|marker| marker as f64 / 10.0)
            .collect::<Vec<_>>();

        assert_eq!(
            beagle_initial_pbwt_ranges(&positions, 4).unwrap(),
            vec![(0, 25), (20, 50), (45, 75), (70, 100)]
        );
        assert_eq!(
            beagle_initial_pbwt_ranges(&positions, 1).unwrap(),
            vec![(0, 100)]
        );
    }

    #[test]
    fn chunk_alignment_selects_the_nearest_het_inside_the_overlap() {
        let rows = (0..10)
            .map(|marker| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: if marker % 2 == 0 {
                        [Some(0), Some(1)]
                    } else {
                        [Some(0), Some(0)]
                    },
                    phased: true,
                }]),
            })
            .collect::<Vec<_>>();
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            cm_positions: (0..10).map(|marker| marker as f64).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 10,
        };
        let first_hets = first_heterozygote_by_sample(&window, 1);

        assert_eq!(
            initial_alignment_het(&window, 0, first_hets[0], 0, 3, 5, 8),
            Some(6)
        );
        assert_eq!(
            initial_alignment_het(&window, 0, first_hets[0], 0, 3, 7, 8),
            Some(6)
        );
    }

    #[test]
    fn phased_input_orientation_is_preserved_by_initialization() {
        let window = phased_heterozygote_window();

        let initialized = initialize_with_threads(&window, 17, 3).unwrap();

        for (input, output) in window.rows.iter().zip(&initialized.rows) {
            for sample in 0..4 {
                assert_eq!(
                    input.calls.get(sample).unwrap().alleles,
                    output.calls.get(sample).unwrap().alleles
                );
            }
        }
    }

    #[test]
    fn initial_phase_partition_follows_beagle_worker_schedule() {
        let mut window = phased_heterozygote_window();
        for (marker, row) in window.rows.iter_mut().enumerate() {
            for sample in 0..4 {
                let mut call = row.calls.get(sample).unwrap();
                if (marker + sample) % 3 == 0 {
                    call.phased = false;
                }
                row.calls.set_call(sample, call).unwrap();
            }
        }

        let one = initialize_with_threads(&window, 2026, 1).unwrap();
        let eight = initialize_with_threads(&window, 2026, 8).unwrap();

        let one_calls = one
            .rows
            .iter()
            .map(|row| {
                (0..4)
                    .map(|sample| row.calls.get(sample).unwrap().alleles)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let eight_calls = eight
            .rows
            .iter()
            .map(|row| {
                (0..4)
                    .map(|sample| row.calls.get(sample).unwrap().alleles)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_ne!(one_calls, eight_calls);
    }

    #[test]
    fn initialization_uses_beagle_reverse_haplotypes_for_missing_alleles() {
        let rows = (0..3)
            .map(|marker| {
                let calls = (0..5)
                    .map(|sample| {
                        let alleles = if sample == 0 {
                            if marker < 2 {
                                [Some(0), Some(1)]
                            } else {
                                [None, Some(0)]
                            }
                        } else if marker < 2 {
                            [Some(0), Some(1)]
                        } else {
                            [Some(1), Some(0)]
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
                        id: format!("v{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(calls),
                }
            })
            .collect::<Vec<_>>();
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            cm_positions: vec![0.0, 0.01, 0.02],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 3,
        };

        let initialized = initialize(&window, 17).unwrap();

        assert_eq!(
            initialized.rows[2].calls.get(0).unwrap().alleles,
            [Some(0), Some(0)],
            "missing alleles follow the Beagle reverse phase after forward alignment"
        );
    }

    #[test]
    fn reference_haplotypes_participate_in_initial_pbwt_imputation() {
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: 1,
                    id: "v1".to_owned(),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall::missing(false)]),
            }],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut reference = PackedHaplotypes::empty_state_major(2, 1, 1).unwrap();
        reference.set_allele(0, 0, Some(1)).unwrap();
        reference.set_allele(0, 1, Some(1)).unwrap();
        reference.mark_complete();

        let initialized =
            initialize_with_reference_pool(&window, Some(&reference), 17, &pool).unwrap();

        assert_eq!(
            initialized.rows[0].calls.get(0).unwrap().alleles,
            [Some(1), Some(1)],
            "reference haplotypes should be visible to initial PBWT neighbor imputation"
        );
    }

    #[test]
    fn compact_and_wide_donor_storage_preserve_ids_and_missing_values() {
        let mut compact = DonorProposalStorage::new(4, true);
        let mut wide = DonorProposalStorage::new(4, false);
        match (&mut compact, &mut wide) {
            (
                DonorProposalStorage::Compact(compact_values),
                DonorProposalStorage::Wide(wide_values),
            ) => {
                compact_values[0] = 3;
                compact_values[2] = 11;
                wide_values[0] = 3;
                wide_values[2] = 11;
            }
            _ => panic!("donor storage variants have unexpected layouts"),
        }

        for storage in [&compact, &wide] {
            assert_eq!(storage.get(0), Some(3));
            assert_eq!(storage.get(1), None);
            assert_eq!(storage.get(2), Some(11));
            assert_eq!(storage.raw_pair(0), [3, -1]);
            assert_eq!(storage.raw_pair(1), [-1, 11]);
        }
    }
}
