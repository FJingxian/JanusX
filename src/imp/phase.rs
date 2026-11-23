//! Progressive per-sample phasing over bounded windows.

use super::composite;
use super::hmm;
use super::pbwt;
use super::phase_baum;
use super::types::{
    validate_phased_reference_calls, CompositePanel, DiploidCall, DiploidCalls, HMMConfig,
    HMMParameterStats, HMMParameterTrace, HMMWorkspace, ImpHmmTrace, PackedHaplotypes, PhaseConfig,
    PhaseWindow, PhasedRow, PhasedWindow, PosteriorWindow, ResolvedDiagnosticSite,
};
use rayon::prelude::*;
use rayon::ThreadPool;
use std::borrow::Cow;
use std::cell::RefCell;
use std::fs::File;
use std::io::Write as IoWrite;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

struct SampleUpdate {
    calls: SampleCallUpdate,
    diagnostic_trace: Vec<ImpHmmTrace>,
    unphased_heterozygote_count: usize,
    phase_swap_count: usize,
}

enum SampleCallUpdate {
    Full(DiploidCalls),
    Sparse { changes: Vec<(usize, DiploidCall)> },
}

fn sample_call_update(update: phase_baum::JointPhaseUpdate) -> SampleCallUpdate {
    if update.calls.is_empty() {
        SampleCallUpdate::Sparse {
            changes: update.changes,
        }
    } else {
        SampleCallUpdate::Full(DiploidCalls::from(update.calls))
    }
}

fn apply_sample_updates_in_place(
    rows: &mut [PhasedRow],
    updates: &[SampleUpdate],
    marker_indices: &[usize],
    mut packed_snapshot: Option<&mut PackedHaplotypes>,
) -> Result<bool, String> {
    let sample_count = rows.first().map_or(0, |row| row.calls.len());
    if updates.len() > sample_count {
        return Err(format!(
            "phase update count {} exceeds sample count {sample_count}",
            updates.len()
        ));
    }
    let mut packed_valid = packed_snapshot.is_some();
    for (sample, update) in updates.iter().enumerate() {
        match &update.calls {
            SampleCallUpdate::Full(calls) => {
                for (local_marker, &marker) in marker_indices.iter().enumerate() {
                    let call = calls
                        .get(local_marker)
                        .ok_or_else(|| "phase update is missing a window marker".to_owned())?;
                    rows[marker].calls.set_call_unchecked(sample, call);
                    if packed_valid {
                        if let Some(packed) = packed_snapshot.as_deref_mut() {
                            packed_valid = packed.overwrite_complete_biallelic_call(
                                local_marker,
                                sample,
                                call,
                            );
                        }
                    }
                }
            }
            SampleCallUpdate::Sparse { changes } => {
                for &(local_marker, call) in changes {
                    let marker = *marker_indices.get(local_marker).ok_or_else(|| {
                        "sparse phase update marker is outside the phase window".to_owned()
                    })?;
                    rows[marker].calls.set_call_unchecked(sample, call);
                    if packed_valid {
                        if let Some(packed) = packed_snapshot.as_deref_mut() {
                            packed_valid = packed.overwrite_complete_biallelic_call(
                                local_marker,
                                sample,
                                call,
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(packed_valid)
}

fn unchanged_sample_update(_snapshot: &PhasedWindow, _sample: usize) -> SampleUpdate {
    SampleUpdate {
        calls: SampleCallUpdate::Sparse {
            changes: Vec::new(),
        },
        diagnostic_trace: Vec::new(),
        unphased_heterozygote_count: 0,
        phase_swap_count: 0,
    }
}

struct LowFrequencyAlleleMarginals {
    values: Vec<f32>,
    offsets: Vec<usize>,
}

#[derive(Clone)]
struct Stage2AnchorPosterior {
    hmm_marker: usize,
    probabilities0: Vec<f32>,
    probabilities1: Vec<f32>,
}

#[derive(Default, Clone, Copy)]
struct CallbackDonorCounts {
    target_donor_states: u64,
    reference_fast_states: u64,
    reference_fallback_states: u64,
    missing_donor_states: u64,
    target_heterozygous_rows: u64,
    target_partial_or_missing_rows: u64,
    probability_calculations: u64,
    donor_source_reads: u64,
    rare_flag_queries: u64,
    marginal_additions: u64,
}

impl CallbackDonorCounts {
    fn add_assign(&mut self, other: Self) {
        self.target_donor_states += other.target_donor_states;
        self.reference_fast_states += other.reference_fast_states;
        self.reference_fallback_states += other.reference_fallback_states;
        self.missing_donor_states += other.missing_donor_states;
        self.target_heterozygous_rows += other.target_heterozygous_rows;
        self.target_partial_or_missing_rows += other.target_partial_or_missing_rows;
        self.probability_calculations += other.probability_calculations;
        self.donor_source_reads += other.donor_source_reads;
        self.rare_flag_queries += other.rare_flag_queries;
        self.marginal_additions += other.marginal_additions;
    }
}

struct Stage2MarginalRun {
    marginals: [LowFrequencyAlleleMarginals; 2],
    anchor_posteriors: Vec<Stage2AnchorPosterior>,
    mismatch_matrix_bytes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage2HmmBackend {
    Production,
    MismatchMatrix,
}

struct PhaseWorkerWorkspace {
    baum: phase_baum::PhaseBaumWorkspace,
    panel: Option<CompositePanel>,
    donors: composite::DonorSegmentWorkspace,
}

#[derive(Default)]
struct PhaseEmWorkspace {
    hmm: hmm::EmPairWorkspace,
    panel: Option<CompositePanel>,
    donors: composite::DonorSegmentWorkspace,
    observations0: Vec<Option<u16>>,
    observations1: Vec<Option<u16>>,
}

thread_local! {
    static PHASE_WORKER_WORKSPACES:
        RefCell<Option<(PhaseWorkerWorkspace, PhaseWorkerWorkspace)>> =
        const { RefCell::new(None) };
    static PHASE_EM_WORKSPACE: RefCell<Option<PhaseEmWorkspace>> = const { RefCell::new(None) };
}

fn with_phase_worker_workspaces<R>(
    action: impl FnOnce(&mut PhaseWorkerWorkspace, &mut PhaseWorkerWorkspace) -> R,
) -> R {
    PHASE_WORKER_WORKSPACES.with(|slot| {
        let mut guard = slot.borrow_mut();
        let (first, second) = guard.get_or_insert_with(|| {
            (
                PhaseWorkerWorkspace {
                    baum: phase_baum::PhaseBaumWorkspace::default(),
                    panel: None,
                    donors: composite::DonorSegmentWorkspace::default(),
                },
                PhaseWorkerWorkspace {
                    baum: phase_baum::PhaseBaumWorkspace::default(),
                    panel: None,
                    donors: composite::DonorSegmentWorkspace::default(),
                },
            )
        });
        action(first, second)
    })
}

fn with_phase_em_workspace<R>(action: impl FnOnce(&mut PhaseEmWorkspace) -> R) -> R {
    PHASE_EM_WORKSPACE.with(|slot| {
        let mut guard = slot.borrow_mut();
        action(guard.get_or_insert_with(PhaseEmWorkspace::default))
    })
}

/// Release per-worker phase caches after all joint HMM rounds complete.
///
/// `PHASE_WORKER_WORKSPACES` deliberately retains two fully grown scratch
/// workspaces on each Rayon worker so phase rounds do not repeatedly allocate.
/// Those caches are dead before Stage 2 and imputation, but thread-local
/// storage otherwise keeps them resident until the pool is dropped. Releasing
/// the whole worker-local objects here changes no HMM input or arithmetic and
/// prevents their memory from overlapping the later panels, PBWT indexes, and
/// posterior buffers.
pub(crate) fn release_worker_workspaces(pool: &rayon::ThreadPool) {
    pool.broadcast(|_| {
        PHASE_WORKER_WORKSPACES.with(|slot| {
            drop(slot.borrow_mut().take());
        });
        PHASE_EM_WORKSPACE.with(|slot| {
            drop(slot.borrow_mut().take());
        });
    });
}

/// Release EM-only scratch after the final burn-in parameter update.
///
/// The joint PhaseBaum workspace must remain cached across rounds, but the
/// separate EM pair workspace is never touched after parameter fitting ends.
/// Dropping only this worker-local cache avoids reallocating the phase
/// workspace while preventing dead EM arrays from overlapping the remaining
/// phasing rounds.
fn release_phase_em_workspaces(pool: &rayon::ThreadPool) {
    pool.broadcast(|_| {
        PHASE_EM_WORKSPACE.with(|slot| {
            drop(slot.borrow_mut().take());
        });
    });
}

#[derive(Debug, Eq, PartialEq)]
struct Stage2SparsePlan {
    anchor_markers: Vec<usize>,
    unresolved_markers: Vec<usize>,
}

fn stage2_sparse_plan(
    input: &PhaseWindow,
    sample: usize,
    low_frequency_mask: &[bool],
    stage1_marker_indices: &[usize],
) -> Result<Stage2SparsePlan, String> {
    if low_frequency_mask.len() != input.rows.len()
        || stage1_marker_indices.is_empty()
        || stage1_marker_indices
            .iter()
            .any(|&marker| marker >= input.rows.len())
        || stage1_marker_indices
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
    {
        return Err("Stage2 sparse marker inputs are inconsistent".to_owned());
    }
    let mut unresolved_markers = Vec::new();
    let mut needed_anchors = vec![false; stage1_marker_indices.len()];
    for (marker, &is_low_frequency) in low_frequency_mask.iter().enumerate() {
        if !is_low_frequency {
            continue;
        }
        let call = input.rows[marker].calls.get(sample).unwrap();
        if matches!(call.alleles, [Some(left), Some(right)] if left == right) {
            continue;
        }
        unresolved_markers.push(marker);
        let right = stage1_marker_indices.partition_point(|&anchor| anchor < marker);
        if let Some(left) = right.checked_sub(1) {
            needed_anchors[left] = true;
        }
        if right < stage1_marker_indices.len() {
            needed_anchors[right] = true;
        }
    }
    let anchor_markers = stage1_marker_indices
        .iter()
        .zip(needed_anchors)
        .filter_map(|(&marker, needed)| needed.then_some(marker))
        .collect();
    Ok(Stage2SparsePlan {
        anchor_markers,
        unresolved_markers,
    })
}

impl LowFrequencyAlleleMarginals {
    fn at(&self, marker: usize) -> &[f32] {
        &self.values[self.offsets[marker]..self.offsets[marker + 1]]
    }
}

const MAX_EM_INITIALIZATION_ROUNDS: usize = 15;
const MAX_EM_TARGET_SAMPLES: usize = 500;
static SAMPLE_MOSAIC_NANOS: AtomicU64 = AtomicU64::new(0);
static SAMPLE_HMM_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_PANEL_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_HMM_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_RESULT_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_SHADOW_HMM_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_SHADOW_RESULT_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_SHADOW_MATRIX_BYTES: AtomicU64 = AtomicU64::new(0);
static STAGE2_SHADOW_ANCHOR_POSTERIORS: AtomicU64 = AtomicU64::new(0);
static STAGE2_SHADOW_MARGINAL_CELLS: AtomicU64 = AtomicU64::new(0);
static STAGE2_SHADOW_GT_CELLS: AtomicU64 = AtomicU64::new(0);
static STAGE2_SHADOW_RUNS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_ANCHOR_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_INTERVAL_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_WEIGHT_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_ROW_NANOS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_ROWS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_TARGET_DONOR_STATES: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_REFERENCE_FAST_STATES: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_REFERENCE_FALLBACK_STATES: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_MISSING_DONOR_STATES: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_TARGET_HET_ROWS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_TARGET_PARTIAL_OR_MISSING_ROWS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_PROBABILITY_CALCULATIONS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_DONOR_SOURCE_READS: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_RARE_FLAG_QUERIES: AtomicU64 = AtomicU64::new(0);
static STAGE2_CALLBACK_MARGINAL_ADDITIONS: AtomicU64 = AtomicU64::new(0);
static EM_MOSAIC_NANOS: AtomicU64 = AtomicU64::new(0);
static EM_HMM_NANOS: AtomicU64 = AtomicU64::new(0);

fn phase_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some()
            || std::env::var_os("JANUSX_IMP_PHASE_PROFILE").is_some()
            || std::env::var_os("JANUSX_IMP_PHASE_DETAIL_PROFILE").is_some()
    })
}

fn stage2_mismatch_shadow_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JANUSX_IMP_STAGE2_MISMATCH_SHADOW").is_some())
}

fn stage2_callback_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JANUSX_IMP_STAGE2_CALLBACK_PROFILE").is_some())
}

fn pack_complete_biallelic_window(
    window: &PhasedWindow,
    cache_budget: usize,
) -> Result<Option<PackedHaplotypes>, String> {
    // The donor mosaic is rebuilt independently for every target sample. Pack
    // the shared phased snapshot once per round so each sample only gathers
    // its donor haplotypes from compact bitplanes.
    let n_samples = window.rows.first().map_or(0, |row| row.calls.len());
    if n_samples == 0 || window.rows.iter().any(|row| row.calls.len() != n_samples) {
        return Ok(None);
    }
    let n_haplotypes = n_samples
        .checked_mul(2)
        .ok_or_else(|| "phase snapshot haplotype count overflow".to_owned())?;
    let packed_bytes = n_haplotypes
        .checked_mul(window.rows.len())
        .map(|bits| bits.div_ceil(64))
        .and_then(|words| words.checked_mul(std::mem::size_of::<u64>()))
        .ok_or_else(|| "phase snapshot cache size overflow".to_owned())?;
    if packed_bytes > cache_budget {
        return Ok(None);
    }
    PackedHaplotypes::complete_biallelic_state_major(window)
}

fn phase_packed_cache_budget(config: &PhaseConfig) -> usize {
    std::env::var("JX_IMP_PHASE_PACKED_CACHE_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(config.packed_cache_budget_bytes)
}

fn validate_marker_projection(
    source_marker_count: usize,
    marker_indices: &[usize],
) -> Result<(), String> {
    if marker_indices.is_empty()
        || marker_indices
            .iter()
            .any(|&marker| marker >= source_marker_count)
        || marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("phase marker projection must be nonempty, unique, and ordered".to_owned());
    }
    Ok(())
}

fn project_phase_window(
    window: &PhaseWindow,
    marker_indices: &[usize],
) -> Result<PhaseWindow, String> {
    validate_marker_projection(window.rows.len(), marker_indices)?;
    Ok(PhaseWindow {
        chrom: window.chrom.clone(),
        index: window.index,
        rows: marker_indices
            .iter()
            .map(|&marker| window.rows[marker].clone())
            .collect(),
        cm_positions: marker_indices
            .iter()
            .map(|&marker| window.cm_positions[marker])
            .collect(),
        incoming_overlap_len: marker_indices
            .partition_point(|&marker| marker < window.incoming_overlap_len),
        emit_start: marker_indices.partition_point(|&marker| marker < window.emit_start),
        emit_end: marker_indices.partition_point(|&marker| marker < window.emit_end),
    })
}

fn project_phased_window(
    window: &PhasedWindow,
    marker_indices: &[usize],
) -> Result<PhasedWindow, String> {
    validate_marker_projection(window.rows.len(), marker_indices)?;
    Ok(PhasedWindow {
        chrom: window.chrom.clone(),
        index: window.index,
        rows: marker_indices
            .iter()
            .map(|&marker| window.rows[marker].clone())
            .collect(),
        emit_start: marker_indices.partition_point(|&marker| marker < window.emit_start),
        emit_end: marker_indices.partition_point(|&marker| marker < window.emit_end),
    })
}

/// Preserve Beagle's physical recombination rate when the phasing population
/// grows from target haplotypes to target plus reference haplotypes.
fn phase_hmm_for_haplotype_count(
    mut config: HMMConfig,
    haplotype_count: usize,
) -> Result<HMMConfig, String> {
    if haplotype_count == 0 || config.haplotype_count == 0 {
        return Err("phase HMM haplotype counts must be positive".to_owned());
    }
    if config.haplotype_count != haplotype_count {
        config.recombination_intensity = (f64::from(config.recombination_intensity)
            * config.haplotype_count as f64
            / haplotype_count as f64) as f32;
        config.haplotype_count = haplotype_count;
    }
    Ok(config)
}

fn phase_hmm_for_haplotypes(
    config: HMMConfig,
    haplotype_count: usize,
) -> Result<HMMConfig, String> {
    let mut config = phase_hmm_for_haplotype_count(config, haplotype_count)?;
    // Beagle's PhaseData derives pMismatch from the total phasing
    // haplotype count. Its `err` option is consumed later by ImpData and
    // does not override this Stage1 value.
    config.err = f64::from(super::types::default_err(haplotype_count)? as f32);
    Ok(config)
}

/// Pack phased reference haplotypes at target marker positions.
///
/// Reference-only markers are intentionally omitted. A target marker absent
/// from the reference remains missing in the packed matrix and is excluded
/// from the Stage1 PBWT by the caller.
#[cfg(test)]
pub(crate) fn with_reference_haplotypes(
    target: &PhaseWindow,
    reference: &PhasedWindow,
) -> Result<(PackedHaplotypes, usize, Vec<bool>), String> {
    let alignment = super::impute::ReferenceMarkerAlignment::new(&target.rows, &reference.rows)?;
    with_reference_haplotypes_aligned(target, reference, &alignment)
}

pub(crate) fn with_reference_haplotypes_aligned(
    target: &PhaseWindow,
    reference: &PhasedWindow,
    alignment: &super::impute::ReferenceMarkerAlignment,
) -> Result<(PackedHaplotypes, usize, Vec<bool>), String> {
    let target_samples = target.rows.first().map_or(0, |row| row.calls.len());
    let reference_samples = reference.rows.first().map_or(0, |row| row.calls.len());
    if target_samples == 0 || reference_samples == 0 {
        return Err(
            "reference-assisted phasing requires positive target and reference sample counts"
                .to_owned(),
        );
    }
    if target.chrom != reference.chrom {
        return Err(format!(
            "target and reference chromosomes differ for reference-assisted phasing: {} versus {}",
            target.chrom, reference.chrom
        ));
    }
    if target
        .rows
        .iter()
        .any(|row| row.calls.len() != target_samples)
        || reference
            .rows
            .iter()
            .any(|row| row.calls.len() != reference_samples)
    {
        return Err("reference-assisted phasing input has inconsistent sample counts".to_owned());
    }
    for row in &reference.rows {
        validate_phased_reference_calls(row)?;
    }
    if alignment.len() != target.rows.len() {
        return Err("reference marker alignment does not match the phase window".to_owned());
    }

    let reference_haplotypes = reference_samples
        .checked_mul(2)
        .ok_or_else(|| "reference-assisted phasing haplotype count overflow".to_owned())?;
    let reference_present = alignment.reference_present().collect::<Vec<_>>();
    let all_present = reference_present.iter().all(|&present| present);
    let mut packed = if all_present {
        PackedHaplotypes::empty_complete_biallelic_state_major(
            reference_haplotypes,
            target.rows.len(),
        )?
    } else {
        PackedHaplotypes::empty_state_major(reference_haplotypes, target.rows.len(), 1)?
    };
    for marker in 0..target.rows.len() {
        if let Some(reference_index) = alignment.reference_row(marker) {
            let swap = alignment.reference_swap(marker);
            let reference_row = &reference.rows[reference_index];
            for (sample, call) in reference_row.calls.iter().enumerate() {
                let [first, second] = call.alleles;
                let first = first.map(|allele| {
                    if swap {
                        swap_biallelic_phase_allele(allele)
                    } else {
                        allele
                    }
                });
                let second = second.map(|allele| {
                    if swap {
                        swap_biallelic_phase_allele(allele)
                    } else {
                        allele
                    }
                });
                packed.set_allele(marker, sample * 2, first)?;
                packed.set_allele(marker, sample * 2 + 1, second)?;
            }
        }
    }
    if all_present {
        packed.mark_complete();
    }

    Ok((packed, target_samples, reference_present))
}

/// Pre-packed state-major 1-bit reference haplotypes and rare-carrier flags
/// extracted directly from a `CompactReferenceWindow` before spilling it to
/// disk for phasing.
pub(crate) struct PrepackedPhaseReference {
    pub(crate) packed: PackedHaplotypes,
    pub(crate) target_samples: usize,
    pub(crate) reference_present: Vec<bool>,
    pub(crate) low_frequency_data: (Vec<bool>, Vec<Vec<bool>>),
}

/// Pack aligned reference haplotypes directly from a 1-bit
/// `CompactReferenceWindow` (either resident in RAM or already spilled to disk)
/// into state-major `PackedHaplotypes` without decoding 4-bit `DiploidCalls` or
/// cloning `VariantMeta`.
pub(crate) fn pack_compact_reference_haplotypes_aligned(
    target: &PhaseWindow,
    reference: &mut super::reference_store::CompactReferenceWindow,
    alignment: &super::impute::ReferenceMarkerAlignment,
) -> Result<(PackedHaplotypes, usize, Vec<bool>), String> {
    let target_samples = target.rows.first().map_or(0, |row| row.calls.len());
    let reference_samples = reference.n_samples;
    if target_samples == 0 || reference_samples == 0 {
        return Err(
            "reference-assisted phasing requires positive target and reference sample counts"
                .to_owned(),
        );
    }
    if target.chrom != reference.chrom {
        return Err(format!(
            "target and reference chromosomes differ for reference-assisted phasing: {} versus {}",
            target.chrom, reference.chrom
        ));
    }
    if target
        .rows
        .iter()
        .any(|row| row.calls.len() != target_samples)
    {
        return Err("reference-assisted phasing input has inconsistent sample counts".to_owned());
    }
    if alignment.len() != target.rows.len() {
        return Err("reference marker alignment does not match the phase window".to_owned());
    }

    let reference_haplotypes = reference_samples
        .checked_mul(2)
        .ok_or_else(|| "reference-assisted phasing haplotype count overflow".to_owned())?;
    let words_per_marker = reference.words_per_marker;
    let reference_present = alignment.reference_present().collect::<Vec<_>>();
    let all_present = reference_present.iter().all(|&present| present);
    let mut packed = if all_present {
        PackedHaplotypes::empty_complete_biallelic_state_major(
            reference_haplotypes,
            target.rows.len(),
        )?
    } else {
        PackedHaplotypes::empty_state_major(reference_haplotypes, target.rows.len(), 1)?
    };
    match reference.alleles {
        super::reference_store::ReferenceWindowAlleles::Resident(ref words) => {
            for marker in 0..target.rows.len() {
                if let Some(reference_index) = alignment.reference_row(marker) {
                    let start = reference_index
                        .checked_mul(words_per_marker)
                        .ok_or_else(|| "compact reference marker offset overflow".to_owned())?;
                    let end = start
                        .checked_add(words_per_marker)
                        .ok_or_else(|| "compact reference marker offset overflow".to_owned())?;
                    let row_words = words.get(start..end).ok_or_else(|| {
                        format!("compact reference marker index {reference_index} is out of bounds")
                    })?;
                    packed.write_fresh_state_major_biallelic_marker_from_words(
                        marker,
                        row_words,
                        alignment.reference_swap(marker),
                    )?;
                }
            }
        }
        super::reference_store::ReferenceWindowAlleles::Spilled(ref mut spill) => {
            let spill_markers = spill.n_markers();
            if spill.words_per_marker() != words_per_marker {
                return Err("spilled reference cache word stride differs from window".to_owned());
            }
            let label = spill.path_display().to_string();
            let mut reader = spill.open_reader()?;
            let mut row_buf = vec![0u64; words_per_marker];
            let mut cursor = 0usize;
            for marker in 0..target.rows.len() {
                if let Some(reference_index) = alignment.reference_row(marker) {
                    if reference_index >= spill_markers || reference_index < cursor {
                        return Err(format!(
                            "spilled reference alignment index {reference_index} is out of order (cursor {cursor})"
                        ));
                    }
                    while cursor <= reference_index {
                        super::reference_store::read_exact_u64_words(
                            &mut reader,
                            &mut row_buf,
                            &label,
                        )?;
                        cursor += 1;
                    }
                    packed.write_fresh_state_major_biallelic_marker_from_words(
                        marker,
                        &row_buf,
                        alignment.reference_swap(marker),
                    )?;
                }
            }
        }
    }
    if all_present {
        packed.mark_complete();
    }

    Ok((packed, target_samples, reference_present))
}

fn swap_biallelic_phase_allele(allele: u16) -> u16 {
    if allele == 0 {
        1
    } else {
        0
    }
}

pub(crate) fn splice_phase_overlap(
    window: &mut PhaseWindow,
    previous_carry: &[PhasedRow],
) -> Result<(), String> {
    let overlap = window.incoming_overlap_len;
    if overlap == 0 {
        return Ok(());
    }
    if overlap > window.rows.len() || overlap > previous_carry.len() {
        return Err(format!(
            "phase overlap has {overlap} rows, but current/previous windows contain {}/{} rows",
            window.rows.len(),
            previous_carry.len()
        ));
    }
    let carry_start = previous_carry.len() - overlap;
    let n_samples = window.rows[0].calls.len();
    for marker in 0..overlap {
        let current = &mut window.rows[marker];
        let previous = &previous_carry[carry_start + marker];
        if current.meta != previous.meta {
            return Err(format!(
                "phase overlap variant differs at current {}:{} and previous {}:{}",
                current.meta.chrom, current.meta.pos, previous.meta.chrom, previous.meta.pos
            ));
        }
        if previous.calls.len() != n_samples {
            return Err("phase overlap sample counts differ between adjacent windows".to_owned());
        }
        // Complete previous calls carry phase orientation across the window
        // boundary. Unresolved calls leave the current input untouched so the
        // next phase/imputation pass can infer them with the full window data.
        for (sample, call) in previous.calls.iter().enumerate() {
            if call.is_complete() {
                current.calls.set_call_unchecked(sample, call);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn phase_window(
    window: &PhaseWindow,
    config: &PhaseConfig,
    seed: u64,
    ibd2_intervals: &[super::types::Ibd2Interval],
    pool: &ThreadPool,
    reference_window: Option<&PhasedWindow>,
    diagnostic_site: Option<&ResolvedDiagnosticSite>,
) -> Result<
    (
        PhasedWindow,
        HMMConfig,
        Vec<HMMParameterTrace>,
        Vec<ImpHmmTrace>,
    ),
    String,
> {
    let target_window = window;
    let reference_alignment = reference_window
        .map(|reference| {
            super::impute::ReferenceMarkerAlignment::new(&target_window.rows, &reference.rows)
        })
        .transpose()?;
    phase_window_prepared(
        window,
        window,
        config,
        seed,
        ibd2_intervals,
        pool,
        reference_window,
        reference_alignment.as_ref(),
        None,
        diagnostic_site,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_window_prepared(
    window: &PhaseWindow,
    carrier_window: &PhaseWindow,
    config: &PhaseConfig,
    seed: u64,
    ibd2_intervals: &[super::types::Ibd2Interval],
    pool: &ThreadPool,
    reference_window: Option<&PhasedWindow>,
    reference_alignment: Option<&super::impute::ReferenceMarkerAlignment>,
    low_frequency_mask: Option<&[bool]>,
    diagnostic_site: Option<&ResolvedDiagnosticSite>,
) -> Result<
    (
        PhasedWindow,
        HMMConfig,
        Vec<HMMParameterTrace>,
        Vec<ImpHmmTrace>,
    ),
    String,
> {
    phase_window_prepared_impl(
        window,
        carrier_window,
        config,
        seed,
        ibd2_intervals,
        pool,
        reference_window,
        reference_alignment,
        None,
        low_frequency_mask,
        diagnostic_site,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_window_prepared_with_prepacked(
    window: &PhaseWindow,
    carrier_window: &PhaseWindow,
    config: &PhaseConfig,
    seed: u64,
    ibd2_intervals: &[super::types::Ibd2Interval],
    pool: &ThreadPool,
    prepacked_reference: Option<PrepackedPhaseReference>,
    low_frequency_mask: Option<&[bool]>,
    diagnostic_site: Option<&ResolvedDiagnosticSite>,
) -> Result<
    (
        PhasedWindow,
        HMMConfig,
        Vec<HMMParameterTrace>,
        Vec<ImpHmmTrace>,
    ),
    String,
> {
    phase_window_prepared_impl(
        window,
        carrier_window,
        config,
        seed,
        ibd2_intervals,
        pool,
        None,
        None,
        prepacked_reference,
        low_frequency_mask,
        diagnostic_site,
    )
}

#[allow(clippy::too_many_arguments)]
fn phase_window_prepared_impl(
    window: &PhaseWindow,
    carrier_window: &PhaseWindow,
    config: &PhaseConfig,
    seed: u64,
    ibd2_intervals: &[super::types::Ibd2Interval],
    pool: &ThreadPool,
    reference_window: Option<&PhasedWindow>,
    reference_alignment: Option<&super::impute::ReferenceMarkerAlignment>,
    prepacked_reference: Option<PrepackedPhaseReference>,
    low_frequency_mask: Option<&[bool]>,
    diagnostic_site: Option<&ResolvedDiagnosticSite>,
) -> Result<
    (
        PhasedWindow,
        HMMConfig,
        Vec<HMMParameterTrace>,
        Vec<ImpHmmTrace>,
    ),
    String,
> {
    let target_window = window;
    if reference_window.is_some() != reference_alignment.is_some() {
        return Err("prepared reference phasing requires its marker alignment".to_owned());
    }
    if low_frequency_mask.is_some_and(|mask| mask.len() != target_window.rows.len()) {
        return Err("prepared low-frequency mask does not match the phase window".to_owned());
    }
    if carrier_window.rows.len() != target_window.rows.len()
        || carrier_window
            .rows
            .iter()
            .zip(&target_window.rows)
            .any(|(carrier, target)| {
                carrier.meta != target.meta || carrier.calls.len() != target.calls.len()
            })
    {
        return Err("carrier-count window does not match the phase window".to_owned());
    }
    let (reference_phase_storage, precomputed_low_frequency) =
        if let Some(pre) = prepacked_reference {
            (
                Some((pre.packed, pre.target_samples, pre.reference_present)),
                Some(pre.low_frequency_data),
            )
        } else {
            (
                reference_window
                    .zip(reference_alignment)
                    .map(|(reference, alignment)| {
                        with_reference_haplotypes_aligned(target_window, reference, alignment)
                    })
                    .transpose()?,
                None,
            )
        };
    let target_only_samples = target_window.rows[0].calls.len();
    let (reference_packed, target_samples, reference_present) = if reference_phase_storage
        .as_ref()
        .is_some_and(|(_, _, present)| {
            present.iter().filter(|&&is_present| is_present).count() >= 2
        }) {
        let (packed, count, present) = reference_phase_storage.as_ref().unwrap();
        (Some(packed), *count, Some(present.as_slice()))
    } else {
        (None, target_only_samples, None)
    };
    target_window.validate(target_samples)?;
    let reference_samples = reference_packed.map_or(0, |packed| packed.n_haplotypes / 2);
    let reference_packed_bytes = reference_packed.map_or(0, PackedHaplotypes::bytes);
    let n_samples = target_samples
        .checked_add(reference_samples)
        .ok_or_else(|| "reference-assisted phasing sample count overflow".to_owned())?;
    let mut effective_config = *config;
    effective_config.hmm = phase_hmm_for_haplotypes(config.hmm, n_samples.saturating_mul(2))?;
    let config = &effective_config;
    let profile = phase_profiling_enabled();
    let profile_started = profile.then(Instant::now);
    let profile_mosaic_start = SAMPLE_MOSAIC_NANOS.load(Ordering::Relaxed);
    let profile_hmm_start = SAMPLE_HMM_NANOS.load(Ordering::Relaxed);
    let em_mosaic_start = EM_MOSAIC_NANOS.load(Ordering::Relaxed);
    let em_hmm_start = EM_HMM_NANOS.load(Ordering::Relaxed);
    let stage2_panel_start = STAGE2_PANEL_NANOS.load(Ordering::Relaxed);
    let stage2_hmm_start = STAGE2_HMM_NANOS.load(Ordering::Relaxed);
    let stage2_result_start = STAGE2_RESULT_NANOS.load(Ordering::Relaxed);
    let pair_profile_start = phase_baum::pair_detail_profile_nanos();
    let prepare_detail_start = phase_baum::prepare_detail_profile_nanos();
    let mosaic_detail_start = composite::mosaic_detail_profile_nanos();
    let phase_count_profile_start = phase_baum::count_profile_totals();
    phase_baum::reset_window_memory_peaks();
    let phase_memory_profile_start = phase_baum::memory_profile_totals();
    let mut initialize_elapsed = Duration::ZERO;
    let mut low_frequency_elapsed = Duration::ZERO;
    let mut em_initialize_elapsed = Duration::ZERO;
    let mut donor_index_elapsed = Duration::ZERO;
    let mut em_update_elapsed = Duration::ZERO;
    let mut sample_update_elapsed = Duration::ZERO;
    let mut pack_elapsed = Duration::ZERO;
    let mut packed_cache_bytes = 0usize;
    let mut packed_cache_enabled_rounds = 0usize;
    let mut packed_cache_disabled_rounds = 0usize;
    let mut skipped_sample_updates = 0usize;
    let mut rebuild_elapsed = Duration::ZERO;
    let mut packed_refresh_elapsed = Duration::ZERO;
    let mut stage2_elapsed = Duration::ZERO;
    let mut donor_step_count = 0usize;
    let mut sample_mosaic_detail = [0u64; 7];
    let n_markers = target_window.rows.len();
    let any_unphased_heterozygote = phase_baum::has_initial_unphased_heterozygote(target_window);
    let packed_cache_budget = phase_packed_cache_budget(config);
    if n_samples < 2 {
        let current = pbwt::initialize_with_pool(target_window, seed, pool)?;
        return Ok((current, config.hmm, Vec::new(), Vec::new()));
    }
    let has_missing = target_window
        .rows
        .iter()
        .flat_map(|row| row.calls.iter())
        .any(|call| !call.is_complete());
    if !any_unphased_heterozygote && !has_missing {
        let current = pbwt::initialize_with_pool(target_window, seed, pool)?;
        return Ok((current, config.hmm, Vec::new(), Vec::new()));
    }
    let low_frequency_started = profile.then(Instant::now);
    let input_low_frequency_data = match precomputed_low_frequency {
        Some(data) => data,
        None => super::impute::low_frequency_marker_data_with_alignment(
            carrier_window,
            reference_window,
            reference_alignment,
        )?,
    };
    if low_frequency_mask.is_some_and(|mask| mask != input_low_frequency_data.0.as_slice()) {
        return Err(
            "prepared low-frequency mask does not match the original carrier window".to_owned(),
        );
    }
    let low_frequency_mask = match low_frequency_mask {
        Some(mask) => {
            super::impute::use_two_stage_phase_from_mask(mask).then_some(Cow::Borrowed(mask))
        }
        None => super::impute::use_two_stage_phase_from_mask(&input_low_frequency_data.0)
            .then_some(Cow::Owned(input_low_frequency_data.0.clone())),
    };
    low_frequency_elapsed += low_frequency_started.map_or(Duration::ZERO, |start| start.elapsed());
    let mut low_frequency_hmm_markers = low_frequency_mask
        .as_deref()
        .map(|mask| {
            mask.iter()
                .enumerate()
                .filter_map(|(marker, &is_low)| (!is_low).then_some(marker))
                .collect::<Vec<_>>()
        })
        .filter(|markers| !markers.is_empty());
    if let (Some(markers), Some(reference_present)) =
        (low_frequency_hmm_markers.as_mut(), reference_present)
    {
        markers.retain(|&marker| reference_present[marker]);
        if markers.is_empty() {
            low_frequency_hmm_markers = None;
        }
    }
    let low_frequency_hmm_cm_positions = low_frequency_hmm_markers.as_ref().map(|markers| {
        markers
            .iter()
            .map(|&marker| window.cm_positions[marker])
            .collect::<Vec<_>>()
    });
    let full_marker_indices = (0..n_markers).collect::<Vec<_>>();
    let mut stage1_marker_indices = low_frequency_hmm_markers
        .clone()
        .unwrap_or_else(|| full_marker_indices.clone());
    if let Some(reference_present) = reference_present {
        stage1_marker_indices.retain(|&marker| reference_present[marker]);
    }
    if reference_present.is_some() && stage1_marker_indices.len() < 2 {
        return Err(
            "reference-assisted phasing requires at least two shared Stage1 markers".to_owned(),
        );
    }
    let needs_stage1_projection =
        low_frequency_mask.is_some() || stage1_marker_indices.len() != full_marker_indices.len();
    let stage1_window_storage = needs_stage1_projection
        .then(|| project_phase_window(window, &stage1_marker_indices))
        .transpose()?;
    let stage1_window = stage1_window_storage.as_ref().unwrap_or(window);
    let stage1_reference_packed_storage = reference_packed
        .filter(|_| needs_stage1_projection)
        .map(|packed| {
            let mut projected = packed.project_state_major_biallelic(&stage1_marker_indices)?;
            // The Stage1 marker map contains only markers shared with the
            // validated, complete reference panel.
            projected.mark_complete();
            Ok::<_, String>(projected)
        })
        .transpose()?;
    let stage1_reference_packed = stage1_reference_packed_storage
        .as_ref()
        .or(reference_packed);
    let initialize_started = profile.then(Instant::now);
    let (mut current, stage1_initial_storage) = if stage1_window_storage.is_some() {
        (
            pbwt::initialize_with_pool(target_window, seed, pool)?,
            Some(pbwt::initialize_with_reference_pool(
                stage1_window,
                stage1_reference_packed,
                seed,
                pool,
            )?),
        )
    } else {
        (
            pbwt::initialize_with_reference_pool(
                stage1_window,
                stage1_reference_packed,
                seed,
                pool,
            )?,
            None,
        )
    };
    initialize_elapsed += initialize_started.map_or(Duration::ZERO, |start| start.elapsed());
    if let Some(stage1_initial) = stage1_initial_storage.as_ref() {
        for (local, &full_marker) in stage1_marker_indices.iter().enumerate() {
            current.rows[full_marker].calls = stage1_initial.rows[local].calls.clone();
        }
    }
    let mosaic_segment_cm = phase_ibs_step_cm(&stage1_window.cm_positions);
    let cluster_templates_started = profile.then(Instant::now);
    let cluster_templates = pool.install(|| {
        (0..target_samples)
            .into_par_iter()
            .map(|sample| phase_baum::build_cluster_template(stage1_window, sample))
            .collect::<Result<Vec<_>, String>>()
    })?;
    let cluster_templates_elapsed =
        cluster_templates_started.map_or(Duration::ZERO, |start| start.elapsed());

    let total_rounds = if any_unphased_heterozygote || has_missing {
        config.hmm.burnin.saturating_add(config.hmm.iterations)
    } else {
        1
    };
    let mut hmm_config = config.hmm;
    let mut parameter_trace = vec![parameter_trace_row(
        target_window,
        "initial",
        0,
        &hmm_config,
    )];
    let diagnostic_marker = diagnostic_site
        .filter(|site| site.chrom == target_window.chrom)
        .and_then(|site| {
            target_window
                .rows
                .iter()
                .position(|row| row.meta.pos == site.position)
        });
    let mut hmm_trace = Vec::new();
    let mut persistent_packed_stage1_snapshot: Option<PackedHaplotypes> = None;
    let mut precomputed_round0_index: Option<pbwt::PhasePbwtIndex> = None;
    if hmm_config.em {
        let em_initialize_started = profile.then(Instant::now);
        let stage1_snapshot_storage = needs_stage1_projection
            .then(|| project_phased_window(&current, &stage1_marker_indices))
            .transpose()?;
        let stage1_snapshot = stage1_snapshot_storage.as_ref().unwrap_or(&current);
        let pack_started = profile.then(Instant::now);
        let packed_stage1_snapshot_storage = if low_frequency_mask.is_some() {
            pack_complete_biallelic_window(stage1_snapshot, packed_cache_budget)?
        } else {
            None
        };
        if low_frequency_mask.is_none() && persistent_packed_stage1_snapshot.is_none() {
            persistent_packed_stage1_snapshot =
                pack_complete_biallelic_window(stage1_snapshot, packed_cache_budget)?;
        }
        let packed_stage1_snapshot = packed_stage1_snapshot_storage
            .as_ref()
            .or(persistent_packed_stage1_snapshot.as_ref());
        let stage1_haplotype_view = composite::PhaseHaplotypeView::new(
            stage1_snapshot,
            packed_stage1_snapshot,
            stage1_reference_packed,
            target_samples,
        )?;
        pack_elapsed += pack_started.map_or(Duration::ZERO, |start| start.elapsed());
        let initialization_candidate_count = phase_candidate_count(
            0,
            config.hmm.burnin,
            config.hmm.iterations,
            n_samples.saturating_mul(2),
        );
        let initialization_index = pbwt::build_phase_donor_index_with_pool_view(
            &stage1_haplotype_view,
            &stage1_window.cm_positions,
            mosaic_segment_cm,
            pbwt::PbwtDirection::Reverse,
            initialization_candidate_count,
            ibd2_intervals,
            pool,
            seed,
        )?;
        initialize_em_parameters(
            stage1_window,
            stage1_snapshot,
            &stage1_haplotype_view,
            &initialization_index,
            mosaic_segment_cm,
            target_samples,
            config.hmm.phase_states,
            seed,
            &mut hmm_config,
            &mut parameter_trace,
            pool,
        )?;
        precomputed_round0_index = Some(initialization_index);
        em_initialize_elapsed +=
            em_initialize_started.map_or(Duration::ZERO, |start| start.elapsed());
    }
    let markers_per_finished_word = 64usize;
    let mut finished =
        vec![vec![0u64; n_markers.div_ceil(markers_per_finished_word)]; target_samples];
    let mut trailing_masked =
        vec![vec![0u64; stage1_window.rows.len().div_ceil(64)]; target_samples];
    let mut transition_factors = Vec::new();
    let mut transition_intensity_bits = None;
    let mut round = 0usize;
    while round < total_rounds {
        let round_profile = profile && std::env::var_os("JANUSX_IMP_PHASE_ROUND_PROFILE").is_some();
        let round_profile_started = round_profile.then(Instant::now);
        let round_pair_profile_start = phase_baum::pair_detail_profile_nanos();
        let round_prepare_detail_start = phase_baum::prepare_detail_profile_nanos();
        let round_count_profile_start = phase_baum::count_profile_totals();
        let round_donor_index_start = donor_index_elapsed;
        let round_pack_start = pack_elapsed;
        let round_rebuild_start = rebuild_elapsed;
        let round_em_update_start = em_update_elapsed;
        let pbwt_direction = pbwt::PbwtDirection::for_iteration(round);
        let lr_threshold = phase_lr_threshold(
            round,
            config.hmm.burnin,
            config.hmm.iterations,
            config.hmm.initial_lr,
        );
        let snapshot = PhasedWindow {
            chrom: current.chrom.clone(),
            index: current.index,
            rows: std::mem::take(&mut current.rows),
            emit_start: current.emit_start,
            emit_end: current.emit_end,
        };
        let stage1_snapshot_storage = needs_stage1_projection
            .then(|| project_phased_window(&snapshot, &stage1_marker_indices))
            .transpose()?;
        let stage1_snapshot = stage1_snapshot_storage.as_ref().unwrap_or(&snapshot);
        let pack_started = profile.then(Instant::now);
        let packed_stage1_snapshot_storage = if low_frequency_mask.is_some() {
            pack_complete_biallelic_window(stage1_snapshot, packed_cache_budget)?
        } else {
            None
        };
        if low_frequency_mask.is_none() && persistent_packed_stage1_snapshot.is_none() {
            persistent_packed_stage1_snapshot =
                pack_complete_biallelic_window(stage1_snapshot, packed_cache_budget)?;
        }
        let packed_stage1_snapshot = packed_stage1_snapshot_storage
            .as_ref()
            .or(persistent_packed_stage1_snapshot.as_ref());
        if profile {
            if let Some(packed) = packed_stage1_snapshot {
                packed_cache_bytes = packed_cache_bytes.max(packed.bytes());
                packed_cache_enabled_rounds += 1;
            } else {
                packed_cache_disabled_rounds += 1;
            }
        }
        let stage1_haplotype_view = composite::PhaseHaplotypeView::new(
            stage1_snapshot,
            packed_stage1_snapshot,
            stage1_reference_packed,
            target_samples,
        )?;
        pack_elapsed += pack_started.map_or(Duration::ZERO, |start| start.elapsed());
        let update_burnin_parameters = hmm_config.em && round > 0 && round < config.hmm.burnin;
        let candidate_count = phase_candidate_count(
            round,
            config.hmm.burnin,
            config.hmm.iterations,
            n_samples.saturating_mul(2),
        );
        // Both parameter fitting and joint PhaseBaum use the Beagle stage-1
        // marker set when the window has a separate low-frequency stage.
        let donor_index_started = profile.then(Instant::now);
        let phase_pbwt_index = if round == 0 && precomputed_round0_index.is_some() {
            precomputed_round0_index.take()
        } else {
            Some(pbwt::build_phase_donor_index_with_pool_view(
                &stage1_haplotype_view,
                &stage1_window.cm_positions,
                mosaic_segment_cm,
                pbwt_direction,
                candidate_count,
                ibd2_intervals,
                pool,
                phase_round_seed(seed, round),
            )?)
        };
        if let Some(index) = phase_pbwt_index.as_ref() {
            donor_step_count = index.step_ranges.len();
        }
        donor_index_elapsed += donor_index_started.map_or(Duration::ZERO, |start| start.elapsed());
        if update_burnin_parameters {
            let em_update_started = profile.then(Instant::now);
            let em_index = phase_pbwt_index
                .as_ref()
                .ok_or_else(|| "EM parameter estimation requires a donor index".to_owned())?;
            let stats = estimate_em_parameters(
                stage1_window,
                stage1_snapshot,
                &stage1_haplotype_view,
                em_index,
                &stage1_window.cm_positions,
                mosaic_segment_cm,
                target_samples,
                config.hmm.phase_states,
                seed,
                &mut pbwt::JavaRandom::new(phase_round_seed(seed, round)),
                &hmm_config,
                pool,
            )?;
            update_em_parameters(&mut hmm_config, stats);
            parameter_trace.push(parameter_trace_row(
                target_window,
                "burnin_update",
                round,
                &hmm_config,
            ));
            em_update_elapsed += em_update_started.map_or(Duration::ZERO, |start| start.elapsed());
        }
        let intensity_bits = hmm_config.recombination_intensity.to_bits();
        if transition_intensity_bits != Some(intensity_bits) {
            transition_factors = phase_baum::marker_no_recombination_probabilities(
                &stage1_window.cm_positions,
                hmm_config.recombination_intensity,
            )?;
            transition_intensity_bits = Some(intensity_bits);
        }
        parameter_trace.push(parameter_trace_row(
            target_window,
            "phase_start",
            round,
            &hmm_config,
        ));
        let sample_update_started = profile.then(Instant::now);
        let sample_mosaic_detail_start = composite::mosaic_detail_profile_nanos();
        let updates = pool.install(|| {
            finished
                .par_chunks_mut(2)
                .zip(trailing_masked.par_chunks_mut(2))
                .enumerate()
                .map(|(pair, pair_masks)| {
                    with_phase_worker_workspaces(|workspace0, workspace1| {
                        let (finished_pair, trailing_pair) = pair_masks;
                        // Keep tracing out of the production update. A scalar diagnostic
                        // update can differ from the paired kernel, and its calls would
                        // become donors in later rounds and windows.
                        let diagnostic_offset = diagnostic_site
                            .filter(|site| site.sample_index / 2 == pair)
                            .map(|site| site.sample_index - pair * 2)
                            .filter(|&offset| offset < finished_pair.len());
                        let mut diagnostic_finished =
                            diagnostic_offset.map(|offset| finished_pair[offset].to_vec());
                        let mut diagnostic_trailing =
                            diagnostic_offset.map(|offset| trailing_pair[offset].to_vec());
                        let (mut updates, skipped_samples) =
                            if finished_pair.len() == 2 && packed_stage1_snapshot.is_some() {
                                let sample0 = pair * 2;
                                let sample1 = sample0 + 1;
                                let (finished0, finished1) = finished_pair.split_at_mut(1);
                                let (trailing0, trailing1) = trailing_pair.split_at_mut(1);
                                let (update0, update1, skipped_samples) = phase_sample_pair_event(
                                    stage1_window,
                                    stage1_snapshot,
                                    &stage1_haplotype_view,
                                    phase_pbwt_index.as_ref().expect("paired phase PBWT index"),
                                    [sample0, sample1],
                                    &cluster_templates[sample0],
                                    &cluster_templates[sample1],
                                    [&mut finished0[0], &mut finished1[0]],
                                    [&mut trailing0[0], &mut trailing1[0]],
                                    any_unphased_heterozygote,
                                    &stage1_marker_indices,
                                    lr_threshold,
                                    mosaic_segment_cm,
                                    &hmm_config,
                                    config.hmm.phase_states,
                                    &transition_factors,
                                    [
                                        phase_sample_seed(seed, round, sample0),
                                        phase_sample_seed(seed, round, sample1),
                                    ],
                                    packed_stage1_snapshot,
                                    workspace0,
                                    workspace1,
                                )?;
                                (vec![update0, update1], skipped_samples)
                            } else {
                                let mut updates = Vec::with_capacity(finished_pair.len());
                                for (offset, (finished_sample, trailing_masked_sample)) in
                                    finished_pair.iter_mut().zip(trailing_pair).enumerate()
                                {
                                    let sample = pair * 2 + offset;
                                    let workspace = if offset == 0 {
                                        &mut workspace0.baum
                                    } else {
                                        &mut workspace1.baum
                                    };
                                    updates.push(phase_sample(
                                        stage1_window,
                                        stage1_snapshot,
                                        &stage1_haplotype_view,
                                        None,
                                        None,
                                        phase_pbwt_index.as_ref(),
                                        None,
                                        None,
                                        sample,
                                        finished_sample,
                                        trailing_masked_sample,
                                        round,
                                        None,
                                        None,
                                        &stage1_marker_indices,
                                        lr_threshold,
                                        any_unphased_heterozygote,
                                        None,
                                        None,
                                        None,
                                        false,
                                        None,
                                        &stage1_window.cm_positions,
                                        &full_marker_indices,
                                        mosaic_segment_cm,
                                        ibd2_intervals,
                                        &hmm_config,
                                        config.hmm.phase_states,
                                        phase_sample_seed(seed, round, sample),
                                        workspace,
                                    )?);
                                }
                                (updates, 0usize)
                            };
                        if let (Some(offset), Some(finished_copy), Some(trailing_copy)) = (
                            diagnostic_offset,
                            diagnostic_finished.as_mut(),
                            diagnostic_trailing.as_mut(),
                        ) {
                            if let Some(site) = diagnostic_site
                                .filter(|site| site.sample_index == pair * 2 + offset)
                            {
                                let sample = pair * 2 + offset;
                                let workspace = if offset == 0 {
                                    &mut workspace0.baum
                                } else {
                                    &mut workspace1.baum
                                };
                                let diagnostic_update = phase_sample(
                                    stage1_window,
                                    stage1_snapshot,
                                    &stage1_haplotype_view,
                                    None,
                                    None,
                                    phase_pbwt_index.as_ref(),
                                    None,
                                    None,
                                    sample,
                                    finished_copy,
                                    trailing_copy,
                                    round,
                                    Some(site),
                                    diagnostic_marker,
                                    &stage1_marker_indices,
                                    lr_threshold,
                                    any_unphased_heterozygote,
                                    None,
                                    None,
                                    None,
                                    false,
                                    None,
                                    &stage1_window.cm_positions,
                                    &full_marker_indices,
                                    mosaic_segment_cm,
                                    ibd2_intervals,
                                    &hmm_config,
                                    config.hmm.phase_states,
                                    phase_sample_seed(seed, round, sample),
                                    workspace,
                                )?;
                                updates[offset].diagnostic_trace =
                                    diagnostic_update.diagnostic_trace;
                            }
                        }
                        Ok((updates, skipped_samples))
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        let sample_mosaic_detail_end = composite::mosaic_detail_profile_nanos();
        for (total, (end, start)) in sample_mosaic_detail.iter_mut().zip(
            sample_mosaic_detail_end
                .into_iter()
                .zip(sample_mosaic_detail_start),
        ) {
            *total = total.saturating_add(end.saturating_sub(start));
        }
        skipped_sample_updates = skipped_sample_updates
            .saturating_add(updates.iter().map(|(_, skipped)| *skipped).sum::<usize>());
        let updates = updates
            .into_iter()
            .flat_map(|(updates, _)| updates)
            .collect::<Vec<_>>();
        let round_sample_update_elapsed =
            sample_update_started.map_or(Duration::ZERO, |start| start.elapsed());
        sample_update_elapsed += round_sample_update_elapsed;

        for update in &updates {
            hmm_trace.extend(update.diagnostic_trace.iter().cloned());
        }
        let unphased_heterozygote_count = updates
            .iter()
            .map(|update| update.unphased_heterozygote_count)
            .sum::<usize>();
        let phase_swap_count = updates
            .iter()
            .map(|update| update.phase_swap_count)
            .sum::<usize>();
        let next_round = next_beagle_phase_iteration(
            round,
            config.hmm.burnin,
            phase_swap_count,
            unphased_heterozygote_count,
        );
        let refresh_packed_snapshot = low_frequency_mask.is_none()
            && next_round < total_rounds
            && persistent_packed_stage1_snapshot.is_some();
        let PhasedWindow {
            chrom,
            index,
            mut rows,
            emit_start,
            emit_end,
        } = snapshot;
        let rebuild_started = profile.then(Instant::now);
        pool.install(|| {
            rows.par_iter_mut().for_each(|row| {
                row.dosage = None;
                row.ap1 = None;
                row.ap2 = None;
                row.gp = None;
                row.info_af = None;
                row.info_dr2 = None;
                row.imputed = false;
            })
        });
        let packed_refresh_started = profile.then(Instant::now);
        let packed_valid = apply_sample_updates_in_place(
            &mut rows,
            &updates,
            &stage1_marker_indices,
            if refresh_packed_snapshot {
                persistent_packed_stage1_snapshot.as_mut()
            } else {
                None
            },
        )?;
        if refresh_packed_snapshot && !packed_valid {
            persistent_packed_stage1_snapshot = None;
        }
        packed_refresh_elapsed +=
            packed_refresh_started.map_or(Duration::ZERO, |start| start.elapsed());
        let round_rebuild_elapsed = rebuild_started.map_or(Duration::ZERO, |start| start.elapsed());
        rebuild_elapsed += round_rebuild_elapsed;
        current = PhasedWindow {
            chrom,
            index,
            rows,
            emit_start,
            emit_end,
        };
        if let Some(round_started) = round_profile_started {
            let pair_end = phase_baum::pair_detail_profile_nanos();
            let prepare_end = phase_baum::prepare_detail_profile_nanos();
            let count_end = phase_baum::count_profile_totals();
            let pair_delta: [u64; 3] = std::array::from_fn(|index| {
                pair_end[index].saturating_sub(round_pair_profile_start[index])
            });
            let prepare_delta: [u64; 4] = std::array::from_fn(|index| {
                prepare_end[index].saturating_sub(round_prepare_detail_start[index])
            });
            let count_delta: [u64; 12] = std::array::from_fn(|index| {
                count_end[index].saturating_sub(round_count_profile_start[index])
            });
            let pairs = target_samples.div_ceil(2).max(1);
            let mosaic_nanos = sample_mosaic_detail_end
                .into_iter()
                .zip(sample_mosaic_detail_start)
                .map(|(end, start)| end.saturating_sub(start))
                .sum::<u64>();
            let donor_index_s = donor_index_elapsed.saturating_sub(round_donor_index_start);
            let pack_s = pack_elapsed.saturating_sub(round_pack_start);
            let rebuild_s = rebuild_elapsed.saturating_sub(round_rebuild_start);
            let em_update_s = em_update_elapsed.saturating_sub(round_em_update_start);
            let round_total_s = round_started.elapsed();
            let other_s = round_total_s
                .saturating_sub(round_sample_update_elapsed)
                .saturating_sub(donor_index_s)
                .saturating_sub(pack_s)
                .saturating_sub(rebuild_s)
                .saturating_sub(em_update_s);
            eprintln!(
                "JANUSX_IMP_PHASE_ROUND_PROFILE window={}:{} round={} pairs={} targets={} round_total_s={:.6} sample_update_s={:.6} other_s={:.6} donor_index_s={:.6} pack_s={:.6} em_update_s={:.6} rebuild_s={:.6} avg_pair_prepare_s={:.6} avg_pair_backward_s={:.6} avg_pair_forward_s={:.6} prepare_cluster_s={:.6} prepare_transition_s={:.6} prepare_emission_s={:.6} prepare_workspace_s={:.6} mosaic_s={:.6} sample_updates={} clusters={} haplotype_state_updates={} emission_rows={} single_track_preparations={}",
                target_window.chrom,
                target_window.index,
                round,
                pairs,
                target_samples,
                round_total_s.as_secs_f64(),
                round_sample_update_elapsed.as_secs_f64(),
                other_s.as_secs_f64(),
                donor_index_s.as_secs_f64(),
                pack_s.as_secs_f64(),
                em_update_s.as_secs_f64(),
                rebuild_s.as_secs_f64(),
                pair_delta[0] as f64 / 1e9 / pairs as f64,
                pair_delta[1] as f64 / 1e9 / pairs as f64,
                pair_delta[2] as f64 / 1e9 / pairs as f64,
                prepare_delta[0] as f64 / 1e9,
                prepare_delta[1] as f64 / 1e9,
                prepare_delta[2] as f64 / 1e9,
                prepare_delta[3] as f64 / 1e9,
                mosaic_nanos as f64 / 1e9,
                count_delta[0],
                count_delta[1],
                count_delta[3],
                count_delta[8],
                count_delta[11],
            );
        }
        if let (Ok(target), Ok(prefix)) = (
            std::env::var("JANUSX_TRACE_PHASE_WINDOW"),
            std::env::var("JANUSX_TRACE_PHASE_PREFIX"),
        ) {
            let trace_round = std::env::var("JANUSX_TRACE_PHASE_ROUND")
                .ok()
                .and_then(|value| value.parse::<usize>().ok());
            if target == current.index.to_string()
                && trace_round.is_none_or(|traced| traced == round)
            {
                let trace_sample = std::env::var("JANUSX_TRACE_SAMPLE_INDEX")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|&sample| sample < target_samples);
                let traced_samples = trace_sample.map_or(target_samples, |_| 1);
                let path = format!("{prefix}.window{}.round{round}.tsv", current.index);
                let file = File::create(&path).map_err(|error| {
                    format!("cannot create temporary phased-window trace {path}: {error}")
                })?;
                let capacity = stage1_marker_indices
                    .len()
                    .saturating_mul(traced_samples)
                    .saturating_mul(28)
                    .min(8 * 1024 * 1024);
                let mut writer = std::io::BufWriter::with_capacity(capacity.max(8192), file);
                writeln!(writer, "SAMPLE\tCHROM\tPOS\tALLELE0\tALLELE1")
                    .map_err(|error| format!("write phased-window trace {path}: {error}"))?;
                for sample in 0..target_samples {
                    if trace_sample.is_some_and(|traced| traced != sample) {
                        continue;
                    }
                    for &marker in &stage1_marker_indices {
                        let row = &current.rows[marker];
                        let call = row.calls.get(sample).ok_or_else(|| {
                            format!("phased-window trace sample {sample} is out of range")
                        })?;
                        writeln!(
                            writer,
                            "{}\t{}\t{}\t{}\t{}",
                            sample,
                            current.chrom,
                            row.meta.pos,
                            call.alleles[0].map_or_else(|| ".".to_owned(), |a| a.to_string()),
                            call.alleles[1].map_or_else(|| ".".to_owned(), |a| a.to_string()),
                        )
                        .map_err(|error| format!("write phased-window trace {path}: {error}"))?;
                    }
                }
                writer
                    .flush()
                    .map_err(|error| format!("flush phased-window trace {path}: {error}"))?;
            }
        }
        if hmm_config.em && update_burnin_parameters && next_round >= config.hmm.burnin {
            release_phase_em_workspaces(pool);
        }
        round = next_round;
    }
    // No later stage reuses the PhaseBaum scratch cached by the round loop.
    // Drop it before Stage 2 / imputation allocates its own large buffers.
    release_worker_workspaces(pool);
    if let Some(mask) = low_frequency_mask.as_deref() {
        let panel_detail_start = composite::stage2_panel_profile_totals();
        let callback_anchor_start = STAGE2_CALLBACK_ANCHOR_NANOS.load(Ordering::Relaxed);
        let callback_interval_start = STAGE2_CALLBACK_INTERVAL_NANOS.load(Ordering::Relaxed);
        let callback_weight_start = STAGE2_CALLBACK_WEIGHT_NANOS.load(Ordering::Relaxed);
        let callback_row_start = STAGE2_CALLBACK_ROW_NANOS.load(Ordering::Relaxed);
        let callback_rows_start = STAGE2_CALLBACK_ROWS.load(Ordering::Relaxed);
        let callback_target_donor_start =
            STAGE2_CALLBACK_TARGET_DONOR_STATES.load(Ordering::Relaxed);
        let callback_ref_fast_start = STAGE2_CALLBACK_REFERENCE_FAST_STATES.load(Ordering::Relaxed);
        let callback_ref_fallback_start =
            STAGE2_CALLBACK_REFERENCE_FALLBACK_STATES.load(Ordering::Relaxed);
        let callback_missing_donor_start =
            STAGE2_CALLBACK_MISSING_DONOR_STATES.load(Ordering::Relaxed);
        let callback_target_het_start = STAGE2_CALLBACK_TARGET_HET_ROWS.load(Ordering::Relaxed);
        let callback_target_partial_start =
            STAGE2_CALLBACK_TARGET_PARTIAL_OR_MISSING_ROWS.load(Ordering::Relaxed);
        let callback_probability_start =
            STAGE2_CALLBACK_PROBABILITY_CALCULATIONS.load(Ordering::Relaxed);
        let callback_donor_read_start = STAGE2_CALLBACK_DONOR_SOURCE_READS.load(Ordering::Relaxed);
        let callback_rare_query_start = STAGE2_CALLBACK_RARE_FLAG_QUERIES.load(Ordering::Relaxed);
        let callback_marginal_add_start =
            STAGE2_CALLBACK_MARGINAL_ADDITIONS.load(Ordering::Relaxed);
        let shadow_hmm_start = STAGE2_SHADOW_HMM_NANOS.load(Ordering::Relaxed);
        let shadow_result_start = STAGE2_SHADOW_RESULT_NANOS.load(Ordering::Relaxed);
        let shadow_matrix_bytes_start = STAGE2_SHADOW_MATRIX_BYTES.load(Ordering::Relaxed);
        let shadow_anchor_posteriors_start =
            STAGE2_SHADOW_ANCHOR_POSTERIORS.load(Ordering::Relaxed);
        let shadow_marginal_cells_start = STAGE2_SHADOW_MARGINAL_CELLS.load(Ordering::Relaxed);
        let shadow_gt_cells_start = STAGE2_SHADOW_GT_CELLS.load(Ordering::Relaxed);
        let shadow_runs_start = STAGE2_SHADOW_RUNS.load(Ordering::Relaxed);
        let stage2_started = profile.then(Instant::now);
        let mut stage2_plan_elapsed = Duration::ZERO;
        let mut stage2_snapshot_elapsed = Duration::ZERO;
        let mut stage2_pbwt_elapsed = Duration::ZERO;
        let mut stage2_sample_elapsed = Duration::ZERO;
        let mut stage2_apply_elapsed = Duration::ZERO;
        let plan_started = profile.then(Instant::now);
        let stage2_sparse_plans = pool.install(|| {
            (0..target_samples)
                .into_par_iter()
                .map(|sample| stage2_sparse_plan(window, sample, mask, &stage1_marker_indices))
                .collect::<Result<Vec<_>, _>>()
        })?;
        stage2_plan_elapsed += plan_started.map_or(Duration::ZERO, |start| start.elapsed());
        let stage2_unresolved_samples = stage2_sparse_plans
            .iter()
            .filter(|plan| !plan.unresolved_markers.is_empty())
            .count();
        let stage2_unresolved_markers = stage2_sparse_plans
            .iter()
            .map(|plan| plan.unresolved_markers.len())
            .sum::<usize>();
        let stage2_anchor_markers = stage2_sparse_plans
            .iter()
            .map(|plan| plan.anchor_markers.len())
            .sum::<usize>();
        let stage2_skipped_samples = stage2_sparse_plans
            .iter()
            .filter(|plan| plan.anchor_markers.is_empty())
            .count();
        let stage2_stage_profile = std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some()
            || std::env::var_os("JANUSX_IMP_STAGE_PROFILE").is_some();
        hmm::reset_stage2_forward_profile_peaks();
        let stage2_forward_start = hmm::stage2_forward_profile_totals();
        if stage2_unresolved_samples > 0 {
            if stage2_stage_profile {
                let unix_s = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0.0, |duration| duration.as_secs_f64());
                if let Some((vm_rss_bytes, vm_hwm_bytes)) =
                    super::memory::read_process_rss_hwm_bytes()
                {
                    eprintln!(
                        "JANUSX_IMP_STAGE_TIMESTAMP stage=stage2_start window={}:{} unix_s={unix_s:.6} vm_rss_bytes={vm_rss_bytes} vm_hwm_bytes={vm_hwm_bytes}",
                        target_window.chrom, target_window.index
                    );
                } else {
                    eprintln!(
                        "JANUSX_IMP_STAGE_TIMESTAMP stage=stage2_start window={}:{} unix_s={unix_s:.6}",
                        target_window.chrom, target_window.index
                    );
                }
            }
            let snapshot = PhasedWindow {
                chrom: current.chrom.clone(),
                index: current.index,
                rows: std::mem::take(&mut current.rows),
                emit_start: current.emit_start,
                emit_end: current.emit_end,
            };
            let snapshot_started = profile.then(Instant::now);
            let full_haplotype_view = composite::PhaseHaplotypeView::new(
                &snapshot,
                None,
                reference_packed,
                target_samples,
            )?;
            let stage1_snapshot = project_phased_window(&snapshot, &stage1_marker_indices)?;
            let packed_stage1_snapshot =
                pack_complete_biallelic_window(&stage1_snapshot, packed_cache_budget)?;
            let stage1_haplotype_view = composite::PhaseHaplotypeView::new(
                &stage1_snapshot,
                packed_stage1_snapshot.as_ref(),
                stage1_reference_packed,
                target_samples,
            )?;
            stage2_snapshot_elapsed +=
                snapshot_started.map_or(Duration::ZERO, |start| start.elapsed());
            let max_carriers = super::impute::beagle_rare_carrier_limit(n_samples);
            let low_frequency_allele_flags = &input_low_frequency_data.1;
            let max_backoff_steps = (0.3f32 / mosaic_segment_cm as f32).round_ties_even() as usize;
            let stage2_seed = stage2_phase_seed(seed, total_rounds);
            let pbwt_started = profile.then(Instant::now);
            let low_frequency_phase_pbwt_index =
                pbwt::build_low_frequency_phase_donor_index_view_with_pool(
                    &stage1_haplotype_view,
                    &full_haplotype_view,
                    Some(carrier_window),
                    &stage1_window.cm_positions,
                    &stage1_marker_indices,
                    mosaic_segment_cm,
                    max_carriers,
                    10,
                    max_backoff_steps,
                    ibd2_intervals,
                    stage2_seed,
                    pool,
                )?;
            stage2_pbwt_elapsed += pbwt_started.map_or(Duration::ZERO, |start| start.elapsed());
            let sample_started = profile.then(Instant::now);
            let updates = pool.install(|| {
                finished
                    .par_iter_mut()
                    .zip(trailing_masked.par_iter_mut())
                    .enumerate()
                    .map_init(
                        phase_baum::PhaseBaumWorkspace::default,
                        |phase_workspace, (sample, (finished_sample, trailing_masked_sample))| {
                            phase_sample(
                                window,
                                &snapshot,
                                &full_haplotype_view,
                                Some(&stage1_haplotype_view),
                                None,
                                None,
                                Some(&low_frequency_phase_pbwt_index),
                                Some(low_frequency_allele_flags),
                                sample,
                                finished_sample,
                                trailing_masked_sample,
                                total_rounds,
                                None,
                                None,
                                &full_marker_indices,
                                1.0,
                                any_unphased_heterozygote,
                                Some(mask),
                                low_frequency_hmm_markers.as_deref(),
                                low_frequency_hmm_cm_positions.as_deref(),
                                true,
                                Some(&stage2_sparse_plans[sample]),
                                &target_window.cm_positions,
                                &full_marker_indices,
                                mosaic_segment_cm,
                                ibd2_intervals,
                                &hmm_config,
                                low_frequency_phase_state_count(config.hmm.phase_states),
                                stage2_seed,
                                phase_workspace,
                            )
                        },
                    )
                    .collect::<Result<Vec<_>, _>>()
            })?;
            stage2_sample_elapsed += sample_started.map_or(Duration::ZERO, |start| start.elapsed());
            let PhasedWindow {
                chrom,
                index,
                mut rows,
                emit_start,
                emit_end,
            } = snapshot;
            let apply_started = profile.then(Instant::now);
            pool.install(|| {
                rows.par_iter_mut().for_each(|row| {
                    row.dosage = None;
                    row.ap1 = None;
                    row.ap2 = None;
                    row.gp = None;
                    row.info_af = None;
                    row.info_dr2 = None;
                    row.imputed = false;
                })
            });
            apply_sample_updates_in_place(&mut rows, &updates, &full_marker_indices, None)?;
            current = PhasedWindow {
                chrom,
                index,
                rows,
                emit_start,
                emit_end,
            };
            stage2_apply_elapsed += apply_started.map_or(Duration::ZERO, |start| start.elapsed());
            if stage2_stage_profile {
                let unix_s = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0.0, |duration| duration.as_secs_f64());
                if let Some((vm_rss_bytes, vm_hwm_bytes)) =
                    super::memory::read_process_rss_hwm_bytes()
                {
                    eprintln!(
                        "JANUSX_IMP_STAGE_TIMESTAMP stage=stage2_end window={}:{} unix_s={unix_s:.6} vm_rss_bytes={vm_rss_bytes} vm_hwm_bytes={vm_hwm_bytes}",
                        target_window.chrom, target_window.index
                    );
                } else {
                    eprintln!(
                        "JANUSX_IMP_STAGE_TIMESTAMP stage=stage2_end window={}:{} unix_s={unix_s:.6}",
                        target_window.chrom, target_window.index
                    );
                }
                let stage2_forward = hmm::stage2_forward_profile_totals();
                eprintln!(
                    "JANUSX_IMP_STAGE2_FORWARD_PROFILE window={}:{} calls={} checkpointed_calls={} budget_bytes={} max_selected_markers={} max_states={} max_full_forward_bytes={} max_active_forward_bytes={}",
                    target_window.chrom,
                    target_window.index,
                    stage2_forward
                        .calls
                        .saturating_sub(stage2_forward_start.calls),
                    stage2_forward
                        .checkpointed_calls
                        .saturating_sub(stage2_forward_start.checkpointed_calls),
                    stage2_forward.budget_bytes.unwrap_or(0),
                    stage2_forward.max_selected_markers,
                    stage2_forward.max_states,
                    stage2_forward.max_full_forward_bytes,
                    stage2_forward.max_active_forward_bytes,
                );
            }
        }
        stage2_elapsed += stage2_started.map_or(Duration::ZERO, |start| start.elapsed());
        if profile {
            eprintln!(
                "JANUSX_IMP_STAGE2_SPARSE_PROFILE window={}:{} skipped_samples={} unresolved_samples={} unresolved_markers={} anchor_markers={} full_anchor_markers={} plan_s={:.6} snapshot_s={:.6} pbwt_s={:.6} panel_s={:.6} panel_packed_calls={} panel_scalar_calls={} panel_packed_s={:.6} panel_scalar_s={:.6} callback_rows={} callback_target_donor_states={} callback_reference_fast_states={} callback_reference_fallback_states={} callback_missing_donor_states={} callback_target_het_rows={} callback_target_partial_or_missing_rows={} callback_probability_calculations={} callback_donor_source_reads={} callback_rare_flag_queries={} callback_marginal_additions={} callback_anchor_s={:.6} callback_interval_s={:.6} callback_weight_s={:.6} callback_row_s={:.6} hmm_s={:.6} result_s={:.6} samples_s={:.6} apply_s={:.6} total_s={:.6} shadow_runs={} shadow_hmm_s={:.6} shadow_result_s={:.6} shadow_matrix_bytes_sum={} shadow_anchor_posteriors={} shadow_marginal_cells={} shadow_gt_cells={}",
                target_window.chrom,
                target_window.index,
                stage2_skipped_samples,
                stage2_unresolved_samples,
                stage2_unresolved_markers,
                stage2_anchor_markers,
                target_samples.saturating_mul(stage1_marker_indices.len()),
                stage2_plan_elapsed.as_secs_f64(),
                stage2_snapshot_elapsed.as_secs_f64(),
                stage2_pbwt_elapsed.as_secs_f64(),
                (STAGE2_PANEL_NANOS.load(Ordering::Relaxed) - stage2_panel_start) as f64 / 1e9,
                (composite::stage2_panel_profile_totals()[2]
                    .saturating_sub(panel_detail_start[2])),
                (composite::stage2_panel_profile_totals()[3]
                    .saturating_sub(panel_detail_start[3])),
                (composite::stage2_panel_profile_totals()[0]
                    .saturating_sub(panel_detail_start[0])) as f64
                    / 1e9,
                (composite::stage2_panel_profile_totals()[1]
                    .saturating_sub(panel_detail_start[1])) as f64
                    / 1e9,
                STAGE2_CALLBACK_ROWS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_rows_start),
                STAGE2_CALLBACK_TARGET_DONOR_STATES
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_target_donor_start),
                STAGE2_CALLBACK_REFERENCE_FAST_STATES
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_ref_fast_start),
                STAGE2_CALLBACK_REFERENCE_FALLBACK_STATES
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_ref_fallback_start),
                STAGE2_CALLBACK_MISSING_DONOR_STATES
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_missing_donor_start),
                STAGE2_CALLBACK_TARGET_HET_ROWS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_target_het_start),
                STAGE2_CALLBACK_TARGET_PARTIAL_OR_MISSING_ROWS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_target_partial_start),
                STAGE2_CALLBACK_PROBABILITY_CALCULATIONS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_probability_start),
                STAGE2_CALLBACK_DONOR_SOURCE_READS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_donor_read_start),
                STAGE2_CALLBACK_RARE_FLAG_QUERIES
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_rare_query_start),
                STAGE2_CALLBACK_MARGINAL_ADDITIONS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_marginal_add_start),
                STAGE2_CALLBACK_ANCHOR_NANOS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_anchor_start) as f64
                    / 1e9,
                STAGE2_CALLBACK_INTERVAL_NANOS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_interval_start) as f64
                    / 1e9,
                STAGE2_CALLBACK_WEIGHT_NANOS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_weight_start) as f64
                    / 1e9,
                STAGE2_CALLBACK_ROW_NANOS
                    .load(Ordering::Relaxed)
                    .saturating_sub(callback_row_start) as f64
                    / 1e9,
                (STAGE2_HMM_NANOS.load(Ordering::Relaxed) - stage2_hmm_start) as f64 / 1e9,
                (STAGE2_RESULT_NANOS.load(Ordering::Relaxed) - stage2_result_start) as f64 / 1e9,
                stage2_sample_elapsed.as_secs_f64(),
                stage2_apply_elapsed.as_secs_f64(),
                stage2_elapsed.as_secs_f64(),
                STAGE2_SHADOW_RUNS
                    .load(Ordering::Relaxed)
                    .saturating_sub(shadow_runs_start),
                STAGE2_SHADOW_HMM_NANOS
                    .load(Ordering::Relaxed)
                    .saturating_sub(shadow_hmm_start) as f64
                    / 1e9,
                STAGE2_SHADOW_RESULT_NANOS
                    .load(Ordering::Relaxed)
                    .saturating_sub(shadow_result_start) as f64
                    / 1e9,
                STAGE2_SHADOW_MATRIX_BYTES
                    .load(Ordering::Relaxed)
                    .saturating_sub(shadow_matrix_bytes_start),
                STAGE2_SHADOW_ANCHOR_POSTERIORS
                    .load(Ordering::Relaxed)
                    .saturating_sub(shadow_anchor_posteriors_start),
                STAGE2_SHADOW_MARGINAL_CELLS
                    .load(Ordering::Relaxed)
                    .saturating_sub(shadow_marginal_cells_start),
                STAGE2_SHADOW_GT_CELLS
                    .load(Ordering::Relaxed)
                    .saturating_sub(shadow_gt_cells_start),
            );
        } else {
            let shadow_runs = STAGE2_SHADOW_RUNS
                .load(Ordering::Relaxed)
                .saturating_sub(shadow_runs_start);
            if shadow_runs > 0 {
                eprintln!(
                    "JANUSX_IMP_STAGE2_MISMATCH_SHADOW window={}:{} runs={} hmm_s={:.6} result_s={:.6} matrix_bytes_sum={} anchor_posteriors={} marginal_cells={} gt_cells={}",
                    target_window.chrom,
                    target_window.index,
                    shadow_runs,
                    STAGE2_SHADOW_HMM_NANOS
                        .load(Ordering::Relaxed)
                        .saturating_sub(shadow_hmm_start) as f64
                        / 1e9,
                    STAGE2_SHADOW_RESULT_NANOS
                        .load(Ordering::Relaxed)
                        .saturating_sub(shadow_result_start) as f64
                        / 1e9,
                    STAGE2_SHADOW_MATRIX_BYTES
                        .load(Ordering::Relaxed)
                        .saturating_sub(shadow_matrix_bytes_start),
                    STAGE2_SHADOW_ANCHOR_POSTERIORS
                        .load(Ordering::Relaxed)
                        .saturating_sub(shadow_anchor_posteriors_start),
                    STAGE2_SHADOW_MARGINAL_CELLS
                        .load(Ordering::Relaxed)
                        .saturating_sub(shadow_marginal_cells_start),
                    STAGE2_SHADOW_GT_CELLS
                        .load(Ordering::Relaxed)
                        .saturating_sub(shadow_gt_cells_start),
                );
            }
        }
    }
    if let Some(started) = profile_started {
        let pair_profile = phase_baum::pair_detail_profile_nanos();
        let prepare_detail = phase_baum::prepare_detail_profile_nanos();
        let mosaic_detail = composite::mosaic_detail_profile_nanos();
        let em_initialization_rounds = parameter_trace
            .iter()
            .filter(|trace| trace.stage == "em_init")
            .count();
        eprintln!(
            "JANUSX_IMP_PHASE_PROFILE window={}:{} rounds={} markers={} samples={} reference_samples={} total_haplotypes={} reference_packed_bytes={} donor_steps={} skipped_sample_updates={} packed_cache_budget={} packed_cache_bytes={} packed_cache_ready_rounds={} packed_cache_missing_rounds={} cluster_templates_s={:.6} initialize_s={:.6} low_frequency_s={:.6} pack_s={:.6} em_initialization_rounds={} em_initialize_s={:.6} em_mosaic_s={:.6} em_hmm_s={:.6} donor_index_s={:.6} em_update_s={:.6} sample_update_s={:.6} sample_mosaic_s={:.6} sample_select_s={:.6} sample_fill_s={:.6} sample_setup_s={:.6} sample_setup_input_s={:.6} sample_setup_panel_s={:.6} sample_fallback_s={:.6} sample_source_s={:.6} sample_hmm_s={:.6} pair_prepare_s={:.6} pair_backward_s={:.6} pair_forward_s={:.6} rebuild_s={:.6} packed_refresh_s={:.6} stage2_s={:.6} total_s={:.6}",
            target_window.chrom,
            target_window.index,
            total_rounds,
            n_markers,
            target_samples,
            reference_samples,
            n_samples.saturating_mul(2),
            reference_packed_bytes,
            donor_step_count,
            skipped_sample_updates,
            packed_cache_budget,
            packed_cache_bytes,
            packed_cache_enabled_rounds,
            packed_cache_disabled_rounds,
            cluster_templates_elapsed.as_secs_f64(),
            initialize_elapsed.as_secs_f64(),
            low_frequency_elapsed.as_secs_f64(),
            pack_elapsed.as_secs_f64(),
            em_initialization_rounds,
            em_initialize_elapsed.as_secs_f64(),
            (EM_MOSAIC_NANOS.load(Ordering::Relaxed) - em_mosaic_start) as f64 / 1e9,
            (EM_HMM_NANOS.load(Ordering::Relaxed) - em_hmm_start) as f64 / 1e9,
            donor_index_elapsed.as_secs_f64(),
            em_update_elapsed.as_secs_f64(),
            sample_update_elapsed.as_secs_f64(),
            (SAMPLE_MOSAIC_NANOS.load(Ordering::Relaxed) - profile_mosaic_start) as f64 / 1e9,
            sample_mosaic_detail[0] as f64 / 1e9,
            sample_mosaic_detail[1] as f64 / 1e9,
            sample_mosaic_detail[2] as f64 / 1e9,
            sample_mosaic_detail[5] as f64 / 1e9,
            sample_mosaic_detail[6] as f64 / 1e9,
            sample_mosaic_detail[3] as f64 / 1e9,
            sample_mosaic_detail[4] as f64 / 1e9,
            (SAMPLE_HMM_NANOS.load(Ordering::Relaxed) - profile_hmm_start) as f64 / 1e9,
            pair_profile[0].saturating_sub(pair_profile_start[0]) as f64 / 1e9,
            pair_profile[1].saturating_sub(pair_profile_start[1]) as f64 / 1e9,
            pair_profile[2].saturating_sub(pair_profile_start[2]) as f64 / 1e9,
            rebuild_elapsed.as_secs_f64(),
            packed_refresh_elapsed.as_secs_f64(),
            stage2_elapsed.as_secs_f64(),
            started.elapsed().as_secs_f64(),
        );
        eprintln!(
            "JANUSX_IMP_PREPARE_PROFILE clusters_s={:.6} transitions_s={:.6} emissions_s={:.6} workspace_s={:.6}",
            prepare_detail[0].saturating_sub(prepare_detail_start[0]) as f64 / 1e9,
            prepare_detail[1].saturating_sub(prepare_detail_start[1]) as f64 / 1e9,
            prepare_detail[2].saturating_sub(prepare_detail_start[2]) as f64 / 1e9,
            prepare_detail[3].saturating_sub(prepare_detail_start[3]) as f64 / 1e9,
        );
        eprintln!(
            "JANUSX_IMP_MOSAIC_PROFILE select_s={:.6} fill_s={:.6} setup_s={:.6} fallback_s={:.6} source_s={:.6} setup_input_s={:.6} setup_panel_s={:.6}",
            mosaic_detail[0].saturating_sub(mosaic_detail_start[0]) as f64 / 1e9,
            mosaic_detail[1].saturating_sub(mosaic_detail_start[1]) as f64 / 1e9,
            mosaic_detail[2].saturating_sub(mosaic_detail_start[2]) as f64 / 1e9,
            mosaic_detail[3].saturating_sub(mosaic_detail_start[3]) as f64 / 1e9,
            mosaic_detail[4].saturating_sub(mosaic_detail_start[4]) as f64 / 1e9,
            mosaic_detail[5].saturating_sub(mosaic_detail_start[5]) as f64 / 1e9,
            mosaic_detail[6].saturating_sub(mosaic_detail_start[6]) as f64 / 1e9,
        );
    }
    let count_totals = phase_baum::count_profile_totals();
    let count_deltas: [u64; 12] = std::array::from_fn(|index| {
        count_totals[index].saturating_sub(phase_count_profile_start[index])
    });
    if count_deltas.iter().any(|&value| value != 0) {
        let phase_haplotype_state_updates = count_deltas[3];
        let phase_forward_track_state_updates = phase_haplotype_state_updates.saturating_mul(3);
        let phase_backward_track_state_updates = phase_haplotype_state_updates.saturating_mul(3);
        let memory_profile = phase_baum::memory_profile_totals();
        let checkpointed_preparations = memory_profile
            .checkpointed_preparations
            .saturating_sub(phase_memory_profile_start.checkpointed_preparations);
        eprintln!(
            "JANUSX_IMP_PHASE_COUNTS_PROFILE window={}:{} sample_updates={} clusters={} states={} state_cluster_updates={phase_haplotype_state_updates} phase_haplotype_state_updates={phase_haplotype_state_updates} phase_forward_track_state_updates={phase_forward_track_state_updates} phase_backward_track_state_updates={phase_backward_track_state_updates} unphased_events={} missing_or_masked={} missing_events={} masked_heterozygote_events={} emission_rows={} beta_cache_allocated_bytes={} process_peak_beta_cache_bytes={} single_track_preparations={} window_aggregate_peak_beta_cache_bytes={} process_aggregate_peak_beta_cache_bytes={} aggregate_live_beta_cache_bytes={} window_aggregate_peak_event_workspace_bytes={} process_aggregate_peak_event_workspace_bytes={} aggregate_live_event_workspace_bytes={} checkpointed_preparations={}",
            target_window.chrom,
            target_window.index,
            count_deltas[0],
            count_deltas[1],
            count_deltas[2],
            count_deltas[4],
            count_deltas[5],
            count_deltas[6],
            count_deltas[7],
            count_deltas[8],
            count_deltas[9],
            count_totals[10],
            count_deltas[11],
            memory_profile.window_peak_live_beta_cache_bytes,
            memory_profile.process_peak_live_beta_cache_bytes,
            memory_profile.live_beta_cache_bytes,
            memory_profile.window_peak_live_event_workspace_bytes,
            memory_profile.process_peak_live_event_workspace_bytes,
            memory_profile.live_event_workspace_bytes,
            checkpointed_preparations,
        );
    }
    Ok((current, hmm_config, parameter_trace, hmm_trace))
}

fn phase_sample(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    haplotype_view: &composite::PhaseHaplotypeView<'_>,
    packed_stage1_view: Option<&composite::PhaseHaplotypeView<'_>>,
    pbwt_index: Option<&pbwt::PbwtIndex>,
    phase_pbwt_index: Option<&pbwt::PhasePbwtIndex>,
    low_frequency_phase_pbwt_index: Option<&pbwt::LowFrequencyPhasePbwtIndex>,
    low_frequency_allele_flags: Option<&[Vec<bool>]>,
    sample: usize,
    finished: &mut [u64],
    persistent_trailing_mask: &mut [u64],
    phase_iteration: usize,
    diagnostic_site: Option<&ResolvedDiagnosticSite>,
    diagnostic_marker: Option<usize>,
    phase_marker_indices: &[usize],
    lr_threshold: f64,
    any_unphased_heterozygote: bool,
    low_frequency_mask: Option<&[bool]>,
    low_frequency_hmm_markers: Option<&[usize]>,
    low_frequency_hmm_cm_positions: Option<&[f64]>,
    low_frequency_stage: bool,
    stage2_sparse_plan: Option<&Stage2SparsePlan>,
    cm_positions: &[f64],
    full_marker_indices: &[usize],
    mosaic_segment_cm: f64,
    ibd2_intervals: &[super::types::Ibd2Interval],
    config: &HMMConfig,
    n_states: usize,
    seed: u64,
    phase_workspace: &mut phase_baum::PhaseBaumWorkspace,
) -> Result<SampleUpdate, String> {
    if !low_frequency_stage
        && diagnostic_site.is_none()
        && !phase_baum::sample_requires_phase_baum(
            input,
            sample,
            finished,
            any_unphased_heterozygote,
        )
    {
        return Ok(unchanged_sample_update(snapshot, sample));
    }
    if low_frequency_stage
        && stage2_sparse_plan.is_some_and(|plan| plan.unresolved_markers.is_empty())
    {
        return Ok(unchanged_sample_update(snapshot, sample));
    }
    let profile_sample = sample == 0 && std::env::var_os("JANUSX_IMP_DETAIL_PROFILE").is_some();
    let profile_started = profile_sample.then(Instant::now);
    let diagnostic_marker_for_sample = diagnostic_site
        .filter(|site| site.sample_index == sample)
        .and(diagnostic_marker);
    if !low_frequency_stage {
        let phase_pbwt_index =
            phase_pbwt_index.ok_or_else(|| "phase PBWT donor index is missing".to_owned())?;
        let minimum_lifetime_steps = ((1.0f32 / mosaic_segment_cm as f32).ceil() as usize).max(200);
        let mosaic_started = profile_sample.then(Instant::now);
        let panel = composite::build_joint_phase_mosaic_view(
            haplotype_view,
            phase_pbwt_index,
            sample,
            n_states,
            minimum_lifetime_steps,
            seed,
        )?;
        let mosaic_elapsed = mosaic_started.map_or(Duration::ZERO, |start| start.elapsed());
        if let (Ok(target), Ok(prefix)) = (
            std::env::var("JANUSX_TRACE_PANEL_TARGET"),
            std::env::var("JANUSX_TRACE_PANEL_PREFIX"),
        ) {
            let trace_all_samples = std::env::var("JANUSX_TRACE_PANEL_ALL_SAMPLES")
                .is_ok_and(|window| window == input.index.to_string());
            if target == format!("{}:{}", input.index, sample) || trace_all_samples {
                use std::fmt::Write as _;
                let mut contents = String::with_capacity(
                    input
                        .rows
                        .len()
                        .saturating_mul(panel.haplotypes.n_haplotypes + 32),
                );
                writeln!(
                    &mut contents,
                    "CHROM\tPOS\t{}",
                    (0..panel.haplotypes.n_haplotypes)
                        .map(|state| format!("STATE{state}"))
                        .collect::<Vec<_>>()
                        .join("\t")
                )
                .unwrap();
                for (marker, row) in input.rows.iter().enumerate() {
                    write!(&mut contents, "{}\t{}", input.chrom, row.meta.pos).unwrap();
                    for state in 0..panel.haplotypes.n_haplotypes {
                        match panel.haplotypes.allele(marker, state) {
                            Some(allele) => write!(&mut contents, "\t{allele}").unwrap(),
                            None => contents.push_str("\t."),
                        }
                    }
                    contents.push('\n');
                }
                let path = format!(
                    "{prefix}.window{}.round{}.sample{}.tsv",
                    input.index, phase_iteration, sample
                );
                std::fs::write(path, contents).map_err(|error| {
                    format!("cannot write temporary phase-panel trace: {error}")
                })?;
            }
        }
        let trace_marker = diagnostic_marker_for_sample.and_then(|marker| {
            phase_marker_indices
                .iter()
                .position(|&full_marker| full_marker == marker)
        });
        let mut phase_finished = vec![0u64; input.rows.len().div_ceil(64)];
        for (local, &full_marker) in phase_marker_indices.iter().enumerate() {
            if bit_is_set(finished, full_marker) {
                mark_finished(&mut phase_finished, local);
            }
        }
        let mut diagnostic_trace = Vec::new();
        update_trailing_unphased_het_mask(
            input,
            sample,
            &phase_finished,
            lr_threshold < 50.0,
            persistent_trailing_mask,
        );
        let update = phase_baum::update_sample_with_diagnostic_and_workspace(
            input,
            snapshot,
            sample,
            &phase_finished,
            any_unphased_heterozygote,
            persistent_trailing_mask,
            &panel,
            config,
            trace_marker,
            phase_workspace,
            |trace| {
                let Some(site) = diagnostic_site else {
                    return;
                };
                diagnostic_trace.push(ImpHmmTrace {
                    chrom: input.chrom.clone(),
                    position: input.rows[trace.marker].meta.pos,
                    sample: site.sample.clone(),
                    window_index: input.index,
                    round: phase_iteration,
                    cluster_kind: trace.cluster_kind.map(str::to_owned),
                    transition_probability: trace.transition_probability,
                    state_alleles: vec![trace.state_alleles.clone(); 3],
                    emission_probabilities: trace.emission_probabilities,
                    forward_probabilities: trace.forward_probabilities,
                    backward_probabilities: trace.backward_probabilities,
                    posterior_normalization: trace.posterior_normalization,
                    phase_swap: trace.phase_swap,
                    phase_log_odds: trace.phase_log_odds,
                    phase_threshold: lr_threshold,
                    selected_alleles: trace.selected_alleles,
                });
            },
        )?;
        if let Some(started) = profile_started {
            eprintln!(
                "JANUSX_IMP_PHASE_HMM_PROFILE window={}:{} round={} states={} markers={} mosaic_s={:.6} hmm_s={:.6} total_s={:.6}",
                input.chrom,
                input.index,
                phase_iteration,
                panel.haplotypes.n_haplotypes,
                input.rows.len(),
                mosaic_elapsed.as_secs_f64(),
                started.elapsed().as_secs_f64() - mosaic_elapsed.as_secs_f64(),
                started.elapsed().as_secs_f64(),
            );
        }
        if any_unphased_heterozygote && !update.phase_confidences.is_empty() {
            let log_threshold = lr_threshold.ln();
            for &(local_marker, log_likelihood_ratio) in &update.phase_confidences {
                if lr_threshold.is_finite() && log_likelihood_ratio >= log_threshold {
                    if let Some(&full_marker) = phase_marker_indices.get(local_marker) {
                        mark_finished(finished, full_marker);
                    }
                }
            }
        }
        let unphased_heterozygote_count = update.unphased_heterozygote_count;
        let phase_swap_count = update.phase_swap_count;
        return Ok(SampleUpdate {
            calls: sample_call_update(update),
            diagnostic_trace,
            unphased_heterozygote_count,
            phase_swap_count,
        });
    }

    let trace_marker = diagnostic_marker_for_sample.and_then(|marker| {
        phase_marker_indices
            .iter()
            .position(|&full_marker| full_marker == marker)
    });
    let n_markers = input.rows.len();
    let max_allele = input
        .rows
        .iter()
        .map(|row| row.meta.allele_count().saturating_sub(1) as u16)
        .max()
        .unwrap_or(1);
    let width = allele_width(max_allele);
    let mut observations0 = PackedHaplotypes::empty(1, n_markers, width)?;
    let mut observations1 = PackedHaplotypes::empty(1, n_markers, width)?;
    let mut needs_hap1 = false;
    for (marker, row) in input.rows.iter().enumerate() {
        if low_frequency_stage {
            // Beagle's Stage2 HMM is built from CodedSteps.allHaps(), which
            // contains the phased Stage1 output rather than the original
            // target calls. It also builds independent state panels for both
            // target haplotypes even when their Stage1 marker sequences happen
            // to be homozygous.
            needs_hap1 = true;
            if low_frequency_mask
                .ok_or_else(|| "low-frequency stage is missing its marker mask".to_owned())?[marker]
            {
                continue;
            }
            let [first, second] = snapshot.rows[marker].calls.get(sample).unwrap().alleles;
            if first.is_none() && second.is_none() {
                continue;
            }
            observations0.set_allele(marker, 0, first)?;
            observations1.set_allele(marker, 0, second)?;
            continue;
        }
        if low_frequency_mask.is_some_and(|mask| mask.get(marker).copied().unwrap_or(false)) {
            needs_hap1 = true;
            continue;
        }
        let call = row.calls.get(sample).unwrap();
        match call.alleles {
            [Some(a), Some(b)] if a == b => {
                observations0.set_allele(marker, 0, Some(a))?;
                observations1.set_allele(marker, 0, Some(a))?;
            }
            [Some(a), Some(b)] if call.phased => {
                observations0.set_allele(marker, 0, Some(a))?;
                observations1.set_allele(marker, 0, Some(b))?;
                needs_hap1 = true;
            }
            [Some(_), Some(_)] => {
                let fixed = snapshot.rows[marker].calls.get(sample).unwrap();
                observations0.set_allele(marker, 0, fixed.alleles[0])?;
                observations1.set_allele(marker, 0, fixed.alleles[1])?;
                needs_hap1 = true;
            }
            [Some(a), None] => {
                observations0.set_allele(marker, 0, Some(a))?;
                needs_hap1 = true;
            }
            [None, Some(b)] => {
                // Preserve the observed allele's slot when phase is known;
                // otherwise place it on the first haplotype consistently.
                if call.phased {
                    observations1.set_allele(marker, 0, Some(b))?;
                    needs_hap1 = true;
                } else {
                    observations0.set_allele(marker, 0, Some(b))?;
                    needs_hap1 = true;
                }
            }
            [None, None] => needs_hap1 = true,
        }
    }

    if low_frequency_stage {
        if let Some(low_frequency_pbwt_index) = low_frequency_phase_pbwt_index {
            return phase_low_frequency_sample_beagle(
                input,
                snapshot,
                haplotype_view,
                packed_stage1_view,
                sample,
                diagnostic_site,
                diagnostic_marker_for_sample,
                trace_marker,
                observations0,
                observations1,
                low_frequency_mask
                    .ok_or_else(|| "low-frequency stage is missing its marker mask".to_owned())?,
                low_frequency_hmm_markers.ok_or_else(|| {
                    "low-frequency stage is missing its Stage1 markers".to_owned()
                })?,
                low_frequency_hmm_cm_positions
                    .ok_or_else(|| "low-frequency stage is missing its Stage1 map".to_owned())?,
                cm_positions,
                low_frequency_allele_flags
                    .ok_or_else(|| "low-frequency allele flags are missing".to_owned())?,
                low_frequency_pbwt_index,
                stage2_sparse_plan.ok_or_else(|| {
                    "low-frequency Stage2 is missing its sparse marker plan".to_owned()
                })?,
                config,
                n_states,
                mosaic_segment_cm,
                phase_iteration,
                lr_threshold,
                seed,
                &mut phase_workspace.low_frequency_hmm,
            );
        }
    }

    let reference_panel = composite::build_phase_reference_view(
        haplotype_view,
        input,
        sample,
        n_states,
        ibd2_intervals,
        low_frequency_mask,
        pbwt_index.ok_or_else(|| "legacy scaffold PBWT index is missing".to_owned())?,
    )?;
    let allele_counts = low_frequency_stage.then(|| {
        input
            .rows
            .iter()
            .map(|row| row.meta.allele_count())
            .collect::<Vec<_>>()
    });
    let sparse_hmm_markers = if low_frequency_stage {
        low_frequency_hmm_markers
    } else {
        None
    };
    let sparse_hmm_cm_positions = if low_frequency_stage {
        low_frequency_hmm_cm_positions
    } else {
        None
    };
    let mut panel0 = composite::build_target_mosaic(
        &reference_panel,
        &observations0,
        full_marker_indices,
        cm_positions,
        mosaic_segment_cm,
    )?;
    let mut workspace0 = Default::default();
    let posterior0 = phase_forward_backward(
        &observations0,
        &mut panel0,
        cm_positions,
        sparse_hmm_markers,
        sparse_hmm_cm_positions,
        allele_counts.as_deref(),
        config,
        &mut workspace0,
    )?;
    let diagnostic0 = if let Some(marker) = trace_marker {
        trace_hmm_marker(&observations0, &panel0, cm_positions, marker, config)?
    } else {
        None
    };
    drop(panel0);
    drop(workspace0);
    let mut diagnostic1 = None;
    let posterior1 = if needs_hap1 {
        let mut panel1 = composite::build_target_mosaic(
            &reference_panel,
            &observations1,
            full_marker_indices,
            cm_positions,
            mosaic_segment_cm,
        )?;
        let mut workspace1 = Default::default();
        let posterior = phase_forward_backward(
            &observations1,
            &mut panel1,
            cm_positions,
            sparse_hmm_markers,
            sparse_hmm_cm_positions,
            allele_counts.as_deref(),
            config,
            &mut workspace1,
        )?;
        diagnostic1 = if let Some(marker) = trace_marker {
            trace_hmm_marker(&observations1, &panel1, cm_positions, marker, config)?
        } else {
            None
        };
        drop(panel1);
        drop(workspace1);
        Some(posterior)
    } else {
        None
    };
    let mut result = Vec::with_capacity(n_markers);
    let mut phase_confidences = Vec::new();
    let mut rng = seed;
    for (marker, row) in input.rows.iter().enumerate() {
        let call = row.calls.get(sample).unwrap();
        let p0 = posterior0
            .allele_probabilities_at(marker)
            .ok_or_else(|| "missing first-haplotype allele posterior".to_owned())?;
        let p1 = match posterior1.as_ref() {
            Some(posterior) => posterior
                .allele_probabilities_at(marker)
                .ok_or_else(|| "missing second-haplotype allele posterior".to_owned())?,
            None => p0,
        };
        let p0_without_marker = posterior0
            .leave_one_out_allele_probabilities_at(marker)
            .unwrap_or(p0);
        let p1_without_marker = match posterior1.as_ref() {
            Some(posterior) => posterior
                .leave_one_out_allele_probabilities_at(marker)
                .unwrap_or(p1),
            None => p0_without_marker,
        };
        let is_low_frequency =
            low_frequency_mask.is_some_and(|mask| mask.get(marker).copied().unwrap_or(false));
        let alleles = if low_frequency_stage && !is_low_frequency {
            snapshot.rows[marker].calls.get(sample).unwrap().alleles
        } else if is_low_frequency && !low_frequency_stage {
            snapshot.rows[marker].calls.get(sample).unwrap().alleles
        } else if is_low_frequency {
            phase_low_frequency_call(
                call,
                snapshot.rows[marker].calls.get(sample).unwrap(),
                p0,
                p1,
                any_unphased_heterozygote,
                &mut rng,
            )
        } else {
            match call.alleles {
                [Some(a), Some(b)] if a == b => [Some(a), Some(a)],
                [Some(a), Some(b)] if call.phased => [Some(a), Some(b)],
                [Some(a), Some(b)] => {
                    let oriented = if bit_is_set(finished, marker) {
                        let fixed = snapshot.rows[marker].calls.get(sample).unwrap();
                        fixed.alleles[0].unwrap_or(a)
                    } else {
                        let (oriented, confidence) = diplotype_orientation(
                            p0_without_marker,
                            p1_without_marker,
                            a,
                            b,
                            &mut rng,
                        );
                        phase_confidences.push((marker, confidence));
                        oriented
                    };
                    let other = if oriented == a { b } else { a };
                    [Some(oriented), Some(other)]
                }
                [Some(a), None] => [Some(a), Some(sample_allele(p1, &mut rng))],
                [None, Some(b)] if call.phased => [Some(sample_allele(p0, &mut rng)), Some(b)],
                [None, Some(b)] => [Some(b), Some(sample_allele(p1, &mut rng))],
                [None, None] => [Some(allele_map(p0)), Some(allele_map(p1))],
            }
        };
        result.push(DiploidCall {
            alleles,
            phased: true,
        });
    }
    if any_unphased_heterozygote && !phase_confidences.is_empty() {
        let log_threshold = lr_threshold.ln();
        for &(marker, log_likelihood_ratio) in &phase_confidences {
            if lr_threshold.is_finite() && log_likelihood_ratio >= log_threshold {
                mark_finished(finished, marker);
            }
        }
    }
    let mut diagnostic_trace = Vec::new();
    if let (Some(marker), Some(diagnostic0), Some(diagnostic1), Some(site)) =
        (trace_marker, diagnostic0, diagnostic1, diagnostic_site)
    {
        diagnostic_trace.push(ImpHmmTrace {
            chrom: input.chrom.clone(),
            position: input.rows[marker].meta.pos,
            sample: site.sample.clone(),
            window_index: input.index,
            round: phase_iteration,
            cluster_kind: None,
            transition_probability: diagnostic0.recursion.transition_probability,
            state_alleles: vec![diagnostic0.state_alleles, diagnostic1.state_alleles],
            emission_probabilities: vec![
                diagnostic0.recursion.emission_probabilities,
                diagnostic1.recursion.emission_probabilities,
            ],
            forward_probabilities: vec![
                diagnostic0.recursion.forward_probabilities,
                diagnostic1.recursion.forward_probabilities,
            ],
            backward_probabilities: vec![
                diagnostic0.recursion.backward_probabilities,
                diagnostic1.recursion.backward_probabilities,
            ],
            posterior_normalization: vec![
                diagnostic0.recursion.posterior_normalization as f64,
                diagnostic1.recursion.posterior_normalization as f64,
            ],
            phase_swap: None,
            phase_log_odds: None,
            phase_threshold: lr_threshold,
            selected_alleles: result[marker].alleles,
        });
    }
    Ok(SampleUpdate {
        calls: SampleCallUpdate::Full(DiploidCalls::from(result)),
        diagnostic_trace,
        unphased_heterozygote_count: 0,
        phase_swap_count: 0,
    })
}

#[cfg(test)]
fn stage2_observation_alleles(
    snapshot: &PhasedWindow,
    low_frequency_mask: &[bool],
    sample: usize,
) -> Result<Vec<[Option<u16>; 2]>, String> {
    if snapshot.rows.len() != low_frequency_mask.len() {
        return Err("Stage2 marker mask does not match the phased snapshot".to_owned());
    }
    snapshot
        .rows
        .iter()
        .zip(low_frequency_mask)
        .map(|(row, &is_low_frequency)| {
            if is_low_frequency {
                return Ok([None, None]);
            }
            row.calls
                .get(sample)
                .map(|call| call.alleles)
                .ok_or_else(|| "Stage2 observation sample is out of range".to_owned())
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn phase_sample_pair_event(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    haplotype_view: &composite::PhaseHaplotypeView<'_>,
    phase_pbwt_index: &pbwt::PhasePbwtIndex,
    samples: [usize; 2],
    cluster_template0: &phase_baum::PhaseClusterTemplate,
    cluster_template1: &phase_baum::PhaseClusterTemplate,
    finished: [&mut [u64]; 2],
    trailing_masked: [&mut [u64]; 2],
    any_unphased_heterozygote: bool,
    phase_marker_indices: &[usize],
    lr_threshold: f64,
    mosaic_segment_cm: f64,
    config: &HMMConfig,
    n_states: usize,
    transition_factors: &[f32],
    seeds: [u64; 2],
    packed_snapshot: Option<&PackedHaplotypes>,
    workspace0: &mut PhaseWorkerWorkspace,
    workspace1: &mut PhaseWorkerWorkspace,
) -> Result<(SampleUpdate, SampleUpdate, usize), String> {
    let needs_update = [
        phase_baum::sample_requires_phase_baum(
            input,
            samples[0],
            finished[0],
            any_unphased_heterozygote,
        ),
        phase_baum::sample_requires_phase_baum(
            input,
            samples[1],
            finished[1],
            any_unphased_heterozygote,
        ),
    ];
    if !needs_update[0] && !needs_update[1] {
        return Ok((
            unchanged_sample_update(snapshot, samples[0]),
            unchanged_sample_update(snapshot, samples[1]),
            2,
        ));
    }
    if !needs_update[0] || !needs_update[1] {
        let active = usize::from(!needs_update[0]);
        let workspace = if active == 0 { workspace0 } else { workspace1 };
        let update = phase_sample(
            input,
            snapshot,
            haplotype_view,
            None,
            None,
            Some(phase_pbwt_index),
            None,
            None,
            samples[active],
            finished[active],
            trailing_masked[active],
            0,
            None,
            None,
            phase_marker_indices,
            lr_threshold,
            any_unphased_heterozygote,
            None,
            None,
            None,
            false,
            None,
            &input.cm_positions,
            phase_marker_indices,
            mosaic_segment_cm,
            &[],
            config,
            n_states,
            seeds[active],
            &mut workspace.baum,
        )?;
        return if active == 0 {
            Ok((update, unchanged_sample_update(snapshot, samples[1]), 1))
        } else {
            Ok((unchanged_sample_update(snapshot, samples[0]), update, 1))
        };
    }
    let minimum_lifetime_steps = ((1.0f32 / mosaic_segment_cm as f32).ceil() as usize).max(200);
    let profile = phase_profiling_enabled();
    let mosaic_started = profile.then(Instant::now);
    composite::build_joint_phase_mosaic_reuse_validated_view(
        haplotype_view,
        phase_pbwt_index,
        samples[0],
        n_states,
        minimum_lifetime_steps,
        seeds[0],
        &mut workspace0.panel,
        &mut workspace0.donors,
    )?;
    composite::build_joint_phase_mosaic_reuse_validated_view(
        haplotype_view,
        phase_pbwt_index,
        samples[1],
        n_states,
        minimum_lifetime_steps,
        seeds[1],
        &mut workspace1.panel,
        &mut workspace1.donors,
    )?;
    if let Some(started) = mosaic_started {
        SAMPLE_MOSAIC_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let panel0 = workspace0
        .panel
        .as_ref()
        .ok_or_else(|| "first paired phase panel was not constructed".to_owned())?;
    let panel1 = workspace1
        .panel
        .as_ref()
        .ok_or_else(|| "second paired phase panel was not constructed".to_owned())?;
    let mut phase_finished0 = vec![0u64; input.rows.len().div_ceil(64)];
    let mut phase_finished1 = vec![0u64; input.rows.len().div_ceil(64)];
    for (local, &full_marker) in phase_marker_indices.iter().enumerate() {
        if bit_is_set(finished[0], full_marker) {
            mark_finished(&mut phase_finished0, local);
        }
        if bit_is_set(finished[1], full_marker) {
            mark_finished(&mut phase_finished1, local);
        }
    }
    update_trailing_unphased_het_mask(
        input,
        samples[0],
        &phase_finished0,
        lr_threshold < 50.0,
        trailing_masked[0],
    );
    update_trailing_unphased_het_mask(
        input,
        samples[1],
        &phase_finished1,
        lr_threshold < 50.0,
        trailing_masked[1],
    );
    let hmm_started = profile.then(Instant::now);
    let (update0, update1) = phase_baum::update_sample_pair(
        input,
        snapshot,
        samples[0],
        samples[1],
        cluster_template0,
        cluster_template1,
        &phase_finished0,
        &phase_finished1,
        any_unphased_heterozygote,
        trailing_masked[0],
        trailing_masked[1],
        panel0,
        panel1,
        packed_snapshot,
        transition_factors,
        config,
        &mut workspace0.baum,
        &mut workspace1.baum,
    )?;
    if let Some(started) = hmm_started {
        SAMPLE_HMM_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let finish_update =
        |finished: &mut [u64], update: phase_baum::JointPhaseUpdate| -> SampleUpdate {
            if any_unphased_heterozygote && !update.phase_confidences.is_empty() {
                let log_threshold = lr_threshold.ln();
                for &(local_marker, log_likelihood_ratio) in &update.phase_confidences {
                    if lr_threshold.is_finite() && log_likelihood_ratio >= log_threshold {
                        if let Some(&full_marker) = phase_marker_indices.get(local_marker) {
                            mark_finished(finished, full_marker);
                        }
                    }
                }
            }
            let unphased_heterozygote_count = update.unphased_heterozygote_count;
            let phase_swap_count = update.phase_swap_count;
            SampleUpdate {
                calls: sample_call_update(update),
                diagnostic_trace: Vec::new(),
                unphased_heterozygote_count,
                phase_swap_count,
            }
        };
    Ok((
        finish_update(finished[0], update0),
        finish_update(finished[1], update1),
        0,
    ))
}

fn phase_low_frequency_sample_beagle(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    haplotype_view: &composite::PhaseHaplotypeView<'_>,
    packed_stage1_view: Option<&composite::PhaseHaplotypeView<'_>>,
    sample: usize,
    diagnostic_site: Option<&ResolvedDiagnosticSite>,
    diagnostic_marker_for_sample: Option<usize>,
    trace_marker: Option<usize>,
    observations0: PackedHaplotypes,
    observations1: PackedHaplotypes,
    low_frequency_mask: &[bool],
    stage1_marker_indices: &[usize],
    stage1_cm_positions: &[f64],
    full_cm_positions: &[f64],
    low_frequency_allele_flags: &[Vec<bool>],
    pbwt_index: &pbwt::LowFrequencyPhasePbwtIndex,
    sparse_plan: &Stage2SparsePlan,
    config: &HMMConfig,
    n_states: usize,
    mosaic_segment_cm: f64,
    phase_iteration: usize,
    lr_threshold: f64,
    seed: u64,
    hmm_workspaces: &mut [HMMWorkspace; 2],
) -> Result<SampleUpdate, String> {
    if low_frequency_mask.len() != input.rows.len()
        || low_frequency_allele_flags.len() != input.rows.len()
        || full_cm_positions.len() != input.rows.len()
    {
        return Err("Stage2 marker arrays do not match the input window".to_owned());
    }
    if sparse_plan.unresolved_markers.is_empty() {
        return Ok(unchanged_sample_update(snapshot, sample));
    }
    let minimum_lifetime_steps = ((1.0f32 / mosaic_segment_cm as f32).ceil() as usize).max(200);
    let profile = phase_profiling_enabled();
    let panel_started = profile.then(Instant::now);
    let (mut panel0, donor_segments0) = if let Some(packed_stage1_view) = packed_stage1_view {
        composite::build_low_frequency_phase_mosaic_view_on_markers_with_stage1_view(
            haplotype_view,
            packed_stage1_view,
            pbwt_index,
            sample,
            0,
            n_states,
            minimum_lifetime_steps,
            seed,
            stage1_marker_indices,
        )?
    } else {
        composite::build_low_frequency_phase_mosaic_view_on_markers(
            haplotype_view,
            pbwt_index,
            sample,
            0,
            n_states,
            minimum_lifetime_steps,
            seed,
            stage1_marker_indices,
        )?
    };
    let (mut panel1, donor_segments1) = if let Some(packed_stage1_view) = packed_stage1_view {
        composite::build_low_frequency_phase_mosaic_view_on_markers_with_stage1_view(
            haplotype_view,
            packed_stage1_view,
            pbwt_index,
            sample,
            1,
            n_states,
            minimum_lifetime_steps,
            seed,
            stage1_marker_indices,
        )?
    } else {
        composite::build_low_frequency_phase_mosaic_view_on_markers(
            haplotype_view,
            pbwt_index,
            sample,
            1,
            n_states,
            minimum_lifetime_steps,
            seed,
            stage1_marker_indices,
        )?
    };
    if let Some(started) = panel_started {
        STAGE2_PANEL_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let shadow_enabled = stage2_mismatch_shadow_enabled();
    let (hmm_workspace0, hmm_workspace1) = hmm_workspaces.split_at_mut(1);
    let hmm_started = profile.then(Instant::now);
    let production = low_frequency_haplotype_marginals_pair(
        input,
        haplotype_view,
        sample,
        &observations0,
        &mut panel0,
        &observations1,
        &mut panel1,
        &donor_segments0,
        &donor_segments1,
        sparse_plan,
        low_frequency_mask,
        low_frequency_allele_flags,
        stage1_marker_indices,
        stage1_cm_positions,
        full_cm_positions,
        config,
        &mut hmm_workspace0[0],
        &mut hmm_workspace1[0],
        Stage2HmmBackend::Production,
        shadow_enabled,
    )?;
    if let Some(started) = hmm_started {
        STAGE2_HMM_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let diagnostic0 = if let Some(marker) = trace_marker {
        trace_hmm_marker(&observations0, &panel0, full_cm_positions, marker, config)?
    } else {
        None
    };
    let diagnostic1 = if let Some(marker) = trace_marker {
        trace_hmm_marker(&observations1, &panel1, full_cm_positions, marker, config)?
    } else {
        None
    };

    let result_started = profile.then(Instant::now);
    let result = stage2_sample_calls_from_marginals(
        input,
        snapshot,
        sample,
        low_frequency_mask,
        &production.marginals,
        seed,
    )?;
    if let Some(started) = result_started {
        STAGE2_RESULT_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let mut diagnostic_trace = Vec::new();
    if shadow_enabled {
        let shadow_hmm_started = Instant::now();
        let shadow = low_frequency_haplotype_marginals_pair(
            input,
            haplotype_view,
            sample,
            &observations0,
            &mut panel0,
            &observations1,
            &mut panel1,
            &donor_segments0,
            &donor_segments1,
            sparse_plan,
            low_frequency_mask,
            low_frequency_allele_flags,
            stage1_marker_indices,
            stage1_cm_positions,
            full_cm_positions,
            config,
            &mut HMMWorkspace::default(),
            &mut HMMWorkspace::default(),
            Stage2HmmBackend::MismatchMatrix,
            true,
        )?;
        STAGE2_SHADOW_HMM_NANOS.fetch_add(
            shadow_hmm_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        compare_stage2_anchor_posteriors(
            input,
            sample,
            stage1_marker_indices,
            &production.anchor_posteriors,
            &shadow.anchor_posteriors,
        )?;
        compare_stage2_marginals(input, sample, &production.marginals, &shadow.marginals)?;
        let shadow_result_started = Instant::now();
        let shadow_result = stage2_sample_calls_from_marginals(
            input,
            snapshot,
            sample,
            low_frequency_mask,
            &shadow.marginals,
            seed,
        )?;
        compare_stage2_calls(input, sample, low_frequency_mask, &result, &shadow_result)?;
        STAGE2_SHADOW_RESULT_NANOS.fetch_add(
            shadow_result_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        STAGE2_SHADOW_MATRIX_BYTES
            .fetch_add(shadow.mismatch_matrix_bytes as u64, Ordering::Relaxed);
        STAGE2_SHADOW_ANCHOR_POSTERIORS
            .fetch_add(shadow.anchor_posteriors.len() as u64, Ordering::Relaxed);
        STAGE2_SHADOW_MARGINAL_CELLS.fetch_add(
            shadow
                .marginals
                .iter()
                .map(|marginal| marginal.values.len() as u64)
                .sum::<u64>(),
            Ordering::Relaxed,
        );
        STAGE2_SHADOW_GT_CELLS.fetch_add(
            low_frequency_mask
                .iter()
                .filter(|&&enabled| enabled)
                .count() as u64,
            Ordering::Relaxed,
        );
        STAGE2_SHADOW_RUNS.fetch_add(1, Ordering::Relaxed);
    }
    drop(panel0);
    drop(donor_segments0);
    drop(panel1);
    drop(donor_segments1);
    if let (Some(marker), Some(diagnostic0), Some(diagnostic1), Some(site)) = (
        diagnostic_marker_for_sample,
        diagnostic0,
        diagnostic1,
        diagnostic_site,
    ) {
        diagnostic_trace.push(ImpHmmTrace {
            chrom: input.chrom.clone(),
            position: input.rows[marker].meta.pos,
            sample: site.sample.clone(),
            window_index: input.index,
            round: phase_iteration,
            cluster_kind: None,
            transition_probability: diagnostic0.recursion.transition_probability,
            state_alleles: vec![diagnostic0.state_alleles, diagnostic1.state_alleles],
            emission_probabilities: vec![
                diagnostic0.recursion.emission_probabilities,
                diagnostic1.recursion.emission_probabilities,
            ],
            forward_probabilities: vec![
                diagnostic0.recursion.forward_probabilities,
                diagnostic1.recursion.forward_probabilities,
            ],
            backward_probabilities: vec![
                diagnostic0.recursion.backward_probabilities,
                diagnostic1.recursion.backward_probabilities,
            ],
            posterior_normalization: vec![
                diagnostic0.recursion.posterior_normalization as f64,
                diagnostic1.recursion.posterior_normalization as f64,
            ],
            phase_swap: None,
            phase_log_odds: None,
            phase_threshold: lr_threshold,
            selected_alleles: result[marker].alleles,
        });
    }
    Ok(SampleUpdate {
        calls: SampleCallUpdate::Full(DiploidCalls::from(result)),
        diagnostic_trace,
        unphased_heterozygote_count: 0,
        phase_swap_count: 0,
    })
}

#[cfg(test)]
#[allow(dead_code)]
fn low_frequency_haplotype_marginals(
    input: &PhaseWindow,
    haplotype_view: &composite::PhaseHaplotypeView<'_>,
    sample: usize,
    observations: &PackedHaplotypes,
    panel: &mut super::types::CompositePanel,
    donor_segments: &[Vec<composite::DonorHapSegment>],
    sparse_plan: &Stage2SparsePlan,
    low_frequency_mask: &[bool],
    low_frequency_allele_flags: &[Vec<bool>],
    stage1_marker_indices: &[usize],
    stage1_cm_positions: &[f64],
    full_cm_positions: &[f64],
    config: &HMMConfig,
    workspace: &mut HMMWorkspace,
) -> Result<LowFrequencyAlleleMarginals, String> {
    let marker_count = input.rows.len();
    let state_count = panel.haplotypes.n_haplotypes;
    if panel.haplotypes.n_markers != marker_count
        || donor_segments.len() != state_count
        || stage1_marker_indices.is_empty()
        || stage1_marker_indices.len() != stage1_cm_positions.len()
        || stage1_marker_indices
            .iter()
            .any(|&marker| marker >= marker_count)
        || stage1_marker_indices
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
        || full_cm_positions.len() != marker_count
        || stage1_cm_positions.iter().any(|cm| !cm.is_finite())
        || sparse_plan
            .anchor_markers
            .iter()
            .any(|marker| stage1_marker_indices.binary_search(marker).is_err())
        || sparse_plan
            .unresolved_markers
            .iter()
            .any(|&marker| marker >= marker_count || !low_frequency_mask[marker])
    {
        return Err("Stage2 marginal arrays have inconsistent dimensions".to_owned());
    }
    let mut offsets = Vec::with_capacity(marker_count + 1);
    offsets.push(0usize);
    for row in &input.rows {
        offsets.push(
            offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(row.meta.allele_count())
                .ok_or_else(|| "Stage2 allele marginal size overflow".to_owned())?,
        );
    }
    let mut marginals = vec![0.0f32; *offsets.last().unwrap()];
    let sparse_observations = select_observations(observations, stage1_marker_indices)?;
    panel.marker_indices = stage1_marker_indices.to_vec();
    let selected_hmm_markers = sparse_plan
        .anchor_markers
        .iter()
        .map(|marker| {
            stage1_marker_indices
                .binary_search(marker)
                .expect("validated Stage2 anchors belong to the Stage1 marker set")
        })
        .collect::<Vec<_>>();
    let mut donor_cursor = Stage2DonorCursor::new(donor_segments, state_count);
    // The HMM emits posterior vectors from right to left while unwinding its
    // backward recursion. Keep the right anchor and wait for the next (left)
    // anchor so each Stage2 target uses its adjacent Stage1 markers.
    let mut right_anchor: Option<(usize, usize, Vec<f32>)> = None;
    hmm::forward_backward_with_selected_state_callback(
        &sparse_observations,
        panel,
        stage1_cm_positions,
        config,
        workspace,
        &selected_hmm_markers,
        |hmm_marker, state_probabilities| {
            let anchor_index = stage2_anchor_index(&selected_hmm_markers, hmm_marker)
                .expect("HMM callback marker belongs to the selected Stage2 anchors");
            let left_marker = stage1_marker_indices[hmm_marker];
            let donor_haplotypes = donor_cursor.row_for_anchor(left_marker);
            if let Some((_, right_marker, right_probabilities)) = right_anchor.as_ref() {
                // The interval is left-closed/right-open: a Stage1 marker is
                // evaluated with its own state vector, and becomes the left
                // anchor of the following interpolation interval.
                let start = sparse_plan
                    .unresolved_markers
                    .partition_point(|&marker| marker < left_marker);
                let end = sparse_plan
                    .unresolved_markers
                    .partition_point(|&marker| marker < *right_marker);
                for &marker in &sparse_plan.unresolved_markers[start..end] {
                    let (left_weight, right_weight) = interpolation_weights(
                        marker,
                        left_marker,
                        *right_marker,
                        full_cm_positions,
                    );
                    accumulate_low_frequency_allele_row(
                        input,
                        Some(haplotype_view),
                        sample,
                        marker,
                        state_probabilities,
                        right_probabilities,
                        left_weight,
                        right_weight,
                        donor_haplotypes,
                        low_frequency_mask,
                        low_frequency_allele_flags,
                        false,
                        false,
                        &offsets,
                        &mut marginals,
                    );
                }
            } else {
                // The suffix beyond the final Stage1 marker uses that final
                // posterior without interpolation.
                let start = sparse_plan
                    .unresolved_markers
                    .partition_point(|&marker| marker < left_marker);
                for &marker in &sparse_plan.unresolved_markers[start..] {
                    accumulate_low_frequency_allele_row(
                        input,
                        Some(haplotype_view),
                        sample,
                        marker,
                        state_probabilities,
                        state_probabilities,
                        1.0,
                        0.0,
                        donor_haplotypes,
                        low_frequency_mask,
                        low_frequency_allele_flags,
                        false,
                        false,
                        &offsets,
                        &mut marginals,
                    );
                }
            }
            right_anchor = Some((anchor_index, left_marker, state_probabilities.to_vec()));
            Ok(())
        },
    )?;
    if let Some((_first_anchor_index, first_marker, first_probabilities)) = right_anchor {
        // Markers preceding the first Stage1 marker use the first posterior.
        let first_donor_row = donor_cursor.row_for_anchor(first_marker);
        let end = sparse_plan
            .unresolved_markers
            .partition_point(|&marker| marker < first_marker);
        for &marker in &sparse_plan.unresolved_markers[..end] {
            accumulate_low_frequency_allele_row(
                input,
                Some(haplotype_view),
                sample,
                marker,
                &first_probabilities,
                &first_probabilities,
                1.0,
                0.0,
                first_donor_row,
                low_frequency_mask,
                low_frequency_allele_flags,
                false,
                false,
                &offsets,
                &mut marginals,
            );
        }
    }
    // Beagle compares the unscaled f32 allele totals returned by
    // unscaledAlProbs() and imputeAllele(). Normalizing each marginal changes
    // tie and threshold behaviour in the final f32 bits.
    Ok(LowFrequencyAlleleMarginals {
        values: marginals,
        offsets,
    })
}

#[allow(clippy::too_many_arguments)]
fn low_frequency_haplotype_marginals_pair(
    input: &PhaseWindow,
    haplotype_view: &composite::PhaseHaplotypeView<'_>,
    sample: usize,
    observations0: &PackedHaplotypes,
    panel0: &mut super::types::CompositePanel,
    observations1: &PackedHaplotypes,
    panel1: &mut super::types::CompositePanel,
    donor_segments0: &[Vec<composite::DonorHapSegment>],
    donor_segments1: &[Vec<composite::DonorHapSegment>],
    sparse_plan: &Stage2SparsePlan,
    low_frequency_mask: &[bool],
    low_frequency_allele_flags: &[Vec<bool>],
    stage1_marker_indices: &[usize],
    stage1_cm_positions: &[f64],
    full_cm_positions: &[f64],
    config: &HMMConfig,
    workspace0: &mut HMMWorkspace,
    workspace1: &mut HMMWorkspace,
    backend: Stage2HmmBackend,
    capture_anchor_posteriors: bool,
) -> Result<Stage2MarginalRun, String> {
    let marker_count = input.rows.len();
    let state_count0 = panel0.haplotypes.n_haplotypes;
    let state_count1 = panel1.haplotypes.n_haplotypes;
    if panel0.haplotypes.n_markers != stage1_marker_indices.len()
        || panel1.haplotypes.n_markers != stage1_marker_indices.len()
        || donor_segments0.len() != state_count0
        || donor_segments1.len() != state_count1
        || stage1_marker_indices.is_empty()
        || stage1_marker_indices.len() != stage1_cm_positions.len()
        || stage1_marker_indices
            .iter()
            .any(|&marker| marker >= marker_count)
        || stage1_marker_indices
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
        || full_cm_positions.len() != marker_count
        || stage1_cm_positions.iter().any(|cm| !cm.is_finite())
        || sparse_plan
            .anchor_markers
            .iter()
            .any(|marker| stage1_marker_indices.binary_search(marker).is_err())
        || sparse_plan
            .unresolved_markers
            .iter()
            .any(|&marker| marker >= marker_count || !low_frequency_mask[marker])
    {
        return Err("Stage2 paired marginal arrays have inconsistent dimensions".to_owned());
    }
    let mut offsets = Vec::with_capacity(marker_count + 1);
    offsets.push(0usize);
    for row in &input.rows {
        offsets.push(
            offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(row.meta.allele_count())
                .ok_or_else(|| "Stage2 allele marginal size overflow".to_owned())?,
        );
    }
    let marginal_size = *offsets.last().unwrap();
    let mut marginals0 = vec![0.0f32; marginal_size];
    let mut marginals1 = vec![0.0f32; marginal_size];
    let sparse_observations0 = select_observations(observations0, stage1_marker_indices)?;
    let sparse_observations1 = select_observations(observations1, stage1_marker_indices)?;
    panel0.marker_indices = (0..stage1_marker_indices.len()).collect();
    panel1.marker_indices = (0..stage1_marker_indices.len()).collect();
    let selected_hmm_markers = sparse_plan
        .anchor_markers
        .iter()
        .map(|marker| {
            stage1_marker_indices
                .binary_search(marker)
                .expect("validated Stage2 anchors belong to the Stage1 marker set")
        })
        .collect::<Vec<_>>();
    let mut donor_cursor0 = Stage2DonorCursor::new(donor_segments0, state_count0);
    let mut donor_cursor1 = Stage2DonorCursor::new(donor_segments1, state_count1);
    // Callbacks arrive in decreasing marker order. One buffer per haplotype
    // holds the current right anchor and is overwritten bit-for-bit after the
    // interpolation interval has been accumulated.
    let mut right_probabilities0 = vec![0.0f32; state_count0];
    let mut right_probabilities1 = vec![0.0f32; state_count1];
    let mut right_anchor0: Option<(usize, usize)> = None;
    let mut right_anchor1: Option<(usize, usize)> = None;
    let reference_fast = haplotype_view.has_complete_state_major_reference();
    let callback_profile = stage2_callback_profiling_enabled();
    let mut callback_counts = CallbackDonorCounts::default();
    let mut anchor_posteriors = Vec::new();
    let mut callback = |hmm_marker: usize,
                        state_probabilities0: &[f32],
                        state_probabilities1: &[f32]|
     -> Result<(), String> {
        if capture_anchor_posteriors {
            anchor_posteriors.push(Stage2AnchorPosterior {
                hmm_marker,
                probabilities0: state_probabilities0.to_vec(),
                probabilities1: state_probabilities1.to_vec(),
            });
        }
        let anchor_profile_started = callback_profile.then(Instant::now);
        let anchor_index = stage2_anchor_index(&selected_hmm_markers, hmm_marker)
            .expect("HMM callback marker belongs to the selected Stage2 anchors");
        let left_marker = stage1_marker_indices[hmm_marker];
        let donor_row0 = donor_cursor0.row_for_anchor(left_marker);
        let donor_row1 = donor_cursor1.row_for_anchor(left_marker);
        if let Some(started) = anchor_profile_started {
            STAGE2_CALLBACK_ANCHOR_NANOS
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if let (Some((_, right_marker)), Some(_)) = (right_anchor0.as_ref(), right_anchor1.as_ref())
        {
            let interval_profile_started = callback_profile.then(Instant::now);
            let start = sparse_plan
                .unresolved_markers
                .partition_point(|&marker| marker < left_marker);
            let end = sparse_plan
                .unresolved_markers
                .partition_point(|&marker| marker < *right_marker);
            if let Some(started) = interval_profile_started {
                STAGE2_CALLBACK_INTERVAL_NANOS
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            for &marker in &sparse_plan.unresolved_markers[start..end] {
                let weight_profile_started = callback_profile.then(Instant::now);
                let (left_weight, right_weight) =
                    interpolation_weights(marker, left_marker, *right_marker, full_cm_positions);
                if let Some(started) = weight_profile_started {
                    STAGE2_CALLBACK_WEIGHT_NANOS
                        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                let row_profile_started = callback_profile.then(Instant::now);
                let first_counts = accumulate_low_frequency_allele_row(
                    input,
                    Some(haplotype_view),
                    sample,
                    marker,
                    state_probabilities0,
                    &right_probabilities0,
                    left_weight,
                    right_weight,
                    donor_row0,
                    low_frequency_mask,
                    low_frequency_allele_flags,
                    reference_fast,
                    callback_profile,
                    &offsets,
                    &mut marginals0,
                );
                let second_counts = accumulate_low_frequency_allele_row(
                    input,
                    Some(haplotype_view),
                    sample,
                    marker,
                    state_probabilities1,
                    &right_probabilities1,
                    left_weight,
                    right_weight,
                    donor_row1,
                    low_frequency_mask,
                    low_frequency_allele_flags,
                    reference_fast,
                    callback_profile,
                    &offsets,
                    &mut marginals1,
                );
                if callback_profile {
                    let mut row_counts = first_counts;
                    row_counts.add_assign(second_counts);
                    callback_counts.add_assign(row_counts);
                }
                if let Some(started) = row_profile_started {
                    STAGE2_CALLBACK_ROW_NANOS
                        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    STAGE2_CALLBACK_ROWS.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            let interval_profile_started = callback_profile.then(Instant::now);
            let start = sparse_plan
                .unresolved_markers
                .partition_point(|&marker| marker < left_marker);
            if let Some(started) = interval_profile_started {
                STAGE2_CALLBACK_INTERVAL_NANOS
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            for &marker in &sparse_plan.unresolved_markers[start..] {
                let row_profile_started = callback_profile.then(Instant::now);
                let first_counts = accumulate_low_frequency_allele_row(
                    input,
                    Some(haplotype_view),
                    sample,
                    marker,
                    state_probabilities0,
                    state_probabilities0,
                    1.0,
                    0.0,
                    donor_row0,
                    low_frequency_mask,
                    low_frequency_allele_flags,
                    reference_fast,
                    callback_profile,
                    &offsets,
                    &mut marginals0,
                );
                let second_counts = accumulate_low_frequency_allele_row(
                    input,
                    Some(haplotype_view),
                    sample,
                    marker,
                    state_probabilities1,
                    state_probabilities1,
                    1.0,
                    0.0,
                    donor_row1,
                    low_frequency_mask,
                    low_frequency_allele_flags,
                    reference_fast,
                    callback_profile,
                    &offsets,
                    &mut marginals1,
                );
                if callback_profile {
                    let mut row_counts = first_counts;
                    row_counts.add_assign(second_counts);
                    callback_counts.add_assign(row_counts);
                }
                if let Some(started) = row_profile_started {
                    STAGE2_CALLBACK_ROW_NANOS
                        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    STAGE2_CALLBACK_ROWS.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        debug_assert_eq!(state_probabilities0.len(), right_probabilities0.len());
        debug_assert_eq!(state_probabilities1.len(), right_probabilities1.len());
        right_probabilities0.copy_from_slice(state_probabilities0);
        right_probabilities1.copy_from_slice(state_probabilities1);
        right_anchor0 = Some((anchor_index, left_marker));
        right_anchor1 = Some((anchor_index, left_marker));
        if callback_profile {
            STAGE2_CALLBACK_TARGET_DONOR_STATES
                .fetch_add(callback_counts.target_donor_states, Ordering::Relaxed);
            STAGE2_CALLBACK_REFERENCE_FAST_STATES
                .fetch_add(callback_counts.reference_fast_states, Ordering::Relaxed);
            STAGE2_CALLBACK_REFERENCE_FALLBACK_STATES
                .fetch_add(callback_counts.reference_fallback_states, Ordering::Relaxed);
            STAGE2_CALLBACK_MISSING_DONOR_STATES
                .fetch_add(callback_counts.missing_donor_states, Ordering::Relaxed);
            STAGE2_CALLBACK_TARGET_HET_ROWS
                .fetch_add(callback_counts.target_heterozygous_rows, Ordering::Relaxed);
            STAGE2_CALLBACK_TARGET_PARTIAL_OR_MISSING_ROWS.fetch_add(
                callback_counts.target_partial_or_missing_rows,
                Ordering::Relaxed,
            );
            STAGE2_CALLBACK_PROBABILITY_CALCULATIONS
                .fetch_add(callback_counts.probability_calculations, Ordering::Relaxed);
            STAGE2_CALLBACK_DONOR_SOURCE_READS
                .fetch_add(callback_counts.donor_source_reads, Ordering::Relaxed);
            STAGE2_CALLBACK_RARE_FLAG_QUERIES
                .fetch_add(callback_counts.rare_flag_queries, Ordering::Relaxed);
            STAGE2_CALLBACK_MARGINAL_ADDITIONS
                .fetch_add(callback_counts.marginal_additions, Ordering::Relaxed);
            callback_counts = CallbackDonorCounts::default();
        }
        Ok(())
    };
    let mismatch_matrix_bytes = match backend {
        Stage2HmmBackend::Production => {
            hmm::forward_backward_with_selected_state_callbacks_pair(
                &sparse_observations0,
                panel0,
                &sparse_observations1,
                panel1,
                stage1_cm_positions,
                config,
                workspace0,
                workspace1,
                &selected_hmm_markers,
                &mut callback,
            )?;
            0
        }
        Stage2HmmBackend::MismatchMatrix => {
            hmm::forward_backward_with_selected_state_callbacks_pair_mismatch(
                &sparse_observations0,
                panel0,
                &sparse_observations1,
                panel1,
                stage1_cm_positions,
                config,
                workspace0,
                workspace1,
                &selected_hmm_markers,
                &mut callback,
            )?
        }
    };
    if let (
        Some((_first_anchor_index0, first_marker0)),
        Some((_first_anchor_index1, first_marker1)),
    ) = (right_anchor0, right_anchor1)
    {
        if first_marker0 != first_marker1 {
            return Err("paired Stage2 first-anchor markers differ".to_owned());
        }
        let donor_row0 = donor_cursor0.row_for_anchor(first_marker0);
        let donor_row1 = donor_cursor1.row_for_anchor(first_marker1);
        let interval_profile_started = callback_profile.then(Instant::now);
        let end = sparse_plan
            .unresolved_markers
            .partition_point(|&marker| marker < first_marker0);
        if let Some(started) = interval_profile_started {
            STAGE2_CALLBACK_INTERVAL_NANOS
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let mut suffix_counts = CallbackDonorCounts::default();
        for &marker in &sparse_plan.unresolved_markers[..end] {
            let row_profile_started = callback_profile.then(Instant::now);
            let first_counts = accumulate_low_frequency_allele_row(
                input,
                Some(haplotype_view),
                sample,
                marker,
                &right_probabilities0,
                &right_probabilities0,
                1.0,
                0.0,
                donor_row0,
                low_frequency_mask,
                low_frequency_allele_flags,
                reference_fast,
                callback_profile,
                &offsets,
                &mut marginals0,
            );
            let second_counts = accumulate_low_frequency_allele_row(
                input,
                Some(haplotype_view),
                sample,
                marker,
                &right_probabilities1,
                &right_probabilities1,
                1.0,
                0.0,
                donor_row1,
                low_frequency_mask,
                low_frequency_allele_flags,
                reference_fast,
                callback_profile,
                &offsets,
                &mut marginals1,
            );
            if callback_profile {
                let mut row_counts = first_counts;
                row_counts.add_assign(second_counts);
                suffix_counts.add_assign(row_counts);
            }
            if let Some(started) = row_profile_started {
                STAGE2_CALLBACK_ROW_NANOS
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                STAGE2_CALLBACK_ROWS.fetch_add(1, Ordering::Relaxed);
            }
        }
        if callback_profile {
            STAGE2_CALLBACK_TARGET_DONOR_STATES
                .fetch_add(suffix_counts.target_donor_states, Ordering::Relaxed);
            STAGE2_CALLBACK_REFERENCE_FAST_STATES
                .fetch_add(suffix_counts.reference_fast_states, Ordering::Relaxed);
            STAGE2_CALLBACK_REFERENCE_FALLBACK_STATES
                .fetch_add(suffix_counts.reference_fallback_states, Ordering::Relaxed);
            STAGE2_CALLBACK_MISSING_DONOR_STATES
                .fetch_add(suffix_counts.missing_donor_states, Ordering::Relaxed);
            STAGE2_CALLBACK_TARGET_HET_ROWS
                .fetch_add(suffix_counts.target_heterozygous_rows, Ordering::Relaxed);
            STAGE2_CALLBACK_TARGET_PARTIAL_OR_MISSING_ROWS.fetch_add(
                suffix_counts.target_partial_or_missing_rows,
                Ordering::Relaxed,
            );
            STAGE2_CALLBACK_PROBABILITY_CALCULATIONS
                .fetch_add(suffix_counts.probability_calculations, Ordering::Relaxed);
            STAGE2_CALLBACK_DONOR_SOURCE_READS
                .fetch_add(suffix_counts.donor_source_reads, Ordering::Relaxed);
            STAGE2_CALLBACK_RARE_FLAG_QUERIES
                .fetch_add(suffix_counts.rare_flag_queries, Ordering::Relaxed);
            STAGE2_CALLBACK_MARGINAL_ADDITIONS
                .fetch_add(suffix_counts.marginal_additions, Ordering::Relaxed);
        }
    }
    Ok(Stage2MarginalRun {
        marginals: [
            LowFrequencyAlleleMarginals {
                values: marginals0,
                offsets: offsets.clone(),
            },
            LowFrequencyAlleleMarginals {
                values: marginals1,
                offsets,
            },
        ],
        anchor_posteriors,
        mismatch_matrix_bytes,
    })
}

fn stage2_sample_calls_from_marginals(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    low_frequency_mask: &[bool],
    marginals: &[LowFrequencyAlleleMarginals; 2],
    seed: u64,
) -> Result<Vec<DiploidCall>, String> {
    if low_frequency_mask.len() != input.rows.len()
        || snapshot.rows.len() != input.rows.len()
        || marginals[0].offsets.len() != input.rows.len().saturating_add(1)
        || marginals[1].offsets.len() != input.rows.len().saturating_add(1)
    {
        return Err("Stage2 sample-call dimensions are inconsistent".to_owned());
    }
    let mut random = pbwt::JavaRandom::new(seed.wrapping_add(sample as u64));
    let mut result = Vec::with_capacity(input.rows.len());
    for marker in 0..input.rows.len() {
        if !low_frequency_mask[marker] {
            result.push(
                snapshot.rows[marker]
                    .calls
                    .get(sample)
                    .ok_or_else(|| "Stage2 snapshot sample is out of range".to_owned())?,
            );
            continue;
        }
        let call = input.rows[marker]
            .calls
            .get(sample)
            .ok_or_else(|| "Stage2 input sample is out of range".to_owned())?;
        let alleles = match call.alleles {
            [Some(a), Some(b)] if a == b => [Some(a), Some(a)],
            [Some(a), Some(b)] => {
                let p0 = marginals[0].at(marker);
                let p1 = marginals[1].at(marker);
                let forward = probability(p0, a) * probability(p1, b);
                let reverse = probability(p0, b) * probability(p1, a);
                let oriented = if forward < reverse || (forward == reverse && random.next_boolean())
                {
                    b
                } else {
                    a
                };
                [Some(oriented), Some(if oriented == a { b } else { a })]
            }
            [_, _] => {
                let p0 = marginals[0].at(marker);
                let p1 = marginals[1].at(marker);
                [Some(beagle_allele_map(p0)), Some(beagle_allele_map(p1))]
            }
        };
        result.push(DiploidCall {
            alleles,
            phased: true,
        });
    }
    Ok(result)
}

fn compare_stage2_anchor_posteriors(
    input: &PhaseWindow,
    sample: usize,
    stage1_marker_indices: &[usize],
    production: &[Stage2AnchorPosterior],
    mismatch: &[Stage2AnchorPosterior],
) -> Result<(), String> {
    if production.len() != mismatch.len() {
        return Err(format!(
            "Stage2 mismatch shadow anchor-count mismatch window={}:{} sample={} production={} mismatch={}",
            input.chrom,
            input.index,
            sample,
            production.len(),
            mismatch.len()
        ));
    }
    for (anchor_index, (production, mismatch)) in production.iter().zip(mismatch).enumerate() {
        if production.hmm_marker != mismatch.hmm_marker {
            return Err(format!(
                "Stage2 mismatch shadow anchor-order mismatch window={}:{} sample={} anchor={} production_hmm_marker={} mismatch_hmm_marker={}",
                input.chrom,
                input.index,
                sample,
                anchor_index,
                production.hmm_marker,
                mismatch.hmm_marker
            ));
        }
        let stage1_marker = stage1_marker_indices
            .get(production.hmm_marker)
            .copied()
            .unwrap_or(usize::MAX);
        for (track, (production, mismatch)) in [
            (&production.probabilities0, &mismatch.probabilities0),
            (&production.probabilities1, &mismatch.probabilities1),
        ]
        .into_iter()
        .enumerate()
        {
            if production.len() != mismatch.len() {
                return Err(format!(
                    "Stage2 mismatch shadow posterior-length mismatch window={}:{} sample={} stage1_marker={} track={} production={} mismatch={}",
                    input.chrom,
                    input.index,
                    sample,
                    stage1_marker,
                    track,
                    production.len(),
                    mismatch.len()
                ));
            }
            for (state, (&production, &mismatch)) in
                production.iter().zip(mismatch.iter()).enumerate()
            {
                if production.to_bits() != mismatch.to_bits() {
                    return Err(format!(
                        "Stage2 mismatch shadow anchor-posterior mismatch window={}:{} sample={} stage1_marker={} track={track} state={state} production={} mismatch={} production_bits={:#010x} mismatch_bits={:#010x}",
                        input.chrom,
                        input.index,
                        sample,
                        stage1_marker,
                        production,
                        mismatch,
                        production.to_bits(),
                        mismatch.to_bits()
                    ));
                }
            }
        }
    }
    Ok(())
}

fn compare_stage2_marginals(
    input: &PhaseWindow,
    sample: usize,
    production: &[LowFrequencyAlleleMarginals; 2],
    mismatch: &[LowFrequencyAlleleMarginals; 2],
) -> Result<(), String> {
    for (track, (production, mismatch)) in production.iter().zip(mismatch).enumerate() {
        if production.offsets != mismatch.offsets {
            return Err(format!(
                "Stage2 mismatch shadow marginal-offset mismatch window={}:{} sample={sample} track={track} production_len={} mismatch_len={}",
                input.chrom,
                input.index,
                production.offsets.len(),
                mismatch.offsets.len()
            ));
        }
        if production.values.len() != mismatch.values.len() {
            return Err(format!(
                "Stage2 mismatch shadow marginal-length mismatch window={}:{} sample={sample} track={track} production={} mismatch={}",
                input.chrom,
                input.index,
                production.values.len(),
                mismatch.values.len()
            ));
        }
        for (marker, (&start, &end)) in production
            .offsets
            .iter()
            .zip(production.offsets.iter().skip(1))
            .enumerate()
        {
            for allele in start..end {
                let production_value = production.values[allele];
                let mismatch_value = mismatch.values[allele];
                if production_value.to_bits() != mismatch_value.to_bits() {
                    return Err(format!(
                        "Stage2 mismatch shadow marginal mismatch window={}:{} sample={sample} marker={marker} track={track} allele={} production={production_value} mismatch={mismatch_value} production_bits={:#010x} mismatch_bits={:#010x}",
                        input.chrom,
                        input.index,
                        allele - start,
                        production_value.to_bits(),
                        mismatch_value.to_bits()
                    ));
                }
            }
        }
    }
    Ok(())
}

fn compare_stage2_calls(
    input: &PhaseWindow,
    sample: usize,
    low_frequency_mask: &[bool],
    production: &[DiploidCall],
    mismatch: &[DiploidCall],
) -> Result<(), String> {
    if production.len() != mismatch.len() || production.len() != input.rows.len() {
        return Err(format!(
            "Stage2 mismatch shadow GT-length mismatch window={}:{} sample={sample} production={} mismatch={} markers={}",
            input.chrom,
            input.index,
            production.len(),
            mismatch.len(),
            input.rows.len()
        ));
    }
    for (marker, (&production, &mismatch)) in production.iter().zip(mismatch.iter()).enumerate() {
        if production != mismatch {
            return Err(format!(
                "Stage2 mismatch shadow GT mismatch window={}:{} sample={sample} marker={marker} low_frequency={} production={production:?} mismatch={mismatch:?}",
                input.chrom,
                input.index,
                low_frequency_mask.get(marker).copied().unwrap_or(false)
            ));
        }
    }
    Ok(())
}

fn stage2_anchor_index(selected_hmm_markers: &[usize], hmm_marker: usize) -> Option<usize> {
    selected_hmm_markers.binary_search(&hmm_marker).ok()
}

fn interpolation_weights(
    marker: usize,
    left_marker: usize,
    right_marker: usize,
    cm_positions: &[f64],
) -> (f32, f32) {
    if right_marker <= left_marker {
        return (1.0, 0.0);
    }
    let span = cm_positions[right_marker] - cm_positions[left_marker];
    if span <= f64::EPSILON {
        return (1.0, 0.0);
    }
    let left = ((cm_positions[right_marker] - cm_positions[marker]) / span) as f32;
    (left, 1.0 - left)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn accumulate_low_frequency_alleles(
    input: &PhaseWindow,
    haplotype_view: Option<&composite::PhaseHaplotypeView<'_>>,
    target_sample: usize,
    marker: usize,
    donor_state_marker: usize,
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    left_weight: f32,
    right_weight: f32,
    panel: &super::types::CompositePanel,
    donor_segments: &[Vec<composite::DonorHapSegment>],
    low_frequency_mask: &[bool],
    low_frequency_allele_flags: &[Vec<bool>],
    offsets: &[usize],
    marginal_values: &mut [f32],
) {
    let donor_haplotypes = donor_haplotypes_at_anchors(
        donor_segments,
        &[donor_state_marker],
        panel.haplotypes.n_haplotypes,
    );
    accumulate_low_frequency_allele_row(
        input,
        haplotype_view,
        target_sample,
        marker,
        left_probabilities,
        right_probabilities,
        left_weight,
        right_weight,
        &donor_haplotypes,
        low_frequency_mask,
        low_frequency_allele_flags,
        false,
        false,
        offsets,
        marginal_values,
    );
}

struct Stage2DonorCursor<'a> {
    donor_segments: &'a [Vec<composite::DonorHapSegment>],
    segment_cursors: Vec<usize>,
    row: Vec<usize>,
    last_anchor: Option<usize>,
}

impl<'a> Stage2DonorCursor<'a> {
    fn new(donor_segments: &'a [Vec<composite::DonorHapSegment>], state_count: usize) -> Self {
        debug_assert_eq!(donor_segments.len(), state_count);
        let segment_cursors = donor_segments.iter().map(Vec::len).collect();
        Self {
            donor_segments,
            segment_cursors,
            row: vec![usize::MAX; state_count],
            last_anchor: None,
        }
    }

    fn row_for_anchor(&mut self, anchor: usize) -> &[usize] {
        if self.last_anchor == Some(anchor) {
            return &self.row;
        }
        if self.last_anchor.is_some_and(|prev| anchor > prev) {
            for (cursor, segments) in self.segment_cursors.iter_mut().zip(self.donor_segments) {
                *cursor = segments.len();
            }
        }
        for ((cursor, segments), slot) in self
            .segment_cursors
            .iter_mut()
            .zip(self.donor_segments)
            .zip(self.row.iter_mut())
        {
            while *cursor > 0 && segments[*cursor - 1].start_marker > anchor {
                *cursor -= 1;
            }
            *slot = if *cursor > 0 {
                let segment = &segments[*cursor - 1];
                if anchor < segment.end_marker {
                    segment.donor_haplotype
                } else {
                    usize::MAX
                }
            } else {
                usize::MAX
            };
        }
        self.last_anchor = Some(anchor);
        &self.row
    }
}

#[cfg(test)]
fn donor_haplotypes_at_anchors(
    donor_segments: &[Vec<composite::DonorHapSegment>],
    anchors: &[usize],
    state_count: usize,
) -> Vec<usize> {
    const MISSING_DONOR: usize = usize::MAX;
    debug_assert_eq!(donor_segments.len(), state_count);
    let mut donors = vec![MISSING_DONOR; anchors.len().saturating_mul(state_count)];
    for (state, segments) in donor_segments.iter().enumerate() {
        let mut segment_index = 0usize;
        for (anchor_index, &anchor) in anchors.iter().enumerate() {
            while segments
                .get(segment_index)
                .is_some_and(|segment| segment.end_marker <= anchor)
            {
                segment_index += 1;
            }
            if let Some(segment) = segments.get(segment_index) {
                if segment.start_marker <= anchor {
                    donors[anchor_index * state_count + state] = segment.donor_haplotype;
                }
            }
        }
    }
    donors
}

#[allow(clippy::too_many_arguments)]
fn accumulate_low_frequency_allele_row(
    input: &PhaseWindow,
    haplotype_view: Option<&composite::PhaseHaplotypeView<'_>>,
    target_sample: usize,
    marker: usize,
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    left_weight: f32,
    right_weight: f32,
    donor_haplotypes: &[usize],
    low_frequency_mask: &[bool],
    low_frequency_allele_flags: &[Vec<bool>],
    reference_fast: bool,
    profile_counts: bool,
    offsets: &[usize],
    marginal_values: &mut [f32],
) -> CallbackDonorCounts {
    if profile_counts {
        accumulate_low_frequency_allele_row_impl::<true>(
            input,
            haplotype_view,
            target_sample,
            marker,
            left_probabilities,
            right_probabilities,
            left_weight,
            right_weight,
            donor_haplotypes,
            low_frequency_mask,
            low_frequency_allele_flags,
            reference_fast,
            offsets,
            marginal_values,
        )
    } else {
        accumulate_low_frequency_allele_row_impl::<false>(
            input,
            haplotype_view,
            target_sample,
            marker,
            left_probabilities,
            right_probabilities,
            left_weight,
            right_weight,
            donor_haplotypes,
            low_frequency_mask,
            low_frequency_allele_flags,
            reference_fast,
            offsets,
            marginal_values,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_low_frequency_allele_row_impl<const PROFILE_COUNTS: bool>(
    input: &PhaseWindow,
    haplotype_view: Option<&composite::PhaseHaplotypeView<'_>>,
    target_sample: usize,
    marker: usize,
    left_probabilities: &[f32],
    right_probabilities: &[f32],
    left_weight: f32,
    right_weight: f32,
    donor_haplotypes: &[usize],
    low_frequency_mask: &[bool],
    low_frequency_allele_flags: &[Vec<bool>],
    reference_fast: bool,
    offsets: &[usize],
    marginal_values: &mut [f32],
) -> CallbackDonorCounts {
    let mut counts = CallbackDonorCounts::default();
    if !low_frequency_mask[marker] {
        return counts;
    }
    let target_call = input.rows[marker].calls.get(target_sample).unwrap();
    if matches!(target_call.alleles, [Some(a), Some(b)] if a == b) {
        return counts;
    }
    if PROFILE_COUNTS {
        match target_call.alleles {
            [Some(first), Some(second)] if first != second => {
                counts.target_heterozygous_rows = 1;
            }
            _ => {
                counts.target_partial_or_missing_rows = 1;
            }
        }
    }
    let probabilities = &mut marginal_values[offsets[marker]..offsets[marker + 1]];
    if probabilities.is_empty() {
        return counts;
    }
    let state_count = donor_haplotypes.len();
    let rare_flags = &low_frequency_allele_flags[marker];
    if probabilities.len() == 2 {
        let target_alleles = target_call.alleles;
        let target_first_rare = target_alleles[0].is_some_and(|allele| {
            rare_flags
                .get(usize::from(allele))
                .copied()
                .unwrap_or(false)
        });
        let target_second_rare = target_alleles[1].is_some_and(|allele| {
            rare_flags
                .get(usize::from(allele))
                .copied()
                .unwrap_or(false)
        });
        if PROFILE_COUNTS {
            counts.rare_flag_queries += target_alleles.iter().flatten().count() as u64;
        }
        for state in 0..state_count {
            if PROFILE_COUNTS {
                counts.probability_calculations += 1;
            }
            let probability =
                left_probabilities[state] * left_weight + right_probabilities[state] * right_weight;
            if probability <= 0.0 {
                continue;
            }
            let donor_haplotype = donor_haplotypes[state];
            if donor_haplotype == usize::MAX {
                if PROFILE_COUNTS {
                    counts.missing_donor_states += 1;
                }
                continue;
            }
            if PROFILE_COUNTS {
                counts.donor_source_reads += 1;
            }
            let donor_sample = donor_haplotype / 2;
            let donor_is_reference = donor_sample >= input.rows[marker].calls.len();
            let donor_alleles = if let Some(call) = input.rows[marker].calls.get(donor_sample) {
                if PROFILE_COUNTS {
                    counts.target_donor_states += 1;
                }
                call.alleles
            } else if let Some(view) = haplotype_view.filter(|view| donor_sample < view.n_samples())
            {
                if PROFILE_COUNTS {
                    if reference_fast {
                        counts.reference_fast_states += 1;
                    } else {
                        counts.reference_fallback_states += 1;
                    }
                }
                let bit = donor_haplotype & 1;
                if reference_fast {
                    let [first, second] = unsafe {
                        view.complete_reference_biallelic_pair_unchecked(marker, donor_sample, bit)
                    };
                    [Some(first), Some(second)]
                } else {
                    [
                        view.allele(marker, donor_sample, bit),
                        view.allele(marker, donor_sample, bit ^ 1),
                    ]
                }
            } else {
                if PROFILE_COUNTS {
                    counts.missing_donor_states += 1;
                }
                continue;
            };
            let [Some(donor_first), Some(donor_second)] = donor_alleles else {
                continue;
            };
            if PROFILE_COUNTS {
                counts.rare_flag_queries += 2;
            }
            let first_is_rare = rare_flags
                .get(usize::from(donor_first))
                .copied()
                .unwrap_or(false);
            let second_is_rare = rare_flags
                .get(usize::from(donor_second))
                .copied()
                .unwrap_or(false);
            let mut add = |allele: u16, contribution: f32| {
                if let Some(value) = probabilities.get_mut(usize::from(allele)) {
                    if PROFILE_COUNTS {
                        counts.marginal_additions += 1;
                    }
                    *value += contribution;
                }
            };
            if let [Some(target_first), Some(target_second)] = target_alleles {
                if target_first == target_second || donor_first == donor_second {
                    add(donor_first, probability);
                } else {
                    let donor_has_first =
                        donor_first == target_first || donor_second == target_first;
                    let donor_has_second =
                        donor_first == target_second || donor_second == target_second;
                    let match_first = target_first_rare && donor_has_first;
                    let match_second = target_second_rare && donor_has_second;
                    if match_first ^ match_second {
                        add(
                            if match_first {
                                target_first
                            } else {
                                target_second
                            },
                            probability,
                        );
                    }
                }
            } else if donor_first == donor_second || donor_is_reference {
                add(donor_first, probability);
            } else if first_is_rare ^ second_is_rare {
                add(
                    if first_is_rare {
                        donor_first
                    } else {
                        donor_second
                    },
                    probability * 0.55,
                );
                add(
                    if first_is_rare {
                        donor_second
                    } else {
                        donor_first
                    },
                    probability * 0.45,
                );
            } else {
                add(donor_first, probability * 0.5);
                add(donor_second, probability * 0.5);
            }
        }
        return counts;
    }
    for state in 0..state_count {
        if PROFILE_COUNTS {
            counts.probability_calculations += 1;
        }
        let probability =
            left_probabilities[state] * left_weight + right_probabilities[state] * right_weight;
        if probability <= 0.0 {
            continue;
        }
        let donor_haplotype = donor_haplotypes[state];
        if donor_haplotype == usize::MAX {
            if PROFILE_COUNTS {
                counts.missing_donor_states += 1;
            }
            continue;
        }
        if PROFILE_COUNTS {
            counts.donor_source_reads += 1;
        }
        // Beagle reads donor alleles at the target marker from the original
        // unphased GT. The Stage1 snapshot must not leak an imputed allele
        // into Stage2 when a donor genotype was masked.
        let donor_sample = donor_haplotype / 2;
        let donor_is_reference = donor_sample >= input.rows[marker].calls.len();
        let donor_alleles = if let Some(call) = input.rows[marker].calls.get(donor_sample) {
            if PROFILE_COUNTS {
                counts.target_donor_states += 1;
            }
            call.alleles
        } else if let Some(view) = haplotype_view.filter(|view| donor_sample < view.n_samples()) {
            // Reference donors use the selected phased haplotype. Target
            // donors above retain the original GT/missingness semantics.
            if PROFILE_COUNTS {
                if reference_fast {
                    counts.reference_fast_states += 1;
                } else {
                    counts.reference_fallback_states += 1;
                }
            }
            let bit = donor_haplotype & 1;
            if reference_fast {
                let [first, second] = unsafe {
                    view.complete_reference_biallelic_pair_unchecked(marker, donor_sample, bit)
                };
                [Some(first), Some(second)]
            } else {
                [
                    view.allele(marker, donor_sample, bit),
                    view.allele(marker, donor_sample, bit ^ 1),
                ]
            }
        } else {
            if PROFILE_COUNTS {
                counts.missing_donor_states += 1;
            }
            continue;
        };
        let [Some(donor_first), Some(donor_second)] = donor_alleles else {
            continue;
        };
        if PROFILE_COUNTS {
            counts.rare_flag_queries += 2;
        }
        let first_is_rare = rare_flags
            .get(usize::from(donor_first))
            .copied()
            .unwrap_or(false);
        let second_is_rare = rare_flags
            .get(usize::from(donor_second))
            .copied()
            .unwrap_or(false);
        let mut add = |probabilities: &mut [f32], allele: u16, contribution: f32| {
            if let Some(value) = probabilities.get_mut(usize::from(allele)) {
                if PROFILE_COUNTS {
                    counts.marginal_additions += 1;
                }
                *value += contribution;
            }
        };
        if let [Some(target_first), Some(target_second)] = target_call.alleles {
            if target_first == target_second {
                add(probabilities, donor_first, probability);
            } else if donor_first == donor_second {
                add(probabilities, donor_first, probability);
            } else {
                if PROFILE_COUNTS {
                    counts.rare_flag_queries += 2;
                }
                let target_first_is_rare = rare_flags
                    .get(usize::from(target_first))
                    .copied()
                    .unwrap_or(false);
                let target_second_is_rare = rare_flags
                    .get(usize::from(target_second))
                    .copied()
                    .unwrap_or(false);
                let donor_has_first = donor_first == target_first || donor_second == target_first;
                let donor_has_second =
                    donor_first == target_second || donor_second == target_second;
                let match_first = target_first_is_rare && donor_has_first;
                let match_second = target_second_is_rare && donor_has_second;
                if match_first ^ match_second {
                    add(
                        probabilities,
                        if match_first {
                            target_first
                        } else {
                            target_second
                        },
                        probability,
                    );
                }
            }
        } else if donor_first == donor_second || donor_is_reference {
            add(probabilities, donor_first, probability);
        } else {
            if first_is_rare ^ second_is_rare {
                add(
                    probabilities,
                    if first_is_rare {
                        donor_first
                    } else {
                        donor_second
                    },
                    probability * 0.55,
                );
                add(
                    probabilities,
                    if first_is_rare {
                        donor_second
                    } else {
                        donor_first
                    },
                    probability * 0.45,
                );
            } else {
                add(probabilities, donor_first, probability * 0.5);
                add(probabilities, donor_second, probability * 0.5);
            }
        }
    }
    counts
}

fn beagle_allele_map(probabilities: &[f32]) -> u16 {
    let mut best_allele = 0usize;
    let mut best_probability = f32::NEG_INFINITY;
    for (allele, &probability) in probabilities.iter().enumerate() {
        if probability > best_probability {
            best_allele = allele;
            best_probability = probability;
        }
    }
    best_allele as u16
}

#[cfg(test)]
fn low_frequency_allele_flags(window: &PhaseWindow, max_carriers: usize) -> Vec<Vec<bool>> {
    low_frequency_allele_flags_rows(&window.rows, max_carriers)
}

#[cfg(test)]
fn low_frequency_allele_flags_rows(
    rows: &[super::types::VariantRow],
    max_carriers: usize,
) -> Vec<Vec<bool>> {
    rows.iter()
        .map(|row| {
            let mut counts = vec![0usize; row.meta.allele_count()];
            for call in row.calls.iter() {
                let [first, second] = call.alleles;
                if let Some(allele) = first {
                    counts[usize::from(allele)] = counts[usize::from(allele)]
                        .saturating_add(1)
                        .min(max_carriers + 1);
                }
                if let Some(allele) = second {
                    if Some(allele) != first {
                        counts[usize::from(allele)] = counts[usize::from(allele)]
                            .saturating_add(1)
                            .min(max_carriers + 1);
                    }
                }
            }
            counts
                .into_iter()
                .map(|count| count <= max_carriers)
                .collect()
        })
        .collect()
}

#[derive(Debug)]
struct HMMMarkerDiagnostic {
    recursion: hmm::HMMRecursionDiagnostic,
    state_alleles: Vec<Option<u16>>,
}

fn trace_hmm_marker(
    observations: &PackedHaplotypes,
    panel: &CompositePanel,
    full_cm_positions: &[f64],
    full_marker: usize,
    config: &HMMConfig,
) -> Result<Option<HMMMarkerDiagnostic>, String> {
    let Some(hmm_marker) = panel
        .marker_indices
        .iter()
        .position(|&marker| marker == full_marker)
    else {
        return Ok(None);
    };
    if full_cm_positions.len() != observations.n_markers
        || panel.marker_indices.len() > observations.n_markers
        || panel
            .marker_indices
            .iter()
            .any(|&marker| marker >= full_cm_positions.len())
    {
        return Err("diagnostic HMM marker mapping dimensions differ".to_owned());
    }
    let mapped = panel.marker_indices.len() != observations.n_markers
        || panel
            .marker_indices
            .iter()
            .enumerate()
            .any(|(marker, &mapped_marker)| marker != mapped_marker);
    let selected_observations;
    let hmm_observations = if mapped {
        selected_observations = select_observations(observations, &panel.marker_indices)?;
        &selected_observations
    } else {
        observations
    };
    let hmm_cm_positions = panel
        .marker_indices
        .iter()
        .map(|&marker| full_cm_positions[marker])
        .collect::<Vec<_>>();
    let mut workspace = HMMWorkspace::default();
    let mut recursion = None;
    hmm::forward_backward_with_diagnostics(
        hmm_observations,
        panel,
        &hmm_cm_positions,
        config,
        &mut workspace,
        &[hmm_marker],
        |diagnostic| {
            recursion = Some(diagnostic);
            Ok(())
        },
    )?;
    let state_alleles = (0..panel.haplotypes.n_haplotypes)
        .map(|state| {
            panel
                .haplotypes
                .allele(panel.marker_indices[hmm_marker], state)
        })
        .collect();
    Ok(recursion.map(|recursion| HMMMarkerDiagnostic {
        recursion,
        state_alleles,
    }))
}

fn phase_forward_backward(
    observations: &PackedHaplotypes,
    panel: &mut super::types::CompositePanel,
    cm_positions: &[f64],
    sparse_hmm_markers: Option<&[usize]>,
    sparse_hmm_cm_positions: Option<&[f64]>,
    allele_counts: Option<&[usize]>,
    config: &HMMConfig,
    workspace: &mut super::types::HMMWorkspace,
) -> Result<PosteriorWindow, String> {
    let Some(marker_indices) = sparse_hmm_markers.filter(|markers| !markers.is_empty()) else {
        return hmm::forward_backward(observations, panel, cm_positions, config, workspace);
    };
    let marker_cm_positions =
        sparse_hmm_cm_positions.ok_or_else(|| "sparse HMM marker map is missing".to_owned())?;
    let allele_counts =
        allele_counts.ok_or_else(|| "sparse HMM allele counts are missing".to_owned())?;
    if marker_indices.len() != marker_cm_positions.len()
        || allele_counts.len() != cm_positions.len()
    {
        return Err("sparse HMM marker, map, and allele-count lengths differ".to_owned());
    }
    let sparse_observations = select_observations(observations, marker_indices)?;
    panel.marker_indices = marker_indices.to_vec();
    forward_backward_interpolated_alleles(
        &sparse_observations,
        panel,
        marker_cm_positions,
        cm_positions,
        allele_counts,
        config,
        workspace,
    )
}

fn select_observations(
    observations: &PackedHaplotypes,
    marker_indices: &[usize],
) -> Result<PackedHaplotypes, String> {
    if marker_indices.is_empty()
        || marker_indices
            .iter()
            .any(|&marker| marker >= observations.n_markers)
        || marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("sparse HMM observation marker indices are invalid".to_owned());
    }
    let mut selected =
        PackedHaplotypes::empty(1, marker_indices.len(), observations.bits_per_allele)?;
    for (selected_marker, &source_marker) in marker_indices.iter().enumerate() {
        selected.set_allele(selected_marker, 0, observations.allele(source_marker, 0))?;
    }
    Ok(selected)
}

fn forward_backward_interpolated_alleles(
    observations: &PackedHaplotypes,
    panel: &super::types::CompositePanel,
    hmm_cm_positions: &[f64],
    full_cm_positions: &[f64],
    allele_counts: &[usize],
    config: &HMMConfig,
    workspace: &mut super::types::HMMWorkspace,
) -> Result<PosteriorWindow, String> {
    let marker_count = panel.haplotypes.n_markers;
    if full_cm_positions.len() != marker_count || allele_counts.len() != marker_count {
        return Err("interpolated HMM full marker dimensions differ".to_owned());
    }
    let mut allele_offsets = Vec::with_capacity(marker_count.saturating_add(1));
    allele_offsets.push(0usize);
    for &allele_count in allele_counts {
        let next = allele_offsets
            .last()
            .copied()
            .unwrap()
            .checked_add(allele_count.max(1))
            .ok_or_else(|| "interpolated HMM posterior size overflow".to_owned())?;
        allele_offsets.push(next);
    }
    let mut allele_probabilities = vec![0.0f32; *allele_offsets.last().unwrap_or(&0)];
    let state_count = panel.haplotypes.n_haplotypes;
    let mut right_hmm_marker: Option<usize> = None;
    let mut right_probabilities = vec![0.0f32; state_count];
    let mut interpolated_probabilities = vec![0.0f32; state_count];
    let mut rightmost_probabilities = None;
    let mut left_endpoint = None;

    hmm::forward_backward_with_state_callback(
        observations,
        panel,
        hmm_cm_positions,
        config,
        workspace,
        |hmm_marker, state_probabilities| {
            let marker = panel.marker_indices[hmm_marker];
            let start = allele_offsets[marker];
            let end = allele_offsets[marker + 1];
            fill_marker_allele_probabilities(
                panel,
                marker,
                state_probabilities,
                &mut allele_probabilities[start..end],
            );

            if let Some(right_marker) = right_hmm_marker {
                let right_union_marker = panel.marker_indices[right_marker];
                let span_cm = hmm_cm_positions[right_marker] - hmm_cm_positions[hmm_marker];
                for union_marker in marker + 1..right_union_marker {
                    let alpha = if span_cm > f64::EPSILON {
                        ((full_cm_positions[union_marker] - hmm_cm_positions[hmm_marker]) / span_cm)
                            .clamp(0.0, 1.0) as f32
                    } else {
                        0.0
                    };
                    for (state, &left_probability) in state_probabilities.iter().enumerate() {
                        interpolated_probabilities[state] =
                            left_probability * (1.0 - alpha) + right_probabilities[state] * alpha;
                    }
                    let start = allele_offsets[union_marker];
                    let end = allele_offsets[union_marker + 1];
                    fill_marker_allele_probabilities(
                        panel,
                        union_marker,
                        &interpolated_probabilities,
                        &mut allele_probabilities[start..end],
                    );
                }
            }
            if hmm_marker == 0 {
                left_endpoint = Some(state_probabilities.to_vec());
            }
            if right_hmm_marker.is_none() {
                rightmost_probabilities = Some(state_probabilities.to_vec());
            }
            right_probabilities.copy_from_slice(state_probabilities);
            right_hmm_marker = Some(hmm_marker);
            Ok(())
        },
    )?;

    if let Some(probabilities) = left_endpoint {
        let first_marker = panel.marker_indices[0];
        for marker in 0..first_marker {
            let start = allele_offsets[marker];
            let end = allele_offsets[marker + 1];
            fill_marker_allele_probabilities(
                panel,
                marker,
                &probabilities,
                &mut allele_probabilities[start..end],
            );
        }
    }
    if let Some(probabilities) = rightmost_probabilities {
        let last_hmm_marker = observations.n_markers - 1;
        let last_marker = panel.marker_indices[last_hmm_marker];
        for marker in last_marker + 1..marker_count {
            let start = allele_offsets[marker];
            let end = allele_offsets[marker + 1];
            fill_marker_allele_probabilities(
                panel,
                marker,
                &probabilities,
                &mut allele_probabilities[start..end],
            );
        }
    }

    Ok(PosteriorWindow {
        allele_probabilities,
        leave_one_out_allele_probabilities: Vec::new(),
        allele_offsets,
        state_probabilities: Vec::new(),
        parameter_stats: Default::default(),
    })
}

fn fill_marker_allele_probabilities(
    panel: &super::types::CompositePanel,
    marker: usize,
    state_probabilities: &[f32],
    allele_probabilities: &mut [f32],
) {
    allele_probabilities.fill(0.0);
    if allele_probabilities.is_empty() {
        return;
    }
    let missing_share = 1.0 / allele_probabilities.len() as f32;
    for (state, &probability) in state_probabilities.iter().enumerate() {
        match panel.haplotypes.allele(marker, state) {
            Some(allele) => {
                if let Some(value) = allele_probabilities.get_mut(usize::from(allele)) {
                    *value += probability;
                } else {
                    for value in allele_probabilities.iter_mut() {
                        *value += probability * missing_share;
                    }
                }
            }
            None => {
                for value in allele_probabilities.iter_mut() {
                    *value += probability * missing_share;
                }
            }
        }
    }
    let sum = allele_probabilities.iter().sum::<f32>();
    if sum > f32::MIN_POSITIVE {
        for probability in allele_probabilities {
            *probability /= sum;
        }
    } else {
        allele_probabilities.fill(missing_share);
    }
}

pub(crate) fn phase_ibs_step_cm(cm_positions: &[f64]) -> f64 {
    let mut distances = cm_positions
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    if distances.is_empty() {
        return 1.0;
    }
    distances.sort_by(f64::total_cmp);
    let middle = distances.len() / 2;
    let lower = distances[(distances.len() - 1) / 2];
    let upper = distances[middle];
    // Beagle's default step scale is 3 times the median genetic distance
    // between all adjacent stage-1 markers, including zero-length intervals,
    // with the median sum and scale rounded in f32.
    let median = 0.5f32 * ((lower + upper) as f32);
    f64::from(3.0f32 * median).max(f64::from(f32::MIN_POSITIVE))
}

pub(crate) fn beagle_phase_marker_map(window: &PhaseWindow) -> Result<Vec<f64>, String> {
    if window.rows.is_empty()
        || window.rows.len() != window.cm_positions.len()
        || window.cm_positions.iter().any(|cm| !cm.is_finite())
        || window.cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("phase marker map dimensions or coordinates are invalid".to_owned());
    }
    if window.rows.len() == 1 {
        return Ok(window.cm_positions.clone());
    }
    let first_raw = window.cm_positions[0];
    let last_raw = *window.cm_positions.last().unwrap();
    let first_pos = window.rows.first().unwrap().meta.pos;
    let last_pos = window.rows.last().unwrap().meta.pos;
    if first_pos == last_pos {
        return Err("phase marker map cannot span one physical position".to_owned());
    }
    let minimum_step =
        ((last_raw - first_raw).abs() / (last_pos as f64 - first_pos as f64).abs()).max(1.0e-8);
    let mut effective = Vec::with_capacity(window.cm_positions.len());
    effective.push(first_raw);
    let mut previous_raw = first_raw;
    for &raw in &window.cm_positions[1..] {
        let previous_effective = *effective.last().unwrap();
        effective.push(previous_effective + (raw - previous_raw).max(minimum_step));
        previous_raw = raw;
    }
    Ok(effective)
}

fn phase_candidate_count(
    iteration: usize,
    burnin: usize,
    iterations: usize,
    haplotype_count: usize,
) -> usize {
    const BURNIN_CANDIDATES: usize = 100;
    const MAX_PHASE_CANDIDATES: usize = 90;
    const MIN_PHASE_CANDIDATES: usize = 5;

    let total_iterations = burnin.saturating_add(iterations);
    let requested = if iteration < burnin {
        BURNIN_CANDIDATES
    } else if iterations > 0 && iteration < total_iterations {
        let remaining = total_iterations - iteration;
        ((remaining as f64 / iterations as f64 * MAX_PHASE_CANDIDATES as f64).round() as usize)
            .max(MIN_PHASE_CANDIDATES)
    } else {
        BURNIN_CANDIDATES
    };
    requested.min(haplotype_count)
}

fn phase_round_seed(window_seed: u64, round: usize) -> u64 {
    // Beagle's PhaseData.seed() is its master phase seed plus the current
    // iteration. PBWT donor and PhaseBaum sample streams therefore change
    // between rounds, while all work within one round uses this value plus a
    // deterministic step or sample offset.
    window_seed.wrapping_add(round as u64)
}

fn phase_sample_seed(window_seed: u64, round: usize, sample: usize) -> u64 {
    phase_round_seed(window_seed, round).wrapping_add(sample as u64)
}

fn low_frequency_phase_state_count(stage1_phase_states: usize) -> usize {
    (stage1_phase_states / 2).max(1)
}

fn stage2_phase_seed(window_seed: u64, completed_rounds: usize) -> u64 {
    // The Beagle driver increments PhaseData.it after each Stage1 round.
    // Stage2 therefore sees PhaseData.seed() == master seed + total rounds.
    window_seed.wrapping_add(completed_rounds as u64)
}

fn next_beagle_phase_iteration(
    current_iteration: usize,
    burnin: usize,
    phase_swap_count: usize,
    unphased_heterozygote_count: usize,
) -> usize {
    let next_iteration = current_iteration.saturating_add(1);
    if next_iteration < burnin
        && unphased_heterozygote_count > 0
        && phase_swap_count as f64 / unphased_heterozygote_count as f64 <= 0.01
    {
        burnin
    } else {
        next_iteration
    }
}

fn parameter_trace_row(
    window: &PhaseWindow,
    stage: &str,
    round: usize,
    config: &HMMConfig,
) -> HMMParameterTrace {
    HMMParameterTrace {
        chrom: window.chrom.clone(),
        window_index: window.index,
        stage: stage.to_owned(),
        round,
        ne: config.ne,
        err: config.err,
    }
}

fn initialize_em_parameters(
    window: &PhaseWindow,
    snapshot: &PhasedWindow,
    haplotype_view: &composite::PhaseHaplotypeView<'_>,
    pbwt_index: &pbwt::PhasePbwtIndex,
    mosaic_segment_cm: f64,
    target_samples: usize,
    max_states: usize,
    seed: u64,
    config: &mut HMMConfig,
    trace: &mut Vec<HMMParameterTrace>,
    pool: &ThreadPool,
) -> Result<(), String> {
    let mut previous_intensity = recombination_intensity(config);
    let mut sample_random = pbwt::JavaRandom::new(seed);
    for initialization_round in 1..=MAX_EM_INITIALIZATION_ROUNDS {
        let stats = estimate_em_parameters(
            window,
            snapshot,
            haplotype_view,
            pbwt_index,
            &window.cm_positions,
            mosaic_segment_cm,
            target_samples,
            max_states,
            seed,
            &mut sample_random,
            config,
            pool,
        )?;
        update_em_parameters(config, stats);
        trace.push(parameter_trace_row(
            window,
            "em_init",
            initialization_round as usize,
            config,
        ));
        let intensity = recombination_intensity(config);
        if em_initialization_converged(previous_intensity, intensity) {
            break;
        }
        previous_intensity = intensity;
    }
    Ok(())
}

fn estimate_em_parameters(
    window: &PhaseWindow,
    snapshot: &PhasedWindow,
    haplotype_view: &composite::PhaseHaplotypeView<'_>,
    pbwt_index: &pbwt::PhasePbwtIndex,
    cm_positions: &[f64],
    mosaic_segment_cm: f64,
    target_samples: usize,
    max_states: usize,
    phase_seed: u64,
    sample_random: &mut pbwt::JavaRandom,
    config: &HMMConfig,
    pool: &ThreadPool,
) -> Result<HMMParameterStats, String> {
    let available_samples = snapshot.rows.first().map_or(0, |row| row.calls.len());
    if target_samples == 0 || target_samples > available_samples {
        return Ok(HMMParameterStats::default());
    }
    let mut sample_indices = (0..target_samples).collect::<Vec<_>>();
    if sample_indices.len() > MAX_EM_TARGET_SAMPLES {
        for index in 0..MAX_EM_TARGET_SAMPLES {
            let remaining = sample_indices.len() - index;
            let selected = index + sample_random.next_int(remaining);
            sample_indices.swap(index, selected);
        }
        sample_indices.truncate(MAX_EM_TARGET_SAMPLES);
    }

    let min_lifetime_steps = ((1.0f32 / mosaic_segment_cm as f32).ceil() as usize).max(200);
    let transition_data = hmm::prepare_em_transition_data(cm_positions, config)?;
    let sample_stats = pool.install(|| {
        sample_indices
            .par_iter()
            .map(|&sample| {
                with_phase_em_workspace(|workspace| {
                    estimate_sample_parameters(
                        window,
                        snapshot,
                        haplotype_view,
                        pbwt_index,
                        cm_positions,
                        &transition_data,
                        sample,
                        max_states,
                        min_lifetime_steps,
                        phase_seed,
                        config,
                        workspace,
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    Ok(aggregate_beagle_em_stats(sample_stats))
}

#[allow(clippy::too_many_arguments)]
fn estimate_sample_parameters(
    window: &PhaseWindow,
    snapshot: &PhasedWindow,
    haplotype_view: &composite::PhaseHaplotypeView<'_>,
    pbwt_index: &pbwt::PhasePbwtIndex,
    cm_positions: &[f64],
    transition_data: &hmm::EmTransitionData,
    sample: usize,
    max_states: usize,
    min_lifetime_steps: usize,
    phase_seed: u64,
    config: &HMMConfig,
    workspace: &mut PhaseEmWorkspace,
) -> Result<HMMParameterStats, String> {
    let profile = phase_profiling_enabled();
    let mosaic_started = profile.then(Instant::now);
    composite::build_joint_phase_mosaic_reuse_validated_view(
        haplotype_view,
        pbwt_index,
        sample,
        max_states,
        min_lifetime_steps,
        phase_seed.wrapping_add(sample as u64),
        &mut workspace.panel,
        &mut workspace.donors,
    )?;
    if let Some(started) = mosaic_started {
        EM_MOSAIC_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let panel = workspace
        .panel
        .as_ref()
        .ok_or_else(|| "EM phase panel was not constructed".to_owned())?;
    // Beagle skips parameter updates for samples with a single composite state.
    if panel.haplotypes.n_haplotypes <= 1 {
        return Ok(HMMParameterStats::default());
    }
    workspace.observations0.clear();
    workspace.observations1.clear();
    workspace.observations0.reserve(window.rows.len());
    workspace.observations1.reserve(window.rows.len());
    for row in &snapshot.rows {
        let call = row.calls.get(sample).unwrap();
        workspace.observations0.push(call.alleles[0]);
        workspace.observations1.push(call.alleles[1]);
    }
    let mut stats = HMMParameterStats::default();
    let hmm_started = profile.then(Instant::now);
    if workspace.observations0 == workspace.observations1 {
        let track_stats = hmm::forward_backward_for_em_single_observation_with_transition_data(
            &workspace.observations0,
            panel,
            cm_positions,
            config,
            transition_data,
        )?;
        stats.add_assign(track_stats);
        stats.add_assign(track_stats);
    } else {
        let (stats0, stats1) = hmm::forward_backward_for_em_pair_with_workspace(
            &workspace.observations0,
            &workspace.observations1,
            panel,
            cm_positions,
            config,
            transition_data,
            &mut workspace.hmm,
        )?;
        stats.add_assign(stats0);
        stats.add_assign(stats1);
    }
    if let Some(started) = hmm_started {
        EM_HMM_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if std::env::var("JANUSX_TRACE_EM_SAMPLE").is_ok_and(|targets| {
        targets
            .split(',')
            .any(|target| target == format!("{}:{}", window.chrom, window.index))
    }) {
        let phase_hash = snapshot.rows.iter().fold(0u64, |hash, row| {
            row.calls
                .get(sample)
                .unwrap()
                .alleles
                .into_iter()
                .fold(hash, |hash, allele| {
                    hash.wrapping_mul(31)
                        .wrapping_add(allele.map(i64::from).unwrap_or(-1) as u64)
                })
        });
        eprintln!(
            "JANUSX_EM_SAMPLE\tWINDOW={}:{}\tSAMPLE={}\tSTATES={}\tPHASEHASH={:016x}\tERRBITS={:016x}\tRECOMBBITS={:08x}\tMM={:.17}\tMMCOUNT={}\tDIST={:.17}\tSWITCH={:.17}",
            window.chrom,
            window.index,
            sample,
            panel.haplotypes.n_haplotypes,
            phase_hash,
            config.err.to_bits(),
            config.recombination_intensity.to_bits(),
            stats.mismatch_probability_sum,
            stats.mismatch_marker_count,
            stats.switch_distance_cm_sum,
            stats.switch_probability_sum,
        );
    }
    Ok(stats)
}

fn em_initialization_converged(previous_intensity: f32, intensity: f32) -> bool {
    previous_intensity.is_finite()
        && previous_intensity > 0.0
        && intensity.is_finite()
        && f64::from((intensity - previous_intensity).abs()) <= 0.1 * f64::from(previous_intensity)
}

#[cfg(test)]
fn em_sample_should_be_analyzed(index: usize, sample_count: usize) -> bool {
    index < sample_count
}

/// Reduces per-sample sufficient statistics in the same stable order as
/// Beagle's `ParamEstimates`. Java emits one record per target sample, sorts
/// mismatch records by (mismatch sum, marker count), sorts recombination
/// records by (distance, switch sum), then accumulates each field in `double`.
fn aggregate_beagle_em_stats(sample_stats: Vec<HMMParameterStats>) -> HMMParameterStats {
    let mut mismatch_data = sample_stats
        .iter()
        .filter(|stats| {
            stats.mismatch_marker_count > 0
                && stats.mismatch_probability_sum > 0.0
                && stats.mismatch_probability_sum.is_finite()
        })
        .map(|stats| (stats.mismatch_probability_sum, stats.mismatch_marker_count))
        .collect::<Vec<_>>();
    mismatch_data.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });

    let mut switch_data = sample_stats
        .iter()
        .filter(|stats| {
            stats.switch_distance_cm_sum > 0.0
                && stats.switch_probability_sum > 0.0
                && stats.switch_distance_cm_sum.is_finite()
                && stats.switch_probability_sum.is_finite()
        })
        .map(|stats| (stats.switch_distance_cm_sum, stats.switch_probability_sum))
        .collect::<Vec<_>>();
    switch_data.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });

    let mut aggregate = HMMParameterStats::default();
    for (mismatch_sum, marker_count) in mismatch_data {
        aggregate.mismatch_marker_count += marker_count;
        aggregate.mismatch_probability_sum += mismatch_sum;
    }
    for (distance_sum, switch_sum) in switch_data {
        aggregate.switch_distance_cm_sum += distance_sum;
        aggregate.switch_probability_sum += switch_sum;
    }
    aggregate
}

fn recombination_intensity(config: &HMMConfig) -> f32 {
    config.recombination_intensity
}

fn update_em_parameters(config: &mut HMMConfig, stats: HMMParameterStats) {
    let mut mismatch_probability_sum = 0.0f64;
    let mut mismatch_marker_count = 0usize;
    let mut switch_probability_sum = 0.0f64;
    let mut switch_distance_cm_sum = 0.0f64;
    mismatch_probability_sum += stats.mismatch_probability_sum;
    mismatch_marker_count += stats.mismatch_marker_count;
    switch_probability_sum += stats.switch_probability_sum;
    switch_distance_cm_sum += stats.switch_distance_cm_sum;
    if mismatch_marker_count > 0 {
        let estimated_err = (mismatch_probability_sum / mismatch_marker_count as f64) as f32;
        if estimated_err.is_finite() && f64::from(estimated_err) > config.err {
            config.err = f64::from(estimated_err.clamp(0.0, 0.5 - f32::EPSILON));
        }
    }
    if switch_distance_cm_sum > 0.0 {
        let estimated_intensity = (switch_probability_sum / switch_distance_cm_sum) as f32;
        if estimated_intensity.is_finite() && estimated_intensity > 0.0 {
            config.recombination_intensity = estimated_intensity;
            let estimated_ne = 25.0_f32 * estimated_intensity * config.haplotype_count as f32;
            if estimated_ne.is_finite() && estimated_ne > 0.0 {
                config.ne = f64::from(estimated_ne).ceil().min(u64::MAX as f64) as u64;
            }
        }
    }
}

fn diplotype_orientation(
    hap0: &[f32],
    hap1: &[f32],
    allele_a: u16,
    allele_b: u16,
    state: &mut u64,
) -> (u16, f64) {
    let forward = f64::from(probability(hap0, allele_a)) * f64::from(probability(hap1, allele_b));
    let reverse = f64::from(probability(hap0, allele_b)) * f64::from(probability(hap1, allele_a));
    let total = forward + reverse;
    if !total.is_finite() || total <= f64::MIN_POSITIVE {
        return (
            if next_random(state) & 1 == 0 {
                allele_a
            } else {
                allele_b
            },
            0.0,
        );
    }
    let orientation = if (forward - reverse).abs() <= f64::EPSILON {
        if next_random(state) & 1 == 0 {
            allele_a
        } else {
            allele_b
        }
    } else if forward > reverse {
        allele_a
    } else {
        allele_b
    };
    let log_confidence = (forward.max(reverse) / forward.min(reverse).max(f64::MIN_POSITIVE)).ln();
    (orientation, log_confidence)
}

fn phase_low_frequency_call(
    input: DiploidCall,
    snapshot: DiploidCall,
    hap0: &[f32],
    hap1: &[f32],
    _any_unphased_heterozygote: bool,
    seed: &mut u64,
) -> [Option<u16>; 2] {
    match input.alleles {
        [Some(a), Some(b)] if a == b => [Some(a), Some(a)],
        [Some(_), Some(_)] if input.phased => snapshot.alleles,
        [Some(a), Some(b)] => {
            let (oriented, _) = diplotype_orientation(hap0, hap1, a, b, seed);
            let other = if oriented == a { b } else { a };
            [Some(oriented), Some(other)]
        }
        [Some(a), None] => [Some(a), Some(allele_map(hap1))],
        [None, Some(b)] if input.phased => [Some(allele_map(hap0)), Some(b)],
        [None, Some(b)] => [Some(b), Some(allele_map(hap1))],
        [None, None] => [Some(allele_map(hap0)), Some(allele_map(hap1))],
    }
}

fn bit_is_set(bits: &[u64], marker: usize) -> bool {
    bits.get(marker >> 6)
        .is_some_and(|word| word & (1u64 << (marker & 63)) != 0)
}

fn mark_finished(bits: &mut [u64], marker: usize) {
    if let Some(word) = bits.get_mut(marker >> 6) {
        *word |= 1u64 << (marker & 63);
    }
}

fn phase_lr_threshold(iteration: usize, burnin: usize, iterations: usize, initial_lr: f64) -> f64 {
    if iteration < burnin {
        return f64::INFINITY;
    }
    if iterations <= 1 || iteration >= burnin.saturating_add(iterations).saturating_sub(1) {
        return 1.0;
    }
    let regular_iteration = iteration - burnin;
    let intervals = iterations - 1;
    let exponent = (intervals - regular_iteration) as f64 / intervals as f64;
    4.0 * (initial_lr / 4.0).powf(exponent)
}

#[cfg(test)]
fn trailing_unphased_het_mask(
    window: &PhaseWindow,
    sample: usize,
    finished: &[u64],
    enabled: bool,
) -> Vec<bool> {
    let mut persistent_mask = vec![0u64; window.rows.len().div_ceil(64)];
    compute_trailing_unphased_het_mask_bits(
        window,
        sample,
        finished,
        enabled,
        &mut persistent_mask,
    );
    (0..window.rows.len())
        .map(|marker| bit_is_set(&persistent_mask, marker))
        .collect()
}

fn update_trailing_unphased_het_mask(
    window: &PhaseWindow,
    sample: usize,
    finished: &[u64],
    enabled: bool,
    persistent_mask: &mut [u64],
) {
    let marker_count = window.rows.len();
    debug_assert_eq!(persistent_mask.len(), marker_count.div_ceil(64));
    for (persistent_word, &finished_word) in persistent_mask.iter_mut().zip(finished) {
        *persistent_word &= !finished_word;
    }
    compute_trailing_unphased_het_mask_bits(window, sample, finished, enabled, persistent_mask);
}

fn compute_trailing_unphased_het_mask_bits(
    window: &PhaseWindow,
    sample: usize,
    finished: &[u64],
    enabled: bool,
    persistent_mask: &mut [u64],
) {
    if !enabled || sample >= window.rows.first().map_or(0, |row| row.calls.len()) {
        return;
    }

    let mut pending = Vec::with_capacity(3);
    let mut first_het_seen = false;
    let apply_pending = |pending: &[usize], persistent_mask: &mut [u64]| {
        if !(2..=3).contains(&pending.len()) {
            return;
        }
        let last_masked = pending.len() - 2;
        if last_masked == 0
            || window.rows[pending[last_masked]].meta.pos - window.rows[pending[0]].meta.pos
                <= 3_000
        {
            for &marker in &pending[..=last_masked] {
                mark_finished(persistent_mask, marker);
            }
        }
    };

    for (marker, row) in window.rows.iter().enumerate() {
        let call = row.calls.get(sample).unwrap();
        if call.is_heterozygous() {
            let initially_unphased = marker >= window.incoming_overlap_len && first_het_seen;
            first_het_seen = true;
            if bit_is_set(persistent_mask, marker) {
                continue;
            }
            if !initially_unphased || bit_is_set(finished, marker) {
                apply_pending(&pending, persistent_mask);
                pending.clear();
            } else {
                pending.push(marker);
            }
        }
    }
    apply_pending(&pending, persistent_mask);
}

fn sample_allele(probabilities: &[f32], state: &mut u64) -> u16 {
    if probabilities.is_empty() {
        return 0;
    }
    let total = probabilities
        .iter()
        .map(|value| value.max(0.0))
        .sum::<f32>();
    if total <= f32::EPSILON {
        return allele_map(probabilities);
    }
    let draw = (next_random(state) as f64 / u64::MAX as f64) as f32 * total;
    let mut cumulative = 0.0f32;
    for (allele, &probability) in probabilities.iter().enumerate() {
        cumulative += probability.max(0.0);
        if draw <= cumulative {
            return allele as u16;
        }
    }
    probabilities.len().saturating_sub(1) as u16
}

fn allele_map(probabilities: &[f32]) -> u16 {
    probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(allele, _)| allele as u16)
        .unwrap_or(0)
}

fn probability(probabilities: &[f32], allele: u16) -> f32 {
    probabilities
        .get(usize::from(allele))
        .copied()
        .unwrap_or(0.0)
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

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{
        accumulate_low_frequency_alleles, aggregate_beagle_em_stats, beagle_allele_map,
        beagle_phase_marker_map, donor_haplotypes_at_anchors, em_initialization_converged,
        em_sample_should_be_analyzed, forward_backward_interpolated_alleles, interpolation_weights,
        low_frequency_allele_flags, low_frequency_phase_state_count, next_beagle_phase_iteration,
        phase_candidate_count, phase_hmm_for_haplotypes, phase_ibs_step_cm,
        phase_low_frequency_call, phase_lr_threshold, phase_round_seed, phase_sample,
        phase_sample_seed, phase_window, phase_window_prepared, project_phase_window,
        project_phased_window, splice_phase_overlap, stage2_anchor_index,
        stage2_observation_alleles, stage2_phase_seed, stage2_sparse_plan, trace_hmm_marker,
        trailing_unphased_het_mask, update_trailing_unphased_het_mask, with_reference_haplotypes,
        MAX_EM_INITIALIZATION_ROUNDS,
    };
    use crate::imp::pbwt;
    use crate::imp::types::{
        CompositePanel, DiploidCall, DiploidCalls, HMMConfig, HMMParameterStats, HMMWorkspace,
        PackedHaplotypes, PhaseConfig, PhaseWindow, PhasedRow, PhasedWindow,
        ResolvedDiagnosticSite, VariantMeta, VariantRow,
    };
    use rayon::ThreadPoolBuilder;

    fn em_initialization_window() -> PhaseWindow {
        let n_samples = 4;
        let n_markers = 32;
        let rows = (0..n_markers)
            .map(|marker| {
                let calls = (0..n_samples)
                    .map(|sample| {
                        if sample == 0 {
                            DiploidCall {
                                alleles: [Some(0), Some(1)],
                                phased: false,
                            }
                        } else {
                            let bit = ((marker * (sample + 1) + sample) % 7) < 3;
                            let allele = u16::from(bit);
                            DiploidCall {
                                alleles: [Some(allele), Some(allele)],
                                phased: true,
                            }
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
            .collect();
        PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            cm_positions: (0..n_markers).map(|marker| marker as f64 * 0.001).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: n_markers,
        }
    }

    #[test]
    fn reference_haplotypes_are_packed_at_target_markers() {
        let target = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![
                VariantRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: 10,
                        id: "target-a".to_owned(),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: false,
                    }]),
                },
                VariantRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: 30,
                        id: "target-b".to_owned(),
                        ref_allele: "C".to_owned(),
                        alt_alleles: vec!["T".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: false,
                    }]),
                },
            ],
            cm_positions: vec![0.1, 0.3],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 2,
        };
        let reference = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![
                PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: 10,
                        id: "reference-a".to_owned(),
                        ref_allele: "G".to_owned(),
                        alt_alleles: vec!["A".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(0), Some(1)],
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
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: 20,
                        id: "reference-only".to_owned(),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["C".to_owned()],
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
                },
                PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: 30,
                        id: "reference-b".to_owned(),
                        ref_allele: "C".to_owned(),
                        alt_alleles: vec!["T".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(0), Some(1)],
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

        let (packed_reference, target_samples, reference_present) =
            with_reference_haplotypes(&target, &reference).unwrap();

        assert_eq!(target_samples, 1);
        assert_eq!(reference_present, vec![true, true]);
        assert_eq!(packed_reference.n_markers, target.rows.len());
        assert_eq!(packed_reference.n_haplotypes, 2);
        assert_eq!(packed_reference.allele(0, 0), Some(1));
        assert_eq!(packed_reference.allele(0, 1), Some(0));
        assert_eq!(packed_reference.allele(1, 0), Some(0));
        assert_eq!(packed_reference.allele(1, 1), Some(1));
    }

    #[test]
    fn reference_phase_window_uses_combined_haplotypes_and_returns_target_samples() {
        let n_markers = 8usize;
        let target_rows = (0..n_markers)
            .map(|marker| {
                let calls = [
                    DiploidCall {
                        alleles: if marker == 3 {
                            [Some(0), Some(1)]
                        } else {
                            let allele = u16::from(marker % 2 == 0);
                            [Some(allele), Some(allele)]
                        },
                        phased: false,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: false,
                    },
                ];
                VariantRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("target-{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(calls.to_vec()),
                }
            })
            .collect();
        let target = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: target_rows,
            cm_positions: (0..n_markers).map(|marker| marker as f64 * 0.01).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: n_markers,
        };
        let reference_rows = (0..n_markers)
            .filter(|&marker| marker != 3)
            .map(|marker| PhasedRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("reference-{marker}"),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [
                        Some(u16::from(marker % 2 == 0)),
                        Some(u16::from(marker % 2 != 0)),
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
            .collect();
        let reference = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: reference_rows,
            emit_start: 0,
            emit_end: n_markers,
        };
        let config = PhaseConfig {
            window_cm: 40.0,
            overlap_cm: 2.0,
            window_markers: 4_000_000,
            seed: 11,
            hmm: HMMConfig {
                ne: 100_000,
                recombination_intensity: (0.04_f32 * 100_000.0) / 4.0,
                err: 0.001,
                em: false,
                burnin: 0,
                iterations: 1,
                initial_lr: 1.0,
                phase_states: 4,
                haplotype_count: 4,
            },
            packed_cache_budget_bytes: 256 * 1024 * 1024,
        };
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

        let (phased, fitted, _, _) =
            phase_window(&target, &config, 11, &[], &pool, Some(&reference), None).unwrap();

        assert_eq!(phased.rows[0].calls.len(), 2);
        assert_eq!(fitted.haplotype_count, 6);
    }

    #[test]
    fn paired_phase_falls_back_when_missing_calls_disable_packed_snapshot() {
        let mut window = em_initialization_window();
        window.rows[0]
            .calls
            .set_call(0, DiploidCall::missing(false))
            .unwrap();
        let config = PhaseConfig {
            window_cm: 40.0,
            overlap_cm: 2.0,
            window_markers: 4_000_000,
            seed: 11,
            hmm: HMMConfig {
                ne: 100_000,
                recombination_intensity: (0.04_f32 * 100_000.0) / 8.0,
                err: 0.01,
                em: false,
                burnin: 0,
                iterations: 1,
                initial_lr: 100_000.0,
                phase_states: 6,
                haplotype_count: 8,
            },
            packed_cache_budget_bytes: 256 * 1024 * 1024,
        };
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        phase_window(&window, &config, 11, &[], &pool, None, None).unwrap();
    }

    #[test]
    fn phase_overlap_splices_the_matching_previous_carry_suffix() {
        let meta = |position: u64| VariantMeta {
            chrom: "1".to_owned(),
            pos: position,
            id: format!("v{position}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let mut window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 1,
            rows: vec![
                VariantRow {
                    meta: meta(2),
                    calls: DiploidCalls::from_calls(vec![DiploidCall::missing(false)]),
                },
                VariantRow {
                    meta: meta(3),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: false,
                    }]),
                },
            ],
            cm_positions: vec![0.1, 0.2],
            incoming_overlap_len: 1,
            emit_start: 0,
            emit_end: 2,
        };
        let previous_carry = vec![
            PhasedRow {
                meta: meta(1),
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
                meta: meta(2),
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
                imputed: true,
            },
        ];

        splice_phase_overlap(&mut window, &previous_carry).unwrap();

        assert_eq!(
            window.rows[0].calls.get(0).unwrap(),
            DiploidCall {
                alleles: [Some(1), Some(0)],
                phased: true,
            }
        );
        assert_eq!(
            window.rows[1].calls.get(0).unwrap().alleles,
            [Some(0), Some(1)],
            "splicing only replaces the incoming overlap prefix"
        );
    }

    #[test]
    fn phase_overlap_preserves_current_calls_for_unresolved_scaffold_overlap() {
        let meta = |position: u64| VariantMeta {
            chrom: "1".to_owned(),
            pos: position,
            id: format!("v{position}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let mut window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 1,
            rows: vec![
                VariantRow {
                    meta: meta(2),
                    calls: DiploidCalls::from_calls(vec![DiploidCall::missing(false)]),
                },
                VariantRow {
                    meta: meta(3),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: false,
                    }]),
                },
            ],
            cm_positions: vec![0.1, 0.2],
            incoming_overlap_len: 1,
            emit_start: 0,
            emit_end: 2,
        };
        let previous_carry = vec![
            PhasedRow {
                meta: meta(1),
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: false,
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
                calls: DiploidCalls::from_calls(vec![DiploidCall::missing(false)]),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            },
        ];

        splice_phase_overlap(&mut window, &previous_carry).unwrap();

        assert_eq!(
            window.rows[0].calls.get(0).unwrap(),
            DiploidCall::missing(false)
        );
    }

    #[test]
    fn stage2_reference_donor_uses_its_selected_phased_allele() {
        let mut input = em_initialization_window();
        input.rows[0].calls = DiploidCalls::from_calls(vec![DiploidCall::missing(false); 4]);
        let snapshot = PhasedWindow {
            chrom: input.chrom.clone(),
            index: 0,
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
            emit_start: 0,
            emit_end: input.rows.len(),
        };
        let mut reference = PackedHaplotypes::empty(2, input.rows.len(), 1).unwrap();
        for marker in 0..input.rows.len() {
            reference.set_allele(marker, 0, Some(0)).unwrap();
            reference.set_allele(marker, 1, Some(1)).unwrap();
        }
        let view = super::composite::PhaseHaplotypeView::new(&snapshot, None, Some(&reference), 4)
            .unwrap();
        let panel = CompositePanel {
            haplotypes: PackedHaplotypes::empty(1, input.rows.len(), 1).unwrap(),
            source_samples: vec![4],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let segments = [vec![super::composite::DonorHapSegment {
            start_marker: 0,
            end_marker: input.rows.len(),
            donor_haplotype: 9,
        }]];
        let mut marginal = [0.0; 2];
        accumulate_low_frequency_alleles(
            &input,
            Some(&view),
            0,
            0,
            0,
            &[1.0],
            &[1.0],
            1.0,
            0.0,
            &panel,
            &segments,
            &[true],
            &[vec![false, true]],
            &[0, 2],
            &mut marginal,
        );
        assert_eq!(marginal, [0.0, 1.0]);
    }

    #[test]
    fn stage2_low_frequency_missing_allele_uses_beagle_rare_allele_weight() {
        let meta = VariantMeta {
            chrom: "1".to_owned(),
            pos: 1,
            id: "rare".to_owned(),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let calls = DiploidCalls::from_calls(vec![
            DiploidCall::missing(false),
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            },
        ]);
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: meta.clone(),
                calls,
            }],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let _snapshot = PhasedWindow {
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
        let mut haplotypes = PackedHaplotypes::empty(1, 1, 1).unwrap();
        haplotypes.set_allele(0, 0, Some(0)).unwrap();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let donor_segments = [vec![super::composite::DonorHapSegment {
            start_marker: 0,
            end_marker: 1,
            donor_haplotype: 2,
        }]];
        let low_frequency_mask = [true];
        let allele_flags = [vec![false, true]];
        let offsets = [0usize, 2];
        let mut marginal = [0.0f32; 2];

        accumulate_low_frequency_alleles(
            &input,
            None,
            0,
            0,
            0,
            &[1.0],
            &[1.0],
            1.0,
            0.0,
            &panel,
            &donor_segments,
            &low_frequency_mask,
            &allele_flags,
            &offsets,
            &mut marginal,
        );

        assert!((marginal[0] - 0.45).abs() < 1e-6);
        assert!((marginal[1] - 0.55).abs() < 1e-6);
    }

    #[test]
    fn stage2_hmm_observations_use_the_phased_stage1_snapshot() {
        let snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: [[Some(0), Some(1)], [Some(1), Some(0)], [Some(1), Some(1)]]
                .into_iter()
                .enumerate()
                .map(|(marker, alleles)| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles,
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
            emit_end: 3,
        };

        let observations = stage2_observation_alleles(&snapshot, &[false, true, false], 0).unwrap();

        assert_eq!(
            observations,
            vec![[Some(0), Some(1)], [None, None], [Some(1), Some(1)],]
        );
    }

    #[test]
    fn stage2_sparse_plan_keeps_only_anchors_around_unresolved_low_frequency_calls() {
        let calls = || {
            DiploidCalls::from_calls(vec![
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: true,
                },
            ])
        };
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: (0..7)
                .map(|marker| VariantRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: calls(),
                })
                .collect(),
            cm_positions: (0..7).map(|marker| marker as f64 * 0.1).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 7,
        };
        let low_frequency_mask = [false, true, false, true, false, true, false];

        let plan = stage2_sparse_plan(&input, 0, &low_frequency_mask, &[0, 2, 4, 6]).unwrap();

        assert_eq!(plan.unresolved_markers, vec![1, 3, 5]);
        assert_eq!(plan.anchor_markers, vec![0, 2, 4, 6]);
    }

    #[test]
    fn donor_anchor_table_matches_point_lookup() {
        let segments = vec![
            vec![
                super::composite::DonorHapSegment {
                    start_marker: 0,
                    end_marker: 4,
                    donor_haplotype: 6,
                },
                super::composite::DonorHapSegment {
                    start_marker: 4,
                    end_marker: 9,
                    donor_haplotype: 8,
                },
            ],
            vec![super::composite::DonorHapSegment {
                start_marker: 0,
                end_marker: 9,
                donor_haplotype: 10,
            }],
        ];
        let anchors = [0, 3, 4, 8, 9];
        let table = donor_haplotypes_at_anchors(&segments, &anchors, segments.len());
        let mut cursor = super::Stage2DonorCursor::new(&segments, segments.len());
        for (anchor_index, &anchor) in anchors.iter().enumerate().rev() {
            let row = cursor.row_for_anchor(anchor);
            assert_eq!(
                row,
                &table[anchor_index * segments.len()..(anchor_index + 1) * segments.len()]
            );
        }
        // Repeated query for the leftmost anchor (used for prefix unresolved markers) is idempotent.
        assert_eq!(cursor.row_for_anchor(anchors[0]), &table[0..segments.len()]);
        for (anchor_index, &anchor) in anchors.iter().enumerate() {
            for (state, state_segments) in segments.iter().enumerate() {
                assert_eq!(
                    table[anchor_index * segments.len() + state],
                    super::composite::donor_haplotype_at(state_segments, anchor)
                        .unwrap_or(usize::MAX)
                );
            }
        }
    }

    #[test]
    fn stage2_anchor_index_uses_the_anchor_ordinal_not_the_marker_ordinal() {
        let selected_hmm_markers = [0, 3, 7, 11];
        assert_eq!(stage2_anchor_index(&selected_hmm_markers, 0), Some(0));
        assert_eq!(stage2_anchor_index(&selected_hmm_markers, 7), Some(2));
        assert_eq!(stage2_anchor_index(&selected_hmm_markers, 11), Some(3));
        assert_eq!(stage2_anchor_index(&selected_hmm_markers, 8), None);
    }

    #[test]
    fn stage2_sample_with_no_unresolved_markers_skips_observation_setup() {
        let meta = VariantMeta {
            chrom: "1".to_owned(),
            pos: 1,
            id: "v1".to_owned(),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: meta.clone(),
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: true,
                }]),
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
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(1), Some(1)],
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
        let sparse_plan = super::Stage2SparsePlan {
            anchor_markers: Vec::new(),
            unresolved_markers: Vec::new(),
        };
        let haplotype_view = super::composite::PhaseHaplotypeView::target(&snapshot);
        let mut finished = [0u64];
        let mut trailing_masked = [0u64];
        let mut phase_workspace = super::phase_baum::PhaseBaumWorkspace::default();
        let config = HMMConfig {
            ne: 100_000,
            recombination_intensity: 500.0,
            err: 0.01,
            em: false,
            burnin: 0,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 8,
            haplotype_count: 2,
        };

        let update = phase_sample(
            &input,
            &snapshot,
            &haplotype_view,
            None,
            None,
            None,
            None,
            None,
            0,
            &mut finished,
            &mut trailing_masked,
            0,
            None,
            None,
            &[0],
            1.0,
            false,
            Some(&[true]),
            Some(&[]),
            Some(&[]),
            true,
            Some(&sparse_plan),
            &input.cm_positions,
            &[0],
            0.01,
            &[],
            &config,
            0,
            7,
            &mut phase_workspace,
        )
        .expect("a Stage2 sample with no unresolved markers should be skipped");

        match &update.calls {
            super::SampleCallUpdate::Full(calls) => {
                assert_eq!(calls.get(0).unwrap().alleles, [Some(1), Some(1)]);
            }
            super::SampleCallUpdate::Sparse { changes } => {
                assert!(changes.is_empty());
                assert_eq!(
                    snapshot.rows[0].calls.get(0).unwrap().alleles,
                    [Some(1), Some(1)]
                );
            }
        }
        assert!(update.diagnostic_trace.is_empty());
    }

    #[test]
    fn stage2_does_not_use_stage1_imputation_for_a_masked_donor_gt() {
        let meta = VariantMeta {
            chrom: "1".to_owned(),
            pos: 1,
            id: "masked_rare".to_owned(),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![VariantRow {
                meta: meta.clone(),
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall::missing(false),
                    DiploidCall::missing(false),
                ]),
            }],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let _snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta,
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall::missing(true),
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
                imputed: true,
            }],
            emit_start: 0,
            emit_end: 1,
        };
        let panel = CompositePanel {
            haplotypes: PackedHaplotypes::empty(1, 1, 1).unwrap(),
            source_samples: vec![1],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let mut marginal = [0.0f32; 2];

        accumulate_low_frequency_alleles(
            &input,
            None,
            0,
            0,
            0,
            &[1.0],
            &[1.0],
            1.0,
            0.0,
            &panel,
            &[vec![super::composite::DonorHapSegment {
                start_marker: 0,
                end_marker: 1,
                donor_haplotype: 2,
            }]],
            &[true],
            &[vec![false, true]],
            &[0, 2],
            &mut marginal,
        );

        assert_eq!(marginal, [0.0, 0.0]);
    }

    #[test]
    fn stage2_uses_left_anchor_donor_across_the_interpolation_interval() {
        let meta = |pos| VariantMeta {
            chrom: "1".to_owned(),
            pos,
            id: format!("rare_{pos}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let calls = || {
            DiploidCalls::from_calls(vec![
                DiploidCall::missing(false),
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: true,
                },
            ])
        };
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![
                VariantRow {
                    meta: meta(1),
                    calls: calls(),
                },
                VariantRow {
                    meta: meta(2),
                    calls: calls(),
                },
            ],
            cm_positions: vec![0.0, 0.1],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 2,
        };
        let _snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
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
            emit_start: 0,
            emit_end: 2,
        };
        let panel = CompositePanel {
            haplotypes: PackedHaplotypes::empty(1, 2, 1).unwrap(),
            source_samples: vec![1],
            marker_indices: vec![0, 1],
            allele_counts: vec![2, 2],
        };
        let segments = [vec![
            super::composite::DonorHapSegment {
                start_marker: 0,
                end_marker: 1,
                donor_haplotype: 2,
            },
            super::composite::DonorHapSegment {
                start_marker: 1,
                end_marker: 2,
                donor_haplotype: 4,
            },
        ]];
        let mut marginal = [0.0f32; 4];

        accumulate_low_frequency_alleles(
            &input,
            None,
            0,
            1,
            0,
            &[1.0],
            &[1.0],
            1.0,
            0.0,
            &panel,
            &segments,
            &[true, true],
            &[vec![false, true], vec![false, true]],
            &[0, 2, 4],
            &mut marginal,
        );

        assert!((marginal[2] - 0.45).abs() < 1e-6);
        assert!((marginal[3] - 0.55).abs() < 1e-6);
    }

    #[test]
    fn stage2_low_frequency_heterozygote_conditioning_and_ties_match_beagle() {
        assert_eq!(
            interpolation_weights(1, 0, 2, &[0.0, 1.0, 4.0]),
            (0.75, 0.25)
        );
        assert_eq!(beagle_allele_map(&[0.5, 0.5]), 0);

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
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: false,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                ]),
            }],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };
        let _snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta,
                calls: input.rows[0].calls.clone(),
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
        let flags = low_frequency_allele_flags(&input, 2);
        assert_eq!(flags, [vec![false, true]]);
        let mut haplotypes = PackedHaplotypes::empty(1, 1, 1).unwrap();
        haplotypes.set_allele(0, 0, Some(0)).unwrap();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let mut marginal = [0.0f32; 2];
        accumulate_low_frequency_alleles(
            &input,
            None,
            0,
            0,
            0,
            &[1.0],
            &[1.0],
            1.0,
            0.0,
            &panel,
            &[vec![super::composite::DonorHapSegment {
                start_marker: 0,
                end_marker: 1,
                donor_haplotype: 2,
            }]],
            &[true],
            &flags,
            &[0, 2],
            &mut marginal,
        );
        assert_eq!(marginal, [0.0, 1.0]);
    }

    #[test]
    fn stage2_unclassified_heterozygous_donor_contributes_each_allele_equally() {
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
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall::missing(false),
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
        let _snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta,
                calls: input.rows[0].calls.clone(),
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
        let mut haplotypes = PackedHaplotypes::empty(1, 1, 1).unwrap();
        haplotypes.set_allele(0, 0, Some(0)).unwrap();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let mut marginal = [0.0f32; 2];

        accumulate_low_frequency_alleles(
            &input,
            None,
            0,
            0,
            0,
            &[1.0],
            &[1.0],
            1.0,
            0.0,
            &panel,
            &[vec![super::composite::DonorHapSegment {
                start_marker: 0,
                end_marker: 1,
                donor_haplotype: 2,
            }]],
            &[true],
            &[vec![false, false]],
            &[0, 2],
            &mut marginal,
        );

        assert_eq!(marginal, [0.5, 0.5]);
    }

    #[test]
    fn stage2_low_frequency_allele_flags_count_carrier_samples() {
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
                        alleles: [Some(1), Some(1)],
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
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                ]),
            }],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 1,
        };

        assert_eq!(low_frequency_allele_flags(&input, 3), [vec![false, true]]);
    }

    #[test]
    fn em_initializes_parameters_before_first_phase_even_without_burnin_rounds() {
        let window = em_initialization_window();
        let initial_err = 1.0e-8;
        let config = PhaseConfig {
            window_cm: 40.0,
            overlap_cm: 2.0,
            window_markers: 4_000_000,
            seed: 11,
            hmm: HMMConfig {
                ne: 100_000,
                recombination_intensity: (0.04_f32 * 100_000.0) / 8.0,
                err: initial_err,
                em: true,
                burnin: 0,
                iterations: 1,
                initial_lr: 100_000.0,
                phase_states: 6,
                haplotype_count: 8,
            },
            packed_cache_budget_bytes: 256 * 1024 * 1024,
        };
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

        let (_, fitted, trace, _) =
            phase_window(&window, &config, 11, &[], &pool, None, None).unwrap();

        assert!(
            fitted.err > initial_err,
            "EM initialization should update err before phasing even when burnin=0; got {}",
            fitted.err
        );
        assert_eq!(trace.first().unwrap().stage, "initial");
        assert!(trace.iter().any(|row| row.stage == "em_init"));
        assert!(
            trace.iter().filter(|row| row.stage == "em_init").count()
                <= MAX_EM_INITIALIZATION_ROUNDS
        );
        assert_eq!(
            trace
                .iter()
                .find(|row| row.stage == "phase_start")
                .unwrap()
                .round,
            0
        );
    }

    #[test]
    fn em_initialization_converges_at_ten_percent_intensity_change() {
        assert!(em_initialization_converged(100.0, 110.0));
        assert!(em_initialization_converged(100.0, 90.0));
        assert!(!em_initialization_converged(100.0, 110.0001));
        assert!(!em_initialization_converged(0.0, 0.0));
    }

    #[test]
    fn em_sampling_processes_every_selected_target_like_fast_beagle() {
        assert!(em_sample_should_be_analyzed(0, 64));
        assert!(em_sample_should_be_analyzed(49, 64));
        assert!(em_sample_should_be_analyzed(63, 64));
        assert!(!em_sample_should_be_analyzed(64, 64));
    }

    #[test]
    fn em_sample_stats_use_beagle_sorted_double_reductions() {
        let sample_stats = vec![
            HMMParameterStats {
                mismatch_probability_sum: 1_000_000_000.0,
                mismatch_marker_count: 1_000_000_000,
                switch_probability_sum: 1_000_000_000.0,
                switch_distance_cm_sum: 1_000_000_000.0,
            },
            HMMParameterStats {
                mismatch_probability_sum: 0.25,
                mismatch_marker_count: 1,
                switch_probability_sum: 0.25,
                switch_distance_cm_sum: 1.0,
            },
            HMMParameterStats {
                mismatch_probability_sum: 0.5,
                mismatch_marker_count: 1,
                switch_probability_sum: 0.5,
                switch_distance_cm_sum: 1.0,
            },
        ];

        let aggregate = aggregate_beagle_em_stats(sample_stats);

        assert_eq!(aggregate.mismatch_marker_count, 1_000_000_002);
        assert_eq!(aggregate.mismatch_probability_sum, 1_000_000_000.75);
        assert_eq!(aggregate.switch_distance_cm_sum, 1_000_000_002.0);
        assert_eq!(aggregate.switch_probability_sum, 1_000_000_000.75);
    }

    #[test]
    fn burnin_em_updates_are_traced_before_their_phase_round() {
        let window = em_initialization_window();
        let config = PhaseConfig {
            window_cm: 40.0,
            overlap_cm: 2.0,
            window_markers: 4_000_000,
            seed: 11,
            hmm: HMMConfig {
                ne: 100_000,
                recombination_intensity: (0.04_f32 * 100_000.0) / 8.0,
                err: 1.0e-8,
                em: true,
                burnin: 2,
                iterations: 1,
                initial_lr: 100_000.0,
                phase_states: 6,
                haplotype_count: 8,
            },
            packed_cache_budget_bytes: 256 * 1024 * 1024,
        };
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

        let (_, _, trace, _) = phase_window(&window, &config, 11, &[], &pool, None, None).unwrap();

        let update = trace
            .iter()
            .position(|row| row.stage == "burnin_update" && row.round == 1)
            .unwrap();
        let phase = trace
            .iter()
            .position(|row| row.stage == "phase_start" && row.round == 1)
            .unwrap();
        assert!(update < phase);
        assert_eq!(trace[update].ne, trace[phase].ne);
        assert_eq!(trace[update].err, trace[phase].err);
    }

    #[test]
    fn two_stage_window_emits_hmm_diagnostic_for_common_masked_site() {
        let n_samples = 8;
        let n_markers = 16;
        let rows = (0..n_markers)
            .map(|marker| {
                let calls = (0..n_samples)
                    .map(|sample| {
                        if marker == 0 && sample == 0 {
                            DiploidCall::missing(false)
                        } else if marker < 4 || sample == 1 {
                            DiploidCall {
                                alleles: [Some(0), Some(1)],
                                phased: sample != 0,
                            }
                        } else {
                            DiploidCall {
                                alleles: [Some(0), Some(0)],
                                phased: true,
                            }
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
            .collect();
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 3,
            rows,
            cm_positions: (0..n_markers).map(|marker| marker as f64 * 0.001).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: n_markers,
        };
        let config = PhaseConfig {
            window_cm: 40.0,
            overlap_cm: 2.0,
            window_markers: 4_000_000,
            seed: 11,
            hmm: HMMConfig {
                ne: 100_000,
                recombination_intensity: (0.04_f32 * 100_000.0) / (n_samples * 2) as f32,
                err: 0.001,
                em: false,
                burnin: 0,
                iterations: 1,
                initial_lr: 1.0,
                phase_states: 4,
                haplotype_count: n_samples * 2,
            },
            packed_cache_budget_bytes: 256 * 1024 * 1024,
        };
        let diagnostic_site = ResolvedDiagnosticSite {
            chrom: "1".to_owned(),
            position: 1,
            sample: "s0".to_owned(),
            sample_index: 0,
        };
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

        let (phased, _, _, trace) = phase_window(
            &window,
            &config,
            11,
            &[],
            &pool,
            None,
            Some(&diagnostic_site),
        )
        .unwrap();

        assert!(
            !trace.is_empty(),
            "selected high-frequency marker must be traced"
        );
        assert!(trace
            .iter()
            .all(|row| row.position == diagnostic_site.position));
        assert!(trace
            .iter()
            .all(|row| row.emission_probabilities.len() == 3));
        assert!(trace.iter().all(|row| row.forward_probabilities.len() == 3));
        assert!(trace
            .iter()
            .all(|row| row.backward_probabilities.len() == 3));
        assert!(trace.iter().all(|row| row.state_alleles.len() == 3));
        let selected = phased.rows[0].calls.get(0).unwrap();
        assert_eq!(trace.last().unwrap().selected_alleles, selected.alleles);
    }

    #[test]
    fn missing_event_remains_missing_across_phase_rounds() {
        let n_samples = 4;
        let n_markers = 8;
        let rows = (0..n_markers)
            .map(|marker| {
                let calls = (0..n_samples)
                    .map(|sample| match (marker, sample) {
                        (0, 3) => DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: false,
                        },
                        (1, 0) => DiploidCall {
                            alleles: [None, Some(1)],
                            phased: false,
                        },
                        _ => DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: true,
                        },
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
            .collect();
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            cm_positions: (0..n_markers).map(|marker| marker as f64 * 0.001).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: n_markers,
        };
        let reference = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: (0..n_markers)
                .map(|marker| PhasedRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("r{}", marker + 1),
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
                })
                .collect(),
            emit_start: 0,
            emit_end: n_markers,
        };
        let config = PhaseConfig {
            window_cm: 40.0,
            overlap_cm: 0.0,
            window_markers: 4_000_000,
            seed: 11,
            hmm: HMMConfig {
                ne: 100_000,
                recombination_intensity: (0.04_f32 * 100_000.0) / (n_samples * 2) as f32,
                err: 0.001,
                em: false,
                burnin: 1,
                iterations: 2,
                initial_lr: 100_000.0,
                phase_states: 4,
                haplotype_count: n_samples * 2,
            },
            packed_cache_budget_bytes: 256 * 1024 * 1024,
        };
        let diagnostic_site = ResolvedDiagnosticSite {
            chrom: "1".to_owned(),
            position: 2,
            sample: "s0".to_owned(),
            sample_index: 0,
        };
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

        let window_seed = pbwt::beagle_window_seed(config.seed, 0);
        let (phased, _, _, trace) = phase_window(
            &window,
            &config,
            window_seed,
            &[],
            &pool,
            Some(&reference),
            Some(&diagnostic_site),
        )
        .unwrap();

        assert_eq!(
            trace.iter().map(|row| row.round).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let estimated = phased.rows[1].calls.get(0).unwrap();
        assert!(
            estimated.alleles.iter().all(Option::is_some),
            "the HMM must resolve the missing haplotype allele; observed {estimated:?}"
        );
        assert_eq!(
            trace
                .iter()
                .map(|row| row.cluster_kind.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("missing"), Some("missing"), Some("missing")],
            "an original missing GT must remain a missing cluster after its allele estimate is filled"
        );
    }

    #[test]
    fn prepared_phase_rejects_low_frequency_mask_from_spliced_genotypes() {
        let window = em_initialization_window();
        let config = PhaseConfig {
            window_cm: 40.0,
            overlap_cm: 0.0,
            window_markers: 4_000_000,
            seed: 11,
            hmm: HMMConfig {
                ne: 100_000,
                recombination_intensity: 0.002,
                err: 0.001,
                em: false,
                burnin: 0,
                iterations: 1,
                initial_lr: 100_000.0,
                phase_states: 4,
                haplotype_count: 8,
            },
            packed_cache_budget_bytes: 256 * 1024 * 1024,
        };
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let spliced_mask = vec![false; window.rows.len()];

        let error = phase_window_prepared(
            &window,
            &window,
            &config,
            11,
            &[],
            &pool,
            None,
            None,
            Some(&spliced_mask),
            None,
        )
        .unwrap_err();

        assert_eq!(
            error,
            "prepared low-frequency mask does not match the original carrier window"
        );
    }

    #[test]
    fn stage2_seed_includes_beagle_completed_rounds() {
        let phase_seed = 20261101;
        assert_eq!(stage2_phase_seed(phase_seed, 0), phase_seed);
        assert_eq!(
            stage2_phase_seed(phase_seed, 20),
            phase_seed.wrapping_add(20)
        );
    }

    #[test]
    fn trace_hmm_marker_maps_sparse_hmm_index_to_full_panel_marker() {
        let mut observations = PackedHaplotypes::empty(1, 3, 1).unwrap();
        observations.set_allele(0, 0, Some(0)).unwrap();
        observations.set_allele(1, 0, None).unwrap();
        observations.set_allele(2, 0, Some(1)).unwrap();
        let mut haplotypes = PackedHaplotypes::empty(2, 3, 1).unwrap();
        for (marker, alleles) in [[0, 1], [1, 0], [1, 0]].into_iter().enumerate() {
            for (state, allele) in alleles.into_iter().enumerate() {
                haplotypes.set_allele(marker, state, Some(allele)).unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1, 2],
            marker_indices: vec![0, 2],
            allele_counts: vec![2; 3],
        };
        let config = HMMConfig {
            ne: 100_000,
            recombination_intensity: (0.04_f32 * 100_000.0) / 4.0,
            err: 0.001,
            em: false,
            burnin: 0,
            iterations: 1,
            initial_lr: 1.0,
            phase_states: 2,
            haplotype_count: 4,
        };

        let diagnostic = trace_hmm_marker(&observations, &panel, &[0.0, 0.01, 0.02], 2, &config)
            .unwrap()
            .unwrap();

        assert_eq!(diagnostic.recursion.marker, 1);
        assert_eq!(diagnostic.state_alleles, vec![Some(1), Some(0)]);
        assert_eq!(diagnostic.recursion.emission_probabilities.len(), 2);
    }

    #[test]
    fn stage1_projection_keeps_high_frequency_marker_order_and_phase_calls() {
        let window = em_initialization_window();
        let marker_indices = [1, 4, 9, 17];
        let projected = project_phase_window(&window, &marker_indices).unwrap();
        let phased = super::pbwt::initialize(&window, 11).unwrap();
        let projected_phased = project_phased_window(&phased, &marker_indices).unwrap();

        assert_eq!(projected.rows.len(), marker_indices.len());
        assert!(projected
            .cm_positions
            .iter()
            .zip([0.001, 0.004, 0.009, 0.017])
            .all(|(&actual, expected)| (actual - expected).abs() < 1.0e-12));
        assert_eq!(projected.rows[0].meta.pos, 2);
        assert_eq!(projected.rows[3].meta.pos, 18);
        assert_eq!(projected_phased.rows.len(), marker_indices.len());
        assert_eq!(
            projected_phased.rows[2].meta.pos,
            window.rows[marker_indices[2]].meta.pos
        );
        for (local, &full) in marker_indices.iter().enumerate() {
            assert_eq!(
                projected_phased.rows[local].calls.get(0).unwrap(),
                phased.rows[full].calls.get(0).unwrap()
            );
        }
    }

    #[test]
    fn fully_missing_call_uses_per_haplotype_posterior_modes() {
        let input = DiploidCall::missing(false);
        let snapshot = DiploidCall::missing(true);
        let mut seed = 7;

        let call = phase_low_frequency_call(
            input,
            snapshot,
            &[0.51, 0.49],
            &[0.51, 0.49],
            false,
            &mut seed,
        );

        assert_eq!(call, [Some(0), Some(0)]);
    }

    #[test]
    fn low_frequency_stage_preserves_input_phased_heterozygote_orientation() {
        let input = DiploidCall {
            alleles: [Some(0), Some(1)],
            phased: true,
        };
        let snapshot = input;
        let mut seed = 7;

        let call =
            phase_low_frequency_call(input, snapshot, &[0.0, 1.0], &[1.0, 0.0], true, &mut seed);

        assert_eq!(call, [Some(0), Some(1)]);
    }

    #[test]
    fn phase_lr_threshold_matches_beagle_schedule() {
        assert!(phase_lr_threshold(2, 3, 12, 100_000.0).is_infinite());
        assert_eq!(phase_lr_threshold(3, 3, 12, 100_000.0), 100_000.0);
        assert!(phase_lr_threshold(13, 3, 12, 100_000.0) > 4.0);
        assert_eq!(phase_lr_threshold(14, 3, 12, 100_000.0), 1.0);
        assert_eq!(phase_lr_threshold(3, 3, 1, 100_000.0), 1.0);
    }

    #[test]
    fn phase_candidate_count_matches_beagle_iteration_schedule() {
        assert_eq!(phase_candidate_count(0, 3, 12, 128), 100);
        assert_eq!(phase_candidate_count(3, 3, 12, 128), 90);
        assert_eq!(phase_candidate_count(13, 3, 12, 128), 15);
        assert_eq!(phase_candidate_count(14, 3, 12, 128), 8);
        assert_eq!(phase_candidate_count(14, 3, 12, 6), 6);
    }

    #[test]
    fn phase_round_seed_matches_beagle_phase_data_iteration_seed() {
        let master = 10_374_988_707_383_432_852u64;
        assert_eq!(phase_round_seed(master, 0), master);
        assert_eq!(phase_round_seed(master, 1), master.wrapping_add(1));
        assert_eq!(phase_sample_seed(master, 3, 207), master.wrapping_add(210));
    }

    #[test]
    fn phase_ibs_step_includes_zero_length_map_intervals() {
        let positions = [0.0, 0.0, 0.0, 0.01, 0.02];
        assert_eq!(
            phase_ibs_step_cm(&positions),
            f64::from(3.0f32 * (0.5f32 * 0.01f32))
        );
    }

    #[test]
    fn phase_marker_map_enforces_beagle_minimum_physical_distance() {
        let mut window = em_initialization_window();
        window.rows.truncate(4);
        for (marker, row) in window.rows.iter_mut().enumerate() {
            row.meta.pos = 10 + marker as u64 * 10;
        }
        window.cm_positions = vec![0.0, 5.0, 10.0, 10.1];
        let map = beagle_phase_marker_map(&window).unwrap();
        let minimum_step = 10.1 / 30.0;

        assert_eq!(&map[..3], &[0.0, 5.0, 10.0]);
        assert!((map[3] - (10.0 + minimum_step)).abs() < 1e-14);
        assert!(map
            .windows(2)
            .all(|pair| pair[1] - pair[0] + 1e-14 >= minimum_step));
    }

    #[test]
    fn stage1_error_matches_beagle_li_stephens_not_imputation_error() {
        let config = HMMConfig {
            ne: 100_000,
            recombination_intensity: 0.5,
            err: 0.0001,
            em: false,
            burnin: 5,
            iterations: 15,
            initial_lr: 100_000.0,
            phase_states: 1_000,
            haplotype_count: 1_348,
        };
        let total_haplotypes = 6_734;
        let phase = phase_hmm_for_haplotypes(config, total_haplotypes).unwrap();
        let expected =
            f64::from(super::super::types::default_err(total_haplotypes).unwrap() as f32);

        assert_eq!(phase.err, expected);
        assert!(phase.err < config.err);
        assert_eq!(phase.haplotype_count, total_haplotypes);
    }

    #[test]
    fn burnin_early_advance_matches_beagle_swap_rate_rule() {
        assert_eq!(next_beagle_phase_iteration(0, 3, 0, 200), 3);
        assert_eq!(next_beagle_phase_iteration(0, 3, 3, 200), 1);
        assert_eq!(next_beagle_phase_iteration(1, 3, 1, 200), 3);
        assert_eq!(next_beagle_phase_iteration(2, 3, 0, 200), 3);
        assert_eq!(next_beagle_phase_iteration(0, 3, 0, 0), 1);
    }

    #[test]
    fn low_frequency_stage_uses_half_the_stage1_state_limit() {
        assert_eq!(low_frequency_phase_state_count(280), 140);
        assert_eq!(low_frequency_phase_state_count(1), 1);
    }

    fn trailing_het_window(calls: Vec<DiploidCall>, positions: Vec<u64>) -> PhaseWindow {
        let rows = calls
            .into_iter()
            .zip(positions)
            .enumerate()
            .map(|(marker, (call, pos))| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![call]),
            })
            .collect::<Vec<_>>();
        PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            cm_positions: (0..rows.len()).map(|marker| marker as f64 * 0.01).collect(),
            emit_start: 0,
            emit_end: rows.len(),
            rows,
            incoming_overlap_len: 0,
        }
    }

    fn unphased_het() -> DiploidCall {
        DiploidCall {
            alleles: [Some(0), Some(1)],
            phased: false,
        }
    }

    #[test]
    fn trailing_mask_only_applies_when_beagle_lr_is_below_fifty() {
        let window = trailing_het_window(vec![unphased_het(), unphased_het()], vec![10, 20]);
        let disabled = trailing_unphased_het_mask(&window, 0, &[0], false);
        let enabled = trailing_unphased_het_mask(&window, 0, &[0], true);

        assert_eq!(disabled, vec![false, false]);
        assert_eq!(enabled, vec![false, false]);
    }

    #[test]
    fn trailing_mask_matches_one_two_and_three_het_rules() {
        let one = trailing_het_window(vec![unphased_het()], vec![10]);
        let two = trailing_het_window(vec![unphased_het(), unphased_het()], vec![10, 20]);
        let three = trailing_het_window(
            vec![unphased_het(), unphased_het(), unphased_het()],
            vec![10, 20, 30],
        );
        let distant_three = trailing_het_window(
            vec![unphased_het(), unphased_het(), unphased_het()],
            vec![10, 3_011, 3_012],
        );

        assert_eq!(trailing_unphased_het_mask(&one, 0, &[0], true), vec![false]);
        assert_eq!(
            trailing_unphased_het_mask(&two, 0, &[0], true),
            vec![false, false]
        );
        assert_eq!(
            trailing_unphased_het_mask(&three, 0, &[0], true),
            vec![false, true, false]
        );
        assert_eq!(
            trailing_unphased_het_mask(&distant_three, 0, &[0], true),
            vec![false, true, false]
        );
    }

    #[test]
    fn trailing_mask_persists_until_the_masked_heterozygote_is_phased() {
        let window = trailing_het_window(
            vec![unphased_het(), unphased_het(), unphased_het()],
            vec![10, 20, 30],
        );
        let mut persistent = vec![0u64];
        update_trailing_unphased_het_mask(&window, 0, &[0], true, &mut persistent);
        assert_eq!(persistent, vec![0b010]);

        // The second trailing heterozygote becomes confidently phased. Beagle
        // keeps the previously masked first heterozygote masked across rounds.
        update_trailing_unphased_het_mask(&window, 0, &[1 << 2], true, &mut persistent);
        assert_eq!(persistent, vec![0b010]);

        // Once the masked heterozygote itself is phased, it leaves the mask.
        update_trailing_unphased_het_mask(
            &window,
            0,
            &[(1 << 1) | (1 << 2)],
            true,
            &mut persistent,
        );
        assert_eq!(persistent, vec![0]);
    }

    #[test]
    fn trailing_mask_spans_homozygous_clusters_but_stops_at_phased_het() {
        let homozygous = DiploidCall {
            alleles: [Some(0), Some(0)],
            phased: false,
        };
        let phased_het = DiploidCall {
            alleles: [Some(0), Some(1)],
            phased: true,
        };
        let window = trailing_het_window(
            vec![unphased_het(), homozygous, unphased_het()],
            vec![10, 11, 12],
        );
        assert_eq!(
            trailing_unphased_het_mask(&window, 0, &[0], true),
            vec![false, false, false]
        );

        let bounded = trailing_het_window(
            vec![unphased_het(), phased_het, unphased_het(), unphased_het()],
            vec![10, 11, 12, 13],
        );
        assert_eq!(
            trailing_unphased_het_mask(&bounded, 0, &[0], true),
            vec![false, true, true, false]
        );
    }

    #[test]
    fn low_frequency_scaffold_interpolates_and_extends_endpoint_posteriors() {
        let mut haplotypes = PackedHaplotypes::empty(2, 5, 1).unwrap();
        let alleles = [[0, 0, 1, 1, 1], [1, 1, 0, 0, 0]];
        for marker in 0..5 {
            for state in 0..2 {
                haplotypes
                    .set_allele(marker, state, Some(alleles[state][marker]))
                    .unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1, 2],
            marker_indices: vec![1, 3],
            allele_counts: vec![2; 5],
        };
        let mut observations = PackedHaplotypes::empty(1, 2, 1).unwrap();
        observations.set_allele(0, 0, Some(0)).unwrap();
        observations.set_allele(1, 0, Some(1)).unwrap();
        let sparse_cm = [1.0, 3.0];
        let full_cm = [0.0, 1.0, 2.0, 3.0, 4.0];
        let config = HMMConfig {
            ne: 100,
            recombination_intensity: (0.04_f32 * 100.0) / 4.0,
            err: 0.01,
            em: false,
            burnin: 0,
            iterations: 1,
            initial_lr: 1.0,
            phase_states: 2,
            haplotype_count: 4,
        };
        let mut state_posteriors = vec![Vec::new(); 2];
        crate::imp::hmm::forward_backward_with_state_callback(
            &observations,
            &panel,
            &sparse_cm,
            &config,
            &mut HMMWorkspace::default(),
            |marker, probabilities| {
                state_posteriors[marker] = probabilities.to_vec();
                Ok(())
            },
        )
        .unwrap();

        let posterior = forward_backward_interpolated_alleles(
            &observations,
            &panel,
            &sparse_cm,
            &full_cm,
            &[2; 5],
            &config,
            &mut HMMWorkspace::default(),
        )
        .unwrap();
        let allele1_mass = |marker: usize, states: &[f32]| {
            (0..2)
                .filter(|&state| panel.haplotypes.allele(marker, state) == Some(1))
                .map(|state| states[state])
                .sum::<f32>()
        };

        for marker in 0..5 {
            let expected_alt = match marker {
                0 | 1 => allele1_mass(marker, &state_posteriors[0]),
                2 => {
                    (allele1_mass(marker, &state_posteriors[0])
                        + allele1_mass(marker, &state_posteriors[1]))
                        / 2.0
                }
                3 | 4 => allele1_mass(marker, &state_posteriors[1]),
                _ => unreachable!(),
            };
            let offset = marker * 2;
            assert!((posterior.allele_probabilities[offset + 1] - expected_alt).abs() < 1.0e-6);
            assert!(
                (posterior.allele_probabilities[offset]
                    + posterior.allele_probabilities[offset + 1]
                    - 1.0)
                    .abs()
                    < 1.0e-6
            );
        }
    }
}
