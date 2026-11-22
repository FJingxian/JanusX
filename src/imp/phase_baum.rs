//! Three-track genotype/haplotype phasing update.

use super::types::{
    read_bit_chunk, CompositePanel, DiploidCall, HMMConfig, HMMWorkspace, PackedHaplotypes,
    PhaseWindow, PhasedWindow,
};
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64 as DetailAtomicU64, Ordering as DetailOrdering};
use std::sync::OnceLock;
use std::time::Instant;

static PAIR_PREPARE_NANOS: DetailAtomicU64 = DetailAtomicU64::new(0);
static PAIR_BACKWARD_NANOS: DetailAtomicU64 = DetailAtomicU64::new(0);
static PAIR_FORWARD_NANOS: DetailAtomicU64 = DetailAtomicU64::new(0);
static PREPARE_CLUSTERS_NANOS: DetailAtomicU64 = DetailAtomicU64::new(0);
static PREPARE_TRANSITIONS_NANOS: DetailAtomicU64 = DetailAtomicU64::new(0);
static PREPARE_EMISSIONS_NANOS: DetailAtomicU64 = DetailAtomicU64::new(0);
static PREPARE_WORKSPACE_NANOS: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_SAMPLES: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_CLUSTERS: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_STATES: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_STATE_CLUSTER_UPDATES: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_UNPHASED_EVENTS: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_MISSING_OR_MASKED: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_MISSING_CLUSTERS: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_MASKED_HETEROZYGOTE_CLUSTERS: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_EMISSION_ROWS: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_BETA_CACHE_BYTES: DetailAtomicU64 = DetailAtomicU64::new(0);
static PEAK_BETA_CACHE_BYTES: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_SINGLE_TRACK_SAMPLES: DetailAtomicU64 = DetailAtomicU64::new(0);
static LIVE_BETA_CACHE_BYTES: DetailAtomicU64 = DetailAtomicU64::new(0);
static WINDOW_PEAK_LIVE_BETA_CACHE_BYTES: DetailAtomicU64 = DetailAtomicU64::new(0);
static PEAK_LIVE_BETA_CACHE_BYTES: DetailAtomicU64 = DetailAtomicU64::new(0);
static LIVE_EVENT_WORKSPACE_BYTES: DetailAtomicU64 = DetailAtomicU64::new(0);
static WINDOW_PEAK_LIVE_EVENT_WORKSPACE_BYTES: DetailAtomicU64 = DetailAtomicU64::new(0);
static PEAK_LIVE_EVENT_WORKSPACE_BYTES: DetailAtomicU64 = DetailAtomicU64::new(0);
static COUNT_CHECKPOINTED_PREPARATIONS: DetailAtomicU64 = DetailAtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PhaseBaumMemoryProfile {
    pub(crate) live_beta_cache_bytes: u64,
    pub(crate) window_peak_live_beta_cache_bytes: u64,
    pub(crate) process_peak_live_beta_cache_bytes: u64,
    pub(crate) live_event_workspace_bytes: u64,
    pub(crate) window_peak_live_event_workspace_bytes: u64,
    pub(crate) process_peak_live_event_workspace_bytes: u64,
    pub(crate) checkpointed_preparations: u64,
}

pub(crate) fn reset_window_memory_peaks() {
    WINDOW_PEAK_LIVE_BETA_CACHE_BYTES.store(
        LIVE_BETA_CACHE_BYTES.load(DetailOrdering::Relaxed),
        DetailOrdering::Relaxed,
    );
    WINDOW_PEAK_LIVE_EVENT_WORKSPACE_BYTES.store(
        LIVE_EVENT_WORKSPACE_BYTES.load(DetailOrdering::Relaxed),
        DetailOrdering::Relaxed,
    );
}

pub(crate) fn memory_profile_totals() -> PhaseBaumMemoryProfile {
    PhaseBaumMemoryProfile {
        live_beta_cache_bytes: LIVE_BETA_CACHE_BYTES.load(DetailOrdering::Relaxed),
        window_peak_live_beta_cache_bytes: WINDOW_PEAK_LIVE_BETA_CACHE_BYTES
            .load(DetailOrdering::Relaxed),
        process_peak_live_beta_cache_bytes: PEAK_LIVE_BETA_CACHE_BYTES
            .load(DetailOrdering::Relaxed),
        live_event_workspace_bytes: LIVE_EVENT_WORKSPACE_BYTES.load(DetailOrdering::Relaxed),
        window_peak_live_event_workspace_bytes: WINDOW_PEAK_LIVE_EVENT_WORKSPACE_BYTES
            .load(DetailOrdering::Relaxed),
        process_peak_live_event_workspace_bytes: PEAK_LIVE_EVENT_WORKSPACE_BYTES
            .load(DetailOrdering::Relaxed),
        checkpointed_preparations: COUNT_CHECKPOINTED_PREPARATIONS.load(DetailOrdering::Relaxed),
    }
}

#[inline]
fn update_live_bytes(
    tracked: &mut usize,
    new_bytes: usize,
    live: &DetailAtomicU64,
    window_peak: &DetailAtomicU64,
    process_peak: &DetailAtomicU64,
) {
    let old = *tracked;
    if new_bytes == old {
        return;
    }
    *tracked = new_bytes;
    if new_bytes > old {
        let delta = (new_bytes - old) as u64;
        let current = live.fetch_add(delta, DetailOrdering::Relaxed) + delta;
        window_peak.fetch_max(current, DetailOrdering::Relaxed);
        process_peak.fetch_max(current, DetailOrdering::Relaxed);
    } else {
        let delta = (old - new_bytes) as u64;
        live.fetch_sub(delta, DetailOrdering::Relaxed);
    }
}

#[inline]
fn vec_cap_bytes<T>(vec: &Vec<T>) -> usize {
    vec.capacity().saturating_mul(std::mem::size_of::<T>())
}

fn packed_emission_scratch_bytes(scratch: &PackedPhaseEmissionScratch) -> usize {
    vec_cap_bytes(&scratch.observed)
        .saturating_add(vec_cap_bytes(&scratch.mismatched))
        .saturating_add(
            scratch
                .target_observed
                .iter()
                .map(vec_cap_bytes)
                .fold(0usize, usize::saturating_add),
        )
        .saturating_add(
            scratch
                .target_one
                .iter()
                .map(vec_cap_bytes)
                .fold(0usize, usize::saturating_add),
        )
        .saturating_add(vec_cap_bytes(&scratch.donor_column))
}

const DEFAULT_PHASE_BETA_CACHE_BUDGET_BYTES: usize = 32 * 1024 * 1024;

fn parse_beta_cache_budget_mb(raw: &str) -> Result<Option<usize>, ()> {
    let mb = raw.trim().parse::<usize>().map_err(|_| ())?;
    if mb == 0 {
        Ok(None)
    } else {
        mb.checked_mul(1024 * 1024).map(Some).ok_or(())
    }
}

fn phase_beta_cache_budget_bytes() -> Option<usize> {
    match std::env::var("JX_IMP_PHASE_BETA_BUDGET_MB") {
        Ok(value) => parse_beta_cache_budget_mb(&value)
            .unwrap_or(Some(DEFAULT_PHASE_BETA_CACHE_BUDGET_BYTES)),
        Err(_) => Some(DEFAULT_PHASE_BETA_CACHE_BUDGET_BYTES),
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PairedHmmTraceConfig {
    event_target: Option<(usize, usize)>,
    sample_target: Option<usize>,
    missing_target: Option<(u64, usize, usize)>,
}

fn parse_paired_hmm_event_target(target: &str) -> Option<(usize, usize)> {
    let (sample, marker) = target.split_once(':')?;
    Some((sample.parse().ok()?, marker.parse().ok()?))
}

fn parse_paired_hmm_missing_target(target: &str) -> Option<(u64, usize, usize)> {
    let mut fields = target.split(':');
    let window = fields.next()?.parse().ok()?;
    let sample = fields.next()?.parse().ok()?;
    let marker = fields.next()?.parse().ok()?;
    fields.next().is_none().then_some((window, sample, marker))
}

fn paired_hmm_trace_config() -> &'static PairedHmmTraceConfig {
    static CONFIG: OnceLock<PairedHmmTraceConfig> = OnceLock::new();
    CONFIG.get_or_init(|| PairedHmmTraceConfig {
        event_target: std::env::var("JANUSX_TRACE_PAIRED_HMM_EVENT")
            .ok()
            .and_then(|target| parse_paired_hmm_event_target(&target)),
        sample_target: std::env::var("JANUSX_TRACE_PAIRED_HMM_SAMPLE")
            .ok()
            .and_then(|target| target.parse().ok()),
        missing_target: std::env::var("JANUSX_TRACE_PAIRED_HMM_TARGET")
            .ok()
            .and_then(|target| parse_paired_hmm_missing_target(&target)),
    })
}

pub(crate) fn pair_detail_profile_nanos() -> [u64; 3] {
    [
        PAIR_PREPARE_NANOS.load(DetailOrdering::Relaxed),
        PAIR_BACKWARD_NANOS.load(DetailOrdering::Relaxed),
        PAIR_FORWARD_NANOS.load(DetailOrdering::Relaxed),
    ]
}

pub(crate) fn prepare_detail_profile_nanos() -> [u64; 4] {
    [
        PREPARE_CLUSTERS_NANOS.load(DetailOrdering::Relaxed),
        PREPARE_TRANSITIONS_NANOS.load(DetailOrdering::Relaxed),
        PREPARE_EMISSIONS_NANOS.load(DetailOrdering::Relaxed),
        PREPARE_WORKSPACE_NANOS.load(DetailOrdering::Relaxed),
    ]
}

fn detail_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some()
            || std::env::var_os("JANUSX_IMP_PHASE_DETAIL_PROFILE").is_some()
            || std::env::var_os("JANUSX_IMP_DETAIL_PROFILE").is_some()
    })
}

fn count_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("JANUSX_IMP_COUNT_PROFILE").is_some()
            || std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some()
            || std::env::var_os("JANUSX_IMP_PHASE_PROFILE").is_some()
            || std::env::var_os("JANUSX_IMP_PHASE_DETAIL_PROFILE").is_some()
    })
}

pub(crate) fn count_profile_totals() -> [u64; 12] {
    [
        COUNT_SAMPLES.load(DetailOrdering::Relaxed),
        COUNT_CLUSTERS.load(DetailOrdering::Relaxed),
        COUNT_STATES.load(DetailOrdering::Relaxed),
        COUNT_STATE_CLUSTER_UPDATES.load(DetailOrdering::Relaxed),
        COUNT_UNPHASED_EVENTS.load(DetailOrdering::Relaxed),
        COUNT_MISSING_OR_MASKED.load(DetailOrdering::Relaxed),
        COUNT_MISSING_CLUSTERS.load(DetailOrdering::Relaxed),
        COUNT_MASKED_HETEROZYGOTE_CLUSTERS.load(DetailOrdering::Relaxed),
        COUNT_EMISSION_ROWS.load(DetailOrdering::Relaxed),
        COUNT_BETA_CACHE_BYTES.load(DetailOrdering::Relaxed),
        PEAK_BETA_CACHE_BYTES.load(DetailOrdering::Relaxed),
        COUNT_SINGLE_TRACK_SAMPLES.load(DetailOrdering::Relaxed),
    ]
}

const CHECKPOINT_STRIDE: usize = 256;
const MAX_CLUSTER_CM: f64 = 0.005;
const MAX_CLUSTER_MARKERS: usize = 255;
const FULL_EMISSION_CACHE_BUDGET_BYTES: usize = 8 * 1024 * 1024;
const EMISSION_UNOBSERVED: u8 = 0;
const EMISSION_MATCH: u8 = 1;
const EMISSION_MISMATCH: u8 = 2;
const NO_EVENT_SLOT: usize = usize::MAX;

#[inline(always)]
fn emission_row_count(kind: ClusterKind) -> usize {
    match kind {
        ClusterKind::Homozygous => 1,
        ClusterKind::Heterozygous { .. } => 2,
        ClusterKind::MaskedHeterozygote | ClusterKind::Missing => 0,
    }
}

#[inline(always)]
fn emission_row_offset(
    kind: ClusterKind,
    cluster_offset: usize,
    state_count: usize,
    track: usize,
) -> Option<usize> {
    match kind {
        ClusterKind::Homozygous => Some(cluster_offset),
        ClusterKind::Heterozygous { .. } if track > 0 => {
            Some(cluster_offset + (track - 1) * state_count)
        }
        ClusterKind::Heterozygous { .. }
        | ClusterKind::MaskedHeterozygote
        | ClusterKind::Missing => None,
    }
}

#[derive(Debug)]
pub(crate) struct JointPhaseUpdate {
    pub(crate) calls: Vec<DiploidCall>,
    pub(crate) changes: Vec<(usize, DiploidCall)>,
    pub(crate) phase_confidences: Vec<(usize, f64)>,
    pub(crate) unphased_heterozygote_count: usize,
    pub(crate) phase_swap_count: usize,
}

#[derive(Debug)]
pub(crate) struct PhaseBaumDiagnostic {
    pub(crate) marker: usize,
    pub(crate) cluster_kind: Option<&'static str>,
    pub(crate) transition_probability: f32,
    pub(crate) state_alleles: Vec<Option<u16>>,
    pub(crate) emission_probabilities: Vec<Vec<f32>>,
    pub(crate) forward_probabilities: Vec<Vec<f32>>,
    pub(crate) backward_probabilities: Vec<Vec<f32>>,
    pub(crate) posterior_normalization: Vec<f64>,
    pub(crate) phase_swap: Option<bool>,
    pub(crate) phase_log_odds: Option<f64>,
    pub(crate) selected_alleles: [Option<u16>; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClusterKind {
    Homozygous,
    Heterozygous { unphased: bool },
    MaskedHeterozygote,
    Missing,
}

#[derive(Debug, Clone, Copy)]
struct PhaseCluster {
    start: usize,
    end: usize,
    kind: ClusterKind,
}

#[derive(Debug, Clone, Copy)]
enum PhaseClusterTemplateKind {
    Homozygous,
    Heterozygous,
    Missing,
}

#[derive(Debug, Default)]
pub(crate) struct PhaseClusterTemplate {
    sample: usize,
    sizes: Vec<u8>,
    kinds: Vec<PhaseClusterTemplateKind>,
}

#[derive(Default)]
struct PackedPhaseEmissionScratch {
    observed: Vec<[u64; 3]>,
    mismatched: Vec<[u64; 3]>,
    target_observed: [Vec<u64>; 3],
    target_one: [Vec<u64>; 3],
    donor_column: Vec<u64>,
}

#[derive(Default)]
pub(crate) struct PhaseBaumWorkspace {
    clusters: Vec<PhaseCluster>,
    transitions: Vec<f32>,
    emission_codes: Vec<u8>,
    emission_code_offsets: Vec<usize>,
    emission_probabilities: Vec<[f32; 2]>,
    packed_emission_scratch: PackedPhaseEmissionScratch,
    slots: Vec<usize>,
    beta_cache: Vec<MaybeUninit<f32>>,
    beta_checkpoints: Vec<MaybeUninit<f32>>,
    beta_local: Vec<f32>,
    beta_event_input: Vec<f32>,
    beta_event: Vec<f32>,
    alpha: Vec<f32>,
    tracked_beta_bytes: usize,
    tracked_event_workspace_bytes: usize,
    tracked_scratch_bytes: usize,
    pub(crate) low_frequency_hmm: [HMMWorkspace; 2],
}

impl Drop for PhaseBaumWorkspace {
    fn drop(&mut self) {
        update_live_bytes(
            &mut self.tracked_beta_bytes,
            0,
            &LIVE_BETA_CACHE_BYTES,
            &WINDOW_PEAK_LIVE_BETA_CACHE_BYTES,
            &PEAK_LIVE_BETA_CACHE_BYTES,
        );
        update_live_bytes(
            &mut self.tracked_event_workspace_bytes,
            0,
            &LIVE_EVENT_WORKSPACE_BYTES,
            &WINDOW_PEAK_LIVE_EVENT_WORKSPACE_BYTES,
            &PEAK_LIVE_EVENT_WORKSPACE_BYTES,
        );
        update_live_bytes(
            &mut self.tracked_scratch_bytes,
            0,
            &LIVE_EVENT_WORKSPACE_BYTES,
            &WINDOW_PEAK_LIVE_EVENT_WORKSPACE_BYTES,
            &PEAK_LIVE_EVENT_WORKSPACE_BYTES,
        );
    }
}

/// Compact per-sample PhaseBaum workspace. Beagle stores mismatch rows and
/// derives emissions in both HMM passes; this stores the equivalent three
/// emission codes per state and retains only event posterior vectors.
struct PreparedEventSample {
    sample: usize,
    trace_config: PairedHmmTraceConfig,
    avx2: bool,
    single_track: bool,
    checkpointed: bool,
    complete_emissions: bool,
    clusters: Vec<PhaseCluster>,
    transitions: Vec<f32>,
    emission_codes: Vec<u8>,
    emission_code_offsets: Vec<usize>,
    emission_probabilities: Vec<[f32; 2]>,
    slots: Vec<usize>,
    beta_cache: Vec<MaybeUninit<f32>>,
    beta_checkpoints: Vec<MaybeUninit<f32>>,
    beta_local: Vec<f32>,
    beta_event_input: Vec<f32>,
    beta_event: Vec<f32>,
    backward_cursor: usize,
    backward_active: bool,
    alpha: Vec<f32>,
    forward_sums: [f32; 3],
    changes: Vec<(usize, DiploidCall)>,
    phase_confidences: Vec<(usize, f64)>,
    swap_haps: bool,
    phase_swap_count: usize,
    unphased_heterozygote_count: usize,
    forward_cursor: usize,
    forward_active: bool,
    tracked_beta_bytes: usize,
    tracked_event_workspace_bytes: usize,
}

impl Drop for PreparedEventSample {
    fn drop(&mut self) {
        update_live_bytes(
            &mut self.tracked_beta_bytes,
            0,
            &LIVE_BETA_CACHE_BYTES,
            &WINDOW_PEAK_LIVE_BETA_CACHE_BYTES,
            &PEAK_LIVE_BETA_CACHE_BYTES,
        );
        update_live_bytes(
            &mut self.tracked_event_workspace_bytes,
            0,
            &LIVE_EVENT_WORKSPACE_BYTES,
            &WINDOW_PEAK_LIVE_EVENT_WORKSPACE_BYTES,
            &PEAK_LIVE_EVENT_WORKSPACE_BYTES,
        );
    }
}

/// A slash-separated heterozygote in the new part of a window activates
/// phasing. Once active, Beagle's initializer treats later heterozygotes as
/// unresolved events; a fully phased window keeps its supplied orientation.
pub(crate) fn has_initial_unphased_heterozygote(input: &PhaseWindow) -> bool {
    input
        .rows
        .iter()
        .skip(input.incoming_overlap_len)
        .flat_map(|row| row.calls.iter())
        .any(|call| call.is_heterozygous() && !call.phased)
}

pub(crate) fn sample_requires_phase_baum(
    input: &PhaseWindow,
    sample: usize,
    finished: &[u64],
    any_unphased_heterozygote: bool,
) -> bool {
    input.rows.iter().enumerate().any(|(marker, row)| {
        let call = row
            .calls
            .get(sample)
            .expect("sample is in the phase window");
        match call.alleles {
            [Some(left), Some(right)] if left == right || call.phased => false,
            [Some(_), Some(_)] => {
                any_unphased_heterozygote
                    && marker >= input.incoming_overlap_len
                    && !bit_is_set(finished, marker)
            }
            _ => true,
        }
    })
}

/// Update one sample against a single shared donor panel. The genotype track
/// carries homozygous evidence; the haplotype tracks carry oriented allele
/// evidence and are reinitialized from the genotype track at each unresolved
/// heterozygote. Backward checkpoints are retained per 256 clusters, while
/// only allele-level hard calls survive the window pass.
pub(crate) fn update_sample(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    finished: &[u64],
    any_unphased_heterozygote: bool,
    panel: &CompositePanel,
    config: &HMMConfig,
) -> Result<JointPhaseUpdate, String> {
    let unmasked = vec![0u64; input.rows.len().div_ceil(64)];
    update_sample_with_diagnostic(
        input,
        snapshot,
        sample,
        finished,
        any_unphased_heterozygote,
        &unmasked,
        panel,
        config,
        None,
        |_| {},
    )
}

pub(crate) fn update_sample_with_diagnostic<F>(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    finished: &[u64],
    any_unphased_heterozygote: bool,
    masked_heterozygotes: &[u64],
    panel: &CompositePanel,
    config: &HMMConfig,
    diagnostic_marker: Option<usize>,
    diagnostic_callback: F,
) -> Result<JointPhaseUpdate, String>
where
    F: FnMut(PhaseBaumDiagnostic),
{
    let mut workspace = PhaseBaumWorkspace::default();
    update_sample_with_diagnostic_and_workspace(
        input,
        snapshot,
        sample,
        finished,
        any_unphased_heterozygote,
        masked_heterozygotes,
        panel,
        config,
        diagnostic_marker,
        &mut workspace,
        diagnostic_callback,
    )
}

pub(crate) fn update_sample_with_diagnostic_and_workspace<F>(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    finished: &[u64],
    any_unphased_heterozygote: bool,
    masked_heterozygotes: &[u64],
    panel: &CompositePanel,
    config: &HMMConfig,
    diagnostic_marker: Option<usize>,
    workspace: &mut PhaseBaumWorkspace,
    mut diagnostic_callback: F,
) -> Result<JointPhaseUpdate, String>
where
    F: FnMut(PhaseBaumDiagnostic),
{
    let marker_count = input.rows.len();
    let state_count = panel.haplotypes.n_haplotypes;
    if marker_count == 0
        || snapshot.rows.len() != marker_count
        || input.cm_positions.len() != marker_count
        || panel.haplotypes.n_markers != marker_count
        || panel.marker_indices.len() != marker_count
        || panel.allele_counts.len() != marker_count
        || state_count == 0
        || config.ne == 0
        || config.haplotype_count == 0
        || !config.err.is_finite()
        || !(0.0..0.5).contains(&config.err)
    {
        return Err("joint phase HMM dimensions or parameters are invalid".to_owned());
    }
    let sample_count = input.rows[0].calls.len();
    if sample >= sample_count
        || snapshot
            .rows
            .iter()
            .any(|row| row.calls.len() != sample_count)
        || input.rows.iter().any(|row| row.calls.len() != sample_count)
        || finished.len() < marker_count.div_ceil(64)
    {
        return Err("joint phase HMM sample or progressive-state dimensions differ".to_owned());
    }
    if input.cm_positions.iter().any(|cm| !cm.is_finite())
        || input.cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("joint phase HMM map must be finite and nondecreasing".to_owned());
    }

    if masked_heterozygotes.len() != marker_count.div_ceil(64) {
        return Err("joint phase HMM trailing-mask dimensions differ".to_owned());
    }
    let clusters = build_clusters(
        input,
        sample,
        finished,
        any_unphased_heterozygote,
        masked_heterozygotes,
    );
    if clusters.is_empty() {
        return Err("joint phase HMM has no marker clusters".to_owned());
    }
    let unphased_heterozygote_count = clusters
        .iter()
        .filter(|&&cluster| is_unphased_event(cluster))
        .count();
    let cluster_count = clusters.len();
    let mut transitions = vec![0.0f32; cluster_count];
    for cluster in 1..cluster_count {
        let previous_marker = clusters[cluster - 1].end - 1;
        let current_marker = clusters[cluster].end - 1;
        transitions[cluster] = cluster_transition_probability(
            &input.cm_positions,
            previous_marker,
            current_marker,
            state_count,
            config.recombination_intensity,
        );
    }
    if diagnostic_marker.is_none() && panel.haplotypes.is_state_major() {
        return update_sample_event_coded(
            input,
            snapshot,
            sample,
            panel,
            config,
            clusters,
            transitions,
            workspace,
        );
    }
    if diagnostic_marker.is_some()
        && std::env::var("JANUSX_TRACE_HMM_DETAIL_TARGET")
            .is_ok_and(|target| target == format!("{}:{}", input.index, sample))
    {
        if let Ok(prefix) = std::env::var("JANUSX_TRACE_HMM_DETAIL_PREFIX") {
            use std::fmt::Write as _;
            let mut contents = String::new();
            writeln!(
                &mut contents,
                "CLUSTER\tSTART_POS\tEND_POS\tKIND\tTRANSITION\tEMISSION0\tEMISSION1\tEMISSION2"
            )
            .unwrap();
            for (index, &cluster) in clusters.iter().enumerate() {
                write!(
                    &mut contents,
                    "{}\t{}\t{}\t{:?}\t{}",
                    index,
                    input.rows[cluster.start].meta.pos,
                    input.rows[cluster.end - 1].meta.pos,
                    cluster.kind,
                    transitions[index]
                )
                .unwrap();
                for track in 0..3 {
                    contents.push('\t');
                    for state in 0..state_count {
                        if state > 0 {
                            contents.push(',');
                        }
                        write!(
                            &mut contents,
                            "{}",
                            cluster_emission(
                                cluster, track, false, state, input, snapshot, sample, panel,
                                config.err,
                            )
                        )
                        .unwrap();
                    }
                }
                contents.push('\n');
            }
            let path = format!("{prefix}.window{}.sample{}.tsv", input.index, sample);
            if !std::path::Path::new(&path).exists() {
                std::fs::write(path, contents).map_err(|error| {
                    format!("cannot write temporary HMM cluster trace: {error}")
                })?;
            }
        }
    }

    let block_count = cluster_count.div_ceil(CHECKPOINT_STRIDE);
    let track_values = state_count
        .checked_mul(3)
        .ok_or_else(|| "joint HMM state vector size overflow".to_owned())?;
    let checkpoint_len = block_count
        .checked_mul(track_values)
        .ok_or_else(|| "joint HMM checkpoint size overflow".to_owned())?;
    let mut checkpoints = vec![0.0f32; checkpoint_len];
    // A single cluster's three emission vectors are enough for both the
    // forward and backward recurrences; avoid caching emissions per window.
    let mut emission_scratch = vec![0.0f32; track_values];
    // Per-state mismatch/observed flags let emission filling walk markers
    // first, keeping each packed donor marker row hot in cache.
    let mut mismatch_scratch = vec![[0u8; 3]; state_count];
    let mut packed_emission_scratch = PackedPhaseEmissionScratch::default();
    let mut beta_local = vec![1.0f32 / state_count as f32; track_values];
    let mut beta_carry = beta_local.clone();
    let mut beta_projected = vec![0.0f32; track_values];
    let trace_backward_prefix = if diagnostic_marker.is_some()
        && std::env::var("JANUSX_TRACE_BWD_TARGET")
            .is_ok_and(|target| target == format!("{}:{}", input.index, sample))
    {
        std::env::var("JANUSX_TRACE_BWD_PREFIX").ok()
    } else {
        None
    };
    let mut backward_trace = String::new();
    if trace_backward_prefix.is_some() {
        use std::fmt::Write as _;
        writeln!(&mut backward_trace, "CLUSTER\tSTAGE\tBETA0\tBETA1\tBETA2").unwrap();
        write_beta_trace_row(
            &mut backward_trace,
            cluster_count - 1,
            "INIT",
            &beta_local,
            state_count,
        );
    }
    let full_emission_values = cluster_count
        .checked_mul(track_values)
        .ok_or_else(|| "joint HMM emission cache size overflow".to_owned())?;
    let cache_budget = std::env::var("JX_IMP_PHASE_EMISSION_CACHE_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(FULL_EMISSION_CACHE_BUDGET_BYTES);
    let use_full_emission_cache = panel.haplotypes.is_state_major()
        && full_emission_values
            .checked_mul(std::mem::size_of::<f32>())
            .is_some_and(|bytes| bytes <= cache_budget);
    let mut full_emission_cache =
        use_full_emission_cache.then(|| vec![0.0f32; full_emission_values]);

    // Store only the backward vector at each block's right edge. An event
    // posterior or a missing-call posterior is reconstructed within a block.
    for cluster in (0..cluster_count).rev() {
        let block = cluster / CHECKPOINT_STRIDE;
        let block_end = ((block + 1) * CHECKPOINT_STRIDE).min(cluster_count) - 1;
        if cluster == block_end {
            let offset = block * track_values;
            checkpoints[offset..offset + track_values].copy_from_slice(&beta_local);
        }
        if cluster > 0 {
            if let Some(cache) = full_emission_cache.as_mut() {
                let offset = cluster * track_values;
                let cached_emissions = &mut cache[offset..offset + track_values];
                fill_cluster_emissions(
                    cached_emissions,
                    clusters[cluster],
                    false,
                    input,
                    snapshot,
                    sample,
                    panel,
                    config.err,
                    &mut mismatch_scratch,
                    &mut packed_emission_scratch,
                );
                backward_project_from_emissions(
                    &beta_carry,
                    &mut beta_projected,
                    cached_emissions,
                    transitions[cluster],
                );
            } else {
                backward_project(
                    &beta_carry,
                    &mut beta_projected,
                    clusters[cluster],
                    transitions[cluster],
                    input,
                    snapshot,
                    sample,
                    panel,
                    config.err,
                    &mut emission_scratch,
                    &mut mismatch_scratch,
                    &mut packed_emission_scratch,
                );
            }
            std::mem::swap(&mut beta_local, &mut beta_projected);
            beta_carry.copy_from_slice(&beta_local);
            if is_unphased_event(clusters[cluster]) {
                copy_track(&mut beta_carry, 0, 1, state_count);
                copy_track(&mut beta_carry, 0, 2, state_count);
            }
        }
    }
    if let Some(cache) = full_emission_cache.as_mut() {
        fill_cluster_emissions(
            &mut cache[..track_values],
            clusters[0],
            false,
            input,
            snapshot,
            sample,
            panel,
            config.err,
            &mut mismatch_scratch,
            &mut packed_emission_scratch,
        );
    }

    let mut alpha = vec![1.0 / state_count as f32; track_values];
    let mut forward_sums = [1.0f32; 3];
    let mut beta_block = Vec::<f32>::new();
    // Backward and forward otherwise rebuild the same non-event emissions in
    // each checkpoint block. Reuse them and swap haplotype tracks for phase
    // orientation during the forward pass.
    let mut emission_block = Vec::<f32>::new();
    let mut beta_event = vec![0.0f32; track_values];
    let mut projected_scratch = vec![0.0f32; track_values];
    let mut output = snapshot
        .rows
        .iter()
        .map(|row| row.calls.get(sample).unwrap())
        .collect::<Vec<_>>();
    let mut phase_confidences = Vec::new();
    let mut swap_haps = false;
    let mut phase_swap_count = 0usize;

    for block in 0..block_count {
        let start = block * CHECKPOINT_STRIDE;
        let end = ((block + 1) * CHECKPOINT_STRIDE).min(cluster_count);
        let block_len = end - start;
        beta_block.resize(block_len * track_values, 0.0);
        emission_block.resize(block_len * track_values, 0.0);
        let checkpoint_offset = block * track_values;
        beta_local
            .copy_from_slice(&checkpoints[checkpoint_offset..checkpoint_offset + track_values]);
        beta_carry.copy_from_slice(&beta_local);
        if end < cluster_count && is_unphased_event(clusters[end]) {
            copy_track(&mut beta_carry, 0, 1, state_count);
            copy_track(&mut beta_carry, 0, 2, state_count);
        }
        for cluster in (start..end).rev() {
            let local = cluster - start;
            let offset = local * track_values;
            beta_block[offset..offset + track_values].copy_from_slice(&beta_local);
            if full_emission_cache.is_none() {
                let cached_emissions = &mut emission_block[offset..offset + track_values];
                fill_cluster_emissions(
                    cached_emissions,
                    clusters[cluster],
                    false,
                    input,
                    snapshot,
                    sample,
                    panel,
                    config.err,
                    &mut mismatch_scratch,
                    &mut packed_emission_scratch,
                );
            }
            if cluster > start {
                if let Some(cache) = full_emission_cache.as_ref() {
                    let cache_offset = cluster * track_values;
                    backward_project_from_emissions(
                        &beta_carry,
                        &mut projected_scratch,
                        &cache[cache_offset..cache_offset + track_values],
                        transitions[cluster],
                    );
                } else {
                    let cached_emissions = &emission_block[offset..offset + track_values];
                    backward_project_from_emissions(
                        &beta_carry,
                        &mut projected_scratch,
                        cached_emissions,
                        transitions[cluster],
                    );
                }
                beta_local.copy_from_slice(&projected_scratch);
                beta_carry.copy_from_slice(&beta_local);
                if is_unphased_event(clusters[cluster]) {
                    if trace_backward_prefix.is_some() {
                        write_beta_trace_row(
                            &mut backward_trace,
                            cluster - 1,
                            "BEFORE_EVENT_RESET",
                            &beta_carry,
                            state_count,
                        );
                    }
                    copy_track(&mut beta_carry, 0, 1, state_count);
                    copy_track(&mut beta_carry, 0, 2, state_count);
                    if trace_backward_prefix.is_some() {
                        write_beta_trace_row(
                            &mut backward_trace,
                            cluster - 1,
                            "AFTER_EVENT_RESET",
                            &beta_carry,
                            state_count,
                        );
                    }
                } else if trace_backward_prefix.is_some() {
                    write_beta_trace_row(
                        &mut backward_trace,
                        cluster - 1,
                        "AFTER_STEP",
                        &beta_carry,
                        state_count,
                    );
                }
            }
        }

        if block + 1 == block_count {
            if let Some(prefix) = &trace_backward_prefix {
                let path = format!("{prefix}.window{}.sample{}.tsv", input.index, sample);
                if !std::path::Path::new(&path).exists() {
                    std::fs::write(path, &backward_trace).map_err(|error| {
                        format!("cannot write temporary HMM backward trace: {error}")
                    })?;
                }
            }
        }

        for cluster_index in start..end {
            let cluster = clusters[cluster_index];
            let beta_offset = (cluster_index - start) * track_values;
            let current_beta = &beta_block[beta_offset..beta_offset + track_values];
            let is_event = is_unphased_event(cluster);
            let mut confidence = None;
            let mut selected_phase_swap = None;
            let mut selected_phase_log_odds = None;
            if is_event {
                projected_scratch.copy_from_slice(current_beta);
                if cluster_index + 1 < cluster_count
                    && is_unphased_event(clusters[cluster_index + 1])
                {
                    copy_track(&mut projected_scratch, 0, 1, state_count);
                    copy_track(&mut projected_scratch, 0, 2, state_count);
                }
                if let Some(cache) = full_emission_cache.as_ref() {
                    let offset = cluster_index * track_values;
                    backward_project_from_emissions(
                        &projected_scratch,
                        &mut beta_event,
                        &cache[offset..offset + track_values],
                        transitions[cluster_index],
                    );
                } else {
                    backward_project(
                        &projected_scratch,
                        &mut beta_event,
                        cluster,
                        transitions[cluster_index],
                        input,
                        snapshot,
                        sample,
                        panel,
                        config.err,
                        &mut emission_scratch,
                        &mut mismatch_scratch,
                        &mut packed_emission_scratch,
                    );
                }
                let (swap, log_odds) = if cluster_index == 0 {
                    choose_phase_swap(
                        &alpha[state_count..2 * state_count],
                        &alpha[2 * state_count..3 * state_count],
                        &beta_event[state_count..2 * state_count],
                        &beta_event[2 * state_count..3 * state_count],
                    )?
                } else {
                    choose_phase_swap(
                        &alpha[state_count..2 * state_count],
                        &alpha[2 * state_count..3 * state_count],
                        &beta_event[state_count..2 * state_count],
                        &beta_event[2 * state_count..3 * state_count],
                    )?
                };
                if swap_haps != swap {
                    phase_swap_count += 1;
                }
                swap_haps = swap;
                confidence = Some(log_odds);
                selected_phase_swap = Some(swap);
                selected_phase_log_odds = Some(log_odds);
                alpha.copy_within(0..state_count, state_count);
                alpha.copy_within(0..state_count, 2 * state_count);
                forward_sums[1] = forward_sums[0];
                forward_sums[2] = forward_sums[0];
            }

            let cached_emissions = if let Some(cache) = full_emission_cache.as_ref() {
                let offset = cluster_index * track_values;
                &cache[offset..offset + track_values]
            } else {
                &emission_block[(cluster_index - start) * track_values
                    ..(cluster_index - start + 1) * track_values]
            };
            for track in 0..3 {
                let begin = track * state_count;
                let finish = begin + state_count;
                let source_track = if swap_haps && track > 0 {
                    3 - track
                } else {
                    track
                };
                let source_begin = source_track * state_count;
                let source_finish = source_begin + state_count;
                let track_emissions = &cached_emissions[source_begin..source_finish];
                forward_sums[track] = beagle_forward_update(
                    &mut alpha[begin..finish],
                    forward_sums[track],
                    transitions[cluster_index],
                    |state| track_emissions[state],
                );
            }
            if let Some(log_odds) = confidence {
                let marker = cluster.start;
                let snapshot_call = snapshot.rows[marker].calls.get(sample).unwrap();
                let [Some(a), Some(b)] = snapshot_call.alleles else {
                    return Err("unphased phase event has an incomplete genotype".to_owned());
                };
                output[marker] = DiploidCall {
                    alleles: if swap_haps {
                        [Some(b), Some(a)]
                    } else {
                        [Some(a), Some(b)]
                    },
                    phased: true,
                };
                phase_confidences.push((marker, log_odds));
            } else if cluster.kind == ClusterKind::MaskedHeterozygote {
                let marker = cluster.start;
                let beta0 = if swap_haps { 2 } else { 1 };
                let beta1 = if swap_haps { 1 } else { 2 };
                let allele_count = panel.allele_counts[marker].max(1);
                let (probabilities0, probabilities1) = hap_allele_posteriors(
                    &alpha[state_count..2 * state_count],
                    &current_beta[beta0 * state_count..(beta0 + 1) * state_count],
                    &alpha[2 * state_count..3 * state_count],
                    &current_beta[beta1 * state_count..(beta1 + 1) * state_count],
                    panel,
                    marker,
                    allele_count,
                );
                let [Some(mut a), Some(mut b)] =
                    snapshot.rows[marker].calls.get(sample).unwrap().alleles
                else {
                    return Err("masked phase event has an incomplete genotype".to_owned());
                };
                // Beagle stores the current orientation in SamplePhase. The
                // unphased input GT may have the opposite allele order.
                if swap_haps {
                    std::mem::swap(&mut a, &mut b);
                }
                let no_swap =
                    allele_probability(&probabilities0, a) * allele_probability(&probabilities1, b);
                let swap =
                    allele_probability(&probabilities0, b) * allele_probability(&probabilities1, a);
                let selected_swap = swap > no_swap;
                let no_swap_f64 = f64::from(no_swap);
                let swap_f64 = f64::from(swap);
                let smaller = no_swap_f64.min(swap_f64);
                let log_odds = if smaller > f64::MIN_POSITIVE {
                    (no_swap_f64.max(swap_f64) / smaller).ln()
                } else if no_swap_f64.max(swap_f64) > 0.0 {
                    f64::INFINITY
                } else {
                    // Beagle compares float path products directly. A 0 vs 0
                    // score therefore satisfies either threshold inequality
                    // once burn-in ends; represent that outcome as pass.
                    f64::INFINITY
                };
                output[marker] = DiploidCall {
                    alleles: if selected_swap {
                        [Some(b), Some(a)]
                    } else {
                        [Some(a), Some(b)]
                    },
                    phased: true,
                };
                phase_confidences.push((marker, log_odds));
                selected_phase_swap = Some(selected_swap);
                selected_phase_log_odds = Some(log_odds);
            } else if let ClusterKind::Heterozygous { .. } = cluster.kind {
                let marker = cluster.start;
                let snapshot_call = snapshot.rows[marker].calls.get(sample).unwrap();
                let mut alleles = snapshot_call.alleles;
                if swap_haps {
                    alleles.swap(0, 1);
                }
                output[marker] = DiploidCall {
                    alleles,
                    phased: true,
                };
            } else if cluster.kind == ClusterKind::Homozygous {
                for marker in cluster.start..cluster.end {
                    let input_call = input.rows[marker].calls.get(sample).unwrap();
                    output[marker] = DiploidCall {
                        alleles: input_call.alleles,
                        phased: true,
                    };
                }
            } else {
                let marker = cluster.start;
                let beta0 = if swap_haps { 2 } else { 1 };
                let beta1 = if swap_haps { 1 } else { 2 };
                let allele_count = panel.allele_counts[marker].max(1);
                let (probabilities0, probabilities1) = hap_allele_posteriors(
                    &alpha[state_count..2 * state_count],
                    &current_beta[beta0 * state_count..(beta0 + 1) * state_count],
                    &alpha[2 * state_count..3 * state_count],
                    &current_beta[beta1 * state_count..(beta1 + 1) * state_count],
                    panel,
                    marker,
                    allele_count,
                );
                let input_call = input.rows[marker].calls.get(sample).unwrap();
                let allele0 = modal_allele(&probabilities0, input, marker);
                let allele1 = modal_allele(&probabilities1, input, marker);
                let alleles = match input_call.alleles {
                    [Some(a), Some(b)] => [Some(a), Some(b)],
                    [Some(a), None] => [Some(a), Some(allele1)],
                    [None, Some(b)] if input_call.phased => [Some(allele0), Some(b)],
                    [None, Some(b)] => [Some(b), Some(allele1)],
                    [None, None] => [Some(allele0), Some(allele1)],
                };
                output[marker] = DiploidCall {
                    alleles,
                    phased: true,
                };
            }

            if let Some(marker) =
                diagnostic_marker.filter(|&marker| cluster.start <= marker && marker < cluster.end)
            {
                fill_cluster_emissions(
                    &mut emission_scratch,
                    cluster,
                    swap_haps,
                    input,
                    snapshot,
                    sample,
                    panel,
                    config.err,
                    &mut mismatch_scratch,
                    &mut packed_emission_scratch,
                );
                let emission_probabilities = (0..3)
                    .map(|track| {
                        let start = track * state_count;
                        emission_scratch[start..start + state_count].to_vec()
                    })
                    .collect::<Vec<_>>();
                let forward_probabilities = (0..3)
                    .map(|track| alpha[track * state_count..(track + 1) * state_count].to_vec())
                    .collect::<Vec<_>>();
                let backward_probabilities = (0..3)
                    .map(|track| {
                        current_beta[track * state_count..(track + 1) * state_count].to_vec()
                    })
                    .collect::<Vec<_>>();
                let posterior_normalization = (0..3)
                    .map(|track| {
                        forward_probabilities[track]
                            .iter()
                            .zip(&backward_probabilities[track])
                            .map(|(&forward, &backward)| f64::from(forward) * f64::from(backward))
                            .sum::<f64>()
                    })
                    .collect();
                diagnostic_callback(PhaseBaumDiagnostic {
                    marker,
                    cluster_kind: Some(cluster_kind_name(cluster.kind)),
                    transition_probability: transitions[cluster_index],
                    state_alleles: (0..state_count)
                        .map(|state| panel.haplotypes.allele(marker, state))
                        .collect(),
                    emission_probabilities,
                    forward_probabilities,
                    backward_probabilities,
                    posterior_normalization,
                    phase_swap: selected_phase_swap,
                    phase_log_odds: selected_phase_log_odds,
                    selected_alleles: output[marker].alleles,
                });
            }
        }
    }

    Ok(JointPhaseUpdate {
        calls: output,
        changes: Vec::new(),
        phase_confidences,
        unphased_heterozygote_count,
        phase_swap_count,
    })
}

#[allow(clippy::too_many_arguments)]
fn update_sample_event_coded(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    panel: &CompositePanel,
    config: &HMMConfig,
    clusters: Vec<PhaseCluster>,
    transitions: Vec<f32>,
    workspace: &mut PhaseBaumWorkspace,
) -> Result<JointPhaseUpdate, String> {
    let mut run = prepare_event_sample_from_clusters(
        input,
        snapshot,
        sample,
        clusters,
        transitions,
        panel,
        config,
        None,
        workspace,
        false,
    )?;
    run_backward_single(&mut run);
    run_forward_single(&mut run, input, snapshot, panel)?;
    recycle_event_workspace(workspace, &mut run);
    Ok(finish_event_update(run, snapshot, true))
}

fn finish_event_update(
    mut run: PreparedEventSample,
    snapshot: &PhasedWindow,
    materialize_calls: bool,
) -> JointPhaseUpdate {
    let calls = if materialize_calls {
        let mut calls = snapshot
            .rows
            .iter()
            .map(|row| {
                let mut call = row.calls.get(run.sample).unwrap();
                if matches!(call.alleles, [Some(left), Some(right)] if left == right) {
                    call.phased = true;
                }
                call
            })
            .collect::<Vec<_>>();
        for &(marker, call) in &run.changes {
            calls[marker] = call;
        }
        calls
    } else {
        Vec::new()
    };
    JointPhaseUpdate {
        calls,
        changes: std::mem::take(&mut run.changes),
        phase_confidences: std::mem::take(&mut run.phase_confidences),
        unphased_heterozygote_count: run.unphased_heterozygote_count,
        phase_swap_count: run.phase_swap_count,
    }
}

fn recycle_event_workspace(workspace: &mut PhaseBaumWorkspace, run: &mut PreparedEventSample) {
    workspace.clusters = std::mem::take(&mut run.clusters);
    workspace.transitions = std::mem::take(&mut run.transitions);
    workspace.emission_codes = std::mem::take(&mut run.emission_codes);
    workspace.emission_code_offsets = std::mem::take(&mut run.emission_code_offsets);
    workspace.emission_probabilities = std::mem::take(&mut run.emission_probabilities);
    workspace.slots = std::mem::take(&mut run.slots);
    workspace.beta_cache = std::mem::take(&mut run.beta_cache);
    workspace.beta_checkpoints = std::mem::take(&mut run.beta_checkpoints);
    workspace.beta_local = std::mem::take(&mut run.beta_local);
    workspace.beta_event_input = std::mem::take(&mut run.beta_event_input);
    workspace.beta_event = std::mem::take(&mut run.beta_event);
    workspace.alpha = std::mem::take(&mut run.alpha);
    workspace.tracked_beta_bytes = std::mem::replace(&mut run.tracked_beta_bytes, 0);
    workspace.tracked_event_workspace_bytes =
        std::mem::replace(&mut run.tracked_event_workspace_bytes, 0);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_sample_pair(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample0: usize,
    sample1: usize,
    cluster_template0: &PhaseClusterTemplate,
    cluster_template1: &PhaseClusterTemplate,
    finished0: &[u64],
    finished1: &[u64],
    any_unphased_heterozygote: bool,
    masked_heterozygotes0: &[u64],
    masked_heterozygotes1: &[u64],
    panel0: &CompositePanel,
    panel1: &CompositePanel,
    packed_target: Option<&PackedHaplotypes>,
    transition_factors: &[f32],
    config: &HMMConfig,
    workspace0: &mut PhaseBaumWorkspace,
    workspace1: &mut PhaseBaumWorkspace,
) -> Result<(JointPhaseUpdate, JointPhaseUpdate), String> {
    update_sample_pair_impl(
        input,
        snapshot,
        sample0,
        sample1,
        cluster_template0,
        cluster_template1,
        finished0,
        finished1,
        any_unphased_heterozygote,
        masked_heterozygotes0,
        masked_heterozygotes1,
        panel0,
        panel1,
        packed_target,
        transition_factors,
        config,
        workspace0,
        workspace1,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn update_sample_pair_impl(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample0: usize,
    sample1: usize,
    cluster_template0: &PhaseClusterTemplate,
    cluster_template1: &PhaseClusterTemplate,
    finished0: &[u64],
    finished1: &[u64],
    any_unphased_heterozygote: bool,
    masked_heterozygotes0: &[u64],
    masked_heterozygotes1: &[u64],
    panel0: &CompositePanel,
    panel1: &CompositePanel,
    packed_target: Option<&PackedHaplotypes>,
    transition_factors: &[f32],
    config: &HMMConfig,
    workspace0: &mut PhaseBaumWorkspace,
    workspace1: &mut PhaseBaumWorkspace,
    collapse_equivalent_tracks: bool,
) -> Result<(JointPhaseUpdate, JointPhaseUpdate), String> {
    update_sample_pair_impl_with_budget(
        input,
        snapshot,
        sample0,
        sample1,
        cluster_template0,
        cluster_template1,
        finished0,
        finished1,
        any_unphased_heterozygote,
        masked_heterozygotes0,
        masked_heterozygotes1,
        panel0,
        panel1,
        packed_target,
        transition_factors,
        config,
        workspace0,
        workspace1,
        collapse_equivalent_tracks,
        phase_beta_cache_budget_bytes(),
    )
}

#[allow(clippy::too_many_arguments)]
fn update_sample_pair_impl_with_budget(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample0: usize,
    sample1: usize,
    cluster_template0: &PhaseClusterTemplate,
    cluster_template1: &PhaseClusterTemplate,
    finished0: &[u64],
    finished1: &[u64],
    any_unphased_heterozygote: bool,
    masked_heterozygotes0: &[u64],
    masked_heterozygotes1: &[u64],
    panel0: &CompositePanel,
    panel1: &CompositePanel,
    packed_target: Option<&PackedHaplotypes>,
    transition_factors: &[f32],
    config: &HMMConfig,
    workspace0: &mut PhaseBaumWorkspace,
    workspace1: &mut PhaseBaumWorkspace,
    collapse_equivalent_tracks: bool,
    beta_cache_budget_bytes: Option<usize>,
) -> Result<(JointPhaseUpdate, JointPhaseUpdate), String> {
    let print_pair_detail =
        sample0 == 0 || std::env::var_os("JANUSX_IMP_PHASE_PAIR_PROFILE_ALL").is_some();
    let profile_started = (print_pair_detail
        && std::env::var_os("JANUSX_IMP_PHASE_DETAIL_PROFILE").is_some())
    .then(Instant::now);
    let detail_profile = detail_profiling_enabled();
    let detail_started = detail_profile.then(Instant::now);
    if sample0 == sample1 {
        return Err(format!(
            "paired phase HMM received the same sample twice: {sample0}"
        ));
    }
    if !panel0.haplotypes.is_state_major() || !panel1.haplotypes.is_state_major() {
        return Err(format!(
            "paired phase HMM panels for samples {sample0}/{sample1} are not state-major"
        ));
    }
    if panel0.haplotypes.n_markers != input.rows.len()
        || panel1.haplotypes.n_markers != input.rows.len()
    {
        return Err(format!(
            "paired phase HMM marker counts differ for samples {sample0}/{sample1}: \
             panels {}/{}, input {}",
            panel0.haplotypes.n_markers,
            panel1.haplotypes.n_markers,
            input.rows.len()
        ));
    }
    let mask_words = input.rows.len().div_ceil(64);
    if masked_heterozygotes0.len() != mask_words || masked_heterozygotes1.len() != mask_words {
        return Err("paired phase HMM trailing-mask dimensions differ".to_owned());
    }
    let clusters_started = detail_profile.then(Instant::now);
    let mut clusters0 = std::mem::take(&mut workspace0.clusters);
    build_clusters_from_template_into(
        cluster_template0,
        input,
        sample0,
        finished0,
        any_unphased_heterozygote,
        masked_heterozygotes0,
        &mut clusters0,
    )?;
    let mut clusters1 = std::mem::take(&mut workspace1.clusters);
    build_clusters_from_template_into(
        cluster_template1,
        input,
        sample1,
        finished1,
        any_unphased_heterozygote,
        masked_heterozygotes1,
        &mut clusters1,
    )?;
    let cluster_count0 = clusters0.len();
    let cluster_count1 = clusters1.len();
    if count_profiling_enabled() {
        let states0 = panel0.haplotypes.n_haplotypes;
        let states1 = panel1.haplotypes.n_haplotypes;
        let unphased = clusters0
            .iter()
            .chain(&clusters1)
            .filter(|&&cluster| is_unphased_event(cluster))
            .count() as u64;
        let missing_or_masked = clusters0
            .iter()
            .chain(&clusters1)
            .filter(|&&cluster| {
                matches!(
                    cluster.kind,
                    ClusterKind::Missing | ClusterKind::MaskedHeterozygote
                )
            })
            .count() as u64;
        let missing_clusters = clusters0
            .iter()
            .chain(&clusters1)
            .filter(|&&cluster| cluster.kind == ClusterKind::Missing)
            .count() as u64;
        let masked_heterozygote_clusters = clusters0
            .iter()
            .chain(&clusters1)
            .filter(|&&cluster| cluster.kind == ClusterKind::MaskedHeterozygote)
            .count() as u64;
        let emission_rows = clusters0
            .iter()
            .chain(&clusters1)
            .map(|&cluster| emission_row_count(cluster.kind) as u64)
            .sum();
        let state_cluster_updates = cluster_count0
            .checked_mul(states0)
            .and_then(|first| {
                cluster_count1
                    .checked_mul(states1)
                    .and_then(|second| first.checked_add(second))
            })
            .ok_or_else(|| "HMM count profile overflow".to_owned())?
            as u64;
        COUNT_SAMPLES.fetch_add(2, DetailOrdering::Relaxed);
        COUNT_CLUSTERS.fetch_add(
            (cluster_count0 + cluster_count1) as u64,
            DetailOrdering::Relaxed,
        );
        COUNT_STATES.fetch_add((states0 + states1) as u64, DetailOrdering::Relaxed);
        COUNT_STATE_CLUSTER_UPDATES.fetch_add(state_cluster_updates, DetailOrdering::Relaxed);
        COUNT_UNPHASED_EVENTS.fetch_add(unphased, DetailOrdering::Relaxed);
        COUNT_MISSING_OR_MASKED.fetch_add(missing_or_masked, DetailOrdering::Relaxed);
        COUNT_MISSING_CLUSTERS.fetch_add(missing_clusters, DetailOrdering::Relaxed);
        COUNT_MASKED_HETEROZYGOTE_CLUSTERS
            .fetch_add(masked_heterozygote_clusters, DetailOrdering::Relaxed);
        COUNT_EMISSION_ROWS.fetch_add(emission_rows, DetailOrdering::Relaxed);
    }
    if let Some(started) = clusters_started {
        PREPARE_CLUSTERS_NANOS
            .fetch_add(started.elapsed().as_nanos() as u64, DetailOrdering::Relaxed);
    }
    let clusters_elapsed =
        profile_started.map_or(std::time::Duration::ZERO, |start| start.elapsed());
    let transitions_started = detail_profile.then(Instant::now);
    let mut transitions0 = std::mem::take(&mut workspace0.transitions);
    phase_transitions_into(
        &clusters0,
        panel0.haplotypes.n_haplotypes,
        transition_factors,
        &mut transitions0,
    );
    let mut transitions1 = std::mem::take(&mut workspace1.transitions);
    phase_transitions_into(
        &clusters1,
        panel1.haplotypes.n_haplotypes,
        transition_factors,
        &mut transitions1,
    );
    if let Some(started) = transitions_started {
        PREPARE_TRANSITIONS_NANOS
            .fetch_add(started.elapsed().as_nanos() as u64, DetailOrdering::Relaxed);
    }
    let mut run0 = prepare_event_sample_from_clusters_with_budget(
        input,
        snapshot,
        sample0,
        clusters0,
        transitions0,
        panel0,
        config,
        packed_target,
        workspace0,
        collapse_equivalent_tracks,
        beta_cache_budget_bytes,
    )?;
    let mut run1 = prepare_event_sample_from_clusters_with_budget(
        input,
        snapshot,
        sample1,
        clusters1,
        transitions1,
        panel1,
        config,
        packed_target,
        workspace1,
        collapse_equivalent_tracks,
        beta_cache_budget_bytes,
    )?;
    let detail_prepared = detail_profile.then(Instant::now);
    let prepare_elapsed =
        profile_started.map_or(std::time::Duration::ZERO, |start| start.elapsed());
    if run0.single_track && run1.single_track {
        run_backward_pair_single_track(&mut run0, &mut run1);
    } else {
        run_backward_pair(&mut run0, &mut run1);
    }
    let detail_backward = detail_profile.then(Instant::now);
    let backward_elapsed =
        profile_started.map_or(std::time::Duration::ZERO, |start| start.elapsed());
    if run0.single_track && run1.single_track {
        run_forward_pair_single_track(&mut run0, &mut run1, input, snapshot, panel0, panel1)?;
    } else {
        run_forward_pair(&mut run0, &mut run1, input, snapshot, panel0, panel1)?;
    }
    let detail_forward = detail_profile.then(Instant::now);
    let forward_elapsed =
        profile_started.map_or(std::time::Duration::ZERO, |start| start.elapsed());
    #[cfg(target_arch = "x86_64")]
    if let (Some(started), Some(prepared), Some(backward), Some(forward)) = (
        detail_started,
        detail_prepared,
        detail_backward,
        detail_forward,
    ) {
        PAIR_PREPARE_NANOS.fetch_add(
            prepared.duration_since(started).as_nanos() as u64,
            DetailOrdering::Relaxed,
        );
        PAIR_BACKWARD_NANOS.fetch_add(
            backward.duration_since(prepared).as_nanos() as u64,
            DetailOrdering::Relaxed,
        );
        PAIR_FORWARD_NANOS.fetch_add(
            forward.duration_since(backward).as_nanos() as u64,
            DetailOrdering::Relaxed,
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (
        detail_started,
        detail_prepared,
        detail_backward,
        detail_forward,
    );
    if let Some(started) = profile_started {
        eprintln!(
            "JANUSX_IMP_PHASE_DETAIL window={}:{} sample={} clusters0={} clusters1={} states0={} states1={} clusters_s={:.6} prepare_s={:.6} backward_s={:.6} forward_s={:.6} total_s={:.6}",
            input.chrom,
            input.index,
            sample0,
            cluster_count0,
            cluster_count1,
            panel0.haplotypes.n_haplotypes,
            panel1.haplotypes.n_haplotypes,
            clusters_elapsed.as_secs_f64(),
            prepare_elapsed.as_secs_f64(),
            (backward_elapsed - prepare_elapsed).as_secs_f64(),
            (forward_elapsed - backward_elapsed).as_secs_f64(),
            started.elapsed().as_secs_f64(),
        );
    }
    recycle_event_workspace(workspace0, &mut run0);
    recycle_event_workspace(workspace1, &mut run1);
    Ok((
        finish_event_update(run0, snapshot, false),
        finish_event_update(run1, snapshot, false),
    ))
}

#[inline]
fn phase_transitions_into(
    clusters: &[PhaseCluster],
    state_count: usize,
    factors: &[f32],
    transitions: &mut Vec<f32>,
) {
    transitions.clear();
    transitions.resize(clusters.len(), 0.0);
    for cluster in 1..clusters.len() {
        let previous_marker = clusters[cluster - 1].end - 1;
        let current_marker = clusters[cluster].end - 1;
        transitions[cluster] = cluster_transition_probability_cached(
            factors,
            previous_marker,
            current_marker,
            state_count,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_event_sample_from_clusters(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    clusters: Vec<PhaseCluster>,
    transitions: Vec<f32>,
    panel: &CompositePanel,
    config: &HMMConfig,
    packed_target: Option<&PackedHaplotypes>,
    workspace: &mut PhaseBaumWorkspace,
    collapse_equivalent_tracks: bool,
) -> Result<PreparedEventSample, String> {
    prepare_event_sample_from_clusters_with_budget(
        input,
        snapshot,
        sample,
        clusters,
        transitions,
        panel,
        config,
        packed_target,
        workspace,
        collapse_equivalent_tracks,
        phase_beta_cache_budget_bytes(),
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_event_sample_from_clusters_with_budget(
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    clusters: Vec<PhaseCluster>,
    transitions: Vec<f32>,
    panel: &CompositePanel,
    config: &HMMConfig,
    packed_target: Option<&PackedHaplotypes>,
    workspace: &mut PhaseBaumWorkspace,
    collapse_equivalent_tracks: bool,
    beta_cache_budget_bytes: Option<usize>,
) -> Result<PreparedEventSample, String> {
    let profile_started = (sample == 0
        && std::env::var_os("JANUSX_IMP_PHASE_DETAIL_PROFILE").is_some())
    .then(Instant::now);
    let detail_profile = detail_profiling_enabled();
    let state_count = panel.haplotypes.n_haplotypes;
    let single_track = collapse_equivalent_tracks
        && clusters
            .iter()
            .all(|cluster| matches!(cluster.kind, ClusterKind::Homozygous | ClusterKind::Missing));
    if !panel.haplotypes.is_state_major() || panel.haplotypes.bits_per_allele != 1 {
        return Err("phase emission state layout is not state-major biallelic".to_owned());
    }
    let complete = panel.haplotypes.is_complete();
    let panel_marker_count = panel.haplotypes.n_markers;
    let (_, allele_bits, missing_bits) = panel
        .haplotypes
        .state_major_biallelic_words(0)
        .ok_or_else(|| "phase emission state layout is not state-major".to_owned())?;
    let track_values = state_count
        .checked_mul(3)
        .ok_or_else(|| "event HMM state vector size overflow".to_owned())?;
    let emission_started = detail_profile.then(Instant::now);
    let mut emission_code_offsets = std::mem::take(&mut workspace.emission_code_offsets);
    emission_code_offsets.clear();
    emission_code_offsets.reserve(clusters.len());
    let mut emission_code_count = 0usize;
    for cluster in &clusters {
        emission_code_offsets.push(emission_code_count);
        let rows = emission_row_count(cluster.kind);
        let row_values = state_count
            .checked_mul(rows)
            .ok_or_else(|| "event HMM emission size overflow".to_owned())?;
        emission_code_count = emission_code_count
            .checked_add(row_values)
            .ok_or_else(|| "event HMM emission size overflow".to_owned())?;
    }
    let mut emission_codes = std::mem::take(&mut workspace.emission_codes);
    emission_codes.clear();
    emission_codes.reserve(emission_code_count);
    let mut emission_probabilities = std::mem::take(&mut workspace.emission_probabilities);
    emission_probabilities.clear();
    emission_probabilities.reserve(clusters.len());
    for &cluster in &clusters {
        let raw_mismatch = ((cluster.end - cluster.start) as f32 * config.err as f32).min(0.5f32);
        let mismatch_probability = raw_mismatch.max(f32::MIN_POSITIVE);
        let match_probability = (1.0f32 - raw_mismatch).max(f32::MIN_POSITIVE);
        emission_probabilities.push([match_probability, mismatch_probability]);
    }
    {
        let uninitialized = &mut emission_codes.spare_capacity_mut()[..emission_code_count];
        let column_filled = if complete {
            if let Some(packed_target) = packed_target {
                fill_state_major_column_emission_codes(
                    uninitialized,
                    &clusters,
                    &emission_code_offsets,
                    sample,
                    panel,
                    packed_target,
                    &mut workspace.packed_emission_scratch,
                )?
            } else {
                false
            }
        } else {
            false
        };
        if !column_filled {
            for (cluster_index, &cluster) in clusters.iter().enumerate() {
                let offset = emission_code_offsets[cluster_index];
                let rows = emission_row_count(cluster.kind);
                fill_state_major_emission_codes(
                    &mut uninitialized[offset..offset + rows * state_count],
                    cluster,
                    input,
                    snapshot,
                    sample,
                    allele_bits,
                    missing_bits,
                    panel_marker_count,
                    state_count,
                    complete,
                    packed_target,
                    &mut workspace.packed_emission_scratch,
                )?;
            }
        }
    }
    // Every byte is initialized before the length becomes observable.
    unsafe {
        emission_codes.set_len(emission_code_count);
    }
    if let Some(started) = emission_started {
        PREPARE_EMISSIONS_NANOS
            .fetch_add(started.elapsed().as_nanos() as u64, DetailOrdering::Relaxed);
    }
    let emission_elapsed =
        profile_started.map_or(std::time::Duration::ZERO, |start| start.elapsed());
    let workspace_started = detail_profile.then(Instant::now);
    let needed = |cluster: PhaseCluster| {
        is_unphased_event(cluster)
            || matches!(
                cluster.kind,
                ClusterKind::MaskedHeterozygote | ClusterKind::Missing
            )
    };
    let mut slots = std::mem::take(&mut workspace.slots);
    slots.clear();
    slots.resize(clusters.len(), NO_EVENT_SLOT);
    let mut needed_count = 0usize;
    for (cluster_index, &cluster) in clusters.iter().enumerate() {
        if needed(cluster) {
            slots[cluster_index] = needed_count;
            needed_count += 1;
        }
    }
    let beta_cache_tracks_per_slot = if single_track { 1 } else { 2 };
    let beta_cache_values_per_slot = state_count
        .checked_mul(beta_cache_tracks_per_slot)
        .ok_or_else(|| "event HMM haplotype backward size overflow".to_owned())?;
    let full_beta_cache_len = needed_count
        .checked_mul(beta_cache_values_per_slot)
        .ok_or_else(|| "event HMM backward size overflow".to_owned())?;
    let full_beta_cache_bytes = full_beta_cache_len.saturating_mul(std::mem::size_of::<f32>());
    let checkpointed = beta_cache_budget_bytes
        .is_some_and(|budget| needed_count > 0 && full_beta_cache_bytes > budget);
    let mut beta_cache = std::mem::take(&mut workspace.beta_cache);
    let mut beta_checkpoints = std::mem::take(&mut workspace.beta_checkpoints);
    if checkpointed {
        let block_count = clusters.len().div_ceil(CHECKPOINT_STRIDE);
        let mut max_block_slots = 0usize;
        for block in 0..block_count {
            let block_start = block * CHECKPOINT_STRIDE;
            let block_end = ((block + 1) * CHECKPOINT_STRIDE).min(clusters.len());
            let mut block_slot = 0usize;
            for cluster_index in block_start..block_end {
                if slots[cluster_index] != NO_EVENT_SLOT {
                    slots[cluster_index] = block_slot;
                    block_slot += 1;
                }
            }
            max_block_slots = max_block_slots.max(block_slot);
        }
        let block_beta_cache_len = max_block_slots
            .checked_mul(beta_cache_values_per_slot)
            .ok_or_else(|| "event HMM block backward cache size overflow".to_owned())?;
        let checkpoint_values_per_block = if single_track { state_count } else { track_values };
        let checkpoint_len = block_count
            .checked_mul(checkpoint_values_per_block)
            .ok_or_else(|| "event HMM backward checkpoint size overflow".to_owned())?;
        beta_cache.clear();
        beta_cache.resize_with(block_beta_cache_len, MaybeUninit::uninit);
        if beta_cache_budget_bytes.is_some_and(|budget| vec_cap_bytes(&beta_cache) > budget) {
            beta_cache.shrink_to(block_beta_cache_len);
        }
        beta_checkpoints.clear();
        beta_checkpoints.resize_with(checkpoint_len, MaybeUninit::uninit);
        COUNT_CHECKPOINTED_PREPARATIONS.fetch_add(1, DetailOrdering::Relaxed);
    } else {
        beta_cache.clear();
        beta_cache.resize_with(full_beta_cache_len, MaybeUninit::uninit);
        beta_checkpoints.clear();
        beta_checkpoints.shrink_to(0);
    }
    if count_profiling_enabled() {
        let bytes = vec_cap_bytes(&beta_cache).saturating_add(vec_cap_bytes(&beta_checkpoints))
            as u64;
        COUNT_BETA_CACHE_BYTES.fetch_add(bytes, DetailOrdering::Relaxed);
        PEAK_BETA_CACHE_BYTES.fetch_max(bytes, DetailOrdering::Relaxed);
        if single_track {
            COUNT_SINGLE_TRACK_SAMPLES.fetch_add(1, DetailOrdering::Relaxed);
        }
    }
    let mut beta_local = std::mem::take(&mut workspace.beta_local);
    beta_local.clear();
    beta_local.resize(track_values, 1.0f32 / state_count as f32);
    let mut beta_event_input = std::mem::take(&mut workspace.beta_event_input);
    beta_event_input.clear();
    beta_event_input.resize(track_values, 0.0);
    let mut beta_event = std::mem::take(&mut workspace.beta_event);
    beta_event.clear();
    beta_event.resize(track_values, 0.0);
    let mut alpha = std::mem::take(&mut workspace.alpha);
    alpha.clear();
    alpha.resize(track_values, 1.0f32 / state_count as f32);
    let unphased_heterozygote_count = clusters
        .iter()
        .filter(|&&cluster| is_unphased_event(cluster))
        .count();
    let scratch_bytes = packed_emission_scratch_bytes(&workspace.packed_emission_scratch);
    update_live_bytes(
        &mut workspace.tracked_scratch_bytes,
        scratch_bytes,
        &LIVE_EVENT_WORKSPACE_BYTES,
        &WINDOW_PEAK_LIVE_EVENT_WORKSPACE_BYTES,
        &PEAK_LIVE_EVENT_WORKSPACE_BYTES,
    );
    let mut tracked_beta_bytes = std::mem::replace(&mut workspace.tracked_beta_bytes, 0);
    let mut tracked_event_workspace_bytes =
        std::mem::replace(&mut workspace.tracked_event_workspace_bytes, 0);
    let beta_bytes = vec_cap_bytes(&beta_cache).saturating_add(vec_cap_bytes(&beta_checkpoints));
    let event_workspace_bytes = vec_cap_bytes(&clusters)
        .saturating_add(vec_cap_bytes(&transitions))
        .saturating_add(vec_cap_bytes(&emission_codes))
        .saturating_add(vec_cap_bytes(&emission_code_offsets))
        .saturating_add(vec_cap_bytes(&emission_probabilities))
        .saturating_add(vec_cap_bytes(&slots))
        .saturating_add(vec_cap_bytes(&beta_local))
        .saturating_add(vec_cap_bytes(&beta_event_input))
        .saturating_add(vec_cap_bytes(&beta_event))
        .saturating_add(vec_cap_bytes(&alpha));
    update_live_bytes(
        &mut tracked_beta_bytes,
        beta_bytes,
        &LIVE_BETA_CACHE_BYTES,
        &WINDOW_PEAK_LIVE_BETA_CACHE_BYTES,
        &PEAK_LIVE_BETA_CACHE_BYTES,
    );
    update_live_bytes(
        &mut tracked_event_workspace_bytes,
        event_workspace_bytes,
        &LIVE_EVENT_WORKSPACE_BYTES,
        &WINDOW_PEAK_LIVE_EVENT_WORKSPACE_BYTES,
        &PEAK_LIVE_EVENT_WORKSPACE_BYTES,
    );
    if let Some(started) = workspace_started {
        PREPARE_WORKSPACE_NANOS
            .fetch_add(started.elapsed().as_nanos() as u64, DetailOrdering::Relaxed);
    }
    let workspace_elapsed =
        profile_started.map_or(std::time::Duration::ZERO, |start| start.elapsed());
    if let Some(started) = profile_started {
        eprintln!(
            "JANUSX_IMP_PHASE_PREPARE_DETAIL sample={} markers={} clusters={} states={} emission_s={:.6} workspace_s={:.6} total_s={:.6}",
            sample,
            input.rows.len(),
            emission_code_offsets.len(),
            state_count,
            emission_elapsed.as_secs_f64(),
            (workspace_elapsed - emission_elapsed).as_secs_f64(),
            started.elapsed().as_secs_f64(),
        );
    }
    Ok(PreparedEventSample {
        sample,
        trace_config: *paired_hmm_trace_config(),
        avx2: phase_avx2_available(),
        single_track,
        checkpointed,
        complete_emissions: panel.haplotypes.is_complete(),
        backward_cursor: clusters.len() - 1,
        clusters,
        transitions,
        emission_codes,
        emission_code_offsets,
        emission_probabilities,
        slots,
        beta_cache,
        beta_checkpoints,
        beta_local,
        beta_event_input,
        beta_event,
        backward_active: true,
        alpha,
        forward_sums: [1.0; 3],
        changes: Vec::new(),
        phase_confidences: Vec::new(),
        swap_haps: false,
        phase_swap_count: 0,
        unphased_heterozygote_count,
        forward_cursor: 0,
        forward_active: true,
        tracked_beta_bytes,
        tracked_event_workspace_bytes,
    })
}

#[inline]
fn backward_store_needed(run: &mut PreparedEventSample) {
    if run.single_track {
        backward_store_single_track(run);
        return;
    }
    let cluster_index = run.backward_cursor;
    let cluster_count = run.clusters.len();
    let state_count = run.beta_local.len() / 3;
    let cluster = run.clusters[cluster_index];
    if run.checkpointed {
        let block = cluster_index / CHECKPOINT_STRIDE;
        let block_end = ((block + 1) * CHECKPOINT_STRIDE).min(cluster_count) - 1;
        if cluster_index == block_end {
            let track_values = state_count * 3;
            let checkpoint_offset = block * track_values;
            // SAFETY: `beta_checkpoints` was sized to `block_count * 3 * state_count`
            // and each block's right edge is visited once during the backward pass.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    run.beta_local.as_ptr(),
                    run.beta_checkpoints
                        .as_mut_ptr()
                        .add(checkpoint_offset)
                        .cast::<f32>(),
                    track_values,
                );
            }
        }
        if is_unphased_event(cluster) {
            run.beta_event_input.copy_from_slice(&run.beta_local);
            if cluster_index + 1 < cluster_count
                && is_unphased_event(run.clusters[cluster_index + 1])
            {
                copy_track(&mut run.beta_event_input, 0, 1, state_count);
                copy_track(&mut run.beta_event_input, 0, 2, state_count);
            }
            let emission_offset = run.emission_code_offsets[cluster_index];
            backward_project_event_from_codes_three(
                &run.beta_event_input,
                &mut run.beta_event,
                &run.emission_codes,
                emission_offset,
                cluster.kind,
                run.transitions[cluster_index],
                run.emission_probabilities[cluster_index],
                run.complete_emissions,
                run.avx2,
            );
        }
        return;
    }
    let cache_values = state_count * 2;
    let slot = run.slots[cluster_index];
    if slot == NO_EVENT_SLOT {
        return;
    }
    let offset = slot * cache_values;
    if !is_unphased_event(cluster) {
        // SAFETY: every event/masked/missing slot is visited exactly once by
        // the backward pass, and this destination was allocated with
        // `MaybeUninit::uninit` in the same call.
        unsafe {
            std::ptr::copy_nonoverlapping(
                run.beta_local.as_ptr().add(state_count),
                run.beta_cache.as_mut_ptr().add(offset).cast::<f32>(),
                cache_values,
            );
        }
        return;
    }
    run.beta_event_input.copy_from_slice(&run.beta_local);
    if cluster_index + 1 < cluster_count && is_unphased_event(run.clusters[cluster_index + 1]) {
        copy_track(&mut run.beta_event_input, 0, 1, state_count);
        copy_track(&mut run.beta_event_input, 0, 2, state_count);
    }
    let emission_offset = run.emission_code_offsets[cluster_index];
    backward_project_event_from_codes_three(
        &run.beta_event_input,
        &mut run.beta_event,
        &run.emission_codes,
        emission_offset,
        cluster.kind,
        run.transitions[cluster_index],
        run.emission_probabilities[cluster_index],
        run.complete_emissions,
        run.avx2,
    );
    // SAFETY: see the non-event branch above.
    unsafe {
        std::ptr::copy_nonoverlapping(
            run.beta_event.as_ptr().add(state_count),
            run.beta_cache.as_mut_ptr().add(offset).cast::<f32>(),
            cache_values,
        );
    }
}

#[inline]
fn backward_store_single_track(run: &mut PreparedEventSample) {
    let cluster_index = run.backward_cursor;
    let cluster_count = run.clusters.len();
    let state_count = run.beta_local.len() / 3;
    if run.checkpointed {
        let block = cluster_index / CHECKPOINT_STRIDE;
        let block_end = ((block + 1) * CHECKPOINT_STRIDE).min(cluster_count) - 1;
        if cluster_index == block_end {
            let checkpoint_offset = block * state_count;
            // SAFETY: `beta_checkpoints` was sized to `block_count * state_count`
            // for single-track runs and each block edge is written once.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    run.beta_local.as_ptr(),
                    run.beta_checkpoints
                        .as_mut_ptr()
                        .add(checkpoint_offset)
                        .cast::<f32>(),
                    state_count,
                );
            }
        }
        return;
    }
    let slot = run.slots[cluster_index];
    if slot == NO_EVENT_SLOT {
        return;
    }
    debug_assert!(matches!(
        run.clusters[cluster_index].kind,
        ClusterKind::Missing
    ));
    let offset = slot * state_count;
    // SAFETY: each missing cluster owns one slot, and the backward pass visits
    // it exactly once before the forward pass reads the shared track.
    unsafe {
        let destination = run.beta_cache.as_mut_ptr().add(offset).cast::<f32>();
        std::ptr::copy_nonoverlapping(run.beta_local.as_ptr(), destination, state_count);
    }
}

#[inline]
fn ensure_forward_block_ready(run: &mut PreparedEventSample) {
    if run.checkpointed && run.forward_cursor % CHECKPOINT_STRIDE == 0 {
        reconstruct_backward_block(run, run.forward_cursor / CHECKPOINT_STRIDE);
    }
}

fn reconstruct_backward_block(run: &mut PreparedEventSample, block: usize) {
    let cluster_count = run.clusters.len();
    let block_start = block * CHECKPOINT_STRIDE;
    let block_end_excl = ((block + 1) * CHECKPOINT_STRIDE).min(cluster_count);
    if !run.slots[block_start..block_end_excl]
        .iter()
        .any(|&slot| slot != NO_EVENT_SLOT)
    {
        return;
    }
    let state_count = run.beta_local.len() / 3;
    if run.single_track {
        let checkpoint_offset = block * state_count;
        // SAFETY: the full backward pass initialized every block checkpoint
        // before the forward pass began.
        unsafe {
            std::ptr::copy_nonoverlapping(
                run.beta_checkpoints
                    .as_ptr()
                    .add(checkpoint_offset)
                    .cast::<f32>(),
                run.beta_local.as_mut_ptr(),
                state_count,
            );
        }
        duplicate_first_track(&mut run.beta_local, state_count);
        for cluster_index in (block_start..block_end_excl).rev() {
            let slot = run.slots[cluster_index];
            if slot != NO_EVENT_SLOT {
                let offset = slot * state_count;
                // SAFETY: `beta_cache` holds `max_block_slots * state_count` values.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        run.beta_local.as_ptr(),
                        run.beta_cache.as_mut_ptr().add(offset).cast::<f32>(),
                        state_count,
                    );
                }
            }
            if cluster_index > block_start {
                let cluster = run.clusters[cluster_index];
                let emission_offset = run.emission_code_offsets[cluster_index];
                backward_project_single_from_codes(
                    &mut run.beta_local[..state_count],
                    &run.emission_codes,
                    emission_offset,
                    cluster.kind,
                    run.transitions[cluster_index],
                    run.emission_probabilities[cluster_index],
                    run.complete_emissions,
                    run.avx2,
                );
                duplicate_first_track(&mut run.beta_local, state_count);
            }
        }
        return;
    }

    let track_values = state_count * 3;
    let cache_values = state_count * 2;
    let checkpoint_offset = block * track_values;
    // SAFETY: the full backward pass initialized every block checkpoint
    // before the forward pass began.
    unsafe {
        std::ptr::copy_nonoverlapping(
            run.beta_checkpoints
                .as_ptr()
                .add(checkpoint_offset)
                .cast::<f32>(),
            run.beta_local.as_mut_ptr(),
            track_values,
        );
    }
    for cluster_index in (block_start..block_end_excl).rev() {
        let cluster = run.clusters[cluster_index];
        let slot = run.slots[cluster_index];
        if slot != NO_EVENT_SLOT {
            let offset = slot * cache_values;
            if !is_unphased_event(cluster) {
                // SAFETY: `beta_cache` holds `max_block_slots * 2 * state_count` values.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        run.beta_local.as_ptr().add(state_count),
                        run.beta_cache.as_mut_ptr().add(offset).cast::<f32>(),
                        cache_values,
                    );
                }
            } else {
                run.beta_event_input.copy_from_slice(&run.beta_local);
                if cluster_index + 1 < cluster_count
                    && is_unphased_event(run.clusters[cluster_index + 1])
                {
                    copy_track(&mut run.beta_event_input, 0, 1, state_count);
                    copy_track(&mut run.beta_event_input, 0, 2, state_count);
                }
                let emission_offset = run.emission_code_offsets[cluster_index];
                backward_project_event_from_codes_three(
                    &run.beta_event_input,
                    &mut run.beta_event,
                    &run.emission_codes,
                    emission_offset,
                    cluster.kind,
                    run.transitions[cluster_index],
                    run.emission_probabilities[cluster_index],
                    run.complete_emissions,
                    run.avx2,
                );
                // SAFETY: see the non-event branch above.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        run.beta_event.as_ptr().add(state_count),
                        run.beta_cache.as_mut_ptr().add(offset).cast::<f32>(),
                        cache_values,
                    );
                }
            }
        }
        if cluster_index + 1 < cluster_count
            && is_unphased_event(run.clusters[cluster_index + 1])
        {
            copy_track(&mut run.beta_local, 0, 1, state_count);
            copy_track(&mut run.beta_local, 0, 2, state_count);
        }
        if cluster_index > block_start {
            if is_unphased_event(cluster) {
                backward_apply_stored_event(run);
            } else {
                let emission_offset = run.emission_code_offsets[cluster_index];
                backward_project_from_codes_three(
                    &mut run.beta_local,
                    &run.emission_codes,
                    emission_offset,
                    cluster.kind,
                    run.transitions[cluster_index],
                    run.emission_probabilities[cluster_index],
                    run.complete_emissions,
                    run.avx2,
                );
            }
        }
    }
}

#[inline]
fn beta_cache_slot(run: &PreparedEventSample, slot: usize) -> &[f32] {
    let state_count = run.beta_local.len() / 3;
    let stride = state_count * if run.single_track { 1 } else { 2 };
    let offset = slot * stride;
    debug_assert!(offset + stride <= run.beta_cache.len());
    // SAFETY: the complete backward pass initializes every allocated slot
    // before it becomes inactive; forward passes only read after that point.
    unsafe { std::slice::from_raw_parts(run.beta_cache.as_ptr().add(offset).cast::<f32>(), stride) }
}

#[inline]
fn beta_tracks(run: &PreparedEventSample, slot: usize) -> (&[f32], &[f32]) {
    let state_count = run.beta_local.len() / 3;
    let beta = beta_cache_slot(run, slot);
    if run.single_track {
        let track = &beta[..state_count];
        (track, track)
    } else {
        (&beta[..state_count], &beta[state_count..2 * state_count])
    }
}

#[inline]
fn backward_apply_stored_event(run: &mut PreparedEventSample) {
    run.beta_local.copy_from_slice(&run.beta_event);
}

#[inline]
fn duplicate_first_track(values: &mut [f32], state_count: usize) {
    values.copy_within(0..state_count, state_count);
    values.copy_within(0..state_count, 2 * state_count);
}

#[inline]
fn backward_reset_for_following_event(run: &mut PreparedEventSample) {
    if run.backward_cursor + 1 >= run.clusters.len()
        || !is_unphased_event(run.clusters[run.backward_cursor + 1])
    {
        return;
    }
    // Beagle first projects the event into the current missing/masked
    // posterior, then resets the haplotype tracks before projecting the
    // current cluster. The caller stores the current event slot first so the
    // two orientations remain available to phaseHet.
    let state_count = run.beta_local.len() / 3;
    copy_track(&mut run.beta_local, 0, 1, state_count);
    copy_track(&mut run.beta_local, 0, 2, state_count);
}

#[inline]
fn run_backward_single(run: &mut PreparedEventSample) {
    while run.backward_active {
        backward_store_needed(run);
        backward_reset_for_following_event(run);
        if run.backward_cursor > 0 {
            let cluster = run.clusters[run.backward_cursor];
            if is_unphased_event(cluster) {
                backward_apply_stored_event(run);
            } else {
                let emission_offset = run.emission_code_offsets[run.backward_cursor];
                backward_project_from_codes_three(
                    &mut run.beta_local,
                    &run.emission_codes,
                    emission_offset,
                    cluster.kind,
                    run.transitions[run.backward_cursor],
                    run.emission_probabilities[run.backward_cursor],
                    run.complete_emissions,
                    run.avx2,
                );
            }
        }
        if run.backward_cursor == 0 {
            run.backward_active = false;
        } else {
            run.backward_cursor -= 1;
        }
    }
}

fn run_backward_single_track(run: &mut PreparedEventSample) {
    debug_assert!(run.single_track);
    while run.backward_active {
        backward_store_single_track(run);
        if run.backward_cursor > 0 {
            let state_count = run.beta_local.len() / 3;
            let cluster = run.clusters[run.backward_cursor];
            let emission_offset = run.emission_code_offsets[run.backward_cursor];
            backward_project_single_from_codes(
                &mut run.beta_local[..state_count],
                &run.emission_codes,
                emission_offset,
                cluster.kind,
                run.transitions[run.backward_cursor],
                run.emission_probabilities[run.backward_cursor],
                run.complete_emissions,
                run.avx2,
            );
            duplicate_first_track(&mut run.beta_local, state_count);
        }
        if run.backward_cursor == 0 {
            run.backward_active = false;
        } else {
            run.backward_cursor -= 1;
        }
    }
}

fn run_backward_pair_single_track(run0: &mut PreparedEventSample, run1: &mut PreparedEventSample) {
    debug_assert!(run0.single_track && run1.single_track);
    while run0.backward_active && run1.backward_active {
        backward_store_single_track(run0);
        backward_store_single_track(run1);
        let advance0 = run0.backward_cursor > 0;
        let advance1 = run1.backward_cursor > 0;
        if advance0 && advance1 {
            let states0 = run0.beta_local.len() / 3;
            let states1 = run1.beta_local.len() / 3;
            let (probabilities0, _) = run0.beta_local.split_at_mut(states0);
            let (probabilities1, _) = run1.beta_local.split_at_mut(states1);
            backward_project_two_from_codes(
                probabilities0,
                &run0.emission_codes,
                run0.emission_code_offsets[run0.backward_cursor],
                run0.clusters[run0.backward_cursor].kind,
                run0.transitions[run0.backward_cursor],
                run0.emission_probabilities[run0.backward_cursor],
                run0.complete_emissions,
                probabilities1,
                &run1.emission_codes,
                run1.emission_code_offsets[run1.backward_cursor],
                run1.clusters[run1.backward_cursor].kind,
                run1.transitions[run1.backward_cursor],
                run1.emission_probabilities[run1.backward_cursor],
                run1.complete_emissions,
                run0.avx2,
            );
            duplicate_first_track(&mut run0.beta_local, states0);
            duplicate_first_track(&mut run1.beta_local, states1);
        } else if advance0 {
            let state_count = run0.beta_local.len() / 3;
            backward_project_single_from_codes(
                &mut run0.beta_local[..state_count],
                &run0.emission_codes,
                run0.emission_code_offsets[run0.backward_cursor],
                run0.clusters[run0.backward_cursor].kind,
                run0.transitions[run0.backward_cursor],
                run0.emission_probabilities[run0.backward_cursor],
                run0.complete_emissions,
                run0.avx2,
            );
            duplicate_first_track(&mut run0.beta_local, state_count);
        } else if advance1 {
            let state_count = run1.beta_local.len() / 3;
            backward_project_single_from_codes(
                &mut run1.beta_local[..state_count],
                &run1.emission_codes,
                run1.emission_code_offsets[run1.backward_cursor],
                run1.clusters[run1.backward_cursor].kind,
                run1.transitions[run1.backward_cursor],
                run1.emission_probabilities[run1.backward_cursor],
                run1.complete_emissions,
                run1.avx2,
            );
            duplicate_first_track(&mut run1.beta_local, state_count);
        }
        if run0.backward_cursor == 0 {
            run0.backward_active = false;
        } else {
            run0.backward_cursor -= 1;
        }
        if run1.backward_cursor == 0 {
            run1.backward_active = false;
        } else {
            run1.backward_cursor -= 1;
        }
    }
    if run0.backward_active {
        run_backward_single_track(run0);
    }
    if run1.backward_active {
        run_backward_single_track(run1);
    }
}

fn run_backward_pair(run0: &mut PreparedEventSample, run1: &mut PreparedEventSample) {
    while run0.backward_active && run1.backward_active {
        backward_store_needed(run0);
        backward_store_needed(run1);
        backward_reset_for_following_event(run0);
        backward_reset_for_following_event(run1);
        let advance0 = run0.backward_cursor > 0;
        let advance1 = run1.backward_cursor > 0;
        let stored_event0 = advance0 && is_unphased_event(run0.clusters[run0.backward_cursor]);
        let stored_event1 = advance1 && is_unphased_event(run1.clusters[run1.backward_cursor]);
        match (advance0, advance1, stored_event0, stored_event1) {
            (true, true, false, false) => {
                backward_project_from_codes_six(
                    &mut run0.beta_local,
                    &run0.emission_codes,
                    run0.emission_code_offsets[run0.backward_cursor],
                    run0.clusters[run0.backward_cursor].kind,
                    run0.transitions[run0.backward_cursor],
                    run0.emission_probabilities[run0.backward_cursor],
                    run0.complete_emissions,
                    &mut run1.beta_local,
                    &run1.emission_codes,
                    run1.emission_code_offsets[run1.backward_cursor],
                    run1.clusters[run1.backward_cursor].kind,
                    run1.transitions[run1.backward_cursor],
                    run1.emission_probabilities[run1.backward_cursor],
                    run1.complete_emissions,
                    run1.avx2,
                );
            }
            (true, true, true, true) => {
                backward_apply_stored_event(run0);
                backward_apply_stored_event(run1);
            }
            (true, true, true, false) => {
                backward_apply_stored_event(run0);
                backward_project_from_codes_three(
                    &mut run1.beta_local,
                    &run1.emission_codes,
                    run1.emission_code_offsets[run1.backward_cursor],
                    run1.clusters[run1.backward_cursor].kind,
                    run1.transitions[run1.backward_cursor],
                    run1.emission_probabilities[run1.backward_cursor],
                    run1.complete_emissions,
                    run1.avx2,
                );
            }
            (true, true, false, true) => {
                backward_project_from_codes_three(
                    &mut run0.beta_local,
                    &run0.emission_codes,
                    run0.emission_code_offsets[run0.backward_cursor],
                    run0.clusters[run0.backward_cursor].kind,
                    run0.transitions[run0.backward_cursor],
                    run0.emission_probabilities[run0.backward_cursor],
                    run0.complete_emissions,
                    run0.avx2,
                );
                backward_apply_stored_event(run1);
            }
            (true, false, _, _) => {
                backward_project_from_codes_three(
                    &mut run0.beta_local,
                    &run0.emission_codes,
                    run0.emission_code_offsets[run0.backward_cursor],
                    run0.clusters[run0.backward_cursor].kind,
                    run0.transitions[run0.backward_cursor],
                    run0.emission_probabilities[run0.backward_cursor],
                    run0.complete_emissions,
                    run0.avx2,
                );
            }
            (false, true, _, _) => {
                backward_project_from_codes_three(
                    &mut run1.beta_local,
                    &run1.emission_codes,
                    run1.emission_code_offsets[run1.backward_cursor],
                    run1.clusters[run1.backward_cursor].kind,
                    run1.transitions[run1.backward_cursor],
                    run1.emission_probabilities[run1.backward_cursor],
                    run1.complete_emissions,
                    run1.avx2,
                );
            }
            (false, false, _, _) => {}
        }
        if run0.backward_cursor == 0 {
            run0.backward_active = false;
        } else {
            run0.backward_cursor -= 1;
        }
        if run1.backward_cursor == 0 {
            run1.backward_active = false;
        } else {
            run1.backward_cursor -= 1;
        }
    }
    if run0.backward_active {
        run_backward_single(run0);
    }
    if run1.backward_active {
        run_backward_single(run1);
    }
}

fn forward_begin_cluster(run: &mut PreparedEventSample) -> Result<Option<(bool, f64)>, String> {
    let state_count = run.alpha.len() / 3;
    let cluster = run.clusters[run.forward_cursor];
    if !is_unphased_event(cluster) {
        return Ok(None);
    }
    let slot = run.slots[run.forward_cursor];
    if slot == NO_EVENT_SLOT {
        return Err("event HMM is missing an event posterior".to_owned());
    }
    let (beta0, beta1) = beta_tracks(run, slot);
    let (swap, log_odds) = choose_phase_swap(
        &run.alpha[state_count..2 * state_count],
        &run.alpha[2 * state_count..3 * state_count],
        beta0,
        beta1,
    )?;
    let trace_event = run.trace_config.event_target == Some((run.sample, cluster.start));
    let trace_all_events = run.trace_config.sample_target == Some(run.sample);
    if trace_event || trace_all_events {
        let (no_swap, swapped) = phase_swap_products(
            &run.alpha[state_count..2 * state_count],
            &run.alpha[2 * state_count..3 * state_count],
            beta0,
            beta1,
        )?;
        let java_float_hash = |values: &[f32]| {
            values.iter().fold(1i32, |hash, value| {
                hash.wrapping_mul(31).wrapping_add(value.to_bits() as i32)
            })
        };
        eprintln!(
            "JANUSX_PAIRED_HMM_EVENT\tSAMPLE={}\tMARKER={}\tSTATES={state_count}\tFORWARD1_HASH={}\tFORWARD2_HASH={}\tBACKWARD1_HASH={}\tBACKWARD2_HASH={}\tNO_SWAP={no_swap:e}\tSWAP={swapped:e}\tSELECTED_SWAP={swap}\tLOG_ODDS={log_odds:.17e}",
            run.sample,
            cluster.start,
            java_float_hash(&run.alpha[state_count..2 * state_count]),
            java_float_hash(&run.alpha[2 * state_count..3 * state_count]),
            java_float_hash(beta0),
            java_float_hash(beta1),
        );
    }
    if run.swap_haps != swap {
        run.phase_swap_count += 1;
    }
    run.swap_haps = swap;
    run.alpha.copy_within(0..state_count, state_count);
    run.alpha.copy_within(0..state_count, 2 * state_count);
    run.forward_sums[1] = run.forward_sums[0];
    run.forward_sums[2] = run.forward_sums[0];
    Ok(Some((swap, log_odds)))
}

fn forward_finish_cluster(
    run: &mut PreparedEventSample,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    panel: &CompositePanel,
    confidence: Option<f64>,
) -> Result<(), String> {
    let state_count = panel.haplotypes.n_haplotypes;
    let cluster = run.clusters[run.forward_cursor];
    if let Some(log_odds) = confidence {
        let marker = cluster.start;
        let snapshot_call = snapshot.rows[marker].calls.get(run.sample).unwrap();
        let [Some(a), Some(b)] = snapshot_call.alleles else {
            return Err("unphased phase event has an incomplete genotype".to_owned());
        };
        run.changes.push((
            marker,
            DiploidCall {
                alleles: if run.swap_haps {
                    [Some(b), Some(a)]
                } else {
                    [Some(a), Some(b)]
                },
                phased: true,
            },
        ));
        run.phase_confidences.push((marker, log_odds));
    } else if cluster.kind == ClusterKind::MaskedHeterozygote {
        let marker = cluster.start;
        let slot = run.slots[run.forward_cursor];
        if slot == NO_EVENT_SLOT {
            return Err("event HMM is missing a masked posterior".to_owned());
        }
        let (beta0_track, beta1_track) = beta_tracks(run, slot);
        let beta0 = if run.swap_haps { 1 } else { 0 };
        let beta1 = if run.swap_haps { 0 } else { 1 };
        let allele_count = panel.allele_counts[marker].max(1);
        let (probabilities0, probabilities1) = hap_allele_posteriors(
            &run.alpha[state_count..2 * state_count],
            if beta0 == 0 { beta0_track } else { beta1_track },
            &run.alpha[2 * state_count..3 * state_count],
            if beta1 == 0 { beta0_track } else { beta1_track },
            panel,
            marker,
            allele_count,
        );
        let [Some(mut a), Some(mut b)] =
            snapshot.rows[marker].calls.get(run.sample).unwrap().alleles
        else {
            return Err("masked phase event has an incomplete genotype".to_owned());
        };
        if run.swap_haps {
            std::mem::swap(&mut a, &mut b);
        }
        let no_swap =
            allele_probability(&probabilities0, a) * allele_probability(&probabilities1, b);
        let swap = allele_probability(&probabilities0, b) * allele_probability(&probabilities1, a);
        let no_swap_f64 = f64::from(no_swap);
        let swap_f64 = f64::from(swap);
        let log_odds = if no_swap_f64.min(swap_f64) > f64::MIN_POSITIVE {
            (no_swap_f64.max(swap_f64) / no_swap_f64.min(swap_f64)).ln()
        } else if no_swap_f64.max(swap_f64) > 0.0 {
            f64::INFINITY
        } else {
            f64::INFINITY
        };
        run.changes.push((
            marker,
            DiploidCall {
                alleles: if swap > no_swap {
                    [Some(b), Some(a)]
                } else {
                    [Some(a), Some(b)]
                },
                phased: true,
            },
        ));
        run.phase_confidences.push((marker, log_odds));
    } else if let ClusterKind::Heterozygous { .. } = cluster.kind {
        let marker = cluster.start;
        let mut alleles = snapshot.rows[marker].calls.get(run.sample).unwrap().alleles;
        if run.swap_haps {
            alleles.swap(0, 1);
        }
        run.changes.push((
            marker,
            DiploidCall {
                alleles,
                phased: true,
            },
        ));
    } else if cluster.kind == ClusterKind::Homozygous {
        // Homozygous calls were marked phased when the output snapshot was
        // copied, so the forward pass does not need to rewrite this range.
    } else {
        let marker = cluster.start;
        let slot = run.slots[run.forward_cursor];
        if slot == NO_EVENT_SLOT {
            return Err("event HMM is missing a missing-GT posterior".to_owned());
        }
        let (beta0_track, beta1_track) = beta_tracks(run, slot);
        let beta0 = if run.swap_haps { 1 } else { 0 };
        let beta1 = if run.swap_haps { 0 } else { 1 };
        let allele_count = panel.allele_counts[marker].max(1);
        let (probabilities0, probabilities1) = hap_allele_posteriors(
            &run.alpha[state_count..2 * state_count],
            if beta0 == 0 { beta0_track } else { beta1_track },
            &run.alpha[2 * state_count..3 * state_count],
            if beta1 == 0 { beta0_track } else { beta1_track },
            panel,
            marker,
            allele_count,
        );
        if run.trace_config.missing_target == Some((input.index, run.sample, marker)) {
            let java_float_hash = |values: &[f32]| {
                values.iter().fold(1i32, |hash, value| {
                    hash.wrapping_mul(31).wrapping_add(value.to_bits() as i32)
                })
            };
            let java_state_hash = (0..state_count).fold(1i32, |hash, state| {
                hash.wrapping_mul(31)
                    .wrapping_add(panel.haplotypes.allele(marker, state).map_or(-1, i32::from))
            });
            eprintln!(
                "JANUSX_PAIRED_HMM_MISSING\tWINDOW={}\tSAMPLE={}\tMARKER={marker}\tSTATES={state_count}\tTRANSITION={}\tPANEL_COMPLETE={}\tSTATE_HASH={java_state_hash}\tFORWARD1_HASH={}\tFORWARD2_HASH={}\tBACKWARD1_HASH={}\tBACKWARD2_HASH={}\tPROBABILITIES0={probabilities0:?}\tPROBABILITIES1={probabilities1:?}",
                input.index,
                run.sample,
                run.transitions[run.forward_cursor],
                panel.haplotypes.is_complete(),
                java_float_hash(&run.alpha[state_count..2 * state_count]),
                java_float_hash(&run.alpha[2 * state_count..3 * state_count]),
                java_float_hash(if beta0 == 0 {
                    beta0_track
                } else {
                    beta1_track
                }),
                java_float_hash(if beta1 == 0 {
                    beta0_track
                } else {
                    beta1_track
                }),
            );
        }
        let input_call = input.rows[marker].calls.get(run.sample).unwrap();
        let allele0 = modal_allele(&probabilities0, input, marker);
        let allele1 = modal_allele(&probabilities1, input, marker);
        let alleles = match input_call.alleles {
            [Some(a), Some(b)] => [Some(a), Some(b)],
            [Some(a), None] => [Some(a), Some(allele1)],
            [None, Some(b)] if input_call.phased => [Some(allele0), Some(b)],
            [None, Some(b)] => [Some(b), Some(allele1)],
            [None, None] => [Some(allele0), Some(allele1)],
        };
        run.changes.push((
            marker,
            DiploidCall {
                alleles,
                phased: true,
            },
        ));
    }
    run.forward_cursor += 1;
    if run.forward_cursor == run.clusters.len() {
        run.forward_active = false;
    }
    Ok(())
}

fn run_forward_single(
    run: &mut PreparedEventSample,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    panel: &CompositePanel,
) -> Result<(), String> {
    while run.forward_active {
        ensure_forward_block_ready(run);
        let confidence = forward_begin_cluster(run)?;
        let state_count = run.alpha.len() / 3;
        let cluster_kind = run.clusters[run.forward_cursor].kind;
        let cluster_offset = run.emission_code_offsets[run.forward_cursor];
        let probabilities = run.emission_probabilities[run.forward_cursor];
        let source_tracks = if run.swap_haps {
            [0usize, 2, 1]
        } else {
            [0usize, 1, 2]
        };
        let rows = source_tracks
            .map(|track| emission_row_offset(cluster_kind, cluster_offset, state_count, track));
        run.forward_sums = beagle_forward_update_codes_three(
            &mut run.alpha,
            &run.emission_codes,
            rows,
            run.forward_sums,
            run.transitions[run.forward_cursor],
            probabilities,
            state_count,
            run.complete_emissions,
            run.avx2,
        );
        forward_finish_cluster(
            run,
            input,
            snapshot,
            panel,
            confidence.map(|(_, log_odds)| log_odds),
        )?;
    }
    Ok(())
}

fn run_forward_single_track(
    run: &mut PreparedEventSample,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    panel: &CompositePanel,
) -> Result<(), String> {
    debug_assert!(run.single_track);
    while run.forward_active {
        ensure_forward_block_ready(run);
        let state_count = run.alpha.len() / 3;
        let cluster = run.clusters[run.forward_cursor];
        let cluster_offset = run.emission_code_offsets[run.forward_cursor];
        let row = emission_row_offset(cluster.kind, cluster_offset, state_count, 0);
        let switch = run.transitions[run.forward_cursor];
        let scale = (1.0f32 - switch) / run.forward_sums[0];
        let shift = switch / state_count as f32;
        forward_values_from_codes(
            &mut run.alpha[..state_count],
            &run.emission_codes,
            row,
            scale,
            shift,
            run.emission_probabilities[run.forward_cursor],
            run.avx2,
            run.complete_emissions,
        );
        let sum = sum_lane(&run.alpha[..state_count]);
        duplicate_first_track(&mut run.alpha, state_count);
        run.forward_sums = [sum; 3];
        forward_finish_cluster(run, input, snapshot, panel, None)?;
    }
    Ok(())
}

fn run_forward_pair_single_track(
    run0: &mut PreparedEventSample,
    run1: &mut PreparedEventSample,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    panel0: &CompositePanel,
    panel1: &CompositePanel,
) -> Result<(), String> {
    debug_assert!(run0.single_track && run1.single_track);
    while run0.forward_active && run1.forward_active {
        ensure_forward_block_ready(run0);
        ensure_forward_block_ready(run1);
        let states0 = run0.alpha.len() / 3;
        let states1 = run1.alpha.len() / 3;
        let cluster0 = run0.clusters[run0.forward_cursor];
        let cluster1 = run1.clusters[run1.forward_cursor];
        let switch0 = run0.transitions[run0.forward_cursor];
        let switch1 = run1.transitions[run1.forward_cursor];
        let scale0 = (1.0f32 - switch0) / run0.forward_sums[0];
        let scale1 = (1.0f32 - switch1) / run1.forward_sums[0];
        let shift0 = switch0 / states0 as f32;
        let shift1 = switch1 / states1 as f32;
        let (alpha0, _) = run0.alpha.split_at_mut(states0);
        let (alpha1, _) = run1.alpha.split_at_mut(states1);
        forward_values_from_codes(
            alpha0,
            &run0.emission_codes,
            emission_row_offset(
                cluster0.kind,
                run0.emission_code_offsets[run0.forward_cursor],
                states0,
                0,
            ),
            scale0,
            shift0,
            run0.emission_probabilities[run0.forward_cursor],
            run0.avx2,
            run0.complete_emissions,
        );
        forward_values_from_codes(
            alpha1,
            &run1.emission_codes,
            emission_row_offset(
                cluster1.kind,
                run1.emission_code_offsets[run1.forward_cursor],
                states1,
                0,
            ),
            scale1,
            shift1,
            run1.emission_probabilities[run1.forward_cursor],
            run1.avx2,
            run1.complete_emissions,
        );
        let sums = sum_lanes_2([alpha0, alpha1]);
        duplicate_first_track(&mut run0.alpha, states0);
        duplicate_first_track(&mut run1.alpha, states1);
        run0.forward_sums = [sums[0]; 3];
        run1.forward_sums = [sums[1]; 3];
        forward_finish_cluster(run0, input, snapshot, panel0, None)?;
        forward_finish_cluster(run1, input, snapshot, panel1, None)?;
    }
    if run0.forward_active {
        run_forward_single_track(run0, input, snapshot, panel0)?;
    }
    if run1.forward_active {
        run_forward_single_track(run1, input, snapshot, panel1)?;
    }
    Ok(())
}

fn run_forward_pair(
    run0: &mut PreparedEventSample,
    run1: &mut PreparedEventSample,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    panel0: &CompositePanel,
    panel1: &CompositePanel,
) -> Result<(), String> {
    while run0.forward_active && run1.forward_active {
        ensure_forward_block_ready(run0);
        ensure_forward_block_ready(run1);
        let confidence0 = forward_begin_cluster(run0)?;
        let confidence1 = forward_begin_cluster(run1)?;
        debug_assert_eq!(run0.alpha.len(), 3 * panel0.haplotypes.n_haplotypes);
        debug_assert_eq!(run1.alpha.len(), 3 * panel1.haplotypes.n_haplotypes);
        let states0 = panel0.haplotypes.n_haplotypes;
        let states1 = panel1.haplotypes.n_haplotypes;
        let tracks0 = if run0.swap_haps {
            [0usize, 2, 1]
        } else {
            [0usize, 1, 2]
        };
        let tracks1 = if run1.swap_haps {
            [0usize, 2, 1]
        } else {
            [0usize, 1, 2]
        };
        let rows0 = tracks0.map(|track| {
            emission_row_offset(
                run0.clusters[run0.forward_cursor].kind,
                run0.emission_code_offsets[run0.forward_cursor],
                states0,
                track,
            )
        });
        let rows1 = tracks1.map(|track| {
            emission_row_offset(
                run1.clusters[run1.forward_cursor].kind,
                run1.emission_code_offsets[run1.forward_cursor],
                states1,
                track,
            )
        });
        let (sums0, sums1) = beagle_forward_update_codes_six(
            &mut run0.alpha,
            &run0.emission_codes,
            rows0,
            run0.forward_sums,
            run0.transitions[run0.forward_cursor],
            run0.emission_probabilities[run0.forward_cursor],
            states0,
            &mut run1.alpha,
            &run1.emission_codes,
            rows1,
            run1.forward_sums,
            run1.transitions[run1.forward_cursor],
            run1.emission_probabilities[run1.forward_cursor],
            states1,
            run0.complete_emissions,
            run1.complete_emissions,
            run0.avx2,
        );
        run0.forward_sums = sums0;
        run1.forward_sums = sums1;
        forward_finish_cluster(
            run0,
            input,
            snapshot,
            panel0,
            confidence0.map(|(_, log_odds)| log_odds),
        )?;
        forward_finish_cluster(
            run1,
            input,
            snapshot,
            panel1,
            confidence1.map(|(_, log_odds)| log_odds),
        )?;
    }
    if run0.forward_active {
        run_forward_single(run0, input, snapshot, panel0)?;
    }
    if run1.forward_active {
        run_forward_single(run1, input, snapshot, panel1)?;
    }
    Ok(())
}

pub(crate) fn choose_phase_swap(
    forward_hap0: &[f32],
    forward_hap1: &[f32],
    backward_hap0: &[f32],
    backward_hap1: &[f32],
) -> Result<(bool, f64), String> {
    let (no_swap, swap) =
        phase_swap_products(forward_hap0, forward_hap1, backward_hap0, backward_hap1)?;
    let no_swap_f64 = f64::from(no_swap);
    let swap_f64 = f64::from(swap);
    let smaller = no_swap_f64.min(swap_f64);
    let confidence = if smaller > f64::MIN_POSITIVE {
        (no_swap_f64.max(swap_f64) / smaller).ln()
    } else if no_swap_f64.max(swap_f64) > 0.0 {
        f64::INFINITY
    } else {
        f64::INFINITY
    };
    Ok((swap > no_swap, confidence))
}

fn phase_swap_products(
    forward_hap0: &[f32],
    forward_hap1: &[f32],
    backward_hap0: &[f32],
    backward_hap1: &[f32],
) -> Result<(f32, f32), String> {
    if forward_hap0.is_empty()
        || forward_hap0.len() != forward_hap1.len()
        || forward_hap0.len() != backward_hap0.len()
        || forward_hap0.len() != backward_hap1.len()
    {
        return Err("phase-odds vectors must have the same positive length".to_owned());
    }
    if forward_hap0
        .iter()
        .chain(forward_hap1)
        .chain(backward_hap0)
        .chain(backward_hap1)
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("phase-odds vectors must contain finite nonnegative values".to_owned());
    }
    // PhaseBaum2 accumulates each inner product and compares the two path
    // products in Java `float`; interleave the four independent chains while
    // preserving state-order accumulation within each chain.
    let mut f0b0 = 0.0f32;
    let mut f0b1 = 0.0f32;
    let mut f1b0 = 0.0f32;
    let mut f1b1 = 0.0f32;
    for (((forward0, forward1), backward0), backward1) in forward_hap0
        .iter()
        .zip(forward_hap1)
        .zip(backward_hap0)
        .zip(backward_hap1)
    {
        f0b0 += *forward0 * *backward0;
        f0b1 += *forward0 * *backward1;
        f1b0 += *forward1 * *backward0;
        f1b1 += *forward1 * *backward1;
    }
    let no_swap = f0b0 * f1b1;
    let swap = f0b1 * f1b0;
    Ok((no_swap, swap))
}

pub(crate) fn build_cluster_template(
    input: &PhaseWindow,
    sample: usize,
) -> Result<PhaseClusterTemplate, String> {
    if input.rows.is_empty() || input.rows.iter().any(|row| sample >= row.calls.len()) {
        return Err("phase cluster template sample is outside the window".to_owned());
    }
    build_cluster_template_from_calls(
        input.rows.len(),
        &input.cm_positions,
        input.rows[0].calls.len(),
        sample,
        |marker| input.rows[marker].calls.get(sample).unwrap(),
    )
}

fn build_cluster_template_from_calls<F>(
    marker_count: usize,
    cm_positions: &[f64],
    sample_count: usize,
    sample: usize,
    mut call_at: F,
) -> Result<PhaseClusterTemplate, String>
where
    F: FnMut(usize) -> DiploidCall,
{
    if marker_count == 0 || sample >= sample_count || cm_positions.len() != marker_count {
        return Err("phase cluster template dimensions are invalid".to_owned());
    }
    let mut template = PhaseClusterTemplate::default();
    template.sample = sample;
    let mut marker = 0usize;
    while marker < marker_count {
        let call = call_at(marker);
        match call.alleles {
            [Some(a), Some(b)] if a == b => {
                let start = marker;
                marker += 1;
                while marker < marker_count {
                    if marker - start >= MAX_CLUSTER_MARKERS
                        || cm_positions[marker] - cm_positions[start] > MAX_CLUSTER_CM
                    {
                        break;
                    }
                    let next = call_at(marker);
                    if !matches!(next.alleles, [Some(x), Some(y)] if x == y) {
                        break;
                    }
                    marker += 1;
                }
                template
                    .sizes
                    .push(u8::try_from(marker - start).map_err(|_| {
                        "phase cluster template exceeded the marker-cluster limit".to_owned()
                    })?);
                template.kinds.push(PhaseClusterTemplateKind::Homozygous);
            }
            [Some(_), Some(_)] => {
                template.sizes.push(1);
                template.kinds.push(PhaseClusterTemplateKind::Heterozygous);
                marker += 1;
            }
            _ => {
                template.sizes.push(1);
                template.kinds.push(PhaseClusterTemplateKind::Missing);
                marker += 1;
            }
        }
    }
    Ok(template)
}

fn build_clusters_from_template_into(
    template: &PhaseClusterTemplate,
    input: &PhaseWindow,
    sample: usize,
    finished: &[u64],
    any_unphased_heterozygote: bool,
    masked_heterozygotes: &[u64],
    clusters: &mut Vec<PhaseCluster>,
) -> Result<(), String> {
    if template.sizes.len() != template.kinds.len() {
        return Err("phase cluster template dimensions differ".to_owned());
    }
    if template.sample != sample {
        return Err("phase cluster template belongs to a different sample".to_owned());
    }
    clusters.clear();
    clusters.reserve(template.sizes.len());
    let mut start = 0usize;
    let mut first_het_seen = false;
    for (&size, &template_kind) in template.sizes.iter().zip(&template.kinds) {
        let end = start
            .checked_add(usize::from(size))
            .ok_or_else(|| "phase cluster template endpoint overflow".to_owned())?;
        if end > input.rows.len() {
            return Err("phase cluster template exceeds the phase window".to_owned());
        }
        let kind = match template_kind {
            PhaseClusterTemplateKind::Homozygous => ClusterKind::Homozygous,
            PhaseClusterTemplateKind::Missing => ClusterKind::Missing,
            PhaseClusterTemplateKind::Heterozygous => {
                let unresolved = any_unphased_heterozygote
                    && start >= input.incoming_overlap_len
                    && first_het_seen
                    && !bit_is_set(finished, start);
                first_het_seen = true;
                if unresolved && bit_is_set(masked_heterozygotes, start) {
                    ClusterKind::MaskedHeterozygote
                } else {
                    ClusterKind::Heterozygous {
                        unphased: unresolved,
                    }
                }
            }
        };
        clusters.push(PhaseCluster { start, end, kind });
        start = end;
    }
    if start != input.rows.len() {
        return Err("phase cluster template does not cover the phase window".to_owned());
    }
    if clusters.is_empty() {
        return Err("phase cluster template has no clusters".to_owned());
    }
    Ok(())
}

fn build_clusters(
    input: &PhaseWindow,
    sample: usize,
    finished: &[u64],
    any_unphased_heterozygote: bool,
    masked_heterozygotes: &[u64],
) -> Vec<PhaseCluster> {
    let mut clusters = Vec::new();
    build_clusters_into(
        input,
        sample,
        finished,
        any_unphased_heterozygote,
        masked_heterozygotes,
        &mut clusters,
    );
    clusters
}

fn build_clusters_into(
    input: &PhaseWindow,
    sample: usize,
    finished: &[u64],
    any_unphased_heterozygote: bool,
    masked_heterozygotes: &[u64],
    clusters: &mut Vec<PhaseCluster>,
) {
    clusters.clear();
    let mut marker = 0usize;
    let mut first_het_seen = false;
    while marker < input.rows.len() {
        let call = input.rows[marker].calls.get(sample).unwrap();
        match call.alleles {
            [Some(a), Some(b)] if a == b => {
                let start = marker;
                marker += 1;
                while marker < input.rows.len() {
                    if marker - start >= MAX_CLUSTER_MARKERS
                        || input.cm_positions[marker] - input.cm_positions[start] > MAX_CLUSTER_CM
                    {
                        break;
                    }
                    let next = input.rows[marker].calls.get(sample).unwrap();
                    if !matches!(next.alleles, [Some(x), Some(y)] if x == y) {
                        break;
                    }
                    marker += 1;
                }
                clusters.push(PhaseCluster {
                    start,
                    end: marker,
                    kind: ClusterKind::Homozygous,
                });
            }
            [Some(_), Some(_)] => {
                let unresolved = any_unphased_heterozygote
                    && marker >= input.incoming_overlap_len
                    && first_het_seen
                    && !bit_is_set(finished, marker);
                first_het_seen = true;
                clusters.push(PhaseCluster {
                    start: marker,
                    end: marker + 1,
                    kind: if unresolved && bit_is_set(masked_heterozygotes, marker) {
                        ClusterKind::MaskedHeterozygote
                    } else {
                        ClusterKind::Heterozygous {
                            unphased: unresolved,
                        }
                    },
                });
                marker += 1;
            }
            _ => {
                clusters.push(PhaseCluster {
                    start: marker,
                    end: marker + 1,
                    kind: ClusterKind::Missing,
                });
                marker += 1;
            }
        }
    }
}

#[inline(always)]
fn is_unphased_event(cluster: PhaseCluster) -> bool {
    matches!(cluster.kind, ClusterKind::Heterozygous { unphased: true })
}

#[inline]
fn cluster_kind_name(kind: ClusterKind) -> &'static str {
    match kind {
        ClusterKind::Homozygous => "homozygous",
        ClusterKind::Heterozygous { unphased: true } => "unphased_heterozygote",
        ClusterKind::Heterozygous { unphased: false } => "phased_heterozygote",
        ClusterKind::MaskedHeterozygote => "masked_heterozygote",
        ClusterKind::Missing => "missing",
    }
}

#[inline(always)]
fn bit_is_set(bits: &[u64], marker: usize) -> bool {
    bits.get(marker / 64)
        .is_some_and(|word| word & (1u64 << (marker % 64)) != 0)
}

fn cluster_transition_probability(
    cm_positions: &[f64],
    previous_marker: usize,
    current_marker: usize,
    state_count: usize,
    recombination_intensity: f32,
) -> f32 {
    if state_count <= 1
        || current_marker <= previous_marker
        || current_marker >= cm_positions.len()
        || !recombination_intensity.is_finite()
        || recombination_intensity <= 0.0
    {
        return 0.0;
    }
    let mut probability_no_recombination = 1.0f32;
    for marker in previous_marker + 1..=current_marker {
        let distance_cm = (cm_positions[marker] - cm_positions[marker - 1]) as f32;
        let marker_probability =
            -(-f64::from(recombination_intensity) * f64::from(distance_cm)).exp_m1() as f32;
        probability_no_recombination *= 1.0f32 - marker_probability;
    }
    1.0f32 - probability_no_recombination
}

pub(crate) fn marker_no_recombination_probabilities(
    cm_positions: &[f64],
    recombination_intensity: f32,
) -> Result<Vec<f32>, String> {
    if cm_positions.is_empty() {
        return Err("phase transition map is empty".to_owned());
    }
    if !recombination_intensity.is_finite() || recombination_intensity <= 0.0 {
        return Err("phase recombination intensity must be positive and finite".to_owned());
    }
    if cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("phase transition map must be finite and nondecreasing".to_owned());
    }
    let mut factors = vec![1.0f32; cm_positions.len()];
    for marker in 1..cm_positions.len() {
        let distance_cm = (cm_positions[marker] - cm_positions[marker - 1]) as f32;
        let marker_probability =
            -(-f64::from(recombination_intensity) * f64::from(distance_cm)).exp_m1() as f32;
        factors[marker] = 1.0f32 - marker_probability;
    }
    Ok(factors)
}

#[inline]
fn cluster_transition_probability_cached(
    factors: &[f32],
    previous_marker: usize,
    current_marker: usize,
    state_count: usize,
) -> f32 {
    if state_count <= 1 || current_marker <= previous_marker || current_marker >= factors.len() {
        return 0.0;
    }
    let mut probability_no_recombination = 1.0f32;
    for &factor in &factors[previous_marker + 1..=current_marker] {
        probability_no_recombination *= factor;
    }
    1.0f32 - probability_no_recombination
}

fn write_beta_trace_row(
    contents: &mut String,
    cluster: usize,
    stage: &str,
    beta: &[f32],
    state_count: usize,
) {
    use std::fmt::Write as _;
    write!(contents, "{cluster}\t{stage}").unwrap();
    for track in 0..3 {
        let start = track * state_count;
        let end = start + state_count;
        let sum = beta[start..end]
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>();
        let scale = if sum.is_finite() && sum > f64::MIN_POSITIVE {
            (1.0 / sum) as f32
        } else {
            1.0
        };
        contents.push('\t');
        for state in 0..state_count {
            if state > 0 {
                contents.push(',');
            }
            write!(contents, "{}", beta[start + state] * scale).unwrap();
        }
    }
    contents.push('\n');
}

fn backward_project(
    beta_after: &[f32],
    projected: &mut [f32],
    cluster: PhaseCluster,
    switch: f32,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    panel: &CompositePanel,
    err: f64,
    emission_scratch: &mut [f32],
    mismatch_scratch: &mut [[u8; 3]],
    packed_emission_scratch: &mut PackedPhaseEmissionScratch,
) {
    fill_cluster_emissions(
        emission_scratch,
        cluster,
        false,
        input,
        snapshot,
        sample,
        panel,
        err,
        mismatch_scratch,
        packed_emission_scratch,
    );
    backward_project_from_emissions(beta_after, projected, emission_scratch, switch);
}

#[inline]
fn backward_project_from_codes(
    beta_after: &[f32],
    projected: &mut [f32],
    codes: &[u8],
    cluster_offset: usize,
    cluster_kind: ClusterKind,
    switch: f32,
    probabilities: [f32; 2],
) {
    let state_count = beta_after.len() / 3;
    let probability_table = [1.0, probabilities[0], probabilities[1]];
    debug_assert_eq!(beta_after.len(), projected.len());
    for track in 0..3 {
        let begin = track * state_count;
        let end = begin + state_count;
        let mut emission_sum = 0.0f32;
        if let Some(row) = emission_row_offset(cluster_kind, cluster_offset, state_count, track) {
            for state in 0..state_count {
                let value =
                    beta_after[begin + state] * probability_table[codes[row + state] as usize];
                projected[begin + state] = value;
                emission_sum += value;
            }
        } else {
            for state in begin..end {
                let value = beta_after[state] * probabilities[0];
                projected[state] = value;
                emission_sum += value;
            }
        }
        let shift = switch / state_count as f32;
        let scale = (1.0f32 - switch) / emission_sum;
        for probability in &mut projected[begin..end] {
            *probability = scale * *probability + shift;
        }
    }
}

#[inline]
fn project_coded_emissions(
    beta_after: &[f32],
    projected: &mut [f32],
    codes: &[u8],
    row: Option<usize>,
    probabilities: [f32; 2],
    avx2: bool,
    complete: bool,
) {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    debug_assert_eq!(beta_after.len(), projected.len());
    if let Some(row) = row {
        let codes = &codes[row..row + beta_after.len()];
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            unsafe {
                if complete {
                    project_binary_emissions_avx2(
                        beta_after,
                        projected,
                        codes,
                        probabilities[0],
                        probabilities[1],
                    );
                } else {
                    project_coded_emissions_avx2(
                        beta_after,
                        projected,
                        codes,
                        probabilities[0],
                        probabilities[1],
                    );
                }
            }
            return;
        }
        for state in 0..beta_after.len() {
            let emission = if complete {
                if codes[state] == EMISSION_MATCH {
                    probabilities[0]
                } else {
                    probabilities[1]
                }
            } else {
                match codes[state] {
                    EMISSION_MATCH => probabilities[0],
                    EMISSION_MISMATCH => probabilities[1],
                    _ => 1.0,
                }
            };
            projected[state] = beta_after[state] * emission;
        }
    } else {
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            unsafe {
                project_constant_emissions_avx2(beta_after, projected, probabilities[0]);
            }
            return;
        }
        for state in 0..beta_after.len() {
            projected[state] = beta_after[state] * probabilities[0];
        }
    }
}

#[inline]
fn normalize_projected_codes(projected: &mut [f32], switch: f32, sum: f32, avx2: bool) {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    let shift = switch / projected.len() as f32;
    let scale = (1.0f32 - switch) / sum;
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        unsafe {
            normalize_projected_codes_avx2(projected, scale, shift);
        }
        return;
    }
    for probability in projected {
        *probability = scale * *probability + shift;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn phase_avx2_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::env::var_os("JANUSX_IMP_DISABLE_SIMD").is_none()
            && std::arch::is_x86_feature_detected!("avx2")
    })
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn phase_avx2_available() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn emission_vector(
    codes: &[u8],
    offset: usize,
    match_probability: f32,
    mismatch_probability: f32,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let code_bytes = _mm_loadl_epi64(codes.as_ptr().add(offset) as *const __m128i);
    let code_values = _mm256_cvtepu8_epi32(code_bytes);
    let matches = _mm256_cmpeq_epi32(code_values, _mm256_set1_epi32(i32::from(EMISSION_MATCH)));
    let mismatches =
        _mm256_cmpeq_epi32(code_values, _mm256_set1_epi32(i32::from(EMISSION_MISMATCH)));
    let ones = _mm256_set1_ps(1.0);
    let emissions = _mm256_blendv_ps(
        ones,
        _mm256_set1_ps(match_probability),
        _mm256_castsi256_ps(matches),
    );
    _mm256_blendv_ps(
        emissions,
        _mm256_set1_ps(mismatch_probability),
        _mm256_castsi256_ps(mismatches),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn project_coded_emissions_avx2(
    beta_after: &[f32],
    projected: &mut [f32],
    codes: &[u8],
    match_probability: f32,
    mismatch_probability: f32,
) {
    use std::arch::x86_64::*;
    let mut state = 0usize;
    while state + 8 <= beta_after.len() {
        let beta = _mm256_loadu_ps(beta_after.as_ptr().add(state));
        let emissions = emission_vector(codes, state, match_probability, mismatch_probability);
        _mm256_storeu_ps(
            projected.as_mut_ptr().add(state),
            _mm256_mul_ps(beta, emissions),
        );
        state += 8;
    }
    for state in state..beta_after.len() {
        let emission = match codes[state] {
            EMISSION_MATCH => match_probability,
            EMISSION_MISMATCH => mismatch_probability,
            _ => 1.0,
        };
        projected[state] = beta_after[state] * emission;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn project_binary_emissions_avx2(
    beta_after: &[f32],
    projected: &mut [f32],
    codes: &[u8],
    match_probability: f32,
    mismatch_probability: f32,
) {
    use std::arch::x86_64::*;
    let match_vector = _mm256_set1_ps(match_probability);
    let mismatch_vector = _mm256_set1_ps(mismatch_probability);
    let mut state = 0usize;
    while state + 8 <= beta_after.len() {
        let beta = _mm256_loadu_ps(beta_after.as_ptr().add(state));
        let code_bytes = _mm_loadl_epi64(codes.as_ptr().add(state) as *const __m128i);
        let code_values = _mm256_cvtepu8_epi32(code_bytes);
        let mismatch_mask =
            _mm256_cmpeq_epi32(code_values, _mm256_set1_epi32(i32::from(EMISSION_MISMATCH)));
        let emissions = _mm256_blendv_ps(
            match_vector,
            mismatch_vector,
            _mm256_castsi256_ps(mismatch_mask),
        );
        _mm256_storeu_ps(
            projected.as_mut_ptr().add(state),
            _mm256_mul_ps(beta, emissions),
        );
        state += 8;
    }
    for state in state..beta_after.len() {
        let emission = if codes[state] == EMISSION_MATCH {
            match_probability
        } else {
            mismatch_probability
        };
        projected[state] = beta_after[state] * emission;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn project_constant_emissions_avx2(
    beta_after: &[f32],
    projected: &mut [f32],
    emission: f32,
) {
    use std::arch::x86_64::*;
    let emission_value = emission;
    let emission = _mm256_set1_ps(emission);
    let mut state = 0usize;
    while state + 8 <= beta_after.len() {
        let beta = _mm256_loadu_ps(beta_after.as_ptr().add(state));
        _mm256_storeu_ps(
            projected.as_mut_ptr().add(state),
            _mm256_mul_ps(beta, emission),
        );
        state += 8;
    }
    for state in state..beta_after.len() {
        projected[state] = beta_after[state] * emission_value;
    }
}

#[inline]
fn project_coded_emissions_in_place(
    probabilities: &mut [f32],
    codes: &[u8],
    row: Option<usize>,
    emission_probabilities: [f32; 2],
    avx2: bool,
    complete: bool,
) {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    if let Some(row) = row {
        let codes = &codes[row..row + probabilities.len()];
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            unsafe {
                if complete {
                    project_binary_emissions_in_place_avx2(
                        probabilities,
                        codes,
                        emission_probabilities[0],
                        emission_probabilities[1],
                    );
                } else {
                    project_coded_emissions_in_place_avx2(
                        probabilities,
                        codes,
                        emission_probabilities[0],
                        emission_probabilities[1],
                    );
                }
            }
            return;
        }
        for (probability, &code) in probabilities.iter_mut().zip(codes) {
            let emission = if complete {
                if code == EMISSION_MATCH {
                    emission_probabilities[0]
                } else {
                    emission_probabilities[1]
                }
            } else {
                match code {
                    EMISSION_MATCH => emission_probabilities[0],
                    EMISSION_MISMATCH => emission_probabilities[1],
                    _ => 1.0,
                }
            };
            *probability *= emission;
        }
    } else {
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            unsafe {
                project_constant_emissions_in_place_avx2(probabilities, emission_probabilities[0]);
            }
            return;
        }
        for probability in probabilities {
            *probability *= emission_probabilities[0];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn project_binary_emissions_in_place_avx2(
    probabilities: &mut [f32],
    codes: &[u8],
    match_probability: f32,
    mismatch_probability: f32,
) {
    use std::arch::x86_64::*;
    let match_vector = _mm256_set1_ps(match_probability);
    let mismatch_vector = _mm256_set1_ps(mismatch_probability);
    let mismatch_code = _mm256_set1_epi32(i32::from(EMISSION_MISMATCH));
    let mut state = 0usize;
    while state + 8 <= probabilities.len() {
        let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
        let code_bytes = _mm_loadl_epi64(codes.as_ptr().add(state) as *const __m128i);
        let code_values = _mm256_cvtepu8_epi32(code_bytes);
        let mismatch_mask = _mm256_cmpeq_epi32(code_values, mismatch_code);
        let emissions = _mm256_blendv_ps(
            match_vector,
            mismatch_vector,
            _mm256_castsi256_ps(mismatch_mask),
        );
        _mm256_storeu_ps(
            probabilities.as_mut_ptr().add(state),
            _mm256_mul_ps(values, emissions),
        );
        state += 8;
    }
    for state in state..probabilities.len() {
        let emission = if codes[state] == EMISSION_MATCH {
            match_probability
        } else {
            mismatch_probability
        };
        probabilities[state] *= emission;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn project_coded_emissions_in_place_avx2(
    probabilities: &mut [f32],
    codes: &[u8],
    match_probability: f32,
    mismatch_probability: f32,
) {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_ps(1.0);
    let matches = _mm256_set1_epi32(i32::from(EMISSION_MATCH));
    let mismatches = _mm256_set1_epi32(i32::from(EMISSION_MISMATCH));
    let match_vector = _mm256_set1_ps(match_probability);
    let mismatch_vector = _mm256_set1_ps(mismatch_probability);
    let mut state = 0usize;
    while state + 8 <= probabilities.len() {
        let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
        let code_bytes = _mm_loadl_epi64(codes.as_ptr().add(state) as *const __m128i);
        let code_values = _mm256_cvtepu8_epi32(code_bytes);
        let match_mask = _mm256_cmpeq_epi32(code_values, matches);
        let mismatch_mask = _mm256_cmpeq_epi32(code_values, mismatches);
        let emissions = _mm256_blendv_ps(ones, match_vector, _mm256_castsi256_ps(match_mask));
        let emissions = _mm256_blendv_ps(
            emissions,
            mismatch_vector,
            _mm256_castsi256_ps(mismatch_mask),
        );
        _mm256_storeu_ps(
            probabilities.as_mut_ptr().add(state),
            _mm256_mul_ps(values, emissions),
        );
        state += 8;
    }
    for state in state..probabilities.len() {
        let emission = match codes[state] {
            EMISSION_MATCH => match_probability,
            EMISSION_MISMATCH => mismatch_probability,
            _ => 1.0,
        };
        probabilities[state] *= emission;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn project_constant_emissions_in_place_avx2(probabilities: &mut [f32], emission: f32) {
    use std::arch::x86_64::*;
    let scalar_emission = emission;
    let emission = _mm256_set1_ps(emission);
    let mut state = 0usize;
    while state + 8 <= probabilities.len() {
        let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
        _mm256_storeu_ps(
            probabilities.as_mut_ptr().add(state),
            _mm256_mul_ps(values, emission),
        );
        state += 8;
    }
    for probability in &mut probabilities[state..] {
        *probability *= scalar_emission;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_projected_codes_avx2(projected: &mut [f32], scale: f32, shift: f32) {
    use std::arch::x86_64::*;
    let scalar_scale = scale;
    let scalar_shift = shift;
    let scale = _mm256_set1_ps(scale);
    let shift = _mm256_set1_ps(shift);
    let mut state = 0usize;
    while state + 8 <= projected.len() {
        let values = _mm256_loadu_ps(projected.as_ptr().add(state));
        let updated = _mm256_add_ps(_mm256_mul_ps(scale, values), shift);
        _mm256_storeu_ps(projected.as_mut_ptr().add(state), updated);
        state += 8;
    }
    for value in &mut projected[state..] {
        *value = scalar_scale * *value + scalar_shift;
    }
}

#[inline]
fn sum_lane(lane: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &value in lane {
        sum += value;
    }
    sum
}

#[inline]
fn sum_lanes_2(lanes: [&[f32]; 2]) -> [f32; 2] {
    let common = lanes[0].len().min(lanes[1].len());
    let mut sums = [0.0f32; 2];
    for state in 0..common {
        sums[0] += lanes[0][state];
        sums[1] += lanes[1][state];
    }
    for (lane, sum) in lanes.into_iter().zip(&mut sums) {
        for &value in &lane[common..] {
            *sum += value;
        }
    }
    sums
}

#[inline]
fn sum_lanes_3(lanes: [&[f32]; 3]) -> [f32; 3] {
    let common = lanes[0].len().min(lanes[1].len()).min(lanes[2].len());
    let mut sums = [0.0f32; 3];
    for ((lane0, lane1), lane2) in lanes[0][..common]
        .iter()
        .zip(&lanes[1][..common])
        .zip(&lanes[2][..common])
    {
        sums[0] += *lane0;
        sums[1] += *lane1;
        sums[2] += *lane2;
    }
    for (lane, sum) in lanes.into_iter().zip(&mut sums) {
        for &value in &lane[common..] {
            *sum += value;
        }
    }
    sums
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn sum_lanes_6(
    lane00: &[f32],
    lane01: &[f32],
    lane02: &[f32],
    lane10: &[f32],
    lane11: &[f32],
    lane12: &[f32],
) -> [f32; 6] {
    let common = lane00
        .len()
        .min(lane01.len())
        .min(lane02.len())
        .min(lane10.len())
        .min(lane11.len())
        .min(lane12.len());
    let mut sums = [0.0f32; 6];
    for state in 0..common {
        sums[0] += lane00[state];
        sums[1] += lane01[state];
        sums[2] += lane02[state];
        sums[3] += lane10[state];
        sums[4] += lane11[state];
        sums[5] += lane12[state];
    }
    for (lane, sum) in [lane00, lane01, lane02, lane10, lane11, lane12]
        .into_iter()
        .zip(&mut sums)
    {
        for &value in &lane[common..] {
            *sum += value;
        }
    }
    sums
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn backward_project_single_from_codes(
    probabilities: &mut [f32],
    codes: &[u8],
    cluster_offset: usize,
    cluster_kind: ClusterKind,
    switch: f32,
    emission_probabilities: [f32; 2],
    complete: bool,
    avx2: bool,
) {
    let state_count = probabilities.len();
    let row = emission_row_offset(cluster_kind, cluster_offset, state_count, 0);
    project_coded_emissions_in_place(
        probabilities,
        codes,
        row,
        emission_probabilities,
        avx2,
        complete,
    );
    let sum = sum_lane(probabilities);
    normalize_projected_codes(probabilities, switch, sum, avx2);
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn backward_project_two_from_codes(
    probabilities0: &mut [f32],
    codes0: &[u8],
    cluster_offset0: usize,
    cluster_kind0: ClusterKind,
    switch0: f32,
    emission_probabilities0: [f32; 2],
    complete0: bool,
    probabilities1: &mut [f32],
    codes1: &[u8],
    cluster_offset1: usize,
    cluster_kind1: ClusterKind,
    switch1: f32,
    emission_probabilities1: [f32; 2],
    complete1: bool,
    avx2: bool,
) {
    let state_count0 = probabilities0.len();
    let state_count1 = probabilities1.len();
    project_coded_emissions_in_place(
        probabilities0,
        codes0,
        emission_row_offset(cluster_kind0, cluster_offset0, state_count0, 0),
        emission_probabilities0,
        avx2,
        complete0,
    );
    project_coded_emissions_in_place(
        probabilities1,
        codes1,
        emission_row_offset(cluster_kind1, cluster_offset1, state_count1, 0),
        emission_probabilities1,
        avx2,
        complete1,
    );
    let sums = sum_lanes_2([probabilities0, probabilities1]);
    normalize_projected_codes(probabilities0, switch0, sums[0], avx2);
    normalize_projected_codes(probabilities1, switch1, sums[1], avx2);
}

#[inline]
fn backward_project_from_codes_three(
    probabilities_by_track: &mut [f32],
    codes: &[u8],
    cluster_offset: usize,
    cluster_kind: ClusterKind,
    switch: f32,
    probabilities: [f32; 2],
    complete: bool,
    avx2: bool,
) {
    let state_count = probabilities_by_track.len() / 3;
    let rows = [
        emission_row_offset(cluster_kind, cluster_offset, state_count, 0),
        emission_row_offset(cluster_kind, cluster_offset, state_count, 1),
        emission_row_offset(cluster_kind, cluster_offset, state_count, 2),
    ];
    let (probabilities01, probabilities2) = probabilities_by_track.split_at_mut(2 * state_count);
    let (probabilities0, probabilities1) = probabilities01.split_at_mut(state_count);
    project_coded_emissions_in_place(
        probabilities0,
        codes,
        rows[0],
        probabilities,
        avx2,
        complete,
    );
    project_coded_emissions_in_place(
        probabilities1,
        codes,
        rows[1],
        probabilities,
        avx2,
        complete,
    );
    project_coded_emissions_in_place(
        probabilities2,
        codes,
        rows[2],
        probabilities,
        avx2,
        complete,
    );
    let sums = sum_lanes_3([probabilities0, probabilities1, probabilities2]);
    normalize_projected_codes(probabilities0, switch, sums[0], avx2);
    normalize_projected_codes(probabilities1, switch, sums[1], avx2);
    normalize_projected_codes(probabilities2, switch, sums[2], avx2);
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn backward_project_event_from_codes_three(
    beta_after: &[f32],
    projected: &mut [f32],
    codes: &[u8],
    cluster_offset: usize,
    cluster_kind: ClusterKind,
    switch: f32,
    probabilities: [f32; 2],
    complete: bool,
    avx2: bool,
) {
    let state_count = beta_after.len() / 3;
    let rows = [
        emission_row_offset(cluster_kind, cluster_offset, state_count, 0),
        emission_row_offset(cluster_kind, cluster_offset, state_count, 1),
        emission_row_offset(cluster_kind, cluster_offset, state_count, 2),
    ];
    let (projected01, projected2) = projected.split_at_mut(2 * state_count);
    let (projected0, projected1) = projected01.split_at_mut(state_count);
    project_coded_emissions(
        &beta_after[..state_count],
        projected0,
        codes,
        rows[0],
        probabilities,
        avx2,
        complete,
    );
    project_coded_emissions(
        &beta_after[state_count..2 * state_count],
        projected1,
        codes,
        rows[1],
        probabilities,
        avx2,
        complete,
    );
    project_coded_emissions(
        &beta_after[2 * state_count..],
        projected2,
        codes,
        rows[2],
        probabilities,
        avx2,
        complete,
    );
    let sums = sum_lanes_3([projected0, projected1, projected2]);
    normalize_projected_codes(projected0, switch, sums[0], avx2);
    normalize_projected_codes(projected1, switch, sums[1], avx2);
    normalize_projected_codes(projected2, switch, sums[2], avx2);
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn backward_project_from_codes_six(
    probabilities_by_track0: &mut [f32],
    codes0: &[u8],
    cluster_offset0: usize,
    cluster_kind0: ClusterKind,
    switch0: f32,
    probabilities0: [f32; 2],
    complete0: bool,
    probabilities_by_track1: &mut [f32],
    codes1: &[u8],
    cluster_offset1: usize,
    cluster_kind1: ClusterKind,
    switch1: f32,
    probabilities1: [f32; 2],
    complete1: bool,
    avx2: bool,
) {
    let states0 = probabilities_by_track0.len() / 3;
    let states1 = probabilities_by_track1.len() / 3;
    let rows0 = [
        emission_row_offset(cluster_kind0, cluster_offset0, states0, 0),
        emission_row_offset(cluster_kind0, cluster_offset0, states0, 1),
        emission_row_offset(cluster_kind0, cluster_offset0, states0, 2),
    ];
    let rows1 = [
        emission_row_offset(cluster_kind1, cluster_offset1, states1, 0),
        emission_row_offset(cluster_kind1, cluster_offset1, states1, 1),
        emission_row_offset(cluster_kind1, cluster_offset1, states1, 2),
    ];
    let (probabilities00, probabilities02) = probabilities_by_track0.split_at_mut(2 * states0);
    let (probabilities00, probabilities01) = probabilities00.split_at_mut(states0);
    let (probabilities10, probabilities12) = probabilities_by_track1.split_at_mut(2 * states1);
    let (probabilities10, probabilities11) = probabilities10.split_at_mut(states1);
    project_coded_emissions_in_place(
        probabilities00,
        codes0,
        rows0[0],
        probabilities0,
        avx2,
        complete0,
    );
    project_coded_emissions_in_place(
        probabilities01,
        codes0,
        rows0[1],
        probabilities0,
        avx2,
        complete0,
    );
    project_coded_emissions_in_place(
        probabilities02,
        codes0,
        rows0[2],
        probabilities0,
        avx2,
        complete0,
    );
    project_coded_emissions_in_place(
        probabilities10,
        codes1,
        rows1[0],
        probabilities1,
        avx2,
        complete1,
    );
    project_coded_emissions_in_place(
        probabilities11,
        codes1,
        rows1[1],
        probabilities1,
        avx2,
        complete1,
    );
    project_coded_emissions_in_place(
        probabilities12,
        codes1,
        rows1[2],
        probabilities1,
        avx2,
        complete1,
    );
    let sums = sum_lanes_6(
        probabilities00,
        probabilities01,
        probabilities02,
        probabilities10,
        probabilities11,
        probabilities12,
    );
    normalize_projected_codes(probabilities00, switch0, sums[0], avx2);
    normalize_projected_codes(probabilities01, switch0, sums[1], avx2);
    normalize_projected_codes(probabilities02, switch0, sums[2], avx2);
    normalize_projected_codes(probabilities10, switch1, sums[3], avx2);
    normalize_projected_codes(probabilities11, switch1, sums[4], avx2);
    normalize_projected_codes(probabilities12, switch1, sums[5], avx2);
}

fn backward_project_from_emissions(
    beta_after: &[f32],
    projected: &mut [f32],
    emissions: &[f32],
    switch: f32,
) {
    let state_count = beta_after.len() / 3;
    debug_assert_eq!(beta_after.len(), projected.len());
    debug_assert_eq!(beta_after.len(), emissions.len());
    for track in 0..3 {
        let begin = track * state_count;
        let end = begin + state_count;
        beagle_backward_update_from(
            &beta_after[begin..end],
            &emissions[begin..end],
            &mut projected[begin..end],
            switch,
        );
    }
}

fn beagle_forward_update<F>(
    probabilities: &mut [f32],
    previous_sum: f32,
    switch: f32,
    mut emission: F,
) -> f32
where
    F: FnMut(usize) -> f32,
{
    let state_count = probabilities.len();
    let shift = switch / state_count as f32;
    let scale = (1.0f32 - switch) / previous_sum;
    let mut sum = 0.0f32;
    for (state, probability) in probabilities.iter_mut().enumerate() {
        *probability = emission(state) * (scale * *probability + shift);
        sum += *probability;
    }
    sum
}

#[inline]
fn beagle_forward_update_codes(
    probabilities: &mut [f32],
    codes: &[u8],
    previous_sum: f32,
    switch: f32,
    match_probability: f32,
    mismatch_probability: f32,
) -> f32 {
    let state_count = probabilities.len();
    let shift = switch / state_count as f32;
    let scale = (1.0f32 - switch) / previous_sum;
    let mut sum = 0.0f32;
    for (state, probability) in probabilities.iter_mut().enumerate() {
        let emission = match codes[state] {
            EMISSION_MATCH => match_probability,
            EMISSION_MISMATCH => mismatch_probability,
            _ => 1.0,
        };
        *probability = emission * (scale * *probability + shift);
        sum += *probability;
    }
    sum
}

#[inline]
fn forward_values_from_codes(
    probabilities: &mut [f32],
    codes: &[u8],
    row: Option<usize>,
    scale: f32,
    shift: f32,
    emission_probabilities: [f32; 2],
    avx2: bool,
    complete: bool,
) {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    if let Some(row) = row {
        let codes = &codes[row..row + probabilities.len()];
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            unsafe {
                if complete {
                    forward_binary_values_avx2(
                        probabilities,
                        codes,
                        scale,
                        shift,
                        emission_probabilities[0],
                        emission_probabilities[1],
                    );
                } else {
                    forward_values_from_codes_avx2(
                        probabilities,
                        codes,
                        scale,
                        shift,
                        emission_probabilities[0],
                        emission_probabilities[1],
                    );
                }
            }
            return;
        }
        for (state, probability) in probabilities.iter_mut().enumerate() {
            let emission = if complete {
                if codes[state] == EMISSION_MATCH {
                    emission_probabilities[0]
                } else {
                    emission_probabilities[1]
                }
            } else {
                match codes[state] {
                    EMISSION_MATCH => emission_probabilities[0],
                    EMISSION_MISMATCH => emission_probabilities[1],
                    _ => 1.0,
                }
            };
            *probability = emission * (scale * *probability + shift);
        }
    } else {
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            unsafe {
                forward_constant_values_avx2(
                    probabilities,
                    scale,
                    shift,
                    emission_probabilities[0],
                );
            }
            return;
        }
        for probability in probabilities {
            *probability = emission_probabilities[0] * (scale * *probability + shift);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn forward_values_from_codes_avx2(
    probabilities: &mut [f32],
    codes: &[u8],
    scale: f32,
    shift: f32,
    match_probability: f32,
    mismatch_probability: f32,
) {
    use std::arch::x86_64::*;
    let scale_vector = _mm256_set1_ps(scale);
    let shift_vector = _mm256_set1_ps(shift);
    let mut state = 0usize;
    while state + 8 <= probabilities.len() {
        let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
        let scaled = _mm256_add_ps(_mm256_mul_ps(scale_vector, values), shift_vector);
        let emissions = emission_vector(codes, state, match_probability, mismatch_probability);
        _mm256_storeu_ps(
            probabilities.as_mut_ptr().add(state),
            _mm256_mul_ps(emissions, scaled),
        );
        state += 8;
    }
    for state in state..probabilities.len() {
        let emission = match codes[state] {
            EMISSION_MATCH => match_probability,
            EMISSION_MISMATCH => mismatch_probability,
            _ => 1.0,
        };
        probabilities[state] = emission * (scale * probabilities[state] + shift);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn forward_binary_values_avx2(
    probabilities: &mut [f32],
    codes: &[u8],
    scale: f32,
    shift: f32,
    match_probability: f32,
    mismatch_probability: f32,
) {
    use std::arch::x86_64::*;
    let scale_vector = _mm256_set1_ps(scale);
    let shift_vector = _mm256_set1_ps(shift);
    let match_vector = _mm256_set1_ps(match_probability);
    let mismatch_vector = _mm256_set1_ps(mismatch_probability);
    let mut state = 0usize;
    while state + 8 <= probabilities.len() {
        let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
        let scaled = _mm256_add_ps(_mm256_mul_ps(scale_vector, values), shift_vector);
        let code_bytes = _mm_loadl_epi64(codes.as_ptr().add(state) as *const __m128i);
        let code_values = _mm256_cvtepu8_epi32(code_bytes);
        let mismatch_mask =
            _mm256_cmpeq_epi32(code_values, _mm256_set1_epi32(i32::from(EMISSION_MISMATCH)));
        let emissions = _mm256_blendv_ps(
            match_vector,
            mismatch_vector,
            _mm256_castsi256_ps(mismatch_mask),
        );
        _mm256_storeu_ps(
            probabilities.as_mut_ptr().add(state),
            _mm256_mul_ps(emissions, scaled),
        );
        state += 8;
    }
    for state in state..probabilities.len() {
        let emission = if codes[state] == EMISSION_MATCH {
            match_probability
        } else {
            mismatch_probability
        };
        probabilities[state] = emission * (scale * probabilities[state] + shift);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn forward_constant_values_avx2(
    probabilities: &mut [f32],
    scale: f32,
    shift: f32,
    emission: f32,
) {
    use std::arch::x86_64::*;
    let scale_value = scale;
    let shift_value = shift;
    let emission_value = emission;
    let scale = _mm256_set1_ps(scale);
    let shift = _mm256_set1_ps(shift);
    let emission = _mm256_set1_ps(emission);
    let mut state = 0usize;
    while state + 8 <= probabilities.len() {
        let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
        let scaled = _mm256_add_ps(_mm256_mul_ps(scale, values), shift);
        _mm256_storeu_ps(
            probabilities.as_mut_ptr().add(state),
            _mm256_mul_ps(emission, scaled),
        );
        state += 8;
    }
    for probability in &mut probabilities[state..] {
        *probability = emission_value * (scale_value * *probability + shift_value);
    }
}

#[inline]
fn beagle_forward_update_codes_three(
    probabilities: &mut [f32],
    codes: &[u8],
    rows: [Option<usize>; 3],
    previous_sums: [f32; 3],
    switch: f32,
    emission_probabilities: [f32; 2],
    state_count: usize,
    complete: bool,
    avx2: bool,
) -> [f32; 3] {
    debug_assert_eq!(probabilities.len(), 3 * state_count);
    debug_assert_eq!(rows.len(), 3);
    let shift = switch / state_count as f32;
    let scales = previous_sums.map(|sum| (1.0f32 - switch) / sum);
    let (probabilities01, probabilities2) = probabilities.split_at_mut(2 * state_count);
    let (probabilities0, probabilities1) = probabilities01.split_at_mut(state_count);
    forward_values_from_codes(
        probabilities0,
        codes,
        rows[0],
        scales[0],
        shift,
        emission_probabilities,
        avx2,
        complete,
    );
    forward_values_from_codes(
        probabilities1,
        codes,
        rows[1],
        scales[1],
        shift,
        emission_probabilities,
        avx2,
        complete,
    );
    forward_values_from_codes(
        probabilities2,
        codes,
        rows[2],
        scales[2],
        shift,
        emission_probabilities,
        avx2,
        complete,
    );
    sum_lanes_3([probabilities0, probabilities1, probabilities2])
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn beagle_forward_update_codes_six(
    probabilities0: &mut [f32],
    codes0: &[u8],
    rows0: [Option<usize>; 3],
    previous_sums0: [f32; 3],
    switch0: f32,
    emission_probabilities0: [f32; 2],
    states0: usize,
    probabilities1: &mut [f32],
    codes1: &[u8],
    rows1: [Option<usize>; 3],
    previous_sums1: [f32; 3],
    switch1: f32,
    emission_probabilities1: [f32; 2],
    states1: usize,
    complete0: bool,
    complete1: bool,
    avx2: bool,
) -> ([f32; 3], [f32; 3]) {
    debug_assert_eq!(probabilities0.len(), 3 * states0);
    debug_assert_eq!(probabilities1.len(), 3 * states1);
    let shifts = [switch0 / states0 as f32, switch1 / states1 as f32];
    let scales0 = previous_sums0.map(|sum| (1.0f32 - switch0) / sum);
    let scales1 = previous_sums1.map(|sum| (1.0f32 - switch1) / sum);
    #[cfg(target_arch = "x86_64")]
    if avx2 && complete0 && complete1 {
        let (sums0, sums1) = unsafe {
            beagle_forward_update_codes_six_complete_avx2(
                probabilities0,
                codes0,
                rows0,
                scales0,
                shifts[0],
                emission_probabilities0,
                states0,
                probabilities1,
                codes1,
                rows1,
                scales1,
                shifts[1],
                emission_probabilities1,
                states1,
            )
        };
        return (sums0, sums1);
    }
    let (p00, p02) = probabilities0.split_at_mut(2 * states0);
    let (p00, p01) = p00.split_at_mut(states0);
    let (p10, p12) = probabilities1.split_at_mut(2 * states1);
    let (p10, p11) = p10.split_at_mut(states1);
    forward_values_from_codes(
        p00,
        codes0,
        rows0[0],
        scales0[0],
        shifts[0],
        emission_probabilities0,
        avx2,
        complete0,
    );
    forward_values_from_codes(
        p01,
        codes0,
        rows0[1],
        scales0[1],
        shifts[0],
        emission_probabilities0,
        avx2,
        complete0,
    );
    forward_values_from_codes(
        p02,
        codes0,
        rows0[2],
        scales0[2],
        shifts[0],
        emission_probabilities0,
        avx2,
        complete0,
    );
    forward_values_from_codes(
        p10,
        codes1,
        rows1[0],
        scales1[0],
        shifts[1],
        emission_probabilities1,
        avx2,
        complete1,
    );
    forward_values_from_codes(
        p11,
        codes1,
        rows1[1],
        scales1[1],
        shifts[1],
        emission_probabilities1,
        avx2,
        complete1,
    );
    forward_values_from_codes(
        p12,
        codes1,
        rows1[2],
        scales1[2],
        shifts[1],
        emission_probabilities1,
        avx2,
        complete1,
    );
    let sums = sum_lanes_6(p00, p01, p02, p10, p11, p12);
    ([sums[0], sums[1], sums[2]], [sums[3], sums[4], sums[5]])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn beagle_forward_update_codes_six_complete_avx2(
    probabilities0: &mut [f32],
    codes0: &[u8],
    rows0: [Option<usize>; 3],
    scales0: [f32; 3],
    shift0: f32,
    emission_probabilities0: [f32; 2],
    states0: usize,
    probabilities1: &mut [f32],
    codes1: &[u8],
    rows1: [Option<usize>; 3],
    scales1: [f32; 3],
    shift1: f32,
    emission_probabilities1: [f32; 2],
    states1: usize,
) -> ([f32; 3], [f32; 3]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(probabilities0.len(), 3 * states0);
    debug_assert_eq!(probabilities1.len(), 3 * states1);
    let (p00, p02) = probabilities0.split_at_mut(2 * states0);
    let (p00, p01) = p00.split_at_mut(states0);
    let (p10, p12) = probabilities1.split_at_mut(2 * states1);
    let (p10, p11) = p10.split_at_mut(states1);
    let shift0_vector = _mm256_set1_ps(shift0);
    let shift1_vector = _mm256_set1_ps(shift1);
    let match0_vector = _mm256_set1_ps(emission_probabilities0[0]);
    let mismatch0_vector = _mm256_set1_ps(emission_probabilities0[1]);
    let match1_vector = _mm256_set1_ps(emission_probabilities1[0]);
    let mismatch1_vector = _mm256_set1_ps(emission_probabilities1[1]);
    let update0 = |probabilities: &mut [f32], row: Option<usize>, scale_value: f32| {
        let scale = _mm256_set1_ps(scale_value);
        if let Some(row) = row {
            let codes = &codes0[row..row + probabilities.len()];
            let mut state = 0usize;
            while state + 8 <= probabilities.len() {
                let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
                let scaled = _mm256_add_ps(_mm256_mul_ps(scale, values), shift0_vector);
                let code_bytes = _mm_loadl_epi64(codes.as_ptr().add(state).cast::<__m128i>());
                let code_values = _mm256_cvtepu8_epi32(code_bytes);
                let mismatch_mask = _mm256_cmpeq_epi32(
                    code_values,
                    _mm256_set1_epi32(i32::from(EMISSION_MISMATCH)),
                );
                let emissions = _mm256_blendv_ps(
                    match0_vector,
                    mismatch0_vector,
                    _mm256_castsi256_ps(mismatch_mask),
                );
                _mm256_storeu_ps(
                    probabilities.as_mut_ptr().add(state),
                    _mm256_mul_ps(emissions, scaled),
                );
                state += 8;
            }
            for state in state..probabilities.len() {
                let emission = if codes[state] == EMISSION_MATCH {
                    emission_probabilities0[0]
                } else {
                    emission_probabilities0[1]
                };
                probabilities[state] = emission
                    * (scale_value * probabilities[state] + _mm256_cvtss_f32(shift0_vector));
            }
        } else {
            let emission = _mm256_set1_ps(emission_probabilities0[0]);
            let mut state = 0usize;
            while state + 8 <= probabilities.len() {
                let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
                let scaled = _mm256_add_ps(_mm256_mul_ps(scale, values), shift0_vector);
                _mm256_storeu_ps(
                    probabilities.as_mut_ptr().add(state),
                    _mm256_mul_ps(emission, scaled),
                );
                state += 8;
            }
            for state in state..probabilities.len() {
                probabilities[state] = emission_probabilities0[0]
                    * (scale_value * probabilities[state] + _mm256_cvtss_f32(shift0_vector));
            }
        }
    };
    update0(p00, rows0[0], scales0[0]);
    update0(p01, rows0[1], scales0[1]);
    update0(p02, rows0[2], scales0[2]);
    let update1 = |probabilities: &mut [f32], row: Option<usize>, scale_value: f32| {
        let scale = _mm256_set1_ps(scale_value);
        if let Some(row) = row {
            let codes = &codes1[row..row + probabilities.len()];
            let mut state = 0usize;
            while state + 8 <= probabilities.len() {
                let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
                let scaled = _mm256_add_ps(_mm256_mul_ps(scale, values), shift1_vector);
                let code_bytes = _mm_loadl_epi64(codes.as_ptr().add(state).cast::<__m128i>());
                let code_values = _mm256_cvtepu8_epi32(code_bytes);
                let mismatch_mask = _mm256_cmpeq_epi32(
                    code_values,
                    _mm256_set1_epi32(i32::from(EMISSION_MISMATCH)),
                );
                let emissions = _mm256_blendv_ps(
                    match1_vector,
                    mismatch1_vector,
                    _mm256_castsi256_ps(mismatch_mask),
                );
                _mm256_storeu_ps(
                    probabilities.as_mut_ptr().add(state),
                    _mm256_mul_ps(emissions, scaled),
                );
                state += 8;
            }
            for state in state..probabilities.len() {
                let emission = if codes[state] == EMISSION_MATCH {
                    emission_probabilities1[0]
                } else {
                    emission_probabilities1[1]
                };
                probabilities[state] = emission
                    * (scale_value * probabilities[state] + _mm256_cvtss_f32(shift1_vector));
            }
        } else {
            let emission = _mm256_set1_ps(emission_probabilities1[0]);
            let mut state = 0usize;
            while state + 8 <= probabilities.len() {
                let values = _mm256_loadu_ps(probabilities.as_ptr().add(state));
                let scaled = _mm256_add_ps(_mm256_mul_ps(scale, values), shift1_vector);
                _mm256_storeu_ps(
                    probabilities.as_mut_ptr().add(state),
                    _mm256_mul_ps(emission, scaled),
                );
                state += 8;
            }
            for state in state..probabilities.len() {
                probabilities[state] = emission_probabilities1[0]
                    * (scale_value * probabilities[state] + _mm256_cvtss_f32(shift1_vector));
            }
        }
    };
    update1(p10, rows1[0], scales1[0]);
    update1(p11, rows1[1], scales1[1]);
    update1(p12, rows1[2], scales1[2]);
    let sums = sum_lanes_6(p00, p01, p02, p10, p11, p12);
    ([sums[0], sums[1], sums[2]], [sums[3], sums[4], sums[5]])
}

fn beagle_backward_update<E>(probabilities: &mut [f32], switch: f32, mut emission: E)
where
    E: FnMut(usize) -> f32,
{
    let state_count = probabilities.len();
    let mut emission_sum = 0.0f32;
    for (state, probability) in probabilities.iter_mut().enumerate() {
        *probability *= emission(state);
        emission_sum += *probability;
    }
    let shift = switch / state_count as f32;
    let scale = (1.0f32 - switch) / emission_sum;
    for probability in probabilities {
        *probability = scale * *probability + shift;
    }
}

#[inline]
fn beagle_backward_update_from(
    beta_after: &[f32],
    emissions: &[f32],
    projected: &mut [f32],
    switch: f32,
) {
    debug_assert_eq!(beta_after.len(), emissions.len());
    debug_assert_eq!(beta_after.len(), projected.len());
    let state_count = beta_after.len();
    let mut emission_sum = 0.0f32;
    for state in 0..state_count {
        let value = beta_after[state] * emissions[state];
        projected[state] = value;
        emission_sum += value;
    }
    let shift = switch / state_count as f32;
    let scale = (1.0f32 - switch) / emission_sum;
    for probability in projected {
        *probability = scale * *probability + shift;
    }
}

fn cluster_emission(
    cluster: PhaseCluster,
    track: usize,
    swap_haps: bool,
    state: usize,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    panel: &CompositePanel,
    err: f64,
) -> f32 {
    let mismatch_probability = ((cluster.end - cluster.start) as f32 * err as f32).min(0.5f32);
    if matches!(
        cluster.kind,
        ClusterKind::Missing | ClusterKind::MaskedHeterozygote
    ) || (track == 0 && matches!(cluster.kind, ClusterKind::Heterozygous { .. }))
    {
        // Beagle sets mismatch flags to zero for missing/masked clusters and
        // the genotype track at heterozygotes, but still applies the match
        // probability to each of those clusters.
        return (1.0 - mismatch_probability).max(f32::MIN_POSITIVE);
    }
    let mut mismatch = false;
    let mut compared = false;
    for marker in cluster.start..cluster.end {
        let target_allele = match cluster.kind {
            ClusterKind::Homozygous => input.rows[marker].calls.get(sample).unwrap().alleles[0],
            ClusterKind::Heterozygous { .. } => {
                let hap = observation_haplotype(track, swap_haps);
                snapshot.rows[marker].calls.get(sample).unwrap().alleles[hap]
            }
            ClusterKind::Missing => None,
            ClusterKind::MaskedHeterozygote => None,
        };
        let donor_allele = panel.haplotypes.allele(marker, state);
        if let (Some(target_allele), Some(donor_allele)) = (target_allele, donor_allele) {
            compared = true;
            mismatch |= target_allele != donor_allele;
        }
    }
    if !compared {
        return 1.0;
    }
    if mismatch {
        mismatch_probability.max(f32::MIN_POSITIVE)
    } else {
        (1.0 - mismatch_probability).max(f32::MIN_POSITIVE)
    }
}

fn fill_state_major_biallelic_cluster_emissions(
    emissions: &mut [f32],
    cluster: PhaseCluster,
    swap_haps: bool,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    panel: &CompositePanel,
    err: f64,
    scratch: &mut PackedPhaseEmissionScratch,
) -> bool {
    let state_count = panel.haplotypes.n_haplotypes;
    if !panel.haplotypes.is_state_major()
        || panel.haplotypes.bits_per_allele != 1
        || emissions.len() != 3 * state_count
        || !matches!(
            cluster.kind,
            ClusterKind::Homozygous | ClusterKind::Heterozygous { .. }
        )
    {
        return false;
    }

    let marker_count = cluster.end - cluster.start;
    let marker_word_count = marker_count.div_ceil(64);
    for track in 0..3 {
        scratch.target_observed[track].resize(marker_word_count, 0);
        scratch.target_observed[track].fill(0);
        scratch.target_one[track].resize(marker_word_count, 0);
        scratch.target_one[track].fill(0);
    }

    let heterozygous = matches!(cluster.kind, ClusterKind::Heterozygous { .. });
    let hap0 = observation_haplotype(1, swap_haps);
    let hap1 = observation_haplotype(2, swap_haps);
    for marker in cluster.start..cluster.end {
        let bit = marker - cluster.start;
        let word = bit >> 6;
        let mask = 1u64 << (bit & 63);
        if heterozygous {
            for (track, haplotype) in [(1usize, hap0), (2usize, hap1)] {
                match snapshot.rows[marker].calls.get_allele(sample, haplotype) {
                    Some(0) => scratch.target_observed[track][word] |= mask,
                    Some(1) => {
                        scratch.target_observed[track][word] |= mask;
                        scratch.target_one[track][word] |= mask;
                    }
                    Some(_) => return false,
                    None => {}
                }
            }
        } else {
            match input.rows[marker].calls.get_allele(sample, 0) {
                Some(0) => scratch.target_observed[0][word] |= mask,
                Some(1) => {
                    scratch.target_observed[0][word] |= mask;
                    scratch.target_one[0][word] |= mask;
                }
                Some(_) => return false,
                None => {}
            }
        }
    }

    let track_start = if heterozygous { 1 } else { 0 };
    let track_end = if heterozygous { 3 } else { 1 };
    let raw_mismatch = ((cluster.end - cluster.start) as f32 * err as f32).min(0.5f32);
    let mismatch_probability = raw_mismatch.max(f32::MIN_POSITIVE);
    let match_probability = (1.0 - raw_mismatch).max(f32::MIN_POSITIVE);
    if heterozygous {
        emissions[..state_count].fill(match_probability);
    }
    for state in 0..state_count {
        let Some((state_start, allele_bits, missing_bits)) =
            panel.haplotypes.state_major_biallelic_words(state)
        else {
            return false;
        };
        for track in track_start..track_end {
            let mut observed = false;
            let mut mismatched = false;
            for word in 0..marker_word_count {
                let first = word * 64;
                let count = (marker_count - first).min(64);
                let bits = if count == 64 {
                    u64::MAX
                } else {
                    (1u64 << count) - 1
                };
                let source_start = state_start + cluster.start + first;
                let donor_one = read_bit_chunk(allele_bits, source_start, count);
                let donor_missing = read_bit_chunk(missing_bits, source_start, count);
                let compared = scratch.target_observed[track][word] & !donor_missing & bits;
                observed |= compared != 0;
                mismatched |= compared & (donor_one ^ scratch.target_one[track][word]) != 0;
            }
            emissions[track * state_count + state] = if !observed {
                1.0
            } else if mismatched {
                mismatch_probability
            } else {
                match_probability
            };
        }
    }

    if !heterozygous {
        let (track0, rest) = emissions.split_at_mut(state_count);
        let (track1, track2) = rest.split_at_mut(state_count);
        track1.copy_from_slice(track0);
        track2.copy_from_slice(track0);
    }
    true
}

#[inline]
fn mark_mismatch_states(
    codes: &mut [MaybeUninit<u8>],
    donor_columns: &[u64],
    code_base: usize,
    target_column: u64,
    marker_mask: u64,
    avx2: bool,
) {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        unsafe {
            mark_mismatch_states_avx2(codes, donor_columns, code_base, target_column, marker_mask);
        }
        return;
    }
    for (state, &donor_column) in donor_columns.iter().enumerate() {
        if ((donor_column ^ target_column) & marker_mask) != 0 {
            codes[code_base + state].write(EMISSION_MISMATCH);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mark_mismatch_states_avx2(
    codes: &mut [MaybeUninit<u8>],
    donor_columns: &[u64],
    code_base: usize,
    target_column: u64,
    marker_mask: u64,
) {
    use std::arch::x86_64::*;
    let target = _mm256_set1_epi64x(target_column as i64);
    let mask = _mm256_set1_epi64x(marker_mask as i64);
    let zero = _mm256_setzero_si256();
    let mut state = 0usize;
    while state + 4 <= donor_columns.len() {
        let donor = _mm256_loadu_si256(donor_columns.as_ptr().add(state).cast::<__m256i>());
        let differ = _mm256_and_si256(_mm256_xor_si256(donor, target), mask);
        let equal = _mm256_cmpeq_epi64(differ, zero);
        let equal_mask = _mm256_movemask_pd(_mm256_castsi256_pd(equal)) as usize;
        let mut mismatches = (!equal_mask) & 0b1111;
        while mismatches != 0 {
            let lane = mismatches.trailing_zeros() as usize;
            codes[code_base + state + lane].write(EMISSION_MISMATCH);
            mismatches &= mismatches - 1;
        }
        state += 4;
    }
    for (offset, &donor_column) in donor_columns[state..].iter().enumerate() {
        if ((donor_column ^ target_column) & marker_mask) != 0 {
            codes[code_base + state + offset].write(EMISSION_MISMATCH);
        }
    }
}

#[inline]
fn mark_mismatch_state_pairs(
    codes: &mut [MaybeUninit<u8>],
    donor_columns: &[u64],
    first_code_base: usize,
    second_code_base: usize,
    first_target_column: u64,
    second_target_column: u64,
    marker_mask: u64,
    avx2: bool,
) {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        unsafe {
            mark_mismatch_state_pairs_avx2(
                codes,
                donor_columns,
                first_code_base,
                second_code_base,
                first_target_column,
                second_target_column,
                marker_mask,
            );
        }
        return;
    }
    for (state, &donor_column) in donor_columns.iter().enumerate() {
        if ((donor_column ^ first_target_column) & marker_mask) != 0 {
            codes[first_code_base + state].write(EMISSION_MISMATCH);
        }
        if ((donor_column ^ second_target_column) & marker_mask) != 0 {
            codes[second_code_base + state].write(EMISSION_MISMATCH);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mark_mismatch_state_pairs_avx2(
    codes: &mut [MaybeUninit<u8>],
    donor_columns: &[u64],
    first_code_base: usize,
    second_code_base: usize,
    first_target_column: u64,
    second_target_column: u64,
    marker_mask: u64,
) {
    use std::arch::x86_64::*;
    let first_target = _mm256_set1_epi64x(first_target_column as i64);
    let second_target = _mm256_set1_epi64x(second_target_column as i64);
    let mask = _mm256_set1_epi64x(marker_mask as i64);
    let zero = _mm256_setzero_si256();
    let mut state = 0usize;
    while state + 4 <= donor_columns.len() {
        let donor = _mm256_loadu_si256(donor_columns.as_ptr().add(state).cast::<__m256i>());
        let first_differ = _mm256_and_si256(_mm256_xor_si256(donor, first_target), mask);
        let second_differ = _mm256_and_si256(_mm256_xor_si256(donor, second_target), mask);
        let first_equal_mask =
            _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(first_differ, zero)))
                as usize;
        let second_equal_mask =
            _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(second_differ, zero)))
                as usize;
        let mut first_mismatches = (!first_equal_mask) & 0b1111;
        while first_mismatches != 0 {
            let lane = first_mismatches.trailing_zeros() as usize;
            codes[first_code_base + state + lane].write(EMISSION_MISMATCH);
            first_mismatches &= first_mismatches - 1;
        }
        let mut second_mismatches = (!second_equal_mask) & 0b1111;
        while second_mismatches != 0 {
            let lane = second_mismatches.trailing_zeros() as usize;
            codes[second_code_base + state + lane].write(EMISSION_MISMATCH);
            second_mismatches &= second_mismatches - 1;
        }
        state += 4;
    }
    for (offset, &donor_column) in donor_columns[state..].iter().enumerate() {
        if ((donor_column ^ first_target_column) & marker_mask) != 0 {
            codes[first_code_base + state + offset].write(EMISSION_MISMATCH);
        }
        if ((donor_column ^ second_target_column) & marker_mask) != 0 {
            codes[second_code_base + state + offset].write(EMISSION_MISMATCH);
        }
    }
}

/// Fill emission rows through a 64-marker column cache.
///
/// Composite states remain state-major for inexpensive segment copying. This
/// kernel gathers one marker word from every state once, then lets every
/// cluster in that word compare against its target haplotype and update the
/// final compact emission bytes directly.
fn fill_state_major_column_emission_codes(
    codes: &mut [MaybeUninit<u8>],
    clusters: &[PhaseCluster],
    code_offsets: &[usize],
    sample: usize,
    panel: &CompositePanel,
    packed_target: &PackedHaplotypes,
    scratch: &mut PackedPhaseEmissionScratch,
) -> Result<bool, String> {
    let packed = &panel.haplotypes;
    let state_count = packed.n_haplotypes;
    let marker_count = packed.n_markers;
    if !packed.is_complete()
        || !packed.is_state_major()
        || packed.bits_per_allele != 1
        || !packed_target.is_complete()
        || !packed_target.is_state_major()
        || packed_target.bits_per_allele != 1
        || packed_target.n_markers != marker_count
        || packed_target.n_haplotypes <= sample * 2 + 1
        || state_count == 0
        || marker_count == 0
        || clusters.len() != code_offsets.len()
    {
        return Ok(false);
    }
    let rows = clusters
        .iter()
        .map(|cluster| emission_row_count(cluster.kind))
        .sum::<usize>();
    let expected_codes = rows
        .checked_mul(state_count)
        .ok_or_else(|| "column emission dimensions overflow".to_owned())?;
    if codes.len() != expected_codes {
        return Err(format!(
            "column emission output has {} values; expected {expected_codes}",
            codes.len()
        ));
    }
    let mut expected_offset = 0usize;
    for (cluster, &offset) in clusters.iter().zip(code_offsets) {
        if offset != expected_offset {
            return Err("column emission row offsets are inconsistent".to_owned());
        }
        expected_offset = expected_offset
            .checked_add(
                emission_row_count(cluster.kind)
                    .checked_mul(state_count)
                    .ok_or_else(|| "column emission offset overflow".to_owned())?,
            )
            .ok_or_else(|| "column emission offset overflow".to_owned())?;
    }
    if expected_offset != expected_codes {
        return Err("column emission row offsets do not cover the output".to_owned());
    }

    for (cluster, &offset) in clusters.iter().zip(code_offsets) {
        let row_values = emission_row_count(cluster.kind) * state_count;
        if row_values != 0 {
            codes[offset..offset + row_values].fill(MaybeUninit::new(EMISSION_MATCH));
        }
    }
    scratch.donor_column.clear();
    scratch.donor_column.resize(state_count, 0);
    let (_, donor_bits, _) = packed
        .state_major_biallelic_words(0)
        .ok_or_else(|| "column emission panel is not state-major".to_owned())?;
    let (target0_start, target0_bits, _) = packed_target
        .state_major_biallelic_words(sample * 2)
        .ok_or_else(|| "column emission first target haplotype is missing".to_owned())?;
    let (target1_start, target1_bits, _) = packed_target
        .state_major_biallelic_words(sample * 2 + 1)
        .ok_or_else(|| "column emission second target haplotype is missing".to_owned())?;
    let avx2 = phase_avx2_available();

    let mut cluster_start_index = 0usize;
    for word in 0..marker_count.div_ceil(64) {
        let word_start = word * 64;
        let word_count = (marker_count - word_start).min(64);
        let word_end = word_start + word_count;
        while cluster_start_index < clusters.len()
            && clusters[cluster_start_index].end <= word_start
        {
            cluster_start_index += 1;
        }
        for state in 0..state_count {
            scratch.donor_column[state] =
                read_bit_chunk(donor_bits, state * marker_count + word_start, word_count);
        }
        let target0_column = read_bit_chunk(target0_bits, target0_start + word_start, word_count);
        let target1_column = read_bit_chunk(target1_bits, target1_start + word_start, word_count);

        let mut cluster_index = cluster_start_index;
        while let Some(cluster) = clusters.get(cluster_index).copied() {
            if cluster.start >= word_end {
                break;
            }
            let rows = emission_row_count(cluster.kind);
            if rows != 0 {
                let overlap_start = cluster.start.max(word_start);
                let overlap_end = cluster.end.min(word_end);
                let count = overlap_end - overlap_start;
                let shift = overlap_start - word_start;
                let mask = if count == 64 {
                    u64::MAX
                } else {
                    ((1u64 << count) - 1) << shift
                };
                let row_base = code_offsets[cluster_index];
                match rows {
                    1 => {
                        mark_mismatch_states(
                            codes,
                            &scratch.donor_column,
                            row_base,
                            target0_column,
                            mask,
                            avx2,
                        );
                    }
                    2 => {
                        mark_mismatch_state_pairs(
                            codes,
                            &scratch.donor_column,
                            row_base,
                            row_base + state_count,
                            target0_column,
                            target1_column,
                            mask,
                            avx2,
                        );
                    }
                    _ => unreachable!("phase clusters have at most two emission rows"),
                }
            }
            cluster_index += 1;
        }
    }

    Ok(true)
}

/// Beagle shares one mismatch row for a homozygous cluster and two rows for
/// a heterozygous cluster. Missing and masked clusters need no stored row.
fn fill_state_major_emission_codes(
    codes: &mut [MaybeUninit<u8>],
    cluster: PhaseCluster,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    allele_bits: &[u64],
    missing_bits: &[u64],
    panel_marker_count: usize,
    state_count: usize,
    complete: bool,
    packed_target: Option<&PackedHaplotypes>,
    scratch: &mut PackedPhaseEmissionScratch,
) -> Result<(), String> {
    let expected_rows = emission_row_count(cluster.kind);
    debug_assert_eq!(codes.len(), expected_rows * state_count);
    if expected_rows == 0 {
        return Ok(());
    }

    let marker_count = cluster.end - cluster.start;
    if marker_count == 1 {
        let marker = cluster.start;
        if expected_rows == 2 {
            let target0 = snapshot.rows[marker]
                .calls
                .get_allele(sample, observation_haplotype(1, false));
            let target1 = snapshot.rows[marker]
                .calls
                .get_allele(sample, observation_haplotype(2, false));
            if target0.is_some_and(|allele| allele > 1) || target1.is_some_and(|allele| allele > 1)
            {
                return Err("phase emission target is not biallelic".to_owned());
            }
            let mut bit = marker;
            for state in 0..state_count {
                let word = bit >> 6;
                let shift = bit & 63;
                let donor_allele = unsafe { (*allele_bits.get_unchecked(word) >> shift) & 1 };
                if complete {
                    codes[state].write(single_marker_emission_code_for_allele(
                        donor_allele,
                        target0,
                    ));
                    codes[state_count + state].write(single_marker_emission_code_for_allele(
                        donor_allele,
                        target1,
                    ));
                } else {
                    let missing = unsafe { (*missing_bits.get_unchecked(word) >> shift) & 1 } != 0;
                    if missing {
                        codes[state].write(EMISSION_UNOBSERVED);
                        codes[state_count + state].write(EMISSION_UNOBSERVED);
                    } else {
                        codes[state].write(single_marker_emission_code_for_allele(
                            donor_allele,
                            target0,
                        ));
                        codes[state_count + state].write(single_marker_emission_code_for_allele(
                            donor_allele,
                            target1,
                        ));
                    }
                }
                bit += panel_marker_count;
            }
        } else {
            let target = input.rows[marker].calls.get_allele(sample, 0);
            if target.is_some_and(|allele| allele > 1) {
                return Err("phase emission target is not biallelic".to_owned());
            }
            let mut bit = marker;
            for state in 0..state_count {
                codes[state].write(if complete {
                    complete_single_marker_emission_code(allele_bits, bit, target)
                } else {
                    single_marker_emission_code(allele_bits, missing_bits, bit, target)
                });
                bit += panel_marker_count;
            }
        }
        return Ok(());
    }

    let marker_word_count = marker_count.div_ceil(64);
    for track in 0..3 {
        scratch.target_observed[track].resize(marker_word_count, 0);
        scratch.target_observed[track].fill(0);
        scratch.target_one[track].resize(marker_word_count, 0);
        scratch.target_one[track].fill(0);
    }

    let heterozygous = expected_rows == 2;
    let packed_target = packed_target.filter(|packed| {
        packed.is_state_major()
            && packed.is_complete()
            && packed.bits_per_allele == 1
            && packed.n_markers == panel_marker_count
            && packed.n_haplotypes > sample * 2 + 1
    });
    if let Some(packed_target) = packed_target {
        let target_states = if heterozygous {
            [sample * 2, sample * 2 + 1]
        } else {
            [sample * 2, sample * 2]
        };
        for word in 0..marker_word_count {
            let first = word * 64;
            let count = (marker_count - first).min(64);
            let bits = if count == 64 {
                u64::MAX
            } else {
                (1u64 << count) - 1
            };
            let track_start = if heterozygous { 1 } else { 0 };
            let track_end = if heterozygous { 3 } else { 1 };
            for track in track_start..track_end {
                let target_state = target_states[track - track_start];
                let (state_start, target_alleles, _) = packed_target
                    .state_major_biallelic_words(target_state)
                    .ok_or_else(|| "packed phase target layout is invalid".to_owned())?;
                scratch.target_observed[track][word] = bits;
                scratch.target_one[track][word] =
                    read_bit_chunk(target_alleles, state_start + cluster.start + first, count);
            }
        }
    } else {
        let hap0 = observation_haplotype(1, false);
        let hap1 = observation_haplotype(2, false);
        for marker in cluster.start..cluster.end {
            let bit = marker - cluster.start;
            let word = bit >> 6;
            let mask = 1u64 << (bit & 63);
            if heterozygous {
                for (track, haplotype) in [(1usize, hap0), (2usize, hap1)] {
                    match snapshot.rows[marker].calls.get_allele(sample, haplotype) {
                        Some(0) => scratch.target_observed[track][word] |= mask,
                        Some(1) => {
                            scratch.target_observed[track][word] |= mask;
                            scratch.target_one[track][word] |= mask;
                        }
                        Some(_) => return Err("phase emission target is not biallelic".to_owned()),
                        None => {}
                    }
                }
            } else {
                match input.rows[marker].calls.get_allele(sample, 0) {
                    Some(0) => scratch.target_observed[0][word] |= mask,
                    Some(1) => {
                        scratch.target_observed[0][word] |= mask;
                        scratch.target_one[0][word] |= mask;
                    }
                    Some(_) => return Err("phase emission target is not biallelic".to_owned()),
                    None => {}
                }
            }
        }
    }

    for state in 0..state_count {
        let state_start = state * panel_marker_count;
        for track in 0..expected_rows {
            let source_track = if heterozygous { track + 1 } else { 0 };
            let mut observed = false;
            let mut mismatched = false;
            for word in 0..marker_word_count {
                let first = word * 64;
                let count = (marker_count - first).min(64);
                let bits = if count == 64 {
                    u64::MAX
                } else {
                    (1u64 << count) - 1
                };
                let source_start = state_start + cluster.start + first;
                let donor_one = read_bit_chunk(allele_bits, source_start, count);
                let compared = if complete {
                    scratch.target_observed[source_track][word] & bits
                } else {
                    let donor_missing = read_bit_chunk(missing_bits, source_start, count);
                    scratch.target_observed[source_track][word] & !donor_missing & bits
                };
                observed |= compared != 0;
                mismatched |= compared & (donor_one ^ scratch.target_one[source_track][word]) != 0;
            }
            codes[track * state_count + state].write(if !observed {
                EMISSION_UNOBSERVED
            } else if mismatched {
                EMISSION_MISMATCH
            } else {
                EMISSION_MATCH
            });
        }
    }
    Ok(())
}

#[inline(always)]
fn single_marker_emission_code_for_allele(donor_allele: u64, target: Option<u16>) -> u8 {
    let Some(target) = target else {
        return EMISSION_UNOBSERVED;
    };
    if donor_allele == u64::from(target) {
        EMISSION_MATCH
    } else {
        EMISSION_MISMATCH
    }
}

#[inline(always)]
fn single_marker_emission_code(
    allele_bits: &[u64],
    missing_bits: &[u64],
    bit: usize,
    target: Option<u16>,
) -> u8 {
    let Some(target) = target else {
        return EMISSION_UNOBSERVED;
    };
    let word = bit >> 6;
    let shift = bit & 63;
    let missing = unsafe { *missing_bits.get_unchecked(word) };
    if (missing >> shift) & 1 != 0 {
        EMISSION_UNOBSERVED
    } else if ((unsafe { *allele_bits.get_unchecked(word) } >> shift) & 1) == u64::from(target) {
        EMISSION_MATCH
    } else {
        EMISSION_MISMATCH
    }
}

#[inline(always)]
fn complete_single_marker_emission_code(
    allele_bits: &[u64],
    bit: usize,
    target: Option<u16>,
) -> u8 {
    let Some(target) = target else {
        return EMISSION_UNOBSERVED;
    };
    let word = unsafe { *allele_bits.get_unchecked(bit >> 6) };
    if ((word >> (bit & 63)) & 1) == u64::from(target) {
        EMISSION_MATCH
    } else {
        EMISSION_MISMATCH
    }
}

fn fill_biallelic_cluster_emissions(
    emissions: &mut [f32],
    cluster: PhaseCluster,
    swap_haps: bool,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    panel: &CompositePanel,
    err: f64,
    scratch: &mut PackedPhaseEmissionScratch,
) -> bool {
    let state_count = panel.haplotypes.n_haplotypes;
    if panel.haplotypes.is_state_major() {
        return fill_state_major_biallelic_cluster_emissions(
            emissions, cluster, swap_haps, input, snapshot, sample, panel, err, scratch,
        );
    }
    if panel.haplotypes.bits_per_allele != 1
        || emissions.len() != 3 * state_count
        || !matches!(
            cluster.kind,
            ClusterKind::Homozygous | ClusterKind::Heterozygous { .. }
        )
    {
        return false;
    }

    let word_count = state_count.div_ceil(64);
    scratch.observed.resize(word_count, [0; 3]);
    scratch.mismatched.resize(word_count, [0; 3]);
    let heterozygous = matches!(cluster.kind, ClusterKind::Heterozygous { .. });
    if heterozygous {
        for flags in &mut scratch.observed {
            flags[1] = 0;
            flags[2] = 0;
        }
        for flags in &mut scratch.mismatched {
            flags[1] = 0;
            flags[2] = 0;
        }
    } else {
        for flags in &mut scratch.observed {
            flags[0] = 0;
        }
        for flags in &mut scratch.mismatched {
            flags[0] = 0;
        }
    }

    let mismatch_probability = ((cluster.end - cluster.start) as f32 * err as f32)
        .min(0.5f32)
        .max(f32::MIN_POSITIVE);
    let match_probability = (1.0 - mismatch_probability).max(f32::MIN_POSITIVE);
    let hap0 = observation_haplotype(1, swap_haps);
    let hap1 = observation_haplotype(2, swap_haps);

    for marker in cluster.start..cluster.end {
        let targets = match cluster.kind {
            ClusterKind::Homozygous => {
                let allele = input.rows[marker].calls.get_allele(sample, 0);
                [allele, allele, allele]
            }
            ClusterKind::Heterozygous { .. } => [
                None,
                snapshot.rows[marker].calls.get_allele(sample, hap0),
                snapshot.rows[marker].calls.get_allele(sample, hap1),
            ],
            ClusterKind::MaskedHeterozygote | ClusterKind::Missing => return false,
        };
        if targets.iter().flatten().any(|&allele| allele > 1) {
            return false;
        }

        for word in 0..word_count {
            let first_state = word * 64;
            let Some((allele_word, missing_word)) =
                panel.haplotypes.biallelic_marker_word(marker, first_state)
            else {
                return false;
            };
            let states_in_word = (state_count - first_state).min(64);
            let state_mask = if states_in_word == 64 {
                u64::MAX
            } else {
                (1u64 << states_in_word) - 1
            };
            let observed_word = !missing_word & state_mask;
            let first_track = if heterozygous { 1 } else { 0 };
            for track in first_track..3 {
                let Some(target_allele) = targets[track] else {
                    continue;
                };
                scratch.observed[word][track] |= observed_word;
                let mismatched_word = if target_allele == 0 {
                    allele_word
                } else {
                    !allele_word
                };
                scratch.mismatched[word][track] |= mismatched_word & observed_word;
            }
        }
    }

    if heterozygous {
        emissions[..state_count].fill(match_probability);
        for track in 1..3 {
            let track_offset = track * state_count;
            for state in 0..state_count {
                let word = state >> 6;
                let bit = 1u64 << (state & 63);
                let probability = if scratch.observed[word][track] & bit == 0 {
                    1.0
                } else if scratch.mismatched[word][track] & bit != 0 {
                    mismatch_probability
                } else {
                    match_probability
                };
                emissions[track_offset + state] = probability;
            }
        }
    } else {
        for state in 0..state_count {
            let word = state >> 6;
            let bit = 1u64 << (state & 63);
            emissions[state] = if scratch.observed[word][0] & bit == 0 {
                1.0
            } else if scratch.mismatched[word][0] & bit != 0 {
                mismatch_probability
            } else {
                match_probability
            };
        }
        let (track0, rest) = emissions.split_at_mut(state_count);
        let (track1, track2) = rest.split_at_mut(state_count);
        track1.copy_from_slice(track0);
        track2.copy_from_slice(track0);
    }
    true
}

/// Fill all three emissions together in marker-major order. This decodes the
/// target call once per marker and walks the packed donor row contiguously;
/// scratch is bounded by the number of panel states, independent of window size.
fn fill_cluster_emissions(
    emissions: &mut [f32],
    cluster: PhaseCluster,
    swap_haps: bool,
    input: &PhaseWindow,
    snapshot: &PhasedWindow,
    sample: usize,
    panel: &CompositePanel,
    err: f64,
    mismatch_scratch: &mut [[u8; 3]],
    packed_emission_scratch: &mut PackedPhaseEmissionScratch,
) {
    let state_count = panel.haplotypes.n_haplotypes;
    debug_assert_eq!(emissions.len(), 3 * state_count);
    debug_assert_eq!(mismatch_scratch.len(), state_count);
    if fill_biallelic_cluster_emissions(
        emissions,
        cluster,
        swap_haps,
        input,
        snapshot,
        sample,
        panel,
        err,
        packed_emission_scratch,
    ) {
        return;
    }
    let mismatch_probability = ((cluster.end - cluster.start) as f32 * err as f32).min(0.5f32);
    let match_probability = (1.0 - mismatch_probability).max(f32::MIN_POSITIVE);

    match cluster.kind {
        ClusterKind::Missing | ClusterKind::MaskedHeterozygote => {
            emissions.fill(match_probability);
        }
        ClusterKind::Homozygous => {
            mismatch_scratch.fill([0; 3]);
            for marker in cluster.start..cluster.end {
                let target_allele = input.rows[marker].calls.get(sample).unwrap().alleles[0];
                for (state, flags) in mismatch_scratch.iter_mut().enumerate() {
                    let donor_allele = panel.haplotypes.allele(marker, state);
                    if let (Some(target_allele), Some(donor_allele)) = (target_allele, donor_allele)
                    {
                        flags[0] |= 1;
                        flags[0] |= u8::from(target_allele != donor_allele) << 1;
                    }
                }
            }
            for (state, flags) in mismatch_scratch.iter().enumerate() {
                let probability = if flags[0] & 1 == 0 {
                    1.0
                } else if flags[0] & 2 != 0 {
                    mismatch_probability.max(f32::MIN_POSITIVE)
                } else {
                    match_probability
                };
                for track in 0..3 {
                    emissions[track * state_count + state] = probability;
                }
            }
        }
        ClusterKind::Heterozygous { .. } => {
            emissions[..state_count].fill(match_probability);
            let hap0 = observation_haplotype(1, swap_haps);
            let hap1 = observation_haplotype(2, swap_haps);
            mismatch_scratch.fill([0; 3]);
            for marker in cluster.start..cluster.end {
                let target_alleles = snapshot.rows[marker].calls.get(sample).unwrap().alleles;
                for (state, flags) in mismatch_scratch.iter_mut().enumerate() {
                    let donor_allele = panel.haplotypes.allele(marker, state);
                    for (index, hap) in [hap0, hap1].into_iter().enumerate() {
                        if let (Some(target_allele), Some(donor_allele)) =
                            (target_alleles[hap], donor_allele)
                        {
                            let flags = &mut flags[index + 1];
                            *flags |= 1;
                            *flags |= u8::from(target_allele != donor_allele) << 1;
                        }
                    }
                }
            }
            for (state, flags) in mismatch_scratch.iter().enumerate() {
                for index in 0..2 {
                    let status = flags[index + 1];
                    let probability = if status & 1 == 0 {
                        1.0
                    } else if status & 2 != 0 {
                        mismatch_probability.max(f32::MIN_POSITIVE)
                    } else {
                        match_probability
                    };
                    emissions[(index + 1) * state_count + state] = probability;
                }
            }
        }
    }
}

enum AlleleProbabilitySet {
    Small { values: [f32; 4], len: usize },
    Large(Vec<f32>),
}

impl AlleleProbabilitySet {
    fn new(len: usize) -> Self {
        if len <= 4 {
            Self::Small {
                values: [0.0; 4],
                len,
            }
        } else {
            Self::Large(vec![0.0; len])
        }
    }
}

impl std::fmt::Debug for AlleleProbabilitySet {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(formatter)
    }
}

impl Deref for AlleleProbabilitySet {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Small { values, len } => &values[..*len],
            Self::Large(values) => values,
        }
    }
}

impl DerefMut for AlleleProbabilitySet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Small { values, len } => &mut values[..*len],
            Self::Large(values) => values,
        }
    }
}

fn hap_allele_posteriors(
    forward0: &[f32],
    backward0: &[f32],
    forward1: &[f32],
    backward1: &[f32],
    panel: &CompositePanel,
    marker: usize,
    allele_count: usize,
) -> (AlleleProbabilitySet, AlleleProbabilitySet) {
    let mut probabilities0 = AlleleProbabilitySet::new(allele_count);
    let mut probabilities1 = AlleleProbabilitySet::new(allele_count);
    if let Some((marker_count, allele_words, missing_words)) =
        panel.haplotypes.state_major_biallelic_marker(marker)
    {
        if let Some(bit_count) = panel
            .haplotypes
            .n_haplotypes
            .checked_mul(marker_count)
            .filter(|&count| count != 0)
        {
            let required_words = bit_count.div_ceil(64);
            if allele_words.len() >= required_words && panel.haplotypes.is_complete() {
                for state in 0..panel.haplotypes.n_haplotypes {
                    let mass0 = forward0[state] * backward0[state];
                    let mass1 = forward1[state] * backward1[state];
                    let valid0 = mass0 > 0.0 && mass0.is_finite();
                    let valid1 = mass1 > 0.0 && mass1.is_finite();
                    if !valid0 && !valid1 {
                        continue;
                    }
                    let index = state * marker_count + marker;
                    let word = index >> 6;
                    let shift = index & 63;
                    // `bit_count` covers every `(state, marker)` pair, `allele_words`
                    // has at least `required_words`, and `is_complete()` guarantees no
                    // missing alleles even when `missing_words` is omitted.
                    let allele =
                        unsafe { ((*allele_words.get_unchecked(word) >> shift) & 1) as u16 };
                    let allele = usize::from(allele);
                    if valid0 {
                        if let Some(probability) = probabilities0.get_mut(allele) {
                            *probability += mass0;
                        }
                    }
                    if valid1 {
                        if let Some(probability) = probabilities1.get_mut(allele) {
                            *probability += mass1;
                        }
                    }
                }
                return (probabilities0, probabilities1);
            }
            if allele_words.len() >= required_words && missing_words.len() >= required_words {
                for state in 0..panel.haplotypes.n_haplotypes {
                    let mass0 = forward0[state] * backward0[state];
                    let mass1 = forward1[state] * backward1[state];
                    let valid0 = mass0 > 0.0 && mass0.is_finite();
                    let valid1 = mass1 > 0.0 && mass1.is_finite();
                    if !valid0 && !valid1 {
                        continue;
                    }
                    let index = state * marker_count + marker;
                    let word = index >> 6;
                    let shift = index & 63;
                    // `bit_count` covers every `(state, marker)` pair and both
                    // planes have at least `required_words`; this avoids the
                    // repeated layout and bounds checks of `allele()` in the
                    // hottest event loop.
                    let missing = unsafe { (*missing_words.get_unchecked(word) >> shift) & 1 != 0 };
                    if missing {
                        continue;
                    }
                    let allele =
                        unsafe { ((*allele_words.get_unchecked(word) >> shift) & 1) as u16 };
                    let allele = usize::from(allele);
                    if valid0 {
                        if let Some(probability) = probabilities0.get_mut(allele) {
                            *probability += mass0;
                        }
                    }
                    if valid1 {
                        if let Some(probability) = probabilities1.get_mut(allele) {
                            *probability += mass1;
                        }
                    }
                }
                return (probabilities0, probabilities1);
            }
        }
    }
    for state in 0..panel.haplotypes.n_haplotypes {
        // Beagle multiplies forward and backward state probabilities in
        // `float`, then accumulates allele totals in a `float[]`. Keeping the
        // same operation widths is necessary for tied hard calls.
        let mass0 = forward0[state] * backward0[state];
        let mass1 = forward1[state] * backward1[state];
        let valid0 = mass0 > 0.0 && mass0.is_finite();
        let valid1 = mass1 > 0.0 && mass1.is_finite();
        if !valid0 && !valid1 {
            continue;
        }
        let allele = panel.haplotypes.allele(marker, state);
        if let Some(allele) = allele {
            let allele = usize::from(allele);
            if valid0 {
                if let Some(probability) = probabilities0.get_mut(allele) {
                    *probability += mass0;
                }
            }
            if valid1 {
                if let Some(probability) = probabilities1.get_mut(allele) {
                    *probability += mass1;
                }
            }
        }
    }
    (probabilities0, probabilities1)
}

fn modal_allele(probabilities: &[f32], input: &PhaseWindow, marker: usize) -> u16 {
    if let Some((allele, _)) = probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
    {
        if probabilities.iter().any(|&probability| probability > 0.0) {
            return allele as u16;
        }
    }
    let mut counts = vec![0usize; input.rows[marker].meta.allele_count().max(1)];
    for allele in input.rows[marker]
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
        .map_or(0, |(allele, _)| allele as u16)
}

fn allele_probability(probabilities: &[f32], allele: u16) -> f32 {
    probabilities
        .get(usize::from(allele))
        .copied()
        .unwrap_or(0.0)
}

#[inline(always)]
fn observation_haplotype(track: usize, swap_haps: bool) -> usize {
    match (track, swap_haps) {
        (1, false) | (2, true) => 0,
        (1, true) | (2, false) => 1,
        _ => 0,
    }
}

#[inline(always)]
fn copy_track(values: &mut [f32], source: usize, target: usize, state_count: usize) {
    if source == target {
        return;
    }
    let source_start = source * state_count;
    let target_start = target * state_count;
    values.copy_within(source_start..source_start + state_count, target_start);
}

#[cfg(test)]
mod tests {
    use super::{
        backward_project_event_from_codes_three, backward_project_from_codes_three,
        beagle_backward_update, beagle_backward_update_from, beagle_forward_update,
        beagle_forward_update_codes, beagle_forward_update_codes_six,
        beagle_forward_update_codes_three, build_cluster_template, build_clusters,
        build_clusters_into, choose_phase_swap, cluster_emission, cluster_transition_probability,
        cluster_transition_probability_cached, emission_row_count,
        fill_biallelic_cluster_emissions, fill_cluster_emissions,
        fill_state_major_column_emission_codes, fill_state_major_emission_codes,
        hap_allele_posteriors, has_initial_unphased_heterozygote,
        marker_no_recombination_probabilities, parse_paired_hmm_event_target,
        parse_paired_hmm_missing_target, run_backward_pair, run_backward_single,
        sample_requires_phase_baum, update_sample_pair_impl, update_sample_with_diagnostic,
        update_sample_with_diagnostic_and_workspace, ClusterKind, PackedPhaseEmissionScratch,
        PhaseBaumWorkspace, PhaseCluster, PreparedEventSample, EMISSION_MATCH, EMISSION_MISMATCH,
        NO_EVENT_SLOT,
    };
    use crate::imp::types::{
        CompositePanel, DiploidCall, DiploidCalls, HMMConfig, PackedHaplotypes, PhaseWindow,
        PhasedRow, PhasedWindow, VariantMeta, VariantRow,
    };
    use std::mem::MaybeUninit;

    #[test]
    fn cluster_transition_uses_beagle_float_intensity_and_marker_products() {
        let intensity = 0.123_456_79f32;
        let cm_positions = [0.0f64, 0.003, 0.005];
        let mut probability_no_recombination = 1.0f32;
        for pair in cm_positions.windows(2) {
            let distance = (pair[1] - pair[0]) as f32;
            let marker_probability = -(-f64::from(intensity) * f64::from(distance)).exp_m1() as f32;
            probability_no_recombination *= 1.0f32 - marker_probability;
        }
        let expected = 1.0f32 - probability_no_recombination;
        let observed = cluster_transition_probability(&cm_positions, 0, 2, 4, intensity);

        assert_eq!(observed.to_bits(), expected.to_bits());
    }

    #[test]
    fn cached_cluster_transitions_match_uncached_bitwise() {
        let cm_positions = [0.0f64, 0.001, 0.004, 0.009, 0.015, 0.022, 0.031];
        let intensity = 0.123_456_79f32;
        let factors = marker_no_recombination_probabilities(&cm_positions, intensity).unwrap();

        for (previous, current) in [(0, 1), (1, 3), (2, 6), (4, 5), (0, 6)] {
            let expected =
                cluster_transition_probability(&cm_positions, previous, current, 4, intensity);
            let observed = cluster_transition_probability_cached(&factors, previous, current, 4);
            assert_eq!(observed.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn phase_forward_update_matches_beagle_float_order_without_normalizing() {
        let mut expected = [0.137_123_45_f32, 0.287_654_3, 0.575_222_25];
        let emissions = [0.963_21_f32, 0.000_728_799_4, 0.417_31];
        let previous_sum = 0.876_543_2_f32;
        let switch = 0.123_456_79_f32;
        let shift = switch / expected.len() as f32;
        let scale = (1.0_f32 - switch) / previous_sum;
        let mut expected_sum = 0.0_f32;
        for state in 0..expected.len() {
            expected[state] = emissions[state] * (scale * expected[state] + shift);
            expected_sum += expected[state];
        }

        let mut observed = [0.137_123_45_f32, 0.287_654_3, 0.575_222_25];
        let observed_sum = beagle_forward_update(&mut observed, previous_sum, switch, |state| {
            emissions[state]
        });

        assert_eq!(observed_sum.to_bits(), expected_sum.to_bits());
        assert_eq!(
            observed.map(f32::to_bits),
            expected.map(f32::to_bits),
            "forward vectors must retain Beagle's unnormalized float scale"
        );
    }

    #[test]
    fn paired_hmm_trace_targets_are_parsed_without_per_cluster_allocation() {
        assert_eq!(parse_paired_hmm_event_target("12:345"), Some((12, 345)));
        assert_eq!(parse_paired_hmm_event_target("12"), None);
        assert_eq!(
            parse_paired_hmm_missing_target("3:12:345"),
            Some((3, 12, 345))
        );
        assert_eq!(parse_paired_hmm_missing_target("3:12"), None);
        assert_eq!(parse_paired_hmm_missing_target("3:12:345:6"), None);
    }

    #[test]
    fn phase_code_forward_update_matches_generic_emission_kernel() {
        let initial = [0.137_123_45_f32, 0.287_654_3, 0.575_222_25, 0.412_3];
        let codes = [0u8, 1, 2, 0];
        let match_probability = 0.997f32;
        let mismatch_probability = 0.003f32;
        let probabilities = [1.0, match_probability, mismatch_probability];
        let mut expected = initial;
        let mut observed = initial;

        let expected_sum = beagle_forward_update(&mut expected, 1.0, 0.123_456_79, |state| {
            probabilities[codes[state] as usize]
        });
        let observed_sum = beagle_forward_update_codes(
            &mut observed,
            &codes,
            1.0,
            0.123_456_79,
            match_probability,
            mismatch_probability,
        );

        assert_eq!(observed_sum.to_bits(), expected_sum.to_bits());
        assert_eq!(
            observed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected.map(f32::to_bits)
        );
    }

    #[test]
    fn three_track_code_forward_matches_independent_track_updates_bitwise() {
        let state_count = 5usize;
        let codes = [0u8, 1, 2, 0, 1, 1, 0, 2, 1, 0];
        let initial = (0..3 * state_count)
            .map(|state| (state as f32 + 1.0) / 37.0)
            .collect::<Vec<_>>();
        let rows = [Some(0), Some(state_count), None];
        let probabilities = [0.997f32, 0.003f32];
        let switch = 0.017f32;
        let mut expected = initial.clone();
        let mut sums = [1.0f32; 3];
        for track in 0..3 {
            let begin = track * state_count;
            let end = begin + state_count;
            sums[track] = match rows[track] {
                Some(row) => beagle_forward_update_codes(
                    &mut expected[begin..end],
                    &codes[row..row + state_count],
                    sums[track],
                    switch,
                    probabilities[0],
                    probabilities[1],
                ),
                None => {
                    beagle_forward_update(&mut expected[begin..end], sums[track], switch, |_| {
                        probabilities[0]
                    })
                }
            };
        }
        let mut observed = initial;
        let avx2 = super::phase_avx2_available();
        let observed_sums = beagle_forward_update_codes_three(
            &mut observed,
            &codes,
            rows,
            [1.0f32; 3],
            switch,
            probabilities,
            state_count,
            false,
            avx2,
        );

        assert_eq!(observed_sums.map(f32::to_bits), sums.map(f32::to_bits));
        assert_eq!(observed, expected);
    }

    #[test]
    fn three_track_code_backward_matches_sequential_tracks_bitwise() {
        let beta = (0..18)
            .map(|state| (state as f32 + 1.0) / 53.0)
            .collect::<Vec<_>>();
        let codes = [0u8, 1, 2, 0, 1, 1, 1, 0, 2, 1, 0, 2];
        let cluster = PhaseCluster {
            start: 0,
            end: 1,
            kind: super::ClusterKind::Homozygous,
        };
        let mut expected = vec![0.0f32; beta.len()];
        let mut observed = beta.clone();
        let probabilities = [0.997f32, 0.003f32];
        let avx2 = super::phase_avx2_available();

        backward_project_event_from_codes_three(
            &beta,
            &mut expected,
            &codes,
            0,
            cluster.kind,
            0.017,
            probabilities,
            false,
            avx2,
        );
        backward_project_from_codes_three(
            &mut observed,
            &codes,
            0,
            cluster.kind,
            0.017,
            probabilities,
            false,
            avx2,
        );

        assert_eq!(observed, expected);
    }

    #[test]
    fn six_track_code_forward_matches_two_three_track_updates_bitwise() {
        let first_states = 5usize;
        let second_states = 4usize;
        let first_codes = [0u8, 1, 2, 0, 1, 1, 0, 2, 1, 0];
        let second_codes = [1u8, 0, 1, 2, 2, 0, 1, 1, 0, 2, 1, 0];
        let first_rows = [Some(0), Some(first_states), None];
        let second_rows = [Some(0), Some(second_states), Some(0)];
        let first_probabilities = [0.997f32, 0.003f32];
        let second_probabilities = [0.994f32, 0.006f32];
        let first_switch = 0.017f32;
        let second_switch = 0.023f32;
        let avx2 = super::phase_avx2_available();
        let mut expected_first = (0..3 * first_states)
            .map(|state| (state as f32 + 1.0) / 37.0)
            .collect::<Vec<_>>();
        let mut expected_second = (0..3 * second_states)
            .map(|state| (state as f32 + 2.0) / 43.0)
            .collect::<Vec<_>>();
        let expected_first_sums = beagle_forward_update_codes_three(
            &mut expected_first,
            &first_codes,
            first_rows,
            [1.0f32; 3],
            first_switch,
            first_probabilities,
            first_states,
            false,
            avx2,
        );
        let expected_second_sums = beagle_forward_update_codes_three(
            &mut expected_second,
            &second_codes,
            second_rows,
            [1.0f32; 3],
            second_switch,
            second_probabilities,
            second_states,
            false,
            avx2,
        );
        let mut observed_first = (0..3 * first_states)
            .map(|state| (state as f32 + 1.0) / 37.0)
            .collect::<Vec<_>>();
        let mut observed_second = (0..3 * second_states)
            .map(|state| (state as f32 + 2.0) / 43.0)
            .collect::<Vec<_>>();

        let observed_sums = beagle_forward_update_codes_six(
            &mut observed_first,
            &first_codes,
            first_rows,
            [1.0f32; 3],
            first_switch,
            first_probabilities,
            first_states,
            &mut observed_second,
            &second_codes,
            second_rows,
            [1.0f32; 3],
            second_switch,
            second_probabilities,
            second_states,
            false,
            false,
            avx2,
        );

        assert_eq!(observed_first, expected_first);
        assert_eq!(observed_second, expected_second);
        assert_eq!(observed_sums.0, expected_first_sums);
        assert_eq!(observed_sums.1, expected_second_sums);
    }

    #[test]
    fn six_track_code_backward_matches_two_three_track_updates_bitwise() {
        let first_states = 5usize;
        let second_states = 4usize;
        let first_codes = [0u8, 1, 2, 0, 1];
        let second_codes = [1u8, 0, 2, 1];
        let first_beta = (0..3 * first_states)
            .map(|state| (state as f32 + 1.0) / 53.0)
            .collect::<Vec<_>>();
        let second_beta = (0..3 * second_states)
            .map(|state| (state as f32 + 2.0) / 61.0)
            .collect::<Vec<_>>();
        let cluster = PhaseCluster {
            start: 0,
            end: 1,
            kind: super::ClusterKind::Homozygous,
        };
        let first_probabilities = [0.997f32, 0.003f32];
        let second_probabilities = [0.994f32, 0.006f32];
        let avx2 = super::phase_avx2_available();
        let mut expected_first = vec![0.0f32; first_beta.len()];
        let mut expected_second = vec![0.0f32; second_beta.len()];
        backward_project_event_from_codes_three(
            &first_beta,
            &mut expected_first,
            &first_codes,
            0,
            cluster.kind,
            0.017,
            first_probabilities,
            false,
            avx2,
        );
        backward_project_event_from_codes_three(
            &second_beta,
            &mut expected_second,
            &second_codes,
            0,
            cluster.kind,
            0.023,
            second_probabilities,
            false,
            avx2,
        );
        let mut observed_first = first_beta.clone();
        let mut observed_second = second_beta.clone();

        super::backward_project_from_codes_six(
            &mut observed_first,
            &first_codes,
            0,
            cluster.kind,
            0.017,
            first_probabilities,
            false,
            &mut observed_second,
            &second_codes,
            0,
            cluster.kind,
            0.023,
            second_probabilities,
            false,
            avx2,
        );

        assert_eq!(observed_first, expected_first);
        assert_eq!(observed_second, expected_second);
    }

    #[test]
    fn paired_backward_preserves_tracks_for_missing_before_unphased_event() {
        const STATES: usize = 2;

        let make_run = || {
            let clusters = vec![
                PhaseCluster {
                    start: 0,
                    end: 1,
                    kind: ClusterKind::Homozygous,
                },
                PhaseCluster {
                    start: 1,
                    end: 2,
                    kind: ClusterKind::Missing,
                },
                PhaseCluster {
                    start: 2,
                    end: 3,
                    kind: ClusterKind::Heterozygous { unphased: true },
                },
                PhaseCluster {
                    start: 3,
                    end: 4,
                    kind: ClusterKind::Homozygous,
                },
            ];
            PreparedEventSample {
                sample: 0,
                trace_config: Default::default(),
                avx2: false,
                single_track: false,
                checkpointed: false,
                complete_emissions: false,
                clusters,
                transitions: vec![0.0, 0.2, 0.15, 0.05],
                emission_codes: vec![
                    EMISSION_MATCH,
                    EMISSION_MISMATCH,
                    EMISSION_MATCH,
                    EMISSION_MISMATCH,
                    EMISSION_MISMATCH,
                    EMISSION_MATCH,
                    EMISSION_MATCH,
                    EMISSION_MISMATCH,
                ],
                emission_code_offsets: vec![0, 2, 2, 6],
                emission_probabilities: vec![[0.99, 0.01]; 4],
                slots: vec![NO_EVENT_SLOT, 0, 1, NO_EVENT_SLOT],
                beta_cache: vec![MaybeUninit::uninit(); 2 * 2 * STATES],
                beta_checkpoints: Vec::new(),
                beta_local: vec![1.0f32 / STATES as f32; 3 * STATES],
                beta_event_input: vec![0.0; 3 * STATES],
                beta_event: vec![0.0; 3 * STATES],
                backward_cursor: 3,
                backward_active: true,
                alpha: vec![1.0f32 / STATES as f32; 3 * STATES],
                forward_sums: [1.0; 3],
                changes: Vec::new(),
                phase_confidences: Vec::new(),
                swap_haps: false,
                phase_swap_count: 0,
                unphased_heterozygote_count: 1,
                forward_cursor: 0,
                forward_active: true,
                tracked_beta_bytes: 0,
                tracked_event_workspace_bytes: 0,
            }
        };
        let cache = |run: &PreparedEventSample| {
            (0..run.beta_cache.len())
                .map(|index| unsafe { *run.beta_cache[index].assume_init_ref() })
                .collect::<Vec<_>>()
        };

        let mut single = make_run();
        run_backward_single(&mut single);
        let cache_values = cache(&single);
        let missing = &cache_values[..2 * STATES];
        assert_ne!(
            &missing[..STATES],
            &missing[STATES..],
            "the missing cluster must retain the two tracks produced by the following event"
        );

        let mut paired0 = make_run();
        let mut paired1 = make_run();
        run_backward_pair(&mut paired0, &mut paired1);
        assert_eq!(cache(&paired0), cache_values);
        assert_eq!(cache(&paired1), cache_values);
    }

    #[test]
    fn allele_totals_preserve_beagle_float_accumulation() {
        let reference_alleles = [0u16, 1, 0, 1, 0];
        let mut haplotypes = PackedHaplotypes::empty_state_major(5, 1, 1).unwrap();
        for (state, allele) in reference_alleles.into_iter().enumerate() {
            haplotypes.set_allele(0, state, Some(allele)).unwrap();
        }
        haplotypes.mark_complete();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0; 5],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let forward = [0.1f32, 0.2, 0.3, 0.4, 0.5];
        let backward = [0.5f32, 0.4, 0.3, 0.2, 0.1];
        let (observed0, observed1) =
            hap_allele_posteriors(&forward, &backward, &forward, &backward, &panel, 0, 2);
        let mut expected = [0.0f32; 2];
        for state in 0..forward.len() {
            expected[usize::from(reference_alleles[state])] += forward[state] * backward[state];
        }

        for observed in [observed0, observed1] {
            assert_eq!(
                observed
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected.map(f32::to_bits)
            );
            assert_ne!(
                observed[0] + observed[1],
                1.0,
                "Beagle selects from raw allele totals rather than normalized totals"
            );
        }
    }

    #[test]
    fn paired_allele_totals_match_state_and_marker_major_layouts() {
        let reference_alleles = [0u16, 1, 0, 1, 1, 0, 0, 1];
        let mut state_major =
            PackedHaplotypes::empty_state_major(reference_alleles.len(), 1, 1).unwrap();
        let mut marker_major = PackedHaplotypes::empty(reference_alleles.len(), 1, 1).unwrap();
        for (state, &allele) in reference_alleles.iter().enumerate() {
            state_major.set_allele(0, state, Some(allele)).unwrap();
            marker_major.set_allele(0, state, Some(allele)).unwrap();
        }
        state_major.mark_complete();
        marker_major.mark_complete();
        let state_panel = CompositePanel {
            haplotypes: state_major,
            source_samples: vec![0; reference_alleles.len()],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let marker_panel = CompositePanel {
            haplotypes: marker_major,
            source_samples: vec![0; reference_alleles.len()],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let forward0 = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let backward0 = [0.8f32, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let forward1 = [0.8f32, 0.6, 0.4, 0.2, 0.1, 0.3, 0.5, 0.7];
        let backward1 = [0.1f32, 0.3, 0.5, 0.7, 0.8, 0.6, 0.4, 0.2];
        let state_values = hap_allele_posteriors(
            &forward0,
            &backward0,
            &forward1,
            &backward1,
            &state_panel,
            0,
            2,
        );
        let marker_values = hap_allele_posteriors(
            &forward0,
            &backward0,
            &forward1,
            &backward1,
            &marker_panel,
            0,
            2,
        );
        assert_eq!(
            state_values
                .0
                .iter()
                .chain(state_values.1.iter())
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            marker_values
                .0
                .iter()
                .chain(marker_values.1.iter())
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn phase_backward_update_matches_beagle_float_order() {
        let mut expected = [0.137_123_45_f32, 0.287_654_3, 0.575_222_25];
        let emissions = [0.963_21_f32, 0.000_728_799_4, 0.417_31];
        let switch = 0.123_456_79_f32;
        let mut emission_sum = 0.0_f32;
        for state in 0..expected.len() {
            expected[state] *= emissions[state];
            emission_sum += expected[state];
        }
        let shift = switch / expected.len() as f32;
        let scale = (1.0_f32 - switch) / emission_sum;
        for value in &mut expected {
            *value = scale * *value + shift;
        }

        let mut observed = [0.137_123_45_f32, 0.287_654_3, 0.575_222_25];
        beagle_backward_update(&mut observed, switch, |state| emissions[state]);

        assert_eq!(
            observed.map(f32::to_bits),
            expected.map(f32::to_bits),
            "backward updates must use Beagle's f32 sum and must not renormalize"
        );
    }

    #[test]
    fn backward_update_from_matches_copy_then_update_bitwise() {
        let beta_after = [0.137_123_45_f32, 0.287_654_3, 0.575_222_25, 0.000_013_7];
        let emissions = [0.963_21_f32, 0.000_728_799_4, 0.417_31, 0.812_3];
        let switch = 0.123_456_79_f32;
        let mut expected = beta_after;
        beagle_backward_update(&mut expected, switch, |state| emissions[state]);
        let mut actual = [0.0f32; 4];

        beagle_backward_update_from(&beta_after, &emissions, &mut actual, switch);

        assert_eq!(
            actual.map(f32::to_bits),
            expected.map(f32::to_bits),
            "the fused backward projection must preserve Beagle's operation order"
        );
    }

    #[test]
    fn genotype_track_applies_beagle_match_emission_at_heterozygotes() {
        let row = VariantRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: 1,
                id: "v1".to_owned(),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: DiploidCalls::from_calls(vec![DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: false,
            }]),
        };
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            emit_start: 0,
            emit_end: 1,
            rows: vec![row.clone()],
            cm_positions: vec![0.0],
            incoming_overlap_len: 0,
        };
        let snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: vec![PhasedRow {
                meta: row.meta,
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
            }],
            emit_start: 0,
            emit_end: 1,
        };
        let mut haplotypes = PackedHaplotypes::empty(2, 1, 1).unwrap();
        haplotypes.set_allele(0, 0, Some(0)).unwrap();
        haplotypes.set_allele(0, 1, Some(1)).unwrap();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let cluster = PhaseCluster {
            start: 0,
            end: 1,
            kind: super::ClusterKind::Heterozygous { unphased: false },
        };
        let err = 0.01_f64;
        let expected = 1.0_f32 - err as f32;

        for state in 0..2 {
            let observed =
                cluster_emission(cluster, 0, false, state, &input, &snapshot, 0, &panel, err);
            assert_eq!(
                observed.to_bits(),
                expected.to_bits(),
                "Beagle's genotype track has zero mismatch at a heterozygote but still applies 1-clustEm"
            );
        }
    }

    #[test]
    fn repeated_phase_updates_reuse_event_workspace_allocation() {
        let make_row = |marker: usize| VariantRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: marker as u64 + 1,
                id: format!("v{}", marker + 1),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: DiploidCalls::from_calls(vec![DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: false,
            }]),
        };
        let input_rows = vec![make_row(0), make_row(1)];
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            emit_start: 0,
            emit_end: 2,
            rows: input_rows.clone(),
            cm_positions: vec![0.0, 0.001],
            incoming_overlap_len: 0,
        };
        let snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: input_rows
                .into_iter()
                .map(|row| PhasedRow {
                    meta: row.meta,
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
                })
                .collect(),
            emit_start: 0,
            emit_end: 2,
        };
        let mut haplotypes = PackedHaplotypes::empty_state_major(2, 2, 1).unwrap();
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
        let config = HMMConfig {
            ne: 500,
            recombination_intensity: 0.01,
            err: 0.01,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 2,
        };
        let finished = [0u64];
        let masked = [0u64];
        let mut workspace = PhaseBaumWorkspace::default();

        update_sample_with_diagnostic_and_workspace(
            &input,
            &snapshot,
            0,
            &finished,
            true,
            &masked,
            &panel,
            &config,
            None,
            &mut workspace,
            |_| {},
        )
        .unwrap();
        let allocation = workspace.emission_codes.as_ptr();
        let capacity = workspace.emission_codes.capacity();
        let scratch_allocation = workspace.packed_emission_scratch.target_observed[0].as_ptr();
        let scratch_capacity = workspace.packed_emission_scratch.target_observed[0].capacity();
        update_sample_with_diagnostic_and_workspace(
            &input,
            &snapshot,
            0,
            &finished,
            true,
            &masked,
            &panel,
            &config,
            None,
            &mut workspace,
            |_| {},
        )
        .unwrap();

        assert!(capacity > 0);
        assert_eq!(workspace.emission_codes.as_ptr(), allocation);
        assert_eq!(workspace.emission_codes.capacity(), capacity);
        assert!(scratch_capacity > 0);
        assert_eq!(
            workspace.packed_emission_scratch.target_observed[0].as_ptr(),
            scratch_allocation
        );
        assert_eq!(
            workspace.packed_emission_scratch.target_observed[0].capacity(),
            scratch_capacity
        );
    }

    #[test]
    fn fused_cluster_emissions_match_independent_track_emissions_bitwise() {
        let make_row = |marker: usize, call: DiploidCall| super::super::types::VariantRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: marker as u64 + 1,
                id: format!("v{}", marker + 1),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: DiploidCalls::from_calls(vec![call]),
        };
        let input_rows = vec![
            make_row(
                0,
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: false,
                },
            ),
            make_row(
                1,
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
            ),
            make_row(2, DiploidCall::missing(false)),
        ];
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            emit_start: 0,
            emit_end: input_rows.len(),
            rows: input_rows.clone(),
            cm_positions: vec![0.0, 0.001, 0.002],
            incoming_overlap_len: 0,
        };
        let snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: input_rows
                .iter()
                .map(|row| PhasedRow {
                    meta: row.meta.clone(),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: row.calls.get(0).unwrap().alleles,
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
            emit_end: input_rows.len(),
        };
        let mut haplotypes = PackedHaplotypes::empty(3, 3, 1).unwrap();
        for (marker, alleles) in [
            [Some(0), Some(1), None],
            [Some(1), Some(0), Some(1)],
            [None, Some(1), Some(0)],
        ]
        .into_iter()
        .enumerate()
        {
            for (haplotype, allele) in alleles.into_iter().enumerate() {
                haplotypes.set_allele(marker, haplotype, allele).unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1],
            marker_indices: vec![0, 1, 2],
            allele_counts: vec![2, 2, 2],
        };
        let clusters = [
            PhaseCluster {
                start: 0,
                end: 1,
                kind: super::ClusterKind::Homozygous,
            },
            PhaseCluster {
                start: 1,
                end: 2,
                kind: super::ClusterKind::Heterozygous { unphased: true },
            },
            PhaseCluster {
                start: 1,
                end: 2,
                kind: super::ClusterKind::MaskedHeterozygote,
            },
            PhaseCluster {
                start: 2,
                end: 3,
                kind: super::ClusterKind::Missing,
            },
        ];
        let err = 0.013_579_f64;

        for cluster in clusters {
            for swap_haps in [false, true] {
                let mut fused = vec![0.0f32; 3 * panel.haplotypes.n_haplotypes];
                let mut mismatch_scratch = vec![[0u8; 3]; panel.haplotypes.n_haplotypes];
                let mut packed_emission_scratch = PackedPhaseEmissionScratch::default();
                fill_cluster_emissions(
                    &mut fused,
                    cluster,
                    swap_haps,
                    &input,
                    &snapshot,
                    0,
                    &panel,
                    err,
                    &mut mismatch_scratch,
                    &mut packed_emission_scratch,
                );
                for track in 0..3 {
                    for state in 0..panel.haplotypes.n_haplotypes {
                        let expected = cluster_emission(
                            cluster,
                            track,
                            track > 0 && swap_haps,
                            state,
                            &input,
                            &snapshot,
                            0,
                            &panel,
                            err,
                        );
                        assert_eq!(
                            fused[track * panel.haplotypes.n_haplotypes + state].to_bits(),
                            expected.to_bits(),
                            "cluster={cluster:?}, track={track}, state={state}, swap={swap_haps}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn packed_biallelic_cluster_emissions_match_scalar_across_unaligned_rows() {
        let target_calls = [
            [Some(0), Some(0)],
            [Some(1), Some(1)],
            [Some(0), Some(1)],
            [Some(0), Some(1)],
            [None, None],
        ];
        let input_rows = target_calls
            .iter()
            .enumerate()
            .map(|(marker, alleles)| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: *alleles,
                    phased: false,
                }]),
            })
            .collect::<Vec<_>>();
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            emit_start: 0,
            emit_end: input_rows.len(),
            rows: input_rows.clone(),
            cm_positions: (0..input_rows.len())
                .map(|marker| marker as f64 * 0.001)
                .collect(),
            incoming_overlap_len: 0,
        };
        let snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: input_rows
                .iter()
                .map(|row| PhasedRow {
                    meta: row.meta.clone(),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: row.calls.get(0).unwrap().alleles,
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
            emit_end: input_rows.len(),
        };
        let mut haplotypes = PackedHaplotypes::empty(70, input_rows.len(), 1).unwrap();
        for marker in 0..input_rows.len() {
            for state in 0..70 {
                let allele = if (state * 5 + marker) % 17 == 0 {
                    None
                } else {
                    Some(((state * 3 + marker * 7) % 2) as u16)
                };
                haplotypes.set_allele(marker, state, allele).unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0; 70],
            marker_indices: (0..input_rows.len()).collect(),
            allele_counts: vec![2; input_rows.len()],
        };
        let clusters = [
            PhaseCluster {
                start: 2,
                end: 4,
                kind: super::ClusterKind::Heterozygous { unphased: true },
            },
            PhaseCluster {
                start: 0,
                end: 2,
                kind: super::ClusterKind::Homozygous,
            },
        ];
        let err = 0.013_579_f64;
        let mut scratch = PackedPhaseEmissionScratch::default();

        for cluster in clusters {
            for swap_haps in [false, true] {
                let mut actual = vec![0.0f32; 3 * 70];
                assert!(fill_biallelic_cluster_emissions(
                    &mut actual,
                    cluster,
                    swap_haps,
                    &input,
                    &snapshot,
                    0,
                    &panel,
                    err,
                    &mut scratch,
                ));
                for track in 0..3 {
                    for state in 0..70 {
                        let expected = cluster_emission(
                            cluster,
                            track,
                            track > 0 && swap_haps,
                            state,
                            &input,
                            &snapshot,
                            0,
                            &panel,
                            err,
                        );
                        assert_eq!(
                            actual[track * 70 + state].to_bits(),
                            expected.to_bits(),
                            "cluster={cluster:?}, track={track}, state={state}, swap={swap_haps}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn column_cached_emission_codes_match_per_cluster_encoding() {
        const MARKERS: usize = 130;
        const STATES: usize = 70;
        let target0 = (0..MARKERS)
            .map(|marker| ((marker * 7 + marker / 11) % 2) as u16)
            .collect::<Vec<_>>();
        let target1 = target0
            .iter()
            .enumerate()
            .map(|(marker, &allele)| {
                if matches!(marker, 3 | 70) {
                    1 - allele
                } else {
                    allele
                }
            })
            .collect::<Vec<_>>();
        let mut packed_target = PackedHaplotypes::empty_state_major(4, MARKERS, 1).unwrap();
        for marker in 0..MARKERS {
            packed_target
                .set_allele(marker, 0, Some(target0[marker]))
                .unwrap();
            packed_target
                .set_allele(marker, 1, Some(target1[marker]))
                .unwrap();
            packed_target.set_allele(marker, 2, Some(0)).unwrap();
            packed_target.set_allele(marker, 3, Some(0)).unwrap();
        }
        packed_target.mark_complete();
        let mut panel_haplotypes = PackedHaplotypes::empty_state_major(STATES, MARKERS, 1).unwrap();
        for marker in 0..MARKERS {
            for state in 0..STATES {
                let allele = ((state * 13 + marker * 5 + marker / 3) % 2) as u16;
                panel_haplotypes
                    .set_allele(marker, state, Some(allele))
                    .unwrap();
            }
        }
        panel_haplotypes.mark_complete();
        let panel = CompositePanel {
            haplotypes: panel_haplotypes,
            source_samples: vec![1; STATES],
            marker_indices: (0..MARKERS).collect(),
            allele_counts: vec![2; MARKERS],
        };
        let input_rows = (0..MARKERS)
            .map(|marker| {
                let target = if marker == 4 {
                    [None, None]
                } else if target0[marker] == target1[marker] {
                    [Some(target0[marker]), Some(target0[marker])]
                } else {
                    [Some(target0[marker]), Some(target1[marker])]
                };
                VariantRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{}", marker + 1),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(vec![
                        DiploidCall {
                            alleles: target,
                            phased: false,
                        },
                        DiploidCall {
                            alleles: [Some(0), Some(0)],
                            phased: false,
                        },
                    ]),
                }
            })
            .collect::<Vec<_>>();
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: input_rows.clone(),
            cm_positions: (0..MARKERS).map(|marker| marker as f64 * 0.001).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: MARKERS,
        };
        let snapshot = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: input_rows
                .into_iter()
                .enumerate()
                .map(|(marker, row)| PhasedRow {
                    meta: row.meta,
                    calls: DiploidCalls::from_calls(vec![
                        DiploidCall {
                            alleles: [Some(target0[marker]), Some(target1[marker])],
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
                })
                .collect(),
            emit_start: 0,
            emit_end: MARKERS,
        };
        let clusters = [
            PhaseCluster {
                start: 0,
                end: 3,
                kind: ClusterKind::Homozygous,
            },
            PhaseCluster {
                start: 3,
                end: 4,
                kind: ClusterKind::Heterozygous { unphased: true },
            },
            PhaseCluster {
                start: 4,
                end: 5,
                kind: ClusterKind::Missing,
            },
            PhaseCluster {
                start: 5,
                end: 70,
                kind: ClusterKind::Homozygous,
            },
            PhaseCluster {
                start: 70,
                end: 71,
                kind: ClusterKind::Heterozygous { unphased: true },
            },
            PhaseCluster {
                start: 71,
                end: MARKERS,
                kind: ClusterKind::Homozygous,
            },
        ];
        let mut offsets = Vec::new();
        let mut expected_count = 0usize;
        for cluster in &clusters {
            offsets.push(expected_count);
            expected_count += emission_row_count(cluster.kind) * STATES;
        }
        let mut expected = vec![MaybeUninit::uninit(); expected_count];
        let mut scratch = PackedPhaseEmissionScratch::default();
        for (cluster_index, &cluster) in clusters.iter().enumerate() {
            let offset = offsets[cluster_index];
            let rows = emission_row_count(cluster.kind);
            fill_state_major_emission_codes(
                &mut expected[offset..offset + rows * STATES],
                cluster,
                &input,
                &snapshot,
                0,
                &panel.haplotypes.allele_bits,
                &panel.haplotypes.missing_bits,
                MARKERS,
                STATES,
                true,
                Some(&packed_target),
                &mut scratch,
            )
            .unwrap();
        }
        let expected = expected
            .into_iter()
            .map(|code| unsafe { code.assume_init() })
            .collect::<Vec<_>>();
        let mut actual = vec![MaybeUninit::uninit(); expected_count];
        assert!(fill_state_major_column_emission_codes(
            &mut actual,
            &clusters,
            &offsets,
            0,
            &panel,
            &packed_target,
            &mut scratch,
        )
        .unwrap());
        let actual = actual
            .into_iter()
            .map(|code| unsafe { code.assume_init() })
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    fn homozygous_window(cm_positions: Vec<f64>) -> PhaseWindow {
        let rows = cm_positions
            .iter()
            .enumerate()
            .map(|(marker, _)| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: false,
                }]),
            })
            .collect::<Vec<_>>();
        PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            emit_start: 0,
            emit_end: rows.len(),
            rows,
            cm_positions,
            incoming_overlap_len: 0,
        }
    }

    #[test]
    fn phase_odds_choose_crossed_haplotype_pairing() {
        let forward_hap0 = [0.99, 0.01];
        let forward_hap1 = [0.01, 0.99];
        let backward_hap0 = [0.01, 0.99];
        let backward_hap1 = [0.99, 0.01];

        let (swap, confidence) =
            choose_phase_swap(&forward_hap0, &forward_hap1, &backward_hap0, &backward_hap1)
                .unwrap();

        assert!(swap, "the crossed pairing has the larger posterior odds");
        assert!(confidence > 7.0, "confidence is the absolute log odds");
    }

    #[test]
    fn phase_odds_follow_beagle_f32_accumulation_at_near_ties() {
        let forward_hap0 = [0.009_175_353_f32, 1.885_623_9];
        let forward_hap1 = [0.007_582_690_6_f32, 1.902_900_6];
        let backward_hap0 = [0.001_395_352_9_f32, 5.207_5];
        let backward_hap1 = [0.001_474_939_2_f32, 5.677_529_3];
        let mut p11 = 0.0_f32;
        let mut p12 = 0.0_f32;
        let mut p21 = 0.0_f32;
        let mut p22 = 0.0_f32;
        for state in 0..forward_hap0.len() {
            p11 += forward_hap0[state] * backward_hap0[state];
            p12 += forward_hap0[state] * backward_hap1[state];
            p21 += forward_hap1[state] * backward_hap0[state];
            p22 += forward_hap1[state] * backward_hap1[state];
        }
        assert!(
            p11 * p22 < p12 * p21,
            "Beagle's float calculation selects a swap: p11={p11:?} p12={p12:?} p21={p21:?} p22={p22:?}"
        );

        let (swap, _) =
            choose_phase_swap(&forward_hap0, &forward_hap1, &backward_hap0, &backward_hap1)
                .unwrap();
        assert!(swap, "phase selection must use Beagle's f32 arithmetic");
    }

    #[test]
    fn zero_phase_scores_pass_beagle_threshold_after_burnin() {
        let zeros = [0.0f32; 4];
        let (swap, confidence) = choose_phase_swap(&zeros, &zeros, &zeros, &zeros).unwrap();

        assert!(!swap);
        assert_eq!(confidence, f64::INFINITY);
    }

    #[test]
    fn phase_odds_are_invariant_to_track_normalization() {
        let forward_hap0 = [0.7, 0.3];
        let forward_hap1 = [0.2, 0.8];
        let backward_hap0 = [0.4, 0.6];
        let backward_hap1 = [0.9, 0.1];
        let original =
            choose_phase_swap(&forward_hap0, &forward_hap1, &backward_hap0, &backward_hap1)
                .unwrap();
        let scaled = choose_phase_swap(&[7.0, 3.0], &[2.0, 8.0], &[4.0, 6.0], &[9.0, 1.0]).unwrap();

        assert_eq!(original.0, scaled.0);
        assert!((original.1 - scaled.1).abs() < 1.0e-6);
    }

    #[test]
    fn two_heterozygote_phase_log_odds_match_joint_path_enumeration() {
        let rows = (0..2)
            .map(|marker| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker + 1,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                }]),
            })
            .collect::<Vec<_>>();
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: rows.clone(),
            cm_positions: vec![0.0, 0.04],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 2,
        };
        let snapshot = PhasedWindow {
            chrom: input.chrom.clone(),
            index: input.index,
            rows: rows
                .iter()
                .map(|row| PhasedRow {
                    meta: row.meta.clone(),
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
                })
                .collect(),
            emit_start: 0,
            emit_end: 2,
        };
        let mut haplotypes = PackedHaplotypes::empty(2, 2, 1).unwrap();
        for marker in 0..2 {
            haplotypes.set_allele(marker, 0, Some(0)).unwrap();
            haplotypes.set_allele(marker, 1, Some(1)).unwrap();
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1, 2],
            marker_indices: vec![0, 1],
            allele_counts: vec![2, 2],
        };
        let config = HMMConfig {
            ne: 500,
            recombination_intensity: (0.04_f32 * 500.0) / 4.0,
            err: 0.05,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 4,
        };

        let mut trace = Vec::new();
        let update = update_sample_with_diagnostic(
            &input,
            &snapshot,
            0,
            &[0],
            true,
            &[0],
            &panel,
            &config,
            Some(1),
            |row| trace.push(row),
        )
        .unwrap();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].phase_swap, Some(false));

        let err = config.err;
        let emission = |observed: u16, state_allele: u16| {
            if observed == state_allele {
                1.0 - err
            } else {
                err
            }
        };
        let initial_track = |observed: u16| {
            let mut probabilities = [emission(observed, 0), emission(observed, 1)];
            let total = probabilities.iter().sum::<f64>();
            for probability in &mut probabilities {
                *probability /= total;
            }
            probabilities
        };
        let switch = 1.0
            - (-4.0 * config.ne as f64 * (input.cm_positions[1] - input.cm_positions[0])
                / 100.0
                / config.haplotype_count as f64)
                .exp();
        let track_likelihood = |initial: &[f64; 2], next_observed: u16| {
            (0..2)
                .flat_map(|previous| {
                    (0..2).map(move |next| {
                        let transition = if previous == next {
                            1.0 - switch + switch / 2.0
                        } else {
                            switch / 2.0
                        };
                        initial[previous] * transition * emission(next_observed, next as u16)
                    })
                })
                .sum::<f64>()
        };
        let hap0_at_first = initial_track(0);
        let hap1_at_first = initial_track(1);
        let no_swap = track_likelihood(&hap0_at_first, 0) * track_likelihood(&hap1_at_first, 1);
        let swap = track_likelihood(&hap0_at_first, 1) * track_likelihood(&hap1_at_first, 0);
        assert!(no_swap > swap);
        let expected_log_odds = (no_swap / swap).ln();
        assert!((trace[0].phase_log_odds.unwrap() - expected_log_odds).abs() < 1.0e-5);
        assert_eq!(update.calls[0].alleles, [Some(0), Some(1)]);
        assert_eq!(update.calls[1].alleles, [Some(0), Some(1)]);
    }

    #[test]
    fn homozygous_clusters_match_beagle_cm_and_marker_limits() {
        let marker_limited = homozygous_window(vec![0.0; 300]);
        let marker_clusters = build_clusters(&marker_limited, 0, &[0], false, &[]);
        assert_eq!(
            marker_clusters
                .iter()
                .map(|cluster: &PhaseCluster| cluster.end - cluster.start)
                .collect::<Vec<_>>(),
            vec![255, 45]
        );

        let cm_limited = homozygous_window((0..100).map(|marker| marker as f64 * 0.0001).collect());
        let cm_clusters = build_clusters(&cm_limited, 0, &[0, 0], false, &[]);
        assert_eq!(
            cm_clusters
                .iter()
                .map(|cluster: &PhaseCluster| cluster.end - cluster.start)
                .collect::<Vec<_>>(),
            vec![51, 49]
        );
    }

    #[test]
    fn collapsed_single_track_matches_three_track_homozygous_and_missing_hmm() {
        let sample_calls = [
            [
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [Some(0), None],
                    phased: false,
                },
            ],
            [
                DiploidCall::missing(false),
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: false,
                },
            ],
            [
                DiploidCall {
                    alleles: [Some(1), Some(1)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [None, Some(1)],
                    phased: false,
                },
            ],
            [
                DiploidCall::missing(false),
                DiploidCall {
                    alleles: [Some(1), Some(1)],
                    phased: false,
                },
            ],
            [
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [None, Some(0)],
                    phased: true,
                },
            ],
            [
                DiploidCall {
                    alleles: [Some(1), Some(1)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: false,
                },
            ],
        ];
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: sample_calls
                .iter()
                .enumerate()
                .map(|(marker, calls)| VariantRow {
                    meta: VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls: DiploidCalls::from_calls(calls.to_vec()),
                })
                .collect(),
            cm_positions: (0..6).map(|marker| marker as f64 * 0.01).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 6,
        };
        let snapshot = PhasedWindow {
            chrom: input.chrom.clone(),
            index: input.index,
            rows: input
                .rows
                .iter()
                .map(|row| PhasedRow {
                    meta: row.meta.clone(),
                    calls: DiploidCalls::from_calls(
                        row.calls
                            .iter()
                            .map(|call| DiploidCall {
                                alleles: call.alleles.map(|allele| allele.or(Some(0))),
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
            emit_end: 6,
        };
        let packed_target = PackedHaplotypes::complete_biallelic_state_major(&snapshot)
            .unwrap()
            .unwrap();
        let mut haplotypes = PackedHaplotypes::empty_state_major(2, 6, 1).unwrap();
        let reference_alleles = [
            [0u16, 0],
            [0u16, 1],
            [1u16, 0],
            [1u16, 0],
            [0u16, 1],
            [1u16, 1],
        ];
        for (marker, alleles) in reference_alleles.into_iter().enumerate() {
            haplotypes.set_allele(marker, 0, Some(alleles[0])).unwrap();
            haplotypes.set_allele(marker, 1, Some(alleles[1])).unwrap();
        }
        haplotypes.mark_complete();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![2, 3],
            marker_indices: (0..6).collect(),
            allele_counts: vec![2; 6],
        };
        let template0 = build_cluster_template(&input, 0).unwrap();
        let template1 = build_cluster_template(&input, 1).unwrap();
        let finished = [0u64];
        let masked = [0u64];
        let config = HMMConfig {
            ne: 1_000,
            recombination_intensity: 10.0,
            err: 0.01,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 4,
        };
        let transition_factors = marker_no_recombination_probabilities(
            &input.cm_positions,
            config.recombination_intensity,
        )
        .unwrap();

        let run = |collapse| {
            let mut workspace0 = PhaseBaumWorkspace::default();
            let mut workspace1 = PhaseBaumWorkspace::default();
            let updates = update_sample_pair_impl(
                &input,
                &snapshot,
                0,
                1,
                &template0,
                &template1,
                &finished,
                &finished,
                false,
                &masked,
                &masked,
                &panel,
                &panel,
                Some(&packed_target),
                &transition_factors,
                &config,
                &mut workspace0,
                &mut workspace1,
                collapse,
            )
            .unwrap();
            (updates, workspace0.beta_cache.len())
        };

        let three_track = run(false);
        let single_track = run(true);
        let ((single0, single1), single_cache_len) = single_track;
        let ((three0, three1), three_cache_len) = three_track;
        assert_eq!(single0.changes, three0.changes);
        assert_eq!(single1.changes, three1.changes);
        assert_eq!(single0.phase_confidences, three0.phase_confidences);
        assert_eq!(single1.phase_confidences, three1.phase_confidences);
        assert_eq!(single_cache_len * 2, three_cache_len);
    }

    #[test]
    fn selected_missing_site_emits_three_track_recursion_trace() {
        let meta = |position| VariantMeta {
            chrom: "1".to_owned(),
            pos: position,
            id: format!("v{position}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let rows = vec![
            VariantRow {
                meta: meta(1),
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: false,
                    },
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    },
                ]),
            },
            VariantRow {
                meta: meta(2),
                calls: DiploidCalls::from_calls(vec![
                    DiploidCall::missing(false),
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: true,
                    },
                ]),
            },
        ];
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 3,
            rows: rows.clone(),
            cm_positions: vec![0.0, 0.1],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 2,
        };
        let snapshot = PhasedWindow {
            chrom: input.chrom.clone(),
            index: input.index,
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
            emit_end: 2,
        };
        let mut haplotypes = PackedHaplotypes::empty(2, 2, 1).unwrap();
        for state in 0..2 {
            haplotypes.set_allele(0, state, Some(0)).unwrap();
        }
        haplotypes.set_allele(1, 0, Some(0)).unwrap();
        haplotypes.set_allele(1, 1, Some(1)).unwrap();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1, 1],
            marker_indices: vec![0, 1],
            allele_counts: vec![2, 2],
        };
        let config = HMMConfig {
            ne: 1_000,
            recombination_intensity: (0.04_f32 * 1_000.0) / 4.0,
            err: 0.01,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 4,
        };
        let mut trace = Vec::new();
        let update = update_sample_with_diagnostic(
            &input,
            &snapshot,
            0,
            &[0],
            false,
            &[0],
            &panel,
            &config,
            Some(1),
            |row| trace.push(row),
        )
        .unwrap();

        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].marker, 1);
        assert_eq!(trace[0].emission_probabilities.len(), 3);
        assert_eq!(trace[0].forward_probabilities.len(), 3);
        assert_eq!(trace[0].backward_probabilities.len(), 3);
        assert_eq!(trace[0].state_alleles, vec![Some(0), Some(1)]);
        assert!(trace[0]
            .posterior_normalization
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
        assert_eq!(trace[0].phase_swap, None);
        assert_eq!(trace[0].selected_alleles, update.calls[1].alleles);
    }

    #[test]
    fn mixed_window_uses_first_het_as_anchor_and_rephases_later_hets() {
        let rows = [true, false, true]
            .into_iter()
            .enumerate()
            .map(|(marker, phased)| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased,
                }]),
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

        let clusters = build_clusters(&window, 0, &[0], true, &[0]);

        assert!(has_initial_unphased_heterozygote(&window));

        assert!(matches!(
            clusters[0].kind,
            super::ClusterKind::Heterozygous { unphased: false }
        ));
        assert!(matches!(
            clusters[1].kind,
            super::ClusterKind::Heterozygous { unphased: true }
        ));
        assert!(matches!(
            clusters[2].kind,
            super::ClusterKind::Heterozygous { unphased: true }
        ));
    }

    #[test]
    fn fully_phased_heterozygotes_do_not_activate_rephasing() {
        let rows = (0..3)
            .map(|marker| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: true,
                }]),
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

        assert!(!has_initial_unphased_heterozygote(&window));
    }

    #[test]
    fn resolved_samples_skip_phase_baum_updates() {
        let rows = [
            (
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: true,
                },
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: true,
                },
            ),
            (
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: true,
                },
                DiploidCall::missing(false),
            ),
            (
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
                DiploidCall {
                    alleles: [Some(1), Some(0)],
                    phased: false,
                },
            ),
        ]
        .into_iter()
        .enumerate()
        .map(|(marker, calls)| VariantRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: marker as u64 + 1,
                id: format!("v{marker}"),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls: DiploidCalls::from_calls(vec![calls.0, calls.1]),
        })
        .collect();
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            cm_positions: vec![0.0, 0.01, 0.02],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 3,
        };

        let resolved_sample_finished = [1u64 << 2];
        assert!(!sample_requires_phase_baum(
            &window,
            0,
            &resolved_sample_finished,
            true
        ));
        assert!(sample_requires_phase_baum(
            &window,
            1,
            &resolved_sample_finished,
            true
        ));
    }

    #[test]
    fn cluster_builder_reuses_worker_capacity() {
        let rows = (0..6)
            .map(|marker| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{marker}"),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![if marker == 3 {
                    DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: false,
                    }
                } else {
                    DiploidCall {
                        alleles: [Some(0), Some(0)],
                        phased: true,
                    }
                }]),
            })
            .collect();
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            cm_positions: (0..6).map(|marker| marker as f64 * 0.01).collect(),
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 6,
        };
        let mut clusters = Vec::new();

        build_clusters_into(&window, 0, &[0], true, &[], &mut clusters);
        let allocation = clusters.as_ptr();
        let capacity = clusters.capacity();
        build_clusters_into(&window, 0, &[0], true, &[], &mut clusters);

        assert!(capacity > 0);
        assert_eq!(clusters.as_ptr(), allocation);
        assert_eq!(clusters.capacity(), capacity);
    }

    #[test]
    fn first_heterozygote_in_overlap_anchors_following_window_phase() {
        let rows = (0..3)
            .map(|marker| VariantRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{}", marker + 1),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(vec![DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: true,
                }]),
            })
            .collect::<Vec<_>>();
        let window = PhaseWindow {
            chrom: "1".to_owned(),
            index: 1,
            rows,
            cm_positions: vec![0.0, 0.01, 0.02],
            incoming_overlap_len: 1,
            emit_start: 1,
            emit_end: 3,
        };

        let clusters = build_clusters(&window, 0, &[0], true, &[0]);

        assert!(matches!(
            clusters[0].kind,
            super::ClusterKind::Heterozygous { unphased: false }
        ));
        assert!(matches!(
            clusters[1].kind,
            super::ClusterKind::Heterozygous { unphased: true }
        ));
        assert!(matches!(
            clusters[2].kind,
            super::ClusterKind::Heterozygous { unphased: true }
        ));
    }

    #[test]
    fn masked_heterozygote_keeps_unordered_gt_and_beagle_match_emission() {
        let meta = |pos| VariantMeta {
            chrom: "1".to_owned(),
            pos,
            id: format!("v{pos}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        };
        let input = PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows: [5, 10]
                .into_iter()
                .map(|pos| VariantRow {
                    meta: meta(pos),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: [Some(0), Some(1)],
                        phased: false,
                    }]),
                })
                .collect(),
            cm_positions: vec![0.0, 0.01],
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: 2,
        };
        let snapshot = PhasedWindow {
            chrom: input.chrom.clone(),
            index: 0,
            rows: input
                .rows
                .iter()
                .enumerate()
                .map(|(marker, row)| PhasedRow {
                    meta: row.meta.clone(),
                    calls: DiploidCalls::from_calls(vec![DiploidCall {
                        alleles: if marker == 1 {
                            [Some(1), Some(0)]
                        } else {
                            [Some(0), Some(1)]
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
                .collect(),
            emit_start: 0,
            emit_end: 2,
        };
        let mut haplotypes = PackedHaplotypes::empty(2, 2, 1).unwrap();
        for marker in 0..2 {
            haplotypes.set_allele(marker, 0, Some(0)).unwrap();
            haplotypes.set_allele(marker, 1, Some(1)).unwrap();
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1, 2],
            marker_indices: vec![0, 1],
            allele_counts: vec![2, 2],
        };
        let config = HMMConfig {
            ne: 1_000,
            recombination_intensity: (0.04_f32 * 1_000.0) / 4.0,
            err: 0.01,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 4,
        };
        let mut trace = Vec::new();

        let update = update_sample_with_diagnostic(
            &input,
            &snapshot,
            0,
            &[0],
            true,
            &[0b10],
            &panel,
            &config,
            Some(1),
            |row| trace.push(row),
        )
        .unwrap();

        assert_eq!(update.calls[1].alleles, [Some(0), Some(1)]);
        assert_eq!(update.phase_confidences.len(), 1);
        assert_eq!(update.phase_confidences[0].0, 1);
        assert!(update.phase_confidences[0].1.is_finite());
        assert_eq!(trace[0].emission_probabilities, vec![vec![0.99, 0.99]; 3]);
        assert_eq!(trace[0].phase_swap, Some(true));
    }

    #[test]
    fn beta_cache_budget_env_parser_handles_disabled_and_positive_mb() {
        assert_eq!(super::parse_beta_cache_budget_mb("0"), Ok(None));
        assert_eq!(super::parse_beta_cache_budget_mb(" 0 "), Ok(None));
        assert_eq!(super::parse_beta_cache_budget_mb(""), Err(()));
        assert_eq!(super::parse_beta_cache_budget_mb("invalid"), Err(()));
        assert_eq!(
            super::parse_beta_cache_budget_mb("32"),
            Ok(Some(32 * 1024 * 1024))
        );
    }

    #[test]
    fn checkpointed_beta_cache_matches_full_cache_bitwise_single_and_three_track() {
        const MARKERS: usize = 600;
        const STATES: usize = 8;

        let make_panel = |seed: usize| {
            let mut haplotypes = PackedHaplotypes::empty_state_major(STATES, MARKERS, 1).unwrap();
            for marker in 0..MARKERS {
                for state in 0..STATES {
                    let allele = (((marker * 7 + state * 13 + seed * 3) / 5) % 2) as u16;
                    haplotypes.set_allele(marker, state, Some(allele)).unwrap();
                }
            }
            haplotypes.mark_complete();
            CompositePanel {
                haplotypes,
                source_samples: vec![2; STATES],
                marker_indices: (0..MARKERS).collect(),
                allele_counts: vec![2; MARKERS],
            }
        };

        // Case 1: single-track (homozygous + missing across CHECKPOINT_STRIDE=256 boundaries)
        // Case 2: three-track (unphased het, consecutive unphased het across 255/256, masked het, missing)
        for three_track in [false, true] {
            let mut masked0 = vec![0u64; MARKERS.div_ceil(64)];
            let mut masked1 = vec![0u64; MARKERS.div_ceil(64)];
            let rows = (0..MARKERS)
                .map(|marker| {
                    let call0 = if !three_track {
                        if marker % 5 == 0 {
                            let a = ((marker / 5) % 2) as u16;
                            DiploidCall {
                                alleles: [Some(a), Some(a)],
                                phased: false,
                            }
                        } else {
                            DiploidCall::missing(false)
                        }
                    } else if matches!(marker, 2 | 120 | 255 | 256 | 257 | 511 | 512) {
                        DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: false,
                        }
                    } else if marker == 258 {
                        masked0[marker / 64] |= 1u64 << (marker % 64);
                        DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: false,
                        }
                    } else if marker % 4 == 0 {
                        let a = ((marker / 4) % 2) as u16;
                        DiploidCall {
                            alleles: [Some(a), Some(a)],
                            phased: false,
                        }
                    } else {
                        DiploidCall::missing(false)
                    };
                    let call1 = if !three_track {
                        if marker % 7 == 0 {
                            let a = ((marker / 7) % 2) as u16;
                            DiploidCall {
                                alleles: [Some(a), Some(a)],
                                phased: false,
                            }
                        } else {
                            DiploidCall::missing(false)
                        }
                    } else if matches!(marker, 3 | 254 | 255 | 256 | 400 | 512) {
                        DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: false,
                        }
                    } else if marker == 257 {
                        masked1[marker / 64] |= 1u64 << (marker % 64);
                        DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: false,
                        }
                    } else if marker % 6 == 0 {
                        let a = ((marker / 6) % 2) as u16;
                        DiploidCall {
                            alleles: [Some(a), Some(a)],
                            phased: false,
                        }
                    } else {
                        DiploidCall::missing(false)
                    };
                    VariantRow {
                        meta: VariantMeta {
                            chrom: "1".to_owned(),
                            pos: marker as u64 + 1,
                            id: format!("v{marker}"),
                            ref_allele: "A".to_owned(),
                            alt_alleles: vec!["G".to_owned()],
                        },
                        calls: DiploidCalls::from_calls(vec![call0, call1]),
                    }
                })
                .collect::<Vec<_>>();
            let input = PhaseWindow {
                chrom: "1".to_owned(),
                index: 0,
                rows,
                cm_positions: (0..MARKERS).map(|m| m as f64 * 0.01).collect(),
                incoming_overlap_len: 0,
                emit_start: 0,
                emit_end: MARKERS,
            };
            let snapshot = PhasedWindow {
                chrom: input.chrom.clone(),
                index: input.index,
                rows: input
                    .rows
                    .iter()
                    .map(|row| PhasedRow {
                        meta: row.meta.clone(),
                        calls: DiploidCalls::from_calls(
                            row.calls
                                .iter()
                                .map(|call| DiploidCall {
                                    alleles: call.alleles.map(|a| a.or(Some(0))),
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
                emit_end: MARKERS,
            };
            let packed_target = PackedHaplotypes::complete_biallelic_state_major(&snapshot)
                .unwrap()
                .unwrap();
            let panel0 = make_panel(0);
            let panel1 = make_panel(1);
            let template0 = build_cluster_template(&input, 0).unwrap();
            let template1 = build_cluster_template(&input, 1).unwrap();
            let finished = vec![0u64; MARKERS.div_ceil(64)];
            let config = HMMConfig {
                ne: 1_000,
                recombination_intensity: 5.0,
                err: 0.01,
                em: false,
                burnin: 1,
                iterations: 1,
                initial_lr: 100_000.0,
                phase_states: STATES,
                haplotype_count: 16,
            };
            let transition_factors = marker_no_recombination_probabilities(
                &input.cm_positions,
                config.recombination_intensity,
            )
            .unwrap();

            let mut full_ws0 = PhaseBaumWorkspace::default();
            let mut full_ws1 = PhaseBaumWorkspace::default();
            let (full0, full1) = super::update_sample_pair_impl_with_budget(
                &input,
                &snapshot,
                0,
                1,
                &template0,
                &template1,
                &finished,
                &finished,
                three_track,
                &masked0,
                &masked1,
                &panel0,
                &panel1,
                Some(&packed_target),
                &transition_factors,
                &config,
                &mut full_ws0,
                &mut full_ws1,
                true,
                None,
            )
            .unwrap();

            let mut ckpt_ws0 = PhaseBaumWorkspace::default();
            let mut ckpt_ws1 = PhaseBaumWorkspace::default();
            let (ckpt0, ckpt1) = super::update_sample_pair_impl_with_budget(
                &input,
                &snapshot,
                0,
                1,
                &template0,
                &template1,
                &finished,
                &finished,
                three_track,
                &masked0,
                &masked1,
                &panel0,
                &panel1,
                Some(&packed_target),
                &transition_factors,
                &config,
                &mut ckpt_ws0,
                &mut ckpt_ws1,
                true,
                Some(1),
            )
            .unwrap();

            for (full, ckpt) in [(&full0, &ckpt0), (&full1, &ckpt1)] {
                assert_eq!(ckpt.changes, full.changes);
                assert_eq!(ckpt.phase_swap_count, full.phase_swap_count);
                assert_eq!(
                    ckpt.unphased_heterozygote_count,
                    full.unphased_heterozygote_count
                );
                assert_eq!(ckpt.phase_confidences.len(), full.phase_confidences.len());
                for ((m_c, c_c), (m_f, c_f)) in
                    ckpt.phase_confidences.iter().zip(&full.phase_confidences)
                {
                    assert_eq!(m_c, m_f);
                    assert_eq!(c_c.to_bits(), c_f.to_bits());
                }
            }

            let full_beta_bytes = full_ws0.tracked_beta_bytes;
            let ckpt_beta_bytes = ckpt_ws0.tracked_beta_bytes;
            assert!(
                ckpt_beta_bytes < full_beta_bytes,
                "checkpointed beta storage ({ckpt_beta_bytes} B) should be smaller than full ({full_beta_bytes} B) for three_track={three_track}"
            );
            assert!(full_ws0.tracked_event_workspace_bytes > 0);
            assert!(ckpt_ws0.tracked_event_workspace_bytes > 0);
            let mem_while_alive = super::memory_profile_totals();
            assert!(mem_while_alive.process_peak_live_beta_cache_bytes >= full_beta_bytes as u64);
            assert!(mem_while_alive.window_peak_live_beta_cache_bytes >= full_beta_bytes as u64);
            assert!(mem_while_alive.process_peak_live_event_workspace_bytes > 0);
            assert!(mem_while_alive.window_peak_live_event_workspace_bytes > 0);

            // Reusing a workspace previously grown for full cache must shrink below full size
            // when a tight budget is enabled dynamically.
            let _ = super::update_sample_pair_impl_with_budget(
                &input,
                &snapshot,
                0,
                1,
                &template0,
                &template1,
                &finished,
                &finished,
                three_track,
                &masked0,
                &masked1,
                &panel0,
                &panel1,
                Some(&packed_target),
                &transition_factors,
                &config,
                &mut full_ws0,
                &mut full_ws1,
                true,
                Some(1),
            )
            .unwrap();
            assert!(
                full_ws0.tracked_beta_bytes < full_beta_bytes,
                "recycled workspace beta storage ({} B) should shrink below previous full cache ({full_beta_bytes} B) when checkpoint budget is enabled",
                full_ws0.tracked_beta_bytes
            );

            // Reusing a checkpointed workspace after disabling the budget must release
            // beta_checkpoints capacity and drop it from tracked_beta_bytes.
            assert!(ckpt_ws0.beta_checkpoints.capacity() > 0);
            let (reverted0, reverted1) = super::update_sample_pair_impl_with_budget(
                &input,
                &snapshot,
                0,
                1,
                &template0,
                &template1,
                &finished,
                &finished,
                three_track,
                &masked0,
                &masked1,
                &panel0,
                &panel1,
                Some(&packed_target),
                &transition_factors,
                &config,
                &mut ckpt_ws0,
                &mut ckpt_ws1,
                true,
                None,
            )
            .unwrap();
            assert_eq!(ckpt_ws0.beta_checkpoints.capacity(), 0);
            assert_eq!(ckpt_ws1.beta_checkpoints.capacity(), 0);
            assert_eq!(
                ckpt_ws0.tracked_beta_bytes,
                super::vec_cap_bytes(&ckpt_ws0.beta_cache)
            );
            assert_eq!(reverted0.changes, full0.changes);
            assert_eq!(reverted1.changes, full1.changes);

            super::reset_window_memory_peaks();
            let after_reset = super::memory_profile_totals();
            assert!(
                after_reset.window_peak_live_beta_cache_bytes >= after_reset.live_beta_cache_bytes
            );
            assert!(
                after_reset.process_peak_live_beta_cache_bytes
                    >= after_reset.window_peak_live_beta_cache_bytes
            );

            drop(full_ws0);
            drop(full_ws1);
            drop(ckpt_ws0);
            drop(ckpt_ws1);
        }
    }
}
