//! Conservative active-window memory estimates and worker budgeting.

use super::types::{
    DiploidCalls, Ibd2Interval, MemoryEstimate, PhasedRow, VariantMeta, VariantRow,
};

pub(crate) trait RowStorage {
    fn meta(&self) -> &VariantMeta;
    fn calls_bytes(&self) -> usize;
    fn row_struct_bytes(&self) -> usize;
}

impl RowStorage for VariantRow {
    fn meta(&self) -> &VariantMeta {
        &self.meta
    }

    fn calls_bytes(&self) -> usize {
        self.calls.bytes()
    }

    fn row_struct_bytes(&self) -> usize {
        std::mem::size_of::<VariantRow>()
    }
}

impl RowStorage for PhasedRow {
    fn meta(&self) -> &VariantMeta {
        &self.meta
    }

    fn calls_bytes(&self) -> usize {
        self.calls.bytes()
    }

    fn row_struct_bytes(&self) -> usize {
        std::mem::size_of::<PhasedRow>()
    }
}

const PHASE_CHECKPOINT_STRIDE: usize = super::phase_baum::CHECKPOINT_STRIDE;
const STAGE2_CHECKPOINT_STRIDE: usize = super::hmm::STAGE2_FORWARD_CHECKPOINT_STRIDE;

/// Read current process `(VmRSS, VmHWM)` in bytes from `/proc/self/status` on Linux.
/// Returns `None` on platforms without procfs.
pub(crate) fn read_process_rss_hwm_bytes() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut rss_kb = None;
    let mut hwm_kb = None;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss_kb = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<u64>().ok());
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm_kb = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<u64>().ok());
        }
    }
    Some((rss_kb?.saturating_mul(1024), hwm_kb?.saturating_mul(1024)))
}

/// Bound per-worker Stage 1 PhaseBaum backward cache (2 paired samples/worker,
/// two haplotype tracks) using the active `JX_IMP_PHASE_BETA_BUDGET_MB` budget.
pub(crate) fn budgeted_phase_checkpoint_bytes_per_worker(
    target_markers: usize,
    phase_states: usize,
    budget_bytes: Option<usize>,
) -> Result<usize, String> {
    // A paired PhaseBaum workspace keeps the two haplotype tracks in its beta
    // cache. The genotype track is a separate alpha vector and is accounted
    // for by the HMM worker scratch.
    let per_marker_two_track = phase_states
        .checked_mul(2 * std::mem::size_of::<f32>())
        .ok_or_else(|| "phase checkpoint per-marker overflow".to_owned())?;
    let full_per_sample = target_markers
        .checked_mul(per_marker_two_track)
        .ok_or_else(|| "phase full beta cache overflow".to_owned())?;
    let per_sample = if let Some(budget) = budget_bytes {
        if full_per_sample <= budget {
            full_per_sample
        } else {
            // The checkpointed runtime stores only the event slots in one
            // marker block plus one state vector per block. A block can contain
            // at most 256 markers, and every marker can be an event.
            let block_markers = PHASE_CHECKPOINT_STRIDE.min(target_markers);
            let checkpoints = target_markers.div_ceil(PHASE_CHECKPOINT_STRIDE);
            block_markers
                .checked_add(checkpoints)
                .and_then(|slots| slots.checked_mul(per_marker_two_track))
                .ok_or_else(|| "phase budgeted checkpoint overflow".to_owned())?
        }
    } else {
        full_per_sample
    };
    per_sample
        .checked_mul(2)
        .ok_or_else(|| "phase pair-worker beta cache overflow".to_owned())
}

/// Bound per-worker selected-forward / imputation checkpoint storage (2
/// haplotypes/worker, single-track) using the active
/// `JX_IMP_STAGE2_FORWARD_BUDGET_MB` budget and selected anchor count.
pub(crate) fn budgeted_imputation_checkpoint_bytes_per_worker(
    selected_markers: usize,
    union_markers: usize,
    imp_states: usize,
    budget_bytes: Option<usize>,
) -> Result<usize, String> {
    let per_marker = imp_states
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "imputation checkpoint per-marker overflow".to_owned())?;
    let full_per_hap = selected_markers
        .checked_mul(per_marker)
        .ok_or_else(|| "imputation full forward cache overflow".to_owned())?;
    let per_hap = if let Some(budget) = budget_bytes {
        let checkpoints_required = union_markers > STAGE2_CHECKPOINT_STRIDE
            && selected_markers > 0
            && full_per_hap > budget;
        if !checkpoints_required {
            full_per_hap
        } else {
            // Stage 2 checkpoints are indexed over every union marker. Keep
            // the budget allowance plus the independently allocated checkpoint
            // state vectors; Vec capacity may remain above the requested
            // shrink target, so counting only the active block would
            // underestimate the real allocation.
            let checkpoint_blocks = union_markers.div_ceil(STAGE2_CHECKPOINT_STRIDE);
            let checkpoint_bytes = checkpoint_blocks
                .checked_mul(per_marker)
                .ok_or_else(|| "imputation checkpoint state storage overflow".to_owned())?;
            budget
                .checked_add(checkpoint_bytes)
                .ok_or_else(|| "imputation budgeted checkpoint overflow".to_owned())?
                .min(full_per_hap)
        }
    } else {
        full_per_hap
    };
    per_hap
        .checked_mul(2)
        .ok_or_else(|| "imputation pair-worker checkpoint overflow".to_owned())
}

/// Update imputation/Stage-2 checkpoint bytes per worker using the actual
/// selected anchor marker count rather than the full union marker count.
pub(crate) fn set_imputation_selected_markers(
    estimate: &mut MemoryEstimate,
    selected_markers: usize,
    union_markers: usize,
    imp_states: usize,
) -> Result<(), String> {
    estimate.imputation_checkpoint_bytes_per_worker =
        budgeted_imputation_checkpoint_bytes_per_worker(
            selected_markers,
            union_markers,
            imp_states,
            super::hmm::stage2_forward_budget_bytes(),
        )?;
    recompute_stage_totals(estimate)
}

pub(crate) fn set_phase_stage2_selected_markers(
    estimate: &mut MemoryEstimate,
    selected_markers: usize,
    target_markers: usize,
    phase_states: usize,
) -> Result<(), String> {
    let stage2_states = (phase_states / 2).max(1);
    estimate.phase_stage2_checkpoint_bytes_per_worker =
        budgeted_imputation_checkpoint_bytes_per_worker(
            selected_markers,
            target_markers,
            stage2_states,
            super::hmm::stage2_forward_budget_bytes(),
        )?;
    recompute_stage_totals(estimate)
}

/// Recompute lifecycle stage peaks (phasing sub-stages: initial PBWT, Stage 1,
/// and Stage 2; imputation+streaming output; and pure output) and set
/// `total_bytes` to their maximum rather than summing mutually exclusive stages.
pub(crate) fn recompute_stage_totals(estimate: &mut MemoryEstimate) -> Result<(), String> {
    let phase_common_fixed = estimate
        .input_window_bytes
        .checked_add(estimate.phase_snapshot_bytes)
        .ok_or_else(|| "phase lifecycle common fixed memory estimate overflow".to_owned())?;

    let phase_init_pbwt_stage_bytes = if estimate.phase_initial_pbwt_bytes > 0 {
        phase_common_fixed
            .checked_add(estimate.phase_initial_pbwt_bytes)
            .ok_or_else(|| "initial PBWT lifecycle memory estimate overflow".to_owned())?
    } else {
        0
    };

    let phase_stage1_fixed = phase_common_fixed
        .checked_add(estimate.composite_panel_bytes)
        .and_then(|value| value.checked_add(estimate.phase_stage1_pbwt_donors_bytes))
        .ok_or_else(|| "phase stage1 fixed memory estimate overflow".to_owned())?;
    let phase_stage1_per_worker = estimate
        .checkpoint_bytes_per_worker
        .checked_add(estimate.phase_event_workspace_bytes_per_worker)
        .and_then(|value| value.checked_add(estimate.scratch_bytes_per_worker))
        .ok_or_else(|| "phase stage1 per-worker memory estimate overflow".to_owned())?;
    let phase_stage1_worker_bytes = phase_stage1_per_worker
        .checked_mul(estimate.worker_count)
        .ok_or_else(|| "phase stage1 worker pool memory estimate overflow".to_owned())?;
    let phase_stage1_bytes = phase_stage1_fixed
        .checked_add(phase_stage1_worker_bytes)
        .ok_or_else(|| "phase stage1 total memory estimate overflow".to_owned())?;

    let phase_stage2_bytes = if estimate.phase_stage2_pbwt_donors_bytes > 0 {
        let stage2_fixed = phase_common_fixed
            .checked_add(estimate.phase_stage2_pbwt_donors_bytes)
            .ok_or_else(|| "phase stage2 fixed memory estimate overflow".to_owned())?;
        let stage2_per_worker = estimate
            .phase_stage2_checkpoint_bytes_per_worker
            .checked_add(estimate.phase_stage2_scratch_bytes_per_worker)
            .ok_or_else(|| "phase stage2 per-worker memory estimate overflow".to_owned())?;
        let stage2_worker_bytes = stage2_per_worker
            .checked_mul(estimate.worker_count)
            .ok_or_else(|| "phase stage2 worker pool memory estimate overflow".to_owned())?;
        stage2_fixed
            .checked_add(stage2_worker_bytes)
            .ok_or_else(|| "phase stage2 total memory estimate overflow".to_owned())?
    } else {
        0
    };

    let phase_stage_bytes = phase_init_pbwt_stage_bytes
        .max(phase_stage1_bytes)
        .max(phase_stage2_bytes);

    let imputation_active = estimate.imputation_fixed_bytes > 0
        || estimate.imputation_checkpoint_bytes_per_worker > 0
        || estimate.imputation_scratch_bytes_per_worker > 0;
    let imputation_stage_bytes = if imputation_active {
        let imp_fixed = estimate
            .input_window_bytes
            .checked_add(estimate.imputation_fixed_bytes)
            .and_then(|value| value.checked_add(estimate.output_buffer_bytes))
            .ok_or_else(|| "imputation lifecycle fixed memory estimate overflow".to_owned())?;
        let imp_per_worker = estimate
            .imputation_checkpoint_bytes_per_worker
            .checked_add(estimate.imputation_scratch_bytes_per_worker)
            .and_then(|value| value.checked_add(estimate.output_scratch_bytes_per_worker))
            .ok_or_else(|| "imputation lifecycle per-worker memory estimate overflow".to_owned())?;
        let imp_worker_bytes = imp_per_worker
            .checked_mul(estimate.worker_count)
            .ok_or_else(|| {
                "imputation lifecycle worker pool memory estimate overflow".to_owned()
            })?;
        imp_fixed
            .checked_add(imp_worker_bytes)
            .ok_or_else(|| "imputation lifecycle total memory estimate overflow".to_owned())?
    } else {
        0
    };

    let output_active =
        estimate.output_buffer_bytes > 0 || estimate.output_scratch_bytes_per_worker > 0;
    let output_stage_bytes = if output_active {
        let out_worker_bytes = estimate
            .output_scratch_bytes_per_worker
            .checked_mul(estimate.worker_count)
            .ok_or_else(|| "output lifecycle worker pool memory estimate overflow".to_owned())?;
        estimate
            .output_buffer_bytes
            .checked_add(out_worker_bytes)
            .ok_or_else(|| "output lifecycle total memory estimate overflow".to_owned())?
    } else {
        0
    };

    estimate.phase_init_pbwt_stage_bytes = phase_init_pbwt_stage_bytes;
    estimate.phase_stage1_bytes = phase_stage1_bytes;
    estimate.phase_stage2_bytes = phase_stage2_bytes;
    estimate.phase_stage_bytes = phase_stage_bytes;
    estimate.imputation_stage_bytes = imputation_stage_bytes;
    estimate.output_stage_bytes = output_stage_bytes;
    estimate.total_bytes = phase_stage_bytes
        .max(imputation_stage_bytes)
        .max(output_stage_bytes);
    Ok(())
}

/// Reserve bounded streaming-writer memory for one BGZF batch. Each worker
/// can hold a raw block, an encoded block, and a temporary Deflate buffer.
/// The extra fixed block is the writer's active fill buffer; 64 KiB also
/// conservatively covers the file BufWriter and BGZF framing overhead.
pub(crate) fn include_bgzf_writer_workspace(estimate: &mut MemoryEstimate) -> Result<(), String> {
    const BLOCK_ALLOWANCE: usize = 65_536;
    const FIXED_BYTES: usize = BLOCK_ALLOWANCE + 8 * 1024;
    const WORKER_BYTES: usize = 3 * BLOCK_ALLOWANCE;

    estimate.output_buffer_bytes = estimate
        .output_buffer_bytes
        .checked_add(FIXED_BYTES)
        .ok_or_else(|| "BGZF writer workspace estimate overflow".to_owned())?;
    estimate.output_scratch_bytes_per_worker = estimate
        .output_scratch_bytes_per_worker
        .checked_add(WORKER_BYTES)
        .ok_or_else(|| "BGZF writer workspace estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

pub(crate) fn estimate_window(
    target_samples: usize,
    reference_samples: usize,
    target_markers: usize,
    union_markers: usize,
    phase_states: usize,
    imp_states: usize,
    bits_per_allele: u8,
    workers: usize,
) -> Result<MemoryEstimate, String> {
    if target_samples == 0
        || target_markers == 0
        || union_markers < target_markers
        || phase_states == 0
        || imp_states == 0
        || bits_per_allele == 0
        || workers == 0
    {
        return Err("memory estimate dimensions and worker count must be positive".to_owned());
    }
    let phase_samples = target_samples
        .checked_add(reference_samples)
        .ok_or_else(|| "phase sample count overflow".to_owned())?;
    let checked_mul = |a: usize, b: usize, label: &str| {
        a.checked_mul(b)
            .ok_or_else(|| format!("{label} memory estimate overflow"))
    };
    let checked_add = |a: usize, b: usize, label: &str| {
        a.checked_add(b)
            .ok_or_else(|| format!("{label} memory estimate overflow"))
    };
    let haplotypes = checked_mul(phase_samples, 2, "haplotype count")?;
    // DiploidCalls stores common biallelic hard calls in half a byte/sample.
    // Extra storage for complex partial-missing or multiallelic calls is added
    // from the actual active rows once they have been read.
    let input_row_bytes = checked_add(target_samples.div_ceil(2), 256, "input row")?;
    // The input cursor holds the active window while cloning its overlap into
    // the next cursor state; reserve two target-window row sets. Reference
    // rows are accounted separately by `include_reference_window`.
    let input_window_bytes = checked_mul(
        checked_mul(target_markers, input_row_bytes, "input window")?,
        2,
        "input window and overlap",
    )?;
    let haplotype_bytes_per_state = union_markers
        .checked_mul(usize::from(bits_per_allele) + 1)
        .ok_or_else(|| "haplotype memory estimate overflow".to_owned())?
        .div_ceil(8);
    // Reserve one packed haplotype matrix and three haplotype-sized PBWT
    // buffers. The sort uses order/inverse/reordered arrays; phase rounds
    // reuse this allowance for the shared order/inverse index.
    let allele_buckets = 1usize
        .checked_shl(u32::from(bits_per_allele))
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| "PBWT bucket memory estimate overflow".to_owned())?;
    let pbwt_order_bytes = checked_mul(
        haplotypes,
        3 * std::mem::size_of::<usize>(),
        "PBWT order arrays",
    )?;
    let pbwt_bucket_bytes = checked_mul(
        allele_buckets,
        2 * std::mem::size_of::<usize>(),
        "PBWT counting-sort buckets",
    )?;
    let phase_haplotype_bytes = checked_add(
        checked_add(
            checked_mul(haplotypes, haplotype_bytes_per_state, "PBWT haplotypes")?,
            pbwt_order_bytes,
            "PBWT workspace",
        )?,
        pbwt_bucket_bytes,
        "PBWT workspace",
    )?;
    let progressive_phase_bytes = checked_mul(
        phase_samples,
        union_markers
            .div_ceil(64)
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| "progressive phase-state memory estimate overflow".to_owned())?,
        "progressive phase state",
    )?;
    let phase_snapshot_bytes = checked_add(
        phase_haplotype_bytes,
        progressive_phase_bytes,
        "phase snapshot",
    )?;
    // The phasing path constructs one shared composite panel for the genotype
    // and two haplotype tracks inside each worker.
    let composite_panel_bytes = 0usize;
    let segment_markers = PHASE_CHECKPOINT_STRIDE.min(union_markers);
    let marker_marginal_scratch = checked_mul(
        union_markers,
        100,
        "HMM transitions, marker marginals, and phase ranking",
    )?;

    // Phasing lifecycle per-worker workspace (bounded by `phase_states` and the
    // active `JX_IMP_PHASE_BETA_BUDGET_MB` budget across 2 paired samples/worker).
    let checkpoint_bytes_per_worker = budgeted_phase_checkpoint_bytes_per_worker(
        target_markers,
        phase_states,
        super::phase_baum::phase_beta_cache_budget_bytes(),
    )?;
    // Dedicated PhaseBaum dynamic event workspace for 2 paired samples/worker.
    // A marker is an upper bound for the number of event clusters. The
    // emission-code term assumes one observed row per marker; this is a
    // calibration heuristic rather than a two-row heterozygous worst case.
    let event_meta_per_pair = checked_mul(
        target_markers,
        2 * super::phase_baum::EVENT_META_BYTES_PER_CLUSTER,
        "phase event metadata workspace",
    )?;
    let event_emission_codes_per_pair = checked_mul(
        target_markers,
        phase_states,
        "phase event emission_codes workspace",
    )?;
    let phase_event_workspace_bytes_per_worker = checked_add(
        event_meta_per_pair,
        event_emission_codes_per_pair,
        "phase event workspace",
    )?;
    let phase_per_marker_scratch = checked_mul(
        phase_states,
        6 * std::mem::size_of::<f32>(),
        "phase HMM scratch",
    )?;
    // update_sample keeps local/carry/projected backward arrays, forward
    // tracks, event and block scratch, plus the genotype alpha vector.
    let phase_hmm_vector_scratch = checked_mul(
        checked_mul(phase_states, 19, "phase HMM vector scratch")?,
        std::mem::size_of::<f32>(),
        "phase HMM vector scratch",
    )?;
    let phase_panel_bytes = checked_mul(
        phase_states,
        haplotype_bytes_per_state,
        "phase reference panel",
    )?
    .checked_add(
        union_markers
            .checked_mul(2 * std::mem::size_of::<usize>())
            .ok_or_else(|| "phase panel marker-index estimate overflow".to_owned())?,
    )
    .and_then(|value| {
        phase_states
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|samples| value.checked_add(samples))
    })
    .ok_or_else(|| "phase reference panel memory estimate overflow".to_owned())?;
    let phase_hmm_scratch = checked_add(
        checked_mul(
            segment_markers,
            phase_per_marker_scratch,
            "phase HMM segment scratch",
        )?,
        checked_add(
            phase_hmm_vector_scratch,
            marker_marginal_scratch,
            "phase HMM scratch",
        )?,
        "phase HMM scratch",
    )?;
    let mosaic_rank_states = phase_states
        .checked_mul(2)
        .ok_or_else(|| "mosaic ranking memory estimate overflow".to_owned())?;
    let mosaic_rank_scratch = checked_mul(
        mosaic_rank_states,
        std::mem::size_of::<i64>() + 2 * std::mem::size_of::<usize>(),
        "mosaic ranking scratch",
    )?;
    let scratch_bytes_per_worker = checked_add(
        checked_add(
            phase_panel_bytes,
            phase_hmm_scratch,
            "per-worker phase panel and HMM scratch",
        )?,
        mosaic_rank_scratch,
        "per-worker phase panel, HMM, and mosaic scratch",
    )?;

    // Imputation lifecycle per-worker workspace (single-track, bounded by
    // `imp_states`, target/selected markers, and the active Stage 2 budget).
    let imputation_checkpoint_bytes_per_worker = budgeted_imputation_checkpoint_bytes_per_worker(
        target_markers,
        union_markers,
        imp_states,
        super::hmm::stage2_forward_budget_bytes(),
    )?;
    let imputation_per_marker_scratch = checked_mul(
        imp_states,
        2 * std::mem::size_of::<f32>(),
        "imputation HMM scratch",
    )?;
    let imputation_hmm_vector_scratch = checked_mul(
        checked_mul(imp_states, 8, "imputation HMM vector scratch")?,
        std::mem::size_of::<f32>(),
        "imputation HMM vector scratch",
    )?;
    let mosaic_panel_bytes =
        checked_mul(imp_states, haplotype_bytes_per_state, "target mosaic panel")?
            .checked_add(
                union_markers
                    .checked_mul(2 * std::mem::size_of::<usize>())
                    .ok_or_else(|| "target mosaic marker-index estimate overflow".to_owned())?,
            )
            .and_then(|value| {
                imp_states
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|samples| value.checked_add(samples))
            })
            .ok_or_else(|| "target mosaic panel memory estimate overflow".to_owned())?;
    let imputation_hmm_scratch = checked_add(
        checked_mul(
            segment_markers,
            imputation_per_marker_scratch,
            "imputation HMM segment scratch",
        )?,
        checked_add(
            imputation_hmm_vector_scratch,
            marker_marginal_scratch,
            "imputation HMM scratch",
        )?,
        "imputation HMM scratch",
    )?;
    let imputation_scratch_bytes_per_worker = checked_add(
        mosaic_panel_bytes,
        imputation_hmm_scratch,
        "per-worker imputation panel and HMM scratch",
    )?;

    // Stage 2 low-frequency phasing per-worker workspace (2 haplotypes/worker,
    // `stage2_states = (phase_states / 2).max(1)`).
    let stage2_states = (phase_states / 2).max(1);
    let phase_stage2_checkpoint_bytes_per_worker =
        budgeted_imputation_checkpoint_bytes_per_worker(
            target_markers,
            target_markers,
            stage2_states,
            super::hmm::stage2_forward_budget_bytes(),
        )?;
    let stage2_panel_bytes = checked_mul(
        stage2_states,
        haplotype_bytes_per_state,
        "stage2 reference panel",
    )?
    .checked_add(
        target_markers
            .checked_mul(2 * std::mem::size_of::<usize>())
            .ok_or_else(|| "stage2 panel marker-index estimate overflow".to_owned())?,
    )
    .and_then(|value| {
        stage2_states
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|samples| value.checked_add(samples))
    })
    .ok_or_else(|| "stage2 reference panel memory estimate overflow".to_owned())?;
    // Two haplotype panels + paired marginals + streaming Stage2DonorCursor.
    let phase_stage2_scratch_bytes_per_worker = checked_add(
        checked_mul(stage2_panel_bytes, 2, "stage2 paired panels")?,
        checked_add(
            checked_mul(target_markers, 4 * std::mem::size_of::<f32>(), "stage2 marginals")?,
            checked_mul(stage2_states, 4 * std::mem::size_of::<usize>(), "stage2 donor cursors")?,
            "stage2 marginals and donor cursors",
        )?,
        "per-worker stage2 scratch",
    )?;

    let output_buffer_bytes = checked_mul(target_samples, 256, "output buffer")?;
    let mut estimate = MemoryEstimate {
        input_window_bytes,
        phase_snapshot_bytes,
        phase_initial_pbwt_bytes: 0,
        phase_stage1_pbwt_donors_bytes: 0,
        phase_stage2_pbwt_donors_bytes: 0,
        composite_panel_bytes,
        checkpoint_bytes_per_worker,
        phase_event_workspace_bytes_per_worker,
        scratch_bytes_per_worker,
        phase_stage2_checkpoint_bytes_per_worker,
        phase_stage2_scratch_bytes_per_worker,
        output_buffer_bytes,
        imputation_fixed_bytes: 0,
        imputation_checkpoint_bytes_per_worker,
        imputation_scratch_bytes_per_worker,
        output_scratch_bytes_per_worker: 0,
        phase_init_pbwt_stage_bytes: 0,
        phase_stage1_bytes: 0,
        phase_stage2_bytes: 0,
        phase_stage_bytes: 0,
        imputation_stage_bytes: 0,
        output_stage_bytes: 0,
        worker_count: workers,
        total_bytes: 0,
    };
    recompute_stage_totals(&mut estimate)?;
    Ok(estimate)
}

/// Account for the two dense, per-marker tracks retained by the initial
/// reverse/forward PBWT passes plus both live per-haplotype phasers. The
/// marker bound is computed from the same fixed logical chunks used at run
/// time, so this remains independent of the Rayon worker count.
pub(crate) fn include_initial_phase_pbwt_workspace_exact(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    reference_samples: usize,
    n_markers: usize,
    max_chunk_markers: usize,
    bits_per_allele: u8,
    workers: usize,
) -> Result<(), String> {
    if target_samples == 0
        || n_markers == 0
        || max_chunk_markers == 0
        || bits_per_allele == 0
        || workers == 0
    {
        return Err("initial PBWT memory dimensions must be positive".to_owned());
    }
    let target_haplotypes = target_samples
        .checked_mul(2)
        .ok_or_else(|| "initial PBWT haplotype count overflow".to_owned())?;
    let phase_samples = target_samples
        .checked_add(reference_samples)
        .ok_or_else(|| "initial PBWT phase sample count overflow".to_owned())?;
    let phase_haplotypes = phase_samples
        .checked_mul(2)
        .ok_or_else(|| "initial PBWT phase haplotype count overflow".to_owned())?;
    let (retained_forward_bytes, live_pair_bytes) = if bits_per_allele == 1 {
        // Biallelic PhaseTrack::Packed stores an allele bit and a missing bit
        // for each target haplotype. Completed chunks retain one forward
        // track; each active worker can additionally hold reverse+forward
        // tracks for one logical chunk.
        let retained_bits = target_haplotypes
            .checked_mul(n_markers)
            .and_then(|cells| cells.checked_mul(2))
            .ok_or_else(|| "initial PBWT retained track overflow".to_owned())?;
        let live_bits = target_haplotypes
            .checked_mul(max_chunk_markers)
            .and_then(|cells| cells.checked_mul(2 * 2 * workers))
            .ok_or_else(|| "initial PBWT live tracks overflow".to_owned())?;
        (retained_bits.div_ceil(8), live_bits.div_ceil(8))
    } else {
        let retained = target_haplotypes
            .checked_mul(n_markers)
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "initial scalar PBWT retained track overflow".to_owned())?;
        let live = target_haplotypes
            .checked_mul(max_chunk_markers)
            .and_then(|cells| cells.checked_mul(2 * std::mem::size_of::<u16>()))
            .and_then(|bytes| bytes.checked_mul(workers))
            .ok_or_else(|| "initial scalar PBWT live tracks overflow".to_owned())?;
        (retained, live)
    };
    let marker_words_bytes = target_haplotypes
        .div_ceil(64)
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| "initial PBWT marker words overflow".to_owned())?;
    // Each live PbwtRecPhaser owns order/inverse/reorder arrays, allele values,
    // and per-allele buckets. Include headroom for their Vec capacities, the
    // per-sample het/alignment vectors, and small allocator metadata.
    let phaser_bytes = phase_haplotypes
        .checked_mul(100)
        .and_then(|bytes| {
            phase_samples
                .checked_mul(64)
                .and_then(|sample_bytes| bytes.checked_add(sample_bytes))
        })
        .and_then(|bytes| bytes.checked_add(16 * 1024))
        .ok_or_else(|| "initial PBWT phaser memory estimate overflow".to_owned())?;
    let bytes = retained_forward_bytes
        .checked_add(live_pair_bytes)
        .and_then(|value| value.checked_add(marker_words_bytes))
        .and_then(|value| value.checked_add(phaser_bytes))
        .ok_or_else(|| "initial PBWT workspace memory estimate overflow".to_owned())?;
    estimate.phase_initial_pbwt_bytes = estimate
        .phase_initial_pbwt_bytes
        .checked_add(bytes)
        .ok_or_else(|| "initial PBWT phase memory estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

#[cfg(test)]
fn include_initial_phase_pbwt_workspace(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    max_chunk_markers: usize,
) -> Result<(), String> {
    include_initial_phase_pbwt_workspace_exact(
        estimate,
        target_samples,
        target_samples,
        max_chunk_markers,
        max_chunk_markers,
        2,
        2,
    )
}

pub(crate) fn include_reference_window(
    estimate: &mut MemoryEstimate,
    reference_samples: usize,
    n_markers: usize,
) -> Result<(), String> {
    let per_row = reference_samples
        .checked_add(1)
        .ok_or_else(|| "reference row memory estimate overflow".to_owned())?
        .div_ceil(2)
        .checked_add(256)
        .ok_or_else(|| "reference row memory estimate overflow".to_owned())?;
    let bytes = n_markers
        .checked_mul(per_row)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "reference window memory estimate overflow".to_owned())?;
    estimate.input_window_bytes = estimate
        .input_window_bytes
        .checked_add(bytes)
        .ok_or_else(|| "reference window memory estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

/// Account for a compact 1-bit reference window (`CompactReferenceWindow`).
/// When spilled to disk during phasing (P1-1b), only per-marker metadata stays
/// resident; when kept in RAM (P1-1a), one 1-bit haplotype bitplane is
/// retained alongside metadata.
pub(crate) fn include_compact_reference_window(
    estimate: &mut MemoryEstimate,
    reference_samples: usize,
    n_markers: usize,
    spilled: bool,
) -> Result<(), String> {
    let allele_per_row = if spilled {
        0
    } else {
        reference_samples
            .checked_mul(2)
            .ok_or_else(|| "compact reference haplotype count overflow".to_owned())?
            .div_ceil(64)
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| "compact reference row memory estimate overflow".to_owned())?
    };
    let per_row = allele_per_row
        .checked_add(256)
        .ok_or_else(|| "compact reference row memory estimate overflow".to_owned())?;
    let bytes = n_markers
        .checked_mul(per_row)
        .ok_or_else(|| "compact reference window memory estimate overflow".to_owned())?;
    estimate.input_window_bytes = estimate
        .input_window_bytes
        .checked_add(bytes)
        .ok_or_else(|| "compact reference window memory estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

pub(crate) fn include_imp_ibs_reference_index(
    estimate: &mut MemoryEstimate,
    reference_haplotypes: usize,
    n_markers: usize,
    step_count: usize,
    workers: usize,
) -> Result<(), String> {
    if reference_haplotypes == 0 || n_markers == 0 || step_count == 0 || workers == 0 {
        return Err("imputation IBS index memory dimensions must be positive".to_owned());
    }
    // ImpIbsReferenceIndex retains three marker/step range vectors, one
    // partitioning marker-index vector, and one sorted hash/donor vector per
    // genetic step. The segment count is the number of imp-step intervals,
    // not the number of markers.
    let range_bytes = step_count
        .checked_mul(3 * 2 * std::mem::size_of::<usize>())
        .ok_or_else(|| "imputation IBS step range memory estimate overflow".to_owned())?;
    let marker_index_bytes = n_markers
        .checked_mul(std::mem::size_of::<usize>())
        .ok_or_else(|| "imputation IBS marker index memory estimate overflow".to_owned())?;
    let per_haplotype_step_bytes = std::mem::size_of::<u64>() + std::mem::size_of::<u32>();
    let retained_step_bytes = reference_haplotypes
        .checked_mul(step_count)
        .and_then(|count| count.checked_mul(per_haplotype_step_bytes))
        .ok_or_else(|| "imputation IBS hash index memory estimate overflow".to_owned())?;
    // Each Rayon worker sorts a temporary (u64, u32) donor entry for the
    // current step. This scratch belongs to the worker lifecycle rather than
    // the fixed shared index, so changing the worker count can re-price it.
    let construction_entry_bytes_per_worker = reference_haplotypes
        .checked_mul(std::mem::size_of::<(u64, u32)>())
        .ok_or_else(|| "imputation IBS construction scratch estimate overflow".to_owned())?;
    let bytes = range_bytes
        .checked_add(marker_index_bytes)
        .and_then(|value| value.checked_add(retained_step_bytes))
        .ok_or_else(|| "imputation IBS index memory estimate overflow".to_owned())?;
    estimate.imputation_fixed_bytes = estimate
        .imputation_fixed_bytes
        .checked_add(bytes)
        .ok_or_else(|| "imputation IBS index memory estimate overflow".to_owned())?;
    estimate.imputation_scratch_bytes_per_worker = estimate
        .imputation_scratch_bytes_per_worker
        .checked_add(construction_entry_bytes_per_worker)
        .ok_or_else(|| "imputation IBS construction scratch estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

pub(crate) fn include_phase_pbwt_donors(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    step_count: usize,
) -> Result<(), String> {
    let bytes = phase_pbwt_donor_storage_bytes(target_samples, step_count, 1)?;
    estimate.phase_stage1_pbwt_donors_bytes = estimate
        .phase_stage1_pbwt_donors_bytes
        .checked_add(bytes)
        .ok_or_else(|| "phase stage1 PBWT donor index memory estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

pub(crate) fn include_low_frequency_phase_pbwt_donors(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    step_count: usize,
) -> Result<(), String> {
    // Two-stage phasing retains 1 proposal copy during Stage 1 and 2 copies
    // during Stage 2; these do not coexist with each other.
    let stage1_bytes = phase_pbwt_donor_storage_bytes(target_samples, step_count, 1)?;
    let stage2_bytes = phase_pbwt_donor_storage_bytes(target_samples, step_count, 2)?;
    estimate.phase_stage1_pbwt_donors_bytes = estimate
        .phase_stage1_pbwt_donors_bytes
        .checked_add(stage1_bytes)
        .ok_or_else(|| "phase stage1 PBWT donor index memory estimate overflow".to_owned())?;
    estimate.phase_stage2_pbwt_donors_bytes = estimate
        .phase_stage2_pbwt_donors_bytes
        .checked_add(stage2_bytes)
        .ok_or_else(|| "phase stage2 PBWT donor index memory estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

fn phase_pbwt_donor_storage_bytes(
    target_samples: usize,
    step_count: usize,
    proposal_copies: usize,
) -> Result<usize, String> {
    let target_haplotypes = target_samples
        .checked_mul(2)
        .ok_or_else(|| "phase PBWT haplotype count overflow".to_owned())?;
    let proposal_bytes = target_haplotypes
        .checked_mul(step_count)
        .and_then(|count| count.checked_mul(proposal_copies))
        .and_then(|count| count.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| "phase PBWT donor proposal memory estimate overflow".to_owned())?;
    let step_range_bytes = step_count
        .checked_mul(2 * std::mem::size_of::<usize>())
        .ok_or_else(|| "phase PBWT step range memory estimate overflow".to_owned())?;
    // The biallelic sequence fast path retains 256 bytes/haplotype for packed
    // step words, plus its open-address table, code/order/divergence arrays,
    // updater buckets, and low-frequency carrier scratch. 512 bytes per
    // haplotype bounds these simultaneously live vectors with capacity slack.
    let scan_scratch_bytes = target_haplotypes
        .checked_mul(512)
        .and_then(|bytes| bytes.checked_add(16 * 1024))
        .ok_or_else(|| "phase PBWT scan scratch memory estimate overflow".to_owned())?;
    proposal_bytes
        .checked_add(step_range_bytes)
        .and_then(|value| value.checked_add(scan_scratch_bytes))
        .ok_or_else(|| "phase PBWT donor index memory estimate overflow".to_owned())
}

pub(crate) fn include_variant_storage<T: RowStorage>(
    estimate: &mut MemoryEstimate,
    rows: &[T],
    n_samples: usize,
    copies: usize,
) -> Result<(), String> {
    let extra = variant_storage_excess_sum(rows, n_samples)?;
    include_row_storage_excess(estimate, extra, copies)
}

pub(crate) fn variant_storage_excess_sum<T: RowStorage>(
    rows: &[T],
    n_samples: usize,
) -> Result<usize, String> {
    rows.iter().try_fold(0usize, |sum, row| {
        sum.checked_add(row_storage_excess(row, n_samples)?)
            .ok_or_else(|| "variant storage estimate overflow".to_owned())
    })
}

pub(crate) fn row_storage_excess<T: RowStorage>(
    row: &T,
    n_samples: usize,
) -> Result<usize, String> {
    let estimated_per_row = n_samples
        .div_ceil(2)
        .checked_add(256)
        .ok_or_else(|| "variant storage estimate overflow".to_owned())?;
    let meta = row.meta();
    let allele_strings = meta.alt_alleles.iter().try_fold(
        meta.alt_alleles
            .capacity()
            .checked_mul(std::mem::size_of::<String>())
            .ok_or_else(|| "variant metadata estimate overflow".to_owned())?,
        |sum, allele| {
            sum.checked_add(allele.capacity())
                .ok_or_else(|| "variant metadata estimate overflow".to_owned())
        },
    )?;
    let actual_per_row = row
        .row_struct_bytes()
        .checked_add(row.calls_bytes())
        .and_then(|bytes| bytes.checked_add(meta.chrom.capacity()))
        .and_then(|bytes| bytes.checked_add(meta.id.capacity()))
        .and_then(|bytes| bytes.checked_add(meta.ref_allele.capacity()))
        .and_then(|bytes| bytes.checked_add(allele_strings))
        .ok_or_else(|| "variant storage estimate overflow".to_owned())?;
    Ok(actual_per_row.saturating_sub(estimated_per_row))
}

pub(crate) fn row_marginal_excess<T: RowStorage>(row: &T) -> usize {
    row.meta().allele_count().saturating_sub(2)
}

pub(crate) fn variant_marginal_excess_sum<T: RowStorage>(rows: &[T]) -> Result<usize, String> {
    rows.iter().try_fold(0usize, |sum, row| {
        sum.checked_add(row_marginal_excess(row))
            .ok_or_else(|| "multiallelic posterior memory estimate overflow".to_owned())
    })
}

pub(crate) fn include_row_storage_excess(
    estimate: &mut MemoryEstimate,
    per_copy_bytes: usize,
    copies: usize,
) -> Result<(), String> {
    let bytes = per_copy_bytes
        .checked_mul(copies)
        .ok_or_else(|| "variant storage estimate overflow".to_owned())?;
    estimate.input_window_bytes = estimate
        .input_window_bytes
        .checked_add(bytes)
        .ok_or_else(|| "variant storage estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

pub(crate) fn phase_expansion_fixed_bytes(n_samples: usize) -> Result<usize, String> {
    // One SampleUpdate plus its DiploidCalls header per target sample, and a
    // small-vector capacity allowance for that sample's complex calls.
    let update_record_bytes = std::mem::size_of::<DiploidCalls>()
        .checked_add(2 * std::mem::size_of::<f64>())
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<usize>()))
        .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?
        .max(128);
    n_samples
        .checked_mul(
            update_record_bytes
                .checked_add(4 * std::mem::size_of::<(usize, super::types::DiploidCall)>())
                .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?,
        )
        .ok_or_else(|| "phase-call memory estimate overflow".to_owned())
}

pub(crate) fn phase_expansion_row_bytes(
    row: &VariantRow,
    n_samples: usize,
) -> Result<usize, String> {
    phase_expansion_row_bytes_with_extra_samples(row, n_samples, 0)
}

pub(crate) fn phase_expansion_row_bytes_with_extra_samples(
    row: &VariantRow,
    target_samples: usize,
    reference_samples: usize,
) -> Result<usize, String> {
    if row.calls.len() != target_samples {
        return Err("phase-call memory estimate sample count differs from target row".to_owned());
    }
    let phase_samples = target_samples
        .checked_add(reference_samples)
        .ok_or_else(|| "phase-call sample count overflow".to_owned())?;
    let mut possible_complex = 0usize;
    for call in row.calls.iter() {
        let has_non_biallelic_allele = call.alleles.into_iter().flatten().any(|allele| allele > 1);
        if has_non_biallelic_allele || (row.meta.allele_count() > 2 && !call.is_complete()) {
            possible_complex = possible_complex
                .checked_add(1)
                .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?;
        }
    }

    // DiploidCalls::from_iter grows a Vec for complex calls. Bound the common
    // geometric-growth slack with two spare entries plus the initial capacity.
    let complex_capacity = if possible_complex == 0 {
        0
    } else {
        possible_complex
            .checked_mul(2)
            .and_then(|count| count.checked_add(4))
            .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?
    };
    let allele_string_bytes = row.meta.alt_alleles.iter().try_fold(
        row.meta
            .alt_alleles
            .capacity()
            .checked_mul(std::mem::size_of::<String>())
            .ok_or_else(|| "phase metadata memory estimate overflow".to_owned())?,
        |sum, allele| {
            sum.checked_add(allele.capacity())
                .ok_or_else(|| "phase metadata memory estimate overflow".to_owned())
        },
    )?;
    let metadata_heap_bytes = row
        .meta
        .chrom
        .capacity()
        .checked_add(row.meta.id.capacity())
        .and_then(|bytes| bytes.checked_add(row.meta.ref_allele.capacity()))
        .and_then(|bytes| bytes.checked_add(allele_string_bytes))
        .ok_or_else(|| "phase metadata memory estimate overflow".to_owned())?;
    let phased_row_bytes = std::mem::size_of::<PhasedRow>()
        .checked_add(metadata_heap_bytes)
        .and_then(|bytes| bytes.checked_add(phase_samples.div_ceil(2)))
        .and_then(|bytes| {
            complex_capacity
                .checked_mul(std::mem::size_of::<(usize, super::types::DiploidCall)>())
                .and_then(|complex| bytes.checked_add(complex))
        })
        .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?;
    let phased_rows = phased_row_bytes
        .checked_mul(4)
        .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?;
    let updates = target_samples
        .div_ceil(2)
        .checked_add(
            possible_complex
                .checked_mul(2)
                .and_then(|count| {
                    count.checked_mul(std::mem::size_of::<(usize, super::types::DiploidCall)>())
                })
                .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?,
        )
        .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?;
    phased_rows
        .checked_add(updates)
        .ok_or_else(|| "phase-call memory estimate overflow".to_owned())
}

pub(crate) fn phase_expansion_window_bytes(
    rows: &[VariantRow],
    n_samples: usize,
) -> Result<usize, String> {
    phase_expansion_window_bytes_with_extra_samples(rows, n_samples, 0)
}

pub(crate) fn phase_expansion_window_bytes_with_extra_samples(
    rows: &[VariantRow],
    target_samples: usize,
    reference_samples: usize,
) -> Result<usize, String> {
    let phase_samples = target_samples
        .checked_add(reference_samples)
        .ok_or_else(|| "phase-call sample count overflow".to_owned())?;
    let mut bytes = phase_expansion_fixed_bytes(phase_samples)?;
    for row in rows {
        bytes = bytes
            .checked_add(phase_expansion_row_bytes_with_extra_samples(
                row,
                target_samples,
                reference_samples,
            )?)
            .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?;
    }
    Ok(bytes)
}

pub(crate) fn packed_state_major_biallelic_bytes(
    n_haplotypes: usize,
    n_markers: usize,
) -> Result<usize, String> {
    if n_haplotypes == 0 || n_markers == 0 {
        return Ok(0);
    }
    let values = n_haplotypes
        .checked_mul(n_markers)
        .ok_or_else(|| "packed haplotype dimensions overflow".to_owned())?;
    values
        .div_ceil(64)
        .checked_mul(2 * std::mem::size_of::<u64>())
        .ok_or_else(|| "packed haplotype memory estimate overflow".to_owned())
}

pub(crate) fn include_phase_expansion(
    estimate: &mut MemoryEstimate,
    bytes: usize,
) -> Result<(), String> {
    estimate.phase_snapshot_bytes = estimate
        .phase_snapshot_bytes
        .checked_add(bytes)
        .ok_or_else(|| "phase-call memory estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

pub(crate) fn include_ibd2_intervals(
    estimate: &mut MemoryEstimate,
    intervals: &Vec<Ibd2Interval>,
) -> Result<(), String> {
    // The merge pass can return fewer intervals while retaining the original
    // Vec capacity. Account for the allocation that remains live afterward.
    let bytes = intervals
        .capacity()
        .checked_mul(std::mem::size_of::<Ibd2Interval>())
        .ok_or_else(|| "IBD2 interval memory estimate overflow".to_owned())?;
    estimate.phase_snapshot_bytes = estimate
        .phase_snapshot_bytes
        .checked_add(bytes)
        .ok_or_else(|| "IBD2 interval memory estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

pub(crate) fn include_multiallelic_marginal_storage(
    estimate: &mut MemoryEstimate,
    extra_allele_marginals: usize,
) -> Result<(), String> {
    let imp_bytes_per_worker = extra_allele_marginals
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "multiallelic posterior memory estimate overflow".to_owned())?;
    // Phasing can retain posterior vectors for both haplotypes at once;
    // imputation evaluates one haplotype at a time.
    let phase_bytes_per_worker = imp_bytes_per_worker
        .checked_mul(2)
        .ok_or_else(|| "multiallelic posterior memory estimate overflow".to_owned())?;
    estimate.scratch_bytes_per_worker = estimate
        .scratch_bytes_per_worker
        .checked_add(phase_bytes_per_worker)
        .ok_or_else(|| "multiallelic posterior memory estimate overflow".to_owned())?;
    estimate.imputation_scratch_bytes_per_worker = estimate
        .imputation_scratch_bytes_per_worker
        .checked_add(imp_bytes_per_worker)
        .ok_or_else(|| "multiallelic imputation posterior memory estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

/// Account for the production streaming path used by `impute_window_stream`.
/// Sample posterior batches are bounded independently, and output rows and
/// posterior tiles are assembled from fixed-size chunks rather than retaining
/// a marker × sample matrix for the full window.
pub(crate) fn include_streamed_imputation_window(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    n_markers: usize,
    reference_haplotypes: usize,
    workers: usize,
) -> Result<(), String> {
    if target_samples == 0 || n_markers == 0 || reference_haplotypes == 0 || workers == 0 {
        return Err("streamed imputation memory dimensions must be positive".to_owned());
    }
    const POSTERIOR_TILE_BYTES: usize = 32 * 1024 * 1024;
    const DEFAULT_POSTERIOR_BATCH_BYTES: usize = 256 * 1024 * 1024;
    let posterior_batch_bytes = std::env::var("JX_IMP_POSTERIOR_BATCH_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_POSTERIOR_BATCH_BYTES / (1024 * 1024))
        .clamp(16, 512)
        .saturating_mul(1024 * 1024);

    // A store tile is copied into posterior values and assembled rows. Reserve
    // two posterior tile copies plus one output-row tile.
    let tile_bytes = POSTERIOR_TILE_BYTES
        .checked_mul(3)
        .ok_or_else(|| "streamed imputation tile estimate overflow".to_owned())?;
    // The packed reference retains one allele bit per reference haplotype and
    // marker, plus one target-only/missing flag per marker.
    let packed_reference_bytes = reference_haplotypes
        .checked_mul(n_markers)
        .and_then(|bits| bits.div_ceil(8).checked_add(n_markers))
        .ok_or_else(|| "packed reference memory estimate overflow".to_owned())?;
    // HaplotypeScratch retains Option<u16> query values, and may retain one
    // full marker vector of f32 per active worker when projection markers are
    // not dense.
    let query_scratch_per_worker = n_markers
        .checked_mul(std::mem::size_of::<Option<u16>>())
        .ok_or_else(|| "streamed imputation query scratch overflow".to_owned())?;
    let full_projection_scratch_per_worker = n_markers
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "streamed imputation projection scratch overflow".to_owned())?;
    let statistics_bytes = n_markers
        .checked_mul(2 * std::mem::size_of::<f64>())
        .ok_or_else(|| "streamed imputation statistics estimate overflow".to_owned())?;
    let metadata_bytes = n_markers
        .checked_mul(256)
        .ok_or_else(|| "streamed imputation metadata estimate overflow".to_owned())?;
    let bytes = posterior_batch_bytes
        .checked_add(tile_bytes)
        .and_then(|value| value.checked_add(packed_reference_bytes))
        .and_then(|value| value.checked_add(statistics_bytes))
        .and_then(|value| value.checked_add(metadata_bytes))
        .ok_or_else(|| "streamed imputation memory estimate overflow".to_owned())?;
    estimate.imputation_fixed_bytes = estimate
        .imputation_fixed_bytes
        .checked_add(bytes)
        .ok_or_else(|| "streamed imputation fixed estimate overflow".to_owned())?;
    estimate.imputation_scratch_bytes_per_worker = estimate
        .imputation_scratch_bytes_per_worker
        .checked_add(query_scratch_per_worker)
        .and_then(|value| value.checked_add(full_projection_scratch_per_worker))
        .ok_or_else(|| "streamed imputation worker scratch estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

/// Replace the full dense donor mosaic allowance with the compact IBS layout
/// used by reference imputation: state haplotypes at target HMM markers plus
/// donor runs bounded by the imp-step count.
pub(crate) fn set_compact_target_mosaic_storage(
    estimate: &mut MemoryEstimate,
    union_markers: usize,
    hmm_markers: usize,
    requested_states: usize,
    reference_haplotypes: usize,
    step_count: usize,
) -> Result<(), String> {
    if union_markers == 0
        || hmm_markers == 0
        || hmm_markers > union_markers
        || requested_states == 0
        || reference_haplotypes == 0
        || step_count == 0
    {
        return Err("compact imputation mosaic memory dimensions are invalid".to_owned());
    }
    let old_states = requested_states;
    let new_states = requested_states.min(reference_haplotypes);
    let old_state_bytes = union_markers
        .checked_mul(2)
        .ok_or_else(|| "old target mosaic memory estimate overflow".to_owned())?
        .div_ceil(8);
    let old_panel_bytes = old_state_bytes
        .checked_mul(old_states)
        .and_then(|value| {
            union_markers
                .checked_mul(2 * std::mem::size_of::<usize>())
                .and_then(|metadata| value.checked_add(metadata))
        })
        .and_then(|value| {
            old_states
                .checked_mul(std::mem::size_of::<usize>())
                .and_then(|metadata| value.checked_add(metadata))
        })
        .ok_or_else(|| "old target mosaic memory estimate overflow".to_owned())?;

    let new_state_bytes = hmm_markers
        .checked_mul(2)
        .ok_or_else(|| "compact target mosaic memory estimate overflow".to_owned())?
        .div_ceil(8);
    let segment_struct_bytes = std::mem::size_of::<(usize, usize, Vec<usize>)>();
    let new_panel_bytes = new_state_bytes
        .checked_mul(new_states)
        .and_then(|value| {
            hmm_markers
                .checked_mul(2 * std::mem::size_of::<usize>())
                .and_then(|metadata| value.checked_add(metadata))
        })
        .and_then(|value| {
            new_states
                .checked_mul(std::mem::size_of::<usize>())
                .and_then(|metadata| value.checked_add(metadata))
        })
        .and_then(|value| {
            step_count
                .checked_mul(
                    new_states
                        .checked_mul(std::mem::size_of::<usize>())?
                        .checked_add(segment_struct_bytes)?,
                )
                .and_then(|donors| value.checked_add(donors))
        })
        .ok_or_else(|| "compact target mosaic memory estimate overflow".to_owned())?;

    estimate.imputation_scratch_bytes_per_worker = estimate
        .imputation_scratch_bytes_per_worker
        .checked_sub(old_panel_bytes)
        .and_then(|value| value.checked_add(new_panel_bytes))
        .ok_or_else(|| "compact target mosaic scratch estimate overflow".to_owned())?;
    recompute_stage_totals(estimate)
}

pub(crate) fn set_worker_count(
    estimate: &mut MemoryEstimate,
    workers: usize,
) -> Result<(), String> {
    if workers == 0 {
        return Err("threads must be a positive integer".to_owned());
    }
    estimate.worker_count = workers;
    recompute_stage_totals(estimate)
}

pub(crate) fn derive_seed(
    master: u64,
    chrom: &str,
    window_index: u64,
    sample_index: usize,
    iteration: u32,
) -> u64 {
    // FNV-1a over a canonical byte stream, followed by SplitMix64 avalanche.
    // This is stable across Rust versions and independent of thread scheduling.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in master
        .to_le_bytes()
        .into_iter()
        .chain(chrom.as_bytes().iter().copied())
        .chain([0xff])
        .chain(window_index.to_le_bytes())
        .chain((sample_index as u64).to_le_bytes())
        .chain(iteration.to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let mut z = hash.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::{
        budgeted_imputation_checkpoint_bytes_per_worker,
        budgeted_phase_checkpoint_bytes_per_worker, estimate_window, include_bgzf_writer_workspace,
        include_imp_ibs_reference_index, include_initial_phase_pbwt_workspace,
        include_low_frequency_phase_pbwt_donors, include_phase_pbwt_donors,
        include_streamed_imputation_window, set_compact_target_mosaic_storage,
        set_imputation_selected_markers, set_phase_stage2_selected_markers, set_worker_count,
    };
    use crate::imp::types::MemoryEstimate;

    #[test]
    fn initial_pbwt_estimate_includes_both_chunk_tracks() {
        let mut estimate = MemoryEstimate::default();
        let before = estimate.total_bytes;
        let expected_track_bytes = 10 * 11 * 2 * std::mem::size_of::<u16>();

        include_initial_phase_pbwt_workspace(&mut estimate, 5, 11).unwrap();

        assert!(estimate.total_bytes - before >= expected_track_bytes);
        assert_eq!(estimate.phase_init_pbwt_stage_bytes, estimate.phase_initial_pbwt_bytes);
        assert_eq!(estimate.total_bytes, estimate.phase_init_pbwt_stage_bytes);
    }

    #[test]
    fn phase_donor_estimate_covers_sequence_encoding_scratch() {
        let mut estimate = MemoryEstimate::default();

        include_phase_pbwt_donors(&mut estimate, 4, 3).unwrap();

        let haplotypes = 8usize;
        let minimum_scan_scratch = haplotypes * 512 + 16 * 1024;
        assert!(estimate.phase_stage1_pbwt_donors_bytes >= minimum_scan_scratch);
        assert_eq!(estimate.phase_stage1_bytes, estimate.phase_stage1_pbwt_donors_bytes);
        assert_eq!(estimate.total_bytes, estimate.phase_stage1_bytes);
    }

    #[test]
    fn compact_mosaic_estimate_replaces_dense_state_matrix_per_worker() {
        let union_markers = 1_000_000usize;
        let hmm_markers = 50_000usize;
        let states = 1_600usize;
        let workers = 8usize;
        let old_panel_bytes = (union_markers * 2).div_ceil(8) * states
            + union_markers * 2 * std::mem::size_of::<usize>()
            + states * std::mem::size_of::<usize>();
        let mut estimate = MemoryEstimate {
            imputation_scratch_bytes_per_worker: old_panel_bytes + 123,
            worker_count: workers,
            imputation_stage_bytes: (old_panel_bytes + 123) * workers,
            total_bytes: (old_panel_bytes + 123) * workers,
            ..MemoryEstimate::default()
        };

        set_compact_target_mosaic_storage(
            &mut estimate,
            union_markers,
            hmm_markers,
            states,
            2_388,
            120,
        )
        .unwrap();

        assert!(estimate.imputation_scratch_bytes_per_worker < old_panel_bytes + 123);
        assert_eq!(
            estimate.imputation_stage_bytes,
            estimate.imputation_scratch_bytes_per_worker * workers
        );
        assert_eq!(estimate.total_bytes, estimate.imputation_stage_bytes);
    }

    #[test]
    fn lifecycle_stage_peaks_do_not_sum_mutually_exclusive_stages() {
        let mut estimate =
            estimate_window(674, 2_693, 196_759, 196_759, 1_000, 2_000, 1, 8).unwrap();
        assert!(estimate.phase_event_workspace_bytes_per_worker > 0);
        let phase_before_imp = estimate.phase_stage_bytes;
        let imp_before_phase_donors = estimate.imputation_stage_bytes;

        include_phase_pbwt_donors(&mut estimate, 674 + 2_693, 200).unwrap();
        assert!(estimate.phase_stage_bytes > phase_before_imp);
        assert_eq!(estimate.imputation_stage_bytes, imp_before_phase_donors);

        let phase_after_donors = estimate.phase_stage_bytes;
        include_imp_ibs_reference_index(&mut estimate, 2_693 * 2, 196_759, 200, 8).unwrap();
        include_streamed_imputation_window(&mut estimate, 674, 196_759, 2_693 * 2, 8).unwrap();
        include_bgzf_writer_workspace(&mut estimate).unwrap();

        assert_eq!(estimate.phase_stage_bytes, phase_after_donors);
        assert!(estimate.imputation_stage_bytes > imp_before_phase_donors);
        // Pure output stage does not retain the full input window.
        assert!(estimate.output_stage_bytes < estimate.input_window_bytes);
        assert_eq!(
            estimate.total_bytes,
            estimate
                .phase_stage_bytes
                .max(estimate.imputation_stage_bytes)
                .max(estimate.output_stage_bytes)
        );
        let naive_sum = estimate.phase_stage_bytes + estimate.imputation_fixed_bytes;
        assert!(estimate.total_bytes < naive_sum);

        // Selecting fewer anchors shrinks imputation checkpoint storage without affecting phasing.
        let imp_ckpt_before = estimate.imputation_checkpoint_bytes_per_worker;
        set_imputation_selected_markers(&mut estimate, 2_000, 196_759, 2_000).unwrap();
        assert!(estimate.imputation_checkpoint_bytes_per_worker <= imp_ckpt_before);
        assert_eq!(estimate.phase_stage_bytes, phase_after_donors);

        set_worker_count(&mut estimate, 4).unwrap();
        assert_eq!(estimate.worker_count, 4);
        assert_eq!(
            estimate.total_bytes,
            estimate
                .phase_stage_bytes
                .max(estimate.imputation_stage_bytes)
                .max(estimate.output_stage_bytes)
        );
    }

    #[test]
    fn budgeted_checkpoint_helpers_cap_full_cache_and_respect_disable() {
        let phase_budgeted =
            budgeted_phase_checkpoint_bytes_per_worker(196_759, 1_000, Some(32 * 1024 * 1024))
                .unwrap();
        let phase_unbudgeted =
            budgeted_phase_checkpoint_bytes_per_worker(196_759, 1_000, None).unwrap();
        assert!(phase_budgeted < phase_unbudgeted);
        assert_eq!(
            phase_budgeted,
            2 * (super::PHASE_CHECKPOINT_STRIDE
                + 196_759_usize.div_ceil(super::PHASE_CHECKPOINT_STRIDE))
                * 1_000
                * 2
                * std::mem::size_of::<f32>()
        );

        let imp_budgeted = budgeted_imputation_checkpoint_bytes_per_worker(
            70_000,
            196_759,
            2_000,
            Some(16 * 1024 * 1024),
        )
        .unwrap();
        let imp_unbudgeted =
            budgeted_imputation_checkpoint_bytes_per_worker(70_000, 196_759, 2_000, None).unwrap();
        assert!(imp_budgeted < imp_unbudgeted);
        assert!(imp_budgeted > 2 * 16 * 1024 * 1024);
    }

    #[test]
    fn worker_count_reprices_imputation_scratch_but_not_fixed_storage() {
        let mut estimate =
            estimate_window(674, 2_693, 196_759, 196_759, 1_000, 2_000, 1, 8).unwrap();
        include_imp_ibs_reference_index(&mut estimate, 2_693 * 2, 196_759, 200, 8).unwrap();
        include_streamed_imputation_window(&mut estimate, 674, 196_759, 2_693 * 2, 8).unwrap();

        let fixed = estimate.imputation_fixed_bytes;
        set_worker_count(&mut estimate, 1).unwrap();
        let one_worker_stage = estimate.imputation_stage_bytes;
        assert_eq!(estimate.imputation_fixed_bytes, fixed);

        set_worker_count(&mut estimate, 8).unwrap();
        let eight_worker_stage = estimate.imputation_stage_bytes;
        let fixed_stage = estimate.input_window_bytes
            + estimate.imputation_fixed_bytes
            + estimate.output_buffer_bytes;
        assert_eq!(estimate.imputation_fixed_bytes, fixed);
        assert!(eight_worker_stage > one_worker_stage);
        assert_eq!(
            eight_worker_stage,
            fixed_stage
                + 8 * (estimate.imputation_checkpoint_bytes_per_worker
                    + estimate.imputation_scratch_bytes_per_worker
                    + estimate.output_scratch_bytes_per_worker)
        );
    }

    #[test]
    fn phase_sub_lifecycles_take_max_instead_of_summing() {
        let mut estimate =
            estimate_window(674, 2_693, 196_759, 196_759, 1_000, 2_000, 1, 8).unwrap();
        include_initial_phase_pbwt_workspace(&mut estimate, 674, 24_595).unwrap();
        include_low_frequency_phase_pbwt_donors(&mut estimate, 674 + 2_693, 52_444).unwrap();
        set_phase_stage2_selected_markers(&mut estimate, 70_000, 196_759, 1_000).unwrap();

        assert!(estimate.phase_init_pbwt_stage_bytes > 0);
        assert!(estimate.phase_stage1_bytes > 0);
        assert!(estimate.phase_stage2_bytes > 0);
        assert_eq!(
            estimate.phase_stage_bytes,
            estimate
                .phase_init_pbwt_stage_bytes
                .max(estimate.phase_stage1_bytes)
                .max(estimate.phase_stage2_bytes)
        );
        // Initial PBWT, Stage 1 donors, and Stage 2 donors must not be summed together.
        let naive_phase_sum = estimate.input_window_bytes
            + estimate.phase_snapshot_bytes
            + estimate.phase_initial_pbwt_bytes
            + estimate.phase_stage2_pbwt_donors_bytes
            + 8 * (estimate.checkpoint_bytes_per_worker
                + estimate.phase_event_workspace_bytes_per_worker
                + estimate.scratch_bytes_per_worker);
        assert!(estimate.phase_stage_bytes < naive_phase_sum);
    }
}
