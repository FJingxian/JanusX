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

/// Reserve bounded streaming-writer memory for one BGZF batch. Each worker
/// can hold a raw block, an encoded block, and a temporary Deflate buffer.
/// The extra fixed block is the writer's active fill buffer; 64 KiB also
/// conservatively covers the file BufWriter and BGZF framing overhead.
pub(crate) fn include_bgzf_writer_workspace(estimate: &mut MemoryEstimate) -> Result<(), String> {
    const BLOCK_ALLOWANCE: usize = 65_536;
    const FIXED_BYTES: usize = BLOCK_ALLOWANCE + 8 * 1024;
    const WORKER_BYTES: usize = 3 * BLOCK_ALLOWANCE;

    let worker_bytes = WORKER_BYTES
        .checked_mul(estimate.worker_count)
        .ok_or_else(|| "BGZF writer workspace estimate overflow".to_owned())?;
    estimate.output_buffer_bytes = estimate
        .output_buffer_bytes
        .checked_add(FIXED_BYTES)
        .ok_or_else(|| "BGZF writer workspace estimate overflow".to_owned())?;
    estimate.scratch_bytes_per_worker = estimate
        .scratch_bytes_per_worker
        .checked_add(WORKER_BYTES)
        .ok_or_else(|| "BGZF writer workspace estimate overflow".to_owned())?;
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(FIXED_BYTES)
        .and_then(|value| value.checked_add(worker_bytes))
        .ok_or_else(|| "total BGZF writer workspace estimate overflow".to_owned())?;
    Ok(())
}

pub(crate) fn estimate_window(
    target_samples: usize,
    reference_samples: usize,
    n_markers: usize,
    phase_states: usize,
    imp_states: usize,
    bits_per_allele: u8,
    workers: usize,
) -> Result<MemoryEstimate, String> {
    if target_samples == 0
        || n_markers == 0
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
        checked_mul(n_markers, input_row_bytes, "input window")?,
        2,
        "input window and overlap",
    )?;
    let haplotype_bytes_per_state = n_markers
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
        n_markers
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
    let checkpoint_stride = 256usize;
    let checkpoints = n_markers.div_ceil(checkpoint_stride).saturating_add(1);
    let hmm_states = phase_states.max(imp_states);
    let checkpoint_bytes_per_worker = checked_mul(
        checked_mul(
            checked_mul(checkpoints, hmm_states, "HMM checkpoints")?,
            std::mem::size_of::<f32>(),
            "HMM checkpoints",
        )?,
        3,
        "three-track HMM checkpoints",
    )?;
    // Three forward tracks and their matching emission caches are bounded by
    // one checkpoint block per worker.
    let segment_markers = checkpoint_stride.min(n_markers);
    let per_marker_scratch =
        checked_mul(hmm_states, 6 * std::mem::size_of::<f32>(), "HMM scratch")?;
    let marker_marginal_scratch = checked_mul(
        n_markers,
        100,
        "HMM transitions, marker marginals, and phase ranking",
    )?;
    // update_sample keeps local/carry/projected backward arrays, forward
    // tracks, event and block scratch, plus the genotype alpha vector.
    let hmm_vector_scratch = checked_mul(
        checked_mul(hmm_states, 19, "HMM vector scratch")?,
        std::mem::size_of::<f32>(),
        "HMM vector scratch",
    )?;
    let phase_panel_bytes = checked_mul(
        phase_states,
        haplotype_bytes_per_state,
        "phase reference panel",
    )?
    .checked_add(
        n_markers
            .checked_mul(2 * std::mem::size_of::<usize>())
            .ok_or_else(|| "phase panel marker-index estimate overflow".to_owned())?,
    )
    .and_then(|value| {
        phase_states
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|samples| value.checked_add(samples))
    })
    .ok_or_else(|| "phase reference panel memory estimate overflow".to_owned())?;
    // Reference imputation still uses a separate target-specific mosaic panel.
    let mosaic_panel_bytes =
        checked_mul(imp_states, haplotype_bytes_per_state, "target mosaic panel")?
            .checked_add(
                n_markers
                    .checked_mul(2 * std::mem::size_of::<usize>())
                    .ok_or_else(|| "target mosaic marker-index estimate overflow".to_owned())?,
            )
            .and_then(|value| {
                imp_states
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|samples| value.checked_add(samples))
            })
            .ok_or_else(|| "target mosaic panel memory estimate overflow".to_owned())?;
    let hmm_scratch = checked_add(
        checked_mul(segment_markers, per_marker_scratch, "HMM segment scratch")?,
        checked_add(hmm_vector_scratch, marker_marginal_scratch, "HMM scratch")?,
        "HMM scratch",
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
            checked_add(
                phase_panel_bytes,
                mosaic_panel_bytes,
                "per-worker composite and imputation panels",
            )?,
            hmm_scratch,
            "per-worker panels and HMM scratch",
        )?,
        mosaic_rank_scratch,
        "per-worker panels, HMM, and mosaic scratch",
    )?;
    let output_buffer_bytes = checked_mul(target_samples, 256, "output buffer")?;
    let per_worker = checked_add(
        checkpoint_bytes_per_worker,
        scratch_bytes_per_worker,
        "per-worker HMM",
    )?;
    let worker_bytes = checked_mul(workers, per_worker, "worker pool")?;
    let total_bytes = [
        input_window_bytes,
        phase_snapshot_bytes,
        composite_panel_bytes,
        output_buffer_bytes,
        worker_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |sum, item| checked_add(sum, item, "total window"))?;
    Ok(MemoryEstimate {
        input_window_bytes,
        phase_snapshot_bytes,
        composite_panel_bytes,
        checkpoint_bytes_per_worker,
        scratch_bytes_per_worker,
        output_buffer_bytes,
        worker_count: workers,
        total_bytes,
    })
}

/// Account for the two dense, per-marker tracks retained by the initial
/// reverse/forward PBWT passes plus both live per-haplotype phasers. The
/// marker bound is computed from the same fixed logical chunks used at run
/// time, so this remains independent of the Rayon worker count.
pub(crate) fn include_initial_phase_pbwt_workspace(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    max_chunk_markers: usize,
) -> Result<(), String> {
    let haplotypes = target_samples
        .checked_mul(2)
        .ok_or_else(|| "initial PBWT haplotype count overflow".to_owned())?;
    let track_bytes = haplotypes
        .checked_mul(max_chunk_markers)
        .and_then(|cells| cells.checked_mul(2 * std::mem::size_of::<u16>()))
        .ok_or_else(|| "initial PBWT track memory estimate overflow".to_owned())?;
    // Each live PbwtRecPhaser owns order/inverse/reorder arrays, allele values,
    // and per-allele buckets. Include headroom for their Vec capacities, the
    // per-sample het/alignment vectors, and small allocator metadata.
    let phaser_bytes = haplotypes
        .checked_mul(100)
        .and_then(|bytes| {
            target_samples
                .checked_mul(64)
                .and_then(|sample_bytes| bytes.checked_add(sample_bytes))
        })
        .and_then(|bytes| bytes.checked_add(16 * 1024))
        .ok_or_else(|| "initial PBWT phaser memory estimate overflow".to_owned())?;
    let bytes = track_bytes
        .checked_add(phaser_bytes)
        .ok_or_else(|| "initial PBWT workspace memory estimate overflow".to_owned())?;
    estimate.phase_snapshot_bytes = estimate
        .phase_snapshot_bytes
        .checked_add(bytes)
        .ok_or_else(|| "initial PBWT phase memory estimate overflow".to_owned())?;
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(bytes)
        .ok_or_else(|| "total initial PBWT memory estimate overflow".to_owned())?;
    Ok(())
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
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(bytes)
        .ok_or_else(|| "total reference memory estimate overflow".to_owned())?;
    Ok(())
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
    // current step. These entries coexist before the compact vectors are
    // filled.
    let construction_entry_bytes = reference_haplotypes
        .checked_mul(workers)
        .and_then(|count| count.checked_mul(std::mem::size_of::<(u64, u32)>()))
        .ok_or_else(|| "imputation IBS construction scratch estimate overflow".to_owned())?;
    let bytes = range_bytes
        .checked_add(marker_index_bytes)
        .and_then(|value| value.checked_add(retained_step_bytes))
        .and_then(|value| value.checked_add(construction_entry_bytes))
        .ok_or_else(|| "imputation IBS index memory estimate overflow".to_owned())?;
    estimate.phase_snapshot_bytes = estimate
        .phase_snapshot_bytes
        .checked_add(bytes)
        .ok_or_else(|| "imputation IBS index memory estimate overflow".to_owned())?;
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(bytes)
        .ok_or_else(|| "total imputation IBS index memory estimate overflow".to_owned())?;
    Ok(())
}

pub(crate) fn include_phase_pbwt_donors(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    step_count: usize,
) -> Result<(), String> {
    include_phase_pbwt_donor_storage(estimate, target_samples, step_count, 1)
}

pub(crate) fn include_low_frequency_phase_pbwt_donors(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    step_count: usize,
) -> Result<(), String> {
    include_phase_pbwt_donor_storage(estimate, target_samples, step_count, 2)
}

fn include_phase_pbwt_donor_storage(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    step_count: usize,
    proposal_copies: usize,
) -> Result<(), String> {
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
    let bytes = proposal_bytes
        .checked_add(step_range_bytes)
        .and_then(|value| value.checked_add(scan_scratch_bytes))
        .ok_or_else(|| "phase PBWT donor index memory estimate overflow".to_owned())?;
    estimate.phase_snapshot_bytes = estimate
        .phase_snapshot_bytes
        .checked_add(bytes)
        .ok_or_else(|| "phase PBWT donor index memory estimate overflow".to_owned())?;
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(bytes)
        .ok_or_else(|| "total phase PBWT donor index memory estimate overflow".to_owned())?;
    Ok(())
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
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(bytes)
        .ok_or_else(|| "total variant storage estimate overflow".to_owned())?;
    Ok(())
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
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(bytes)
        .ok_or_else(|| "total phase-call memory estimate overflow".to_owned())?;
    Ok(())
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
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(bytes)
        .ok_or_else(|| "total IBD2 memory estimate overflow".to_owned())?;
    Ok(())
}

pub(crate) fn include_multiallelic_marginal_storage(
    estimate: &mut MemoryEstimate,
    extra_allele_marginals: usize,
) -> Result<(), String> {
    let bytes_per_worker = extra_allele_marginals
        .checked_mul(std::mem::size_of::<f32>())
        // Phasing can retain the posterior vectors for both haplotypes at
        // once; imputation uses one at a time, so two is the safe bound.
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| "multiallelic posterior memory estimate overflow".to_owned())?;
    let total_bytes = bytes_per_worker
        .checked_mul(estimate.worker_count)
        .ok_or_else(|| "multiallelic posterior memory estimate overflow".to_owned())?;
    estimate.scratch_bytes_per_worker = estimate
        .scratch_bytes_per_worker
        .checked_add(bytes_per_worker)
        .ok_or_else(|| "multiallelic posterior memory estimate overflow".to_owned())?;
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(total_bytes)
        .ok_or_else(|| "total multiallelic posterior estimate overflow".to_owned())?;
    Ok(())
}

/// Account for the production streaming path used by `impute_window_stream`.
/// Sample posterior batches are bounded independently, and output rows and
/// posterior tiles are assembled from fixed-size chunks rather than retaining
/// a marker × sample matrix for the full window.
pub(crate) fn include_streamed_imputation_window(
    estimate: &mut MemoryEstimate,
    target_samples: usize,
    n_markers: usize,
    workers: usize,
) -> Result<(), String> {
    if target_samples == 0 || n_markers == 0 || workers == 0 {
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
    // HaplotypeScratch can retain one full marker vector of f32 per active
    // worker when projection markers are not dense.
    let full_projection_scratch = n_markers
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| bytes.checked_mul(workers))
        .ok_or_else(|| "streamed imputation projection scratch overflow".to_owned())?;
    let statistics_bytes = n_markers
        .checked_mul(2 * std::mem::size_of::<f64>())
        .ok_or_else(|| "streamed imputation statistics estimate overflow".to_owned())?;
    let metadata_bytes = n_markers
        .checked_mul(256)
        .ok_or_else(|| "streamed imputation metadata estimate overflow".to_owned())?;
    let bytes = posterior_batch_bytes
        .checked_add(tile_bytes)
        .and_then(|value| value.checked_add(full_projection_scratch))
        .and_then(|value| value.checked_add(statistics_bytes))
        .and_then(|value| value.checked_add(metadata_bytes))
        .ok_or_else(|| "streamed imputation memory estimate overflow".to_owned())?;
    estimate.output_buffer_bytes = estimate
        .output_buffer_bytes
        .checked_add(bytes)
        .ok_or_else(|| "streamed imputation output estimate overflow".to_owned())?;
    estimate.total_bytes = estimate
        .total_bytes
        .checked_add(bytes)
        .ok_or_else(|| "total streamed imputation estimate overflow".to_owned())?;
    Ok(())
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

    estimate.scratch_bytes_per_worker = estimate
        .scratch_bytes_per_worker
        .checked_sub(old_panel_bytes)
        .and_then(|value| value.checked_add(new_panel_bytes))
        .ok_or_else(|| "compact target mosaic scratch estimate overflow".to_owned())?;
    let old_workers = old_panel_bytes
        .checked_mul(estimate.worker_count)
        .ok_or_else(|| "old worker mosaic estimate overflow".to_owned())?;
    let new_workers = new_panel_bytes
        .checked_mul(estimate.worker_count)
        .ok_or_else(|| "compact worker mosaic estimate overflow".to_owned())?;
    estimate.total_bytes = estimate
        .total_bytes
        .checked_sub(old_workers)
        .and_then(|value| value.checked_add(new_workers))
        .ok_or_else(|| "compact total mosaic estimate overflow".to_owned())?;
    Ok(())
}

pub(crate) fn set_worker_count(
    estimate: &mut MemoryEstimate,
    workers: usize,
) -> Result<(), String> {
    if workers == 0 {
        return Err("threads must be a positive integer".to_owned());
    }
    let fixed_bytes = estimate
        .input_window_bytes
        .checked_add(estimate.phase_snapshot_bytes)
        .and_then(|value| value.checked_add(estimate.composite_panel_bytes))
        .and_then(|value| value.checked_add(estimate.output_buffer_bytes))
        .ok_or_else(|| "fixed window memory estimate overflow".to_string())?;
    let per_worker = estimate
        .checkpoint_bytes_per_worker
        .checked_add(estimate.scratch_bytes_per_worker)
        .ok_or_else(|| "per-worker memory estimate overflow".to_string())?;
    let worker_bytes = per_worker
        .checked_mul(workers)
        .ok_or_else(|| "worker pool memory estimate overflow".to_string())?;
    estimate.worker_count = workers;
    estimate.total_bytes = fixed_bytes
        .checked_add(worker_bytes)
        .ok_or_else(|| "total window memory estimate overflow".to_string())?;
    Ok(())
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
        include_initial_phase_pbwt_workspace, include_phase_pbwt_donors,
        set_compact_target_mosaic_storage,
    };
    use crate::imp::types::MemoryEstimate;

    #[test]
    fn initial_pbwt_estimate_includes_both_chunk_tracks() {
        let mut estimate = MemoryEstimate::default();
        let before = estimate.total_bytes;
        let expected_track_bytes = 10 * 11 * 2 * std::mem::size_of::<u16>();

        include_initial_phase_pbwt_workspace(&mut estimate, 5, 11).unwrap();

        assert!(estimate.total_bytes - before >= expected_track_bytes);
        assert_eq!(estimate.total_bytes, estimate.phase_snapshot_bytes);
    }

    #[test]
    fn phase_donor_estimate_covers_sequence_encoding_scratch() {
        let mut estimate = MemoryEstimate::default();

        include_phase_pbwt_donors(&mut estimate, 4, 3).unwrap();

        let haplotypes = 8usize;
        let minimum_scan_scratch = haplotypes * 512 + 16 * 1024;
        assert!(estimate.phase_snapshot_bytes >= minimum_scan_scratch);
        assert_eq!(estimate.total_bytes, estimate.phase_snapshot_bytes);
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
            scratch_bytes_per_worker: old_panel_bytes + 123,
            worker_count: workers,
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

        assert!(estimate.scratch_bytes_per_worker < old_panel_bytes + 123);
        assert_eq!(
            estimate.total_bytes,
            estimate.scratch_bytes_per_worker * workers
        );
    }
}
