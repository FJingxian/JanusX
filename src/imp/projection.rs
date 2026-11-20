//! Exact reference sequence groups used only for allele projection.

use super::composite::PackedBiallelicReference;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::collections::HashMap;

const BLOCK_MARKERS: usize = 64;

struct SequenceBlock {
    donor_sequences: Vec<u32>,
    sequences: Vec<u64>,
}

pub(crate) struct ReferenceSequenceIndex {
    blocks: Vec<Option<SequenceBlock>>,
}

#[derive(Default)]
pub(crate) struct SequenceProjectionScratch {
    stamps: Vec<u32>,
    slots: Vec<usize>,
    generation: u32,
    groups: Vec<(u64, f32, f32)>,
}

impl ReferenceSequenceIndex {
    pub(crate) fn new(reference: &PackedBiallelicReference, pool: &ThreadPool) -> Self {
        let block_count = reference.n_markers().div_ceil(BLOCK_MARKERS);
        // Bound the additional index independently of reference/window size.
        // Oversized windows retain the original packed projection kernel.
        const INDEX_BUDGET_BYTES: usize = 128 * 1024 * 1024;
        if block_count
            .saturating_mul(reference.n_haplotypes())
            .saturating_mul(12)
            > INDEX_BUDGET_BYTES
        {
            return Self {
                blocks: (0..block_count).map(|_| None).collect(),
            };
        }
        let blocks = pool.install(|| {
            (0..reference.n_markers().div_ceil(BLOCK_MARKERS))
                .into_par_iter()
                .map(|block| {
                    let start = block * BLOCK_MARKERS;
                    let end = (start + BLOCK_MARKERS).min(reference.n_markers());
                    let mut alleles = vec![0u64; reference.n_haplotypes()];
                    for marker in start..end {
                        let words = reference.complete_marker_words(marker)?;
                        let mask = 1u64 << (marker - start);
                        for (word_index, &word) in words.iter().enumerate() {
                            let mut bits = word;
                            while bits != 0 {
                                let donor = word_index * 64 + bits.trailing_zeros() as usize;
                                if donor < alleles.len() {
                                    alleles[donor] |= mask;
                                }
                                bits &= bits - 1;
                            }
                        }
                    }
                    let mut ids = HashMap::new();
                    let mut sequences = Vec::new();
                    let donor_sequences = alleles
                        .into_iter()
                        .map(|sequence| {
                            *ids.entry(sequence).or_insert_with(|| {
                                let id = sequences.len() as u32;
                                sequences.push(sequence);
                                id
                            })
                        })
                        .collect();
                    Some(SequenceBlock {
                        donor_sequences,
                        sequences,
                    })
                })
                .collect()
        });
        Self { blocks }
    }

    /// Equal sequences are compared as complete 64-bit allele vectors, so
    /// grouping cannot merge a hash collision. All HMM/retention decisions
    /// precede this operation; only the order of floating sums changes.
    pub(crate) fn project_run(
        &self,
        start: usize,
        end: usize,
        donors: &[usize],
        left: &[f32],
        right: Option<&[f32]>,
        cm: &[f64],
        left_cm: f64,
        span_cm: f64,
        output: &mut [f32],
        scratch: &mut SequenceProjectionScratch,
    ) -> bool {
        if start >= end {
            return true;
        }
        let first = start / BLOCK_MARKERS;
        let last = (end - 1) / BLOCK_MARKERS;
        if self.blocks[first..=last].iter().any(Option::is_none) {
            return false;
        }
        let total_left = left.iter().copied().sum::<f32>();
        let total_right = right.map_or(total_left, |r| r.iter().copied().sum::<f32>());
        for block_id in first..=last {
            let block = self.blocks[block_id].as_ref().unwrap();
            scratch.stamps.resize(block.sequences.len(), 0);
            scratch.slots.resize(block.sequences.len(), 0);
            scratch.generation = scratch.generation.wrapping_add(1);
            if scratch.generation == 0 {
                scratch.stamps.fill(0);
                scratch.generation = 1;
            }
            scratch.groups.clear();
            for (state, &donor) in donors.iter().enumerate() {
                let id = block.donor_sequences[donor] as usize;
                if scratch.stamps[id] != scratch.generation {
                    scratch.stamps[id] = scratch.generation;
                    scratch.slots[id] = scratch.groups.len();
                    scratch.groups.push((block.sequences[id], 0.0, 0.0));
                }
                let group = &mut scratch.groups[scratch.slots[id]];
                group.1 += left[state];
                group.2 += right.map_or(left[state], |r| r[state]);
            }
            let base = block_id * BLOCK_MARKERS;
            let lo = start.max(base) - base;
            let hi = end.min(base + BLOCK_MARKERS) - base;
            let mask = (u64::MAX << lo) & (u64::MAX >> (64 - hi));
            let mut sums_left = [0.0f32; BLOCK_MARKERS];
            let mut sums_right = [0.0f32; BLOCK_MARKERS];
            for &(sequence, p0, p1) in &scratch.groups {
                let mut bits = sequence & mask;
                while bits != 0 {
                    let marker = bits.trailing_zeros() as usize;
                    sums_left[marker] += p0;
                    sums_right[marker] += p1;
                    bits &= bits - 1;
                }
            }
            for offset in lo..hi {
                let marker = base + offset;
                let alpha = if right.is_some() && span_cm > f64::EPSILON {
                    ((cm[marker] - left_cm) / span_cm).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                let weight = (1.0 - alpha) * total_left + alpha * total_right;
                let value = (1.0 - alpha) * sums_left[offset] + alpha * sums_right[offset];
                let probability = if weight > f32::MIN_POSITIVE {
                    (value / weight).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                // Reproduce the original state-order evaluation at hard-GT
                // decision boundaries, where float regrouping could flip GT.
                let decision_margin = 32.0 * f32::EPSILON * donors.len().max(1) as f32;
                output[marker] = if (probability - 0.5).abs() <= decision_margin {
                    let mut numerator = 0.0f32;
                    let mut denominator = 0.0f32;
                    for (state, &donor) in donors.iter().enumerate() {
                        let weight = if let Some(r) = right {
                            (1.0 - alpha) * left[state] + alpha * r[state]
                        } else {
                            left[state]
                        };
                        denominator += weight;
                        let sequence = block.sequences[block.donor_sequences[donor] as usize];
                        if sequence & (1 << offset) != 0 {
                            numerator += weight;
                        }
                    }
                    if denominator > f32::MIN_POSITIVE {
                        (numerator / denominator).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                } else {
                    probability
                };
            }
        }
        true
    }
}
