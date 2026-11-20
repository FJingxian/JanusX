//! IBS tracks without the Cartesian expansion of switch points and states.

use super::composite::{self, DonorHapSegment, PackedBiallelicReference};
use super::types::PackedHaplotypes;

pub(crate) struct IbsTargetMosaic {
    pub(crate) tracks: Vec<Vec<DonorHapSegment>>,
    pub(crate) n_markers: usize,
}

impl IbsTargetMosaic {
    pub(crate) fn new(
        reference: &PackedBiallelicReference,
        step_ranges: &[(usize, usize)],
        candidates: &[Vec<usize>],
        max_states: usize,
        target_haplotype: usize,
        seed: u64,
    ) -> Result<Self, String> {
        let n_markers = reference.n_markers();
        if n_markers == 0
            || reference.n_haplotypes() == 0
            || max_states == 0
            || step_ranges.last().is_none_or(|r| r.1 != n_markers)
        {
            return Err("imputation IBS track dimensions are invalid".to_owned());
        }
        let mut tracks = composite::select_ibs_donor_segments(
            step_ranges,
            candidates,
            reference.n_haplotypes(),
            max_states,
        )?;
        if tracks.is_empty() {
            tracks = composite::ibs_fallback_donor_haplotypes(
                reference.n_haplotypes(),
                max_states,
                target_haplotype,
                seed,
            )
            .into_iter()
            .map(|donor_haplotype| {
                vec![DonorHapSegment {
                    start_marker: 0,
                    end_marker: n_markers,
                    donor_haplotype,
                }]
            })
            .collect();
        }
        for track in &tracks {
            let mut end = 0;
            for segment in track {
                if segment.start_marker != end
                    || segment.end_marker <= end
                    || segment.end_marker > n_markers
                    || segment.donor_haplotype >= reference.n_haplotypes()
                {
                    return Err(
                        "imputation IBS track does not partition the reference window".to_owned(),
                    );
                }
                end = segment.end_marker;
            }
            if end != n_markers {
                return Err("imputation IBS track ends before the reference window".to_owned());
            }
        }
        Ok(Self { tracks, n_markers })
    }

    pub(crate) fn n_states(&self) -> usize {
        self.tracks.len()
    }

    pub(crate) fn donor_at(&self, marker: usize, state: usize) -> usize {
        let track = &self.tracks[state];
        track[track.partition_point(|segment| segment.end_marker <= marker)].donor_haplotype
    }

    /// Bound a run by changes in the requested states only. Inactive state
    /// switches neither change the projection nor require work or storage.
    pub(crate) fn active_donor_run_at(
        &self,
        marker: usize,
        states: &[usize],
        donors: &mut Vec<usize>,
    ) -> Result<usize, String> {
        if marker >= self.n_markers {
            return Err("projection marker exceeds IBS tracks".to_owned());
        }
        donors.clear();
        let mut end = self.n_markers;
        for &state in states {
            let track = &self.tracks[state];
            let segment = track[track.partition_point(|segment| segment.end_marker <= marker)];
            end = end.min(segment.end_marker);
            donors.push(segment.donor_haplotype);
        }
        Ok(end)
    }

    pub(crate) fn pack_marker_words(
        &self,
        reference: &PackedBiallelicReference,
        marker: usize,
        allele_words: &mut [u64],
        missing_words: &mut [u64],
    ) -> Result<(), String> {
        if marker >= self.n_markers
            || allele_words.len() != self.n_states().div_ceil(64)
            || missing_words.len() != allele_words.len()
        {
            return Err("IBS projection word dimensions are invalid".to_owned());
        }
        allele_words.fill(0);
        missing_words.fill(0);
        for state in 0..self.n_states() {
            let bit = 1u64 << (state & 63);
            match reference.allele_unchecked(marker, self.donor_at(marker, state)) {
                Some(1) => allele_words[state >> 6] |= bit,
                None => missing_words[state >> 6] |= bit,
                _ => {}
            }
        }
        Ok(())
    }

    pub(crate) fn pack_marker(
        &self,
        reference: &PackedBiallelicReference,
        marker: usize,
        output: &mut PackedHaplotypes,
    ) -> Result<(), String> {
        if marker >= self.n_markers {
            return Err("projection marker exceeds IBS tracks".to_owned());
        }
        for state in 0..self.n_states() {
            output.set_allele(
                0,
                state,
                reference.allele_unchecked(marker, self.donor_at(marker, state)),
            )?;
        }
        Ok(())
    }
}
