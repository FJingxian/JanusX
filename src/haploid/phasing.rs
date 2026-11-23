//! 1-pass lightweight whole-chromosome PBWT phasing for unphased heterozygous
//! diploid individuals. Inbred/DH lines and already-phased samples bypass this
//! pass completely with zero overhead.

use super::pbwt::PbwtState;
use super::types::BitHaplotypeMatrix;

/// Sample-level ploidy classification and haplotype mapping within a chromosome
/// `BitHaplotypeMatrix`.
#[derive(Debug, Clone)]
pub struct SampleHaplotypeLayout {
    pub n_samples: usize,
    /// `true` if sample `s` is treated as inbred/DH (1 pseudo-haplotype).
    pub is_inbred: Vec<bool>,
    /// Starting haplotype column in `BitHaplotypeMatrix` for sample `s`.
    pub hap_offset: Vec<usize>,
    /// Number of haplotypes (`1` for inbred, `2` for diploid) for sample `s`.
    pub hap_count: Vec<u8>,
    /// Reverse lookup: for each haplotype column `h`, `(sample_idx, hap_idx_within_sample)`.
    pub hap_to_sample: Vec<(u32, u8)>,
    pub total_haps: usize,
}

impl SampleHaplotypeLayout {
    pub fn from_inbred_flags(is_inbred: Vec<bool>) -> Self {
        let n_samples = is_inbred.len();
        let mut hap_offset = Vec::with_capacity(n_samples);
        let mut hap_count = Vec::with_capacity(n_samples);
        let mut hap_to_sample = Vec::new();
        let mut offset = 0usize;
        for (s, &inbred) in is_inbred.iter().enumerate() {
            hap_offset.push(offset);
            let count = if inbred { 1u8 } else { 2u8 };
            hap_count.push(count);
            for h in 0..count {
                hap_to_sample.push((s as u32, h));
            }
            offset += count as usize;
        }
        Self {
            n_samples,
            is_inbred,
            hap_offset,
            hap_count,
            hap_to_sample,
            total_haps: offset,
        }
    }
}

/// A disjoint VCF `PS` phase block on a chromosome for a diploid sample.
///
/// `anchor_marker` is always the FIRST phased marker in the block
/// (`all_phased_markers[0]`) so block orientation is decided BEFORE PBWT
/// advances past any marker in the block (guaranteeing PBWT prefix/divergence
/// state never becomes stale relative to flipped markers). `first_het_marker`
/// records the first complete heterozygote in the block to break ties when
/// `anchor_marker` is a partial-missing phased call (e.g. `0|.`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisjointPhaseBlock {
    pub anchor_marker: u32,
    pub first_het_marker: Option<u32>,
    pub all_phased_markers: Vec<u32>,
}

/// 1-pass whole-chromosome PBWT phaser for unphased diploid heterozygotes and
/// disjoint VCF `PS` phase blocks.
///
/// - For each single unphased heterozygote (`unphased_het_positions[sample]`),
///   we orient `(0, 1)` vs `(1, 0)` using the longest-matching PBWT neighbors.
/// - For each disjoint VCF `PS` block (`disjoint_ps_blocks[sample]`), all
///   phased calls inside the block (including partial-missing calls like `0|.`)
///   already have known relative phase; at `block.anchor_marker` (the earliest
///   phased marker in the block, strictly before PBWT processes any marker in
///   the block) we decide once whether to keep or flip all
///   `block.all_phased_markers` atomically.
///
/// Complexity: O(H * M) time, O(H) extra memory — runs in seconds per chromosome
/// and feeds directly into error-tolerant `hap-IBD`.
pub fn phase_unphased_diploids_one_pass(
    matrix: &mut BitHaplotypeMatrix,
    layout: &SampleHaplotypeLayout,
    unphased_het_positions: &[Vec<u32>],
    disjoint_ps_blocks: &[Vec<DisjointPhaseBlock>],
) {
    let n_markers = matrix.n_markers;
    let n_haps = matrix.n_haps;
    if n_markers < 2 || n_haps < 4 {
        return;
    }
    let any_unphased = unphased_het_positions.iter().any(|v| !v.is_empty());
    let any_ps_blocks = disjoint_ps_blocks
        .iter()
        .any(|blocks| blocks.iter().any(|b| !b.all_phased_markers.is_empty()));
    if !any_unphased && !any_ps_blocks {
        return;
    }

    // Track per-sample cursor into sorted `unphased_het_positions[sample]` and
    // `disjoint_ps_blocks[sample]` (sorted by `anchor_marker`).
    let mut het_cursors = vec![0usize; layout.n_samples];
    let mut ps_cursors = vec![0usize; layout.n_samples];
    let mut rank_of_hap = vec![0usize; n_haps];
    let mut pbwt = PbwtState::new(n_haps);
    let mut active_unphased_samples = Vec::new();
    let mut active_ps_blocks: Vec<(usize, usize)> = Vec::new(); // (sample, block_idx)

    for marker in 0..n_markers {
        let marker_u32 = marker as u32;
        active_unphased_samples.clear();
        active_ps_blocks.clear();
        for s in 0..layout.n_samples {
            if layout.is_inbred[s] {
                continue;
            }
            if let Some(list) = unphased_het_positions.get(s) {
                let c = het_cursors[s];
                if c < list.len() && list[c] == marker_u32 {
                    active_unphased_samples.push(s);
                    het_cursors[s] = c + 1;
                }
            }
            if let Some(blocks) = disjoint_ps_blocks.get(s) {
                while ps_cursors[s] < blocks.len()
                    && blocks[ps_cursors[s]].all_phased_markers.is_empty()
                {
                    ps_cursors[s] += 1;
                }
                while ps_cursors[s] < blocks.len()
                    && blocks[ps_cursors[s]].anchor_marker == marker_u32
                {
                    active_ps_blocks.push((s, ps_cursors[s]));
                    ps_cursors[s] += 1;
                }
            }
        }

        if (!active_unphased_samples.is_empty() || !active_ps_blocks.is_empty()) && marker > 0 {
            for (rank, &hap) in pbwt.prefix_order.iter().enumerate() {
                rank_of_hap[hap as usize] = rank;
            }

            // 1) Orient individual unphased heterozygotes.
            let mut single_decisions: Vec<(usize, u8, u8)> =
                Vec::with_capacity(active_unphased_samples.len());
            for &s in &active_unphased_samples {
                let h0 = layout.hap_offset[s];
                let h1 = h0 + 1;
                let (w0_zero, w0_one) = neighbor_allele_support_at(
                    &pbwt,
                    &rank_of_hap,
                    matrix,
                    marker,
                    marker,
                    h0,
                    s,
                    layout,
                );
                let (w1_zero, w1_one) = neighbor_allele_support_at(
                    &pbwt,
                    &rank_of_hap,
                    matrix,
                    marker,
                    marker,
                    h1,
                    s,
                    layout,
                );
                let score_01 = w0_zero + w1_one;
                let score_10 = w0_one + w1_zero;
                if score_10 > score_01 {
                    single_decisions.push((h0, 1, 0));
                } else {
                    single_decisions.push((h0, 0, 1));
                }
            }
            for (h0, a0, a1) in single_decisions {
                matrix.set_allele(marker, h0, Some(a0));
                matrix.set_allele(marker, h0 + 1, Some(a1));
            }

            // 2) Orient each disjoint PS block atomically at its earliest phased
            //    marker (before PBWT advances past any marker in the block).
            let mut blocks_to_flip: Vec<(usize, usize)> = Vec::new();
            for &(s, bc) in &active_ps_blocks {
                let block = &disjoint_ps_blocks[s][bc];
                let h0 = layout.hap_offset[s];
                let h1 = h0 + 1;
                let (mut score_keep, mut score_flip) = score_block_orientation_at(
                    &pbwt,
                    &rank_of_hap,
                    matrix,
                    marker,
                    marker,
                    h0,
                    h1,
                    s,
                    layout,
                );
                if let Some(het_m_u32) = block.first_het_marker {
                    let het_m = het_m_u32 as usize;
                    if het_m != marker && het_m < n_markers {
                        let (hk, hf) = score_block_orientation_at(
                            &pbwt,
                            &rank_of_hap,
                            matrix,
                            marker,
                            het_m,
                            h0,
                            h1,
                            s,
                            layout,
                        );
                        score_keep += hk;
                        score_flip += hf;
                    }
                }
                if score_flip > score_keep {
                    blocks_to_flip.push((s, bc));
                }
            }
            for (s, bc) in blocks_to_flip {
                let h0 = layout.hap_offset[s];
                let h1 = h0 + 1;
                for &m_u32 in &disjoint_ps_blocks[s][bc].all_phased_markers {
                    let m = m_u32 as usize;
                    let a0 = matrix.allele(m, h0);
                    let a1 = matrix.allele(m, h1);
                    matrix.set_allele(m, h0, a1);
                    matrix.set_allele(m, h1, a0);
                }
            }
        }

        pbwt.advance(marker, |hap| matrix.allele(marker, hap));
    }
}

#[allow(clippy::too_many_arguments)]
fn score_block_orientation_at(
    pbwt: &PbwtState,
    rank_of_hap: &[usize],
    matrix: &BitHaplotypeMatrix,
    pbwt_marker: usize,
    query_marker: usize,
    h0: usize,
    h1: usize,
    owner_sample: usize,
    layout: &SampleHaplotypeLayout,
) -> (f64, f64) {
    let cur_a0 = matrix.allele(query_marker, h0);
    let cur_a1 = matrix.allele(query_marker, h1);
    if cur_a0 == cur_a1 {
        return (0.0, 0.0);
    }
    let (w0_zero, w0_one) = neighbor_allele_support_at(
        pbwt,
        rank_of_hap,
        matrix,
        pbwt_marker,
        query_marker,
        h0,
        owner_sample,
        layout,
    );
    let (w1_zero, w1_one) = neighbor_allele_support_at(
        pbwt,
        rank_of_hap,
        matrix,
        pbwt_marker,
        query_marker,
        h1,
        owner_sample,
        layout,
    );
    match (cur_a0, cur_a1) {
        (Some(0), Some(1)) => (w0_zero + w1_one, w0_one + w1_zero),
        (Some(1), Some(0)) => (w0_one + w1_zero, w0_zero + w1_one),
        (Some(0), None) => (w0_zero, w1_zero),
        (Some(1), None) => (w0_one, w1_one),
        (None, Some(0)) => (w1_zero, w0_zero),
        (None, Some(1)) => (w1_one, w0_one),
        _ => (0.0, 0.0),
    }
}

#[allow(clippy::too_many_arguments)]
fn neighbor_allele_support_at(
    pbwt: &PbwtState,
    rank_of_hap: &[usize],
    matrix: &BitHaplotypeMatrix,
    pbwt_marker: usize,
    query_marker: usize,
    hap: usize,
    owner_sample: usize,
    layout: &SampleHaplotypeLayout,
) -> (f64, f64) {
    const WINDOW: usize = 4;
    let rank = rank_of_hap[hap];
    let mut w_zero = 0.0f64;
    let mut w_one = 0.0f64;

    // Scan upward from `rank - 1`
    let mut min_div = 0u32;
    let mut steps = 0usize;
    let mut r = rank;
    while r > 0 && steps < WINDOW {
        let div = pbwt.divergence[r];
        if div > min_div {
            min_div = div;
        }
        r -= 1;
        steps += 1;
        let nbr = pbwt.prefix_order[r] as usize;
        if layout.hap_to_sample[nbr].0 as usize == owner_sample {
            continue;
        }
        if let Some(allele) = matrix.allele(query_marker, nbr) {
            let match_len = pbwt_marker.saturating_sub(min_div as usize) as f64 + 1.0;
            if allele == 0 {
                w_zero += match_len;
            } else {
                w_one += match_len;
            }
        }
    }

    // Scan downward from `rank + 1`
    min_div = 0;
    steps = 0;
    let mut r = rank + 1;
    while r < pbwt.n_haps && steps < WINDOW {
        let div = pbwt.divergence[r];
        if div > min_div {
            min_div = div;
        }
        let nbr = pbwt.prefix_order[r] as usize;
        r += 1;
        steps += 1;
        if layout.hap_to_sample[nbr].0 as usize == owner_sample {
            continue;
        }
        if let Some(allele) = matrix.allele(query_marker, nbr) {
            let match_len = pbwt_marker.saturating_sub(min_div as usize) as f64 + 1.0;
            if allele == 0 {
                w_zero += match_len;
            } else {
                w_one += match_len;
            }
        }
    }

    (w_zero, w_one)
}

#[cfg(test)]
mod tests {
    use super::{phase_unphased_diploids_one_pass, DisjointPhaseBlock, SampleHaplotypeLayout};
    use crate::haploid::types::BitHaplotypeMatrix;

    #[test]
    fn one_pass_pbwt_phasing_recovers_template_phase_for_unphased_diploid() {
        // 3 phased/inbred template lines (samples 0, 1, 2) + 1 unphased diploid F1
        // (sample 3) whose true haplotypes are template 0 and template 1.
        let n_markers = 120;
        let layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true, true, false]);
        let mut matrix = BitHaplotypeMatrix::new(layout.total_haps, n_markers);
        let mut unphased_hets = vec![Vec::new(); layout.n_samples];
        let ps_blocks = vec![Vec::new(); layout.n_samples];

        let mut true_h0 = Vec::with_capacity(n_markers);
        let mut true_h1 = Vec::with_capacity(n_markers);
        for m in 0..n_markers {
            let t0 = ((m * 5 + 1) % 2) as u8;
            let t1 = ((m * 7 + 2) % 3 == 0) as u8;
            let t2 = ((m * 11 + 3) % 2) as u8;
            matrix.set_allele(m, 0, Some(t0));
            matrix.set_allele(m, 1, Some(t1));
            matrix.set_allele(m, 2, Some(t2));
            true_h0.push(t0);
            true_h1.push(t1);

            if t0 == t1 {
                matrix.set_allele(m, 3, Some(t0));
                matrix.set_allele(m, 4, Some(t1));
            } else {
                // Intentionally scramble unphased het orientation as (0, 1) everywhere.
                matrix.set_allele(m, 3, Some(0));
                matrix.set_allele(m, 4, Some(1));
                unphased_hets[3].push(m as u32);
            }
        }

        phase_unphased_diploids_one_pass(&mut matrix, &layout, &unphased_hets, &ps_blocks);

        // After burn-in of a few markers, phased haplotypes should match (true_h0, true_h1)
        // or its global swap (true_h1, true_h0) with very high concordance.
        let mut match_direct = 0usize;
        let mut match_swapped = 0usize;
        for m in 10..n_markers {
            let a0 = matrix.allele(m, 3).unwrap();
            let a1 = matrix.allele(m, 4).unwrap();
            if a0 == true_h0[m] && a1 == true_h1[m] {
                match_direct += 1;
            }
            if a0 == true_h1[m] && a1 == true_h0[m] {
                match_swapped += 1;
            }
        }
        let best = match_direct.max(match_swapped);
        assert!(
            best >= 100,
            "expected >= 100/110 phased markers to match template, got {best}"
        );
    }

    #[test]
    fn disjoint_ps_block_flips_atomically_including_partial_missing_phased_calls() {
        // 2 inbred templates + 1 diploid sample whose true haplotypes are (t0, t1).
        // First half [0..40) is in PS=1 (correctly oriented).
        // Second half [40..80) is in PS=2 (globally flipped relative to PS=1),
        // containing two complete hets (m=40, m=50) AND a partial-missing phased
        // call `0|.` at m=45. Flipping PS=2 must flip m=40, m=45 (`0|.` -> `.|0`),
        // and m=50 together atomically.
        let n_markers = 80;
        let layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true, false]);
        let mut matrix = BitHaplotypeMatrix::new(layout.total_haps, n_markers);
        let unphased_hets = vec![Vec::new(); layout.n_samples];
        let mut ps_blocks = vec![Vec::new(); layout.n_samples];

        for m in 0..n_markers {
            let t0 = ((m * 3 + 1) % 2) as u8;
            let t1 = ((m * 5 + 2) % 3 == 0) as u8;
            matrix.set_allele(m, 0, Some(t0));
            matrix.set_allele(m, 1, Some(t1));
            if m < 40 {
                matrix.set_allele(m, 2, Some(t0));
                matrix.set_allele(m, 3, Some(t1));
            } else if m == 40 || m == 50 {
                if m == 40 {
                    matrix.set_allele(m, 0, Some(0));
                    matrix.set_allele(m, 1, Some(1));
                } else {
                    matrix.set_allele(m, 0, Some(1));
                    matrix.set_allele(m, 1, Some(0));
                }
                matrix.set_allele(m, 2, Some(1));
                matrix.set_allele(m, 3, Some(0));
            } else if m == 45 {
                // Partial-missing phased call `0|.` inside PS=2
                matrix.set_allele(m, 2, Some(0));
                matrix.set_allele(m, 3, None);
            } else {
                matrix.set_allele(m, 2, Some(t0));
                matrix.set_allele(m, 3, Some(t0));
                matrix.set_allele(m, 1, Some(t0));
            }
        }
        ps_blocks[2].push(DisjointPhaseBlock {
            anchor_marker: 40,
            first_het_marker: Some(40),
            all_phased_markers: vec![40, 45, 50],
        });

        phase_unphased_diploids_one_pass(&mut matrix, &layout, &unphased_hets, &ps_blocks);

        // Complete hets m=40 and m=50 flipped together from (1, 0) to (0, 1).
        assert_eq!(
            (matrix.allele(40, 2), matrix.allele(40, 3)),
            (Some(0), Some(1))
        );
        assert_eq!(
            (matrix.allele(50, 2), matrix.allele(50, 3)),
            (Some(0), Some(1))
        );
        // Partial-missing phased call m=45 flipped together from `0|.` to `.|0`.
        assert_eq!(
            (matrix.allele(45, 2), matrix.allele(45, 3)),
            (None, Some(0))
        );
    }

    #[test]
    fn ps_block_flips_before_pbwt_advances_past_leading_partial_missing() {
        // 4-marker test where PS=2 on sample 2 has a partial-missing phased call `0|.`
        // at marker 1 BEFORE its first complete het at marker 2.
        // Because `anchor_marker = 1` (earliest phased marker in PS=2) and
        // `first_het_marker = Some(2)`, PS=2 flips at marker 1 BEFORE `pbwt.advance(1)`.
        // Verify both marker 1 (`0|.` -> `.|0`) and marker 2 (`1|0` -> `0|1`) flip
        // atomically and leave the matrix consistent with the PBWT prefix at marker 3.
        let n_markers = 4;
        let layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true, false, false]);
        let mut matrix = BitHaplotypeMatrix::new(layout.total_haps, n_markers);
        let mut unphased_hets = vec![Vec::new(); layout.n_samples];
        let mut ps_blocks = vec![Vec::new(); layout.n_samples];

        // Marker 0: templates (0, 1), sample 2 (0|1), sample 3 (0|1)
        matrix.set_allele(0, 0, Some(0));
        matrix.set_allele(0, 1, Some(1));
        matrix.set_allele(0, 2, Some(0));
        matrix.set_allele(0, 3, Some(1));
        matrix.set_allele(0, 4, Some(0));
        matrix.set_allele(0, 5, Some(1));

        // Marker 1: templates (1, 0); sample 2 has partial-missing `0|.` in PS=2
        // (flipped relative to templates -> should become `.|0`); sample 3 has (1|0).
        matrix.set_allele(1, 0, Some(1));
        matrix.set_allele(1, 1, Some(0));
        matrix.set_allele(1, 2, Some(0));
        matrix.set_allele(1, 3, None);
        matrix.set_allele(1, 4, Some(1));
        matrix.set_allele(1, 5, Some(0));

        // Marker 2: templates (0, 1); sample 2 has complete het `1|0` in PS=2
        // (also flipped relative to templates -> should become `0|1`); sample 3 has (0|1).
        matrix.set_allele(2, 0, Some(0));
        matrix.set_allele(2, 1, Some(1));
        matrix.set_allele(2, 2, Some(1));
        matrix.set_allele(2, 3, Some(0));
        matrix.set_allele(2, 4, Some(0));
        matrix.set_allele(2, 5, Some(1));

        // Marker 3: templates (1, 0), sample 2 (1|0), sample 3 unphased het `0/1`
        matrix.set_allele(3, 0, Some(1));
        matrix.set_allele(3, 1, Some(0));
        matrix.set_allele(3, 2, Some(1));
        matrix.set_allele(3, 3, Some(0));
        matrix.set_allele(3, 4, Some(0));
        matrix.set_allele(3, 5, Some(1));
        unphased_hets[3].push(3);

        ps_blocks[2].push(DisjointPhaseBlock {
            anchor_marker: 1,
            first_het_marker: Some(2),
            all_phased_markers: vec![1, 2],
        });

        phase_unphased_diploids_one_pass(&mut matrix, &layout, &unphased_hets, &ps_blocks);

        // Sample 2's PS=2 flipped atomically at marker 1 before PBWT advanced past marker 1.
        assert_eq!(
            (matrix.allele(1, 2), matrix.allele(1, 3)),
            (None, Some(0))
        );
        assert_eq!(
            (matrix.allele(2, 2), matrix.allele(2, 3)),
            (Some(0), Some(1))
        );
        // Sample 3's unphased het at marker 3 phases cleanly to (1, 0) matching templates + sample 2.
        assert_eq!(
            (matrix.allele(3, 4), matrix.allele(3, 5)),
            (Some(1), Some(0))
        );
    }
}
