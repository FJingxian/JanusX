//! Whole-chromosome Positional Burrows-Wheeler Transform (PBWT) engine
//! for O(H * M) seed discovery in both pairwise IBD and multi-parent ancestry.

use super::types::BitHaplotypeMatrix;

/// Candidate maximal exact match emitted during a PBWT scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PbwtSeed {
    pub hap_a: u32,
    pub hap_b: u32,
    pub start_marker: u32,
    pub end_marker: u32,
}

/// Streaming 3-alphabet (`0`, `1`, `missing`) PBWT state requiring O(H) RAM
/// regardless of chromosome marker count. Missing alleles (`None`) are routed
/// into a dedicated third bucket with reset divergence so missing genotypes
/// can never create or extend false exact-match seeds against allele `0`.
#[derive(Debug, Clone)]
pub struct PbwtState {
    pub n_haps: usize,
    /// `prefix_order[i]` is the haplotype index at rank `i` in prefix-reversed order.
    pub prefix_order: Vec<u32>,
    /// `divergence[i]` is the start marker index of the longest suffix match
    /// between `prefix_order[i - 1]` and `prefix_order[i]`.
    pub divergence: Vec<u32>,
    a0: Vec<u32>,
    a1: Vec<u32>,
    a_miss: Vec<u32>,
    d0: Vec<u32>,
    d1: Vec<u32>,
    d_miss: Vec<u32>,
}

impl PbwtState {
    pub fn new(n_haps: usize) -> Self {
        Self {
            n_haps,
            prefix_order: (0..n_haps as u32).collect(),
            divergence: vec![0u32; n_haps],
            a0: Vec::with_capacity(n_haps),
            a1: Vec::with_capacity(n_haps),
            a_miss: Vec::new(),
            d0: Vec::with_capacity(n_haps),
            d1: Vec::with_capacity(n_haps),
            d_miss: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.prefix_order.clear();
        self.prefix_order.extend(0..self.n_haps as u32);
        self.divergence.fill(0);
    }

    /// Advance PBWT state across one marker.
    ///
    /// Observed alleles `Some(0)` and `Some(1)` follow Durbin's Algorithm 2.
    /// Missing alleles (`None`) are routed to a separate third bucket with
    /// divergence reset to `marker + 1`, preventing missing genotypes from
    /// ever matching `0`, `1`, or other missing genotypes as exact seeds.
    pub fn advance<F>(&mut self, marker: usize, allele_at: F)
    where
        F: Fn(usize) -> Option<u8>,
    {
        self.a0.clear();
        self.a1.clear();
        self.a_miss.clear();
        self.d0.clear();
        self.d1.clear();
        self.d_miss.clear();
        let next_div = (marker + 1) as u32;
        let mut p = next_div;
        let mut q = next_div;
        for idx in 0..self.n_haps {
            let hap = self.prefix_order[idx];
            let div = self.divergence[idx];
            if div > p {
                p = div;
            }
            if div > q {
                q = div;
            }
            match allele_at(hap as usize) {
                Some(0) => {
                    self.a0.push(hap);
                    self.d0.push(p);
                    p = 0;
                }
                Some(_) => {
                    self.a1.push(hap);
                    self.d1.push(q);
                    q = 0;
                }
                None => {
                    // Missing allele breaks exact suffix matching immediately.
                    self.a_miss.push(hap);
                    self.d_miss.push(next_div);
                }
            }
        }
        let n0 = self.a0.len();
        let n1 = self.a1.len();
        self.prefix_order[..n0].copy_from_slice(&self.a0);
        self.prefix_order[n0..n0 + n1].copy_from_slice(&self.a1);
        self.prefix_order[n0 + n1..].copy_from_slice(&self.a_miss);
        self.divergence[..n0].copy_from_slice(&self.d0);
        self.divergence[n0..n0 + n1].copy_from_slice(&self.d1);
        self.divergence[n0 + n1..].copy_from_slice(&self.d_miss);
    }

    /// Emit all maximal matches ending at `marker` (i.e., diverging at `marker`)
    /// whose start marker is `<= max_start_marker`.
    ///
    /// When `ref_hap_count` is `Some(k)`, only cross-group `(target, ref)` pairs
    /// where one haplotype is `< k` (reference) and the other is `>= k` (target)
    /// are emitted.
    pub fn emit_maximal_matches<F, E>(
        &self,
        marker: usize,
        max_start_marker: u32,
        ref_hap_count: Option<usize>,
        allele_at: F,
        mut on_seed: E,
    ) where
        F: Fn(usize) -> Option<u8>,
        E: FnMut(PbwtSeed),
    {
        if self.n_haps < 2 {
            return;
        }
        let mut block_start = 0usize;
        for idx in 1..=self.n_haps {
            let split = idx == self.n_haps || self.divergence[idx] > max_start_marker;
            if !split {
                continue;
            }
            if idx - block_start >= 2 {
                self.emit_block_matches(
                    block_start,
                    idx,
                    marker,
                    ref_hap_count,
                    &allele_at,
                    &mut on_seed,
                );
            }
            block_start = idx;
        }
    }

    fn emit_block_matches<F, E>(
        &self,
        block_start: usize,
        block_end: usize,
        marker: usize,
        ref_hap_count: Option<usize>,
        allele_at: &F,
        on_seed: &mut E,
    ) where
        F: Fn(usize) -> Option<u8>,
        E: FnMut(PbwtSeed),
    {
        // A pair is right-maximal at `marker` when its two alleles at `marker`
        // are not both the same observed allele (`Some(0)` or `Some(1)`).
        let mut first_obs: Option<Option<u8>> = None;
        let mut has_split = false;
        for rank in block_start..block_end {
            let a = allele_at(self.prefix_order[rank] as usize);
            if a.is_none() {
                has_split = true;
                break;
            }
            match first_obs {
                None => first_obs = Some(a),
                Some(prev) if prev != a => {
                    has_split = true;
                    break;
                }
                _ => {}
            }
        }
        if !has_split {
            return;
        }
        for i in block_start..block_end {
            let hap_i = self.prefix_order[i];
            let allele_i = allele_at(hap_i as usize);
            let mut match_start = 0u32;
            for j in (i + 1)..block_end {
                let div_j = self.divergence[j];
                if div_j > match_start {
                    match_start = div_j;
                }
                let hap_j = self.prefix_order[j];
                let allele_j = allele_at(hap_j as usize);
                // If both share the same observed allele at `marker`, the match continues.
                if allele_i.is_some() && allele_i == allele_j {
                    continue;
                }
                if let Some(k) = ref_hap_count {
                    let i_is_ref = (hap_i as usize) < k;
                    let j_is_ref = (hap_j as usize) < k;
                    if i_is_ref == j_is_ref {
                        continue;
                    }
                }
                let (a, b) = if hap_i < hap_j {
                    (hap_i, hap_j)
                } else {
                    (hap_j, hap_i)
                };
                on_seed(PbwtSeed {
                    hap_a: a,
                    hap_b: b,
                    start_marker: match_start,
                    end_marker: marker as u32,
                });
            }
        }
    }

    /// Flush all long matches surviving to the end of the chromosome (`n_markers`).
    pub fn flush_terminal_matches<E>(
        &self,
        n_markers: usize,
        max_start_marker: u32,
        ref_hap_count: Option<usize>,
        mut on_seed: E,
    ) where
        E: FnMut(PbwtSeed),
    {
        if self.n_haps < 2 || n_markers == 0 {
            return;
        }
        let mut block_start = 0usize;
        for idx in 1..=self.n_haps {
            let split = idx == self.n_haps || self.divergence[idx] > max_start_marker;
            if !split {
                continue;
            }
            if idx - block_start >= 2 {
                for i in block_start..idx {
                    let hap_i = self.prefix_order[i];
                    let mut match_start = 0u32;
                    for j in (i + 1)..idx {
                        let div_j = self.divergence[j];
                        if div_j > match_start {
                            match_start = div_j;
                        }
                        let hap_j = self.prefix_order[j];
                        if let Some(k) = ref_hap_count {
                            let i_is_ref = (hap_i as usize) < k;
                            let j_is_ref = (hap_j as usize) < k;
                            if i_is_ref == j_is_ref {
                                continue;
                            }
                        }
                        let (a, b) = if hap_i < hap_j {
                            (hap_i, hap_j)
                        } else {
                            (hap_j, hap_i)
                        };
                        on_seed(PbwtSeed {
                            hap_a: a,
                            hap_b: b,
                            start_marker: match_start,
                            end_marker: n_markers as u32,
                        });
                    }
                }
            }
            block_start = idx;
        }
    }
}

/// Run a whole-chromosome PBWT seed discovery sweep over `matrix`, emitting all
/// maximal exact seeds of genetic length `>= min_seed_cm` and marker count `>= min_markers`.
pub fn scan_chromosome_seeds(
    matrix: &BitHaplotypeMatrix,
    cm_positions: &[f64],
    min_seed_cm: f64,
    min_markers: usize,
    ref_hap_count: Option<usize>,
) -> Vec<PbwtSeed> {
    let n_markers = matrix.n_markers;
    if n_markers < min_markers.max(2) || matrix.n_haps < 2 {
        return Vec::new();
    }
    debug_assert_eq!(cm_positions.len(), n_markers);
    let mut pbwt = PbwtState::new(matrix.n_haps);
    let mut seeds = Vec::new();
    let mut cm_left = 0usize;
    for marker in 0..n_markers {
        let cur_cm = cm_positions[marker];
        while cm_left < marker && (cur_cm - cm_positions[cm_left]) >= min_seed_cm {
            cm_left += 1;
        }
        let max_start_by_cm = cm_left.checked_sub(1);
        let max_start_by_count = marker.checked_sub(min_markers);
        if let (Some(s_cm), Some(s_cnt)) = (max_start_by_cm, max_start_by_count) {
            let max_start = s_cm.min(s_cnt) as u32;
            pbwt.emit_maximal_matches(
                marker,
                max_start,
                ref_hap_count,
                |hap| matrix.allele(marker, hap),
                |seed| {
                    let len_markers = seed.end_marker.saturating_sub(seed.start_marker) as usize;
                    if len_markers >= min_markers
                        && (cm_positions[seed.end_marker.saturating_sub(1) as usize]
                            - cm_positions[seed.start_marker as usize])
                            >= min_seed_cm
                    {
                        seeds.push(seed);
                    }
                },
            );
        }
        pbwt.advance(marker, |hap| matrix.allele(marker, hap));
    }
    let last_cm = cm_positions[n_markers - 1];
    let mut terminal_left = 0usize;
    while terminal_left < n_markers && (last_cm - cm_positions[terminal_left]) >= min_seed_cm {
        terminal_left += 1;
    }
    if let (Some(s_cm), Some(s_cnt)) = (
        terminal_left.checked_sub(1),
        n_markers.checked_sub(min_markers),
    ) {
        let max_start = s_cm.min(s_cnt) as u32;
        pbwt.flush_terminal_matches(n_markers, max_start, ref_hap_count, |seed| {
            let len_markers = seed.end_marker.saturating_sub(seed.start_marker) as usize;
            if len_markers >= min_markers
                && (cm_positions[seed.end_marker.saturating_sub(1) as usize]
                    - cm_positions[seed.start_marker as usize])
                    >= min_seed_cm
            {
                seeds.push(seed);
            }
        });
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::{scan_chromosome_seeds, BitHaplotypeMatrix};

    #[test]
    fn pbwt_finds_exact_shared_segment_and_respects_ref_filter() {
        let n_markers = 200;
        let n_haps = 4; // 0,1 = ref; 2,3 = target
        let mut matrix = BitHaplotypeMatrix::new(n_haps, n_markers);
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect(); // 10 cM total
        for m in 0..n_markers {
            let a0 = ((m * 7 + 3) % 2) as u8;
            let a1 = ((m * 11 + 1) % 2) as u8;
            let a2 = if (40..140).contains(&m) { a0 } else { 1 - a0 };
            let a3 = a1;
            matrix.set_allele(m, 0, Some(a0));
            matrix.set_allele(m, 1, Some(a1));
            matrix.set_allele(m, 2, Some(a2));
            matrix.set_allele(m, 3, Some(a3));
        }

        let seeds = scan_chromosome_seeds(&matrix, &cm, 2.0, 30, Some(2));
        assert!(seeds
            .iter()
            .any(|s| s.hap_a == 0 && s.hap_b == 2 && s.start_marker <= 40 && s.end_marker >= 140));
        assert!(seeds
            .iter()
            .any(|s| s.hap_a == 1 && s.hap_b == 3 && s.start_marker == 0 && s.end_marker == 200));
        assert!(seeds.iter().all(|s| (s.hap_a < 2) != (s.hap_b < 2)));
    }

    #[test]
    fn pbwt_never_emits_seeds_for_all_missing_haplotype_against_all_zero() {
        let n_markers = 200;
        let mut matrix = BitHaplotypeMatrix::new(2, n_markers);
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            matrix.set_allele(m, 0, Some(0));
            matrix.set_allele(m, 1, None);
        }
        let seeds = scan_chromosome_seeds(&matrix, &cm, 1.0, 10, None);
        assert!(seeds.is_empty(), "all-missing sample must not match all-0");
    }
}
