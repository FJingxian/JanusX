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
    scan_chromosome_seeds_cancellable(
        matrix,
        cm_positions,
        min_seed_cm,
        min_markers,
        ref_hap_count,
        None,
    )
    .unwrap_or_default()
}

/// Cancellable whole-chromosome PBWT seed discovery sweep over `matrix`.
///
/// Automatically partitions long chromosomes into overlapping chunks across the
/// active Rayon thread pool: each chunk warms up PBWT strictly before
/// `cm_positions[own_start - 1] - min_seed_cm` and `own_start - min_markers`,
/// initializes chunk divergence to `warmup_start`, and recovers the exact true
/// left start down to marker `0` whenever a match reaches `warmup_start`.
/// This guarantees thread-count-invariant seeds (identical to single-threaded
/// PBWT) while saturating all CPU cores.
pub fn scan_chromosome_seeds_cancellable(
    matrix: &BitHaplotypeMatrix,
    cm_positions: &[f64],
    min_seed_cm: f64,
    min_markers: usize,
    ref_hap_count: Option<usize>,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Vec<PbwtSeed>, String> {
    use rayon::prelude::*;

    if is_cancelled(cancelled) {
        return Err("Interrupted".to_string());
    }
    let n_markers = matrix.n_markers;
    if n_markers < min_markers.max(2) || matrix.n_haps < 2 {
        return Ok(Vec::new());
    }
    debug_assert_eq!(cm_positions.len(), n_markers);

    let threads = rayon::current_num_threads().max(1);
    let total_cm = (cm_positions[n_markers - 1] - cm_positions[0]).max(0.0);
    // Use chunked parallel PBWT when there is enough work to amortize warm-up overlap.
    let n_chunks = if threads <= 1
        || n_markers < min_markers.saturating_mul(8).max(4096)
        || total_cm < min_seed_cm * 3.0
    {
        1
    } else {
        let max_by_cm = ((total_cm / (min_seed_cm * 2.0)).floor() as usize).max(1);
        threads.min(max_by_cm)
    };

    if n_chunks <= 1 {
        return scan_chunk_seeds(
            matrix,
            cm_positions,
            0,
            0,
            n_markers,
            true,
            min_seed_cm,
            min_markers,
            ref_hap_count,
            cancelled,
        );
    }

    let chunk_size = n_markers.div_ceil(n_chunks);
    let chunks: Vec<(usize, usize, usize, bool)> = (0..n_chunks)
        .filter_map(|c| {
            let own_start = c * chunk_size;
            if own_start >= n_markers {
                return None;
            }
            let own_end = ((c + 1) * chunk_size).min(n_markers);
            let is_last = own_end == n_markers;
            let warmup_start = if own_start == 0 {
                0
            } else {
                // A seed ending at `marker = own_start` has last matched marker `own_start - 1`.
                // Warm up far enough left that its genetic length `cm[own_start - 1] - cm[warmup_start]`
                // is already `>= min_seed_cm` (or `warmup_start == 0`) AND marker count `>= min_markers`.
                let last_matched_cm = cm_positions[own_start - 1];
                let target_cm = last_matched_cm - min_seed_cm;
                let mut w_cm = own_start - 1;
                while w_cm > 0 && cm_positions[w_cm] > target_cm {
                    w_cm -= 1;
                }
                let w_cnt = own_start.saturating_sub(min_markers);
                w_cm.min(w_cnt)
            };
            Some((warmup_start, own_start, own_end, is_last))
        })
        .collect();

    let chunk_results: Result<Vec<Vec<PbwtSeed>>, String> = chunks
        .into_par_iter()
        .map(|(warmup_start, own_start, own_end, is_last)| {
            scan_chunk_seeds(
                matrix,
                cm_positions,
                warmup_start,
                own_start,
                own_end,
                is_last,
                min_seed_cm,
                min_markers,
                ref_hap_count,
                cancelled,
            )
        })
        .collect();

    let parts = chunk_results?;
    let total_len: usize = parts.iter().map(Vec::len).sum();
    let mut all = Vec::with_capacity(total_len);
    for part in parts {
        all.extend(part);
    }
    Ok(all)
}

#[inline]
fn is_cancelled(cancelled: Option<&std::sync::atomic::AtomicBool>) -> bool {
    cancelled
        .map(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(false)
}

#[inline]
fn recover_exact_left_start(
    matrix: &BitHaplotypeMatrix,
    hap_a: usize,
    hap_b: usize,
    mut start_m: usize,
) -> usize {
    while start_m > 0 {
        let prev = start_m - 1;
        match (matrix.allele(prev, hap_a), matrix.allele(prev, hap_b)) {
            (Some(a), Some(b)) if a == b => start_m = prev,
            _ => break,
        }
    }
    start_m
}

/// O(H_active * warmup_start) reverse partition-refinement + O(1) RMQ index
/// for recovering exact left start markers of arbitrarily many seeds touching
/// `warmup_start` without per-pair prefix rescanning.
struct LeftBoundaryIndex {
    rank_of: Vec<u32>,
    rmq: Vec<Vec<u32>>,
}

impl LeftBoundaryIndex {
    fn build(
        matrix: &BitHaplotypeMatrix,
        warmup_start: usize,
        boundary_seeds: &[PbwtSeed],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Self, String> {
        let mut seen = vec![false; matrix.n_haps];
        let mut order: Vec<u32> = Vec::new();
        for seed in boundary_seeds {
            let a = seed.hap_a as usize;
            let b = seed.hap_b as usize;
            if !seen[a] {
                seen[a] = true;
                order.push(seed.hap_a);
            }
            if !seen[b] {
                seen[b] = true;
                order.push(seed.hap_b);
            }
        }
        let k = order.len();
        let mut div = vec![0u32; k];
        if k >= 2 && warmup_start > 0 {
            let mut blocks: Vec<(usize, usize)> = vec![(0, k)];
            let mut next_blocks: Vec<(usize, usize)> = Vec::new();
            let mut b0: Vec<u32> = Vec::with_capacity(k);
            let mut b1: Vec<u32> = Vec::with_capacity(k);
            let mut b_miss: Vec<u32> = Vec::new();

            for m in (0..warmup_start).rev() {
                if (m & 255) == 0 && is_cancelled(cancelled) {
                    return Err("Interrupted".to_string());
                }
                if blocks.is_empty() {
                    break;
                }
                next_blocks.clear();
                let div_val = (m + 1) as u32;
                for &(b_start, b_end) in &blocks {
                    b0.clear();
                    b1.clear();
                    b_miss.clear();
                    for &hap in &order[b_start..b_end] {
                        match matrix.allele(m, hap as usize) {
                            Some(0) => b0.push(hap),
                            Some(_) => b1.push(hap),
                            None => b_miss.push(hap),
                        }
                    }
                    let n0 = b0.len();
                    let n1 = b1.len();
                    let p1 = b_start + n0;
                    let pm = p1 + n1;
                    order[b_start..p1].copy_from_slice(&b0);
                    order[p1..pm].copy_from_slice(&b1);
                    order[pm..b_end].copy_from_slice(&b_miss);

                    if n0 > 0 && n1 > 0 {
                        div[p1] = div_val;
                    }
                    // Every missing allele breaks exact matching against both observed
                    // buckets and every other missing haplotype at marker `m`.
                    for idx in pm..b_end {
                        if idx > b_start {
                            div[idx] = div_val;
                        }
                    }
                    if n0 >= 2 {
                        next_blocks.push((b_start, p1));
                    }
                    if n1 >= 2 {
                        next_blocks.push((p1, pm));
                    }
                }
                std::mem::swap(&mut blocks, &mut next_blocks);
            }
        }

        let mut rank_of = vec![u32::MAX; matrix.n_haps];
        for (rank, &hap) in order.iter().enumerate() {
            rank_of[hap as usize] = rank as u32;
        }

        let mut rmq: Vec<Vec<u32>> = Vec::new();
        if k > 0 {
            let levels = (usize::BITS - k.leading_zeros()) as usize;
            rmq.reserve(levels);
            rmq.push(div);
            for lvl in 1..levels {
                let half = 1usize << (lvl - 1);
                let span = 1usize << lvl;
                let prev = &rmq[lvl - 1];
                let cur_len = k + 1 - span;
                let mut cur = Vec::with_capacity(cur_len);
                for i in 0..cur_len {
                    cur.push(prev[i].max(prev[i + half]));
                }
                rmq.push(cur);
            }
        }

        Ok(Self { rank_of, rmq })
    }

    #[inline]
    fn exact_start(&self, hap_a: usize, hap_b: usize) -> usize {
        let ra = self.rank_of[hap_a] as usize;
        let rb = self.rank_of[hap_b] as usize;
        let (lo, hi) = if ra < rb { (ra + 1, rb) } else { (rb + 1, ra) };
        let len = hi - lo + 1;
        let lvl = (usize::BITS - 1 - len.leading_zeros()) as usize;
        self.rmq[lvl][lo].max(self.rmq[lvl][hi + 1 - (1usize << lvl)]) as usize
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_chunk_seeds(
    matrix: &BitHaplotypeMatrix,
    cm_positions: &[f64],
    warmup_start: usize,
    own_start: usize,
    own_end: usize,
    is_last: bool,
    min_seed_cm: f64,
    min_markers: usize,
    ref_hap_count: Option<usize>,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Vec<PbwtSeed>, String> {
    let mut pbwt = PbwtState::new(matrix.n_haps);
    if warmup_start > 0 {
        pbwt.divergence.fill(warmup_start as u32);
    }
    let mut seeds = Vec::new();
    let mut boundary_seeds = Vec::new();
    let mut cm_left = warmup_start;

    for marker in warmup_start..own_end {
        if (marker & 255) == 0 && is_cancelled(cancelled) {
            return Err("Interrupted".to_string());
        }
        if marker >= own_start {
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
                        let start_m = (seed.start_marker as usize).max(warmup_start);
                        if warmup_start > 0 && start_m == warmup_start {
                            boundary_seeds.push(PbwtSeed {
                                start_marker: start_m as u32,
                                ..seed
                            });
                            return;
                        }
                        let end_m = seed.end_marker as usize;
                        let len_markers = end_m.saturating_sub(start_m);
                        if len_markers >= min_markers
                            && (cm_positions[end_m.saturating_sub(1)] - cm_positions[start_m])
                                >= min_seed_cm
                        {
                            seeds.push(PbwtSeed {
                                start_marker: start_m as u32,
                                ..seed
                            });
                        }
                    },
                );
            }
        }
        pbwt.advance(marker, |hap| matrix.allele(marker, hap));
    }

    if is_last {
        if is_cancelled(cancelled) {
            return Err("Interrupted".to_string());
        }
        let last_cm = cm_positions[own_end - 1];
        let mut terminal_left = warmup_start;
        while terminal_left < own_end && (last_cm - cm_positions[terminal_left]) >= min_seed_cm {
            terminal_left += 1;
        }
        if let (Some(s_cm), Some(s_cnt)) = (
            terminal_left.checked_sub(1),
            own_end.checked_sub(min_markers),
        ) {
            let max_start = s_cm.min(s_cnt) as u32;
            pbwt.flush_terminal_matches(own_end, max_start, ref_hap_count, |seed| {
                let start_m = (seed.start_marker as usize).max(warmup_start);
                if warmup_start > 0 && start_m == warmup_start {
                    boundary_seeds.push(PbwtSeed {
                        start_marker: start_m as u32,
                        ..seed
                    });
                    return;
                }
                let end_m = seed.end_marker as usize;
                let len_markers = end_m.saturating_sub(start_m);
                if len_markers >= min_markers
                    && (cm_positions[end_m.saturating_sub(1)] - cm_positions[start_m])
                        >= min_seed_cm
                {
                    seeds.push(PbwtSeed {
                        start_marker: start_m as u32,
                        ..seed
                    });
                }
            });
        }
    }

    if !boundary_seeds.is_empty() {
        if boundary_seeds.len() <= 64 {
            for seed in boundary_seeds {
                let start_m = recover_exact_left_start(
                    matrix,
                    seed.hap_a as usize,
                    seed.hap_b as usize,
                    seed.start_marker as usize,
                );
                let end_m = seed.end_marker as usize;
                let len_markers = end_m.saturating_sub(start_m);
                if len_markers >= min_markers
                    && (cm_positions[end_m.saturating_sub(1)] - cm_positions[start_m])
                        >= min_seed_cm
                {
                    seeds.push(PbwtSeed {
                        start_marker: start_m as u32,
                        ..seed
                    });
                }
            }
        } else {
            let index =
                LeftBoundaryIndex::build(matrix, warmup_start, &boundary_seeds, cancelled)?;
            seeds.reserve(boundary_seeds.len());
            for seed in boundary_seeds {
                let start_m = index.exact_start(seed.hap_a as usize, seed.hap_b as usize);
                let end_m = seed.end_marker as usize;
                let len_markers = end_m.saturating_sub(start_m);
                if len_markers >= min_markers
                    && (cm_positions[end_m.saturating_sub(1)] - cm_positions[start_m])
                        >= min_seed_cm
                {
                    seeds.push(PbwtSeed {
                        start_marker: start_m as u32,
                        ..seed
                    });
                }
            }
        }
    }
    Ok(seeds)
}

#[cfg(test)]
mod tests {
    use super::{scan_chromosome_seeds, BitHaplotypeMatrix, PbwtSeed};

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

    #[test]
    fn parallel_pbwt_is_bit_identical_to_single_threaded_across_chunk_boundaries() {
        // 4,096 markers triggers 2-chunk and 4-chunk parallel PBWT splits at 2048 and 1024/2048/3072.
        // Includes:
        //  - (0, 2): exact match ending precisely at chunk boundary 2048 (`[2030, 2048)`, 18 markers / 17.0 cM)
        //  - (1, 3): full-length chromosome match (`[0, 4096)`, 4095.0 cM) crossing all chunk boundaries
        let n_markers = 4096usize;
        let n_haps = 4usize; // 0,1 = ref; 2,3 = target
        let mut matrix = BitHaplotypeMatrix::new(n_haps, n_markers);
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 1.0).collect(); // 1.0 cM spacing
        for m in 0..n_markers {
            let a0 = ((m * 7 + 3) % 2) as u8;
            let a1 = ((m * 11 + 1) % 3 == 0) as u8;
            let a2 = if (2030..2048).contains(&m) {
                a0
            } else {
                1 - a0
            };
            let a3 = a1;
            matrix.set_allele(m, 0, Some(a0));
            matrix.set_allele(m, 1, Some(a1));
            matrix.set_allele(m, 2, Some(a2));
            matrix.set_allele(m, 3, Some(a3));
        }

        let run_with_threads = |threads: usize, ref_k: Option<usize>| -> Vec<PbwtSeed> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut seeds =
                pool.install(|| scan_chromosome_seeds(&matrix, &cm, 15.0, 15, ref_k));
            seeds.sort_unstable_by_key(|s| (s.hap_a, s.hap_b, s.start_marker, s.end_marker));
            seeds
        };

        for ref_k in [None, Some(2)] {
            let s1 = run_with_threads(1, ref_k);
            let s2 = run_with_threads(2, ref_k);
            let s4 = run_with_threads(4, ref_k);
            assert!(
                s1.iter().any(|s| {
                    s.hap_a == 0 && s.hap_b == 2 && s.start_marker == 2030 && s.end_marker == 2048
                }),
                "expected exact boundary seed [2030, 2048) for ref_k={ref_k:?}"
            );
            assert!(
                s1.iter().any(|s| {
                    s.hap_a == 1 && s.hap_b == 3 && s.start_marker == 0 && s.end_marker == 4096
                }),
                "expected full-length seed [0, 4096) for ref_k={ref_k:?}"
            );
            assert_eq!(
                s1, s2,
                "2-thread PBWT seeds diverged from 1-thread for ref_k={ref_k:?}"
            );
            assert_eq!(
                s1, s4,
                "4-thread PBWT seeds diverged from 1-thread for ref_k={ref_k:?}"
            );
        }
    }

    #[test]
    fn parallel_pbwt_left_boundary_index_matches_single_threaded_on_high_similarity_panel() {
        // 64 haplotypes x 4,096 markers (> 64 boundary seeds to exercise LeftBoundaryIndex RMQ):
        //  - haps 0..32: identical across all 4,096 markers
        //  - haps 32..48: diverge from haps 0..32 at marker 500, identical on [501, 4096)
        //  - haps 48..64: missing (`None`) at marker 1200, identical to haps 0..32 on [1201, 4096)
        let n_markers = 4096usize;
        let n_haps = 64usize;
        let mut matrix = BitHaplotypeMatrix::new(n_haps, n_markers);
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            let base = ((m * 5 + 1) % 2) as u8;
            for h in 0..32 {
                matrix.set_allele(m, h, Some(base));
            }
            for h in 32..48 {
                let a = if m == 500 { 1 - base } else { base };
                matrix.set_allele(m, h, Some(a));
            }
            for h in 48..64 {
                let a = if m == 1200 { None } else { Some(base) };
                matrix.set_allele(m, h, a);
            }
        }

        let run_with_threads = |threads: usize| -> Vec<PbwtSeed> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut seeds = pool.install(|| scan_chromosome_seeds(&matrix, &cm, 5.0, 20, None));
            seeds.sort_unstable_by_key(|s| (s.hap_a, s.hap_b, s.start_marker, s.end_marker));
            seeds
        };

        let s1 = run_with_threads(1);
        let s2 = run_with_threads(2);
        let s4 = run_with_threads(4);
        assert_eq!(s1, s2, "2-thread RMQ boundary recovery diverged from 1-thread");
        assert_eq!(s1, s4, "4-thread RMQ boundary recovery diverged from 1-thread");
        // Spot-check exact starts for each group:
        assert!(s4.iter().any(|s| s.hap_a == 0 && s.hap_b == 1 && s.start_marker == 0 && s.end_marker == 4096));
        assert!(s4.iter().any(|s| s.hap_a == 0 && s.hap_b == 32 && s.start_marker == 501 && s.end_marker == 4096));
        assert!(s4.iter().any(|s| s.hap_a == 48 && s.hap_b == 49 && s.start_marker == 1201 && s.end_marker == 4096));
    }
}
