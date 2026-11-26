//! Multi-parent ancestry / mosaic painting (Mode 2: with `--ref`) using
//! whole-chromosome PBWT candidate narrowing + genetic-step Viterbi decoding,
//! plus Union-of-Breakpoints sparse ancestry dosage matrix for `src/hgwas`.

use super::pbwt::scan_chromosome_seeds_cancellable;
use super::phasing::SampleHaplotypeLayout;
use super::types::{AncestrySegment, BitHaplotypeMatrix, HaploidIbdConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// Exact population-level union-of-breakpoints sparse ancestry dosage matrix
/// for one chromosome, stored in bin-major (CSC-like) layout so `src/hgwas`
/// can scan genetic bins in O(1) seek without materializing a dense 10M-SNP matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChromosomeAncestryBinsSparse {
    pub chrom_idx: u16,
    pub n_target_samples: usize,
    pub n_parents: usize,
    pub bin_starts_bp: Vec<u64>,
    pub bin_ends_bp: Vec<u64>,
    pub bin_starts_cm: Vec<f32>,
    pub bin_ends_cm: Vec<f32>,
    /// Length `n_bins + 1`: entries for bin `b` lie in `bin_ptr[b]..bin_ptr[b + 1]`.
    pub bin_ptr: Vec<u64>,
    /// Target sample index for each nonzero `(bin, sample, parent)` entry.
    pub sample_idx: Vec<u32>,
    /// Reference parent sample index for each nonzero entry.
    pub parent_idx: Vec<u32>,
    /// Parent dosage in `{0.5, 1.0}` (1.0 for inbred/DH or homozygous diploid ancestry).
    pub dosage: Vec<f32>,
}

/// Paint multi-parent ancestry mosaic segments for all target haplotypes on one
/// chromosome against a reference parent panel.
///
/// `joint_matrix` lays out reference haplotypes in columns `0..ref_layout.total_haps`
/// and target haplotypes in columns `ref_layout.total_haps..joint_matrix.n_haps`.
#[inline]
fn is_cancelled(cancelled: Option<&AtomicBool>) -> bool {
    cancelled
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

/// In-place 64x64 bit-matrix transpose (Hacker's Delight Fig. 7-2).
/// Converts 64 marker-major 64-haplotype words into 64 haplotype-major 64-marker words.
#[inline]
fn transpose_64x64(a: &mut [u64; 64]) {
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    let mut j: usize = 32;
    while j != 0 {
        let mut k = 0usize;
        while k < 64 {
            for i in 0..j {
                let x = a[k + i];
                let y = a[k + i + j];
                let t = ((x >> j) ^ y) & m;
                a[k + i + j] = y ^ t;
                a[k + i] = x ^ (t << j);
            }
            k += j << 1;
        }
        j >>= 1;
        m ^= m << j;
    }
}

/// Haplotype-major 64-bit packed view of `BitHaplotypeMatrix` for O(M / 64)
/// hardware-POPCNT mismatch counting with sequential L1 cache access.
struct HaplotypeMajorBits {
    words_per_hap: usize,
    allele_words: Vec<u64>,
    missing_words: Vec<u64>,
}

impl HaplotypeMajorBits {
    fn from_matrix(
        matrix: &BitHaplotypeMatrix,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self, String> {
        let n_haps = matrix.n_haps;
        let n_markers = matrix.n_markers;
        let words_per_hap = n_markers.div_ceil(64);
        let stride = matrix.words_per_marker;
        let mut allele_words = vec![0u64; n_haps * words_per_hap];
        let mut missing_words = vec![0u64; n_haps * words_per_hap];

        if n_haps == 0 || n_markers == 0 || stride == 0 {
            return Ok(Self {
                words_per_hap,
                allele_words,
                missing_words,
            });
        }

        let group_words = 64 * words_per_hap;
        allele_words
            .par_chunks_mut(group_words)
            .zip(missing_words.par_chunks_mut(group_words))
            .enumerate()
            .try_for_each(|(hap_word_idx, (out_a, out_m))| -> Result<(), String> {
                let haps_in_group = (n_haps - hap_word_idx * 64).min(64);
                let mut block_a = [0u64; 64];
                let mut block_m = [0u64; 64];
                for w_idx in 0..words_per_hap {
                    if (w_idx & 255) == 0 && is_cancelled(cancelled) {
                        return Err("Interrupted".to_string());
                    }
                    let m_start = w_idx * 64;
                    let m_len = (n_markers - m_start).min(64);
                    for b in 0..m_len {
                        let src = (m_start + b) * stride + hap_word_idx;
                        block_a[b] = matrix.allele_bits[src];
                        block_m[b] = matrix.missing_bits[src];
                    }
                    for b in m_len..64 {
                        block_a[b] = 0;
                        block_m[b] = 0;
                    }
                    transpose_64x64(&mut block_a);
                    transpose_64x64(&mut block_m);
                    for h in 0..haps_in_group {
                        let dst = h * words_per_hap + w_idx;
                        out_a[dst] = block_a[h];
                        out_m[dst] = block_m[h];
                    }
                }
                Ok(())
            })?;

        Ok(Self {
            words_per_hap,
            allele_words,
            missing_words,
        })
    }

    #[inline]
    fn count_mismatches(
        &self,
        hap_a: usize,
        hap_b: usize,
        start_marker: usize,
        end_marker: usize,
    ) -> (u32, u32) {
        if start_marker >= end_marker || self.words_per_hap == 0 {
            return (0, 0);
        }
        let base_a = hap_a * self.words_per_hap;
        let base_b = hap_b * self.words_per_hap;
        let w_start = start_marker >> 6;
        let w_end = (end_marker - 1) >> 6;
        let start_mask = !0u64 << (start_marker & 63);
        let end_rem = end_marker & 63;
        let end_mask = if end_rem == 0 {
            !0u64
        } else {
            (1u64 << end_rem) - 1
        };

        if w_start == w_end {
            let mask = start_mask & end_mask;
            let valid = !(self.missing_words[base_a + w_start]
                | self.missing_words[base_b + w_start])
                & mask;
            let mis = ((self.allele_words[base_a + w_start] ^ self.allele_words[base_b + w_start])
                & valid)
                .count_ones();
            return (valid.count_ones(), mis);
        }

        let all_a = &self.allele_words[base_a..base_a + self.words_per_hap];
        let all_b = &self.allele_words[base_b..base_b + self.words_per_hap];
        let miss_a = &self.missing_words[base_a..base_a + self.words_per_hap];
        let miss_b = &self.missing_words[base_b..base_b + self.words_per_hap];

        let v0 = !(miss_a[w_start] | miss_b[w_start]) & start_mask;
        let mut comp = v0.count_ones();
        let mut mis = ((all_a[w_start] ^ all_b[w_start]) & v0).count_ones();

        for w in (w_start + 1)..w_end {
            let valid = !(miss_a[w] | miss_b[w]);
            comp += valid.count_ones();
            mis += ((all_a[w] ^ all_b[w]) & valid).count_ones();
        }

        let ve = !(miss_a[w_end] | miss_b[w_end]) & end_mask;
        comp += ve.count_ones();
        mis += ((all_a[w_end] ^ all_b[w_end]) & ve).count_ones();
        (comp, mis)
    }

    /// Count exact mismatches and measure their genetic-coordinate dispersion
    /// directly at mismatch marker indices (completely invariant to `ancestry_step_cm`).
    /// Returns `(total_mismatches, mis_clusters, mis_span_cm)`, opening a new cluster
    /// only when consecutive mismatch markers are separated by `>= cluster_gap_cm`.
    #[inline]
    fn mismatch_dispersion(
        &self,
        hap_a: usize,
        hap_b: usize,
        start_marker: usize,
        end_marker: usize,
        cm_positions: &[f64],
        cluster_gap_cm: f64,
    ) -> (u32, u32, f64) {
        if start_marker >= end_marker || self.words_per_hap == 0 {
            return (0, 0, 0.0);
        }
        let base_a = hap_a * self.words_per_hap;
        let base_b = hap_b * self.words_per_hap;
        let w_start = start_marker >> 6;
        let w_end = (end_marker - 1) >> 6;
        let start_mask = !0u64 << (start_marker & 63);
        let end_rem = end_marker & 63;
        let end_mask = if end_rem == 0 {
            !0u64
        } else {
            (1u64 << end_rem) - 1
        };

        let all_a = &self.allele_words[base_a..base_a + self.words_per_hap];
        let all_b = &self.allele_words[base_b..base_b + self.words_per_hap];
        let miss_a = &self.missing_words[base_a..base_a + self.words_per_hap];
        let miss_b = &self.missing_words[base_b..base_b + self.words_per_hap];

        let mut total_mis = 0u32;
        let mut mis_clusters = 0u32;
        let mut first_mis_cm: Option<f64> = None;
        let mut last_mis_cm = 0.0f64;
        let mut scan_word = |w: usize, mask: u64| {
            let valid = !(miss_a[w] | miss_b[w]) & mask;
            let mut mis_bits = (all_a[w] ^ all_b[w]) & valid;
            while mis_bits != 0 {
                let bit = mis_bits.trailing_zeros() as usize;
                mis_bits &= mis_bits - 1;
                let m = (w << 6) + bit;
                let m_cm = cm_positions[m];
                total_mis = total_mis.saturating_add(1);
                if first_mis_cm.is_none() || (m_cm - last_mis_cm) >= cluster_gap_cm {
                    mis_clusters = mis_clusters.saturating_add(1);
                }
                if first_mis_cm.is_none() {
                    first_mis_cm = Some(m_cm);
                }
                last_mis_cm = m_cm;
            }
        };

        if w_start == w_end {
            scan_word(w_start, start_mask & end_mask);
        } else {
            scan_word(w_start, start_mask);
            for w in (w_start + 1)..w_end {
                scan_word(w, !0u64);
            }
            scan_word(w_end, end_mask);
        }
        let mis_span_cm = first_mis_cm
            .map(|lo| (last_mis_cm - lo).max(0.0))
            .unwrap_or(0.0);
        (total_mis, mis_clusters, mis_span_cm)
    }

    #[inline]
    fn is_observed(&self, hap: usize, marker: usize) -> bool {
        let w = marker >> 6;
        ((self.missing_words[hap * self.words_per_hap + w] >> (marker & 63)) & 1) == 0
    }

    /// Collect exact marker indices in `[start_marker, end_marker)` where `hap_a` and
    /// `hap_b` are both observed and carry different alleles.
    fn collect_mismatch_markers(
        &self,
        hap_a: usize,
        hap_b: usize,
        start_marker: usize,
        end_marker: usize,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        if start_marker >= end_marker || self.words_per_hap == 0 {
            return out;
        }
        let base_a = hap_a * self.words_per_hap;
        let base_b = hap_b * self.words_per_hap;
        let w_start = start_marker >> 6;
        let w_end = (end_marker - 1) >> 6;
        let start_mask = !0u64 << (start_marker & 63);
        let end_rem = end_marker & 63;
        let end_mask = if end_rem == 0 {
            !0u64
        } else {
            (1u64 << end_rem) - 1
        };
        let all_a = &self.allele_words[base_a..base_a + self.words_per_hap];
        let all_b = &self.allele_words[base_b..base_b + self.words_per_hap];
        let miss_a = &self.missing_words[base_a..base_a + self.words_per_hap];
        let miss_b = &self.missing_words[base_b..base_b + self.words_per_hap];
        let mut scan_word = |w: usize, mask: u64| {
            let valid = !(miss_a[w] | miss_b[w]) & mask;
            let mut mis_bits = (all_a[w] ^ all_b[w]) & valid;
            while mis_bits != 0 {
                let bit = mis_bits.trailing_zeros() as usize;
                mis_bits &= mis_bits - 1;
                out.push((w << 6) + bit);
            }
        };
        if w_start == w_end {
            scan_word(w_start, start_mask & end_mask);
        } else {
            scan_word(w_start, start_mask);
            for w in (w_start + 1)..w_end {
                scan_word(w, !0u64);
            }
            scan_word(w_end, end_mask);
        }
        out
    }

    /// Find the largest marker index in `[search_start, before_marker)` where `hap_a` and `hap_b` mismatch.
    fn find_nearest_mismatch_left(
        &self,
        hap_a: usize,
        hap_b: usize,
        search_start: usize,
        before_marker: usize,
    ) -> Option<usize> {
        if search_start >= before_marker || self.words_per_hap == 0 {
            return None;
        }
        let base_a = hap_a * self.words_per_hap;
        let base_b = hap_b * self.words_per_hap;
        let w_lo = search_start >> 6;
        let w_hi = (before_marker - 1) >> 6;
        let lo_mask = !0u64 << (search_start & 63);
        let hi_rem = before_marker & 63;
        let hi_mask = if hi_rem == 0 {
            !0u64
        } else {
            (1u64 << hi_rem) - 1
        };
        for w in (w_lo..=w_hi).rev() {
            let mut mask = !0u64;
            if w == w_lo {
                mask &= lo_mask;
            }
            if w == w_hi {
                mask &= hi_mask;
            }
            let valid = !(self.missing_words[base_a + w] | self.missing_words[base_b + w]) & mask;
            let mis_bits = (self.allele_words[base_a + w] ^ self.allele_words[base_b + w]) & valid;
            if mis_bits != 0 {
                let bit = 63 - mis_bits.leading_zeros() as usize;
                return Some((w << 6) + bit);
            }
        }
        None
    }

    /// Find the smallest marker index in `(after_marker, search_end)` where `hap_a` and `hap_b` mismatch.
    fn find_nearest_mismatch_right(
        &self,
        hap_a: usize,
        hap_b: usize,
        after_marker: usize,
        search_end: usize,
    ) -> Option<usize> {
        let start = after_marker + 1;
        if start >= search_end || self.words_per_hap == 0 {
            return None;
        }
        let base_a = hap_a * self.words_per_hap;
        let base_b = hap_b * self.words_per_hap;
        let w_lo = start >> 6;
        let w_hi = (search_end - 1) >> 6;
        let lo_mask = !0u64 << (start & 63);
        let hi_rem = search_end & 63;
        let hi_mask = if hi_rem == 0 {
            !0u64
        } else {
            (1u64 << hi_rem) - 1
        };
        for w in w_lo..=w_hi {
            let mut mask = !0u64;
            if w == w_lo {
                mask &= lo_mask;
            }
            if w == w_hi {
                mask &= hi_mask;
            }
            let valid = !(self.missing_words[base_a + w] | self.missing_words[base_b + w]) & mask;
            let mis_bits = (self.allele_words[base_a + w] ^ self.allele_words[base_b + w]) & valid;
            if mis_bits != 0 {
                let bit = mis_bits.trailing_zeros() as usize;
                return Some((w << 6) + bit);
            }
        }
        None
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn count_mismatches_weighted(
        &self,
        hap_a: usize,
        hap_b: usize,
        start_marker: usize,
        end_marker: usize,
        weights: &RefAlleleWeightPlanes,
        match_reward: f32,
        flat_mismatch_delta: f32,
        freq_mismatch_delta: f32,
    ) -> (u32, f32, f32) {
        if start_marker >= end_marker || self.words_per_hap == 0 {
            return (0, 0.0, 0.0);
        }
        let base_a = hap_a * self.words_per_hap;
        let base_b = hap_b * self.words_per_hap;
        let w_start = start_marker >> 6;
        let w_end = (end_marker - 1) >> 6;
        let start_mask = !0u64 << (start_marker & 63);
        let end_rem = end_marker & 63;
        let end_mask = if end_rem == 0 {
            !0u64
        } else {
            (1u64 << end_rem) - 1
        };

        let all_a = &self.allele_words[base_a..base_a + self.words_per_hap];
        let all_b = &self.allele_words[base_b..base_b + self.words_per_hap];
        let miss_a = &self.missing_words[base_a..base_a + self.words_per_hap];
        let miss_b = &self.missing_words[base_b..base_b + self.words_per_hap];

        let mut comp_freq = 0u32;
        let mut mis_freq = 0u32;
        let mut comp_flat = 0u32;
        let mut mis_flat = 0u32;
        let mut qsum = 0u64;
        let mut accum_word = |w: usize, mask: u64| {
            let valid = !(miss_a[w] | miss_b[w]) & mask;
            if valid == 0 {
                return;
            }
            let diff = all_a[w] ^ all_b[w];
            let active = weights.freq_active_plane[w];
            let valid_freq = valid & active;
            if valid_freq != 0 {
                comp_freq += valid_freq.count_ones();
                mis_freq += (diff & valid_freq).count_ones();
                let v1 = valid_freq & all_a[w];
                let v0 = valid_freq & !all_a[w];
                for b in 0..6 {
                    let bits = (v0 & weights.w0_planes[b][w]) | (v1 & weights.w1_planes[b][w]);
                    qsum += (bits.count_ones() as u64) << b;
                }
            }
            let valid_flat = valid & !active;
            if valid_flat != 0 {
                comp_flat += valid_flat.count_ones();
                mis_flat += (diff & valid_flat).count_ones();
            }
        };

        if w_start == w_end {
            accum_word(w_start, start_mask & end_mask);
        } else {
            accum_word(w_start, start_mask);
            for w in (w_start + 1)..w_end {
                accum_word(w, !0u64);
            }
            accum_word(w_end, end_mask);
        }
        let comp_total = comp_freq + comp_flat;
        let info_freq = qsum as f32 * (1.0 / 16.0);
        let info_flat = comp_flat as f32 * match_reward;
        let info_total = info_freq + info_flat;
        let raw_freq = info_freq - mis_freq as f32 * freq_mismatch_delta;
        let raw_flat = info_flat - mis_flat as f32 * flat_mismatch_delta;
        (comp_total, info_total, raw_freq + raw_flat)
    }
}

const FREQ_MODEL_MIN_REF_HAPS: usize = 6;
const FREQ_ERR_RATE: f64 = 0.0025;

/// 6-bit fixed-point (`1/16` resolution, `[0.0, 3.9375]`) bitplanes of
/// `ln((1 - err) / af(allele))` precomputed from the reference panel so
/// frequency-aware Bayes LLR emissions run at 64 markers per hardware POPCNT.
/// Sites with fewer than `FREQ_MODEL_MIN_REF_HAPS` observed reference
/// haplotypes have their `freq_active_plane` bit cleared and fall back to the
/// exact legacy flat `+0.7 / -4.5` scoring model.
struct RefAlleleWeightPlanes {
    freq_active_plane: Vec<u64>,
    w0_planes: [Vec<u64>; 6],
    w1_planes: [Vec<u64>; 6],
}

impl RefAlleleWeightPlanes {
    fn from_reference_panel(
        matrix: &BitHaplotypeMatrix,
        n_ref_haps: usize,
        err_rate: f64,
    ) -> Option<Self> {
        if n_ref_haps < FREQ_MODEL_MIN_REF_HAPS
            || matrix.n_markers == 0
            || matrix.words_per_marker == 0
        {
            return None;
        }
        let n_markers = matrix.n_markers;
        let words_per_hap = n_markers.div_ceil(64);
        let stride = matrix.words_per_marker;
        let ref_full_words = n_ref_haps / 64;
        let ref_rem = n_ref_haps & 63;
        let ref_tail_mask = if ref_rem == 0 {
            0u64
        } else {
            (1u64 << ref_rem) - 1
        };

        let per_word: Vec<(u64, [u64; 6], [u64; 6])> = (0..words_per_hap)
            .into_par_iter()
            .map(|w_idx| {
                let m_start = w_idx * 64;
                let m_len = (n_markers - m_start).min(64);
                let mut freq_active = 0u64;
                let mut w0 = [0u64; 6];
                let mut w1 = [0u64; 6];
                for bit in 0..m_len {
                    let m = m_start + bit;
                    let row_off = m * stride;
                    let mut obs_ref = 0u32;
                    let mut c1 = 0u32;
                    for rw in 0..ref_full_words {
                        let valid = !matrix.missing_bits[row_off + rw];
                        obs_ref += valid.count_ones();
                        c1 += (matrix.allele_bits[row_off + rw] & valid).count_ones();
                    }
                    if ref_rem != 0 {
                        let valid = !matrix.missing_bits[row_off + ref_full_words] & ref_tail_mask;
                        obs_ref += valid.count_ones();
                        c1 += (matrix.allele_bits[row_off + ref_full_words] & valid).count_ones();
                    }
                    if (obs_ref as usize) < FREQ_MODEL_MIN_REF_HAPS {
                        continue;
                    }
                    let bit_mask = 1u64 << bit;
                    freq_active |= bit_mask;
                    let p1 = ((c1 as f64 + 0.5) / (obs_ref as f64 + 1.0)).clamp(0.02, 0.98);
                    let af0 = 1.0 - p1;
                    let af1 = p1;
                    let q0 = ((((1.0 - err_rate) / af0).ln().clamp(0.0, 63.0 / 16.0)) * 16.0)
                        .round() as u32;
                    let q1 = ((((1.0 - err_rate) / af1).ln().clamp(0.0, 63.0 / 16.0)) * 16.0)
                        .round() as u32;
                    for b in 0..6 {
                        if (q0 >> b) & 1 != 0 {
                            w0[b] |= bit_mask;
                        }
                        if (q1 >> b) & 1 != 0 {
                            w1[b] |= bit_mask;
                        }
                    }
                }
                (freq_active, w0, w1)
            })
            .collect();

        let mut freq_active_plane = vec![0u64; words_per_hap];
        let mut w0_planes = std::array::from_fn(|_| vec![0u64; words_per_hap]);
        let mut w1_planes = std::array::from_fn(|_| vec![0u64; words_per_hap]);
        let mut any_freq_active = false;
        for (w_idx, (active, w0, w1)) in per_word.into_iter().enumerate() {
            if active != 0 {
                any_freq_active = true;
            }
            freq_active_plane[w_idx] = active;
            for b in 0..6 {
                w0_planes[b][w_idx] = w0[b];
                w1_planes[b][w_idx] = w1[b];
            }
        }
        if !any_freq_active {
            return None;
        }
        Some(Self {
            freq_active_plane,
            w0_planes,
            w1_planes,
        })
    }
}

/// One-sided Binomial log-likelihood-ratio deviance `2 * [m*ln(p_hat/p0) + (n-m)*ln((1-p_hat)/(1-p0))]`
/// testing whether observed mismatch rate `m/n` significantly exceeds the baseline error rate `p0`.
#[inline]
fn one_sided_binomial_deviance(m: f64, n: f64, p0: f64) -> f64 {
    if n <= 0.0 || m <= 0.0 {
        return 0.0;
    }
    let p_hat = (m / n).clamp(0.0, 1.0);
    if p_hat <= p0 {
        return 0.0;
    }
    let p0 = p0.clamp(1e-6, 1.0 - 1e-6);
    let term1 = m * (p_hat / p0).ln();
    let term2 = if m < n {
        (n - m) * ((1.0 - p_hat) / (1.0 - p0)).ln()
    } else {
        0.0
    };
    (2.0 * (term1 + term2)).max(0.0)
}

pub fn paint_chromosome_ancestry(
    joint_matrix: &BitHaplotypeMatrix,
    ref_layout: &SampleHaplotypeLayout,
    target_layout: &SampleHaplotypeLayout,
    chrom_idx: u16,
    positions_bp: &[u64],
    cm_positions: &[f64],
    config: &HaploidIbdConfig,
) -> (Vec<AncestrySegment>, ChromosomeAncestryBinsSparse) {
    paint_chromosome_ancestry_cancellable(
        joint_matrix,
        ref_layout,
        target_layout,
        chrom_idx,
        positions_bp,
        cm_positions,
        config,
        None,
    )
    .unwrap_or_default()
}

/// Cancellable variant of `paint_chromosome_ancestry` that checks `cancelled`
/// during PBWT seed scanning, fallback donor scoring, and per-haplotype Viterbi decoding.
#[allow(clippy::too_many_arguments)]
pub fn paint_chromosome_ancestry_cancellable(
    joint_matrix: &BitHaplotypeMatrix,
    ref_layout: &SampleHaplotypeLayout,
    target_layout: &SampleHaplotypeLayout,
    chrom_idx: u16,
    positions_bp: &[u64],
    cm_positions: &[f64],
    config: &HaploidIbdConfig,
    cancelled: Option<&AtomicBool>,
) -> Result<(Vec<AncestrySegment>, ChromosomeAncestryBinsSparse), String> {
    if is_cancelled(cancelled) {
        return Err("Interrupted".to_string());
    }
    let n_markers = joint_matrix.n_markers;
    let n_ref_haps = ref_layout.total_haps;
    let n_target_haps = target_layout.total_haps;
    if n_markers == 0 || n_ref_haps == 0 || n_target_haps == 0 {
        return Ok((Vec::new(), ChromosomeAncestryBinsSparse::default()));
    }

    // Step 1: Partition the chromosome into genetic steps (~0.05 cM) so even
    // 10M-SNP chromosomes decode in ~2,000-4,000 lightweight Viterbi steps.
    let step_ranges = build_genetic_step_ranges(cm_positions, config.ancestry_step_cm);
    let n_steps = step_ranges.len();

    // Step 2: If reference panel is larger than `ancestry_top_donors`, run a
    // single whole-chromosome PBWT sweep (cross-group target->ref only) to
    // select the top candidate reference haplotypes per target haplotype.
    let per_target_candidates: Vec<Vec<usize>> = if n_ref_haps <= config.ancestry_top_donors {
        vec![(0..n_ref_haps).collect(); n_target_haps]
    } else {
        let seeds = scan_chromosome_seeds_cancellable(
            joint_matrix,
            cm_positions,
            (config.min_seed_cm * 0.5).max(0.2),
            config.min_markers.min(10).max(4),
            Some(n_ref_haps),
            cancelled,
        )?;
        let mut counts: Vec<HashMap<usize, u64>> = vec![HashMap::new(); n_target_haps];
        for seed in seeds {
            let (ref_h, tgt_h) = if (seed.hap_a as usize) < n_ref_haps {
                (seed.hap_a as usize, seed.hap_b as usize - n_ref_haps)
            } else {
                (seed.hap_b as usize, seed.hap_a as usize - n_ref_haps)
            };
            let span = seed.end_marker.saturating_sub(seed.start_marker) as u64;
            *counts[tgt_h].entry(ref_h).or_insert(0) += span;
        }
        counts
            .into_par_iter()
            .enumerate()
            .map(|(tgt_local_h, map)| -> Result<Vec<usize>, String> {
                if is_cancelled(cancelled) {
                    return Err("Interrupted".to_string());
                }
                let mut ranked: Vec<(usize, u64)> = map.into_iter().collect();
                ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let mut cands: Vec<usize> = ranked
                    .into_iter()
                    .take(config.ancestry_top_donors)
                    .map(|(h, _)| h)
                    .collect();
                if cands.is_empty() {
                    // Order-independent evidence fallback when PBWT emitted zero seeds
                    // (e.g. short chromosomes/regions below `min_seed_cm` or high divergence):
                    // score all reference haplotypes by actual allele match evidence instead
                    // of taking the first `ancestry_top_donors` by file order.
                    let tgt_col = n_ref_haps + tgt_local_h;
                    let stride = (n_markers / 4096).max(1);
                    let mut scored: Vec<(usize, bool, f32, u32)> = (0..n_ref_haps)
                        .map(|ref_h| {
                            let (comp, mis) = if stride == 1 {
                                joint_matrix.count_mismatches(tgt_col, ref_h, 0, n_markers)
                            } else {
                                let mut c = 0u32;
                                let mut m_err = 0u32;
                                let mut m = 0usize;
                                while m < n_markers {
                                    if let (Some(a), Some(b)) = (
                                        joint_matrix.allele(m, tgt_col),
                                        joint_matrix.allele(m, ref_h),
                                    ) {
                                        c += 1;
                                        if a != b {
                                            m_err += 1;
                                        }
                                    }
                                    m += stride;
                                }
                                (c, m_err)
                            };
                            let matched = comp.saturating_sub(mis);
                            let score = if comp > 0 {
                                matched as f32 * 0.7 - mis as f32 * 4.5
                            } else {
                                f32::NEG_INFINITY
                            };
                            (ref_h, comp > 0, score, matched)
                        })
                        .collect();
                    scored.sort_by(|a, b| {
                        b.1.cmp(&a.1)
                            .then_with(|| b.2.total_cmp(&a.2))
                            .then_with(|| b.3.cmp(&a.3))
                            .then_with(|| a.0.cmp(&b.0))
                    });
                    cands.extend(
                        scored
                            .into_iter()
                            .take(config.ancestry_top_donors)
                            .map(|(h, _, _, _)| h),
                    );
                }
                Ok(cands)
            })
            .collect::<Result<Vec<Vec<usize>>, String>>()?
    };

    // Step 3: Parallel step-level Viterbi mosaic decoding per target haplotype.
    // Transpose `joint_matrix` once in parallel into 64-bit haplotype-major words
    // so emission mismatch counts run at 64 markers per hardware POPCNT instruction
    // with sequential L1 cache access.
    let hap_major = HaplotypeMajorBits::from_matrix(joint_matrix, cancelled)?;
    let err_rate = if config.ancestry_error_rate.is_finite()
        && config.ancestry_error_rate > 0.0
        && config.ancestry_error_rate < 0.25
    {
        config.ancestry_error_rate
    } else {
        FREQ_ERR_RATE
    };
    let ref_weights =
        RefAlleleWeightPlanes::from_reference_panel(joint_matrix, n_ref_haps, err_rate);
    // Log-likelihood ratio scores relative to `Uncalled` background:
    //   - For multi-parent panels (`n_ref_haps >= 6`), use empirical reference allele
    //     frequencies `af(t)` with `err = ancestry_error_rate` (default `0.0025`):
    //     `match = ln((1 - err) / af(t))`, `mismatch = match - ln((1 - err) / err)`.
    //     This suppresses false-positive IBS on common alleles for out-of-panel
    //     founders while preserving full rare-allele signal.
    //   - For tiny synthetic panels (`n_ref_haps < 6`), fall back to uniform `p = 0.5`
    //     (`match = +0.7`, `mismatch = -4.5`).
    let mismatch_penalty: f32 = 4.5;
    let match_reward: f32 = 0.7;
    let flat_mismatch_delta: f32 = match_reward + mismatch_penalty;
    let default_freq_mismatch_delta: f32 = ((1.0 - FREQ_ERR_RATE) / FREQ_ERR_RATE).ln() as f32;
    let freq_mismatch_delta: f32 = ((1.0 - err_rate) / err_rate).ln() as f32;
    let switch_penalty = config.ancestry_switch_penalty.max(0.5);
    let step_eff_markers_cap = config.ancestry_step_max_markers as f32;

    let per_hap_segments: Result<Vec<Vec<AncestrySegment>>, String> = (0..n_target_haps)
        .into_par_iter()
        .map(|tgt_local_h| {
            if is_cancelled(cancelled) {
                return Err("Interrupted".to_string());
            }
            let tgt_col = n_ref_haps + tgt_local_h;
            let (target_sample, target_hap) = target_layout.hap_to_sample[tgt_local_h];
            let candidates = &per_target_candidates[tgt_local_h];
            let k = candidates.len();
            if k == 0 || n_steps == 0 {
                return Ok(Vec::new());
            }

            // Standard (k + 1)-state Viterbi DP with neutral missing-data emission:
            //   - states `0..k`: candidate reference haplotypes; a step with zero
            //     comparable markers emits neutral `0.0` so scattered missing calls
            //     carry the hidden state forward instead of killing it or granting
            //     free zero-penalty switches.
            //   - state `k`: explicit `Uncalled` background state with emission `0.0`.
            // Every state transition between distinct states pays `switch_penalty`.
            let uncalled_state = k;
            let n_states = k + 1;
            let mut prev_dp = vec![f32::NEG_INFINITY; n_states];
            let mut curr_dp = vec![f32::NEG_INFINITY; n_states];
            // Use u32 backpointers so large `--ancestry-top-donors` (e.g. >= 65536)
            // never wrap `uncalled_state` to 0.
            let mut backptr = vec![uncalled_state as u32; n_steps * n_states];
            let mut step_scores = vec![0.0f32; n_steps * k];
            let mut step_infos = vec![0.0f32; n_steps * k];
            let mut step_comparable = vec![0u32; n_steps * k];
            let mut step_max_comps = vec![0u32; n_steps];
            let mut step_max_infos = vec![0.0f32; n_steps];
            let mut cand_total_comp = vec![0u64; k];

            // Precompute per-step raw LLR emissions (`step_scores`), match-information
            // potentials (`step_infos`), and comparable counts (`step_comparable`).
            // Raw `step_scores` are kept unmodified for worklist gap resolution and
            // segment confidence/ambiguity reporting.
            for (step_idx, &(m_start, m_end)) in step_ranges.iter().enumerate() {
                if (step_idx & 255) == 0 && is_cancelled(cancelled) {
                    return Err("Interrupted".to_string());
                }
                let row_off = step_idx * k;
                let mut step_max_comp = 0u32;
                let mut step_max_info = 0.0f32;
                for (c_idx, &ref_col) in candidates.iter().enumerate() {
                    let (comp, info_sum, raw_score) = if let Some(ref weights) = ref_weights {
                        hap_major.count_mismatches_weighted(
                            tgt_col,
                            ref_col,
                            m_start,
                            m_end,
                            weights,
                            match_reward,
                            flat_mismatch_delta,
                            freq_mismatch_delta,
                        )
                    } else {
                        let (c, m) = hap_major.count_mismatches(tgt_col, ref_col, m_start, m_end);
                        let info = c as f32 * match_reward;
                        (c, info, info - m as f32 * flat_mismatch_delta)
                    };
                    step_comparable[row_off + c_idx] = comp;
                    step_infos[row_off + c_idx] = info_sum;
                    step_scores[row_off + c_idx] = raw_score;
                    cand_total_comp[c_idx] = cand_total_comp[c_idx].saturating_add(comp as u64);
                    if comp > step_max_comp {
                        step_max_comp = comp;
                    }
                    if info_sum > step_max_info {
                        step_max_info = info_sum;
                    }
                }
                step_max_comps[step_idx] = step_max_comp;
                step_max_infos[step_idx] = step_max_info;
                if step_eff_markers_cap > 0.0 && (step_max_comp as f32) > step_eff_markers_cap {
                    let step_scale = step_eff_markers_cap / step_max_comp as f32;
                    step_max_infos[step_idx] *= step_scale;
                    for score in &mut step_scores[row_off..row_off + k] {
                        *score *= step_scale;
                    }
                    for info in &mut step_infos[row_off..row_off + k] {
                        *info *= step_scale;
                    }
                }
            }

            // Local step-level missingness compensation for Viterbi DP only:
            // Scales by `FOUNDER_MISSING_CREDIT_RATIO` (< 1.0) times the step's
            // frequency-weighted missing information `max_info - cand_info`, so
            // missingness NEVER beats a local match, low-information major-allele
            // sites receive proportionally tiny credit, and founders missing
            // elsewhere get ZERO bonus where locally observed.
            // All-missing founders (`cand_total_comp[c] == 0`) always emit `0.0`.
            const FOUNDER_MISSING_CREDIT_RATIO: f32 = 0.75;

            for step_idx in 0..n_steps {
                if (step_idx & 255) == 0 && is_cancelled(cancelled) {
                    return Err("Interrupted".to_string());
                }
                let row_off = step_idx * k;
                let bp_off = step_idx * n_states;
                let step_max_comp = step_max_comps[step_idx];
                let step_max_info = step_max_infos[step_idx];
                let dp_emit_for = |c_idx: usize| -> f32 {
                    let raw = step_scores[row_off + c_idx];
                    if step_max_comp == 0 || cand_total_comp[c_idx] == 0 {
                        return raw;
                    }
                    let missing_info = (step_max_info - step_infos[row_off + c_idx]).max(0.0);
                    if missing_info == 0.0 {
                        raw
                    } else {
                        raw + FOUNDER_MISSING_CREDIT_RATIO * missing_info
                    }
                };

                if step_idx == 0 {
                    for c_idx in 0..k {
                        curr_dp[c_idx] = dp_emit_for(c_idx);
                        backptr[bp_off + c_idx] = c_idx as u32;
                    }
                    curr_dp[uncalled_state] = 0.0;
                    backptr[bp_off + uncalled_state] = uncalled_state as u32;
                } else {
                    let mut best_prev_val = prev_dp[uncalled_state];
                    let mut best_prev_idx = uncalled_state;
                    for c_prev in 0..k {
                        let val = prev_dp[c_prev];
                        if val > best_prev_val {
                            best_prev_val = val;
                            best_prev_idx = c_prev;
                        }
                    }
                    let switch_val = best_prev_val - switch_penalty;
                    for s_idx in 0..n_states {
                        let emit = if s_idx < k { dp_emit_for(s_idx) } else { 0.0 };
                        let stay = prev_dp[s_idx];
                        if stay >= switch_val {
                            curr_dp[s_idx] = stay + emit;
                            backptr[bp_off + s_idx] = s_idx as u32;
                        } else {
                            curr_dp[s_idx] = switch_val + emit;
                            backptr[bp_off + s_idx] = best_prev_idx as u32;
                        }
                    }
                }
                std::mem::swap(&mut prev_dp, &mut curr_dp);
            }

            // Backtrack optimal path across `0..=k` (ties prefer `uncalled_state`).
            let mut best_end_idx = uncalled_state;
            let mut best_end_val = prev_dp[uncalled_state];
            for c_idx in 0..k {
                let val = prev_dp[c_idx];
                if val > best_end_val {
                    best_end_val = val;
                    best_end_idx = c_idx;
                }
            }
            let mut path = vec![uncalled_state; n_steps];
            let mut cur_idx = best_end_idx;
            for step_idx in (0..n_steps).rev() {
                path[step_idx] = cur_idx;
                cur_idx = backptr[step_idx * n_states + cur_idx] as usize;
            }

            // Output-layer missing-gap resolution (keeps DP neutral 0.0 emission
            // and uniform switch_penalty intact while distinguishing target
            // missingness from chosen-parent missingness):
            //   - Resolve each zero-comparable run via a sub-interval worklist:
            //     when a qualified alternative parent (`alt_comp >= 4 && alt_score > 0`)
            //     wins a sub-interval, assign it only to its observable steps and
            //     re-queue every remaining contiguous zero-comparable sub-run so
            //     multiple sequential parents inside one gap (e.g. P1 then P2)
            //     are each recovered and adjacent long all-missing runs become
            //     Uncalled.
            //   - When no qualified alternative exists, bridge short target-missing
            //     gaps and short same-parent-flanked isolated founder-missing gaps,
            //     while masking long unobservable gaps (>= 10 markers AND >= 0.5 cM)
            //     to Uncalled.
            const LONG_MISSING_GAP_MIN_MARKERS: usize = 10;
            const LONG_MISSING_GAP_MIN_CM: f64 = 0.5;
            const ALT_EVIDENCE_MIN_MARKERS: u32 = 4;
            let mut called_path: Vec<Option<usize>> = (0..n_steps)
                .map(|step_idx| {
                    let st = path[step_idx];
                    if st < k {
                        Some(st)
                    } else {
                        None
                    }
                })
                .collect();
            let mut worklist: Vec<(usize, usize, usize)> = Vec::new();
            let mut gap_step = 0usize;
            while gap_step < n_steps {
                let Some(st) = called_path[gap_step] else {
                    gap_step += 1;
                    continue;
                };
                if step_comparable[gap_step * k + st] > 0 {
                    gap_step += 1;
                    continue;
                }
                let mut gap_end = gap_step + 1;
                while gap_end < n_steps
                    && called_path[gap_end] == Some(st)
                    && step_comparable[gap_end * k + st] == 0
                {
                    gap_end += 1;
                }
                worklist.push((gap_step, gap_end, st));
                while let Some((sub_start, sub_end, cur_st)) = worklist.pop() {
                    if sub_start >= sub_end {
                        continue;
                    }
                    let m_lo = step_ranges[sub_start].0;
                    let m_hi = step_ranges[sub_end - 1].1;
                    let gap_markers = m_hi.saturating_sub(m_lo);
                    let gap_cm = if m_hi > m_lo {
                        (cm_positions[m_hi - 1] - cm_positions[m_lo]).max(0.0)
                    } else {
                        0.0
                    };
                    let is_long_gap = gap_markers >= LONG_MISSING_GAP_MIN_MARKERS
                        && gap_cm >= LONG_MISSING_GAP_MIN_CM;

                    // Filter qualified alternatives (`alt_comp >= ALT_EVIDENCE_MIN_MARKERS && alt_score > 0`)
                    // using raw comparable marker counts and raw scores so gap reassignment
                    // and boundary edges uniformly enforce the raw marker floor.
                    let mut best_qual_idx: Option<usize> = None;
                    let mut best_qual_score = f64::NEG_INFINITY;
                    let mut best_qual_comp: u32 = 0;
                    for ci in 0..k {
                        if ci == cur_st {
                            continue;
                        }
                        let mut alt_comp = 0u32;
                        let mut alt_score = 0.0f64;
                        for s_idx in sub_start..sub_end {
                            let off = s_idx * k + ci;
                            let comp = step_comparable[off];
                            if comp > 0 {
                                alt_comp = alt_comp.saturating_add(comp);
                                alt_score += step_scores[off] as f64;
                            }
                        }
                        if alt_comp >= ALT_EVIDENCE_MIN_MARKERS
                            && alt_score > 0.0
                            && (alt_score > best_qual_score
                                || (alt_score == best_qual_score && alt_comp > best_qual_comp))
                        {
                            best_qual_score = alt_score;
                            best_qual_comp = alt_comp;
                            best_qual_idx = Some(ci);
                        }
                    }

                    if let Some(alt_ci) = best_qual_idx {
                        for s_idx in sub_start..sub_end {
                            let off = s_idx * k + alt_ci;
                            if step_comparable[off] > 0 && step_scores[off] >= 0.0 {
                                called_path[s_idx] = Some(alt_ci);
                            }
                        }
                        let mut rem_step = sub_start;
                        while rem_step < sub_end {
                            let off = rem_step * k + alt_ci;
                            if step_comparable[off] > 0 && step_scores[off] >= 0.0 {
                                rem_step += 1;
                                continue;
                            }
                            let mut rem_end = rem_step + 1;
                            while rem_end < sub_end {
                                let roff = rem_end * k + alt_ci;
                                if step_comparable[roff] > 0 && step_scores[roff] >= 0.0 {
                                    break;
                                }
                                rem_end += 1;
                            }
                            // If another candidate has qualified positive evidence across this
                            // non-supporting (missing or mismatched) sub-run of `alt_ci`,
                            // re-queue the sub-run for independent evaluation so local
                            // contradictions are not overwritten by `alt_ci`'s overall score.
                            let mut has_qual_other = false;
                            for other_ci in 0..k {
                                if other_ci == alt_ci || other_ci == cur_st {
                                    continue;
                                }
                                let mut o_comp = 0u32;
                                let mut o_score = 0.0f64;
                                for s_idx in rem_step..rem_end {
                                    let o_off = s_idx * k + other_ci;
                                    let c = step_comparable[o_off];
                                    if c > 0 {
                                        o_comp = o_comp.saturating_add(c);
                                        o_score += step_scores[o_off] as f64;
                                    }
                                }
                                if o_comp >= ALT_EVIDENCE_MIN_MARKERS && o_score > 0.0 {
                                    has_qual_other = true;
                                    break;
                                }
                            }
                            if has_qual_other {
                                worklist.push((rem_step, rem_end, alt_ci));
                            } else {
                                for s_idx in rem_step..rem_end {
                                    if step_comparable[s_idx * k + alt_ci] > 0 {
                                        called_path[s_idx] = Some(alt_ci);
                                    }
                                }
                                let mut z_step = rem_step;
                                while z_step < rem_end {
                                    if step_comparable[z_step * k + alt_ci] > 0 {
                                        z_step += 1;
                                        continue;
                                    }
                                    let mut z_end = z_step + 1;
                                    while z_end < rem_end
                                        && step_comparable[z_end * k + alt_ci] == 0
                                    {
                                        z_end += 1;
                                    }
                                    let internal_to_alt = z_step > sub_start
                                        && z_end < sub_end
                                        && called_path[z_step - 1] == Some(alt_ci)
                                        && called_path[z_end] == Some(alt_ci);
                                    if internal_to_alt {
                                        for slot in &mut called_path[z_step..z_end] {
                                            *slot = Some(alt_ci);
                                        }
                                        worklist.push((z_step, z_end, alt_ci));
                                    } else {
                                        worklist.push((z_step, z_end, cur_st));
                                    }
                                    z_step = z_end;
                                }
                            }
                            rem_step = rem_end;
                        }
                    } else if is_long_gap {
                        for slot in &mut called_path[sub_start..sub_end] {
                            *slot = None;
                        }
                    }
                }
                gap_step = gap_end;
            }

            // Enforce the raw comparable marker floor (`ALT_EVIDENCE_MIN_MARKERS = 4`)
            // on any DP-emitted segment that relied on missingness compensation
            // (`run_comp < run_max_comp`): a candidate with < 4 raw comparable markers
            // across a run cannot insert a spurious island solely by accumulating
            // `FOUNDER_MISSING_CREDIT` over a local target error burst.
            loop {
                let mut changed = false;
                let mut run_start = 0usize;
                while run_start < n_steps {
                    let Some(c_idx) = called_path[run_start] else {
                        run_start += 1;
                        continue;
                    };
                    let (parent_sample, parent_hap) = ref_layout.hap_to_sample[candidates[c_idx]];
                    let mut run_end = run_start + 1;
                    while run_end < n_steps {
                        let Some(next_c) = called_path[run_end] else {
                            break;
                        };
                        let (np, nh) = ref_layout.hap_to_sample[candidates[next_c]];
                        if np != parent_sample || nh != parent_hap {
                            break;
                        }
                        run_end += 1;
                    }
                    let mut run_comp: u32 = 0;
                    let mut run_max_comp: u32 = 0;
                    let mut run_info = 0.0f64;
                    let mut run_max_info = 0.0f64;
                    let mut run_raw_score = 0.0f64;
                    for s_idx in run_start..run_end {
                        let chosen = called_path[s_idx].unwrap_or(c_idx);
                        let off = s_idx * k + chosen;
                        run_comp = run_comp.saturating_add(step_comparable[off]);
                        run_max_comp = run_max_comp.saturating_add(step_max_comps[s_idx]);
                        run_info += step_infos[off] as f64;
                        run_max_info += step_max_infos[s_idx] as f64;
                        run_raw_score += step_scores[off] as f64;
                    }
                    if run_comp < ALT_EVIDENCE_MIN_MARKERS
                        && (run_comp < run_max_comp || run_info + 1e-4 < run_max_info)
                    {
                        let left_c = if run_start > 0 {
                            called_path[run_start - 1]
                        } else {
                            None
                        };
                        let right_c = if run_end < n_steps {
                            called_path[run_end]
                        } else {
                            None
                        };
                        let raw_for_cand = |cand: usize| -> f64 {
                            let mut sum = 0.0f64;
                            for s_idx in run_start..run_end {
                                sum += step_scores[s_idx * k + cand] as f64;
                            }
                            sum
                        };
                        let sw = switch_penalty as f64;
                        let (flank, stands_without_credit) = match (left_c, right_c) {
                            (Some(lc), Some(rc))
                                if ref_layout.hap_to_sample[candidates[lc]]
                                    == ref_layout.hap_to_sample[candidates[rc]] =>
                            {
                                let flank_raw = raw_for_cand(lc);
                                let uncalled_val = -2.0 * sw;
                                let best_alt = flank_raw.max(uncalled_val);
                                let fallback = if uncalled_val > flank_raw {
                                    None
                                } else {
                                    Some(lc)
                                };
                                (
                                    fallback,
                                    run_raw_score > 0.0 && (run_raw_score - 2.0 * sw) > best_alt,
                                )
                            }
                            (Some(lc), Some(rc)) => {
                                let best_nb_raw = raw_for_cand(lc).max(raw_for_cand(rc));
                                (
                                    None,
                                    run_raw_score > 0.0 && (run_raw_score - sw) > best_nb_raw,
                                )
                            }
                            (Some(nb), None) | (None, Some(nb)) => {
                                let nb_raw = raw_for_cand(nb);
                                (None, run_raw_score > 0.0 && (run_raw_score - sw) > nb_raw)
                            }
                            (None, None) => (None, run_raw_score > 0.0),
                        };
                        if !stands_without_credit {
                            for slot in &mut called_path[run_start..run_end] {
                                *slot = flank;
                            }
                            changed = true;
                        }
                    }
                    run_start = run_end;
                }
                if !changed {
                    break;
                }
            }

            // Collapse consecutive CALLED steps with the same `(parent_sample, parent_hap)`
            // track into proto-segments so intra-parent haplotype switches split cleanly,
            // then apply out-of-panel mosaic/excess-mismatch rejection and sample-calibrated
            // IBD-fit confidence for multi-parent frequency-model panels.
            struct ProtoSeg {
                start_step: usize,
                end_step: usize,
                c_idx: usize,
                ref_col: usize,
                parent_sample: u32,
                parent_hap: u8,
                seg_comp: u32,
                seg_mis: u32,
                mis_clusters: u32,
                mis_span_cm: f64,
                start_cm: f64,
                end_cm: f64,
                len_cm: f64,
                best_seg_score: f64,
                second_best_score: f64,
                raw_conf: f64,
                ambiguous: u16,
                div_uncalled_left: bool,
                div_uncalled_right: bool,
                interior_mosaic_ibs: bool,
                rejected: bool,
            }

            let mut proto_segs: Vec<ProtoSeg> = Vec::new();
            let mut seg_start_step = 0usize;
            while seg_start_step < n_steps {
                let Some(c_idx) = called_path[seg_start_step] else {
                    seg_start_step += 1;
                    continue;
                };
                let ref_col = candidates[c_idx];
                let (parent_sample, parent_hap) = ref_layout.hap_to_sample[ref_col];
                let mut seg_end_step = seg_start_step + 1;
                while seg_end_step < n_steps {
                    let Some(next_c_idx) = called_path[seg_end_step] else {
                        break;
                    };
                    let next_ref = candidates[next_c_idx];
                    let (next_parent, next_parent_hap) = ref_layout.hap_to_sample[next_ref];
                    if next_parent != parent_sample || next_parent_hap != parent_hap {
                        break;
                    }
                    seg_end_step += 1;
                }

                let mut seg_comparable: u32 = 0;
                let mut cand_totals = vec![0.0f64; k];
                let mut cand_comp = vec![0u32; k];
                for s_idx in seg_start_step..seg_end_step {
                    let chosen = called_path[s_idx].unwrap_or(c_idx);
                    let row_off = s_idx * k;
                    seg_comparable =
                        seg_comparable.saturating_add(step_comparable[row_off + chosen]);
                    for ci in 0..k {
                        cand_totals[ci] += step_scores[row_off + ci] as f64;
                        cand_comp[ci] = cand_comp[ci].saturating_add(step_comparable[row_off + ci]);
                    }
                }
                if seg_comparable == 0 {
                    seg_start_step = seg_end_step;
                    continue;
                }

                let best_seg_score = cand_totals[c_idx];
                let mut second_best_score = f64::NEG_INFINITY;
                let mut ambiguous = 0u16;
                let mut exp_sum = 0.0f64;
                for ci in 0..k {
                    if cand_comp[ci] == 0 {
                        continue;
                    }
                    let score = cand_totals[ci];
                    if ci != c_idx && score > second_best_score {
                        second_best_score = score;
                    }
                    if (best_seg_score - score) <= 0.5 {
                        ambiguous = ambiguous.saturating_add(1);
                    }
                    let delta = (score - best_seg_score).max(-40.0);
                    exp_sum += delta.exp();
                }
                // Include the `Uncalled` background hypothesis (score = 0.0) in the
                // posterior denominator so poorly-fitting out-of-panel segments with low
                // or negative raw LLR cannot report inflated `confidence = 1.0`.
                let uncalled_delta = (0.0 - best_seg_score).clamp(-40.0, 40.0);
                exp_sum += uncalled_delta.exp();
                let raw_conf = if exp_sum > 0.0 {
                    (1.0 / exp_sum).clamp(0.0, 1.0)
                } else {
                    1.0
                };

                let m_start = step_ranges[seg_start_step].0;
                let m_end = step_ranges[seg_end_step - 1].1;
                let (seg_mis, mis_clusters, mis_span_cm) = hap_major.mismatch_dispersion(
                    tgt_col,
                    ref_col,
                    m_start,
                    m_end,
                    cm_positions,
                    0.5,
                );
                let start_cm = cm_positions[m_start];
                let end_cm = cm_positions[m_end - 1];
                let len_cm = (end_cm - start_cm).max(0.0);

                proto_segs.push(ProtoSeg {
                    start_step: seg_start_step,
                    end_step: seg_end_step,
                    c_idx,
                    ref_col,
                    parent_sample,
                    parent_hap,
                    seg_comp: seg_comparable,
                    seg_mis,
                    mis_clusters,
                    mis_span_cm,
                    start_cm,
                    end_cm,
                    len_cm,
                    best_seg_score,
                    second_best_score,
                    raw_conf,
                    ambiguous: ambiguous.max(1),
                    div_uncalled_left: false,
                    div_uncalled_right: false,
                    interior_mosaic_ibs: false,
                    rejected: false,
                });
                seg_start_step = seg_end_step;
            }

            // Out-of-panel mosaic/excess-mismatch rejection & sample-baseline confidence
            // ranking adjustment (active only when the multi-parent frequency model is active).
            let mut baseline_err = 0.003f64;
            if ref_weights.is_some() && !proto_segs.is_empty() {
                // Mark which proto-segments border a *divergence* Uncalled gap (where the
                // target had observed markers yet no panel founder beat Uncalled).
                for i in 0..proto_segs.len() {
                    let left_gap_start = if i == 0 {
                        0
                    } else {
                        proto_segs[i - 1].end_step
                    };
                    let left_gap_end = proto_segs[i].start_step;
                    if left_gap_start < left_gap_end {
                        proto_segs[i].div_uncalled_left =
                            (left_gap_start..left_gap_end).any(|s| step_max_comps[s] > 0);
                    }
                    let right_gap_start = proto_segs[i].end_step;
                    let right_gap_end = if i + 1 < proto_segs.len() {
                        proto_segs[i + 1].start_step
                    } else {
                        n_steps
                    };
                    if right_gap_start < right_gap_end {
                        proto_segs[i].div_uncalled_right =
                            (right_gap_start..right_gap_end).any(|s| step_max_comps[s] > 0);
                    }
                }

                const MOSAIC_SEG_MAX_CM: f64 = 6.0;
                const MOSAIC_GAP_MAX_CM: f64 = 6.0;

                // Estimate this target haplotype's own clean baseline mismatch rate from
                // the lower quartile (25th percentile, marker-weighted) of substantial
                // segments (`seg_comp >= 50`). Uniformly noisy samples automatically get
                // a higher `baseline_err` so they are not falsely rejected.
                let mut substantial: Vec<(f64, u32)> = proto_segs
                    .iter()
                    .filter(|s| s.seg_comp >= 50)
                    .map(|s| (s.seg_mis as f64 / s.seg_comp as f64, s.seg_comp))
                    .collect();
                if !substantial.is_empty() {
                    substantial.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let total_sub_comp: u64 = substantial.iter().map(|&(_, c)| c as u64).sum();
                    let q25_target = total_sub_comp / 4;
                    let mut acc = 0u64;
                    let mut q25_rate = substantial[0].0;
                    for &(rate, comp) in &substantial {
                        acc += comp as u64;
                        if acc >= q25_target {
                            q25_rate = rate;
                            break;
                        }
                    }
                    baseline_err = q25_rate.clamp(0.003, 0.05);
                }

                // Rule B1: Reject contiguous rapid-switching mosaic runs whose full
                // genetic span (including any short embedded divergence Uncalled gaps
                // `<= 6.0 cM`) exhibits anomalously high switch density and excess
                // mismatch over `baseline_err`. Only *internal* short divergence gaps
                // count toward `has_internal_div_uncalled` (never outer gaps outside the
                // run). Every segment inside a run is protected independently when it has
                // `<= 2` mismatches, `raw_conf >= 0.99`, and either uncontested positive
                // evidence over Uncalled (`second_best <= 0 && best >= max(0.5 * sw, 6)`) or
                // a decisive lead over a positive runner-up (`best - second_best >= sw + 1`).
                let sw_f64 = switch_penalty as f64;
                let uncontested_floor = (0.5 * sw_f64).max(6.0);
                let contested_margin = sw_f64 + 1.0;
                let is_strong_single_seg = |seg: &ProtoSeg| -> bool {
                    let margin_over_alt = seg.best_seg_score - seg.second_best_score.max(0.0);
                    seg.seg_mis <= 2
                        && seg.raw_conf >= 0.99
                        && if seg.second_best_score <= 0.0 {
                            seg.best_seg_score >= uncontested_floor
                        } else {
                            margin_over_alt >= contested_margin
                        }
                };
                let mut idx = 0usize;
                while idx < proto_segs.len() {
                    if proto_segs[idx].len_cm > MOSAIC_SEG_MAX_CM {
                        idx += 1;
                        continue;
                    }
                    let mut end = idx + 1;
                    while end < proto_segs.len() && proto_segs[end].len_cm <= MOSAIC_SEG_MAX_CM {
                        let direct_adj = proto_segs[end - 1].end_step == proto_segs[end].start_step;
                        let gap_cm =
                            (proto_segs[end].start_cm - proto_segs[end - 1].end_cm).max(0.0);
                        let short_div_gap =
                            proto_segs[end - 1].div_uncalled_right && gap_cm <= MOSAIC_GAP_MAX_CM;
                        if !direct_adj && !short_div_gap {
                            break;
                        }
                        end += 1;
                    }
                    let run_len = end - idx;
                    if run_len >= 3 {
                        let mut run_comp = 0u32;
                        let mut run_mis = 0u32;
                        let mut run_mis_clusters = 0u32;
                        let mut weak_segs = 0usize;
                        for seg in &proto_segs[idx..end] {
                            run_comp = run_comp.saturating_add(seg.seg_comp);
                            run_mis = run_mis.saturating_add(seg.seg_mis);
                            run_mis_clusters = run_mis_clusters.saturating_add(seg.mis_clusters);
                            if !is_strong_single_seg(seg) {
                                weak_segs += 1;
                            }
                        }
                        let has_internal_div_uncalled =
                            (idx..end - 1).any(|j| proto_segs[j].div_uncalled_right);
                        let span_cm =
                            (proto_segs[end - 1].end_cm - proto_segs[idx].start_cm).max(0.0);
                        let avg_span_cm = span_cm / run_len as f64;
                        let run_mis_rate = run_mis as f64 / run_comp.max(1) as f64;
                        let density_qualifies = (run_len >= 4
                            && weak_segs >= 3
                            && avg_span_cm < 2.5)
                            || (has_internal_div_uncalled && weak_segs >= 2 && avg_span_cm < 3.0);
                        if density_qualifies
                            && run_mis >= 3
                            && run_mis_clusters >= 3
                            && run_mis_rate > (baseline_err * 2.0).max(0.006)
                            && one_sided_binomial_deviance(
                                run_mis as f64,
                                run_comp as f64,
                                baseline_err,
                            ) >= 8.0
                        {
                            for seg in &mut proto_segs[idx..end] {
                                if !is_strong_single_seg(seg) {
                                    seg.rejected = true;
                                }
                            }
                            for pos in (idx + 1)..(end - 1) {
                                if !proto_segs[pos].rejected
                                    && proto_segs[idx..pos].iter().any(|s| s.rejected)
                                    && proto_segs[pos + 1..end].iter().any(|s| s.rejected)
                                {
                                    proto_segs[pos].interior_mosaic_ibs = true;
                                }
                            }
                        }
                    }
                    idx = end;
                }

                // Rule B3: Reject standalone segments with *genetically dispersed* excess
                // mismatch (`>= 3` marker-coordinate mismatch clusters separated by
                // `>= 0.5 cM`, spanning `>= 1.0 cM` and `>= 35%` of segment genetic length).
                // Fully invariant to `ancestry_step_cm` and preserves contiguous noise blocks.
                for seg in &mut proto_segs {
                    if seg.rejected || seg.seg_comp < 60 || seg.seg_mis < 5 {
                        continue;
                    }
                    let mis_rate = seg.seg_mis as f64 / seg.seg_comp as f64;
                    let span_frac = if seg.len_cm > 1e-6 {
                        seg.mis_span_cm / seg.len_cm
                    } else {
                        0.0
                    };
                    if mis_rate >= (baseline_err * 3.0).max(0.015)
                        && seg.mis_clusters >= 3
                        && seg.mis_span_cm >= 1.0
                        && span_frac >= 0.35
                        && one_sided_binomial_deviance(
                            seg.seg_mis as f64,
                            seg.seg_comp as f64,
                            baseline_err,
                        ) >= 16.0
                    {
                        seg.rejected = true;
                    }
                }

                // Rule B2: Single non-cascading pass rejecting short-to-medium stitched
                // bridges bordering *original* DP divergence Uncalled gaps (or directly
                // adjacent B1/B3 rejections — never feeding B2's own rejections forward
                // and never propagating across pure-missing gaps). Protects unambiguous
                // strong single-founder segments on single-sided edges.
                let pre_b2_rejected: Vec<bool> = proto_segs.iter().map(|s| s.rejected).collect();
                for i in 0..proto_segs.len() {
                    if proto_segs[i].rejected {
                        continue;
                    }
                    let left_adj_b13 = i > 0
                        && pre_b2_rejected[i - 1]
                        && proto_segs[i - 1].end_step == proto_segs[i].start_step;
                    let right_adj_b13 = i + 1 < proto_segs.len()
                        && pre_b2_rejected[i + 1]
                        && proto_segs[i].end_step == proto_segs[i + 1].start_step;
                    let left_div = proto_segs[i].div_uncalled_left || left_adj_b13;
                    let right_div = proto_segs[i].div_uncalled_right || right_adj_b13;
                    if !left_div && !right_div {
                        continue;
                    }
                    proto_segs[i].div_uncalled_left = left_div;
                    proto_segs[i].div_uncalled_right = right_div;
                    let seg = &proto_segs[i];
                    if seg.len_cm > 8.0
                        || seg.seg_mis < 2
                        || seg.mis_clusters < 2
                        || is_strong_single_seg(seg)
                    {
                        continue;
                    }
                    let mis_rate = seg.seg_mis as f64 / seg.seg_comp.max(1) as f64;
                    let dev = one_sided_binomial_deviance(
                        seg.seg_mis as f64,
                        seg.seg_comp as f64,
                        baseline_err,
                    );
                    let sandwiched_reject = left_div
                        && right_div
                        && mis_rate > (baseline_err * 1.5).max(0.006)
                        && dev >= 4.0;
                    let edge_reject = mis_rate > (baseline_err * 2.0).max(0.008) && dev >= 6.0;
                    if sandwiched_reject || edge_reject {
                        proto_segs[i].rejected = true;
                    }
                }

                // Clear rejected segments from `called_path` so surviving neighbors stop
                // cleanly at their own last marker rather than bleeding into Uncalled gaps.
                for seg in &proto_segs {
                    if seg.rejected {
                        for slot in &mut called_path[seg.start_step..seg.end_step] {
                            *slot = None;
                        }
                    }
                }
            }

            let mut out = Vec::with_capacity(proto_segs.len());
            for seg in &proto_segs {
                if seg.rejected {
                    continue;
                }
                // Composite segment confidence / ranking score: combines competitor +
                // Uncalled posterior (`raw_conf`) with an LD-damped excess-mismatch
                // penalty (capped at 10 effective SNPs/cM and weighted by genetically
                // separated mismatch clusters so dense LD bursts do not over-penalize
                // true in-panel segments).
                let confidence = if ref_weights.is_some() {
                    let n_eff_cap = (seg.len_cm.max(0.5) * 10.0).max(20.0);
                    let (m_eff, n_eff) = if (seg.seg_comp as f64) > n_eff_cap {
                        let scale = n_eff_cap / seg.seg_comp as f64;
                        (seg.seg_mis as f64 * scale, n_eff_cap)
                    } else {
                        (seg.seg_mis as f64, seg.seg_comp as f64)
                    };
                    let dev = one_sided_binomial_deviance(m_eff, n_eff, baseline_err);
                    let ibd_fit_factor = if dev <= 0.0 {
                        1.0
                    } else {
                        let spread = (seg.mis_clusters as f64 / 4.0).clamp(0.15, 1.0);
                        let eff_ll_penalty = 0.5 * dev * spread;
                        let prior_log_odds = if seg.div_uncalled_left
                            || seg.div_uncalled_right
                            || seg.len_cm < 3.0
                        {
                            2.0
                        } else {
                            4.5
                        };
                        1.0 / (1.0 + (eff_ll_penalty - prior_log_odds).clamp(-40.0, 40.0).exp())
                    };
                    let structural_discount = if seg.interior_mosaic_ibs {
                        0.75
                    } else if seg.div_uncalled_left && seg.div_uncalled_right && seg.len_cm < 3.0 {
                        0.80
                    } else if (seg.div_uncalled_left || seg.div_uncalled_right) && seg.len_cm < 3.0
                    {
                        0.85
                    } else {
                        1.0
                    };
                    (seg.raw_conf * ibd_fit_factor * structural_discount).clamp(0.0, 1.0) as f32
                } else {
                    seg.raw_conf as f32
                };

                let m_start = step_ranges[seg.start_step].0;
                let m_end = step_ranges[seg.end_step - 1].1;
                // Share the exact recombination boundary with the next segment ONLY
                // when immediately followed by another called segment; otherwise stop
                // at this segment's own last marker so uncalled gaps remain uncalled.
                let followed_by_called =
                    seg.end_step < n_steps && called_path[seg.end_step].is_some();

                // Post-Viterbi marker-level short-introgression refinement: recovers
                // compact 250-500 kb introgressions diluted by 0.05 cM step boundaries or
                // slightly below `2 * switch_penalty` (e.g. 3-4 diagnostic SNPs) inside
                // clean multi-parent background segments, without lowering global
                // `switch_penalty`. Disabled when `ancestry_step_max_markers > 0` or on
                // noisy segments/samples.
                let mut splices: Vec<(usize, usize, u32, u8, f32, u16, u32)> = Vec::new();
                if let Some(ref weights) = ref_weights {
                    let seg_mis_rate = seg.seg_mis as f64 / seg.seg_comp.max(1) as f64;
                    if config.ancestry_step_max_markers == 0
                        && baseline_err <= 0.015
                        && seg.seg_mis >= 3
                        && seg.seg_comp >= 50
                        && seg_mis_rate <= 0.025
                    {
                        const CLUSTER_MAX_GAP_BP: u64 = 400_000;
                        const CLUSTER_MAX_GAP_CM: f64 = 0.6;
                        const CLUSTER_MAX_SPAN_BP: u64 = 750_000;
                        const CLUSTER_MAX_SPAN_CM: f64 = 1.2;
                        let mis_markers = hap_major.collect_mismatch_markers(
                            tgt_col,
                            seg.ref_col,
                            m_start,
                            m_end,
                        );
                        let mut c_start = 0usize;
                        let mut last_splice_end = m_start;
                        while c_start < mis_markers.len() {
                            let mut c_end = c_start + 1;
                            while c_end < mis_markers.len() {
                                let prev_m = mis_markers[c_end - 1];
                                let next_m = mis_markers[c_end];
                                let gap_bp = positions_bp[next_m].saturating_sub(positions_bp[prev_m]);
                                let gap_cm = (cm_positions[next_m] - cm_positions[prev_m]).max(0.0);
                                if gap_bp <= CLUSTER_MAX_GAP_BP && gap_cm <= CLUSTER_MAX_GAP_CM {
                                    c_end += 1;
                                } else {
                                    break;
                                }
                            }
                            let cluster = &mis_markers[c_start..c_end];
                            if cluster.len() >= 3 {
                                let mut sub_windows: Vec<(usize, usize)> = Vec::with_capacity(3);
                                sub_windows.push((0, cluster.len() - 1));
                                if cluster.len() >= 4 {
                                    sub_windows.push((1, cluster.len() - 1));
                                    sub_windows.push((0, cluster.len() - 2));
                                }
                                for (lo_i, hi_i) in sub_windows {
                                    let m_lo = cluster[lo_i];
                                    let m_hi = cluster[hi_i];
                                    if m_lo < last_splice_end {
                                        continue;
                                    }
                                    let span_bp = positions_bp[m_hi].saturating_sub(positions_bp[m_lo]);
                                    let span_cm = (cm_positions[m_hi] - cm_positions[m_lo]).max(0.0);
                                    if span_bp > CLUSTER_MAX_SPAN_BP || span_cm > CLUSTER_MAX_SPAN_CM {
                                        continue;
                                    }
                                    let (_, _, bg_raw) = hap_major.count_mismatches_weighted(
                                        tgt_col,
                                        seg.ref_col,
                                        m_lo,
                                        m_hi + 1,
                                        weights,
                                        match_reward,
                                        flat_mismatch_delta,
                                        freq_mismatch_delta,
                                    );
                                    let mut best_alt: Option<(usize, u32, f32)> = None;
                                    let mut zero_mis_parents: Vec<u32> = Vec::new();
                                    for (alt_ci, &alt_ref_col) in candidates.iter().enumerate() {
                                        if alt_ci == seg.c_idx {
                                            continue;
                                        }
                                        let (alt_comp, alt_mis) = hap_major.count_mismatches(
                                            tgt_col,
                                            alt_ref_col,
                                            m_lo,
                                            m_hi + 1,
                                        );
                                        if alt_mis != 0 || alt_comp == 0 {
                                            continue;
                                        }
                                        let (alt_parent, _) = ref_layout.hap_to_sample[alt_ref_col];
                                        zero_mis_parents.push(alt_parent);
                                        let obs_diag = cluster[lo_i..=hi_i]
                                            .iter()
                                            .filter(|&&dm| hap_major.is_observed(alt_ref_col, dm))
                                            .count() as u32;
                                        if obs_diag < 3 {
                                            continue;
                                        }
                                        let min_comp = if obs_diag == 3 { 5 } else { 4 };
                                        if alt_comp < min_comp {
                                            continue;
                                        }
                                        let (_, _, alt_raw) = hap_major.count_mismatches_weighted(
                                            tgt_col,
                                            alt_ref_col,
                                            m_lo,
                                            m_hi + 1,
                                            weights,
                                            match_reward,
                                            flat_mismatch_delta,
                                            freq_mismatch_delta,
                                        );
                                        let min_alt_raw = if obs_diag == 3 { 3.5f32 } else { 3.0f32 };
                                        let min_ll_gain = 16.0f32
                                            * (freq_mismatch_delta / default_freq_mismatch_delta);
                                        let ll_gain = alt_raw - bg_raw;
                                        if alt_raw >= min_alt_raw && ll_gain >= min_ll_gain {
                                            if best_alt
                                                .map(|(_, bc, br)| {
                                                    alt_raw > br || (alt_raw == br && alt_comp > bc)
                                                })
                                                .unwrap_or(true)
                                            {
                                                best_alt = Some((alt_ci, alt_comp, alt_raw));
                                            }
                                        }
                                    }
                                    let Some((best_ci, _, _)) = best_alt else {
                                        continue;
                                    };
                                    zero_mis_parents.sort_unstable();
                                    zero_mis_parents.dedup();
                                    if zero_mis_parents.len() > 3 {
                                        continue;
                                    }
                                    let alt_ref_col = candidates[best_ci];
                                    let prev_alt_mis = hap_major.find_nearest_mismatch_left(
                                        tgt_col,
                                        alt_ref_col,
                                        last_splice_end,
                                        m_lo,
                                    );
                                    let next_alt_mis = hap_major.find_nearest_mismatch_right(
                                        tgt_col,
                                        alt_ref_col,
                                        m_hi,
                                        m_end,
                                    );
                                    let splice_start = prev_alt_mis
                                        .map(|p| {
                                            let bp_l = positions_bp[p];
                                            let bp_r = positions_bp[m_lo];
                                            let mid_bp = bp_l + (bp_r.saturating_sub(bp_l)) / 2;
                                            let lo = p + 1;
                                            lo + positions_bp[lo..=m_lo]
                                                .partition_point(|&bp| bp < mid_bp)
                                        })
                                        .unwrap_or(m_lo)
                                        .max(last_splice_end)
                                        .min(m_lo);
                                    let splice_end = next_alt_mis
                                        .map(|n| {
                                            let bp_l = positions_bp[m_hi];
                                            let bp_r = positions_bp[n];
                                            let mid_bp = bp_l + (bp_r.saturating_sub(bp_l)) / 2;
                                            let lo = m_hi + 1;
                                            lo + positions_bp[lo..n]
                                                .partition_point(|&bp| bp <= mid_bp)
                                        })
                                        .unwrap_or(m_hi + 1)
                                        .max(m_hi + 1)
                                        .min(m_end);
                                    if splice_start < splice_end {
                                        let (alt_parent, alt_hap) =
                                            ref_layout.hap_to_sample[alt_ref_col];
                                        let (sp_comp, _, sp_best_raw) =
                                            hap_major.count_mismatches_weighted(
                                                tgt_col,
                                                alt_ref_col,
                                                splice_start,
                                                splice_end,
                                                weights,
                                                match_reward,
                                                flat_mismatch_delta,
                                                freq_mismatch_delta,
                                            );
                                        let mut sp_ambig = 0u16;
                                        let mut sp_exp_sum = 0.0f64;
                                        for &cand_ref_col in candidates {
                                            let (cc, _, cs) = hap_major.count_mismatches_weighted(
                                                tgt_col,
                                                cand_ref_col,
                                                splice_start,
                                                splice_end,
                                                weights,
                                                match_reward,
                                                flat_mismatch_delta,
                                                freq_mismatch_delta,
                                            );
                                            if cc == 0 {
                                                continue;
                                            }
                                            if (sp_best_raw as f64 - cs as f64) <= 0.5 {
                                                sp_ambig = sp_ambig.saturating_add(1);
                                            }
                                            sp_exp_sum +=
                                                ((cs as f64 - sp_best_raw as f64).max(-40.0)).exp();
                                        }
                                        sp_exp_sum +=
                                            ((0.0 - sp_best_raw as f64).clamp(-40.0, 40.0)).exp();
                                        let sp_conf = if sp_exp_sum > 0.0 {
                                            (1.0 / sp_exp_sum).clamp(0.0, 1.0) as f32
                                        } else {
                                            1.0
                                        };
                                        splices.push((
                                            splice_start,
                                            splice_end,
                                            alt_parent,
                                            alt_hap,
                                            sp_conf,
                                            sp_ambig.max(1),
                                            sp_comp,
                                        ));
                                        last_splice_end = splice_end;
                                        break;
                                    }
                                }
                            }
                            c_start = c_end;
                        }
                    }
                }

                if splices.is_empty() {
                    let start_bp = positions_bp[m_start];
                    let start_cm = cm_positions[m_start] as f32;
                    let (end_bp, end_cm) = if followed_by_called && m_end < n_markers {
                        (
                            positions_bp[m_end].saturating_sub(1).max(start_bp),
                            cm_positions[m_end] as f32,
                        )
                    } else {
                        (positions_bp[m_end - 1], cm_positions[m_end - 1] as f32)
                    };
                    out.push(AncestrySegment {
                        target_sample,
                        hap: target_hap,
                        chrom_idx,
                        start_bp,
                        end_bp,
                        start_cm,
                        end_cm,
                        parent_sample: seg.parent_sample,
                        parent_hap: seg.parent_hap,
                        confidence,
                        ambiguous_parents: seg.ambiguous,
                        n_markers: seg.seg_comp,
                    });
                } else {
                    let mut cursor = m_start;
                    for &(sp_start, sp_end, alt_p, alt_h, sp_conf, sp_ambig, sp_comp) in &splices {
                        if cursor < sp_start {
                            let (left_comp, _) =
                                hap_major.count_mismatches(tgt_col, seg.ref_col, cursor, sp_start);
                            if left_comp > 0 {
                                let s_bp = positions_bp[cursor];
                                out.push(AncestrySegment {
                                    target_sample,
                                    hap: target_hap,
                                    chrom_idx,
                                    start_bp: s_bp,
                                    end_bp: positions_bp[sp_start].saturating_sub(1).max(s_bp),
                                    start_cm: cm_positions[cursor] as f32,
                                    end_cm: cm_positions[sp_start] as f32,
                                    parent_sample: seg.parent_sample,
                                    parent_hap: seg.parent_hap,
                                    confidence,
                                    ambiguous_parents: seg.ambiguous,
                                    n_markers: left_comp,
                                });
                            }
                        }
                        let sp_s_bp = positions_bp[sp_start];
                        let sp_has_right = sp_end < m_end || (followed_by_called && m_end < n_markers);
                        let (sp_e_bp, sp_e_cm) = if sp_has_right && sp_end < n_markers {
                            (
                                positions_bp[sp_end].saturating_sub(1).max(sp_s_bp),
                                cm_positions[sp_end] as f32,
                            )
                        } else {
                            (positions_bp[sp_end - 1], cm_positions[sp_end - 1] as f32)
                        };
                        out.push(AncestrySegment {
                            target_sample,
                            hap: target_hap,
                            chrom_idx,
                            start_bp: sp_s_bp,
                            end_bp: sp_e_bp,
                            start_cm: cm_positions[sp_start] as f32,
                            end_cm: sp_e_cm,
                            parent_sample: alt_p,
                            parent_hap: alt_h,
                            confidence: sp_conf,
                            ambiguous_parents: sp_ambig,
                            n_markers: sp_comp,
                        });
                        cursor = sp_end;
                    }
                    if cursor < m_end {
                        let (right_comp, _) =
                            hap_major.count_mismatches(tgt_col, seg.ref_col, cursor, m_end);
                        if right_comp > 0 {
                            let s_bp = positions_bp[cursor];
                            let (e_bp, e_cm) = if followed_by_called && m_end < n_markers {
                                (
                                    positions_bp[m_end].saturating_sub(1).max(s_bp),
                                    cm_positions[m_end] as f32,
                                )
                            } else {
                                (positions_bp[m_end - 1], cm_positions[m_end - 1] as f32)
                            };
                            out.push(AncestrySegment {
                                target_sample,
                                hap: target_hap,
                                chrom_idx,
                                start_bp: s_bp,
                                end_bp: e_bp,
                                start_cm: cm_positions[cursor] as f32,
                                end_cm: e_cm,
                                parent_sample: seg.parent_sample,
                                parent_hap: seg.parent_hap,
                                confidence,
                                ambiguous_parents: seg.ambiguous,
                                n_markers: right_comp,
                            });
                        }
                    }
                }
            }
            Ok(out)
        })
        .collect();
    let segments: Vec<AncestrySegment> = per_hap_segments?.into_iter().flatten().collect();
    // Release the transposed haplotype-major bitplane before allocating sparse bins.
    drop(hap_major);

    if is_cancelled(cancelled) {
        return Err("Interrupted".to_string());
    }

    // Step 4: Build exact Union-of-Breakpoints sparse ancestry dosage matrix.
    let sparse_bins = build_union_breakpoint_sparse_matrix(
        chrom_idx,
        target_layout,
        ref_layout.n_samples,
        &segments,
    );
    Ok((segments, sparse_bins))
}

fn build_genetic_step_ranges(cm_positions: &[f64], step_cm: f64) -> Vec<(usize, usize)> {
    let n = cm_positions.len();
    if n == 0 {
        return Vec::new();
    }
    let step = if step_cm.is_finite() && step_cm > 0.0 {
        step_cm
    } else {
        0.05
    };
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < n {
        let limit_cm = cm_positions[start] + step;
        let mut end = start + 1;
        while end < n && cm_positions[end] < limit_cm {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }
    ranges
}

/// Compress all ancestry segments on a chromosome into the exact population-wide
/// Union-of-Breakpoints sparse dosage matrix (bin-major CSC-like layout) using
/// lock-free block-partitioned parallel sort-and-coalesce on the active Rayon pool.
pub fn build_union_breakpoint_sparse_matrix(
    chrom_idx: u16,
    target_layout: &SampleHaplotypeLayout,
    n_parents: usize,
    segments: &[AncestrySegment],
) -> ChromosomeAncestryBinsSparse {
    build_union_breakpoint_sparse_matrix_cancellable(
        chrom_idx,
        target_layout,
        n_parents,
        segments,
        None,
    )
    .unwrap_or_default()
}

/// Cancellable variant of `build_union_breakpoint_sparse_matrix`.
pub fn build_union_breakpoint_sparse_matrix_cancellable(
    chrom_idx: u16,
    target_layout: &SampleHaplotypeLayout,
    n_parents: usize,
    segments: &[AncestrySegment],
    cancelled: Option<&AtomicBool>,
) -> Result<ChromosomeAncestryBinsSparse, String> {
    if is_cancelled(cancelled) {
        return Err("Interrupted".to_owned());
    }
    if segments.is_empty() || target_layout.n_samples == 0 || n_parents == 0 {
        return Ok(ChromosomeAncestryBinsSparse {
            chrom_idx,
            n_target_samples: target_layout.n_samples,
            n_parents,
            ..ChromosomeAncestryBinsSparse::default()
        });
    }

    // Collect unique sorted breakpoint coordinates `(bp, cm)` in parallel.
    // Because adjacent segments share `left.end_bp + 1 == right.start_bp`,
    // recombination boundaries deduplicate to a single breakpoint with zero empty bins.
    let mut bp_to_cm: Vec<(u64, f32)> = Vec::with_capacity(segments.len() * 2);
    for seg in segments {
        bp_to_cm.push((seg.start_bp, seg.start_cm));
        bp_to_cm.push((seg.end_bp.saturating_add(1), seg.end_cm));
    }
    bp_to_cm.par_sort_by_key(|&(bp, _)| bp);
    bp_to_cm.dedup_by_key(|(bp, _)| *bp);
    if is_cancelled(cancelled) {
        return Err("Interrupted".to_owned());
    }
    if bp_to_cm.len() < 2 {
        return Ok(ChromosomeAncestryBinsSparse {
            chrom_idx,
            n_target_samples: target_layout.n_samples,
            n_parents,
            ..ChromosomeAncestryBinsSparse::default()
        });
    }

    let n_bins = bp_to_cm.len() - 1;

    // Parallel binary-search mapping of every segment onto `[b_start, b_end)`.
    let seg_ranges: Vec<(u32, u32, u32, u32, f32)> = segments
        .par_iter()
        .filter_map(|seg| {
            let b_start = bp_to_cm
                .binary_search_by_key(&seg.start_bp, |&(bp, _)| bp)
                .unwrap_or(0);
            let b_end = bp_to_cm
                .binary_search_by_key(&seg.end_bp.saturating_add(1), |&(bp, _)| bp)
                .unwrap_or(n_bins)
                .min(n_bins);
            if b_start >= b_end {
                return None;
            }
            let weight = if target_layout.is_inbred[seg.target_sample as usize] {
                1.0f32
            } else {
                0.5f32
            };
            Some((
                b_start as u32,
                b_end as u32,
                seg.target_sample,
                seg.parent_sample,
                weight,
            ))
        })
        .collect();

    // Partition bins into disjoint contiguous blocks so each Rayon worker builds
    // its bins independently with cache-local Vec sort+coalesce (zero HashMaps).
    const BIN_BLOCK: usize = 256;
    let n_blocks = n_bins.div_ceil(BIN_BLOCK);
    let mut block_counts = vec![0usize; n_blocks];
    for &(b_start, b_end, _, _, _) in &seg_ranges {
        let blk0 = (b_start as usize) / BIN_BLOCK;
        let blk1 = ((b_end as usize) - 1) / BIN_BLOCK;
        for c in &mut block_counts[blk0..=blk1] {
            *c += 1;
        }
    }
    let mut block_ptr = Vec::with_capacity(n_blocks + 1);
    let mut total_refs = 0usize;
    block_ptr.push(0usize);
    for &c in &block_counts {
        total_refs += c;
        block_ptr.push(total_refs);
    }
    let mut block_write = block_ptr[..n_blocks].to_vec();
    let mut block_segs = vec![0u32; total_refs];
    for (seg_idx, &(b_start, b_end, _, _, _)) in seg_ranges.iter().enumerate() {
        let blk0 = (b_start as usize) / BIN_BLOCK;
        let blk1 = ((b_end as usize) - 1) / BIN_BLOCK;
        let seg_u32 = seg_idx as u32;
        for blk in blk0..=blk1 {
            let pos = block_write[blk];
            block_segs[pos] = seg_u32;
            block_write[blk] = pos + 1;
        }
    }
    drop(block_counts);
    drop(block_write);

    struct BlockOut {
        starts_bp: Vec<u64>,
        ends_bp: Vec<u64>,
        starts_cm: Vec<f32>,
        ends_cm: Vec<f32>,
        bin_lens: Vec<u32>,
        sample_idx: Vec<u32>,
        parent_idx: Vec<u32>,
        dosage: Vec<f32>,
    }

    let block_outputs: Vec<BlockOut> = (0..n_blocks)
        .into_par_iter()
        .map(|blk| -> Result<BlockOut, String> {
            if is_cancelled(cancelled) {
                return Err("Interrupted".to_owned());
            }
            let b0 = blk * BIN_BLOCK;
            let b1 = (b0 + BIN_BLOCK).min(n_bins);
            let local_n = b1 - b0;
            let slice = &block_segs[block_ptr[blk]..block_ptr[blk + 1]];
            let mut local_bins: Vec<Vec<(u32, u32, f32)>> = vec![Vec::new(); local_n];
            for &seg_idx in slice {
                let (b_start, b_end, s, p, w) = seg_ranges[seg_idx as usize];
                let lo = (b_start as usize).max(b0);
                let hi = (b_end as usize).min(b1);
                for b in lo..hi {
                    local_bins[b - b0].push((s, p, w));
                }
            }

            let mut out = BlockOut {
                starts_bp: Vec::with_capacity(local_n),
                ends_bp: Vec::with_capacity(local_n),
                starts_cm: Vec::with_capacity(local_n),
                ends_cm: Vec::with_capacity(local_n),
                bin_lens: Vec::with_capacity(local_n),
                sample_idx: Vec::new(),
                parent_idx: Vec::new(),
                dosage: Vec::new(),
            };
            for (offset, mut entries) in local_bins.into_iter().enumerate() {
                if entries.is_empty() {
                    continue;
                }
                entries.sort_unstable_by_key(|&(s, p, _)| (s, p));
                let b = b0 + offset;
                out.starts_bp.push(bp_to_cm[b].0);
                out.ends_bp.push(bp_to_cm[b + 1].0.saturating_sub(1));
                out.starts_cm.push(bp_to_cm[b].1);
                out.ends_cm.push(bp_to_cm[b + 1].1);

                let before = out.sample_idx.len();
                let mut cur_s = entries[0].0;
                let mut cur_p = entries[0].1;
                let mut cur_d = entries[0].2;
                for &(s, p, w) in &entries[1..] {
                    if s == cur_s && p == cur_p {
                        cur_d += w;
                    } else {
                        out.sample_idx.push(cur_s);
                        out.parent_idx.push(cur_p);
                        out.dosage.push(cur_d);
                        cur_s = s;
                        cur_p = p;
                        cur_d = w;
                    }
                }
                out.sample_idx.push(cur_s);
                out.parent_idx.push(cur_p);
                out.dosage.push(cur_d);
                out.bin_lens.push((out.sample_idx.len() - before) as u32);
            }
            Ok(out)
        })
        .collect::<Result<Vec<BlockOut>, String>>()?;

    let total_bins: usize = block_outputs.iter().map(|b| b.starts_bp.len()).sum();
    let total_nnz: usize = block_outputs.iter().map(|b| b.sample_idx.len()).sum();
    let mut out_bin_starts_bp = Vec::with_capacity(total_bins);
    let mut out_bin_ends_bp = Vec::with_capacity(total_bins);
    let mut out_bin_starts_cm = Vec::with_capacity(total_bins);
    let mut out_bin_ends_cm = Vec::with_capacity(total_bins);
    let mut bin_ptr = Vec::with_capacity(total_bins + 1);
    let mut sample_idx = Vec::with_capacity(total_nnz);
    let mut parent_idx = Vec::with_capacity(total_nnz);
    let mut dosage = Vec::with_capacity(total_nnz);
    bin_ptr.push(0u64);
    for blk in block_outputs {
        out_bin_starts_bp.extend(blk.starts_bp);
        out_bin_ends_bp.extend(blk.ends_bp);
        out_bin_starts_cm.extend(blk.starts_cm);
        out_bin_ends_cm.extend(blk.ends_cm);
        sample_idx.extend(blk.sample_idx);
        parent_idx.extend(blk.parent_idx);
        dosage.extend(blk.dosage);
        let mut cursor = *bin_ptr.last().unwrap_or(&0);
        for len in blk.bin_lens {
            cursor += len as u64;
            bin_ptr.push(cursor);
        }
    }

    Ok(ChromosomeAncestryBinsSparse {
        chrom_idx,
        n_target_samples: target_layout.n_samples,
        n_parents,
        bin_starts_bp: out_bin_starts_bp,
        bin_ends_bp: out_bin_ends_bp,
        bin_starts_cm: out_bin_starts_cm,
        bin_ends_cm: out_bin_ends_cm,
        bin_ptr,
        sample_idx,
        parent_idx,
        dosage,
    })
}

#[cfg(test)]
mod tests {
    use super::paint_chromosome_ancestry;
    use crate::haploid::phasing::SampleHaplotypeLayout;
    use crate::haploid::types::{BitHaplotypeMatrix, HaploidIbdConfig};

    #[test]
    fn multi_parent_ancestry_recovers_dh_recombination_breakpoint_and_sparse_bins() {
        // 4 inbred maize parents + 2 DH progeny:
        //   DH 0: parent 0 on [0, 150), parent 2 on [150, 300)
        //   DH 1: parent 1 on [0, 300)
        let n_markers = 300;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 4]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 2]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let p0 = ((m * 3 + 1) % 2) as u8;
            let p1 = ((m * 5 + 2) % 3 == 0) as u8;
            let p2 = ((m * 7 + 4) % 2) as u8;
            let p3 = ((m * 11 + 1) % 5 < 2) as u8;
            joint.set_allele(m, 0, Some(p0));
            joint.set_allele(m, 1, Some(p1));
            joint.set_allele(m, 2, Some(p2));
            joint.set_allele(m, 3, Some(p3));

            let dh0 = if m < 150 { p0 } else { p2 };
            let dh1 = p1;
            joint.set_allele(m, 4, Some(dh0));
            joint.set_allele(m, 5, Some(dh1));
        }

        let cfg = HaploidIbdConfig {
            ancestry_step_cm: 0.25,
            ancestry_switch_penalty: 8.0,
            ..HaploidIbdConfig::default()
        };
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);

        let dh0_segs: Vec<_> = segs.iter().filter(|s| s.target_sample == 0).collect();
        let dh1_segs: Vec<_> = segs.iter().filter(|s| s.target_sample == 1).collect();
        assert_eq!(dh0_segs.len(), 2, "DH 0 should have 2 ancestry blocks");
        assert_eq!(dh0_segs[0].parent_sample, 0);
        assert_eq!(dh0_segs[1].parent_sample, 2);
        assert_eq!(dh0_segs[0].end_bp + 1, dh0_segs[1].start_bp);
        assert_eq!(dh1_segs.len(), 1, "DH 1 should have 1 ancestry block");
        assert_eq!(dh1_segs[0].parent_sample, 1);

        // Union of breakpoints across DH0 (1 shared boundary) + DH1 (0 breakpoints) = 2 bins.
        assert_eq!(bins.bin_starts_bp.len(), 2);
        assert_eq!(bins.dosage.len(), 4); // 2 samples x 2 bins
        assert!(bins.dosage.iter().all(|&d| (d - 1.0).abs() < 1e-6));
    }

    #[test]
    fn all_missing_target_leaves_ancestry_unassigned_with_zero_dosage() {
        let n_markers = 150;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 3]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 1]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            joint.set_allele(m, 0, Some((m % 2) as u8));
            joint.set_allele(m, 1, Some(((m + 1) % 2) as u8));
            joint.set_allele(m, 2, Some(((m / 2) % 2) as u8));
            joint.set_allele(m, 3, None); // all-missing target
        }
        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert!(
            segs.is_empty(),
            "all-missing target must produce 0 ancestry segments"
        );
        assert!(
            bins.dosage.is_empty(),
            "all-missing target must produce 0 sparse dosages"
        );
    }

    #[test]
    fn zero_evidence_parent_cannot_beat_mismatched_parent_for_ancestry_dosage() {
        // Parent 0 is all-missing (0 comparable sites), Parent 1 mismatches target
        // at all 40 markers. Parent 0 must NEVER be emitted with dosage=1.0.
        let n_markers = 40;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            joint.set_allele(m, 0, None); // all-missing parent 0
            joint.set_allele(m, 1, Some(1)); // parent 1 mismatches target everywhere
            joint.set_allele(m, 2, Some(0)); // target
        }
        let cfg = HaploidIbdConfig {
            ancestry_step_cm: 0.25,
            ..HaploidIbdConfig::default()
        };
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert!(
            segs.is_empty(),
            "zero-evidence parent must not produce ancestry segments, got {segs:?}"
        );
        assert!(bins.dosage.is_empty());
    }

    #[test]
    fn internal_missing_gap_breaks_ancestry_segment_with_zero_gap_dosage() {
        // Target matches Parent 0 on [0..40) and [60..100), with 20 consecutive
        // all-missing markers on [40..60). Must break into 2 segments and leave
        // the missing gap uncalled.
        let n_markers = 100;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            joint.set_allele(m, 0, Some(p0));
            joint.set_allele(m, 1, Some(p1));
            if (40..60).contains(&m) {
                joint.set_allele(m, 2, None);
            } else {
                joint.set_allele(m, 2, Some(p0));
            }
        }
        let cfg = HaploidIbdConfig {
            ancestry_step_cm: 0.25,
            ..HaploidIbdConfig::default()
        };
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "20-marker missing gap must break segment into 2"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 0);
        assert_eq!(segs[0].end_bp, pos[39]);
        assert_eq!(segs[1].start_bp, pos[60]);
        assert_eq!(segs[0].n_markers, 40);
        assert_eq!(segs[1].n_markers, 40);
        // Gap [pos[40]..pos[59]] must not appear in called sparse bins.
        assert_eq!(bins.bin_starts_bp.len(), 2);
        assert_eq!(bins.bin_ends_bp[0], pos[39]);
        assert_eq!(bins.bin_starts_bp[1], pos[60]);
    }

    #[test]
    fn missing_parent_cannot_suppress_observed_ancestry_in_second_half() {
        // [0..20): Parent 0 matches 20/20, Parent 1 is all-missing.
        // [20..40): Parent 0 is all-missing, Parent 1 matches 20/20.
        // Default switch_penalty (12.0) exceeds 20 * 0.25 = 5.0, yet BOTH halves
        // must be recovered because a missing parent cannot hold the Viterbi path.
        let n_markers = 40;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            let a = (m % 2) as u8;
            if m < 20 {
                joint.set_allele(m, 0, Some(a));
                joint.set_allele(m, 1, None);
            } else {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, Some(a));
            }
            joint.set_allele(m, 2, Some(a));
        }
        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "both observed halves must be recovered, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].n_markers, 20);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[1].n_markers, 20);
        assert_eq!(bins.bin_starts_bp.len(), 2);
    }

    #[test]
    fn diploid_parent_haplotype_switch_splits_ancestry_segments_by_track() {
        // Single diploid parent (hap0 & hap1 complementary); target matches
        // hap0 on [0..20) and hap1 on [20..40). Must emit 2 segments recording
        // parent_hap=0 (20 markers) and parent_hap=1 (20 markers).
        let n_markers = 40;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![false]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        for m in 0..n_markers {
            let h0 = (m % 2) as u8;
            let h1 = 1 - h0;
            joint.set_allele(m, 0, Some(h0));
            joint.set_allele(m, 1, Some(h1));
            joint.set_allele(m, 2, Some(if m < 20 { h0 } else { h1 }));
        }
        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "intra-parent track switch must split segment"
        );
        assert_eq!(
            (segs[0].parent_sample, segs[0].parent_hap, segs[0].n_markers),
            (0, 0, 20)
        );
        assert_eq!(
            (segs[1].parent_sample, segs[1].parent_hap, segs[1].n_markers),
            (0, 1, 20)
        );
        assert_eq!(segs[0].end_bp + 1, segs[1].start_bp);
        assert_eq!(bins.bin_starts_bp.len(), 2);
        assert!(bins.dosage.iter().all(|&d| (d - 1.0).abs() < 1e-6));
    }

    #[test]
    fn zero_seed_short_region_with_more_than_top_donors_finds_last_parent() {
        // 65 inbred reference parents (> default ancestry_top_donors=64), 40 markers
        // spanning 0.39 cM (< min_seed_cm * 0.5 = 0.5 cM -> 0 PBWT seeds).
        // Only parent #64 (the 65th parent) matches the target; candidate selection
        // must be order-independent and recover parent #64 across all 40 markers.
        let n_parents = 65usize;
        let n_markers = 40usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.01).collect(); // 0.39 cM total

        for m in 0..n_markers {
            let target_allele = (m % 2) as u8;
            for p in 0..(n_parents - 1) {
                joint.set_allele(m, p, Some(1 - target_allele));
            }
            joint.set_allele(m, n_parents - 1, Some(target_allele));
            joint.set_allele(m, n_parents, Some(target_allele));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "65th parent must be recovered despite 0 PBWT seeds, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, (n_parents - 1) as u32);
        assert_eq!(segs[0].n_markers, n_markers as u32);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }

    #[test]
    fn haplotype_major_popcnt_matches_marker_major_across_all_ranges() {
        let n_haps = 131usize; // non-multiple of 64
        let n_markers = 263usize; // non-multiple of 64
        let mut matrix = BitHaplotypeMatrix::new(n_haps, n_markers);
        for m in 0..n_markers {
            for h in 0..n_haps {
                let v = match (m * 17 + h * 31) % 11 {
                    0 => None,
                    1..=5 => Some(0),
                    _ => Some(1),
                };
                matrix.set_allele(m, h, v);
            }
        }
        let hap_major = super::HaplotypeMajorBits::from_matrix(&matrix, None).unwrap();
        let ranges = [
            (0, 1),
            (0, 64),
            (0, 65),
            (13, 47),
            (60, 70),
            (63, 193),
            (128, 263),
            (0, n_markers),
        ];
        for &(s, e) in &ranges {
            for (ha, hb) in [(0, 1), (0, 63), (63, 64), (64, 130), (129, 130)] {
                assert_eq!(
                    hap_major.count_mismatches(ha, hb, s, e),
                    matrix.count_mismatches(ha, hb, s, e),
                    "mismatch for haps ({ha}, {hb}) range [{s}, {e})"
                );
            }
        }
    }

    #[test]
    fn high_density_local_error_burst_does_not_fragment_or_drop_to_uncalled() {
        // High-density WGS scenario: 2,000 markers over 1.0 cM (100 markers per 0.05 cM step).
        // Target matches P0 on [0..1000) and P1 on [1000..2000), with a 25-marker local
        // imputation error burst (25% mismatch in step 5, markers 500..525) flipped to P1.
        // With opt-in step cap (`ancestry_step_max_markers = 8`), the burst must neither
        // trigger a spurious switch nor drop to `Uncalled`, recovering 2 contiguous segments.
        let n_markers = 2_000usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 500).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.0005).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            joint.set_allele(m, 0, Some(p0));
            joint.set_allele(m, 1, Some(p1));
            let tgt = if m < 1_000 {
                if (500..525).contains(&m) {
                    p1
                } else {
                    p0
                }
            } else {
                p1
            };
            joint.set_allele(m, 2, Some(tgt));
        }

        let cfg = HaploidIbdConfig {
            ancestry_step_max_markers: 8,
            ..HaploidIbdConfig::default()
        };
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "opt-in step cap must prevent 25-SNP error burst from fragmenting, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[0].end_bp + 1, segs[1].start_bp);
        assert_eq!(bins.bin_starts_bp.len(), 2);
    }

    #[test]
    fn default_unscaled_ancestry_recovers_short_single_step_segment() {
        // Default `ancestry_step_max_markers == 0` (unscaled): a true 0.05 cM (100-SNP)
        // short segment from P1 embedded inside a P0 background must be recovered.
        let n_markers = 2_000usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 500).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.0005).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            joint.set_allele(m, 0, Some(p0));
            joint.set_allele(m, 1, Some(p1));
            let tgt = if (500..600).contains(&m) { p1 } else { p0 };
            joint.set_allele(m, 2, Some(tgt));
        }

        let cfg = HaploidIbdConfig::default();
        assert_eq!(cfg.ancestry_step_max_markers, 0);
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            3,
            "default unscaled decoding must recover true 0.05 cM short segment, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[1].n_markers, 100);
        assert_eq!(segs[2].parent_sample, 0);
    }

    #[test]
    fn shared_step_scale_prevents_low_coverage_decoy_from_beating_full_parent() {
        // High-density step (100 markers per 0.05 cM) with opt-in `ancestry_step_max_markers = 8`.
        // P0 is the true full-coverage parent with 1% genotyping error (99/100 match per step).
        // P1 is a low-coverage decoy observed at only 8% of sites (8/100), matching all 8 sites.
        // Step-wide shared scaling must preserve raw candidate ranking and choose P0 everywhere.
        let n_markers = 1_000usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 500).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.0005).collect();

        for m in 0..n_markers {
            let a = (m % 2) as u8;
            joint.set_allele(m, 0, Some(a));
            if (m % 100) < 8 {
                joint.set_allele(m, 1, Some(a));
            } else {
                joint.set_allele(m, 1, None);
            }
            // 1% genotyping error on target at offset 50 of each 100-SNP step
            let tgt = if (m % 100) == 50 { 1 - a } else { a };
            joint.set_allele(m, 2, Some(tgt));
        }

        let cfg = HaploidIbdConfig {
            ancestry_step_max_markers: 8,
            ..HaploidIbdConfig::default()
        };
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "shared step scaling must keep full-coverage true parent P0 over 8% decoy P1, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].n_markers, n_markers as u32);
    }

    #[test]
    fn scattered_missingness_bridges_parent_without_free_switch() {
        // 200 markers at 0.05 cM/marker (1 marker per step under default ancestry_step_cm).
        // True parent is P0 everywhere; 1/3 of target calls are missing, and a single
        // genotyping error immediately after a missing step matches P1. Viterbi must
        // bridge scattered missing steps with neutral 0.0 emission and preserve
        // switch_penalty, emitting a single contiguous P0 segment.
        let n_markers = 200usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            joint.set_allele(m, 0, Some(p0));
            joint.set_allele(m, 1, Some(p1));
            let tgt = if m % 3 == 0 {
                None
            } else if m == 91 {
                Some(p1)
            } else {
                Some(p0)
            };
            joint.set_allele(m, 2, tgt);
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "scattered missingness must bridge P0 without spurious gaps/switches, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].start_bp, pos[0]);
        assert_eq!(segs[0].end_bp, pos[n_markers - 1]);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }

    #[test]
    fn chosen_parent_missing_gap_recovers_supported_alt_parent_in_p0_p1_p0() {
        // P0-P1-P0 where target is completely observed, P0 is all-missing in the
        // middle (8 or 15 markers), and P1 matches 100% in the middle. Both cases
        // must recover P1 rather than filling P0 or outputting Uncalled.
        for mid_len in [8usize, 15usize] {
            let left_len = 40usize;
            let right_len = 40usize;
            let n_markers = left_len + mid_len + right_len;
            let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
            let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
            let mut joint =
                BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
            let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
            let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

            for m in 0..n_markers {
                let p0 = (m % 2) as u8;
                let p1 = 1 - p0;
                if m >= left_len && m < left_len + mid_len {
                    joint.set_allele(m, 0, None);
                    joint.set_allele(m, 1, Some(p1));
                    joint.set_allele(m, 2, Some(p1));
                } else {
                    joint.set_allele(m, 0, Some(p0));
                    joint.set_allele(m, 1, Some(p1));
                    joint.set_allele(m, 2, Some(p0));
                }
            }

            let cfg = HaploidIbdConfig::default();
            let (segs, bins) =
                paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
            assert_eq!(
                segs.len(),
                3,
                "mid_len={mid_len}: expected P0-P1-P0 (3 segments), got {segs:?}"
            );
            assert_eq!(segs[0].parent_sample, 0);
            assert_eq!(segs[1].parent_sample, 1);
            assert_eq!(segs[1].n_markers, mid_len as u32);
            assert_eq!(segs[2].parent_sample, 0);
            assert_eq!(bins.bin_starts_bp.len(), 3);
        }
    }

    #[test]
    fn alt_parent_recovery_does_not_overwrite_adjacent_long_missing_gap() {
        // 40 markers P0, then 20 markers where P0 is all-missing: first 4 markers
        // match P1, remaining 16 markers (>= 10 markers & >= 0.5 cM) have zero
        // observations for all parents, then 40 markers P0. Must emit P0, P1 (4
        // markers), Uncalled gap [44..60), and P0.
        let n_markers = 100usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.1).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            if (40..44).contains(&m) {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p1));
            } else if (44..60).contains(&m) {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, None);
                joint.set_allele(m, 2, None);
            } else {
                joint.set_allele(m, 0, Some(p0));
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            3,
            "expected [P0, P1, P0] with Uncalled gap, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[1].n_markers, 4);
        assert_eq!(segs[1].end_bp, pos[43]);
        assert_eq!(segs[2].parent_sample, 0);
        assert_eq!(segs[2].start_bp, pos[60]);
        assert_eq!(bins.bin_starts_bp.len(), 3);
    }

    #[test]
    fn alt_parent_selection_filters_by_min_markers_before_picking_best_score() {
        // P0-mid-P0 with 3 parents (P0, P1, P2): on the 8-marker middle where P0
        // is all-missing, P1 has 8 comparable markers (7 match, 1 mismatch -> score
        // ~0.4 > 0, qualified), while P2 has only 3 matching markers (score ~2.1 > 0.4,
        // unqualified). Must select P1 rather than letting P2 shadow it.
        let n_markers = 88usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 3]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            if (40..48).contains(&m) {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, Some(if m == 47 { 1 - p1 } else { p1 }));
                joint.set_allele(m, 2, if m < 43 { Some(p1) } else { None });
                joint.set_allele(m, 3, Some(p1));
            } else {
                joint.set_allele(m, 0, Some(p0));
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p1));
                joint.set_allele(m, 3, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(segs.len(), 3, "expected [P0, P1, P0], got {segs:?}");
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[1].n_markers, 8);
        assert_eq!(segs[2].parent_sample, 0);
        assert_eq!(bins.bin_starts_bp.len(), 3);
    }

    #[test]
    fn chosen_parent_missing_gap_recovers_multiple_sequential_alt_parents() {
        // P0-[15 P1 + 14 P2]-P0: 29-marker gap where target is complete, P0 is
        // all-missing, P1 matches the first 15 markers (P2 missing), and P2
        // matches the last 14 markers (P1 missing). Must emit 4 segments:
        // [P0, P1(15), P2(14), P0] rather than blanking the P2 tail to Uncalled.
        let left_len = 40usize;
        let p1_len = 15usize;
        let p2_len = 14usize;
        let right_len = 40usize;
        let n_markers = left_len + p1_len + p2_len + right_len;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 3]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            let p2 = ((m / 2) % 2) as u8;
            if m >= left_len && m < left_len + p1_len {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, None);
                joint.set_allele(m, 3, Some(p1));
            } else if m >= left_len + p1_len && m < left_len + p1_len + p2_len {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, None);
                joint.set_allele(m, 2, Some(p2));
                joint.set_allele(m, 3, Some(p2));
            } else {
                joint.set_allele(m, 0, Some(p0));
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p2));
                joint.set_allele(m, 3, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            4,
            "expected [P0, P1, P2, P0] (4 segments), got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[1].n_markers, p1_len as u32);
        assert_eq!(segs[2].parent_sample, 2);
        assert_eq!(segs[2].n_markers, p2_len as u32);
        assert_eq!(segs[3].parent_sample, 0);
        assert_eq!(bins.bin_starts_bp.len(), 4);
    }

    #[test]
    fn alt_parent_recovery_re_evaluates_contradicted_tail_for_overlapping_parent() {
        // 36-marker gap where P0 is all-missing: P1 matches the first 32 markers
        // and mismatches the last 4 markers, while P2 matches the last 4 markers.
        // P1's higher overall score must not overwrite its locally contradicted tail;
        // output must recover 4 segments: [P0, P1(32), P2(4), P0].
        let left_len = 40usize;
        let p1_len = 32usize;
        let p2_len = 4usize;
        let right_len = 40usize;
        let n_markers = left_len + p1_len + p2_len + right_len;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 3]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            if m >= left_len && m < left_len + p1_len {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, None);
                joint.set_allele(m, 3, Some(p1));
            } else if m >= left_len + p1_len && m < left_len + p1_len + p2_len {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(1 - p1));
                joint.set_allele(m, 3, Some(1 - p1));
            } else {
                joint.set_allele(m, 0, Some(p0));
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p1));
                joint.set_allele(m, 3, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            4,
            "expected [P0, P1(32), P2(4), P0] (4 segments), got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[1].n_markers, p1_len as u32);
        assert_eq!(segs[2].parent_sample, 2);
        assert_eq!(segs[2].n_markers, p2_len as u32);
        assert_eq!(segs[3].parent_sample, 0);
        assert_eq!(bins.bin_starts_bp.len(), 4);
    }

    #[test]
    fn asymmetric_founder_missingness_does_not_let_low_missing_parent_swallow_true_founder() {
        // 4 founders over 400 markers: P0 is the true founder across the entire
        // chromosome with 50% missingness (0% mismatch on observed sites), while
        // P1/P2/P3 have 0% missingness and share 95% IBS with P0 (1 mismatch
        // every 20 markers). Call-rate-normalized emissions must recover P0
        // across the whole chromosome rather than letting 0%-missing P1 swallow it.
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; 4]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let true_a = (m % 2) as u8;
            // P0: 50% missing, 0% mismatch
            joint.set_allele(m, 0, if m % 2 == 0 { Some(true_a) } else { None });
            // P1..P3: 0% missing, 95% IBS with P0 (mismatch every 20 markers at distinct phases)
            for p in 1..4usize {
                let mis = (m + p * 7) % 20 == 0;
                joint.set_allele(m, p, Some(if mis { 1 - true_a } else { true_a }));
            }
            joint.set_allele(m, 4, Some(true_a));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "50%-missing true founder P0 must not be swallowed by 0%-missing 95%-IBS founders, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].start_bp, pos[0]);
        assert_eq!(segs[0].end_bp, pos[n_markers - 1]);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }

    #[test]
    fn local_matching_decoy_missing_elsewhere_cannot_intrude_into_full_parent() {
        // P0 matches all 400 markers. Decoy P1 is missing on 360 markers and
        // observed only on the middle 40 markers where it is identical to P0.
        // Missingness elsewhere must never give P1 a local score advantage over P0.
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let a = (m % 2) as u8;
            joint.set_allele(m, 0, Some(a));
            joint.set_allele(
                m,
                1,
                if (180..220).contains(&m) {
                    Some(a)
                } else {
                    None
                },
            );
            joint.set_allele(m, 2, Some(a));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "decoy missing elsewhere must not intrude into full-coverage matching P0, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].n_markers, n_markers as u32);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }

    #[test]
    fn two_marker_p0_missing_gap_does_not_insert_spurious_p1_segment() {
        // P0 matches everywhere except a 2-marker missing gap in the middle where
        // P1 happens to match both markers. Raw comparable floor (>= 4) must keep
        // P0 bridged rather than inserting a 2-marker P1 segment.
        let n_markers = 82usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            if (40..42).contains(&m) {
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p1));
            } else {
                joint.set_allele(m, 0, Some(p0));
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "2-marker P0 missing gap must stay bridged as P0, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }

    #[test]
    fn dp_switch_requiring_missing_credit_below_raw_floor_does_not_insert_spurious_island() {
        // 400 markers where P0 is the true parent everywhere except 5 consecutive
        // target genotyping errors on [198..203). P1 mismatches P0 outside the burst,
        // matches the target on only 1 of the 5 burst markers (m=200), and is missing
        // on the other 4 burst markers. DP missingness credit (+2.9 vs P0 -22.5) must
        // not bypass the 4-marker raw comparable floor to insert a 1-marker P1 island.
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            joint.set_allele(m, 0, Some(p0));
            if (198..203).contains(&m) {
                joint.set_allele(m, 1, if m == 200 { Some(p1) } else { None });
                joint.set_allele(m, 2, Some(p1));
            } else {
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "1-marker P1 relying on missingness credit must not insert a spurious island, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].start_bp, pos[0]);
        assert_eq!(segs[0].end_bp, pos[n_markers - 1]);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }

    #[test]
    fn short_true_segment_standing_without_missing_credit_is_preserved() {
        // 400 markers where P0 matches everywhere except 5 consecutive markers
        // [198..203) where P0 mismatches all 5 (raw -22.5). True P1 matches 3 of
        // the 5 markers (raw +2.1) and is missing on 2. Even without missingness
        // credit, P1 beats P0 by 24.6 > 2 * switch_penalty (24.0), so P1 must be
        // preserved as [P0, P1(3), P0].
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        // Strictly > default ancestry_step_cm (0.05) so every marker is its own step.
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.06).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            joint.set_allele(m, 0, Some(p0));
            if (198..203).contains(&m) {
                let p1_obs = m == 198 || m == 200 || m == 202;
                joint.set_allele(m, 1, if p1_obs { Some(p1) } else { None });
                joint.set_allele(m, 2, Some(p1));
            } else {
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            3,
            "3-match/2-miss P1 standing on uncompensated score must be preserved, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[1].n_markers, 3);
        assert_eq!(segs[2].parent_sample, 0);
        assert_eq!(bins.bin_starts_bp.len(), 3);
    }

    #[test]
    fn mismatched_single_marker_alt_loses_to_uncalled_in_same_parent_flank() {
        // 400 markers where P0 matches everywhere except 10 consecutive markers
        // [195..205) where P0 mismatches all 10 (raw -45.0). P1 has only 1 observed
        // marker in the burst (m=200), which ALSO mismatches (raw -4.5), and is
        // missing on the other 9 (compensated DP score = -4.5 + 9 * 0.55 = +0.45 > 0,
        // so DP enters P1). Uncompensated: Uncalled (-24.0) beats both P1 (-28.5)
        // and P0 (-45.0), so the burst must be left Uncalled ([P0, P0]).
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.06).collect();

        for m in 0..n_markers {
            let p0 = (m % 2) as u8;
            let p1 = 1 - p0;
            joint.set_allele(m, 0, Some(p0));
            if (195..205).contains(&m) {
                joint.set_allele(m, 1, if m == 200 { Some(p0) } else { None });
                joint.set_allele(m, 2, Some(p1));
            } else {
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, 2, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "mismatched 1-marker P1 must lose to Uncalled (-24 vs -28.5), got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].end_bp, pos[194]);
        assert_eq!(segs[1].parent_sample, 0);
        assert_eq!(segs[1].start_bp, pos[205]);
        assert_eq!(bins.bin_starts_bp.len(), 2);
    }

    #[test]
    fn out_of_panel_hidden_founder_is_rejected_to_uncalled_under_frequency_model() {
        // 12-founder reference panel (>= FREQ_MODEL_MIN_REF_HAPS) with skewed
        // allele frequencies (2/12 carry minor allele 1, 10/12 carry major allele 0).
        // Target is an out-of-panel hidden founder sharing exact 542/600 = 90.33% IBS
        // with P0 (all 0s; 58 mismatches on target minor-allele sites). Under the old
        // flat +0.7/-4.5 model, P0 scored +118.4 and forced a full-chromosome mis-call;
        // under the frequency-aware model, P0 scores ~-151.5 and is rejected to Uncalled.
        let n_parents = 12usize;
        let n_markers = 600usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            // P0 carries major allele 0 everywhere; 2 distinct founders in 1..12
            // carry minor allele 1 at every site (af1 ≈ 17%, af0 ≈ 83%).
            let minor_a = 1 + (m % (n_parents - 1));
            let minor_b = 1 + ((m + 5) % (n_parents - 1));
            joint.set_allele(m, 0, Some(0));
            for p in 1..n_parents {
                let allele = if p == minor_a || p == minor_b {
                    1u8
                } else {
                    0u8
                };
                joint.set_allele(m, p, Some(allele));
            }
            // Hidden founder carries minor allele 1 at 58 sites (542/600 = 90.33% IBS with P0).
            let tgt = if m % 10 == 0 && m < 580 { 1u8 } else { 0u8 };
            joint.set_allele(m, n_parents, Some(tgt));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert!(
            segs.is_empty(),
            "out-of-panel founder with 90.33% IBS on skewed-MAF sites must be rejected to Uncalled, got {segs:?}"
        );
        assert!(bins.dosage.is_empty());
    }

    #[test]
    fn adding_all_missing_sixth_founder_preserves_five_founder_ancestry() {
        // 5 observed founders where P0 matches target with 12% mismatch (24/200):
        // under the legacy flat +0.7/-4.5 model, P0 scores +15.2 and is called.
        // Adding a 6th all-missing founder raises n_ref_haps to 6 (>= FREQ_MODEL_MIN_REF_HAPS),
        // but every site still has obs_ref = 5 < 6 and must fall back to the exact
        // legacy +0.7/-4.5 model, producing identical segments instead of Uncalled.
        let n_markers = 200usize;
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();
        let cfg = HaploidIbdConfig::default();

        let run_panel = |add_all_missing_sixth: bool| {
            let n_parents = if add_all_missing_sixth { 6 } else { 5 };
            let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
            let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
            let mut joint =
                BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
            for m in 0..n_markers {
                let p0 = (m % 2) as u8;
                joint.set_allele(m, 0, Some(p0));
                for p in 1..5usize {
                    joint.set_allele(m, p, Some(1 - p0));
                }
                if add_all_missing_sixth {
                    joint.set_allele(m, 5, None);
                }
                let is_err = m % 25 == 0 || m % 25 == 8 || m % 25 == 16; // 12% mismatch
                joint.set_allele(m, n_parents, Some(if is_err { 1 - p0 } else { p0 }));
            }
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg).0
        };

        let segs_5 = run_panel(false);
        let segs_6 = run_panel(true);
        assert_eq!(
            segs_5.len(),
            1,
            "5-founder panel must call P0, got {segs_5:?}"
        );
        assert_eq!(segs_5[0].parent_sample, 0);
        assert_eq!(
            segs_6.len(),
            segs_5.len(),
            "adding an all-missing 6th founder must not alter ancestry output: 5={segs_5:?}, 6={segs_6:?}"
        );
        assert_eq!(segs_6[0].parent_sample, segs_5[0].parent_sample);
        assert_eq!(segs_6[0].start_bp, segs_5[0].start_bp);
        assert_eq!(segs_6[0].end_bp, segs_5[0].end_bp);
        assert_eq!(segs_6[0].n_markers, segs_5[0].n_markers);
        assert!((segs_6[0].confidence - segs_5[0].confidence).abs() < 1e-6);
    }

    #[test]
    fn equal_comparable_counts_with_different_site_information_receives_credit() {
        // 16-founder panel, 50 steps of 8 markers each (400 markers total).
        // Every founder observes exactly 4 markers per step (`step_max_comp == 4`):
        //   - P0 observes the 4 common-allele sites (off 4..8) with 0 mismatches.
        //   - P1 observes the 4 rare-allele sites (off 0..4) with 1 mismatch (25% error).
        //   - P2..P8 observe off 0..4 (all mismatching target); P9..P15 observe off 4..8
        //     (each with 1 mismatch per step), keeping `obs_ref = 8 >= 6` at every site.
        // Even though `comp(P0) == comp(P1) == step_max_comp == 4`, P0 has lower
        // observed site information (`cand_info < step_max_info`) and must receive
        // `0.75 * (step_max_info - cand_info)` credit so 0%-error P0 beats 25%-error P1.
        let n_parents = 16usize;
        let n_steps = 50usize;
        let n_markers = n_steps * 8;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        // Each 8-marker block shares one genetic coordinate; blocks are spaced 0.06 cM (> 0.05).
        let cm: Vec<f64> = (0..n_markers).map(|m| (m / 8) as f64 * 0.06).collect();

        for m in 0..n_markers {
            let off = m % 8;
            if off < 4 {
                // Rare-allele half of step: target = 1, P0 missing, P1 has 1 mismatch at off=0,
                // P2..P8 carry 0 (so obs_ref = 8, minor allele 1 is rare).
                joint.set_allele(m, 0, None);
                joint.set_allele(m, 1, Some(if off == 0 { 0 } else { 1 }));
                for p in 2..9usize {
                    joint.set_allele(m, p, Some(0));
                }
                for p in 9..n_parents {
                    joint.set_allele(m, p, None);
                }
                joint.set_allele(m, n_parents, Some(1));
            } else {
                // Common-allele half of step: target = 0, P0 matches all 4, P1..P8 missing,
                // P9..P15 carry mostly 0 with 1 mismatch each (obs_ref = 8, major allele 0 common).
                joint.set_allele(m, 0, Some(0));
                for p in 1..9usize {
                    joint.set_allele(m, p, None);
                }
                for p in 9..n_parents {
                    let mis = off == 4 + ((p - 9) % 4);
                    joint.set_allele(m, p, Some(if mis { 1 } else { 0 }));
                }
                joint.set_allele(m, n_parents, Some(0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "0%-error P0 observing common sites must beat 25%-error P1 observing rare sites despite equal comp==4, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].n_markers, (n_steps * 4) as u32);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }

    #[test]
    fn multi_parent_asymmetric_missingness_still_recovers_high_missing_true_founder() {
        // 12-founder panel (>= FREQ_MODEL_MIN_REF_HAPS): P0 is the true founder
        // with 50% missingness (0% mismatch), while P1..P11 have 0% missingness
        // and share 95% IBS with P0. Frequency-scaled missingness credit must
        // recover P0 across the entire chromosome.
        let n_parents = 12usize;
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            let true_a = if m % 4 == 0 { 1u8 } else { 0u8 };
            joint.set_allele(m, 0, if m % 2 == 0 { Some(true_a) } else { None });
            for p in 1..n_parents {
                let mis = (m + p * 7) % 20 == 0;
                joint.set_allele(m, p, Some(if mis { 1 - true_a } else { true_a }));
            }
            joint.set_allele(m, n_parents, Some(true_a));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "50%-missing true founder P0 must be recovered in 12-founder frequency model, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].start_bp, pos[0]);
        assert_eq!(segs[0].end_bp, pos[n_markers - 1]);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }

    #[test]
    fn low_llr_segment_confidence_reflects_uncalled_competition() {
        // Single 3-marker segment where P0 matches all 3 markers (raw LLR ≈ 2.1)
        // and P1 mismatches all 3 (raw LLR ≈ -13.5). Including Uncalled (0.0) in
        // the softmax denominator must keep confidence < 0.95 rather than 1.0000.
        let n_markers = 3usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true, true]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.06).collect();

        for m in 0..n_markers {
            let a = (m % 2) as u8;
            joint.set_allele(m, 0, Some(a));
            joint.set_allele(m, 1, Some(1 - a));
            joint.set_allele(m, 2, Some(a));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].parent_sample, 0);
        assert!(
            segs[0].confidence < 0.95,
            "3-marker segment with raw LLR ~2.1 must reflect Uncalled competition (confidence={})",
            segs[0].confidence
        );
    }

    /// Deterministic 64-bit SplitMix hash producing uncorrelated alleles per `(m, p)`.
    fn splitmix_allele_bits(m: usize, p: usize) -> u64 {
        let mut z = ((m as u64) << 16) ^ (p as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Deterministic pairwise-distinct synthetic founder allele generator (~25% MAF).
    fn distinct_founder_allele(m: usize, p: usize) -> u8 {
        ((splitmix_allele_bits(m, p) & 3) == 0) as u8
    }

    /// Deterministic pairwise-distinct synthetic founder allele generator (~50% MAF).
    fn balanced_founder_allele(m: usize, p: usize) -> u8 {
        (splitmix_allele_bits(m, p) & 1) as u8
    }

    #[test]
    fn out_of_panel_mosaic_stitching_run_is_rejected_to_uncalled() {
        // 12 independent balanced ~50%-MAF founders over 600 markers (0.045 cM/marker).
        // [0..300) cleanly matches visible founder P0 (13.5 cM, 0 mismatches -> baseline=0.003).
        // [300..600) has 6 short 2.25 cM blocks (50 markers each, P1..P6; avg span < 2.5 cM):
        // 5 weak blocks carry 3 mismatches each (6% error, raw LLR ~+16.4 > switch_penalty),
        // while interior block P4 has 0 mismatches and strong unambiguous single-founder LLR.
        // Rule B1 must hard-reject all 5 weak mosaic blocks to Uncalled while preserving
        // strong 0-mismatch P4 with a mosaic-interior confidence discount of 0.75.
        let n_parents = 12usize;
        let n_markers = 600usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.045).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
            }
            if m >= 300 {
                let blk = (m - 300) / 50; // 0..6 -> P1..P6
                let off = (m - 300) % 50;
                let src_a = joint.allele(m, 1 + blk).unwrap();
                let flip = blk != 3 && (off == 10 || off == 25 || off == 40);
                joint.set_allele(m, n_parents, Some(if flip { 1 - src_a } else { src_a }));
            } else {
                joint.set_allele(m, n_parents, joint.allele(m, 0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "5 weak mosaic blocks must be rejected while preserving strong 0-mismatch P4, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert!(segs[0].end_bp >= pos[295] && segs[0].end_bp <= pos[305]);
        assert_eq!(segs[0].ambiguous_parents, 1);
        assert!(segs[0].confidence > 0.99);
        assert_eq!(segs[1].parent_sample, 4);
        assert!(
            (segs[1].confidence - 0.75).abs() < 1e-3,
            "interior short strong segment in rejected mosaic must get 0.75 discount, got {}",
            segs[1].confidence
        );
        assert_eq!(bins.bin_starts_bp.len(), 2);
    }

    #[test]
    fn out_of_panel_excess_mismatch_segment_confidence_is_calibrated_below_clean_segment() {
        // 12 pairwise-distinct founders over 600 markers (0.06 cM/marker = 36 cM).
        // [0..400) cleanly matches P0 (0 mismatches -> baseline_err=0.003, ambiguous=1).
        // [400..600) matches P1 with 4 well-separated mismatches (2.0%, 4 clusters > 0.5 cM,
        // below B3's hard-rejection count floor of 5). Both segments are emitted, P0 stays
        // > 0.995, and P1's composite confidence drops well below 0.90.
        let n_parents = 12usize;
        let n_markers = 600usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.06).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(distinct_founder_allele(m, p)));
            }
            let tgt = if m < 400 {
                joint.allele(m, 0).unwrap()
            } else {
                let p1_a = joint.allele(m, 1).unwrap();
                let flip = m == 425 || m == 475 || m == 525 || m == 575;
                if flip {
                    1 - p1_a
                } else {
                    p1_a
                }
            };
            joint.set_allele(m, n_parents, Some(tgt));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(segs.len(), 2, "expected [P0, P1], got {segs:?}");
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[0].ambiguous_parents, 1);
        assert_eq!(segs[1].parent_sample, 1);
        assert!(segs[0].confidence > 0.995);
        assert!(
            segs[1].confidence < 0.90,
            "2% excess-mismatch segment must have composite confidence < 0.90, got {}",
            segs[1].confidence
        );
    }

    #[test]
    fn uniformly_noisy_in_panel_sample_is_not_falsely_rejected() {
        // 12 pairwise-distinct founders over 600 markers: true P0 on [0..300) and true P1
        // on [300..600), both carrying uniform ~2.7% genotyping error (8 mismatches each).
        // Sample-baseline calibration adapts `baseline_err` to ~2.7%, preserving both
        // true in-panel segments without false out-of-panel rejection.
        let n_parents = 12usize;
        let n_markers = 600usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.06).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(distinct_founder_allele(m, p)));
            }
            let base = if m < 300 {
                joint.allele(m, 0).unwrap()
            } else {
                joint.allele(m, 1).unwrap()
            };
            let off = m % 300;
            let flip = off == 20
                || off == 55
                || off == 90
                || off == 125
                || off == 160
                || off == 195
                || off == 230
                || off == 265;
            joint.set_allele(m, n_parents, Some(if flip { 1 - base } else { base }));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "uniformly noisy in-panel sample must keep both P0 and P1, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(bins.bin_starts_bp.len(), 2);
    }

    #[test]
    fn true_short_recombination_series_without_uncalled_gaps_is_preserved() {
        // Regression for P1 (B1 & B2 non-cascading): 12 uncorrelated ~50%-MAF founders
        // so each 40-marker / 2.2 cM true block with 2 errors scores ~+14 > switch_penalty=12
        // and is emitted by Viterbi. P0 (200 markers) is followed by 4 true short
        // recombination segments P1..P4 and a 22 cM outer divergence Uncalled gap.
        // Neither B1 nor B2 may wipe or cascade through P1..P4.
        let n_parents = 12usize;
        let n_markers = 760usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.055).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
            }
            let tgt = if m < 200 {
                joint.allele(m, 0).unwrap()
            } else if m < 360 {
                let blk = (m - 200) / 40; // 0..4 -> P1, P2, P3, P4 (2.2 cM each)
                let off = (m - 200) % 40;
                let src_a = joint.allele(m, 1 + blk).unwrap();
                let flip = off == 12 || off == 28;
                if flip {
                    1 - src_a
                } else {
                    src_a
                }
            } else {
                // Outer 22 cM divergence Uncalled gap beyond P4
                for p in 0..n_parents {
                    joint.set_allele(m, p, Some(0));
                }
                1u8
            };
            joint.set_allele(m, n_parents, Some(tgt));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            5,
            "true P0-P1-P2-P3-P4 short recombination series before 22 cM gap must be preserved, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[2].parent_sample, 2);
        assert_eq!(segs[3].parent_sample, 3);
        assert_eq!(segs[4].parent_sample, 4);
    }

    #[test]
    fn sandwiched_strong_single_cluster_true_segment_is_preserved_by_b2() {
        // Regression for P1 (B2 sandwiched_reject): 12 uncorrelated ~50%-MAF founders.
        // Clean P0 flanks establish baseline_err=0.003; a 90-SNP (3.6 cM) true P1 segment
        // sandwiched between two divergence Uncalled gaps has only 2 errors spaced 0.12 cM
        // apart (1 cluster, strong raw LLR ~+48, raw_conf=1.0). B2 must preserve P1.
        let n_parents = 12usize;
        let n_markers = 810usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.04).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
            }
            let tgt = if m < 300 || m >= 510 {
                joint.allele(m, 0).unwrap()
            } else if (360..450).contains(&m) {
                // True 90-SNP (3.6 cM) P1 with 2 errors spaced 3 markers (0.12 cM) apart
                let base = joint.allele(m, 1).unwrap();
                if m == 400 || m == 403 {
                    1 - base
                } else {
                    base
                }
            } else {
                // Divergence Uncalled gaps on both sides of P1
                for p in 0..n_parents {
                    joint.set_allele(m, p, Some(0));
                }
                1u8
            };
            joint.set_allele(m, n_parents, Some(tgt));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            3,
            "sandwiched strong single-cluster true P1 segment must be preserved by B2, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[2].parent_sample, 0);
    }

    #[test]
    fn b3_mismatch_clustering_is_invariant_to_large_ancestry_step_cm() {
        // Regression for P1 (B3 step-size invariance): 12 distinct founders (0.01 cM/marker).
        // [0..400) is clean P0 (4 cM); [400..1400) is true P1 (10 cM, 1,000 SNPs) with 35
        // errors spaced 0.12 cM (< 0.5 cM) apart across 4.08 cM (1 single contiguous cluster).
        // Even with `--ancestry-step-cm 1.0`, marker-coordinate clustering must count 1
        // cluster and preserve P1.
        let n_parents = 12usize;
        let n_markers = 1_400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 1_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.01).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                let x = m
                    .wrapping_mul(2_654_435_761usize)
                    .wrapping_add(p.wrapping_mul(2_246_822_519usize) ^ 0x9e37_79b9usize);
                let a = (((x ^ (x >> 13)) & 1) == 1) as u8;
                joint.set_allele(m, p, Some(a));
            }
            let tgt = if m < 400 {
                joint.allele(m, 0).unwrap()
            } else {
                let base = joint.allele(m, 1).unwrap();
                // 35 errors spaced 12 markers (0.12 cM) apart across [650..1059) = 4.08 cM
                let is_err = m >= 650 && m <= 1058 && (m - 650) % 12 == 0;
                if is_err {
                    1 - base
                } else {
                    base
                }
            };
            joint.set_allele(m, n_parents, Some(tgt));
        }

        let cfg = HaploidIbdConfig {
            ancestry_step_cm: 1.0,
            ..HaploidIbdConfig::default()
        };
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "contiguous 0.12 cM-spaced error block must remain 1 cluster under step_cm=1.0, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
    }

    #[test]
    fn uncalled_flanked_true_segment_below_sample_baseline_is_preserved() {
        // Regression for P1 (B2): 12 distinct founders (balanced ~50% MAF so 60-marker P1
        // scores ~+34 > 2*switch_penalty=24 and is emitted by Viterbi). P0 flanks carry
        // 2.67% error (baseline_err ≈ 2.67%), while the 2.1 cM true P1 segment sandwiched
        // between two divergence Uncalled gaps has only 1/60 = 1.67% mismatch (< baseline).
        // B2 must preserve P1.
        let n_parents = 12usize;
        let n_markers = 780usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.035).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                let x = m
                    .wrapping_mul(2_654_435_761usize)
                    .wrapping_add(p.wrapping_mul(2_246_822_519usize) ^ 0x9e37_79b9usize);
                let a = (((x ^ (x >> 13)) & 1) == 1) as u8;
                joint.set_allele(m, p, Some(a));
            }
            let tgt = if m < 300 || m >= 480 {
                let base = joint.allele(m, 0).unwrap();
                let local = if m < 300 { m } else { m - 480 };
                let flip = (local % 37) == 10 && local < 296; // 8/300 = 2.67%
                if flip {
                    1 - base
                } else {
                    base
                }
            } else if (360..420).contains(&m) {
                // True 2.1 cM P1 segment (60 markers) with 1/60 = 1.67% mismatch (< 2.67% baseline)
                let base = joint.allele(m, 1).unwrap();
                if m == 390 {
                    1 - base
                } else {
                    base
                }
            } else {
                // Divergence Uncalled gap: all refs 0, target 1
                for p in 0..n_parents {
                    joint.set_allele(m, p, Some(0));
                }
                1u8
            };
            joint.set_allele(m, n_parents, Some(tgt));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            3,
            "Uncalled-flanked 2.1 cM true P1 segment cleaner than baseline must be preserved, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[2].parent_sample, 0);
    }

    #[test]
    fn b2_does_not_propagate_rejection_across_pure_missing_gap() {
        // Regression for P1 (B2): true P5 segment (3.15 cM, 70 markers, 2 errors) separated
        // by a 22.5 cM all-missing target gap from a B1-rejected 6-block out-of-panel mosaic.
        // B2 must NEVER propagate rejection across the pure-missing gap to delete P5, and
        // the entire far-side mosaic must be rejected to Uncalled.
        let n_parents = 12usize;
        let n_markers = 1_120usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.045).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
            }
            if m < 250 {
                // Clean P0 establishing baseline_err = 0.003
                joint.set_allele(m, n_parents, joint.allele(m, 0));
            } else if m < 320 {
                // True P5 (70 markers, 3.15 cM, 2 mismatches -> 2.86%)
                let base = joint.allele(m, 5).unwrap();
                let flip = m == 270 || m == 300;
                joint.set_allele(m, n_parents, Some(if flip { 1 - base } else { base }));
            } else if m < 820 {
                // 500 markers = 22.5 cM pure all-missing target gap
                joint.set_allele(m, n_parents, None);
            } else {
                // 6-block B1-rejected out-of-panel mosaic on [820..1120) (50 markers/2.25 cM each)
                let blk = (m - 820) / 50;
                let off = (m - 820) % 50;
                let src_a = joint.allele(m, 1 + blk).unwrap();
                let flip = off == 10 || off == 25 || off == 40;
                joint.set_allele(m, n_parents, Some(if flip { 1 - src_a } else { src_a }));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert!(
            segs.iter()
                .any(|s| s.parent_sample == 0 && s.n_markers >= 240),
            "clean P0 must be preserved, got {segs:?}"
        );
        assert!(
            segs.iter().any(|s| s.parent_sample == 5 && s.n_markers >= 60),
            "B2 must not propagate across a 22.5 cM all-missing gap to delete true P5, got {segs:?}"
        );
        assert!(
            segs.iter().all(|s| s.end_bp < pos[820]),
            "far-side 6-block mosaic on [820..1120) must be completely rejected, got {segs:?}"
        );
    }

    #[test]
    fn high_density_contiguous_local_error_cluster_is_not_rejected_by_b3() {
        // Regression for P1 (B3): 12 distinct founders at high density (0.008 cM/marker).
        // [0..400) is clean P0 (3.2 cM, 0 mismatches -> baseline=0.003); [400..800) is true
        // P1 (3.2 cM, 400 SNPs) with 10 genotyping errors forming a single contiguous 1.15 cM
        // local noise block (consecutive errors spaced 0.128 cM < 0.5 cM -> exactly 1 cluster).
        // B3 must count consecutive-mismatch gaps and preserve P1.
        let n_parents = 12usize;
        let n_markers = 800usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 1_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.008).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                // Balanced ~50% MAF so the 10-error local block stays in P1 under Viterbi.
                let x = m
                    .wrapping_mul(2_654_435_761usize)
                    .wrapping_add(p.wrapping_mul(2_246_822_519usize) ^ 0x9e37_79b9usize);
                let a = (((x ^ (x >> 13)) & 1) == 1) as u8;
                joint.set_allele(m, p, Some(a));
            }
            let tgt = if m < 400 {
                joint.allele(m, 0).unwrap()
            } else {
                let base = joint.allele(m, 1).unwrap();
                // 10 errors spaced 16 markers (0.128 cM < 0.5 cM) apart across [520..665):
                // total span = 144 * 0.008 = 1.152 cM (> 1.0 cM, > 35% of 3.2 cM), yet 1 cluster.
                let is_burst_err = m >= 520 && m <= 664 && (m - 520) % 16 == 0;
                if is_burst_err {
                    1 - base
                } else {
                    base
                }
            };
            joint.set_allele(m, n_parents, Some(tgt));
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            2,
            "true 400-SNP P1 segment with a single contiguous 1.15 cM 10-error block must be preserved, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
    }

    #[test]
    fn short_clean_anchor_with_mosaic_run_does_not_pollute_baseline_err() {
        // Regression for P1 (baseline_err contamination): 50 clean markers of P0 (3.0 cM)
        // followed by 6 x 40-marker (2.4 cM) stitched mosaic blocks (~2.5% mismatch).
        // Because no segment exceeds MOSAIC_SEG_MAX_CM (6.0 cM), the mosaic blocks must
        // not inflate baseline_err; B1 must reject all 6 mosaic blocks (including P6)
        // while preserving the clean P0 anchor.
        let n_parents = 12usize;
        let n_markers = 290usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.06).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(distinct_founder_allele(m, p)));
            }
            if m >= 50 {
                let blk = (m - 50) / 40;
                let off = (m - 50) % 40;
                let primary = 1 + blk;
                let sibling = 7 + (blk % 5);
                let src_a = joint.allele(m, primary).unwrap();
                joint.set_allele(m, sibling, Some(src_a));
                let flip = (blk == 0 && (off == 12 || off == 28))
                    || ((blk == 1 || blk == 2 || blk == 4 || blk == 5) && off == 20);
                joint.set_allele(m, n_parents, Some(if flip { 1 - src_a } else { src_a }));
            } else {
                joint.set_allele(m, n_parents, joint.allele(m, 0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "50-marker P0 anchor + 6x40 mosaic must reject all 6 mosaic blocks including P6, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(bins.bin_starts_bp.len(), 1);
    }

    #[test]
    fn true_short_recombination_with_competing_founder_is_preserved() {
        // Regression for P1 (B1 over-rejection with positive competitor): 12 balanced founders.
        // Clean P0 (200 markers) + 4 true short 50-marker (2.4 cM) blocks P1..P4 with 1 error
        // each; on each block a related founder shares most alleles (giving second_best > 0)
        // but diverges at 3 diagnostic sites so the true founder still leads by > switch_penalty.
        // B1 must preserve all 5 segments rather than deleting true segments for second_best > 0.
        let n_parents = 12usize;
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.048).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
            }
            if m < 200 {
                joint.set_allele(m, n_parents, joint.allele(m, 0));
            } else {
                let blk = (m - 200) / 50;
                let off = (m - 200) % 50;
                let primary = 1 + blk;
                let cousin = 7 + blk;
                let src_a = joint.allele(m, primary).unwrap();
                // Cousin shares 47/50 sites (second_best > 0) but differs at 3 diagnostic sites.
                let cousin_diff = off == 10 || off == 25 || off == 40;
                joint.set_allele(m, cousin, Some(if cousin_diff { 1 - src_a } else { src_a }));
                let flip = off == 18;
                joint.set_allele(m, n_parents, Some(if flip { 1 - src_a } else { src_a }));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            5,
            "true short recombination blocks beating a positive competitor by > switch_penalty must be preserved, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[2].parent_sample, 2);
        assert_eq!(segs[3].parent_sample, 3);
        assert_eq!(segs[4].parent_sample, 4);
    }

    #[test]
    fn short_500kb_introgression_below_viterbi_switch_threshold_is_recovered() {
        // 12 balanced founders over 400 markers (31,250 bp and 0.025 cM per marker ->
        // 16 markers = 500 kb / 0.40 cM). Background is P0; on [200..216) target carries
        // a true 500 kb P1 introgression with only 4 diagnostic SNPs (201, 205, 210, 214),
        // so 4 * 5.99 = 23.96 < 2 * switch_penalty (24.0) and Viterbi alone misses it.
        // Marker-level short-introgression refinement must recover [P0, P1, P0] with exact
        // contiguous breakpoints.
        let n_parents = 12usize;
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 31_250).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.025).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
            }
            let p0 = joint.allele(m, 0).unwrap();
            if (200..216).contains(&m) {
                // P1 differs from P0 at exactly 4 diagnostic markers inside the 500 kb block.
                let diag = m == 201 || m == 205 || m == 210 || m == 214;
                let p1 = if diag { 1 - p0 } else { p0 };
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, n_parents, Some(p1));
            } else {
                // Ensure flanking markers immediately outside [200..216) distinguish P0 from P1.
                if m == 199 || m == 216 {
                    joint.set_allele(m, 1, Some(1 - p0));
                }
                joint.set_allele(m, n_parents, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, bins) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            3,
            "500 kb 4-diagnostic-SNP introgression must be recovered as [P0, P1, P0], got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[2].parent_sample, 0);
        assert_eq!(segs[0].end_bp + 1, segs[1].start_bp);
        assert_eq!(segs[1].end_bp + 1, segs[2].start_bp);
        assert_eq!(segs[1].start_bp, pos[200]);
        assert_eq!(segs[1].end_bp + 1, pos[216]);
        assert_eq!(bins.bin_starts_bp.len(), 3);
    }

    #[test]
    fn short_250kb_three_diagnostic_introgression_is_recovered() {
        // 12 balanced founders over 400 markers (31,250 bp and 0.025 cM per marker ->
        // 8 markers = 250 kb / 0.20 cM). On [200..208), true P2 introgression carries 3
        // diagnostic SNPs (201, 204, 206) spanning 6 markers (alt_comp=6 >= 5, ll_gain~18).
        // Refinement must recover [P0, P2, P0].
        let n_parents = 12usize;
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 31_250).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.025).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
            }
            let p0 = joint.allele(m, 0).unwrap();
            if (200..208).contains(&m) {
                let diag = m == 201 || m == 204 || m == 206;
                let p2 = if diag { 1 - p0 } else { p0 };
                joint.set_allele(m, 2, Some(p2));
                joint.set_allele(m, n_parents, Some(p2));
            } else {
                if m == 199 || m == 208 {
                    joint.set_allele(m, 2, Some(1 - p0));
                }
                joint.set_allele(m, n_parents, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            3,
            "250 kb 3-diagnostic-SNP introgression must be recovered as [P0, P2, P0], got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 2);
        assert_eq!(segs[2].parent_sample, 0);
    }

    #[test]
    fn common_allele_correlated_noise_burst_is_not_turned_into_false_introgression() {
        // 12 skewed-MAF founders (2/12 carry minor allele 1, 10/12 carry major allele 0).
        // True parent is P0 everywhere; 3 minor-allele sites of P0 inside a 200 kb window
        // are flipped by correlated noise to major allele 0 (matching 9+ other founders).
        // Refinement must NOT splice a spurious introgression (low alt_raw + high tie count).
        let n_parents = 12usize;
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 31_250).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.025).collect();

        for m in 0..n_markers {
            if m == 200 || m == 202 || m == 205 {
                // P0 and P1 carry minor allele 1; P2..P11 carry major allele 0.
                joint.set_allele(m, 0, Some(1));
                joint.set_allele(m, 1, Some(1));
                for p in 2..n_parents {
                    joint.set_allele(m, p, Some(0));
                }
                // Correlated noise flips target from 1 -> major allele 0.
                joint.set_allele(m, n_parents, Some(0));
            } else {
                for p in 0..n_parents {
                    let a = if (200..=205).contains(&m) {
                        0u8
                    } else {
                        distinct_founder_allele(m, p)
                    };
                    joint.set_allele(m, p, Some(a));
                }
                joint.set_allele(m, n_parents, joint.allele(m, 0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            1,
            "common-allele correlated noise burst must not splice a false introgression, got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
    }

    #[test]
    fn short_introgression_breakpoints_use_physical_midpoint_under_uneven_density() {
        // Non-uniform marker spacing around a 4-diagnostic-SNP P1 introgression:
        //   - Left flank mismatch p=199 at 6.00 Mb; markers 200..203 tightly clustered
        //     at 6.005..6.020 Mb; first diagnostic m_lo=204 at 6.20 Mb.
        //     Physical midpoint is 6.10 Mb -> splice_start must map to marker 204
        //     (6.20 Mb), NOT index midpoint marker 202 (6.015 Mb).
        //   - Last diagnostic m_hi=210 at 6.38 Mb; markers 211..214 tightly clustered
        //     at 6.560..6.575 Mb; right flank mismatch n=215 at 6.58 Mb.
        //     Physical midpoint is 6.48 Mb -> splice_end must map to marker 211
        //     (6.560 Mb), NOT index midpoint marker 213 (6.570 Mb).
        let n_parents = 12usize;
        let n_markers = 400usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let mut pos = vec![0u64; n_markers];
        for m in 0..n_markers {
            pos[m] = if m < 200 {
                (m as u64 + 1) * 30_000
            } else if m < 204 {
                6_000_000 + ((m - 199) as u64) * 5_000
            } else if m <= 210 {
                6_200_000 + ((m - 204) as u64) * 30_000
            } else if m < 215 {
                6_560_000 + ((m - 211) as u64) * 5_000
            } else if m == 215 {
                6_580_000
            } else {
                6_580_000 + ((m - 215) as u64) * 30_000
            };
        }
        let cm: Vec<f64> = pos.iter().map(|&bp| bp as f64 * 1e-6).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
            }
            let p0 = joint.allele(m, 0).unwrap();
            if (200..215).contains(&m) {
                let diag = m == 204 || m == 206 || m == 208 || m == 210;
                let p1 = if diag { 1 - p0 } else { p0 };
                joint.set_allele(m, 1, Some(p1));
                joint.set_allele(m, n_parents, Some(p1));
            } else {
                if m == 199 || m == 215 {
                    joint.set_allele(m, 1, Some(1 - p0));
                }
                joint.set_allele(m, n_parents, Some(p0));
            }
        }

        let cfg = HaploidIbdConfig::default();
        let (segs, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg);
        assert_eq!(
            segs.len(),
            3,
            "uneven-density 4-diagnostic introgression must be recovered as [P0, P1, P0], got {segs:?}"
        );
        assert_eq!(segs[0].parent_sample, 0);
        assert_eq!(segs[1].parent_sample, 1);
        assert_eq!(segs[2].parent_sample, 0);
        assert_eq!(segs[1].start_bp, pos[204]);
        assert_eq!(segs[1].end_bp + 1, pos[211]);
        assert_eq!(segs[0].end_bp + 1, segs[1].start_bp);
        assert_eq!(segs[1].end_bp + 1, segs[2].start_bp);
    }

    #[test]
    fn short_introgression_tie_count_includes_all_zero_mismatch_candidates() {
        // P1 satisfies all obs_diag/alt_comp/alt_raw/ll_gain thresholds on a 3-diagnostic
        // window [201..=206]. When P2, P3, P4 are partially observed (alt_comp=2, alt_mis=0)
        // and fail obs_diag >= 3, they must still count toward the zero-mismatch parent tie
        // limit (total 4 > 3), preventing P1 from being spliced. When P2..P4 carry a mismatch,
        // total zero-mismatch parents drops to 1 and P1 is recovered.
        let n_parents = 12usize;
        let n_markers = 400usize;
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 31_250).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.025).collect();
        let cfg = HaploidIbdConfig::default();

        let run = |extra_zero_mis_ties: bool| {
            let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
            let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
            let mut joint =
                BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
            for m in 0..n_markers {
                for p in 0..n_parents {
                    joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
                }
                let p0 = joint.allele(m, 0).unwrap();
                if (200..208).contains(&m) {
                    let diag = m == 201 || m == 204 || m == 206;
                    let p1 = if diag { 1 - p0 } else { p0 };
                    joint.set_allele(m, 1, Some(p1));
                    joint.set_allele(m, n_parents, Some(p1));
                    if (201..=206).contains(&m) {
                        for p in 2..=4usize {
                            if m == 202 || m == 203 {
                                let a = if extra_zero_mis_ties { p1 } else { 1 - p1 };
                                joint.set_allele(m, p, Some(a));
                            } else {
                                joint.set_allele(m, p, None);
                            }
                        }
                    }
                } else {
                    if m == 199 || m == 208 {
                        joint.set_allele(m, 1, Some(1 - p0));
                    }
                    joint.set_allele(m, n_parents, Some(p0));
                }
            }
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &cfg).0
        };

        let segs_unique = run(false);
        assert_eq!(
            segs_unique.len(),
            3,
            "with only 1 zero-mismatch parent, P1 must be recovered, got {segs_unique:?}"
        );
        assert_eq!(segs_unique[1].parent_sample, 1);

        let segs_tied = run(true);
        assert_eq!(
            segs_tied.len(),
            1,
            "3 additional low-information zero-mismatch parents (total 4 > 3) must block splice, got {segs_tied:?}"
        );
        assert_eq!(segs_tied[0].parent_sample, 0);
    }

    #[test]
    fn higher_ancestry_error_rate_retains_noisy_in_panel_segment_without_uncalled() {
        // 12 balanced founders over 300 markers; target matches P0 with 14% uniform
        // error (42/300). At default ancestry_error_rate=0.0025 (delta=5.99), per-site
        // expected LLR ~0.69 - 0.14*5.99 = -0.15 < 0 so Viterbi drops P0 to Uncalled.
        // Raising ancestry_error_rate=0.02 (delta=3.89) makes expected LLR ~+0.13 > 0
        // (while 0.14 < 3 * 0.05 stays below B3), so P0 is retained across the chromosome.
        let n_parents = 12usize;
        let n_markers = 300usize;
        let ref_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n_parents]);
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true]);
        let mut joint =
            BitHaplotypeMatrix::new(ref_layout.total_haps + tgt_layout.total_haps, n_markers);
        let pos: Vec<u64> = (0..n_markers).map(|m| (m as u64 + 1) * 10_000).collect();
        let cm: Vec<f64> = (0..n_markers).map(|m| m as f64 * 0.05).collect();

        for m in 0..n_markers {
            for p in 0..n_parents {
                joint.set_allele(m, p, Some(balanced_founder_allele(m, p)));
            }
            let p0 = joint.allele(m, 0).unwrap();
            let flip = (m % 50) % 7 == 0 && (m % 50) < 49; // 14% uniform mismatch (42/300)
            joint.set_allele(m, n_parents, Some(if flip { 1 - p0 } else { p0 }));
        }

        let strict_cfg = HaploidIbdConfig::default();
        let (segs_strict, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &strict_cfg);
        assert!(
            segs_strict.is_empty(),
            "14% uniform mismatch must fall to Uncalled under strict 0.0025 error rate, got {segs_strict:?}"
        );

        let tolerant_cfg = HaploidIbdConfig {
            ancestry_error_rate: 0.02,
            ..HaploidIbdConfig::default()
        };
        let (segs_tolerant, _) =
            paint_chromosome_ancestry(&joint, &ref_layout, &tgt_layout, 0, &pos, &cm, &tolerant_cfg);
        assert_eq!(
            segs_tolerant.len(),
            1,
            "ancestry_error_rate=0.02 must retain 14%-error in-panel P0, got {segs_tolerant:?}"
        );
        assert_eq!(segs_tolerant[0].parent_sample, 0);
    }
}
