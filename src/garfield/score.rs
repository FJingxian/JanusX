#![allow(dead_code)]

use crate::bitwise::{and_popcount, popcount};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;

const BATCH_PAR_MIN_ROWS: usize = 64;
#[cfg(target_arch = "x86_64")]
const SUM_Y_MASKLOAD_MIN_POPCNT: u32 = 10;
#[cfg(target_arch = "aarch64")]
const SUM_Y_NEON_MIN_POPCNT: u32 = 14;

#[cfg(target_arch = "x86_64")]
const SUM_Y_MASK4_LUT: [[i64; 4]; 16] = [
    [0, 0, 0, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [-1, -1, 0, 0],
    [0, 0, -1, 0],
    [-1, 0, -1, 0],
    [0, -1, -1, 0],
    [-1, -1, -1, 0],
    [0, 0, 0, -1],
    [-1, 0, 0, -1],
    [0, -1, 0, -1],
    [-1, -1, 0, -1],
    [0, 0, -1, -1],
    [-1, 0, -1, -1],
    [0, -1, -1, -1],
    [-1, -1, -1, -1],
];

#[cfg(target_arch = "aarch64")]
const SUM_Y_MASK2_LUT: [[u64; 2]; 4] = [[0, 0], [u64::MAX, 0], [0, u64::MAX], [u64::MAX, u64::MAX]];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuousRuleScore {
    pub score: f64,
    pub raw_score: f64,
    pub mean_hit: f64,
    pub mean_miss: f64,
    pub support_frac: f64,
    pub dosage_maf: f64,
    pub n_hit: usize,
    pub n_ge2: usize,
    pub n_miss: usize,
}

#[inline]
fn words_for_samples(n: usize) -> usize {
    n.div_ceil(64)
}

#[inline]
fn should_parallel_rows(n_rows: usize) -> bool {
    n_rows >= BATCH_PAR_MIN_ROWS && rayon::current_num_threads() > 1
}

#[inline]
fn require_n_valid(n: usize, y_len: usize, bit_words: usize, ctx: &str) -> Result<(), String> {
    if n > y_len {
        return Err(format!("{ctx}: n_samples={} exceeds y length={}", n, y_len));
    }
    let bit_cap = bit_words.saturating_mul(64);
    if n > bit_cap {
        return Err(format!(
            "{ctx}: n_samples={} exceeds bit capacity={} ({} words)",
            n, bit_cap, bit_words
        ));
    }
    Ok(())
}

#[inline]
fn validate_binary_y(y: &[u8], n_samples: usize, ctx: &str) -> Result<(), String> {
    if let Some((idx, bad)) = y.iter().take(n_samples).enumerate().find(|(_, v)| **v > 1) {
        return Err(format!("{ctx}: y must be binary 0/1; found y[{idx}]={bad}"));
    }
    Ok(())
}

#[inline]
pub fn validate_continuous_y(y: &[f64], n_samples: usize, ctx: &str) -> Result<(), String> {
    if y.len() < n_samples {
        return Err(format!(
            "{ctx}: n_samples={} exceeds y length={}",
            n_samples,
            y.len()
        ));
    }
    if let Some((idx, bad)) = y
        .iter()
        .take(n_samples)
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(format!(
            "{ctx}: y must be finite for continuous mode; found y[{idx}]={bad}"
        ));
    }
    Ok(())
}

#[inline]
fn invalid_rule_score() -> ContinuousRuleScore {
    ContinuousRuleScore {
        score: f64::NEG_INFINITY,
        raw_score: f64::NEG_INFINITY,
        mean_hit: f64::NAN,
        mean_miss: f64::NAN,
        support_frac: f64::NAN,
        dosage_maf: f64::NAN,
        n_hit: 0,
        n_ge2: 0,
        n_miss: 0,
    }
}

#[inline]
fn minor_allele_frac(af: f64) -> f64 {
    if !af.is_finite() {
        return 0.0;
    }
    let af = af.clamp(0.0, 1.0);
    af.min(1.0 - af)
}

#[inline]
pub fn binary_maf_from_n_hit(n_samples: usize, n_hit: usize) -> f64 {
    if n_samples == 0 {
        return 0.0;
    }
    minor_allele_frac((n_hit as f64) / (n_samples as f64))
}

#[inline]
pub fn dosage_maf_from_dual_counts(n_samples: usize, n_ge1: usize, n_ge2: usize) -> f64 {
    if n_samples == 0 {
        return 0.0;
    }
    minor_allele_frac(((n_ge1.saturating_add(n_ge2)) as f64) / (2.0 * (n_samples as f64)))
}

#[inline]
fn pack_binary_y_to_bits(y: &[u8], n_samples: usize) -> (Vec<u64>, u64) {
    let words = words_for_samples(n_samples).max(1);
    let mut out = vec![0u64; words];
    let mut y_pos = 0u64;
    for (i, &yv) in y.iter().take(n_samples).enumerate() {
        if yv != 0 {
            out[i >> 6] |= 1u64 << (i & 63);
            y_pos += 1;
        }
    }
    (out, y_pos)
}

#[inline]
fn masked_xpos_tp(bits: &[u64], y_bits: &[u64], n_samples: usize) -> (u64, u64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;

    let mut x_pos = 0u64;
    let mut tp = 0u64;

    if full_words > 0 {
        x_pos += popcount(&bits[..full_words]);
        tp += and_popcount(&bits[..full_words], &y_bits[..full_words]);
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let xb = bits[full_words] & mask;
        let yb = y_bits[full_words] & mask;
        x_pos += xb.count_ones() as u64;
        tp += (xb & yb).count_ones() as u64;
    }

    (x_pos, tp)
}

#[inline]
fn confusion_from_packed_ybits(
    bits: &[u64],
    y_bits: &[u64],
    y_pos: u64,
    n_samples: usize,
) -> (u64, u64, u64, u64) {
    let (x_pos, tp) = masked_xpos_tp(bits, y_bits, n_samples);
    let fp = x_pos.saturating_sub(tp);
    let fnv = y_pos.saturating_sub(tp);
    let tn = (n_samples as u64).saturating_sub(tp + fp + fnv);
    (tp, tn, fp, fnv)
}

#[inline]
fn ba_from_confusion(tp: u64, tn: u64, fp: u64, fnv: u64) -> f64 {
    let pos = tp + fnv;
    let neg = tn + fp;
    match (pos, neg) {
        (0, 0) => f64::NAN,
        (0, _) => (tn as f64) / (neg as f64),
        (_, 0) => (tp as f64) / (pos as f64),
        _ => {
            let tpr = (tp as f64) / (pos as f64);
            let tnr = (tn as f64) / (neg as f64);
            0.5 * (tpr + tnr)
        }
    }
}

#[inline]
fn mcc_from_confusion(tp: u64, tn: u64, fp: u64, fnv: u64) -> f64 {
    let num = (tp as f64) * (tn as f64) - (fp as f64) * (fnv as f64);
    let a = (tp + fp) as f64;
    let b = (tp + fnv) as f64;
    let c = (tn + fp) as f64;
    let d = (tn + fnv) as f64;
    let den = (a * b * c * d).sqrt();
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

#[inline]
fn total_y_sum(y: &[f64], n_samples: usize) -> f64 {
    y.iter().take(n_samples).copied().sum::<f64>()
}

#[inline]
fn y_sum_and_sq(y: &[f64], n_samples: usize) -> (f64, f64) {
    let mut sy = 0.0_f64;
    let mut syy = 0.0_f64;
    for &v in y.iter().take(n_samples) {
        sy += v;
        syy += v * v;
    }
    (sy, syy)
}

#[inline]
fn count_sum_y_where_bit1(bits: &[u64], y: &[f64], n_samples: usize) -> (usize, f64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;

    let mut n1 = 0usize;
    let mut s1 = 0.0_f64;

    for (w_idx, &word) in bits.iter().take(full_words).enumerate() {
        n1 += word.count_ones() as usize;
        let mut w = word;
        let base = w_idx << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            s1 += y[base + tz];
            w &= w - 1;
        }
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let mut w = bits[full_words] & mask;
        n1 += w.count_ones() as usize;
        let base = full_words << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            s1 += y[base + tz];
            w &= w - 1;
        }
    }

    (n1, s1)
}

#[inline]
fn sum_y_where_bit1(bits: &[u64], y: &[f64], n_samples: usize) -> f64 {
    count_sum_y_where_bit1(bits, y, n_samples).1
}

#[inline]
fn count_sum_y_where_dual_packed(
    ge1_bits: &[u64],
    ge2_bits: &[u64],
    y: &[f64],
    n_samples: usize,
) -> (usize, usize, f64, f64) {
    let (n_ge1, sum_ge1) = count_sum_y_where_bit1(ge1_bits, y, n_samples);
    let (n_ge2, sum_ge2) = count_sum_y_where_bit1(ge2_bits, y, n_samples);
    (n_ge1, n_ge2, sum_ge1, sum_ge2)
}

#[inline]
pub fn dual_packed_summary(
    ge1_bits: &[u64],
    ge2_bits: &[u64],
    y: &[f64],
    n_samples: usize,
) -> (usize, usize, f64, f64) {
    count_sum_y_where_dual_packed(ge1_bits, ge2_bits, y, n_samples)
}

#[inline]
fn cont_mean_diff_corr_from_row(
    y: &[f64],
    bits: &[u64],
    n_samples: usize,
    y_sum: f64,
    y_sq_sum: f64,
) -> (f64, f64) {
    let (n1_usize, s1) = count_sum_y_where_bit1(bits, y, n_samples);
    let n1 = n1_usize as u64;
    let n0 = (n_samples as u64).saturating_sub(n1);

    let mean_diff = if n1 == 0 || n0 == 0 {
        f64::NAN
    } else {
        let mean1 = s1 / (n1 as f64);
        let mean0 = (y_sum - s1) / (n0 as f64);
        mean1 - mean0
    };

    let n = n_samples as f64;
    let sx = n1 as f64;
    let sxy = s1;

    let vxx_num = n * sx - sx * sx;
    let vyy_num = n * y_sq_sum - y_sum * y_sum;
    let corr = if !(vxx_num > 0.0 && vyy_num > 0.0) {
        0.0
    } else {
        let cov_num = n * sxy - sx * y_sum;
        (cov_num / (vxx_num.sqrt() * vyy_num.sqrt())).clamp(-1.0, 1.0)
    };

    (mean_diff, corr)
}

#[inline]
pub fn support_size_packed(bits: &[u64], n_samples: usize) -> usize {
    if n_samples == 0 || bits.is_empty() {
        return 0usize;
    }
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let mut n1 = if full_words > 0 {
        popcount(&bits[..full_words]) as usize
    } else {
        0usize
    };
    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        n1 += (bits[full_words] & mask).count_ones() as usize;
    }
    n1
}

#[inline]
fn sum_y_from_word_scalar(y: &[f64], base: usize, mut word: u64) -> f64 {
    let mut s = 0.0_f64;
    while word != 0 {
        let tz = word.trailing_zeros() as usize;
        s += y[base + tz];
        word &= word - 1;
    }
    s
}

#[inline]
fn sum_y_where_both1_scalar(lhs: &[u64], rhs: &[u64], y: &[f64], n_samples: usize) -> f64 {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;

    let mut s1 = 0.0_f64;

    for w_idx in 0..full_words {
        let base = w_idx << 6;
        s1 += sum_y_from_word_scalar(y, base, lhs[w_idx] & rhs[w_idx]);
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let base = full_words << 6;
        s1 += sum_y_from_word_scalar(y, base, (lhs[full_words] & rhs[full_words]) & mask);
    }

    s1
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_y_where_both1_avx2(lhs: &[u64], rhs: &[u64], y: &[f64], n_samples: usize) -> f64 {
    use core::arch::x86_64::{
        __m256d, __m256i, _mm256_add_pd, _mm256_loadu_pd, _mm256_loadu_si256, _mm256_maskload_pd,
        _mm256_setzero_pd, _mm256_storeu_pd,
    };

    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let y_ptr = y.as_ptr();
    let mut scalar_tail = 0.0_f64;
    let mut acc0: __m256d = _mm256_setzero_pd();
    let mut acc1: __m256d = _mm256_setzero_pd();
    let mut acc2: __m256d = _mm256_setzero_pd();
    let mut acc3: __m256d = _mm256_setzero_pd();

    let mut accumulate_word = |base: usize, word: u64, valid_bits: usize| {
        if word == 0 {
            return;
        }
        if word.count_ones() <= SUM_Y_MASKLOAD_MIN_POPCNT {
            scalar_tail += sum_y_from_word_scalar(y, base, word);
            return;
        }
        let blocks = valid_bits.div_ceil(4);
        for block_idx in 0..blocks {
            let nib = ((word >> (block_idx << 2)) & 0xFu64) as usize;
            if nib == 0 {
                continue;
            }
            let ptr = unsafe { y_ptr.add(base + (block_idx << 2)) };
            let vals = if nib == 0xF {
                unsafe { _mm256_loadu_pd(ptr) }
            } else {
                let mask =
                    unsafe { _mm256_loadu_si256(SUM_Y_MASK4_LUT[nib].as_ptr().cast::<__m256i>()) };
                unsafe { _mm256_maskload_pd(ptr, mask) }
            };
            match block_idx & 3 {
                0 => acc0 = _mm256_add_pd(acc0, vals),
                1 => acc1 = _mm256_add_pd(acc1, vals),
                2 => acc2 = _mm256_add_pd(acc2, vals),
                _ => acc3 = _mm256_add_pd(acc3, vals),
            }
        }
    };

    for w_idx in 0..full_words {
        accumulate_word(w_idx << 6, lhs[w_idx] & rhs[w_idx], 64);
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        accumulate_word(
            full_words << 6,
            (lhs[full_words] & rhs[full_words]) & mask,
            rem,
        );
    }

    let acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    let mut lanes = [0.0_f64; 4];
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    scalar_tail + lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_y_where_both1_neon(lhs: &[u64], rhs: &[u64], y: &[f64], n_samples: usize) -> f64 {
    use core::arch::aarch64::{
        float64x2_t, uint64x2_t, vaddq_f64, vandq_u64, vdupq_n_f64, vld1q_f64, vld1q_u64,
        vreinterpretq_f64_u64, vreinterpretq_u64_f64, vst1q_f64,
    };

    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let y_ptr = y.as_ptr();
    let mut scalar_tail = 0.0_f64;
    let mut acc0: float64x2_t = vdupq_n_f64(0.0);
    let mut acc1: float64x2_t = vdupq_n_f64(0.0);
    let mut acc2: float64x2_t = vdupq_n_f64(0.0);
    let mut acc3: float64x2_t = vdupq_n_f64(0.0);

    let mut accumulate_pair = |pair_idx: usize, ptr: *const f64, bits2: usize| {
        if bits2 == 0 {
            return;
        }
        let vals = if bits2 == 0x3 {
            unsafe { vld1q_f64(ptr) }
        } else {
            let mask: uint64x2_t = unsafe { vld1q_u64(SUM_Y_MASK2_LUT[bits2].as_ptr()) };
            let raw = unsafe { vld1q_f64(ptr) };
            vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(raw), mask))
        };
        match pair_idx & 3 {
            0 => acc0 = vaddq_f64(acc0, vals),
            1 => acc1 = vaddq_f64(acc1, vals),
            2 => acc2 = vaddq_f64(acc2, vals),
            _ => acc3 = vaddq_f64(acc3, vals),
        }
    };

    for w_idx in 0..full_words {
        let word = lhs[w_idx] & rhs[w_idx];
        if word == 0 {
            continue;
        }
        let base = w_idx << 6;
        if word.count_ones() <= SUM_Y_NEON_MIN_POPCNT {
            scalar_tail += sum_y_from_word_scalar(y, base, word);
            continue;
        }
        for block_idx in 0..16usize {
            let nib = ((word >> (block_idx << 2)) & 0xFu64) as usize;
            if nib == 0 {
                continue;
            }
            let block_ptr = unsafe { y_ptr.add(base + (block_idx << 2)) };
            let low2 = nib & 0x3;
            let high2 = nib >> 2;
            accumulate_pair(block_idx << 1, block_ptr, low2);
            if high2 != 0 {
                accumulate_pair((block_idx << 1) + 1, unsafe { block_ptr.add(2) }, high2);
            }
        }
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        scalar_tail += sum_y_from_word_scalar(
            y,
            full_words << 6,
            (lhs[full_words] & rhs[full_words]) & mask,
        );
    }

    let acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
    let mut lanes = [0.0_f64; 2];
    unsafe { vst1q_f64(lanes.as_mut_ptr(), acc) };
    scalar_tail + lanes[0] + lanes[1]
}

#[inline]
pub fn sum_y_where_both1(lhs: &[u64], rhs: &[u64], y: &[f64], n_samples: usize) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        // The current NEON kernel underperforms the scalar path on Apple Silicon
        // for both sparse and dense supports, so keep AArch64 on scalar for now.
        sum_y_where_both1_scalar(lhs, rhs, y, n_samples)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { sum_y_where_both1_avx2(lhs, rhs, y, n_samples) };
        }
        sum_y_where_both1_scalar(lhs, rhs, y, n_samples)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        sum_y_where_both1_scalar(lhs, rhs, y, n_samples)
    }
}

/// Balanced Accuracy for binary `y` (0/1) vs packed 0/1 bit-vector.
pub fn score_binary_ba_packed(y: &[u8], bits: &[u64], n_samples: usize) -> f64 {
    if n_samples == 0 {
        return f64::NAN;
    }
    if require_n_valid(n_samples, y.len(), bits.len(), "score_binary_ba_packed").is_err() {
        return f64::NAN;
    }

    let (y_bits, y_pos) = pack_binary_y_to_bits(y, n_samples);
    let (tp, tn, fp, fnv) = confusion_from_packed_ybits(bits, &y_bits, y_pos, n_samples);
    ba_from_confusion(tp, tn, fp, fnv)
}

/// Matthews Correlation Coefficient for binary `y` (0/1) vs packed 0/1 bit-vector.
pub fn score_binary_mcc_packed(y: &[u8], bits: &[u64], n_samples: usize) -> f64 {
    if n_samples == 0 {
        return f64::NAN;
    }
    if require_n_valid(n_samples, y.len(), bits.len(), "score_binary_mcc_packed").is_err() {
        return f64::NAN;
    }

    let (y_bits, y_pos) = pack_binary_y_to_bits(y, n_samples);
    let (tp, tn, fp, fnv) = confusion_from_packed_ybits(bits, &y_bits, y_pos, n_samples);
    mcc_from_confusion(tp, tn, fp, fnv)
}

/// Mean difference (`mean(x=1) - mean(x=0)`) for continuous `y` vs packed 0/1 bit-vector.
pub fn score_cont_mean_diff_packed(y: &[f64], bits: &[u64], n_samples: usize) -> f64 {
    if n_samples == 0 {
        return f64::NAN;
    }
    if require_n_valid(
        n_samples,
        y.len(),
        bits.len(),
        "score_cont_mean_diff_packed",
    )
    .is_err()
    {
        return f64::NAN;
    }

    let (sy, syy) = y_sum_and_sq(y, n_samples);
    let (md, _r) = cont_mean_diff_corr_from_row(y, bits, n_samples, sy, syy);
    md
}

/// Pearson correlation between continuous `y` and packed 0/1 bit-vector.
pub fn score_cont_corr_packed(y: &[f64], bits: &[u64], n_samples: usize) -> f64 {
    if n_samples == 0 {
        return f64::NAN;
    }
    if require_n_valid(n_samples, y.len(), bits.len(), "score_cont_corr_packed").is_err() {
        return f64::NAN;
    }

    let (sy, syy) = y_sum_and_sq(y, n_samples);
    let (_md, r) = cont_mean_diff_corr_from_row(y, bits, n_samples, sy, syy);
    r
}

/// Batch scoring for binary `y` against many packed bit-vectors.
///
/// `bits_flat` is row-major with `n_rows` rows and `row_words` words per row.
/// Returns `(ba, mcc)`, both length `n_rows`.
pub fn score_binary_ba_mcc_batch_packed(
    y: &[u8],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> (Vec<f64>, Vec<f64>) {
    if n_rows == 0 {
        return (Vec::new(), Vec::new());
    }

    let needed_words = words_for_samples(n_samples).max(1);
    if n_samples == 0
        || require_n_valid(
            n_samples,
            y.len(),
            needed_words,
            "score_binary_ba_mcc_batch_packed",
        )
        .is_err()
        || row_words < needed_words
        || bits_flat.len() < n_rows.saturating_mul(row_words)
    {
        return (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows]);
    }

    let (y_bits, y_pos) = pack_binary_y_to_bits(y, n_samples);

    let pairs: Vec<(f64, f64)> = if should_parallel_rows(n_rows) {
        (0..n_rows)
            .into_par_iter()
            .map(|r| {
                let start = r * row_words;
                let row = &bits_flat[start..start + needed_words];
                let (tp, tn, fp, fnv) = confusion_from_packed_ybits(row, &y_bits, y_pos, n_samples);
                (
                    ba_from_confusion(tp, tn, fp, fnv),
                    mcc_from_confusion(tp, tn, fp, fnv),
                )
            })
            .collect()
    } else {
        (0..n_rows)
            .map(|r| {
                let start = r * row_words;
                let row = &bits_flat[start..start + needed_words];
                let (tp, tn, fp, fnv) = confusion_from_packed_ybits(row, &y_bits, y_pos, n_samples);
                (
                    ba_from_confusion(tp, tn, fp, fnv),
                    mcc_from_confusion(tp, tn, fp, fnv),
                )
            })
            .collect()
    };

    pairs.into_iter().unzip()
}

/// Batch scoring for continuous `y` against many packed bit-vectors.
///
/// `bits_flat` is row-major with `n_rows` rows and `row_words` words per row.
/// Returns `(mean_diff, corr)`, both length `n_rows`.
pub fn score_cont_mean_diff_corr_batch_packed(
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> (Vec<f64>, Vec<f64>) {
    if n_rows == 0 {
        return (Vec::new(), Vec::new());
    }

    let needed_words = words_for_samples(n_samples).max(1);
    if n_samples == 0
        || require_n_valid(
            n_samples,
            y.len(),
            needed_words,
            "score_cont_mean_diff_corr_batch_packed",
        )
        .is_err()
        || row_words < needed_words
        || bits_flat.len() < n_rows.saturating_mul(row_words)
    {
        return (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows]);
    }

    let (sy, syy) = y_sum_and_sq(y, n_samples);

    let pairs: Vec<(f64, f64)> = if should_parallel_rows(n_rows) {
        (0..n_rows)
            .into_par_iter()
            .map(|r| {
                let start = r * row_words;
                let row = &bits_flat[start..start + needed_words];
                cont_mean_diff_corr_from_row(y, row, n_samples, sy, syy)
            })
            .collect()
    } else {
        (0..n_rows)
            .map(|r| {
                let start = r * row_words;
                let row = &bits_flat[start..start + needed_words];
                cont_mean_diff_corr_from_row(y, row, n_samples, sy, syy)
            })
            .collect()
    };

    pairs.into_iter().unzip()
}

#[inline]
pub fn score_cont_centered_gain_from_sum_and_n_hit(
    total_sum: f64,
    sum_hit: f64,
    n_samples: usize,
    n_hit: usize,
) -> ContinuousRuleScore {
    let n_miss = n_samples.saturating_sub(n_hit);
    if n_hit == 0 || n_miss == 0 {
        return ContinuousRuleScore {
            score: 0.0,
            raw_score: 0.0,
            mean_hit: f64::NAN,
            mean_miss: f64::NAN,
            support_frac: (n_hit as f64) / (n_samples as f64),
            dosage_maf: binary_maf_from_n_hit(n_samples, n_hit),
            n_hit,
            n_ge2: 0,
            n_miss,
        };
    }

    let mean_hit = sum_hit / (n_hit as f64);
    let sum_miss = total_sum - sum_hit;
    let mean_miss = sum_miss / (n_miss as f64);
    let p = (n_hit as f64) / (n_samples as f64);
    let centered_sum_hit = sum_hit - p * total_sum;
    let raw = (n_samples as f64) * centered_sum_hit * centered_sum_hit
        / ((n_hit as f64) * (n_miss as f64));
    ContinuousRuleScore {
        score: raw,
        raw_score: raw,
        mean_hit,
        mean_miss,
        support_frac: p,
        dosage_maf: binary_maf_from_n_hit(n_samples, n_hit),
        n_hit,
        n_ge2: 0,
        n_miss,
    }
}

pub fn score_cont_centered_gain_packed_with_n_hit(
    y: &[f64],
    bits: &[u64],
    n_samples: usize,
    total_sum: f64,
    n_hit: usize,
) -> ContinuousRuleScore {
    const CTX: &str = "score_cont_centered_gain_packed_with_n_hit";
    if n_samples == 0 || require_n_valid(n_samples, y.len(), bits.len(), CTX).is_err() {
        return invalid_rule_score();
    }
    if validate_continuous_y(y, n_samples, CTX).is_err() {
        return invalid_rule_score();
    }
    if n_hit > n_samples {
        return invalid_rule_score();
    }

    let sum_hit = sum_y_where_bit1(bits, y, n_samples);
    score_cont_centered_gain_from_sum_and_n_hit(total_sum, sum_hit, n_samples, n_hit)
}

pub fn score_cont_centered_gain_packed_with_sum(
    y: &[f64],
    bits: &[u64],
    n_samples: usize,
    total_sum: f64,
) -> ContinuousRuleScore {
    const CTX: &str = "score_cont_centered_gain_packed_with_sum";
    if n_samples == 0 || require_n_valid(n_samples, y.len(), bits.len(), CTX).is_err() {
        return invalid_rule_score();
    }
    if validate_continuous_y(y, n_samples, CTX).is_err() {
        return invalid_rule_score();
    }

    let n_hit = support_size_packed(bits, n_samples);
    score_cont_centered_gain_packed_with_n_hit(y, bits, n_samples, total_sum, n_hit)
}

pub fn score_cont_weighted_mean_diff_packed(
    y: &[f64],
    bits: &[u64],
    n_samples: usize,
) -> ContinuousRuleScore {
    score_cont_centered_gain_packed_with_sum(y, bits, n_samples, total_y_sum(y, n_samples))
}

#[inline]
pub fn score_cont_centered_gain_dual_from_summary(
    total_sum: f64,
    n_samples: usize,
    n_ge1: usize,
    n_ge2: usize,
    sum_ge1: f64,
    sum_ge2: f64,
) -> ContinuousRuleScore {
    let n_hit = n_ge1;
    let n_miss = n_samples.saturating_sub(n_hit);
    let support_frac = (n_hit as f64) / (n_samples as f64);
    let dosage_maf = dosage_maf_from_dual_counts(n_samples, n_ge1, n_ge2);
    let mean_hit = if n_hit == 0 {
        f64::NAN
    } else {
        sum_ge1 / (n_hit as f64)
    };
    let mean_miss = if n_miss == 0 {
        f64::NAN
    } else {
        (total_sum - sum_ge1) / (n_miss as f64)
    };

    let n = n_samples as f64;
    let sx = (n_ge1 + n_ge2) as f64;
    let sx2 = (n_ge1 + 3usize.saturating_mul(n_ge2)) as f64;
    let sxy = sum_ge1 + sum_ge2;
    let vxx_num = n * sx2 - sx * sx;
    let raw = if !(vxx_num > 0.0) {
        0.0
    } else {
        let cov_num = n * sxy - sx * total_sum;
        (cov_num * cov_num) / (n * vxx_num)
    };

    ContinuousRuleScore {
        score: raw,
        raw_score: raw,
        mean_hit,
        mean_miss,
        support_frac,
        dosage_maf,
        n_hit,
        n_ge2,
        n_miss,
    }
}

pub fn score_cont_centered_gain_dual_packed_with_sum(
    y: &[f64],
    ge1_bits: &[u64],
    ge2_bits: &[u64],
    n_samples: usize,
    total_sum: f64,
) -> ContinuousRuleScore {
    const CTX: &str = "score_cont_centered_gain_dual_packed_with_sum";
    if n_samples == 0
        || require_n_valid(n_samples, y.len(), ge1_bits.len(), CTX).is_err()
        || require_n_valid(n_samples, y.len(), ge2_bits.len(), CTX).is_err()
    {
        return invalid_rule_score();
    }
    if validate_continuous_y(y, n_samples, CTX).is_err() {
        return invalid_rule_score();
    }

    let (n_ge1, n_ge2, sum_ge1, sum_ge2) = dual_packed_summary(ge1_bits, ge2_bits, y, n_samples);
    score_cont_centered_gain_dual_from_summary(total_sum, n_samples, n_ge1, n_ge2, sum_ge1, sum_ge2)
}

pub fn score_cont_centered_gain_dual_packed(
    y: &[f64],
    ge1_bits: &[u64],
    ge2_bits: &[u64],
    n_samples: usize,
) -> ContinuousRuleScore {
    score_cont_centered_gain_dual_packed_with_sum(
        y,
        ge1_bits,
        ge2_bits,
        n_samples,
        total_y_sum(y, n_samples),
    )
}

#[inline]
fn readonly_u8_to_cow<'a>(arr: &'a PyReadonlyArray1<'a, u8>) -> Cow<'a, [u8]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[inline]
fn readonly_f64_to_cow<'a>(arr: &'a PyReadonlyArray1<'a, f64>) -> Cow<'a, [f64]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[inline]
fn readonly_u64_to_cow<'a>(arr: &'a PyReadonlyArray1<'a, u64>) -> Cow<'a, [u64]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[inline]
fn readonly_u64_2d_to_flat_cow<'a>(arr: &'a PyReadonlyArray2<'a, u64>) -> Cow<'a, [u64]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[pyfunction(name = "score_binary_ba")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_binary_ba_py(
    y: PyReadonlyArray1<'_, u8>,
    bits: PyReadonlyArray1<'_, u64>,
    n_samples: Option<usize>,
) -> PyResult<f64> {
    let yv = readonly_u8_to_cow(&y);
    let bv = readonly_u64_to_cow(&bits);
    let n = n_samples.unwrap_or(yv.len());
    require_n_valid(n, yv.len(), bv.len(), "score_binary_ba").map_err(PyRuntimeError::new_err)?;
    validate_binary_y(yv.as_ref(), n, "score_binary_ba").map_err(PyRuntimeError::new_err)?;
    Ok(score_binary_ba_packed(yv.as_ref(), bv.as_ref(), n))
}

#[pyfunction(name = "score_binary_mcc")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_binary_mcc_py(
    y: PyReadonlyArray1<'_, u8>,
    bits: PyReadonlyArray1<'_, u64>,
    n_samples: Option<usize>,
) -> PyResult<f64> {
    let yv = readonly_u8_to_cow(&y);
    let bv = readonly_u64_to_cow(&bits);
    let n = n_samples.unwrap_or(yv.len());
    require_n_valid(n, yv.len(), bv.len(), "score_binary_mcc").map_err(PyRuntimeError::new_err)?;
    validate_binary_y(yv.as_ref(), n, "score_binary_mcc").map_err(PyRuntimeError::new_err)?;
    Ok(score_binary_mcc_packed(yv.as_ref(), bv.as_ref(), n))
}

#[pyfunction(name = "score_cont_mean_diff")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_cont_mean_diff_py(
    y: PyReadonlyArray1<'_, f64>,
    bits: PyReadonlyArray1<'_, u64>,
    n_samples: Option<usize>,
) -> PyResult<f64> {
    let yv = readonly_f64_to_cow(&y);
    let bv = readonly_u64_to_cow(&bits);
    let n = n_samples.unwrap_or(yv.len());
    require_n_valid(n, yv.len(), bv.len(), "score_cont_mean_diff")
        .map_err(PyRuntimeError::new_err)?;
    Ok(score_cont_mean_diff_packed(yv.as_ref(), bv.as_ref(), n))
}

#[pyfunction(name = "score_cont_corr")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_cont_corr_py(
    y: PyReadonlyArray1<'_, f64>,
    bits: PyReadonlyArray1<'_, u64>,
    n_samples: Option<usize>,
) -> PyResult<f64> {
    let yv = readonly_f64_to_cow(&y);
    let bv = readonly_u64_to_cow(&bits);
    let n = n_samples.unwrap_or(yv.len());
    require_n_valid(n, yv.len(), bv.len(), "score_cont_corr").map_err(PyRuntimeError::new_err)?;
    Ok(score_cont_corr_packed(yv.as_ref(), bv.as_ref(), n))
}

#[pyfunction(name = "score_binary_ba_mcc_batch")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_binary_ba_mcc_batch_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, u8>,
    bits: PyReadonlyArray2<'py, u64>,
    n_samples: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let yv = readonly_u8_to_cow(&y);
    let bits_arr = bits.as_array();
    if bits_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "score_binary_ba_mcc_batch: bits must be 2D",
        ));
    }

    let n_rows = bits_arr.shape()[0];
    let row_words = bits_arr.shape()[1];
    let n = n_samples.unwrap_or(yv.len());
    let needed_words = words_for_samples(n).max(1);

    require_n_valid(n, yv.len(), needed_words, "score_binary_ba_mcc_batch")
        .map_err(PyRuntimeError::new_err)?;
    validate_binary_y(yv.as_ref(), n, "score_binary_ba_mcc_batch")
        .map_err(PyRuntimeError::new_err)?;

    if row_words < needed_words {
        return Err(PyRuntimeError::new_err(format!(
            "score_binary_ba_mcc_batch: bits row_words={} is smaller than required {} for n_samples={}",
            row_words, needed_words, n
        )));
    }

    let bv = readonly_u64_2d_to_flat_cow(&bits);
    let (ba, mcc) =
        score_binary_ba_mcc_batch_packed(yv.as_ref(), bv.as_ref(), row_words, n_rows, n);

    let out_ba = PyArray1::<f64>::zeros(py, [n_rows], false);
    let out_mcc = PyArray1::<f64>::zeros(py, [n_rows], false);

    let out_ba_slice = unsafe {
        out_ba.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("score_binary_ba_mcc_batch: out_ba not contiguous")
        })?
    };
    let out_mcc_slice = unsafe {
        out_mcc.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("score_binary_ba_mcc_batch: out_mcc not contiguous")
        })?
    };
    out_ba_slice.copy_from_slice(&ba);
    out_mcc_slice.copy_from_slice(&mcc);

    Ok((out_ba, out_mcc))
}

#[pyfunction(name = "score_cont_mean_diff_corr_batch")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_cont_mean_diff_corr_batch_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    bits: PyReadonlyArray2<'py, u64>,
    n_samples: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let yv = readonly_f64_to_cow(&y);
    let bits_arr = bits.as_array();
    if bits_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "score_cont_mean_diff_corr_batch: bits must be 2D",
        ));
    }

    let n_rows = bits_arr.shape()[0];
    let row_words = bits_arr.shape()[1];
    let n = n_samples.unwrap_or(yv.len());
    let needed_words = words_for_samples(n).max(1);

    require_n_valid(n, yv.len(), needed_words, "score_cont_mean_diff_corr_batch")
        .map_err(PyRuntimeError::new_err)?;

    if row_words < needed_words {
        return Err(PyRuntimeError::new_err(format!(
            "score_cont_mean_diff_corr_batch: bits row_words={} is smaller than required {} for n_samples={}",
            row_words, needed_words, n
        )));
    }

    let bv = readonly_u64_2d_to_flat_cow(&bits);
    let (md, corr) =
        score_cont_mean_diff_corr_batch_packed(yv.as_ref(), bv.as_ref(), row_words, n_rows, n);

    let out_md = PyArray1::<f64>::zeros(py, [n_rows], false);
    let out_corr = PyArray1::<f64>::zeros(py, [n_rows], false);

    let out_md_slice = unsafe {
        out_md.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("score_cont_mean_diff_corr_batch: out_md not contiguous")
        })?
    };
    let out_corr_slice = unsafe {
        out_corr.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("score_cont_mean_diff_corr_batch: out_corr not contiguous")
        })?
    };
    out_md_slice.copy_from_slice(&md);
    out_corr_slice.copy_from_slice(&corr);

    Ok((out_md, out_corr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    fn pack01(v: &[u8]) -> Vec<u64> {
        let words = v.len().div_ceil(64);
        let mut out = vec![0u64; words.max(1)];
        for (i, &b) in v.iter().enumerate() {
            if b != 0 {
                out[i >> 6] |= 1u64 << (i & 63);
            }
        }
        out
    }

    fn median_ns_per_call(samples_ns: &mut [f64]) -> f64 {
        samples_ns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        samples_ns[samples_ns.len() / 2]
    }

    fn make_sum_y_bench_inputs<F>(n_samples: usize, pred: F) -> (Vec<u64>, Vec<u64>, Vec<f64>)
    where
        F: Fn(usize) -> (bool, bool),
    {
        let mut lhs = vec![0u64; words_for_samples(n_samples).max(1)];
        let mut rhs = vec![0u64; words_for_samples(n_samples).max(1)];
        let mut y = Vec::<f64>::with_capacity(n_samples);
        for i in 0..n_samples {
            let (a, b) = pred(i);
            if a {
                lhs[i >> 6] |= 1u64 << (i & 63);
            }
            if b {
                rhs[i >> 6] |= 1u64 << (i & 63);
            }
            y.push(((i as f64) * 0.03125) - 7.0 + ((i % 13) as f64) * 0.0075);
        }
        (lhs, rhs, y)
    }

    fn bench_sum_y_backend<F>(
        lhs: &[u64],
        rhs: &[u64],
        y: &[f64],
        n_samples: usize,
        iters: usize,
        mut f: F,
    ) -> (f64, f64)
    where
        F: FnMut(&[u64], &[u64], &[f64], usize) -> f64,
    {
        black_box(f(lhs, rhs, y, n_samples));
        let mut samples = [0.0_f64; 5];
        let mut sink = 0.0_f64;
        for slot in samples.iter_mut() {
            let t0 = Instant::now();
            for _ in 0..iters {
                sink += black_box(f(lhs, rhs, y, n_samples));
            }
            let elapsed = t0.elapsed().as_secs_f64();
            *slot = elapsed * 1e9 / (iters as f64);
        }
        black_box(sink);
        let median = median_ns_per_call(samples.as_mut_slice());
        let checksum = f(lhs, rhs, y, n_samples);
        (median, checksum)
    }

    #[test]
    fn test_binary_scores_perfect() {
        let y = [0u8, 1, 1, 0, 1, 0, 0, 1];
        let b = pack01(&y);
        assert!((score_binary_ba_packed(&y, &b, y.len()) - 1.0).abs() < 1e-12);
        assert!((score_binary_mcc_packed(&y, &b, y.len()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_binary_scores_half_random() {
        let y = [0u8, 0, 1, 1];
        let x = [0u8, 1, 0, 1];
        let b = pack01(&x);
        assert!((score_binary_ba_packed(&y, &b, y.len()) - 0.5).abs() < 1e-12);
        assert!(score_binary_mcc_packed(&y, &b, y.len()).abs() < 1e-12);
    }

    #[test]
    fn test_binary_tail_mask_ignored() {
        let n = 70usize;
        let mut y = vec![0u8; n];
        let mut x = vec![0u8; n];
        for i in (0..n).step_by(3) {
            y[i] = 1;
        }
        for i in (1..n).step_by(4) {
            x[i] = 1;
        }
        let mut b = pack01(&x);
        b[1] |= !((1u64 << (n & 63)) - 1u64);

        let ba1 = score_binary_ba_packed(&y, &b, n);
        let mcc1 = score_binary_mcc_packed(&y, &b, n);

        let mut b_clean = b.clone();
        b_clean[1] &= (1u64 << (n & 63)) - 1u64;
        let ba2 = score_binary_ba_packed(&y, &b_clean, n);
        let mcc2 = score_binary_mcc_packed(&y, &b_clean, n);

        assert!((ba1 - ba2).abs() < 1e-12);
        assert!((mcc1 - mcc2).abs() < 1e-12);
    }

    #[test]
    fn test_cont_mean_diff_and_corr() {
        let y = [1.0_f64, 2.0, 3.0, 4.0];
        let x = [0u8, 0, 1, 1];
        let b = pack01(&x);
        let md = score_cont_mean_diff_packed(&y, &b, y.len());
        let r = score_cont_corr_packed(&y, &b, y.len());
        assert!((md - 2.0).abs() < 1e-12);
        assert!((r - 0.894_427_190_999_915_9).abs() < 1e-12);
    }

    #[test]
    fn test_cont_corr_constant_bit() {
        let y = [1.0_f64, 2.0, 3.0, 4.0];
        let x = [1u8, 1, 1, 1];
        let b = pack01(&x);
        assert_eq!(score_cont_corr_packed(&y, &b, y.len()), 0.0);
        assert!(score_cont_mean_diff_packed(&y, &b, y.len()).is_nan());
    }

    #[test]
    fn test_sum_y_where_both1_matches_scalar_reference() {
        let n = 137usize;
        let y = (0..n)
            .map(|i| ((i as f64) * 0.125) - 3.0 + ((i % 7) as f64) * 0.05)
            .collect::<Vec<_>>();
        let lhs_raw = (0..n)
            .map(|i| ((i % 3) == 0 || (i % 11) == 1 || (i >= 96 && i < 128)) as u8)
            .collect::<Vec<_>>();
        let rhs_raw = (0..n)
            .map(|i| ((i % 5) <= 1 || (i % 17) == 4 || (i >= 64 && i < 120)) as u8)
            .collect::<Vec<_>>();
        let lhs = pack01(lhs_raw.as_slice());
        let rhs = pack01(rhs_raw.as_slice());
        let expected = sum_y_where_both1_scalar(lhs.as_slice(), rhs.as_slice(), y.as_slice(), n);
        let got = sum_y_where_both1(lhs.as_slice(), rhs.as_slice(), y.as_slice(), n);
        assert!((got - expected).abs() < 1e-12);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sum_y_where_both1_avx2_matches_scalar_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 130usize;
        let y = (0..n)
            .map(|i| ((i as f64) * 0.2) - 5.0 + ((i % 9) as f64) * 0.03)
            .collect::<Vec<_>>();
        let lhs_raw = (0..n)
            .map(|i| ((i % 2) == 0 || (i >= 32 && i < 96)) as u8)
            .collect::<Vec<_>>();
        let rhs_raw = (0..n)
            .map(|i| ((i % 4) != 1 || (i >= 80 && i < 128)) as u8)
            .collect::<Vec<_>>();
        let lhs = pack01(lhs_raw.as_slice());
        let rhs = pack01(rhs_raw.as_slice());
        let expected = sum_y_where_both1_scalar(lhs.as_slice(), rhs.as_slice(), y.as_slice(), n);
        let got =
            unsafe { sum_y_where_both1_avx2(lhs.as_slice(), rhs.as_slice(), y.as_slice(), n) };
        assert!((got - expected).abs() < 1e-12);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sum_y_where_both1_neon_matches_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let n = 130usize;
        let y = (0..n)
            .map(|i| ((i as f64) * 0.2) - 5.0 + ((i % 9) as f64) * 0.03)
            .collect::<Vec<_>>();
        let lhs_raw = (0..n)
            .map(|i| ((i % 2) == 0 || (i >= 32 && i < 96)) as u8)
            .collect::<Vec<_>>();
        let rhs_raw = (0..n)
            .map(|i| ((i % 4) != 1 || (i >= 80 && i < 128)) as u8)
            .collect::<Vec<_>>();
        let lhs = pack01(lhs_raw.as_slice());
        let rhs = pack01(rhs_raw.as_slice());
        let expected = sum_y_where_both1_scalar(lhs.as_slice(), rhs.as_slice(), y.as_slice(), n);
        let got =
            unsafe { sum_y_where_both1_neon(lhs.as_slice(), rhs.as_slice(), y.as_slice(), n) };
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    #[ignore]
    fn bench_sum_y_where_both1_backends() {
        let n_samples = 65_536usize;
        let sparse = make_sum_y_bench_inputs(n_samples, |i| {
            let lhs = (i % 29) == 0 || (i % 31) == 1 || (i % 37) == 3;
            let rhs = (i % 41) == 0 || (i % 43) == 2 || (i % 47) == 4;
            (lhs, rhs)
        });
        let dense = make_sum_y_bench_inputs(n_samples, |i| {
            let lhs = ((i & 1) == 0) || ((i % 7) <= 2) || ((i % 19) <= 6);
            let rhs = ((i % 3) != 1) || ((i % 11) <= 4) || ((i % 23) <= 7);
            (lhs, rhs)
        });
        let iters = 512usize;

        let run_case = |label: &str, lhs: &[u64], rhs: &[u64], y: &[f64]| {
            let (scalar_ns, scalar_sum) =
                bench_sum_y_backend(lhs, rhs, y, n_samples, iters, sum_y_where_both1_scalar);
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    let (neon_ns, neon_sum) =
                        bench_sum_y_backend(lhs, rhs, y, n_samples, iters, |a, b, c, n| unsafe {
                            sum_y_where_both1_neon(a, b, c, n)
                        });
                    let diff = (scalar_sum - neon_sum).abs();
                    let tol = scalar_sum.abs().max(1.0) * 1e-10;
                    assert!(
                        diff <= tol,
                        "checksum drift too large: diff={diff} tol={tol}"
                    );
                    eprintln!(
                        "[sum_y bench][{label}] scalar={scalar_ns:.1} ns/call neon={neon_ns:.1} ns/call speedup={:.2}x diff={diff:.3e}",
                        scalar_ns / neon_ns,
                    );
                    return;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    let (avx2_ns, avx2_sum) =
                        bench_sum_y_backend(lhs, rhs, y, n_samples, iters, |a, b, c, n| unsafe {
                            sum_y_where_both1_avx2(a, b, c, n)
                        });
                    let diff = (scalar_sum - avx2_sum).abs();
                    let tol = scalar_sum.abs().max(1.0) * 1e-10;
                    assert!(
                        diff <= tol,
                        "checksum drift too large: diff={diff} tol={tol}"
                    );
                    eprintln!(
                        "[sum_y bench][{label}] scalar={scalar_ns:.1} ns/call avx2={avx2_ns:.1} ns/call speedup={:.2}x diff={diff:.3e}",
                        scalar_ns / avx2_ns,
                    );
                    return;
                }
            }
            eprintln!(
                "[sum_y bench][{label}] simd backend unavailable; scalar={scalar_ns:.1} ns/call"
            );
        };

        run_case(
            "sparse",
            sparse.0.as_slice(),
            sparse.1.as_slice(),
            sparse.2.as_slice(),
        );
        run_case(
            "dense",
            dense.0.as_slice(),
            dense.1.as_slice(),
            dense.2.as_slice(),
        );
    }

    #[test]
    fn test_binary_batch_matches_single() {
        let n = 100usize;
        let y: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
        let row_words = words_for_samples(n) + 1;
        let n_rows = 4usize;

        let mut bits_flat = vec![0u64; n_rows * row_words];
        for r in 0..n_rows {
            for i in 0..n {
                let b = (((i + r) % (r + 2)) == 0) as u8;
                if b != 0 {
                    bits_flat[r * row_words + (i >> 6)] |= 1u64 << (i & 63);
                }
            }
            bits_flat[r * row_words + row_words - 1] = u64::MAX;
        }

        let (ba, mcc) = score_binary_ba_mcc_batch_packed(&y, &bits_flat, row_words, n_rows, n);

        for r in 0..n_rows {
            let row = &bits_flat[r * row_words..r * row_words + words_for_samples(n)];
            let ba_s = score_binary_ba_packed(&y, row, n);
            let mcc_s = score_binary_mcc_packed(&y, row, n);
            assert!((ba[r] - ba_s).abs() < 1e-12);
            assert!((mcc[r] - mcc_s).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cont_batch_matches_single() {
        let n = 96usize;
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 - 7.0).collect();
        let row_words = words_for_samples(n);
        let n_rows = 3usize;

        let mut bits_flat = vec![0u64; n_rows * row_words];
        for r in 0..n_rows {
            for i in 0..n {
                if ((i * (r + 1)) % 7) < 3 {
                    bits_flat[r * row_words + (i >> 6)] |= 1u64 << (i & 63);
                }
            }
        }

        let (md, corr) =
            score_cont_mean_diff_corr_batch_packed(&y, &bits_flat, row_words, n_rows, n);

        for r in 0..n_rows {
            let row = &bits_flat[r * row_words..(r + 1) * row_words];
            let md_s = score_cont_mean_diff_packed(&y, row, n);
            let corr_s = score_cont_corr_packed(&y, row, n);
            if md_s.is_nan() {
                assert!(md[r].is_nan());
            } else {
                assert!((md[r] - md_s).abs() < 1e-12);
            }
            assert!((corr[r] - corr_s).abs() < 1e-12);
        }
    }

    #[test]
    fn test_score_cont_weighted_mean_diff_basic() {
        let y = vec![5.0, 4.0, -1.0, -2.0];
        let bits = vec![0b0011_u64];
        let sc = score_cont_weighted_mean_diff_packed(&y, &bits, y.len());
        assert_eq!(sc.n_hit, 2);
        assert_eq!(sc.n_miss, 2);
        assert!((sc.mean_hit - 4.5).abs() < 1e-12);
        assert!((sc.mean_miss + 1.5).abs() < 1e-12);
        assert!((sc.raw_score - 36.0).abs() < 1e-12);
        assert!((sc.score - 36.0).abs() < 1e-12);
    }

    #[test]
    fn test_score_cont_weighted_mean_diff_centers_nonzero_total_sum() {
        let y = vec![2.0, 2.0, 0.0, 0.0];
        let bits = vec![0b0011_u64];
        let sc = score_cont_weighted_mean_diff_packed(&y, &bits, y.len());
        assert_eq!(sc.n_hit, 2);
        assert_eq!(sc.n_miss, 2);
        assert!((sc.mean_hit - 2.0).abs() < 1e-12);
        assert!((sc.mean_miss - 0.0).abs() < 1e-12);
        assert!((sc.raw_score - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_score_cont_weighted_mean_diff_extreme_support_is_zero() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let bits_all_zero = vec![0u64];
        let bits_all_one = vec![0b1111_u64];
        let sc0 = score_cont_weighted_mean_diff_packed(&y, &bits_all_zero, y.len());
        let sc1 = score_cont_weighted_mean_diff_packed(&y, &bits_all_one, y.len());
        assert_eq!(sc0.score, 0.0);
        assert_eq!(sc1.score, 0.0);
        assert_eq!(sc0.n_hit, 0);
        assert_eq!(sc1.n_miss, 0);
    }

    #[test]
    fn test_support_size_packed_masks_tail() {
        let bits = vec![u64::MAX];
        assert_eq!(support_size_packed(&bits, 5), 5);
        assert_eq!(support_size_packed(&bits, 64), 64);
    }
}
