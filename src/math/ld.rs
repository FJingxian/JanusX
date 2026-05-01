use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::bedmath::{code_to_dosage_or_zero, PackedByteLut, PackedPairLut};

#[derive(Clone, Copy, Default)]
pub(crate) struct PackedRowStats {
    pub mean: f64,
    pub std: f64,
    pub maf: f64,
    pub non_missing: usize,
    pub has_missing: bool,
}

static PRUNE_DOT_TOTAL_CALLS: AtomicU64 = AtomicU64::new(0);
static PRUNE_DOT_AVX2_CALLS: AtomicU64 = AtomicU64::new(0);
static PRUNE_DOT_NEON_CALLS: AtomicU64 = AtomicU64::new(0);
static PRUNE_DOT_SCALAR_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(target_arch = "x86_64")]
#[inline]
fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(false)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn prune_force_scalar_runtime() -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    *FORCE_SCALAR.get_or_init(|| env_truthy("JANUSX_PRUNE_FORCE_SCALAR"))
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn prune_force_scalar_runtime() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn prune_avx2_runtime_available() -> bool {
    static AVX2_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVX2_AVAILABLE.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn prune_avx2_runtime_available() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn prune_neon_runtime_available() -> bool {
    true
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn prune_neon_runtime_available() -> bool {
    false
}

pub(crate) fn packed_prune_kernel_stats(
    reset: bool,
) -> (String, bool, bool, bool, u64, u64, u64, u64, f64) {
    if reset {
        PRUNE_DOT_TOTAL_CALLS.store(0, Ordering::Relaxed);
        PRUNE_DOT_AVX2_CALLS.store(0, Ordering::Relaxed);
        PRUNE_DOT_NEON_CALLS.store(0, Ordering::Relaxed);
        PRUNE_DOT_SCALAR_CALLS.store(0, Ordering::Relaxed);
    }
    let total = PRUNE_DOT_TOTAL_CALLS.load(Ordering::Relaxed);
    let avx2_calls = PRUNE_DOT_AVX2_CALLS.load(Ordering::Relaxed);
    let neon_calls = PRUNE_DOT_NEON_CALLS.load(Ordering::Relaxed);
    let scalar_calls = PRUNE_DOT_SCALAR_CALLS.load(Ordering::Relaxed);

    let force_scalar = prune_force_scalar_runtime();
    let avx2_available = prune_avx2_runtime_available();
    let neon_available = prune_neon_runtime_available();
    let backend = if cfg!(target_arch = "x86_64") {
        if force_scalar {
            "x86_64+scalar(forced)".to_string()
        } else if avx2_available {
            "x86_64+avx2".to_string()
        } else {
            "x86_64+scalar".to_string()
        }
    } else if cfg!(target_arch = "aarch64") {
        "aarch64+neon".to_string()
    } else {
        "scalar".to_string()
    };
    let avx2_hit_rate = if total > 0 {
        (avx2_calls as f64) / (total as f64)
    } else {
        0.0_f64
    };
    (
        backend,
        avx2_available,
        neon_available,
        force_scalar,
        total,
        avx2_calls,
        neon_calls,
        scalar_calls,
        avx2_hit_rate,
    )
}

#[inline]
pub(crate) fn dot_nomiss_pair_from_packed(
    row_i: &[u8],
    row_j: &[u8],
    n_samples: usize,
    pair_lut: &PackedPairLut,
    code4_lut: &[[u8; 4]; 256],
) -> f64 {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut dot = 0.0_f64;
    for b in 0..full_bytes {
        let idx = ((row_i[b] as usize) << 8) | (row_j[b] as usize);
        dot += pair_lut.dot_obs0[idx] as f64;
    }
    if rem > 0 {
        let ci = &code4_lut[row_i[full_bytes] as usize];
        let cj = &code4_lut[row_j[full_bytes] as usize];
        for lane in 0..rem {
            let vi = code_to_dosage_or_zero(ci[lane]) as f64;
            let vj = code_to_dosage_or_zero(cj[lane]) as f64;
            dot += vi * vj;
        }
    }
    dot
}

#[inline]
pub(crate) fn r2_pairwise_complete_from_packed(
    row_i: &[u8],
    row_j: &[u8],
    n_samples: usize,
    pair_lut: &PackedPairLut,
    code4_lut: &[[u8; 4]; 256],
) -> Option<f64> {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut n_obs: u64 = 0;
    let mut sum_i: u64 = 0;
    let mut sum_j: u64 = 0;
    let mut sum_i2: u64 = 0;
    let mut sum_j2: u64 = 0;
    let mut sum_ij: u64 = 0;

    for b in 0..full_bytes {
        let idx = ((row_i[b] as usize) << 8) | (row_j[b] as usize);
        n_obs = n_obs.saturating_add(pair_lut.obs_ct[idx] as u64);
        sum_i = sum_i.saturating_add(pair_lut.obs_sum_i[idx] as u64);
        sum_j = sum_j.saturating_add(pair_lut.obs_sum_j[idx] as u64);
        sum_i2 = sum_i2.saturating_add(pair_lut.obs_sum_i2[idx] as u64);
        sum_j2 = sum_j2.saturating_add(pair_lut.obs_sum_j2[idx] as u64);
        sum_ij = sum_ij.saturating_add(pair_lut.obs_sum_ij[idx] as u64);
    }

    if rem > 0 {
        let ci = &code4_lut[row_i[full_bytes] as usize];
        let cj = &code4_lut[row_j[full_bytes] as usize];
        for lane in 0..rem {
            let ai = ci[lane];
            let aj = cj[lane];
            if ai == 0b01 || aj == 0b01 {
                continue;
            }
            let vi = code_to_dosage_or_zero(ai) as u64;
            let vj = code_to_dosage_or_zero(aj) as u64;
            n_obs = n_obs.saturating_add(1);
            sum_i = sum_i.saturating_add(vi);
            sum_j = sum_j.saturating_add(vj);
            sum_i2 = sum_i2.saturating_add(vi.saturating_mul(vi));
            sum_j2 = sum_j2.saturating_add(vj.saturating_mul(vj));
            sum_ij = sum_ij.saturating_add(vi.saturating_mul(vj));
        }
    }

    if n_obs <= 1 {
        return None;
    }
    let n = n_obs as f64;
    let si = sum_i as f64;
    let sj = sum_j as f64;
    let si2 = sum_i2 as f64;
    let sj2 = sum_j2 as f64;
    let sij = sum_ij as f64;

    let cov_num = sij * n - si * sj;
    let var_i_num = si2 * n - si * si;
    let var_j_num = sj2 * n - sj * sj;
    let denom = var_i_num * var_j_num;
    if !(denom.is_finite() && denom > 0.0_f64 && cov_num.is_finite()) {
        return None;
    }
    Some((cov_num * cov_num) / denom)
}

pub(crate) fn build_bitplanes_u64(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    n_samples: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> (Vec<u64>, Vec<u64>, Vec<u64>, usize, Vec<u64>) {
    let words = n_samples.div_ceil(64);
    if m == 0 || words == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), 0, Vec::new());
    }

    let mut lut_h = [0u8; 256];
    let mut lut_l = [0u8; 256];
    let mut lut_m = [0u8; 256];
    for b in 0u16..=255 {
        let byte = b as u8;
        let mut hb = 0u8;
        let mut lb = 0u8;
        let mut mb = 0u8;
        for lane in 0..4usize {
            let code = (byte >> (lane * 2)) & 0b11;
            if ((code >> 1) & 1) != 0 {
                hb |= 1u8 << lane;
            }
            if (code & 1) != 0 {
                lb |= 1u8 << lane;
            }
            if code == 0b01 {
                mb |= 1u8 << lane;
            }
        }
        lut_h[byte as usize] = hb;
        lut_l[byte as usize] = lb;
        lut_m[byte as usize] = mb;
    }

    let mut word_masks = vec![u64::MAX; words];
    let rem = n_samples % 64;
    if rem != 0 {
        word_masks[words - 1] = (1u64 << rem) - 1u64;
    }
    let last_mask = word_masks[words - 1];

    let mut h_bits = vec![0u64; m * words];
    let mut l_bits = vec![0u64; m * words];
    let mut m_bits = vec![0u64; m * words];
    let mut run = || {
        h_bits
            .par_chunks_mut(words)
            .zip(l_bits.par_chunks_mut(words))
            .zip(m_bits.par_chunks_mut(words))
            .enumerate()
            .for_each(|(row_idx, ((hrow, lrow), mrow))| {
                let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                for (bi, &b) in row.iter().enumerate() {
                    let w = bi >> 4;
                    let sh = ((bi & 15) << 2) as u32;
                    hrow[w] |= (lut_h[b as usize] as u64) << sh;
                    lrow[w] |= (lut_l[b as usize] as u64) << sh;
                    mrow[w] |= (lut_m[b as usize] as u64) << sh;
                }
                hrow[words - 1] &= last_mask;
                lrow[words - 1] &= last_mask;
                mrow[words - 1] &= last_mask;
            });
    };
    if let Some(tp) = pool {
        tp.install(&mut run);
    } else {
        run();
    }
    (h_bits, l_bits, m_bits, words, word_masks)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcount_u8x32_avx2(
    v: core::arch::x86_64::__m256i,
    lut4: core::arch::x86_64::__m256i,
    low_mask: core::arch::x86_64::__m256i,
    zero: core::arch::x86_64::__m256i,
) -> u64 {
    use core::arch::x86_64::{
        _mm256_add_epi8, _mm256_and_si256, _mm256_extract_epi64, _mm256_sad_epu8,
        _mm256_shuffle_epi8, _mm256_srli_epi16,
    };
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    let cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut4, lo), _mm256_shuffle_epi8(lut4, hi));
    let sum64 = _mm256_sad_epu8(cnt, zero);
    (_mm256_extract_epi64(sum64, 0) as u64)
        + (_mm256_extract_epi64(sum64, 1) as u64)
        + (_mm256_extract_epi64(sum64, 2) as u64)
        + (_mm256_extract_epi64(sum64, 3) as u64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_nomiss_pair_bitplanes_avx2(
    hi: &[u64],
    li: &[u64],
    hj: &[u64],
    lj: &[u64],
    full_words: usize,
) -> (u64, usize) {
    use core::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_setr_epi8,
        _mm256_setzero_si256,
    };

    let lut4 = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f_i8);
    let zero = _mm256_setzero_si256();

    let mut acc = 0u64;
    let mut w = 0usize;
    while w + 4 <= full_words {
        let hx = _mm256_loadu_si256(hi.as_ptr().add(w) as *const __m256i);
        let lx = _mm256_loadu_si256(li.as_ptr().add(w) as *const __m256i);
        let hy = _mm256_loadu_si256(hj.as_ptr().add(w) as *const __m256i);
        let ly = _mm256_loadu_si256(lj.as_ptr().add(w) as *const __m256i);

        acc += popcount_u8x32_avx2(_mm256_and_si256(hx, hy), lut4, low_mask, zero)
            + popcount_u8x32_avx2(_mm256_and_si256(hx, ly), lut4, low_mask, zero)
            + popcount_u8x32_avx2(_mm256_and_si256(lx, hy), lut4, low_mask, zero)
            + popcount_u8x32_avx2(_mm256_and_si256(lx, ly), lut4, low_mask, zero);
        w += 4;
    }
    (acc, w)
}

#[inline]
pub(crate) fn dot_nomiss_pair_bitplanes(
    i: usize,
    j: usize,
    h_bits: &[u64],
    l_bits: &[u64],
    words: usize,
    word_masks: &[u64],
) -> f64 {
    PRUNE_DOT_TOTAL_CALLS.fetch_add(1, Ordering::Relaxed);
    if words == 0 {
        PRUNE_DOT_SCALAR_CALLS.fetch_add(1, Ordering::Relaxed);
        return 0.0_f64;
    }
    let oi = i * words;
    let oj = j * words;
    let hi = &h_bits[oi..oi + words];
    let li = &l_bits[oi..oi + words];
    let hj = &h_bits[oj..oj + words];
    let lj = &l_bits[oj..oj + words];

    let mut acc: u64 = 0;
    let full_words = words - 1;
    let mut w = 0usize;
    #[cfg(target_arch = "aarch64")]
    let vector_hit = true;
    #[cfg(not(target_arch = "aarch64"))]
    let mut vector_hit = false;

    #[cfg(target_arch = "x86_64")]
    {
        if prune_avx2_runtime_available() && !prune_force_scalar_runtime() {
            let (acc_avx2, w_avx2) =
                unsafe { dot_nomiss_pair_bitplanes_avx2(hi, li, hj, lj, full_words) };
            acc += acc_avx2;
            w = w_avx2;
            vector_hit = true;
            PRUNE_DOT_AVX2_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{
            vandq_u64, vcntq_u8, vgetq_lane_u64, vld1q_u64, vpaddlq_u16, vpaddlq_u32, vpaddlq_u8,
            vreinterpretq_u8_u64,
        };

        #[inline(always)]
        unsafe fn popcount_u64x2(v: core::arch::aarch64::uint64x2_t) -> u64 {
            let cnt8 = vcntq_u8(vreinterpretq_u8_u64(v));
            let sum16 = vpaddlq_u8(cnt8);
            let sum32 = vpaddlq_u16(sum16);
            let sum64 = vpaddlq_u32(sum32);
            vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1)
        }

        while w + 2 <= full_words {
            let hx = unsafe { vld1q_u64(hi.as_ptr().add(w)) };
            let lx = unsafe { vld1q_u64(li.as_ptr().add(w)) };
            let hy = unsafe { vld1q_u64(hj.as_ptr().add(w)) };
            let ly = unsafe { vld1q_u64(lj.as_ptr().add(w)) };
            unsafe {
                acc += popcount_u64x2(vandq_u64(hx, hy))
                    + popcount_u64x2(vandq_u64(hx, ly))
                    + popcount_u64x2(vandq_u64(lx, hy))
                    + popcount_u64x2(vandq_u64(lx, ly));
            }
            w += 2;
        }
        PRUNE_DOT_NEON_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    if !vector_hit {
        PRUNE_DOT_SCALAR_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    while w + 4 <= full_words {
        let hx0 = hi[w];
        let lx0 = li[w];
        let hy0 = hj[w];
        let ly0 = lj[w];
        let hx1 = hi[w + 1];
        let lx1 = li[w + 1];
        let hy1 = hj[w + 1];
        let ly1 = lj[w + 1];
        let hx2 = hi[w + 2];
        let lx2 = li[w + 2];
        let hy2 = hj[w + 2];
        let ly2 = lj[w + 2];
        let hx3 = hi[w + 3];
        let lx3 = li[w + 3];
        let hy3 = hj[w + 3];
        let ly3 = lj[w + 3];

        acc += (hx0 & hy0).count_ones() as u64
            + (hx0 & ly0).count_ones() as u64
            + (lx0 & hy0).count_ones() as u64
            + (lx0 & ly0).count_ones() as u64
            + (hx1 & hy1).count_ones() as u64
            + (hx1 & ly1).count_ones() as u64
            + (lx1 & hy1).count_ones() as u64
            + (lx1 & ly1).count_ones() as u64
            + (hx2 & hy2).count_ones() as u64
            + (hx2 & ly2).count_ones() as u64
            + (lx2 & hy2).count_ones() as u64
            + (lx2 & ly2).count_ones() as u64
            + (hx3 & hy3).count_ones() as u64
            + (hx3 & ly3).count_ones() as u64
            + (lx3 & hy3).count_ones() as u64
            + (lx3 & ly3).count_ones() as u64;
        w += 4;
    }
    while w < full_words {
        let hx = hi[w];
        let lx = li[w];
        let hy = hj[w];
        let ly = lj[w];
        acc += (hx & hy).count_ones() as u64
            + (hx & ly).count_ones() as u64
            + (lx & hy).count_ones() as u64
            + (lx & ly).count_ones() as u64;
        w += 1;
    }

    let tail = words - 1;
    let m = word_masks[tail];
    let hx = hi[tail] & m;
    let lx = li[tail] & m;
    let hy = hj[tail] & m;
    let ly = lj[tail] & m;
    acc += (hx & hy).count_ones() as u64
        + (hx & ly).count_ones() as u64
        + (lx & hy).count_ones() as u64
        + (lx & ly).count_ones() as u64;
    acc as f64
}

#[inline]
pub(crate) fn classify_ld_pair_by_maf(maf_i: f64, maf_j: f64, eps: f64) -> u8 {
    if maf_i < (1.0_f64 - eps) * maf_j {
        1u8
    } else {
        2u8
    }
}

#[inline]
pub(crate) fn compute_packed_row_stats(
    row: &[u8],
    n_samples: usize,
    byte_lut: &PackedByteLut,
) -> PackedRowStats {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;

    let mut non_missing: usize = 0;
    let mut alt_sum: usize = 0;
    let mut sq_sum: usize = 0;
    for &b in row.iter().take(full_bytes) {
        let idx = b as usize;
        non_missing += byte_lut.nonmiss[idx] as usize;
        alt_sum += byte_lut.alt_sum[idx] as usize;
        sq_sum += byte_lut.sq_sum[idx] as usize;
    }
    if rem > 0 {
        let codes = &byte_lut.code4[row[full_bytes] as usize];
        for &code in codes.iter().take(rem) {
            match code {
                0b00 => {
                    non_missing += 1;
                }
                0b10 => {
                    non_missing += 1;
                    alt_sum += 1;
                    sq_sum += 1;
                }
                0b11 => {
                    non_missing += 1;
                    alt_sum += 2;
                    sq_sum += 4;
                }
                _ => {}
            }
        }
    }

    if non_missing > 0 {
        let obs_n = non_missing as f64;
        let sum_g = alt_sum as f64;
        let sum_g2 = sq_sum as f64;
        let p = sum_g / (2.0_f64 * obs_n);
        let maf = p.min(1.0_f64 - p);
        let mean = sum_g / obs_n;
        let ss = (sum_g2 - (sum_g * sum_g / obs_n)).max(0.0_f64);
        let var = ss / denom;
        let std = var.max(1e-12_f64).sqrt();
        PackedRowStats {
            mean,
            std,
            maf,
            non_missing,
            has_missing: non_missing < n_samples,
        }
    } else {
        PackedRowStats {
            mean: 0.0_f64,
            std: 1e-6_f64,
            maf: 0.0_f64,
            non_missing: 0usize,
            has_missing: true,
        }
    }
}

fn build_row_bitplanes_u64(row: &[u8], n_samples: usize) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let words = n_samples.div_ceil(64);
    if words == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    static ROW_LUT: OnceLock<([u8; 256], [u8; 256], [u8; 256])> = OnceLock::new();
    let (lut_h, lut_l, lut_m) = ROW_LUT.get_or_init(|| {
        let mut lh = [0u8; 256];
        let mut ll = [0u8; 256];
        let mut lm = [0u8; 256];
        for b in 0u16..=255 {
            let byte = b as u8;
            let mut hb = 0u8;
            let mut lb = 0u8;
            let mut mb = 0u8;
            for lane in 0..4usize {
                let code = (byte >> (lane * 2)) & 0b11;
                if ((code >> 1) & 1) != 0 {
                    hb |= 1u8 << lane;
                }
                if (code & 1) != 0 {
                    lb |= 1u8 << lane;
                }
                if code == 0b01 {
                    mb |= 1u8 << lane;
                }
            }
            lh[byte as usize] = hb;
            ll[byte as usize] = lb;
            lm[byte as usize] = mb;
        }
        (lh, ll, lm)
    });

    let mut hi = vec![0u64; words];
    let mut lo = vec![0u64; words];
    let mut mi = vec![0u64; words];
    for (bi, &b) in row.iter().enumerate() {
        let w = bi >> 4;
        let sh = ((bi & 15) << 2) as u32;
        hi[w] |= (lut_h[b as usize] as u64) << sh;
        lo[w] |= (lut_l[b as usize] as u64) << sh;
        mi[w] |= (lut_m[b as usize] as u64) << sh;
    }
    let rem = n_samples % 64;
    if rem != 0 {
        let tail_mask = (1u64 << rem) - 1u64;
        hi[words - 1] &= tail_mask;
        lo[words - 1] &= tail_mask;
        mi[words - 1] &= tail_mask;
    }
    (hi, lo, mi)
}

#[inline]
pub(crate) fn build_row_bitplanes_u64_with_aux(
    row: &[u8],
    n_samples: usize,
) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    let (hi, lo, mi) = build_row_bitplanes_u64(row, n_samples);
    let words = hi.len();
    if words == 0 {
        return (hi, lo, mi, Vec::new(), Vec::new());
    }
    let mut ai = vec![0u64; words];
    let mut vi = vec![0u64; words];
    for w in 0..words {
        ai[w] = hi[w] & lo[w];
        vi[w] = !mi[w];
    }
    let rem = n_samples % 64;
    if rem != 0 {
        let tail_mask = (1u64 << rem) - 1u64;
        vi[words - 1] &= tail_mask;
    }
    (hi, lo, mi, ai, vi)
}

#[inline]
pub(crate) fn r2_pairwise_complete_bitplanes(
    hi: &[u64],
    li: &[u64],
    mi: &[u64],
    hj: &[u64],
    lj: &[u64],
    mj: &[u64],
    word_masks: &[u64],
) -> Option<f64> {
    let words = hi.len();
    if words == 0 {
        return None;
    }
    debug_assert_eq!(li.len(), words);
    debug_assert_eq!(mi.len(), words);
    debug_assert_eq!(hj.len(), words);
    debug_assert_eq!(lj.len(), words);
    debug_assert_eq!(mj.len(), words);
    debug_assert_eq!(word_masks.len(), words);

    let mut n_obs: u64 = 0;
    let mut sum_i: u64 = 0;
    let mut sum_j: u64 = 0;
    let mut sum_i2: u64 = 0;
    let mut sum_j2: u64 = 0;
    let mut sum_ij: u64 = 0;

    let full_words = words.saturating_sub(1);
    for w in 0..full_words {
        let hiw = hi[w];
        let liw = li[w];
        let hjw = hj[w];
        let ljw = lj[w];

        let valid = !(mi[w] | mj[w]);
        if valid == 0 {
            continue;
        }

        let ai = hiw & liw;
        let aj = hjw & ljw;
        let hv_i = hiw & valid;
        let hv_j = hjw & valid;
        let av_i = ai & valid;
        let av_j = aj & valid;

        let cnt_n = valid.count_ones() as u64;
        let cnt_hi = hv_i.count_ones() as u64;
        let cnt_hj = hv_j.count_ones() as u64;
        let cnt_ai = av_i.count_ones() as u64;
        let cnt_aj = av_j.count_ones() as u64;

        n_obs = n_obs.saturating_add(cnt_n);
        sum_i = sum_i.saturating_add(cnt_hi.saturating_add(cnt_ai));
        sum_j = sum_j.saturating_add(cnt_hj.saturating_add(cnt_aj));
        sum_i2 = sum_i2.saturating_add(cnt_hi.saturating_add(3u64.saturating_mul(cnt_ai)));
        sum_j2 = sum_j2.saturating_add(cnt_hj.saturating_add(3u64.saturating_mul(cnt_aj)));

        let c_hh = (hv_i & hv_j).count_ones() as u64;
        let c_ha = (hv_i & av_j).count_ones() as u64;
        let c_ah = (av_i & hv_j).count_ones() as u64;
        let c_aa = (av_i & av_j).count_ones() as u64;
        sum_ij = sum_ij.saturating_add(
            c_hh.saturating_add(c_ha)
                .saturating_add(c_ah)
                .saturating_add(c_aa),
        );
    }

    let tail = words - 1;
    let hiw = hi[tail];
    let liw = li[tail];
    let hjw = hj[tail];
    let ljw = lj[tail];
    let valid = (!(mi[tail] | mj[tail])) & word_masks[tail];
    if valid != 0 {
        let ai = hiw & liw;
        let aj = hjw & ljw;
        let hv_i = hiw & valid;
        let hv_j = hjw & valid;
        let av_i = ai & valid;
        let av_j = aj & valid;

        let cnt_n = valid.count_ones() as u64;
        let cnt_hi = hv_i.count_ones() as u64;
        let cnt_hj = hv_j.count_ones() as u64;
        let cnt_ai = av_i.count_ones() as u64;
        let cnt_aj = av_j.count_ones() as u64;

        n_obs = n_obs.saturating_add(cnt_n);
        sum_i = sum_i.saturating_add(cnt_hi.saturating_add(cnt_ai));
        sum_j = sum_j.saturating_add(cnt_hj.saturating_add(cnt_aj));
        sum_i2 = sum_i2.saturating_add(cnt_hi.saturating_add(3u64.saturating_mul(cnt_ai)));
        sum_j2 = sum_j2.saturating_add(cnt_hj.saturating_add(3u64.saturating_mul(cnt_aj)));

        let c_hh = (hv_i & hv_j).count_ones() as u64;
        let c_ha = (hv_i & av_j).count_ones() as u64;
        let c_ah = (av_i & hv_j).count_ones() as u64;
        let c_aa = (av_i & av_j).count_ones() as u64;
        sum_ij = sum_ij.saturating_add(
            c_hh.saturating_add(c_ha)
                .saturating_add(c_ah)
                .saturating_add(c_aa),
        );
    }

    if n_obs <= 1 {
        return None;
    }
    let n = n_obs as f64;
    let si = sum_i as f64;
    let sj = sum_j as f64;
    let si2 = sum_i2 as f64;
    let sj2 = sum_j2 as f64;
    let sij = sum_ij as f64;

    let cov_num = sij * n - si * sj;
    let var_i_num = si2 * n - si * si;
    let var_j_num = sj2 * n - sj * sj;
    let denom = var_i_num * var_j_num;
    if !(denom.is_finite() && denom > 0.0_f64 && cov_num.is_finite()) {
        return None;
    }
    Some((cov_num * cov_num) / denom)
}

#[inline]
pub(crate) fn r2_pairwise_complete_bitplanes_cached_masks(
    hi: &[u64],
    ai: &[u64],
    vi: &[u64],
    hj: &[u64],
    aj: &[u64],
    vj: &[u64],
) -> Option<f64> {
    let words = hi.len();
    if words == 0 {
        return None;
    }
    debug_assert_eq!(ai.len(), words);
    debug_assert_eq!(vi.len(), words);
    debug_assert_eq!(hj.len(), words);
    debug_assert_eq!(aj.len(), words);
    debug_assert_eq!(vj.len(), words);

    let mut n_obs: u64 = 0;
    let mut sum_i: u64 = 0;
    let mut sum_j: u64 = 0;
    let mut sum_i2: u64 = 0;
    let mut sum_j2: u64 = 0;
    let mut sum_ij: u64 = 0;

    for w in 0..words {
        let valid = vi[w] & vj[w];
        if valid == 0 {
            continue;
        }

        let hv_i = hi[w] & valid;
        let hv_j = hj[w] & valid;
        let av_i = ai[w] & valid;
        let av_j = aj[w] & valid;

        let cnt_n = valid.count_ones() as u64;
        let cnt_hi = hv_i.count_ones() as u64;
        let cnt_hj = hv_j.count_ones() as u64;
        let cnt_ai = av_i.count_ones() as u64;
        let cnt_aj = av_j.count_ones() as u64;

        n_obs = n_obs.saturating_add(cnt_n);
        sum_i = sum_i.saturating_add(cnt_hi.saturating_add(cnt_ai));
        sum_j = sum_j.saturating_add(cnt_hj.saturating_add(cnt_aj));
        sum_i2 = sum_i2.saturating_add(cnt_hi.saturating_add(3u64.saturating_mul(cnt_ai)));
        sum_j2 = sum_j2.saturating_add(cnt_hj.saturating_add(3u64.saturating_mul(cnt_aj)));

        let c_hh = (hv_i & hv_j).count_ones() as u64;
        let c_ha = (hv_i & av_j).count_ones() as u64;
        let c_ah = (av_i & hv_j).count_ones() as u64;
        let c_aa = (av_i & av_j).count_ones() as u64;
        sum_ij = sum_ij.saturating_add(
            c_hh.saturating_add(c_ha)
                .saturating_add(c_ah)
                .saturating_add(c_aa),
        );
    }

    if n_obs <= 1 {
        return None;
    }
    let n = n_obs as f64;
    let si = sum_i as f64;
    let sj = sum_j as f64;
    let si2 = sum_i2 as f64;
    let sj2 = sum_j2 as f64;
    let sij = sum_ij as f64;

    let cov_num = sij * n - si * sj;
    let var_i_num = si2 * n - si * si;
    let var_j_num = sj2 * n - sj * sj;
    let denom = var_i_num * var_j_num;
    if !(denom.is_finite() && denom > 0.0_f64 && cov_num.is_finite()) {
        return None;
    }
    Some((cov_num * cov_num) / denom)
}

#[inline]
pub(crate) fn dot_nomiss_row_bitplanes(
    hi: &[u64],
    li: &[u64],
    hj: &[u64],
    lj: &[u64],
    word_masks: &[u64],
) -> f64 {
    PRUNE_DOT_TOTAL_CALLS.fetch_add(1, Ordering::Relaxed);
    if hi.is_empty() {
        PRUNE_DOT_SCALAR_CALLS.fetch_add(1, Ordering::Relaxed);
        return 0.0_f64;
    }
    let words = hi.len();
    debug_assert_eq!(li.len(), words);
    debug_assert_eq!(hj.len(), words);
    debug_assert_eq!(lj.len(), words);
    debug_assert_eq!(word_masks.len(), words);

    let mut acc: u64 = 0;
    let full_words = words - 1;
    let mut w = 0usize;
    #[cfg(target_arch = "aarch64")]
    let vector_hit = true;
    #[cfg(not(target_arch = "aarch64"))]
    let mut vector_hit = false;

    #[cfg(target_arch = "x86_64")]
    {
        if prune_avx2_runtime_available() && !prune_force_scalar_runtime() {
            let (acc_avx2, w_avx2) =
                unsafe { dot_nomiss_pair_bitplanes_avx2(hi, li, hj, lj, full_words) };
            acc += acc_avx2;
            w = w_avx2;
            vector_hit = true;
            PRUNE_DOT_AVX2_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{
            vandq_u64, vcntq_u8, vgetq_lane_u64, vld1q_u64, vpaddlq_u16, vpaddlq_u32, vpaddlq_u8,
            vreinterpretq_u8_u64,
        };

        #[inline(always)]
        unsafe fn popcount_u64x2(v: core::arch::aarch64::uint64x2_t) -> u64 {
            let cnt8 = vcntq_u8(vreinterpretq_u8_u64(v));
            let sum16 = vpaddlq_u8(cnt8);
            let sum32 = vpaddlq_u16(sum16);
            let sum64 = vpaddlq_u32(sum32);
            vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1)
        }

        while w + 2 <= full_words {
            let hx = unsafe { vld1q_u64(hi.as_ptr().add(w)) };
            let lx = unsafe { vld1q_u64(li.as_ptr().add(w)) };
            let hy = unsafe { vld1q_u64(hj.as_ptr().add(w)) };
            let ly = unsafe { vld1q_u64(lj.as_ptr().add(w)) };
            unsafe {
                acc += popcount_u64x2(vandq_u64(hx, hy))
                    + popcount_u64x2(vandq_u64(hx, ly))
                    + popcount_u64x2(vandq_u64(lx, hy))
                    + popcount_u64x2(vandq_u64(lx, ly));
            }
            w += 2;
        }
        PRUNE_DOT_NEON_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    if !vector_hit {
        PRUNE_DOT_SCALAR_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    while w + 4 <= full_words {
        let hx0 = hi[w];
        let lx0 = li[w];
        let hy0 = hj[w];
        let ly0 = lj[w];
        let hx1 = hi[w + 1];
        let lx1 = li[w + 1];
        let hy1 = hj[w + 1];
        let ly1 = lj[w + 1];
        let hx2 = hi[w + 2];
        let lx2 = li[w + 2];
        let hy2 = hj[w + 2];
        let ly2 = lj[w + 2];
        let hx3 = hi[w + 3];
        let lx3 = li[w + 3];
        let hy3 = hj[w + 3];
        let ly3 = lj[w + 3];

        acc += (hx0 & hy0).count_ones() as u64
            + (hx0 & ly0).count_ones() as u64
            + (lx0 & hy0).count_ones() as u64
            + (lx0 & ly0).count_ones() as u64
            + (hx1 & hy1).count_ones() as u64
            + (hx1 & ly1).count_ones() as u64
            + (lx1 & hy1).count_ones() as u64
            + (lx1 & ly1).count_ones() as u64
            + (hx2 & hy2).count_ones() as u64
            + (hx2 & ly2).count_ones() as u64
            + (lx2 & hy2).count_ones() as u64
            + (lx2 & ly2).count_ones() as u64
            + (hx3 & hy3).count_ones() as u64
            + (hx3 & ly3).count_ones() as u64
            + (lx3 & hy3).count_ones() as u64
            + (lx3 & ly3).count_ones() as u64;
        w += 4;
    }
    while w < full_words {
        let hx = hi[w];
        let lx = li[w];
        let hy = hj[w];
        let ly = lj[w];
        acc += (hx & hy).count_ones() as u64
            + (hx & ly).count_ones() as u64
            + (lx & hy).count_ones() as u64
            + (lx & ly).count_ones() as u64;
        w += 1;
    }

    let tail = words - 1;
    let m = word_masks[tail];
    let hx = hi[tail] & m;
    let lx = li[tail] & m;
    let hy = hj[tail] & m;
    let ly = lj[tail] & m;
    acc += (hx & hy).count_ones() as u64
        + (hx & ly).count_ones() as u64
        + (lx & hy).count_ones() as u64
        + (lx & ly).count_ones() as u64;
    acc as f64
}
