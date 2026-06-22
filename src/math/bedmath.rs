use rayon::prelude::*;
#[cfg(target_arch = "x86")]
use std::arch::x86 as x86_simd;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as x86_simd;
use std::sync::{Arc, OnceLock};

use crate::gfcore::{block_rows_from_memory_target_mb, parse_positive_env_f64};

const BED_DECODE_SIMD_DEFAULT: &str = match option_env!("JANUSX_BED_DECODE_SIMD_DEFAULT") {
    Some(v) => v,
    None => "0",
};
const BED_SUBSET_SIMD_DEFAULT: &str = match option_env!("JANUSX_BED_SUBSET_SIMD_DEFAULT") {
    Some(v) => v,
    None => "1",
};

#[inline]
pub(crate) fn decode_plink_bed_hardcall(code: u8) -> Option<f64> {
    match code {
        0b00 => Some(0.0),
        0b10 => Some(1.0),
        0b11 => Some(2.0),
        _ => None, // 0b01 => missing
    }
}

pub(crate) struct PackedByteLut {
    pub(crate) nonmiss: [u8; 256],
    pub(crate) alt_sum: [u8; 256],
    pub(crate) het_sum: [u8; 256],
    pub(crate) sq_sum: [u8; 256],
    pub(crate) code4: [[u8; 4]; 256],
}

pub(crate) fn packed_byte_lut() -> &'static PackedByteLut {
    static LUT: OnceLock<PackedByteLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut nonmiss = [0u8; 256];
        let mut alt_sum = [0u8; 256];
        let mut het_sum = [0u8; 256];
        let mut sq_sum = [0u8; 256];
        let mut code4 = [[0u8; 4]; 256];
        for b in 0u16..=255 {
            let byte = b as u8;
            let mut nm = 0u8;
            let mut alt = 0u8;
            let mut het = 0u8;
            let mut sq = 0u8;
            let mut codes = [0u8; 4];
            for lane in 0..4usize {
                let code = (byte >> (lane * 2)) & 0b11;
                codes[lane] = code;
                match code {
                    0b00 => {
                        nm += 1;
                    }
                    0b10 => {
                        nm += 1;
                        alt += 1;
                        het += 1;
                        sq += 1;
                    }
                    0b11 => {
                        nm += 1;
                        alt += 2;
                        sq += 4;
                    }
                    _ => {}
                }
            }
            let idx = byte as usize;
            nonmiss[idx] = nm;
            alt_sum[idx] = alt;
            het_sum[idx] = het;
            sq_sum[idx] = sq;
            code4[idx] = codes;
        }
        PackedByteLut {
            nonmiss,
            alt_sum,
            het_sum,
            sq_sum,
            code4,
        }
    })
}

#[inline]
pub(crate) fn packed_row_missing_count(row: &[u8], n_samples: usize) -> usize {
    let byte_lut = packed_byte_lut();
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut non_missing = 0usize;
    for &b in row.iter().take(full_bytes) {
        non_missing += byte_lut.nonmiss[b as usize] as usize;
    }
    if rem > 0 {
        let codes = &byte_lut.code4[row[full_bytes] as usize];
        for &code in codes.iter().take(rem) {
            if code != 0b01 {
                non_missing += 1;
            }
        }
    }
    n_samples.saturating_sub(non_missing)
}

#[inline]
pub(crate) fn packed_row_missing_count_selected(
    row: &[u8],
    n_samples: usize,
    sample_idx: &[usize],
) -> usize {
    if is_identity_indices(sample_idx, n_samples) {
        return packed_row_missing_count(row, n_samples);
    }
    let mut missing = 0usize;
    for &sid in sample_idx.iter() {
        let code = (row[sid >> 2] >> ((sid & 3) * 2)) & 0b11;
        if code == 0b01 {
            missing += 1;
        }
    }
    missing
}

pub(crate) struct PackedPairLut {
    pub(crate) dot_obs0: [u8; 65536],
    pub(crate) obs_ct: [u8; 65536],
    pub(crate) obs_sum_i: [u8; 65536],
    pub(crate) obs_sum_j: [u8; 65536],
    pub(crate) obs_sum_i2: [u8; 65536],
    pub(crate) obs_sum_j2: [u8; 65536],
    pub(crate) obs_sum_ij: [u8; 65536],
}

#[inline]
pub(crate) fn code_to_dosage_or_zero(code: u8) -> u8 {
    match code {
        0b00 => 0,
        0b10 => 1,
        0b11 => 2,
        _ => 0, // missing
    }
}

pub(crate) fn packed_pair_lut() -> &'static PackedPairLut {
    static LUT: OnceLock<PackedPairLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let code4 = &packed_byte_lut().code4;
        let mut dot_obs0 = [0u8; 65536];
        let mut obs_ct = [0u8; 65536];
        let mut obs_sum_i = [0u8; 65536];
        let mut obs_sum_j = [0u8; 65536];
        let mut obs_sum_i2 = [0u8; 65536];
        let mut obs_sum_j2 = [0u8; 65536];
        let mut obs_sum_ij = [0u8; 65536];

        for bi in 0u16..=255 {
            for bj in 0u16..=255 {
                let idx = ((bi as usize) << 8) | (bj as usize);
                let ci = &code4[bi as usize];
                let cj = &code4[bj as usize];

                let mut dot_v: u8 = 0;
                let mut n_obs: u8 = 0;
                let mut sum_i_obs: u8 = 0;
                let mut sum_j_obs: u8 = 0;
                let mut sum_i2_obs: u8 = 0;
                let mut sum_j2_obs: u8 = 0;
                let mut sum_ij_obs: u8 = 0;
                for lane in 0..4usize {
                    let a = ci[lane];
                    let b = cj[lane];
                    let ga = code_to_dosage_or_zero(a);
                    let gb = code_to_dosage_or_zero(b);
                    dot_v = dot_v.saturating_add(ga.saturating_mul(gb));
                    if a != 0b01 && b != 0b01 {
                        n_obs = n_obs.saturating_add(1);
                        sum_i_obs = sum_i_obs.saturating_add(ga);
                        sum_j_obs = sum_j_obs.saturating_add(gb);
                        sum_i2_obs = sum_i2_obs.saturating_add(ga.saturating_mul(ga));
                        sum_j2_obs = sum_j2_obs.saturating_add(gb.saturating_mul(gb));
                        sum_ij_obs = sum_ij_obs.saturating_add(ga.saturating_mul(gb));
                    }
                }
                dot_obs0[idx] = dot_v;
                obs_ct[idx] = n_obs;
                obs_sum_i[idx] = sum_i_obs;
                obs_sum_j[idx] = sum_j_obs;
                obs_sum_i2[idx] = sum_i2_obs;
                obs_sum_j2[idx] = sum_j2_obs;
                obs_sum_ij[idx] = sum_ij_obs;
            }
        }

        PackedPairLut {
            dot_obs0,
            obs_ct,
            obs_sum_i,
            obs_sum_j,
            obs_sum_i2,
            obs_sum_j2,
            obs_sum_ij,
        }
    })
}

#[inline]
pub(crate) fn is_identity_indices(sample_idx: &[usize], n_samples: usize) -> bool {
    if sample_idx.len() != n_samples {
        return false;
    }
    sample_idx.iter().enumerate().all(|(i, &v)| i == v)
}

#[inline]
fn decode_row_centered_full_lut_scalar(
    row: &[u8],
    n_samples: usize,
    code4_lut: &[[u8; 4]; 256],
    value_lut: &[f32; 4],
    out_row: &mut [f32],
) {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut col = 0usize;
    for &b in row.iter().take(full_bytes) {
        let codes = &code4_lut[b as usize];
        out_row[col] = value_lut[codes[0] as usize];
        out_row[col + 1] = value_lut[codes[1] as usize];
        out_row[col + 2] = value_lut[codes[2] as usize];
        out_row[col + 3] = value_lut[codes[3] as usize];
        col += 4;
    }
    if rem > 0 {
        let codes = &code4_lut[row[full_bytes] as usize];
        for lane in 0..rem {
            out_row[col + lane] = value_lut[codes[lane] as usize];
        }
    }
}

#[inline]
fn bed_decode_simd_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let raw = std::env::var("JX_BED_DECODE_SIMD")
            .unwrap_or_else(|_| BED_DECODE_SIMD_DEFAULT.to_string());
        let key = raw.trim().to_ascii_lowercase();
        matches!(key.as_str(), "1" | "true" | "on" | "yes")
    })
}

const BED_DECODE_SIMD_MIN_BYTES: usize = 32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn bed_decode_avx2_runtime_available() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn decode_row_centered_full_lut_avx2(
    row: &[u8],
    full_bytes: usize,
    code4_lut: &[[u8; 4]; 256],
    value_lut: &[f32; 4],
    out_row: &mut [f32],
) {
    use x86_simd::*;
    const CHUNK_BYTES: usize = 16;
    const CHUNK_CODES: usize = CHUNK_BYTES * 4;
    let mut code_buf = [0u8; CHUNK_CODES];
    let mut byte_pos = 0usize;
    let mut out_pos = 0usize;
    let idx0 = _mm256_set1_epi32(0);
    let idx1 = _mm256_set1_epi32(1);
    let idx2 = _mm256_set1_epi32(2);
    let v0 = _mm256_set1_ps(value_lut[0]);
    let v1 = _mm256_set1_ps(value_lut[1]);
    let v2 = _mm256_set1_ps(value_lut[2]);
    let v3 = _mm256_set1_ps(value_lut[3]);

    while byte_pos + CHUNK_BYTES <= full_bytes {
        let src = &row[byte_pos..byte_pos + CHUNK_BYTES];
        for (k, &b) in src.iter().enumerate() {
            let codes = code4_lut[b as usize];
            let dst = k * 4;
            code_buf[dst] = codes[0];
            code_buf[dst + 1] = codes[1];
            code_buf[dst + 2] = codes[2];
            code_buf[dst + 3] = codes[3];
        }

        for cst in (0..CHUNK_CODES).step_by(8) {
            let cp = &code_buf[cst..cst + 8];
            let idx = _mm256_setr_epi32(
                cp[0] as i32,
                cp[1] as i32,
                cp[2] as i32,
                cp[3] as i32,
                cp[4] as i32,
                cp[5] as i32,
                cp[6] as i32,
                cp[7] as i32,
            );
            let mut outv = v3;
            let m2 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, idx2));
            outv = _mm256_blendv_ps(outv, v2, m2);
            let m1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, idx1));
            outv = _mm256_blendv_ps(outv, v1, m1);
            let m0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, idx0));
            outv = _mm256_blendv_ps(outv, v0, m0);
            _mm256_storeu_ps(out_row.as_mut_ptr().add(out_pos + cst), outv);
        }
        byte_pos += CHUNK_BYTES;
        out_pos += CHUNK_CODES;
    }

    if byte_pos < full_bytes {
        let tail = &row[byte_pos..full_bytes];
        let tail_codes = tail.len() * 4;
        for (k, &b) in tail.iter().enumerate() {
            let codes = code4_lut[b as usize];
            let dst = k * 4;
            code_buf[dst] = codes[0];
            code_buf[dst + 1] = codes[1];
            code_buf[dst + 2] = codes[2];
            code_buf[dst + 3] = codes[3];
        }
        let simd_codes = tail_codes - (tail_codes % 8);
        for cst in (0..simd_codes).step_by(8) {
            let cp = &code_buf[cst..cst + 8];
            let idx = _mm256_setr_epi32(
                cp[0] as i32,
                cp[1] as i32,
                cp[2] as i32,
                cp[3] as i32,
                cp[4] as i32,
                cp[5] as i32,
                cp[6] as i32,
                cp[7] as i32,
            );
            let mut outv = v3;
            let m2 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, idx2));
            outv = _mm256_blendv_ps(outv, v2, m2);
            let m1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, idx1));
            outv = _mm256_blendv_ps(outv, v1, m1);
            let m0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, idx0));
            outv = _mm256_blendv_ps(outv, v0, m0);
            _mm256_storeu_ps(out_row.as_mut_ptr().add(out_pos + cst), outv);
        }
        for t in simd_codes..tail_codes {
            out_row[out_pos + t] = value_lut[code_buf[t] as usize];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn bed_decode_neon_runtime_available() -> bool {
    static NEON: OnceLock<bool> = OnceLock::new();
    *NEON.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_row_centered_full_lut_neon(
    row: &[u8],
    full_bytes: usize,
    code4_lut: &[[u8; 4]; 256],
    value_lut: &[f32; 4],
    out_row: &mut [f32],
) {
    use std::arch::aarch64::*;
    const CHUNK_BYTES: usize = 16;
    const CHUNK_CODES: usize = CHUNK_BYTES * 4;
    let mut code_buf = [0u8; CHUNK_CODES];
    let mut byte_pos = 0usize;
    let mut out_pos = 0usize;
    let idx0 = vdupq_n_u32(0);
    let idx1 = vdupq_n_u32(1);
    let idx2 = vdupq_n_u32(2);
    let v0 = vdupq_n_f32(value_lut[0]);
    let v1 = vdupq_n_f32(value_lut[1]);
    let v2 = vdupq_n_f32(value_lut[2]);
    let v3 = vdupq_n_f32(value_lut[3]);

    while byte_pos + CHUNK_BYTES <= full_bytes {
        let src = &row[byte_pos..byte_pos + CHUNK_BYTES];
        for (k, &b) in src.iter().enumerate() {
            let codes = code4_lut[b as usize];
            let dst = k * 4;
            code_buf[dst] = codes[0];
            code_buf[dst + 1] = codes[1];
            code_buf[dst + 2] = codes[2];
            code_buf[dst + 3] = codes[3];
        }

        for cst in (0..CHUNK_CODES).step_by(4) {
            let idx_buf = [
                code_buf[cst] as u32,
                code_buf[cst + 1] as u32,
                code_buf[cst + 2] as u32,
                code_buf[cst + 3] as u32,
            ];
            let idx_u32 = vld1q_u32(idx_buf.as_ptr());
            let mut outv = v3;
            let m2 = vceqq_u32(idx_u32, idx2);
            outv = vbslq_f32(m2, v2, outv);
            let m1 = vceqq_u32(idx_u32, idx1);
            outv = vbslq_f32(m1, v1, outv);
            let m0 = vceqq_u32(idx_u32, idx0);
            outv = vbslq_f32(m0, v0, outv);
            vst1q_f32(out_row.as_mut_ptr().add(out_pos + cst), outv);
        }
        byte_pos += CHUNK_BYTES;
        out_pos += CHUNK_CODES;
    }

    if byte_pos < full_bytes {
        let tail = &row[byte_pos..full_bytes];
        let tail_codes = tail.len() * 4;
        for (k, &b) in tail.iter().enumerate() {
            let codes = code4_lut[b as usize];
            let dst = k * 4;
            code_buf[dst] = codes[0];
            code_buf[dst + 1] = codes[1];
            code_buf[dst + 2] = codes[2];
            code_buf[dst + 3] = codes[3];
        }
        let simd_codes = tail_codes - (tail_codes % 4);
        for cst in (0..simd_codes).step_by(4) {
            let idx_buf = [
                code_buf[cst] as u32,
                code_buf[cst + 1] as u32,
                code_buf[cst + 2] as u32,
                code_buf[cst + 3] as u32,
            ];
            let idx_u32 = vld1q_u32(idx_buf.as_ptr());
            let mut outv = v3;
            let m2 = vceqq_u32(idx_u32, idx2);
            outv = vbslq_f32(m2, v2, outv);
            let m1 = vceqq_u32(idx_u32, idx1);
            outv = vbslq_f32(m1, v1, outv);
            let m0 = vceqq_u32(idx_u32, idx0);
            outv = vbslq_f32(m0, v0, outv);
            vst1q_f32(out_row.as_mut_ptr().add(out_pos + cst), outv);
        }
        for t in simd_codes..tail_codes {
            out_row[out_pos + t] = value_lut[code_buf[t] as usize];
        }
    }
}

#[inline]
pub(crate) fn decode_row_centered_full_lut(
    row: &[u8],
    n_samples: usize,
    code4_lut: &[[u8; 4]; 256],
    value_lut: &[f32; 4],
    out_row: &mut [f32],
) {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    if full_bytes >= BED_DECODE_SIMD_MIN_BYTES && bed_decode_simd_enabled() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if bed_decode_avx2_runtime_available() {
            unsafe {
                decode_row_centered_full_lut_avx2(row, full_bytes, code4_lut, value_lut, out_row);
            }
            if rem > 0 {
                let base = full_bytes * 4;
                let codes = &code4_lut[row[full_bytes] as usize];
                for lane in 0..rem {
                    out_row[base + lane] = value_lut[codes[lane] as usize];
                }
            }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if bed_decode_neon_runtime_available() {
            unsafe {
                decode_row_centered_full_lut_neon(row, full_bytes, code4_lut, value_lut, out_row);
            }
            if rem > 0 {
                let base = full_bytes * 4;
                let codes = &code4_lut[row[full_bytes] as usize];
                for lane in 0..rem {
                    out_row[base + lane] = value_lut[codes[lane] as usize];
                }
            }
            return;
        }
    }
    decode_row_centered_full_lut_scalar(row, n_samples, code4_lut, value_lut, out_row);
}

#[inline]
pub(crate) fn decode_row_centered_full_lut_f64(
    row: &[u8],
    n_samples: usize,
    code4_lut: &[[u8; 4]; 256],
    value_lut: &[f64; 4],
    out_row: &mut [f64],
) {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut col = 0usize;
    for &b in row.iter().take(full_bytes) {
        let codes = &code4_lut[b as usize];
        out_row[col] = value_lut[codes[0] as usize];
        out_row[col + 1] = value_lut[codes[1] as usize];
        out_row[col + 2] = value_lut[codes[2] as usize];
        out_row[col + 3] = value_lut[codes[3] as usize];
        col += 4;
    }
    if rem > 0 {
        let codes = &code4_lut[row[full_bytes] as usize];
        for lane in 0..rem {
            out_row[col + lane] = value_lut[codes[lane] as usize];
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum SubsetDecodePlan {
    Gather {
        byte_idx: Vec<usize>,
        shift_bits: Vec<u8>,
    },
    NearFull {
        n_samples_full: usize,
        kept_ranges: Vec<(usize, usize)>,
    },
}

#[inline]
fn maybe_build_near_full_kept_ranges(
    sample_idx: &[usize],
    n_samples: usize,
) -> Option<Vec<(usize, usize)>> {
    if sample_idx.is_empty() || sample_idx.len() >= n_samples {
        return None;
    }
    let excluded_n = n_samples.saturating_sub(sample_idx.len());
    if excluded_n == 0 || excluded_n.saturating_mul(4) > sample_idx.len() {
        return None;
    }
    let mut kept_ranges = Vec::<(usize, usize)>::with_capacity(excluded_n.saturating_add(1));
    let mut start = sample_idx[0];
    if start >= n_samples {
        return None;
    }
    let mut prev = start;
    for &sid in sample_idx.iter().skip(1) {
        if sid >= n_samples || sid <= prev {
            return None;
        }
        if sid == prev + 1 {
            prev = sid;
            continue;
        }
        kept_ranges.push((start, prev + 1));
        start = sid;
        prev = sid;
    }
    kept_ranges.push((start, prev + 1));
    Some(kept_ranges)
}

impl SubsetDecodePlan {
    #[inline]
    pub(crate) fn from_sample_idx(sample_idx: &[usize]) -> Self {
        let mut byte_idx = Vec::with_capacity(sample_idx.len());
        let mut shift_bits = Vec::with_capacity(sample_idx.len());
        for &sid in sample_idx {
            byte_idx.push(sid >> 2);
            shift_bits.push(((sid & 3) * 2) as u8);
        }
        Self::Gather {
            byte_idx,
            shift_bits,
        }
    }

    #[inline]
    pub(crate) fn from_sample_idx_with_n_samples(sample_idx: &[usize], n_samples: usize) -> Self {
        if let Some(kept_ranges) = maybe_build_near_full_kept_ranges(sample_idx, n_samples) {
            return Self::NearFull {
                n_samples_full: n_samples,
                kept_ranges,
            };
        }
        Self::from_sample_idx(sample_idx)
    }
}

#[inline]
fn compact_full_decode_by_kept_ranges(
    full_row: &[f32],
    kept_ranges: &[(usize, usize)],
    out_row: &mut [f32],
) {
    let mut dst = 0usize;
    for &(start, end) in kept_ranges {
        let len = end.saturating_sub(start);
        out_row[dst..dst + len].copy_from_slice(&full_row[start..end]);
        dst += len;
    }
    debug_assert_eq!(dst, out_row.len());
}

#[inline]
fn compact_full_decode_by_kept_ranges_f64_with_stats(
    full_row: &[f64],
    kept_ranges: &[(usize, usize)],
    rhs: &[f64],
    out_row: &mut [f64],
) -> (f64, f64) {
    let mut dst = 0usize;
    let mut sy = 0.0_f64;
    let mut ss = 0.0_f64;
    for &(start, end) in kept_ranges {
        let len = end.saturating_sub(start);
        let src = &full_row[start..end];
        let rhs_slice = &rhs[dst..dst + len];
        let dst_slice = &mut out_row[dst..dst + len];
        dst_slice.copy_from_slice(src);
        for (g, &rv) in src.iter().zip(rhs_slice.iter()) {
            sy += *g * rv;
            ss += *g * *g;
        }
        dst += len;
    }
    debug_assert_eq!(dst, out_row.len());
    (sy, ss)
}

#[inline]
fn subset_plan_full_scratch_len(plan: Option<&SubsetDecodePlan>) -> usize {
    match plan {
        Some(SubsetDecodePlan::NearFull { n_samples_full, .. }) => *n_samples_full,
        _ => 0usize,
    }
}

#[inline]
fn decode_subset_plan_row_f32(
    row: &[u8],
    plan: &SubsetDecodePlan,
    code4_lut: &[[u8; 4]; 256],
    value_lut: &[f32; 4],
    full_row_scratch: &mut [f32],
    out_row: &mut [f32],
) {
    match plan {
        SubsetDecodePlan::Gather { .. } => {
            decode_subset_with_plan(row, plan, value_lut, out_row);
        }
        SubsetDecodePlan::NearFull {
            n_samples_full,
            kept_ranges,
        } => {
            debug_assert!(full_row_scratch.len() >= *n_samples_full);
            let full_row = &mut full_row_scratch[..*n_samples_full];
            decode_row_centered_full_lut(row, *n_samples_full, code4_lut, value_lut, full_row);
            compact_full_decode_by_kept_ranges(full_row, kept_ranges, out_row);
        }
    }
}

#[inline]
fn decode_subset_with_plan_f64(
    row: &[u8],
    plan: &SubsetDecodePlan,
    value_lut: &[f64; 4],
    out_row: &mut [f64],
) {
    let SubsetDecodePlan::Gather {
        byte_idx,
        shift_bits,
    } = plan
    else {
        unreachable!("subset gather decode requires Gather plan");
    };
    let n = byte_idx.len();
    for t in 0..n {
        let code = (row[byte_idx[t]] >> shift_bits[t]) & 0b11;
        out_row[t] = value_lut[code as usize];
    }
}

#[inline]
fn decode_subset_plan_row_f64_with_stats(
    row: &[u8],
    plan: &SubsetDecodePlan,
    code4_lut: &[[u8; 4]; 256],
    value_lut: &[f64; 4],
    rhs: &[f64],
    full_row_scratch: &mut [f64],
    out_row: &mut [f64],
) -> (f64, f64) {
    match plan {
        SubsetDecodePlan::Gather { .. } => {
            decode_subset_with_plan_f64(row, plan, value_lut, out_row);
            let mut sy = 0.0_f64;
            let mut ss = 0.0_f64;
            for (g, &rv) in out_row.iter().zip(rhs.iter()) {
                sy += *g * rv;
                ss += *g * *g;
            }
            (sy, ss)
        }
        SubsetDecodePlan::NearFull {
            n_samples_full,
            kept_ranges,
        } => {
            debug_assert!(full_row_scratch.len() >= *n_samples_full);
            let full_row = &mut full_row_scratch[..*n_samples_full];
            decode_row_centered_full_lut_f64(row, *n_samples_full, code4_lut, value_lut, full_row);
            compact_full_decode_by_kept_ranges_f64_with_stats(full_row, kept_ranges, rhs, out_row)
        }
    }
}

#[inline]
fn bed_subset_decode_simd_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let raw = std::env::var("JX_BED_SUBSET_SIMD")
            .unwrap_or_else(|_| BED_SUBSET_SIMD_DEFAULT.to_string());
        let key = raw.trim().to_ascii_lowercase();
        !matches!(key.as_str(), "0" | "false" | "off" | "no")
    })
}

const BED_SUBSET_SIMD_MIN_LEN: usize = 32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn bed_subset_decode_avx2_runtime_available() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn decode_subset_with_plan_avx2(
    row: &[u8],
    plan: &SubsetDecodePlan,
    value_lut: &[f32; 4],
    out_row: &mut [f32],
) {
    use x86_simd::*;
    let SubsetDecodePlan::Gather {
        byte_idx,
        shift_bits,
    } = plan
    else {
        unreachable!("AVX2 subset gather decode requires Gather plan");
    };
    let n = byte_idx.len();
    let idx0 = _mm256_set1_epi32(0);
    let idx1 = _mm256_set1_epi32(1);
    let idx2 = _mm256_set1_epi32(2);
    let v0 = _mm256_set1_ps(value_lut[0]);
    let v1 = _mm256_set1_ps(value_lut[1]);
    let v2 = _mm256_set1_ps(value_lut[2]);
    let v3 = _mm256_set1_ps(value_lut[3]);
    let mut j = 0usize;
    while j + 8 <= n {
        let c0 = ((row[byte_idx[j]] >> shift_bits[j]) & 0b11) as i32;
        let c1 = ((row[byte_idx[j + 1]] >> shift_bits[j + 1]) & 0b11) as i32;
        let c2 = ((row[byte_idx[j + 2]] >> shift_bits[j + 2]) & 0b11) as i32;
        let c3 = ((row[byte_idx[j + 3]] >> shift_bits[j + 3]) & 0b11) as i32;
        let c4 = ((row[byte_idx[j + 4]] >> shift_bits[j + 4]) & 0b11) as i32;
        let c5 = ((row[byte_idx[j + 5]] >> shift_bits[j + 5]) & 0b11) as i32;
        let c6 = ((row[byte_idx[j + 6]] >> shift_bits[j + 6]) & 0b11) as i32;
        let c7 = ((row[byte_idx[j + 7]] >> shift_bits[j + 7]) & 0b11) as i32;
        let idx = _mm256_setr_epi32(c0, c1, c2, c3, c4, c5, c6, c7);
        let mut outv = v3;
        let m2 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, idx2));
        outv = _mm256_blendv_ps(outv, v2, m2);
        let m1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, idx1));
        outv = _mm256_blendv_ps(outv, v1, m1);
        let m0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, idx0));
        outv = _mm256_blendv_ps(outv, v0, m0);
        _mm256_storeu_ps(out_row.as_mut_ptr().add(j), outv);
        j += 8;
    }
    for t in j..n {
        let code = (row[byte_idx[t]] >> shift_bits[t]) & 0b11;
        out_row[t] = value_lut[code as usize];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn bed_subset_decode_neon_runtime_available() -> bool {
    static NEON: OnceLock<bool> = OnceLock::new();
    *NEON.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_subset_with_plan_neon(
    row: &[u8],
    plan: &SubsetDecodePlan,
    value_lut: &[f32; 4],
    out_row: &mut [f32],
) {
    use std::arch::aarch64::*;
    let SubsetDecodePlan::Gather {
        byte_idx,
        shift_bits,
    } = plan
    else {
        unreachable!("NEON subset gather decode requires Gather plan");
    };
    let n = byte_idx.len();
    let idx0 = vdupq_n_u32(0);
    let idx1 = vdupq_n_u32(1);
    let idx2 = vdupq_n_u32(2);
    let v0 = vdupq_n_f32(value_lut[0]);
    let v1 = vdupq_n_f32(value_lut[1]);
    let v2 = vdupq_n_f32(value_lut[2]);
    let v3 = vdupq_n_f32(value_lut[3]);
    let mut j = 0usize;
    while j + 4 <= n {
        let codes = [
            ((row[byte_idx[j]] >> shift_bits[j]) & 0b11) as u32,
            ((row[byte_idx[j + 1]] >> shift_bits[j + 1]) & 0b11) as u32,
            ((row[byte_idx[j + 2]] >> shift_bits[j + 2]) & 0b11) as u32,
            ((row[byte_idx[j + 3]] >> shift_bits[j + 3]) & 0b11) as u32,
        ];
        let idx = vld1q_u32(codes.as_ptr());
        let mut outv = v3;
        let m2 = vceqq_u32(idx, idx2);
        outv = vbslq_f32(m2, v2, outv);
        let m1 = vceqq_u32(idx, idx1);
        outv = vbslq_f32(m1, v1, outv);
        let m0 = vceqq_u32(idx, idx0);
        outv = vbslq_f32(m0, v0, outv);
        vst1q_f32(out_row.as_mut_ptr().add(j), outv);
        j += 4;
    }
    for t in j..n {
        let code = (row[byte_idx[t]] >> shift_bits[t]) & 0b11;
        out_row[t] = value_lut[code as usize];
    }
}

#[inline]
pub(crate) fn decode_subset_with_plan(
    row: &[u8],
    plan: &SubsetDecodePlan,
    value_lut: &[f32; 4],
    out_row: &mut [f32],
) {
    let SubsetDecodePlan::Gather {
        byte_idx,
        shift_bits,
    } = plan
    else {
        unreachable!("subset gather decode requires Gather plan");
    };
    let n = byte_idx.len();
    if n >= BED_SUBSET_SIMD_MIN_LEN && bed_subset_decode_simd_enabled() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if bed_subset_decode_avx2_runtime_available() {
            unsafe { decode_subset_with_plan_avx2(row, plan, value_lut, out_row) };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if bed_subset_decode_neon_runtime_available() {
            unsafe { decode_subset_with_plan_neon(row, plan, value_lut, out_row) };
            return;
        }
    }
    for t in 0..n {
        let code = (row[byte_idx[t]] >> shift_bits[t]) & 0b11;
        out_row[t] = value_lut[code as usize];
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_mean_imputed_additive_packed_block_rows_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    row_start: usize,
    out: &mut [f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Ok(());
    }
    if out.len() % n_out != 0 {
        return Err(
            "decode_mean_imputed_additive_packed_block_rows_f32: output length mismatch"
                .to_string(),
        );
    }
    let cur_rows = out.len() / n_out;
    if cur_rows == 0 {
        return Ok(());
    }
    let subset_plan = if full_sample_fast {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
            sample_idx, n_samples,
        ))
    };
    let full_scratch_len = subset_plan_full_scratch_len(subset_plan.as_ref());

    let decode_one = |local_row: usize, out_row: &mut [f32], full_row_scratch: &mut [f32]| {
        let row_idx_local = row_start + local_row;
        let row_idx_packed = packed_row_indices
            .map(|idx| idx[row_idx_local])
            .unwrap_or(row_idx_local);
        let row =
            &packed_flat[row_idx_packed * bytes_per_snp..(row_idx_packed + 1) * bytes_per_snp];
        let mean_g = (2.0_f32 * row_maf[row_idx_local]).clamp(0.0_f32, 2.0_f32);
        let value_lut: [f32; 4] = if row_flip[row_idx_local] {
            [2.0_f32, mean_g, 1.0_f32, 0.0_f32]
        } else {
            [0.0_f32, mean_g, 1.0_f32, 2.0_f32]
        };
        if full_sample_fast {
            decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, out_row);
            return;
        }
        if let Some(plan) = subset_plan.as_ref() {
            decode_subset_plan_row_f32(row, plan, code4_lut, &value_lut, full_row_scratch, out_row);
        }
    };

    if let Some(tp) = pool {
        tp.install(|| {
            out.par_chunks_mut(n_out).enumerate().for_each_init(
                || vec![0.0_f32; full_scratch_len],
                |full_row_scratch, (local_row, out_row)| {
                    decode_one(local_row, out_row, full_row_scratch.as_mut_slice())
                },
            );
        });
    } else {
        let mut full_row_scratch = vec![0.0_f32; full_scratch_len];
        for (local_row, out_row) in out.chunks_mut(n_out).enumerate() {
            decode_one(local_row, out_row, full_row_scratch.as_mut_slice());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_mean_imputed_additive_packed_block_rows_f64_stats(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    row_start: usize,
    rhs: &[f64],
    out: &mut [f64],
    rhs_dot_out: &mut [f64],
    ss_out: &mut [f64],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Ok(());
    }
    if rhs.len() != n_out {
        return Err(
            "decode_mean_imputed_additive_packed_block_rows_f64_stats: rhs length mismatch"
                .to_string(),
        );
    }
    if out.len() % n_out != 0 {
        return Err(
            "decode_mean_imputed_additive_packed_block_rows_f64_stats: output length mismatch"
                .to_string(),
        );
    }
    let cur_rows = out.len() / n_out;
    if cur_rows == 0 {
        return Ok(());
    }
    if rhs_dot_out.len() != cur_rows || ss_out.len() != cur_rows {
        return Err(
            "decode_mean_imputed_additive_packed_block_rows_f64_stats: stats length mismatch"
                .to_string(),
        );
    }
    let subset_plan = if full_sample_fast {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
            sample_idx, n_samples,
        ))
    };
    let full_scratch_len = subset_plan_full_scratch_len(subset_plan.as_ref());

    let decode_one = |local_row: usize,
                      out_row: &mut [f64],
                      rhs_dot_slot: &mut f64,
                      ss_slot: &mut f64,
                      full_row_scratch: &mut [f64]| {
        let row_idx_local = row_start + local_row;
        let row_idx_packed = packed_row_indices
            .map(|idx| idx[row_idx_local])
            .unwrap_or(row_idx_local);
        let row =
            &packed_flat[row_idx_packed * bytes_per_snp..(row_idx_packed + 1) * bytes_per_snp];
        let mean_g = (2.0_f64 * row_maf[row_idx_local] as f64).clamp(0.0_f64, 2.0_f64);
        let value_lut: [f64; 4] = if row_flip[row_idx_local] {
            [2.0_f64, mean_g, 1.0_f64, 0.0_f64]
        } else {
            [0.0_f64, mean_g, 1.0_f64, 2.0_f64]
        };
        let (sy, ss) = if full_sample_fast {
            decode_row_centered_full_lut_f64(row, n_samples, code4_lut, &value_lut, out_row);
            let mut sy = 0.0_f64;
            let mut ss = 0.0_f64;
            for (g, &rv) in out_row.iter().zip(rhs.iter()) {
                sy += *g * rv;
                ss += *g * *g;
            }
            (sy, ss)
        } else {
            decode_subset_plan_row_f64_with_stats(
                row,
                subset_plan.as_ref().expect("subset plan missing"),
                code4_lut,
                &value_lut,
                rhs,
                full_row_scratch,
                out_row,
            )
        };
        *rhs_dot_slot = sy;
        *ss_slot = ss;
    };

    if let Some(tp) = pool {
        tp.install(|| {
            out.par_chunks_mut(n_out)
                .zip(rhs_dot_out.par_iter_mut())
                .zip(ss_out.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || vec![0.0_f64; full_scratch_len],
                    |full_row_scratch, (local_row, ((out_row, rhs_dot_slot), ss_slot))| {
                        decode_one(
                            local_row,
                            out_row,
                            rhs_dot_slot,
                            ss_slot,
                            full_row_scratch.as_mut_slice(),
                        )
                    },
                );
        });
    } else {
        let mut full_row_scratch = vec![0.0_f64; full_scratch_len];
        for (local_row, ((out_row, rhs_dot_slot), ss_slot)) in out
            .chunks_mut(n_out)
            .zip(rhs_dot_out.iter_mut())
            .zip(ss_out.iter_mut())
            .enumerate()
        {
            decode_one(
                local_row,
                out_row,
                rhs_dot_slot,
                ss_slot,
                full_row_scratch.as_mut_slice(),
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_standardized_packed_block_rows_f32_with_plan(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    subset_plan: Option<&SubsetDecodePlan>,
    packed_row_indices: Option<&[usize]>,
    row_start: usize,
    out: &mut [f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Ok(());
    }
    if out.len() % n_out != 0 {
        return Err("decode_standardized_packed_block_f32: output length mismatch".to_string());
    }
    let cur_rows = out.len() / n_out;
    if cur_rows == 0 {
        return Ok(());
    }
    let local_subset_plan = if full_sample_fast {
        None
    } else {
        subset_plan.cloned().or_else(|| {
            Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
                sample_idx, n_samples,
            ))
        })
    };
    let full_scratch_len = subset_plan_full_scratch_len(local_subset_plan.as_ref());

    let decode_one = |local_row: usize, out_row: &mut [f32], full_row_scratch: &mut [f32]| {
        let row_idx_local = row_start + local_row;
        let row_idx_packed = packed_row_indices
            .map(|idx| idx[row_idx_local])
            .unwrap_or(row_idx_local);
        let row =
            &packed_flat[row_idx_packed * bytes_per_snp..(row_idx_packed + 1) * bytes_per_snp];
        let mean_g = row_mean[row_idx_local];
        let inv_sd = row_inv_sd[row_idx_local];
        let value_lut: [f32; 4] = if row_flip[row_idx_local] {
            [
                (2.0_f32 - mean_g) * inv_sd,
                0.0_f32,
                (1.0_f32 - mean_g) * inv_sd,
                (0.0_f32 - mean_g) * inv_sd,
            ]
        } else {
            [
                (0.0_f32 - mean_g) * inv_sd,
                0.0_f32,
                (1.0_f32 - mean_g) * inv_sd,
                (2.0_f32 - mean_g) * inv_sd,
            ]
        };
        if full_sample_fast {
            decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, out_row);
            return;
        }
        if let Some(plan) = local_subset_plan.as_ref() {
            decode_subset_plan_row_f32(row, plan, code4_lut, &value_lut, full_row_scratch, out_row);
        }
    };

    if let Some(tp) = pool {
        tp.install(|| {
            out.par_chunks_mut(n_out).enumerate().for_each_init(
                || vec![0.0_f32; full_scratch_len],
                |full_row_scratch, (local_row, out_row)| {
                    decode_one(local_row, out_row, full_row_scratch.as_mut_slice())
                },
            );
        });
    } else {
        let mut full_row_scratch = vec![0.0_f32; full_scratch_len];
        for (local_row, out_row) in out.chunks_mut(n_out).enumerate() {
            decode_one(local_row, out_row, full_row_scratch.as_mut_slice());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_standardized_packed_block_rows_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    row_start: usize,
    out: &mut [f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    decode_standardized_packed_block_rows_f32_with_plan(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_idx,
        full_sample_fast,
        None,
        packed_row_indices,
        row_start,
        out,
        code4_lut,
        pool,
    )
}

pub(crate) fn decode_standardized_packed_block_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    row_start: usize,
    out: &mut [f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    decode_standardized_packed_block_rows_f32(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_idx,
        full_sample_fast,
        None,
        row_start,
        out,
        code4_lut,
        pool,
    )
}

#[inline]
pub(crate) fn adaptive_grm_block_rows(
    requested_block_rows: usize,
    m: usize,
    n_out: usize,
    subset_full_scratch_len: usize,
    threads: usize,
) -> usize {
    let base = requested_block_rows.max(1).min(m.max(1));
    let exact_mode = std::env::var("JX_GRM_BLOCK_EXACT")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);
    if exact_mode {
        return base;
    }
    let target_mb = parse_positive_env_f64(&[
        "JX_GRM_PACKED_BLOCK_TARGET_MB",
        "JX_GRM_BLOCK_TARGET_MB",
        "JX_BED_BLOCK_TARGET_MB",
        "JANUSX_BED_BLOCK_TARGET_MB",
    ])
    .unwrap_or(1024.0_f64);
    let workers = if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
    };
    let reserve_bytes = subset_full_scratch_len
        .saturating_mul(4)
        .saturating_mul(workers.max(1));
    let row_bytes = n_out.saturating_mul(4).max(1);
    let cap_rows =
        block_rows_from_memory_target_mb(target_mb, row_bytes, m.max(1), 1, 1, reserve_bytes);
    base.min(cap_rows)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn decode_subset_row_from_full_scratch(
    row: &[u8],
    n_samples: usize,
    sample_idx: &[usize],
    flip: bool,
    default_mean_g: f32,
    method: usize,
    eps: f32,
    code4_lut: &[[u8; 4]; 256],
    full_row: &mut [f32],
    out_row: &mut [f32],
) -> (f64, f64) {
    debug_assert!(full_row.len() >= n_samples);
    debug_assert_eq!(out_row.len(), sample_idx.len());
    let n = sample_idx.len();

    let value_lut: [f32; 4] = if flip {
        [2.0_f32, -9.0_f32, 1.0_f32, 0.0_f32]
    } else {
        [0.0_f32, -9.0_f32, 1.0_f32, 2.0_f32]
    };
    decode_row_centered_full_lut(
        row,
        n_samples,
        code4_lut,
        &value_lut,
        &mut full_row[..n_samples],
    );

    let mut obs_n: usize = 0;
    let mut obs_sum = 0.0_f64;
    let mut obs_sq = 0.0_f64;
    for (j, &sid) in sample_idx.iter().enumerate() {
        let gv = full_row[sid];
        out_row[j] = gv;
        if gv >= 0.0_f32 {
            let g = gv as f64;
            obs_n += 1;
            obs_sum += g;
            obs_sq += g * g;
        }
    }

    let (mean_g, var_centered, var_standardized) = if obs_n > 0 {
        let mean = obs_sum / obs_n as f64;
        let p_sub = (0.5_f64 * mean).clamp(0.0, 1.0);
        let v_std = (2.0_f64 * p_sub * (1.0_f64 - p_sub)).max(0.0);
        let ss = (obs_sq - (obs_sum * obs_sum) / obs_n as f64).max(0.0);
        let v_center = (ss / n as f64).max(0.0);
        (mean as f32, v_center as f32, v_std as f32)
    } else {
        (default_mean_g, 0.0_f32, 0.0_f32)
    };

    let std_scale = if method == 2 {
        if var_standardized > eps {
            1.0_f32 / var_standardized.sqrt()
        } else {
            0.0_f32
        }
    } else {
        1.0_f32
    };
    for v in out_row.iter_mut() {
        let gv = if *v >= 0.0_f32 { *v } else { mean_g };
        *v = (gv - mean_g) * std_scale;
    }

    let row_sum = (mean_g as f64) * (n as f64);
    (var_centered as f64, row_sum)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn decode_subset_row_from_full_scratch_f64(
    row: &[u8],
    n_samples: usize,
    sample_idx: &[usize],
    flip: bool,
    default_mean_g: f64,
    method: usize,
    eps: f64,
    code4_lut: &[[u8; 4]; 256],
    full_row: &mut [f64],
    out_row: &mut [f64],
) -> (f64, f64) {
    debug_assert!(full_row.len() >= n_samples);
    debug_assert_eq!(out_row.len(), sample_idx.len());
    let n = sample_idx.len();

    let value_lut: [f64; 4] = if flip {
        [2.0_f64, -9.0_f64, 1.0_f64, 0.0_f64]
    } else {
        [0.0_f64, -9.0_f64, 1.0_f64, 2.0_f64]
    };
    decode_row_centered_full_lut_f64(
        row,
        n_samples,
        code4_lut,
        &value_lut,
        &mut full_row[..n_samples],
    );

    let mut obs_n: usize = 0;
    let mut obs_sum = 0.0_f64;
    let mut obs_sq = 0.0_f64;
    for (j, &sid) in sample_idx.iter().enumerate() {
        let gv = full_row[sid];
        out_row[j] = gv;
        if gv >= 0.0_f64 {
            obs_n += 1;
            obs_sum += gv;
            obs_sq += gv * gv;
        }
    }

    let (mean_g, var_centered, var_standardized) = if obs_n > 0 {
        let mean = obs_sum / obs_n as f64;
        let p_sub = (0.5_f64 * mean).clamp(0.0, 1.0);
        let v_std = (2.0_f64 * p_sub * (1.0_f64 - p_sub)).max(0.0);
        let ss = (obs_sq - (obs_sum * obs_sum) / obs_n as f64).max(0.0);
        let v_center = (ss / n as f64).max(0.0);
        (mean, v_center, v_std)
    } else {
        (default_mean_g, 0.0_f64, 0.0_f64)
    };

    let std_scale = if method == 2 {
        if var_standardized > eps {
            1.0_f64 / var_standardized.sqrt()
        } else {
            0.0_f64
        }
    } else {
        1.0_f64
    };
    for v in out_row.iter_mut() {
        let gv = if *v >= 0.0_f64 { *v } else { mean_g };
        *v = (gv - mean_g) * std_scale;
    }

    let row_sum = mean_g * (n as f64);
    (var_centered, row_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_codes(codes: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; codes.len().div_ceil(4)];
        for (i, &code) in codes.iter().enumerate() {
            out[i >> 2] |= (code & 0b11) << ((i & 3) * 2);
        }
        out
    }

    #[test]
    fn near_full_additive_decode_matches_gather_decode() {
        let codes = [0u8, 2, 3, 1, 0, 3, 2, 0];
        let packed = pack_codes(&codes);
        let sample_idx = vec![0usize, 1, 2, 3, 5, 6, 7];
        let code4_lut = &packed_byte_lut().code4;
        let row_flip = [false];
        let row_maf = [0.25_f32];
        let mut out_near_full = vec![0.0_f32; sample_idx.len()];
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed.as_slice(),
            packed.len(),
            codes.len(),
            &row_flip,
            &row_maf,
            sample_idx.as_slice(),
            false,
            None,
            0,
            out_near_full.as_mut_slice(),
            code4_lut,
            None,
        )
        .expect("near-full additive decode should succeed");

        let mean_g = 2.0_f32 * row_maf[0];
        let value_lut = [0.0_f32, mean_g, 1.0_f32, 2.0_f32];
        let gather_plan = SubsetDecodePlan::from_sample_idx(sample_idx.as_slice());
        let mut out_gather = vec![0.0_f32; sample_idx.len()];
        decode_subset_with_plan(
            packed.as_slice(),
            &gather_plan,
            &value_lut,
            out_gather.as_mut_slice(),
        );

        assert_eq!(out_near_full, out_gather);
    }

    #[test]
    fn near_full_standardized_decode_matches_gather_decode() {
        let codes = [3u8, 2, 0, 1, 3, 2, 0, 2];
        let packed = pack_codes(&codes);
        let sample_idx = vec![0usize, 1, 2, 4, 5, 6, 7];
        let code4_lut = &packed_byte_lut().code4;
        let row_flip = [true];
        let row_mean = [0.6_f32];
        let row_inv_sd = [1.25_f32];
        let near_full_plan =
            SubsetDecodePlan::from_sample_idx_with_n_samples(sample_idx.as_slice(), codes.len());
        let gather_plan = SubsetDecodePlan::from_sample_idx(sample_idx.as_slice());
        let mut out_near_full = vec![0.0_f32; sample_idx.len()];
        let mut out_gather = vec![0.0_f32; sample_idx.len()];
        decode_standardized_packed_block_rows_f32_with_plan(
            packed.as_slice(),
            packed.len(),
            codes.len(),
            &row_flip,
            &row_mean,
            &row_inv_sd,
            sample_idx.as_slice(),
            false,
            Some(&near_full_plan),
            None,
            0,
            out_near_full.as_mut_slice(),
            code4_lut,
            None,
        )
        .expect("near-full standardized decode should succeed");
        decode_standardized_packed_block_rows_f32_with_plan(
            packed.as_slice(),
            packed.len(),
            codes.len(),
            &row_flip,
            &row_mean,
            &row_inv_sd,
            sample_idx.as_slice(),
            false,
            Some(&gather_plan),
            None,
            0,
            out_gather.as_mut_slice(),
            code4_lut,
            None,
        )
        .expect("gather standardized decode should succeed");

        for (lhs, rhs) in out_near_full.iter().zip(out_gather.iter()) {
            assert!((lhs - rhs).abs() <= 1e-6_f32);
        }
    }
}
