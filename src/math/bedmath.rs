use rayon::prelude::*;
#[cfg(target_arch = "x86")]
use std::arch::x86 as x86_simd;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as x86_simd;
use std::sync::{Arc, OnceLock};

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
    pub(crate) sq_sum: [u8; 256],
    pub(crate) code4: [[u8; 4]; 256],
}

pub(crate) fn packed_byte_lut() -> &'static PackedByteLut {
    static LUT: OnceLock<PackedByteLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut nonmiss = [0u8; 256];
        let mut alt_sum = [0u8; 256];
        let mut sq_sum = [0u8; 256];
        let mut code4 = [[0u8; 4]; 256];
        for b in 0u16..=255 {
            let byte = b as u8;
            let mut nm = 0u8;
            let mut alt = 0u8;
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
            sq_sum[idx] = sq;
            code4[idx] = codes;
        }
        PackedByteLut {
            nonmiss,
            alt_sum,
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
        let raw = std::env::var("JX_BED_DECODE_SIMD").unwrap_or_else(|_| "0".to_string());
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
            let p = code_buf.as_ptr().add(cst);
            let idx_u8 = vld1_u8(p);
            let idx_u16 = vmovl_u8(idx_u8);
            let idx_u32 = vmovl_u16(vget_low_u16(idx_u16));
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
            let p = code_buf.as_ptr().add(cst);
            let idx_u8 = vld1_u8(p);
            let idx_u16 = vmovl_u8(idx_u8);
            let idx_u32 = vmovl_u16(vget_low_u16(idx_u16));
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
pub(crate) struct SubsetDecodePlan {
    byte_idx: Vec<usize>,
    shift_bits: Vec<u8>,
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
        Self {
            byte_idx,
            shift_bits,
        }
    }
}

#[inline]
fn bed_subset_decode_simd_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let raw = std::env::var("JX_BED_SUBSET_SIMD").unwrap_or_else(|_| "1".to_string());
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
    let n = plan.byte_idx.len();
    let idx0 = _mm256_set1_epi32(0);
    let idx1 = _mm256_set1_epi32(1);
    let idx2 = _mm256_set1_epi32(2);
    let v0 = _mm256_set1_ps(value_lut[0]);
    let v1 = _mm256_set1_ps(value_lut[1]);
    let v2 = _mm256_set1_ps(value_lut[2]);
    let v3 = _mm256_set1_ps(value_lut[3]);
    let mut j = 0usize;
    while j + 8 <= n {
        let c0 = ((row[plan.byte_idx[j]] >> plan.shift_bits[j]) & 0b11) as i32;
        let c1 = ((row[plan.byte_idx[j + 1]] >> plan.shift_bits[j + 1]) & 0b11) as i32;
        let c2 = ((row[plan.byte_idx[j + 2]] >> plan.shift_bits[j + 2]) & 0b11) as i32;
        let c3 = ((row[plan.byte_idx[j + 3]] >> plan.shift_bits[j + 3]) & 0b11) as i32;
        let c4 = ((row[plan.byte_idx[j + 4]] >> plan.shift_bits[j + 4]) & 0b11) as i32;
        let c5 = ((row[plan.byte_idx[j + 5]] >> plan.shift_bits[j + 5]) & 0b11) as i32;
        let c6 = ((row[plan.byte_idx[j + 6]] >> plan.shift_bits[j + 6]) & 0b11) as i32;
        let c7 = ((row[plan.byte_idx[j + 7]] >> plan.shift_bits[j + 7]) & 0b11) as i32;
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
        let code = (row[plan.byte_idx[t]] >> plan.shift_bits[t]) & 0b11;
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
    let n = plan.byte_idx.len();
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
            ((row[plan.byte_idx[j]] >> plan.shift_bits[j]) & 0b11) as u32,
            ((row[plan.byte_idx[j + 1]] >> plan.shift_bits[j + 1]) & 0b11) as u32,
            ((row[plan.byte_idx[j + 2]] >> plan.shift_bits[j + 2]) & 0b11) as u32,
            ((row[plan.byte_idx[j + 3]] >> plan.shift_bits[j + 3]) & 0b11) as u32,
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
        let code = (row[plan.byte_idx[t]] >> plan.shift_bits[t]) & 0b11;
        out_row[t] = value_lut[code as usize];
    }
}

#[inline]
fn decode_subset_with_plan(
    row: &[u8],
    plan: &SubsetDecodePlan,
    value_lut: &[f32; 4],
    out_row: &mut [f32],
) {
    let n = plan.byte_idx.len();
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
        let code = (row[plan.byte_idx[t]] >> plan.shift_bits[t]) & 0b11;
        out_row[t] = value_lut[code as usize];
    }
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
    let subset_plan = if full_sample_fast {
        None
    } else {
        Some(SubsetDecodePlan::from_sample_idx(sample_idx))
    };

    let decode_one = |local_row: usize, out_row: &mut [f32]| {
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
        if let Some(plan) = subset_plan.as_ref() {
            decode_subset_with_plan(row, plan, &value_lut, out_row);
        }
    };

    if let Some(tp) = pool {
        tp.install(|| {
            out.par_chunks_mut(n_out)
                .enumerate()
                .for_each(|(local_row, out_row)| decode_one(local_row, out_row));
        });
    } else {
        for (local_row, out_row) in out.chunks_mut(n_out).enumerate() {
            decode_one(local_row, out_row);
        }
    }
    Ok(())
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
    let target_mb = std::env::var("JX_GRM_PACKED_BLOCK_TARGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(64.0_f64);
    if !(target_mb.is_finite() && target_mb > 0.0) {
        return base;
    }
    let target_bytes = (target_mb * 1024.0_f64 * 1024.0_f64) as usize;
    if target_bytes == 0 {
        return base;
    }
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
    if reserve_bytes >= target_bytes {
        return 1;
    }
    let cap_rows = ((target_bytes - reserve_bytes) / row_bytes).max(1);
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
