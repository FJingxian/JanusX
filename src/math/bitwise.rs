use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;
use std::hint::black_box;
use std::sync::OnceLock;
use std::time::Instant;

const PAR_REDUCE_MIN_WORDS: usize = 16 * 1024;
const PAR_MUTATE_MIN_WORDS: usize = 16 * 1024;
const PAR_CHUNK_WORDS: usize = 16 * 1024;

#[inline]
fn should_parallelize(len: usize, min_len: usize) -> bool {
    len >= min_len && rayon::current_num_threads() > 1
}

#[inline]
fn parse_env_bool(name: &str) -> bool {
    std::env::var(name).map_or(false, |v| {
        let t = v.trim().to_ascii_lowercase();
        matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
    })
}

#[inline]
fn parse_env_bool_default(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |v| {
        let t = v.trim().to_ascii_lowercase();
        if matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on") {
            true
        } else if matches!(t.as_str(), "0" | "false" | "no" | "n" | "off") {
            false
        } else {
            default
        }
    })
}

#[inline]
fn parse_env_token(name: &str) -> Option<String> {
    std::env::var(name).ok().and_then(|v| {
        let t = v.trim().to_ascii_lowercase();
        if t.is_empty() {
            None
        } else {
            Some(t)
        }
    })
}

#[inline]
fn force_scalar_runtime() -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    *FORCE_SCALAR.get_or_init(|| parse_env_bool("JANUSX_BITWISE_FORCE_SCALAR"))
}

#[inline]
fn force_simd_reduce_runtime() -> bool {
    static FORCE_SIMD_REDUCE: OnceLock<bool> = OnceLock::new();
    *FORCE_SIMD_REDUCE.get_or_init(|| parse_env_bool("JANUSX_BITWISE_FORCE_SIMD_REDUCE"))
}

#[inline]
fn force_simd_mutate_runtime() -> bool {
    static FORCE_SIMD_MUTATE: OnceLock<bool> = OnceLock::new();
    *FORCE_SIMD_MUTATE.get_or_init(|| parse_env_bool("JANUSX_BITWISE_FORCE_SIMD_MUTATE"))
}

#[inline]
fn bitwise_autotune_enabled() -> bool {
    static AUTO: OnceLock<bool> = OnceLock::new();
    *AUTO.get_or_init(|| parse_env_bool_default("JANUSX_BITWISE_AUTOTUNE", true))
}

#[inline]
fn bitwise_autotune_words() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("JANUSX_BITWISE_AUTOTUNE_WORDS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1024)
            .unwrap_or(1 << 16)
    })
}

#[inline]
fn bitwise_autotune_iters() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("JANUSX_BITWISE_AUTOTUNE_ITERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(6)
    })
}

#[inline]
fn bitwise_backend_debug() -> bool {
    static DBG: OnceLock<bool> = OnceLock::new();
    *DBG.get_or_init(|| parse_env_bool("JANUSX_BITWISE_BACKEND_DEBUG"))
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ReduceBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MutateBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[inline]
fn reduce_backend_name(b: ReduceBackend) -> &'static str {
    match b {
        ReduceBackend::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        ReduceBackend::Avx2 => "avx2",
        #[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
        ReduceBackend::Avx512 => "avx512",
        #[cfg(target_arch = "aarch64")]
        ReduceBackend::Neon => "neon",
    }
}

#[inline]
fn mutate_backend_name(b: MutateBackend) -> &'static str {
    match b {
        MutateBackend::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        MutateBackend::Avx2 => "avx2",
        #[cfg(target_arch = "aarch64")]
        MutateBackend::Neon => "neon",
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_runtime_available() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
#[inline]
fn avx512_vpopcntdq_runtime_available() -> bool {
    static AVX512_VPOPCNT: OnceLock<bool> = OnceLock::new();
    *AVX512_VPOPCNT.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
    })
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_runtime_available() -> bool {
    static NEON: OnceLock<bool> = OnceLock::new();
    *NEON.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

#[inline]
fn popcount_serial(words: &[u64]) -> u64 {
    let mut a0 = 0u64;
    let mut a1 = 0u64;
    let mut a2 = 0u64;
    let mut a3 = 0u64;
    let mut i = 0usize;
    let n = words.len();

    while i + 4 <= n {
        a0 += words[i].count_ones() as u64;
        a1 += words[i + 1].count_ones() as u64;
        a2 += words[i + 2].count_ones() as u64;
        a3 += words[i + 3].count_ones() as u64;
        i += 4;
    }
    while i < n {
        a0 += words[i].count_ones() as u64;
        i += 1;
    }
    a0 + a1 + a2 + a3
}

#[inline]
fn and_popcount_serial(lhs: &[u64], rhs: &[u64]) -> u64 {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut a0 = 0u64;
    let mut a1 = 0u64;
    let mut a2 = 0u64;
    let mut a3 = 0u64;
    let mut i = 0usize;
    let n = lhs.len();

    while i + 4 <= n {
        a0 += (lhs[i] & rhs[i]).count_ones() as u64;
        a1 += (lhs[i + 1] & rhs[i + 1]).count_ones() as u64;
        a2 += (lhs[i + 2] & rhs[i + 2]).count_ones() as u64;
        a3 += (lhs[i + 3] & rhs[i + 3]).count_ones() as u64;
        i += 4;
    }
    while i < n {
        a0 += (lhs[i] & rhs[i]).count_ones() as u64;
        i += 1;
    }
    a0 + a1 + a2 + a3
}

#[inline]
fn bitand_assign_serial(dst: &mut [u64], rhs: &[u64]) {
    debug_assert_eq!(dst.len(), rhs.len());
    for (d, &r) in dst.iter_mut().zip(rhs.iter()) {
        *d &= r;
    }
}

#[inline]
fn bitor_into_serial(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d |= s;
    }
}

#[inline]
fn bitnot_in_place_serial(words: &mut [u64]) {
    for w in words.iter_mut() {
        *w = !*w;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcount_u8x32_avx2(
    v: core::arch::x86_64::__m256i,
    lut4: core::arch::x86_64::__m256i,
    low_mask: core::arch::x86_64::__m256i,
    zero: core::arch::x86_64::__m256i,
) -> u64 {
    use core::arch::x86_64::*;
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    let cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut4, lo), _mm256_shuffle_epi8(lut4, hi));
    let sum64 = _mm256_sad_epu8(cnt, zero);
    (_mm256_extract_epi64(sum64, 0) as u64)
        + (_mm256_extract_epi64(sum64, 1) as u64)
        + (_mm256_extract_epi64(sum64, 2) as u64)
        + (_mm256_extract_epi64(sum64, 3) as u64)
}

#[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn popcount_simd_avx512(words: &[u64]) -> u64 {
    use core::arch::x86_64::*;
    let mut acc = _mm512_setzero_si512();
    let mut i = 0usize;
    let n = words.len();
    while i + 8 <= n {
        let v = _mm512_loadu_si512(words.as_ptr().add(i) as *const _);
        let c = _mm512_popcnt_epi64(v);
        acc = _mm512_add_epi64(acc, c);
        i += 8;
    }
    let mut lanes = [0u64; 8];
    _mm512_storeu_si512(lanes.as_mut_ptr() as *mut _, acc);
    lanes.iter().copied().sum::<u64>() + popcount_serial(&words[i..])
}

#[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn and_popcount_simd_avx512(lhs: &[u64], rhs: &[u64]) -> u64 {
    use core::arch::x86_64::*;
    let mut acc = _mm512_setzero_si512();
    let mut i = 0usize;
    let n = lhs.len();
    while i + 8 <= n {
        let lv = _mm512_loadu_si512(lhs.as_ptr().add(i) as *const _);
        let rv = _mm512_loadu_si512(rhs.as_ptr().add(i) as *const _);
        let c = _mm512_popcnt_epi64(_mm512_and_si512(lv, rv));
        acc = _mm512_add_epi64(acc, c);
        i += 8;
    }
    let mut lanes = [0u64; 8];
    _mm512_storeu_si512(lanes.as_mut_ptr() as *mut _, acc);
    lanes.iter().copied().sum::<u64>() + and_popcount_serial(&lhs[i..], &rhs[i..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcount_simd_avx2(words: &[u64]) -> u64 {
    use core::arch::x86_64::*;
    let lut4 = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f_i8);
    let zero = _mm256_setzero_si256();
    let mut acc = 0u64;
    let mut i = 0usize;
    let n = words.len();
    while i + 4 <= n {
        let v = _mm256_loadu_si256(words.as_ptr().add(i) as *const __m256i);
        acc += popcount_u8x32_avx2(v, lut4, low_mask, zero);
        i += 4;
    }
    acc + popcount_serial(&words[i..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_popcount_simd_avx2(lhs: &[u64], rhs: &[u64]) -> u64 {
    use core::arch::x86_64::*;
    let lut4 = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f_i8);
    let zero = _mm256_setzero_si256();
    let mut acc = 0u64;
    let mut i = 0usize;
    let n = lhs.len();
    while i + 4 <= n {
        let lv = _mm256_loadu_si256(lhs.as_ptr().add(i) as *const __m256i);
        let rv = _mm256_loadu_si256(rhs.as_ptr().add(i) as *const __m256i);
        acc += popcount_u8x32_avx2(_mm256_and_si256(lv, rv), lut4, low_mask, zero);
        i += 4;
    }
    acc + and_popcount_serial(&lhs[i..], &rhs[i..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bitand_assign_simd_avx2(dst: &mut [u64], rhs: &[u64]) {
    use core::arch::x86_64::*;
    let mut i = 0usize;
    let n = dst.len();
    while i + 4 <= n {
        let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
        let r = _mm256_loadu_si256(rhs.as_ptr().add(i) as *const __m256i);
        let out = _mm256_and_si256(d, r);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, out);
        i += 4;
    }
    bitand_assign_serial(&mut dst[i..], &rhs[i..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bitor_into_simd_avx2(dst: &mut [u64], src: &[u64]) {
    use core::arch::x86_64::*;
    let mut i = 0usize;
    let n = dst.len();
    while i + 4 <= n {
        let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
        let s = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
        let out = _mm256_or_si256(d, s);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, out);
        i += 4;
    }
    bitor_into_serial(&mut dst[i..], &src[i..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bitnot_in_place_simd_avx2(words: &mut [u64]) {
    use core::arch::x86_64::*;
    let all = _mm256_set1_epi64x(-1);
    let mut i = 0usize;
    let n = words.len();
    while i + 4 <= n {
        let v = _mm256_loadu_si256(words.as_ptr().add(i) as *const __m256i);
        let out = _mm256_xor_si256(v, all);
        _mm256_storeu_si256(words.as_mut_ptr().add(i) as *mut __m256i, out);
        i += 4;
    }
    bitnot_in_place_serial(&mut words[i..]);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn popcount_u64x2_neon(v: core::arch::aarch64::uint64x2_t) -> u64 {
    use core::arch::aarch64::*;
    let cnt8 = vcntq_u8(vreinterpretq_u8_u64(v));
    let sum16 = vpaddlq_u8(cnt8);
    let sum32 = vpaddlq_u16(sum16);
    let sum64 = vpaddlq_u32(sum32);
    vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1)
}

#[cfg(target_arch = "aarch64")]
unsafe fn popcount_simd_neon(words: &[u64]) -> u64 {
    use core::arch::aarch64::*;
    let mut acc = 0u64;
    let mut i = 0usize;
    let n = words.len();
    while i + 2 <= n {
        let v = vld1q_u64(words.as_ptr().add(i));
        acc += popcount_u64x2_neon(v);
        i += 2;
    }
    acc + popcount_serial(&words[i..])
}

#[cfg(target_arch = "aarch64")]
unsafe fn and_popcount_simd_neon(lhs: &[u64], rhs: &[u64]) -> u64 {
    use core::arch::aarch64::*;
    let mut acc = 0u64;
    let mut i = 0usize;
    let n = lhs.len();
    while i + 2 <= n {
        let lv = vld1q_u64(lhs.as_ptr().add(i));
        let rv = vld1q_u64(rhs.as_ptr().add(i));
        acc += popcount_u64x2_neon(vandq_u64(lv, rv));
        i += 2;
    }
    acc + and_popcount_serial(&lhs[i..], &rhs[i..])
}

#[cfg(target_arch = "aarch64")]
unsafe fn bitand_assign_simd_neon(dst: &mut [u64], rhs: &[u64]) {
    use core::arch::aarch64::*;
    let mut i = 0usize;
    let n = dst.len();
    while i + 2 <= n {
        let d = vld1q_u64(dst.as_ptr().add(i));
        let r = vld1q_u64(rhs.as_ptr().add(i));
        vst1q_u64(dst.as_mut_ptr().add(i), vandq_u64(d, r));
        i += 2;
    }
    bitand_assign_serial(&mut dst[i..], &rhs[i..]);
}

#[cfg(target_arch = "aarch64")]
unsafe fn bitor_into_simd_neon(dst: &mut [u64], src: &[u64]) {
    use core::arch::aarch64::*;
    let mut i = 0usize;
    let n = dst.len();
    while i + 2 <= n {
        let d = vld1q_u64(dst.as_ptr().add(i));
        let s = vld1q_u64(src.as_ptr().add(i));
        vst1q_u64(dst.as_mut_ptr().add(i), vorrq_u64(d, s));
        i += 2;
    }
    bitor_into_serial(&mut dst[i..], &src[i..]);
}

#[cfg(target_arch = "aarch64")]
unsafe fn bitnot_in_place_simd_neon(words: &mut [u64]) {
    use core::arch::aarch64::*;
    let all = vdupq_n_u64(u64::MAX);
    let mut i = 0usize;
    let n = words.len();
    while i + 2 <= n {
        let v = vld1q_u64(words.as_ptr().add(i));
        vst1q_u64(words.as_mut_ptr().add(i), veorq_u64(v, all));
        i += 2;
    }
    bitnot_in_place_serial(&mut words[i..]);
}

#[inline]
fn popcount_chunk_by_backend(words: &[u64], backend: ReduceBackend) -> u64 {
    match backend {
        ReduceBackend::Scalar => popcount_serial(words),
        #[cfg(target_arch = "x86_64")]
        ReduceBackend::Avx2 => {
            if avx2_runtime_available() {
                // SAFETY: gated by runtime AVX2 detection.
                unsafe { popcount_simd_avx2(words) }
            } else {
                popcount_serial(words)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
        ReduceBackend::Avx512 => {
            if avx512_vpopcntdq_runtime_available() {
                // SAFETY: gated by runtime AVX-512 VPOPCNTDQ detection.
                unsafe { popcount_simd_avx512(words) }
            } else if avx2_runtime_available() {
                // SAFETY: gated by runtime AVX2 detection.
                unsafe { popcount_simd_avx2(words) }
            } else {
                popcount_serial(words)
            }
        }
        #[cfg(target_arch = "aarch64")]
        ReduceBackend::Neon => {
            if neon_runtime_available() {
                // SAFETY: gated by runtime NEON detection.
                unsafe { popcount_simd_neon(words) }
            } else {
                popcount_serial(words)
            }
        }
    }
}

#[inline]
fn and_popcount_chunk_by_backend(lhs: &[u64], rhs: &[u64], backend: ReduceBackend) -> u64 {
    match backend {
        ReduceBackend::Scalar => and_popcount_serial(lhs, rhs),
        #[cfg(target_arch = "x86_64")]
        ReduceBackend::Avx2 => {
            if avx2_runtime_available() {
                // SAFETY: gated by runtime AVX2 detection.
                unsafe { and_popcount_simd_avx2(lhs, rhs) }
            } else {
                and_popcount_serial(lhs, rhs)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
        ReduceBackend::Avx512 => {
            if avx512_vpopcntdq_runtime_available() {
                // SAFETY: gated by runtime AVX-512 VPOPCNTDQ detection.
                unsafe { and_popcount_simd_avx512(lhs, rhs) }
            } else if avx2_runtime_available() {
                // SAFETY: gated by runtime AVX2 detection.
                unsafe { and_popcount_simd_avx2(lhs, rhs) }
            } else {
                and_popcount_serial(lhs, rhs)
            }
        }
        #[cfg(target_arch = "aarch64")]
        ReduceBackend::Neon => {
            if neon_runtime_available() {
                // SAFETY: gated by runtime NEON detection.
                unsafe { and_popcount_simd_neon(lhs, rhs) }
            } else {
                and_popcount_serial(lhs, rhs)
            }
        }
    }
}

#[inline]
fn bitand_assign_chunk_by_backend(dst: &mut [u64], rhs: &[u64], backend: MutateBackend) {
    match backend {
        MutateBackend::Scalar => bitand_assign_serial(dst, rhs),
        #[cfg(target_arch = "x86_64")]
        MutateBackend::Avx2 => {
            if avx2_runtime_available() {
                // SAFETY: gated by runtime AVX2 detection.
                unsafe { bitand_assign_simd_avx2(dst, rhs) };
            } else {
                bitand_assign_serial(dst, rhs);
            }
        }
        #[cfg(target_arch = "aarch64")]
        MutateBackend::Neon => {
            if neon_runtime_available() {
                // SAFETY: gated by runtime NEON detection.
                unsafe { bitand_assign_simd_neon(dst, rhs) };
            } else {
                bitand_assign_serial(dst, rhs);
            }
        }
    }
}

#[inline]
fn bitor_into_chunk_by_backend(dst: &mut [u64], src: &[u64], backend: MutateBackend) {
    match backend {
        MutateBackend::Scalar => bitor_into_serial(dst, src),
        #[cfg(target_arch = "x86_64")]
        MutateBackend::Avx2 => {
            if avx2_runtime_available() {
                // SAFETY: gated by runtime AVX2 detection.
                unsafe { bitor_into_simd_avx2(dst, src) };
            } else {
                bitor_into_serial(dst, src);
            }
        }
        #[cfg(target_arch = "aarch64")]
        MutateBackend::Neon => {
            if neon_runtime_available() {
                // SAFETY: gated by runtime NEON detection.
                unsafe { bitor_into_simd_neon(dst, src) };
            } else {
                bitor_into_serial(dst, src);
            }
        }
    }
}

#[inline]
fn bitnot_in_place_chunk_by_backend(words: &mut [u64], backend: MutateBackend) {
    match backend {
        MutateBackend::Scalar => bitnot_in_place_serial(words),
        #[cfg(target_arch = "x86_64")]
        MutateBackend::Avx2 => {
            if avx2_runtime_available() {
                // SAFETY: gated by runtime AVX2 detection.
                unsafe { bitnot_in_place_simd_avx2(words) };
            } else {
                bitnot_in_place_serial(words);
            }
        }
        #[cfg(target_arch = "aarch64")]
        MutateBackend::Neon => {
            if neon_runtime_available() {
                // SAFETY: gated by runtime NEON detection.
                unsafe { bitnot_in_place_simd_neon(words) };
            } else {
                bitnot_in_place_serial(words);
            }
        }
    }
}

#[inline]
fn popcount_with_backend(words: &[u64], backend: ReduceBackend) -> u64 {
    if words.is_empty() {
        return 0;
    }
    if should_parallelize(words.len(), PAR_REDUCE_MIN_WORDS) {
        words
            .par_chunks(PAR_CHUNK_WORDS)
            .map(|chunk| popcount_chunk_by_backend(chunk, backend))
            .sum::<u64>()
    } else {
        popcount_chunk_by_backend(words, backend)
    }
}

#[inline]
fn and_popcount_with_backend(lhs: &[u64], rhs: &[u64], backend: ReduceBackend) -> u64 {
    if lhs.is_empty() {
        return 0;
    }
    if should_parallelize(lhs.len(), PAR_REDUCE_MIN_WORDS) {
        lhs.par_chunks(PAR_CHUNK_WORDS)
            .zip(rhs.par_chunks(PAR_CHUNK_WORDS))
            .map(|(l, r)| and_popcount_chunk_by_backend(l, r, backend))
            .sum::<u64>()
    } else {
        and_popcount_chunk_by_backend(lhs, rhs, backend)
    }
}

#[inline]
fn available_reduce_simd_backends() -> Vec<ReduceBackend> {
    let mut out = Vec::new();
    #[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
    {
        if avx512_vpopcntdq_runtime_available() {
            out.push(ReduceBackend::Avx512);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_runtime_available() {
            out.push(ReduceBackend::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_runtime_available() {
            out.push(ReduceBackend::Neon);
        }
    }
    out
}

#[inline]
fn available_mutate_simd_backends() -> Vec<MutateBackend> {
    let mut out = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_runtime_available() {
            out.push(MutateBackend::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_runtime_available() {
            out.push(MutateBackend::Neon);
        }
    }
    out
}

#[inline]
fn make_autotune_words(n: usize, seed: u64) -> Vec<u64> {
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15u64;
    let mut out = vec![0u64; n];
    for v in out.iter_mut() {
        x ^= x << 7;
        x ^= x >> 9;
        x = x.wrapping_mul(0xD134_2543_DE82_EF95);
        *v = x;
    }
    out
}

#[inline]
fn autotune_reduce_backend(cands: &[ReduceBackend], allow_scalar: bool) -> ReduceBackend {
    if cands.is_empty() {
        return ReduceBackend::Scalar;
    }
    if !bitwise_autotune_enabled() {
        return cands[0];
    }

    let n_words = bitwise_autotune_words();
    let iters = bitwise_autotune_iters();
    let a = make_autotune_words(n_words, 0x1234_5678_9ABC_DEF0);
    let b = make_autotune_words(n_words, 0x0FED_CBA9_8765_4321);

    let mut options: Vec<ReduceBackend> = Vec::new();
    if allow_scalar {
        options.push(ReduceBackend::Scalar);
    }
    options.extend_from_slice(cands);

    let mut best = options[0];
    let mut best_t = f64::INFINITY;
    for &backend in options.iter() {
        let mut sink = 0u64;
        let _ = and_popcount_with_backend(&a, &b, backend);
        let t0 = Instant::now();
        for _ in 0..iters {
            sink ^= and_popcount_with_backend(&a, &b, backend);
        }
        let dt = t0.elapsed().as_secs_f64();
        black_box(sink);
        if dt < best_t {
            best_t = dt;
            best = backend;
        }
    }
    best
}

#[inline]
fn choose_reduce_backend() -> ReduceBackend {
    if force_scalar_runtime() {
        return ReduceBackend::Scalar;
    }
    let simd = available_reduce_simd_backends();
    if simd.is_empty() {
        return ReduceBackend::Scalar;
    }

    if let Some(req) = parse_env_token("JANUSX_BITWISE_REDUCE_BACKEND") {
        match req.as_str() {
            "scalar" => return ReduceBackend::Scalar,
            "simd" => return autotune_reduce_backend(&simd, false),
            #[cfg(target_arch = "x86_64")]
            "avx2" => {
                if simd.contains(&ReduceBackend::Avx2) {
                    return ReduceBackend::Avx2;
                }
            }
            #[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
            "avx512" => {
                if simd.contains(&ReduceBackend::Avx512) {
                    return ReduceBackend::Avx512;
                }
            }
            #[cfg(target_arch = "aarch64")]
            "neon" => {
                if simd.contains(&ReduceBackend::Neon) {
                    return ReduceBackend::Neon;
                }
            }
            _ => {}
        }
    }

    if force_simd_reduce_runtime() {
        return autotune_reduce_backend(&simd, false);
    }
    autotune_reduce_backend(&simd, true)
}

#[inline]
fn resolve_reduce_backend() -> ReduceBackend {
    static BACKEND: OnceLock<ReduceBackend> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        let b = choose_reduce_backend();
        if bitwise_backend_debug() {
            eprintln!("[bitwise] reduce_backend={}", reduce_backend_name(b));
        }
        b
    })
}

#[inline]
fn choose_mutate_backend() -> MutateBackend {
    if force_scalar_runtime() {
        return MutateBackend::Scalar;
    }
    let simd = available_mutate_simd_backends();
    if simd.is_empty() {
        return MutateBackend::Scalar;
    }

    if let Some(req) = parse_env_token("JANUSX_BITWISE_MUTATE_BACKEND") {
        match req.as_str() {
            "scalar" => return MutateBackend::Scalar,
            "simd" => return simd[0],
            #[cfg(target_arch = "x86_64")]
            "avx2" => {
                if simd.contains(&MutateBackend::Avx2) {
                    return MutateBackend::Avx2;
                }
            }
            #[cfg(target_arch = "aarch64")]
            "neon" => {
                if simd.contains(&MutateBackend::Neon) {
                    return MutateBackend::Neon;
                }
            }
            _ => {}
        }
    }

    if force_simd_mutate_runtime() {
        return simd[0];
    }

    match resolve_reduce_backend() {
        #[cfg(target_arch = "x86_64")]
        ReduceBackend::Avx2 | ReduceBackend::Avx512 => {
            if simd.contains(&MutateBackend::Avx2) {
                return MutateBackend::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        ReduceBackend::Neon => {
            if simd.contains(&MutateBackend::Neon) {
                return MutateBackend::Neon;
            }
        }
        ReduceBackend::Scalar => {}
    }
    MutateBackend::Scalar
}

#[inline]
fn resolve_mutate_backend() -> MutateBackend {
    static BACKEND: OnceLock<MutateBackend> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        let b = choose_mutate_backend();
        if bitwise_backend_debug() {
            eprintln!("[bitwise] mutate_backend={}", mutate_backend_name(b));
        }
        b
    })
}

/// Count the number of 1-bits in a packed `u64` bit-vector.
pub fn popcount(words: &[u64]) -> u64 {
    popcount_with_backend(words, resolve_reduce_backend())
}

/// Count the number of 1-bits in `lhs & rhs` for packed `u64` bit-vectors.
pub fn and_popcount(lhs: &[u64], rhs: &[u64]) -> u64 {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "and_popcount: length mismatch (lhs={}, rhs={})",
        lhs.len(),
        rhs.len()
    );
    if lhs.is_empty() {
        return 0;
    }
    and_popcount_with_backend(lhs, rhs, resolve_reduce_backend())
}

/// In-place `dst &= rhs` for packed `u64` bit-vectors.
pub fn bitand_assign(dst: &mut [u64], rhs: &[u64]) {
    assert_eq!(
        dst.len(),
        rhs.len(),
        "bitand_assign: length mismatch (dst={}, rhs={})",
        dst.len(),
        rhs.len()
    );
    if dst.is_empty() {
        return;
    }
    let backend = resolve_mutate_backend();
    if should_parallelize(dst.len(), PAR_MUTATE_MIN_WORDS) {
        dst.par_chunks_mut(PAR_CHUNK_WORDS)
            .zip(rhs.par_chunks(PAR_CHUNK_WORDS))
            .for_each(|(d, r)| bitand_assign_chunk_by_backend(d, r, backend));
    } else {
        bitand_assign_chunk_by_backend(dst, rhs, backend);
    }
}

/// In-place `dst |= src` for packed `u64` bit-vectors.
pub fn bitor_into(dst: &mut [u64], src: &[u64]) {
    assert_eq!(
        dst.len(),
        src.len(),
        "bitor_into: length mismatch (dst={}, src={})",
        dst.len(),
        src.len()
    );
    if dst.is_empty() {
        return;
    }
    let backend = resolve_mutate_backend();
    if should_parallelize(dst.len(), PAR_MUTATE_MIN_WORDS) {
        dst.par_chunks_mut(PAR_CHUNK_WORDS)
            .zip(src.par_chunks(PAR_CHUNK_WORDS))
            .for_each(|(d, s)| bitor_into_chunk_by_backend(d, s, backend));
    } else {
        bitor_into_chunk_by_backend(dst, src, backend);
    }
}

/// In-place bitwise NOT within `n_valid_bits` and clear all out-of-range bits.
///
/// Semantics:
/// - Invert bits in `[0, n_valid_bits)`.
/// - Clear bits in `[n_valid_bits, words.len() * 64)`.
pub fn bitnot_masked(words: &mut [u64], n_valid_bits: usize) {
    if words.is_empty() {
        return;
    }
    if n_valid_bits == 0 {
        words.fill(0);
        return;
    }

    let active_words = n_valid_bits.div_ceil(64);
    assert!(
        active_words <= words.len(),
        "bitnot_masked: n_valid_bits={} requires {} words, got {}",
        n_valid_bits,
        active_words,
        words.len()
    );

    let (active, tail) = words.split_at_mut(active_words);
    let backend = resolve_mutate_backend();
    if should_parallelize(active.len(), PAR_MUTATE_MIN_WORDS) {
        active
            .par_chunks_mut(PAR_CHUNK_WORDS)
            .for_each(|chunk| bitnot_in_place_chunk_by_backend(chunk, backend));
    } else {
        bitnot_in_place_chunk_by_backend(active, backend);
    }

    let rem = n_valid_bits & 63;
    if rem != 0 {
        let tail_mask = (1u64 << rem) - 1u64;
        if let Some(last) = active.last_mut() {
            *last &= tail_mask;
        }
    }

    if !tail.is_empty() {
        if should_parallelize(tail.len(), PAR_MUTATE_MIN_WORDS) {
            tail.par_chunks_mut(PAR_CHUNK_WORDS)
                .for_each(|chunk| chunk.fill(0));
        } else {
            tail.fill(0);
        }
    }
}

#[inline]
fn readonly_u64_to_cow<'a>(arr: &'a PyReadonlyArray1<'a, u64>) -> Cow<'a, [u64]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[pyfunction(name = "popcount")]
pub fn popcount_py(words: PyReadonlyArray1<'_, u64>) -> PyResult<u64> {
    let words_cow = readonly_u64_to_cow(&words);
    Ok(popcount(words_cow.as_ref()))
}

#[pyfunction(name = "and_popcount")]
pub fn and_popcount_py(
    lhs: PyReadonlyArray1<'_, u64>,
    rhs: PyReadonlyArray1<'_, u64>,
) -> PyResult<u64> {
    let lhs_cow = readonly_u64_to_cow(&lhs);
    let rhs_cow = readonly_u64_to_cow(&rhs);
    if lhs_cow.len() != rhs_cow.len() {
        return Err(PyRuntimeError::new_err(format!(
            "and_popcount: length mismatch (lhs={}, rhs={})",
            lhs_cow.len(),
            rhs_cow.len()
        )));
    }
    Ok(and_popcount(lhs_cow.as_ref(), rhs_cow.as_ref()))
}

#[pyfunction(name = "bitand_assign")]
pub fn bitand_assign_py(
    mut dst: PyReadwriteArray1<'_, u64>,
    rhs: PyReadonlyArray1<'_, u64>,
) -> PyResult<()> {
    let dst_slice = dst
        .as_slice_mut()
        .map_err(|_| PyRuntimeError::new_err("bitand_assign: dst must be contiguous"))?;
    let rhs_cow = readonly_u64_to_cow(&rhs);
    if dst_slice.len() != rhs_cow.len() {
        return Err(PyRuntimeError::new_err(format!(
            "bitand_assign: length mismatch (dst={}, rhs={})",
            dst_slice.len(),
            rhs_cow.len()
        )));
    }
    bitand_assign(dst_slice, rhs_cow.as_ref());
    Ok(())
}

#[pyfunction(name = "bitor_into")]
pub fn bitor_into_py(
    mut dst: PyReadwriteArray1<'_, u64>,
    src: PyReadonlyArray1<'_, u64>,
) -> PyResult<()> {
    let dst_slice = dst
        .as_slice_mut()
        .map_err(|_| PyRuntimeError::new_err("bitor_into: dst must be contiguous"))?;
    let src_cow = readonly_u64_to_cow(&src);
    if dst_slice.len() != src_cow.len() {
        return Err(PyRuntimeError::new_err(format!(
            "bitor_into: length mismatch (dst={}, src={})",
            dst_slice.len(),
            src_cow.len()
        )));
    }
    bitor_into(dst_slice, src_cow.as_ref());
    Ok(())
}

#[pyfunction(name = "bitnot_masked")]
pub fn bitnot_masked_py(
    mut words: PyReadwriteArray1<'_, u64>,
    n_valid_bits: usize,
) -> PyResult<()> {
    let words_slice = words
        .as_slice_mut()
        .map_err(|_| PyRuntimeError::new_err("bitnot_masked: words must be contiguous"))?;
    if n_valid_bits > words_slice.len().saturating_mul(64) {
        return Err(PyRuntimeError::new_err(format!(
            "bitnot_masked: n_valid_bits={} exceeds capacity {} bits",
            n_valid_bits,
            words_slice.len().saturating_mul(64)
        )));
    }
    bitnot_masked(words_slice, n_valid_bits);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    fn make_words(n: usize, seed: u64) -> Vec<u64> {
        let mut x = seed ^ 0x9E37_79B9_7F4A_7C15u64;
        let mut out = vec![0u64; n];
        for v in out.iter_mut() {
            x ^= x << 7;
            x ^= x >> 9;
            x = x.wrapping_mul(0xD134_2543_DE82_EF95);
            *v = x;
        }
        out
    }

    fn as_gbps(bytes: f64, sec: f64) -> f64 {
        if sec <= 0.0 {
            0.0
        } else {
            bytes / sec / 1e9
        }
    }

    #[test]
    fn test_popcount_basic() {
        let v = [0u64, 1u64, u64::MAX, 0xF0F0_F0F0_F0F0_F0F0];
        assert_eq!(popcount(&v), 0 + 1 + 64 + 32);
    }

    #[test]
    fn test_and_popcount_basic() {
        let a = [0b1011u64, 0b1100u64, u64::MAX];
        let b = [0b0110u64, 0b0101u64, 0];
        assert_eq!(and_popcount(&a, &b), 1 + 1 + 0);
    }

    #[test]
    fn test_bitand_assign_and_bitor_into() {
        let mut x = [0b1111u64, 0b0011u64];
        let y = [0b1010u64, 0b0110u64];
        bitand_assign(&mut x, &y);
        assert_eq!(x, [0b1010, 0b0010]);

        let z = [0b0100u64, 0b1000u64];
        bitor_into(&mut x, &z);
        assert_eq!(x, [0b1110, 0b1010]);
    }

    #[test]
    fn test_bitnot_masked_partial_tail() {
        let mut bits = [0u64, 0xFFFF_0000_0000_0000u64, u64::MAX];
        // Keep only first 70 bits => 2 active words, last 58 bits in word#1 must be cleared.
        bitnot_masked(&mut bits, 70);

        assert_eq!(bits[0], u64::MAX);
        let expected_mask = (1u64 << 6) - 1u64;
        assert_eq!(bits[1], (!0xFFFF_0000_0000_0000u64) & expected_mask);
        assert_eq!(bits[2], 0u64);
    }

    #[test]
    fn test_bitnot_masked_zero_bits() {
        let mut bits = [u64::MAX, 123, 456];
        bitnot_masked(&mut bits, 0);
        assert_eq!(bits, [0, 0, 0]);
    }

    #[test]
    #[ignore]
    fn bench_bitwise_throughput() {
        let n_words = std::env::var("JANUSX_BITWISE_BENCH_WORDS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2_000_000);
        let iters = std::env::var("JANUSX_BITWISE_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(24);

        let a = make_words(n_words, 0x1234_5678_9ABC_DEF0);
        let b = make_words(n_words, 0x0FED_CBA9_8765_4321);

        let mut sink = 0u64;
        let force_scalar = force_scalar_runtime();
        let force_simd_reduce = force_simd_reduce_runtime();
        let force_simd_mutate = force_simd_mutate_runtime();
        println!(
            "bitwise_bench mode={} simd_reduce={} simd_mutate={} threads={} n_words={} iters={}",
            if force_scalar { "scalar" } else { "simd_auto" },
            force_simd_reduce,
            force_simd_mutate,
            rayon::current_num_threads(),
            n_words,
            iters
        );

        let t0 = Instant::now();
        for _ in 0..iters {
            sink ^= popcount(&a);
        }
        let d_pop = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            sink ^= and_popcount(&a, &b);
        }
        let d_and_pop = t1.elapsed();

        let mut dsta = a.clone();
        let t2 = Instant::now();
        for i in 0..iters {
            bitand_assign(&mut dsta, &b);
            sink ^= dsta[i % dsta.len()];
        }
        let d_and_assign = t2.elapsed();

        let mut dsto = a.clone();
        let t3 = Instant::now();
        for i in 0..iters {
            bitor_into(&mut dsto, &b);
            sink ^= dsto[i % dsto.len()];
        }
        let d_or_into = t3.elapsed();

        let mut dstn = a.clone();
        let valid_bits = n_words * 64 - 7;
        let t4 = Instant::now();
        for i in 0..iters {
            bitnot_masked(&mut dstn, valid_bits);
            sink ^= dstn[i % dstn.len()];
        }
        let d_not = t4.elapsed();

        let bytes_pop = (n_words * 8 * iters) as f64;
        let bytes_and_pop = (n_words * 16 * iters) as f64;
        let bytes_and_assign = (n_words * 24 * iters) as f64;
        let bytes_or_into = (n_words * 24 * iters) as f64;
        let bytes_not = (n_words * 16 * iters) as f64;

        println!(
            "popcount      {:>8.3} ms  {:>8.2} GB/s",
            d_pop.as_secs_f64() * 1e3,
            as_gbps(bytes_pop, d_pop.as_secs_f64())
        );
        println!(
            "and_popcount  {:>8.3} ms  {:>8.2} GB/s",
            d_and_pop.as_secs_f64() * 1e3,
            as_gbps(bytes_and_pop, d_and_pop.as_secs_f64())
        );
        println!(
            "bitand_assign {:>8.3} ms  {:>8.2} GB/s",
            d_and_assign.as_secs_f64() * 1e3,
            as_gbps(bytes_and_assign, d_and_assign.as_secs_f64())
        );
        println!(
            "bitor_into    {:>8.3} ms  {:>8.2} GB/s",
            d_or_into.as_secs_f64() * 1e3,
            as_gbps(bytes_or_into, d_or_into.as_secs_f64())
        );
        println!(
            "bitnot_masked {:>8.3} ms  {:>8.2} GB/s",
            d_not.as_secs_f64() * 1e3,
            as_gbps(bytes_not, d_not.as_secs_f64())
        );

        black_box(sink);
    }
}
