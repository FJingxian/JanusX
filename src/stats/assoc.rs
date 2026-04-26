use matrixmultiply::sgemm;
use nalgebra::{DMatrix, DVector};
use numpy::ndarray::{Array1, Array2};
use numpy::PyArray1;
use numpy::PyReadonlyArray3;
use numpy::PyUntypedArrayMethods;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::brent::brent_minimize;
use crate::linalg::{
    chi2_sf_df1, cholesky_inplace, cholesky_logdet, cholesky_solve_into, normal_sf,
};

// =============================================================================
// Common utilities
// =============================================================================

const _INTERRUPTED_MSG: &str = "Interrupted by user (Ctrl+C).";

#[inline]
fn _check_ctrlc() -> Result<(), String> {
    Python::attach(|py| py.check_signals()).map_err(|_| _INTERRUPTED_MSG.to_string())
}

#[inline]
fn _map_err_string_to_py(err: String) -> PyErr {
    if err.contains("Interrupted by user (Ctrl+C)") {
        PyKeyboardInterrupt::new_err(err)
    } else {
        PyRuntimeError::new_err(err)
    }
}

#[inline]
fn _env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(false)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

thread_local! {
    static LOCAL_RAYON_POOLS: RefCell<HashMap<usize, Arc<rayon::ThreadPool>>> =
        RefCell::new(HashMap::new());
}

#[inline]
fn get_cached_pool(threads: usize) -> PyResult<Option<Arc<rayon::ThreadPool>>> {
    if threads == 0 {
        return Ok(None);
    }
    LOCAL_RAYON_POOLS.with(|cell| {
        let mut pools = cell.borrow_mut();
        if let Some(tp) = pools.get(&threads) {
            return Ok(Some(Arc::clone(tp)));
        }
        let tp = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        );
        pools.insert(threads, Arc::clone(&tp));
        Ok(Some(tp))
    })
}

/// Solve `A x = b` from the in-place Cholesky factor `L` stored in `a`.
fn cholesky_solve(a: &[f64], dim: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; dim];
    for i in 0..dim {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * dim + k] * y[k];
        }
        y[i] = sum / a[i * dim + i];
    }

    let mut x = vec![0.0_f64; dim];
    for ii in 0..dim {
        let i = dim - 1 - ii;
        let mut sum = y[i];
        for k in (i + 1)..dim {
            sum -= a[k * dim + i] * x[k];
        }
        x[i] = sum / a[i * dim + i];
    }
    x
}

// =============================================================================
// Student-t p-value (for GLM)
// =============================================================================

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let maxit = 200;
    let eps = 3.0e-14;
    let fpmin = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=maxit {
        let m2 = 2.0 * (m as f64);

        let mut aa = (m as f64) * (b - (m as f64)) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + (m as f64)) * (qab + (m as f64)) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

fn betai(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    let ln_beta = libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);

    if x < (a + 1.0) / (a + b + 2.0) {
        let front = ((a * x.ln()) + (b * (1.0 - x).ln()) - ln_beta).exp() / a;
        front * betacf(a, b, x)
    } else {
        let front = ((b * (1.0 - x).ln()) + (a * x.ln()) - ln_beta).exp() / b;
        1.0 - front * betacf(b, a, 1.0 - x)
    }
}

#[inline]
fn student_t_p_two_sided(t: f64, df: i32) -> f64 {
    if df <= 0 {
        return f64::NAN;
    }
    if !t.is_finite() {
        return if t.is_nan() {
            f64::NAN
        } else {
            f64::MIN_POSITIVE
        };
    }

    let v = df as f64;
    let x = v / (v + t * t);
    let a = v / 2.0;
    let b = 0.5;

    let mut p = betai(a, b, x);
    if !p.is_finite() {
        p = 1.0;
    }
    p.clamp(f64::MIN_POSITIVE, 1.0)
}

// =============================================================================
// GLM float32 fast path (glmf32) with thread-local scratch
// =============================================================================

#[inline]
fn xs_t_ixx_into(xs: &[f64], ixx: &[f64], q0: usize, out_b21: &mut [f64]) {
    debug_assert_eq!(out_b21.len(), q0);
    for j in 0..q0 {
        let mut acc = 0.0;
        for k in 0..q0 {
            acc += xs[k] * ixx[k * q0 + j];
        }
        out_b21[j] = acc;
    }
}

#[inline]
fn build_ixxs_into(ixx: &[f64], b21: &[f64], invb22: f64, q0: usize, out_ixxs: &mut [f64]) {
    let dim = q0 + 1;
    debug_assert_eq!(out_ixxs.len(), dim * dim);

    for r in 0..q0 {
        for c in 0..q0 {
            out_ixxs[r * dim + c] = ixx[r * q0 + c] + invb22 * (b21[r] * b21[c]);
        }
    }

    out_ixxs[q0 * dim + q0] = invb22;

    for j in 0..q0 {
        let v = -invb22 * b21[j];
        out_ixxs[q0 * dim + j] = v;
        out_ixxs[j * dim + q0] = v;
    }
}

#[inline]
fn matvec_into(a: &[f64], dim: usize, rhs: &[f64], out: &mut [f64]) {
    debug_assert_eq!(rhs.len(), dim);
    debug_assert_eq!(out.len(), dim);
    for r in 0..dim {
        let row = &a[r * dim..(r + 1) * dim];
        out[r] = row.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum();
    }
}

struct GlmScratch {
    xs: Vec<f64>,   // q0
    b21: Vec<f64>,  // q0
    rhs: Vec<f64>,  // q0+1
    beta: Vec<f64>, // q0+1
    ixxs: Vec<f64>, // (q0+1)^2
}

impl GlmScratch {
    fn new(q0: usize) -> Self {
        let dim = q0 + 1;
        Self {
            xs: vec![0.0; q0],
            b21: vec![0.0; q0],
            rhs: vec![0.0; dim],
            beta: vec![0.0; dim],
            ixxs: vec![0.0; dim * dim],
        }
    }
    #[inline]
    fn reset_xs(&mut self) {
        self.xs.fill(0.0);
    }
}

#[inline]
fn decode_plink_bed_hardcall(code: u8) -> Option<f64> {
    match code {
        0b00 => Some(0.0),
        0b10 => Some(1.0),
        0b11 => Some(2.0),
        _ => None, // 0b01 => missing
    }
}

struct PackedByteLut {
    nonmiss: [u8; 256],
    alt_sum: [u8; 256],
    sq_sum: [u8; 256],
    code4: [[u8; 4]; 256],
}

fn packed_byte_lut() -> &'static PackedByteLut {
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

struct PackedPairLut {
    dot_obs0: [u8; 65536],
    sum_j_when_i_miss: [u8; 65536],
    sum_i_when_j_miss: [u8; 65536],
    miss_both: [u8; 65536],
}

#[inline]
fn _code_to_dosage_or_zero(code: u8) -> u8 {
    match code {
        0b00 => 0,
        0b10 => 1,
        0b11 => 2,
        _ => 0, // missing
    }
}

fn packed_pair_lut() -> &'static PackedPairLut {
    static LUT: OnceLock<PackedPairLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let code4 = &packed_byte_lut().code4;
        let mut dot_obs0 = [0u8; 65536];
        let mut sum_j_when_i_miss = [0u8; 65536];
        let mut sum_i_when_j_miss = [0u8; 65536];
        let mut miss_both = [0u8; 65536];

        for bi in 0u16..=255 {
            for bj in 0u16..=255 {
                let idx = ((bi as usize) << 8) | (bj as usize);
                let ci = &code4[bi as usize];
                let cj = &code4[bj as usize];

                let mut dot_v: u8 = 0;
                let mut sj_imiss: u8 = 0;
                let mut si_jmiss: u8 = 0;
                let mut mb: u8 = 0;
                for lane in 0..4usize {
                    let a = ci[lane];
                    let b = cj[lane];
                    let ga = _code_to_dosage_or_zero(a);
                    let gb = _code_to_dosage_or_zero(b);
                    dot_v = dot_v.saturating_add(ga.saturating_mul(gb));
                    if a == 0b01 {
                        sj_imiss = sj_imiss.saturating_add(gb);
                        if b == 0b01 {
                            mb = mb.saturating_add(1);
                        }
                    }
                    if b == 0b01 {
                        si_jmiss = si_jmiss.saturating_add(ga);
                    }
                }
                dot_obs0[idx] = dot_v;
                sum_j_when_i_miss[idx] = sj_imiss;
                sum_i_when_j_miss[idx] = si_jmiss;
                miss_both[idx] = mb;
            }
        }

        PackedPairLut {
            dot_obs0,
            sum_j_when_i_miss,
            sum_i_when_j_miss,
            miss_both,
        }
    })
}

#[derive(Clone, Copy, Default)]
struct PackedRowStats {
    mean: f64,
    std: f64,
    maf: f64,
    has_missing: bool,
}

#[inline]
fn dot_imputed_pair_from_packed(
    row_i: &[u8],
    row_j: &[u8],
    n_samples: usize,
    mean_i: f64,
    mean_j: f64,
    pair_lut: &PackedPairLut,
    code4_lut: &[[u8; 4]; 256],
) -> f64 {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut dot = 0.0_f64;

    for b in 0..full_bytes {
        let idx = ((row_i[b] as usize) << 8) | (row_j[b] as usize);
        dot += pair_lut.dot_obs0[idx] as f64;
        dot += mean_i * pair_lut.sum_j_when_i_miss[idx] as f64;
        dot += mean_j * pair_lut.sum_i_when_j_miss[idx] as f64;
        dot += mean_i * mean_j * pair_lut.miss_both[idx] as f64;
    }

    if rem > 0 {
        let ci = &code4_lut[row_i[full_bytes] as usize];
        let cj = &code4_lut[row_j[full_bytes] as usize];
        for lane in 0..rem {
            let vi = match ci[lane] {
                0b00 => 0.0_f64,
                0b10 => 1.0_f64,
                0b11 => 2.0_f64,
                _ => mean_i,
            };
            let vj = match cj[lane] {
                0b00 => 0.0_f64,
                0b10 => 1.0_f64,
                0b11 => 2.0_f64,
                _ => mean_j,
            };
            dot += vi * vj;
        }
    }
    dot
}

#[inline]
fn dot_nomiss_pair_from_packed(
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
            let vi = _code_to_dosage_or_zero(ci[lane]) as f64;
            let vj = _code_to_dosage_or_zero(cj[lane]) as f64;
            dot += vi * vj;
        }
    }
    dot
}

static PRUNE_DOT_TOTAL_CALLS: AtomicU64 = AtomicU64::new(0);
static PRUNE_DOT_AVX2_CALLS: AtomicU64 = AtomicU64::new(0);
static PRUNE_DOT_NEON_CALLS: AtomicU64 = AtomicU64::new(0);
static PRUNE_DOT_SCALAR_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(target_arch = "x86_64")]
#[inline]
fn prune_force_scalar_runtime() -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    *FORCE_SCALAR.get_or_init(|| _env_truthy("JANUSX_PRUNE_FORCE_SCALAR"))
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

#[pyfunction]
#[pyo3(signature = (reset=false))]
pub fn packed_prune_kernel_stats(
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

fn build_bitplanes_u64(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    n_samples: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> (Vec<u64>, Vec<u64>, usize, Vec<u64>) {
    let words = (n_samples + 63) / 64;
    if m == 0 || words == 0 {
        return (Vec::new(), Vec::new(), 0, Vec::new());
    }

    let mut lut_h = [0u8; 256];
    let mut lut_l = [0u8; 256];
    for b in 0u16..=255 {
        let byte = b as u8;
        let mut hb = 0u8;
        let mut lb = 0u8;
        for lane in 0..4usize {
            let code = (byte >> (lane * 2)) & 0b11;
            if ((code >> 1) & 1) != 0 {
                hb |= 1u8 << lane;
            }
            if (code & 1) != 0 {
                lb |= 1u8 << lane;
            }
        }
        lut_h[byte as usize] = hb;
        lut_l[byte as usize] = lb;
    }

    let mut word_masks = vec![u64::MAX; words];
    let rem = n_samples % 64;
    if rem != 0 {
        word_masks[words - 1] = (1u64 << rem) - 1u64;
    }
    let last_mask = word_masks[words - 1];

    let mut h_bits = vec![0u64; m * words];
    let mut l_bits = vec![0u64; m * words];
    let mut run = || {
        h_bits
            .par_chunks_mut(words)
            .zip(l_bits.par_chunks_mut(words))
            .enumerate()
            .for_each(|(row_idx, (hrow, lrow))| {
                let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                for (bi, &b) in row.iter().enumerate() {
                    let w = bi >> 4;
                    let sh = ((bi & 15) << 2) as u32;
                    hrow[w] |= (lut_h[b as usize] as u64) << sh;
                    lrow[w] |= (lut_l[b as usize] as u64) << sh;
                }
                hrow[words - 1] &= last_mask;
                lrow[words - 1] &= last_mask;
            });
    };
    if let Some(tp) = pool {
        tp.install(&mut run);
    } else {
        run();
    }
    (h_bits, l_bits, words, word_masks)
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
    // AVX2 chunk: 4 * u64 (256 bit) per iteration.
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
fn dot_nomiss_pair_bitplanes(
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

        // Explicit SIMD (NEON): 2x u64 lanes per iteration.
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

    // Unrolled popcount accumulation on full words (no tail mask).
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

    // Tail word with sample-count mask.
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

fn prune_one_chrom_packed(
    idx_list: &[usize],
    pos_vec: &[i64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    stats: &[PackedRowStats],
    pair_lut: &PackedPairLut,
    code4_lut: &[[u8; 4]; 256],
    bitplane_h: &[u64],
    bitplane_l: &[u64],
    bitplane_words: usize,
    bitplane_masks: &[u64],
    enable_intra_chrom_parallel: bool,
    intra_parallel_min_neighbors: usize,
) -> Vec<bool> {
    let l = idx_list.len();
    let mut dropped = vec![false; l];
    if l <= 1 {
        return dropped;
    }

    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let n_samples_f = n_samples as f64;
    let eps = 1e-12_f64;
    let step = step_variants.max(1);
    let use_bp = window_bp.is_some();
    let mut pos_sorted = true;
    for k in 1..l {
        if pos_vec[idx_list[k]] < pos_vec[idx_list[k - 1]] {
            pos_sorted = false;
            break;
        }
    }
    let mut bp_end_ptr = 1usize;

    let mut block_start = 0usize;
    while block_start < l {
        let block_end = (block_start + step).min(l);
        for li in block_start..block_end {
            if dropped[li] {
                continue;
            }
            let end = if use_bp {
                let bp = window_bp.unwrap_or(1);
                if pos_sorted {
                    if bp_end_ptr < li + 1 {
                        bp_end_ptr = li + 1;
                    }
                    let target = pos_vec[idx_list[li]].saturating_add(bp);
                    while bp_end_ptr < l && pos_vec[idx_list[bp_end_ptr]] <= target {
                        bp_end_ptr += 1;
                    }
                    bp_end_ptr
                } else {
                    let mut e = li + 1;
                    let p0 = pos_vec[idx_list[li]];
                    while e < l {
                        let d = pos_vec[idx_list[e]].saturating_sub(p0);
                        if d <= bp {
                            e += 1;
                        } else if pos_vec[idx_list[e]] > p0 {
                            break;
                        } else {
                            e += 1;
                        }
                    }
                    e
                }
            } else {
                (li + window_variants.unwrap_or(1)).min(l)
            };
            if end <= li + 1 {
                continue;
            }

            let gi = idx_list[li];
            let row_i = &packed_flat[gi * bytes_per_snp..(gi + 1) * bytes_per_snp];
            let st_i = stats[gi];
            let mut drop_i = false;
            let mut drop_low: Vec<usize> = Vec::new();
            let classify_lj = |lj: usize| -> Option<u8> {
                if dropped[lj] {
                    return None;
                }
                let gj = idx_list[lj];
                let st_j = stats[gj];

                let dot_imp = if !st_i.has_missing && !st_j.has_missing && bitplane_words > 0 {
                    dot_nomiss_pair_bitplanes(
                        gi,
                        gj,
                        bitplane_h,
                        bitplane_l,
                        bitplane_words,
                        bitplane_masks,
                    )
                } else {
                    let row_j = &packed_flat[gj * bytes_per_snp..(gj + 1) * bytes_per_snp];
                    if !st_i.has_missing && !st_j.has_missing {
                        dot_nomiss_pair_from_packed(row_i, row_j, n_samples, pair_lut, code4_lut)
                    } else {
                        dot_imputed_pair_from_packed(
                            row_i, row_j, n_samples, st_i.mean, st_j.mean, pair_lut, code4_lut,
                        )
                    }
                };
                let cov = dot_imp - n_samples_f * st_i.mean * st_j.mean;
                let denom_corr = denom * st_i.std * st_j.std;
                let corr = if denom_corr > 0.0_f64 {
                    cov / denom_corr
                } else {
                    0.0_f64
                };
                let r2 = corr * corr;
                if !(r2.is_finite() && r2 >= r2_threshold) {
                    return None;
                }
                if st_j.maf > st_i.maf + eps {
                    Some(1)
                } else if st_j.maf + eps < st_i.maf {
                    Some(2)
                } else {
                    None
                }
            };

            let n_neighbors = end.saturating_sub(li + 1);
            if enable_intra_chrom_parallel && n_neighbors >= intra_parallel_min_neighbors {
                let decisions: Vec<(usize, u8)> = ((li + 1)..end)
                    .into_par_iter()
                    .filter_map(|lj| classify_lj(lj).map(|tag| (lj, tag)))
                    .collect();
                if decisions.iter().any(|(_, tag)| *tag == 1u8) {
                    drop_i = true;
                } else {
                    drop_low.extend(decisions.into_iter().filter_map(|(lj, tag)| {
                        if tag == 2u8 {
                            Some(lj)
                        } else {
                            None
                        }
                    }));
                }
            } else {
                for lj in (li + 1)..end {
                    if let Some(tag) = classify_lj(lj) {
                        if tag == 1u8 {
                            drop_i = true;
                            break;
                        }
                        drop_low.push(lj);
                    }
                }
            }

            if drop_i {
                dropped[li] = true;
            } else if !drop_low.is_empty() {
                for j in drop_low {
                    dropped[j] = true;
                }
            }
        }
        block_start = block_start.saturating_add(step);
    }
    dropped
}

fn bed_packed_ld_prune_keep(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    n_samples: usize,
    chrom_vec: &[i32],
    pos_vec: &[i64],
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    threads: usize,
) -> Result<Vec<bool>, String> {
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    if m == 0 {
        return Ok(Vec::new());
    }
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        ));
    }
    if packed_flat.len() != m.saturating_mul(bytes_per_snp) {
        return Err(format!(
            "packed payload length mismatch: got {}, expected {}",
            packed_flat.len(),
            m.saturating_mul(bytes_per_snp)
        ));
    }
    if chrom_vec.len() != m {
        return Err(format!(
            "chrom_codes length mismatch: got {}, expected {m}",
            chrom_vec.len()
        ));
    }
    if pos_vec.len() != m {
        return Err(format!(
            "positions length mismatch: got {}, expected {m}",
            pos_vec.len()
        ));
    }
    if !(r2_threshold.is_finite() && r2_threshold > 0.0 && r2_threshold <= 1.0) {
        return Err("r2_threshold must be finite and in (0, 1]".to_string());
    }
    if step_variants == 0 {
        return Err("step_variants must be > 0".to_string());
    }
    if window_bp.is_none() && window_variants.is_none() {
        return Err("provide one of window_bp or window_variants".to_string());
    }
    if let Some(bp) = window_bp {
        if bp <= 0 {
            return Err("window_bp must be > 0".to_string());
        }
    }
    if let Some(wv) = window_variants {
        if wv == 0 {
            return Err("window_variants must be > 0".to_string());
        }
    }

    let mut stats = vec![PackedRowStats::default(); m];
    let byte_lut = packed_byte_lut();
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    {
        let mut run = || {
            stats.par_iter_mut().enumerate().for_each(|(row_idx, st)| {
                let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
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
                    *st = PackedRowStats {
                        mean,
                        std,
                        maf,
                        has_missing: non_missing < n_samples,
                    };
                } else {
                    *st = PackedRowStats {
                        mean: 0.0_f64,
                        std: 1e-6_f64,
                        maf: 0.0_f64,
                        has_missing: true,
                    };
                }
            });
        };
        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    }

    let mut by_chr: HashMap<i32, Vec<usize>> = HashMap::new();
    for i in 0..m {
        by_chr.entry(chrom_vec[i]).or_default().push(i);
    }
    let chrom_groups: Vec<Vec<usize>> = by_chr.into_values().collect();

    let pair_lut = packed_pair_lut();
    let (bitplane_h, bitplane_l, bitplane_words, bitplane_masks) =
        build_bitplanes_u64(packed_flat, m, bytes_per_snp, n_samples, pool.as_ref());
    // If there is a single chromosome group, parallelize heavy inner window scans
    // to avoid the "only one outer task" bottleneck.
    let enable_intra_chrom_parallel = chrom_groups.len() == 1;
    let intra_parallel_min_neighbors = 96usize;
    let dropped_groups: Vec<Vec<bool>> = {
        let run = || {
            chrom_groups
                .par_iter()
                .map(|idx_list| {
                    prune_one_chrom_packed(
                        idx_list,
                        pos_vec,
                        packed_flat,
                        bytes_per_snp,
                        n_samples,
                        window_bp,
                        window_variants,
                        step_variants,
                        r2_threshold,
                        &stats,
                        pair_lut,
                        &byte_lut.code4,
                        &bitplane_h,
                        &bitplane_l,
                        bitplane_words,
                        &bitplane_masks,
                        enable_intra_chrom_parallel,
                        intra_parallel_min_neighbors,
                    )
                })
                .collect::<Vec<Vec<bool>>>()
        };
        if let Some(tp) = &pool {
            tp.install(run)
        } else {
            run()
        }
    };

    let mut keep = vec![true; m];
    for (idx_list, dropped) in chrom_groups.iter().zip(dropped_groups.iter()) {
        for (local_idx, &global_idx) in idx_list.iter().enumerate() {
            if dropped[local_idx] {
                keep[global_idx] = false;
            }
        }
    }
    Ok(keep)
}

fn normalize_plink_prefix(p: &str) -> String {
    let s = p.trim();
    let low = s.to_ascii_lowercase();
    if low.ends_with(".bed") || low.ends_with(".bim") || low.ends_with(".fam") {
        return s[..s.len() - 4].to_string();
    }
    s.to_string()
}

fn read_bim_lines_chrom_pos(prefix: &str) -> Result<(Vec<String>, Vec<i32>, Vec<i64>), String> {
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| format!("{bim_path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut lines: Vec<String> = Vec::new();
    let mut chrom_codes: Vec<i32> = Vec::new();
    let mut positions: Vec<i64> = Vec::new();
    let mut chr_map: HashMap<String, i32> = HashMap::new();
    let mut next_code: i32 = 0;
    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{bim_path}:{}: {}", line_no + 1, e))?;
        let t = l.trim();
        if t.is_empty() {
            continue;
        }
        let toks: Vec<&str> = t.split_whitespace().collect();
        if toks.len() < 4 {
            return Err(format!(
                "{bim_path}:{}: malformed BIM row (need >=4 columns)",
                line_no + 1
            ));
        }
        let chr_raw = toks[0].to_string();
        let pos = toks[3].parse::<i64>().map_err(|_| {
            format!(
                "{bim_path}:{}: invalid POS column '{}'",
                line_no + 1,
                toks[3]
            )
        })?;
        let code = if let Some(&c) = chr_map.get(&chr_raw) {
            c
        } else {
            let c = next_code;
            chr_map.insert(chr_raw, c);
            next_code = next_code.saturating_add(1);
            c
        };
        lines.push(t.to_string());
        chrom_codes.push(code);
        positions.push(pos);
    }
    if lines.is_empty() {
        return Err(format!("no variant rows in {bim_path}"));
    }
    Ok((lines, chrom_codes, positions))
}

fn read_bed_payload(prefix: &str, n_samples: usize, n_snps: usize) -> Result<Vec<u8>, String> {
    let bed_path = format!("{prefix}.bed");
    let mut f = File::open(&bed_path).map_err(|e| format!("{bed_path}: {e}"))?;
    let mut bytes = Vec::<u8>::new();
    f.read_to_end(&mut bytes)
        .map_err(|e| format!("{bed_path}: {e}"))?;
    if bytes.len() < 3 {
        return Err(format!("{bed_path}: invalid BED header"));
    }
    if bytes[0] != 0x6C || bytes[1] != 0x1B || bytes[2] != 0x01 {
        return Err(format!(
            "{bed_path}: unsupported BED header (expect SNP-major 0x6C 0x1B 0x01)"
        ));
    }
    let bps = (n_samples + 3) / 4;
    let expect = n_snps.saturating_mul(bps);
    let got = bytes.len().saturating_sub(3);
    if got != expect {
        return Err(format!(
            "{bed_path}: payload size mismatch, got {got}, expected {expect} (n_snps={n_snps}, n_samples={n_samples})"
        ));
    }
    Ok(bytes[3..].to_vec())
}

fn write_pruned_plink(
    src_prefix: &str,
    out_prefix: &str,
    keep: &[bool],
    bed_payload: &[u8],
    bytes_per_snp: usize,
    bim_lines: &[String],
) -> Result<(usize, usize), String> {
    if keep.len() != bim_lines.len() {
        return Err(format!(
            "keep mask length mismatch: keep={}, bim_rows={}",
            keep.len(),
            bim_lines.len()
        ));
    }
    if bed_payload.len() != keep.len().saturating_mul(bytes_per_snp) {
        return Err(format!(
            "bed payload mismatch: got {}, expected {}",
            bed_payload.len(),
            keep.len().saturating_mul(bytes_per_snp)
        ));
    }
    let out_bed = format!("{out_prefix}.bed");
    let out_bim = format!("{out_prefix}.bim");
    let out_fam = format!("{out_prefix}.fam");
    let src_fam = format!("{src_prefix}.fam");
    if let Some(parent) = Path::new(&out_bed).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    let mut wbed = BufWriter::new(File::create(&out_bed).map_err(|e| format!("{out_bed}: {e}"))?);
    wbed.write_all(&[0x6C, 0x1B, 0x01])
        .map_err(|e| format!("{out_bed}: {e}"))?;
    let mut wbim = BufWriter::new(File::create(&out_bim).map_err(|e| format!("{out_bim}: {e}"))?);

    let mut kept = 0usize;
    for i in 0..keep.len() {
        if !keep[i] {
            continue;
        }
        let s = i.saturating_mul(bytes_per_snp);
        let e = s.saturating_add(bytes_per_snp);
        wbed.write_all(&bed_payload[s..e])
            .map_err(|e| format!("{out_bed}: {e}"))?;
        wbim.write_all(bim_lines[i].as_bytes())
            .and_then(|_| wbim.write_all(b"\n"))
            .map_err(|e| format!("{out_bim}: {e}"))?;
        kept = kept.saturating_add(1);
    }
    wbed.flush().map_err(|e| format!("{out_bed}: {e}"))?;
    wbim.flush().map_err(|e| format!("{out_bim}: {e}"))?;

    fs::copy(&src_fam, &out_fam).map_err(|e| format!("{src_fam} -> {out_fam}: {e}"))?;
    Ok((kept, keep.len()))
}

#[inline]
fn is_identity_indices(sample_idx: &[usize], n_samples: usize) -> bool {
    if sample_idx.len() != n_samples {
        return false;
    }
    sample_idx.iter().enumerate().all(|(i, &v)| i == v)
}

#[inline]
fn decode_row_centered_full_lut(
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
fn adaptive_grm_block_rows(
    requested_block_rows: usize,
    m: usize,
    n_out: usize,
    subset_full_scratch_len: usize,
    threads: usize,
) -> usize {
    let base = requested_block_rows.max(1).min(m.max(1));
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

#[inline]
fn decode_subset_row_from_full_scratch(
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

    // Decode full row once into per-thread scratch:
    // non-missing => {0,1,2}, missing => -9.
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
        // Match subset semantics with mean-imputation on selected samples.
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
fn decode_grm_block(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip_vec: &[bool],
    row_maf_vec: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    method: usize,
    eps: f32,
    code4_lut: &[[u8; 4]; 256],
    row_start: usize,
    row_end: usize,
    n: usize,
    out_block: &mut [f32],
    out_varsum: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    if cur_rows == 0 {
        return Ok(());
    }
    if out_block.len() < cur_rows.saturating_mul(n) {
        return Err("decode_grm_block: out_block too small".to_string());
    }
    if out_varsum.len() < cur_rows {
        return Err("decode_grm_block: out_varsum too small".to_string());
    }
    let block = &mut out_block[..cur_rows * n];
    let varsum = &mut out_varsum[..cur_rows];

    let mut decode_run = || {
        if full_sample_fast {
            block
                .par_chunks_mut(n)
                .zip(varsum.par_iter_mut())
                .enumerate()
                .for_each(|(off, (out_row, row_varsum_dst))| {
                    let idx = row_start + off;
                    let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                    let flip = row_flip_vec[idx];
                    let p = row_maf_vec[idx].clamp(0.0, 0.5);
                    let mean_g = 2.0_f32 * p;
                    let var = 2.0_f32 * p * (1.0_f32 - p);
                    let std_scale = if method == 2 {
                        if var > eps {
                            1.0_f32 / var.sqrt()
                        } else {
                            0.0_f32
                        }
                    } else {
                        1.0_f32
                    };
                    let value_lut: [f32; 4] = if flip {
                        [
                            (2.0_f32 - mean_g) * std_scale,
                            0.0_f32, // missing -> mean imputation => centered 0
                            (1.0_f32 - mean_g) * std_scale,
                            (0.0_f32 - mean_g) * std_scale,
                        ]
                    } else {
                        [
                            (0.0_f32 - mean_g) * std_scale,
                            0.0_f32, // missing -> mean imputation => centered 0
                            (1.0_f32 - mean_g) * std_scale,
                            (2.0_f32 - mean_g) * std_scale,
                        ]
                    };
                    decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, out_row);
                    if method == 1 {
                        *row_varsum_dst = var as f64;
                    }
                });
        } else {
            block
                .par_chunks_mut(n)
                .zip(varsum.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || vec![0.0_f32; n_samples],
                    |full_row, (off, (out_row, row_varsum_dst))| {
                        let idx = row_start + off;
                        let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                        let flip = row_flip_vec[idx];
                        let p = row_maf_vec[idx].clamp(0.0, 0.5);
                        let default_mean_g = 2.0_f32 * p;
                        let (var_centered, _row_sum) = decode_subset_row_from_full_scratch(
                            row,
                            n_samples,
                            sample_idx,
                            flip,
                            default_mean_g,
                            method,
                            eps,
                            code4_lut,
                            full_row.as_mut_slice(),
                            out_row,
                        );
                        if method == 1 {
                            *row_varsum_dst = var_centered;
                        }
                    },
                );
        }
    };

    if let Some(tp) = pool {
        tp.install(&mut decode_run);
    } else {
        decode_run();
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
type CblasInt = std::os::raw::c_int;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const CBLAS_COL_MAJOR: CblasInt = 102;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const CBLAS_NO_TRANS: CblasInt = 111;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const CBLAS_TRANS: CblasInt = 112;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SgemmBackend {
    Accelerate,
    OpenBlas,
    Blas,
    Rust,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
static SGEMM_BACKEND: OnceLock<SgemmBackend> = OnceLock::new();

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const HAS_OPENBLAS_BACKEND: bool = cfg!(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
));
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const HAS_ACCELERATE_BACKEND: bool = cfg!(target_os = "macos");
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const HAS_BLAS_BACKEND: bool = cfg!(all(
    target_os = "linux",
    jx_blas_available,
    not(all(feature = "blas-openblas", jx_openblas_available))
));

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn default_sgemm_backend() -> SgemmBackend {
    #[cfg(target_os = "macos")]
    {
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        return SgemmBackend::Accelerate;
    }
    #[cfg(target_os = "windows")]
    {
        return SgemmBackend::OpenBlas;
    }
    #[cfg(target_os = "linux")]
    {
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        if HAS_BLAS_BACKEND {
            return SgemmBackend::Blas;
        }
        return SgemmBackend::Rust;
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn resolve_sgemm_backend() -> SgemmBackend {
    let default_backend = default_sgemm_backend();
    let req = std::env::var("JX_RUST_BLAS_BACKEND")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "auto".to_string());
    match req.as_str() {
        "" | "auto" => default_backend,
        "openblas" => {
            if HAS_OPENBLAS_BACKEND {
                SgemmBackend::OpenBlas
            } else {
                default_backend
            }
        }
        "accelerate" => {
            if HAS_ACCELERATE_BACKEND {
                SgemmBackend::Accelerate
            } else {
                default_backend
            }
        }
        "blas" => {
            if HAS_BLAS_BACKEND {
                SgemmBackend::Blas
            } else {
                default_backend
            }
        }
        "rust" => SgemmBackend::Rust,
        _ => default_backend,
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn selected_sgemm_backend() -> SgemmBackend {
    *SGEMM_BACKEND.get_or_init(resolve_sgemm_backend)
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_accelerate(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
}

#[cfg(all(
    target_os = "linux",
    jx_blas_available,
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
#[link(name = "blas")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_blas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
}

#[cfg(all(
    target_os = "macos",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas0)
))]
#[link(name = "openblas")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
}

#[cfg(all(
    target_os = "macos",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas0
))]
#[link(name = "openblas.0")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
}

#[cfg(all(
    target_os = "linux",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas0)
))]
#[link(name = "openblas")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
}

#[cfg(all(
    target_os = "linux",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas0
))]
#[link(name = "openblas.0")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
}

#[cfg(all(target_os = "windows", not(jx_openblas_link_openblas_plain)))]
#[link(name = "libopenblas", kind = "static")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
}

#[cfg(all(target_os = "windows", jx_openblas_link_openblas_plain))]
#[link(name = "openblas", kind = "static")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_sgemm_rust(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    b: *const f32,
    ldb: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust SGEMM fallback expects column-major order"
    );
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);

    let a_cols = if transa == CBLAS_NO_TRANS { k } else { m };
    let b_cols = if transb == CBLAS_NO_TRANS { n } else { k };
    let a_slice = std::slice::from_raw_parts(a, lda.saturating_mul(a_cols));
    let b_slice = std::slice::from_raw_parts(b, ldb.saturating_mul(b_cols));
    let c_slice = std::slice::from_raw_parts_mut(c, ldc.saturating_mul(n));

    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0_f32;
            for p in 0..k {
                let av = if transa == CBLAS_NO_TRANS {
                    a_slice[row + p * lda]
                } else {
                    a_slice[p + row * lda]
                };
                let bv = if transb == CBLAS_NO_TRANS {
                    b_slice[p + col * ldb]
                } else {
                    b_slice[col + p * ldb]
                };
                acc += av * bv;
            }
            let idx = row + col * ldc;
            c_slice[idx] = alpha * acc + beta * c_slice[idx];
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_sgemm_dispatch(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    b: *const f32,
    ldb: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(target_os = "macos")]
            {
                cblas_sgemm_accelerate(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(any(
                target_os = "windows",
                all(feature = "blas-openblas", jx_openblas_available)
            ))]
            {
                cblas_sgemm_openblas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_sgemm_blas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Rust => {
            cblas_sgemm_rust(
                order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            );
            return;
        }
    }

    #[cfg(target_os = "macos")]
    {
        cblas_sgemm_accelerate(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(target_os = "linux", feature = "blas-openblas", jx_openblas_available))]
    {
        cblas_sgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_sgemm_blas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        not(jx_blas_available),
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_sgemm_rust(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(target_os = "windows")]
    {
        cblas_sgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

#[pyfunction]
pub fn rust_sgemm_backend() -> String {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        match selected_sgemm_backend() {
            SgemmBackend::Accelerate => "accelerate".to_string(),
            SgemmBackend::OpenBlas => "openblas".to_string(),
            SgemmBackend::Blas => "blas".to_string(),
            SgemmBackend::Rust => "rust".to_string(),
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "unsupported".to_string()
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn grm_rankk_update_cblas(
    grm: &mut [f32],
    block: &[f32],
    cur_rows: usize,
    n: usize,
    copy_rhs: bool,
    beta_zero_accum: bool,
) {
    let rhs_owned = if copy_rhs { Some(block.to_vec()) } else { None };
    let rhs_ptr = match rhs_owned.as_ref() {
        Some(v) => v.as_ptr(),
        None => block.as_ptr(),
    };
    if beta_zero_accum {
        let mut tmp = vec![0.0_f32; n * n];
        unsafe {
            // `block` is row-major (cur_rows, n), which is layout-equivalent to
            // a column-major matrix with shape (n, cur_rows). Use col-major GEMM
            // to compute: tmp = block^T @ block == A @ A^T.
            cblas_sgemm_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n as CblasInt,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                rhs_ptr,
                n as CblasInt,
                0.0,
                tmp.as_mut_ptr(),
                n as CblasInt,
            );
        }
        for (g, t) in grm.iter_mut().zip(tmp.iter()) {
            *g += *t;
        }
    } else {
        unsafe {
            cblas_sgemm_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n as CblasInt,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                rhs_ptr,
                n as CblasInt,
                1.0,
                grm.as_mut_ptr(),
                n as CblasInt,
            );
        }
    }
}

#[inline]
fn grm_rankk_update(
    grm: &mut [f32],
    block: &[f32],
    cur_rows: usize,
    n: usize,
    cblas_copy_rhs: bool,
    cblas_beta_zero_accum: bool,
) -> Result<(), String> {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        grm_rankk_update_cblas(
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            cblas_beta_zero_accum,
        );
        return Ok(());
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = (
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            cblas_beta_zero_accum,
        );
        Err(
            "Packed GRM requires CBLAS backend on this platform; fallback to streaming GRM."
                .to_string(),
        )
    }
}

fn parse_index_vec_i64(indices: &[i64], upper_bound: usize, label: &str) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(indices.len());
    for (i, &v) in indices.iter().enumerate() {
        if v < 0 {
            return Err(PyRuntimeError::new_err(format!(
                "{label}[{i}] must be >= 0, got {v}"
            )));
        }
        let u = v as usize;
        if u >= upper_bound {
            return Err(PyRuntimeError::new_err(format!(
                "{label}[{i}] out of range: {u} >= {upper_bound}"
            )));
        }
        out.push(u);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (packed, n_samples))]
pub fn bed_packed_row_flip_mask<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let byte_lut = packed_byte_lut();
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;

    let out = PyArray1::<bool>::zeros(py, [m], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    out_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(row_idx, dst)| {
            let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
            let mut alt_sum = 0usize;
            let mut non_missing = 0usize;
            for &b in row.iter().take(full_bytes) {
                let idx = b as usize;
                non_missing += byte_lut.nonmiss[idx] as usize;
                alt_sum += byte_lut.alt_sum[idx] as usize;
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
                        }
                        0b11 => {
                            non_missing += 1;
                            alt_sum += 2;
                        }
                        _ => {}
                    }
                }
            }
            if non_missing == 0 {
                *dst = false;
            } else {
                let p = (alt_sum as f64) / (2.0 * non_missing as f64);
                *dst = p > 0.5;
            }
        });

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_indices,
    row_flip,
    row_maf,
    sample_indices=None
))]
pub fn bed_packed_decode_rows_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip.len()
        )));
    }
    if row_maf.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        )));
    }

    let row_idx_raw = row_indices.as_slice()?;
    let row_idx = parse_index_vec_i64(row_idx_raw, m, "row_indices")?;

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n_out = sample_idx.len();

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };

    let mut out = vec![0.0_f32; row_idx.len() * n_out];
    out.par_chunks_mut(n_out)
        .enumerate()
        .for_each(|(i_row, out_row)| {
            let src_row = row_idx[i_row];
            let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
            let flip = row_flip[src_row];
            let mean_g = (2.0_f32 * row_maf[src_row]).max(0.0);
            for (j, &sidx) in sample_idx.iter().enumerate() {
                let b = row[sidx >> 2];
                let code = (b >> ((sidx & 3) * 2)) & 0b11;
                let mut gv = match code {
                    0b00 => 0.0_f32,
                    0b10 => 1.0_f32,
                    0b11 => 2.0_f32,
                    _ => mean_g, // missing
                };
                if flip && code != 0b01 {
                    gv = 2.0_f32 - gv;
                }
                out_row[j] = gv;
            }
        });

    let arr = PyArray2::<f32>::zeros(py, [row_idx.len(), n_out], false).into_bound();
    let arr_slice = unsafe {
        arr.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    arr_slice.copy_from_slice(&out);
    Ok(arr)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    chrom_codes,
    positions,
    window_bp=None,
    window_variants=None,
    step_variants=1,
    r2_threshold=0.2,
    threads=0
))]
pub fn bed_packed_ld_prune_maf_priority<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    chrom_codes: PyReadonlyArray1<'py, i32>,
    positions: PyReadonlyArray1<'py, i64>,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(r2_threshold.is_finite() && r2_threshold > 0.0 && r2_threshold <= 1.0) {
        return Err(PyRuntimeError::new_err(
            "r2_threshold must be finite and in (0, 1]",
        ));
    }
    if step_variants == 0 {
        return Err(PyRuntimeError::new_err("step_variants must be > 0"));
    }
    if window_bp.is_none() && window_variants.is_none() {
        return Err(PyRuntimeError::new_err(
            "provide one of window_bp or window_variants",
        ));
    }
    if let Some(bp) = window_bp {
        if bp <= 0 {
            return Err(PyRuntimeError::new_err("window_bp must be > 0"));
        }
    }
    if let Some(wv) = window_variants {
        if wv == 0 {
            return Err(PyRuntimeError::new_err("window_variants must be > 0"));
        }
    }

    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }
    let chrom_vec: Cow<[i32]> = match chrom_codes.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(chrom_codes.as_array().iter().copied().collect()),
    };
    let pos_vec: Cow<[i64]> = match positions.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(positions.as_array().iter().copied().collect()),
    };
    if chrom_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "chrom_codes length mismatch: got {}, expected {m}",
            chrom_vec.len()
        )));
    }
    if pos_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "positions length mismatch: got {}, expected {m}",
            pos_vec.len()
        )));
    }

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };

    let keep = py
        .detach(|| {
            bed_packed_ld_prune_keep(
                &packed_flat,
                m,
                bytes_per_snp,
                n_samples,
                &chrom_vec,
                &pos_vec,
                window_bp,
                window_variants,
                step_variants,
                r2_threshold,
                threads,
            )
        })
        .map_err(_map_err_string_to_py)?;

    let out = PyArray1::<bool>::zeros(py, [m], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&keep);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    src_prefix,
    out_prefix,
    window_bp=None,
    window_variants=None,
    step_variants=1,
    r2_threshold=0.2,
    threads=0
))]
pub fn bed_prune_to_plink_rust(
    py: Python<'_>,
    src_prefix: String,
    out_prefix: String,
    window_bp: Option<i64>,
    window_variants: Option<usize>,
    step_variants: usize,
    r2_threshold: f64,
    threads: usize,
) -> PyResult<(usize, usize)> {
    let src = normalize_plink_prefix(&src_prefix);
    let out = normalize_plink_prefix(&out_prefix);
    if src.is_empty() || out.is_empty() {
        return Err(PyRuntimeError::new_err(
            "src_prefix/out_prefix must not be empty",
        ));
    }

    let (bim_lines, chrom_codes, positions) =
        read_bim_lines_chrom_pos(&src).map_err(_map_err_string_to_py)?;
    let n_snps = bim_lines.len();
    let n_samples = crate::gfcore::read_fam(&src)
        .map_err(_map_err_string_to_py)?
        .len();
    if n_samples == 0 || n_snps == 0 {
        return Err(PyRuntimeError::new_err(
            "empty PLINK input (no samples or no SNPs)",
        ));
    }

    let bed_payload = read_bed_payload(&src, n_samples, n_snps).map_err(_map_err_string_to_py)?;
    let bytes_per_snp = (n_samples + 3) / 4;

    let keep = py
        .detach(|| {
            bed_packed_ld_prune_keep(
                &bed_payload,
                n_snps,
                bytes_per_snp,
                n_samples,
                &chrom_codes,
                &positions,
                window_bp,
                window_variants,
                step_variants,
                r2_threshold,
                threads,
            )
        })
        .map_err(_map_err_string_to_py)?;

    let (kept, total) =
        write_pruned_plink(&src, &out, &keep, &bed_payload, bytes_per_snp, &bim_lines)
            .map_err(_map_err_string_to_py)?;
    Ok((kept, total))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices,
    threads=0
))]
pub fn bed_packed_decode_stats_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }
    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;

    let (blk, sum_rows, sq_sum_rows) = py
        .detach(|| -> Result<_, String> {
            let mut blk = vec![0.0_f64; m * n_out];
            let mut sum_rows = vec![0.0_f64; m];
            let mut sq_sum_rows = vec![0.0_f64; m];

            let mut run = || {
                blk.par_chunks_mut(n_out)
                    .zip(sum_rows.par_iter_mut())
                    .zip(sq_sum_rows.par_iter_mut())
                    .enumerate()
                    .for_each(|(row_idx, ((out_row, sum_dst), sq_dst))| {
                        let row =
                            &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                        let flip = row_flip_vec[row_idx];
                        let mean_g = 2.0_f64 * row_maf_vec[row_idx] as f64;
                        let mut sum = 0.0_f64;
                        let mut sq_sum = 0.0_f64;
                        for (j, &sid) in sample_idx.iter().enumerate() {
                            let b = row[sid >> 2];
                            let code = (b >> ((sid & 3) * 2)) & 0b11;
                            let mut gv = match decode_plink_bed_hardcall(code) {
                                Some(v) => v,
                                None => mean_g,
                            };
                            if flip && code != 0b01 {
                                gv = 2.0_f64 - gv;
                            }
                            out_row[j] = gv;
                            sum += gv;
                            sq_sum += gv * gv;
                        }
                        *sum_dst = sum;
                        *sq_dst = sq_sum;
                    });
            };
            if let Some(tp) = &pool {
                tp.install(run);
            } else {
                run();
            }
            Ok((blk, sum_rows, sq_sum_rows))
        })
        .map_err(_map_err_string_to_py)?;

    let blk_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((m, n_out), blk)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let sum_arr = PyArray1::from_owned_array(py, Array1::from_vec(sum_rows)).into_bound();
    let sq_arr = PyArray1::from_owned_array(py, Array1::from_vec(sq_sum_rows)).into_bound();
    Ok((blk_arr, sum_arr, sq_arr))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices,
    m_alpha,
    m_mean,
    alpha_sum,
    mean_sq,
    mean_malpha,
    m_var_sum,
    block_rows=4096,
    threads=0
))]
pub fn cross_grm_times_alpha_packed_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    m_alpha: PyReadonlyArray1<'py, f64>,
    m_mean: PyReadonlyArray1<'py, f64>,
    alpha_sum: f64,
    mean_sq: f64,
    mean_malpha: f64,
    m_var_sum: f64,
    block_rows: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(m_var_sum.is_finite() && m_var_sum > 0.0) {
        return Err(PyRuntimeError::new_err(
            "m_var_sum must be finite and > 0 for compact cross-GRM prediction",
        ));
    }
    if !(alpha_sum.is_finite() && mean_sq.is_finite() && mean_malpha.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "alpha_sum/mean_sq/mean_malpha must be finite",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "packed must contain at least one SNP row",
        ));
    }
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }

    let m_alpha_vec: Cow<[f64]> = match m_alpha.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(m_alpha.as_array().iter().copied().collect()),
    };
    let m_mean_vec: Cow<[f64]> = match m_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(m_mean.as_array().iter().copied().collect()),
    };
    if m_alpha_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "m_alpha length mismatch: got {}, expected {m}",
            m_alpha_vec.len()
        )));
    }
    if m_mean_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "m_mean length mismatch: got {}, expected {m}",
            m_mean_vec.len()
        )));
    }

    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let full_sample_fast = is_identity_indices(&sample_idx, n_samples);

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;

    let row_step = block_rows.max(1).min(m.max(1));
    let byte_lut = packed_byte_lut();

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    let m_alpha_f32: Vec<f32> = m_alpha_vec.iter().map(|&v| v as f32).collect();
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    let m_mean_f32: Vec<f32> = m_mean_vec.iter().map(|&v| v as f32).collect();

    let (term1, term2) = py
        .detach(|| -> Result<(Vec<f64>, Vec<f64>), String> {
            let mut term1 = vec![0.0_f64; n_out];
            let mut term2 = vec![0.0_f64; n_out];
            let mut block = vec![0.0_f32; row_step * n_out];
            let mut tick = 0usize;

            #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
            let mut tmp1 = vec![0.0_f32; n_out];
            #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
            let mut tmp2 = vec![0.0_f32; n_out];

            let mut run = || -> Result<(), String> {
                for st in (0..m).step_by(row_step) {
                    let ed = (st + row_step).min(m);
                    let cur_rows = ed - st;
                    let blk_slice = &mut block[..cur_rows * n_out];

                    if full_sample_fast {
                        blk_slice.par_chunks_mut(n_out).enumerate().for_each(
                            |(local_row, out_row)| {
                                let row_idx = st + local_row;
                                let row = &packed_flat
                                    [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                let mean_g = 2.0_f32 * row_maf_vec[row_idx];
                                let value_lut = if row_flip_vec[row_idx] {
                                    [2.0_f32, mean_g, 1.0_f32, 0.0_f32]
                                } else {
                                    [0.0_f32, mean_g, 1.0_f32, 2.0_f32]
                                };
                                decode_row_centered_full_lut(
                                    row,
                                    n_samples,
                                    &byte_lut.code4,
                                    &value_lut,
                                    out_row,
                                );
                            },
                        );
                    } else {
                        blk_slice.par_chunks_mut(n_out).enumerate().for_each(
                            |(local_row, out_row)| {
                                let row_idx = st + local_row;
                                let row = &packed_flat
                                    [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                let flip = row_flip_vec[row_idx];
                                let mean_g = 2.0_f32 * row_maf_vec[row_idx];
                                for (j, &sid) in sample_idx.iter().enumerate() {
                                    let b = row[sid >> 2];
                                    let code = (b >> ((sid & 3) * 2)) & 0b11;
                                    let mut gv = match code {
                                        0b00 => 0.0_f32,
                                        0b10 => 1.0_f32,
                                        0b11 => 2.0_f32,
                                        _ => mean_g,
                                    };
                                    if flip && code != 0b01 {
                                        gv = 2.0_f32 - gv;
                                    }
                                    out_row[j] = gv;
                                }
                            },
                        );
                    }

                    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
                    {
                        let alpha_blk = &m_alpha_f32[st..ed];
                        let mean_blk = &m_mean_f32[st..ed];
                        unsafe {
                            // `blk_slice` is row-major (cur_rows, n_out), which is layout-equivalent
                            // to col-major (n_out, cur_rows). Compute tmp = blk^T @ vec.
                            cblas_sgemm_dispatch(
                                CBLAS_COL_MAJOR,
                                CBLAS_NO_TRANS,
                                CBLAS_NO_TRANS,
                                n_out as CblasInt,
                                1 as CblasInt,
                                cur_rows as CblasInt,
                                1.0,
                                blk_slice.as_ptr(),
                                n_out as CblasInt,
                                alpha_blk.as_ptr(),
                                cur_rows as CblasInt,
                                0.0,
                                tmp1.as_mut_ptr(),
                                n_out as CblasInt,
                            );
                            cblas_sgemm_dispatch(
                                CBLAS_COL_MAJOR,
                                CBLAS_NO_TRANS,
                                CBLAS_NO_TRANS,
                                n_out as CblasInt,
                                1 as CblasInt,
                                cur_rows as CblasInt,
                                1.0,
                                blk_slice.as_ptr(),
                                n_out as CblasInt,
                                mean_blk.as_ptr(),
                                cur_rows as CblasInt,
                                0.0,
                                tmp2.as_mut_ptr(),
                                n_out as CblasInt,
                            );
                        }
                        for j in 0..n_out {
                            term1[j] += tmp1[j] as f64;
                            term2[j] += tmp2[j] as f64;
                        }
                    }

                    #[cfg(not(any(
                        target_os = "macos",
                        target_os = "linux",
                        target_os = "windows"
                    )))]
                    {
                        let alpha_blk = &m_alpha_vec[st..ed];
                        let mean_blk = &m_mean_vec[st..ed];
                        for j in 0..n_out {
                            let mut acc1 = 0.0_f64;
                            let mut acc2 = 0.0_f64;
                            for r in 0..cur_rows {
                                let gv = blk_slice[r * n_out + j] as f64;
                                acc1 += gv * alpha_blk[r];
                                acc2 += gv * mean_blk[r];
                            }
                            term1[j] += acc1;
                            term2[j] += acc2;
                        }
                    }

                    tick += cur_rows;
                    if tick >= row_step.saturating_mul(64).max(1) {
                        _check_ctrlc()?;
                        tick = 0;
                    }
                }
                Ok(())
            };

            if let Some(tp) = &pool {
                tp.install(run)?;
            } else {
                run()?;
            }
            Ok((term1, term2))
        })
        .map_err(_map_err_string_to_py)?;

    let const_term = mean_sq * alpha_sum - mean_malpha;
    let inv_var = 1.0_f64 / m_var_sum;
    let mut out = vec![0.0_f64; n_out];
    for j in 0..n_out {
        out[j] = (term1[j] - term2[j] * alpha_sum + const_term) * inv_var;
    }

    let out_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n_out, 1), out)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok(out_arr)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices,
    alpha,
    block_rows=4096,
    threads=0
))]
pub fn packed_malpha_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    alpha: PyReadonlyArray1<'py, f64>,
    block_rows: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "packed must contain at least one SNP row",
        ));
    }
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }

    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let alpha_vec: Cow<[f64]> = match alpha.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(alpha.as_array().iter().copied().collect()),
    };
    if alpha_vec.len() != n_out {
        return Err(PyRuntimeError::new_err(format!(
            "alpha length mismatch: got {}, expected {} (len(sample_indices))",
            alpha_vec.len(),
            n_out
        )));
    }
    let full_sample_fast = is_identity_indices(&sample_idx, n_samples);

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;
    let row_step = block_rows.max(1).min(m.max(1));
    let byte_lut = packed_byte_lut();

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    let alpha_f32: Vec<f32> = alpha_vec.iter().map(|&v| v as f32).collect();

    let out = py
        .detach(|| -> Result<Vec<f64>, String> {
            let mut out = vec![0.0_f64; m];
            let mut block = vec![0.0_f32; row_step * n_out];
            let mut tick = 0usize;

            #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
            let mut tmp = vec![0.0_f32; row_step];

            let mut run = || -> Result<(), String> {
                for st in (0..m).step_by(row_step) {
                    let ed = (st + row_step).min(m);
                    let cur_rows = ed - st;
                    let blk_slice = &mut block[..cur_rows * n_out];

                    if full_sample_fast {
                        blk_slice.par_chunks_mut(n_out).enumerate().for_each(
                            |(local_row, out_row)| {
                                let row_idx = st + local_row;
                                let row = &packed_flat
                                    [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                let mean_g = 2.0_f32 * row_maf_vec[row_idx];
                                let value_lut = if row_flip_vec[row_idx] {
                                    [2.0_f32, mean_g, 1.0_f32, 0.0_f32]
                                } else {
                                    [0.0_f32, mean_g, 1.0_f32, 2.0_f32]
                                };
                                decode_row_centered_full_lut(
                                    row,
                                    n_samples,
                                    &byte_lut.code4,
                                    &value_lut,
                                    out_row,
                                );
                            },
                        );
                    } else {
                        blk_slice.par_chunks_mut(n_out).enumerate().for_each(
                            |(local_row, out_row)| {
                                let row_idx = st + local_row;
                                let row = &packed_flat
                                    [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                let flip = row_flip_vec[row_idx];
                                let mean_g = 2.0_f32 * row_maf_vec[row_idx];
                                for (j, &sid) in sample_idx.iter().enumerate() {
                                    let b = row[sid >> 2];
                                    let code = (b >> ((sid & 3) * 2)) & 0b11;
                                    let mut gv = match code {
                                        0b00 => 0.0_f32,
                                        0b10 => 1.0_f32,
                                        0b11 => 2.0_f32,
                                        _ => mean_g,
                                    };
                                    if flip && code != 0b01 {
                                        gv = 2.0_f32 - gv;
                                    }
                                    out_row[j] = gv;
                                }
                            },
                        );
                    }

                    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
                    {
                        let out_blk = &mut out[st..ed];
                        let tmp_blk = &mut tmp[..cur_rows];
                        unsafe {
                            // `blk_slice` is row-major (cur_rows, n_out), layout-equivalent to
                            // col-major (n_out, cur_rows). We need:
                            // out_blk = blk_slice @ alpha
                            // -> out_blk(col-major cur_rows x 1) = A^T (cur_rows x n_out) * alpha (n_out x 1)
                            cblas_sgemm_dispatch(
                                CBLAS_COL_MAJOR,
                                CBLAS_TRANS,
                                CBLAS_NO_TRANS,
                                cur_rows as CblasInt,
                                1 as CblasInt,
                                n_out as CblasInt,
                                1.0,
                                blk_slice.as_ptr(),
                                n_out as CblasInt,
                                alpha_f32.as_ptr(),
                                n_out as CblasInt,
                                0.0,
                                tmp_blk.as_mut_ptr(),
                                cur_rows as CblasInt,
                            );
                        }
                        for i in 0..cur_rows {
                            out_blk[i] = tmp_blk[i] as f64;
                        }
                    }

                    #[cfg(not(any(
                        target_os = "macos",
                        target_os = "linux",
                        target_os = "windows"
                    )))]
                    {
                        let out_blk = &mut out[st..ed];
                        for r in 0..cur_rows {
                            let mut acc = 0.0_f64;
                            let row = &blk_slice[r * n_out..(r + 1) * n_out];
                            for j in 0..n_out {
                                acc += row[j] as f64 * alpha_vec[j];
                            }
                            out_blk[r] = acc;
                        }
                    }

                    tick += cur_rows;
                    if tick >= row_step.saturating_mul(64).max(1) {
                        _check_ctrlc()?;
                        tick = 0;
                    }
                }
                Ok(())
            };

            if let Some(tp) = &pool {
                tp.install(run)?;
            } else {
                run()?;
            }
            Ok(out)
        })
        .map_err(_map_err_string_to_py)?;

    let out_arr = PyArray1::from_owned_array(py, Array1::from_vec(out)).into_bound();
    Ok(out_arr)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    method=1,
    block_cols=2048,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if method != 1 && method != 2 {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method={method}; expected 1 (centered) or 2 (standardized)"
        )));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "packed must contain at least one SNP row",
        ));
    }
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let full_sample_fast = is_identity_indices(&sample_idx, n_samples);

    let varsum_full = if method == 1 && full_sample_fast {
        let mut acc = 0.0_f64;
        for &maf in row_maf_vec.iter() {
            let p = maf as f64;
            let v = 2.0_f64 * p * (1.0_f64 - p);
            if v.is_finite() && v > 0.0 {
                acc += v;
            }
        }
        if !(acc.is_finite() && acc > 0.0) {
            return Err(PyRuntimeError::new_err(
                "invalid centered GRM denominator: sum(2p(1-p)) <= 0",
            ));
        }
        acc
    } else {
        0.0_f64
    };

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;
    let row_step = adaptive_grm_block_rows(
        block_cols.max(1),
        m.max(1),
        n,
        if full_sample_fast { 0 } else { n_samples },
        threads,
    );
    let block_target_mb = std::env::var("JX_GRM_PACKED_BLOCK_TARGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(64.0_f64);
    let stage_timing = _env_truthy("JX_GRM_PACKED_STAGE_TIMING");
    let notify_step = if progress_every == 0 {
        row_step
    } else {
        progress_every.max(1)
    };
    let cblas_copy_rhs = std::env::var("JX_GRM_PACKED_CBLAS_COPY_RHS")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("1") || s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let cblas_beta_zero_accum = std::env::var("JX_GRM_PACKED_CBLAS_BETA0")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        // Default ON: avoids observed diagonal drift for very large SNP blocks.
        .unwrap_or(true);

    let grm_vec = py
        .detach(move || -> Result<Vec<f32>, String> {
            let total_t0 = Instant::now();
            let mut decode_secs = 0.0_f64;
            let mut gemm_secs = 0.0_f64;
            let mut grm = vec![0.0_f32; n * n];
            let mut varsum_acc = varsum_full;
            let eps = 1e-12_f32;
            let byte_lut = packed_byte_lut();

            let mut block_a = vec![0.0_f32; row_step * n];
            let mut block_b = vec![0.0_f32; row_step * n];
            let mut varsum_a = vec![0.0_f64; row_step];
            let mut varsum_b = vec![0.0_f64; row_step];
            // Default strategy: always use dual-buffer overlap when there is
            // enough work and more than one thread.
            let pipeline_overlap = (threads != 1) && (m > row_step);

            let mut cur_start = 0usize;
            let mut cur_end = row_step.min(m);
            let mut use_a = true;
            let mut last_notified = 0usize;

            let decode_t0 = Instant::now();
            decode_grm_block(
                packed_flat.as_ref(),
                bytes_per_snp,
                n_samples,
                row_flip_vec.as_ref(),
                row_maf_vec.as_ref(),
                &sample_idx,
                full_sample_fast,
                method,
                eps,
                &byte_lut.code4,
                cur_start,
                cur_end,
                n,
                &mut block_a,
                &mut varsum_a,
                pool.as_ref(),
            )?;
            decode_secs += decode_t0.elapsed().as_secs_f64();

            loop {
                let cur_rows = cur_end.saturating_sub(cur_start);
                if cur_rows == 0 {
                    break;
                }
                let next_start = cur_end;
                let next_end = (next_start + row_step).min(m);
                let has_next = next_start < m;

                if pipeline_overlap && has_next {
                    if use_a {
                        let (gemm_dt, decode_dt) =
                            std::thread::scope(|scope| -> Result<(f64, f64), String> {
                                let decode_handle = scope.spawn(|| {
                                    let dt0 = Instant::now();
                                    let r = decode_grm_block(
                                        packed_flat.as_ref(),
                                        bytes_per_snp,
                                        n_samples,
                                        row_flip_vec.as_ref(),
                                        row_maf_vec.as_ref(),
                                        &sample_idx,
                                        full_sample_fast,
                                        method,
                                        eps,
                                        &byte_lut.code4,
                                        next_start,
                                        next_end,
                                        n,
                                        &mut block_b,
                                        &mut varsum_b,
                                        pool.as_ref(),
                                    );
                                    (r, dt0.elapsed().as_secs_f64())
                                });

                                let gt0 = Instant::now();
                                let gemm_res = grm_rankk_update(
                                    &mut grm,
                                    &block_a[..cur_rows * n],
                                    cur_rows,
                                    n,
                                    cblas_copy_rhs,
                                    cblas_beta_zero_accum,
                                );
                                let gemm_dt = gt0.elapsed().as_secs_f64();

                                let (decode_res, decode_dt) = decode_handle
                                    .join()
                                    .map_err(|_| "decode worker thread panicked".to_string())?;
                                gemm_res?;
                                decode_res?;
                                Ok((gemm_dt, decode_dt))
                            })?;
                        gemm_secs += gemm_dt;
                        decode_secs += decode_dt;
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_a[..cur_rows].iter().sum::<f64>();
                        }
                    } else {
                        let (gemm_dt, decode_dt) =
                            std::thread::scope(|scope| -> Result<(f64, f64), String> {
                                let decode_handle = scope.spawn(|| {
                                    let dt0 = Instant::now();
                                    let r = decode_grm_block(
                                        packed_flat.as_ref(),
                                        bytes_per_snp,
                                        n_samples,
                                        row_flip_vec.as_ref(),
                                        row_maf_vec.as_ref(),
                                        &sample_idx,
                                        full_sample_fast,
                                        method,
                                        eps,
                                        &byte_lut.code4,
                                        next_start,
                                        next_end,
                                        n,
                                        &mut block_a,
                                        &mut varsum_a,
                                        pool.as_ref(),
                                    );
                                    (r, dt0.elapsed().as_secs_f64())
                                });

                                let gt0 = Instant::now();
                                let gemm_res = grm_rankk_update(
                                    &mut grm,
                                    &block_b[..cur_rows * n],
                                    cur_rows,
                                    n,
                                    cblas_copy_rhs,
                                    cblas_beta_zero_accum,
                                );
                                let gemm_dt = gt0.elapsed().as_secs_f64();

                                let (decode_res, decode_dt) = decode_handle
                                    .join()
                                    .map_err(|_| "decode worker thread panicked".to_string())?;
                                gemm_res?;
                                decode_res?;
                                Ok((gemm_dt, decode_dt))
                            })?;
                        gemm_secs += gemm_dt;
                        decode_secs += decode_dt;
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_b[..cur_rows].iter().sum::<f64>();
                        }
                    }
                } else {
                    if use_a {
                        let gemm_t0 = Instant::now();
                        grm_rankk_update(
                            &mut grm,
                            &block_a[..cur_rows * n],
                            cur_rows,
                            n,
                            cblas_copy_rhs,
                            cblas_beta_zero_accum,
                        )?;
                        gemm_secs += gemm_t0.elapsed().as_secs_f64();
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_a[..cur_rows].iter().sum::<f64>();
                        }
                        if has_next {
                            let decode_t0 = Instant::now();
                            decode_grm_block(
                                packed_flat.as_ref(),
                                bytes_per_snp,
                                n_samples,
                                row_flip_vec.as_ref(),
                                row_maf_vec.as_ref(),
                                &sample_idx,
                                full_sample_fast,
                                method,
                                eps,
                                &byte_lut.code4,
                                next_start,
                                next_end,
                                n,
                                &mut block_b,
                                &mut varsum_b,
                                pool.as_ref(),
                            )?;
                            decode_secs += decode_t0.elapsed().as_secs_f64();
                        }
                    } else {
                        let gemm_t0 = Instant::now();
                        grm_rankk_update(
                            &mut grm,
                            &block_b[..cur_rows * n],
                            cur_rows,
                            n,
                            cblas_copy_rhs,
                            cblas_beta_zero_accum,
                        )?;
                        gemm_secs += gemm_t0.elapsed().as_secs_f64();
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_b[..cur_rows].iter().sum::<f64>();
                        }
                        if has_next {
                            let decode_t0 = Instant::now();
                            decode_grm_block(
                                packed_flat.as_ref(),
                                bytes_per_snp,
                                n_samples,
                                row_flip_vec.as_ref(),
                                row_maf_vec.as_ref(),
                                &sample_idx,
                                full_sample_fast,
                                method,
                                eps,
                                &byte_lut.code4,
                                next_start,
                                next_end,
                                n,
                                &mut block_a,
                                &mut varsum_a,
                                pool.as_ref(),
                            )?;
                            decode_secs += decode_t0.elapsed().as_secs_f64();
                        }
                    }
                }

                let done = cur_end;
                if (done >= last_notified.saturating_add(notify_step)) || (done == m) {
                    last_notified = done;
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done, m))?;
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                    } else {
                        Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                    }
                }

                if !has_next {
                    break;
                }
                cur_start = next_start;
                cur_end = next_end;
                use_a = !use_a;
            }

            let scale = if method == 1 {
                if !(varsum_acc.is_finite() && varsum_acc > 0.0) {
                    return Err("invalid centered GRM denominator in subset path".to_string());
                }
                varsum_acc as f32
            } else {
                m as f32
            };
            if !(scale.is_finite() && scale > 0.0) {
                return Err("invalid GRM scale factor".to_string());
            }
            let inv_scale = 1.0_f32 / scale;
            for i in 0..n {
                let ii = i * n + i;
                grm[ii] *= inv_scale;
                for j in 0..i {
                    let idx_ij = i * n + j;
                    let idx_ji = j * n + i;
                    let v = 0.5_f32 * (grm[idx_ij] + grm[idx_ji]) * inv_scale;
                    grm[idx_ij] = v;
                    grm[idx_ji] = v;
                }
            }
            if stage_timing {
                let total_secs = total_t0.elapsed().as_secs_f64();
                let stage_sum = decode_secs + gemm_secs;
                let overlap_secs = (stage_sum - total_secs).max(0.0_f64);
                let other_secs = if stage_sum <= total_secs {
                    total_secs - stage_sum
                } else {
                    0.0_f64
                };
                let to_pct = |x: f64| -> f64 {
                    if total_secs > 0.0 {
                        x * 100.0 / total_secs
                    } else {
                        0.0
                    }
                };
                eprintln!(
                    "GRM stage timing: decode={:.3}s ({:.1}%), gemm={:.3}s ({:.1}%), \
other={:.3}s ({:.1}%), overlap={:.3}s ({:.1}%), total={:.3}s, row_step={}, \
block_target_mb={:.1}, n_samples={}, full_sample={}, threads={}",
                    decode_secs,
                    to_pct(decode_secs),
                    gemm_secs,
                    to_pct(gemm_secs),
                    other_secs,
                    to_pct(other_secs),
                    overlap_secs,
                    to_pct(overlap_secs),
                    total_secs,
                    row_step,
                    block_target_mb,
                    n,
                    if full_sample_fast { "yes" } else { "no" },
                    if threads > 0 {
                        threads
                    } else {
                        rayon::current_num_threads()
                    }
                );
            }
            Ok(grm)
        })
        .map_err(_map_err_string_to_py)?;

    let grm_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n, n), grm_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok(grm_arr)
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    method=1,
    maf_threshold=0.02,
    max_missing_rate=0.05,
    block_cols=2048,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_bed_f32<'py>(
    py: Python<'py>,
    prefix: String,
    method: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, usize, usize)> {
    let (packed_arr, miss_arr, maf_arr, _std_arr, n_samples) =
        crate::gfreader::load_bed_2bit_packed(py, prefix)?;

    let miss_ro = miss_arr.readonly();
    let miss_vec: Vec<f32> = match miss_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => miss_ro.as_array().iter().copied().collect(),
    };
    let maf_ro = maf_arr.readonly();
    let maf_vec: Vec<f32> = match maf_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => maf_ro.as_array().iter().copied().collect(),
    };
    if miss_vec.len() != maf_vec.len() {
        return Err(PyRuntimeError::new_err(format!(
            "packed BED stat length mismatch: miss={}, maf={}",
            miss_vec.len(),
            maf_vec.len()
        )));
    }
    let m_total = maf_vec.len();
    if m_total == 0 {
        return Err(PyRuntimeError::new_err(
            "No SNPs found in packed BED input.",
        ));
    }

    let maf_thr = maf_threshold.clamp(0.0, 0.5);
    let miss_thr = max_missing_rate.clamp(0.0, 1.0);
    let mut keep_idx: Vec<usize> = Vec::with_capacity(m_total);
    for i in 0..m_total {
        let maf_i = maf_vec[i];
        if maf_i < maf_thr || maf_i > (1.0 - maf_thr) {
            continue;
        }
        if miss_vec[i] > miss_thr {
            continue;
        }
        keep_idx.push(i);
    }
    if keep_idx.is_empty() {
        return Err(PyRuntimeError::new_err(
            "No SNPs remained after packed BED filtering; GRM is empty.",
        ));
    }
    let eff_m = keep_idx.len();

    if eff_m == m_total {
        let row_flip = bed_packed_row_flip_mask(py, packed_arr.readonly(), n_samples)?;
        let grm = grm_packed_f32(
            py,
            packed_arr.readonly(),
            n_samples,
            row_flip.readonly(),
            maf_arr.readonly(),
            None,
            method,
            block_cols,
            threads,
            progress_callback,
            progress_every,
        )?;
        return Ok((grm, eff_m, n_samples));
    }

    let packed_ro = packed_arr.readonly();
    let packed_view = packed_ro.as_array();
    if packed_view.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed BED payload must be 2D (m, bytes_per_snp).",
        ));
    }
    let bytes_per_snp = packed_view.shape()[1];
    let packed_slice: Cow<[u8]> = match packed_ro.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_view.iter().copied().collect()),
    };

    let mut packed_keep = vec![0_u8; eff_m * bytes_per_snp];
    for (dst_row, &src_row) in keep_idx.iter().enumerate() {
        let src_off = src_row * bytes_per_snp;
        let dst_off = dst_row * bytes_per_snp;
        packed_keep[dst_off..dst_off + bytes_per_snp]
            .copy_from_slice(&packed_slice[src_off..src_off + bytes_per_snp]);
    }
    let maf_keep: Vec<f32> = keep_idx.iter().map(|&i| maf_vec[i]).collect();

    let packed_keep_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((eff_m, bytes_per_snp), packed_keep)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let maf_keep_arr = PyArray1::from_owned_array(py, Array1::from_vec(maf_keep)).into_bound();

    let row_flip = bed_packed_row_flip_mask(py, packed_keep_arr.readonly(), n_samples)?;
    let grm = grm_packed_f32(
        py,
        packed_keep_arr.readonly(),
        n_samples,
        row_flip.readonly(),
        maf_keep_arr.readonly(),
        None,
        method,
        block_cols,
        threads,
        progress_callback,
        progress_every,
    )?;
    Ok((grm, eff_m, n_samples))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    method=1,
    block_cols=2048,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_f32_with_stats<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f64>>, f64)> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if method != 1 && method != 2 {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method={method}; expected 1 (centered) or 2 (standardized)"
        )));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "packed must contain at least one SNP row",
        ));
    }
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let full_sample_fast = is_identity_indices(&sample_idx, n_samples);

    let varsum_full = if method == 1 && full_sample_fast {
        let mut acc = 0.0_f64;
        for &maf in row_maf_vec.iter() {
            let p = maf as f64;
            let v = 2.0_f64 * p * (1.0_f64 - p);
            if v.is_finite() && v > 0.0 {
                acc += v;
            }
        }
        if !(acc.is_finite() && acc > 0.0) {
            return Err(PyRuntimeError::new_err(
                "invalid centered GRM denominator: sum(2p(1-p)) <= 0",
            ));
        }
        acc
    } else {
        0.0_f64
    };

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;
    let row_step = adaptive_grm_block_rows(
        block_cols.max(1),
        m.max(1),
        n,
        if full_sample_fast { 0 } else { n_samples },
        threads,
    );
    let notify_step = if progress_every == 0 {
        row_step
    } else {
        progress_every.max(1)
    };
    let cblas_copy_rhs = std::env::var("JX_GRM_PACKED_CBLAS_COPY_RHS")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("1") || s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let cblas_beta_zero_accum = std::env::var("JX_GRM_PACKED_CBLAS_BETA0")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        // Default ON: avoids observed diagonal drift for very large SNP blocks.
        .unwrap_or(true);

    let (grm_vec, row_sum_vec, varsum_ret) = py
        .detach(move || -> Result<(Vec<f32>, Vec<f64>, f64), String> {
            let mut grm = vec![0.0_f32; n * n];
            let mut block = vec![0.0_f32; row_step * n];
            let mut varsum_acc = varsum_full;
            let eps = 1e-12_f32;
            let byte_lut = packed_byte_lut();
            let mut row_sum_all = vec![0.0_f64; m];

            let mut last_notified = 0usize;
            for row_start in (0..m).step_by(row_step) {
                let row_end = (row_start + row_step).min(m);
                let cur_rows = row_end - row_start;
                let cur_block = &mut block[..cur_rows * n];
                let mut block_varsum = vec![0.0_f64; cur_rows];
                let mut block_rowsum = vec![0.0_f64; cur_rows];

                let mut decode_run = || {
                    if full_sample_fast {
                        cur_block
                            .par_chunks_mut(n)
                            .zip(block_varsum.par_iter_mut())
                            .zip(block_rowsum.par_iter_mut())
                            .enumerate()
                            .for_each(|(off, ((out_row, row_varsum_dst), row_sum_dst))| {
                                let idx = row_start + off;
                                let row =
                                    &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                                let flip = row_flip_vec[idx];
                                let p = row_maf_vec[idx].clamp(0.0, 0.5);
                                let mean_g = 2.0_f32 * p;
                                let var = 2.0_f32 * p * (1.0_f32 - p);
                                let std_scale = if method == 2 {
                                    if var > eps {
                                        1.0_f32 / var.sqrt()
                                    } else {
                                        0.0_f32
                                    }
                                } else {
                                    1.0_f32
                                };
                                let value_lut: [f32; 4] = if flip {
                                    [
                                        (2.0_f32 - mean_g) * std_scale,
                                        0.0_f32, // missing -> mean imputation => centered 0
                                        (1.0_f32 - mean_g) * std_scale,
                                        (0.0_f32 - mean_g) * std_scale,
                                    ]
                                } else {
                                    [
                                        (0.0_f32 - mean_g) * std_scale,
                                        0.0_f32, // missing -> mean imputation => centered 0
                                        (1.0_f32 - mean_g) * std_scale,
                                        (2.0_f32 - mean_g) * std_scale,
                                    ]
                                };
                                decode_row_centered_full_lut(
                                    row,
                                    n_samples,
                                    &byte_lut.code4,
                                    &value_lut,
                                    out_row,
                                );
                                if method == 1 {
                                    *row_varsum_dst = var as f64;
                                }
                                *row_sum_dst = (mean_g as f64) * (n as f64);
                            });
                    } else {
                        cur_block
                            .par_chunks_mut(n)
                            .zip(block_varsum.par_iter_mut())
                            .zip(block_rowsum.par_iter_mut())
                            .enumerate()
                            .for_each_init(
                                || vec![0.0_f32; n_samples],
                                |full_row, (off, ((out_row, row_varsum_dst), row_sum_dst))| {
                                    let idx = row_start + off;
                                    let row = &packed_flat
                                        [idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                                    let flip = row_flip_vec[idx];
                                    let p = row_maf_vec[idx].clamp(0.0, 0.5);
                                    let default_mean_g = 2.0_f32 * p;
                                    let (var_centered, row_sum) =
                                        decode_subset_row_from_full_scratch(
                                            row,
                                            n_samples,
                                            &sample_idx,
                                            flip,
                                            default_mean_g,
                                            method,
                                            eps,
                                            &byte_lut.code4,
                                            full_row.as_mut_slice(),
                                            out_row,
                                        );
                                    if method == 1 {
                                        *row_varsum_dst = var_centered;
                                    }
                                    *row_sum_dst = row_sum;
                                },
                            );
                    }
                };

                if let Some(tp) = &pool {
                    tp.install(&mut decode_run);
                } else {
                    decode_run();
                }

                grm_rankk_update(
                    &mut grm,
                    cur_block,
                    cur_rows,
                    n,
                    cblas_copy_rhs,
                    cblas_beta_zero_accum,
                )?;
                if method == 1 && !full_sample_fast {
                    varsum_acc += block_varsum.iter().sum::<f64>();
                }
                row_sum_all[row_start..row_end].copy_from_slice(&block_rowsum);

                let done = row_end;
                if (done >= last_notified.saturating_add(notify_step)) || (done == m) {
                    last_notified = done;
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done, m))?;
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                    } else {
                        Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                    }
                }
            }

            let scale = if method == 1 {
                if !(varsum_acc.is_finite() && varsum_acc > 0.0) {
                    return Err("invalid centered GRM denominator in subset path".to_string());
                }
                varsum_acc as f32
            } else {
                m as f32
            };
            if !(scale.is_finite() && scale > 0.0) {
                return Err("invalid GRM scale factor".to_string());
            }
            let inv_scale = 1.0_f32 / scale;
            for i in 0..n {
                let ii = i * n + i;
                grm[ii] *= inv_scale;
                for j in 0..i {
                    let idx_ij = i * n + j;
                    let idx_ji = j * n + i;
                    let v = 0.5_f32 * (grm[idx_ij] + grm[idx_ji]) * inv_scale;
                    grm[idx_ij] = v;
                    grm[idx_ji] = v;
                }
            }
            let varsum_ret = if method == 1 { varsum_acc } else { m as f64 };
            Ok((grm, row_sum_all, varsum_ret))
        })
        .map_err(_map_err_string_to_py)?;

    let grm_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n, n), grm_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let row_sum_arr = PyArray1::from_owned_array(py, Array1::from_vec(row_sum_vec)).into_bound();
    Ok((grm_arr, row_sum_arr, varsum_ret))
}

#[pyfunction]
#[pyo3(signature = (
    y,
    x,
    ixx,
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    step=10000,
    threads=0
))]
pub fn glmf32_packed<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    step: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let ixx_arr = ixx.as_array();
    let packed_arr = packed.as_array();

    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip.len()
        )));
    }
    if row_maf.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        )));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };

    let n = y.len();
    if sample_idx.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (xn, q0) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
        return Err(PyRuntimeError::new_err("ixx must be (q0,q0)"));
    }
    if n <= q0 + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n too small: require n > q0+1, got n={n}, q0={q0}"
        )));
    }

    let row_stride = q0 + 3;
    let dim = q0 + 1;
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let ixx_flat: Cow<[f64]> = match ixx.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ixx_arr.iter().copied().collect()),
    };
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };

    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();

    let out = PyArray2::<f64>::zeros(py, [m, row_stride], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    let pool = get_cached_pool(threads)?;
    py.detach(|| {
        let mut runner = || {
            let mut i_marker = 0usize;
            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);
                let block = &mut out_slice[i_marker * row_stride..(i_marker + cnt) * row_stride];
                block.par_chunks_mut(row_stride).enumerate().for_each_init(
                    || GlmScratch::new(q0),
                    |scr, (l, row_out)| {
                        let idx = i_marker + l;
                        scr.reset_xs();

                        let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                        let flip = row_flip[idx];
                        let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);

                        let mut sy = 0.0_f64;
                        let mut ss = 0.0_f64;
                        for (k, &sidx) in sample_idx.iter().enumerate() {
                            let b = row[sidx >> 2];
                            let code = (b >> ((sidx & 3) * 2)) & 0b11;
                            let mut gv = match decode_plink_bed_hardcall(code) {
                                Some(v) => v,
                                None => mean_g,
                            };
                            if flip && code != 0b01 {
                                gv = 2.0 - gv;
                            }

                            sy += gv * y[k];
                            ss += gv * gv;
                            let xrow = &x_flat[k * q0..(k + 1) * q0];
                            for j in 0..q0 {
                                scr.xs[j] += xrow[j] * gv;
                            }
                        }

                        xs_t_ixx_into(&scr.xs, &ixx_flat, q0, &mut scr.b21);
                        let t2 = dot(&scr.b21, &scr.xs);
                        let b22 = ss - t2;

                        let (invb22, df) = if b22 < 1e-8 {
                            (0.0, (n as i32) - (q0 as i32))
                        } else {
                            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                        };
                        if df <= 0 {
                            row_out.fill(f64::NAN);
                            return;
                        }

                        build_ixxs_into(&ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);
                        scr.rhs[..q0].copy_from_slice(&xy);
                        scr.rhs[q0] = sy;
                        matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);

                        let beta_rhs = dot(&scr.beta, &scr.rhs);
                        let ve = (yy - beta_rhs) / (df as f64);

                        for ff in 0..dim {
                            let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
                            let t = scr.beta[ff] / se;
                            row_out[2 + ff] = student_t_p_two_sided(t, df);
                        }

                        if invb22 == 0.0 {
                            row_out[0] = f64::NAN;
                            row_out[1] = f64::NAN;
                            row_out[2 + q0] = f64::NAN;
                        } else {
                            row_out[0] = scr.beta[q0];
                            row_out[1] = (scr.ixxs[q0 * dim + q0] * ve).sqrt();
                        }
                    },
                );
                i_marker += cnt;
            }
        };

        if let Some(p) = &pool {
            p.install(runner);
        } else {
            runner();
        }
    });

    Ok(out)
}

#[inline]
fn decode_packed_rows_to_sample_major(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_indices: &[usize],
    sample_idx: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
) -> Vec<f64> {
    let n = sample_idx.len();
    let k = row_indices.len();
    let mut out = vec![0.0_f64; n * k]; // row-major: (n_samples_used, k_rows)
    for (col, &row_idx) in row_indices.iter().enumerate() {
        let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
        let flip = row_flip[row_idx];
        let mean_g = 2.0_f64 * row_maf[row_idx] as f64;
        for (i, &sidx) in sample_idx.iter().enumerate() {
            let b = row[sidx >> 2];
            let code = (b >> ((sidx & 3) * 2)) & 0b11;
            let mut gv = match decode_plink_bed_hardcall(code) {
                Some(v) => v,
                None => mean_g,
            };
            if flip && code != 0b01 {
                gv = 2.0 - gv;
            }
            out[i * k + col] = gv;
        }
    }
    out
}

#[inline]
fn decode_dense_rows_to_sample_major(
    g_arr: &numpy::ndarray::ArrayView2<'_, f32>,
    g_slice_opt: Option<&[f32]>,
    n: usize,
    row_indices: &[usize],
) -> Vec<f64> {
    let k = row_indices.len();
    let mut out = vec![0.0_f64; n * k]; // row-major: (n_samples, k_rows)
    for (col, &row_idx) in row_indices.iter().enumerate() {
        if let Some(g_slice) = g_slice_opt {
            let row = &g_slice[row_idx * n..(row_idx + 1) * n];
            for i in 0..n {
                out[i * k + col] = row[i] as f64;
            }
        } else {
            for i in 0..n {
                out[i * k + col] = g_arr[(row_idx, i)] as f64;
            }
        }
    }
    out
}

fn select_lead_indices(sz: i64, n_lead: usize, pvalue: &[f64], pos: &[i64]) -> Vec<usize> {
    let m = pvalue.len();
    if m == 0 || n_lead == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        let ba = pos[a].div_euclid(sz);
        let bb = pos[b].div_euclid(sz);
        match ba.cmp(&bb) {
            std::cmp::Ordering::Equal => pvalue[a].total_cmp(&pvalue[b]),
            other => other,
        }
    });

    let mut lead: Vec<usize> = Vec::new();
    let mut last_bin: Option<i64> = None;
    for &idx in order.iter() {
        let b = pos[idx].div_euclid(sz);
        if last_bin.map_or(true, |lb| lb != b) {
            lead.push(idx);
            last_bin = Some(b);
        }
    }

    lead.sort_by(|&a, &b| pvalue[a].total_cmp(&pvalue[b]));
    if lead.len() > n_lead {
        lead.truncate(n_lead);
    }
    lead.sort_unstable();
    lead
}

fn solve_linear_system_stable(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let p = a.nrows();
    if p == 0 {
        return Some(DVector::zeros(0));
    }
    let eye = DMatrix::<f64>::identity(p, p);
    for ridge in [0.0_f64, 1e-10, 1e-8, 1e-6] {
        let mut a_use = a.clone();
        if ridge > 0.0 {
            a_use += &eye * ridge;
        }
        if let Some(ch) = a_use.cholesky() {
            return Some(ch.solve(b));
        }
    }
    let lu = a.clone().lu();
    lu.solve(b)
}

fn farmcpu_ll_score_from_sample_major(
    y: &[f64],
    x: &[f64], // row-major (n, p)
    n: usize,
    p: usize,
    snp_pool_sample_major: &[f64], // row-major (n, k)
    k: usize,
    delta_exp_start: f64,
    delta_exp_end: f64,
    delta_step: f64,
    svd_eps: f64,
) -> Result<f64, String> {
    if n == 0 || p == 0 {
        return Err("invalid shape in ll score: empty y/X".to_string());
    }
    if y.len() != n {
        return Err("invalid y length in ll score".to_string());
    }
    if x.len() != n * p {
        return Err("invalid X length in ll score".to_string());
    }
    if snp_pool_sample_major.len() != n * k {
        return Err("invalid snp_pool length in ll score".to_string());
    }
    if !(delta_step.is_finite() && delta_step > 0.0) {
        return Err("delta_step must be positive and finite".to_string());
    }

    let yvec = DVector::<f64>::from_column_slice(y);
    let xmat = DMatrix::<f64>::from_row_slice(n, p, x);
    let snp_pool = if k > 0 {
        DMatrix::<f64>::from_row_slice(n, k, snp_pool_sample_major)
    } else {
        DMatrix::<f64>::zeros(n, 0)
    };

    let mut d_start = delta_exp_start;
    let mut d_end = delta_exp_end;

    if k > 0 {
        // Match Python logic: if any SNP variance is zero, force a degenerate high-delta grid.
        let mut has_zero_var = n <= 1;
        if !has_zero_var {
            for col in 0..k {
                let mut mean = 0.0_f64;
                for i in 0..n {
                    mean += snp_pool[(i, col)];
                }
                mean /= n as f64;
                let mut ss = 0.0_f64;
                for i in 0..n {
                    let d = snp_pool[(i, col)] - mean;
                    ss += d * d;
                }
                let var = ss / ((n - 1) as f64);
                if var <= 0.0 || !var.is_finite() {
                    has_zero_var = true;
                    break;
                }
            }
        }
        if has_zero_var {
            d_start = 100.0;
            d_end = 100.0;
        }
    }

    let (u1, d): (DMatrix<f64>, Vec<f64>) = if k == 0 {
        (DMatrix::<f64>::zeros(n, 0), Vec::new())
    } else {
        let svd = snp_pool.svd(true, false);
        let u = svd
            .u
            .ok_or_else(|| "SVD failed to produce U in ll score".to_string())?;
        let s = svd.singular_values;
        let keep: Vec<usize> = (0..s.len()).filter(|&i| s[i] > svd_eps).collect();
        if keep.is_empty() {
            (DMatrix::<f64>::zeros(n, 0), Vec::new())
        } else {
            let r = keep.len();
            let mut u1_data = Vec::with_capacity(n * r);
            for i in 0..n {
                for &c in keep.iter() {
                    u1_data.push(u[(i, c)]);
                }
            }
            let d = keep.iter().map(|&c| s[c] * s[c]).collect::<Vec<_>>();
            (DMatrix::<f64>::from_row_slice(n, r, &u1_data), d)
        }
    };

    let r = d.len();
    let u1tx = if r > 0 {
        u1.transpose() * &xmat
    } else {
        DMatrix::<f64>::zeros(0, p)
    };
    let u1ty = if r > 0 {
        u1.transpose() * &yvec
    } else {
        DVector::<f64>::zeros(0)
    };
    let x_u = if r > 0 {
        &xmat - &u1 * &u1tx
    } else {
        xmat.clone()
    };
    let y_u = if r > 0 {
        &yvec - &u1 * &u1ty
    } else {
        yvec.clone()
    };

    let xtx_u = x_u.transpose() * &x_u;
    let xty_u = x_u.transpose() * &y_u;

    let mut best_ll = f64::NEG_INFINITY;
    let mut expv = d_start;
    let end = d_end + 1e-12;
    while expv <= end {
        let delta = expv.exp();
        if !(delta.is_finite() && delta > 0.0) {
            expv += delta_step;
            continue;
        }

        let mut beta1 = DMatrix::<f64>::zeros(p, p);
        let mut beta3 = DVector::<f64>::zeros(p);
        let mut part12 = 0.0_f64;

        if r > 0 {
            for t in 0..r {
                let dt = d[t] + delta;
                if !(dt.is_finite() && dt > 0.0) {
                    continue;
                }
                let w = 1.0 / dt;
                part12 += dt.ln();
                for i in 0..p {
                    let vi = u1tx[(t, i)];
                    beta3[i] += vi * w * u1ty[t];
                    for j in 0..p {
                        beta1[(i, j)] += vi * w * u1tx[(t, j)];
                    }
                }
            }
        }

        let a = beta1 + (&xtx_u / delta);
        let b = beta3 + (&xty_u / delta);
        let Some(beta) = solve_linear_system_stable(&a, &b) else {
            expv += delta_step;
            continue;
        };

        let part11 = (n as f64) * (2.0 * PI).ln();
        let part13 = ((n - r) as f64) * delta.ln();
        let part1 = -0.5 * (part11 + part12 + part13);

        let mut part221 = 0.0_f64;
        if r > 0 {
            for t in 0..r {
                let mut pred = 0.0_f64;
                for j in 0..p {
                    pred += u1tx[(t, j)] * beta[j];
                }
                let resid = u1ty[t] - pred;
                part221 += (resid * resid) / (d[t] + delta);
            }
        }

        let resid_i = &y_u - &x_u * &beta;
        let part222 = resid_i.dot(&resid_i) / delta;
        let part22 = part221 + part222;
        if !(part22.is_finite() && part22 > 0.0) {
            expv += delta_step;
            continue;
        }
        let part2 = -0.5 * ((n as f64) + (n as f64) * (part22 / (n as f64)).ln());
        let ll = part1 + part2;
        if ll > best_ll {
            best_ll = ll;
        }
        expv += delta_step;
    }

    if !best_ll.is_finite() {
        return Err("failed to evaluate valid LL in farmcpu ll scorer".to_string());
    }
    Ok(-2.0 * best_ll)
}

fn farmcpu_super_keep_from_sample_major(
    sample_major: &[f64], // row-major (n, k)
    n: usize,
    k: usize,
    pval: &[f64],
    thr: f64,
) -> Vec<bool> {
    let mut keep = vec![true; k];
    if k == 0 || n == 0 {
        return keep;
    }

    let mut centered = vec![0.0_f64; k * n]; // col-major per SNP
    let mut std = vec![0.0_f64; k];
    for c in 0..k {
        let mut mean = 0.0_f64;
        for i in 0..n {
            mean += sample_major[i * k + c];
        }
        mean /= n as f64;

        let mut ss = 0.0_f64;
        for i in 0..n {
            let z = sample_major[i * k + c] - mean;
            centered[c * n + i] = z;
            ss += z * z;
        }
        std[c] = if n > 1 {
            (ss / ((n - 1) as f64)).sqrt()
        } else {
            0.0
        };
    }

    for i in 0..k {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..k {
            if !keep[j] {
                continue;
            }
            let denom = ((n.saturating_sub(1)) as f64) * std[i] * std[j];
            let cij = if denom > 0.0 && denom.is_finite() {
                let mut dot_ij = 0.0_f64;
                for t in 0..n {
                    dot_ij += centered[i * n + t] * centered[j * n + t];
                }
                dot_ij / denom
            } else {
                f64::NAN
            };
            if cij >= thr || cij <= -thr {
                if pval[i] >= pval[j] {
                    keep[i] = false;
                } else {
                    keep[j] = false;
                    break;
                }
            }
        }
    }
    keep
}

#[pyfunction]
#[pyo3(signature = (
    sz,
    n_lead,
    pvalue,
    pos,
    packed,
    n_samples,
    row_flip,
    row_maf,
    y,
    x,
    sample_indices=None,
    delta_exp_start=-5.0,
    delta_exp_end=5.0,
    delta_step=0.1,
    svd_eps=1e-8
))]
pub fn farmcpu_rem_packed(
    py: Python<'_>,
    sz: i64,
    n_lead: usize,
    pvalue: PyReadonlyArray1<'_, f64>,
    pos: PyReadonlyArray1<'_, i64>,
    packed: PyReadonlyArray2<'_, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'_, bool>,
    row_maf: PyReadonlyArray1<'_, f32>,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    sample_indices: Option<PyReadonlyArray1<'_, i64>>,
    delta_exp_start: f64,
    delta_exp_end: f64,
    delta_step: f64,
    svd_eps: f64,
) -> PyResult<(f64, Vec<i64>)> {
    if sz <= 0 {
        return Err(PyRuntimeError::new_err("sz must be > 0"));
    }
    let pvalue = pvalue.as_slice()?;
    let pos = pos.as_slice()?;
    if pvalue.len() != pos.len() {
        return Err(PyRuntimeError::new_err(
            "pvalue and pos length mismatch in farmcpu_rem_packed",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    if m != pvalue.len() {
        return Err(PyRuntimeError::new_err(format!(
            "packed row count mismatch: packed m={m}, pvalue len={}",
            pvalue.len()
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let n = y.len();
    if x_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    let p = x_arr.shape()[1];
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    if sample_idx.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let leadidx = select_lead_indices(sz, n_lead, pvalue, pos);
    let k = leadidx.len();
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let (score, leadidx_i64) = py.detach(|| -> PyResult<(f64, Vec<i64>)> {
        let snp_pool_sample_major = decode_packed_rows_to_sample_major(
            &packed_flat,
            bytes_per_snp,
            &leadidx,
            &sample_idx,
            row_flip,
            row_maf,
        );

        let score = farmcpu_ll_score_from_sample_major(
            y,
            &x_flat,
            n,
            p,
            &snp_pool_sample_major,
            k,
            delta_exp_start,
            delta_exp_end,
            delta_step,
            svd_eps,
        )
        .map_err(PyRuntimeError::new_err)?;

        let leadidx_i64 = leadidx.iter().map(|&v| v as i64).collect::<Vec<_>>();
        Ok((score, leadidx_i64))
    })?;
    Ok((score, leadidx_i64))
}

#[pyfunction]
#[pyo3(signature = (
    row_indices,
    pvalue,
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    thr=0.7
))]
pub fn farmcpu_super_packed<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, i64>,
    pvalue: PyReadonlyArray1<'py, f64>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    thr: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ridx_raw = row_indices.as_slice()?;
    let pval = pvalue.as_slice()?;
    if ridx_raw.len() != pval.len() {
        return Err(PyRuntimeError::new_err(
            "row_indices and pvalue length mismatch in farmcpu_super_packed",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let ridx = parse_index_vec_i64(ridx_raw, m, "row_indices")?;
    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    let k = ridx.len();

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let keep = py.detach(|| {
        let sample_major = decode_packed_rows_to_sample_major(
            &packed_flat,
            bytes_per_snp,
            &ridx,
            &sample_idx,
            row_flip,
            row_maf,
        );
        farmcpu_super_keep_from_sample_major(&sample_major, n, k, pval, thr)
    });

    let out = PyArray1::<bool>::zeros(py, [k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&keep);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    sz,
    n_lead,
    pvalue,
    pos,
    g,
    y,
    x,
    delta_exp_start=-5.0,
    delta_exp_end=5.0,
    delta_step=0.1,
    svd_eps=1e-8
))]
pub fn farmcpu_rem_dense(
    py: Python<'_>,
    sz: i64,
    n_lead: usize,
    pvalue: PyReadonlyArray1<'_, f64>,
    pos: PyReadonlyArray1<'_, i64>,
    g: PyReadonlyArray2<'_, f32>,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    delta_exp_start: f64,
    delta_exp_end: f64,
    delta_step: f64,
    svd_eps: f64,
) -> PyResult<(f64, Vec<i64>)> {
    if sz <= 0 {
        return Err(PyRuntimeError::new_err("sz must be > 0"));
    }
    let pvalue = pvalue.as_slice()?;
    let pos = pos.as_slice()?;
    if pvalue.len() != pos.len() {
        return Err(PyRuntimeError::new_err(
            "pvalue and pos length mismatch in farmcpu_rem_dense",
        ));
    }

    let g_arr = g.as_array();
    if g_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("g must be 2D (m, n)"));
    }
    let m = g_arr.shape()[0];
    let n = g_arr.shape()[1];
    if m != pvalue.len() {
        return Err(PyRuntimeError::new_err(format!(
            "g row count mismatch: g m={m}, pvalue len={}",
            pvalue.len()
        )));
    }

    let y = y.as_slice()?;
    if y.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "y length mismatch: got {}, expected n={n}",
            y.len()
        )));
    }

    let x_arr = x.as_array();
    if x_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    let p = x_arr.shape()[1];
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let g_slice_opt = g.as_slice().ok();

    let leadidx = select_lead_indices(sz, n_lead, pvalue, pos);
    let k = leadidx.len();

    let (score, leadidx_i64) = py.detach(|| -> PyResult<(f64, Vec<i64>)> {
        let snp_pool_sample_major =
            decode_dense_rows_to_sample_major(&g_arr, g_slice_opt, n, &leadidx);
        let score = farmcpu_ll_score_from_sample_major(
            y,
            &x_flat,
            n,
            p,
            &snp_pool_sample_major,
            k,
            delta_exp_start,
            delta_exp_end,
            delta_step,
            svd_eps,
        )
        .map_err(PyRuntimeError::new_err)?;
        let leadidx_i64 = leadidx.iter().map(|&v| v as i64).collect::<Vec<_>>();
        Ok((score, leadidx_i64))
    })?;
    Ok((score, leadidx_i64))
}

#[pyfunction]
#[pyo3(signature = (
    row_indices,
    pvalue,
    g,
    thr=0.7
))]
pub fn farmcpu_super_dense<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, i64>,
    pvalue: PyReadonlyArray1<'py, f64>,
    g: PyReadonlyArray2<'py, f32>,
    thr: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ridx_raw = row_indices.as_slice()?;
    let pval = pvalue.as_slice()?;
    if ridx_raw.len() != pval.len() {
        return Err(PyRuntimeError::new_err(
            "row_indices and pvalue length mismatch in farmcpu_super_dense",
        ));
    }

    let g_arr = g.as_array();
    if g_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("g must be 2D (m, n)"));
    }
    let m = g_arr.shape()[0];
    let n = g_arr.shape()[1];
    let ridx = parse_index_vec_i64(ridx_raw, m, "row_indices")?;
    let k = ridx.len();
    let g_slice_opt = g.as_slice().ok();

    let keep = py.detach(|| {
        let sample_major = decode_dense_rows_to_sample_major(&g_arr, g_slice_opt, n, &ridx);
        farmcpu_super_keep_from_sample_major(&sample_major, n, k, pval, thr)
    });

    let out = PyArray1::<bool>::zeros(py, [k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&keep);
    Ok(out)
}

/// Fast GLM interface:
/// y: (n,) float64
/// X: (n, q0) float64
/// ixx: (q0, q0) float64
/// G: (m, n) float32 (marker rows)
///
/// Returns: (m, q0 + 3) float64
///   col0: beta_snp
///   col1: se_snp
///   col2..: p-values for coefficients (q0 covariates + snp)
#[pyfunction]
#[pyo3(signature = (y, x, ixx, g, step=10000, threads=0))]
pub fn glmf32<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray2<'py, f32>,
    step: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let ixx_arr = ixx.as_array();
    let g_arr = g.as_array();

    let n = y.len();
    let (xn, q0) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
        return Err(PyRuntimeError::new_err("ixx must be (q0,q0)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "G must be shape (m, n) for float32 fast path",
        ));
    }
    if n <= q0 + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n too small: require n > q0+1, got n={n}, q0={q0}"
        )));
    }
    let g_slice_opt = g.as_slice().ok();

    let m = g_arr.shape()[0];
    let row_stride = q0 + 3;
    let dim = q0 + 1;

    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let ixx_flat: Cow<[f64]> = match ixx.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ixx_arr.iter().copied().collect()),
    };

    let mut xy = vec![0.0; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();

    let out = PyArray2::<f64>::zeros(py, [m, row_stride], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    let pool = get_cached_pool(threads)?;

    py.detach(|| {
        let mut runner = || {
            let mut i_marker = 0usize;
            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);
                let block = &mut out_slice[i_marker * row_stride..(i_marker + cnt) * row_stride];

                block.par_chunks_mut(row_stride).enumerate().for_each_init(
                    || GlmScratch::new(q0),
                    |scr, (l, row_out)| {
                        let idx = i_marker + l;
                        scr.reset_xs();

                        let mut sy = 0.0_f64;
                        let mut ss = 0.0_f64;

                        if let Some(gs) = g_slice_opt {
                            let grow = &gs[idx * n..(idx + 1) * n];
                            for k in 0..n {
                                let gv = grow[k] as f64;
                                sy += gv * y[k];
                                ss += gv * gv;

                                let row = &x_flat[k * q0..(k + 1) * q0];
                                for j in 0..q0 {
                                    scr.xs[j] += row[j] * gv;
                                }
                            }
                        } else {
                            for k in 0..n {
                                let gv = g_arr[(idx, k)] as f64; // float32 -> f64
                                sy += gv * y[k];
                                ss += gv * gv;

                                let row = &x_flat[k * q0..(k + 1) * q0];
                                for j in 0..q0 {
                                    scr.xs[j] += row[j] * gv;
                                }
                            }
                        }

                        xs_t_ixx_into(&scr.xs, &ixx_flat, q0, &mut scr.b21);
                        let t2 = dot(&scr.b21, &scr.xs);
                        let b22 = ss - t2;

                        let (invb22, df) = if b22 < 1e-8 {
                            (0.0, (n as i32) - (q0 as i32))
                        } else {
                            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                        };
                        if df <= 0 {
                            row_out.fill(f64::NAN);
                            return;
                        }

                        build_ixxs_into(&ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);

                        scr.rhs[..q0].copy_from_slice(&xy);
                        scr.rhs[q0] = sy;

                        matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);

                        let beta_rhs = dot(&scr.beta, &scr.rhs);
                        let ve = (yy - beta_rhs) / (df as f64);

                        for ff in 0..dim {
                            let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
                            let t = scr.beta[ff] / se;
                            row_out[2 + ff] = student_t_p_two_sided(t, df);
                        }

                        if invb22 == 0.0 {
                            row_out[0] = f64::NAN;
                            row_out[1] = f64::NAN;
                            row_out[2 + q0] = f64::NAN;
                        } else {
                            let beta_snp = scr.beta[q0];
                            let se_snp = (scr.ixxs[q0 * dim + q0] * ve).sqrt();
                            row_out[0] = beta_snp;
                            row_out[1] = se_snp;
                        }
                    },
                );

                i_marker += cnt;
            }
        };

        if let Some(p) = &pool {
            p.install(runner);
        } else {
            runner();
        }
    });

    Ok(out)
}

/// Full GLM interface (returns beta/se/p for all coefficients).
///
/// Returns: (m, 3 * (q0 + 1)) float64
///   For each coefficient j (0..q0 covariates, q0 is SNP):
///     col[3*j+0] = beta_j
///     col[3*j+1] = se_j
///     col[3*j+2] = p_j
#[pyfunction]
#[pyo3(signature = (y, x, ixx, g, step=10000, threads=0))]
pub fn glmf32_full<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray2<'py, f32>,
    step: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let ixx_arr = ixx.as_array();
    let g_arr = g.as_array();

    let n = y.len();
    let (xn, q0) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
        return Err(PyRuntimeError::new_err("ixx must be (q0,q0)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "G must be shape (m, n) for float32 fast path",
        ));
    }
    if n <= q0 + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n too small: require n > q0+1, got n={n}, q0={q0}"
        )));
    }
    let g_slice_opt = g.as_slice().ok();

    let m = g_arr.shape()[0];
    let dim = q0 + 1;
    let row_stride = dim * 3;

    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let ixx_flat: Cow<[f64]> = match ixx.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ixx_arr.iter().copied().collect()),
    };

    let mut xy = vec![0.0; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();

    let out = PyArray2::<f64>::zeros(py, [m, row_stride], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    let pool = get_cached_pool(threads)?;

    py.detach(|| {
        let mut runner = || {
            let mut i_marker = 0usize;
            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);
                let block = &mut out_slice[i_marker * row_stride..(i_marker + cnt) * row_stride];

                block.par_chunks_mut(row_stride).enumerate().for_each_init(
                    || GlmScratch::new(q0),
                    |scr, (l, row_out)| {
                        let idx = i_marker + l;
                        scr.reset_xs();

                        let mut sy = 0.0_f64;
                        let mut ss = 0.0_f64;

                        if let Some(gs) = g_slice_opt {
                            let grow = &gs[idx * n..(idx + 1) * n];
                            for k in 0..n {
                                let gv = grow[k] as f64;
                                sy += gv * y[k];
                                ss += gv * gv;

                                let row = &x_flat[k * q0..(k + 1) * q0];
                                for j in 0..q0 {
                                    scr.xs[j] += row[j] * gv;
                                }
                            }
                        } else {
                            for k in 0..n {
                                let gv = g_arr[(idx, k)] as f64;
                                sy += gv * y[k];
                                ss += gv * gv;

                                let row = &x_flat[k * q0..(k + 1) * q0];
                                for j in 0..q0 {
                                    scr.xs[j] += row[j] * gv;
                                }
                            }
                        }

                        xs_t_ixx_into(&scr.xs, &ixx_flat, q0, &mut scr.b21);
                        let t2 = dot(&scr.b21, &scr.xs);
                        let b22 = ss - t2;

                        let (invb22, df) = if b22 < 1e-8 {
                            (0.0, (n as i32) - (q0 as i32))
                        } else {
                            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                        };
                        if df <= 0 {
                            row_out.fill(f64::NAN);
                            return;
                        }

                        build_ixxs_into(&ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);

                        scr.rhs[..q0].copy_from_slice(&xy);
                        scr.rhs[q0] = sy;

                        matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);

                        let beta_rhs = dot(&scr.beta, &scr.rhs);
                        let ve = (yy - beta_rhs) / (df as f64);

                        for ff in 0..dim {
                            let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
                            let t = scr.beta[ff] / se;
                            let base = 3 * ff;
                            row_out[base] = scr.beta[ff];
                            row_out[base + 1] = se;
                            row_out[base + 2] = student_t_p_two_sided(t, df);
                        }

                        if invb22 == 0.0 {
                            let base = 3 * q0;
                            row_out[base] = f64::NAN;
                            row_out[base + 1] = f64::NAN;
                            row_out[base + 2] = f64::NAN;
                        }
                    },
                );

                i_marker += cnt;
            }
        };

        if let Some(p) = &pool {
            p.install(runner);
        } else {
            runner();
        }
    });

    Ok(out)
}

// =============================================================================
// LMM REML chunk (lmm_reml_chunk_f32)
// =============================================================================

fn reml_loglike(
    log10_lbd: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: Option<&[f64]>,
    n: usize,
    p_cov: usize,
) -> f64 {
    let use_snp = snp.is_some();
    let snp = snp.unwrap_or(&[]);
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return -1e8;
    }

    if use_snp && snp.len() != n {
        return -1e8;
    }
    let p = p_cov + if use_snp { 1 } else { 0 };
    if n <= p {
        return -1e8;
    }

    let mut v = vec![0.0_f64; n];
    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return -1e8;
        }
        v[i] = vv;
        vinv[i] = 1.0 / vv;
    }

    let dim = p;
    let mut xtv_inv_x = vec![0.0_f64; dim * dim];
    let mut xtv_inv_y = vec![0.0_f64; dim];

    for i in 0..n {
        let vi = vinv[i];
        let yi = y[i];
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xtv_inv_y[r] += vi * xir * yi;

            for c in 0..=r {
                let xic = if c < p_cov {
                    xcov[i * p_cov + c]
                } else {
                    snp[i]
                };
                xtv_inv_x[r * dim + c] += vi * xir * xic;
            }
        }
    }

    let ridge = 1e-6;
    for r in 0..dim {
        xtv_inv_x[r * dim + r] += ridge;
        for c in 0..r {
            let vrc = xtv_inv_x[r * dim + c];
            xtv_inv_x[c * dim + r] = vrc;
        }
    }

    if cholesky_inplace(&mut xtv_inv_x, dim).is_none() {
        return -1e8;
    }
    let log_det_xtv_inv_x = cholesky_logdet(&xtv_inv_x, dim);
    let beta = cholesky_solve(&xtv_inv_x, dim, &xtv_inv_y);

    let mut r_vec = vec![0.0_f64; n];
    for i in 0..n {
        let mut xb = 0.0;
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xb += xir * beta[r];
        }
        r_vec[i] = y[i] - xb;
    }

    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        rtv_invr += vinv[i] * r_vec[i] * r_vec[i];
    }

    let log_det_v: f64 = v.iter().map(|vv| vv.ln()).sum();
    let n_f = n as f64;
    let p_f = dim as f64;

    let total_log = (n_f - p_f) * (rtv_invr.ln()) + log_det_v + log_det_xtv_inv_x;
    let c = (n_f - p_f) * ((n_f - p_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let reml = c - 0.5 * total_log;

    if !reml.is_finite() {
        -1e8
    } else {
        reml
    }
}

fn ml_loglike(
    log10_lbd: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: Option<&[f64]>,
    n: usize,
    p_cov: usize,
) -> f64 {
    let use_snp = snp.is_some();
    let snp = snp.unwrap_or(&[]);
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return -1e8;
    }

    if use_snp && snp.len() != n {
        return -1e8;
    }
    let p = p_cov + if use_snp { 1 } else { 0 };
    if n <= p {
        return -1e8;
    }

    let mut v = vec![0.0_f64; n];
    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return -1e8;
        }
        v[i] = vv;
        vinv[i] = 1.0 / vv;
    }

    let dim = p;
    let mut xtv_inv_x = vec![0.0_f64; dim * dim];
    let mut xtv_inv_y = vec![0.0_f64; dim];

    for i in 0..n {
        let vi = vinv[i];
        let yi = y[i];
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xtv_inv_y[r] += vi * xir * yi;

            for c in 0..=r {
                let xic = if c < p_cov {
                    xcov[i * p_cov + c]
                } else {
                    snp[i]
                };
                xtv_inv_x[r * dim + c] += vi * xir * xic;
            }
        }
    }

    let ridge = 1e-6;
    for r in 0..dim {
        xtv_inv_x[r * dim + r] += ridge;
        for c in 0..r {
            let vrc = xtv_inv_x[r * dim + c];
            xtv_inv_x[c * dim + r] = vrc;
        }
    }

    if cholesky_inplace(&mut xtv_inv_x, dim).is_none() {
        return -1e8;
    }
    let beta = cholesky_solve(&xtv_inv_x, dim, &xtv_inv_y);

    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        let mut xb = 0.0;
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xb += xir * beta[r];
        }
        let ri = y[i] - xb;
        rtv_invr += vinv[i] * ri * ri;
    }

    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return -1e8;
    }

    let log_det_v: f64 = v.iter().map(|vv| vv.ln()).sum();
    let n_f = n as f64;

    let total_log = n_f * (rtv_invr.ln()) + log_det_v;
    let c = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let ml = c - 0.5 * total_log;

    if !ml.is_finite() {
        -1e8
    } else {
        ml
    }
}

fn final_beta_se(
    log10_lbd: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: &[f64],
    n: usize,
    p_cov: usize,
) -> (f64, f64, f64) {
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let p = p_cov + 1;
    if n <= p {
        return (f64::NAN, f64::NAN, lbd);
    }

    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return (f64::NAN, f64::NAN, lbd);
        }
        vinv[i] = 1.0 / vv;
    }

    let dim = p;
    let mut xtv_inv_x = vec![0.0_f64; dim * dim];
    let mut xtv_inv_y = vec![0.0_f64; dim];

    for i in 0..n {
        let vi = vinv[i];
        let yi = y[i];
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xtv_inv_y[r] += vi * xir * yi;

            for c in 0..=r {
                let xic = if c < p_cov {
                    xcov[i * p_cov + c]
                } else {
                    snp[i]
                };
                xtv_inv_x[r * dim + c] += vi * xir * xic;
            }
        }
    }

    let ridge = 1e-6;
    for r in 0..dim {
        xtv_inv_x[r * dim + r] += ridge;
        for c in 0..r {
            let vrc = xtv_inv_x[r * dim + c];
            xtv_inv_x[c * dim + r] = vrc;
        }
    }

    if cholesky_inplace(&mut xtv_inv_x, dim).is_none() {
        return (f64::NAN, f64::NAN, lbd);
    }
    let beta = cholesky_solve(&xtv_inv_x, dim, &xtv_inv_y);

    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        let mut xb = 0.0;
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xb += xir * beta[r];
        }
        let ri = y[i] - xb;
        rtv_invr += vinv[i] * ri * ri;
    }

    let n_f = n as f64;
    let p_f = p as f64;
    let sigma2 = rtv_invr / (n_f - p_f);

    let k = dim - 1;
    let mut e = vec![0.0_f64; dim];
    e[k] = 1.0;
    let x = cholesky_solve(&xtv_inv_x, dim, &e);
    let var_beta_k = sigma2 * x[k];
    if var_beta_k <= 0.0 || !var_beta_k.is_finite() {
        return (f64::NAN, f64::NAN, lbd);
    }

    (beta[k], var_beta_k.sqrt(), lbd)
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, max_iter=50, tol=1e-2))]
pub fn lmm_reml_null_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<(f64, f64, f64)> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let (best_log10_lbd, best_cost, ml) = py.detach(|| {
        let (best_log10_lbd, best_cost) = brent_minimize(
            |x| -reml_loglike(x, s, &xcov_flat, y, None, n, p_cov),
            low,
            high,
            tol,
            max_iter,
        );
        let ml = ml_loglike(best_log10_lbd, s, &xcov_flat, y, None, n, p_cov);
        (best_log10_lbd, best_cost, ml)
    });
    let reml = -best_cost;
    let lbd = 10.0_f64.powf(best_log10_lbd);
    Ok((lbd, ml, reml))
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd))]
pub fn ml_loglike_null_f32<'py>(
    _py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
) -> PyResult<f64> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let ml = ml_loglike(log10_lbd, s, &xcov_flat, y, None, n, p_cov);
    Ok(ml)
}

#[inline]
fn apply_p_diag_vec(
    xcov: &[f64],
    n: usize,
    p: usize,
    w: &[f64],
    c_inv: &[f64],
    v: &[f64],
    out: &mut [f64],
    tmp: &mut [f64],
    xtmp: &mut [f64],
    cxtmp: &mut [f64],
) {
    debug_assert_eq!(w.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(tmp.len(), n);
    debug_assert_eq!(xtmp.len(), p);
    debug_assert_eq!(cxtmp.len(), p);

    for i in 0..n {
        tmp[i] = w[i] * v[i];
    }
    for r in 0..p {
        let mut acc = 0.0_f64;
        for i in 0..n {
            acc += xcov[i * p + r] * tmp[i];
        }
        xtmp[r] = acc;
    }
    for r in 0..p {
        let mut acc = 0.0_f64;
        for c in 0..p {
            acc += c_inv[r * p + c] * xtmp[c];
        }
        cxtmp[r] = acc;
    }
    for i in 0..n {
        let mut xcu = 0.0_f64;
        for r in 0..p {
            xcu += xcov[i * p + r] * cxtmp[r];
        }
        out[i] = tmp[i] - w[i] * xcu;
    }
}

#[inline]
fn trace_p_d(
    s: &[f64],
    xcov: &[f64],
    n: usize,
    p: usize,
    w: &[f64],
    c_inv: &[f64],
    use_s_as_d: bool,
) -> f64 {
    let mut tr_wd = 0.0_f64;
    for i in 0..n {
        let d = if use_s_as_d { s[i] } else { 1.0 };
        tr_wd += w[i] * d;
    }

    // M = X^T diag(w^2 * d) X
    let mut m = vec![0.0_f64; p * p];
    for i in 0..n {
        let d = if use_s_as_d { s[i] } else { 1.0 };
        let wi2d = w[i] * w[i] * d;
        let base = i * p;
        for r in 0..p {
            let xir = xcov[base + r];
            for c in 0..=r {
                let xic = xcov[base + c];
                m[r * p + c] += wi2d * xir * xic;
            }
        }
    }
    for r in 0..p {
        for c in 0..r {
            m[c * p + r] = m[r * p + c];
        }
    }

    // tr(C^{-1} M) is not right; we already pass C = (X^T W X)^{-1}.
    // Need tr(C * M).
    let mut tr_cm = 0.0_f64;
    for r in 0..p {
        for c in 0..p {
            tr_cm += c_inv[r * p + c] * m[c * p + r];
        }
    }
    tr_wd - tr_cm
}

fn ai_reml_eval(
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    sigma_g2: f64,
    sigma_e2: f64,
) -> Option<(f64, f64, f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if !sigma_g2.is_finite() || !sigma_e2.is_finite() || sigma_g2 <= 0.0 || sigma_e2 <= 0.0 {
        return None;
    }
    if n <= p {
        return None;
    }

    let mut w = vec![0.0_f64; n];
    let mut log_det_v = 0.0_f64;
    for i in 0..n {
        let vi = sigma_g2 * s[i] + sigma_e2;
        if !vi.is_finite() || vi <= 0.0 {
            return None;
        }
        w[i] = 1.0 / vi;
        log_det_v += vi.ln();
    }

    // A = X^T W X, b = X^T W y
    let mut a = vec![0.0_f64; p * p];
    let mut b = vec![0.0_f64; p];
    for i in 0..n {
        let wi = w[i];
        let yi = y[i];
        let base = i * p;
        for r in 0..p {
            let xir = xcov[base + r];
            b[r] += wi * xir * yi;
            for c in 0..=r {
                let xic = xcov[base + c];
                a[r * p + c] += wi * xir * xic;
            }
        }
    }
    for r in 0..p {
        a[r * p + r] += 1e-8;
        for c in 0..r {
            a[c * p + r] = a[r * p + c];
        }
    }

    let mut l = a.clone();
    cholesky_inplace(&mut l, p)?;
    let log_det_xtv_inv_x = cholesky_logdet(&l, p);

    let beta = cholesky_solve(&l, p, &b);

    // z = V^{-1} r
    let mut z = vec![0.0_f64; n];
    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        let base = i * p;
        let mut xb = 0.0_f64;
        for r in 0..p {
            xb += xcov[base + r] * beta[r];
        }
        let ri = y[i] - xb;
        z[i] = w[i] * ri;
        rtv_invr += ri * z[i];
    }
    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return None;
    }

    // C = (X^T W X)^{-1}
    let mut c_inv = vec![0.0_f64; p * p];
    let mut e = vec![0.0_f64; p];
    let mut x = vec![0.0_f64; p];
    for col in 0..p {
        e.fill(0.0);
        e[col] = 1.0;
        x.fill(0.0);
        cholesky_solve_into(&l, p, &e, &mut x);
        for row in 0..p {
            c_inv[row * p + col] = x[row];
        }
    }

    let n_f = n as f64;
    let p_f = p as f64;
    let reml_total = (n_f - p_f) * rtv_invr.ln() + log_det_v + log_det_xtv_inv_x;
    let reml_c = (n_f - p_f) * ((n_f - p_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let reml = reml_c - 0.5 * reml_total;

    let ml_total = n_f * rtv_invr.ln() + log_det_v;
    let ml_c = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let ml = ml_c - 0.5 * ml_total;

    Some((reml, ml, rtv_invr, w, z, c_inv, beta))
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, max_iter=100, tol=1e-6, min_var=1e-12))]
pub fn ai_reml_null_f64<'py>(
    _py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
    min_var: f64,
) -> PyResult<(f64, f64, f64, f64, f64, usize, bool)> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if n <= p_cov {
        return Err(PyRuntimeError::new_err("n must be > p_cov"));
    }
    let min_var = if min_var.is_finite() && min_var > 0.0 {
        min_var
    } else {
        1e-12
    };
    let tol = if tol.is_finite() && tol > 0.0 {
        tol
    } else {
        1e-6
    };

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };

    let mean_y = y.iter().copied().sum::<f64>() / (n as f64);
    let mut var_y = 0.0_f64;
    for &yi in y {
        let d = yi - mean_y;
        var_y += d * d;
    }
    var_y /= (n.max(2) - 1) as f64;
    if !var_y.is_finite() || var_y <= 0.0 {
        var_y = 1.0;
    }
    let mut sigma_g2 = (0.5 * var_y).max(min_var);
    let mut sigma_e2 = (0.5 * var_y).max(min_var);

    let mut converged = false;
    let mut used_iter = 0usize;

    let mut state = ai_reml_eval(s, &xcov_flat, y, n, p_cov, sigma_g2, sigma_e2)
        .ok_or_else(|| PyRuntimeError::new_err("AIREML init failed"))?;

    for it in 0..max_iter {
        used_iter = it + 1;
        let (reml_curr, _ml_curr, _q_curr, w, z, c_inv, _beta) = &state;

        let tr_g = trace_p_d(s, &xcov_flat, n, p_cov, w, c_inv, true);
        let tr_e = trace_p_d(s, &xcov_flat, n, p_cov, w, c_inv, false);

        let mut q_g = 0.0_f64;
        let mut q_e = 0.0_f64;
        for i in 0..n {
            q_g += s[i] * z[i] * z[i];
            q_e += z[i] * z[i];
        }
        let score_g = -0.5 * (tr_g - q_g);
        let score_e = -0.5 * (tr_e - q_e);

        let mut dz_g = vec![0.0_f64; n];
        let mut dz_e = vec![0.0_f64; n];
        for i in 0..n {
            dz_g[i] = s[i] * z[i];
            dz_e[i] = z[i];
        }

        let mut p_dz_g = vec![0.0_f64; n];
        let mut p_dz_e = vec![0.0_f64; n];
        let mut tmp = vec![0.0_f64; n];
        let mut xtmp = vec![0.0_f64; p_cov];
        let mut cxtmp = vec![0.0_f64; p_cov];

        apply_p_diag_vec(
            &xcov_flat,
            n,
            p_cov,
            w,
            c_inv,
            &dz_g,
            &mut p_dz_g,
            &mut tmp,
            &mut xtmp,
            &mut cxtmp,
        );
        apply_p_diag_vec(
            &xcov_flat,
            n,
            p_cov,
            w,
            c_inv,
            &dz_e,
            &mut p_dz_e,
            &mut tmp,
            &mut xtmp,
            &mut cxtmp,
        );

        let mut ai_gg = 0.0_f64;
        let mut ai_ge = 0.0_f64;
        let mut ai_ee = 0.0_f64;
        for i in 0..n {
            ai_gg += dz_g[i] * p_dz_g[i];
            ai_ge += dz_g[i] * p_dz_e[i];
            ai_ee += dz_e[i] * p_dz_e[i];
        }
        ai_gg *= 0.5;
        ai_ge *= 0.5;
        ai_ee *= 0.5;

        let ridge = 1e-10;
        ai_gg += ridge;
        ai_ee += ridge;
        let det = ai_gg * ai_ee - ai_ge * ai_ge;
        if !det.is_finite() || det.abs() < 1e-18 {
            break;
        }
        let delta_g = (score_g * ai_ee - score_e * ai_ge) / det;
        let delta_e = (ai_gg * score_e - ai_ge * score_g) / det;
        if !delta_g.is_finite() || !delta_e.is_finite() {
            break;
        }

        let mut accepted = false;
        let mut step = 1.0_f64;
        let mut next_state = None;
        let mut next_sg = sigma_g2;
        let mut next_se = sigma_e2;

        for _ in 0..24 {
            let cand_sg = (sigma_g2 + step * delta_g).max(min_var);
            let cand_se = (sigma_e2 + step * delta_e).max(min_var);
            if let Some(st) = ai_reml_eval(s, &xcov_flat, y, n, p_cov, cand_sg, cand_se) {
                if st.0.is_finite() && st.0 >= *reml_curr - 1e-12 {
                    accepted = true;
                    next_state = Some(st);
                    next_sg = cand_sg;
                    next_se = cand_se;
                    break;
                }
            }
            step *= 0.5;
            if step < 1e-8 {
                break;
            }
        }

        if !accepted {
            break;
        }
        let rel_g = (next_sg - sigma_g2).abs() / sigma_g2.max(min_var);
        let rel_e = (next_se - sigma_e2).abs() / sigma_e2.max(min_var);
        sigma_g2 = next_sg;
        sigma_e2 = next_se;
        if let Some(st) = next_state {
            state = st;
        } else {
            break;
        }
        if rel_g.max(rel_e) < tol {
            converged = true;
            break;
        }
    }

    let (_reml, _ml, q, _w, _z, _c_inv, _beta) = &state;
    let n_f = n as f64;
    let p_f = p_cov as f64;
    let sigma_g2_out = (q / (n_f - p_f)).max(min_var);
    let sigma_e2_out = (sigma_e2 / sigma_g2).max(min_var) * sigma_g2_out;
    let lbd = (sigma_e2_out / sigma_g2_out).max(min_var);
    let reml = state.0;
    let ml = state.1;

    Ok((
        lbd,
        ml,
        reml,
        sigma_g2_out,
        sigma_e2_out,
        used_iter,
        converged,
    ))
}

#[inline]
fn trace_ab(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    let (nr, nc) = a.shape();
    debug_assert_eq!(b.shape(), (nr, nc));
    let mut s = 0.0_f64;
    for r in 0..nr {
        for c in 0..nc {
            s += a[(r, c)] * b[(c, r)];
        }
    }
    s
}

#[pyfunction]
#[pyo3(signature = (k_stack, xcov, y, max_iter=100, tol=1e-6, min_var=1e-12, trace_probes=8, trace_seed=42))]
pub fn ai_reml_multi_f64<'py>(
    py: Python<'py>,
    k_stack: PyReadonlyArray3<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
    min_var: f64,
    trace_probes: usize,
    trace_seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64, usize, bool)> {
    let kshape = k_stack.shape();
    if kshape.len() != 3 {
        return Err(PyRuntimeError::new_err("k_stack must be 3D (q, n, n)"));
    }
    let q = kshape[0];
    let n = kshape[1];
    if kshape[2] != n {
        return Err(PyRuntimeError::new_err("k_stack must be (q, n, n)"));
    }
    if q == 0 {
        return Err(PyRuntimeError::new_err(
            "k_stack requires at least one kernel",
        ));
    }

    let y_slice = y.as_slice()?;
    if y_slice.len() != n {
        return Err(PyRuntimeError::new_err("len(y) must equal n in k_stack"));
    }

    let x_arr = xcov.as_array();
    let (xn, p) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err(
            "xcov.n_rows must equal n in k_stack",
        ));
    }
    if p == 0 {
        return Err(PyRuntimeError::new_err(
            "xcov must have at least one column",
        ));
    }
    if n <= p {
        return Err(PyRuntimeError::new_err("n must be > p in xcov"));
    }

    let min_var = if min_var.is_finite() && min_var > 0.0 {
        min_var
    } else {
        1e-12
    };
    let tol = if tol.is_finite() && tol > 0.0 {
        tol
    } else {
        1e-6
    };

    let x_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let x_mat = DMatrix::<f64>::from_row_slice(n, p, &x_flat);
    let x_t = x_mat.transpose();
    let y_vec = DVector::<f64>::from_row_slice(y_slice);

    let mut kmats: Vec<DMatrix<f64>> = Vec::with_capacity(q);
    if let Ok(ks) = k_stack.as_slice() {
        for qi in 0..q {
            let off = qi * n * n;
            let mut km = DMatrix::<f64>::from_row_slice(n, n, &ks[off..off + n * n]);
            km = (&km + km.transpose()) * 0.5;
            kmats.push(km);
        }
    } else {
        let k_arr = k_stack.as_array();
        for qi in 0..q {
            let mut buf = vec![0.0_f64; n * n];
            for r in 0..n {
                for c in 0..n {
                    buf[r * n + c] = k_arr[(qi, r, c)];
                }
            }
            let mut km = DMatrix::<f64>::from_row_slice(n, n, &buf);
            km = (&km + km.transpose()) * 0.5;
            kmats.push(km);
        }
    }
    let eye = DMatrix::<f64>::identity(n, n);
    let use_exact_trace = trace_probes == 0 || n <= 768;
    let probes_n = trace_probes.max(1);
    let mut trace_probes_vec: Vec<DVector<f64>> = Vec::new();
    if !use_exact_trace {
        let mut rng = StdRng::seed_from_u64(trace_seed);
        trace_probes_vec.reserve(probes_n);
        for _ in 0..probes_n {
            let mut z = DVector::<f64>::zeros(n);
            for i in 0..n {
                z[i] = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 };
            }
            trace_probes_vec.push(z);
        }
    }

    let mut var_y = {
        let mean = y_slice.iter().copied().sum::<f64>() / (n as f64);
        let mut s2 = 0.0_f64;
        for &yi in y_slice {
            let d = yi - mean;
            s2 += d * d;
        }
        s2 / ((n.max(2) - 1) as f64)
    };
    if !var_y.is_finite() || var_y <= 0.0 {
        var_y = 1.0;
    }

    let m = q + 1; // q kernels + residual
    let mut theta = DVector::<f64>::from_element(m, (var_y / (m as f64)).max(min_var));
    let mut converged = false;
    let mut used_iter = 0usize;

    let mut last_ml = f64::NAN;
    let mut last_reml = f64::NAN;

    for it in 0..max_iter {
        used_iter = it + 1;

        // Build V = sum_j theta_j K_j + theta_e I
        let mut v = eye.clone() * theta[m - 1];
        for j in 0..q {
            v += &kmats[j] * theta[j];
        }
        let Some(chol_v) = v.clone().cholesky() else {
            break;
        };

        let vinv_x = chol_v.solve(&x_mat);
        let xt_vinv_x = &x_t * &vinv_x;
        let Some(chol_x) = xt_vinv_x.clone().cholesky() else {
            break;
        };
        let c_inv = chol_x.inverse();

        let vinv_y = chol_v.solve(&y_vec);
        let xt_vinv_y = &x_t * &vinv_y;
        let beta = chol_x.solve(&xt_vinv_y);

        let r_vec = &y_vec - &x_mat * beta;
        let vinv_r = chol_v.solve(&r_vec);
        let proj = &vinv_x * (&c_inv * (&x_t * &vinv_r));
        let alpha = &vinv_r - proj; // alpha = P y
        let qval = r_vec.dot(&vinv_r);
        if !qval.is_finite() || qval <= 0.0 {
            break;
        }

        let lv = chol_v.l();
        let mut log_det_v = 0.0_f64;
        for i in 0..n {
            log_det_v += 2.0 * lv[(i, i)].ln();
        }
        let lx = chol_x.l();
        let mut log_det_xt = 0.0_f64;
        for i in 0..p {
            log_det_xt += 2.0 * lx[(i, i)].ln();
        }
        let n_f = n as f64;
        let p_f = p as f64;
        let reml_total = (n_f - p_f) * qval.ln() + log_det_v + log_det_xt;
        let reml_c = (n_f - p_f) * ((n_f - p_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
        let reml_curr = reml_c - 0.5 * reml_total;
        let ml_total = n_f * qval.ln() + log_det_v;
        let ml_c = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
        let ml_curr = ml_c - 0.5 * ml_total;
        if !reml_curr.is_finite() || !ml_curr.is_finite() {
            break;
        }
        last_reml = reml_curr;
        last_ml = ml_curr;

        let mut score = DVector::<f64>::zeros(m);
        let mut ka_list: Vec<DVector<f64>> = Vec::with_capacity(m);
        let mut pka_list: Vec<DVector<f64>> = Vec::with_capacity(m);
        let b_t = vinv_x.transpose();
        let vinv_exact = if use_exact_trace {
            Some(chol_v.inverse())
        } else {
            None
        };
        let mut vinv_probe_vec: Vec<DVector<f64>> = Vec::new();
        if !use_exact_trace {
            vinv_probe_vec.reserve(probes_n);
            for z in &trace_probes_vec {
                vinv_probe_vec.push(chol_v.solve(z));
            }
        }

        for j in 0..m {
            let kj = if j < q { &kmats[j] } else { &eye };

            let ka = kj * &alpha;
            let quad = alpha.dot(&ka);
            ka_list.push(ka.clone());

            let tr1 = if let Some(vinv) = &vinv_exact {
                trace_ab(vinv, kj)
            } else {
                // Hutchinson trace estimate for tr(V^{-1} K_j) using cached V^{-1}z.
                let mut s = 0.0_f64;
                for (z, vinv_z) in trace_probes_vec.iter().zip(vinv_probe_vec.iter()) {
                    let kz = kj * z;
                    s += vinv_z.dot(&kz);
                }
                s / probes_n as f64
            };

            // Exact correction tr((X'V^{-1}X)^{-1} X'V^{-1}K_jV^{-1}X)
            // = tr(C^{-1} B'K_jB), where B = V^{-1}X.
            let kb = kj * &vinv_x;
            let bt_kb = &b_t * &kb;
            let tr2 = trace_ab(&c_inv, &bt_kb);
            let tr_pk = tr1 - tr2;
            score[j] = -0.5 * (tr_pk - quad);

            let vinv_ka = chol_v.solve(&ka);
            let pka = &vinv_ka - &vinv_x * (&c_inv * (&b_t * &ka));
            pka_list.push(pka);
        }

        let mut ai = DMatrix::<f64>::zeros(m, m);
        for a in 0..m {
            for b in 0..=a {
                let v_ab = 0.5 * ka_list[a].dot(&pka_list[b]);
                ai[(a, b)] = v_ab;
                ai[(b, a)] = v_ab;
            }
        }
        for d in 0..m {
            ai[(d, d)] += 1e-10;
        }

        let Some(delta) = ai.lu().solve(&score) else {
            break;
        };
        if !delta.iter().all(|v| v.is_finite()) {
            break;
        }

        let mut accepted = false;
        let mut step = 1.0_f64;
        let mut theta_next = theta.clone();
        let mut reml_next = reml_curr;
        let mut ml_next = ml_curr;

        for _ in 0..24 {
            let mut cand = theta.clone();
            for j in 0..m {
                cand[j] = (theta[j] + step * delta[j]).max(min_var);
            }

            let mut v_c = eye.clone() * cand[m - 1];
            for j in 0..q {
                v_c += &kmats[j] * cand[j];
            }
            let Some(chol_vc) = v_c.clone().cholesky() else {
                step *= 0.5;
                if step < 1e-8 {
                    break;
                }
                continue;
            };
            let vinv_xc = chol_vc.solve(&x_mat);
            let xt_vinv_xc = &x_t * &vinv_xc;
            let Some(chol_xc) = xt_vinv_xc.clone().cholesky() else {
                step *= 0.5;
                if step < 1e-8 {
                    break;
                }
                continue;
            };
            let vinv_yc = chol_vc.solve(&y_vec);
            let xt_vinv_yc = &x_t * &vinv_yc;
            let beta_c = chol_xc.solve(&xt_vinv_yc);
            let r_c = &y_vec - &x_mat * beta_c;
            let vinv_rc = chol_vc.solve(&r_c);
            let q_c = r_c.dot(&vinv_rc);
            if !q_c.is_finite() || q_c <= 0.0 {
                step *= 0.5;
                if step < 1e-8 {
                    break;
                }
                continue;
            }
            let lv_c = chol_vc.l();
            let mut log_det_vc = 0.0_f64;
            for i in 0..n {
                log_det_vc += 2.0 * lv_c[(i, i)].ln();
            }
            let lx_c = chol_xc.l();
            let mut log_det_xc = 0.0_f64;
            for i in 0..p {
                log_det_xc += 2.0 * lx_c[(i, i)].ln();
            }
            let reml_total_c = (n_f - p_f) * q_c.ln() + log_det_vc + log_det_xc;
            let reml_cand = reml_c - 0.5 * reml_total_c;
            let ml_total_c = n_f * q_c.ln() + log_det_vc;
            let ml_cand = ml_c - 0.5 * ml_total_c;
            if reml_cand.is_finite() && reml_cand >= reml_curr - 1e-12 {
                accepted = true;
                theta_next = cand;
                reml_next = reml_cand;
                ml_next = ml_cand;
                break;
            }
            step *= 0.5;
            if step < 1e-8 {
                break;
            }
        }

        if !accepted {
            break;
        }

        let mut rel_max = 0.0_f64;
        for j in 0..m {
            let rel = (theta_next[j] - theta[j]).abs() / theta[j].max(min_var);
            if rel > rel_max {
                rel_max = rel;
            }
        }
        theta = theta_next;
        last_reml = reml_next;
        last_ml = ml_next;

        if rel_max < tol {
            converged = true;
            break;
        }
    }

    let theta_arr = PyArray1::<f64>::from_vec(py, theta.iter().copied().collect()).into_bound();
    Ok((theta_arr, last_ml, last_reml, used_iter, converged))
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, g_rot_chunk, max_iter=50, tol=1e-2, threads=0, nullml=None))]
pub fn lmm_reml_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    g_rot_chunk: PyReadonlyArray2<'py, f32>,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let g_arr = g_rot_chunk.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("g_rot_chunk must be (m_chunk, n)"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let m_chunk = g_arr.shape()[0];
    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let g_slice_opt = g_rot_chunk.as_slice().ok();

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let beta_se_p = PyArray2::<f64>::zeros(py, [m_chunk, out_cols], false).into_bound();

    let beta_se_p_slice: &mut [f64] = unsafe {
        beta_se_p
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("beta_se_p not contiguous"))?
    };
    // For REML chunk scanning, rebuilding a local pool is consistently faster
    // than thread-local pool reuse in CLI LMM benchmarks.
    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    py.detach(|| {
        let mut run = || {
            beta_se_p_slice
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || vec![0.0_f64; n],
                    |snp_vec, (idx, out_row)| {
                        if let Some(gs) = g_slice_opt {
                            let row = &gs[idx * n..(idx + 1) * n];
                            for i in 0..n {
                                snp_vec[i] = row[i] as f64;
                            }
                        } else {
                            let row = g_arr.row(idx);
                            for i in 0..n {
                                snp_vec[i] = row[i] as f64;
                            }
                        }

                        let (best_log10_lbd, _best_cost) = brent_minimize(
                            |x| -reml_loglike(x, s, &xcov_flat, y, Some(&snp_vec[..]), n, p_cov),
                            low,
                            high,
                            tol,
                            max_iter,
                        );

                        let (beta, se, _lbd) =
                            final_beta_se(best_log10_lbd, s, &xcov_flat, y, &snp_vec, n, p_cov);

                        let p = if beta.is_finite() && se.is_finite() && se > 0.0 {
                            let z = beta / se;
                            (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta;
                        out_row[1] = se;
                        out_row[2] = if p.is_finite() { p } else { 1.0 };

                        if with_plrt {
                            let ml = ml_loglike(
                                best_log10_lbd,
                                s,
                                &xcov_flat,
                                y,
                                Some(&snp_vec[..]),
                                n,
                                p_cov,
                            );
                            out_row[3] = if ml.is_finite() {
                                let mut stat = 2.0 * (ml - nullml_val);
                                if !stat.is_finite() || stat < 0.0 {
                                    stat = 0.0;
                                }
                                chi2_sf_df1(stat)
                            } else {
                                1.0
                            };
                        }
                    },
                );
        };

        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    });

    Ok(beta_se_p)
}

#[inline]
fn rotate_snp_block_with_ut(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
) {
    if rows == 0 || n == 0 {
        return;
    }
    // C(rows, n) = A(rows, n) * U(n, n)
    // We store U^T in row-major (`u_t`). For matrixmultiply strides, expose U by
    // setting B row stride = 1 and col stride = n, i.e. B(i,j)=u_t[j*n + i].
    unsafe {
        sgemm(
            rows,
            n,
            n,
            1.0,
            snp_block.as_ptr(),
            n as isize,
            1,
            u_t.as_ptr(),
            1,
            n as isize,
            0.0,
            out_block.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
fn choose_rotate_tile_rows(rows: usize, thread_hint: usize) -> usize {
    if rows <= 1 {
        return 1;
    }
    let threads = thread_hint.max(1);
    // Avoid splitting when there is no real parallelism (or chunk is tiny):
    // one large SGEMM is much faster than many tiny calls in this case.
    if threads <= 1 || rows <= 32 {
        return rows;
    }
    // Keep enough independent tiles to feed the rayon pool, but avoid tiles
    // so small that SGEMM launch overhead dominates.
    let target_tasks = threads.saturating_mul(2).max(1);
    let mut tile_rows = (rows + target_tasks - 1) / target_tasks;
    tile_rows = tile_rows.clamp(16, 128);
    tile_rows.min(rows)
}

#[inline]
fn rotate_snp_block_with_ut_parallel(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    tile_rows: usize,
) {
    if rows == 0 || n == 0 {
        return;
    }

    let tile_rows = tile_rows.max(1).min(rows);
    if tile_rows >= rows {
        rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
        return;
    }

    out_block
        .par_chunks_mut(tile_rows * n)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * tile_rows;
            let rows_here = out_chunk.len() / n;
            let a_start = row_start * n;
            let a_end = a_start + rows_here * n;
            rotate_snp_block_with_ut(&snp_block[a_start..a_end], rows_here, n, u_t, out_chunk);
        });
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, snp_chunk, u_t, max_iter=50, tol=1e-2, threads=0, nullml=None, rotate_block_rows=256))]
pub fn lmm_reml_chunk_from_snp_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m_chunk, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let m_chunk = snp_arr.shape()[0];
    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let beta_se_p = PyArray2::<f64>::zeros(py, [m_chunk, out_cols], false).into_bound();
    let beta_se_p_slice: &mut [f64] = unsafe {
        beta_se_p
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("beta_se_p not contiguous"))?
    };

    // Keep same behavior as lmm_reml_chunk_f32: dedicated pool can be faster here.
    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    let block_rows = rotate_block_rows.max(1);

    py.detach(|| {
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            out_block
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || vec![0.0_f64; n],
                    |snp_vec, (idx, out_row)| {
                        let row = &g_block[idx * n..(idx + 1) * n];
                        for i in 0..n {
                            snp_vec[i] = row[i] as f64;
                        }

                        let (best_log10_lbd, _best_cost) = brent_minimize(
                            |x| -reml_loglike(x, s, &xcov_flat, y, Some(&snp_vec[..]), n, p_cov),
                            low,
                            high,
                            tol,
                            max_iter,
                        );

                        let (beta, se, _lbd) =
                            final_beta_se(best_log10_lbd, s, &xcov_flat, y, &snp_vec, n, p_cov);

                        let p = if beta.is_finite() && se.is_finite() && se > 0.0 {
                            let z = beta / se;
                            (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta;
                        out_row[1] = se;
                        out_row[2] = if p.is_finite() { p } else { 1.0 };

                        if with_plrt {
                            let ml = ml_loglike(
                                best_log10_lbd,
                                s,
                                &xcov_flat,
                                y,
                                Some(&snp_vec[..]),
                                n,
                                p_cov,
                            );
                            out_row[3] = if ml.is_finite() {
                                let mut stat = 2.0 * (ml - nullml_val);
                                if !stat.is_finite() || stat < 0.0 {
                                    stat = 0.0;
                                }
                                chi2_sf_df1(stat)
                            } else {
                                1.0
                            };
                        }
                    },
                );
        };

        let mut start = 0usize;
        while start < m_chunk {
            let rows = (m_chunk - start).min(block_rows);
            let snp_block = &snp_flat[start * n..(start + rows) * n];
            let rot_block = &mut rot_buf[..rows * n];
            let rotate_tile_rows = choose_rotate_tile_rows(
                rows,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                },
            );

            let out_block = &mut beta_se_p_slice[start * out_cols..(start + rows) * out_cols];
            if let Some(tp) = &pool {
                tp.install(|| {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                    run_rows(rot_block, out_block);
                });
            } else {
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    rotate_tile_rows,
                );
                run_rows(rot_block, out_block);
            }
            start += rows;
        }
    });

    Ok(beta_se_p)
}

// ------------------------------------------------------------
// Helpers: dot loops
// ------------------------------------------------------------

#[inline]
fn dot_loop(a: &[f64], b: &[f64]) -> f64 {
    dot_loop_simd(a, b)
}

#[inline]
fn dot_loop_unrolled(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut i = 0usize;
    while i + 3 < n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = s0 + s1 + s2 + s3;
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_loop_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0usize;
    let mut acc: __m256d = _mm256_setzero_pd();
    while i + 4 <= n {
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };
        acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
        i += 4;
    }
    let mut lanes = [0.0_f64; 4];
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    let mut s = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_loop_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0usize;
    let mut acc = vdupq_n_f64(0.0);
    while i + 2 <= n {
        let va = unsafe { vld1q_f64(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_f64(b.as_ptr().add(i)) };
        acc = vaddq_f64(acc, vmulq_f64(va, vb));
        i += 2;
    }
    let mut lanes = [0.0_f64; 2];
    unsafe { vst1q_f64(lanes.as_mut_ptr(), acc) };
    let mut s = lanes[0] + lanes[1];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[inline]
fn dot_loop_simd(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_loop_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_loop_neon(a, b) };
    }
    #[allow(unreachable_code)]
    dot_loop_unrolled(a, b)
}

struct AssocScratch {
    c: Vec<f64>,       // len p
    a_inv_c: Vec<f64>, // len p
}
impl AssocScratch {
    fn new(p: usize) -> Self {
        Self {
            c: vec![0.0; p],
            a_inv_c: vec![0.0; p],
        }
    }
    #[inline]
    fn reset(&mut self) {
        self.c.fill(0.0);
    }
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd, g_rot_chunk, threads=0, nullml=None))]
pub fn lmm_assoc_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    g_rot_chunk: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let g_arr = g_rot_chunk.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("g_rot_chunk must be (m, n)"));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let m = g_arr.shape()[0];
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound(); // beta, se, p, (plrt)
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    // Use contiguous slice directly (no copy)
    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let p = p_cov;

    // Build weights W = V^{-1} = 1/(s + lbd) (store as f32 to reduce bandwidth)
    let mut w = vec![0.0_f32; n];
    let mut log_det_v = 0.0_f64;
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return Err(PyRuntimeError::new_err("non-positive s[i]+lbd"));
        }
        w[i] = (1.0 / vv) as f32;
        log_det_v += vv.ln();
    }

    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    // Precompute A = X'WX, b = X'Wy, yWy
    let mut a = vec![0.0_f64; p * p];
    let mut b = vec![0.0_f64; p];
    let mut ywy = 0.0_f64;

    for i in 0..n {
        let wi = w[i] as f64;
        let yi = y[i];
        ywy += wi * yi * yi;

        let base = i * p;
        for r in 0..p {
            let xir = xcov_slice[base + r];
            b[r] += wi * xir * yi;
            for c in 0..=r {
                let xic = xcov_slice[base + c];
                a[r * p + c] += wi * xir * xic;
            }
        }
    }

    // symmetrize + ridge
    let ridge = 1e-6;
    for r in 0..p {
        a[r * p + r] += ridge;
        for c in 0..r {
            let vrc = a[r * p + c];
            a[c * p + r] = vrc;
        }
    }

    // Cholesky(A) in-place; now a stores L
    if cholesky_inplace(&mut a, p).is_none() {
        return Err(PyRuntimeError::new_err("X'WX not SPD"));
    }

    // Solve A^{-1} b once (no-alloc version into tmp)
    let mut a_inv_b = vec![0.0_f64; p];
    cholesky_solve_into(&a, p, &b, &mut a_inv_b);

    // b'A^{-1}b is constant
    let b_aib = dot_loop(&b, &a_inv_b);

    let df = (n as i32) - (p as i32) - 1;
    if df <= 0 {
        return Err(PyRuntimeError::new_err("df <= 0"));
    }

    // Thread pool
    let pool = get_cached_pool(threads)?;

    py.detach(|| {
        let mut run = || {
            out_slice
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || AssocScratch::new(p),
                    |scr, (idx, out_row)| {
                        scr.reset();

                        // c = X'Wg , d = g'Wg, e = g'Wy
                        let mut d = 0.0_f64;
                        let mut e = 0.0_f64;

                        for i in 0..n {
                            let gi = g_arr[(idx, i)] as f64;
                            let wi = w[i] as f64;
                            let yi = y[i];
                            let base = i * p;

                            let wgi = wi * gi;
                            d += wgi * gi;
                            e += wgi * yi;

                            // c[r] += wgi * xcov[i, r]
                            for r in 0..p {
                                scr.c[r] += wgi * xcov_slice[base + r];
                            }
                        }

                        // a_inv_c = A^{-1} c (no alloc)
                        cholesky_solve_into(&a, p, &scr.c, &mut scr.a_inv_c);

                        // ct_aic = c' A^{-1} c
                        let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                        let schur = d - ct_aic;

                        if schur <= 1e-12 || !schur.is_finite() {
                            out_row[0] = f64::NAN;
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        // ct_aib = c' A^{-1} b, computed in one loop without dot().
                        let mut ct_aib = 0.0_f64;
                        for r in 0..p {
                            ct_aib += scr.c[r] * a_inv_b[r];
                        }

                        let num = e - ct_aib;
                        let beta_g = num / schur;

                        // rWr = yWy - [ b'A^{-1}b + (e - c'A^{-1}b)^2 / schur ]
                        let q = b_aib + (num * num) / schur;
                        let rwr = (ywy - q).max(0.0);
                        let sigma2 = rwr / (df as f64);

                        let se_g = (sigma2 / schur).sqrt();
                        let pval = if se_g.is_finite() && se_g > 0.0 && beta_g.is_finite() {
                            let z = (beta_g / se_g).abs();
                            (2.0 * normal_sf(z)).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta_g;
                        out_row[1] = se_g;
                        out_row[2] = pval;
                        if with_plrt {
                            let ml = if rwr > 0.0 && rwr.is_finite() {
                                let total_log = n_f * rwr.ln() + log_det_v;
                                c_ml - 0.5 * total_log
                            } else {
                                f64::NAN
                            };
                            let mut stat = if ml.is_finite() {
                                2.0 * (ml - nullml_val)
                            } else {
                                0.0
                            };
                            if !stat.is_finite() || stat < 0.0 {
                                stat = 0.0;
                            }
                            out_row[3] = chi2_sf_df1(stat);
                        }
                    },
                );
        };

        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    });

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd, snp_chunk, u_t, threads=0, nullml=None, rotate_block_rows=512))]
pub fn lmm_assoc_chunk_from_snp_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let m = snp_arr.shape()[0];
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };
    let p = p_cov;

    // Build weights W = V^{-1} = 1/(s + lbd) (store as f32 to reduce bandwidth)
    let mut w = vec![0.0_f32; n];
    let mut log_det_v = 0.0_f64;
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return Err(PyRuntimeError::new_err("non-positive s[i]+lbd"));
        }
        w[i] = (1.0 / vv) as f32;
        log_det_v += vv.ln();
    }

    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    // Precompute A = X'WX, b = X'Wy, yWy
    let mut a = vec![0.0_f64; p * p];
    let mut b = vec![0.0_f64; p];
    let mut ywy = 0.0_f64;

    for i in 0..n {
        let wi = w[i] as f64;
        let yi = y[i];
        ywy += wi * yi * yi;

        let base = i * p;
        for r in 0..p {
            let xir = xcov_slice[base + r];
            b[r] += wi * xir * yi;
            for c in 0..=r {
                let xic = xcov_slice[base + c];
                a[r * p + c] += wi * xir * xic;
            }
        }
    }

    // symmetrize + ridge
    let ridge = 1e-6;
    for r in 0..p {
        a[r * p + r] += ridge;
        for c in 0..r {
            let vrc = a[r * p + c];
            a[c * p + r] = vrc;
        }
    }

    // Cholesky(A) in-place; now a stores L
    if cholesky_inplace(&mut a, p).is_none() {
        return Err(PyRuntimeError::new_err("X'WX not SPD"));
    }

    // Solve A^{-1} b once (no-alloc version into tmp)
    let mut a_inv_b = vec![0.0_f64; p];
    cholesky_solve_into(&a, p, &b, &mut a_inv_b);

    // b'A^{-1}b is constant
    let b_aib = dot_loop(&b, &a_inv_b);

    let df = (n as i32) - (p as i32) - 1;
    if df <= 0 {
        return Err(PyRuntimeError::new_err("df <= 0"));
    }

    let pool = get_cached_pool(threads)?;
    let block_rows = rotate_block_rows.max(1);

    py.detach(|| {
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            out_block
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || AssocScratch::new(p),
                    |scr, (idx, out_row)| {
                        scr.reset();
                        let row = &g_block[idx * n..(idx + 1) * n];

                        // c = X'Wg , d = g'Wg, e = g'Wy
                        let mut d = 0.0_f64;
                        let mut e = 0.0_f64;

                        for i in 0..n {
                            let gi = row[i] as f64;
                            let wi = w[i] as f64;
                            let yi = y[i];
                            let base = i * p;

                            let wgi = wi * gi;
                            d += wgi * gi;
                            e += wgi * yi;

                            for r in 0..p {
                                scr.c[r] += wgi * xcov_slice[base + r];
                            }
                        }

                        cholesky_solve_into(&a, p, &scr.c, &mut scr.a_inv_c);
                        let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                        let schur = d - ct_aic;

                        if schur <= 1e-12 || !schur.is_finite() {
                            out_row[0] = f64::NAN;
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        let mut ct_aib = 0.0_f64;
                        for r in 0..p {
                            ct_aib += scr.c[r] * a_inv_b[r];
                        }

                        let num = e - ct_aib;
                        let beta_g = num / schur;

                        let q = b_aib + (num * num) / schur;
                        let rwr = (ywy - q).max(0.0);
                        let sigma2 = rwr / (df as f64);

                        let se_g = (sigma2 / schur).sqrt();
                        let pval = if se_g.is_finite() && se_g > 0.0 && beta_g.is_finite() {
                            let z = (beta_g / se_g).abs();
                            (2.0 * normal_sf(z)).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta_g;
                        out_row[1] = se_g;
                        out_row[2] = pval;
                        if with_plrt {
                            let ml = if rwr > 0.0 && rwr.is_finite() {
                                let total_log = n_f * rwr.ln() + log_det_v;
                                c_ml - 0.5 * total_log
                            } else {
                                f64::NAN
                            };
                            let mut stat = if ml.is_finite() {
                                2.0 * (ml - nullml_val)
                            } else {
                                0.0
                            };
                            if !stat.is_finite() || stat < 0.0 {
                                stat = 0.0;
                            }
                            out_row[3] = chi2_sf_df1(stat);
                        }
                    },
                );
        };

        let mut start = 0usize;
        while start < m {
            let rows = (m - start).min(block_rows);
            let snp_block = &snp_flat[start * n..(start + rows) * n];
            let rot_block = &mut rot_buf[..rows * n];
            let rotate_tile_rows = choose_rotate_tile_rows(
                rows,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                },
            );

            let out_block = &mut out_slice[start * out_cols..(start + rows) * out_cols];
            if let Some(tp) = &pool {
                tp.install(|| {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                    run_rows(rot_block, out_block);
                });
            } else {
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    rotate_tile_rows,
                );
                run_rows(rot_block, out_block);
            }
            start += rows;
        }
    });

    Ok(out)
}

// =============================================================================
// FaST-LMM fixed-lambda association (fastlmm_assoc_chunk_f32)
// =============================================================================

struct FastlmmAssocScratch {
    xtv_inv_x: Vec<f64>,
    xtv_inv_y: Vec<f64>,
    beta: Vec<f64>,
    rhs: Vec<f64>,
    work: Vec<f64>,
    u1_xtsnp: Vec<f64>,
    u2_xtsnp: Vec<f64>,
}

impl FastlmmAssocScratch {
    fn new(p: usize) -> Self {
        let dim = p + 1;
        Self {
            xtv_inv_x: vec![0.0; dim * dim],
            xtv_inv_y: vec![0.0; dim],
            beta: vec![0.0; dim],
            rhs: vec![0.0; dim],
            work: vec![0.0; dim],
            u1_xtsnp: vec![0.0; p],
            u2_xtsnp: vec![0.0; p],
        }
    }
}

fn precompute_u2_base_fastlmm(
    u2tx: &[f64],
    u2ty: &[f64],
    n: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut u2_xtx = vec![0.0_f64; p * p];
    let mut u2_xty = vec![0.0_f64; p];

    for i in 0..n {
        let base = i * p;
        let yi = u2ty[i];
        for r in 0..p {
            let xir = u2tx[base + r];
            u2_xty[r] += xir * yi;
            for c in 0..=r {
                u2_xtx[r * p + c] += xir * u2tx[base + c];
            }
        }
    }

    for r in 0..p {
        for c in 0..r {
            let vrc = u2_xtx[r * p + c];
            u2_xtx[c * p + r] = vrc;
        }
    }

    (u2_xtx, u2_xty)
}

#[pyfunction]
#[pyo3(signature = (s, u1tx, u2tx, u1ty, u2ty, log10_lbd, u1tsnp_chunk, u2tsnp_chunk, threads=0, nullml=None))]
pub fn fastlmm_assoc_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    u1tx: PyReadonlyArray2<'py, f64>,
    u2tx: PyReadonlyArray2<'py, f64>,
    u1ty: PyReadonlyArray1<'py, f64>,
    u2ty: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    u1tsnp_chunk: PyReadonlyArray2<'py, f32>,
    u2tsnp_chunk: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let u1ty = u1ty.as_slice()?;
    let u2ty = u2ty.as_slice()?;

    let u1tx_arr = u1tx.as_array();
    let u2tx_arr = u2tx.as_array();
    let u1tsnp_arr = u1tsnp_chunk.as_array();
    let u2tsnp_arr = u2tsnp_chunk.as_array();

    let k = s.len();
    if k == 0 {
        return Err(PyRuntimeError::new_err("empty s"));
    }
    let (k1, p) = (u1tx_arr.shape()[0], u1tx_arr.shape()[1]);
    if k1 != k {
        return Err(PyRuntimeError::new_err("u1tx rows must equal len(s)"));
    }
    if u1ty.len() != k {
        return Err(PyRuntimeError::new_err("u1ty len must equal len(s)"));
    }
    let (n, p2) = (u2tx_arr.shape()[0], u2tx_arr.shape()[1]);
    if p2 != p {
        return Err(PyRuntimeError::new_err(
            "u1tx/u2tx must have same column count",
        ));
    }
    if u2ty.len() != n {
        return Err(PyRuntimeError::new_err("u2ty len must equal u2tx rows"));
    }

    let (m1, k2) = (u1tsnp_arr.shape()[0], u1tsnp_arr.shape()[1]);
    let (m2, n2) = (u2tsnp_arr.shape()[0], u2tsnp_arr.shape()[1]);
    if k2 != k {
        return Err(PyRuntimeError::new_err("u1tsnp_chunk must be (m, k)"));
    }
    if n2 != n {
        return Err(PyRuntimeError::new_err("u2tsnp_chunk must be (m, n)"));
    }
    if m1 != m2 {
        return Err(PyRuntimeError::new_err(
            "u1tsnp_chunk and u2tsnp_chunk must have same row count",
        ));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err("n must be > p+1"));
    }

    let u1tx_slice: &[f64] = u1tx
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u1tx must be contiguous (C-order)"))?;
    let u2tx_slice: &[f64] = u2tx
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u2tx must be contiguous (C-order)"))?;
    let u1tsnp_slice: &[f32] = u1tsnp_chunk
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u1tsnp_chunk must be contiguous (C-order)"))?;
    let u2tsnp_slice: &[f32] = u2tsnp_chunk
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u2tsnp_chunk must be contiguous (C-order)"))?;

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }
    let v2_inv = 1.0 / lbd;

    let mut v1_inv = vec![0.0_f64; k];
    for i in 0..k {
        let v1 = s[i] + lbd;
        if v1 <= 0.0 {
            return Err(PyRuntimeError::new_err("non-positive s[i]+lbd"));
        }
        v1_inv[i] = 1.0 / v1;
    }

    let (u2_xtx, u2_xty) = precompute_u2_base_fastlmm(u2tx_slice, u2ty, n, p);

    let mut base_a = vec![0.0_f64; p * p];
    for r in 0..p {
        for c in 0..=r {
            base_a[r * p + c] = v2_inv * u2_xtx[r * p + c];
        }
    }

    for i in 0..k {
        let vi = v1_inv[i];
        let base = i * p;
        for r in 0..p {
            let xir = u1tx_slice[base + r];
            for c in 0..=r {
                base_a[r * p + c] += vi * xir * u1tx_slice[base + c];
            }
        }
    }

    for r in 0..p {
        for c in 0..r {
            let vrc = base_a[r * p + c];
            base_a[c * p + r] = vrc;
        }
    }

    let mut base_b = vec![0.0_f64; p];
    for r in 0..p {
        base_b[r] = v2_inv * u2_xty[r];
    }
    for i in 0..k {
        let vi = v1_inv[i];
        let yi = u1ty[i];
        let base = i * p;
        for r in 0..p {
            base_b[r] += vi * u1tx_slice[base + r] * yi;
        }
    }

    let log_det_v: f64 =
        s.iter().map(|v| (v + lbd).ln()).sum::<f64>() + ((n - k) as f64) * lbd.ln();
    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    let m = m1;
    let out_cols = if with_plrt { 4 } else { 3 };
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let df = (n as isize) - (p as isize) - 1;
    let df_f = df as f64;

    let pool = get_cached_pool(threads)?;

    py.detach(|| {
        let mut run = || {
            out_slice
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || FastlmmAssocScratch::new(p),
                    |scr, (idx, out_row)| {
                        scr.u1_xtsnp.fill(0.0);
                        scr.u2_xtsnp.fill(0.0);

                        let u1_row = &u1tsnp_slice[idx * k..(idx + 1) * k];
                        let u2_row = &u2tsnp_slice[idx * n..(idx + 1) * n];

                        let mut u1_snp_snp = 0.0_f64;
                        let mut u1_snp_ty = 0.0_f64;
                        for i in 0..k {
                            let gi = u1_row[i] as f64;
                            let vi = v1_inv[i];
                            u1_snp_snp += vi * gi * gi;
                            u1_snp_ty += vi * gi * u1ty[i];
                            let base = i * p;
                            for r in 0..p {
                                scr.u1_xtsnp[r] += vi * u1tx_slice[base + r] * gi;
                            }
                        }

                        let mut u2_snp_snp = 0.0_f64;
                        let mut u2_snp_ty = 0.0_f64;
                        for i in 0..n {
                            let gi = u2_row[i] as f64;
                            u2_snp_snp += gi * gi;
                            u2_snp_ty += gi * u2ty[i];
                            let base = i * p;
                            for r in 0..p {
                                scr.u2_xtsnp[r] += u2tx_slice[base + r] * gi;
                            }
                        }

                        let dim = p + 1;
                        scr.xtv_inv_x[..p * p].copy_from_slice(&base_a);
                        scr.xtv_inv_y[..p].copy_from_slice(&base_b);

                        for r in 0..p {
                            let c = scr.u1_xtsnp[r] + v2_inv * scr.u2_xtsnp[r];
                            scr.xtv_inv_x[p * dim + r] = c;
                            scr.xtv_inv_x[r * dim + p] = c;
                        }
                        scr.xtv_inv_x[p * dim + p] = u1_snp_snp + v2_inv * u2_snp_snp;
                        scr.xtv_inv_y[p] = u1_snp_ty + v2_inv * u2_snp_ty;

                        let ridge = 1e-6;
                        for r in 0..dim {
                            scr.xtv_inv_x[r * dim + r] += ridge;
                        }

                        if cholesky_inplace(&mut scr.xtv_inv_x, dim).is_none() {
                            out_row[0] = f64::NAN;
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        cholesky_solve_into(&scr.xtv_inv_x, dim, &scr.xtv_inv_y, &mut scr.beta);

                        let mut r1_sum = 0.0_f64;
                        for i in 0..k {
                            let mut xb = 0.0_f64;
                            let base = i * p;
                            for r in 0..p {
                                xb += u1tx_slice[base + r] * scr.beta[r];
                            }
                            xb += (u1_row[i] as f64) * scr.beta[p];
                            let ri = u1ty[i] - xb;
                            r1_sum += v1_inv[i] * ri * ri;
                        }

                        let mut r2_sum = 0.0_f64;
                        for i in 0..n {
                            let mut xb = 0.0_f64;
                            let base = i * p;
                            for r in 0..p {
                                xb += u2tx_slice[base + r] * scr.beta[r];
                            }
                            xb += (u2_row[i] as f64) * scr.beta[p];
                            let ri = u2ty[i] - xb;
                            r2_sum += ri * ri;
                        }

                        let rtv_invr = r1_sum + v2_inv * r2_sum;
                        if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
                            out_row[0] = scr.beta[p];
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        let sigma2 = rtv_invr / df_f;
                        if !sigma2.is_finite() || sigma2 <= 0.0 {
                            out_row[0] = scr.beta[p];
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        scr.rhs.fill(0.0);
                        scr.rhs[p] = 1.0;
                        cholesky_solve_into(&scr.xtv_inv_x, dim, &scr.rhs, &mut scr.work);
                        let var_beta = sigma2 * scr.work[p];
                        let se = if var_beta.is_finite() && var_beta > 0.0 {
                            var_beta.sqrt()
                        } else {
                            f64::NAN
                        };

                        let beta = scr.beta[p];
                        let pval = if beta.is_finite() && se.is_finite() && se > 0.0 {
                            let z = (beta / se).abs();
                            (2.0 * normal_sf(z)).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta;
                        out_row[1] = se;
                        out_row[2] = pval;
                        if with_plrt {
                            let ml = if rtv_invr.is_finite() && rtv_invr > 0.0 {
                                let total_log = n_f * rtv_invr.ln() + log_det_v;
                                c_ml - 0.5 * total_log
                            } else {
                                f64::NAN
                            };
                            let mut stat = if ml.is_finite() {
                                2.0 * (ml - nullml_val)
                            } else {
                                0.0
                            };
                            if !stat.is_finite() || stat < 0.0 {
                                stat = 0.0;
                            }
                            out_row[3] = chi2_sf_df1(stat);
                        }
                    },
                );
        };

        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    });

    Ok(out)
}

#[derive(Clone, Copy)]
enum PackedGeneticModel {
    Add,
    Dom,
    Rec,
    Het,
}

impl PackedGeneticModel {
    fn parse(text: &str) -> PyResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "add" => Ok(Self::Add),
            "dom" => Ok(Self::Dom),
            "rec" => Ok(Self::Rec),
            "het" => Ok(Self::Het),
            _ => Err(PyRuntimeError::new_err(
                "model must be one of: add, dom, rec, het",
            )),
        }
    }

    #[inline]
    fn apply(self, g: f64) -> f64 {
        match self {
            Self::Add => g,
            Self::Dom => {
                if g > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Rec => {
                if (g - 2.0).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Het => {
                if (g - 1.0).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

struct PackedNullEval {
    lbd: f64,
    ml: f64,
    reml: f64,
    log_det_v: f64,
    v1_inv: Vec<f64>,
    base_a: Vec<f64>,
    base_b: Vec<f64>,
}

fn fastlmm_null_eval_packed(
    log10_lbd: f64,
    tau: f64,
    s: &[f64],
    u1tx: &[f64],
    u2tx: &[f64],
    u1ty: &[f64],
    u2ty: &[f64],
    u2_xtx: &[f64],
    u2_xty: &[f64],
    k: usize,
    n: usize,
    p: usize,
    c_reml: f64,
    c_ml: f64,
) -> Option<PackedNullEval> {
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return None;
    }
    let v2_denom = lbd + tau;
    if !v2_denom.is_finite() || v2_denom <= 0.0 {
        return None;
    }
    let v2_inv = 1.0 / v2_denom;

    let mut v1_inv = vec![0.0_f64; k];
    let mut log_det_v = 0.0_f64;
    for i in 0..k {
        let v1 = s[i] + v2_denom;
        if !v1.is_finite() || v1 <= 0.0 {
            return None;
        }
        v1_inv[i] = 1.0 / v1;
        log_det_v += v1.ln();
    }
    log_det_v += ((n - k) as f64) * v2_denom.ln();

    let mut base_a = vec![0.0_f64; p * p];
    for r in 0..p {
        for c in 0..=r {
            base_a[r * p + c] = v2_inv * u2_xtx[r * p + c];
        }
    }
    for i in 0..k {
        let vi = v1_inv[i];
        let base = i * p;
        for r in 0..p {
            let xir = u1tx[base + r];
            for c in 0..=r {
                base_a[r * p + c] += vi * xir * u1tx[base + c];
            }
        }
    }
    for r in 0..p {
        for c in 0..r {
            let vrc = base_a[r * p + c];
            base_a[c * p + r] = vrc;
        }
    }

    let mut base_b = vec![0.0_f64; p];
    for r in 0..p {
        base_b[r] = v2_inv * u2_xty[r];
    }
    for i in 0..k {
        let vi = v1_inv[i];
        let yi = u1ty[i];
        let base = i * p;
        for r in 0..p {
            base_b[r] += vi * u1tx[base + r] * yi;
        }
    }

    let mut chol_a = base_a.clone();
    let ridge = 1e-6_f64;
    for i in 0..p {
        chol_a[i * p + i] += ridge;
    }
    if cholesky_inplace(&mut chol_a, p).is_none() {
        return None;
    }
    let log_det_xtv = cholesky_logdet(&chol_a, p);

    let mut beta = vec![0.0_f64; p];
    cholesky_solve_into(&chol_a, p, &base_b, &mut beta);

    let mut r1_sum = 0.0_f64;
    for i in 0..k {
        let mut xb = 0.0_f64;
        let base = i * p;
        for r in 0..p {
            xb += u1tx[base + r] * beta[r];
        }
        let ri = u1ty[i] - xb;
        r1_sum += v1_inv[i] * ri * ri;
    }

    let mut r2_sum = 0.0_f64;
    for i in 0..n {
        let mut xb = 0.0_f64;
        let base = i * p;
        for r in 0..p {
            xb += u2tx[base + r] * beta[r];
        }
        let ri = u2ty[i] - xb;
        r2_sum += ri * ri;
    }

    let rtv_invr = r1_sum + v2_inv * r2_sum;
    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return None;
    }

    let n_f = n as f64;
    let n_minus_p_f = (n - p) as f64;
    let total_reml = n_minus_p_f * rtv_invr.ln() + log_det_v + log_det_xtv;
    let reml = c_reml - 0.5 * total_reml;

    let total_ml = n_f * rtv_invr.ln() + log_det_v;
    let ml = c_ml - 0.5 * total_ml;

    Some(PackedNullEval {
        lbd,
        ml,
        reml,
        log_det_v,
        v1_inv,
        base_a,
        base_b,
    })
}

struct PackedFastlmmScratch {
    g: Vec<f64>,
    u1: Vec<f64>,
    u2: Vec<f64>,
    xtv_inv_x: Vec<f64>,
    xtv_inv_y: Vec<f64>,
    beta: Vec<f64>,
    rhs: Vec<f64>,
    work: Vec<f64>,
    u1_xtsnp: Vec<f64>,
    u2_xtsnp: Vec<f64>,
}

impl PackedFastlmmScratch {
    fn new(n: usize, k: usize, p: usize) -> Self {
        let dim = p + 1;
        Self {
            g: vec![0.0; n],
            u1: vec![0.0; k],
            u2: vec![0.0; n],
            xtv_inv_x: vec![0.0; dim * dim],
            xtv_inv_y: vec![0.0; dim],
            beta: vec![0.0; dim],
            rhs: vec![0.0; dim],
            work: vec![0.0; dim],
            u1_xtsnp: vec![0.0; p],
            u2_xtsnp: vec![0.0; p],
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    u,
    s,
    y,
    x=None,
    sample_indices=None,
    low=-5.0,
    high=5.0,
    max_iter=50,
    tol=1e-2,
    tau=0.0,
    threads=0,
    model="add",
    progress_callback=None,
    progress_every=0
))]
pub fn fastlmm_assoc_packed_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    u: PyReadonlyArray2<'py, f32>,
    s: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    tau: f64,
    threads: usize,
    model: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(f64, f64, f64, Bound<'py, PyArray2<f64>>)> {
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }
    if !(tau.is_finite() && tau >= 0.0) {
        return Err(PyRuntimeError::new_err("tau must be finite and >= 0"));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let gm = PackedGeneticModel::parse(model)?;

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip.len()
        )));
    }
    if row_maf.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        )));
    }

    let y = y.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y must not be empty"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        if sidx_slice.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                sidx_slice.len()
            )));
        }
        parse_index_vec_i64(sidx_slice, n_samples, "sample_indices")?
    } else {
        if n != n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "len(y)={} must equal n_samples={} when sample_indices is not provided",
                n, n_samples
            )));
        }
        (0..n).collect()
    };

    let u_arr = u.as_array();
    if u_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("u must be 2D (n_samples, k)"));
    }
    let (u_n, k_full) = (u_arr.shape()[0], u_arr.shape()[1]);
    if u_n != n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "u row count mismatch: got {u_n}, expected {n_samples}"
        )));
    }
    if k_full == 0 {
        return Err(PyRuntimeError::new_err("u must have at least 1 component"));
    }

    let s_f32 = s.as_slice()?;
    if s_f32.len() != k_full {
        return Err(PyRuntimeError::new_err(format!(
            "s length mismatch: got {}, expected {k_full}",
            s_f32.len()
        )));
    }
    let k = k_full.min(n.saturating_sub(1));
    if k == 0 {
        return Err(PyRuntimeError::new_err(
            "effective rank is 0 after accounting for sample size",
        ));
    }
    let s_vec: Vec<f64> = s_f32[..k].iter().map(|&v| v as f64).collect();

    let p0 = match &x {
        Some(xarr) => {
            let xa = xarr.as_array();
            if xa.ndim() != 2 {
                return Err(PyRuntimeError::new_err("x must be 2D (n, p0)"));
            }
            let xr = xa.shape()[0];
            let xc = xa.shape()[1];
            if xr != n {
                return Err(PyRuntimeError::new_err(format!(
                    "x row count mismatch: got {xr}, expected {n}"
                )));
            }
            xc
        }
        None => 0usize,
    };
    let p = p0 + 1;
    if n <= p {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p for null model: n={n}, p={p}"
        )));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p+1 for SNP tests: n={n}, p={p}"
        )));
    }

    let u_slice: &[f32] = u
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("u must be contiguous (C-order)"))?;
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let x_slice_opt: Option<&[f64]> = match &x {
        Some(xarr) => Some(
            xarr.as_slice()
                .map_err(|_| PyRuntimeError::new_err("x must be contiguous (C-order)"))?,
        ),
        None => None,
    };

    let mut x_full = vec![0.0_f64; n * p];
    for i in 0..n {
        x_full[i * p] = 1.0;
    }
    if let Some(x_slice) = x_slice_opt {
        for i in 0..n {
            let src = &x_slice[i * p0..(i + 1) * p0];
            let dst = &mut x_full[i * p + 1..(i + 1) * p];
            dst.copy_from_slice(src);
        }
    }

    let mut u_sub = vec![0.0_f32; n * k];
    for (i, &sid) in sample_idx.iter().enumerate() {
        let src = &u_slice[sid * k_full..sid * k_full + k];
        let dst = &mut u_sub[i * k..(i + 1) * k];
        dst.copy_from_slice(src);
    }

    let mut u1ty = vec![0.0_f64; k];
    let mut u1tx = vec![0.0_f64; k * p];
    for i in 0..n {
        let yi = y[i];
        let ub = i * k;
        let xb = i * p;
        for r in 0..k {
            let ur = u_sub[ub + r] as f64;
            u1ty[r] += ur * yi;
            for c in 0..p {
                u1tx[r * p + c] += ur * x_full[xb + c];
            }
        }
    }

    let mut u2ty = vec![0.0_f64; n];
    let mut u2tx = vec![0.0_f64; n * p];
    for i in 0..n {
        let ub = i * k;
        let xb = i * p;
        let mut y_proj = 0.0_f64;
        for r in 0..k {
            y_proj += (u_sub[ub + r] as f64) * u1ty[r];
        }
        u2ty[i] = y[i] - y_proj;

        for c in 0..p {
            let mut x_proj = 0.0_f64;
            for r in 0..k {
                x_proj += (u_sub[ub + r] as f64) * u1tx[r * p + c];
            }
            u2tx[xb + c] = x_full[xb + c] - x_proj;
        }
    }

    let (u2_xtx, u2_xty) = precompute_u2_base_fastlmm(&u2tx, &u2ty, n, p);

    let n_f = n as f64;
    let n_minus_p_f = (n - p) as f64;
    let c_reml = n_minus_p_f * (n_minus_p_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    let (best_log10, _best_cost) = brent_minimize(
        |x0| match fastlmm_null_eval_packed(
            x0, tau, &s_vec, &u1tx, &u2tx, &u1ty, &u2ty, &u2_xtx, &u2_xty, k, n, p, c_reml, c_ml,
        ) {
            Some(v) if v.reml.is_finite() => -v.reml,
            _ => 1e100,
        },
        low,
        high,
        tol,
        max_iter,
    );

    let best = fastlmm_null_eval_packed(
        best_log10, tau, &s_vec, &u1tx, &u2tx, &u1ty, &u2ty, &u2_xtx, &u2_xty, k, n, p, c_reml,
        c_ml,
    )
    .ok_or_else(|| PyRuntimeError::new_err("failed to evaluate null model at optimum"))?;

    let lbd = best.lbd;
    let ml0 = best.ml;
    let reml0 = best.reml;
    let log_det_v = best.log_det_v;
    let v1_inv = best.v1_inv;
    let base_a = best.base_a;
    let base_b = best.base_b;
    let v2_denom = lbd + tau;
    let v2_inv = 1.0 / v2_denom;

    let df = (n as isize) - (p as isize) - 1;
    let df_f = df as f64;
    if df <= 0 {
        return Err(PyRuntimeError::new_err("invalid df <= 0 in packed fastlmm"));
    }

    let out_cols = 4usize;
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let pool = get_cached_pool(threads)?;
    let progress_block = if progress_every == 0 {
        m.max(1)
    } else {
        progress_every.max(1)
    };

    py.detach(move || -> PyResult<()> {
        for row_start in (0..m).step_by(progress_block) {
            let row_end = (row_start + progress_block).min(m);
            {
                let out_blk = &mut out_slice[row_start * out_cols..row_end * out_cols];
                let mut run_block = || {
                    out_blk.par_chunks_mut(out_cols).enumerate().for_each_init(
                        || PackedFastlmmScratch::new(n, k, p),
                        |scr, (off, out_row)| {
                            let idx = row_start + off;
                            let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                            let flip = row_flip[idx];
                            let mean_g = (2.0_f64 * (row_maf[idx] as f64)).max(0.0);

                            let mut sum_g = 0.0_f64;
                            for (j, &sid) in sample_idx.iter().enumerate() {
                                let b = row[sid >> 2];
                                let code = (b >> ((sid & 3) * 2)) & 0b11;
                                let mut gv = match code {
                                    0b00 => 0.0_f64,
                                    0b10 => 1.0_f64,
                                    0b11 => 2.0_f64,
                                    _ => mean_g,
                                };
                                if flip && code != 0b01 {
                                    gv = 2.0_f64 - gv;
                                }
                                gv = gm.apply(gv);
                                scr.g[j] = gv;
                                sum_g += gv;
                            }
                            let g_mean = sum_g / (n as f64);
                            for j in 0..n {
                                scr.g[j] -= g_mean;
                            }

                            scr.u1.fill(0.0);
                            for i in 0..n {
                                let gi = scr.g[i];
                                let ub = i * k;
                                for r in 0..k {
                                    scr.u1[r] += gi * (u_sub[ub + r] as f64);
                                }
                            }

                            for i in 0..n {
                                let ub = i * k;
                                let mut proj = 0.0_f64;
                                for r in 0..k {
                                    proj += scr.u1[r] * (u_sub[ub + r] as f64);
                                }
                                scr.u2[i] = scr.g[i] - proj;
                            }

                            scr.u1_xtsnp.fill(0.0);
                            scr.u2_xtsnp.fill(0.0);

                            let mut u1_snp_snp = 0.0_f64;
                            let mut u1_snp_ty = 0.0_f64;
                            for i in 0..k {
                                let gi = scr.u1[i];
                                let vi = v1_inv[i];
                                u1_snp_snp += vi * gi * gi;
                                u1_snp_ty += vi * gi * u1ty[i];
                                let base = i * p;
                                for r in 0..p {
                                    scr.u1_xtsnp[r] += vi * u1tx[base + r] * gi;
                                }
                            }

                            let mut u2_snp_snp = 0.0_f64;
                            let mut u2_snp_ty = 0.0_f64;
                            for i in 0..n {
                                let gi = scr.u2[i];
                                u2_snp_snp += gi * gi;
                                u2_snp_ty += gi * u2ty[i];
                                let base = i * p;
                                for r in 0..p {
                                    scr.u2_xtsnp[r] += u2tx[base + r] * gi;
                                }
                            }

                            let dim = p + 1;
                            scr.xtv_inv_x.fill(0.0);
                            scr.xtv_inv_y.fill(0.0);
                            for r in 0..p {
                                for c in 0..p {
                                    scr.xtv_inv_x[r * dim + c] = base_a[r * p + c];
                                }
                                scr.xtv_inv_y[r] = base_b[r];
                            }
                            for r in 0..p {
                                let cross = scr.u1_xtsnp[r] + v2_inv * scr.u2_xtsnp[r];
                                scr.xtv_inv_x[p * dim + r] = cross;
                                scr.xtv_inv_x[r * dim + p] = cross;
                            }
                            scr.xtv_inv_x[p * dim + p] = u1_snp_snp + v2_inv * u2_snp_snp;
                            scr.xtv_inv_y[p] = u1_snp_ty + v2_inv * u2_snp_ty;

                            let ridge = 1e-6_f64;
                            for r in 0..dim {
                                scr.xtv_inv_x[r * dim + r] += ridge;
                            }

                            if cholesky_inplace(&mut scr.xtv_inv_x, dim).is_none() {
                                out_row[0] = f64::NAN;
                                out_row[1] = f64::NAN;
                                out_row[2] = f64::NAN;
                                out_row[3] = 1.0;
                                return;
                            }
                            cholesky_solve_into(&scr.xtv_inv_x, dim, &scr.xtv_inv_y, &mut scr.beta);

                            let mut r1_sum = 0.0_f64;
                            for i in 0..k {
                                let mut xb = 0.0_f64;
                                let base = i * p;
                                for r in 0..p {
                                    xb += u1tx[base + r] * scr.beta[r];
                                }
                                xb += scr.u1[i] * scr.beta[p];
                                let ri = u1ty[i] - xb;
                                r1_sum += v1_inv[i] * ri * ri;
                            }

                            let mut r2_sum = 0.0_f64;
                            for i in 0..n {
                                let mut xb = 0.0_f64;
                                let base = i * p;
                                for r in 0..p {
                                    xb += u2tx[base + r] * scr.beta[r];
                                }
                                xb += scr.u2[i] * scr.beta[p];
                                let ri = u2ty[i] - xb;
                                r2_sum += ri * ri;
                            }

                            let rtv_invr = r1_sum + v2_inv * r2_sum;
                            if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
                                out_row[0] = scr.beta[p];
                                out_row[1] = f64::NAN;
                                out_row[2] = f64::NAN;
                                out_row[3] = 1.0;
                                return;
                            }

                            let sigma2 = rtv_invr / df_f;
                            if !sigma2.is_finite() || sigma2 <= 0.0 {
                                out_row[0] = scr.beta[p];
                                out_row[1] = f64::NAN;
                                out_row[2] = f64::NAN;
                                out_row[3] = 1.0;
                                return;
                            }

                            scr.rhs.fill(0.0);
                            scr.rhs[p] = 1.0;
                            cholesky_solve_into(&scr.xtv_inv_x, dim, &scr.rhs, &mut scr.work);
                            let var_beta = sigma2 * scr.work[p];
                            let se = if var_beta.is_finite() && var_beta > 0.0 {
                                var_beta.sqrt()
                            } else {
                                f64::NAN
                            };
                            let beta = scr.beta[p];
                            let pval = if beta.is_finite() && se.is_finite() && se > 0.0 {
                                let z = (beta / se).abs();
                                (2.0 * normal_sf(z)).clamp(f64::MIN_POSITIVE, 1.0)
                            } else {
                                1.0
                            };

                            let ml = c_ml - 0.5 * (n_f * rtv_invr.ln() + log_det_v);
                            let mut stat = if ml.is_finite() {
                                2.0 * (ml - ml0)
                            } else {
                                0.0
                            };
                            if !stat.is_finite() || stat < 0.0 {
                                stat = 0.0;
                            }
                            let plrt = chi2_sf_df1(stat);

                            out_row[0] = beta;
                            out_row[1] = se;
                            out_row[2] = pval;
                            out_row[3] = plrt;
                        },
                    );
                };
                if let Some(tp) = &pool {
                    tp.install(run_block);
                } else {
                    run_block();
                }
            }

            let done = row_end;
            if let Some(cb) = progress_callback.as_ref() {
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (done, m))?;
                    Ok(())
                })?;
            } else {
                Python::attach(|py2| py2.check_signals())?;
            }
        }
        Ok(())
    })?;

    Ok((lbd, ml0, reml0, out))
}
