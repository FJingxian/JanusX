// GWAS linear-model scan on the residualized scale:
// M_X = I - X(X'X)^{-1}X'
// beta_hat = (g' M_X y) / (g' M_X g)
// rss(beta_hat) = y' M_X y - (g' M_X y)^2 / (g' M_X g)
// sigma2_hat = rss(beta_hat) / (n - rank(X) - 1)
// se(beta_hat) = sqrt(sigma2_hat / (g' M_X g))
//
// The maintained CLI path for LM is the windowed BED/memmap scan below.

use memmap2::Mmap;
use nalgebra::DMatrix;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::thread;

use crate::assoc2tsv::{
    append_assoc_block_from_arrays, append_assoc_block_from_core_sites, resolve_assoc_tsv_metadata,
    send_text_buf, AssocArraysBlock, AssocCoreSitesBlock, AssocMissBlock, AssocResultLayout,
};
use crate::bedmath::{
    adaptive_grm_block_rows, decode_mean_imputed_additive_packed_block_rows_f32, packed_byte_lut,
};
use crate::blas::{
    cblas_dgemm_dispatch, rust_sgemm_prefers_rayon_rowmajor_f32_kernel, BlasThreadGuard, CblasInt,
    CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use crate::decode::{
    decode_centered_block_packed_model_f32, AdditiveDecodePlan,
    PackedGeneticModel as PackedDecodeGeneticModel,
};
use crate::gfcore;
use crate::gfcore::BedSnpIter;
use crate::gfcore::BimChunkReader;
use crate::gfreader::{
    build_sample_selection, count_packed_row_counts,
    count_packed_row_counts_selected_with_excluded, prepare_bed_logic_meta_owned_for_stats_samples,
};
use crate::gload::WindowedBedMatrix;
use crate::he::{row_major_block_mul_mat_f32, row_major_block_mul_mat_f32_small_rhs};
use crate::linalg::sanitize_assoc_pvalue;
use crate::stats_common::{check_ctrlc, get_cached_pool, parse_index_vec_i64, AsyncTsvWriter};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct LmMemmapMetaCacheKey {
    prefix: String,
    bed_file_len: u64,
    bed_modified_ns: u128,
    sample_len: usize,
    sample_hash: u64,
    maf_bits: u32,
    miss_bits: u32,
    het_bits: u32,
    snps_only: bool,
}

fn lm_memmap_meta_cache(
) -> &'static Mutex<HashMap<LmMemmapMetaCacheKey, Arc<crate::gfreader::PreparedBedLogicMetaOwned>>>
{
    static CACHE: OnceLock<
        Mutex<HashMap<LmMemmapMetaCacheKey, Arc<crate::gfreader::PreparedBedLogicMetaOwned>>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[inline]
fn lm_memmap_modified_ns(meta: &fs::Metadata) -> u128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0u128)
}

#[inline]
fn lm_memmap_sample_hash(sample_idx: Option<&[usize]>) -> (usize, u64) {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let len = sample_idx.map(|v| v.len()).unwrap_or(0usize);
    len.hash(&mut hasher);
    if let Some(idx) = sample_idx {
        for &sid in idx {
            sid.hash(&mut hasher);
        }
    }
    (len, hasher.finish())
}

fn lm_memmap_meta_cache_key(
    prefix: &str,
    sample_idx: Option<&[usize]>,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
) -> Result<LmMemmapMetaCacheKey, String> {
    let bed_path = format!("{prefix}.bed");
    let meta = fs::metadata(&bed_path).map_err(|e| format!("metadata {bed_path}: {e}"))?;
    let (sample_len, sample_hash) = lm_memmap_sample_hash(sample_idx);
    Ok(LmMemmapMetaCacheKey {
        prefix: prefix.to_string(),
        bed_file_len: meta.len(),
        bed_modified_ns: lm_memmap_modified_ns(&meta),
        sample_len,
        sample_hash,
        maf_bits: maf_threshold.to_bits(),
        miss_bits: max_missing_rate.to_bits(),
        het_bits: het_threshold.to_bits(),
        snps_only,
    })
}

fn prepare_bed_logic_meta_owned_for_stats_samples_cached(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    stats_sample_indices: Option<&[usize]>,
) -> Result<Arc<crate::gfreader::PreparedBedLogicMetaOwned>, String> {
    let key = lm_memmap_meta_cache_key(
        prefix,
        stats_sample_indices,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
    )?;
    if let Some(hit) = lm_memmap_meta_cache()
        .lock()
        .map_err(|_| "LM memmap metadata cache lock poisoned".to_string())?
        .get(&key)
        .cloned()
    {
        return Ok(hit);
    }
    let prepared = Arc::new(prepare_bed_logic_meta_owned_for_stats_samples(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        stats_sample_indices,
        false,
    )?);
    let mut cache = lm_memmap_meta_cache()
        .lock()
        .map_err(|_| "LM memmap metadata cache lock poisoned".to_string())?;
    if cache.len() >= 16 && !cache.contains_key(&key) {
        cache.clear();
    }
    cache.insert(key, Arc::clone(&prepared));
    Ok(prepared)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn pinv_svd_square(a: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let q = a.nrows();
    if q != a.ncols() {
        return Err("pinv_svd_square expects a square matrix".to_string());
    }
    let svd = a.clone().svd(true, true);
    let u = svd.u.ok_or_else(|| "SVD failed to produce U".to_string())?;
    let vt = svd
        .v_t
        .ok_or_else(|| "SVD failed to produce V^T".to_string())?;
    let mut s_inv = DMatrix::<f64>::zeros(q, q);
    let smax = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let rcond = 1e-12_f64;
    let cutoff = rcond * smax.max(1.0);
    for i in 0..q {
        let s = svd.singular_values[i];
        if s.is_finite() && s > cutoff {
            s_inv[(i, i)] = 1.0 / s;
        }
    }
    let v = vt.transpose();
    Ok(v * s_inv * u.transpose())
}

pub(crate) fn ixx_from_x_qr(x_flat: &[f64], n: usize, q0: usize) -> Result<Vec<f64>, String> {
    if q0 == 0 {
        return Err("X has zero columns".to_string());
    }
    if x_flat.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if n <= q0 {
        return Err(format!("n too small for LM design: n={n}, q0={q0}"));
    }

    let xmat = DMatrix::<f64>::from_row_slice(n, q0, x_flat);
    let qr = xmat.qr();
    let r = qr.r();
    if r.nrows() != q0 || r.ncols() != q0 {
        return Err(format!(
            "unexpected R shape from QR: got {}x{}, expected {}x{}",
            r.nrows(),
            r.ncols(),
            q0,
            q0
        ));
    }

    let eye = DMatrix::<f64>::identity(q0, q0);
    let r_inv = if let Some(inv) = r.clone().qr().solve(&eye) {
        inv
    } else {
        pinv_svd_square(&r)?
    };
    let ixx = &r_inv * r_inv.transpose();
    Ok(ixx.as_slice().to_vec())
}

#[derive(Clone, Debug)]
pub(crate) struct LmQrProjection {
    q: Vec<f64>,
    q_f32: Vec<f32>,
    qty: Vec<f64>,
    y_resid: Vec<f64>,
    rss0: f64,
    rank: usize,
    n: usize,
}

impl LmQrProjection {
    pub(crate) fn from_design(
        x_flat: &[f64],
        y: &[f64],
        n: usize,
        q0: usize,
    ) -> Result<Self, String> {
        if x_flat.len() != n.saturating_mul(q0) {
            return Err("X shape mismatch".to_string());
        }
        if y.len() != n {
            return Err("y length mismatch".to_string());
        }
        if n == 0 {
            return Err("empty LM design".to_string());
        }
        if q0 == 0 {
            return Err("X has zero columns".to_string());
        }

        // Modified Gram-Schmidt is fast and memory-light here because q is tiny
        // compared with n in GWAS covariate/QTN designs.
        let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(q0);
        for col in 0..q0 {
            let mut v = vec![0.0_f64; n];
            let mut col_norm2 = 0.0_f64;
            for i in 0..n {
                let value = x_flat[i * q0 + col];
                if !value.is_finite() {
                    return Err("LM design contains non-finite values".to_string());
                }
                v[i] = value;
                col_norm2 += value * value;
            }
            if col_norm2 <= 0.0 {
                continue;
            }
            for qv in q_cols.iter() {
                let coeff = dot(&v, qv);
                if coeff != 0.0 {
                    for i in 0..n {
                        v[i] -= coeff * qv[i];
                    }
                }
            }
            // A second pass improves orthogonality for nearly collinear covariates.
            for qv in q_cols.iter() {
                let coeff = dot(&v, qv);
                if coeff != 0.0 {
                    for i in 0..n {
                        v[i] -= coeff * qv[i];
                    }
                }
            }
            let norm2 = dot(&v, &v);
            let tol = 1e-12_f64 * col_norm2.max(1.0);
            if norm2 <= tol || !norm2.is_finite() {
                continue;
            }
            let inv_norm = 1.0 / norm2.sqrt();
            for value in v.iter_mut() {
                *value *= inv_norm;
            }
            q_cols.push(v);
        }

        let rank = q_cols.len();
        if n <= rank + 1 {
            return Err(format!(
                "n too small: require n > rank(X)+1, got n={n}, rank={rank}"
            ));
        }

        let mut q = vec![0.0_f64; n.saturating_mul(rank)];
        for (j, qv) in q_cols.iter().enumerate() {
            for i in 0..n {
                q[i * rank + j] = qv[i];
            }
        }
        let mut qty = vec![0.0_f64; rank];
        for j in 0..rank {
            let mut acc = 0.0_f64;
            for i in 0..n {
                acc += q[i * rank + j] * y[i];
            }
            qty[j] = acc;
        }
        let mut y_resid = vec![0.0_f64; n];
        for i in 0..n {
            let mut fitted = 0.0_f64;
            for j in 0..rank {
                fitted += q[i * rank + j] * qty[j];
            }
            let value = y[i] - fitted;
            if !value.is_finite() {
                return Err("LM QR residualization produced non-finite values".to_string());
            }
            y_resid[i] = value;
        }
        let rss0 = y_resid.iter().map(|v| v * v).sum::<f64>();
        if !rss0.is_finite() {
            return Err("LM QR residual RSS is not finite".to_string());
        }
        let q_f32 = cast_f64_slice_to_f32(&q)?;
        Ok(Self {
            q,
            q_f32,
            qty,
            y_resid,
            rss0,
            rank,
            n,
        })
    }

    #[inline]
    pub(crate) fn q(&self) -> &[f64] {
        self.q.as_slice()
    }

    #[inline]
    pub(crate) fn q_f32(&self) -> &[f32] {
        self.q_f32.as_slice()
    }

    #[inline]
    pub(crate) fn y_resid(&self) -> &[f64] {
        self.y_resid.as_slice()
    }

    #[inline]
    pub(crate) fn rss0(&self) -> f64 {
        self.rss0
    }

    #[inline]
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }
}

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
pub(crate) fn student_t_p_two_sided(t: f64, df: i32) -> f64 {
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

#[inline]
fn chi2_sf_df1(stat: f64) -> f64 {
    if !stat.is_finite() || stat < 0.0 {
        return f64::NAN;
    }
    let p = libm::erfc((0.5 * stat).sqrt());
    if !p.is_finite() {
        return 1.0;
    }
    p.clamp(f64::MIN_POSITIVE, 1.0)
}

#[inline]
pub(crate) fn lm_plrt_from_t2(t2: f64, n_obs: usize, df: i32) -> f64 {
    if df <= 0 || !t2.is_finite() || t2 < 0.0 {
        return f64::NAN;
    }
    let stat = (n_obs as f64) * (1.0 + t2 / (df as f64)).ln();
    chi2_sf_df1(stat)
}

#[inline]
pub(crate) fn normalize_plink_prefix_local(prefix: &str) -> String {
    let s = prefix.trim();
    let low = s.to_ascii_lowercase();
    if low.ends_with(".bed") || low.ends_with(".bim") || low.ends_with(".fam") {
        s[..s.len() - 4].to_string()
    } else {
        s.to_string()
    }
}

#[inline]
pub(crate) fn is_simple_snp_allele(a: &str) -> bool {
    let t = a.trim().to_ascii_uppercase();
    if t.len() != 1 {
        return false;
    }
    matches!(t.as_bytes()[0], b'A' | b'C' | b'G' | b'T')
}

#[inline]
pub(crate) fn lm_resolve_snp_name(snp: &str, chrom: &str, pos: i32) -> String {
    if snp.is_empty() || snp == "." {
        format!("{chrom}_{pos}")
    } else {
        snp.to_string()
    }
}

#[inline]
fn transform_model_value(v: f32, model: &str) -> f32 {
    match model {
        "add" => v,
        "dom" => {
            if (v - 1.0).abs() <= 1e-6 || (v - 2.0).abs() <= 1e-6 {
                1.0
            } else {
                0.0
            }
        }
        "rec" => {
            if (v - 2.0).abs() <= 1e-6 {
                1.0
            } else {
                0.0
            }
        }
        "het" => {
            if (v - 1.0).abs() <= 1e-6 {
                1.0
            } else {
                0.0
            }
        }
        _ => v,
    }
}

#[inline]
fn cast_f64_slice_to_f32(input: &[f64]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(input.len());
    for &value in input {
        if !value.is_finite() {
            return Err("LM memmap scan received non-finite dense operand".to_string());
        }
        out.push(value as f32);
    }
    Ok(out)
}

#[inline]
pub(crate) fn pack_lm_scan_rhs_f32(
    y: &[f64],
    x_rhs: Option<&[f32]>,
    q: usize,
) -> Result<Vec<f32>, String> {
    let n = y.len();
    let rhs_cols = q
        .checked_add(1)
        .ok_or_else(|| "LM memmap RHS column overflow".to_string())?;
    let y_f32 = cast_f64_slice_to_f32(y)?;
    let mut rhs = vec![0.0_f32; n.saturating_mul(rhs_cols)];
    match (x_rhs, q) {
        (Some(x), q_cols) => {
            if x.len() != n.saturating_mul(q_cols) {
                return Err(format!(
                    "LM memmap RHS design mismatch: len(x_rhs)={}, expected {}",
                    x.len(),
                    n.saturating_mul(q_cols)
                ));
            }
            for i in 0..n {
                let dst = &mut rhs[i * rhs_cols..(i + 1) * rhs_cols];
                dst[0] = y_f32[i];
                dst[1..].copy_from_slice(&x[i * q_cols..(i + 1) * q_cols]);
            }
        }
        (None, 0) => {
            for i in 0..n {
                rhs[i] = y_f32[i];
            }
        }
        (None, q_cols) => {
            return Err(format!("LM memmap RHS missing dense design for q={q_cols}"));
        }
    }
    Ok(rhs)
}

fn row_major_block_first_col_f32_to_f64(
    block: &[f32],
    rows: usize,
    stride: usize,
    out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(stride));
    debug_assert_eq!(out.len(), rows);
    let use_parallel = pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(stride) >= 16_384usize;
    if use_parallel {
        let mut run = || {
            out.par_iter_mut().enumerate().for_each(|(r, dst)| {
                *dst = block[r * stride] as f64;
            });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }
    for r in 0..rows {
        out[r] = block[r * stride] as f64;
    }
}

fn row_major_block_sumsq_f64(
    block: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(out.len(), rows);
    let use_parallel = pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(cols) >= 16_384usize;
    if use_parallel {
        let mut run = || {
            out.par_iter_mut().enumerate().for_each(|(r, dst)| {
                let row = &block[r * cols..(r + 1) * cols];
                let mut ss = 0.0_f64;
                for &value in row {
                    let vf = value as f64;
                    ss += vf * vf;
                }
                *dst = ss;
            });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }
    for r in 0..rows {
        let row = &block[r * cols..(r + 1) * cols];
        let mut ss = 0.0_f64;
        for &value in row {
            let vf = value as f64;
            ss += vf * vf;
        }
        out[r] = ss;
    }
}

#[inline]
fn lm_parallel_small_rhs_enabled() -> bool {
    if let Ok(raw) = std::env::var("JX_LM_ROWMAJOR_F32_KERNEL") {
        let key = raw.trim().to_ascii_lowercase();
        return !matches!(
            key.as_str(),
            "0" | "false" | "off" | "no" | "blas" | "gemm" | "serial"
        );
    }
    true
}

#[inline]
fn lm_max_output_rows_env() -> Option<usize> {
    for key in ["JX_LM_MAX_OUTPUT_ROWS", "JX_GWAS_MAX_OUTPUT_ROWS"] {
        if let Ok(raw) = std::env::var(key) {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(value) = trimmed.parse::<usize>() {
                return Some(value);
            }
        }
    }
    None
}

#[inline]
fn lm_prefer_parallel_small_rhs(
    rows: usize,
    cols: usize,
    n_rhs: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> bool {
    if n_rhs == 0 || n_rhs > 64 || !lm_parallel_small_rhs_enabled() {
        return false;
    }
    let Some(tp) = pool else {
        return false;
    };
    tp.current_num_threads() > 1
        && rows.saturating_mul(cols).saturating_mul(n_rhs) >= 1_048_576usize
}

fn row_major_block_mul_mat_f32_lm(
    block: &[f32],
    rows: usize,
    cols: usize,
    rhs: &[f32],
    n_rhs: usize,
    out: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(rhs.len(), cols.saturating_mul(n_rhs));
    debug_assert_eq!(out.len(), rows.saturating_mul(n_rhs));
    if !lm_prefer_parallel_small_rhs(rows, cols, n_rhs, pool) {
        row_major_block_mul_mat_f32(block, rows, cols, rhs, n_rhs, out, pool);
        return;
    }
    row_major_block_mul_mat_f32_small_rhs(block, rows, cols, rhs, n_rhs, out, pool);
}

#[inline]
fn lm_assoc_from_qr_projection(
    qg: &[f64],
    gy_resid: f64,
    ss: f64,
    ctx: &LmQrProjection,
) -> [f64; 5] {
    lm_assoc_from_projection_residual(qg, gy_resid, ss, ctx.rss0, ctx.n, ctx.rank)
}

#[inline]
fn lm_assoc_from_projection_residual(
    qg: &[f64],
    gy_resid: f64,
    ss: f64,
    rss0: f64,
    n: usize,
    rank: usize,
) -> [f64; 5] {
    debug_assert_eq!(qg.len(), rank);
    let q_quad = qg.iter().map(|v| v * v).sum::<f64>();
    let gg_resid = ss - q_quad;
    let df = (n as i32) - (rank as i32) - 1;
    if !(gg_resid.is_finite() && gg_resid > 1e-8) || df <= 0 {
        return [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    }

    let beta_snp = gy_resid / gg_resid;
    let rss1 = (rss0 - gy_resid * beta_snp).max(0.0);
    let ve = rss1 / (df as f64);
    if !(ve.is_finite() && ve > 0.0) {
        return [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    }

    let se_snp = (ve / gg_resid).sqrt();
    if !beta_snp.is_finite() || !se_snp.is_finite() || se_snp <= 0.0 {
        return [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    }
    let t = beta_snp / se_snp;
    let stat = (n as f64) * (1.0 + (t * t) / (df as f64)).ln();
    let pwald = student_t_p_two_sided(t, df);
    let plrt = lm_plrt_from_t2(t * t, n, df);
    [beta_snp, se_snp, stat, pwald, plrt]
}

#[inline]
fn lm_assoc_from_qr_projection_raw(
    qg: &[f64],
    gy_raw: f64,
    ss: f64,
    ctx: &LmQrProjection,
) -> [f64; 5] {
    debug_assert_eq!(qg.len(), ctx.rank);
    let gy_resid = gy_raw - dot(qg, &ctx.qty);
    lm_assoc_from_qr_projection(qg, gy_resid, ss, ctx)
}

#[inline]
fn lm_assoc_from_projection_raw_values(
    qg: &[f64],
    qty: &[f64],
    gy_raw: f64,
    ss: f64,
    rss0: f64,
    n: usize,
    rank: usize,
) -> [f64; 5] {
    debug_assert_eq!(qg.len(), rank);
    debug_assert_eq!(qty.len(), rank);
    let gy_resid = gy_raw - dot(qg, qty);
    lm_assoc_from_projection_residual(qg, gy_resid, ss, rss0, n, rank)
}

#[derive(Clone, Copy)]
struct LmUnifiedSnpCounts {
    maf: f32,
    missing_count: usize,
}

struct LmUnifiedStreamingChunk {
    g_block: Vec<f32>,
    qg_block: Vec<f32>,
    gy_block: Vec<f64>,
    ss_block: Vec<f64>,
    out_block: Vec<f64>,
    miss_block: Vec<usize>,
    indices: Vec<usize>,
    flip: Vec<bool>,
    maf: Vec<f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    a0: Vec<String>,
    a1: Vec<String>,
    ctx_ids: Vec<usize>,
    rows: usize,
    scanned_to: usize,
}

impl LmUnifiedStreamingChunk {
    fn new(capacity: usize, n: usize, q_rank: usize) -> Self {
        Self {
            g_block: vec![0.0_f32; capacity.saturating_mul(n)],
            qg_block: vec![0.0_f32; capacity.saturating_mul(q_rank.saturating_add(1).max(1))],
            gy_block: vec![0.0_f64; capacity],
            ss_block: vec![0.0_f64; capacity],
            out_block: vec![0.0_f64; capacity.saturating_mul(5)],
            miss_block: vec![0usize; capacity],
            indices: Vec::with_capacity(capacity),
            flip: Vec::with_capacity(capacity),
            maf: Vec::with_capacity(capacity),
            chrom: Vec::with_capacity(capacity),
            pos: Vec::with_capacity(capacity),
            snp: Vec::with_capacity(capacity),
            a0: Vec::with_capacity(capacity),
            a1: Vec::with_capacity(capacity),
            ctx_ids: Vec::with_capacity(capacity),
            rows: 0usize,
            scanned_to: 0usize,
        }
    }

    fn clear(&mut self) {
        self.indices.clear();
        self.flip.clear();
        self.maf.clear();
        self.chrom.clear();
        self.pos.clear();
        self.snp.clear();
        self.a0.clear();
        self.a1.clear();
        self.ctx_ids.clear();
        self.rows = 0usize;
        self.scanned_to = 0usize;
    }
}

#[derive(Clone, Debug)]
struct LmStage2CompactContext {
    col_indices: Vec<usize>,
    chol_l: Vec<f64>,
    qty: Vec<f64>,
    rss0: f64,
    rank: usize,
    n: usize,
}

#[inline]
fn lm_forward_solve_lower_inplace(l: &[f64], dim: usize, rhs: &mut [f64]) {
    debug_assert_eq!(l.len(), dim.saturating_mul(dim));
    debug_assert_eq!(rhs.len(), dim);
    for i in 0..dim {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= l[i * dim + k] * rhs[k];
        }
        rhs[i] = sum / l[i * dim + i];
    }
}

fn lm_stage2_build_x_all_f32(
    x_base: &[f64],
    qtn_cov: &[f64],
    n: usize,
    base_cols: usize,
    qtn_cols: usize,
) -> Result<Vec<f32>, String> {
    if x_base.len() != n.saturating_mul(base_cols) {
        return Err("x_base shape mismatch".to_string());
    }
    if qtn_cov.len() != n.saturating_mul(qtn_cols) {
        return Err("qtn_cov shape mismatch".to_string());
    }
    let q_all = base_cols.saturating_add(qtn_cols);
    if q_all == 0 {
        return Err("stage2 design has zero columns".to_string());
    }
    let mut out = vec![0.0_f32; n.saturating_mul(q_all)];
    for i in 0..n {
        let dst = &mut out[i * q_all..(i + 1) * q_all];
        for c in 0..base_cols {
            let value = x_base[i * base_cols + c];
            if !value.is_finite() {
                return Err("x_base contains non-finite values".to_string());
            }
            dst[c] = value as f32;
        }
        for c in 0..qtn_cols {
            let value = qtn_cov[i * qtn_cols + c];
            if !value.is_finite() {
                return Err("qtn_cov contains non-finite values".to_string());
            }
            dst[base_cols + c] = value as f32;
        }
    }
    Ok(out)
}

fn lm_stage2_xtx_from_design_f64(
    x_base: &[f64],
    qtn_cov: &[f64],
    n: usize,
    base_cols: usize,
    qtn_cols: usize,
) -> Result<Vec<f64>, String> {
    if x_base.len() != n.saturating_mul(base_cols) {
        return Err("x_base shape mismatch".to_string());
    }
    if qtn_cov.len() != n.saturating_mul(qtn_cols) {
        return Err("qtn_cov shape mismatch".to_string());
    }
    let q_all = base_cols.saturating_add(qtn_cols);
    if q_all == 0 {
        return Err("stage2 design has zero columns".to_string());
    }
    let mut xtx = vec![0.0_f64; q_all.saturating_mul(q_all)];
    if base_cols > 0 && n > 0 {
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                base_cols as CblasInt,
                base_cols as CblasInt,
                n as CblasInt,
                1.0,
                x_base.as_ptr(),
                base_cols as CblasInt,
                x_base.as_ptr(),
                base_cols as CblasInt,
                0.0,
                xtx.as_mut_ptr(),
                q_all as CblasInt,
            );
        }
    }
    if qtn_cols > 0 && n > 0 {
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                base_cols as CblasInt,
                qtn_cols as CblasInt,
                n as CblasInt,
                1.0,
                x_base.as_ptr(),
                base_cols as CblasInt,
                qtn_cov.as_ptr(),
                qtn_cols as CblasInt,
                0.0,
                xtx.as_mut_ptr().add(base_cols),
                q_all as CblasInt,
            );
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                qtn_cols as CblasInt,
                qtn_cols as CblasInt,
                n as CblasInt,
                1.0,
                qtn_cov.as_ptr(),
                qtn_cols as CblasInt,
                qtn_cov.as_ptr(),
                qtn_cols as CblasInt,
                0.0,
                xtx.as_mut_ptr().add(base_cols * q_all + base_cols),
                q_all as CblasInt,
            );
        }
        for r in 0..base_cols {
            for c in 0..qtn_cols {
                xtx[(base_cols + c) * q_all + r] = xtx[r * q_all + base_cols + c];
            }
        }
    }
    Ok(xtx)
}

fn lm_stage2_xty_from_design(
    x_base: &[f64],
    qtn_cov: &[f64],
    y: &[f64],
    n: usize,
    base_cols: usize,
    qtn_cols: usize,
) -> Result<Vec<f64>, String> {
    let q_all = base_cols.saturating_add(qtn_cols);
    if y.len() != n {
        return Err("y length mismatch".to_string());
    }
    if x_base.len() != n.saturating_mul(base_cols) {
        return Err("x_base shape mismatch".to_string());
    }
    if qtn_cov.len() != n.saturating_mul(qtn_cols) {
        return Err("qtn_cov shape mismatch".to_string());
    }
    let xty: Vec<f64> = (0..q_all)
        .into_par_iter()
        .map(|col| {
            let mut acc = 0.0_f64;
            if col < base_cols {
                for i in 0..n {
                    acc += x_base[i * base_cols + col] * y[i];
                }
            } else {
                let qcol = col - base_cols;
                for i in 0..n {
                    acc += qtn_cov[i * qtn_cols + qcol] * y[i];
                }
            }
            acc
        })
        .collect();
    Ok(xty)
}

fn lm_stage2_context_from_subset_xtx(
    xtx_all: &[f64],
    xty_all: &[f64],
    yy: f64,
    n: usize,
    q_all: usize,
    col_indices: Vec<usize>,
) -> Result<LmStage2CompactContext, String> {
    if col_indices.is_empty() {
        return Err("stage2 context has zero columns".to_string());
    }

    // Rank-revealing compact QR via Cholesky on X'X.  We never materialize Q:
    // for each SNP, Q'g is obtained as solve(L, X'g), where X'X = L L'.
    let requested_rank = col_indices.len();
    let mut active_cols = Vec::<usize>::with_capacity(requested_rank);
    let mut rows_l: Vec<Vec<f64>> = Vec::with_capacity(requested_rank);
    for &src_col in col_indices.iter() {
        if src_col >= q_all {
            return Err(format!(
                "stage2 context column {src_col} out of range {q_all}"
            ));
        }
        let diag = xtx_all[src_col * q_all + src_col];
        if !diag.is_finite() || diag <= 0.0 {
            continue;
        }
        let cur_rank = active_cols.len();
        let mut row = vec![0.0_f64; cur_rank + 1];
        for j in 0..cur_rank {
            let active_col = active_cols[j];
            let mut sum = xtx_all[src_col * q_all + active_col];
            if !sum.is_finite() {
                return Err("stage2 compact QR received non-finite X'X".to_string());
            }
            for k in 0..j {
                sum -= row[k] * rows_l[j][k];
            }
            let pivot = rows_l[j][j];
            if pivot <= 0.0 || !pivot.is_finite() {
                return Err("stage2 compact QR produced invalid Cholesky pivot".to_string());
            }
            row[j] = sum / pivot;
        }
        let projected = row[..cur_rank].iter().map(|v| v * v).sum::<f64>();
        let residual_diag = diag - projected;
        let tol = 1e-12_f64 * diag.abs().max(1.0);
        if residual_diag <= tol || !residual_diag.is_finite() {
            continue;
        }
        row[cur_rank] = residual_diag.sqrt();
        active_cols.push(src_col);
        rows_l.push(row);
    }

    let rank = active_cols.len();
    if rank == 0 {
        return Err("stage2 compact QR failed: all design columns are rank deficient".to_string());
    }
    if n <= rank + 1 {
        return Err(format!(
            "n too small: require n > rank+1, got n={n}, rank={rank}"
        ));
    }
    let mut chol_l = vec![0.0_f64; rank.saturating_mul(rank)];
    for i in 0..rank {
        for j in 0..=i {
            chol_l[i * rank + j] = rows_l[i][j];
        }
    }
    let mut qty = vec![0.0_f64; rank];
    for (i, &src) in active_cols.iter().enumerate() {
        qty[i] = xty_all[src];
    }
    lm_forward_solve_lower_inplace(&chol_l, rank, &mut qty);
    let y_fit_ss = qty.iter().map(|v| v * v).sum::<f64>();
    let rss0 = (yy - y_fit_ss).max(0.0);
    if !rss0.is_finite() {
        return Err("stage2 compact QR produced non-finite residual RSS".to_string());
    }
    Ok(LmStage2CompactContext {
        col_indices: active_cols,
        chol_l,
        qty,
        rss0,
        rank,
        n,
    })
}

fn parse_lm_context_exclude_qtn<'py>(
    context_exclude_qtn: Bound<'py, PyAny>,
    qtn_cols: usize,
) -> PyResult<Vec<Vec<usize>>> {
    let outer = context_exclude_qtn.cast::<PyList>().map_err(|_| {
        PyValueError::new_err("context_exclude_qtn must be a list of qtn-index lists")
    })?;
    let mut out = Vec::<Vec<usize>>::with_capacity(outer.len());
    for (idx, item) in outer.iter().enumerate() {
        let inner = item.cast::<PyList>().map_err(|_| {
            PyValueError::new_err(format!("context_exclude_qtn[{idx}] must be a list"))
        })?;
        let mut vals = Vec::<usize>::with_capacity(inner.len());
        let mut seen = std::collections::HashSet::<usize>::with_capacity(inner.len());
        for value in inner.iter() {
            let qidx: usize = value.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "context_exclude_qtn[{idx}] contains a non-integer value"
                ))
            })?;
            if qidx >= qtn_cols {
                return Err(PyValueError::new_err(format!(
                    "context_exclude_qtn[{idx}] index {qidx} out of range {qtn_cols}"
                )));
            }
            if seen.insert(qidx) {
                vals.push(qidx);
            }
        }
        vals.sort_unstable();
        out.push(vals);
    }
    Ok(out)
}

fn lm_stage2_build_compact_contexts(
    xtx_all: &[f64],
    xty_all: &[f64],
    yy: f64,
    n: usize,
    base_cols: usize,
    qtn_cols: usize,
    context_exclude_qtn: &[Vec<usize>],
    threads: usize,
) -> Result<Vec<LmStage2CompactContext>, String> {
    let q_all = base_cols.saturating_add(qtn_cols);
    let build_one = |exclude: &Vec<usize>| -> Result<LmStage2CompactContext, String> {
        let mut excluded = vec![false; qtn_cols];
        for &idx in exclude {
            excluded[idx] = true;
        }
        let mut cols = Vec::<usize>::with_capacity(q_all.saturating_sub(exclude.len()));
        cols.extend(0..base_cols);
        for qidx in 0..qtn_cols {
            if !excluded[qidx] {
                cols.push(base_cols + qidx);
            }
        }
        lm_stage2_context_from_subset_xtx(xtx_all, xty_all, yy, n, q_all, cols)
    };
    if context_exclude_qtn.len() <= 1 {
        return context_exclude_qtn.iter().map(build_one).collect();
    }
    let ctx_threads = threads.max(1).min(context_exclude_qtn.len()).min(8);
    let pool = get_cached_pool(ctx_threads).map_err(|e| e.to_string())?;
    if let Some(tp) = pool.as_ref() {
        tp.install(|| context_exclude_qtn.par_iter().map(build_one).collect())
    } else {
        context_exclude_qtn.iter().map(build_one).collect()
    }
}

#[inline]
fn lm_stage2_compact_assoc_from_gx(
    ctx: &LmStage2CompactContext,
    gx_all_row: &[f32],
    gy_raw: f64,
    ss: f64,
    qg_tmp: &mut [f64],
) -> [f64; 5] {
    debug_assert!(qg_tmp.len() >= ctx.rank);
    for (dst, &src_col) in ctx.col_indices.iter().enumerate() {
        qg_tmp[dst] = gx_all_row[src_col] as f64;
    }
    lm_forward_solve_lower_inplace(&ctx.chol_l, ctx.rank, &mut qg_tmp[..ctx.rank]);
    lm_assoc_from_projection_raw_values(
        &qg_tmp[..ctx.rank],
        &ctx.qty,
        gy_raw,
        ss,
        ctx.rss0,
        ctx.n,
        ctx.rank,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn lm_stream_bed_additive_windowed_unified(
    norm_prefix: &str,
    y_raw: &[f64],
    qr_ctx: &LmQrProjection,
    out_tsv_path: &str,
    sample_indices: &[usize],
    maf_threshold: f32,
    max_missing_rate: f32,
    snps_only: bool,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: usize,
) -> Result<(usize, usize), String> {
    let n = y_raw.len();
    if sample_indices.len() != n {
        return Err(format!(
            "sample_ids length mismatch: selected n={} but len(y)={n}",
            sample_indices.len()
        ));
    }

    let full_samples = gfcore::read_fam(norm_prefix)?;
    let n_samples_full = full_samples.len();
    if n_samples_full == 0 {
        return Err("no samples in PLINK FAM".to_string());
    }
    if let Some(max_sid) = sample_indices.iter().copied().max() {
        if max_sid >= n_samples_full {
            return Err(format!(
                "sample index {max_sid} out of bounds for {n_samples_full} BED samples"
            ));
        }
    }

    let mut bed_window = WindowedBedMatrix::open(norm_prefix, mmap_window_mb)?;
    if bed_window.n_samples_full() != n_samples_full {
        return Err(format!(
            "BED/FAM sample count mismatch: window={}, fam={n_samples_full}",
            bed_window.n_samples_full()
        ));
    }
    let bytes_per_snp = bed_window.bytes_per_snp();
    let n_snps = bed_window.n_source_snps();
    let mut bim_reader = BimChunkReader::open(norm_prefix)?;

    let decode_plan = AdditiveDecodePlan::from_sample_indices(n_samples_full, sample_indices);
    let sample_identity = decode_plan.sample_identity();
    let use_selected = !sample_identity;
    let selected_excluded_sample_indices = decode_plan.selected_excluded_sample_indices();
    let sample_subset_plan = decode_plan.subset_decode_plan();
    let dense_subset_pos = decode_plan.dense_subset_pos();

    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let q_rank = qr_ctx.rank;
    let rhs_cols = q_rank.saturating_add(1);
    let rhs_f32 = pack_lm_scan_rhs_f32(
        y_raw,
        if q_rank > 0 {
            Some(qr_ctx.q_f32.as_slice())
        } else {
            None
        },
        q_rank,
    )?;
    let scan_chunk_snps = chunk_size.max(1).min(n_snps.max(1));
    let progress_block = if progress_every == 0 {
        scan_chunk_snps.max(1)
    } else {
        progress_every.max(1)
    };
    let mut next_progress_emit = progress_block.min(n_snps).max(1);
    let output_row_limit = lm_max_output_rows_env();
    let mut written_total = 0usize;
    let mut kept_total = 0usize;
    let mut text = String::with_capacity(scan_chunk_snps * 112);
    let mut xts_tmp = vec![0.0_f64; q_rank.max(1)];

    let lm_parallel_rhs =
        lm_prefer_parallel_small_rhs(scan_chunk_snps.max(1), n, rhs_cols, pool.as_ref());
    let blas_threads = if lm_parallel_rhs || rust_sgemm_prefers_rayon_rowmajor_f32_kernel() {
        1usize
    } else {
        threads.max(1)
    };
    let _blas_guard = BlasThreadGuard::enter(blas_threads);

    let writer = AsyncTsvWriter::with_config(
        out_tsv_path,
        b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n",
        64 * 1024 * 1024,
        16,
    )?;

    let mut chunk_start = 0usize;
    let producer_err = Arc::new(OnceLock::<String>::new());
    let producer_err_bg = Arc::clone(&producer_err);
    let producer = |chunk: &mut LmUnifiedStreamingChunk| -> bool {
        if let Err(err) = check_ctrlc() {
            let _ = producer_err_bg.set(err);
            return false;
        }
        if chunk_start >= n_snps {
            return false;
        }
        chunk.clear();
        let chunk_end = (chunk_start + scan_chunk_snps).min(n_snps);
        let chunk_packed = match bed_window.read_source_range(chunk_start, chunk_end) {
            Ok(slice) => slice,
            Err(e) => {
                let _ = producer_err_bg.set(e);
                return false;
            }
        };
        let chunk_sites = match bim_reader.read_range(chunk_start, chunk_end) {
            Ok(sites) => sites,
            Err(e) => {
                let _ = producer_err_bg.set(e);
                return false;
            }
        };

        let counts: Vec<Option<LmUnifiedSnpCounts>> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|snp_idx| {
                let local_idx = snp_idx - chunk_start;
                let row = &chunk_packed[local_idx * bytes_per_snp..(local_idx + 1) * bytes_per_snp];
                let (missing, het, hom_alt) = if use_selected {
                    count_packed_row_counts_selected_with_excluded(
                        row,
                        n_samples_full,
                        sample_indices,
                        selected_excluded_sample_indices,
                    )
                } else {
                    count_packed_row_counts(row, n_samples_full)
                };
                let non_missing = n.saturating_sub(missing);
                let miss_rate = if n > 0 {
                    missing as f32 / n as f32
                } else {
                    1.0_f32
                };
                if miss_rate > max_missing_rate {
                    return None;
                }
                if non_missing == 0 {
                    return if maf_threshold > 0.0 {
                        None
                    } else {
                        Some(LmUnifiedSnpCounts {
                            maf: 0.0_f32,
                            missing_count: missing,
                        })
                    };
                }

                let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                let alt_freq = alt_sum as f32 / (2.0_f32 * non_missing as f32);
                let maf = alt_freq.min(1.0_f32 - alt_freq);
                if maf < maf_threshold {
                    return None;
                }

                Some(LmUnifiedSnpCounts {
                    maf: alt_freq,
                    missing_count: missing,
                })
            })
            .collect();

        for (offset, (cnts_opt, site)) in
            counts.into_iter().zip(chunk_sites.into_iter()).enumerate()
        {
            let cnts = match cnts_opt {
                Some(c) => c,
                None => continue,
            };
            if snps_only
                && (!is_simple_snp_allele(&site.ref_allele)
                    || !is_simple_snp_allele(&site.alt_allele))
            {
                continue;
            }
            let snp_name = lm_resolve_snp_name(&site.snp, &site.chrom, site.pos);
            chunk.indices.push(offset);
            chunk.flip.push(false);
            chunk.maf.push(cnts.maf);
            chunk.miss_block[chunk.rows] = cnts.missing_count;
            chunk.chrom.push(site.chrom);
            chunk.pos.push(site.pos as i64);
            chunk.snp.push(snp_name);
            chunk.a0.push(site.ref_allele);
            chunk.a1.push(site.alt_allele);
            chunk.rows += 1;
        }

        if chunk.rows > 0 {
            decode_centered_block_packed_model_f32(
                chunk_packed,
                bytes_per_snp,
                &chunk.flip,
                &chunk.maf,
                Some(&chunk.indices),
                0,
                chunk.rows,
                if sample_identity { n_samples_full } else { n },
                PackedDecodeGeneticModel::Add,
                sample_identity,
                sample_subset_plan,
                dense_subset_pos,
                &mut chunk.g_block[..chunk.rows * n],
            );
        }

        chunk.scanned_to = chunk_end;
        chunk_start = chunk_end;
        chunk_start < n_snps
    };

    let consumer = |chunk: &mut LmUnifiedStreamingChunk| {
        let rows = chunk.rows;
        if rows > 0 {
            let g_slice = &chunk.g_block[..rows * n];
            let gx_slice = &mut chunk.qg_block[..rows * rhs_cols];
            row_major_block_mul_mat_f32_lm(
                g_slice,
                rows,
                n,
                rhs_f32.as_slice(),
                rhs_cols,
                gx_slice,
                pool.as_ref(),
            );
            row_major_block_first_col_f32_to_f64(
                gx_slice,
                rows,
                rhs_cols,
                &mut chunk.gy_block[..rows],
                pool.as_ref(),
            );
            row_major_block_sumsq_f64(g_slice, rows, n, &mut chunk.ss_block[..rows], pool.as_ref());
            if q_rank > 0 {
                for row_idx in 0..rows {
                    let qg_row = &gx_slice[row_idx * rhs_cols + 1..row_idx * rhs_cols + 1 + q_rank];
                    for j in 0..q_rank {
                        xts_tmp[j] = qg_row[j] as f64;
                    }
                    let stats = lm_assoc_from_qr_projection_raw(
                        &xts_tmp[..q_rank],
                        chunk.gy_block[row_idx],
                        chunk.ss_block[row_idx],
                        qr_ctx,
                    );
                    chunk.out_block[row_idx * 5..(row_idx + 1) * 5].copy_from_slice(&stats);
                }
            } else {
                for row_idx in 0..rows {
                    let stats = lm_assoc_from_qr_projection_raw(
                        &xts_tmp[..q_rank],
                        chunk.gy_block[row_idx],
                        chunk.ss_block[row_idx],
                        qr_ctx,
                    );
                    chunk.out_block[row_idx * 5..(row_idx + 1) * 5].copy_from_slice(&stats);
                }
            }

            let write_rows = output_row_limit
                .map(|limit| limit.saturating_sub(written_total).min(rows))
                .unwrap_or(rows);
            if write_rows > 0 {
                text.clear();
                if text.capacity() < write_rows * 112 {
                    text.reserve(write_rows * 112 - text.capacity());
                }
                append_assoc_block_from_arrays(
                    &mut text,
                    AssocResultLayout::PrecomputedChisqBasic3 {
                        row_stride: 5,
                        beta_col: 0,
                        se_col: 1,
                        chisq_col: 2,
                        raw_p_col: 3,
                    },
                    "add",
                    AssocArraysBlock {
                        chrom: &chunk.chrom[..write_rows],
                        pos: &chunk.pos[..write_rows],
                        snp: &chunk.snp[..write_rows],
                        allele0: &chunk.a0[..write_rows],
                        allele1: &chunk.a1[..write_rows],
                        maf: &chunk.maf[..write_rows],
                        miss: AssocMissBlock::CountUsize(&chunk.miss_block[..write_rows]),
                    },
                    &chunk.out_block[..write_rows * 5],
                )?;
                send_text_buf(&writer, &mut text)?;
                written_total = written_total.saturating_add(write_rows);
            }
        }

        kept_total = kept_total.saturating_add(rows);
        if chunk.scanned_to >= next_progress_emit {
            if let Some(cb) = progress_callback {
                let _ = Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (chunk.scanned_to.min(n_snps), n_snps))?;
                    Ok(())
                });
            } else {
                check_ctrlc()?;
            }
            next_progress_emit = (chunk.scanned_to / progress_block + 1)
                .saturating_mul(progress_block)
                .min(n_snps);
        }

        Ok::<(), String>(())
    };

    crate::pipeline::run_double_buffer(
        2,
        || LmUnifiedStreamingChunk::new(scan_chunk_snps, n, q_rank),
        producer,
        consumer,
    )?;
    if let Some(err) = producer_err.get() {
        return Err(err.clone());
    }
    bim_reader.ensure_exhausted(n_snps)?;
    if let Some(cb) = progress_callback {
        let _ = Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (n_snps, n_snps))?;
            Ok(())
        });
    }
    writer.finish()?;
    Ok((kept_total, n_snps))
}

#[inline]
fn lm_segment_context_for_active_idx(
    segments: &[LmStreamSegment],
    cursor: &mut usize,
    active_idx: usize,
) -> Result<usize, String> {
    if segments.is_empty() {
        return Ok(0usize);
    }
    while *cursor < segments.len() && active_idx >= segments[*cursor].end {
        *cursor += 1;
    }
    if *cursor >= segments.len() {
        return Err(format!(
            "active marker index {active_idx} exceeds segment coverage ending at {}",
            segments.last().map(|s| s.end).unwrap_or(0)
        ));
    }
    let seg = &segments[*cursor];
    if active_idx < seg.start {
        return Ok(0usize);
    }
    Ok(seg.ctx)
}

fn lm_build_source_window_map(
    source_windows: &[LmSourceWindow],
    n_contexts: usize,
) -> Result<HashMap<String, Vec<(i64, i64, usize)>>, String> {
    let mut map = HashMap::<String, Vec<(i64, i64, usize)>>::new();
    for win in source_windows {
        if win.start > win.end {
            return Err(format!(
                "source window has start > end on {}: {} > {}",
                win.chrom, win.start, win.end
            ));
        }
        if win.ctx >= n_contexts {
            return Err(format!(
                "source window context index {} exceeds context length {}",
                win.ctx, n_contexts
            ));
        }
        map.entry(win.chrom.clone())
            .or_default()
            .push((win.start, win.end, win.ctx));
    }
    for windows in map.values_mut() {
        windows.sort_unstable_by_key(|v| (v.0, v.1, v.2));
        for pair in windows.windows(2) {
            if pair[1].0 <= pair[0].1 {
                return Err("source windows must be non-overlapping within chromosome".to_string());
            }
        }
    }
    Ok(map)
}

#[inline]
fn lm_source_window_context(
    source_window_map: &HashMap<String, Vec<(i64, i64, usize)>>,
    chrom: &str,
    pos: i64,
) -> usize {
    let Some(windows) = source_window_map.get(chrom) else {
        return 0usize;
    };
    let idx = windows.partition_point(|w| w.0 <= pos);
    if idx == 0 {
        return 0usize;
    }
    let (start, end, ctx) = windows[idx - 1];
    if pos >= start && pos <= end {
        ctx
    } else {
        0usize
    }
}

#[allow(clippy::too_many_arguments)]
fn lm_stream_bed_segments_compact_windowed_unified(
    norm_prefix: &str,
    y_raw: &[f64],
    x_all_f32: &[f32],
    q_all: usize,
    compact_contexts: &[LmStage2CompactContext],
    segment_plan: &[LmStreamSegment],
    source_windows: &[LmSourceWindow],
    out_tsv_path: &str,
    sample_indices: &[usize],
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: usize,
) -> Result<(usize, usize), String> {
    let n = y_raw.len();
    if compact_contexts.is_empty() {
        return Err("compact stage2 contexts must not be empty".to_string());
    }
    if q_all == 0 || x_all_f32.len() != n.saturating_mul(q_all) {
        return Err("compact stage2 design shape mismatch".to_string());
    }
    if sample_indices.len() != n {
        return Err(format!(
            "sample_ids length mismatch: selected n={} but len(y)={n}",
            sample_indices.len()
        ));
    }
    let full_samples = gfcore::read_fam(norm_prefix)?;
    let n_samples_full = full_samples.len();
    if n_samples_full == 0 {
        return Err("no samples in PLINK FAM".to_string());
    }
    if let Some(max_sid) = sample_indices.iter().copied().max() {
        if max_sid >= n_samples_full {
            return Err(format!(
                "sample index {max_sid} out of bounds for {n_samples_full} BED samples"
            ));
        }
    }

    let mut bed_window = WindowedBedMatrix::open(norm_prefix, mmap_window_mb)?;
    let bytes_per_snp = bed_window.bytes_per_snp();
    let n_snps = bed_window.n_source_snps();
    let mut bim_reader = BimChunkReader::open(norm_prefix)?;

    let decode_plan = AdditiveDecodePlan::from_sample_indices(n_samples_full, sample_indices);
    let sample_identity = decode_plan.sample_identity();
    let use_selected = !sample_identity;
    let selected_excluded_sample_indices = decode_plan.selected_excluded_sample_indices();
    let sample_subset_plan = decode_plan.subset_decode_plan();
    let dense_subset_pos = decode_plan.dense_subset_pos();

    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let max_rank = compact_contexts
        .iter()
        .map(|ctx| ctx.rank)
        .max()
        .unwrap_or(0usize);
    let rhs_cols = q_all.saturating_add(1);
    let rhs_f32 = pack_lm_scan_rhs_f32(y_raw, Some(x_all_f32), q_all)?;
    let source_window_map = if source_windows.is_empty() {
        None
    } else {
        Some(lm_build_source_window_map(
            source_windows,
            compact_contexts.len(),
        )?)
    };
    let scan_chunk_snps = chunk_size.max(1).min(n_snps.max(1));
    let progress_block = if progress_every == 0 {
        scan_chunk_snps.max(1)
    } else {
        progress_every.max(1)
    };
    let mut next_progress_emit = progress_block.min(n_snps).max(1);
    let output_row_limit = lm_max_output_rows_env();
    let mut written_total = 0usize;
    let mut kept_total = 0usize;
    let mut text = String::with_capacity(scan_chunk_snps * 112);
    let mut qg_tmp = vec![0.0_f64; max_rank.max(1)];

    let lm_parallel_rhs =
        lm_prefer_parallel_small_rhs(scan_chunk_snps.max(1), n, rhs_cols, pool.as_ref());
    let blas_threads = if lm_parallel_rhs || rust_sgemm_prefers_rayon_rowmajor_f32_kernel() {
        1usize
    } else {
        threads.max(1)
    };
    let _blas_guard = BlasThreadGuard::enter(blas_threads);

    let writer = AsyncTsvWriter::with_config(
        out_tsv_path,
        b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n",
        64 * 1024 * 1024,
        16,
    )?;

    let mut chunk_start = 0usize;
    let mut active_idx = 0usize;
    let mut segment_cursor = 0usize;
    let producer_err = Arc::new(OnceLock::<String>::new());
    let producer_err_bg = Arc::clone(&producer_err);
    let producer = |chunk: &mut LmUnifiedStreamingChunk| -> bool {
        if chunk_start >= n_snps {
            return false;
        }
        chunk.clear();
        let chunk_end = (chunk_start + scan_chunk_snps).min(n_snps);
        let chunk_packed = match bed_window.read_source_range(chunk_start, chunk_end) {
            Ok(slice) => slice,
            Err(e) => {
                let _ = producer_err_bg.set(e);
                return false;
            }
        };
        let chunk_sites = match bim_reader.read_range(chunk_start, chunk_end) {
            Ok(sites) => sites,
            Err(e) => {
                let _ = producer_err_bg.set(e);
                return false;
            }
        };

        let counts: Vec<Option<LmUnifiedSnpCounts>> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|snp_idx| {
                let local_idx = snp_idx - chunk_start;
                let row = &chunk_packed[local_idx * bytes_per_snp..(local_idx + 1) * bytes_per_snp];
                let (missing, het, hom_alt) = if use_selected {
                    count_packed_row_counts_selected_with_excluded(
                        row,
                        n_samples_full,
                        sample_indices,
                        selected_excluded_sample_indices,
                    )
                } else {
                    count_packed_row_counts(row, n_samples_full)
                };
                let non_missing = n.saturating_sub(missing);
                let miss_rate = if n > 0 {
                    missing as f32 / n as f32
                } else {
                    1.0_f32
                };
                if miss_rate > max_missing_rate {
                    return None;
                }
                if non_missing == 0 {
                    return if maf_threshold > 0.0 {
                        None
                    } else {
                        Some(LmUnifiedSnpCounts {
                            maf: 0.0_f32,
                            missing_count: missing,
                        })
                    };
                }
                if het_threshold > 0.0 {
                    let het_rate = het as f32 / non_missing as f32;
                    if het_rate > het_threshold {
                        return None;
                    }
                }
                let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                let alt_freq = alt_sum as f32 / (2.0_f32 * non_missing as f32);
                let maf = alt_freq.min(1.0_f32 - alt_freq);
                if maf < maf_threshold {
                    return None;
                }

                Some(LmUnifiedSnpCounts {
                    maf: alt_freq,
                    missing_count: missing,
                })
            })
            .collect();

        for (offset, (cnts_opt, site)) in
            counts.into_iter().zip(chunk_sites.into_iter()).enumerate()
        {
            let cnts = match cnts_opt {
                Some(c) => c,
                None => continue,
            };
            if snps_only
                && (!is_simple_snp_allele(&site.ref_allele)
                    || !is_simple_snp_allele(&site.alt_allele))
            {
                continue;
            }
            let ctx_id = if let Some(window_map) = source_window_map.as_ref() {
                lm_source_window_context(window_map, &site.chrom, site.pos as i64)
            } else {
                match lm_segment_context_for_active_idx(
                    segment_plan,
                    &mut segment_cursor,
                    active_idx,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = producer_err_bg.set(e);
                        return false;
                    }
                }
            };
            if ctx_id >= compact_contexts.len() {
                let _ = producer_err_bg.set(format!(
                    "segment context index {ctx_id} exceeds compact context length {}",
                    compact_contexts.len()
                ));
                return false;
            }
            let snp_name = lm_resolve_snp_name(&site.snp, &site.chrom, site.pos);
            chunk.indices.push(offset);
            chunk.flip.push(false);
            chunk.maf.push(cnts.maf);
            chunk.miss_block[chunk.rows] = cnts.missing_count;
            chunk.chrom.push(site.chrom);
            chunk.pos.push(site.pos as i64);
            chunk.snp.push(snp_name);
            chunk.a0.push(site.ref_allele);
            chunk.a1.push(site.alt_allele);
            chunk.ctx_ids.push(ctx_id);
            chunk.rows += 1;
            active_idx = active_idx.saturating_add(1);
        }

        if chunk.rows > 0 {
            decode_centered_block_packed_model_f32(
                chunk_packed,
                bytes_per_snp,
                &chunk.flip,
                &chunk.maf,
                Some(&chunk.indices),
                0,
                chunk.rows,
                if sample_identity { n_samples_full } else { n },
                PackedDecodeGeneticModel::Add,
                sample_identity,
                sample_subset_plan,
                dense_subset_pos,
                &mut chunk.g_block[..chunk.rows * n],
            );
        }

        chunk.scanned_to = chunk_end;
        chunk_start = chunk_end;
        chunk_start < n_snps
    };

    let consumer = |chunk: &mut LmUnifiedStreamingChunk| {
        let rows = chunk.rows;
        if rows > 0 {
            let g_slice = &chunk.g_block[..rows * n];
            let gx_slice = &mut chunk.qg_block[..rows * rhs_cols];
            row_major_block_mul_mat_f32_lm(
                g_slice,
                rows,
                n,
                rhs_f32.as_slice(),
                rhs_cols,
                gx_slice,
                pool.as_ref(),
            );
            row_major_block_first_col_f32_to_f64(
                gx_slice,
                rows,
                rhs_cols,
                &mut chunk.gy_block[..rows],
                pool.as_ref(),
            );
            row_major_block_sumsq_f64(g_slice, rows, n, &mut chunk.ss_block[..rows], pool.as_ref());

            for row_idx in 0..rows {
                let ctx = &compact_contexts[chunk.ctx_ids[row_idx]];
                let gx_row = &gx_slice[row_idx * rhs_cols + 1..row_idx * rhs_cols + 1 + q_all];
                let stats = lm_stage2_compact_assoc_from_gx(
                    ctx,
                    gx_row,
                    chunk.gy_block[row_idx],
                    chunk.ss_block[row_idx],
                    &mut qg_tmp,
                );
                chunk.out_block[row_idx * 5..(row_idx + 1) * 5].copy_from_slice(&stats);
            }

            let write_rows = output_row_limit
                .map(|limit| limit.saturating_sub(written_total).min(rows))
                .unwrap_or(rows);
            if write_rows > 0 {
                text.clear();
                if text.capacity() < write_rows * 112 {
                    text.reserve(write_rows * 112 - text.capacity());
                }
                append_assoc_block_from_arrays(
                    &mut text,
                    AssocResultLayout::PrecomputedChisqBasic3 {
                        row_stride: 5,
                        beta_col: 0,
                        se_col: 1,
                        chisq_col: 2,
                        raw_p_col: 3,
                    },
                    "add",
                    AssocArraysBlock {
                        chrom: &chunk.chrom[..write_rows],
                        pos: &chunk.pos[..write_rows],
                        snp: &chunk.snp[..write_rows],
                        allele0: &chunk.a0[..write_rows],
                        allele1: &chunk.a1[..write_rows],
                        maf: &chunk.maf[..write_rows],
                        miss: AssocMissBlock::CountUsize(&chunk.miss_block[..write_rows]),
                    },
                    &chunk.out_block[..write_rows * 5],
                )?;
                send_text_buf(&writer, &mut text)?;
                written_total = written_total.saturating_add(write_rows);
            }
        }

        kept_total = kept_total.saturating_add(rows);
        if chunk.scanned_to >= next_progress_emit {
            if let Some(cb) = progress_callback {
                let _ = Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (chunk.scanned_to.min(n_snps), n_snps))?;
                    Ok(())
                });
            }
            next_progress_emit = (chunk.scanned_to / progress_block + 1)
                .saturating_mul(progress_block)
                .min(n_snps);
        }

        Ok::<(), String>(())
    };

    crate::pipeline::run_double_buffer(
        2,
        || LmUnifiedStreamingChunk::new(scan_chunk_snps, n, q_all),
        producer,
        consumer,
    )?;
    if let Some(err) = producer_err.get() {
        return Err(err.clone());
    }
    bim_reader.ensure_exhausted(n_snps)?;
    if source_window_map.is_none() && !segment_plan.is_empty() {
        let expected = segment_plan.last().map(|s| s.end).unwrap_or(0usize);
        if active_idx != expected {
            return Err(format!(
                "segment active marker count mismatch: streamed={active_idx}, segments_end={expected}"
            ));
        }
    }
    if let Some(cb) = progress_callback {
        let _ = Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (n_snps, n_snps))?;
            Ok(())
        });
    }
    writer.finish()?;
    Ok((kept_total, n_snps))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn lm_stream_bed_additive_prepared_unified(
    norm_prefix: &str,
    qr_ctx: &LmQrProjection,
    rhs_add_f32: &[f32],
    out_tsv_path: &str,
    sample_indices: &[usize],
    prepared_meta_input: (Vec<i64>, Vec<bool>, Vec<f32>, Vec<f32>),
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> Result<(usize, usize), String> {
    let n = qr_ctx.n;
    if sample_indices.len() != n {
        return Err(format!(
            "sample_ids length mismatch: selected n={} but len(y)={n}",
            sample_indices.len()
        ));
    }
    let q_rank = qr_ctx.rank;
    let rhs_add_cols = q_rank.saturating_add(1);
    let (row_idx64, row_flip, row_missing, row_maf) = prepared_meta_input;
    let m = row_idx64.len();
    if row_flip.len() != m || row_missing.len() != m || row_maf.len() != m {
        return Err(format!(
            "prepared row metadata length mismatch: row_indices={m}, row_flip={}, row_missing={}, row_maf={}",
            row_flip.len(),
            row_missing.len(),
            row_maf.len(),
        ));
    }

    let full_samples = gfcore::read_fam(norm_prefix)?;
    let n_samples_full = full_samples.len();
    if n_samples_full == 0 {
        return Err("no samples in PLINK FAM".to_string());
    }
    if let Some(max_sid) = sample_indices.iter().copied().max() {
        if max_sid >= n_samples_full {
            return Err(format!(
                "sample index {max_sid} out of bounds for {n_samples_full} BED samples"
            ));
        }
    }
    let bytes_per_snp = (n_samples_full + 3) / 4;
    if bytes_per_snp == 0 {
        return Err("bytes_per_snp is zero".to_string());
    }
    let bed_path = format!("{norm_prefix}.bed");
    let mut bed_window = if let Some(window_mb) = mmap_window_mb.filter(|&v| v > 0) {
        Some(WindowedBedMatrix::open(norm_prefix, window_mb)?)
    } else {
        None
    };
    let full_mmap = if bed_window.is_none() {
        let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
        if mmap.len() < 3 || mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err("Only SNP-major BED supported".to_string());
        }
        let data_len = mmap.len() - 3;
        if data_len % bytes_per_snp != 0 {
            return Err(format!(
                "BED payload length {data_len} not a multiple of {bytes_per_snp}"
            ));
        }
        Some(mmap)
    } else {
        None
    };
    let n_snps = if let Some(window) = bed_window.as_ref() {
        window.n_source_snps()
    } else {
        let mmap = full_mmap
            .as_ref()
            .ok_or_else(|| "internal error: missing BED source".to_string())?;
        (mmap.len() - 3) / bytes_per_snp
    };
    let source_rows = parse_index_vec_i64(row_idx64.as_slice(), n_snps, "row_indices")
        .map_err(|e| e.to_string())?;
    let mut prev_src: Option<usize> = None;
    for &src_row in source_rows.iter() {
        if let Some(prev) = prev_src {
            if src_row < prev {
                return Err(
                    "prepared row_indices must be sorted in ascending BED order".to_string()
                );
            }
        }
        prev_src = Some(src_row);
    }
    let total_scan_units = source_rows.len();

    let decode_plan = AdditiveDecodePlan::from_sample_indices(n_samples_full, sample_indices);
    let sample_identity = decode_plan.sample_identity();

    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let scan_block_rows =
        adaptive_grm_block_rows(chunk_size.max(512), total_scan_units, n, 0usize, threads).max(1);
    let progress_block = if progress_every == 0 {
        scan_block_rows.max(1)
    } else {
        progress_every.max(1)
    };
    let output_row_limit = lm_max_output_rows_env();
    let mut written_total = 0usize;
    let code4_lut = &packed_byte_lut().code4;
    let lm_parallel_rhs =
        lm_prefer_parallel_small_rhs(scan_block_rows.max(1), n, rhs_add_cols, pool.as_ref());
    let blas_threads = if lm_parallel_rhs || rust_sgemm_prefers_rayon_rowmajor_f32_kernel() {
        1usize
    } else {
        threads.max(1)
    };
    let _blas_guard = BlasThreadGuard::enter(blas_threads);
    let writer = AsyncTsvWriter::with_config(
        out_tsv_path,
        b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n",
        64 * 1024 * 1024,
        16,
    )?;
    let mut bim_reader = BimChunkReader::open(norm_prefix)?;
    let mut block = vec![0.0_f32; scan_block_rows * n];
    let mut gy_block = vec![0.0_f64; scan_block_rows];
    let mut xts_block = vec![0.0_f32; scan_block_rows * rhs_add_cols.max(1)];
    let mut row_ss = vec![0.0_f64; scan_block_rows];
    let mut out_buf = vec![0.0_f64; scan_block_rows * 5];
    let mut xts_tmp = vec![0.0_f64; q_rank.max(1)];
    let mut text = String::with_capacity(scan_block_rows * 112);
    let mut rel_row_indices = Vec::<usize>::with_capacity(scan_block_rows);
    let mut contig_row_indices = Vec::<usize>::with_capacity(scan_block_rows);
    let mut row_start = 0usize;
    let mut next_progress_emit = progress_block.min(total_scan_units).max(1);
    let mut kept_total = 0usize;

    while row_start < total_scan_units {
        check_ctrlc()?;
        let row_end = (row_start + scan_block_rows).min(total_scan_units);
        let rows_here = row_end - row_start;
        let source_rows_batch = &source_rows[row_start..row_end];
        let row_flip_batch = &row_flip[row_start..row_end];
        let row_maf_batch = &row_maf[row_start..row_end];
        let row_missing_batch = &row_missing[row_start..row_end];
        let chunk_sites = bim_reader.read_selected_rows(source_rows_batch)?;

        let (chunk_packed, local_row_indices): (&[u8], &[usize]) = match bed_window.as_mut() {
            Some(window) => {
                let slice = window.prepare_source_rows(source_rows_batch, &mut rel_row_indices)?;
                (slice, rel_row_indices.as_slice())
            }
            None => {
                let mmap = full_mmap
                    .as_ref()
                    .ok_or_else(|| "internal error: missing full BED mmap".to_string())?;
                let packed = &mmap[3..];
                let src_start = source_rows_batch[0];
                let src_end = source_rows_batch[source_rows_batch.len() - 1] + 1;
                let start_byte = src_start * bytes_per_snp;
                let end_byte = src_end * bytes_per_snp;
                contig_row_indices.clear();
                contig_row_indices.extend(
                    source_rows_batch
                        .iter()
                        .map(|&src_row| src_row.saturating_sub(src_start)),
                );
                (&packed[start_byte..end_byte], contig_row_indices.as_slice())
            }
        };

        let block_slice = &mut block[..rows_here * n];
        let gy_slice = &mut gy_block[..rows_here];
        let xts_slice = &mut xts_block[..rows_here * rhs_add_cols];
        let ss_slice = &mut row_ss[..rows_here];
        let out_slice = &mut out_buf[..rows_here * 5];
        decode_mean_imputed_additive_packed_block_rows_f32(
            chunk_packed,
            bytes_per_snp,
            n_samples_full,
            row_flip_batch,
            row_maf_batch,
            sample_indices,
            sample_identity,
            Some(local_row_indices),
            0,
            block_slice,
            code4_lut,
            pool.as_ref(),
        )?;
        row_major_block_mul_mat_f32_lm(
            block_slice,
            rows_here,
            n,
            rhs_add_f32,
            rhs_add_cols,
            xts_slice,
            pool.as_ref(),
        );
        row_major_block_first_col_f32_to_f64(
            xts_slice,
            rows_here,
            rhs_add_cols,
            gy_slice,
            pool.as_ref(),
        );
        row_major_block_sumsq_f64(block_slice, rows_here, n, ss_slice, pool.as_ref());
        for local_idx in 0..rows_here {
            let xts_row =
                &xts_slice[local_idx * rhs_add_cols + 1..local_idx * rhs_add_cols + 1 + q_rank];
            for j in 0..q_rank {
                xts_tmp[j] = xts_row[j] as f64;
            }
            let stats = lm_assoc_from_qr_projection_raw(
                &xts_tmp[..q_rank],
                gy_slice[local_idx],
                ss_slice[local_idx],
                qr_ctx,
            );
            out_slice[local_idx * 5..(local_idx + 1) * 5].copy_from_slice(&stats);
        }

        let write_rows = output_row_limit
            .map(|limit| limit.saturating_sub(written_total).min(rows_here))
            .unwrap_or(rows_here);
        if write_rows > 0 {
            text.clear();
            let miss_counts: Vec<i64> = row_missing_batch[..write_rows]
                .iter()
                .map(|&v| v.round() as i64)
                .collect();
            append_assoc_block_from_core_sites(
                &mut text,
                AssocResultLayout::PrecomputedChisqBasic3 {
                    row_stride: 5,
                    beta_col: 0,
                    se_col: 1,
                    chisq_col: 2,
                    raw_p_col: 3,
                },
                "add",
                AssocCoreSitesBlock {
                    sites: &chunk_sites[..write_rows],
                    maf: &row_maf_batch[..write_rows],
                    miss: AssocMissBlock::CountI64(miss_counts.as_slice()),
                },
                &out_slice[..write_rows * 5],
            )?;
            send_text_buf(&writer, &mut text)?;
            written_total = written_total.saturating_add(write_rows);
        }

        if let Some(cb) = progress_callback {
            if row_end >= next_progress_emit || row_end == total_scan_units {
                while next_progress_emit <= row_end {
                    next_progress_emit = next_progress_emit.saturating_add(progress_block);
                    if next_progress_emit == 0 {
                        break;
                    }
                }
                let done_now = row_end.min(total_scan_units);
                let _ = Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (done_now, total_scan_units))?;
                    Ok(())
                });
            }
        } else if row_end >= next_progress_emit || row_end == total_scan_units {
            while next_progress_emit <= row_end {
                next_progress_emit = next_progress_emit.saturating_add(progress_block);
                if next_progress_emit == 0 {
                    break;
                }
            }
            check_ctrlc()?;
        }
        kept_total = kept_total.saturating_add(rows_here);
        row_start = row_end;
    }

    if let Some(cb) = progress_callback {
        let _ = Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (total_scan_units, total_scan_units))?;
            Ok(())
        });
    }
    writer.finish()?;
    Ok((kept_total, total_scan_units))
}

/// End-to-end LM streaming scan on PLINK BED:
/// read BED chunk -> per-row filter/model/center -> parallel LM assoc -> write TSV
#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    x,
    ixx,
    out_tsv,
    sample_ids=None,
    row_indices=None,
    row_flip=None,
    row_missing=None,
    row_maf=None,
    maf_threshold=0.0,
    max_missing_rate=1.0,
    genetic_model="add",
    het_threshold=0.02,
    snps_only=false,
    chunk_size=10000,
    threads=0,
    progress_callback=None,
    progress_every=0,
    mmap_window_mb=None,
))]
pub fn lm_stream_bed_to_tsv(
    py: Python<'_>,
    prefix: String,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ixx: Option<PyReadonlyArray2<'_, f64>>,
    out_tsv: String,
    sample_ids: Option<Vec<String>>,
    row_indices: Option<PyReadonlyArray1<'_, i64>>,
    row_flip: Option<PyReadonlyArray1<'_, bool>>,
    row_missing: Option<PyReadonlyArray1<'_, f32>>,
    row_maf: Option<PyReadonlyArray1<'_, f32>>,
    maf_threshold: f32,
    max_missing_rate: f32,
    genetic_model: &str,
    het_threshold: f32,
    snps_only: bool,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<(usize, usize)> {
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be > 0"));
    }
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err(PyValueError::new_err(
            "maf_threshold must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err(PyValueError::new_err(
            "max_missing_rate must be within [0, 1.0]",
        ));
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 1.0]",
        ));
    }
    let model = genetic_model.trim().to_ascii_lowercase();
    if !matches!(model.as_str(), "add" | "dom" | "rec" | "het") {
        return Err(PyValueError::new_err(
            "genetic_model must be one of: add, dom, rec, het",
        ));
    }
    if let Some(window_mb) = mmap_window_mb {
        if window_mb == 0 {
            return Err(PyValueError::new_err("mmap_window_mb must be > 0"));
        }
    }

    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let n = y.len();
    let (xn, q0) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    if n <= q0 + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n too small: require n > q0+1, got n={n}, q0={q0}"
        )));
    }

    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    if let Some(ixx_in) = ixx {
        let ixx_arr = ixx_in.as_array();
        if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
            return Err(PyRuntimeError::new_err("ixx must be (q0,q0)"));
        }
    }
    let qr_ctx =
        LmQrProjection::from_design(x_flat.as_ref(), y, n, q0).map_err(PyRuntimeError::new_err)?;
    let q_rank = qr_ctx.rank;
    let additive_model = model.as_str() == "add";
    let x_f32_add = if additive_model {
        Some(qr_ctx.q_f32.clone())
    } else {
        None
    };
    let y_raw = y.to_vec();
    let rhs_add_cols = q_rank.saturating_add(1);
    let rhs_add_f32 = if additive_model {
        Some(
            pack_lm_scan_rhs_f32(y_raw.as_slice(), x_f32_add.as_deref(), q_rank)
                .map_err(PyRuntimeError::new_err)?,
        )
    } else {
        None
    };
    let prepared_meta_input = match (row_indices, row_flip, row_missing, row_maf) {
        (None, None, None, None) => None,
        (Some(row_idx), Some(row_flip), Some(row_missing), Some(row_maf)) => {
            let row_idx_vec = row_idx.as_slice()?.to_vec();
            let row_flip_vec = match row_flip.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_flip.as_array().iter().copied().collect(),
            };
            let row_missing_vec = match row_missing.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_missing.as_array().iter().copied().collect(),
            };
            let row_maf_vec = match row_maf.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_maf.as_array().iter().copied().collect(),
            };
            Some((row_idx_vec, row_flip_vec, row_missing_vec, row_maf_vec))
        }
        _ => {
            return Err(PyRuntimeError::new_err(
                "prepared row metadata must provide all or none of: row_indices, row_flip, row_missing, row_maf",
            ))
        }
    };

    let norm_prefix = normalize_plink_prefix_local(&prefix);
    if additive_model {
        if let Some(prepared_meta_input) = prepared_meta_input {
            let samples = gfcore::read_fam(&norm_prefix).map_err(PyRuntimeError::new_err)?;
            let (sample_indices, _sample_ids_ordered) =
                build_sample_selection(&samples, sample_ids, None)
                    .map_err(PyRuntimeError::new_err)?;
            if sample_indices.len() != n {
                return Err(PyRuntimeError::new_err(format!(
                    "sample_ids length mismatch: selected n={} but len(y)={n}",
                    sample_indices.len()
                )));
            }
            let out_tsv_path = out_tsv.clone();
            return py.detach(move || -> PyResult<(usize, usize)> {
                lm_stream_bed_additive_prepared_unified(
                    norm_prefix.as_str(),
                    &qr_ctx,
                    rhs_add_f32
                        .as_ref()
                        .expect("additive RHS missing")
                        .as_slice(),
                    out_tsv_path.as_str(),
                    sample_indices.as_slice(),
                    prepared_meta_input,
                    chunk_size,
                    threads,
                    progress_callback.as_ref(),
                    progress_every,
                    mmap_window_mb,
                )
                .map_err(PyRuntimeError::new_err)
            });
        }
        if let Some(window_mb) = mmap_window_mb {
            let samples = gfcore::read_fam(&norm_prefix).map_err(PyRuntimeError::new_err)?;
            let (sample_indices, _sample_ids_ordered) =
                build_sample_selection(&samples, sample_ids, None)
                    .map_err(PyRuntimeError::new_err)?;
            if sample_indices.len() != n {
                return Err(PyRuntimeError::new_err(format!(
                    "sample_ids length mismatch: selected n={} but len(y)={n}",
                    sample_indices.len()
                )));
            }
            let out_tsv_path = out_tsv.clone();
            return py.detach(move || -> PyResult<(usize, usize)> {
                lm_stream_bed_additive_windowed_unified(
                    norm_prefix.as_str(),
                    y_raw.as_slice(),
                    &qr_ctx,
                    out_tsv_path.as_str(),
                    sample_indices.as_slice(),
                    maf_threshold,
                    max_missing_rate,
                    snps_only,
                    chunk_size,
                    threads,
                    progress_callback.as_ref(),
                    progress_every,
                    window_mb,
                )
                .map_err(PyRuntimeError::new_err)
            });
        }
    }

    let mut it = if let Some(window_mb) = mmap_window_mb {
        BedSnpIter::new_with_fill_window(&norm_prefix, 0.0, 1.0, false, false, 0.02, window_mb)
            .map_err(PyRuntimeError::new_err)?
    } else {
        BedSnpIter::new_with_fill(&norm_prefix, 0.0, 1.0, false, false, 0.02)
            .map_err(PyRuntimeError::new_err)?
    };
    let (sample_indices, _sample_ids_ordered) =
        build_sample_selection(&it.samples, sample_ids, None).map_err(PyRuntimeError::new_err)?;
    if sample_indices.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "sample_ids length mismatch: selected n={} but len(y)={n}",
            sample_indices.len()
        )));
    }
    let full_samples = sample_indices.len() == it.n_samples();
    let total_snp_hint = it.n_snps();
    let windowed_mode = it.is_windowed();

    let out_tsv_path = out_tsv.clone();

    let pool = get_cached_pool(threads)?;
    let progress_block = if progress_every == 0 {
        chunk_size.max(1)
    } else {
        progress_every.max(1)
    };

    py.detach(move || -> PyResult<(usize, usize)> {
        let writer = AsyncTsvWriter::with_config(
            &out_tsv_path,
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n",
            64 * 1024 * 1024,
            16,
        )
        .map_err(PyRuntimeError::new_err)?;

        let mut chunk_rows: Vec<f32> = Vec::with_capacity(chunk_size * n);
        let mut chunk_sites: Vec<gfcore::SiteInfo> = Vec::with_capacity(chunk_size);
        let mut chunk_maf: Vec<f32> = Vec::with_capacity(chunk_size);
        let mut chunk_miss: Vec<f32> = Vec::with_capacity(chunk_size);
        let mut window_rows: Vec<f32> = vec![0.0_f32; chunk_size * n];
        let mut window_sites: Vec<gfcore::SiteInfo> = Vec::with_capacity(chunk_size);
        let mut window_counts: Vec<gfcore::DecodedSnpRowCounts> = Vec::with_capacity(chunk_size);
        let mut kept_window_rows: Vec<usize> = Vec::with_capacity(chunk_size);
        let mut out_buf: Vec<f64> = vec![0.0_f64; chunk_size * 5];
        let mut text = String::with_capacity(chunk_size * 112);
        let mut gy_block: Vec<f64> = vec![0.0_f64; chunk_size];
        let mut xts_block: Vec<f32> = vec![0.0_f32; chunk_size * rhs_add_cols.max(1)];
        let mut row_ss: Vec<f64> = vec![0.0_f64; chunk_size];
        let mut xts_tmp: Vec<f64> = vec![0.0_f64; q_rank.max(1)];
        let mut kept_total: usize = 0;
        let mut seen_total: usize = 0;
        let mut scan_idx: usize = 0;
        let total_scan = total_snp_hint;
        let mut next_progress_emit: usize = progress_block;
        let output_row_limit = lm_max_output_rows_env();
        let mut written_total: usize = 0;
        let code4_lut = &packed_byte_lut().code4;
        let _blas_guard = if additive_model {
            let lm_parallel_rhs =
                lm_prefer_parallel_small_rhs(chunk_size.max(1), n, rhs_add_cols, pool.as_ref());
            let blas_threads = if lm_parallel_rhs || rust_sgemm_prefers_rayon_rowmajor_f32_kernel()
            {
                1usize
            } else {
                threads.max(1)
            };
            Some(BlasThreadGuard::enter(blas_threads))
        } else {
            None
        };

        let finalize_row = |row_sub: &mut [f32],
                            site: &mut gfcore::SiteInfo,
                            pre_counts: Option<gfcore::DecodedSnpRowCounts>|
         -> Option<(f32, f32)> {
            let stats = if let Some(counts) = pre_counts {
                gfcore::process_snp_row_with_precomputed_counts_preserve_alt(
                    row_sub,
                    &mut site.ref_allele,
                    &mut site.alt_allele,
                    counts,
                    maf_threshold,
                    max_missing_rate,
                    true,
                    model.as_str() != "add",
                    het_threshold,
                )
            } else {
                gfcore::process_snp_row_with_stats_preserve_alt(
                    row_sub,
                    &mut site.ref_allele,
                    &mut site.alt_allele,
                    maf_threshold,
                    max_missing_rate,
                    true,
                    model.as_str() != "add",
                    het_threshold,
                )
            };
            let Some(stats) = stats else {
                return None;
            };

            let mut sum_model = 0.0_f64;
            let mut maf_sum = 0.0_f64;
            for v in row_sub.iter_mut() {
                let mv = transform_model_value(*v, model.as_str());
                *v = mv;
                sum_model += mv as f64;
                maf_sum += mv as f64;
            }
            let mean_model = (sum_model / n as f64) as f32;
            for v in row_sub.iter_mut() {
                *v -= mean_model;
            }
            let maf_val = if model.as_str() == "add" {
                ((maf_sum / n as f64) * 0.5_f64) as f32
            } else {
                (maf_sum / n as f64) as f32
            };
            Some((maf_val, stats.missing_count as f32))
        };

        let prepare_row = |mut row_sub: Vec<f32>,
                           mut site: gfcore::SiteInfo|
         -> Option<(Vec<f32>, gfcore::SiteInfo, f32, f32)> {
            let (maf_val, miss_val) = finalize_row(&mut row_sub, &mut site, None)?;
            if snps_only
                && (!is_simple_snp_allele(&site.ref_allele)
                    || !is_simple_snp_allele(&site.alt_allele))
            {
                return None;
            }
            Some((row_sub, site, maf_val, miss_val))
        };

        if additive_model && !windowed_mode {
            let prepared = prepare_bed_logic_meta_owned_for_stats_samples_cached(
                norm_prefix.as_str(),
                maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                if full_samples {
                    None
                } else {
                    Some(sample_indices.as_slice())
                },
            )
            .map_err(PyRuntimeError::new_err)?;
            let bed_path = format!("{norm_prefix}.bed");
            let bed_file = File::open(&bed_path)
                .map_err(|e| PyRuntimeError::new_err(format!("open {bed_path}: {e}")))?;
            let bed_mmap = unsafe { Mmap::map(&bed_file) }
                .map_err(|e| PyRuntimeError::new_err(format!("mmap {bed_path}: {e}")))?;
            if bed_mmap.len() < 3 {
                return Err(PyRuntimeError::new_err("BED too small"));
            }
            if bed_mmap[0] != 0x6C || bed_mmap[1] != 0x1B || bed_mmap[2] != 0x01 {
                return Err(PyRuntimeError::new_err("Only SNP-major BED supported"));
            }
            let packed_src = &bed_mmap[3..];
            let kept_rows_total = prepared.sites.len();
            let scan_block_rows =
                adaptive_grm_block_rows(chunk_size.max(512), kept_rows_total, n, 0usize, threads)
                    .max(1);
            let mut block = vec![0.0_f32; scan_block_rows * n];
            let mut row_start = 0usize;
            while row_start < kept_rows_total {
                let row_end = (row_start + scan_block_rows).min(kept_rows_total);
                let rows_here = row_end - row_start;
                let block_slice = &mut block[..rows_here * n];
                let gy_slice = &mut gy_block[..rows_here];
                let xts_slice = &mut xts_block[..rows_here * rhs_add_cols];
                let ss_slice = &mut row_ss[..rows_here];
                let out_slice = &mut out_buf[..rows_here * 5];
                decode_mean_imputed_additive_packed_block_rows_f32(
                    packed_src,
                    prepared.bytes_per_snp,
                    prepared.n_samples,
                    &prepared.row_flip,
                    &prepared.maf,
                    sample_indices.as_slice(),
                    full_samples,
                    Some(prepared.row_source_indices.as_slice()),
                    row_start,
                    block_slice,
                    code4_lut,
                    pool.as_ref(),
                )
                .map_err(PyRuntimeError::new_err)?;
                row_major_block_mul_mat_f32_lm(
                    block_slice,
                    rows_here,
                    n,
                    rhs_add_f32
                        .as_ref()
                        .expect("additive RHS missing")
                        .as_slice(),
                    rhs_add_cols,
                    xts_slice,
                    pool.as_ref(),
                );
                row_major_block_first_col_f32_to_f64(
                    xts_slice,
                    rows_here,
                    rhs_add_cols,
                    gy_slice,
                    pool.as_ref(),
                );
                row_major_block_sumsq_f64(block_slice, rows_here, n, ss_slice, pool.as_ref());
                for local_idx in 0..rows_here {
                    let xts_row = &xts_slice
                        [local_idx * rhs_add_cols + 1..local_idx * rhs_add_cols + 1 + q_rank];
                    for j in 0..q_rank {
                        xts_tmp[j] = xts_row[j] as f64;
                    }
                    let stats = lm_assoc_from_qr_projection_raw(
                        &xts_tmp[..q_rank],
                        gy_slice[local_idx],
                        ss_slice[local_idx],
                        &qr_ctx,
                    );
                    out_slice[local_idx * 5..(local_idx + 1) * 5].copy_from_slice(&stats);
                }
                let write_rows = output_row_limit
                    .map(|limit| limit.saturating_sub(written_total).min(rows_here))
                    .unwrap_or(rows_here);
                if write_rows > 0 {
                    text.clear();
                    let miss_counts: Vec<i64> = (0..write_rows)
                        .map(|local_idx| {
                            let row_idx = row_start + local_idx;
                            (prepared.missing_rate[row_idx] as f64 * n as f64).round() as i64
                        })
                        .collect();
                    append_assoc_block_from_core_sites(
                        &mut text,
                        AssocResultLayout::PrecomputedChisqBasic3 {
                            row_stride: 5,
                            beta_col: 0,
                            se_col: 1,
                            chisq_col: 2,
                            raw_p_col: 3,
                        },
                        "add",
                        AssocCoreSitesBlock {
                            sites: &prepared.sites[row_start..row_start + write_rows],
                            maf: &prepared.maf[row_start..row_start + write_rows],
                            miss: AssocMissBlock::CountI64(miss_counts.as_slice()),
                        },
                        &out_slice[..write_rows * 5],
                    )
                    .map_err(PyRuntimeError::new_err)?;
                    send_text_buf(&writer, &mut text).map_err(PyRuntimeError::new_err)?;
                    written_total = written_total.saturating_add(write_rows);
                }
                let done_now = prepared
                    .row_source_indices
                    .get(row_end.saturating_sub(1))
                    .copied()
                    .unwrap_or(0usize)
                    .saturating_add(1);
                if let Some(cb) = progress_callback.as_ref() {
                    if done_now >= next_progress_emit || row_end == kept_rows_total {
                        while next_progress_emit <= done_now {
                            next_progress_emit = next_progress_emit.saturating_add(progress_block);
                            if next_progress_emit == 0 {
                                break;
                            }
                        }
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done_now.min(total_snp_hint), total_snp_hint))?;
                            Ok(())
                        })?;
                    }
                }
                kept_total = kept_total.saturating_add(rows_here);
                row_start = row_end;
            }
            if let Some(cb) = progress_callback.as_ref() {
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (total_snp_hint, total_snp_hint))?;
                    Ok(())
                })?;
            }
            writer.finish().map_err(PyRuntimeError::new_err)?;
            return Ok((kept_total, total_snp_hint));
        }

        loop {
            chunk_rows.clear();
            chunk_sites.clear();
            chunk_maf.clear();
            chunk_miss.clear();
            if scan_idx >= total_scan {
                break;
            }
            let batch_len = (chunk_size).min(total_scan.saturating_sub(scan_idx));
            seen_total = seen_total.saturating_add(batch_len);
            if let Some(cb) = progress_callback.as_ref() {
                if seen_total >= next_progress_emit {
                    while next_progress_emit <= seen_total {
                        next_progress_emit = next_progress_emit.saturating_add(progress_block);
                        if next_progress_emit == 0 {
                            break;
                        }
                    }
                    let done_now = seen_total.min(total_snp_hint);
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (done_now, total_snp_hint))?;
                        Ok(())
                    })?;
                }
            }

            let m = if windowed_mode {
                window_sites.clear();
                window_counts.clear();
                kept_window_rows.clear();
                let mut decoded_rows = 0usize;
                for row_idx in 0..batch_len {
                    let row_sub = &mut window_rows[row_idx * n..(row_idx + 1) * n];
                    let maybe = if full_samples {
                        it.next_snp_raw_into_with_counts(row_sub)
                    } else {
                        it.next_snp_selected_raw_into_with_counts(&sample_indices, row_sub)
                    };
                    if let Some((_snp_idx, site, counts)) = maybe {
                        window_sites.push(site);
                        window_counts.push(counts);
                        decoded_rows += 1;
                    } else {
                        break;
                    }
                }
                scan_idx = scan_idx.saturating_add(batch_len);

                let keep_meta: Vec<Option<(f32, f32)>> = if decoded_rows >= 64 {
                    if let Some(p) = &pool {
                        p.install(|| {
                            window_rows[..decoded_rows * n]
                                .par_chunks_mut(n)
                                .zip(window_sites.par_iter_mut())
                                .zip(window_counts.par_iter().copied())
                                .map(|((row_sub, site), counts)| {
                                    finalize_row(row_sub, site, Some(counts)).and_then(
                                        |(maf_val, miss_val)| {
                                            if snps_only
                                                && (!is_simple_snp_allele(&site.ref_allele)
                                                    || !is_simple_snp_allele(&site.alt_allele))
                                            {
                                                None
                                            } else {
                                                Some((maf_val, miss_val))
                                            }
                                        },
                                    )
                                })
                                .collect()
                        })
                    } else {
                        (0..decoded_rows)
                            .map(|idx| {
                                let row_sub = &mut window_rows[idx * n..(idx + 1) * n];
                                let site = &mut window_sites[idx];
                                let counts = window_counts[idx];
                                finalize_row(row_sub, site, Some(counts)).and_then(
                                    |(maf_val, miss_val)| {
                                        if snps_only
                                            && (!is_simple_snp_allele(&site.ref_allele)
                                                || !is_simple_snp_allele(&site.alt_allele))
                                        {
                                            None
                                        } else {
                                            Some((maf_val, miss_val))
                                        }
                                    },
                                )
                            })
                            .collect()
                    }
                } else {
                    (0..decoded_rows)
                        .map(|idx| {
                            let row_sub = &mut window_rows[idx * n..(idx + 1) * n];
                            let site = &mut window_sites[idx];
                            let counts = window_counts[idx];
                            finalize_row(row_sub, site, Some(counts)).and_then(
                                |(maf_val, miss_val)| {
                                    if snps_only
                                        && (!is_simple_snp_allele(&site.ref_allele)
                                            || !is_simple_snp_allele(&site.alt_allele))
                                    {
                                        None
                                    } else {
                                        Some((maf_val, miss_val))
                                    }
                                },
                            )
                        })
                        .collect()
                };

                chunk_sites.clear();
                chunk_maf.clear();
                chunk_miss.clear();
                for (idx, site) in window_sites.drain(..).enumerate() {
                    if let Some((maf_val, miss_val)) = keep_meta[idx] {
                        kept_window_rows.push(idx);
                        chunk_sites.push(site);
                        chunk_maf.push(maf_val);
                        chunk_miss.push(miss_val);
                    }
                }
                window_counts.clear();
                chunk_sites.len()
            } else if let Some(p) = &pool {
                let decoded: Vec<(Vec<f32>, gfcore::SiteInfo, f32, f32)> = p.install(|| {
                    let end_idx = (scan_idx + batch_len).min(total_scan);
                    (scan_idx..end_idx)
                        .into_par_iter()
                        .filter_map(|snp_idx| {
                            let maybe = if full_samples {
                                it.get_snp_row_raw(snp_idx)
                            } else {
                                it.get_snp_row_selected_raw(snp_idx, &sample_indices)
                            };
                            maybe.and_then(|(row_sub, site)| prepare_row(row_sub, site))
                        })
                        .collect()
                });
                scan_idx = scan_idx.saturating_add(batch_len);
                chunk_rows.clear();
                chunk_sites.clear();
                chunk_maf.clear();
                chunk_miss.clear();
                for (row_sub, site, maf_val, miss_val) in decoded.into_iter() {
                    chunk_rows.extend_from_slice(&row_sub);
                    chunk_sites.push(site);
                    chunk_maf.push(maf_val);
                    chunk_miss.push(miss_val);
                }
                chunk_sites.len()
            } else {
                let mut decoded: Vec<(Vec<f32>, gfcore::SiteInfo, f32, f32)> =
                    Vec::with_capacity(batch_len);
                let end_idx = (scan_idx + batch_len).min(total_scan);
                for snp_idx in scan_idx..end_idx {
                    let maybe = if full_samples {
                        it.get_snp_row_raw(snp_idx)
                    } else {
                        it.get_snp_row_selected_raw(snp_idx, &sample_indices)
                    };
                    if let Some((row_sub, site)) = maybe {
                        if let Some(ok) = prepare_row(row_sub, site) {
                            decoded.push(ok);
                        }
                    }
                }
                scan_idx = scan_idx.saturating_add(batch_len);
                chunk_rows.clear();
                chunk_sites.clear();
                chunk_maf.clear();
                chunk_miss.clear();
                for (row_sub, site, maf_val, miss_val) in decoded.into_iter() {
                    chunk_rows.extend_from_slice(&row_sub);
                    chunk_sites.push(site);
                    chunk_maf.push(maf_val);
                    chunk_miss.push(miss_val);
                }
                chunk_sites.len()
            };
            if m == 0 {
                continue;
            }

            let out = &mut out_buf[..m * 5];
            if additive_model {
                let block_rows: &[f32] = if windowed_mode {
                    let dense_window_rows = kept_window_rows
                        .iter()
                        .take(m)
                        .enumerate()
                        .all(|(dst_idx, &src_idx)| dst_idx == src_idx);
                    if dense_window_rows {
                        &window_rows[..m * n]
                    } else {
                        chunk_rows.clear();
                        for &src_idx in kept_window_rows.iter().take(m) {
                            chunk_rows
                                .extend_from_slice(&window_rows[src_idx * n..(src_idx + 1) * n]);
                        }
                        &chunk_rows[..m * n]
                    }
                } else {
                    &chunk_rows[..m * n]
                };
                let gy_slice = &mut gy_block[..m];
                let xts_slice = &mut xts_block[..m * rhs_add_cols];
                let ss_slice = &mut row_ss[..m];
                row_major_block_mul_mat_f32_lm(
                    block_rows,
                    m,
                    n,
                    rhs_add_f32
                        .as_ref()
                        .expect("additive RHS missing")
                        .as_slice(),
                    rhs_add_cols,
                    xts_slice,
                    pool.as_ref(),
                );
                row_major_block_first_col_f32_to_f64(
                    xts_slice,
                    m,
                    rhs_add_cols,
                    gy_slice,
                    pool.as_ref(),
                );
                row_major_block_sumsq_f64(block_rows, m, n, ss_slice, pool.as_ref());
                for idx in 0..m {
                    let xts_row =
                        &xts_slice[idx * rhs_add_cols + 1..idx * rhs_add_cols + 1 + q_rank];
                    for j in 0..q_rank {
                        xts_tmp[j] = xts_row[j] as f64;
                    }
                    let stats = lm_assoc_from_qr_projection_raw(
                        &xts_tmp[..q_rank],
                        gy_slice[idx],
                        ss_slice[idx],
                        &qr_ctx,
                    );
                    out[idx * 5..(idx + 1) * 5].copy_from_slice(&stats);
                }
            } else if windowed_mode {
                let mut runner = || {
                    out.par_chunks_mut(5).enumerate().for_each_init(
                        || GlmScratch::new(q_rank),
                        |scr, (idx, row_out)| {
                            let src_idx = kept_window_rows[idx];
                            let grow = &window_rows[src_idx * n..(src_idx + 1) * n];
                            scr.reset_xs();
                            let mut gy_resid = 0.0_f64;
                            let mut ss = 0.0_f64;

                            for k in 0..n {
                                let gv = grow[k] as f64;
                                gy_resid += gv * qr_ctx.y_resid[k];
                                ss += gv * gv;
                                let qrow = &qr_ctx.q[k * q_rank..(k + 1) * q_rank];
                                for j in 0..q_rank {
                                    scr.xs[j] += qrow[j] * gv;
                                }
                            }

                            row_out.copy_from_slice(&lm_assoc_from_qr_projection(
                                &scr.xs, gy_resid, ss, &qr_ctx,
                            ));
                        },
                    );
                };
                if let Some(p) = &pool {
                    p.install(&mut runner);
                } else {
                    runner();
                }
            } else {
                let mut runner = || {
                    out.par_chunks_mut(5).enumerate().for_each_init(
                        || GlmScratch::new(q_rank),
                        |scr, (idx, row_out)| {
                            let grow = &chunk_rows[idx * n..(idx + 1) * n];
                            scr.reset_xs();
                            let mut gy_resid = 0.0_f64;
                            let mut ss = 0.0_f64;

                            for k in 0..n {
                                let gv = grow[k] as f64;
                                gy_resid += gv * qr_ctx.y_resid[k];
                                ss += gv * gv;
                                let qrow = &qr_ctx.q[k * q_rank..(k + 1) * q_rank];
                                for j in 0..q_rank {
                                    scr.xs[j] += qrow[j] * gv;
                                }
                            }

                            row_out.copy_from_slice(&lm_assoc_from_qr_projection(
                                &scr.xs, gy_resid, ss, &qr_ctx,
                            ));
                        },
                    );
                };
                if let Some(p) = &pool {
                    p.install(&mut runner);
                } else {
                    runner();
                }
            }

            let write_rows = output_row_limit
                .map(|limit| limit.saturating_sub(written_total).min(m))
                .unwrap_or(m);
            if write_rows > 0 {
                text.clear();
                let miss_counts: Vec<i64> = chunk_miss[..write_rows]
                    .iter()
                    .map(|v| v.round() as i64)
                    .collect();
                append_assoc_block_from_core_sites(
                    &mut text,
                    AssocResultLayout::PrecomputedChisqBasic3 {
                        row_stride: 5,
                        beta_col: 0,
                        se_col: 1,
                        chisq_col: 2,
                        raw_p_col: 3,
                    },
                    model.as_str(),
                    AssocCoreSitesBlock {
                        sites: &chunk_sites[..write_rows],
                        maf: &chunk_maf[..write_rows],
                        miss: AssocMissBlock::CountI64(miss_counts.as_slice()),
                    },
                    &out[..write_rows * 5],
                )
                .map_err(PyRuntimeError::new_err)?;
                send_text_buf(&writer, &mut text).map_err(PyRuntimeError::new_err)?;
                written_total = written_total.saturating_add(write_rows);
            }

            kept_total = kept_total.saturating_add(m);
        }
        if let Some(cb) = progress_callback.as_ref() {
            Python::attach(|py2| -> PyResult<()> {
                py2.check_signals()?;
                cb.call1(py2, (seen_total.min(total_snp_hint), total_snp_hint))?;
                Ok(())
            })?;
        }
        writer.finish().map_err(PyRuntimeError::new_err)?;
        Ok((kept_total, seen_total))
    })
}

#[derive(Clone, Debug)]
struct LmStreamSegment {
    start: usize,
    end: usize,
    ctx: usize,
}

#[derive(Clone, Debug)]
struct LmSourceWindow {
    chrom: String,
    start: i64,
    end: i64,
    ctx: usize,
}

fn parse_lm_segments_unbounded<'py>(
    segments: Bound<'py, PyAny>,
    n_contexts: usize,
) -> PyResult<Vec<LmStreamSegment>> {
    let seg_list = segments.cast::<PyList>().map_err(|_| {
        PyValueError::new_err("segments must be a list of (start, end, context_index) tuples")
    })?;
    let mut out = Vec::<LmStreamSegment>::with_capacity(seg_list.len());
    let mut prev_end = 0usize;
    for (idx, item) in seg_list.iter().enumerate() {
        let (start, end, ctx): (usize, usize, usize) = item.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "segments[{idx}] must be a (start, end, context_index) tuple"
            ))
        })?;
        if start > end {
            return Err(PyValueError::new_err(format!(
                "segments[{idx}] has start > end: {start} > {end}"
            )));
        }
        if ctx >= n_contexts {
            return Err(PyValueError::new_err(format!(
                "segments[{idx}] context index {ctx} exceeds x_contexts length {n_contexts}"
            )));
        }
        if start < prev_end {
            return Err(PyValueError::new_err(format!(
                "segments must be sorted and non-overlapping; segments[{idx}] starts at {start}, previous end {prev_end}"
            )));
        }
        if start < end {
            out.push(LmStreamSegment { start, end, ctx });
        }
        prev_end = end;
    }
    Ok(out)
}

fn parse_lm_source_windows<'py>(
    source_windows: Option<Bound<'py, PyAny>>,
    n_contexts: usize,
) -> PyResult<Vec<LmSourceWindow>> {
    let Some(source_windows) = source_windows else {
        return Ok(Vec::new());
    };
    let win_list = source_windows.cast::<PyList>().map_err(|_| {
        PyValueError::new_err(
            "source_windows must be None or a list of (chrom, start, end, context_index) tuples",
        )
    })?;
    let mut out = Vec::<LmSourceWindow>::with_capacity(win_list.len());
    for (idx, item) in win_list.iter().enumerate() {
        let (chrom, start, end, ctx): (String, i64, i64, usize) = item.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "source_windows[{idx}] must be a (chrom, start, end, context_index) tuple"
            ))
        })?;
        if start > end {
            return Err(PyValueError::new_err(format!(
                "source_windows[{idx}] has start > end: {start} > {end}"
            )));
        }
        if ctx >= n_contexts {
            return Err(PyValueError::new_err(format!(
                "source_windows[{idx}] context index {ctx} exceeds context length {n_contexts}"
            )));
        }
        out.push(LmSourceWindow {
            chrom,
            start,
            end,
            ctx,
        });
    }
    Ok(out)
}

/// Compact QR-projected LM streaming scan over ordered active marker segments.
///
/// This is the stage2-optimized variant.  Python passes one base design matrix,
/// one QTN covariate matrix, and per-context QTN columns to exclude; Rust keeps
/// only compact Cholesky/QR factors for each context instead of materializing Q.
#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    x_base,
    qtn_cov,
    context_exclude_qtn,
    segments,
    out_tsv,
    sample_ids=None,
    maf_threshold=0.0,
    max_missing_rate=1.0,
    het_threshold=0.02,
    snps_only=false,
    chunk_size=10000,
    threads=0,
    progress_callback=None,
    progress_every=0,
    mmap_window_mb=None,
    setup_callback=None,
    source_windows=None,
))]
pub fn lm_stream_bed_segments_compact_to_tsv<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    x_base: PyReadonlyArray2<'py, f64>,
    qtn_cov: PyReadonlyArray2<'py, f64>,
    context_exclude_qtn: Bound<'py, PyAny>,
    segments: Bound<'py, PyAny>,
    out_tsv: String,
    sample_ids: Option<Vec<String>>,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
    setup_callback: Option<Py<PyAny>>,
    source_windows: Option<Bound<'py, PyAny>>,
) -> PyResult<(usize, usize)> {
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be > 0"));
    }
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err(PyValueError::new_err(
            "maf_threshold must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err(PyValueError::new_err(
            "max_missing_rate must be within [0, 1.0]",
        ));
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 1.0]",
        ));
    }
    let window_mb = mmap_window_mb
        .filter(|v| *v > 0)
        .ok_or_else(|| PyValueError::new_err("mmap_window_mb must be provided and > 0"))?;

    let y_slice = y.as_slice()?;
    let n = y_slice.len();
    if n == 0 {
        return Err(PyValueError::new_err("y must not be empty"));
    }
    let x_base_arr = x_base.as_array();
    let qtn_arr = qtn_cov.as_array();
    if x_base_arr.ndim() != 2 || qtn_arr.ndim() != 2 {
        return Err(PyValueError::new_err("x_base and qtn_cov must be 2D"));
    }
    let base_rows = x_base_arr.shape()[0];
    let base_cols = x_base_arr.shape()[1];
    let qtn_rows = qtn_arr.shape()[0];
    let qtn_cols = qtn_arr.shape()[1];
    if base_rows != n {
        return Err(PyValueError::new_err(format!(
            "x_base rows must equal len(y): rows={base_rows}, len(y)={n}"
        )));
    }
    if qtn_rows != n {
        return Err(PyValueError::new_err(format!(
            "qtn_cov rows must equal len(y): rows={qtn_rows}, len(y)={n}"
        )));
    }
    if base_cols == 0 {
        return Err(PyValueError::new_err(
            "x_base must have at least one column",
        ));
    }

    let x_base_flat: Cow<[f64]> = match x_base.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_base_arr.iter().copied().collect()),
    };
    let qtn_flat: Cow<[f64]> = match qtn_cov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(qtn_arr.iter().copied().collect()),
    };
    let context_exclude = parse_lm_context_exclude_qtn(context_exclude_qtn, qtn_cols)?;
    if context_exclude.is_empty() {
        return Err(PyValueError::new_err(
            "context_exclude_qtn must contain at least one context",
        ));
    }

    let norm_prefix = normalize_plink_prefix_local(&prefix);
    let samples = gfcore::read_fam(&norm_prefix).map_err(PyRuntimeError::new_err)?;
    let (sample_indices, _sample_ids_ordered) =
        build_sample_selection(&samples, sample_ids, None).map_err(PyRuntimeError::new_err)?;
    if sample_indices.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "sample_ids length mismatch: selected n={} but len(y)={n}",
            sample_indices.len()
        )));
    }

    let y_raw = y_slice.to_vec();
    let yy = y_raw.iter().map(|v| v * v).sum::<f64>();
    let q_all = base_cols.saturating_add(qtn_cols);
    if let Some(cb) = setup_callback.as_ref() {
        cb.call1(
            py,
            ("Building compact QR contexts", "start", 0usize, 0usize),
        )?;
    }
    let x_all_f32 = lm_stage2_build_x_all_f32(
        x_base_flat.as_ref(),
        qtn_flat.as_ref(),
        n,
        base_cols,
        qtn_cols,
    )
    .map_err(PyValueError::new_err)?;
    let xtx_all = {
        let _blas_guard = BlasThreadGuard::enter(threads.max(1));
        lm_stage2_xtx_from_design_f64(
            x_base_flat.as_ref(),
            qtn_flat.as_ref(),
            n,
            base_cols,
            qtn_cols,
        )
        .map_err(PyRuntimeError::new_err)?
    };
    let xty_all = lm_stage2_xty_from_design(
        x_base_flat.as_ref(),
        qtn_flat.as_ref(),
        y_raw.as_slice(),
        n,
        base_cols,
        qtn_cols,
    )
    .map_err(PyRuntimeError::new_err)?;
    let compact_contexts = {
        let _blas_guard = BlasThreadGuard::enter(1);
        lm_stage2_build_compact_contexts(
            xtx_all.as_slice(),
            xty_all.as_slice(),
            yy,
            n,
            base_cols,
            qtn_cols,
            context_exclude.as_slice(),
            threads,
        )
        .map_err(PyRuntimeError::new_err)?
    };
    if let Some(cb) = setup_callback.as_ref() {
        cb.call1(
            py,
            (
                "Building compact QR contexts",
                "finish",
                compact_contexts.len(),
                q_all,
            ),
        )?;
    }
    let segment_plan = parse_lm_segments_unbounded(segments, compact_contexts.len())?;
    let source_window_plan = parse_lm_source_windows(source_windows, compact_contexts.len())?;
    let out_tsv_path = out_tsv.clone();

    py.detach(move || -> PyResult<(usize, usize)> {
        lm_stream_bed_segments_compact_windowed_unified(
            norm_prefix.as_str(),
            y_raw.as_slice(),
            x_all_f32.as_slice(),
            q_all,
            compact_contexts.as_slice(),
            segment_plan.as_slice(),
            source_window_plan.as_slice(),
            out_tsv_path.as_str(),
            sample_indices.as_slice(),
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            chunk_size,
            threads,
            progress_callback.as_ref(),
            progress_every,
            window_mb,
        )
        .map_err(PyRuntimeError::new_err)
    })
}

struct GlmScratch {
    xs: Vec<f64>, // q0
}

impl GlmScratch {
    fn new(q0: usize) -> Self {
        Self { xs: vec![0.0; q0] }
    }
    #[inline]
    fn reset_xs(&mut self) {
        self.xs.fill(0.0);
    }
}

/// Unified LM block formula using centered/residualized approach with BLAS-3.
///
/// Precomputes r_y = M_X y = y - X @ C @ (X^T @ y) where C = (X^T X)^{-1}.
/// Then decodes genotype in blocks and uses sgemm for X^T @ G_b,
/// with double buffering to overlap decode and compute.
///
/// Formula per block:
///   U = X^T @ G_b          (sgemm, f32)
///   a = G_b @ r_y          (sgemm, f32)
///   d = colsum(G_b^2)      (f64)
///   V = C @ U              (f64)
///   s = d - colsum(U ⊙ V)  (f64)
///   beta = a / s
///   se = sqrt(ve / s) where ve = (yy_r - a*beta) / (n - q0 - 1)
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
    chunk_size=10000,
    threads=0
))]
pub fn lm_block_assoc_packed<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    chunk_size: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if chunk_size == 0 {
        return Err(PyRuntimeError::new_err("chunk_size must be > 0"));
    }
    let y_slice = y.as_slice()?;
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

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = y_slice.len();
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
    let df = (n as i32) - (q0 as i32) - 1;

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

    // ---- Precompute r_y = M_X y = y - X @ C @ (X^T @ y) ----
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y_slice[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    // C @ xy = ixx @ xy -> projection weights
    let mut c_xy = vec![0.0_f64; q0];
    for i in 0..q0 {
        let mut acc = 0.0_f64;
        let row = &ixx_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            acc += row[j] * xy[j];
        }
        c_xy[i] = acc;
    }
    // r_y = y - X @ (C @ xy)
    let mut ry = vec![0.0_f64; n];
    let mut yy_r = 0.0_f64;
    for i in 0..n {
        let xrow = &x_flat[i * q0..(i + 1) * q0];
        let mut pred = 0.0_f64;
        for j in 0..q0 {
            pred += xrow[j] * c_xy[j];
        }
        let resid = y_slice[i] - pred;
        ry[i] = resid;
        yy_r += resid * resid;
    }

    // Convert r_y and X to f32 for sgemm
    let ry_f32: Vec<f32> = ry.iter().map(|&v| v as f32).collect();
    let x_f32: Vec<f32> = x_flat.iter().map(|&v| v as f32).collect();

    // Output: beta(0), se(1), pwald(2), plrt(3)
    let out = PyArray2::<f64>::zeros(py, [m, 4], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    let code4_lut = &packed_byte_lut().code4;
    let full_sample_fast = sample_idx.iter().enumerate().all(|(i, &s)| i == s);
    let pool = get_cached_pool(threads)?;
    let block_rows = chunk_size.min(m).max(1);

    py.detach(|| -> PyResult<()> {
        let runner = || -> PyResult<()> {
            let g_buf_len = block_rows * n;
            let (task_tx, task_rx) = mpsc::sync_channel::<(usize, usize, Vec<f32>)>(2);
            let (done_tx, done_rx) = mpsc::sync_channel::<(usize, usize, Vec<f32>)>(2);

            let decode_pool: Option<Arc<rayon::ThreadPool>> = None;

            thread::scope(|scope| {
                // Decode thread
                scope.spawn(move || {
                    while let Ok((start, rows, mut buf)) = task_rx.recv() {
                        let buf_slice = &mut buf[..rows * n];
                        let _ = decode_mean_imputed_additive_packed_block_rows_f32(
                            &packed_flat,
                            bytes_per_snp,
                            n_samples,
                            &row_flip,
                            &row_maf,
                            &sample_idx,
                            full_sample_fast,
                            None,
                            start,
                            buf_slice,
                            code4_lut,
                            decode_pool.as_ref(),
                        );
                        if done_tx.send((start, rows, buf)).is_err() {
                            break;
                        }
                    }
                });

                // Seed pipeline with 2 buffers
                let mut next_start = 0usize;
                let mut inflight = 0usize;
                for _ in 0..2 {
                    if next_start >= m {
                        break;
                    }
                    let rows = (m - next_start).min(block_rows);
                    task_tx
                        .send((next_start, rows, vec![0.0_f32; g_buf_len]))
                        .expect("lm block pipeline task send");
                    next_start += rows;
                    inflight += 1;
                }

                // Process completed blocks
                while inflight > 0 {
                    let (start, rows, g_buf) = done_rx.recv().expect("lm block pipeline recv");
                    inflight -= 1;

                    let g_block = &g_buf[..rows * n];

                    // ---- U = G_b @ X (rows x q0, f32) ----
                    let mut u_block = vec![0.0_f32; rows * q0];
                    if q0 > 0 {
                        row_major_block_mul_mat_f32_lm(
                            g_block,
                            rows,
                            n,
                            &x_f32,
                            q0,
                            &mut u_block,
                            pool.as_ref(),
                        );
                    }

                    // ---- a = G_b @ r_y (rows, f32) ----
                    let mut a_block = vec![0.0_f32; rows];
                    row_major_block_mul_mat_f32_lm(
                        g_block,
                        rows,
                        n,
                        &ry_f32,
                        1,
                        &mut a_block,
                        pool.as_ref(),
                    );

                    // ---- d = colsum(G_b^2) (rows, f64) ----
                    let mut d_block = vec![0.0_f64; rows];
                    row_major_block_sumsq_f64(g_block, rows, n, &mut d_block, pool.as_ref());

                    // ---- V = C @ U, s = d - colsum(U ⊙ V), per-SNP stats in f64 ----
                    for j in 0..rows {
                        // Extract U[j,:] as f64 row vector
                        let mut uj = vec![0.0_f64; q0];
                        for k in 0..q0 {
                            uj[k] = u_block[j * q0 + k] as f64;
                        }
                        // V_j = C @ U_j
                        let mut vj = vec![0.0_f64; q0];
                        for k in 0..q0 {
                            let crow = &ixx_flat[k * q0..(k + 1) * q0];
                            let mut acc = 0.0_f64;
                            for t in 0..q0 {
                                acc += crow[t] * uj[t];
                            }
                            vj[k] = acc;
                        }
                        // correction = sum_k U_j[k] * V_j[k]
                        let mut correction = 0.0_f64;
                        for k in 0..q0 {
                            correction += uj[k] * vj[k];
                        }
                        let s_j = d_block[j] - correction;

                        let a_j = a_block[j] as f64;

                        let (beta_j, se_j, pwald, plrt) = if s_j < 1e-12 || !s_j.is_finite() {
                            (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
                        } else {
                            let b = a_j / s_j;
                            let rss = (yy_r - b * a_j).max(0.0);
                            let ve = rss / (df as f64);
                            if ve <= 0.0 {
                                (b, f64::NAN, f64::NAN, f64::NAN)
                            } else {
                                let se = (ve / s_j).sqrt();
                                let t = b / se;
                                let t2 = t * t;
                                let p = student_t_p_two_sided(t, df);
                                let plrt = lm_plrt_from_t2(t2, n, df);
                                (b, se, p, plrt)
                            }
                        };

                        let out_idx = start + j;
                        out_slice[out_idx * 4] = beta_j;
                        out_slice[out_idx * 4 + 1] = se_j;
                        out_slice[out_idx * 4 + 2] = pwald;
                        out_slice[out_idx * 4 + 3] = plrt;
                    }

                    // Recycle buffer for next decode
                    if next_start < m {
                        let next_rows = (m - next_start).min(block_rows);
                        task_tx
                            .send((next_start, next_rows, g_buf))
                            .expect("lm block pipeline task resend");
                        next_start += next_rows;
                        inflight += 1;
                    }
                }

                drop(task_tx);
            });

            Ok(())
        };

        runner()
    })?;

    Ok(out)
}

/// Block LM formula with per-trait filtering and incremental TSV output.
///
/// Precomputes r_y = M_X y (residualized phenotype), then filters SNPs
/// using trait-specific MAF/missing/het computed on the selected samples.
/// Surviving SNPs are processed in blocks via sgemm (BLAS-3) with double
/// buffering to overlap decode and compute.
#[pyfunction]
#[pyo3(signature = (
    y,
    x,
    ixx,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_missing,
    chrom,
    pos,
    snp,
    allele0,
    allele1,
    out_tsv,
    maf_threshold=0.0,
    max_missing_rate=1.0,
    het_threshold=0.0,
    sample_indices=None,
    row_indices=None,
    chunk_size=10000,
    threads=0,
    progress_callback=None,
    progress_every=0,
    bed_prefix=None
))]
pub fn lm_block_assoc_packed_to_tsv<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_missing: PyReadonlyArray1<'py, f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    bed_prefix: Option<&str>,
) -> PyResult<(usize, usize)> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if chunk_size == 0 {
        return Err(PyRuntimeError::new_err("chunk_size must be > 0"));
    }
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err(PyValueError::new_err(
            "maf_threshold must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err(PyValueError::new_err(
            "max_missing_rate must be within [0, 1.0]",
        ));
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 1.0]",
        ));
    }

    let y_slice = y.as_slice()?;
    let x_arr = x.as_array();
    let ixx_arr = ixx.as_array();
    let packed_arr = packed.as_array();

    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m_packed = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;

    let row_idx: Option<Vec<usize>> = if let Some(ridx) = row_indices {
        Some(parse_index_vec_i64(
            ridx.as_slice()?,
            m_packed,
            "row_indices",
        )?)
    } else {
        None
    };
    let m = row_idx.as_ref().map(|v| v.len()).unwrap_or(m_packed);

    if row_flip.len() != m {
        return Err(PyRuntimeError::new_err(format!("row_flip length mismatch")));
    }
    if row_maf.len() != m {
        return Err(PyRuntimeError::new_err(format!("row_maf length mismatch")));
    }
    if row_missing.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_missing length mismatch"
        )));
    }
    let (chrom, pos, snp, allele0, allele1) = resolve_assoc_tsv_metadata(
        bed_prefix,
        chrom,
        pos,
        snp,
        allele0,
        allele1,
        row_idx.as_deref(),
        m,
    )
    .map_err(PyRuntimeError::new_err)?;

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = y_slice.len();
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
    let df = (n as i32) - (q0 as i32) - 1;

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

    // ---- Precompute r_y = M_X y = y - X @ ixx @ (X^T @ y) ----
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y_slice[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let mut c_xy = vec![0.0_f64; q0];
    for i in 0..q0 {
        let mut acc = 0.0_f64;
        let ixx_row = &ixx_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            acc += ixx_row[j] * xy[j];
        }
        c_xy[i] = acc;
    }
    let mut ry = vec![0.0_f64; n];
    let mut yy_r = 0.0_f64;
    for i in 0..n {
        let xrow = &x_flat[i * q0..(i + 1) * q0];
        let mut pred = 0.0_f64;
        for j in 0..q0 {
            pred += xrow[j] * c_xy[j];
        }
        let resid = y_slice[i] - pred;
        ry[i] = resid;
        yy_r += resid * resid;
    }
    let ry_f32: Vec<f32> = ry.iter().map(|&v| v as f32).collect();
    let x_f32: Vec<f32> = x_flat.iter().map(|&v| v as f32).collect();

    // Convert slices to owned Vecs for 'static lifetime in py.detach closure
    let row_flip_owned: Vec<bool> = row_flip.to_vec();
    let row_maf_owned: Vec<f32> = row_maf.to_vec();
    let row_missing_owned: Vec<f32> = row_missing.to_vec();

    let code4_lut = &packed_byte_lut().code4;
    let full_sample_fast = sample_idx.iter().enumerate().all(|(i, &s)| i == s);
    let pool = get_cached_pool(threads)?;
    let block_rows = chunk_size.min(m).max(1);

    let out_path = out_tsv.to_string();
    let progress_block = if progress_every == 0 {
        chunk_size.max(1)
    } else {
        progress_every.max(1)
    };

    py.detach(move || -> PyResult<(usize, usize)> {
        let row_flip = row_flip_owned;
        let row_maf = row_maf_owned;
        let row_missing = row_missing_owned;
        let writer = AsyncTsvWriter::with_config(
            &out_path,
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n",
            64 * 1024 * 1024,
            4,
        )
        .map_err(PyRuntimeError::new_err)?;

        let g_buf_len = block_rows * n;
        let (task_tx, task_rx) = mpsc::sync_channel::<(usize, usize, Vec<f32>)>(2);
        let (done_tx, done_rx) = mpsc::sync_channel::<(usize, usize, Vec<f32>)>(2);
        let decode_pool: Option<Arc<rayon::ThreadPool>> = None;
        let row_source_indices: Option<Vec<usize>> = row_idx.clone();

        let mut text_buf = String::with_capacity(block_rows * 112);
        let mut kept_total: usize = 0;

        thread::scope(|scope| {
            // Decode thread
            let decode_flat = packed_flat.clone();
            let decode_flip = row_flip.clone();
            let decode_maf = row_maf.clone();
            let decode_sidx = sample_idx.clone();
            let decode_row_src = row_source_indices.clone();
            scope.spawn(move || {
                while let Ok((start, rows, mut buf)) = task_rx.recv() {
                    let buf_slice = &mut buf[..rows * n];
                    let _ = decode_mean_imputed_additive_packed_block_rows_f32(
                        &decode_flat,
                        bytes_per_snp,
                        n_samples,
                        &decode_flip,
                        &decode_maf,
                        &decode_sidx,
                        full_sample_fast,
                        decode_row_src.as_deref(),
                        start,
                        buf_slice,
                        code4_lut,
                        decode_pool.as_ref(),
                    );
                    if done_tx.send((start, rows, buf)).is_err() {
                        break;
                    }
                }
            });

            // Seed pipeline with 2 buffers
            let mut next_start = 0usize;
            let mut inflight = 0usize;
            for _ in 0..2 {
                if next_start >= m {
                    break;
                }
                let rows = (m - next_start).min(block_rows);
                task_tx
                    .send((next_start, rows, vec![0.0_f32; g_buf_len]))
                    .expect("lm block tsv pipeline task send");
                next_start += rows;
                inflight += 1;
            }

            // Process completed blocks
            while inflight > 0 {
                let (start, rows, g_buf) = done_rx.recv().expect("lm block tsv pipeline recv");
                inflight -= 1;
                let g_block = &g_buf[..rows * n];

                // ---- U = G_b @ X (rows x q0, f32) ----
                let mut u_block = vec![0.0_f32; rows * q0];
                if q0 > 0 {
                    row_major_block_mul_mat_f32_lm(
                        g_block,
                        rows,
                        n,
                        &x_f32,
                        q0,
                        &mut u_block,
                        pool.as_ref(),
                    );
                }

                // ---- a = G_b @ r_y (rows, f32) ----
                let mut a_block = vec![0.0_f32; rows];
                row_major_block_mul_mat_f32_lm(
                    g_block,
                    rows,
                    n,
                    &ry_f32,
                    1,
                    &mut a_block,
                    pool.as_ref(),
                );

                // ---- d = colsum(G_b^2) (rows, f64) ----
                let mut d_block = vec![0.0_f64; rows];
                row_major_block_sumsq_f64(g_block, rows, n, &mut d_block, pool.as_ref());

                // ---- V = ixx @ U, per-SNP corrections in f64 ----
                text_buf.clear();
                let mut out_block = vec![0.0_f64; rows * 5];
                let mut miss_counts = vec![0_i64; rows];
                for j in 0..rows {
                    let mut uj = vec![0.0_f64; q0];
                    for k in 0..q0 {
                        uj[k] = u_block[j * q0 + k] as f64;
                    }
                    let mut vj = vec![0.0_f64; q0];
                    for k in 0..q0 {
                        let crow = &ixx_flat[k * q0..(k + 1) * q0];
                        let mut acc = 0.0_f64;
                        for t in 0..q0 {
                            acc += crow[t] * uj[t];
                        }
                        vj[k] = acc;
                    }
                    let mut correction = 0.0_f64;
                    for k in 0..q0 {
                        correction += uj[k] * vj[k];
                    }
                    let s_j = d_block[j] - correction;
                    let a_j = a_block[j] as f64;

                    let (beta_j, se_j, pwald, plrt) = if s_j < 1e-12 || !s_j.is_finite() {
                        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
                    } else {
                        let b = a_j / s_j;
                        let rss = (yy_r - b * a_j).max(0.0);
                        let ve = rss / (df as f64);
                        if ve <= 0.0 {
                            (b, f64::NAN, f64::NAN, f64::NAN)
                        } else {
                            let se = (ve / s_j).sqrt();
                            let t = b / se;
                            let p = student_t_p_two_sided(t, df);
                            let plrt = lm_plrt_from_t2(t * t, n, df);
                            (b, se, p, plrt)
                        }
                    };

                    let fi = start + j;
                    let chisq = if beta_j.is_finite() && se_j.is_finite() && se_j > 0.0 {
                        let t = beta_j / se_j;
                        t * t
                    } else {
                        f64::NAN
                    };
                    let pwald = sanitize_assoc_pvalue(beta_j, se_j, pwald);
                    let _ = plrt;
                    miss_counts[j] = (row_missing[fi] * n_samples as f32) as i64;
                    let base = j * 5;
                    out_block[base] = beta_j;
                    out_block[base + 1] = se_j;
                    out_block[base + 2] = chisq;
                    out_block[base + 3] = pwald;
                    out_block[base + 4] = plrt;
                }

                append_assoc_block_from_arrays(
                    &mut text_buf,
                    AssocResultLayout::PrecomputedChisqBasic3 {
                        row_stride: 5,
                        beta_col: 0,
                        se_col: 1,
                        chisq_col: 2,
                        raw_p_col: 3,
                    },
                    "add",
                    AssocArraysBlock {
                        chrom: &chrom[start..start + rows],
                        pos: &pos[start..start + rows],
                        snp: &snp[start..start + rows],
                        allele0: &allele0[start..start + rows],
                        allele1: &allele1[start..start + rows],
                        maf: &row_maf[start..start + rows],
                        miss: AssocMissBlock::CountI64(miss_counts.as_slice()),
                    },
                    out_block.as_slice(),
                )
                .expect("LM packed TSV formatting should be length-consistent");
                writer
                    .send(std::mem::take(&mut text_buf).into_bytes())
                    .expect("lm block tsv writer send");
                kept_total += rows;

                // Progress callback (best-effort inside thread::scope)
                if let Some(ref cb) = progress_callback {
                    if kept_total % progress_block == 0 || start + rows >= m {
                        let _ = Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (kept_total.min(m), m))?;
                            Ok(())
                        });
                    }
                }

                // Recycle buffer for next decode
                if next_start < m {
                    let next_rows = (m - next_start).min(block_rows);
                    task_tx
                        .send((next_start, next_rows, g_buf))
                        .expect("lm block tsv pipeline task resend");
                    next_start += next_rows;
                    inflight += 1;
                }
            }

            drop(task_tx);
        });

        if let Some(ref cb) = progress_callback {
            Python::attach(|py2| -> PyResult<()> {
                py2.check_signals()?;
                cb.call1(py2, (m, m))?;
                Ok(())
            })?;
        }

        writer.finish().map_err(PyRuntimeError::new_err)?;
        Ok((kept_total, m))
    })
}

/// Dense block LM formula using the same residualized-scale Schur complement
/// as `lm_block_assoc_packed`, but taking an already-decoded SNP-major matrix.
///
/// Returns columns: beta, se, pwald, plrt.
#[pyfunction]
#[pyo3(signature = (y, x, ixx, g, chunk_size=10000, threads=0))]
pub fn lm_block_assoc_f32<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray2<'py, f32>,
    chunk_size: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if chunk_size == 0 {
        return Err(PyRuntimeError::new_err("chunk_size must be > 0"));
    }
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
        return Err(PyRuntimeError::new_err("g must be shape (m, n)"));
    }
    if n <= q0 + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n too small: require n > q0+1, got n={n}, q0={q0}"
        )));
    }
    let m = g_arr.shape()[0];
    let df = (n as i32) - (q0 as i32) - 1;

    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let ixx_flat: Cow<[f64]> = match ixx.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ixx_arr.iter().copied().collect()),
    };

    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let mut c_xy = vec![0.0_f64; q0];
    for i in 0..q0 {
        let mut acc = 0.0_f64;
        let row = &ixx_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            acc += row[j] * xy[j];
        }
        c_xy[i] = acc;
    }
    let mut ry = vec![0.0_f64; n];
    let mut yy_r = 0.0_f64;
    for i in 0..n {
        let xrow = &x_flat[i * q0..(i + 1) * q0];
        let mut pred = 0.0_f64;
        for j in 0..q0 {
            pred += xrow[j] * c_xy[j];
        }
        let resid = y[i] - pred;
        ry[i] = resid;
        yy_r += resid * resid;
    }

    let x_f32: Vec<f32> = x_flat.iter().map(|&v| v as f32).collect();
    let ry_f32: Vec<f32> = ry.iter().map(|&v| v as f32).collect();
    let g_slice_opt = g.as_slice().ok();

    let out = PyArray2::<f64>::zeros(py, [m, 4], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    let pool = get_cached_pool(threads)?;
    let block_rows = chunk_size.min(m).max(1);

    py.detach(|| -> PyResult<()> {
        let mut runner = || -> PyResult<()> {
            let mut g_buf = if g_slice_opt.is_some() {
                Vec::<f32>::new()
            } else {
                vec![0.0_f32; block_rows * n]
            };
            let mut u_block = vec![0.0_f32; block_rows * q0];
            let mut a_block = vec![0.0_f32; block_rows];
            let mut d_block = vec![0.0_f64; block_rows];
            let mut i_marker = 0usize;
            while i_marker < m {
                let rows = (m - i_marker).min(block_rows);
                let g_block: &[f32] = if let Some(gs) = g_slice_opt {
                    &gs[i_marker * n..(i_marker + rows) * n]
                } else {
                    let dst = &mut g_buf[..rows * n];
                    for r in 0..rows {
                        for c in 0..n {
                            dst[r * n + c] = g_arr[(i_marker + r, c)];
                        }
                    }
                    dst
                };

                if q0 > 0 {
                    row_major_block_mul_mat_f32_lm(
                        g_block,
                        rows,
                        n,
                        &x_f32,
                        q0,
                        &mut u_block[..rows * q0],
                        pool.as_ref(),
                    );
                }
                row_major_block_mul_mat_f32_lm(
                    g_block,
                    rows,
                    n,
                    &ry_f32,
                    1,
                    &mut a_block[..rows],
                    pool.as_ref(),
                );
                row_major_block_sumsq_f64(g_block, rows, n, &mut d_block[..rows], pool.as_ref());

                for j in 0..rows {
                    let mut correction = 0.0_f64;
                    for k in 0..q0 {
                        let uk = u_block[j * q0 + k] as f64;
                        let row = &ixx_flat[k * q0..(k + 1) * q0];
                        let mut vk = 0.0_f64;
                        for t in 0..q0 {
                            vk += row[t] * (u_block[j * q0 + t] as f64);
                        }
                        correction += uk * vk;
                    }
                    let schur = d_block[j] - correction;
                    let out_idx = (i_marker + j) * 4;
                    if !(schur.is_finite() && schur > 1e-12) {
                        out_slice[out_idx..out_idx + 4].fill(f64::NAN);
                        continue;
                    }
                    let a_j = a_block[j] as f64;
                    let beta = a_j / schur;
                    let rss = (yy_r - beta * a_j).max(0.0);
                    let ve = rss / (df as f64);
                    if !(ve.is_finite() && ve > 0.0) {
                        out_slice[out_idx] = beta;
                        out_slice[out_idx + 1] = f64::NAN;
                        out_slice[out_idx + 2] = f64::NAN;
                        out_slice[out_idx + 3] = f64::NAN;
                        continue;
                    }
                    let se = (ve / schur).sqrt();
                    if !(beta.is_finite() && se.is_finite() && se > 0.0) {
                        out_slice[out_idx..out_idx + 4].fill(f64::NAN);
                        continue;
                    }
                    let t = beta / se;
                    out_slice[out_idx] = beta;
                    out_slice[out_idx + 1] = se;
                    out_slice[out_idx + 2] = student_t_p_two_sided(t, df);
                    out_slice[out_idx + 3] = lm_plrt_from_t2(t * t, n, df);
                }
                i_marker += rows;
            }
            Ok(())
        };

        if let Some(p) = &pool {
            p.install(runner)
        } else {
            runner()
        }
    })?;

    Ok(out)
}
