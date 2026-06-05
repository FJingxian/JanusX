use memmap2::Mmap;
use nalgebra::DMatrix;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::thread;

use crate::bedmath::{
    adaptive_grm_block_rows, decode_mean_imputed_additive_packed_block_rows_f32,
    decode_plink_bed_hardcall, is_identity_indices, packed_byte_lut,
};
use crate::blas::{
    cblas_sgemm_dispatch, rust_sgemm_prefers_rayon_rowmajor_f32_kernel, BlasThreadGuard, CblasInt,
    CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use crate::gfcore;
use crate::gfcore::BedSnpIter;
use crate::gfreader::{
    build_sample_selection, prepare_bed_logic_meta_owned_for_stats_samples,
};
use crate::gload::{GenotypeMatrix, UnifiedInput};
use crate::he::row_major_block_mul_mat_f32;
use crate::linalg::format_chisq_value;
use crate::stats_common::{get_cached_pool, parse_index_vec_i64, AsyncTsvWriter};

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

fn ixx_from_x_qr(x_flat: &[f64], n: usize, q0: usize) -> Result<Vec<f64>, String> {
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

#[pyfunction]
#[pyo3(signature = (y, x))]
pub fn glm_ixx_from_x_qr<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y_slice = y.as_slice()?;
    let x_arr = x.as_array();
    if x_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("x must be a 2D matrix"));
    }
    let n = y_slice.len();
    let xn = x_arr.shape()[0];
    let q0 = x_arr.shape()[1];
    if xn != n {
        return Err(PyRuntimeError::new_err(format!(
            "x rows must equal len(y): rows={}, len(y)={n}",
            xn
        )));
    }
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let ixx = ixx_from_x_qr(x_flat.as_ref(), n, q0).map_err(PyRuntimeError::new_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((q0, q0), ixx)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr).into_bound())
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
fn lm_plrt_from_t2(t2: f64, n_obs: usize, df: i32) -> f64 {
    if df <= 0 || !t2.is_finite() || t2 < 0.0 {
        return f64::NAN;
    }
    let stat = (n_obs as f64) * (1.0 + t2 / (df as f64)).ln();
    chi2_sf_df1(stat)
}

#[inline]
fn normalize_plink_prefix_local(prefix: &str) -> String {
    let s = prefix.trim();
    let low = s.to_ascii_lowercase();
    if low.ends_with(".bed") || low.ends_with(".bim") || low.ends_with(".fam") {
        s[..s.len() - 4].to_string()
    } else {
        s.to_string()
    }
}

#[inline]
fn is_simple_snp_allele(a: &str) -> bool {
    let t = a.trim().to_ascii_uppercase();
    if t.len() != 1 {
        return false;
    }
    matches!(t.as_bytes()[0], b'A' | b'C' | b'G' | b'T')
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
fn transform_alleles_by_model(a0: &str, a1: &str, model: &str) -> (String, String) {
    if model == "add" {
        return (a0.to_string(), a1.to_string());
    }
    let hom0 = format!("{a0}{a0}");
    let het = format!("{a0}{a1}");
    let hom1 = format!("{a1}{a1}");
    match model {
        "dom" => (hom0, format!("{het}/{hom1}")),
        "rec" => (format!("{het}/{hom0}"), hom1),
        "het" => (format!("{hom0}/{hom1}"), het),
        _ => (a0.to_string(), a1.to_string()),
    }
}

#[inline]
fn decode_plink_dosage_with_mean(code: u8, mean_g: f64, flip: bool) -> f64 {
    if !flip {
        match code {
            0b00 => 0.0,
            0b10 => 1.0,
            0b11 => 2.0,
            _ => mean_g, // 0b01 => missing
        }
    } else {
        match code {
            0b00 => 2.0,
            0b10 => 1.0,
            0b11 => 0.0,
            _ => mean_g, // 0b01 => missing
        }
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
fn lm_assoc_from_centered_projection(
    xts: &[f64],
    sy: f64,
    ss: f64,
    ixx: &[f64],
    xy: &[f64],
    xy_quad: f64,
    q0: usize,
    n: usize,
    yy: f64,
    b21: &mut [f64],
) -> [f64; 5] {
    xs_t_ixx_into(xts, ixx, q0, b21);
    let x_quad = dot(b21, xts);
    let b22 = ss - x_quad;
    let df = (n as i32) - (q0 as i32) - 1;
    if !(b22.is_finite() && b22 > 1e-8) || df <= 0 {
        return [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    }

    let numer = sy - dot(b21, xy);
    let invb22 = 1.0_f64 / b22;
    let beta_snp = numer * invb22;
    let beta_rhs = xy_quad + (numer * beta_snp);
    let ve = (yy - beta_rhs) / (df as f64);
    if !(ve.is_finite() && ve > 0.0) {
        return [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    }

    let se_snp = (invb22 * ve).sqrt();
    if !beta_snp.is_finite() || !se_snp.is_finite() || se_snp <= 0.0 {
        return [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    }
    let t = beta_snp / se_snp;
    let stat = (n as f64) * (1.0 + (t * t) / (df as f64)).ln();
    let pwald = student_t_p_two_sided(t, df);
    let plrt = lm_plrt_from_t2(t * t, n, df);
    [beta_snp, se_snp, stat, pwald, plrt]
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
    let ixx_flat: Cow<[f64]> = if let Some(ixx_in) = ixx {
        let ixx_arr = ixx_in.as_array();
        if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
            return Err(PyRuntimeError::new_err("ixx must be (q0,q0)"));
        }
        match ixx_in.as_slice() {
            Ok(s) => Cow::Owned(s.to_vec()),
            Err(_) => Cow::Owned(ixx_arr.iter().copied().collect()),
        }
    } else {
        Cow::Owned(ixx_from_x_qr(x_flat.as_ref(), n, q0).map_err(PyRuntimeError::new_err)?)
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
    let dim = q0 + 1;
    let additive_model = model.as_str() == "add";
    let y_f32_add = if additive_model {
        Some(cast_f64_slice_to_f32(y).map_err(PyRuntimeError::new_err)?)
    } else {
        None
    };
    let x_f32_add = if additive_model {
        Some(cast_f64_slice_to_f32(x_flat.as_ref()).map_err(PyRuntimeError::new_err)?)
    } else {
        None
    };
    let mut xy_c_proj = vec![0.0_f64; q0];
    xs_t_ixx_into(&xy, &ixx_flat, q0, &mut xy_c_proj);
    let xy_quad = dot(&xy_c_proj, &xy);

    let norm_prefix = normalize_plink_prefix_local(&prefix);
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
            b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
            64 * 1024 * 1024,
            4,
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
        let mut sy_block: Vec<f32> = vec![0.0_f32; chunk_size];
        let mut xts_block: Vec<f32> = vec![0.0_f32; chunk_size * q0];
        let mut row_ss: Vec<f64> = vec![0.0_f64; chunk_size];
        let mut xts_tmp: Vec<f64> = vec![0.0_f64; q0];
        let mut b21_tmp: Vec<f64> = vec![0.0_f64; q0];
        let mut kept_total: usize = 0;
        let mut seen_total: usize = 0;
        let mut scan_idx: usize = 0;
        let total_scan = total_snp_hint;
        let mut next_progress_emit: usize = progress_block;
        let code4_lut = &packed_byte_lut().code4;
        let _blas_guard = if additive_model {
            let blas_threads = if rust_sgemm_prefers_rayon_rowmajor_f32_kernel() {
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
                gfcore::process_snp_row_with_precomputed_counts(
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
                gfcore::process_snp_row_with_stats(
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
                stats.maf
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
                adaptive_grm_block_rows(progress_block.max(1), kept_rows_total, n, 0usize, threads)
                    .max(1);
            let mut block = vec![0.0_f32; scan_block_rows * n];
            let mut row_start = 0usize;
            while row_start < kept_rows_total {
                let row_end = (row_start + scan_block_rows).min(kept_rows_total);
                let rows_here = row_end - row_start;
                let block_slice = &mut block[..rows_here * n];
                let sy_slice = &mut sy_block[..rows_here];
                let xts_slice = &mut xts_block[..rows_here * q0];
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
                row_major_block_mul_mat_f32(
                    block_slice,
                    rows_here,
                    n,
                    y_f32_add
                        .as_ref()
                        .expect("additive y_f32 missing")
                        .as_slice(),
                    1usize,
                    sy_slice,
                    pool.as_ref(),
                );
                row_major_block_mul_mat_f32(
                    block_slice,
                    rows_here,
                    n,
                    x_f32_add
                        .as_ref()
                        .expect("additive x_f32 missing")
                        .as_slice(),
                    q0,
                    xts_slice,
                    pool.as_ref(),
                );
                row_major_block_sumsq_f64(block_slice, rows_here, n, ss_slice, pool.as_ref());
                for local_idx in 0..rows_here {
                    let xts_row = &xts_slice[local_idx * q0..(local_idx + 1) * q0];
                    for j in 0..q0 {
                        xts_tmp[j] = xts_row[j] as f64;
                    }
                    let stats = lm_assoc_from_centered_projection(
                        &xts_tmp,
                        sy_slice[local_idx] as f64,
                        ss_slice[local_idx],
                        &ixx_flat,
                        &xy,
                        xy_quad,
                        q0,
                        n,
                        yy,
                        &mut b21_tmp,
                    );
                    out_slice[local_idx * 5..(local_idx + 1) * 5].copy_from_slice(&stats);
                }
                text.clear();
                for local_idx in 0..rows_here {
                    let row_idx = row_start + local_idx;
                    let site = &prepared.sites[row_idx];
                    let r = &out_slice[local_idx * 5..(local_idx + 1) * 5];
                    let chisq_txt = format_chisq_value(r[2]);
                    let miss_count =
                        (prepared.missing_rate[row_idx] as f64 * n as f64).round() as i64;
                    let _ = write!(
                        text,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                        site.chrom,
                        site.pos,
                        site.snp,
                        site.ref_allele,
                        site.alt_allele,
                        prepared.maf[row_idx],
                        miss_count,
                        r[0],
                        r[1],
                        chisq_txt,
                        r[3],
                        r[4]
                    );
                }
                let payload = std::mem::take(&mut text).into_bytes();
                writer.send(payload).map_err(PyRuntimeError::new_err)?;
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
                let sy_slice = &mut sy_block[..m];
                let xts_slice = &mut xts_block[..m * q0];
                let ss_slice = &mut row_ss[..m];
                row_major_block_mul_mat_f32(
                    block_rows,
                    m,
                    n,
                    y_f32_add
                        .as_ref()
                        .expect("additive y_f32 missing")
                        .as_slice(),
                    1usize,
                    sy_slice,
                    pool.as_ref(),
                );
                row_major_block_mul_mat_f32(
                    block_rows,
                    m,
                    n,
                    x_f32_add
                        .as_ref()
                        .expect("additive x_f32 missing")
                        .as_slice(),
                    q0,
                    xts_slice,
                    pool.as_ref(),
                );
                row_major_block_sumsq_f64(block_rows, m, n, ss_slice, pool.as_ref());
                for idx in 0..m {
                    let xts_row = &xts_slice[idx * q0..(idx + 1) * q0];
                    for j in 0..q0 {
                        xts_tmp[j] = xts_row[j] as f64;
                    }
                    let stats = lm_assoc_from_centered_projection(
                        &xts_tmp,
                        sy_slice[idx] as f64,
                        ss_slice[idx],
                        &ixx_flat,
                        &xy,
                        xy_quad,
                        q0,
                        n,
                        yy,
                        &mut b21_tmp,
                    );
                    out[idx * 5..(idx + 1) * 5].copy_from_slice(&stats);
                }
            } else if windowed_mode {
                let mut runner = || {
                    out.par_chunks_mut(5).enumerate().for_each_init(
                        || GlmScratch::new(q0),
                        |scr, (idx, row_out)| {
                            let src_idx = kept_window_rows[idx];
                            let grow = &window_rows[src_idx * n..(src_idx + 1) * n];
                            scr.reset_xs();
                            let mut sy = 0.0_f64;
                            let mut ss = 0.0_f64;

                            for k in 0..n {
                                let gv = grow[k] as f64;
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
                            let beta_snp = scr.beta[q0];
                            let se_snp = (scr.ixxs[q0 * dim + q0] * ve).sqrt();
                            if invb22 == 0.0
                                || !beta_snp.is_finite()
                                || !se_snp.is_finite()
                                || se_snp <= 0.0
                            {
                                row_out.fill(f64::NAN);
                                return;
                            }
                            let t = beta_snp / se_snp;
                            let pwald = student_t_p_two_sided(t, df);
                            let stat = (n as f64) * (1.0 + (t * t) / (df as f64)).ln();
                            let plrt = lm_plrt_from_t2(t * t, n, df);
                            row_out[0] = beta_snp;
                            row_out[1] = se_snp;
                            row_out[2] = stat;
                            row_out[3] = pwald;
                            row_out[4] = plrt;
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
                        || GlmScratch::new(q0),
                        |scr, (idx, row_out)| {
                            let grow = &chunk_rows[idx * n..(idx + 1) * n];
                            scr.reset_xs();
                            let mut sy = 0.0_f64;
                            let mut ss = 0.0_f64;

                            for k in 0..n {
                                let gv = grow[k] as f64;
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
                            let beta_snp = scr.beta[q0];
                            let se_snp = (scr.ixxs[q0 * dim + q0] * ve).sqrt();
                            if invb22 == 0.0
                                || !beta_snp.is_finite()
                                || !se_snp.is_finite()
                                || se_snp <= 0.0
                            {
                                row_out.fill(f64::NAN);
                                return;
                            }
                            let t = beta_snp / se_snp;
                            let pwald = student_t_p_two_sided(t, df);
                            let stat = (n as f64) * (1.0 + (t * t) / (df as f64)).ln();
                            let plrt = lm_plrt_from_t2(t * t, n, df);
                            row_out[0] = beta_snp;
                            row_out[1] = se_snp;
                            row_out[2] = stat;
                            row_out[3] = pwald;
                            row_out[4] = plrt;
                        },
                    );
                };
                if let Some(p) = &pool {
                    p.install(&mut runner);
                } else {
                    runner();
                }
            }

            text.clear();
            for i in 0..m {
                let site = &chunk_sites[i];
                let (a0, a1) =
                    transform_alleles_by_model(&site.ref_allele, &site.alt_allele, model.as_str());
                let r = &out[i * 5..(i + 1) * 5];
                let chisq_txt = format_chisq_value(r[2]);
                let _ = write!(
                    text,
                    "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                    site.chrom,
                    site.pos,
                    site.snp,
                    a0,
                    a1,
                    chunk_maf[i],
                    chunk_miss[i].round() as i64,
                    r[0],
                    r[1],
                    chisq_txt,
                    r[3],
                    r[4]
                );
            }
            let payload = std::mem::take(&mut text).into_bytes();
            writer.send(payload).map_err(PyRuntimeError::new_err)?;

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
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn glmf32_packed_assoc<'py>(
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
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
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

    // Columns: beta, se, pwald, plrt
    let out = PyArray2::<f64>::zeros(py, [m, 4], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    let pool = get_cached_pool(threads)?;
    let progress_block = if progress_every == 0 {
        step.max(1)
    } else {
        progress_every.max(1)
    };
    py.detach(move || -> PyResult<()> {
        let mut runner = || -> PyResult<()> {
            let mut i_marker = 0usize;
            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);
                let block = &mut out_slice[i_marker * 4..(i_marker + cnt) * 4];
                block.par_chunks_mut(4).enumerate().for_each_init(
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
                        let beta_snp = scr.beta[q0];
                        let se_snp = (scr.ixxs[q0 * dim + q0] * ve).sqrt();

                        if invb22 == 0.0
                            || !beta_snp.is_finite()
                            || !se_snp.is_finite()
                            || se_snp <= 0.0
                        {
                            row_out.fill(f64::NAN);
                            return;
                        }

                        let t = beta_snp / se_snp;
                        let pwald = student_t_p_two_sided(t, df);
                        let plrt = lm_plrt_from_t2(t * t, n, df);
                        row_out[0] = beta_snp;
                        row_out[1] = se_snp;
                        row_out[2] = pwald;
                        row_out[3] = plrt;
                    },
                );
                let done = (i_marker + cnt).min(m);
                if done % progress_block == 0 || done == m {
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done, m))?;
                            Ok(())
                        })?;
                    }
                }
                i_marker += cnt;
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

    let _blas_guard = BlasThreadGuard::enter(threads.max(1));

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

                    // ---- U = X^T @ G_b (q0 x rows, f32) ----
                    let mut u_block = vec![0.0_f32; q0 * rows];
                    if q0 > 0 {
                        unsafe {
                            cblas_sgemm_dispatch(
                                CBLAS_ROW_MAJOR,
                                CBLAS_TRANS,
                                CBLAS_TRANS,
                                q0 as CblasInt,
                                rows as CblasInt,
                                n as CblasInt,
                                1.0_f32,
                                x_f32.as_ptr(),
                                q0 as CblasInt,
                                g_block.as_ptr(),
                                n as CblasInt,
                                0.0_f32,
                                u_block.as_mut_ptr(),
                                rows as CblasInt,
                            );
                        }
                    }

                    // ---- a = G_b @ r_y (rows, f32) ----
                    let mut a_block = vec![0.0_f32; rows];
                    unsafe {
                        cblas_sgemm_dispatch(
                            CBLAS_ROW_MAJOR,
                            CBLAS_NO_TRANS,
                            CBLAS_NO_TRANS,
                            rows as CblasInt,
                            1 as CblasInt,
                            n as CblasInt,
                            1.0_f32,
                            g_block.as_ptr(),
                            n as CblasInt,
                            ry_f32.as_ptr(),
                            1 as CblasInt,
                            0.0_f32,
                            a_block.as_mut_ptr(),
                            1 as CblasInt,
                        );
                    }

                    // ---- d = colsum(G_b^2) (rows, f64) ----
                    let mut d_block = vec![0.0_f64; rows];
                    for j in 0..rows {
                        let row_g = &g_block[j * n..(j + 1) * n];
                        let mut ss = 0.0_f64;
                        for &v in row_g {
                            let vf = v as f64;
                            ss += vf * vf;
                        }
                        d_block[j] = ss;
                    }

                    // ---- V = C @ U, s = d - colsum(U ⊙ V), per-SNP stats in f64 ----
                    for j in 0..rows {
                        // Extract U[:,j] as f64 column vector
                        let mut uj = vec![0.0_f64; q0];
                        for k in 0..q0 {
                            uj[k] = u_block[k * rows + j] as f64;
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
    progress_every=0
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
    if chrom.len() != m
        || pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
    {
        return Err(PyRuntimeError::new_err(format!(
            "TSV metadata length mismatch"
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
    let _blas_guard = BlasThreadGuard::enter(threads.max(1));
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
            b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
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

                // ---- U = X^T @ G_b (q0 x rows, f32) ----
                let mut u_block = vec![0.0_f32; q0 * rows];
                if q0 > 0 {
                    unsafe {
                        cblas_sgemm_dispatch(
                            CBLAS_ROW_MAJOR,
                            CBLAS_TRANS,
                            CBLAS_TRANS,
                            q0 as CblasInt,
                            rows as CblasInt,
                            n as CblasInt,
                            1.0_f32,
                            x_f32.as_ptr(),
                            q0 as CblasInt,
                            g_block.as_ptr(),
                            n as CblasInt,
                            0.0_f32,
                            u_block.as_mut_ptr(),
                            rows as CblasInt,
                        );
                    }
                }

                // ---- a = G_b @ r_y (rows, f32) ----
                let mut a_block = vec![0.0_f32; rows];
                unsafe {
                    cblas_sgemm_dispatch(
                        CBLAS_ROW_MAJOR,
                        CBLAS_NO_TRANS,
                        CBLAS_NO_TRANS,
                        rows as CblasInt,
                        1 as CblasInt,
                        n as CblasInt,
                        1.0_f32,
                        g_block.as_ptr(),
                        n as CblasInt,
                        ry_f32.as_ptr(),
                        1 as CblasInt,
                        0.0_f32,
                        a_block.as_mut_ptr(),
                        1 as CblasInt,
                    );
                }

                // ---- d = colsum(G_b^2) (rows, f64) ----
                let mut d_block = vec![0.0_f64; rows];
                for j in 0..rows {
                    let row_g = &g_block[j * n..(j + 1) * n];
                    let mut ss = 0.0_f64;
                    for &v in row_g {
                        let vf = v as f64;
                        ss += vf * vf;
                    }
                    d_block[j] = ss;
                }

                // ---- V = ixx @ U, per-SNP corrections in f64 ----
                text_buf.clear();
                for j in 0..rows {
                    let mut uj = vec![0.0_f64; q0];
                    for k in 0..q0 {
                        uj[k] = u_block[k * rows + j] as f64;
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
                    let row_maf_val = row_maf[fi];
                    let miss_count = (row_missing[fi] * n_samples as f32) as i64;
                    let chisq = if beta_j.is_finite() && se_j.is_finite() && se_j > 0.0 {
                        let t = beta_j / se_j;
                        t * t
                    } else {
                        f64::NAN
                    };
                    let chisq_txt = format_chisq_value(chisq);

                    let _ = write!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                        chrom[fi],
                        pos[fi],
                        snp[fi],
                        allele0[fi],
                        allele1[fi],
                        row_maf_val,
                        miss_count,
                        beta_j,
                        se_j,
                        chisq_txt,
                        pwald,
                        plrt,
                    );
                }

                writer
                    .send(text_buf.as_bytes().to_vec())
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
    sample_indices=None,
    row_indices=None,
    step=10000,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn glmf32_packed_assoc_to_tsv(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ixx: Option<PyReadonlyArray2<'_, f64>>,
    packed: PyReadonlyArray2<'_, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'_, bool>,
    row_maf: PyReadonlyArray1<'_, f32>,
    row_missing: PyReadonlyArray1<'_, f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: &str,
    sample_indices: Option<PyReadonlyArray1<'_, i64>>,
    row_indices: Option<PyReadonlyArray1<'_, i64>>,
    step: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<usize> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let y = y.as_slice()?;
    let x_arr = x.as_array();
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
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
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
    if row_missing.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_missing length mismatch: got {}, expected {m}",
            row_missing.len()
        )));
    }
    if chrom.len() != m
        || pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
    {
        return Err(PyRuntimeError::new_err(format!(
            "TSV metadata length mismatch: rows={m}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        )));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let sample_idx_is_identity = is_identity_indices(&sample_idx, n_samples);
    let (sample_byte_idx, sample_lane_idx): (Vec<usize>, Vec<usize>) = if sample_idx_is_identity {
        (Vec::new(), Vec::new())
    } else {
        let mut byte_idx = Vec::with_capacity(sample_idx.len());
        let mut lane_idx = Vec::with_capacity(sample_idx.len());
        for &s in &sample_idx {
            byte_idx.push(s >> 2);
            lane_idx.push(s & 3);
        }
        (byte_idx, lane_idx)
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
    if n <= q0 + 1 {
        return Err(PyRuntimeError::new_err(format!(
            "n too small: require n > q0+1, got n={n}, q0={q0}"
        )));
    }

    let dim = q0 + 1;
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let ixx_flat: Cow<[f64]> = if let Some(ixx_in) = ixx {
        let ixx_arr = ixx_in.as_array();
        if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
            return Err(PyRuntimeError::new_err("ixx must be (q0,q0)"));
        }
        match ixx_in.as_slice() {
            Ok(s) => Cow::Owned(s.to_vec()),
            Err(_) => Cow::Owned(ixx_arr.iter().copied().collect()),
        }
    } else {
        Cow::Owned(ixx_from_x_qr(x_flat.as_ref(), n, q0).map_err(PyRuntimeError::new_err)?)
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

    let out_path = out_tsv.to_string();

    let pool = get_cached_pool(threads)?;
    let step = step.max(1);
    let progress_block = if progress_every == 0 {
        step
    } else {
        progress_every.max(1)
    };
    py.detach(move || -> PyResult<usize> {
        let writer = AsyncTsvWriter::with_config(
            &out_path,
            b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
            64 * 1024 * 1024,
            4,
        )
        .map_err(PyRuntimeError::new_err)?;

        let mut out_block = vec![0.0_f64; step * 5];
        let mut miss_block = vec![0usize; step];
        let mut text_buf = String::with_capacity(step * 96);
        let code4_lut = &packed_byte_lut().code4;

        let mut runner = || -> PyResult<()> {
            let mut i_marker = 0usize;
            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);
                let block = &mut out_block[..cnt * 5];
                let miss_sub = &mut miss_block[..cnt];
                block
                    .par_chunks_mut(5)
                    .zip(miss_sub.par_iter_mut())
                    .enumerate()
                    .for_each_init(
                        || GlmScratch::new(q0),
                        |scr, (l, (row_out, miss_out))| {
                            let idx = i_marker + l;
                            let src_row = row_idx.as_ref().map(|v| v[idx]).unwrap_or(idx);
                            scr.reset_xs();

                            let row = &packed_flat
                                [src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                            let flip = row_flip[idx];
                            let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);

                            let mut sy = 0.0_f64;
                            let mut ss = 0.0_f64;
                            let mut missing_ct = 0usize;
                            if sample_idx_is_identity {
                                let full_bytes = n_samples / 4;
                                let rem = n_samples % 4;
                                let mut k = 0usize;
                                for &b in row.iter().take(full_bytes) {
                                    let codes = &code4_lut[b as usize];
                                    for &code in codes.iter().take(4) {
                                        if code == 0b01 {
                                            missing_ct += 1;
                                        }
                                        let gv = decode_plink_dosage_with_mean(code, mean_g, flip);
                                        sy += gv * y[k];
                                        ss += gv * gv;
                                        let xrow = &x_flat[k * q0..(k + 1) * q0];
                                        for j in 0..q0 {
                                            scr.xs[j] += xrow[j] * gv;
                                        }
                                        k += 1;
                                    }
                                }
                                if rem > 0 {
                                    let codes = &code4_lut[row[full_bytes] as usize];
                                    for &code in codes.iter().take(rem) {
                                        if code == 0b01 {
                                            missing_ct += 1;
                                        }
                                        let gv = decode_plink_dosage_with_mean(code, mean_g, flip);
                                        sy += gv * y[k];
                                        ss += gv * gv;
                                        let xrow = &x_flat[k * q0..(k + 1) * q0];
                                        for j in 0..q0 {
                                            scr.xs[j] += xrow[j] * gv;
                                        }
                                        k += 1;
                                    }
                                }
                            } else {
                                for k in 0..n {
                                    let b = row[sample_byte_idx[k]];
                                    let code = code4_lut[b as usize][sample_lane_idx[k]];
                                    if code == 0b01 {
                                        missing_ct += 1;
                                    }
                                    let gv = decode_plink_dosage_with_mean(code, mean_g, flip);

                                    sy += gv * y[k];
                                    ss += gv * gv;
                                    let xrow = &x_flat[k * q0..(k + 1) * q0];
                                    for j in 0..q0 {
                                        scr.xs[j] += xrow[j] * gv;
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
                            let beta_snp = scr.beta[q0];
                            let se_snp = (scr.ixxs[q0 * dim + q0] * ve).sqrt();

                            if invb22 == 0.0
                                || !beta_snp.is_finite()
                                || !se_snp.is_finite()
                                || se_snp <= 0.0
                            {
                                *miss_out = missing_ct;
                                row_out.fill(f64::NAN);
                                return;
                            }

                            let t = beta_snp / se_snp;
                            let pwald = student_t_p_two_sided(t, df);
                            let stat = (n as f64) * (1.0 + (t * t) / (df as f64)).ln();
                            let plrt = lm_plrt_from_t2(t * t, n, df);
                            row_out[0] = beta_snp;
                            row_out[1] = se_snp;
                            row_out[2] = stat;
                            row_out[3] = pwald;
                            row_out[4] = plrt;
                            *miss_out = missing_ct;
                        },
                    );

                text_buf.clear();
                if text_buf.capacity() < cnt * 96 {
                    text_buf.reserve(cnt * 96 - text_buf.capacity());
                }
                for l in 0..cnt {
                    let idx = i_marker + l;
                    let base = l * 5;
                    let chisq_txt = format_chisq_value(block[base + 2]);
                    let _ = write!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                        chrom[idx],
                        pos[idx],
                        snp[idx],
                        allele0[idx],
                        allele1[idx],
                        row_maf[idx],
                        miss_block[l],
                        block[base],
                        block[base + 1],
                        chisq_txt,
                        block[base + 3],
                        block[base + 4],
                    );
                }
                let payload = std::mem::take(&mut text_buf).into_bytes();
                writer.send(payload).map_err(PyRuntimeError::new_err)?;

                let done = (i_marker + cnt).min(m);
                if done % progress_block == 0 || done == m {
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done, m))?;
                            Ok(())
                        })?;
                    }
                }
                i_marker += cnt;
            }
            Ok(())
        };

        let run_res = if let Some(p) = &pool {
            p.install(runner)
        } else {
            runner()
        };
        let writer_res = writer.finish().map_err(PyRuntimeError::new_err);
        run_res?;
        writer_res?;
        Ok(m)
    })
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
/// Unified additive LM scan — works with any GenotypeMatrix backend.
/// Both mmap (BedMmapMatrix) and packed (PackedBedMatrix) call this function.
#[allow(dead_code)]
pub(crate) fn lm_unified_scan_to_tsv<G: GenotypeMatrix>(
    input: &mut UnifiedInput<G>, y: &[f64], x_flat: &[f64], ixx_flat: &[f64],
    out_tsv: &str, sample_idx: &[usize], sample_identity: bool,
    chunk_size: usize, threads: usize,
) -> Result<(usize, usize), String> {
    let n = y.len(); let q0 = x_flat.len() / n; let m = input.n_markers();
    if m == 0 { return Ok((0, 0)); }
    let pool = crate::stats_common::get_cached_pool(threads).map_err(|e| e.to_string())?;
    let y_f32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let x_f32: Vec<f32> = x_flat.iter().map(|&v| v as f32).collect();
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n { let xr = &x_flat[i*q0..(i+1)*q0]; for j in 0..q0 { xy[j] += xr[j] * y[i]; } }
    let mut xyc = vec![0.0_f64; q0]; xs_t_ixx_into(&xy, ixx_flat, q0, &mut xyc);
    let xy_quad = dot(&xyc, &xy);
    let blk = crate::bedmath::adaptive_grm_block_rows(chunk_size.max(512), m, n, 0, threads).max(1);
    let mut buf = vec![0.0_f32; blk * n];
    let mut sy_s = vec![0.0_f32; blk]; let mut xts_s = vec![0.0_f32; blk * q0];
    let mut ss_s = vec![0.0_f64; blk]; let mut ob = vec![0.0_f64; blk * 5];
    let mut xt = vec![0.0_f64; q0]; let mut b2 = vec![0.0_f64; q0];
    let mut text = String::with_capacity(blk * 112);
    let writer = AsyncTsvWriter::with_config(out_tsv,
        b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
        64*1024*1024, 4).map_err(|e| e.to_string())?;
    let mut rs = 0usize; let mut kt = 0usize;
    while rs < m {
        let re = (rs + blk).min(m); let rh = re - rs; let bs = &mut buf[..rh * n];
        input.matrix.decode_additive_block(&input.stats, rs, bs, sample_idx, sample_identity, pool.as_ref())?;
        row_major_block_mul_mat_f32(bs, rh, n, &y_f32, 1, &mut sy_s[..rh], pool.as_ref());
        row_major_block_mul_mat_f32(bs, rh, n, &x_f32, q0, &mut xts_s[..rh*q0], pool.as_ref());
        row_major_block_sumsq_f64(bs, rh, n, &mut ss_s[..rh], pool.as_ref());
        for li in 0..rh {
            for j in 0..q0 { xt[j] = xts_s[li*q0 + j] as f64; }
            let st = lm_assoc_from_centered_projection(&xt, sy_s[li] as f64, ss_s[li], ixx_flat, &xy, xy_quad, q0, n, yy, &mut b2);
            ob[li*5..(li+1)*5].copy_from_slice(&st);
        }
        text.clear();
        for li in 0..rh {
            let s = &ob[li*5..(li+1)*5];
            let _ = write!(text, "{}\t.\t.\t.\t.\t.\t.\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                rs+li, s[0], s[1], format_chisq_value(s[4]), s[2], s[3]);
        }
        writer.send(std::mem::take(&mut text).into_bytes())?;
        kt += rh; rs = re;
    }
    writer.finish()?;
    Ok((kt, m))
}
