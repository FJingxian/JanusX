use nalgebra::{DMatrix, DVector};
use matrixmultiply::{dgemm, sgemm};
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
use std::sync::Arc;

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
    Python::attach(|py| py.check_signals())
        .map_err(|_| _INTERRUPTED_MSG.to_string())
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

fn parse_index_vec_i64(
    indices: &[i64],
    upper_bound: usize,
    label: &str,
) -> PyResult<Vec<usize>> {
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
        return Err(PyRuntimeError::new_err("packed must be 2D (m, bytes_per_snp)"));
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

    let out = PyArray1::<bool>::zeros(py, [m], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    out_slice.par_iter_mut().enumerate().for_each(|(row_idx, dst)| {
        let row =
            &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
        let mut alt_sum = 0.0_f64;
        let mut non_missing = 0usize;
        for l in 0..n_samples {
            let b = row[l >> 2];
            let code = (b >> ((l & 3) * 2)) & 0b11;
            match code {
                0b00 => {
                    non_missing += 1;
                }
                0b10 => {
                    non_missing += 1;
                    alt_sum += 1.0;
                }
                0b11 => {
                    non_missing += 1;
                    alt_sum += 2.0;
                }
                _ => {}
            }
        }
        if non_missing == 0 {
            *dst = false;
        } else {
            let p = alt_sum / (2.0 * non_missing as f64);
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
        return Err(PyRuntimeError::new_err("packed must be 2D (m, bytes_per_snp)"));
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
            let row =
                &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
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

#[inline]
unsafe fn dgemm_row_major_accumulate(
    m: usize,
    k: usize,
    n: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    dgemm(
        m,
        k,
        n,
        1.0,
        a,
        k as isize,
        1,
        b,
        n as isize,
        1,
        1.0,
        c,
        n as isize,
        1,
    );
}

#[inline]
unsafe fn dgemm_row_major_accumulate_transpose_b(
    m: usize,
    k: usize,
    n: usize,
    a: *const f64,
    b_t: *const f64,
    c: *mut f64,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    dgemm(
        m,
        k,
        n,
        1.0,
        a,
        k as isize,
        1,
        b_t,
        1,
        k as isize,
        1.0,
        c,
        n as isize,
        1,
    );
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices,
    x,
    y,
    sample_block=8192,
    threads=0
))]
pub fn bed_packed_fit_stats_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    sample_block: usize,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("packed must be 2D (m, bytes_per_snp)"));
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
    let n_train = sample_idx.len();
    if n_train == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }

    let x_arr = x.as_array();
    if x_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("x must be a 2D array"));
    }
    if x_arr.shape()[0] != n_train {
        return Err(PyRuntimeError::new_err(format!(
            "x row count mismatch: got {}, expected {n_train}",
            x_arr.shape()[0]
        )));
    }
    let p = x_arr.shape()[1];

    let y_arr = y.as_array();
    if y_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("y must be a 2D array"));
    }
    if y_arr.shape()[0] != n_train || y_arr.shape()[1] != 1 {
        return Err(PyRuntimeError::new_err(format!(
            "y must be shaped (n_train, 1). Got {:?}, expected ({n_train}, 1)",
            y_arr.shape()
        )));
    }

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let y_flat: Cow<[f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(y_arr.iter().copied().collect()),
    };

    let step = sample_block.max(1);
    let pool = get_cached_pool(threads)?;

    let (sum_rows, sq_sum_rows, mx, my, gram) = py
        .detach(|| -> Result<_, String> {
            let mut sum_rows = vec![0.0_f64; m];
            let mut sq_sum_rows = vec![0.0_f64; m];
            let mut mx = vec![0.0_f64; m * p];
            let mut my = vec![0.0_f64; m];
            let mut gram = vec![0.0_f64; m * m];

            for st in (0..n_train).step_by(step) {
                if (st & 15usize) == 0usize {
                    _check_ctrlc()?;
                }
                let ed = (st + step).min(n_train);
                let blk_cols = ed - st;
                let sidx_blk = &sample_idx[st..ed];
                let mut blk = vec![0.0_f64; m * blk_cols];
                let mut blk_sum = vec![0.0_f64; m];
                let mut blk_sq_sum = vec![0.0_f64; m];

                let mut decode_block = || {
                    blk.par_chunks_mut(blk_cols)
                        .zip(blk_sum.par_iter_mut())
                        .zip(blk_sq_sum.par_iter_mut())
                        .enumerate()
                        .for_each(|(row_idx, ((out_row, sum_dst), sq_dst))| {
                            let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                            let flip = row_flip_vec[row_idx];
                            let mean_g = 2.0_f64 * row_maf_vec[row_idx] as f64;
                            let mut sum = 0.0_f64;
                            let mut sq_sum = 0.0_f64;
                            for (j, &sid) in sidx_blk.iter().enumerate() {
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
                    tp.install(decode_block);
                } else {
                    decode_block();
                }

                for i in 0..m {
                    sum_rows[i] += blk_sum[i];
                    sq_sum_rows[i] += blk_sq_sum[i];
                }

                unsafe {
                    dgemm_row_major_accumulate(
                        m,
                        blk_cols,
                        p,
                        blk.as_ptr(),
                        x_flat[st * p..].as_ptr(),
                        mx.as_mut_ptr(),
                    );
                    dgemm_row_major_accumulate(
                        m,
                        blk_cols,
                        1,
                        blk.as_ptr(),
                        y_flat[st..].as_ptr(),
                        my.as_mut_ptr(),
                    );
                    dgemm_row_major_accumulate_transpose_b(
                        m,
                        blk_cols,
                        m,
                        blk.as_ptr(),
                        blk.as_ptr(),
                        gram.as_mut_ptr(),
                    );
                }
            }

            Ok((sum_rows, sq_sum_rows, mx, my, gram))
        })
        .map_err(_map_err_string_to_py)?;

    let sum_arr = PyArray1::from_owned_array(py, Array1::from_vec(sum_rows)).into_bound();
    let sq_sum_arr = PyArray1::from_owned_array(py, Array1::from_vec(sq_sum_rows)).into_bound();
    let mx_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((m, p), mx)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let my_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((m, 1), my)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let gram_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((m, m), gram)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok((sum_arr, sq_sum_arr, mx_arr, my_arr, gram_arr))
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
        return Err(PyRuntimeError::new_err("packed must be 2D (m, bytes_per_snp)"));
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
        return Err(PyRuntimeError::new_err("packed must be 2D (m, bytes_per_snp)"));
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
        return Err(PyRuntimeError::new_err("packed must be 2D (m, bytes_per_snp)"));
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
            rotate_snp_block_with_ut(
                &snp_block[a_start..a_end],
                rows_here,
                n,
                u_t,
                out_chunk,
            );
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
        return Err(PyRuntimeError::new_err("u_t must be (n, n) and row-major U^T"));
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
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd,
        _mm256_storeu_pd,
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
    use std::arch::aarch64::{
        vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64,
    };

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
        return Err(PyRuntimeError::new_err("u_t must be (n, n) and row-major U^T"));
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
        return Err(PyRuntimeError::new_err("packed must be 2D (m, bytes_per_snp)"));
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
            x0,
            tau,
            &s_vec,
            &u1tx,
            &u2tx,
            &u1ty,
            &u2ty,
            &u2_xtx,
            &u2_xty,
            k,
            n,
            p,
            c_reml,
            c_ml,
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
        best_log10,
        tau,
        &s_vec,
        &u1tx,
        &u2tx,
        &u1ty,
        &u2ty,
        &u2_xtx,
        &u2_xty,
        k,
        n,
        p,
        c_reml,
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
                    out_blk
                        .par_chunks_mut(out_cols)
                        .enumerate()
                        .for_each_init(
                            || PackedFastlmmScratch::new(n, k, p),
                            |scr, (off, out_row)| {
                                let idx = row_start + off;
                                let row =
                                    &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
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
                                cholesky_solve_into(
                                    &scr.xtv_inv_x,
                                    dim,
                                    &scr.xtv_inv_y,
                                    &mut scr.beta,
                                );

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
                                let mut stat = if ml.is_finite() { 2.0 * (ml - ml0) } else { 0.0 };
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
