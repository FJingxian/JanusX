use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;

use crate::bedmath::decode_plink_bed_hardcall;
use crate::stats_common::{get_cached_pool, parse_index_vec_i64};

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
