use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use rayon::prelude::*;

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

fn student_t_cdf(t: f64, df: i32) -> f64 {
    if df <= 0 {
        return f64::NAN;
    }
    if !t.is_finite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }

    let v = df as f64;
    let x = v / (v + t * t);
    let a = v / 2.0;
    let b = 0.5;
    let ib = betai(a, b, x);

    if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

// fn student_t_p_two_sided(t: f64, df: i32) -> f64 {
//     let cdf = student_t_cdf(t.abs(), df);
//     let mut p = 2.0 * (1.0 - cdf);
//     if p < 0.0 {
//         p = 0.0;
//     }
//     if p > 1.0 {
//         p = 1.0;
//     }
//     p
// }

fn student_t_p_two_sided(t: f64, df: i32) -> f64 {
    if df <= 0 { return f64::NAN; }
    if !t.is_finite() { return 0.0; }

    let v = df as f64;
    let tt = t.abs();
    let x = v / (v + tt * tt);      // in (0,1]
    let a = v / 2.0;
    let b = 0.5;

    // ✅ two-sided p-value == regularized incomplete beta I_x(a,b)
    // avoids catastrophic cancellation from 1 - cdf
    let mut p = betai(a, b, x);

    if p < 0.0 { p = 0.0; }
    if p > 1.0 { p = 1.0; }
    p
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[allow(non_snake_case)]
fn xs_t_iXX(xs: &[f64], ixx: &[f64], q0: usize) -> Vec<f64> {
    let mut b21 = vec![0.0; q0];
    for j in 0..q0 {
        let mut acc = 0.0;
        for k in 0..q0 {
            acc += xs[k] * ixx[k * q0 + j];
        }
        b21[j] = acc;
    }
    b21
}

#[allow(non_snake_case)]
fn build_iXXs(iXX: &[f64], b21: &[f64], invb22: f64, q0: usize) -> Vec<f64> {
    let dim = q0 + 1;
    let mut ixxs = vec![0.0; dim * dim];

    for r in 0..q0 {
        for c in 0..q0 {
            ixxs[r * dim + c] = iXX[r * q0 + c] + invb22 * (b21[r] * b21[c]);
        }
    }
    ixxs[q0 * dim + q0] = invb22;

    for j in 0..q0 {
        let v = -invb22 * b21[j];
        ixxs[q0 * dim + j] = v;
        ixxs[j * dim + q0] = v;
    }
    ixxs
}

#[inline]
fn matvec(a: &[f64], dim: usize, rhs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    for r in 0..dim {
        let row = &a[r * dim..(r + 1) * dim];
        out[r] = row.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum();
    }
    out
}

/// Fast GLM for:
/// y: (n,) float64
/// X: (n, q0) float64
/// iXX: (q0, q0) float64
/// G: (m, n) int8   (marker rows)  <-- no transpose
///
/// Return: (m, q0+3) float64
#[pyfunction]
#[pyo3(signature = (y, x, ixx, g, step=10000, threads=0))]
fn glmi8<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray2<'py, i8>,
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
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "X.n_rows must equal len(y)",
        ));
    }
    if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "iXX must be (q0,q0)",
        ));
    }
    if g_arr.shape()[1] != n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "G must be shape (m, n) for int8 fast path",
        ));
    }
    let m = g_arr.shape()[0];
    let row_stride = q0 + 3;

    // flatten X and iXX for fast indexing
    let x_flat: Vec<f64> = x_arr.iter().cloned().collect();
    let ixx_flat: Vec<f64> = ixx_arr.iter().cloned().collect();

    // precompute xy and yy
    let mut xy = vec![0.0; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();

    // allocate output
    let out = PyArray2::<f64>::zeros(py, [m, row_stride], false);

    // IMPORTANT: borrow output as mutable slice, then parallel-fill chunks_mut safely
    // This is safe because each thread writes a disjoint row slice.
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("output array not contiguous")
        })?
    };

    // optional pool
    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    py.detach(|| {
        let mut runner = || {
            let mut i_marker = 0usize;

            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);

                // split output into mutable row chunks for this block
                let block = &mut out_slice[i_marker * row_stride..(i_marker + cnt) * row_stride];

                block
                    .par_chunks_mut(row_stride)
                    .enumerate()
                    .for_each(|(l, row_out)| {
                        let idx = i_marker + l;

                        let mut sy = 0.0_f64;
                        let mut ss = 0.0_f64;
                        let mut xs = vec![0.0_f64; q0];

                        for k in 0..n {
                            let gv = g_arr[(idx, k)] as f64;
                            sy += gv * y[k];
                            ss += gv * gv;

                            let row = &x_flat[k * q0..(k + 1) * q0];
                            for j in 0..q0 {
                                xs[j] += row[j] * gv;
                            }
                        }

                        let b21 = xs_t_iXX(&xs, &ixx_flat, q0);
                        let t2 = dot(&b21, &xs);
                        let b22 = ss - t2;

                        let (invb22, df) = if b22 < 1e-8 {
                            (0.0, (n as i32) - (q0 as i32))
                        } else {
                            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                        };

                        let dim = q0 + 1;
                        let ixxs = build_iXXs(&ixx_flat, &b21, invb22, q0);

                        let mut rhs = vec![0.0_f64; dim];
                        rhs[..q0].copy_from_slice(&xy);
                        rhs[q0] = sy;

                        let beta = matvec(&ixxs, dim, &rhs);
                        let beta_rhs = dot(&beta, &rhs);
                        let ve = (yy - beta_rhs) / (df as f64);

                        // pvalues for all coefficients
                        for ff in 0..dim {
                            let se = (ixxs[ff * dim + ff] * ve).sqrt();
                            let t = beta[ff] / se;
                            row_out[2 + ff] = student_t_p_two_sided(t, df);
                        }

                        // beta/se for SNP
                        if invb22 == 0.0 {
                            row_out[0] = f64::NAN;
                            row_out[1] = f64::NAN;
                            row_out[2 + q0] = f64::NAN;
                        } else {
                            let beta_snp = beta[q0];
                            let se_snp = (ixxs[q0 * dim + q0] * ve).sqrt();
                            row_out[0] = beta_snp;
                            row_out[1] = se_snp;
                        }
                    });

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

#[pymodule]
fn jxglm_rs(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(glmi8, m)?)?;
    Ok(())
}