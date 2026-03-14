use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

const EPS64: f64 = 1e-5;
const EPS32: f32 = 1e-5;

#[inline]
fn clip64(v: f64) -> f64 {
    v.clamp(EPS64, 1.0 - EPS64)
}

#[inline]
fn clip32(v: f32) -> f32 {
    v.clamp(EPS32, 1.0 - EPS32)
}

#[pyfunction]
pub fn admx_set_threads(threads: usize) -> PyResult<()> {
    if threads == 0 {
        return Err(PyRuntimeError::new_err(
            "admx_set_threads expects a positive thread count",
        ));
    }
    match ThreadPoolBuilder::new().num_threads(threads).build_global() {
        Ok(()) => Ok(()),
        Err(_) => Ok(()),
    }
}

#[pyfunction]
pub fn admx_multiply_at_omega<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
    omega: PyReadonlyArray2<'py, f32>,
    f: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let g = g.as_array();
    let omega = omega.as_array();
    let f = f.as_array();
    let (m, n) = g.dim();
    let (m2, kp) = omega.dim();
    if m != m2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G is ({m},{n}), omega is ({m2},{kp}), expected omega rows == G rows"
        )));
    }
    if f.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: f length={} expected {}",
            f.len(),
            m
        )));
    }

    let mut out_vec = vec![0.0_f32; n * kp];
    out_vec
        .par_chunks_mut(kp)
        .enumerate()
        .for_each(|(l, out_row)| {
            for i in 0..m {
                let gv = g[(i, l)];
                if gv == 3 {
                    continue;
                }
                let centered = gv as f32 - 2.0_f32 * f[i];
                for j in 0..kp {
                    out_row[j] += centered * omega[(i, j)];
                }
            }
        });
    let out = PyArray2::<f32>::zeros(py, [n, kp], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&out_vec);
    Ok(out)
}

#[pyfunction]
pub fn admx_multiply_a_omega<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
    omega: PyReadonlyArray2<'py, f32>,
    f: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let g = g.as_array();
    let omega = omega.as_array();
    let f = f.as_array();
    let (m, n) = g.dim();
    let (n2, kp) = omega.dim();
    if n != n2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G is ({m},{n}), omega is ({n2},{kp}), expected omega rows == G cols"
        )));
    }
    if f.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: f length={} expected {}",
            f.len(),
            m
        )));
    }

    let mut out_vec = vec![0.0_f32; m * kp];
    out_vec
        .par_chunks_mut(kp)
        .enumerate()
        .for_each(|(i, row)| {
            let f2 = 2.0_f32 * f[i];
            for j in 0..kp {
                let mut acc = 0.0_f32;
                for l in 0..n {
                    let gv = g[(i, l)];
                    if gv == 3 {
                        continue;
                    }
                    let centered = gv as f32 - f2;
                    acc += centered * omega[(l, j)];
                }
                row[j] = acc;
            }
        });
    let out = PyArray2::<f32>::zeros(py, [m, kp], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&out_vec);
    Ok(out)
}

#[pyfunction]
pub fn admx_allele_frequency<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let g = g.as_array();
    let (m, n) = g.dim();
    let mut out_vec = vec![0.0_f32; m];
    out_vec.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let mut sum_val = 0.0_f32;
        let mut denom = 0.0_f32;
        for j in 0..n {
            let v = g[(i, j)];
            if v == 3 {
                continue;
            }
            sum_val += v as f32;
            denom += 2.0;
        }
        *dst = if denom > 0.0 { sum_val / denom } else { 0.0 };
    });
    let out = PyArray1::<f32>::zeros(py, [m], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&out_vec);
    Ok(out)
}

#[pyfunction]
pub fn admx_loglikelihood(
    g: PyReadonlyArray2<'_, u8>,
    p: PyReadonlyArray2<'_, f64>,
    q: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let g = g.as_array();
    let p = p.as_array();
    let q = q.as_array();
    let (m, n) = g.dim();
    let (m2, k) = p.dim();
    let (n2, k2) = q.dim();
    if m != m2 || n != n2 || k != k2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G=({m},{n}), P=({m2},{k}), Q=({n2},{k2})"
        )));
    }
    let ll: f64 = (0..m)
        .into_par_iter()
        .map(|i| {
            let mut part = 0.0_f64;
            for j in 0..n {
                let gv = g[(i, j)];
                if gv == 3 {
                    continue;
                }
                let mut rec = 0.0_f64;
                for kk in 0..k {
                    rec += p[(i, kk)] * q[(j, kk)];
                }
                rec = rec.clamp(1e-12, 1.0 - 1e-12);
                let g_d = gv as f64;
                part += g_d * rec.ln() + (2.0 - g_d) * (1.0 - rec).ln();
            }
            part
        })
        .sum();
    Ok(ll)
}

#[pyfunction]
pub fn admx_rmse_f32(q1: PyReadonlyArray2<'_, f32>, q2: PyReadonlyArray2<'_, f32>) -> PyResult<f32> {
    let q1 = q1.as_array();
    let q2 = q2.as_array();
    if q1.dim() != q2.dim() {
        return Err(PyRuntimeError::new_err("shape mismatch in admx_rmse_f32"));
    }
    let (n, k) = q1.dim();
    let t = (n * k) as f32;
    if t <= 0.0 {
        return Ok(0.0);
    }
    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in 0..k {
            let d = (q1[(i, j)] - q2[(i, j)]) as f64;
            acc += d * d;
        }
    }
    Ok(((acc as f32) / t).sqrt())
}

#[pyfunction]
pub fn admx_rmse_f64(q1: PyReadonlyArray2<'_, f64>, q2: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    let q1 = q1.as_array();
    let q2 = q2.as_array();
    if q1.dim() != q2.dim() {
        return Err(PyRuntimeError::new_err("shape mismatch in admx_rmse_f64"));
    }
    let (n, k) = q1.dim();
    let t = (n * k) as f64;
    if t <= 0.0 {
        return Ok(0.0);
    }
    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in 0..k {
            let d = q1[(i, j)] - q2[(i, j)];
            acc += d * d;
        }
    }
    Ok((acc / t).sqrt())
}

#[pyfunction]
pub fn admx_kl_divergence(q1: PyReadonlyArray2<'_, f64>, q2: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    let q1 = q1.as_array();
    let q2 = q2.as_array();
    if q1.dim() != q2.dim() {
        return Err(PyRuntimeError::new_err("shape mismatch in admx_kl_divergence"));
    }
    let (n, k) = q1.dim();
    if n == 0 {
        return Ok(0.0);
    }
    let eps = 1e-10_f64;
    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in 0..k {
            let ai = q1[(i, j)];
            let bi = q2[(i, j)];
            let mid = 0.5 * (ai + bi);
            acc += ai * ((ai / mid) + eps).ln();
        }
    }
    Ok(acc / n as f64)
}

#[pyfunction]
pub fn admx_map_q_f32<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let q = q.as_array();
    let (n, k) = q.dim();
    let out = PyArray2::<f32>::zeros(py, [n, k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    for i in 0..n {
        let row = &mut out_slice[i * k..(i + 1) * k];
        let mut sum = 0.0_f32;
        for j in 0..k {
            let v = clip32(q[(i, j)]);
            row[j] = v;
            sum += v;
        }
        if sum <= 0.0 {
            let v = 1.0_f32 / (k as f32).max(1.0);
            row.fill(v);
        } else {
            for j in 0..k {
                row[j] /= sum;
            }
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn admx_map_p_f32<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let p = p.as_array();
    let (m, k) = p.dim();
    let out = PyArray2::<f32>::zeros(py, [m, k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    for i in 0..m {
        let row = &mut out_slice[i * k..(i + 1) * k];
        for j in 0..k {
            row[j] = clip32(p[(i, j)]);
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn admx_em_step<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
    p: PyReadonlyArray2<'py, f64>,
    q: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let g = g.as_array();
    let p = p.as_array();
    let q = q.as_array();
    let (m, n) = g.dim();
    let (m2, k) = p.dim();
    let (n2, k2) = q.dim();
    if m != m2 || n != n2 || k != k2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G=({m},{n}), P=({m2},{k}), Q=({n2},{k2})"
        )));
    }

    let p_src = p
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("P not contiguous"))?;
    let q_src = q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("Q not contiguous"))?;

    // Pass 1: compute P_EM row-wise (parallel over SNP rows).
    let mut p_em_vec = vec![0.0_f64; m * k];
    p_em_vec
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, out_row)| {
            let p_row = &p_src[i * k..(i + 1) * k];
            let mut a = vec![0.0_f64; k];
            let mut b = vec![0.0_f64; k];
            for col in 0..n {
                let gv = g[(i, col)];
                if gv == 3 {
                    continue;
                }
                let q_row = &q_src[col * k..(col + 1) * k];
                let mut rec = 0.0_f64;
                for kk in 0..k {
                    rec += p_row[kk] * q_row[kk];
                }
                rec = rec.clamp(1e-12, 1.0 - 1e-12);
                let g_f = gv as f64;
                let aa = g_f / rec;
                let bb = (2.0 - g_f) / (1.0 - rec);
                for kk in 0..k {
                    a[kk] += q_row[kk] * aa;
                    b[kk] += q_row[kk] * bb;
                }
            }
            for kk in 0..k {
                let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                let v = if denom.abs() < 1e-14 {
                    p_row[kk]
                } else {
                    (a[kk] * p_row[kk]) / denom
                };
                out_row[kk] = clip64(v);
            }
        });

    // Pass 2: compute Q_EM block-wise to avoid allocating thread-local n*k buffers.
    let mut q_em_vec = vec![0.0_f64; n * k];
    let block_cols = 256_usize.max(k).min(n.max(1));
    for block_start in (0..n).step_by(block_cols) {
        let block_len = (n - block_start).min(block_cols);
        let (q_bat_blk, t_blk) = (0..m)
            .into_par_iter()
            .fold(
                || (vec![0.0_f64; block_len], vec![0.0_f64; block_len * k]),
                |(mut qb, mut tb), i| {
                    let p_row = &p_src[i * k..(i + 1) * k];
                    for off in 0..block_len {
                        let col = block_start + off;
                        let gv = g[(i, col)];
                        if gv == 3 {
                            continue;
                        }
                        qb[off] += 2.0;
                        let q_row = &q_src[col * k..(col + 1) * k];
                        let mut rec = 0.0_f64;
                        for kk in 0..k {
                            rec += p_row[kk] * q_row[kk];
                        }
                        rec = rec.clamp(1e-12, 1.0 - 1e-12);
                        let g_f = gv as f64;
                        let aa = g_f / rec;
                        let bb = (2.0 - g_f) / (1.0 - rec);
                        let t_row = &mut tb[off * k..(off + 1) * k];
                        for kk in 0..k {
                            t_row[kk] += p_row[kk] * (aa - bb) + bb;
                        }
                    }
                    (qb, tb)
                },
            )
            .reduce(
                || (vec![0.0_f64; block_len], vec![0.0_f64; block_len * k]),
                |(mut qb1, mut tb1), (qb2, tb2)| {
                    for off in 0..block_len {
                        qb1[off] += qb2[off];
                    }
                    for idx in 0..(block_len * k) {
                        tb1[idx] += tb2[idx];
                    }
                    (qb1, tb1)
                },
            );

        for off in 0..block_len {
            let col = block_start + off;
            let out_row = &mut q_em_vec[col * k..(col + 1) * k];
            let q_row = &q_src[col * k..(col + 1) * k];
            if q_bat_blk[off] <= 0.0 {
                for kk in 0..k {
                    out_row[kk] = clip64(q_row[kk]);
                }
            } else {
                let inv = 1.0 / q_bat_blk[off];
                let t_row = &t_blk[off * k..(off + 1) * k];
                for kk in 0..k {
                    out_row[kk] = clip64(q_row[kk] * t_row[kk] * inv);
                }
            }
            let sum = out_row.iter().sum::<f64>();
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f64).max(1.0);
                out_row.fill(v);
            } else {
                for kk in 0..k {
                    out_row[kk] /= sum;
                }
            }
        }
    }

    let p_em = PyArray2::<f64>::zeros(py, [m, k], false).into_bound();
    let p_em_slice = unsafe {
        p_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output P_EM not contiguous"))?
    };
    p_em_slice.copy_from_slice(&p_em_vec);
    let q_em = PyArray2::<f64>::zeros(py, [n, k], false).into_bound();
    let q_em_slice = unsafe {
        q_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output Q_EM not contiguous"))?
    };
    q_em_slice.copy_from_slice(&q_em_vec);
    Ok((p_em, q_em))
}

#[pyfunction]
pub fn admx_adam_update_p<'py>(
    py: Python<'py>,
    p0: PyReadonlyArray2<'py, f64>,
    p1: PyReadonlyArray2<'py, f64>,
    m_p: PyReadonlyArray2<'py, f64>,
    v_p: PyReadonlyArray2<'py, f64>,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let p0 = p0.as_array();
    let p1 = p1.as_array();
    let m_p = m_p.as_array();
    let v_p = v_p.as_array();
    if p0.dim() != p1.dim() || p0.dim() != m_p.dim() || p0.dim() != v_p.dim() {
        return Err(PyRuntimeError::new_err(
            "shape mismatch in admx_adam_update_p (p0/p1/m_p/v_p)",
        ));
    }
    let (m, k) = p0.dim();
    let mut p_out = vec![0.0_f64; m * k];
    let mut m_out = vec![0.0_f64; m * k];
    let mut v_out = vec![0.0_f64; m * k];
    let one_b1 = 1.0 - beta1;
    let one_b2 = 1.0 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0 - beta1_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta1_t)
    };
    let v_scale = if (1.0 - beta2_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta2_t)
    };
    let p0s = p0
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("p0 not contiguous"))?;
    let p1s = p1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("p1 not contiguous"))?;
    let m0s = m_p
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("m_p not contiguous"))?;
    let v0s = v_p
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("v_p not contiguous"))?;
    p_out
        .par_chunks_mut(k)
        .zip(m_out.par_chunks_mut(k))
        .zip(v_out.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((p_row, m_row), v_row))| {
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = p1s[idx] - p0s[idx];
                let mcur = beta1 * m0s[idx] + one_b1 * delta;
                let vcur = beta2 * v0s[idx] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                p_row[j] = clip64(p0s[idx] + step);
                m_row[j] = mcur;
                v_row[j] = vcur;
            }
        });
    let p_new = PyArray2::<f64>::zeros(py, [m, k], false).into_bound();
    let m_new = PyArray2::<f64>::zeros(py, [m, k], false).into_bound();
    let v_new = PyArray2::<f64>::zeros(py, [m, k], false).into_bound();
    let p_slice = unsafe {
        p_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("p_new not contiguous"))?
    };
    let m_slice = unsafe {
        m_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_new not contiguous"))?
    };
    let v_slice = unsafe {
        v_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_new not contiguous"))?
    };
    p_slice.copy_from_slice(&p_out);
    m_slice.copy_from_slice(&m_out);
    v_slice.copy_from_slice(&v_out);
    Ok((p_new, m_new, v_new))
}

#[pyfunction]
pub fn admx_adam_update_q<'py>(
    py: Python<'py>,
    q0: PyReadonlyArray2<'py, f64>,
    q1: PyReadonlyArray2<'py, f64>,
    m_q: PyReadonlyArray2<'py, f64>,
    v_q: PyReadonlyArray2<'py, f64>,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let q0 = q0.as_array();
    let q1 = q1.as_array();
    let m_q = m_q.as_array();
    let v_q = v_q.as_array();
    if q0.dim() != q1.dim() || q0.dim() != m_q.dim() || q0.dim() != v_q.dim() {
        return Err(PyRuntimeError::new_err(
            "shape mismatch in admx_adam_update_q (q0/q1/m_q/v_q)",
        ));
    }
    let (n, k) = q0.dim();
    let mut q_out = vec![0.0_f64; n * k];
    let mut m_out = vec![0.0_f64; n * k];
    let mut v_out = vec![0.0_f64; n * k];
    let one_b1 = 1.0 - beta1;
    let one_b2 = 1.0 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0 - beta1_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta1_t)
    };
    let v_scale = if (1.0 - beta2_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta2_t)
    };
    let q0s = q0
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("q0 not contiguous"))?;
    let q1s = q1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("q1 not contiguous"))?;
    let m0s = m_q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("m_q not contiguous"))?;
    let v0s = v_q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("v_q not contiguous"))?;
    q_out
        .par_chunks_mut(k)
        .zip(m_out.par_chunks_mut(k))
        .zip(v_out.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((q_row, m_row), v_row))| {
            let mut sum = 0.0_f64;
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = q1s[idx] - q0s[idx];
                let mcur = beta1 * m0s[idx] + one_b1 * delta;
                let vcur = beta2 * v0s[idx] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                let qv = clip64(q0s[idx] + step);
                q_row[j] = qv;
                m_row[j] = mcur;
                v_row[j] = vcur;
                sum += qv;
            }
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f64).max(1.0);
                q_row.fill(v);
            } else {
                for j in 0..k {
                    q_row[j] /= sum;
                }
            }
        });
    let q_new = PyArray2::<f64>::zeros(py, [n, k], false).into_bound();
    let m_new = PyArray2::<f64>::zeros(py, [n, k], false).into_bound();
    let v_new = PyArray2::<f64>::zeros(py, [n, k], false).into_bound();
    let q_slice = unsafe {
        q_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("q_new not contiguous"))?
    };
    let m_slice = unsafe {
        m_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_new not contiguous"))?
    };
    let v_slice = unsafe {
        v_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_new not contiguous"))?
    };
    q_slice.copy_from_slice(&q_out);
    m_slice.copy_from_slice(&m_out);
    v_slice.copy_from_slice(&v_out);
    Ok((q_new, m_new, v_new))
}

#[pyfunction]
pub fn admx_em_step_inplace<'py>(
    g: PyReadonlyArray2<'py, u8>,
    p: PyReadonlyArray2<'py, f64>,
    q: PyReadonlyArray2<'py, f64>,
    p_em: Bound<'py, PyArray2<f64>>,
    q_em: Bound<'py, PyArray2<f64>>,
) -> PyResult<()> {
    let g = g.as_array();
    let p = p.as_array();
    let q = q.as_array();
    let (m, n) = g.dim();
    let (m2, k) = p.dim();
    let (n2, k2) = q.dim();
    if m != m2 || n != n2 || k != k2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G=({m},{n}), P=({m2},{k}), Q=({n2},{k2})"
        )));
    }
    let p_shape = p_em.shape();
    let q_shape = q_em.shape();
    if p_shape.len() != 2
        || q_shape.len() != 2
        || p_shape[0] != m
        || p_shape[1] != k
        || q_shape[0] != n
        || q_shape[1] != k
    {
        return Err(PyRuntimeError::new_err(format!(
            "output shape mismatch: p_em={:?}, q_em={:?}, expected ({m},{k}) and ({n},{k})",
            p_shape, q_shape
        )));
    }

    let p_src = p
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("P not contiguous"))?;
    let q_src = q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("Q not contiguous"))?;
    let p_em_slice = unsafe {
        p_em
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("p_em is not contiguous"))?
    };
    let q_em_slice = unsafe {
        q_em
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("q_em is not contiguous"))?
    };

    p_em_slice
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, out_row)| {
            let p_row = &p_src[i * k..(i + 1) * k];
            let mut a = vec![0.0_f64; k];
            let mut b = vec![0.0_f64; k];
            for col in 0..n {
                let gv = g[(i, col)];
                if gv == 3 {
                    continue;
                }
                let q_row = &q_src[col * k..(col + 1) * k];
                let mut rec = 0.0_f64;
                for kk in 0..k {
                    rec += p_row[kk] * q_row[kk];
                }
                rec = rec.clamp(1e-12, 1.0 - 1e-12);
                let g_f = gv as f64;
                let aa = g_f / rec;
                let bb = (2.0 - g_f) / (1.0 - rec);
                for kk in 0..k {
                    a[kk] += q_row[kk] * aa;
                    b[kk] += q_row[kk] * bb;
                }
            }
            for kk in 0..k {
                let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                let v = if denom.abs() < 1e-14 {
                    p_row[kk]
                } else {
                    (a[kk] * p_row[kk]) / denom
                };
                out_row[kk] = clip64(v);
            }
        });

    let block_cols = 256_usize.max(k).min(n.max(1));
    for block_start in (0..n).step_by(block_cols) {
        let block_len = (n - block_start).min(block_cols);
        let (q_bat_blk, t_blk) = (0..m)
            .into_par_iter()
            .fold(
                || (vec![0.0_f64; block_len], vec![0.0_f64; block_len * k]),
                |(mut qb, mut tb), i| {
                    let p_row = &p_src[i * k..(i + 1) * k];
                    for off in 0..block_len {
                        let col = block_start + off;
                        let gv = g[(i, col)];
                        if gv == 3 {
                            continue;
                        }
                        qb[off] += 2.0;
                        let q_row = &q_src[col * k..(col + 1) * k];
                        let mut rec = 0.0_f64;
                        for kk in 0..k {
                            rec += p_row[kk] * q_row[kk];
                        }
                        rec = rec.clamp(1e-12, 1.0 - 1e-12);
                        let g_f = gv as f64;
                        let aa = g_f / rec;
                        let bb = (2.0 - g_f) / (1.0 - rec);
                        let t_row = &mut tb[off * k..(off + 1) * k];
                        for kk in 0..k {
                            t_row[kk] += p_row[kk] * (aa - bb) + bb;
                        }
                    }
                    (qb, tb)
                },
            )
            .reduce(
                || (vec![0.0_f64; block_len], vec![0.0_f64; block_len * k]),
                |(mut qb1, mut tb1), (qb2, tb2)| {
                    for off in 0..block_len {
                        qb1[off] += qb2[off];
                    }
                    for idx in 0..(block_len * k) {
                        tb1[idx] += tb2[idx];
                    }
                    (qb1, tb1)
                },
            );

        for off in 0..block_len {
            let col = block_start + off;
            let out_row = &mut q_em_slice[col * k..(col + 1) * k];
            let q_row = &q_src[col * k..(col + 1) * k];
            if q_bat_blk[off] <= 0.0 {
                for kk in 0..k {
                    out_row[kk] = clip64(q_row[kk]);
                }
            } else {
                let inv = 1.0 / q_bat_blk[off];
                let t_row = &t_blk[off * k..(off + 1) * k];
                for kk in 0..k {
                    out_row[kk] = clip64(q_row[kk] * t_row[kk] * inv);
                }
            }
            let sum = out_row.iter().sum::<f64>();
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f64).max(1.0);
                out_row.fill(v);
            } else {
                for kk in 0..k {
                    out_row[kk] /= sum;
                }
            }
        }
    }
    Ok(())
}

#[pyfunction]
pub fn admx_adam_update_p_inplace<'py>(
    p0: Bound<'py, PyArray2<f64>>,
    p1: PyReadonlyArray2<'py, f64>,
    m_p: Bound<'py, PyArray2<f64>>,
    v_p: Bound<'py, PyArray2<f64>>,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
) -> PyResult<()> {
    let p1 = p1.as_array();
    let p1s = p1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("p1 not contiguous"))?;

    let p0_shape = p0.shape();
    let m_shape = m_p.shape();
    let v_shape = v_p.shape();
    if p0_shape.len() != 2 || m_shape != p0_shape || v_shape != p0_shape {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch in admx_adam_update_p_inplace: p0={:?}, m_p={:?}, v_p={:?}",
            p0_shape, m_shape, v_shape
        )));
    }
    let m = p0_shape[0];
    let k = p0_shape[1];
    if p1.shape() != [m, k] {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: p1={:?}, expected [{},{}]",
            p1.shape(),
            m,
            k
        )));
    }

    let p0s = unsafe {
        p0.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("p0 is not contiguous"))?
    };
    let m0s = unsafe {
        m_p.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_p is not contiguous"))?
    };
    let v0s = unsafe {
        v_p.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_p is not contiguous"))?
    };

    let one_b1 = 1.0 - beta1;
    let one_b2 = 1.0 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0 - beta1_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta1_t)
    };
    let v_scale = if (1.0 - beta2_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta2_t)
    };

    p0s.par_chunks_mut(k)
        .zip(m0s.par_chunks_mut(k))
        .zip(v0s.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((p_row, m_row), v_row))| {
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = p1s[idx] - p_row[j];
                let mcur = beta1 * m_row[j] + one_b1 * delta;
                let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                p_row[j] = clip64(p_row[j] + step);
                m_row[j] = mcur;
                v_row[j] = vcur;
            }
        });
    Ok(())
}

#[pyfunction]
pub fn admx_adam_update_q_inplace<'py>(
    q0: Bound<'py, PyArray2<f64>>,
    q1: PyReadonlyArray2<'py, f64>,
    m_q: Bound<'py, PyArray2<f64>>,
    v_q: Bound<'py, PyArray2<f64>>,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
) -> PyResult<()> {
    let q1 = q1.as_array();
    let q1s = q1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("q1 not contiguous"))?;

    let q0_shape = q0.shape();
    let m_shape = m_q.shape();
    let v_shape = v_q.shape();
    if q0_shape.len() != 2 || m_shape != q0_shape || v_shape != q0_shape {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch in admx_adam_update_q_inplace: q0={:?}, m_q={:?}, v_q={:?}",
            q0_shape, m_shape, v_shape
        )));
    }
    let n = q0_shape[0];
    let k = q0_shape[1];
    if q1.shape() != [n, k] {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: q1={:?}, expected [{},{}]",
            q1.shape(),
            n,
            k
        )));
    }

    let q0s = unsafe {
        q0.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("q0 is not contiguous"))?
    };
    let m0s = unsafe {
        m_q.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_q is not contiguous"))?
    };
    let v0s = unsafe {
        v_q.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_q is not contiguous"))?
    };

    let one_b1 = 1.0 - beta1;
    let one_b2 = 1.0 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0 - beta1_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta1_t)
    };
    let v_scale = if (1.0 - beta2_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta2_t)
    };

    q0s.par_chunks_mut(k)
        .zip(m0s.par_chunks_mut(k))
        .zip(v0s.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((q_row, m_row), v_row))| {
            let mut sum = 0.0_f64;
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = q1s[idx] - q_row[j];
                let mcur = beta1 * m_row[j] + one_b1 * delta;
                let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                let qv = clip64(q_row[j] + step);
                q_row[j] = qv;
                m_row[j] = mcur;
                v_row[j] = vcur;
                sum += qv;
            }
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f64).max(1.0);
                q_row.fill(v);
            } else {
                for j in 0..k {
                    q_row[j] /= sum;
                }
            }
        });
    Ok(())
}
