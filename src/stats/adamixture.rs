use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;

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

    let out = PyArray2::<f32>::zeros(py, [n, kp], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    for i in 0..m {
        let f2 = 2.0_f32 * f[i];
        for l in 0..n {
            let gv = g[(i, l)];
            if gv == 3 {
                continue;
            }
            let centered = gv as f32 - f2;
            let out_row = &mut out_slice[l * kp..(l + 1) * kp];
            for j in 0..kp {
                out_row[j] += centered * omega[(i, j)];
            }
        }
    }
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

    let out = PyArray2::<f32>::zeros(py, [m, kp], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    for i in 0..m {
        let f2 = 2.0_f32 * f[i];
        let row = &mut out_slice[i * kp..(i + 1) * kp];
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
    }
    Ok(out)
}

#[pyfunction]
pub fn admx_allele_frequency<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let g = g.as_array();
    let (m, n) = g.dim();
    let out = PyArray1::<f32>::zeros(py, [m], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    for i in 0..m {
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
        out_slice[i] = if denom > 0.0 { sum_val / denom } else { 0.0 };
    }
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
    let mut ll = 0.0_f64;
    for i in 0..m {
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
            ll += g_d * rec.ln() + (2.0 - g_d) * (1.0 - rec).ln();
        }
    }
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

    let p_em = PyArray2::<f64>::zeros(py, [m, k], false).into_bound();
    let q_em = PyArray2::<f64>::zeros(py, [n, k], false).into_bound();
    let p_em_slice = unsafe {
        p_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output P_EM not contiguous"))?
    };
    let q_em_slice = unsafe {
        q_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output Q_EM not contiguous"))?
    };

    let mut t_acc = vec![0.0_f64; n * k];
    let mut q_bat = vec![0.0_f64; n];
    let mut a = vec![0.0_f64; k];
    let mut b = vec![0.0_f64; k];

    for i in 0..m {
        a.fill(0.0);
        b.fill(0.0);
        let p_row = &p.as_slice().ok_or_else(|| PyRuntimeError::new_err("P not contiguous"))?[i * k..(i + 1) * k];
        for col in 0..n {
            let gv = g[(i, col)];
            if gv == 3 {
                continue;
            }
            q_bat[col] += 2.0;
            let q_row = &q.as_slice().ok_or_else(|| PyRuntimeError::new_err("Q not contiguous"))?[col * k..(col + 1) * k];
            let mut rec = 0.0_f64;
            for kk in 0..k {
                rec += p_row[kk] * q_row[kk];
            }
            rec = rec.clamp(1e-12, 1.0 - 1e-12);
            let g_f = gv as f64;
            let aa = g_f / rec;
            let bb = (2.0 - g_f) / (1.0 - rec);
            let t_row = &mut t_acc[col * k..(col + 1) * k];
            for kk in 0..k {
                a[kk] += q_row[kk] * aa;
                b[kk] += q_row[kk] * bb;
                t_row[kk] += p_row[kk] * (aa - bb) + bb;
            }
        }
        let out_row = &mut p_em_slice[i * k..(i + 1) * k];
        for kk in 0..k {
            let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
            let v = if denom.abs() < 1e-14 {
                p_row[kk]
            } else {
                (a[kk] * p_row[kk]) / denom
            };
            out_row[kk] = clip64(v);
        }
    }

    let q_src = q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("Q not contiguous"))?;
    for col in 0..n {
        let out_row = &mut q_em_slice[col * k..(col + 1) * k];
        let q_row = &q_src[col * k..(col + 1) * k];
        if q_bat[col] <= 0.0 {
            for kk in 0..k {
                out_row[kk] = clip64(q_row[kk]);
            }
        } else {
            let inv = 1.0 / q_bat[col];
            let t_row = &t_acc[col * k..(col + 1) * k];
            for kk in 0..k {
                out_row[kk] = clip64(q_row[kk] * t_row[kk] * inv);
            }
        }
        let mut sum = out_row.iter().sum::<f64>();
        if sum <= 0.0 || !sum.is_finite() {
            let v = 1.0 / (k as f64).max(1.0);
            out_row.fill(v);
            sum = 1.0;
        }
        for kk in 0..k {
            out_row[kk] /= sum;
        }
    }
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
    for idx in 0..(m * k) {
        let delta = p1s[idx] - p0s[idx];
        let mcur = beta1 * m0s[idx] + one_b1 * delta;
        let vcur = beta2 * v0s[idx] + one_b2 * delta * delta;
        let m_hat = mcur * m_scale;
        let v_hat = vcur * v_scale;
        let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
        p_slice[idx] = clip64(p0s[idx] + step);
        m_slice[idx] = mcur;
        v_slice[idx] = vcur;
    }
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
    for i in 0..n {
        let row = &mut q_slice[i * k..(i + 1) * k];
        let mut sum = 0.0_f64;
        for j in 0..k {
            let idx = i * k + j;
            let delta = q1s[idx] - q0s[idx];
            let mcur = beta1 * m0s[idx] + one_b1 * delta;
            let vcur = beta2 * v0s[idx] + one_b2 * delta * delta;
            let m_hat = mcur * m_scale;
            let v_hat = vcur * v_scale;
            let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
            let qv = clip64(q0s[idx] + step);
            row[j] = qv;
            m_slice[idx] = mcur;
            v_slice[idx] = vcur;
            sum += qv;
        }
        if sum <= 0.0 || !sum.is_finite() {
            let v = 1.0 / (k as f64).max(1.0);
            row.fill(v);
        } else {
            for j in 0..k {
                row[j] /= sum;
            }
        }
    }
    Ok((q_new, m_new, v_new))
}
