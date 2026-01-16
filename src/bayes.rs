use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{ChiSquared, Gamma, StandardNormal};

fn array1_to_vec(arr: PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn array2_to_vec(arr: PyReadonlyArray2<f64>) -> Vec<f64> {
    let view = arr.as_array();
    let (n, p) = view.dim();
    let mut out = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            out.push(view[[i, j]]);
        }
    }
    out
}

fn array2_to_vec_snp_major(arr: PyReadonlyArray2<f64>) -> Vec<f64> {
    let view = arr.as_array();
    let (n, p) = view.dim();
    let mut out = Vec::with_capacity(n * p);
    for j in 0..p {
        for i in 0..n {
            out.push(view[[i, j]]);
        }
    }
    out
}

fn bayesa_core_impl(
    y: &[f64],
    m: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    min_abs_beta: f64,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += v * v;
            msum += v;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0)
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let s0_e = match s0_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if s0_e <= 0.0 {
        return Err("S0_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = vec![0.0; q];
    let mut x2_x = vec![0.0; q];
    for k in 0..q {
        let mut s2 = 0.0;
        for i in 0..n {
            let v = x[i * q + k];
            s2 += v * v;
        }
        x2_x[k] = s2;
    }

    let mut r = y.to_vec();
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs / var_e + x2_x[k] * alpha[k] / var_e;
            let c = x2_x[k] / var_e + 1.0 / var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
            if alpha[k].abs() < min_abs_beta {
                alpha[k] = min_abs_beta;
            }
        }

        for j in 0..p {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += m[j * n + i] * r[i];
            }
            rhs = rhs / var_e + x2[j] * beta[j] / var_e;
            let c = x2[j] / var_e + 1.0 / var_b[j];
            let z_beta: f64 = rng.sample(StandardNormal);
            let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;

            let delta = beta[j] - new_beta;
            for i in 0..n {
                r[i] += delta * m[j * n + i];
            }
            beta[j] = new_beta;
            if beta[j].abs() < min_abs_beta {
                beta[j] = min_abs_beta;
            }
        }

        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
        }

        let mut tmp_rate = 0.0;
        for j in 0..p {
            tmp_rate += 1.0 / var_b[j];
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);

        let mut ss_e = 0.0;
        for i in 0..n {
            ss_e += r[i] * r[i];
        }
        ss_e += s0_e;
        var_e = ss_e / rng.sample(chi_e);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    Ok((beta_sum, alpha_sum))
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    s0_e = None,
    min_abs_beta = 1e-9,
    seed = None
))]
pub fn bayesa(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: PyReadonlyArray2<f64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    s0_e: Option<f64>,
    min_abs_beta: f64,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if min_abs_beta <= 0.0 {
        return Err(PyValueError::new_err("min_abs_beta must be > 0"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }

    let y_vec = array1_to_vec(y);
    let n = y_vec.len();
    let m_shape = m.shape();
    if m_shape[0] != n {
        return Err(PyValueError::new_err("M rows must match len(y)"));
    }
    let p = m_shape[1];
    let m_vec = array2_to_vec_snp_major(m);

    let (x_vec, q) = match x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            (array2_to_vec(arr), q)
        }
        None => (vec![1.0; n], 1usize),
    };

    let result = py.allow_threads(|| {
        bayesa_core_impl(
            &y_vec,
            &m_vec,
            &x_vec,
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            s0_e,
            min_abs_beta,
            seed,
        )
    });

    match result {
        Ok((beta, alpha)) => {
            let beta_py = beta.into_pyarray_bound(py).unbind();
            let alpha_py = alpha.into_pyarray_bound(py).unbind();
            Ok((beta_py, alpha_py))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}
