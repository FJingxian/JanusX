use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use std::borrow::Cow;

use crate::linalg::{cholesky_inplace, normal_sf};
use crate::reml::{cholesky_solve, run_rotated_assoc_block_f32};
use crate::stats_common::get_cached_pool;

struct JointAssocState {
    xtwx: Vec<f64>,
    xtwy: Vec<f64>,
    rhs: Vec<f64>,
}

impl JointAssocState {
    #[inline]
    fn new(dim: usize) -> Self {
        Self {
            xtwx: vec![0.0_f64; dim.saturating_mul(dim)],
            xtwy: vec![0.0_f64; dim],
            rhs: vec![0.0_f64; dim],
        }
    }
}

#[inline]
fn init_output_nan(out_row: &mut [f64]) {
    for v in out_row.iter_mut() {
        *v = f64::NAN;
    }
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd, snp1_chunk, snp2_chunk, combo_chunk, threads=4))]
pub fn fvlmm2_assoc_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    snp1_chunk: PyReadonlyArray2<'py, f32>,
    snp2_chunk: PyReadonlyArray2<'py, f32>,
    combo_chunk: PyReadonlyArray2<'py, f32>,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let g1_arr = snp1_chunk.as_array();
    let g2_arr = snp2_chunk.as_array();
    let gc_arr = combo_chunk.as_array();

    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y_rot must not be empty"));
    }
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err(
            "xcov.n_rows must equal len(y_rot)",
        ));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if !(log10_lbd.is_finite()) {
        return Err(PyRuntimeError::new_err("log10_lbd must be finite"));
    }

    let (rows, g1_n) = (g1_arr.shape()[0], g1_arr.shape()[1]);
    if g1_n != n {
        return Err(PyRuntimeError::new_err(
            "snp1_chunk.n_cols must equal len(y_rot)",
        ));
    }
    if g2_arr.shape() != [rows, n] {
        return Err(PyRuntimeError::new_err(
            "snp2_chunk must have the same shape as snp1_chunk",
        ));
    }
    if gc_arr.shape() != [rows, n] {
        return Err(PyRuntimeError::new_err(
            "combo_chunk must have the same shape as snp1_chunk",
        ));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err(
            "log10_lbd must map to a finite positive lambda",
        ));
    }
    let dim = p_cov.saturating_add(3);
    if n <= dim {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p_cov + 3 for the joint test, got n={n}, p_cov={p_cov}"
        )));
    }

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let g1_flat: Cow<[f32]> = match snp1_chunk.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(g1_arr.iter().copied().collect()),
    };
    let g2_flat: Cow<[f32]> = match snp2_chunk.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(g2_arr.iter().copied().collect()),
    };
    let gc_flat: Cow<[f32]> = match combo_chunk.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(gc_arr.iter().copied().collect()),
    };

    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if !vv.is_finite() || vv <= 0.0 {
            return Err(PyRuntimeError::new_err(
                "S + lambda must stay finite and positive for all samples",
            ));
        }
        vinv[i] = 1.0 / vv;
    }

    let mut base_xtwx = vec![0.0_f64; dim * dim];
    let mut base_xtwy = vec![0.0_f64; dim];
    for i in 0..n {
        let vi = vinv[i];
        let yi = y[i];
        if !yi.is_finite() {
            return Err(PyRuntimeError::new_err("y_rot contains non-finite values"));
        }
        for r in 0..p_cov {
            let xir = xcov_flat[i * p_cov + r];
            if !xir.is_finite() {
                return Err(PyRuntimeError::new_err("xcov contains non-finite values"));
            }
            base_xtwy[r] += vi * xir * yi;
            for c in 0..p_cov {
                let xic = xcov_flat[i * p_cov + c];
                base_xtwx[r * dim + c] += vi * xir * xic;
            }
        }
    }

    let pool = get_cached_pool(threads)?;
    let out_vec = py.detach(|| {
        let mut out = vec![f64::NAN; rows.saturating_mul(9)];
        let ridge = 1e-6_f64;
        run_rotated_assoc_block_f32(
            &g1_flat,
            rows,
            n,
            &mut out,
            9,
            pool.as_ref(),
            || JointAssocState::new(dim),
            |idx, g1_row_f32, state, out_row| {
                init_output_nan(out_row);

                state.xtwx.copy_from_slice(&base_xtwx);
                state.xtwy.copy_from_slice(&base_xtwy);
                for v in state.xtwy[p_cov..].iter_mut() {
                    *v = 0.0;
                }

                let row_off = idx * n;
                let g2_row = &g2_flat[row_off..row_off + n];
                let gc_row = &gc_flat[row_off..row_off + n];
                let j1 = p_cov;
                let j2 = p_cov + 1;
                let jc = p_cov + 2;

                for i in 0..n {
                    let g1 = g1_row_f32[i] as f64;
                    let g2 = g2_row[i] as f64;
                    let gc = gc_row[i] as f64;
                    if !(g1.is_finite() && g2.is_finite() && gc.is_finite()) {
                        return;
                    }
                    let vi = vinv[i];
                    let yi = y[i];

                    state.xtwy[j1] += vi * g1 * yi;
                    state.xtwy[j2] += vi * g2 * yi;
                    state.xtwy[jc] += vi * gc * yi;

                    for c in 0..p_cov {
                        let xic = xcov_flat[i * p_cov + c];
                        let v1 = vi * xic * g1;
                        let v2 = vi * xic * g2;
                        let vc = vi * xic * gc;
                        state.xtwx[c * dim + j1] += v1;
                        state.xtwx[j1 * dim + c] += v1;
                        state.xtwx[c * dim + j2] += v2;
                        state.xtwx[j2 * dim + c] += v2;
                        state.xtwx[c * dim + jc] += vc;
                        state.xtwx[jc * dim + c] += vc;
                    }

                    let g11 = vi * g1 * g1;
                    let g12 = vi * g1 * g2;
                    let g1c = vi * g1 * gc;
                    let g22 = vi * g2 * g2;
                    let g2c = vi * g2 * gc;
                    let gcc = vi * gc * gc;

                    state.xtwx[j1 * dim + j1] += g11;
                    state.xtwx[j1 * dim + j2] += g12;
                    state.xtwx[j2 * dim + j1] += g12;
                    state.xtwx[j1 * dim + jc] += g1c;
                    state.xtwx[jc * dim + j1] += g1c;
                    state.xtwx[j2 * dim + j2] += g22;
                    state.xtwx[j2 * dim + jc] += g2c;
                    state.xtwx[jc * dim + j2] += g2c;
                    state.xtwx[jc * dim + jc] += gcc;
                }

                for d in 0..dim {
                    state.xtwx[d * dim + d] += ridge;
                }
                if cholesky_inplace(&mut state.xtwx, dim).is_none() {
                    return;
                }
                let beta = cholesky_solve(&state.xtwx, dim, &state.xtwy);
                if beta.len() != dim {
                    return;
                }

                let mut rtv_invr = 0.0_f64;
                for i in 0..n {
                    let g1 = g1_row_f32[i] as f64;
                    let g2 = g2_row[i] as f64;
                    let gc = gc_row[i] as f64;
                    let mut xb = g1 * beta[j1] + g2 * beta[j2] + gc * beta[jc];
                    for c in 0..p_cov {
                        xb += xcov_flat[i * p_cov + c] * beta[c];
                    }
                    let ri = y[i] - xb;
                    rtv_invr += vinv[i] * ri * ri;
                }

                let df = (n - dim) as f64;
                if !(rtv_invr.is_finite() && rtv_invr > 0.0 && df > 0.0) {
                    return;
                }
                let sigma2 = rtv_invr / df;
                if !(sigma2.is_finite() && sigma2 > 0.0) {
                    return;
                }

                let joint_cols = [j1, j2, jc];
                for (out_idx, &coef_idx) in joint_cols.iter().enumerate() {
                    state.rhs.fill(0.0);
                    state.rhs[coef_idx] = 1.0;
                    let inv_col = cholesky_solve(&state.xtwx, dim, &state.rhs);
                    if inv_col.len() != dim {
                        return;
                    }
                    let var_beta = sigma2 * inv_col[coef_idx];
                    if !(var_beta.is_finite() && var_beta > 0.0) {
                        return;
                    }
                    let se = var_beta.sqrt();
                    let beta_j = beta[coef_idx];
                    let z = beta_j / se;
                    let p = if z.is_finite() {
                        (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
                    } else {
                        f64::NAN
                    };
                    let base = out_idx * 3;
                    out_row[base] = beta_j;
                    out_row[base + 1] = se;
                    out_row[base + 2] = p;
                }
            },
        );
        out
    });

    let out_arr = Array2::from_shape_vec((rows, 9), out_vec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, out_arr).into_bound())
}
