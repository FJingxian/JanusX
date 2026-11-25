//! General Multi-Kernel BLUP (MK-BLUP) solver and nonlinear/embedding kernel builders.
//!
//! Solves the linear mixed model with an arbitrary number of symmetric PSD kernels
//! `K_1, ..., K_R`:
//!
//! ```text
//! y = X * beta + sum_{k=1}^R u_k + e,
//! u_k ~ N(0, sigma_k^2 * K_k),   e ~ N(0, sigma_e^2 * I_n)
//! ```
//!
//! Variance components are estimated via a hybrid AI-REML + EM-REML solver with
//! active-set boundary handling, backtracking line search, and KKT optimality
//! checks. Dense `n x n` factorizations and solves use blocked Cholesky + BLAS
//! `dgemm` (`crate::blas::cblas_dgemm_dispatch`).

use std::f64::consts::PI;

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::blas::{cblas_dgemm_dispatch, CblasInt, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS};
use crate::linalg::{cholesky_inplace, cholesky_logdet, cholesky_solve_into};

const CHOL_BLOCK: usize = 64;

#[derive(Clone, Debug)]
pub struct MkBlupFitConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub min_var_ratio: f64,
    pub ridge: f64,
}

impl Default for MkBlupFitConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-6,
            min_var_ratio: 1e-8,
            ridge: 1e-8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MkBlupFitResult {
    /// Estimated genetic variance components `[sigma_1^2, ..., sigma_R^2]`.
    pub sigma2: Vec<f64>,
    /// Estimated residual variance `sigma_e^2`.
    pub sigma_e2: f64,
    /// Per-kernel marginal variance proportions `(sigma_k^2 * trace(K_k) / n) / total_var`.
    pub h2_components: Vec<f64>,
    /// Total genetic variance proportion `sum_k (sigma_k^2 * trace(K_k) / n) / total_var`.
    pub h2_total: f64,
    /// Fixed-effect GLS estimates `beta` (length `p`).
    pub beta: Vec<f64>,
    /// Dual weights `alpha = P * y = V^{-1} * (y - X * beta)` (length `n`).
    pub alpha: Vec<f64>,
    /// Per-kernel BLUPs `u_k = sigma_k^2 * K_k * alpha` (each length `n`).
    pub u_components: Vec<Vec<f64>>,
    /// Fitted training predictions `X * beta + sum_k u_k` (length `n`).
    pub y_hat: Vec<f64>,
    /// Converged REML log-likelihood.
    pub reml: f64,
    /// Whether the solver reached KKT/tolerance optimality before `max_iter`.
    pub converged: bool,
    /// Number of iterations executed.
    pub used_iter: usize,
}

#[inline]
fn dgemm_rm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if beta == 0.0 {
            for r in 0..m {
                c[r * ldc..r * ldc + n].fill(0.0);
            }
        } else if (beta - 1.0).abs() > f64::EPSILON {
            for r in 0..m {
                for v in &mut c[r * ldc..r * ldc + n] {
                    *v *= beta;
                }
            }
        }
        return;
    }
    let ta = if trans_a { CBLAS_TRANS } else { CBLAS_NO_TRANS };
    let tb = if trans_b { CBLAS_TRANS } else { CBLAS_NO_TRANS };
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_ROW_MAJOR,
            ta,
            tb,
            m as CblasInt,
            n as CblasInt,
            k as CblasInt,
            alpha,
            a.as_ptr(),
            lda as CblasInt,
            b.as_ptr(),
            ldb as CblasInt,
            beta,
            c.as_mut_ptr(),
            ldc as CblasInt,
        );
    }
}

#[inline]
fn dgemm_nn(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    dgemm_rm(false, false, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
}

#[inline]
fn dgemm_tn(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    dgemm_rm(true, false, m, n, k, 1.0, a, m, b, n, 0.0, c, n);
}

#[inline]
fn dgemm_aat(n: usize, k: usize, a: &[f64], c: &mut [f64]) {
    dgemm_rm(false, true, n, n, k, 1.0, a, k, a, k, 0.0, c, n);
}

#[inline]
fn dgemm_nt_sub(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    dgemm_rm(false, true, m, n, k, -1.0, a, k, b, k, 1.0, c, n);
}

/// Validate that `k_mat` is an `n x n` finite, symmetric matrix with non-negative
/// diagonal entries (necessary for PSD kernels).
fn validate_symmetric_kernel(k_mat: &[f64], n: usize, label: &str) -> Result<(), String> {
    let nn = n
        .checked_mul(n)
        .ok_or_else(|| format!("{label}: n*n overflow for n={n}"))?;
    if n == 0 || k_mat.len() != nn {
        return Err(format!(
            "{label}: invalid kernel shape (len={}, n={n})",
            k_mat.len()
        ));
    }
    if k_mat.iter().any(|v| !v.is_finite()) {
        return Err(format!("{label}: kernel contains non-finite values"));
    }
    for i in 0..n {
        let dii = k_mat[i * n + i];
        if dii < 0.0 {
            return Err(format!(
                "{label}: kernel has negative diagonal {dii} at index {i}"
            ));
        }
        for j in (i + 1)..n {
            let a_ij = k_mat[i * n + j];
            let a_ji = k_mat[j * n + i];
            let diff = (a_ij - a_ji).abs();
            if diff > 1e-6 * a_ij.abs().max(a_ji.abs()) {
                return Err(format!(
                    "{label}: kernel is not symmetric at ({i}, {j}) (diff={diff})"
                ));
            }
        }
    }
    Ok(())
}

/// Blocked right-looking Cholesky factorization `A = L * L^T` in row-major order.
/// Stores `L` in the lower triangle and zeros the strict upper triangle.
fn blocked_cholesky_lower_inplace(a: &mut [f64], n: usize) -> Option<()> {
    if a.len() != n.checked_mul(n)? {
        return None;
    }
    for j0 in (0..n).step_by(CHOL_BLOCK) {
        let jb = (n - j0).min(CHOL_BLOCK);
        for i in 0..jb {
            let gi = j0 + i;
            for j in 0..=i {
                let gj = j0 + j;
                let mut sum = a[gi * n + gj];
                for k in 0..j {
                    let gk = j0 + k;
                    sum -= a[gi * n + gk] * a[gj * n + gk];
                }
                if i == j {
                    if !sum.is_finite() || sum <= 1e-18 {
                        return None;
                    }
                    a[gi * n + gj] = sum.sqrt();
                } else {
                    a[gi * n + gj] = sum / a[gj * n + gj];
                }
            }
            for j in (i + 1)..jb {
                a[gi * n + (j0 + j)] = 0.0;
            }
        }

        let rem = n - (j0 + jb);
        if rem == 0 {
            continue;
        }
        let r0 = j0 + jb;
        let (top_part, bot_part) = a.split_at_mut(r0 * n);
        let l11_rows = &top_part[j0 * n..r0 * n];
        bot_part.par_chunks_mut(n).for_each(|row| {
            for j in 0..jb {
                let mut sum = row[j0 + j];
                for k in 0..j {
                    sum -= row[j0 + k] * l11_rows[j * n + (j0 + k)];
                }
                row[j0 + j] = sum / l11_rows[j * n + (j0 + j)];
            }
        });

        let mut l21 = vec![0.0_f64; rem * jb];
        for r in 0..rem {
            let src_off = (r0 + r) * n + j0;
            l21[r * jb..(r + 1) * jb].copy_from_slice(&a[src_off..src_off + jb]);
        }
        let a22_ptr_off = r0 * n + r0;
        dgemm_rm(
            false,
            true,
            rem,
            rem,
            jb,
            -1.0,
            &l21,
            jb,
            &l21,
            jb,
            1.0,
            &mut a[a22_ptr_off..],
            n,
        );
    }
    for i in 0..n {
        for j in (i + 1)..n {
            a[i * n + j] = 0.0;
        }
    }
    Some(())
}

/// Invert a lower-triangular matrix `L` and compute `V^{-1} = L^{-T} * L^{-1}`
/// using BLAS `dgemm`.
fn cholesky_invert_spd(l: &[f64], n: usize) -> Option<Vec<f64>> {
    if l.len() != n.checked_mul(n)? {
        return None;
    }
    let mut linv_cols = vec![0.0_f64; n * n];
    linv_cols
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(col, dst_col)| {
            dst_col[col] = 1.0 / l[col * n + col];
            for row in (col + 1)..n {
                let mut sum = 0.0_f64;
                let row_off = row * n;
                for k in col..row {
                    sum += l[row_off + k] * dst_col[k];
                }
                dst_col[row] = -sum / l[row_off + row];
            }
        });
    let mut v_inv = vec![0.0_f64; n * n];
    dgemm_aat(n, n, &linv_cols, &mut v_inv);
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (v_inv[i * n + j] + v_inv[j * n + i]);
            v_inv[i * n + j] = avg;
            v_inv[j * n + i] = avg;
        }
    }
    Some(v_inv)
}

#[inline]
fn sym_matvec(mat: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    out.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let row = &mat[i * n..(i + 1) * n];
        let mut acc = 0.0_f64;
        for j in 0..n {
            acc += row[j] * x[j];
        }
        *dst = acc;
    });
    out
}

#[inline]
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn frobenius_inner(a: &[f64], b: &[f64]) -> f64 {
    a.par_chunks(4096)
        .zip(b.par_chunks(4096))
        .map(|(ca, cb)| ca.iter().zip(cb.iter()).map(|(&x, &y)| x * y).sum::<f64>())
        .sum()
}

#[inline]
fn trace_f64(a: &[f64], n: usize) -> f64 {
    (0..n).map(|i| a[i * n + i]).sum()
}

fn build_v_matrix(
    kernels: &[&[f64]],
    n: usize,
    sigma2: &[f64],
    sigma_e2: f64,
    ridge: f64,
) -> Vec<f64> {
    let mut v = vec![0.0_f64; n * n];
    for (k_mat, &s2) in kernels.iter().zip(sigma2.iter()) {
        if s2 == 0.0 {
            continue;
        }
        v.par_chunks_mut(4096)
            .zip(k_mat.par_chunks(4096))
            .for_each(|(dst, src)| {
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s2 * s;
                }
            });
    }
    let diag_add = sigma_e2 + ridge;
    for i in 0..n {
        v[i * n + i] += diag_add;
    }
    v
}

struct MkRemlState {
    reml: f64,
    p_mat: Vec<f64>,
    py: Vec<f64>,
    beta: Vec<f64>,
}

fn eval_mk_reml(
    kernels: &[&[f64]],
    x_cov: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    sigma2: &[f64],
    sigma_e2: f64,
    ridge: f64,
) -> Option<MkRemlState> {
    if !sigma_e2.is_finite() || sigma_e2 <= 0.0 {
        return None;
    }
    if sigma2.iter().any(|&s| !s.is_finite() || s < 0.0) {
        return None;
    }
    let mut l_v = build_v_matrix(kernels, n, sigma2, sigma_e2, ridge);
    blocked_cholesky_lower_inplace(&mut l_v, n)?;
    let log_det_v = cholesky_logdet(&l_v, n);
    let v_inv = cholesky_invert_spd(&l_v, n)?;

    let mut v_inv_x = vec![0.0_f64; n * p];
    dgemm_nn(n, p, n, &v_inv, x_cov, &mut v_inv_x);

    let mut xtvx = vec![0.0_f64; p * p];
    dgemm_tn(p, p, n, x_cov, &v_inv_x, &mut xtvx);
    for r in 0..p {
        xtvx[r * p + r] += 1e-10;
        for c in 0..r {
            let avg = 0.5 * (xtvx[r * p + c] + xtvx[c * p + r]);
            xtvx[r * p + c] = avg;
            xtvx[c * p + r] = avg;
        }
    }
    let mut l_x = xtvx;
    cholesky_inplace(&mut l_x, p)?;
    let log_det_xtvx = cholesky_logdet(&l_x, p);

    let mut c_inv = vec![0.0_f64; p * p];
    let mut e_vec = vec![0.0_f64; p];
    let mut sol = vec![0.0_f64; p];
    for col in 0..p {
        e_vec.fill(0.0);
        e_vec[col] = 1.0;
        sol.fill(0.0);
        cholesky_solve_into(&l_x, p, &e_vec, &mut sol);
        for row in 0..p {
            c_inv[row * p + col] = sol[row];
        }
    }

    let mut xtvy = vec![0.0_f64; p];
    for col in 0..p {
        let mut acc = 0.0_f64;
        for i in 0..n {
            acc += v_inv_x[i * p + col] * y[i];
        }
        xtvy[col] = acc;
    }
    let mut beta = vec![0.0_f64; p];
    cholesky_solve_into(&l_x, p, &xtvy, &mut beta);

    let mut vx_cinv = vec![0.0_f64; n * p];
    dgemm_nn(n, p, p, &v_inv_x, &c_inv, &mut vx_cinv);
    let mut p_mat = v_inv;
    dgemm_nt_sub(n, n, p, &vx_cinv, &v_inv_x, &mut p_mat);

    let py = sym_matvec(&p_mat, n, y);
    let ypy = dot_f64(y, &py);
    if !ypy.is_finite() || ypy < 0.0 {
        return None;
    }
    let df = (n - p) as f64;
    let reml = -0.5 * (log_det_v + log_det_xtvx + ypy + df * (2.0 * PI).ln());
    if !reml.is_finite() {
        return None;
    }
    Some(MkRemlState {
        reml,
        p_mat,
        py,
        beta,
    })
}

/// Fit a general Multi-Kernel BLUP model with `R >= 1` symmetric PSD kernels
/// using hybrid AI-REML + EM-REML.
///
/// Precondition: every kernel in `kernels` should be symmetric positive
/// semi-definite (finite, symmetric, non-negative diagonal entries are
/// validated directly; full eigendecomposition PSD verification is left to
/// callers for large `n`).
pub fn mkblup_fit_reml(
    kernels: &[&[f64]],
    n: usize,
    y: &[f64],
    x_cov: Option<(&[f64], usize)>,
    cfg: &MkBlupFitConfig,
) -> Result<MkBlupFitResult, String> {
    let r_cnt = kernels.len();
    if r_cnt == 0 {
        return Err("mkblup_fit_reml: at least one kernel is required".to_string());
    }
    if n < 2 {
        return Err("mkblup_fit_reml: n must be >= 2".to_string());
    }
    if y.len() != n {
        return Err(format!(
            "mkblup_fit_reml: y length mismatch (got {}, expected {n})",
            y.len()
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err("mkblup_fit_reml: y contains non-finite values".to_string());
    }
    for (idx, k_mat) in kernels.iter().enumerate() {
        let lbl = format!("mkblup_fit_reml kernel[{idx}]");
        validate_symmetric_kernel(k_mat, n, &lbl)?;
    }

    let (x_owned, p) = match x_cov {
        Some((x_slice, p_cols)) => {
            if p_cols == 0 || x_slice.len() != n.saturating_mul(p_cols) {
                return Err(format!(
                    "mkblup_fit_reml: x_cov shape mismatch (len={}, n={n}, p={p_cols})",
                    x_slice.len()
                ));
            }
            if x_slice.iter().any(|v| !v.is_finite()) {
                return Err("mkblup_fit_reml: x_cov contains non-finite values".to_string());
            }
            (x_slice.to_vec(), p_cols)
        }
        None => (vec![1.0_f64; n], 1usize),
    };
    if n <= p {
        return Err(format!(
            "mkblup_fit_reml: n ({n}) must exceed fixed-effect columns p ({p})"
        ));
    }

    let max_iter = cfg.max_iter.max(1);
    let tol = if cfg.tol.is_finite() && cfg.tol > 0.0 {
        cfg.tol
    } else {
        1e-6
    };
    let ridge = if cfg.ridge.is_finite() && cfg.ridge >= 0.0 {
        cfg.ridge
    } else {
        1e-8
    };

    let mean_y = y.iter().sum::<f64>() / (n as f64);
    let mut var_y = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f64>() / ((n - 1) as f64);
    if !var_y.is_finite() || var_y <= 1e-12 {
        var_y = 1.0;
    }
    let min_var = (var_y * cfg.min_var_ratio.max(1e-12)).max(1e-12);
    // Normalize each kernel internally by mean(diag(K_k)) and optimize marginal
    // variance parameters tau_k = sigma_k^2 * mean(diag(K_k)) so initialization,
    // min_var, AI damping, and KKT checks are strictly invariant to K_k -> c * K_k.
    let inv_scales: Vec<f64> = kernels
        .iter()
        .map(|k_mat| {
            let md = trace_f64(k_mat, n) / (n as f64);
            if md.is_finite() && md > 0.0 {
                1.0 / md
            } else {
                0.0
            }
        })
        .collect();
    let n_nonzero_k = inv_scales.iter().filter(|&&is| is > 0.0).count().max(1);
    let mut tau: Vec<f64> = inv_scales
        .iter()
        .map(|&is| {
            if is > 0.0 {
                (0.5 * var_y / (n_nonzero_k as f64)).max(min_var)
            } else {
                0.0
            }
        })
        .collect();
    let mut sigma_e2 = (0.5 * var_y).max(min_var);
    let tau_to_sigma2 = |t: &[f64]| -> Vec<f64> {
        t.iter()
            .zip(inv_scales.iter())
            .map(|(&tk, &is)| tk * is)
            .collect()
    };
    let init_s2 = tau_to_sigma2(&tau);
    let mut state = eval_mk_reml(kernels, &x_owned, y, n, p, &init_s2, sigma_e2, ridge)
        .ok_or_else(|| "mkblup_fit_reml: initial covariance factorization failed".to_string())?;

    let m_comp = r_cnt + 1;
    let grad_scale = var_y / (n as f64);
    let mut converged = false;
    let mut used_iter = 0usize;

    for it in 0..max_iter {
        used_iter = it + 1;

        let mut score = vec![0.0_f64; m_comp];
        let mut f_vecs: Vec<Vec<f64>> = Vec::with_capacity(m_comp);
        let mut tr_pk = vec![0.0_f64; m_comp];
        let mut quad_k = vec![0.0_f64; m_comp];

        for k in 0..r_cnt {
            let inv_s = inv_scales[k];
            if inv_s <= 0.0 {
                f_vecs.push(vec![0.0_f64; n]);
                continue;
            }
            let mut fk = sym_matvec(kernels[k], n, &state.py);
            for v in &mut fk {
                *v *= inv_s;
            }
            let qk = dot_f64(&state.py, &fk);
            let trk = frobenius_inner(&state.p_mat, kernels[k]) * inv_s;
            tr_pk[k] = trk;
            quad_k[k] = qk;
            score[k] = -0.5 * (trk - qk);
            f_vecs.push(fk);
        }
        let q_e = dot_f64(&state.py, &state.py);
        let tr_e = trace_f64(&state.p_mat, n);
        tr_pk[r_cnt] = tr_e;
        quad_k[r_cnt] = q_e;
        score[r_cnt] = -0.5 * (tr_e - q_e);
        f_vecs.push(state.py.clone());

        // Check projected KKT optimality on tau_k >= 0, sigma_e^2 >= min_var.
        let mut active = vec![true; m_comp];
        let mut kkt_ok = true;
        for k in 0..r_cnt {
            if inv_scales[k] <= 0.0 {
                active[k] = false;
                continue;
            }
            let at_bound = tau[k] <= min_var * 1.01;
            let g_norm = score[k] * grad_scale;
            if at_bound && score[k] <= 0.0 {
                active[k] = false;
                if g_norm > tol {
                    kkt_ok = false;
                }
            } else if g_norm.abs() > tol {
                kkt_ok = false;
            }
        }
        let e_at_bound = sigma_e2 <= min_var * 1.01;
        let ge_norm = score[r_cnt] * grad_scale;
        if e_at_bound && score[r_cnt] <= 0.0 {
            active[r_cnt] = false;
            if ge_norm > tol {
                kkt_ok = false;
            }
        } else if ge_norm.abs() > tol {
            kkt_ok = false;
        }

        let active_idx: Vec<usize> = (0..m_comp).filter(|&k| active[k]).collect();
        if kkt_ok || active_idx.is_empty() {
            converged = true;
            break;
        }

        let mut p_f_vecs: Vec<Vec<f64>> = Vec::with_capacity(m_comp);
        for fk in &f_vecs {
            p_f_vecs.push(sym_matvec(&state.p_mat, n, fk));
        }
        let mut ai = vec![0.0_f64; m_comp * m_comp];
        for k in 0..m_comp {
            for l in 0..=k {
                let val = 0.5 * dot_f64(&f_vecs[k], &p_f_vecs[l]);
                ai[k * m_comp + l] = val;
                ai[l * m_comp + k] = val;
            }
        }

        let mut delta = vec![0.0_f64; m_comp];
        let mut ai_step_ok = false;
        let a_dim = active_idx.len();
        let mut ai_sub = vec![0.0_f64; a_dim * a_dim];
        let mut g_sub = vec![0.0_f64; a_dim];
        let mut diag_max = 0.0_f64;
        for (ii, &ki) in active_idx.iter().enumerate() {
            g_sub[ii] = score[ki];
            for (jj, &kj) in active_idx.iter().enumerate() {
                ai_sub[ii * a_dim + jj] = ai[ki * m_comp + kj];
            }
            diag_max = diag_max.max(ai_sub[ii * a_dim + ii].abs());
        }
        let damp = (diag_max * 1e-8).max(1e-10);
        for ii in 0..a_dim {
            ai_sub[ii * a_dim + ii] += damp;
        }
        if cholesky_inplace(&mut ai_sub, a_dim).is_some() {
            let mut d_sub = vec![0.0_f64; a_dim];
            cholesky_solve_into(&ai_sub, a_dim, &g_sub, &mut d_sub);
            if d_sub.iter().all(|v| v.is_finite()) {
                for (ii, &ki) in active_idx.iter().enumerate() {
                    delta[ki] = d_sub[ii];
                }
                ai_step_ok = true;
            }
        }

        let mut accepted = false;
        let mut next_tau = tau.clone();
        let mut next_sigma_e2 = sigma_e2;
        let mut next_state: Option<MkRemlState> = None;

        if ai_step_ok {
            let mut step = 1.0_f64;
            for _ in 0..16 {
                let mut cand_t = vec![0.0_f64; r_cnt];
                for k in 0..r_cnt {
                    cand_t[k] = if active[k] {
                        (tau[k] + step * delta[k]).max(0.0)
                    } else {
                        0.0
                    };
                }
                let cand_se = if active[r_cnt] {
                    (sigma_e2 + step * delta[r_cnt]).max(min_var)
                } else {
                    min_var
                };
                let cand_s2 = tau_to_sigma2(&cand_t);
                let cand = eval_mk_reml(kernels, &x_owned, y, n, p, &cand_s2, cand_se, ridge);
                if let Some(st) = cand {
                    if st.reml >= state.reml - 1e-10 {
                        accepted = true;
                        next_tau = cand_t;
                        next_sigma_e2 = cand_se;
                        next_state = Some(st);
                        break;
                    }
                }
                step *= 0.5;
            }
        }

        if !accepted {
            let n_f = n as f64;
            let mut em_t = vec![0.0_f64; r_cnt];
            for k in 0..r_cnt {
                if !active[k] {
                    em_t[k] = 0.0;
                } else {
                    let s = tau[k].max(min_var);
                    let upd = s + (s * s / n_f) * (quad_k[k] - tr_pk[k]);
                    em_t[k] = upd.max(0.0);
                }
            }
            let em_se = if !active[r_cnt] {
                min_var
            } else {
                let se = sigma_e2.max(min_var);
                (se + (se * se / n_f) * (quad_k[r_cnt] - tr_pk[r_cnt])).max(min_var)
            };
            let em_s2 = tau_to_sigma2(&em_t);
            let cand = eval_mk_reml(kernels, &x_owned, y, n, p, &em_s2, em_se, ridge);
            if let Some(st) = cand {
                if st.reml >= state.reml - 1e-10 {
                    accepted = true;
                    next_tau = em_t;
                    next_sigma_e2 = em_se;
                    next_state = Some(st);
                }
            }
        }

        if !accepted {
            converged = false;
            break;
        }

        let e_denom = sigma_e2.abs().max(var_y * 1e-4).max(min_var);
        let mut max_rel = (next_sigma_e2 - sigma_e2).abs() / e_denom;
        for k in 0..r_cnt {
            let denom = tau[k].abs().max(var_y * 1e-4).max(min_var);
            max_rel = max_rel.max((next_tau[k] - tau[k]).abs() / denom);
        }
        let reml_gain = next_state
            .as_ref()
            .map(|st| (st.reml - state.reml).abs())
            .unwrap_or(0.0);

        tau = next_tau;
        sigma_e2 = next_sigma_e2;
        if let Some(st) = next_state {
            state = st;
        }
        if max_rel < tol && reml_gain < tol * 10.0 {
            converged = true;
            break;
        }
    }

    let sigma2 = tau_to_sigma2(&tau);
    let mut u_components: Vec<Vec<f64>> = Vec::with_capacity(r_cnt);
    let mut y_hat = vec![0.0_f64; n];
    for i in 0..n {
        let mut xb = 0.0_f64;
        for c in 0..p {
            xb += x_owned[i * p + c] * state.beta[c];
        }
        y_hat[i] = xb;
    }
    for k in 0..r_cnt {
        let s2 = sigma2[k];
        let mut uk = if s2 > 0.0 {
            sym_matvec(kernels[k], n, &state.py)
        } else {
            vec![0.0_f64; n]
        };
        if s2 > 0.0 {
            for (u_val, yh) in uk.iter_mut().zip(y_hat.iter_mut()) {
                *u_val *= s2;
                *yh += *u_val;
            }
        }
        u_components.push(uk);
    }
    let total_var = (tau.iter().sum::<f64>() + sigma_e2).max(1e-18);
    let h2_components: Vec<f64> = tau.iter().map(|&tk| tk / total_var).collect();
    let h2_total: f64 = h2_components.iter().sum();

    Ok(MkBlupFitResult {
        sigma2,
        sigma_e2,
        h2_components,
        h2_total,
        beta: state.beta,
        alpha: state.py,
        u_components,
        y_hat,
        reml: state.reml,
        converged,
        used_iter,
    })
}

/// Predict out-of-sample test phenotypes from fitted MK-BLUP weights:
/// `y_pred = X_test * beta + sum_k sigma_k^2 * K_{k,test,train} * alpha`.
pub fn mkblup_predict(
    cross_kernels: &[&[f64]],
    n_test: usize,
    n_train: usize,
    sigma2: &[f64],
    alpha: &[f64],
    beta: &[f64],
    x_test: Option<(&[f64], usize)>,
) -> Result<Vec<f64>, String> {
    let r_cnt = cross_kernels.len();
    if r_cnt == 0 || r_cnt != sigma2.len() {
        return Err(format!(
            "mkblup_predict: kernel count ({r_cnt}) must match sigma2 length ({})",
            sigma2.len()
        ));
    }
    if sigma2.iter().any(|&v| !v.is_finite() || v < 0.0) {
        return Err("mkblup_predict: sigma2 must be finite and non-negative".to_string());
    }
    if alpha.len() != n_train {
        return Err(format!(
            "mkblup_predict: alpha length ({}) != n_train ({n_train})",
            alpha.len()
        ));
    }
    if alpha.iter().any(|v| !v.is_finite()) {
        return Err("mkblup_predict: alpha contains non-finite values".to_string());
    }
    if beta.iter().any(|v| !v.is_finite()) {
        return Err("mkblup_predict: beta contains non-finite values".to_string());
    }
    let expected_len = n_test
        .checked_mul(n_train)
        .ok_or_else(|| "mkblup_predict: n_test * n_train overflow".to_string())?;
    for (idx, k_mat) in cross_kernels.iter().enumerate() {
        if k_mat.len() != expected_len {
            return Err(format!(
                "mkblup_predict: cross_kernel[{idx}] length ({}) != {expected_len}",
                k_mat.len()
            ));
        }
        if k_mat.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "mkblup_predict: cross_kernel[{idx}] contains non-finite values"
            ));
        }
    }

    let mut y_pred = vec![0.0_f64; n_test];
    match x_test {
        Some((x_slice, p)) => {
            if p != beta.len() || x_slice.len() != n_test.saturating_mul(p) {
                let x_len = x_slice.len();
                let b_len = beta.len();
                return Err(format!(
                    "mkblup_predict: x_test mismatch (len={x_len}, n={n_test}, p={p}, b={b_len})"
                ));
            }
            if x_slice.iter().any(|v| !v.is_finite()) {
                return Err("mkblup_predict: x_test contains non-finite values".to_string());
            }
            for i in 0..n_test {
                let mut acc = 0.0_f64;
                for c in 0..p {
                    acc += x_slice[i * p + c] * beta[c];
                }
                y_pred[i] = acc;
            }
        }
        None => {
            if beta.len() != 1 {
                return Err(format!(
                    "mkblup_predict: x_test omitted but beta has {} columns (expected 1)",
                    beta.len()
                ));
            }
            y_pred.fill(beta[0]);
        }
    }

    for (k_mat, &s2) in cross_kernels.iter().zip(sigma2.iter()) {
        if s2 == 0.0 {
            continue;
        }
        y_pred.par_iter_mut().enumerate().for_each(|(i, dst)| {
            let row = &k_mat[i * n_train..(i + 1) * n_train];
            let mut acc = 0.0_f64;
            for j in 0..n_train {
                acc += row[j] * alpha[j];
            }
            *dst += s2 * acc;
        });
    }
    Ok(y_pred)
}

/// Build a Gaussian RBF (RKHS) kernel from an additive GRM `K_A`:
/// `K_RBF[i, j] = exp(-h * max(0, K_A[i,i] + K_A[j,j] - 2 * K_A[i,j]))`.
///
/// When `bandwidth` is `None`, uses `h = 1 / mean_{i < j}(D_ij^2)`.
pub fn rbf_kernel_from_grm(
    k_a: &[f64],
    n: usize,
    bandwidth: Option<f64>,
) -> Result<(Vec<f64>, f64), String> {
    validate_symmetric_kernel(k_a, n, "rbf_kernel_from_grm")?;
    let diag: Vec<f64> = (0..n).map(|i| k_a[i * n + i]).collect();
    let h = match bandwidth {
        Some(v) => {
            if !v.is_finite() || v <= 0.0 {
                return Err(format!(
                    "rbf_kernel_from_grm: bandwidth must be > 0, got {v}"
                ));
            }
            v
        }
        None => {
            if n <= 1 {
                1.0
            } else {
                let mut sum_d2 = 0.0_f64;
                let mut pairs = 0usize;
                for i in 0..n {
                    let di = diag[i];
                    let row = &k_a[i * n..(i + 1) * n];
                    for j in (i + 1)..n {
                        let d2 = (di + diag[j] - 2.0 * row[j]).max(0.0);
                        sum_d2 += d2;
                        pairs += 1;
                    }
                }
                let mean_d2 = if pairs > 0 {
                    sum_d2 / (pairs as f64)
                } else {
                    1.0
                };
                if mean_d2 > 1e-12 && mean_d2.is_finite() {
                    1.0 / mean_d2
                } else {
                    1.0
                }
            }
        }
    };

    let mut out = vec![0.0_f64; n * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, dst_row)| {
        let di = diag[i];
        let src_row = &k_a[i * n..(i + 1) * n];
        for j in 0..n {
            if i == j {
                dst_row[j] = 1.0;
            } else {
                let d2 = (di + diag[j] - 2.0 * src_row[j]).max(0.0);
                dst_row[j] = (-h * d2).exp();
            }
        }
    });
    Ok((out, h))
}

/// Build an epistasis (additive x additive) Hadamard kernel `K_AA = K_A .* K_A`,
/// scaled so `mean(diag(K_AA)) == 1`.
pub fn epistasis_kernel_from_grm(k_a: &[f64], n: usize) -> Result<Vec<f64>, String> {
    validate_symmetric_kernel(k_a, n, "epistasis_kernel_from_grm")?;
    let diag_mean = (0..n).map(|i| k_a[i * n + i].powi(2)).sum::<f64>() / (n as f64);
    let scale = if diag_mean > 1e-12 && diag_mean.is_finite() {
        1.0 / diag_mean
    } else {
        1.0
    };
    let mut out = vec![0.0_f64; n * n];
    out.par_chunks_mut(4096)
        .zip(k_a.par_chunks(4096))
        .for_each(|(dst, src)| {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = s * s * scale;
            }
        });
    Ok(out)
}

/// Build an Arc-Cosine kernel (infinite-width ReLU neural network kernel, Cho & Saul 2009)
/// of `degree` 0 or 1 directly from `K_A`, scaled so `mean(diag) == 1`.
pub fn arccosine_kernel_from_grm(k_a: &[f64], n: usize, degree: u8) -> Result<Vec<f64>, String> {
    validate_symmetric_kernel(k_a, n, "arccosine_kernel_from_grm")?;
    if degree > 1 {
        return Err(format!(
            "arccosine_kernel_from_grm: degree must be 0 or 1, got {degree}"
        ));
    }
    let norms: Vec<f64> = (0..n).map(|i| k_a[i * n + i].max(0.0).sqrt()).collect();
    let mut out = vec![0.0_f64; n * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, dst_row)| {
        let ni = norms[i];
        let src_row = &k_a[i * n..(i + 1) * n];
        for j in 0..n {
            let nj = norms[j];
            let denom = ni * nj;
            if denom <= 1e-15 {
                dst_row[j] = 0.0;
                continue;
            }
            let cos_t = (src_row[j] / denom).clamp(-1.0, 1.0);
            let theta = cos_t.acos();
            dst_row[j] = if degree == 0 {
                1.0 - theta / PI
            } else {
                (denom / PI) * (theta.sin() + (PI - theta) * cos_t)
            };
        }
    });
    let diag_mean = (0..n).map(|i| out[i * n + i]).sum::<f64>() / (n as f64);
    if diag_mean > 1e-12 && diag_mean.is_finite() {
        let inv = 1.0 / diag_mean;
        out.par_chunks_mut(4096).for_each(|chunk| {
            for v in chunk {
                *v *= inv;
            }
        });
    }
    Ok(out)
}

/// Build a linear kernel `K = Z * Z^T` from sample embeddings `Z` (`n x d`, e.g., JEPA embeddings),
/// optionally column-centering `Z` and trace-normalizing `mean(diag(K)) == 1`.
pub fn linear_kernel_from_embeddings(
    z: &[f64],
    n: usize,
    d: usize,
    center_cols: bool,
    normalize_trace: bool,
) -> Result<Vec<f64>, String> {
    if n == 0 || d == 0 || z.len() != n.saturating_mul(d) {
        return Err(format!(
            "linear_kernel_from_embeddings: invalid shape (len={}, n={n}, d={d})",
            z.len()
        ));
    }
    if z.iter().any(|v| !v.is_finite()) {
        return Err("linear_kernel_from_embeddings: z contains non-finite values".to_string());
    }
    let z_work: Vec<f64>;
    let z_ref: &[f64] = if center_cols {
        let mut means = vec![0.0_f64; d];
        for i in 0..n {
            let row = &z[i * d..(i + 1) * d];
            for j in 0..d {
                means[j] += row[j];
            }
        }
        let inv_n = 1.0 / (n as f64);
        for m in &mut means {
            *m *= inv_n;
        }
        let mut centered = z.to_vec();
        centered.par_chunks_mut(d).for_each(|row| {
            for j in 0..d {
                row[j] -= means[j];
            }
        });
        z_work = centered;
        &z_work
    } else {
        z
    };

    let mut out = vec![0.0_f64; n * n];
    dgemm_aat(n, d, z_ref, &mut out);
    if normalize_trace {
        let diag_mean = (0..n).map(|i| out[i * n + i]).sum::<f64>() / (n as f64);
        if diag_mean > 1e-12 && diag_mean.is_finite() {
            let inv = 1.0 / diag_mean;
            out.par_chunks_mut(4096).for_each(|chunk| {
                for v in chunk {
                    *v *= inv;
                }
            });
        }
    }
    Ok(out)
}

fn extract_2d_f64(arr: &PyReadonlyArray2<'_, f64>) -> (Vec<f64>, usize, usize) {
    let view = arr.as_array();
    let (rows, cols) = (view.nrows(), view.ncols());
    let flat = match arr.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => view.iter().copied().collect(),
    };
    (flat, rows, cols)
}

#[pyfunction]
#[pyo3(signature = (kernels, y, x_cov=None, max_iter=50, tol=1e-6, ridge=1e-8))]
pub fn mkblup_fit_reml_f64<'py>(
    py: Python<'py>,
    kernels: Vec<PyReadonlyArray2<'py, f64>>,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    max_iter: usize,
    tol: f64,
    ridge: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if kernels.is_empty() {
        return Err(PyRuntimeError::new_err(
            "mkblup_fit_reml_f64: kernels list must not be empty",
        ));
    }
    let y_vec: Vec<f64> = match y.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y.as_array().iter().copied().collect(),
    };
    let n = y_vec.len();
    let mut k_vecs: Vec<Vec<f64>> = Vec::with_capacity(kernels.len());
    for (idx, k_arr) in kernels.iter().enumerate() {
        let (flat, rows, cols) = extract_2d_f64(k_arr);
        if rows != n || cols != n {
            return Err(PyRuntimeError::new_err(format!(
                "mkblup_fit_reml_f64: kernel[{idx}] shape ({rows}, {cols}) != ({n}, {n})"
            )));
        }
        k_vecs.push(flat);
    }
    let x_owned = x_cov.map(|x_arr| {
        let (flat, rows, cols) = extract_2d_f64(&x_arr);
        (flat, rows, cols)
    });
    if let Some((_, rows, _)) = x_owned.as_ref() {
        if *rows != n {
            return Err(PyRuntimeError::new_err(format!(
                "mkblup_fit_reml_f64: x_cov rows ({rows}) != n ({n})"
            )));
        }
    }

    let cfg = MkBlupFitConfig {
        max_iter,
        tol,
        min_var_ratio: 1e-8,
        ridge,
    };

    let res = py
        .detach(move || {
            let k_refs: Vec<&[f64]> = k_vecs.iter().map(Vec::as_slice).collect();
            let x_arg = x_owned.as_ref().map(|(v, _, c)| (v.as_slice(), *c));
            mkblup_fit_reml(&k_refs, n, &y_vec, x_arg, &cfg)
        })
        .map_err(PyRuntimeError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("sigma2", PyArray1::from_vec(py, res.sigma2))?;
    out.set_item("sigma_e2", res.sigma_e2)?;
    out.set_item("h2_components", PyArray1::from_vec(py, res.h2_components))?;
    out.set_item("h2_total", res.h2_total)?;
    out.set_item("beta", PyArray1::from_vec(py, res.beta))?;
    out.set_item("alpha", PyArray1::from_vec(py, res.alpha))?;
    let u_list = PyList::empty(py);
    for uk in res.u_components {
        u_list.append(PyArray1::from_vec(py, uk))?;
    }
    out.set_item("u_components", u_list)?;
    out.set_item("y_hat", PyArray1::from_vec(py, res.y_hat))?;
    out.set_item("reml", res.reml)?;
    out.set_item("converged", res.converged)?;
    out.set_item("used_iter", res.used_iter)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (cross_kernels, sigma2, alpha, beta, x_test=None))]
pub fn mkblup_predict_f64<'py>(
    py: Python<'py>,
    cross_kernels: Vec<PyReadonlyArray2<'py, f64>>,
    sigma2: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    beta: PyReadonlyArray1<'py, f64>,
    x_test: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if cross_kernels.is_empty() {
        return Err(PyRuntimeError::new_err(
            "mkblup_predict_f64: cross_kernels must not be empty",
        ));
    }
    let sigma2_vec: Vec<f64> = match sigma2.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => sigma2.as_array().iter().copied().collect(),
    };
    let alpha_vec: Vec<f64> = match alpha.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => alpha.as_array().iter().copied().collect(),
    };
    let beta_vec: Vec<f64> = match beta.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => beta.as_array().iter().copied().collect(),
    };
    let n_train = alpha_vec.len();
    let mut k_vecs: Vec<Vec<f64>> = Vec::with_capacity(cross_kernels.len());
    let mut n_test = 0usize;
    for (idx, k_arr) in cross_kernels.iter().enumerate() {
        let (flat, rows, cols) = extract_2d_f64(k_arr);
        if idx == 0 {
            n_test = rows;
        }
        if rows != n_test || cols != n_train {
            return Err(PyRuntimeError::new_err(format!(
                "mkblup_predict_f64: kernel[{idx}] shape ({rows}, {cols}) != ({n_test}, {n_train})"
            )));
        }
        k_vecs.push(flat);
    }
    let x_owned = x_test.map(|x_arr| {
        let (flat, rows, cols) = extract_2d_f64(&x_arr);
        (flat, rows, cols)
    });
    if let Some((_, rows, _)) = x_owned.as_ref() {
        if *rows != n_test {
            return Err(PyRuntimeError::new_err(format!(
                "mkblup_predict_f64: x_test rows ({rows}) != n_test ({n_test})"
            )));
        }
    }

    let pred = py
        .detach(move || {
            let k_refs: Vec<&[f64]> = k_vecs.iter().map(Vec::as_slice).collect();
            let x_arg = x_owned.as_ref().map(|(v, _, c)| (v.as_slice(), *c));
            let s2 = &sigma2_vec;
            let al = &alpha_vec;
            let be = &beta_vec;
            mkblup_predict(&k_refs, n_test, n_train, s2, al, be, x_arg)
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyArray1::from_vec(py, pred))
}

#[pyfunction]
#[pyo3(signature = (k_a, bandwidth=None))]
pub fn mkblup_rbf_kernel_from_grm_f64<'py>(
    py: Python<'py>,
    k_a: PyReadonlyArray2<'py, f64>,
    bandwidth: Option<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, f64)> {
    let (flat, rows, cols) = extract_2d_f64(&k_a);
    if rows != cols || rows == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "mkblup_rbf_kernel_from_grm_f64: non-square or empty k_a ({rows}, {cols})"
        )));
    }
    let (rbf, h) = py
        .detach(move || rbf_kernel_from_grm(&flat, rows, bandwidth))
        .map_err(PyRuntimeError::new_err)?;
    let arr = Array2::from_shape_vec((rows, rows), rbf)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((PyArray2::from_owned_array(py, arr), h))
}

#[pyfunction]
#[pyo3(signature = (k_a, degree=1))]
pub fn mkblup_arccosine_kernel_from_grm_f64<'py>(
    py: Python<'py>,
    k_a: PyReadonlyArray2<'py, f64>,
    degree: u8,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (flat, rows, cols) = extract_2d_f64(&k_a);
    if rows != cols || rows == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "mkblup_arccosine_kernel_from_grm_f64: non-square or empty k_a ({rows}, {cols})"
        )));
    }
    let out = py
        .detach(move || arccosine_kernel_from_grm(&flat, rows, degree))
        .map_err(PyRuntimeError::new_err)?;
    let arr = Array2::from_shape_vec((rows, rows), out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
pub fn mkblup_epistasis_kernel_from_grm_f64<'py>(
    py: Python<'py>,
    k_a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (flat, rows, cols) = extract_2d_f64(&k_a);
    if rows != cols || rows == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "mkblup_epistasis_kernel_from_grm_f64: non-square or empty k_a ({rows}, {cols})"
        )));
    }
    let out = py
        .detach(move || epistasis_kernel_from_grm(&flat, rows))
        .map_err(PyRuntimeError::new_err)?;
    let arr = Array2::from_shape_vec((rows, rows), out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
#[pyo3(signature = (z, center_cols=true, normalize_trace=true))]
pub fn mkblup_linear_kernel_from_embeddings_f64<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    center_cols: bool,
    normalize_trace: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (flat, rows, cols) = extract_2d_f64(&z);
    if rows == 0 || cols == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "mkblup_linear_kernel_from_embeddings_f64: empty z ({rows}, {cols})"
        )));
    }
    let out = py
        .detach(move || {
            linear_kernel_from_embeddings(&flat, rows, cols, center_cols, normalize_trace)
        })
        .map_err(PyRuntimeError::new_err)?;
    let arr = Array2::from_shape_vec((rows, rows), out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aireml::ai_reml_null_from_spectral as spectral_reml;

    fn lcg_normal(state: &mut u64) -> f64 {
        let mut next_u = || {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let u1 = next_u().max(1e-12);
        let u2 = next_u();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    fn submatrix(mat: &[f64], n_full: usize, rows: &[usize], cols: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0_f64; rows.len() * cols.len()];
        for (ri, &r) in rows.iter().enumerate() {
            for (ci, &c) in cols.iter().enumerate() {
                out[ri * cols.len() + ci] = mat[r * n_full + c];
            }
        }
        out
    }

    #[test]
    fn mkblup_single_kernel_matches_spectral_aireml_and_is_scale_invariant() {
        let n = 80usize;
        let m = 200usize;
        let mut rng = 20261125u64;
        let mut z = vec![0.0_f64; n * m];
        for v in &mut z {
            *v = lcg_normal(&mut rng);
        }
        let k_a = linear_kernel_from_embeddings(&z, n, m, true, true).expect("build K_A");
        let mut l_k = k_a.clone();
        for i in 0..n {
            l_k[i * n + i] += 1e-6;
        }
        blocked_cholesky_lower_inplace(&mut l_k, n).expect("chol K_A");
        let g_eps: Vec<f64> = (0..n).map(|_| lcg_normal(&mut rng)).collect();
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let mut u_i = 0.0;
            for j in 0..=i {
                u_i += l_k[i * n + j] * g_eps[j];
            }
            y[i] = 2.5 + 0.8_f64.sqrt() * u_i + 0.4_f64.sqrt() * lcg_normal(&mut rng);
        }

        let cfg = MkBlupFitConfig::default();
        let mk = mkblup_fit_reml(&[&k_a], n, &y, None, &cfg).expect("mkblup 1-kernel");
        assert!(mk.converged);

        // Compare variance ratio lambda against spectral AI-REML in src/math/aireml.rs.
        let (s, u, _) = crate::eigh::symmetric_eigh_f64_row_major(&k_a, n).expect("eigh");
        let mut ut_y = vec![0.0_f64; n];
        let mut ut_1 = vec![0.0_f64; n];
        for i in 0..n {
            let mut ay = 0.0;
            let mut a1 = 0.0;
            for j in 0..n {
                let u_ji = u[j * n + i];
                ay += u_ji * y[j];
                a1 += u_ji;
            }
            ut_y[i] = ay;
            ut_1[i] = a1;
        }
        let spec = spectral_reml(&s, &ut_1, &ut_y, n, 1, 60, 1e-7, 1e-9).expect("spectral");

        let mk_lbd = mk.sigma_e2 / mk.sigma2[0];
        let rel_lbd = (mk_lbd - spec.lbd).abs() / spec.lbd.max(1e-12);
        assert!(
            rel_lbd < 1e-3,
            "lambda mismatch: mk={mk_lbd} vs spec={}",
            spec.lbd
        );

        // Rescaling K_A -> c * K_A across extreme scales must leave REML, h2, and y_hat invariant.
        for c in [1e-9_f64, 2.0_f64, 1e9_f64] {
            let k_scaled: Vec<f64> = k_a.iter().map(|&v| c * v).collect();
            let mk_s = mkblup_fit_reml(&[&k_scaled], n, &y, None, &cfg).expect("mkblup scaled");
            assert!(mk_s.converged);
            assert!((mk_s.sigma2[0] * c - mk.sigma2[0]).abs() < 1e-4);
            assert!((mk_s.reml - mk.reml).abs() < 1e-6);
            assert!((mk_s.h2_total - mk.h2_total).abs() < 1e-6);
            assert!((mk_s.h2_components[0] - mk.h2_components[0]).abs() < 1e-6);
            for (a, b) in mk.y_hat.iter().zip(mk_s.y_hat.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn mkblup_two_kernel_ka_plus_rbf_recovers_variance_and_improves_reml() {
        let n = 120usize;
        let n_train = 90usize;
        let m = 24usize;
        let mut rng = 77712345u64;
        let mut z = vec![0.0_f64; n * m];
        for v in &mut z {
            *v = lcg_normal(&mut rng);
        }
        let k_a = linear_kernel_from_embeddings(&z, n, m, true, true).expect("K_A");
        let (k_rbf, h) = rbf_kernel_from_grm(&k_a, n, None).expect("K_RBF");
        assert!(h > 0.0 && h.is_finite());

        let mut l_a = k_a.clone();
        let mut l_r = k_rbf.clone();
        for i in 0..n {
            l_a[i * n + i] += 1e-5;
            l_r[i * n + i] += 1e-5;
        }
        blocked_cholesky_lower_inplace(&mut l_a, n).expect("chol K_A");
        blocked_cholesky_lower_inplace(&mut l_r, n).expect("chol K_RBF");
        let ea: Vec<f64> = (0..n).map(|_| lcg_normal(&mut rng)).collect();
        let er: Vec<f64> = (0..n).map(|_| lcg_normal(&mut rng)).collect();
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let mut ua = 0.0;
            let mut ur = 0.0;
            for j in 0..=i {
                ua += l_a[i * n + j] * ea[j];
                ur += l_r[i * n + j] * er[j];
            }
            let eps = 0.1_f64.sqrt() * lcg_normal(&mut rng);
            y[i] = 1.0 + 0.15_f64.sqrt() * ua + 0.75_f64.sqrt() * ur + eps;
        }

        let train_idx: Vec<usize> = (0..n_train).collect();
        let test_idx: Vec<usize> = (n_train..n).collect();
        let y_tr = &y[..n_train];

        let ka_tr = submatrix(&k_a, n, &train_idx, &train_idx);
        let ka_te = submatrix(&k_a, n, &test_idx, &train_idx);
        let kr_tr = submatrix(&k_rbf, n, &train_idx, &train_idx);
        let kr_te = submatrix(&k_rbf, n, &test_idx, &train_idx);

        let cfg = MkBlupFitConfig::default();
        let fit_1k = mkblup_fit_reml(&[&ka_tr], n_train, y_tr, None, &cfg).expect("1k fit");
        let fit_2k = mkblup_fit_reml(&[&ka_tr, &kr_tr], n_train, y_tr, None, &cfg).expect("2k fit");
        assert!(fit_1k.converged);
        assert!(fit_2k.converged);
        assert!(
            fit_2k.reml > fit_1k.reml + 1.0,
            "expected 2k REML ({}) > 1k REML ({}) + 1.0",
            fit_2k.reml,
            fit_1k.reml
        );
        assert!(fit_2k.h2_components[1] > 0.2, "expected non-trivial RBF h2");

        let pred_2k = mkblup_predict(
            &[&ka_te, &kr_te],
            test_idx.len(),
            n_train,
            &fit_2k.sigma2,
            &fit_2k.alpha,
            &fit_2k.beta,
            None,
        )
        .expect("2k pred");
        for i in 0..test_idx.len() {
            let mut manual = fit_2k.beta[0];
            for j in 0..n_train {
                manual += fit_2k.sigma2[0] * ka_te[i * n_train + j] * fit_2k.alpha[j];
                manual += fit_2k.sigma2[1] * kr_te[i * n_train + j] * fit_2k.alpha[j];
            }
            assert!((pred_2k[i] - manual).abs() < 1e-12);
        }
    }

    #[test]
    fn mkblup_three_kernel_with_embedding_kernel_and_validation_guards() {
        let n = 64usize;
        let mut rng = 99887766u64;
        let z_snp: Vec<f64> = (0..n * 40).map(|_| lcg_normal(&mut rng)).collect();
        let z_jepa: Vec<f64> = (0..n * 16).map(|_| lcg_normal(&mut rng)).collect();
        let k_a = linear_kernel_from_embeddings(&z_snp, n, 40, true, true).expect("K_A");
        let (k_rbf, _) = rbf_kernel_from_grm(&k_a, n, None).expect("K_RBF");
        let k_jepa = linear_kernel_from_embeddings(&z_jepa, n, 16, true, true).expect("K_JEPA");
        let k_arc = arccosine_kernel_from_grm(&k_a, n, 1).expect("K_Arc");
        let k_aa = epistasis_kernel_from_grm(&k_a, n).expect("K_AA");

        assert!((trace_f64(&k_arc, n) / (n as f64) - 1.0).abs() < 1e-10);
        assert!((trace_f64(&k_aa, n) / (n as f64) - 1.0).abs() < 1e-10);
        assert!((k_rbf[n + 2] - k_rbf[2 * n + 1]).abs() < 1e-12);

        let mut x_cov = vec![0.0_f64; n * 2];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            x_cov[i * 2] = 1.0;
            x_cov[i * 2 + 1] = lcg_normal(&mut rng);
            let eps = 0.3 * lcg_normal(&mut rng);
            y[i] = 1.2 - 0.7 * x_cov[i * 2 + 1] + 0.5 * z_jepa[i * 16] + eps;
        }
        let cfg = MkBlupFitConfig::default();
        let fit = mkblup_fit_reml(&[&k_a, &k_rbf, &k_jepa], n, &y, Some((&x_cov, 2)), &cfg)
            .expect("3k fit");
        assert!(fit.converged);
        assert_eq!(fit.sigma2.len(), 3);
        assert_eq!(fit.beta.len(), 2);
        assert!(fit.h2_total >= 0.0 && fit.h2_total <= 1.0 + 1e-9);
        assert!((fit.beta[1] - (-0.7)).abs() < 0.35);

        // Input validation guards: reject NaN, asymmetric, negative diag, and negative sigma2.
        let mut k_nan = k_a.clone();
        k_nan[1] = f64::NAN;
        assert!(rbf_kernel_from_grm(&k_nan, n, None).is_err());
        assert!(epistasis_kernel_from_grm(&k_nan, n).is_err());
        assert!(arccosine_kernel_from_grm(&k_nan, n, 1).is_err());

        let mut k_asym = k_a.clone();
        k_asym[1] += 1.0;
        assert!(mkblup_fit_reml(&[&k_asym], n, &y, None, &cfg).is_err());

        // Small-scale and tiny-scale asymmetric matrices must both be rejected.
        for scale in [1e-6_f64, 1e-15_f64] {
            let k_small_asym = [scale, 0.9 * scale, 0.1 * scale, scale];
            assert!(rbf_kernel_from_grm(&k_small_asym, 2, None).is_err());
            assert!(epistasis_kernel_from_grm(&k_small_asym, 2).is_err());
            assert!(arccosine_kernel_from_grm(&k_small_asym, 2, 1).is_err());
            assert!(mkblup_fit_reml(&[&k_small_asym], 2, &[1.0, 2.0], None, &cfg).is_err());
        }

        // Tiny and heterogeneous-scale negative diagonals and asymmetries must be rejected.
        let k_tiny_neg = [-1e-16_f64, 0.0, 0.0, 1e-16];
        assert!(mkblup_fit_reml(&[&k_tiny_neg], 2, &[1.0, 2.0], None, &cfg).is_err());
        let k_het_neg = [1e12_f64, 0.0, 0.0, -1.0];
        assert!(rbf_kernel_from_grm(&k_het_neg, 2, None).is_err());
        assert!(mkblup_fit_reml(&[&k_het_neg], 2, &[1.0, 2.0], None, &cfg).is_err());
        let k_het_asym = [1e12_f64, 0.0, 1e-3, 1.0];
        assert!(rbf_kernel_from_grm(&k_het_asym, 2, None).is_err());
        assert!(mkblup_fit_reml(&[&k_het_asym], 2, &[1.0, 2.0], None, &cfg).is_err());

        let mut k_neg_diag = k_a.clone();
        k_neg_diag[0] = -0.5;
        assert!(mkblup_fit_reml(&[&k_neg_diag], n, &y, None, &cfg).is_err());

        assert!(mkblup_predict(&[&k_a], n, n, &[-0.1], &fit.alpha, &[0.0], None).is_err());
        assert!(mkblup_predict(&[&k_nan], n, n, &[0.5], &fit.alpha, &[0.0], None).is_err());
    }
}
