use std::f64::consts::PI;

use crate::linalg::{cholesky_inplace, cholesky_logdet, cholesky_solve_into};

#[derive(Clone, Copy, Debug)]
pub struct AiRemlNullResult {
    pub lbd: f64,
    pub ml: f64,
    pub reml: f64,
    pub sigma_g2: f64,
    pub sigma_e2: f64,
    pub used_iter: usize,
    pub converged: bool,
}

#[inline]
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

    let mut tr_cm = 0.0_f64;
    for r in 0..p {
        for c in 0..p {
            tr_cm += c_inv[r * p + c] * m[c * p + r];
        }
    }
    tr_wd - tr_cm
}

#[allow(clippy::type_complexity)]
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

pub fn ai_reml_null_from_spectral(
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    n: usize,
    p_cov: usize,
    max_iter: usize,
    tol: f64,
    min_var: f64,
) -> Result<AiRemlNullResult, String> {
    if y.len() != n {
        return Err(format!(
            "ai_reml_null_from_spectral: y length mismatch: got {}, expected {n}",
            y.len()
        ));
    }
    if s.len() != n {
        return Err(format!(
            "ai_reml_null_from_spectral: s length mismatch: got {}, expected {n}",
            s.len()
        ));
    }
    if xcov.len() != n.saturating_mul(p_cov) {
        return Err(format!(
            "ai_reml_null_from_spectral: xcov length mismatch: got {}, expected {}",
            xcov.len(),
            n.saturating_mul(p_cov)
        ));
    }
    if n <= p_cov {
        return Err("ai_reml_null_from_spectral: n must be > p_cov".to_string());
    }
    if max_iter == 0 {
        return Err("ai_reml_null_from_spectral: max_iter must be > 0".to_string());
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err("ai_reml_null_from_spectral: y contains non-finite values".to_string());
    }
    if s.iter().any(|v| !v.is_finite()) {
        return Err("ai_reml_null_from_spectral: s contains non-finite values".to_string());
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

    let mut state = ai_reml_eval(s, xcov, y, n, p_cov, sigma_g2, sigma_e2)
        .ok_or_else(|| "AIREML init failed".to_string())?;

    for it in 0..max_iter {
        used_iter = it + 1;
        let (reml_curr, _ml_curr, _q_curr, w, z, c_inv, _beta) = &state;

        let tr_g = trace_p_d(s, xcov, n, p_cov, w, c_inv, true);
        let tr_e = trace_p_d(s, xcov, n, p_cov, w, c_inv, false);

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
            xcov,
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
            xcov,
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
            if let Some(st) = ai_reml_eval(s, xcov, y, n, p_cov, cand_sg, cand_se) {
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

    Ok(AiRemlNullResult {
        lbd,
        ml: state.1,
        reml: state.0,
        sigma_g2: sigma_g2_out,
        sigma_e2: sigma_e2_out,
        used_iter,
        converged,
    })
}
