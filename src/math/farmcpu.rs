use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

use crate::bedmath::decode_plink_bed_hardcall;

#[inline]
pub(crate) fn decode_packed_rows_to_sample_major(
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
pub(crate) fn decode_dense_rows_to_sample_major(
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

pub(crate) fn select_lead_indices(
    sz: i64,
    n_lead: usize,
    pvalue: &[f64],
    pos: &[i64],
) -> Vec<usize> {
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

pub(crate) fn farmcpu_ll_score_from_sample_major(
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

pub(crate) fn farmcpu_super_keep_from_sample_major(
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
