use super::*;
use crate::bedmath::decode_plink_bed_hardcall;
use crate::gs_native::grm_rankk_update;

#[allow(clippy::too_many_arguments)]
pub(crate) fn gblup_marker_fast_packed(
    threads: usize,
    eff_m: usize,
    n_train: usize,
    train_idx: &[usize],
    y_vec: &[f64],
    test_idx: &[usize],
    train_pred_abs: &[usize],
    sample_chunk: usize,
    packed_keep: &[u8],
    bytes_per_snp: usize,
    row_flip_keep: &[bool],
    maf_keep: &[f32],
    g_eps: f64,
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
    pool_ref: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64, f64, f64, String, f64), String> {
    // Enforce OpenBLAS threads inside Rust hot path so eig stage
    // is not affected by external Python-side thread drift.
    let _blas_guard = OpenBlasThreadGuard::enter(threads.max(1));
    let m = eff_m;
    let n = n_train;
    if n <= 1 || m == 0 {
        return Err("GBLUP marker-space FaST requires n>1 and m>0.".to_string());
    }

    let y_sum = y_vec.iter().sum::<f64>();
    let y_mean = y_sum / (n as f64);
    let y_center_ss = y_vec
        .iter()
        .map(|v| {
            let d = *v - y_mean;
            d * d
        })
        .sum::<f64>();

    let mut gram_raw = vec![0.0_f32; m * m];
    let mut row_sum = vec![0.0_f64; m];
    let mut row_sq_sum = vec![0.0_f64; m];
    let mut my_raw = vec![0.0_f64; m];
    let mut decode_blk = vec![0.0_f32; m * sample_chunk];
    let mut gram_blk_t = vec![0.0_f32; sample_chunk * m];
    let mut tick = 0usize;

    for st in (0..n).step_by(sample_chunk) {
        let ed = (st + sample_chunk).min(n);
        let cur = ed - st;
        let idx_blk = &train_idx[st..ed];
        let y_blk = &y_vec[st..ed];
        let blk_slice = &mut decode_blk[..m * cur];
        let mut blk_sum = vec![0.0_f64; m];
        let mut blk_sq = vec![0.0_f64; m];
        let mut blk_my = vec![0.0_f64; m];

        let mut decode_run = || {
            blk_slice
                .par_chunks_mut(cur)
                .zip(blk_sum.par_iter_mut())
                .zip(blk_sq.par_iter_mut())
                .zip(blk_my.par_iter_mut())
                .enumerate()
                .for_each(|(row_idx, (((out_row, sum_dst), sq_dst), my_dst))| {
                    let row = &packed_keep[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                    let flip = row_flip_keep[row_idx];
                    let mean_g = 2.0_f64 * (maf_keep[row_idx] as f64);
                    let mut sum = 0.0_f64;
                    let mut sq = 0.0_f64;
                    let mut my = 0.0_f64;
                    for (j, &sid) in idx_blk.iter().enumerate() {
                        let b = row[sid >> 2];
                        let code = (b >> ((sid & 3) * 2)) & 0b11;
                        let mut gv = match decode_plink_bed_hardcall(code) {
                            Some(v) => v,
                            None => mean_g,
                        };
                        if flip && code != 0b01 {
                            gv = 2.0_f64 - gv;
                        }
                        out_row[j] = gv as f32;
                        sum += gv;
                        sq += gv * gv;
                        my += gv * y_blk[j];
                    }
                    *sum_dst = sum;
                    *sq_dst = sq;
                    *my_dst = my;
                });
        };
        if let Some(tp) = pool_ref {
            tp.install(&mut decode_run);
        } else {
            decode_run();
        }

        for j in 0..m {
            row_sum[j] += blk_sum[j];
            row_sq_sum[j] += blk_sq[j];
            my_raw[j] += blk_my[j];
        }

        let blk_t = &mut gram_blk_t[..cur * m];
        for row in 0..m {
            let src = &blk_slice[row * cur..(row + 1) * cur];
            for col in 0..cur {
                blk_t[col * m + row] = src[col];
            }
        }
        grm_rankk_update(&mut gram_raw, blk_t, cur, m, false, true)?;

        tick += cur;
        if tick >= sample_chunk.saturating_mul(16).max(1) {
            check_ctrlc()?;
            tick = 0;
        }
    }

    let mut m_mean = vec![0.0_f64; m];
    let n_f = n as f64;
    for j in 0..m {
        m_mean[j] = row_sum[j] / n_f;
    }
    let mut var_sum = 0.0_f64;
    for j in 0..m {
        let v = row_sq_sum[j] / n_f - m_mean[j] * m_mean[j];
        if v.is_finite() && v > 0.0 {
            var_sum += v;
        }
    }
    if !(var_sum.is_finite() && var_sum > 0.0) {
        var_sum = 1e-12_f64;
    }
    let scale = var_sum.sqrt().max(1e-12_f64);

    let mut gram = vec![0.0_f64; m * m];
    for i in 0..m {
        let mi = m_mean[i];
        for j in 0..m {
            let idx = i * m + j;
            let centered = (gram_raw[idx] as f64) - n_f * mi * m_mean[j];
            gram[idx] = centered / var_sum;
        }
    }
    for i in 0..m {
        gram[i * m + i] += g_eps;
    }
    for i in 0..m {
        for j in 0..i {
            let v = 0.5_f64 * (gram[i * m + j] + gram[j * m + i]);
            gram[i * m + j] = v;
            gram[j * m + i] = v;
        }
    }

    let mut my_centered = vec![0.0_f64; m];
    for j in 0..m {
        my_centered[j] = (my_raw[j] - m_mean[j] * y_sum) / scale;
    }

    let t_evd = Instant::now();
    let (eval_all, evec_all, evd_backend_inner) = symmetric_eigh_f64_row_major(&gram, m)?;
    let evd_elapsed_inner = t_evd.elapsed().as_secs_f64();
    if eval_all.is_empty() {
        return Err("marker-space FaST eigendecomposition returned empty spectrum.".to_string());
    }
    let max_eval = *eval_all.last().unwrap_or(&0.0_f64);
    let tol_eval = f64::EPSILON * max_eval.max(1.0_f64) * (m.max(1) as f64);
    let mut keep_start = 0usize;
    while keep_start < eval_all.len() && eval_all[keep_start] <= tol_eval {
        keep_start += 1;
    }
    if keep_start >= eval_all.len() {
        return Err("Numerically singular marker matrix in packed marker-space FaST.".to_string());
    }

    let r = eval_all.len() - keep_start;
    let mut s = vec![0.0_f64; r];
    let mut v = vec![0.0_f64; m * r];
    for k in 0..r {
        s[k] = eval_all[keep_start + k];
        for row in 0..m {
            v[row * r + k] = evec_all[row * m + (keep_start + k)];
        }
    }
    let svals: Vec<f64> = s.iter().map(|x| x.sqrt()).collect();
    let inv_s: Vec<f64> = svals.iter().map(|x| 1.0_f64 / x.max(1e-18_f64)).collect();

    let mut y_proj = vec![0.0_f64; r];
    for k in 0..r {
        let mut acc = 0.0_f64;
        for row in 0..m {
            acc += v[row * r + k] * my_centered[row];
        }
        y_proj[k] = acc * inv_s[k];
    }
    let y_proj_sq = y_proj.iter().map(|x| x * x).sum::<f64>();
    let etas2_sq = (y_center_ss - y_proj_sq).max(0.0_f64);
    let n_null = n.saturating_sub(r + 1);

    let v_floor = 1e-12_f64;
    let n_eff = (n.saturating_sub(1)).max(1) as f64;
    let c_reml = n_eff * (n_eff.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let c_ml = (n as f64) * ((n as f64).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let eval_reml = |log10_lbd: f64| -> Option<(f64, f64, f64)> {
        let lbd = 10.0_f64.powf(log10_lbd);
        if !(lbd.is_finite() && lbd > 0.0) {
            return None;
        }
        let mut quad_main = 0.0_f64;
        let mut log_det_v = 0.0_f64;
        for k in 0..r {
            let vk = (s[k] + lbd).max(v_floor);
            quad_main += (y_proj[k] * y_proj[k]) / vk;
            log_det_v += vk.ln();
        }
        if n_null > 0 {
            quad_main += etas2_sq / lbd.max(v_floor);
            log_det_v += (n_null as f64) * lbd.ln();
        }
        let rtv = quad_main.max(v_floor);
        let reml = c_reml - 0.5_f64 * (n_eff * rtv.ln() + log_det_v);
        let ml = c_ml - 0.5_f64 * (((n as f64) * rtv.ln()) + log_det_v);
        if !(reml.is_finite() && ml.is_finite()) {
            return None;
        }
        Some((reml, ml, rtv))
    };

    let (best_log10, _best_cost) = brent_minimize(
        |x0| match eval_reml(x0) {
            Some((reml_v, _, _)) => -reml_v,
            None => 1e100,
        },
        low,
        high,
        tol,
        max_iter,
    );
    let (reml_best, ml_best, rtv_invr) = eval_reml(best_log10)
        .ok_or_else(|| "GBLUP marker-space REML optimization failed.".to_string())?;
    let lambda = 10.0_f64.powf(best_log10);

    let mut rhs = vec![0.0_f64; r];
    for k in 0..r {
        rhs[k] = y_proj[k] / (s[k] + lambda).max(v_floor);
    }
    let mut tmp_u = vec![0.0_f64; r];
    for k in 0..r {
        tmp_u[k] = inv_s[k] * ((s[k] + g_eps) * rhs[k]);
    }
    let mut q_u = vec![0.0_f64; m];
    for row in 0..m {
        let mut acc = 0.0_f64;
        for k in 0..r {
            acc += v[row * r + k] * tmp_u[k];
        }
        q_u[row] = acc;
    }
    let mu_dot_u = m_mean
        .iter()
        .zip(q_u.iter())
        .map(|(a, b)| (*a) * (*b))
        .sum::<f64>();

    let predict_samples = |sample_abs: &[usize]| -> Result<Vec<f64>, String> {
        let n_out = sample_abs.len();
        if n_out == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0.0_f64; n_out];
        let mut run = || {
            out.par_iter_mut().enumerate().for_each(|(i, dst)| {
                let sid = sample_abs[i];
                let mut acc = 0.0_f64;
                for row_idx in 0..m {
                    let row = &packed_keep[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                    let b = row[sid >> 2];
                    let code = (b >> ((sid & 3) * 2)) & 0b11;
                    let mean_g = 2.0_f64 * (maf_keep[row_idx] as f64);
                    let mut gv = match decode_plink_bed_hardcall(code) {
                        Some(v0) => v0,
                        None => mean_g,
                    };
                    if row_flip_keep[row_idx] && code != 0b01 {
                        gv = 2.0_f64 - gv;
                    }
                    acc += gv * q_u[row_idx];
                }
                *dst = y_mean + (acc - mu_dot_u) / scale;
            });
        };
        if let Some(tp) = pool_ref {
            tp.install(&mut run);
        } else {
            run();
        }
        Ok(out)
    };

    let pred_train_local = predict_samples(train_pred_abs)?;
    let pred_test_local = predict_samples(test_idx)?;

    let sigma_g2 = rtv_invr / n_eff.max(1.0);
    let sigma_e2 = lambda * sigma_g2;
    let mean_s = s.iter().copied().sum::<f64>() / (n as f64);
    let var_g = sigma_g2 * mean_s.max(0.0_f64);
    let denom = var_g + sigma_e2;
    let pve = if denom.is_finite() && denom > 0.0 {
        var_g / denom
    } else {
        f64::NAN
    };

    Ok((
        pred_train_local,
        pred_test_local,
        pve,
        lambda,
        ml_best,
        reml_best,
        format!("marker_fast:{}", evd_backend_inner),
        evd_elapsed_inner,
    ))
}
