use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::{prelude::*, BoundObject};
use rand::rngs::{OsRng, StdRng};
use rand::{Rng, SeedableRng, TryRngCore};
use rand_distr::{Beta, ChiSquared, Gamma, StandardNormal};
use rayon::prelude::*;
use std::borrow::Cow;
use std::env;
use std::sync::OnceLock;

fn array1_to_vec(arr: &PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn array2_to_vec(arr: &PyReadonlyArray2<f64>) -> Vec<f64> {
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

fn parse_index_vec_i64(indices: &[i64], upper_bound: usize, label: &str) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(indices.len());
    for (i, &v) in indices.iter().enumerate() {
        if v < 0 {
            return Err(PyValueError::new_err(format!(
                "{label}[{i}] must be >= 0, got {v}"
            )));
        }
        let u = v as usize;
        if u >= upper_bound {
            return Err(PyValueError::new_err(format!(
                "{label}[{i}] out of range: {u} >= {upper_bound}"
            )));
        }
        out.push(u);
    }
    Ok(out)
}

struct PackedByteLut {
    code4: [[u8; 4]; 256],
}

fn packed_byte_lut() -> &'static PackedByteLut {
    static LUT: OnceLock<PackedByteLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut code4 = [[0u8; 4]; 256];
        for b in 0u16..=255 {
            let byte = b as u8;
            for lane in 0..4usize {
                code4[byte as usize][lane] = (byte >> (lane * 2)) & 0b11;
            }
        }
        PackedByteLut { code4 }
    })
}

struct SampleByteDecodePlan {
    group_bytes: Vec<usize>,
    group_offsets: Vec<usize>,
    out_pos: Vec<usize>,
    lanes: Vec<u8>,
    group_fast_lane0123: Vec<bool>,
    identity_order: bool,
}

fn build_sample_byte_decode_plan(sample_idx: &[usize]) -> SampleByteDecodePlan {
    let n = sample_idx.len();
    let mut entries: Vec<(usize, u8, usize)> = Vec::with_capacity(n);
    for (out_pos, &sid) in sample_idx.iter().enumerate() {
        entries.push((sid >> 2, (sid & 3) as u8, out_pos));
    }
    entries.sort_unstable_by_key(|(byte_idx, _, _)| *byte_idx);

    let mut group_bytes: Vec<usize> = Vec::new();
    let mut group_offsets: Vec<usize> = Vec::new();
    let mut out_pos: Vec<usize> = Vec::with_capacity(n);
    let mut lanes: Vec<u8> = Vec::with_capacity(n);

    group_offsets.push(0);
    let mut last_byte: Option<usize> = None;
    for (byte_idx, lane, out) in entries {
        if last_byte != Some(byte_idx) {
            group_bytes.push(byte_idx);
            if !out_pos.is_empty() {
                group_offsets.push(out_pos.len());
            }
            last_byte = Some(byte_idx);
        }
        out_pos.push(out);
        lanes.push(lane);
    }
    group_offsets.push(out_pos.len());

    let identity_order = out_pos.iter().enumerate().all(|(i, &v)| i == v);
    let mut group_fast_lane0123 = Vec::with_capacity(group_bytes.len());
    for g in 0..group_bytes.len() {
        let st = group_offsets[g];
        let ed = group_offsets[g + 1];
        let fast = (ed - st) == 4
            && lanes[st] == 0
            && lanes[st + 1] == 1
            && lanes[st + 2] == 2
            && lanes[st + 3] == 3;
        group_fast_lane0123.push(fast);
    }

    SampleByteDecodePlan {
        group_bytes,
        group_offsets,
        out_pos,
        lanes,
        group_fast_lane0123,
        identity_order,
    }
}

fn parse_env_truthy(raw: &str) -> Option<bool> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    if matches!(s.as_str(), "1" | "true" | "yes" | "y" | "on") {
        return Some(true);
    }
    if matches!(s.as_str(), "0" | "false" | "no" | "n" | "off") {
        return Some(false);
    }
    None
}

fn bayes_packed_parallel_decode_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        if let Ok(raw) = env::var("JX_BAYES_PAR_DECODE") {
            if let Some(v) = parse_env_truthy(&raw) {
                return v;
            }
        }
        true
    })
}

#[inline]
fn bayes_packed_block_rows(n: usize, p: usize) -> usize {
    if n == 0 || p == 0 {
        return 1;
    }
    let default_target_bytes: usize = 32 * 1024 * 1024;
    let bytes_per_row = n.saturating_mul(std::mem::size_of::<f64>()).max(1);
    let mut rows = (default_target_bytes / bytes_per_row).max(1);
    rows = rows.clamp(8, 2048);
    if let Ok(raw) = env::var("JX_BAYES_PACKED_ROW_BLOCK") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            if v > 0 {
                rows = v;
            }
        }
    }
    rows.clamp(1, p)
}

#[inline]
fn decode_packed_standardized_row(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_idx: usize,
    sample_plan: &SampleByteDecodePlan,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    code4_lut: &[[u8; 4]; 256],
    dst: &mut [f64],
) {
    let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
    let flip = row_flip[row_idx];
    let mean_g = 2.0_f64 * row_maf[row_idx] as f64;
    let mu = row_mean[row_idx] as f64;
    let inv = row_inv_sd[row_idx] as f64;
    let (g0, g2) = if flip {
        (2.0_f64, 0.0_f64)
    } else {
        (0.0_f64, 2.0_f64)
    };
    let code_val = [
        (g0 - mu) * inv,      // 0b00
        (mean_g - mu) * inv,  // 0b01 (missing -> mean impute)
        (1.0_f64 - mu) * inv, // 0b10
        (g2 - mu) * inv,      // 0b11
    ];

    if sample_plan.identity_order {
        let mut k = 0usize;
        for g in 0..sample_plan.group_bytes.len() {
            let codes = &code4_lut[row[sample_plan.group_bytes[g]] as usize];
            if sample_plan.group_fast_lane0123[g] {
                dst[k] = code_val[codes[0] as usize];
                dst[k + 1] = code_val[codes[1] as usize];
                dst[k + 2] = code_val[codes[2] as usize];
                dst[k + 3] = code_val[codes[3] as usize];
                k += 4;
            } else {
                let st = sample_plan.group_offsets[g];
                let ed = sample_plan.group_offsets[g + 1];
                for t in st..ed {
                    dst[k] = code_val[codes[sample_plan.lanes[t] as usize] as usize];
                    k += 1;
                }
            }
        }
    } else {
        for g in 0..sample_plan.group_bytes.len() {
            let codes = &code4_lut[row[sample_plan.group_bytes[g]] as usize];
            let st = sample_plan.group_offsets[g];
            let ed = sample_plan.group_offsets[g + 1];
            for t in st..ed {
                dst[sample_plan.out_pos[t]] =
                    code_val[codes[sample_plan.lanes[t] as usize] as usize];
            }
        }
    }
}

#[inline]
fn decode_packed_block_standardized_into(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_start: usize,
    row_end: usize,
    sample_plan: &SampleByteDecodePlan,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    code4_lut: &[[u8; 4]; 256],
    out_block: &mut [f64],
    n: usize,
) {
    let block_rows = row_end - row_start;
    debug_assert_eq!(out_block.len(), block_rows * n);
    debug_assert_eq!(sample_plan.out_pos.len(), n);

    let use_parallel_decode = bayes_packed_parallel_decode_enabled()
        && block_rows >= 32
        && n >= 256
        && rayon::current_num_threads() > 1;
    if use_parallel_decode {
        out_block
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(off, dst)| {
                let row_idx = row_start + off;
                decode_packed_standardized_row(
                    packed_flat,
                    bytes_per_snp,
                    row_idx,
                    sample_plan,
                    row_flip,
                    row_maf,
                    row_mean,
                    row_inv_sd,
                    code4_lut,
                    dst,
                );
            });
        return;
    }

    for off in 0..block_rows {
        let row_idx = row_start + off;
        let dst = &mut out_block[off * n..(off + 1) * n];
        decode_packed_standardized_row(
            packed_flat,
            bytes_per_snp,
            row_idx,
            sample_plan,
            row_flip,
            row_maf,
            row_mean,
            row_inv_sd,
            code4_lut,
            dst,
        );
    }
}

fn genetic_variance_from_residual(
    y: &[f64],
    r: &[f64],
    x: &[f64],
    alpha: &[f64],
    n: usize,
    q: usize,
) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut mean_g = 0.0;
    let mut m2 = 0.0;
    for i in 0..n {
        let mut xa = 0.0;
        for k in 0..q {
            xa += x[i * q + k] * alpha[k];
        }
        let g = y[i] - r[i] - xa;
        let delta = g - mean_g;
        mean_g += delta / (i as f64 + 1.0);
        let delta2 = g - mean_g;
        m2 += delta * delta2;
    }
    m2 / (n as f64 - 1.0)
}

fn bayesb_core_impl(
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
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
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
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
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
    let mut d = vec![0u8; p];
    for j in 0..p {
        d[j] = if rng.random::<f64>() < prob_in_init {
            1
        } else {
            0
        };
    }
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;
    let mut prob_in = prob_in_init;

    let counts_in = counts * prob_in_init;
    let counts_out = counts - counts_in;

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
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();
        let c1 = -0.5 * inv_var_e;

        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let mut xe = 0.0;
            for i in 0..n {
                xe += r[i] * m_j[i];
            }
            let b = beta[j];
            let d_old = d[j];
            let d_rss = if d_old == 1 {
                -b * b * x2[j] - 2.0 * b * xe
            } else {
                b * b * x2[j] - 2.0 * b * xe
            };
            let log_odds = log_odds_prior + c1 * d_rss;
            let p_in = if log_odds >= 0.0 {
                1.0 / (1.0 + (-log_odds).exp())
            } else {
                let e = log_odds.exp();
                e / (1.0 + e)
            };
            let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };

            if new_d != d_old {
                if new_d > d_old {
                    let delta = -b;
                    for i in 0..n {
                        r[i] += delta * m_j[i];
                    }
                    // After r <- r - b * m_j, update xe = <r, m_j> algebraically.
                    // This is equivalent to recomputing the full dot product, but avoids
                    // an extra O(n) pass on every 0->1 indicator switch.
                    xe -= b * x2[j];
                } else {
                    let delta = b;
                    for i in 0..n {
                        r[i] += delta * m_j[i];
                    }
                }
            }
            d[j] = new_d;

            if d[j] == 0 {
                let z_beta: f64 = rng.sample(StandardNormal);
                beta[j] = var_b[j].sqrt() * z_beta;
            } else {
                let rhs = (x2[j] * b + xe) * inv_var_e;
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                let z_beta: f64 = rng.sample(StandardNormal);
                let tmp = rhs / c + (1.0 / c).sqrt() * z_beta;

                let delta = b - tmp;
                for i in 0..n {
                    r[i] += delta * m_j[i];
                }
                beta[j] = tmp;
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

        let mut mrk_in = 0.0;
        for j in 0..p {
            mrk_in += d[j] as f64;
        }
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

        let mut ss_e = 0.0;
        for i in 0..n {
            ss_e += r[i] * r[i];
        }
        ss_e += s0_e;
        var_e = ss_e / rng.sample(chi_e);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }

    Ok((beta_sum, alpha_sum, varb_sum, var_e_sum, h2_mean, var_h2))
}

fn bayescpi_core_impl(
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
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
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
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
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
    let mut d = vec![0u8; p];
    for j in 0..p {
        d[j] = if rng.random::<f64>() < prob_in_init {
            1
        } else {
            0
        };
    }
    let mut var_b = s0_b;
    let mut prob_in = prob_in_init;

    let counts_in = counts * prob_in_init;
    let counts_out = counts - counts_in;

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
    let mut varb_sum = 0.0;
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + p as f64).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();
        let c1 = -0.5 * inv_var_e;

        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let mut xe = 0.0;
            for i in 0..n {
                xe += r[i] * m_j[i];
            }
            let b = beta[j];
            let d_old = d[j];
            let d_rss = if d_old == 1 {
                -b * b * x2[j] - 2.0 * b * xe
            } else {
                b * b * x2[j] - 2.0 * b * xe
            };
            let log_odds = log_odds_prior + c1 * d_rss;
            let p_in = if log_odds >= 0.0 {
                1.0 / (1.0 + (-log_odds).exp())
            } else {
                let e = log_odds.exp();
                e / (1.0 + e)
            };
            let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };

            if new_d != d_old {
                if new_d > d_old {
                    let delta = -b;
                    for i in 0..n {
                        r[i] += delta * m_j[i];
                    }
                    // After r <- r - b * m_j, update xe = <r, m_j> algebraically.
                    // This avoids an extra O(n) dot-product pass when toggling 0->1.
                    xe -= b * x2[j];
                } else {
                    let delta = b;
                    for i in 0..n {
                        r[i] += delta * m_j[i];
                    }
                }
            }
            d[j] = new_d;

            if d[j] == 0 {
                let z_beta: f64 = rng.sample(StandardNormal);
                beta[j] = var_b.sqrt() * z_beta;
            } else {
                let rhs = (x2[j] * b + xe) * inv_var_e;
                let c = x2[j] * inv_var_e + 1.0 / var_b;
                let z_beta: f64 = rng.sample(StandardNormal);
                let tmp = rhs / c + (1.0 / c).sqrt() * z_beta;

                let delta = b - tmp;
                for i in 0..n {
                    r[i] += delta * m_j[i];
                }
                beta[j] = tmp;
            }
        }

        let mut ss_b = 0.0;
        for j in 0..p {
            ss_b += beta[j] * beta[j];
        }
        ss_b += s0_b;
        var_b = ss_b / rng.sample(chi_b);

        let mut mrk_in = 0.0;
        for j in 0..p {
            mrk_in += d[j] as f64;
        }
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

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
            varb_sum += var_b;
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
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
    varb_sum *= inv_keep;
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }

    Ok((beta_sum, alpha_sum, varb_sum, var_e_sum, h2_mean, var_h2))
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
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64), String> {
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
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
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
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
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
            rhs = rhs * inv_var_e + x2[j] * beta[j] * inv_var_e;
            let c = x2[j] * inv_var_e + 1.0 / var_b[j];
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
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    Ok((beta_sum, alpha_sum, varb_sum, var_e_sum, h2_mean, var_h2))
}

fn bayesa_packed_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
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
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let sample_plan = build_sample_byte_decode_plan(sample_idx);
    let code4_lut = &packed_byte_lut().code4;
    let block_rows = bayes_packed_block_rows(n, p);
    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    let mut m_block = vec![0.0_f64; block_rows * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_packed_block_standardized_into(
            packed_flat,
            bytes_per_snp,
            st,
            ed,
            &sample_plan,
            row_flip,
            row_maf,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut m_block[..br * n],
            n,
        );
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += v * v;
                msum += v;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
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
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
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

        for st in (0..p).step_by(block_rows) {
            let ed = (st + block_rows).min(p);
            let br = ed - st;
            decode_packed_block_standardized_into(
                packed_flat,
                bytes_per_snp,
                st,
                ed,
                &sample_plan,
                row_flip,
                row_maf,
                row_mean,
                row_inv_sd,
                code4_lut,
                &mut m_block[..br * n],
                n,
            );
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let mut rhs = 0.0;
                for i in 0..n {
                    rhs += m_row[i] * r[i];
                }
                rhs = rhs * inv_var_e + x2[j] * beta[j] * inv_var_e;
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                let z_beta: f64 = rng.sample(StandardNormal);
                let new_beta = rhs / c + (1.0 / c).sqrt() * z_beta;

                let delta = beta[j] - new_beta;
                for i in 0..n {
                    r[i] += delta * m_row[i];
                }
                beta[j] = new_beta;
                if beta[j].abs() < min_abs_beta {
                    beta[j] = min_abs_beta;
                }
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
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    Ok((beta_sum, alpha_sum, varb_sum, var_e_sum, h2_mean, var_h2))
}

fn bayesb_packed_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
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
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let sample_plan = build_sample_byte_decode_plan(sample_idx);
    let code4_lut = &packed_byte_lut().code4;
    let block_rows = bayes_packed_block_rows(n, p);
    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    let mut m_block = vec![0.0_f64; block_rows * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_packed_block_standardized_into(
            packed_flat,
            bytes_per_snp,
            st,
            ed,
            &sample_plan,
            row_flip,
            row_maf,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut m_block[..br * n],
            n,
        );
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += v * v;
                msum += v;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
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
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
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
    let mut d = vec![0u8; p];
    for item in &mut d {
        *item = if rng.random::<f64>() < prob_in_init {
            1
        } else {
            0
        };
    }
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;
    let mut prob_in = prob_in_init;
    let counts_in = counts * prob_in_init;
    let counts_out = counts - counts_in;

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
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();
        let c1 = -0.5 * inv_var_e;

        for st in (0..p).step_by(block_rows) {
            let ed = (st + block_rows).min(p);
            let br = ed - st;
            decode_packed_block_standardized_into(
                packed_flat,
                bytes_per_snp,
                st,
                ed,
                &sample_plan,
                row_flip,
                row_maf,
                row_mean,
                row_inv_sd,
                code4_lut,
                &mut m_block[..br * n],
                n,
            );
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let mut xe = 0.0;
                for i in 0..n {
                    xe += r[i] * m_row[i];
                }
                let b = beta[j];
                let d_old = d[j];
                let d_rss = if d_old == 1 {
                    -b * b * x2[j] - 2.0 * b * xe
                } else {
                    b * b * x2[j] - 2.0 * b * xe
                };
                let log_odds = log_odds_prior + c1 * d_rss;
                let p_in = if log_odds >= 0.0 {
                    1.0 / (1.0 + (-log_odds).exp())
                } else {
                    let e = log_odds.exp();
                    e / (1.0 + e)
                };
                let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };

                if new_d != d_old {
                    if new_d > d_old {
                        let delta = -b;
                        for i in 0..n {
                            r[i] += delta * m_row[i];
                        }
                        xe -= b * x2[j];
                    } else {
                        let delta = b;
                        for i in 0..n {
                            r[i] += delta * m_row[i];
                        }
                    }
                }
                d[j] = new_d;

                if d[j] == 0 {
                    let z_beta: f64 = rng.sample(StandardNormal);
                    beta[j] = var_b[j].sqrt() * z_beta;
                } else {
                    let rhs = (x2[j] * b + xe) * inv_var_e;
                    let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                    let z_beta: f64 = rng.sample(StandardNormal);
                    let tmp = rhs / c + (1.0 / c).sqrt() * z_beta;
                    let delta = b - tmp;
                    for i in 0..n {
                        r[i] += delta * m_row[i];
                    }
                    beta[j] = tmp;
                }
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

        let mut mrk_in = 0.0;
        for &dj in &d {
            mrk_in += dj as f64;
        }
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

        let mut ss_e = 0.0;
        for i in 0..n {
            ss_e += r[i] * r[i];
        }
        ss_e += s0_e;
        var_e = ss_e / rng.sample(chi_e);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept (check burnin/thin)".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }

    Ok((beta_sum, alpha_sum, varb_sum, var_e_sum, h2_mean, var_h2))
}

fn bayescpi_packed_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    df0_e: f64,
    s0_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64, f64, f64), String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let sample_plan = build_sample_byte_decode_plan(sample_idx);
    let code4_lut = &packed_byte_lut().code4;
    let block_rows = bayes_packed_block_rows(n, p);
    let mut x2 = vec![0.0; p];
    let mut mean_x = vec![0.0; p];
    let mut m_block = vec![0.0_f64; block_rows * n];
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_packed_block_standardized_into(
            packed_flat,
            bytes_per_snp,
            st,
            ed,
            &sample_plan,
            row_flip,
            row_maf,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut m_block[..br * n],
            n,
        );
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += v * v;
                msum += v;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
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
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_init
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
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
    let mut d = vec![0u8; p];
    for item in &mut d {
        *item = if rng.random::<f64>() < prob_in_init {
            1
        } else {
            0
        };
    }
    let mut var_b = s0_b;
    let mut prob_in = prob_in_init;
    let counts_in = counts * prob_in_init;
    let counts_out = counts - counts_in;

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
    let mut varb_sum = 0.0;
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + p as f64).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();
        let c1 = -0.5 * inv_var_e;

        for st in (0..p).step_by(block_rows) {
            let ed = (st + block_rows).min(p);
            let br = ed - st;
            decode_packed_block_standardized_into(
                packed_flat,
                bytes_per_snp,
                st,
                ed,
                &sample_plan,
                row_flip,
                row_maf,
                row_mean,
                row_inv_sd,
                code4_lut,
                &mut m_block[..br * n],
                n,
            );
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let mut xe = 0.0;
                for i in 0..n {
                    xe += r[i] * m_row[i];
                }
                let b = beta[j];
                let d_old = d[j];
                let d_rss = if d_old == 1 {
                    -b * b * x2[j] - 2.0 * b * xe
                } else {
                    b * b * x2[j] - 2.0 * b * xe
                };
                let log_odds = log_odds_prior + c1 * d_rss;
                let p_in = if log_odds >= 0.0 {
                    1.0 / (1.0 + (-log_odds).exp())
                } else {
                    let e = log_odds.exp();
                    e / (1.0 + e)
                };
                let new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };

                if new_d != d_old {
                    if new_d > d_old {
                        let delta = -b;
                        for i in 0..n {
                            r[i] += delta * m_row[i];
                        }
                        xe -= b * x2[j];
                    } else {
                        let delta = b;
                        for i in 0..n {
                            r[i] += delta * m_row[i];
                        }
                    }
                }
                d[j] = new_d;

                if d[j] == 0 {
                    let z_beta: f64 = rng.sample(StandardNormal);
                    beta[j] = var_b.sqrt() * z_beta;
                } else {
                    let rhs = (x2[j] * b + xe) * inv_var_e;
                    let c = x2[j] * inv_var_e + 1.0 / var_b;
                    let z_beta: f64 = rng.sample(StandardNormal);
                    let tmp = rhs / c + (1.0 / c).sqrt() * z_beta;
                    let delta = b - tmp;
                    for i in 0..n {
                        r[i] += delta * m_row[i];
                    }
                    beta[j] = tmp;
                }
            }
        }

        let mut ss_b = 0.0;
        for &bj in &beta {
            ss_b += bj * bj;
        }
        ss_b += s0_b;
        var_b = ss_b / rng.sample(chi_b);

        let mut mrk_in = 0.0;
        for &dj in &d {
            mrk_in += dj as f64;
        }
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
        prob_in = rng.sample(beta_dist);

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
            varb_sum += var_b;
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
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
    varb_sum *= inv_keep;
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }

    Ok((beta_sum, alpha_sum, varb_sum, var_e_sum, h2_mean, var_h2))
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
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
)> {
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

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_shape = m.shape();
    if m_shape[1] != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_shape[0];
    let m_vec: Cow<'_, [f64]> = match m.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array2_to_vec(&m)),
    };

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesa_core_impl(
            y_vec.as_ref(),
            m_vec.as_ref(),
            x_vec.as_ref(),
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
        Ok((beta, alpha, varb, vare, h2_mean, var_h2)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((beta_py, alpha_py, varb_py, vare, h2_mean, var_h2))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
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
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    seed = None
))]
pub fn bayesb(
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
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
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
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_shape = m.shape();
    if m_shape[1] != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_shape[0];
    let m_vec: Cow<'_, [f64]> = match m.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array2_to_vec(&m)),
    };

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesb_core_impl(
            y_vec.as_ref(),
            m_vec.as_ref(),
            x_vec.as_ref(),
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
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
        )
    });

    match result {
        Ok((beta, alpha, varb, vare, h2_mean, var_h2)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((beta_py, alpha_py, varb_py, vare, h2_mean, var_h2))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
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
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    seed = None
))]
pub fn bayescpi(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: PyReadonlyArray2<f64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, f64, f64, f64, f64)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = s0_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_e must be > 0"));
        }
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_shape = m.shape();
    if m_shape[1] != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_shape[0];
    let m_vec: Cow<'_, [f64]> = match m.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array2_to_vec(&m)),
    };

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayescpi_core_impl(
            y_vec.as_ref(),
            m_vec.as_ref(),
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
        )
    });

    match result {
        Ok((beta, alpha, varb_mean, vare, h2_mean, var_h2)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            Ok((beta_py, alpha_py, varb_mean, vare, h2_mean, var_h2))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
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
pub fn bayesa_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
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
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
)> {
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
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };

    let result = py.detach(|| {
        bayesa_packed_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
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
        Ok((beta, alpha, varb, vare, h2_mean, var_h2)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((beta_py, alpha_py, varb_py, vare, h2_mean, var_h2))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    seed = None
))]
pub fn bayesb_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
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
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };

    let result = py.detach(|| {
        bayesb_packed_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
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
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
        )
    });

    match result {
        Ok((beta, alpha, varb, vare, h2_mean, var_h2)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((beta_py, alpha_py, varb_py, vare, h2_mean, var_h2))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    df0_e = 5.0,
    s0_e = None,
    seed = None
))]
pub fn bayescpi_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    df0_e: f64,
    s0_e: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, f64, f64, f64, f64)> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = s0_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_e must be > 0"));
        }
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };

    let result = py.detach(|| {
        bayescpi_packed_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            df0_e,
            s0_e,
            seed,
        )
    });

    match result {
        Ok((beta, alpha, varb_mean, vare, h2_mean, var_h2)) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            Ok((beta_py, alpha_py, varb_mean, vare, h2_mean, var_h2))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}
