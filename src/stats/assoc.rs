use crate::bedmath::{decode_plink_bed_hardcall, packed_row_missing_count_selected};
use crate::linalg::{chi2_sf_df1, chi2_stat_df1_from_sf, format_chisq_value};
use crate::math_farmcpu::{
    decode_dense_rows_to_sample_major, decode_packed_rows_to_sample_major,
    farmcpu_ll_score_from_sample_major, farmcpu_super_keep_from_sample_major, select_lead_indices,
};
use crate::stats_common::{get_cached_pool, parse_index_vec_i64, AsyncTsvWriter};
use nalgebra::DMatrix;
use numpy::PyArray1;
use numpy::PyArrayMethods;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

// FarmCPU helper kernels moved to `math/farmcpu.rs`.

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

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

#[inline]
fn student_t_p_two_sided(t: f64, df: i32) -> f64 {
    if df <= 0 {
        return f64::NAN;
    }
    if !t.is_finite() {
        return if t.is_nan() {
            f64::NAN
        } else {
            f64::MIN_POSITIVE
        };
    }
    let v = df as f64;
    let x = v / (v + t * t);
    let a = v / 2.0;
    let b = 0.5;
    let mut p = betai(a, b, x);
    if !p.is_finite() {
        p = 1.0;
    }
    p.clamp(f64::MIN_POSITIVE, 1.0)
}

#[inline]
fn lm_plrt_from_t2(t2: f64, n_obs: usize, df: i32) -> f64 {
    if df <= 0 || !t2.is_finite() || t2 < 0.0 {
        return f64::NAN;
    }
    let stat = (n_obs as f64) * (1.0 + t2 / (df as f64)).ln();
    chi2_sf_df1(stat)
}

#[inline]
fn xs_t_ixx_into(xs: &[f64], ixx: &[f64], q0: usize, out_b21: &mut [f64]) {
    for j in 0..q0 {
        let mut acc = 0.0;
        for k in 0..q0 {
            acc += xs[k] * ixx[k * q0 + j];
        }
        out_b21[j] = acc;
    }
}

#[inline]
fn build_ixxs_into(ixx: &[f64], b21: &[f64], invb22: f64, q0: usize, out_ixxs: &mut [f64]) {
    let dim = q0 + 1;
    for r in 0..q0 {
        for c in 0..q0 {
            out_ixxs[r * dim + c] = ixx[r * q0 + c] + invb22 * (b21[r] * b21[c]);
        }
    }
    out_ixxs[q0 * dim + q0] = invb22;
    for j in 0..q0 {
        let v = -invb22 * b21[j];
        out_ixxs[q0 * dim + j] = v;
        out_ixxs[j * dim + q0] = v;
    }
}

#[inline]
fn matvec_into(a: &[f64], dim: usize, rhs: &[f64], out: &mut [f64]) {
    for r in 0..dim {
        let row = &a[r * dim..(r + 1) * dim];
        out[r] = row.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum();
    }
}

struct GlmScratch {
    xs: Vec<f64>,
    b21: Vec<f64>,
    rhs: Vec<f64>,
    beta: Vec<f64>,
    ixxs: Vec<f64>,
}

impl GlmScratch {
    fn new(q0: usize) -> Self {
        let dim = q0 + 1;
        Self {
            xs: vec![0.0; q0],
            b21: vec![0.0; q0],
            rhs: vec![0.0; dim],
            beta: vec![0.0; dim],
            ixxs: vec![0.0; dim * dim],
        }
    }
    #[inline]
    fn reset_xs(&mut self) {
        self.xs.fill(0.0);
    }
}

fn pinv_xtx(xtx: &[f64], q: usize) -> Result<Vec<f64>, String> {
    if q == 0 {
        return Ok(Vec::new());
    }
    let a = DMatrix::<f64>::from_row_slice(q, q, xtx);
    let svd = a.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| "SVD failed to produce U for X'X pseudo-inverse".to_string())?;
    let vt = svd
        .v_t
        .ok_or_else(|| "SVD failed to produce V^T for X'X pseudo-inverse".to_string())?;
    let mut s_inv = DMatrix::<f64>::zeros(q, q);
    let smax = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let rcond = 1e-12_f64;
    let cutoff = rcond * smax.max(1.0);
    for i in 0..q {
        let s = svd.singular_values[i];
        if s.is_finite() && s > cutoff {
            s_inv[(i, i)] = 1.0 / s;
        }
    }
    let v = vt.transpose();
    let ixx = v * s_inv * u.transpose();
    Ok(ixx.as_slice().to_vec())
}

fn build_x_with_qtn_packed(
    base_x: &[f64],
    n: usize,
    q_base: usize,
    qtn_idx: &[usize],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
) -> Result<(Vec<f64>, usize), String> {
    if base_x.len() != n * q_base {
        return Err("base X shape mismatch".to_string());
    }
    let k = qtn_idx.len();
    let q_total = q_base + k;
    let mut out = vec![0.0_f64; n * q_total];
    for i in 0..n {
        let src = &base_x[i * q_base..(i + 1) * q_base];
        let dst = &mut out[i * q_total..i * q_total + q_base];
        dst.copy_from_slice(src);
    }
    if k > 0 {
        let decoded = decode_packed_rows_to_sample_major(
            packed_flat,
            bytes_per_snp,
            qtn_idx,
            row_indices,
            sample_idx,
            row_flip,
            row_maf,
        )?;
        for i in 0..n {
            let src = &decoded[i * k..(i + 1) * k];
            let dst = &mut out[i * q_total + q_base..(i + 1) * q_total];
            dst.copy_from_slice(src);
        }
    }
    Ok((out, q_total))
}

fn scan_packed_full_matrix(
    y: &[f64],
    x_flat: &[f64],   // row-major (n, q0)
    ixx_flat: &[f64], // row-major (q0, q0)
    packed_flat: &[u8],
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    threads: usize,
) -> Result<(Vec<f64>, usize, usize), String> {
    let n = y.len();
    if n == 0 {
        return Err("empty y".to_string());
    }
    if sample_idx.len() != n {
        return Err("sample_idx length mismatch".to_string());
    }
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let bytes_per_snp = (n_samples + 3) / 4;
    if bytes_per_snp == 0 {
        return Err("invalid packed shape".to_string());
    }
    if packed_flat.len() % bytes_per_snp != 0 {
        return Err("packed length is not divisible by bytes_per_snp".to_string());
    }
    let m_packed = packed_flat.len() / bytes_per_snp;
    let m = row_indices.map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m || row_maf.len() != m {
        return Err("row_flip/row_maf length mismatch".to_string());
    }
    let q0 = if n == 0 { 0 } else { x_flat.len() / n };
    if x_flat.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("ixx shape mismatch".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!("n too small for LM core: n={n}, q={q0}"));
    }

    let row_stride = q0 + 3;
    let dim = q0 + 1;
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let mut out = vec![f64::NAN; m * row_stride];
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let mut runner = || {
        out.par_chunks_mut(row_stride).enumerate().for_each_init(
            || GlmScratch::new(q0),
            |scr, (idx, row_out)| {
                scr.reset_xs();
                let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                let flip = row_flip[idx];
                let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);
                let mut sy = 0.0_f64;
                let mut ss = 0.0_f64;
                for (k, &sidx) in sample_idx.iter().enumerate() {
                    let b = row[sidx >> 2];
                    let code = (b >> ((sidx & 3) * 2)) & 0b11;
                    let mut gv = match decode_plink_bed_hardcall(code) {
                        Some(v) => v,
                        None => mean_g,
                    };
                    if flip && code != 0b01 {
                        gv = 2.0 - gv;
                    }
                    sy += gv * y[k];
                    ss += gv * gv;
                    let xrow = &x_flat[k * q0..(k + 1) * q0];
                    for j in 0..q0 {
                        scr.xs[j] += xrow[j] * gv;
                    }
                }
                xs_t_ixx_into(&scr.xs, ixx_flat, q0, &mut scr.b21);
                let t2 = dot(&scr.b21, &scr.xs);
                let b22 = ss - t2;
                let (invb22, df) = if b22 < 1e-8 {
                    (0.0, (n as i32) - (q0 as i32))
                } else {
                    (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                };
                if df <= 0 {
                    row_out.fill(f64::NAN);
                    return;
                }
                build_ixxs_into(ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);
                scr.rhs[..q0].copy_from_slice(&xy);
                scr.rhs[q0] = sy;
                matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);
                let beta_rhs = dot(&scr.beta, &scr.rhs);
                let ve = (yy - beta_rhs) / (df as f64);
                for ff in 0..dim {
                    let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
                    let t = scr.beta[ff] / se;
                    row_out[2 + ff] = student_t_p_two_sided(t, df);
                }
                if invb22 == 0.0 {
                    row_out[0] = f64::NAN;
                    row_out[1] = f64::NAN;
                    row_out[2 + q0] = f64::NAN;
                } else {
                    row_out[0] = scr.beta[q0];
                    row_out[1] = (scr.ixxs[q0 * dim + q0] * ve).sqrt();
                }
            },
        );
    };
    if let Some(p) = &pool {
        p.install(&mut runner);
    } else {
        runner();
    }
    Ok((out, m, row_stride))
}

fn fit_packed_full_rows(
    y: &[f64],
    x_flat: &[f64],   // row-major (n, q0)
    ixx_flat: &[f64], // row-major (q0, q0)
    packed_flat: &[u8],
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    selected_rows: &[usize],
) -> Result<(Vec<f64>, usize), String> {
    let n = y.len();
    if n == 0 {
        return Err("empty y".to_string());
    }
    if sample_idx.len() != n {
        return Err("sample_idx length mismatch".to_string());
    }
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let bytes_per_snp = (n_samples + 3) / 4;
    if bytes_per_snp == 0 {
        return Err("invalid packed shape".to_string());
    }
    if packed_flat.len() % bytes_per_snp != 0 {
        return Err("packed length is not divisible by bytes_per_snp".to_string());
    }
    let m_packed = packed_flat.len() / bytes_per_snp;
    let m = row_indices.map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m || row_maf.len() != m {
        return Err("row_flip/row_maf length mismatch".to_string());
    }
    let q0 = if n == 0 { 0 } else { x_flat.len() / n };
    if x_flat.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("ixx shape mismatch".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!("n too small for LM core: n={n}, q={q0}"));
    }

    let dim = q0 + 1;
    let row_stride = dim * 3;
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let mut out = vec![f64::NAN; selected_rows.len() * row_stride];
    let mut scr = GlmScratch::new(q0);
    for (row_idx, &idx) in selected_rows.iter().enumerate() {
        if idx >= m {
            return Err(format!(
                "selected row index out of bounds: idx={idx}, m={m}"
            ));
        }
        scr.reset_xs();
        let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
        if src_row >= m_packed {
            return Err(format!(
                "packed row index out of bounds: idx={src_row}, packed_rows={m_packed}"
            ));
        }
        let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
        let flip = row_flip[idx];
        let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);
        let mut sy = 0.0_f64;
        let mut ss = 0.0_f64;
        for (k, &sidx) in sample_idx.iter().enumerate() {
            let b = row[sidx >> 2];
            let code = (b >> ((sidx & 3) * 2)) & 0b11;
            let mut gv = match decode_plink_bed_hardcall(code) {
                Some(v) => v,
                None => mean_g,
            };
            if flip && code != 0b01 {
                gv = 2.0 - gv;
            }
            sy += gv * y[k];
            ss += gv * gv;
            let xrow = &x_flat[k * q0..(k + 1) * q0];
            for j in 0..q0 {
                scr.xs[j] += xrow[j] * gv;
            }
        }
        xs_t_ixx_into(&scr.xs, ixx_flat, q0, &mut scr.b21);
        let t2 = dot(&scr.b21, &scr.xs);
        let b22 = ss - t2;
        let (invb22, df) = if b22 < 1e-8 {
            (0.0, (n as i32) - (q0 as i32))
        } else {
            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
        };
        let row_out = &mut out[row_idx * row_stride..(row_idx + 1) * row_stride];
        if df <= 0 {
            row_out.fill(f64::NAN);
            continue;
        }
        build_ixxs_into(ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);
        scr.rhs[..q0].copy_from_slice(&xy);
        scr.rhs[q0] = sy;
        matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);
        let beta_rhs = dot(&scr.beta, &scr.rhs);
        let ve = (yy - beta_rhs) / (df as f64);
        for ff in 0..dim {
            let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
            let t = scr.beta[ff] / se;
            let base = 3 * ff;
            row_out[base] = scr.beta[ff];
            row_out[base + 1] = se;
            row_out[base + 2] = student_t_p_two_sided(t, df);
        }
        if invb22 == 0.0 {
            let base = 3 * q0;
            row_out[base] = f64::NAN;
            row_out[base + 1] = f64::NAN;
            row_out[base + 2] = f64::NAN;
        }
    }
    Ok((out, row_stride))
}

fn summarize_qtn_from_fem(
    fem: &[f64],
    m: usize,
    row_stride: usize,
    q_base: usize,
    qtn_idx: &[usize],
) -> (Vec<f64>, Vec<usize>) {
    let k_qtn = qtn_idx.len();
    let mut qtn_min_p = vec![1.0_f64; k_qtn];
    let mut qtn_best_rows = vec![0usize; k_qtn];
    for i in 0..m {
        let base = i * row_stride;
        for j in 0..k_qtn {
            let col = 2 + q_base + j;
            let p = fem[base + col];
            if p.is_finite() && p < qtn_min_p[j] {
                qtn_min_p[j] = p;
                qtn_best_rows[j] = i;
            }
        }
    }
    (qtn_min_p, qtn_best_rows)
}

#[allow(clippy::too_many_arguments)]
fn write_farmcpu_packed_main_scan(
    out_writer: &AsyncTsvWriter,
    qtn_writer: Option<&AsyncTsvWriter>,
    qtn_path_for_err: &str,
    y: &[f64],
    x_qtn: &[f64],
    ixx_flat: &[f64],
    packed_flat: &[u8],
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    threads: usize,
    chrom: &[String],
    pos: &[i64],
    snp: &[String],
    allele0: &[String],
    allele1: &[String],
    miss_counts: &[usize],
    _q_base: usize,
    qtn_lookup: &HashMap<usize, usize>,
    qtn_min_p: &[f64],
    qtn_beta: &[f64],
    qtn_se: &[f64],
) -> Result<(usize, usize), String> {
    let n = y.len();
    if n == 0 {
        return Err("empty phenotype vector".to_string());
    }
    if sample_idx.len() != n {
        return Err("sample_idx length mismatch".to_string());
    }
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let q0 = x_qtn.len() / n;
    if x_qtn.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("ixx shape mismatch".to_string());
    }
    let m = chrom.len();
    if pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
        || row_flip.len() != m
        || row_maf.len() != m
        || miss_counts.len() != m
    {
        return Err("FarmCPU metadata length mismatch during streaming scan".to_string());
    }

    let bytes_per_snp = (n_samples + 3) / 4;
    if bytes_per_snp == 0 {
        return Err("invalid packed shape".to_string());
    }
    if packed_flat.len() % bytes_per_snp != 0 {
        return Err("packed length is not divisible by bytes_per_snp".to_string());
    }
    let m_packed = packed_flat.len() / bytes_per_snp;
    if row_indices
        .map(|v| v.iter().any(|&idx| idx >= m_packed))
        .unwrap_or(false)
    {
        return Err("row_indices out of bounds for packed rows".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!("n too small for LM core: n={n}, q={q0}"));
    }

    let dim = q0 + 1;
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_qtn[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let df = (n as i32) - (q0 as i32) - 1;
    let can_plrt = df > 0;
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let row_block = 8192usize;
    let mut stats_buf = vec![f64::NAN; row_block * 3];
    let mut text_buf = String::with_capacity(row_block * 128);
    let mut qtn_text_buf_opt = if qtn_writer.is_some() {
        Some(String::with_capacity(
            row_block.min(qtn_lookup.len().max(1)) * 128,
        ))
    } else {
        None
    };
    let mut qtn_rows_written = 0usize;

    for row_start in (0..m).step_by(row_block) {
        let row_end = (row_start + row_block).min(m);
        let n_block = row_end - row_start;
        let stats_slice = &mut stats_buf[..n_block * 3];
        let mut runner = || {
            stats_slice.par_chunks_mut(3).enumerate().for_each_init(
                || GlmScratch::new(q0),
                |scr, (local_idx, stat_out)| {
                    let idx = row_start + local_idx;
                    scr.reset_xs();
                    let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                    let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                    let flip = row_flip[idx];
                    let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);
                    let mut sy = 0.0_f64;
                    let mut ss = 0.0_f64;
                    for (k, &sidx) in sample_idx.iter().enumerate() {
                        let b = row[sidx >> 2];
                        let code = (b >> ((sidx & 3) * 2)) & 0b11;
                        let mut gv = match decode_plink_bed_hardcall(code) {
                            Some(v) => v,
                            None => mean_g,
                        };
                        if flip && code != 0b01 {
                            gv = 2.0 - gv;
                        }
                        sy += gv * y[k];
                        ss += gv * gv;
                        let xrow = &x_qtn[k * q0..(k + 1) * q0];
                        for j in 0..q0 {
                            scr.xs[j] += xrow[j] * gv;
                        }
                    }
                    xs_t_ixx_into(&scr.xs, ixx_flat, q0, &mut scr.b21);
                    let t2 = dot(&scr.b21, &scr.xs);
                    let b22 = ss - t2;
                    let (invb22, df_row) = if b22 < 1e-8 {
                        (0.0, (n as i32) - (q0 as i32))
                    } else {
                        (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                    };
                    if df_row <= 0 || invb22 == 0.0 {
                        stat_out[0] = f64::NAN;
                        stat_out[1] = f64::NAN;
                        stat_out[2] = f64::NAN;
                        return;
                    }
                    build_ixxs_into(ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);
                    scr.rhs[..q0].copy_from_slice(&xy);
                    scr.rhs[q0] = sy;
                    matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);
                    let beta_rhs = dot(&scr.beta, &scr.rhs);
                    let ve = (yy - beta_rhs) / (df_row as f64);
                    let beta = scr.beta[q0];
                    let se = (scr.ixxs[q0 * dim + q0] * ve).sqrt();
                    let t = beta / se;
                    stat_out[0] = beta;
                    stat_out[1] = se;
                    stat_out[2] = student_t_p_two_sided(t, df_row);
                },
            );
        };
        if let Some(p) = &pool {
            p.install(&mut runner);
        } else {
            runner();
        }

        text_buf.clear();
        if let Some(ref mut qtn_text_buf) = qtn_text_buf_opt {
            qtn_text_buf.clear();
        }
        for local_idx in 0..n_block {
            let i = row_start + local_idx;
            let stat_base = local_idx * 3;
            let mut beta = stats_slice[stat_base];
            let mut se = stats_slice[stat_base + 1];
            let mut pwald = stats_slice[stat_base + 2];
            if !pwald.is_finite() {
                pwald = 1.0;
            }
            if let Some(&j) = qtn_lookup.get(&i) {
                beta = qtn_beta[j];
                se = qtn_se[j];
                if qtn_min_p[j].is_finite() {
                    pwald = qtn_min_p[j];
                }
            }
            let plrt = if can_plrt && beta.is_finite() && se.is_finite() && se > 0.0 {
                let t2 = (beta / se) * (beta / se);
                lm_plrt_from_t2(t2, n, df)
            } else {
                f64::NAN
            };
            let chisq = if plrt.is_finite() {
                chi2_stat_df1_from_sf(plrt)
            } else if beta.is_finite() && se.is_finite() && se > 0.0 {
                let z = beta / se;
                z * z
            } else {
                f64::NAN
            };
            let chisq_txt = format_chisq_value(chisq);
            let _ = write!(
                text_buf,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                chrom[i],
                pos[i],
                snp[i],
                allele0[i],
                allele1[i],
                row_maf[i],
                miss_counts[i],
                beta,
                se,
                chisq_txt,
                pwald,
                plrt
            );
            if let Some(ref mut qtn_text_buf) = qtn_text_buf_opt {
                if qtn_lookup.contains_key(&i) {
                    qtn_rows_written += 1;
                    let _ = write!(
                        qtn_text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                        chrom[i],
                        pos[i],
                        snp[i],
                        allele0[i],
                        allele1[i],
                        row_maf[i],
                        miss_counts[i],
                        beta,
                        se,
                        chisq_txt,
                        pwald,
                        plrt
                    );
                }
            }
        }

        if !text_buf.is_empty() {
            out_writer
                .send(text_buf.as_bytes().to_vec())
                .map_err(|e| e.to_string())?;
        }
        if let (Some(q_writer), Some(qtn_text_buf)) = (qtn_writer, qtn_text_buf_opt.as_mut()) {
            if !qtn_text_buf.is_empty() {
                q_writer
                    .send(qtn_text_buf.as_bytes().to_vec())
                    .map_err(|e| format!("{qtn_path_for_err}: {e}"))?;
            }
        }
    }

    Ok((m, qtn_rows_written))
}

#[pyfunction]
#[pyo3(signature = (
    sz,
    n_lead,
    pvalue,
    pos,
    packed,
    n_samples,
    row_flip,
    row_maf,
    y,
    x,
    sample_indices=None,
    delta_exp_start=-5.0,
    delta_exp_end=5.0,
    delta_step=0.1,
    svd_eps=1e-8
))]
pub fn farmcpu_rem_packed(
    py: Python<'_>,
    sz: i64,
    n_lead: usize,
    pvalue: PyReadonlyArray1<'_, f64>,
    pos: PyReadonlyArray1<'_, i64>,
    packed: PyReadonlyArray2<'_, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'_, bool>,
    row_maf: PyReadonlyArray1<'_, f32>,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    sample_indices: Option<PyReadonlyArray1<'_, i64>>,
    delta_exp_start: f64,
    delta_exp_end: f64,
    delta_step: f64,
    svd_eps: f64,
) -> PyResult<(f64, Vec<i64>)> {
    if sz <= 0 {
        return Err(PyRuntimeError::new_err("sz must be > 0"));
    }
    let pvalue = pvalue.as_slice()?;
    let pos = pos.as_slice()?;
    if pvalue.len() != pos.len() {
        return Err(PyRuntimeError::new_err(
            "pvalue and pos length mismatch in farmcpu_rem_packed",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    if m != pvalue.len() {
        return Err(PyRuntimeError::new_err(format!(
            "packed row count mismatch: packed m={m}, pvalue len={}",
            pvalue.len()
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let n = y.len();
    if x_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    let p = x_arr.shape()[1];
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    if sample_idx.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let leadidx = select_lead_indices(sz, n_lead, pvalue, pos);
    let k = leadidx.len();
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let (score, leadidx_i64) = py.detach(|| -> PyResult<(f64, Vec<i64>)> {
        let snp_pool_sample_major = decode_packed_rows_to_sample_major(
            &packed_flat,
            bytes_per_snp,
            &leadidx,
            None,
            &sample_idx,
            row_flip,
            row_maf,
        )
        .map_err(PyRuntimeError::new_err)?;

        let score = farmcpu_ll_score_from_sample_major(
            y,
            &x_flat,
            n,
            p,
            &snp_pool_sample_major,
            k,
            delta_exp_start,
            delta_exp_end,
            delta_step,
            svd_eps,
        )
        .map_err(PyRuntimeError::new_err)?;

        let leadidx_i64 = leadidx.iter().map(|&v| v as i64).collect::<Vec<_>>();
        Ok((score, leadidx_i64))
    })?;
    Ok((score, leadidx_i64))
}

#[pyfunction]
#[pyo3(signature = (
    row_indices,
    pvalue,
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    thr=0.7
))]
pub fn farmcpu_super_packed<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, i64>,
    pvalue: PyReadonlyArray1<'py, f64>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    thr: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ridx_raw = row_indices.as_slice()?;
    let pval = pvalue.as_slice()?;
    if ridx_raw.len() != pval.len() {
        return Err(PyRuntimeError::new_err(
            "row_indices and pvalue length mismatch in farmcpu_super_packed",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let ridx = parse_index_vec_i64(ridx_raw, m, "row_indices")?;
    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    let k = ridx.len();

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let keep = py.detach(|| -> PyResult<Vec<bool>> {
        let sample_major = decode_packed_rows_to_sample_major(
            &packed_flat,
            bytes_per_snp,
            &ridx,
            None,
            &sample_idx,
            row_flip,
            row_maf,
        )
        .map_err(PyRuntimeError::new_err)?;
        Ok(farmcpu_super_keep_from_sample_major(
            &sample_major,
            n,
            k,
            pval,
            thr,
        ))
    })?;

    let out = PyArray1::<bool>::zeros(py, [k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&keep);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    sz,
    n_lead,
    pvalue,
    pos,
    g,
    y,
    x,
    delta_exp_start=-5.0,
    delta_exp_end=5.0,
    delta_step=0.1,
    svd_eps=1e-8
))]
pub fn farmcpu_rem_dense(
    py: Python<'_>,
    sz: i64,
    n_lead: usize,
    pvalue: PyReadonlyArray1<'_, f64>,
    pos: PyReadonlyArray1<'_, i64>,
    g: PyReadonlyArray2<'_, f32>,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    delta_exp_start: f64,
    delta_exp_end: f64,
    delta_step: f64,
    svd_eps: f64,
) -> PyResult<(f64, Vec<i64>)> {
    if sz <= 0 {
        return Err(PyRuntimeError::new_err("sz must be > 0"));
    }
    let pvalue = pvalue.as_slice()?;
    let pos = pos.as_slice()?;
    if pvalue.len() != pos.len() {
        return Err(PyRuntimeError::new_err(
            "pvalue and pos length mismatch in farmcpu_rem_dense",
        ));
    }

    let g_arr = g.as_array();
    if g_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("g must be 2D (m, n)"));
    }
    let m = g_arr.shape()[0];
    let n = g_arr.shape()[1];
    if m != pvalue.len() {
        return Err(PyRuntimeError::new_err(format!(
            "g row count mismatch: g m={m}, pvalue len={}",
            pvalue.len()
        )));
    }

    let y = y.as_slice()?;
    if y.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "y length mismatch: got {}, expected n={n}",
            y.len()
        )));
    }

    let x_arr = x.as_array();
    if x_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    let p = x_arr.shape()[1];
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let g_slice_opt = g.as_slice().ok();

    let leadidx = select_lead_indices(sz, n_lead, pvalue, pos);
    let k = leadidx.len();

    let (score, leadidx_i64) = py.detach(|| -> PyResult<(f64, Vec<i64>)> {
        let snp_pool_sample_major =
            decode_dense_rows_to_sample_major(&g_arr, g_slice_opt, n, &leadidx);
        let score = farmcpu_ll_score_from_sample_major(
            y,
            &x_flat,
            n,
            p,
            &snp_pool_sample_major,
            k,
            delta_exp_start,
            delta_exp_end,
            delta_step,
            svd_eps,
        )
        .map_err(PyRuntimeError::new_err)?;
        let leadidx_i64 = leadidx.iter().map(|&v| v as i64).collect::<Vec<_>>();
        Ok((score, leadidx_i64))
    })?;
    Ok((score, leadidx_i64))
}

#[pyfunction]
#[pyo3(signature = (
    row_indices,
    pvalue,
    g,
    thr=0.7
))]
pub fn farmcpu_super_dense<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, i64>,
    pvalue: PyReadonlyArray1<'py, f64>,
    g: PyReadonlyArray2<'py, f32>,
    thr: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ridx_raw = row_indices.as_slice()?;
    let pval = pvalue.as_slice()?;
    if ridx_raw.len() != pval.len() {
        return Err(PyRuntimeError::new_err(
            "row_indices and pvalue length mismatch in farmcpu_super_dense",
        ));
    }

    let g_arr = g.as_array();
    if g_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("g must be 2D (m, n)"));
    }
    let m = g_arr.shape()[0];
    let n = g_arr.shape()[1];
    let ridx = parse_index_vec_i64(ridx_raw, m, "row_indices")?;
    let k = ridx.len();
    let g_slice_opt = g.as_slice().ok();

    let keep = py.detach(|| {
        let sample_major = decode_dense_rows_to_sample_major(&g_arr, g_slice_opt, n, &ridx);
        farmcpu_super_keep_from_sample_major(&sample_major, n, k, pval, thr)
    });

    let out = PyArray1::<bool>::zeros(py, [k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&keep);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    y,
    x_cov,
    chrom,
    pos,
    snp,
    allele0,
    allele1,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_missing,
    out_tsv,
    sample_indices=None,
    row_indices=None,
    threshold=0.05,
    max_iter=30,
    qtn_bound=None,
    nbin=5,
    szbin=vec![5e5, 5e6, 5e7],
    threads=0,
    progress_callback=None,
    pseudo_tsv=None
))]
pub fn farmcpu_packed_to_tsv(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    x_cov: PyReadonlyArray2<'_, f64>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    packed: PyReadonlyArray2<'_, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'_, bool>,
    row_maf: PyReadonlyArray1<'_, f32>,
    row_missing: PyReadonlyArray1<'_, f32>,
    out_tsv: &str,
    sample_indices: Option<PyReadonlyArray1<'_, i64>>,
    row_indices: Option<PyReadonlyArray1<'_, i64>>,
    threshold: f64,
    max_iter: usize,
    qtn_bound: Option<usize>,
    nbin: usize,
    szbin: Vec<f64>,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    pseudo_tsv: Option<&str>,
) -> PyResult<(usize, usize, usize)> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(threshold.is_finite() && threshold > 0.0) {
        return Err(PyRuntimeError::new_err("threshold must be finite and > 0"));
    }
    let y = y.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("empty phenotype vector"));
    }

    let x_cov_arr = x_cov.as_array();
    if x_cov_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("x_cov must be 2D (n, p_cov)"));
    }
    if x_cov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err(format!(
            "x_cov rows mismatch: got {}, expected n={n}",
            x_cov_arr.shape()[0]
        )));
    }
    let p_cov = x_cov_arr.shape()[1];
    let q_base = 1 + p_cov; // intercept + covariates

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    if sample_idx.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m_packed = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let row_idx: Option<Vec<usize>> = if let Some(ridx) = row_indices {
        Some(parse_index_vec_i64(
            ridx.as_slice()?,
            m_packed,
            "row_indices",
        )?)
    } else {
        None
    };
    let m = row_idx.as_ref().map(|v| v.len()).unwrap_or(m_packed);
    if m == 0 {
        return Err(PyRuntimeError::new_err("empty packed marker matrix"));
    }
    if chrom.len() != m
        || pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
    {
        return Err(PyRuntimeError::new_err(format!(
            "metadata length mismatch: m={m}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        )));
    }
    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m || row_missing.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row metadata length mismatch: row_flip={}, row_maf={}, row_missing={}, m={m}",
            row_flip.len(),
            row_maf.len(),
            row_missing.len(),
        )));
    }
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    if packed_flat.len() != m_packed * bytes_per_snp {
        return Err(PyRuntimeError::new_err("invalid packed flattened length"));
    }

    let x_cov_flat: Cow<[f64]> = match x_cov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_cov_arr.iter().copied().collect()),
    };
    let mut base_x = vec![0.0_f64; n * q_base];
    for i in 0..n {
        base_x[i * q_base] = 1.0;
        let src = &x_cov_flat[i * p_cov..(i + 1) * p_cov];
        let dst = &mut base_x[i * q_base + 1..(i + 1) * q_base];
        dst.copy_from_slice(src);
    }

    let mut uniq_chrom = chrom.clone();
    uniq_chrom.sort();
    uniq_chrom.dedup();
    let mut chr_block: HashMap<String, i64> = HashMap::with_capacity(uniq_chrom.len());
    for (i, c) in uniq_chrom.into_iter().enumerate() {
        chr_block.insert(c, i as i64);
    }
    let global_pos: Vec<i64> = (0..m)
        .map(|i| {
            let block = *chr_block.get(&chrom[i]).unwrap_or(&0_i64);
            pos[i].saturating_add(block.saturating_mul(1_000_000_000_000_i64))
        })
        .collect();

    let mut qtn_idx: Vec<usize> = Vec::new();
    let qtn_threshold_eff = threshold / (m as f64);
    let max_iter_i = max_iter.max(1);
    let qb = qtn_bound.unwrap_or_else(|| {
        let nf = n as f64;
        if nf <= 2.0 {
            1
        } else {
            let den = nf.log10();
            if !den.is_finite() || den <= 0.0 {
                1
            } else {
                ((nf / den).sqrt().floor() as usize).max(1)
            }
        }
    });
    let qb_eff = qb.max(1);
    let nbin_den = nbin.max(1);
    let nbin_step = (qb_eff / nbin_den).max(1);
    let mut nbin_vals: Vec<usize> = (nbin_step..=qb_eff).step_by(nbin_step).collect();
    if nbin_vals.is_empty() {
        nbin_vals.push(qb_eff);
    }
    let mut szbin_i64: Vec<i64> = szbin
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.round() as i64)
        .map(|v| v.max(1))
        .collect();
    if szbin_i64.is_empty() {
        szbin_i64 = vec![500_000_i64, 5_000_000_i64, 50_000_000_i64];
    }
    let pool = get_cached_pool(threads)?;

    py.detach(|| -> PyResult<()> {
        let mut final_model_cache: Option<(Vec<f64>, usize, Vec<f64>, Vec<f64>, Vec<usize>)> =
            None;
        for it_idx in 0..max_iter_i {
            let (x_qtn, q_total) = build_x_with_qtn_packed(
                &base_x,
                n,
                q_base,
                &qtn_idx,
                &packed_flat,
                bytes_per_snp,
                row_idx.as_deref(),
                &sample_idx,
                row_flip,
                row_maf,
            )
            .map_err(PyRuntimeError::new_err)?;

            let mut xtx = vec![0.0_f64; q_total * q_total];
            for i in 0..n {
                let row = &x_qtn[i * q_total..(i + 1) * q_total];
                for a in 0..q_total {
                    let va = row[a];
                    for b in 0..q_total {
                        xtx[a * q_total + b] += va * row[b];
                    }
                }
            }
            let ixx = pinv_xtx(&xtx, q_total).map_err(PyRuntimeError::new_err)?;
            let (mut fem, _m_scan, row_stride) = scan_packed_full_matrix(
                y,
                &x_qtn,
                &ixx,
                &packed_flat,
                n_samples,
                row_flip,
                row_maf,
                row_idx.as_deref(),
                &sample_idx,
                threads,
            )
            .map_err(PyRuntimeError::new_err)?;
            for i in 0..m {
                let base = i * row_stride;
                for c in 2..row_stride {
                    let v = fem[base + c];
                    if !v.is_finite() {
                        fem[base + c] = 1.0;
                    }
                }
            }

            let k_qtn = qtn_idx.len();
            let mut qtn_min_p = vec![1.0_f64; k_qtn];
            if k_qtn > 0 {
                for i in 0..m {
                    let base = i * row_stride;
                    for j in 0..k_qtn {
                        let col = 2 + q_base + j;
                        let p = fem[base + col];
                        if p.is_finite() && p < qtn_min_p[j] {
                            qtn_min_p[j] = p;
                        }
                    }
                }
            }

            let mut femp = vec![1.0_f64; m];
            for i in 0..m {
                femp[i] = fem[i * row_stride + row_stride - 1];
            }
            for (j, &idx) in qtn_idx.iter().enumerate() {
                if idx < m && qtn_min_p[j].is_finite() {
                    femp[idx] = qtn_min_p[j];
                }
            }

            let has_signal = femp
                .iter()
                .any(|&p| p.is_finite() && p <= qtn_threshold_eff);
            if !has_signal {
                let (final_qtn_min_p, final_qtn_best_rows) =
                    summarize_qtn_from_fem(&fem, m, row_stride, q_base, &qtn_idx);
                final_model_cache =
                    Some((x_qtn, q_total, ixx, final_qtn_min_p, final_qtn_best_rows));
                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (it_idx + 1, max_iter_i))?;
                        Ok(())
                    })?;
                }
                break;
            }

            let mut combine: Vec<(i64, usize)> = Vec::new();
            for &sz in szbin_i64.iter() {
                for &nn in nbin_vals.iter() {
                    combine.push((sz, nn));
                }
            }
            if combine.is_empty() {
                let (final_qtn_min_p, final_qtn_best_rows) =
                    summarize_qtn_from_fem(&fem, m, row_stride, q_base, &qtn_idx);
                final_model_cache =
                    Some((x_qtn, q_total, ixx, final_qtn_min_p, final_qtn_best_rows));
                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (it_idx + 1, max_iter_i))?;
                        Ok(())
                    })?;
                }
                break;
            }

            let rem_task = || {
                combine
                    .par_iter()
                    .map(|&(sz, nn)| {
                        let leadidx = select_lead_indices(sz, nn, &femp, &global_pos);
                        let sample_major = decode_packed_rows_to_sample_major(
                            &packed_flat,
                            bytes_per_snp,
                            &leadidx,
                            row_idx.as_deref(),
                            &sample_idx,
                            row_flip,
                            row_maf,
                        );
                        let sample_major = match sample_major {
                            Ok(v) => v,
                            Err(_) => return (f64::INFINITY, leadidx),
                        };
                        let score = farmcpu_ll_score_from_sample_major(
                            y,
                            &x_qtn,
                            n,
                            q_total,
                            &sample_major,
                            leadidx.len(),
                            -5.0,
                            5.0,
                            0.1,
                            1e-8,
                        )
                        .unwrap_or(f64::INFINITY);
                        (score, leadidx)
                    })
                    .collect::<Vec<(f64, Vec<usize>)>>()
            };
            let rem_res = if let Some(p) = &pool {
                p.install(rem_task)
            } else {
                rem_task()
            };
            if rem_res.is_empty() {
                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (it_idx + 1, max_iter_i))?;
                        Ok(())
                    })?;
                }
                break;
            }
            let mut best_i = 0usize;
            let mut best_score = rem_res[0].0;
            for (i, (s, _)) in rem_res.iter().enumerate().skip(1) {
                if s.total_cmp(&best_score).is_lt() {
                    best_score = *s;
                    best_i = i;
                }
            }
            let opt_lead = &rem_res[best_i].1;
            let mut qtn_union = qtn_idx.clone();
            qtn_union.extend(opt_lead.iter().copied());
            qtn_union.sort_unstable();
            qtn_union.dedup();

            if it_idx >= 1 && !qtn_union.is_empty() {
                let prev_set: HashSet<usize> = qtn_idx.iter().copied().collect();
                qtn_union.retain(|&idx| {
                    let p = femp[idx];
                    (p.is_finite() && p < qtn_threshold_eff) || prev_set.contains(&idx)
                });
            }

            let qtn_next: Vec<usize> = if qtn_union.is_empty() {
                Vec::new()
            } else {
                let sample_major = decode_packed_rows_to_sample_major(
                    &packed_flat,
                    bytes_per_snp,
                    &qtn_union,
                    row_idx.as_deref(),
                    &sample_idx,
                    row_flip,
                    row_maf,
                )
                .map_err(PyRuntimeError::new_err)?;
                let pvals: Vec<f64> = qtn_union.iter().map(|&ix| femp[ix]).collect();
                let keep = farmcpu_super_keep_from_sample_major(
                    &sample_major,
                    n,
                    qtn_union.len(),
                    &pvals,
                    0.7,
                );
                qtn_union
                    .iter()
                    .zip(keep.iter())
                    .filter_map(|(&ix, &k)| if k { Some(ix) } else { None })
                    .collect()
            };

            if let Some(cb) = progress_callback.as_ref() {
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (it_idx + 1, max_iter_i))?;
                    Ok(())
                })?;
            }

            if qtn_next == qtn_idx {
                let (final_qtn_min_p, final_qtn_best_rows) =
                    summarize_qtn_from_fem(&fem, m, row_stride, q_base, &qtn_idx);
                final_model_cache =
                    Some((x_qtn, q_total, ixx, final_qtn_min_p, final_qtn_best_rows));
                break;
            }
            qtn_idx = qtn_next;
        }

        let (x_qtn, _q_total, ixx, qtn_min_p, qtn_best_rows) =
            if let Some(cached) = final_model_cache.take() {
                cached
            } else {
                let (x_qtn, q_total) = build_x_with_qtn_packed(
                    &base_x,
                    n,
                    q_base,
                    &qtn_idx,
                    &packed_flat,
                    bytes_per_snp,
                    row_idx.as_deref(),
                    &sample_idx,
                    row_flip,
                    row_maf,
                )
                .map_err(PyRuntimeError::new_err)?;
                let mut xtx = vec![0.0_f64; q_total * q_total];
                for i in 0..n {
                    let row = &x_qtn[i * q_total..(i + 1) * q_total];
                    for a in 0..q_total {
                        let va = row[a];
                        for b in 0..q_total {
                            xtx[a * q_total + b] += va * row[b];
                        }
                    }
                }
                let ixx = pinv_xtx(&xtx, q_total).map_err(PyRuntimeError::new_err)?;
                let (mut fem, _m_scan, row_stride) = scan_packed_full_matrix(
                    y,
                    &x_qtn,
                    &ixx,
                    &packed_flat,
                    n_samples,
                    row_flip,
                    row_maf,
                    row_idx.as_deref(),
                    &sample_idx,
                    threads,
                )
                .map_err(PyRuntimeError::new_err)?;
                for i in 0..m {
                    let base = i * row_stride;
                    for c in 2..row_stride {
                        let v = fem[base + c];
                        if !v.is_finite() {
                            fem[base + c] = 1.0;
                        }
                    }
                }
                let (qtn_min_p, qtn_best_rows) =
                    summarize_qtn_from_fem(&fem, m, row_stride, q_base, &qtn_idx);
                (x_qtn, q_total, ixx, qtn_min_p, qtn_best_rows)
            };
        let k_qtn = qtn_idx.len();
        let mut qtn_beta = vec![f64::NAN; k_qtn];
        let mut qtn_se = vec![f64::NAN; k_qtn];
        if k_qtn > 0 {
            let (qtn_full, qtn_full_stride) = fit_packed_full_rows(
                y,
                &x_qtn,
                &ixx,
                &packed_flat,
                n_samples,
                row_flip,
                row_maf,
                row_idx.as_deref(),
                &sample_idx,
                &qtn_best_rows,
            )
            .map_err(PyRuntimeError::new_err)?;
            for j in 0..k_qtn {
                let coef_idx = q_base + j;
                let base = j * qtn_full_stride + 3 * coef_idx;
                qtn_beta[j] = qtn_full[base];
                qtn_se[j] = qtn_full[base + 1];
            }
        }
        let mut qtn_lookup: HashMap<usize, usize> = HashMap::with_capacity(k_qtn);
        for (j, &idx) in qtn_idx.iter().enumerate() {
            qtn_lookup.insert(idx, j);
        }

        let mut miss_counts = vec![0usize; m];
        let mut fill_miss = || {
            miss_counts
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, dst)| {
                    let src_row = row_idx.as_ref().map(|v| v[idx]).unwrap_or(idx);
                    let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                    *dst = packed_row_missing_count_selected(row, n_samples, &sample_idx);
                });
        };
        if let Some(tp) = &pool {
            tp.install(fill_miss);
        } else {
            fill_miss();
        }

        let out_path = out_tsv.to_string();
        let writer = AsyncTsvWriter::with_config(
            &out_path,
            b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
            64 * 1024 * 1024,
            4,
        )
        .map_err(PyRuntimeError::new_err)?;

        let mut qtn_writer_opt: Option<AsyncTsvWriter> = None;
        let mut qtn_path_for_err = String::new();
        if let Some(pseudo_path) = pseudo_tsv {
            if !qtn_idx.is_empty() {
                qtn_path_for_err = pseudo_path.to_string();
                qtn_writer_opt = Some(
                    AsyncTsvWriter::with_config(
                        &qtn_path_for_err,
                        b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
                        16 * 1024 * 1024,
                        4,
                    )
                    .map_err(PyRuntimeError::new_err)?,
                );
            }
        }

        let (_written_rows, qtn_rows_written) = write_farmcpu_packed_main_scan(
            &writer,
            qtn_writer_opt.as_ref(),
            &qtn_path_for_err,
            y,
            &x_qtn,
            &ixx,
            &packed_flat,
            n_samples,
            row_flip,
            row_maf,
            row_idx.as_deref(),
            &sample_idx,
            threads,
            &chrom,
            &pos,
            &snp,
            &allele0,
            &allele1,
            &miss_counts,
            q_base,
            &qtn_lookup,
            &qtn_min_p,
            &qtn_beta,
            &qtn_se,
        )
        .map_err(PyRuntimeError::new_err)?;
        writer.finish().map_err(PyRuntimeError::new_err)?;
        if let Some(q_writer) = qtn_writer_opt.take() {
            q_writer.finish().map_err(|e| {
                PyRuntimeError::new_err(format!("{qtn_path_for_err}: {e}"))
            })?;
        }
        let _ = qtn_rows_written;
        Ok(())
    })?;

    Ok((m, qtn_idx.len(), m))
}

#[pyfunction]
#[pyo3(signature = (
    chrom,
    pos,
    snp,
    allele0,
    allele1,
    maf,
    row_missing,
    beta,
    se,
    pwald,
    n_obs,
    df,
    out_tsv,
    qtn_idx=None,
    pseudo_tsv=None
))]
pub fn farmcpu_write_assoc_tsv(
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    maf: PyReadonlyArray1<'_, f32>,
    row_missing: PyReadonlyArray1<'_, f32>,
    beta: PyReadonlyArray1<'_, f64>,
    se: PyReadonlyArray1<'_, f64>,
    pwald: PyReadonlyArray1<'_, f64>,
    n_obs: usize,
    df: usize,
    out_tsv: &str,
    qtn_idx: Option<PyReadonlyArray1<'_, i64>>,
    pseudo_tsv: Option<&str>,
) -> PyResult<(usize, usize)> {
    let m = chrom.len();
    if pos.len() != m || snp.len() != m || allele0.len() != m || allele1.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "FarmCPU metadata length mismatch: chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        )));
    }
    let maf = maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;
    let beta = beta.as_slice()?;
    let se = se.as_slice()?;
    let pwald = pwald.as_slice()?;
    if maf.len() != m
        || row_missing.len() != m
        || beta.len() != m
        || se.len() != m
        || pwald.len() != m
    {
        return Err(PyRuntimeError::new_err(format!(
            "FarmCPU vector length mismatch: m={m}, maf={}, miss={}, beta={}, se={}, pwald={}",
            maf.len(),
            row_missing.len(),
            beta.len(),
            se.len(),
            pwald.len()
        )));
    }

    let n_f = n_obs as f64;
    let df_f = df as f64;
    let can_plrt = n_obs > 0 && df > 0;

    let writer = AsyncTsvWriter::with_config(
        out_tsv,
        b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
        4 * 1024 * 1024,
        4,
    )
    .map_err(PyRuntimeError::new_err)?;
    let mut text_buf = String::with_capacity(8192 * 112);

    for i in 0..m {
        let b = beta[i];
        let s = se[i];
        let plrt = if can_plrt && b.is_finite() && s.is_finite() && s > 0.0 {
            let t2 = (b / s) * (b / s);
            let stat = n_f * (1.0 + t2 / df_f).ln();
            chi2_sf_df1(stat)
        } else {
            f64::NAN
        };
        let chisq = if plrt.is_finite() {
            chi2_stat_df1_from_sf(plrt)
        } else if b.is_finite() && s.is_finite() && s > 0.0 {
            let z = b / s;
            z * z
        } else {
            f64::NAN
        };
        let chisq_txt = format_chisq_value(chisq);
        writeln!(
            text_buf,
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}",
            chrom[i],
            pos[i],
            snp[i],
            allele0[i],
            allele1[i],
            maf[i],
            row_missing[i].round() as i64,
            b,
            s,
            chisq_txt,
            pwald[i],
            plrt
        )
        .map_err(|e| PyRuntimeError::new_err(format!("format row {i} for {out_tsv}: {e}")))?;
        if (i + 1) % 8192 == 0 {
            writer
                .send(text_buf.as_bytes().to_vec())
                .map_err(PyRuntimeError::new_err)?;
            text_buf.clear();
        }
    }
    if !text_buf.is_empty() {
        writer
            .send(text_buf.as_bytes().to_vec())
            .map_err(PyRuntimeError::new_err)?;
    }
    writer.finish().map_err(PyRuntimeError::new_err)?;

    let mut qtn_written = 0usize;
    if let (Some(qidx_arr), Some(pseudo_path)) = (qtn_idx, pseudo_tsv) {
        let qidx_raw = qidx_arr.as_slice()?;
        let mut uniq: Vec<usize> = Vec::with_capacity(qidx_raw.len());
        let mut seen: HashSet<usize> = HashSet::with_capacity(qidx_raw.len());
        for &v in qidx_raw.iter() {
            if v < 0 {
                continue;
            }
            let idx = v as usize;
            if idx >= m {
                continue;
            }
            if seen.insert(idx) {
                uniq.push(idx);
            }
        }
        if !uniq.is_empty() {
            let q_writer = AsyncTsvWriter::with_config(
                pseudo_path,
                b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
                512 * 1024,
                4,
            )
            .map_err(PyRuntimeError::new_err)?;
            let mut q_text_buf = String::with_capacity(uniq.len().min(8192).max(1) * 112);
            for &idx in uniq.iter() {
                let b = beta[idx];
                let s = se[idx];
                let plrt = if can_plrt && b.is_finite() && s.is_finite() && s > 0.0 {
                    let t2 = (b / s) * (b / s);
                    let stat = n_f * (1.0 + t2 / df_f).ln();
                    chi2_sf_df1(stat)
                } else {
                    f64::NAN
                };
                let chisq = if plrt.is_finite() {
                    chi2_stat_df1_from_sf(plrt)
                } else if b.is_finite() && s.is_finite() && s > 0.0 {
                    let z = b / s;
                    z * z
                } else {
                    f64::NAN
                };
                let chisq_txt = format_chisq_value(chisq);
                writeln!(
                    q_text_buf,
                    "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}",
                    chrom[idx],
                    pos[idx],
                    snp[idx],
                    allele0[idx],
                    allele1[idx],
                    maf[idx],
                    row_missing[idx].round() as i64,
                    b,
                    s,
                    chisq_txt,
                    pwald[idx],
                    plrt
                )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("format qtn row {idx} for {pseudo_path}: {e}"))
                })?;
                if q_text_buf.len() >= 512 * 1024 {
                    q_writer
                        .send(q_text_buf.as_bytes().to_vec())
                        .map_err(PyRuntimeError::new_err)?;
                    q_text_buf.clear();
                }
            }
            if !q_text_buf.is_empty() {
                q_writer
                    .send(q_text_buf.as_bytes().to_vec())
                    .map_err(PyRuntimeError::new_err)?;
            }
            q_writer.finish().map_err(PyRuntimeError::new_err)?;
            qtn_written = uniq.len();
        }
    }

    Ok((m, qtn_written))
}

// ---------------------------------------------------------------------------
// Dense genotype helpers for FarmCPU
// ---------------------------------------------------------------------------

fn scan_dense_full_matrix(
    y: &[f64],
    x_flat: &[f64],   // row-major (n, q0)
    ixx_flat: &[f64], // row-major (q0, q0)
    g: &[f32],        // row-major (m, n) — SNP-major dense genotype
    m: usize,
    n: usize,
    threads: usize,
) -> Result<(Vec<f64>, usize, usize), String> {
    if n == 0 || m == 0 {
        return Err("empty y or g".to_string());
    }
    if g.len() != m * n {
        return Err("g shape mismatch".to_string());
    }
    let q0 = if n == 0 { 0 } else { x_flat.len() / n };
    if x_flat.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("ixx shape mismatch".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!("n too small for LM core: n={n}, q={q0}"));
    }
    let row_stride = q0 + 3;
    let dim = q0 + 1;
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let mut out = vec![f64::NAN; m * row_stride];
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let mut runner = || {
        out.par_chunks_mut(row_stride).enumerate().for_each_init(
            || GlmScratch::new(q0),
            |scr, (snp_idx, row_out)| {
                scr.reset_xs();
                let g_row = &g[snp_idx * n..(snp_idx + 1) * n];
                let mut sy = 0.0_f64;
                let mut ss = 0.0_f64;
                for (k, &gv) in g_row.iter().enumerate() {
                    let gv = gv as f64;
                    sy += gv * y[k];
                    ss += gv * gv;
                    let xrow = &x_flat[k * q0..(k + 1) * q0];
                    for j in 0..q0 {
                        scr.xs[j] += xrow[j] * gv;
                    }
                }
                xs_t_ixx_into(&scr.xs, ixx_flat, q0, &mut scr.b21);
                let t2 = dot(&scr.b21, &scr.xs);
                let b22 = ss - t2;
                let (invb22, df) = if b22 < 1e-8 {
                    (0.0, (n as i32) - (q0 as i32))
                } else {
                    (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                };
                if df <= 0 {
                    row_out.fill(f64::NAN);
                    return;
                }
                build_ixxs_into(ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);
                scr.rhs[..q0].copy_from_slice(&xy);
                scr.rhs[q0] = sy;
                matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);
                let beta_rhs = dot(&scr.beta, &scr.rhs);
                let ve = (yy - beta_rhs) / (df as f64);
                for ff in 0..dim {
                    let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
                    let t = scr.beta[ff] / se;
                    row_out[2 + ff] = student_t_p_two_sided(t, df);
                }
                if invb22 == 0.0 {
                    row_out[0] = f64::NAN;
                    row_out[1] = f64::NAN;
                    row_out[2 + q0] = f64::NAN;
                } else {
                    row_out[0] = scr.beta[q0];
                    row_out[1] = (scr.ixxs[q0 * dim + q0] * ve).sqrt();
                }
            },
        );
    };
    if let Some(p) = &pool {
        p.install(&mut runner);
    } else {
        runner();
    }
    Ok((out, m, row_stride))
}

fn fit_dense_full_rows(
    y: &[f64],
    x_flat: &[f64],
    ixx_flat: &[f64],
    g: &[f32],         // row-major (m, n)
    m: usize,
    n: usize,
    selected_rows: &[usize],
) -> Result<(Vec<f64>, usize), String> {
    if n == 0 || m == 0 {
        return Err("empty y or g".to_string());
    }
    if g.len() != m * n {
        return Err("g shape mismatch".to_string());
    }
    let q0 = if n == 0 { 0 } else { x_flat.len() / n };
    if x_flat.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("ixx shape mismatch".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!("n too small for LM core: n={n}, q={q0}"));
    }
    let dim = q0 + 1;
    let row_stride = dim * 3;
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let mut out = vec![f64::NAN; selected_rows.len() * row_stride];
    let mut scr = GlmScratch::new(q0);
    for (out_idx, &snp_idx) in selected_rows.iter().enumerate() {
        if snp_idx >= m {
            return Err(format!("selected row index out of bounds: idx={snp_idx}, m={m}"));
        }
        scr.reset_xs();
        let g_row = &g[snp_idx * n..(snp_idx + 1) * n];
        let mut sy = 0.0_f64;
        let mut ss = 0.0_f64;
        for (k, &gv) in g_row.iter().enumerate() {
            let gv = gv as f64;
            sy += gv * y[k];
            ss += gv * gv;
            let xrow = &x_flat[k * q0..(k + 1) * q0];
            for j in 0..q0 {
                scr.xs[j] += xrow[j] * gv;
            }
        }
        xs_t_ixx_into(&scr.xs, ixx_flat, q0, &mut scr.b21);
        let t2 = dot(&scr.b21, &scr.xs);
        let b22 = ss - t2;
        let (invb22, df) = if b22 < 1e-8 {
            (0.0, (n as i32) - (q0 as i32))
        } else {
            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
        };
        let row_out = &mut out[out_idx * row_stride..(out_idx + 1) * row_stride];
        if df <= 0 {
            row_out.fill(f64::NAN);
            continue;
        }
        build_ixxs_into(ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);
        scr.rhs[..q0].copy_from_slice(&xy);
        scr.rhs[q0] = sy;
        matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);
        let beta_rhs = dot(&scr.beta, &scr.rhs);
        let ve = (yy - beta_rhs) / (df as f64);
        for ff in 0..dim {
            let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
            let t = scr.beta[ff] / se;
            let base = 3 * ff;
            row_out[base] = scr.beta[ff];
            row_out[base + 1] = se;
            row_out[base + 2] = student_t_p_two_sided(t, df);
        }
        if invb22 == 0.0 {
            let base = 3 * q0;
            row_out[base] = f64::NAN;
            row_out[base + 1] = f64::NAN;
            row_out[base + 2] = f64::NAN;
        }
    }
    Ok((out, row_stride))
}

fn build_x_with_qtn_dense(
    base_x: &[f64],
    n: usize,
    q_base: usize,
    qtn_idx: &[usize],
    g: &[f32],
    m_g: usize,
    n_g: usize,
) -> Result<(Vec<f64>, usize), String> {
    if n_g != n {
        return Err("g.n != n".to_string());
    }
    if base_x.len() != n * q_base {
        return Err("base X shape mismatch".to_string());
    }
    let k = qtn_idx.len();
    let q_total = q_base + k;
    let mut out = vec![0.0_f64; n * q_total];
    for i in 0..n {
        let src = &base_x[i * q_base..(i + 1) * q_base];
        let dst = &mut out[i * q_total..i * q_total + q_base];
        dst.copy_from_slice(src);
    }
    for (col, &snp_idx) in qtn_idx.iter().enumerate() {
        if snp_idx >= m_g {
            return Err(format!("qtn idx out of bounds: idx={snp_idx}, m={m_g}"));
        }
        let g_row = &g[snp_idx * n..(snp_idx + 1) * n];
        for i in 0..n {
            out[i * q_total + q_base + col] = g_row[i] as f64;
        }
    }
    Ok((out, q_total))
}

// =============================================================================
// farmcpu_dense — full FarmCPU loop for dense genotype, returns numpy arrays
// =============================================================================

#[pyfunction]
#[pyo3(signature = (
    y,
    x_cov,
    g,
    chrom,
    pos,
    threshold=0.05,
    max_iter=30,
    qtn_bound=None,
    nbin=5,
    szbin=vec![5e5, 5e6, 5e7],
    threads=0,
    progress_callback=None
))]
#[allow(clippy::too_many_arguments)]
pub fn farmcpu_dense<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray2<'py, f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    threshold: f64,
    max_iter: usize,
    qtn_bound: Option<usize>,
    nbin: usize,
    szbin: Vec<f64>,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>, // beta
    Bound<'py, PyArray1<f64>>, // se
    Bound<'py, PyArray1<f64>>, // pwald
    Bound<'py, PyArray1<i64>>, // qtn_idx
    usize,                      // n_pseudo_qtn
    usize,                      // n_obs
    usize,                      // df_lrt
)> {
    let y_slice = y.as_slice()?;
    let n = y_slice.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("empty phenotype vector"));
    }
    let x_cov_arr = x_cov.as_array();
    if x_cov_arr.ndim() != 2 || x_cov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("x_cov must be 2D (n, p_cov)"));
    }
    let p_cov = x_cov_arr.shape()[1];
    let q_base = 1 + p_cov;

    let g_arr = g.as_array();
    if g_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("g must be 2D (m, n)"));
    }
    let m = g_arr.shape()[0];
    let n_g = g_arr.shape()[1];
    if n_g != n {
        return Err(PyRuntimeError::new_err(format!(
            "g sample count mismatch: g.n={n_g}, len(y)={n}"
        )));
    }
    if m == 0 {
        return Err(PyRuntimeError::new_err("empty genotype matrix"));
    }
    if chrom.len() != m || pos.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "metadata length mismatch: m={m}, chrom={}, pos={}",
            chrom.len(),
            pos.len()
        )));
    }
    let n_obs = n;
    let df_lrt = n.saturating_sub(q_base);
    if !(threshold.is_finite() && threshold > 0.0) {
        return Err(PyRuntimeError::new_err("threshold must be finite and > 0"));
    }

    // Build base X with intercept
    let x_cov_flat: Cow<[f64]> = match x_cov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_cov_arr.iter().copied().collect()),
    };
    let mut base_x = vec![0.0_f64; n * q_base];
    for i in 0..n {
        base_x[i * q_base] = 1.0;
        let src = &x_cov_flat[i * p_cov..(i + 1) * p_cov];
        let dst = &mut base_x[i * q_base + 1..(i + 1) * q_base];
        dst.copy_from_slice(src);
    }

    // Map chrom to blocks for global positions
    let mut uniq_chrom: Vec<String> = chrom.iter().cloned().collect();
    uniq_chrom.sort();
    uniq_chrom.dedup();
    let chr_block: HashMap<String, i64> = uniq_chrom
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), i as i64))
        .collect();
    let global_pos: Vec<i64> = (0..m)
        .map(|i| {
            let block = *chr_block.get(&chrom[i]).unwrap_or(&0_i64);
            pos[i].saturating_add(block.saturating_mul(1_000_000_000_000_i64))
        })
        .collect();

    let qtn_threshold_eff = threshold / (m as f64);
    let max_iter_i = max_iter.max(1);
    let qb = qtn_bound.unwrap_or_else(|| {
        let nf = n as f64;
        if nf <= 2.0 { 1 } else {
            let den = nf.log10();
            if !den.is_finite() || den <= 0.0 { 1 } else {
                ((nf / den).sqrt().floor() as usize).max(1)
            }
        }
    });
    let qb_eff = qb.max(1);
    let nbin_den = nbin.max(1);
    let nbin_step = (qb_eff / nbin_den).max(1);
    let mut nbin_vals: Vec<usize> = (nbin_step..=qb_eff).step_by(nbin_step).collect();
    if nbin_vals.is_empty() {
        nbin_vals.push(qb_eff);
    }
    let mut szbin_i64: Vec<i64> = szbin
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.round() as i64)
        .map(|v| v.max(1))
        .collect();
    if szbin_i64.is_empty() {
        szbin_i64 = vec![500_000_i64, 5_000_000_i64, 50_000_000_i64];
    }

    // Flatten g to slice
    let g_flat: Cow<[f32]> = match g.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(g_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;

    py.detach(|| -> PyResult<(
        Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>, usize,
    )> {
        let mut qtn_idx: Vec<usize> = Vec::new();
        let mut final_model_cache: Option<(
            Vec<f64>,     // x_qtn
            usize,        // q_total
            Vec<f64>,     // ixx
            Vec<f64>,     // qtn_min_p
            Vec<usize>,   // qtn_best_rows
        )> = None;

        for it_idx in 0..max_iter_i {
            let (x_qtn, q_total) = build_x_with_qtn_dense(
                &base_x, n, q_base, &qtn_idx, &g_flat, m, n,
            ).map_err(PyRuntimeError::new_err)?;

            let mut xtx = vec![0.0_f64; q_total * q_total];
            for i in 0..n {
                let row = &x_qtn[i * q_total..(i + 1) * q_total];
                for a in 0..q_total {
                    let va = row[a];
                    for b in 0..q_total {
                        xtx[a * q_total + b] += va * row[b];
                    }
                }
            }
            let ixx = pinv_xtx(&xtx, q_total).map_err(PyRuntimeError::new_err)?;
            let (mut fem, _m_scan, row_stride) = scan_dense_full_matrix(
                y_slice, &x_qtn, &ixx, &g_flat, m, n, threads,
            ).map_err(PyRuntimeError::new_err)?;
            for i in 0..m {
                let base = i * row_stride;
                for c in 2..row_stride {
                    let v = fem[base + c];
                    if !v.is_finite() { fem[base + c] = 1.0; }
                }
            }

            let k_qtn = qtn_idx.len();
            let mut qtn_min_p = vec![1.0_f64; k_qtn];
            if k_qtn > 0 {
                for i in 0..m {
                    let base = i * row_stride;
                    for j in 0..k_qtn {
                        let col = 2 + q_base + j;
                        let p = fem[base + col];
                        if p.is_finite() && p < qtn_min_p[j] {
                            qtn_min_p[j] = p;
                        }
                    }
                }
            }

            let mut femp = vec![1.0_f64; m];
            for i in 0..m {
                femp[i] = fem[i * row_stride + row_stride - 1];
            }
            for (j, &idx) in qtn_idx.iter().enumerate() {
                if idx < m && qtn_min_p[j].is_finite() {
                    femp[idx] = qtn_min_p[j];
                }
            }

            let has_signal = femp.iter().any(|&p| p.is_finite() && p <= qtn_threshold_eff);
            if !has_signal {
                let (final_qtn_min_p, final_qtn_best_rows) =
                    summarize_qtn_from_fem(&fem, m, row_stride, q_base, &qtn_idx);
                final_model_cache = Some((x_qtn, q_total, ixx, final_qtn_min_p, final_qtn_best_rows));
                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (it_idx + 1, max_iter_i))?;
                        Ok(())
                    })?;
                }
                break;
            }

            let mut combine: Vec<(i64, usize)> = Vec::new();
            for &sz in szbin_i64.iter() {
                for &nn in nbin_vals.iter() {
                    combine.push((sz, nn));
                }
            }
            if combine.is_empty() {
                let (final_qtn_min_p, final_qtn_best_rows) =
                    summarize_qtn_from_fem(&fem, m, row_stride, q_base, &qtn_idx);
                final_model_cache = Some((x_qtn, q_total, ixx, final_qtn_min_p, final_qtn_best_rows));
                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (it_idx + 1, max_iter_i))?;
                        Ok(())
                    })?;
                }
                break;
            }

            let g_flat_rem = g_flat.as_ref();
            let x_qtn_rem = x_qtn.as_slice();
            let femp_rem = femp.as_slice();
            let global_pos_rem = global_pos.as_slice();
            let rem_task = || {
                combine
                    .par_iter()
                    .map(|&(sz, nn)| {
                        let leadidx = select_lead_indices(sz, nn, femp_rem, global_pos_rem);
                        let k = leadidx.len();
                        let mut sample_major = vec![0.0_f64; n * k];
                        for (col, &row_idx) in leadidx.iter().enumerate() {
                            let row = &g_flat_rem[row_idx * n..(row_idx + 1) * n];
                            for i in 0..n {
                                sample_major[i * k + col] = row[i] as f64;
                            }
                        }
                        let score = farmcpu_ll_score_from_sample_major(
                            y_slice, x_qtn_rem, n, q_total,
                            &sample_major, k,
                            -5.0, 5.0, 0.1, 1e-8,
                        ).unwrap_or(f64::INFINITY);
                        (score, leadidx)
                    })
                    .collect::<Vec<(f64, Vec<usize>)>>()
            };
            let rem_res = if let Some(p) = &pool {
                p.install(rem_task)
            } else {
                rem_task()
            };
            if rem_res.is_empty() {
                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (it_idx + 1, max_iter_i))?;
                        Ok(())
                    })?;
                }
                break;
            }
            let mut best_i = 0usize;
            let mut best_score = rem_res[0].0;
            for (i, (s, _)) in rem_res.iter().enumerate().skip(1) {
                if s.total_cmp(&best_score).is_lt() {
                    best_score = *s;
                    best_i = i;
                }
            }
            let opt_lead = &rem_res[best_i].1;
            let mut qtn_union = qtn_idx.clone();
            qtn_union.extend(opt_lead.iter().copied());
            qtn_union.sort_unstable();
            qtn_union.dedup();

            if it_idx >= 1 && !qtn_union.is_empty() {
                let prev_set: HashSet<usize> = qtn_idx.iter().copied().collect();
                qtn_union.retain(|&idx| {
                    let p = femp[idx];
                    (p.is_finite() && p < qtn_threshold_eff) || prev_set.contains(&idx)
                });
            }

            let qtn_next: Vec<usize> = if qtn_union.is_empty() {
                Vec::new()
            } else {
                let k_u = qtn_union.len();
                let mut sample_major = vec![0.0_f64; n * k_u];
                for (col, &row_idx) in qtn_union.iter().enumerate() {
                    let row = &g_flat[row_idx * n..(row_idx + 1) * n];
                    for i in 0..n {
                        sample_major[i * k_u + col] = row[i] as f64;
                    }
                }
                let pvals: Vec<f64> = qtn_union.iter().map(|&ix| femp[ix]).collect();
                let keep = farmcpu_super_keep_from_sample_major(
                    &sample_major, n, k_u, &pvals, 0.7,
                );
                qtn_union.iter().zip(keep.iter())
                    .filter_map(|(&ix, &k)| if k { Some(ix) } else { None })
                    .collect()
            };

            if let Some(cb) = progress_callback.as_ref() {
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (it_idx + 1, max_iter_i))?;
                    Ok(())
                })?;
            }

            if qtn_next == qtn_idx {
                let (final_qtn_min_p, final_qtn_best_rows) =
                    summarize_qtn_from_fem(&fem, m, row_stride, q_base, &qtn_idx);
                final_model_cache = Some((x_qtn, q_total, ixx, final_qtn_min_p, final_qtn_best_rows));
                break;
            }
            qtn_idx = qtn_next;
        }

        // Final scan
        let (x_qtn, _q_total, ixx, qtn_min_p, qtn_best_rows) =
            if let Some(cached) = final_model_cache.take() {
                cached
            } else {
                let (x_qtn_f, q_total_f) = build_x_with_qtn_dense(
                    &base_x, n, q_base, &qtn_idx, &g_flat, m, n,
                ).map_err(PyRuntimeError::new_err)?;
                let mut xtx = vec![0.0_f64; q_total_f * q_total_f];
                for i in 0..n {
                    let row = &x_qtn_f[i * q_total_f..(i + 1) * q_total_f];
                    for a in 0..q_total_f {
                        let va = row[a];
                        for b in 0..q_total_f {
                            xtx[a * q_total_f + b] += va * row[b];
                        }
                    }
                }
                let ixx_f = pinv_xtx(&xtx, q_total_f).map_err(PyRuntimeError::new_err)?;
                let (mut fem_f, _m_scan_f, row_stride_f) = scan_dense_full_matrix(
                    y_slice, &x_qtn_f, &ixx_f, &g_flat, m, n, threads,
                ).map_err(PyRuntimeError::new_err)?;
                for i in 0..m {
                    let base = i * row_stride_f;
                    for c in 2..row_stride_f {
                        let v = fem_f[base + c];
                        if !v.is_finite() { fem_f[base + c] = 1.0; }
                    }
                }
                let (qtn_min_p_f, qtn_best_rows_f) =
                    summarize_qtn_from_fem(&fem_f, m, row_stride_f, q_base, &qtn_idx);
                (x_qtn_f, q_total_f, ixx_f, qtn_min_p_f, qtn_best_rows_f)
            };

        // Extract final beta/se/p from the last scan together with QTN beta/se overrides
        let (mut fem_final, _mf, row_stride_final) = scan_dense_full_matrix(
            y_slice, &x_qtn, &ixx, &g_flat, m, n, threads,
        ).map_err(PyRuntimeError::new_err)?;
        for i in 0..m {
            let base = i * row_stride_final;
            for c in 2..row_stride_final {
                let v = fem_final[base + c];
                if !v.is_finite() { fem_final[base + c] = 1.0; }
            }
        }

        let k_qtn_final = qtn_idx.len();
        let mut beta = vec![f64::NAN; m];
        let mut se = vec![f64::NAN; m];
        let mut pval = vec![f64::NAN; m];

        for i in 0..m {
            let base = i * row_stride_final;
            beta[i] = fem_final[base];
            se[i] = fem_final[base + 1];
            pval[i] = fem_final[base + row_stride_final - 1];
        }
        for (j, &idx) in qtn_idx.iter().enumerate() {
            if idx < m {
                pval[idx] = qtn_min_p[j];
            }
        }

        if k_qtn_final > 0 {
            let (qtn_full, qtn_full_stride) = fit_dense_full_rows(
                y_slice, &x_qtn, &ixx, &g_flat, m, n, &qtn_best_rows,
            ).map_err(PyRuntimeError::new_err)?;
            for j in 0..k_qtn_final {
                let coef_idx = q_base + j;
                let base = j * qtn_full_stride + 3 * coef_idx;
                let qidx = qtn_idx[j];
                if qidx < m {
                    beta[qidx] = qtn_full[base];
                    se[qidx] = qtn_full[base + 1];
                }
            }
        }

        Ok((beta, se, pval, qtn_idx.clone(), k_qtn_final))
    }).map(|(beta, se, pval, qtn_idx, n_pseudo_qtn)| {
        let beta_arr = PyArray1::<f64>::from_vec(py, beta).into_bound();
        let se_arr = PyArray1::<f64>::from_vec(py, se).into_bound();
        let pval_arr = PyArray1::<f64>::from_vec(py, pval).into_bound();
        let qtn_arr = PyArray1::<i64>::from_vec(
            py,
            qtn_idx.iter().map(|&x| x as i64).collect(),
        ).into_bound();
        (beta_arr, se_arr, pval_arr, qtn_arr, n_pseudo_qtn, n_obs, df_lrt)
    })
}

// =============================================================================
