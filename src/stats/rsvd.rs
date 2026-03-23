use nalgebra::{DMatrix, SymmetricEigen};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::borrow::Cow;

const _INTERRUPTED_MSG: &str = "Interrupted by user (Ctrl+C).";

#[inline]
fn _check_ctrlc() -> Result<(), String> {
    Python::attach(|py| py.check_signals())
        .map_err(|_| _INTERRUPTED_MSG.to_string())
}

#[inline]
fn _map_err_string_to_py(err: String) -> PyErr {
    if err.contains("Interrupted by user (Ctrl+C)") {
        PyKeyboardInterrupt::new_err(err)
    } else {
        PyRuntimeError::new_err(err)
    }
}

#[inline]
fn decode_plink_2bit(row: &[u8], sample_idx: usize) -> u8 {
    let b = row[sample_idx / 4];
    (b >> ((sample_idx % 4) * 2)) & 0b11
}

#[inline]
fn centered_from_plink_code(code: u8, f2: f32, flip: bool) -> Option<f32> {
    let mut g = match code {
        0b00 => 0.0_f32,
        0b10 => 1.0_f32,
        0b11 => 2.0_f32,
        _ => return None,
    };
    if flip {
        g = 2.0_f32 - g;
    }
    Some(g - f2)
}

fn packed_subset_row_stats(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    sample_idx: &[usize],
) -> Result<(Vec<f32>, Vec<bool>, f64), String> {
    if packed_flat.len() != m * bytes_per_snp {
        return Err("packed buffer length mismatch in packed_subset_row_stats".to_string());
    }
    let n = sample_idx.len();
    if n == 0 {
        return Err("sample_idx is empty in packed_subset_row_stats".to_string());
    }
    let mut row_freq = vec![0.0_f32; m];
    let mut row_flip = vec![false; m];
    let mut varsum = 0.0_f64;

    for i in 0..m {
        if (i & 1023usize) == 0usize {
            _check_ctrlc()?;
        }
        let row = &packed_flat[i * bytes_per_snp..(i + 1) * bytes_per_snp];
        let mut non_missing = 0usize;
        let mut alt_sum = 0.0_f64;
        for &sid in sample_idx.iter() {
            let code = decode_plink_2bit(row, sid);
            match code {
                0b00 => {
                    non_missing += 1;
                }
                0b10 => {
                    non_missing += 1;
                    alt_sum += 1.0;
                }
                0b11 => {
                    non_missing += 1;
                    alt_sum += 2.0;
                }
                _ => {}
            }
        }
        if non_missing == 0 {
            row_freq[i] = 0.0;
            row_flip[i] = false;
            continue;
        }
        let p = alt_sum / (2.0 * non_missing as f64);
        let flip = p > 0.5;
        let p_minor = if flip { 1.0 - p } else { p };
        row_freq[i] = p_minor as f32;
        row_flip[i] = flip;
        varsum += 2.0_f64 * p_minor * (1.0_f64 - p_minor);
    }

    if !(varsum.is_finite() && varsum > 0.0) {
        return Err("invalid scaling denominator in packed subset RSVD".to_string());
    }
    Ok((row_freq, row_flip, varsum))
}

fn compute_a_omega_packed_subset(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    sample_idx: &[usize],
    row_freq: &[f32],
    row_flip: &[bool],
    omega: &[f32],
    kp: usize,
) -> Result<Vec<f32>, String> {
    let n_sub = sample_idx.len();
    if packed_flat.len() != m * bytes_per_snp {
        return Err("packed buffer length mismatch in packed subset A*Omega".to_string());
    }
    if row_freq.len() != m || row_flip.len() != m {
        return Err("row stats length mismatch in packed subset A*Omega".to_string());
    }
    if omega.len() != n_sub * kp {
        return Err("omega shape mismatch in packed subset A*Omega".to_string());
    }
    let mut out = vec![0.0_f32; m * kp];
    // Smaller row blocks improve Ctrl+C responsiveness between Rayon kernels.
    let block_rows = 512usize.max(1).min(m.max(1));
    for row_start in (0..m).step_by(block_rows) {
        _check_ctrlc()?;
        let row_end = (row_start + block_rows).min(m);
        let out_blk = &mut out[row_start * kp..row_end * kp];
        out_blk
            .par_chunks_mut(kp)
            .enumerate()
            .for_each(|(off, out_row)| {
                let i = row_start + off;
                let row = &packed_flat[i * bytes_per_snp..(i + 1) * bytes_per_snp];
                let f2 = 2.0_f32 * row_freq[i];
                let flip = row_flip[i];
                for (l, &sid) in sample_idx.iter().enumerate() {
                    let code = decode_plink_2bit(row, sid);
                    if let Some(centered) = centered_from_plink_code(code, f2, flip) {
                        let omega_row = &omega[l * kp..(l + 1) * kp];
                        for j in 0..kp {
                            out_row[j] += centered * omega_row[j];
                        }
                    }
                }
            });
    }
    Ok(out)
}

fn compute_at_omega_packed_subset(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    sample_idx: &[usize],
    row_freq: &[f32],
    row_flip: &[bool],
    omega: &[f32],
    kp: usize,
    tile_cols: usize,
) -> Result<Vec<f32>, String> {
    let n_sub = sample_idx.len();
    if packed_flat.len() != m * bytes_per_snp {
        return Err("packed buffer length mismatch in packed subset A^T*Omega".to_string());
    }
    if row_freq.len() != m || row_flip.len() != m {
        return Err("row stats length mismatch in packed subset A^T*Omega".to_string());
    }
    if omega.len() != m * kp {
        return Err("omega shape mismatch in packed subset A^T*Omega".to_string());
    }
    let mut out = vec![0.0_f32; n_sub * kp];
    let tile = tile_cols.max(1).min(n_sub);
    for block_start in (0..n_sub).step_by(tile) {
        _check_ctrlc()?;
        let block_len = (n_sub - block_start).min(tile);
        let blk = (0..m)
            .into_par_iter()
            .fold(
                || vec![0.0_f32; block_len * kp],
                |mut local, i| {
                    let row = &packed_flat[i * bytes_per_snp..(i + 1) * bytes_per_snp];
                    let f2 = 2.0_f32 * row_freq[i];
                    let flip = row_flip[i];
                    let omega_row = &omega[i * kp..(i + 1) * kp];
                    for off in 0..block_len {
                        let l = block_start + off;
                        let sid = sample_idx[l];
                        let code = decode_plink_2bit(row, sid);
                        if let Some(centered) = centered_from_plink_code(code, f2, flip) {
                            let dst = &mut local[off * kp..(off + 1) * kp];
                            for j in 0..kp {
                                dst[j] += centered * omega_row[j];
                            }
                        }
                    }
                    local
                },
            )
            .reduce(
                || vec![0.0_f32; block_len * kp],
                |mut a, b| {
                    for idx in 0..(block_len * kp) {
                        a[idx] += b[idx];
                    }
                    a
                },
            );
        out[block_start * kp..(block_start + block_len) * kp].copy_from_slice(&blk);
    }
    Ok(out)
}

fn random_omega(m: usize, kp: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = vec![0.0_f32; m * kp];
    for v in omega.iter_mut() {
        *v = rng.sample::<f64, _>(StandardNormal) as f32;
    }
    omega
}

fn parse_index_vec_i64(src: &[i64], n_total: usize, name: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::with_capacity(src.len());
    for (i, &v) in src.iter().enumerate() {
        if v < 0 {
            return Err(format!("{name}[{i}] must be >= 0, got {v}"));
        }
        let u = v as usize;
        if u >= n_total {
            return Err(format!(
                "{name}[{i}] out of range: {u} >= {n_total}"
            ));
        }
        out.push(u);
    }
    Ok(out)
}

fn rsvd_tile_cols_env() -> usize {
    match std::env::var("JANUSX_ADMX_RSVD_TILE") {
        Ok(v) => v.trim().parse::<usize>().ok().unwrap_or(1024).max(1),
        Err(_) => 1024,
    }
}

fn thin_svd_from_tall(
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if rows == 0 || cols == 0 {
        return Err("invalid matrix shape for thin SVD".to_string());
    }
    if x.len() != rows * cols {
        return Err(format!(
            "matrix buffer size mismatch in thin SVD: len={}, expected={}",
            x.len(),
            rows * cols
        ));
    }

    let mut gram = vec![0.0_f64; cols * cols];
    for r in 0..rows {
        if (r & 1023usize) == 0usize {
            _check_ctrlc()?;
        }
        let row = &x[r * cols..(r + 1) * cols];
        for i in 0..cols {
            let xi = row[i] as f64;
            for j in 0..=i {
                gram[i * cols + j] += xi * (row[j] as f64);
            }
        }
    }
    for i in 0..cols {
        for j in 0..i {
            gram[j * cols + i] = gram[i * cols + j];
        }
    }

    let eig = SymmetricEigen::new(DMatrix::from_row_slice(cols, cols, &gram));
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));

    let mut s = vec![0.0_f32; cols];
    let mut v = vec![0.0_f32; cols * cols];
    for (new_col, &src_col) in order.iter().enumerate() {
        let eval = eig.eigenvalues[src_col].max(1e-12);
        s[new_col] = (eval as f32).sqrt();
        for r in 0..cols {
            v[r * cols + new_col] = eig.eigenvectors[(r, src_col)] as f32;
        }
    }

    let mut u = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let x_row = &x[r * cols..(r + 1) * cols];
        let u_row = &mut u[r * cols..(r + 1) * cols];
        for c in 0..cols {
            let mut acc = 0.0_f64;
            for t in 0..cols {
                acc += (x_row[t] as f64) * (v[t * cols + c] as f64);
            }
            let inv = if s[c] > 1e-12 {
                1.0_f64 / (s[c] as f64)
            } else {
                0.0_f64
            };
            u_row[c] = (acc * inv) as f32;
        }
    }
    Ok((u, s, v))
}

fn spectral_proxy_from_columns(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut s = vec![0.0_f32; cols];
    for c in 0..cols {
        let mut ss = 0.0_f64;
        for r in 0..rows {
            let v = x[r * cols + c] as f64;
            ss += v * v;
        }
        s[c] = (ss.max(0.0)).sqrt() as f32;
    }
    s.sort_by(|a, b| b.total_cmp(a));
    s
}

fn qr_normalize_mgs(
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if x.len() != rows * cols {
        return Err("buffer size mismatch in qr_normalize_mgs".to_string());
    }
    let mut q = vec![0.0_f32; rows * cols];
    let mut rdiag = vec![0.0_f32; cols];
    let mut v = vec![0.0_f64; rows];

    for j in 0..cols {
        for r in 0..rows {
            v[r] = x[r * cols + j] as f64;
        }

        for i in 0..j {
            let mut dot = 0.0_f64;
            for r in 0..rows {
                dot += (q[r * cols + i] as f64) * v[r];
            }
            for r in 0..rows {
                v[r] -= dot * (q[r * cols + i] as f64);
            }
        }

        let mut nrm2 = 0.0_f64;
        for r in 0..rows {
            nrm2 += v[r] * v[r];
        }
        let nrm = nrm2.sqrt();
        if !(nrm.is_finite() && nrm > 1e-12) {
            return Err(format!(
                "QR normalization failed: near-dependent column at j={j}"
            ));
        }
        rdiag[j] = nrm as f32;
        let inv = 1.0_f64 / nrm;
        for r in 0..rows {
            q[r * cols + j] = (v[r] * inv) as f32;
        }
    }
    Ok((q, rdiag))
}

fn lu_normalize_with_qr_fallback(
    x: &[f32],
    rows: usize,
    cols: usize,
    lu_eps: f64,
    cond_min_ratio: f64,
) -> Result<(Vec<f32>, bool), String> {
    if x.len() != rows * cols {
        return Err("buffer size mismatch in lu_normalize_with_qr_fallback".to_string());
    }
    if rows < cols {
        let (q_qr, _) = qr_normalize_mgs(x, rows, cols)?;
        return Ok((q_qr, false));
    }

    let mut a = vec![0.0_f64; rows * cols];
    for i in 0..(rows * cols) {
        a[i] = x[i] as f64;
    }
    let mut piv: Vec<usize> = (0..rows).collect();

    let mut dmin = f64::INFINITY;
    let mut dmax = 0.0_f64;
    let mut lu_ok = true;

    for j in 0..cols {
        let mut piv_row = j;
        let mut piv_abs = 0.0_f64;
        for i in j..rows {
            let v = a[i * cols + j].abs();
            if v > piv_abs {
                piv_abs = v;
                piv_row = i;
            }
        }
        if !(piv_abs.is_finite() && piv_abs > lu_eps) {
            lu_ok = false;
            break;
        }

        if piv_row != j {
            for c in 0..cols {
                a.swap(j * cols + c, piv_row * cols + c);
            }
            piv.swap(j, piv_row);
        }

        let ajj = a[j * cols + j];
        let ad = ajj.abs();
        if !(ad.is_finite() && ad > lu_eps) {
            lu_ok = false;
            break;
        }
        if ad < dmin {
            dmin = ad;
        }
        if ad > dmax {
            dmax = ad;
        }

        let inv = 1.0_f64 / ajj;
        for i in (j + 1)..rows {
            let lij = a[i * cols + j] * inv;
            a[i * cols + j] = lij;
            for c in (j + 1)..cols {
                a[i * cols + c] -= lij * a[j * cols + c];
            }
        }
    }

    if lu_ok {
        if !(dmax.is_finite() && dmax > lu_eps) {
            lu_ok = false;
        } else if !(dmin.is_finite() && dmin > lu_eps) {
            lu_ok = false;
        } else {
            let ratio = dmin / dmax;
            if !(ratio.is_finite() && ratio >= cond_min_ratio) {
                lu_ok = false;
            }
        }
    }

    if !lu_ok {
        let (q_qr, _) = qr_normalize_mgs(x, rows, cols)?;
        return Ok((q_qr, false));
    }

    // A now stores compact LU of (P * X). Build Q := P^T * L (same column space as X).
    let mut q = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        let orig_row = piv[i];
        let dst = &mut q[orig_row * cols..(orig_row + 1) * cols];
        for j in 0..cols {
            let val = if i < cols {
                if i == j {
                    1.0_f64
                } else if i > j {
                    a[i * cols + j]
                } else {
                    0.0_f64
                }
            } else {
                a[i * cols + j]
            };
            dst[j] = val as f32;
        }
    }

    // Column scaling keeps numerics bounded while preserving span.
    for j in 0..cols {
        let mut ss = 0.0_f64;
        for i in 0..rows {
            let v = q[i * cols + j] as f64;
            ss += v * v;
        }
        let nrm = ss.sqrt();
        if !(nrm.is_finite() && nrm > lu_eps) {
            let (q_qr, _) = qr_normalize_mgs(x, rows, cols)?;
            return Ok((q_qr, false));
        }
        let inv = 1.0_f64 / nrm;
        for i in 0..rows {
            q[i * cols + j] = ((q[i * cols + j] as f64) * inv) as f32;
        }
    }
    Ok((q, true))
}

pub fn rsvd_packed_subset(
    packed_flat: &[u8],
    m: usize,
    bytes_per_snp: usize,
    sample_idx: &[usize],
    k: usize,
    seed: u64,
    power: usize,
    tol: f32,
    tile_cols: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>, usize), String> {
    if k == 0 {
        return Err("k must be > 0".to_string());
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err("tol must be positive and finite".to_string());
    }
    let n = sample_idx.len();
    if n == 0 {
        return Err("sample_idx must not be empty".to_string());
    }
    let (row_freq, row_flip, varsum) =
        packed_subset_row_stats(packed_flat, m, bytes_per_snp, sample_idx)?;
    let k_eff = k.min(n);
    let kp = (k_eff + 10).max(20).min(m.max(1)).min(n.max(1));

    let omega = random_omega(m, kp, seed);
    let mut y = compute_at_omega_packed_subset(
        packed_flat,
        m,
        bytes_per_snp,
        sample_idx,
        &row_freq,
        &row_flip,
        &omega,
        kp,
        tile_cols,
    )?;
    let (mut q, _) = qr_normalize_mgs(&y, n, kp)?;

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    let lu_eps = 1e-10_f64;
    let lu_cond_ratio = 1e-8_f64;
    let mut q_is_qr = true;
    // Power iteration acceleration:
    // - first (power-1) rounds: LU normalization with automatic QR fallback on ill-conditioning
    // - final round (or early-exit finalization): QR normalization for stability
    for it in 0..power {
        _check_ctrlc()?;
        let g_small = compute_a_omega_packed_subset(
            packed_flat,
            m,
            bytes_per_snp,
            sample_idx,
            &row_freq,
            &row_flip,
            &q,
            kp,
        )?;
        y = compute_at_omega_packed_subset(
            packed_flat,
            m,
            bytes_per_snp,
            sample_idx,
            &row_freq,
            &row_flip,
            &g_small,
            kp,
            tile_cols,
        )?;
        for idx in 0..(n * kp) {
            y[idx] -= alpha * q[idx];
        }
        let force_qr = (it + 1) >= power;
        if force_qr {
            let (q_new, _) = qr_normalize_mgs(&y, n, kp)?;
            q = q_new;
            q_is_qr = true;
        } else {
            let (q_new, used_lu) = lu_normalize_with_qr_fallback(
                &y,
                n,
                kp,
                lu_eps,
                lu_cond_ratio,
            )?;
            q = q_new;
            q_is_qr = !used_lu;
        }
        let s_y = spectral_proxy_from_columns(&y, n, kp);

        if it > 0 {
            let mut max_rel = 0.0_f32;
            for i in 0..k_eff {
                let sk_now = s_y[i] + alpha;
                let denom = sk_now.max(1e-12);
                let rel = ((sk_now - sk[i]).abs()) / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                sk[i] = sk_now;
            }
            if max_rel < tol {
                if !q_is_qr {
                    let (q_new, _) = qr_normalize_mgs(&q, n, kp)?;
                    q = q_new;
                    q_is_qr = true;
                }
                break;
            }
        } else {
            for i in 0..kp {
                sk[i] = s_y[i] + alpha;
            }
        }

        let tail = s_y[kp - 1];
        if alpha < tail {
            alpha = 0.5 * (alpha + tail);
        }
    }
    if !q_is_qr {
        let (q_new, _) = qr_normalize_mgs(&q, n, kp)?;
        q = q_new;
    }

    let g_small = compute_a_omega_packed_subset(
        packed_flat,
        m,
        bytes_per_snp,
        sample_idx,
        &row_freq,
        &row_flip,
        &q,
        kp,
    )?;
    let (_, s_all, v_small) = thin_svd_from_tall(&g_small, m, kp)?;

    let scale = varsum as f32;
    let mut eigvals = vec![0.0_f32; k_eff];
    for i in 0..k_eff {
        eigvals[i] = (s_all[i] * s_all[i]) / scale;
    }

    let mut eigvecs_sample = vec![0.0_f32; n * k_eff];
    for r in 0..n {
        let q_row = &q[r * kp..(r + 1) * kp];
        let out_row = &mut eigvecs_sample[r * k_eff..(r + 1) * k_eff];
        for c in 0..k_eff {
            let mut acc = 0.0_f64;
            for t in 0..kp {
                acc += (q_row[t] as f64) * (v_small[t * kp + c] as f64);
            }
            out_row[c] = acc as f32;
        }
    }

    Ok((eigvals, eigvecs_sample, row_freq, row_flip, k_eff))
}

#[pyfunction(name = "rsvd_packed_subset")]
#[pyo3(signature = (
    packed,
    n_samples,
    k,
    sample_indices=None,
    seed=42,
    power=5,
    tol=1e-1
))]
pub fn py_rsvd_packed_subset<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    k: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    seed: u64,
    power: usize,
    tol: f32,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<bool>>,
)> {
    if k == 0 {
        return Err(PyRuntimeError::new_err("k must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("packed must be 2D (m, bytes_per_snp)"));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    if m == 0 {
        return Err(PyRuntimeError::new_err("packed matrix has zero SNP rows"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        parse_index_vec_i64(sidx_slice, n_samples, "sample_indices")
            .map_err(PyRuntimeError::new_err)?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };

    let tile_cols = rsvd_tile_cols_env();
    let (eigvals, eigvecs_sample, row_freq, row_flip, k_eff) = py
        .detach(|| {
            rsvd_packed_subset(
                &packed_flat,
                m,
                bytes_per_snp,
                &sample_idx,
                k,
                seed,
                power,
                tol,
                tile_cols,
            )
        })
        .map_err(_map_err_string_to_py)?;

    let eval_arr = PyArray1::<f32>::zeros(py, [k_eff], false).into_bound();
    let evec_arr = PyArray2::<f32>::zeros(py, [n, k_eff], false).into_bound();
    let maf_arr = PyArray1::<f32>::zeros(py, [m], false).into_bound();
    let flip_arr = PyArray1::<bool>::zeros(py, [m], false).into_bound();

    let eval_slice = unsafe {
        eval_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("eigvals output not contiguous"))?
    };
    let evec_slice = unsafe {
        evec_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("sample eigvecs output not contiguous"))?
    };
    let maf_slice = unsafe {
        maf_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("row maf output not contiguous"))?
    };
    let flip_slice = unsafe {
        flip_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("row flip output not contiguous"))?
    };

    eval_slice.copy_from_slice(&eigvals);
    evec_slice.copy_from_slice(&eigvecs_sample);
    maf_slice.copy_from_slice(&row_freq);
    flip_slice.copy_from_slice(&row_flip);
    Ok((eval_arr, evec_arr, maf_arr, flip_arr))
}
