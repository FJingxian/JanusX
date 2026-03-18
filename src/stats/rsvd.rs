use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

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
    out.par_chunks_mut(kp).enumerate().for_each(|(i, out_row)| {
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
    let kp = (k_eff + 10).max(20).min(m.max(1));

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
    let (mut q, _, _) = thin_svd_from_tall(&y, n, kp)?;
    let mut s_y = vec![0.0_f32; kp];

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    for it in 0..power {
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
        let (q_new, s_new, _) = thin_svd_from_tall(&y, n, kp)?;
        q = q_new;
        s_y = s_new;

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
