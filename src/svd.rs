use std::borrow::Cow;
use std::sync::Arc;

use matrixmultiply::dgemm;
use nalgebra::{DMatrix, SymmetricEigen};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rand::Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::gfcore::BedSnpIter;
use crate::gfreader::{build_sample_selection, build_snp_indices};

fn matmul_a_b(a: &[f64], m: usize, n: usize, b: &[f64], l: usize, block_rows: usize) -> Vec<f64> {
    let mut y = vec![0.0; m * l];
    let block = block_rows.max(1).min(m);
    let mut i = 0usize;
    while i < m {
        let i_end = (i + block).min(m);
        for row in i..i_end {
            let a_row = &a[row * n..(row + 1) * n];
            let y_row = &mut y[row * l..(row + 1) * l];
            for k in 0..n {
                let aik = a_row[k];
                if aik != 0.0 {
                    let b_row = &b[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += aik * b_row[j];
                    }
                }
            }
        }
        i = i_end;
    }
    y
}

fn matmul_at_b(a: &[f64], m: usize, n: usize, b: &[f64], l: usize, block_rows: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * l];
    let block = block_rows.max(1).min(m);
    let mut i = 0usize;
    while i < m {
        let i_end = (i + block).min(m);
        for row in i..i_end {
            let a_row = &a[row * n..(row + 1) * n];
            let b_row = &b[row * l..(row + 1) * l];
            for k in 0..n {
                let aik = a_row[k];
                if aik != 0.0 {
                    let out_row = &mut out[k * l..(k + 1) * l];
                    for j in 0..l {
                        out_row[j] += aik * b_row[j];
                    }
                }
            }
        }
        i = i_end;
    }
    out
}

fn matmul_qt_a(q: &[f64], m: usize, l: usize, a: &[f64], n: usize, block_rows: usize) -> Vec<f64> {
    let mut out = vec![0.0; l * n];
    let block = block_rows.max(1).min(m);
    let mut i = 0usize;
    while i < m {
        let i_end = (i + block).min(m);
        for row in i..i_end {
            let q_row = &q[row * l..(row + 1) * l];
            let a_row = &a[row * n..(row + 1) * n];
            for j in 0..l {
                let qij = q_row[j];
                if qij != 0.0 {
                    let out_row = &mut out[j * n..(j + 1) * n];
                    for k in 0..n {
                        out_row[k] += qij * a_row[k];
                    }
                }
            }
        }
        i = i_end;
    }
    out
}

fn gram_b(b: &[f64], l: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; l * l];
    for i in 0..l {
        let row_i = &b[i * n..(i + 1) * n];
        for j in i..l {
            let row_j = &b[j * n..(j + 1) * n];
            let mut sum = 0.0;
            for k in 0..n {
                sum += row_i[k] * row_j[k];
            }
            c[i * l + j] = sum;
            c[j * l + i] = sum;
        }
    }
    c
}

fn dmatrix_to_rowmajor(mat: &DMatrix<f64>) -> Vec<f64> {
    let (m, n) = mat.shape();
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = mat[(i, j)];
        }
    }
    out
}

fn transpose_rowmajor(src: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut dst = vec![0.0; rows * cols];
    for i in 0..rows {
        let row = &src[i * cols..(i + 1) * cols];
        for j in 0..cols {
            dst[j * rows + i] = row[j];
        }
    }
    dst
}

#[derive(Clone, Copy)]
struct RowStats {
    mean: f64,
    inv_std: f64,
}

fn gemm_rowmajor(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    rsa: isize,
    csa: isize,
    b: &[f64],
    rsb: isize,
    csb: isize,
    c: &mut [f64],
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    debug_assert!(c.len() >= m * n);
    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            rsa,
            csa,
            b.as_ptr(),
            rsb,
            csb,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

fn compute_u_row(q_row: &[f64], m: usize, l: usize, u_b_row: &[f64], k: usize) -> Vec<f64> {
    let mut u_row = vec![0.0; m * k];
    if m == 0 || l == 0 || k == 0 {
        return u_row;
    }
    gemm_rowmajor(
        m,
        l,
        k,
        q_row,
        l as isize,
        1,
        u_b_row,
        k as isize,
        1,
        &mut u_row,
    );
    u_row
}

fn compute_vt_from_b(u_b_row: &[f64], k: usize, l: usize, b: &[f64], n: usize, s: &[f64]) -> Vec<f64> {
    let mut vt = vec![0.0; k * n];
    if k == 0 || l == 0 || n == 0 {
        return vt;
    }
    gemm_rowmajor(
        k,
        l,
        n,
        u_b_row,
        1,
        k as isize,
        b,
        n as isize,
        1,
        &mut vt,
    );
    for col in 0..k {
        let sigma = s[col];
        if sigma == 0.0 {
            let row = &mut vt[col * n..(col + 1) * n];
            row.fill(0.0);
            continue;
        }
        let inv_sigma = 1.0 / sigma;
        let row = &mut vt[col * n..(col + 1) * n];
        for val in row.iter_mut() {
            *val *= inv_sigma;
        }
    }
    vt
}

trait GenoIter {
    fn n_samples(&self) -> usize;
    fn next_row(&mut self) -> Option<Vec<f32>>;
}

struct BedSnpIterSelect {
    it: BedSnpIter,
    snp_indices: Option<Arc<Vec<usize>>>,
    snp_pos: usize,
    sample_indices: Arc<Vec<usize>>,
    full_samples: bool,
}

impl GenoIter for BedSnpIter {
    fn n_samples(&self) -> usize {
        BedSnpIter::n_samples(self)
    }

    fn next_row(&mut self) -> Option<Vec<f32>> {
        BedSnpIter::next_snp(self).map(|(row, _)| row)
    }
}

impl GenoIter for BedSnpIterSelect {
    fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    fn next_row(&mut self) -> Option<Vec<f32>> {
        if let Some(ref snp_indices) = self.snp_indices {
            while self.snp_pos < snp_indices.len() {
                let snp_idx = snp_indices[self.snp_pos];
                self.snp_pos += 1;
                if let Some((row, _site)) = self.it.get_snp_row(snp_idx) {
                    if self.full_samples {
                        return Some(row);
                    }
                    let mut out = Vec::with_capacity(self.sample_indices.len());
                    for &idx in self.sample_indices.iter() {
                        out.push(row[idx]);
                    }
                    return Some(out);
                }
            }
            return None;
        }

        match self.it.next_snp() {
            Some((row, _site)) => {
                if self.full_samples {
                    Some(row)
                } else {
                    let mut out = Vec::with_capacity(self.sample_indices.len());
                    for &idx in self.sample_indices.iter() {
                        out.push(row[idx]);
                    }
                    Some(out)
                }
            }
            None => None,
        }
    }
}

#[inline]
fn row_mean_invstd(row: &[f32], center: bool, scale: bool) -> (f64, f64) {
    if !center && !scale {
        return (0.0, 1.0);
    }
    let sum: f64 = row.iter().map(|&v| v as f64).sum();
    let mean = sum / row.len() as f64;
    if !scale {
        return (mean, 1.0);
    }
    let p = (mean / 2.0).clamp(0.0, 1.0);
    let var = 2.0 * p * (1.0 - p);
    if var <= 0.0 {
        return (mean, 0.0);
    }
    (mean, 1.0 / var.sqrt())
}

#[inline]
fn row_mean_invstd_selected(
    row: &[f32],
    sample_indices: &[usize],
    center: bool,
    scale: bool,
    full_samples: bool,
) -> (f64, f64) {
    if !center && !scale {
        return (0.0, 1.0);
    }
    if full_samples {
        return row_mean_invstd(row, center, scale);
    }
    let sum: f64 = sample_indices.iter().map(|&i| row[i] as f64).sum();
    let mean = sum / sample_indices.len() as f64;
    if !scale {
        return (mean, 1.0);
    }
    let p = (mean / 2.0).clamp(0.0, 1.0);
    let var = 2.0 * p * (1.0 - p);
    if var <= 0.0 {
        return (mean, 0.0);
    }
    (mean, 1.0 / var.sqrt())
}

fn compute_y_with_iter<I: GenoIter>(
    mut it: I,
    omega: &[f64],
    l: usize,
    center: bool,
    scale: bool,
) -> Result<(Vec<f64>, usize, Option<Vec<RowStats>>), String> {
    let n = it.n_samples();
    if omega.len() != n * l {
        return Err("omega size mismatch".into());
    }
    let cache_stats = center || scale;
    let mut stats = if cache_stats { Some(Vec::new()) } else { None };
    let mut y: Vec<f64> = Vec::new();
    while let Some(row) = it.next_row() {
        let base = y.len();
        y.resize(base + l, 0.0);
        let y_row = &mut y[base..base + l];
        if !center && !scale {
            for k in 0..n {
                let val = row[k] as f64;
                if val != 0.0 {
                    let omega_row = &omega[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += val * omega_row[j];
                    }
                }
            }
        } else {
            let (mean, inv_std) = row_mean_invstd(&row, center, scale);
            if let Some(ref mut stats_vec) = stats {
                stats_vec.push(RowStats { mean, inv_std });
            }
            if scale && inv_std == 0.0 {
                continue;
            }
            for k in 0..n {
                let mut val = row[k] as f64 - mean;
                if scale {
                    val *= inv_std;
                }
                if val != 0.0 {
                    let omega_row = &omega[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += val * omega_row[j];
                    }
                }
            }
        }
    }
    Ok((y, n, stats))
}

fn compute_y_rows_bed_parallel(
    it: &BedSnpIter,
    snp_indices: Option<&[usize]>,
    sample_indices: &[usize],
    full_samples: bool,
    mat: &[f64],
    l: usize,
    center: bool,
    scale: bool,
    retained: Option<&[usize]>,
    stats: Option<&[RowStats]>,
    pool: &rayon::ThreadPool,
) -> Result<(Vec<f64>, Vec<usize>, Option<Vec<RowStats>>), String> {
    let n_samples = sample_indices.len();
    if mat.len() != n_samples * l {
        return Err("projection matrix size mismatch".into());
    }
    if stats.is_some() && retained.is_none() {
        return Err("stats require retained SNP list".into());
    }
    if let (Some(stats), Some(retained)) = (stats, retained) {
        if stats.len() != retained.len() {
            return Err("stats length mismatch".into());
        }
    }
    let block_snps = 256usize;
    let cache_stats = stats.is_none() && (center || scale);
    let use_stats = stats.is_some() && (center || scale);
    let stats_slice = stats.unwrap_or(&[]);

    let (total, mode_list): (usize, Option<&[usize]>) = if let Some(retained_idx) = retained {
        (retained_idx.len(), Some(retained_idx))
    } else if let Some(list) = snp_indices {
        (list.len(), Some(list))
    } else {
        (it.n_snps(), None)
    };
    if total == 0 {
        let stats_out = if cache_stats { Some(Vec::new()) } else { None };
        return Ok((Vec::new(), Vec::new(), stats_out));
    }

    let blocks: Vec<(usize, usize)> = (0..total)
        .step_by(block_snps)
        .map(|start| (start, (start + block_snps).min(total)))
        .collect();

    let results: Vec<(usize, Vec<usize>, Vec<f64>, Vec<RowStats>)> = pool.install(|| {
        blocks
            .par_iter()
            .enumerate()
            .map(|(block_idx, (start, end))| {
                let mut kept: Vec<usize> = Vec::new();
                let mut rows: Vec<f64> = Vec::new();
                let mut stats_block: Vec<RowStats> = Vec::new();
                for i in *start..*end {
                    let snp_idx = if let Some(list) = mode_list {
                        list[i]
                    } else {
                        i
                    };
                    let Some((row, _)) = it.get_snp_row(snp_idx) else {
                        if retained.is_some() {
                            return Err("SNP missing during pass".to_string());
                        }
                        continue;
                    };
                    let (mean, inv_std) = if !center && !scale {
                        (0.0, 1.0)
                    } else if use_stats {
                        let stat = stats_slice
                            .get(i)
                            .ok_or_else(|| "stats length mismatch".to_string())?;
                        (stat.mean, stat.inv_std)
                    } else {
                        row_mean_invstd_selected(
                            &row,
                            sample_indices,
                            center,
                            scale,
                            full_samples,
                        )
                    };
                    if scale && inv_std == 0.0 {
                        if retained.is_some() {
                            return Err("SNP variance is zero in later pass".to_string());
                        }
                        continue;
                    }
                    let base = rows.len();
                    rows.resize(base + l, 0.0);
                    let y_row = &mut rows[base..base + l];
                    if !center && !scale {
                        if full_samples {
                            for k in 0..n_samples {
                                let val = row[k] as f64;
                                if val != 0.0 {
                                    let mat_row = &mat[k * l..(k + 1) * l];
                                    for j in 0..l {
                                        y_row[j] += val * mat_row[j];
                                    }
                                }
                            }
                        } else {
                            for (sel_pos, &orig_idx) in sample_indices.iter().enumerate() {
                                let val = row[orig_idx] as f64;
                                if val != 0.0 {
                                    let mat_row = &mat[sel_pos * l..(sel_pos + 1) * l];
                                    for j in 0..l {
                                        y_row[j] += val * mat_row[j];
                                    }
                                }
                            }
                        }
                    } else {
                        if full_samples {
                            for k in 0..n_samples {
                                let mut val = row[k] as f64 - mean;
                                if scale {
                                    val *= inv_std;
                                }
                                if val != 0.0 {
                                    let mat_row = &mat[k * l..(k + 1) * l];
                                    for j in 0..l {
                                        y_row[j] += val * mat_row[j];
                                    }
                                }
                            }
                        } else {
                            for (sel_pos, &orig_idx) in sample_indices.iter().enumerate() {
                                let mut val = row[orig_idx] as f64 - mean;
                                if scale {
                                    val *= inv_std;
                                }
                                if val != 0.0 {
                                    let mat_row = &mat[sel_pos * l..(sel_pos + 1) * l];
                                    for j in 0..l {
                                        y_row[j] += val * mat_row[j];
                                    }
                                }
                            }
                        }
                    }
                    kept.push(snp_idx);
                    if cache_stats {
                        stats_block.push(RowStats { mean, inv_std });
                    }
                }
                Ok((block_idx, kept, rows, stats_block))
            })
            .collect::<Result<Vec<_>, String>>()
    })?;

    let mut results = results;
    results.sort_by_key(|(idx, _, _, _)| *idx);

    let mut y: Vec<f64> = Vec::new();
    let mut kept: Vec<usize> = Vec::new();
    let mut stats_out: Option<Vec<RowStats>> = if cache_stats { Some(Vec::new()) } else { None };
    for (_idx, mut blk_kept, blk_rows, blk_stats) in results {
        kept.append(&mut blk_kept);
        y.extend_from_slice(&blk_rows);
        if let Some(ref mut stats_vec) = stats_out {
            stats_vec.extend_from_slice(&blk_stats);
        }
    }

    Ok((y, kept, stats_out))
}

fn compute_at_y_bed_parallel(
    it: &BedSnpIter,
    retained: &[usize],
    sample_indices: &[usize],
    full_samples: bool,
    y: &[f64],
    l: usize,
    center: bool,
    scale: bool,
    stats: Option<&[RowStats]>,
    pool: &rayon::ThreadPool,
) -> Result<Vec<f64>, String> {
    let n_samples = sample_indices.len();
    if y.len() != retained.len() * l {
        return Err("Y size mismatch".into());
    }
    if let Some(stats) = stats {
        if stats.len() != retained.len() {
            return Err("stats length mismatch in A^T*Y".into());
        }
    }
    let block_snps = 256usize;
    let blocks: Vec<(usize, usize)> = (0..retained.len())
        .step_by(block_snps)
        .map(|start| (start, (start + block_snps).min(retained.len())))
        .collect();

    let z = pool.install(|| {
        blocks
            .par_iter()
            .map(|(start, end)| {
                let mut z_block = vec![0.0f64; n_samples * l];
                for row_idx in *start..*end {
                    let snp_idx = retained[row_idx];
                    let y_row = &y[row_idx * l..(row_idx + 1) * l];
                    let Some((row, _)) = it.get_snp_row(snp_idx) else {
                        return Err("SNP missing during A^T*Y".to_string());
                    };
                    if !center && !scale {
                        if full_samples {
                            for k in 0..n_samples {
                                let val = row[k] as f64;
                                if val != 0.0 {
                                    let z_row = &mut z_block[k * l..(k + 1) * l];
                                    for j in 0..l {
                                        z_row[j] += val * y_row[j];
                                    }
                                }
                            }
                        } else {
                            for (sel_pos, &orig_idx) in sample_indices.iter().enumerate() {
                                let val = row[orig_idx] as f64;
                                if val != 0.0 {
                                    let z_row = &mut z_block[sel_pos * l..(sel_pos + 1) * l];
                                    for j in 0..l {
                                        z_row[j] += val * y_row[j];
                                    }
                                }
                            }
                        }
                    } else {
                        let (mean, inv_std) = if let Some(stats) = stats {
                            let stat = stats
                                .get(row_idx)
                                .ok_or_else(|| "stats length mismatch in A^T*Y".to_string())?;
                            (stat.mean, stat.inv_std)
                        } else {
                            row_mean_invstd_selected(
                                &row,
                                sample_indices,
                                center,
                                scale,
                                full_samples,
                            )
                        };
                        if scale && inv_std == 0.0 {
                            return Err("SNP variance is zero in later pass".to_string());
                        }
                        if full_samples {
                            for k in 0..n_samples {
                                let mut val = row[k] as f64 - mean;
                                if scale {
                                    val *= inv_std;
                                }
                                if val != 0.0 {
                                    let z_row = &mut z_block[k * l..(k + 1) * l];
                                    for j in 0..l {
                                        z_row[j] += val * y_row[j];
                                    }
                                }
                            }
                        } else {
                            for (sel_pos, &orig_idx) in sample_indices.iter().enumerate() {
                                let mut val = row[orig_idx] as f64 - mean;
                                if scale {
                                    val *= inv_std;
                                }
                                if val != 0.0 {
                                    let z_row = &mut z_block[sel_pos * l..(sel_pos + 1) * l];
                                    for j in 0..l {
                                        z_row[j] += val * y_row[j];
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(z_block)
            })
            .reduce(
                || Ok(vec![0.0f64; n_samples * l]),
                |a, b| {
                    let mut a = a?;
                    let b = b?;
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += y;
                    }
                    Ok(a)
                },
            )
    })?;

    Ok(z)
}

fn compute_qt_a_bed_parallel(
    it: &BedSnpIter,
    retained: &[usize],
    sample_indices: &[usize],
    full_samples: bool,
    q: &[f64],
    l: usize,
    center: bool,
    scale: bool,
    stats: Option<&[RowStats]>,
    pool: &rayon::ThreadPool,
) -> Result<Vec<f64>, String> {
    let n_samples = sample_indices.len();
    if q.len() != retained.len() * l {
        return Err("Q size mismatch".into());
    }
    if let Some(stats) = stats {
        if stats.len() != retained.len() {
            return Err("stats length mismatch in Q^T*A".into());
        }
    }
    let block_snps = 256usize;
    let blocks: Vec<(usize, usize)> = (0..retained.len())
        .step_by(block_snps)
        .map(|start| (start, (start + block_snps).min(retained.len())))
        .collect();

    let b = pool.install(|| {
        blocks
            .par_iter()
            .map(|(start, end)| {
                let mut b_block = vec![0.0f64; l * n_samples];
                for row_idx in *start..*end {
                    let snp_idx = retained[row_idx];
                    let q_row = &q[row_idx * l..(row_idx + 1) * l];
                    let Some((row, _)) = it.get_snp_row(snp_idx) else {
                        return Err("SNP missing during Q^T*A".to_string());
                    };
                    if !center && !scale {
                        if full_samples {
                            for k in 0..n_samples {
                                let val = row[k] as f64;
                                if val != 0.0 {
                                    for j in 0..l {
                                        b_block[j * n_samples + k] += q_row[j] * val;
                                    }
                                }
                            }
                        } else {
                            for (sel_pos, &orig_idx) in sample_indices.iter().enumerate() {
                                let val = row[orig_idx] as f64;
                                if val != 0.0 {
                                    for j in 0..l {
                                        b_block[j * n_samples + sel_pos] += q_row[j] * val;
                                    }
                                }
                            }
                        }
                    } else {
                        let (mean, inv_std) = if let Some(stats) = stats {
                            let stat = stats
                                .get(row_idx)
                                .ok_or_else(|| "stats length mismatch in Q^T*A".to_string())?;
                            (stat.mean, stat.inv_std)
                        } else {
                            row_mean_invstd_selected(
                                &row,
                                sample_indices,
                                center,
                                scale,
                                full_samples,
                            )
                        };
                        if scale && inv_std == 0.0 {
                            return Err("SNP variance is zero in later pass".to_string());
                        }
                        if full_samples {
                            for k in 0..n_samples {
                                let mut val = row[k] as f64 - mean;
                                if scale {
                                    val *= inv_std;
                                }
                                if val != 0.0 {
                                    for j in 0..l {
                                        b_block[j * n_samples + k] += q_row[j] * val;
                                    }
                                }
                            }
                        } else {
                            for (sel_pos, &orig_idx) in sample_indices.iter().enumerate() {
                                let mut val = row[orig_idx] as f64 - mean;
                                if scale {
                                    val *= inv_std;
                                }
                                if val != 0.0 {
                                    for j in 0..l {
                                        b_block[j * n_samples + sel_pos] += q_row[j] * val;
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(b_block)
            })
            .reduce(
                || Ok(vec![0.0f64; l * n_samples]),
                |a, b| {
                    let mut a = a?;
                    let b = b?;
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += y;
                    }
                    Ok(a)
                },
            )
    })?;

    Ok(b)
}

fn randomized_svd_bed_parallel(
    it: &BedSnpIter,
    snp_indices: Option<&[usize]>,
    sample_indices: &[usize],
    full_samples: bool,
    k: usize,
    oversample: usize,
    n_iter: usize,
    center: bool,
    scale: bool,
    seed: u64,
    threads: usize,
    need_u: bool,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize), String> {
    let n = sample_indices.len();
    if n == 0 {
        return Err("no samples selected".into());
    }
    let min_dim = n;
    if k == 0 || k > min_dim {
        return Err(format!("k must be in 1..={min_dim}"));
    }
    let l = k.checked_add(oversample).ok_or("k + oversample overflow")?;
    if l == 0 || l > min_dim {
        return Err("k + oversample must be in 1..=n_samples".into());
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| e.to_string())?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = vec![0.0; n * l];
    for i in 0..n {
        let row = &mut omega[i * l..(i + 1) * l];
        for j in 0..l {
            row[j] = rng.sample(StandardNormal);
        }
    }

    let (mut y, retained, stats) = compute_y_rows_bed_parallel(
        it,
        snp_indices,
        sample_indices,
        full_samples,
        &omega,
        l,
        center,
        scale,
        None,
        None,
        &pool,
    )?;
    let m_eff = retained.len();
    if m_eff == 0 {
        return Err("no variants left after QC".into());
    }
    if l > m_eff {
        return Err("k + oversample exceeds retained variant count".into());
    }

    for _ in 0..n_iter {
        let z = compute_at_y_bed_parallel(
            it,
            &retained,
            sample_indices,
            full_samples,
            &y,
            l,
            center,
            scale,
            stats.as_deref(),
            &pool,
        )?;
        let (y_new, retained_new, _stats_new) = compute_y_rows_bed_parallel(
            it,
            None,
            sample_indices,
            full_samples,
            &z,
            l,
            center,
            scale,
            Some(&retained),
            stats.as_deref(),
            &pool,
        )?;
        if retained_new.len() != retained.len() {
            return Err("inconsistent SNP filtering across passes".into());
        }
        y = y_new;
    }

    let y_mat = DMatrix::from_row_slice(m_eff, l, &y);
    let qr = y_mat.qr();
    let q_full = qr.q();
    let q = q_full.columns(0, l).into_owned();
    let q_row = dmatrix_to_rowmajor(&q);

    let b = compute_qt_a_bed_parallel(
        it,
        &retained,
        sample_indices,
        full_samples,
        &q_row,
        l,
        center,
        scale,
        stats.as_deref(),
        &pool,
    )?;

    let c = gram_b(&b, l, n);
    let c_mat = DMatrix::from_row_slice(l, l, &c);
    let eig = SymmetricEigen::new(c_mat);

    let mut pairs: Vec<(f64, usize)> = eig
        .eigenvalues
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut s = Vec::with_capacity(k);
    let mut u_b = DMatrix::<f64>::zeros(l, k);
    for (col, (val, idx)) in pairs.into_iter().take(k).enumerate() {
        let sigma = val.max(0.0).sqrt();
        s.push(sigma);
        let ucol = eig.eigenvectors.column(idx);
        u_b.set_column(col, &ucol);
    }

    let u_b_row = dmatrix_to_rowmajor(&u_b);
    let u_row = if need_u {
        compute_u_row(&q_row, m_eff, l, &u_b_row, k)
    } else {
        Vec::new()
    };
    let vt = compute_vt_from_b(&u_b_row, k, l, &b, n, &s);

    Ok((u_row, s, vt, m_eff, n))
}

fn compute_at_y_with_iter<I: GenoIter>(
    mut it: I,
    y: &[f64],
    l: usize,
    center: bool,
    scale: bool,
    stats: Option<&[RowStats]>,
) -> Result<(Vec<f64>, usize), String> {
    let n = it.n_samples();
    if let Some(stats) = stats {
        if stats.len() * l != y.len() {
            return Err("stats length mismatch in A^T*Y".into());
        }
    }
    let mut z = vec![0.0f64; n * l];
    let mut row_idx: usize = 0;
    while let Some(row) = it.next_row() {
        let y_row = y
            .get(row_idx * l..(row_idx + 1) * l)
            .ok_or_else(|| "row count mismatch in A^T*Y".to_string())?;
        if !center && !scale {
            for k in 0..n {
                let val = row[k] as f64;
                if val != 0.0 {
                    let z_row = &mut z[k * l..(k + 1) * l];
                    for j in 0..l {
                        z_row[j] += val * y_row[j];
                    }
                }
            }
        } else {
            let (mean, inv_std) = if let Some(stats) = stats {
                let stat = stats
                    .get(row_idx)
                    .ok_or_else(|| "stats length mismatch in A^T*Y".to_string())?;
                (stat.mean, stat.inv_std)
            } else {
                row_mean_invstd(&row, center, scale)
            };
            if scale && inv_std == 0.0 {
                row_idx += 1;
                continue;
            }
            for k in 0..n {
                let mut val = row[k] as f64 - mean;
                if scale {
                    val *= inv_std;
                }
                if val != 0.0 {
                    let z_row = &mut z[k * l..(k + 1) * l];
                    for j in 0..l {
                        z_row[j] += val * y_row[j];
                    }
                }
            }
        }
        row_idx += 1;
    }
    if row_idx * l != y.len() {
        return Err("row count mismatch in A^T*Y".into());
    }
    Ok((z, n))
}

fn compute_a_z_with_iter<I: GenoIter>(
    mut it: I,
    z: &[f64],
    l: usize,
    center: bool,
    scale: bool,
    stats: Option<&[RowStats]>,
) -> Result<(Vec<f64>, usize), String> {
    let n = it.n_samples();
    if z.len() != n * l {
        return Err("Z size mismatch".into());
    }
    let mut y: Vec<f64> = Vec::new();
    let mut row_idx: usize = 0;
    while let Some(row) = it.next_row() {
        let base = y.len();
        y.resize(base + l, 0.0);
        let y_row = &mut y[base..base + l];
        if !center && !scale {
            for k in 0..n {
                let val = row[k] as f64;
                if val != 0.0 {
                    let z_row = &z[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += val * z_row[j];
                    }
                }
            }
        } else {
            let (mean, inv_std) = if let Some(stats) = stats {
                let stat = stats
                    .get(row_idx)
                    .ok_or_else(|| "stats length mismatch in A*Z".to_string())?;
                (stat.mean, stat.inv_std)
            } else {
                row_mean_invstd(&row, center, scale)
            };
            if scale && inv_std == 0.0 {
                row_idx += 1;
                continue;
            }
            for k in 0..n {
                let mut val = row[k] as f64 - mean;
                if scale {
                    val *= inv_std;
                }
                if val != 0.0 {
                    let z_row = &z[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += val * z_row[j];
                    }
                }
            }
        }
        row_idx += 1;
    }
    if let Some(stats) = stats {
        if row_idx != stats.len() {
            return Err("stats length mismatch in A*Z".into());
        }
    }
    Ok((y, n))
}

fn compute_qt_a_with_iter<I: GenoIter>(
    mut it: I,
    q: &[f64],
    l: usize,
    center: bool,
    scale: bool,
    stats: Option<&[RowStats]>,
) -> Result<(Vec<f64>, usize), String> {
    let n = it.n_samples();
    let mut b = vec![0.0f64; l * n];
    let mut row_idx: usize = 0;
    if let Some(stats) = stats {
        if stats.len() * l != q.len() {
            return Err("stats length mismatch in Q^T*A".into());
        }
    }
    while let Some(row) = it.next_row() {
        let q_row = q
            .get(row_idx * l..(row_idx + 1) * l)
            .ok_or_else(|| "row count mismatch in Q^T*A".to_string())?;
        if !center && !scale {
            for k in 0..n {
                let val = row[k] as f64;
                if val != 0.0 {
                    for j in 0..l {
                        b[j * n + k] += q_row[j] * val;
                    }
                }
            }
        } else {
            let (mean, inv_std) = if let Some(stats) = stats {
                let stat = stats
                    .get(row_idx)
                    .ok_or_else(|| "stats length mismatch in Q^T*A".to_string())?;
                (stat.mean, stat.inv_std)
            } else {
                row_mean_invstd(&row, center, scale)
            };
            if scale && inv_std == 0.0 {
                row_idx += 1;
                continue;
            }
            for k in 0..n {
                let mut val = row[k] as f64 - mean;
                if scale {
                    val *= inv_std;
                }
                if val != 0.0 {
                    for j in 0..l {
                        b[j * n + k] += q_row[j] * val;
                    }
                }
            }
        }
        row_idx += 1;
    }
    if row_idx * l != q.len() {
        return Err("row count mismatch in Q^T*A".into());
    }
    Ok((b, n))
}

fn randomized_svd_streaming<I, F>(
    mut make_iter: F,
    k: usize,
    oversample: usize,
    n_iter: usize,
    center: bool,
    scale: bool,
    seed: u64,
    need_u: bool,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize), String>
where
    I: GenoIter,
    F: FnMut() -> Result<I, String>,
{
    if k == 0 {
        return Err("k must be > 0".into());
    }
    let l = k.checked_add(oversample).ok_or("k + oversample overflow")?;

    let iter0 = make_iter()?;
    let n = iter0.n_samples();
    if n == 0 {
        return Err("no samples in genotype input".into());
    }
    if l == 0 || l > n {
        return Err("k + oversample must be in 1..=n_samples".into());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = vec![0.0; n * l];
    for i in 0..n {
        let row = &mut omega[i * l..(i + 1) * l];
        for j in 0..l {
            row[j] = rng.sample(StandardNormal);
        }
    }

    let (mut y, n_y, stats) = compute_y_with_iter(iter0, &omega, l, center, scale)?;
    if n_y != n {
        return Err("sample count mismatch".into());
    }
    if y.len() % l != 0 {
        return Err("Y size mismatch".into());
    }
    let m_eff = y.len() / l;
    if m_eff == 0 {
        return Err("no variants left after QC".into());
    }

    for _ in 0..n_iter {
        let iter_z = make_iter()?;
        let (z, n_z) = compute_at_y_with_iter(iter_z, &y, l, center, scale, stats.as_deref())?;
        if n_z != n {
            return Err("sample count mismatch".into());
        }
        let iter_y = make_iter()?;
        let (y_new, n_y2) = compute_a_z_with_iter(iter_y, &z, l, center, scale, stats.as_deref())?;
        if n_y2 != n {
            return Err("sample count mismatch".into());
        }
        if y_new.len() % l != 0 {
            return Err("Y size mismatch".into());
        }
        let m_new = y_new.len() / l;
        if m_new != m_eff {
            return Err("inconsistent SNP filtering across passes".into());
        }
        y = y_new;
    }

    if l > m_eff {
        return Err("k + oversample exceeds retained variant count".into());
    }
    if k > m_eff.min(n) {
        return Err("k exceeds matrix rank limit".into());
    }

    let y_mat = DMatrix::from_row_slice(m_eff, l, &y);
    let q_full = y_mat.qr().q();
    let q = q_full.columns(0, l).into_owned();
    let q_row = dmatrix_to_rowmajor(&q);

    let iter_b = make_iter()?;
    let (b, n_b) = compute_qt_a_with_iter(iter_b, &q_row, l, center, scale, stats.as_deref())?;
    if n_b != n {
        return Err("sample count mismatch".into());
    }

    let c = gram_b(&b, l, n);
    let c_mat = DMatrix::from_row_slice(l, l, &c);
    let eig = SymmetricEigen::new(c_mat);

    let mut pairs: Vec<(f64, usize)> = eig
        .eigenvalues
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut s = Vec::with_capacity(k);
    let mut u_b = DMatrix::<f64>::zeros(l, k);
    for (col, (val, idx)) in pairs.into_iter().take(k).enumerate() {
        let sigma = val.max(0.0).sqrt();
        s.push(sigma);
        let ucol = eig.eigenvectors.column(idx);
        u_b.set_column(col, &ucol);
    }

    let u_b_row = dmatrix_to_rowmajor(&u_b);
    let u_row = if need_u {
        compute_u_row(&q_row, m_eff, l, &u_b_row, k)
    } else {
        Vec::new()
    };
    let vt = compute_vt_from_b(&u_b_row, k, l, &b, n, &s);

    Ok((u_row, s, vt, m_eff, n))
}

fn randomized_svd_blocked(
    a: &[f64],
    m: usize,
    n: usize,
    k: usize,
    oversample: usize,
    n_iter: usize,
    block_rows: usize,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if m == 0 || n == 0 {
        return Err("input matrix must be non-empty".into());
    }
    let min_dim = m.min(n);
    if k == 0 || k > min_dim {
        return Err(format!("k must be in 1..={min_dim}"));
    }
    let l = k.checked_add(oversample).ok_or("k + oversample overflow")?;
    if l == 0 || l > min_dim {
        return Err("k + oversample must be in 1..=min(m,n)".into());
    }
    if a.len() != m * n {
        return Err("input matrix size mismatch".into());
    }
    let block = if block_rows == 0 { m } else { block_rows.min(m) };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = vec![0.0; n * l];
    for k in 0..n {
        let row = &mut omega[k * l..(k + 1) * l];
        for j in 0..l {
            row[j] = rng.sample(StandardNormal);
        }
    }

    let mut y = matmul_a_b(a, m, n, &omega, l, block);
    for _ in 0..n_iter {
        let z = matmul_at_b(a, m, n, &y, l, block);
        y = matmul_a_b(a, m, n, &z, l, block);
    }

    let y_mat = DMatrix::from_row_slice(m, l, &y);
    let q_full = y_mat.qr().q();
    let q = q_full.columns(0, l).into_owned();
    let q_row = dmatrix_to_rowmajor(&q);

    let b = matmul_qt_a(&q_row, m, l, a, n, block);
    let c = gram_b(&b, l, n);
    let c_mat = DMatrix::from_row_slice(l, l, &c);
    let eig = SymmetricEigen::new(c_mat);

    let mut pairs: Vec<(f64, usize)> = eig
        .eigenvalues
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut s = Vec::with_capacity(k);
    let mut u_b = DMatrix::<f64>::zeros(l, k);
    for (col, (val, idx)) in pairs.into_iter().take(k).enumerate() {
        let sigma = val.max(0.0).sqrt();
        s.push(sigma);
        let ucol = eig.eigenvectors.column(idx);
        u_b.set_column(col, &ucol);
    }

    let u_b_row = dmatrix_to_rowmajor(&u_b);
    let u_row = compute_u_row(&q_row, m, l, &u_b_row, k);
    let vt = compute_vt_from_b(&u_b_row, k, l, &b, n, &s);

    Ok((u_row, s, vt))
}

#[pyfunction]
#[pyo3(signature = (a, k, oversample=10, n_iter=2, block_rows=4096, seed=0))]
/// Blocked randomized SVD for a dense matrix.
/// Returns (U, S, Vt) with shapes (m, k), (k,), (k, n).
/// Non-contiguous inputs are copied into a row-major buffer.
pub fn block_randomized_svd<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    k: usize,
    oversample: usize,
    n_iter: usize,
    block_rows: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let a_arr = a.as_array();
    let (m, n) = (a_arr.shape()[0], a_arr.shape()[1]);
    let a_data: Cow<[f64]> = match a.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(a_arr.iter().cloned().collect()),
    };

    let result = py.allow_threads(|| {
        randomized_svd_blocked(
            a_data.as_ref(),
            m,
            n,
            k,
            oversample,
            n_iter,
            block_rows,
            seed,
        )
    });
    let (u_vec, s_vec, vt_vec) =
        result.map_err(|e| PyRuntimeError::new_err(e))?;

    let u_arr = Array2::from_shape_vec((m, k), u_vec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let vt_arr = Array2::from_shape_vec((k, n), vt_vec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let u_np = u_arr.into_pyarray_bound(py);
    let s_np = s_vec.into_pyarray_bound(py);
    let vt_np = vt_arr.into_pyarray_bound(py);

    Ok((u_np, s_np, vt_np))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    k,
    oversample=10,
    n_iter=2,
    maf=0.0,
    miss=1.0,
    center=true,
    scale=false,
    impute=true,
    seed=0,
    threads=0,
    snp_range=None,
    snp_indices=None,
    bim_range=None,
    sample_ids=None,
    sample_indices=None,
    return_vt=true,
))]
/// Blocked randomized SVD for PLINK bed/bim/fam (SNP-major).
/// Returns (U, S, Vt) for the transposed orientation:
///   U: (n_samples, k), Vt: (k, n_snps_retained).
/// Set return_vt=false to skip SNP-side vectors.
/// Optional SNP/sample selection follows the BED reader rules.
pub fn block_randomized_svd_bed<'py>(
    py: Python<'py>,
    prefix: &str,
    k: usize,
    oversample: usize,
    n_iter: usize,
    maf: f32,
    miss: f32,
    center: bool,
    scale: bool,
    impute: bool,
    seed: u64,
    threads: usize,
    snp_range: Option<(usize, usize)>,
    snp_indices: Option<Vec<usize>>,
    bim_range: Option<(String, i32, i32)>,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
    return_vt: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
)> {
    let center = center || scale;
    let prefix_s = prefix.to_string();
    let meta_it = BedSnpIter::new_with_fill(&prefix_s, maf, miss, impute)
        .map_err(PyRuntimeError::new_err)?;
    let (sample_indices, _sample_ids) = build_sample_selection(
        &meta_it.samples,
        sample_ids,
        sample_indices,
    ).map_err(PyRuntimeError::new_err)?;
    let snp_indices = build_snp_indices(
        &meta_it.sites,
        snp_range,
        snp_indices,
        bim_range,
    ).map_err(PyRuntimeError::new_err)?;
    let full_samples = sample_indices.len() == meta_it.n_samples()
        && sample_indices.iter().enumerate().all(|(i, idx)| *idx == i);

    let sample_indices = Arc::new(sample_indices);
    let snp_indices = snp_indices.map(Arc::new);

    let need_u = return_vt;
    let result = if threads > 1 {
        let it = BedSnpIter::new_with_fill(&prefix_s, maf, miss, impute)
            .map_err(PyRuntimeError::new_err)?;
        py.allow_threads(|| {
            randomized_svd_bed_parallel(
                &it,
                snp_indices.as_deref().map(|v| v.as_slice()),
                sample_indices.as_ref(),
                full_samples,
                k,
                oversample,
                n_iter,
                center,
                scale,
                seed,
                threads,
                need_u,
            )
        })
    } else {
        py.allow_threads(|| {
            randomized_svd_streaming::<BedSnpIterSelect, _>(
                || {
                    let it = BedSnpIter::new_with_fill(&prefix_s, maf, miss, impute)?;
                    Ok(BedSnpIterSelect {
                        it,
                        snp_indices: snp_indices.clone(),
                        snp_pos: 0,
                        sample_indices: sample_indices.clone(),
                        full_samples,
                    })
                },
                k,
                oversample,
                n_iter,
                center,
                scale,
                seed,
                need_u,
            )
        })
    };
    let (u_vec, s_vec, vt_vec, m, n) = result.map_err(PyRuntimeError::new_err)?;
    let u_out = transpose_rowmajor(&vt_vec, k, n);

    let u_arr = Array2::from_shape_vec((n, k), u_out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let u_np = u_arr.into_pyarray_bound(py);
    let s_np = s_vec.into_pyarray_bound(py);
    let vt_np = if return_vt {
        let vt_out = transpose_rowmajor(&u_vec, m, k);
        let vt_arr = Array2::from_shape_vec((k, m), vt_out)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Some(vt_arr.into_pyarray_bound(py))
    } else {
        None
    };

    Ok((u_np, s_np, vt_np))
}
