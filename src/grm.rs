use nalgebra::{DMatrix, SymmetricEigen};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::gfcore::{BedSnpIter, VcfSnpIter};

#[derive(Clone, Copy)]
pub enum InputKind<'a> {
    Bed { prefix: &'a str },
    Vcf { path: &'a str },
}

fn pass1_den(kind: InputKind, maf: f32, miss: f32) -> Result<(f64, usize), String> {
    let mut den: f64 = 0.0;
    let mut m_eff: usize = 0;

    match kind {
        InputKind::Bed { prefix } => {
            let mut it = BedSnpIter::new(prefix, maf, miss)?;
            while let Some((row, _site)) = it.next_snp() {
                let mean_g = row.iter().map(|&x| x as f64).sum::<f64>() / row.len() as f64;
                let p = (mean_g / 2.0).clamp(0.0, 1.0);
                den += 2.0 * p * (1.0 - p);
                m_eff += 1;
            }
        }
        InputKind::Vcf { path } => {
            let mut it = VcfSnpIter::new(path, maf, miss)?;
            while let Some((row, _site)) = it.next_snp() {
                let mean_g = row.iter().map(|&x| x as f64).sum::<f64>() / row.len() as f64;
                let p = (mean_g / 2.0).clamp(0.0, 1.0);
                den += 2.0 * p * (1.0 - p);
                m_eff += 1;
            }
        }
    }

    if den <= 0.0 || m_eff == 0 {
        return Err("No variants left after QC, den=0".into());
    }
    Ok((den, m_eff))
}

/// 单线程：Y = (Z^T (Z X)) / den
fn matmul_g_serial(kind: InputKind, maf: f32, miss: f32, den: f64, x: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let n = x.nrows();
    let l = x.ncols();
    let mut y = DMatrix::<f64>::zeros(n, l);

    match kind {
        InputKind::Bed { prefix } => {
            let mut it = BedSnpIter::new(prefix, maf, miss)?;
            while let Some((row_f32, _)) = it.next_snp() {
                let mean_g = row_f32.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

                let mut t = vec![0.0f64; l];
                for i in 0..n {
                    let zi = row_f32[i] as f64 - mean_g;
                    if zi != 0.0 {
                        let xi = x.row(i);
                        for j in 0..l { t[j] += zi * xi[j]; }
                    }
                }
                for i in 0..n {
                    let zi = row_f32[i] as f64 - mean_g;
                    if zi != 0.0 {
                        for j in 0..l { y[(i, j)] += zi * t[j]; }
                    }
                }
            }
        }
        InputKind::Vcf { path } => {
            let mut it = VcfSnpIter::new(path, maf, miss)?;
            while let Some((row_f32, _)) = it.next_snp() {
                let mean_g = row_f32.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

                let mut t = vec![0.0f64; l];
                for i in 0..n {
                    let zi = row_f32[i] as f64 - mean_g;
                    if zi != 0.0 {
                        let xi = x.row(i);
                        for j in 0..l { t[j] += zi * xi[j]; }
                    }
                }
                for i in 0..n {
                    let zi = row_f32[i] as f64 - mean_g;
                    if zi != 0.0 {
                        for j in 0..l { y[(i, j)] += zi * t[j]; }
                    }
                }
            }
        }
    }

    Ok(y.scale(1.0 / den))
}

/// BED 多线程：并行按 snp_idx 遍历（需要 BedSnpIter::get_snp_row）
fn matmul_g_bed_parallel(prefix: &str, maf: f32, miss: f32, den: f64, x: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let it = BedSnpIter::new(prefix, maf, miss)?;
    let n = x.nrows();
    let l = x.ncols();
    let n_snps = it.n_snps();

    // 每线程局部 y_local，再 reduce
    let y = (0..n_snps)
        .into_par_iter()
        .fold(
            || DMatrix::<f64>::zeros(n, l),
            |mut y_local, snp_idx| {
                if let Some((row_f32, _)) = it.get_snp_row(snp_idx) {
                    let mean_g = row_f32.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

                    let mut t = vec![0.0f64; l];
                    for i in 0..n {
                        let zi = row_f32[i] as f64 - mean_g;
                        if zi != 0.0 {
                            let xi = x.row(i);
                            for j in 0..l { t[j] += zi * xi[j]; }
                        }
                    }

                    for i in 0..n {
                        let zi = row_f32[i] as f64 - mean_g;
                        if zi != 0.0 {
                            for j in 0..l { y_local[(i, j)] += zi * t[j]; }
                        }
                    }
                }
                y_local
            }
        )
        .reduce(
            || DMatrix::<f64>::zeros(n, l),
            |mut a, b| { a += b; a }
        );

    Ok(y.scale(1.0 / den))
}

/// B = Q^T G Q：串行版本（VCF 用）
fn small_b_serial(kind: InputKind, maf: f32, miss: f32, den: f64, q: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let n = q.nrows();
    let l = q.ncols();
    let mut b = DMatrix::<f64>::zeros(l, l);

    match kind {
        InputKind::Bed { prefix } => {
            let mut it = BedSnpIter::new(prefix, maf, miss)?;
            while let Some((row_f32, _)) = it.next_snp() {
                let mean_g = row_f32.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

                let mut u = vec![0.0f64; l];
                for i in 0..n {
                    let zi = row_f32[i] as f64 - mean_g;
                    if zi != 0.0 {
                        let qi = q.row(i);
                        for j in 0..l { u[j] += zi * qi[j]; }
                    }
                }
                for a in 0..l {
                    for c in 0..l {
                        b[(a, c)] += u[a] * u[c];
                    }
                }
            }
        }
        InputKind::Vcf { path } => {
            let mut it = VcfSnpIter::new(path, maf, miss)?;
            while let Some((row_f32, _)) = it.next_snp() {
                let mean_g = row_f32.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

                let mut u = vec![0.0f64; l];
                for i in 0..n {
                    let zi = row_f32[i] as f64 - mean_g;
                    if zi != 0.0 {
                        let qi = q.row(i);
                        for j in 0..l { u[j] += zi * qi[j]; }
                    }
                }
                for a in 0..l {
                    for c in 0..l {
                        b[(a, c)] += u[a] * u[c];
                    }
                }
            }
        }
    }

    Ok(b.scale(1.0 / den))
}

/// BED 并行 small_b（更快）
fn small_b_bed_parallel(prefix: &str, maf: f32, miss: f32, den: f64, q: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let it = BedSnpIter::new(prefix, maf, miss)?;
    let n = q.nrows();
    let l = q.ncols();
    let n_snps = it.n_snps();

    let b = (0..n_snps)
        .into_par_iter()
        .fold(
            || DMatrix::<f64>::zeros(l, l),
            |mut b_local, snp_idx| {
                if let Some((row_f32, _)) = it.get_snp_row(snp_idx) {
                    let mean_g = row_f32.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

                    let mut u = vec![0.0f64; l];
                    for i in 0..n {
                        let zi = row_f32[i] as f64 - mean_g;
                        if zi != 0.0 {
                            let qi = q.row(i);
                            for j in 0..l { u[j] += zi * qi[j]; }
                        }
                    }
                    for a in 0..l {
                        for c in 0..l {
                            b_local[(a, c)] += u[a] * u[c];
                        }
                    }
                }
                b_local
            }
        )
        .reduce(
            || DMatrix::<f64>::zeros(l, l),
            |mut a, b| { a += b; a }
        );

    Ok(b.scale(1.0 / den))
}

/// Randomized PCA of the GRM
pub fn randomized_grm_pca(
    kind: InputKind,
    k: usize,
    oversample: usize,
    n_iter: usize,
    maf: f32,
    miss: f32,
    seed: u64,
    threads: usize,   // <-- 新增：线程数
) -> Result<(Vec<f64>, DMatrix<f64>), String> {
    let (den, _m_eff) = pass1_den(kind, maf, miss)?;

    let n = match kind {
        InputKind::Bed { prefix } => BedSnpIter::new(prefix, maf, miss)?.n_samples(),
        InputKind::Vcf { path } => VcfSnpIter::new(path, maf, miss)?.n_samples(),
    };

    let l = k + oversample;
    if l == 0 || l > n {
        return Err("Invalid k/oversample".into());
    }

    // 线程池（局部，不污染全局）
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| e.to_string())?;

    // Omega: n×l （修复 StandardNormal）
    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = DMatrix::<f64>::zeros(n, l);
    for i in 0..n {
        for j in 0..l {
            omega[(i, j)] = rng.sample(StandardNormal);
        }
    }

    let y = pool.install(|| -> Result<DMatrix<f64>, String> {
        // BED 用并行版本；VCF 用串行版本（线程池主要用于后续可能并行的计算）
        let mut yy = match kind {
            InputKind::Bed { prefix } => matmul_g_bed_parallel(prefix, maf, miss, den, &omega)?,
            InputKind::Vcf { .. } => matmul_g_serial(kind, maf, miss, den, &omega)?,
        };

        for _ in 0..n_iter {
            yy = match kind {
                InputKind::Bed { prefix } => matmul_g_bed_parallel(prefix, maf, miss, den, &yy)?,
                InputKind::Vcf { .. } => matmul_g_serial(kind, maf, miss, den, &yy)?,
            };
        }
        Ok(yy)
    })?;

    // QR
    let qr = y.clone().qr();
    let q = qr.q();

    // B = Q^T G Q（BED 并行）
    let b = pool.install(|| -> Result<DMatrix<f64>, String> {
        match kind {
            InputKind::Bed { prefix } => small_b_bed_parallel(prefix, maf, miss, den, &q),
            InputKind::Vcf { .. } => small_b_serial(kind, maf, miss, den, &q),
        }
    })?;

    // eig(B)
    let eig = SymmetricEigen::new(b);
    let mut pairs: Vec<(f64, usize)> = eig.eigenvalues.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let take = &pairs[pairs.len() - k..];
    let mut evals = Vec::with_capacity(k);
    let mut u_top = DMatrix::<f64>::zeros(l, k);
    for (col_out, (val, idx)) in take.iter().enumerate() {
        evals.push(*val);
        let ucol = eig.eigenvectors.column(*idx);
        u_top.set_column(col_out, &ucol);
    }

    let evecs = q * u_top; // n×k
    Ok((evals, evecs))
}

fn nalgebra_to_ndarray_rowmajor(mat: &DMatrix<f64>) -> Array2<f64> {
    let n = mat.nrows();
    let m = mat.ncols();
    let mut out = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            out[i * m + j] = mat[(i, j)];
        }
    }
    Array2::from_shape_vec((n, m), out).unwrap()
}

#[pyfunction]
pub fn grm_pca_bed<'py>(
    py: Python<'py>,
    prefix: &str,
    k: usize,
    oversample: usize,
    n_iter: usize,
    maf: f32,
    miss: f32,
    seed: u64,
    threads: usize, // <-- 新增
) -> PyResult<(pyo3::Bound<'py, PyArray1<f64>>, pyo3::Bound<'py, PyArray2<f64>>)> {
    let (evals, evecs) = randomized_grm_pca(
        InputKind::Bed { prefix },
        k, oversample, n_iter, maf, miss, seed, threads
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // 修复弃用：into_pyarray_bound
    let evals_np = evals.into_pyarray_bound(py);

    // 修复 PyArray2::from_shape_vec：用 ndarray -> numpy
    let arr2 = nalgebra_to_ndarray_rowmajor(&evecs);
    let evecs_np = arr2.into_pyarray_bound(py);

    Ok((evals_np, evecs_np))
}

#[pyfunction]
pub fn grm_pca_vcf<'py>(
    py: Python<'py>,
    path: &str,
    k: usize,
    oversample: usize,
    n_iter: usize,
    maf: f32,
    miss: f32,
    seed: u64,
    threads: usize, // <-- 新增（VCF 目前主要是将来扩展；gzip 解析仍偏串行）
) -> PyResult<(pyo3::Bound<'py, PyArray1<f64>>, pyo3::Bound<'py, PyArray2<f64>>)> {
    let (evals, evecs) = randomized_grm_pca(
        InputKind::Vcf { path },
        k, oversample, n_iter, maf, miss, seed, threads
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let evals_np = evals.into_pyarray_bound(py);

    let arr2 = nalgebra_to_ndarray_rowmajor(&evecs);
    let evecs_np = arr2.into_pyarray_bound(py);

    Ok((evals_np, evecs_np))
}
