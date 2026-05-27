use rayon::prelude::*;
use std::any::Any;
use std::sync::Arc;

use crate::bedmath::{
    adaptive_grm_block_rows, decode_standardized_packed_block_rows_f32, is_identity_indices,
    packed_byte_lut,
};
use crate::gfreader::{prepare_bed_2bit_packed_owned_for_stats_samples, PreparedBedPackedOwned};
use crate::he::{build_row_standardization_stats, RowStdStats};
use crate::pcg::pcg_solve_f32_with_guess_into;

const DEFAULT_STANDARDIZE_EPS32: f32 = 1e-12_f32;
const DEFAULT_PACKED_BLOCK_ROWS: usize = 1024;
const DEFAULT_ACTIVE_CACHE_TARGET_MB: usize = 512;
const DEFAULT_ACTIVE_CACHE_MIN_ROWS: usize = 256;
const LASSO_PAR_THRESHOLD: usize = 16_384;
const LASSO_ROW_CHUNK: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LassoSolverKind {
    AdmmPcg,
    ActiveSetCd,
}

#[derive(Clone, Debug)]
pub struct LassoConfig {
    pub lambda: f32,
    pub rho: f32,
    pub max_admm_iter: usize,
    pub max_pcg_iter: usize,
    pub admm_abs_tol: f32,
    pub admm_rel_tol: f32,
    pub pcg_tol: f64,
    pub pcg_tiny: f64,
    pub over_relaxation: f32,
    pub fit_intercept: bool,
    pub penalty_weights: Option<Vec<f32>>,
    pub nz_tol: f32,
    pub max_active_outer_iter: usize,
    pub max_active_cd_sweeps: usize,
    pub active_cd_tol: f32,
    pub active_kkt_tol: f32,
    pub active_cache_target_mb: usize,
    pub active_cache_max_rows: usize,
}

impl Default for LassoConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0_f32,
            rho: 1.0_f32,
            max_admm_iter: 100usize,
            max_pcg_iter: 100usize,
            admm_abs_tol: 1e-4_f32,
            admm_rel_tol: 1e-3_f32,
            pcg_tol: 1e-6_f64,
            pcg_tiny: 1e-20_f64,
            over_relaxation: 1.0_f32,
            fit_intercept: true,
            penalty_weights: None,
            nz_tol: 1e-6_f32,
            max_active_outer_iter: 0usize,
            max_active_cd_sweeps: 0usize,
            active_cd_tol: 0.0_f32,
            active_kkt_tol: 0.0_f32,
            active_cache_target_mb: 0usize,
            active_cache_max_rows: 0usize,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LassoResult {
    pub solver: LassoSolverKind,
    pub beta: Vec<f32>,
    pub intercept: f32,
    pub beta_raw: Vec<f32>,
    pub intercept_raw: f32,
    pub converged: bool,
    pub admm_iters: usize,
    pub pcg_iters_last: usize,
    pub pcg_converged_last: bool,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub objective: f64,
    pub nnz: usize,
    pub rho_effective: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LassoCompareStats {
    pub max_abs_beta: f32,
    pub max_abs_beta_raw: f32,
    pub abs_intercept: f32,
    pub abs_intercept_raw: f32,
    pub abs_objective: f64,
    pub nnz_a: usize,
    pub nnz_b: usize,
    pub support_overlap: usize,
    pub support_union: usize,
    pub support_jaccard: f32,
}

#[derive(Clone, Debug)]
pub struct DenseLassoDesign {
    x: Vec<f32>,
    n_features: usize,
    n_samples: usize,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    pool: Option<Arc<rayon::ThreadPool>>,
}

#[derive(Clone, Debug)]
pub struct PackedBedLassoConfig {
    pub sample_idx: Option<Vec<usize>>,
    pub threads: usize,
    pub block_rows: usize,
    pub use_train_maf: bool,
    pub standardize_eps: f32,
}

impl Default for PackedBedLassoConfig {
    fn default() -> Self {
        Self {
            sample_idx: None,
            threads: 0usize,
            block_rows: 0usize,
            use_train_maf: true,
            standardize_eps: DEFAULT_STANDARDIZE_EPS32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PackedBedLassoDesign {
    packed: Vec<u8>,
    bytes_per_snp: usize,
    n_samples_full: usize,
    n_samples_used: usize,
    n_features: usize,
    sample_idx: Vec<usize>,
    full_sample_fast: bool,
    row_flip: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    block_rows: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
}

pub trait LassoDesignMatrix: Any {
    fn n_features(&self) -> usize;
    fn n_samples(&self) -> usize;
    fn xty(&self, y: &[f32]) -> Result<Vec<f32>, String>;
    fn diag_xtx(&self) -> Result<Vec<f32>, String>;
    fn apply_x_into(&self, beta: &[f32], out: &mut [f32]) -> Result<(), String>;
    fn apply_xt_into(&self, sample_vec: &[f32], out: &mut [f32]) -> Result<(), String>;
    fn row_standardization_stats(&self) -> Option<(&[f32], &[f32])> {
        None
    }

    fn apply_x(&self, beta: &[f32]) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0_f32; self.n_samples()];
        self.apply_x_into(beta, &mut out)?;
        Ok(out)
    }

    fn apply_xt(&self, sample_vec: &[f32]) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0_f32; self.n_features()];
        self.apply_xt_into(sample_vec, &mut out)?;
        Ok(out)
    }

    fn apply_xtx_plus_rho_into(
        &self,
        beta: &[f32],
        rho: f32,
        sample_scratch: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        self.apply_x_into(beta, sample_scratch)?;
        self.apply_xt_into(sample_scratch, out)?;
        for (dst, &bj) in out.iter_mut().zip(beta.iter()) {
            *dst += rho * bj;
        }
        Ok(())
    }

    fn apply_xtx_plus_rho(&self, beta: &[f32], rho: f32) -> Result<Vec<f32>, String> {
        let mut sample_scratch = vec![0.0_f32; self.n_samples()];
        let mut out = vec![0.0_f32; self.n_features()];
        self.apply_xtx_plus_rho_into(beta, rho, &mut sample_scratch, &mut out)?;
        Ok(out)
    }
}

#[inline]
fn build_thread_pool(threads: usize) -> Result<Option<Arc<rayon::ThreadPool>>, String> {
    if threads == 0 {
        return Ok(None);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| format!("rayon pool: {e}"))?;
    Ok(Some(Arc::new(pool)))
}

#[inline]
fn normalize_sample_idx(
    sample_idx: Option<Vec<usize>>,
    n_samples: usize,
) -> Result<Vec<usize>, String> {
    match sample_idx {
        Some(idx) => {
            if idx.is_empty() {
                return Err("sample_idx is empty".to_string());
            }
            let mut seen = vec![false; n_samples];
            for (i, &sid) in idx.iter().enumerate() {
                if sid >= n_samples {
                    return Err(format!(
                        "sample_idx[{i}] out of range: {sid} >= {n_samples}"
                    ));
                }
                if seen[sid] {
                    return Err(format!("sample_idx contains duplicate entry: {sid}"));
                }
                seen[sid] = true;
            }
            Ok(idx)
        }
        None => Ok((0..n_samples).collect()),
    }
}

#[inline]
fn validate_finite_slice(name: &str, x: &[f32]) -> Result<(), String> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(format!("{name} contains non-finite values"));
    }
    Ok(())
}

#[inline]
fn dot_f32_f64(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.len() >= LASSO_PAR_THRESHOLD {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum()
    } else {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum()
    }
}

#[inline]
fn l2_norm_f32(x: &[f32]) -> f64 {
    dot_f32_f64(x, x).sqrt()
}

#[inline]
fn l1_weighted_norm_f32(beta: &[f32], weights: &[f32]) -> f64 {
    beta.iter()
        .zip(weights.iter())
        .map(|(b, w)| (b.abs() as f64) * (*w as f64))
        .sum()
}

#[inline]
fn soft_threshold(v: f32, thr: f32) -> f32 {
    if v > thr {
        v - thr
    } else if v < -thr {
        v + thr
    } else {
        0.0_f32
    }
}

#[inline]
fn effective_active_outer_iter(cfg: &LassoConfig) -> usize {
    if cfg.max_active_outer_iter > 0 {
        cfg.max_active_outer_iter
    } else {
        cfg.max_admm_iter.max(1)
    }
}

#[inline]
fn effective_active_cd_sweeps(cfg: &LassoConfig) -> usize {
    if cfg.max_active_cd_sweeps > 0 {
        cfg.max_active_cd_sweeps
    } else {
        cfg.max_pcg_iter.max(1)
    }
}

#[inline]
fn effective_active_cd_tol(cfg: &LassoConfig) -> f32 {
    if cfg.active_cd_tol > 0.0_f32 {
        cfg.active_cd_tol
    } else {
        cfg.admm_abs_tol.max(1e-6_f32)
    }
}

#[inline]
fn effective_active_kkt_tol(cfg: &LassoConfig) -> f32 {
    if cfg.active_kkt_tol > 0.0_f32 {
        cfg.active_kkt_tol
    } else {
        cfg.admm_abs_tol.max(1e-5_f32)
    }
}

#[inline]
fn effective_active_cache_target_mb(cfg: &LassoConfig) -> usize {
    if cfg.active_cache_target_mb > 0 {
        cfg.active_cache_target_mb
    } else {
        std::env::var("JX_LASSO_ACTIVE_TARGET_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_ACTIVE_CACHE_TARGET_MB)
    }
}

#[inline]
fn effective_active_cache_cap_rows(
    cfg: &LassoConfig,
    n_samples: usize,
    n_features: usize,
) -> usize {
    if cfg.active_cache_max_rows > 0 {
        return cfg.active_cache_max_rows.clamp(1usize, n_features.max(1));
    }
    let row_bytes = n_samples
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1usize);
    let target_mb = effective_active_cache_target_mb(cfg);
    ((target_mb.saturating_mul(1024 * 1024)) / row_bytes)
        .max(DEFAULT_ACTIVE_CACHE_MIN_ROWS)
        .min(n_features.max(1))
}

pub fn compare_lasso_results(
    a: &LassoResult,
    b: &LassoResult,
    nz_tol: f32,
) -> Result<LassoCompareStats, String> {
    if a.beta.len() != b.beta.len() {
        return Err(format!(
            "beta length mismatch: {} vs {}",
            a.beta.len(),
            b.beta.len()
        ));
    }
    if a.beta_raw.len() != b.beta_raw.len() {
        return Err(format!(
            "beta_raw length mismatch: {} vs {}",
            a.beta_raw.len(),
            b.beta_raw.len()
        ));
    }

    let mut max_abs_beta = 0.0_f32;
    let mut max_abs_beta_raw = 0.0_f32;
    let mut nnz_a = 0usize;
    let mut nnz_b = 0usize;
    let mut support_overlap = 0usize;
    let mut support_union = 0usize;

    for j in 0..a.beta.len() {
        max_abs_beta = max_abs_beta.max((a.beta[j] - b.beta[j]).abs());
        let a_nz = a.beta[j].abs() > nz_tol;
        let b_nz = b.beta[j].abs() > nz_tol;
        nnz_a += usize::from(a_nz);
        nnz_b += usize::from(b_nz);
        support_overlap += usize::from(a_nz && b_nz);
        support_union += usize::from(a_nz || b_nz);
    }
    for j in 0..a.beta_raw.len() {
        max_abs_beta_raw = max_abs_beta_raw.max((a.beta_raw[j] - b.beta_raw[j]).abs());
    }

    let support_jaccard = if support_union == 0 {
        1.0_f32
    } else {
        support_overlap as f32 / support_union as f32
    };

    Ok(LassoCompareStats {
        max_abs_beta,
        max_abs_beta_raw,
        abs_intercept: (a.intercept - b.intercept).abs(),
        abs_intercept_raw: (a.intercept_raw - b.intercept_raw).abs(),
        abs_objective: (a.objective - b.objective).abs(),
        nnz_a,
        nnz_b,
        support_overlap,
        support_union,
        support_jaccard,
    })
}

#[inline]
fn standardize_dense_rows_inplace(
    x: &mut [f32],
    n_features: usize,
    n_samples: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut row_mean = vec![0.0_f32; n_features];
    let mut row_inv_sd = vec![0.0_f32; n_features];
    for row_idx in 0..n_features {
        let row = &mut x[row_idx * n_samples..(row_idx + 1) * n_samples];
        let mut obs_n = 0usize;
        let mut obs_sum = 0.0_f64;
        for &v in row.iter() {
            if v.is_finite() {
                obs_n += 1;
                obs_sum += v as f64;
            }
        }
        if obs_n == 0 {
            row.fill(0.0_f32);
            continue;
        }
        let mean = (obs_sum / obs_n as f64) as f32;
        let mut ss = 0.0_f64;
        for &v in row.iter() {
            if v.is_finite() {
                let d = (v - mean) as f64;
                ss += d * d;
            }
        }
        let var = (ss / n_samples.max(1) as f64) as f32;
        row_mean[row_idx] = mean;
        if var > eps {
            let inv_sd = 1.0_f32 / var.sqrt();
            row_inv_sd[row_idx] = inv_sd;
            for cell in row.iter_mut() {
                let centered = if cell.is_finite() {
                    *cell - mean
                } else {
                    0.0_f32
                };
                *cell = centered * inv_sd;
            }
        } else {
            row.fill(0.0_f32);
        }
    }
    (row_mean, row_inv_sd)
}

#[inline]
fn row_major_block_mul_vec_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);
    if rows.saturating_mul(cols) >= LASSO_PAR_THRESHOLD {
        let mut run = || {
            out.par_iter_mut().enumerate().for_each(|(r, dst)| {
                let row = &block[r * cols..(r + 1) * cols];
                let mut acc = 0.0_f32;
                for c in 0..cols {
                    acc += row[c] * vec[c];
                }
                *dst = acc;
            });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }
    for r in 0..rows {
        let row = &block[r * cols..(r + 1) * cols];
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += row[c] * vec[c];
        }
        out[r] = acc;
    }
}

#[inline]
fn row_major_block_t_mul_vec_accum_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);
    if rows.saturating_mul(cols) >= LASSO_PAR_THRESHOLD {
        let chunk_rows = LASSO_ROW_CHUNK.min(rows.max(1));
        let mut run = || {
            let partial = block
                .par_chunks(chunk_rows * cols)
                .zip(vec.par_chunks(chunk_rows))
                .fold(
                    || vec![0.0_f32; cols],
                    |mut acc, (block_chunk, vec_chunk)| {
                        for (r, &vr) in vec_chunk.iter().enumerate() {
                            let row = &block_chunk[r * cols..(r + 1) * cols];
                            for c in 0..cols {
                                acc[c] += row[c] * vr;
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![0.0_f32; cols],
                    |mut left, right| {
                        for c in 0..cols {
                            left[c] += right[c];
                        }
                        left
                    },
                );
            for c in 0..cols {
                out[c] += partial[c];
            }
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }
    for r in 0..rows {
        let vr = vec[r];
        let row = &block[r * cols..(r + 1) * cols];
        for c in 0..cols {
            out[c] += row[c] * vr;
        }
    }
}

#[inline]
fn row_major_row_sumsq_f64(row: &[f32]) -> f64 {
    row.iter()
        .map(|&v| {
            let vf = v as f64;
            vf * vf
        })
        .sum()
}

fn active_cd_sweep_dense_f32(
    active_dense: &[f32],
    n_samples: usize,
    active_beta: &mut [f32],
    active_diag: &[f32],
    active_weights: &[f32],
    residual: &mut [f32],
    lambda: f32,
) -> f32 {
    let mut max_update = 0.0_f32;
    for k in 0..active_beta.len() {
        let row = &active_dense[k * n_samples..(k + 1) * n_samples];
        let bj_old = active_beta[k];
        let corr = dot_f32_f64(row, residual) as f32 + active_diag[k] * bj_old;
        let bj_new = soft_threshold(corr, lambda * active_weights[k]) / active_diag[k];
        let delta = bj_new - bj_old;
        if delta != 0.0_f32 {
            for i in 0..n_samples {
                residual[i] -= row[i] * delta;
            }
            active_beta[k] = bj_new;
            max_update = max_update.max(delta.abs());
        }
    }
    max_update
}

#[allow(clippy::too_many_arguments)]
fn active_cd_sweep_streaming_f32(
    design: &PackedBedLassoDesign,
    active_rows: &[usize],
    active_beta: &mut [f32],
    active_diag: &[f32],
    active_weights: &[f32],
    residual: &mut [f32],
    lambda: f32,
    stream_block_rows: usize,
    stream_block: &mut [f32],
    sel_flip: &mut Vec<bool>,
    sel_mean: &mut Vec<f32>,
    sel_inv_sd: &mut Vec<f32>,
) -> Result<f32, String> {
    let n_samples = design.n_samples();
    let mut max_update = 0.0_f32;
    for chunk_start in (0..active_rows.len()).step_by(stream_block_rows.max(1)) {
        let chunk_end = (chunk_start + stream_block_rows).min(active_rows.len());
        let rows = &active_rows[chunk_start..chunk_end];
        let rows_len = rows.len();
        let block = &mut stream_block[..rows_len * n_samples];
        design
            .decode_selected_rows_into_with_scratch(rows, block, sel_flip, sel_mean, sel_inv_sd)?;
        for local_k in 0..rows_len {
            let k = chunk_start + local_k;
            let row = &block[local_k * n_samples..(local_k + 1) * n_samples];
            let bj_old = active_beta[k];
            let corr = dot_f32_f64(row, residual) as f32 + active_diag[k] * bj_old;
            let bj_new = soft_threshold(corr, lambda * active_weights[k]) / active_diag[k];
            let delta = bj_new - bj_old;
            if delta != 0.0_f32 {
                for i in 0..n_samples {
                    residual[i] -= row[i] * delta;
                }
                active_beta[k] = bj_new;
                max_update = max_update.max(delta.abs());
            }
        }
    }
    Ok(max_update)
}

impl DenseLassoDesign {
    pub fn from_standardized_rows(
        x: Vec<f32>,
        n_features: usize,
        n_samples: usize,
    ) -> Result<Self, String> {
        if n_features == 0 || n_samples == 0 {
            return Err("dense lasso design requires n_features > 0 and n_samples > 0".to_string());
        }
        if x.len() != n_features.saturating_mul(n_samples) {
            return Err(format!(
                "dense lasso shape mismatch: x.len()={} expected {}",
                x.len(),
                n_features.saturating_mul(n_samples)
            ));
        }
        validate_finite_slice("dense design", &x)?;
        Ok(Self {
            x,
            n_features,
            n_samples,
            row_mean: vec![0.0_f32; n_features],
            row_inv_sd: vec![1.0_f32; n_features],
            pool: None,
        })
    }

    pub fn from_raw_rows_standardized(
        mut x: Vec<f32>,
        n_features: usize,
        n_samples: usize,
    ) -> Result<Self, String> {
        if n_features == 0 || n_samples == 0 {
            return Err("dense lasso design requires n_features > 0 and n_samples > 0".to_string());
        }
        if x.len() != n_features.saturating_mul(n_samples) {
            return Err(format!(
                "dense lasso shape mismatch: x.len()={} expected {}",
                x.len(),
                n_features.saturating_mul(n_samples)
            ));
        }
        let (row_mean, row_inv_sd) = standardize_dense_rows_inplace(
            &mut x,
            n_features,
            n_samples,
            DEFAULT_STANDARDIZE_EPS32,
        );
        Ok(Self {
            x,
            n_features,
            n_samples,
            row_mean,
            row_inv_sd,
            pool: None,
        })
    }

    pub fn with_threads(mut self, threads: usize) -> Result<Self, String> {
        self.pool = build_thread_pool(threads)?;
        Ok(self)
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn row_stats(&self) -> (&[f32], &[f32]) {
        (&self.row_mean, &self.row_inv_sd)
    }
}

impl LassoDesignMatrix for DenseLassoDesign {
    fn n_features(&self) -> usize {
        self.n_features
    }

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn xty(&self, y: &[f32]) -> Result<Vec<f32>, String> {
        if y.len() != self.n_samples {
            return Err(format!(
                "dense xty length mismatch: got {}, expected {}",
                y.len(),
                self.n_samples
            ));
        }
        validate_finite_slice("response", y)?;
        let mut out = vec![0.0_f32; self.n_features];
        row_major_block_mul_vec_f32(
            &self.x,
            self.n_features,
            self.n_samples,
            y,
            &mut out,
            self.pool.as_ref(),
        );
        Ok(out)
    }

    fn diag_xtx(&self) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0_f32; self.n_features];
        for (row_idx, dst) in out.iter_mut().enumerate() {
            let row = &self.x[row_idx * self.n_samples..(row_idx + 1) * self.n_samples];
            *dst = row_major_row_sumsq_f64(row) as f32;
        }
        Ok(out)
    }

    fn apply_x_into(&self, beta: &[f32], out: &mut [f32]) -> Result<(), String> {
        if beta.len() != self.n_features {
            return Err(format!(
                "dense apply_x length mismatch: got {}, expected {}",
                beta.len(),
                self.n_features
            ));
        }
        if out.len() != self.n_samples {
            return Err(format!(
                "dense apply_x output length mismatch: got {}, expected {}",
                out.len(),
                self.n_samples
            ));
        }
        validate_finite_slice("beta", beta)?;
        out.fill(0.0_f32);
        row_major_block_t_mul_vec_accum_f32(
            &self.x,
            self.n_features,
            self.n_samples,
            beta,
            out,
            self.pool.as_ref(),
        );
        Ok(())
    }

    fn apply_xt_into(&self, sample_vec: &[f32], out: &mut [f32]) -> Result<(), String> {
        if sample_vec.len() != self.n_samples {
            return Err(format!(
                "dense apply_xt length mismatch: got {}, expected {}",
                sample_vec.len(),
                self.n_samples
            ));
        }
        if out.len() != self.n_features {
            return Err(format!(
                "dense apply_xt output length mismatch: got {}, expected {}",
                out.len(),
                self.n_features
            ));
        }
        validate_finite_slice("sample_vec", sample_vec)?;
        row_major_block_mul_vec_f32(
            &self.x,
            self.n_features,
            self.n_samples,
            sample_vec,
            out,
            self.pool.as_ref(),
        );
        Ok(())
    }

    fn row_standardization_stats(&self) -> Option<(&[f32], &[f32])> {
        Some((&self.row_mean, &self.row_inv_sd))
    }
}

impl PackedBedLassoDesign {
    pub fn from_packed_rows(
        packed: Vec<u8>,
        n_samples_full: usize,
        row_flip: Vec<bool>,
        row_maf: Vec<f32>,
        cfg: PackedBedLassoConfig,
    ) -> Result<Self, String> {
        let PackedBedLassoConfig {
            sample_idx,
            threads,
            block_rows,
            use_train_maf,
            standardize_eps,
        } = cfg;
        if n_samples_full == 0 {
            return Err("packed lasso design requires n_samples_full > 0".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        if bytes_per_snp == 0 || packed.len() % bytes_per_snp != 0 {
            return Err("packed lasso packed length mismatch".to_string());
        }
        let n_features = packed.len() / bytes_per_snp;
        if row_flip.len() != n_features {
            return Err(format!(
                "row_flip length mismatch: got {}, expected {}",
                row_flip.len(),
                n_features
            ));
        }
        if row_maf.len() != n_features {
            return Err(format!(
                "row_maf length mismatch: got {}, expected {}",
                row_maf.len(),
                n_features
            ));
        }
        let sample_idx = normalize_sample_idx(sample_idx, n_samples_full)?;
        let pool = build_thread_pool(threads)?;
        let row_stats = build_row_standardization_stats(
            &packed,
            bytes_per_snp,
            &row_flip,
            &row_maf,
            &sample_idx,
            None,
            standardize_eps.max(0.0_f32),
            use_train_maf,
            pool.as_ref(),
        )?;
        Self::from_parts(
            packed,
            bytes_per_snp,
            n_samples_full,
            sample_idx,
            row_flip,
            row_stats,
            block_rows,
            threads,
            pool,
        )
    }

    pub fn from_bed_prefix(
        prefix: &str,
        maf_threshold: f32,
        max_missing_rate: f32,
        het_threshold: f32,
        snps_only: bool,
        cfg: PackedBedLassoConfig,
    ) -> Result<Self, String> {
        let prepared = prepare_bed_2bit_packed_owned_for_stats_samples(
            prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            cfg.sample_idx.as_deref(),
        )?;
        Self::from_prepared(prepared, cfg)
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples_used
    }

    pub fn row_stats(&self) -> (&[f32], &[f32]) {
        (&self.row_mean, &self.row_inv_sd)
    }

    fn from_prepared(
        prepared: PreparedBedPackedOwned,
        cfg: PackedBedLassoConfig,
    ) -> Result<Self, String> {
        Self::from_packed_rows(
            prepared.packed,
            prepared.n_samples,
            prepared.row_flip,
            prepared.maf,
            cfg,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        packed: Vec<u8>,
        bytes_per_snp: usize,
        n_samples_full: usize,
        sample_idx: Vec<usize>,
        row_flip: Vec<bool>,
        row_stats: RowStdStats,
        requested_block_rows: usize,
        threads: usize,
        pool: Option<Arc<rayon::ThreadPool>>,
    ) -> Result<Self, String> {
        let n_features = row_flip.len();
        if row_stats.row_mean.len() != n_features || row_stats.row_inv_sd.len() != n_features {
            return Err("packed lasso row standardization stats length mismatch".to_string());
        }
        let n_samples_used = sample_idx.len();
        let full_sample_fast = is_identity_indices(&sample_idx, n_samples_full);
        let base_rows = if requested_block_rows == 0 {
            DEFAULT_PACKED_BLOCK_ROWS
        } else {
            requested_block_rows
        };
        let block_rows =
            adaptive_grm_block_rows(base_rows, n_features, n_samples_used, 0usize, threads).max(1);
        Ok(Self {
            packed,
            bytes_per_snp,
            n_samples_full,
            n_samples_used,
            n_features,
            sample_idx,
            full_sample_fast,
            row_flip,
            row_mean: row_stats.row_mean,
            row_inv_sd: row_stats.row_inv_sd,
            block_rows,
            pool,
        })
    }

    fn decode_block(&self, row_start: usize, out: &mut [f32], rows: usize) -> Result<(), String> {
        let code4_lut = &packed_byte_lut().code4;
        decode_standardized_packed_block_rows_f32(
            &self.packed,
            self.bytes_per_snp,
            self.n_samples_full,
            &self.row_flip,
            &self.row_mean,
            &self.row_inv_sd,
            &self.sample_idx,
            self.full_sample_fast,
            None,
            row_start,
            &mut out[..rows * self.n_samples_used],
            code4_lut,
            self.pool.as_ref(),
        )
    }

    fn fill_selected_row_stats(
        &self,
        rows: &[usize],
        sel_flip: &mut Vec<bool>,
        sel_mean: &mut Vec<f32>,
        sel_inv_sd: &mut Vec<f32>,
    ) -> Result<(), String> {
        sel_flip.clear();
        sel_mean.clear();
        sel_inv_sd.clear();
        sel_flip.reserve(rows.len().saturating_sub(sel_flip.capacity()));
        sel_mean.reserve(rows.len().saturating_sub(sel_mean.capacity()));
        sel_inv_sd.reserve(rows.len().saturating_sub(sel_inv_sd.capacity()));
        for &idx in rows {
            if idx >= self.n_features {
                return Err(format!(
                    "selected row index out of bounds: {idx} >= {}",
                    self.n_features
                ));
            }
            sel_flip.push(self.row_flip[idx]);
            sel_mean.push(self.row_mean[idx]);
            sel_inv_sd.push(self.row_inv_sd[idx]);
        }
        Ok(())
    }

    pub(crate) fn decode_selected_rows_into_with_scratch(
        &self,
        rows: &[usize],
        out: &mut [f32],
        sel_flip: &mut Vec<bool>,
        sel_mean: &mut Vec<f32>,
        sel_inv_sd: &mut Vec<f32>,
    ) -> Result<(), String> {
        let expected = rows.len().saturating_mul(self.n_samples_used);
        if out.len() != expected {
            return Err(format!(
                "selected row decode output length mismatch: got {}, expected {}",
                out.len(),
                expected
            ));
        }
        if rows.is_empty() {
            return Ok(());
        }
        self.fill_selected_row_stats(rows, sel_flip, sel_mean, sel_inv_sd)?;
        let code4_lut = &packed_byte_lut().code4;
        decode_standardized_packed_block_rows_f32(
            &self.packed,
            self.bytes_per_snp,
            self.n_samples_full,
            sel_flip,
            sel_mean,
            sel_inv_sd,
            &self.sample_idx,
            self.full_sample_fast,
            Some(rows),
            0,
            out,
            code4_lut,
            self.pool.as_ref(),
        )?;
        Ok(())
    }

    pub(crate) fn decode_selected_rows_into(
        &self,
        rows: &[usize],
        out: &mut [f32],
    ) -> Result<(), String> {
        let mut sel_flip = Vec::with_capacity(rows.len());
        let mut sel_mean = Vec::with_capacity(rows.len());
        let mut sel_inv_sd = Vec::with_capacity(rows.len());
        self.decode_selected_rows_into_with_scratch(
            rows,
            out,
            &mut sel_flip,
            &mut sel_mean,
            &mut sel_inv_sd,
        )
    }

    fn decode_selected_rows(&self, rows: &[usize]) -> Result<Vec<f32>, String> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![0.0_f32; rows.len() * self.n_samples_used];
        self.decode_selected_rows_into(rows, &mut out)?;
        Ok(out)
    }
}

impl LassoDesignMatrix for PackedBedLassoDesign {
    fn n_features(&self) -> usize {
        self.n_features
    }

    fn n_samples(&self) -> usize {
        self.n_samples_used
    }

    fn xty(&self, y: &[f32]) -> Result<Vec<f32>, String> {
        if y.len() != self.n_samples_used {
            return Err(format!(
                "packed xty length mismatch: got {}, expected {}",
                y.len(),
                self.n_samples_used
            ));
        }
        validate_finite_slice("response", y)?;
        let mut out = vec![0.0_f32; self.n_features];
        let mut block = vec![0.0_f32; self.block_rows * self.n_samples_used];
        let mut tmp = vec![0.0_f32; self.block_rows];
        for row_start in (0..self.n_features).step_by(self.block_rows) {
            let row_end = (row_start + self.block_rows).min(self.n_features);
            let rows = row_end - row_start;
            self.decode_block(row_start, &mut block, rows)?;
            row_major_block_mul_vec_f32(
                &block[..rows * self.n_samples_used],
                rows,
                self.n_samples_used,
                y,
                &mut tmp[..rows],
                self.pool.as_ref(),
            );
            out[row_start..row_end].copy_from_slice(&tmp[..rows]);
        }
        Ok(out)
    }

    fn diag_xtx(&self) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0_f32; self.n_features];
        let mut block = vec![0.0_f32; self.block_rows * self.n_samples_used];
        for row_start in (0..self.n_features).step_by(self.block_rows) {
            let row_end = (row_start + self.block_rows).min(self.n_features);
            let rows = row_end - row_start;
            self.decode_block(row_start, &mut block, rows)?;
            for local_row in 0..rows {
                let row =
                    &block[local_row * self.n_samples_used..(local_row + 1) * self.n_samples_used];
                out[row_start + local_row] = row_major_row_sumsq_f64(row) as f32;
            }
        }
        Ok(out)
    }

    fn apply_x_into(&self, beta: &[f32], out: &mut [f32]) -> Result<(), String> {
        if beta.len() != self.n_features {
            return Err(format!(
                "packed apply_x length mismatch: got {}, expected {}",
                beta.len(),
                self.n_features
            ));
        }
        if out.len() != self.n_samples_used {
            return Err(format!(
                "packed apply_x output length mismatch: got {}, expected {}",
                out.len(),
                self.n_samples_used
            ));
        }
        validate_finite_slice("beta", beta)?;
        out.fill(0.0_f32);
        let mut block = vec![0.0_f32; self.block_rows * self.n_samples_used];
        for row_start in (0..self.n_features).step_by(self.block_rows) {
            let row_end = (row_start + self.block_rows).min(self.n_features);
            let rows = row_end - row_start;
            self.decode_block(row_start, &mut block, rows)?;
            row_major_block_t_mul_vec_accum_f32(
                &block[..rows * self.n_samples_used],
                rows,
                self.n_samples_used,
                &beta[row_start..row_end],
                out,
                self.pool.as_ref(),
            );
        }
        Ok(())
    }

    fn apply_xt_into(&self, sample_vec: &[f32], out: &mut [f32]) -> Result<(), String> {
        if sample_vec.len() != self.n_samples_used {
            return Err(format!(
                "packed apply_xt length mismatch: got {}, expected {}",
                sample_vec.len(),
                self.n_samples_used
            ));
        }
        if out.len() != self.n_features {
            return Err(format!(
                "packed apply_xt output length mismatch: got {}, expected {}",
                out.len(),
                self.n_features
            ));
        }
        validate_finite_slice("sample_vec", sample_vec)?;
        let mut block = vec![0.0_f32; self.block_rows * self.n_samples_used];
        let mut tmp = vec![0.0_f32; self.block_rows];
        for row_start in (0..self.n_features).step_by(self.block_rows) {
            let row_end = (row_start + self.block_rows).min(self.n_features);
            let rows = row_end - row_start;
            self.decode_block(row_start, &mut block, rows)?;
            row_major_block_mul_vec_f32(
                &block[..rows * self.n_samples_used],
                rows,
                self.n_samples_used,
                sample_vec,
                &mut tmp[..rows],
                self.pool.as_ref(),
            );
            out[row_start..row_end].copy_from_slice(&tmp[..rows]);
        }
        Ok(())
    }

    fn row_standardization_stats(&self) -> Option<(&[f32], &[f32])> {
        Some((&self.row_mean, &self.row_inv_sd))
    }
}

pub fn fit_lasso_f32<D: LassoDesignMatrix>(
    design: &D,
    y: &[f32],
    cfg: &LassoConfig,
) -> Result<LassoResult, String> {
    if let Some(packed) = (design as &dyn Any).downcast_ref::<PackedBedLassoDesign>() {
        return fit_lasso_packed_active_f32(packed, y, cfg);
    }
    fit_lasso_f32_reference(design, y, cfg)
}

pub fn fit_lasso_f32_reference<D: LassoDesignMatrix>(
    design: &D,
    y: &[f32],
    cfg: &LassoConfig,
) -> Result<LassoResult, String> {
    let n_samples = design.n_samples();
    let n_features = design.n_features();
    if n_samples == 0 || n_features == 0 {
        return Err("lasso requires n_samples > 0 and n_features > 0".to_string());
    }
    if y.len() != n_samples {
        return Err(format!(
            "response length mismatch: got {}, expected {}",
            y.len(),
            n_samples
        ));
    }
    validate_finite_slice("response", y)?;
    if !cfg.lambda.is_finite() || cfg.lambda < 0.0_f32 {
        return Err("lasso lambda must be finite and >= 0".to_string());
    }
    if !cfg.rho.is_finite() || cfg.rho <= 0.0_f32 {
        return Err("lasso rho must be finite and > 0".to_string());
    }
    if cfg.max_admm_iter == 0 || cfg.max_pcg_iter == 0 {
        return Err("lasso iteration limits must be > 0".to_string());
    }
    if !cfg.admm_abs_tol.is_finite() || cfg.admm_abs_tol < 0.0_f32 {
        return Err("lasso admm_abs_tol must be finite and >= 0".to_string());
    }
    if !cfg.admm_rel_tol.is_finite() || cfg.admm_rel_tol < 0.0_f32 {
        return Err("lasso admm_rel_tol must be finite and >= 0".to_string());
    }
    if !cfg.pcg_tol.is_finite() || cfg.pcg_tol < 0.0_f64 {
        return Err("lasso pcg_tol must be finite and >= 0".to_string());
    }
    if !cfg.pcg_tiny.is_finite() || cfg.pcg_tiny <= 0.0_f64 {
        return Err("lasso pcg_tiny must be finite and > 0".to_string());
    }
    if !cfg.over_relaxation.is_finite()
        || cfg.over_relaxation <= 0.0_f32
        || cfg.over_relaxation >= 2.0_f32
    {
        return Err("lasso over_relaxation must be within (0, 2)".to_string());
    }
    let penalty_weights = if let Some(weights) = cfg.penalty_weights.as_ref() {
        if weights.len() != n_features {
            return Err(format!(
                "penalty_weights length mismatch: got {}, expected {}",
                weights.len(),
                n_features
            ));
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0_f32) {
            return Err("penalty_weights must be finite and >= 0".to_string());
        }
        weights.clone()
    } else {
        vec![1.0_f32; n_features]
    };

    let intercept = if cfg.fit_intercept {
        (y.iter().map(|&v| v as f64).sum::<f64>() / n_samples as f64) as f32
    } else {
        0.0_f32
    };
    let rho_effective = cfg.rho * ((n_samples.max(1) as f32) / 10.0_f32);
    let mut y_centered = vec![0.0_f32; n_samples];
    for (dst, &src) in y_centered.iter_mut().zip(y.iter()) {
        *dst = if cfg.fit_intercept {
            src - intercept
        } else {
            src
        };
    }

    let xty = design.xty(&y_centered)?;
    let diag_xtx = design.diag_xtx()?;
    let mut diag_inv = vec![0.0_f32; n_features];
    for j in 0..n_features {
        let denom = (diag_xtx[j] + rho_effective).max(cfg.pcg_tiny as f32);
        diag_inv[j] = 1.0_f32 / denom;
    }

    let mut beta = vec![0.0_f32; n_features];
    for j in 0..n_features {
        beta[j] = xty[j] * diag_inv[j];
    }
    let mut z = vec![0.0_f32; n_features];
    for j in 0..n_features {
        z[j] = soft_threshold(beta[j], cfg.lambda * penalty_weights[j] / rho_effective);
    }
    let mut u = vec![0.0_f32; n_features];
    for j in 0..n_features {
        u[j] = beta[j] - z[j];
    }

    let mut converged = false;
    let mut primal_residual = 1.0_f64;
    let mut dual_residual = f64::INFINITY;
    let mut pcg_iters_last = 0usize;
    let mut pcg_converged_last = false;
    let mut admm_iters = 0usize;
    let sqrt_p = (n_features as f64).sqrt();
    let mut rhs = vec![0.0_f32; n_features];
    let mut x_hat = vec![0.0_f32; n_features];
    let mut sample_scratch = vec![0.0_f32; n_samples];
    let mut z_prev = vec![0.0_f32; n_features];
    let mut beta_minus_z = vec![0.0_f32; n_features];
    let mut z_delta = vec![0.0_f32; n_features];

    for iter in 0..cfg.max_admm_iter {
        z_prev.copy_from_slice(&z);
        for j in 0..n_features {
            rhs[j] = xty[j] + rho_effective * (z[j] - u[j]);
        }
        let pcg_tol_cap = 1e-2_f64.max(cfg.pcg_tol);
        let pcg_tol_iter = cfg.pcg_tol.max(0.1_f64 * primal_residual).min(pcg_tol_cap);
        let pcg = pcg_solve_f32_with_guess_into(
            &rhs,
            Some(&beta),
            cfg.max_pcg_iter,
            pcg_tol_iter,
            cfg.pcg_tiny,
            |p, out| design.apply_xtx_plus_rho_into(p, rho_effective, &mut sample_scratch, out),
            |r, out| {
                for j in 0..n_features {
                    out[j] = r[j] * diag_inv[j];
                }
            },
            |_iter_done, _iter_total, _rel_res| Ok(()),
        )?;
        beta = pcg.x;
        pcg_iters_last = pcg.iters;
        pcg_converged_last = pcg.converged;

        let alpha = cfg.over_relaxation;
        for j in 0..n_features {
            x_hat[j] = alpha * beta[j] + (1.0_f32 - alpha) * z_prev[j];
            z[j] = soft_threshold(
                x_hat[j] + u[j],
                cfg.lambda * penalty_weights[j] / rho_effective,
            );
        }
        for j in 0..n_features {
            u[j] += x_hat[j] - z[j];
        }

        for j in 0..n_features {
            beta_minus_z[j] = x_hat[j] - z[j];
            z_delta[j] = z[j] - z_prev[j];
        }
        primal_residual = l2_norm_f32(&beta_minus_z);
        dual_residual = (rho_effective as f64) * l2_norm_f32(&z_delta);
        let eps_primal = sqrt_p * cfg.admm_abs_tol as f64
            + (cfg.admm_rel_tol as f64) * l2_norm_f32(&x_hat).max(l2_norm_f32(&z));
        let eps_dual = sqrt_p * cfg.admm_abs_tol as f64
            + (cfg.admm_rel_tol as f64) * (rho_effective as f64) * l2_norm_f32(&u);
        admm_iters = iter + 1;
        if primal_residual <= eps_primal && dual_residual <= eps_dual {
            converged = true;
            break;
        }
    }

    design.apply_x_into(&z, &mut sample_scratch)?;
    let mut rss = 0.0_f64;
    for (&pred, &obs) in sample_scratch.iter().zip(y_centered.iter()) {
        let d = pred as f64 - obs as f64;
        rss += d * d;
    }
    let objective =
        0.5_f64 * rss + (cfg.lambda as f64) * l1_weighted_norm_f32(&z, &penalty_weights);
    let nnz = z.iter().filter(|&&v| v.abs() > cfg.nz_tol).count();
    let (beta_raw, intercept_raw) =
        if let Some((row_mean, row_inv_sd)) = design.row_standardization_stats() {
            let mut beta_raw = vec![0.0_f32; n_features];
            let mut intercept_raw = intercept;
            for j in 0..n_features {
                if row_inv_sd[j] > 0.0_f32 {
                    beta_raw[j] = z[j] * row_inv_sd[j];
                    intercept_raw -= beta_raw[j] * row_mean[j];
                }
            }
            (beta_raw, intercept_raw)
        } else {
            (z.clone(), intercept)
        };

    Ok(LassoResult {
        solver: LassoSolverKind::AdmmPcg,
        beta: z,
        intercept,
        beta_raw,
        intercept_raw,
        converged,
        admm_iters,
        pcg_iters_last,
        pcg_converged_last,
        primal_residual,
        dual_residual,
        objective,
        nnz,
        rho_effective,
    })
}

pub fn fit_lasso_packed_active_f32(
    design: &PackedBedLassoDesign,
    y: &[f32],
    cfg: &LassoConfig,
) -> Result<LassoResult, String> {
    let n_samples = design.n_samples();
    let n_features = design.n_features();
    if n_samples == 0 || n_features == 0 {
        return Err("lasso requires n_samples > 0 and n_features > 0".to_string());
    }
    if y.len() != n_samples {
        return Err(format!(
            "response length mismatch: got {}, expected {}",
            y.len(),
            n_samples
        ));
    }
    validate_finite_slice("response", y)?;
    if !cfg.lambda.is_finite() || cfg.lambda < 0.0_f32 {
        return Err("lasso lambda must be finite and >= 0".to_string());
    }
    if let Some(weights) = cfg.penalty_weights.as_ref() {
        if weights.len() != n_features {
            return Err(format!(
                "penalty_weights length mismatch: got {}, expected {}",
                weights.len(),
                n_features
            ));
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0_f32) {
            return Err("penalty_weights must be finite and >= 0".to_string());
        }
    }
    if !cfg.active_cd_tol.is_finite() || cfg.active_cd_tol < 0.0_f32 {
        return Err("lasso active_cd_tol must be finite and >= 0".to_string());
    }
    if !cfg.active_kkt_tol.is_finite() || cfg.active_kkt_tol < 0.0_f32 {
        return Err("lasso active_kkt_tol must be finite and >= 0".to_string());
    }

    let penalty_weights = if let Some(weights) = cfg.penalty_weights.as_ref() {
        weights.clone()
    } else {
        vec![1.0_f32; n_features]
    };

    let intercept = if cfg.fit_intercept {
        (y.iter().map(|&v| v as f64).sum::<f64>() / n_samples as f64) as f32
    } else {
        0.0_f32
    };
    let mut y_centered = vec![0.0_f32; n_samples];
    for (dst, &src) in y_centered.iter_mut().zip(y.iter()) {
        *dst = if cfg.fit_intercept {
            src - intercept
        } else {
            src
        };
    }

    let xty = design.xty(&y_centered)?;
    let diag_xtx = design.diag_xtx()?;
    let mut active_rows = Vec::<usize>::new();
    let mut active_mask = vec![false; n_features];
    let mut active_diag = Vec::<f32>::new();
    let mut active_weights = Vec::<f32>::new();
    let mut active_beta = Vec::<f32>::new();
    let screen_eps = cfg.admm_abs_tol.max(1e-6_f32);
    let mut lambda_support = 0usize;
    for j in 0..n_features {
        let thr = cfg.lambda * penalty_weights[j];
        if xty[j].abs() > thr + screen_eps {
            active_mask[j] = true;
            active_rows.push(j);
            active_diag.push(diag_xtx[j].max(cfg.pcg_tiny as f32));
            active_weights.push(penalty_weights[j]);
            active_beta.push(0.0_f32);
            lambda_support += 1;
        }
    }

    if lambda_support == 0 {
        let beta = vec![0.0_f32; n_features];
        return Ok(LassoResult {
            solver: LassoSolverKind::ActiveSetCd,
            beta: beta.clone(),
            intercept,
            beta_raw: beta,
            intercept_raw: intercept,
            converged: true,
            admm_iters: 0,
            pcg_iters_last: 0,
            pcg_converged_last: true,
            primal_residual: 0.0_f64,
            dual_residual: 0.0_f64,
            objective: 0.5_f64 * dot_f32_f64(&y_centered, &y_centered) + 0.0_f64,
            nnz: 0,
            rho_effective: 0.0_f32,
        });
    }

    let active_cap = effective_active_cache_cap_rows(cfg, n_samples, n_features);
    let stream_block_rows = design.block_rows.max(1).min(active_cap.max(1));
    let mut active_dense = if active_rows.len() <= active_cap {
        Some(design.decode_selected_rows(&active_rows)?)
    } else {
        None
    };
    let mut residual = y_centered.clone();
    let mut grad = vec![0.0_f32; n_features];
    let mut stream_block = vec![0.0_f32; stream_block_rows * n_samples];
    let mut stream_sel_flip = Vec::with_capacity(stream_block_rows);
    let mut stream_sel_mean = Vec::with_capacity(stream_block_rows);
    let mut stream_sel_inv_sd = Vec::with_capacity(stream_block_rows);
    let mut converged = false;
    let mut outer_iters = 0usize;
    let mut inner_sweeps_last = 0usize;
    let max_outer = effective_active_outer_iter(cfg).max(1);
    let max_sweeps = effective_active_cd_sweeps(cfg).max(1);
    let cd_tol = effective_active_cd_tol(cfg);
    let kkt_tol = effective_active_kkt_tol(cfg);
    let mut max_update_last = f32::INFINITY;

    for outer in 0..max_outer {
        inner_sweeps_last = 0;
        for _ in 0..max_sweeps {
            let max_update = if let Some(active_dense) = active_dense.as_ref() {
                active_cd_sweep_dense_f32(
                    active_dense,
                    n_samples,
                    &mut active_beta,
                    &active_diag,
                    &active_weights,
                    &mut residual,
                    cfg.lambda,
                )
            } else {
                active_cd_sweep_streaming_f32(
                    design,
                    &active_rows,
                    &mut active_beta,
                    &active_diag,
                    &active_weights,
                    &mut residual,
                    cfg.lambda,
                    stream_block_rows,
                    &mut stream_block,
                    &mut stream_sel_flip,
                    &mut stream_sel_mean,
                    &mut stream_sel_inv_sd,
                )?
            };
            inner_sweeps_last += 1;
            max_update_last = max_update;
            if max_update <= cd_tol {
                break;
            }
        }

        grad.fill(0.0_f32);
        design.apply_xt_into(&residual, &mut grad)?;
        let mut violators = Vec::<usize>::new();
        for j in 0..n_features {
            if !active_mask[j] {
                let thr = cfg.lambda * penalty_weights[j] + kkt_tol;
                if grad[j].abs() > thr {
                    violators.push(j);
                }
            }
        }

        outer_iters = outer + 1;
        if violators.is_empty() {
            converged = true;
            break;
        }

        match active_dense.as_mut() {
            Some(active_dense_buf) if active_rows.len() + violators.len() <= active_cap => {
                let new_dense = design.decode_selected_rows(&violators)?;
                active_dense_buf.extend_from_slice(&new_dense);
            }
            Some(_) => {
                active_dense = None;
            }
            None => {}
        }
        for &j in &violators {
            active_mask[j] = true;
            active_rows.push(j);
            active_diag.push(diag_xtx[j].max(cfg.pcg_tiny as f32));
            active_weights.push(penalty_weights[j]);
            active_beta.push(0.0_f32);
        }
    }

    let mut beta = vec![0.0_f32; n_features];
    for (k, &j) in active_rows.iter().enumerate() {
        beta[j] = active_beta[k];
    }

    let mut fitted = vec![0.0_f32; n_samples];
    design.apply_x_into(&beta, &mut fitted)?;
    let mut rss = 0.0_f64;
    for (&pred, &obs) in fitted.iter().zip(y_centered.iter()) {
        let d = pred as f64 - obs as f64;
        rss += d * d;
    }
    let objective =
        0.5_f64 * rss + (cfg.lambda as f64) * l1_weighted_norm_f32(&beta, &penalty_weights);
    let nnz = beta.iter().filter(|&&v| v.abs() > cfg.nz_tol).count();
    let (beta_raw, intercept_raw) =
        if let Some((row_mean, row_inv_sd)) = design.row_standardization_stats() {
            let mut beta_raw = vec![0.0_f32; n_features];
            let mut intercept_raw = intercept;
            for j in 0..n_features {
                if row_inv_sd[j] > 0.0_f32 {
                    beta_raw[j] = beta[j] * row_inv_sd[j];
                    intercept_raw -= beta_raw[j] * row_mean[j];
                }
            }
            (beta_raw, intercept_raw)
        } else {
            (beta.clone(), intercept)
        };

    Ok(LassoResult {
        solver: LassoSolverKind::ActiveSetCd,
        beta,
        intercept,
        beta_raw,
        intercept_raw,
        converged,
        admm_iters: outer_iters,
        pcg_iters_last: inner_sweeps_last,
        pcg_converged_last: converged,
        primal_residual: max_update_last as f64,
        dual_residual: 0.0_f64,
        objective,
        nnz,
        rho_effective: 0.0_f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_codes(codes: &[u8], n_samples: usize) -> Vec<u8> {
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut out = vec![0u8; bytes_per_snp];
        for (sid, &code) in codes.iter().enumerate() {
            out[sid >> 2] |= code << ((sid & 3) * 2);
        }
        out
    }

    fn pack_dosages_02(dosages: &[u8], n_samples: usize) -> Vec<u8> {
        let mut codes = Vec::with_capacity(dosages.len());
        for &g in dosages {
            codes.push(match g {
                0 => 0b00,
                2 => 0b11,
                _ => panic!("only 0/2 dosages supported in this test helper"),
            });
        }
        pack_codes(&codes, n_samples)
    }

    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    fn synthetic_streaming_case(
        n_features: usize,
        n_samples: usize,
        n_signal: usize,
    ) -> (PackedBedLassoDesign, Vec<f32>) {
        let mut packed = Vec::new();
        let mut row_maf = Vec::with_capacity(n_features);
        let row_flip = vec![false; n_features];
        let mut x_std = vec![0.0_f32; n_features * n_samples];

        for j in 0..n_features {
            let mut dosages = vec![0u8; n_samples];
            let threshold = 3_u64 + (j % 4) as u64;
            for (i, g) in dosages.iter_mut().enumerate() {
                let h = splitmix64(((j as u64 + 1) << 32) ^ (i as u64 + 17));
                *g = if h % 11 < threshold { 2u8 } else { 0u8 };
            }
            dosages[j % n_samples] = 0u8;
            dosages[(j * 7 + 11) % n_samples] = 2u8;

            let mean = dosages.iter().map(|&g| g as f32).sum::<f32>() / n_samples as f32;
            let var = dosages
                .iter()
                .map(|&g| {
                    let d = g as f32 - mean;
                    d * d
                })
                .sum::<f32>()
                / n_samples as f32;
            let inv_sd = 1.0_f32 / var.sqrt().max(1e-6_f32);
            let maf = (0.5_f32 * mean).min(1.0_f32 - 0.5_f32 * mean);
            row_maf.push(maf);

            for i in 0..n_samples {
                x_std[j * n_samples + i] = (dosages[i] as f32 - mean) * inv_sd;
            }
            packed.extend_from_slice(&pack_dosages_02(&dosages, n_samples));
        }

        let mut y = vec![0.0_f32; n_samples];
        for j in 0..n_signal.min(n_features) {
            let weight = 0.02_f32 * (1 + (j % 5) as i32) as f32;
            let row = &x_std[j * n_samples..(j + 1) * n_samples];
            for i in 0..n_samples {
                y[i] += weight * row[i];
            }
        }
        for (i, yi) in y.iter_mut().enumerate() {
            *yi += (((i * 17 + 5) % 23) as f32 - 11.0_f32) * 1e-3_f32;
        }

        let design = PackedBedLassoDesign::from_packed_rows(
            packed,
            n_samples,
            row_flip,
            row_maf,
            PackedBedLassoConfig {
                use_train_maf: false,
                ..PackedBedLassoConfig::default()
            },
        )
        .unwrap();
        (design, y)
    }

    #[test]
    fn dense_lasso_matches_orthogonal_solution() {
        let x = vec![
            -1.0_f32, -1.0_f32, 1.0_f32, 1.0_f32, -1.0_f32, 1.0_f32, -1.0_f32, 1.0_f32,
        ];
        let design = DenseLassoDesign::from_standardized_rows(x, 2, 4).unwrap();
        let y = vec![-3.5_f32, -2.5_f32, 2.5_f32, 3.5_f32];
        let cfg = LassoConfig {
            lambda: 1.0_f32,
            rho: 1.0_f32,
            max_admm_iter: 200usize,
            max_pcg_iter: 100usize,
            ..LassoConfig::default()
        };
        let res = fit_lasso_f32(&design, &y, &cfg).unwrap();
        assert!(res.converged);
        assert_eq!(res.solver, LassoSolverKind::AdmmPcg);
        assert!((res.beta[0] - 2.75_f32).abs() < 1e-3_f32);
        assert!((res.beta[1] - 0.25_f32).abs() < 1e-3_f32);
    }

    #[test]
    fn dense_lasso_exposes_raw_scale_coefficients() {
        let x = vec![0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32];
        let design = DenseLassoDesign::from_raw_rows_standardized(x, 1, 4).unwrap();
        let y = vec![0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32];
        let cfg = LassoConfig {
            lambda: 0.0_f32,
            rho: 1.0_f32,
            max_admm_iter: 200usize,
            max_pcg_iter: 100usize,
            ..LassoConfig::default()
        };
        let res = fit_lasso_f32(&design, &y, &cfg).unwrap();
        assert_eq!(res.solver, LassoSolverKind::AdmmPcg);
        assert!((res.beta_raw[0] - 1.0_f32).abs() < 1e-3_f32);
        assert!(res.intercept_raw.abs() < 1e-3_f32);
    }

    #[test]
    fn packed_lasso_matches_dense_small_case() {
        let n_samples = 4usize;
        let row1 = pack_codes(&[0b00, 0b11, 0b00, 0b11], n_samples);
        let row2 = pack_codes(&[0b00, 0b00, 0b11, 0b11], n_samples);
        let mut packed = Vec::new();
        packed.extend_from_slice(&row1);
        packed.extend_from_slice(&row2);
        let packed_design = PackedBedLassoDesign::from_packed_rows(
            packed,
            n_samples,
            vec![false, false],
            vec![0.5_f32, 0.5_f32],
            PackedBedLassoConfig {
                use_train_maf: false,
                ..PackedBedLassoConfig::default()
            },
        )
        .unwrap();

        let y = vec![-3.5_f32, 2.5_f32, -2.5_f32, 3.5_f32];
        let cfg = LassoConfig {
            lambda: 1.0_f32,
            rho: 1.0_f32,
            max_admm_iter: 200usize,
            max_pcg_iter: 100usize,
            ..LassoConfig::default()
        };
        let ref_res = fit_lasso_f32_reference(&packed_design, &y, &cfg).unwrap();
        let fast_res = fit_lasso_f32(&packed_design, &y, &cfg).unwrap();

        assert_eq!(ref_res.solver, LassoSolverKind::AdmmPcg);
        assert_eq!(fast_res.solver, LassoSolverKind::ActiveSetCd);
        assert!(fast_res.converged);
        assert!((ref_res.beta[0] - fast_res.beta[0]).abs() < 1e-3_f32);
        assert!((ref_res.beta[1] - fast_res.beta[1]).abs() < 1e-3_f32);
    }

    #[test]
    fn packed_lasso_streaming_matches_reference() {
        let (packed_design, y) = synthetic_streaming_case(96, 256, 80);
        let cfg = LassoConfig {
            lambda: 0.5_f32,
            rho: 1.0_f32,
            max_admm_iter: 250usize,
            max_pcg_iter: 80usize,
            admm_abs_tol: 1e-5_f32,
            admm_rel_tol: 1e-4_f32,
            active_cd_tol: 1e-5_f32,
            active_kkt_tol: 5e-5_f32,
            active_cache_max_rows: 64usize,
            ..LassoConfig::default()
        };

        let ref_res = fit_lasso_f32_reference(&packed_design, &y, &cfg).unwrap();
        let fast_res = fit_lasso_f32(&packed_design, &y, &cfg).unwrap();
        let cmp = compare_lasso_results(&ref_res, &fast_res, 1e-4_f32).unwrap();
        let mut ref_fit = vec![0.0_f32; packed_design.n_samples()];
        let mut fast_fit = vec![0.0_f32; packed_design.n_samples()];
        packed_design
            .apply_x_into(&ref_res.beta, &mut ref_fit)
            .unwrap();
        packed_design
            .apply_x_into(&fast_res.beta, &mut fast_fit)
            .unwrap();
        let max_fit_diff = ref_fit
            .iter()
            .zip(fast_fit.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        eprintln!(
            "lasso compare: beta={:.3e} beta_raw={:.3e} obj={:.3e} fit={:.3e} jaccard={:.3}",
            cmp.max_abs_beta,
            cmp.max_abs_beta_raw,
            cmp.abs_objective,
            max_fit_diff,
            cmp.support_jaccard
        );

        assert_eq!(ref_res.solver, LassoSolverKind::AdmmPcg);
        assert_eq!(fast_res.solver, LassoSolverKind::ActiveSetCd);
        assert!(fast_res.converged, "{cmp:?}");
        assert!(cmp.max_abs_beta < 1e-2_f32, "{cmp:?}");
        assert!(cmp.max_abs_beta_raw < 1e-2_f32, "{cmp:?}");
        assert!(cmp.abs_objective < 5e-3_f64, "{cmp:?}");
        assert!(cmp.support_jaccard > 0.95_f32, "{cmp:?}");
        assert!(
            max_fit_diff < 5e-3_f32,
            "max_fit_diff={max_fit_diff}, {cmp:?}"
        );
    }
}
