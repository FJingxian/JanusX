use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use crate::bedmath::{decode_standardized_packed_block_f32, is_identity_indices, packed_byte_lut};
use crate::blas::{
    cblas_sgemm_dispatch, CblasInt, OpenBlasThreadGuard, CBLAS_COL_MAJOR, CBLAS_NO_TRANS,
    CBLAS_TRANS, rust_sgemm_prefers_rayon_rowmajor_f32_kernel,
};
use crate::linalg::{cholesky_inplace, cholesky_solve_into};
use crate::packed::bed_packed_row_flip_mask;
use crate::stats_common::{env_truthy, get_cached_pool, map_err_string_to_py, parse_index_vec_i64};

pub const HE_BOUNDARY_INTERIOR: u8 = 0;
pub const HE_BOUNDARY_SIGMA_G_ZERO: u8 = 1;
pub const HE_BOUNDARY_SIGMA_E_ZERO: u8 = 2;
pub const HE_BOUNDARY_ORIGIN: u8 = 3;

#[derive(Clone, Debug)]
pub struct HePcgResult {
    pub sigma_g2: f64,
    pub sigma_e2: f64,
    pub h2: f64,
    pub lambda: f64,
    pub converged: bool,
    pub iters: usize,
    pub rel_res: f64,
    pub m_effective: usize,
    pub tr_k: f64,
    pub tr_p: f64,
    pub tr_k2: f64,
    pub tr_k2_solve: f64,
    pub y_ky: f64,
    pub y_y: f64,
    pub nnls_projected: bool,
    pub boundary_status: u8,
    pub exact_trace_used: bool,
}

#[derive(Clone, Debug)]
pub struct RowStdStats {
    pub row_mean: Vec<f32>,
    pub row_inv_sd: Vec<f32>,
    pub m_effective: usize,
}

#[derive(Clone, Debug)]
struct CovariateProjector {
    n: usize,
    p: usize,
    x: Vec<f64>,
    chol_xtx: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ProjectionWorkspace {
    xtv: Vec<f64>,
    beta: Vec<f64>,
}

impl ProjectionWorkspace {
    fn new(p: usize) -> Self {
        Self {
            xtv: vec![0.0_f64; p],
            beta: vec![0.0_f64; p],
        }
    }
}

#[derive(Clone, Debug)]
struct GrmApplyWorkspace {
    block: Vec<f32>,
    tmp: Vec<f32>,
}

impl GrmApplyWorkspace {
    fn new() -> Self {
        Self {
            block: Vec::new(),
            tmp: Vec::new(),
        }
    }

    fn ensure(&mut self, row_step: usize, n_out: usize, rhs_cols: usize) {
        let need_block = row_step.saturating_mul(n_out);
        if self.block.len() < need_block {
            self.block.resize(need_block, 0.0_f32);
        }
        let need_tmp = row_step.saturating_mul(rhs_cols);
        if self.tmp.len() < need_tmp {
            self.tmp.resize(need_tmp, 0.0_f32);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct HeApplyTiming {
    decode_secs: f64,
    xmul_secs: f64,
    xtmul_secs: f64,
}

impl HeApplyTiming {
    #[inline]
    fn gemm_secs(&self) -> f64 {
        self.xmul_secs + self.xtmul_secs
    }
}

const SMALL_RHS_PAR_ROW_BLOCK: usize = 64;

fn emit_he_stage_timing(
    label: &str,
    batch_done: usize,
    batch_total: usize,
    train_maf_secs: f64,
    decode_secs: f64,
    xmul_secs: f64,
    xtmul_secs: f64,
    trace_secs: f64,
    total_secs: f64,
) {
    let total = total_secs.max(1e-12_f64);
    let other_secs =
        (total_secs - train_maf_secs - decode_secs - xmul_secs - xtmul_secs - trace_secs)
            .max(0.0);
    let pct = |x: f64| -> f64 { (x * 100.0) / total };
    eprintln!(
        "HE stage timing {} batch={}/{} train_maf={:.3}s ({:.1}%) decode={:.3}s ({:.1}%) xmul={:.3}s ({:.1}%) xtmul={:.3}s ({:.1}%) trace={:.3}s ({:.1}%) other={:.3}s ({:.1}%) total={:.3}s",
        label,
        batch_done,
        batch_total,
        train_maf_secs,
        pct(train_maf_secs),
        decode_secs,
        pct(decode_secs),
        xmul_secs,
        pct(xmul_secs),
        xtmul_secs,
        pct(xtmul_secs),
        trace_secs,
        pct(trace_secs),
        other_secs,
        pct(other_secs),
        total_secs,
    );
}

impl CovariateProjector {
    fn from_optional_covariates(
        y_len: usize,
        x_cov: Option<&[f64]>,
        p_cov: usize,
    ) -> Result<Self, String> {
        let n = y_len;
        if n == 0 {
            return Err("CovariateProjector requires n > 0".to_string());
        }
        if let Some(x) = x_cov {
            if p_cov == 0 {
                return Err("x_cov provided but p_cov == 0".to_string());
            }
            if x.len() != n.saturating_mul(p_cov) {
                return Err(format!(
                    "x_cov length mismatch: got {}, expected {}",
                    x.len(),
                    n.saturating_mul(p_cov)
                ));
            }
            if x.iter().any(|v| !v.is_finite()) {
                return Err("x_cov contains non-finite values".to_string());
            }
        } else if p_cov != 0 {
            return Err("p_cov > 0 but x_cov is None".to_string());
        }

        let p = 1usize.saturating_add(p_cov);
        if n <= p {
            return Err(format!(
                "HE projection requires n > rank(X): n={n}, p(with intercept)={p}"
            ));
        }

        let mut x = vec![0.0_f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0_f64;
        }
        if let Some(x_cov_flat) = x_cov {
            for i in 0..n {
                let src = &x_cov_flat[i * p_cov..(i + 1) * p_cov];
                let dst = &mut x[i * p + 1..(i + 1) * p];
                dst.copy_from_slice(src);
            }
        }

        let mut xtx = vec![0.0_f64; p * p];
        for i in 0..n {
            let row = &x[i * p..(i + 1) * p];
            for a in 0..p {
                let va = row[a];
                for b in 0..=a {
                    xtx[a * p + b] += va * row[b];
                }
            }
        }
        for a in 0..p {
            for b in 0..a {
                xtx[b * p + a] = xtx[a * p + b];
            }
        }

        cholesky_inplace(&mut xtx, p).ok_or_else(|| {
            format!("HE covariate projection failed: X'X is singular or ill-conditioned (p={p})")
        })?;

        Ok(Self {
            n,
            p,
            x,
            chol_xtx: xtx,
        })
    }

    #[inline]
    fn tr_p(&self) -> f64 {
        ((self.n as f64) - (self.p as f64)).max(1.0_f64)
    }

    fn project_vec_f64_in_place(
        &self,
        v: &mut [f64],
        ws: &mut ProjectionWorkspace,
    ) -> Result<(), String> {
        if v.len() != self.n {
            return Err(format!(
                "project_vec_f64_in_place length mismatch: got {}, expected {}",
                v.len(),
                self.n
            ));
        }
        ws.xtv.fill(0.0_f64);
        for i in 0..self.n {
            let vi = v[i];
            let row = &self.x[i * self.p..(i + 1) * self.p];
            for (a, &x_ia) in row.iter().enumerate() {
                ws.xtv[a] += x_ia * vi;
            }
        }
        cholesky_solve_into(&self.chol_xtx, self.p, &ws.xtv, &mut ws.beta);
        for i in 0..self.n {
            let row = &self.x[i * self.p..(i + 1) * self.p];
            let mut xb = 0.0_f64;
            for (a, &x_ia) in row.iter().enumerate() {
                xb += x_ia * ws.beta[a];
            }
            v[i] -= xb;
        }
        Ok(())
    }

    fn project_mat_f32_in_place(
        &self,
        mat: &mut [f32],
        n_cols: usize,
        ws: &mut ProjectionWorkspace,
    ) -> Result<(), String> {
        if mat.len() != self.n.saturating_mul(n_cols) {
            return Err(format!(
                "project_mat_f32_in_place length mismatch: got {}, expected {}",
                mat.len(),
                self.n.saturating_mul(n_cols)
            ));
        }
        if n_cols == 0 {
            return Ok(());
        }
        for c in 0..n_cols {
            ws.xtv.fill(0.0_f64);
            for i in 0..self.n {
                let vi = mat[i * n_cols + c] as f64;
                let row = &self.x[i * self.p..(i + 1) * self.p];
                for (a, &x_ia) in row.iter().enumerate() {
                    ws.xtv[a] += x_ia * vi;
                }
            }
            cholesky_solve_into(&self.chol_xtx, self.p, &ws.xtv, &mut ws.beta);
            for i in 0..self.n {
                let row = &self.x[i * self.p..(i + 1) * self.p];
                let mut xb = 0.0_f64;
                for (a, &x_ia) in row.iter().enumerate() {
                    xb += x_ia * ws.beta[a];
                }
                let idx = i * n_cols + c;
                mat[idx] -= xb as f32;
            }
        }
        Ok(())
    }
}

#[inline]
fn dot_f32_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

#[inline]
fn oriented_minor_freq_from_input(raw_freq: f32, flip: bool) -> Result<f32, String> {
    if !raw_freq.is_finite() {
        return Err("row_maf contains non-finite values".to_string());
    }
    let af = raw_freq.clamp(0.0_f32, 1.0_f32);
    // Backward-compatible interpretation:
    // - <=0.5 is assumed to already be MAF.
    // - >0.5 is treated as allele-frequency input; when flip=true we convert to
    //   minor-allele frequency, otherwise keep orientation consistent with decode.
    let p = if af <= 0.5_f32 {
        af
    } else if flip {
        1.0_f32 - af
    } else {
        af
    };
    Ok(p.clamp(0.0_f32, 1.0_f32))
}

#[inline]
fn prefer_parallel_small_rhs(
    rows: usize,
    cols: usize,
    n_rhs: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> bool {
    if n_rhs == 0 {
        return false;
    }
    let Some(tp) = pool else {
        return false;
    };
    if tp.current_num_threads() <= 1 {
        return false;
    }
    n_rhs <= 16
        && rows.saturating_mul(cols).saturating_mul(n_rhs) >= 1_048_576
        && rust_sgemm_prefers_rayon_rowmajor_f32_kernel()
}

pub fn build_row_standardization_stats(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    std_eps32: f32,
    use_train_maf: bool,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<RowStdStats, String> {
    let m = row_flip.len();
    if row_maf.len() != m {
        return Err("build_row_standardization_stats: length mismatch".to_string());
    }
    if packed_flat.len() != m.saturating_mul(bytes_per_snp) {
        return Err("build_row_standardization_stats: packed length mismatch".to_string());
    }
    let mut row_mean = vec![0.0_f32; m];
    let mut row_inv_sd = vec![0.0_f32; m];
    let compute_row = |j: usize, mean_slot: &mut f32, inv_sd_slot: &mut f32| -> Result<usize, String> {
        let flip = row_flip[j];
        let mut p = oriented_minor_freq_from_input(row_maf[j], flip)?;
        if use_train_maf {
            let row = &packed_flat[j * bytes_per_snp..(j + 1) * bytes_per_snp];
            let mut non_missing = 0usize;
            let mut alt_sum = 0usize;
            for &sid in sample_idx {
                let code = (row[sid >> 2] >> ((sid & 3) * 2)) & 0b11;
                match code {
                    0b00 => {
                        non_missing += 1;
                    }
                    0b10 => {
                        non_missing += 1;
                        alt_sum += 1;
                    }
                    0b11 => {
                        non_missing += 1;
                        alt_sum += 2;
                    }
                    _ => {}
                }
            }
            if non_missing > 0 {
                let dosage_sum = if flip {
                    2usize.saturating_mul(non_missing).saturating_sub(alt_sum)
                } else {
                    alt_sum
                };
                p = (dosage_sum as f32) / (2.0_f32 * non_missing as f32);
            }
        }

        let p = p.clamp(0.0_f32, 1.0_f32);
        let mean = 2.0_f32 * p;
        let var = (2.0_f32 * p * (1.0_f32 - p)).max(0.0_f32);
        *mean_slot = mean;
        if var > std_eps32 {
            *inv_sd_slot = 1.0_f32 / var.sqrt();
            Ok(1usize)
        } else {
            *inv_sd_slot = 0.0_f32;
            Ok(0usize)
        }
    };
    let m_effective = if let Some(tp) = pool {
        tp.install(|| {
            row_mean
                .par_iter_mut()
                .zip(row_inv_sd.par_iter_mut())
                .enumerate()
                .map(|(j, (mean_slot, inv_sd_slot))| compute_row(j, mean_slot, inv_sd_slot))
                .try_reduce(|| 0usize, |acc, v| Ok(acc + v))
        })?
    } else {
        let mut acc = 0usize;
        for j in 0..m {
            acc += compute_row(j, &mut row_mean[j], &mut row_inv_sd[j])?;
        }
        acc
    };
    Ok(RowStdStats {
        row_mean,
        row_inv_sd,
        m_effective,
    })
}

#[inline]
fn he_residual_sq_2x2(a00: f64, a01: f64, a11: f64, b0: f64, b1: f64, x0: f64, x1: f64) -> f64 {
    let r0 = a00.mul_add(x0, a01 * x1) - b0;
    let r1 = a01.mul_add(x0, a11 * x1) - b1;
    r0 * r0 + r1 * r1
}

#[inline]
fn he_project_nnls_2x2(
    a00: f64,
    a01: f64,
    a11: f64,
    b0: f64,
    b1: f64,
    x0_unconstrained: f64,
    x1_unconstrained: f64,
) -> (f64, f64, bool, u8) {
    let mut best = (0.0_f64, 0.0_f64, f64::INFINITY, HE_BOUNDARY_ORIGIN);
    let mut consider = |x0: f64, x1: f64, status: u8| {
        if !x0.is_finite() || !x1.is_finite() {
            return;
        }
        if x0 < 0.0_f64 || x1 < 0.0_f64 {
            return;
        }
        let obj = he_residual_sq_2x2(a00, a01, a11, b0, b1, x0, x1);
        if obj.is_finite() && obj < best.2 {
            best = (x0, x1, obj, status);
        }
    };

    // Candidate A: unconstrained HE solution (if already feasible).
    consider(x0_unconstrained, x1_unconstrained, HE_BOUNDARY_INTERIOR);

    // Candidate B: sigma_g2 = 0 boundary, solve least-squares for sigma_e2 >= 0.
    let col1_norm2 = a01 * a01 + a11 * a11;
    if col1_norm2.is_finite() && col1_norm2 > 0.0_f64 {
        let x1 = ((a01 * b0 + a11 * b1) / col1_norm2).max(0.0_f64);
        consider(0.0_f64, x1, HE_BOUNDARY_SIGMA_G_ZERO);
    }

    // Candidate C: sigma_e2 = 0 boundary, solve least-squares for sigma_g2 >= 0.
    let col0_norm2 = a00 * a00 + a01 * a01;
    if col0_norm2.is_finite() && col0_norm2 > 0.0_f64 {
        let x0 = ((a00 * b0 + a01 * b1) / col0_norm2).max(0.0_f64);
        consider(x0, 0.0_f64, HE_BOUNDARY_SIGMA_E_ZERO);
    }

    // Candidate D: origin.
    consider(0.0_f64, 0.0_f64, HE_BOUNDARY_ORIGIN);

    if best.2.is_finite() {
        let scale = x0_unconstrained
            .abs()
            .max(x1_unconstrained.abs())
            .max(1.0_f64);
        let proj_tol = 1e-10_f64 * scale;
        let projected = best.3 != HE_BOUNDARY_INTERIOR
            || (best.0 - x0_unconstrained).abs() > proj_tol
            || (best.1 - x1_unconstrained).abs() > proj_tol;
        (best.0, best.1, projected, best.3)
    } else {
        (0.0_f64, 0.0_f64, true, HE_BOUNDARY_ORIGIN)
    }
}

#[inline]
fn he_solve_2x2(a00: f64, a01: f64, a11: f64, b0: f64, b1: f64) -> Result<(f64, f64), String> {
    if !a00.is_finite()
        || !a01.is_finite()
        || !a11.is_finite()
        || !b0.is_finite()
        || !b1.is_finite()
    {
        return Err("HE 2x2 solve received non-finite inputs".to_string());
    }
    let det = a00.mul_add(a11, -(a01 * a01));
    let det_scale = (a00.abs() + a11.abs() + 2.0_f64 * a01.abs()).max(1.0_f64);
    let det_floor = det_scale * det_scale * f64::EPSILON;
    if !det.is_finite() || det.abs() <= det_floor {
        return Err(format!(
            "HE 2x2 solve is singular/ill-conditioned: det={det}, floor={det_floor}"
        ));
    }
    let x0 = (b0.mul_add(a11, -(b1 * a01))) / det;
    let x1 = (a00.mul_add(b1, -(a01 * b0))) / det;
    if !x0.is_finite() || !x1.is_finite() {
        return Err("HE 2x2 solve produced non-finite outputs".to_string());
    }
    Ok((x0, x1))
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn row_major_block_mul_mat_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    rhs: &[f32], // row-major (cols, n_rhs)
    n_rhs: usize,
    out: &mut [f32], // row-major (rows, n_rhs)
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(rhs.len(), cols.saturating_mul(n_rhs));
    debug_assert_eq!(out.len(), rows.saturating_mul(n_rhs));
    if n_rhs == 0 {
        return;
    }
    if prefer_parallel_small_rhs(rows, cols, n_rhs, pool) {
        let row_block = SMALL_RHS_PAR_ROW_BLOCK.max(1);
        let mut run = || {
            out.par_chunks_mut(n_rhs * row_block)
                .enumerate()
                .for_each(|(chunk_id, out_chunk)| {
                    out_chunk.fill(0.0_f32);
                    let row_start = chunk_id * row_block;
                    let rows_here = out_chunk.len() / n_rhs;
                    for local_r in 0..rows_here {
                        let r = row_start + local_r;
                        let out_row = &mut out_chunk[local_r * n_rhs..(local_r + 1) * n_rhs];
                        let row = &block[r * cols..(r + 1) * cols];
                        for c in 0..cols {
                            let a = row[c];
                            if a == 0.0_f32 {
                                continue;
                            }
                            let rhs_row = &rhs[c * n_rhs..(c + 1) * n_rhs];
                            for t in 0..n_rhs {
                                out_row[t] += a * rhs_row[t];
                            }
                        }
                    }                    
                });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        // Row-major C = A * B is equivalent to column-major C^T = B^T * A^T.
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            n_rhs as CblasInt,
            rows as CblasInt,
            cols as CblasInt,
            1.0,
            rhs.as_ptr(),
            n_rhs as CblasInt,
            block.as_ptr(),
            cols as CblasInt,
            0.0,
            out.as_mut_ptr(),
            n_rhs as CblasInt,
        );
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let row_block = SMALL_RHS_PAR_ROW_BLOCK.max(1);
        out.par_chunks_mut(n_rhs * row_block)
            .enumerate()
            .for_each(|(chunk_id, out_chunk)| {
                out_chunk.fill(0.0_f32);
                let row_start = chunk_id * row_block;
                let rows_here = out_chunk.len() / n_rhs;
                for local_r in 0..rows_here {
                    let r = row_start + local_r;
                    let out_row = &mut out_chunk[local_r * n_rhs..(local_r + 1) * n_rhs];
                    let row = &block[r * cols..(r + 1) * cols];
                    for c in 0..cols {
                        let a = row[c];
                        if a == 0.0_f32 {
                            continue;
                        }
                        let rhs_row = &rhs[c * n_rhs..(c + 1) * n_rhs];
                        for t in 0..n_rhs {
                            out_row[t] += a * rhs_row[t];
                        }
                    }
                }
            });
    }
}

#[inline]
fn row_major_block_t_mul_mat_accum_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    rhs: &[f32], // row-major (rows, n_rhs)
    n_rhs: usize,
    out: &mut [f32], // row-major (cols, n_rhs)
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(rhs.len(), rows.saturating_mul(n_rhs));
    debug_assert_eq!(out.len(), cols.saturating_mul(n_rhs));
    if n_rhs == 0 {
        return;
    }
    if prefer_parallel_small_rhs(rows, cols, n_rhs, pool) {
        let col_block = SMALL_RHS_PAR_ROW_BLOCK.max(1);
        let mut run = || {
            out.par_chunks_mut(n_rhs * col_block)
                .enumerate()
                .for_each(|(chunk_id, out_chunk)| {
                    let col_start = chunk_id * col_block;
                    let cols_here = out_chunk.len() / n_rhs;
                    for local_c in 0..cols_here {
                        let c = col_start + local_c;
                        let out_row = &mut out_chunk[local_c * n_rhs..(local_c + 1) * n_rhs];
                        for r in 0..rows {
                            let a = block[r * cols + c];
                            if a == 0.0_f32 {
                                continue;
                            }
                            let rhs_row = &rhs[r * n_rhs..(r + 1) * n_rhs];
                            for t in 0..n_rhs {
                                out_row[t] += a * rhs_row[t];
                            }
                        }
                    }
                });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        // Row-major C += A^T * B  <=>  column-major C^T += B^T * A.
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            n_rhs as CblasInt,
            cols as CblasInt,
            rows as CblasInt,
            1.0,
            rhs.as_ptr(),
            n_rhs as CblasInt,
            block.as_ptr(),
            cols as CblasInt,
            1.0,
            out.as_mut_ptr(),
            n_rhs as CblasInt,
        );
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let col_block = SMALL_RHS_PAR_ROW_BLOCK.max(1);
        out.par_chunks_mut(n_rhs * col_block)
            .enumerate()
            .for_each(|(chunk_id, out_chunk)| {
                let col_start = chunk_id * col_block;
                let cols_here = out_chunk.len() / n_rhs;
                for local_c in 0..cols_here {
                    let c = col_start + local_c;
                    let out_row = &mut out_chunk[local_c * n_rhs..(local_c + 1) * n_rhs];
                    for r in 0..rows {
                        let a = block[r * cols + c];
                        if a == 0.0_f32 {
                            continue;
                        }
                        let rhs_row = &rhs[r * n_rhs..(r + 1) * n_rhs];
                        for t in 0..n_rhs {
                            out_row[t] += a * rhs_row[t];
                        }
                    }
                }
            });
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_standardized_block_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    row_start: usize,
    out: &mut [f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    decode_standardized_packed_block_f32(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_idx,
        full_sample_fast,
        row_start,
        out,
        code4_lut,
        pool,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_grm_to_mat_f32_with_workspace(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    block_rows: usize,
    rhs: &[f32], // row-major (n_out, n_rhs)
    n_rhs: usize,
    m_scale: f32,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
    workspace: &mut GrmApplyWorkspace,
    timing: Option<&mut HeApplyTiming>,
    out: &mut [f32], // row-major (n_out, n_rhs)
) -> Result<(), String> {
    let m = row_flip.len();
    let n_out = sample_idx.len();
    if rhs.len() != n_out.saturating_mul(n_rhs) {
        return Err(format!(
            "apply_grm_to_mat_f32_with_workspace RHS length mismatch: got {}, expected {}",
            rhs.len(),
            n_out.saturating_mul(n_rhs)
        ));
    }
    if out.len() != n_out.saturating_mul(n_rhs) {
        return Err(format!(
            "apply_grm_to_mat_f32_with_workspace out length mismatch: got {}, expected {}",
            out.len(),
            n_out.saturating_mul(n_rhs)
        ));
    }
    out.fill(0.0_f32);
    if n_rhs == 0 {
        return Ok(());
    }
    if m == 0 || n_out == 0 {
        return Ok(());
    }
    let row_step = block_rows.max(1).min(m.max(1));
    workspace.ensure(row_step, n_out, n_rhs);
    let mut decode_acc = 0.0_f64;
    let mut xmul_acc = 0.0_f64;
    let mut xtmul_acc = 0.0_f64;

    for st in (0..m).step_by(row_step) {
        let ed = (st + row_step).min(m);
        let cur_rows = ed - st;
        let blk_slice = &mut workspace.block[..cur_rows * n_out];
        let t_decode = Instant::now();
        decode_standardized_block_f32(
            packed_flat,
            bytes_per_snp,
            n_samples,
            row_flip,
            row_mean,
            row_inv_sd,
            sample_idx,
            full_sample_fast,
            st,
            blk_slice,
            code4_lut,
            pool,
        )?;
        decode_acc += t_decode.elapsed().as_secs_f64();
        let tmp_slice = &mut workspace.tmp[..cur_rows * n_rhs];
        let t_xmul = Instant::now();
        row_major_block_mul_mat_f32(blk_slice, cur_rows, n_out, rhs, n_rhs, tmp_slice, pool);
        xmul_acc += t_xmul.elapsed().as_secs_f64();
        let t_xtmul = Instant::now();
        row_major_block_t_mul_mat_accum_f32(blk_slice, cur_rows, n_out, tmp_slice, n_rhs, out, pool);
        xtmul_acc += t_xtmul.elapsed().as_secs_f64();
    }
    let inv_m = 1.0_f32 / m_scale.max(1.0_f32);
    out.iter_mut().for_each(|v| *v *= inv_m);
    if let Some(t) = timing {
        t.decode_secs += decode_acc;
        t.xmul_secs += xmul_acc;
        t.xtmul_secs += xtmul_acc;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_grm_to_vec_f32_with_workspace(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    block_rows: usize,
    vec_in: &[f32],
    m_scale: f32,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
    workspace: &mut GrmApplyWorkspace,
    timing: Option<&mut HeApplyTiming>,
) -> Result<Vec<f32>, String> {
    let n_out = sample_idx.len();
    if vec_in.len() != n_out {
        return Err(format!(
            "apply_grm_to_vec_f32_with_workspace length mismatch: len(vec_in)={} != n_samples_subset={n_out}",
            vec_in.len()
        ));
    }
    let mut out = vec![0.0_f32; n_out];
    apply_grm_to_mat_f32_with_workspace(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_mean,
        row_inv_sd,
        sample_idx,
        full_sample_fast,
        block_rows,
        vec_in,
        1,
        m_scale,
        code4_lut,
        pool,
        workspace,
        timing,
        &mut out,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn he_variance_components_packed(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    y: &[f64],
    trace_samples: usize,
    block_rows: usize,
    std_eps: f64,
    use_train_maf: bool,
    max_iter: usize,
    tol: f64,
    seed: u64,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<HePcgResult, String> {
    he_variance_components_packed_with_covariates(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_maf,
        sample_idx,
        y,
        None,
        0,
        trace_samples,
        8,
        block_rows,
        std_eps,
        use_train_maf,
        max_iter,
        tol,
        seed,
        false,
        256,
        pool,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn he_variance_components_packed_with_covariates(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    y: &[f64],
    x_cov: Option<&[f64]>,
    p_cov: usize,
    trace_samples: usize,
    trace_probe_batch: usize,
    block_rows: usize,
    std_eps: f64,
    use_train_maf: bool,
    max_iter: usize,
    tol: f64,
    seed: u64,
    exact_trace_debug: bool,
    exact_trace_max_n: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<HePcgResult, String> {
    let stage_timing = env_truthy("JX_GS_HE_STAGE_TIMING");
    let stage_log_every = std::env::var("JX_GS_HE_STAGE_LOG_EVERY")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(8usize);
    let total_t0 = Instant::now();
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let m = row_flip.len();
    if m == 0 {
        return Err("SNP row count must be > 0".to_string());
    }
    if row_maf.len() != m {
        return Err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        ));
    }
    if packed_flat.len() != m.saturating_mul(bytes_per_snp) {
        return Err(format!(
            "packed payload size mismatch: got {}, expected {}",
            packed_flat.len(),
            m.saturating_mul(bytes_per_snp)
        ));
    }
    if sample_idx.is_empty() {
        return Err("sample_idx must not be empty".to_string());
    }
    if y.len() != sample_idx.len() {
        return Err(format!(
            "y length mismatch: got {}, expected {}",
            y.len(),
            sample_idx.len()
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err("y contains non-finite values".to_string());
    }
    if x_cov.is_none() && p_cov != 0 {
        return Err("p_cov > 0 but x_cov is None".to_string());
    }
    if x_cov.is_some() && p_cov == 0 {
        return Err("x_cov provided but p_cov == 0".to_string());
    }
    if trace_samples == 0 {
        return Err("trace_samples must be > 0".to_string());
    }
    if trace_probe_batch == 0 {
        return Err("trace_probe_batch must be > 0".to_string());
    }
    if !(std_eps.is_finite() && std_eps > 0.0_f64) {
        return Err("std_eps must be finite and > 0".to_string());
    }
    if max_iter == 0 {
        return Err("max_iter must be > 0".to_string());
    }
    if !(tol.is_finite() && tol > 0.0_f64) {
        return Err("tol must be finite and > 0".to_string());
    }

    let std_eps32 = std_eps.max(1e-12_f64) as f32;
    let train_maf_t0 = Instant::now();
    let row_stats = build_row_standardization_stats(
        packed_flat,
        bytes_per_snp,
        row_flip,
        row_maf,
        sample_idx,
        std_eps32,
        use_train_maf,
        pool,
    )?;
    let train_maf_secs = train_maf_t0.elapsed().as_secs_f64();
    if stage_timing {
        emit_he_stage_timing(
            "stage=train_maf",
            0,
            0,
            train_maf_secs,
            0.0,
            0.0,
            0.0,
            0.0,
            total_t0.elapsed().as_secs_f64(),
        );
    }
    let m_effective = row_stats.m_effective;
    if m_effective == 0 {
        return Err("No effective SNPs after std_eps filtering".to_string());
    }

    let n = sample_idx.len();
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let m_scale = m_effective as f32;
    let projector = CovariateProjector::from_optional_covariates(n, x_cov, p_cov)?;
    let mut proj_ws = ProjectionWorkspace::new(projector.p);
    let mut y_proj = y.to_vec();
    projector.project_vec_f64_in_place(&mut y_proj, &mut proj_ws)?;
    let y_f32: Vec<f32> = y_proj.iter().map(|v| *v as f32).collect();
    let code4_lut = &packed_byte_lut().code4;
    let mut grm_ws = GrmApplyWorkspace::new();
    let mut ky_apply_t = HeApplyTiming::default();

    let k_y = apply_grm_to_vec_f32_with_workspace(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        &row_stats.row_mean,
        &row_stats.row_inv_sd,
        sample_idx,
        full_sample_fast,
        block_rows,
        &y_f32,
        m_scale,
        code4_lut,
        pool,
        &mut grm_ws,
        if stage_timing {
            Some(&mut ky_apply_t)
        } else {
            None
        },
    )?;
    let y_ky = dot_f32_f64(&y_f32, &k_y);
    let y_y = dot_f32_f64(&y_f32, &y_f32);

    let batch_cols = trace_probe_batch.max(1);
    let mut probe_batch = vec![0.0_f32; n.saturating_mul(batch_cols)];
    let mut kv_batch = vec![0.0_f32; n.saturating_mul(batch_cols)];
    let mut tr_k_acc = 0.0_f64;
    let mut tr_k2_acc = 0.0_f64;
    let exact_trace_used = exact_trace_debug && n <= exact_trace_max_n.max(1);
    let total_trace_batches = if exact_trace_used {
        n.div_ceil(batch_cols)
    } else {
        trace_samples.div_ceil(batch_cols)
    };
    let trace_t0 = Instant::now();
    let mut trace_apply_t = HeApplyTiming::default();
    let mut trace_batches_done = 0usize;

    if exact_trace_used {
        let mut col_start = 0usize;
        while col_start < n {
            let cur = (n - col_start).min(batch_cols);
            probe_batch.fill(0.0_f32);
            for t in 0..cur {
                let i = col_start + t;
                probe_batch[i * batch_cols + t] = 1.0_f32;
            }
            projector.project_mat_f32_in_place(&mut probe_batch, batch_cols, &mut proj_ws)?;
            kv_batch.fill(0.0_f32);
            apply_grm_to_mat_f32_with_workspace(
                packed_flat,
                bytes_per_snp,
                n_samples,
                row_flip,
                &row_stats.row_mean,
                &row_stats.row_inv_sd,
                sample_idx,
                full_sample_fast,
                block_rows,
                &probe_batch,
                batch_cols,
                m_scale,
                code4_lut,
                pool,
                &mut grm_ws,
                if stage_timing {
                    Some(&mut trace_apply_t)
                } else {
                    None
                },
                &mut kv_batch,
            )?;
            projector.project_mat_f32_in_place(&mut kv_batch, batch_cols, &mut proj_ws)?;

            for t in 0..cur {
                let i = col_start + t;
                tr_k_acc += kv_batch[i * batch_cols + t] as f64;
                let mut col_norm2 = 0.0_f64;
                for r in 0..n {
                    let v = kv_batch[r * batch_cols + t] as f64;
                    col_norm2 += v * v;
                }
                tr_k2_acc += col_norm2;
            }
            col_start += cur;
            trace_batches_done += 1;
            if stage_timing
                && ((trace_batches_done % stage_log_every) == 0
                    || trace_batches_done == total_trace_batches)
            {
                let trace_secs = (trace_t0.elapsed().as_secs_f64()
                    - trace_apply_t.decode_secs
                    - trace_apply_t.gemm_secs())
                    .max(0.0_f64);
                emit_he_stage_timing(
                    "stage=trace",
                    trace_batches_done,
                    total_trace_batches,
                    train_maf_secs,
                    ky_apply_t.decode_secs + trace_apply_t.decode_secs,
                    ky_apply_t.xmul_secs + trace_apply_t.xmul_secs,
                    ky_apply_t.xtmul_secs + trace_apply_t.xtmul_secs,
                    trace_secs,
                    total_t0.elapsed().as_secs_f64(),
                );
            }
        }
    } else {
        let mut b_start = 0usize;
        while b_start < trace_samples {
            let cur = (trace_samples - b_start).min(batch_cols);
            probe_batch.fill(0.0_f32);
            for t in 0..cur {
                let probe_idx = b_start + t;
                let mut state =
                    splitmix64(seed ^ ((probe_idx as u64).wrapping_mul(0x517CC1B727220A95)));
                for i in 0..n {
                    state = splitmix64(state);
                    probe_batch[i * batch_cols + t] =
                        if (state & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
                }
            }
            projector.project_mat_f32_in_place(&mut probe_batch, batch_cols, &mut proj_ws)?;
            kv_batch.fill(0.0_f32);
            apply_grm_to_mat_f32_with_workspace(
                packed_flat,
                bytes_per_snp,
                n_samples,
                row_flip,
                &row_stats.row_mean,
                &row_stats.row_inv_sd,
                sample_idx,
                full_sample_fast,
                block_rows,
                &probe_batch,
                batch_cols,
                m_scale,
                code4_lut,
                pool,
                &mut grm_ws,
                if stage_timing {
                    Some(&mut trace_apply_t)
                } else {
                    None
                },
                &mut kv_batch,
            )?;
            projector.project_mat_f32_in_place(&mut kv_batch, batch_cols, &mut proj_ws)?;

            for t in 0..cur {
                let mut trk_one = 0.0_f64;
                let mut trk2_one = 0.0_f64;
                for i in 0..n {
                    let z = probe_batch[i * batch_cols + t] as f64;
                    let v = kv_batch[i * batch_cols + t] as f64;
                    trk_one += z * v;
                    trk2_one += v * v;
                }
                tr_k_acc += trk_one;
                tr_k2_acc += trk2_one;
            }
            b_start += cur;
            trace_batches_done += 1;
            if stage_timing
                && ((trace_batches_done % stage_log_every) == 0
                    || trace_batches_done == total_trace_batches)
            {
                let trace_secs = (trace_t0.elapsed().as_secs_f64()
                    - trace_apply_t.decode_secs
                    - trace_apply_t.gemm_secs())
                    .max(0.0_f64);
                emit_he_stage_timing(
                    "stage=trace",
                    trace_batches_done,
                    total_trace_batches,
                    train_maf_secs,
                    ky_apply_t.decode_secs + trace_apply_t.decode_secs,
                    ky_apply_t.xmul_secs + trace_apply_t.xmul_secs,
                    ky_apply_t.xtmul_secs + trace_apply_t.xtmul_secs,
                    trace_secs,
                    total_t0.elapsed().as_secs_f64(),
                );
            }
        }
    }
    if stage_timing && total_trace_batches == 0 {
        emit_he_stage_timing(
            "stage=trace",
            0,
            0,
            train_maf_secs,
            ky_apply_t.decode_secs,
            ky_apply_t.xmul_secs,
            ky_apply_t.xtmul_secs,
            0.0,
            total_t0.elapsed().as_secs_f64(),
        );
    }
    let tr_k = if exact_trace_used {
        tr_k_acc
    } else {
        tr_k_acc / (trace_samples as f64)
    };
    let tr_k2 = if exact_trace_used {
        tr_k2_acc
    } else {
        tr_k2_acc / (trace_samples as f64)
    };
    if !(tr_k.is_finite() && tr_k > 0.0_f64) {
        return Err(format!(
            "estimated Tr(PKP) is invalid: {tr_k}. Try increasing trace_samples."
        ));
    }
    if !(tr_k2.is_finite() && tr_k2 > 0.0_f64) {
        return Err(format!(
            "estimated Tr((PKP)^2) is invalid: {tr_k2}. Try increasing trace_samples."
        ));
    }

    let tr_p = projector.tr_p();
    // Stochastic trace estimation can slightly undershoot the PSD lower bound.
    // Add a tiny floor to keep the 2x2 normal matrix strictly SPD for direct solve.
    let tr_k2_floor = (tr_k * tr_k) / tr_p + tr_p * 1e-6_f64;
    let tr_k2_solve = tr_k2.max(tr_k2_floor);
    if tr_k2_solve > tr_k2 * 1.05_f64 {
        return Err(format!(
            "Tr((PKP)^2) stochastic estimate violates PSD bound too much: raw={tr_k2}, adjusted={tr_k2_solve}. Increase trace_samples."
        ));
    }
    let a00 = tr_k2_solve;
    let a01 = tr_k;
    let a11 = tr_p;
    let b0 = y_ky;
    let b1 = y_y;
    let (sigma_unconstrained_g2, sigma_unconstrained_e2) = he_solve_2x2(a00, a01, a11, b0, b1)?;
    let (sigma_g2, sigma_e2, nnls_projected, boundary_status) = he_project_nnls_2x2(
        a00,
        a01,
        a11,
        b0,
        b1,
        sigma_unconstrained_g2,
        sigma_unconstrained_e2,
    );
    let rel_res = {
        let res0 = a00.mul_add(sigma_g2, a01 * sigma_e2) - b0;
        let res1 = a01.mul_add(sigma_g2, a11 * sigma_e2) - b1;
        let rhs_norm = (b0 * b0 + b1 * b1).sqrt().max(1e-20_f64);
        ((res0 * res0 + res1 * res1).sqrt()) / rhs_norm
    };
    let converged = rel_res.is_finite() && rel_res <= tol.max(1e-12_f64);
    let h2 = {
        let denom = sigma_g2 + sigma_e2;
        if denom.is_finite() && denom > 0.0_f64 {
            sigma_g2 / denom
        } else {
            f64::NAN
        }
    };
    let lambda = if sigma_g2.is_finite() && sigma_g2 > 0.0_f64 {
        sigma_e2 / sigma_g2
    } else {
        f64::INFINITY
    };
    if stage_timing {
        let trace_secs = (trace_t0.elapsed().as_secs_f64()
            - trace_apply_t.decode_secs
            - trace_apply_t.gemm_secs())
            .max(0.0_f64);
        emit_he_stage_timing(
            "final",
            trace_batches_done,
            total_trace_batches,
            train_maf_secs,
            ky_apply_t.decode_secs + trace_apply_t.decode_secs,
            ky_apply_t.xmul_secs + trace_apply_t.xmul_secs,
            ky_apply_t.xtmul_secs + trace_apply_t.xtmul_secs,
            trace_secs,
            total_t0.elapsed().as_secs_f64(),
        );
    }

    Ok(HePcgResult {
        sigma_g2,
        sigma_e2,
        h2,
        lambda,
        converged,
        iters: 1,
        rel_res,
        m_effective,
        tr_k,
        tr_p,
        tr_k2,
        tr_k2_solve,
        y_ky,
        y_y,
        nnls_projected,
        boundary_status,
        exact_trace_used,
    })
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    train_sample_indices,
    y_train,
    site_keep=None,
    trace_samples=32,
    trace_probe_batch=64,
    tol=1e-6_f64,
    max_iter=32,
    block_rows=4096,
    std_eps=1e-12_f64,
    use_train_maf=true,
    exact_trace_debug=false,
    exact_trace_max_n=256,
    threads=0,
    seed=20260512_u64,
    packed=None,
    packed_n_samples=0,
    maf=None,
    row_flip=None,
    x_cov=None,
    blas_threads=0
))]
pub fn he_pcg_bed<'py>(
    py: Python<'py>,
    prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    y_train: PyReadonlyArray1<'py, f64>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    trace_samples: usize,
    trace_probe_batch: usize,
    tol: f64,
    max_iter: usize,
    block_rows: usize,
    std_eps: f64,
    use_train_maf: bool,
    exact_trace_debug: bool,
    exact_trace_max_n: usize,
    threads: usize,
    seed: u64,
    packed: Option<PyReadonlyArray2<'py, u8>>,
    packed_n_samples: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    blas_threads: usize,
) -> PyResult<(
    f64,
    f64,
    f64,
    bool,
    usize,
    f64,
    usize,
    f64,
    f64,
    f64,
    f64,
    f64,
)> {
    if trace_samples == 0 {
        return Err(PyRuntimeError::new_err("trace_samples must be > 0"));
    }
    if trace_probe_batch == 0 {
        return Err(PyRuntimeError::new_err("trace_probe_batch must be > 0"));
    }
    if max_iter == 0 {
        return Err(PyRuntimeError::new_err("max_iter must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0_f64) {
        return Err(PyRuntimeError::new_err("tol must be finite and > 0"));
    }
    if !(std_eps.is_finite() && std_eps > 0.0_f64) {
        return Err(PyRuntimeError::new_err("std_eps must be finite and > 0"));
    }

    let use_external_packed =
        packed.is_some() || maf.is_some() || row_flip.is_some() || packed_n_samples > 0;
    let mut loaded_packed_arr = None;
    let mut loaded_maf_arr = None;
    let mut external_maf_ro: Option<PyReadonlyArray1<'py, f32>> = None;
    let mut external_row_flip_ro: Option<PyReadonlyArray1<'py, bool>> = None;

    let n_samples: usize;
    let packed_ro: PyReadonlyArray2<'py, u8>;
    if use_external_packed {
        packed_ro = packed.ok_or_else(|| {
            PyRuntimeError::new_err("he_pcg_bed: packed payload path requires `packed` argument.")
        })?;
        external_maf_ro = Some(maf.ok_or_else(|| {
            PyRuntimeError::new_err("he_pcg_bed: packed payload path requires `maf` argument.")
        })?);
        external_row_flip_ro = Some(row_flip.ok_or_else(|| {
            PyRuntimeError::new_err("he_pcg_bed: packed payload path requires `row_flip` argument.")
        })?);
        if packed_n_samples == 0 {
            return Err(PyRuntimeError::new_err(
                "he_pcg_bed: packed payload path requires packed_n_samples > 0.",
            ));
        }
        n_samples = packed_n_samples;
    } else {
        let (packed_arr, _miss_arr, maf_arr, _std_arr, n_samples_loaded) =
            crate::gfreader::load_bed_2bit_packed(py, prefix)?;
        if n_samples_loaded == 0 {
            return Err(PyRuntimeError::new_err("No samples found in BED input."));
        }
        n_samples = n_samples_loaded;
        loaded_packed_arr = Some(packed_arr);
        loaded_maf_arr = Some(maf_arr);
        packed_ro = loaded_packed_arr
            .as_ref()
            .expect("packed array must exist")
            .readonly();
    }

    let packed_view = packed_ro.as_array();
    if packed_view.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed BED payload must be 2D (m, bytes_per_snp).",
        ));
    }
    let m_total = packed_view.shape()[0];
    if m_total == 0 {
        return Err(PyRuntimeError::new_err("No SNP rows found in BED input."));
    }
    let bytes_per_snp = packed_view.shape()[1];
    let expected_bps = n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let packed_flat: Cow<[u8]> = match packed_ro.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_view.iter().copied().collect()),
    };

    let maf_full: Vec<f32> = if use_external_packed {
        let maf_ro = external_maf_ro
            .as_ref()
            .expect("external maf must exist for packed payload path");
        match maf_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => maf_ro.as_array().iter().copied().collect(),
        }
    } else {
        let maf_ro = loaded_maf_arr
            .as_ref()
            .expect("maf array must exist")
            .readonly();
        match maf_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => maf_ro.as_array().iter().copied().collect(),
        }
    };
    if maf_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "maf length mismatch: got {}, expected {m_total}",
            maf_full.len()
        )));
    }

    let row_flip_full: Vec<bool> = if use_external_packed {
        let row_flip_ro = external_row_flip_ro
            .as_ref()
            .expect("external row_flip must exist for packed payload path");
        match row_flip_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_flip_ro.as_array().iter().copied().collect(),
        }
    } else {
        let row_flip_full_arr = bed_packed_row_flip_mask(
            py,
            loaded_packed_arr
                .as_ref()
                .expect("packed array must exist")
                .readonly(),
            n_samples,
        )?;
        let row_flip_ro = row_flip_full_arr.readonly();
        match row_flip_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_flip_ro.as_array().iter().copied().collect(),
        }
    };
    if row_flip_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m_total}",
            row_flip_full.len()
        )));
    }

    let mut eff_m = m_total;
    let mut packed_keep: Cow<[u8]> = Cow::Borrowed(packed_flat.as_ref());
    let mut maf_keep: Cow<[f32]> = Cow::Borrowed(maf_full.as_slice());
    let mut row_flip_keep: Cow<[bool]> = Cow::Borrowed(row_flip_full.as_slice());
    if let Some(mask) = site_keep {
        let mask_vec: Vec<bool> = match mask.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mask.as_array().iter().copied().collect(),
        };
        if mask_vec.len() != m_total {
            return Err(PyRuntimeError::new_err(format!(
                "site_keep length mismatch: got {}, expected {m_total}",
                mask_vec.len()
            )));
        }
        let keep_idx: Vec<usize> = mask_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &k)| if k { Some(i) } else { None })
            .collect();
        if keep_idx.is_empty() {
            return Err(PyRuntimeError::new_err(
                "No SNPs remained after applying site_keep mask.",
            ));
        }
        let keep_is_identity = (keep_idx.len() == m_total)
            && keep_idx
                .iter()
                .enumerate()
                .all(|(dst_row, &src_row)| dst_row == src_row);
        if !keep_is_identity {
            eff_m = keep_idx.len();
            let mut packed_subset = vec![0_u8; eff_m * bytes_per_snp];
            let mut maf_subset = vec![0.0_f32; eff_m];
            let mut row_flip_subset = vec![false; eff_m];
            for (dst_row, &src_row) in keep_idx.iter().enumerate() {
                let src_off = src_row * bytes_per_snp;
                let dst_off = dst_row * bytes_per_snp;
                packed_subset[dst_off..dst_off + bytes_per_snp]
                    .copy_from_slice(&packed_flat[src_off..src_off + bytes_per_snp]);
                maf_subset[dst_row] = maf_full[src_row];
                row_flip_subset[dst_row] = row_flip_full[src_row];
            }
            packed_keep = Cow::Owned(packed_subset);
            maf_keep = Cow::Owned(maf_subset);
            row_flip_keep = Cow::Owned(row_flip_subset);
        }
    }

    let train_idx = parse_index_vec_i64(
        train_sample_indices.as_slice()?,
        n_samples,
        "train_sample_indices",
    )?;
    if train_idx.is_empty() {
        return Err(PyRuntimeError::new_err(
            "train_sample_indices must not be empty.",
        ));
    }
    let y_vec_f64: Vec<f64> = match y_train.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y_train.as_array().iter().copied().collect(),
    };
    if y_vec_f64.len() != train_idx.len() {
        return Err(PyRuntimeError::new_err(format!(
            "y_train length mismatch: got {}, expected {}",
            y_vec_f64.len(),
            train_idx.len()
        )));
    }
    if y_vec_f64.iter().any(|v| !v.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "y_train contains non-finite values.",
        ));
    }

    let (x_cov_train, p_cov): (Option<Vec<f64>>, usize) = if let Some(x_cov_ro) = x_cov {
        let x_arr = x_cov_ro.as_array();
        if x_arr.ndim() != 2 {
            return Err(PyRuntimeError::new_err("x_cov must be 2D (n, p_cov)"));
        }
        let rows = x_arr.shape()[0];
        let p = x_arr.shape()[1];
        if p == 0 {
            return Err(PyRuntimeError::new_err(
                "x_cov must have at least one column",
            ));
        }
        let x_flat: Cow<[f64]> = match x_cov_ro.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
        };
        if rows == n_samples {
            let mut out = vec![0.0_f64; train_idx.len() * p];
            for (ri, &sid) in train_idx.iter().enumerate() {
                let src = &x_flat[sid * p..(sid + 1) * p];
                out[ri * p..(ri + 1) * p].copy_from_slice(src);
            }
            (Some(out), p)
        } else if rows == train_idx.len() {
            (Some(x_flat.as_ref().to_vec()), p)
        } else {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov rows mismatch: got {rows}, expected either n_samples={n_samples} or n_train={}",
                train_idx.len()
            )));
        }
    } else {
        (None, 0usize)
    };

    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let he_res = py
        .detach(move || {
            // Keep OpenBLAS stage control independent from Rayon pool sizing when an
            // explicit BLAS cap is provided, but preserve the legacy behavior of
            // following `threads` when `blas_threads` is left at its default.
            let blas_threads_effective = if blas_threads > 0 {
                blas_threads.max(1)
            } else if threads > 0 {
                threads.max(1)
            } else {
                0
            };
            let _blas_guard = if blas_threads_effective > 0 {
                Some(OpenBlasThreadGuard::enter(blas_threads_effective))
            } else {
                None
            };
            he_variance_components_packed_with_covariates(
                packed_keep.as_ref(),
                bytes_per_snp,
                n_samples,
                row_flip_keep.as_ref(),
                maf_keep.as_ref(),
                &train_idx,
                &y_vec_f64,
                x_cov_train.as_deref(),
                p_cov,
                trace_samples,
                trace_probe_batch,
                block_rows,
                std_eps,
                use_train_maf,
                max_iter,
                tol,
                seed,
                exact_trace_debug,
                exact_trace_max_n,
                pool_ref,
            )
        })
        .map_err(map_err_string_to_py)?;

    Ok((
        he_res.sigma_g2,
        he_res.sigma_e2,
        he_res.h2,
        he_res.converged,
        he_res.iters,
        he_res.rel_res,
        he_res.m_effective.min(eff_m),
        he_res.tr_k2,
        he_res.y_ky,
        he_res.y_y,
        he_res.lambda,
        he_res.tr_k2_solve,
    ))
}
