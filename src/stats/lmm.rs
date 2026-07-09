// Exact LMM / LMM2 scan code:
// V_lambda = K + lambda I
// P_lambda = V_lambda^{-1} - V_lambda^{-1}X(X'V_lambda^{-1}X)^{-1}X'V_lambda^{-1}
// beta_hat = (g' P_lambda y) / (g' P_lambda g)
// sigma2_hat = (y' P_lambda y) / (n - rank(X) - 1)
// se(beta_hat) = sqrt(sigma2_hat / (g' P_lambda g))
//
// LMM optimizes lambda per SNP against the REML objective and reports Wald beta/se/pwald.
// LMM2 uses the same REML beta/se for Wald output, then runs an additional ML optimization
// to emit per-SNP lambda/ml/plrt.

use matrixmultiply::sgemm;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::assoc2tsv::{
    append_assoc_block_from_arrays, resolve_assoc_tsv_metadata, send_text_buf, AssocArraysBlock,
    AssocMissBlock, AssocResultCols, AssocResultLayout,
};
use crate::bedmath::packed_row_missing_count_selected;
use crate::blas::{
    cblas_sgemm_dispatch, rust_sgemm_backend_tag, BlasThreadGuard, CblasInt, CBLAS_NO_TRANS,
    CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use crate::brent::{brent_minimize, brent_minimize_with_init};
use crate::decode::{
    decode_centered_block_packed_model_f32, AdditiveDecodePlan, PackedGeneticModel,
};
use crate::gfcore;
use crate::gfcore::BimChunkReader;
use crate::gfreader::{
    count_packed_row_counts, count_packed_row_counts_selected_with_excluded, is_simple_snp_allele,
};
use crate::gload::WindowedBedMatrix;
use crate::linalg::{chi2_sf_df1, cholesky_inplace, cholesky_solve_into, normal_sf};
use crate::stats_common::{env_truthy, get_cached_pool, parse_index_vec_i64, AsyncTsvWriter};
use memmap2::Mmap;
use std::fs::File;

use crate::reml::{final_beta_se, ml_loglike, reml_loglike, run_rotated_assoc_block_f32};

struct RotatedRemlAssocState {
    snp_vec: Vec<f64>,
    last_log10_lbd: Option<f64>,
}

struct RotatedLmm2AssocState {
    snp_vec: Vec<f64>,
    last_log10_lbd_reml: Option<f64>,
    last_log10_lbd_ml: Option<f64>,
}

#[inline]
fn copy_rotated_snp_row_to_f64(src: &[f32], dst: &mut [f64]) -> f64 {
    let mut ssq = 0.0_f64;
    for i in 0..src.len() {
        let v = src[i] as f64;
        dst[i] = v;
        ssq += v * v;
    }
    ssq
}

#[inline]
fn fill_invalid_rotated_assoc_row(out_row: &mut [f64], with_plrt: bool) {
    out_row[0] = f64::NAN;
    out_row[1] = f64::NAN;
    out_row[2] = 1.0;
    if with_plrt {
        out_row[3] = 1.0;
    }
}

#[inline]
fn fill_invalid_rotated_lmm2_row(out_row: &mut [f64]) {
    out_row[0] = f64::NAN;
    out_row[1] = f64::NAN;
    out_row[2] = 1.0;
    out_row[3] = f64::NAN;
    out_row[4] = f64::NAN;
    out_row[5] = 1.0;
}

#[allow(clippy::too_many_arguments)]
fn run_rotated_reml_assoc_block_f32(
    g_block: &[f32],
    rows: usize,
    n: usize,
    s: &[f64],
    xcov_flat: &[f64],
    y: &[f64],
    p_cov: usize,
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
    init_log10_lbd: Option<f64>,
    seed_with_init_guess: bool,
    carry_warm_start: bool,
    nullml: Option<f64>,
    out: &mut [f64],
    out_cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    run_rotated_assoc_block_f32(
        g_block,
        rows,
        n,
        out,
        out_cols,
        pool,
        || RotatedRemlAssocState {
            snp_vec: vec![0.0_f64; n],
            last_log10_lbd: None,
        },
        |_, row, state, out_row| {
            let snp_ssq = copy_rotated_snp_row_to_f64(row, &mut state.snp_vec);
            if !snp_ssq.is_finite() || snp_ssq <= 1e-12 {
                fill_invalid_rotated_assoc_row(out_row, with_plrt);
                return;
            }

            let init_guess = if carry_warm_start {
                state.last_log10_lbd.or(init_log10_lbd)
            } else if seed_with_init_guess {
                init_log10_lbd
            } else {
                None
            };
            let (best_log10_lbd, _best_cost) = if let Some(guess) = init_guess {
                brent_minimize_with_init(
                    |x| -reml_loglike(x, s, xcov_flat, y, Some(&state.snp_vec[..]), n, p_cov),
                    low,
                    high,
                    tol,
                    max_iter,
                    Some(guess),
                )
            } else {
                brent_minimize(
                    |x| -reml_loglike(x, s, xcov_flat, y, Some(&state.snp_vec[..]), n, p_cov),
                    low,
                    high,
                    tol,
                    max_iter,
                )
            };
            if carry_warm_start {
                state.last_log10_lbd = Some(best_log10_lbd);
            }

            let (beta, se, _lbd) =
                final_beta_se(best_log10_lbd, s, xcov_flat, y, &state.snp_vec, n, p_cov);
            let pwald = if beta.is_finite() && se.is_finite() && se > 0.0 {
                let z = beta / se;
                (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
            } else {
                fill_invalid_rotated_assoc_row(out_row, with_plrt);
                return;
            };

            out_row[0] = beta;
            out_row[1] = se;
            out_row[2] = if pwald.is_finite() { pwald } else { 1.0 };

            if with_plrt {
                let ml = ml_loglike(
                    best_log10_lbd,
                    s,
                    xcov_flat,
                    y,
                    Some(&state.snp_vec[..]),
                    n,
                    p_cov,
                );
                out_row[3] = if ml.is_finite() {
                    let mut stat = 2.0 * (ml - nullml_val);
                    if !stat.is_finite() || stat < 0.0 {
                        stat = 0.0;
                    }
                    chi2_sf_df1(stat)
                } else {
                    1.0
                };
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn run_rotated_lmm2_assoc_block_f32(
    g_block: &[f32],
    rows: usize,
    n: usize,
    s: &[f64],
    xcov_flat: &[f64],
    y: &[f64],
    p_cov: usize,
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
    init_log10_lbd_reml: Option<f64>,
    init_log10_lbd_ml: Option<f64>,
    use_warm_start: bool,
    nullml: f64,
    out: &mut [f64],
    out_cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    run_rotated_assoc_block_f32(
        g_block,
        rows,
        n,
        out,
        out_cols,
        pool,
        || RotatedLmm2AssocState {
            snp_vec: vec![0.0_f64; n],
            last_log10_lbd_reml: None,
            last_log10_lbd_ml: None,
        },
        |_, row, state, out_row| {
            let snp_ssq = copy_rotated_snp_row_to_f64(row, &mut state.snp_vec);
            if !snp_ssq.is_finite() || snp_ssq <= 1e-12 {
                fill_invalid_rotated_lmm2_row(out_row);
                return;
            }

            let reml_init_guess = if use_warm_start {
                state
                    .last_log10_lbd_reml
                    .or(init_log10_lbd_reml)
                    .or(init_log10_lbd_ml)
            } else {
                init_log10_lbd_reml.or(init_log10_lbd_ml)
            };
            let (best_log10_lbd_reml, _best_reml_cost) = brent_minimize_with_init(
                |x| -reml_loglike(x, s, xcov_flat, y, Some(&state.snp_vec[..]), n, p_cov),
                low,
                high,
                tol,
                max_iter,
                reml_init_guess,
            );
            if use_warm_start {
                state.last_log10_lbd_reml = Some(best_log10_lbd_reml);
            }

            let (beta, se, lambda_reml) = final_beta_se(
                best_log10_lbd_reml,
                s,
                xcov_flat,
                y,
                &state.snp_vec,
                n,
                p_cov,
            );
            let pwald = if beta.is_finite() && se.is_finite() && se > 0.0 {
                let z = beta / se;
                (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
            } else {
                fill_invalid_rotated_lmm2_row(out_row);
                return;
            };

            let ml_init_guess = if use_warm_start {
                state
                    .last_log10_lbd_ml
                    .or(Some(best_log10_lbd_reml))
                    .or(init_log10_lbd_ml)
                    .or(init_log10_lbd_reml)
            } else {
                Some(best_log10_lbd_reml)
                    .or(init_log10_lbd_ml)
                    .or(init_log10_lbd_reml)
            };
            let (best_log10_lbd_ml, best_ml_cost) = brent_minimize_with_init(
                |x| -ml_loglike(x, s, xcov_flat, y, Some(&state.snp_vec[..]), n, p_cov),
                low,
                high,
                tol,
                max_iter,
                ml_init_guess,
            );
            if use_warm_start {
                state.last_log10_lbd_ml = Some(best_log10_lbd_ml);
            }

            let mut ml_alt = -best_ml_cost;
            if !ml_alt.is_finite() {
                ml_alt = ml_loglike(
                    best_log10_lbd_ml,
                    s,
                    xcov_flat,
                    y,
                    Some(&state.snp_vec[..]),
                    n,
                    p_cov,
                );
            }
            let mut stat = if ml_alt.is_finite() {
                2.0 * (ml_alt - nullml)
            } else {
                0.0
            };
            if !stat.is_finite() || stat < 0.0 {
                stat = 0.0;
            }
            let plrt = chi2_sf_df1(stat);

            out_row[0] = beta;
            out_row[1] = se;
            out_row[2] = if pwald.is_finite() { pwald } else { 1.0 };
            out_row[3] = lambda_reml;
            out_row[4] = ml_alt;
            out_row[5] = if plrt.is_finite() { plrt } else { 1.0 };
        },
    );
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, g_rot_chunk, max_iter=50, tol=1e-2, threads=0, nullml=None))]
pub fn lmm_reml_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    g_rot_chunk: PyReadonlyArray2<'py, f32>,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let g_arr = g_rot_chunk.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("g_rot_chunk must be (m_chunk, n)"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let m_chunk = g_arr.shape()[0];
    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let g_slice_opt = g_rot_chunk.as_slice().ok();

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let beta_se_p = PyArray2::<f64>::zeros(py, [m_chunk, out_cols], false).into_bound();

    let beta_se_p_slice: &mut [f64] = unsafe {
        beta_se_p
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("beta_se_p not contiguous"))?
    };
    // For REML chunk scanning, rebuilding a local pool is consistently faster
    // than thread-local pool reuse in CLI LMM benchmarks.
    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    py.detach(|| {
        if let Some(gs) = g_slice_opt {
            if let Some(tp) = &pool {
                tp.install(|| {
                    run_rotated_reml_assoc_block_f32(
                        gs,
                        m_chunk,
                        n,
                        s,
                        &xcov_flat,
                        y,
                        p_cov,
                        low,
                        high,
                        tol,
                        max_iter,
                        None,
                        false,
                        false,
                        if with_plrt { Some(nullml_val) } else { None },
                        beta_se_p_slice,
                        out_cols,
                        None,
                    );
                });
            } else {
                run_rotated_reml_assoc_block_f32(
                    gs,
                    m_chunk,
                    n,
                    s,
                    &xcov_flat,
                    y,
                    p_cov,
                    low,
                    high,
                    tol,
                    max_iter,
                    None,
                    false,
                    false,
                    if with_plrt { Some(nullml_val) } else { None },
                    beta_se_p_slice,
                    out_cols,
                    None,
                );
            }
        } else {
            let mut run = || {
                beta_se_p_slice
                    .par_chunks_mut(out_cols)
                    .enumerate()
                    .for_each_init(
                        || vec![0.0_f64; n],
                        |snp_vec, (idx, out_row)| {
                            let row = g_arr.row(idx);
                            for i in 0..n {
                                snp_vec[i] = row[i] as f64;
                            }

                            let (best_log10_lbd, _best_cost) = brent_minimize(
                                |x| {
                                    -reml_loglike(x, s, &xcov_flat, y, Some(&snp_vec[..]), n, p_cov)
                                },
                                low,
                                high,
                                tol,
                                max_iter,
                            );

                            let (beta, se, _lbd) =
                                final_beta_se(best_log10_lbd, s, &xcov_flat, y, &snp_vec, n, p_cov);

                            let p = if beta.is_finite() && se.is_finite() && se > 0.0 {
                                let z = beta / se;
                                (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
                            } else {
                                1.0
                            };

                            out_row[0] = beta;
                            out_row[1] = se;
                            out_row[2] = if p.is_finite() { p } else { 1.0 };

                            if with_plrt {
                                let ml = ml_loglike(
                                    best_log10_lbd,
                                    s,
                                    &xcov_flat,
                                    y,
                                    Some(&snp_vec[..]),
                                    n,
                                    p_cov,
                                );
                                out_row[3] = if ml.is_finite() {
                                    let mut stat = 2.0 * (ml - nullml_val);
                                    if !stat.is_finite() || stat < 0.0 {
                                        stat = 0.0;
                                    }
                                    chi2_sf_df1(stat)
                                } else {
                                    1.0
                                };
                            }
                        },
                    );
            };

            if let Some(tp) = &pool {
                tp.install(run);
            } else {
                run();
            }
        }
    });

    Ok(beta_se_p)
}

#[inline]

fn rotate_snp_block_with_ut(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
) {
    if rows == 0 || n == 0 {
        return;
    }
    // C(rows, n) = A(rows, n) * U(n, n)
    // We store U^T in row-major (`u_t`). For matrixmultiply strides, expose U by
    // setting B row stride = 1 and col stride = n, i.e. B(i,j)=u_t[j*n + i].
    unsafe {
        sgemm(
            rows,
            n,
            n,
            1.0,
            snp_block.as_ptr(),
            n as isize,
            1,
            u_t.as_ptr(),
            1,
            n as isize,
            0.0,
            out_block.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
fn env_positive_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
}

#[inline]
fn choose_rotate_row_tile_rows_fvlmm(rows: usize, n: usize, thread_hint: usize) -> usize {
    if let Some(v) = env_positive_usize("JX_GWAS_ROTATE_ROW_TILE") {
        return v.max(1).min(rows.max(1));
    }
    if n < 2048 {
        return choose_rotate_tile_rows(rows, thread_hint);
    }
    let target_mb = env_positive_usize("JX_GWAS_ROTATE_ROW_TILE_MB").unwrap_or_else(|| {
        if n >= 16384 {
            8usize
        } else if n >= 8192 {
            6usize
        } else {
            4usize
        }
    });
    let target_bytes = target_mb.saturating_mul(1024 * 1024);
    let bytes_per_row = n
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    let mut tile_rows = (target_bytes / bytes_per_row).max(16).min(rows.max(1));
    let max_tile = if n >= 16384 {
        256usize
    } else if n >= 8192 {
        192usize
    } else {
        128usize
    };
    tile_rows = tile_rows.clamp(16, max_tile.min(rows.max(1)));
    tile_rows.min(rows.max(1))
}

#[inline]
fn choose_rotate_col_block_cols_fvlmm(n: usize) -> usize {
    if let Some(v) = env_positive_usize("JX_GWAS_ROTATE_COL_BLOCK") {
        return v.max(1).min(n.max(1));
    }
    if n < 2048 {
        return n.max(1);
    }
    let target_mb = env_positive_usize("JX_GWAS_ROTATE_COL_BLOCK_MB").unwrap_or_else(|| {
        if n >= 16384 {
            12usize
        } else if n >= 8192 {
            16usize
        } else {
            8usize
        }
    });
    let target_bytes = target_mb.saturating_mul(1024 * 1024);
    let bytes_per_col = n.saturating_mul(std::mem::size_of::<f32>()).max(1);
    let cols = (target_bytes / bytes_per_col).max(64);
    cols.clamp(64, 1024).min(n.max(1))
}

#[inline]
fn rotate_snp_block_with_ut_parallel_blocked(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    row_tile_rows: usize,
    col_block_cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    if rows == 0 || n == 0 {
        return;
    }
    debug_assert!(snp_block.len() >= rows.saturating_mul(n));
    debug_assert!(u_t.len() >= n.saturating_mul(n));
    debug_assert!(out_block.len() >= rows.saturating_mul(n));
    let tile_rows = row_tile_rows.max(1).min(rows);
    let col_block = col_block_cols.max(1).min(n);

    let mut run_parallel = || {
        out_block
            .par_chunks_mut(tile_rows * n)
            .enumerate()
            .for_each_init(
                || vec![0.0_f32; tile_rows.saturating_mul(col_block)],
                |scratch, (chunk_idx, out_chunk)| {
                    let row_start = chunk_idx * tile_rows;
                    let rows_here = out_chunk.len() / n;
                    let a_start = row_start * n;
                    let a_end = a_start + rows_here * n;
                    let a_block = &snp_block[a_start..a_end];
                    if col_block >= n {
                        rotate_snp_block_with_ut(a_block, rows_here, n, u_t, out_chunk);
                        return;
                    }
                    for col_start in (0..n).step_by(col_block) {
                        let cols_here = (n - col_start).min(col_block);
                        let scratch_sub = &mut scratch[..rows_here * cols_here];
                        matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
                            a_block,
                            rows_here,
                            n,
                            &u_t[col_start * n..(col_start + cols_here) * n],
                            cols_here,
                            scratch_sub,
                        );
                        for r in 0..rows_here {
                            let src = &scratch_sub[r * cols_here..(r + 1) * cols_here];
                            let dst =
                                &mut out_chunk[r * n + col_start..r * n + col_start + cols_here];
                            dst.copy_from_slice(src);
                        }
                    }
                },
            );
    };

    if tile_rows >= rows {
        if col_block >= n {
            rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
            return;
        }
        let mut scratch = vec![0.0_f32; rows.saturating_mul(col_block)];
        for col_start in (0..n).step_by(col_block) {
            let cols_here = (n - col_start).min(col_block);
            let scratch_sub = &mut scratch[..rows * cols_here];
            matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
                snp_block,
                rows,
                n,
                &u_t[col_start * n..(col_start + cols_here) * n],
                cols_here,
                scratch_sub,
            );
            for r in 0..rows {
                let src = &scratch_sub[r * cols_here..(r + 1) * cols_here];
                let dst = &mut out_block[r * n + col_start..r * n + col_start + cols_here];
                dst.copy_from_slice(src);
            }
        }
        return;
    }

    if let Some(tp) = pool {
        tp.install(run_parallel);
    } else {
        run_parallel();
    }
}

#[inline]
fn assoc_rotate_prefers_rayon_rowmajor_f32_kernel() -> bool {
    for env_name in ["JX_LMM_ROTATE_F32_KERNEL", "JX_ROWMAJOR_F32_KERNEL"] {
        if let Ok(raw) = std::env::var(env_name) {
            let norm = raw.trim().to_ascii_lowercase();
            match norm.as_str() {
                "rayon" | "parallel" | "custom" => return true,
                "blas" | "gemm" | "serial" => return false,
                _ => {}
            }
        }
    }
    // For LMM/FvLMM association rotate-proj, default to BLAS-backed GEMM
    // across platforms. This path is large dense projection, unlike some
    // HE/PCG row-major kernels that still prefer custom Rayon tiling.
    false
}

#[inline]
fn rotate_snp_block_with_ut_blas(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    threads: usize,
    proj_pool: Option<&Arc<rayon::ThreadPool>>,
) {
    if rows == 0 || n == 0 {
        return;
    }
    debug_assert!(snp_block.len() >= rows.saturating_mul(n));
    debug_assert!(u_t.len() >= n.saturating_mul(n));
    debug_assert!(out_block.len() >= rows.saturating_mul(n));

    if assoc_rotate_prefers_rayon_rowmajor_f32_kernel() {
        let thread_hint = if threads > 0 {
            threads
        } else {
            rayon::current_num_threads()
        };
        let row_tile = choose_rotate_row_tile_rows_fvlmm(rows, n, thread_hint);
        let col_block = choose_rotate_col_block_cols_fvlmm(n);
        rotate_snp_block_with_ut_parallel_blocked(
            snp_block, rows, n, u_t, out_block, row_tile, col_block, proj_pool,
        );
        return;
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        let _blas_guard = BlasThreadGuard::enter(threads.max(1));
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            rows as CblasInt,
            n as CblasInt,
            n as CblasInt,
            1.0_f32,
            snp_block.as_ptr(),
            n as CblasInt,
            u_t.as_ptr(),
            n as CblasInt,
            0.0_f32,
            out_block.as_mut_ptr(),
            n as CblasInt,
        );
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
    }
}

#[inline]
fn matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
    a: &[f32],
    m: usize,
    k: usize,
    b_rowmajor_nk: &[f32],
    n: usize,
    out: &mut [f32],
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    debug_assert!(a.len() >= m.saturating_mul(k));
    debug_assert!(b_rowmajor_nk.len() >= n.saturating_mul(k));
    debug_assert!(out.len() >= m.saturating_mul(n));
    // C(m, n) = A(m, k) * B^T(k, n)
    // `b_rowmajor_nk` stores B(n, k) row-major. Expose B^T by setting
    // B strides to: row_stride=1, col_stride=k.
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b_rowmajor_nk.as_ptr(),
            1,
            k as isize,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
fn choose_rotate_tile_rows(rows: usize, thread_hint: usize) -> usize {
    if rows <= 1 {
        return 1;
    }
    let threads = thread_hint.max(1);
    // Avoid splitting when there is no real parallelism (or chunk is tiny):
    // one large SGEMM is much faster than many tiny calls in this case.
    if threads <= 1 || rows <= 32 {
        return rows;
    }
    // Keep enough independent tiles to feed the rayon pool, but avoid tiles
    // so small that SGEMM launch overhead dominates.
    let target_tasks = threads.saturating_mul(2).max(1);
    let mut tile_rows = (rows + target_tasks - 1) / target_tasks;
    tile_rows = tile_rows.clamp(16, 128);
    tile_rows.min(rows)
}

#[inline]
fn rotate_snp_block_with_ut_parallel(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    tile_rows: usize,
) {
    if rows == 0 || n == 0 {
        return;
    }

    let tile_rows = tile_rows.max(1).min(rows);
    if tile_rows >= rows {
        rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
        return;
    }

    out_block
        .par_chunks_mut(tile_rows * n)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * tile_rows;
            let rows_here = out_chunk.len() / n;
            let a_start = row_start * n;
            let a_end = a_start + rows_here * n;
            rotate_snp_block_with_ut(&snp_block[a_start..a_end], rows_here, n, u_t, out_chunk);
        });
}

#[inline]
fn record_elapsed_nanos(acc: &AtomicU64, t0: Instant) {
    let nanos = t0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    acc.fetch_add(nanos, Ordering::Relaxed);
}

#[inline]
fn elapsed_nanos_to_secs(acc: &AtomicU64) -> f64 {
    (acc.load(Ordering::Relaxed) as f64) * 1e-9
}

/// Per-SNP filter results from parallel counting (Copy-friendly, no heap strings).
struct SnpCounts {
    flip: bool,
    maf: f32,
    miss_rate: f32,
    missing_count: usize,
}

/// Double-buffer for the streaming BED → TSV pipeline.
struct StreamingChunk {
    // Stage 1 output: count + decode
    g_block: Vec<f32>,
    miss_block: Vec<usize>,
    indices: Vec<usize>,
    flip: Vec<bool>,
    maf: Vec<f32>,
    miss_rate: Vec<f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    a0: Vec<String>,
    a1: Vec<String>,
    // Stage 2 output
    rot_block: Vec<f32>,
    out_block: Vec<f64>,
    // Bookkeeping
    rows: usize,
    scanned_to: usize,
}

impl StreamingChunk {
    fn new(capacity: usize, n: usize, out_cols: usize) -> Self {
        Self {
            g_block: vec![0.0_f32; capacity.saturating_mul(n)],
            miss_block: vec![0usize; capacity],
            indices: Vec::with_capacity(capacity),
            flip: Vec::with_capacity(capacity),
            maf: Vec::with_capacity(capacity),
            miss_rate: Vec::with_capacity(capacity),
            chrom: Vec::with_capacity(capacity),
            pos: Vec::with_capacity(capacity),
            snp: Vec::with_capacity(capacity),
            a0: Vec::with_capacity(capacity),
            a1: Vec::with_capacity(capacity),
            rot_block: vec![0.0_f32; capacity.saturating_mul(n)],
            out_block: vec![0.0_f64; capacity.saturating_mul(out_cols)],
            rows: 0,
            scanned_to: 0,
        }
    }

    fn clear(&mut self) {
        self.indices.clear();
        self.flip.clear();
        self.maf.clear();
        self.miss_rate.clear();
        self.chrom.clear();
        self.pos.clear();
        self.snp.clear();
        self.a0.clear();
        self.a1.clear();
        self.rows = 0;
        self.scanned_to = 0;
    }
}

struct UnifiedBedScanSummary {
    total_rows: usize,
    total_secs: f64,
    pipeline_secs: f64,
    count_secs: f64,
    meta_secs: f64,
    decode_secs: f64,
    proj_secs: f64,
    assoc_secs: f64,
    tsv_secs: f64,
    n_snps: usize,
    n: usize,
    scan_chunk_snps: usize,
    proj_threads: usize,
    assoc_threads: usize,
}

struct PreparedBedScanMeta {
    row_indices: Vec<i64>,
    row_flip: Vec<bool>,
    row_maf: Vec<f32>,
    miss_counts: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn run_unified_bed_scan_to_tsv_common<RunRows, FormatRows>(
    bed_prefix: &str,
    out_tsv: &str,
    ut_flat: &[f32],
    n: usize,
    gm: PackedGeneticModel,
    sample_ids: Option<Vec<String>>,
    maf_thr: f32,
    miss_thr: f32,
    het_thr: f32,
    snps_only: bool,
    threads: usize,
    out_cols: usize,
    rotate_block_rows: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
    prepared_meta: Option<&PreparedBedScanMeta>,
    header: &[u8],
    text_capacity_per_row: usize,
    run_rows: RunRows,
    format_rows: FormatRows,
) -> Result<UnifiedBedScanSummary, String>
where
    RunRows: Fn(&[f32], usize, &mut [f64]),
    FormatRows: Fn(&mut String, &StreamingChunk, usize),
{
    let total_t0 = Instant::now();
    let count_nanos = AtomicU64::new(0);
    let meta_nanos = AtomicU64::new(0);
    let decode_nanos = AtomicU64::new(0);
    let proj_nanos = AtomicU64::new(0);
    let assoc_nanos = AtomicU64::new(0);
    let tsv_nanos = AtomicU64::new(0);

    let full_samples = gfcore::read_fam(bed_prefix)?;
    let n_samples_full = full_samples.len();
    if n_samples_full == 0 {
        return Err("no samples in PLINK FAM".to_string());
    }

    let sample_idx: Vec<usize> = match &sample_ids {
        Some(ids) => {
            let id_to_idx: std::collections::HashMap<&str, usize> = full_samples
                .iter()
                .enumerate()
                .map(|(i, name)| (name.as_str(), i))
                .collect();
            ids.iter()
                .map(|sid| {
                    id_to_idx
                        .get(sid.as_str())
                        .copied()
                        .ok_or_else(|| format!("sample '{sid}' not found in PLINK FAM"))
                })
                .collect::<Result<Vec<usize>, String>>()?
        }
        None => (0..n_samples_full).collect(),
    };
    if sample_idx.len() != n {
        return Err(format!(
            "sample_ids length {} != expected sample count {n}",
            sample_idx.len()
        ));
    }

    let bytes_per_snp = (n_samples_full + 3) / 4;
    if bytes_per_snp == 0 {
        return Err("bytes_per_snp is zero".to_string());
    }
    let bed_path = format!("{bed_prefix}.bed");
    let mut bed_window = if let Some(window_mb) = mmap_window_mb.filter(|&v| v > 0) {
        Some(WindowedBedMatrix::open(bed_prefix, window_mb)?)
    } else {
        None
    };
    let full_mmap = if bed_window.is_none() {
        let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
        if mmap.len() < 3 || mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err("only SNP-major BED supported".to_string());
        }
        let data_len = mmap.len() - 3;
        if data_len % bytes_per_snp != 0 {
            return Err(format!(
                "BED payload length {data_len} not a multiple of {bytes_per_snp}"
            ));
        }
        Some(mmap)
    } else {
        None
    };
    let n_snps = if let Some(window) = bed_window.as_ref() {
        window.n_source_snps()
    } else {
        let mmap = full_mmap
            .as_ref()
            .ok_or_else(|| "internal error: missing BED source".to_string())?;
        (mmap.len() - 3) / bytes_per_snp
    };
    let mut bim_reader = BimChunkReader::open(bed_prefix)?;
    let prepared_source_rows = if let Some(meta) = prepared_meta {
        let m = meta.row_indices.len();
        if meta.row_flip.len() != m || meta.row_maf.len() != m || meta.miss_counts.len() != m {
            return Err(format!(
                "prepared row metadata length mismatch: row_indices={m}, row_flip={}, row_maf={}, row_missing={}",
                meta.row_flip.len(),
                meta.row_maf.len(),
                meta.miss_counts.len(),
            ));
        }
        let source_rows = parse_index_vec_i64(meta.row_indices.as_slice(), n_snps, "row_indices")
            .map_err(|e| e.to_string())?;
        let mut prev_src: Option<usize> = None;
        for &src_row in source_rows.iter() {
            if let Some(prev) = prev_src {
                if src_row < prev {
                    return Err(
                        "prepared row_indices must be sorted in ascending BED order".to_string()
                    );
                }
            }
            prev_src = Some(src_row);
        }
        Some(source_rows)
    } else {
        None
    };
    let total_scan_units = prepared_source_rows
        .as_ref()
        .map(|rows| rows.len())
        .unwrap_or(n_snps);

    let decode_plan = AdditiveDecodePlan::from_sample_indices(n_samples_full, &sample_idx);
    let sample_identity = decode_plan.sample_identity();
    let use_selected = !sample_identity;
    let selected_excluded_sample_indices = decode_plan.selected_excluded_sample_indices();
    let sample_subset_plan = decode_plan.subset_decode_plan();
    let dense_subset_pos = decode_plan.dense_subset_pos();

    let proj_threads = stage_proj_threads_or(threads);
    let assoc_threads = stage_assoc_threads_or(threads, proj_threads);
    let proj_pool = get_cached_pool(proj_threads).map_err(|e| format!("{e}"))?;

    let writer = AsyncTsvWriter::with_config(out_tsv, header, 64 * 1024 * 1024, 16)?;

    let scan_chunk_snps = rotate_block_rows.max(1).min(total_scan_units.max(1));
    let mut text_buf = String::with_capacity(scan_chunk_snps * text_capacity_per_row);
    let mut total_rows = 0usize;

    let progress_block = if progress_every == 0 {
        0usize
    } else {
        progress_every.max(1)
    };
    let mut next_progress_emit = if progress_block > 0 {
        progress_block.min(total_scan_units).max(1)
    } else {
        0
    };

    let mut chunk_start = 0usize;
    let mut rel_row_indices: Vec<usize> = Vec::with_capacity(scan_chunk_snps);
    let producer_err = Arc::new(OnceLock::<String>::new());
    let producer_err_bg = Arc::clone(&producer_err);
    let producer = |chunk: &mut StreamingChunk| -> bool {
        if chunk_start >= total_scan_units {
            return false;
        }
        chunk.clear();
        let chunk_end = (chunk_start + scan_chunk_snps).min(total_scan_units);
        let prepared_batch =
            prepared_source_rows
                .as_ref()
                .zip(prepared_meta)
                .map(|(source_rows_all, meta)| {
                    (
                        &source_rows_all[chunk_start..chunk_end],
                        &meta.row_flip[chunk_start..chunk_end],
                        &meta.row_maf[chunk_start..chunk_end],
                        &meta.miss_counts[chunk_start..chunk_end],
                    )
                });
        let prepared_src_start = prepared_batch
            .as_ref()
            .and_then(|(source_rows, _, _, _)| source_rows.first().copied());
        let prepared_uses_window = prepared_batch.is_some() && bed_window.is_some();
        let chunk_packed = if let Some((source_rows, _, _, _)) = prepared_batch.as_ref() {
            if source_rows.is_empty() {
                chunk.scanned_to = chunk_end;
                chunk_start = chunk_end;
                return chunk_start < total_scan_units;
            }
            match bed_window.as_mut() {
                Some(window) => match window.prepare_source_rows(source_rows, &mut rel_row_indices)
                {
                    Ok(slice) => slice,
                    Err(e) => {
                        let _ = producer_err_bg.set(e);
                        return false;
                    }
                },
                None => {
                    let mmap = match full_mmap.as_ref() {
                        Some(mmap) => mmap,
                        None => {
                            let _ = producer_err_bg
                                .set("internal error: missing full BED mmap".to_string());
                            return false;
                        }
                    };
                    let packed = &mmap[3..];
                    let src_start = source_rows[0];
                    let src_end = source_rows[source_rows.len() - 1] + 1;
                    let start_byte = src_start * bytes_per_snp;
                    let end_byte = src_end * bytes_per_snp;
                    &packed[start_byte..end_byte]
                }
            }
        } else {
            match bed_window.as_mut() {
                Some(window) => match window.read_source_range(chunk_start, chunk_end) {
                    Ok(slice) => slice,
                    Err(e) => {
                        let _ = producer_err_bg.set(e);
                        return false;
                    }
                },
                None => {
                    let mmap = match full_mmap.as_ref() {
                        Some(mmap) => mmap,
                        None => {
                            let _ = producer_err_bg
                                .set("internal error: missing full BED mmap".to_string());
                            return false;
                        }
                    };
                    let packed = &mmap[3..];
                    let start_byte = chunk_start * bytes_per_snp;
                    let end_byte = chunk_end * bytes_per_snp;
                    &packed[start_byte..end_byte]
                }
            }
        };
        let chunk_sites = if let Some((source_rows, _, _, _)) = prepared_batch.as_ref() {
            match bim_reader.read_selected_rows(source_rows) {
                Ok(sites) => sites,
                Err(e) => {
                    let _ = producer_err_bg.set(e);
                    return false;
                }
            }
        } else {
            match bim_reader.read_range(chunk_start, chunk_end) {
                Ok(sites) => sites,
                Err(e) => {
                    let _ = producer_err_bg.set(e);
                    return false;
                }
            }
        };

        if let Some((source_rows, row_flip_batch, row_maf_batch, miss_count_batch)) =
            prepared_batch.as_ref()
        {
            let src_start = prepared_src_start.unwrap_or(0usize);
            for (offset, site) in chunk_sites.into_iter().enumerate() {
                let snp_name = resolve_snp_name(&site.snp, &site.chrom, site.pos);
                let local_src = if prepared_uses_window {
                    rel_row_indices[offset]
                } else {
                    source_rows[offset].saturating_sub(src_start)
                };
                chunk.indices.push(local_src);
                chunk.flip.push(row_flip_batch[offset]);
                chunk.maf.push(row_maf_batch[offset]);
                chunk.miss_rate.push(0.0_f32);
                chunk.miss_block[chunk.rows] = miss_count_batch[offset];
                chunk.chrom.push(site.chrom);
                chunk.pos.push(site.pos as i64);
                chunk.snp.push(snp_name);
                chunk.a0.push(site.ref_allele);
                chunk.a1.push(site.alt_allele);
                chunk.rows += 1;
            }
        } else {
            let count_t0 = Instant::now();
            let counts: Vec<Option<SnpCounts>> = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|snp_idx| {
                    let local_idx = snp_idx - chunk_start;
                    let row =
                        &chunk_packed[local_idx * bytes_per_snp..(local_idx + 1) * bytes_per_snp];
                    let (missing, het, hom_alt) = if use_selected {
                        count_packed_row_counts_selected_with_excluded(
                            row,
                            n_samples_full,
                            &sample_idx,
                            selected_excluded_sample_indices,
                        )
                    } else {
                        count_packed_row_counts(row, n_samples_full)
                    };
                    let non_missing = n.saturating_sub(missing);

                    let miss_rate = if n > 0 {
                        missing as f32 / n as f32
                    } else {
                        1.0_f32
                    };
                    if miss_rate > miss_thr {
                        return None;
                    }

                    if non_missing == 0 {
                        return if maf_thr > 0.0 {
                            None
                        } else {
                            Some(SnpCounts {
                                flip: false,
                                maf: 0.0_f32,
                                miss_rate,
                                missing_count: missing,
                            })
                        };
                    }

                    if het_thr > 0.0 {
                        let het_rate = het as f32 / non_missing as f32;
                        if het_rate > het_thr {
                            return None;
                        }
                    }

                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                    let alt_freq = alt_sum as f32 / (2.0_f32 * non_missing as f32);
                    let maf_v = alt_freq.min(1.0_f32 - alt_freq);
                    if maf_v < maf_thr {
                        return None;
                    }

                    Some(SnpCounts {
                        flip: false,
                        maf: alt_freq,
                        miss_rate,
                        missing_count: missing,
                    })
                })
                .collect();
            record_elapsed_nanos(&count_nanos, count_t0);

            let meta_t0 = Instant::now();
            for (offset, (cnts_opt, site)) in
                counts.into_iter().zip(chunk_sites.into_iter()).enumerate()
            {
                let cnts = match cnts_opt {
                    Some(c) => c,
                    None => continue,
                };
                if snps_only
                    && (!is_simple_snp_allele(&site.ref_allele)
                        || !is_simple_snp_allele(&site.alt_allele))
                {
                    continue;
                }
                let snp_name = resolve_snp_name(&site.snp, &site.chrom, site.pos);
                chunk.indices.push(offset);
                chunk.flip.push(cnts.flip);
                chunk.maf.push(cnts.maf);
                chunk.miss_rate.push(cnts.miss_rate);
                chunk.miss_block[chunk.rows] = cnts.missing_count;
                chunk.chrom.push(site.chrom);
                chunk.pos.push(site.pos as i64);
                chunk.snp.push(snp_name);
                chunk.a0.push(site.ref_allele);
                chunk.a1.push(site.alt_allele);
                chunk.rows += 1;
            }
            record_elapsed_nanos(&meta_nanos, meta_t0);
        }

        if chunk.rows > 0 {
            let decode_t0 = Instant::now();
            decode_centered_block_packed_model_f32(
                chunk_packed,
                bytes_per_snp,
                &chunk.flip,
                &chunk.maf,
                Some(&chunk.indices),
                0,
                chunk.rows,
                if sample_identity { n_samples_full } else { n },
                gm,
                sample_identity,
                sample_subset_plan,
                dense_subset_pos,
                &mut chunk.g_block[..chunk.rows * n],
            );
            record_elapsed_nanos(&decode_nanos, decode_t0);
        }

        chunk.scanned_to = chunk_end;
        chunk_start = chunk_end;
        chunk_start < total_scan_units
    };

    let consumer = |chunk: &mut StreamingChunk| {
        let rows = chunk.rows;
        if rows > 0 {
            let proj_t0 = Instant::now();
            rotate_snp_block_with_ut_blas(
                &chunk.g_block[..rows * n],
                rows,
                n,
                ut_flat,
                &mut chunk.rot_block[..rows * n],
                proj_threads,
                proj_pool.as_ref(),
            );
            record_elapsed_nanos(&proj_nanos, proj_t0);
            let assoc_t0 = Instant::now();
            run_rows(
                &chunk.rot_block[..rows * n],
                rows,
                &mut chunk.out_block[..rows * out_cols],
            );
            record_elapsed_nanos(&assoc_nanos, assoc_t0);

            let tsv_t0 = Instant::now();
            text_buf.clear();
            if text_buf.capacity() < rows * text_capacity_per_row {
                text_buf.reserve(rows * text_capacity_per_row - text_buf.capacity());
            }
            format_rows(&mut text_buf, chunk, rows);
            let payload = std::mem::take(&mut text_buf).into_bytes();
            writer.send(payload)?;
            record_elapsed_nanos(&tsv_nanos, tsv_t0);
        }

        total_rows += rows;
        if progress_block > 0 && chunk.scanned_to >= next_progress_emit {
            if let Some(ref cb) = progress_callback {
                let _ = Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(
                        py2,
                        (chunk.scanned_to.min(total_scan_units), total_scan_units),
                    )?;
                    Ok(())
                });
            }
            next_progress_emit = (chunk.scanned_to / progress_block + 1)
                .saturating_mul(progress_block)
                .min(total_scan_units);
        }

        Ok::<(), String>(())
    };

    let pipeline_t0 = Instant::now();
    crate::pipeline::run_double_buffer(
        2,
        || StreamingChunk::new(scan_chunk_snps, n, out_cols),
        producer,
        consumer,
    )?;
    let pipeline_secs = pipeline_t0.elapsed().as_secs_f64();
    if let Some(err) = producer_err.get() {
        return Err(err.clone());
    }
    if prepared_meta.is_none() {
        bim_reader.ensure_exhausted(n_snps)?;
    }

    if let Some(ref cb) = progress_callback {
        let _ = Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (total_scan_units, total_scan_units))?;
            Ok(())
        });
    }

    let writer_t0 = Instant::now();
    writer.finish()?;
    record_elapsed_nanos(&tsv_nanos, writer_t0);

    Ok(UnifiedBedScanSummary {
        total_rows,
        total_secs: total_t0.elapsed().as_secs_f64(),
        pipeline_secs,
        count_secs: elapsed_nanos_to_secs(&count_nanos),
        meta_secs: elapsed_nanos_to_secs(&meta_nanos),
        decode_secs: elapsed_nanos_to_secs(&decode_nanos),
        proj_secs: elapsed_nanos_to_secs(&proj_nanos),
        assoc_secs: elapsed_nanos_to_secs(&assoc_nanos),
        tsv_secs: elapsed_nanos_to_secs(&tsv_nanos),
        n_snps: total_scan_units,
        n,
        scan_chunk_snps,
        proj_threads,
        assoc_threads,
    })
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, snp_chunk, u_t, max_iter=50, tol=1e-2, threads=0, nullml=None, rotate_block_rows=256))]
pub fn lmm_reml_chunk_from_snp_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m_chunk, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }

    let m_chunk = snp_arr.shape()[0];
    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let beta_se_p = PyArray2::<f64>::zeros(py, [m_chunk, out_cols], false).into_bound();
    let beta_se_p_slice: &mut [f64] = unsafe {
        beta_se_p
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("beta_se_p not contiguous"))?
    };

    // Keep same behavior as lmm_reml_chunk_f32: dedicated pool can be faster here.
    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    let block_rows = rotate_block_rows.max(1);

    py.detach(|| {
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            run_rotated_reml_assoc_block_f32(
                g_block,
                out_block.len() / out_cols,
                n,
                s,
                &xcov_flat,
                y,
                p_cov,
                low,
                high,
                tol,
                max_iter,
                None,
                false,
                false,
                if with_plrt { Some(nullml_val) } else { None },
                out_block,
                out_cols,
                None,
            );
        };

        let mut start = 0usize;
        while start < m_chunk {
            let rows = (m_chunk - start).min(block_rows);
            let snp_block = &snp_flat[start * n..(start + rows) * n];
            let rot_block = &mut rot_buf[..rows * n];
            let rotate_tile_rows = choose_rotate_tile_rows(
                rows,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                },
            );

            let out_block = &mut beta_se_p_slice[start * out_cols..(start + rows) * out_cols];
            if let Some(tp) = &pool {
                tp.install(|| {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                    run_rows(rot_block, out_block);
                });
            } else {
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    rotate_tile_rows,
                );
                run_rows(rot_block, out_block);
            }
            start += rows;
        }
    });

    Ok(beta_se_p)
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, snp_chunk, u_t, nullml, max_iter=50, tol=1e-2, threads=0, rotate_block_rows=256))]
pub fn lmm_reml_lmm2_chunk_from_snp_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    nullml: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m_chunk, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !nullml.is_finite() {
        return Err(PyRuntimeError::new_err("nullml must be finite"));
    }

    let m_chunk = snp_arr.shape()[0];
    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let out_cols = 6usize;
    let out = PyArray2::<f64>::zeros(py, [m_chunk, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("lmm2 output not contiguous"))?
    };

    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    let block_rows = rotate_block_rows.max(1);

    py.detach(|| {
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            run_rotated_lmm2_assoc_block_f32(
                g_block,
                out_block.len() / out_cols,
                n,
                s,
                &xcov_flat,
                y,
                p_cov,
                low,
                high,
                tol,
                max_iter,
                None,
                None,
                false,
                nullml,
                out_block,
                out_cols,
                None,
            );
        };

        let mut start = 0usize;
        while start < m_chunk {
            let rows = (m_chunk - start).min(block_rows);
            let snp_block = &snp_flat[start * n..(start + rows) * n];
            let rot_block = &mut rot_buf[..rows * n];
            let rotate_tile_rows = choose_rotate_tile_rows(
                rows,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                },
            );
            let out_block = &mut out_slice[start * out_cols..(start + rows) * out_cols];
            if let Some(tp) = &pool {
                tp.install(|| {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                    run_rows(rot_block, out_block);
                });
            } else {
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    rotate_tile_rows,
                );
                run_rows(rot_block, out_block);
            }
            start += rows;
        }
    });

    Ok(out)
}

// ------------------------------------------------------------
// Helpers: dot loops
// ------------------------------------------------------------

#[inline]
fn dot_loop(a: &[f64], b: &[f64]) -> f64 {
    dot_loop_simd(a, b)
}

#[inline]
fn dot_loop_unrolled(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut i = 0usize;
    while i + 3 < n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = s0 + s1 + s2 + s3;
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_loop_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0usize;
    let mut acc: __m256d = _mm256_setzero_pd();
    while i + 4 <= n {
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };
        acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
        i += 4;
    }
    let mut lanes = [0.0_f64; 4];
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    let mut s = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_loop_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0usize;
    let mut acc = vdupq_n_f64(0.0);
    while i + 2 <= n {
        let va = unsafe { vld1q_f64(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_f64(b.as_ptr().add(i)) };
        acc = vaddq_f64(acc, vmulq_f64(va, vb));
        i += 2;
    }
    let mut lanes = [0.0_f64; 2];
    unsafe { vst1q_f64(lanes.as_mut_ptr(), acc) };
    let mut s = lanes[0] + lanes[1];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[inline]
fn dot_loop_simd(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_loop_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_loop_neon(a, b) };
    }
    #[allow(unreachable_code)]
    dot_loop_unrolled(a, b)
}

struct AssocScratch {
    c: Vec<f64>,       // len p
    a_inv_c: Vec<f64>, // len p
}
impl AssocScratch {
    fn new(p: usize) -> Self {
        Self {
            c: vec![0.0; p],
            a_inv_c: vec![0.0; p],
        }
    }
    #[inline]
    fn reset(&mut self) {
        self.c.fill(0.0);
    }
}

fn stage_proj_threads_or(requested_threads: usize) -> usize {
    env_positive_usize("JX_FVLMM_PROJ_THREADS")
        .or_else(|| env_positive_usize("JX_MLM_RUST_THREADS"))
        .unwrap_or(requested_threads)
}

#[inline]
fn normalize_default_assoc_threads_for_backend(
    backend_tag: &str,
    proj_threads: usize,
    assoc_threads: usize,
) -> usize {
    let proj = proj_threads.max(1);
    let assoc = assoc_threads.max(1);
    if backend_tag == "openblas" {
        assoc.min(proj)
    } else {
        assoc
    }
}

#[inline]
fn stage_assoc_threads_or(requested_threads: usize, proj_threads: usize) -> usize {
    if let Some(explicit) = env_positive_usize("JX_FVLMM_ASSOC_THREADS") {
        return explicit;
    }
    let default_threads = env_positive_usize("JX_MLM_RUST_THREADS").unwrap_or(requested_threads);
    normalize_default_assoc_threads_for_backend(
        rust_sgemm_backend_tag(),
        proj_threads,
        default_threads,
    )
}

#[inline]
fn missing_count_from_rate(v: f32, n_samples: usize) -> usize {
    if !v.is_finite() || v <= 0.0 {
        0usize
    } else {
        ((v as f64) * (n_samples as f64)).round().max(0.0) as usize
    }
}

#[inline]
fn missing_rate_from_count(v: usize, n_samples: usize) -> f32 {
    if n_samples == 0 {
        0.0_f32
    } else {
        (v as f32) / (n_samples as f32)
    }
}

#[inline]
fn resolve_snp_name(snp: &str, chrom: &str, pos: i32) -> String {
    if !snp.is_empty() && snp != "." {
        snp.to_string()
    } else {
        format!("{chrom}_{pos}")
    }
}

fn fill_packed_missing_block(
    miss_sub: &mut [usize],
    sample_identity: bool,
    row_missing: &[f32],
    row_idx: Option<&[usize]>,
    start: usize,
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    sample_idx: &[usize],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    if sample_identity {
        if let Some(tp) = pool {
            tp.install(|| {
                miss_sub.par_iter_mut().enumerate().for_each(|(off, dst)| {
                    let idx = start + off;
                    // row_missing is aligned to the active/output row order; only packed
                    // row fetches should use row_idx indirection.
                    *dst = missing_count_from_rate(row_missing[idx], n_samples);
                });
            });
        } else {
            miss_sub.iter_mut().enumerate().for_each(|(off, dst)| {
                let idx = start + off;
                *dst = missing_count_from_rate(row_missing[idx], n_samples);
            });
        }
        return;
    }

    if let Some(tp) = pool {
        tp.install(|| {
            miss_sub.par_iter_mut().enumerate().for_each(|(off, dst)| {
                let idx = start + off;
                let src_row = row_idx.map(|v| v[idx]).unwrap_or(idx);
                let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                *dst = packed_row_missing_count_selected(row, n_samples, sample_idx);
            });
        });
    } else {
        miss_sub.iter_mut().enumerate().for_each(|(off, dst)| {
            let idx = start + off;
            let src_row = row_idx.map(|v| v[idx]).unwrap_or(idx);
            let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
            *dst = packed_row_missing_count_selected(row, n_samples, sample_idx);
        });
    }
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd, g_rot_chunk, threads=0, nullml=None))]
pub fn lmm_assoc_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    g_rot_chunk: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let g_arr = g_rot_chunk.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("g_rot_chunk must be (m, n)"));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let m = g_arr.shape()[0];
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound(); // beta, se, p, (plrt)
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    // Use contiguous slice directly (no copy)
    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let p = p_cov;

    // Build weights W = V^{-1} = 1/(s + lbd) (store as f32 to reduce bandwidth)
    let mut w = vec![0.0_f32; n];
    let mut log_det_v = 0.0_f64;
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return Err(PyRuntimeError::new_err("non-positive s[i]+lbd"));
        }
        w[i] = (1.0 / vv) as f32;
        log_det_v += vv.ln();
    }

    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    // Precompute A = X'WX, b = X'Wy, yWy
    let mut a = vec![0.0_f64; p * p];
    let mut b = vec![0.0_f64; p];
    let mut ywy = 0.0_f64;

    for i in 0..n {
        let wi = w[i] as f64;
        let yi = y[i];
        ywy += wi * yi * yi;

        let base = i * p;
        for r in 0..p {
            let xir = xcov_slice[base + r];
            b[r] += wi * xir * yi;
            for c in 0..=r {
                let xic = xcov_slice[base + c];
                a[r * p + c] += wi * xir * xic;
            }
        }
    }

    let ridge = 1e-6;
    for r in 0..p {
        a[r * p + r] += ridge;
        for c in 0..r {
            let vrc = a[r * p + c];
            a[c * p + r] = vrc;
        }
    }

    if cholesky_inplace(&mut a, p).is_none() {
        return Err(PyRuntimeError::new_err("X'WX not SPD"));
    }

    let mut a_inv_b = vec![0.0_f64; p];
    cholesky_solve_into(&a, p, &b, &mut a_inv_b);

    // b'A^{-1}b is constant
    let b_aib = dot_loop(&b, &a_inv_b);

    let df = (n as i32) - (p as i32) - 1;
    if df <= 0 {
        return Err(PyRuntimeError::new_err("df <= 0"));
    }

    // Thread pool
    let pool = get_cached_pool(threads)?;

    py.detach(|| {
        let mut run = || {
            out_slice
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || AssocScratch::new(p),
                    |scr, (idx, out_row)| {
                        scr.reset();

                        // c = X'Wg , d = g'Wg, e = g'Wy
                        let mut d = 0.0_f64;
                        let mut e = 0.0_f64;

                        for i in 0..n {
                            let gi = g_arr[(idx, i)] as f64;
                            let wi = w[i] as f64;
                            let yi = y[i];
                            let base = i * p;

                            let wgi = wi * gi;
                            d += wgi * gi;
                            e += wgi * yi;

                            // c[r] += wgi * xcov[i, r]
                            for r in 0..p {
                                scr.c[r] += wgi * xcov_slice[base + r];
                            }
                        }

                        cholesky_solve_into(&a, p, &scr.c, &mut scr.a_inv_c);

                        // ct_aic = c' A^{-1} c
                        let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                        let schur = d - ct_aic;
                        if schur <= 1e-12 || !schur.is_finite() {
                            out_row[0] = f64::NAN;
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        // ct_aib = c' A^{-1} b, computed in one loop without dot().
                        let mut ct_aib = 0.0_f64;
                        for r in 0..p {
                            ct_aib += scr.c[r] * a_inv_b[r];
                        }

                        let num = e - ct_aib;
                        let beta_g = num / schur;

                        // rWr = yWy - [ b'A^{-1}b + (e - c'A^{-1}b)^2 / schur ]
                        let q = b_aib + (num * num) / schur;
                        let rwr = (ywy - q).max(0.0);
                        let sigma2 = rwr / (df as f64);

                        let se_g = (sigma2 / schur).sqrt();
                        let pval = if se_g.is_finite() && se_g > 0.0 && beta_g.is_finite() {
                            let z = (beta_g / se_g).abs();
                            (2.0 * normal_sf(z)).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta_g;
                        out_row[1] = se_g;
                        out_row[2] = pval;
                        if with_plrt {
                            let ml = if rwr > 0.0 && rwr.is_finite() {
                                let total_log = n_f * rwr.ln() + log_det_v;
                                c_ml - 0.5 * total_log
                            } else {
                                f64::NAN
                            };
                            let mut stat = if ml.is_finite() {
                                2.0 * (ml - nullml_val)
                            } else {
                                0.0
                            };
                            if !stat.is_finite() || stat < 0.0 {
                                stat = 0.0;
                            }
                            out_row[3] = chi2_sf_df1(stat);
                        }
                    },
                );
        };

        if let Some(tp) = &pool {
            tp.install(run);
        } else {
            run();
        }
    });

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, log10_lbd, snp_chunk, u_t, threads=0, nullml=None, rotate_block_rows=512))]
pub fn lmm_assoc_chunk_from_snp_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    log10_lbd: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    threads: usize,
    nullml: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let snp_arr = snp_chunk.as_array();
    let ut_arr = u_t.as_array();

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if snp_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("snp_chunk must be (m, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return Err(PyRuntimeError::new_err("invalid log10_lbd"));
    }

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };

    let m = snp_arr.shape()[0];
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("out not contiguous"))?
    };

    let xcov_slice: &[f64] = xcov
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("xcov must be contiguous (C-order)"))?;
    let snp_flat: Cow<[f32]> = match snp_chunk.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(snp_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };
    let p = p_cov;

    // Build weights W = V^{-1} = 1/(s + lbd) (store as f32 to reduce bandwidth)
    let mut w = vec![0.0_f32; n];
    let mut log_det_v = 0.0_f64;
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return Err(PyRuntimeError::new_err("non-positive s[i]+lbd"));
        }
        w[i] = (1.0 / vv) as f32;
        log_det_v += vv.ln();
    }

    let n_f = n as f64;
    let c_ml = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;

    // Precompute A = X'WX, b = X'Wy, yWy
    let mut a = vec![0.0_f64; p * p];
    let mut b = vec![0.0_f64; p];
    let mut ywy = 0.0_f64;

    for i in 0..n {
        let wi = w[i] as f64;
        let yi = y[i];
        ywy += wi * yi * yi;

        let base = i * p;
        for r in 0..p {
            let xir = xcov_slice[base + r];
            b[r] += wi * xir * yi;
            for c in 0..=r {
                let xic = xcov_slice[base + c];
                a[r * p + c] += wi * xir * xic;
            }
        }
    }

    let ridge = 1e-6;
    for r in 0..p {
        a[r * p + r] += ridge;
        for c in 0..r {
            let vrc = a[r * p + c];
            a[c * p + r] = vrc;
        }
    }

    if cholesky_inplace(&mut a, p).is_none() {
        return Err(PyRuntimeError::new_err("X'WX not SPD"));
    }

    let mut a_inv_b = vec![0.0_f64; p];
    cholesky_solve_into(&a, p, &b, &mut a_inv_b);

    // b'A^{-1}b is constant
    let b_aib = dot_loop(&b, &a_inv_b);

    let df = (n as i32) - (p as i32) - 1;
    if df <= 0 {
        return Err(PyRuntimeError::new_err("df <= 0"));
    }

    let pool = get_cached_pool(threads)?;
    let block_rows = rotate_block_rows.max(1);

    py.detach(|| {
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            out_block
                .par_chunks_mut(out_cols)
                .enumerate()
                .for_each_init(
                    || AssocScratch::new(p),
                    |scr, (idx, out_row)| {
                        scr.reset();
                        let row = &g_block[idx * n..(idx + 1) * n];

                        // c = X'Wg , d = g'Wg, e = g'Wy
                        let mut d = 0.0_f64;
                        let mut e = 0.0_f64;

                        for i in 0..n {
                            let gi = row[i] as f64;
                            let wi = w[i] as f64;
                            let yi = y[i];
                            let base = i * p;

                            let wgi = wi * gi;
                            d += wgi * gi;
                            e += wgi * yi;

                            for r in 0..p {
                                scr.c[r] += wgi * xcov_slice[base + r];
                            }
                        }

                        cholesky_solve_into(&a, p, &scr.c, &mut scr.a_inv_c);
                        let ct_aic = dot_loop(&scr.c, &scr.a_inv_c);
                        let schur = d - ct_aic;
                        if schur <= 1e-12 || !schur.is_finite() {
                            out_row[0] = f64::NAN;
                            out_row[1] = f64::NAN;
                            out_row[2] = f64::NAN;
                            return;
                        }

                        let mut ct_aib = 0.0_f64;
                        for r in 0..p {
                            ct_aib += scr.c[r] * a_inv_b[r];
                        }

                        let num = e - ct_aib;
                        let beta_g = num / schur;

                        let q = b_aib + (num * num) / schur;
                        let rwr = (ywy - q).max(0.0);
                        let sigma2 = rwr / (df as f64);

                        let se_g = (sigma2 / schur).sqrt();
                        let pval = if se_g.is_finite() && se_g > 0.0 && beta_g.is_finite() {
                            let z = (beta_g / se_g).abs();
                            (2.0 * normal_sf(z)).clamp(f64::MIN_POSITIVE, 1.0)
                        } else {
                            1.0
                        };

                        out_row[0] = beta_g;
                        out_row[1] = se_g;
                        out_row[2] = pval;
                        if with_plrt {
                            let ml = if rwr > 0.0 && rwr.is_finite() {
                                let total_log = n_f * rwr.ln() + log_det_v;
                                c_ml - 0.5 * total_log
                            } else {
                                f64::NAN
                            };
                            let mut stat = if ml.is_finite() {
                                2.0 * (ml - nullml_val)
                            } else {
                                0.0
                            };
                            if !stat.is_finite() || stat < 0.0 {
                                stat = 0.0;
                            }
                            out_row[3] = chi2_sf_df1(stat);
                        }
                    },
                );
        };

        let mut start = 0usize;
        while start < m {
            let rows = (m - start).min(block_rows);
            let snp_block = &snp_flat[start * n..(start + rows) * n];
            let rot_block = &mut rot_buf[..rows * n];
            let rotate_tile_rows = choose_rotate_tile_rows(
                rows,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                },
            );

            let out_block = &mut out_slice[start * out_cols..(start + rows) * out_cols];
            if let Some(tp) = &pool {
                tp.install(|| {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                    run_rows(rot_block, out_block);
                });
            } else {
                rotate_snp_block_with_ut_parallel(
                    snp_block,
                    rows,
                    n,
                    &ut_flat,
                    rot_block,
                    rotate_tile_rows,
                );
                run_rows(rot_block, out_block);
            }
            start += rows;
        }
    });

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    bed_prefix,
    out_tsv,
    s, xcov, y_rot, u_t,
    maf_thr, miss_thr, het_thr,
    genetic_model = "add",
    snps_only = false,
    sample_ids = None,
    row_indices = None,
    row_flip = None,
    row_missing = None,
    row_maf = None,
    low = -5.0,
    high = 5.0,
    max_iter = 30,
    tol = 1e-2,
    threads = 0,
    nullml = None,
    init_log10_lbd = None,
    rotate_block_rows = 512,
    progress_callback = None,
    progress_every = 0,
    mmap_window_mb = None,
))]
pub fn lmm_reml_assoc_bed_to_tsv_f32<'py>(
    py: Python<'py>,
    bed_prefix: &str,
    out_tsv: &str,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    u_t: PyReadonlyArray2<'py, f32>,
    maf_thr: f32,
    miss_thr: f32,
    het_thr: f32,
    genetic_model: &str,
    snps_only: bool,
    sample_ids: Option<Vec<String>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    row_maf: Option<PyReadonlyArray1<'py, f32>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
    init_log10_lbd: Option<f64>,
    rotate_block_rows: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<usize> {
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }

    let s_slice = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let ut_arr = u_t.as_array();

    let n = y.len();
    let p = xcov_arr.shape()[1];
    if xcov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s_slice.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("u_t must be (n, n) row-major U^T"));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err("n must be > p+1"));
    }

    let gm = PackedGeneticModel::parse(genetic_model)?;
    let with_plrt = nullml.is_some();
    let out_cols = if with_plrt { 4usize } else { 3usize };
    let nullml_val = nullml.unwrap_or(0.0);
    let init_log10_lbd = init_log10_lbd
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(low, high));
    let prepared_meta = match (row_indices, row_flip, row_missing, row_maf) {
        (None, None, None, None) => None,
        (Some(row_idx), Some(row_flip), Some(row_missing), Some(row_maf)) => {
            let row_idx_vec = row_idx.as_slice()?.to_vec();
            let row_flip_vec = match row_flip.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_flip.as_array().iter().copied().collect(),
            };
            let row_missing_vec = match row_missing.as_slice() {
                Ok(slc) => slc
                    .iter()
                    .map(|&v| missing_count_from_rate(v, n))
                    .collect(),
                Err(_) => row_missing
                    .as_array()
                    .iter()
                    .copied()
                    .map(|v| missing_count_from_rate(v, n))
                    .collect(),
            };
            let row_maf_vec = match row_maf.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_maf.as_array().iter().copied().collect(),
            };
            Some(PreparedBedScanMeta {
                row_indices: row_idx_vec,
                row_flip: row_flip_vec,
                row_maf: row_maf_vec,
                miss_counts: row_missing_vec,
            })
        }
        _ => {
            return Err(PyRuntimeError::new_err(
                "prepared row metadata must provide all or none of: row_indices, row_flip, row_missing, row_maf",
            ))
        }
    };

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let bed_prefix_owned = bed_prefix.to_string();
    let out_tsv_owned = out_tsv.to_string();
    let stage_timing = env_truthy("JX_LMM_UNIFIED_STAGE_TIMING");
    let use_warm_start = !env_truthy("JX_LMM_UNIFIED_NO_WARM_START");

    py.detach(move || -> Result<usize, String> {
        let assoc_threads = stage_assoc_threads_or(threads, threads);
        let assoc_pool = get_cached_pool(assoc_threads).map_err(|e| format!("{e}"))?;
        let header: &[u8] = if with_plrt {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n"
        } else {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n"
        };

        let summary = run_unified_bed_scan_to_tsv_common(
            &bed_prefix_owned,
            &out_tsv_owned,
            &ut_flat,
            n,
            gm,
            sample_ids,
            maf_thr,
            miss_thr,
            het_thr,
            snps_only,
            threads,
            out_cols,
            rotate_block_rows,
            progress_callback,
            progress_every,
            mmap_window_mb,
            prepared_meta.as_ref(),
            header,
            128,
            |g_block, rows, out_block| {
                run_rotated_reml_assoc_block_f32(
                    g_block,
                    rows,
                    n,
                    s_slice,
                    &xcov_flat,
                    y,
                    p,
                    low,
                    high,
                    tol,
                    max_iter,
                    init_log10_lbd,
                    use_warm_start,
                    use_warm_start,
                    if with_plrt { Some(nullml_val) } else { None },
                    out_block,
                    out_cols,
                    assoc_pool.as_ref(),
                );
            },
            |text_buf, chunk, rows| {
                let miss_rate_buf: Vec<f32> = chunk.miss_block[..rows]
                    .iter()
                    .map(|&v| missing_rate_from_count(v, n))
                    .collect();
                append_assoc_block_from_arrays(
                    text_buf,
                    AssocResultLayout::ResultCols {
                        schema: if with_plrt {
                            AssocResultCols::Plrt4
                        } else {
                            AssocResultCols::Basic3
                        },
                        row_stride: out_cols,
                    },
                    "add",
                    AssocArraysBlock {
                        chrom: &chunk.chrom[..rows],
                        pos: &chunk.pos[..rows],
                        snp: &chunk.snp[..rows],
                        allele0: &chunk.a0[..rows],
                        allele1: &chunk.a1[..rows],
                        maf: &chunk.maf[..rows],
                        miss: AssocMissBlock::Rate(miss_rate_buf.as_slice()),
                    },
                    &chunk.out_block[..rows * out_cols],
                )
                .expect("LMM TSV formatting should be length-consistent");
            },
        )?;

        if stage_timing {
            let to_pct = |x: f64| -> f64 {
                if summary.total_secs > 0.0 {
                    x * 100.0 / summary.total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "LMM unified timing: count_qc={:.3}s ({:.1}%), metadata={:.3}s ({:.1}%), decode={:.3}s ({:.1}%), project={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), tsv={:.3}s ({:.1}%), pipeline_wall={:.3}s, total={:.3}s, rows={}, snps={}, n={}, chunk={}, proj_threads={}, assoc_threads={}",
                summary.count_secs,
                to_pct(summary.count_secs),
                summary.meta_secs,
                to_pct(summary.meta_secs),
                summary.decode_secs,
                to_pct(summary.decode_secs),
                summary.proj_secs,
                to_pct(summary.proj_secs),
                summary.assoc_secs,
                to_pct(summary.assoc_secs),
                summary.tsv_secs,
                to_pct(summary.tsv_secs),
                summary.pipeline_secs,
                summary.total_secs,
                summary.total_rows,
                summary.n_snps,
                summary.n,
                summary.scan_chunk_snps,
                summary.proj_threads,
                summary.assoc_threads,
            );
        }
        Ok(summary.total_rows)
    })
    .map_err(PyRuntimeError::new_err)
}

/// Single-entry Rust BED -> TSV path for exact LMM2 association.
///
/// This mirrors the unified LMM BED scan, but computes per-SNP REML
/// beta/se/pwald together with a second per-SNP ML optimization for
/// `lambda/ml/plrt`.
#[pyfunction]
#[pyo3(signature = (
    bed_prefix,
    out_tsv,
    s, xcov, y_rot, u_t,
    maf_thr, miss_thr, het_thr,
    genetic_model = "add",
    snps_only = false,
    sample_ids = None,
    row_indices = None,
    row_flip = None,
    row_missing = None,
    row_maf = None,
    low = -5.0,
    high = 5.0,
    max_iter = 30,
    tol = 1e-2,
    threads = 0,
    nullml = None,
    init_log10_lbd_reml = None,
    init_log10_lbd_ml = None,
    rotate_block_rows = 512,
    progress_callback = None,
    progress_every = 0,
    mmap_window_mb = None,
))]
pub fn lmm_reml_lmm2_assoc_bed_to_tsv_f32<'py>(
    py: Python<'py>,
    bed_prefix: &str,
    out_tsv: &str,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    u_t: PyReadonlyArray2<'py, f32>,
    maf_thr: f32,
    miss_thr: f32,
    het_thr: f32,
    genetic_model: &str,
    snps_only: bool,
    sample_ids: Option<Vec<String>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    row_maf: Option<PyReadonlyArray1<'py, f32>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    nullml: Option<f64>,
    init_log10_lbd_reml: Option<f64>,
    init_log10_lbd_ml: Option<f64>,
    rotate_block_rows: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<usize> {
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }

    let s_slice = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let ut_arr = u_t.as_array();

    let n = y.len();
    let p = xcov_arr.shape()[1];
    if xcov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s_slice.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("u_t must be (n, n) row-major U^T"));
    }
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err("n must be > p+1"));
    }

    let gm = PackedGeneticModel::parse(genetic_model)?;
    let out_cols = 6usize;
    let init_log10_lbd_reml = init_log10_lbd_reml
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(low, high));
    let init_log10_lbd_ml = init_log10_lbd_ml
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(low, high));
    let prepared_meta = match (row_indices, row_flip, row_missing, row_maf) {
        (None, None, None, None) => None,
        (Some(row_idx), Some(row_flip), Some(row_missing), Some(row_maf)) => {
            let row_idx_vec = row_idx.as_slice()?.to_vec();
            let row_flip_vec = match row_flip.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_flip.as_array().iter().copied().collect(),
            };
            let row_missing_vec = match row_missing.as_slice() {
                Ok(slc) => slc
                    .iter()
                    .map(|&v| missing_count_from_rate(v, n))
                    .collect(),
                Err(_) => row_missing
                    .as_array()
                    .iter()
                    .copied()
                    .map(|v| missing_count_from_rate(v, n))
                    .collect(),
            };
            let row_maf_vec = match row_maf.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_maf.as_array().iter().copied().collect(),
            };
            Some(PreparedBedScanMeta {
                row_indices: row_idx_vec,
                row_flip: row_flip_vec,
                row_maf: row_maf_vec,
                miss_counts: row_missing_vec,
            })
        }
        _ => {
            return Err(PyRuntimeError::new_err(
                "prepared row metadata must provide all or none of: row_indices, row_flip, row_missing, row_maf",
            ))
        }
    };

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let bed_prefix_owned = bed_prefix.to_string();
    let out_tsv_owned = out_tsv.to_string();
    let stage_timing =
        env_truthy("JX_LMM2_UNIFIED_STAGE_TIMING") || env_truthy("JX_LMM_UNIFIED_STAGE_TIMING");
    let use_warm_start = !(env_truthy("JX_LMM2_UNIFIED_NO_WARM_START")
        || env_truthy("JX_LMM_UNIFIED_NO_WARM_START"));

    py.detach(move || -> Result<usize, String> {
        let null_opt_t0 = Instant::now();
        let nullml_val = if let Some(v) = nullml {
            if !v.is_finite() {
                return Err("nullml must be finite when provided".to_string());
            }
            v
        } else {
            let init_guess = init_log10_lbd_ml.or(init_log10_lbd_reml);
            let (best_log10_lbd_ml, best_ml_cost) = brent_minimize_with_init(
                |x0| -ml_loglike(x0, s_slice, &xcov_flat, y, None, n, p),
                low,
                high,
                tol,
                max_iter,
                init_guess,
            );
            let mut ml0 = -best_ml_cost;
            if !ml0.is_finite() {
                ml0 = ml_loglike(best_log10_lbd_ml, s_slice, &xcov_flat, y, None, n, p);
            }
            if !ml0.is_finite() {
                return Err("failed to optimize null ML for LMM2 unified scan".to_string());
            }
            ml0
        };
        let null_opt_secs = null_opt_t0.elapsed().as_secs_f64();

        let header: &[u8] =
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tlambda\tml\tplrt\n";
        let assoc_threads = stage_assoc_threads_or(threads, threads);
        let assoc_pool = get_cached_pool(assoc_threads).map_err(|e| format!("{e}"))?;

        let summary = run_unified_bed_scan_to_tsv_common(
            &bed_prefix_owned,
            &out_tsv_owned,
            &ut_flat,
            n,
            gm,
            sample_ids,
            maf_thr,
            miss_thr,
            het_thr,
            snps_only,
            threads,
            out_cols,
            rotate_block_rows,
            progress_callback,
            progress_every,
            mmap_window_mb,
            prepared_meta.as_ref(),
            header,
            160,
            |g_block, rows, out_block| {
                run_rotated_lmm2_assoc_block_f32(
                    g_block,
                    rows,
                    n,
                    s_slice,
                    &xcov_flat,
                    y,
                    p,
                    low,
                    high,
                    tol,
                    max_iter,
                    init_log10_lbd_reml,
                    init_log10_lbd_ml,
                    use_warm_start,
                    nullml_val,
                    out_block,
                    out_cols,
                    assoc_pool.as_ref(),
                );
            },
            |text_buf, chunk, rows| {
                let miss_rate_buf: Vec<f32> = chunk.miss_block[..rows]
                    .iter()
                    .map(|&v| missing_rate_from_count(v, n))
                    .collect();
                append_assoc_block_from_arrays(
                    text_buf,
                    AssocResultLayout::ResultCols {
                        schema: AssocResultCols::Lmm2_6,
                        row_stride: out_cols,
                    },
                    "add",
                    AssocArraysBlock {
                        chrom: &chunk.chrom[..rows],
                        pos: &chunk.pos[..rows],
                        snp: &chunk.snp[..rows],
                        allele0: &chunk.a0[..rows],
                        allele1: &chunk.a1[..rows],
                        maf: &chunk.maf[..rows],
                        miss: AssocMissBlock::Rate(miss_rate_buf.as_slice()),
                    },
                    &chunk.out_block[..rows * out_cols],
                )
                .expect("LMM2 TSV formatting should be length-consistent");
            },
        )?;

        if stage_timing {
            let to_pct = |x: f64| -> f64 {
                if summary.total_secs > 0.0 {
                    x * 100.0 / summary.total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "LMM2 unified timing: null_ml_opt={:.3}s ({:.1}%), count_qc={:.3}s ({:.1}%), metadata={:.3}s ({:.1}%), decode={:.3}s ({:.1}%), project={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), tsv={:.3}s ({:.1}%), pipeline_wall={:.3}s, total={:.3}s, rows={}, snps={}, n={}, chunk={}, proj_threads={}, assoc_threads={}",
                null_opt_secs,
                to_pct(null_opt_secs),
                summary.count_secs,
                to_pct(summary.count_secs),
                summary.meta_secs,
                to_pct(summary.meta_secs),
                summary.decode_secs,
                to_pct(summary.decode_secs),
                summary.proj_secs,
                to_pct(summary.proj_secs),
                summary.assoc_secs,
                to_pct(summary.assoc_secs),
                summary.tsv_secs,
                to_pct(summary.tsv_secs),
                summary.pipeline_secs,
                summary.total_secs,
                summary.total_rows,
                summary.n_snps,
                summary.n,
                summary.scan_chunk_snps,
                summary.proj_threads,
                summary.assoc_threads,
            );
        }
        Ok(summary.total_rows)
    })
    .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    s,
    xcov,
    y_rot,
    u_t,
    sample_indices=None,
    row_indices=None,
    low=-5.0,
    high=5.0,
    max_iter=50,
    tol=1e-2,
    threads=0,
    model="add",
    progress_callback=None,
    progress_every=0,
    nullml=None,
    init_log10_lbd=None,
    rotate_block_rows=256
))]
pub fn lmm_reml_assoc_packed_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    u_t: PyReadonlyArray2<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    model: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    nullml: Option<f64>,
    init_log10_lbd: Option<f64>,
    rotate_block_rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }
    let gm = PackedGeneticModel::parse(model)?;
    let init_log10_lbd = init_log10_lbd
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(low, high));

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

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
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
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let y = y_rot.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y_rot must not be empty"));
    }
    let s = s.as_slice()?;
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        if sidx_slice.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                sidx_slice.len()
            )));
        }
        parse_index_vec_i64(sidx_slice, n_samples, "sample_indices")?
    } else {
        if n != n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "len(y_rot)={} must equal n_samples={} when sample_indices is not provided",
                n, n_samples
            )));
        }
        (0..n_samples).collect()
    };
    let decode_plan = AdditiveDecodePlan::from_sample_indices(n_samples, &sample_idx);
    let sample_identity = decode_plan.sample_identity();
    let sample_subset_plan = decode_plan.subset_decode_plan();
    let dense_subset_pos = decode_plan.dense_subset_pos();

    let xcov_arr = xcov.as_array();
    if xcov_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("xcov must be 2D (n, p_cov)"));
    }
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("xcov.n_rows must equal len(y_rot)"));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let ut_arr = u_t.as_array();
    if ut_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("u_t must be 2D (n, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4 } else { 3 };
    let out = PyArray2::<f64>::zeros(py, [m, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    let pool = get_cached_pool(threads)?;
    let block_rows = rotate_block_rows.max(1);
    let progress_block = if progress_every == 0 {
        block_rows.max(1)
    } else {
        progress_every.max(1)
    };
    let stage_timing = env_truthy("JX_LMM_PACKED_STAGE_TIMING");

    py.detach(move || -> PyResult<()> {
        let total_t0 = Instant::now();
        let mut decode_secs = 0.0_f64;
        let mut proj_secs = 0.0_f64;
        let mut assoc_secs = 0.0_f64;

        let mut snp_buf = vec![0.0_f32; block_rows * n];
        let mut rot_buf = vec![0.0_f32; block_rows * n];

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            run_rotated_reml_assoc_block_f32(
                g_block,
                out_block.len() / out_cols,
                n,
                s,
                &xcov_flat,
                y,
                p_cov,
                low,
                high,
                tol,
                max_iter,
                init_log10_lbd,
                true,
                true,
                if with_plrt { Some(nullml_val) } else { None },
                out_block,
                out_cols,
                pool.as_ref(),
            );
        };

        for row_start in (0..m).step_by(progress_block) {
            let row_end = (row_start + progress_block).min(m);
            let mut start = row_start;
            while start < row_end {
                let rows = (row_end - start).min(block_rows);
                let snp_block = &mut snp_buf[..rows * n];
                let rot_block = &mut rot_buf[..rows * n];

                let dt0 = Instant::now();
                decode_centered_block_packed_model_f32(
                    &packed_flat,
                    bytes_per_snp,
                    row_flip,
                    row_maf,
                    row_idx.as_deref(),
                    start,
                    rows,
                    n,
                    gm,
                    sample_identity,
                    sample_subset_plan,
                    dense_subset_pos,
                    snp_block,
                );
                decode_secs += dt0.elapsed().as_secs_f64();

                let rotate_tile_rows = choose_rotate_tile_rows(
                    rows,
                    if threads > 0 {
                        threads
                    } else {
                        rayon::current_num_threads()
                    },
                );
                let out_block = &mut out_slice[start * out_cols..(start + rows) * out_cols];

                let pt0 = Instant::now();
                if let Some(tp) = &pool {
                    tp.install(|| {
                        rotate_snp_block_with_ut_parallel(
                            snp_block,
                            rows,
                            n,
                            &ut_flat,
                            rot_block,
                            rotate_tile_rows,
                        );
                    });
                } else {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                }
                proj_secs += pt0.elapsed().as_secs_f64();

                let at0 = Instant::now();
                if let Some(tp) = &pool {
                    tp.install(|| run_rows(rot_block, out_block));
                } else {
                    run_rows(rot_block, out_block);
                }
                assoc_secs += at0.elapsed().as_secs_f64();
                start += rows;
            }

            if let Some(cb) = progress_callback.as_ref() {
                let done = row_end;
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (done, m))?;
                    Ok(())
                })?;
            }
        }
        if stage_timing {
            let total_secs = total_t0.elapsed().as_secs_f64();
            let other_secs = (total_secs - decode_secs - proj_secs - assoc_secs).max(0.0_f64);
            let to_pct = |x: f64| -> f64 {
                if total_secs > 0.0 {
                    x * 100.0 / total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "LMM packed timing: decode={:.3}s ({:.1}%), proj={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, threads={}",
                decode_secs,
                to_pct(decode_secs),
                proj_secs,
                to_pct(proj_secs),
                assoc_secs,
                to_pct(assoc_secs),
                other_secs,
                to_pct(other_secs),
                total_secs,
                m,
                n,
                if threads > 0 { threads } else { rayon::current_num_threads() }
            );
        }
        Ok(())
    })?;

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_missing,
    s,
    xcov,
    y_rot,
    u_t,
    chrom,
    pos,
    snp,
    allele0,
    allele1,
    out_tsv,
    sample_indices=None,
    row_indices=None,
    low=-5.0,
    high=5.0,
    max_iter=50,
    tol=1e-2,
    threads=0,
    model="add",
    progress_callback=None,
    progress_every=0,
    nullml=None,
    init_log10_lbd=None,
    rotate_block_rows=256,
    bed_prefix=None
))]
pub fn lmm_reml_assoc_packed_f32_to_tsv<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_missing: PyReadonlyArray1<'py, f32>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    u_t: PyReadonlyArray2<'py, f32>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: &str,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    model: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    nullml: Option<f64>,
    init_log10_lbd: Option<f64>,
    rotate_block_rows: usize,
    bed_prefix: Option<&str>,
) -> PyResult<usize> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }
    let gm = PackedGeneticModel::parse(model)?;
    let init_log10_lbd = init_log10_lbd
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(low, high));

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
    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;
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
    if row_flip.len() != m || row_maf.len() != m || row_missing.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf/row_missing length mismatch with packed rows",
        ));
    }
    let (chrom, pos, snp, allele0, allele1) = resolve_assoc_tsv_metadata(
        bed_prefix,
        chrom,
        pos,
        snp,
        allele0,
        allele1,
        row_idx.as_deref(),
        m,
    )
    .map_err(PyRuntimeError::new_err)?;

    let y = y_rot.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y_rot must not be empty"));
    }
    let s = s.as_slice()?;
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        let sidx_slice = sidx.as_slice()?;
        if sidx_slice.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                sidx_slice.len()
            )));
        }
        parse_index_vec_i64(sidx_slice, n_samples, "sample_indices")?
    } else {
        if n != n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "len(y_rot)={} must equal n_samples={} when sample_indices is not provided",
                n, n_samples
            )));
        }
        (0..n_samples).collect()
    };
    let decode_plan = AdditiveDecodePlan::from_sample_indices(n_samples, &sample_idx);
    let sample_identity = decode_plan.sample_identity();
    let sample_subset_plan = decode_plan.subset_decode_plan();
    let dense_subset_pos = decode_plan.dense_subset_pos();

    let xcov_arr = xcov.as_array();
    if xcov_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("xcov must be 2D (n, p_cov)"));
    }
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("xcov.n_rows must equal len(y_rot)"));
    }
    if n <= p_cov + 1 {
        return Err(PyRuntimeError::new_err("n must be > p_cov+1"));
    }

    let ut_arr = u_t.as_array();
    if ut_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("u_t must be 2D (n, n)"));
    }
    if ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err(
            "u_t must be (n, n) and row-major U^T",
        ));
    }

    let xcov_flat: Cow<[f64]> = match xcov.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(xcov_arr.iter().copied().collect()),
    };
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let ut_flat: Cow<[f32]> = match u_t.as_slice() {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(ut_arr.iter().copied().collect()),
    };

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let out_cols = if with_plrt { 4usize } else { 3usize };

    let pool = get_cached_pool(threads)?;
    let block_rows = rotate_block_rows.max(1);
    let progress_block = if progress_every == 0 {
        block_rows.max(1)
    } else {
        progress_every.max(1)
    };
    let stage_timing = env_truthy("JX_LMM_PACKED_STAGE_TIMING");
    let out_tsv_path = out_tsv.to_string();

    py.detach(move || -> PyResult<()> {
        let total_t0 = Instant::now();
        let mut decode_secs = 0.0_f64;
        let mut proj_secs = 0.0_f64;
        let mut assoc_secs = 0.0_f64;
        let mut tsv_secs = 0.0_f64;

        let writer = AsyncTsvWriter::with_config(
            &out_tsv_path,
            if with_plrt {
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n"
                    .as_slice()
            } else {
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n"
                    .as_slice()
            },
            64 * 1024 * 1024,
            4,
        )
        .map_err(PyRuntimeError::new_err)?;

        let mut snp_buf = vec![0.0_f32; block_rows * n];
        let mut rot_buf = vec![0.0_f32; block_rows * n];
        let mut out_block = vec![0.0_f64; block_rows * out_cols];
        let mut miss_block = vec![0usize; block_rows];
        let mut text_buf = String::with_capacity(block_rows * 128);

        let run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            run_rotated_reml_assoc_block_f32(
                g_block,
                out_block.len() / out_cols,
                n,
                s,
                &xcov_flat,
                y,
                p_cov,
                low,
                high,
                tol,
                max_iter,
                init_log10_lbd,
                true,
                true,
                if with_plrt { Some(nullml_val) } else { None },
                out_block,
                out_cols,
                pool.as_ref(),
            );
        };

        for row_start in (0..m).step_by(progress_block) {
            let row_end = (row_start + progress_block).min(m);
            let mut start = row_start;
            while start < row_end {
                let rows = (row_end - start).min(block_rows);
                let snp_block = &mut snp_buf[..rows * n];
                let rot_block = &mut rot_buf[..rows * n];

                let dt0 = Instant::now();
                decode_centered_block_packed_model_f32(
                    &packed_flat,
                    bytes_per_snp,
                    row_flip,
                    row_maf,
                    row_idx.as_deref(),
                    start,
                    rows,
                    n,
                    gm,
                    sample_identity,
                    sample_subset_plan,
                    dense_subset_pos,
                    snp_block,
                );
                decode_secs += dt0.elapsed().as_secs_f64();

                let miss_sub = &mut miss_block[..rows];
                fill_packed_missing_block(
                    miss_sub,
                    sample_identity,
                    row_missing,
                    row_idx.as_deref(),
                    start,
                    &packed_flat,
                    bytes_per_snp,
                    n_samples,
                    &sample_idx,
                    pool.as_ref(),
                );

                let rotate_tile_rows = choose_rotate_tile_rows(
                    rows,
                    if threads > 0 {
                        threads
                    } else {
                        rayon::current_num_threads()
                    },
                );
                let pt0 = Instant::now();
                if let Some(tp) = &pool {
                    tp.install(|| {
                        rotate_snp_block_with_ut_parallel(
                            snp_block,
                            rows,
                            n,
                            &ut_flat,
                            rot_block,
                            rotate_tile_rows,
                        );
                    });
                } else {
                    rotate_snp_block_with_ut_parallel(
                        snp_block,
                        rows,
                        n,
                        &ut_flat,
                        rot_block,
                        rotate_tile_rows,
                    );
                }
                proj_secs += pt0.elapsed().as_secs_f64();

                let out_sub = &mut out_block[..rows * out_cols];
                let at0 = Instant::now();
                if let Some(tp) = &pool {
                    tp.install(|| run_rows(rot_block, out_sub));
                } else {
                    run_rows(rot_block, out_sub);
                }
                assoc_secs += at0.elapsed().as_secs_f64();

                let tt0 = Instant::now();
                text_buf.clear();
                if text_buf.capacity() < rows * 128 {
                    text_buf.reserve(rows * 128 - text_buf.capacity());
                }
                let miss_rate_buf: Vec<f32> = miss_sub
                    .iter()
                    .map(|&v| missing_rate_from_count(v, n))
                    .collect();
                append_assoc_block_from_arrays(
                    &mut text_buf,
                    AssocResultLayout::ResultCols {
                        schema: if with_plrt {
                            AssocResultCols::Plrt4
                        } else {
                            AssocResultCols::Basic3
                        },
                        row_stride: out_cols,
                    },
                    "add",
                    AssocArraysBlock {
                        chrom: &chrom[start..start + rows],
                        pos: &pos[start..start + rows],
                        snp: &snp[start..start + rows],
                        allele0: &allele0[start..start + rows],
                        allele1: &allele1[start..start + rows],
                        maf: &row_maf[start..start + rows],
                        miss: AssocMissBlock::Rate(miss_rate_buf.as_slice()),
                    },
                    &out_sub[..rows * out_cols],
                )
                .map_err(PyRuntimeError::new_err)?;
                send_text_buf(&writer, &mut text_buf).map_err(PyRuntimeError::new_err)?;
                tsv_secs += tt0.elapsed().as_secs_f64();

                start += rows;
            }

            if let Some(cb) = progress_callback.as_ref() {
                let done = row_end;
                Python::attach(|py2| -> PyResult<()> {
                    py2.check_signals()?;
                    cb.call1(py2, (done, m))?;
                    Ok(())
                })?;
            } else {
                Python::attach(|py2| py2.check_signals())?;
            }
        }

        let wt0 = Instant::now();
        let writer_result = writer.finish().map_err(PyRuntimeError::new_err);
        tsv_secs += wt0.elapsed().as_secs_f64();
        writer_result?;

        if stage_timing {
            let total_secs = total_t0.elapsed().as_secs_f64();
            let other_secs =
                (total_secs - decode_secs - proj_secs - assoc_secs - tsv_secs).max(0.0_f64);
            let to_pct = |x: f64| -> f64 {
                if total_secs > 0.0 {
                    x * 100.0 / total_secs
                } else {
                    0.0
                }
            };
            eprintln!(
                "LMM packed timing: decode={:.3}s ({:.1}%), proj={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), tsv={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, threads={}",
                decode_secs,
                to_pct(decode_secs),
                proj_secs,
                to_pct(proj_secs),
                assoc_secs,
                to_pct(assoc_secs),
                tsv_secs,
                to_pct(tsv_secs),
                other_secs,
                to_pct(other_secs),
                total_secs,
                m,
                n,
                if threads > 0 {
                    threads
                } else {
                    rayon::current_num_threads()
                }
            );
        }
        Ok(())
    })?;

    Ok(m)
}
