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

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::assoc2tsv::{
    append_assoc_block_from_arrays, resolve_assoc_tsv_metadata, send_text_buf, AssocArraysBlock,
    AssocMissBlock, AssocResultCols, AssocResultLayout,
};
use crate::brent::brent_minimize_with_init;
use crate::decode::{
    decode_centered_block_packed_model_f32, AdditiveDecodePlan, PackedGeneticModel,
};
use crate::gfcore;
use crate::gfcore::BimChunkReader;
use crate::gfreader::{evaluate_packed_bed_row_qc, is_simple_snp_allele};
use crate::gload::WindowedBedMatrix;
use crate::kmer::KfileAssocSource;
use crate::linalg::{chi2_sf_df1, cholesky_gram_inplace, cholesky_solve_into, normal_sf};
use crate::lmm_common::{
    choose_rotate_tile_rows, dot_loop, elapsed_nanos_to_secs, fill_packed_missing_block,
    missing_count_from_rate, missing_rate_from_count, record_elapsed_nanos, resolve_snp_name,
    rotate_snp_block_with_ut_blas, rotate_snp_block_with_ut_parallel, stage_assoc_threads_or,
    stage_proj_threads_or, AssocScratch, SnpCounts, StreamingChunk,
};
use crate::reml::{
    final_beta_se_with_workspace, ml_loglike, ml_loglike_with_workspace,
    reml_brent_minimize_with_workspace, run_rotated_assoc_block_f32, MlWorkspace, RemlWorkspace,
};
use crate::stats_common::{env_truthy, get_cached_pool, parse_index_vec_i64, AsyncTsvWriter};
use memmap2::Mmap;

struct RotatedRemlAssocState {
    snp_vec: Vec<f64>,
    reml_workspace: RemlWorkspace,
    last_log10_lbd: Option<f64>,
}

const LMM_DETERMINISTIC_WARM_START_BLOCK_ROWS: usize = 16;

struct RotatedRemlWarmStartState {
    assoc: RotatedRemlAssocState,
    rows_in_group: usize,
}

impl RotatedRemlWarmStartState {
    fn new(n: usize, p_cov: usize) -> Self {
        Self {
            assoc: RotatedRemlAssocState {
                snp_vec: vec![0.0_f64; n],
                reml_workspace: RemlWorkspace::new(p_cov + 1),
                last_log10_lbd: None,
            },
            rows_in_group: 0,
        }
    }
}

struct RotatedLmm2AssocState {
    snp_vec: Vec<f64>,
    reml_workspace: RemlWorkspace,
    ml_workspace: MlWorkspace,
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

#[inline]
fn unified_lmm_reml_start_policy(no_warm_start: bool) -> (bool, bool) {
    // Reuse the previous marker's optimum within fixed-size groups. The scanner
    // carries a partial group's state across read chunks so `-mem` does not
    // change which marker receives which initial lambda.
    (!no_warm_start, !no_warm_start)
}

#[allow(clippy::too_many_arguments)]
fn process_rotated_reml_assoc_row(
    row: &[f32],
    s: &[f64],
    xcov_flat: &[f64],
    y: &[f64],
    n: usize,
    p_cov: usize,
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
    init_log10_lbd: Option<f64>,
    seed_with_init_guess: bool,
    carry_warm_start: bool,
    nullml_val: f64,
    with_plrt: bool,
    state: &mut RotatedRemlAssocState,
    out_row: &mut [f64],
) {
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
    let (best_log10_lbd, _best_cost, last_eval_log10_lbd) = reml_brent_minimize_with_workspace(
        low,
        high,
        tol,
        max_iter,
        init_guess,
        s,
        xcov_flat,
        y,
        &state.snp_vec,
        n,
        p_cov,
        &mut state.reml_workspace,
    );
    if carry_warm_start {
        state.last_log10_lbd = Some(best_log10_lbd);
    }

    let (beta, se, _lbd, _reused_fit) = final_beta_se_with_workspace(
        best_log10_lbd,
        s,
        xcov_flat,
        y,
        &state.snp_vec,
        n,
        p_cov,
        last_eval_log10_lbd,
        &mut state.reml_workspace,
    );
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
    debug_assert!(g_block.len() >= rows.saturating_mul(n));
    debug_assert!(out.len() >= rows.saturating_mul(out_cols));
    if rows == 0 || n == 0 || out_cols == 0 {
        return;
    }
    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let init_state = || RotatedRemlAssocState {
        snp_vec: vec![0.0_f64; n],
        reml_workspace: RemlWorkspace::new(p_cov + 1),
        last_log10_lbd: None,
    };
    let process_row = |row: &[f32], state: &mut RotatedRemlAssocState, out_row: &mut [f64]| {
        process_rotated_reml_assoc_row(
            row,
            s,
            xcov_flat,
            y,
            n,
            p_cov,
            low,
            high,
            tol,
            max_iter,
            init_log10_lbd,
            seed_with_init_guess,
            carry_warm_start,
            nullml_val,
            with_plrt,
            state,
            out_row,
        );
    };

    if carry_warm_start {
        let fixed_rows = LMM_DETERMINISTIC_WARM_START_BLOCK_ROWS;
        let mut run = || {
            out[..rows * out_cols]
                .par_chunks_mut(fixed_rows * out_cols)
                .enumerate()
                .for_each_init(init_state, |state, (block_idx, out_block)| {
                    let block_start = block_idx * fixed_rows;
                    let block_rows = out_block.len() / out_cols;
                    let g_start = block_start * n;
                    let g_end = g_start + block_rows * n;
                    let g_block = &g_block[g_start..g_end];
                    state.last_log10_lbd = None;
                    for (row_idx, out_row) in out_block.chunks_mut(out_cols).enumerate() {
                        let row_start = row_idx * n;
                        process_row(&g_block[row_start..row_start + n], state, out_row);
                    }
                });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
    } else {
        run_rotated_assoc_block_f32(
            g_block,
            rows,
            n,
            out,
            out_cols,
            pool,
            init_state,
            |_, row, state, out_row| process_row(row, state, out_row),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_rotated_reml_assoc_block_f32_with_warm_state(
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
    nullml: Option<f64>,
    state: &mut RotatedRemlWarmStartState,
    out: &mut [f64],
    out_cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert!(g_block.len() >= rows.saturating_mul(n));
    debug_assert!(out.len() >= rows.saturating_mul(out_cols));
    if rows == 0 || n == 0 || out_cols == 0 {
        return;
    }

    let with_plrt = nullml.is_some();
    let nullml_val = nullml.unwrap_or(0.0);
    let mut row_start = 0usize;

    if state.rows_in_group > 0 {
        let prefix_rows = (LMM_DETERMINISTIC_WARM_START_BLOCK_ROWS - state.rows_in_group).min(rows);
        for row_idx in 0..prefix_rows {
            let row = &g_block[row_idx * n..(row_idx + 1) * n];
            let out_row = &mut out[row_idx * out_cols..(row_idx + 1) * out_cols];
            process_rotated_reml_assoc_row(
                row,
                s,
                xcov_flat,
                y,
                n,
                p_cov,
                low,
                high,
                tol,
                max_iter,
                init_log10_lbd,
                seed_with_init_guess,
                true,
                nullml_val,
                with_plrt,
                &mut state.assoc,
                out_row,
            );
        }
        state.rows_in_group += prefix_rows;
        row_start += prefix_rows;
        if state.rows_in_group == LMM_DETERMINISTIC_WARM_START_BLOCK_ROWS {
            state.rows_in_group = 0;
            state.assoc.last_log10_lbd = None;
        }
    }

    let remaining_rows = rows - row_start;
    let full_rows = (remaining_rows / LMM_DETERMINISTIC_WARM_START_BLOCK_ROWS)
        * LMM_DETERMINISTIC_WARM_START_BLOCK_ROWS;
    if full_rows > 0 {
        let full_start = row_start;
        let full_end = full_start + full_rows;
        run_rotated_reml_assoc_block_f32(
            &g_block[full_start * n..full_end * n],
            full_rows,
            n,
            s,
            xcov_flat,
            y,
            p_cov,
            low,
            high,
            tol,
            max_iter,
            init_log10_lbd,
            seed_with_init_guess,
            true,
            nullml,
            &mut out[full_start * out_cols..full_end * out_cols],
            out_cols,
            pool,
        );
        row_start = full_end;
    }

    if row_start < rows {
        if state.rows_in_group == 0 {
            state.assoc.last_log10_lbd = None;
        }
        for row_idx in row_start..rows {
            let row = &g_block[row_idx * n..(row_idx + 1) * n];
            let out_row = &mut out[row_idx * out_cols..(row_idx + 1) * out_cols];
            process_rotated_reml_assoc_row(
                row,
                s,
                xcov_flat,
                y,
                n,
                p_cov,
                low,
                high,
                tol,
                max_iter,
                init_log10_lbd,
                seed_with_init_guess,
                true,
                nullml_val,
                with_plrt,
                &mut state.assoc,
                out_row,
            );
            state.rows_in_group += 1;
            if state.rows_in_group == LMM_DETERMINISTIC_WARM_START_BLOCK_ROWS {
                state.rows_in_group = 0;
                state.assoc.last_log10_lbd = None;
            }
        }
    }
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
            reml_workspace: RemlWorkspace::new(p_cov + 1),
            ml_workspace: MlWorkspace::new(p_cov + 1),
        },
        |_, row, state, out_row| {
            let snp_ssq = copy_rotated_snp_row_to_f64(row, &mut state.snp_vec);
            if !snp_ssq.is_finite() || snp_ssq <= 1e-12 {
                fill_invalid_rotated_lmm2_row(out_row);
                return;
            }

            let reml_init_guess = init_log10_lbd_reml.or(init_log10_lbd_ml);
            let (best_log10_lbd_reml, _best_reml_cost, last_eval_log10_lbd) =
                reml_brent_minimize_with_workspace(
                    low,
                    high,
                    tol,
                    max_iter,
                    reml_init_guess,
                    s,
                    xcov_flat,
                    y,
                    &state.snp_vec,
                    n,
                    p_cov,
                    &mut state.reml_workspace,
                );
            let (beta, se, lambda_reml, _reused_fit) = final_beta_se_with_workspace(
                best_log10_lbd_reml,
                s,
                xcov_flat,
                y,
                &state.snp_vec,
                n,
                p_cov,
                last_eval_log10_lbd,
                &mut state.reml_workspace,
            );
            let pwald = if beta.is_finite() && se.is_finite() && se > 0.0 {
                let z = beta / se;
                (2.0 * normal_sf(z.abs())).clamp(f64::MIN_POSITIVE, 1.0)
            } else {
                fill_invalid_rotated_lmm2_row(out_row);
                return;
            };

            // ML optimization starts from this SNP's REML optimum. Do not
            // carry an optimum across SNPs: Rayon scheduling must not affect
            // the per-marker result.
            let ml_init_guess = Some(best_log10_lbd_reml);
            let (best_log10_lbd_ml, best_ml_cost) = brent_minimize_with_init(
                |x| {
                    -ml_loglike_with_workspace(
                        x,
                        s,
                        xcov_flat,
                        y,
                        Some(&state.snp_vec[..]),
                        n,
                        p_cov,
                        &mut state.ml_workspace,
                    )
                },
                low,
                high,
                tol,
                max_iter,
                ml_init_guess,
            );
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
    let xcov_flat: Cow<[f64]> = match crate::stats_common::array_standard_slice(&xcov) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&xcov_arr)),
    };
    let g_slice_opt = crate::stats_common::array_standard_slice(&g_rot_chunk).ok();

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
                        || (vec![0.0_f64; n], RemlWorkspace::new(p_cov + 1)),
                        |(snp_vec, reml_workspace), (idx, out_row)| {
                            let row = g_arr.row(idx);
                            for i in 0..n {
                                snp_vec[i] = row[i] as f64;
                            }

                            let (best_log10_lbd, _best_cost, last_eval_log10_lbd) =
                                reml_brent_minimize_with_workspace(
                                    low,
                                    high,
                                    tol,
                                    max_iter,
                                    None,
                                    s,
                                    &xcov_flat,
                                    y,
                                    snp_vec,
                                    n,
                                    p_cov,
                                    reml_workspace,
                                );

                            let (beta, se, _lbd, _reused_fit) = final_beta_se_with_workspace(
                                best_log10_lbd,
                                s,
                                &xcov_flat,
                                y,
                                snp_vec,
                                n,
                                p_cov,
                                last_eval_log10_lbd,
                                reml_workspace,
                            );

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
    mut run_rows: RunRows,
    format_rows: FormatRows,
) -> Result<UnifiedBedScanSummary, String>
where
    RunRows: FnMut(&[f32], usize, &mut [f64]),
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
    let resident_bed = crate::gload::lookup_resident_packed_bed(bed_prefix)
        .filter(|res| res.n_samples_full == n_samples_full && res.bytes_per_snp == bytes_per_snp);
    let pgen_source = resident_bed.is_none() && crate::pgen::path_uses_pgen(bed_prefix);
    let mut bed_window = None;
    let mut full_mmap = None;
    let n_snps = if let Some(res) = resident_bed.as_ref() {
        res.n_snps
    } else if pgen_source {
        crate::pgen::count_pgen_variants(bed_prefix, n_samples_full)?
    } else {
        let bed_path = format!("{bed_prefix}.bed");
        bed_window = if let Some(window_mb) = mmap_window_mb.filter(|&v| v > 0) {
            Some(WindowedBedMatrix::open_lm_lmm(bed_prefix, window_mb)?)
        } else {
            None
        };
        full_mmap = if bed_window.is_none() {
            let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
            let mmap =
                unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
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
        if let Some(window) = bed_window.as_ref() {
            window.n_source_snps()
        } else {
            let mmap = full_mmap
                .as_ref()
                .ok_or_else(|| "internal error: missing BED source".to_string())?;
            (mmap.len() - 3) / bytes_per_snp
        }
    };
    let mut pgen_reader = if pgen_source {
        let pgen_prefix = crate::pgen::normalize_pgen_prefix(bed_prefix);
        let pgen_path = std::path::PathBuf::from(format!("{pgen_prefix}.pgen"));
        Some(crate::pgen::PgenReader::open(
            &pgen_path,
            n_samples_full,
            n_snps,
        )?)
    } else {
        None
    };
    let mut pgen_packed: Vec<u8> = Vec::new();
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
        // Validate the caller-owned index table once, but do not duplicate it
        // as a marker-sized usize table. Conversion is bounded to each chunk.
        let source_rows = meta.row_indices.as_slice();
        let mut prev_src: Option<usize> = None;
        for (i, &index) in source_rows.iter().enumerate() {
            if index < 0 {
                return Err(format!("row_indices[{i}] must be >= 0, got {index}"));
            }
            if index as usize >= n_snps {
                return Err(format!(
                    "row_indices[{i}] out of range: {index} >= {n_snps}"
                ));
            }
            let src_row = index as usize;
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
    let mut prepared_chunk_rows: Vec<usize> =
        Vec::with_capacity(if prepared_source_rows.is_some() {
            scan_chunk_snps
        } else {
            0
        });
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
                        {
                            prepared_chunk_rows.clear();
                            prepared_chunk_rows.extend(
                                source_rows_all[chunk_start..chunk_end]
                                    .iter()
                                    .map(|&i| i as usize),
                            );
                            prepared_chunk_rows.as_slice()
                        },
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
            if let Some(res) = resident_bed.as_ref() {
                let src_start = source_rows[0];
                let src_end = source_rows[source_rows.len() - 1] + 1;
                &res.packed[src_start * bytes_per_snp..src_end * bytes_per_snp]
            } else if pgen_source {
                let src_start = source_rows[0];
                let src_end = source_rows[source_rows.len() - 1] + 1;
                let rows = src_end - src_start;
                pgen_packed.resize(rows * bytes_per_snp, 0);
                let reader = match pgen_reader.as_mut() {
                    Some(reader) => reader,
                    None => {
                        let _ =
                            producer_err_bg.set("internal error: missing PGEN reader".to_string());
                        return false;
                    }
                };
                if let Err(e) =
                    reader.read_source_block(src_start, rows, pgen_packed.as_mut_slice())
                {
                    let _ = producer_err_bg.set(e);
                    return false;
                }
                pgen_packed.as_slice()
            } else {
                match bed_window.as_mut() {
                    Some(window) => {
                        match window.prepare_source_rows(source_rows, &mut rel_row_indices) {
                            Ok(slice) => slice,
                            Err(e) => {
                                let _ = producer_err_bg.set(e);
                                return false;
                            }
                        }
                    }
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
            }
        } else if let Some(res) = resident_bed.as_ref() {
            &res.packed[chunk_start * bytes_per_snp..chunk_end * bytes_per_snp]
        } else if pgen_source {
            let rows = chunk_end - chunk_start;
            pgen_packed.resize(rows * bytes_per_snp, 0);
            let reader = match pgen_reader.as_mut() {
                Some(reader) => reader,
                None => {
                    let _ = producer_err_bg.set("internal error: missing PGEN reader".to_string());
                    return false;
                }
            };
            if let Err(e) = reader.read_source_block(chunk_start, rows, pgen_packed.as_mut_slice())
            {
                let _ = producer_err_bg.set(e);
                return false;
            }
            pgen_packed.as_slice()
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
                    let qc = evaluate_packed_bed_row_qc(
                        row,
                        n_samples_full,
                        if use_selected {
                            Some(sample_idx.as_slice())
                        } else {
                            None
                        },
                        if use_selected {
                            selected_excluded_sample_indices
                        } else {
                            None
                        },
                        n,
                        maf_thr,
                        miss_thr,
                        het_thr,
                    );
                    if !qc.keep {
                        return None;
                    }
                    Some(SnpCounts {
                        flip: qc.row_flip,
                        maf: qc.alt_freq,
                        miss_rate: qc.missing_rate,
                        missing_count: qc.missing_count,
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
            writer.send_text(&mut text_buf)?;
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
        || StreamingChunk::new_decoded(scan_chunk_snps, n, out_cols),
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

fn lmm_reml_chunk_from_snp_common<'py, F>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    snp_chunk: PyReadonlyArray2<'py, f32>,
    u_t: PyReadonlyArray2<'py, f32>,
    out_cols: usize,
    out_err_msg: &'static str,
    threads: usize,
    rotate_block_rows: usize,
    run_rows: F,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    F: Fn(&[f32], usize, usize, &[f64], &[f64], &[f64], usize, &mut [f64]) + Sync,
{
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
    let xcov_flat: Cow<[f64]> = match crate::stats_common::array_standard_slice(&xcov) {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&xcov_arr)),
    };
    let snp_flat: Cow<[f32]> = match crate::stats_common::array_standard_slice(&snp_chunk) {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&snp_arr)),
    };
    let ut_flat: Cow<[f32]> = match crate::stats_common::array_standard_slice(&u_t) {
        Ok(s0) => Cow::Borrowed(s0),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&ut_arr)),
    };

    let out = PyArray2::<f64>::zeros(py, [m_chunk, out_cols], false).into_bound();
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err(out_err_msg))?
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
                    run_rows(rot_block, rows, n, s, &xcov_flat, y, p_cov, out_block);
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
                run_rows(rot_block, rows, n, s, &xcov_flat, y, p_cov, out_block);
            }
            start += rows;
        }
    });

    Ok(out)
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
    let out_cols = if nullml.is_some() { 4 } else { 3 };
    lmm_reml_chunk_from_snp_common(
        py,
        s,
        xcov,
        y_rot,
        low,
        high,
        snp_chunk,
        u_t,
        out_cols,
        "beta_se_p not contiguous",
        threads,
        rotate_block_rows,
        |g_block, rows, n, s, xcov_flat, y, p_cov, out_block| {
            run_rotated_reml_assoc_block_f32(
                g_block, rows, n, s, xcov_flat, y, p_cov, low, high, tol, max_iter, None, false,
                false, nullml, out_block, out_cols, None,
            );
        },
    )
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
    if !nullml.is_finite() {
        return Err(PyRuntimeError::new_err("nullml must be finite"));
    }
    let out_cols = 6usize;
    lmm_reml_chunk_from_snp_common(
        py,
        s,
        xcov,
        y_rot,
        low,
        high,
        snp_chunk,
        u_t,
        out_cols,
        "lmm2 output not contiguous",
        threads,
        rotate_block_rows,
        |g_block, rows, n, s, xcov_flat, y, p_cov, out_block| {
            run_rotated_lmm2_assoc_block_f32(
                g_block, rows, n, s, xcov_flat, y, p_cov, low, high, tol, max_iter, None, None,
                nullml, out_block, out_cols, None,
            );
        },
    )
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
    let xcov_slice_storage = crate::stats_common::array_standard_cow(&xcov);
    let xcov_slice: &[f64] = &*xcov_slice_storage;
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

    for r in 0..p {
        for c in 0..r {
            let vrc = a[r * p + c];
            a[c * p + r] = vrc;
        }
    }

    if cholesky_gram_inplace(&mut a, p).is_none() {
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

    let xcov_slice_storage = crate::stats_common::array_standard_cow(&xcov);
    let xcov_slice: &[f64] = &*xcov_slice_storage;
    let snp_flat: Cow<[f32]> = match crate::stats_common::array_standard_slice(&snp_chunk) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&snp_arr)),
    };
    let ut_flat: Cow<[f32]> = match crate::stats_common::array_standard_slice(&u_t) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&ut_arr)),
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

    for r in 0..p {
        for c in 0..r {
            let vrc = a[r * p + c];
            a[c * p + r] = vrc;
        }
    }

    if cholesky_gram_inplace(&mut a, p).is_none() {
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
            let row_flip_vec = match crate::stats_common::array_standard_slice(&row_flip) {
                Ok(slc) => slc.to_vec(),
                Err(_) => crate::stats_common::copy_array_standard_layout(&row_flip.as_array()),
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
            let row_maf_vec = match crate::stats_common::array_standard_slice(&row_maf) {
                Ok(slc) => slc.to_vec(),
                Err(_) => crate::stats_common::copy_array_standard_layout(&row_maf.as_array()),
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

    let xcov_flat: Cow<[f64]> = match crate::stats_common::array_standard_slice(&xcov) {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&xcov_arr)),
    };
    let ut_flat: Cow<[f32]> = match crate::stats_common::array_standard_slice(&u_t) {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&ut_arr)),
    };

    let bed_prefix_owned = bed_prefix.to_string();
    let out_tsv_owned = out_tsv.to_string();
    let stage_timing = env_truthy("JX_LMM_UNIFIED_STAGE_TIMING");
    let (seed_with_null_lambda, carry_snp_lambda) =
        unified_lmm_reml_start_policy(env_truthy("JX_LMM_UNIFIED_NO_WARM_START"));

    py.detach(move || -> Result<usize, String> {
        let assoc_threads = stage_assoc_threads_or(threads, threads);
        let assoc_pool = get_cached_pool(assoc_threads).map_err(|e| format!("{e}"))?;
        let header: &[u8] = if with_plrt {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n"
        } else {
            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n"
        };

        let mut warm_state = RotatedRemlWarmStartState::new(n, p);
        let run_rows = |g_block: &[f32], rows: usize, out_block: &mut [f64]| {
            if carry_snp_lambda {
                run_rotated_reml_assoc_block_f32_with_warm_state(
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
                    seed_with_null_lambda,
                    if with_plrt { Some(nullml_val) } else { None },
                    &mut warm_state,
                    out_block,
                    out_cols,
                    assoc_pool.as_ref(),
                );
            } else {
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
                    seed_with_null_lambda,
                    false,
                    if with_plrt { Some(nullml_val) } else { None },
                    out_block,
                    out_cols,
                    assoc_pool.as_ref(),
                );
            }
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
            run_rows,
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

/// Native streaming LMM scan over a k-file bitset.
///
/// This keeps the same rotated REML kernel, warm-start policy and projection
/// path as the unified BED scanner.  Only the genotype source and the compact
/// four-column k-file writer differ.
#[pyfunction]
#[pyo3(signature = (
    kfile_prefix,
    out_tsv,
    s,
    xcov,
    y_rot,
    u_t,
    sample_indices,
    low=-5.0,
    high=5.0,
    max_iter=30,
    tol=1e-2,
    threads=0,
    init_log10_lbd=None,
    rotate_block_rows=512,
    progress_callback=None,
    progress_every=0,
    genetic_model="add",
))]
pub fn lmm_reml_assoc_kfile_to_tsv_f32<'py>(
    py: Python<'py>,
    kfile_prefix: String,
    out_tsv: String,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    u_t: PyReadonlyArray2<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    init_log10_lbd: Option<f64>,
    rotate_block_rows: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    genetic_model: &str,
) -> PyResult<usize> {
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }
    if !genetic_model.trim().eq_ignore_ascii_case("add") {
        return Err(PyRuntimeError::new_err(
            "k-file native LMM currently supports only genetic_model='add'",
        ));
    }
    let s_slice = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let ut_arr = u_t.as_array();
    let n = y.len();
    if s_slice.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if xcov_arr.ndim() != 2 || xcov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err(
            "Xcov must be a 2D array with Xcov.n_rows == len(y_rot)",
        ));
    }
    let p = xcov_arr.shape()[1];
    if n <= p + 1 {
        return Err(PyRuntimeError::new_err("n must be > p+1"));
    }
    if ut_arr.ndim() != 2 || ut_arr.shape()[0] != n || ut_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("u_t must be (n, n) row-major U^T"));
    }
    let sample_indices_vec: Vec<usize> = sample_indices
        .as_slice()?
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .map_err(|_| PyValueError::new_err("sample_indices must be non-negative"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    if sample_indices_vec.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length {} does not match len(y_rot) {n}",
            sample_indices_vec.len()
        )));
    }
    let xcov_flat: Cow<[f64]> = match crate::stats_common::array_standard_slice(&xcov) {
        Ok(values) => Cow::Borrowed(values),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&xcov_arr)),
    };
    let ut_flat: Cow<[f32]> = match crate::stats_common::array_standard_slice(&u_t) {
        Ok(values) => Cow::Borrowed(values),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&ut_arr)),
    };
    let init_log10_lbd = init_log10_lbd
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(low, high));
    let prefix_owned = kfile_prefix;
    let out_tsv_owned = out_tsv;
    let block_rows = rotate_block_rows.max(1);
    let progress_block = if progress_every == 0 {
        block_rows
    } else {
        progress_every.max(1)
    };

    py.detach(move || -> PyResult<usize> {
        let source = KfileAssocSource::open(Path::new(&prefix_owned), &sample_indices_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let total_rows = source.n_kmers();
        let pool = get_cached_pool(threads).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let writer = AsyncTsvWriter::with_config(
            &out_tsv_owned,
            b"maf\tbeta\tse\tpwald\n",
            64 * 1024 * 1024,
            16,
        )
        .map_err(PyRuntimeError::new_err)?;
        let mut rot_block = vec![0.0_f32; block_rows.saturating_mul(n)];
        let mut out_block = vec![0.0_f64; block_rows.saturating_mul(3)];
        let mut warm_state = RotatedRemlWarmStartState::new(n, p);
        let mut text = String::with_capacity(block_rows.saturating_mul(96));
        let mut done = 0usize;
        let mut next_progress = progress_block.min(total_rows).max(1);
        let overlap = pool
            .as_ref()
            .map(|pool| pool.current_num_threads() > 1)
            .unwrap_or(false);
        let scan_done = source
            .for_each_centered_block(block_rows, overlap, |row_start, rows, snp_sub, maf| {
                if row_start != done {
                    return Err(format!(
                        "k-file LMM source returned non-sequential row_start={row_start}, expected={done}"
                    ));
                }
            let rot_sub = &mut rot_block[..rows * n];
            rotate_snp_block_with_ut_blas(
                snp_sub,
                rows,
                n,
                ut_flat.as_ref(),
                rot_sub,
                threads.max(1),
                pool.as_ref(),
            );
            let out_sub = &mut out_block[..rows * 3];
            run_rotated_reml_assoc_block_f32_with_warm_state(
                rot_sub,
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
                true,
                None,
                &mut warm_state,
                out_sub,
                3,
                pool.as_ref(),
            );
            for row in 0..rows {
                let out_row = &out_sub[row * 3..(row + 1) * 3];
                let _ = writeln!(
                    text,
                    "{:.4}\t{:.4}\t{:.4}\t{:.4e}",
                    maf[row], out_row[0], out_row[1], out_row[2],
                );
            }
            writer
                .send_text(&mut text)
                .map_err(|error| error.to_string())?;
            text.clear();
            done = row_start.saturating_add(rows);
            if let Some(cb) = progress_callback.as_ref() {
                if done >= next_progress || done == total_rows {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (done.min(total_rows), total_rows))?;
                        Ok(())
                    })
                    .map_err(|error| error.to_string())?;
                    while next_progress <= done {
                        next_progress = next_progress.saturating_add(progress_block);
                        if next_progress == 0 {
                            break;
                        }
                    }
                }
            } else {
                Python::attach(|py2| py2.check_signals()).map_err(|error| error.to_string())?;
            }
                Ok(())
            })
            .map_err(PyRuntimeError::new_err)?;
        done = scan_done;
        if let Some(cb) = progress_callback.as_ref() {
            Python::attach(|py2| -> PyResult<()> {
                py2.check_signals()?;
                cb.call1(py2, (total_rows, total_rows))?;
                Ok(())
            })?;
        }
        writer.finish().map_err(PyRuntimeError::new_err)?;
        Ok(done)
    })
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
            let row_flip_vec = match crate::stats_common::array_standard_slice(&row_flip) {
                Ok(slc) => slc.to_vec(),
                Err(_) => crate::stats_common::copy_array_standard_layout(&row_flip.as_array()),
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
            let row_maf_vec = match crate::stats_common::array_standard_slice(&row_maf) {
                Ok(slc) => slc.to_vec(),
                Err(_) => crate::stats_common::copy_array_standard_layout(&row_maf.as_array()),
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

    let xcov_flat: Cow<[f64]> = match crate::stats_common::array_standard_slice(&xcov) {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&xcov_arr)),
    };
    let ut_flat: Cow<[f32]> = match crate::stats_common::array_standard_slice(&u_t) {
        Ok(slc) => Cow::Borrowed(slc),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&ut_arr)),
    };

    let bed_prefix_owned = bed_prefix.to_string();
    let out_tsv_owned = out_tsv.to_string();
    let stage_timing =
        env_truthy("JX_LMM2_UNIFIED_STAGE_TIMING") || env_truthy("JX_LMM_UNIFIED_STAGE_TIMING");
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

    let xcov_flat: Cow<[f64]> = match crate::stats_common::array_standard_slice(&xcov) {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&xcov_arr)),
    };
    let packed_flat: Cow<[u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };
    let ut_flat: Cow<[f32]> = match crate::stats_common::array_standard_slice(&u_t) {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&ut_arr)),
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
        let mut warm_state = RotatedRemlWarmStartState::new(n, p_cov);

        let mut run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            run_rotated_reml_assoc_block_f32_with_warm_state(
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
                if with_plrt { Some(nullml_val) } else { None },
                &mut warm_state,
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

    let xcov_flat: Cow<[f64]> = match crate::stats_common::array_standard_slice(&xcov) {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&xcov_arr)),
    };
    let packed_flat: Cow<[u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };
    let ut_flat: Cow<[f32]> = match crate::stats_common::array_standard_slice(&u_t) {
        Ok(v) => Cow::Borrowed(v),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&ut_arr)),
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
        let mut warm_state = RotatedRemlWarmStartState::new(n, p_cov);
        let mut out_block = vec![0.0_f64; block_rows * out_cols];
        let mut miss_block = vec![0usize; block_rows];
        let mut text_buf = String::with_capacity(block_rows * 128);

        let mut run_rows = |g_block: &[f32], out_block: &mut [f64]| {
            run_rotated_reml_assoc_block_f32_with_warm_state(
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
                if with_plrt { Some(nullml_val) } else { None },
                &mut warm_state,
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
#[cfg(test)]
mod tests {
    use super::{
        run_rotated_lmm2_assoc_block_f32, run_rotated_reml_assoc_block_f32_with_warm_state,
        unified_lmm_reml_start_policy, RotatedRemlWarmStartState,
    };
    use std::sync::Arc;

    #[test]
    fn unified_lmm_uses_fixed_null_seed_with_deterministic_snp_carry() {
        assert_eq!(unified_lmm_reml_start_policy(false), (true, true));
        assert_eq!(unified_lmm_reml_start_policy(true), (false, false));
    }

    #[test]
    fn lmm_snp_warm_start_is_invariant_to_outer_chunk_boundaries() {
        let n = 12usize;
        let rows = 32usize;
        let s = [
            0.15_f64, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.55, 1.8, 2.1,
        ];
        let xcov = [1.0_f64; 12];
        let y = [
            -0.8_f64, 0.2, 1.1, -0.4, 0.7, 1.4, -1.2, 0.5, 0.9, -0.1, 1.8, -0.6,
        ];
        let mut genotypes = vec![0.0_f32; rows * n];
        for marker in 0..rows {
            for sample in 0..n {
                genotypes[marker * n + sample] =
                    ((marker * 7 + sample * (marker % 5 + 1)) % 3) as f32;
            }
        }
        let one_thread_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap(),
        );
        let four_thread_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap(),
        );

        let run_chunks = |chunk_rows: &[usize], pool: &Arc<rayon::ThreadPool>| {
            let mut output = vec![0.0_f64; rows * 3];
            let mut warm_state = RotatedRemlWarmStartState::new(n, 1);
            let mut row_start = 0usize;
            for &chunk_len in chunk_rows {
                let row_end = row_start + chunk_len;
                run_rotated_reml_assoc_block_f32_with_warm_state(
                    &genotypes[row_start * n..row_end * n],
                    chunk_len,
                    n,
                    &s,
                    &xcov,
                    &y,
                    1,
                    -5.0,
                    5.0,
                    0.2,
                    4,
                    Some(-0.5),
                    true,
                    None,
                    &mut warm_state,
                    &mut output[row_start * 3..row_end * 3],
                    3,
                    Some(pool),
                );
                row_start = row_end;
            }
            assert_eq!(row_start, rows, "chunk lengths must cover all markers");
            output
        };

        let one_chunk = run_chunks(&[rows], &one_thread_pool);
        let split_at_non_group_boundary = run_chunks(&[13, 19], &one_thread_pool);
        let many_small_chunks = run_chunks(&[5, 11, 7, 9], &four_thread_pool);
        for (label, compared) in [
            ("13/19 split", &split_at_non_group_boundary),
            ("5/11/7/9 split with four threads", &many_small_chunks),
        ] {
            for (idx, (&whole, &split)) in one_chunk.iter().zip(compared).enumerate() {
                assert!(
                    whole == split || (whole.is_nan() && split.is_nan()),
                    "association result changed at cell {idx} for {label}: {whole:?} != {split:?}"
                );
            }
        }
    }

    #[test]
    fn lmm2_per_snp_results_do_not_depend_on_neighbor_markers() {
        let n = 6usize;
        let s = [0.2_f64, 0.4, 0.8, 1.1, 1.7, 2.3];
        let xcov = [1.0_f64; 6];
        let y = [1.0_f64, 2.0, 1.0, 3.0, 0.0, 2.0];
        let genotypes = [
            0.0_f32, 1.0, 2.0, 1.0, 0.0, 2.0, // marker 1
            2.0_f32, 2.0, 0.0, 0.0, 1.0, 1.0, // marker 2
            0.0_f32, 0.0, 1.0, 2.0, 2.0, 1.0, // marker 3
        ];
        let mut together = [0.0_f64; 18];
        run_rotated_lmm2_assoc_block_f32(
            &genotypes,
            3,
            n,
            &s,
            &xcov,
            &y,
            1,
            -5.0,
            5.0,
            1e-4,
            30,
            Some(-0.4),
            Some(0.1),
            -10.0,
            &mut together,
            6,
            None,
        );

        for marker in 0..3 {
            let mut alone = [0.0_f64; 6];
            run_rotated_lmm2_assoc_block_f32(
                &genotypes[marker * n..(marker + 1) * n],
                1,
                n,
                &s,
                &xcov,
                &y,
                1,
                -5.0,
                5.0,
                1e-4,
                30,
                Some(-0.4),
                Some(0.1),
                -10.0,
                &mut alone,
                6,
                None,
            );
            assert_eq!(&together[marker * 6..(marker + 1) * 6], &alone);
        }
    }
}
