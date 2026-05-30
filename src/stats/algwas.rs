use memmap2::{MmapMut, MmapOptions};
use nalgebra::{DMatrix, DVector};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::active_path::{
    run_active_kkt_path, validate_active_path_state, ActivePathSolveConfig, ActivePathState,
};
use crate::bedmath::{
    adaptive_grm_block_rows, decode_plink_bed_hardcall, decode_row_centered_full_lut_f64,
    decode_standardized_packed_block_rows_f32, is_identity_indices, packed_byte_lut,
};
use crate::blas::{
    cblas_daxpy_dispatch, cblas_dgemm_dispatch, CblasInt, OpenBlasThreadGuard, CBLAS_COL_MAJOR,
    CBLAS_NO_TRANS, CBLAS_TRANS,
};
use crate::he::build_row_standardization_stats;
use crate::linalg::{chi2_sf_df1, chi2_stat_df1_from_sf, format_chisq_value};
use crate::math_farmcpu::decode_packed_rows_to_sample_major;
use crate::pcg::{pcg_solve, IdentityPreconditioner, PcgOperator};
use crate::stats_common::{get_cached_pool, parse_index_vec_i64, AsyncTsvWriter};

const DEFAULT_STANDARDIZE_EPS32: f32 = 1e-12_f32;
const DEFAULT_STAGE1_PATH_STEPS: usize = 64;
const DEFAULT_STAGE1_LAMBDA_MIN_RATIO: f32 = 0.001_f32;
const DEFAULT_STAGE1_BLOCK_ROWS: usize = 1024usize;
const DEFAULT_STAGE1_ACTIVE_TARGET_MB: usize = 512usize;
const DEFAULT_STAGE1_ACTIVE_MIN_ROWS: usize = 256usize;
const DEFAULT_STAGE1_ALASSO_GAMMA: f32 = 1.0_f32;
const DEFAULT_STAGE1_ALASSO_RIDGE_LAMBDA: f32 = 0.001_f32;
const DEFAULT_STAGE1_ALASSO_WEIGHT_FLOOR: f32 = 1e-8_f32;
const DEFAULT_STAGE1_ALASSO_WEIGHT_CAP: f32 = 1e8_f32;
const DEFAULT_STAGE1_ALASSO_WEIGHT_SCREEN: usize = 4096usize;
const DEFAULT_STAGE1_ALASSO_INITIAL_WORKING_SET: usize = 4096usize;
const DEFAULT_STAGE1_ALASSO_STRONG_SET: usize = 16384usize;
const DEFAULT_STAGE1_ALASSO_PCG_MAX_ITERS: usize = 64usize;
const DEFAULT_STAGE1_ALASSO_PCG_TOL: f64 = 1e-5_f64;
const DEFAULT_STAGE1_ALASSO_DENSE_WEIGHT_MAX_CELLS: usize = 32_000_000usize;
const DEFAULT_STAGE1_EBIC_GAMMA: f64 = 0.5_f64;
const DEFAULT_STAGE1_MSGPS_STEP: usize = 20_000usize;
const DEFAULT_STAGE1_MSGPS_STEP_MAX: usize = 200_000usize;
const DEFAULT_STAGE1_MSGPS_PMAX: usize = 300usize;
const DEFAULT_STAGE1_MSGPS_XTX_CACHE_MB: usize = 512usize;
const DEFAULT_STAGE1_MSGPS_XTX_LOG_EVERY: usize = 32usize;
const DEFAULT_STAGE1_AUTO_EXACT_MAX_FEATURES: usize = 32_768usize;
const DEFAULT_STAGE1_AUTO_EXACT_MAX_CELLS: usize = 64_000_000usize;
const DEFAULT_STAGE1_EXACT_PCG_MIN_N: usize = 4096usize;
const ALG_LASSO_PAR_THRESHOLD: usize = 16_384usize;
const ALG_ROW_CHUNK: usize = 64usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AlgwasStage1Mode {
    Auto,
    PackedExactMsgps,
    StreamActive,
    DenseMsgps,
}

#[inline]
fn algwas_stage1_mode() -> AlgwasStage1Mode {
    let raw = std::env::var("JX_ALGWAS_STAGE1_MODE").ok();
    algwas_stage1_mode_from_raw(raw.as_deref())
}

#[inline]
fn algwas_stage1_mode_from_raw(raw: Option<&str>) -> AlgwasStage1Mode {
    let raw = raw.unwrap_or("").trim().to_ascii_lowercase();
    match raw.as_str() {
        "" => AlgwasStage1Mode::StreamActive,
        "auto" => AlgwasStage1Mode::Auto,
        "dense" | "msgps" => AlgwasStage1Mode::DenseMsgps,
        "stream" | "approx" | "active" => AlgwasStage1Mode::StreamActive,
        "packed" | "exact" | "packed_exact" | "exact_packed" => AlgwasStage1Mode::PackedExactMsgps,
        _ => AlgwasStage1Mode::PackedExactMsgps,
    }
}

#[inline]
fn algwas_stage1_auto_limits() -> (usize, usize) {
    let max_features = std::env::var("JX_ALGWAS_STAGE1_AUTO_EXACT_MAX_FEATURES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_STAGE1_AUTO_EXACT_MAX_FEATURES);
    let max_cells = std::env::var("JX_ALGWAS_STAGE1_AUTO_EXACT_MAX_CELLS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_STAGE1_AUTO_EXACT_MAX_CELLS);
    (max_features, max_cells)
}

#[inline]
fn algwas_stage1_mode_resolved(mode: AlgwasStage1Mode, n: usize, p: usize) -> AlgwasStage1Mode {
    let (max_features, max_cells) = algwas_stage1_auto_limits();
    algwas_stage1_mode_resolved_with_limits(mode, n, p, max_features, max_cells)
}

#[inline]
fn algwas_stage1_mode_resolved_with_limits(
    mode: AlgwasStage1Mode,
    n: usize,
    p: usize,
    max_features: usize,
    max_cells: usize,
) -> AlgwasStage1Mode {
    match mode {
        AlgwasStage1Mode::Auto => {
            if p <= max_features && n.saturating_mul(p) <= max_cells {
                AlgwasStage1Mode::PackedExactMsgps
            } else {
                AlgwasStage1Mode::StreamActive
            }
        }
        other => other,
    }
}

#[inline]
fn algwas_stage1_mode_log_enabled() -> bool {
    env_var_truthy("JX_ALGWAS_STAGE1_MODE_LOG") || std::env::var_os("JX_ALGWAS_DEBUG").is_some()
}

#[inline]
fn algwas_stage1_dense_weights_enabled() -> bool {
    match std::env::var("JX_ALGWAS_STAGE1_DENSE_WEIGHTS") {
        Ok(v) => {
            let raw = v.trim().to_ascii_lowercase();
            !matches!(raw.as_str(), "0" | "false" | "no" | "off")
        }
        Err(_) => false,
    }
}

#[inline]
fn algwas_stage1_timing_enabled() -> bool {
    env_var_truthy("JX_ALGWAS_STAGE1_TIMING")
}

#[inline]
fn algwas_stage1_ebic_gamma() -> f64 {
    std::env::var("JX_ALGWAS_STAGE1_EBIC_GAMMA")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0_f64)
        .unwrap_or(DEFAULT_STAGE1_EBIC_GAMMA)
}

#[inline]
fn algwas_stage1_site_cap(n: usize) -> usize {
    let default_cap = ((n as f64).sqrt().floor() as usize).clamp(1usize, 50usize);
    std::env::var("JX_ALGWAS_STAGE1_SITE_CAP")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_cap)
}

#[inline]
fn algwas_stage1_ld_prune_r2() -> f64 {
    std::env::var("JX_ALGWAS_STAGE1_LD_R2")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0_f64)
        .unwrap_or(0.8_f64)
        .clamp(1e-6_f64, 1.0_f64)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AlgwasStage1SelectCriterion {
    Ebic,
    Bic,
}

#[inline]
fn algwas_stage1_select_criterion() -> AlgwasStage1SelectCriterion {
    match std::env::var("JX_ALGWAS_STAGE1_SELECT") {
        Ok(v) => {
            let raw = v.trim().to_ascii_lowercase();
            match raw.as_str() {
                "" | "bic" => AlgwasStage1SelectCriterion::Bic,
                "ebic" => AlgwasStage1SelectCriterion::Ebic,
                _ => AlgwasStage1SelectCriterion::Bic,
            }
        }
        Err(_) => AlgwasStage1SelectCriterion::Bic,
    }
}

#[inline]
fn env_var_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let raw = v.trim().to_ascii_lowercase();
            !matches!(raw.as_str(), "" | "0" | "false" | "no" | "off")
        }
        Err(_) => false,
    }
}

#[inline]
fn algwas_xtx_cache_log_enabled() -> bool {
    env_var_truthy("JX_ALGWAS_XTX_CACHE_LOG") || std::env::var_os("JX_ALGWAS_DEBUG").is_some()
}

#[inline]
fn algwas_xtx_cache_log_every() -> usize {
    std::env::var("JX_ALGWAS_XTX_CACHE_LOG_EVERY")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_STAGE1_MSGPS_XTX_LOG_EVERY)
}

#[inline]
fn algwas_exact_pcg_enabled(n: usize) -> bool {
    if env_var_truthy("JX_ALGWAS_STAGE1_EXACT_DIRECT") {
        return false;
    }
    if env_var_truthy("JX_ALGWAS_STAGE1_EXACT_PCG") {
        return true;
    }
    let min_n = std::env::var("JX_ALGWAS_STAGE1_EXACT_PCG_MIN_N")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_STAGE1_EXACT_PCG_MIN_N);
    n >= min_n
}

#[inline]
fn format_bytes_binary(n: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = n as f64;
    let mut unit_idx = 0usize;
    while value >= 1024.0_f64 && unit_idx + 1 < UNITS.len() {
        value /= 1024.0_f64;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{n} {}", UNITS[unit_idx])
    } else {
        format!("{value:.1} {}", UNITS[unit_idx])
    }
}

#[inline]
fn dot_f32_f64(a: &[f32], b: &[f32], pool: Option<&Arc<rayon::ThreadPool>>) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && a.len() >= ALG_LASSO_PAR_THRESHOLD
    {
        let run = || {
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| (*x as f64) * (*y as f64))
                .sum()
        };
        if let Some(tp) = pool {
            return tp.install(run);
        }
        return run();
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

#[inline]
fn log_choose_ln(n: usize, k: usize) -> f64 {
    if k == 0 || k >= n {
        return 0.0_f64;
    }
    let k_eff = k.min(n - k);
    libm::lgamma((n + 1) as f64)
        - libm::lgamma((k_eff + 1) as f64)
        - libm::lgamma((n - k_eff + 1) as f64)
}

#[inline]
fn algwas_bic_from_rss_df(n: usize, rss: f64, df: f64) -> f64 {
    (n as f64) * (rss.max(1e-12_f64) / n as f64).ln() + df * (n as f64).ln()
}

#[inline]
fn algwas_ebic_from_bic(bic: f64, n_features: usize, nnz: usize, gamma: f64) -> f64 {
    bic + 2.0_f64 * gamma.max(0.0_f64) * log_choose_ln(n_features, nnz)
}

#[inline]
fn algwas_stage1_path_score(
    point: &AlgwasStage1PathPoint,
    criterion: AlgwasStage1SelectCriterion,
) -> f64 {
    match criterion {
        AlgwasStage1SelectCriterion::Ebic => point.ebic,
        AlgwasStage1SelectCriterion::Bic => point.bic,
    }
}

#[inline]
fn best_stage1_path_index(
    path: &[AlgwasStage1PathPoint],
    criterion: AlgwasStage1SelectCriterion,
) -> usize {
    path.iter()
        .enumerate()
        .min_by(|a, b| {
            algwas_stage1_path_score(a.1, criterion)
                .total_cmp(&algwas_stage1_path_score(b.1, criterion))
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0usize)
}

#[inline]
fn validate_finite_slice_f32(name: &str, x: &[f32]) -> Result<(), String> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(format!("{name} contains non-finite values"));
    }
    Ok(())
}

#[inline]
fn standardized_value_lut_algwas(row_flip: bool, mean_g: f32, inv_sd: f32) -> [f32; 4] {
    if row_flip {
        [
            (2.0_f32 - mean_g) * inv_sd,
            0.0_f32,
            (1.0_f32 - mean_g) * inv_sd,
            (0.0_f32 - mean_g) * inv_sd,
        ]
    } else {
        [
            (0.0_f32 - mean_g) * inv_sd,
            0.0_f32,
            (1.0_f32 - mean_g) * inv_sd,
            (2.0_f32 - mean_g) * inv_sd,
        ]
    }
}

#[inline]
fn packed_row_standardized_dot_full_algwas(
    row: &[u8],
    n_samples: usize,
    sample_vec: &[f32],
    value_lut: &[f32; 4],
    code4_lut: &[[u8; 4]; 256],
) -> f64 {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut sum = 0.0_f64;
    let mut col = 0usize;
    for &b in row.iter().take(full_bytes) {
        let codes = &code4_lut[b as usize];
        sum += (value_lut[codes[0] as usize] as f64) * (sample_vec[col] as f64);
        sum += (value_lut[codes[1] as usize] as f64) * (sample_vec[col + 1] as f64);
        sum += (value_lut[codes[2] as usize] as f64) * (sample_vec[col + 2] as f64);
        sum += (value_lut[codes[3] as usize] as f64) * (sample_vec[col + 3] as f64);
        col += 4;
    }
    if rem > 0 {
        let codes = &code4_lut[row[full_bytes] as usize];
        for lane in 0..rem {
            sum += (value_lut[codes[lane] as usize] as f64) * (sample_vec[col + lane] as f64);
        }
    }
    sum
}

#[inline]
fn packed_row_standardized_dot_selected_algwas(
    row: &[u8],
    sample_idx: &[usize],
    sample_vec: &[f32],
    value_lut: &[f32; 4],
) -> f64 {
    let mut sum = 0.0_f64;
    for (local_i, &sid) in sample_idx.iter().enumerate() {
        let code = (row[sid >> 2] >> ((sid & 3) * 2)) & 0b11;
        sum += (value_lut[code as usize] as f64) * (sample_vec[local_i] as f64);
    }
    sum
}

#[inline]
fn packed_row_standardized_sum_full_algwas(
    row: &[u8],
    n_samples: usize,
    value_lut: &[f32; 4],
    code4_lut: &[[u8; 4]; 256],
) -> f64 {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut sum = 0.0_f64;
    for &b in row.iter().take(full_bytes) {
        let codes = &code4_lut[b as usize];
        sum += value_lut[codes[0] as usize] as f64;
        sum += value_lut[codes[1] as usize] as f64;
        sum += value_lut[codes[2] as usize] as f64;
        sum += value_lut[codes[3] as usize] as f64;
    }
    if rem > 0 {
        let codes = &code4_lut[row[full_bytes] as usize];
        for lane in 0..rem {
            sum += value_lut[codes[lane] as usize] as f64;
        }
    }
    sum
}

#[inline]
fn packed_row_standardized_sum_selected_algwas(
    row: &[u8],
    sample_idx: &[usize],
    value_lut: &[f32; 4],
) -> f64 {
    let mut sum = 0.0_f64;
    for &sid in sample_idx {
        let code = (row[sid >> 2] >> ((sid & 3) * 2)) & 0b11;
        sum += value_lut[code as usize] as f64;
    }
    sum
}

#[inline]
fn packed_row_standardized_sumsq_full_algwas(
    row: &[u8],
    n_samples: usize,
    value_lut: &[f32; 4],
    code4_lut: &[[u8; 4]; 256],
) -> f64 {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let value_sq = [
        (value_lut[0] as f64) * (value_lut[0] as f64),
        0.0_f64,
        (value_lut[2] as f64) * (value_lut[2] as f64),
        (value_lut[3] as f64) * (value_lut[3] as f64),
    ];
    let mut sum = 0.0_f64;
    for &b in row.iter().take(full_bytes) {
        let codes = &code4_lut[b as usize];
        sum += value_sq[codes[0] as usize];
        sum += value_sq[codes[1] as usize];
        sum += value_sq[codes[2] as usize];
        sum += value_sq[codes[3] as usize];
    }
    if rem > 0 {
        let codes = &code4_lut[row[full_bytes] as usize];
        for lane in 0..rem {
            sum += value_sq[codes[lane] as usize];
        }
    }
    sum
}

#[inline]
fn packed_row_standardized_sumsq_selected_algwas(
    row: &[u8],
    sample_idx: &[usize],
    value_lut: &[f32; 4],
) -> f64 {
    let value_sq = [
        (value_lut[0] as f64) * (value_lut[0] as f64),
        0.0_f64,
        (value_lut[2] as f64) * (value_lut[2] as f64),
        (value_lut[3] as f64) * (value_lut[3] as f64),
    ];
    let mut sum = 0.0_f64;
    for &sid in sample_idx {
        let code = (row[sid >> 2] >> ((sid & 3) * 2)) & 0b11;
        sum += value_sq[code as usize];
    }
    sum
}

#[inline]
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn daxpy_inplace_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    if alpha == 0.0_f64 || x.is_empty() {
        return;
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        if x.len() <= (CblasInt::MAX as usize) {
            unsafe {
                cblas_daxpy_dispatch(
                    x.len() as CblasInt,
                    alpha,
                    x.as_ptr(),
                    1 as CblasInt,
                    y.as_mut_ptr(),
                    1 as CblasInt,
                );
            }
            return;
        }
    }
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = alpha.mul_add(*xi, *yi);
    }
}

#[inline]
fn current_pool_threads(pool: Option<&Arc<rayon::ThreadPool>>) -> usize {
    pool.map(|tp| tp.current_num_threads()).unwrap_or(1).max(1)
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

#[allow(dead_code)]
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
    if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(cols) >= ALG_LASSO_PAR_THRESHOLD
    {
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
    if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(cols) >= ALG_LASSO_PAR_THRESHOLD
    {
        let chunk_rows = ALG_ROW_CHUNK.min(rows.max(1));
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
fn row_major_block_t_mul_vec_accum_f64(
    block: &[f64],
    rows: usize,
    cols: usize,
    vec: &[f64],
    out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);
    if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(cols) >= ALG_LASSO_PAR_THRESHOLD
    {
        let chunk_rows = ALG_ROW_CHUNK.min(rows.max(1));
        let mut run = || {
            let partial = block
                .par_chunks(chunk_rows * cols)
                .zip(vec.par_chunks(chunk_rows))
                .fold(
                    || vec![0.0_f64; cols],
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
                    || vec![0.0_f64; cols],
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
fn row_major_block_mul_vec_f64(
    block: &[f64],
    rows: usize,
    cols: usize,
    vec: &[f64],
    out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);
    if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(cols) >= ALG_LASSO_PAR_THRESHOLD
    {
        let mut run = || {
            out.par_iter_mut().enumerate().for_each(|(r, dst)| {
                let row = &block[r * cols..(r + 1) * cols];
                let mut acc = 0.0_f64;
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
        let mut acc = 0.0_f64;
        for c in 0..cols {
            acc += row[c] * vec[c];
        }
        out[r] = acc;
    }
}

#[inline]
fn residualize_block_rows_in_place_f64(
    block: &mut [f64],
    rows: usize,
    cols: usize,
    proj: &CovariateProjection,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    if rows == 0 || cols == 0 {
        return;
    }
    if proj.is_intercept_only() {
        let mut run = || {
            block.par_chunks_mut(cols).for_each(|row| {
                let mean = row.iter().sum::<f64>() * proj.inv_n;
                for v in row.iter_mut() {
                    *v -= mean;
                }
            });
        };
        if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false) {
            if let Some(tp) = pool {
                tp.install(run);
            } else {
                run();
            }
        } else {
            for row in block.chunks_mut(cols) {
                let mean = row.iter().sum::<f64>() * proj.inv_n;
                for v in row.iter_mut() {
                    *v -= mean;
                }
            }
        }
        return;
    }
    if !pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false) {
        let mut ztrow = vec![0.0_f64; proj.q];
        let mut coeff = vec![0.0_f64; proj.q];
        for row in block.chunks_mut(cols) {
            proj.ztv_f64(row, &mut ztrow);
            proj.apply_inv(&ztrow, &mut coeff);
            for i in 0..cols {
                let mut projv = 0.0_f64;
                let zrow = &proj.z[i * proj.q..(i + 1) * proj.q];
                for a in 0..proj.q {
                    projv += zrow[a] * coeff[a];
                }
                row[i] -= projv;
            }
        }
        return;
    }
    let mut run = || {
        block.par_chunks_mut(cols).for_each_init(
            || (vec![0.0_f64; proj.q], vec![0.0_f64; proj.q]),
            |(ztrow, coeff), row| {
                proj.ztv_f64(row, ztrow);
                proj.apply_inv(ztrow, coeff);
                for i in 0..cols {
                    let mut projv = 0.0_f64;
                    let zrow = &proj.z[i * proj.q..(i + 1) * proj.q];
                    for a in 0..proj.q {
                        projv += zrow[a] * coeff[a];
                    }
                    row[i] -= projv;
                }
            },
        );
    };
    if let Some(tp) = pool {
        tp.install(run);
    } else {
        run();
    }
}

#[inline]
fn center_and_scale_block_rows_in_place_f64(
    block: &mut [f64],
    rows: usize,
    cols: usize,
    scales_out: Option<&mut [f64]>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    if rows == 0 || cols == 0 {
        return;
    }

    let do_row = |row: &mut [f64], scale_dst: Option<&mut f64>| {
        let mut mean = 0.0_f64;
        for &v in row.iter() {
            mean += v;
        }
        mean /= cols as f64;
        let mut ss = 0.0_f64;
        for v in row.iter_mut() {
            let d = *v - mean;
            *v = d;
            ss += d * d;
        }
        let scale = if ss > 0.0_f64 && ss.is_finite() {
            1.0_f64 / ss.sqrt()
        } else {
            0.0_f64
        };
        if scale != 0.0_f64 {
            for v in row.iter_mut() {
                *v *= scale;
            }
        }
        if let Some(dst) = scale_dst {
            *dst = scale;
        }
    };

    match scales_out {
        Some(scales) => {
            debug_assert_eq!(scales.len(), rows);
            if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false) {
                let mut run = || {
                    block
                        .par_chunks_mut(cols)
                        .zip(scales.par_iter_mut())
                        .for_each(|(row, scale_dst)| do_row(row, Some(scale_dst)));
                };
                if let Some(tp) = pool {
                    tp.install(run);
                } else {
                    run();
                }
            } else {
                for (row, scale_dst) in block.chunks_mut(cols).zip(scales.iter_mut()) {
                    do_row(row, Some(scale_dst));
                }
            }
        }
        None => {
            if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false) {
                let mut run = || {
                    block
                        .par_chunks_mut(cols)
                        .for_each(|row| do_row(row, None::<&mut f64>));
                };
                if let Some(tp) = pool {
                    tp.install(run);
                } else {
                    run();
                }
            } else {
                for row in block.chunks_mut(cols) {
                    do_row(row, None::<&mut f64>);
                }
            }
        }
    }
}

#[inline]
fn gram_add_aat_f64(gram: &mut [f64], block: &[f64], rows: usize, n: usize, is_first_block: bool) {
    debug_assert_eq!(gram.len(), n.saturating_mul(n));
    debug_assert_eq!(block.len(), rows.saturating_mul(n));
    if rows == 0 || n == 0 {
        return;
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            n as CblasInt,
            n as CblasInt,
            rows as CblasInt,
            1.0_f64,
            block.as_ptr(),
            n as CblasInt,
            block.as_ptr(),
            n as CblasInt,
            if is_first_block { 0.0_f64 } else { 1.0_f64 },
            gram.as_mut_ptr(),
            n as CblasInt,
        );
        return;
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        if is_first_block {
            gram.fill(0.0_f64);
        }
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f64;
                for r in 0..rows {
                    acc += block[r * n + i] * block[r * n + j];
                }
                gram[i * n + j] += acc;
            }
        }
    }
}

#[inline]
fn residualize_block_rows_in_place(
    block: &mut [f32],
    rows: usize,
    cols: usize,
    proj: &CovariateProjection,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    if rows == 0 || cols == 0 {
        return;
    }
    if proj.is_intercept_only() {
        let mut run = || {
            block.par_chunks_mut(cols).for_each(|row| {
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() * proj.inv_n;
                for v in row.iter_mut() {
                    *v -= mean as f32;
                }
            });
        };
        if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false) {
            if let Some(tp) = pool {
                tp.install(run);
            } else {
                run();
            }
        } else {
            for row in block.chunks_mut(cols) {
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() * proj.inv_n;
                for v in row.iter_mut() {
                    *v -= mean as f32;
                }
            }
        }
        return;
    }
    if !pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false) {
        let mut ztrow = vec![0.0_f64; proj.q];
        let mut coeff = vec![0.0_f64; proj.q];
        for row in block.chunks_mut(cols) {
            proj.ztv_f32(row, &mut ztrow);
            proj.apply_inv(&ztrow, &mut coeff);
            for i in 0..cols {
                let mut projv = 0.0_f64;
                let zrow = &proj.z[i * proj.q..(i + 1) * proj.q];
                for a in 0..proj.q {
                    projv += zrow[a] * coeff[a];
                }
                row[i] -= projv as f32;
            }
        }
        return;
    }
    let mut run = || {
        block.par_chunks_mut(cols).enumerate().for_each_init(
            || (vec![0.0_f64; proj.q], vec![0.0_f64; proj.q]),
            |(ztrow, coeff), (_local, row)| {
                proj.ztv_f32(row, ztrow);
                proj.apply_inv(ztrow, coeff);
                for i in 0..cols {
                    let mut projv = 0.0_f64;
                    let zrow = &proj.z[i * proj.q..(i + 1) * proj.q];
                    for a in 0..proj.q {
                        projv += zrow[a] * coeff[a];
                    }
                    row[i] -= projv as f32;
                }
            },
        );
    };
    if let Some(tp) = pool {
        tp.install(run);
    } else {
        run();
    }
}

fn center_and_scale_columns_in_place(
    x: &mut DMatrix<f64>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> (Vec<f64>, Vec<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let mut means = vec![0.0_f64; p];
    let mut inv_norm = vec![0.0_f64; p];
    let data = x.as_mut_slice();
    if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false) {
        let mut run = || {
            data.par_chunks_mut(n)
                .zip(means.par_iter_mut())
                .zip(inv_norm.par_iter_mut())
                .for_each(|((col, mean_dst), inv_dst)| {
                    let mut mean = 0.0_f64;
                    for &v in col.iter() {
                        mean += v;
                    }
                    mean /= n as f64;
                    *mean_dst = mean;
                    let mut ss = 0.0_f64;
                    for v in col.iter_mut() {
                        let d = *v - mean;
                        *v = d;
                        ss += d * d;
                    }
                    *inv_dst = if ss > 0.0_f64 && ss.is_finite() {
                        1.0_f64 / ss.sqrt()
                    } else {
                        0.0_f64
                    };
                    if *inv_dst != 0.0_f64 {
                        for v in col.iter_mut() {
                            *v *= *inv_dst;
                        }
                    }
                });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
    } else {
        for ((col, mean_dst), inv_dst) in data
            .chunks_mut(n)
            .zip(means.iter_mut())
            .zip(inv_norm.iter_mut())
        {
            let mut mean = 0.0_f64;
            for &v in col.iter() {
                mean += v;
            }
            mean /= n as f64;
            *mean_dst = mean;
            let mut ss = 0.0_f64;
            for v in col.iter_mut() {
                let d = *v - mean;
                *v = d;
                ss += d * d;
            }
            *inv_dst = if ss > 0.0_f64 && ss.is_finite() {
                1.0_f64 / ss.sqrt()
            } else {
                0.0_f64
            };
            if *inv_dst != 0.0_f64 {
                for v in col.iter_mut() {
                    *v *= *inv_dst;
                }
            }
        }
    }
    (means, inv_norm)
}

fn residualize_dense_columns_in_place(
    x: &mut DMatrix<f64>,
    proj: &CovariateProjection,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let n = x.nrows();
    if n != proj.n {
        return Err("dense residualization length mismatch".to_string());
    }
    let data = x.as_mut_slice();
    let mut run = || {
        data.par_chunks_mut(n).enumerate().for_each(|(_j, col)| {
            let mut ztv = vec![0.0_f64; proj.q];
            let mut coeff = vec![0.0_f64; proj.q];
            proj.ztv_f64(col, &mut ztv);
            proj.apply_inv(&ztv, &mut coeff);
            for i in 0..n {
                let zrow = &proj.z[i * proj.q..(i + 1) * proj.q];
                let mut projv = 0.0_f64;
                for a in 0..proj.q {
                    projv += zrow[a] * coeff[a];
                }
                col[i] -= projv;
            }
        });
    };
    if pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false) {
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
    } else {
        for col in data.chunks_mut(n) {
            let mut ztv = vec![0.0_f64; proj.q];
            let mut coeff = vec![0.0_f64; proj.q];
            proj.ztv_f64(col, &mut ztv);
            proj.apply_inv(&ztv, &mut coeff);
            for i in 0..n {
                let zrow = &proj.z[i * proj.q..(i + 1) * proj.q];
                let mut projv = 0.0_f64;
                for a in 0..proj.q {
                    projv += zrow[a] * coeff[a];
                }
                col[i] -= projv;
            }
        }
    }
    Ok(())
}

fn dual_ridge_solve_alpha(
    x: &DMatrix<f64>,
    y: &[f64],
    ridge_lambda: f64,
    _max_iter: usize,
    _tol: f64,
) -> Result<Vec<f64>, String> {
    let n = x.nrows();
    if y.len() != n {
        return Err("dual ridge y length mismatch".to_string());
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut k = x * x.transpose();
    if ridge_lambda != 0.0_f64 {
        for i in 0..n {
            k[(i, i)] += ridge_lambda;
        }
    }
    let yv = DVector::<f64>::from_column_slice(y);
    if let Some(chol) = k.clone().cholesky() {
        let alpha = chol.solve(&yv);
        return Ok(alpha.as_slice().to_vec());
    }
    let lu = k.lu();
    if let Some(alpha) = lu.solve(&yv) {
        return Ok(alpha.as_slice().to_vec());
    }
    Err("failed to solve dense dual ridge system".to_string())
}

fn dual_ridge_beta(
    x: &DMatrix<f64>,
    y: &[f64],
    ridge_lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, String> {
    let alpha = dual_ridge_solve_alpha(x, y, ridge_lambda, max_iter, tol)?;
    let alpha_vec = DVector::<f64>::from_column_slice(&alpha);
    let beta = x.transpose() * alpha_vec;
    Ok(beta.as_slice().to_vec())
}

fn primal_beta_from_gram(xtx: &DMatrix<f64>, xty: &DVector<f64>) -> Result<Vec<f64>, String> {
    if xtx.nrows() != xtx.ncols() || xtx.nrows() != xty.len() {
        return Err("primal beta shape mismatch".to_string());
    }
    if xtx.nrows() == 0 {
        return Ok(Vec::new());
    }
    if let Some(chol) = xtx.clone().cholesky() {
        let beta = chol.solve(xty);
        return Ok(beta.as_slice().to_vec());
    }
    let lu = xtx.clone().lu();
    if let Some(beta) = lu.solve(xty) {
        return Ok(beta.as_slice().to_vec());
    }
    Err("failed to solve primal normal equations".to_string())
}

fn msgps_alasso_weight_beta(
    x: &DMatrix<f64>,
    y: &[f64],
    ridge_lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, String> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return Ok(vec![0.0_f64; p]);
    }
    if p < n {
        let mut xtx = x.transpose() * x;
        for j in 0..p {
            xtx[(j, j)] += ridge_lambda;
        }
        let xty = x.transpose() * DVector::<f64>::from_column_slice(y);
        return primal_beta_from_gram(&xtx, &xty);
    }
    dual_ridge_beta(x, y, ridge_lambda, max_iter, tol)
}

fn msgps_standardized_beta_ols(
    x: &DMatrix<f64>,
    y: &[f64],
    ridge_lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, String> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return Ok(vec![0.0_f64; p]);
    }
    if n <= p {
        return dual_ridge_beta(x, y, ridge_lambda, max_iter, tol);
    }
    let xtx = x.transpose() * x;
    let det = xtx.clone().lu().determinant();
    let xty = x.transpose() * DVector::<f64>::from_column_slice(y);
    if !det.is_finite() || det < 1e-3_f64 {
        let mut xtx_ridge = xtx;
        for j in 0..p {
            xtx_ridge[(j, j)] += ridge_lambda;
        }
        return primal_beta_from_gram(&xtx_ridge, &xty);
    }
    primal_beta_from_gram(&xtx, &xty)
}

#[inline]
fn matvec_row_major(a: &[f64], dim: usize, rhs: &[f64], out: &mut [f64]) {
    for r in 0..dim {
        let row = &a[r * dim..(r + 1) * dim];
        out[r] = row.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum();
    }
}

#[inline]
fn pinv_xtx(xtx: &[f64], q: usize) -> Result<Vec<f64>, String> {
    if q == 0 {
        return Ok(Vec::new());
    }
    let a = DMatrix::<f64>::from_row_slice(q, q, xtx);
    let svd = a.svd(true, true);
    let u = svd.u.ok_or_else(|| "SVD failed to produce U".to_string())?;
    let vt = svd
        .v_t
        .ok_or_else(|| "SVD failed to produce V^T".to_string())?;
    let mut s_inv = DMatrix::<f64>::zeros(q, q);
    let smax = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = 1e-12_f64 * smax.max(1.0_f64);
    for i in 0..q {
        let s = svd.singular_values[i];
        if s.is_finite() && s > cutoff {
            s_inv[(i, i)] = 1.0_f64 / s;
        }
    }
    let v = vt.transpose();
    Ok((v * s_inv * u.transpose()).as_slice().to_vec())
}

#[inline]
fn weighted_lambda_max(xty: &[f32], weights: &[f32]) -> f32 {
    let mut out = 0.0_f32;
    for j in 0..xty.len() {
        let w = weights[j];
        if w > 0.0_f32 {
            out = out.max(xty[j].abs() / w);
        }
    }
    out
}

#[derive(Clone, Debug)]
struct CachedMsgpsStdRow {
    values: Vec<f64>,
    scale: f64,
}

#[derive(Debug)]
struct MsgpsXtxCacheEntry {
    spill_slot: usize,
    last_access: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct MsgpsXtxBlockKey {
    row_idx: usize,
    block_idx: usize,
}

#[derive(Debug)]
struct MsgpsXtxHotBlockEntry {
    values: Vec<f64>,
    last_access: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MsgpsXtxCacheStats {
    hot_blocks: usize,
    hot_block_bytes: usize,
    spill_rows: usize,
    spill_file_bytes: usize,
}

#[derive(Default)]
struct MsgpsXtxScratch {
    row_ids: Vec<usize>,
    block: Vec<f64>,
    tmp: Vec<f64>,
}

impl MsgpsXtxScratch {
    fn prepare(&mut self, block_rows: usize, n_samples: usize) {
        let block_len = block_rows.saturating_mul(n_samples);
        if self.block.len() < block_len {
            self.block.resize(block_len, 0.0_f64);
        }
        if self.tmp.len() < block_rows {
            self.tmp.resize(block_rows, 0.0_f64);
        }
        self.row_ids.clear();
        self.row_ids.reserve(block_rows);
    }
}

#[inline]
fn fill_contiguous_row_ids(buf: &mut Vec<usize>, row_start: usize, row_end: usize) {
    buf.clear();
    buf.extend(row_start..row_end);
}

struct MsgpsXtxSpill {
    path: PathBuf,
    file: File,
    mmap: Option<MmapMut>,
    entry_len: usize,
    block_len: usize,
    entry_bytes: usize,
    slots_capacity: usize,
}

impl MsgpsXtxSpill {
    fn create_unique_file(dir: &Path) -> Result<(File, PathBuf), String> {
        create_dir_all(dir)
            .map_err(|e| format!("failed to create xtx spill dir {}: {e}", dir.display()))?;
        let pid = std::process::id();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for attempt in 0..32usize {
            let path = dir.join(format!("janusx_algwas_xtx_{pid}_{ts}_{attempt}.bin"));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => return Ok((file, path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(format!(
                        "failed to create xtx spill file {}: {e}",
                        path.display()
                    ));
                }
            }
        }
        Err("failed to allocate unique xtx spill file".to_string())
    }

    fn new(entry_len: usize, block_len: usize) -> Result<Self, String> {
        let entry_bytes = entry_len
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| "xtx spill entry size overflow".to_string())?;
        let spill_dir = std::env::var("JX_ALGWAS_XTX_SPILL_DIR")
            .ok()
            .map(|s| PathBuf::from(s.trim()))
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(std::env::temp_dir);
        let (file, path) = Self::create_unique_file(&spill_dir)?;
        Ok(Self {
            path,
            file,
            mmap: None,
            entry_len,
            block_len: block_len.max(1),
            entry_bytes,
            slots_capacity: 0,
        })
    }

    fn ensure_capacity(&mut self, slots_needed: usize) -> Result<(), String> {
        if slots_needed <= self.slots_capacity || self.entry_bytes == 0 {
            return Ok(());
        }
        let mut new_slots = self.slots_capacity.max(1);
        while new_slots < slots_needed {
            let doubled = new_slots.saturating_mul(2);
            if doubled <= new_slots {
                new_slots = slots_needed;
                break;
            }
            new_slots = doubled;
        }
        let new_bytes = new_slots
            .checked_mul(self.entry_bytes)
            .ok_or_else(|| "xtx spill file size overflow".to_string())?;
        self.mmap = None;
        self.file.set_len(new_bytes as u64).map_err(|e| {
            format!(
                "failed to resize xtx spill file {}: {e}",
                self.path.display()
            )
        })?;
        let mmap = unsafe {
            MmapOptions::new()
                .len(new_bytes)
                .map_mut(&self.file)
                .map_err(|e| {
                    format!("failed to mmap xtx spill file {}: {e}", self.path.display())
                })?
        };
        self.mmap = Some(mmap);
        self.slots_capacity = new_slots;
        Ok(())
    }

    #[inline]
    fn n_blocks(&self) -> usize {
        self.entry_len.div_ceil(self.block_len.max(1))
    }

    #[inline]
    fn block_bounds(&self, block_idx: usize) -> Result<(usize, usize), String> {
        let n_blocks = self.n_blocks();
        if block_idx >= n_blocks {
            return Err(format!(
                "xtx spill block out of range: {block_idx} >= {n_blocks}"
            ));
        }
        let start = block_idx
            .checked_mul(self.block_len)
            .ok_or_else(|| "xtx spill block offset overflow".to_string())?;
        let end = (start + self.block_len).min(self.entry_len);
        Ok((start, end))
    }

    fn write_block(&mut self, slot: usize, block_idx: usize, values: &[f64]) -> Result<(), String> {
        let (elem_start, elem_end) = self.block_bounds(block_idx)?;
        if values.len() != elem_end - elem_start {
            return Err("xtx spill block write length mismatch".to_string());
        }
        self.ensure_capacity(slot + 1)?;
        if self.entry_bytes == 0 {
            return Ok(());
        }
        let entry_start = slot
            .checked_mul(self.entry_bytes)
            .ok_or_else(|| "xtx spill slot offset overflow".to_string())?;
        let start = entry_start
            .checked_add(
                elem_start
                    .checked_mul(size_of::<f64>())
                    .ok_or_else(|| "xtx spill block byte offset overflow".to_string())?,
            )
            .ok_or_else(|| "xtx spill block start overflow".to_string())?;
        let end = start
            .checked_add(values.len().saturating_mul(size_of::<f64>()))
            .ok_or_else(|| "xtx spill block end overflow".to_string())?;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                values.len().saturating_mul(size_of::<f64>()),
            )
        };
        let mmap = self
            .mmap
            .as_mut()
            .ok_or_else(|| "xtx spill mmap unavailable".to_string())?;
        mmap[start..end].copy_from_slice(bytes);
        Ok(())
    }

    fn read_block_vec(&self, slot: usize, block_idx: usize) -> Result<Vec<f64>, String> {
        if slot >= self.slots_capacity {
            return Err(format!(
                "xtx spill slot out of range: {slot} >= {}",
                self.slots_capacity
            ));
        }
        let (elem_start, elem_end) = self.block_bounds(block_idx)?;
        if self.entry_bytes == 0 || elem_start == elem_end {
            return Ok(Vec::new());
        }
        let entry_start = slot
            .checked_mul(self.entry_bytes)
            .ok_or_else(|| "xtx spill slot offset overflow".to_string())?;
        let start = entry_start
            .checked_add(
                elem_start
                    .checked_mul(size_of::<f64>())
                    .ok_or_else(|| "xtx spill block byte offset overflow".to_string())?,
            )
            .ok_or_else(|| "xtx spill block start overflow".to_string())?;
        let end = start
            .checked_add((elem_end - elem_start).saturating_mul(size_of::<f64>()))
            .ok_or_else(|| "xtx spill block end overflow".to_string())?;
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| "xtx spill mmap unavailable".to_string())?;
        let bytes = &mmap[start..end];
        let ptr = bytes.as_ptr() as *const f64;
        debug_assert_eq!(ptr.align_offset(std::mem::align_of::<f64>()), 0);
        let vals = unsafe { std::slice::from_raw_parts(ptr, elem_end - elem_start) };
        Ok(vals.to_vec())
    }

    #[inline]
    fn allocated_bytes(&self) -> usize {
        self.slots_capacity.saturating_mul(self.entry_bytes)
    }
}

impl Drop for MsgpsXtxSpill {
    fn drop(&mut self) {
        let _ = self.mmap.take();
        let _ = std::fs::remove_file(&self.path);
    }
}

struct MsgpsXtxCache {
    entries: HashMap<usize, MsgpsXtxCacheEntry>,
    hot_blocks: HashMap<MsgpsXtxBlockKey, MsgpsXtxHotBlockEntry>,
    hot_lru: BinaryHeap<Reverse<(u64, MsgpsXtxBlockKey)>>,
    spill: MsgpsXtxSpill,
    next_spill_slot: usize,
    hot_budget_bytes: usize,
    hot_bytes: usize,
    touch_counter: u64,
    log_enabled: bool,
    log_every: usize,
    insert_count: usize,
    hot_event_count: usize,
}

impl MsgpsXtxCache {
    fn new(vector_len: usize, cache_cap_hint: usize, block_len: usize) -> Result<Self, String> {
        let cap = cache_cap_hint.max(1);
        let block_bytes = block_len
            .max(1)
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| "xtx hot block size overflow".to_string())?;
        let cache_mb = std::env::var("JX_ALGWAS_XTX_CACHE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_STAGE1_MSGPS_XTX_CACHE_MB);
        let budget_bytes = cache_mb
            .saturating_mul(1024usize * 1024usize)
            .max(block_bytes);
        Ok(Self {
            entries: HashMap::with_capacity(cap),
            hot_blocks: HashMap::new(),
            hot_lru: BinaryHeap::new(),
            spill: MsgpsXtxSpill::new(vector_len, block_len)?,
            next_spill_slot: 0,
            hot_budget_bytes: budget_bytes,
            hot_bytes: 0,
            touch_counter: 0,
            log_enabled: algwas_xtx_cache_log_enabled(),
            log_every: algwas_xtx_cache_log_every(),
            insert_count: 0,
            hot_event_count: 0,
        })
    }

    fn contains_key(&self, row_idx: usize) -> bool {
        self.entries.contains_key(&row_idx)
    }

    fn reserve_slot(&mut self) -> usize {
        let slot = self.next_spill_slot;
        self.next_spill_slot = self.next_spill_slot.saturating_add(1);
        slot
    }

    fn write_block(&mut self, slot: usize, block_idx: usize, values: &[f64]) -> Result<(), String> {
        self.spill.write_block(slot, block_idx, values)
    }

    fn hot_block_bytes(values: &[f64]) -> usize {
        values.len().saturating_mul(size_of::<f64>())
    }

    fn evict_hot_blocks_if_needed(&mut self, protect: Option<MsgpsXtxBlockKey>) {
        while self.hot_bytes > self.hot_budget_bytes {
            let Some(Reverse((last_access, key))) = self.hot_lru.pop() else {
                break;
            };
            if Some(key) == protect {
                self.hot_lru.push(Reverse((last_access, key)));
                break;
            }
            let Some(entry) = self.hot_blocks.get(&key) else {
                continue;
            };
            if entry.last_access != last_access {
                continue;
            }
            let removed = self.hot_blocks.remove(&key).unwrap();
            self.hot_bytes = self
                .hot_bytes
                .saturating_sub(Self::hot_block_bytes(&removed.values));
        }
    }

    fn insert_hot_block(&mut self, key: MsgpsXtxBlockKey, values: Vec<f64>) -> Result<(), String> {
        let bytes = Self::hot_block_bytes(&values);
        let prev_stats = self.stats();
        self.touch_counter = self.touch_counter.wrapping_add(1);
        let touch = self.touch_counter;
        self.hot_event_count = self.hot_event_count.saturating_add(1);
        if let Some(old) = self.hot_blocks.insert(
            key,
            MsgpsXtxHotBlockEntry {
                values,
                last_access: touch,
            },
        ) {
            self.hot_bytes = self
                .hot_bytes
                .saturating_sub(Self::hot_block_bytes(&old.values));
        }
        self.hot_bytes = self.hot_bytes.saturating_add(bytes);
        self.hot_lru.push(Reverse((touch, key)));
        self.evict_hot_blocks_if_needed(Some(key));
        self.maybe_log_hot_update("promote", key, prev_stats);
        Ok(())
    }

    fn commit_spilled_row(&mut self, row_idx: usize, spill_slot: usize) {
        let prev_stats = self.stats();
        self.touch_counter = self.touch_counter.wrapping_add(1);
        self.entries.insert(
            row_idx,
            MsgpsXtxCacheEntry {
                spill_slot,
                last_access: self.touch_counter,
            },
        );
        self.insert_count = self.insert_count.saturating_add(1);
        self.maybe_log_update("insert", row_idx, prev_stats);
    }

    fn with_row_block<R>(
        &mut self,
        row_idx: usize,
        block_idx: usize,
        f: impl FnOnce(&[f64]) -> R,
    ) -> Result<R, String> {
        self.touch_counter = self.touch_counter.wrapping_add(1);
        let touch = self.touch_counter;
        let slot = {
            let entry = self
                .entries
                .get_mut(&row_idx)
                .ok_or_else(|| format!("xtx cache missing row {row_idx}"))?;
            entry.last_access = touch;
            entry.spill_slot
        };
        let key = MsgpsXtxBlockKey { row_idx, block_idx };
        if let Some((ptr, len)) = {
            let hot = self.hot_blocks.get_mut(&key);
            if let Some(entry) = hot {
                entry.last_access = touch;
                self.hot_lru.push(Reverse((touch, key)));
                Some((entry.values.as_ptr(), entry.values.len()))
            } else {
                None
            }
        } {
            let vals = unsafe { std::slice::from_raw_parts(ptr, len) };
            return Ok(f(vals));
        }

        let values = self.spill.read_block_vec(slot, block_idx)?;
        let out = f(&values);
        self.insert_hot_block(key, values)?;
        Ok(out)
    }

    fn for_each_block(
        &mut self,
        row_idx: usize,
        mut f: impl FnMut(usize, &[f64]) -> Result<(), String>,
    ) -> Result<(), String> {
        let n_blocks = self.spill.n_blocks();
        for block_idx in 0..n_blocks {
            let block_start = block_idx
                .checked_mul(self.spill.block_len)
                .ok_or_else(|| "xtx cache block start overflow".to_string())?;
            self.with_row_block(row_idx, block_idx, |vals| f(block_start, vals))??;
        }
        Ok(())
    }

    fn stats(&self) -> MsgpsXtxCacheStats {
        let hot_blocks = self.hot_blocks.len();
        let hot_block_bytes = self.hot_bytes;
        let spill_rows = self.entries.len();
        let spill_file_bytes = self.spill.allocated_bytes();
        MsgpsXtxCacheStats {
            hot_blocks,
            hot_block_bytes,
            spill_rows,
            spill_file_bytes,
        }
    }

    #[inline]
    fn is_spill_active(&self) -> bool {
        !self.entries.is_empty()
    }

    fn log_line(&self, label: &str, row_idx: Option<usize>) {
        let stats = self.stats();
        let row_part = row_idx
            .map(|idx| format!(", row={idx}"))
            .unwrap_or_default();
        eprintln!(
            "ALGWAS xtx cache {label}: hot_cache_blocks={} hot_cache_bytes={} spill_rows={} spill_file_size={}{}",
            stats.hot_blocks,
            format_bytes_binary(stats.hot_block_bytes),
            stats.spill_rows,
            format_bytes_binary(stats.spill_file_bytes),
            row_part,
        );
    }

    fn maybe_log_update(&self, label: &str, row_idx: usize, prev_stats: MsgpsXtxCacheStats) {
        if !self.log_enabled {
            return;
        }
        let stats = self.stats();
        let spill_changed = stats.spill_rows != prev_stats.spill_rows
            || stats.spill_file_bytes != prev_stats.spill_file_bytes;
        let periodic = self.log_every > 0 && self.insert_count % self.log_every == 0;
        if spill_changed || periodic {
            self.log_line(label, Some(row_idx));
        }
    }

    fn maybe_log_hot_update(
        &self,
        label: &str,
        key: MsgpsXtxBlockKey,
        prev_stats: MsgpsXtxCacheStats,
    ) {
        if !self.log_enabled {
            return;
        }
        let stats = self.stats();
        let hot_changed = stats.hot_blocks != prev_stats.hot_blocks
            || stats.hot_block_bytes != prev_stats.hot_block_bytes;
        let periodic = self.log_every > 0 && self.hot_event_count % self.log_every == 0;
        if hot_changed && periodic {
            self.log_line(label, Some(key.row_idx));
        }
    }

    fn maybe_log_final_summary(&self) {
        if self.log_enabled || self.is_spill_active() {
            self.log_line("final", None);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct F32TotalOrd(f32);

impl Eq for F32TotalOrd {}

impl PartialOrd for F32TotalOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F32TotalOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[inline]
fn algwas_stage1_weight_screen_cap(n_features: usize, qtn_bound: Option<usize>) -> usize {
    if n_features == 0 {
        return 0usize;
    }
    let env_cap = std::env::var("JX_ALGWAS_STAGE1_WEIGHT_SCREEN")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_STAGE1_ALASSO_WEIGHT_SCREEN);
    let qtn_floor = qtn_bound
        .unwrap_or(DEFAULT_STAGE1_MSGPS_PMAX)
        .saturating_mul(8)
        .max(512usize);
    env_cap.max(qtn_floor).min(n_features)
}

#[inline]
fn algwas_stage1_initial_working_set_cap(n_features: usize) -> usize {
    if n_features == 0 {
        return 0usize;
    }
    let env_cap = std::env::var("JX_ALGWAS_STAGE1_INITIAL_WORKING_SET")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_STAGE1_ALASSO_INITIAL_WORKING_SET);
    env_cap.min(n_features)
}

#[inline]
fn algwas_stage1_strong_set_cap(n_features: usize, active_cap: usize, active_len: usize) -> usize {
    if n_features == 0 || active_len >= active_cap {
        return 0usize;
    }
    let env_cap = std::env::var("JX_ALGWAS_STAGE1_STRONG_SET")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_STAGE1_ALASSO_STRONG_SET);
    env_cap
        .min(active_cap.saturating_sub(active_len))
        .min(n_features.saturating_sub(active_len))
}

fn top_feature_indices_by_score(scores: &[f32], k: usize) -> Vec<usize> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }
    if k >= scores.len() {
        let mut out: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter_map(|(idx, &score)| {
                if score.is_finite() && score > 0.0_f32 {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        out.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        return out;
    }

    let mut heap = BinaryHeap::<Reverse<(F32TotalOrd, usize)>>::with_capacity(k + 1);
    for (idx, &score) in scores.iter().enumerate() {
        if !score.is_finite() || score <= 0.0_f32 {
            continue;
        }
        let item = Reverse((F32TotalOrd(score), idx));
        if heap.len() < k {
            heap.push(item);
            continue;
        }
        if let Some(peek) = heap.peek() {
            if item.0 .0 > peek.0 .0 {
                let _ = heap.pop();
                heap.push(item);
            }
        }
    }

    let mut out = Vec::<usize>::with_capacity(heap.len());
    while let Some(Reverse((_score, idx))) = heap.pop() {
        out.push(idx);
    }
    out.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    out
}

fn select_stage1_lm_screen_rows(xty: &[f32], diag_xtx: &[f32], cap: usize) -> Vec<usize> {
    if cap == 0 || xty.is_empty() || diag_xtx.len() != xty.len() {
        return Vec::new();
    }
    let mut heap = BinaryHeap::<Reverse<(F32TotalOrd, usize)>>::with_capacity(cap + 1);
    for j in 0..xty.len() {
        let diag = diag_xtx[j];
        if diag.is_finite() && diag > 0.0_f32 {
            let score = xty[j].abs() / diag.sqrt().max(1e-12_f32);
            if score.is_finite() && score > 0.0_f32 {
                let item = Reverse((F32TotalOrd(score), j));
                if heap.len() < cap {
                    heap.push(item);
                } else if let Some(peek) = heap.peek() {
                    if item.0 .0 > peek.0 .0 {
                        let _ = heap.pop();
                        heap.push(item);
                    }
                }
            }
        }
    }

    let mut out = Vec::<usize>::with_capacity(heap.len());
    while let Some(Reverse((_score, idx))) = heap.pop() {
        out.push(idx);
    }
    out.sort_unstable_by(|&a, &b| {
        let da = diag_xtx[a].max(1e-12_f32).sqrt().max(1e-12_f32);
        let db = diag_xtx[b].max(1e-12_f32).sqrt().max(1e-12_f32);
        let sa = xty[a].abs() / da;
        let sb = xty[b].abs() / db;
        sb.total_cmp(&sa)
    });
    out
}

fn select_stage1_strong_rule_rows(
    last_grad: &[f32],
    active_mask: &[bool],
    penalty_weights: &[f32],
    prev_lambda: f32,
    next_lambda: f32,
    cap: usize,
) -> Vec<usize> {
    if cap == 0
        || last_grad.is_empty()
        || active_mask.len() != last_grad.len()
        || penalty_weights.len() != last_grad.len()
        || !prev_lambda.is_finite()
        || !next_lambda.is_finite()
        || next_lambda >= prev_lambda
    {
        return Vec::new();
    }
    let strong_thr = (2.0_f32 * next_lambda - prev_lambda).max(0.0_f32);
    let mut heap = BinaryHeap::<Reverse<(F32TotalOrd, usize)>>::with_capacity(cap + 1);
    for j in 0..last_grad.len() {
        if active_mask[j] {
            continue;
        }
        let w = penalty_weights[j].max(1e-12_f32);
        let score = last_grad[j].abs() / w;
        if !score.is_finite() || score <= strong_thr {
            continue;
        }
        let item = Reverse((F32TotalOrd(score), j));
        if heap.len() < cap {
            heap.push(item);
        } else if let Some(peek) = heap.peek() {
            if item.0 .0 > peek.0 .0 {
                let _ = heap.pop();
                heap.push(item);
            }
        }
    }
    let mut out = Vec::<usize>::with_capacity(heap.len());
    while let Some(Reverse((_score, idx))) = heap.pop() {
        out.push(idx);
    }
    out.sort_unstable_by(|&a, &b| {
        let wa = penalty_weights[a].max(1e-12_f32);
        let wb = penalty_weights[b].max(1e-12_f32);
        let sa = last_grad[a].abs() / wa;
        let sb = last_grad[b].abs() / wb;
        sb.total_cmp(&sa)
    });
    out
}

fn compute_msgps_alasso_init(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y_resid: &[f32],
    diag_xtx: &[f32],
    xty_hint: Option<&[f32]>,
    cfg: &AlgwasConfig,
) -> Result<AlgwasStage1AdaptiveInit, String> {
    let n_features = design.n_features();
    if y_resid.len() != design.n_samples() {
        return Err("alasso weight residual length mismatch".to_string());
    }
    if diag_xtx.len() != n_features {
        return Err("alasso weight diag length mismatch".to_string());
    }
    if let Some(xty) = xty_hint {
        if xty.len() != n_features {
            return Err("alasso weight xty length mismatch".to_string());
        }
    }

    let ridge_lambda = DEFAULT_STAGE1_ALASSO_RIDGE_LAMBDA.max(1e-8_f32);
    let dense_cells = n_features.saturating_mul(design.n_samples());
    if algwas_stage1_dense_weights_enabled()
        && dense_cells > 0
        && dense_cells <= DEFAULT_STAGE1_ALASSO_DENSE_WEIGHT_MAX_CELLS
    {
        let mut dense = vec![0.0_f32; dense_cells];
        let all_rows: Vec<usize> = (0..n_features).collect();
        design.decode_rows_standardized_into(&all_rows, &mut dense)?;
        residualize_block_rows_in_place(
            &mut dense,
            n_features,
            design.n_samples(),
            proj,
            design.pool.as_ref(),
        );
        let mut x64 = Vec::<f64>::with_capacity(dense.len());
        x64.extend(dense.iter().map(|&v| v as f64));
        let x = DMatrix::<f64>::from_row_slice(n_features, design.n_samples(), &x64);
        let mut gram = x.transpose() * &x;
        for i in 0..design.n_samples() {
            gram[(i, i)] += ridge_lambda as f64;
        }
        let rhs =
            DVector::<f64>::from_iterator(design.n_samples(), y_resid.iter().map(|&v| v as f64));
        let alpha = if let Some(chol) = gram.clone().cholesky() {
            chol.solve(&rhs)
        } else if let Some(sol) = gram.clone().lu().solve(&rhs) {
            sol
        } else {
            return Err("failed to solve dense ALGWAS ridge weight system".to_string());
        };
        let beta = x * alpha;
        let gamma = DEFAULT_STAGE1_ALASSO_GAMMA.max(1e-6_f32) as f64;
        let mut weights = vec![0.0_f32; n_features];
        let mut beta_init = vec![0.0_f32; n_features];
        for j in 0..n_features {
            let b = beta[j].abs().max(DEFAULT_STAGE1_ALASSO_WEIGHT_FLOOR as f64);
            let w = b.powf(-gamma);
            beta_init[j] = if beta[j].is_finite() {
                beta[j] as f32
            } else {
                0.0_f32
            };
            weights[j] = if w.is_finite() {
                (w as f32).min(DEFAULT_STAGE1_ALASSO_WEIGHT_CAP)
            } else {
                DEFAULT_STAGE1_ALASSO_WEIGHT_CAP
            };
        }
        return Ok(AlgwasStage1AdaptiveInit {
            penalty_weights: weights,
            beta_init,
        });
    }

    let rhs_owned;
    let rhs: &[f32] = if let Some(xty) = xty_hint {
        xty
    } else {
        rhs_owned = design.xty_residualized(y_resid)?;
        &rhs_owned
    };
    let mut beta0 = vec![0.0_f32; n_features];
    for j in 0..n_features {
        let denom = (diag_xtx[j] + ridge_lambda).max(1e-12_f32);
        beta0[j] = rhs[j] / denom;
    }
    let score_cap = (DEFAULT_STAGE1_ALASSO_DENSE_WEIGHT_MAX_CELLS / design.n_samples().max(1))
        .max(1usize)
        .min(n_features);
    let screen_cap = algwas_stage1_weight_screen_cap(n_features, cfg.qtn_bound).min(score_cap);
    let mut scores = vec![0.0_f32; n_features];
    for j in 0..n_features {
        scores[j] = beta0[j].abs();
    }
    let selected_rows = top_feature_indices_by_score(&scores, screen_cap);
    if !selected_rows.is_empty() {
        let mut dense = vec![0.0_f32; selected_rows.len() * design.n_samples()];
        design.decode_rows_standardized_into(&selected_rows, &mut dense)?;
        let mut x64 = Vec::<f64>::with_capacity(dense.len());
        x64.extend(dense.iter().map(|&v| v as f64));
        let mut x_sel = DMatrix::<f64>::from_vec(design.n_samples(), selected_rows.len(), x64);
        residualize_dense_columns_in_place(&mut x_sel, proj, design.pool.as_ref())?;
        let y64: Vec<f64> = y_resid.iter().map(|&v| v as f64).collect();
        if let Ok(beta_sel) = msgps_alasso_weight_beta(
            &x_sel,
            &y64,
            ridge_lambda as f64,
            cfg.lasso
                .max_pcg_iter
                .max(DEFAULT_STAGE1_ALASSO_PCG_MAX_ITERS),
            cfg.lasso.pcg_tol.max(DEFAULT_STAGE1_ALASSO_PCG_TOL),
        ) {
            for (local, &idx) in selected_rows.iter().enumerate() {
                if local < beta_sel.len() {
                    beta0[idx] = beta_sel[local] as f32;
                }
            }
        }
    }

    let gamma = DEFAULT_STAGE1_ALASSO_GAMMA.max(1e-6_f32);
    let mut weights = vec![0.0_f32; n_features];
    for j in 0..n_features {
        let b = beta0[j].abs().max(DEFAULT_STAGE1_ALASSO_WEIGHT_FLOOR);
        let w = b.powf(-gamma);
        weights[j] = if w.is_finite() {
            w.min(DEFAULT_STAGE1_ALASSO_WEIGHT_CAP)
        } else {
            DEFAULT_STAGE1_ALASSO_WEIGHT_CAP
        };
    }
    for bj in &mut beta0 {
        if !bj.is_finite() {
            *bj = 0.0_f32;
        }
    }
    Ok(AlgwasStage1AdaptiveInit {
        penalty_weights: weights,
        beta_init: beta0,
    })
}

#[derive(Clone, Debug)]
struct AlgwasStage1AdaptiveInit {
    penalty_weights: Vec<f32>,
    beta_init: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct AlgwasStage1PathPoint {
    pub lambda: f32,
    pub bic: f64,
    pub ebic: f64,
    pub rss: f64,
    pub nnz: usize,
    pub df: f64,
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct AlgwasStage1Result {
    pub selected_indices: Vec<usize>,
    pub beta: Vec<f32>,
    pub lambda_best: f32,
    pub bic_best: f64,
    pub ebic_best: f64,
    pub best_step: usize,
    pub path: Vec<AlgwasStage1PathPoint>,
    pub converged: bool,
}

#[derive(Clone, Debug)]
struct AlgwasStage1Aux {
    last_kkt_grad: Option<Vec<f32>>,
    last_lambda: Option<f32>,
}

type AlgwasStage1WarmState = ActivePathState<AlgwasStage1Aux>;

#[derive(Clone, Debug)]
pub struct AlgwasConfig {
    pub lambda_steps: usize,
    pub lambda_min_ratio: f32,
    pub qtn_bound: Option<usize>,
    pub lasso: crate::lasso::LassoConfig,
}

impl Default for AlgwasConfig {
    fn default() -> Self {
        Self {
            lambda_steps: DEFAULT_STAGE1_PATH_STEPS,
            lambda_min_ratio: DEFAULT_STAGE1_LAMBDA_MIN_RATIO,
            qtn_bound: None,
            lasso: crate::lasso::LassoConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
struct CovariateProjection {
    n: usize,
    inv_n: f64,
    q: usize,
    z: Vec<f64>,
    ztz_inv: Vec<f64>,
}

impl CovariateProjection {
    fn new(x_cov: &[f64], n: usize, p_cov: usize) -> Result<Self, String> {
        if n == 0 {
            return Err("covariate projection requires n > 0".to_string());
        }
        if p_cov > 0 && x_cov.len() != n.saturating_mul(p_cov) {
            return Err(format!(
                "x_cov length mismatch: got {}, expected {}",
                x_cov.len(),
                n.saturating_mul(p_cov)
            ));
        }
        if x_cov.iter().any(|v| !v.is_finite()) {
            return Err("x_cov contains non-finite values".to_string());
        }
        let q = 1usize.saturating_add(p_cov);
        if n <= q {
            return Err(format!(
                "covariate projection requires n > q with intercept: n={n}, q={q}"
            ));
        }
        let mut z = vec![0.0_f64; n * q];
        for i in 0..n {
            z[i * q] = 1.0_f64;
            if p_cov > 0 {
                let src = &x_cov[i * p_cov..(i + 1) * p_cov];
                let dst = &mut z[i * q + 1..(i + 1) * q];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = s;
                }
            }
        }
        let mut ztz = vec![0.0_f64; q * q];
        for i in 0..n {
            let row = &z[i * q..(i + 1) * q];
            for a in 0..q {
                let va = row[a];
                for b in 0..=a {
                    ztz[a * q + b] += va * row[b];
                }
            }
        }
        for a in 0..q {
            for b in 0..a {
                ztz[b * q + a] = ztz[a * q + b];
            }
        }
        let ztz_inv = pinv_xtx(&ztz, q)?;
        Ok(Self {
            n,
            inv_n: 1.0_f64 / n as f64,
            q,
            z,
            ztz_inv,
        })
    }

    #[inline]
    fn is_intercept_only(&self) -> bool {
        self.q == 1
    }

    #[inline]
    fn ztv_f32(&self, v: &[f32], out: &mut [f64]) {
        debug_assert_eq!(v.len(), self.n);
        debug_assert_eq!(out.len(), self.q);
        if self.is_intercept_only() {
            out[0] = v.iter().map(|&x| x as f64).sum::<f64>();
            return;
        }
        out.fill(0.0_f64);
        for i in 0..self.n {
            let vi = v[i] as f64;
            let row = &self.z[i * self.q..(i + 1) * self.q];
            for c in 0..self.q {
                out[c] += row[c] * vi;
            }
        }
    }

    #[inline]
    fn ztv_f64(&self, v: &[f64], out: &mut [f64]) {
        debug_assert_eq!(v.len(), self.n);
        debug_assert_eq!(out.len(), self.q);
        if self.is_intercept_only() {
            out[0] = v.iter().sum::<f64>();
            return;
        }
        out.fill(0.0_f64);
        for i in 0..self.n {
            let vi = v[i];
            let row = &self.z[i * self.q..(i + 1) * self.q];
            for c in 0..self.q {
                out[c] += row[c] * vi;
            }
        }
    }

    #[inline]
    fn apply_inv(&self, rhs: &[f64], out: &mut [f64]) {
        debug_assert_eq!(rhs.len(), self.q);
        debug_assert_eq!(out.len(), self.q);
        if self.is_intercept_only() {
            out[0] = rhs[0] * self.inv_n;
            return;
        }
        matvec_row_major(&self.ztz_inv, self.q, rhs, out);
    }

    fn residualize_y(&self, y: &[f64]) -> Result<Vec<f32>, String> {
        if y.len() != self.n {
            return Err("y length mismatch in residualization".to_string());
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err("y contains non-finite values".to_string());
        }
        if self.is_intercept_only() {
            let mean = y.iter().sum::<f64>() * self.inv_n;
            return Ok(y.iter().map(|&v| (v - mean) as f32).collect());
        }
        let mut zty = vec![0.0_f64; self.q];
        self.ztv_f64(y, &mut zty);
        let mut coeff = vec![0.0_f64; self.q];
        self.apply_inv(&zty, &mut coeff);
        let mut out = vec![0.0_f32; self.n];
        for i in 0..self.n {
            let mut proj = 0.0_f64;
            let row = &self.z[i * self.q..(i + 1) * self.q];
            for c in 0..self.q {
                proj += row[c] * coeff[c];
            }
            out[i] = (y[i] - proj) as f32;
        }
        Ok(out)
    }
}

#[derive(Clone, Debug)]
struct AlgwasPackedDesign<'a> {
    packed: Cow<'a, [u8]>,
    bytes_per_snp: usize,
    n_samples_full: usize,
    n_samples_used: usize,
    n_features: usize,
    sample_idx: Vec<usize>,
    full_sample_fast: bool,
    row_flip: Vec<bool>,
    row_maf: Vec<f32>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    row_indices: Option<Vec<usize>>,
    block_rows: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl<'a> AlgwasPackedDesign<'a> {
    #[inline]
    fn standardized_row_value_lut(&self, row_idx: usize) -> [f32; 4] {
        standardized_value_lut_algwas(
            self.row_flip[row_idx],
            self.row_mean[row_idx],
            self.row_inv_sd[row_idx],
        )
    }

    #[inline]
    fn packed_row_slice(&self, row_idx: usize) -> &[u8] {
        let packed_idx = self
            .row_indices
            .as_ref()
            .map(|v| v[row_idx])
            .unwrap_or(row_idx);
        &self.packed[packed_idx * self.bytes_per_snp..(packed_idx + 1) * self.bytes_per_snp]
    }

    #[inline]
    fn standardized_row_dot_into(
        &self,
        row_idx: usize,
        sample_vec: &[f32],
        code4_lut: &[[u8; 4]; 256],
    ) -> f64 {
        let row = self.packed_row_slice(row_idx);
        let value_lut = self.standardized_row_value_lut(row_idx);
        if self.full_sample_fast {
            packed_row_standardized_dot_full_algwas(
                row,
                self.n_samples_used,
                sample_vec,
                &value_lut,
                code4_lut,
            )
        } else {
            packed_row_standardized_dot_selected_algwas(
                row,
                &self.sample_idx,
                sample_vec,
                &value_lut,
            )
        }
    }

    #[inline]
    fn standardized_row_sum_sumsq(&self, row_idx: usize, code4_lut: &[[u8; 4]; 256]) -> (f64, f64) {
        let row = self.packed_row_slice(row_idx);
        let value_lut = self.standardized_row_value_lut(row_idx);
        if self.full_sample_fast {
            (
                packed_row_standardized_sum_full_algwas(
                    row,
                    self.n_samples_used,
                    &value_lut,
                    code4_lut,
                ),
                packed_row_standardized_sumsq_full_algwas(
                    row,
                    self.n_samples_used,
                    &value_lut,
                    code4_lut,
                ),
            )
        } else {
            (
                packed_row_standardized_sum_selected_algwas(row, &self.sample_idx, &value_lut),
                packed_row_standardized_sumsq_selected_algwas(row, &self.sample_idx, &value_lut),
            )
        }
    }

    fn from_parts(
        packed: Cow<'a, [u8]>,
        n_samples_full: usize,
        row_flip: Vec<bool>,
        row_maf: Vec<f32>,
        sample_idx: Vec<usize>,
        row_indices: Option<Vec<usize>>,
        threads: usize,
    ) -> Result<Self, String> {
        if n_samples_full == 0 {
            return Err("packed design requires n_samples_full > 0".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        if bytes_per_snp == 0 || packed.len() % bytes_per_snp != 0 {
            return Err("packed design length mismatch".to_string());
        }
        let n_features = row_flip.len();
        if row_maf.len() != n_features {
            return Err("row_maf length mismatch".to_string());
        }
        if let Some(row_idx) = row_indices.as_ref() {
            if row_idx.len() != n_features {
                return Err("row_indices length mismatch".to_string());
            }
            if row_idx.iter().any(|&v| v >= packed.len() / bytes_per_snp) {
                return Err("row_indices out of range".to_string());
            }
        } else if packed.len() / bytes_per_snp != n_features {
            return Err("packed row count mismatch".to_string());
        }
        let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
        let row_stats = build_row_standardization_stats(
            &packed,
            bytes_per_snp,
            &row_flip,
            &row_maf,
            &sample_idx,
            row_indices.as_deref(),
            DEFAULT_STANDARDIZE_EPS32,
            false,
            pool.as_ref(),
        )?;
        let n_samples_used = sample_idx.len();
        let sample_idx_len = n_samples_used;
        let full_sample_fast = is_identity_indices(&sample_idx, n_samples_full);
        let base_rows = DEFAULT_STAGE1_BLOCK_ROWS;
        Ok(Self {
            packed,
            bytes_per_snp,
            n_samples_full,
            n_samples_used,
            n_features,
            sample_idx,
            full_sample_fast,
            row_flip,
            row_maf,
            row_mean: row_stats.row_mean,
            row_inv_sd: row_stats.row_inv_sd,
            row_indices,
            block_rows: adaptive_grm_block_rows(
                base_rows,
                n_features,
                sample_idx_len,
                0usize,
                threads,
            )
            .max(1),
            pool,
        })
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.n_features
    }

    #[inline]
    fn n_samples(&self) -> usize {
        self.n_samples_used
    }

    fn decode_all_rows_raw_feature_major_f64(&self) -> Result<DMatrix<f64>, String> {
        let mut data = vec![0.0_f64; self.n_samples_used * self.n_features];
        let mut fill = || {
            data.par_chunks_mut(self.n_samples_used)
                .enumerate()
                .try_for_each(|(idx, col_out)| -> Result<(), String> {
                    let packed_idx = self.row_indices.as_ref().map(|v| v[idx]).unwrap_or(idx);
                    let row = &self.packed
                        [packed_idx * self.bytes_per_snp..(packed_idx + 1) * self.bytes_per_snp];
                    let flip = self.row_flip[idx];
                    let fallback_mean =
                        (2.0_f64 * self.row_maf[idx] as f64).clamp(0.0_f64, 2.0_f64);
                    let mut sum = 0.0_f64;
                    let mut n_obs = 0usize;
                    for (i, &sidx) in self.sample_idx.iter().enumerate() {
                        let b = row[sidx >> 2];
                        let code = (b >> ((sidx & 3) * 2)) & 0b11;
                        let maybe = decode_plink_bed_hardcall(code);
                        if let Some(mut gv) = maybe {
                            if flip {
                                gv = 2.0_f64 - gv;
                            }
                            col_out[i] = gv;
                            sum += gv;
                            n_obs += 1;
                        } else {
                            col_out[i] = f64::NAN;
                        }
                    }
                    let mean = if n_obs > 0 {
                        sum / n_obs as f64
                    } else {
                        fallback_mean
                    };
                    for v in col_out.iter_mut() {
                        if !v.is_finite() {
                            *v = mean;
                        }
                    }
                    Ok(())
                })
        };
        if let Some(tp) = self.pool.as_ref() {
            tp.install(fill)?;
        } else {
            fill()?;
        }
        Ok(DMatrix::<f64>::from_vec(
            self.n_samples_used,
            self.n_features,
            data,
        ))
    }

    fn decode_rows_raw_into_f64(&self, rows: &[usize], out: &mut [f64]) -> Result<(), String> {
        if rows.is_empty() {
            return Ok(());
        }
        if out.len() != rows.len() * self.n_samples_used {
            return Err("raw decode output length mismatch".to_string());
        }
        let n = self.n_samples_used;
        let full_sample_fast = self.full_sample_fast;
        let sample_idx = &self.sample_idx;
        let packed = &self.packed;
        let bytes_per_snp = self.bytes_per_snp;
        let row_flip = &self.row_flip;
        let row_maf = &self.row_maf;
        let row_indices = self.row_indices.as_ref();
        let code4_lut = &packed_byte_lut().code4;
        let decode_one = |local_row: usize, out_row: &mut [f64]| -> Result<(), String> {
            let row_idx_local = rows[local_row];
            if row_idx_local >= self.n_features {
                return Err(format!("selected row index out of range: {row_idx_local}"));
            }
            let row_idx_packed = row_indices
                .map(|idx| idx[row_idx_local])
                .unwrap_or(row_idx_local);
            let row = &packed[row_idx_packed * bytes_per_snp..(row_idx_packed + 1) * bytes_per_snp];
            let flip = row_flip[row_idx_local];
            let fallback_mean = (2.0_f64 * row_maf[row_idx_local] as f64).clamp(0.0_f64, 2.0_f64);
            let mut sum = 0.0_f64;
            let mut n_obs = 0usize;
            if full_sample_fast {
                let value_lut: [f64; 4] = if flip {
                    [2.0_f64, -9.0_f64, 1.0_f64, 0.0_f64]
                } else {
                    [0.0_f64, -9.0_f64, 1.0_f64, 2.0_f64]
                };
                decode_row_centered_full_lut_f64(
                    row,
                    self.n_samples_full,
                    code4_lut,
                    &value_lut,
                    out_row,
                );
                for v in out_row.iter_mut().take(n) {
                    if *v >= 0.0_f64 {
                        sum += *v;
                        n_obs += 1;
                    } else {
                        *v = f64::NAN;
                    }
                }
            } else {
                for (i, &sidx) in sample_idx.iter().enumerate() {
                    let b = row[sidx >> 2];
                    let code = (b >> ((sidx & 3) * 2)) & 0b11;
                    let maybe = decode_plink_bed_hardcall(code);
                    if let Some(mut gv) = maybe {
                        if flip {
                            gv = 2.0_f64 - gv;
                        }
                        out_row[i] = gv;
                        sum += gv;
                        n_obs += 1;
                    } else {
                        out_row[i] = f64::NAN;
                    }
                }
            }
            let mean = if n_obs > 0 {
                sum / n_obs as f64
            } else {
                fallback_mean
            };
            for v in out_row.iter_mut() {
                if !v.is_finite() {
                    *v = mean;
                }
            }
            Ok(())
        };
        if current_pool_threads(self.pool.as_ref()) > 1 {
            if let Some(tp) = self.pool.as_ref() {
                tp.install(|| {
                    out.par_chunks_mut(n)
                        .enumerate()
                        .try_for_each(|(local_row, out_row)| decode_one(local_row, out_row))
                })?;
            } else {
                for (local_row, out_row) in out.chunks_mut(n).enumerate() {
                    decode_one(local_row, out_row)?;
                }
            }
        } else {
            for (local_row, out_row) in out.chunks_mut(n).enumerate() {
                decode_one(local_row, out_row)?;
            }
        }
        Ok(())
    }

    fn decode_rows_resid_base_into_f64(
        &self,
        rows: &[usize],
        proj: &CovariateProjection,
        out: &mut [f64],
    ) -> Result<(), String> {
        self.decode_rows_raw_into_f64(rows, out)?;
        if proj.q > 1 {
            residualize_block_rows_in_place_f64(
                out,
                rows.len(),
                self.n_samples_used,
                proj,
                self.pool.as_ref(),
            );
        }
        Ok(())
    }

    fn decode_rows_resid_standardized_into_f64(
        &self,
        rows: &[usize],
        proj: &CovariateProjection,
        out: &mut [f64],
        scales_out: Option<&mut [f64]>,
    ) -> Result<(), String> {
        self.decode_rows_resid_base_into_f64(rows, proj, out)?;
        center_and_scale_block_rows_in_place_f64(
            out,
            rows.len(),
            self.n_samples_used,
            scales_out,
            self.pool.as_ref(),
        );
        Ok(())
    }

    fn decode_rows_standardized_into(&self, rows: &[usize], out: &mut [f32]) -> Result<(), String> {
        if rows.is_empty() {
            return Ok(());
        }
        if out.len() != rows.len() * self.n_samples_used {
            return Err("decode output length mismatch".to_string());
        }
        let mut sel_flip = Vec::with_capacity(rows.len());
        let mut sel_mean = Vec::with_capacity(rows.len());
        let mut sel_inv_sd = Vec::with_capacity(rows.len());
        let mut packed_rows = Vec::with_capacity(rows.len());
        for &idx in rows {
            if idx >= self.n_features {
                return Err(format!("selected row index out of range: {idx}"));
            }
            sel_flip.push(self.row_flip[idx]);
            sel_mean.push(self.row_mean[idx]);
            sel_inv_sd.push(self.row_inv_sd[idx]);
            packed_rows.push(self.row_indices.as_ref().map(|v| v[idx]).unwrap_or(idx));
        }
        let code4_lut = &packed_byte_lut().code4;
        decode_standardized_packed_block_rows_f32(
            &self.packed,
            self.bytes_per_snp,
            self.n_samples_full,
            &sel_flip,
            &sel_mean,
            &sel_inv_sd,
            &self.sample_idx,
            self.full_sample_fast,
            Some(&packed_rows),
            0,
            out,
            code4_lut,
            self.pool.as_ref(),
        )?;
        Ok(())
    }

    fn xty_residualized(&self, y_resid: &[f32]) -> Result<Vec<f32>, String> {
        if y_resid.len() != self.n_samples_used {
            return Err("residualized response length mismatch".to_string());
        }
        validate_finite_slice_f32("residualized response", y_resid)?;
        let code4_lut = &packed_byte_lut().code4;
        let row_step = self.block_rows.max(1);
        let mut out = vec![0.0_f32; self.n_features];
        if let Some(tp) = self.pool.as_ref() {
            tp.install(|| {
                out.par_chunks_mut(row_step)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let row_start = chunk_idx * row_step;
                        for (local_row, dst) in out_chunk.iter_mut().enumerate() {
                            let row_idx = row_start + local_row;
                            if row_idx >= self.n_features {
                                break;
                            }
                            *dst =
                                self.standardized_row_dot_into(row_idx, y_resid, code4_lut) as f32;
                        }
                    });
            });
        } else {
            for row_idx in 0..self.n_features {
                out[row_idx] = self.standardized_row_dot_into(row_idx, y_resid, code4_lut) as f32;
            }
        }
        Ok(out)
    }

    fn diag_xtx_residualized(&self, proj: &CovariateProjection) -> Result<Vec<f32>, String> {
        if proj.is_intercept_only() {
            let code4_lut = &packed_byte_lut().code4;
            let row_step = self.block_rows.max(1);
            let mut out = vec![0.0_f32; self.n_features];
            if let Some(tp) = self.pool.as_ref() {
                tp.install(|| {
                    out.par_chunks_mut(row_step)
                        .enumerate()
                        .for_each(|(chunk_idx, out_chunk)| {
                            let row_start = chunk_idx * row_step;
                            for (local_row, dst) in out_chunk.iter_mut().enumerate() {
                                let row_idx = row_start + local_row;
                                if row_idx >= self.n_features {
                                    break;
                                }
                                let (row_sum, row_ss) =
                                    self.standardized_row_sum_sumsq(row_idx, code4_lut);
                                *dst =
                                    (row_ss - row_sum * row_sum * proj.inv_n).max(1e-12_f64) as f32;
                            }
                        });
                });
            } else {
                for (row_idx, dst) in out.iter_mut().enumerate() {
                    let (row_sum, row_ss) = self.standardized_row_sum_sumsq(row_idx, code4_lut);
                    *dst = (row_ss - row_sum * row_sum * proj.inv_n).max(1e-12_f64) as f32;
                }
            }
            return Ok(out);
        }
        let mut out = vec![0.0_f32; self.n_features];
        let mut block = vec![0.0_f32; self.block_rows * self.n_samples_used];
        let mut row_ids = Vec::<usize>::with_capacity(self.block_rows);
        for row_start in (0..self.n_features).step_by(self.block_rows) {
            let row_end = (row_start + self.block_rows).min(self.n_features);
            let rows = row_end - row_start;
            fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
            self.decode_rows_standardized_into(&row_ids, &mut block[..rows * self.n_samples_used])?;
            let row_block = &block[..rows * self.n_samples_used];
            let out_slice = &mut out[row_start..row_end];
            if proj.is_intercept_only() {
                let mut run = || {
                    out_slice
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(local, dst)| {
                            let row = &row_block
                                [local * self.n_samples_used..(local + 1) * self.n_samples_used];
                            let mut sum = 0.0_f64;
                            let mut ss = 0.0_f64;
                            for &v in row {
                                let vf = v as f64;
                                sum += vf;
                                ss += vf * vf;
                            }
                            *dst = (ss - sum * sum * proj.inv_n).max(1e-12_f64) as f32;
                        });
                };
                if let Some(tp) = self.pool.as_ref() {
                    tp.install(run);
                } else {
                    run();
                }
                continue;
            }
            let mut run = || {
                out_slice.par_iter_mut().enumerate().for_each_init(
                    || (vec![0.0_f64; proj.q], vec![0.0_f64; proj.q]),
                    |(ztrow, coeff), (local, dst)| {
                        let row = &row_block
                            [local * self.n_samples_used..(local + 1) * self.n_samples_used];
                        proj.ztv_f32(row, ztrow);
                        proj.apply_inv(ztrow, coeff);
                        let mut proj_norm = 0.0_f64;
                        for a in 0..proj.q {
                            proj_norm += ztrow[a] * coeff[a];
                        }
                        *dst = (dot_f32_f64(row, row, None) - proj_norm).max(1e-12_f64) as f32;
                    },
                );
            };
            if let Some(tp) = self.pool.as_ref() {
                tp.install(run);
            } else {
                run();
            }
        }
        Ok(out)
    }

    fn scan_kkt_violators_bitwise_orthogonal(
        &self,
        residual: &[f32],
        active_mask: &[bool],
        penalty_weights: &[f32],
        lambda: f32,
        kkt_tol: f32,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        if residual.len() != self.n_samples_used {
            return Err(format!(
                "residual length mismatch: got {}, expected {}",
                residual.len(),
                self.n_samples_used
            ));
        }
        if active_mask.len() != self.n_features {
            return Err(format!(
                "active_mask length mismatch: got {}, expected {}",
                active_mask.len(),
                self.n_features
            ));
        }
        if penalty_weights.len() != self.n_features {
            return Err(format!(
                "penalty_weights length mismatch: got {}, expected {}",
                penalty_weights.len(),
                self.n_features
            ));
        }
        validate_finite_slice_f32("residual", residual)?;
        let code4_lut = &packed_byte_lut().code4;
        let row_step = self.block_rows.max(1);
        if let Some(tp) = self.pool.as_ref() {
            let chunks = tp.install(|| {
                (0..self.n_features)
                    .step_by(row_step)
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|row_start| {
                        let row_end = (row_start + row_step).min(self.n_features);
                        let mut local = Vec::<usize>::new();
                        let mut local_grad = vec![0.0_f32; row_end - row_start];
                        for row_idx in row_start..row_end {
                            let grad =
                                self.standardized_row_dot_into(row_idx, residual, code4_lut) as f32;
                            local_grad[row_idx - row_start] = grad;
                            if active_mask[row_idx] {
                                continue;
                            }
                            let thr = (lambda * penalty_weights[row_idx] + kkt_tol) as f64;
                            if (grad as f64).abs() > thr {
                                local.push(row_idx);
                            }
                        }
                        (row_start, local, local_grad)
                    })
                    .collect::<Vec<(usize, Vec<usize>, Vec<f32>)>>()
            });
            let mut out = Vec::<usize>::new();
            let mut grad = vec![0.0_f32; self.n_features];
            for (row_start, local, local_grad) in chunks {
                out.extend(local);
                grad[row_start..row_start + local_grad.len()].copy_from_slice(&local_grad);
            }
            return Ok((out, grad));
        }
        let mut out = Vec::<usize>::new();
        let mut grad = vec![0.0_f32; self.n_features];
        for row_idx in 0..self.n_features {
            let g = self.standardized_row_dot_into(row_idx, residual, code4_lut) as f32;
            grad[row_idx] = g;
            if active_mask[row_idx] {
                continue;
            }
            let thr = (lambda * penalty_weights[row_idx] + kkt_tol) as f64;
            if (g as f64).abs() > thr {
                out.push(row_idx);
            }
        }
        Ok((out, grad))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn apply_xt_resid_into(
        &self,
        proj: &CovariateProjection,
        sample_vec: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        if sample_vec.len() != self.n_samples_used {
            return Err("apply_xt length mismatch".to_string());
        }
        if out.len() != self.n_features {
            return Err("apply_xt output length mismatch".to_string());
        }
        let mut block = vec![0.0_f32; self.block_rows * self.n_samples_used];
        let mut row_ids = Vec::<usize>::with_capacity(self.block_rows);
        if proj.is_intercept_only() {
            let sample_mean = sample_vec.iter().map(|&v| v as f64).sum::<f64>() * proj.inv_n;
            for row_start in (0..self.n_features).step_by(self.block_rows) {
                let row_end = (row_start + self.block_rows).min(self.n_features);
                let rows = row_end - row_start;
                fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
                self.decode_rows_standardized_into(
                    &row_ids,
                    &mut block[..rows * self.n_samples_used],
                )?;
                let row_block = &block[..rows * self.n_samples_used];
                let out_slice = &mut out[row_start..row_end];
                let mut run = || {
                    out_slice
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(local, dst)| {
                            let row = &row_block
                                [local * self.n_samples_used..(local + 1) * self.n_samples_used];
                            let mut dot_gv = 0.0_f64;
                            let mut row_sum = 0.0_f64;
                            for (&g, &sv) in row.iter().zip(sample_vec.iter()) {
                                let gf = g as f64;
                                dot_gv += gf * sv as f64;
                                row_sum += gf;
                            }
                            *dst = (dot_gv - row_sum * sample_mean) as f32;
                        });
                };
                if let Some(tp) = self.pool.as_ref() {
                    tp.install(run);
                } else {
                    run();
                }
            }
            return Ok(());
        }
        let mut ztv = vec![0.0_f64; proj.q];
        let mut proj_rhs = vec![0.0_f64; proj.q];
        proj.ztv_f32(sample_vec, &mut ztv);
        proj.apply_inv(&ztv, &mut proj_rhs);
        for row_start in (0..self.n_features).step_by(self.block_rows) {
            let row_end = (row_start + self.block_rows).min(self.n_features);
            let rows = row_end - row_start;
            fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
            self.decode_rows_standardized_into(&row_ids, &mut block[..rows * self.n_samples_used])?;
            let row_block = &block[..rows * self.n_samples_used];
            let out_slice = &mut out[row_start..row_end];
            let mut run = || {
                out_slice.par_iter_mut().enumerate().for_each_init(
                    || (vec![0.0_f64; proj.q], vec![0.0_f64; proj.q]),
                    |(ztrow, coeff), (local, dst)| {
                        let row = &row_block
                            [local * self.n_samples_used..(local + 1) * self.n_samples_used];
                        let dot_gv = dot_f32_f64(row, sample_vec, None);
                        proj.ztv_f32(row, ztrow);
                        proj.apply_inv(ztrow, coeff);
                        let mut corr = 0.0_f64;
                        for a in 0..proj.q {
                            corr += ztrow[a] * proj_rhs[a];
                        }
                        *dst = (dot_gv - corr) as f32;
                    },
                );
            };
            if let Some(tp) = self.pool.as_ref() {
                tp.install(run);
            } else {
                run();
            }
        }
        Ok(())
    }

    fn apply_x_resid_into(
        &self,
        proj: &CovariateProjection,
        beta: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        if beta.len() != self.n_features {
            return Err("apply_x length mismatch".to_string());
        }
        if out.len() != self.n_samples_used {
            return Err("apply_x output length mismatch".to_string());
        }
        out.fill(0.0_f32);
        let mut block = vec![0.0_f32; self.block_rows * self.n_samples_used];
        if proj.is_intercept_only() {
            let mut mean_correction = 0.0_f64;
            let mut row_ids = Vec::<usize>::with_capacity(self.block_rows);
            for row_start in (0..self.n_features).step_by(self.block_rows) {
                let row_end = (row_start + self.block_rows).min(self.n_features);
                let rows = row_end - row_start;
                fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
                self.decode_rows_standardized_into(
                    &row_ids,
                    &mut block[..rows * self.n_samples_used],
                )?;
                let row_block = &block[..rows * self.n_samples_used];
                row_major_block_t_mul_vec_accum_f32(
                    row_block,
                    rows,
                    self.n_samples_used,
                    &beta[row_start..row_end],
                    out,
                    self.pool.as_ref(),
                );
                for local in 0..rows {
                    let bj = beta[row_start + local];
                    if bj == 0.0_f32 {
                        continue;
                    }
                    let row =
                        &row_block[local * self.n_samples_used..(local + 1) * self.n_samples_used];
                    let row_sum = row.iter().map(|&v| v as f64).sum::<f64>();
                    mean_correction += row_sum * bj as f64 * proj.inv_n;
                }
            }
            if mean_correction != 0.0_f64 {
                for v in out.iter_mut() {
                    *v -= mean_correction as f32;
                }
            }
            return Ok(());
        }
        let mut ztrow = vec![0.0_f64; proj.q];
        let mut zt_accum = vec![0.0_f64; proj.q];
        let mut proj_coeff = vec![0.0_f64; proj.q];
        for row_start in (0..self.n_features).step_by(self.block_rows) {
            let row_end = (row_start + self.block_rows).min(self.n_features);
            let rows = row_end - row_start;
            let row_ids: Vec<usize> = (row_start..row_end).collect();
            self.decode_rows_standardized_into(&row_ids, &mut block[..rows * self.n_samples_used])?;
            let row_block = &block[..rows * self.n_samples_used];
            row_major_block_t_mul_vec_accum_f32(
                row_block,
                rows,
                self.n_samples_used,
                &beta[row_start..row_end],
                out,
                self.pool.as_ref(),
            );
            for local in 0..rows {
                let k = row_start + local;
                let bj = beta[k];
                if bj == 0.0_f32 {
                    continue;
                }
                let row =
                    &row_block[local * self.n_samples_used..(local + 1) * self.n_samples_used];
                proj.ztv_f32(row, &mut ztrow);
                for a in 0..proj.q {
                    zt_accum[a] += ztrow[a] * (bj as f64);
                }
            }
        }
        proj.apply_inv(&zt_accum, &mut proj_coeff);
        for i in 0..self.n_samples_used {
            let zrow = &proj.z[i * proj.q..(i + 1) * proj.q];
            let mut corr = 0.0_f64;
            for a in 0..proj.q {
                corr += zrow[a] * proj_coeff[a];
            }
            out[i] -= corr as f32;
        }
        Ok(())
    }
}

fn ensure_msgps_std_row_cached(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    row_idx: usize,
    cache: &mut HashMap<usize, CachedMsgpsStdRow>,
) -> Result<(), String> {
    if cache.contains_key(&row_idx) {
        return Ok(());
    }
    let mut row = vec![0.0_f64; design.n_samples()];
    let mut scale = vec![0.0_f64; 1];
    design.decode_rows_resid_standardized_into_f64(&[row_idx], proj, &mut row, Some(&mut scale))?;
    cache.insert(
        row_idx,
        CachedMsgpsStdRow {
            values: row,
            scale: scale[0],
        },
    );
    Ok(())
}

fn ensure_msgps_xtx_cached(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    row_idx: usize,
    std_cache: &mut HashMap<usize, CachedMsgpsStdRow>,
    xtx_cache: &mut MsgpsXtxCache,
    scratch: &mut MsgpsXtxScratch,
) -> Result<(), String> {
    if xtx_cache.contains_key(row_idx) {
        return Ok(());
    }
    ensure_msgps_std_row_cached(design, proj, row_idx, std_cache)?;
    let pivot = &std_cache
        .get(&row_idx)
        .ok_or_else(|| "missing cached standardized row".to_string())?
        .values;
    let n = design.n_samples();
    let p = design.n_features();
    let block_rows = design.block_rows.max(1);
    let spill_slot = xtx_cache.reserve_slot();
    scratch.prepare(block_rows, n);
    let n_blocks = p.div_ceil(block_rows);
    for block_idx in (0..n_blocks).rev() {
        let row_start = block_idx * block_rows;
        let row_end = (row_start + block_rows).min(p);
        let rows = row_end - row_start;
        fill_contiguous_row_ids(&mut scratch.row_ids, row_start, row_end);
        design.decode_rows_resid_standardized_into_f64(
            &scratch.row_ids,
            proj,
            &mut scratch.block[..rows * n],
            None,
        )?;
        row_major_block_mul_vec_f64(
            &scratch.block[..rows * n],
            rows,
            n,
            pivot,
            &mut scratch.tmp[..rows],
            design.pool.as_ref(),
        );
        xtx_cache.write_block(spill_slot, block_idx, &scratch.tmp[..rows])?;
        xtx_cache.insert_hot_block(
            MsgpsXtxBlockKey { row_idx, block_idx },
            scratch.tmp[..rows].to_vec(),
        )?;
    }
    xtx_cache.commit_spilled_row(row_idx, spill_slot);
    Ok(())
}

#[inline]
fn algwas_stage1_active_cap(cfg: &AlgwasConfig, n_samples: usize, n_features: usize) -> usize {
    if cfg.lasso.active_cache_max_rows > 0 {
        cfg.lasso
            .active_cache_max_rows
            .clamp(1usize, n_features.max(1))
    } else {
        let row_bytes = n_samples
            .saturating_mul(std::mem::size_of::<f32>())
            .max(1usize);
        let target_mb = std::env::var("JX_LASSO_ACTIVE_TARGET_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_STAGE1_ACTIVE_TARGET_MB);
        ((target_mb.saturating_mul(1024 * 1024)) / row_bytes)
            .max(DEFAULT_STAGE1_ACTIVE_MIN_ROWS)
            .min(n_features.max(1))
    }
}

#[inline]
fn decode_rows_resid_standardized_into_f32(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    rows: &[usize],
    out: &mut [f32],
) -> Result<(), String> {
    design.decode_rows_standardized_into(rows, out)?;
    residualize_block_rows_in_place(
        out,
        rows.len(),
        design.n_samples(),
        proj,
        design.pool.as_ref(),
    );
    Ok(())
}

fn validate_algwas_stage1_warm_state(
    state: &AlgwasStage1WarmState,
    n_features: usize,
    n_samples: usize,
) -> Result<(), String> {
    validate_active_path_state(state, n_features, n_samples, |aux| {
        if let Some(last_grad) = aux.last_kkt_grad.as_ref() {
            if last_grad.len() != n_features {
                return Err(format!(
                    "last_kkt_grad length mismatch: got {}, expected {}",
                    last_grad.len(),
                    n_features
                ));
            }
        }
        Ok(())
    })
}

fn extend_algwas_stage1_active_set(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    state: &mut AlgwasStage1WarmState,
    rows: &[usize],
    diag_xtx: &[f32],
    penalty_weights: &[f32],
    active_cap: usize,
) -> Result<usize, String> {
    if rows.is_empty() {
        return Ok(0usize);
    }
    let n_features = design.n_features();
    let n_samples = design.n_samples();
    let mut add_rows = Vec::<usize>::with_capacity(rows.len());
    for &j in rows {
        if j < n_features && !state.active_mask[j] {
            add_rows.push(j);
        }
    }
    if add_rows.is_empty() {
        return Ok(0usize);
    }
    if let Some(ref mut dense) = state.active_dense {
        if state.active_rows.len() + add_rows.len() <= active_cap {
            let mut residualized = vec![0.0_f32; add_rows.len() * n_samples];
            decode_rows_resid_standardized_into_f32(design, proj, &add_rows, &mut residualized)?;
            dense.extend_from_slice(&residualized);
        } else {
            state.active_dense = None;
        }
    }
    for &j in &add_rows {
        state.active_mask[j] = true;
        state.active_rows.push(j);
        state.active_diag.push(diag_xtx[j].max(1e-12_f32));
        state.active_weights.push(penalty_weights[j]);
        state.active_beta.push(0.0_f32);
    }
    Ok(add_rows.len())
}

fn initialize_algwas_stage1_warm_state(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y_resid: &[f32],
    xty: &[f32],
    diag_xtx: &[f32],
    penalty_weights: &[f32],
    lambda: f32,
    _cfg: &AlgwasConfig,
    beta_init: Option<&[f32]>,
    initial_working_set: Option<&[usize]>,
    active_cap: usize,
    cd_tol: f32,
    kkt_tol: f32,
) -> Result<AlgwasStage1WarmState, String> {
    let n_samples = design.n_samples();
    let n_features = design.n_features();
    let mut active_rows = Vec::<usize>::new();
    let mut active_mask = vec![false; n_features];
    let mut active_diag = Vec::<f32>::new();
    let mut active_weights = Vec::<f32>::new();
    let mut active_beta = Vec::<f32>::new();

    if let Some(beta0) = beta_init {
        if beta0.len() != n_features {
            return Err("beta_init length mismatch".to_string());
        }
        for j in 0..n_features {
            let bj = beta0[j];
            if bj.abs() > cd_tol {
                active_mask[j] = true;
                active_rows.push(j);
                active_diag.push(diag_xtx[j].max(1e-12_f32));
                active_weights.push(penalty_weights[j]);
                active_beta.push(bj);
            }
        }
    }
    if let Some(rows) = initial_working_set {
        for &j in rows {
            if j >= n_features || active_mask[j] {
                continue;
            }
            active_mask[j] = true;
            active_rows.push(j);
            active_diag.push(diag_xtx[j].max(1e-12_f32));
            active_weights.push(penalty_weights[j]);
            active_beta.push(0.0_f32);
        }
    } else {
        for j in 0..n_features {
            if !active_mask[j] {
                let thr = lambda * penalty_weights[j];
                if xty[j].abs() > thr + kkt_tol {
                    active_mask[j] = true;
                    active_rows.push(j);
                    active_diag.push(diag_xtx[j].max(1e-12_f32));
                    active_weights.push(penalty_weights[j]);
                    active_beta.push(0.0_f32);
                }
            }
        }
    }

    let mut residual = y_resid.to_vec();
    let active_dense = if !active_rows.is_empty() && active_rows.len() <= active_cap {
        let mut dense = vec![0.0_f32; active_rows.len() * n_samples];
        decode_rows_resid_standardized_into_f32(design, proj, &active_rows, &mut dense)?;
        if beta_init.is_some() {
            for (k, &bj) in active_beta.iter().enumerate() {
                if bj == 0.0_f32 {
                    continue;
                }
                let row = &dense[k * n_samples..(k + 1) * n_samples];
                for i in 0..n_samples {
                    residual[i] -= row[i] * bj;
                }
            }
        }
        Some(dense)
    } else {
        if let Some(beta0) = beta_init {
            let mut fitted = vec![0.0_f32; n_samples];
            design.apply_x_resid_into(
                proj,
                &active_mask_to_beta(&active_mask, beta0, n_features),
                &mut fitted,
            )?;
            for i in 0..n_samples {
                residual[i] -= fitted[i];
            }
        }
        None
    };

    Ok(AlgwasStage1WarmState {
        active_rows,
        active_mask,
        active_diag,
        active_weights,
        active_beta,
        active_dense,
        residual,
        aux: AlgwasStage1Aux {
            last_kkt_grad: None,
            last_lambda: None,
        },
    })
}

fn solve_algwas_stage1_active_from_stats(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y_resid: &[f32],
    xty: &[f32],
    diag_xtx: &[f32],
    penalty_weights: &[f32],
    lambda: f32,
    cfg: &AlgwasConfig,
    beta_init: Option<&[f32]>,
    initial_working_set: Option<&[usize]>,
    warm_state: Option<AlgwasStage1WarmState>,
) -> Result<(Vec<f32>, f64, bool, usize, AlgwasStage1WarmState), String> {
    let stage1_timing = algwas_stage1_timing_enabled();
    let solve_t0 = stage1_timing.then(Instant::now);
    let mut init_secs = 0.0_f64;
    let mut dense_restore_secs = 0.0_f64;
    let mut cd_dense_secs = 0.0_f64;
    let mut cd_stream_decode_secs = 0.0_f64;
    let mut cd_stream_proj_secs = 0.0_f64;
    let mut kkt_secs = 0.0_f64;
    let mut expand_secs = 0.0_f64;
    let mut strong_seed_secs = 0.0_f64;
    let mut total_strong_seed = 0usize;
    let n_samples = design.n_samples();
    let n_features = design.n_features();
    if y_resid.len() != n_samples || xty.len() != n_features || diag_xtx.len() != n_features {
        return Err("stage1 shape mismatch".to_string());
    }
    if penalty_weights.len() != n_features {
        return Err("penalty_weights length mismatch".to_string());
    }
    if lambda < 0.0_f32 || !lambda.is_finite() {
        return Err("lambda must be finite and >= 0".to_string());
    }
    let max_outer = if cfg.lasso.max_active_outer_iter > 0 {
        cfg.lasso.max_active_outer_iter
    } else {
        cfg.lasso.max_admm_iter.max(1)
    };
    let max_sweeps = if cfg.lasso.max_active_cd_sweeps > 0 {
        cfg.lasso.max_active_cd_sweeps
    } else {
        cfg.lasso.max_pcg_iter.max(1)
    };
    let cd_tol = if cfg.lasso.active_cd_tol > 0.0_f32 {
        cfg.lasso.active_cd_tol
    } else {
        cfg.lasso.admm_abs_tol.max(1e-6_f32)
    };
    let kkt_tol = if cfg.lasso.active_kkt_tol > 0.0_f32 {
        cfg.lasso.active_kkt_tol
    } else {
        cfg.lasso.admm_abs_tol.max(1e-5_f32)
    };
    let active_cap = algwas_stage1_active_cap(cfg, n_samples, n_features);
    let state_init_t0 = stage1_timing.then(Instant::now);
    let state = if let Some(state) = warm_state {
        validate_algwas_stage1_warm_state(&state, n_features, n_samples)?;
        state
    } else {
        initialize_algwas_stage1_warm_state(
            design,
            proj,
            y_resid,
            xty,
            diag_xtx,
            penalty_weights,
            lambda,
            cfg,
            beta_init,
            initial_working_set,
            active_cap,
            cd_tol,
            kkt_tol,
        )?
    };
    if let Some(t0) = state_init_t0 {
        init_secs += t0.elapsed().as_secs_f64();
    }
    let mut ztrow = vec![0.0_f64; proj.q];
    let mut coeff = vec![0.0_f64; proj.q];
    let mut block = vec![0.0_f32; active_cap.max(1) * n_samples];
    let (state, stats) = run_active_kkt_path(
        state,
        ActivePathSolveConfig {
            lambda,
            active_cap,
            max_outer,
            max_sweeps,
            cd_tol,
            kkt_tol,
        },
        |rows| {
            let restore_t0 = stage1_timing.then(Instant::now);
            let mut dense = vec![0.0_f32; rows.len() * n_samples];
            decode_rows_resid_standardized_into_f32(design, proj, rows, &mut dense)?;
            if let Some(t0) = restore_t0 {
                dense_restore_secs += t0.elapsed().as_secs_f64();
            }
            Ok(dense)
        },
        |state| {
            if let (Some(last_grad), Some(prev_lambda)) =
                (state.aux.last_kkt_grad.as_ref(), state.aux.last_lambda)
            {
                let strong_cap =
                    algwas_stage1_strong_set_cap(n_features, active_cap, state.active_rows.len());
                if strong_cap > 0 {
                    let strong_t0 = stage1_timing.then(Instant::now);
                    let strong_rows = select_stage1_strong_rule_rows(
                        last_grad,
                        &state.active_mask,
                        penalty_weights,
                        prev_lambda,
                        lambda,
                        strong_cap,
                    );
                    let added = extend_algwas_stage1_active_set(
                        design,
                        proj,
                        state,
                        &strong_rows,
                        diag_xtx,
                        penalty_weights,
                        active_cap,
                    )?;
                    total_strong_seed += added;
                    if let Some(t0) = strong_t0 {
                        strong_seed_secs += t0.elapsed().as_secs_f64();
                    }
                    return Ok(added);
                }
            }
            Ok(0usize)
        },
        |state, lambda_now| {
            let cd_t0 = stage1_timing.then(Instant::now);
            let active_len = state.active_rows.len();
            let dense = state
                .active_dense
                .as_ref()
                .ok_or_else(|| "missing active_dense during ALGWAS dense CD sweep".to_string())?;
            let mut max_update = 0.0_f32;
            for k in 0..active_len {
                let row = &dense[k * n_samples..(k + 1) * n_samples];
                let bj_old = state.active_beta[k];
                let corr = dot_f32_f64(row, &state.residual, design.pool.as_ref()) as f32
                    + state.active_diag[k] * bj_old;
                let bj_new = soft_threshold(corr, lambda_now * state.active_weights[k])
                    / state.active_diag[k];
                let delta = bj_new - bj_old;
                if delta != 0.0_f32 {
                    for i in 0..n_samples {
                        state.residual[i] -= row[i] * delta;
                    }
                    state.active_beta[k] = bj_new;
                    max_update = max_update.max(delta.abs());
                }
            }
            if let Some(t0) = cd_t0 {
                cd_dense_secs += t0.elapsed().as_secs_f64();
            }
            Ok(max_update)
        },
        |state, lambda_now, active_cap_now| {
            let active_len = state.active_rows.len();
            let mut max_update = 0.0_f32;
            for chunk_start in (0..active_len).step_by(active_cap_now.max(1)) {
                let chunk_end = (chunk_start + active_cap_now.max(1)).min(active_len);
                let rows = &state.active_rows[chunk_start..chunk_end];
                let rows_len = rows.len();
                let decode_t0 = stage1_timing.then(Instant::now);
                design.decode_rows_standardized_into(rows, &mut block[..rows_len * n_samples])?;
                if let Some(t0) = decode_t0 {
                    cd_stream_decode_secs += t0.elapsed().as_secs_f64();
                }
                for local_k in 0..rows_len {
                    let k = chunk_start + local_k;
                    let row = &mut block[local_k * n_samples..(local_k + 1) * n_samples];
                    let proj_t0 = stage1_timing.then(Instant::now);
                    proj.ztv_f32(row, &mut ztrow);
                    proj.apply_inv(&ztrow, &mut coeff);
                    for i in 0..n_samples {
                        let zrow = &proj.z[i * proj.q..(i + 1) * proj.q];
                        let mut projv = 0.0_f64;
                        for a in 0..proj.q {
                            projv += zrow[a] * coeff[a];
                        }
                        row[i] -= projv as f32;
                    }
                    let bj_old = state.active_beta[k];
                    let corr = dot_f32_f64(row, &state.residual, design.pool.as_ref()) as f32
                        + state.active_diag[k] * bj_old;
                    let bj_new = soft_threshold(corr, lambda_now * state.active_weights[k])
                        / state.active_diag[k];
                    let delta = bj_new - bj_old;
                    if delta != 0.0_f32 {
                        for i in 0..n_samples {
                            state.residual[i] -= row[i] * delta;
                        }
                        state.active_beta[k] = bj_new;
                        max_update = max_update.max(delta.abs());
                    }
                    if let Some(t0) = proj_t0 {
                        cd_stream_proj_secs += t0.elapsed().as_secs_f64();
                    }
                }
            }
            Ok(max_update)
        },
        |state, lambda_now, kkt_tol_now| {
            let kkt_t0 = stage1_timing.then(Instant::now);
            let (violators, last_grad) = design.scan_kkt_violators_bitwise_orthogonal(
                &state.residual,
                &state.active_mask,
                penalty_weights,
                lambda_now,
                kkt_tol_now,
            )?;
            state.aux.last_kkt_grad = Some(last_grad);
            state.aux.last_lambda = Some(lambda_now);
            if let Some(t0) = kkt_t0 {
                kkt_secs += t0.elapsed().as_secs_f64();
            }
            Ok(violators)
        },
        |state, rows, active_cap_now| {
            let expand_t0 = stage1_timing.then(Instant::now);
            let added = extend_algwas_stage1_active_set(
                design,
                proj,
                state,
                rows,
                diag_xtx,
                penalty_weights,
                active_cap_now,
            )?;
            if let Some(t0) = expand_t0 {
                expand_secs += t0.elapsed().as_secs_f64();
            }
            Ok(added)
        },
    )?;

    total_strong_seed = stats.total_seed_rows;
    let total_cd_sweeps = stats.total_sweeps;
    let total_violators = stats.total_violators;
    let peak_active = stats.peak_active;

    let mut beta = vec![0.0_f32; n_features];
    for (k, &j) in state.active_rows.iter().enumerate() {
        beta[j] = state.active_beta[k];
    }
    let rss = dot_f32_f64(&state.residual, &state.residual, design.pool.as_ref());
    if let Some(t0) = solve_t0 {
        eprintln!(
            "ALGWAS stage1 solve timing lambda={:.6e} active_end={} active_peak={} outer={} sweeps={} strong_seed={} violators={} init={:.3}s restore={:.3}s strong={:.3}s cd_dense={:.3}s cd_stream_decode={:.3}s cd_stream_proj={:.3}s kkt={:.3}s expand={:.3}s total={:.3}s",
            lambda as f64,
            state.active_rows.len(),
            peak_active,
            stats.outer_iters,
            total_cd_sweeps,
            total_strong_seed,
            total_violators,
            init_secs,
            dense_restore_secs,
            strong_seed_secs,
            cd_dense_secs,
            cd_stream_decode_secs,
            cd_stream_proj_secs,
            kkt_secs,
            expand_secs,
            t0.elapsed().as_secs_f64(),
        );
    }
    Ok((
        beta,
        rss,
        stats.converged,
        stats.outer_iters.max(stats.sweeps_last),
        state,
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
fn fit_algwas_stage1_active_from_stats(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y_resid: &[f32],
    xty: &[f32],
    diag_xtx: &[f32],
    penalty_weights: &[f32],
    lambda: f32,
    cfg: &AlgwasConfig,
    beta_init: Option<&[f32]>,
    initial_working_set: Option<&[usize]>,
) -> Result<(Vec<f32>, f64, bool, usize), String> {
    let (beta, rss, converged, iters, _warm_state) = solve_algwas_stage1_active_from_stats(
        design,
        proj,
        y_resid,
        xty,
        diag_xtx,
        penalty_weights,
        lambda,
        cfg,
        beta_init,
        initial_working_set,
        None,
    )?;
    Ok((beta, rss, converged, iters))
}

#[inline]
fn active_mask_to_beta(mask: &[bool], beta_init: &[f32], n_features: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n_features];
    for j in 0..n_features {
        if mask[j] {
            out[j] = beta_init[j];
        }
    }
    out
}

#[inline]
fn algwas_pair_r2_from_rows(
    row_i: &[f64],
    mean_i: f64,
    ss_i: f64,
    row_j: &[f64],
    mean_j: f64,
    ss_j: f64,
) -> f64 {
    if !(ss_i > 0.0_f64 && ss_j > 0.0_f64) {
        return 0.0_f64;
    }
    let mut cov = 0.0_f64;
    for idx in 0..row_i.len() {
        cov += (row_i[idx] - mean_i) * (row_j[idx] - mean_j);
    }
    let r2 = (cov * cov) / (ss_i * ss_j);
    if r2.is_finite() {
        r2.clamp(0.0_f64, 1.0_f64)
    } else {
        0.0_f64
    }
}

fn algwas_stage1_ld_prune_selected(
    design: &AlgwasPackedDesign<'_>,
    chrom: &[String],
    pos: &[i64],
    beta: &[f32],
    selected_indices: &[usize],
) -> Result<Vec<usize>, String> {
    let k = selected_indices.len();
    if k <= 1 {
        return Ok(selected_indices.to_vec());
    }
    let r2_threshold = algwas_stage1_ld_prune_r2();
    if !(r2_threshold.is_finite() && r2_threshold > 0.0_f64) {
        return Ok(selected_indices.to_vec());
    }
    let n = design.n_samples();
    let mut decoded = vec![0.0_f64; k * n];
    design.decode_rows_raw_into_f64(selected_indices, &mut decoded)?;
    let mut row_mean = vec![0.0_f64; k];
    let mut row_ss = vec![0.0_f64; k];
    for local in 0..k {
        let row = &decoded[local * n..(local + 1) * n];
        let mean = row.iter().sum::<f64>() / (n as f64);
        let mut ss = 0.0_f64;
        for &v in row {
            let d = v - mean;
            ss += d * d;
        }
        row_mean[local] = mean;
        row_ss[local] = ss;
    }

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        let ga = selected_indices[a];
        let gb = selected_indices[b];
        beta[gb]
            .abs()
            .total_cmp(&beta[ga].abs())
            .then_with(|| chrom[ga].cmp(&chrom[gb]))
            .then_with(|| pos[ga].cmp(&pos[gb]))
            .then_with(|| ga.cmp(&gb))
    });

    let mut kept_local = Vec::<usize>::with_capacity(k);
    'candidate: for &li in &order {
        let global_i = selected_indices[li];
        let row_i = &decoded[li * n..(li + 1) * n];
        for &lj in &kept_local {
            let global_j = selected_indices[lj];
            if chrom[global_i] != chrom[global_j] {
                continue;
            }
            let row_j = &decoded[lj * n..(lj + 1) * n];
            let r2 = algwas_pair_r2_from_rows(
                row_i,
                row_mean[li],
                row_ss[li],
                row_j,
                row_mean[lj],
                row_ss[lj],
            );
            if r2 >= r2_threshold {
                continue 'candidate;
            }
        }
        kept_local.push(li);
    }

    let mut kept = kept_local
        .into_iter()
        .map(|local| selected_indices[local])
        .collect::<Vec<_>>();
    kept.sort_unstable();
    Ok(kept)
}

fn fit_stage1_path_streaming(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y: &[f64],
    cfg: &AlgwasConfig,
    progress_callback: Option<&Py<PyAny>>,
) -> Result<AlgwasStage1Result, String> {
    let stage1_timing = algwas_stage1_timing_enabled();
    let stage1_total_t0 = stage1_timing.then(Instant::now);
    let ebic_gamma = algwas_stage1_ebic_gamma();
    let n = design.n_samples();
    let p = design.n_features();
    let lambda_steps = cfg.lambda_steps.max(1);
    let progress_total = lambda_steps + 3;
    if n == 0 || p == 0 {
        return Ok(AlgwasStage1Result {
            selected_indices: Vec::new(),
            beta: vec![0.0_f32; p],
            lambda_best: 0.0_f32,
            bic_best: f64::INFINITY,
            ebic_best: f64::INFINITY,
            best_step: 0usize,
            path: Vec::new(),
            converged: true,
        });
    }

    if let Some(cb) = progress_callback {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (0usize, progress_total))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }

    let y_t0 = stage1_timing.then(Instant::now);
    let y_resid = proj.residualize_y(y)?;
    let y_secs = y_t0.map(|t0| t0.elapsed().as_secs_f64()).unwrap_or(0.0_f64);
    let diag_t0 = stage1_timing.then(Instant::now);
    let diag_xtx = design.diag_xtx_residualized(proj)?;
    let diag_secs = diag_t0
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    if let Some(cb) = progress_callback {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (1usize, progress_total))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }
    let xty_t0 = stage1_timing.then(Instant::now);
    let xty = design.xty_residualized(&y_resid)?;
    let xty_secs = xty_t0
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    if let Some(cb) = progress_callback {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (2usize, progress_total))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }
    let weight_t0 = stage1_timing.then(Instant::now);
    let adaptive_init =
        compute_msgps_alasso_init(design, proj, &y_resid, &diag_xtx, Some(&xty), cfg)?;
    let mut penalty_weights = adaptive_init.penalty_weights;
    let mut beta_init_full = adaptive_init.beta_init;
    let weight_secs = weight_t0
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or(0.0_f64);
    for w in &mut penalty_weights {
        if !w.is_finite() || *w <= 0.0_f32 {
            *w = 1.0_f32;
        }
    }
    for bj in &mut beta_init_full {
        if !bj.is_finite() {
            *bj = 0.0_f32;
        }
    }
    let initial_working_cap = algwas_stage1_initial_working_set_cap(p);
    let initial_working_set = select_stage1_lm_screen_rows(&xty, &diag_xtx, initial_working_cap);
    let mut beta_init_seed = vec![0.0_f32; p];
    for &j in &initial_working_set {
        if j < p {
            beta_init_seed[j] = beta_init_full[j];
        }
    }
    if std::env::var_os("JX_ALGWAS_DEBUG").is_some() {
        eprintln!(
            "ALGWAS stage1 LM screen: selected {} / {} candidates (cap={})",
            initial_working_set.len(),
            p,
            initial_working_cap
        );
    }
    if let Some(cb) = progress_callback {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (3usize, progress_total))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }

    let lambda_min_ratio = cfg.lambda_min_ratio.clamp(1e-6_f32, 1.0_f32);
    let lambda_max = weighted_lambda_max(&xty, &penalty_weights);
    if !lambda_max.is_finite() || lambda_max <= 0.0_f32 {
        if let Some(cb) = progress_callback {
            Python::attach(|py| -> PyResult<()> {
                py.check_signals()?;
                cb.call1(py, (progress_total, progress_total))?;
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        }
        let rss0 = dot_f32_f64(&y_resid, &y_resid, design.pool.as_ref());
        let path = vec![AlgwasStage1PathPoint {
            lambda: 0.0_f32,
            bic: (n as f64) * (rss0.max(1e-12_f64) / n as f64).ln(),
            ebic: (n as f64) * (rss0.max(1e-12_f64) / n as f64).ln(),
            rss: rss0,
            nnz: 0,
            df: 0.0_f64,
            converged: true,
        }];
        return Ok(AlgwasStage1Result {
            selected_indices: Vec::new(),
            beta: vec![0.0_f32; p],
            lambda_best: 0.0_f32,
            bic_best: path[0].bic,
            ebic_best: path[0].ebic,
            best_step: 0usize,
            path,
            converged: true,
        });
    }

    let lambda_min = (lambda_max * lambda_min_ratio).max(lambda_max * 1e-6_f32);
    let lambda_span = if lambda_max > 0.0_f32 && lambda_min > 0.0_f32 {
        (lambda_min / lambda_max).max(1e-6_f32)
    } else {
        1.0_f32
    };

    let mut path = Vec::<AlgwasStage1PathPoint>::with_capacity(lambda_steps + 1);
    let rss0 = dot_f32_f64(&y_resid, &y_resid, design.pool.as_ref());
    let bic0 = algwas_bic_from_rss_df(n, rss0, 0.0_f64);
    let ebic0 = algwas_ebic_from_bic(bic0, p, 0usize, ebic_gamma);
    path.push(AlgwasStage1PathPoint {
        lambda: 0.0_f32,
        bic: bic0,
        ebic: ebic0,
        rss: rss0,
        nnz: 0,
        df: 0.0_f64,
        converged: true,
    });

    let selection_criterion = algwas_stage1_select_criterion();
    let early_stop_cap = algwas_stage1_site_cap(n);
    let mut warm_state: Option<AlgwasStage1WarmState> = None;
    let mut best_idx = 0usize;
    let mut best_selected_score = algwas_stage1_path_score(&path[0], selection_criterion);
    let mut best_beta_std = vec![0.0_f32; p];
    let mut best_converged = true;
    let mut path_solve_secs = 0.0_f64;
    let mut last_progress_done = 3usize;

    for step in 0..lambda_steps {
        let t = if lambda_steps <= 1 {
            0.0_f32
        } else {
            step as f32 / (lambda_steps - 1) as f32
        };
        let lambda = lambda_max * lambda_span.powf(t);
        let use_seed_rows = warm_state.is_none();
        let solve_t0 = stage1_timing.then(Instant::now);
        let (beta_std, rss, converged, _iters, next_warm_state) =
            solve_algwas_stage1_active_from_stats(
                design,
                proj,
                &y_resid,
                &xty,
                &diag_xtx,
                &penalty_weights,
                lambda,
                cfg,
                if use_seed_rows {
                    Some(&beta_init_seed)
                } else {
                    None
                },
                if use_seed_rows {
                    Some(&initial_working_set)
                } else {
                    None
                },
                warm_state.take(),
            )?;
        if let Some(t0) = solve_t0 {
            path_solve_secs += t0.elapsed().as_secs_f64();
        }
        let nnz = beta_std.iter().filter(|&&v| v != 0.0_f32).count();
        let df = nnz as f64;
        let bic = algwas_bic_from_rss_df(n, rss, df);
        let ebic = algwas_ebic_from_bic(bic, p, nnz, ebic_gamma);
        path.push(AlgwasStage1PathPoint {
            lambda,
            bic,
            ebic,
            rss,
            nnz,
            df,
            converged,
        });
        let point_ref = path
            .last()
            .ok_or_else(|| "ALGWAS stage1 path unexpectedly empty after push".to_string())?;
        let selected_score = algwas_stage1_path_score(point_ref, selection_criterion);
        if selected_score < best_selected_score {
            best_selected_score = selected_score;
            best_idx = step + 1;
            best_beta_std.clone_from(&beta_std);
            best_converged = converged;
        }
        warm_state = Some(next_warm_state);
        if let Some(cb) = progress_callback {
            let done = step + 4;
            last_progress_done = done;
            Python::attach(|py| -> PyResult<()> {
                py.check_signals()?;
                cb.call1(py, (done, progress_total))?;
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        }
        if nnz > early_stop_cap {
            break;
        }
    }

    let best_bic_idx = best_stage1_path_index(&path, AlgwasStage1SelectCriterion::Bic);
    let best_ebic_idx = best_stage1_path_index(&path, AlgwasStage1SelectCriterion::Ebic);
    let best_point = path
        .get(best_idx)
        .ok_or_else(|| "ALGWAS stage1 path unexpectedly empty".to_string())?;
    let mut beta_best = vec![0.0_f32; p];
    let mut selected_indices = Vec::<usize>::new();
    for j in 0..p {
        let bj = best_beta_std[j] * design.row_inv_sd[j];
        beta_best[j] = bj;
        if bj != 0.0_f32 {
            selected_indices.push(j);
        }
    }
    if let Some(qb) = cfg.qtn_bound {
        if selected_indices.len() > qb {
            selected_indices.sort_by(|&a, &b| beta_best[b].abs().total_cmp(&beta_best[a].abs()));
            selected_indices.truncate(qb);
            selected_indices.sort_unstable();
        }
    }

    if let Some(t0) = stage1_total_t0 {
        let total_secs = t0.elapsed().as_secs_f64().max(1e-12_f64);
        eprintln!(
            "ALGWAS stage1 path timing: y_resid={:.3}s ({:.1}%) diag_xtx={:.3}s ({:.1}%) xty={:.3}s ({:.1}%) weights={:.3}s ({:.1}%) lambda_path={:.3}s ({:.1}%) total={:.3}s",
            y_secs,
            100.0_f64 * y_secs / total_secs,
            diag_secs,
            100.0_f64 * diag_secs / total_secs,
            xty_secs,
            100.0_f64 * xty_secs / total_secs,
            weight_secs,
            100.0_f64 * weight_secs / total_secs,
            path_solve_secs,
            100.0_f64 * path_solve_secs / total_secs,
            total_secs,
        );
    }

    if let Some(cb) = progress_callback {
        let done = last_progress_done.max(progress_total);
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (done, progress_total))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }

    Ok(AlgwasStage1Result {
        selected_indices,
        beta: beta_best,
        lambda_best: best_point.lambda,
        bic_best: path[best_bic_idx].bic,
        ebic_best: path[best_ebic_idx].ebic,
        best_step: best_idx,
        path,
        converged: best_converged,
    })
}

fn reconstruct_msgps_beta(
    n_features: usize,
    betahat_index: &[i32],
    step: usize,
    delta_t: f64,
) -> Vec<f64> {
    let mut beta = vec![0.0_f64; n_features];
    let stop = step.min(betahat_index.len().saturating_sub(1));
    for &idx_signed in betahat_index.iter().take(stop + 1).skip(1) {
        if idx_signed == 0 {
            continue;
        }
        let sign = if idx_signed > 0 { 1.0_f64 } else { -1.0_f64 };
        let idx = (idx_signed.unsigned_abs() as usize).saturating_sub(1);
        if idx < n_features {
            beta[idx] += sign * delta_t;
        }
    }
    beta
}

#[inline]
fn msgps_near_zero_tol(delta_t: f64) -> f64 {
    delta_t.abs().max(1.0_f64) * 1e-12_f64
}

#[inline]
fn msgps_beta_sign_with_memory(beta: f64, remembered_sign: i8, zero_tol: f64) -> i8 {
    if beta > zero_tol {
        1
    } else if beta < -zero_tol {
        -1
    } else {
        remembered_sign
    }
}

fn compute_msgps_df_modified(
    r_mat: &DMatrix<f64>,
    betahat_index_adj: &[usize],
    increment_vec: &[f64],
    step_adj: usize,
) -> Vec<f64> {
    let q_n = r_mat.nrows();
    let mut df = vec![0.0_f64; step_adj + 1];
    if q_n == 0 || step_adj == 0 {
        return df;
    }
    let mut bt0 = DMatrix::<f64>::zeros(q_n, q_n);
    let mut q_n_adj = 0usize;
    for step in 1..=step_adj {
        let jstar = betahat_index_adj[step].saturating_sub(1);
        if q_n_adj == jstar && q_n_adj < q_n {
            let new_idx = q_n_adj;
            q_n_adj += 1;
            for i in 0..q_n_adj {
                bt0[(new_idx, i)] = 0.0_f64;
                bt0[(i, new_idx)] = 0.0_f64;
            }
            bt0[(new_idx, new_idx)] = 1.0_f64;
        }
        if q_n_adj == 0 {
            continue;
        }
        let inc = increment_vec[step];
        let mut r_j = vec![0.0_f64; q_n_adj];
        for i in 0..q_n_adj {
            r_j[i] = r_mat[(i, jstar)];
        }
        let mut r_cov = vec![0.0_f64; q_n_adj];
        for i in 0..q_n_adj {
            let mut acc = 0.0_f64;
            for k in 0..q_n_adj {
                acc += bt0[(k, i)] * r_j[k];
            }
            r_cov[i] = acc;
        }
        for i in 0..q_n_adj {
            for j in 0..q_n_adj {
                bt0[(i, j)] -= inc * r_j[i] * r_cov[j];
            }
        }
        let mut tr = 0.0_f64;
        for i in 0..q_n_adj {
            tr += bt0[(i, i)];
        }
        df[step] = q_n_adj as f64 - tr;
    }
    df
}

struct ExactPackedStage1Prelude {
    weights: Vec<f64>,
    y_centered: Vec<f64>,
    xty0: Vec<f64>,
    beta_ols_abs_sum: f64,
}

#[derive(Clone, Copy)]
enum ExactPackedOperatorMode {
    ResidualizedBase,
    ResidualizedStandardized,
}

fn decode_exact_packed_rows_into(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    mode: ExactPackedOperatorMode,
    rows: &[usize],
    out: &mut [f64],
) -> Result<(), String> {
    match mode {
        ExactPackedOperatorMode::ResidualizedBase => {
            design.decode_rows_resid_base_into_f64(rows, proj, out)
        }
        ExactPackedOperatorMode::ResidualizedStandardized => {
            design.decode_rows_resid_standardized_into_f64(rows, proj, out, None)
        }
    }
}

fn apply_packed_sample_space_operator_into(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    mode: ExactPackedOperatorMode,
    ridge_lambda: f64,
    sample_vec: &[f64],
    row_ids: &mut Vec<usize>,
    block: &mut [f64],
    feature_scratch: &mut [f64],
    out: &mut [f64],
) -> Result<(), String> {
    let n = design.n_samples();
    let p = design.n_features();
    if sample_vec.len() != n || out.len() != n {
        return Err("sample-space operator length mismatch".to_string());
    }
    if feature_scratch.len() != p {
        return Err("sample-space operator feature scratch length mismatch".to_string());
    }
    feature_scratch.fill(0.0_f64);
    out.fill(0.0_f64);
    for row_start in (0..p).step_by(design.block_rows) {
        let row_end = (row_start + design.block_rows).min(p);
        let rows = row_end - row_start;
        fill_contiguous_row_ids(row_ids, row_start, row_end);
        decode_exact_packed_rows_into(design, proj, mode, row_ids, &mut block[..rows * n])?;
        row_major_block_mul_vec_f64(
            &block[..rows * n],
            rows,
            n,
            sample_vec,
            &mut feature_scratch[row_start..row_end],
            design.pool.as_ref(),
        );
    }
    for row_start in (0..p).step_by(design.block_rows) {
        let row_end = (row_start + design.block_rows).min(p);
        let rows = row_end - row_start;
        fill_contiguous_row_ids(row_ids, row_start, row_end);
        decode_exact_packed_rows_into(design, proj, mode, row_ids, &mut block[..rows * n])?;
        row_major_block_t_mul_vec_accum_f64(
            &block[..rows * n],
            rows,
            n,
            &feature_scratch[row_start..row_end],
            out,
            design.pool.as_ref(),
        );
    }
    if ridge_lambda != 0.0_f64 {
        for i in 0..n {
            out[i] += ridge_lambda * sample_vec[i];
        }
    }
    Ok(())
}

struct PackedSampleSpaceOperator<'ctx, 'data> {
    design: &'ctx AlgwasPackedDesign<'data>,
    proj: &'ctx CovariateProjection,
    mode: ExactPackedOperatorMode,
    ridge_lambda: f64,
    row_ids: &'ctx mut Vec<usize>,
    block: &'ctx mut [f64],
    feature_scratch: &'ctx mut [f64],
}

impl PcgOperator<f64> for PackedSampleSpaceOperator<'_, '_> {
    #[inline]
    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
        apply_packed_sample_space_operator_into(
            self.design,
            self.proj,
            self.mode,
            self.ridge_lambda,
            input,
            self.row_ids,
            self.block,
            self.feature_scratch,
            output,
        )
    }
}

fn solve_packed_dual_ridge_alpha_pcg(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    mode: ExactPackedOperatorMode,
    y: &[f64],
    ridge_lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, String> {
    let n = design.n_samples();
    if y.len() != n {
        return Err("packed dual ridge RHS length mismatch".to_string());
    }
    let mut feature_scratch = vec![0.0_f64; design.n_features()];
    let mut block = vec![0.0_f64; design.block_rows.max(1) * n];
    let mut row_ids = Vec::<usize>::with_capacity(design.block_rows.max(1));
    let max_iter_use = max_iter.max(1).max(256).min(n.max(1));
    let tol_use = if tol.is_finite() && tol > 0.0_f64 {
        tol.min(DEFAULT_STAGE1_ALASSO_PCG_TOL)
    } else {
        DEFAULT_STAGE1_ALASSO_PCG_TOL
    };
    let tiny_use = ridge_lambda.abs().max(1e-12_f64);
    let apply_a = PackedSampleSpaceOperator {
        design,
        proj,
        mode,
        ridge_lambda,
        row_ids: &mut row_ids,
        block: &mut block,
        feature_scratch: &mut feature_scratch,
    };
    let pcg = pcg_solve(
        y,
        None,
        max_iter_use,
        tol_use,
        tiny_use,
        apply_a,
        IdentityPreconditioner,
        |_iter_done, _iter_total, _rel_res| Ok(()),
    )?;
    if pcg.converged {
        return Ok(pcg.x);
    }
    Err(format!(
        "exact packed PCG did not converge: iters={} rel_res={:.3e}",
        pcg.iters, pcg.rel_res
    ))
}

fn compute_exact_packed_stage1_prelude_pcg(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y: &[f64],
    cfg: &AlgwasConfig,
) -> Result<ExactPackedStage1Prelude, String> {
    let n = design.n_samples();
    let p = design.n_features();
    let y_resid_f32 = proj.residualize_y(y)?;
    let has_cov = proj.q > 1;
    let y_base: Vec<f64> = if has_cov {
        y_resid_f32.iter().map(|&v| v as f64).collect()
    } else {
        y.to_vec()
    };
    let ridge_lambda = DEFAULT_STAGE1_ALASSO_RIDGE_LAMBDA as f64;
    let pcg_max_iter = cfg
        .lasso
        .max_pcg_iter
        .max(DEFAULT_STAGE1_ALASSO_PCG_MAX_ITERS);
    let pcg_tol = cfg.lasso.pcg_tol;
    let alpha_base = solve_packed_dual_ridge_alpha_pcg(
        design,
        proj,
        ExactPackedOperatorMode::ResidualizedBase,
        &y_base,
        ridge_lambda,
        pcg_max_iter,
        pcg_tol,
    )?;

    let block_rows = design.block_rows.max(1);
    let mut block = vec![0.0_f64; block_rows * n];
    let mut tmp = vec![0.0_f64; block_rows];
    let mut row_ids = Vec::<usize>::with_capacity(block_rows);
    let mut weights = vec![0.0_f64; p];
    for row_start in (0..p).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(p);
        let rows = row_end - row_start;
        fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
        design.decode_rows_resid_base_into_f64(&row_ids, proj, &mut block[..rows * n])?;
        row_major_block_mul_vec_f64(
            &block[..rows * n],
            rows,
            n,
            &alpha_base,
            &mut tmp[..rows],
            design.pool.as_ref(),
        );
        for local in 0..rows {
            let beta = tmp[local].abs();
            weights[row_start + local] = if beta == 0.0_f64 {
                f64::INFINITY
            } else {
                beta.powf(-(DEFAULT_STAGE1_ALASSO_GAMMA as f64))
            };
        }
    }

    let y_mean = y_base.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y_base.iter().map(|v| v - y_mean).collect();
    let alpha_std = solve_packed_dual_ridge_alpha_pcg(
        design,
        proj,
        ExactPackedOperatorMode::ResidualizedStandardized,
        &y_centered,
        ridge_lambda,
        pcg_max_iter,
        pcg_tol,
    )?;

    let mut xty0 = vec![0.0_f64; p];
    let mut beta_ols_abs_sum = 0.0_f64;
    for row_start in (0..p).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(p);
        let rows = row_end - row_start;
        fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
        design.decode_rows_resid_standardized_into_f64(
            &row_ids,
            proj,
            &mut block[..rows * n],
            None,
        )?;
        row_major_block_mul_vec_f64(
            &block[..rows * n],
            rows,
            n,
            &y_centered,
            &mut tmp[..rows],
            design.pool.as_ref(),
        );
        xty0[row_start..row_end].copy_from_slice(&tmp[..rows]);
        row_major_block_mul_vec_f64(
            &block[..rows * n],
            rows,
            n,
            &alpha_std,
            &mut tmp[..rows],
            design.pool.as_ref(),
        );
        beta_ols_abs_sum += tmp[..rows].iter().map(|v| v.abs()).sum::<f64>();
    }

    Ok(ExactPackedStage1Prelude {
        weights,
        y_centered,
        xty0,
        beta_ols_abs_sum,
    })
}

fn compute_exact_packed_stage1_prelude_dense(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y: &[f64],
) -> Result<ExactPackedStage1Prelude, String> {
    let n = design.n_samples();
    let p = design.n_features();
    let _blas_guard = OpenBlasThreadGuard::enter(current_pool_threads(design.pool.as_ref()));
    let y_resid_f32 = proj.residualize_y(y)?;
    let has_cov = proj.q > 1;
    let y_base: Vec<f64> = if has_cov {
        y_resid_f32.iter().map(|&v| v as f64).collect()
    } else {
        y.to_vec()
    };
    let mut gram_base = vec![0.0_f64; n * n];
    let mut gram_std = vec![0.0_f64; n * n];
    let block_rows = design.block_rows.max(1);
    let mut raw_block = vec![0.0_f64; block_rows * n];
    let mut tmp = vec![0.0_f64; block_rows];
    let mut row_ids = Vec::<usize>::with_capacity(block_rows);

    for row_start in (0..p).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(p);
        let rows = row_end - row_start;
        fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
        design.decode_rows_raw_into_f64(&row_ids, &mut raw_block[..rows * n])?;
        if has_cov {
            residualize_block_rows_in_place_f64(
                &mut raw_block[..rows * n],
                rows,
                n,
                proj,
                design.pool.as_ref(),
            );
        }
        gram_add_aat_f64(
            &mut gram_base,
            &raw_block[..rows * n],
            rows,
            n,
            row_start == 0,
        );
    }
    let ridge_lambda = DEFAULT_STAGE1_ALASSO_RIDGE_LAMBDA as f64;
    for i in 0..n {
        gram_base[i * n + i] += ridge_lambda;
    }
    let alpha_base = primal_beta_from_gram(
        &DMatrix::<f64>::from_row_slice(n, n, &gram_base),
        &DVector::<f64>::from_column_slice(&y_base),
    )?;

    let mut weights = vec![0.0_f64; p];
    for row_start in (0..p).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(p);
        let rows = row_end - row_start;
        fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
        design.decode_rows_raw_into_f64(&row_ids, &mut raw_block[..rows * n])?;
        if has_cov {
            residualize_block_rows_in_place_f64(
                &mut raw_block[..rows * n],
                rows,
                n,
                proj,
                design.pool.as_ref(),
            );
        }
        row_major_block_mul_vec_f64(
            &raw_block[..rows * n],
            rows,
            n,
            &alpha_base,
            &mut tmp[..rows],
            design.pool.as_ref(),
        );
        for local in 0..rows {
            let beta = tmp[local].abs();
            weights[row_start + local] = if beta == 0.0_f64 {
                f64::INFINITY
            } else {
                beta.powf(-(DEFAULT_STAGE1_ALASSO_GAMMA as f64))
            };
        }
    }

    let y_mean = y_base.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y_base.iter().map(|v| v - y_mean).collect();
    for i in 0..n {
        gram_std[i * n + i] = 0.0_f64;
    }
    for row_start in (0..p).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(p);
        let rows = row_end - row_start;
        fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
        design.decode_rows_resid_standardized_into_f64(
            &row_ids,
            proj,
            &mut raw_block[..rows * n],
            None,
        )?;
        gram_add_aat_f64(
            &mut gram_std,
            &raw_block[..rows * n],
            rows,
            n,
            row_start == 0,
        );
    }
    for i in 0..n {
        gram_std[i * n + i] += ridge_lambda;
    }
    let alpha_std = primal_beta_from_gram(
        &DMatrix::<f64>::from_row_slice(n, n, &gram_std),
        &DVector::<f64>::from_column_slice(&y_centered),
    )?;

    let mut xty0 = vec![0.0_f64; p];
    let mut beta_ols_abs_sum = 0.0_f64;
    for row_start in (0..p).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(p);
        let rows = row_end - row_start;
        fill_contiguous_row_ids(&mut row_ids, row_start, row_end);
        design.decode_rows_resid_standardized_into_f64(
            &row_ids,
            proj,
            &mut raw_block[..rows * n],
            None,
        )?;
        row_major_block_mul_vec_f64(
            &raw_block[..rows * n],
            rows,
            n,
            &y_centered,
            &mut tmp[..rows],
            design.pool.as_ref(),
        );
        xty0[row_start..row_end].copy_from_slice(&tmp[..rows]);
        row_major_block_mul_vec_f64(
            &raw_block[..rows * n],
            rows,
            n,
            &alpha_std,
            &mut tmp[..rows],
            design.pool.as_ref(),
        );
        beta_ols_abs_sum += tmp[..rows].iter().map(|v| v.abs()).sum::<f64>();
    }

    Ok(ExactPackedStage1Prelude {
        weights,
        y_centered,
        xty0,
        beta_ols_abs_sum,
    })
}

fn fit_stage1_path_exact_packed(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y: &[f64],
    cfg: &AlgwasConfig,
    progress_callback: Option<&Py<PyAny>>,
) -> Result<AlgwasStage1Result, String> {
    let ebic_gamma = algwas_stage1_ebic_gamma();
    let n = design.n_samples();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Ok(AlgwasStage1Result {
            selected_indices: Vec::new(),
            beta: vec![0.0_f32; p],
            lambda_best: 0.0_f32,
            bic_best: f64::INFINITY,
            ebic_best: f64::INFINITY,
            best_step: 0usize,
            path: Vec::new(),
            converged: true,
        });
    }
    if n > p {
        return fit_stage1_path_dense_msgps(design, proj, y, cfg, progress_callback);
    }

    let prelude = if algwas_exact_pcg_enabled(n) {
        match compute_exact_packed_stage1_prelude_pcg(design, proj, y, cfg) {
            Ok(stats) => stats,
            Err(err) => {
                if env_var_truthy("JX_ALGWAS_STAGE1_EXACT_PCG") {
                    return Err(err);
                }
                if std::env::var_os("JX_ALGWAS_DEBUG").is_some() {
                    eprintln!("ALGWAS exact packed PCG fallback to dense prelude: {err}");
                }
                compute_exact_packed_stage1_prelude_dense(design, proj, y)?
            }
        }
    } else {
        compute_exact_packed_stage1_prelude_dense(design, proj, y)?
    };
    let ExactPackedStage1Prelude {
        weights,
        y_centered,
        xty0,
        beta_ols_abs_sum,
    } = prelude;

    let step_ref = DEFAULT_STAGE1_MSGPS_STEP as f64;
    let delta_t = beta_ols_abs_sum / step_ref.max(1.0_f64);
    if !delta_t.is_finite() || delta_t <= 0.0_f64 {
        if let Some(cb) = progress_callback {
            Python::attach(|py| -> PyResult<()> {
                py.check_signals()?;
                cb.call1(py, (1usize, 1usize))?;
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        }
        return Ok(AlgwasStage1Result {
            selected_indices: Vec::new(),
            beta: vec![0.0_f32; p],
            lambda_best: 0.0_f32,
            bic_best: f64::INFINITY,
            ebic_best: f64::INFINITY,
            best_step: 0usize,
            path: vec![AlgwasStage1PathPoint {
                lambda: 0.0_f32,
                bic: f64::INFINITY,
                ebic: f64::INFINITY,
                rss: y_centered.iter().map(|v| v * v).sum(),
                nnz: 0,
                df: 0.0_f64,
                converged: true,
            }],
            converged: true,
        });
    }

    let alpha_n2 = 2.0_f64 / n as f64;
    let threshold = alpha_n2 * delta_t;
    let step_max = DEFAULT_STAGE1_MSGPS_STEP_MAX;
    let pmax = cfg
        .qtn_bound
        .unwrap_or(DEFAULT_STAGE1_MSGPS_PMAX)
        .min(p.max(1));
    let early_stop_cap = algwas_stage1_site_cap(n);

    let mut g: Vec<f64> = xty0.iter().map(|v| alpha_n2 * *v).collect();
    let mut residual = DVector::<f64>::from_column_slice(&y_centered);
    let mut beta_std = vec![0.0_f64; p];
    let mut beta_sign_state = vec![0i8; p];
    let mut selected_order = vec![0usize; p];
    let mut selected_cols = Vec::<usize>::new();
    let mut betahat_index = vec![0i32; step_max + 1];
    let mut betahat_index_adj = vec![0usize; step_max + 1];
    let mut tuning = vec![0.0_f64; step_max + 1];
    let mut tuning_stand = vec![0.0_f64; step_max + 1];
    let mut rss = vec![0.0_f64; step_max + 1];
    let mut increment_vec = vec![0.0_f64; step_max + 1];
    let mut sum_lambda = vec![0.0_f64; step_max + 1];
    rss[0] = residual.dot(&residual);

    let mut last_jstar: Option<usize> = None;
    let mut last_sign = 1.0_f64;
    let mut step_adj_raw = step_max + 1;
    let mut include_last_step = false;
    let mut converged = false;
    let cache_cap = pmax.saturating_add(1).min(p.max(1));
    let block_rows = design.block_rows.max(1);
    let mut std_cache = HashMap::<usize, CachedMsgpsStdRow>::with_capacity(cache_cap);
    let mut xtx_cache = MsgpsXtxCache::new(p, cache_cap, block_rows)?;
    let mut xtx_scratch = MsgpsXtxScratch::default();
    let beta_zero_tol = msgps_near_zero_tol(delta_t);

    if let Some(cb) = progress_callback {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (0usize, 0usize))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }

    for step in 1..=step_max {
        if let Some(prev_jstar) = last_jstar {
            ensure_msgps_xtx_cached(
                design,
                proj,
                prev_jstar,
                &mut std_cache,
                &mut xtx_cache,
                &mut xtx_scratch,
            )?;
            let scale = -threshold * last_sign;
            xtx_cache.for_each_block(prev_jstar, |block_start, xtxj| {
                let block_end = block_start + xtxj.len();
                daxpy_inplace_f64(scale, xtxj, &mut g[block_start..block_end]);
                Ok(())
            })?;
        }

        let mut max_idx = 0usize;
        let mut max_val = 0.0_f64;
        let mut below_thr = 0usize;
        let mut sum_abs_lambda = 0.0_f64;
        for j in 0..p {
            let absg = g[j].abs();
            let lambda_raw = g[j] / weights[j];
            let mut score = lambda_raw.abs();
            sum_abs_lambda += lambda_raw.abs();
            let beta_sign =
                msgps_beta_sign_with_memory(beta_std[j], beta_sign_state[j], beta_zero_tol);
            if beta_sign != 0
                && lambda_raw.signum() != 0.0_f64
                && (lambda_raw.signum() as i8) != beta_sign
                && absg > threshold
            {
                score += 1e12_f64;
            }
            if absg < threshold {
                score = 0.0_f64;
                below_thr += 1;
            }
            if score > max_val {
                max_val = score;
                max_idx = j;
            }
        }

        let lambda_j = g[max_idx] / weights[max_idx];
        let sign = if lambda_j > 0.0_f64 {
            1.0_f64
        } else {
            -1.0_f64
        };
        if selected_order[max_idx] == 0 {
            selected_cols.push(max_idx);
            selected_order[max_idx] = selected_cols.len();
        }
        betahat_index_adj[step] = selected_order[max_idx];

        let beta_prev = beta_std[max_idx];
        beta_std[max_idx] += delta_t * sign;
        if beta_std[max_idx] > beta_zero_tol {
            beta_sign_state[max_idx] = 1;
        } else if beta_std[max_idx] < -beta_zero_tol {
            beta_sign_state[max_idx] = -1;
        }
        betahat_index[step] = ((max_idx + 1) as i32) * if sign > 0.0_f64 { 1 } else { -1 };
        increment_vec[step] = if g[max_idx].abs() > 0.0_f64 {
            threshold / g[max_idx].abs()
        } else {
            0.0_f64
        };

        ensure_msgps_std_row_cached(design, proj, max_idx, &mut std_cache)?;
        let col = &std_cache.get(&max_idx).unwrap().values;
        daxpy_inplace_f64(-delta_t * sign, col, residual.as_mut_slice());
        rss[step] = residual.dot(&residual);
        let delta_pen = weights[max_idx] * (beta_std[max_idx].abs() - beta_prev.abs());
        tuning[step] = tuning[step - 1] + delta_pen;
        tuning_stand[step] =
            tuning_stand[step - 1] + std_cache.get(&max_idx).unwrap().scale * delta_pen;
        sum_lambda[step] = sum_abs_lambda;

        last_jstar = Some(max_idx);
        last_sign = sign;
        if selected_cols.len() > early_stop_cap {
            step_adj_raw = step.saturating_add(1);
            include_last_step = true;
            break;
        }
        if below_thr == p || max_val <= 0.0_f64 {
            converged = true;
            step_adj_raw = step;
            break;
        }
        if selected_cols.len() == pmax + 1 {
            step_adj_raw = step;
            break;
        }
        if step > 20 {
            let cur = sum_lambda[step];
            if (step - 20..step).any(|k| (sum_lambda[k] - cur).abs() == 0.0_f64) {
                converged = true;
                step_adj_raw = step;
                break;
            }
        }
        if let Some(cb) = progress_callback {
            if step <= 1024 || step % 64 == 0 || step == step_max {
                Python::attach(|py| -> PyResult<()> {
                    py.check_signals()?;
                    cb.call1(py, (step, 0usize))?;
                    Ok(())
                })
                .map_err(|e| e.to_string())?;
            }
        }
    }

    let step_adj = if include_last_step {
        step_adj_raw.min(step_max)
    } else {
        step_adj_raw.saturating_sub(1).min(step_max)
    };
    let q_n = selected_cols.len();
    let df = if q_n > 0 {
        let mut x_sel = Vec::<f64>::with_capacity(n * q_n);
        for &col_idx in selected_cols.iter().take(q_n) {
            ensure_msgps_std_row_cached(design, proj, col_idx, &mut std_cache)?;
            x_sel.extend_from_slice(&std_cache.get(&col_idx).unwrap().values);
        }
        let x_sel = DMatrix::<f64>::from_vec(n, q_n, x_sel);
        let r = x_sel.qr().r();
        compute_msgps_df_modified(&r, &betahat_index_adj, &increment_vec, step_adj)
    } else {
        vec![0.0_f64; step_adj + 1]
    };

    let tau2 = if n <= p {
        let mean = y_centered.iter().sum::<f64>() / n as f64;
        let mut ss = 0.0_f64;
        for &v in &y_centered {
            let d = v - mean;
            ss += d * d;
        }
        ss / (n as f64 - 1.0_f64).max(1.0_f64)
    } else {
        let resid_ols = DVector::<f64>::from_column_slice(&y_centered)
            - (DMatrix::<f64>::from_vec(n, p, {
                let mut x_std = Vec::<f64>::with_capacity(n * p);
                for j in 0..p {
                    ensure_msgps_std_row_cached(design, proj, j, &mut std_cache)?;
                    x_std.extend_from_slice(&std_cache.get(&j).unwrap().values);
                }
                x_std
            }) * DVector::<f64>::from_column_slice(&beta_std));
        resid_ols.dot(&resid_ols) / ((n - p).max(1) as f64)
    }
    .max(1e-12_f64);

    let selection_criterion = algwas_stage1_select_criterion();
    let mut path = Vec::<AlgwasStage1PathPoint>::with_capacity(step_adj + 1);
    for step in 0..=step_adj {
        let beta_step = reconstruct_msgps_beta(p, &betahat_index, step, delta_t);
        let nnz = beta_step.iter().filter(|&&v| v != 0.0_f64).count();
        let bic = (n as f64) * (2.0_f64 * std::f64::consts::PI * tau2).ln()
            + (rss[step] / tau2)
            + (n as f64).ln() * df.get(step).copied().unwrap_or(0.0_f64);
        let ebic = algwas_ebic_from_bic(bic, p, nnz, ebic_gamma);
        path.push(AlgwasStage1PathPoint {
            lambda: tuning_stand[step] as f32,
            bic,
            ebic,
            rss: rss[step],
            nnz,
            df: df.get(step).copied().unwrap_or(0.0_f64),
            converged,
        });
    }

    let best_bic_idx = best_stage1_path_index(&path, AlgwasStage1SelectCriterion::Bic);
    let best_ebic_idx = best_stage1_path_index(&path, AlgwasStage1SelectCriterion::Ebic);
    let best_step = best_stage1_path_index(&path, selection_criterion);
    let beta_best_std = reconstruct_msgps_beta(p, &betahat_index, best_step, delta_t);
    let mut beta_best = vec![0.0_f32; p];
    let mut selected_indices = Vec::<usize>::new();
    for j in 0..p {
        let bj_std = beta_best_std[j];
        if bj_std != 0.0_f64 {
            ensure_msgps_std_row_cached(design, proj, j, &mut std_cache)?;
            let bj = bj_std * std_cache.get(&j).unwrap().scale;
            beta_best[j] = bj as f32;
            if bj != 0.0_f64 {
                selected_indices.push(j);
            }
        }
    }
    if let Some(qb) = cfg.qtn_bound {
        if selected_indices.len() > qb {
            selected_indices.sort_by(|&a, &b| beta_best[b].abs().total_cmp(&beta_best[a].abs()));
            selected_indices.truncate(qb);
            selected_indices.sort_unstable();
        }
    }

    if let Some(cb) = progress_callback {
        let final_step = step_adj.max(1);
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (final_step, final_step))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }
    xtx_cache.maybe_log_final_summary();

    Ok(AlgwasStage1Result {
        selected_indices,
        beta: beta_best,
        lambda_best: path.get(best_step).map(|p| p.lambda).unwrap_or(0.0_f32),
        bic_best: path
            .get(best_bic_idx)
            .map(|p| p.bic)
            .unwrap_or(f64::INFINITY),
        ebic_best: path
            .get(best_ebic_idx)
            .map(|p| p.ebic)
            .unwrap_or(f64::INFINITY),
        best_step,
        path,
        converged,
    })
}

fn fit_stage1_path_dense_msgps(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y: &[f64],
    cfg: &AlgwasConfig,
    progress_callback: Option<&Py<PyAny>>,
) -> Result<AlgwasStage1Result, String> {
    let ebic_gamma = algwas_stage1_ebic_gamma();
    let n = design.n_samples();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Ok(AlgwasStage1Result {
            selected_indices: Vec::new(),
            beta: vec![0.0_f32; p],
            lambda_best: 0.0_f32,
            bic_best: f64::INFINITY,
            ebic_best: f64::INFINITY,
            best_step: 0usize,
            path: Vec::new(),
            converged: true,
        });
    }

    let mut x_base = design.decode_all_rows_raw_feature_major_f64()?;
    let mut y_base = y.to_vec();
    if proj.q > 1 {
        residualize_dense_columns_in_place(&mut x_base, proj, design.pool.as_ref())?;
        let y_resid = proj.residualize_y(y)?;
        y_base = y_resid.into_iter().map(|v| v as f64).collect();
    }

    let ridge_lambda = DEFAULT_STAGE1_ALASSO_RIDGE_LAMBDA as f64;
    let alasso_gamma = DEFAULT_STAGE1_ALASSO_GAMMA as f64;
    let base_beta = msgps_alasso_weight_beta(
        &x_base,
        &y_base,
        ridge_lambda,
        cfg.lasso
            .max_pcg_iter
            .max(DEFAULT_STAGE1_ALASSO_PCG_MAX_ITERS),
        cfg.lasso.pcg_tol.max(DEFAULT_STAGE1_ALASSO_PCG_TOL),
    )?;
    let mut weights = vec![0.0_f64; p];
    for j in 0..p {
        let bj = base_beta[j].abs();
        weights[j] = if bj == 0.0_f64 {
            f64::INFINITY
        } else {
            bj.powf(-alasso_gamma)
        };
    }

    let mut x_std = x_base.clone();
    let (_row_means, standardize_vec) =
        center_and_scale_columns_in_place(&mut x_std, design.pool.as_ref());
    let y_mean = y_base.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y_base.iter().map(|v| v - y_mean).collect();
    let beta_ols = msgps_standardized_beta_ols(
        &x_std,
        &y_centered,
        ridge_lambda,
        cfg.lasso
            .max_pcg_iter
            .max(DEFAULT_STAGE1_ALASSO_PCG_MAX_ITERS),
        cfg.lasso.pcg_tol.max(DEFAULT_STAGE1_ALASSO_PCG_TOL),
    )?;
    let step_ref = DEFAULT_STAGE1_MSGPS_STEP as f64;
    let delta_t = beta_ols.iter().map(|v| v.abs()).sum::<f64>() / step_ref.max(1.0_f64);
    if !delta_t.is_finite() || delta_t <= 0.0_f64 {
        if let Some(cb) = progress_callback {
            Python::attach(|py| -> PyResult<()> {
                py.check_signals()?;
                cb.call1(py, (1usize, 1usize))?;
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        }
        return Ok(AlgwasStage1Result {
            selected_indices: Vec::new(),
            beta: vec![0.0_f32; p],
            lambda_best: 0.0_f32,
            bic_best: f64::INFINITY,
            ebic_best: f64::INFINITY,
            best_step: 0usize,
            path: vec![AlgwasStage1PathPoint {
                lambda: 0.0_f32,
                bic: f64::INFINITY,
                ebic: f64::INFINITY,
                rss: y_centered.iter().map(|v| v * v).sum(),
                nnz: 0,
                df: 0.0_f64,
                converged: true,
            }],
            converged: true,
        });
    }

    let alpha_n2 = 2.0_f64 / n as f64;
    let threshold = alpha_n2 * delta_t;
    let step_max = DEFAULT_STAGE1_MSGPS_STEP_MAX;
    let pmax = cfg
        .qtn_bound
        .unwrap_or(DEFAULT_STAGE1_MSGPS_PMAX)
        .min(p.max(1));
    let early_stop_cap = algwas_stage1_site_cap(n);

    let xty0 = x_std.transpose() * DVector::<f64>::from_column_slice(&y_centered);
    let mut g: Vec<f64> = xty0.iter().map(|v| alpha_n2 * *v).collect();
    let mut residual = DVector::<f64>::from_column_slice(&y_centered);
    let mut beta_std = vec![0.0_f64; p];
    let mut beta_sign_state = vec![0i8; p];
    let mut selected_order = vec![0usize; p];
    let mut selected_cols = Vec::<usize>::new();
    let mut xtx_cache = vec![Vec::<f64>::new(); p];
    let mut betahat_index = vec![0i32; step_max + 1];
    let mut betahat_index_adj = vec![0usize; step_max + 1];
    let mut tuning = vec![0.0_f64; step_max + 1];
    let mut tuning_stand = vec![0.0_f64; step_max + 1];
    let mut rss = vec![0.0_f64; step_max + 1];
    let mut increment_vec = vec![0.0_f64; step_max + 1];
    let mut sum_lambda = vec![0.0_f64; step_max + 1];
    rss[0] = residual.dot(&residual);

    let mut last_jstar: Option<usize> = None;
    let mut last_sign = 1.0_f64;
    let mut step_adj_raw = step_max + 1;
    let mut include_last_step = false;
    let mut converged = false;
    let beta_zero_tol = msgps_near_zero_tol(delta_t);
    if let Some(cb) = progress_callback {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (0usize, 0usize))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }
    for step in 1..=step_max {
        if let Some(prev_jstar) = last_jstar {
            if xtx_cache[prev_jstar].is_empty() {
                let col = x_std.column(prev_jstar).into_owned();
                let xtx = x_std.transpose() * col;
                xtx_cache[prev_jstar] = xtx.as_slice().to_vec();
            }
            let xtxj = &xtx_cache[prev_jstar];
            let scale = -threshold * last_sign;
            daxpy_inplace_f64(scale, xtxj, &mut g);
        }

        let mut max_idx = 0usize;
        let mut max_val = 0.0_f64;
        let mut below_thr = 0usize;
        let mut sum_abs_lambda = 0.0_f64;
        for j in 0..p {
            let absg = g[j].abs();
            let lambda_raw = g[j] / weights[j];
            let mut score = lambda_raw.abs();
            sum_abs_lambda += lambda_raw.abs();
            let beta_sign =
                msgps_beta_sign_with_memory(beta_std[j], beta_sign_state[j], beta_zero_tol);
            if beta_sign != 0
                && lambda_raw.signum() != 0.0_f64
                && (lambda_raw.signum() as i8) != beta_sign
                && absg > threshold
            {
                score += 1e12_f64;
            }
            if absg < threshold {
                score = 0.0_f64;
                below_thr += 1;
            }
            if score > max_val {
                max_val = score;
                max_idx = j;
            }
        }

        let lambda_j = g[max_idx] / weights[max_idx];
        let sign = if lambda_j > 0.0_f64 {
            1.0_f64
        } else {
            -1.0_f64
        };
        if selected_order[max_idx] == 0 {
            selected_cols.push(max_idx);
            selected_order[max_idx] = selected_cols.len();
        }
        betahat_index_adj[step] = selected_order[max_idx];

        let beta_prev = beta_std[max_idx];
        beta_std[max_idx] += delta_t * sign;
        if beta_std[max_idx] > beta_zero_tol {
            beta_sign_state[max_idx] = 1;
        } else if beta_std[max_idx] < -beta_zero_tol {
            beta_sign_state[max_idx] = -1;
        }
        betahat_index[step] = ((max_idx + 1) as i32) * if sign > 0.0_f64 { 1 } else { -1 };
        increment_vec[step] = if g[max_idx].abs() > 0.0_f64 {
            threshold / g[max_idx].abs()
        } else {
            0.0_f64
        };

        let col = x_std.column(max_idx);
        daxpy_inplace_f64(-delta_t * sign, col.as_slice(), residual.as_mut_slice());
        rss[step] = residual.dot(&residual);
        let delta_pen = weights[max_idx] * (beta_std[max_idx].abs() - beta_prev.abs());
        tuning[step] = tuning[step - 1] + delta_pen;
        tuning_stand[step] = tuning_stand[step - 1] + standardize_vec[max_idx] * delta_pen;
        sum_lambda[step] = sum_abs_lambda;

        last_jstar = Some(max_idx);
        last_sign = sign;
        if selected_cols.len() > early_stop_cap {
            step_adj_raw = step.saturating_add(1);
            include_last_step = true;
            break;
        }
        if below_thr == p || max_val <= 0.0_f64 {
            converged = true;
            step_adj_raw = step;
            break;
        }
        if selected_cols.len() == pmax + 1 {
            step_adj_raw = step;
            break;
        }
        if step > 20 {
            let cur = sum_lambda[step];
            if (step - 20..step).any(|k| (sum_lambda[k] - cur).abs() == 0.0_f64) {
                converged = true;
                step_adj_raw = step;
                break;
            }
        }
        if let Some(cb) = progress_callback {
            if step <= 1024 || step % 64 == 0 || step == step_max {
                Python::attach(|py| -> PyResult<()> {
                    py.check_signals()?;
                    cb.call1(py, (step, 0usize))?;
                    Ok(())
                })
                .map_err(|e| e.to_string())?;
            }
        }
    }
    let step_adj = if include_last_step {
        step_adj_raw.min(step_max)
    } else {
        step_adj_raw.saturating_sub(1).min(step_max)
    };

    let q_n = selected_cols.len();
    let df = if q_n > 0 {
        let mut x_sel = Vec::<f64>::with_capacity(n * q_n);
        for &col_idx in selected_cols.iter().take(q_n) {
            x_sel.extend_from_slice(x_std.column(col_idx).as_slice());
        }
        let x_sel = DMatrix::<f64>::from_vec(n, q_n, x_sel);
        let r = x_sel.qr().r();
        compute_msgps_df_modified(&r, &betahat_index_adj, &increment_vec, step_adj)
    } else {
        vec![0.0_f64; step_adj + 1]
    };

    let tau2 = if n <= p {
        let mean = y_centered.iter().sum::<f64>() / n as f64;
        let mut ss = 0.0_f64;
        for &v in &y_centered {
            let d = v - mean;
            ss += d * d;
        }
        ss / (n as f64 - 1.0_f64).max(1.0_f64)
    } else {
        let resid_ols = DVector::<f64>::from_column_slice(&y_centered)
            - (&x_std * DVector::<f64>::from_column_slice(&beta_ols));
        resid_ols.dot(&resid_ols) / ((n - p).max(1) as f64)
    }
    .max(1e-12_f64);

    let selection_criterion = algwas_stage1_select_criterion();
    let mut path = Vec::<AlgwasStage1PathPoint>::with_capacity(step_adj + 1);
    for step in 0..=step_adj {
        let beta_step = reconstruct_msgps_beta(p, &betahat_index, step, delta_t);
        let nnz = beta_step.iter().filter(|&&v| v != 0.0_f64).count();
        let bic = (n as f64) * (2.0_f64 * std::f64::consts::PI * tau2).ln()
            + (rss[step] / tau2)
            + (n as f64).ln() * df.get(step).copied().unwrap_or(0.0_f64);
        let ebic = algwas_ebic_from_bic(bic, p, nnz, ebic_gamma);
        path.push(AlgwasStage1PathPoint {
            lambda: tuning_stand[step] as f32,
            bic,
            ebic,
            rss: rss[step],
            nnz,
            df: df.get(step).copied().unwrap_or(0.0_f64),
            converged,
        });
    }

    let best_bic_idx = best_stage1_path_index(&path, AlgwasStage1SelectCriterion::Bic);
    let best_ebic_idx = best_stage1_path_index(&path, AlgwasStage1SelectCriterion::Ebic);
    let best_step = best_stage1_path_index(&path, selection_criterion);
    let beta_best_std = reconstruct_msgps_beta(p, &betahat_index, best_step, delta_t);
    let mut beta_best = vec![0.0_f32; p];
    let mut selected_indices = Vec::<usize>::new();
    for j in 0..p {
        let bj = beta_best_std[j] * standardize_vec[j];
        beta_best[j] = bj as f32;
        if bj != 0.0_f64 {
            selected_indices.push(j);
        }
    }
    if let Some(qb) = cfg.qtn_bound {
        if selected_indices.len() > qb {
            selected_indices.sort_by(|&a, &b| beta_best[b].abs().total_cmp(&beta_best[a].abs()));
            selected_indices.truncate(qb);
            selected_indices.sort_unstable();
        }
    }

    if let Some(cb) = progress_callback {
        let final_step = step_adj.max(1);
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (final_step, final_step))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }

    if std::env::var_os("JX_ALGWAS_DEBUG").is_some() {
        if let Ok(path) = std::env::var("JX_ALGWAS_DEBUG_BETAHAT_PATH") {
            if let Ok(file) = File::create(&path) {
                let mut writer = BufWriter::new(file);
                for v in betahat_index.iter().take(step_adj + 1) {
                    let _ = writeln!(writer, "{v}");
                }
                let _ = writer.flush();
            }
        }
        let df_144 = df.get(144).copied().unwrap_or(f64::NAN);
        let df_303 = df.get(303).copied().unwrap_or(f64::NAN);
        let bic_144 = if 144 <= step_adj {
            let rss_144 = rss[144];
            (n as f64) * (2.0_f64 * std::f64::consts::PI * tau2).ln()
                + (rss_144 / tau2)
                + (n as f64).ln() * df_144
        } else {
            f64::NAN
        };
        let bic_303 = if 303 <= step_adj {
            let rss_303 = rss[303];
            (n as f64) * (2.0_f64 * std::f64::consts::PI * tau2).ln()
                + (rss_303 / tau2)
                + (n as f64).ln() * df_303
        } else {
            f64::NAN
        };
        let df_2645 = df.get(2645).copied().unwrap_or(f64::NAN);
        let df_2679 = df.get(2679).copied().unwrap_or(f64::NAN);
        let bic_2645 = if 2645 <= step_adj {
            let rss_2645 = rss[2645];
            (n as f64) * (2.0_f64 * std::f64::consts::PI * tau2).ln()
                + (rss_2645 / tau2)
                + (n as f64).ln() * df_2645
        } else {
            f64::NAN
        };
        let bic_2679 = if 2679 <= step_adj {
            let rss_2679 = rss[2679];
            (n as f64) * (2.0_f64 * std::f64::consts::PI * tau2).ln()
                + (rss_2679 / tau2)
                + (n as f64).ln() * df_2679
        } else {
            f64::NAN
        };
        let first20 = betahat_index
            .iter()
            .take(20)
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "ALGWAS stage1 debug: delta_t={:.15e}, msgps_step_adj={}, best_step={}, best_tuning={:.4e}, best_bic={:.4e}, selected={}, df144={:.4e}, df303={:.4e}, bic144={:.4e}, bic303={:.4e}, df2645={:.4e}, bic2645={:.4e}, df2679={:.4e}, bic2679={:.4e}, first20=[{}]",
            delta_t,
            step_adj,
            best_step,
            tuning_stand[best_step],
            path.get(best_bic_idx).map(|p| p.bic).unwrap_or(f64::NAN),
            selected_indices.len(),
            df_144,
            df_303,
            bic_144,
            bic_303,
            df_2645,
            bic_2645,
            df_2679,
            bic_2679,
            first20
        );
    }

    Ok(AlgwasStage1Result {
        selected_indices,
        beta: beta_best,
        lambda_best: tuning_stand[best_step] as f32,
        bic_best: path
            .get(best_bic_idx)
            .map(|p| p.bic)
            .unwrap_or(f64::INFINITY),
        ebic_best: path
            .get(best_ebic_idx)
            .map(|p| p.ebic)
            .unwrap_or(f64::INFINITY),
        best_step,
        path,
        converged,
    })
}

fn compute_background_model(
    y: &[f64],
    x_bg: &[f64],
    n: usize,
    q: usize,
) -> Result<(Vec<f64>, Vec<f64>, f64, i32), String> {
    if n == 0 || q == 0 {
        return Err("background model requires non-empty X".to_string());
    }
    if x_bg.len() != n.saturating_mul(q) {
        return Err("background X shape mismatch".to_string());
    }
    if n <= q {
        return Err(format!("n too small for background model: n={n}, q={q}"));
    }
    let mut xtx = vec![0.0_f64; q * q];
    let mut xty = vec![0.0_f64; q];
    for i in 0..n {
        let row = &x_bg[i * q..(i + 1) * q];
        let yi = y[i];
        for a in 0..q {
            let va = row[a];
            xty[a] += va * yi;
            for b in 0..=a {
                xtx[a * q + b] += va * row[b];
            }
        }
    }
    for a in 0..q {
        for b in 0..a {
            xtx[b * q + a] = xtx[a * q + b];
        }
    }
    let ixx = pinv_xtx(&xtx, q)?;
    let mut beta = vec![0.0_f64; q];
    matvec_row_major(&ixx, q, &xty, &mut beta);
    let mut rss = 0.0_f64;
    for i in 0..n {
        let row = &x_bg[i * q..(i + 1) * q];
        let fit = row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum::<f64>();
        let d = y[i] - fit;
        rss += d * d;
    }
    Ok((beta, ixx, rss, (n as i32) - (q as i32)))
}

fn row_missing_count_from_float(v: f32) -> usize {
    if !v.is_finite() || v <= 0.0_f32 {
        0usize
    } else {
        v.round().max(0.0_f32) as usize
    }
}

#[allow(clippy::too_many_arguments)]
fn write_stage2_tsv(
    y: &[f64],
    x_cov: &[f64],
    chrom: &[String],
    pos: &[i64],
    snp: &[String],
    allele0: &[String],
    allele1: &[String],
    packed: Cow<'_, [u8]>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_missing: &[f32],
    out_tsv: &str,
    sample_indices: &[usize],
    row_indices: Option<&[usize]>,
    selected_indices: &[usize],
    threads: usize,
    step: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    pseudo_tsv: Option<&str>,
) -> Result<(usize, usize, usize), String> {
    let n = y.len();
    let p_cov = if n == 0 { 0 } else { x_cov.len() / n };
    if x_cov.len() != n.saturating_mul(p_cov) {
        return Err("x_cov shape mismatch".to_string());
    }
    let q_base = 1usize.saturating_add(p_cov);
    let k = selected_indices.len();
    let q_total = q_base + k;
    if chrom.len() != row_flip.len()
        || pos.len() != row_flip.len()
        || snp.len() != row_flip.len()
        || allele0.len() != row_flip.len()
        || allele1.len() != row_flip.len()
        || row_maf.len() != row_flip.len()
        || row_missing.len() != row_flip.len()
    {
        return Err("metadata length mismatch".to_string());
    }
    if sample_indices.len() != n {
        return Err("sample index length mismatch".to_string());
    }
    let mut x_bg = vec![0.0_f64; n * q_total];
    for i in 0..n {
        x_bg[i * q_total] = 1.0_f64;
        if p_cov > 0 {
            let src = &x_cov[i * p_cov..(i + 1) * p_cov];
            let dst = &mut x_bg[i * q_total + 1..i * q_total + q_base];
            dst.copy_from_slice(src);
        }
    }
    if k > 0 {
        let qtn_rows = decode_packed_rows_to_sample_major(
            &packed,
            n_samples.div_ceil(4),
            selected_indices,
            row_indices,
            sample_indices,
            row_flip,
            row_maf,
        )?;
        for i in 0..n {
            let src = &qtn_rows[i * k..(i + 1) * k];
            let dst = &mut x_bg[i * q_total + q_base..(i + 1) * q_total];
            dst.copy_from_slice(src);
        }
    }

    let mut bg_xty = vec![0.0_f64; q_total];
    for i in 0..n {
        let yi = y[i];
        let row = &x_bg[i * q_total..(i + 1) * q_total];
        for j in 0..q_total {
            bg_xty[j] += row[j] * yi;
        }
    }
    let (bg_beta, bg_ixx, bg_rss, bg_df) = compute_background_model(y, &x_bg, n, q_total)?;
    let mut bg_se = vec![f64::NAN; q_total];
    if bg_df > 0 && bg_rss.is_finite() {
        let ve = bg_rss / (bg_df as f64);
        for j in 0..q_total {
            bg_se[j] = (bg_ixx[j * q_total + j] * ve).sqrt();
        }
    }
    let mut qtn_beta = vec![f64::NAN; k];
    let mut qtn_se = vec![f64::NAN; k];
    let mut qtn_pwald = vec![f64::NAN; k];
    let mut qtn_plrt = vec![f64::NAN; k];
    let mut qtn_chisq = vec![f64::NAN; k];
    for j in 0..k {
        let coef_idx = q_base + j;
        if coef_idx < q_total && bg_se[coef_idx].is_finite() && bg_se[coef_idx] > 0.0_f64 {
            let beta = bg_beta[coef_idx];
            let se = bg_se[coef_idx];
            let t = beta / se;
            qtn_beta[j] = beta;
            qtn_se[j] = se;
            qtn_pwald[j] = student_t_p_two_sided(t, bg_df);
            qtn_plrt[j] = lm_plrt_from_t2(t * t, n, bg_df);
            qtn_chisq[j] = chi2_stat_df1_from_sf(qtn_plrt[j]);
        }
    }

    let qtn_lookup: HashMap<usize, usize> = selected_indices
        .iter()
        .enumerate()
        .map(|(j, &idx)| (idx, j))
        .collect();
    let bytes_per_snp = n_samples.div_ceil(4);
    let packed_flat = packed;
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let step = step.max(1);
    let progress_block = if progress_every == 0 {
        step
    } else {
        progress_every.max(1)
    };
    let writer = AsyncTsvWriter::with_config(
        out_tsv,
        b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
        64 * 1024 * 1024,
        4,
    )?;

    let pseudo_writer = if let Some(path) = pseudo_tsv {
        if k > 0 {
            Some(AsyncTsvWriter::with_config(
                path,
                b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n",
                16 * 1024 * 1024,
                4,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    let row_stride = 5usize;
    let mut out_block = vec![0.0_f64; step * row_stride];
    let mut text_buf = String::with_capacity(step * 128);
    let mut pseudo_text_buf = String::with_capacity(k.min(step).max(1) * 128);
    let mut written_rows = 0usize;
    let mut pseudo_rows = 0usize;
    let mut qtn_written = 0usize;

    let mut i = 0usize;
    while i < row_indices.map(|v| v.len()).unwrap_or(row_flip.len()) {
        let m = row_indices.map(|v| v.len()).unwrap_or(row_flip.len());
        let cnt = std::cmp::min(step, m - i);
        let block_slice = &mut out_block[..cnt * row_stride];
        let mut runner = || {
            block_slice
                .par_chunks_mut(row_stride)
                .enumerate()
                .for_each_init(
                    || GlmScratch::new(q_total),
                    |scr, (l, row_out)| {
                        let idx = i + l;
                        let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                        let row =
                            &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                        let flip = row_flip[idx];
                        let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0_f64);
                        scr.reset_xs();
                        let mut sy = 0.0_f64;
                        let mut ss = 0.0_f64;
                        for (k, &sidx) in sample_indices.iter().enumerate() {
                            let b = row[sidx >> 2];
                            let code = (b >> ((sidx & 3) * 2)) & 0b11;
                            let mut gv = match decode_plink_bed_hardcall(code) {
                                Some(v) => v,
                                None => mean_g,
                            };
                            if flip && code != 0b01 {
                                gv = 2.0_f64 - gv;
                            }
                            sy += gv * y[k];
                            ss += gv * gv;
                            let xrow = &x_bg[k * q_total..(k + 1) * q_total];
                            for j in 0..q_total {
                                scr.xs[j] += xrow[j] * gv;
                            }
                        }
                        xs_t_ixx_into(&scr.xs, &bg_ixx, q_total, &mut scr.b21);
                        let t2 = dot_f64(&scr.b21, &scr.xs);
                        let b22 = ss - t2;
                        let (invb22, df) = if b22 < 1e-8_f64 {
                            (0.0_f64, bg_df)
                        } else {
                            (1.0_f64 / b22, bg_df - 1)
                        };
                        if df <= 0 {
                            row_out.fill(f64::NAN);
                            return;
                        }
                        build_ixxs_into(&bg_ixx, &scr.b21, invb22, q_total, &mut scr.ixxs);
                        scr.rhs[..q_total].copy_from_slice(&bg_xty);
                        scr.rhs[q_total] = sy;
                        matvec_row_major(&scr.ixxs, q_total + 1, &scr.rhs, &mut scr.beta);
                        let beta_rhs = dot_f64(&scr.beta, &scr.rhs);
                        let ve = (y.iter().map(|v| v * v).sum::<f64>() - beta_rhs) / (df as f64);
                        let beta_snp = scr.beta[q_total];
                        let se_snp = (scr.ixxs[q_total * (q_total + 1) + q_total] * ve).sqrt();
                        if invb22 == 0.0
                            || !beta_snp.is_finite()
                            || !se_snp.is_finite()
                            || se_snp <= 0.0
                        {
                            row_out.fill(f64::NAN);
                            return;
                        }
                        let t = beta_snp / se_snp;
                        let pwald = student_t_p_two_sided(t, df);
                        let plrt = lm_plrt_from_t2(t * t, n, df);
                        let stat = chi2_stat_df1_from_sf(pwald);
                        row_out[0] = beta_snp;
                        row_out[1] = se_snp;
                        row_out[2] = stat;
                        row_out[3] = pwald;
                        row_out[4] = plrt;
                    },
                );
        };
        if let Some(p) = &pool {
            p.install(runner);
        } else {
            runner();
        }

        text_buf.clear();
        pseudo_text_buf.clear();
        for l in 0..cnt {
            let idx = i + l;
            let base = l * row_stride;
            let mut beta = out_block[base];
            let mut se = out_block[base + 1];
            let mut chisq = out_block[base + 2];
            let mut pwald = out_block[base + 3];
            let mut plrt = out_block[base + 4];
            if let Some(&j) = qtn_lookup.get(&idx) {
                beta = qtn_beta[j];
                se = qtn_se[j];
                pwald = qtn_pwald[j];
                plrt = qtn_plrt[j];
                chisq = qtn_chisq[j];
                qtn_written += 1;
            }
            let miss = row_missing_count_from_float(row_missing[idx]);
            let _ = write!(
                text_buf,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                chrom[idx],
                pos[idx],
                snp[idx],
                allele0[idx],
                allele1[idx],
                row_maf[idx],
                miss,
                beta,
                se,
                format_chisq_value(chisq),
                pwald,
                plrt
            );
            if qtn_lookup.contains_key(&idx) {
                let _ = write!(
                    pseudo_text_buf,
                    "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}\n",
                    chrom[idx],
                    pos[idx],
                    snp[idx],
                    allele0[idx],
                    allele1[idx],
                    row_maf[idx],
                    miss,
                    beta,
                    se,
                    format_chisq_value(chisq),
                    pwald,
                    plrt
                );
            }
        }
        if !text_buf.is_empty() {
            writer.send(text_buf.as_bytes().to_vec())?;
        }
        if let Some(pw) = pseudo_writer.as_ref() {
            if !pseudo_text_buf.is_empty() {
                pw.send(pseudo_text_buf.as_bytes().to_vec())?;
            }
        }

        written_rows += cnt;
        if let Some(cb) = progress_callback.as_ref() {
            let done = written_rows;
            if done % progress_block == 0 || done == m {
                Python::attach(|py| -> PyResult<()> {
                    py.check_signals()?;
                    cb.call1(py, (done, m))?;
                    Ok(())
                })
                .map_err(|e| e.to_string())?;
            }
        }
        i += cnt;
    }
    writer.finish()?;
    if let Some(pw) = pseudo_writer {
        pw.finish()?;
        pseudo_rows = qtn_written;
    }
    Ok((k, pseudo_rows, written_rows))
}

#[derive(Clone, Debug)]
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
            xs: vec![0.0_f64; q0],
            b21: vec![0.0_f64; q0],
            rhs: vec![0.0_f64; dim],
            beta: vec![0.0_f64; dim],
            ixxs: vec![0.0_f64; dim * dim],
        }
    }

    #[inline]
    fn reset_xs(&mut self) {
        self.xs.fill(0.0_f64);
    }
}

#[inline]
fn xs_t_ixx_into(xs: &[f64], ixx: &[f64], q0: usize, out_b21: &mut [f64]) {
    for j in 0..q0 {
        let mut acc = 0.0_f64;
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
    let a = v / 2.0_f64;
    let b = 0.5_f64;
    let mut p = betai(a, b, x);
    if !p.is_finite() {
        p = 1.0_f64;
    }
    p.clamp(f64::MIN_POSITIVE, 1.0_f64)
}

#[inline]
fn lm_plrt_from_t2(t2: f64, n_obs: usize, df: i32) -> f64 {
    if df <= 0 || !t2.is_finite() || t2 < 0.0_f64 {
        return f64::NAN;
    }
    let stat = (n_obs as f64) * (1.0_f64 + t2 / (df as f64)).ln();
    chi2_sf_df1(stat)
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let maxit = 200usize;
    let eps = 3.0e-14_f64;
    let fpmin = 1.0e-300_f64;

    let qab = a + b;
    let qap = a + 1.0_f64;
    let qam = a - 1.0_f64;

    let mut c = 1.0_f64;
    let mut d = 1.0_f64 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0_f64 / d;
    let mut h = d;

    for m in 1..=maxit {
        let m2 = 2.0_f64 * (m as f64);
        let mut aa = (m as f64) * (b - (m as f64)) * x / ((qam + m2) * (a + m2));
        d = 1.0_f64 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0_f64 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0_f64 / d;
        h *= d * c;

        aa = -(a + (m as f64)) * (qab + (m as f64)) * x / ((a + m2) * (qap + m2));
        d = 1.0_f64 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0_f64 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0_f64 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0_f64).abs() < eps {
            break;
        }
    }
    h
}

fn betai(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0_f64..=1.0_f64).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0_f64 {
        return 0.0_f64;
    }
    if x == 1.0_f64 {
        return 1.0_f64;
    }
    let ln_beta = libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);
    if x < (a + 1.0_f64) / (a + b + 2.0_f64) {
        let front = ((a * x.ln()) + (b * (1.0_f64 - x).ln()) - ln_beta).exp() / a;
        front * betacf(a, b, x)
    } else {
        let front = ((b * (1.0_f64 - x).ln()) + (a * x.ln()) - ln_beta).exp() / b;
        1.0_f64 - front * betacf(b, a, 1.0_f64 - x)
    }
}

fn fit_stage1_path(
    design: &AlgwasPackedDesign<'_>,
    proj: &CovariateProjection,
    y: &[f64],
    cfg: &AlgwasConfig,
    progress_callback: Option<&Py<PyAny>>,
) -> Result<AlgwasStage1Result, String> {
    let requested_mode = algwas_stage1_mode();
    let mode = algwas_stage1_mode_resolved(requested_mode, design.n_samples(), design.n_features());
    if algwas_stage1_mode_log_enabled() && matches!(requested_mode, AlgwasStage1Mode::Auto) {
        let (max_features, max_cells) = algwas_stage1_auto_limits();
        let chosen = match mode {
            AlgwasStage1Mode::PackedExactMsgps => "packed_exact",
            AlgwasStage1Mode::StreamActive => "stream_active",
            AlgwasStage1Mode::DenseMsgps => "dense_msgps",
            AlgwasStage1Mode::Auto => "auto",
        };
        eprintln!(
            "ALGWAS stage1 auto selected: {} (n={}, p={}, cells={}, max_features={}, max_cells={})",
            chosen,
            design.n_samples(),
            design.n_features(),
            design.n_samples().saturating_mul(design.n_features()),
            max_features,
            max_cells,
        );
    }
    match mode {
        AlgwasStage1Mode::PackedExactMsgps => {
            fit_stage1_path_exact_packed(design, proj, y, cfg, progress_callback)
        }
        AlgwasStage1Mode::StreamActive => {
            fit_stage1_path_streaming(design, proj, y, cfg, progress_callback)
        }
        AlgwasStage1Mode::DenseMsgps => {
            fit_stage1_path_dense_msgps(design, proj, y, cfg, progress_callback)
        }
        AlgwasStage1Mode::Auto => {
            Err("ALGWAS stage1 auto mode was not resolved before dispatch".to_string())
        }
    }
}

#[inline]
fn algwas_stage1_summary_path(out_tsv: &str) -> String {
    if let Some(prefix) = out_tsv.strip_suffix(".tsv") {
        format!("{prefix}.stage1.tsv")
    } else {
        format!("{out_tsv}.stage1.tsv")
    }
}

fn write_algwas_stage1_summary_tsv(
    out_tsv: &str,
    stage1: &AlgwasStage1Result,
) -> Result<String, String> {
    let summary_path = algwas_stage1_summary_path(out_tsv);
    let file = File::create(&summary_path).map_err(|e| format!("create {}: {e}", summary_path))?;
    let mut writer = BufWriter::with_capacity(256 * 1024, file);
    writer
        .write_all(
            b"step\tlambda\trss\tnnz\tdf\tbic\tebic\tconverged\tselected_by_bic\tselected_by_ebic\tselected_by_model\n",
        )
        .map_err(|e| format!("write header {}: {e}", summary_path))?;
    let best_bic_idx = stage1
        .path
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.bic.total_cmp(&b.1.bic))
        .map(|(idx, _)| idx)
        .unwrap_or(0usize);
    let best_ebic_idx = stage1
        .path
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.ebic.total_cmp(&b.1.ebic))
        .map(|(idx, _)| idx)
        .unwrap_or(0usize);
    for (step, point) in stage1.path.iter().enumerate() {
        writeln!(
            writer,
            "{step}\t{:.8e}\t{:.8e}\t{}\t{:.8e}\t{:.8e}\t{:.8e}\t{}\t{}\t{}\t{}",
            point.lambda as f64,
            point.rss,
            point.nnz,
            point.df,
            point.bic,
            point.ebic,
            if point.converged { 1 } else { 0 },
            if step == best_bic_idx { 1 } else { 0 },
            if step == best_ebic_idx { 1 } else { 0 },
            if step == stage1.best_step { 1 } else { 0 },
        )
        .map_err(|e| format!("write {}: {e}", summary_path))?;
    }
    writer
        .flush()
        .map_err(|e| format!("flush {}: {e}", summary_path))?;
    Ok(summary_path)
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
    qtn_bound=None,
    lambda_steps=24,
    lambda_min_ratio=0.05,
    scan_step=10000,
    stage1_progress_callback=None,
    threads=0,
    progress_callback=None,
    pseudo_tsv=None
))]
pub fn algwas_packed_to_tsv(
    _py: Python<'_>,
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
    qtn_bound: Option<usize>,
    lambda_steps: usize,
    lambda_min_ratio: f32,
    scan_step: usize,
    stage1_progress_callback: Option<Py<PyAny>>,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    pseudo_tsv: Option<&str>,
) -> PyResult<(usize, usize, usize)> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let y = y.as_slice()?;
    let x_cov_arr = x_cov.as_array();
    if x_cov_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("x_cov must be 2D (n, p_cov)"));
    }
    let n = y.len();
    let p_cov = x_cov_arr.shape()[1];
    if x_cov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("x_cov rows must equal len(y)"));
    }
    let x_cov_flat: Cow<[f64]> = match x_cov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_cov_arr.iter().copied().collect()),
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

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;
    let row_idx: Option<Vec<usize>> = if let Some(ridx) = row_indices {
        Some(parse_index_vec_i64(
            ridx.as_slice()?,
            row_flip.len(),
            "row_indices",
        )?)
    } else {
        None
    };
    let m = row_idx.as_ref().map(|v| v.len()).unwrap_or(row_flip.len());
    if row_flip.len() != m || row_maf.len() != m || row_missing.len() != m {
        return Err(PyRuntimeError::new_err("row metadata length mismatch"));
    }
    if chrom.len() != m || pos.len() != m || snp.len() != m || allele0.len() != m || allele1.len() != m {
        return Err(PyRuntimeError::new_err("metadata length mismatch"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("packed must be 2D"));
    }
    let m_packed = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    if packed_flat.len() != m_packed * bytes_per_snp {
        return Err(PyRuntimeError::new_err("invalid packed flattened length"));
    }

    let design = AlgwasPackedDesign::from_parts(
        packed_flat.clone(),
        n_samples,
        row_flip.to_vec(),
        row_maf.to_vec(),
        sample_idx.clone(),
        row_idx.clone(),
        threads,
    )
    .map_err(PyRuntimeError::new_err)?;
    let cov_proj =
        CovariateProjection::new(x_cov_flat.as_ref(), n, p_cov).map_err(PyRuntimeError::new_err)?;
    let alg_cfg = AlgwasConfig {
        lambda_steps: lambda_steps.max(1),
        lambda_min_ratio: lambda_min_ratio.clamp(1e-6_f32, 1.0_f32),
        qtn_bound,
        lasso: crate::lasso::LassoConfig::default(),
    };

    let (qtn_count, pseudo_rows, written_rows) = _py
        .detach(move || -> Result<(usize, usize, usize), String> {
            let stage1 = fit_stage1_path(
                &design,
                &cov_proj,
                y,
                &alg_cfg,
                stage1_progress_callback.as_ref(),
            )?;
            write_algwas_stage1_summary_tsv(out_tsv, &stage1)?;
            let mut selected_indices = stage1.selected_indices.clone();
            if selected_indices.len() > 1 {
                selected_indices = algwas_stage1_ld_prune_selected(
                    &design,
                    &chrom,
                    &pos,
                    &stage1.beta,
                    &selected_indices,
                )?;
            }

            write_stage2_tsv(
                y,
                x_cov_flat.as_ref(),
                &chrom,
                &pos,
                &snp,
                &allele0,
                &allele1,
                packed_flat,
                n_samples,
                row_flip,
                row_maf,
                row_missing,
                out_tsv,
                &sample_idx,
                row_idx.as_deref(),
                &selected_indices,
                threads,
                scan_step.max(1),
                progress_callback,
                scan_step.max(1),
                pseudo_tsv,
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok((qtn_count, pseudo_rows, written_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use std::fs;
    use std::path::Path;

    fn pack_dosages_012(dosages: &[u8], n_samples: usize) -> Vec<u8> {
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut out = vec![0u8; bytes_per_snp];
        for (sid, &g) in dosages.iter().enumerate() {
            let code = match g {
                0 => 0b00,
                1 => 0b10,
                2 => 0b11,
                _ => panic!("test helper only supports 0/1/2 dosages"),
            };
            out[sid >> 2] |= code << ((sid & 3) * 2);
        }
        out
    }

    fn maf_from_row(dosages: &[u8]) -> f32 {
        let af = dosages.iter().map(|&g| g as f32).sum::<f32>() / (2.0_f32 * dosages.len() as f32);
        af.min(1.0_f32 - af)
    }

    fn standardized_row(dosages: &[u8], maf: f32) -> Vec<f64> {
        let mean = 2.0_f64 * maf as f64;
        let inv_sd = 1.0_f64 / (2.0_f64 * maf as f64 * (1.0_f64 - maf as f64)).sqrt();
        dosages
            .iter()
            .map(|&g| ((g as f64) - mean) * inv_sd)
            .collect()
    }

    fn build_test_design(rows: &[Vec<u8>]) -> (AlgwasPackedDesign<'static>, Vec<f32>) {
        let n_samples = rows[0].len();
        let mut packed = Vec::<u8>::new();
        let mut row_maf = Vec::<f32>::with_capacity(rows.len());
        for row in rows {
            row_maf.push(maf_from_row(row));
            packed.extend_from_slice(&pack_dosages_012(row, n_samples));
        }
        let design = AlgwasPackedDesign::from_parts(
            Cow::Owned(packed),
            n_samples,
            vec![false; rows.len()],
            row_maf.clone(),
            (0..n_samples).collect(),
            None,
            0,
        )
        .unwrap();
        (design, row_maf)
    }

    fn load_mouse_hs1940_stage1_fixture() -> (AlgwasPackedDesign<'static>, Vec<f64>) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("test");
        let geno_path = root.join("mouse_hs1940.txt");
        let pheno_path = root.join("mouse_hs1940.pheno.txt");
        let geno_txt = fs::read_to_string(&geno_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", geno_path.display()));
        let mut rows = Vec::<Vec<u8>>::new();
        for (lineno, line) in geno_txt.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row = line
                .split_whitespace()
                .map(|tok| {
                    tok.parse::<u8>().unwrap_or_else(|e| {
                        panic!("parse genotype at line {} token {tok}: {e}", lineno + 1)
                    })
                })
                .collect::<Vec<_>>();
            rows.push(row);
        }
        let y = fs::read_to_string(&pheno_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", pheno_path.display()))
            .lines()
            .skip(1)
            .map(|line| {
                let mut it = line.split_whitespace();
                let _iid = it.next().unwrap_or("");
                let value = it.next().unwrap_or("");
                value
                    .parse::<f64>()
                    .unwrap_or_else(|e| panic!("parse phenotype line {line:?}: {e}"))
            })
            .collect::<Vec<_>>();
        let (design, _row_maf) = build_test_design(&rows);
        (design, y)
    }

    #[test]
    fn msgps_alasso_weights_match_dense_ridge_reference() {
        let rows = vec![
            vec![0u8, 0, 1, 1, 2, 2],
            vec![0u8, 1, 0, 1, 0, 1],
            vec![2u8, 1, 2, 1, 0, 0],
        ];
        let n_samples = rows[0].len();
        let n_features = rows.len();
        let (design, row_maf) = build_test_design(&rows);
        let proj = CovariateProjection::new(&[], n_samples, 0).unwrap();
        let y = vec![0.0_f64, 0.2, 1.0, 1.2, 2.1, 2.0];
        let y_resid = proj.residualize_y(&y).unwrap();
        let diag_xtx = design.diag_xtx_residualized(&proj).unwrap();
        let cfg = AlgwasConfig::default();
        let adaptive_init =
            compute_msgps_alasso_init(&design, &proj, &y_resid, &diag_xtx, None, &cfg).unwrap();
        let weights = adaptive_init.penalty_weights;

        let y_mean = y.iter().sum::<f64>() / n_samples as f64;
        let y_centered: Vec<f64> = y.iter().map(|v| v - y_mean).collect();
        let mut x_data = vec![0.0_f64; n_samples * n_features];
        for (j, row) in rows.iter().enumerate() {
            let xj = standardized_row(row, row_maf[j]);
            for i in 0..n_samples {
                x_data[i * n_features + j] = xj[i];
            }
        }
        let x = DMatrix::<f64>::from_row_slice(n_samples, n_features, &x_data);
        let yv = DVector::<f64>::from_row_slice(&y_centered);
        let xtx = x.transpose() * &x + DMatrix::<f64>::identity(n_features, n_features) * 0.001_f64;
        let xty = x.transpose() * yv;
        let beta = xtx.lu().solve(&xty).unwrap();

        for j in 0..n_features {
            let want = beta[j]
                .abs()
                .max(DEFAULT_STAGE1_ALASSO_WEIGHT_FLOOR as f64)
                .powf(-(DEFAULT_STAGE1_ALASSO_GAMMA as f64));
            let got = weights[j] as f64;
            let rel = ((got - want).abs()) / want.max(1e-8_f64);
            assert!(rel < 1e-3_f64, "j={j} got={got} want={want} rel={rel}");
        }
        assert!(weights[0] < weights[1]);
        assert!(weights[0] < weights[2]);
    }

    #[test]
    fn algwas_stage1_selects_strong_signal() {
        let rows = vec![
            vec![0u8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            vec![0u8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            vec![2u8, 2, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1],
            vec![0u8, 0, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1],
        ];
        let n_samples = rows[0].len();
        let (design, row_maf) = build_test_design(&rows);
        let proj = CovariateProjection::new(&[], n_samples, 0).unwrap();
        let x0 = standardized_row(&rows[0], row_maf[0]);
        let x1 = standardized_row(&rows[1], row_maf[1]);
        let y: Vec<f64> = (0..n_samples)
            .map(|i| 5.0_f64 * x0[i] + 0.35_f64 * x1[i] + ((i % 3) as f64 - 1.0_f64) * 0.02_f64)
            .collect();
        let cfg = AlgwasConfig {
            lambda_steps: 32,
            lambda_min_ratio: 1e-3_f32,
            qtn_bound: None,
            lasso: crate::lasso::LassoConfig {
                max_admm_iter: 128,
                max_pcg_iter: 96,
                active_cd_tol: 1e-5_f32,
                active_kkt_tol: 5e-5_f32,
                ..crate::lasso::LassoConfig::default()
            },
        };
        let res = fit_stage1_path(&design, &proj, &y, &cfg, None).unwrap();
        assert!(
            res.selected_indices.contains(&0),
            "selected={:?} lambda_best={} bic_best={}",
            res.selected_indices,
            res.lambda_best,
            res.bic_best
        );
        assert!(!res.selected_indices.is_empty());
    }

    #[test]
    fn algwas_stage1_auto_mode_switches_by_scale() {
        assert_eq!(
            algwas_stage1_mode_from_raw(None),
            AlgwasStage1Mode::StreamActive
        );
        assert_eq!(
            algwas_stage1_mode_from_raw(Some("auto")),
            AlgwasStage1Mode::Auto
        );
        assert_eq!(
            algwas_stage1_mode_resolved_with_limits(
                AlgwasStage1Mode::Auto,
                1940,
                8960,
                DEFAULT_STAGE1_AUTO_EXACT_MAX_FEATURES,
                DEFAULT_STAGE1_AUTO_EXACT_MAX_CELLS,
            ),
            AlgwasStage1Mode::PackedExactMsgps
        );
        assert_eq!(
            algwas_stage1_mode_resolved_with_limits(
                AlgwasStage1Mode::Auto,
                1940,
                10_000_000,
                DEFAULT_STAGE1_AUTO_EXACT_MAX_FEATURES,
                DEFAULT_STAGE1_AUTO_EXACT_MAX_CELLS,
            ),
            AlgwasStage1Mode::StreamActive
        );
        assert_eq!(
            algwas_stage1_mode_resolved_with_limits(
                AlgwasStage1Mode::Auto,
                200_000,
                1024,
                DEFAULT_STAGE1_AUTO_EXACT_MAX_FEATURES,
                DEFAULT_STAGE1_AUTO_EXACT_MAX_CELLS,
            ),
            AlgwasStage1Mode::StreamActive
        );
    }

    #[test]
    fn intercept_only_stream_stats_match_manual_formulas() {
        let rows = vec![
            vec![0u8, 0, 1, 1, 2, 2],
            vec![0u8, 1, 0, 1, 0, 1],
            vec![2u8, 1, 2, 1, 0, 0],
        ];
        let n_samples = rows[0].len();
        let (design, row_maf) = build_test_design(&rows);
        let proj = CovariateProjection::new(&[], n_samples, 0).unwrap();
        let diag_xtx = design.diag_xtx_residualized(&proj).unwrap();
        let y = vec![0.0_f64, 0.2, 1.0, 1.2, 2.1, 2.0];
        let y_resid = proj.residualize_y(&y).unwrap();
        let mut xty = vec![0.0_f32; design.n_features()];
        design
            .apply_xt_resid_into(&proj, &y_resid, &mut xty)
            .unwrap();

        let y_mean = y.iter().sum::<f64>() / n_samples as f64;
        let y_centered: Vec<f64> = y.iter().map(|v| v - y_mean).collect();
        for j in 0..rows.len() {
            let x = standardized_row(&rows[j], row_maf[j]);
            let row_sum = x.iter().sum::<f64>();
            let row_ss = x.iter().map(|v| v * v).sum::<f64>();
            let want_diag = (row_ss - row_sum * row_sum / n_samples as f64).max(1e-12_f64);
            let got_diag = diag_xtx[j] as f64;
            let rel_diag = (got_diag - want_diag).abs() / want_diag.max(1e-8_f64);
            assert!(
                rel_diag < 1e-5_f64,
                "diag j={j} got={got_diag} want={want_diag} rel={rel_diag}"
            );

            let want_xty = x
                .iter()
                .zip(y_centered.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let got_xty = xty[j] as f64;
            let rel_xty = (got_xty - want_xty).abs() / want_xty.abs().max(1e-8_f64);
            assert!(
                rel_xty < 1e-5_f64,
                "xty j={j} got={got_xty} want={want_xty} rel={rel_xty}"
            );
        }
    }

    #[test]
    fn algwas_stage1_lm_screen_prefers_strong_marginal_signal() {
        let xty = vec![0.5_f32, 8.0_f32, 3.0_f32, 0.1_f32];
        let diag = vec![1.0_f32, 4.0_f32, 9.0_f32, 1.0_f32];
        let rows = select_stage1_lm_screen_rows(&xty, &diag, 2);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], 1);
        assert_eq!(rows[1], 2);
    }

    #[test]
    fn algwas_stage1_site_cap_default_matches_spec() {
        assert_eq!(algwas_stage1_site_cap(1), 1);
        assert_eq!(algwas_stage1_site_cap(12), 3);
        assert_eq!(algwas_stage1_site_cap(144), 12);
        assert_eq!(algwas_stage1_site_cap(10_000), 50);
    }

    #[test]
    fn algwas_stage1_ld_prune_removes_redundant_same_chrom_sites() {
        let rows = vec![
            vec![0u8, 0, 1, 1, 2, 2, 1, 1],
            vec![0u8, 0, 1, 1, 2, 2, 1, 1],
            vec![2u8, 1, 2, 1, 0, 0, 1, 1],
        ];
        let (design, _row_maf) = build_test_design(&rows);
        let chrom = vec!["1".to_string(), "1".to_string(), "2".to_string()];
        let pos = vec![100_i64, 120_i64, 200_i64];
        let beta = vec![0.4_f32, 1.2_f32, 0.8_f32];
        let selected = vec![0usize, 1usize, 2usize];

        let kept = algwas_stage1_ld_prune_selected(&design, &chrom, &pos, &beta, &selected)
            .expect("ld prune should succeed");

        assert_eq!(kept, vec![1usize, 2usize]);
    }

    #[test]
    fn algwas_stage1_stream_warm_state_matches_independent_refits() {
        let rows = vec![
            vec![0u8, 0, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1],
            vec![0u8, 1, 2, 1, 0, 1, 2, 1, 0, 1, 0, 1],
            vec![2u8, 2, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1],
            vec![1u8, 1, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2],
            vec![0u8, 1, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1],
        ];
        let n_samples = rows[0].len();
        let (design, row_maf) = build_test_design(&rows);
        let x_cov: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64 - 5.5_f64) * 0.15_f64)
            .collect();
        let proj = CovariateProjection::new(&x_cov, n_samples, 1).unwrap();
        let x0 = standardized_row(&rows[0], row_maf[0]);
        let x2 = standardized_row(&rows[2], row_maf[2]);
        let x4 = standardized_row(&rows[4], row_maf[4]);
        let y: Vec<f64> = (0..n_samples)
            .map(|i| {
                2.5_f64 * x0[i] - 0.9_f64 * x2[i] + 0.35_f64 * x4[i] + 0.2_f64 * x_cov[i]
                    - 0.03_f64 * (i as f64)
            })
            .collect();
        let cfg = AlgwasConfig {
            lambda_steps: 10,
            lambda_min_ratio: 1e-2_f32,
            qtn_bound: None,
            lasso: crate::lasso::LassoConfig {
                max_admm_iter: 128,
                max_pcg_iter: 96,
                active_cd_tol: 1e-6_f32,
                active_kkt_tol: 5e-6_f32,
                ..crate::lasso::LassoConfig::default()
            },
        };

        let y_resid = proj.residualize_y(&y).unwrap();
        let diag_xtx = design.diag_xtx_residualized(&proj).unwrap();
        let xty = design.xty_residualized(&y_resid).unwrap();
        let adaptive_init =
            compute_msgps_alasso_init(&design, &proj, &y_resid, &diag_xtx, Some(&xty), &cfg)
                .unwrap();
        let mut penalty_weights = adaptive_init.penalty_weights;
        for w in &mut penalty_weights {
            if !w.is_finite() || *w <= 0.0_f32 {
                *w = 1.0_f32;
            }
        }
        let initial_working_set = select_stage1_lm_screen_rows(
            &xty,
            &diag_xtx,
            algwas_stage1_initial_working_set_cap(design.n_features()),
        );
        let lambda_max = weighted_lambda_max(&xty, &penalty_weights);
        let lambda_min = (lambda_max * cfg.lambda_min_ratio).max(lambda_max * 1e-6_f32);
        let lambda_span = (lambda_min / lambda_max).max(1e-6_f32);

        let mut beta_prev: Option<Vec<f32>> = None;
        let mut warm_state: Option<AlgwasStage1WarmState> = None;
        for step in 0..cfg.lambda_steps {
            let t = if cfg.lambda_steps <= 1 {
                0.0_f32
            } else {
                step as f32 / (cfg.lambda_steps - 1) as f32
            };
            let lambda = lambda_max * lambda_span.powf(t);
            let (beta_refit, rss_refit, conv_refit, _iters_refit) =
                fit_algwas_stage1_active_from_stats(
                    &design,
                    &proj,
                    &y_resid,
                    &xty,
                    &diag_xtx,
                    &penalty_weights,
                    lambda,
                    &cfg,
                    beta_prev.as_deref(),
                    Some(&initial_working_set),
                )
                .unwrap();
            let use_seed_rows = warm_state.is_none();
            let (beta_warm, rss_warm, conv_warm, _iters_warm, next_warm_state) =
                solve_algwas_stage1_active_from_stats(
                    &design,
                    &proj,
                    &y_resid,
                    &xty,
                    &diag_xtx,
                    &penalty_weights,
                    lambda,
                    &cfg,
                    None,
                    if use_seed_rows {
                        Some(&initial_working_set)
                    } else {
                        None
                    },
                    warm_state.take(),
                )
                .unwrap();

            assert_eq!(conv_refit, conv_warm, "lambda step={step} lambda={lambda}");
            for (j, (a, b)) in beta_refit.iter().zip(beta_warm.iter()).enumerate() {
                let diff = (*a as f64 - *b as f64).abs();
                assert!(
                    diff < 5e-5_f64,
                    "beta mismatch step={step} lambda={lambda} j={j} refit={a} warm={b} diff={diff}"
                );
            }
            let rss_diff = (rss_refit - rss_warm).abs();
            assert!(
                rss_diff < 5e-5_f64,
                "rss mismatch step={step} lambda={lambda} refit={rss_refit} warm={rss_warm} diff={rss_diff}"
            );

            beta_prev = Some(beta_refit);
            warm_state = Some(next_warm_state);
        }
    }

    #[test]
    fn algwas_stage1_exact_packed_matches_dense_reference() {
        let rows = vec![
            vec![0u8, 0, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1],
            vec![0u8, 1, 2, 1, 0, 1, 2, 1, 0, 1, 0, 1],
            vec![2u8, 2, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1],
            vec![1u8, 1, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2],
        ];
        let n_samples = rows[0].len();
        let (design, row_maf) = build_test_design(&rows);
        let x_cov: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64 - 5.5_f64) * 0.2_f64)
            .collect();
        let proj = CovariateProjection::new(&x_cov, n_samples, 1).unwrap();
        let x0 = standardized_row(&rows[0], row_maf[0]);
        let x2 = standardized_row(&rows[2], row_maf[2]);
        let y: Vec<f64> = (0..n_samples)
            .map(|i| {
                2.25_f64 * x0[i] - 0.75_f64 * x2[i]
                    + 0.1_f64 * x_cov[i]
                    + ((i % 4) as f64 - 1.5_f64) * 0.01_f64
            })
            .collect();
        let cfg = AlgwasConfig {
            lambda_steps: 24,
            lambda_min_ratio: 1e-3_f32,
            qtn_bound: None,
            lasso: crate::lasso::LassoConfig {
                max_admm_iter: 128,
                max_pcg_iter: 96,
                active_cd_tol: 1e-5_f32,
                active_kkt_tol: 5e-5_f32,
                ..crate::lasso::LassoConfig::default()
            },
        };
        let dense = fit_stage1_path_dense_msgps(&design, &proj, &y, &cfg, None).unwrap();
        let exact = fit_stage1_path_exact_packed(&design, &proj, &y, &cfg, None).unwrap();
        assert_eq!(dense.selected_indices, exact.selected_indices);
        assert_eq!(dense.beta.len(), exact.beta.len());
        for (j, (a, b)) in dense.beta.iter().zip(exact.beta.iter()).enumerate() {
            let diff = (*a as f64 - *b as f64).abs();
            assert!(
                diff < 1e-4_f64,
                "beta mismatch j={j} dense={a} exact={b} diff={diff}"
            );
        }
        assert!((dense.lambda_best - exact.lambda_best).abs() < 1e-5_f32);
        assert_eq!(dense.path.len(), exact.path.len());
    }

    #[test]
    fn algwas_exact_packed_pcg_prelude_matches_dense_reference() {
        let rows = vec![
            vec![0u8, 0, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1],
            vec![0u8, 1, 2, 1, 0, 1, 2, 1, 0, 1, 0, 1],
            vec![2u8, 2, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1],
            vec![1u8, 1, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2],
        ];
        let n_samples = rows[0].len();
        let (design, row_maf) = build_test_design(&rows);
        let x_cov: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64 - 5.5_f64) * 0.2_f64)
            .collect();
        let proj = CovariateProjection::new(&x_cov, n_samples, 1).unwrap();
        let x0 = standardized_row(&rows[0], row_maf[0]);
        let x2 = standardized_row(&rows[2], row_maf[2]);
        let y: Vec<f64> = (0..n_samples)
            .map(|i| {
                2.25_f64 * x0[i] - 0.75_f64 * x2[i]
                    + 0.1_f64 * x_cov[i]
                    + ((i % 4) as f64 - 1.5_f64) * 0.01_f64
            })
            .collect();
        let cfg = AlgwasConfig {
            lambda_steps: 24,
            lambda_min_ratio: 1e-3_f32,
            qtn_bound: None,
            lasso: crate::lasso::LassoConfig {
                max_admm_iter: 128,
                max_pcg_iter: 96,
                active_cd_tol: 1e-5_f32,
                active_kkt_tol: 5e-5_f32,
                ..crate::lasso::LassoConfig::default()
            },
        };
        let dense = compute_exact_packed_stage1_prelude_dense(&design, &proj, &y).unwrap();
        let pcg = compute_exact_packed_stage1_prelude_pcg(&design, &proj, &y, &cfg).unwrap();
        assert_eq!(dense.weights.len(), pcg.weights.len());
        assert_eq!(dense.xty0.len(), pcg.xty0.len());
        for (a, b) in dense.y_centered.iter().zip(pcg.y_centered.iter()) {
            assert!((a - b).abs() < 1e-8_f64);
        }
        for (j, (a, b)) in dense.weights.iter().zip(pcg.weights.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-4_f64,
                "weight mismatch j={j} dense={a} pcg={b} diff={diff}"
            );
        }
        for (j, (a, b)) in dense.xty0.iter().zip(pcg.xty0.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-4_f64,
                "xty0 mismatch j={j} dense={a} pcg={b} diff={diff}"
            );
        }
        let diff = (dense.beta_ols_abs_sum - pcg.beta_ols_abs_sum).abs();
        assert!(
            diff < 1e-4_f64,
            "beta_ols_abs_sum mismatch dense={} pcg={} diff={diff}",
            dense.beta_ols_abs_sum,
            pcg.beta_ols_abs_sum
        );
    }

    #[test]
    #[ignore = "diagnostic regression against mouse_hs1940 stage1 semantics"]
    fn debug_mouse_hs1940_stage1_metrics() {
        let (design, y) = load_mouse_hs1940_stage1_fixture();
        let proj = CovariateProjection::new(&[], y.len(), 0).unwrap();
        let cfg = AlgwasConfig::default();
        std::env::set_var("JX_ALGWAS_DEBUG", "1");
        std::env::set_var("JX_ALGWAS_STAGE1_MODE", "dense");
        let dense = fit_stage1_path_dense_msgps(&design, &proj, &y, &cfg, None).unwrap();
        std::env::set_var("JX_ALGWAS_STAGE1_MODE", "packed");
        let exact = fit_stage1_path_exact_packed(&design, &proj, &y, &cfg, None).unwrap();
        eprintln!(
            "mouse_hs1940 dense: selected={} lambda_best={:.12e} bic_best={:.12e} path_len={}",
            dense.selected_indices.len(),
            dense.lambda_best,
            dense.bic_best,
            dense.path.len()
        );
        eprintln!(
            "mouse_hs1940 exact: selected={} lambda_best={:.12e} bic_best={:.12e} path_len={}",
            exact.selected_indices.len(),
            exact.lambda_best,
            exact.bic_best,
            exact.path.len()
        );
    }
}
