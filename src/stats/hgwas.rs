//! Haploid / multi-parent ancestry GWAS (hGWAS) over streaming `.jhibd` archives.
//!
//! At each genomic bin `b`, local founder ancestry dosage `Z_b` (`n x P_b`, sparse)
//! is modeled as a random effect (Wei & Xu 2016 *Genetics* / Xiao et al. 2016 *New Phytologist*):
//!
//! ```text
//! y = X beta + g + Z_b u_b + eps,
//! g ~ N(0, sigma_g^2 K),   u_b ~ N(0, sigma_b^2 I),   eps ~ N(0, sigma_e^2 I)
//! ```
//!
//! Two exact likelihood-ratio test (LRT) modes are provided (no Wald test):
//! - **FVLMM** (`exact_lmm = false`): fixes the background polygenic variance ratio
//!   `lambda_g = sigma_g^2 / sigma_e^2` from the null model (`H_0: sigma_b^2 = 0`),
//!   precomputes the exact original-space projection `M_0 = U P_0 U^T` and `v_0 = U P_0 y_rot`
//!   once, and maintains `(r_b = Z_b^T v_0, A_b = Z_b^T M_0 Z_b, B_b = M_0 Z_b)` via
//!   incremental rank-1 sparse-diff updates across adjacent Union-of-Breakpoints bins.
//! - **LMM** (`exact_lmm = true`): re-estimates all three variance components
//!   `(sigma_g^2, sigma_b^2, sigma_e^2)` per bin via profile REML over `(lambda_g, tau)`,
//!   maintaining rotated `Z_rot = U^T Z` incrementally across adjacent bins with a
//!   precomputed `BackgroundState` grid and tight-bracket Brent refinement.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::blas::{
    cblas_dgemm_dispatch, BlasThreadGuard, CblasInt, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use crate::brent::brent_minimize_with_init;
use crate::eigh::symmetric_eigh_f64_row_major;
use crate::haploid::ancestry::ChromosomeAncestryBinsSparse;
use crate::haploid::sparse_io::{read_sparse_archive, HaploidSparseArchive, HaploidSparsePayload};
use crate::haploid::types::HaploidIbdMode;
use crate::linalg::{
    chi2_sf_df1, cholesky_gram_inplace, cholesky_logdet, cholesky_solve_into,
    validate_fixed_effect_design,
};

const MIN_DIAG_FLOOR: f64 = 1e-12;
const MIN_RSS_FLOOR: f64 = 1e-30;
const TAU_LOG_LOW: f64 = -18.0;
const TAU_LOG_HIGH: f64 = 18.0;
const DEFAULT_BIN_BATCH_COLS: usize = 512;
/// Number of bins processed per independent incremental chunk (resets floating-point
/// drift periodically while keeping chunk-init overhead < 1% of scan time).
const CHUNK_BINS: usize = 1024;
/// Maximum `p x p` size handled by the zero-allocation inline Jacobi eigensolver.
const INLINE_JACOBI_MAX_P: usize = 64;

/// Output statistics for a single bin.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HgwasBinResult {
    pub(crate) n_parents: usize,
    pub(crate) sigma_g2: f64,
    pub(crate) sigma_b2: f64,
    pub(crate) sigma_e2: f64,
    pub(crate) h2_bin: f64,
    pub(crate) lrt: f64,
    pub(crate) plrt: f64,
}

/// Null-model REML quantities evaluated at a given `lambda_g = sigma_g^2 / sigma_e^2`.
#[derive(Clone, Debug)]
struct BackgroundState {
    lambda_g: f64,
    /// `w[i] = 1.0 / (lambda_g * s[i] + 1.0)`
    w: Vec<f64>,
    /// `log|D| = sum_i ln(lambda_g * s[i] + 1.0)`
    logdet_d: f64,
    /// `V_x = W X_rot L_x^{-T}` stored in `(q x n)` row-major so `V_x * z_col` is contiguous.
    vx_t: Vec<f64>,
    /// `log|X^T W X|`
    logdet_xwx: f64,
    /// `P_0 y = W y - W X (X^T W X)^{-1} X^T W y` (length `n`).
    p0_y: Vec<f64>,
    /// `RSS_0 = y^T P_0 y`.
    rss0: f64,
    /// Null REML log-likelihood at `tau = 0`.
    reml_ll0: f64,
    /// `sigma_e2_0 = RSS_0 / (n - q)`.
    sigma_e2_0: f64,
}

/// Exact 50:50 mixture p-value for boundary variance-component LRT (`H_0: sigma_b^2 = 0`).
#[inline]
pub(crate) fn variance_component_lrt_pvalue(lrt: f64) -> f64 {
    if !lrt.is_finite() || lrt <= 0.0 {
        return 1.0;
    }
    let p = 0.5 * chi2_sf_df1(lrt);
    if p.is_finite() {
        p.clamp(f64::MIN_POSITIVE, 1.0)
    } else {
        1.0
    }
}

fn load_multparent_archive(path: &str) -> Result<HaploidSparseArchive, String> {
    let archive = read_sparse_archive(Path::new(path))?;
    if archive.mode != HaploidIbdMode::MultiParentAncestry {
        return Err(format!(
            "hGWAS requires a MultiParentAncestry .jhibd archive, got {:?}",
            archive.mode
        ));
    }
    Ok(archive)
}

fn build_sample_remap(
    n_archive_samples: usize,
    sample_indices: Option<&[i64]>,
) -> Result<(Vec<i32>, usize), String> {
    if let Some(sel) = sample_indices {
        if sel.is_empty() {
            return Err("sample_indices must not be empty".to_string());
        }
        let mut remap = vec![-1i32; n_archive_samples];
        for (new_idx, &raw_idx) in sel.iter().enumerate() {
            if raw_idx < 0 || (raw_idx as usize) >= n_archive_samples {
                return Err(format!(
                    "sample_indices[{}]={} out of range 0..{}",
                    new_idx, raw_idx, n_archive_samples
                ));
            }
            let slot = &mut remap[raw_idx as usize];
            if *slot >= 0 {
                return Err(format!(
                    "duplicate sample index {} in sample_indices",
                    raw_idx
                ));
            }
            *slot = new_idx as i32;
        }
        Ok((remap, sel.len()))
    } else {
        let remap = (0..n_archive_samples).map(|i| i as i32).collect();
        Ok((remap, n_archive_samples))
    }
}

fn validate_incremental_sparse_bins(
    chrom_bins: &[ChromosomeAncestryBinsSparse],
    n_archive_samples: usize,
    p_total: usize,
) -> Result<(), String> {
    for cb in chrom_bins {
        let n_bins = cb.bin_starts_bp.len();
        if n_bins == 0 {
            continue;
        }
        if cb.bin_ptr.len() != n_bins + 1 {
            return Err(format!(
                "invalid .jhibd bin index: chromosome {} has {} bins but {} bin_ptr entries",
                cb.chrom_idx,
                n_bins,
                cb.bin_ptr.len()
            ));
        }
        if cb.sample_idx.len() != cb.parent_idx.len() || cb.sample_idx.len() != cb.dosage.len() {
            return Err(format!(
                "invalid .jhibd sparse columns on chromosome {}",
                cb.chrom_idx
            ));
        }
        if cb.n_target_samples > n_archive_samples {
            return Err(format!(
                "invalid .jhibd sample count: chromosome {} declares {}, archive has {}",
                cb.chrom_idx, cb.n_target_samples, n_archive_samples
            ));
        }
        if cb.bin_ptr[0] != 0 {
            return Err(format!(
                "invalid .jhibd bin index: chromosome {} does not start at entry 0",
                cb.chrom_idx
            ));
        }

        for b in 0..n_bins {
            let start = usize::try_from(cb.bin_ptr[b])
                .map_err(|_| format!("bin_ptr overflow on chromosome {}", cb.chrom_idx))?;
            let end = usize::try_from(cb.bin_ptr[b + 1])
                .map_err(|_| format!("bin_ptr overflow on chromosome {}", cb.chrom_idx))?;
            if start > end || end > cb.sample_idx.len() {
                return Err(format!(
                    "invalid .jhibd bin range on chromosome {} bin {}",
                    cb.chrom_idx, b
                ));
            }

            let mut previous: Option<(u32, u32)> = None;
            for pos in start..end {
                let sample = cb.sample_idx[pos];
                let parent = cb.parent_idx[pos];
                if sample as usize >= cb.n_target_samples || parent as usize >= p_total {
                    return Err(format!(
                        "out-of-range ancestry entry on chromosome {} bin {}",
                        cb.chrom_idx, b
                    ));
                }
                let dosage = cb.dosage[pos];
                if !dosage.is_finite() || dosage < 0.0 {
                    return Err(format!(
                        "invalid dosage on chromosome {} bin {}",
                        cb.chrom_idx, b
                    ));
                }
                let key = (sample, parent);
                if previous.is_some_and(|prev| key <= prev) {
                    return Err(format!(
                        "ancestry entries must be strictly sorted and unique on chromosome {} bin {}",
                        cb.chrom_idx, b
                    ));
                }
                previous = Some(key);
            }
        }
    }
    Ok(())
}

/// Evaluate background REML state at a fixed `lambda_g = sigma_g^2 / sigma_e^2 >= 0`.
fn evaluate_background_state(
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    q: usize,
    lambda_g: f64,
) -> Option<BackgroundState> {
    let df = (n.checked_sub(q)?) as f64;
    if df <= 0.0 || !lambda_g.is_finite() || lambda_g < 0.0 {
        return None;
    }

    let mut w = vec![0.0_f64; n];
    let mut logdet_d = 0.0_f64;
    for i in 0..n {
        let d = (lambda_g * s[i] + 1.0).max(MIN_DIAG_FLOOR);
        w[i] = 1.0 / d;
        logdet_d += d.ln();
    }

    let mut xwx = vec![0.0_f64; q * q];
    let mut xwy = vec![0.0_f64; q];
    let mut ywy = 0.0_f64;
    for i in 0..n {
        let wi = w[i];
        let yi = y_rot[i];
        ywy += wi * yi * yi;
        let xi = &x_rot[i * q..(i + 1) * q];
        for a in 0..q {
            let wxa = wi * xi[a];
            xwy[a] += wxa * yi;
            let row = a * q;
            for b in 0..=a {
                xwx[row + b] += wxa * xi[b];
            }
        }
    }
    for a in 0..q {
        for b in (a + 1)..q {
            xwx[a * q + b] = xwx[b * q + a];
        }
    }

    cholesky_gram_inplace(&mut xwx, q)?;
    let logdet_xwx = cholesky_logdet(&xwx, q);
    let mut beta = vec![0.0_f64; q];
    cholesky_solve_into(&xwx, q, &xwy, &mut beta);

    let quad_x: f64 = xwy.iter().zip(&beta).map(|(a, b)| a * b).sum();
    let rss0 = (ywy - quad_x).max(MIN_RSS_FLOOR);
    let sigma_e2_0 = (rss0 / df).max(MIN_RSS_FLOOR);

    let mut p0_y = vec![0.0_f64; n];
    // Precompute V_x^T = L_x^{-1} X_rot^T W in (q x n) row-major so that
    // (X_rot^T W Z_rot)^T (X_rot^T W X_rot)^{-1} (X_rot^T W Z_rot) = (V_x^T Z_rot)^T (V_x^T Z_rot).
    let mut vx_t = vec![0.0_f64; q * n];
    let mut tmp = vec![0.0_f64; q];
    for i in 0..n {
        let wi = w[i];
        let xi = &x_rot[i * q..(i + 1) * q];
        let mut xb = 0.0_f64;
        for a in 0..q {
            xb += xi[a] * beta[a];
            let mut sum = wi * xi[a];
            let l_row = a * q;
            for k in 0..a {
                sum -= xwx[l_row + k] * tmp[k];
            }
            let val = sum / xwx[l_row + a];
            tmp[a] = val;
            vx_t[a * n + i] = val;
        }
        p0_y[i] = wi * (y_rot[i] - xb);
    }

    let const_term = df * ((2.0 * std::f64::consts::PI).ln() + 1.0);
    let reml_ll0 = -0.5 * (const_term + df * sigma_e2_0.ln() + logdet_d + logdet_xwx);

    Some(BackgroundState {
        lambda_g,
        w,
        logdet_d,
        vx_t,
        logdet_xwx,
        p0_y,
        rss0,
        reml_ll0,
        sigma_e2_0,
    })
}

/// Optimize the null model (`tau = 0`) over `lambda_g` in `[10^low, 10^high]`
/// (plus boundary `lambda_g = 0`) to obtain the exact REML null log-likelihood.
fn fit_null_reml_background(
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    q: usize,
    log10_low: f64,
    log10_high: f64,
    init_lambda_g: Option<f64>,
) -> Result<BackgroundState, String> {
    let mut best = evaluate_background_state(s, x_rot, y_rot, n, q, 0.0)
        .ok_or_else(|| "failed to evaluate null REML at lambda_g=0".to_string())?;

    let mut best_log10 = log10_low;
    let n_grid = 25usize;
    for g in 0..=n_grid {
        let frac = (g as f64) / (n_grid as f64);
        let lg = log10_low + frac * (log10_high - log10_low);
        let lam = 10.0_f64.powf(lg);
        if let Some(st) = evaluate_background_state(s, x_rot, y_rot, n, q, lam) {
            if st.reml_ll0 > best.reml_ll0 {
                best = st;
                best_log10 = lg;
            }
        }
    }

    if let Some(init_lam) = init_lambda_g {
        if init_lam.is_finite() && init_lam > 0.0 {
            let lg = init_lam.log10().clamp(log10_low, log10_high);
            if let Some(st) = evaluate_background_state(s, x_rot, y_rot, n, q, 10.0_f64.powf(lg)) {
                if st.reml_ll0 > best.reml_ll0 {
                    best = st;
                    best_log10 = lg;
                }
            }
        }
    }

    let step = (log10_high - log10_low) / (n_grid as f64);
    let b_lo = (best_log10 - step).max(log10_low);
    let b_hi = (best_log10 + step).min(log10_high);
    let (opt_lg, _) = brent_minimize_with_init(
        |lg| {
            evaluate_background_state(s, x_rot, y_rot, n, q, 10.0_f64.powf(lg))
                .map(|st| -st.reml_ll0)
                .unwrap_or(f64::INFINITY)
        },
        b_lo,
        b_hi,
        1e-6,
        60,
        Some(best_log10),
    );
    if let Some(st) = evaluate_background_state(s, x_rot, y_rot, n, q, 10.0_f64.powf(opt_lg)) {
        if st.reml_ll0 > best.reml_ll0 {
            best = st;
        }
    }
    Ok(best)
}

/// Fast zero-allocation cyclic Jacobi eigensolver for small symmetric `p x p` matrices
/// (`p <= INLINE_JACOBI_MAX_P`). Overwrites `a` with diagonal eigenvalues and writes
/// eigenvectors as columns in row-major `v` (`v[row * p + col]`).
fn jacobi_sym_eigh_inplace(a: &mut [f64], v: &mut [f64], p: usize) {
    // Initialize V = I.
    for val in v.iter_mut() {
        *val = 0.0;
    }
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    if p <= 1 {
        return;
    }

    let mut diag_scale = 0.0_f64;
    for i in 0..p {
        let d = a[i * p + i].abs();
        if d > diag_scale {
            diag_scale = d;
        }
    }
    let eps = 1e-13 * diag_scale.max(1.0);
    let max_sweeps = 24usize;

    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0_f64;
        for row in 0..p {
            let r_off = row * p;
            for col in (row + 1)..p {
                let val = a[r_off + col].abs();
                if val > max_off {
                    max_off = val;
                }
            }
        }
        if max_off <= eps {
            break;
        }
        let thresh = 0.2 * max_off;
        for i in 0..p {
            for j in (i + 1)..p {
                let aij = a[i * p + j];
                if aij.abs() <= thresh && max_off > 10.0 * eps {
                    continue;
                }
                if aij.abs() <= 1e-30 {
                    continue;
                }
                let aii = a[i * p + i];
                let ajj = a[j * p + j];
                let tau = (ajj - aii) / (2.0 * aij);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[i * p + i] = aii - t * aij;
                a[j * p + j] = ajj + t * aij;
                a[i * p + j] = 0.0;
                a[j * p + i] = 0.0;

                for k in 0..p {
                    if k != i && k != j {
                        let aki = a[k * p + i];
                        let akj = a[k * p + j];
                        let new_ki = c * aki - s * akj;
                        let new_kj = s * aki + c * akj;
                        a[k * p + i] = new_ki;
                        a[i * p + k] = new_ki;
                        a[k * p + j] = new_kj;
                        a[j * p + k] = new_kj;
                    }
                }
                for k in 0..p {
                    let vki = v[k * p + i];
                    let vkj = v[k * p + j];
                    v[k * p + i] = c * vki - s * vkj;
                    v[k * p + j] = s * vki + c * vkj;
                }
            }
        }
    }
}

/// Scratch workspace reused across bins inside each Rayon worker thread to avoid
/// per-bin heap allocations.
struct TauSolverScratch {
    a_work: Vec<f64>,
    v_work: Vec<f64>,
    mu: Vec<f64>,
    u2: Vec<f64>,
    r_sub: Vec<f64>,
}

impl TauSolverScratch {
    fn with_capacity(p_max: usize) -> Self {
        Self {
            a_work: vec![0.0_f64; p_max * p_max],
            v_work: vec![0.0_f64; p_max * p_max],
            mu: Vec::with_capacity(p_max),
            u2: Vec::with_capacity(p_max),
            r_sub: Vec::with_capacity(p_max),
        }
    }
}

/// Exact 1D REML optimization over `tau = sigma_b^2 / sigma_e^2 >= 0` given the
/// `p x p` symmetric PSD matrix `A = Z^T P_0 Z` (stored in `scratch.a_work[..p*p]`)
/// and vector `r = Z^T P_0 y` (stored in `scratch.r_sub[..p]`).
///
/// Uses zero-allocation Jacobi eigendecomposition (for `p <= 64`), an exact
/// analytical certificate for `tau = 0`, and safeguarded analytical Newton-Raphson
/// on `d(ll)/d(ln tau) = 0` (converging in ~4-6 iterations to `< 1e-10`).
fn solve_tau_from_a_r(
    bg: &BackgroundState,
    df: f64,
    p: usize,
    scratch: &mut TauSolverScratch,
) -> (f64, f64, f64) {
    if p == 0 {
        return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
    }

    let mut trace_a = 0.0_f64;
    let mut r_norm2 = 0.0_f64;
    for c in 0..p {
        trace_a += scratch.a_work[c * p + c];
        r_norm2 += scratch.r_sub[c] * scratch.r_sub[c];
    }
    if !trace_a.is_finite() || trace_a <= 1e-14 || !r_norm2.is_finite() || r_norm2 <= 1e-30 {
        return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
    }

    scratch.mu.clear();
    scratch.u2.clear();
    let mu_tol = 1e-12 * trace_a.max(1.0);
    let mut rss_proj_max = 0.0_f64;

    if p <= INLINE_JACOBI_MAX_P {
        let a_slice = &mut scratch.a_work[..p * p];
        let v_slice = &mut scratch.v_work[..p * p];
        jacobi_sym_eigh_inplace(a_slice, v_slice, p);
        for j in 0..p {
            let mj = a_slice[j * p + j];
            if mj > mu_tol && mj.is_finite() {
                let mut uj = 0.0_f64;
                for c in 0..p {
                    uj += v_slice[c * p + j] * scratch.r_sub[c];
                }
                let uj2 = uj * uj;
                scratch.mu.push(mj);
                scratch.u2.push(uj2);
                rss_proj_max += uj2 / mj;
            }
        }
    } else {
        let (eigvals, eigvecs, _) = match symmetric_eigh_f64_row_major(&scratch.a_work[..p * p], p)
        {
            Ok(res) => res,
            Err(_) => return (bg.reml_ll0, 0.0, bg.sigma_e2_0),
        };
        for j in 0..p {
            let mj = eigvals[j];
            if mj > mu_tol && mj.is_finite() {
                let mut uj = 0.0_f64;
                for c in 0..p {
                    uj += eigvecs[c * p + j] * scratch.r_sub[c];
                }
                let uj2 = uj * uj;
                scratch.mu.push(mj);
                scratch.u2.push(uj2);
                rss_proj_max += uj2 / mj;
            }
        }
    }

    if scratch.mu.is_empty() {
        return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
    }
    let rss_perp = (bg.rss0 - rss_proj_max).max(0.0);

    // Exact mathematical certificate for boundary optimum tau = 0:
    // For every tau >= 0, RSS(tau) >= rss_perp, so
    //   d(ll)/d(tau) = 0.5 * sum_j [mu_j / (1 + tau*mu_j)] * [(df / RSS(tau)) * (u_j^2 / (mu_j * (1 + tau*mu_j))) - 1].
    // If max_j ((df / rss_perp) * (u_j^2 / mu_j)) <= 1.0, then d(ll)/d(tau) <= 0 for ALL tau >= 0!
    if rss_perp > MIN_RSS_FLOOR {
        let mut max_ratio = 0.0_f64;
        for (&mj, &uj2) in scratch.mu.iter().zip(&scratch.u2) {
            let ratio = uj2 / mj;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        if (df / rss_perp) * max_ratio <= 1.0 {
            return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
        }
    }

    let const_term = df * ((2.0 * std::f64::consts::PI).ln() + 1.0);
    let base_logdet = bg.logdet_d + bg.logdet_xwx;
    let mu_sl = &scratch.mu;
    let u2_sl = &scratch.u2;

    // Evaluate (ll, sigma_e2, d_ll / d(ln tau), d2_ll / d(ln tau)^2) in O(p).
    let eval_reml_derivs = |tau: f64| -> (f64, f64, f64, f64) {
        let mut logdet_m = 0.0_f64;
        let mut rss_var = 0.0_f64;
        let mut s1 = 0.0_f64; // sum_j t_j
        let mut s2 = 0.0_f64; // sum_j t_j * (1 - t_j)
        let mut r1 = 0.0_f64; // sum_j (u_j^2 * tau) / (1 + tau*mu_j)^2
        let mut r2 = 0.0_f64; // d(r1)/d(ln tau)
        for (&mj, &uj2) in mu_sl.iter().zip(u2_sl.iter()) {
            let tm = tau * mj;
            let inv_d = 1.0 / (1.0 + tm);
            logdet_m += (1.0 + tm).ln();
            let qj = uj2 * inv_d;
            rss_var += qj / mj;
            let tj = tm * inv_d; // tau*mu / (1 + tau*mu)
            s1 += tj;
            s2 += tj * inv_d;
            let r1j = tau * uj2 * inv_d * inv_d;
            r1 += r1j;
            r2 += r1j * (1.0 - 2.0 * tm * inv_d);
        }
        let rss = (rss_perp + rss_var).max(MIN_RSS_FLOOR);
        let sigma_e2 = (rss / df).max(MIN_RSS_FLOOR);
        let ll = -0.5 * (const_term + df * sigma_e2.ln() + base_logdet + logdet_m);
        // d(rss)/d(ln tau) = -r1; d2(rss)/d(ln tau)^2 = -r2.
        // g = d(ll)/d(ln tau) = 0.5 * (df * r1 / rss - s1).
        let inv_rss = 1.0 / rss;
        let g = 0.5 * (df * r1 * inv_rss - s1);
        let h = 0.5 * (df * (r2 * inv_rss + (r1 * inv_rss) * (r1 * inv_rss)) - s2);
        (ll, sigma_e2, g, h)
    };

    // Fast coarse scan over 9 log-spaced probes to locate global basin(s).
    const PROBES: [f64; 9] = [-14.0, -10.0, -6.0, -3.0, -1.0, 1.0, 3.0, 7.0, 13.0];
    let mut best_ll = bg.reml_ll0;
    let mut best_tau = 0.0_f64;
    let mut best_se2 = bg.sigma_e2_0;
    let mut best_idx: Option<usize> = None;
    let mut probe_g = [0.0_f64; 9];

    for (k, &lt) in PROBES.iter().enumerate() {
        let tau = lt.exp();
        let (ll, se2, g, _) = eval_reml_derivs(tau);
        probe_g[k] = g;
        if ll > best_ll {
            best_ll = ll;
            best_tau = tau;
            best_se2 = se2;
            best_idx = Some(k);
        }
    }

    let score_slope = (df / bg.rss0) * u2_sl.iter().sum::<f64>() - mu_sl.iter().sum::<f64>();
    if best_idx.is_none() && score_slope <= 0.0 {
        // Also check whether any probe had positive slope (interior peak between probes).
        let mut any_pos = false;
        for (k, &g) in probe_g.iter().enumerate() {
            if g > 0.0 {
                best_idx = Some(k);
                any_pos = true;
                break;
            }
        }
        if !any_pos {
            return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
        }
    }

    // Establish a tight bracket around the best probe (or positive-slope region).
    let (mut lo, mut hi, mut x) = if let Some(k) = best_idx {
        let lo_k = if k == 0 { TAU_LOG_LOW } else { PROBES[k - 1] };
        let hi_k = if k + 1 >= PROBES.len() {
            TAU_LOG_HIGH
        } else {
            PROBES[k + 1]
        };
        (lo_k, hi_k, PROBES[k])
    } else {
        // Positive score slope at tau=0 but all probes <= ll0 -> optimum is at very small tau.
        (TAU_LOG_LOW, PROBES[0], 0.5 * (TAU_LOG_LOW + PROBES[0]))
    };

    // Safeguarded Newton-Raphson on g(lt) = 0 within [lo, hi].
    for _iter in 0..16 {
        let tau = x.exp();
        let (ll, se2, g, h) = eval_reml_derivs(tau);
        if ll > best_ll {
            best_ll = ll;
            best_tau = tau;
            best_se2 = se2;
        }
        if g.abs() < 1e-9 || (hi - lo) < 1e-8 {
            break;
        }
        if g > 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let next_x = if h < -1e-12 {
            let cand = x - g / h;
            if cand > lo && cand < hi {
                cand
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };
        if (next_x - x).abs() < 1e-8 {
            let t_fin = next_x.exp();
            let (ll_f, se2_f, _, _) = eval_reml_derivs(t_fin);
            if ll_f > best_ll {
                best_ll = ll_f;
                best_tau = t_fin;
                best_se2 = se2_f;
            }
            break;
        }
        x = next_x;
    }

    (best_ll, best_tau, best_se2)
}

/// Given rotated active columns `z_rot_cols` (slice of `p` contiguous `n`-vectors)
/// and a fixed `BackgroundState`, form `A = Z^T P_0 Z` and `r = Z^T P_0 y` and
/// solve the exact 1D REML optimum over `tau >= 0`.
fn optimize_tau_for_background_cols(
    bg: &BackgroundState,
    z_rot_cols: &[&[f64]],
    n: usize,
    q: usize,
    scratch: &mut TauSolverScratch,
) -> (f64, f64, f64) {
    let p = z_rot_cols.len();
    if p == 0 {
        return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
    }
    let df = (n - q) as f64;
    let w = &bg.w;
    let p0_y = &bg.p0_y;

    scratch.r_sub.clear();
    scratch.r_sub.resize(p, 0.0);
    // Compute V = V_x^T Z_rot (q x p) where vx_t is (q x n).
    let mut v_qp = vec![0.0_f64; q * p];
    let mut wz_col = vec![0.0_f64; n];

    for c in 0..p {
        let zc = z_rot_cols[c];
        let mut rc = 0.0_f64;
        for i in 0..n {
            let zci = zc[i];
            rc += zci * p0_y[i];
            wz_col[i] = w[i] * zci;
        }
        scratch.r_sub[c] = rc;

        for a in 0..q {
            let vxa = &bg.vx_t[a * n..(a + 1) * n];
            let mut dot = 0.0_f64;
            for i in 0..n {
                dot += vxa[i] * zc[i];
            }
            v_qp[a * p + c] = dot;
        }

        let row = c * p;
        for d in 0..=c {
            let zd = z_rot_cols[d];
            let mut zwz_cd = 0.0_f64;
            for i in 0..n {
                zwz_cd += wz_col[i] * zd[i];
            }
            let mut sub = 0.0_f64;
            for a in 0..q {
                sub += v_qp[a * p + c] * v_qp[a * p + d];
            }
            let val = zwz_cd - sub;
            scratch.a_work[row + d] = val;
            scratch.a_work[d * p + c] = val;
        }
    }

    solve_tau_from_a_r(bg, df, p, scratch)
}

/// Build the exact original-space projection matrix `M_0 = U P_0 U^T` (`n x n`, row-major)
/// and vector `v_0 = U P_0 y_rot` (`n`) once before scanning all bins in FVLMM mode.
fn build_fvlmm_original_space_projection(
    u_mat: &[f64],
    bg: &BackgroundState,
    n: usize,
    q: usize,
    threads: usize,
) -> (Vec<f64>, Vec<f64>) {
    // 1. uw[i, j] = U[i, j] * w[j]  ->  M_0 = uw * U^T = U W U^T.
    let mut uw = vec![0.0_f64; n * n];
    for i in 0..n {
        let u_row = &u_mat[i * n..(i + 1) * n];
        let uw_row = &mut uw[i * n..(i + 1) * n];
        for j in 0..n {
            uw_row[j] = u_row[j] * bg.w[j];
        }
    }
    let mut m0 = vec![0.0_f64; n * n];
    {
        let _blas_guard = BlasThreadGuard::enter(threads.max(1));
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n as CblasInt,
                n as CblasInt,
                n as CblasInt,
                1.0,
                uw.as_ptr(),
                n as CblasInt,
                u_mat.as_ptr(),
                n as CblasInt,
                0.0,
                m0.as_mut_ptr(),
                n as CblasInt,
            );
        }
    }

    // 2. Subtract (U V_x)(U V_x)^T where bg.vx_t is V_x^T (q x n, row-major).
    //    Compute u_vx (n x q, row-major) = U (n x n) * (V_x^T)^T (n x q).
    if q > 0 {
        let mut u_vx = vec![0.0_f64; n * q];
        {
            let _blas_guard = BlasThreadGuard::enter(threads.max(1));
            unsafe {
                cblas_dgemm_dispatch(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_TRANS,
                    n as CblasInt,
                    q as CblasInt,
                    n as CblasInt,
                    1.0,
                    u_mat.as_ptr(),
                    n as CblasInt,
                    bg.vx_t.as_ptr(),
                    n as CblasInt,
                    0.0,
                    u_vx.as_mut_ptr(),
                    q as CblasInt,
                );
                cblas_dgemm_dispatch(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_TRANS,
                    n as CblasInt,
                    n as CblasInt,
                    q as CblasInt,
                    -1.0,
                    u_vx.as_ptr(),
                    q as CblasInt,
                    u_vx.as_ptr(),
                    q as CblasInt,
                    1.0,
                    m0.as_mut_ptr(),
                    n as CblasInt,
                );
            }
        }
    }

    // Enforce exact symmetry on M_0.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m0[i * n + j] + m0[j * n + i]);
            m0[i * n + j] = avg;
            m0[j * n + i] = avg;
        }
    }

    // 3. v_0 = U * p0_y.
    let mut v0 = vec![0.0_f64; n];
    for i in 0..n {
        let u_row = &u_mat[i * n..(i + 1) * n];
        let mut sum = 0.0_f64;
        for j in 0..n {
            sum += u_row[j] * bg.p0_y[j];
        }
        v0[i] = sum;
    }

    (m0, v0)
}

/// Sparse entry delta between two adjacent bins: `(sample_new, parent_raw, old_val, new_val)`.
#[inline]
fn diff_sorted_bin_slices(
    cb: &ChromosomeAncestryBinsSparse,
    prev_start: usize,
    prev_end: usize,
    cur_start: usize,
    cur_end: usize,
    sample_remap: &[i32],
    p_total: usize,
    out_deltas: &mut Vec<(usize, usize, f64, f64)>,
) {
    out_deltas.clear();
    let mut i = prev_start;
    let mut j = cur_start;

    while i < prev_end && j < cur_end {
        let key_i = (cb.sample_idx[i], cb.parent_idx[i]);
        let key_j = (cb.sample_idx[j], cb.parent_idx[j]);
        if key_i == key_j {
            let d_old = cb.dosage[i] as f64;
            let d_new = cb.dosage[j] as f64;
            if (d_new - d_old).abs() > 1e-12 {
                let s_raw = key_i.0 as usize;
                let p_raw = key_i.1 as usize;
                if s_raw < sample_remap.len() && p_raw < p_total {
                    let s_new = sample_remap[s_raw];
                    if s_new >= 0 && d_old.is_finite() && d_new.is_finite() {
                        out_deltas.push((s_new as usize, p_raw, d_old, d_new));
                    }
                }
            }
            i += 1;
            j += 1;
        } else if key_i < key_j {
            let s_raw = key_i.0 as usize;
            let p_raw = key_i.1 as usize;
            let d_old = cb.dosage[i] as f64;
            if s_raw < sample_remap.len() && p_raw < p_total && d_old != 0.0 && d_old.is_finite() {
                let s_new = sample_remap[s_raw];
                if s_new >= 0 {
                    out_deltas.push((s_new as usize, p_raw, d_old, 0.0));
                }
            }
            i += 1;
        } else {
            let s_raw = key_j.0 as usize;
            let p_raw = key_j.1 as usize;
            let d_new = cb.dosage[j] as f64;
            if s_raw < sample_remap.len() && p_raw < p_total && d_new != 0.0 && d_new.is_finite() {
                let s_new = sample_remap[s_raw];
                if s_new >= 0 {
                    out_deltas.push((s_new as usize, p_raw, 0.0, d_new));
                }
            }
            j += 1;
        }
    }

    while i < prev_end {
        let s_raw = cb.sample_idx[i] as usize;
        let p_raw = cb.parent_idx[i] as usize;
        let d_old = cb.dosage[i] as f64;
        if s_raw < sample_remap.len() && p_raw < p_total && d_old != 0.0 && d_old.is_finite() {
            let s_new = sample_remap[s_raw];
            if s_new >= 0 {
                out_deltas.push((s_new as usize, p_raw, d_old, 0.0));
            }
        }
        i += 1;
    }

    while j < cur_end {
        let s_raw = cb.sample_idx[j] as usize;
        let p_raw = cb.parent_idx[j] as usize;
        let d_new = cb.dosage[j] as f64;
        if s_raw < sample_remap.len() && p_raw < p_total && d_new != 0.0 && d_new.is_finite() {
            let s_new = sample_remap[s_raw];
            if s_new >= 0 {
                out_deltas.push((s_new as usize, p_raw, 0.0, d_new));
            }
        }
        j += 1;
    }
}

#[inline]
fn build_result_from_opt(
    p_active: usize,
    best_ll: f64,
    best_tau: f64,
    best_se2: f64,
    best_lg: f64,
    null_reml_ll0: f64,
    c_k: f64,
    c_b: f64,
) -> HgwasBinResult {
    let lrt = (2.0 * (best_ll - null_reml_ll0)).max(0.0);
    let plrt = variance_component_lrt_pvalue(lrt);
    let sigma_e2 = best_se2;
    let sigma_g2 = best_lg * sigma_e2;
    let sigma_b2 = best_tau * sigma_e2;
    let denom = sigma_g2 * c_k + sigma_b2 * c_b + sigma_e2;
    let h2_bin = if denom > 0.0 && denom.is_finite() {
        ((sigma_b2 * c_b) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    HgwasBinResult {
        n_parents: p_active,
        sigma_g2,
        sigma_b2,
        sigma_e2,
        h2_bin,
        lrt,
        plrt,
    }
}

/// Process a chunk of bins `[b_start, b_end)` in **FVLMM** mode using exact incremental
/// rank-1 updates of `(r, A, B)` in original sample space (`M_0 = U P_0 U^T`).
#[allow(clippy::too_many_arguments)]
fn scan_chunk_fvlmm(
    cb: &ChromosomeAncestryBinsSparse,
    b_start: usize,
    b_end: usize,
    sample_remap: &[i32],
    n: usize,
    q: usize,
    p_total: usize,
    c_k: f64,
    null_bg: &BackgroundState,
    m0: &[f64],
    v0: &[f64],
) -> Vec<HgwasBinResult> {
    let df = (n - q) as f64;
    let inv_n = 1.0 / (n as f64);
    let mut out = Vec::with_capacity(b_end - b_start);

    let mut r_full = vec![0.0_f64; p_total];
    let mut a_full = vec![0.0_f64; p_total * p_total];
    let mut b_full = vec![0.0_f64; p_total * n];
    let mut p_sum = vec![0.0_f64; p_total];
    let mut p_sq_sum = vec![0.0_f64; p_total];
    let mut p_nnz = vec![0u32; p_total];
    let mut active_parents = Vec::with_capacity(p_total);
    let mut deltas: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(64);
    let mut scratch = TauSolverScratch::with_capacity(p_total);

    let apply_delta = |s_new: usize,
                       p_raw: usize,
                       d_old: f64,
                       d_new: f64,
                       r_f: &mut [f64],
                       a_f: &mut [f64],
                       b_f: &mut [f64],
                       ps: &mut [f64],
                       psq: &mut [f64],
                       pnnz: &mut [u32]| {
        let delta = d_new - d_old;
        ps[p_raw] += delta;
        psq[p_raw] += d_new * d_new - d_old * d_old;
        if d_old == 0.0 && d_new != 0.0 {
            pnnz[p_raw] = pnnz[p_raw].saturating_add(1);
        } else if d_old != 0.0 && d_new == 0.0 {
            pnnz[p_raw] = pnnz[p_raw].saturating_sub(1);
        }

        r_f[p_raw] += delta * v0[s_new];
        let m0_row = &m0[s_new * n..(s_new + 1) * n];
        for p2 in 0..p_total {
            let d_bq = delta * b_f[p2 * n + s_new];
            a_f[p_raw * p_total + p2] += d_bq;
            a_f[p2 * p_total + p_raw] += d_bq;
        }
        a_f[p_raw * p_total + p_raw] += delta * delta * m0_row[s_new];
        let bp = &mut b_f[p_raw * n..(p_raw + 1) * n];
        for j in 0..n {
            bp[j] += delta * m0_row[j];
        }
    };

    let mut prev_start = 0usize;
    let mut prev_end = 0usize;
    let mut last_res = HgwasBinResult {
        n_parents: 0,
        sigma_g2: null_bg.lambda_g * null_bg.sigma_e2_0,
        sigma_b2: 0.0,
        sigma_e2: null_bg.sigma_e2_0,
        h2_bin: 0.0,
        lrt: 0.0,
        plrt: 1.0,
    };

    for b in b_start..b_end {
        let cur_start = cb.bin_ptr[b] as usize;
        let cur_end = cb.bin_ptr[b + 1] as usize;

        if b == b_start {
            // Initialize from scratch using only nonzero entries of bin `b_start`.
            for pos in cur_start..cur_end {
                let s_raw = cb.sample_idx[pos] as usize;
                let p_raw = cb.parent_idx[pos] as usize;
                if s_raw >= sample_remap.len() || p_raw >= p_total {
                    continue;
                }
                let s_new = sample_remap[s_raw];
                if s_new < 0 {
                    continue;
                }
                let d = cb.dosage[pos] as f64;
                if d == 0.0 || !d.is_finite() {
                    continue;
                }
                let si = s_new as usize;
                p_sum[p_raw] += d;
                p_sq_sum[p_raw] += d * d;
                p_nnz[p_raw] = p_nnz[p_raw].saturating_add(1);
                r_full[p_raw] += d * v0[si];
                let m0_row = &m0[si * n..(si + 1) * n];
                let bp = &mut b_full[p_raw * n..(p_raw + 1) * n];
                for j in 0..n {
                    bp[j] += d * m0_row[j];
                }
            }
            // Form initial A = Z^T B directly from nonzero entries of bin `b_start`.
            for pos in cur_start..cur_end {
                let s_raw = cb.sample_idx[pos] as usize;
                let p_raw = cb.parent_idx[pos] as usize;
                if s_raw >= sample_remap.len() || p_raw >= p_total {
                    continue;
                }
                let s_new = sample_remap[s_raw];
                if s_new < 0 {
                    continue;
                }
                let d = cb.dosage[pos] as f64;
                if d == 0.0 || !d.is_finite() {
                    continue;
                }
                let si = s_new as usize;
                let row = p_raw * p_total;
                for p2 in 0..p_total {
                    if p_nnz[p2] > 0 {
                        a_full[row + p2] += d * b_full[p2 * n + si];
                    }
                }
            }
            for p1 in 0..p_total {
                for p2 in (p1 + 1)..p_total {
                    let avg = 0.5 * (a_full[p1 * p_total + p2] + a_full[p2 * p_total + p1]);
                    a_full[p1 * p_total + p2] = avg;
                    a_full[p2 * p_total + p1] = avg;
                }
            }
        } else {
            diff_sorted_bin_slices(
                cb,
                prev_start,
                prev_end,
                cur_start,
                cur_end,
                sample_remap,
                p_total,
                &mut deltas,
            );
            if deltas.is_empty() {
                out.push(last_res);
                prev_start = cur_start;
                prev_end = cur_end;
                continue;
            }
            for &(s_new, p_raw, d_old, d_new) in &deltas {
                apply_delta(
                    s_new,
                    p_raw,
                    d_old,
                    d_new,
                    &mut r_full,
                    &mut a_full,
                    &mut b_full,
                    &mut p_sum,
                    &mut p_sq_sum,
                    &mut p_nnz,
                );
            }
        }

        prev_start = cur_start;
        prev_end = cur_end;

        active_parents.clear();
        let mut centered_trace_sum = 0.0_f64;
        for p in 0..p_total {
            if p_nnz[p] > 0 {
                active_parents.push(p);
                let s1 = p_sum[p];
                let mean = s1 * inv_n;
                centered_trace_sum += (p_sq_sum[p] - s1 * mean).max(0.0);
            }
        }
        let p_active = active_parents.len();
        let c_b = centered_trace_sum * inv_n;

        if p_active == 0 || c_b <= 1e-14 {
            last_res = HgwasBinResult {
                n_parents: p_active,
                sigma_g2: null_bg.lambda_g * null_bg.sigma_e2_0,
                sigma_b2: 0.0,
                sigma_e2: null_bg.sigma_e2_0,
                h2_bin: 0.0,
                lrt: 0.0,
                plrt: 1.0,
            };
            out.push(last_res);
            continue;
        }

        scratch.r_sub.clear();
        for (i_loc, &pi) in active_parents.iter().enumerate() {
            scratch.r_sub.push(r_full[pi]);
            let dst_row = i_loc * p_active;
            let src_row = pi * p_total;
            for (j_loc, &pj) in active_parents.iter().enumerate() {
                scratch.a_work[dst_row + j_loc] = a_full[src_row + pj];
            }
        }

        let (best_ll, best_tau, best_se2) = solve_tau_from_a_r(null_bg, df, p_active, &mut scratch);
        last_res = build_result_from_opt(
            p_active,
            best_ll,
            best_tau,
            best_se2,
            null_bg.lambda_g,
            null_bg.reml_ll0,
            c_k,
            c_b,
        );
        out.push(last_res);
    }

    out
}

/// Process a chunk of bins `[b_start, b_end)` in exact **LMM** mode using incremental
/// `Z_rot = U^T Z` updates, a precomputed `BackgroundState` grid, and tight-bracket Brent.
#[allow(clippy::too_many_arguments)]
fn scan_chunk_lmm(
    cb: &ChromosomeAncestryBinsSparse,
    b_start: usize,
    b_end: usize,
    sample_remap: &[i32],
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    u_mat: &[f64],
    n: usize,
    q: usize,
    p_total: usize,
    c_k: f64,
    null_bg: &BackgroundState,
    cached_grid: &[(f64, BackgroundState)],
    grid_step: f64,
    log10_low: f64,
    log10_high: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<HgwasBinResult> {
    let inv_n = 1.0 / (n as f64);
    let mut out = Vec::with_capacity(b_end - b_start);

    // `z_rot` stored as (p_total x n) row-major so each parent column is contiguous.
    let mut z_rot = vec![0.0_f64; p_total * n];
    let mut p_sum = vec![0.0_f64; p_total];
    let mut p_sq_sum = vec![0.0_f64; p_total];
    let mut p_nnz = vec![0u32; p_total];
    let mut active_parents = Vec::with_capacity(p_total);
    let mut deltas: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(64);
    let mut scratch = TauSolverScratch::with_capacity(p_total);

    let mut prev_start = 0usize;
    let mut prev_end = 0usize;
    let mut last_res = HgwasBinResult {
        n_parents: 0,
        sigma_g2: null_bg.lambda_g * null_bg.sigma_e2_0,
        sigma_b2: 0.0,
        sigma_e2: null_bg.sigma_e2_0,
        h2_bin: 0.0,
        lrt: 0.0,
        plrt: 1.0,
    };

    for b in b_start..b_end {
        let cur_start = cb.bin_ptr[b] as usize;
        let cur_end = cb.bin_ptr[b + 1] as usize;

        if b == b_start {
            for pos in cur_start..cur_end {
                let s_raw = cb.sample_idx[pos] as usize;
                let p_raw = cb.parent_idx[pos] as usize;
                if s_raw >= sample_remap.len() || p_raw >= p_total {
                    continue;
                }
                let s_new = sample_remap[s_raw];
                if s_new < 0 {
                    continue;
                }
                let d = cb.dosage[pos] as f64;
                if d == 0.0 || !d.is_finite() {
                    continue;
                }
                let si = s_new as usize;
                p_sum[p_raw] += d;
                p_sq_sum[p_raw] += d * d;
                p_nnz[p_raw] = p_nnz[p_raw].saturating_add(1);
                let u_row = &u_mat[si * n..(si + 1) * n];
                let zp = &mut z_rot[p_raw * n..(p_raw + 1) * n];
                for j in 0..n {
                    zp[j] += d * u_row[j];
                }
            }
        } else {
            diff_sorted_bin_slices(
                cb,
                prev_start,
                prev_end,
                cur_start,
                cur_end,
                sample_remap,
                p_total,
                &mut deltas,
            );
            if deltas.is_empty() {
                out.push(last_res);
                prev_start = cur_start;
                prev_end = cur_end;
                continue;
            }
            for &(s_new, p_raw, d_old, d_new) in &deltas {
                let delta = d_new - d_old;
                p_sum[p_raw] += delta;
                p_sq_sum[p_raw] += d_new * d_new - d_old * d_old;
                if d_old == 0.0 && d_new != 0.0 {
                    p_nnz[p_raw] = p_nnz[p_raw].saturating_add(1);
                } else if d_old != 0.0 && d_new == 0.0 {
                    p_nnz[p_raw] = p_nnz[p_raw].saturating_sub(1);
                }
                let u_row = &u_mat[s_new * n..(s_new + 1) * n];
                let zp = &mut z_rot[p_raw * n..(p_raw + 1) * n];
                for j in 0..n {
                    zp[j] += delta * u_row[j];
                }
            }
        }

        prev_start = cur_start;
        prev_end = cur_end;

        active_parents.clear();
        let mut centered_trace_sum = 0.0_f64;
        for p in 0..p_total {
            if p_nnz[p] > 0 {
                active_parents.push(p);
                let s1 = p_sum[p];
                let mean = s1 * inv_n;
                centered_trace_sum += (p_sq_sum[p] - s1 * mean).max(0.0);
            }
        }
        let p_active = active_parents.len();
        let c_b = centered_trace_sum * inv_n;

        if p_active == 0 || c_b <= 1e-14 {
            last_res = HgwasBinResult {
                n_parents: p_active,
                sigma_g2: null_bg.lambda_g * null_bg.sigma_e2_0,
                sigma_b2: 0.0,
                sigma_e2: null_bg.sigma_e2_0,
                h2_bin: 0.0,
                lrt: 0.0,
                plrt: 1.0,
            };
            out.push(last_res);
            continue;
        }

        let col_refs: Vec<&[f64]> = active_parents
            .iter()
            .map(|&p| &z_rot[p * n..(p + 1) * n])
            .collect();

        // Step 1: Evaluate at null_bg (exact FVLMM warm-start).
        let (mut best_ll, mut best_tau, mut best_se2) =
            optimize_tau_for_background_cols(null_bg, &col_refs, n, q, &mut scratch);
        let mut best_lg = null_bg.lambda_g;
        let mut best_log10 = if null_bg.lambda_g > 0.0 {
            null_bg.lambda_g.log10().clamp(log10_low, log10_high)
        } else {
            log10_low
        };

        // Step 2: Evaluate cached BackgroundState grid (0.0 + log-spaced grid) with zero
        // repeated Cholesky/background recomputation.
        for (lg_log10, bg_cand) in cached_grid {
            let (ll, tau, se2) =
                optimize_tau_for_background_cols(bg_cand, &col_refs, n, q, &mut scratch);
            if ll > best_ll {
                best_ll = ll;
                best_tau = tau;
                best_se2 = se2;
                best_lg = bg_cand.lambda_g;
                if lg_log10.is_finite() {
                    best_log10 = *lg_log10;
                }
            }
        }

        // Step 3: Refine lambda_g via tight-bracket Brent ONLY when a positive tau was found.
        if best_tau > 0.0 && best_ll > null_bg.reml_ll0 {
            let b_lo = (best_log10 - grid_step).max(log10_low);
            let b_hi = (best_log10 + grid_step).min(log10_high);
            let mut local_scratch = TauSolverScratch::with_capacity(p_active);
            let (opt_log10, _) = brent_minimize_with_init(
                |lg_log10| {
                    let lam = 10.0_f64.powf(lg_log10);
                    evaluate_background_state(s, x_rot, y_rot, n, q, lam)
                        .map(|bg| {
                            -optimize_tau_for_background_cols(
                                &bg,
                                &col_refs,
                                n,
                                q,
                                &mut local_scratch,
                            )
                            .0
                        })
                        .unwrap_or(f64::INFINITY)
                },
                b_lo,
                b_hi,
                tol.max(1e-3),
                max_iter.clamp(8, 20),
                Some(best_log10),
            );
            let opt_lam = 10.0_f64.powf(opt_log10);
            if let Some(bg) = evaluate_background_state(s, x_rot, y_rot, n, q, opt_lam) {
                let (ll, tau, se2) =
                    optimize_tau_for_background_cols(&bg, &col_refs, n, q, &mut scratch);
                if ll > best_ll {
                    best_ll = ll;
                    best_tau = tau;
                    best_se2 = se2;
                    best_lg = opt_lam;
                }
            }
        }

        last_res = build_result_from_opt(
            p_active,
            best_ll,
            best_tau,
            best_se2,
            best_lg,
            null_bg.reml_ll0,
            c_k,
            c_b,
        );
        out.push(last_res);
    }

    out
}

/// Extract a single bin's sparse ancestry entries into dense `n x p_active` column-major
/// columns inside a batch matrix `z_batch` (used by ancestry GRM construction).
fn populate_bin_into_batch(
    cb: &ChromosomeAncestryBinsSparse,
    bin_idx: usize,
    sample_remap: &[i32],
    n: usize,
    z_batch: &mut Vec<f64>,
    col_offset: usize,
    parent_to_local: &mut [i32],
    col_sums: &mut Vec<f64>,
    col_sq_sums: &mut Vec<f64>,
) -> (usize, f64) {
    let start = cb.bin_ptr[bin_idx] as usize;
    let end = cb.bin_ptr[bin_idx + 1] as usize;
    if start >= end {
        return (0, 0.0);
    }

    let mut active_parents: Vec<usize> = Vec::new();
    for pos in start..end {
        let s_raw = cb.sample_idx[pos] as usize;
        if s_raw >= sample_remap.len() || sample_remap[s_raw] < 0 {
            continue;
        }
        let d = cb.dosage[pos] as f64;
        if d == 0.0 || !d.is_finite() {
            continue;
        }
        let p_raw = cb.parent_idx[pos] as usize;
        if p_raw < parent_to_local.len() && parent_to_local[p_raw] < 0 {
            parent_to_local[p_raw] = active_parents.len() as i32;
            active_parents.push(p_raw);
        }
    }

    let p_active = active_parents.len();
    if p_active == 0 {
        return (0, 0.0);
    }

    let old_len = z_batch.len();
    z_batch.resize(old_len + p_active * n, 0.0);
    col_sums.clear();
    col_sums.resize(p_active, 0.0);
    col_sq_sums.clear();
    col_sq_sums.resize(p_active, 0.0);

    for pos in start..end {
        let s_raw = cb.sample_idx[pos] as usize;
        if s_raw >= sample_remap.len() {
            continue;
        }
        let s_new = sample_remap[s_raw];
        if s_new < 0 {
            continue;
        }
        let p_raw = cb.parent_idx[pos] as usize;
        if p_raw >= parent_to_local.len() {
            continue;
        }
        let local_p = parent_to_local[p_raw];
        if local_p < 0 {
            continue;
        }
        let d = cb.dosage[pos] as f64;
        if d == 0.0 || !d.is_finite() {
            continue;
        }
        let c = local_p as usize;
        let idx = (col_offset + c) * n + (s_new as usize);
        z_batch[idx] += d;
    }

    let inv_n = 1.0 / (n as f64);
    let mut centered_trace_sum = 0.0_f64;
    for c in 0..p_active {
        let col = &z_batch[(col_offset + c) * n..(col_offset + c + 1) * n];
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        for &v in col {
            s1 += v;
            s2 += v * v;
        }
        let mean = s1 * inv_n;
        let var_sum = (s2 - s1 * mean).max(0.0);
        centered_trace_sum += var_sum;
        col_sums[c] = mean;
        col_sq_sums[c] = var_sum;
    }

    for &p_raw in &active_parents {
        parent_to_local[p_raw] = -1;
    }

    (p_active, centered_trace_sum * inv_n)
}

/// Core streaming ancestry GRM builder from `HaploidSparseArchive`.
///
/// `method == 1`: centered ancestry GRM `K = Z_c Z_c^T / (tr(Z_c Z_c^T) / n)`.
/// `method == 2`: standardized ancestry GRM `K = Z_s Z_s^T / m_eff`.
pub(crate) fn build_ancestry_grm_from_archive(
    archive: &HaploidSparseArchive,
    sample_indices: Option<&[i64]>,
    method: i32,
    threads: usize,
) -> Result<(Vec<f64>, usize, usize), String> {
    if method != 1 && method != 2 {
        return Err(format!("GRM method must be 1 or 2, got {}", method));
    }
    let chrom_bins = match &archive.payload {
        HaploidSparsePayload::MultiParentAncestry { chrom_bins, .. } => chrom_bins,
        _ => {
            return Err("hGWAS GRM requires MultiParentAncestry payload".to_string());
        }
    };
    let (sample_remap, n) = build_sample_remap(archive.target_sample_ids.len(), sample_indices)?;
    if n == 0 {
        return Err("empty sample selection for ancestry GRM".to_string());
    }

    let n_parents = archive.reference_sample_ids.len().max(1);
    let mut parent_to_local = vec![-1i32; n_parents];
    let mut col_sums = Vec::with_capacity(n_parents);
    let mut col_sq_sums = Vec::with_capacity(n_parents);

    let mut grm = vec![0.0_f64; n * n];
    let mut z_batch: Vec<f64> = Vec::with_capacity(DEFAULT_BIN_BATCH_COLS * n);
    let mut batch_cols = 0usize;
    let mut eff_bins = 0usize;
    let mut eff_cols = 0usize;
    let inv_n = 1.0 / (n as f64);

    let flush_batch = |z_colmajor: &mut Vec<f64>, cols: &mut usize, grm_acc: &mut [f64]| {
        if *cols == 0 {
            return;
        }
        let c = *cols;
        let _blas_guard = BlasThreadGuard::enter(threads.max(1));
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                n as CblasInt,
                n as CblasInt,
                c as CblasInt,
                1.0,
                z_colmajor.as_ptr(),
                n as CblasInt,
                z_colmajor.as_ptr(),
                n as CblasInt,
                1.0,
                grm_acc.as_mut_ptr(),
                n as CblasInt,
            );
        }
        z_colmajor.clear();
        *cols = 0;
    };

    for cb in chrom_bins {
        let n_bins = cb.bin_starts_bp.len();
        for b in 0..n_bins {
            let col_off = batch_cols;
            let (p_active, _) = populate_bin_into_batch(
                cb,
                b,
                &sample_remap,
                n,
                &mut z_batch,
                col_off,
                &mut parent_to_local,
                &mut col_sums,
                &mut col_sq_sums,
            );
            if p_active == 0 {
                continue;
            }
            let mut kept_in_bin = 0usize;
            for c in 0..p_active {
                let mean = col_sums[c];
                let var = col_sq_sums[c] * inv_n;
                let col_slice = &mut z_batch[(col_off + c) * n..(col_off + c + 1) * n];
                if var <= 1e-12 {
                    col_slice.fill(0.0);
                    continue;
                }
                if method == 1 {
                    for v in col_slice.iter_mut() {
                        *v -= mean;
                    }
                } else {
                    let inv_sd = 1.0 / var.sqrt();
                    for v in col_slice.iter_mut() {
                        *v = (*v - mean) * inv_sd;
                    }
                }
                kept_in_bin += 1;
            }
            if kept_in_bin > 0 {
                eff_bins += 1;
                eff_cols += kept_in_bin;
            }
            batch_cols += p_active;
            if batch_cols >= DEFAULT_BIN_BATCH_COLS {
                flush_batch(&mut z_batch, &mut batch_cols, &mut grm);
            }
        }
    }
    flush_batch(&mut z_batch, &mut batch_cols, &mut grm);

    if eff_cols == 0 {
        return Err("no polymorphic ancestry bins found in .jhibd archive".to_string());
    }

    let denom = if method == 1 {
        let tr: f64 = (0..n).map(|i| grm[i * n + i]).sum();
        if !tr.is_finite() || tr <= 0.0 {
            return Err("ancestry GRM has non-positive trace".to_string());
        }
        tr * inv_n
    } else {
        eff_cols as f64
    };
    let scale = 1.0 / denom;
    for v in &mut grm {
        *v *= scale;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (grm[i * n + j] + grm[j * n + i]);
            grm[i * n + j] = avg;
            grm[j * n + i] = avg;
        }
    }

    Ok((grm, n, eff_bins))
}

/// Core streaming hGWAS scan over a `HaploidSparseArchive`, writing results to `out_tsv`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_hgwas_archive_to_tsv(
    archive: &HaploidSparseArchive,
    out_tsv: &str,
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    u_t: &[f64],
    n: usize,
    q: usize,
    sample_indices: Option<&[i64]>,
    exact_lmm: bool,
    lambda_g_null: Option<f64>,
    log10_low: f64,
    log10_high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<(usize, f64), String> {
    let chrom_bins = match &archive.payload {
        HaploidSparsePayload::MultiParentAncestry { chrom_bins, .. } => chrom_bins,
        _ => {
            return Err("hGWAS requires MultiParentAncestry payload in .jhibd".to_string());
        }
    };
    if s.len() != n || y_rot.len() != n || x_rot.len() != n * q || u_t.len() != n * n {
        return Err("dimension mismatch in rotated inputs for hGWAS".to_string());
    }
    validate_fixed_effect_design(x_rot, n, q).map_err(|e| e.to_string())?;
    let (sample_remap, n_sel) =
        build_sample_remap(archive.target_sample_ids.len(), sample_indices)?;
    if n_sel != n {
        return Err(format!(
            "selected sample count {} does not match phenotype length {}",
            n_sel, n
        ));
    }

    let c_k = (s.iter().sum::<f64>() / (n as f64)).max(1e-12);
    let null_bg = if let Some(lg_fixed) = lambda_g_null {
        if !exact_lmm && lg_fixed.is_finite() && lg_fixed >= 0.0 {
            evaluate_background_state(s, x_rot, y_rot, n, q, lg_fixed)
                .ok_or_else(|| "failed to evaluate fixed null background state".to_string())?
        } else {
            fit_null_reml_background(s, x_rot, y_rot, n, q, log10_low, log10_high, Some(lg_fixed))?
        }
    } else {
        fit_null_reml_background(s, x_rot, y_rot, n, q, log10_low, log10_high, None)?
    };

    // Transpose U^T -> U (n x n, row-major) once so U[i, :] is contiguous for both
    // FVLMM M_0 construction and LMM incremental Z_rot updates.
    let mut u_mat = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            u_mat[i * n + j] = u_t[j * n + i];
        }
    }

    let fvlmm_proj = if !exact_lmm {
        Some(build_fvlmm_original_space_projection(
            &u_mat, &null_bg, n, q, threads,
        ))
    } else {
        None
    };

    // Precompute cached BackgroundState grid for exact LMM once before scanning.
    let n_grid_lmm = 8usize;
    let grid_step = (log10_high - log10_low) / (n_grid_lmm as f64);
    let cached_grid: Vec<(f64, BackgroundState)> = if exact_lmm {
        let mut grid = Vec::with_capacity(n_grid_lmm + 2);
        if let Some(bg0) = evaluate_background_state(s, x_rot, y_rot, n, q, 0.0) {
            grid.push((f64::NEG_INFINITY, bg0));
        }
        for g in 0..=n_grid_lmm {
            let lg = log10_low + (g as f64) * grid_step;
            let lam = 10.0_f64.powf(lg);
            if let Some(bg) = evaluate_background_state(s, x_rot, y_rot, n, q, lam) {
                grid.push((lg, bg));
            }
        }
        grid
    } else {
        Vec::new()
    };

    let p_total = archive.reference_sample_ids.len().max(1);
    validate_incremental_sparse_bins(chrom_bins, archive.target_sample_ids.len(), p_total)?;

    let total_bins: usize = chrom_bins.iter().map(|cb| cb.bin_starts_bp.len()).sum();
    let file = File::create(out_tsv)
        .map_err(|e| format!("failed to create hGWAS TSV '{}': {}", out_tsv, e))?;
    let mut writer = BufWriter::with_capacity(512 * 1024, file);
    writeln!(
        writer,
        "chrom\tpos\tn_parents\tsigma_g2\tsigma_b2\tsigma_e2\th2_bin\tlrt\tplrt"
    )
    .map_err(|e| format!("failed to write hGWAS TSV header: {}", e))?;

    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| format!("failed to create rayon thread pool: {}", e))?,
        )
    } else {
        None
    };

    // Build flat chunk descriptors `(chrom_vec_idx, b_start, b_end)`.
    let mut chunks: Vec<(usize, usize, usize)> = Vec::new();
    for (ci, cb) in chrom_bins.iter().enumerate() {
        let n_bins = cb.bin_starts_bp.len();
        let mut b = 0usize;
        while b < n_bins {
            let b_end = (b + CHUNK_BINS).min(n_bins);
            chunks.push((ci, b, b_end));
            b = b_end;
        }
    }

    // Process waves of parallel chunks so memory stays bounded and progress updates smoothly.
    let wave_size = (threads.max(1) * 4).clamp(4, 64);
    let mut bins_done = 0usize;

    for wave in chunks.chunks(wave_size) {
        let run_wave = || -> Vec<Vec<HgwasBinResult>> {
            wave.par_iter()
                .map(|&(ci, b_start, b_end)| {
                    let cb = &chrom_bins[ci];
                    if let Some((ref m0, ref v0)) = fvlmm_proj {
                        scan_chunk_fvlmm(
                            cb,
                            b_start,
                            b_end,
                            &sample_remap,
                            n,
                            q,
                            p_total,
                            c_k,
                            &null_bg,
                            m0,
                            v0,
                        )
                    } else {
                        scan_chunk_lmm(
                            cb,
                            b_start,
                            b_end,
                            &sample_remap,
                            s,
                            x_rot,
                            y_rot,
                            &u_mat,
                            n,
                            q,
                            p_total,
                            c_k,
                            &null_bg,
                            &cached_grid,
                            grid_step,
                            log10_low,
                            log10_high,
                            max_iter,
                            tol,
                        )
                    }
                })
                .collect()
        };

        let wave_results = if let Some(tp) = pool.as_ref() {
            tp.install(run_wave)
        } else {
            run_wave()
        };

        for (&(ci, b_start, b_end), chunk_res) in wave.iter().zip(&wave_results) {
            let cb = &chrom_bins[ci];
            let chrom_name = archive
                .chrom_names
                .get(cb.chrom_idx as usize)
                .map(|s| s.as_str())
                .unwrap_or("chr?");
            for (offset, res) in chunk_res.iter().enumerate() {
                let b = b_start + offset;
                let start_bp = cb.bin_starts_bp[b];
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6}\t{:.6}\t{:.6e}",
                    chrom_name,
                    start_bp,
                    res.n_parents,
                    res.sigma_g2,
                    res.sigma_b2,
                    res.sigma_e2,
                    res.h2_bin,
                    res.lrt,
                    res.plrt
                )
                .map_err(|e| format!("failed to write hGWAS TSV row: {}", e))?;
            }
            bins_done += b_end - b_start;
        }
        on_progress(bins_done, total_bins);
    }

    writer
        .flush()
        .map_err(|e| format!("failed to flush hGWAS TSV: {}", e))?;

    let null_pve = {
        let sg2 = null_bg.lambda_g * null_bg.sigma_e2_0;
        let denom = sg2 * c_k + null_bg.sigma_e2_0;
        if denom > 0.0 {
            (sg2 * c_k / denom).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    Ok((bins_done, null_pve))
}

// ============================================================
// PyO3 bindings
// ============================================================

/// Inspect metadata of a `.jhibd` archive for hGWAS CLI setup.
#[pyfunction]
pub fn hgwas_inspect_jhibd(py: Python<'_>, jhibd_path: String) -> PyResult<Py<PyAny>> {
    let archive = py
        .detach(move || load_multparent_archive(&jhibd_path))
        .map_err(PyValueError::new_err)?;
    let n_bins: usize = match &archive.payload {
        HaploidSparsePayload::MultiParentAncestry { chrom_bins, .. } => {
            chrom_bins.iter().map(|cb| cb.bin_starts_bp.len()).sum()
        }
        _ => 0,
    };
    let dict = PyDict::new(py);
    dict.set_item("sample_ids", archive.target_sample_ids)?;
    dict.set_item("parent_ids", archive.reference_sample_ids)?;
    dict.set_item("chrom_names", archive.chrom_names)?;
    dict.set_item("n_bins", n_bins)?;
    dict.set_item("version", archive.version)?;
    Ok(dict.into_any().unbind())
}

/// Build an ancestry GRM directly from `.jhibd` founder dosage bins.
#[pyfunction]
#[pyo3(signature = (jhibd_path, sample_indices=None, method=1, threads=0))]
pub fn hgwas_grm_from_jhibd_f64<'py>(
    py: Python<'py>,
    jhibd_path: String,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: i32,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
    let sel_vec: Option<Vec<i64>> = sample_indices.map(|arr| arr.as_slice().unwrap().to_vec());
    let (grm, n, eff_bins) = py
        .detach(move || {
            let archive = load_multparent_archive(&jhibd_path)?;
            build_ancestry_grm_from_archive(&archive, sel_vec.as_deref(), method, threads)
        })
        .map_err(PyRuntimeError::new_err)?;
    let arr = PyArray2::<f64>::from_vec2(
        py,
        &(0..n)
            .map(|i| grm[i * n..(i + 1) * n].to_vec())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("failed to create GRM array: {}", e)))?;
    Ok((arr, eff_bins))
}

/// Streaming hGWAS scan over `.jhibd` into TSV using either FVLMM (`exact_lmm=False`)
/// or exact per-bin LMM (`exact_lmm=True`).
#[pyfunction]
#[pyo3(signature = (
    jhibd_path,
    out_tsv,
    s,
    x_rot,
    y_rot,
    u_t,
    sample_indices=None,
    exact_lmm=false,
    lambda_g_null=None,
    low=-5.0,
    high=5.0,
    max_iter=30,
    tol=1e-2,
    threads=0,
    progress_callback=None
))]
#[allow(clippy::too_many_arguments)]
pub fn hgwas_scan_jhibd_to_tsv_f64(
    py: Python<'_>,
    jhibd_path: String,
    out_tsv: String,
    s: PyReadonlyArray1<'_, f64>,
    x_rot: PyReadonlyArray2<'_, f64>,
    y_rot: PyReadonlyArray1<'_, f64>,
    u_t: PyReadonlyArray2<'_, f64>,
    sample_indices: Option<PyReadonlyArray1<'_, i64>>,
    exact_lmm: bool,
    lambda_g_null: Option<f64>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<(usize, f64)> {
    let s_vec = s
        .as_slice()
        .map_err(|_| PyValueError::new_err("s must be contiguous f64"))?
        .to_vec();
    let y_vec = y_rot
        .as_slice()
        .map_err(|_| PyValueError::new_err("y_rot must be contiguous f64"))?
        .to_vec();
    let x_shape = x_rot.shape();
    let n = s_vec.len();
    if y_vec.len() != n || x_shape[0] != n || x_shape[1] == 0 {
        return Err(PyValueError::new_err(
            "dimension mismatch between s, y_rot, and x_rot",
        ));
    }
    let q = x_shape[1];
    let x_vec = x_rot
        .as_slice()
        .map_err(|_| PyValueError::new_err("x_rot must be contiguous f64"))?
        .to_vec();
    let u_shape = u_t.shape();
    if u_shape[0] != n || u_shape[1] != n {
        return Err(PyValueError::new_err("u_t must have shape (n, n)"));
    }
    let u_vec = u_t
        .as_slice()
        .map_err(|_| PyValueError::new_err("u_t must be contiguous f64"))?
        .to_vec();
    let sel_vec: Option<Vec<i64>> = sample_indices
        .map(|arr| {
            arr.as_slice()
                .map(|sl| sl.to_vec())
                .map_err(|_| PyValueError::new_err("sample_indices must be contiguous i64"))
        })
        .transpose()?;

    let cb_ref = progress_callback.map(|cb| cb.clone_ref(py));
    let result = py
        .detach(move || {
            let archive = load_multparent_archive(&jhibd_path)?;
            scan_hgwas_archive_to_tsv(
                &archive,
                &out_tsv,
                &s_vec,
                &x_vec,
                &y_vec,
                &u_vec,
                n,
                q,
                sel_vec.as_deref(),
                exact_lmm,
                lambda_g_null,
                low,
                high,
                max_iter,
                tol,
                threads,
                |done, total| {
                    if let Some(cb) = cb_ref.as_ref() {
                        let _ = Python::attach(|py_inner| cb.call1(py_inner, (done, total)));
                    }
                },
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haploid::sparse_io::write_sparse_archive;
    use crate::linalg::{cholesky_inplace, cholesky_logdet, cholesky_solve_into};

    /// Brute-force dense n x n REML log-likelihood for `lambda_g * K + tau * Z Z^T + I`
    /// (matches Wei & Xu 2016 / Xiao et al. 2016 `nph.13814` random-model REML directly).
    fn dense_reml_ll(
        k: &[f64],
        z: &[f64],
        x: &[f64],
        y: &[f64],
        n: usize,
        q: usize,
        p: usize,
        lambda_g: f64,
        tau: f64,
    ) -> f64 {
        let mut h = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut zzt = 0.0_f64;
                for c in 0..p {
                    zzt += z[i * p + c] * z[j * p + c];
                }
                h[i * n + j] = lambda_g * k[i * n + j] + tau * zzt + if i == j { 1.0 } else { 0.0 };
            }
        }
        let mut l = h.clone();
        cholesky_inplace(&mut l, n).expect("dense H cholesky");
        let logdet_h = cholesky_logdet(&l, n);

        let mut hinv_y = vec![0.0_f64; n];
        cholesky_solve_into(&l, n, y, &mut hinv_y);

        let mut hinv_x = vec![0.0_f64; n * q];
        let mut col_in = vec![0.0_f64; n];
        let mut col_out = vec![0.0_f64; n];
        for a in 0..q {
            for i in 0..n {
                col_in[i] = x[i * q + a];
            }
            cholesky_solve_into(&l, n, &col_in, &mut col_out);
            for i in 0..n {
                hinv_x[i * q + a] = col_out[i];
            }
        }

        let mut xhx = vec![0.0_f64; q * q];
        let mut xhy = vec![0.0_f64; q];
        for i in 0..n {
            for a in 0..q {
                xhy[a] += x[i * q + a] * hinv_y[i];
                for b in 0..q {
                    xhx[a * q + b] += x[i * q + a] * hinv_x[i * q + b];
                }
            }
        }
        let mut lx = xhx.clone();
        cholesky_gram_inplace(&mut lx, q).expect("dense X^T H^{-1} X cholesky");
        let logdet_xhx = cholesky_logdet(&lx, q);

        let mut beta = vec![0.0_f64; q];
        cholesky_solve_into(&lx, q, &xhy, &mut beta);
        let yhy: f64 = y.iter().zip(&hinv_y).map(|(a, b)| a * b).sum();
        let quad: f64 = xhy.iter().zip(&beta).map(|(a, b)| a * b).sum();
        let rss = (yhy - quad).max(MIN_RSS_FLOOR);
        let df = (n - q) as f64;
        let se2 = rss / df;
        let const_term = df * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        -0.5 * (const_term + df * se2.ln() + logdet_h + logdet_xhx)
    }

    #[test]
    fn jacobi_small_eigh_matches_lapack() {
        let p = 64usize;
        let mut state: u64 = 0xfeed_face_cafe_0001;
        let mut next_f64 = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut m = vec![0.0_f64; p * p];
        for v in &mut m {
            *v = next_f64();
        }
        let mut a = vec![0.0_f64; p * p];
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0;
                for k in 0..p {
                    s += m[i * p + k] * m[j * p + k];
                }
                a[i * p + j] = s;
            }
        }
        let a_orig = a.clone();
        let (ref_evals, _, _) = symmetric_eigh_f64_row_major(&a, p).unwrap();
        let mut a_j = a.clone();
        let mut v_j = vec![0.0_f64; p * p];
        jacobi_sym_eigh_inplace(&mut a_j, &mut v_j, p);
        let mut j_evals: Vec<f64> = (0..p).map(|i| a_j[i * p + i]).collect();
        j_evals.sort_by(|x, y| x.total_cmp(y));
        for (e1, e2) in ref_evals.iter().zip(&j_evals) {
            assert!((e1 - e2).abs() < 1e-10, "eigenvalue mismatch: {e1} vs {e2}");
        }
        for j in 0..p {
            let eval = a_j[j * p + j];
            let mut residual = 0.0_f64;
            for i in 0..p {
                let mut av = 0.0_f64;
                for k in 0..p {
                    av += a_orig[i * p + k] * v_j[k * p + j];
                }
                residual += (av - eval * v_j[i * p + j]).powi(2);
            }
            assert!(
                residual < 1e-10,
                "eigenvector residual too large for column {j}: {residual}"
            );
        }
    }

    #[test]
    fn fvlmm_woodbury_reml_matches_dense_cholesky() {
        let n = 40usize;
        let q = 2usize;
        let p = 5usize;

        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next_f64 = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };

        let mut g_raw = vec![0.0_f64; n * 15];
        for v in &mut g_raw {
            *v = next_f64();
        }
        let mut k = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for m in 0..15 {
                    sum += g_raw[i * 15 + m] * g_raw[j * 15 + m];
                }
                k[i * n + j] = sum / 15.0;
            }
        }

        let mut x = vec![0.0_f64; n * q];
        for i in 0..n {
            x[i * q] = 1.0;
            x[i * q + 1] = next_f64();
        }

        let mut z = vec![0.0_f64; n * p];
        let parent_effects = [1.5, -1.0, 0.4, -0.8, 0.9];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let par = i % p;
            z[i * p + par] = 1.0;
            y[i] = 2.0 + 0.5 * x[i * q + 1] + parent_effects[par] + 0.3 * next_f64();
        }

        let (s, u, _) = symmetric_eigh_f64_row_major(&k, n).expect("eigh K");
        let mut u_t = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                u_t[i * n + j] = u[j * n + i];
            }
        }
        let mut x_rot = vec![0.0_f64; n * q];
        let mut y_rot = vec![0.0_f64; n];
        let mut z_rot_cols = vec![vec![0.0_f64; n]; p];
        for i in 0..n {
            for j in 0..n {
                let uij = u_t[i * n + j];
                y_rot[i] += uij * y[j];
                for a in 0..q {
                    x_rot[i * q + a] += uij * x[j * q + a];
                }
                for c in 0..p {
                    z_rot_cols[c][i] += uij * z[j * p + c];
                }
            }
        }

        let lambda_g = 1.25_f64;
        let bg = evaluate_background_state(&s, &x_rot, &y_rot, n, q, lambda_g).unwrap();
        let dense_ll0 = dense_reml_ll(&k, &z, &x, &y, n, q, p, lambda_g, 0.0);
        assert!(
            (bg.reml_ll0 - dense_ll0).abs() < 1e-9,
            "null REML mismatch: {} vs {}",
            bg.reml_ll0,
            dense_ll0
        );

        let col_refs: Vec<&[f64]> = z_rot_cols.iter().map(|v| v.as_slice()).collect();
        let mut scratch = TauSolverScratch::with_capacity(p);
        let (woodbury_ll1, opt_tau, _) =
            optimize_tau_for_background_cols(&bg, &col_refs, n, q, &mut scratch);
        let dense_ll1 = dense_reml_ll(&k, &z, &x, &y, n, q, p, lambda_g, opt_tau);
        assert!(opt_tau > 0.1, "expected positive tau for causal bin");
        assert!(
            (woodbury_ll1 - dense_ll1).abs() < 1e-8,
            "alt REML mismatch: {} vs {}",
            woodbury_ll1,
            dense_ll1
        );
    }

    #[test]
    fn incremental_fvlmm_and_lmm_match_dense_reml_across_multi_bin_transitions() {
        let n = 48usize;
        let q = 2usize;
        let p_total = 6usize;
        let n_bins = 16usize;

        let mut state: u64 = 0x9876_5432_10fe_dcba;
        fn lcg_u64(state: &mut u64) -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *state
        }
        fn lcg_f64(state: &mut u64) -> f64 {
            ((lcg_u64(state) >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }

        // Construct K, X, y.
        let mut g_raw = vec![0.0_f64; n * 20];
        for v in &mut g_raw {
            *v = lcg_f64(&mut state);
        }
        let mut k = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for m in 0..20 {
                    sum += g_raw[i * 20 + m] * g_raw[j * 20 + m];
                }
                k[i * n + j] = sum / 20.0;
            }
        }
        let mut x = vec![0.0_f64; n * q];
        for i in 0..n {
            x[i * q] = 1.0;
            x[i * q + 1] = lcg_f64(&mut state);
        }

        // Build incremental Union-of-Breakpoints bins where 0-4 samples switch per bin.
        let mut cur_par: Vec<usize> = (0..n).map(|i| i % p_total).collect();
        let mut bin_starts_bp = Vec::new();
        let mut bin_ends_bp = Vec::new();
        let mut bin_starts_cm = Vec::new();
        let mut bin_ends_cm = Vec::new();
        let mut bin_ptr = vec![0u64; n_bins + 1];
        let mut sample_idx = Vec::new();
        let mut parent_idx = Vec::new();
        let mut dosage = Vec::new();
        let mut z_per_bin: Vec<Vec<f64>> = Vec::new();

        for b in 0..n_bins {
            if b > 0 && b != 5 {
                // Switch 3 samples on non-identical bins; keep bin 5 identical to bin 4.
                for _ in 0..3 {
                    let si = (lcg_u64(&mut state) as usize) % n;
                    cur_par[si] =
                        (cur_par[si] + 1 + ((lcg_u64(&mut state) as usize) % (p_total - 1)))
                            % p_total;
                }
            }
            bin_starts_bp.push((b as u64) * 10_000 + 1);
            bin_ends_bp.push(((b + 1) as u64) * 10_000);
            bin_starts_cm.push(b as f32 * 0.1);
            bin_ends_cm.push((b + 1) as f32 * 0.1);
            let mut z_dense = vec![0.0_f64; n * p_total];
            for i in 0..n {
                let d = if i % 4 == 0 { 0.5_f32 } else { 1.0_f32 };
                sample_idx.push(i as u32);
                parent_idx.push(cur_par[i] as u32);
                dosage.push(d);
                z_dense[i * p_total + cur_par[i]] = d as f64;
            }
            bin_ptr[b + 1] = sample_idx.len() as u64;
            z_per_bin.push(z_dense);
        }

        let effects = [2.2, -1.8, 1.0, -1.2, 0.7, -0.9];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let par_causal = (0..p_total)
                .find(|&p| z_per_bin[8][i * p_total + p] > 0.0)
                .unwrap();
            y[i] = 3.0
                + 0.4 * x[i * q + 1]
                + effects[par_causal] * z_per_bin[8][i * p_total + par_causal]
                + 0.35 * lcg_f64(&mut state);
        }

        let (s, u, _) = symmetric_eigh_f64_row_major(&k, n).unwrap();
        let mut u_t = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                u_t[i * n + j] = u[j * n + i];
            }
        }
        let mut x_rot = vec![0.0_f64; n * q];
        let mut y_rot = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                let uij = u_t[i * n + j];
                y_rot[i] += uij * y[j];
                for a in 0..q {
                    x_rot[i * q + a] += uij * x[j * q + a];
                }
            }
        }

        let cb = ChromosomeAncestryBinsSparse {
            chrom_idx: 0,
            n_target_samples: n,
            n_parents: p_total,
            bin_starts_bp,
            bin_ends_bp,
            bin_starts_cm,
            bin_ends_cm,
            bin_ptr,
            sample_idx,
            parent_idx,
            dosage,
        };
        let remap: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let null_bg = fit_null_reml_background(&s, &x_rot, &y_rot, n, q, -5.0, 5.0, None).unwrap();
        let c_k = (s.iter().sum::<f64>() / (n as f64)).max(1e-12);
        let (m0, v0) = build_fvlmm_original_space_projection(&u, &null_bg, n, q, 1);

        let fv_res = scan_chunk_fvlmm(
            &cb, 0, n_bins, &remap, n, q, p_total, c_k, &null_bg, &m0, &v0,
        );
        assert!((fv_res[5].lrt - fv_res[4].lrt).abs() < 1e-12);

        // Verify every bin's FVLMM LRT against brute-force dense Cholesky REML.
        for b in 0..n_bins {
            let tau_hat = if fv_res[b].sigma_e2 > 0.0 {
                fv_res[b].sigma_b2 / fv_res[b].sigma_e2
            } else {
                0.0
            };
            let dense_ll0 = dense_reml_ll(
                &k,
                &z_per_bin[b],
                &x,
                &y,
                n,
                q,
                p_total,
                null_bg.lambda_g,
                0.0,
            );
            let dense_ll1 = dense_reml_ll(
                &k,
                &z_per_bin[b],
                &x,
                &y,
                n,
                q,
                p_total,
                null_bg.lambda_g,
                tau_hat,
            );
            let dense_lrt = (2.0 * (dense_ll1 - dense_ll0)).max(0.0);
            assert!(
                (fv_res[b].lrt - dense_lrt).abs() < 1e-7,
                "bin {b} FVLMM LRT mismatch: fast={} vs dense={}",
                fv_res[b].lrt,
                dense_lrt
            );
        }

        let n_grid_lmm = 8usize;
        let grid_step = 10.0_f64 / (n_grid_lmm as f64);
        let mut cached_grid = Vec::with_capacity(n_grid_lmm + 2);
        if let Some(bg0) = evaluate_background_state(&s, &x_rot, &y_rot, n, q, 0.0) {
            cached_grid.push((f64::NEG_INFINITY, bg0));
        }
        for g in 0..=n_grid_lmm {
            let lg = -5.0 + (g as f64) * grid_step;
            if let Some(bg) = evaluate_background_state(&s, &x_rot, &y_rot, n, q, 10.0_f64.powf(lg))
            {
                cached_grid.push((lg, bg));
            }
        }
        let lmm_res = scan_chunk_lmm(
            &cb,
            0,
            n_bins,
            &remap,
            &s,
            &x_rot,
            &y_rot,
            &u,
            n,
            q,
            p_total,
            c_k,
            &null_bg,
            &cached_grid,
            grid_step,
            -5.0,
            5.0,
            30,
            1e-2,
        );
        assert_eq!(lmm_res.len(), n_bins);

        // The chunked LMM path must agree with a cold-start LMM evaluation of each bin,
        // and both must reproduce the brute-force dense Cholesky likelihood at the
        // selected (lambda_g, tau).
        for b in 0..n_bins {
            let fresh = scan_chunk_lmm(
                &cb,
                b,
                b + 1,
                &remap,
                &s,
                &x_rot,
                &y_rot,
                &u,
                n,
                q,
                p_total,
                c_k,
                &null_bg,
                &cached_grid,
                grid_step,
                -5.0,
                5.0,
                30,
                1e-2,
            );
            assert_eq!(fresh.len(), 1);
            assert!(
                (lmm_res[b].lrt - fresh[0].lrt).abs() < 1e-7,
                "bin {b} incremental LMM mismatch: fast={} cold={}",
                lmm_res[b].lrt,
                fresh[0].lrt
            );
            assert!(
                (lmm_res[b].sigma_g2 - fresh[0].sigma_g2).abs() < 1e-7,
                "bin {b} incremental LMM sigma_g2 mismatch"
            );
            assert!(
                (lmm_res[b].sigma_b2 - fresh[0].sigma_b2).abs() < 1e-7,
                "bin {b} incremental LMM sigma_b2 mismatch"
            );
            assert!(
                (lmm_res[b].sigma_e2 - fresh[0].sigma_e2).abs() < 1e-7,
                "bin {b} incremental LMM sigma_e2 mismatch"
            );

            let lambda_hat = if lmm_res[b].sigma_e2 > 0.0 {
                lmm_res[b].sigma_g2 / lmm_res[b].sigma_e2
            } else {
                0.0
            };
            let tau_hat = if lmm_res[b].sigma_e2 > 0.0 {
                lmm_res[b].sigma_b2 / lmm_res[b].sigma_e2
            } else {
                0.0
            };
            let dense_ll0 = dense_reml_ll(
                &k,
                &z_per_bin[b],
                &x,
                &y,
                n,
                q,
                p_total,
                null_bg.lambda_g,
                0.0,
            );
            let dense_ll1 = dense_reml_ll(
                &k,
                &z_per_bin[b],
                &x,
                &y,
                n,
                q,
                p_total,
                lambda_hat,
                tau_hat,
            );
            let dense_lrt = (2.0 * (dense_ll1 - dense_ll0)).max(0.0);
            assert!(
                (lmm_res[b].lrt - dense_lrt).abs() < 1e-7,
                "bin {b} LMM LRT mismatch: fast={} vs dense={}",
                lmm_res[b].lrt,
                dense_lrt
            );
        }
    }

    #[test]
    fn end_to_end_jhibd_hgwas_detects_causal_bin_for_fvlmm_and_lmm() {
        let n = 60usize;
        let n_parents = 6usize;
        let n_bins = 20usize;
        let causal_bin = 7usize;

        let mut bin_starts_bp = Vec::with_capacity(n_bins);
        let mut bin_ends_bp = Vec::with_capacity(n_bins);
        let mut bin_starts_cm = Vec::with_capacity(n_bins);
        let mut bin_ends_cm = Vec::with_capacity(n_bins);
        let mut bin_ptr = vec![0u64; n_bins + 1];
        let mut sample_idx = Vec::new();
        let mut parent_idx = Vec::new();
        let mut dosage = Vec::new();

        let mut state: u64 = 0xdead_beef_cafe_babe;
        let mut next_u64 = || -> u64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };

        let mut causal_parent = vec![0usize; n];
        for b in 0..n_bins {
            bin_starts_bp.push((b as u64) * 100_000 + 1);
            bin_ends_bp.push(((b + 1) as u64) * 100_000);
            bin_starts_cm.push(b as f32);
            bin_ends_cm.push((b + 1) as f32);
            for i in 0..n {
                let p = (next_u64() as usize) % n_parents;
                if b == causal_bin {
                    causal_parent[i] = p;
                }
                sample_idx.push(i as u32);
                parent_idx.push(p as u32);
                dosage.push(1.0_f32);
            }
            bin_ptr[b + 1] = sample_idx.len() as u64;
        }

        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: (0..n).map(|i| format!("S{:03}", i)).collect(),
            reference_sample_ids: (0..n_parents).map(|p| format!("P{}", p)).collect(),
            chrom_names: vec!["chr1".to_string()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments: Vec::new(),
                chrom_bins: vec![ChromosomeAncestryBinsSparse {
                    chrom_idx: 0,
                    n_target_samples: n,
                    n_parents,
                    bin_starts_bp,
                    bin_ends_bp,
                    bin_starts_cm,
                    bin_ends_cm,
                    bin_ptr,
                    sample_idx,
                    parent_idx,
                    dosage,
                }],
            },
        };

        let tmp_dir = std::env::temp_dir();
        let jhibd_path = tmp_dir.join("test_hgwas_roundtrip.jhibd");
        let fvlmm_tsv = tmp_dir.join("test_hgwas_fvlmm.tsv");
        let lmm_tsv = tmp_dir.join("test_hgwas_lmm.tsv");
        write_sparse_archive(&jhibd_path, &archive).expect("write jhibd");
        let loaded = load_multparent_archive(jhibd_path.to_str().unwrap()).expect("read jhibd");

        let (grm, n_grm, eff_bins) =
            build_ancestry_grm_from_archive(&loaded, None, 1, 1).expect("build GRM");
        assert_eq!(n_grm, n);
        assert_eq!(eff_bins, n_bins);

        let (s, u, _) = symmetric_eigh_f64_row_major(&grm, n).expect("eigh");
        let mut u_t = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                u_t[i * n + j] = u[j * n + i];
            }
        }
        let effects = [2.5, -2.0, 1.5, -1.5, 0.5, -1.0];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let eps = ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64) * 0.4 - 0.2;
            y[i] = 10.0 + effects[causal_parent[i]] + eps;
        }
        let mut x_rot = vec![0.0_f64; n];
        let mut y_rot = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                let uij = u_t[i * n + j];
                x_rot[i] += uij;
                y_rot[i] += uij * y[j];
            }
        }

        let (n_fv, _) = scan_hgwas_archive_to_tsv(
            &loaded,
            fvlmm_tsv.to_str().unwrap(),
            &s,
            &x_rot,
            &y_rot,
            &u_t,
            n,
            1,
            None,
            false,
            None,
            -5.0,
            5.0,
            30,
            1e-2,
            1,
            |_, _| {},
        )
        .expect("fvlmm scan");
        assert_eq!(n_fv, n_bins);

        let (n_lmm, _) = scan_hgwas_archive_to_tsv(
            &loaded,
            lmm_tsv.to_str().unwrap(),
            &s,
            &x_rot,
            &y_rot,
            &u_t,
            n,
            1,
            None,
            true,
            None,
            -5.0,
            5.0,
            30,
            1e-2,
            1,
            |_, _| {},
        )
        .expect("lmm scan");
        assert_eq!(n_lmm, n_bins);

        for tsv in [&fvlmm_tsv, &lmm_tsv] {
            let text = std::fs::read_to_string(tsv).expect("read tsv");
            let mut lines = text.lines();
            let header = lines.next().expect("header");
            assert_eq!(
                header,
                "chrom\tpos\tn_parents\tsigma_g2\tsigma_b2\tsigma_e2\th2_bin\tlrt\tplrt"
            );
            assert!(!header.contains("pwald"));
            let pvals: Vec<f64> = lines
                .enumerate()
                .map(|(bin, ln)| {
                    let cols: Vec<&str> = ln.split('\t').collect();
                    assert_eq!(cols.len(), 9, "unexpected hGWAS row width");
                    assert_eq!(
                        cols[1],
                        (bin as u64 * 100_000 + 1).to_string(),
                        "pos must be the bin start"
                    );
                    cols[8].parse::<f64>().expect("parse plrt")
                })
                .collect();
            assert_eq!(pvals.len(), n_bins);
            assert!(
                pvals[causal_bin] < 1e-6,
                "causal bin p-value too large: {}",
                pvals[causal_bin]
            );
            let min_null = pvals
                .iter()
                .enumerate()
                .filter(|&(idx, _)| idx != causal_bin)
                .map(|(_, &p)| p)
                .fold(1.0_f64, f64::min);
            assert!(
                pvals[causal_bin] < min_null * 1e-4,
                "causal bin should dominate null bins"
            );
        }

        let _ = std::fs::remove_file(jhibd_path);
        let _ = std::fs::remove_file(fvlmm_tsv);
        let _ = std::fs::remove_file(lmm_tsv);
    }
}
