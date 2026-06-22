/*!
Packed FarmCPU implementation.

Notation
========

Let

  - `y` be the phenotype vector
  - `X0 = [1, C]` be the intercept plus user covariates
  - `S_t` be the pseudo-QTN index set at iteration `t`
  - `X_t = [X0, G_{S_t}]` be the background design at iteration `t`
  - `p_t(i)` be the stage1 FEM p-value for marker `i` conditional on `X_t`
  - `L_t(s, b)` be the lead set selected by window size `s` and lead count `b`
  - `tau` be the stage1 signal threshold supplied by the caller

Common stage1 pieces
====================

Both routes run a conditional FEM scan on `X_t` and score REM candidate lead sets
by minimizing the FarmCPU log-likelihood over `(window_bp, n_lead)` grid pairs.

Raw route: `-farmcpu`
=========================

This route follows the classic FEM/REM/SUPER chain.

  1. FEM: compute `p_t(i)` conditional on `X_t`
  2. REM: choose `L_t* = argmin_{s,b} REM(L_t(s,b) | X_t, y)`
  3. SUPER update: `S_{t+1} = SUPER(S_t ∪ L_t*)`
     with the classic rMVP-style redundancy rule `|r| >= 0.7`
  4. Current pseudo-QTN effect p-values are fed back into the update chain
  5. Stop when no marker passes `tau`, or `S_{t+1} == S_t`, or a 2-cycle is detected

Unified route: `-frgwas`
=========================

This route keeps the FEM background scan, adds the same REM scorer, and differs
from the raw route in how pseudo-QTNs are retained across stage1 and how the
final stage2 window scan is prepared.

  1. FEM: compute `p_t(i)` conditional on `X_t`
  2. REM: choose `L_t* = argmin_{s,b} REM(L_t(s,b) | X_t, y)`
  3. Record additional significant window representatives `R_t`
  4. Build the stage1 candidate union `U_t = S_t ∪ L_t* ∪ R_t`
  5. Strictly merge `U_t` within the current iteration using `r^2 >= 0.8`
     to obtain the next background set `S_{t+1}`
  6. Add every stage1-selected pseudo-QTN into a persistent seen-set and mask
     those rows out of later FEM candidate selection so they no longer compete
     for new lead slots
  7. Stop when no unmasked marker passes `tau`, or `S_{t+1} == S_t`, or a
     2-cycle is detected
  8. After stage1 converges, run one final relaxed merge with `r^2 >= 0.5`
     before the stage2 merged-window local re-scan

Final scan
==========

After stage1 converges, both routes perform the packed association scan using the
final background design. The unified route additionally enables merged-window
local re-scans after the relaxed final merge. Pseudo-QTN rows are written for
both routes; when a unified-route pseudo-QTN falls inside a local re-scan
window, its output statistics come from that local conditional refit.
*/

use crate::assoc2tsv::resolve_assoc_tsv_metadata;
use crate::bedmath::{decode_plink_bed_hardcall, packed_row_missing_count_selected};
use crate::blas::{
    cblas_dgemm_dispatch, CblasInt, OpenBlasThreadGuard, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR,
};
use crate::brent::brent_minimize;
use crate::eigh::symmetric_eigh_f64_row_major;
use crate::linalg::{
    chisq_from_beta_se_and_optional_plrt, format_chisq_value, sanitize_assoc_pvalue,
};
use crate::stats_common::{get_cached_pool, parse_index_vec_i64, AsyncTsvWriter};
use nalgebra::DMatrix;
use numpy::PyArray1;
use numpy::PyArrayMethods;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::f64::consts::LN_10;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FarmcpuRoute {
    Unified,
    Raw,
}

impl FarmcpuRoute {
    #[inline]
    fn from_raw(raw: bool) -> Self {
        if raw {
            Self::Raw
        } else {
            Self::Unified
        }
    }

    #[inline]
    fn enable_local_window_merge(self) -> bool {
        matches!(self, Self::Unified)
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[derive(Clone, Debug)]
struct FarmcpuExactRemPrepared {
    n: usize,
    p: usize,
    rank_x: usize,
    x_flat: Vec<f64>,
    ixx: Vec<f64>,
    y_resid: Vec<f64>,
    y_resid_ss: f64,
}

fn resolve_assoc_chrom_pos_metadata(
    bed_prefix: Option<&str>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    row_indices: Option<&[usize]>,
    expected_len: usize,
) -> Result<(Vec<String>, Vec<i64>), String> {
    let all_empty = chrom.is_empty() && pos.is_empty();
    if !all_empty {
        if chrom.len() != expected_len || pos.len() != expected_len {
            return Err(format!(
                "TSV chrom/pos metadata length mismatch: rows={expected_len}, chrom={}, pos={}",
                chrom.len(),
                pos.len(),
            ));
        }
        return Ok((chrom, pos));
    }

    let prefix = bed_prefix
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "empty chrom/pos metadata requires non-empty bed_prefix".to_string())?;
    let bim_path = format!("{prefix}.bim");
    let select = match row_indices {
        Some(idx) if idx.is_empty() => return Ok((Vec::new(), Vec::new())),
        Some(idx) => Some(idx),
        None => None,
    };
    if let Some(idx) = select {
        if !idx.windows(2).all(|w| w[0] <= w[1]) {
            let (chrom2, pos2, _snp2, _allele02, _allele12) =
                crate::gfcore::read_bim_columns(prefix, Some(idx))?;
            let pos64 = pos2.into_iter().map(|v| v as i64).collect::<Vec<i64>>();
            if chrom2.len() != expected_len || pos64.len() != expected_len {
                return Err(format!(
                    "BIM chrom/pos metadata length mismatch: expected={expected_len}, chrom={}, pos={}",
                    chrom2.len(),
                    pos64.len(),
                ));
            }
            return Ok((chrom2, pos64));
        }
    }

    let file = File::open(&bim_path).map_err(|e| format!("{bim_path}: {e}"))?;
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut chrom_out = Vec::<String>::with_capacity(expected_len);
    let mut pos_out = Vec::<i64>::with_capacity(expected_len);
    let mut want_ptr = 0usize;
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("{bim_path}:{}: {}", line_no + 1, e))?;
        if let Some(idx) = select {
            if want_ptr >= idx.len() {
                break;
            }
            if line_no != idx[want_ptr] {
                continue;
            }
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 4 {
            return Err(format!(
                "{bim_path}:{}: malformed BIM row (need >=4 columns)",
                line_no + 1
            ));
        }
        let pos_val = toks[3].parse::<i64>().map_err(|_| {
            format!(
                "{bim_path}:{}: invalid POS column '{}'",
                line_no + 1,
                toks[3]
            )
        })?;
        chrom_out.push(toks[0].to_string());
        pos_out.push(pos_val);
        if select.is_some() {
            want_ptr += 1;
        }
    }

    if let Some(idx) = select {
        if want_ptr != idx.len() {
            return Err(format!(
                "BIM ended early: needed row {} but only resolved {} selected rows from {}",
                idx[want_ptr], want_ptr, bim_path
            ));
        }
    }
    if chrom_out.len() != expected_len || pos_out.len() != expected_len {
        return Err(format!(
            "BIM chrom/pos metadata length mismatch: expected={expected_len}, chrom={}, pos={}",
            chrom_out.len(),
            pos_out.len(),
        ));
    }
    Ok((chrom_out, pos_out))
}

#[inline]
pub(crate) fn decode_packed_rows_to_sample_major(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_indices: &[usize],
    packed_row_lookup: Option<&[usize]>,
    sample_idx: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
) -> Result<Vec<f64>, String> {
    if bytes_per_snp == 0 {
        return Err("invalid bytes_per_snp=0 in packed decode".to_string());
    }
    if packed_flat.len() % bytes_per_snp != 0 {
        return Err("packed length is not divisible by bytes_per_snp in packed decode".to_string());
    }
    let n = sample_idx.len();
    let k = row_indices.len();
    let packed_rows = packed_flat.len() / bytes_per_snp;
    let mut out = vec![0.0_f64; n * k];
    for (col, &local_row_idx) in row_indices.iter().enumerate() {
        if local_row_idx >= row_flip.len() || local_row_idx >= row_maf.len() {
            return Err(format!(
                "row metadata index out of bounds in packed decode: idx={}, row_flip={}, row_maf={}",
                local_row_idx,
                row_flip.len(),
                row_maf.len()
            ));
        }
        let packed_row_idx = if let Some(lookup) = packed_row_lookup {
            *lookup.get(local_row_idx).ok_or_else(|| {
                format!(
                    "packed row lookup index out of bounds in packed decode: idx={}, lookup_len={}",
                    local_row_idx,
                    lookup.len()
                )
            })?
        } else {
            local_row_idx
        };
        if packed_row_idx >= packed_rows {
            return Err(format!(
                "packed row index out of bounds in packed decode: idx={}, packed_rows={}",
                packed_row_idx, packed_rows
            ));
        }
        let row =
            &packed_flat[packed_row_idx * bytes_per_snp..(packed_row_idx + 1) * bytes_per_snp];
        let flip = row_flip[local_row_idx];
        let mean_g = 2.0_f64 * row_maf[local_row_idx] as f64;
        for (i, &sidx) in sample_idx.iter().enumerate() {
            let b = row[sidx >> 2];
            let code = (b >> ((sidx & 3) * 2)) & 0b11;
            let mut gv = match decode_plink_bed_hardcall(code) {
                Some(v) => v,
                None => mean_g,
            };
            if flip && code != 0b01 {
                gv = 2.0 - gv;
            }
            out[i * k + col] = gv;
        }
    }
    Ok(out)
}

pub(crate) fn select_lead_indices(
    sz: i64,
    n_lead: usize,
    pvalue: &[f64],
    pos: &[i64],
) -> Vec<usize> {
    let m = pvalue.len();
    if m == 0 || n_lead == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        let ba = pos[a].div_euclid(sz);
        let bb = pos[b].div_euclid(sz);
        match ba.cmp(&bb) {
            std::cmp::Ordering::Equal => pvalue[a].total_cmp(&pvalue[b]),
            other => other,
        }
    });

    let mut lead: Vec<usize> = Vec::new();
    let mut last_bin: Option<i64> = None;
    for &idx in order.iter() {
        let b = pos[idx].div_euclid(sz);
        if last_bin.map_or(true, |lb| lb != b) {
            lead.push(idx);
            last_bin = Some(b);
        }
    }

    lead.sort_by(|&a, &b| pvalue[a].total_cmp(&pvalue[b]));
    if lead.len() > n_lead {
        lead.truncate(n_lead);
    }
    lead.sort_unstable();
    lead
}

fn farmcpu_exact_reml_log10_bounds_from_legacy_delta(
    delta_exp_start: f64,
    delta_exp_end: f64,
) -> (f64, f64) {
    let mut low = delta_exp_start.min(delta_exp_end) / LN_10;
    let mut high = delta_exp_start.max(delta_exp_end) / LN_10;
    if !low.is_finite() || !high.is_finite() {
        low = -6.0;
        high = 6.0;
    }
    if low == high {
        low -= 1.0;
        high += 1.0;
    }
    (low, high)
}

#[inline]
fn farmcpu_exact_reml_cost_from_spectrum(
    lambda: f64,
    eigvals: &[f64],
    y_proj: &[f64],
    y_resid_ss: f64,
    n_eff: usize,
) -> f64 {
    if !(lambda.is_finite() && lambda > 0.0) {
        return f64::INFINITY;
    }
    let r = eigvals.len();
    if r != y_proj.len() || n_eff == 0 || n_eff < r {
        return f64::INFINITY;
    }
    let mut quad = 0.0_f64;
    let mut log_det = 0.0_f64;
    let mut y_proj_ss = 0.0_f64;
    for k in 0..r {
        let s = eigvals[k];
        let yk = y_proj[k];
        if !(s.is_finite() && s >= 0.0 && yk.is_finite()) {
            return f64::INFINITY;
        }
        let vk = s + lambda;
        if !(vk.is_finite() && vk > 0.0) {
            return f64::INFINITY;
        }
        quad += (yk * yk) / vk;
        log_det += vk.ln();
        y_proj_ss += yk * yk;
    }
    let null_df = n_eff.saturating_sub(r);
    let null_ss = (y_resid_ss - y_proj_ss).max(0.0_f64);
    if null_df > 0 {
        quad += null_ss / lambda;
        log_det += (null_df as f64) * lambda.ln();
    }
    if !(quad.is_finite() && quad > 0.0 && log_det.is_finite()) {
        return f64::INFINITY;
    }
    0.5_f64 * ((n_eff as f64) * quad.ln() + log_det)
}

fn prepare_farmcpu_exact_rem(
    y: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
) -> Result<FarmcpuExactRemPrepared, String> {
    if n == 0 || p == 0 {
        return Err("invalid shape in ll score: empty y/X".to_string());
    }
    if y.len() != n {
        return Err("invalid y length in ll score".to_string());
    }
    if x.len() != n * p {
        return Err("invalid X length in ll score".to_string());
    }
    let (ixx, rank_x) = pinv_for_row_major_x_with_rank(x, n, p)?;
    if rank_x == 0 || rank_x >= n {
        return Err(format!(
            "invalid fixed-effect rank for exact FarmCPU REML: rank_x={rank_x}, n={n}"
        ));
    }
    let mut xty = vec![0.0_f64; p];
    for i in 0..n {
        let yi = y[i];
        let row = &x[i * p..(i + 1) * p];
        for a in 0..p {
            xty[a] += row[a] * yi;
        }
    }
    let mut beta_y = vec![0.0_f64; p];
    matvec_into(&ixx, p, &xty, &mut beta_y);
    let mut y_resid = vec![0.0_f64; n];
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        y_resid[i] = y[i] - dot(row, &beta_y);
    }
    let y_resid_ss = y_resid.iter().map(|v| v * v).sum::<f64>();
    Ok(FarmcpuExactRemPrepared {
        n,
        p,
        rank_x,
        x_flat: x.to_vec(),
        ixx,
        y_resid,
        y_resid_ss,
    })
}

fn farmcpu_ll_score_from_sample_major_prepared(
    prepared: &FarmcpuExactRemPrepared,
    snp_pool_sample_major: &[f64],
    k: usize,
    delta_exp_start: f64,
    delta_exp_end: f64,
    reml_tol: f64,
    reml_max_iter: usize,
    eig_eps: f64,
) -> Result<f64, String> {
    if k == 0 {
        return Err("FarmCPU exact REML scorer received zero candidate markers".to_string());
    }
    if snp_pool_sample_major.len() != prepared.n * k {
        return Err("invalid snp_pool length in ll score".to_string());
    }
    if !(reml_tol.is_finite() && reml_tol > 0.0) {
        return Err("exact FarmCPU REML requires finite reml_tol > 0".to_string());
    }
    if reml_max_iter == 0 {
        return Err("exact FarmCPU REML requires reml_max_iter > 0".to_string());
    }
    let n_eff = prepared.n.saturating_sub(prepared.rank_x);
    if n_eff == 0 {
        return Err("exact FarmCPU REML has zero effective degrees of freedom".to_string());
    }

    let mut xtg = vec![0.0_f64; prepared.p * k];
    for i in 0..prepared.n {
        let x_row = &prepared.x_flat[i * prepared.p..(i + 1) * prepared.p];
        let g_row = &snp_pool_sample_major[i * k..(i + 1) * k];
        for a in 0..prepared.p {
            let xa = x_row[a];
            for j in 0..k {
                xtg[a * k + j] += xa * g_row[j];
            }
        }
    }

    let mut x_proj_g = vec![0.0_f64; prepared.p * k];
    for a in 0..prepared.p {
        let ixx_row = &prepared.ixx[a * prepared.p..(a + 1) * prepared.p];
        for j in 0..k {
            let mut acc = 0.0_f64;
            for b in 0..prepared.p {
                acc += ixx_row[b] * xtg[b * k + j];
            }
            x_proj_g[a * k + j] = acc;
        }
    }

    let mut g_resid = vec![0.0_f64; prepared.n * k];
    for i in 0..prepared.n {
        let x_row = &prepared.x_flat[i * prepared.p..(i + 1) * prepared.p];
        let g_row = &snp_pool_sample_major[i * k..(i + 1) * k];
        let out_row = &mut g_resid[i * k..(i + 1) * k];
        for j in 0..k {
            let mut fit = 0.0_f64;
            for a in 0..prepared.p {
                fit += x_row[a] * x_proj_g[a * k + j];
            }
            out_row[j] = g_row[j] - fit;
        }
    }

    let mut a_star = vec![0.0_f64; k * k];
    let mut z = vec![0.0_f64; k];
    for i in 0..prepared.n {
        let g_row = &g_resid[i * k..(i + 1) * k];
        let yr = prepared.y_resid[i];
        for a in 0..k {
            let ga = g_row[a];
            z[a] += ga * yr;
            for b in a..k {
                a_star[a * k + b] += ga * g_row[b];
            }
        }
    }
    for a in 0..k {
        for b in 0..a {
            a_star[a * k + b] = a_star[b * k + a];
        }
    }

    // Stage1 evaluates several (window, lead-count) combinations in parallel,
    // so keep each exact eigh single-threaded to avoid nested BLAS oversubscription.
    let _blas_guard = OpenBlasThreadGuard::enter(1);
    let (evals_all, evecs_all, _eig_backend) = symmetric_eigh_f64_row_major(&a_star, k)?;
    if evals_all.is_empty() {
        return Err("exact FarmCPU REML eigendecomposition returned empty spectrum".to_string());
    }
    let max_eval = evals_all.last().copied().unwrap_or(0.0_f64).max(0.0_f64);
    let tol = eig_eps
        .max(1e-12_f64)
        .max(f64::EPSILON * max_eval.max(1.0_f64) * (k.max(1) as f64));
    let mut keep_start = evals_all
        .iter()
        .position(|&v| v > tol)
        .ok_or_else(|| "exact FarmCPU REML found no positive spectrum".to_string())?;
    let rank_pos = k.saturating_sub(keep_start);
    if rank_pos > n_eff {
        keep_start = k.saturating_sub(n_eff);
    }
    let eigvals = &evals_all[keep_start..];
    let rank = eigvals.len();
    if rank == 0 || rank > n_eff {
        return Err(format!(
            "invalid exact FarmCPU REML rank after projection: rank={rank}, n_eff={n_eff}"
        ));
    }

    let mut coeff = vec![0.0_f64; rank];
    for comp in 0..rank {
        let eig_col = keep_start + comp;
        let mut acc = 0.0_f64;
        for row in 0..k {
            acc += evecs_all[row * k + eig_col] * z[row];
        }
        coeff[comp] = acc;
    }
    let mut y_proj = vec![0.0_f64; rank];
    for comp in 0..rank {
        y_proj[comp] = coeff[comp] / eigvals[comp].sqrt().max(1e-18_f64);
    }

    let (log10_low, log10_high) =
        farmcpu_exact_reml_log10_bounds_from_legacy_delta(delta_exp_start, delta_exp_end);
    let (best_log10, best_cost) = brent_minimize(
        |x0| {
            let lambda = 10.0_f64.powf(x0);
            farmcpu_exact_reml_cost_from_spectrum(
                lambda,
                eigvals,
                y_proj.as_slice(),
                prepared.y_resid_ss,
                n_eff,
            )
        },
        log10_low,
        log10_high,
        reml_tol,
        reml_max_iter,
    );
    let lambda_best = 10.0_f64.powf(best_log10);
    let final_cost = farmcpu_exact_reml_cost_from_spectrum(
        lambda_best,
        eigvals,
        y_proj.as_slice(),
        prepared.y_resid_ss,
        n_eff,
    );
    if !best_cost.is_finite() || !final_cost.is_finite() {
        return Err("exact FarmCPU REML failed to evaluate a finite optimum".to_string());
    }
    Ok(2.0_f64 * final_cost)
}

pub(crate) fn farmcpu_ll_score_from_sample_major(
    y: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
    snp_pool_sample_major: &[f64],
    k: usize,
    delta_exp_start: f64,
    delta_exp_end: f64,
    delta_step: f64,
    svd_eps: f64,
) -> Result<f64, String> {
    let _ = delta_step;
    let prepared = prepare_farmcpu_exact_rem(y, x, n, p)?;
    farmcpu_ll_score_from_sample_major_prepared(
        &prepared,
        snp_pool_sample_major,
        k,
        delta_exp_start,
        delta_exp_end,
        1e-4_f64,
        50usize,
        svd_eps,
    )
}

pub(crate) fn farmcpu_super_keep_from_sample_major(
    sample_major: &[f64],
    n: usize,
    k: usize,
    pval: &[f64],
    thr: f64,
) -> Vec<bool> {
    let mut keep = vec![true; k];
    if k == 0 || n == 0 {
        return keep;
    }

    let mut centered = vec![0.0_f64; k * n];
    let mut std = vec![0.0_f64; k];
    for c in 0..k {
        let mut mean = 0.0_f64;
        for i in 0..n {
            mean += sample_major[i * k + c];
        }
        mean /= n as f64;

        let mut ss = 0.0_f64;
        for i in 0..n {
            let z = sample_major[i * k + c] - mean;
            centered[c * n + i] = z;
            ss += z * z;
        }
        std[c] = if n > 1 {
            (ss / ((n - 1) as f64)).sqrt()
        } else {
            0.0
        };
    }

    for i in 0..k {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..k {
            if !keep[j] {
                continue;
            }
            let denom = ((n.saturating_sub(1)) as f64) * std[i] * std[j];
            let cij = if denom > 0.0 && denom.is_finite() {
                let mut dot_ij = 0.0_f64;
                for t in 0..n {
                    dot_ij += centered[i * n + t] * centered[j * n + t];
                }
                dot_ij / denom
            } else {
                f64::NAN
            };
            if cij >= thr || cij <= -thr {
                if pval[i] >= pval[j] {
                    keep[i] = false;
                } else {
                    keep[j] = false;
                    break;
                }
            }
        }
    }
    keep
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let maxit = 200;
    let eps = 3.0e-14;
    let fpmin = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=maxit {
        let m2 = 2.0 * (m as f64);
        let mut aa = (m as f64) * (b - (m as f64)) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + (m as f64)) * (qab + (m as f64)) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

fn betai(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    let ln_beta = libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);
    if x < (a + 1.0) / (a + b + 2.0) {
        let front = ((a * x.ln()) + (b * (1.0 - x).ln()) - ln_beta).exp() / a;
        front * betacf(a, b, x)
    } else {
        let front = ((b * (1.0 - x).ln()) + (a * x.ln()) - ln_beta).exp() / b;
        1.0 - front * betacf(b, a, 1.0 - x)
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
    let a = v / 2.0;
    let b = 0.5;
    let mut p = betai(a, b, x);
    if !p.is_finite() {
        p = 1.0;
    }
    p.clamp(f64::MIN_POSITIVE, 1.0)
}

#[inline]
fn xs_t_ixx_into(xs: &[f64], ixx: &[f64], q0: usize, out_b21: &mut [f64]) {
    for j in 0..q0 {
        let mut acc = 0.0;
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
fn matvec_into(a: &[f64], dim: usize, rhs: &[f64], out: &mut [f64]) {
    for r in 0..dim {
        let row = &a[r * dim..(r + 1) * dim];
        out[r] = row.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum();
    }
}

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
            xs: vec![0.0; q0],
            b21: vec![0.0; q0],
            rhs: vec![0.0; dim],
            beta: vec![0.0; dim],
            ixxs: vec![0.0; dim * dim],
        }
    }
    #[inline]
    fn reset_xs(&mut self) {
        self.xs.fill(0.0);
    }
}

fn pinv_xtx_with_rank(xtx: &[f64], q: usize) -> Result<(Vec<f64>, usize), String> {
    if q == 0 {
        return Ok((Vec::new(), 0usize));
    }
    let a = DMatrix::<f64>::from_row_slice(q, q, xtx);
    let svd = a.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| "SVD failed to produce U for X'X pseudo-inverse".to_string())?;
    let vt = svd
        .v_t
        .ok_or_else(|| "SVD failed to produce V^T for X'X pseudo-inverse".to_string())?;
    let mut s_inv = DMatrix::<f64>::zeros(q, q);
    let smax = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let rcond = 1e-12_f64;
    let cutoff = rcond * smax.max(1.0);
    let mut rank = 0usize;
    for i in 0..q {
        let s = svd.singular_values[i];
        if s.is_finite() && s > cutoff {
            s_inv[(i, i)] = 1.0 / s;
            rank += 1;
        }
    }
    let v = vt.transpose();
    let ixx = v * s_inv * u.transpose();
    Ok((ixx.as_slice().to_vec(), rank))
}

fn pinv_xtx(xtx: &[f64], q: usize) -> Result<Vec<f64>, String> {
    let (ixx, _rank) = pinv_xtx_with_rank(xtx, q)?;
    Ok(ixx)
}

#[inline]
fn farmcpu_final_window_bp(szbin: &[i64]) -> i64 {
    if let Ok(raw) = std::env::var("JX_FARMCPU_FINAL_WINDOW_BP") {
        if let Ok(v) = raw.trim().parse::<i64>() {
            if v >= 0 {
                return v;
            }
        }
    }
    szbin
        .iter()
        .copied()
        .filter(|&v| v > 0)
        .min()
        .unwrap_or(500_000_i64)
}

fn pinv_for_row_major_x_with_rank(
    x_flat: &[f64],
    n: usize,
    q: usize,
) -> Result<(Vec<f64>, usize), String> {
    if x_flat.len() != n.saturating_mul(q) {
        return Err("X shape mismatch for X'X pseudo-inverse".to_string());
    }
    let mut xtx = vec![0.0_f64; q * q];
    for i in 0..n {
        let row = &x_flat[i * q..(i + 1) * q];
        for a in 0..q {
            let va = row[a];
            for b in 0..q {
                xtx[a * q + b] += va * row[b];
            }
        }
    }
    pinv_xtx_with_rank(&xtx, q)
}

fn pinv_for_row_major_x(x_flat: &[f64], n: usize, q: usize) -> Result<Vec<f64>, String> {
    let (ixx, _rank) = pinv_for_row_major_x_with_rank(x_flat, n, q)?;
    Ok(ixx)
}

fn farmcpu_window_representatives(
    sorted_candidates: Vec<(f64, usize)>,
    chrom: &[String],
    pos: &[i64],
    window_bp: i64,
    max_keep: usize,
) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(max_keep.min(sorted_candidates.len()));
    let bp = window_bp.max(0) as u64;
    'candidate: for (_p, idx) in sorted_candidates {
        if idx >= chrom.len() || idx >= pos.len() {
            continue;
        }
        for &kept in out.iter() {
            if kept < chrom.len() && kept < pos.len() && chrom[kept] == chrom[idx] {
                if pos[kept].abs_diff(pos[idx]) <= bp {
                    continue 'candidate;
                }
            }
        }
        out.push(idx);
        if out.len() >= max_keep {
            break;
        }
    }
    out.sort_unstable();
    out
}

#[derive(Clone, Debug)]
struct FarmcpuFinalWindow {
    chrom: String,
    start: i64,
    end: i64,
    qtn_model_positions: Vec<usize>,
}

#[inline]
fn farmcpu_stage1_ld_merge_r2_threshold() -> f64 {
    if let Ok(raw) = std::env::var("JX_FARMCPU_STAGE1_LD_MERGE_R2") {
        if let Ok(v) = raw.trim().parse::<f64>() {
            if v.is_finite() {
                return v;
            }
        }
    }
    0.8
}

#[inline]
fn farmcpu_final_ld_merge_r2_threshold() -> f64 {
    if let Ok(raw) = std::env::var("JX_FARMCPU_FINAL_LD_MERGE_R2") {
        if let Ok(v) = raw.trim().parse::<f64>() {
            if v.is_finite() {
                return v;
            }
        }
    }
    0.5
}

fn find_parent(parent: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while parent[r] != r {
        r = parent[r];
    }
    let mut y = x;
    while parent[y] != y {
        let next = parent[y];
        parent[y] = r;
        y = next;
    }
    r
}

fn union_parent(parent: &mut [usize], a: usize, b: usize) {
    let ra = find_parent(parent, a);
    let rb = find_parent(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

fn sample_major_r2(sample_major: &[f64], n: usize, k: usize, a: usize, b: usize) -> f64 {
    if n == 0 || a >= k || b >= k {
        return 0.0;
    }
    let mut ma = 0.0_f64;
    let mut mb = 0.0_f64;
    for i in 0..n {
        ma += sample_major[i * k + a];
        mb += sample_major[i * k + b];
    }
    ma /= n as f64;
    mb /= n as f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    let mut cov = 0.0_f64;
    for i in 0..n {
        let da = sample_major[i * k + a] - ma;
        let db = sample_major[i * k + b] - mb;
        va += da * da;
        vb += db * db;
        cov += da * db;
    }
    let den = va * vb;
    if den <= 0.0 || !den.is_finite() {
        0.0
    } else {
        ((cov * cov) / den).clamp(0.0, 1.0)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_farmcpu_final_windows(
    qtn_idx: &[usize],
    chrom: &[String],
    pos: &[i64],
    final_window_bp: i64,
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    ld_thr: f64,
    merge_overlapping_windows: bool,
) -> Result<Vec<FarmcpuFinalWindow>, String> {
    let k = qtn_idx.len();
    if k == 0 || final_window_bp < 0 {
        return Ok(Vec::new());
    }
    let bp = final_window_bp.max(0);
    let mut parent: Vec<usize> = (0..k).collect();
    let mut starts = vec![0_i64; k];
    let mut ends = vec![0_i64; k];
    for (j, &idx) in qtn_idx.iter().enumerate() {
        if idx >= chrom.len() || idx >= pos.len() {
            return Err("QTN index out of metadata bounds during window merge".to_string());
        }
        starts[j] = pos[idx].saturating_sub(bp);
        ends[j] = pos[idx].saturating_add(bp);
    }

    if merge_overlapping_windows {
        for a in 0..k {
            for b in (a + 1)..k {
                let ia = qtn_idx[a];
                let ib = qtn_idx[b];
                if chrom[ia] == chrom[ib] && starts[a] <= ends[b] && starts[b] <= ends[a] {
                    union_parent(&mut parent, a, b);
                }
            }
        }
    }

    if k > 1 && ld_thr.is_finite() && ld_thr > 0.0 {
        let sample_major = decode_packed_rows_to_sample_major(
            packed_flat,
            bytes_per_snp,
            qtn_idx,
            row_indices,
            sample_idx,
            row_flip,
            row_maf,
        )?;
        for a in 0..k {
            for b in (a + 1)..k {
                let ia = qtn_idx[a];
                let ib = qtn_idx[b];
                if chrom[ia] != chrom[ib] {
                    continue;
                }
                let r2 = sample_major_r2(&sample_major, sample_idx.len(), k, a, b);
                if r2 >= ld_thr {
                    union_parent(&mut parent, a, b);
                }
            }
        }
    }

    let mut grouped: HashMap<usize, FarmcpuFinalWindow> = HashMap::new();
    for j in 0..k {
        let root = find_parent(&mut parent, j);
        let idx = qtn_idx[j];
        grouped
            .entry(root)
            .and_modify(|w| {
                w.start = w.start.min(starts[j]);
                w.end = w.end.max(ends[j]);
                w.qtn_model_positions.push(j);
            })
            .or_insert_with(|| FarmcpuFinalWindow {
                chrom: chrom[idx].clone(),
                start: starts[j],
                end: ends[j],
                qtn_model_positions: vec![j],
            });
    }
    let mut windows: Vec<FarmcpuFinalWindow> = grouped.into_values().collect();
    windows.sort_by(|a, b| {
        a.chrom
            .cmp(&b.chrom)
            .then_with(|| a.start.cmp(&b.start))
            .then_with(|| a.end.cmp(&b.end))
    });

    for w in windows.iter_mut() {
        w.qtn_model_positions.sort_unstable();
        w.qtn_model_positions.dedup();
    }
    if !merge_overlapping_windows {
        return Ok(windows);
    }

    let mut merged: Vec<FarmcpuFinalWindow> = Vec::with_capacity(windows.len());
    for w in windows {
        if let Some(last) = merged.last_mut() {
            if last.chrom == w.chrom && w.start <= last.end {
                last.end = last.end.max(w.end);
                last.qtn_model_positions.extend(w.qtn_model_positions);
                last.qtn_model_positions.sort_unstable();
                last.qtn_model_positions.dedup();
                continue;
            }
        }
        merged.push(w);
    }
    Ok(merged)
}

#[allow(clippy::too_many_arguments)]
fn farmcpu_prune_qtn_by_merged_windows(
    qtn_idx: Vec<usize>,
    qtn_scores: &HashMap<usize, f64>,
    chrom: &[String],
    pos: &[i64],
    final_window_bp: i64,
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    ld_thr: f64,
    merge_overlapping_windows: bool,
    max_keep: usize,
) -> Result<Vec<usize>, String> {
    if qtn_idx.is_empty() || max_keep == 0 {
        return Ok(Vec::new());
    }
    let windows = build_farmcpu_final_windows(
        &qtn_idx,
        chrom,
        pos,
        final_window_bp,
        packed_flat,
        bytes_per_snp,
        row_indices,
        sample_idx,
        row_flip,
        row_maf,
        ld_thr,
        merge_overlapping_windows,
    )?;
    let mut reps: Vec<(f64, usize)> = Vec::with_capacity(windows.len());
    for window in windows.iter() {
        let mut best: Option<(f64, usize)> = None;
        for &model_pos in window.qtn_model_positions.iter() {
            let idx = qtn_idx[model_pos];
            let mut p = qtn_scores.get(&idx).copied().unwrap_or(1.0);
            if !p.is_finite() {
                p = 1.0;
            }
            match best {
                Some((bp, bi)) if p.total_cmp(&bp).is_ge() || (p == bp && idx >= bi) => {}
                _ => best = Some((p, idx)),
            }
        }
        if let Some(v) = best {
            reps.push(v);
        }
    }
    reps.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    reps.truncate(max_keep);
    let mut out: Vec<usize> = reps.into_iter().map(|(_, idx)| idx).collect();
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn farmcpu_best_rem_lead_set(
    y: &[f64],
    x_qtn: &[f64],
    q_total: usize,
    femp: &[f64],
    global_pos: &[i64],
    szbin_i64: &[i64],
    nbin_vals: &[usize],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    pool: Option<&std::sync::Arc<rayon::ThreadPool>>,
) -> Result<Vec<usize>, String> {
    let n = y.len();
    let rem_prepared = prepare_farmcpu_exact_rem(y, x_qtn, n, q_total)?;
    let mut combine: Vec<(i64, usize)> = Vec::new();
    for &sz in szbin_i64.iter() {
        for &nn in nbin_vals.iter() {
            combine.push((sz, nn));
        }
    }
    if combine.is_empty() {
        return Ok(Vec::new());
    }

    let rem_task = || {
        combine
            .par_iter()
            .map(|&(sz, nn)| {
                let leadidx = select_lead_indices(sz, nn, femp, global_pos);
                let sample_major = decode_packed_rows_to_sample_major(
                    packed_flat,
                    bytes_per_snp,
                    &leadidx,
                    row_indices,
                    sample_idx,
                    row_flip,
                    row_maf,
                );
                let sample_major = match sample_major {
                    Ok(v) => v,
                    Err(_) => return (f64::INFINITY, leadidx),
                };
                let score = farmcpu_ll_score_from_sample_major_prepared(
                    &rem_prepared,
                    &sample_major,
                    leadidx.len(),
                    -5.0,
                    5.0,
                    1e-4,
                    50,
                    1e-8,
                )
                .unwrap_or(f64::INFINITY);
                (score, leadidx)
            })
            .collect::<Vec<(f64, Vec<usize>)>>()
    };
    let rem_res = if let Some(p) = pool {
        p.install(rem_task)
    } else {
        rem_task()
    };
    if rem_res.is_empty() {
        return Ok(Vec::new());
    }
    let mut best_i = 0usize;
    let mut best_score = rem_res[0].0;
    for (i, (s, _)) in rem_res.iter().enumerate().skip(1) {
        if s.total_cmp(&best_score).is_lt() {
            best_score = *s;
            best_i = i;
        }
    }
    Ok(rem_res[best_i].1.clone())
}

fn build_x_with_qtn_packed(
    base_x: &[f64],
    n: usize,
    q_base: usize,
    qtn_idx: &[usize],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
) -> Result<(Vec<f64>, usize), String> {
    if base_x.len() != n * q_base {
        return Err("base X shape mismatch".to_string());
    }
    let k = qtn_idx.len();
    let q_total = q_base + k;
    let mut out = vec![0.0_f64; n * q_total];
    for i in 0..n {
        let src = &base_x[i * q_base..(i + 1) * q_base];
        let dst = &mut out[i * q_total..i * q_total + q_base];
        dst.copy_from_slice(src);
    }
    if k > 0 {
        let decoded = decode_packed_rows_to_sample_major(
            packed_flat,
            bytes_per_snp,
            qtn_idx,
            row_indices,
            sample_idx,
            row_flip,
            row_maf,
        )?;
        for i in 0..n {
            let src = &decoded[i * k..(i + 1) * k];
            let dst = &mut out[i * q_total + q_base..(i + 1) * q_total];
            dst.copy_from_slice(src);
        }
    }
    Ok((out, q_total))
}

#[allow(dead_code)]
fn scan_packed_full_matrix(
    y: &[f64],
    x_flat: &[f64],   // row-major (n, q0)
    ixx_flat: &[f64], // row-major (q0, q0)
    packed_flat: &[u8],
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    threads: usize,
) -> Result<(Vec<f64>, usize, usize), String> {
    let n = y.len();
    if n == 0 {
        return Err("empty y".to_string());
    }
    if sample_idx.len() != n {
        return Err("sample_idx length mismatch".to_string());
    }
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let bytes_per_snp = (n_samples + 3) / 4;
    if bytes_per_snp == 0 {
        return Err("invalid packed shape".to_string());
    }
    if packed_flat.len() % bytes_per_snp != 0 {
        return Err("packed length is not divisible by bytes_per_snp".to_string());
    }
    let m_packed = packed_flat.len() / bytes_per_snp;
    let m = row_indices.map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m || row_maf.len() != m {
        return Err("row_flip/row_maf length mismatch".to_string());
    }
    let q0 = if n == 0 { 0 } else { x_flat.len() / n };
    if x_flat.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("ixx shape mismatch".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!("n too small for LM core: n={n}, q={q0}"));
    }

    let row_stride = q0 + 3;
    let dim = q0 + 1;
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let mut out = vec![f64::NAN; m * row_stride];
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let mut runner = || {
        out.par_chunks_mut(row_stride).enumerate().for_each_init(
            || GlmScratch::new(q0),
            |scr, (idx, row_out)| {
                scr.reset_xs();
                let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                let flip = row_flip[idx];
                let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);
                let mut sy = 0.0_f64;
                let mut ss = 0.0_f64;
                for (k, &sidx) in sample_idx.iter().enumerate() {
                    let b = row[sidx >> 2];
                    let code = (b >> ((sidx & 3) * 2)) & 0b11;
                    let mut gv = match decode_plink_bed_hardcall(code) {
                        Some(v) => v,
                        None => mean_g,
                    };
                    if flip && code != 0b01 {
                        gv = 2.0 - gv;
                    }
                    sy += gv * y[k];
                    ss += gv * gv;
                    let xrow = &x_flat[k * q0..(k + 1) * q0];
                    for j in 0..q0 {
                        scr.xs[j] += xrow[j] * gv;
                    }
                }
                xs_t_ixx_into(&scr.xs, ixx_flat, q0, &mut scr.b21);
                let t2 = dot(&scr.b21, &scr.xs);
                let b22 = ss - t2;
                let (invb22, df) = if b22 < 1e-8 {
                    (0.0, (n as i32) - (q0 as i32))
                } else {
                    (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                };
                if df <= 0 {
                    row_out.fill(f64::NAN);
                    return;
                }
                build_ixxs_into(ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);
                scr.rhs[..q0].copy_from_slice(&xy);
                scr.rhs[q0] = sy;
                matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);
                let beta_rhs = dot(&scr.beta, &scr.rhs);
                let ve = (yy - beta_rhs) / (df as f64);
                for ff in 0..dim {
                    let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
                    let t = scr.beta[ff] / se;
                    row_out[2 + ff] = student_t_p_two_sided(t, df);
                }
                if invb22 == 0.0 {
                    row_out[0] = f64::NAN;
                    row_out[1] = f64::NAN;
                    row_out[2 + q0] = f64::NAN;
                } else {
                    row_out[0] = scr.beta[q0];
                    row_out[1] = (scr.ixxs[q0 * dim + q0] * ve).sqrt();
                }
            },
        );
    };
    if let Some(p) = &pool {
        p.install(&mut runner);
    } else {
        runner();
    }
    Ok((out, m, row_stride))
}

fn scan_packed_full_matrix_reduced(
    y: &[f64],
    x_flat: &[f64],   // row-major (n, q0)
    ixx_flat: &[f64], // row-major (q0, q0)
    packed_flat: &[u8],
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    q_base: usize,
    threads: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = y.len();
    if n == 0 {
        return Err("empty y".to_string());
    }
    if sample_idx.len() != n {
        return Err("sample_idx length mismatch".to_string());
    }
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let bytes_per_snp = (n_samples + 3) / 4;
    if bytes_per_snp == 0 {
        return Err("invalid packed shape".to_string());
    }
    if packed_flat.len() % bytes_per_snp != 0 {
        return Err("packed length is not divisible by bytes_per_snp".to_string());
    }
    let m_packed = packed_flat.len() / bytes_per_snp;
    let m = row_indices.map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m || row_maf.len() != m {
        return Err("row_flip/row_maf length mismatch".to_string());
    }
    let q0 = if n == 0 { 0 } else { x_flat.len() / n };
    if x_flat.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("ixx shape mismatch".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!("n too small for LM core: n={n}, q={q0}"));
    }
    if q_base > q0 {
        return Err(format!(
            "q_base exceeds background design width: q_base={q_base}, q0={q0}"
        ));
    }

    let k_qtn = q0.saturating_sub(q_base);
    let dim = q0 + 1;
    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let mut marker_p = vec![1.0_f64; m];
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let chunk_rows = 256usize;
    let mut runner = || {
        marker_p
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .map_init(
                || GlmScratch::new(q0),
                |scr, (chunk_idx, p_chunk)| {
                    let mut local_qtn_min = vec![1.0_f64; k_qtn];
                    let start = chunk_idx * chunk_rows;
                    for (off, p_out) in p_chunk.iter_mut().enumerate() {
                        let idx = start + off;
                        scr.reset_xs();
                        let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                        let row =
                            &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                        let flip = row_flip[idx];
                        let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);
                        let mut sy = 0.0_f64;
                        let mut ss = 0.0_f64;
                        for (k, &sidx) in sample_idx.iter().enumerate() {
                            let b = row[sidx >> 2];
                            let code = (b >> ((sidx & 3) * 2)) & 0b11;
                            let mut gv = match decode_plink_bed_hardcall(code) {
                                Some(v) => v,
                                None => mean_g,
                            };
                            if flip && code != 0b01 {
                                gv = 2.0 - gv;
                            }
                            sy += gv * y[k];
                            ss += gv * gv;
                            let xrow = &x_flat[k * q0..(k + 1) * q0];
                            for j in 0..q0 {
                                scr.xs[j] += xrow[j] * gv;
                            }
                        }
                        xs_t_ixx_into(&scr.xs, ixx_flat, q0, &mut scr.b21);
                        let t2 = dot(&scr.b21, &scr.xs);
                        let b22 = ss - t2;
                        let (invb22, df) = if b22 < 1e-8 {
                            (0.0, (n as i32) - (q0 as i32))
                        } else {
                            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                        };
                        if df <= 0 {
                            *p_out = 1.0;
                            continue;
                        }
                        build_ixxs_into(ixx_flat, &scr.b21, invb22, q0, &mut scr.ixxs);
                        scr.rhs[..q0].copy_from_slice(&xy);
                        scr.rhs[q0] = sy;
                        matvec_into(&scr.ixxs, dim, &scr.rhs, &mut scr.beta);
                        let beta_rhs = dot(&scr.beta, &scr.rhs);
                        let ve = (yy - beta_rhs) / (df as f64);
                        if invb22 == 0.0 || !ve.is_finite() || ve < 0.0 {
                            *p_out = 1.0;
                            continue;
                        }

                        let mut marker_p_here = 1.0_f64;
                        for ff in 0..dim {
                            let se = (scr.ixxs[ff * dim + ff] * ve).sqrt();
                            let p = if se.is_finite() && se > 0.0 && scr.beta[ff].is_finite() {
                                student_t_p_two_sided(scr.beta[ff] / se, df)
                            } else {
                                1.0_f64
                            };
                            if ff >= q_base && ff < q0 {
                                let slot = ff - q_base;
                                if slot < local_qtn_min.len()
                                    && p.is_finite()
                                    && p < local_qtn_min[slot]
                                {
                                    local_qtn_min[slot] = p;
                                }
                            } else if ff == q0 {
                                marker_p_here = sanitize_assoc_pvalue(scr.beta[ff], se, p);
                            }
                        }
                        *p_out = marker_p_here;
                    }
                    local_qtn_min
                },
            )
            .reduce(
                || vec![1.0_f64; k_qtn],
                |mut acc, local| {
                    for (dst, src) in acc.iter_mut().zip(local.iter()) {
                        if src.is_finite() && src.total_cmp(dst).is_lt() {
                            *dst = *src;
                        }
                    }
                    acc
                },
            )
    };
    let qtn_min_p = if let Some(p) = &pool {
        p.install(&mut runner)
    } else {
        runner()
    };
    Ok((marker_p, qtn_min_p))
}

#[allow(clippy::too_many_arguments)]
fn farmcpu_conditional_subset_stats(
    y: &[f64],
    x_bg: &[f64],
    ixx_flat: &[f64],
    packed_flat: &[u8],
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    target_rows: &[usize],
    threads: usize,
) -> Result<Vec<[f64; 3]>, String> {
    let n = y.len();
    if n == 0 {
        return Err("empty phenotype vector".to_string());
    }
    if sample_idx.len() != n {
        return Err("sample_idx length mismatch".to_string());
    }
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let q0 = x_bg.len() / n;
    if x_bg.len() != n * q0 {
        return Err("background X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("background inverse X'X shape mismatch".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!("n too small for conditional scan: n={n}, q={q0}"));
    }
    let bytes_per_snp = n_samples.div_ceil(4);
    if bytes_per_snp == 0 || packed_flat.len() % bytes_per_snp != 0 {
        return Err("invalid packed shape for conditional scan".to_string());
    }
    let m_packed = packed_flat.len() / bytes_per_snp;
    let m = row_indices.map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m || row_maf.len() != m {
        return Err("row_flip/row_maf length mismatch".to_string());
    }
    if target_rows.iter().any(|&idx| idx >= m) {
        return Err("conditional scan target row out of range".to_string());
    }

    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_bg[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let bg_df = (n as i32) - (q0 as i32);
    if bg_df <= 0 {
        return Err(format!(
            "n too small for conditional background: n={n}, q={q0}"
        ));
    }
    let mut bg_beta = vec![0.0_f64; q0];
    matvec_into(ixx_flat, q0, &xy, &mut bg_beta);
    let bg_rss = (yy - dot(&bg_beta, &xy)).max(0.0);
    let mut y_resid = vec![0.0_f64; n];
    for i in 0..n {
        let row = &x_bg[i * q0..(i + 1) * q0];
        y_resid[i] = y[i] - dot(row, &bg_beta);
    }

    let mut out = vec![[f64::NAN; 3]; target_rows.len()];
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let mut runner = || {
        out.par_iter_mut()
            .zip(target_rows.par_iter())
            .for_each_init(
                || GlmScratch::new(q0),
                |scr, (stat_out, &idx)| {
                    scr.reset_xs();
                    let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                    if src_row >= m_packed {
                        return;
                    }
                    let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                    let flip = row_flip[idx];
                    let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);
                    let mut sy_resid = 0.0_f64;
                    let mut ss = 0.0_f64;
                    for (k, &sidx) in sample_idx.iter().enumerate() {
                        let b = row[sidx >> 2];
                        let code = (b >> ((sidx & 3) * 2)) & 0b11;
                        let mut gv = match decode_plink_bed_hardcall(code) {
                            Some(v) => v,
                            None => mean_g,
                        };
                        if flip && code != 0b01 {
                            gv = 2.0 - gv;
                        }
                        sy_resid += gv * y_resid[k];
                        ss += gv * gv;
                        let xrow = &x_bg[k * q0..(k + 1) * q0];
                        for j in 0..q0 {
                            scr.xs[j] += xrow[j] * gv;
                        }
                    }
                    xs_t_ixx_into(&scr.xs, ixx_flat, q0, &mut scr.b21);
                    let b22 = ss - dot(&scr.b21, &scr.xs);
                    let df = bg_df - 1;
                    if df <= 0 || b22 <= 1e-8 || !b22.is_finite() {
                        return;
                    }
                    let beta = sy_resid / b22;
                    let rss_full = (bg_rss - (sy_resid * sy_resid / b22)).max(0.0);
                    let ve = rss_full / (df as f64);
                    let se = (ve / b22).sqrt();
                    if !beta.is_finite() || !se.is_finite() || se <= 0.0 {
                        return;
                    }
                    let pwald = student_t_p_two_sided(beta / se, df);
                    *stat_out = [beta, se, pwald];
                },
            );
    };
    if let Some(tp) = &pool {
        tp.install(&mut runner);
    } else {
        runner();
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn farmcpu_scan_packed_marker_pvalues(
    y: &[f64],
    x_bg: &[f64],
    ixx_flat: &[f64],
    packed_flat: &[u8],
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    threads: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = y.len();
    if n == 0 {
        return Err("empty phenotype vector".to_string());
    }
    if sample_idx.len() != n {
        return Err("sample_idx length mismatch".to_string());
    }
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let q0 = x_bg.len() / n;
    if x_bg.len() != n * q0 {
        return Err("background X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("background inverse X'X shape mismatch".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!(
            "n too small for FarmCPU stage1 scan: n={n}, q={q0}"
        ));
    }
    let bytes_per_snp = n_samples.div_ceil(4);
    if bytes_per_snp == 0 || packed_flat.len() % bytes_per_snp != 0 {
        return Err("invalid packed shape for FarmCPU stage1 scan".to_string());
    }
    let m_packed = packed_flat.len() / bytes_per_snp;
    let m = row_indices.map(|v| v.len()).unwrap_or(m_packed);
    if row_flip.len() != m || row_maf.len() != m {
        return Err("row_flip/row_maf length mismatch".to_string());
    }
    if row_indices
        .map(|v| v.iter().any(|&idx| idx >= m_packed))
        .unwrap_or(false)
    {
        return Err("row_indices out of bounds for packed rows".to_string());
    }

    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_bg[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let bg_df = (n as i32) - (q0 as i32);
    if bg_df <= 0 {
        return Err(format!(
            "n too small for FarmCPU stage1 background: n={n}, q={q0}"
        ));
    }
    let mut bg_beta = vec![0.0_f64; q0];
    matvec_into(ixx_flat, q0, &xy, &mut bg_beta);
    let bg_rss = (yy - dot(&bg_beta, &xy)).max(0.0);
    let bg_ve = bg_rss / (bg_df as f64);
    let mut bg_pwald = vec![1.0_f64; q0];
    for j in 0..q0 {
        let se = (ixx_flat[j * q0 + j] * bg_ve).sqrt();
        if se.is_finite() && se > 0.0 && bg_beta[j].is_finite() {
            bg_pwald[j] = student_t_p_two_sided(bg_beta[j] / se, bg_df);
        }
    }
    let mut y_resid = vec![0.0_f64; n];
    for i in 0..n {
        let row = &x_bg[i * q0..(i + 1) * q0];
        y_resid[i] = y[i] - dot(row, &bg_beta);
    }

    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let row_block = 8192usize;
    let mut geno_buf = vec![0.0_f64; row_block * n];
    let mut xs_buf = vec![0.0_f64; row_block * q0];
    let mut b21_buf = vec![0.0_f64; row_block * q0];
    let mut sy_resid_buf = vec![0.0_f64; row_block];
    let mut ss_buf = vec![0.0_f64; row_block];
    let mut pvals = vec![1.0_f64; m];

    for row_start in (0..m).step_by(row_block) {
        let row_end = (row_start + row_block).min(m);
        let n_block = row_end - row_start;
        let geno_slice = &mut geno_buf[..n_block * n];
        let sy_slice = &mut sy_resid_buf[..n_block];
        let ss_slice = &mut ss_buf[..n_block];
        let mut decode_runner = || {
            geno_slice
                .par_chunks_mut(n)
                .zip(sy_slice.par_iter_mut())
                .zip(ss_slice.par_iter_mut())
                .enumerate()
                .for_each(|(local_idx, ((geno_row, sy_out), ss_out))| {
                    let idx = row_start + local_idx;
                    let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                    let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                    let flip = row_flip[idx];
                    let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);
                    let mut sy_resid = 0.0_f64;
                    let mut ss = 0.0_f64;
                    for (k, &sidx) in sample_idx.iter().enumerate() {
                        let b = row[sidx >> 2];
                        let code = (b >> ((sidx & 3) * 2)) & 0b11;
                        let mut gv = match decode_plink_bed_hardcall(code) {
                            Some(v) => v,
                            None => mean_g,
                        };
                        if flip && code != 0b01 {
                            gv = 2.0 - gv;
                        }
                        geno_row[k] = gv;
                        sy_resid += gv * y_resid[k];
                        ss += gv * gv;
                    }
                    *sy_out = sy_resid;
                    *ss_out = ss;
                });
        };
        if let Some(p) = &pool {
            p.install(&mut decode_runner);
        } else {
            decode_runner();
        }

        let xs_slice = &mut xs_buf[..n_block * q0];
        let b21_slice = &mut b21_buf[..n_block * q0];
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                n_block as CblasInt,
                q0 as CblasInt,
                n as CblasInt,
                1.0_f64,
                geno_slice.as_ptr(),
                n as CblasInt,
                x_bg.as_ptr(),
                q0 as CblasInt,
                0.0_f64,
                xs_slice.as_mut_ptr(),
                q0 as CblasInt,
            );
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                n_block as CblasInt,
                q0 as CblasInt,
                q0 as CblasInt,
                1.0_f64,
                xs_slice.as_ptr(),
                q0 as CblasInt,
                ixx_flat.as_ptr(),
                q0 as CblasInt,
                0.0_f64,
                b21_slice.as_mut_ptr(),
                q0 as CblasInt,
            );
        }

        let p_slice = &mut pvals[row_start..row_end];
        let mut stat_runner = || {
            p_slice
                .par_iter_mut()
                .enumerate()
                .for_each(|(local_idx, p_out)| {
                    let xs = &xs_slice[local_idx * q0..(local_idx + 1) * q0];
                    let b21 = &b21_slice[local_idx * q0..(local_idx + 1) * q0];
                    let b22 = ss_slice[local_idx] - dot(b21, xs);
                    let df_row = bg_df - 1;
                    if df_row <= 0 || b22 <= 1e-8 || !b22.is_finite() {
                        *p_out = 1.0;
                        return;
                    }
                    let sy_resid = sy_slice[local_idx];
                    let beta = sy_resid / b22;
                    let rss_full = (bg_rss - (sy_resid * sy_resid / b22)).max(0.0);
                    let ve = rss_full / (df_row as f64);
                    let se = (ve / b22).sqrt();
                    if !beta.is_finite() || !se.is_finite() || se <= 0.0 {
                        *p_out = 1.0;
                        return;
                    }
                    *p_out = student_t_p_two_sided(beta / se, df_row);
                });
        };
        if let Some(p) = &pool {
            p.install(&mut stat_runner);
        } else {
            stat_runner();
        }
    }

    Ok((pvals, bg_pwald))
}

#[allow(clippy::too_many_arguments)]
fn write_farmcpu_packed_main_scan(
    out_writer: &AsyncTsvWriter,
    qtn_writer: Option<&AsyncTsvWriter>,
    qtn_path_for_err: &str,
    y: &[f64],
    x_qtn: &[f64],
    ixx_flat: &[f64],
    packed_flat: &[u8],
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    threads: usize,
    chrom: &[String],
    pos: &[i64],
    snp: &[String],
    allele0: &[String],
    allele1: &[String],
    miss_counts: &[usize],
    q_base: usize,
    qtn_lookup: &HashMap<usize, usize>,
    qtn_idx: &[usize],
    final_window_bp: i64,
    enable_local_window_merge: bool,
) -> Result<(usize, usize), String> {
    let n = y.len();
    if n == 0 {
        return Err("empty phenotype vector".to_string());
    }
    if sample_idx.len() != n {
        return Err("sample_idx length mismatch".to_string());
    }
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let q0 = x_qtn.len() / n;
    if x_qtn.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("ixx shape mismatch".to_string());
    }
    let m = chrom.len();
    if pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
        || row_flip.len() != m
        || row_maf.len() != m
        || miss_counts.len() != m
    {
        return Err("FarmCPU metadata length mismatch during streaming scan".to_string());
    }

    let bytes_per_snp = (n_samples + 3) / 4;
    if bytes_per_snp == 0 {
        return Err("invalid packed shape".to_string());
    }
    if packed_flat.len() % bytes_per_snp != 0 {
        return Err("packed length is not divisible by bytes_per_snp".to_string());
    }
    let m_packed = packed_flat.len() / bytes_per_snp;
    if row_indices
        .map(|v| v.iter().any(|&idx| idx >= m_packed))
        .unwrap_or(false)
    {
        return Err("row_indices out of bounds for packed rows".to_string());
    }
    if n <= q0 + 1 {
        return Err(format!("n too small for LM core: n={n}, q={q0}"));
    }

    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_qtn[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let bg_df = (n as i32) - (q0 as i32);
    if bg_df <= 0 {
        return Err(format!(
            "n too small for FarmCPU background model: n={n}, q={q0}"
        ));
    }
    let mut bg_beta = vec![0.0_f64; q0];
    matvec_into(ixx_flat, q0, &xy, &mut bg_beta);
    let bg_rss = (yy - dot(&bg_beta, &xy)).max(0.0);
    let bg_ve = bg_rss / (bg_df as f64);
    let mut bg_se = vec![f64::NAN; q0];
    let mut bg_pwald = vec![f64::NAN; q0];
    for j in 0..q0 {
        let se = (ixx_flat[j * q0 + j] * bg_ve).sqrt();
        bg_se[j] = se;
        if se.is_finite() && se > 0.0 && bg_beta[j].is_finite() {
            bg_pwald[j] = student_t_p_two_sided(bg_beta[j] / se, bg_df);
        }
    }
    let mut y_resid = vec![0.0_f64; n];
    for i in 0..n {
        let row = &x_qtn[i * q0..(i + 1) * q0];
        y_resid[i] = y[i] - dot(row, &bg_beta);
    }

    let mut local_overrides: HashMap<usize, [f64; 3]> = HashMap::new();
    if enable_local_window_merge && final_window_bp >= 0 && !qtn_idx.is_empty() {
        let mut base_x = vec![0.0_f64; n * q_base];
        for i in 0..n {
            let src = &x_qtn[i * q0..i * q0 + q_base];
            let dst = &mut base_x[i * q_base..(i + 1) * q_base];
            dst.copy_from_slice(src);
        }
        let windows = build_farmcpu_final_windows(
            qtn_idx,
            chrom,
            pos,
            final_window_bp,
            packed_flat,
            bytes_per_snp,
            row_indices,
            sample_idx,
            row_flip,
            row_maf,
            farmcpu_final_ld_merge_r2_threshold(),
            true,
        )?;
        let mut chr_to_rows: HashMap<&str, Vec<usize>> = HashMap::new();
        for i in 0..m {
            chr_to_rows.entry(chrom[i].as_str()).or_default().push(i);
        }
        for window in windows.iter() {
            let mut window_rows = Vec::<usize>::new();
            if let Some(rows) = chr_to_rows.get(window.chrom.as_str()) {
                for &i in rows {
                    if pos[i] >= window.start && pos[i] <= window.end {
                        window_rows.push(i);
                    }
                }
            }
            if window_rows.is_empty() {
                continue;
            }
            let mut local_qtn = Vec::<usize>::with_capacity(
                qtn_idx
                    .len()
                    .saturating_sub(window.qtn_model_positions.len()),
            );
            for (j, &idx) in qtn_idx.iter().enumerate() {
                if window.qtn_model_positions.binary_search(&j).is_err() {
                    local_qtn.push(idx);
                }
            }
            let (x_local, q_local) = build_x_with_qtn_packed(
                &base_x,
                n,
                q_base,
                &local_qtn,
                packed_flat,
                bytes_per_snp,
                row_indices,
                sample_idx,
                row_flip,
                row_maf,
            )?;
            let ixx_local = pinv_for_row_major_x(&x_local, n, q_local)?;
            let local_stats = farmcpu_conditional_subset_stats(
                y,
                &x_local,
                &ixx_local,
                packed_flat,
                n_samples,
                row_flip,
                row_maf,
                row_indices,
                sample_idx,
                &window_rows,
                threads,
            )?;
            for (&idx, stat) in window_rows.iter().zip(local_stats.into_iter()) {
                local_overrides.insert(idx, stat);
            }
        }
    }

    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let row_block = 8192usize;
    let row_stride = 3usize;
    let mut geno_buf = vec![0.0_f64; row_block * n];
    let mut xs_buf = vec![0.0_f64; row_block * q0];
    let mut b21_buf = vec![0.0_f64; row_block * q0];
    let mut sy_resid_buf = vec![0.0_f64; row_block];
    let mut ss_buf = vec![0.0_f64; row_block];
    let mut stats_buf = vec![f64::NAN; row_block * 3];
    let mut text_buf = String::with_capacity(row_block * 128);
    let mut qtn_text_buf_opt = if qtn_writer.is_some() {
        Some(String::with_capacity(
            row_block.min(qtn_lookup.len().max(1)) * 128,
        ))
    } else {
        None
    };
    let mut qtn_rows_written = 0usize;

    for row_start in (0..m).step_by(row_block) {
        let row_end = (row_start + row_block).min(m);
        let n_block = row_end - row_start;
        let stats_slice = &mut stats_buf[..n_block * row_stride];
        let geno_slice = &mut geno_buf[..n_block * n];
        let sy_slice = &mut sy_resid_buf[..n_block];
        let ss_slice = &mut ss_buf[..n_block];
        let mut decode_runner = || {
            geno_slice
                .par_chunks_mut(n)
                .zip(sy_slice.par_iter_mut())
                .zip(ss_slice.par_iter_mut())
                .enumerate()
                .for_each(|(local_idx, ((geno_row, sy_out), ss_out))| {
                    let idx = row_start + local_idx;
                    let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                    let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                    let flip = row_flip[idx];
                    let mean_g = (2.0_f64 * row_maf[idx] as f64).max(0.0);
                    let mut sy_resid = 0.0_f64;
                    let mut ss = 0.0_f64;
                    for (k, &sidx) in sample_idx.iter().enumerate() {
                        let b = row[sidx >> 2];
                        let code = (b >> ((sidx & 3) * 2)) & 0b11;
                        let mut gv = match decode_plink_bed_hardcall(code) {
                            Some(v) => v,
                            None => mean_g,
                        };
                        if flip && code != 0b01 {
                            gv = 2.0 - gv;
                        }
                        geno_row[k] = gv;
                        sy_resid += gv * y_resid[k];
                        ss += gv * gv;
                    }
                    *sy_out = sy_resid;
                    *ss_out = ss;
                });
        };
        if let Some(p) = &pool {
            p.install(&mut decode_runner);
        } else {
            decode_runner();
        }

        let xs_slice = &mut xs_buf[..n_block * q0];
        let b21_slice = &mut b21_buf[..n_block * q0];
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                n_block as CblasInt,
                q0 as CblasInt,
                n as CblasInt,
                1.0_f64,
                geno_slice.as_ptr(),
                n as CblasInt,
                x_qtn.as_ptr(),
                q0 as CblasInt,
                0.0_f64,
                xs_slice.as_mut_ptr(),
                q0 as CblasInt,
            );
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                n_block as CblasInt,
                q0 as CblasInt,
                q0 as CblasInt,
                1.0_f64,
                xs_slice.as_ptr(),
                q0 as CblasInt,
                ixx_flat.as_ptr(),
                q0 as CblasInt,
                0.0_f64,
                b21_slice.as_mut_ptr(),
                q0 as CblasInt,
            );
        }

        let mut stat_runner = || {
            stats_slice
                .par_chunks_mut(row_stride)
                .enumerate()
                .for_each(|(local_idx, stat_out)| {
                    let idx = row_start + local_idx;
                    if qtn_lookup.contains_key(&idx) || local_overrides.contains_key(&idx) {
                        stat_out[0] = f64::NAN;
                        stat_out[1] = f64::NAN;
                        stat_out[2] = f64::NAN;
                        return;
                    }
                    let xs = &xs_slice[local_idx * q0..(local_idx + 1) * q0];
                    let b21 = &b21_slice[local_idx * q0..(local_idx + 1) * q0];
                    let b22 = ss_slice[local_idx] - dot(b21, xs);
                    let df_row = bg_df - 1;
                    if df_row <= 0 || b22 <= 1e-8 || !b22.is_finite() {
                        stat_out[0] = f64::NAN;
                        stat_out[1] = f64::NAN;
                        stat_out[2] = f64::NAN;
                        return;
                    }
                    let sy_resid = sy_slice[local_idx];
                    let beta = sy_resid / b22;
                    let rss_full = (bg_rss - (sy_resid * sy_resid / b22)).max(0.0);
                    let ve = rss_full / (df_row as f64);
                    let se = (ve / b22).sqrt();
                    if !beta.is_finite() || !se.is_finite() || se <= 0.0 {
                        stat_out[0] = f64::NAN;
                        stat_out[1] = f64::NAN;
                        stat_out[2] = f64::NAN;
                        return;
                    }
                    let t = beta / se;
                    stat_out[0] = beta;
                    stat_out[1] = se;
                    stat_out[2] = student_t_p_two_sided(t, df_row);
                });
        };
        if let Some(p) = &pool {
            p.install(&mut stat_runner);
        } else {
            stat_runner();
        }

        text_buf.clear();
        if let Some(ref mut qtn_text_buf) = qtn_text_buf_opt {
            qtn_text_buf.clear();
        }
        for local_idx in 0..n_block {
            let i = row_start + local_idx;
            let stat_base = local_idx * 3;
            let mut beta = stats_slice[stat_base];
            let mut se = stats_slice[stat_base + 1];
            let mut pwald = stats_slice[stat_base + 2];
            if !pwald.is_finite() {
                pwald = 1.0;
            }
            if let Some(stat) = local_overrides.get(&i) {
                beta = stat[0];
                se = stat[1];
                pwald = sanitize_assoc_pvalue(beta, se, stat[2]);
            } else if let Some(&j) = qtn_lookup.get(&i) {
                let coef_idx = q_base + j;
                if coef_idx < q0 {
                    beta = bg_beta[coef_idx];
                    se = bg_se[coef_idx];
                    pwald = sanitize_assoc_pvalue(beta, se, bg_pwald[coef_idx]);
                } else {
                    beta = f64::NAN;
                    se = f64::NAN;
                    pwald = 1.0;
                }
            }
            let chisq = chisq_from_beta_se_and_optional_plrt(beta, se, None);
            let chisq_txt = format_chisq_value(chisq);
            let _ = write!(
                text_buf,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\n",
                chrom[i],
                pos[i],
                snp[i],
                allele0[i],
                allele1[i],
                row_maf[i],
                miss_counts[i],
                beta,
                se,
                chisq_txt,
                pwald
            );
            if let Some(ref mut qtn_text_buf) = qtn_text_buf_opt {
                if qtn_lookup.contains_key(&i) {
                    qtn_rows_written += 1;
                    let _ = write!(
                        qtn_text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{}\n",
                        chrom[i],
                        pos[i],
                        snp[i],
                        allele0[i],
                        allele1[i],
                        row_maf[i],
                        miss_counts[i],
                        beta,
                        se,
                        chisq_txt,
                        pwald,
                        row_indices.map(|v| v[i]).unwrap_or(i)
                    );
                }
            }
        }

        if !text_buf.is_empty() {
            out_writer
                .send(text_buf.as_bytes().to_vec())
                .map_err(|e| e.to_string())?;
        }
        if let (Some(q_writer), Some(qtn_text_buf)) = (qtn_writer, qtn_text_buf_opt.as_mut()) {
            if !qtn_text_buf.is_empty() {
                q_writer
                    .send(qtn_text_buf.as_bytes().to_vec())
                    .map_err(|e| format!("{qtn_path_for_err}: {e}"))?;
            }
        }
    }

    Ok((m, qtn_rows_written))
}

fn write_farmcpu_pseudo_qtn_only(
    out_tsv: &str,
    y: &[f64],
    x_qtn: &[f64],
    ixx_flat: &[f64],
    q_base: usize,
    qtn_idx: &[usize],
    chrom: &[String],
    pos: &[i64],
    snp: &[String],
    allele0: &[String],
    allele1: &[String],
    row_maf: &[f32],
    miss_counts: &[usize],
    source_rows: &[usize],
) -> Result<usize, String> {
    let n = y.len();
    if n == 0 {
        return Err("empty phenotype vector".to_string());
    }
    let q0 = x_qtn.len() / n;
    if x_qtn.len() != n * q0 {
        return Err("X shape mismatch".to_string());
    }
    if ixx_flat.len() != q0 * q0 {
        return Err("ixx shape mismatch".to_string());
    }
    let k_qtn = qtn_idx.len();
    if chrom.len() != k_qtn
        || pos.len() != k_qtn
        || snp.len() != k_qtn
        || allele0.len() != k_qtn
        || allele1.len() != k_qtn
        || row_maf.len() != k_qtn
        || miss_counts.len() != k_qtn
        || source_rows.len() != k_qtn
    {
        return Err("FarmCPU pseudo-QTN metadata length mismatch".to_string());
    }

    let mut xy = vec![0.0_f64; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_qtn[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let bg_df = (n as i32) - (q0 as i32);
    if bg_df <= 0 {
        return Err(format!(
            "n too small for FarmCPU background model: n={n}, q={q0}"
        ));
    }
    let mut bg_beta = vec![0.0_f64; q0];
    matvec_into(ixx_flat, q0, &xy, &mut bg_beta);
    let bg_rss = (yy - dot(&bg_beta, &xy)).max(0.0);
    let bg_ve = bg_rss / (bg_df as f64);

    let writer = AsyncTsvWriter::with_config(
        out_tsv,
        b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tsource_row\n",
        16 * 1024 * 1024,
        4,
    )?;
    let mut text_buf = String::with_capacity(k_qtn.max(1) * 128);
    for j in 0..k_qtn {
        let coef_idx = q_base + j;
        let (beta, se, pwald) = if coef_idx < q0 {
            let se = (ixx_flat[coef_idx * q0 + coef_idx] * bg_ve).sqrt();
            let pwald = if se.is_finite() && se > 0.0 && bg_beta[coef_idx].is_finite() {
                student_t_p_two_sided(bg_beta[coef_idx] / se, bg_df)
            } else {
                1.0_f64
            };
            (
                bg_beta[coef_idx],
                se,
                sanitize_assoc_pvalue(bg_beta[coef_idx], se, pwald),
            )
        } else {
            (f64::NAN, f64::NAN, 1.0_f64)
        };
        let chisq = chisq_from_beta_se_and_optional_plrt(beta, se, None);
        let chisq_txt = format_chisq_value(chisq);
        let _ = write!(
            text_buf,
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{}\n",
            chrom[j],
            pos[j],
            snp[j],
            allele0[j],
            allele1[j],
            row_maf[j],
            miss_counts[j],
            beta,
            se,
            chisq_txt,
            pwald,
            source_rows[j]
        );
    }
    if !text_buf.is_empty() {
        writer.send(text_buf.into_bytes())?;
    }
    writer.finish()?;
    Ok(k_qtn)
}

#[pyfunction]
#[pyo3(signature = (
    sz,
    n_lead,
    pvalue,
    pos,
    packed,
    n_samples,
    row_flip,
    row_maf,
    y,
    x,
    sample_indices=None,
    delta_exp_start=-5.0,
    delta_exp_end=5.0,
    delta_step=0.1,
    svd_eps=1e-8
))]
pub fn farmcpu_rem_packed(
    py: Python<'_>,
    sz: i64,
    n_lead: usize,
    pvalue: PyReadonlyArray1<'_, f64>,
    pos: PyReadonlyArray1<'_, i64>,
    packed: PyReadonlyArray2<'_, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'_, bool>,
    row_maf: PyReadonlyArray1<'_, f32>,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    sample_indices: Option<PyReadonlyArray1<'_, i64>>,
    delta_exp_start: f64,
    delta_exp_end: f64,
    delta_step: f64,
    svd_eps: f64,
) -> PyResult<(f64, Vec<i64>)> {
    if sz <= 0 {
        return Err(PyRuntimeError::new_err("sz must be > 0"));
    }
    let pvalue = pvalue.as_slice()?;
    let pos = pos.as_slice()?;
    if pvalue.len() != pos.len() {
        return Err(PyRuntimeError::new_err(
            "pvalue and pos length mismatch in farmcpu_rem_packed",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    if m != pvalue.len() {
        return Err(PyRuntimeError::new_err(format!(
            "packed row count mismatch: packed m={m}, pvalue len={}",
            pvalue.len()
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let n = y.len();
    if x_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    let p = x_arr.shape()[1];
    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
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

    let leadidx = select_lead_indices(sz, n_lead, pvalue, pos);
    let k = leadidx.len();
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let (score, leadidx_i64) = py.detach(|| -> PyResult<(f64, Vec<i64>)> {
        let snp_pool_sample_major = decode_packed_rows_to_sample_major(
            &packed_flat,
            bytes_per_snp,
            &leadidx,
            None,
            &sample_idx,
            row_flip,
            row_maf,
        )
        .map_err(PyRuntimeError::new_err)?;

        let score = farmcpu_ll_score_from_sample_major(
            y,
            &x_flat,
            n,
            p,
            &snp_pool_sample_major,
            k,
            delta_exp_start,
            delta_exp_end,
            delta_step,
            svd_eps,
        )
        .map_err(PyRuntimeError::new_err)?;

        let leadidx_i64 = leadidx.iter().map(|&v| v as i64).collect::<Vec<_>>();
        Ok((score, leadidx_i64))
    })?;
    Ok((score, leadidx_i64))
}

#[pyfunction]
#[pyo3(signature = (
    row_indices,
    pvalue,
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    thr=0.7
))]
pub fn farmcpu_super_packed<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, i64>,
    pvalue: PyReadonlyArray1<'py, f64>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    thr: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ridx_raw = row_indices.as_slice()?;
    let pval = pvalue.as_slice()?;
    if ridx_raw.len() != pval.len() {
        return Err(PyRuntimeError::new_err(
            "row_indices and pvalue length mismatch in farmcpu_super_packed",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m {
        return Err(PyRuntimeError::new_err(
            "row_flip/row_maf length mismatch with packed rows",
        ));
    }

    let ridx = parse_index_vec_i64(ridx_raw, m, "row_indices")?;
    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    let k = ridx.len();

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let keep = py.detach(|| -> PyResult<Vec<bool>> {
        let sample_major = decode_packed_rows_to_sample_major(
            &packed_flat,
            bytes_per_snp,
            &ridx,
            None,
            &sample_idx,
            row_flip,
            row_maf,
        )
        .map_err(PyRuntimeError::new_err)?;
        Ok(farmcpu_super_keep_from_sample_major(
            &sample_major,
            n,
            k,
            pval,
            thr,
        ))
    })?;

    let out = PyArray1::<bool>::zeros(py, [k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&keep);
    Ok(out)
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
    threshold=1e-6,
    max_iter=30,
    qtn_bound=None,
    nbin=5,
    szbin=vec![5e5, 5e6, 5e7],
    threads=0,
    progress_callback=None,
    pseudo_tsv=None,
    bed_prefix=None,
    raw=false,
    skip_main_scan=false
))]
pub fn farmcpu_packed_to_tsv(
    py: Python<'_>,
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
    threshold: f64,
    max_iter: usize,
    qtn_bound: Option<usize>,
    nbin: usize,
    szbin: Vec<f64>,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    pseudo_tsv: Option<&str>,
    bed_prefix: Option<&str>,
    raw: bool,
    skip_main_scan: bool,
) -> PyResult<(usize, usize, usize)> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(threshold.is_finite() && threshold > 0.0) {
        return Err(PyRuntimeError::new_err("threshold must be finite and > 0"));
    }
    let y = y.as_slice()?;
    let n = y.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("empty phenotype vector"));
    }

    let x_cov_arr = x_cov.as_array();
    if x_cov_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("x_cov must be 2D (n, p_cov)"));
    }
    if x_cov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err(format!(
            "x_cov rows mismatch: got {}, expected n={n}",
            x_cov_arr.shape()[0]
        )));
    }
    let p_cov = x_cov_arr.shape()[1];
    let q_base = 1 + p_cov; // intercept + covariates

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
    if m == 0 {
        return Err(PyRuntimeError::new_err("empty packed marker matrix"));
    }
    let (chrom, pos) =
        resolve_assoc_chrom_pos_metadata(bed_prefix, chrom, pos, row_idx.as_deref(), m)
            .map_err(PyRuntimeError::new_err)?;
    let string_metadata = if skip_main_scan {
        None
    } else {
        Some(
            resolve_assoc_tsv_metadata(
                bed_prefix,
                chrom.clone(),
                pos.clone(),
                snp,
                allele0,
                allele1,
                row_idx.as_deref(),
                m,
            )
            .map_err(PyRuntimeError::new_err)?,
        )
    };
    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;
    if row_flip.len() != m || row_maf.len() != m || row_missing.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row metadata length mismatch: row_flip={}, row_maf={}, row_missing={}, m={m}",
            row_flip.len(),
            row_maf.len(),
            row_missing.len(),
        )));
    }
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    if packed_flat.len() != m_packed * bytes_per_snp {
        return Err(PyRuntimeError::new_err("invalid packed flattened length"));
    }

    let x_cov_flat: Cow<[f64]> = match x_cov.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_cov_arr.iter().copied().collect()),
    };
    let mut base_x = vec![0.0_f64; n * q_base];
    for i in 0..n {
        base_x[i * q_base] = 1.0;
        let src = &x_cov_flat[i * p_cov..(i + 1) * p_cov];
        let dst = &mut base_x[i * q_base + 1..(i + 1) * q_base];
        dst.copy_from_slice(src);
    }

    let mut uniq_chrom = chrom.clone();
    uniq_chrom.sort();
    uniq_chrom.dedup();
    let mut chr_block: HashMap<String, i64> = HashMap::with_capacity(uniq_chrom.len());
    for (i, c) in uniq_chrom.into_iter().enumerate() {
        chr_block.insert(c, i as i64);
    }
    let global_pos: Vec<i64> = (0..m)
        .map(|i| {
            let block = *chr_block.get(&chrom[i]).unwrap_or(&0_i64);
            pos[i].saturating_add(block.saturating_mul(1_000_000_000_000_i64))
        })
        .collect();

    let mut qtn_idx: Vec<usize> = Vec::new();
    let mut qtn_prev_idx: Option<Vec<usize>> = None;
    let route = FarmcpuRoute::from_raw(raw);
    let qtn_threshold_eff = threshold;
    let max_iter_i = max_iter.max(1);
    let qb = qtn_bound.unwrap_or_else(|| {
        let nf = n as f64;
        if nf <= 2.0 {
            1
        } else {
            let den = nf.log10();
            if !den.is_finite() || den <= 0.0 {
                1
            } else {
                ((nf / den).sqrt().floor() as usize).max(1)
            }
        }
    });
    let qb_eff = qb.max(1);
    let nbin_den = nbin.max(1);
    let nbin_step = (qb_eff / nbin_den).max(1);
    let mut nbin_vals: Vec<usize> = (nbin_step..=qb_eff).step_by(nbin_step).collect();
    if nbin_vals.is_empty() {
        nbin_vals.push(qb_eff);
    }
    let mut szbin_i64: Vec<i64> = szbin
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.round() as i64)
        .map(|v| v.max(1))
        .collect();
    if szbin_i64.is_empty() {
        szbin_i64 = vec![500_000_i64, 5_000_000_i64, 50_000_000_i64];
    }
    let final_window_bp = farmcpu_final_window_bp(&szbin_i64);
    let qtn_stage_cap = qtn_bound
        .map(|v| v.max(1))
        .unwrap_or_else(|| qb_eff.saturating_mul(nbin_den).max(qb_eff));
    let stage1_ld_thr = farmcpu_stage1_ld_merge_r2_threshold();
    let final_ld_thr = farmcpu_final_ld_merge_r2_threshold();
    let pool = get_cached_pool(threads)?;

    let (written_rows, qtn_count, qtn_rows_written) = py.detach(|| -> PyResult<(usize, usize, usize)> {
        let mut final_model_cache: Option<(Vec<f64>, usize, Vec<f64>)> = None;
        let mut seen_qtn_idx: HashSet<usize> = HashSet::new();
        let mut qtn_best_score: HashMap<usize, f64> = HashMap::new();
        if matches!(route, FarmcpuRoute::Unified) {
            for it_idx in 0..max_iter_i {
                let (x_qtn, q_total) = build_x_with_qtn_packed(
                    &base_x,
                    n,
                    q_base,
                    &qtn_idx,
                    &packed_flat,
                    bytes_per_snp,
                    row_idx.as_deref(),
                    &sample_idx,
                    row_flip,
                    row_maf,
                )
                .map_err(PyRuntimeError::new_err)?;
                let ixx =
                    pinv_for_row_major_x(&x_qtn, n, q_total).map_err(PyRuntimeError::new_err)?;
                let (femp, _bg_pwald) = farmcpu_scan_packed_marker_pvalues(
                    y,
                    &x_qtn,
                    &ixx,
                    &packed_flat,
                    n_samples,
                    row_flip,
                    row_maf,
                    row_idx.as_deref(),
                    &sample_idx,
                    threads,
                )
                .map_err(PyRuntimeError::new_err)?;
                let mut femp_masked = femp.clone();
                for &idx in seen_qtn_idx.iter() {
                    if idx < femp_masked.len() {
                        femp_masked[idx] = 1.0;
                    }
                }
                let mut candidates: Vec<(f64, usize)> = femp_masked
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(idx, p)| {
                        if p.is_finite() && p <= qtn_threshold_eff {
                            Some((p, idx))
                        } else {
                            None
                        }
                    })
                    .collect();
                candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                if candidates.is_empty() {
                    final_model_cache = Some((x_qtn, q_total, ixx));
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (it_idx + 1, max_iter_i))?;
                            Ok(())
                        })?;
                    }
                    break;
                }
                let opt_lead = farmcpu_best_rem_lead_set(
                    y,
                    &x_qtn,
                    q_total,
                    &femp_masked,
                    &global_pos,
                    &szbin_i64,
                    &nbin_vals,
                    &packed_flat,
                    bytes_per_snp,
                    row_idx.as_deref(),
                    &sample_idx,
                    row_flip,
                    row_maf,
                    pool.as_ref(),
                )
                .map_err(PyRuntimeError::new_err)?;
                let mut qtn_union = qtn_idx.clone();
                qtn_union.extend(farmcpu_window_representatives(
                    candidates.clone(),
                    &chrom,
                    &pos,
                    final_window_bp,
                    qtn_stage_cap,
                ));
                qtn_union.extend(opt_lead.iter().copied());
                qtn_union.sort_unstable();
                qtn_union.dedup();
                let mut qtn_score_map: HashMap<usize, f64> =
                    HashMap::with_capacity(qtn_union.len());
                for &idx in qtn_union.iter() {
                    let mut score = qtn_best_score.get(&idx).copied().unwrap_or_else(|| {
                        let p = femp.get(idx).copied().unwrap_or(1.0);
                        if p.is_finite() {
                            p
                        } else {
                            1.0
                        }
                    });
                    if !score.is_finite() {
                        score = 1.0;
                    }
                    qtn_score_map.insert(idx, score);
                }
                let qtn_next = farmcpu_prune_qtn_by_merged_windows(
                    qtn_union,
                    &qtn_score_map,
                    &chrom,
                    &pos,
                    final_window_bp,
                    &packed_flat,
                    bytes_per_snp,
                    row_idx.as_deref(),
                    &sample_idx,
                    row_flip,
                    row_maf,
                    stage1_ld_thr,
                    false,
                    qtn_stage_cap,
                )
                .map_err(PyRuntimeError::new_err)?;
                for &idx in qtn_next.iter() {
                    seen_qtn_idx.insert(idx);
                    let mut score = qtn_score_map.get(&idx).copied().unwrap_or(1.0);
                    if !score.is_finite() {
                        score = 1.0;
                    }
                    qtn_best_score
                        .entry(idx)
                        .and_modify(|best| {
                            if score.total_cmp(best).is_lt() {
                                *best = score;
                            }
                        })
                        .or_insert(score);
                }

                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (it_idx + 1, max_iter_i))?;
                        Ok(())
                    })?;
                }

                if qtn_next == qtn_idx {
                    final_model_cache = Some((x_qtn, q_total, ixx));
                    break;
                }
                if qtn_prev_idx
                    .as_ref()
                    .map(|prev| *prev == qtn_next)
                    .unwrap_or(false)
                {
                    qtn_idx = qtn_next;
                    break;
                }
                qtn_prev_idx = Some(qtn_idx.clone());
                qtn_idx = qtn_next;
            }
        } else {
            for it_idx in 0..max_iter_i {
                let (x_qtn, q_total) = build_x_with_qtn_packed(
                    &base_x,
                    n,
                    q_base,
                    &qtn_idx,
                    &packed_flat,
                    bytes_per_snp,
                    row_idx.as_deref(),
                    &sample_idx,
                    row_flip,
                    row_maf,
                )
                .map_err(PyRuntimeError::new_err)?;

                let mut xtx = vec![0.0_f64; q_total * q_total];
                for i in 0..n {
                    let row = &x_qtn[i * q_total..(i + 1) * q_total];
                    for a in 0..q_total {
                        let va = row[a];
                        for b in 0..q_total {
                            xtx[a * q_total + b] += va * row[b];
                        }
                    }
                }
                let ixx = pinv_xtx(&xtx, q_total).map_err(PyRuntimeError::new_err)?;
                let (mut femp, qtn_min_p) = scan_packed_full_matrix_reduced(
                    y,
                    &x_qtn,
                    &ixx,
                    &packed_flat,
                    n_samples,
                    row_flip,
                    row_maf,
                    row_idx.as_deref(),
                    &sample_idx,
                    q_base,
                    threads,
                )
                .map_err(PyRuntimeError::new_err)?;
                for (j, &idx) in qtn_idx.iter().enumerate() {
                    if idx < m && qtn_min_p[j].is_finite() {
                        femp[idx] = qtn_min_p[j];
                    }
                }

                let has_signal = femp
                    .iter()
                    .any(|&p| p.is_finite() && p <= qtn_threshold_eff);
                if !has_signal {
                    final_model_cache = Some((x_qtn, q_total, ixx));
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (it_idx + 1, max_iter_i))?;
                            Ok(())
                        })?;
                    }
                    break;
                }

                let opt_lead = farmcpu_best_rem_lead_set(
                    y,
                    &x_qtn,
                    q_total,
                    &femp,
                    &global_pos,
                    &szbin_i64,
                    &nbin_vals,
                    &packed_flat,
                    bytes_per_snp,
                    row_idx.as_deref(),
                    &sample_idx,
                    row_flip,
                    row_maf,
                    pool.as_ref(),
                )
                .map_err(PyRuntimeError::new_err)?;
                let mut qtn_union = qtn_idx.clone();
                qtn_union.extend(opt_lead.into_iter());
                qtn_union.sort_unstable();
                qtn_union.dedup();

                let qtn_next: Vec<usize> = if qtn_union.is_empty() {
                    Vec::new()
                } else {
                    let sample_major = decode_packed_rows_to_sample_major(
                        &packed_flat,
                        bytes_per_snp,
                        &qtn_union,
                        row_idx.as_deref(),
                        &sample_idx,
                        row_flip,
                        row_maf,
                    )
                    .map_err(PyRuntimeError::new_err)?;
                    let pvals: Vec<f64> = qtn_union.iter().map(|&ix| femp[ix]).collect();
                    let keep = farmcpu_super_keep_from_sample_major(
                        &sample_major,
                        n,
                        qtn_union.len(),
                        &pvals,
                        0.7,
                    );
                    qtn_union
                        .iter()
                        .zip(keep.iter())
                        .filter_map(|(&ix, &k)| if k { Some(ix) } else { None })
                        .collect()
                };

                if let Some(cb) = progress_callback.as_ref() {
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (it_idx + 1, max_iter_i))?;
                        Ok(())
                    })?;
                }

                if qtn_next == qtn_idx {
                    final_model_cache = Some((x_qtn, q_total, ixx));
                    break;
                }
                if qtn_prev_idx
                    .as_ref()
                    .map(|prev| *prev == qtn_next)
                    .unwrap_or(false)
                {
                    qtn_idx = qtn_next;
                    break;
                }
                qtn_prev_idx = Some(qtn_idx.clone());
                qtn_idx = qtn_next;
            }
        }

        if matches!(route, FarmcpuRoute::Unified) && !qtn_idx.is_empty() {
            let mut qtn_score_map: HashMap<usize, f64> = HashMap::with_capacity(qtn_idx.len());
            for &idx in qtn_idx.iter() {
                let mut score = qtn_best_score.get(&idx).copied().unwrap_or(1.0);
                if !score.is_finite() {
                    score = 1.0;
                }
                qtn_score_map.insert(idx, score);
            }
            let qtn_final = farmcpu_prune_qtn_by_merged_windows(
                qtn_idx.clone(),
                &qtn_score_map,
                &chrom,
                &pos,
                final_window_bp,
                &packed_flat,
                bytes_per_snp,
                row_idx.as_deref(),
                &sample_idx,
                row_flip,
                row_maf,
                final_ld_thr,
                true,
                qtn_idx.len(),
            )
            .map_err(PyRuntimeError::new_err)?;
            if qtn_final != qtn_idx {
                qtn_idx = qtn_final;
                final_model_cache = None;
            }
        }

        let (x_qtn, _q_total, ixx) = if let Some(cached) = final_model_cache.take() {
            cached
        } else {
            let (x_qtn, q_total) = build_x_with_qtn_packed(
                &base_x,
                n,
                q_base,
                &qtn_idx,
                &packed_flat,
                bytes_per_snp,
                row_idx.as_deref(),
                &sample_idx,
                row_flip,
                row_maf,
            )
            .map_err(PyRuntimeError::new_err)?;
            let mut xtx = vec![0.0_f64; q_total * q_total];
            for i in 0..n {
                let row = &x_qtn[i * q_total..(i + 1) * q_total];
                for a in 0..q_total {
                    let va = row[a];
                    for b in 0..q_total {
                        xtx[a * q_total + b] += va * row[b];
                    }
                }
            }
            let ixx = pinv_xtx(&xtx, q_total).map_err(PyRuntimeError::new_err)?;
            (x_qtn, q_total, ixx)
        };
        let k_qtn = qtn_idx.len();
        let mut qtn_lookup: HashMap<usize, usize> = HashMap::with_capacity(k_qtn);
        for (j, &idx) in qtn_idx.iter().enumerate() {
            qtn_lookup.insert(idx, j);
        }

        let qtn_rows_written = if skip_main_scan && !qtn_idx.is_empty() {
            if let Some(pseudo_path) = pseudo_tsv {
                if !qtn_idx.is_empty() {
                    let qtn_row_lookup = qtn_idx
                        .iter()
                        .map(|&idx| row_idx.as_ref().map(|v| v[idx]).unwrap_or(idx))
                        .collect::<Vec<usize>>();
                    let (q_chrom, q_pos, q_snp, q_allele0, q_allele1) =
                        resolve_assoc_tsv_metadata(
                            bed_prefix,
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            Some(qtn_row_lookup.as_slice()),
                            qtn_idx.len(),
                        )
                        .map_err(PyRuntimeError::new_err)?;
                    let qtn_maf = qtn_idx.iter().map(|&idx| row_maf[idx]).collect::<Vec<f32>>();
                    let qtn_miss = qtn_idx
                        .iter()
                        .map(|&idx| {
                            let src_row = row_idx.as_ref().map(|v| v[idx]).unwrap_or(idx);
                            let row =
                                &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                            packed_row_missing_count_selected(row, n_samples, &sample_idx)
                        })
                        .collect::<Vec<usize>>();
                    write_farmcpu_pseudo_qtn_only(
                        pseudo_path,
                        y,
                        &x_qtn,
                        &ixx,
                        q_base,
                        &qtn_idx,
                        &q_chrom,
                        &q_pos,
                        &q_snp,
                        &q_allele0,
                        &q_allele1,
                        &qtn_maf,
                        &qtn_miss,
                        qtn_row_lookup.as_slice(),
                    )
                    .map_err(PyRuntimeError::new_err)?
                } else {
                    0usize
                }
            } else {
                0usize
            }
        } else {
            let owned_meta = if string_metadata.is_none() {
                Some(resolve_assoc_tsv_metadata(
                    bed_prefix,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    row_idx.as_deref(),
                    m,
                )
                .map_err(PyRuntimeError::new_err)?)
            } else {
                None
            };
            let meta_ref = if let Some(meta) = string_metadata.as_ref() {
                meta
            } else {
                owned_meta.as_ref().expect("owned metadata just assigned")
            };
            let (_chrom_all, _pos_all, snp_all, allele0_all, allele1_all) = meta_ref;
            let mut miss_counts = vec![0usize; m];
            let mut fill_miss = || {
                miss_counts
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(idx, dst)| {
                        let src_row = row_idx.as_ref().map(|v| v[idx]).unwrap_or(idx);
                        let row =
                            &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                        *dst = packed_row_missing_count_selected(row, n_samples, &sample_idx);
                    });
            };
            if let Some(tp) = &pool {
                tp.install(fill_miss);
            } else {
                fill_miss();
            }

            let out_path = out_tsv.to_string();
            let writer = AsyncTsvWriter::with_config(
                &out_path,
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n",
                64 * 1024 * 1024,
                4,
            )
            .map_err(PyRuntimeError::new_err)?;

            let mut qtn_writer_opt: Option<AsyncTsvWriter> = None;
            let mut qtn_path_for_err = String::new();
            if let Some(pseudo_path) = pseudo_tsv {
                if !qtn_idx.is_empty() {
                    qtn_path_for_err = pseudo_path.to_string();
                    qtn_writer_opt = Some(
                        AsyncTsvWriter::with_config(
                            &qtn_path_for_err,
                            b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tsource_row\n",
                            16 * 1024 * 1024,
                            4,
                        )
                        .map_err(PyRuntimeError::new_err)?,
                    );
                }
            }

            let (written_rows, qtn_rows_written) = write_farmcpu_packed_main_scan(
                &writer,
                qtn_writer_opt.as_ref(),
                &qtn_path_for_err,
                y,
                &x_qtn,
                &ixx,
                &packed_flat,
                n_samples,
                row_flip,
                row_maf,
                row_idx.as_deref(),
                &sample_idx,
                threads,
                &chrom,
                &pos,
                snp_all,
                allele0_all,
                allele1_all,
                &miss_counts,
                q_base,
                &qtn_lookup,
                &qtn_idx,
                final_window_bp,
                route.enable_local_window_merge(),
            )
            .map_err(PyRuntimeError::new_err)?;
            writer.finish().map_err(PyRuntimeError::new_err)?;
            if let Some(q_writer) = qtn_writer_opt.take() {
                q_writer
                    .finish()
                    .map_err(|e| PyRuntimeError::new_err(format!("{qtn_path_for_err}: {e}")))?;
            }
            return Ok((written_rows, qtn_idx.len(), qtn_rows_written));
        };
        Ok((0usize, qtn_idx.len(), qtn_rows_written))
    })?;

    Ok((written_rows, qtn_count, qtn_rows_written))
}

#[pyfunction]
#[pyo3(signature = (
    chrom,
    pos,
    snp,
    allele0,
    allele1,
    maf,
    row_missing,
    beta,
    se,
    pwald,
    n_obs,
    df,
    out_tsv,
    qtn_idx=None,
    pseudo_tsv=None
))]
pub fn farmcpu_write_assoc_tsv(
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    maf: PyReadonlyArray1<'_, f32>,
    row_missing: PyReadonlyArray1<'_, f32>,
    beta: PyReadonlyArray1<'_, f64>,
    se: PyReadonlyArray1<'_, f64>,
    pwald: PyReadonlyArray1<'_, f64>,
    n_obs: usize,
    df: usize,
    out_tsv: &str,
    qtn_idx: Option<PyReadonlyArray1<'_, i64>>,
    pseudo_tsv: Option<&str>,
) -> PyResult<(usize, usize)> {
    let m = chrom.len();
    if pos.len() != m || snp.len() != m || allele0.len() != m || allele1.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "FarmCPU metadata length mismatch: chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        )));
    }
    let maf = maf.as_slice()?;
    let row_missing = row_missing.as_slice()?;
    let beta = beta.as_slice()?;
    let se = se.as_slice()?;
    let pwald = pwald.as_slice()?;
    if maf.len() != m
        || row_missing.len() != m
        || beta.len() != m
        || se.len() != m
        || pwald.len() != m
    {
        return Err(PyRuntimeError::new_err(format!(
            "FarmCPU vector length mismatch: m={m}, maf={}, miss={}, beta={}, se={}, pwald={}",
            maf.len(),
            row_missing.len(),
            beta.len(),
            se.len(),
            pwald.len()
        )));
    }

    let _ = (n_obs, df);

    let writer = AsyncTsvWriter::with_config(
        out_tsv,
        b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n",
        4 * 1024 * 1024,
        4,
    )
    .map_err(PyRuntimeError::new_err)?;
    let mut text_buf = String::with_capacity(8192 * 112);

    for i in 0..m {
        let b = beta[i];
        let s = se[i];
        let chisq = chisq_from_beta_se_and_optional_plrt(b, s, None);
        let chisq_txt = format_chisq_value(chisq);
        let pwald_txt = sanitize_assoc_pvalue(b, s, pwald[i]);
        writeln!(
            text_buf,
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}",
            chrom[i],
            pos[i],
            snp[i],
            allele0[i],
            allele1[i],
            maf[i],
            row_missing[i].round() as i64,
            b,
            s,
            chisq_txt,
            pwald_txt
        )
        .map_err(|e| PyRuntimeError::new_err(format!("format row {i} for {out_tsv}: {e}")))?;
        if (i + 1) % 8192 == 0 {
            writer
                .send(text_buf.as_bytes().to_vec())
                .map_err(PyRuntimeError::new_err)?;
            text_buf.clear();
        }
    }
    if !text_buf.is_empty() {
        writer
            .send(text_buf.as_bytes().to_vec())
            .map_err(PyRuntimeError::new_err)?;
    }
    writer.finish().map_err(PyRuntimeError::new_err)?;

    let mut qtn_written = 0usize;
    if let (Some(qidx_arr), Some(pseudo_path)) = (qtn_idx, pseudo_tsv) {
        let qidx_raw = qidx_arr.as_slice()?;
        let mut uniq: Vec<usize> = Vec::with_capacity(qidx_raw.len());
        let mut seen: HashSet<usize> = HashSet::with_capacity(qidx_raw.len());
        for &v in qidx_raw.iter() {
            if v < 0 {
                continue;
            }
            let idx = v as usize;
            if idx >= m {
                continue;
            }
            if seen.insert(idx) {
                uniq.push(idx);
            }
        }
        if !uniq.is_empty() {
            let q_writer = AsyncTsvWriter::with_config(
                pseudo_path,
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n",
                512 * 1024,
                4,
            )
            .map_err(PyRuntimeError::new_err)?;
            let mut q_text_buf = String::with_capacity(uniq.len().min(8192).max(1) * 112);
            for &idx in uniq.iter() {
                let b = beta[idx];
                let s = se[idx];
                let chisq = chisq_from_beta_se_and_optional_plrt(b, s, None);
                let chisq_txt = format_chisq_value(chisq);
                let pwald_txt = sanitize_assoc_pvalue(b, s, pwald[idx]);
                writeln!(
                    q_text_buf,
                    "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}",
                    chrom[idx],
                    pos[idx],
                    snp[idx],
                    allele0[idx],
                    allele1[idx],
                    maf[idx],
                    row_missing[idx].round() as i64,
                    b,
                    s,
                    chisq_txt,
                    pwald_txt
                )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("format qtn row {idx} for {pseudo_path}: {e}"))
                })?;
                if q_text_buf.len() >= 512 * 1024 {
                    q_writer
                        .send(q_text_buf.as_bytes().to_vec())
                        .map_err(PyRuntimeError::new_err)?;
                    q_text_buf.clear();
                }
            }
            if !q_text_buf.is_empty() {
                q_writer
                    .send(q_text_buf.as_bytes().to_vec())
                    .map_err(PyRuntimeError::new_err)?;
            }
            q_writer.finish().map_err(PyRuntimeError::new_err)?;
            qtn_written = uniq.len();
        }
    }

    Ok((m, qtn_written))
}

// =============================================================================
