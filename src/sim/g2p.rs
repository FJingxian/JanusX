use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::eigh::symmetric_eigh_f64_row_major;
use crate::gfcore as core;
use crate::gfcore::{BedSnpIter, HmpSnpIter, TxtSnpIter, VcfSnpIter};
use crate::gfreader::prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window;
use crate::linalg::cholesky_inplace;

#[derive(Clone, Debug)]
struct SimSiteRecord {
    chrom: String,
    chrom_norm: String,
    pos: i32,
    ref_allele: String,
    alt_allele: String,
    maf: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackgroundDist {
    Normal,
    Gamma,
    Laplace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LogicGateMode {
    A,
    Na,
    An,
    Nan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LogicEffectModel {
    Gate,
    CenteredInteraction,
}

#[derive(Clone, Debug)]
struct LogicPoolSpec {
    pool_indices: Vec<usize>,
    mode: LogicGateMode,
}

#[derive(Clone, Debug)]
struct LogicSampledSpec {
    pool_indices: Vec<usize>,
    mode: LogicGateMode,
    size: usize,
}

#[derive(Clone, Debug)]
enum MixedPlannedTerm {
    Additive(usize),
    Logic(LogicSampledSpec),
}

#[derive(Clone, Debug)]
struct CausalTerm {
    members: Vec<usize>,
    mode: Option<LogicGateMode>,
    values: Vec<f64>,
    effect: f64,
    label: String,
}

const SAMPLE_PAR_THRESHOLD: usize = 10_000;
const SAMPLE_PAR_CHUNK: usize = 4096;
const DEFAULT_CAUSAL_SERIES_ALPHA: f64 = 0.9;
const BED_SIM_FAST_WINDOW_MB: usize = 64;

struct G2pSimConfig {
    path_or_prefix: String,
    delimiter: Option<String>,
    maf_threshold: f32,
    causal_maf_min: f32,
    max_missing_rate: f32,
    het_threshold: Option<f32>,
    seed: u64,
    residual_var: f64,
    bg_pve: f64,
    causal_count: usize,
    causal_pve: Option<f64>,
    bim_ranges: Vec<(String, i32, i32)>,
    bim_range_groups: Vec<Vec<(String, i32, i32)>>,
    logic_mode: Option<String>,
    logic_size_weights: Option<Vec<f64>>,
    logic_gate_count: Option<usize>,
    logic_k_min: usize,
    logic_k_max: usize,
    logic_ld_max: f64,
    logic_het_max: f64,
    logic_af_min: f64,
    logic_af_max: f64,
    logic_max_iter: usize,
    logic_window_bp: Option<i32>,
    logic_effect_model: LogicEffectModel,
    snps_only: bool,
    pheno_prefix: Option<String>,
    fixed_effects_path: Option<String>,
    random_effects_path: Option<String>,
    causal_sites_path: Option<String>,
    grm: Option<Vec<f64>>,
    grm_n: Option<usize>,
    trait_name: Option<String>,
    na_rate: f64,
    progress_callback: Option<Py<PyAny>>,
    progress_total_hint: Option<usize>,
    progress_every: usize,
}

struct G2pSimResult {
    sample_ids: Vec<String>,
    phenotype: Vec<f64>,
    trait_name: String,
    causal_sites: Vec<(String, i32, i32)>,
    fixed_rows: Vec<(usize, String, String, String, String, f64)>,
    n_background_sites: usize,
    n_causal_terms: usize,
    bg_pve: f64,
    causal_pve: f64,
    residual_var: f64,
    logic_effect_model: String,
    background_source: String,
    background_factorization: String,
    realized_summary: RealizedSummary,
}

struct RealizedSummary {
    mean_y: f64,
    var_y: f64,
    mean_causal: f64,
    mean_background: f64,
    mean_residual: f64,
    var_causal: f64,
    var_background: f64,
    var_residual: f64,
    cov_causal_background: f64,
    cov_causal_residual: f64,
    cov_background_residual: f64,
    pve_causal: f64,
    pve_background: f64,
    pve_residual: f64,
}

struct PreparedBedFastPath {
    prefix: String,
    sample_ids: Vec<String>,
    sites: Vec<SimSiteRecord>,
    row_source_indices: Vec<usize>,
}

enum SourceReader {
    Bed(BedSnpIter),
    Vcf(VcfSnpIter),
    Hmp(HmpSnpIter),
    Txt(TxtSnpIter),
}

impl SourceReader {
    fn sample_ids(&self) -> &[String] {
        match self {
            Self::Bed(it) => &it.samples,
            Self::Vcf(it) => &it.samples,
            Self::Hmp(it) => &it.samples,
            Self::Txt(it) => &it.samples,
        }
    }

    fn next_row_raw(&mut self) -> Option<(Vec<f32>, core::SiteInfo)> {
        match self {
            Self::Bed(it) => it.next_snp_raw(),
            Self::Vcf(it) => it.next_snp_raw(),
            Self::Hmp(it) => it.next_snp_raw(),
            Self::Txt(it) => it.next_snp(),
        }
    }
}

#[inline]
fn normalize_chrom(chrom: &str) -> String {
    let s = chrom.trim();
    if s.len() >= 3 && s[..3].eq_ignore_ascii_case("chr") {
        s[3..].trim().to_ascii_uppercase()
    } else {
        s.to_ascii_uppercase()
    }
}

#[inline]
fn is_simple_snp_allele(a: &str) -> bool {
    matches!(
        a.trim().to_ascii_uppercase().as_str(),
        "A" | "C" | "G" | "T"
    )
}

#[inline]
fn mean_f64(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

#[inline]
fn variance_f64(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mu = mean_f64(x);
    let mut acc = 0.0_f64;
    for &v in x.iter() {
        let d = v - mu;
        acc += d * d;
    }
    acc / x.len() as f64
}

#[inline]
fn variance_scale_factor(raw_var: f64, target_var: f64) -> f64 {
    if target_var > 0.0 && raw_var > 1e-12 {
        (target_var / raw_var).sqrt()
    } else {
        0.0
    }
}

#[inline]
fn covariance_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return 0.0;
    }
    let mx = mean_f64(x);
    let my = mean_f64(y);
    let mut acc = 0.0_f64;
    for (&vx, &vy) in x.iter().zip(y.iter()) {
        acc += (vx - mx) * (vy - my);
    }
    acc / x.len() as f64
}

#[inline]
fn centered_row_to_owned_f64(row: &[f32]) -> Vec<f64> {
    if row.is_empty() {
        return Vec::new();
    }
    let mu = row.iter().map(|&v| v as f64).sum::<f64>() / row.len() as f64;
    let mut out = Vec::with_capacity(row.len());
    for &v in row.iter() {
        out.push(v as f64 - mu);
    }
    out
}

#[inline]
fn axpy_inplace(dst: &mut [f64], src: &[f64], alpha: f64) {
    debug_assert_eq!(dst.len(), src.len());
    if alpha == 0.0 || dst.is_empty() {
        return;
    }
    if dst.len() >= SAMPLE_PAR_THRESHOLD {
        dst.par_chunks_mut(SAMPLE_PAR_CHUNK)
            .zip(src.par_chunks(SAMPLE_PAR_CHUNK))
            .for_each(|(dst_chunk, src_chunk)| {
                for (d, &s) in dst_chunk.iter_mut().zip(src_chunk.iter()) {
                    *d += alpha * s;
                }
            });
    } else {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d += alpha * s;
        }
    }
}

#[inline]
fn logic_effect_model_name(model: LogicEffectModel) -> &'static str {
    match model {
        LogicEffectModel::Gate => "gate",
        LogicEffectModel::CenteredInteraction => "centered_interaction",
    }
}

fn validate_grm_matrix(grm: &[f64], n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("GRM must not be empty.".to_string());
    }
    if grm.len() != n.saturating_mul(n) {
        return Err(format!(
            "GRM payload length mismatch: got {}, expected {}",
            grm.len(),
            n.saturating_mul(n)
        ));
    }
    for i in 0..n {
        if !grm[i * n + i].is_finite() {
            return Err(format!(
                "GRM diagonal contains non-finite value at row {i}."
            ));
        }
        for j in 0..=i {
            let a = grm[i * n + j];
            let b = grm[j * n + i];
            if !a.is_finite() || !b.is_finite() {
                return Err(format!("GRM contains non-finite value at ({i}, {j})."));
            }
            if (a - b).abs() > 1e-8_f64 {
                return Err(format!("GRM must be symmetric; mismatch at ({i}, {j})."));
            }
        }
    }
    Ok(())
}

fn spd_cholesky_with_jitter(matrix: &[f64], dim: usize, label: &str) -> Result<Vec<f64>, String> {
    if matrix.len() != dim.saturating_mul(dim) {
        return Err(format!(
            "{label} shape mismatch: got {}, expected {}",
            matrix.len(),
            dim.saturating_mul(dim)
        ));
    }
    let mut chol = matrix.to_vec();
    if cholesky_inplace(&mut chol, dim).is_some() {
        return Ok(chol);
    }
    let diag_scale = (0..dim)
        .map(|i| matrix[i * dim + i].abs())
        .fold(1.0_f64, f64::max);
    for rel in [1e-12_f64, 1e-10_f64, 1e-8_f64, 1e-6_f64, 1e-4_f64] {
        let jitter = diag_scale * rel;
        chol.copy_from_slice(matrix);
        for i in 0..dim {
            chol[i * dim + i] += jitter;
        }
        if cholesky_inplace(&mut chol, dim).is_some() {
            return Ok(chol);
        }
    }
    Err(format!("{label} is not SPD even after diagonal jitter"))
}

#[derive(Clone, Debug)]
enum SamplingFactor {
    CholeskyLower(Vec<f64>),
    DenseSquareRoot(Vec<f64>),
}

fn lower_triangular_matvec(l: &[f64], dim: usize, rhs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(l.len(), dim.saturating_mul(dim));
    debug_assert_eq!(rhs.len(), dim);
    let mut out = vec![0.0_f64; dim];
    for i in 0..dim {
        let mut acc = 0.0_f64;
        for j in 0..=i {
            acc += l[i * dim + j] * rhs[j];
        }
        out[i] = acc;
    }
    out
}

fn dense_row_major_matvec(a: &[f64], dim: usize, rhs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), dim.saturating_mul(dim));
    debug_assert_eq!(rhs.len(), dim);
    let mut out = vec![0.0_f64; dim];
    for i in 0..dim {
        let row = &a[i * dim..(i + 1) * dim];
        let mut acc = 0.0_f64;
        for (v, z) in row.iter().zip(rhs.iter()) {
            acc += *v * *z;
        }
        out[i] = acc;
    }
    out
}

fn sample_factor_matvec(factor: &SamplingFactor, dim: usize, rhs: &[f64]) -> Vec<f64> {
    match factor {
        SamplingFactor::CholeskyLower(l) => lower_triangular_matvec(l, dim, rhs),
        SamplingFactor::DenseSquareRoot(l) => dense_row_major_matvec(l, dim, rhs),
    }
}

fn diag_trace_f64(matrix: &[f64], dim: usize) -> f64 {
    (0..dim).map(|i| matrix[i * dim + i]).sum::<f64>()
}

#[inline]
fn processed_row_minor_allele_frequency(row: &[f32]) -> f32 {
    if row.is_empty() {
        return 0.0_f32;
    }
    let mut n_obs = 0usize;
    let mut sum = 0.0_f64;
    for &v in row.iter() {
        if v.is_finite() && v >= 0.0_f32 {
            n_obs += 1;
            sum += v as f64;
        }
    }
    if n_obs == 0 {
        return 0.0_f32;
    }
    let af = (0.5_f64 * (sum / n_obs as f64)).clamp(0.0_f64, 1.0_f64);
    af.min(1.0_f64 - af) as f32
}

#[inline]
fn site_passes_causal_maf(site: &SimSiteRecord, causal_maf_min: f32) -> bool {
    let thr = causal_maf_min.max(0.0_f32);
    site.maf.is_finite() && site.maf + 1e-8_f32 >= thr
}

fn eigh_psd_square_root_with_clipped_negatives(
    matrix: &[f64],
    dim: usize,
    label: &str,
) -> Result<(Vec<f64>, f64), String> {
    let (evals, evecs, _backend) = symmetric_eigh_f64_row_major(matrix, dim)
        .map_err(|e| format!("{label} eigh fallback failed: {e}"))?;
    if evals.len() != dim || evecs.len() != dim.saturating_mul(dim) {
        return Err(format!(
            "{label} eigh fallback shape mismatch: evals={}, evecs={}, dim={dim}",
            evals.len(),
            evecs.len(),
        ));
    }

    let mut root = vec![0.0_f64; dim * dim];
    let mut trace_psd = 0.0_f64;
    for k in 0..dim {
        let lambda = evals[k];
        if !lambda.is_finite() {
            return Err(format!(
                "{label} eigh fallback produced non-finite eigenvalue at index {k}: {lambda}"
            ));
        }
        let lambda_clip = lambda.max(0.0_f64);
        trace_psd += lambda_clip;
        if lambda_clip <= 0.0_f64 {
            continue;
        }
        let scale = lambda_clip.sqrt();
        for i in 0..dim {
            root[i * dim + k] = evecs[i * dim + k] * scale;
        }
    }
    if !trace_psd.is_finite() || trace_psd <= 0.0_f64 {
        return Err(format!(
            "{label} eigh fallback produced non-positive PSD trace after clipping: {trace_psd}"
        ));
    }
    Ok((root, trace_psd))
}

fn build_sampling_factor_with_fallback(
    matrix: &[f64],
    dim: usize,
    label: &str,
) -> Result<(SamplingFactor, f64), String> {
    let trace = diag_trace_f64(matrix, dim);
    if let Ok(chol) = spd_cholesky_with_jitter(matrix, dim, label) {
        if !trace.is_finite() || trace <= 0.0_f64 {
            return Err(format!(
                "{label} trace must be finite and > 0 for trace-scaled sampling."
            ));
        }
        return Ok((SamplingFactor::CholeskyLower(chol), trace));
    }
    let (root, trace_psd) = eigh_psd_square_root_with_clipped_negatives(matrix, dim, label)?;
    Ok((SamplingFactor::DenseSquareRoot(root), trace_psd))
}

fn sampling_factor_method_name(factor: &SamplingFactor) -> &'static str {
    match factor {
        SamplingFactor::CholeskyLower(_) => "cholesky",
        SamplingFactor::DenseSquareRoot(_) => "eigh-clipped",
    }
}

fn build_causal_series_effects(count: usize, alpha: f64, rng: &mut StdRng) -> Vec<f64> {
    let mut effects = Vec::with_capacity(count);
    let mut current = alpha;
    for _ in 0..count {
        effects.push(current);
        current *= alpha;
    }
    effects.shuffle(rng);
    effects
}

fn sample_gaussian_noise_with_variance(n: usize, variance: f64, rng: &mut StdRng) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    if variance <= 0.0 || n == 0 {
        return out;
    }
    let sd = variance.sqrt();
    for v in out.iter_mut() {
        *v = StandardNormal.sample(rng);
        *v *= sd;
    }
    out
}

fn sample_background_effects_from_grm_trace_scaled(
    grm: &[f64],
    n: usize,
    target_var: f64,
    rng: &mut StdRng,
) -> Result<(Vec<f64>, String), String> {
    validate_grm_matrix(grm, n)?;
    if target_var <= 0.0 || n == 0 {
        return Ok((vec![0.0_f64; n], "none".to_string()));
    }
    let (factor, trace_for_scale) = build_sampling_factor_with_fallback(grm, n, "GRM")?;
    let mut z = vec![0.0_f64; n];
    for zi in z.iter_mut() {
        *zi = StandardNormal.sample(rng);
    }
    let mut out = sample_factor_matvec(&factor, n, &z);
    let scale = ((n as f64) * target_var / trace_for_scale).sqrt();
    for v in out.iter_mut() {
        *v *= scale;
    }
    Ok((out, sampling_factor_method_name(&factor).to_string()))
}

fn realized_share(component_var: f64, total_var: f64) -> f64 {
    if total_var > 1e-12 {
        component_var / total_var
    } else {
        0.0
    }
}

fn build_realized_summary(
    y: &[f64],
    causal: &[f64],
    background: &[f64],
    residual: &[f64],
) -> RealizedSummary {
    let var_y = variance_f64(y);
    let var_causal = variance_f64(causal);
    let var_background = variance_f64(background);
    let var_residual = variance_f64(residual);
    RealizedSummary {
        mean_y: mean_f64(y),
        var_y,
        mean_causal: mean_f64(causal),
        mean_background: mean_f64(background),
        mean_residual: mean_f64(residual),
        var_causal,
        var_background,
        var_residual,
        cov_causal_background: covariance_f64(causal, background),
        cov_causal_residual: covariance_f64(causal, residual),
        cov_background_residual: covariance_f64(background, residual),
        pve_causal: realized_share(var_causal, var_y),
        pve_background: realized_share(var_background, var_y),
        pve_residual: realized_share(var_residual, var_y),
    }
}

fn collapse_to_logic_bin01(row: &[f32], het_max: f64) -> Option<Vec<u8>> {
    if row.is_empty() {
        return None;
    }
    let mut valid: Vec<u8> = Vec::with_capacity(row.len());
    let mut valid_idx: Vec<usize> = Vec::with_capacity(row.len());
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            continue;
        }
        let r = v.round();
        let g = if r <= 0.0 {
            0u8
        } else if r >= 2.0 {
            2u8
        } else {
            1u8
        };
        valid.push(g);
        valid_idx.push(i);
    }
    if valid.is_empty() {
        return None;
    }
    let het = valid.iter().filter(|&&g| g == 1).count() as f64 / valid.len() as f64;
    if het > het_max {
        return None;
    }
    let c0 = valid.iter().filter(|&&g| g == 0).count();
    let c2 = valid.iter().filter(|&&g| g == 2).count();
    let mode02 = if c2 > c0 { 2u8 } else { 0u8 };
    let mut out = vec![mode02; row.len()];
    for (&idx, &g) in valid_idx.iter().zip(valid.iter()) {
        out[idx] = if g == 1 { mode02 } else { g };
    }
    Some(
        out.into_iter()
            .map(|g| if g > 0 { 1u8 } else { 0u8 })
            .collect(),
    )
}

#[inline]
fn binary_r2(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i] as f64;
        sb += b[i] as f64;
    }
    let ma = sa / n as f64;
    let mb = sb / n as f64;
    let mut cov = 0.0_f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    for i in 0..n {
        let da = a[i] as f64 - ma;
        let db = b[i] as f64 - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 1e-12 || vb <= 1e-12 {
        return 1.0;
    }
    let r = cov / (va.sqrt() * vb.sqrt());
    let r2 = r * r;
    if r2.is_finite() {
        r2.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

#[inline]
fn logic_mode_code(mode: LogicGateMode) -> &'static str {
    match mode {
        LogicGateMode::A => "a",
        LogicGateMode::Na => "na",
        LogicGateMode::An => "an",
        LogicGateMode::Nan => "nan",
    }
}

#[inline]
fn logic_member_negated(mode: LogicGateMode, idx: usize) -> bool {
    match mode {
        LogicGateMode::A | LogicGateMode::Na => false,
        LogicGateMode::An => idx > 0,
        LogicGateMode::Nan => true,
    }
}

#[inline]
fn logic_output_negated(mode: LogicGateMode) -> bool {
    matches!(mode, LogicGateMode::Na)
}

fn logic_gate_literal_rows(rows: &[Vec<u8>], mode: LogicGateMode) -> Result<Vec<Vec<u8>>, String> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let n = rows[0].len();
    let mut out = Vec::with_capacity(rows.len());
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != n {
            return Err("logic gate member rows have inconsistent sample lengths".to_string());
        }
        let negated = logic_member_negated(mode, row_idx);
        let literal = row
            .iter()
            .map(|&v| {
                let bit = if v == 0 { 0u8 } else { 1u8 };
                if negated {
                    1u8 - bit
                } else {
                    bit
                }
            })
            .collect::<Vec<_>>();
        out.push(literal);
    }
    Ok(out)
}

fn logic_gate_indicator_from_literals(rows: &[Vec<u8>], output_negated: bool) -> Vec<u8> {
    if rows.is_empty() {
        return Vec::new();
    }
    let n = rows[0].len();
    let mut raw = vec![0u8; n];
    for i in 0..n {
        let mut ok = true;
        for row in rows.iter() {
            if row[i] == 0 {
                ok = false;
                break;
            }
        }
        raw[i] = if ok { 1u8 } else { 0u8 };
    }
    if output_negated {
        for v in raw.iter_mut() {
            *v = 1u8 - *v;
        }
    }
    raw
}

#[cfg(test)]
fn logic_gate_indicator(rows: &[Vec<u8>], mode: LogicGateMode) -> Result<Vec<u8>, String> {
    let literal_rows = logic_gate_literal_rows(rows, mode)?;
    Ok(logic_gate_indicator_from_literals(
        literal_rows.as_slice(),
        logic_output_negated(mode),
    ))
}

fn logic_gate_centered_values(indicator: &[u8]) -> Vec<f64> {
    let raw = indicator.iter().map(|&v| v as f64).collect::<Vec<_>>();
    let mu = mean_f64(&raw);
    raw.into_iter().map(|v| v - mu).collect()
}

fn solve_linear_system(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    if a.len() != n * n || b.len() != n {
        return None;
    }
    for k in 0..n {
        let mut pivot = k;
        let mut pivot_abs = a[k * n + k].abs();
        for i in (k + 1)..n {
            let cand = a[i * n + k].abs();
            if cand > pivot_abs {
                pivot = i;
                pivot_abs = cand;
            }
        }
        if pivot_abs <= 1e-12 {
            return None;
        }
        if pivot != k {
            for j in 0..n {
                a.swap(k * n + j, pivot * n + j);
            }
            b.swap(k, pivot);
        }
        let diag = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / diag;
            if factor == 0.0 {
                continue;
            }
            a[i * n + k] = 0.0;
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    let mut x = vec![0.0_f64; n];
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut rhs = b[i];
        for j in (i + 1)..n {
            rhs -= a[i * n + j] * x[j];
        }
        let diag = a[i * n + i];
        if diag.abs() <= 1e-12 {
            return None;
        }
        x[i] = rhs / diag;
    }
    Some(x)
}

fn residualize_logic_indicator_against_main_effects(
    indicator: &[u8],
    member_rows: &[Vec<u8>],
) -> Result<Vec<f64>, String> {
    if indicator.is_empty() {
        return Err("logic indicator values are empty".to_string());
    }
    let n = indicator.len();
    let p = 1usize + member_rows.len();
    for row in member_rows.iter() {
        if row.len() != n {
            return Err("logic gate member rows have inconsistent sample lengths".to_string());
        }
    }

    let mut gram = vec![0.0_f64; p * p];
    let mut rhs = vec![0.0_f64; p];
    let ridge = 1e-8_f64;
    for i in 0..n {
        let yi = indicator[i] as f64;
        rhs[0] += yi;
        gram[0] += 1.0;
        for a in 0..member_rows.len() {
            let xa = member_rows[a][i] as f64;
            rhs[a + 1] += xa * yi;
            gram[(a + 1) * p] += xa;
            gram[a + 1] += xa;
            gram[(a + 1) * p + (a + 1)] += xa * xa;
            for b in (a + 1)..member_rows.len() {
                let xb = member_rows[b][i] as f64;
                let v = xa * xb;
                gram[(a + 1) * p + (b + 1)] += v;
                gram[(b + 1) * p + (a + 1)] += v;
            }
        }
    }
    for d in 0..p {
        gram[d * p + d] += ridge;
    }
    let beta = solve_linear_system(gram, rhs, p)
        .ok_or_else(|| "failed to solve centered-interaction projection system".to_string())?;
    let mut resid = vec![0.0_f64; n];
    for i in 0..n {
        let mut fit = beta[0];
        for (j, row) in member_rows.iter().enumerate() {
            fit += beta[j + 1] * row[i] as f64;
        }
        resid[i] = indicator[i] as f64 - fit;
    }
    Ok(resid)
}

fn term_label(sites: &[SimSiteRecord], members: &[usize], mode: Option<LogicGateMode>) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(members.len());
    for (member_idx, &idx) in members.iter().enumerate() {
        let s = &sites[idx];
        let base = format!("{}_{}[{}>{}]", s.chrom, s.pos, s.ref_allele, s.alt_allele);
        parts.push(
            if mode
                .map(|gate_mode| logic_member_negated(gate_mode, member_idx))
                .unwrap_or(false)
            {
                format!("!{base}")
            } else {
                base
            },
        );
    }
    if let Some(gate_mode) = mode {
        let expr = parts.join("&");
        if logic_output_negated(gate_mode) {
            format!("!({expr})")
        } else {
            expr
        }
    } else {
        parts.join("")
    }
}

fn normalize_plink_prefix_local(p: &str) -> String {
    let low = p.to_ascii_lowercase();
    for ext in [".bed", ".bim", ".fam"] {
        if low.ends_with(ext) {
            let keep = p.len().saturating_sub(ext.len());
            return p[..keep].to_string();
        }
    }
    p.to_string()
}

fn open_source_reader(
    path_or_prefix: &str,
    delimiter: Option<&str>,
) -> Result<SourceReader, String> {
    let p = path_or_prefix.trim();
    if p.is_empty() {
        return Err("path_or_prefix must not be empty".to_string());
    }
    let low = p.to_ascii_lowercase();
    if low.ends_with(".vcf") || low.ends_with(".vcf.gz") {
        return Ok(SourceReader::Vcf(VcfSnpIter::new_with_fill(
            p, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    if low.ends_with(".hmp") || low.ends_with(".hmp.gz") {
        return Ok(SourceReader::Hmp(HmpSnpIter::new_with_fill(
            p, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    if low.ends_with(".txt")
        || low.ends_with(".tsv")
        || low.ends_with(".csv")
        || low.ends_with(".npy")
        || low.ends_with(".bin")
    {
        return Ok(SourceReader::Txt(TxtSnpIter::new(p, delimiter)?));
    }
    let prefix = normalize_plink_prefix_local(p);
    if Path::new(&(prefix.clone() + ".bed")).exists()
        && Path::new(&(prefix.clone() + ".bim")).exists()
        && Path::new(&(prefix.clone() + ".fam")).exists()
    {
        return Ok(SourceReader::Bed(BedSnpIter::new_with_fill(
            &prefix, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    if Path::new(&(p.to_string() + ".npy")).exists()
        || Path::new(&(p.to_string() + ".txt")).exists()
        || Path::new(&(p.to_string() + ".tsv")).exists()
        || Path::new(&(p.to_string() + ".csv")).exists()
        || Path::new(&(p.to_string() + ".bin")).exists()
    {
        return Ok(SourceReader::Txt(TxtSnpIter::new(p, delimiter)?));
    }
    Err(
        "Unable to infer genotype input type. Provide a VCF/HMP path, a PLINK prefix, or a FILE matrix path/prefix."
            .to_string(),
    )
}

fn iterate_filtered_rows<F>(
    path_or_prefix: &str,
    delimiter: Option<&str>,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: Option<f32>,
    snps_only: bool,
    mut callback: F,
) -> Result<Vec<String>, String>
where
    F: FnMut(usize, &[f32], &core::SiteInfo) -> Result<(), String>,
{
    let mut reader = open_source_reader(path_or_prefix, delimiter)?;
    let sample_ids = reader.sample_ids().to_vec();
    let mut kept = 0usize;
    while let Some((mut row, mut site)) = reader.next_row_raw() {
        let keep = core::process_snp_row(
            &mut row,
            &mut site.ref_allele,
            &mut site.alt_allele,
            maf_threshold,
            max_missing_rate,
            true,
            het_threshold.is_some(),
            het_threshold.unwrap_or(1.0_f32),
        );
        if !keep {
            continue;
        }
        if snps_only
            && (!is_simple_snp_allele(&site.ref_allele) || !is_simple_snp_allele(&site.alt_allele))
        {
            continue;
        }
        callback(kept, &row, &site)?;
        kept = kept.saturating_add(1);
    }
    Ok(sample_ids)
}

#[inline]
fn is_plink_bed_prefix_available(path_or_prefix: &str) -> Option<String> {
    let prefix = normalize_plink_prefix_local(path_or_prefix.trim());
    if prefix.is_empty() {
        return None;
    }
    if Path::new(&(prefix.clone() + ".bed")).exists()
        && Path::new(&(prefix.clone() + ".bim")).exists()
        && Path::new(&(prefix.clone() + ".fam")).exists()
    {
        Some(prefix)
    } else {
        None
    }
}

fn build_kept_bed_sites(
    prefix: &str,
    site_keep: &[bool],
    kept_alt_freq: &[f32],
) -> Result<Vec<SimSiteRecord>, String> {
    let sites_all = core::read_bim(prefix)?;
    if sites_all.len() != site_keep.len() {
        return Err(format!(
            "BED/BIM keep-mask mismatch: keep_mask={}, bim_rows={}",
            site_keep.len(),
            sites_all.len()
        ));
    }
    let mut kept_idx = 0usize;
    let mut out = Vec::with_capacity(kept_alt_freq.len());
    for (src_idx, site) in sites_all.into_iter().enumerate() {
        if !site_keep[src_idx] {
            continue;
        }
        let alt_freq = kept_alt_freq
            .get(kept_idx)
            .copied()
            .ok_or_else(|| format!("kept alt-frequency missing for kept site index {kept_idx}"))?;
        let flip = alt_freq > 0.5_f32;
        let (ref_allele, alt_allele) = if flip {
            (site.alt_allele, site.ref_allele)
        } else {
            (site.ref_allele, site.alt_allele)
        };
        out.push(SimSiteRecord {
            chrom: site.chrom.clone(),
            chrom_norm: normalize_chrom(&site.chrom),
            pos: site.pos,
            ref_allele,
            alt_allele,
            maf: alt_freq.min(1.0_f32 - alt_freq).max(0.0_f32),
        });
        kept_idx = kept_idx.saturating_add(1);
    }
    if kept_idx != kept_alt_freq.len() {
        return Err(format!(
            "kept alt-frequency count mismatch after BIM filter: used={}, expected={}",
            kept_idx,
            kept_alt_freq.len()
        ));
    }
    Ok(out)
}

fn try_prepare_bed_fast_path(
    path_or_prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: Option<f32>,
    snps_only: bool,
    allow_fast_path: bool,
) -> Result<Option<PreparedBedFastPath>, String> {
    if !allow_fast_path {
        return Ok(None);
    }
    let Some(prefix) = is_plink_bed_prefix_available(path_or_prefix) else {
        return Ok(None);
    };
    let sample_ids = core::read_fam(&prefix)?;
    if sample_ids.is_empty() {
        return Err("no samples found in PLINK input after BED fast-path prep".to_string());
    }
    let prepared = prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window(
        &prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold.unwrap_or(1.0_f32),
        snps_only,
        None,
        true,
        Some(BED_SIM_FAST_WINDOW_MB),
        rayon::current_num_threads().max(1),
    )?;
    if prepared.n_samples != sample_ids.len() {
        return Err(format!(
            "BED fast-path sample count mismatch: fam={}, prepared={}",
            sample_ids.len(),
            prepared.n_samples
        ));
    }
    let sites = build_kept_bed_sites(
        &prefix,
        prepared.site_keep.as_slice(),
        prepared.maf.as_slice(),
    )?;
    Ok(Some(PreparedBedFastPath {
        prefix,
        sample_ids,
        sites,
        row_source_indices: prepared.row_source_indices,
    }))
}

fn orient_and_impute_minor_coded_row_inplace(
    row: &mut [f32],
    raw_alt_sum: f64,
    non_missing: usize,
) -> Result<(), String> {
    if non_missing == 0 {
        return Err(
            "cannot orient/impute a genotype row with zero non-missing samples".to_string(),
        );
    }
    let flip = raw_alt_sum > non_missing as f64;
    if flip {
        for v in row.iter_mut() {
            if *v >= 0.0 {
                *v = 2.0_f32 - *v;
            }
        }
    }
    let oriented_alt_sum = if flip {
        2.0_f64 * non_missing as f64 - raw_alt_sum
    } else {
        raw_alt_sum
    };
    let imputed = (oriented_alt_sum / non_missing as f64) as f32;
    for v in row.iter_mut() {
        if *v < 0.0 {
            *v = imputed;
        }
    }
    Ok(())
}

fn decode_bed_rows_by_kept_index(
    prefix: &str,
    row_source_indices: &[usize],
    kept_indices: &[usize],
    progress_callback: Option<&Py<PyAny>>,
    progress_stage: &str,
    progress_done_offset: usize,
    progress_total: usize,
    progress_every: usize,
) -> Result<HashMap<usize, Vec<f32>>, String> {
    if kept_indices.is_empty() {
        return Ok(HashMap::new());
    }
    let mut kept_unique = kept_indices.to_vec();
    kept_unique.sort_unstable();
    kept_unique.dedup();
    let mut bed_iter = BedSnpIter::new_for_grm_window(prefix, BED_SIM_FAST_WINDOW_MB)?;
    let mut row_buf = vec![-9.0_f32; bed_iter.n_samples()];
    let mut out = HashMap::with_capacity(kept_unique.len());
    let mut last_notified = progress_done_offset;
    for (seen, kept_idx) in kept_unique.into_iter().enumerate() {
        let src_row = *row_source_indices.get(kept_idx).ok_or_else(|| {
            format!("kept BED row index out of range during causal decode: {kept_idx}")
        })?;
        bed_iter.ensure_window_for_snp(src_row)?;
        let (raw_alt_sum, non_missing) = bed_iter
            .decode_snp_raw_into_with_stats_at(src_row, row_buf.as_mut_slice())
            .ok_or_else(|| format!("failed to decode BED source row {src_row}"))?;
        let mut row = row_buf.clone();
        orient_and_impute_minor_coded_row_inplace(row.as_mut_slice(), raw_alt_sum, non_missing)?;
        out.insert(kept_idx, row);
        if progress_total > 0 {
            let done_now = progress_done_offset.saturating_add(seen.saturating_add(1));
            g2p_progress_notify(
                progress_callback,
                progress_stage,
                done_now,
                progress_total,
                progress_every,
                &mut last_notified,
                false,
            )?;
        }
    }
    if progress_total > 0 {
        let done_now = progress_done_offset.saturating_add(out.len());
        g2p_progress_notify(
            progress_callback,
            progress_stage,
            done_now,
            progress_total,
            progress_every,
            &mut last_notified,
            true,
        )?;
    }
    Ok(out)
}

fn site_in_range(site: &SimSiteRecord, range: &(String, i32, i32)) -> bool {
    site.chrom_norm == normalize_chrom(&range.0) && site.pos >= range.1 && site.pos <= range.2
}

fn build_range_pools(
    sites: &[SimSiteRecord],
    ranges: &[(String, i32, i32)],
) -> Result<Vec<Vec<usize>>, String> {
    let mut out = Vec::with_capacity(ranges.len());
    for (ri, rg) in ranges.iter().enumerate() {
        let mut idx = Vec::new();
        for (i, site) in sites.iter().enumerate() {
            if site_in_range(site, rg) {
                idx.push(i);
            }
        }
        if idx.is_empty() {
            return Err(format!(
                "bimrange[{ri}] has no eligible sites after QC: {}:{}-{}",
                rg.0, rg.1, rg.2
            ));
        }
        out.push(idx);
    }
    Ok(out)
}

fn build_range_group_pools(
    sites: &[SimSiteRecord],
    range_groups: &[Vec<(String, i32, i32)>],
) -> Result<Vec<Vec<usize>>, String> {
    let mut out = Vec::with_capacity(range_groups.len());
    for (gi, group) in range_groups.iter().enumerate() {
        if group.is_empty() {
            return Err(format!("causal_group[{gi}] must contain at least one range"));
        }
        let mut idx: Vec<usize> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for rg in group.iter() {
            for (i, site) in sites.iter().enumerate() {
                if site_in_range(site, rg) && seen.insert(i) {
                    idx.push(i);
                }
            }
        }
        if idx.is_empty() {
            let ranges_txt = group
                .iter()
                .map(|rg| format!("{}:{}-{}", rg.0, rg.1, rg.2))
                .collect::<Vec<String>>()
                .join(";");
            return Err(format!(
                "causal_group[{gi}] has no eligible sites after QC: {ranges_txt}"
            ));
        }
        out.push(idx);
    }
    Ok(out)
}

fn build_constraint_pools(
    sites: &[SimSiteRecord],
    ranges: &[(String, i32, i32)],
    range_groups: &[Vec<(String, i32, i32)>],
) -> Result<Vec<Vec<usize>>, String> {
    if !ranges.is_empty() && !range_groups.is_empty() {
        return Err("bim_ranges and bim_range_groups cannot both be set".to_string());
    }
    if !range_groups.is_empty() {
        build_range_group_pools(sites, range_groups)
    } else {
        build_range_pools(sites, ranges)
    }
}

fn sample_without_replacement(
    pool: &[usize],
    k: usize,
    rng: &mut StdRng,
) -> Result<Vec<usize>, String> {
    if k > pool.len() {
        return Err(format!(
            "cannot sample {k} unique sites from pool size {}",
            pool.len()
        ));
    }
    let mut tmp = pool.to_vec();
    tmp.shuffle(rng);
    tmp.truncate(k);
    Ok(tmp)
}

fn select_additive_indices(
    sites: &[SimSiteRecord],
    causal_count: usize,
    constraint_pools: &[Vec<usize>],
    causal_maf_min: f32,
    rng: &mut StdRng,
) -> Result<Vec<usize>, String> {
    if sites.is_empty() {
        return Err("no eligible sites remain after QC".to_string());
    }
    if causal_count == 0 && constraint_pools.is_empty() {
        return Ok(Vec::new());
    }
    if causal_count < constraint_pools.len() {
        return Err(format!(
            "causal_count must be >= number of causal constraint groups: causal_count={}, groups={}",
            causal_count,
            constraint_pools.len()
        ));
    }
    let mut selected: Vec<usize> = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();
    for (ri, pool) in constraint_pools.iter().enumerate() {
        let avail: Vec<usize> = pool
            .iter()
            .copied()
            .filter(|idx| {
                !used.contains(idx) && site_passes_causal_maf(&sites[*idx], causal_maf_min)
            })
            .collect();
        if avail.is_empty() {
            return Err(format!(
                "causal constraint group[{ri}] has no causal-site candidates after lmaf filtering: lmaf={:.4}",
                causal_maf_min
            ));
        }
        let pick = avail[rng.random_range(0..avail.len())];
        used.insert(pick);
        selected.push(pick);
    }
    let target = causal_count;
    let eligible_all: Vec<usize> = sites
        .iter()
        .enumerate()
        .filter_map(|(idx, site)| site_passes_causal_maf(site, causal_maf_min).then_some(idx))
        .collect();
    if target > eligible_all.len() {
        return Err(format!(
            "requested causal sites exceed lmaf-filtered eligible site count: target={target}, eligible={}, lmaf={:.4}",
            eligible_all.len(),
            causal_maf_min,
        ));
    }
    if selected.len() < target {
        let mut rest: Vec<usize> = eligible_all
            .into_iter()
            .filter(|idx| !used.contains(idx))
            .collect();
        rest.shuffle(rng);
        for idx in rest.into_iter().take(target - selected.len()) {
            used.insert(idx);
            selected.push(idx);
        }
    }
    if selected.len() != target {
        return Err(format!(
            "unable to draw enough unique causal sites: target={target}, got={}",
            selected.len()
        ));
    }
    selected.sort_unstable();
    Ok(selected)
}

fn reservoir_sample(mut pool: Vec<usize>, cap: usize, rng: &mut StdRng) -> Vec<usize> {
    if pool.len() <= cap {
        return pool;
    }
    pool.shuffle(rng);
    pool.truncate(cap);
    pool
}

fn logic_mode_from_str(mode: &str, rng: &mut StdRng) -> Result<LogicGateMode, String> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "a" | "and" => Ok(LogicGateMode::A),
        "na" => Ok(LogicGateMode::Na),
        "an" => Ok(LogicGateMode::An),
        "nan" => Ok(LogicGateMode::Nan),
        "r" => Ok(match rng.random_range(0..4) {
            0 => LogicGateMode::A,
            1 => LogicGateMode::Na,
            2 => LogicGateMode::An,
            _ => LogicGateMode::Nan,
        }),
        other => Err(format!("unsupported logic mode: {other}")),
    }
}

fn validate_logic_size_weights(weights: &[f64]) -> Result<(), String> {
    if weights.is_empty() {
        return Err("logic_size_weights must not be empty".to_string());
    }
    let mut any_positive = false;
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(format!(
                "logic_size_weights[{}] must be finite and >= 0, got {}",
                i, w
            ));
        }
        if w > 0.0 {
            any_positive = true;
        }
    }
    if !any_positive {
        return Err("logic_size_weights must contain at least one positive entry".to_string());
    }
    Ok(())
}

fn sample_term_size_with_limit(
    weights: &[f64],
    max_size: usize,
    rng: &mut StdRng,
) -> Result<usize, String> {
    let mut filtered: Vec<(usize, f64)> = Vec::new();
    let mut sum = 0.0_f64;
    for (i, &w) in weights.iter().enumerate() {
        let size = i + 1;
        if size > max_size || w <= 0.0 {
            continue;
        }
        filtered.push((size, w));
        sum += w;
    }
    if filtered.is_empty() || !(sum.is_finite() && sum > 0.0) {
        return Err(format!(
            "logic_size_weights has no positive mass for term sizes <= {max_size}"
        ));
    }
    let mut draw = rng.random::<f64>() * sum;
    for (size, w) in filtered.into_iter() {
        if draw <= w {
            return Ok(size);
        }
        draw -= w;
    }
    Ok(max_size.min(weights.len()).max(1))
}

fn choose_unique_site_from_pool(
    pool: &[usize],
    used: &HashSet<usize>,
    ctx: &str,
    rng: &mut StdRng,
) -> Result<usize, String> {
    let avail: Vec<usize> = pool.iter().copied().filter(|idx| !used.contains(idx)).collect();
    if avail.is_empty() {
        return Err(format!("{ctx}: no unused candidate sites remain"));
    }
    Ok(avail[rng.random_range(0..avail.len())])
}

fn build_mixed_logic_term_plan(
    sites: &[SimSiteRecord],
    constraint_pools: &[Vec<usize>],
    causal_count: usize,
    logic_mode: &str,
    logic_size_weights: &[f64],
    causal_maf_min: f32,
    logic_window_bp: Option<i32>,
    rng: &mut StdRng,
) -> Result<Vec<MixedPlannedTerm>, String> {
    validate_logic_size_weights(logic_size_weights)?;
    if sites.is_empty() {
        return Err("no eligible sites remain after QC".to_string());
    }
    if causal_count < constraint_pools.len() {
        return Err(format!(
            "causal_count must be >= number of causal constraint groups: causal_count={}, groups={}",
            causal_count,
            constraint_pools.len()
        ));
    }

    let eligible_all: Vec<usize> = sites
        .iter()
        .enumerate()
        .filter_map(|(idx, site)| site_passes_causal_maf(site, causal_maf_min).then_some(idx))
        .collect();
    if eligible_all.is_empty() {
        return Err(format!(
            "no causal-site candidates remain after lmaf filtering: lmaf={:.4}",
            causal_maf_min
        ));
    }

    let mut range_pools: Vec<Vec<usize>> = Vec::with_capacity(constraint_pools.len());
    for (ri, pool) in constraint_pools.iter().enumerate() {
        let filtered: Vec<usize> = pool
            .iter()
            .copied()
            .into_iter()
            .filter(|idx| site_passes_causal_maf(&sites[*idx], causal_maf_min))
            .collect();
        if filtered.is_empty() {
            return Err(format!(
                "causal constraint group[{ri}] has no causal-site candidates after lmaf filtering: lmaf={:.4}",
                causal_maf_min
            ));
        }
        range_pools.push(filtered);
    }

    let total_terms = causal_count;
    if total_terms == 0 {
        return Ok(Vec::new());
    }

    let mut plan: Vec<MixedPlannedTerm> = Vec::with_capacity(total_terms);
    let mut used_additive: HashSet<usize> = HashSet::new();

    for (ri, pool) in range_pools.iter().enumerate() {
        let size = sample_term_size_with_limit(logic_size_weights, pool.len(), rng)?;
        if size == 1 {
            let pick = choose_unique_site_from_pool(
                pool.as_slice(),
                &used_additive,
                &format!("bimrange[{ri}]"),
                rng,
            )?;
            used_additive.insert(pick);
            plan.push(MixedPlannedTerm::Additive(pick));
        } else {
            plan.push(MixedPlannedTerm::Logic(LogicSampledSpec {
                pool_indices: reservoir_sample(pool.clone(), 1024, rng),
                mode: logic_mode_from_str(logic_mode, rng)?,
                size,
            }));
        }
    }

    for ti in range_pools.len()..total_terms {
        let size = sample_term_size_with_limit(logic_size_weights, eligible_all.len(), rng)?;
        if size == 1 {
            let pick = choose_unique_site_from_pool(
                eligible_all.as_slice(),
                &used_additive,
                &format!("causal_term[{ti}]"),
                rng,
            )?;
            used_additive.insert(pick);
            plan.push(MixedPlannedTerm::Additive(pick));
            continue;
        }

        let pool = if let Some(window_bp) = logic_window_bp {
            let mut tries = 0usize;
            let mut found: Option<Vec<usize>> = None;
            while tries < 128 {
                tries += 1;
                let anchor_idx = eligible_all[rng.random_range(0..eligible_all.len())];
                let anchor = &sites[anchor_idx];
                let lo = anchor.pos.saturating_sub(window_bp);
                let hi = anchor.pos.saturating_add(window_bp);
                let mut cand = Vec::new();
                for &idx in eligible_all.iter() {
                    let site = &sites[idx];
                    if site.chrom_norm == anchor.chrom_norm && site.pos >= lo && site.pos <= hi {
                        cand.push(idx);
                    }
                }
                if cand.len() >= size {
                    found = Some(reservoir_sample(cand, 1024, rng));
                    break;
                }
            }
            found.unwrap_or_else(|| reservoir_sample(eligible_all.clone(), 1024, rng))
        } else {
            reservoir_sample(eligible_all.clone(), 1024, rng)
        };
        if pool.len() < size {
            return Err(format!(
                "unable to build a logic-gate candidate pool for term {ti} with requested size {size}"
            ));
        }
        plan.push(MixedPlannedTerm::Logic(LogicSampledSpec {
            pool_indices: pool,
            mode: logic_mode_from_str(logic_mode, rng)?,
            size,
        }));
    }

    Ok(plan)
}

fn build_logic_pool_specs(
    sites: &[SimSiteRecord],
    constraint_pools: &[Vec<usize>],
    causal_count: usize,
    logic_gate_count: Option<usize>,
    logic_mode: &str,
    logic_k_min: usize,
    causal_maf_min: f32,
    logic_window_bp: Option<i32>,
    rng: &mut StdRng,
) -> Result<Vec<LogicPoolSpec>, String> {
    if sites.is_empty() {
        return Err("no eligible sites remain after QC".to_string());
    }
    if logic_k_min == 0 {
        return Err("logic_k_min must be > 0".to_string());
    }
    let mut out: Vec<LogicPoolSpec> = Vec::new();
    for (ri, pool) in constraint_pools.iter().enumerate() {
        let pool: Vec<usize> = pool
            .iter()
            .copied()
            .into_iter()
            .filter(|idx| site_passes_causal_maf(&sites[*idx], causal_maf_min))
            .collect();
        if pool.len() < logic_k_min {
            return Err(format!(
                "causal constraint group[{ri}] does not contain enough lmaf-filtered sites for logic gate size: required >= {logic_k_min}, got {}, lmaf={:.4}",
                pool.len(),
                causal_maf_min,
            ));
        }
        out.push(LogicPoolSpec {
            pool_indices: reservoir_sample(pool, 1024, rng),
            mode: logic_mode_from_str(logic_mode, rng)?,
        });
    }
    let eligible_all: Vec<usize> = sites
        .iter()
        .enumerate()
        .filter_map(|(idx, site)| site_passes_causal_maf(site, causal_maf_min).then_some(idx))
        .collect();
    if eligible_all.len() < logic_k_min {
        return Err(format!(
            "unable to build a logic-gate candidate pool with at least {logic_k_min} lmaf-filtered sites: eligible={}, lmaf={:.4}",
            eligible_all.len(),
            causal_maf_min,
        ));
    }
    let requested_target = logic_gate_count.unwrap_or(causal_count.max(1));
    if requested_target < out.len() {
        return Err(format!(
            "requested logic term count must be >= number of causal constraint groups: requested_terms={}, groups={}",
            requested_target,
            out.len()
        ));
    }
    let target = requested_target;
    while out.len() < target {
        let pool = if let Some(window_bp) = logic_window_bp {
            let mut tries = 0usize;
            let mut found: Option<Vec<usize>> = None;
            while tries < 128 {
                tries += 1;
                let anchor_idx = rng.random_range(0..sites.len());
                let anchor = &sites[anchor_idx];
                let lo = anchor.pos.saturating_sub(window_bp);
                let hi = anchor.pos.saturating_add(window_bp);
                let mut cand = Vec::new();
                for (i, site) in sites.iter().enumerate() {
                    if site.chrom_norm == anchor.chrom_norm
                        && site.pos >= lo
                        && site.pos <= hi
                        && site_passes_causal_maf(site, causal_maf_min)
                    {
                        cand.push(i);
                    }
                }
                if cand.len() >= logic_k_min {
                    found = Some(reservoir_sample(cand, 1024, rng));
                    break;
                }
            }
            found.unwrap_or_else(|| reservoir_sample(eligible_all.clone(), 1024, rng))
        } else {
            reservoir_sample(eligible_all.clone(), 1024, rng)
        };
        if pool.len() < logic_k_min {
            return Err(format!(
                "unable to build a logic-gate candidate pool with at least {logic_k_min} lmaf-filtered sites"
            ));
        }
        out.push(LogicPoolSpec {
            pool_indices: pool,
            mode: logic_mode_from_str(logic_mode, rng)?,
        });
    }
    Ok(out)
}

fn select_logic_terms(
    pool_specs: &[LogicPoolSpec],
    row_map: &HashMap<usize, Vec<f32>>,
    sites: &[SimSiteRecord],
    logic_k_min: usize,
    logic_k_max: usize,
    logic_ld_max: f64,
    logic_het_max: f64,
    causal_maf_min: f32,
    logic_af_min: f64,
    logic_af_max: f64,
    logic_max_iter: usize,
    logic_effect_model: LogicEffectModel,
    rng: &mut StdRng,
) -> Result<Vec<CausalTerm>, String> {
    if logic_k_min == 0 {
        return Err("logic_k_min must be > 0".to_string());
    }
    if logic_k_max < logic_k_min {
        return Err("logic_k_max must be >= logic_k_min".to_string());
    }
    let af_center = 0.5 * (logic_af_min + logic_af_max);
    let mut used_global: HashSet<usize> = HashSet::new();
    let mut out = Vec::with_capacity(pool_specs.len());

    for (ti, spec) in pool_specs.iter().enumerate() {
        let mut bin_map: HashMap<usize, Vec<u8>> = HashMap::new();
        for &idx in spec.pool_indices.iter() {
            if let Some(row) = row_map.get(&idx) {
                if let Some(bin) = collapse_to_logic_bin01(row, logic_het_max) {
                    bin_map.insert(idx, bin);
                }
            }
        }
        if bin_map.len() < logic_k_min {
            return Err(format!(
                "logic-gate candidate pool {ti} has too few usable sites after heterozygosity filtering: need >= {logic_k_min}, got {}",
                bin_map.len()
            ));
        }

        let mut best: Option<(Vec<usize>, Vec<f64>, f64)> = None;
        for _ in 0..logic_max_iter.max(1) {
            let prefer_unused: Vec<usize> = spec
                .pool_indices
                .iter()
                .copied()
                .filter(|idx| bin_map.contains_key(idx) && !used_global.contains(idx))
                .collect();
            let all_avail: Vec<usize> = spec
                .pool_indices
                .iter()
                .copied()
                .filter(|idx| bin_map.contains_key(idx))
                .collect();
            let pool = if prefer_unused.len() >= logic_k_min {
                prefer_unused
            } else {
                all_avail
            };
            if pool.len() < logic_k_min {
                break;
            }
            let k_hi = logic_k_max.min(pool.len());
            let k = if k_hi == logic_k_min {
                logic_k_min
            } else {
                rng.random_range(logic_k_min..=k_hi)
            };
            let members = sample_without_replacement(&pool, k, rng)?;
            let mut ld_ok = true;
            if logic_ld_max < 0.999_999 {
                for a in 0..members.len() {
                    for b in (a + 1)..members.len() {
                        let r2 = binary_r2(
                            bin_map
                                .get(&members[a])
                                .ok_or_else(|| "missing logic row".to_string())?,
                            bin_map
                                .get(&members[b])
                                .ok_or_else(|| "missing logic row".to_string())?,
                        );
                        if r2 > logic_ld_max + 1e-12 {
                            ld_ok = false;
                            break;
                        }
                    }
                    if !ld_ok {
                        break;
                    }
                }
            }
            if !ld_ok {
                continue;
            }
            let gate_rows: Vec<Vec<u8>> = members
                .iter()
                .map(|idx| {
                    bin_map
                        .get(idx)
                        .cloned()
                        .ok_or_else(|| "missing logic row".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let literal_rows = logic_gate_literal_rows(&gate_rows, spec.mode)?;
            let gate_indicator = logic_gate_indicator_from_literals(
                literal_rows.as_slice(),
                logic_output_negated(spec.mode),
            );
            let raw_af = gate_indicator.iter().filter(|&&v| v != 0).count() as f64
                / gate_indicator.len() as f64;
            let gate_maf = raw_af.min(1.0_f64 - raw_af);
            let gate_values = match logic_effect_model {
                LogicEffectModel::Gate => logic_gate_centered_values(&gate_indicator),
                LogicEffectModel::CenteredInteraction => {
                    residualize_logic_indicator_against_main_effects(
                        &gate_indicator,
                        literal_rows.as_slice(),
                    )?
                }
            };
            let var = variance_f64(&gate_values);
            if var <= 1e-12 {
                continue;
            }
            if gate_maf + 1e-12_f64 >= causal_maf_min as f64
                && best
                    .as_ref()
                    .map(|(_, _, af)| (raw_af - af_center).abs() < (*af - af_center).abs())
                    .unwrap_or(true)
            {
                best = Some((members.clone(), gate_values.clone(), raw_af));
            }
            if raw_af >= logic_af_min
                && raw_af <= logic_af_max
                && gate_maf + 1e-12_f64 >= causal_maf_min as f64
            {
                let label = term_label(sites, &members, Some(spec.mode));
                for idx in members.iter() {
                    used_global.insert(*idx);
                }
                out.push(CausalTerm {
                    members,
                    mode: Some(spec.mode),
                    values: gate_values,
                    effect: 0.0,
                    label,
                });
                break;
            }
        }

        if out.len() != ti + 1 {
            if let Some((members, gate, _af)) = best {
                let label = term_label(sites, &members, Some(spec.mode));
                for idx in members.iter() {
                    used_global.insert(*idx);
                }
                out.push(CausalTerm {
                    members,
                    mode: Some(spec.mode),
                    values: gate,
                    effect: 0.0,
                    label,
                });
            } else {
                return Err(format!(
                    "unable to build a valid logic gate for pool {ti}; try relaxing gate size / LD / AF / heterozygosity constraints (gate_size={}..{}, lmaf={:.4})",
                    logic_k_min,
                    logic_k_max,
                    causal_maf_min,
                ));
            }
        }
    }
    Ok(out)
}

fn select_logic_terms_sampled_specs(
    specs: &[LogicSampledSpec],
    row_map: &HashMap<usize, Vec<f32>>,
    sites: &[SimSiteRecord],
    logic_ld_max: f64,
    logic_het_max: f64,
    causal_maf_min: f32,
    logic_af_min: f64,
    logic_af_max: f64,
    logic_max_iter: usize,
    logic_effect_model: LogicEffectModel,
    initial_used: &HashSet<usize>,
    rng: &mut StdRng,
) -> Result<Vec<CausalTerm>, String> {
    let af_center = 0.5 * (logic_af_min + logic_af_max);
    let mut used_global = initial_used.clone();
    let mut out = Vec::with_capacity(specs.len());

    for (ti, spec) in specs.iter().enumerate() {
        let mut bin_map: HashMap<usize, Vec<u8>> = HashMap::new();
        for &idx in spec.pool_indices.iter() {
            if let Some(row) = row_map.get(&idx) {
                if let Some(bin) = collapse_to_logic_bin01(row, logic_het_max) {
                    bin_map.insert(idx, bin);
                }
            }
        }
        if bin_map.len() < spec.size {
            return Err(format!(
                "logic-gate candidate pool {ti} has too few usable sites after heterozygosity filtering: need >= {}, got {}",
                spec.size,
                bin_map.len()
            ));
        }

        let mut best: Option<(Vec<usize>, Vec<f64>, f64)> = None;
        for _ in 0..logic_max_iter.max(1) {
            let pool: Vec<usize> = spec
                .pool_indices
                .iter()
                .copied()
                .filter(|idx| bin_map.contains_key(idx) && !used_global.contains(idx))
                .collect();
            if pool.len() < spec.size {
                break;
            }
            let members = sample_without_replacement(&pool, spec.size, rng)?;
            let mut ld_ok = true;
            if logic_ld_max < 0.999_999 {
                for a in 0..members.len() {
                    for b in (a + 1)..members.len() {
                        let r2 = binary_r2(
                            bin_map
                                .get(&members[a])
                                .ok_or_else(|| "missing logic row".to_string())?,
                            bin_map
                                .get(&members[b])
                                .ok_or_else(|| "missing logic row".to_string())?,
                        );
                        if r2 > logic_ld_max + 1e-12 {
                            ld_ok = false;
                            break;
                        }
                    }
                    if !ld_ok {
                        break;
                    }
                }
            }
            if !ld_ok {
                continue;
            }

            let gate_rows: Vec<Vec<u8>> = members
                .iter()
                .map(|idx| {
                    bin_map
                        .get(idx)
                        .cloned()
                        .ok_or_else(|| "missing logic row".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let literal_rows = logic_gate_literal_rows(&gate_rows, spec.mode)?;
            let gate_indicator = logic_gate_indicator_from_literals(
                literal_rows.as_slice(),
                logic_output_negated(spec.mode),
            );
            let raw_af = gate_indicator.iter().filter(|&&v| v != 0).count() as f64
                / gate_indicator.len() as f64;
            let gate_maf = raw_af.min(1.0_f64 - raw_af);
            let gate_values = match logic_effect_model {
                LogicEffectModel::Gate => logic_gate_centered_values(&gate_indicator),
                LogicEffectModel::CenteredInteraction => {
                    residualize_logic_indicator_against_main_effects(
                        &gate_indicator,
                        literal_rows.as_slice(),
                    )?
                }
            };
            let var = variance_f64(&gate_values);
            if var <= 1e-12 {
                continue;
            }
            if gate_maf + 1e-12_f64 >= causal_maf_min as f64
                && best
                    .as_ref()
                    .map(|(_, _, af)| (raw_af - af_center).abs() < (*af - af_center).abs())
                    .unwrap_or(true)
            {
                best = Some((members.clone(), gate_values.clone(), raw_af));
            }
            if raw_af >= logic_af_min
                && raw_af <= logic_af_max
                && gate_maf + 1e-12_f64 >= causal_maf_min as f64
            {
                let label = term_label(sites, &members, Some(spec.mode));
                for idx in members.iter() {
                    used_global.insert(*idx);
                }
                out.push(CausalTerm {
                    members,
                    mode: Some(spec.mode),
                    values: gate_values,
                    effect: 0.0,
                    label,
                });
                break;
            }
        }

        if out.len() != ti + 1 {
            if let Some((members, gate, _af)) = best {
                let label = term_label(sites, &members, Some(spec.mode));
                for idx in members.iter() {
                    used_global.insert(*idx);
                }
                out.push(CausalTerm {
                    members,
                    mode: Some(spec.mode),
                    values: gate,
                    effect: 0.0,
                    label,
                });
            } else {
                return Err(format!(
                    "unable to build a valid logic gate for term {ti}; try relaxing LD / AF / heterozygosity constraints (gate_size={}, lmaf={:.4})",
                    spec.size,
                    causal_maf_min,
                ));
            }
        }
    }
    Ok(out)
}

fn build_additive_term(
    idx: usize,
    row_map: &HashMap<usize, Vec<f32>>,
    sites: &[SimSiteRecord],
) -> Result<CausalTerm, String> {
    let row = row_map
        .get(&idx)
        .ok_or_else(|| format!("selected causal site row missing for index {idx}"))?;
    let values = centered_row_to_owned_f64(row);
    if variance_f64(&values) <= 1e-12 {
        return Err(format!(
            "selected causal site has zero variance after centering: {}:{}",
            sites[idx].chrom, sites[idx].pos
        ));
    }
    Ok(CausalTerm {
        members: vec![idx],
        mode: None,
        values,
        effect: 0.0,
        label: term_label(sites, &[idx], None),
    })
}

fn build_additive_terms(
    selected_indices: &[usize],
    row_map: &HashMap<usize, Vec<f32>>,
    sites: &[SimSiteRecord],
) -> Result<Vec<CausalTerm>, String> {
    let mut out = Vec::with_capacity(selected_indices.len());
    for &idx in selected_indices.iter() {
        out.push(build_additive_term(idx, row_map, sites)?);
    }
    Ok(out)
}

fn materialize_mixed_terms_from_plan(
    plan: &[MixedPlannedTerm],
    row_map: &HashMap<usize, Vec<f32>>,
    sites: &[SimSiteRecord],
    logic_ld_max: f64,
    logic_het_max: f64,
    causal_maf_min: f32,
    logic_af_min: f64,
    logic_af_max: f64,
    logic_max_iter: usize,
    logic_effect_model: LogicEffectModel,
    rng: &mut StdRng,
) -> Result<Vec<CausalTerm>, String> {
    let mut additive_used: HashSet<usize> = HashSet::new();
    let mut logic_specs: Vec<LogicSampledSpec> = Vec::new();
    for item in plan.iter() {
        match item {
            MixedPlannedTerm::Additive(idx) => {
                additive_used.insert(*idx);
            }
            MixedPlannedTerm::Logic(spec) => logic_specs.push(spec.clone()),
        }
    }
    let mut logic_terms_iter = select_logic_terms_sampled_specs(
        logic_specs.as_slice(),
        row_map,
        sites,
        logic_ld_max,
        logic_het_max,
        causal_maf_min,
        logic_af_min,
        logic_af_max,
        logic_max_iter,
        logic_effect_model,
        &additive_used,
        rng,
    )?
    .into_iter();
    let mut out: Vec<CausalTerm> = Vec::with_capacity(plan.len());
    for item in plan.iter() {
        match item {
            MixedPlannedTerm::Additive(idx) => out.push(build_additive_term(*idx, row_map, sites)?),
            MixedPlannedTerm::Logic(_) => out.push(
                logic_terms_iter
                    .next()
                    .ok_or_else(|| "internal error: mixed logic term materialization underflow".to_string())?,
            ),
        }
    }
    if logic_terms_iter.next().is_some() {
        return Err("internal error: mixed logic term materialization overflow".to_string());
    }
    Ok(out)
}

fn write_pheno_files(
    prefix: &str,
    sample_ids: &[String],
    y: &[f64],
    trait_name: &str,
    na_rate: f64,
    seed: u64,
) -> Result<(), String> {
    if sample_ids.len() != y.len() {
        return Err(format!(
            "sample/phenotype length mismatch: ids={}, y={}",
            sample_ids.len(),
            y.len()
        ));
    }
    let pheno_path = format!("{prefix}.pheno");
    let pheno_txt_path = format!("{prefix}.pheno.txt");
    let pheno_na_path = format!("{prefix}.pheno.NA.txt");

    let mut w3 = BufWriter::new(File::create(&pheno_path).map_err(|e| e.to_string())?);
    for (sid, &yv) in sample_ids.iter().zip(y.iter()) {
        writeln!(w3, "{sid}\t{sid}\t{yv:.6}").map_err(|e| e.to_string())?;
    }
    w3.flush().map_err(|e| e.to_string())?;

    let mut w2 = BufWriter::new(File::create(&pheno_txt_path).map_err(|e| e.to_string())?);
    writeln!(w2, "IID\t{trait_name}").map_err(|e| e.to_string())?;
    for (sid, &yv) in sample_ids.iter().zip(y.iter()) {
        writeln!(w2, "{sid}\t{yv:.6}").map_err(|e| e.to_string())?;
    }
    w2.flush().map_err(|e| e.to_string())?;

    let mut na_idx: Vec<usize> = (0..sample_ids.len()).collect();
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD9A3_5C71_4B1E_208Du64);
    na_idx.shuffle(&mut rng);
    let k = ((sample_ids.len() as f64) * na_rate.clamp(0.0, 1.0)).round() as usize;
    let na_set: HashSet<usize> = na_idx.into_iter().take(k).collect();
    let mut wna = BufWriter::new(File::create(&pheno_na_path).map_err(|e| e.to_string())?);
    writeln!(wna, "IID\t{trait_name}").map_err(|e| e.to_string())?;
    for (i, sid) in sample_ids.iter().enumerate() {
        if na_set.contains(&i) {
            writeln!(wna, "{sid}\tNA").map_err(|e| e.to_string())?;
        } else {
            writeln!(wna, "{sid}\t{:.6}", y[i]).map_err(|e| e.to_string())?;
        }
    }
    wna.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_sample_background_effects(
    path: &str,
    sample_ids: &[String],
    effects: &[f64],
    source: &str,
) -> Result<(), String> {
    if sample_ids.len() != effects.len() {
        return Err(format!(
            "sample background effect length mismatch: sample_ids={}, effects={}",
            sample_ids.len(),
            effects.len()
        ));
    }
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(w, "sample_index\tsample_id\tsource\trole\teffect").map_err(|e| e.to_string())?;
    for (i, (sid, &eff)) in sample_ids.iter().zip(effects.iter()).enumerate() {
        writeln!(w, "{}\t{}\t{}\tbackground\t{eff:.10}", i + 1, sid, source)
            .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_fixed_effects(
    path: &str,
    terms: &[CausalTerm],
    sites: &[SimSiteRecord],
) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(w, "term_id\tkind\tlogic\tsites\tlabel\teffect").map_err(|e| e.to_string())?;
    for (i, term) in terms.iter().enumerate() {
        let kind = if term.mode.is_some() {
            "logic_gate"
        } else {
            "additive"
        };
        let logic = term.mode.map(logic_mode_code).unwrap_or("single");
        let site_text = term
            .members
            .iter()
            .map(|&idx| format!("{}:{}", sites[idx].chrom, sites[idx].pos))
            .collect::<Vec<String>>()
            .join(";");
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{:.10}",
            i + 1,
            kind,
            logic,
            site_text,
            term.label,
            term.effect,
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_causal_sites(
    path: &str,
    terms: &[CausalTerm],
    sites: &[SimSiteRecord],
) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    let mut seen: HashSet<(String, i32)> = HashSet::new();
    for term in terms.iter() {
        for &idx in term.members.iter() {
            let site = &sites[idx];
            let key = (site.chrom.clone(), site.pos);
            if !seen.insert(key) {
                continue;
            }
            writeln!(w, "{}\t{}\t{}", site.chrom, site.pos, site.pos).map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn parse_background_dist(name: &str) -> Result<BackgroundDist, String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "normal" | "gaussian" => Ok(BackgroundDist::Normal),
        "gamma" => Ok(BackgroundDist::Gamma),
        "laplace" => Ok(BackgroundDist::Laplace),
        other => Err(format!("unsupported background distribution: {other}")),
    }
}

fn parse_logic_effect_model(name: &str) -> Result<LogicEffectModel, String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "" | "gate" | "raw" | "mean_centered_gate" => Ok(LogicEffectModel::Gate),
        "centered" | "centered_interaction" | "interaction" | "orthogonal" => {
            Ok(LogicEffectModel::CenteredInteraction)
        }
        other => Err(format!("unsupported logic effect model: {other}")),
    }
}

fn g2p_progress_notify(
    progress_callback: Option<&Py<PyAny>>,
    stage: &str,
    done: usize,
    total: usize,
    notify_step: usize,
    last_notified: &mut usize,
    force: bool,
) -> Result<(), String> {
    let done_clamped = if total > 0 { done.min(total) } else { done };
    if !force && done_clamped < last_notified.saturating_add(notify_step.max(1)) {
        return Ok(());
    }
    *last_notified = done_clamped;
    Python::attach(|py2| -> PyResult<()> {
        py2.check_signals()?;
        if let Some(cb) = progress_callback {
            cb.call1(py2, (stage, done_clamped, total))?;
        }
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn g2p_simulate_core(config: G2pSimConfig) -> Result<G2pSimResult, String> {
    if !(0.0..=1.0).contains(&config.bg_pve) {
        return Err("bg_pve must be within [0, 1].".to_string());
    }
    if !(0.0..=1.0).contains(&config.na_rate) {
        return Err("na_rate must be within [0, 1].".to_string());
    }
    if config.residual_var < 0.0 || !config.residual_var.is_finite() {
        return Err("residual_var must be finite and >= 0.".to_string());
    }
    if config.logic_k_min == 0 {
        return Err("logic_k_min must be > 0.".to_string());
    }
    if config.logic_k_max < config.logic_k_min {
        return Err("logic_k_max must be >= logic_k_min.".to_string());
    }
    if !(0.0..=0.5).contains(&config.causal_maf_min) {
        return Err("causal_maf_min must be within [0, 0.5].".to_string());
    }
    if !(0.0..=1.0).contains(&config.logic_ld_max) {
        return Err("logic_ld_max must be within [0, 1].".to_string());
    }
    if !(0.0..=1.0).contains(&config.logic_het_max) {
        return Err("logic_het_max must be within [0, 1].".to_string());
    }
    if let Some(het_threshold) = config.het_threshold {
        if !(0.0..=1.0).contains(&het_threshold) {
            return Err("het_threshold must be within [0, 1].".to_string());
        }
    }
    if !(0.0..=1.0).contains(&config.logic_af_min) || !(0.0..=1.0).contains(&config.logic_af_max) {
        return Err("logic_af_min/logic_af_max must be within [0, 1].".to_string());
    }
    if config.logic_af_min > config.logic_af_max {
        return Err("logic_af_min must be <= logic_af_max.".to_string());
    }

    let logic_requested = config
        .logic_mode
        .as_ref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if config.logic_size_weights.is_some() && !logic_requested {
        return Err("logic_size_weights requires logic_mode to be set.".to_string());
    }
    let mixed_logic_requested = logic_requested
        && config
            .logic_size_weights
            .as_ref()
            .map(|w| !w.is_empty())
            .unwrap_or(false);
    if let Some(weights) = config.logic_size_weights.as_ref() {
        validate_logic_size_weights(weights)?;
    }
    if !config.bim_ranges.is_empty() && !config.bim_range_groups.is_empty() {
        return Err("bim_ranges and bim_range_groups cannot both be set.".to_string());
    }
    let constraint_group_count = if !config.bim_range_groups.is_empty() {
        config.bim_range_groups.len()
    } else {
        config.bim_ranges.len()
    };
    let base_term_count = if mixed_logic_requested {
        config.causal_count
    } else if logic_requested {
        config
            .logic_gate_count
            .unwrap_or(config.causal_count.max(1))
    } else {
        config.causal_count
    };
    if base_term_count < constraint_group_count {
        return Err(format!(
            "requested causal term count must be >= number of causal constraint groups: requested_terms={}, groups={}",
            base_term_count,
            constraint_group_count
        ));
    }
    let effective_term_count = base_term_count;
    let causal_pve_target = config.causal_pve.unwrap_or_else(|| {
        if effective_term_count == 0 {
            0.0
        } else {
            (0.05_f64 * effective_term_count as f64).min(0.95)
        }
    });
    if !(0.0..=1.0).contains(&causal_pve_target) {
        return Err("causal_pve must be within [0, 1].".to_string());
    }

    let residual_var_eff = 1.0 - config.bg_pve - causal_pve_target;
    if residual_var_eff < -1e-12 {
        return Err(
            "bg_pve + causal_pve must be <= 1.0 under the final-variance PVE definition."
                .to_string(),
        );
    }
    let residual_var_eff = residual_var_eff.max(0.0);
    if config.bg_pve > 0.0 && config.grm.is_none() {
        return Err(
            "background sample-space simulation requires a GRM; provide --grm or use a caller that auto-builds cached cGRM."
                .to_string(),
        );
    }
    let needs_causal_scan = effective_term_count > 0 && causal_pve_target > 0.0;
    let scan_passes = 1usize + if needs_causal_scan { 1 } else { 0 };
    let progress_site_total = config.progress_total_hint.unwrap_or(0);
    let progress_overall_total = progress_site_total.saturating_mul(scan_passes);
    let progress_every = config.progress_every.max(1);

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut sites: Vec<SimSiteRecord> = Vec::new();
    let mut bg_progress_seen = 0usize;
    let mut bg_last_notified = 0usize;
    let mut fast_bed_path: Option<PreparedBedFastPath> = None;
    let mut runtime_progress_total = progress_overall_total;
    let sample_ids = if let Some(prepared) = try_prepare_bed_fast_path(
        &config.path_or_prefix,
        config.maf_threshold,
        config.max_missing_rate,
        config.het_threshold,
        config.snps_only,
        true,
    )? {
        sites = prepared.sites;
        let sample_ids = prepared.sample_ids;
        fast_bed_path = Some(PreparedBedFastPath {
            prefix: prepared.prefix,
            sample_ids: Vec::new(),
            sites: Vec::new(),
            row_source_indices: prepared.row_source_indices,
        });
        runtime_progress_total = progress_site_total;
        if progress_site_total > 0 {
            g2p_progress_notify(
                config.progress_callback.as_ref(),
                "background",
                progress_site_total,
                runtime_progress_total,
                progress_every,
                &mut bg_last_notified,
                true,
            )?;
        }
        sample_ids
    } else {
        let sample_ids = iterate_filtered_rows(
            &config.path_or_prefix,
            config.delimiter.as_deref(),
            config.maf_threshold,
            config.max_missing_rate,
            config.het_threshold,
            config.snps_only,
            |_, row, site| {
                bg_progress_seen = bg_progress_seen.saturating_add(1);
                if progress_overall_total > 0 {
                    g2p_progress_notify(
                        config.progress_callback.as_ref(),
                        "background",
                        bg_progress_seen,
                        progress_overall_total,
                        progress_every,
                        &mut bg_last_notified,
                        false,
                    )?;
                }
                sites.push(SimSiteRecord {
                    chrom: site.chrom.clone(),
                    chrom_norm: normalize_chrom(&site.chrom),
                    pos: site.pos,
                    ref_allele: site.ref_allele.clone(),
                    alt_allele: site.alt_allele.clone(),
                    maf: processed_row_minor_allele_frequency(row),
                });
                Ok(())
            },
        )?;
        if progress_overall_total > 0 {
            g2p_progress_notify(
                config.progress_callback.as_ref(),
                "background",
                progress_site_total,
                progress_overall_total,
                progress_every,
                &mut bg_last_notified,
                true,
            )?;
        }
        sample_ids
    };

    if sample_ids.is_empty() {
        return Err("no samples found in genotype input after inspection".to_string());
    }
    if sites.is_empty() {
        return Err("no eligible variants remain after QC filtering".to_string());
    }

    let n = sample_ids.len();
    let residual_effects = sample_gaussian_noise_with_variance(n, residual_var_eff, &mut rng);
    let mut y = residual_effects.clone();
    let bg_var_target = config.bg_pve;
    let (bg_effects, background_source, background_factorization) = if let Some(grm_vec) = config.grm.as_ref() {
        let grm_n = config
            .grm_n
            .ok_or_else(|| "internal error: grm_n missing for provided GRM".to_string())?;
        if grm_n != n {
            return Err(format!(
                "GRM size mismatch: got n={}, expected {} genotype samples",
                grm_n, n
            ));
        }
        let (raw, factorization) = if bg_var_target > 0.0 {
            let mut grm_factor_last_notified = 0usize;
            g2p_progress_notify(
                config.progress_callback.as_ref(),
                "grm_factor",
                0,
                1,
                1,
                &mut grm_factor_last_notified,
                true,
            )?;
            sample_background_effects_from_grm_trace_scaled(
                grm_vec.as_slice(),
                n,
                bg_var_target,
                &mut rng,
            )?
        } else {
            (vec![0.0_f64; n], "none".to_string())
        };
        if bg_var_target > 0.0 {
            let mut grm_factor_last_notified = 0usize;
            g2p_progress_notify(
                config.progress_callback.as_ref(),
                "grm_factor",
                1,
                1,
                1,
                &mut grm_factor_last_notified,
                true,
            )?;
        }
        axpy_inplace(&mut y, raw.as_slice(), 1.0);
        (raw, "grm".to_string(), factorization)
    } else {
        let raw = vec![0.0_f64; n];
        axpy_inplace(&mut y, raw.as_slice(), 1.0);
        (raw, "none".to_string(), "none".to_string())
    };
    let mut causal_component = vec![0.0_f64; n];

    let mut causal_terms: Vec<CausalTerm> = Vec::new();
    if needs_causal_scan {
        let constraint_pools = build_constraint_pools(
            &sites,
            &config.bim_ranges,
            &config.bim_range_groups,
        )?;
        let causal_offset = progress_site_total;
        if let Some(bed_fast) = fast_bed_path.as_ref() {
            if mixed_logic_requested {
                let logic_mode_str = config.logic_mode.as_deref().unwrap_or("a");
                let logic_size_weights = config
                    .logic_size_weights
                    .as_ref()
                    .ok_or_else(|| "internal error: mixed logic weights missing".to_string())?;
                let mut plan_rng = StdRng::seed_from_u64(config.seed ^ 0xB28F_6A91_C547_31D1u64);
                let mixed_plan = build_mixed_logic_term_plan(
                    &sites,
                    constraint_pools.as_slice(),
                    config.causal_count,
                    logic_mode_str,
                    logic_size_weights.as_slice(),
                    config.causal_maf_min,
                    config.logic_window_bp,
                    &mut plan_rng,
                )?;
                let mut need_rows: HashSet<usize> = HashSet::with_capacity(1024);
                for item in mixed_plan.iter() {
                    match item {
                        MixedPlannedTerm::Additive(idx) => {
                            need_rows.insert(*idx);
                        }
                        MixedPlannedTerm::Logic(spec) => {
                            for &idx in spec.pool_indices.iter() {
                                need_rows.insert(idx);
                            }
                        }
                    }
                }
                let need_kept_indices = need_rows.into_iter().collect::<Vec<_>>();
                runtime_progress_total = causal_offset.saturating_add(need_kept_indices.len());
                let progress_stage = if mixed_plan
                    .iter()
                    .all(|item| matches!(item, MixedPlannedTerm::Additive(_)))
                {
                    "causal_additive"
                } else {
                    "causal_logic"
                };
                let row_map = decode_bed_rows_by_kept_index(
                    &bed_fast.prefix,
                    bed_fast.row_source_indices.as_slice(),
                    need_kept_indices.as_slice(),
                    config.progress_callback.as_ref(),
                    progress_stage,
                    causal_offset,
                    runtime_progress_total,
                    progress_every,
                )?;
                let mut logic_rng = StdRng::seed_from_u64(config.seed ^ 0x9F4B_72E0_1A33_4F0Cu64);
                causal_terms = materialize_mixed_terms_from_plan(
                    mixed_plan.as_slice(),
                    &row_map,
                    &sites,
                    config.logic_ld_max,
                    config.logic_het_max,
                    config.causal_maf_min,
                    config.logic_af_min,
                    config.logic_af_max,
                    config.logic_max_iter,
                    config.logic_effect_model,
                    &mut logic_rng,
                )?;
            } else if logic_requested {
                let logic_mode_str = config.logic_mode.as_deref().unwrap_or("a");
                let mut pool_rng = StdRng::seed_from_u64(config.seed ^ 0xB28F_6A91_C547_31D1u64);
                let pool_specs = build_logic_pool_specs(
                    &sites,
                    constraint_pools.as_slice(),
                    config.causal_count,
                    config.logic_gate_count,
                    logic_mode_str,
                    config.logic_k_min,
                    config.causal_maf_min,
                    config.logic_window_bp,
                    &mut pool_rng,
                )?;
                let mut need_rows: HashSet<usize> = HashSet::with_capacity(1024);
                for spec in pool_specs.iter() {
                    for &idx in spec.pool_indices.iter() {
                        need_rows.insert(idx);
                    }
                }
                let need_kept_indices = need_rows.into_iter().collect::<Vec<_>>();
                runtime_progress_total = causal_offset.saturating_add(need_kept_indices.len());
                let row_map = decode_bed_rows_by_kept_index(
                    &bed_fast.prefix,
                    bed_fast.row_source_indices.as_slice(),
                    need_kept_indices.as_slice(),
                    config.progress_callback.as_ref(),
                    "causal_logic",
                    causal_offset,
                    runtime_progress_total,
                    progress_every,
                )?;
                let mut logic_rng = StdRng::seed_from_u64(config.seed ^ 0x9F4B_72E0_1A33_4F0Cu64);
                causal_terms = select_logic_terms(
                    &pool_specs,
                    &row_map,
                    &sites,
                    config.logic_k_min,
                    config.logic_k_max,
                    config.logic_ld_max,
                    config.logic_het_max,
                    config.causal_maf_min,
                    config.logic_af_min,
                    config.logic_af_max,
                    config.logic_max_iter,
                    config.logic_effect_model,
                    &mut logic_rng,
                )?;
            } else {
                let mut sel_rng = StdRng::seed_from_u64(config.seed ^ 0xA54D_3F9E_6721_8CB7u64);
                let selected = select_additive_indices(
                    &sites,
                    config.causal_count,
                    constraint_pools.as_slice(),
                    config.causal_maf_min,
                    &mut sel_rng,
                )?;
                runtime_progress_total = causal_offset.saturating_add(selected.len());
                let row_map = decode_bed_rows_by_kept_index(
                    &bed_fast.prefix,
                    bed_fast.row_source_indices.as_slice(),
                    selected.as_slice(),
                    config.progress_callback.as_ref(),
                    "causal_additive",
                    causal_offset,
                    runtime_progress_total,
                    progress_every,
                )?;
                causal_terms = build_additive_terms(&selected, &row_map, &sites)?;
            }
        } else if mixed_logic_requested {
            let mut causal_progress_seen = 0usize;
            let mut causal_last_notified = 0usize;
            let logic_mode_str = config.logic_mode.as_deref().unwrap_or("a");
            let logic_size_weights = config
                .logic_size_weights
                .as_ref()
                .ok_or_else(|| "internal error: mixed logic weights missing".to_string())?;
            let mut plan_rng = StdRng::seed_from_u64(config.seed ^ 0xB28F_6A91_C547_31D1u64);
            let mixed_plan = build_mixed_logic_term_plan(
                &sites,
                constraint_pools.as_slice(),
                config.causal_count,
                logic_mode_str,
                logic_size_weights.as_slice(),
                config.causal_maf_min,
                config.logic_window_bp,
                &mut plan_rng,
            )?;
            let mut need_rows: HashSet<usize> = HashSet::with_capacity(1024);
            for item in mixed_plan.iter() {
                match item {
                    MixedPlannedTerm::Additive(idx) => {
                        need_rows.insert(*idx);
                    }
                    MixedPlannedTerm::Logic(spec) => {
                        for &idx in spec.pool_indices.iter() {
                            need_rows.insert(idx);
                        }
                    }
                }
            }
            let progress_stage = if mixed_plan
                .iter()
                .all(|item| matches!(item, MixedPlannedTerm::Additive(_)))
            {
                "causal_additive"
            } else {
                "causal_logic"
            };
            let mut row_map: HashMap<usize, Vec<f32>> = HashMap::with_capacity(need_rows.len());
            iterate_filtered_rows(
                &config.path_or_prefix,
                config.delimiter.as_deref(),
                config.maf_threshold,
                config.max_missing_rate,
                config.het_threshold,
                config.snps_only,
                |kept_idx, row, _site| {
                    causal_progress_seen = causal_progress_seen.saturating_add(1);
                    if progress_overall_total > 0 {
                        g2p_progress_notify(
                            config.progress_callback.as_ref(),
                            progress_stage,
                            causal_offset.saturating_add(causal_progress_seen),
                            progress_overall_total,
                            progress_every,
                            &mut causal_last_notified,
                            false,
                        )?;
                    }
                    if need_rows.contains(&kept_idx) {
                        row_map.insert(kept_idx, row.to_vec());
                    }
                    Ok(())
                },
            )?;
            if progress_overall_total > 0 {
                g2p_progress_notify(
                    config.progress_callback.as_ref(),
                    progress_stage,
                    causal_offset.saturating_add(progress_site_total),
                    progress_overall_total,
                    progress_every,
                    &mut causal_last_notified,
                    true,
                )?;
            }
            let mut logic_rng = StdRng::seed_from_u64(config.seed ^ 0x9F4B_72E0_1A33_4F0Cu64);
            causal_terms = materialize_mixed_terms_from_plan(
                mixed_plan.as_slice(),
                &row_map,
                &sites,
                config.logic_ld_max,
                config.logic_het_max,
                config.causal_maf_min,
                config.logic_af_min,
                config.logic_af_max,
                config.logic_max_iter,
                config.logic_effect_model,
                &mut logic_rng,
            )?;
        } else if logic_requested {
            let mut causal_progress_seen = 0usize;
            let mut causal_last_notified = 0usize;
            let logic_mode_str = config.logic_mode.as_deref().unwrap_or("a");
            let mut pool_rng = StdRng::seed_from_u64(config.seed ^ 0xB28F_6A91_C547_31D1u64);
            let pool_specs = build_logic_pool_specs(
                &sites,
                constraint_pools.as_slice(),
                config.causal_count,
                config.logic_gate_count,
                logic_mode_str,
                config.logic_k_min,
                config.causal_maf_min,
                config.logic_window_bp,
                &mut pool_rng,
            )?;
            let mut need_rows: HashSet<usize> = HashSet::with_capacity(1024);
            for spec in pool_specs.iter() {
                for &idx in spec.pool_indices.iter() {
                    need_rows.insert(idx);
                }
            }
            let mut row_map: HashMap<usize, Vec<f32>> = HashMap::with_capacity(need_rows.len());
            iterate_filtered_rows(
                &config.path_or_prefix,
                config.delimiter.as_deref(),
                config.maf_threshold,
                config.max_missing_rate,
                config.het_threshold,
                config.snps_only,
                |kept_idx, row, _site| {
                    causal_progress_seen = causal_progress_seen.saturating_add(1);
                    if progress_overall_total > 0 {
                        g2p_progress_notify(
                            config.progress_callback.as_ref(),
                            "causal_logic",
                            causal_offset.saturating_add(causal_progress_seen),
                            progress_overall_total,
                            progress_every,
                            &mut causal_last_notified,
                            false,
                        )?;
                    }
                    if need_rows.contains(&kept_idx) {
                        row_map.insert(kept_idx, row.to_vec());
                    }
                    Ok(())
                },
            )?;
            if progress_overall_total > 0 {
                g2p_progress_notify(
                    config.progress_callback.as_ref(),
                    "causal_logic",
                    causal_offset.saturating_add(progress_site_total),
                    progress_overall_total,
                    progress_every,
                    &mut causal_last_notified,
                    true,
                )?;
            }
            let mut logic_rng = StdRng::seed_from_u64(config.seed ^ 0x9F4B_72E0_1A33_4F0Cu64);
            causal_terms = select_logic_terms(
                &pool_specs,
                &row_map,
                &sites,
                config.logic_k_min,
                config.logic_k_max,
                config.logic_ld_max,
                config.logic_het_max,
                config.causal_maf_min,
                config.logic_af_min,
                config.logic_af_max,
                config.logic_max_iter,
                config.logic_effect_model,
                &mut logic_rng,
            )?;
        } else {
            let mut causal_progress_seen = 0usize;
            let mut causal_last_notified = 0usize;
            let mut sel_rng = StdRng::seed_from_u64(config.seed ^ 0xA54D_3F9E_6721_8CB7u64);
            let selected = select_additive_indices(
                &sites,
                config.causal_count,
                constraint_pools.as_slice(),
                config.causal_maf_min,
                &mut sel_rng,
            )?;
            let selected_set: HashSet<usize> = selected.iter().copied().collect();
            let mut row_map: HashMap<usize, Vec<f32>> = HashMap::with_capacity(selected.len());
            iterate_filtered_rows(
                &config.path_or_prefix,
                config.delimiter.as_deref(),
                config.maf_threshold,
                config.max_missing_rate,
                config.het_threshold,
                config.snps_only,
                |kept_idx, row, _site| {
                    causal_progress_seen = causal_progress_seen.saturating_add(1);
                    if progress_overall_total > 0 {
                        g2p_progress_notify(
                            config.progress_callback.as_ref(),
                            "causal_additive",
                            causal_offset.saturating_add(causal_progress_seen),
                            progress_overall_total,
                            progress_every,
                            &mut causal_last_notified,
                            false,
                        )?;
                    }
                    if selected_set.contains(&kept_idx) {
                        row_map.insert(kept_idx, row.to_vec());
                    }
                    Ok(())
                },
            )?;
            if progress_overall_total > 0 {
                g2p_progress_notify(
                    config.progress_callback.as_ref(),
                    "causal_additive",
                    causal_offset.saturating_add(progress_site_total),
                    progress_overall_total,
                    progress_every,
                    &mut causal_last_notified,
                    true,
                )?;
            }
            causal_terms = build_additive_terms(&selected, &row_map, &sites)?;
        }
    }

    if !causal_terms.is_empty() && causal_pve_target > 0.0 {
        let gamma0 =
            build_causal_series_effects(causal_terms.len(), DEFAULT_CAUSAL_SERIES_ALPHA, &mut rng);
        let mut causal_score = vec![0.0_f64; n];
        for (coef, term) in gamma0.iter().zip(causal_terms.iter()) {
            axpy_inplace(&mut causal_score, &term.values, *coef);
        }
        let scale = variance_scale_factor(variance_f64(&causal_score), causal_pve_target);
        if scale > 0.0 {
            for (coef, term) in gamma0.iter().zip(causal_terms.iter_mut()) {
                term.effect = scale * *coef;
            }
            for term in causal_terms.iter() {
                axpy_inplace(&mut causal_component, &term.values, term.effect);
                axpy_inplace(&mut y, &term.values, term.effect);
            }
        }
    }
    let realized_summary =
        build_realized_summary(&y, &causal_component, &bg_effects, &residual_effects);

    let logic_suffix = if logic_requested
        && matches!(
            config.logic_effect_model,
            LogicEffectModel::CenteredInteraction
        ) {
        "_centered"
    } else {
        ""
    };
    let default_trait = format!(
        "sim_bg{:.3}_cs{:.3}_{}{}",
        config.bg_pve, causal_pve_target, background_source, logic_suffix,
    );
    let trait_name = config
        .trait_name
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or(default_trait);

    if let Some(prefix) = config.pheno_prefix.as_ref() {
        write_pheno_files(
            prefix,
            &sample_ids,
            &y,
            &trait_name,
            config.na_rate,
            config.seed,
        )?;
    }
    if let Some(path) = config.random_effects_path.as_ref() {
        write_sample_background_effects(
            path,
            &sample_ids,
            &bg_effects,
            background_source.as_str(),
        )?;
    }
    if let Some(path) = config.fixed_effects_path.as_ref() {
        write_fixed_effects(path, &causal_terms, &sites)?;
    }
    if let Some(path) = config.causal_sites_path.as_ref() {
        write_causal_sites(path, &causal_terms, &sites)?;
    }
    if runtime_progress_total > 0 {
        let mut final_last_notified = runtime_progress_total;
        g2p_progress_notify(
            config.progress_callback.as_ref(),
            "finalize",
            runtime_progress_total,
            runtime_progress_total,
            progress_every,
            &mut final_last_notified,
            true,
        )?;
    }

    let causal_sites: Vec<(String, i32, i32)> = {
        let mut seen: HashSet<(String, i32)> = HashSet::new();
        let mut out: Vec<(String, i32, i32)> = Vec::new();
        for term in causal_terms.iter() {
            for &idx in term.members.iter() {
                let site = &sites[idx];
                let key = (site.chrom.clone(), site.pos);
                if seen.insert(key.clone()) {
                    out.push((key.0, key.1, key.1));
                }
            }
        }
        out
    };

    let fixed_rows: Vec<(usize, String, String, String, String, f64)> = causal_terms
        .iter()
        .enumerate()
        .map(|(i, term)| {
            let kind = if term.mode.is_some() {
                "logic_gate".to_string()
            } else {
                "additive".to_string()
            };
            let logic = term
                .mode
                .map(|mode| logic_mode_code(mode).to_string())
                .unwrap_or_else(|| "single".to_string());
            let site_text = term
                .members
                .iter()
                .map(|&idx| format!("{}:{}", sites[idx].chrom, sites[idx].pos))
                .collect::<Vec<String>>()
                .join(";");
            (
                i + 1,
                kind,
                logic,
                site_text,
                term.label.clone(),
                term.effect,
            )
        })
        .collect();

    Ok(G2pSimResult {
        sample_ids,
        phenotype: y,
        trait_name,
        causal_sites,
        fixed_rows,
        n_background_sites: 0,
        n_causal_terms: causal_terms.len(),
        bg_pve: config.bg_pve,
        causal_pve: causal_pve_target,
        residual_var: residual_var_eff,
        logic_effect_model: logic_effect_model_name(config.logic_effect_model).to_string(),
        background_source,
        background_factorization,
        realized_summary,
    })
}

#[pyfunction(name = "g2p_simulate")]
#[pyo3(signature = (
    path_or_prefix,
    chunk_size=100_000,
    maf_threshold=0.02_f32,
    causal_maf_min=0.02_f32,
    max_missing_rate=0.05_f32,
    het_threshold=None,
    seed=1_u64,
    residual_var=1.0_f64,
    bg_pve=0.5_f64,
    background_dist="normal",
    gamma_shape=1.0_f64,
    gamma_scale=1.0_f64,
    laplace_scale=1.0_f64,
    causal_count=1_usize,
    causal_pve=None,
    bim_ranges=None,
    bim_range_groups=None,
    logic_mode=None,
    logic_size_weights=None,
    logic_gate_count=None,
    logic_k_min=2_usize,
    logic_k_max=2_usize,
    logic_ld_max=1.0_f64,
    logic_het_max=1.0_f64,
    logic_af_min=0.0_f64,
    logic_af_max=1.0_f64,
    logic_max_iter=256_usize,
    logic_window_bp=None,
    logic_effect_model="gate",
    delimiter=None,
    snps_only=false,
    pheno_prefix=None,
    fixed_effects_path=None,
    random_effects_path=None,
    causal_sites_path=None,
    grm=None,
    trait_name=None,
    na_rate=0.1_f64,
    progress_callback=None,
    progress_total_hint=None,
    progress_every=10_000_usize,
))]
pub fn g2p_simulate_py<'py>(
    py: Python<'py>,
    path_or_prefix: String,
    chunk_size: usize,
    maf_threshold: f32,
    causal_maf_min: f32,
    max_missing_rate: f32,
    het_threshold: Option<f32>,
    seed: u64,
    residual_var: f64,
    bg_pve: f64,
    background_dist: &str,
    gamma_shape: f64,
    gamma_scale: f64,
    laplace_scale: f64,
    causal_count: usize,
    causal_pve: Option<f64>,
    bim_ranges: Option<Vec<(String, i32, i32)>>,
    bim_range_groups: Option<Vec<Vec<(String, i32, i32)>>>,
    logic_mode: Option<String>,
    logic_size_weights: Option<Vec<f64>>,
    logic_gate_count: Option<usize>,
    logic_k_min: usize,
    logic_k_max: usize,
    logic_ld_max: f64,
    logic_het_max: f64,
    logic_af_min: f64,
    logic_af_max: f64,
    logic_max_iter: usize,
    logic_window_bp: Option<i32>,
    logic_effect_model: &str,
    delimiter: Option<String>,
    snps_only: bool,
    pheno_prefix: Option<String>,
    fixed_effects_path: Option<String>,
    random_effects_path: Option<String>,
    causal_sites_path: Option<String>,
    grm: Option<PyReadonlyArray2<'py, f64>>,
    trait_name: Option<String>,
    na_rate: f64,
    progress_callback: Option<Py<PyAny>>,
    progress_total_hint: Option<usize>,
    progress_every: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if !(0.0..=1.0).contains(&bg_pve) {
        return Err(PyValueError::new_err("bg_pve must be within [0, 1]."));
    }
    if !(0.0..=1.0).contains(&na_rate) {
        return Err(PyValueError::new_err("na_rate must be within [0, 1]."));
    }
    if residual_var < 0.0 || !residual_var.is_finite() {
        return Err(PyValueError::new_err(
            "residual_var must be finite and >= 0.",
        ));
    }
    if let Some(het_threshold) = het_threshold {
        if !(0.0..=1.0).contains(&het_threshold) {
            return Err(PyValueError::new_err(
                "het_threshold must be within [0, 1].",
            ));
        }
    }
    if !(0.0..=0.5).contains(&causal_maf_min) {
        return Err(PyValueError::new_err(
            "causal_maf_min must be within [0, 0.5].",
        ));
    }
    if logic_k_min == 0 {
        return Err(PyValueError::new_err("logic_k_min must be > 0."));
    }
    if logic_k_max < logic_k_min {
        return Err(PyValueError::new_err("logic_k_max must be >= logic_k_min."));
    }
    if !(0.0..=1.0).contains(&logic_ld_max) {
        return Err(PyValueError::new_err("logic_ld_max must be within [0, 1]."));
    }
    if !(0.0..=1.0).contains(&logic_het_max) {
        return Err(PyValueError::new_err(
            "logic_het_max must be within [0, 1].",
        ));
    }
    if !(0.0..=1.0).contains(&logic_af_min) || !(0.0..=1.0).contains(&logic_af_max) {
        return Err(PyValueError::new_err(
            "logic_af_min/logic_af_max must be within [0, 1].",
        ));
    }
    if logic_af_min > logic_af_max {
        return Err(PyValueError::new_err(
            "logic_af_min must be <= logic_af_max.",
        ));
    }
    if let Some(weights) = logic_size_weights.as_ref() {
        validate_logic_size_weights(weights)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    let _ = chunk_size;
    let _bg_dist =
        parse_background_dist(background_dist).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let logic_effect_model = parse_logic_effect_model(logic_effect_model)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let _ = (gamma_shape, gamma_scale, laplace_scale);
    let (grm_vec, grm_n) = if let Some(arr) = grm {
        let mat = arr.as_array();
        if mat.ndim() != 2 || mat.shape()[0] != mat.shape()[1] {
            return Err(PyValueError::new_err(
                "grm must be a square float64 matrix.",
            ));
        }
        let n = mat.shape()[0];
        let vec = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mat.iter().copied().collect(),
        };
        (Some(vec), Some(n))
    } else {
        (None, None)
    };
    let config = G2pSimConfig {
        path_or_prefix,
        delimiter,
        maf_threshold,
        causal_maf_min,
        max_missing_rate,
        het_threshold,
        seed,
        residual_var,
        bg_pve,
        causal_count,
        causal_pve,
        bim_ranges: bim_ranges.unwrap_or_default(),
        bim_range_groups: bim_range_groups.unwrap_or_default(),
        logic_mode,
        logic_size_weights,
        logic_gate_count,
        logic_k_min,
        logic_k_max,
        logic_ld_max,
        logic_het_max,
        logic_af_min,
        logic_af_max,
        logic_max_iter,
        logic_window_bp,
        logic_effect_model,
        snps_only,
        pheno_prefix,
        fixed_effects_path,
        random_effects_path,
        causal_sites_path,
        grm: grm_vec,
        grm_n,
        trait_name,
        na_rate,
        progress_callback,
        progress_total_hint,
        progress_every,
    };
    let sim = py
        .detach(move || g2p_simulate_core(config))
        .map_err(PyRuntimeError::new_err)?;

    let G2pSimResult {
        sample_ids,
        phenotype,
        trait_name,
        causal_sites,
        fixed_rows,
        n_background_sites,
        n_causal_terms,
        bg_pve,
        causal_pve,
        residual_var,
        logic_effect_model,
        background_source,
        background_factorization,
        realized_summary,
    } = sim;

    #[allow(deprecated)]
    let y_arr = PyArray1::from_owned_array(py, Array1::from_vec(phenotype));
    let out = PyDict::new(py);
    out.set_item("sample_ids", sample_ids)?;
    out.set_item("phenotype", y_arr)?;
    out.set_item("trait_name", trait_name)?;
    out.set_item("causal_sites", causal_sites)?;
    out.set_item("fixed_rows", fixed_rows)?;
    out.set_item("n_background_sites", n_background_sites)?;
    out.set_item("n_causal_terms", n_causal_terms)?;
    out.set_item("bg_pve", bg_pve)?;
    out.set_item("causal_pve", causal_pve)?;
    out.set_item("ve", residual_var)?;
    out.set_item("residual_var", residual_var)?;
    out.set_item("logic_effect_model", logic_effect_model)?;
    out.set_item("background_source", background_source)?;
    out.set_item("background_factorization", background_factorization)?;
    let realized = PyDict::new(py);
    realized.set_item("mean_y", realized_summary.mean_y)?;
    realized.set_item("var_y", realized_summary.var_y)?;
    realized.set_item("mean_causal", realized_summary.mean_causal)?;
    realized.set_item("mean_background", realized_summary.mean_background)?;
    realized.set_item("mean_residual", realized_summary.mean_residual)?;
    realized.set_item("var_causal", realized_summary.var_causal)?;
    realized.set_item("var_background", realized_summary.var_background)?;
    realized.set_item("var_residual", realized_summary.var_residual)?;
    realized.set_item(
        "cov_causal_background",
        realized_summary.cov_causal_background,
    )?;
    realized.set_item("cov_causal_residual", realized_summary.cov_causal_residual)?;
    realized.set_item(
        "cov_background_residual",
        realized_summary.cov_background_residual,
    )?;
    realized.set_item("pve_causal", realized_summary.pve_causal)?;
    realized.set_item("pve_background", realized_summary.pve_background)?;
    realized.set_item("pve_residual", realized_summary.pve_residual)?;
    out.set_item("realized_summary", realized)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(prefix: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_{prefix}_{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_centered_interaction_is_orthogonal_to_main_effects() {
        let a = vec![0u8, 0, 1, 1, 0, 1];
        let b = vec![0u8, 1, 0, 1, 1, 1];
        let literal_rows = logic_gate_literal_rows(&[a.clone(), b.clone()], LogicGateMode::A)
            .expect("logic literal rows");
        let gate = logic_gate_indicator_from_literals(&literal_rows, false);
        let z = residualize_logic_indicator_against_main_effects(&gate, &literal_rows)
            .expect("centered interaction residualization");
        let sum_z = z.iter().sum::<f64>();
        let dot_a = z
            .iter()
            .zip(literal_rows[0].iter())
            .map(|(zi, &xi)| zi * xi as f64)
            .sum::<f64>();
        let dot_b = z
            .iter()
            .zip(literal_rows[1].iter())
            .map(|(zi, &xi)| zi * xi as f64)
            .sum::<f64>();
        assert!(sum_z.abs() < 1e-8);
        assert!(dot_a.abs() < 1e-8);
        assert!(dot_b.abs() < 1e-8);
        assert!(variance_f64(&z) > 1e-12);
    }

    #[test]
    fn logic_gate_modes_match_expected_patterns() {
        let a = vec![0u8, 0, 1, 1];
        let b = vec![0u8, 1, 0, 1];
        assert_eq!(
            logic_gate_indicator(&[a.clone(), b.clone()], LogicGateMode::A).unwrap(),
            vec![0u8, 0, 0, 1]
        );
        assert_eq!(
            logic_gate_indicator(&[a.clone(), b.clone()], LogicGateMode::Na).unwrap(),
            vec![1u8, 1, 1, 0]
        );
        assert_eq!(
            logic_gate_indicator(&[a.clone(), b.clone()], LogicGateMode::An).unwrap(),
            vec![0u8, 0, 1, 0]
        );
        assert_eq!(
            logic_gate_indicator(&[a, b], LogicGateMode::Nan).unwrap(),
            vec![1u8, 0, 0, 0]
        );
    }

    #[test]
    fn gaussian_noise_is_sampled_at_requested_scale() {
        let mut rng = StdRng::seed_from_u64(11);
        let eps = sample_gaussian_noise_with_variance(4096, 0.35, &mut rng);
        assert!((variance_f64(&eps) - 0.35).abs() < 0.03);
    }

    #[test]
    fn trace_scaled_identity_grm_matches_direct_gaussian_sampling() {
        let n = 4usize;
        let grm = vec![
            1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut rng_bg = StdRng::seed_from_u64(23);
        let (observed, factorization) =
            sample_background_effects_from_grm_trace_scaled(&grm, n, 0.25, &mut rng_bg).unwrap();
        assert_eq!(factorization, "cholesky");
        let mut rng_direct = StdRng::seed_from_u64(23);
        let expected = sample_gaussian_noise_with_variance(n, 0.25, &mut rng_direct);
        for (obs, exp) in observed.iter().zip(expected.iter()) {
            assert!((obs - exp).abs() < 1e-12);
        }
    }

    #[test]
    fn indefinite_grm_falls_back_to_eigh_and_clips_negative_eigenvalues() {
        let grm = vec![1.0_f64, 2.0_f64, 2.0_f64, 1.0_f64];
        let (factor, trace_psd) = build_sampling_factor_with_fallback(&grm, 2, "GRM").unwrap();
        assert!((trace_psd - 3.0_f64).abs() < 1e-12);
        let root = match factor {
            SamplingFactor::DenseSquareRoot(root) => root,
            SamplingFactor::CholeskyLower(_) => {
                panic!("indefinite GRM should fall back to eigh clipping")
            }
        };

        let mut recon = vec![0.0_f64; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0_f64;
                for k in 0..2 {
                    acc += root[i * 2 + k] * root[j * 2 + k];
                }
                recon[i * 2 + j] = acc;
            }
        }
        let expected = vec![1.5_f64, 1.5_f64, 1.5_f64, 1.5_f64];
        for (obs, exp) in recon.iter().zip(expected.iter()) {
            assert!((obs - exp).abs() < 1e-12);
        }
    }

    #[test]
    fn realized_summary_reports_component_moments_and_covariances() {
        let causal = vec![1.0_f64, -1.0, 0.0];
        let background = vec![0.0_f64, 1.0, -1.0];
        let residual = vec![1.0_f64, 1.0, 1.0];
        let y = vec![2.0_f64, 1.0, 0.0];
        let summary = build_realized_summary(&y, &causal, &background, &residual);
        assert!((summary.mean_y - 1.0).abs() < 1e-12);
        assert!((summary.var_y - (2.0 / 3.0)).abs() < 1e-12);
        assert!(summary.mean_causal.abs() < 1e-12);
        assert!(summary.mean_background.abs() < 1e-12);
        assert!((summary.mean_residual - 1.0).abs() < 1e-12);
        assert!((summary.var_causal - (2.0 / 3.0)).abs() < 1e-12);
        assert!((summary.var_background - (2.0 / 3.0)).abs() < 1e-12);
        assert!(summary.var_residual.abs() < 1e-12);
        assert!((summary.cov_causal_background + (1.0 / 3.0)).abs() < 1e-12);
        assert!(summary.cov_causal_residual.abs() < 1e-12);
        assert!(summary.cov_background_residual.abs() < 1e-12);
        assert!((summary.pve_causal - 1.0).abs() < 1e-12);
        assert!((summary.pve_background - 1.0).abs() < 1e-12);
        assert!(summary.pve_residual.abs() < 1e-12);
    }

    #[test]
    fn causal_series_effects_match_geometric_magnitudes() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut observed = build_causal_series_effects(4, DEFAULT_CAUSAL_SERIES_ALPHA, &mut rng);
        observed.sort_by(|a, b| a.total_cmp(b));
        let mut expected = vec![
            DEFAULT_CAUSAL_SERIES_ALPHA,
            DEFAULT_CAUSAL_SERIES_ALPHA.powi(2),
            DEFAULT_CAUSAL_SERIES_ALPHA.powi(3),
            DEFAULT_CAUSAL_SERIES_ALPHA.powi(4),
        ];
        expected.sort_by(|a, b| a.total_cmp(b));
        for (obs, exp) in observed.iter().zip(expected.iter()) {
            assert!((obs - exp).abs() < 1e-12);
        }
    }

    #[test]
    fn iterate_filtered_rows_keeps_non_acgt_sites_when_snps_only_disabled() {
        let dir = make_temp_dir("g2p_iter_txt");
        let prefix = dir.join("geno");
        let txt_path = dir.join("geno.txt");
        let id_path = dir.join("geno.id");

        {
            let mut f = File::create(&txt_path).unwrap();
            writeln!(f, "0\t1\t2").unwrap();
        }
        {
            let mut f = File::create(&id_path).unwrap();
            writeln!(f, "s1").unwrap();
            writeln!(f, "s2").unwrap();
            writeln!(f, "s3").unwrap();
        }

        let mut kept_false = 0usize;
        let sample_ids = iterate_filtered_rows(
            prefix.to_str().unwrap(),
            None,
            0.0,
            1.0,
            None,
            false,
            |_, row, site| {
                assert_eq!(row, &[0.0_f32, 1.0_f32, 2.0_f32]);
                assert_eq!(site.ref_allele, "N");
                assert_eq!(site.alt_allele, "N");
                kept_false += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(sample_ids, vec!["s1", "s2", "s3"]);
        assert_eq!(kept_false, 1);

        let mut kept_true = 0usize;
        let sample_ids_true = iterate_filtered_rows(
            prefix.to_str().unwrap(),
            None,
            0.0,
            1.0,
            None,
            true,
            |_, _, _| {
                kept_true += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(sample_ids_true, vec!["s1", "s2", "s3"]);
        assert_eq!(kept_true, 0);

        let _ = fs::remove_file(txt_path);
        let _ = fs::remove_file(id_path);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn iterate_filtered_rows_respects_optional_het_threshold() {
        let dir = make_temp_dir("g2p_iter_het");
        let prefix = dir.join("geno");
        let txt_path = dir.join("geno.txt");
        let id_path = dir.join("geno.id");

        {
            let mut f = File::create(&txt_path).unwrap();
            writeln!(f, "0\t1\t2\t1").unwrap();
        }
        {
            let mut f = File::create(&id_path).unwrap();
            writeln!(f, "s1").unwrap();
            writeln!(f, "s2").unwrap();
            writeln!(f, "s3").unwrap();
            writeln!(f, "s4").unwrap();
        }

        let mut kept_disabled = 0usize;
        iterate_filtered_rows(
            prefix.to_str().unwrap(),
            None,
            0.0,
            1.0,
            None,
            false,
            |_, _, _| {
                kept_disabled += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(kept_disabled, 1);

        let mut kept_enabled = 0usize;
        iterate_filtered_rows(
            prefix.to_str().unwrap(),
            None,
            0.0,
            1.0,
            Some(0.25_f32),
            false,
            |_, _, _| {
                kept_enabled += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(kept_enabled, 0);

        let _ = fs::remove_file(txt_path);
        let _ = fs::remove_file(id_path);
        let _ = fs::remove_dir_all(dir);
    }
}
