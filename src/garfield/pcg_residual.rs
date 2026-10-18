use crate::bedmath::is_identity_indices;
use crate::decode::decode_prepared_additive_block_packed_f32;
use crate::garfield::residual::{
    estimate_lmm_lambda_from_grm_rust, null_validation_seed, standardize_residualized_y,
    GarfieldResidualResult,
};
use crate::gblup::{build_grm_from_meta_stream, StreamKernelMode};
use crate::gfcore::read_fam;
use crate::gload::WindowedBedMatrix;
use crate::he::he_variance_components_meta_stream_with_covariates;
use crate::linalg::{cholesky_inplace, cholesky_solve_into};
use crate::pcg::{pcg_preconditioner_fn, pcg_solve, PcgOperator};
use crate::stats_common::get_cached_pool;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::Arc;

// This module intentionally implements an rrBLUP additive projection, not an
// LMM residualizer.  Lambda is selected from a GRM/REML or HE variance-ratio
// estimate, converted to marker space, and the marker coefficients are then
// solved with PCG before y - C*alpha - W*beta is formed.

#[derive(Clone, Debug)]
pub(crate) struct PcgResidualConfig {
    pub block_rows: usize,
    pub threads: usize,
    pub std_eps: f64,
    pub small_n_max: usize,
    pub reml_low: f64,
    pub reml_high: f64,
    pub reml_max_iter: usize,
    pub reml_tol: f64,
    pub pcg_max_iter: usize,
    pub pcg_tol: f64,
    pub he_trace_samples: usize,
    pub he_trace_probe_batch: usize,
    pub he_max_iter: usize,
    pub he_tol: f64,
    pub seed: u64,
}

impl Default for PcgResidualConfig {
    fn default() -> Self {
        Self {
            block_rows: 4096,
            threads: 0,
            std_eps: 1e-12_f64,
            small_n_max: 15_000,
            reml_low: -5.0,
            reml_high: 5.0,
            reml_max_iter: 50,
            reml_tol: 1e-3,
            pcg_max_iter: 100,
            pcg_tol: 1e-4,
            he_trace_samples: 32,
            he_trace_probe_batch: 64,
            he_max_iter: 32,
            he_tol: 1e-6,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
struct PcgLambdaEstimate {
    lambda_vc: f64,
    lambda_marker: f64,
    sigma_g2: f64,
    sigma_e2: f64,
    pve: f64,
    lambda_converged: bool,
    lambda_iters: usize,
    lambda_rel_res: f64,
    m_effective: usize,
    source: String,
    eigh_backend: String,
    eigh_elapsed: f64,
}

struct CovariateProjector {
    n: usize,
    p: usize,
    c: Vec<f64>,
    chol_ctc: Vec<f64>,
}

#[inline]
fn oriented_minor_frequency(raw_af: f32, row_flip: bool) -> f32 {
    let raw_af = raw_af.clamp(0.0_f32, 1.0_f32);
    if row_flip {
        (1.0_f32 - raw_af).min(0.5_f32)
    } else {
        raw_af.min(0.5_f32)
    }
}

impl CovariateProjector {
    fn new(c: &[f64], n: usize, p: usize) -> Result<Self, String> {
        if p == 0 {
            return Err(
                "PCG residualization requires at least one fixed-effect column".to_string(),
            );
        }
        if c.len() != n.saturating_mul(p) {
            return Err(format!(
                "fixed-effect design length mismatch: got {}, expected {}",
                c.len(),
                n.saturating_mul(p)
            ));
        }
        if c.iter().any(|v| !v.is_finite()) {
            return Err("fixed-effect design contains non-finite values".to_string());
        }
        if n <= p {
            return Err(format!(
                "PCG residualization requires n > rank(C): n={n}, p={p}"
            ));
        }

        let mut ctc = vec![0.0_f64; p * p];
        for row in c.chunks_exact(p) {
            for a in 0..p {
                for b in 0..=a {
                    ctc[a * p + b] += row[a] * row[b];
                }
            }
        }
        for a in 0..p {
            for b in 0..a {
                ctc[b * p + a] = ctc[a * p + b];
            }
        }
        cholesky_inplace(&mut ctc, p).ok_or_else(|| {
            format!(
                "PCG residualization fixed-effect design is singular or ill-conditioned (p={p})"
            )
        })?;

        Ok(Self {
            n,
            p,
            c: c.to_vec(),
            chol_ctc: ctc,
        })
    }

    fn coefficients_for_f64(&self, y: &[f64]) -> Result<Vec<f64>, String> {
        if y.len() != self.n {
            return Err(format!(
                "fixed-effect response length mismatch: got {}, expected {}",
                y.len(),
                self.n
            ));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err("fixed-effect response contains non-finite values".to_string());
        }
        let mut cty = vec![0.0_f64; self.p];
        for (i, row) in self.c.chunks_exact(self.p).enumerate() {
            for j in 0..self.p {
                cty[j] += row[j] * y[i];
            }
        }
        let mut beta = vec![0.0_f64; self.p];
        cholesky_solve_into(&self.chol_ctc, self.p, &cty, &mut beta);
        Ok(beta)
    }

    fn project_f64_in_place(&self, values: &mut [f64]) -> Result<(), String> {
        let beta = self.coefficients_for_f64(values)?;
        for (i, row) in self.c.chunks_exact(self.p).enumerate() {
            let fitted = row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum::<f64>();
            values[i] -= fitted;
        }
        Ok(())
    }

    fn project_f32_in_place(&self, values: &mut [f32]) -> Result<(), String> {
        if values.len() != self.n {
            return Err(format!(
                "projected sample vector length mismatch: got {}, expected {}",
                values.len(),
                self.n
            ));
        }
        let as_f64 = values.iter().map(|v| *v as f64).collect::<Vec<_>>();
        let beta = self.coefficients_for_f64(as_f64.as_slice())?;
        for (i, row) in self.c.chunks_exact(self.p).enumerate() {
            let fitted = row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum::<f64>();
            values[i] = (as_f64[i] - fitted) as f32;
        }
        Ok(())
    }

    fn fitted_from_coefficients(&self, beta: &[f64]) -> Result<Vec<f64>, String> {
        if beta.len() != self.p {
            return Err(format!(
                "fixed-effect coefficient length mismatch: got {}, expected {}",
                beta.len(),
                self.p
            ));
        }
        Ok(self
            .c
            .chunks_exact(self.p)
            .map(|row| row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum::<f64>())
            .collect())
    }
}

trait MarkerDesign {
    fn n_markers(&self) -> usize;
    fn n_samples(&self) -> usize;
    fn marker_to_samples(&mut self, marker_beta: &[f32], out: &mut [f32]) -> Result<(), String>;
    fn samples_to_markers(&mut self, samples: &[f32], out: &mut [f32]) -> Result<(), String>;
}

struct BedMarkerDesign {
    matrix: WindowedBedMatrix,
    n_samples_full: usize,
    sample_idx: Vec<usize>,
    full_sample_fast: bool,
    row_source_indices: Vec<usize>,
    row_flip: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    block_rows: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl BedMarkerDesign {
    #[allow(clippy::too_many_arguments)]
    fn open(
        prefix: &str,
        row_source_indices: Vec<usize>,
        row_flip: Vec<bool>,
        row_maf: Vec<f32>,
        sample_idx: Vec<usize>,
        block_rows: usize,
        threads: usize,
        std_eps: f32,
    ) -> Result<(Self, usize), String> {
        if row_source_indices.is_empty() {
            return Err("PCG residualization received zero QC marker rows".to_string());
        }
        if row_source_indices.len() != row_flip.len() || row_flip.len() != row_maf.len() {
            return Err("PCG residualization marker metadata length mismatch".to_string());
        }
        let n_samples_full = read_fam(prefix)
            .map_err(|e| format!("failed to read {prefix}.fam for PCG residualization: {e}"))?
            .len();
        if n_samples_full == 0 {
            return Err("PCG residualization BED input has no samples".to_string());
        }
        if sample_idx.is_empty() || sample_idx.iter().any(|&idx| idx >= n_samples_full) {
            return Err("PCG residualization sample index is empty or out of range".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let window_mb = block_rows
            .max(1)
            .saturating_mul(bytes_per_snp.max(1))
            .saturating_mul(2)
            .div_ceil(1024 * 1024)
            .max(1);
        let matrix = WindowedBedMatrix::open(prefix, window_mb)?;
        if row_source_indices
            .iter()
            .any(|&idx| idx >= matrix.n_source_snps())
        {
            return Err("PCG residualization marker source index is out of range".to_string());
        }
        let mut active_source = Vec::with_capacity(row_source_indices.len());
        let mut active_flip = Vec::with_capacity(row_flip.len());
        let mut active_mean = Vec::with_capacity(row_maf.len());
        let mut active_inv_sd = Vec::with_capacity(row_maf.len());
        for ((&source, &flip), &maf) in row_source_indices
            .iter()
            .zip(row_flip.iter())
            .zip(row_maf.iter())
        {
            if !maf.is_finite() {
                return Err("PCG residualization row MAF contains non-finite values".to_string());
            }
            // `filter_marker_metadata` has already converted raw BED A2
            // frequencies to the oriented minor frequency used here.
            let p = maf.clamp(0.0_f32, 0.5_f32);
            let mean = 2.0_f32 * p;
            let variance = (2.0_f32 * p * (1.0_f32 - p)).max(0.0_f32);
            let inv_sd = if variance > std_eps.max(1e-12_f32) {
                variance.sqrt().recip()
            } else {
                0.0_f32
            };
            if inv_sd > 0.0 && inv_sd.is_finite() {
                active_source.push(source);
                active_flip.push(flip);
                active_mean.push(mean);
                active_inv_sd.push(inv_sd);
            }
        }
        if active_source.is_empty() {
            return Err(
                "PCG residualization found no effective standardized QC markers".to_string(),
            );
        }
        let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
        let full_sample_fast = is_identity_indices(sample_idx.as_slice(), n_samples_full);
        let design = Self {
            matrix,
            n_samples_full,
            sample_idx,
            full_sample_fast,
            row_source_indices: active_source,
            row_flip: active_flip,
            row_mean: active_mean,
            row_inv_sd: active_inv_sd,
            block_rows: block_rows.max(1),
            pool,
        };
        let m_effective = design.row_source_indices.len();
        Ok((design, m_effective))
    }

    fn decode_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let cur_rows = row_end.saturating_sub(row_start);
        if cur_rows == 0 {
            return Ok(());
        }
        let source_rows = &self.row_source_indices[row_start..row_end];
        let mut relative_rows = Vec::with_capacity(cur_rows);
        let packed = self
            .matrix
            .prepare_source_rows(source_rows, &mut relative_rows)?;
        decode_prepared_additive_block_packed_f32(
            packed,
            self.n_samples_full.div_ceil(4),
            self.n_samples_full,
            &self.row_flip[row_start..row_end],
            &self.row_mean[row_start..row_end],
            &self.row_inv_sd[row_start..row_end],
            self.sample_idx.as_slice(),
            self.full_sample_fast,
            None,
            Some(relative_rows.as_slice()),
            0,
            cur_rows,
            out,
            self.pool.as_ref(),
        )
    }

    fn for_each_block<F>(&mut self, mut worker: F) -> Result<(), String>
    where
        F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
    {
        let n = self.sample_idx.len();
        let row_step = self.block_rows.min(self.n_markers().max(1));
        let mut block = vec![0.0_f32; row_step.saturating_mul(n)];
        for row_start in (0..self.n_markers()).step_by(row_step) {
            let row_end = (row_start + row_step).min(self.n_markers());
            let cur_rows = row_end - row_start;
            self.decode_block(row_start, row_end, &mut block[..cur_rows * n])?;
            worker(row_start, row_end, &block[..cur_rows * n])?;
        }
        Ok(())
    }
}

impl MarkerDesign for BedMarkerDesign {
    fn n_markers(&self) -> usize {
        self.row_source_indices.len()
    }

    fn n_samples(&self) -> usize {
        self.sample_idx.len()
    }

    fn marker_to_samples(&mut self, marker_beta: &[f32], out: &mut [f32]) -> Result<(), String> {
        if marker_beta.len() != self.n_markers() || out.len() != self.n_samples() {
            return Err("BED marker_to_samples shape mismatch".to_string());
        }
        out.fill(0.0_f32);
        let pool = self.pool.clone();
        let n_samples = self.n_samples();
        self.for_each_block(|row_start, row_end, block| {
            let rows = row_end - row_start;
            let weights = &marker_beta[row_start..row_end];
            let mut run = || {
                out.par_iter_mut().enumerate().for_each(|(i, dst)| {
                    let mut acc = 0.0_f32;
                    for r in 0..rows {
                        acc += block[r * n_samples + i] * weights[r];
                    }
                    *dst += acc;
                });
            };
            if let Some(tp) = pool.as_ref() {
                tp.install(&mut run);
            } else {
                run();
            }
            Ok(())
        })
    }

    fn samples_to_markers(&mut self, samples: &[f32], out: &mut [f32]) -> Result<(), String> {
        if samples.len() != self.n_samples() || out.len() != self.n_markers() {
            return Err("BED samples_to_markers shape mismatch".to_string());
        }
        let pool = self.pool.clone();
        self.for_each_block(|row_start, row_end, block| {
            let rows = row_end - row_start;
            let out_chunk = &mut out[row_start..row_end];
            let mut run = || {
                out_chunk.par_iter_mut().enumerate().for_each(|(r, dst)| {
                    let row = &block[r * samples.len()..(r + 1) * samples.len()];
                    *dst = row
                        .iter()
                        .zip(samples.iter())
                        .map(|(x, v)| x * v)
                        .sum::<f32>();
                });
            };
            if let Some(tp) = pool.as_ref() {
                tp.install(&mut run);
            } else {
                run();
            }
            debug_assert_eq!(rows, out_chunk.len());
            Ok(())
        })
    }
}

fn filter_marker_metadata(
    row_source_indices: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    std_eps: f32,
) -> Result<(Vec<usize>, Vec<bool>, Vec<f32>), String> {
    if row_source_indices.len() != row_flip.len() || row_flip.len() != row_maf.len() {
        return Err("PCG residualization marker metadata length mismatch".to_string());
    }
    let mut source = Vec::with_capacity(row_source_indices.len());
    let mut flip = Vec::with_capacity(row_flip.len());
    let mut maf = Vec::with_capacity(row_maf.len());
    for ((&source_idx, &row_flip), &row_maf) in row_source_indices
        .iter()
        .zip(row_flip.iter())
        .zip(row_maf.iter())
    {
        if !row_maf.is_finite() {
            return Err("PCG residualization row MAF contains non-finite values".to_string());
        }
        let p = oriented_minor_frequency(row_maf, row_flip);
        let variance = (2.0_f32 * p * (1.0_f32 - p)).max(0.0_f32);
        if variance > std_eps.max(1e-12_f32) {
            source.push(source_idx);
            flip.push(row_flip);
            maf.push(p);
        }
    }
    if source.is_empty() {
        return Err("PCG residualization found no effective standardized QC markers".to_string());
    }
    Ok((source, flip, maf))
}

#[allow(clippy::too_many_arguments)]
fn estimate_lambda(
    prefix: &str,
    row_source_indices: &[usize],
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    y: &[f64],
    x_cov: Option<&[f64]>,
    q_cov: usize,
    add_intercept: bool,
    config: &PcgResidualConfig,
    m_effective: usize,
) -> Result<PcgLambdaEstimate, String> {
    let estimate = if sample_idx.len() <= config.small_n_max {
        let (grm, _, _) = build_grm_from_meta_stream(
            prefix,
            row_source_indices,
            row_flip,
            row_maf,
            sample_idx,
            StreamKernelMode::StandardizedAdditive,
            config.block_rows,
            config.threads,
            None,
            0,
            None::<fn(usize, usize) -> Result<(), String>>,
        )?;
        let reml = estimate_lmm_lambda_from_grm_rust(
            grm.as_slice(),
            sample_idx.len(),
            y,
            x_cov,
            q_cov,
            config.threads,
            config.reml_low,
            config.reml_high,
            config.reml_max_iter,
            config.reml_tol,
            add_intercept,
            config.small_n_max,
            false,
        )?;
        (
            reml.lambda_vc,
            reml.sigma_g2,
            reml.sigma_e2,
            reml.pve,
            reml.converged,
            reml.iters,
            reml.rel_res,
            "small_n_grm_eigh_brent_reml".to_string(),
            reml.eigh_backend,
            reml.eigh_elapsed,
        )
    } else {
        let pool = get_cached_pool(config.threads).map_err(|e| e.to_string())?;
        let he = he_variance_components_meta_stream_with_covariates(
            prefix,
            row_source_indices,
            row_flip,
            row_maf,
            sample_idx,
            y,
            x_cov,
            q_cov,
            config.he_trace_samples,
            config.he_trace_probe_batch,
            config.block_rows,
            config.std_eps,
            false,
            config.he_max_iter,
            config.he_tol,
            config.seed,
            false,
            256,
            None,
            pool.as_ref(),
        )?;
        (
            he.lambda,
            he.sigma_g2,
            he.sigma_e2,
            he.h2,
            he.converged,
            he.iters,
            he.rel_res,
            "large_n_he".to_string(),
            "not_used".to_string(),
            0.0,
        )
    };
    let (
        lambda_vc,
        sigma_g2,
        sigma_e2,
        pve,
        lambda_converged,
        lambda_iters,
        lambda_rel_res,
        source,
        eigh_backend,
        eigh_elapsed,
    ) = estimate;
    let lambda_marker = if lambda_vc.is_finite() && lambda_vc > 0.0 {
        lambda_vc * m_effective as f64
    } else {
        f64::INFINITY
    };
    Ok(PcgLambdaEstimate {
        lambda_vc,
        lambda_marker,
        sigma_g2,
        sigma_e2,
        pve,
        lambda_converged,
        lambda_iters,
        lambda_rel_res,
        m_effective,
        source,
        eigh_backend,
        eigh_elapsed,
    })
}

fn make_null_residual_permutations(
    residualized_y: &[f64],
    repeats: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    if repeats == 0 || residualized_y.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(repeats);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..repeats {
        let mut draw = residualized_y.to_vec();
        draw.shuffle(&mut rng);
        out.push(draw);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn garfield_residualize_pcg_from_bed(
    prefix: &str,
    row_source_indices: Vec<usize>,
    row_flip: Vec<bool>,
    row_maf: Vec<f32>,
    sample_idx: Vec<usize>,
    y: Vec<f64>,
    x_cov: Option<Vec<f64>>,
    q_cov: usize,
    threads: usize,
    add_intercept: bool,
    mut config: PcgResidualConfig,
    null_repeats: usize,
    null_seed: u64,
) -> Result<GarfieldResidualResult, String> {
    if sample_idx.is_empty() {
        return Err("PCG residualization requires at least one selected sample".to_string());
    }
    if y.len() != sample_idx.len() {
        return Err(format!(
            "PCG residualization y length mismatch: got {}, expected {}",
            y.len(),
            sample_idx.len()
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err("PCG residualization y contains non-finite values".to_string());
    }
    if let Some(cov) = x_cov.as_ref() {
        if cov.len() != sample_idx.len().saturating_mul(q_cov) {
            return Err(format!(
                "PCG residualization x_cov length mismatch: got {}, expected {}",
                cov.len(),
                sample_idx.len().saturating_mul(q_cov)
            ));
        }
    }
    config.threads = if threads == 0 {
        config.threads
    } else {
        threads
    };
    let std_eps = config.std_eps.max(1e-12_f64) as f32;
    let (active_source, active_flip, active_maf) = filter_marker_metadata(
        row_source_indices.as_slice(),
        row_flip.as_slice(),
        row_maf.as_slice(),
        std_eps,
    )?;
    let (mut design, m_effective) = BedMarkerDesign::open(
        prefix,
        active_source.clone(),
        active_flip.clone(),
        active_maf.clone(),
        sample_idx.clone(),
        config.block_rows,
        config.threads,
        std_eps,
    )?;
    let (c, p) = super::residual::build_design_matrix(
        x_cov.as_deref(),
        sample_idx.len(),
        q_cov,
        add_intercept,
    )?;
    let lambda = estimate_lambda(
        prefix,
        active_source.as_slice(),
        active_flip.as_slice(),
        active_maf.as_slice(),
        sample_idx.as_slice(),
        y.as_slice(),
        x_cov.as_deref(),
        q_cov,
        add_intercept,
        &config,
        m_effective,
    )?;

    let fit = if lambda.lambda_marker.is_finite() && lambda.lambda_marker > 0.0 {
        fit_projected_rrblup(
            &mut design,
            c.as_slice(),
            p,
            y.as_slice(),
            lambda.lambda_marker,
            config.pcg_max_iter,
            config.pcg_tol,
        )?
    } else {
        let projector = CovariateProjector::new(c.as_slice(), sample_idx.len(), p)?;
        let fixed_effects = projector.coefficients_for_f64(y.as_slice())?;
        let fitted = projector.fitted_from_coefficients(fixed_effects.as_slice())?;
        let residual = y
            .iter()
            .zip(fitted.iter())
            .map(|(yi, fi)| yi - fi)
            .collect::<Vec<_>>();
        ProjectedRrblupFit {
            fixed_effects,
            additive: vec![0.0_f64; sample_idx.len()],
            fitted,
            residual,
            converged: true,
            iters: 0,
            rel_res: 0.0,
        }
    };

    let mut residualized_y = fit.residual.clone();
    standardize_residualized_y(&mut residualized_y)?;
    let null_residualized_y =
        make_null_residual_permutations(residualized_y.as_slice(), null_repeats, null_seed);
    let null_validation_residualized_y = make_null_residual_permutations(
        residualized_y.as_slice(),
        null_repeats,
        null_validation_seed(null_seed),
    );
    let residual_var = if fit.residual.len() > 1 {
        let mean = fit.residual.iter().sum::<f64>() / fit.residual.len() as f64;
        fit.residual
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / (fit.residual.len() - 1) as f64
    } else {
        f64::NAN
    };
    let additive_var = if fit.additive.len() > 1 {
        let mean = fit.additive.iter().sum::<f64>() / fit.additive.len() as f64;
        fit.additive
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / (fit.additive.len() - 1) as f64
    } else {
        f64::NAN
    };
    let pve = if additive_var.is_finite()
        && residual_var.is_finite()
        && additive_var + residual_var > 0.0
    {
        additive_var / (additive_var + residual_var)
    } else {
        lambda.pve
    };
    Ok(GarfieldResidualResult {
        beta: fit.fixed_effects,
        py: fit.residual.clone(),
        residualized_y,
        resid: fit.residual,
        fitted_y: Some(fit.fitted),
        additive_effect: Some(fit.additive),
        lbd: lambda.lambda_vc,
        lambda_marker: lambda.lambda_marker,
        residualization_method: "rrblup_pcg_additive_projection".to_string(),
        lambda_source: lambda.source,
        no_grm_boundary: false,
        pcg_converged: fit.converged,
        pcg_iters: fit.iters,
        pcg_rel_res: fit.rel_res,
        lambda_converged: lambda.lambda_converged,
        lambda_iters: lambda.lambda_iters,
        lambda_rel_res: lambda.lambda_rel_res,
        ml: f64::NAN,
        reml: f64::NAN,
        sigma_g2: lambda.sigma_g2,
        sigma_e2: lambda.sigma_e2,
        pve,
        n_samples: sample_idx.len(),
        n_fixed_effects: p,
        eigh_backend: lambda.eigh_backend,
        eigh_elapsed: lambda.eigh_elapsed,
        eigenvalues: Vec::new(),
        eff_m: Some(lambda.m_effective),
        null_residualized_y,
        null_validation_residualized_y,
        tail_audit_residualized_y: Vec::new(),
    })
}

struct ProjectedMarkerOperator<'a, D> {
    design: &'a mut D,
    projector: &'a CovariateProjector,
    lambda_marker: f32,
}

impl<D: MarkerDesign> PcgOperator<f32> for ProjectedMarkerOperator<'_, D> {
    fn apply(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        if input.len() != self.design.n_markers() || output.len() != input.len() {
            return Err(format!(
                "projected marker operator length mismatch: input={}, output={}, expected={}",
                input.len(),
                output.len(),
                self.design.n_markers()
            ));
        }
        let mut sample = vec![0.0_f32; self.design.n_samples()];
        self.design.marker_to_samples(input, &mut sample)?;
        self.projector.project_f32_in_place(&mut sample)?;
        self.design.samples_to_markers(&sample, output)?;
        for (dst, &src) in output.iter_mut().zip(input.iter()) {
            *dst += self.lambda_marker * src;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ProjectedRrblupFit {
    fixed_effects: Vec<f64>,
    additive: Vec<f64>,
    fitted: Vec<f64>,
    residual: Vec<f64>,
    converged: bool,
    iters: usize,
    rel_res: f64,
}

fn fit_projected_rrblup<D: MarkerDesign>(
    design: &mut D,
    c: &[f64],
    p: usize,
    y: &[f64],
    lambda_marker: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ProjectedRrblupFit, String> {
    if !lambda_marker.is_finite() || lambda_marker <= 0.0 {
        return Err(format!(
            "marker-space PCG lambda must be finite and > 0, got {lambda_marker}"
        ));
    }
    if max_iter == 0 {
        return Err("marker-space PCG max_iter must be > 0".to_string());
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(format!(
            "marker-space PCG tol must be finite and > 0, got {tol}"
        ));
    }
    if y.len() != design.n_samples() {
        return Err(format!(
            "phenotype length mismatch: got {}, expected {}",
            y.len(),
            design.n_samples()
        ));
    }
    let projector = CovariateProjector::new(c, design.n_samples(), p)?;
    let mut y_projected = y.to_vec();
    projector.project_f64_in_place(&mut y_projected)?;
    let y_projected_f32 = y_projected.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let mut rhs = vec![0.0_f32; design.n_markers()];
    design.samples_to_markers(y_projected_f32.as_slice(), &mut rhs)?;

    let lambda32 = lambda_marker as f32;
    if !lambda32.is_finite() || lambda32 <= 0.0 {
        return Err(format!(
            "marker-space PCG lambda is outside f32 range: {lambda_marker}"
        ));
    }
    let n_scale = (design.n_samples() as f32 + lambda32).recip();
    let preconditioner = pcg_preconditioner_fn(move |rhs_in: &[f32], out: &mut [f32]| {
        if rhs_in.len() != out.len() {
            return Err(format!(
                "PCG scalar preconditioner length mismatch: rhs={}, out={}",
                rhs_in.len(),
                out.len()
            ));
        }
        for (dst, &src) in out.iter_mut().zip(rhs_in.iter()) {
            *dst = src * n_scale;
        }
        Ok(())
    });
    let operator = ProjectedMarkerOperator {
        design: &mut *design,
        projector: &projector,
        lambda_marker: lambda32,
    };
    let pcg_fit = pcg_solve(
        rhs.as_slice(),
        None,
        max_iter,
        tol,
        1e-20_f64,
        operator,
        preconditioner,
        |_iter, _max_iter, _rel_res| Ok(()),
    )?;
    let marker_beta = pcg_fit.x;
    let mut additive_f32 = vec![0.0_f32; design.n_samples()];
    design.marker_to_samples(marker_beta.as_slice(), &mut additive_f32)?;
    let additive = additive_f32.iter().map(|v| *v as f64).collect::<Vec<_>>();
    let y_minus_additive = y
        .iter()
        .zip(additive.iter())
        .map(|(yi, gi)| yi - gi)
        .collect::<Vec<_>>();
    let fixed_effects = projector.coefficients_for_f64(y_minus_additive.as_slice())?;
    let fixed_fitted = projector.fitted_from_coefficients(fixed_effects.as_slice())?;
    let fitted = fixed_fitted
        .iter()
        .zip(additive.iter())
        .map(|(fixed, g)| fixed + g)
        .collect::<Vec<_>>();
    let residual = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect::<Vec<_>>();
    Ok(ProjectedRrblupFit {
        fixed_effects,
        additive,
        fitted,
        residual,
        converged: pcg_fit.converged,
        iters: pcg_fit.iters,
        rel_res: pcg_fit.rel_res,
    })
}

#[cfg(test)]
struct DenseMarkerDesign {
    markers: Vec<f32>,
    m: usize,
    n: usize,
}

#[cfg(test)]
impl DenseMarkerDesign {
    fn new(markers: Vec<f32>, m: usize, n: usize) -> Self {
        assert_eq!(markers.len(), m * n);
        Self { markers, m, n }
    }

    fn apply_marker_to_samples(&self, marker_beta: &[f32]) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.n];
        for j in 0..self.m {
            for i in 0..self.n {
                out[i] += self.markers[j * self.n + i] as f64 * marker_beta[j] as f64;
            }
        }
        out
    }
}

#[cfg(test)]
impl MarkerDesign for DenseMarkerDesign {
    fn n_markers(&self) -> usize {
        self.m
    }

    fn n_samples(&self) -> usize {
        self.n
    }

    fn marker_to_samples(&mut self, marker_beta: &[f32], out: &mut [f32]) -> Result<(), String> {
        if marker_beta.len() != self.m || out.len() != self.n {
            return Err("dense marker_to_samples shape mismatch".to_string());
        }
        out.fill(0.0_f32);
        for j in 0..self.m {
            for i in 0..self.n {
                out[i] += self.markers[j * self.n + i] * marker_beta[j];
            }
        }
        Ok(())
    }

    fn samples_to_markers(&mut self, samples: &[f32], out: &mut [f32]) -> Result<(), String> {
        if samples.len() != self.n || out.len() != self.m {
            return Err("dense samples_to_markers shape mismatch".to_string());
        }
        for j in 0..self.m {
            out[j] = (0..self.n)
                .map(|i| self.markers[j * self.n + i] * samples[i])
                .sum();
        }
        Ok(())
    }
}

#[cfg(test)]
fn fit_projected_rrblup_dense(
    design: &DenseMarkerDesign,
    c: &[f64],
    p: usize,
    y: &[f64],
    lambda_marker: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ProjectedRrblupFit, String> {
    let mut design = DenseMarkerDesign {
        markers: design.markers.clone(),
        m: design.m,
        n: design.n,
    };
    fit_projected_rrblup(&mut design, c, p, y, lambda_marker, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::{
        fit_projected_rrblup_dense, make_null_residual_permutations, oriented_minor_frequency,
        DenseMarkerDesign,
    };

    #[test]
    fn bed_frequency_is_oriented_before_standardization() {
        assert!((oriented_minor_frequency(0.2, false) - 0.2).abs() < 1e-7);
        assert!((oriented_minor_frequency(0.8, true) - 0.2).abs() < 1e-7);
        assert!((oriented_minor_frequency(0.5, false) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn projected_rrblup_removes_covariates_and_returns_additive_residual() {
        let n = 6usize;
        let c = vec![
            1.0, -2.5, 1.0, -1.5, 1.0, -0.5, 1.0, 0.5, 1.0, 1.5, 1.0, 2.5,
        ];
        let w = vec![
            -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.5, 0.0, -0.5, -1.0, -0.25,
        ];
        let marker_beta = [0.75_f32, -0.4_f32];
        let y = vec![-6.55, -2.9, 1.3, 3.05, 6.8, 10.55];
        let design = DenseMarkerDesign::new(w, 2, n);
        let fit = fit_projected_rrblup_dense(&design, &c, 2, &y, 0.25, 200, 1e-10)
            .expect("dense projected rrBLUP fit should succeed");

        assert!(fit.converged, "PCG should converge on the synthetic system");
        assert!(
            fit.rel_res <= 1e-8,
            "unexpected PCG residual: {}",
            fit.rel_res
        );
        assert_eq!(fit.residual.len(), n);
        assert!(fit.residual.iter().all(|v| v.is_finite()));
        assert!(fit.fitted.iter().all(|v| v.is_finite()));

        let direct_additive = design.apply_marker_to_samples(&marker_beta);
        for (got, expected) in fit.additive.iter().zip(direct_additive.iter()) {
            assert!(got.is_finite() && expected.is_finite());
        }

        let residual_norm = fit.residual.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            residual_norm < 1.0,
            "residual norm too large: {residual_norm}"
        );
        let intercept_residual = fit.residual.iter().sum::<f64>();
        let covariate_residual = c
            .chunks_exact(2)
            .zip(fit.residual.iter())
            .map(|(row, residual)| row[1] * residual)
            .sum::<f64>();
        assert!(intercept_residual.abs() < 1e-6);
        assert!(covariate_residual.abs() < 1e-6);
        for (yi, (fi, ri)) in y.iter().zip(fit.fitted.iter().zip(fit.residual.iter())) {
            assert!((yi - fi - ri).abs() < 1e-10);
        }
    }

    #[test]
    fn residual_null_permutations_are_seeded_and_preserve_residual_values() {
        let residual = vec![-1.5, -0.5, 0.25, 1.75, 2.0];
        let first = make_null_residual_permutations(&residual, 4, 1234);
        let second = make_null_residual_permutations(&residual, 4, 1234);
        assert_eq!(first, second);
        assert_eq!(first.len(), 4);
        assert!(first.iter().all(|draw| {
            let mut sorted = draw.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            sorted == residual
        }));
        assert!(first.iter().any(|draw| draw != &residual));
    }
}
