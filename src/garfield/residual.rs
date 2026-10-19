use std::borrow::Cow;
use std::f64::consts::PI;

use nalgebra::DMatrix;
use numpy::ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::brent::brent_minimize_with_diagnostics;
use crate::eigh::symmetric_eigh_f64_row_major;
use crate::linalg::{cholesky_inplace, cholesky_logdet};
use crate::stats_common::get_cached_pool;
use std::time::Instant;

use super::pcg_residual::{garfield_residualize_pcg_from_bed, PcgResidualConfig};

#[derive(Debug)]
struct GarfieldResidualFit {
    beta: Vec<f64>,
    resid_rot: Vec<f64>,
    py_rot: Vec<f64>,
    lbd: f64,
    ml: f64,
    reml: f64,
    sigma_g2: f64,
    sigma_e2: f64,
    fixed_rank: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct GarfieldResidualResult {
    pub beta: Vec<f64>,
    pub py: Vec<f64>,
    pub residualized_y: Vec<f64>,
    pub resid: Vec<f64>,
    pub fitted_y: Option<Vec<f64>>,
    pub additive_effect: Option<Vec<f64>>,
    pub lbd: f64,
    pub lambda_marker: f64,
    pub residualization_method: String,
    pub lambda_source: String,
    pub no_grm_boundary: bool,
    pub pcg_converged: bool,
    pub pcg_iters: usize,
    pub pcg_rel_res: f64,
    pub lambda_converged: bool,
    pub lambda_iters: usize,
    pub lambda_rel_res: f64,
    pub ml: f64,
    pub reml: f64,
    pub sigma_g2: f64,
    pub sigma_e2: f64,
    pub pve: f64,
    pub n_samples: usize,
    pub n_fixed_effects: usize,
    pub eigh_backend: String,
    pub eigh_elapsed: f64,
    pub eigenvalues: Vec<f64>,
    pub eff_m: Option<usize>,
    pub null_residualized_y: Vec<Vec<f64>>,
    pub null_validation_residualized_y: Vec<Vec<f64>>,
    /// Independent diagnostic-only null draws for the opt-in Max6 tail audit.
    /// These never enter production threshold calibration or validation.
    pub tail_audit_residualized_y: Vec<Vec<f64>>,
}

/// Derive a deterministic, domain-separated seed for null validation draws.
/// Calibration and validation remain reproducible while using disjoint RNG
/// streams for every root seed.
pub(crate) fn null_validation_seed(calibration_seed: u64) -> u64 {
    let mut value = calibration_seed ^ 0x5641_4C49_4441_5445;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let validation_seed = value ^ (value >> 31);
    if validation_seed == calibration_seed {
        calibration_seed ^ 0x5641_4C49_4441_5445
    } else {
        validation_seed
    }
}

/// Deterministic stream isolated from both the production calibration and
/// validation nulls. Tail-audit draws are diagnostics only.
pub(crate) fn null_tail_audit_seed(root_seed: u64) -> u64 {
    let mut value = root_seed ^ 0x5441_494C_4155_4449;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let audit_seed = value ^ (value >> 31);
    if audit_seed == root_seed || audit_seed == null_validation_seed(root_seed) {
        root_seed ^ 0x5441_494C_4155_4449
    } else {
        audit_seed
    }
}

#[inline]
fn cholesky_solve(a: &[f64], dim: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; dim];
    for i in 0..dim {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * dim + k] * y[k];
        }
        y[i] = sum / a[i * dim + i];
    }

    let mut x = vec![0.0_f64; dim];
    for ii in 0..dim {
        let i = dim - 1 - ii;
        let mut sum = y[i];
        for k in (i + 1)..dim {
            sum -= a[k * dim + i] * x[k];
        }
        x[i] = sum / a[i * dim + i];
    }
    x
}

#[derive(Debug)]
struct FixedEffectMetricSolution {
    inverse: Vec<f64>,
    beta: Vec<f64>,
    log_pdet: f64,
    rank: usize,
}

const NO_GRM_QR_RELATIVE_RANK_TOL: f64 = 1.0e-12;

/// Low-rank representation of the ordinary fixed-effect residual projector.
///
/// For a design matrix `C` with numerical rank `r`, the thin QR/SVD-equivalent
/// basis stores only an orthonormal `Q` with shape `n x r` and represents
///
/// ```text
/// M_C = I - Q Q'
/// ```
///
/// This is the exact no-GRM limit of the LMM projection, but avoids materializing
/// the `n x n` matrix.  The QR with column pivoting supplies a stable rank
/// decision for the usual small covariate design; the retained Q columns are an
/// orthonormal basis even when redundant covariates are present.
#[derive(Clone, Debug)]
pub(crate) struct NoGrmCovariateProjector {
    n_samples: usize,
    rank: usize,
    q: Vec<f64>,
}

impl NoGrmCovariateProjector {
    pub(crate) fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn q(&self) -> &[f64] {
        &self.q
    }

    pub(crate) fn coordinates(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        if values.len() != self.n_samples {
            return Err(format!(
                "no-GRM projector vector length mismatch: got {}, expected {}",
                values.len(),
                self.n_samples
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("no-GRM projector input contains a non-finite value".to_string());
        }
        let mut coordinates = vec![0.0_f64; self.rank];
        for row in 0..self.n_samples {
            let value = values[row];
            for column in 0..self.rank {
                coordinates[column] += self.q[row * self.rank + column] * value;
            }
        }
        Ok(coordinates)
    }

    pub(crate) fn project(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        let coordinates = self.coordinates(values)?;
        let mut projected = values.to_vec();
        for row in 0..self.n_samples {
            let fitted = (0..self.rank)
                .map(|column| self.q[row * self.rank + column] * coordinates[column])
                .sum::<f64>();
            projected[row] -= fitted;
        }
        if projected.iter().any(|value| !value.is_finite()) {
            return Err("no-GRM projector produced a non-finite vector".to_string());
        }
        Ok(projected)
    }

    #[cfg(test)]
    pub(crate) fn projected_dot(&self, left: &[f64], right: &[f64]) -> Result<f64, String> {
        let left_coordinates = self.coordinates(left)?;
        let right_coordinates = self.coordinates(right)?;
        let raw = left
            .iter()
            .zip(right.iter())
            .map(|(&left, &right)| left * right)
            .sum::<f64>();
        Ok(self.projected_dot_from_coordinates(raw, &left_coordinates, &right_coordinates))
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn projected_dot_from_coordinates(
        &self,
        raw_dot: f64,
        left_coordinates: &[f64],
        right_coordinates: &[f64],
    ) -> f64 {
        debug_assert_eq!(left_coordinates.len(), self.rank);
        debug_assert_eq!(right_coordinates.len(), self.rank);
        raw_dot
            - left_coordinates
                .iter()
                .zip(right_coordinates.iter())
                .map(|(&left, &right)| left * right)
                .sum::<f64>()
    }

    /// Diagonal of the represented residual projector `I - Q Q'`.
    pub(crate) fn projected_diagonal(&self) -> Vec<f64> {
        (0..self.n_samples)
            .map(|row| {
                (1.0 - (0..self.rank)
                    .map(|column| {
                        let value = self.q[row * self.rank + column];
                        value * value
                    })
                    .sum::<f64>())
                .max(0.0)
            })
            .collect()
    }
}

/// Fix the design rank once, before the covariance metric changes during REML.
/// Column normalization makes the QR decision independent of covariate units.
/// The single-column fast path preserves the common intercept-only arithmetic.
struct RankAwareFixedDesign {
    x: Vec<f64>,
    rank: usize,
    original_p: usize,
    selected: Vec<usize>,
    upper: Vec<f64>,
    norms: Vec<f64>,
}

impl RankAwareFixedDesign {
    fn expand_beta(&self, beta: &[f64]) -> Vec<f64> {
        // Return a deterministic basic solution in the original column order.
        // Redundant-column coefficients are not individually identifiable;
        // their zero entries do not imply absence of a fixed-effect signal.
        let mut coefficients = beta.to_vec();
        for row in (0..self.rank).rev() {
            for col in row + 1..self.rank {
                coefficients[row] -= self.upper[row * self.rank + col] * coefficients[col];
            }
            coefficients[row] /= self.upper[row * self.rank + row];
        }
        let mut out = vec![0.0; self.original_p];
        for (position, &column) in self.selected.iter().enumerate() {
            out[column] = coefficients[position] / self.norms[column];
        }
        out
    }
}

fn rank_aware_fixed_design(x: &[f64], n: usize, p: usize) -> Result<RankAwareFixedDesign, String> {
    if n == 0
        || p == 0
        || x.len() != n.saturating_mul(p)
        || x.iter().any(|value| !value.is_finite())
    {
        return Err("fixed-effect design must have a finite nonempty n x p payload".to_string());
    }
    if p == 1 {
        if n <= 1 || !x.iter().any(|&value| value != 0.0) {
            return Err("fixed-effect design requires 0 < rank < n".to_string());
        }
        return Ok(RankAwareFixedDesign {
            x: x.to_vec(),
            rank: 1,
            original_p: 1,
            selected: vec![0],
            upper: vec![1.0],
            norms: vec![1.0],
        });
    }
    let norms = (0..p)
        .map(|column| (0..n).fold(0.0_f64, |norm, row| norm.hypot(x[row * p + column])))
        .collect::<Vec<_>>();
    let normalized = DMatrix::from_fn(n, p, |row, column| {
        if norms[column] > 0.0 {
            x[row * p + column] / norms[column]
        } else {
            0.0
        }
    });
    let qr = normalized.col_piv_qr();
    let r = qr.r();
    let max_diagonal = (0..n.min(p))
        .map(|i| r[(i, i)].abs())
        .fold(0.0_f64, f64::max);
    let cutoff = NO_GRM_QR_RELATIVE_RANK_TOL * max_diagonal;
    let rank = (0..n.min(p))
        .take_while(|&i| r[(i, i)].abs() > cutoff)
        .count();
    if rank == 0 || rank >= n {
        return Err(format!(
            "fixed-effect design requires 0 < rank < n; got rank={rank}, n={n}"
        ));
    }
    let mut order = DMatrix::from_fn(1, p, |_, col| col);
    qr.p().permute_columns(&mut order);
    let selected = (0..rank).map(|i| order[(0, i)]).collect();
    let q = qr.q();
    let basis = (0..n)
        .flat_map(|row| (0..rank).map(move |col| (row, col)))
        .map(|(row, col)| q[(row, col)])
        .collect();
    let upper = (0..rank)
        .flat_map(|row| (0..rank).map(move |col| (row, col)))
        .map(|(row, col)| r[(row, col)])
        .collect();
    Ok(RankAwareFixedDesign {
        x: basis,
        rank,
        original_p: p,
        selected,
        upper,
        norms,
    })
}

/// Build a low-rank no-GRM fixed-effect projector using the same design rank.
pub(crate) fn build_no_grm_covariate_projector(
    x: &[f64],
    n: usize,
    p: usize,
) -> Result<NoGrmCovariateProjector, String> {
    if n == 0 || p == 0 || x.len() != n.saturating_mul(p) {
        return Err("no-GRM covariate projector design shape mismatch".to_string());
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err("no-GRM covariate projector design contains a non-finite value".to_string());
    }
    let design = rank_aware_fixed_design(x, n, p)?;
    let rank = design.rank;
    let mut q = design.x;
    if p == 1 {
        let norm = q.iter().fold(0.0_f64, |norm, &v| norm.hypot(v));
        for value in &mut q {
            *value /= norm;
        }
    }
    if q.iter().any(|value| !value.is_finite()) {
        return Err("no-GRM QR produced a non-finite Q basis".to_string());
    }
    Ok(NoGrmCovariateProjector {
        n_samples: n,
        rank,
        q,
    })
}

/// Solve the GLS fixed-effect metric without adding an artificial ridge.
///
/// The exact covariance projection must use the inverse on the span of the
/// fixed-effect design. A Cholesky solve is used for the common full-rank
/// case; a symmetric eigendecomposition supplies the Moore--Penrose inverse
/// for redundant/zero covariates. This keeps `P_C X = 0` while preserving a
/// usable null fit for rank-deficient design matrices.
fn solve_fixed_effect_metric(
    metric: &[f64],
    rhs: &[f64],
    dim: usize,
) -> Option<FixedEffectMetricSolution> {
    if dim == 0 || metric.len() != dim.saturating_mul(dim) || rhs.len() != dim {
        return None;
    }

    let mut cholesky_metric = metric.to_vec();
    let rank_tolerance = (0..dim)
        .map(|i| metric[i * dim + i].abs())
        .fold(0.0_f64, f64::max)
        * 1.0e-12
        * dim as f64;
    if cholesky_inplace(&mut cholesky_metric, dim).is_some()
        && (0..dim).all(|i| cholesky_metric[i * dim + i].powi(2) > rank_tolerance)
    {
        let log_pdet = cholesky_logdet(&cholesky_metric, dim);
        if !log_pdet.is_finite() {
            return None;
        }
        let beta = cholesky_solve(&cholesky_metric, dim, rhs);
        let mut inverse = vec![0.0_f64; dim * dim];
        for column in 0..dim {
            let mut unit = vec![0.0_f64; dim];
            unit[column] = 1.0;
            let solution = cholesky_solve(&cholesky_metric, dim, &unit);
            for row in 0..dim {
                inverse[row * dim + column] = solution[row];
            }
        }
        return Some(FixedEffectMetricSolution {
            inverse,
            beta,
            log_pdet,
            rank: dim,
        });
    }

    let (eigenvalues, eigenvectors, _backend) = symmetric_eigh_f64_row_major(metric, dim).ok()?;
    let max_eigenvalue = eigenvalues
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if !(max_eigenvalue.is_finite() && max_eigenvalue > 0.0) {
        return None;
    }
    let tolerance = max_eigenvalue * 1.0e-12 * dim.max(1) as f64;
    let mut inverse_eigenvalues = vec![0.0_f64; dim];
    let mut log_pdet = 0.0_f64;
    let mut rank = 0usize;
    for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
        if !eigenvalue.is_finite() || eigenvalue < -tolerance {
            return None;
        }
        if eigenvalue > tolerance {
            inverse_eigenvalues[index] = 1.0 / eigenvalue;
            log_pdet += eigenvalue.ln();
            rank += 1;
        }
    }
    if rank == 0 || !log_pdet.is_finite() {
        return None;
    }

    let mut inverse = vec![0.0_f64; dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            let mut value = 0.0_f64;
            for component in 0..dim {
                value += eigenvectors[row * dim + component]
                    * inverse_eigenvalues[component]
                    * eigenvectors[col * dim + component];
            }
            inverse[row * dim + col] = value;
        }
    }
    let beta = (0..dim)
        .map(|row| {
            (0..dim)
                .map(|col| inverse[row * dim + col] * rhs[col])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    Some(FixedEffectMetricSolution {
        inverse,
        beta,
        log_pdet,
        rank,
    })
}

pub(crate) fn build_design_matrix(
    x_cov: Option<&[f64]>,
    n: usize,
    q_cov: usize,
    add_intercept: bool,
) -> Result<(Vec<f64>, usize), String> {
    let p = q_cov + if add_intercept { 1 } else { 0 };
    if p == 0 {
        return Err(
            "at least one fixed-effect column is required; set add_intercept=true or provide x_cov."
                .to_string(),
        );
    }
    let mut x = vec![0.0_f64; n * p];
    for i in 0..n {
        let base = i * p;
        if add_intercept {
            x[base] = 1.0;
            if let Some(cov) = x_cov {
                let cov_base = i * q_cov;
                for j in 0..q_cov {
                    x[base + 1 + j] = cov[cov_base + j];
                }
            }
        } else if let Some(cov) = x_cov {
            let cov_base = i * q_cov;
            for j in 0..q_cov {
                x[base + j] = cov[cov_base + j];
            }
        }
    }
    Ok((x, p))
}

/// Build the ordinary fixed-effect residual projector
///
/// ```text
/// M_C = I - X (X'X)^+ X'
/// ```
///
/// This is the `lambda -> +infinity` limit of the LMM covariance projector
/// when `lambda = sigma_e^2 / sigma_g^2`.  Keeping it as an explicit matrix
/// avoids representing the no-polygenic boundary by an arbitrary finite
/// lambda and, importantly, avoids confusing that boundary with
/// `lambda = 0` (the pure-GRM limit).
fn build_no_grm_projection(x: &[f64], n: usize, p: usize) -> Result<Vec<f64>, String> {
    if x.len() != n.saturating_mul(p) || n == 0 || p == 0 {
        return Err("no-GRM projection design shape mismatch".to_string());
    }
    let mut xtx = vec![0.0_f64; p * p];
    for row in 0..n {
        let base = row * p;
        for left in 0..p {
            let x_left = x[base + left];
            for right in 0..=left {
                xtx[left * p + right] += x_left * x[base + right];
            }
        }
    }
    for left in 0..p {
        for right in 0..left {
            xtx[right * p + left] = xtx[left * p + right];
        }
    }
    let inverse = solve_fixed_effect_metric(&xtx, &vec![0.0; p], p)
        .ok_or_else(|| {
            "fixed-effect metric is singular/indefinite for no-GRM projection".to_string()
        })?
        .inverse;
    let mut out = vec![0.0_f64; n * n];
    for row in 0..n {
        for col in 0..=row {
            let mut value = if row == col { 1.0 } else { 0.0 };
            for left in 0..p {
                let x_left = x[row * p + left];
                for right in 0..p {
                    value -= x_left * inverse[left * p + right] * x[col * p + right];
                }
            }
            if !value.is_finite() {
                return Err("no-GRM projection produced a non-finite value".to_string());
            }
            out[row * n + col] = value;
            out[col * n + row] = value;
        }
    }
    Ok(out)
}

fn rotate_design_and_response(
    u: &[f64],
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    threads: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let dim = p + 1;
    let mut utxy = vec![0.0_f64; n * dim];
    let mut run = || {
        utxy.par_chunks_mut(dim).enumerate().for_each(|(k, row)| {
            for c in 0..p {
                let mut acc = 0.0_f64;
                for i in 0..n {
                    acc += u[i * n + k] * x[i * p + c];
                }
                row[c] = acc;
            }
            let mut accy = 0.0_f64;
            for i in 0..n {
                accy += u[i * n + k] * y[i];
            }
            row[p] = accy;
        });
    };
    if let Some(p0) = &pool {
        p0.install(run);
    } else {
        run();
    }

    let mut x_rot = vec![0.0_f64; n * p];
    let mut y_rot = vec![0.0_f64; n];
    for k in 0..n {
        let src = &utxy[k * dim..(k + 1) * dim];
        x_rot[k * p..(k + 1) * p].copy_from_slice(&src[..p]);
        y_rot[k] = src[p];
    }
    Ok((x_rot, y_rot))
}

fn project_back_from_eigenvectors(
    u: &[f64],
    vec_rot: &[f64],
    n: usize,
    threads: usize,
) -> Result<Vec<f64>, String> {
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let mut out = vec![0.0_f64; n];
    let mut run = || {
        out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += u[i * n + k] * vec_rot[k];
            }
            *out_i = acc;
        });
    };
    if let Some(p0) = &pool {
        p0.install(run);
    } else {
        run();
    }
    Ok(out)
}

pub(crate) fn standardize_residualized_y(values: &mut [f64]) -> Result<(), String> {
    let n = values.len();
    if n < 2 {
        return Err(
            "garfield residualization requires at least 2 samples to standardize Py.".to_string(),
        );
    }
    let mean = values.iter().copied().sum::<f64>() / (n as f64);
    let mut ss = 0.0_f64;
    for value in values.iter() {
        let centered = *value - mean;
        ss += centered * centered;
    }
    let variance = ss / ((n - 1) as f64);
    if !(variance.is_finite() && variance > 0.0) {
        return Err(
            "garfield residualization produced zero-variance Py; cannot standardize.".to_string(),
        );
    }
    let sample_std = variance.sqrt();
    for value in values.iter_mut() {
        *value = (*value - mean) / sample_std;
    }
    Ok(())
}

fn exact_lmm_null_fit_from_rotated(
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    p: usize,
    log10_lbd: f64,
) -> Option<GarfieldResidualFit> {
    if log10_lbd.is_infinite() && log10_lbd.is_sign_positive() {
        return exact_no_grm_fit_from_rotated(x_rot, y_rot, n, p);
    }
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 || n <= p {
        return None;
    }

    let mut v = vec![0.0_f64; n];
    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if !vv.is_finite() || vv <= 0.0 {
            return None;
        }
        v[i] = vv;
        vinv[i] = 1.0 / vv;
    }

    let mut xtv_inv_x = vec![0.0_f64; p * p];
    let mut xtv_inv_y = vec![0.0_f64; p];
    for i in 0..n {
        let vi = vinv[i];
        let yi = y_rot[i];
        let base = i * p;
        for r in 0..p {
            let xir = x_rot[base + r];
            xtv_inv_y[r] += vi * xir * yi;
            for c in 0..=r {
                xtv_inv_x[r * p + c] += vi * xir * x_rot[base + c];
            }
        }
    }

    for r in 0..p {
        for c in 0..r {
            xtv_inv_x[c * p + r] = xtv_inv_x[r * p + c];
        }
    }
    let fixed_metric = solve_fixed_effect_metric(&xtv_inv_x, &xtv_inv_y, p)?;
    let log_det_xtv_inv_x = fixed_metric.log_pdet;
    let beta = fixed_metric.beta;

    let mut resid_rot = vec![0.0_f64; n];
    let mut py_rot = vec![0.0_f64; n];
    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        let base = i * p;
        let mut xb = 0.0_f64;
        for r in 0..p {
            xb += x_rot[base + r] * beta[r];
        }
        let ri = y_rot[i] - xb;
        resid_rot[i] = ri;
        py_rot[i] = vinv[i] * ri;
        rtv_invr += vinv[i] * ri * ri;
    }
    if !rtv_invr.is_finite() || rtv_invr <= 0.0 {
        return None;
    }

    let n_f = n as f64;
    let fixed_rank = fixed_metric.rank;
    if n <= fixed_rank {
        return None;
    }
    let p_f = fixed_rank as f64;
    let log_det_v: f64 = v.iter().map(|vv| vv.ln()).sum();
    let reml_total = (n_f - p_f) * rtv_invr.ln() + log_det_v + log_det_xtv_inv_x;
    let reml_const = (n_f - p_f) * ((n_f - p_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let reml = reml_const - 0.5 * reml_total;
    let ml_total = n_f * rtv_invr.ln() + log_det_v;
    let ml_const = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let ml = ml_const - 0.5 * ml_total;
    if !reml.is_finite() || !ml.is_finite() {
        return None;
    }

    let sigma_g2 = rtv_invr / (n_f - p_f);
    let sigma_e2 = lbd * sigma_g2;
    if !(sigma_g2.is_finite() && sigma_g2 > 0.0 && sigma_e2.is_finite() && sigma_e2 > 0.0) {
        return None;
    }

    Some(GarfieldResidualFit {
        beta,
        resid_rot,
        py_rot,
        lbd,
        ml,
        reml,
        sigma_g2,
        sigma_e2,
        fixed_rank,
    })
}

/// Fit the `sigma_g^2 = 0` boundary model in the GRM eigenbasis.
///
/// The returned `py_rot` is the ordinary fixed-effect residual precision
/// vector.  Its overall scale is removed by the downstream phenotype
/// standardization, while the explicit `sigma_e2` is used by the exact
/// GRM-Max6 score scale.
fn exact_no_grm_fit_from_rotated(
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    p: usize,
) -> Option<GarfieldResidualFit> {
    if n == 0 || p == 0 || x_rot.len() != n.saturating_mul(p) || y_rot.len() != n {
        return None;
    }
    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];
    for row in 0..n {
        let base = row * p;
        for left in 0..p {
            let x_left = x_rot[base + left];
            xty[left] += x_left * y_rot[row];
            for right in 0..=left {
                xtx[left * p + right] += x_left * x_rot[base + right];
            }
        }
    }
    for left in 0..p {
        for right in 0..left {
            xtx[right * p + left] = xtx[left * p + right];
        }
    }
    let fixed_metric = solve_fixed_effect_metric(&xtx, &xty, p)?;
    let mut resid_rot = vec![0.0_f64; n];
    let mut rss = 0.0_f64;
    for row in 0..n {
        let base = row * p;
        let fitted = (0..p)
            .map(|column| x_rot[base + column] * fixed_metric.beta[column])
            .sum::<f64>();
        let residual = y_rot[row] - fitted;
        resid_rot[row] = residual;
        rss += residual * residual;
    }
    let dof = n.checked_sub(fixed_metric.rank)?;
    if dof == 0 || !rss.is_finite() || rss <= 0.0 {
        return None;
    }
    let sigma_e2 = rss / dof as f64;
    if !sigma_e2.is_finite() || sigma_e2 <= 0.0 {
        return None;
    }
    let py_rot = resid_rot
        .iter()
        .map(|&value| value / sigma_e2)
        .collect::<Vec<_>>();
    let n_f = n as f64;
    let rank_f = fixed_metric.rank as f64;
    let reml_total = (n_f - rank_f) * rss.ln() + fixed_metric.log_pdet;
    let reml_const = (n_f - rank_f) * ((n_f - rank_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let reml = reml_const - 0.5 * reml_total;
    let ml_total = n_f * rss.ln();
    let ml_const = n_f * (n_f.ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let ml = ml_const - 0.5 * ml_total;
    if !reml.is_finite() || !ml.is_finite() {
        return None;
    }
    Some(GarfieldResidualFit {
        beta: fixed_metric.beta,
        resid_rot,
        py_rot,
        lbd: f64::INFINITY,
        ml,
        reml,
        sigma_g2: 0.0,
        sigma_e2,
        fixed_rank: fixed_metric.rank,
    })
}

/// Generate conditional parametric null phenotypes for a fitted LMM.
///
/// The draws are made in the GRM eigenbasis, where the fitted covariance is
/// diagonal. Fixed effects are projected out with the same weighted design
/// used by the fitted null model, then the resulting Py vectors are mapped
/// back to sample space and standardized exactly like the observed response.
/// The eigenvectors are deliberately not retained in the residual result;
/// this keeps the O(n²) eigensystem workspace transient instead of doubling
/// the resident memory of a GARFIELD run.
fn generate_lmm_null_residualized(
    s: &[f64],
    u: &[f64],
    x_rot: &[f64],
    n: usize,
    p: usize,
    lbd: f64,
    repeats: usize,
    seed: u64,
    threads: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if repeats == 0 {
        return Ok(Vec::new());
    }
    if s.len() != n || u.len() != n.saturating_mul(n) || x_rot.len() != n.saturating_mul(p) {
        return Err("conditional LMM null workspace shape mismatch".to_string());
    }
    let no_grm_boundary = lbd.is_infinite() && lbd.is_sign_positive();
    if !no_grm_boundary && !(lbd.is_finite() && lbd > 0.0) {
        return Err(format!(
            "conditional LMM null requires positive lbd, got {lbd}"
        ));
    }
    let mut sqrt_v = Vec::with_capacity(n);
    for (i, &sv) in s.iter().enumerate() {
        let vv = if no_grm_boundary {
            1.0
        } else {
            sv.max(0.0) + lbd
        };
        if !(vv.is_finite() && vv > 0.0) {
            return Err(format!(
                "invalid fitted null variance at eigenvalue {i}: {vv}"
            ));
        }
        sqrt_v.push(vv.sqrt());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let y_rot = sqrt_v
            .iter()
            .map(|&scale| scale * rng.sample::<f64, _>(StandardNormal))
            .collect::<Vec<_>>();
        let fit = if no_grm_boundary {
            exact_no_grm_fit_from_rotated(x_rot, &y_rot, n, p)
        } else {
            exact_lmm_null_fit_from_rotated(s, x_rot, &y_rot, n, p, lbd.log10())
        }
        .ok_or_else(|| "failed to evaluate conditional LMM null draw".to_string())?;
        let mut py = project_back_from_eigenvectors(u, &fit.py_rot, n, threads)?;
        standardize_residualized_y(&mut py)?;
        out.push(py);
    }
    Ok(out)
}

fn exact_reml_loglike_from_rotated(
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    p: usize,
    log10_lbd: f64,
) -> f64 {
    exact_lmm_null_fit_from_rotated(s, x_rot, y_rot, n, p, log10_lbd)
        .map(|fit| fit.reml)
        .unwrap_or(-1e100_f64)
}

#[derive(Clone, Debug)]
pub(crate) struct GarfieldLmmLambdaEstimate {
    pub lambda_vc: f64,
    pub sigma_g2: f64,
    pub sigma_e2: f64,
    pub pve: f64,
    pub converged: bool,
    pub iters: usize,
    pub rel_res: f64,
    pub eigh_backend: String,
    pub eigh_elapsed: f64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_lmm_lambda_from_grm_rust(
    grm_vec: &[f64],
    n: usize,
    y_vec: &[f64],
    x_cov_vec: Option<&[f64]>,
    q_cov: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
) -> Result<GarfieldLmmLambdaEstimate, String> {
    if n > exact_n_max {
        return Err(format!(
            "small-sample REML lambda selection requires n <= {exact_n_max}; got n={n}"
        ));
    }
    if y_vec.len() != n {
        return Err(format!(
            "y length mismatch for REML lambda selection: got {}, expected {n}",
            y_vec.len()
        ));
    }
    if y_vec.iter().any(|v| !v.is_finite()) {
        return Err("y contains non-finite values for REML lambda selection".to_string());
    }
    if let Some(cov) = x_cov_vec {
        if cov.len() != n.saturating_mul(q_cov) {
            return Err(format!(
                "x_cov length mismatch for REML lambda selection: got {}, expected {}",
                cov.len(),
                n.saturating_mul(q_cov)
            ));
        }
        if cov.iter().any(|v| !v.is_finite()) {
            return Err("x_cov contains non-finite values for REML lambda selection".to_string());
        }
    }
    validate_grm(grm_vec, n)?;
    let (x_full, p) = build_design_matrix(x_cov_vec, n, q_cov, add_intercept)?;
    if n <= p {
        return Err(format!(
            "REML lambda selection requires n > rank(C): n={n}, p={p}"
        ));
    }
    let driver_t0 = Instant::now();
    let (threads_before, _threads_in_stage, eigh_run, _threads_after) =
        crate::blas::with_eigh_thread_stage(threads, || symmetric_eigh_f64_row_major(grm_vec, n));
    let eigh_elapsed = driver_t0.elapsed().as_secs_f64();
    let (evals, evecs, evd_backend) = eigh_run?;
    if require_lapack && evd_backend == "nalgebra" {
        return Err(
            "REML lambda selection expected LAPACK backend but fell back to nalgebra".to_string(),
        );
    }
    let _ = threads_before;
    let s_vec = evals
        .into_iter()
        .enumerate()
        .map(|(i, value)| {
            if !value.is_finite() {
                Err(format!(
                    "non-finite eigenvalue at index {i} for REML lambda selection"
                ))
            } else {
                Ok(value.max(0.0))
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (x_rot, y_rot) = rotate_design_and_response(&evecs, &x_full, y_vec, n, p, threads)?;
    let brent = brent_minimize_with_diagnostics(
        |log10_lbd| -exact_reml_loglike_from_rotated(&s_vec, &x_rot, &y_rot, n, p, log10_lbd),
        low,
        high,
        tol,
        max_iter,
    );
    let fit = exact_lmm_null_fit_from_rotated(&s_vec, &x_rot, &y_rot, n, p, brent.x)
        .ok_or_else(|| "failed to evaluate small-sample REML lambda estimate".to_string())?;
    let mean_s = s_vec.iter().copied().sum::<f64>() / n as f64;
    let var_g = fit.sigma_g2 * mean_s.max(0.0);
    let denom = var_g + fit.sigma_e2;
    let pve = if denom.is_finite() && denom > 0.0 {
        var_g / denom
    } else {
        f64::NAN
    };
    Ok(GarfieldLmmLambdaEstimate {
        lambda_vc: fit.lbd,
        sigma_g2: fit.sigma_g2,
        sigma_e2: fit.sigma_e2,
        pve,
        converged: brent.converged,
        iters: brent.iterations,
        rel_res: brent.relative_bracket_width,
        eigh_backend: evd_backend.to_string(),
        eigh_elapsed,
    })
}

fn validate_grm(grm: &[f64], n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("GRM must not be empty.".to_string());
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn garfield_residualize_exact_from_grm_rust(
    grm_vec: Vec<f64>,
    n: usize,
    y_vec: Vec<f64>,
    x_cov_vec: Option<Vec<f64>>,
    q_cov: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    eff_m: Option<usize>,
    null_repeats: usize,
    null_seed: u64,
    tail_audit_repeats: usize,
) -> Result<GarfieldResidualResult, String> {
    garfield_residualize_exact_with_eigenvectors_rust(
        grm_vec,
        n,
        y_vec,
        x_cov_vec,
        q_cov,
        threads,
        low,
        high,
        max_iter,
        tol,
        add_intercept,
        exact_n_max,
        require_lapack,
        eff_m,
        null_repeats,
        null_seed,
        tail_audit_repeats,
        None,
    )
}

/// Retain the fitted eigensystem only for immediate exact projection assembly.
/// It is not attached to the residual result or held across observed windows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn garfield_residualize_exact_with_eigenvectors_rust(
    grm_vec: Vec<f64>,
    n: usize,
    y_vec: Vec<f64>,
    x_cov_vec: Option<Vec<f64>>,
    q_cov: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    eff_m: Option<usize>,
    null_repeats: usize,
    null_seed: u64,
    tail_audit_repeats: usize,
    eigenvectors_out: Option<&mut Vec<f64>>,
) -> Result<GarfieldResidualResult, String> {
    if n > exact_n_max {
        return Err(format!(
            "garfield exact residualization currently supports n <= {exact_n_max}; got n={n}. Large-sample HE/PCG residualization is not implemented yet."
        ));
    }
    if y_vec.len() != n {
        return Err(format!(
            "y length mismatch: got {}, expected {n}",
            y_vec.len()
        ));
    }
    if y_vec.iter().any(|v| !v.is_finite()) {
        return Err("y contains non-finite values.".to_string());
    }
    if let Some(cov) = x_cov_vec.as_ref() {
        if cov.len() != n * q_cov {
            return Err(format!(
                "x_cov payload length mismatch: got {}, expected {}",
                cov.len(),
                n * q_cov
            ));
        }
        if cov.iter().any(|v| !v.is_finite()) {
            return Err("x_cov contains non-finite values.".to_string());
        }
    }
    validate_grm(&grm_vec, n)?;

    let (original_design, original_p) =
        build_design_matrix(x_cov_vec.as_deref(), n, q_cov, add_intercept)?;
    let fixed_design = rank_aware_fixed_design(&original_design, n, original_p)?;
    let (x_full, p) = (&fixed_design.x, fixed_design.rank);

    let driver_t0 = Instant::now();
    let (threads_before, _threads_in_stage, eigh_run, _threads_after) =
        crate::blas::with_eigh_thread_stage(threads, || symmetric_eigh_f64_row_major(&grm_vec, n));
    let eigh_elapsed = driver_t0.elapsed().as_secs_f64();
    let (evals, evecs, evd_backend) = eigh_run?;
    if require_lapack && evd_backend == "nalgebra" {
        return Err(
            "garfield residualization expected LAPACK backend but fell back to nalgebra"
                .to_string(),
        );
    }
    let _ = threads_before;

    if evals.len() != n || evecs.len() != n * n {
        return Err(
            "eigen-decomposition output shape mismatch during residualization.".to_string(),
        );
    }
    let mut s_vec = Vec::with_capacity(n);
    for (i, &sv) in evals.iter().enumerate() {
        if !sv.is_finite() {
            return Err(format!(
                "non-finite eigenvalue at index {i} during residualization."
            ));
        }
        s_vec.push(sv.max(0.0));
    }
    let u_vec = evecs;

    let (x_rot, y_rot) = rotate_design_and_response(&u_vec, &x_full, &y_vec, n, p, threads)?;
    let brent = brent_minimize_with_diagnostics(
        |log10_lbd| -exact_reml_loglike_from_rotated(&s_vec, &x_rot, &y_rot, n, p, log10_lbd),
        low,
        high,
        tol,
        max_iter,
    );
    let finite_fit = exact_lmm_null_fit_from_rotated(&s_vec, &x_rot, &y_rot, n, p, brent.x)
        .ok_or_else(|| {
            "garfield exact residualization failed to evaluate the null model at the REML optimum."
                .to_string()
        })?;
    // REML has a valid sigma_g^2=0 boundary at lambda=+infinity.  Compare
    // that restricted model explicitly instead of interpreting the arbitrary
    // finite search upper bound as a numeric lambda value.
    let boundary_fit = exact_no_grm_fit_from_rotated(&x_rot, &y_rot, n, p).ok_or_else(|| {
        "garfield exact residualization failed to evaluate the no-GRM boundary model.".to_string()
    })?;
    let reml_boundary_tolerance =
        1.0e-10 * (1.0 + finite_fit.reml.abs().max(boundary_fit.reml.abs()));
    let no_grm_boundary = boundary_fit.reml + reml_boundary_tolerance >= finite_fit.reml;
    let fit = if no_grm_boundary {
        boundary_fit
    } else {
        finite_fit
    };
    let resid = project_back_from_eigenvectors(&u_vec, &fit.resid_rot, n, threads)?;
    let py_vec = project_back_from_eigenvectors(&u_vec, &fit.py_rot, n, threads)?;
    let mut residualized_y = py_vec.clone();
    standardize_residualized_y(&mut residualized_y)?;
    let null_residualized_y = generate_lmm_null_residualized(
        &s_vec,
        &u_vec,
        &x_rot,
        n,
        p,
        fit.lbd,
        null_repeats,
        null_seed,
        threads,
    )?;
    let null_validation_residualized_y = generate_lmm_null_residualized(
        &s_vec,
        &u_vec,
        &x_rot,
        n,
        p,
        fit.lbd,
        null_repeats,
        null_validation_seed(null_seed),
        threads,
    )?;
    let tail_audit_residualized_y = if tail_audit_repeats > 0 {
        generate_lmm_null_residualized(
            &s_vec,
            &u_vec,
            &x_rot,
            n,
            p,
            fit.lbd,
            tail_audit_repeats,
            null_tail_audit_seed(null_seed),
            threads,
        )?
    } else {
        Vec::new()
    };
    let mean_s = s_vec.iter().copied().sum::<f64>() / (n as f64);
    let var_g = fit.sigma_g2 * mean_s.max(0.0);
    let denom = var_g + fit.sigma_e2;
    let pve = if denom.is_finite() && denom > 0.0 {
        var_g / denom
    } else {
        f64::NAN
    };

    if let Some(out) = eigenvectors_out {
        *out = u_vec;
    }
    Ok(GarfieldResidualResult {
        beta: fixed_design.expand_beta(&fit.beta),
        py: py_vec.clone(),
        residualized_y,
        resid,
        fitted_y: None,
        additive_effect: None,
        lbd: fit.lbd,
        lambda_marker: f64::NAN,
        residualization_method: "exact_lmm".to_string(),
        lambda_source: if no_grm_boundary {
            "reml_boundary_no_grm".to_string()
        } else {
            "reml".to_string()
        },
        no_grm_boundary,
        pcg_converged: true,
        pcg_iters: 0,
        pcg_rel_res: 0.0,
        lambda_converged: brent.converged,
        lambda_iters: brent.iterations,
        lambda_rel_res: brent.relative_bracket_width,
        ml: fit.ml,
        reml: fit.reml,
        sigma_g2: fit.sigma_g2,
        sigma_e2: fit.sigma_e2,
        pve,
        n_samples: n,
        n_fixed_effects: fit.fixed_rank,
        eigh_backend: evd_backend.to_string(),
        eigh_elapsed,
        eigenvalues: s_vec,
        eff_m,
        null_residualized_y,
        null_validation_residualized_y,
        tail_audit_residualized_y,
    })
}

/// Build the exact covariance-metric residual projection used by the
/// candidate GRM-Max6 reranker.
///
/// This is intentionally separate from the residual phenotype fit.  The
/// ordinary GARFIELD path only needs `P_C y`; the Max6 -> GRM-Max6 path also
/// needs the reusable matrix `P_C` for the candidate interaction vectors.  In
/// eigen-coordinates, with `D=(Lambda+lambda I)^-1`,
///
/// ```text
/// P_C = U D U' - U D X_rot (X_rot' D X_rot)^-1 X_rot' D U'
/// ```
///
/// where `X_rot=U'X`.  The returned matrix is row-major and is exact up to
/// the same floating-point/eigensolver arithmetic used by the null fit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_exact_projection_from_grm_rust(
    grm_vec: &[f64],
    n: usize,
    x_cov_vec: Option<&[f64]>,
    q_cov: usize,
    threads: usize,
    lambda_vc: f64,
    add_intercept: bool,
    require_lapack: bool,
) -> Result<Vec<f64>, String> {
    if lambda_vc.is_nan() || lambda_vc.is_sign_negative() {
        return Err(format!(
            "cannot build P_C with NaN/negative lambda={lambda_vc}"
        ));
    }
    validate_grm(grm_vec, n)?;
    let (original_design, original_p) = build_design_matrix(x_cov_vec, n, q_cov, add_intercept)?;
    let fixed_design = rank_aware_fixed_design(&original_design, n, original_p)?;
    let (x_full, p) = (&fixed_design.x, fixed_design.rank);
    if lambda_vc.is_infinite() {
        return build_no_grm_projection(x_full.as_slice(), n, p);
    }
    let driver_t0 = Instant::now();
    let (threads_before, _threads_in_stage, eigh_run, _threads_after) =
        crate::blas::with_eigh_thread_stage(threads, || symmetric_eigh_f64_row_major(grm_vec, n));
    let _eigh_elapsed = driver_t0.elapsed().as_secs_f64();
    let (evals, u_vec, evd_backend) = eigh_run?;
    if require_lapack && evd_backend == "nalgebra" {
        return Err(
            "P_C construction expected LAPACK backend but fell back to nalgebra".to_string(),
        );
    }
    let _ = threads_before;
    if evals.len() != n || u_vec.len() != n.saturating_mul(n) {
        return Err("P_C eigen-decomposition output shape mismatch".to_string());
    }
    let s_vec = evals
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if !value.is_finite() {
                Err(format!(
                    "non-finite eigenvalue at index {index} while building P_C"
                ))
            } else {
                Ok(value.max(0.0))
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    build_exact_projection_from_eigensystem_rust(
        &s_vec,
        &u_vec,
        n,
        x_cov_vec,
        q_cov,
        threads,
        lambda_vc,
        add_intercept,
    )
}

/// Same exact projection algebra with the eigensystem from the observed-y fit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_exact_projection_from_eigensystem_rust(
    s_vec: &[f64],
    u_vec: &[f64],
    n: usize,
    x_cov_vec: Option<&[f64]>,
    q_cov: usize,
    threads: usize,
    lambda_vc: f64,
    add_intercept: bool,
) -> Result<Vec<f64>, String> {
    if lambda_vc.is_nan() || lambda_vc.is_sign_negative() {
        return Err(format!(
            "cannot build P_C with NaN/negative lambda={lambda_vc}"
        ));
    }
    let (original_design, original_p) = build_design_matrix(x_cov_vec, n, q_cov, add_intercept)?;
    let fixed_design = rank_aware_fixed_design(&original_design, n, original_p)?;
    let (x_full, p) = (&fixed_design.x, fixed_design.rank);
    if lambda_vc.is_infinite() {
        return build_no_grm_projection(x_full.as_slice(), n, p);
    }
    let square_len = n
        .checked_mul(n)
        .ok_or_else(|| "P_C dimension overflow".to_string())?;
    if s_vec.len() != n
        || u_vec.len() != square_len
        || s_vec.iter().any(|v| !v.is_finite() || *v < 0.0)
        || u_vec.iter().any(|v| !v.is_finite())
    {
        return Err("invalid retained P_C eigensystem".to_string());
    }
    let (x_rot, _y_rot) = rotate_design_and_response(u_vec, x_full, &vec![0.0; n], n, p, threads)?;
    let mut d = vec![0.0; n];
    for (index, &eigenvalue) in s_vec.iter().enumerate() {
        let denominator = eigenvalue + lambda_vc;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(format!(
                "invalid P_C eigen denominator at index {index}: eigenvalue={eigenvalue}, lambda={lambda_vc}"
            ));
        }
        d[index] = 1.0 / denominator;
    }

    let mut xtdx = vec![0.0; p * p];
    for k in 0..n {
        let dk = d[k];
        for left in 0..p {
            let x_left = x_rot[k * p + left];
            for right in 0..=left {
                xtdx[left * p + right] += dk * x_left * x_rot[k * p + right];
            }
        }
    }
    for left in 0..p {
        for right in 0..left {
            xtdx[right * p + left] = xtdx[left * p + right];
        }
    }
    // Do not regularize the fixed-effect metric here.  The exact covariance
    // projection must annihilate every column of X (in particular the
    // intercept), otherwise the closed-form cell contrasts below are no
    // longer the corresponding quadratic forms.  Redundant/zero covariates
    // are handled by the same rank-aware pseudoinverse used by the null fit.
    let zero_rhs = vec![0.0_f64; p];
    let xtdx_inv = solve_fixed_effect_metric(&xtdx, &zero_rhs, p)
        .ok_or_else(|| "fixed-effect metric is singular/indefinite while building P_C".to_string())?
        .inverse;

    // A = U D X_rot = V^-1 X.  Keeping A explicit makes the final symmetric
    // rank-p correction simple and avoids a second sample-space projection.
    let mut a = vec![0.0; n * p];
    for row in 0..n {
        for column in 0..p {
            let mut value = 0.0;
            for k in 0..n {
                value += u_vec[row * n + k] * d[k] * x_rot[k * p + column];
            }
            a[row * p + column] = value;
        }
    }

    // U D U' is a dense matrix product, not n^2 independent scalar dots.
    // Use the same GARFIELD BLAS dispatcher as the exact candidate kernel.
    // Weight U by D (rather than sqrt(D)) to preserve the reference product
    // definition; only BLAS reduction order changes. Keep explicit symmetry
    // and the existing fixed-effect correction below.
    crate::blas::ensure_garfield_blas_backend()?;
    let mut weighted_u = u_vec.to_vec();
    for row in weighted_u.chunks_exact_mut(n) {
        for (value, &weight) in row.iter_mut().zip(d.iter()) {
            *value *= weight;
        }
    }
    let mut p_c = vec![0.0; n * n];
    super::grm_max6::dgemm_row_major(false, true, n, n, n, &weighted_u, &u_vec, &mut p_c)?;
    drop(weighted_u);
    for row in 0..n {
        for col in 0..=row {
            let p0 = p_c[row * n + col];
            let mut correction = 0.0;
            for left in 0..p {
                let a_left = a[row * p + left];
                for right in 0..p {
                    correction += a_left * xtdx_inv[left * p + right] * a[col * p + right];
                }
            }
            let value = p0 - correction;
            p_c[row * n + col] = value;
            p_c[col * n + row] = value;
        }
    }
    if p_c.iter().any(|value| !value.is_finite()) {
        return Err("P_C construction produced non-finite values".to_string());
    }
    Ok(p_c)
}

fn garfield_residual_result_to_pydict<'py>(
    py: Python<'py>,
    result: GarfieldResidualResult,
) -> PyResult<Bound<'py, PyDict>> {
    let py_arr = PyArray1::from_owned_array(py, Array1::from_vec(result.py)).into_bound();
    let resid_arr = PyArray1::from_owned_array(py, Array1::from_vec(result.resid)).into_bound();
    let beta_arr = PyArray1::from_owned_array(py, Array1::from_vec(result.beta)).into_bound();
    let s_arr = PyArray1::from_owned_array(py, Array1::from_vec(result.eigenvalues)).into_bound();
    let y_resid_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(result.residualized_y)).into_bound();

    let out = PyDict::new(py);
    out.set_item("py", py_arr)?;
    out.set_item("residualized_y", y_resid_arr)?;
    out.set_item("resid", resid_arr)?;
    out.set_item("beta", beta_arr)?;
    if let Some(fitted) = result.fitted_y {
        out.set_item(
            "fitted_y",
            PyArray1::from_owned_array(py, Array1::from_vec(fitted)).into_bound(),
        )?;
    } else {
        out.set_item("fitted_y", py.None())?;
    }
    if let Some(additive) = result.additive_effect {
        out.set_item(
            "additive_effect",
            PyArray1::from_owned_array(py, Array1::from_vec(additive)).into_bound(),
        )?;
    } else {
        out.set_item("additive_effect", py.None())?;
    }
    out.set_item("lbd", result.lbd)?;
    out.set_item("lambda_marker", result.lambda_marker)?;
    out.set_item("residualization_method", result.residualization_method)?;
    out.set_item("lambda_source", result.lambda_source)?;
    out.set_item("no_grm_boundary", result.no_grm_boundary)?;
    out.set_item("pcg_converged", result.pcg_converged)?;
    out.set_item("pcg_iters", result.pcg_iters)?;
    out.set_item("pcg_rel_res", result.pcg_rel_res)?;
    out.set_item("lambda_converged", result.lambda_converged)?;
    out.set_item("lambda_iters", result.lambda_iters)?;
    out.set_item("lambda_rel_res", result.lambda_rel_res)?;
    out.set_item("ml", result.ml)?;
    out.set_item("reml", result.reml)?;
    out.set_item("sigma_g2", result.sigma_g2)?;
    out.set_item("sigma_e2", result.sigma_e2)?;
    out.set_item("pve", result.pve)?;
    out.set_item("n_samples", result.n_samples)?;
    out.set_item("n_fixed_effects", result.n_fixed_effects)?;
    out.set_item("eigh_backend", result.eigh_backend)?;
    out.set_item("eigh_elapsed", result.eigh_elapsed)?;
    out.set_item("eigenvalues", s_arr)?;
    if let Some(m) = result.eff_m {
        out.set_item("eff_m", m)?;
    }
    let null_arrays = PyList::empty(py);
    for values in result.null_residualized_y {
        null_arrays
            .append(PyArray1::from_owned_array(py, Array1::from_vec(values)).into_bound())?;
    }
    out.set_item("null_residualized_y", null_arrays)?;
    let validation_arrays = PyList::empty(py);
    for values in result.null_validation_residualized_y {
        validation_arrays
            .append(PyArray1::from_owned_array(py, Array1::from_vec(values)).into_bound())?;
    }
    out.set_item("null_validation_residualized_y", validation_arrays)?;
    let tail_audit_arrays = PyList::empty(py);
    for values in result.tail_audit_residualized_y {
        tail_audit_arrays
            .append(PyArray1::from_owned_array(py, Array1::from_vec(values)).into_bound())?;
    }
    out.set_item("tail_audit_residualized_y", tail_audit_arrays)?;
    Ok(out)
}

fn garfield_residualize_exact_from_grm_impl<'py>(
    py: Python<'py>,
    grm_vec: Vec<f64>,
    n: usize,
    y_vec: Vec<f64>,
    x_cov_vec: Option<Vec<f64>>,
    q_cov: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    eff_m: Option<usize>,
    null_repeats: usize,
    null_seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let result = garfield_residualize_exact_from_grm_rust(
        grm_vec,
        n,
        y_vec,
        x_cov_vec,
        q_cov,
        threads,
        low,
        high,
        max_iter,
        tol,
        add_intercept,
        exact_n_max,
        require_lapack,
        eff_m,
        null_repeats,
        null_seed,
        0,
    )
    .map_err(PyRuntimeError::new_err)?;

    garfield_residual_result_to_pydict(py, result)
}

#[pyfunction(name = "garfield_residualize_grm")]
#[pyo3(signature = (
    grm,
    y,
    x_cov=None,
    threads=0,
    low=-5.0_f64,
    high=5.0_f64,
    max_iter=50_usize,
    tol=1e-3_f64,
    add_intercept=true,
    exact_n_max=15_000_usize,
    require_lapack=false
))]
pub fn garfield_residualize_grm_py<'py>(
    py: Python<'py>,
    grm: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let grm_arr = grm.as_array();
    if grm_arr.ndim() != 2 || grm_arr.shape()[0] != grm_arr.shape()[1] {
        return Err(PyRuntimeError::new_err(
            "grm must be a square float64 matrix.",
        ));
    }
    let n = grm_arr.shape()[0];
    let grm_vec: Vec<f64> = match crate::stats_common::array_standard_slice(&grm) {
        Ok(s) => s.to_vec(),
        Err(_) => crate::stats_common::copy_array_standard_layout(&grm_arr),
    };
    let y_vec: Vec<f64> = match crate::stats_common::array_standard_slice(&y) {
        Ok(s) => s.to_vec(),
        Err(_) => crate::stats_common::copy_array_standard_layout(&y.as_array()),
    };
    let (x_cov_vec, q_cov) = if let Some(xc) = x_cov {
        let xc_arr = xc.as_array();
        if xc_arr.ndim() != 2 {
            return Err(PyRuntimeError::new_err("x_cov must be 2D (n, q_cov)."));
        }
        if xc_arr.shape()[0] != n {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov row mismatch: got {}, expected {n}",
                xc_arr.shape()[0]
            )));
        }
        let q_cov = xc_arr.shape()[1];
        let cov_vec = match crate::stats_common::array_standard_slice(&xc) {
            Ok(s) => s.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&xc_arr),
        };
        (Some(cov_vec), q_cov)
    } else {
        (None, 0usize)
    };
    garfield_residualize_exact_from_grm_impl(
        py,
        grm_vec,
        n,
        y_vec,
        x_cov_vec,
        q_cov,
        threads,
        low,
        high,
        max_iter,
        tol,
        add_intercept,
        exact_n_max,
        require_lapack,
        None,
        0,
        0,
    )
}

#[pyfunction(name = "garfield_residualize_bed")]
#[pyo3(signature = (
    prefix,
    y,
    x_cov=None,
    maf_threshold=0.02_f32,
    max_missing_rate=0.05_f32,
    block_cols=65_536_usize,
    threads=0_usize,
    progress_callback=None,
    progress_every=0_usize,
    low=-5.0_f64,
    high=5.0_f64,
    max_iter=50_usize,
    tol=1e-3_f64,
    add_intercept=true,
    exact_n_max=15_000_usize,
    require_lapack=false,
    site_keep=None,
    residual_method="exact",
    pcg_max_iter=100_usize,
    pcg_tol=1e-4_f64,
    pcg_small_n_max=15_000_usize,
    null_repeats=0_usize,
    seed=42_u64
))]
pub fn garfield_residualize_bed_py<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    maf_threshold: f32,
    max_missing_rate: f32,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    residual_method: &str,
    pcg_max_iter: usize,
    pcg_tol: f64,
    pcg_small_n_max: usize,
    null_repeats: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let y_vec: Vec<f64> = match crate::stats_common::array_standard_slice(&y) {
        Ok(s) => s.to_vec(),
        Err(_) => crate::stats_common::copy_array_standard_layout(&y.as_array()),
    };
    let (x_cov_vec, q_cov) = if let Some(xc) = x_cov {
        let xc_arr = xc.as_array();
        if xc_arr.ndim() != 2 {
            return Err(PyRuntimeError::new_err("x_cov must be 2D (n, q_cov)."));
        }
        let q_cov = xc_arr.shape()[1];
        let cov_vec = match crate::stats_common::array_standard_slice(&xc) {
            Ok(s) => s.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&xc_arr),
        };
        (Some(cov_vec), q_cov)
    } else {
        (None, 0usize)
    };

    if residual_method.trim().eq_ignore_ascii_case("rrblup_pcg")
        || residual_method.trim().eq_ignore_ascii_case("rrblup-pcg")
        || residual_method.trim().eq_ignore_ascii_case("pcg")
    {
        if pcg_max_iter == 0 {
            return Err(PyValueError::new_err(
                "pcg_max_iter must be > 0 for rrblup_pcg residualization",
            ));
        }
        if !pcg_tol.is_finite() || pcg_tol <= 0.0 {
            return Err(PyValueError::new_err(
                "pcg_tol must be finite and > 0 for rrblup_pcg residualization",
            ));
        }
        if pcg_small_n_max == 0 {
            return Err(PyValueError::new_err(
                "pcg_small_n_max must be > 0 for rrblup_pcg residualization",
            ));
        }
        let n_samples = crate::gfcore::read_fam(&prefix)
            .map_err(PyRuntimeError::new_err)?
            .len();
        let n_snps = crate::gfcore::read_bim(&prefix)
            .map_err(PyRuntimeError::new_err)?
            .len();
        let (row_source_indices, row_flip, row_maf) = if let Some(mask) = site_keep {
            let values = match crate::stats_common::array_standard_slice(&mask) {
                Ok(slice) => slice.to_vec(),
                Err(_) => crate::stats_common::copy_array_standard_layout(&mask.as_array()),
            };
            if values.len() != n_snps {
                return Err(PyRuntimeError::new_err(format!(
                    "site_keep length mismatch: got {}, expected {n_snps}",
                    values.len()
                )));
            }
            let row_source_indices = values
                .iter()
                .enumerate()
                .filter_map(|(idx, keep)| keep.then_some(idx))
                .collect::<Vec<_>>();
            if row_source_indices.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "site_keep retained zero markers for PCG residualization",
                ));
            }
            let meta = crate::gfreader::compute_bed_row_meta_owned_for_source_rows(
                &prefix,
                row_source_indices.as_slice(),
                None,
            )
            .map_err(PyRuntimeError::new_err)?;
            (row_source_indices, meta.row_flip, meta.maf)
        } else {
            let meta =
                crate::gfreader::prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window(
                    &prefix,
                    maf_threshold,
                    max_missing_rate,
                    1.0,
                    false,
                    None,
                    true,
                    None,
                    threads,
                )
                .map_err(PyRuntimeError::new_err)?;
            let row_source_indices = meta.row_source_indices;
            let row_meta = crate::gfreader::compute_bed_row_meta_owned_for_source_rows(
                &prefix,
                row_source_indices.as_slice(),
                None,
            )
            .map_err(PyRuntimeError::new_err)?;
            (row_source_indices, row_meta.row_flip, row_meta.maf)
        };
        if row_source_indices.is_empty() {
            return Err(PyRuntimeError::new_err(
                "PCG residualization retained zero QC markers",
            ));
        }
        if y_vec.len() != n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "y length mismatch: got {}, expected {n_samples} samples from BED input",
                y_vec.len()
            )));
        }
        if let Some(cov) = x_cov_vec.as_ref() {
            if cov.len() != n_samples.saturating_mul(q_cov) {
                return Err(PyRuntimeError::new_err(format!(
                    "x_cov payload length mismatch: got {}, expected {}",
                    cov.len(),
                    n_samples.saturating_mul(q_cov)
                )));
            }
        }
        let config = PcgResidualConfig {
            block_rows: block_cols.max(1),
            threads,
            small_n_max: pcg_small_n_max,
            reml_low: low,
            reml_high: high,
            reml_max_iter: max_iter,
            reml_tol: tol,
            pcg_max_iter,
            pcg_tol,
            seed,
            ..PcgResidualConfig::default()
        };
        let result = garfield_residualize_pcg_from_bed(
            prefix.as_str(),
            row_source_indices,
            row_flip,
            row_maf,
            (0..n_samples).collect::<Vec<_>>(),
            y_vec,
            x_cov_vec,
            q_cov,
            threads,
            add_intercept,
            config,
            null_repeats,
            seed,
        )
        .map_err(PyRuntimeError::new_err)?;
        return garfield_residual_result_to_pydict(py, result);
    }
    if !residual_method.trim().eq_ignore_ascii_case("exact") {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported residual_method '{}'; expected 'exact' or 'rrblup_pcg'",
            residual_method
        )));
    }

    let (grm_f32_arr, eff_m, n_samples) = crate::grm::grm_packed_bed_f32(
        py,
        prefix,
        1,
        maf_threshold,
        max_missing_rate,
        0.0,
        false,
        block_cols,
        threads,
        progress_callback,
        progress_every,
    )?;
    if y_vec.len() != n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "y length mismatch: got {}, expected {n_samples} samples from BED input",
            y_vec.len()
        )));
    }
    if let Some(cov) = x_cov_vec.as_ref() {
        if cov.len() != n_samples * q_cov {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov payload length mismatch: got {}, expected {}",
                cov.len(),
                n_samples * q_cov
            )));
        }
    }
    let grm_f32_ro = grm_f32_arr.readonly();
    let grm_f32: Cow<[f32]> = match crate::stats_common::array_standard_slice(&grm_f32_ro) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &grm_f32_ro.as_array(),
        )),
    };
    let grm_vec: Vec<f64> = grm_f32.iter().map(|&v| v as f64).collect();
    garfield_residualize_exact_from_grm_impl(
        py,
        grm_vec,
        n_samples,
        y_vec,
        x_cov_vec,
        q_cov,
        threads,
        low,
        high,
        max_iter,
        tol,
        add_intercept,
        exact_n_max,
        require_lapack,
        Some(eff_m),
        null_repeats,
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        build_exact_projection_from_grm_rust, build_no_grm_covariate_projector,
        build_no_grm_projection, exact_no_grm_fit_from_rotated, generate_lmm_null_residualized,
        null_tail_audit_seed, null_validation_seed, standardize_residualized_y,
    };

    fn sample_mean(values: &[f64]) -> f64 {
        values.iter().copied().sum::<f64>() / (values.len() as f64)
    }

    fn sample_std(values: &[f64]) -> f64 {
        let mean = sample_mean(values);
        let ss = values
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f64>();
        (ss / ((values.len() - 1) as f64)).sqrt()
    }

    #[test]
    fn standardize_residualized_y_sets_unit_sample_std() {
        let mut values = vec![2.0, 4.0, 6.0, 8.0];
        standardize_residualized_y(&mut values).unwrap();
        assert!(sample_mean(&values).abs() < 1e-12);
        assert!((sample_std(&values) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn standardize_residualized_y_rejects_zero_variance() {
        let mut values = vec![3.0, 3.0, 3.0];
        let err = standardize_residualized_y(&mut values).unwrap_err();
        assert!(err.contains("zero-variance Py"));
    }

    #[test]
    fn conditional_lmm_null_is_seeded_and_finite() {
        let s = vec![0.0, 0.5, 1.0, 2.0, 3.0, 4.0];
        let u = (0..s.len() * s.len())
            .map(|idx| {
                if idx / s.len() == idx % s.len() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let x_rot = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let first =
            generate_lmm_null_residualized(&s, &u, &x_rot, s.len(), 2, 1.0, 8, 1234, 1).unwrap();
        let second =
            generate_lmm_null_residualized(&s, &u, &x_rot, s.len(), 2, 1.0, 8, 1234, 1).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 8);
        assert!(first.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn validation_null_uses_a_deterministic_independent_rng_stream() {
        let calibration_seed = 1234;
        let validation_seed = null_validation_seed(calibration_seed);
        assert_ne!(validation_seed, calibration_seed);
        assert_eq!(validation_seed, null_validation_seed(calibration_seed));

        let s = vec![0.0, 0.5, 1.0, 2.0, 3.0, 4.0];
        let u = (0..s.len() * s.len())
            .map(|idx| {
                if idx / s.len() == idx % s.len() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let x_rot = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let calibration =
            generate_lmm_null_residualized(&s, &u, &x_rot, s.len(), 2, 1.0, 8, calibration_seed, 1)
                .unwrap();
        let validation =
            generate_lmm_null_residualized(&s, &u, &x_rot, s.len(), 2, 1.0, 8, validation_seed, 1)
                .unwrap();
        let validation_again = generate_lmm_null_residualized(
            &s,
            &u,
            &x_rot,
            s.len(),
            2,
            1.0,
            8,
            null_validation_seed(calibration_seed),
            1,
        )
        .unwrap();
        assert_ne!(calibration, validation);
        assert_eq!(validation, validation_again);
    }

    #[test]
    fn tail_audit_null_stream_is_deterministic_and_domain_separated() {
        for root_seed in [0, 1, 1234, u64::MAX] {
            let first = null_tail_audit_seed(root_seed);
            assert_eq!(first, null_tail_audit_seed(root_seed));
            assert_ne!(first, root_seed);
            assert_ne!(first, null_validation_seed(root_seed));
        }
    }

    #[test]
    fn exact_projection_is_symmetric_and_removes_intercept() {
        let n = 4usize;
        let grm = (0..n * n)
            .map(|index| if index / n == index % n { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let p_c =
            build_exact_projection_from_grm_rust(grm.as_slice(), n, None, 0, 1, 2.0, true, false)
                .unwrap();
        for row in 0..n {
            let row_sum = (0..n).map(|col| p_c[row * n + col]).sum::<f64>();
            assert!(row_sum.abs() < 1.0e-10);
            for col in 0..n {
                assert!((p_c[row * n + col] - p_c[col * n + row]).abs() < 1.0e-12);
            }
        }
        assert!((p_c[0] - 0.25).abs() < 1.0e-10);
    }

    #[test]
    fn exact_projection_removes_full_rank_covariate_without_ridge() {
        let n = 5usize;
        let grm = (0..n * n)
            .map(|index| if index / n == index % n { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let covariate = (0..n).map(|value| value as f64).collect::<Vec<_>>();
        let p_c = build_exact_projection_from_grm_rust(
            grm.as_slice(),
            n,
            Some(covariate.as_slice()),
            1,
            1,
            2.0,
            true,
            false,
        )
        .unwrap();
        for row in 0..n {
            let intercept_projection = (0..n).map(|col| p_c[row * n + col]).sum::<f64>();
            let covariate_projection = (0..n)
                .map(|col| p_c[row * n + col] * covariate[col])
                .sum::<f64>();
            assert!(intercept_projection.abs() < 1.0e-10);
            assert!(covariate_projection.abs() < 1.0e-10);
        }
    }

    #[test]
    fn infinite_lambda_uses_no_grm_fixed_effect_projection() {
        let n = 5usize;
        let grm = (0..n * n)
            .map(|index| if index / n == index % n { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let covariate = (0..n).map(|value| value as f64).collect::<Vec<_>>();
        let p_c = build_exact_projection_from_grm_rust(
            grm.as_slice(),
            n,
            Some(covariate.as_slice()),
            1,
            1,
            f64::INFINITY,
            true,
            false,
        )
        .unwrap();

        let mut x = vec![0.0; n * 2];
        for i in 0..n {
            x[i * 2] = 1.0;
            x[i * 2 + 1] = covariate[i];
        }
        let mut xtx = [[0.0_f64; 2]; 2];
        for i in 0..n {
            for left in 0..2 {
                for right in 0..2 {
                    xtx[left][right] += x[i * 2 + left] * x[i * 2 + right];
                }
            }
        }
        let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
        let inv = [
            [xtx[1][1] / det, -xtx[0][1] / det],
            [-xtx[1][0] / det, xtx[0][0] / det],
        ];
        for row in 0..n {
            for col in 0..n {
                let mut expected = if row == col { 1.0 } else { 0.0 };
                for left in 0..2 {
                    for right in 0..2 {
                        expected -= x[row * 2 + left] * inv[left][right] * x[col * 2 + right];
                    }
                }
                assert!((p_c[row * n + col] - expected).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn no_grm_boundary_fit_uses_ols_residual_variance_and_precision() {
        let x_rot = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let y_rot = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fit = exact_no_grm_fit_from_rotated(&x_rot, &y_rot, 5, 1).unwrap();
        assert!(fit.lbd.is_infinite() && fit.lbd.is_sign_positive());
        assert_eq!(fit.sigma_g2, 0.0);
        assert!((fit.sigma_e2 - 2.5).abs() < 1.0e-12);
        assert!(fit.resid_rot.iter().sum::<f64>().abs() < 1.0e-12);
        assert!(fit
            .py_rot
            .iter()
            .zip(fit.resid_rot.iter())
            .all(|(&py, &resid)| (py - resid / fit.sigma_e2).abs() < 1.0e-12));
    }

    #[test]
    fn no_grm_qr_projector_matches_dense_projection_with_covariates() {
        let n = 7usize;
        let p = 3usize;
        let design = (0..n)
            .flat_map(|row| [1.0, row as f64, (row as f64).powi(2)])
            .collect::<Vec<_>>();
        let projector = build_no_grm_covariate_projector(&design, n, p).unwrap();
        let dense = build_no_grm_projection(&design, n, p).unwrap();
        assert_eq!(projector.n_samples(), n);
        assert_eq!(projector.rank(), p);

        let left = (0..n).map(|row| (row as f64) - 1.5).collect::<Vec<_>>();
        let right = (0..n)
            .map(|row| (2 * row) as f64 - 0.25)
            .collect::<Vec<_>>();
        let dense_projected = (0..n)
            .map(|row| {
                (0..n)
                    .map(|col| dense[row * n + col] * right[col])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let compact_projected = projector.project(&right).unwrap();
        for (actual, expected) in compact_projected.iter().zip(dense_projected.iter()) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        let dense_dot = left
            .iter()
            .enumerate()
            .map(|(row, &value)| value * dense_projected[row])
            .sum::<f64>();
        let compact_dot = projector.projected_dot(&left, &right).unwrap();
        assert!((compact_dot - dense_dot).abs() < 1.0e-10);
    }

    #[test]
    fn no_grm_qr_projector_handles_redundant_covariate() {
        let n = 6usize;
        let p = 3usize;
        let design = (0..n)
            .flat_map(|row| [1.0, row as f64, 2.0 * row as f64])
            .collect::<Vec<_>>();
        let projector = build_no_grm_covariate_projector(&design, n, p).unwrap();
        assert_eq!(projector.rank(), 2);
        let intercept = vec![1.0_f64; n];
        let projected = projector.project(&intercept).unwrap();
        assert!(projected.iter().all(|value| value.abs() < 1.0e-10));
    }
}
