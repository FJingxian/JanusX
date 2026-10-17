//! Bayesian marker-effect solvers over additive PLINK BED blocks.
//!
//! Let `Z` denote the standardized additive genotype matrix after marker
//! filtering and sample subsetting, with markers in rows and samples in
//! columns. This module fits
//!
//! `y = X alpha + Z' beta + e,  e ~ N(0, sigma_e^2 I)`.
//!
//! The residual inverse-chi-square prior uses `prior_ss_e = nu_0 * S_0^2` as
//! its sum-of-squares parameter. Consequently, each update has the form
//! `(RSS + prior_ss_e) / chi2(n + nu_0)`; `prior_ss_e` is not `S_0^2` alone.
//!
//! The three maintained priors are:
//!
//! - `BayesA`: each marker has its own variance,
//!   `beta_j | sigma_bj^2 ~ N(0, sigma_bj^2)`,
//!   `sigma_bj^2 ~ scaled-inv-chi^2`.
//! - `BayesB`: marker inclusion indicator
//!   `delta_j ~ Bernoulli(pi)`, inactive markers have `beta_j = 0`, and active
//!   markers follow the BayesA variance hierarchy.
//! - `BayesC`: the same spike-and-slab inclusion structure as BayesB, but
//!   active markers share a common marker variance and `pi` is updated from its
//!   Beta-Binomial posterior.
//!
//! All maintained paths stream standardized BED blocks from either resident
//! packed rows or `WindowedBedMatrix`. Small resident packed inputs may be
//! predecoded once, but the streaming path remains the default maintained route
//! for GS. Each MCMC iteration updates fixed effects, residual variance, marker
//! effects, and inclusion/variance hyperparameters. Production kernels monitor
//! split-chain R-hat for a single chain and pool native chain summaries for
//! multi-chain calls using a recent retained-state window of predictive
//! summaries, discard pre-convergence samples once the threshold is reached,
//! and return means from the next 1000 posterior samples. If the limit is
//! reached without convergence, the next 1000 iterations form the fallback
//! posterior window. Trace-only kernels retain explicit thinning for
//! diagnostic plots.

use nalgebra::{DMatrix, DVector};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAny, PyDict, PyFloat, PyInt, PyTuple};
use pyo3::{prelude::*, BoundObject};
use rand::rngs::{OsRng, StdRng};
use rand::{Rng, SeedableRng, TryRngCore};
use rand_distr::{Beta, ChiSquared, Gamma, StandardNormal};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, OnceLock};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

#[cfg(all(test, target_arch = "aarch64"))]
use std::arch::aarch64::{
    vcvt_f64_f32, vdupq_n_f64, vfmaq_f64, vget_high_f32, vget_low_f32, vld1q_f64, vst1q_f64,
};

use crate::bedmath::{decode_standardized_packed_block_f32, is_identity_indices};
#[cfg(test)]
use crate::blas::cblas_daxpy_dispatch;
use crate::blas::{
    cblas_ddot_dispatch, cblas_dgemm_dispatch, CblasInt, OpenBlasThreadGuard, CBLAS_COL_MAJOR,
    CBLAS_NO_TRANS, CBLAS_TRANS,
};
use crate::decode::decode_prepared_additive_block_packed_f32;
use crate::gload::WindowedBedMatrix;
use crate::stats_common::{get_cached_pool, parse_index_vec_i64_value_error};

/// Dense marker input held for the duration of a native sampler call.
///
/// A contiguous float32 NumPy array is kept as a `PyReadonlyArray2` so its
/// borrow remains active while the GIL is detached.  The sampler can then
/// read the marker rows directly without first allocating and copying a
/// second `Vec<f32>`.  Non-contiguous and float64 inputs use the owned
/// fallback, preserving the old logical row-major conversion semantics.
pub(crate) enum DenseF32Input<'py> {
    Borrowed {
        array: PyReadonlyArray2<'py, f32>,
        rows: usize,
        cols: usize,
    },
    Owned {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
}

impl DenseF32Input<'_> {
    #[inline]
    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Borrowed { rows, .. } | Self::Owned { rows, .. } => *rows,
        }
    }

    #[inline]
    pub(crate) fn cols(&self) -> usize {
        match self {
            Self::Borrowed { cols, .. } | Self::Owned { cols, .. } => *cols,
        }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[f32] {
        match self {
            Self::Borrowed { array, .. } => array
                .as_slice()
                .expect("borrowed dense float32 input must be contiguous"),
            Self::Owned { data, .. } => data,
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed { .. })
    }
}

pub(crate) fn array2_to_f32_input<'py>(
    obj: &Bound<'py, PyAny>,
    label: &str,
) -> PyResult<DenseF32Input<'py>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'py, f32>>() {
        let view = arr.as_array();
        let (rows, cols) = view.dim();
        if crate::stats_common::array_standard_slice(&arr).is_ok() {
            return Ok(DenseF32Input::Borrowed {
                array: arr,
                rows,
                cols,
            });
        }
        return Ok(DenseF32Input::Owned {
            data: view.iter().copied().collect(),
            rows,
            cols,
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'py, f64>>() {
        let view = arr.as_array();
        let (rows, cols) = view.dim();
        return Ok(DenseF32Input::Owned {
            data: view.iter().map(|value| *value as f32).collect(),
            rows,
            cols,
        });
    }
    Err(PyValueError::new_err(format!(
        "{label} must be a 2D float32 or float64 numpy array"
    )))
}

fn array1_to_vec(arr: &PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn array2_to_vec(arr: &PyReadonlyArray2<f64>) -> Vec<f64> {
    let view = arr.as_array();
    let (n, p) = view.dim();
    let mut out = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            out.push(view[[i, j]]);
        }
    }
    out
}

#[inline]
pub(crate) fn copy_f64_to_f32(src: &[f64], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for (out, value) in dst.iter_mut().zip(src.iter()) {
        *out = *value as f32;
    }
}

#[inline]
pub(crate) fn copy_f32_to_f64(src: &[f32], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    for (out, value) in dst.iter_mut().zip(src.iter()) {
        *out = *value as f64;
    }
}

fn parse_optional_index_vec_i64(
    indices: Option<&PyReadonlyArray1<i64>>,
    upper_bound: usize,
    label: &str,
) -> PyResult<Vec<usize>> {
    match indices {
        Some(arr) => parse_index_vec_i64_value_error(arr.as_slice()?, upper_bound, label),
        None => Ok(Vec::new()),
    }
}

fn parse_index_vec_i64_string(
    raw: &[i64],
    upper_bound: usize,
    label: &str,
) -> Result<Vec<usize>, String> {
    parse_index_vec_i64_value_error(raw, upper_bound, label).map_err(|e| e.to_string())
}

#[derive(Debug)]
struct PackedBayesTraceResult {
    beta: Vec<f64>,
    alpha: Vec<f64>,
    vare: f64,
    h2_mean: f64,
    var_h2: f64,
    prob_in_mean: f64,
    n_active_mean: f64,
    iter_trace: Vec<i64>,
    h2_trace: Vec<f64>,
    var_e_trace: Vec<f64>,
    prob_in_trace: Vec<f64>,
    n_active_trace: Vec<f64>,
    full_iter_trace: Vec<i64>,
    full_h2_trace: Vec<f64>,
    full_var_e_trace: Vec<f64>,
    full_prob_in_trace: Vec<f64>,
    full_n_active_trace: Vec<f64>,
    beta_trace_indices: Vec<i64>,
    beta_trace: Vec<f64>,
}

fn posterior_keep_iters(n_iter: usize, burnin: usize, thin: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for it in 0..n_iter {
        if it >= burnin && ((it - burnin) % thin == 0) {
            out.push((it + 1) as i64);
        }
    }
    out
}

pub(crate) const BAYES_AUTO_BURNIN: usize = 500;

#[derive(Clone, Copy, Debug)]
pub(crate) struct BayesSamplingOptions {
    pub(crate) burnin: Option<usize>,
    pub(crate) adaptive_burnin: bool,
    pub(crate) posterior_target: usize,
}

impl BayesSamplingOptions {
    #[inline]
    pub(crate) fn new(
        burnin: Option<usize>,
        adaptive_burnin: bool,
        posterior_target: usize,
    ) -> Self {
        Self {
            burnin,
            adaptive_burnin,
            posterior_target: posterior_target.max(1),
        }
    }
}

#[inline]
fn resolve_burnin(burnin: Option<usize>) -> (usize, bool) {
    match burnin {
        Some(value) => (value, false),
        None => (BAYES_AUTO_BURNIN, true),
    }
}

#[inline]
fn resolve_burnin_mode(burnin: Option<usize>, adaptive_burnin: bool) -> (usize, bool) {
    if adaptive_burnin {
        (burnin.unwrap_or(BAYES_AUTO_BURNIN), true)
    } else {
        resolve_burnin(burnin)
    }
}

#[inline]
fn monitor_end_iteration(n_iter: usize, burnin: usize, thin: usize, adaptive: bool) -> usize {
    let first_retained = burnin.saturating_add(1);
    if !adaptive {
        return first_retained;
    }
    let lower_bound = n_iter.max(first_retained);
    let delta = lower_bound.saturating_sub(first_retained);
    let steps = delta / thin + usize::from(delta % thin != 0);
    first_retained.saturating_add(steps.saturating_mul(thin))
}

// A single MCMC chain does not support the classical multi-chain Gelman--Rubin
// diagnostic.  We therefore report the standard split-chain R-hat calculated
// from each retained scalar report metric (h2, variance components, and model
// hyperparameters).  The two consecutive halves are treated as two chains;
// single-chain convergence uses the maximum finite scalar R-hat.  Adaptive
// multi-chain stopping uses the predictive monitor declared below instead.
// Keep the early-stop criterion aligned with the GS user-facing contract.
// Keep the conventional conservative convergence cutoff for split-chain R-hat.
const BAYES_RHAT_THRESHOLD: f64 = 1.10;
const BAYES_RHAT_WINDOW: usize = 500;
const BAYES_RHAT_MIN_KEEP: usize = BAYES_RHAT_WINDOW;
const BAYES_RHAT_CHECK_EVERY: usize = 50;
const BAYES_RHAT_STABLE_CHECKS: usize = 3;
const BAYES_MAX_POSTERIOR_RESTARTS: usize = 1;
pub(crate) const BAYES_POSTERIOR_SAMPLES: usize = 1000;

// BayesA/B use marker-specific latent variances.  Their mean `var_b` is an
// auxiliary prior state (and for BayesB also includes inactive-marker draws),
// so monitoring it makes convergence spuriously fail without reflecting the
// stability of the fitted genetic/residual components.  BayesC has one shared
// slab variance, so its `var_b` remains a meaningful global monitor.
pub(crate) const BAYES_RHAT_A_NAMES: &[&str] = &["h2", "var_g", "var_e"];
pub(crate) const BAYES_RHAT_B_NAMES: &[&str] = &["h2", "var_g", "var_e", "prob_in", "n_active"];
pub(crate) const BAYES_RHAT_C_NAMES: &[&str] =
    &["h2", "var_g", "var_e", "var_b", "prob_in", "n_active"];
pub(crate) const BAYES_RHAT_R_NAMES: &[&str] = &[
    "h2",
    "var_g",
    "var_e",
    "sigma_lambda2",
    "pi0",
    "pi1",
    "pi2",
    "pi3",
    "n_active",
];

// Adaptive stopping targets predictive state rather than mixture-architecture
// parameters.  The latter remain in the final report, but can mix much more
// slowly than GEBV without materially changing prediction.
pub(crate) const BAYES_PREDICTIVE_MONITOR_NAMES: &[&str] = &[
    "h2",
    "var_g",
    "var_e",
    "gebv_mean",
    "gebv_var",
    "gebv_cov_y",
    "gebv_sketch_0",
    "gebv_sketch_1",
    "gebv_sketch_2",
    "gebv_sketch_3",
    "gebv_sketch_4",
    "gebv_sketch_5",
    "gebv_sketch_6",
    "gebv_sketch_7",
];

#[inline]
pub(crate) fn finite_rhat_max(values: &[f64]) -> f64 {
    // A positive infinite R-hat is a useful diagnostic (zero within-chain
    // variance with a non-zero between-chain shift) and must remain the
    // maximum rather than being hidden as NaN.  NaN and negative values are
    // invalid monitor outputs and invalidate the aggregate statistic.
    if values.is_empty() || values.iter().any(|value| value.is_nan() || *value < 0.0) {
        return f64::NAN;
    }
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Convert the native scalar R-hat vector to a Python mapping.  The mapping
/// is intentionally flat so callers can print `Rhat(h2)`, `Rhat(var_g)`, …
/// without knowing the model-specific tuple layout.  `rhat_max` is the
/// maximum of the reported diagnostics; adaptive multi-chain stopping uses a
/// separate predictive monitor, while these entries remain available for
/// diagnostics and backwards-compatible `rhat_h2` handling.
pub(crate) fn rhat_metrics_to_py<'py>(
    py: Python<'py>,
    names: &[&str],
    values: &[f64],
) -> PyResult<Py<PyAny>> {
    let mapping = PyDict::new(py);
    mapping.set_item("rhat_max", finite_rhat_max(values))?;
    for (name, value) in names.iter().zip(values.iter()) {
        mapping.set_item(*name, *value)?;
    }
    Ok(mapping.into_any().unbind())
}

fn py_tuple_from_items<'py>(py: Python<'py>, items: Vec<Py<PyAny>>) -> PyResult<Py<PyTuple>> {
    Ok(PyTuple::new(py, items)?.unbind())
}

#[inline]
fn py_float_item(py: Python<'_>, value: f64) -> Py<PyAny> {
    PyFloat::new(py, value).into_any().unbind()
}

#[inline]
fn py_usize_item(py: Python<'_>, value: usize) -> Py<PyAny> {
    PyInt::new(py, value as u64).into_any().unbind()
}

/// Resolve the native multi-chain budget. Chains are capped at half of the
/// requested worker budget so marker decoding/BLAS work still has room to run;
/// a one-thread caller always gets one chain.
pub(crate) fn effective_bayes_chains(chains: usize, threads: usize) -> Result<usize, String> {
    if chains == 0 {
        return Err("chains must be > 0".to_string());
    }
    let thread_budget = threads.max(1);
    Ok(chains.min((thread_budget / 2).max(1)))
}

/// Build the native chain scheduler. The pool is deliberately separate from
/// the default Rayon pool so a GS call cannot silently consume unrelated
/// process-wide workers. `threads` is the total Bayesian budget; chain count
/// is capped independently by `effective_bayes_chains`.
pub(crate) fn build_bayes_chain_pool(
    threads: usize,
) -> Result<Option<Arc<rayon::ThreadPool>>, String> {
    if threads <= 1 {
        return Ok(None);
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| format!("Bayes chain thread pool: {error}"))?,
    );
    Ok(Some(pool))
}

/// Run an independent operation on every chain. With no scheduler this keeps
/// the single-chain/low-thread path allocation-free; multi-chain calls use
/// the explicit pool and therefore no longer fall back to a serial `for`.
pub(crate) fn bayes_chain_for_each_mut<T, F>(
    states: &mut [T],
    pool: Option<&Arc<rayon::ThreadPool>>,
    operation: F,
) where
    T: Send,
    F: Fn(&mut T) + Send + Sync,
{
    if let Some(pool) = pool {
        pool.install(|| states.par_iter_mut().for_each(&operation));
    } else {
        states.iter_mut().for_each(operation);
    }
}

pub(crate) fn bayes_chain_try_for_each_mut<T, F>(
    states: &mut [T],
    pool: Option<&Arc<rayon::ThreadPool>>,
    operation: F,
) -> Result<(), String>
where
    T: Send,
    F: Fn(&mut T) -> Result<(), String> + Send + Sync,
{
    if let Some(pool) = pool {
        pool.install(|| states.par_iter_mut().try_for_each(operation))
    } else {
        states.iter_mut().try_for_each(operation)
    }
}

/// Fallible indexed map used for the post-marker hyperparameter update. The
/// returned vector preserves chain order for synchronized R-hat observation.
pub(crate) fn bayes_chain_try_map_mut<T, R, F>(
    states: &mut [T],
    pool: Option<&Arc<rayon::ThreadPool>>,
    operation: F,
) -> Result<Vec<R>, String>
where
    T: Send,
    R: Send,
    F: Fn(&mut T) -> Result<R, String> + Send + Sync,
{
    if let Some(pool) = pool {
        pool.install(|| states.par_iter_mut().map(operation).collect())
    } else {
        states.iter_mut().map(operation).collect()
    }
}

#[inline]
fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive independent chain seeds inside Rust. The default one-chain/no-seed
/// path remains OS-random for compatibility with existing GS runs.
pub(crate) fn bayes_chain_seeds(seed: Option<u64>, chains: usize) -> Vec<Option<u64>> {
    if chains <= 1 {
        return vec![seed];
    }
    let root = seed.unwrap_or_else(|| {
        let mut bytes = [0u8; 8];
        if OsRng.try_fill_bytes(&mut bytes).is_ok() {
            u64::from_le_bytes(bytes)
        } else {
            42
        }
    });
    (0..chains)
        .map(|index| {
            Some(splitmix64(root.wrapping_add(
                (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            )))
        })
        .collect()
}

const BAYES_ADAPTIVE_CHAIN_START_SCALES: [f64; 4] = [0.7, 1.0, 1.4, 2.0];

/// Return a deterministic overdispersed initial-state scale for adaptive
/// multi-chain sampling. This changes only the starting latent state; the
/// prior calibration and all full-conditionals remain unchanged.
pub(crate) fn bayes_chain_start_scale(chain_index: usize, chains: usize, adaptive: bool) -> f64 {
    if !adaptive || chains <= 1 {
        return 1.0;
    }
    match chains {
        2 => [0.7, 2.0][chain_index.min(1)],
        3 => [0.7, 1.0, 2.0][chain_index.min(2)],
        _ => BAYES_ADAPTIVE_CHAIN_START_SCALES[chain_index.min(3)],
    }
}

/// Scale a Bernoulli inclusion probability through its odds, keeping it in
/// (0, 1) while preserving the requested base probability at scale 1.
#[inline]
pub(crate) fn bayes_chain_start_probability(base: f64, scale: f64) -> f64 {
    if scale == 1.0 {
        return base;
    }
    let odds = (base / (1.0 - base)) * scale;
    odds / (1.0 + odds)
}

/// Pool scalar posterior summaries from native chains. The within-chain
/// variance follows the population-moment convention used by the existing
/// kernels; the between-chain term is then added using total variance.
pub(crate) fn aggregate_bayes_h2(
    h2_means: &[f64],
    h2_vars: &[f64],
    chain_rhats: &[f64],
    n_post: usize,
) -> (f64, f64, f64) {
    if h2_means.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mean = h2_means.iter().sum::<f64>() / h2_means.len() as f64;
    let var = h2_vars
        .iter()
        .zip(h2_means.iter())
        .map(|(within, value)| within.max(0.0) + (value - mean) * (value - mean))
        .sum::<f64>()
        / h2_means.len() as f64;
    let mut rhat = chain_rhats
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NAN, f64::max);
    if h2_means.len() > 1 && n_post > 1 {
        let within = h2_vars
            .iter()
            .map(|value| value.max(0.0) * n_post as f64 / (n_post - 1) as f64)
            .sum::<f64>()
            / h2_vars.len() as f64;
        if within > 0.0 && within.is_finite() {
            let between = h2_means
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f64>()
                / (h2_means.len() - 1) as f64
                * n_post as f64;
            let variance_hat =
                ((n_post - 1) as f64 / n_post as f64) * within + between / n_post as f64;
            if variance_hat.is_finite() && variance_hat >= 0.0 {
                let pooled = (variance_hat / within).sqrt().max(1.0);
                rhat = if rhat.is_finite() {
                    rhat.max(pooled)
                } else {
                    pooled
                };
            }
        }
    }
    (mean, var.max(0.0), rhat)
}

#[derive(Debug)]
struct BayesRhatState {
    names: &'static [&'static str],
    window: usize,
    history: Vec<VecDeque<f64>>,
    sum: Vec<f64>,
    sq_sum: Vec<f64>,
    invalid_count: Vec<usize>,
    valid: Vec<bool>,
    observed: usize,
    stable_checks: usize,
}

impl BayesRhatState {
    fn with_window(names: &'static [&'static str], window: usize) -> Self {
        Self {
            names,
            window: window.max(1),
            history: names
                .iter()
                .map(|_| VecDeque::with_capacity(window.max(1)))
                .collect(),
            sum: vec![0.0; names.len()],
            sq_sum: vec![0.0; names.len()],
            invalid_count: vec![0; names.len()],
            valid: vec![true; names.len()],
            observed: 0,
            stable_checks: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        for values in &mut self.history {
            values.clear();
        }
        self.sum.fill(0.0);
        self.sq_sum.fill(0.0);
        self.invalid_count.fill(0);
        self.valid.fill(true);
        self.observed = 0;
        self.stable_checks = 0;
    }

    #[inline]
    fn observe(&mut self, values: &[f64]) -> bool {
        if values.len() != self.names.len() {
            self.valid.fill(false);
            return false;
        }
        for (index, value) in values.iter().copied().enumerate() {
            let old = if self.history[index].len() >= self.window {
                self.history[index].pop_front()
            } else {
                None
            };
            if let Some(old) = old {
                if old.is_finite() {
                    self.sum[index] -= old;
                    self.sq_sum[index] -= old * old;
                } else {
                    self.invalid_count[index] = self.invalid_count[index].saturating_sub(1);
                }
            }
            if value.is_finite() {
                self.sum[index] += value;
                self.sq_sum[index] += value * value;
            } else {
                self.invalid_count[index] += 1;
            }
            self.history[index].push_back(value);
        }
        self.observed = self.observed.saturating_add(1);
        let n = self.history.first().map(VecDeque::len).unwrap_or(0);
        if n < BAYES_RHAT_MIN_KEEP || self.observed % BAYES_RHAT_CHECK_EVERY != 0 {
            return false;
        }
        let rhat = self.rhat_max();
        if rhat.is_finite() && rhat < BAYES_RHAT_THRESHOLD {
            self.stable_checks += 1;
        } else {
            self.stable_checks = 0;
        }
        self.stable_checks >= BAYES_RHAT_STABLE_CHECKS
    }

    #[inline]
    fn value(&self) -> f64 {
        self.values().first().copied().unwrap_or(f64::NAN)
    }

    #[inline]
    fn values(&self) -> Vec<f64> {
        self.history
            .iter()
            .zip(self.sum.iter())
            .zip(self.sq_sum.iter())
            .zip(self.invalid_count.iter())
            .zip(self.valid.iter())
            .map(|((((history, sum), sq_sum), invalid), valid)| {
                if *valid && *invalid == 0 {
                    split_rhat_from_window(history, *sum, *sq_sum)
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    #[inline]
    fn rhat_max(&self) -> f64 {
        finite_rhat_max(&self.values())
    }
}

/// Controls burn-in, R-hat monitoring, and posterior collection. `None` for
/// `burnin` selects adaptive mode with `BAYES_AUTO_BURNIN` warm-up updates;
/// `Some(n)` selects fixed mode with exactly `n` warm-up updates and no R-hat
/// stopping. The first warm-up updates are excluded from both R-hat and the
/// posterior. In adaptive mode, `n_iter` is the monitoring upper bound, but it
/// is extended when necessary to finish warm-up and reach a retained
/// iteration. Once R-hat stabilizes, the pre-trigger summaries are discarded
/// and exactly `posterior_target` subsequent samples are retained. If
/// monitoring reaches its upper bound without convergence, the same posterior
/// collection phase starts as a fallback. An adaptively triggered posterior
/// window is validated once; failed validation gets one bounded retry, and a
/// second failure clears `convergence_iteration` while retaining the samples
/// as a fallback result.
#[derive(Debug)]
pub(crate) struct BayesSamplingController {
    monitor_end: usize,
    burnin: usize,
    adaptive: bool,
    thin: usize,
    posterior_target: usize,
    actual_iterations: usize,
    posterior_samples: usize,
    monitor_rhat_state: BayesRhatState,
    posterior_rhat_state: BayesRhatState,
    convergence_iteration: usize,
    collecting_posterior: bool,
    posterior_validation_restarts: usize,
}

impl BayesSamplingController {
    pub(crate) fn new(
        n_iter: usize,
        burnin: Option<usize>,
        thin: usize,
        rhat_names: &'static [&'static str],
    ) -> Self {
        Self::new_with_options(
            n_iter,
            burnin,
            false,
            thin,
            rhat_names,
            BAYES_POSTERIOR_SAMPLES,
        )
    }

    pub(crate) fn new_with_options(
        n_iter: usize,
        burnin: Option<usize>,
        adaptive_burnin: bool,
        thin: usize,
        rhat_names: &'static [&'static str],
        posterior_target: usize,
    ) -> Self {
        let (burnin, adaptive) = resolve_burnin_mode(burnin, adaptive_burnin);
        let thin = thin.max(1);
        let posterior_target = posterior_target.max(1);
        Self {
            monitor_end: monitor_end_iteration(n_iter, burnin, thin, adaptive),
            burnin,
            adaptive,
            thin,
            posterior_target,
            actual_iterations: 0,
            posterior_samples: 0,
            monitor_rhat_state: BayesRhatState::with_window(rhat_names, BAYES_RHAT_WINDOW),
            posterior_rhat_state: BayesRhatState::with_window(rhat_names, posterior_target),
            convergence_iteration: 0,
            collecting_posterior: false,
            posterior_validation_restarts: 0,
        }
    }

    #[inline]
    pub(crate) fn should_run(&self) -> bool {
        if self.collecting_posterior {
            self.posterior_samples < self.posterior_target
        } else {
            self.actual_iterations < self.monitor_end
        }
    }

    /// Advance one MCMC iteration and report whether its summary is retained.
    #[inline]
    pub(crate) fn begin_iteration(&mut self) -> bool {
        self.actual_iterations += 1;
        let it = self.actual_iterations - 1;
        let retained = it >= self.burnin && ((it - self.burnin) % self.thin == 0);
        if retained && !self.adaptive && !self.collecting_posterior {
            self.collecting_posterior = true;
            self.convergence_iteration = 0;
        }
        retained
    }

    /// Observe a retained h2 sample. Returns true when the monitoring summary
    /// must be cleared before the next iteration starts formal posterior
    /// collection.
    #[inline]
    pub(crate) fn observe(&mut self, values: &[f64]) -> bool {
        if self.collecting_posterior {
            self.posterior_samples += 1;
            if self.adaptive {
                let _ = self.posterior_rhat_state.observe(values);
                if self.posterior_samples >= self.posterior_target && self.convergence_iteration > 0
                {
                    let posterior_rhat = self.posterior_rhat_state.rhat_max();
                    if !posterior_rhat.is_finite() || posterior_rhat >= BAYES_RHAT_THRESHOLD {
                        if self.posterior_validation_restarts < BAYES_MAX_POSTERIOR_RESTARTS {
                            self.posterior_validation_restarts += 1;
                            self.collecting_posterior = false;
                            self.convergence_iteration = 0;
                            self.posterior_samples = 0;
                            self.monitor_rhat_state.reset();
                            self.posterior_rhat_state.reset();
                            let retained_updates = BAYES_RHAT_WINDOW
                                .saturating_add(
                                    BAYES_RHAT_CHECK_EVERY.saturating_mul(BAYES_RHAT_STABLE_CHECKS),
                                )
                                .saturating_mul(self.thin);
                            self.monitor_end = self
                                .monitor_end
                                .max(self.actual_iterations.saturating_add(retained_updates));
                            return true;
                        }
                        // Do not report an adaptive convergence that failed
                        // its final posterior-window validation. The samples
                        // remain available as a fallback result.
                        self.convergence_iteration = 0;
                    }
                }
            }
            return false;
        }

        if self.monitor_rhat_state.observe(values) {
            self.collecting_posterior = true;
            self.convergence_iteration = self.actual_iterations;
            self.posterior_samples = 0;
            self.posterior_rhat_state.reset();
            return true;
        }

        // No convergence by the monitoring upper bound: use the following
        // 1000 iterations as the fallback posterior window. There is no
        // convergence iteration to report in this branch.
        if self.actual_iterations >= self.monitor_end {
            self.collecting_posterior = true;
            self.convergence_iteration = 0;
            self.posterior_samples = 0;
            self.posterior_rhat_state.reset();
            return true;
        }

        false
    }

    #[inline]
    pub(crate) fn posterior_samples(&self) -> usize {
        self.posterior_samples
    }

    #[inline]
    pub(crate) fn posterior_target(&self) -> usize {
        self.posterior_target
    }

    #[inline]
    pub(crate) fn rhat_value(&self) -> f64 {
        if self.collecting_posterior {
            self.posterior_rhat_state.value()
        } else {
            self.monitor_rhat_state.value()
        }
    }

    #[inline]
    pub(crate) fn rhat_values(&self) -> Vec<f64> {
        if self.collecting_posterior {
            self.posterior_rhat_state.values()
        } else {
            self.monitor_rhat_state.values()
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rhat_max(&self) -> f64 {
        if self.collecting_posterior {
            self.posterior_rhat_state.rhat_max()
        } else {
            self.monitor_rhat_state.rhat_max()
        }
    }

    #[inline]
    pub(crate) fn actual_iterations(&self) -> usize {
        self.actual_iterations
    }

    #[inline]
    pub(crate) fn convergence_iteration(&self) -> usize {
        self.convergence_iteration
    }
}

/// Synchronized controller used by the native multi-chain kernels. Every
/// chain contributes report metrics and a predictive monitor vector at the
/// same retained iteration. Adaptive convergence is decided from the maximum
/// between-chain Gelman--Rubin statistic of the predictive vector; report
/// metrics are retained separately for diagnostics. `None` for `burnin`
/// enables this adaptive decision; `Some(n)` skips it and starts the fixed
/// 1000-sample posterior window immediately after the warm-up. Marker blocks
/// and all model state remain in the caller's chain states. A triggered
/// posterior window is validated once and retried at most once before it is
/// reported as fallback.
#[derive(Debug)]
pub(crate) struct BayesMultiChainController {
    rhat_names: &'static [&'static str],
    monitor_names: &'static [&'static str],
    monitor_end: usize,
    burnin: usize,
    adaptive: bool,
    thin: usize,
    posterior_target: usize,
    chains: usize,
    actual_iterations: usize,
    posterior_samples: usize,
    convergence_iteration: usize,
    collecting_posterior: bool,
    stable_checks: usize,
    monitor_n: usize,
    monitor_history: Vec<Vec<VecDeque<f64>>>,
    monitor_valid: Vec<bool>,
    monitor_invalid_count: Vec<Vec<usize>>,
    monitor_sum: Vec<Vec<f64>>,
    monitor_sq_sum: Vec<Vec<f64>>,
    posterior_valid: Vec<bool>,
    posterior_sum: Vec<Vec<f64>>,
    posterior_sq_sum: Vec<Vec<f64>>,
    posterior_monitor_valid: Vec<bool>,
    posterior_monitor_sum: Vec<Vec<f64>>,
    posterior_monitor_sq_sum: Vec<Vec<f64>>,
    posterior_validation_restarts: usize,
}

impl BayesMultiChainController {
    pub(crate) fn new(
        chains: usize,
        n_iter: usize,
        burnin: Option<usize>,
        thin: usize,
        rhat_names: &'static [&'static str],
    ) -> Result<Self, String> {
        Self::new_with_monitor(chains, n_iter, burnin, thin, rhat_names, rhat_names)
    }

    pub(crate) fn new_with_monitor(
        chains: usize,
        n_iter: usize,
        burnin: Option<usize>,
        thin: usize,
        rhat_names: &'static [&'static str],
        monitor_names: &'static [&'static str],
    ) -> Result<Self, String> {
        Self::new_with_monitor_options(
            chains,
            n_iter,
            burnin,
            false,
            thin,
            rhat_names,
            monitor_names,
            BAYES_POSTERIOR_SAMPLES,
        )
    }

    pub(crate) fn new_with_monitor_options(
        chains: usize,
        n_iter: usize,
        burnin: Option<usize>,
        adaptive_burnin: bool,
        thin: usize,
        rhat_names: &'static [&'static str],
        monitor_names: &'static [&'static str],
        posterior_target: usize,
    ) -> Result<Self, String> {
        if chains == 0 {
            return Err("native multi-chain controller requires at least one chain".to_string());
        }
        let (burnin, adaptive) = resolve_burnin_mode(burnin, adaptive_burnin);
        let posterior_target = posterior_target.max(1);
        Ok(Self {
            rhat_names,
            monitor_names,
            monitor_end: monitor_end_iteration(n_iter, burnin, thin.max(1), adaptive),
            burnin,
            adaptive,
            thin: thin.max(1),
            posterior_target,
            chains,
            actual_iterations: 0,
            posterior_samples: 0,
            convergence_iteration: 0,
            collecting_posterior: false,
            stable_checks: 0,
            monitor_n: 0,
            monitor_history: monitor_names
                .iter()
                .map(|_| {
                    (0..chains)
                        .map(|_| VecDeque::with_capacity(BAYES_RHAT_WINDOW))
                        .collect()
                })
                .collect(),
            monitor_valid: vec![true; monitor_names.len()],
            monitor_invalid_count: vec![vec![0; chains]; monitor_names.len()],
            monitor_sum: vec![vec![0.0; chains]; monitor_names.len()],
            monitor_sq_sum: vec![vec![0.0; chains]; monitor_names.len()],
            posterior_valid: vec![true; rhat_names.len()],
            posterior_sum: vec![vec![0.0; chains]; rhat_names.len()],
            posterior_sq_sum: vec![vec![0.0; chains]; rhat_names.len()],
            posterior_monitor_valid: vec![true; monitor_names.len()],
            posterior_monitor_sum: vec![vec![0.0; chains]; monitor_names.len()],
            posterior_monitor_sq_sum: vec![vec![0.0; chains]; monitor_names.len()],
            posterior_validation_restarts: 0,
        })
    }

    #[inline]
    pub(crate) fn should_run(&self) -> bool {
        if self.collecting_posterior {
            self.posterior_samples < self.posterior_target
        } else {
            self.actual_iterations < self.monitor_end
        }
    }

    #[inline]
    pub(crate) fn begin_iteration(&mut self) -> bool {
        self.actual_iterations += 1;
        let it = self.actual_iterations - 1;
        let retained = it >= self.burnin && ((it - self.burnin) % self.thin == 0);
        if retained && !self.adaptive && !self.collecting_posterior {
            self.start_posterior(false);
        }
        retained
    }

    #[inline]
    pub(crate) fn collecting_posterior(&self) -> bool {
        self.collecting_posterior
    }

    fn reset_accumulators(&mut self) {
        for values in &mut self.posterior_sum {
            values.fill(0.0);
        }
        for values in &mut self.posterior_sq_sum {
            values.fill(0.0);
        }
        self.posterior_valid.fill(true);
        self.posterior_monitor_valid.fill(true);
        for values in &mut self.posterior_monitor_sum {
            values.fill(0.0);
        }
        for values in &mut self.posterior_monitor_sq_sum {
            values.fill(0.0);
        }
        self.posterior_samples = 0;
    }

    fn reset_monitor_state(&mut self) {
        for values in &mut self.monitor_history {
            for history in values {
                history.clear();
            }
        }
        self.monitor_valid.fill(true);
        for values in &mut self.monitor_invalid_count {
            values.fill(0);
        }
        for values in &mut self.monitor_sum {
            values.fill(0.0);
        }
        for values in &mut self.monitor_sq_sum {
            values.fill(0.0);
        }
        self.monitor_n = 0;
        self.stable_checks = 0;
    }

    fn start_posterior(&mut self, converged: bool) {
        self.collecting_posterior = true;
        self.convergence_iteration = if converged { self.actual_iterations } else { 0 };
        self.reset_accumulators();
    }

    fn posterior_monitor_rhat(&self) -> f64 {
        if self.posterior_samples < 2 {
            return f64::NAN;
        }
        let values = self
            .posterior_monitor_sum
            .iter()
            .zip(self.posterior_monitor_sq_sum.iter())
            .zip(self.posterior_monitor_valid.iter())
            .map(|((sum, sq_sum), valid)| {
                if *valid {
                    multi_chain_rhat_from_moments(sum, sq_sum, self.posterior_samples)
                } else {
                    f64::NAN
                }
            })
            .collect::<Vec<_>>();
        finite_rhat_max(&values)
    }

    fn restart_adaptive_monitoring(&mut self) {
        self.collecting_posterior = false;
        self.convergence_iteration = 0;
        self.reset_accumulators();
        self.reset_monitor_state();
        let retained_updates = BAYES_RHAT_WINDOW
            .saturating_add(BAYES_RHAT_CHECK_EVERY.saturating_mul(BAYES_RHAT_STABLE_CHECKS))
            .saturating_mul(self.thin);
        self.monitor_end = self
            .monitor_end
            .max(self.actual_iterations.saturating_add(retained_updates));
    }

    fn monitor_rhat(&self) -> f64 {
        let window_n = self
            .monitor_history
            .first()
            .and_then(|chains| chains.first())
            .map(VecDeque::len)
            .unwrap_or(0);
        if window_n < BAYES_RHAT_MIN_KEEP {
            return f64::NAN;
        }
        let values = self
            .monitor_sum
            .iter()
            .zip(self.monitor_sq_sum.iter())
            .zip(self.monitor_valid.iter())
            .zip(self.monitor_invalid_count.iter())
            .map(|(((sum, sq_sum), valid), invalid_count)| {
                if *valid && invalid_count.iter().all(|count| *count == 0) {
                    multi_chain_rhat_from_moments(sum, sq_sum, window_n)
                } else {
                    f64::NAN
                }
            })
            .collect::<Vec<_>>();
        finite_rhat_max(&values)
    }

    /// Observe one synchronized retained sample. Returns true when model
    /// accumulators must be cleared before the next iteration.
    pub(crate) fn observe(&mut self, values: &[Vec<f64>], retained: bool) -> Result<bool, String> {
        self.observe_with_monitor(values, values, retained)
    }

    pub(crate) fn observe_with_monitor(
        &mut self,
        values: &[Vec<f64>],
        monitor_values: &[Vec<f64>],
        retained: bool,
    ) -> Result<bool, String> {
        if !retained {
            return Ok(false);
        }
        if values.len() != self.chains
            || values
                .iter()
                .any(|value| value.len() != self.rhat_names.len())
        {
            return Err("native multi-chain R-hat metric dimensions mismatch".to_string());
        }
        if monitor_values.len() != self.chains
            || monitor_values
                .iter()
                .any(|value| value.len() != self.monitor_names.len())
        {
            return Err("native multi-chain predictive monitor dimensions mismatch".to_string());
        }
        if self.collecting_posterior {
            for (chain, chain_values) in values.iter().enumerate() {
                for (metric, value) in chain_values.iter().copied().enumerate() {
                    if value.is_finite() {
                        self.posterior_sum[metric][chain] += value;
                        self.posterior_sq_sum[metric][chain] += value * value;
                    } else {
                        self.posterior_valid[metric] = false;
                    }
                }
            }
            for (chain, chain_values) in monitor_values.iter().enumerate() {
                for (metric, value) in chain_values.iter().copied().enumerate() {
                    if value.is_finite() {
                        self.posterior_monitor_sum[metric][chain] += value;
                        self.posterior_monitor_sq_sum[metric][chain] += value * value;
                    } else {
                        self.posterior_monitor_valid[metric] = false;
                    }
                }
            }
            self.posterior_samples += 1;
            if self.adaptive
                && self.posterior_samples >= self.posterior_target
                && self.convergence_iteration > 0
            {
                let predictive_rhat = self.posterior_monitor_rhat();
                if !predictive_rhat.is_finite() || predictive_rhat >= BAYES_RHAT_THRESHOLD {
                    if self.posterior_validation_restarts < BAYES_MAX_POSTERIOR_RESTARTS {
                        self.posterior_validation_restarts += 1;
                        self.restart_adaptive_monitoring();
                        return Ok(true);
                    }
                    // A second failed validation is still a failed adaptive
                    // convergence attempt. Keep the posterior window for a
                    // usable fallback result, but never report that it
                    // converged.
                    self.convergence_iteration = 0;
                }
            }
            return Ok(false);
        }
        for (chain, chain_values) in monitor_values.iter().enumerate() {
            for (metric, value) in chain_values.iter().copied().enumerate() {
                let history = &mut self.monitor_history[metric][chain];
                if history.len() >= BAYES_RHAT_WINDOW {
                    if let Some(old) = history.pop_front() {
                        if old.is_finite() {
                            self.monitor_sum[metric][chain] -= old;
                            self.monitor_sq_sum[metric][chain] -= old * old;
                        } else {
                            self.monitor_invalid_count[metric][chain] =
                                self.monitor_invalid_count[metric][chain].saturating_sub(1);
                        }
                    }
                }
                if value.is_finite() {
                    self.monitor_sum[metric][chain] += value;
                    self.monitor_sq_sum[metric][chain] += value * value;
                } else {
                    self.monitor_invalid_count[metric][chain] += 1;
                }
                history.push_back(value);
            }
        }
        self.monitor_n += 1;
        if self.monitor_n >= BAYES_RHAT_MIN_KEEP && self.monitor_n % BAYES_RHAT_CHECK_EVERY == 0 {
            if self.monitor_rhat().is_finite() && self.monitor_rhat() < BAYES_RHAT_THRESHOLD {
                self.stable_checks += 1;
            } else {
                self.stable_checks = 0;
            }
            if self.stable_checks >= BAYES_RHAT_STABLE_CHECKS {
                self.start_posterior(true);
                return Ok(true);
            }
        }
        if self.actual_iterations >= self.monitor_end {
            self.start_posterior(false);
            return Ok(true);
        }
        Ok(false)
    }

    #[inline]
    pub(crate) fn posterior_samples(&self) -> usize {
        self.posterior_samples
    }

    #[inline]
    pub(crate) fn posterior_target(&self) -> usize {
        self.posterior_target
    }

    pub(crate) fn posterior_h2_summary(&self) -> (f64, f64, f64) {
        self.posterior_metric_summary(0)
    }

    fn posterior_metric_summary(&self, metric: usize) -> (f64, f64, f64) {
        if metric >= self.rhat_names.len() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let n = self.posterior_samples.max(1) as f64;
        let means = self.posterior_sum[metric]
            .iter()
            .map(|value| *value / n)
            .collect::<Vec<_>>();
        let vars = self.posterior_sum[metric]
            .iter()
            .zip(self.posterior_sq_sum[metric].iter())
            .map(|(sum, sq_sum)| (sq_sum / n - (sum / n) * (sum / n)).max(0.0))
            .collect::<Vec<_>>();
        if !self.posterior_valid[metric] {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let rhats = vec![f64::NAN; self.chains];
        aggregate_bayes_h2(&means, &vars, &rhats, self.posterior_samples)
    }

    pub(crate) fn posterior_rhat_values(&self) -> Vec<f64> {
        if !self.adaptive {
            return vec![f64::NAN; self.rhat_names.len()];
        }
        self.posterior_sum
            .iter()
            .zip(self.posterior_sq_sum.iter())
            .zip(self.posterior_valid.iter())
            .map(|((sum, sq_sum), valid)| {
                if *valid {
                    multi_chain_rhat_from_moments(sum, sq_sum, self.posterior_samples)
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    #[inline]
    pub(crate) fn actual_iterations(&self) -> usize {
        self.actual_iterations
    }

    #[inline]
    pub(crate) fn convergence_iteration(&self) -> usize {
        self.convergence_iteration
    }
}

fn multi_chain_rhat_from_moments(sum: &[f64], sq_sum: &[f64], n: usize) -> f64 {
    if sum.len() != sq_sum.len() || sum.is_empty() || n < 2 {
        return f64::NAN;
    }
    let n_f = n as f64;
    let means = sum.iter().map(|value| *value / n_f).collect::<Vec<_>>();
    let within = sum
        .iter()
        .zip(sq_sum.iter())
        .map(|(sum, sq_sum)| ((sq_sum - sum * sum / n_f) / (n_f - 1.0)).max(0.0))
        .sum::<f64>()
        / sum.len() as f64;
    let mean = means.iter().sum::<f64>() / means.len() as f64;
    let between = n_f
        * means
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
        / (means.len().saturating_sub(1).max(1) as f64);
    if !within.is_finite() || !between.is_finite() {
        return f64::NAN;
    }
    if within <= f64::MIN_POSITIVE {
        return if between <= f64::MIN_POSITIVE {
            1.0
        } else {
            f64::INFINITY
        };
    }
    let variance_hat = ((n - 1) as f64 / n_f) * within + between / n_f;
    if variance_hat.is_finite() && variance_hat >= 0.0 {
        (variance_hat / within).sqrt().max(1.0)
    } else {
        f64::NAN
    }
}

fn split_rhat_from_window(history: &VecDeque<f64>, total_sum: f64, total_sq_sum: f64) -> f64 {
    let n = history.len();
    if n < 4 {
        return f64::NAN;
    }
    // Use equal halves; an odd final sample is intentionally ignored.
    let chain_n = n / 2;
    if chain_n < 2 {
        return f64::NAN;
    }
    let left_sum = history.iter().take(chain_n).sum::<f64>();
    let left_sq_sum = history
        .iter()
        .take(chain_n)
        .map(|value| value * value)
        .sum::<f64>();
    let split_end = chain_n * 2;
    let split_sum = if split_end == n {
        total_sum
    } else {
        history.iter().take(split_end).sum::<f64>()
    };
    let split_sq_sum = if split_end == n {
        total_sq_sum
    } else {
        history
            .iter()
            .take(split_end)
            .map(|value| value * value)
            .sum::<f64>()
    };
    let right_sum = split_sum - left_sum;
    let right_sq_sum = split_sq_sum - left_sq_sum;
    let mean_left = left_sum / chain_n as f64;
    let mean_right = right_sum / chain_n as f64;
    let var_left =
        ((left_sq_sum - left_sum * left_sum / chain_n as f64) / (chain_n - 1) as f64).max(0.0);
    let var_right =
        ((right_sq_sum - right_sum * right_sum / chain_n as f64) / (chain_n - 1) as f64).max(0.0);
    let within = 0.5 * (var_left + var_right);
    let between = chain_n as f64 * (mean_left - mean_right) * (mean_left - mean_right) / 2.0;
    let var_hat = ((chain_n - 1) as f64 / chain_n as f64) * within + between / chain_n as f64;
    if !within.is_finite() || !var_hat.is_finite() {
        return f64::NAN;
    }
    if within <= f64::MIN_POSITIVE {
        return if var_hat <= f64::MIN_POSITIVE {
            1.0
        } else {
            f64::INFINITY
        };
    }
    (var_hat / within).sqrt().max(1.0)
}

fn fill_beta_trace_row(
    beta_trace: &mut [f64],
    row_idx: usize,
    n_trace_snps: usize,
    trace_snp_indices: &[usize],
    beta: &[f64],
    d: Option<&[u8]>,
) {
    if n_trace_snps == 0 {
        return;
    }
    let row_off = row_idx * n_trace_snps;
    for (k, &j) in trace_snp_indices.iter().enumerate() {
        let val = if let Some(mask) = d {
            (mask[j] as f64) * beta[j]
        } else {
            beta[j]
        };
        beta_trace[row_off + k] = val;
    }
}

fn packed_trace_result_to_pydict<'py>(
    py: Python<'py>,
    res: PackedBayesTraceResult,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py).into_bound();
    let beta_py = res.beta.into_pyarray(py);
    let alpha_py = res.alpha.into_pyarray(py);
    let iter_trace_py = res.iter_trace.into_pyarray(py);
    let h2_trace_py = res.h2_trace.into_pyarray(py);
    let var_e_trace_py = res.var_e_trace.into_pyarray(py);
    let prob_in_trace_py = res.prob_in_trace.into_pyarray(py);
    let n_active_trace_py = res.n_active_trace.into_pyarray(py);
    let full_iter_trace_py = res.full_iter_trace.into_pyarray(py);
    let full_h2_trace_py = res.full_h2_trace.into_pyarray(py);
    let full_var_e_trace_py = res.full_var_e_trace.into_pyarray(py);
    let full_prob_in_trace_py = res.full_prob_in_trace.into_pyarray(py);
    let full_n_active_trace_py = res.full_n_active_trace.into_pyarray(py);
    let beta_trace_indices_py = res.beta_trace_indices.into_pyarray(py);
    let beta_trace_arr = Array2::from_shape_vec(
        (iter_trace_py.len(), beta_trace_indices_py.len()),
        res.beta_trace,
    )
    .map_err(|e| PyValueError::new_err(format!("invalid beta trace shape: {e}")))?;
    let beta_trace_py = PyArray2::from_owned_array(py, beta_trace_arr);
    out.set_item("beta", beta_py)?;
    out.set_item("alpha", alpha_py)?;
    out.set_item("vare", res.vare)?;
    out.set_item("h2_mean", res.h2_mean)?;
    out.set_item("var_h2", res.var_h2)?;
    out.set_item("prob_in_mean", res.prob_in_mean)?;
    out.set_item("n_active_mean", res.n_active_mean)?;
    out.set_item("iter_trace", iter_trace_py)?;
    out.set_item("h2_trace", h2_trace_py)?;
    out.set_item("var_e_trace", var_e_trace_py)?;
    out.set_item("prob_in_trace", prob_in_trace_py)?;
    out.set_item("n_active_trace", n_active_trace_py)?;
    out.set_item("full_iter_trace", full_iter_trace_py)?;
    out.set_item("full_h2_trace", full_h2_trace_py)?;
    out.set_item("full_var_e_trace", full_var_e_trace_py)?;
    out.set_item("full_prob_in_trace", full_prob_in_trace_py)?;
    out.set_item("full_n_active_trace", full_n_active_trace_py)?;
    out.set_item("beta_trace_indices", beta_trace_indices_py)?;
    out.set_item("beta_trace", beta_trace_py)?;
    Ok(out)
}

#[inline]
fn row_major_block_mul_vec_f64(
    block: &[f64],
    rows: usize,
    cols: usize,
    vec: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            rows as CblasInt,
            1 as CblasInt,
            cols as CblasInt,
            1.0_f64,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            cols as CblasInt,
            0.0_f64,
            out.as_mut_ptr(),
            rows as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for r in 0..rows {
            let row = &block[r * cols..(r + 1) * cols];
            let mut acc = 0.0_f64;
            for c in 0..cols {
                acc += row[c] * vec[c];
            }
            out[r] = acc;
        }
    }
}

#[inline]
fn row_major_block_t_mul_vec_f64(
    block: &[f64],
    rows: usize,
    cols: usize,
    vec: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            cols as CblasInt,
            1 as CblasInt,
            rows as CblasInt,
            1.0_f64,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            rows as CblasInt,
            0.0_f64,
            out.as_mut_ptr(),
            cols as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for c in 0..cols {
            let mut acc = 0.0_f64;
            for r in 0..rows {
                acc += block[r * cols + c] * vec[r];
            }
            out[c] = acc;
        }
    }
}

#[inline]
fn row_major_xtx_f64(x: &[f64], n: usize, q: usize, xtx_out: &mut [f64]) {
    debug_assert_eq!(x.len(), n.saturating_mul(q));
    debug_assert_eq!(xtx_out.len(), q.saturating_mul(q));
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            q as CblasInt,
            q as CblasInt,
            n as CblasInt,
            1.0_f64,
            x.as_ptr(),
            q as CblasInt,
            x.as_ptr(),
            q as CblasInt,
            0.0_f64,
            xtx_out.as_mut_ptr(),
            q as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for i in 0..q {
            for j in 0..q {
                let mut acc = 0.0_f64;
                for r in 0..n {
                    acc += x[r * q + i] * x[r * q + j];
                }
                xtx_out[i * q + j] = acc;
            }
        }
    }
}

/// Shared fixed-effect setup for all Bayesian marker samplers.
///
/// The marker sampler only needs the small `q x q` cross-product for its
/// conditional updates.  Initialization is deliberately solved from the
/// original `n x q` design with QR, and falls back to an SVD pseudo-inverse
/// when the QR factor is rank deficient.  No `n x n` matrix is constructed.
pub(crate) struct BayesFixedEffectBackend<'a> {
    x: &'a [f64],
    n: usize,
    q: usize,
    x2_x: Vec<f64>,
    xtx: Vec<f64>,
}

pub(crate) struct BayesFixedEffectInit {
    pub(crate) alpha: Vec<f64>,
    pub(crate) residual: Vec<f64>,
}

/// Data-scaled variance-prior calibration shared by every Bayes backend.
///
/// `residualized_var` is computed from the fixed-effect residual
/// `y - X alpha`, so automatic prior scales do not use variation already
/// explained by covariates. The calibration is immutable after construction;
/// MCMC updates only the sampled variance states.
#[derive(Clone, Copy)]
pub(crate) struct BayesPriorCalibration {
    pub(crate) residualized_var: f64,
    pub(crate) initial_var_e: f64,
    pub(crate) prior_ss_e: f64,
}

impl BayesPriorCalibration {
    pub(crate) fn from_fixed_init(
        fixed_init: &BayesFixedEffectInit,
        r2: f64,
        df0_e: f64,
        prior_ss_e_opt: Option<f64>,
    ) -> Result<Self, String> {
        if !(r2.is_finite() && r2 > 0.0 && r2 < 1.0) {
            return Err("r2 must be finite and in (0, 1)".to_string());
        }
        if !(df0_e.is_finite() && df0_e > 0.0) {
            return Err("df0_e must be finite and > 0".to_string());
        }
        let residualized_var = phenotype_variance(&fixed_init.residual)?;
        let initial_var_e = residualized_var * (1.0 - r2);
        if !(initial_var_e.is_finite() && initial_var_e > 0.0) {
            return Err(
                "varE must be positive; check r2 and residualized phenotype variance".to_string(),
            );
        }
        let prior_ss_e = prior_ss_e_opt.unwrap_or(initial_var_e * (df0_e + 2.0));
        if !(prior_ss_e.is_finite() && prior_ss_e > 0.0) {
            return Err("prior_ss_e must be finite and > 0".to_string());
        }
        Ok(Self {
            residualized_var,
            initial_var_e,
            prior_ss_e,
        })
    }

    pub(crate) fn marker_scale(
        &self,
        msx: f64,
        r2: f64,
        df0_b: f64,
        inclusion_probability: Option<f64>,
        explicit_s0_b: Option<f64>,
    ) -> Result<f64, String> {
        if let Some(value) = explicit_s0_b {
            if value.is_finite() && value > 0.0 {
                return Ok(value);
            }
            return Err("S0_b must be finite and > 0".to_string());
        }
        if !(msx.is_finite() && msx > 0.0) {
            return Err("MSx must be positive to compute S0_b".to_string());
        }
        if !(r2.is_finite() && r2 > 0.0 && r2 < 1.0) {
            return Err("r2 must be finite and in (0, 1)".to_string());
        }
        if !(df0_b.is_finite() && df0_b > 0.0) {
            return Err("df0_b must be finite and > 0".to_string());
        }
        let mut scale = self.residualized_var * r2 / msx * (df0_b + 2.0);
        if let Some(probability) = inclusion_probability {
            if !(probability.is_finite() && probability > 0.0 && probability < 1.0) {
                return Err("prob_in must be finite and in (0, 1)".to_string());
            }
            scale /= probability;
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err("S0_b must be finite and > 0".to_string());
        }
        Ok(scale)
    }
}

impl<'a> BayesFixedEffectBackend<'a> {
    pub(crate) fn new(x: &'a [f64], n: usize, q: usize) -> Result<Self, String> {
        if n == 0 || q == 0 {
            return Err("Bayesian fixed-effect design must be non-empty".to_string());
        }
        if x.len() != n.saturating_mul(q) {
            return Err("Bayesian fixed-effect design dimensions are incompatible".to_string());
        }
        let mut xtx = vec![0.0_f64; q.saturating_mul(q)];
        row_major_xtx_f64(x, n, q, &mut xtx);
        let x2_x = (0..q).map(|k| xtx[k * q + k]).collect();
        Ok(Self { x, n, q, x2_x, xtx })
    }

    #[inline]
    pub(crate) fn x2_x(&self) -> &[f64] {
        &self.x2_x
    }

    #[inline]
    pub(crate) fn xtx(&self) -> &[f64] {
        &self.xtx
    }

    pub(crate) fn initial_state(&self, y: &[f64]) -> Result<BayesFixedEffectInit, String> {
        if y.len() != self.n {
            return Err("Bayesian fixed-effect phenotype length does not match design".to_string());
        }
        let alpha = solve_fixed_effects_qr_svd(self.x, self.n, self.q, y)
            .unwrap_or_else(|| vec![0.0_f64; self.q]);
        let mut residual = vec![0.0_f64; self.n];
        for i in 0..self.n {
            let row = &self.x[i * self.q..(i + 1) * self.q];
            let fitted = row
                .iter()
                .zip(alpha.iter())
                .map(|(value, coefficient)| value * coefficient)
                .sum::<f64>();
            residual[i] = y[i] - fitted;
        }
        Ok(BayesFixedEffectInit { alpha, residual })
    }
}

/// Solve `min_alpha ||y - X alpha||` without forming a sample-by-sample
/// covariance matrix. QR is preferred for full-rank designs; SVD supplies a
/// stable minimum-norm solution for rank-deficient or ill-conditioned designs.
fn solve_fixed_effects_qr_svd(x: &[f64], n: usize, q: usize, y: &[f64]) -> Option<Vec<f64>> {
    if n == 0 || q == 0 || x.len() != n.saturating_mul(q) || y.len() != n {
        return None;
    }
    if !x.iter().all(|value| value.is_finite()) || !y.iter().all(|value| value.is_finite()) {
        return None;
    }

    let xmat = DMatrix::<f64>::from_row_slice(n, q, x);
    let yvec = DVector::<f64>::from_column_slice(y);
    let qr = xmat.clone().qr();
    let r = qr.r();
    let mut qty = yvec;
    qr.q_tr_mul(&mut qty);
    let diag_scale = (0..r.nrows().min(r.ncols()))
        .map(|i| r[(i, i)].abs())
        .fold(0.0_f64, f64::max);
    let qr_cutoff = 1.0e-12_f64 * diag_scale;
    let qr_full_rank = n >= q
        && diag_scale.is_finite()
        && diag_scale > 0.0
        && (0..q).all(|i| r[(i, i)].abs() > qr_cutoff);
    if qr_full_rank {
        let mut solution = vec![0.0_f64; q];
        for ii in 0..q {
            let i = q - 1 - ii;
            let mut rhs = qty[i];
            for j in (i + 1)..q {
                rhs -= r[(i, j)] * solution[j];
            }
            solution[i] = rhs / r[(i, i)];
        }
        if solution.iter().all(|value| value.is_finite()) {
            return Some(solution);
        }
    }

    // Use the original design for the fallback. `nalgebra::QR::r()` stores a
    // sign-normalized triangular factor, while `q_tr_mul` follows the
    // reflector convention of the original factorization; pairing those
    // values in a second SVD can change the projection for rank-deficient
    // designs. The direct SVD remains n x q, never n x n, and is only used
    // after the cheap full-rank QR path is rejected.
    let svd = xmat.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let smax = svd
        .singular_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::max);
    if !(smax.is_finite() && smax > 0.0) {
        return None;
    }
    let cutoff = 1.0e-12_f64 * smax;
    let mut alpha = vec![0.0_f64; q];
    for (i, singular) in svd.singular_values.iter().copied().enumerate() {
        if !(singular.is_finite() && singular > cutoff) {
            continue;
        }
        let mut uy = 0.0_f64;
        for row in 0..n {
            uy += u[(row, i)] * y[row];
        }
        let coefficient = uy / singular;
        for column in 0..q {
            alpha[column] += vt[(i, column)] * coefficient;
        }
    }
    alpha.iter().all(|value| value.is_finite()).then_some(alpha)
}

#[inline]
pub(crate) fn ddot_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return 0.0_f64;
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        if x.len() <= (CblasInt::MAX as usize) {
            unsafe {
                return cblas_ddot_dispatch(
                    x.len() as CblasInt,
                    x.as_ptr(),
                    1 as CblasInt,
                    y.as_ptr(),
                    1 as CblasInt,
                );
            }
        }
    }
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (*xi) * (*yi))
        .sum::<f64>()
}

#[inline]
#[cfg(test)]
pub(crate) fn daxpy_inplace_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
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

/// Return the conditional marker score without materializing `r + x_j beta_j`.
///
/// The maintained residual is `r = y - X beta`, so it still contains the
/// current marker effect.  Adding `beta_j * d_j` recovers
/// `x_j^T (r + x_j beta_j)` with a single dot product over the samples.
#[inline]
#[cfg(test)]
pub(crate) fn marker_conditional_dot_f64(
    residual: &[f64],
    marker: &[f64],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    ddot_f64(residual, marker) + old_beta * marker_ss
}

/// Mixed-precision marker score: f32 genotype storage with an f64 residual
/// and f64 accumulation.  This is intentionally separate from the all-f64
/// helper so callers cannot accidentally downcast the residual or statistic.
#[inline]
#[cfg(test)]
pub(crate) fn marker_conditional_dot_f32_f64(
    residual: &[f64],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    debug_assert_eq!(residual.len(), marker.len());
    // ARM64 NEON converts four f32 marker values to two f64x2 vectors and
    // accumulates in f64.  This avoids a scalar cast on every product while
    // preserving the mixed-precision accumulator contract.
    let (mut dot, mut i) = {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { marker_dot_f32_f64_neon(residual, marker) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            (0.0_f64, 0usize)
        }
    };
    while i + 4 <= marker.len() {
        dot = residual[i].mul_add(marker[i] as f64, dot);
        dot = residual[i + 1].mul_add(marker[i + 1] as f64, dot);
        dot = residual[i + 2].mul_add(marker[i + 2] as f64, dot);
        dot = residual[i + 3].mul_add(marker[i + 3] as f64, dot);
        i += 4;
    }
    while i < marker.len() {
        dot = residual[i].mul_add(marker[i] as f64, dot);
        i += 1;
    }
    dot + old_beta * marker_ss
}

/// Marker score for the hot Gibbs loop when both the standardized marker and
/// maintained marker residual are f32.  Genotypes and the repeatedly streamed
/// residual stay compact, while the score is accumulated in f64 so marker
/// ordering and posterior draws remain stable under mixed precision.
#[inline]
#[cfg(test)]
pub(crate) fn marker_conditional_dot_f32_f32(
    residual: &[f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    marker_conditional_dot_f32_f32_blocked(residual, marker, old_beta, marker_ss)
}

#[inline]
#[cfg(test)]
pub(crate) fn marker_conditional_dot_f32_f32_fast(
    residual: &[f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    debug_assert_eq!(residual.len(), marker.len());
    let (mut dot, mut i) = {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { marker_dot_f32_f32_f64_neon(residual, marker) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            (0.0_f64, 0usize)
        }
    };
    while i < marker.len() {
        dot = (residual[i] as f64).mul_add(marker[i] as f64, dot);
        i += 1;
    }
    dot + old_beta * marker_ss
}

/// Compute the conditional marker score with f32 vector accumulators and a
/// f64 accumulator between SIMD blocks.  The previous kernel converted every
/// f32 lane to f64 before the FMA; that preserved more dot-product precision
/// than the marker input carries, but made the hot path conversion-bound on
/// Apple NEON.  Summing each 4-lane block in f32 and promoting only the block
/// total retains a bounded mixed-precision error while avoiding per-lane f64
/// conversion.
#[inline]
pub(crate) fn marker_conditional_dot_f32_f32_blocked(
    residual: &[f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    debug_assert_eq!(residual.len(), marker.len());
    let (mut dot, mut i) = {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { marker_dot_f32_f32_blocked_neon(residual, marker) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut acc0 = 0.0_f32;
            let mut acc1 = 0.0_f32;
            let mut acc2 = 0.0_f32;
            let mut acc3 = 0.0_f32;
            let limit = marker.len() / 4 * 4;
            let mut i = 0usize;
            while i < limit {
                acc0 += residual[i] * marker[i];
                acc1 += residual[i + 1] * marker[i + 1];
                acc2 += residual[i + 2] * marker[i + 2];
                acc3 += residual[i + 3] * marker[i + 3];
                i += 4;
            }
            (
                f64::from(acc0) + f64::from(acc1) + f64::from(acc2) + f64::from(acc3),
                i,
            )
        }
    };
    while i < marker.len() {
        dot += f64::from(residual[i] * marker[i]);
        i += 1;
    }
    dot + old_beta * marker_ss
}

#[cfg(all(test, target_arch = "aarch64"))]
#[inline]
unsafe fn marker_dot_f32_f32_f64_neon(residual: &[f32], marker: &[f32]) -> (f64, usize) {
    let limit = marker.len() / 8 * 8;
    let mut i = 0usize;
    let mut acc0_lo = vdupq_n_f64(0.0);
    let mut acc0_hi = vdupq_n_f64(0.0);
    let mut acc1_lo = vdupq_n_f64(0.0);
    let mut acc1_hi = vdupq_n_f64(0.0);
    while i < limit {
        let m0 = vld1q_f32(marker.as_ptr().add(i));
        let r0 = vld1q_f32(residual.as_ptr().add(i));
        let m1 = vld1q_f32(marker.as_ptr().add(i + 4));
        let r1 = vld1q_f32(residual.as_ptr().add(i + 4));
        acc0_lo = vfmaq_f64(
            acc0_lo,
            vcvt_f64_f32(vget_low_f32(r0)),
            vcvt_f64_f32(vget_low_f32(m0)),
        );
        acc0_hi = vfmaq_f64(
            acc0_hi,
            vcvt_f64_f32(vget_high_f32(r0)),
            vcvt_f64_f32(vget_high_f32(m0)),
        );
        acc1_lo = vfmaq_f64(
            acc1_lo,
            vcvt_f64_f32(vget_low_f32(r1)),
            vcvt_f64_f32(vget_low_f32(m1)),
        );
        acc1_hi = vfmaq_f64(
            acc1_hi,
            vcvt_f64_f32(vget_high_f32(r1)),
            vcvt_f64_f32(vget_high_f32(m1)),
        );
        i += 8;
    }
    if i + 4 <= marker.len() {
        let m = vld1q_f32(marker.as_ptr().add(i));
        let r = vld1q_f32(residual.as_ptr().add(i));
        acc0_lo = vfmaq_f64(
            acc0_lo,
            vcvt_f64_f32(vget_low_f32(r)),
            vcvt_f64_f32(vget_low_f32(m)),
        );
        acc0_hi = vfmaq_f64(
            acc0_hi,
            vcvt_f64_f32(vget_high_f32(r)),
            vcvt_f64_f32(vget_high_f32(m)),
        );
        i += 4;
    }
    let mut lanes0_lo = [0.0_f64; 2];
    let mut lanes0_hi = [0.0_f64; 2];
    let mut lanes1_lo = [0.0_f64; 2];
    let mut lanes1_hi = [0.0_f64; 2];
    vst1q_f64(lanes0_lo.as_mut_ptr(), acc0_lo);
    vst1q_f64(lanes0_hi.as_mut_ptr(), acc0_hi);
    vst1q_f64(lanes1_lo.as_mut_ptr(), acc1_lo);
    vst1q_f64(lanes1_hi.as_mut_ptr(), acc1_hi);
    (
        lanes0_lo[0]
            + lanes0_lo[1]
            + lanes0_hi[0]
            + lanes0_hi[1]
            + lanes1_lo[0]
            + lanes1_lo[1]
            + lanes1_hi[0]
            + lanes1_hi[1],
        i,
    )
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn marker_dot_f32_f32_blocked_neon(residual: &[f32], marker: &[f32]) -> (f64, usize) {
    let limit = marker.len() / 16 * 16;
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    while i < limit {
        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(residual.as_ptr().add(i)),
            vld1q_f32(marker.as_ptr().add(i)),
        );
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(residual.as_ptr().add(i + 4)),
            vld1q_f32(marker.as_ptr().add(i + 4)),
        );
        acc2 = vfmaq_f32(
            acc2,
            vld1q_f32(residual.as_ptr().add(i + 8)),
            vld1q_f32(marker.as_ptr().add(i + 8)),
        );
        acc3 = vfmaq_f32(
            acc3,
            vld1q_f32(residual.as_ptr().add(i + 12)),
            vld1q_f32(marker.as_ptr().add(i + 12)),
        );
        i += 16;
    }
    let mut lanes0 = [0.0_f32; 4];
    let mut lanes1 = [0.0_f32; 4];
    let mut lanes2 = [0.0_f32; 4];
    let mut lanes3 = [0.0_f32; 4];
    vst1q_f32(lanes0.as_mut_ptr(), acc0);
    vst1q_f32(lanes1.as_mut_ptr(), acc1);
    vst1q_f32(lanes2.as_mut_ptr(), acc2);
    vst1q_f32(lanes3.as_mut_ptr(), acc3);
    let mut dot = lanes0.iter().copied().map(f64::from).sum::<f64>()
        + lanes1.iter().copied().map(f64::from).sum::<f64>()
        + lanes2.iter().copied().map(f64::from).sum::<f64>()
        + lanes3.iter().copied().map(f64::from).sum::<f64>();

    if i + 4 <= marker.len() {
        let acc = vfmaq_f32(
            vdupq_n_f32(0.0),
            vld1q_f32(residual.as_ptr().add(i)),
            vld1q_f32(marker.as_ptr().add(i)),
        );
        let mut lanes = [0.0_f32; 4];
        vst1q_f32(lanes.as_mut_ptr(), acc);
        dot += lanes.iter().copied().map(f64::from).sum::<f64>();
        i += 4;
    }
    (dot, i)
}

#[cfg(all(test, target_arch = "aarch64"))]
#[inline]
unsafe fn marker_dot_f32_f64_neon(residual: &[f64], marker: &[f32]) -> (f64, usize) {
    let limit = marker.len() / 4 * 4;
    let mut i = 0usize;
    let mut acc_lo = vdupq_n_f64(0.0);
    let mut acc_hi = vdupq_n_f64(0.0);
    while i < limit {
        let m = vld1q_f32(marker.as_ptr().add(i));
        let m_lo = vcvt_f64_f32(vget_low_f32(m));
        let m_hi = vcvt_f64_f32(vget_high_f32(m));
        let r_lo = vld1q_f64(residual.as_ptr().add(i));
        let r_hi = vld1q_f64(residual.as_ptr().add(i + 2));
        acc_lo = vfmaq_f64(acc_lo, r_lo, m_lo);
        acc_hi = vfmaq_f64(acc_hi, r_hi, m_hi);
        i += 4;
    }
    let mut lanes_lo = [0.0_f64; 2];
    let mut lanes_hi = [0.0_f64; 2];
    vst1q_f64(lanes_lo.as_mut_ptr(), acc_lo);
    vst1q_f64(lanes_hi.as_mut_ptr(), acc_hi);
    (lanes_lo[0] + lanes_lo[1] + lanes_hi[0] + lanes_hi[1], i)
}

/// Update a maintained residual after replacing one marker effect.
///
/// Since `r = y - X beta`, changing `beta_j` from `old_beta` to `new_beta`
/// requires `r += x_j * (old_beta - new_beta)`.  Keeping this as one BLAS
/// AXPY avoids separate remove/add scans of the sample vector.
#[inline]
#[cfg(test)]
pub(crate) fn update_marker_residual_f64(
    old_beta: f64,
    new_beta: f64,
    marker: &[f64],
    residual: &mut [f64],
) {
    daxpy_inplace_f64(old_beta - new_beta, marker, residual);
}

/// Mixed-precision residual update.  The residual remains f64 for numerical
/// stability while the marker is read from the f32 backend buffer.
#[inline]
#[cfg(test)]
pub(crate) fn update_marker_residual_f32_f64(
    old_beta: f64,
    new_beta: f64,
    marker: &[f32],
    residual: &mut [f64],
) {
    debug_assert_eq!(marker.len(), residual.len());
    let alpha = old_beta - new_beta;
    let mut i = {
        #[cfg(all(test, target_arch = "aarch64"))]
        {
            unsafe { update_marker_residual_f32_f64_neon(alpha, marker, residual) }
        }
        #[cfg(not(all(test, target_arch = "aarch64")))]
        {
            0usize
        }
    };
    while i + 4 <= marker.len() {
        residual[i] = alpha.mul_add(marker[i] as f64, residual[i]);
        residual[i + 1] = alpha.mul_add(marker[i + 1] as f64, residual[i + 1]);
        residual[i + 2] = alpha.mul_add(marker[i + 2] as f64, residual[i + 2]);
        residual[i + 3] = alpha.mul_add(marker[i + 3] as f64, residual[i + 3]);
        i += 4;
    }
    while i < marker.len() {
        residual[i] = alpha.mul_add(marker[i] as f64, residual[i]);
        i += 1;
    }
}

/// f32 marker residual update.  `alpha` is retained in f64 because sampled
/// effects are f64; the per-sample update is intentionally rounded once when
/// written back to the f32 residual buffer.
#[inline]
pub(crate) fn update_marker_residual_f32(
    old_beta: f64,
    new_beta: f64,
    marker: &[f32],
    residual: &mut [f32],
) {
    debug_assert_eq!(marker.len(), residual.len());
    let alpha = (old_beta - new_beta) as f32;
    // Spike components frequently leave a marker at exactly zero.  Avoid a
    // full sample-vector traversal when replacing zero by zero; this is the
    // dominant case for BayesB/C/R and is also safe for all backends.
    if alpha == 0.0_f32 || marker.is_empty() {
        return;
    }
    let mut i = {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { update_marker_residual_f32_neon(alpha, marker, residual) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            0usize
        }
    };
    while i + 4 <= marker.len() {
        residual[i] = alpha.mul_add(marker[i], residual[i]);
        residual[i + 1] = alpha.mul_add(marker[i + 1], residual[i + 1]);
        residual[i + 2] = alpha.mul_add(marker[i + 2], residual[i + 2]);
        residual[i + 3] = alpha.mul_add(marker[i + 3], residual[i + 3]);
        i += 4;
    }
    while i < marker.len() {
        residual[i] = alpha.mul_add(marker[i], residual[i]);
        i += 1;
    }
}

/// Unified marker Gibbs kernel.  The conditional score is reduced once, the
/// caller samples the model-specific new effect from that score, and the
/// maintained f32 residual is updated before returning.  Keeping this
/// transition in one inlined function makes all BayesA/B/C/R backends share
/// the same SIMD dot/update path.  The two vector traversals are intentional:
/// a Gibbs draw cannot update the residual until its conditional draw is known.
#[inline]
pub(crate) fn bayes_marker_update_f32<F>(
    residual: &mut [f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
    choose_beta: F,
) -> (f64, f64)
where
    F: FnOnce(f64) -> f64,
{
    bayes_marker_update_f32_fused(residual, marker, old_beta, marker_ss, choose_beta)
}

/// Fast marker transition for models whose conditional draw cannot fail.
/// Keeping score, callback, and residual replacement in one inline function
/// removes the Result/closure adapter from BayesA/B/C while retaining the
/// required two streaming passes: the new beta is not known until the score
/// reduction has completed.
#[inline]
pub(crate) fn bayes_marker_update_f32_fused<F>(
    residual: &mut [f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
    choose_beta: F,
) -> (f64, f64)
where
    F: FnOnce(f64) -> f64,
{
    let u = marker_conditional_dot_f32_f32_blocked(residual, marker, old_beta, marker_ss);
    let new_beta = choose_beta(u);
    update_marker_residual_f32(old_beta, new_beta, marker, residual);
    (u, new_beta)
}

/// Result-returning form used by BayesR, whose component posterior calculation
/// performs input validation inside the marker callback.
#[inline]
pub(crate) fn bayes_marker_update_f32_result<F>(
    residual: &mut [f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
    choose_beta: F,
) -> Result<(f64, f64), String>
where
    F: FnOnce(f64) -> Result<f64, String>,
{
    let u = marker_conditional_dot_f32_f32_blocked(residual, marker, old_beta, marker_ss);
    let new_beta = choose_beta(u)?;
    update_marker_residual_f32(old_beta, new_beta, marker, residual);
    Ok((u, new_beta))
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn update_marker_residual_f32_neon(
    alpha: f32,
    marker: &[f32],
    residual: &mut [f32],
) -> usize {
    let limit = marker.len() / 4 * 4;
    let alpha_v = vdupq_n_f32(alpha);
    let mut i = 0usize;
    while i < limit {
        let m = vld1q_f32(marker.as_ptr().add(i));
        let r = vld1q_f32(residual.as_ptr().add(i));
        let out = vfmaq_f32(r, m, alpha_v);
        vst1q_f32(residual.as_mut_ptr().add(i), out);
        i += 4;
    }
    i
}

#[cfg(all(test, target_arch = "aarch64"))]
#[inline]
unsafe fn update_marker_residual_f32_f64_neon(
    alpha: f64,
    marker: &[f32],
    residual: &mut [f64],
) -> usize {
    let limit = marker.len() / 4 * 4;
    let alpha_v = vdupq_n_f64(alpha);
    let mut i = 0usize;
    while i < limit {
        let m = vld1q_f32(marker.as_ptr().add(i));
        let m_lo = vcvt_f64_f32(vget_low_f32(m));
        let m_hi = vcvt_f64_f32(vget_high_f32(m));
        let r_lo = vld1q_f64(residual.as_ptr().add(i));
        let r_hi = vld1q_f64(residual.as_ptr().add(i + 2));
        let out_lo = vfmaq_f64(r_lo, m_lo, alpha_v);
        let out_hi = vfmaq_f64(r_hi, m_hi, alpha_v);
        vst1q_f64(residual.as_mut_ptr().add(i), out_lo);
        vst1q_f64(residual.as_mut_ptr().add(i + 2), out_hi);
        i += 4;
    }
    i
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_alpha_gauss_seidel_blas(
    x: &[f64],
    n: usize,
    q: usize,
    inv_var_e: f64,
    inv_var_b_fixed: f64,
    x2_x: &[f64],
    xtx: &[f64],
    alpha: &mut [f64],
    r: &mut [f64],
    xtr_buf: &mut [f64],
    delta_alpha: &mut [f64],
    tmp_n: &mut [f64],
    rng: &mut StdRng,
) {
    debug_assert_eq!(x.len(), n.saturating_mul(q));
    debug_assert_eq!(x2_x.len(), q);
    debug_assert_eq!(xtx.len(), q.saturating_mul(q));
    debug_assert_eq!(alpha.len(), q);
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(xtr_buf.len(), q);
    debug_assert_eq!(delta_alpha.len(), q);
    debug_assert_eq!(tmp_n.len(), n);

    row_major_block_t_mul_vec_f64(x, n, q, r, xtr_buf);
    delta_alpha.fill(0.0_f64);

    for k in 0..q {
        let rhs = xtr_buf[k].mul_add(inv_var_e, x2_x[k] * alpha[k] * inv_var_e);
        let c = x2_x[k].mul_add(inv_var_e, inv_var_b_fixed);
        let z_alpha: f64 = rng.sample(StandardNormal);
        let new_alpha = rhs / c + (1.0_f64 / c).sqrt() * z_alpha;
        let delta = alpha[k] - new_alpha;
        alpha[k] = new_alpha;
        delta_alpha[k] = delta;
        if delta != 0.0_f64 {
            for t in 0..q {
                xtr_buf[t] += delta * xtx[t * q + k];
            }
        }
    }

    row_major_block_mul_vec_f64(x, n, q, delta_alpha, tmp_n);
    for i in 0..n {
        r[i] += tmp_n[i];
    }
}

pub(crate) struct PackedByteLut {
    code4: [[u8; 4]; 256],
}

pub(crate) fn packed_byte_lut() -> &'static PackedByteLut {
    static LUT: OnceLock<PackedByteLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut code4 = [[0u8; 4]; 256];
        for b in 0u16..=255 {
            let byte = b as u8;
            for lane in 0..4usize {
                code4[byte as usize][lane] = (byte >> (lane * 2)) & 0b11;
            }
        }
        PackedByteLut { code4 }
    })
}

impl PackedByteLut {
    pub(crate) fn code4(&self) -> &[[u8; 4]; 256] {
        &self.code4
    }
}

fn parse_env_truthy(raw: &str) -> Option<bool> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    if matches!(s.as_str(), "1" | "true" | "yes" | "y" | "on") {
        return Some(true);
    }
    if matches!(s.as_str(), "0" | "false" | "no" | "n" | "off") {
        return Some(false);
    }
    None
}

#[inline]
fn bayes_packed_double_buffer_enabled() -> bool {
    env::var("JX_BAYES_PACKED_DOUBLE_BUFFER")
        .ok()
        .and_then(|raw| parse_env_truthy(&raw))
        .unwrap_or(true)
}

#[inline]
pub(crate) fn bayes_packed_block_rows(n: usize, p: usize) -> usize {
    if n == 0 || p == 0 {
        return 1;
    }
    let default_target_bytes: usize = 32 * 1024 * 1024;
    let bytes_per_row = n.saturating_mul(std::mem::size_of::<f32>()).max(1);
    let mut rows = (default_target_bytes / bytes_per_row).max(1);
    rows = rows.clamp(8, 2048);
    if let Ok(raw) = env::var("JX_BAYES_PACKED_ROW_BLOCK") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            if v > 0 {
                rows = v;
            }
        }
    }
    rows.clamp(1, p)
}

#[inline]
fn bayes_packed_predecode_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        if let Ok(raw) = env::var("JX_BAYES_PACKED_PREDECODE") {
            if let Some(v) = parse_env_truthy(&raw) {
                return v;
            }
        }
        true
    })
}

#[inline]
fn bayes_packed_predecode_max_mb() -> usize {
    static MAX_MB: OnceLock<usize> = OnceLock::new();
    *MAX_MB.get_or_init(|| {
        if let Ok(raw) = env::var("JX_BAYES_PACKED_PREDECODE_MAX_MB") {
            if let Ok(v) = raw.trim().parse::<usize>() {
                if v > 0 {
                    return v;
                }
            }
        }
        768usize
    })
}

#[inline]
pub(crate) fn bayes_packed_blas_threads() -> usize {
    // Default policy stays "BLAS single-thread + Rayon parallel decode".
    // Env override for experiments:
    //   JX_BAYES_PACKED_BLAS_THREADS=4  -> force BLAS 4 threads in Bayes packed kernels
    //   JX_BAYES_PACKED_BLAS_THREADS=0  -> do not override BLAS threads
    if let Ok(raw) = env::var("JX_BAYES_PACKED_BLAS_THREADS") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            return v;
        }
    }
    1usize
}

#[inline]
fn bayes_packed_should_predecode_dense(n: usize, p: usize) -> bool {
    if !bayes_packed_predecode_enabled() {
        return false;
    }
    let elem_cnt = n.saturating_mul(p);
    if elem_cnt == 0 {
        return false;
    }
    let dense_bytes = elem_cnt.saturating_mul(std::mem::size_of::<f32>());
    let cap_bytes = bayes_packed_predecode_max_mb().saturating_mul(1024 * 1024);
    if dense_bytes > cap_bytes {
        return false;
    }
    // Keep a lightweight guard on huge dimensions even if memory cap allows it.
    elem_cnt <= 80_000_000usize
}

#[inline]
fn bayes_stream_window_mb(n_samples: usize, block_rows: usize) -> usize {
    let bytes_per_snp = n_samples.div_ceil(4).max(1);
    let target_bytes = block_rows
        .max(1)
        .saturating_mul(bytes_per_snp)
        .saturating_mul(2);
    target_bytes.div_ceil(1024 * 1024).max(1)
}

pub(crate) enum BayesPackedSource<'a> {
    Resident {
        packed_flat: &'a [u8],
        bytes_per_snp: usize,
    },
    Windowed(WindowedBedMatrix),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BayesDecodeMode {
    Dense,
    Single,
    Double,
}

#[inline]
pub(crate) fn bayes_decode_mode(p: usize, block_rows: usize) -> BayesDecodeMode {
    if p == 0 || block_rows >= p {
        BayesDecodeMode::Dense
    } else {
        BayesDecodeMode::Double
    }
}

/// Common marker access contract used by every Bayesian marker sampler.
///
/// Marker rows are returned in standardized f32 form.  The fixed-effect
/// residual is synchronized in f64 outside the marker sweep, while the hot
/// marker residual and genotype buffers are f32 and all reductions remain f64.
/// Resident packed BED, windowed BED, and dense inputs therefore share one
/// compact marker path.
pub(crate) trait BayesMarkerBackend {
    fn n_samples(&self) -> usize;
    fn n_markers(&self) -> usize;
    fn block_rows(&self) -> usize;
    fn fill_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        out_block: &mut [f32],
        n: usize,
    ) -> Result<(), String>;

    fn for_each_block<F>(&mut self, mut f: F) -> Result<(), String>
    where
        F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
        Self: Sized,
    {
        let n = self.n_samples();
        let p = self.n_markers();
        let block_rows = self.block_rows().max(1).min(p.max(1));
        let mut block = vec![0.0_f32; block_rows * n];
        for row_start in (0..p).step_by(block_rows) {
            let row_end = (row_start + block_rows).min(p);
            let block_len = (row_end - row_start) * n;
            self.fill_block(row_start, row_end, &mut block[..block_len], n)?;
            f(row_start, row_end, &block[..block_len])?;
        }
        Ok(())
    }
}

/// Dense marker backend used by the in-memory Bayes entry points.
pub(crate) struct DenseBayesBackend<'a> {
    matrix: &'a [f32],
    n: usize,
    p: usize,
    block_rows: usize,
}

impl<'a> DenseBayesBackend<'a> {
    pub(crate) fn new(matrix: &'a [f32], n: usize, p: usize) -> Result<Self, String> {
        if matrix.len() != p.saturating_mul(n) {
            return Err("Bayes dense genotype dimensions are incompatible".to_string());
        }
        Ok(Self {
            matrix,
            n,
            p,
            block_rows: p.min(2048).max(1),
        })
    }
}

impl BayesMarkerBackend for DenseBayesBackend<'_> {
    fn n_samples(&self) -> usize {
        self.n
    }

    fn n_markers(&self) -> usize {
        self.p
    }

    fn block_rows(&self) -> usize {
        self.block_rows
    }

    fn fill_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        out_block: &mut [f32],
        n: usize,
    ) -> Result<(), String> {
        if row_start > row_end || row_end > self.p || n != self.n {
            return Err("Bayes dense block bounds are invalid".to_string());
        }
        let expected = (row_end - row_start).saturating_mul(n);
        if out_block.len() != expected {
            return Err("Bayes dense block buffer has an invalid length".to_string());
        }
        out_block.copy_from_slice(&self.matrix[row_start * n..row_end * n]);
        Ok(())
    }

    /// Dense rows are already resident and contiguous.  Passing a borrowed
    /// slice avoids copying every marker block into a staging buffer on every
    /// MCMC iteration; packed/stream backends keep the default fill path.
    fn for_each_block<F>(&mut self, mut f: F) -> Result<(), String>
    where
        F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
        Self: Sized,
    {
        let n = self.n;
        let p = self.p;
        let block_rows = self.block_rows.max(1).min(p.max(1));
        for row_start in (0..p).step_by(block_rows) {
            let row_end = (row_start + block_rows).min(p);
            f(row_start, row_end, &self.matrix[row_start * n..row_end * n])?;
        }
        Ok(())
    }
}

/// Shared packed/stream BED backend.
///
/// This owns all source-specific state (row/sample mapping, decode scratch,
/// thread pool and source window).  An explicit memory-derived block uses two
/// reusable decode buffers by default; when that block covers every active
/// marker, the source is materialized once and the dense backend is used.
/// Keeping this policy here makes it apply uniformly to every Bayes model.
pub(crate) struct PackedBayesBackend<'a, 'source> {
    source: &'a mut BayesPackedSource<'source>,
    n_samples_total: usize,
    p: usize,
    row_flip: &'a [bool],
    row_mean: &'a [f32],
    row_inv_sd: &'a [f32],
    packed_row_indices: Option<&'a [usize]>,
    sample_indices: &'a [usize],
    full_sample_fast: bool,
    block_rows: usize,
    pool: Option<&'a Arc<rayon::ThreadPool>>,
    scratch_f32: Vec<f32>,
    decode_mode: BayesDecodeMode,
}

impl<'a, 'source> PackedBayesBackend<'a, 'source> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        source: &'a mut BayesPackedSource<'source>,
        n_samples_total: usize,
        p: usize,
        row_flip: &'a [bool],
        row_mean: &'a [f32],
        row_inv_sd: &'a [f32],
        packed_row_indices: Option<&'a [usize]>,
        sample_indices: &'a [usize],
        block_rows: Option<usize>,
        pool: Option<&'a Arc<rayon::ThreadPool>>,
    ) -> Result<Self, String> {
        if p == 0 {
            return Err("Bayes packed genotype has no markers".to_string());
        }
        if row_flip.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
            return Err("Bayes packed row metadata length mismatch".to_string());
        }
        if let Some(indices) = packed_row_indices {
            if indices.len() != p {
                return Err("Bayes packed source-row index length mismatch".to_string());
            }
        }
        if sample_indices.is_empty() || sample_indices.iter().any(|&i| i >= n_samples_total) {
            return Err("Bayes packed sample indices are invalid".to_string());
        }
        let explicit_block_rows = block_rows.is_some();
        let resolved_block_rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(sample_indices.len(), p))
            .max(1)
            .min(p);
        let decode_mode = if explicit_block_rows {
            let base_mode = bayes_decode_mode(p, resolved_block_rows);
            if base_mode == BayesDecodeMode::Double && !bayes_packed_double_buffer_enabled() {
                BayesDecodeMode::Single
            } else {
                base_mode
            }
        } else {
            BayesDecodeMode::Single
        };
        Ok(Self {
            source,
            n_samples_total,
            p,
            row_flip,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_indices,
            full_sample_fast: is_identity_indices(sample_indices, n_samples_total),
            block_rows: resolved_block_rows,
            pool,
            scratch_f32: Vec::new(),
            decode_mode,
        })
    }

    /// Materialize a packed source when the common predecode policy allows it.
    /// An explicit block covering all active markers forces this dense path,
    /// including for a windowed BED source.
    pub(crate) fn maybe_predecode_dense_f32(&mut self) -> Result<Option<Vec<f32>>, String> {
        let force_dense = self.decode_mode == BayesDecodeMode::Dense;
        let dense = maybe_predecode_source_dense_f32(
            &mut self.source,
            self.n_samples_total,
            self.row_flip,
            self.row_mean,
            self.row_inv_sd,
            self.sample_indices,
            self.sample_indices.len(),
            self.p,
            self.packed_row_indices,
            packed_byte_lut().code4(),
            self.pool,
            Some(self.block_rows),
            force_dense,
        )?;
        if dense.is_some() {
            self.decode_mode = BayesDecodeMode::Dense;
        }
        Ok(dense)
    }

    fn duplicate_source(&self) -> Result<BayesPackedSource<'source>, String> {
        match &*self.source {
            BayesPackedSource::Resident {
                packed_flat,
                bytes_per_snp,
            } => Ok(BayesPackedSource::Resident {
                packed_flat,
                bytes_per_snp: *bytes_per_snp,
            }),
            BayesPackedSource::Windowed(matrix) => {
                Ok(BayesPackedSource::Windowed(matrix.duplicate()?))
            }
        }
    }

    pub(crate) fn for_each_block<F>(&mut self, mut f: F) -> Result<(), String>
    where
        F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
    {
        if self.decode_mode == BayesDecodeMode::Double {
            return self.for_each_block_double(&mut f);
        }
        let n = self.sample_indices.len();
        let block_rows = self.block_rows.max(1).min(self.p.max(1));
        let mut block = vec![0.0_f32; block_rows * n];
        for row_start in (0..self.p).step_by(block_rows) {
            let row_end = (row_start + block_rows).min(self.p);
            let block_len = (row_end - row_start) * n;
            self.fill_block(row_start, row_end, &mut block[..block_len], n)?;
            f(row_start, row_end, &block[..block_len])?;
        }
        Ok(())
    }

    fn for_each_block_double<F>(&mut self, f: &mut F) -> Result<(), String>
    where
        F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
    {
        let worker_source = self.duplicate_source()?;
        let n = self.sample_indices.len();
        let p = self.p;
        let block_rows = self.block_rows.max(1).min(p.max(1));
        let n_samples_total = self.n_samples_total;
        let row_flip = self.row_flip;
        let row_mean = self.row_mean;
        let row_inv_sd = self.row_inv_sd;
        let packed_row_indices = self.packed_row_indices;
        let sample_indices = self.sample_indices;
        let full_sample_fast = self.full_sample_fast;
        let pool = self.pool;
        let code4_lut = packed_byte_lut().code4();

        std::thread::scope(|scope| {
            let (request_tx, request_rx) = sync_channel::<(usize, usize, Vec<f32>)>(2);
            let (ready_tx, ready_rx) = sync_channel::<Result<(usize, usize, Vec<f32>), String>>(2);

            scope.spawn(move || {
                let mut source = worker_source;
                let mut scratch_f32 = Vec::<f32>::new();
                while let Ok((row_start, row_end, mut block)) = request_rx.recv() {
                    let block_len = (row_end - row_start) * n;
                    let result = decode_source_block_standardized_into(
                        &mut source,
                        n_samples_total,
                        row_start,
                        row_end,
                        sample_indices,
                        full_sample_fast,
                        packed_row_indices,
                        row_flip,
                        row_mean,
                        row_inv_sd,
                        code4_lut,
                        &mut block[..block_len],
                        n,
                        pool,
                        &mut scratch_f32,
                    )
                    .map(|_| (row_start, row_end, block));
                    if ready_tx.send(result).is_err() {
                        break;
                    }
                }
            });

            let mut blocks = (0..p).step_by(block_rows).map(|row_start| {
                let row_end = (row_start + block_rows).min(p);
                (row_start, row_end)
            });
            let send_request =
                |row_start: usize, row_end: usize, mut block: Vec<f32>| -> Result<(), String> {
                    let block_len = (row_end - row_start) * n;
                    if block.len() != block_len {
                        block.resize(block_len, 0.0);
                    }
                    request_tx
                        .send((row_start, row_end, block))
                        .map_err(|_| "Bayes double-buffer worker stopped".to_string())
                };

            let mut in_flight = 0usize;
            for _ in 0..2 {
                if let Some((row_start, row_end)) = blocks.next() {
                    send_request(row_start, row_end, Vec::new())?;
                    in_flight += 1;
                }
            }

            while in_flight > 0 {
                let ready = ready_rx
                    .recv()
                    .map_err(|_| "Bayes double-buffer worker stopped".to_string())??;
                let (row_start, row_end, block) = ready;
                in_flight -= 1;
                f(row_start, row_end, &block)?;
                if let Some((next_start, next_end)) = blocks.next() {
                    send_request(next_start, next_end, block)?;
                    in_flight += 1;
                }
            }
            Ok(())
        })
    }
}

impl BayesMarkerBackend for PackedBayesBackend<'_, '_> {
    fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    fn n_markers(&self) -> usize {
        self.p
    }

    fn block_rows(&self) -> usize {
        self.block_rows
    }

    fn fill_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        out_block: &mut [f32],
        n: usize,
    ) -> Result<(), String> {
        if row_start > row_end || row_end > self.p || n != self.sample_indices.len() {
            return Err("Bayes packed block bounds are invalid".to_string());
        }
        decode_source_block_standardized_into(
            &mut self.source,
            self.n_samples_total,
            row_start,
            row_end,
            self.sample_indices,
            self.full_sample_fast,
            self.packed_row_indices,
            self.row_flip,
            self.row_mean,
            self.row_inv_sd,
            packed_byte_lut().code4(),
            out_block,
            n,
            self.pool,
            &mut self.scratch_f32,
        )
    }
}

pub(crate) fn build_bayes_source<'a>(
    resident_packed_flat: Option<&'a [u8]>,
    prefix: &str,
    n_samples: usize,
    block_rows: usize,
    mmap_window_mb: Option<usize>,
) -> Result<BayesPackedSource<'a>, String> {
    if let Some(packed_flat) = resident_packed_flat {
        return Ok(BayesPackedSource::Resident {
            packed_flat,
            bytes_per_snp: n_samples.div_ceil(4),
        });
    }
    if prefix.trim().is_empty() {
        return Err("Bayes streaming path requires non-empty prefix".to_string());
    }
    let window_mb = mmap_window_mb
        .map(|v| v.max(1))
        .unwrap_or_else(|| bayes_stream_window_mb(n_samples, block_rows));
    let matrix = WindowedBedMatrix::open(prefix, window_mb)?;
    Ok(BayesPackedSource::Windowed(matrix))
}

#[inline]
pub(crate) fn decode_source_block_standardized_into(
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_start: usize,
    row_end: usize,
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    code4_lut: &[[u8; 4]; 256],
    out_block: &mut [f32],
    n: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
    scratch_f32: &mut Vec<f32>,
) -> Result<(), String> {
    let block_rows = row_end - row_start;
    debug_assert_eq!(out_block.len(), block_rows * n);
    if scratch_f32.len() < block_rows * n {
        scratch_f32.resize(block_rows * n, 0.0_f32);
    }
    let tmp = &mut scratch_f32[..block_rows * n];
    match source {
        BayesPackedSource::Resident {
            packed_flat,
            bytes_per_snp,
        } => {
            decode_standardized_packed_block_f32(
                packed_flat,
                *bytes_per_snp,
                n_samples,
                row_flip,
                row_mean,
                row_inv_sd,
                sample_idx,
                full_sample_fast,
                row_start,
                tmp,
                code4_lut,
                pool,
            )?;
        }
        BayesPackedSource::Windowed(matrix) => {
            let cur_rows = row_end - row_start;
            let bytes_per_snp = matrix.bytes_per_snp();
            let packed_slice = if let Some(indices) = packed_row_indices {
                let source_rows = &indices[row_start..row_end];
                let mut rel_indices = Vec::with_capacity(cur_rows);
                let packed_slice = matrix.prepare_source_rows(source_rows, &mut rel_indices)?;
                decode_prepared_additive_block_packed_f32(
                    packed_slice,
                    bytes_per_snp,
                    n_samples,
                    &row_flip[row_start..row_end],
                    &row_mean[row_start..row_end],
                    &row_inv_sd[row_start..row_end],
                    sample_idx,
                    full_sample_fast,
                    None,
                    Some(rel_indices.as_slice()),
                    0usize,
                    cur_rows,
                    tmp,
                    pool,
                )?;
                out_block.copy_from_slice(tmp);
                return Ok(());
            } else {
                matrix.read_source_range(row_start, row_end)?
            };
            decode_prepared_additive_block_packed_f32(
                packed_slice,
                bytes_per_snp,
                n_samples,
                &row_flip[row_start..row_end],
                &row_mean[row_start..row_end],
                &row_inv_sd[row_start..row_end],
                sample_idx,
                full_sample_fast,
                None,
                None,
                0usize,
                cur_rows,
                tmp,
                pool,
            )?;
        }
    }
    out_block.copy_from_slice(tmp);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_predecode_source_dense_f32(
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    n: usize,
    p: usize,
    packed_row_indices: Option<&[usize]>,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
    block_rows_override: Option<usize>,
    force_dense: bool,
) -> Result<Option<Vec<f32>>, String> {
    if !force_dense && matches!(source, BayesPackedSource::Windowed(_)) {
        return Ok(None);
    }
    if !force_dense && !bayes_packed_should_predecode_dense(n, p) {
        return Ok(None);
    }
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let block_rows = block_rows_override
        .unwrap_or_else(|| bayes_packed_block_rows(n, p))
        .max(1)
        .min(p.max(1));
    let mut dense_f32 = vec![0.0_f32; p * n];
    let mut scratch_f32 = Vec::<f32>::new();
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_source_block_standardized_into(
            source,
            n_samples,
            st,
            ed,
            sample_idx,
            full_sample_fast,
            packed_row_indices,
            row_flip,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut dense_f32[st * n..(st + br) * n],
            n,
            pool,
            &mut scratch_f32,
        )?;
    }
    Ok(Some(dense_f32))
}

#[inline]
fn genetic_value_from_residual(
    y: &[f64],
    r: &[f64],
    x: &[f64],
    alpha: &[f64],
    i: usize,
    q: usize,
) -> f64 {
    let mut xa = 0.0;
    for k in 0..q {
        xa += x[i * q + k] * alpha[k];
    }
    y[i] - r[i] - xa
}

/// Build the low-dimensional predictive state used by adaptive multi-chain
/// stopping.  The fixed sign sketches are cheap linear projections of the
/// current GEBV vector; unlike mixture hyperparameters, they directly detect
/// chains that still disagree on marker allocation and predictions.
pub(crate) fn bayes_predictive_monitor_values(
    y: &[f64],
    r: &[f64],
    x: &[f64],
    alpha: &[f64],
    n: usize,
    q: usize,
    h2: f64,
    var_g: f64,
    var_e: f64,
) -> Vec<f64> {
    if n <= 1 || y.len() < n || r.len() < n || x.len() < n.saturating_mul(q) || alpha.len() < q {
        return vec![f64::NAN; BAYES_PREDICTIVE_MONITOR_NAMES.len()];
    }
    let mut mean_g = 0.0;
    let mut mean_y = 0.0;
    for i in 0..n {
        let g = genetic_value_from_residual(y, r, x, alpha, i, q);
        let count = (i + 1) as f64;
        mean_g += (g - mean_g) / count;
        mean_y += (y[i] - mean_y) / count;
    }
    let mut centered_ss_g = 0.0;
    let mut centered_cross_gy = 0.0;
    let mut sketches = [0.0_f64; 8];
    for i in 0..n {
        let g = genetic_value_from_residual(y, r, x, alpha, i, q);
        let centered_g = g - mean_g;
        let centered_y = y[i] - mean_y;
        centered_ss_g += centered_g * centered_g;
        centered_cross_gy += centered_g * centered_y;
        for (sketch, value) in sketches.iter_mut().enumerate() {
            let key =
                (i as u64).wrapping_add((sketch as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let sign = if splitmix64(key) & 1 == 0 { 1.0 } else { -1.0 };
            *value += sign * centered_g;
        }
    }
    let denominator = (n - 1) as f64;
    let mut values = Vec::with_capacity(BAYES_PREDICTIVE_MONITOR_NAMES.len());
    values.extend([h2, var_g, var_e]);
    values.push(mean_g);
    values.push(centered_ss_g / denominator);
    values.push(centered_cross_gy / denominator);
    values.extend(sketches.into_iter().map(|value| value / n as f64));
    values
}

pub(crate) fn genetic_variance_from_residual(
    y: &[f64],
    r: &[f64],
    x: &[f64],
    alpha: &[f64],
    n: usize,
    q: usize,
) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut mean_g = 0.0;
    let mut m2 = 0.0;
    for i in 0..n {
        let g = genetic_value_from_residual(y, r, x, alpha, i, q);
        let delta = g - mean_g;
        mean_g += delta / (i as f64 + 1.0);
        let delta2 = g - mean_g;
        m2 += delta * delta2;
    }
    m2 / (n as f64 - 1.0)
}

#[inline]
fn bayesb_rate0(shape0: f64, rate0_opt: Option<f64>, s0_b: f64) -> Result<f64, String> {
    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }
    Ok(rate0)
}

#[inline]
fn bayes_inclusion_prior_shapes(prob_in_init: f64, counts: f64) -> (f64, f64) {
    let counts_in = (counts * prob_in_init).max(f64::MIN_POSITIVE);
    let counts_out = (counts * (1.0 - prob_in_init)).max(f64::MIN_POSITIVE);
    (counts_in, counts_out)
}

#[inline]
fn sample_gamma_with_rate<R: Rng + ?Sized>(
    rng: &mut R,
    shape: f64,
    rate: f64,
) -> Result<f64, String> {
    if !(shape.is_finite() && shape > 0.0) {
        return Err("Gamma shape must be positive and finite".to_string());
    }
    if !(rate.is_finite() && rate > 0.0) {
        return Err("Gamma rate must be positive and finite".to_string());
    }
    let gamma = Gamma::new(shape, 1.0).map_err(|e| e.to_string())?;
    Ok(rng.sample(gamma) / rate)
}

#[inline]
fn bayes_positive_floor(x: f64) -> f64 {
    x.max(1e-300_f64)
}

fn bayesb_core_impl(
    y: &[f64],
    m: &[f32],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<BayesBCResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if q == 0 || x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += (v as f64) * (v as f64);
            msum += v as f64;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, Some(prob_in_base), s0_b_opt)?;

    let rate0 = bayesb_rate0(shape0, rate0_opt, s0_b)?;

    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;
    let mut prob_in = prob_in_base;
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_base, counts);

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();

    let mut r = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut pip_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut schedule = BayesSamplingController::new_with_options(
        n_iter,
        sampling.burnin,
        sampling.adaptive_burnin,
        thin,
        BAYES_RHAT_B_NAMES,
        sampling.posterior_target,
    );
    let var_b_fixed = 1e10_f64;

    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let b_old = beta[j];
            let c = x2[j] * inv_var_e + 1.0 / var_b[j];
            if !(c.is_finite() && c > 0.0) {
                return Err("Non-positive posterior precision in BayesB beta update".to_string());
            }
            let mut new_d = 0u8;
            let (_, new_beta) =
                bayes_marker_update_f32(&mut marker_residual, m_j, b_old, x2[j], |xe| {
                    let rhs = xe * inv_var_e;
                    // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                    let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b[j] * c).ln();
                    let log_odds = log_odds_prior + log_bf10;
                    let p_in = if log_odds >= 0.0 {
                        1.0 / (1.0 + (-log_odds).exp())
                    } else {
                        let e = log_odds.exp();
                        e / (1.0 + e)
                    };
                    new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                    if new_d == 1 {
                        let z_beta: f64 = rng.sample(StandardNormal);
                        rhs / c + (1.0 / c).sqrt() * z_beta
                    } else {
                        0.0
                    }
                });
            d[j] = new_d;
            beta[j] = new_beta;
        }

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut n_active = 0usize;
        for j in 0..p {
            if d[j] == 1 {
                let beta2 = beta[j] * beta[j];
                var_b[j] = bayes_positive_floor((s + beta2) / rng.sample(chi_b_active));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err("BayesB var_b became non-finite or non-positive".to_string());
                }
                n_active += 1;
            } else {
                var_b[j] = bayes_positive_floor(s / rng.sample(chi_b_inactive));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err(
                        "BayesB inactive var_b became non-finite or non-positive".to_string()
                    );
                }
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating BayesB S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        s = bayes_positive_floor(sample_gamma_with_rate(&mut rng, tmp_shape, tmp_rate)?);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesB S became non-finite or non-positive".to_string());
        }

        let mrk_in = n_active as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesB var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                pip_sum[j] += d[j] as f64;
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            let rhat_metrics = [h2, var_g, var_e, prob_in, mrk_in];
            if schedule.observe(&rhat_metrics) {
                beta_sum.fill(0.0);
                pip_sum.fill(0.0);
                varb_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
                prob_in_sum = 0.0;
                n_active_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        pip_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;
    let rhat_h2 = schedule.rhat_value();
    let rhat_metrics = schedule.rhat_values();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        pip_sum,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
        rhat_metrics,
    ))
}

fn bayesc_core_impl(
    y: &[f64],
    m: &[f32],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<BayesCResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if q == 0 || x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += (v as f64) * (v as f64);
            msum += v as f64;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, Some(prob_in_base), s0_b_opt)?;

    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = s0_b;
    let mut prob_in = prob_in_base;

    let counts_in = counts * prob_in_base;
    let counts_out = counts - counts_in;

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();

    let mut r = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut pip_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut varb_sum = 0.0;
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut schedule = BayesSamplingController::new_with_options(
        n_iter,
        sampling.burnin,
        sampling.adaptive_burnin,
        thin,
        BAYES_RHAT_C_NAMES,
        sampling.posterior_target,
    );
    let var_b_fixed = 1e10_f64;
    let mut chi_b_cache: HashMap<usize, ChiSquared<f64>> = HashMap::new();

    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let b_old = beta[j];
            let c = x2[j] * inv_var_e + 1.0 / var_b;
            if !(c.is_finite() && c > 0.0) {
                return Err("Non-positive posterior precision in BayesC beta update".to_string());
            }
            let mut new_d = 0u8;
            let (_, new_beta) =
                bayes_marker_update_f32(&mut marker_residual, m_j, b_old, x2[j], |xe| {
                    let rhs = xe * inv_var_e;
                    // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                    let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b * c).ln();
                    let log_odds = log_odds_prior + log_bf10;
                    let p_in = if log_odds >= 0.0 {
                        1.0 / (1.0 + (-log_odds).exp())
                    } else {
                        let e = log_odds.exp();
                        e / (1.0 + e)
                    };
                    new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                    if new_d == 1 {
                        let z_beta: f64 = rng.sample(StandardNormal);
                        rhs / c + (1.0 / c).sqrt() * z_beta
                    } else {
                        0.0
                    }
                });
            d[j] = new_d;
            beta[j] = new_beta;
        }

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut mrk_in_usize = 0usize;
        let mut ss_b = 0.0;
        for j in 0..p {
            if d[j] == 1 {
                ss_b += beta[j] * beta[j];
                mrk_in_usize += 1;
            }
        }
        ss_b += s0_b;
        let chi_b_eff = if let Some(dist) = chi_b_cache.get(&mrk_in_usize) {
            *dist
        } else {
            let dist = ChiSquared::new(df0_b + mrk_in_usize as f64).map_err(|e| e.to_string())?;
            chi_b_cache.insert(mrk_in_usize, dist);
            dist
        };
        var_b = ss_b / rng.sample(chi_b_eff);
        if !(var_b.is_finite() && var_b > 0.0) {
            return Err("BayesC var_b became non-finite or non-positive".to_string());
        }

        let mrk_in = mrk_in_usize as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesC var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                pip_sum[j] += d[j] as f64;
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            varb_sum += var_b;
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            let rhat_metrics = [h2, var_g, var_e, var_b, prob_in, mrk_in];
            if schedule.observe(&rhat_metrics) {
                beta_sum.fill(0.0);
                pip_sum.fill(0.0);
                alpha_sum.fill(0.0);
                varb_sum = 0.0;
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
                prob_in_sum = 0.0;
                n_active_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        pip_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    varb_sum *= inv_keep;
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;
    let rhat_h2 = schedule.rhat_value();
    let rhat_metrics = schedule.rhat_values();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        pip_sum,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
        rhat_metrics,
    ))
}

fn bayesa_core_impl(
    y: &[f64],
    m: &[f32],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    _min_abs_beta: f64,
    seed: Option<u64>,
) -> Result<BayesAResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if q == 0 || x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += (v as f64) * (v as f64);
            msum += v as f64;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, None, s0_b_opt)?;

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();

    let mut r = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut schedule = BayesSamplingController::new_with_options(
        n_iter,
        sampling.burnin,
        sampling.adaptive_burnin,
        thin,
        BAYES_RHAT_A_NAMES,
        sampling.posterior_target,
    );
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        copy_f64_to_f32(&r, &mut marker_residual);
        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let c = x2[j] * inv_var_e + 1.0 / var_b[j];
            let z_beta: f64 = rng.sample(StandardNormal);
            let old_beta = beta[j];
            let (_, new_beta) =
                bayes_marker_update_f32(&mut marker_residual, m_j, old_beta, x2[j], |u| {
                    u * inv_var_e / c + (1.0 / c).sqrt() * z_beta
                });
            beta[j] = new_beta;
        }

        copy_f32_to_f64(&marker_residual, &mut r);
        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
            if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                return Err("BayesA var_b became non-finite or non-positive".to_string());
            }
        }

        let mut tmp_rate = 0.0;
        for j in 0..p {
            tmp_rate += 1.0 / var_b[j];
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesA S became non-finite or non-positive".to_string());
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesA var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                beta_sum[j] += beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            let rhat_metrics = [h2, var_g, var_e];
            if schedule.observe(&rhat_metrics) {
                beta_sum.fill(0.0);
                varb_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let rhat_h2 = schedule.rhat_value();
    let rhat_metrics = schedule.rhat_values();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();
    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
        rhat_metrics,
    ))
}

type BayesAResult = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
    f64,
    f64,
    usize,
    usize,
    usize,
    Vec<f64>,
);

type BayesBCResult = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
    f64,
    f64,
    f64,
    Vec<f64>,
    f64,
    usize,
    usize,
    usize,
    Vec<f64>,
);

type BayesCResult = (
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    Vec<f64>,
    f64,
    usize,
    usize,
    usize,
    Vec<f64>,
);

fn average_f64_vectors(values: &[Vec<f64>]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut output = values[0].clone();
    for value in values.iter().skip(1) {
        for (dst, src) in output.iter_mut().zip(value.iter()) {
            *dst += *src;
        }
    }
    let scale = 1.0 / values.len() as f64;
    for value in &mut output {
        *value *= scale;
    }
    output
}

fn aggregate_bayes_a_results(results: &[BayesAResult]) -> BayesAResult {
    let h2_means = results.iter().map(|value| value.4).collect::<Vec<_>>();
    let h2_vars = results.iter().map(|value| value.5).collect::<Vec<_>>();
    let rhats = results.iter().map(|value| value.6).collect::<Vec<_>>();
    let n_post = results.iter().map(|value| value.9).min().unwrap_or(0);
    let rhat_metrics = average_f64_vectors(
        &results
            .iter()
            .map(|value| value.10.clone())
            .collect::<Vec<_>>(),
    );
    let (h2_mean, var_h2, rhat) = aggregate_bayes_h2(&h2_means, &h2_vars, &rhats, n_post);
    let actual_iterations = results.iter().map(|value| value.7).max().unwrap_or(0);
    let convergence_values = results.iter().map(|value| value.8).collect::<Vec<_>>();
    let convergence_iteration = if convergence_values.iter().all(|value| *value > 0) {
        convergence_values.iter().copied().max().unwrap_or(0)
    } else {
        0
    };
    (
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.0.clone())
                .collect::<Vec<_>>(),
        ),
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.1.clone())
                .collect::<Vec<_>>(),
        ),
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.2.clone())
                .collect::<Vec<_>>(),
        ),
        results.iter().map(|value| value.3).sum::<f64>() / results.len() as f64,
        h2_mean,
        var_h2,
        rhat,
        actual_iterations,
        convergence_iteration,
        n_post,
        rhat_metrics,
    )
}

fn aggregate_bayes_bc_results(results: &[BayesBCResult]) -> BayesBCResult {
    let h2_means = results.iter().map(|value| value.4).collect::<Vec<_>>();
    let h2_vars = results.iter().map(|value| value.5).collect::<Vec<_>>();
    let rhats = results.iter().map(|value| value.9).collect::<Vec<_>>();
    let n_post = results.iter().map(|value| value.12).min().unwrap_or(0);
    let rhat_metrics = average_f64_vectors(
        &results
            .iter()
            .map(|value| value.13.clone())
            .collect::<Vec<_>>(),
    );
    let (h2_mean, var_h2, rhat) = aggregate_bayes_h2(&h2_means, &h2_vars, &rhats, n_post);
    let actual_iterations = results.iter().map(|value| value.10).max().unwrap_or(0);
    let convergence_values = results.iter().map(|value| value.11).collect::<Vec<_>>();
    let convergence_iteration = if convergence_values.iter().all(|value| *value > 0) {
        convergence_values.iter().copied().max().unwrap_or(0)
    } else {
        0
    };
    (
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.0.clone())
                .collect::<Vec<_>>(),
        ),
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.1.clone())
                .collect::<Vec<_>>(),
        ),
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.2.clone())
                .collect::<Vec<_>>(),
        ),
        results.iter().map(|value| value.3).sum::<f64>() / results.len() as f64,
        h2_mean,
        var_h2,
        results.iter().map(|value| value.6).sum::<f64>() / results.len() as f64,
        results.iter().map(|value| value.7).sum::<f64>() / results.len() as f64,
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.8.clone())
                .collect::<Vec<_>>(),
        ),
        rhat,
        actual_iterations,
        convergence_iteration,
        n_post,
        rhat_metrics,
    )
}

fn aggregate_bayes_c_results(results: &[BayesCResult]) -> BayesCResult {
    let h2_means = results.iter().map(|value| value.4).collect::<Vec<_>>();
    let h2_vars = results.iter().map(|value| value.5).collect::<Vec<_>>();
    let rhats = results.iter().map(|value| value.9).collect::<Vec<_>>();
    let n_post = results.iter().map(|value| value.12).min().unwrap_or(0);
    let rhat_metrics = average_f64_vectors(
        &results
            .iter()
            .map(|value| value.13.clone())
            .collect::<Vec<_>>(),
    );
    let (h2_mean, var_h2, rhat) = aggregate_bayes_h2(&h2_means, &h2_vars, &rhats, n_post);
    let actual_iterations = results.iter().map(|value| value.10).max().unwrap_or(0);
    let convergence_values = results.iter().map(|value| value.11).collect::<Vec<_>>();
    let convergence_iteration = if convergence_values.iter().all(|value| *value > 0) {
        convergence_values.iter().copied().max().unwrap_or(0)
    } else {
        0
    };
    (
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.0.clone())
                .collect::<Vec<_>>(),
        ),
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.1.clone())
                .collect::<Vec<_>>(),
        ),
        results.iter().map(|value| value.2).sum::<f64>() / results.len() as f64,
        results.iter().map(|value| value.3).sum::<f64>() / results.len() as f64,
        h2_mean,
        var_h2,
        results.iter().map(|value| value.6).sum::<f64>() / results.len() as f64,
        results.iter().map(|value| value.7).sum::<f64>() / results.len() as f64,
        average_f64_vectors(
            &results
                .iter()
                .map(|value| value.8.clone())
                .collect::<Vec<_>>(),
        ),
        rhat,
        actual_iterations,
        convergence_iteration,
        n_post,
        rhat_metrics,
    )
}

pub(crate) fn marker_sufficient_stats<B: BayesMarkerBackend>(
    backend: &mut B,
    n: usize,
    p: usize,
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    let n_f = n as f64;
    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|row_start, row_end, block| {
        for offset in 0..(row_end - row_start) {
            let row = &block[offset * n..(offset + 1) * n];
            let mut ss = 0.0;
            let mut sum = 0.0;
            for value in row.iter().copied() {
                let value = value as f64;
                ss += value * value;
                sum += value;
            }
            x2[row_start + offset] = ss;
            mean_x[row_start + offset] = sum / n_f;
        }
        Ok(())
    })?;
    let msx = x2.iter().sum::<f64>() / n_f - mean_x.iter().map(|value| value * value).sum::<f64>();
    Ok((x2, mean_x, msx))
}

fn phenotype_variance(y: &[f64]) -> Result<f64, String> {
    if y.len() <= 1 {
        return Err("n must be > 1".to_string());
    }
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let variance = y
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (y.len() - 1) as f64;
    if !(variance.is_finite() && variance > 0.0) {
        return Err("phenotype variance must be positive".to_string());
    }
    Ok(variance)
}

struct BayesAChainState {
    rng: StdRng,
    beta: Vec<f64>,
    var_b: Vec<f64>,
    s: f64,
    var_e: f64,
    alpha: Vec<f64>,
    residual: Vec<f64>,
    marker_residual: Vec<f32>,
    alpha_xtr: Vec<f64>,
    alpha_delta: Vec<f64>,
    alpha_tmp_n: Vec<f64>,
    beta_sum: Vec<f64>,
    varb_sum: Vec<f64>,
    alpha_sum: Vec<f64>,
    var_e_sum: f64,
}

impl BayesAChainState {
    fn new(
        fixed_init: &BayesFixedEffectInit,
        p: usize,
        q: usize,
        s0_b: f64,
        df0_b: f64,
        var_e: f64,
        start_scale: f64,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => {
                let mut seed_bytes = [0u8; 32];
                if OsRng.try_fill_bytes(&mut seed_bytes).is_err() {
                    seed_bytes = [42u8; 32];
                }
                StdRng::from_seed(seed_bytes)
            }
        };
        let initial_s = s0_b * start_scale;
        Self {
            rng,
            beta: vec![0.0; p],
            var_b: vec![initial_s / (df0_b + 2.0); p],
            s: initial_s,
            var_e,
            alpha: fixed_init.alpha.clone(),
            residual: fixed_init.residual.clone(),
            marker_residual: vec![0.0; fixed_init.residual.len()],
            alpha_xtr: vec![0.0; q],
            alpha_delta: vec![0.0; q],
            alpha_tmp_n: vec![0.0; fixed_init.residual.len()],
            beta_sum: vec![0.0; p],
            varb_sum: vec![0.0; p],
            alpha_sum: vec![0.0; q],
            var_e_sum: 0.0,
        }
    }

    fn clear_posterior(&mut self) {
        self.beta_sum.fill(0.0);
        self.varb_sum.fill(0.0);
        self.alpha_sum.fill(0.0);
        self.var_e_sum = 0.0;
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesa_lockstep_core_impl<B: BayesMarkerBackend>(
    backend: &mut B,
    y: &[f64],
    x: &[f64],
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    chain_pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<BayesAResult, String> {
    let n = backend.n_samples();
    let p = backend.n_markers();
    if n <= 1 || y.len() != n {
        return Err("BayesA phenotype/genotype dimensions are incompatible".to_string());
    }
    if q == 0 || x.len() != n.saturating_mul(q) {
        return Err("BayesA covariate dimensions are incompatible".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;
    let (x2, _mean_x, msx) = marker_sufficient_stats(backend, n, p)?;
    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, None, s0_b_opt)?;
    let rate0 = rate0_opt.unwrap_or_else(|| (shape0 - 1.0) / s0_b);
    if !(rate0.is_finite() && rate0 > 0.0) {
        return Err("rate0 must be positive; shape0 must be > 1".to_string());
    }
    let initial_var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let adaptive = sampling.adaptive_burnin || sampling.burnin.is_none();
    let seeds = bayes_chain_seeds(seed, chains);
    let mut states = seeds
        .into_iter()
        .enumerate()
        .map(|(chain_index, chain_seed)| {
            let start_scale = bayes_chain_start_scale(chain_index, chains, adaptive);
            BayesAChainState::new(
                &fixed_init,
                p,
                q,
                s0_b,
                df0_b,
                initial_var_e,
                start_scale,
                chain_seed,
            )
        })
        .collect::<Vec<_>>();
    let mut controller = BayesMultiChainController::new_with_monitor_options(
        chains,
        n_iter,
        sampling.burnin,
        adaptive,
        thin,
        BAYES_RHAT_A_NAMES,
        BAYES_PREDICTIVE_MONITOR_NAMES,
        sampling.posterior_target,
    )?;
    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|error| error.to_string())?;
    let chi_e = ChiSquared::new(n as f64 + df0_e).map_err(|error| error.to_string())?;
    let var_b_fixed = 1.0e10_f64;

    while controller.should_run() {
        let retained = controller.begin_iteration();
        let collect = controller.collecting_posterior();
        bayes_chain_for_each_mut(&mut states, chain_pool, |state| {
            let inv_var_e = 1.0 / state.var_e;
            update_alpha_gauss_seidel_blas(
                x,
                n,
                q,
                inv_var_e,
                1.0 / var_b_fixed,
                &x2_x,
                &xtx,
                &mut state.alpha,
                &mut state.residual,
                &mut state.alpha_xtr,
                &mut state.alpha_delta,
                &mut state.alpha_tmp_n,
                &mut state.rng,
            );
            copy_f64_to_f32(&state.residual, &mut state.marker_residual);
        });
        backend.for_each_block(|row_start, row_end, block| {
            bayes_chain_for_each_mut(&mut states, chain_pool, |state| {
                let inv_var_e = 1.0 / state.var_e;
                for offset in 0..(row_end - row_start) {
                    let j = row_start + offset;
                    let marker = &block[offset * n..(offset + 1) * n];
                    let c = x2[j] * inv_var_e + 1.0 / state.var_b[j];
                    let z_beta: f64 = state.rng.sample(StandardNormal);
                    let old_beta = state.beta[j];
                    let (_, new_beta) = bayes_marker_update_f32(
                        &mut state.marker_residual,
                        marker,
                        old_beta,
                        x2[j],
                        |u| u * inv_var_e / c + (1.0 / c).sqrt() * z_beta,
                    );
                    state.beta[j] = new_beta;
                }
            });
            Ok(())
        })?;

        let chain_summaries = bayes_chain_try_map_mut(&mut states, chain_pool, |state| {
            copy_f32_to_f64(&state.marker_residual, &mut state.residual);
            for j in 0..p {
                state.var_b[j] = bayes_positive_floor(
                    (state.s + state.beta[j] * state.beta[j]) / state.rng.sample(chi_b),
                );
            }
            let tmp_rate = state.var_b.iter().map(|value| 1.0 / value).sum::<f64>() / 2.0 + rate0;
            let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
            state.s =
                bayes_positive_floor(sample_gamma_with_rate(&mut state.rng, tmp_shape, tmp_rate)?);
            state.var_e =
                (ddot_f64(&state.residual, &state.residual) + prior_ss_e) / state.rng.sample(chi_e);
            if !(state.var_e.is_finite() && state.var_e > 0.0) {
                return Err("BayesA var_e became non-finite or non-positive".to_string());
            }
            let var_g = genetic_variance_from_residual(y, &state.residual, x, &state.alpha, n, q);
            let h2_value = var_g / (var_g + state.var_e);
            if collect && retained {
                for j in 0..p {
                    state.beta_sum[j] += state.beta[j];
                    state.varb_sum[j] += state.var_b[j];
                }
                for k in 0..q {
                    state.alpha_sum[k] += state.alpha[k];
                }
                state.var_e_sum += state.var_e;
            }
            Ok((
                vec![h2_value, var_g, state.var_e],
                bayes_predictive_monitor_values(
                    y,
                    &state.residual,
                    x,
                    &state.alpha,
                    n,
                    q,
                    h2_value,
                    var_g,
                    state.var_e,
                ),
            ))
        })?;
        let (metrics, monitor_metrics): (Vec<_>, Vec<_>) = chain_summaries.into_iter().unzip();
        if controller.observe_with_monitor(&metrics, &monitor_metrics, retained)? {
            bayes_chain_for_each_mut(&mut states, chain_pool, |state| {
                state.clear_posterior();
            });
        }
    }
    let n_keep = controller.posterior_samples();
    if n_keep != controller.posterior_target() {
        return Err(format!(
            "BayesA retained {n_keep} posterior samples; expected {}",
            controller.posterior_target()
        ));
    }
    let scale = 1.0 / (n_keep * chains) as f64;
    let beta = (0..p)
        .map(|j| states.iter().map(|state| state.beta_sum[j]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let varbeta = (0..p)
        .map(|j| states.iter().map(|state| state.varb_sum[j]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let alpha = (0..q)
        .map(|k| states.iter().map(|state| state.alpha_sum[k]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let vare = states.iter().map(|state| state.var_e_sum).sum::<f64>() * scale;
    let (h2_mean, var_h2, rhat_h2) = controller.posterior_h2_summary();
    let rhat_metrics = controller.posterior_rhat_values();
    Ok((
        beta,
        alpha,
        varbeta,
        vare,
        h2_mean,
        var_h2,
        rhat_h2,
        controller.actual_iterations(),
        controller.convergence_iteration(),
        n_keep,
        rhat_metrics,
    ))
}

struct BayesBChainState {
    rng: StdRng,
    beta: Vec<f64>,
    d: Vec<u8>,
    var_b: Vec<f64>,
    s: f64,
    prob_in: f64,
    var_e: f64,
    alpha: Vec<f64>,
    residual: Vec<f64>,
    marker_residual: Vec<f32>,
    alpha_xtr: Vec<f64>,
    alpha_delta: Vec<f64>,
    alpha_tmp_n: Vec<f64>,
    beta_sum: Vec<f64>,
    pip_sum: Vec<f64>,
    varb_sum: Vec<f64>,
    alpha_sum: Vec<f64>,
    var_e_sum: f64,
    prob_in_sum: f64,
    n_active_sum: f64,
}

impl BayesBChainState {
    fn new(
        fixed_init: &BayesFixedEffectInit,
        p: usize,
        q: usize,
        s0_b: f64,
        var_e: f64,
        prob_in: f64,
        df0_b: f64,
        start_scale: f64,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => {
                let mut seed_bytes = [0u8; 32];
                if OsRng.try_fill_bytes(&mut seed_bytes).is_err() {
                    seed_bytes = [42u8; 32];
                }
                StdRng::from_seed(seed_bytes)
            }
        };
        let initial_s = s0_b * start_scale;
        Self {
            rng,
            beta: vec![0.0; p],
            d: vec![0; p],
            var_b: vec![initial_s / (df0_b + 2.0); p],
            s: initial_s,
            prob_in,
            var_e,
            alpha: fixed_init.alpha.clone(),
            residual: fixed_init.residual.clone(),
            marker_residual: vec![0.0; fixed_init.residual.len()],
            alpha_xtr: vec![0.0; q],
            alpha_delta: vec![0.0; q],
            alpha_tmp_n: vec![0.0; fixed_init.residual.len()],
            beta_sum: vec![0.0; p],
            pip_sum: vec![0.0; p],
            varb_sum: vec![0.0; p],
            alpha_sum: vec![0.0; q],
            var_e_sum: 0.0,
            prob_in_sum: 0.0,
            n_active_sum: 0.0,
        }
    }

    fn clear_posterior(&mut self) {
        self.beta_sum.fill(0.0);
        self.pip_sum.fill(0.0);
        self.varb_sum.fill(0.0);
        self.alpha_sum.fill(0.0);
        self.var_e_sum = 0.0;
        self.prob_in_sum = 0.0;
        self.n_active_sum = 0.0;
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesb_lockstep_core_impl<B: BayesMarkerBackend>(
    backend: &mut B,
    y: &[f64],
    x: &[f64],
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    chain_pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<BayesBCResult, String> {
    let n = backend.n_samples();
    let p = backend.n_markers();
    if n <= 1 || y.len() != n {
        return Err("BayesB phenotype/genotype dimensions are incompatible".to_string());
    }
    if q == 0 || x.len() != n.saturating_mul(q) {
        return Err("BayesB covariate dimensions are incompatible".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) || counts < 0.0 {
        return Err("prob_in must be in (0, 1) and counts must be >= 0".to_string());
    }
    let (x2, _mean_x, msx) = marker_sufficient_stats(backend, n, p)?;
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("fixed pi must be in (0, 1)".to_string());
    }
    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, Some(prob_in_base), s0_b_opt)?;
    let rate0 = rate0_opt.unwrap_or_else(|| (shape0 - 1.0) / s0_b);
    if !(rate0.is_finite() && rate0 > 0.0) {
        return Err("rate0 must be positive; shape0 must be > 1".to_string());
    }
    let initial_var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_base, counts);
    let adaptive = sampling.adaptive_burnin || sampling.burnin.is_none();
    let seeds = bayes_chain_seeds(seed, chains);
    let mut states = seeds
        .into_iter()
        .enumerate()
        .map(|(chain_index, chain_seed)| {
            let start_scale = bayes_chain_start_scale(chain_index, chains, adaptive);
            let start_prob = if fixed_prob_in_opt.is_some() {
                prob_in_base
            } else {
                bayes_chain_start_probability(prob_in_base, start_scale)
            };
            BayesBChainState::new(
                &fixed_init,
                p,
                q,
                s0_b,
                initial_var_e,
                start_prob,
                df0_b,
                start_scale,
                chain_seed,
            )
        })
        .collect::<Vec<_>>();
    let mut controller = BayesMultiChainController::new_with_monitor_options(
        chains,
        n_iter,
        sampling.burnin,
        adaptive,
        thin,
        BAYES_RHAT_B_NAMES,
        BAYES_PREDICTIVE_MONITOR_NAMES,
        sampling.posterior_target,
    )?;
    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|error| error.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|error| error.to_string())?;
    let chi_e = ChiSquared::new(n as f64 + df0_e).map_err(|error| error.to_string())?;
    let var_b_fixed = 1.0e10_f64;

    while controller.should_run() {
        let retained = controller.begin_iteration();
        let collect = controller.collecting_posterior();
        bayes_chain_for_each_mut(&mut states, chain_pool, |state| {
            let inv_var_e = 1.0 / state.var_e;
            update_alpha_gauss_seidel_blas(
                x,
                n,
                q,
                inv_var_e,
                1.0 / var_b_fixed,
                &x2_x,
                &xtx,
                &mut state.alpha,
                &mut state.residual,
                &mut state.alpha_xtr,
                &mut state.alpha_delta,
                &mut state.alpha_tmp_n,
                &mut state.rng,
            );
            copy_f64_to_f32(&state.residual, &mut state.marker_residual);
        });
        backend.for_each_block(|row_start, row_end, block| {
            bayes_chain_try_for_each_mut(&mut states, chain_pool, |state| {
                let inv_var_e = 1.0 / state.var_e;
                let log_odds_prior = (state.prob_in / (1.0 - state.prob_in)).ln();
                for offset in 0..(row_end - row_start) {
                    let j = row_start + offset;
                    let marker = &block[offset * n..(offset + 1) * n];
                    let c = x2[j] * inv_var_e + 1.0 / state.var_b[j];
                    if !(c.is_finite() && c > 0.0) {
                        return Err("non-positive BayesB posterior precision".to_string());
                    }
                    let old_beta = state.beta[j];
                    let mut new_d = 0u8;
                    let (_, new_beta) = bayes_marker_update_f32(
                        &mut state.marker_residual,
                        marker,
                        old_beta,
                        x2[j],
                        |u| {
                            let rhs = u * inv_var_e;
                            let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (state.var_b[j] * c).ln();
                            let log_odds = log_odds_prior + log_bf10;
                            let p_in = if log_odds >= 0.0 {
                                1.0 / (1.0 + (-log_odds).exp())
                            } else {
                                let e = log_odds.exp();
                                e / (1.0 + e)
                            };
                            new_d = if state.rng.random::<f64>() < p_in {
                                1
                            } else {
                                0
                            };
                            if new_d == 1 {
                                let z: f64 = state.rng.sample(StandardNormal);
                                rhs / c + (1.0 / c).sqrt() * z
                            } else {
                                0.0
                            }
                        },
                    );
                    state.d[j] = new_d;
                    state.beta[j] = new_beta;
                }
                Ok(())
            })
        })?;

        let chain_summaries = bayes_chain_try_map_mut(&mut states, chain_pool, |state| {
            copy_f32_to_f64(&state.marker_residual, &mut state.residual);
            let mut n_active = 0usize;
            for j in 0..p {
                let distribution = if state.d[j] == 1 {
                    chi_b_active
                } else {
                    chi_b_inactive
                };
                let numerator = if state.d[j] == 1 {
                    n_active += 1;
                    state.s + state.beta[j] * state.beta[j]
                } else {
                    state.s
                };
                state.var_b[j] = bayes_positive_floor(numerator / state.rng.sample(distribution));
            }
            let tmp_rate = state.var_b.iter().map(|value| 1.0 / value).sum::<f64>() / 2.0 + rate0;
            let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
            state.s =
                bayes_positive_floor(sample_gamma_with_rate(&mut state.rng, tmp_shape, tmp_rate)?);
            let mrk_in = n_active as f64;
            if fixed_prob_in_opt.is_none() {
                let beta_dist = Beta::new(
                    mrk_in + counts_in + 1.0,
                    (p as f64 - mrk_in) + counts_out + 1.0,
                )
                .map_err(|error| error.to_string())?;
                state.prob_in = state.rng.sample(beta_dist);
            }
            state.var_e =
                (ddot_f64(&state.residual, &state.residual) + prior_ss_e) / state.rng.sample(chi_e);
            if !(state.var_e.is_finite() && state.var_e > 0.0) {
                return Err("BayesB var_e became non-finite or non-positive".to_string());
            }
            let var_g = genetic_variance_from_residual(y, &state.residual, x, &state.alpha, n, q);
            let h2_value = var_g / (var_g + state.var_e);
            if collect && retained {
                for j in 0..p {
                    state.beta_sum[j] += state.d[j] as f64 * state.beta[j];
                    state.pip_sum[j] += state.d[j] as f64;
                    state.varb_sum[j] += state.var_b[j];
                }
                for k in 0..q {
                    state.alpha_sum[k] += state.alpha[k];
                }
                state.var_e_sum += state.var_e;
                state.prob_in_sum += state.prob_in;
                state.n_active_sum += mrk_in;
            }
            Ok((
                vec![h2_value, var_g, state.var_e, state.prob_in, mrk_in],
                bayes_predictive_monitor_values(
                    y,
                    &state.residual,
                    x,
                    &state.alpha,
                    n,
                    q,
                    h2_value,
                    var_g,
                    state.var_e,
                ),
            ))
        })?;
        let (metrics, monitor_metrics): (Vec<_>, Vec<_>) = chain_summaries.into_iter().unzip();
        if controller.observe_with_monitor(&metrics, &monitor_metrics, retained)? {
            bayes_chain_for_each_mut(&mut states, chain_pool, |state| {
                state.clear_posterior();
            });
        }
    }
    let n_keep = controller.posterior_samples();
    if n_keep != controller.posterior_target() {
        return Err(format!(
            "BayesB retained {n_keep} posterior samples; expected {}",
            controller.posterior_target()
        ));
    }
    let scale = 1.0 / (n_keep * chains) as f64;
    let beta = (0..p)
        .map(|j| states.iter().map(|state| state.beta_sum[j]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let pip = (0..p)
        .map(|j| states.iter().map(|state| state.pip_sum[j]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let varbeta = (0..p)
        .map(|j| states.iter().map(|state| state.varb_sum[j]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let alpha = (0..q)
        .map(|k| states.iter().map(|state| state.alpha_sum[k]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let (h2_mean, var_h2, rhat_h2) = controller.posterior_h2_summary();
    let rhat_metrics = controller.posterior_rhat_values();
    Ok((
        beta,
        alpha,
        varbeta,
        states.iter().map(|state| state.var_e_sum).sum::<f64>() * scale,
        h2_mean,
        var_h2,
        states.iter().map(|state| state.prob_in_sum).sum::<f64>() * scale,
        states.iter().map(|state| state.n_active_sum).sum::<f64>() * scale,
        pip,
        rhat_h2,
        controller.actual_iterations(),
        controller.convergence_iteration(),
        n_keep,
        rhat_metrics,
    ))
}

struct BayesCChainState {
    rng: StdRng,
    beta: Vec<f64>,
    d: Vec<u8>,
    var_b: f64,
    prob_in: f64,
    var_e: f64,
    alpha: Vec<f64>,
    residual: Vec<f64>,
    marker_residual: Vec<f32>,
    alpha_xtr: Vec<f64>,
    alpha_delta: Vec<f64>,
    alpha_tmp_n: Vec<f64>,
    beta_sum: Vec<f64>,
    pip_sum: Vec<f64>,
    alpha_sum: Vec<f64>,
    varb_sum: f64,
    var_e_sum: f64,
    prob_in_sum: f64,
    n_active_sum: f64,
}

impl BayesCChainState {
    fn new(
        fixed_init: &BayesFixedEffectInit,
        p: usize,
        q: usize,
        s0_b: f64,
        var_e: f64,
        prob_in: f64,
        start_scale: f64,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => {
                let mut seed_bytes = [0u8; 32];
                if OsRng.try_fill_bytes(&mut seed_bytes).is_err() {
                    seed_bytes = [42u8; 32];
                }
                StdRng::from_seed(seed_bytes)
            }
        };
        let initial_var_b = s0_b * start_scale;
        Self {
            rng,
            beta: vec![0.0; p],
            d: vec![0; p],
            var_b: initial_var_b,
            prob_in,
            var_e,
            alpha: fixed_init.alpha.clone(),
            residual: fixed_init.residual.clone(),
            marker_residual: vec![0.0; fixed_init.residual.len()],
            alpha_xtr: vec![0.0; q],
            alpha_delta: vec![0.0; q],
            alpha_tmp_n: vec![0.0; fixed_init.residual.len()],
            beta_sum: vec![0.0; p],
            pip_sum: vec![0.0; p],
            alpha_sum: vec![0.0; q],
            varb_sum: 0.0,
            var_e_sum: 0.0,
            prob_in_sum: 0.0,
            n_active_sum: 0.0,
        }
    }

    fn clear_posterior(&mut self) {
        self.beta_sum.fill(0.0);
        self.pip_sum.fill(0.0);
        self.alpha_sum.fill(0.0);
        self.varb_sum = 0.0;
        self.var_e_sum = 0.0;
        self.prob_in_sum = 0.0;
        self.n_active_sum = 0.0;
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesc_lockstep_core_impl<B: BayesMarkerBackend>(
    backend: &mut B,
    y: &[f64],
    x: &[f64],
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    chain_pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<BayesCResult, String> {
    let n = backend.n_samples();
    let p = backend.n_markers();
    if n <= 1 || y.len() != n {
        return Err("BayesC phenotype/genotype dimensions are incompatible".to_string());
    }
    if q == 0 || x.len() != n.saturating_mul(q) {
        return Err("BayesC covariate dimensions are incompatible".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;
    if !(prob_in_init > 0.0 && prob_in_init < 1.0) || counts < 0.0 {
        return Err("prob_in must be in (0, 1) and counts must be >= 0".to_string());
    }
    let (x2, _mean_x, msx) = marker_sufficient_stats(backend, n, p)?;
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, Some(prob_in_base), s0_b_opt)?;
    let initial_var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_base, counts);
    let adaptive = sampling.adaptive_burnin || sampling.burnin.is_none();
    let seeds = bayes_chain_seeds(seed, chains);
    let mut states = seeds
        .into_iter()
        .enumerate()
        .map(|(chain_index, chain_seed)| {
            let start_scale = bayes_chain_start_scale(chain_index, chains, adaptive);
            let start_prob = if fixed_prob_in_opt.is_some() {
                prob_in_base
            } else {
                bayes_chain_start_probability(prob_in_base, start_scale)
            };
            BayesCChainState::new(
                &fixed_init,
                p,
                q,
                s0_b,
                initial_var_e,
                start_prob,
                start_scale,
                chain_seed,
            )
        })
        .collect::<Vec<_>>();
    let mut controller = BayesMultiChainController::new_with_monitor_options(
        chains,
        n_iter,
        sampling.burnin,
        adaptive,
        thin,
        BAYES_RHAT_C_NAMES,
        BAYES_PREDICTIVE_MONITOR_NAMES,
        sampling.posterior_target,
    )?;
    let chi_e = ChiSquared::new(n as f64 + df0_e).map_err(|error| error.to_string())?;
    let var_b_fixed = 1.0e10_f64;

    while controller.should_run() {
        let retained = controller.begin_iteration();
        let collect = controller.collecting_posterior();
        bayes_chain_for_each_mut(&mut states, chain_pool, |state| {
            let inv_var_e = 1.0 / state.var_e;
            update_alpha_gauss_seidel_blas(
                x,
                n,
                q,
                inv_var_e,
                1.0 / var_b_fixed,
                &x2_x,
                &xtx,
                &mut state.alpha,
                &mut state.residual,
                &mut state.alpha_xtr,
                &mut state.alpha_delta,
                &mut state.alpha_tmp_n,
                &mut state.rng,
            );
            copy_f64_to_f32(&state.residual, &mut state.marker_residual);
        });
        backend.for_each_block(|row_start, row_end, block| {
            bayes_chain_try_for_each_mut(&mut states, chain_pool, |state| {
                let inv_var_e = 1.0 / state.var_e;
                let log_odds_prior = (state.prob_in / (1.0 - state.prob_in)).ln();
                for offset in 0..(row_end - row_start) {
                    let j = row_start + offset;
                    let marker = &block[offset * n..(offset + 1) * n];
                    let c = x2[j] * inv_var_e + 1.0 / state.var_b;
                    if !(c.is_finite() && c > 0.0) {
                        return Err("non-positive BayesC posterior precision".to_string());
                    }
                    let old_beta = state.beta[j];
                    let mut new_d = 0u8;
                    let (_, new_beta) = bayes_marker_update_f32(
                        &mut state.marker_residual,
                        marker,
                        old_beta,
                        x2[j],
                        |u| {
                            let rhs = u * inv_var_e;
                            let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (state.var_b * c).ln();
                            let log_odds = log_odds_prior + log_bf10;
                            let p_in = if log_odds >= 0.0 {
                                1.0 / (1.0 + (-log_odds).exp())
                            } else {
                                let e = log_odds.exp();
                                e / (1.0 + e)
                            };
                            new_d = if state.rng.random::<f64>() < p_in {
                                1
                            } else {
                                0
                            };
                            if new_d == 1 {
                                let z: f64 = state.rng.sample(StandardNormal);
                                rhs / c + (1.0 / c).sqrt() * z
                            } else {
                                0.0
                            }
                        },
                    );
                    state.d[j] = new_d;
                    state.beta[j] = new_beta;
                }
                Ok(())
            })
        })?;
        let chain_summaries = bayes_chain_try_map_mut(&mut states, chain_pool, |state| {
            copy_f32_to_f64(&state.marker_residual, &mut state.residual);
            let mut active = 0usize;
            let mut ss_b = 0.0;
            for j in 0..p {
                if state.d[j] == 1 {
                    active += 1;
                    ss_b += state.beta[j] * state.beta[j];
                }
            }
            let chi_b =
                ChiSquared::new(df0_b + active as f64).map_err(|error| error.to_string())?;
            state.var_b = (ss_b + s0_b) / state.rng.sample(chi_b);
            if fixed_prob_in_opt.is_none() {
                let mrk_in = active as f64;
                let beta_dist = Beta::new(
                    mrk_in + counts_in + 1.0,
                    (p as f64 - mrk_in) + counts_out + 1.0,
                )
                .map_err(|error| error.to_string())?;
                state.prob_in = state.rng.sample(beta_dist);
            }
            state.var_e =
                (ddot_f64(&state.residual, &state.residual) + prior_ss_e) / state.rng.sample(chi_e);
            if !(state.var_e.is_finite() && state.var_e > 0.0) {
                return Err("BayesC var_e became non-finite or non-positive".to_string());
            }
            let var_g = genetic_variance_from_residual(y, &state.residual, x, &state.alpha, n, q);
            let h2_value = var_g / (var_g + state.var_e);
            if collect && retained {
                for j in 0..p {
                    state.beta_sum[j] += state.d[j] as f64 * state.beta[j];
                    state.pip_sum[j] += state.d[j] as f64;
                }
                for k in 0..q {
                    state.alpha_sum[k] += state.alpha[k];
                }
                state.varb_sum += state.var_b;
                state.var_e_sum += state.var_e;
                state.prob_in_sum += state.prob_in;
                state.n_active_sum += active as f64;
            }
            Ok((
                vec![
                    h2_value,
                    var_g,
                    state.var_e,
                    state.var_b,
                    state.prob_in,
                    active as f64,
                ],
                bayes_predictive_monitor_values(
                    y,
                    &state.residual,
                    x,
                    &state.alpha,
                    n,
                    q,
                    h2_value,
                    var_g,
                    state.var_e,
                ),
            ))
        })?;
        let (metrics, monitor_metrics): (Vec<_>, Vec<_>) = chain_summaries.into_iter().unzip();
        if controller.observe_with_monitor(&metrics, &monitor_metrics, retained)? {
            bayes_chain_for_each_mut(&mut states, chain_pool, |state| {
                state.clear_posterior();
            });
        }
    }
    let n_keep = controller.posterior_samples();
    if n_keep != controller.posterior_target() {
        return Err(format!(
            "BayesC retained {n_keep} posterior samples; expected {}",
            controller.posterior_target()
        ));
    }
    let scale = 1.0 / (n_keep * chains) as f64;
    let beta = (0..p)
        .map(|j| states.iter().map(|state| state.beta_sum[j]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let pip = (0..p)
        .map(|j| states.iter().map(|state| state.pip_sum[j]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let alpha = (0..q)
        .map(|k| states.iter().map(|state| state.alpha_sum[k]).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let (h2_mean, var_h2, rhat_h2) = controller.posterior_h2_summary();
    let rhat_metrics = controller.posterior_rhat_values();
    Ok((
        beta,
        alpha,
        states.iter().map(|state| state.varb_sum).sum::<f64>() * scale,
        states.iter().map(|state| state.var_e_sum).sum::<f64>() * scale,
        h2_mean,
        var_h2,
        states.iter().map(|state| state.prob_in_sum).sum::<f64>() * scale,
        states.iter().map(|state| state.n_active_sum).sum::<f64>() * scale,
        pip,
        rhat_h2,
        controller.actual_iterations(),
        controller.convergence_iteration(),
        n_keep,
        rhat_metrics,
    ))
}

#[allow(clippy::too_many_arguments)]
fn bayesa_multi_core_impl(
    y: &[f64],
    m: &[f32],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    min_abs_beta: f64,
    seed: Option<u64>,
    chains: usize,
    threads: usize,
) -> Result<BayesAResult, String> {
    let effective = effective_bayes_chains(chains, threads)?;
    if effective > 1 {
        let mut backend = DenseBayesBackend::new(m, n, p)?;
        let chain_pool = build_bayes_chain_pool(threads)?;
        return bayesa_lockstep_core_impl(
            &mut backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
            effective,
            chain_pool.as_ref(),
        );
    }
    let seeds = bayes_chain_seeds(seed, effective);
    let mut results = Vec::with_capacity(effective);
    for chain_seed in seeds {
        results.push(bayesa_core_impl(
            y,
            m,
            x,
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            df0_e,
            prior_ss_e_opt,
            min_abs_beta,
            chain_seed,
        )?);
    }
    Ok(aggregate_bayes_a_results(&results))
}

#[allow(clippy::too_many_arguments)]
fn bayesb_multi_core_impl(
    y: &[f64],
    m: &[f32],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    threads: usize,
) -> Result<BayesBCResult, String> {
    let effective = effective_bayes_chains(chains, threads)?;
    if effective > 1 {
        let mut backend = DenseBayesBackend::new(m, n, p)?;
        let chain_pool = build_bayes_chain_pool(threads)?;
        return bayesb_lockstep_core_impl(
            &mut backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
            effective,
            chain_pool.as_ref(),
        );
    }
    let seeds = bayes_chain_seeds(seed, effective);
    let mut results = Vec::with_capacity(effective);
    for chain_seed in seeds {
        results.push(bayesb_core_impl(
            y,
            m,
            x,
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            chain_seed,
        )?);
    }
    Ok(aggregate_bayes_bc_results(&results))
}

#[allow(clippy::too_many_arguments)]
fn bayesc_multi_core_impl(
    y: &[f64],
    m: &[f32],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    threads: usize,
) -> Result<BayesCResult, String> {
    let effective = effective_bayes_chains(chains, threads)?;
    if effective > 1 {
        let mut backend = DenseBayesBackend::new(m, n, p)?;
        let chain_pool = build_bayes_chain_pool(threads)?;
        return bayesc_lockstep_core_impl(
            &mut backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
            effective,
            chain_pool.as_ref(),
        );
    }
    let seeds = bayes_chain_seeds(seed, effective);
    let mut results = Vec::with_capacity(effective);
    for chain_seed in seeds {
        results.push(bayesc_core_impl(
            y,
            m,
            x,
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            chain_seed,
        )?);
    }
    Ok(aggregate_bayes_c_results(&results))
}

fn bayesa_packed_core_impl(
    y: &[f64],
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    _min_abs_beta: f64,
    seed: Option<u64>,
    backend_block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<BayesAResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if q == 0 || x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let mut backend = PackedBayesBackend::new(
        source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        sample_idx,
        backend_block_rows,
        pool,
    )?;
    if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
        return bayesa_core_impl(
            y,
            &m_dense,
            x,
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            df0_e,
            prior_ss_e_opt,
            0.0,
            seed,
        );
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, None, s0_b_opt)?;

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut schedule = BayesSamplingController::new_with_options(
        n_iter,
        sampling.burnin,
        sampling.adaptive_burnin,
        thin,
        BAYES_RHAT_A_NAMES,
        sampling.posterior_target,
    );
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                let z_beta: f64 = rng.sample(StandardNormal);
                let old_beta = beta[j];
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, old_beta, x2[j], |u| {
                        u * inv_var_e / c + (1.0 / c).sqrt() * z_beta
                    });
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
            if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                return Err("BayesA packed var_b became non-finite or non-positive".to_string());
            }
        }

        let mut tmp_rate = 0.0;
        for j in 0..p {
            tmp_rate += 1.0 / var_b[j];
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesA packed S became non-finite or non-positive".to_string());
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesA packed var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                beta_sum[j] += beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            let rhat_metrics = [h2, var_g, var_e];
            if schedule.observe(&rhat_metrics) {
                beta_sum.fill(0.0);
                varb_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let rhat_h2 = schedule.rhat_value();
    let rhat_metrics = schedule.rhat_values();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();
    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
        rhat_metrics,
    ))
}

fn bayesb_packed_core_impl(
    y: &[f64],
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    backend_block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<BayesBCResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if q == 0 || x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let mut backend = PackedBayesBackend::new(
        source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        sample_idx,
        backend_block_rows,
        pool,
    )?;
    if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
        return bayesb_core_impl(
            y,
            &m_dense,
            x,
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
        );
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, Some(prob_in_base), s0_b_opt)?;

    let rate0 = bayesb_rate0(shape0, rate0_opt, s0_b)?;

    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut prob_in = prob_in_base;
    let mut s = s0_b;
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_base, counts);

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut pip_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut schedule = BayesSamplingController::new_with_options(
        n_iter,
        sampling.burnin,
        sampling.adaptive_burnin,
        thin,
        BAYES_RHAT_B_NAMES,
        sampling.posterior_target,
    );
    let var_b_fixed = 1e10_f64;

    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesB packed beta update".to_string(),
                    );
                }
                let mut new_d = 0u8;
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, b_old, x2[j], |xe| {
                        let rhs = xe * inv_var_e;
                        // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                        let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b[j] * c).ln();
                        let log_odds = log_odds_prior + log_bf10;
                        let p_in = if log_odds >= 0.0 {
                            1.0 / (1.0 + (-log_odds).exp())
                        } else {
                            let e = log_odds.exp();
                            e / (1.0 + e)
                        };
                        new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                        if new_d == 1 {
                            let z_beta: f64 = rng.sample(StandardNormal);
                            rhs / c + (1.0 / c).sqrt() * z_beta
                        } else {
                            0.0
                        }
                    });
                d[j] = new_d;
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut n_active = 0usize;
        for j in 0..p {
            if d[j] == 1 {
                let beta2 = beta[j] * beta[j];
                var_b[j] = bayes_positive_floor((s + beta2) / rng.sample(chi_b_active));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err("BayesB packed var_b became non-finite or non-positive".to_string());
                }
                n_active += 1;
            } else {
                var_b[j] = bayes_positive_floor(s / rng.sample(chi_b_inactive));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err(
                        "BayesB packed inactive var_b became non-finite or non-positive"
                            .to_string(),
                    );
                }
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err(
                "Gamma rate became non-positive while updating BayesB packed S".to_string(),
            );
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        s = bayes_positive_floor(sample_gamma_with_rate(&mut rng, tmp_shape, tmp_rate)?);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesB packed S became non-finite or non-positive".to_string());
        }

        let mrk_in = n_active as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesB packed var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                pip_sum[j] += d[j] as f64;
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            let rhat_metrics = [h2, var_g, var_e, prob_in, mrk_in];
            if schedule.observe(&rhat_metrics) {
                beta_sum.fill(0.0);
                pip_sum.fill(0.0);
                varb_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
                prob_in_sum = 0.0;
                n_active_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        pip_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;
    let rhat_h2 = schedule.rhat_value();
    let rhat_metrics = schedule.rhat_values();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        pip_sum,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
        rhat_metrics,
    ))
}

fn bayesc_packed_core_impl(
    y: &[f64],
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    backend_block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<BayesCResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if q == 0 || x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let mut backend = PackedBayesBackend::new(
        source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        sample_idx,
        backend_block_rows,
        pool,
    )?;
    if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
        return bayesc_core_impl(
            y,
            &m_dense,
            x,
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
        );
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, Some(prob_in_base), s0_b_opt)?;

    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = s0_b;
    let mut prob_in = prob_in_base;
    let counts_in = counts * prob_in_base;
    let counts_out = counts - counts_in;

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut pip_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut varb_sum = 0.0;
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut schedule = BayesSamplingController::new_with_options(
        n_iter,
        sampling.burnin,
        sampling.adaptive_burnin,
        thin,
        BAYES_RHAT_C_NAMES,
        sampling.posterior_target,
    );
    let var_b_fixed = 1e10_f64;

    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;
    let mut chi_b_cache: HashMap<usize, ChiSquared<f64>> = HashMap::new();

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                let c = x2[j] * inv_var_e + 1.0 / var_b;
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesC packed beta update".to_string(),
                    );
                }
                let mut new_d = 0u8;
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, b_old, x2[j], |xe| {
                        let rhs = xe * inv_var_e;
                        // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                        let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b * c).ln();
                        let log_odds = log_odds_prior + log_bf10;
                        let p_in = if log_odds >= 0.0 {
                            1.0 / (1.0 + (-log_odds).exp())
                        } else {
                            let e = log_odds.exp();
                            e / (1.0 + e)
                        };
                        new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                        if new_d == 1 {
                            let z_beta: f64 = rng.sample(StandardNormal);
                            rhs / c + (1.0 / c).sqrt() * z_beta
                        } else {
                            0.0
                        }
                    });
                d[j] = new_d;
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut mrk_in_usize = 0usize;
        let mut ss_b = 0.0;
        for j in 0..p {
            if d[j] == 1 {
                ss_b += beta[j] * beta[j];
                mrk_in_usize += 1;
            }
        }
        ss_b += s0_b;
        let chi_b_eff = if let Some(dist) = chi_b_cache.get(&mrk_in_usize) {
            *dist
        } else {
            let dist = ChiSquared::new(df0_b + mrk_in_usize as f64).map_err(|e| e.to_string())?;
            chi_b_cache.insert(mrk_in_usize, dist);
            dist
        };
        var_b = ss_b / rng.sample(chi_b_eff);
        if !(var_b.is_finite() && var_b > 0.0) {
            return Err("BayesC packed var_b became non-finite or non-positive".to_string());
        }

        let mrk_in = mrk_in_usize as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesC packed var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                pip_sum[j] += d[j] as f64;
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            varb_sum += var_b;
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            let rhat_metrics = [h2, var_g, var_e, var_b, prob_in, mrk_in];
            if schedule.observe(&rhat_metrics) {
                beta_sum.fill(0.0);
                pip_sum.fill(0.0);
                alpha_sum.fill(0.0);
                varb_sum = 0.0;
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
                prob_in_sum = 0.0;
                n_active_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        pip_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    varb_sum *= inv_keep;
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;
    let rhat_h2 = schedule.rhat_value();
    let rhat_metrics = schedule.rhat_values();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        pip_sum,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
        rhat_metrics,
    ))
}

pub(crate) fn duplicate_bayes_source<'a>(
    source: &BayesPackedSource<'a>,
) -> Result<BayesPackedSource<'a>, String> {
    match source {
        BayesPackedSource::Resident {
            packed_flat,
            bytes_per_snp,
        } => Ok(BayesPackedSource::Resident {
            packed_flat,
            bytes_per_snp: *bytes_per_snp,
        }),
        BayesPackedSource::Windowed(matrix) => Ok(BayesPackedSource::Windowed(matrix.duplicate()?)),
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesa_packed_multi_core_impl<'a>(
    y: &[f64],
    source: &mut BayesPackedSource<'a>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    min_abs_beta: f64,
    seed: Option<u64>,
    backend_block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
    chains: usize,
    threads: usize,
) -> Result<BayesAResult, String> {
    let effective = effective_bayes_chains(chains, threads)?;
    if effective > 1 {
        let mut backend = PackedBayesBackend::new(
            source,
            n_samples,
            p,
            row_flip,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_idx,
            backend_block_rows,
            pool,
        )?;
        if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
            let mut dense_backend = DenseBayesBackend::new(&m_dense, n, p)?;
            return bayesa_lockstep_core_impl(
                &mut dense_backend,
                y,
                x,
                q,
                n_iter,
                sampling,
                thin,
                r2,
                df0_b,
                shape0,
                rate0_opt,
                s0_b_opt,
                df0_e,
                prior_ss_e_opt,
                seed,
                effective,
                pool,
            );
        }
        return bayesa_lockstep_core_impl(
            &mut backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
            effective,
            pool,
        );
    }
    let seeds = bayes_chain_seeds(seed, effective);
    let mut results = Vec::with_capacity(effective);
    for chain_seed in seeds {
        let mut chain_source = duplicate_bayes_source(source)?;
        results.push(bayesa_packed_core_impl(
            y,
            &mut chain_source,
            n_samples,
            row_flip,
            row_maf,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_idx,
            x,
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            df0_e,
            prior_ss_e_opt,
            min_abs_beta,
            chain_seed,
            backend_block_rows,
            pool,
        )?);
    }
    Ok(aggregate_bayes_a_results(&results))
}

#[allow(clippy::too_many_arguments)]
fn bayesb_packed_multi_core_impl<'a>(
    y: &[f64],
    source: &mut BayesPackedSource<'a>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    backend_block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
    chains: usize,
    threads: usize,
) -> Result<BayesBCResult, String> {
    let effective = effective_bayes_chains(chains, threads)?;
    if effective > 1 {
        let mut backend = PackedBayesBackend::new(
            source,
            n_samples,
            p,
            row_flip,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_idx,
            backend_block_rows,
            pool,
        )?;
        if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
            let mut dense_backend = DenseBayesBackend::new(&m_dense, n, p)?;
            return bayesb_lockstep_core_impl(
                &mut dense_backend,
                y,
                x,
                q,
                n_iter,
                sampling,
                thin,
                r2,
                df0_b,
                shape0,
                rate0_opt,
                s0_b_opt,
                prob_in_init,
                counts,
                fixed_prob_in_opt,
                df0_e,
                prior_ss_e_opt,
                seed,
                effective,
                pool,
            );
        }
        return bayesb_lockstep_core_impl(
            &mut backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
            effective,
            pool,
        );
    }
    let seeds = bayes_chain_seeds(seed, effective);
    let mut results = Vec::with_capacity(effective);
    for chain_seed in seeds {
        let mut chain_source = duplicate_bayes_source(source)?;
        results.push(bayesb_packed_core_impl(
            y,
            &mut chain_source,
            n_samples,
            row_flip,
            row_maf,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_idx,
            x,
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            chain_seed,
            backend_block_rows,
            pool,
        )?);
    }
    Ok(aggregate_bayes_bc_results(&results))
}

#[allow(clippy::too_many_arguments)]
fn bayesc_packed_multi_core_impl<'a>(
    y: &[f64],
    source: &mut BayesPackedSource<'a>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    sampling: BayesSamplingOptions,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    backend_block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
    chains: usize,
    threads: usize,
) -> Result<BayesCResult, String> {
    let effective = effective_bayes_chains(chains, threads)?;
    if effective > 1 {
        let mut backend = PackedBayesBackend::new(
            source,
            n_samples,
            p,
            row_flip,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_idx,
            backend_block_rows,
            pool,
        )?;
        if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
            let mut dense_backend = DenseBayesBackend::new(&m_dense, n, p)?;
            return bayesc_lockstep_core_impl(
                &mut dense_backend,
                y,
                x,
                q,
                n_iter,
                sampling,
                thin,
                r2,
                df0_b,
                s0_b_opt,
                prob_in_init,
                counts,
                fixed_prob_in_opt,
                df0_e,
                prior_ss_e_opt,
                seed,
                effective,
                pool,
            );
        }
        return bayesc_lockstep_core_impl(
            &mut backend,
            y,
            x,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
            effective,
            pool,
        );
    }
    let seeds = bayes_chain_seeds(seed, effective);
    let mut results = Vec::with_capacity(effective);
    for chain_seed in seeds {
        let mut chain_source = duplicate_bayes_source(source)?;
        results.push(bayesc_packed_core_impl(
            y,
            &mut chain_source,
            n_samples,
            row_flip,
            row_maf,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_idx,
            x,
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            chain_seed,
            backend_block_rows,
            pool,
        )?);
    }
    Ok(aggregate_bayes_c_results(&results))
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    prior_ss_e = None,
    min_abs_beta = 1e-9,
    seed = None,
    chains = 1,
    threads = 1,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesa(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: Bound<'_, PyAny>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    min_abs_beta: f64,
    seed: Option<u64>,
    chains: usize,
    threads: usize,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Py<PyTuple>> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_input = array2_to_f32_input(&m, "M")?;
    if m_input.cols() != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_input.rows();
    let m_slice = m_input.as_slice();
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesa_multi_core_impl(
            y_vec.as_ref(),
            m_slice,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            prior_ss_e,
            min_abs_beta,
            seed,
            chains,
            threads,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            posterior_samples,
            rhat_metrics,
        )) => {
            let rhat_py = rhat_metrics_to_py(py, BAYES_RHAT_A_NAMES, &rhat_metrics)?;
            py_tuple_from_items(
                py,
                vec![
                    beta.into_pyarray(py).into_bound().into_any().unbind(),
                    alpha.into_pyarray(py).into_bound().into_any().unbind(),
                    varb.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, vare),
                    py_float_item(py, h2_mean),
                    py_float_item(py, var_h2),
                    py_float_item(py, rhat_h2),
                    py_usize_item(py, actual_iterations),
                    py_usize_item(py, convergence_iteration),
                    py_usize_item(py, posterior_samples),
                    rhat_py,
                ],
            )
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.05,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    seed = None,
    chains = 1,
    threads = 1,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesb(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: Bound<'_, PyAny>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    threads: usize,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Py<PyTuple>> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }

    // BayesB uses a fixed inclusion prior.  ``fixed_pi`` remains an
    // explicit override for callers, while the public API default pins it
    // to ``prob_in`` instead of enabling the Gibbs pi update.
    let fixed_pi = fixed_pi.or(Some(prob_in));
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_input = array2_to_f32_input(&m, "M")?;
    if m_input.cols() != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_input.rows();
    let m_slice = m_input.as_slice();

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesb_multi_core_impl(
            y_vec.as_ref(),
            m_slice,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            chains,
            threads,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
            rhat_metrics,
        )) => {
            let rhat_py = rhat_metrics_to_py(py, BAYES_RHAT_B_NAMES, &rhat_metrics)?;
            py_tuple_from_items(
                py,
                vec![
                    beta.into_pyarray(py).into_bound().into_any().unbind(),
                    alpha.into_pyarray(py).into_bound().into_any().unbind(),
                    varb.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, vare),
                    py_float_item(py, h2_mean),
                    py_float_item(py, var_h2),
                    py_float_item(py, prob_in_mean),
                    py_float_item(py, n_active_mean),
                    pip.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, rhat_h2),
                    py_usize_item(py, actual_iterations),
                    py_usize_item(py, convergence_iteration),
                    rhat_py,
                ],
            )
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.05,
    counts = 5.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    seed = None,
    chains = 1,
    threads = 1,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesc(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: Bound<'_, PyAny>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    seed: Option<u64>,
    chains: usize,
    threads: usize,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Py<PyTuple>> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = prior_ss_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("prior_ss_e must be > 0"));
        }
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_input = array2_to_f32_input(&m, "M")?;
    if m_input.cols() != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_input.rows();
    let m_slice = m_input.as_slice();
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesc_multi_core_impl(
            y_vec.as_ref(),
            m_slice,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            chains,
            threads,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb_mean,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
            rhat_metrics,
        )) => {
            let rhat_py = rhat_metrics_to_py(py, BAYES_RHAT_C_NAMES, &rhat_metrics)?;
            py_tuple_from_items(
                py,
                vec![
                    beta.into_pyarray(py).into_bound().into_any().unbind(),
                    alpha.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, varb_mean),
                    py_float_item(py, vare),
                    py_float_item(py, h2_mean),
                    py_float_item(py, var_h2),
                    py_float_item(py, prob_in_mean),
                    py_float_item(py, n_active_mean),
                    pip.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, rhat_h2),
                    py_usize_item(py, actual_iterations),
                    py_usize_item(py, convergence_iteration),
                    rhat_py,
                ],
            )
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    prior_ss_e = None,
    min_abs_beta = 1e-9,
    threads = 0,
    chains = 1,
    seed = None,
    block_rows = None,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesa_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    min_abs_beta: f64,
    threads: usize,
    chains: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Py<PyTuple>> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_flip.as_array(),
        )),
    };
    let row_maf_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_maf) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_maf.as_array(),
        )),
    };
    let row_mean_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_mean) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_mean.as_array(),
        )),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> =
        match crate::stats_common::array_standard_slice(&row_inv_sd) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                &row_inv_sd.as_array(),
            )),
        };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);

    let result = py.detach(|| {
        let mut source = BayesPackedSource::Resident {
            packed_flat: packed_flat.as_ref(),
            bytes_per_snp,
        };
        bayesa_packed_multi_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            None,
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            prior_ss_e,
            min_abs_beta,
            seed,
            block_rows,
            pool_ref,
            chains,
            threads,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            posterior_samples,
            rhat_metrics,
        )) => {
            let rhat_py = rhat_metrics_to_py(py, BAYES_RHAT_A_NAMES, &rhat_metrics)?;
            py_tuple_from_items(
                py,
                vec![
                    beta.into_pyarray(py).into_bound().into_any().unbind(),
                    alpha.into_pyarray(py).into_bound().into_any().unbind(),
                    varb.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, vare),
                    py_float_item(py, h2_mean),
                    py_float_item(py, var_h2),
                    py_float_item(py, rhat_h2),
                    py_usize_item(py, actual_iterations),
                    py_usize_item(py, convergence_iteration),
                    py_usize_item(py, posterior_samples),
                    rhat_py,
                ],
            )
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.05,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    threads = 0,
    chains = 1,
    seed = None,
    block_rows = None,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesb_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    threads: usize,
    chains: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Py<PyTuple>> {
    // BayesB always fixes pi; an omitted override means use the prior
    // inclusion probability supplied in ``prob_in``.
    let fixed_pi = fixed_pi.or(Some(prob_in));
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_flip.as_array(),
        )),
    };
    let row_maf_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_maf) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_maf.as_array(),
        )),
    };
    let row_mean_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_mean) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_mean.as_array(),
        )),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> =
        match crate::stats_common::array_standard_slice(&row_inv_sd) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                &row_inv_sd.as_array(),
            )),
        };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);

    let result = py.detach(|| {
        let mut source = BayesPackedSource::Resident {
            packed_flat: packed_flat.as_ref(),
            bytes_per_snp,
        };
        bayesb_packed_multi_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            None,
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            block_rows,
            pool_ref,
            chains,
            threads,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
            rhat_metrics,
        )) => {
            let rhat_py = rhat_metrics_to_py(py, BAYES_RHAT_B_NAMES, &rhat_metrics)?;
            py_tuple_from_items(
                py,
                vec![
                    beta.into_pyarray(py).into_bound().into_any().unbind(),
                    alpha.into_pyarray(py).into_bound().into_any().unbind(),
                    varb.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, vare),
                    py_float_item(py, h2_mean),
                    py_float_item(py, var_h2),
                    py_float_item(py, prob_in_mean),
                    py_float_item(py, n_active_mean),
                    pip.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, rhat_h2),
                    py_usize_item(py, actual_iterations),
                    py_usize_item(py, convergence_iteration),
                    rhat_py,
                ],
            )
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.05,
    counts = 5.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    threads = 0,
    chains = 1,
    seed = None,
    block_rows = None,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesc_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    threads: usize,
    chains: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Py<PyTuple>> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = prior_ss_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("prior_ss_e must be > 0"));
        }
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_flip.as_array(),
        )),
    };
    let row_maf_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_maf) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_maf.as_array(),
        )),
    };
    let row_mean_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_mean) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_mean.as_array(),
        )),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> =
        match crate::stats_common::array_standard_slice(&row_inv_sd) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                &row_inv_sd.as_array(),
            )),
        };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);

    let result = py.detach(|| {
        let mut source = BayesPackedSource::Resident {
            packed_flat: packed_flat.as_ref(),
            bytes_per_snp,
        };
        bayesc_packed_multi_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            None,
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            block_rows,
            pool_ref,
            chains,
            threads,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb_mean,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
            rhat_metrics,
        )) => {
            let rhat_py = rhat_metrics_to_py(py, BAYES_RHAT_C_NAMES, &rhat_metrics)?;
            py_tuple_from_items(
                py,
                vec![
                    beta.into_pyarray(py).into_bound().into_any().unbind(),
                    alpha.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, varb_mean),
                    py_float_item(py, vare),
                    py_float_item(py, h2_mean),
                    py_float_item(py, var_h2),
                    py_float_item(py, prob_in_mean),
                    py_float_item(py, n_active_mean),
                    pip.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, rhat_h2),
                    py_usize_item(py, actual_iterations),
                    py_usize_item(py, convergence_iteration),
                    rhat_py,
                ],
            )
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    n_samples,
    row_indices,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    prior_ss_e = None,
    min_abs_beta = 1e-9,
    threads = 0,
    chains = 1,
    seed = None,
    block_rows = None,
    mmap_window_mb = None,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesa_stream_bed(
    py: Python,
    prefix: String,
    y: PyReadonlyArray1<f64>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<i64>,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    min_abs_beta: f64,
    threads: usize,
    chains: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    mmap_window_mb: Option<usize>,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Py<PyTuple>> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let row_idx_raw = row_indices.as_slice()?.to_vec();
    let p = row_idx_raw.len();
    if p == 0 {
        return Err(PyValueError::new_err("row_indices must not be empty"));
    }
    let row_flip_vec: Cow<'_, [bool]> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_flip.as_array(),
        )),
    };
    let row_maf_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_maf) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_maf.as_array(),
        )),
    };
    let row_mean_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_mean) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_mean.as_array(),
        )),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> =
        match crate::stats_common::array_standard_slice(&row_inv_sd) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                &row_inv_sd.as_array(),
            )),
        };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match row_indices",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);
    let result = py.detach(|| {
        let block_rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(n, p))
            .max(1);
        let mut source = build_bayes_source(None, &prefix, n_samples, block_rows, mmap_window_mb)?;
        let n_source = match &source {
            BayesPackedSource::Resident { .. } => 0usize,
            BayesPackedSource::Windowed(matrix) => matrix.n_source_snps(),
        };
        let packed_row_indices =
            parse_index_vec_i64_string(row_idx_raw.as_slice(), n_source, "row_indices")?;
        bayesa_packed_multi_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            Some(packed_row_indices.as_slice()),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            prior_ss_e,
            min_abs_beta,
            seed,
            Some(block_rows),
            pool_ref,
            chains,
            threads,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            posterior_samples,
            rhat_metrics,
        )) => {
            let rhat_py = rhat_metrics_to_py(py, BAYES_RHAT_A_NAMES, &rhat_metrics)?;
            py_tuple_from_items(
                py,
                vec![
                    beta.into_pyarray(py).into_bound().into_any().unbind(),
                    alpha.into_pyarray(py).into_bound().into_any().unbind(),
                    varb.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, vare),
                    py_float_item(py, h2_mean),
                    py_float_item(py, var_h2),
                    py_float_item(py, rhat_h2),
                    py_usize_item(py, actual_iterations),
                    py_usize_item(py, convergence_iteration),
                    py_usize_item(py, posterior_samples),
                    rhat_py,
                ],
            )
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    n_samples,
    row_indices,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.05,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    threads = 0,
    chains = 1,
    seed = None,
    block_rows = None,
    mmap_window_mb = None,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesb_stream_bed(
    py: Python,
    prefix: String,
    y: PyReadonlyArray1<f64>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<i64>,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    threads: usize,
    chains: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    mmap_window_mb: Option<usize>,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Py<PyTuple>> {
    // Keep streamed BED behavior identical to dense and resident packed
    // BayesB: pi is a fixed prior, never a sampled state variable.
    let fixed_pi = fixed_pi.or(Some(prob_in));
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let row_idx_raw = row_indices.as_slice()?.to_vec();
    let p = row_idx_raw.len();
    if p == 0 {
        return Err(PyValueError::new_err("row_indices must not be empty"));
    }
    let row_flip_vec: Cow<'_, [bool]> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_flip.as_array(),
        )),
    };
    let row_maf_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_maf) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_maf.as_array(),
        )),
    };
    let row_mean_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_mean) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_mean.as_array(),
        )),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> =
        match crate::stats_common::array_standard_slice(&row_inv_sd) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                &row_inv_sd.as_array(),
            )),
        };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match row_indices",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);
    let result = py.detach(|| {
        let block_rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(n, p))
            .max(1);
        let mut source = build_bayes_source(None, &prefix, n_samples, block_rows, mmap_window_mb)?;
        let n_source = match &source {
            BayesPackedSource::Resident { .. } => 0usize,
            BayesPackedSource::Windowed(matrix) => matrix.n_source_snps(),
        };
        let packed_row_indices =
            parse_index_vec_i64_string(row_idx_raw.as_slice(), n_source, "row_indices")?;
        bayesb_packed_multi_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            Some(packed_row_indices.as_slice()),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            Some(block_rows),
            pool_ref,
            chains,
            threads,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
            rhat_metrics,
        )) => {
            let rhat_py = rhat_metrics_to_py(py, BAYES_RHAT_B_NAMES, &rhat_metrics)?;
            py_tuple_from_items(
                py,
                vec![
                    beta.into_pyarray(py).into_bound().into_any().unbind(),
                    alpha.into_pyarray(py).into_bound().into_any().unbind(),
                    varb.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, vare),
                    py_float_item(py, h2_mean),
                    py_float_item(py, var_h2),
                    py_float_item(py, prob_in_mean),
                    py_float_item(py, n_active_mean),
                    pip.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, rhat_h2),
                    py_usize_item(py, actual_iterations),
                    py_usize_item(py, convergence_iteration),
                    rhat_py,
                ],
            )
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    n_samples,
    row_indices,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 10000,
    burnin = None,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.05,
    counts = 5.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    threads = 0,
    chains = 1,
    seed = None,
    block_rows = None,
    mmap_window_mb = None,
    adaptive_burnin = false,
    posterior_samples = 1000
))]
pub fn bayesc_stream_bed(
    py: Python,
    prefix: String,
    y: PyReadonlyArray1<f64>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<i64>,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: Option<usize>,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    threads: usize,
    chains: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    mmap_window_mb: Option<usize>,
    adaptive_burnin: bool,
    posterior_samples: usize,
) -> PyResult<Py<PyTuple>> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if posterior_samples == 0 {
        return Err(PyValueError::new_err("posterior_samples must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = prior_ss_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("prior_ss_e must be > 0"));
        }
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let row_idx_raw = row_indices.as_slice()?.to_vec();
    let p = row_idx_raw.len();
    if p == 0 {
        return Err(PyValueError::new_err("row_indices must not be empty"));
    }
    let row_flip_vec: Cow<'_, [bool]> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_flip.as_array(),
        )),
    };
    let row_maf_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_maf) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_maf.as_array(),
        )),
    };
    let row_mean_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_mean) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_mean.as_array(),
        )),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> =
        match crate::stats_common::array_standard_slice(&row_inv_sd) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                &row_inv_sd.as_array(),
            )),
        };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match row_indices",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let sampling = BayesSamplingOptions::new(burnin, adaptive_burnin, posterior_samples);
    let result = py.detach(|| {
        let block_rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(n, p))
            .max(1);
        let mut source = build_bayes_source(None, &prefix, n_samples, block_rows, mmap_window_mb)?;
        let n_source = match &source {
            BayesPackedSource::Resident { .. } => 0usize,
            BayesPackedSource::Windowed(matrix) => matrix.n_source_snps(),
        };
        let packed_row_indices =
            parse_index_vec_i64_string(row_idx_raw.as_slice(), n_source, "row_indices")?;
        bayesc_packed_multi_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            Some(packed_row_indices.as_slice()),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            sampling,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            Some(block_rows),
            pool_ref,
            chains,
            threads,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb_mean,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
            rhat_metrics,
        )) => {
            let rhat_py = rhat_metrics_to_py(py, BAYES_RHAT_C_NAMES, &rhat_metrics)?;
            py_tuple_from_items(
                py,
                vec![
                    beta.into_pyarray(py).into_bound().into_any().unbind(),
                    alpha.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, varb_mean),
                    py_float_item(py, vare),
                    py_float_item(py, h2_mean),
                    py_float_item(py, var_h2),
                    py_float_item(py, prob_in_mean),
                    py_float_item(py, n_active_mean),
                    pip.into_pyarray(py).into_bound().into_any().unbind(),
                    py_float_item(py, rhat_h2),
                    py_usize_item(py, actual_iterations),
                    py_usize_item(py, convergence_iteration),
                    rhat_py,
                ],
            )
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesa_packed_trace_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    trace_snp_indices: &[usize],
    block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<PackedBayesTraceResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if q == 0 || x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;

    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut source = BayesPackedSource::Resident {
        packed_flat,
        bytes_per_snp,
    };
    let mut backend = PackedBayesBackend::new(
        &mut source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        None,
        sample_idx,
        block_rows,
        pool,
    )?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, None, s0_b_opt)?;

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let keep_iters = posterior_keep_iters(n_iter, burnin, thin);
    let n_keep_target = keep_iters.len();
    let n_trace_snps = trace_snp_indices.len();
    let mut h2_trace = Vec::with_capacity(n_keep_target);
    let mut var_e_trace = Vec::with_capacity(n_keep_target);
    let prob_in_trace = vec![f64::NAN; n_keep_target];
    let n_active_trace = vec![f64::NAN; n_keep_target];
    let full_iter_trace: Vec<i64> = (1..=n_iter).map(|v| v as i64).collect();
    let mut full_h2_trace = Vec::with_capacity(n_iter);
    let mut full_var_e_trace = Vec::with_capacity(n_iter);
    let full_prob_in_trace = vec![f64::NAN; n_iter];
    let full_n_active_trace = vec![f64::NAN; n_iter];
    let mut beta_trace = vec![0.0_f64; n_keep_target * n_trace_snps];
    let mut keep_row = 0usize;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                let z_beta: f64 = rng.sample(StandardNormal);
                let old_beta = beta[j];
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, old_beta, x2[j], |u| {
                        u * inv_var_e / c + (1.0 / c).sqrt() * z_beta
                    });
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
            if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                return Err("BayesA packed var_b became non-finite or non-positive".to_string());
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesA packed S became non-finite or non-positive".to_string());
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesA packed var_e became non-finite or non-positive".to_string());
        }

        let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
        let h2 = var_g / (var_g + var_e);
        full_h2_trace.push(h2);
        full_var_e_trace.push(var_e);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            h2_trace.push(h2);
            var_e_trace.push(var_e);
            fill_beta_trace_row(
                &mut beta_trace,
                keep_row,
                n_trace_snps,
                trace_snp_indices,
                &beta,
                None,
            );
            keep_row += 1;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept for the requested trace iterations".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for bj in &mut beta_sum {
        *bj *= inv_keep;
    }
    for ak in &mut alpha_sum {
        *ak *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }

    Ok(PackedBayesTraceResult {
        beta: beta_sum,
        alpha: alpha_sum,
        vare: var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean: f64::NAN,
        n_active_mean: f64::NAN,
        iter_trace: keep_iters,
        h2_trace,
        var_e_trace,
        prob_in_trace,
        n_active_trace,
        full_iter_trace,
        full_h2_trace,
        full_var_e_trace,
        full_prob_in_trace,
        full_n_active_trace,
        beta_trace_indices: trace_snp_indices.iter().map(|&v| v as i64).collect(),
        beta_trace,
    })
}

#[allow(clippy::too_many_arguments)]
fn bayesb_packed_trace_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    trace_snp_indices: &[usize],
    block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<PackedBayesTraceResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if q == 0 || x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;

    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut source = BayesPackedSource::Resident {
        packed_flat,
        bytes_per_snp,
    };
    let mut backend = PackedBayesBackend::new(
        &mut source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        None,
        sample_idx,
        block_rows,
        pool,
    )?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, Some(prob_in_base), s0_b_opt)?;

    let rate0 = bayesb_rate0(shape0, rate0_opt, s0_b)?;

    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut prob_in = prob_in_base;
    let mut s = s0_b;
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_base, counts);

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let keep_iters = posterior_keep_iters(n_iter, burnin, thin);
    let n_keep_target = keep_iters.len();
    let n_trace_snps = trace_snp_indices.len();
    let mut h2_trace = Vec::with_capacity(n_keep_target);
    let mut var_e_trace = Vec::with_capacity(n_keep_target);
    let mut prob_in_trace = Vec::with_capacity(n_keep_target);
    let mut n_active_trace = Vec::with_capacity(n_keep_target);
    let full_iter_trace: Vec<i64> = (1..=n_iter).map(|v| v as i64).collect();
    let mut full_h2_trace = Vec::with_capacity(n_iter);
    let mut full_var_e_trace = Vec::with_capacity(n_iter);
    let mut full_prob_in_trace = Vec::with_capacity(n_iter);
    let mut full_n_active_trace = Vec::with_capacity(n_iter);
    let mut beta_trace = vec![0.0_f64; n_keep_target * n_trace_snps];
    let mut keep_row = 0usize;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesB packed beta update".to_string(),
                    );
                }
                let mut new_d = 0u8;
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, b_old, x2[j], |xe| {
                        let rhs = xe * inv_var_e;
                        let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b[j] * c).ln();
                        let log_odds = log_odds_prior + log_bf10;
                        let p_in = if log_odds >= 0.0 {
                            1.0 / (1.0 + (-log_odds).exp())
                        } else {
                            let e = log_odds.exp();
                            e / (1.0 + e)
                        };
                        new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                        if new_d == 1 {
                            let z_beta: f64 = rng.sample(StandardNormal);
                            rhs / c + (1.0 / c).sqrt() * z_beta
                        } else {
                            0.0
                        }
                    });
                d[j] = new_d;
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut n_active = 0usize;
        for j in 0..p {
            if d[j] == 1 {
                let beta2 = beta[j] * beta[j];
                var_b[j] = bayes_positive_floor((s + beta2) / rng.sample(chi_b_active));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err("BayesB packed var_b became non-finite or non-positive".to_string());
                }
                n_active += 1;
            } else {
                var_b[j] = bayes_positive_floor(s / rng.sample(chi_b_inactive));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err(
                        "BayesB packed inactive var_b became non-finite or non-positive"
                            .to_string(),
                    );
                }
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err(
                "Gamma rate became non-positive while updating BayesB packed trace S".to_string(),
            );
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        s = bayes_positive_floor(sample_gamma_with_rate(&mut rng, tmp_shape, tmp_rate)?);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesB packed trace S became non-finite or non-positive".to_string());
        }

        let mrk_in = n_active as f64;
        let a = mrk_in + counts_in;
        let b = (p as f64 - mrk_in) + counts_out;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesB packed var_e became non-finite or non-positive".to_string());
        }

        let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
        let h2 = var_g / (var_g + var_e);
        full_h2_trace.push(h2);
        full_var_e_trace.push(var_e);
        full_prob_in_trace.push(prob_in);
        full_n_active_trace.push(mrk_in);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += (d[j] as f64) * beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            h2_trace.push(h2);
            var_e_trace.push(var_e);
            prob_in_trace.push(prob_in);
            n_active_trace.push(mrk_in);
            fill_beta_trace_row(
                &mut beta_trace,
                keep_row,
                n_trace_snps,
                trace_snp_indices,
                &beta,
                Some(&d),
            );
            keep_row += 1;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept for the requested trace iterations".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for bj in &mut beta_sum {
        *bj *= inv_keep;
    }
    for ak in &mut alpha_sum {
        *ak *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok(PackedBayesTraceResult {
        beta: beta_sum,
        alpha: alpha_sum,
        vare: var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        iter_trace: keep_iters,
        h2_trace,
        var_e_trace,
        prob_in_trace,
        n_active_trace,
        full_iter_trace,
        full_h2_trace,
        full_var_e_trace,
        full_prob_in_trace,
        full_n_active_trace,
        beta_trace_indices: trace_snp_indices.iter().map(|&v| v as i64).collect(),
        beta_trace,
    })
}

#[allow(clippy::too_many_arguments)]
fn bayesc_packed_trace_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    trace_snp_indices: &[usize],
    block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<PackedBayesTraceResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if q == 0 || x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }
    let fixed_effects = BayesFixedEffectBackend::new(x, n, q)?;
    let fixed_init = fixed_effects.initial_state(y)?;
    let prior_calibration =
        BayesPriorCalibration::from_fixed_init(&fixed_init, r2, df0_e, prior_ss_e_opt)?;

    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut source = BayesPackedSource::Resident {
        packed_flat,
        bytes_per_snp,
    };
    let mut backend = PackedBayesBackend::new(
        &mut source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        None,
        sample_idx,
        block_rows,
        pool,
    )?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let s0_b = prior_calibration.marker_scale(msx, r2, df0_b, Some(prob_in_base), s0_b_opt)?;

    let mut var_e = prior_calibration.initial_var_e;
    let prior_ss_e = prior_calibration.prior_ss_e;

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = s0_b;
    let mut prob_in = prob_in_base;
    let counts_in = counts * prob_in_base;
    let counts_out = counts - counts_in;

    let mut alpha = fixed_init.alpha;
    let x2_x = fixed_effects.x2_x();
    let xtx = fixed_effects.xtx();
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = fixed_init.residual;
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let keep_iters = posterior_keep_iters(n_iter, burnin, thin);
    let n_keep_target = keep_iters.len();
    let n_trace_snps = trace_snp_indices.len();
    let mut h2_trace = Vec::with_capacity(n_keep_target);
    let mut var_e_trace = Vec::with_capacity(n_keep_target);
    let mut prob_in_trace = Vec::with_capacity(n_keep_target);
    let mut n_active_trace = Vec::with_capacity(n_keep_target);
    let full_iter_trace: Vec<i64> = (1..=n_iter).map(|v| v as i64).collect();
    let mut full_h2_trace = Vec::with_capacity(n_iter);
    let mut full_var_e_trace = Vec::with_capacity(n_iter);
    let mut full_prob_in_trace = Vec::with_capacity(n_iter);
    let mut full_n_active_trace = Vec::with_capacity(n_iter);
    let mut beta_trace = vec![0.0_f64; n_keep_target * n_trace_snps];
    let mut keep_row = 0usize;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;
    let mut chi_b_cache: HashMap<usize, ChiSquared<f64>> = HashMap::new();

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                let c = x2[j] * inv_var_e + 1.0 / var_b;
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesC packed beta update".to_string(),
                    );
                }
                let mut new_d = 0u8;
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, b_old, x2[j], |xe| {
                        let rhs = xe * inv_var_e;
                        let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b * c).ln();
                        let log_odds = log_odds_prior + log_bf10;
                        let p_in = if log_odds >= 0.0 {
                            1.0 / (1.0 + (-log_odds).exp())
                        } else {
                            let e = log_odds.exp();
                            e / (1.0 + e)
                        };
                        new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                        if new_d == 1 {
                            let z_beta: f64 = rng.sample(StandardNormal);
                            rhs / c + (1.0 / c).sqrt() * z_beta
                        } else {
                            0.0
                        }
                    });
                d[j] = new_d;
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut mrk_in_usize = 0usize;
        let mut ss_b = 0.0;
        for j in 0..p {
            if d[j] == 1 {
                ss_b += beta[j] * beta[j];
                mrk_in_usize += 1;
            }
        }
        ss_b += s0_b;
        let chi_b_eff = if let Some(dist) = chi_b_cache.get(&mrk_in_usize) {
            *dist
        } else {
            let dist = ChiSquared::new(df0_b + mrk_in_usize as f64).map_err(|e| e.to_string())?;
            chi_b_cache.insert(mrk_in_usize, dist);
            dist
        };
        var_b = ss_b / rng.sample(chi_b_eff);
        if !(var_b.is_finite() && var_b > 0.0) {
            return Err("BayesC packed var_b became non-finite or non-positive".to_string());
        }

        let mrk_in = mrk_in_usize as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesC packed var_e became non-finite or non-positive".to_string());
        }

        let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
        let h2 = var_g / (var_g + var_e);
        full_h2_trace.push(h2);
        full_var_e_trace.push(var_e);
        full_prob_in_trace.push(prob_in);
        full_n_active_trace.push(mrk_in);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += (d[j] as f64) * beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            h2_trace.push(h2);
            var_e_trace.push(var_e);
            prob_in_trace.push(prob_in);
            n_active_trace.push(mrk_in);
            fill_beta_trace_row(
                &mut beta_trace,
                keep_row,
                n_trace_snps,
                trace_snp_indices,
                &beta,
                Some(&d),
            );
            keep_row += 1;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept for the requested trace iterations".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for bj in &mut beta_sum {
        *bj *= inv_keep;
    }
    for ak in &mut alpha_sum {
        *ak *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok(PackedBayesTraceResult {
        beta: beta_sum,
        alpha: alpha_sum,
        vare: var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        iter_trace: keep_iters,
        h2_trace,
        var_e_trace,
        prob_in_trace,
        n_active_trace,
        full_iter_trace,
        full_h2_trace,
        full_var_e_trace,
        full_prob_in_trace,
        full_n_active_trace,
        beta_trace_indices: trace_snp_indices.iter().map(|&v| v as i64).collect(),
        beta_trace,
    })
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    prior_ss_e = None,
    min_abs_beta = 1e-9,
    trace_snp_indices = None,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesa_packed_trace<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    min_abs_beta: f64,
    trace_snp_indices: Option<PyReadonlyArray1<i64>>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_flip.as_array(),
        )),
    };
    let row_maf_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_maf) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_maf.as_array(),
        )),
    };
    let row_mean_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_mean) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_mean.as_array(),
        )),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> =
        match crate::stats_common::array_standard_slice(&row_inv_sd) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                &row_inv_sd.as_array(),
            )),
        };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let trace_idx =
        parse_optional_index_vec_i64(trace_snp_indices.as_ref(), p, "trace_snp_indices")?;

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        bayesa_packed_trace_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            prior_ss_e,
            seed,
            &trace_idx,
            block_rows,
            pool_ref,
        )
    });
    match result {
        Ok(res) => packed_trace_result_to_pydict(py, res),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.05,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    trace_snp_indices = None,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesb_packed_trace<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    trace_snp_indices: Option<PyReadonlyArray1<i64>>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    // The trace entry point follows the same fixed-pi BayesB contract.
    let fixed_pi = fixed_pi.or(Some(prob_in));
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_flip.as_array(),
        )),
    };
    let row_maf_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_maf) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_maf.as_array(),
        )),
    };
    let row_mean_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_mean) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_mean.as_array(),
        )),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> =
        match crate::stats_common::array_standard_slice(&row_inv_sd) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                &row_inv_sd.as_array(),
            )),
        };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let trace_idx =
        parse_optional_index_vec_i64(trace_snp_indices.as_ref(), p, "trace_snp_indices")?;

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        bayesb_packed_trace_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            &trace_idx,
            block_rows,
            pool_ref,
        )
    });
    match result {
        Ok(res) => packed_trace_result_to_pydict(py, res),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.05,
    counts = 5.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    trace_snp_indices = None,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesc_packed_trace<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    trace_snp_indices: Option<PyReadonlyArray1<i64>>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = prior_ss_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("prior_ss_e must be > 0"));
        }
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match crate::stats_common::array_standard_slice(&row_flip) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_flip.as_array(),
        )),
    };
    let row_maf_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_maf) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_maf.as_array(),
        )),
    };
    let row_mean_vec: Cow<'_, [f32]> = match crate::stats_common::array_standard_slice(&row_mean) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
            &row_mean.as_array(),
        )),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> =
        match crate::stats_common::array_standard_slice(&row_inv_sd) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(
                &row_inv_sd.as_array(),
            )),
        };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let trace_idx =
        parse_optional_index_vec_i64(trace_snp_indices.as_ref(), p, "trace_snp_indices")?;

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match crate::stats_common::array_standard_slice(arr) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match crate::stats_common::array_standard_slice(&packed) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(crate::stats_common::copy_array_standard_layout(&packed_arr)),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        bayesc_packed_trace_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            &trace_idx,
            block_rows,
            pool_ref,
        )
    });
    match result {
        Ok(res) => packed_trace_result_to_pydict(py, res),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    #[test]
    fn native_chain_cap_and_seed_derivation_are_stable() {
        assert_eq!(effective_bayes_chains(8, 8).unwrap(), 4);
        assert_eq!(effective_bayes_chains(8, 3).unwrap(), 1);
        assert!(effective_bayes_chains(0, 8).is_err());
        let first = bayes_chain_seeds(Some(123), 4);
        let second = bayes_chain_seeds(Some(123), 4);
        assert_eq!(first, second);
        assert_eq!(first.len(), 4);
        assert!(first.windows(2).all(|pair| pair[0] != pair[1]));
        assert_eq!(bayes_chain_seeds(Some(123), 1), vec![Some(123)]);
    }

    #[test]
    fn adaptive_chain_start_scales_are_overdispersed_but_fixed_mode_is_neutral() {
        let scales = (0..4)
            .map(|index| bayes_chain_start_scale(index, 4, true))
            .collect::<Vec<_>>();
        assert_eq!(scales, vec![0.7, 1.0, 1.4, 2.0]);
        assert_eq!(bayes_chain_start_scale(0, 4, false), 1.0);
        assert_eq!(bayes_chain_start_scale(0, 1, true), 1.0);
        assert_eq!(bayes_chain_start_scale(1, 2, true), 2.0);
        assert_eq!(bayes_chain_start_scale(1, 3, true), 1.0);
    }

    #[test]
    fn adaptive_chain_probability_scales_inclusion_odds() {
        let base = 0.2;
        let lower = bayes_chain_start_probability(base, 0.7);
        let upper = bayes_chain_start_probability(base, 2.0);
        assert_eq!(bayes_chain_start_probability(base, 1.0), base);
        assert!(lower < base && base < upper);
        assert!((lower / (1.0 - lower) - 0.7 * base / (1.0 - base)).abs() < 1e-12);
        assert!((upper / (1.0 - upper) - 2.0 * base / (1.0 - base)).abs() < 1e-12);
    }

    #[test]
    fn adaptive_chain_start_scale_separates_a_b_and_c_states() {
        let fixed_init = BayesFixedEffectInit {
            alpha: vec![0.0],
            residual: vec![1.0, -1.0],
        };
        let a_low = BayesAChainState::new(&fixed_init, 2, 1, 4.0, 5.0, 2.0, 0.7, Some(1));
        let a_high = BayesAChainState::new(&fixed_init, 2, 1, 4.0, 5.0, 2.0, 2.0, Some(2));
        assert!(a_low.s < a_high.s);
        assert!(a_low.var_b[0] < a_high.var_b[0]);

        let b_low = BayesBChainState::new(&fixed_init, 2, 1, 4.0, 2.0, 0.1, 5.0, 0.7, Some(1));
        let b_high = BayesBChainState::new(&fixed_init, 2, 1, 4.0, 2.0, 0.1, 5.0, 2.0, Some(2));
        assert!(b_low.s < b_high.s);
        assert!(b_low.var_b[0] < b_high.var_b[0]);

        let c_low = BayesCChainState::new(&fixed_init, 2, 1, 4.0, 2.0, 0.1, 0.7, Some(1));
        let c_high = BayesCChainState::new(&fixed_init, 2, 1, 4.0, 2.0, 0.1, 2.0, Some(2));
        assert!(c_low.var_b < c_high.var_b);
    }

    #[test]
    fn predictive_monitor_tracks_gebv_state_without_changing_report_metrics() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let residual = [0.5, 1.0, 1.5, 2.0];
        let x = [1.0, 1.0, 1.0, 1.0];
        let alpha = [0.5];
        let values =
            bayes_predictive_monitor_values(&y, &residual, &x, &alpha, 4, 1, 0.25, 0.75, 0.5);
        assert_eq!(values.len(), BAYES_PREDICTIVE_MONITOR_NAMES.len());
        assert_eq!(&values[..3], &[0.25, 0.75, 0.5]);
        assert!((values[3] - 0.75).abs() < 1e-12);
        assert!(values.iter().all(|value| value.is_finite()));

        let changed_residual = [0.25, 1.0, 1.75, 2.5];
        let changed = bayes_predictive_monitor_values(
            &y,
            &changed_residual,
            &x,
            &alpha,
            4,
            1,
            0.25,
            0.75,
            0.5,
        );
        assert!(values[3..]
            .iter()
            .zip(changed[3..].iter())
            .any(|(a, b)| { (a - b).abs() > 1e-12 }));
    }

    #[test]
    fn pooled_chain_rhat_detects_between_chain_shift() {
        let equal = aggregate_bayes_h2(&[0.5, 0.5], &[0.01, 0.01], &[1.01, 1.02], 1000);
        assert!(equal.0 > 0.49 && equal.0 < 0.51);
        assert!(equal.2 >= 1.0);
        let shifted = aggregate_bayes_h2(&[0.1, 0.9], &[0.0001, 0.0001], &[1.01, 1.02], 1000);
        assert!(shifted.2 > 1.10);
    }

    #[test]
    fn native_chain_controller_does_not_converge_on_shifted_constant_chains() {
        let mut controller =
            BayesMultiChainController::new(2, 2000, None, 1, BAYES_RHAT_A_NAMES).unwrap();
        let mut started = false;
        for _ in 0..650 {
            assert!(controller.should_run());
            let retained = controller.begin_iteration();
            if controller
                .observe(&[vec![0.1, 0.2, 0.3], vec![0.9, 0.8, 0.7]], retained)
                .unwrap()
            {
                started = true;
                break;
            }
        }
        assert!(!started);
        assert!(!controller.collecting_posterior());
    }

    #[test]
    fn multi_chain_adaptive_stop_ignores_architecture_report_metrics() {
        let mut controller = BayesMultiChainController::new_with_monitor(
            2,
            2000,
            None,
            1,
            BAYES_RHAT_R_NAMES,
            BAYES_PREDICTIVE_MONITOR_NAMES,
        )
        .unwrap();
        let stable_monitor = vec![0.5; BAYES_PREDICTIVE_MONITOR_NAMES.len()];
        let shifted_report = [
            vec![0.1, 0.2, 0.3, 2.0, 0.1, 0.1, 0.1, 0.1, 1.0],
            vec![0.9, 0.8, 0.7, 8.0, 0.9, 0.9, 0.9, 0.9, 9.0],
        ];
        let mut started = false;
        while controller.should_run() {
            let retained = controller.begin_iteration();
            if controller
                .observe_with_monitor(
                    &shifted_report,
                    &[stable_monitor.clone(), stable_monitor.clone()],
                    retained,
                )
                .unwrap()
            {
                started = true;
                break;
            }
        }
        assert!(started);
        assert!(controller.collecting_posterior());
    }

    #[test]
    fn multi_chain_adaptive_stop_detects_predictive_monitor_shift() {
        let mut controller = BayesMultiChainController::new_with_monitor(
            2,
            2000,
            None,
            1,
            BAYES_RHAT_R_NAMES,
            BAYES_PREDICTIVE_MONITOR_NAMES,
        )
        .unwrap();
        let mut shifted_monitor = [
            vec![0.5; BAYES_PREDICTIVE_MONITOR_NAMES.len()],
            vec![0.5; BAYES_PREDICTIVE_MONITOR_NAMES.len()],
        ];
        for value in &mut shifted_monitor[1] {
            *value = 1.5;
        }
        let report = vec![0.5; BAYES_RHAT_R_NAMES.len()];
        let reports = [report.clone(), report];
        for _ in 0..650 {
            assert!(controller.should_run());
            let retained = controller.begin_iteration();
            assert!(!controller
                .observe_with_monitor(&reports, &shifted_monitor, retained)
                .unwrap());
        }
        assert!(!controller.collecting_posterior());
    }

    #[test]
    fn posterior_predictive_validation_restarts_after_post_trigger_drift() {
        let mut controller = BayesMultiChainController::new_with_monitor(
            2,
            5000,
            None,
            1,
            BAYES_RHAT_R_NAMES,
            BAYES_PREDICTIVE_MONITOR_NAMES,
        )
        .unwrap();
        let report = vec![
            vec![0.5; BAYES_RHAT_R_NAMES.len()],
            vec![0.5; BAYES_RHAT_R_NAMES.len()],
        ];
        let stable_monitor = [
            vec![0.5; BAYES_PREDICTIVE_MONITOR_NAMES.len()],
            vec![0.5; BAYES_PREDICTIVE_MONITOR_NAMES.len()],
        ];
        for _ in 0..BAYES_AUTO_BURNIN {
            let retained = controller.begin_iteration();
            assert!(!retained);
            assert!(!controller
                .observe_with_monitor(&report, &stable_monitor, retained)
                .unwrap());
        }
        let mut triggered = false;
        while !triggered {
            assert!(controller.should_run());
            let retained = controller.begin_iteration();
            triggered = controller
                .observe_with_monitor(&report, &stable_monitor, retained)
                .unwrap();
        }
        assert!(controller.collecting_posterior());
        assert!(controller.convergence_iteration() > 0);

        let divergent_monitor = [
            vec![0.5; BAYES_PREDICTIVE_MONITOR_NAMES.len()],
            vec![1.5; BAYES_PREDICTIVE_MONITOR_NAMES.len()],
        ];
        for index in 0..BAYES_POSTERIOR_SAMPLES {
            let retained = controller.begin_iteration();
            assert!(retained);
            let restarted = controller
                .observe_with_monitor(&report, &divergent_monitor, retained)
                .unwrap();
            if index + 1 < BAYES_POSTERIOR_SAMPLES {
                assert!(!restarted);
            } else {
                assert!(restarted);
            }
        }
        assert!(!controller.collecting_posterior());
        assert_eq!(controller.posterior_samples(), 0);
        assert_eq!(controller.convergence_iteration(), 0);
        assert!(controller.should_run());

        // The retry must also be validated. Exhausting the bounded retry
        // budget may retain a fallback posterior, but it cannot restore a
        // false positive convergence flag.
        let mut retriggered = false;
        while !retriggered {
            assert!(controller.should_run());
            let retained = controller.begin_iteration();
            retriggered = controller
                .observe_with_monitor(&report, &stable_monitor, retained)
                .unwrap();
        }
        assert!(controller.collecting_posterior());
        assert!(controller.convergence_iteration() > 0);
        for index in 0..BAYES_POSTERIOR_SAMPLES {
            let retained = controller.begin_iteration();
            assert!(retained);
            let completed = controller
                .observe_with_monitor(&report, &divergent_monitor, retained)
                .unwrap();
            if index + 1 < BAYES_POSTERIOR_SAMPLES {
                assert!(!completed);
            } else {
                assert!(!completed);
            }
        }
        assert_eq!(controller.convergence_iteration(), 0);
        assert!(!controller.should_run());
    }

    #[test]
    fn single_chain_posterior_validation_clears_false_convergence() {
        let mut controller = BayesSamplingController::new(5000, None, 1, BAYES_RHAT_A_NAMES);
        let stable = [0.5, 0.5, 0.5];
        while !controller.collecting_posterior {
            assert!(controller.should_run());
            let retained = controller.begin_iteration();
            if retained {
                let started = controller.observe(&stable);
                assert_eq!(started, controller.collecting_posterior);
            }
        }
        assert!(controller.convergence_iteration() > 0);

        for index in 0..BAYES_POSTERIOR_SAMPLES {
            let retained = controller.begin_iteration();
            assert!(retained);
            let values = if index < BAYES_POSTERIOR_SAMPLES / 2 {
                stable
            } else {
                [1.5, 1.5, 1.5]
            };
            let restarted = controller.observe(&values);
            if index + 1 < BAYES_POSTERIOR_SAMPLES {
                assert!(!restarted);
            } else {
                assert!(restarted);
            }
        }
        assert!(!controller.collecting_posterior);
        assert_eq!(controller.posterior_samples(), 0);
        assert_eq!(controller.convergence_iteration(), 0);
        assert!(controller.should_run());
    }

    #[test]
    fn chain_controllers_respect_burnin_before_rhat_and_posterior() {
        let stable = [vec![0.5, 0.2, 0.3], vec![0.5, 0.2, 0.3]];
        let mut multi =
            BayesMultiChainController::new(2, 2000, Some(1000), 1, BAYES_RHAT_A_NAMES).unwrap();
        for _ in 0..1000 {
            assert!(multi.should_run());
            let retained = multi.begin_iteration();
            assert!(!retained);
            assert!(!multi.observe(&stable, retained).unwrap());
        }
        assert_eq!(multi.actual_iterations(), 1000);
        assert_eq!(multi.posterior_samples(), 0);
        assert!(!multi.collecting_posterior());

        let mut single = BayesSamplingController::new(2000, Some(1000), 1, BAYES_RHAT_A_NAMES);
        for _ in 0..1000 {
            assert!(single.should_run());
            let retained = single.begin_iteration();
            assert!(!retained);
            if retained {
                assert!(!single.observe(&stable[0]));
            }
        }
        assert_eq!(single.actual_iterations(), 1000);
        assert_eq!(single.posterior_samples(), 0);
    }

    #[test]
    fn rhat_controller_reports_each_metric_and_uses_their_maximum() {
        let mut controller = BayesSamplingController::new(2000, None, 1, BAYES_RHAT_A_NAMES);
        for iteration in 0..1500 {
            assert!(controller.should_run());
            let retained = controller.begin_iteration();
            if retained {
                let drift = iteration as f64 / 1000.0;
                let _ = controller.observe(&[0.5, drift, 0.2]);
            }
        }
        let values = controller.rhat_values();
        assert_eq!(values.len(), BAYES_RHAT_A_NAMES.len());
        assert!(values[1] > values[0]);
        if values.iter().all(|value| value.is_finite()) {
            assert_eq!(controller.rhat_max(), finite_rhat_max(&values));
        } else if values.iter().any(|value| *value == f64::INFINITY) {
            assert_eq!(controller.rhat_max(), f64::INFINITY);
        } else {
            assert!(controller.rhat_max().is_nan());
        }
        assert_eq!(finite_rhat_max(&[1.01, f64::INFINITY]), f64::INFINITY);
        assert!(finite_rhat_max(&[1.01, f64::NAN]).is_nan());
    }

    #[test]
    fn single_chain_rhat_uses_recent_window_not_prefix_history() {
        let mut state = BayesRhatState::with_window(BAYES_RHAT_A_NAMES, BAYES_RHAT_WINDOW);
        for index in 0..BAYES_RHAT_WINDOW {
            let phase = (index % 2) as f64;
            assert!(!state.observe(&[10.0 + phase, 10.0 + phase, 10.0 + phase]));
        }
        for index in 0..BAYES_RHAT_WINDOW {
            let phase = (index % 2) as f64;
            assert!(!state.observe(&[1.0 + phase, 1.0 + phase, 1.0 + phase]));
        }
        assert!(state
            .values()
            .iter()
            .all(|value| *value < BAYES_RHAT_THRESHOLD));
    }

    #[test]
    fn multi_chain_rhat_uses_recent_window_not_prefix_history() {
        let mut controller =
            BayesMultiChainController::new(2, 5000, None, 1, BAYES_RHAT_A_NAMES).unwrap();
        for _ in 0..BAYES_AUTO_BURNIN {
            assert!(controller.should_run());
            let retained = controller.begin_iteration();
            assert!(!retained);
            assert!(!controller
                .observe(&[vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0],], retained,)
                .unwrap());
        }
        for index in 0..(BAYES_RHAT_WINDOW * 2) {
            let phase = (index % 2) as f64;
            let (chain0, chain1) = if index < BAYES_RHAT_WINDOW {
                (
                    vec![10.0 + phase, 10.0 + phase, 10.0 + phase],
                    vec![1.0 + phase, 1.0 + phase, 1.0 + phase],
                )
            } else {
                (
                    vec![1.0 + phase, 1.0 + phase, 1.0 + phase],
                    vec![1.0 + phase, 1.0 + phase, 1.0 + phase],
                )
            };
            assert!(controller.should_run());
            let retained = controller.begin_iteration();
            assert!(retained);
            assert!(!controller.observe(&[chain0, chain1], retained).unwrap());
        }
        assert!(!controller.collecting_posterior());
        assert!(controller.monitor_rhat() < BAYES_RHAT_THRESHOLD);
    }

    #[test]
    fn fixed_burnin_skips_rhat_and_auto_burnin_starts_at_minimum() {
        let stable = [vec![0.5, 0.2, 0.3], vec![0.5, 0.2, 0.3]];
        let mut fixed =
            BayesMultiChainController::new(2, 2000, Some(3), 1, BAYES_RHAT_A_NAMES).unwrap();
        for _ in 0..3 {
            assert!(fixed.should_run());
            let retained = fixed.begin_iteration();
            assert!(!retained);
            assert!(!fixed.observe(&stable, retained).unwrap());
        }
        assert!(!fixed.collecting_posterior());
        assert!(fixed.should_run());
        assert!(fixed.begin_iteration());
        assert!(fixed.collecting_posterior());
        assert!(!fixed.observe(&stable, true).unwrap());
        assert_eq!(fixed.posterior_samples(), 1);
        assert_eq!(fixed.convergence_iteration(), 0);
        assert!(fixed
            .posterior_rhat_values()
            .iter()
            .all(|value| value.is_nan()));

        let mut auto =
            BayesMultiChainController::new(2, 2000, None, 1, BAYES_RHAT_A_NAMES).unwrap();
        for _ in 0..500 {
            assert!(auto.should_run());
            let retained = auto.begin_iteration();
            assert!(!retained);
            assert!(!auto.observe(&stable, retained).unwrap());
        }
        assert!(!auto.collecting_posterior());
        assert_eq!(auto.actual_iterations(), 500);
    }

    #[test]
    fn fixed_burnin_collects_exactly_one_thousand_samples_without_rhat() {
        let stable = [vec![0.5, 0.2, 0.3], vec![0.5, 0.2, 0.3]];

        let mut single = BayesSamplingController::new(1, Some(7), 1, BAYES_RHAT_A_NAMES);
        while single.should_run() {
            let retained = single.begin_iteration();
            if retained {
                assert!(!single.observe(&stable[0]));
            }
        }
        assert_eq!(single.actual_iterations(), 1007);
        assert_eq!(single.posterior_samples(), BAYES_POSTERIOR_SAMPLES);
        assert_eq!(single.convergence_iteration(), 0);
        assert!(single.rhat_value().is_nan());

        let mut multi =
            BayesMultiChainController::new(2, 1, Some(7), 1, BAYES_RHAT_A_NAMES).unwrap();
        while multi.should_run() {
            let retained = multi.begin_iteration();
            assert!(!multi.observe(&stable, retained).unwrap());
        }
        assert_eq!(multi.actual_iterations(), 1007);
        assert_eq!(multi.posterior_samples(), BAYES_POSTERIOR_SAMPLES);
        assert_eq!(multi.convergence_iteration(), 0);
        assert!(multi
            .posterior_rhat_values()
            .iter()
            .all(|value| value.is_nan()));
    }

    #[test]
    fn explicit_burnin_can_keep_adaptive_rhat_with_custom_posterior_target() {
        let stable = [0.5, 0.2, 0.3];
        let mut single = BayesSamplingController::new_with_options(
            5000,
            Some(1000),
            true,
            1,
            BAYES_RHAT_A_NAMES,
            7,
        );
        for _ in 0..1001 {
            assert!(single.should_run());
            let retained = single.begin_iteration();
            if retained {
                let _ = single.observe(&stable);
            }
        }
        assert!(!single.collecting_posterior);
        assert_eq!(single.posterior_samples(), 0);

        let stable_multi = vec![stable.to_vec(), stable.to_vec()];
        let mut multi = BayesMultiChainController::new_with_monitor_options(
            2,
            5000,
            Some(1000),
            true,
            1,
            BAYES_RHAT_A_NAMES,
            BAYES_RHAT_A_NAMES,
            7,
        )
        .unwrap();
        for _ in 0..1001 {
            assert!(multi.should_run());
            let retained = multi.begin_iteration();
            let _ = multi.observe(&stable_multi, retained).unwrap();
        }
        assert!(!multi.collecting_posterior());
        assert_eq!(multi.posterior_samples(), 0);

        let mut fixed_single =
            BayesSamplingController::new_with_options(1, Some(3), false, 1, BAYES_RHAT_A_NAMES, 7);
        while fixed_single.should_run() {
            let retained = fixed_single.begin_iteration();
            if retained {
                let _ = fixed_single.observe(&stable);
            }
        }
        assert_eq!(fixed_single.posterior_samples(), 7);

        let mut fixed_multi = BayesMultiChainController::new_with_monitor_options(
            2,
            1,
            Some(3),
            false,
            1,
            BAYES_RHAT_A_NAMES,
            BAYES_RHAT_A_NAMES,
            7,
        )
        .unwrap();
        while fixed_multi.should_run() {
            let retained = fixed_multi.begin_iteration();
            if retained {
                let _ = fixed_multi.observe(&stable_multi, retained).unwrap();
            }
        }
        assert_eq!(fixed_multi.posterior_samples(), 7);
    }

    #[test]
    fn marker_conditional_update_matches_remove_dot_add_reference() {
        let marker = [0.0_f64, 1.0, 2.0, 1.0, 0.0];
        let residual = [1.5_f64, -0.5, 2.0, 0.25, -1.0];
        let old_beta = -0.37_f64;
        let new_beta = 0.82_f64;
        let d = ddot_f64(&marker, &marker);

        let mut reference_residual = residual;
        daxpy_inplace_f64(old_beta, &marker, &mut reference_residual);
        let reference_u = ddot_f64(&reference_residual, &marker);
        daxpy_inplace_f64(-new_beta, &marker, &mut reference_residual);

        let fast_u = marker_conditional_dot_f64(&residual, &marker, old_beta, d);
        let mut fast_residual = residual;
        update_marker_residual_f64(old_beta, new_beta, &marker, &mut fast_residual);

        assert!((fast_u - reference_u).abs() < 1e-12);
        for (fast, reference) in fast_residual.iter().zip(reference_residual.iter()) {
            assert!((fast - reference).abs() < 1e-12);
        }
    }

    #[test]
    fn bayes_decode_plan_uses_dense_when_one_block_covers_all_markers() {
        assert_eq!(bayes_decode_mode(4, 4), BayesDecodeMode::Dense);
        assert_eq!(bayes_decode_mode(3, 4), BayesDecodeMode::Dense);
        assert_eq!(bayes_decode_mode(5, 4), BayesDecodeMode::Double);
    }

    #[test]
    fn dense_backend_reads_requested_marker_block() {
        let matrix = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut backend = DenseBayesBackend::new(&matrix, 4, 2).expect("valid dense backend");
        assert_eq!(backend.n_samples(), 4);
        assert_eq!(backend.n_markers(), 2);
        assert_eq!(backend.block_rows(), 2);

        let mut block = vec![0.0_f32; 4];
        backend
            .fill_block(1, 2, &mut block, 4)
            .expect("valid dense block");
        assert_eq!(block, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn dense_backend_iterates_borrowed_marker_rows() {
        let matrix = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let matrix_ptr = matrix.as_ptr();
        let mut backend = DenseBayesBackend::new(&matrix, 4, 2).expect("valid dense backend");
        let mut blocks = 0usize;
        backend
            .for_each_block(|row_start, row_end, block| {
                assert_eq!(block.as_ptr(), unsafe { matrix_ptr.add(row_start * 4) });
                assert_eq!(block.len(), (row_end - row_start) * 4);
                blocks += 1;
                Ok(())
            })
            .expect("dense borrowed iteration succeeds");
        assert_eq!(blocks, 1);
    }

    #[test]
    fn mixed_precision_marker_helpers_match_f64_reference() {
        let marker_f32 = [0.0_f32, 1.0, 2.0, 1.0, 0.0];
        let marker_f64 = marker_f32.map(f64::from);
        let residual = [1.5_f64, -0.5, 2.0, 0.25, -1.0];
        let old_beta = -0.37_f64;
        let new_beta = 0.82_f64;
        let d = ddot_f64(&marker_f64, &marker_f64);

        let reference_u = marker_conditional_dot_f64(&residual, &marker_f64, old_beta, d);
        let mixed_u = marker_conditional_dot_f32_f64(&residual, &marker_f32, old_beta, d);
        assert!((mixed_u - reference_u).abs() < 1e-12);

        let mut reference_residual = residual;
        update_marker_residual_f64(old_beta, new_beta, &marker_f64, &mut reference_residual);
        let mut mixed_residual = residual;
        update_marker_residual_f32_f64(old_beta, new_beta, &marker_f32, &mut mixed_residual);
        for (mixed, reference) in mixed_residual.iter().zip(reference_residual.iter()) {
            assert!((mixed - reference).abs() < 1e-12);
        }
    }

    #[test]
    fn f32_residual_marker_helpers_match_f64_reference_with_rounding_bound() {
        let marker = [0.0_f32, 1.0, 2.0, 1.0, 0.0];
        let residual_f64 = [1.5_f64, -0.5, 2.0, 0.25, -1.0];
        let residual_f32 = residual_f64.map(|value| value as f32);
        let marker_f64 = marker.map(f64::from);
        let old_beta = -0.37_f64;
        let new_beta = 0.82_f64;
        let d = ddot_f64(&marker_f64, &marker_f64);

        let reference_u = marker_conditional_dot_f64(&residual_f64, &marker_f64, old_beta, d);
        let f32_u = marker_conditional_dot_f32_f32(&residual_f32, &marker, old_beta, d);
        assert!((f32_u - reference_u).abs() < 1e-6);

        let mut reference_residual = residual_f64;
        update_marker_residual_f64(old_beta, new_beta, &marker_f64, &mut reference_residual);
        let mut f32_out = residual_f32;
        update_marker_residual_f32(old_beta, new_beta, &marker, &mut f32_out);
        for (actual, expected) in f32_out.iter().zip(reference_residual.iter()) {
            assert!((*actual as f64 - *expected).abs() < 2e-6);
        }
    }

    #[test]
    fn fast_f32_marker_dot_matches_f64_reference_with_float32_bound() {
        let marker = [
            0.03125_f32,
            1.0,
            2.0,
            1.0,
            -0.1171875,
            -0.5,
            0.25,
            0.6875,
            -1.375,
            0.203125,
            0.8125,
        ];
        let residual = [
            1.5_f32,
            -0.5,
            2.0,
            0.25,
            -1.0,
            0.75,
            -0.125,
            0.33333334,
            -0.7777778,
            1.2345679,
            -0.44444445,
        ];
        let marker_f64 = marker.map(f64::from);
        let residual_f64 = residual.map(f64::from);
        let marker_ss = marker_f64.iter().map(|v| v * v).sum::<f64>();
        let expected = marker_conditional_dot_f64(&residual_f64, &marker_f64, -0.37, marker_ss);
        let actual = marker_conditional_dot_f32_f32_fast(&residual, &marker, -0.37, marker_ss);
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn blocked_f32_marker_dot_matches_f64_reference() {
        let marker: Vec<f32> = (0..257).map(|i| ((i as f32) * 0.03125).sin()).collect();
        let residual: Vec<f32> = (0..257)
            .map(|i| ((i as f32) * 0.017).cos() * 0.75)
            .collect();
        let marker_f64: Vec<f64> = marker.iter().copied().map(f64::from).collect();
        let residual_f64: Vec<f64> = residual.iter().copied().map(f64::from).collect();
        let marker_ss = marker_f64.iter().map(|v| v * v).sum::<f64>();
        let expected = marker_conditional_dot_f64(&residual_f64, &marker_f64, -0.37, marker_ss);
        let actual = marker_conditional_dot_f32_f32_blocked(&residual, &marker, -0.37, marker_ss);
        assert!((actual - expected).abs() < 2e-5);
    }

    #[test]
    fn fused_marker_update_matches_dot_then_axpy_reference() {
        let marker = [0.0_f32, 1.0, 2.0, 1.0, 0.0, -0.5, 0.25, 1.25];
        let initial = [1.5_f32, -0.5, 2.0, 0.25, -1.0, 0.75, -0.125, 0.33333334];
        let old_beta = -0.37_f64;
        let marker_ss = marker
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        let expected_u =
            marker_conditional_dot_f32_f32_blocked(&initial, &marker, old_beta, marker_ss);
        let expected_beta = 0.125 + 0.03125 * expected_u;
        let mut expected_residual = initial;
        update_marker_residual_f32(old_beta, expected_beta, &marker, &mut expected_residual);

        let mut actual_residual = initial;
        let (actual_u, actual_beta) = bayes_marker_update_f32_fused(
            &mut actual_residual,
            &marker,
            old_beta,
            marker_ss,
            |u| 0.125 + 0.03125 * u,
        );
        assert_eq!(actual_u, expected_u);
        assert_eq!(actual_beta, expected_beta);
        assert_eq!(actual_residual, expected_residual);
    }

    #[test]
    fn dense_f32_input_keeps_contiguous_python_storage_borrowed() {
        Python::initialize();
        Python::attach(|py| {
            let array = numpy::PyArray2::<f32>::zeros(py, (2, 3), false);
            let any = array.as_any();
            let input = array2_to_f32_input(&any, "M").expect("valid float32 matrix");
            assert!(input.is_borrowed());
            assert_eq!(input.rows(), 2);
            assert_eq!(input.cols(), 3);
            assert_eq!(input.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        });
    }

    #[test]
    fn dense_f64_input_uses_owned_float32_fallback() {
        Python::initialize();
        Python::attach(|py| {
            let array = numpy::PyArray2::<f64>::zeros(py, (2, 3), false);
            let any = array.as_any();
            let input = array2_to_f32_input(&any, "M").expect("valid float64 matrix");
            assert!(!input.is_borrowed());
            assert_eq!(input.rows(), 2);
            assert_eq!(input.cols(), 3);
            assert_eq!(input.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        });
    }

    #[test]
    fn unified_marker_update_returns_score_and_updates_residual() {
        let marker = [0.0_f32, 1.0, 2.0, 1.0, 0.0];
        let mut residual = [1.5_f32, -0.5, 2.0, 0.25, -1.0];
        let old_beta = -0.37_f64;
        let new_beta = 0.82_f64;
        let marker_ss = marker
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        let expected_u = marker_conditional_dot_f32_f32(&residual, &marker, old_beta, marker_ss);
        let (u, actual_beta) =
            bayes_marker_update_f32(&mut residual, &marker, old_beta, marker_ss, |_| new_beta);
        assert_eq!(actual_beta, new_beta);
        assert_eq!(u, expected_u);
    }

    #[test]
    fn zero_marker_update_is_a_noop() {
        let marker = [0.0_f32, 1.0, 2.0, 1.0, 0.0];
        let mut residual = [1.5_f32, -0.5, 2.0, 0.25, -1.0];
        let expected = residual;
        update_marker_residual_f32(0.0, 0.0, &marker, &mut residual);
        assert_eq!(residual, expected);
    }

    #[test]
    fn packed_backend_rejects_incomplete_row_metadata() {
        let packed = [0_u8; 2];
        let mut source = BayesPackedSource::Resident {
            packed_flat: &packed,
            bytes_per_snp: 1,
        };
        let result = PackedBayesBackend::new(
            &mut source,
            4,
            2,
            &[false],
            &[0.0, 0.0],
            &[1.0, 1.0],
            None,
            &[0, 1],
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn packed_backend_double_buffer_preserves_marker_order() {
        // Four samples, three SNP rows, two-bit BED codes.  The values are
        // intentionally distinct so an out-of-order prefetch is observable.
        let packed = [0b00_11_10_00_u8, 0b11_00_10_11_u8, 0b10_10_10_10_u8];
        let mut source = BayesPackedSource::Resident {
            packed_flat: &packed,
            bytes_per_snp: 1,
        };
        let mut backend = PackedBayesBackend::new(
            &mut source,
            4,
            3,
            &[false, false, false],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            None,
            &[0, 1, 2, 3],
            Some(1),
            None,
        )
        .expect("valid packed backend");

        let mut starts = Vec::new();
        let mut rows = Vec::new();
        backend
            .for_each_block(|start, end, block| {
                starts.push((start, end));
                rows.push(block.to_vec());
                Ok(())
            })
            .expect("double-buffered scan succeeds");

        assert_eq!(starts, vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![0.0, 1.0, 2.0, 0.0]);
        assert_eq!(rows[1], vec![2.0, 1.0, 0.0, 2.0]);
        assert_eq!(rows[2], vec![1.0, 1.0, 1.0, 1.0]);
    }
}
