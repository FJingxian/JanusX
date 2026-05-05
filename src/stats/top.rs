use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::seq::index::sample as sample_index;
use rand::SeedableRng;
use std::cmp::Ordering;

pub(crate) type TopResult<T> = Result<T, String>;

const WEIGHT_FLOOR: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopMode {
    Auto,
    ExactNewton,
    ExactBfgs,
    MiniBatchAdam,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelSelectMode {
    PerTrait,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelSelectMetric {
    Pearson,
    Spearman,
    Rmse,
    Nrmse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CalibrationMode {
    None,
    AddMean,
    Linear,
}

#[derive(Debug, Clone)]
pub(crate) struct TopConfig {
    pub mode: TopMode,
    pub exact_threshold: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub l2: f64,
    pub damping: f64,
    pub max_backtracking: usize,
    pub line_search_shrink: f64,
    pub line_search_c1: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub normalize_weights: bool,
    pub seed: u64,
    pub model_select_mode: ModelSelectMode,
    pub model_select_metric: ModelSelectMetric,
    pub calibration_mode: CalibrationMode,
}

impl Default for TopConfig {
    fn default() -> Self {
        Self {
            mode: TopMode::Auto,
            exact_threshold: 20_000,
            max_iter: 50,
            tol: 1e-6,
            l2: 1e-4,
            damping: 1e-6,
            max_backtracking: 12,
            line_search_shrink: 0.5,
            line_search_c1: 1e-4,
            batch_size: 1024,
            epochs: 20,
            learning_rate: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            normalize_weights: true,
            seed: 2026,
            model_select_mode: ModelSelectMode::PerTrait,
            model_select_metric: ModelSelectMetric::Pearson,
            calibration_mode: CalibrationMode::Linear,
        }
    }
}

impl TopConfig {
    pub(crate) fn resolve_mode(&self, n: usize) -> TopMode {
        match self.mode {
            TopMode::Auto => {
                if n <= self.exact_threshold {
                    TopMode::ExactNewton
                } else {
                    TopMode::MiniBatchAdam
                }
            }
            other => other,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TopData {
    pub sample_ids: Vec<String>,
    pub trait_names: Vec<String>,
    pub y_true_raw: Vec<f64>,
    pub y_pred_oof_raw: Vec<f64>,
    pub n: usize,
    pub k: usize,
}

impl TopData {
    pub(crate) fn new(
        sample_ids: Vec<String>,
        trait_names: Vec<String>,
        y_true_raw: Vec<f64>,
        y_pred_oof_raw: Vec<f64>,
    ) -> TopResult<Self> {
        let n = sample_ids.len();
        let k = trait_names.len();
        if n == 0 || k == 0 {
            return Err("TopData requires non-empty sample_ids and trait_names.".to_string());
        }
        validate_matrix_len("y_true_raw", &y_true_raw, n, k)?;
        validate_matrix_len("y_pred_oof_raw", &y_pred_oof_raw, n, k)?;
        validate_finite_matrix("y_pred_oof_raw", &y_pred_oof_raw)?;
        Ok(Self {
            sample_ids,
            trait_names,
            y_true_raw,
            y_pred_oof_raw,
            n,
            k,
        })
    }

    pub(crate) fn validate(&self) -> TopResult<()> {
        if self.sample_ids.len() != self.n {
            return Err(format!(
                "TopData sample_ids length mismatch: {} vs n={}",
                self.sample_ids.len(),
                self.n
            ));
        }
        if self.trait_names.len() != self.k {
            return Err(format!(
                "TopData trait_names length mismatch: {} vs k={}",
                self.trait_names.len(),
                self.k
            ));
        }
        validate_matrix_len("y_true_raw", &self.y_true_raw, self.n, self.k)?;
        validate_matrix_len("y_pred_oof_raw", &self.y_pred_oof_raw, self.n, self.k)?;
        validate_finite_matrix("y_pred_oof_raw", &self.y_pred_oof_raw)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LinearCalibration {
    pub intercept: f64,
    pub slope: f64,
}

impl LinearCalibration {
    pub(crate) fn identity() -> Self {
        Self {
            intercept: 0.0,
            slope: 1.0,
        }
    }

    pub(crate) fn add_mean(mean_y: f64) -> Self {
        Self {
            intercept: mean_y,
            slope: 1.0,
        }
    }

    pub(crate) fn fit(y_true: &[f64], y_pred: &[f64]) -> TopResult<Self> {
        if y_true.len() != y_pred.len() {
            return Err("LinearCalibration::fit length mismatch.".to_string());
        }
        let mut n = 0usize;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
            if !yt.is_finite() || !yp.is_finite() {
                continue;
            }
            n += 1;
            sx += yp;
            sy += yt;
            sxx += yp * yp;
            sxy += yp * yt;
        }
        if n == 0 {
            return Err("LinearCalibration::fit found no finite samples.".to_string());
        }
        let n_f = n as f64;
        let mean_x = sx / n_f;
        let mean_y = sy / n_f;
        let var_x = (sxx / n_f) - mean_x * mean_x;
        if !var_x.is_finite() || var_x <= WEIGHT_FLOOR {
            return Ok(Self {
                intercept: mean_y,
                slope: 0.0,
            });
        }
        let cov_xy = (sxy / n_f) - mean_x * mean_y;
        let slope = cov_xy / var_x;
        let intercept = mean_y - slope * mean_x;
        Ok(Self { intercept, slope })
    }

    #[inline]
    pub(crate) fn apply(&self, raw: f64) -> f64 {
        self.intercept + self.slope * raw
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TraitCalibration {
    pub trait_name: String,
    pub selected_gs_model: String,
    pub prediction_type: String,
    pub intercept: f64,
    pub slope: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct Standardizer {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl Standardizer {
    pub(crate) fn fit(data: &[f64], n: usize, k: usize) -> TopResult<Self> {
        validate_matrix_len("standardizer input", data, n, k)?;
        validate_finite_matrix("standardizer input", data)?;
        let mut mean = vec![0.0; k];
        let mut std = vec![1.0; k];
        for t in 0..k {
            let mut sx = 0.0;
            let mut sxx = 0.0;
            for i in 0..n {
                let v = data[i * k + t];
                sx += v;
                sxx += v * v;
            }
            let n_f = n as f64;
            let mu = sx / n_f;
            let var = (sxx / n_f) - mu * mu;
            mean[t] = mu;
            std[t] = if var.is_finite() && var > WEIGHT_FLOOR {
                var.sqrt()
            } else {
                1.0
            };
        }
        Ok(Self { mean, std })
    }

    pub(crate) fn fit_missing(
        data: &[f64],
        observed_mask: &[bool],
        n: usize,
        k: usize,
    ) -> TopResult<Self> {
        validate_matrix_len("standardizer input", data, n, k)?;
        if observed_mask.len() != n.saturating_mul(k) {
            return Err("standardizer observed_mask length mismatch.".to_string());
        }
        let mut mean = vec![0.0; k];
        let mut std = vec![1.0; k];
        for t in 0..k {
            let mut sx = 0.0;
            let mut sxx = 0.0;
            let mut cnt = 0usize;
            for i in 0..n {
                let idx = i * k + t;
                let v = data[idx];
                if observed_mask[idx] && v.is_finite() {
                    sx += v;
                    sxx += v * v;
                    cnt += 1;
                }
            }
            if cnt == 0 {
                mean[t] = 0.0;
                std[t] = 1.0;
                continue;
            }
            let n_f = cnt as f64;
            let mu = sx / n_f;
            let var = (sxx / n_f) - mu * mu;
            mean[t] = mu;
            std[t] = if var.is_finite() && var > WEIGHT_FLOOR {
                var.sqrt()
            } else {
                1.0
            };
        }
        Ok(Self { mean, std })
    }

    pub(crate) fn transform_inplace(&self, data: &mut [f64], n: usize, k: usize) -> TopResult<()> {
        validate_matrix_len("transform_inplace input", data, n, k)?;
        if self.mean.len() != k || self.std.len() != k {
            return Err("Standardizer dimension mismatch.".to_string());
        }
        for i in 0..n {
            let base = i * k;
            for t in 0..k {
                let v = data[base + t];
                if !v.is_finite() {
                    continue;
                }
                let sd = self.std[t].max(WEIGHT_FLOOR);
                data[base + t] = (v - self.mean[t]) / sd;
            }
        }
        Ok(())
    }

    pub(crate) fn transform_row(&self, row: &mut [f64]) -> TopResult<()> {
        if row.len() != self.mean.len() || row.len() != self.std.len() {
            return Err("Standardizer row dimension mismatch.".to_string());
        }
        for t in 0..row.len() {
            let v = row[t];
            if !v.is_finite() {
                continue;
            }
            let sd = self.std[t].max(WEIGHT_FLOOR);
            row[t] = (v - self.mean[t]) / sd;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedTopData {
    pub y_true_std: Vec<f64>,
    pub y_pred_std: Vec<f64>,
    pub observed_mask: Vec<bool>,
    pub observed_samples: Vec<usize>,
    pub missing_aware: bool,
    pub standardizer: Standardizer,
    pub calibrations: Vec<LinearCalibration>,
}

#[derive(Debug, Clone)]
pub(crate) struct TopModel {
    pub trait_names: Vec<String>,
    pub weights: Vec<f64>,
    pub standardizer: Standardizer,
    pub calibrations: Vec<TraitCalibration>,
    pub selected_gs_model: Vec<String>,
    pub train_loss: f64,
    pub n_train: usize,
    pub mode: TopMode,
}

impl TopModel {
    pub(crate) fn new(
        trait_names: Vec<String>,
        weights: Vec<f64>,
        standardizer: Standardizer,
        calibrations: Vec<TraitCalibration>,
        train_loss: f64,
        n_train: usize,
        mode: TopMode,
    ) -> TopResult<Self> {
        if trait_names.is_empty() {
            return Err("TopModel requires at least one trait.".to_string());
        }
        if trait_names.len() != weights.len() || weights.len() != calibrations.len() {
            return Err("TopModel dimension mismatch.".to_string());
        }
        let selected_gs_model = calibrations
            .iter()
            .map(|c| c.selected_gs_model.clone())
            .collect::<Vec<_>>();
        Ok(Self {
            trait_names,
            weights,
            standardizer,
            calibrations,
            selected_gs_model,
            train_loss,
            n_train,
            mode,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TopFitResult {
    pub model: TopModel,
    pub report: TopTrainingReport,
}

#[derive(Debug, Clone)]
pub(crate) struct TopTrainingReport {
    pub mode: TopMode,
    pub optimizer: String,
    pub loss: f64,
    pub grad_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct TopRankRecord {
    pub sample_id: String,
    pub distance: f64,
    pub similarity: f64,
    pub rank: usize,
    pub pred_traits_raw: Vec<f64>,
    pub pred_traits_calibrated: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct TopObjective {
    pub loss: f64,
    pub grad: Vec<f64>,
    pub hess: Vec<f64>,
}

impl TopObjective {
    pub(crate) fn grad_norm(&self) -> f64 {
        l2_norm(&self.grad)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExactTopWorkspace {
    scores: Vec<f64>,
    x_diag: Vec<f64>,
    x_row: Vec<f64>,
    mean_x: Vec<f64>,
    second_x: Vec<f64>,
}

impl ExactTopWorkspace {
    pub(crate) fn new(n: usize, k: usize) -> Self {
        Self {
            scores: vec![0.0; n],
            x_diag: vec![0.0; k],
            x_row: vec![0.0; k],
            mean_x: vec![0.0; k],
            second_x: vec![0.0; k * k],
        }
    }

    fn ensure(&mut self, n: usize, k: usize) {
        if self.scores.len() != n {
            self.scores.resize(n, 0.0);
        }
        if self.x_diag.len() != k {
            self.x_diag.resize(k, 0.0);
        }
        if self.x_row.len() != k {
            self.x_row.resize(k, 0.0);
        }
        if self.mean_x.len() != k {
            self.mean_x.resize(k, 0.0);
        }
        if self.second_x.len() != k * k {
            self.second_x.resize(k * k, 0.0);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MiniBatchWorkspace {
    scores: Vec<f64>,
    x_diag: Vec<f64>,
    x_row: Vec<f64>,
    mean_x: Vec<f64>,
}

impl MiniBatchWorkspace {
    pub(crate) fn new(batch_size: usize, k: usize) -> Self {
        Self {
            scores: vec![0.0; batch_size],
            x_diag: vec![0.0; k],
            x_row: vec![0.0; k],
            mean_x: vec![0.0; k],
        }
    }

    fn ensure(&mut self, batch_size: usize, k: usize) {
        if self.scores.len() != batch_size {
            self.scores.resize(batch_size, 0.0);
        }
        if self.x_diag.len() != k {
            self.x_diag.resize(k, 0.0);
        }
        if self.x_row.len() != k {
            self.x_row.resize(k, 0.0);
        }
        if self.mean_x.len() != k {
            self.mean_x.resize(k, 0.0);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AdamState {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub t: usize,
}

impl AdamState {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            m: vec![0.0; k],
            v: vec![0.0; k],
            t: 0,
        }
    }

    pub(crate) fn step(
        &mut self,
        w: &mut [f64],
        grad: &[f64],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        normalize_weights: bool,
    ) {
        self.t += 1;
        let t_f = self.t as f64;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);
        for i in 0..w.len() {
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grad[i];
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bc1.max(WEIGHT_FLOOR);
            let v_hat = self.v[i] / bc2.max(WEIGHT_FLOOR);
            w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            if !w[i].is_finite() || w[i] < WEIGHT_FLOOR {
                w[i] = WEIGHT_FLOOR;
            }
        }
        if normalize_weights {
            normalize_sum_to_one(w);
        }
    }
}

pub(crate) fn normalize_sum_to_one(w: &mut [f64]) {
    if w.is_empty() {
        return;
    }
    let mut s = 0.0;
    for v in w.iter_mut() {
        if !v.is_finite() || *v < WEIGHT_FLOOR {
            *v = WEIGHT_FLOOR;
        }
        s += *v;
    }
    if !s.is_finite() || s <= 0.0 {
        let val = 1.0 / (w.len() as f64);
        for v in w.iter_mut() {
            *v = val;
        }
        return;
    }
    for v in w.iter_mut() {
        *v /= s;
    }
}

pub(crate) fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let mut n = 0usize;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (&a, &b) in x.iter().zip(y.iter()) {
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        n += 1;
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    if n < 2 {
        return None;
    }
    let n_f = n as f64;
    let cov = sxy - sx * sy / n_f;
    let vx = sxx - sx * sx / n_f;
    let vy = syy - sy * sy / n_f;
    let den = (vx * vy).sqrt();
    if !den.is_finite() || den <= 0.0 {
        None
    } else {
        Some(cov / den)
    }
}

pub(crate) fn rmse(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let mut n = 0usize;
    let mut ss = 0.0;
    for (&a, &b) in x.iter().zip(y.iter()) {
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let d = a - b;
        ss += d * d;
        n += 1;
    }
    if n == 0 {
        None
    } else {
        Some((ss / (n as f64)).sqrt())
    }
}

pub(crate) fn mae(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let mut n = 0usize;
    let mut s = 0.0;
    for (&a, &b) in x.iter().zip(y.iter()) {
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        s += (a - b).abs();
        n += 1;
    }
    if n == 0 {
        None
    } else {
        Some(s / (n as f64))
    }
}

pub(crate) fn nrmse(x: &[f64], y: &[f64]) -> Option<f64> {
    let rmse_v = rmse(x, y)?;
    let (_, std_y) = mean_std(y).ok()?;
    if std_y <= WEIGHT_FLOOR {
        None
    } else {
        Some(rmse_v / std_y)
    }
}

pub(crate) fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    for (&a, &b) in x.iter().zip(y.iter()) {
        if a.is_finite() && b.is_finite() {
            xv.push(a);
            yv.push(b);
        }
    }
    if xv.len() < 2 {
        return None;
    }
    let rx = rank_with_ties(&xv)?;
    let ry = rank_with_ties(&yv)?;
    pearson(&rx, &ry)
}

pub(crate) fn prepare_top_training_data(
    data: &TopData,
    calibration_mode: CalibrationMode,
) -> TopResult<PreparedTopData> {
    data.validate()?;
    let observed_mask = build_observed_mask(&data.y_true_raw);
    let observed_samples = observed_sample_indices(&observed_mask, data.n, data.k);
    let missing_aware = observed_mask.iter().any(|&x| !x);

    let standardizer = if missing_aware {
        Standardizer::fit_missing(&data.y_true_raw, &observed_mask, data.n, data.k)?
    } else {
        Standardizer::fit(&data.y_true_raw, data.n, data.k)?
    };

    let mut calibrations = Vec::with_capacity(data.k);
    for t in 0..data.k {
        let cal = match calibration_mode {
            CalibrationMode::None => LinearCalibration::identity(),
            CalibrationMode::AddMean => LinearCalibration::add_mean(standardizer.mean[t]),
            CalibrationMode::Linear => fit_linear_calibration_with_mask(
                &data.y_true_raw,
                &data.y_pred_oof_raw,
                &observed_mask,
                data.n,
                data.k,
                t,
                standardizer.mean[t],
            ),
        };
        calibrations.push(cal);
    }
    let mut y_true_std = data.y_true_raw.clone();
    let mut y_pred_cal = data.y_pred_oof_raw.clone();
    calibrate_matrix_by_trait_inplace(&mut y_pred_cal, data.n, data.k, &calibrations)?;
    standardizer.transform_inplace(&mut y_true_std, data.n, data.k)?;
    standardizer.transform_inplace(&mut y_pred_cal, data.n, data.k)?;
    Ok(PreparedTopData {
        y_true_std,
        y_pred_std: y_pred_cal,
        observed_mask,
        observed_samples,
        missing_aware,
        standardizer,
        calibrations,
    })
}

fn fit_linear_calibration_with_mask(
    y_true_raw: &[f64],
    y_pred_raw: &[f64],
    observed_mask: &[bool],
    n: usize,
    k: usize,
    trait_idx: usize,
    mean_y_fallback: f64,
) -> LinearCalibration {
    let mut y_t = Vec::with_capacity(n);
    let mut p_t = Vec::with_capacity(n);
    for i in 0..n {
        let idx = i * k + trait_idx;
        if observed_mask[idx] {
            let yt = y_true_raw[idx];
            let yp = y_pred_raw[idx];
            if yt.is_finite() && yp.is_finite() {
                y_t.push(yt);
                p_t.push(yp);
            }
        }
    }
    if y_t.len() < 3 {
        return LinearCalibration::add_mean(mean_y_fallback);
    }
    match LinearCalibration::fit(&y_t, &p_t) {
        Ok(v) => v,
        Err(_) => LinearCalibration::add_mean(mean_y_fallback),
    }
}

fn build_observed_mask(y_true_raw: &[f64]) -> Vec<bool> {
    y_true_raw.iter().map(|v| v.is_finite()).collect()
}

fn observed_sample_indices(observed_mask: &[bool], n: usize, k: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..n {
        let base = i * k;
        let mut any_obs = false;
        for t in 0..k {
            if observed_mask[base + t] {
                any_obs = true;
                break;
            }
        }
        if any_obs {
            out.push(i);
        }
    }
    out
}

pub(crate) fn calibrate_matrix_by_trait_inplace(
    data: &mut [f64],
    n: usize,
    k: usize,
    calibrations: &[LinearCalibration],
) -> TopResult<()> {
    validate_matrix_len("calibrate_matrix_by_trait_inplace", data, n, k)?;
    if calibrations.len() != k {
        return Err("calibration length mismatch.".to_string());
    }
    for i in 0..n {
        let base = i * k;
        for t in 0..k {
            data[base + t] = calibrations[t].apply(data[base + t]);
        }
    }
    Ok(())
}

pub(crate) fn calibrate_matrix_by_trait(
    data: &[f64],
    n: usize,
    k: usize,
    calibrations: &[LinearCalibration],
) -> TopResult<Vec<f64>> {
    let mut out = data.to_vec();
    calibrate_matrix_by_trait_inplace(&mut out, n, k, calibrations)?;
    Ok(out)
}

pub(crate) fn find_target_by_sample_id(data: &TopData, target_id: &str) -> Option<Vec<f64>> {
    let idx = data.sample_ids.iter().position(|x| x == target_id)?;
    let mut out = vec![0.0; data.k];
    for t in 0..data.k {
        out[t] = data.y_true_raw[idx * data.k + t];
    }
    Some(out)
}

pub(crate) fn prepare_target(model: &TopModel, mut target_raw: Vec<f64>) -> TopResult<Vec<f64>> {
    model.standardizer.transform_row(&mut target_raw)?;
    Ok(target_raw)
}

pub(crate) fn exact_objective_with_workspace(
    y_pred_std: &[f64],
    y_true_std: &[f64],
    n: usize,
    k: usize,
    weights: &[f64],
    l2: f64,
    ws: &mut ExactTopWorkspace,
) -> TopResult<TopObjective> {
    validate_matrix_len("y_pred_std", y_pred_std, n, k)?;
    validate_matrix_len("y_true_std", y_true_std, n, k)?;
    if weights.len() != k {
        return Err("exact objective weight length mismatch.".to_string());
    }
    validate_finite_slice("weights", weights)?;
    ws.ensure(n, k);
    ws.mean_x.fill(0.0);
    ws.second_x.fill(0.0);
    let mut loss = 0.0;
    let mut grad = vec![0.0; k];
    let mut hess = vec![0.0; k * k];
    for i in 0..n {
        let pred_row = &y_pred_std[i * k..(i + 1) * k];
        ws.x_diag.fill(0.0);
        for j in 0..n {
            let true_row = &y_true_std[j * k..(j + 1) * k];
            let mut d = 0.0;
            for t in 0..k {
                let diff = (pred_row[t] - true_row[t]).abs();
                d += weights[t] * diff;
                if i == j {
                    ws.x_diag[t] = diff;
                }
            }
            ws.scores[j] = -d;
        }
        let mut max_score = f64::NEG_INFINITY;
        for &s in &ws.scores[..n] {
            if s > max_score {
                max_score = s;
            }
        }
        let mut z = 0.0;
        for &s in &ws.scores[..n] {
            z += (s - max_score).exp();
        }
        if !z.is_finite() || z <= 0.0 {
            return Err("exact objective produced invalid partition function.".to_string());
        }
        let logsumexp = max_score + z.ln();
        loss += -ws.scores[i] + logsumexp;
        ws.mean_x.fill(0.0);
        ws.second_x.fill(0.0);
        for j in 0..n {
            let p = (ws.scores[j] - max_score).exp() / z;
            let true_row = &y_true_std[j * k..(j + 1) * k];
            for t in 0..k {
                ws.x_row[t] = (pred_row[t] - true_row[t]).abs();
            }
            for a in 0..k {
                let xa = ws.x_row[a];
                ws.mean_x[a] += p * xa;
                for b in 0..k {
                    ws.second_x[a * k + b] += p * xa * ws.x_row[b];
                }
            }
        }
        for t in 0..k {
            grad[t] += ws.x_diag[t] - ws.mean_x[t];
        }
        for a in 0..k {
            for b in 0..k {
                hess[a * k + b] += ws.second_x[a * k + b] - ws.mean_x[a] * ws.mean_x[b];
            }
        }
    }
    if l2 > 0.0 {
        for t in 0..k {
            loss += 0.5 * l2 * weights[t] * weights[t];
            grad[t] += l2 * weights[t];
            hess[t * k + t] += l2;
        }
    }
    Ok(TopObjective { loss, grad, hess })
}

pub(crate) fn exact_objective(
    y_pred_std: &[f64],
    y_true_std: &[f64],
    n: usize,
    k: usize,
    weights: &[f64],
    l2: f64,
) -> TopResult<TopObjective> {
    let mut ws = ExactTopWorkspace::new(n, k);
    exact_objective_with_workspace(y_pred_std, y_true_std, n, k, weights, l2, &mut ws)
}

pub(crate) fn minibatch_loss_grad_with_workspace(
    y_pred_std: &[f64],
    y_true_std: &[f64],
    n: usize,
    k: usize,
    batch_indices: &[usize],
    weights: &[f64],
    l2: f64,
    ws: &mut MiniBatchWorkspace,
) -> TopResult<(f64, Vec<f64>)> {
    validate_matrix_len("y_pred_std", y_pred_std, n, k)?;
    validate_matrix_len("y_true_std", y_true_std, n, k)?;
    if weights.len() != k {
        return Err("minibatch objective weight length mismatch.".to_string());
    }
    if batch_indices.is_empty() {
        return Err("minibatch objective requires a non-empty batch.".to_string());
    }
    validate_finite_slice("weights", weights)?;
    let bsz = batch_indices.len();
    ws.ensure(bsz, k);
    let mut loss = 0.0;
    let mut grad = vec![0.0; k];
    for (a_pos, &i) in batch_indices.iter().enumerate() {
        let pred_row = &y_pred_std[i * k..(i + 1) * k];
        ws.x_diag.fill(0.0);
        for (b_pos, &j) in batch_indices.iter().enumerate() {
            let true_row = &y_true_std[j * k..(j + 1) * k];
            let mut d = 0.0;
            for t in 0..k {
                let diff = (pred_row[t] - true_row[t]).abs();
                d += weights[t] * diff;
                if a_pos == b_pos {
                    ws.x_diag[t] = diff;
                }
            }
            ws.scores[b_pos] = -d;
        }
        let mut max_score = f64::NEG_INFINITY;
        for &s in &ws.scores[..bsz] {
            if s > max_score {
                max_score = s;
            }
        }
        let mut z = 0.0;
        for &s in &ws.scores[..bsz] {
            z += (s - max_score).exp();
        }
        if !z.is_finite() || z <= 0.0 {
            return Err("minibatch objective produced invalid partition function.".to_string());
        }
        let logsumexp = max_score + z.ln();
        loss += -ws.scores[a_pos] + logsumexp;
        ws.mean_x.fill(0.0);
        for (b_pos, &j) in batch_indices.iter().enumerate() {
            let p = (ws.scores[b_pos] - max_score).exp() / z;
            let true_row = &y_true_std[j * k..(j + 1) * k];
            for t in 0..k {
                ws.x_row[t] = (pred_row[t] - true_row[t]).abs();
            }
            for t in 0..k {
                ws.mean_x[t] += p * ws.x_row[t];
            }
        }
        for t in 0..k {
            grad[t] += ws.x_diag[t] - ws.mean_x[t];
        }
    }
    let inv_b = 1.0 / (bsz as f64);
    for t in 0..k {
        grad[t] *= inv_b;
    }
    loss *= inv_b;
    if l2 > 0.0 {
        for t in 0..k {
            loss += 0.5 * l2 * weights[t] * weights[t];
            grad[t] += l2 * weights[t];
        }
    }
    Ok((loss, grad))
}

pub(crate) fn minibatch_loss_grad(
    y_pred_std: &[f64],
    y_true_std: &[f64],
    n: usize,
    k: usize,
    batch_indices: &[usize],
    weights: &[f64],
    l2: f64,
) -> TopResult<(f64, Vec<f64>)> {
    let mut ws = MiniBatchWorkspace::new(batch_indices.len(), k);
    minibatch_loss_grad_with_workspace(
        y_pred_std,
        y_true_std,
        n,
        k,
        batch_indices,
        weights,
        l2,
        &mut ws,
    )
}

fn missing_distance_and_grad(
    pred_row: &[f64],
    true_row: &[f64],
    obs_mask_row: &[bool],
    weights: &[f64],
    normalize_by_weight: bool,
    grad_out: &mut [f64],
) -> TopResult<f64> {
    if pred_row.len() != true_row.len()
        || true_row.len() != obs_mask_row.len()
        || weights.len() != true_row.len()
        || grad_out.len() != true_row.len()
    {
        return Err("missing_distance_and_grad dimension mismatch.".to_string());
    }
    let k = true_row.len();
    grad_out.fill(0.0);
    let mut numerator = 0.0;
    let mut denom = if normalize_by_weight {
        WEIGHT_FLOOR
    } else {
        1.0
    };
    let mut observed_cnt = 0usize;
    for t in 0..k {
        if !obs_mask_row[t] {
            continue;
        }
        observed_cnt += 1;
        let x = (pred_row[t] - true_row[t]).abs();
        numerator += weights[t] * x;
        if normalize_by_weight {
            denom += weights[t];
        }
    }
    if observed_cnt == 0 {
        return Err(
            "missing_distance_and_grad received a reference row without observed traits."
                .to_string(),
        );
    }
    let d = numerator / denom.max(WEIGHT_FLOOR);
    for t in 0..k {
        if !obs_mask_row[t] {
            grad_out[t] = 0.0;
            continue;
        }
        let x = (pred_row[t] - true_row[t]).abs();
        if normalize_by_weight {
            grad_out[t] = (x - d) / denom.max(WEIGHT_FLOOR);
        } else {
            grad_out[t] = x;
        }
    }
    Ok(d)
}

pub(crate) fn minibatch_missing_loss_grad_with_workspace(
    prepared: &PreparedTopData,
    batch_indices: &[usize],
    weights: &[f64],
    l2: f64,
    normalize_by_weight: bool,
    ws: &mut MiniBatchWorkspace,
) -> TopResult<(f64, Vec<f64>)> {
    let n = prepared.y_true_std.len() / prepared.standardizer.mean.len();
    let k = prepared.standardizer.mean.len();
    validate_matrix_len("y_pred_std", &prepared.y_pred_std, n, k)?;
    validate_matrix_len("y_true_std", &prepared.y_true_std, n, k)?;
    if prepared.observed_mask.len() != n.saturating_mul(k) {
        return Err("observed_mask length mismatch.".to_string());
    }
    if weights.len() != k {
        return Err("minibatch missing-aware objective weight length mismatch.".to_string());
    }
    if batch_indices.is_empty() {
        return Err("minibatch missing-aware objective requires a non-empty batch.".to_string());
    }
    validate_finite_slice("weights", weights)?;

    let bsz = batch_indices.len();
    ws.ensure(bsz, k);
    let mut loss = 0.0;
    let mut grad = vec![0.0; k];
    for (a_pos, &i) in batch_indices.iter().enumerate() {
        let pred_row = &prepared.y_pred_std[i * k..(i + 1) * k];
        ws.x_diag.fill(0.0);
        let mut max_score = f64::NEG_INFINITY;
        let mut d_ii = None::<f64>;

        // Pass-1: score/logsumexp cache
        for (b_pos, &j) in batch_indices.iter().enumerate() {
            let true_row = &prepared.y_true_std[j * k..(j + 1) * k];
            let mask_row = &prepared.observed_mask[j * k..(j + 1) * k];
            let d = missing_distance_and_grad(
                pred_row,
                true_row,
                mask_row,
                weights,
                normalize_by_weight,
                &mut ws.x_row,
            )?;
            ws.scores[b_pos] = -d;
            if a_pos == b_pos {
                ws.x_diag.copy_from_slice(&ws.x_row);
                d_ii = Some(d);
            }
            if ws.scores[b_pos] > max_score {
                max_score = ws.scores[b_pos];
            }
        }
        let d_diag = d_ii.ok_or_else(|| {
            "missing-aware minibatch failed to capture diagonal distance.".to_string()
        })?;
        let mut z = 0.0;
        for &s in &ws.scores[..bsz] {
            z += (s - max_score).exp();
        }
        if !z.is_finite() || z <= 0.0 {
            return Err(
                "minibatch missing-aware objective produced invalid partition function."
                    .to_string(),
            );
        }
        let logsumexp = max_score + z.ln();
        loss += d_diag + logsumexp;

        // Pass-2: softmax-weighted gradient accumulation
        ws.mean_x.fill(0.0);
        for (b_pos, &j) in batch_indices.iter().enumerate() {
            let true_row = &prepared.y_true_std[j * k..(j + 1) * k];
            let mask_row = &prepared.observed_mask[j * k..(j + 1) * k];
            let _ = missing_distance_and_grad(
                pred_row,
                true_row,
                mask_row,
                weights,
                normalize_by_weight,
                &mut ws.x_row,
            )?;
            let p = (ws.scores[b_pos] - max_score).exp() / z;
            for t in 0..k {
                ws.mean_x[t] += p * ws.x_row[t];
            }
        }
        for t in 0..k {
            grad[t] += ws.x_diag[t] - ws.mean_x[t];
        }
    }

    let inv_b = 1.0 / (bsz as f64);
    for t in 0..k {
        grad[t] *= inv_b;
    }
    loss *= inv_b;
    if l2 > 0.0 {
        for t in 0..k {
            loss += 0.5 * l2 * weights[t] * weights[t];
            grad[t] += l2 * weights[t];
        }
    }
    Ok((loss, grad))
}

pub(crate) fn fit_top_model(data: &TopData, cfg: &TopConfig) -> TopResult<TopFitResult> {
    let prepared = prepare_top_training_data(data, cfg.calibration_mode)?;
    let mode = cfg.resolve_mode(data.n);
    let (weights, report) = if prepared.missing_aware {
        // Missing-aware TOP first version:
        // use normalized-distance mini-batch Adam to avoid heavy Hessian logic
        // and to keep memory bounded on sparse phenotype matrices.
        fit_minibatch_adam_missing(&prepared, cfg)?
    } else {
        match mode {
            TopMode::ExactNewton => fit_exact_newton(&prepared, cfg)?,
            TopMode::ExactBfgs => fit_exact_bfgs(&prepared, cfg)?,
            TopMode::MiniBatchAdam => fit_minibatch_adam(&prepared, cfg)?,
            TopMode::Auto => unreachable!("TopConfig::resolve_mode should have resolved Auto."),
        }
    };
    let mut calibrations = Vec::with_capacity(data.k);
    for t in 0..data.k {
        calibrations.push(TraitCalibration {
            trait_name: data.trait_names[t].clone(),
            selected_gs_model: String::new(),
            prediction_type: "raw_oof".to_string(),
            intercept: prepared.calibrations[t].intercept,
            slope: prepared.calibrations[t].slope,
        });
    }
    let model = TopModel::new(
        data.trait_names.clone(),
        weights,
        prepared.standardizer,
        calibrations,
        report.loss,
        if prepared.missing_aware {
            prepared.observed_samples.len()
        } else {
            data.n
        },
        report.mode,
    )?;
    Ok(TopFitResult { model, report })
}

pub(crate) fn fit_exact_newton(
    prepared: &PreparedTopData,
    cfg: &TopConfig,
) -> TopResult<(Vec<f64>, TopTrainingReport)> {
    let n = prepared.y_true_std.len() / prepared.standardizer.mean.len();
    let k = prepared.standardizer.mean.len();
    if n == 0 || k == 0 {
        return Err("exact TOP requires non-empty data.".to_string());
    }
    let mut weights = vec![1.0 / (k as f64); k];
    let mut ws = ExactTopWorkspace::new(n, k);
    let mut best_weights = weights.clone();
    let mut best_loss = f64::INFINITY;
    let mut converged = false;
    let mut grad_norm = f64::INFINITY;
    let mut iter_done = 0usize;
    for iter in 0..cfg.max_iter {
        let obj = exact_objective_with_workspace(
            &prepared.y_pred_std,
            &prepared.y_true_std,
            n,
            k,
            &weights,
            cfg.l2,
            &mut ws,
        )?;
        iter_done = iter + 1;
        grad_norm = obj.grad_norm();
        if obj.loss < best_loss {
            best_loss = obj.loss;
            best_weights = weights.clone();
        }
        if grad_norm <= cfg.tol {
            converged = true;
            best_loss = obj.loss;
            best_weights = weights.clone();
            break;
        }
        let mut step = solve_damped_linear_system(&obj.hess, &obj.grad, k, cfg.damping)?;
        let mut grad_dot_step = dot(&obj.grad, &step);
        if !grad_dot_step.is_finite() || grad_dot_step >= 0.0 {
            for t in 0..k {
                step[t] = -obj.grad[t];
            }
            grad_dot_step = -dot(&obj.grad, &obj.grad);
        }
        let mut accepted = false;
        let mut alpha = 1.0_f64;
        let mut candidate = weights.clone();
        for _ in 0..cfg.max_backtracking {
            candidate.copy_from_slice(&weights);
            for t in 0..k {
                candidate[t] += alpha * step[t];
                if !candidate[t].is_finite() || candidate[t] < WEIGHT_FLOOR {
                    candidate[t] = WEIGHT_FLOOR;
                }
            }
            if cfg.normalize_weights {
                normalize_sum_to_one(&mut candidate);
            }
            let cand_obj = exact_objective_with_workspace(
                &prepared.y_pred_std,
                &prepared.y_true_std,
                n,
                k,
                &candidate,
                cfg.l2,
                &mut ws,
            )?;
            let armijo_rhs = obj.loss + cfg.line_search_c1 * alpha * grad_dot_step;
            if cand_obj.loss <= armijo_rhs
                || (cand_obj.loss.is_finite() && cand_obj.loss < obj.loss)
            {
                weights = candidate.clone();
                accepted = true;
                if cand_obj.loss < best_loss {
                    best_loss = cand_obj.loss;
                    best_weights = candidate.clone();
                }
                break;
            }
            alpha *= cfg.line_search_shrink.max(0.05);
        }
        if !accepted {
            break;
        }
    }
    if cfg.normalize_weights {
        normalize_sum_to_one(&mut best_weights);
    }
    Ok((
        best_weights,
        TopTrainingReport {
            mode: TopMode::ExactNewton,
            optimizer: "newton".to_string(),
            loss: best_loss,
            grad_norm,
            iterations: iter_done,
            converged,
        },
    ))
}

pub(crate) fn fit_exact_bfgs(
    prepared: &PreparedTopData,
    cfg: &TopConfig,
) -> TopResult<(Vec<f64>, TopTrainingReport)> {
    let n = prepared.y_true_std.len() / prepared.standardizer.mean.len();
    let k = prepared.standardizer.mean.len();
    if n == 0 || k == 0 {
        return Err("exact TOP requires non-empty data.".to_string());
    }
    let mut weights = vec![1.0 / (k as f64); k];
    let mut h_inv = DMatrix::<f64>::identity(k, k);
    let mut ws = ExactTopWorkspace::new(n, k);
    let mut best_weights = weights.clone();
    let mut best_loss = f64::INFINITY;
    let mut converged = false;
    let mut grad_norm = f64::INFINITY;
    let mut iter_done = 0usize;
    let mut prev_grad: Option<Vec<f64>> = None;
    for iter in 0..cfg.max_iter {
        let obj = exact_objective_with_workspace(
            &prepared.y_pred_std,
            &prepared.y_true_std,
            n,
            k,
            &weights,
            cfg.l2,
            &mut ws,
        )?;
        iter_done = iter + 1;
        grad_norm = obj.grad_norm();
        if obj.loss < best_loss {
            best_loss = obj.loss;
            best_weights = weights.clone();
        }
        if grad_norm <= cfg.tol {
            converged = true;
            best_loss = obj.loss;
            best_weights = weights.clone();
            break;
        }
        let grad_vec = DVector::from_column_slice(&obj.grad);
        let step_vec = -&h_inv * grad_vec.clone();
        let mut step = step_vec.iter().copied().collect::<Vec<_>>();
        let mut grad_dot_step = dot(&obj.grad, &step);
        if !grad_dot_step.is_finite() || grad_dot_step >= 0.0 {
            for t in 0..k {
                step[t] = -obj.grad[t];
            }
            grad_dot_step = -dot(&obj.grad, &obj.grad);
        }
        let old_weights = weights.clone();
        let mut accepted = false;
        let mut alpha = 1.0_f64;
        let mut candidate = weights.clone();
        let mut cand_grad = obj.grad.clone();
        for _ in 0..cfg.max_backtracking {
            candidate.copy_from_slice(&old_weights);
            for t in 0..k {
                candidate[t] += alpha * step[t];
                if !candidate[t].is_finite() || candidate[t] < WEIGHT_FLOOR {
                    candidate[t] = WEIGHT_FLOOR;
                }
            }
            if cfg.normalize_weights {
                normalize_sum_to_one(&mut candidate);
            }
            let cand_obj = exact_objective_with_workspace(
                &prepared.y_pred_std,
                &prepared.y_true_std,
                n,
                k,
                &candidate,
                cfg.l2,
                &mut ws,
            )?;
            let armijo_rhs = obj.loss + cfg.line_search_c1 * alpha * grad_dot_step;
            if cand_obj.loss <= armijo_rhs
                || (cand_obj.loss.is_finite() && cand_obj.loss < obj.loss)
            {
                cand_grad = cand_obj.grad.clone();
                weights = candidate.clone();
                accepted = true;
                if cand_obj.loss < best_loss {
                    best_loss = cand_obj.loss;
                    best_weights = candidate.clone();
                }
                break;
            }
            alpha *= cfg.line_search_shrink.max(0.05);
        }
        if !accepted {
            break;
        }
        if let Some(prev) = prev_grad.take() {
            let s = vector_sub(&weights, &old_weights);
            let y = vector_sub(&cand_grad, &prev);
            let ys = dot(&y, &s);
            if ys.is_finite() && ys > 1e-12 {
                let rho = 1.0 / ys;
                let i_mat = DMatrix::<f64>::identity(k, k);
                let s_col = DVector::from_column_slice(&s);
                let y_col = DVector::from_column_slice(&y);
                let sy_t = &s_col * y_col.transpose();
                let ys_t = &y_col * s_col.transpose();
                let ss_t = &s_col * s_col.transpose();
                let left = &i_mat - rho * sy_t;
                let right = &i_mat - rho * ys_t;
                h_inv = left * h_inv * right + rho * ss_t;
            } else {
                h_inv = DMatrix::<f64>::identity(k, k);
            }
        }
        prev_grad = Some(cand_grad);
    }
    if cfg.normalize_weights {
        normalize_sum_to_one(&mut best_weights);
    }
    Ok((
        best_weights,
        TopTrainingReport {
            mode: TopMode::ExactBfgs,
            optimizer: "bfgs".to_string(),
            loss: best_loss,
            grad_norm,
            iterations: iter_done,
            converged,
        },
    ))
}

pub(crate) fn fit_minibatch_adam(
    prepared: &PreparedTopData,
    cfg: &TopConfig,
) -> TopResult<(Vec<f64>, TopTrainingReport)> {
    let n = prepared.y_true_std.len() / prepared.standardizer.mean.len();
    let k = prepared.standardizer.mean.len();
    if n == 0 || k == 0 {
        return Err("mini-batch TOP requires non-empty data.".to_string());
    }
    let mut weights = vec![1.0 / (k as f64); k];
    let mut best_weights = weights.clone();
    let mut best_loss = f64::INFINITY;
    let mut adam = AdamState::new(k);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let batch_size = cfg.batch_size.max(2).min(n);
    let steps_per_epoch = ((n + batch_size - 1) / batch_size).max(1);
    let mut ws = MiniBatchWorkspace::new(batch_size, k);
    let mut grad_norm = f64::INFINITY;
    let mut converged = false;
    let mut iter_done = 0usize;
    for _epoch in 0..cfg.epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_grad_norm = 0.0;
        for _ in 0..steps_per_epoch {
            let batch = sample_batch_indices(&mut rng, n, batch_size);
            let (loss, grad) = minibatch_loss_grad_with_workspace(
                &prepared.y_pred_std,
                &prepared.y_true_std,
                n,
                k,
                &batch,
                &weights,
                cfg.l2,
                &mut ws,
            )?;
            grad_norm = l2_norm(&grad);
            epoch_loss += loss;
            epoch_grad_norm += grad_norm;
            iter_done += 1;
            adam.step(
                &mut weights,
                &grad,
                cfg.learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.normalize_weights,
            );
        }
        let avg_loss = epoch_loss / (steps_per_epoch as f64);
        let avg_grad_norm = epoch_grad_norm / (steps_per_epoch as f64);
        if avg_loss < best_loss {
            best_loss = avg_loss;
            best_weights = weights.clone();
        }
        if avg_grad_norm <= cfg.tol {
            converged = true;
            break;
        }
    }
    if cfg.normalize_weights {
        normalize_sum_to_one(&mut best_weights);
    }
    Ok((
        best_weights,
        TopTrainingReport {
            mode: TopMode::MiniBatchAdam,
            optimizer: "adam".to_string(),
            loss: best_loss,
            grad_norm,
            iterations: iter_done,
            converged,
        },
    ))
}

pub(crate) fn fit_minibatch_adam_missing(
    prepared: &PreparedTopData,
    cfg: &TopConfig,
) -> TopResult<(Vec<f64>, TopTrainingReport)> {
    let n = prepared.y_true_std.len() / prepared.standardizer.mean.len();
    let k = prepared.standardizer.mean.len();
    if n == 0 || k == 0 {
        return Err("missing-aware mini-batch TOP requires non-empty data.".to_string());
    }
    let pool = &prepared.observed_samples;
    if pool.len() < 2 {
        return Err("missing-aware mini-batch TOP requires >=2 observed samples.".to_string());
    }
    let mut weights = vec![1.0 / (k as f64); k];
    let mut best_weights = weights.clone();
    let mut best_loss = f64::INFINITY;
    let mut adam = AdamState::new(k);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let batch_size = cfg.batch_size.max(2).min(pool.len());
    let steps_per_epoch = ((pool.len() + batch_size - 1) / batch_size).max(1);
    let mut ws = MiniBatchWorkspace::new(batch_size, k);
    let mut grad_norm = f64::INFINITY;
    let mut converged = false;
    let mut iter_done = 0usize;

    for _epoch in 0..cfg.epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_grad_norm = 0.0;
        for _ in 0..steps_per_epoch {
            let batch = sample_batch_indices_from_pool(&mut rng, pool, batch_size);
            let (loss, grad) = minibatch_missing_loss_grad_with_workspace(
                prepared, &batch, &weights, cfg.l2, true, &mut ws,
            )?;
            grad_norm = l2_norm(&grad);
            epoch_loss += loss;
            epoch_grad_norm += grad_norm;
            iter_done += 1;
            adam.step(
                &mut weights,
                &grad,
                cfg.learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.normalize_weights,
            );
        }
        let avg_loss = epoch_loss / (steps_per_epoch as f64);
        let avg_grad_norm = epoch_grad_norm / (steps_per_epoch as f64);
        if avg_loss < best_loss {
            best_loss = avg_loss;
            best_weights = weights.clone();
        }
        if avg_grad_norm <= cfg.tol {
            converged = true;
            break;
        }
    }
    if cfg.normalize_weights {
        normalize_sum_to_one(&mut best_weights);
    }
    Ok((
        best_weights,
        TopTrainingReport {
            mode: TopMode::MiniBatchAdam,
            optimizer: "adam-missing-aware".to_string(),
            loss: best_loss,
            grad_norm,
            iterations: iter_done,
            converged,
        },
    ))
}

pub(crate) fn rank_candidates(
    model: &TopModel,
    candidate_ids: &[String],
    y_candidate_pred_raw: &[f64],
    m: usize,
    k: usize,
    target_raw: &[f64],
) -> TopResult<Vec<TopRankRecord>> {
    if candidate_ids.len() != m {
        return Err("candidate_ids length mismatch.".to_string());
    }
    validate_matrix_len("candidate raw predictions", y_candidate_pred_raw, m, k)?;
    if model.weights.len() != k {
        return Err("TopModel weight dimension mismatch.".to_string());
    }
    if model.calibrations.len() != k {
        return Err("TopModel calibration dimension mismatch.".to_string());
    }
    if model.standardizer.mean.len() != k || model.standardizer.std.len() != k {
        return Err("TopModel standardizer dimension mismatch.".to_string());
    }
    let mut target_std = target_raw.to_vec();
    model.standardizer.transform_row(&mut target_std)?;
    let mut target_used = vec![false; k];
    let mut target_weight_denom = WEIGHT_FLOOR;
    for t in 0..k {
        if target_std[t].is_finite() {
            target_used[t] = true;
            target_weight_denom += model.weights[t];
        }
    }
    if !target_used.iter().any(|&x| x) {
        return Err("target values are all missing/non-finite after standardization.".to_string());
    }
    let mut records = Vec::with_capacity(m);
    for i in 0..m {
        let mut pred_raw = vec![0.0; k];
        let mut pred_cal = vec![0.0; k];
        let mut pred_std = vec![0.0; k];
        for t in 0..k {
            let raw = y_candidate_pred_raw[i * k + t];
            pred_raw[t] = raw;
            let cal = model.calibrations[t].intercept + model.calibrations[t].slope * raw;
            pred_cal[t] = cal;
            let sd = model.standardizer.std[t].max(WEIGHT_FLOOR);
            pred_std[t] = (cal - model.standardizer.mean[t]) / sd;
        }
        let mut d_num = 0.0;
        for t in 0..k {
            if !target_used[t] {
                continue;
            }
            d_num += model.weights[t] * (pred_std[t] - target_std[t]).abs();
        }
        let d = d_num / target_weight_denom.max(WEIGHT_FLOOR);
        records.push(TopRankRecord {
            sample_id: candidate_ids[i].clone(),
            distance: d,
            similarity: (-d).exp(),
            rank: 0,
            pred_traits_raw: pred_raw,
            pred_traits_calibrated: pred_cal,
        });
    }
    records.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });
    for (idx, rec) in records.iter_mut().enumerate() {
        rec.rank = idx + 1;
    }
    Ok(records)
}

pub(crate) fn rank_candidates_to_target_sample(
    model: &TopModel,
    candidate_ids: &[String],
    y_candidate_pred_raw: &[f64],
    m: usize,
    k: usize,
    data: &TopData,
    target_id: &str,
) -> TopResult<Vec<TopRankRecord>> {
    let target_raw = find_target_by_sample_id(data, target_id)
        .ok_or_else(|| format!("target sample not found: {target_id}"))?;
    rank_candidates(
        model,
        candidate_ids,
        y_candidate_pred_raw,
        m,
        k,
        &target_raw,
    )
}

pub(crate) fn sample_batch_indices(rng: &mut StdRng, n: usize, batch_size: usize) -> Vec<usize> {
    if batch_size >= n {
        return (0..n).collect();
    }
    sample_index(rng, n, batch_size).into_vec()
}

pub(crate) fn sample_batch_indices_from_pool(
    rng: &mut StdRng,
    pool: &[usize],
    batch_size: usize,
) -> Vec<usize> {
    if batch_size >= pool.len() {
        return pool.to_vec();
    }
    let picked = sample_index(rng, pool.len(), batch_size).into_vec();
    let mut out = Vec::with_capacity(picked.len());
    for idx in picked {
        out.push(pool[idx]);
    }
    out
}

pub(crate) fn solve_damped_linear_system(
    hess: &[f64],
    grad: &[f64],
    k: usize,
    damping: f64,
) -> TopResult<Vec<f64>> {
    if hess.len() != k * k || grad.len() != k {
        return Err("solve_damped_linear_system dimension mismatch.".to_string());
    }
    let mut mat = DMatrix::from_row_slice(k, k, hess);
    let damp = damping.max(1e-12);
    for i in 0..k {
        mat[(i, i)] += damp;
    }
    let rhs = DVector::from_iterator(k, grad.iter().map(|g| -*g));
    if let Some(chol) = Cholesky::new(mat.clone()) {
        let sol = chol.solve(&rhs);
        return Ok(sol.iter().copied().collect());
    }
    if let Some(sol) = mat.lu().solve(&rhs) {
        return Ok(sol.iter().copied().collect());
    }
    Err("failed to solve damped Newton system.".to_string())
}

fn validate_matrix_len(name: &str, data: &[f64], n: usize, k: usize) -> TopResult<()> {
    if data.len() != n.saturating_mul(k) {
        return Err(format!(
            "{name} length mismatch: len={} expected={}",
            data.len(),
            n.saturating_mul(k)
        ));
    }
    Ok(())
}

fn validate_finite_slice(name: &str, data: &[f64]) -> TopResult<()> {
    for (i, v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{name}[{i}] is not finite."));
        }
    }
    Ok(())
}

fn validate_finite_matrix(name: &str, data: &[f64]) -> TopResult<()> {
    validate_finite_slice(name, data)
}

fn mean_std(data: &[f64]) -> TopResult<(f64, f64)> {
    if data.is_empty() {
        return Err("mean_std requires a non-empty slice.".to_string());
    }
    let mut s = 0.0;
    let mut s2 = 0.0;
    let mut n = 0usize;
    for &v in data {
        if !v.is_finite() {
            continue;
        }
        s += v;
        s2 += v * v;
        n += 1;
    }
    if n == 0 {
        return Err("mean_std found no finite samples.".to_string());
    }
    let n_f = n as f64;
    let mean = s / n_f;
    let var = (s2 / n_f) - mean * mean;
    let std = if var.is_finite() && var > WEIGHT_FLOOR {
        var.sqrt()
    } else {
        1.0
    };
    Ok((mean, std))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn vector_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn rank_with_ties(values: &[f64]) -> Option<Vec<f64>> {
    if values.is_empty() {
        return None;
    }
    let mut pairs: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.is_finite() { Some((i, v)) } else { None })
        .collect();
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0usize;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].1 == pairs[i].1 {
            j += 1;
        }
        let avg_rank = 1.0 + ((i + j - 1) as f64) / 2.0;
        for idx in i..j {
            ranks[pairs[idx].0] = avg_rank;
        }
        i = j;
    }
    Some(ranks)
}

fn top_mode_to_str(mode: TopMode) -> &'static str {
    match mode {
        TopMode::Auto => "auto",
        TopMode::ExactNewton => "exact-newton",
        TopMode::ExactBfgs => "exact-bfgs",
        TopMode::MiniBatchAdam => "minibatch-adam",
    }
}

fn parse_top_mode(s: &str) -> Result<TopMode, String> {
    let x = s.trim().to_lowercase();
    match x.as_str() {
        "" | "auto" => Ok(TopMode::Auto),
        "exact-newton" | "exact_newton" | "newton" => Ok(TopMode::ExactNewton),
        "exact-bfgs" | "exact_bfgs" | "bfgs" => Ok(TopMode::ExactBfgs),
        "minibatch-adam" | "minibatch_adam" | "mini-batch-adam" | "adam" => {
            Ok(TopMode::MiniBatchAdam)
        }
        _ => Err(format!(
            "invalid top mode '{s}', expected one of: auto, exact-newton, exact-bfgs, minibatch-adam"
        )),
    }
}

fn parse_calibration_mode(s: &str) -> Result<CalibrationMode, String> {
    let x = s.trim().to_lowercase();
    match x.as_str() {
        "" | "linear" => Ok(CalibrationMode::Linear),
        "none" => Ok(CalibrationMode::None),
        "addmean" | "add_mean" | "add-mean" => Ok(CalibrationMode::AddMean),
        _ => Err(format!(
            "invalid calibration_mode '{s}', expected one of: linear, none, addmean"
        )),
    }
}

fn parse_top_model_from_dict(model: &Bound<'_, PyDict>) -> PyResult<TopModel> {
    let trait_names: Vec<String> = model
        .get_item("trait_names")?
        .ok_or_else(|| PyValueError::new_err("top model missing 'trait_names'"))?
        .extract()?;
    let weights: Vec<f64> = model
        .get_item("weights")?
        .ok_or_else(|| PyValueError::new_err("top model missing 'weights'"))?
        .extract()?;
    let mode_str: String = model
        .get_item("mode")?
        .and_then(|v| v.extract().ok())
        .unwrap_or_else(|| "auto".to_string());
    let mode = parse_top_mode(&mode_str)
        .map_err(|e| PyValueError::new_err(format!("invalid mode: {e}")))?;
    let train_loss = model
        .get_item("train_loss")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(0.0_f64);
    let n_train = model
        .get_item("n_train")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(0_usize);

    let standardizer_obj = model
        .get_item("standardizer")?
        .ok_or_else(|| PyValueError::new_err("top model missing 'standardizer'"))?;
    let standardizer_dict = standardizer_obj
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("top model field 'standardizer' must be dict"))?;
    let mean: Vec<f64> = standardizer_dict
        .get_item("mean")?
        .ok_or_else(|| PyValueError::new_err("top model standardizer missing 'mean'"))?
        .extract()?;
    let std: Vec<f64> = standardizer_dict
        .get_item("std")?
        .ok_or_else(|| PyValueError::new_err("top model standardizer missing 'std'"))?
        .extract()?;
    let standardizer = Standardizer { mean, std };

    let cal_obj = model
        .get_item("calibrations")?
        .ok_or_else(|| PyValueError::new_err("top model missing 'calibrations'"))?;
    let cal_list = cal_obj
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("top model field 'calibrations' must be list"))?;
    let mut calibrations = Vec::with_capacity(cal_list.len());
    for (i, item) in cal_list.iter().enumerate() {
        let d = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("calibrations[{i}] must be dict")))?;
        let trait_name: String = match d.get_item("trait_name")? {
            Some(v) => v.extract()?,
            None => match d.get_item("trait")? {
                Some(v) => v.extract()?,
                None => {
                    return Err(PyValueError::new_err(format!(
                        "calibrations[{i}] missing trait_name"
                    )))
                }
            },
        };
        let selected_gs_model: String = match d.get_item("selected_gs_model")? {
            Some(v) => v.extract().unwrap_or_default(),
            None => match d.get_item("model")? {
                Some(v) => v.extract().unwrap_or_default(),
                None => String::new(),
            },
        };
        let prediction_type: String = match d.get_item("prediction_type")? {
            Some(v) => v.extract().unwrap_or_else(|_| "raw_oof".to_string()),
            None => "raw_oof".to_string(),
        };
        let intercept: f64 = match d.get_item("intercept")? {
            Some(v) => v.extract()?,
            None => {
                return Err(PyValueError::new_err(format!(
                    "calibrations[{i}] missing intercept"
                )))
            }
        };
        let slope: f64 = match d.get_item("slope")? {
            Some(v) => v.extract()?,
            None => {
                return Err(PyValueError::new_err(format!(
                    "calibrations[{i}] missing slope"
                )))
            }
        };
        calibrations.push(TraitCalibration {
            trait_name,
            selected_gs_model,
            prediction_type,
            intercept,
            slope,
        });
    }

    TopModel::new(
        trait_names,
        weights,
        standardizer,
        calibrations,
        train_loss,
        n_train,
        mode,
    )
    .map_err(PyValueError::new_err)
}

fn matrix2_to_row_major(arr: &PyReadonlyArray2<'_, f64>) -> (Vec<f64>, usize, usize) {
    let view = arr.as_array();
    let (n, k) = view.dim();
    let mut out = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k {
            out.push(view[[i, j]]);
        }
    }
    (out, n, k)
}

#[pyfunction(name = "top_fit_model")]
#[pyo3(signature = (
    sample_ids,
    trait_names,
    y_true_raw,
    y_pred_oof_raw,
    selected_models=None,
    mode="auto",
    exact_threshold=20000,
    max_iter=50,
    tol=1e-6,
    l2=1e-4,
    damping=1e-6,
    max_backtracking=12,
    line_search_shrink=0.5,
    line_search_c1=1e-4,
    batch_size=1024,
    epochs=20,
    learning_rate=1e-2,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    normalize_weights=true,
    seed=2026,
    calibration_mode="linear"
))]
pub(crate) fn top_fit_model_py<'py>(
    py: Python<'py>,
    sample_ids: Vec<String>,
    trait_names: Vec<String>,
    y_true_raw: PyReadonlyArray2<'py, f64>,
    y_pred_oof_raw: PyReadonlyArray2<'py, f64>,
    selected_models: Option<Vec<String>>,
    mode: &str,
    exact_threshold: usize,
    max_iter: usize,
    tol: f64,
    l2: f64,
    damping: f64,
    max_backtracking: usize,
    line_search_shrink: f64,
    line_search_c1: f64,
    batch_size: usize,
    epochs: usize,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    normalize_weights: bool,
    seed: u64,
    calibration_mode: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let (y_true_vec, n_true, k_true) = matrix2_to_row_major(&y_true_raw);
    let (y_pred_vec, n_pred, k_pred) = matrix2_to_row_major(&y_pred_oof_raw);
    if n_true != n_pred || k_true != k_pred {
        return Err(PyValueError::new_err(format!(
            "matrix shape mismatch: y_true=({n_true},{k_true}) vs y_pred_oof=({n_pred},{k_pred})"
        )));
    }
    if sample_ids.len() != n_true {
        return Err(PyValueError::new_err(format!(
            "sample_ids length mismatch: {} vs matrix rows={n_true}",
            sample_ids.len()
        )));
    }
    if trait_names.len() != k_true {
        return Err(PyValueError::new_err(format!(
            "trait_names length mismatch: {} vs matrix cols={k_true}",
            trait_names.len()
        )));
    }
    if let Some(sm) = selected_models.as_ref() {
        if sm.len() != k_true {
            return Err(PyValueError::new_err(format!(
                "selected_models length mismatch: {} vs traits={k_true}",
                sm.len()
            )));
        }
    }
    let data = TopData::new(
        sample_ids.clone(),
        trait_names.clone(),
        y_true_vec,
        y_pred_vec,
    )
    .map_err(PyValueError::new_err)?;
    let mut cfg = TopConfig::default();
    cfg.mode = parse_top_mode(mode).map_err(PyValueError::new_err)?;
    cfg.exact_threshold = exact_threshold;
    cfg.max_iter = max_iter.max(1);
    cfg.tol = tol;
    cfg.l2 = l2;
    cfg.damping = damping;
    cfg.max_backtracking = max_backtracking.max(1);
    cfg.line_search_shrink = line_search_shrink;
    cfg.line_search_c1 = line_search_c1;
    cfg.batch_size = batch_size.max(2);
    cfg.epochs = epochs.max(1);
    cfg.learning_rate = learning_rate;
    cfg.beta1 = beta1;
    cfg.beta2 = beta2;
    cfg.eps = eps;
    cfg.normalize_weights = normalize_weights;
    cfg.seed = seed;
    cfg.calibration_mode =
        parse_calibration_mode(calibration_mode).map_err(PyValueError::new_err)?;

    let mut fit = fit_top_model(&data, &cfg).map_err(PyValueError::new_err)?;
    if let Some(sm) = selected_models {
        for (i, x) in sm.iter().enumerate() {
            fit.model.calibrations[i].selected_gs_model = x.clone();
            fit.model.selected_gs_model[i] = x.clone();
        }
    }
    let model = fit.model;
    let report = fit.report;

    let out = PyDict::new(py).into_bound();
    out.set_item("trait_names", model.trait_names.clone())?;
    out.set_item("weights", model.weights.clone())?;
    out.set_item("selected_gs_model", model.selected_gs_model.clone())?;
    out.set_item("train_loss", model.train_loss)?;
    out.set_item("n_train", model.n_train)?;
    out.set_item("mode", top_mode_to_str(model.mode))?;

    let std = PyDict::new(py).into_bound();
    std.set_item("mean", model.standardizer.mean.clone())?;
    std.set_item("std", model.standardizer.std.clone())?;
    out.set_item("standardizer", std.clone())?;

    let cal_list = PyList::empty(py).into_bound();
    for c in &model.calibrations {
        let d = PyDict::new(py).into_bound();
        d.set_item("trait_name", c.trait_name.clone())?;
        d.set_item("selected_gs_model", c.selected_gs_model.clone())?;
        d.set_item("prediction_type", c.prediction_type.clone())?;
        d.set_item("intercept", c.intercept)?;
        d.set_item("slope", c.slope)?;
        cal_list.append(d)?;
    }
    out.set_item("calibrations", cal_list.clone())?;

    let rep = PyDict::new(py).into_bound();
    rep.set_item("mode", top_mode_to_str(report.mode))?;
    rep.set_item("optimizer", report.optimizer.clone())?;
    rep.set_item("loss", report.loss)?;
    rep.set_item("grad_norm", report.grad_norm)?;
    rep.set_item("iterations", report.iterations)?;
    rep.set_item("converged", report.converged)?;
    out.set_item("report", rep.clone())?;
    Ok(out)
}

#[pyfunction(name = "top_rank_to_target_sample")]
#[pyo3(signature = (
    model,
    train_sample_ids,
    trait_names,
    y_true_raw,
    y_pred_oof_raw,
    candidate_ids,
    y_candidate_pred_raw,
    target_sample_id
))]
pub(crate) fn top_rank_to_target_sample_py<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyDict>,
    train_sample_ids: Vec<String>,
    trait_names: Vec<String>,
    y_true_raw: PyReadonlyArray2<'py, f64>,
    y_pred_oof_raw: PyReadonlyArray2<'py, f64>,
    candidate_ids: Vec<String>,
    y_candidate_pred_raw: PyReadonlyArray2<'py, f64>,
    target_sample_id: String,
) -> PyResult<Bound<'py, PyDict>> {
    let parsed_model = parse_top_model_from_dict(model)?;
    let (y_true_vec, n_true, k_true) = matrix2_to_row_major(&y_true_raw);
    let (y_pred_vec, n_pred, k_pred) = matrix2_to_row_major(&y_pred_oof_raw);
    if n_true != n_pred || k_true != k_pred {
        return Err(PyValueError::new_err(format!(
            "training matrix shape mismatch: y_true=({n_true},{k_true}) vs y_pred_oof=({n_pred},{k_pred})"
        )));
    }
    if train_sample_ids.len() != n_true {
        return Err(PyValueError::new_err(format!(
            "train_sample_ids length mismatch: {} vs matrix rows={n_true}",
            train_sample_ids.len()
        )));
    }
    if trait_names.len() != k_true {
        return Err(PyValueError::new_err(format!(
            "trait_names length mismatch: {} vs matrix cols={k_true}",
            trait_names.len()
        )));
    }
    let data = TopData::new(train_sample_ids, trait_names, y_true_vec, y_pred_vec)
        .map_err(PyValueError::new_err)?;

    let (cand_vec, m, k) = matrix2_to_row_major(&y_candidate_pred_raw);
    if candidate_ids.len() != m {
        return Err(PyValueError::new_err(format!(
            "candidate_ids length mismatch: {} vs candidate rows={m}",
            candidate_ids.len()
        )));
    }
    if k != data.k {
        return Err(PyValueError::new_err(format!(
            "candidate matrix trait count mismatch: {k} vs training traits={}",
            data.k
        )));
    }
    let records = rank_candidates_to_target_sample(
        &parsed_model,
        &candidate_ids,
        &cand_vec,
        m,
        k,
        &data,
        &target_sample_id,
    )
    .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py).into_bound();
    out.set_item("target_sample_id", target_sample_id)?;
    let rows = PyList::empty(py).into_bound();
    for r in &records {
        let d = PyDict::new(py).into_bound();
        d.set_item("sample_id", r.sample_id.clone())?;
        d.set_item("distance", r.distance)?;
        d.set_item("similarity", r.similarity)?;
        d.set_item("rank", r.rank)?;
        d.set_item("pred_traits_raw", r.pred_traits_raw.clone())?;
        d.set_item("pred_traits_calibrated", r.pred_traits_calibrated.clone())?;
        rows.append(d)?;
    }
    out.set_item("records", rows.clone())?;
    Ok(out)
}

#[pyfunction(name = "top_rank_to_target_values")]
#[pyo3(signature = (
    model,
    trait_names,
    candidate_ids,
    y_candidate_pred_raw,
    target_values
))]
pub(crate) fn top_rank_to_target_values_py<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyDict>,
    trait_names: Vec<String>,
    candidate_ids: Vec<String>,
    y_candidate_pred_raw: PyReadonlyArray2<'py, f64>,
    target_values: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let parsed_model = parse_top_model_from_dict(model)?;
    let (cand_vec, m, k) = matrix2_to_row_major(&y_candidate_pred_raw);
    if candidate_ids.len() != m {
        return Err(PyValueError::new_err(format!(
            "candidate_ids length mismatch: {} vs candidate rows={m}",
            candidate_ids.len()
        )));
    }
    if trait_names.len() != k {
        return Err(PyValueError::new_err(format!(
            "trait_names length mismatch: {} vs candidate cols={k}",
            trait_names.len()
        )));
    }
    if target_values.len() != k {
        return Err(PyValueError::new_err(format!(
            "target_values length mismatch: {} vs traits={k}",
            target_values.len()
        )));
    }
    let records = rank_candidates(
        &parsed_model,
        &candidate_ids,
        &cand_vec,
        m,
        k,
        &target_values,
    )
    .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py).into_bound();
    out.set_item("target_values", target_values)?;
    let rows = PyList::empty(py).into_bound();
    for r in &records {
        let d = PyDict::new(py).into_bound();
        d.set_item("sample_id", r.sample_id.clone())?;
        d.set_item("distance", r.distance)?;
        d.set_item("similarity", r.similarity)?;
        d.set_item("rank", r.rank)?;
        d.set_item("pred_traits_raw", r.pred_traits_raw.clone())?;
        d.set_item("pred_traits_calibrated", r.pred_traits_calibrated.clone())?;
        rows.append(d)?;
    }
    out.set_item("records", rows.clone())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_data() -> TopData {
        TopData::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec!["t1".into(), "t2".into()],
            vec![
                1.0, 2.0, //
                2.0, 3.0, //
                3.0, 5.0, //
                4.0, 7.0,
            ],
            vec![
                0.9, 2.1, //
                2.1, 2.8, //
                2.8, 5.2, //
                4.2, 7.1,
            ],
        )
        .unwrap()
    }

    #[test]
    fn standardizer_roundtrip() {
        let data = toy_data();
        let std = Standardizer::fit(&data.y_true_raw, data.n, data.k).unwrap();
        let mut mat = data.y_true_raw.clone();
        std.transform_inplace(&mut mat, data.n, data.k).unwrap();
        let mut row = vec![10.0, 11.0];
        let _ = std.transform_row(&mut row);
        assert!(mat.iter().all(|v| v.is_finite()));
        assert!(row.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn calibration_linear() {
        let y_true = vec![2.0, 4.0, 6.0, 8.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        let cal = LinearCalibration::fit(&y_true, &y_pred).unwrap();
        assert!((cal.intercept - 0.0).abs() < 1e-9);
        assert!((cal.slope - 2.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_work() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 4.0];
        assert!(pearson(&x, &y).unwrap() > 0.9);
        assert!(spearman(&x, &y).unwrap() > 0.9);
        assert!(rmse(&x, &y).unwrap() > 0.0);
        assert!(mae(&x, &y).unwrap() > 0.0);
        assert!(nrmse(&x, &y).unwrap() > 0.0);
    }

    #[test]
    fn exact_objective_gradient_smoke() {
        let data = toy_data();
        let prepared = prepare_top_training_data(&data, CalibrationMode::Linear).unwrap();
        let weights = vec![0.5, 0.5];
        let obj = exact_objective(
            &prepared.y_pred_std,
            &prepared.y_true_std,
            data.n,
            data.k,
            &weights,
            1e-4,
        )
        .unwrap();
        assert!(obj.loss.is_finite());
        assert_eq!(obj.grad.len(), 2);
        assert_eq!(obj.hess.len(), 4);
        assert!(obj.grad_norm().is_finite());
    }

    #[test]
    fn adam_normalizes() {
        let mut w = vec![2.0, 3.0, 5.0];
        let mut adam = AdamState::new(3);
        adam.step(&mut w, &[1.0, -1.0, 0.5], 1e-2, 0.9, 0.999, 1e-8, true);
        let s: f64 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rank_candidates_works() {
        let data = toy_data();
        let prepared = prepare_top_training_data(&data, CalibrationMode::Linear).unwrap();
        let model = TopModel::new(
            data.trait_names.clone(),
            vec![0.5, 0.5],
            prepared.standardizer,
            vec![
                TraitCalibration {
                    trait_name: "t1".into(),
                    selected_gs_model: String::new(),
                    prediction_type: "raw_oof".into(),
                    intercept: prepared.calibrations[0].intercept,
                    slope: prepared.calibrations[0].slope,
                },
                TraitCalibration {
                    trait_name: "t2".into(),
                    selected_gs_model: String::new(),
                    prediction_type: "raw_oof".into(),
                    intercept: prepared.calibrations[1].intercept,
                    slope: prepared.calibrations[1].slope,
                },
            ],
            0.0,
            data.n,
            TopMode::ExactNewton,
        )
        .unwrap();
        let target = find_target_by_sample_id(&data, "a").unwrap();
        let recs = rank_candidates(
            &model,
            &data.sample_ids,
            &data.y_pred_oof_raw,
            data.n,
            data.k,
            &target,
        )
        .unwrap();
        assert_eq!(recs.len(), data.n);
        assert_eq!(recs[0].rank, 1);
    }

    #[test]
    fn missing_aware_training_and_rank_work() {
        let data = TopData::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec!["t1".into(), "t2".into()],
            vec![
                1.0,
                f64::NAN, //
                f64::NAN,
                3.0, //
                3.0,
                5.0, //
                f64::NAN,
                7.0,
            ],
            vec![
                0.9, 2.1, //
                2.1, 2.8, //
                2.8, 5.2, //
                4.2, 7.1,
            ],
        )
        .unwrap();
        let mut cfg = TopConfig::default();
        cfg.mode = TopMode::MiniBatchAdam;
        cfg.epochs = 3;
        cfg.batch_size = 2;
        let fit = fit_top_model(&data, &cfg).unwrap();
        assert_eq!(fit.model.weights.len(), 2);
        assert!(fit.model.n_train >= 2);
        let recs = rank_candidates(
            &fit.model,
            &data.sample_ids,
            &data.y_pred_oof_raw,
            data.n,
            data.k,
            &[2.0, f64::NAN],
        )
        .unwrap();
        assert_eq!(recs.len(), data.n);
        assert_eq!(recs[0].rank, 1);
    }
}
