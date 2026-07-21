use crate::ml::common::ResponseKind;
use rayon::prelude::*;

const UNIVARIATE_PAR_MIN_ROWS: usize = 64;
const UNIVARIATE_PAR_MIN_SAMPLES: usize = 256;

#[inline]
fn mcc_from_confusion(tp: u64, tn: u64, fp: u64, fnv: u64) -> f64 {
    let num = (tp as f64) * (tn as f64) - (fp as f64) * (fnv as f64);
    let den = ((tp + fp) as f64) * ((tp + fnv) as f64) * ((tn + fp) as f64) * ((tn + fnv) as f64);
    if den <= 0.0 {
        0.0
    } else {
        num / den.sqrt()
    }
}

#[inline]
fn should_parallel_row_scores(n_rows: usize, n_samples: usize) -> bool {
    rayon::current_num_threads() > 1
        && n_rows >= UNIVARIATE_PAR_MIN_ROWS
        && n_samples >= UNIVARIATE_PAR_MIN_SAMPLES
}

#[inline]
fn corr_score_abs_row(row: &[u8], y: &[f64], n: f64, meany: f64, vary: f64) -> f64 {
    let mut sumx = 0.0f64;
    let mut sumx2 = 0.0f64;
    let mut sumxy = 0.0f64;
    for i in 0..row.len() {
        let x = f64::from(row[i]);
        sumx += x;
        sumx2 += x * x;
        sumxy += x * y[i];
    }
    let meanx = sumx / n;
    let varx = (sumx2 / n) - meanx * meanx;
    if !varx.is_finite() || varx <= 0.0 {
        return 0.0;
    }
    let exy = sumxy / n;
    let cov = exy - meanx * meany;
    let corr = cov / (varx * vary).sqrt();
    corr.abs()
}

#[inline]
fn mean_diff_score_abs_row(row: &[u8], y: &[f64]) -> f64 {
    let mut n1 = 0usize;
    let mut s1 = 0.0f64;
    let mut n0 = 0usize;
    let mut s0 = 0.0f64;
    for i in 0..row.len() {
        if row[i] != 0 {
            n1 += 1;
            s1 += y[i];
        } else {
            n0 += 1;
            s0 += y[i];
        }
    }
    if n0 == 0 || n1 == 0 {
        return 0.0;
    }
    let m1 = s1 / (n1 as f64);
    let m0 = s0 / (n0 as f64);
    (m1 - m0).abs()
}

#[inline]
fn mcc_score_abs_row(row: &[u8], yb: &[u8], y_pos: u64, y_neg: u64) -> f64 {
    let mut tp = 0u64;
    let mut pred_pos = 0u64;
    for i in 0..row.len() {
        if row[i] != 0 {
            pred_pos += 1;
            tp += yb[i] as u64;
        }
    }
    let fp = pred_pos.saturating_sub(tp);
    let fnv = y_pos.saturating_sub(tp);
    let tn = y_neg.saturating_sub(fp);
    mcc_from_confusion(tp, tn, fp, fnv).abs()
}

#[inline]
fn fisher_score_row(row: &[u8], y: &[f64], response: ResponseKind) -> f64 {
    let eps = 1e-12f64;
    let mut n1 = 0usize;
    let mut s1 = 0.0f64;
    let mut q1 = 0.0f64;
    let mut n0 = 0usize;
    let mut s0 = 0.0f64;
    let mut q0 = 0.0f64;
    for i in 0..row.len() {
        let v = y[i];
        if row[i] != 0 {
            n1 += 1;
            s1 += v;
            q1 += v * v;
        } else {
            n0 += 1;
            s0 += v;
            q0 += v * v;
        }
    }
    if n0 == 0 || n1 == 0 {
        return 0.0;
    }

    let m1 = s1 / (n1 as f64);
    let m0 = s0 / (n0 as f64);
    if response == ResponseKind::Binary {
        return (m1 - m0).abs();
    }

    let v1 = (q1 / (n1 as f64) - m1 * m1).max(0.0);
    let v0 = (q0 / (n0 as f64) - m0 * m0).max(0.0);
    let between = (m1 - m0) * (m1 - m0);
    let within = v1 + v0 + eps;
    between / within
}

pub fn feature_scores_abs_corr_dosage_x(x_rows: &[Vec<u8>], y: &[f64]) -> Vec<f64> {
    if y.is_empty() {
        return vec![0.0; x_rows.len()];
    }
    let n = y.len() as f64;
    let sumy: f64 = y.iter().sum();
    let sumy2: f64 = y.iter().map(|v| v * v).sum();
    let meany = sumy / n;
    let vary = (sumy2 / n) - meany * meany;
    if !vary.is_finite() || vary <= 0.0 {
        return vec![0.0; x_rows.len()];
    }
    if should_parallel_row_scores(x_rows.len(), y.len()) {
        x_rows
            .par_iter()
            .map(|row| corr_score_abs_row(row.as_slice(), y, n, meany, vary))
            .collect()
    } else {
        x_rows
            .iter()
            .map(|row| corr_score_abs_row(row.as_slice(), y, n, meany, vary))
            .collect()
    }
}

pub fn feature_scores_abs_corr_stage1(
    sum_x: &[f64],
    sum_x2: &[f64],
    sum_xy: &[f64],
    y: &[f64],
) -> Vec<f64> {
    feature_scores_abs_corr_stage1_with_parallel(sum_x, sum_x2, sum_xy, y, true)
}

pub fn feature_scores_abs_corr_stage1_with_parallel(
    sum_x: &[f64],
    sum_x2: &[f64],
    sum_xy: &[f64],
    y: &[f64],
    allow_parallel: bool,
) -> Vec<f64> {
    if y.is_empty() {
        return vec![0.0; sum_x.len()];
    }
    let n_features = sum_x.len().min(sum_x2.len()).min(sum_xy.len());
    let n = y.len() as f64;
    let sumy: f64 = y.iter().sum();
    let sumy2: f64 = y.iter().map(|v| v * v).sum();
    let meany = sumy / n;
    let vary = (sumy2 / n) - meany * meany;
    if !vary.is_finite() || vary <= 0.0 {
        return vec![0.0; n_features];
    }
    if allow_parallel && should_parallel_row_scores(n_features, y.len()) {
        (0..n_features)
            .into_par_iter()
            .map(|i| {
                let meanx = sum_x[i] / n;
                let varx = (sum_x2[i] / n) - meanx * meanx;
                if !varx.is_finite() || varx <= 0.0 {
                    return 0.0;
                }
                let cov = (sum_xy[i] / n) - meanx * meany;
                (cov / (varx * vary).sqrt()).abs()
            })
            .collect()
    } else {
        (0..n_features)
            .map(|i| {
                let meanx = sum_x[i] / n;
                let varx = (sum_x2[i] / n) - meanx * meanx;
                if !varx.is_finite() || varx <= 0.0 {
                    return 0.0;
                }
                let cov = (sum_xy[i] / n) - meanx * meany;
                (cov / (varx * vary).sqrt()).abs()
            })
            .collect()
    }
}

pub fn feature_scores_abs_mean_diff_binary_x(x_rows: &[Vec<u8>], y: &[f64]) -> Vec<f64> {
    if should_parallel_row_scores(x_rows.len(), y.len()) {
        x_rows
            .par_iter()
            .map(|row| mean_diff_score_abs_row(row.as_slice(), y))
            .collect()
    } else {
        x_rows
            .iter()
            .map(|row| mean_diff_score_abs_row(row.as_slice(), y))
            .collect()
    }
}

pub fn feature_scores_abs_mcc_binary_x(x_rows: &[Vec<u8>], y: &[f64]) -> Vec<f64> {
    let yb: Vec<u8> = y.iter().map(|v| if *v > 0.5 { 1 } else { 0 }).collect();
    let y_pos = yb.iter().map(|v| *v as u64).sum::<u64>();
    let n = yb.len() as u64;
    let y_neg = n.saturating_sub(y_pos);
    if should_parallel_row_scores(x_rows.len(), y.len()) {
        x_rows
            .par_iter()
            .map(|row| mcc_score_abs_row(row.as_slice(), yb.as_slice(), y_pos, y_neg))
            .collect()
    } else {
        x_rows
            .iter()
            .map(|row| mcc_score_abs_row(row.as_slice(), yb.as_slice(), y_pos, y_neg))
            .collect()
    }
}

pub fn feature_scores_fisher_binary_x(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
) -> Vec<f64> {
    if should_parallel_row_scores(x_rows.len(), y.len()) {
        x_rows
            .par_iter()
            .map(|row| fisher_score_row(row.as_slice(), y, response))
            .collect()
    } else {
        x_rows
            .iter()
            .map(|row| fisher_score_row(row.as_slice(), y, response))
            .collect()
    }
}
