use crate::ml::common::ResponseKind;

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

pub fn feature_scores_abs_corr_binary_x(x_rows: &[Vec<u8>], y: &[f64]) -> Vec<f64> {
    let n = y.len() as f64;
    let sumy: f64 = y.iter().sum();
    let sumy2: f64 = y.iter().map(|v| v * v).sum();
    let meany = sumy / n;
    let vary = (sumy2 / n) - meany * meany;
    if !vary.is_finite() || vary <= 0.0 {
        return vec![0.0; x_rows.len()];
    }

    let mut out = vec![0.0f64; x_rows.len()];
    for (r, row) in x_rows.iter().enumerate() {
        let mut n1 = 0usize;
        let mut sum1 = 0.0f64;
        for i in 0..row.len() {
            if row[i] != 0 {
                n1 += 1;
                sum1 += y[i];
            }
        }
        let p = (n1 as f64) / n;
        let varx = p * (1.0 - p);
        if varx <= 0.0 {
            out[r] = 0.0;
            continue;
        }
        let exy = sum1 / n;
        let cov = exy - p * meany;
        let corr = cov / (varx * vary).sqrt();
        out[r] = corr.abs();
    }
    out
}

pub fn feature_scores_abs_mean_diff_binary_x(x_rows: &[Vec<u8>], y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; x_rows.len()];
    for (r, row) in x_rows.iter().enumerate() {
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
            out[r] = 0.0;
            continue;
        }
        let m1 = s1 / (n1 as f64);
        let m0 = s0 / (n0 as f64);
        out[r] = (m1 - m0).abs();
    }
    out
}

pub fn feature_scores_abs_mcc_binary_x(x_rows: &[Vec<u8>], y: &[f64]) -> Vec<f64> {
    let yb: Vec<u8> = y.iter().map(|v| if *v > 0.5 { 1 } else { 0 }).collect();
    let y_pos = yb.iter().map(|v| *v as u64).sum::<u64>();
    let n = yb.len() as u64;
    let y_neg = n.saturating_sub(y_pos);

    let mut out = vec![0.0f64; x_rows.len()];
    for (r, row) in x_rows.iter().enumerate() {
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
        out[r] = mcc_from_confusion(tp, tn, fp, fnv).abs();
    }
    out
}

pub fn feature_scores_fisher_binary_x(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
) -> Vec<f64> {
    let eps = 1e-12f64;
    let mut out = vec![0.0f64; x_rows.len()];
    for (r, row) in x_rows.iter().enumerate() {
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
            out[r] = 0.0;
            continue;
        }

        let m1 = s1 / (n1 as f64);
        let m0 = s0 / (n0 as f64);
        if response == ResponseKind::Binary {
            // For binary response this simplifies to a robust separation score.
            out[r] = (m1 - m0).abs();
            continue;
        }

        let v1 = (q1 / (n1 as f64) - m1 * m1).max(0.0);
        let v0 = (q0 / (n0 as f64) - m0 * m0).max(0.0);
        let between = (m1 - m0) * (m1 - m0);
        let within = v1 + v0 + eps;
        out[r] = between / within;
    }
    out
}
