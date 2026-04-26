use crate::bitwise::{and_popcount, popcount};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;

const BATCH_PAR_MIN_ROWS: usize = 64;

#[inline]
fn words_for_samples(n: usize) -> usize {
    n.div_ceil(64)
}

#[inline]
fn should_parallel_rows(n_rows: usize) -> bool {
    n_rows >= BATCH_PAR_MIN_ROWS && rayon::current_num_threads() > 1
}

#[inline]
fn require_n_valid(n: usize, y_len: usize, bit_words: usize, ctx: &str) -> Result<(), String> {
    if n > y_len {
        return Err(format!("{ctx}: n_samples={} exceeds y length={}", n, y_len));
    }
    let bit_cap = bit_words.saturating_mul(64);
    if n > bit_cap {
        return Err(format!(
            "{ctx}: n_samples={} exceeds bit capacity={} ({} words)",
            n, bit_cap, bit_words
        ));
    }
    Ok(())
}

#[inline]
fn validate_binary_y(y: &[u8], n_samples: usize, ctx: &str) -> Result<(), String> {
    if let Some((idx, bad)) = y.iter().take(n_samples).enumerate().find(|(_, v)| **v > 1) {
        return Err(format!("{ctx}: y must be binary 0/1; found y[{idx}]={bad}"));
    }
    Ok(())
}

#[inline]
fn pack_binary_y_to_bits(y: &[u8], n_samples: usize) -> (Vec<u64>, u64) {
    let words = words_for_samples(n_samples).max(1);
    let mut out = vec![0u64; words];
    let mut y_pos = 0u64;
    for (i, &yv) in y.iter().take(n_samples).enumerate() {
        if yv != 0 {
            out[i >> 6] |= 1u64 << (i & 63);
            y_pos += 1;
        }
    }
    (out, y_pos)
}

#[inline]
fn masked_xpos_tp(bits: &[u64], y_bits: &[u64], n_samples: usize) -> (u64, u64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;

    let mut x_pos = 0u64;
    let mut tp = 0u64;

    if full_words > 0 {
        x_pos += popcount(&bits[..full_words]);
        tp += and_popcount(&bits[..full_words], &y_bits[..full_words]);
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let xb = bits[full_words] & mask;
        let yb = y_bits[full_words] & mask;
        x_pos += xb.count_ones() as u64;
        tp += (xb & yb).count_ones() as u64;
    }

    (x_pos, tp)
}

#[inline]
fn confusion_from_packed_ybits(
    bits: &[u64],
    y_bits: &[u64],
    y_pos: u64,
    n_samples: usize,
) -> (u64, u64, u64, u64) {
    let (x_pos, tp) = masked_xpos_tp(bits, y_bits, n_samples);
    let fp = x_pos.saturating_sub(tp);
    let fnv = y_pos.saturating_sub(tp);
    let tn = (n_samples as u64).saturating_sub(tp + fp + fnv);
    (tp, tn, fp, fnv)
}

#[inline]
fn ba_from_confusion(tp: u64, tn: u64, fp: u64, fnv: u64) -> f64 {
    let pos = tp + fnv;
    let neg = tn + fp;
    match (pos, neg) {
        (0, 0) => f64::NAN,
        (0, _) => (tn as f64) / (neg as f64),
        (_, 0) => (tp as f64) / (pos as f64),
        _ => {
            let tpr = (tp as f64) / (pos as f64);
            let tnr = (tn as f64) / (neg as f64);
            0.5 * (tpr + tnr)
        }
    }
}

#[inline]
fn mcc_from_confusion(tp: u64, tn: u64, fp: u64, fnv: u64) -> f64 {
    let num = (tp as f64) * (tn as f64) - (fp as f64) * (fnv as f64);
    let a = (tp + fp) as f64;
    let b = (tp + fnv) as f64;
    let c = (tn + fp) as f64;
    let d = (tn + fnv) as f64;
    let den = (a * b * c * d).sqrt();
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

#[inline]
fn y_sum_and_sq(y: &[f64], n_samples: usize) -> (f64, f64) {
    let mut sy = 0.0_f64;
    let mut syy = 0.0_f64;
    for &v in y.iter().take(n_samples) {
        sy += v;
        syy += v * v;
    }
    (sy, syy)
}

#[inline]
fn sum_y_where_bit1(bits: &[u64], y: &[f64], n_samples: usize) -> (u64, f64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;

    let mut n1 = 0u64;
    let mut s1 = 0.0_f64;

    for (w_idx, &word) in bits.iter().take(full_words).enumerate() {
        n1 += word.count_ones() as u64;
        let mut w = word;
        let base = w_idx << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            s1 += y[base + tz];
            w &= w - 1;
        }
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let mut w = bits[full_words] & mask;
        n1 += w.count_ones() as u64;
        let base = full_words << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            s1 += y[base + tz];
            w &= w - 1;
        }
    }

    (n1, s1)
}

#[inline]
fn cont_mean_diff_corr_from_row(
    y: &[f64],
    bits: &[u64],
    n_samples: usize,
    y_sum: f64,
    y_sq_sum: f64,
) -> (f64, f64) {
    let (n1, s1) = sum_y_where_bit1(bits, y, n_samples);
    let n0 = (n_samples as u64).saturating_sub(n1);

    let mean_diff = if n1 == 0 || n0 == 0 {
        f64::NAN
    } else {
        let mean1 = s1 / (n1 as f64);
        let mean0 = (y_sum - s1) / (n0 as f64);
        mean1 - mean0
    };

    let n = n_samples as f64;
    let sx = n1 as f64;
    let sxy = s1;

    let vxx_num = n * sx - sx * sx;
    let vyy_num = n * y_sq_sum - y_sum * y_sum;
    let corr = if !(vxx_num > 0.0 && vyy_num > 0.0) {
        0.0
    } else {
        let cov_num = n * sxy - sx * y_sum;
        (cov_num / (vxx_num.sqrt() * vyy_num.sqrt())).clamp(-1.0, 1.0)
    };

    (mean_diff, corr)
}

/// Balanced Accuracy for binary `y` (0/1) vs packed 0/1 bit-vector.
///
/// `bits` stores predictions/features in packed form (LSB-first in each `u64` word).
pub fn score_binary_ba_packed(y: &[u8], bits: &[u64], n_samples: usize) -> f64 {
    if n_samples == 0 {
        return f64::NAN;
    }
    if require_n_valid(n_samples, y.len(), bits.len(), "score_binary_ba_packed").is_err() {
        return f64::NAN;
    }

    let (y_bits, y_pos) = pack_binary_y_to_bits(y, n_samples);
    let (tp, tn, fp, fnv) = confusion_from_packed_ybits(bits, &y_bits, y_pos, n_samples);
    ba_from_confusion(tp, tn, fp, fnv)
}

/// Matthews Correlation Coefficient for binary `y` (0/1) vs packed 0/1 bit-vector.
pub fn score_binary_mcc_packed(y: &[u8], bits: &[u64], n_samples: usize) -> f64 {
    if n_samples == 0 {
        return f64::NAN;
    }
    if require_n_valid(n_samples, y.len(), bits.len(), "score_binary_mcc_packed").is_err() {
        return f64::NAN;
    }

    let (y_bits, y_pos) = pack_binary_y_to_bits(y, n_samples);
    let (tp, tn, fp, fnv) = confusion_from_packed_ybits(bits, &y_bits, y_pos, n_samples);
    mcc_from_confusion(tp, tn, fp, fnv)
}

/// Mean difference (`mean(x=1) - mean(x=0)`) for continuous `y` vs packed 0/1 bit-vector.
pub fn score_cont_mean_diff_packed(y: &[f64], bits: &[u64], n_samples: usize) -> f64 {
    if n_samples == 0 {
        return f64::NAN;
    }
    if require_n_valid(
        n_samples,
        y.len(),
        bits.len(),
        "score_cont_mean_diff_packed",
    )
    .is_err()
    {
        return f64::NAN;
    }

    let (sy, syy) = y_sum_and_sq(y, n_samples);
    let (md, _r) = cont_mean_diff_corr_from_row(y, bits, n_samples, sy, syy);
    md
}

/// Pearson correlation between continuous `y` and packed 0/1 bit-vector.
pub fn score_cont_corr_packed(y: &[f64], bits: &[u64], n_samples: usize) -> f64 {
    if n_samples == 0 {
        return f64::NAN;
    }
    if require_n_valid(n_samples, y.len(), bits.len(), "score_cont_corr_packed").is_err() {
        return f64::NAN;
    }

    let (sy, syy) = y_sum_and_sq(y, n_samples);
    let (_md, r) = cont_mean_diff_corr_from_row(y, bits, n_samples, sy, syy);
    r
}

/// Batch scoring for binary `y` against many packed bit-vectors.
///
/// `bits_flat` is row-major with `n_rows` rows and `row_words` words per row.
/// Returns `(ba, mcc)`, both length `n_rows`.
pub fn score_binary_ba_mcc_batch_packed(
    y: &[u8],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> (Vec<f64>, Vec<f64>) {
    if n_rows == 0 {
        return (Vec::new(), Vec::new());
    }

    let needed_words = words_for_samples(n_samples).max(1);
    if n_samples == 0
        || require_n_valid(
            n_samples,
            y.len(),
            needed_words,
            "score_binary_ba_mcc_batch_packed",
        )
        .is_err()
        || row_words < needed_words
        || bits_flat.len() < n_rows.saturating_mul(row_words)
    {
        return (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows]);
    }

    let (y_bits, y_pos) = pack_binary_y_to_bits(y, n_samples);

    let pairs: Vec<(f64, f64)> = if should_parallel_rows(n_rows) {
        (0..n_rows)
            .into_par_iter()
            .map(|r| {
                let start = r * row_words;
                let row = &bits_flat[start..start + needed_words];
                let (tp, tn, fp, fnv) = confusion_from_packed_ybits(row, &y_bits, y_pos, n_samples);
                (
                    ba_from_confusion(tp, tn, fp, fnv),
                    mcc_from_confusion(tp, tn, fp, fnv),
                )
            })
            .collect()
    } else {
        (0..n_rows)
            .map(|r| {
                let start = r * row_words;
                let row = &bits_flat[start..start + needed_words];
                let (tp, tn, fp, fnv) = confusion_from_packed_ybits(row, &y_bits, y_pos, n_samples);
                (
                    ba_from_confusion(tp, tn, fp, fnv),
                    mcc_from_confusion(tp, tn, fp, fnv),
                )
            })
            .collect()
    };

    pairs.into_iter().unzip()
}

/// Batch scoring for continuous `y` against many packed bit-vectors.
///
/// `bits_flat` is row-major with `n_rows` rows and `row_words` words per row.
/// Returns `(mean_diff, corr)`, both length `n_rows`.
pub fn score_cont_mean_diff_corr_batch_packed(
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> (Vec<f64>, Vec<f64>) {
    if n_rows == 0 {
        return (Vec::new(), Vec::new());
    }

    let needed_words = words_for_samples(n_samples).max(1);
    if n_samples == 0
        || require_n_valid(
            n_samples,
            y.len(),
            needed_words,
            "score_cont_mean_diff_corr_batch_packed",
        )
        .is_err()
        || row_words < needed_words
        || bits_flat.len() < n_rows.saturating_mul(row_words)
    {
        return (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows]);
    }

    let (sy, syy) = y_sum_and_sq(y, n_samples);

    let pairs: Vec<(f64, f64)> = if should_parallel_rows(n_rows) {
        (0..n_rows)
            .into_par_iter()
            .map(|r| {
                let start = r * row_words;
                let row = &bits_flat[start..start + needed_words];
                cont_mean_diff_corr_from_row(y, row, n_samples, sy, syy)
            })
            .collect()
    } else {
        (0..n_rows)
            .map(|r| {
                let start = r * row_words;
                let row = &bits_flat[start..start + needed_words];
                cont_mean_diff_corr_from_row(y, row, n_samples, sy, syy)
            })
            .collect()
    };

    pairs.into_iter().unzip()
}

#[inline]
fn readonly_u8_to_cow<'a>(arr: &'a PyReadonlyArray1<'a, u8>) -> Cow<'a, [u8]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[inline]
fn readonly_f64_to_cow<'a>(arr: &'a PyReadonlyArray1<'a, f64>) -> Cow<'a, [f64]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[inline]
fn readonly_u64_to_cow<'a>(arr: &'a PyReadonlyArray1<'a, u64>) -> Cow<'a, [u64]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[inline]
fn readonly_u64_2d_to_flat_cow<'a>(arr: &'a PyReadonlyArray2<'a, u64>) -> Cow<'a, [u64]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[pyfunction(name = "score_binary_ba")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_binary_ba_py(
    y: PyReadonlyArray1<'_, u8>,
    bits: PyReadonlyArray1<'_, u64>,
    n_samples: Option<usize>,
) -> PyResult<f64> {
    let yv = readonly_u8_to_cow(&y);
    let bv = readonly_u64_to_cow(&bits);
    let n = n_samples.unwrap_or(yv.len());
    require_n_valid(n, yv.len(), bv.len(), "score_binary_ba").map_err(PyRuntimeError::new_err)?;
    validate_binary_y(yv.as_ref(), n, "score_binary_ba").map_err(PyRuntimeError::new_err)?;
    Ok(score_binary_ba_packed(yv.as_ref(), bv.as_ref(), n))
}

#[pyfunction(name = "score_binary_mcc")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_binary_mcc_py(
    y: PyReadonlyArray1<'_, u8>,
    bits: PyReadonlyArray1<'_, u64>,
    n_samples: Option<usize>,
) -> PyResult<f64> {
    let yv = readonly_u8_to_cow(&y);
    let bv = readonly_u64_to_cow(&bits);
    let n = n_samples.unwrap_or(yv.len());
    require_n_valid(n, yv.len(), bv.len(), "score_binary_mcc").map_err(PyRuntimeError::new_err)?;
    validate_binary_y(yv.as_ref(), n, "score_binary_mcc").map_err(PyRuntimeError::new_err)?;
    Ok(score_binary_mcc_packed(yv.as_ref(), bv.as_ref(), n))
}

#[pyfunction(name = "score_cont_mean_diff")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_cont_mean_diff_py(
    y: PyReadonlyArray1<'_, f64>,
    bits: PyReadonlyArray1<'_, u64>,
    n_samples: Option<usize>,
) -> PyResult<f64> {
    let yv = readonly_f64_to_cow(&y);
    let bv = readonly_u64_to_cow(&bits);
    let n = n_samples.unwrap_or(yv.len());
    require_n_valid(n, yv.len(), bv.len(), "score_cont_mean_diff")
        .map_err(PyRuntimeError::new_err)?;
    Ok(score_cont_mean_diff_packed(yv.as_ref(), bv.as_ref(), n))
}

#[pyfunction(name = "score_cont_corr")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_cont_corr_py(
    y: PyReadonlyArray1<'_, f64>,
    bits: PyReadonlyArray1<'_, u64>,
    n_samples: Option<usize>,
) -> PyResult<f64> {
    let yv = readonly_f64_to_cow(&y);
    let bv = readonly_u64_to_cow(&bits);
    let n = n_samples.unwrap_or(yv.len());
    require_n_valid(n, yv.len(), bv.len(), "score_cont_corr").map_err(PyRuntimeError::new_err)?;
    Ok(score_cont_corr_packed(yv.as_ref(), bv.as_ref(), n))
}

#[pyfunction(name = "score_binary_ba_mcc_batch")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_binary_ba_mcc_batch_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, u8>,
    bits: PyReadonlyArray2<'py, u64>,
    n_samples: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let yv = readonly_u8_to_cow(&y);
    let bits_arr = bits.as_array();
    if bits_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "score_binary_ba_mcc_batch: bits must be 2D",
        ));
    }

    let n_rows = bits_arr.shape()[0];
    let row_words = bits_arr.shape()[1];
    let n = n_samples.unwrap_or(yv.len());
    let needed_words = words_for_samples(n).max(1);

    require_n_valid(n, yv.len(), needed_words, "score_binary_ba_mcc_batch")
        .map_err(PyRuntimeError::new_err)?;
    validate_binary_y(yv.as_ref(), n, "score_binary_ba_mcc_batch")
        .map_err(PyRuntimeError::new_err)?;

    if row_words < needed_words {
        return Err(PyRuntimeError::new_err(format!(
            "score_binary_ba_mcc_batch: bits row_words={} is smaller than required {} for n_samples={}",
            row_words, needed_words, n
        )));
    }

    let bv = readonly_u64_2d_to_flat_cow(&bits);
    let (ba, mcc) =
        score_binary_ba_mcc_batch_packed(yv.as_ref(), bv.as_ref(), row_words, n_rows, n);

    let out_ba = PyArray1::<f64>::zeros(py, [n_rows], false);
    let out_mcc = PyArray1::<f64>::zeros(py, [n_rows], false);

    let out_ba_slice = unsafe {
        out_ba.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("score_binary_ba_mcc_batch: out_ba not contiguous")
        })?
    };
    let out_mcc_slice = unsafe {
        out_mcc.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("score_binary_ba_mcc_batch: out_mcc not contiguous")
        })?
    };
    out_ba_slice.copy_from_slice(&ba);
    out_mcc_slice.copy_from_slice(&mcc);

    Ok((out_ba, out_mcc))
}

#[pyfunction(name = "score_cont_mean_diff_corr_batch")]
#[pyo3(signature = (y, bits, n_samples=None))]
pub fn score_cont_mean_diff_corr_batch_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    bits: PyReadonlyArray2<'py, u64>,
    n_samples: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let yv = readonly_f64_to_cow(&y);
    let bits_arr = bits.as_array();
    if bits_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "score_cont_mean_diff_corr_batch: bits must be 2D",
        ));
    }

    let n_rows = bits_arr.shape()[0];
    let row_words = bits_arr.shape()[1];
    let n = n_samples.unwrap_or(yv.len());
    let needed_words = words_for_samples(n).max(1);

    require_n_valid(n, yv.len(), needed_words, "score_cont_mean_diff_corr_batch")
        .map_err(PyRuntimeError::new_err)?;

    if row_words < needed_words {
        return Err(PyRuntimeError::new_err(format!(
            "score_cont_mean_diff_corr_batch: bits row_words={} is smaller than required {} for n_samples={}",
            row_words, needed_words, n
        )));
    }

    let bv = readonly_u64_2d_to_flat_cow(&bits);
    let (md, corr) =
        score_cont_mean_diff_corr_batch_packed(yv.as_ref(), bv.as_ref(), row_words, n_rows, n);

    let out_md = PyArray1::<f64>::zeros(py, [n_rows], false);
    let out_corr = PyArray1::<f64>::zeros(py, [n_rows], false);

    let out_md_slice = unsafe {
        out_md.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("score_cont_mean_diff_corr_batch: out_md not contiguous")
        })?
    };
    let out_corr_slice = unsafe {
        out_corr.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("score_cont_mean_diff_corr_batch: out_corr not contiguous")
        })?
    };
    out_md_slice.copy_from_slice(&md);
    out_corr_slice.copy_from_slice(&corr);

    Ok((out_md, out_corr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack01(v: &[u8]) -> Vec<u64> {
        let words = v.len().div_ceil(64);
        let mut out = vec![0u64; words.max(1)];
        for (i, &b) in v.iter().enumerate() {
            if b != 0 {
                out[i >> 6] |= 1u64 << (i & 63);
            }
        }
        out
    }

    #[test]
    fn test_binary_scores_perfect() {
        let y = [0u8, 1, 1, 0, 1, 0, 0, 1];
        let b = pack01(&y);
        assert!((score_binary_ba_packed(&y, &b, y.len()) - 1.0).abs() < 1e-12);
        assert!((score_binary_mcc_packed(&y, &b, y.len()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_binary_scores_half_random() {
        let y = [0u8, 0, 1, 1];
        let x = [0u8, 1, 0, 1];
        let b = pack01(&x);
        assert!((score_binary_ba_packed(&y, &b, y.len()) - 0.5).abs() < 1e-12);
        assert!(score_binary_mcc_packed(&y, &b, y.len()).abs() < 1e-12);
    }

    #[test]
    fn test_binary_tail_mask_ignored() {
        let n = 70usize;
        let mut y = vec![0u8; n];
        let mut x = vec![0u8; n];
        for i in (0..n).step_by(3) {
            y[i] = 1;
        }
        for i in (1..n).step_by(4) {
            x[i] = 1;
        }
        let mut b = pack01(&x);
        // Dirty tail bits outside n should not affect score.
        b[1] |= !((1u64 << (n & 63)) - 1u64);

        let ba1 = score_binary_ba_packed(&y, &b, n);
        let mcc1 = score_binary_mcc_packed(&y, &b, n);

        let mut b_clean = b.clone();
        b_clean[1] &= (1u64 << (n & 63)) - 1u64;
        let ba2 = score_binary_ba_packed(&y, &b_clean, n);
        let mcc2 = score_binary_mcc_packed(&y, &b_clean, n);

        assert!((ba1 - ba2).abs() < 1e-12);
        assert!((mcc1 - mcc2).abs() < 1e-12);
    }

    #[test]
    fn test_cont_mean_diff_and_corr() {
        let y = [1.0_f64, 2.0, 3.0, 4.0];
        let x = [0u8, 0, 1, 1];
        let b = pack01(&x);
        let md = score_cont_mean_diff_packed(&y, &b, y.len());
        let r = score_cont_corr_packed(&y, &b, y.len());
        assert!((md - 2.0).abs() < 1e-12);
        assert!((r - 0.894_427_190_999_915_9).abs() < 1e-12);
    }

    #[test]
    fn test_cont_corr_constant_bit() {
        let y = [1.0_f64, 2.0, 3.0, 4.0];
        let x = [1u8, 1, 1, 1];
        let b = pack01(&x);
        assert_eq!(score_cont_corr_packed(&y, &b, y.len()), 0.0);
        assert!(score_cont_mean_diff_packed(&y, &b, y.len()).is_nan());
    }

    #[test]
    fn test_binary_batch_matches_single() {
        let n = 100usize;
        let y: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
        let row_words = words_for_samples(n) + 1; // padded row
        let n_rows = 4usize;

        let mut bits_flat = vec![0u64; n_rows * row_words];
        for r in 0..n_rows {
            for i in 0..n {
                let b = (((i + r) % (r + 2)) == 0) as u8;
                if b != 0 {
                    bits_flat[r * row_words + (i >> 6)] |= 1u64 << (i & 63);
                }
            }
            // fill extra word with noise, should be ignored
            bits_flat[r * row_words + row_words - 1] = u64::MAX;
        }

        let (ba, mcc) = score_binary_ba_mcc_batch_packed(&y, &bits_flat, row_words, n_rows, n);

        for r in 0..n_rows {
            let row = &bits_flat[r * row_words..r * row_words + words_for_samples(n)];
            let ba_s = score_binary_ba_packed(&y, row, n);
            let mcc_s = score_binary_mcc_packed(&y, row, n);
            assert!((ba[r] - ba_s).abs() < 1e-12);
            assert!((mcc[r] - mcc_s).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cont_batch_matches_single() {
        let n = 96usize;
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 - 7.0).collect();
        let row_words = words_for_samples(n);
        let n_rows = 3usize;

        let mut bits_flat = vec![0u64; n_rows * row_words];
        for r in 0..n_rows {
            for i in 0..n {
                if ((i * (r + 1)) % 7) < 3 {
                    bits_flat[r * row_words + (i >> 6)] |= 1u64 << (i & 63);
                }
            }
        }

        let (md, corr) =
            score_cont_mean_diff_corr_batch_packed(&y, &bits_flat, row_words, n_rows, n);

        for r in 0..n_rows {
            let row = &bits_flat[r * row_words..(r + 1) * row_words];
            let md_s = score_cont_mean_diff_packed(&y, row, n);
            let corr_s = score_cont_corr_packed(&y, row, n);
            if md_s.is_nan() {
                assert!(md[r].is_nan());
            } else {
                assert!((md[r] - md_s).abs() < 1e-12);
            }
            assert!((corr[r] - corr_s).abs() < 1e-12);
        }
    }
}
