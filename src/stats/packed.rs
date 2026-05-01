use matrixmultiply::sgemm;
use numpy::ndarray::{Array1, Array2};
use numpy::PyArray1;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::bedmath::{
    decode_plink_bed_hardcall, decode_row_centered_full_lut, is_identity_indices, packed_byte_lut,
};
use crate::blas::{cblas_sgemm_dispatch, CblasInt, CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS};
use crate::stats_common::{
    check_ctrlc, get_cached_pool, map_err_string_to_py, parse_index_vec_i64,
};

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn signed_hash_bucket_sign(seed: u64, row_idx: usize, n_buckets: usize) -> (usize, f32) {
    let key = (row_idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let h_bucket = splitmix64(seed ^ key);
    let bucket = (h_bucket % (n_buckets as u64)) as usize;
    let h_sign = splitmix64(seed.wrapping_add(0x517CC1B727220A95) ^ key.rotate_left(17));
    let sign = if (h_sign & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
    (bucket, sign)
}

#[pyfunction]
#[pyo3(signature = (packed, n_samples))]
pub fn bed_packed_row_flip_mask<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }
    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let byte_lut = packed_byte_lut();
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;

    let out = PyArray1::<bool>::zeros(py, [m], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };

    out_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(row_idx, dst)| {
            let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
            let mut alt_sum = 0usize;
            let mut non_missing = 0usize;
            for &b in row.iter().take(full_bytes) {
                let idx = b as usize;
                non_missing += byte_lut.nonmiss[idx] as usize;
                alt_sum += byte_lut.alt_sum[idx] as usize;
            }
            if rem > 0 {
                let codes = &byte_lut.code4[row[full_bytes] as usize];
                for &code in codes.iter().take(rem) {
                    match code {
                        0b00 => {
                            non_missing += 1;
                        }
                        0b10 => {
                            non_missing += 1;
                            alt_sum += 1;
                        }
                        0b11 => {
                            non_missing += 1;
                            alt_sum += 2;
                        }
                        _ => {}
                    }
                }
            }
            if non_missing == 0 {
                *dst = false;
            } else {
                let p = (alt_sum as f64) / (2.0 * non_missing as f64);
                *dst = p > 0.5;
            }
        });

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_indices,
    row_flip,
    row_maf,
    sample_indices=None
))]
pub fn bed_packed_decode_rows_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip = row_flip.as_slice()?;
    let row_maf = row_maf.as_slice()?;
    if row_flip.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip.len()
        )));
    }
    if row_maf.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        )));
    }

    let row_idx_raw = row_indices.as_slice()?;
    let row_idx = parse_index_vec_i64(row_idx_raw, m, "row_indices")?;

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n_out = sample_idx.len();

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };

    let mut out = vec![0.0_f32; row_idx.len() * n_out];
    out.par_chunks_mut(n_out)
        .enumerate()
        .for_each(|(i_row, out_row)| {
            let src_row = row_idx[i_row];
            let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
            let flip = row_flip[src_row];
            let mean_g = (2.0_f32 * row_maf[src_row]).max(0.0);
            for (j, &sidx) in sample_idx.iter().enumerate() {
                let b = row[sidx >> 2];
                let code = (b >> ((sidx & 3) * 2)) & 0b11;
                let mut gv = match code {
                    0b00 => 0.0_f32,
                    0b10 => 1.0_f32,
                    0b11 => 2.0_f32,
                    _ => mean_g,
                };
                if flip && code != 0b01 {
                    gv = 2.0_f32 - gv;
                }
                out_row[j] = gv;
            }
        });

    let arr = PyArray2::<f32>::zeros(py, [row_idx.len(), n_out], false).into_bound();
    let arr_slice = unsafe {
        arr.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    arr_slice.copy_from_slice(&out);
    Ok(arr)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    n_buckets,
    seed=20260428_u64,
    sample_indices=None,
    row_missing=None,
    min_maf=0.0_f32,
    max_missing=1.0_f32,
    standardize=true,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn bed_packed_signed_hash_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    n_buckets: usize,
    seed: u64,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    min_maf: f32,
    max_missing: f32,
    standardize: bool,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, f32, usize)> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if n_buckets == 0 {
        return Err(PyRuntimeError::new_err("n_buckets must be > 0"));
    }
    if !(min_maf.is_finite() && min_maf >= 0.0_f32 && min_maf <= 0.5_f32) {
        return Err(PyRuntimeError::new_err(
            "min_maf must be finite and in [0, 0.5]",
        ));
    }
    if !(max_missing.is_finite() && max_missing >= 0.0_f32 && max_missing <= 1.0_f32) {
        return Err(PyRuntimeError::new_err(
            "max_missing must be finite and in [0, 1]",
        ));
    }

    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }
    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }
    let row_missing_vec: Option<Vec<f32>> = match row_missing {
        Some(arr) => {
            let v: Vec<f32> = match arr.as_slice() {
                Ok(s) => s.to_vec(),
                Err(_) => arr.as_array().iter().copied().collect(),
            };
            if v.len() != m {
                return Err(PyRuntimeError::new_err(format!(
                    "row_missing length mismatch: got {}, expected {m}",
                    v.len()
                )));
            }
            Some(v)
        }
        None => None,
    };

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let sample_lut: Vec<(usize, u8)> = sample_idx
        .iter()
        .map(|&s| (s >> 2, ((s & 3) * 2) as u8))
        .collect();

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let out_len = n_buckets
        .checked_mul(n_out)
        .ok_or_else(|| PyRuntimeError::new_err("hashed output size overflow"))?;
    let pool = get_cached_pool(threads)?;

    let (mut out, scale, kept_snps) = py
        .detach(|| -> Result<(Vec<f32>, f32, usize), String> {
            let mut rows_by_bucket: Vec<Vec<usize>> = (0..n_buckets).map(|_| Vec::new()).collect();
            let mut sign_by_row = vec![0.0_f32; m];
            let mut kept: usize = 0;

            for row_idx in 0..m {
                let maf = row_maf_vec[row_idx];
                if !maf.is_finite() {
                    continue;
                }
                if maf < min_maf || maf > 0.5_f32 {
                    continue;
                }
                if let Some(miss) = row_missing_vec.as_ref().map(|x| x[row_idx]) {
                    if !miss.is_finite() || miss > max_missing {
                        continue;
                    }
                }
                if standardize {
                    let var = 2.0_f32 * maf * (1.0_f32 - maf);
                    if !(var.is_finite() && var > 1e-12_f32) {
                        continue;
                    }
                }
                let (bucket, sign) = signed_hash_bucket_sign(seed, row_idx, n_buckets);
                rows_by_bucket[bucket].push(row_idx);
                sign_by_row[row_idx] = sign;
                kept += 1;
            }

            if kept == 0 {
                return Err(
                    "No SNPs left after signed hashing filters; relax min_maf/max_missing."
                        .to_string(),
                );
            }

            let mut out = vec![0.0_f32; out_len];
            let notify_step = if progress_every == 0 {
                (kept / 200).max(1)
            } else {
                progress_every.max(1)
            };
            let progress_done = AtomicUsize::new(0);
            let mut last_notified = 0usize;
            let bucket_step = 64usize;

            for b0 in (0..n_buckets).step_by(bucket_step) {
                let b1 = (b0 + bucket_step).min(n_buckets);
                let chunk = &rows_by_bucket[b0..b1];
                let chunk_done: usize = chunk.iter().map(|rows| rows.len()).sum();
                let out_st = b0 * n_out;
                let out_ed = b1 * n_out;
                let out_slice = &mut out[out_st..out_ed];

                let mut run = || {
                    out_slice
                        .par_chunks_mut(n_out)
                        .enumerate()
                        .for_each(|(off, out_row)| {
                            let rows = &chunk[off];
                            if rows.is_empty() {
                                return;
                            }
                            for &row_idx in rows.iter() {
                                let row = &packed_flat
                                    [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                let flip = row_flip_vec[row_idx];
                                let maf = row_maf_vec[row_idx].clamp(0.0_f32, 0.5_f32);
                                let mean_g = 2.0_f32 * maf;
                                let denom = if standardize {
                                    (2.0_f32 * maf * (1.0_f32 - maf)).max(1e-12_f32).sqrt()
                                } else {
                                    1.0_f32
                                };
                                let sign = sign_by_row[row_idx];
                                for (j, &(byte_idx, shift_bits)) in sample_lut.iter().enumerate() {
                                    let b = row[byte_idx];
                                    let code = (b >> shift_bits) & 0b11;
                                    let mut gv = match code {
                                        0b00 => 0.0_f32,
                                        0b10 => 1.0_f32,
                                        0b11 => 2.0_f32,
                                        _ => mean_g,
                                    };
                                    if flip && code != 0b01 {
                                        gv = 2.0_f32 - gv;
                                    }
                                    let v = if standardize {
                                        (gv - mean_g) / denom
                                    } else {
                                        gv
                                    };
                                    out_row[j] += sign * v;
                                }
                            }
                        });
                };
                if let Some(tp) = &pool {
                    tp.install(&mut run);
                } else {
                    run();
                }

                let done = progress_done.fetch_add(chunk_done, Ordering::Relaxed) + chunk_done;
                if done >= last_notified.saturating_add(notify_step) || done == kept {
                    last_notified = done;
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done, kept))?;
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                    } else {
                        Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                    }
                }
            }

            let mut sum_diag = 0.0_f64;
            for j in 0..n_out {
                let mut acc = 0.0_f64;
                let mut idx = j;
                for _ in 0..n_buckets {
                    let v = out[idx] as f64;
                    acc += v * v;
                    idx += n_out;
                }
                sum_diag += acc;
            }
            let mean_diag = sum_diag / (n_out as f64);
            let mut scale = mean_diag.sqrt() as f32;
            if !scale.is_finite() || scale <= 0.0_f32 {
                scale = 1.0_f32;
            } else {
                let inv_scale = 1.0_f32 / scale;
                out.iter_mut().for_each(|v| *v *= inv_scale);
            }

            Ok((out, scale, kept))
        })
        .map_err(map_err_string_to_py)?;

    let arr = PyArray2::<f32>::zeros(py, [n_buckets, n_out], false).into_bound();
    let arr_slice = unsafe {
        arr.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    arr_slice.copy_from_slice(&out);
    out.clear();
    Ok((arr, scale, kept_snps))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    n_buckets,
    train_sample_indices,
    y_train,
    seed=20260428_u64,
    row_missing=None,
    min_maf=0.0_f32,
    max_missing=1.0_f32,
    standardize=true,
    threads=0,
    sample_block=1024,
    progress_callback=None,
    progress_every=0
))]
pub fn bed_packed_signed_hash_ztz_stats_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    n_buckets: usize,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    y_train: PyReadonlyArray1<'py, f64>,
    seed: u64,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    min_maf: f32,
    max_missing: f32,
    standardize: bool,
    threads: usize,
    sample_block: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    usize,
)> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if n_buckets == 0 {
        return Err(PyRuntimeError::new_err("n_buckets must be > 0"));
    }
    if !(min_maf.is_finite() && min_maf >= 0.0_f32 && min_maf <= 0.5_f32) {
        return Err(PyRuntimeError::new_err(
            "min_maf must be finite and in [0, 0.5]",
        ));
    }
    if !(max_missing.is_finite() && max_missing >= 0.0_f32 && max_missing <= 1.0_f32) {
        return Err(PyRuntimeError::new_err(
            "max_missing must be finite and in [0, 1]",
        ));
    }

    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }
    let row_missing_vec: Option<Vec<f32>> = match row_missing {
        Some(arr) => {
            let v: Vec<f32> = match arr.as_slice() {
                Ok(s) => s.to_vec(),
                Err(_) => arr.as_array().iter().copied().collect(),
            };
            if v.len() != m {
                return Err(PyRuntimeError::new_err(format!(
                    "row_missing length mismatch: got {}, expected {m}",
                    v.len()
                )));
            }
            Some(v)
        }
        None => None,
    };

    let train_idx = parse_index_vec_i64(
        train_sample_indices.as_slice()?,
        n_samples,
        "train_sample_indices",
    )?;
    if train_idx.is_empty() {
        return Err(PyRuntimeError::new_err(
            "train_sample_indices must not be empty",
        ));
    }
    let n_train = train_idx.len();
    let y_vec: Cow<[f64]> = match y_train.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(y_train.as_array().iter().copied().collect()),
    };
    if y_vec.len() != n_train {
        return Err(PyRuntimeError::new_err(format!(
            "y_train length mismatch: got {}, expected {}",
            y_vec.len(),
            n_train
        )));
    }
    let sample_lut: Vec<(usize, u8)> = train_idx
        .iter()
        .map(|&s| (s >> 2, ((s & 3) * 2) as u8))
        .collect();

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;

    let (gram_vec, row_sum_vec, row_sq_sum_vec, zy_vec, scale, kept_snps) = py
        .detach(
            || -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64, usize), String> {
                let mut rows_by_bucket: Vec<Vec<usize>> =
                    (0..n_buckets).map(|_| Vec::new()).collect();
                let mut sign_by_row = vec![0.0_f32; m];
                let mut kept: usize = 0;
                for row_idx in 0..m {
                    let maf = row_maf_vec[row_idx];
                    if !maf.is_finite() {
                        continue;
                    }
                    if maf < min_maf || maf > 0.5_f32 {
                        continue;
                    }
                    if let Some(miss) = row_missing_vec.as_ref().map(|x| x[row_idx]) {
                        if !miss.is_finite() || miss > max_missing {
                            continue;
                        }
                    }
                    if standardize {
                        let var = 2.0_f32 * maf * (1.0_f32 - maf);
                        if !(var.is_finite() && var > 1e-12_f32) {
                            continue;
                        }
                    }
                    let (bucket, sign) = signed_hash_bucket_sign(seed, row_idx, n_buckets);
                    rows_by_bucket[bucket].push(row_idx);
                    sign_by_row[row_idx] = sign;
                    kept += 1;
                }
                if kept == 0 {
                    return Err(
                        "No SNPs left after signed hashing filters; relax min_maf/max_missing."
                            .to_string(),
                    );
                }

                let mut gram = vec![0.0_f32; n_buckets * n_buckets];
                let mut row_sum = vec![0.0_f64; n_buckets];
                let mut row_sq_sum = vec![0.0_f64; n_buckets];
                let mut zy = vec![0.0_f64; n_buckets];
                let step = sample_block.max(1);
                let notify_step = if progress_every == 0 {
                    step
                } else {
                    progress_every.max(1)
                };
                let mut last_notified = 0usize;

                for st in (0..n_train).step_by(step) {
                    let ed = (st + step).min(n_train);
                    let b = ed - st;
                    if b == 0 {
                        continue;
                    }
                    let lut_blk = &sample_lut[st..ed];
                    let y_blk = &y_vec[st..ed];
                    let mut z_blk = vec![0.0_f32; n_buckets * b];

                    let mut decode_run = || {
                        z_blk
                            .par_chunks_mut(b)
                            .enumerate()
                            .for_each(|(bucket, out_row)| {
                                let rows = &rows_by_bucket[bucket];
                                if rows.is_empty() {
                                    return;
                                }
                                for &row_idx in rows.iter() {
                                    let row = &packed_flat
                                        [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                    let flip = row_flip_vec[row_idx];
                                    let maf = row_maf_vec[row_idx].clamp(0.0_f32, 0.5_f32);
                                    let mean_g = 2.0_f32 * maf;
                                    let denom = if standardize {
                                        (2.0_f32 * maf * (1.0_f32 - maf)).max(1e-12_f32).sqrt()
                                    } else {
                                        1.0_f32
                                    };
                                    let sign = sign_by_row[row_idx];
                                    for (j, &(byte_idx, shift_bits)) in lut_blk.iter().enumerate() {
                                        let bcode = row[byte_idx];
                                        let code = (bcode >> shift_bits) & 0b11;
                                        let mut gv = match code {
                                            0b00 => 0.0_f32,
                                            0b10 => 1.0_f32,
                                            0b11 => 2.0_f32,
                                            _ => mean_g,
                                        };
                                        if flip && code != 0b01 {
                                            gv = 2.0_f32 - gv;
                                        }
                                        let v = if standardize {
                                            (gv - mean_g) / denom
                                        } else {
                                            gv
                                        };
                                        out_row[j] += sign * v;
                                    }
                                }
                            });
                    };
                    if let Some(tp) = &pool {
                        tp.install(&mut decode_run);
                    } else {
                        decode_run();
                    }

                    for bucket in 0..n_buckets {
                        let row = &z_blk[bucket * b..(bucket + 1) * b];
                        let mut sum = 0.0_f64;
                        let mut sq = 0.0_f64;
                        let mut doty = 0.0_f64;
                        for j in 0..b {
                            let v = row[j] as f64;
                            sum += v;
                            sq += v * v;
                            doty += v * y_blk[j];
                        }
                        row_sum[bucket] += sum;
                        row_sq_sum[bucket] += sq;
                        zy[bucket] += doty;
                    }

                    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
                    unsafe {
                        cblas_sgemm_dispatch(
                            CBLAS_COL_MAJOR,
                            CBLAS_TRANS,
                            CBLAS_NO_TRANS,
                            n_buckets as CblasInt,
                            n_buckets as CblasInt,
                            b as CblasInt,
                            1.0,
                            z_blk.as_ptr(),
                            b as CblasInt,
                            z_blk.as_ptr(),
                            b as CblasInt,
                            1.0,
                            gram.as_mut_ptr(),
                            n_buckets as CblasInt,
                        );
                    }
                    #[cfg(not(any(
                        target_os = "macos",
                        target_os = "linux",
                        target_os = "windows"
                    )))]
                    unsafe {
                        sgemm(
                            n_buckets,
                            n_buckets,
                            b,
                            1.0,
                            z_blk.as_ptr(),
                            b as isize,
                            1,
                            z_blk.as_ptr(),
                            b as isize,
                            1,
                            1.0,
                            gram.as_mut_ptr(),
                            n_buckets as isize,
                            1,
                        );
                    }

                    let done = ed;
                    if done >= last_notified.saturating_add(notify_step) || done == n_train {
                        last_notified = done;
                        if let Some(cb) = progress_callback.as_ref() {
                            Python::attach(|py2| -> PyResult<()> {
                                py2.check_signals()?;
                                cb.call1(py2, (done, n_train))?;
                                Ok(())
                            })
                            .map_err(|e| e.to_string())?;
                        } else {
                            Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                        }
                    }
                }

                let mut sum_diag = 0.0_f64;
                for v in row_sq_sum.iter() {
                    sum_diag += *v;
                }
                let mean_diag = sum_diag / (n_train as f64);
                let mut scale = mean_diag.sqrt();
                if !scale.is_finite() || scale <= 0.0_f64 {
                    scale = 1.0_f64;
                }
                let inv = 1.0_f64 / scale;
                let inv2 = inv * inv;
                for v in row_sum.iter_mut() {
                    *v *= inv;
                }
                for v in row_sq_sum.iter_mut() {
                    *v *= inv2;
                }
                for v in zy.iter_mut() {
                    *v *= inv;
                }
                for v in gram.iter_mut() {
                    *v *= inv2 as f32;
                }

                for i in 0..n_buckets {
                    for j in (i + 1)..n_buckets {
                        let a = 0.5_f32 * (gram[i * n_buckets + j] + gram[j * n_buckets + i]);
                        gram[i * n_buckets + j] = a;
                        gram[j * n_buckets + i] = a;
                    }
                }

                let gram_f64: Vec<f64> = gram.into_iter().map(|v| v as f64).collect();
                Ok((gram_f64, row_sum, row_sq_sum, zy, scale, kept))
            },
        )
        .map_err(map_err_string_to_py)?;

    let gram_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n_buckets, n_buckets), gram_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let row_sum_arr = PyArray1::from_owned_array(py, Array1::from_vec(row_sum_vec)).into_bound();
    let row_sq_sum_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(row_sq_sum_vec)).into_bound();
    let zy_arr = PyArray1::from_owned_array(py, Array1::from_vec(zy_vec)).into_bound();
    Ok((
        gram_arr,
        row_sum_arr,
        row_sq_sum_arr,
        zy_arr,
        scale,
        kept_snps,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    n_buckets,
    train_sample_indices,
    test_sample_indices=None,
    seed=20260428_u64,
    row_missing=None,
    min_maf=0.0_f32,
    max_missing=1.0_f32,
    standardize=true,
    threads=0,
    bucket_block=64,
    progress_callback=None,
    progress_every=0
))]
pub fn bed_packed_signed_hash_kernels_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    n_buckets: usize,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    test_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    seed: u64,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    min_maf: f32,
    max_missing: f32,
    standardize: bool,
    threads: usize,
    bucket_block: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, usize)> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if n_buckets == 0 {
        return Err(PyRuntimeError::new_err("n_buckets must be > 0"));
    }
    if !(min_maf.is_finite() && min_maf >= 0.0_f32 && min_maf <= 0.5_f32) {
        return Err(PyRuntimeError::new_err(
            "min_maf must be finite and in [0, 0.5]",
        ));
    }
    if !(max_missing.is_finite() && max_missing >= 0.0_f32 && max_missing <= 1.0_f32) {
        return Err(PyRuntimeError::new_err(
            "max_missing must be finite and in [0, 1]",
        ));
    }

    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }
    let row_missing_vec: Option<Vec<f32>> = match row_missing {
        Some(arr) => {
            let v: Vec<f32> = match arr.as_slice() {
                Ok(s) => s.to_vec(),
                Err(_) => arr.as_array().iter().copied().collect(),
            };
            if v.len() != m {
                return Err(PyRuntimeError::new_err(format!(
                    "row_missing length mismatch: got {}, expected {m}",
                    v.len()
                )));
            }
            Some(v)
        }
        None => None,
    };

    let train_idx = parse_index_vec_i64(
        train_sample_indices.as_slice()?,
        n_samples,
        "train_sample_indices",
    )?;
    if train_idx.is_empty() {
        return Err(PyRuntimeError::new_err(
            "train_sample_indices must not be empty",
        ));
    }
    let test_idx: Vec<usize> = if let Some(sidx) = test_sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "test_sample_indices")?
    } else {
        Vec::new()
    };
    let n_train = train_idx.len();
    let n_test = test_idx.len();
    let train_lut: Vec<(usize, u8)> = train_idx
        .iter()
        .map(|&s| (s >> 2, ((s & 3) * 2) as u8))
        .collect();
    let test_lut: Vec<(usize, u8)> = test_idx
        .iter()
        .map(|&s| (s >> 2, ((s & 3) * 2) as u8))
        .collect();

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;

    let (train_grm_f64, test_train_f64, kept_snps) = py
        .detach(|| -> Result<(Vec<f64>, Vec<f64>, usize), String> {
            let mut rows_by_bucket: Vec<Vec<usize>> = (0..n_buckets).map(|_| Vec::new()).collect();
            let mut sign_by_row = vec![0.0_f32; m];
            let mut kept: usize = 0;
            for row_idx in 0..m {
                let maf = row_maf_vec[row_idx];
                if !maf.is_finite() {
                    continue;
                }
                if maf < min_maf || maf > 0.5_f32 {
                    continue;
                }
                if let Some(miss) = row_missing_vec.as_ref().map(|x| x[row_idx]) {
                    if !miss.is_finite() || miss > max_missing {
                        continue;
                    }
                }
                if standardize {
                    let var = 2.0_f32 * maf * (1.0_f32 - maf);
                    if !(var.is_finite() && var > 1e-12_f32) {
                        continue;
                    }
                }
                let (bucket, sign) = signed_hash_bucket_sign(seed, row_idx, n_buckets);
                rows_by_bucket[bucket].push(row_idx);
                sign_by_row[row_idx] = sign;
                kept += 1;
            }
            if kept == 0 {
                return Err(
                    "No SNPs left after signed hashing filters; relax min_maf/max_missing."
                        .to_string(),
                );
            }

            let train_grm_len = n_train
                .checked_mul(n_train)
                .ok_or_else(|| "train_grm allocation overflow".to_string())?;
            let cross_len = n_test
                .checked_mul(n_train)
                .ok_or_else(|| "test_train_grm allocation overflow".to_string())?;
            let mut train_grm = vec![0.0_f32; train_grm_len];
            let mut test_train = vec![0.0_f32; cross_len];
            let mut var_sum_total = 0.0_f64;
            let block_sz = bucket_block.max(1);
            let notify_step = if progress_every == 0 {
                (kept / 200).max(1)
            } else {
                progress_every.max(1)
            };
            let mut done_rows = 0usize;
            let mut last_notified = 0usize;

            for b0 in (0..n_buckets).step_by(block_sz) {
                let b1 = (b0 + block_sz).min(n_buckets);
                let cur = b1 - b0;
                if cur == 0 {
                    continue;
                }
                let train_blk_len = cur
                    .checked_mul(n_train)
                    .ok_or_else(|| "train block allocation overflow".to_string())?;
                let mut train_blk = vec![0.0_f32; train_blk_len];
                let mut var_blk = vec![0.0_f32; cur];
                let rows_blk = &rows_by_bucket[b0..b1];
                let chunk_done: usize = rows_blk.iter().map(|rows| rows.len()).sum();

                if n_test > 0 {
                    let test_blk_len = cur
                        .checked_mul(n_test)
                        .ok_or_else(|| "test block allocation overflow".to_string())?;
                    let mut test_blk = vec![0.0_f32; test_blk_len];

                    let mut decode_run = || {
                        train_blk
                            .par_chunks_mut(n_train)
                            .zip(test_blk.par_chunks_mut(n_test))
                            .zip(var_blk.par_iter_mut())
                            .enumerate()
                            .for_each(|(off, ((train_row, test_row), var_dst))| {
                                let rows = &rows_blk[off];
                                for &row_idx in rows.iter() {
                                    let row = &packed_flat
                                        [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                    let flip = row_flip_vec[row_idx];
                                    let maf = row_maf_vec[row_idx].clamp(0.0_f32, 0.5_f32);
                                    let mean_g = 2.0_f32 * maf;
                                    let denom = if standardize {
                                        (2.0_f32 * maf * (1.0_f32 - maf)).max(1e-12_f32).sqrt()
                                    } else {
                                        1.0_f32
                                    };
                                    let sign = sign_by_row[row_idx];
                                    for (j, &(byte_idx, shift_bits)) in train_lut.iter().enumerate()
                                    {
                                        let b = row[byte_idx];
                                        let code = (b >> shift_bits) & 0b11;
                                        let mut gv = match code {
                                            0b00 => 0.0_f32,
                                            0b10 => 1.0_f32,
                                            0b11 => 2.0_f32,
                                            _ => mean_g,
                                        };
                                        if flip && code != 0b01 {
                                            gv = 2.0_f32 - gv;
                                        }
                                        let v = if standardize {
                                            (gv - mean_g) / denom
                                        } else {
                                            gv
                                        };
                                        train_row[j] += sign * v;
                                    }
                                    for (j, &(byte_idx, shift_bits)) in test_lut.iter().enumerate()
                                    {
                                        let b = row[byte_idx];
                                        let code = (b >> shift_bits) & 0b11;
                                        let mut gv = match code {
                                            0b00 => 0.0_f32,
                                            0b10 => 1.0_f32,
                                            0b11 => 2.0_f32,
                                            _ => mean_g,
                                        };
                                        if flip && code != 0b01 {
                                            gv = 2.0_f32 - gv;
                                        }
                                        let v = if standardize {
                                            (gv - mean_g) / denom
                                        } else {
                                            gv
                                        };
                                        test_row[j] += sign * v;
                                    }
                                }
                                let mean =
                                    train_row.iter().copied().sum::<f32>() / (n_train as f32);
                                let mut var_acc = 0.0_f32;
                                for v in train_row.iter_mut() {
                                    let c = *v - mean;
                                    *v = c;
                                    var_acc += c * c;
                                }
                                *var_dst = var_acc / (n_train as f32);
                                for v in test_row.iter_mut() {
                                    *v -= mean;
                                }
                            });
                    };
                    if let Some(tp) = &pool {
                        tp.install(&mut decode_run);
                    } else {
                        decode_run();
                    }

                    var_sum_total += var_blk.iter().map(|&v| v as f64).sum::<f64>();
                    unsafe {
                        sgemm(
                            n_train,
                            cur,
                            n_train,
                            1.0_f32,
                            train_blk.as_ptr(),
                            1,
                            n_train as isize,
                            train_blk.as_ptr(),
                            n_train as isize,
                            1,
                            1.0_f32,
                            train_grm.as_mut_ptr(),
                            n_train as isize,
                            1,
                        );
                        sgemm(
                            n_test,
                            cur,
                            n_train,
                            1.0_f32,
                            test_blk.as_ptr(),
                            1,
                            n_test as isize,
                            train_blk.as_ptr(),
                            n_train as isize,
                            1,
                            1.0_f32,
                            test_train.as_mut_ptr(),
                            n_train as isize,
                            1,
                        );
                    }
                } else {
                    let mut decode_run = || {
                        train_blk
                            .par_chunks_mut(n_train)
                            .zip(var_blk.par_iter_mut())
                            .enumerate()
                            .for_each(|(off, (train_row, var_dst))| {
                                let rows = &rows_blk[off];
                                for &row_idx in rows.iter() {
                                    let row = &packed_flat
                                        [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                    let flip = row_flip_vec[row_idx];
                                    let maf = row_maf_vec[row_idx].clamp(0.0_f32, 0.5_f32);
                                    let mean_g = 2.0_f32 * maf;
                                    let denom = if standardize {
                                        (2.0_f32 * maf * (1.0_f32 - maf)).max(1e-12_f32).sqrt()
                                    } else {
                                        1.0_f32
                                    };
                                    let sign = sign_by_row[row_idx];
                                    for (j, &(byte_idx, shift_bits)) in train_lut.iter().enumerate()
                                    {
                                        let b = row[byte_idx];
                                        let code = (b >> shift_bits) & 0b11;
                                        let mut gv = match code {
                                            0b00 => 0.0_f32,
                                            0b10 => 1.0_f32,
                                            0b11 => 2.0_f32,
                                            _ => mean_g,
                                        };
                                        if flip && code != 0b01 {
                                            gv = 2.0_f32 - gv;
                                        }
                                        let v = if standardize {
                                            (gv - mean_g) / denom
                                        } else {
                                            gv
                                        };
                                        train_row[j] += sign * v;
                                    }
                                }
                                let mean =
                                    train_row.iter().copied().sum::<f32>() / (n_train as f32);
                                let mut var_acc = 0.0_f32;
                                for v in train_row.iter_mut() {
                                    let c = *v - mean;
                                    *v = c;
                                    var_acc += c * c;
                                }
                                *var_dst = var_acc / (n_train as f32);
                            });
                    };
                    if let Some(tp) = &pool {
                        tp.install(&mut decode_run);
                    } else {
                        decode_run();
                    }

                    var_sum_total += var_blk.iter().map(|&v| v as f64).sum::<f64>();
                    unsafe {
                        sgemm(
                            n_train,
                            cur,
                            n_train,
                            1.0_f32,
                            train_blk.as_ptr(),
                            1,
                            n_train as isize,
                            train_blk.as_ptr(),
                            n_train as isize,
                            1,
                            1.0_f32,
                            train_grm.as_mut_ptr(),
                            n_train as isize,
                            1,
                        );
                    }
                }

                done_rows += chunk_done;
                if done_rows >= last_notified.saturating_add(notify_step) || done_rows == kept {
                    last_notified = done_rows;
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done_rows, kept))?;
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                    } else {
                        Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                    }
                }
            }

            if !(var_sum_total.is_finite() && var_sum_total > 0.0_f64) {
                return Err("Invalid hashed GBLUP denominator (sum(var)<=0).".to_string());
            }
            let inv = (1.0_f64 / var_sum_total) as f32;
            train_grm.iter_mut().for_each(|v| *v *= inv);
            test_train.iter_mut().for_each(|v| *v *= inv);

            for i in 0..n_train {
                for j in (i + 1)..n_train {
                    let a = 0.5_f32 * (train_grm[i * n_train + j] + train_grm[j * n_train + i]);
                    train_grm[i * n_train + j] = a;
                    train_grm[j * n_train + i] = a;
                }
            }

            let train_grm_f64: Vec<f64> = train_grm.into_iter().map(|v| v as f64).collect();
            let test_train_f64: Vec<f64> = test_train.into_iter().map(|v| v as f64).collect();
            Ok((train_grm_f64, test_train_f64, kept))
        })
        .map_err(map_err_string_to_py)?;

    let train_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n_train, n_train), train_grm_f64)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let cross_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n_test, n_train), test_train_f64)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok((train_arr, cross_arr, kept_snps))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices,
    threads=0
))]
pub fn bed_packed_decode_stats_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    let m = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }
    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;

    let (blk, sum_rows, sq_sum_rows) = py
        .detach(|| -> Result<_, String> {
            let mut blk = vec![0.0_f64; m * n_out];
            let mut sum_rows = vec![0.0_f64; m];
            let mut sq_sum_rows = vec![0.0_f64; m];

            let mut run = || {
                blk.par_chunks_mut(n_out)
                    .zip(sum_rows.par_iter_mut())
                    .zip(sq_sum_rows.par_iter_mut())
                    .enumerate()
                    .for_each(|(row_idx, ((out_row, sum_dst), sq_dst))| {
                        let row =
                            &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                        let flip = row_flip_vec[row_idx];
                        let mean_g = 2.0_f64 * row_maf_vec[row_idx] as f64;
                        let mut sum = 0.0_f64;
                        let mut sq_sum = 0.0_f64;
                        for (j, &sid) in sample_idx.iter().enumerate() {
                            let b = row[sid >> 2];
                            let code = (b >> ((sid & 3) * 2)) & 0b11;
                            let mut gv = match decode_plink_bed_hardcall(code) {
                                Some(v) => v,
                                None => mean_g,
                            };
                            if flip && code != 0b01 {
                                gv = 2.0_f64 - gv;
                            }
                            out_row[j] = gv;
                            sum += gv;
                            sq_sum += gv * gv;
                        }
                        *sum_dst = sum;
                        *sq_dst = sq_sum;
                    });
            };
            if let Some(tp) = &pool {
                tp.install(run);
            } else {
                run();
            }
            Ok((blk, sum_rows, sq_sum_rows))
        })
        .map_err(map_err_string_to_py)?;

    let blk_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((m, n_out), blk)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let sum_arr = PyArray1::from_owned_array(py, Array1::from_vec(sum_rows)).into_bound();
    let sq_arr = PyArray1::from_owned_array(py, Array1::from_vec(sq_sum_rows)).into_bound();
    Ok((blk_arr, sum_arr, sq_arr))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices,
    m_alpha,
    m_mean,
    alpha_sum,
    mean_sq,
    mean_malpha,
    m_var_sum,
    block_rows=4096,
    threads=0
))]
pub fn cross_grm_times_alpha_packed_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    m_alpha: PyReadonlyArray1<'py, f64>,
    m_mean: PyReadonlyArray1<'py, f64>,
    alpha_sum: f64,
    mean_sq: f64,
    mean_malpha: f64,
    m_var_sum: f64,
    block_rows: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if !(m_var_sum.is_finite() && m_var_sum > 0.0) {
        return Err(PyRuntimeError::new_err(
            "m_var_sum must be finite and > 0 for compact cross-GRM prediction",
        ));
    }
    if !(alpha_sum.is_finite() && mean_sq.is_finite() && mean_malpha.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "alpha_sum/mean_sq/mean_malpha must be finite",
        ));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "packed must contain at least one SNP row",
        ));
    }
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }

    let m_alpha_vec: Cow<[f64]> = match m_alpha.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(m_alpha.as_array().iter().copied().collect()),
    };
    let m_mean_vec: Cow<[f64]> = match m_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(m_mean.as_array().iter().copied().collect()),
    };
    if m_alpha_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "m_alpha length mismatch: got {}, expected {m}",
            m_alpha_vec.len()
        )));
    }
    if m_mean_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "m_mean length mismatch: got {}, expected {m}",
            m_mean_vec.len()
        )));
    }

    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let full_sample_fast = is_identity_indices(&sample_idx, n_samples);

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;

    let row_step = block_rows.max(1).min(m.max(1));
    let byte_lut = packed_byte_lut();

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    let m_alpha_f32: Vec<f32> = m_alpha_vec.iter().map(|&v| v as f32).collect();
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    let m_mean_f32: Vec<f32> = m_mean_vec.iter().map(|&v| v as f32).collect();

    let (term1, term2) = py
        .detach(|| -> Result<(Vec<f64>, Vec<f64>), String> {
            let mut term1 = vec![0.0_f64; n_out];
            let mut term2 = vec![0.0_f64; n_out];
            let mut block = vec![0.0_f32; row_step * n_out];
            let mut tick = 0usize;

            #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
            let mut tmp1 = vec![0.0_f32; n_out];
            #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
            let mut tmp2 = vec![0.0_f32; n_out];

            let mut run = || -> Result<(), String> {
                for st in (0..m).step_by(row_step) {
                    let ed = (st + row_step).min(m);
                    let cur_rows = ed - st;
                    let blk_slice = &mut block[..cur_rows * n_out];

                    if full_sample_fast {
                        blk_slice.par_chunks_mut(n_out).enumerate().for_each(
                            |(local_row, out_row)| {
                                let row_idx = st + local_row;
                                let row = &packed_flat
                                    [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                let mean_g = 2.0_f32 * row_maf_vec[row_idx];
                                let value_lut = if row_flip_vec[row_idx] {
                                    [2.0_f32, mean_g, 1.0_f32, 0.0_f32]
                                } else {
                                    [0.0_f32, mean_g, 1.0_f32, 2.0_f32]
                                };
                                decode_row_centered_full_lut(
                                    row,
                                    n_samples,
                                    &byte_lut.code4,
                                    &value_lut,
                                    out_row,
                                );
                            },
                        );
                    } else {
                        blk_slice.par_chunks_mut(n_out).enumerate().for_each(
                            |(local_row, out_row)| {
                                let row_idx = st + local_row;
                                let row = &packed_flat
                                    [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                let flip = row_flip_vec[row_idx];
                                let mean_g = 2.0_f32 * row_maf_vec[row_idx];
                                for (j, &sid) in sample_idx.iter().enumerate() {
                                    let b = row[sid >> 2];
                                    let code = (b >> ((sid & 3) * 2)) & 0b11;
                                    let mut gv = match code {
                                        0b00 => 0.0_f32,
                                        0b10 => 1.0_f32,
                                        0b11 => 2.0_f32,
                                        _ => mean_g,
                                    };
                                    if flip && code != 0b01 {
                                        gv = 2.0_f32 - gv;
                                    }
                                    out_row[j] = gv;
                                }
                            },
                        );
                    }

                    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
                    {
                        let alpha_blk = &m_alpha_f32[st..ed];
                        let mean_blk = &m_mean_f32[st..ed];
                        unsafe {
                            cblas_sgemm_dispatch(
                                CBLAS_COL_MAJOR,
                                CBLAS_NO_TRANS,
                                CBLAS_NO_TRANS,
                                n_out as CblasInt,
                                1 as CblasInt,
                                cur_rows as CblasInt,
                                1.0,
                                blk_slice.as_ptr(),
                                n_out as CblasInt,
                                alpha_blk.as_ptr(),
                                cur_rows as CblasInt,
                                0.0,
                                tmp1.as_mut_ptr(),
                                n_out as CblasInt,
                            );
                            cblas_sgemm_dispatch(
                                CBLAS_COL_MAJOR,
                                CBLAS_NO_TRANS,
                                CBLAS_NO_TRANS,
                                n_out as CblasInt,
                                1 as CblasInt,
                                cur_rows as CblasInt,
                                1.0,
                                blk_slice.as_ptr(),
                                n_out as CblasInt,
                                mean_blk.as_ptr(),
                                cur_rows as CblasInt,
                                0.0,
                                tmp2.as_mut_ptr(),
                                n_out as CblasInt,
                            );
                        }
                        for j in 0..n_out {
                            term1[j] += tmp1[j] as f64;
                            term2[j] += tmp2[j] as f64;
                        }
                    }

                    #[cfg(not(any(
                        target_os = "macos",
                        target_os = "linux",
                        target_os = "windows"
                    )))]
                    {
                        let alpha_blk = &m_alpha_vec[st..ed];
                        let mean_blk = &m_mean_vec[st..ed];
                        for j in 0..n_out {
                            let mut acc1 = 0.0_f64;
                            let mut acc2 = 0.0_f64;
                            for r in 0..cur_rows {
                                let gv = blk_slice[r * n_out + j] as f64;
                                acc1 += gv * alpha_blk[r];
                                acc2 += gv * mean_blk[r];
                            }
                            term1[j] += acc1;
                            term2[j] += acc2;
                        }
                    }

                    tick += cur_rows;
                    if tick >= row_step.saturating_mul(64).max(1) {
                        check_ctrlc()?;
                        tick = 0;
                    }
                }
                Ok(())
            };

            if let Some(tp) = &pool {
                tp.install(run)?;
            } else {
                run()?;
            }
            Ok((term1, term2))
        })
        .map_err(map_err_string_to_py)?;

    let const_term = mean_sq * alpha_sum - mean_malpha;
    let inv_var = 1.0_f64 / m_var_sum;
    let mut out = vec![0.0_f64; n_out];
    for j in 0..n_out {
        out[j] = (term1[j] - term2[j] * alpha_sum + const_term) * inv_var;
    }

    let out_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n_out, 1), out)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok(out_arr)
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices,
    alpha,
    block_rows=4096,
    threads=0
))]
pub fn packed_malpha_f64<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    alpha: PyReadonlyArray1<'py, f64>,
    block_rows: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let m = packed_arr.shape()[0];
    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "packed must contain at least one SNP row",
        ));
    }
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<[bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<[f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m}",
            row_flip_vec.len()
        )));
    }
    if row_maf_vec.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf_vec.len()
        )));
    }

    let sample_idx = parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let alpha_vec: Cow<[f64]> = match alpha.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(alpha.as_array().iter().copied().collect()),
    };
    if alpha_vec.len() != n_out {
        return Err(PyRuntimeError::new_err(format!(
            "alpha length mismatch: got {}, expected {} (len(sample_indices))",
            alpha_vec.len(),
            n_out
        )));
    }
    let full_sample_fast = is_identity_indices(&sample_idx, n_samples);

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;
    let row_step = block_rows.max(1).min(m.max(1));
    let byte_lut = packed_byte_lut();

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    let alpha_f32: Vec<f32> = alpha_vec.iter().map(|&v| v as f32).collect();

    let out = py
        .detach(|| -> Result<Vec<f64>, String> {
            let mut out = vec![0.0_f64; m];
            let mut block = vec![0.0_f32; row_step * n_out];
            let mut tick = 0usize;

            #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
            let mut tmp = vec![0.0_f32; row_step];

            let mut run = || -> Result<(), String> {
                for st in (0..m).step_by(row_step) {
                    let ed = (st + row_step).min(m);
                    let cur_rows = ed - st;
                    let blk_slice = &mut block[..cur_rows * n_out];

                    if full_sample_fast {
                        blk_slice.par_chunks_mut(n_out).enumerate().for_each(
                            |(local_row, out_row)| {
                                let row_idx = st + local_row;
                                let row = &packed_flat
                                    [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                let mean_g = 2.0_f32 * row_maf_vec[row_idx];
                                let value_lut = if row_flip_vec[row_idx] {
                                    [2.0_f32, mean_g, 1.0_f32, 0.0_f32]
                                } else {
                                    [0.0_f32, mean_g, 1.0_f32, 2.0_f32]
                                };
                                decode_row_centered_full_lut(
                                    row,
                                    n_samples,
                                    &byte_lut.code4,
                                    &value_lut,
                                    out_row,
                                );
                            },
                        );
                    } else {
                        blk_slice.par_chunks_mut(n_out).enumerate().for_each(
                            |(local_row, out_row)| {
                                let row_idx = st + local_row;
                                let row = &packed_flat
                                    [row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
                                let flip = row_flip_vec[row_idx];
                                let mean_g = 2.0_f32 * row_maf_vec[row_idx];
                                for (j, &sid) in sample_idx.iter().enumerate() {
                                    let b = row[sid >> 2];
                                    let code = (b >> ((sid & 3) * 2)) & 0b11;
                                    let mut gv = match code {
                                        0b00 => 0.0_f32,
                                        0b10 => 1.0_f32,
                                        0b11 => 2.0_f32,
                                        _ => mean_g,
                                    };
                                    if flip && code != 0b01 {
                                        gv = 2.0_f32 - gv;
                                    }
                                    out_row[j] = gv;
                                }
                            },
                        );
                    }

                    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
                    {
                        let out_blk = &mut out[st..ed];
                        let tmp_blk = &mut tmp[..cur_rows];
                        unsafe {
                            cblas_sgemm_dispatch(
                                CBLAS_COL_MAJOR,
                                CBLAS_TRANS,
                                CBLAS_NO_TRANS,
                                cur_rows as CblasInt,
                                1 as CblasInt,
                                n_out as CblasInt,
                                1.0,
                                blk_slice.as_ptr(),
                                n_out as CblasInt,
                                alpha_f32.as_ptr(),
                                n_out as CblasInt,
                                0.0,
                                tmp_blk.as_mut_ptr(),
                                cur_rows as CblasInt,
                            );
                        }
                        for i in 0..cur_rows {
                            out_blk[i] = tmp_blk[i] as f64;
                        }
                    }

                    #[cfg(not(any(
                        target_os = "macos",
                        target_os = "linux",
                        target_os = "windows"
                    )))]
                    {
                        let out_blk = &mut out[st..ed];
                        for r in 0..cur_rows {
                            let mut acc = 0.0_f64;
                            let row = &blk_slice[r * n_out..(r + 1) * n_out];
                            for j in 0..n_out {
                                acc += row[j] as f64 * alpha_vec[j];
                            }
                            out_blk[r] = acc;
                        }
                    }

                    tick += cur_rows;
                    if tick >= row_step.saturating_mul(64).max(1) {
                        check_ctrlc()?;
                        tick = 0;
                    }
                }
                Ok(())
            };

            if let Some(tp) = &pool {
                tp.install(run)?;
            } else {
                run()?;
            }
            Ok(out)
        })
        .map_err(map_err_string_to_py)?;

    let out_arr = PyArray1::from_owned_array(py, Array1::from_vec(out)).into_bound();
    Ok(out_arr)
}
