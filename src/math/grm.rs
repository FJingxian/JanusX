use super::*;
use crate::gs_native::{grm_rankk_update, grm_rankk_update_f64};

#[allow(clippy::too_many_arguments)]
pub(crate) fn grm_packed_f32_with_stats_impl<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f64>>, f64)> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if method != 1 && method != 2 {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method={method}; expected 1 (centered) or 2 (standardized)"
        )));
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

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let full_sample_fast = is_identity_indices(&sample_idx, n_samples);

    let varsum_full = if method == 1 && full_sample_fast {
        let mut acc = 0.0_f64;
        for &maf in row_maf_vec.iter() {
            let p = maf as f64;
            let v = 2.0_f64 * p * (1.0_f64 - p);
            if v.is_finite() && v > 0.0 {
                acc += v;
            }
        }
        if !(acc.is_finite() && acc > 0.0) {
            return Err(PyRuntimeError::new_err(
                "invalid centered GRM denominator: sum(2p(1-p)) <= 0",
            ));
        }
        acc
    } else {
        0.0_f64
    };

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;
    let row_step = adaptive_grm_block_rows(
        block_cols.max(1),
        m.max(1),
        n,
        if full_sample_fast { 0 } else { n_samples },
        threads,
    );
    let notify_step = if progress_every == 0 {
        row_step
    } else {
        progress_every.max(1)
    };
    let cblas_copy_rhs = std::env::var("JX_GRM_PACKED_CBLAS_COPY_RHS")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("1") || s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let cblas_beta_zero_accum = std::env::var("JX_GRM_PACKED_CBLAS_BETA0")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(true);

    let (grm_vec, row_sum_vec, varsum_ret) = py
        .detach(move || -> Result<(Vec<f32>, Vec<f64>, f64), String> {
            let mut grm = vec![0.0_f32; n * n];
            let mut block = vec![0.0_f32; row_step * n];
            let mut varsum_acc = varsum_full;
            let eps = 1e-12_f32;
            let byte_lut = packed_byte_lut();
            let mut row_sum_all = vec![0.0_f64; m];

            let mut last_notified = 0usize;
            for row_start in (0..m).step_by(row_step) {
                let row_end = (row_start + row_step).min(m);
                let cur_rows = row_end - row_start;
                let cur_block = &mut block[..cur_rows * n];
                let mut block_varsum = vec![0.0_f64; cur_rows];
                let mut block_rowsum = vec![0.0_f64; cur_rows];

                let mut decode_run = || {
                    if full_sample_fast {
                        cur_block
                            .par_chunks_mut(n)
                            .zip(block_varsum.par_iter_mut())
                            .zip(block_rowsum.par_iter_mut())
                            .enumerate()
                            .for_each(|(off, ((out_row, row_varsum_dst), row_sum_dst))| {
                                let idx = row_start + off;
                                let row =
                                    &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                                let p = row_maf_vec[idx].clamp(0.0_f32, 0.5_f32);
                                let mean_g = 2.0_f32 * p;
                                let var = 2.0_f32 * p * (1.0_f32 - p);
                                let std_scale = if method == 2 {
                                    if var > eps {
                                        1.0_f32 / var.sqrt()
                                    } else {
                                        0.0_f32
                                    }
                                } else {
                                    1.0_f32
                                };
                                let value_lut: [f32; 4] = if row_flip_vec[idx] {
                                    [
                                        (2.0_f32 - mean_g) * std_scale,
                                        0.0_f32,
                                        (1.0_f32 - mean_g) * std_scale,
                                        (0.0_f32 - mean_g) * std_scale,
                                    ]
                                } else {
                                    [
                                        (0.0_f32 - mean_g) * std_scale,
                                        0.0_f32,
                                        (1.0_f32 - mean_g) * std_scale,
                                        (2.0_f32 - mean_g) * std_scale,
                                    ]
                                };
                                decode_row_centered_full_lut(
                                    row,
                                    n_samples,
                                    &byte_lut.code4,
                                    &value_lut,
                                    out_row,
                                );
                                if method == 1 {
                                    *row_varsum_dst = var as f64;
                                }
                                *row_sum_dst = (mean_g as f64) * (n as f64);
                            });
                    } else {
                        cur_block
                            .par_chunks_mut(n)
                            .zip(block_varsum.par_iter_mut())
                            .zip(block_rowsum.par_iter_mut())
                            .enumerate()
                            .for_each_init(
                                || vec![0.0_f32; n_samples],
                                |full_row, (off, ((out_row, row_varsum_dst), row_sum_dst))| {
                                    let idx = row_start + off;
                                    let row = &packed_flat
                                        [idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                                    let p = row_maf_vec[idx].clamp(0.0_f32, 0.5_f32);
                                    let default_mean_g = 2.0_f32 * p;
                                    let (var_centered, row_sum) =
                                        decode_subset_row_from_full_scratch(
                                            row,
                                            n_samples,
                                            &sample_idx,
                                            row_flip_vec[idx],
                                            default_mean_g,
                                            method,
                                            eps,
                                            &byte_lut.code4,
                                            full_row.as_mut_slice(),
                                            out_row,
                                        );
                                    if method == 1 {
                                        *row_varsum_dst = var_centered;
                                    }
                                    *row_sum_dst = row_sum;
                                },
                            );
                    }
                };

                if let Some(tp) = &pool {
                    tp.install(&mut decode_run);
                } else {
                    decode_run();
                }

                grm_rankk_update(
                    &mut grm,
                    cur_block,
                    cur_rows,
                    n,
                    cblas_copy_rhs,
                    cblas_beta_zero_accum,
                )?;
                if method == 1 && !full_sample_fast {
                    varsum_acc += block_varsum.iter().sum::<f64>();
                }
                row_sum_all[row_start..row_end].copy_from_slice(&block_rowsum);

                let done = row_end;
                if (done >= last_notified.saturating_add(notify_step)) || (done == m) {
                    last_notified = done;
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done, m))?;
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                    } else {
                        Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                    }
                }
            }

            let scale = if method == 1 {
                if !(varsum_acc.is_finite() && varsum_acc > 0.0) {
                    return Err("invalid centered GRM denominator in subset path".to_string());
                }
                varsum_acc as f32
            } else {
                m as f32
            };
            if !(scale.is_finite() && scale > 0.0) {
                return Err("invalid GRM scale factor".to_string());
            }
            let inv_scale = 1.0_f32 / scale;
            for i in 0..n {
                let ii = i * n + i;
                grm[ii] *= inv_scale;
                for j in 0..i {
                    let idx_ij = i * n + j;
                    let idx_ji = j * n + i;
                    let v = 0.5_f32 * (grm[idx_ij] + grm[idx_ji]) * inv_scale;
                    grm[idx_ij] = v;
                    grm[idx_ji] = v;
                }
            }
            let varsum_ret = if method == 1 { varsum_acc } else { m as f64 };
            Ok((grm, row_sum_all, varsum_ret))
        })
        .map_err(map_err_string_to_py)?;

    let grm_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n, n), grm_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let row_sum_arr = PyArray1::from_owned_array(py, Array1::from_vec(row_sum_vec)).into_bound();
    Ok((grm_arr, row_sum_arr, varsum_ret))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn grm_packed_f64_with_stats_impl<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("n_samples must be > 0"));
    }
    if method != 1 && method != 2 {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method={method}; expected 1 (centered) or 2 (standardized)"
        )));
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

    let sample_idx: Vec<usize> = if let Some(sidx) = sample_indices {
        parse_index_vec_i64(sidx.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    let n = sample_idx.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("sample_indices must not be empty"));
    }
    let full_sample_fast = is_identity_indices(&sample_idx, n_samples);

    let varsum_full = if method == 1 && full_sample_fast {
        let mut acc = 0.0_f64;
        for &maf in row_maf_vec.iter() {
            let p = maf as f64;
            let v = 2.0_f64 * p * (1.0_f64 - p);
            if v.is_finite() && v > 0.0 {
                acc += v;
            }
        }
        if !(acc.is_finite() && acc > 0.0) {
            return Err(PyRuntimeError::new_err(
                "invalid centered GRM denominator: sum(2p(1-p)) <= 0",
            ));
        }
        acc
    } else {
        0.0_f64
    };

    let packed_flat: Cow<[u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool = get_cached_pool(threads)?;
    let row_step = adaptive_grm_block_rows(
        block_cols.max(1),
        m.max(1),
        n,
        if full_sample_fast { 0 } else { n_samples },
        threads,
    );
    let notify_step = if progress_every == 0 {
        row_step
    } else {
        progress_every.max(1)
    };
    let cblas_copy_rhs = std::env::var("JX_GRM_PACKED_CBLAS_COPY_RHS")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("1") || s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let cblas_beta_zero_accum = std::env::var("JX_GRM_PACKED_CBLAS_BETA0")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(true);

    let (grm_vec, row_sum_vec, varsum_ret) = py
        .detach(move || -> Result<(Vec<f64>, Vec<f64>, f64), String> {
            let mut grm = vec![0.0_f64; n * n];
            let mut block = vec![0.0_f64; row_step * n];
            let mut varsum_acc = varsum_full;
            let eps = 1e-12_f64;
            let byte_lut = packed_byte_lut();
            let mut row_sum_all = vec![0.0_f64; m];

            let mut last_notified = 0usize;
            for row_start in (0..m).step_by(row_step) {
                let row_end = (row_start + row_step).min(m);
                let cur_rows = row_end - row_start;
                let cur_block = &mut block[..cur_rows * n];
                let mut block_varsum = vec![0.0_f64; cur_rows];
                let mut block_rowsum = vec![0.0_f64; cur_rows];

                let mut decode_run = || {
                    if full_sample_fast {
                        cur_block
                            .par_chunks_mut(n)
                            .zip(block_varsum.par_iter_mut())
                            .zip(block_rowsum.par_iter_mut())
                            .enumerate()
                            .for_each(|(off, ((out_row, row_varsum_dst), row_sum_dst))| {
                                let idx = row_start + off;
                                let row =
                                    &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                                let p = row_maf_vec[idx].clamp(0.0_f32, 0.5_f32) as f64;
                                let mean_g = 2.0_f64 * p;
                                let var = 2.0_f64 * p * (1.0_f64 - p);
                                let std_scale = if method == 2 {
                                    if var > eps {
                                        1.0_f64 / var.sqrt()
                                    } else {
                                        0.0_f64
                                    }
                                } else {
                                    1.0_f64
                                };
                                let value_lut: [f64; 4] = if row_flip_vec[idx] {
                                    [
                                        (2.0_f64 - mean_g) * std_scale,
                                        0.0_f64,
                                        (1.0_f64 - mean_g) * std_scale,
                                        (0.0_f64 - mean_g) * std_scale,
                                    ]
                                } else {
                                    [
                                        (0.0_f64 - mean_g) * std_scale,
                                        0.0_f64,
                                        (1.0_f64 - mean_g) * std_scale,
                                        (2.0_f64 - mean_g) * std_scale,
                                    ]
                                };
                                decode_row_centered_full_lut_f64(
                                    row,
                                    n_samples,
                                    &byte_lut.code4,
                                    &value_lut,
                                    out_row,
                                );
                                if method == 1 {
                                    *row_varsum_dst = var;
                                }
                                *row_sum_dst = mean_g * (n as f64);
                            });
                    } else {
                        cur_block
                            .par_chunks_mut(n)
                            .zip(block_varsum.par_iter_mut())
                            .zip(block_rowsum.par_iter_mut())
                            .enumerate()
                            .for_each_init(
                                || vec![0.0_f64; n_samples],
                                |full_row, (off, ((out_row, row_varsum_dst), row_sum_dst))| {
                                    let idx = row_start + off;
                                    let row = &packed_flat
                                        [idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                                    let p = row_maf_vec[idx].clamp(0.0_f32, 0.5_f32) as f64;
                                    let default_mean_g = 2.0_f64 * p;
                                    let (var_centered, row_sum) =
                                        decode_subset_row_from_full_scratch_f64(
                                            row,
                                            n_samples,
                                            &sample_idx,
                                            row_flip_vec[idx],
                                            default_mean_g,
                                            method,
                                            eps,
                                            &byte_lut.code4,
                                            full_row.as_mut_slice(),
                                            out_row,
                                        );
                                    if method == 1 {
                                        *row_varsum_dst = var_centered;
                                    }
                                    *row_sum_dst = row_sum;
                                },
                            );
                    }
                };

                if let Some(tp) = &pool {
                    tp.install(&mut decode_run);
                } else {
                    decode_run();
                }

                grm_rankk_update_f64(
                    &mut grm,
                    cur_block,
                    cur_rows,
                    n,
                    cblas_copy_rhs,
                    cblas_beta_zero_accum,
                )?;
                if method == 1 && !full_sample_fast {
                    varsum_acc += block_varsum.iter().sum::<f64>();
                }
                row_sum_all[row_start..row_end].copy_from_slice(&block_rowsum);

                let done = row_end;
                if (done >= last_notified.saturating_add(notify_step)) || (done == m) {
                    last_notified = done;
                    if let Some(cb) = progress_callback.as_ref() {
                        Python::attach(|py2| -> PyResult<()> {
                            py2.check_signals()?;
                            cb.call1(py2, (done, m))?;
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                    } else {
                        Python::attach(|py2| py2.check_signals()).map_err(|e| e.to_string())?;
                    }
                }
            }

            let scale = if method == 1 {
                if !(varsum_acc.is_finite() && varsum_acc > 0.0) {
                    return Err("invalid centered GRM denominator in subset path".to_string());
                }
                varsum_acc
            } else {
                m as f64
            };
            if !(scale.is_finite() && scale > 0.0) {
                return Err("invalid GRM scale factor".to_string());
            }
            let inv_scale = 1.0_f64 / scale;
            for i in 0..n {
                let ii = i * n + i;
                grm[ii] *= inv_scale;
                for j in 0..i {
                    let idx_ij = i * n + j;
                    let idx_ji = j * n + i;
                    let v = 0.5_f64 * (grm[idx_ij] + grm[idx_ji]) * inv_scale;
                    grm[idx_ij] = v;
                    grm[idx_ji] = v;
                }
            }
            let varsum_ret = if method == 1 { varsum_acc } else { m as f64 };
            Ok((grm, row_sum_all, varsum_ret))
        })
        .map_err(map_err_string_to_py)?;

    let grm_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n, n), grm_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let row_sum_arr = PyArray1::from_owned_array(py, Array1::from_vec(row_sum_vec)).into_bound();
    Ok((grm_arr, row_sum_arr, varsum_ret))
}
