use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

use crate::bedmath::{decode_row_centered_full_lut, is_identity_indices, packed_byte_lut};
use crate::blas::{cblas_sgemm_dispatch, CblasInt, CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS};
use crate::packed::bed_packed_row_flip_mask;
use crate::pcg::pcg_solve_f32;
use crate::stats_common::{get_cached_pool, map_err_string_to_py, parse_index_vec_i64};

#[derive(Clone, Copy, Debug)]
pub struct HePcgResult {
    pub sigma_g2: f64,
    pub sigma_e2: f64,
    pub h2: f64,
    pub converged: bool,
    pub iters: usize,
    pub rel_res: f64,
    pub m_effective: usize,
    pub tr_k2: f64,
    pub y_ky: f64,
    pub y_y: f64,
}

#[inline]
fn dot_f32_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn fill_rademacher_f32(out: &mut [f32], seed: u64, probe_idx: usize) {
    let mut state = splitmix64(seed ^ ((probe_idx as u64).wrapping_mul(0x517CC1B727220A95)));
    for v in out.iter_mut() {
        state = splitmix64(state);
        *v = if (state & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
    }
}

#[inline]
fn row_major_block_mul_vec_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            rows as CblasInt,
            1 as CblasInt,
            cols as CblasInt,
            1.0,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            cols as CblasInt,
            0.0,
            out.as_mut_ptr(),
            rows as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for r in 0..rows {
            let row = &block[r * cols..(r + 1) * cols];
            let mut acc = 0.0_f32;
            for c in 0..cols {
                acc += row[c] * vec[c];
            }
            out[r] = acc;
        }
    }
}

#[inline]
fn row_major_block_t_mul_vec_accum_f32(
    block: &[f32],
    rows: usize,
    cols: usize,
    vec: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            cols as CblasInt,
            1 as CblasInt,
            rows as CblasInt,
            1.0,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            rows as CblasInt,
            1.0,
            out.as_mut_ptr(),
            cols as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for r in 0..rows {
            let vr = vec[r];
            let row = &block[r * cols..(r + 1) * cols];
            for c in 0..cols {
                out[c] += row[c] * vr;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_standardized_block_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    row_start: usize,
    out: &mut [f32],
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let n_out = sample_idx.len();
    if n_out == 0 {
        return Ok(());
    }
    if !out.len().is_multiple_of(n_out) {
        return Err("decode_standardized_block_f32: output length mismatch".to_string());
    }
    let cur_rows = out.len() / n_out;
    if cur_rows == 0 {
        return Ok(());
    }

    let decode_one = |local_row: usize, out_row: &mut [f32]| {
        let row_idx = row_start + local_row;
        let row = &packed_flat[row_idx * bytes_per_snp..(row_idx + 1) * bytes_per_snp];
        let mean_g = row_mean[row_idx];
        let inv_sd = row_inv_sd[row_idx];
        if full_sample_fast {
            let value_lut: [f32; 4] = if row_flip[row_idx] {
                [
                    (2.0_f32 - mean_g) * inv_sd,
                    0.0_f32,
                    (1.0_f32 - mean_g) * inv_sd,
                    (0.0_f32 - mean_g) * inv_sd,
                ]
            } else {
                [
                    (0.0_f32 - mean_g) * inv_sd,
                    0.0_f32,
                    (1.0_f32 - mean_g) * inv_sd,
                    (2.0_f32 - mean_g) * inv_sd,
                ]
            };
            decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, out_row);
            return;
        }

        let flip = row_flip[row_idx];
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
            out_row[j] = (gv - mean_g) * inv_sd;
        }
    };

    if let Some(tp) = pool {
        tp.install(|| {
            out.par_chunks_mut(n_out)
                .enumerate()
                .for_each(|(local_row, out_row)| decode_one(local_row, out_row));
        });
    } else {
        for (local_row, out_row) in out.chunks_mut(n_out).enumerate() {
            decode_one(local_row, out_row);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_grm_to_vec_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    block_rows: usize,
    vec_in: &[f32],
    m_scale: f32,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<Vec<f32>, String> {
    let m = row_flip.len();
    let n_out = sample_idx.len();
    if vec_in.len() != n_out {
        return Err(format!(
            "apply_grm_to_vec_f32 length mismatch: len(vec_in)={} != n_samples_subset={n_out}",
            vec_in.len()
        ));
    }
    let mut out = vec![0.0_f32; n_out];
    if m == 0 || n_out == 0 {
        return Ok(out);
    }
    let row_step = block_rows.max(1).min(m.max(1));
    let mut block = vec![0.0_f32; row_step * n_out];
    let mut tmp = vec![0.0_f32; row_step];

    for st in (0..m).step_by(row_step) {
        let ed = (st + row_step).min(m);
        let cur_rows = ed - st;
        let blk_slice = &mut block[..cur_rows * n_out];
        decode_standardized_block_f32(
            packed_flat,
            bytes_per_snp,
            n_samples,
            row_flip,
            row_mean,
            row_inv_sd,
            sample_idx,
            full_sample_fast,
            st,
            blk_slice,
            code4_lut,
            pool,
        )?;
        let tmp_slice = &mut tmp[..cur_rows];
        row_major_block_mul_vec_f32(blk_slice, cur_rows, n_out, vec_in, tmp_slice);
        row_major_block_t_mul_vec_accum_f32(blk_slice, cur_rows, n_out, tmp_slice, &mut out);
    }
    let inv_m = 1.0_f32 / m_scale.max(1.0_f32);
    for v in out.iter_mut() {
        *v *= inv_m;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn he_variance_components_packed(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    y: &[f64],
    trace_samples: usize,
    block_rows: usize,
    std_eps: f64,
    max_iter: usize,
    tol: f64,
    seed: u64,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<HePcgResult, String> {
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    let m = row_flip.len();
    if m == 0 {
        return Err("SNP row count must be > 0".to_string());
    }
    if row_maf.len() != m {
        return Err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        ));
    }
    if packed_flat.len() != m.saturating_mul(bytes_per_snp) {
        return Err(format!(
            "packed payload size mismatch: got {}, expected {}",
            packed_flat.len(),
            m.saturating_mul(bytes_per_snp)
        ));
    }
    if sample_idx.is_empty() {
        return Err("sample_idx must not be empty".to_string());
    }
    if y.len() != sample_idx.len() {
        return Err(format!(
            "y length mismatch: got {}, expected {}",
            y.len(),
            sample_idx.len()
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err("y contains non-finite values".to_string());
    }
    if trace_samples == 0 {
        return Err("trace_samples must be > 0".to_string());
    }
    if !(std_eps.is_finite() && std_eps > 0.0_f64) {
        return Err("std_eps must be finite and > 0".to_string());
    }
    if max_iter == 0 {
        return Err("max_iter must be > 0".to_string());
    }
    if !(tol.is_finite() && tol > 0.0_f64) {
        return Err("tol must be finite and > 0".to_string());
    }

    let std_eps32 = std_eps.max(1e-12_f64) as f32;
    let mut row_mean = vec![0.0_f32; m];
    let mut row_inv_sd = vec![0.0_f32; m];
    let mut m_effective = 0usize;
    for j in 0..m {
        let p = row_maf[j].clamp(0.0_f32, 0.5_f32);
        let mean = 2.0_f32 * p;
        let var = (2.0_f32 * p * (1.0_f32 - p)).max(0.0_f32);
        row_mean[j] = mean;
        if var > std_eps32 {
            row_inv_sd[j] = 1.0_f32 / var.sqrt();
            m_effective += 1;
        } else {
            row_inv_sd[j] = 0.0_f32;
        }
    }
    if m_effective == 0 {
        return Err("No effective SNPs after std_eps filtering".to_string());
    }

    let n = sample_idx.len();
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let m_scale = m_effective as f32;
    let y_f32: Vec<f32> = y.iter().map(|v| *v as f32).collect();
    let code4_lut = &packed_byte_lut().code4;

    let k_y = apply_grm_to_vec_f32(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        &row_mean,
        &row_inv_sd,
        sample_idx,
        full_sample_fast,
        block_rows,
        &y_f32,
        m_scale,
        code4_lut,
        pool,
    )?;
    let y_ky = dot_f32_f64(&y_f32, &k_y);
    let y_y = dot_f32_f64(&y_f32, &y_f32);

    let mut probe = vec![0.0_f32; n];
    let mut tr_k2_acc = 0.0_f64;
    for b in 0..trace_samples {
        fill_rademacher_f32(&mut probe, seed, b);
        let kv = apply_grm_to_vec_f32(
            packed_flat,
            bytes_per_snp,
            n_samples,
            row_flip,
            &row_mean,
            &row_inv_sd,
            sample_idx,
            full_sample_fast,
            block_rows,
            &probe,
            m_scale,
            code4_lut,
            pool,
        )?;
        tr_k2_acc += dot_f32_f64(&kv, &kv);
    }
    let tr_k2 = tr_k2_acc / (trace_samples as f64);
    let tr_k = n as f64;
    if !(tr_k2.is_finite() && tr_k2 > 0.0_f64) {
        return Err(format!(
            "estimated Tr(K^2) is invalid: {tr_k2}. Try increasing trace_samples."
        ));
    }

    // Stochastic trace estimation can slightly undershoot Tr(K) for small probe counts.
    // Add a tiny floor to keep the 2x2 normal matrix SPD for PCG.
    let tr_k2_solve = tr_k2.max(tr_k + tr_k.max(1.0_f64) * 1e-6_f64);
    let a00 = tr_k2_solve as f32;
    let a01 = tr_k as f32;
    let a11 = n as f32;
    let b_vec = [y_ky as f32, y_y as f32];
    let d0_inv = 1.0_f32 / a00.max(1e-12_f32);
    let d1_inv = 1.0_f32 / a11.max(1e-12_f32);

    let pcg_res = pcg_solve_f32(
        &b_vec,
        max_iter,
        tol.max(1e-12_f64),
        1e-20_f64,
        |x| {
            if x.len() != 2 {
                return Err(format!("HE 2x2 apply_a expects len=2, got {}", x.len()));
            }
            Ok(vec![a00 * x[0] + a01 * x[1], a01 * x[0] + a11 * x[1]])
        },
        |r, z| {
            z[0] = d0_inv * r[0];
            z[1] = d1_inv * r[1];
        },
        |_iters_now, _iters_max, _rel_res| Ok(()),
    )?;

    let sigma_g2 = pcg_res.x[0] as f64;
    let sigma_e2 = pcg_res.x[1] as f64;
    let h2 = {
        let denom = sigma_g2 + sigma_e2;
        if denom.is_finite() && denom > 0.0_f64 {
            sigma_g2 / denom
        } else {
            f64::NAN
        }
    };

    Ok(HePcgResult {
        sigma_g2,
        sigma_e2,
        h2,
        converged: pcg_res.converged,
        iters: pcg_res.iters,
        rel_res: pcg_res.rel_res,
        m_effective,
        tr_k2,
        y_ky,
        y_y,
    })
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    train_sample_indices,
    y_train,
    site_keep=None,
    trace_samples=32,
    tol=1e-6_f64,
    max_iter=32,
    block_rows=4096,
    std_eps=1e-12_f64,
    threads=0,
    seed=20260512_u64,
    packed=None,
    packed_n_samples=0,
    maf=None,
    row_flip=None
))]
pub fn he_pcg_bed<'py>(
    py: Python<'py>,
    prefix: String,
    train_sample_indices: PyReadonlyArray1<'py, i64>,
    y_train: PyReadonlyArray1<'py, f64>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    trace_samples: usize,
    tol: f64,
    max_iter: usize,
    block_rows: usize,
    std_eps: f64,
    threads: usize,
    seed: u64,
    packed: Option<PyReadonlyArray2<'py, u8>>,
    packed_n_samples: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<(f64, f64, f64, bool, usize, f64, usize, f64, f64, f64)> {
    if trace_samples == 0 {
        return Err(PyRuntimeError::new_err("trace_samples must be > 0"));
    }
    if max_iter == 0 {
        return Err(PyRuntimeError::new_err("max_iter must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0_f64) {
        return Err(PyRuntimeError::new_err("tol must be finite and > 0"));
    }
    if !(std_eps.is_finite() && std_eps > 0.0_f64) {
        return Err(PyRuntimeError::new_err("std_eps must be finite and > 0"));
    }

    let use_external_packed =
        packed.is_some() || maf.is_some() || row_flip.is_some() || packed_n_samples > 0;
    let mut loaded_packed_arr = None;
    let mut loaded_maf_arr = None;
    let mut external_maf_ro: Option<PyReadonlyArray1<'py, f32>> = None;
    let mut external_row_flip_ro: Option<PyReadonlyArray1<'py, bool>> = None;

    let n_samples: usize;
    let packed_ro: PyReadonlyArray2<'py, u8>;
    if use_external_packed {
        packed_ro = packed.ok_or_else(|| {
            PyRuntimeError::new_err("he_pcg_bed: packed payload path requires `packed` argument.")
        })?;
        external_maf_ro = Some(maf.ok_or_else(|| {
            PyRuntimeError::new_err("he_pcg_bed: packed payload path requires `maf` argument.")
        })?);
        external_row_flip_ro = Some(row_flip.ok_or_else(|| {
            PyRuntimeError::new_err("he_pcg_bed: packed payload path requires `row_flip` argument.")
        })?);
        if packed_n_samples == 0 {
            return Err(PyRuntimeError::new_err(
                "he_pcg_bed: packed payload path requires packed_n_samples > 0.",
            ));
        }
        n_samples = packed_n_samples;
    } else {
        let (packed_arr, _miss_arr, maf_arr, _std_arr, n_samples_loaded) =
            crate::gfreader::load_bed_2bit_packed(py, prefix)?;
        if n_samples_loaded == 0 {
            return Err(PyRuntimeError::new_err("No samples found in BED input."));
        }
        n_samples = n_samples_loaded;
        loaded_packed_arr = Some(packed_arr);
        loaded_maf_arr = Some(maf_arr);
        packed_ro = loaded_packed_arr
            .as_ref()
            .expect("packed array must exist")
            .readonly();
    }

    let packed_view = packed_ro.as_array();
    if packed_view.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed BED payload must be 2D (m, bytes_per_snp).",
        ));
    }
    let m_total = packed_view.shape()[0];
    if m_total == 0 {
        return Err(PyRuntimeError::new_err("No SNP rows found in BED input."));
    }
    let bytes_per_snp = packed_view.shape()[1];
    let expected_bps = n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let packed_flat: Cow<[u8]> = match packed_ro.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_view.iter().copied().collect()),
    };

    let maf_full: Vec<f32> = if use_external_packed {
        let maf_ro = external_maf_ro
            .as_ref()
            .expect("external maf must exist for packed payload path");
        match maf_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => maf_ro.as_array().iter().copied().collect(),
        }
    } else {
        let maf_ro = loaded_maf_arr
            .as_ref()
            .expect("maf array must exist")
            .readonly();
        match maf_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => maf_ro.as_array().iter().copied().collect(),
        }
    };
    if maf_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "maf length mismatch: got {}, expected {m_total}",
            maf_full.len()
        )));
    }

    let row_flip_full: Vec<bool> = if use_external_packed {
        let row_flip_ro = external_row_flip_ro
            .as_ref()
            .expect("external row_flip must exist for packed payload path");
        match row_flip_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_flip_ro.as_array().iter().copied().collect(),
        }
    } else {
        let row_flip_full_arr = bed_packed_row_flip_mask(
            py,
            loaded_packed_arr
                .as_ref()
                .expect("packed array must exist")
                .readonly(),
            n_samples,
        )?;
        let row_flip_ro = row_flip_full_arr.readonly();
        match row_flip_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_flip_ro.as_array().iter().copied().collect(),
        }
    };
    if row_flip_full.len() != m_total {
        return Err(PyRuntimeError::new_err(format!(
            "row_flip length mismatch: got {}, expected {m_total}",
            row_flip_full.len()
        )));
    }

    let mut eff_m = m_total;
    let mut packed_keep: Cow<[u8]> = Cow::Borrowed(packed_flat.as_ref());
    let mut maf_keep: Cow<[f32]> = Cow::Borrowed(maf_full.as_slice());
    let mut row_flip_keep: Cow<[bool]> = Cow::Borrowed(row_flip_full.as_slice());
    if let Some(mask) = site_keep {
        let mask_vec: Vec<bool> = match mask.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mask.as_array().iter().copied().collect(),
        };
        if mask_vec.len() != m_total {
            return Err(PyRuntimeError::new_err(format!(
                "site_keep length mismatch: got {}, expected {m_total}",
                mask_vec.len()
            )));
        }
        let keep_idx: Vec<usize> = mask_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &k)| if k { Some(i) } else { None })
            .collect();
        if keep_idx.is_empty() {
            return Err(PyRuntimeError::new_err(
                "No SNPs remained after applying site_keep mask.",
            ));
        }
        let keep_is_identity = (keep_idx.len() == m_total)
            && keep_idx
                .iter()
                .enumerate()
                .all(|(dst_row, &src_row)| dst_row == src_row);
        if !keep_is_identity {
            eff_m = keep_idx.len();
            let mut packed_subset = vec![0_u8; eff_m * bytes_per_snp];
            let mut maf_subset = vec![0.0_f32; eff_m];
            let mut row_flip_subset = vec![false; eff_m];
            for (dst_row, &src_row) in keep_idx.iter().enumerate() {
                let src_off = src_row * bytes_per_snp;
                let dst_off = dst_row * bytes_per_snp;
                packed_subset[dst_off..dst_off + bytes_per_snp]
                    .copy_from_slice(&packed_flat[src_off..src_off + bytes_per_snp]);
                maf_subset[dst_row] = maf_full[src_row].clamp(0.0_f32, 0.5_f32);
                row_flip_subset[dst_row] = row_flip_full[src_row];
            }
            packed_keep = Cow::Owned(packed_subset);
            maf_keep = Cow::Owned(maf_subset);
            row_flip_keep = Cow::Owned(row_flip_subset);
        }
    }

    let train_idx = parse_index_vec_i64(
        train_sample_indices.as_slice()?,
        n_samples,
        "train_sample_indices",
    )?;
    if train_idx.is_empty() {
        return Err(PyRuntimeError::new_err(
            "train_sample_indices must not be empty.",
        ));
    }
    let y_vec_f64: Vec<f64> = match y_train.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => y_train.as_array().iter().copied().collect(),
    };
    if y_vec_f64.len() != train_idx.len() {
        return Err(PyRuntimeError::new_err(format!(
            "y_train length mismatch: got {}, expected {}",
            y_vec_f64.len(),
            train_idx.len()
        )));
    }
    if y_vec_f64.iter().any(|v| !v.is_finite()) {
        return Err(PyRuntimeError::new_err(
            "y_train contains non-finite values.",
        ));
    }

    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let he_res = py
        .detach(move || {
            he_variance_components_packed(
                packed_keep.as_ref(),
                bytes_per_snp,
                n_samples,
                row_flip_keep.as_ref(),
                maf_keep.as_ref(),
                &train_idx,
                &y_vec_f64,
                trace_samples,
                block_rows,
                std_eps,
                max_iter,
                tol,
                seed,
                pool_ref,
            )
        })
        .map_err(map_err_string_to_py)?;

    Ok((
        he_res.sigma_g2,
        he_res.sigma_e2,
        he_res.h2,
        he_res.converged,
        he_res.iters,
        he_res.rel_res,
        he_res.m_effective.min(eff_m),
        he_res.tr_k2,
        he_res.y_ky,
        he_res.y_y,
    ))
}
