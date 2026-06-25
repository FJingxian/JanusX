use memmap2::MmapOptions;
use nalgebra::DMatrix;
use numpy::ndarray::{Array1, Array2};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
use crate::blas::{lapack_dsyevd_dispatch, lapack_dsyevr_dispatch, CblasInt};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EighDriver {
    Auto,
    Dsyevd,
    Dsyevr,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EighJobz {
    ValuesOnly,
    ValuesAndVectors,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NpyFloatDtype {
    F32Le,
    F64Le,
}

#[inline]
fn copy_col_major_to_row_major(src_col_major: &[f64], n: usize) -> Vec<f64> {
    let mut out_row_major = vec![0.0_f64; n * n];
    for col in 0..n {
        for row in 0..n {
            out_row_major[row * n + col] = src_col_major[row + col * n];
        }
    }
    out_row_major
}

#[inline]
fn array2_view_to_row_major(a_view: numpy::ndarray::ArrayView2<'_, f64>) -> Vec<f64> {
    let nrows = a_view.shape().first().copied().unwrap_or(0usize);
    let ncols = a_view.shape().get(1).copied().unwrap_or(0usize);
    let mut out = vec![0.0_f64; nrows.saturating_mul(ncols)];
    for row in 0..nrows {
        for col in 0..ncols {
            out[row * ncols + col] = a_view[(row, col)];
        }
    }
    out
}

#[inline]
fn pyreadonly_array2_to_row_major_cow<'py>(
    arr: &'py PyReadonlyArray2<'py, f64>,
) -> Cow<'py, [f64]> {
    if let Ok(slice) = arr.as_slice() {
        return Cow::Borrowed(slice);
    }
    Cow::Owned(array2_view_to_row_major(arr.as_array()))
}

#[inline]
fn symmetrize_row_major_to_col_major_f64(
    a_row_major: &[f64],
    n: usize,
    label: &str,
) -> Result<Vec<f64>, String> {
    validate_eigh_dims(n, a_row_major.len(), label)?;
    let mut out = vec![0.0_f64; n * n];
    for row in 0..n {
        for col in row..n {
            let a_rc = a_row_major[row * n + col];
            let a_cr = a_row_major[col * n + row];
            if !a_rc.is_finite() {
                return Err(format!("{label}: non-finite value at ({row}, {col})"));
            }
            if !a_cr.is_finite() {
                return Err(format!("{label}: non-finite value at ({col}, {row})"));
            }
            let v = if row == col {
                a_rc
            } else {
                0.5_f64 * (a_rc + a_cr)
            };
            out[row + col * n] = v;
            out[col + row * n] = v;
        }
    }
    Ok(out)
}

#[inline]
fn validate_eigh_dims(n: usize, len: usize, label: &str) -> Result<(), String> {
    let expect = n
        .checked_mul(n)
        .ok_or_else(|| format!("{label}: matrix element count overflow for n={n}"))?;
    if len != expect {
        return Err(format!(
            "{label}: shape mismatch, len={} expected={expect}",
            len
        ));
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        let max_n = CblasInt::MAX as usize;
        if n > max_n {
            return Err(format!(
                "{label}: n={} exceeds LAPACK integer limit {}",
                n, max_n
            ));
        }
    }
    Ok(())
}

#[inline]
fn validate_subset_indices_usize(
    idx_raw: &[usize],
    limit: usize,
    label: &str,
) -> Result<(), String> {
    if idx_raw.is_empty() {
        return Err(format!("{label}: subset indices must not be empty"));
    }
    for (pos, &u) in idx_raw.iter().enumerate() {
        if u >= limit {
            return Err(format!(
                "{label}: subset index at position {} ({}) exceeds matrix size {}",
                pos, u, limit
            ));
        }
    }
    Ok(())
}

#[inline]
fn lapack_f64_workspace_len(query: f64, label: &str) -> Result<(CblasInt, usize), String> {
    if !query.is_finite() {
        return Err(format!(
            "{label}: non-finite LAPACK workspace query result {query}"
        ));
    }
    let need = query.ceil().max(1.0_f64);
    let max_i32 = CblasInt::MAX as f64;
    if need > max_i32 {
        return Err(format!(
            "{label}: workspace query {} exceeds LAPACK integer limit {}",
            need,
            CblasInt::MAX
        ));
    }
    let need_i = need as CblasInt;
    Ok((need_i, need_i as usize))
}

#[inline]
fn lapack_i32_workspace_len(query: CblasInt, label: &str) -> Result<(CblasInt, usize), String> {
    if query < 1 {
        return Err(format!(
            "{label}: invalid integer workspace query result {query}"
        ));
    }
    Ok((query, query as usize))
}

#[inline]
fn symmetrize_row_major_f64(
    a_row_major: &[f64],
    n: usize,
    label: &str,
) -> Result<Vec<f64>, String> {
    validate_eigh_dims(n, a_row_major.len(), label)?;
    let mut out = vec![0.0_f64; n * n];
    for row in 0..n {
        for col in row..n {
            let a_rc = a_row_major[row * n + col];
            let a_cr = a_row_major[col * n + row];
            if !a_rc.is_finite() {
                return Err(format!("{label}: non-finite value at ({row}, {col})"));
            }
            if !a_cr.is_finite() {
                return Err(format!("{label}: non-finite value at ({col}, {row})"));
            }
            let v = if row == col {
                a_rc
            } else {
                0.5_f64 * (a_rc + a_cr)
            };
            out[row * n + col] = v;
            out[col * n + row] = v;
        }
    }
    Ok(out)
}

#[inline]
fn parse_eigh_driver_token(token: &str) -> EighDriver {
    match token.trim().to_ascii_lowercase().as_str() {
        "auto" => EighDriver::Auto,
        "dsyevr" | "syevr" | "evr" => EighDriver::Dsyevr,
        "" | "dsyevd" | "syevd" | "evd" => EighDriver::Dsyevd,
        _ => EighDriver::Dsyevd,
    }
}

#[inline]
fn parse_eigh_jobz_token(token: &str) -> EighJobz {
    match token.trim().to_ascii_uppercase().as_str() {
        "N" | "NOVEC" | "NOVECTORS" | "EIGVALS" => EighJobz::ValuesOnly,
        "V" | "VEC" | "VECTORS" | "EIGENPAIRS" | "" => EighJobz::ValuesAndVectors,
        _ => EighJobz::ValuesAndVectors,
    }
}

#[inline]
fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(false)
}

#[inline]
fn eigh_jobz_label(jobz: EighJobz) -> &'static str {
    match jobz {
        EighJobz::ValuesOnly => "N",
        EighJobz::ValuesAndVectors => "V",
    }
}

#[inline]
fn is_lapack_backend(name: &str) -> bool {
    name.starts_with("lapack_")
}

#[inline]
fn lapack_dsyevd_backend_label() -> &'static str {
    match crate::blas::rust_eigh_lapack_backend_tag() {
        "openblas_dyn" => "lapack_dsyevd_openblas_dyn",
        "openblas" => "lapack_dsyevd_openblas",
        "accelerate" => "lapack_dsyevd_accelerate",
        "blas" => "lapack_dsyevd_blas",
        "rust" => "lapack_dsyevd_rust",
        _ => "lapack_dsyevd",
    }
}

#[inline]
fn lapack_dsyevr_backend_label() -> &'static str {
    match crate::blas::rust_eigh_lapack_backend_tag() {
        "openblas_dyn" => "lapack_dsyevr_openblas_dyn",
        "openblas" => "lapack_dsyevr_openblas",
        "accelerate" => "lapack_dsyevr_accelerate",
        "blas" => "lapack_dsyevr_blas",
        "rust" => "lapack_dsyevr_rust",
        _ => "lapack_dsyevr",
    }
}

#[inline]
fn resolve_eigh_driver_from_env() -> EighDriver {
    let req = std::env::var("JX_EIGH_DRIVER")
        .ok()
        .unwrap_or_else(|| "auto".to_string());
    parse_eigh_driver_token(&req)
}

#[inline]
fn dsyevd_vectors_workspace_overflows_i32(n: usize) -> bool {
    let n128 = n as u128;
    let need = 1_u128 + 6_u128 * n128 + 2_u128 * n128 * n128;
    need > (CblasInt::MAX as u128)
}

#[inline]
fn resolve_eigh_driver_for_problem(requested: EighDriver, n: usize, jobz: EighJobz) -> EighDriver {
    match requested {
        EighDriver::Dsyevd => EighDriver::Dsyevd,
        EighDriver::Dsyevr => EighDriver::Dsyevr,
        EighDriver::Auto => {
            if matches!(jobz, EighJobz::ValuesAndVectors)
                && dsyevd_vectors_workspace_overflows_i32(n)
            {
                EighDriver::Dsyevr
            } else if env_truthy("JX_EIGH_ALLOW_DSYEVR") {
                EighDriver::Dsyevr
            } else {
                EighDriver::Dsyevd
            }
        }
    }
}

#[inline]
fn parse_npy_shape_local(header: &str) -> Result<(usize, usize), String> {
    let shape_key_pos = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| "NPY header missing shape field".to_string())?;
    let after = &header[shape_key_pos..];
    let open = after
        .find('(')
        .ok_or_else(|| "NPY header has malformed shape tuple".to_string())?;
    let close = after[open + 1..]
        .find(')')
        .ok_or_else(|| "NPY header has malformed shape tuple".to_string())?;
    let inside = &after[open + 1..open + 1 + close];

    let dims: Vec<usize> = inside
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format!("invalid NPY shape dimension: {s}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    match dims.as_slice() {
        [rows] => Ok((*rows, 1)),
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(format!("unsupported NPY shape rank: {:?}", dims)),
    }
}

#[inline]
fn parse_npy_descr_local(header: &str) -> Result<NpyFloatDtype, String> {
    let descr_key_pos = header
        .find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or_else(|| "NPY header missing descr field".to_string())?;
    let after = &header[descr_key_pos..];
    let colon = after
        .find(':')
        .ok_or_else(|| "NPY header has malformed descr field".to_string())?;
    let mut tail = after[colon + 1..].trim_start();
    let q = tail
        .chars()
        .next()
        .ok_or_else(|| "NPY header has empty descr field".to_string())?;
    if q != '\'' && q != '"' {
        return Err("NPY header has malformed descr quote".to_string());
    }
    tail = &tail[1..];
    let end = tail
        .find(q)
        .ok_or_else(|| "NPY header has unterminated descr field".to_string())?;
    let descr = tail[..end].trim().to_ascii_lowercase();
    match descr.as_str() {
        "<f4" | "|f4" | "=f4" => Ok(NpyFloatDtype::F32Le),
        "<f8" | "|f8" | "=f8" => Ok(NpyFloatDtype::F64Le),
        d if d.starts_with(">f4") || d.starts_with(">f8") => {
            Err("big-endian NPY float dtype is not supported".to_string())
        }
        _ => Err(format!("NPY dtype is not supported for eigh: {descr}")),
    }
}

#[inline]
fn parse_npy_header_for_float_matrix(
    bytes: &[u8],
) -> Result<(usize, usize, usize, NpyFloatDtype), String> {
    if bytes.len() < 10 {
        return Err("NPY file too small".to_string());
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("invalid NPY magic".to_string());
    }

    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, header_start) = match major {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10usize)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("NPY file too small for v2/v3 header".to_string());
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12usize)
        }
        _ => return Err(format!("unsupported NPY version: {major}.{minor}")),
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| "NPY header overflow".to_string())?;
    if header_end > bytes.len() {
        return Err("NPY header exceeds file size".to_string());
    }
    let header =
        std::str::from_utf8(&bytes[header_start..header_end]).map_err(|e| e.to_string())?;
    if header.contains("fortran_order': True") || header.contains("fortran_order\": true") {
        return Err("fortran_order=True NPY is not supported".to_string());
    }
    let (rows, cols) = parse_npy_shape_local(header)?;
    let dtype = parse_npy_descr_local(header)?;
    Ok((rows, cols, header_end, dtype))
}

#[inline]
fn apply_diag_shift_row_major(a_row_major: &mut [f64], n: usize, diag_shift: f64) {
    if diag_shift == 0.0 || n == 0 {
        return;
    }
    for i in 0..n {
        a_row_major[i * n + i] += diag_shift;
    }
}

fn load_square_matrix_from_npy_f64(
    path: &str,
    diag_shift: f64,
) -> Result<(Vec<f64>, usize), String> {
    let file = File::open(path).map_err(|e| format!("open NPY failed: {e}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| e.to_string())? };
    let (rows, cols, data_offset, dtype) = parse_npy_header_for_float_matrix(&mmap[..])?;
    if rows == 0 || cols == 0 || rows != cols {
        return Err(format!(
            "GRM/kinship matrix must be non-empty square, got ({rows}, {cols})"
        ));
    }
    let n = rows;
    let n_elem = n
        .checked_mul(n)
        .ok_or_else(|| "matrix element count overflow".to_string())?;
    let bytes_per = match dtype {
        NpyFloatDtype::F32Le => 4usize,
        NpyFloatDtype::F64Le => 8usize,
    };
    let payload_bytes = n_elem
        .checked_mul(bytes_per)
        .ok_or_else(|| "matrix payload size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(payload_bytes)
        .ok_or_else(|| "matrix payload end overflow".to_string())?;
    if data_end > mmap.len() {
        return Err("NPY data truncated".to_string());
    }
    let payload = &mmap[data_offset..data_end];

    let mut out = Vec::<f64>::with_capacity(n_elem);
    match dtype {
        NpyFloatDtype::F64Le => {
            for chunk in payload.chunks_exact(8) {
                let v = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(v);
            }
        }
        NpyFloatDtype::F32Le => {
            for chunk in payload.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
                out.push(v);
            }
        }
    }
    if out.len() != n_elem {
        return Err(format!(
            "decoded matrix length mismatch: got {}, expected {}",
            out.len(),
            n_elem
        ));
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err("matrix contains non-finite values".to_string());
    }
    apply_diag_shift_row_major(&mut out, n, diag_shift);
    Ok((out, n))
}

#[inline]
fn read_npy_float_elem_as_f64(
    payload: &[u8],
    dtype: NpyFloatDtype,
    idx: usize,
) -> Result<f64, String> {
    match dtype {
        NpyFloatDtype::F64Le => {
            let off = idx
                .checked_mul(8usize)
                .ok_or_else(|| "NPY element offset overflow".to_string())?;
            let end = off
                .checked_add(8usize)
                .ok_or_else(|| "NPY element end overflow".to_string())?;
            let chunk = payload
                .get(off..end)
                .ok_or_else(|| "NPY payload truncated while reading f64".to_string())?;
            Ok(f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]))
        }
        NpyFloatDtype::F32Le => {
            let off = idx
                .checked_mul(4usize)
                .ok_or_else(|| "NPY element offset overflow".to_string())?;
            let end = off
                .checked_add(4usize)
                .ok_or_else(|| "NPY element end overflow".to_string())?;
            let chunk = payload
                .get(off..end)
                .ok_or_else(|| "NPY payload truncated while reading f32".to_string())?;
            Ok(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
        }
    }
}

fn load_square_matrix_from_npy_sym_col_major_f64(
    path: &str,
    diag_shift: f64,
) -> Result<(Vec<f64>, usize), String> {
    let file = File::open(path).map_err(|e| format!("open NPY failed: {e}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| e.to_string())? };
    let (rows, cols, data_offset, dtype) = parse_npy_header_for_float_matrix(&mmap[..])?;
    if rows == 0 || cols == 0 || rows != cols {
        return Err(format!(
            "GRM/kinship matrix must be non-empty square, got ({rows}, {cols})"
        ));
    }
    let n = rows;
    let n_elem = n
        .checked_mul(n)
        .ok_or_else(|| "matrix element count overflow".to_string())?;
    let bytes_per = match dtype {
        NpyFloatDtype::F32Le => 4usize,
        NpyFloatDtype::F64Le => 8usize,
    };
    let payload_bytes = n_elem
        .checked_mul(bytes_per)
        .ok_or_else(|| "matrix payload size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(payload_bytes)
        .ok_or_else(|| "matrix payload end overflow".to_string())?;
    if data_end > mmap.len() {
        return Err("NPY data truncated".to_string());
    }
    let payload = &mmap[data_offset..data_end];

    let mut out = vec![0.0_f64; n_elem];
    for row in 0..n {
        for col in row..n {
            let idx_rc = row
                .checked_mul(n)
                .and_then(|v| v.checked_add(col))
                .ok_or_else(|| "NPY index overflow".to_string())?;
            let idx_cr = col
                .checked_mul(n)
                .and_then(|v| v.checked_add(row))
                .ok_or_else(|| "NPY index overflow".to_string())?;
            let a_rc = read_npy_float_elem_as_f64(payload, dtype, idx_rc)?;
            let a_cr = read_npy_float_elem_as_f64(payload, dtype, idx_cr)?;
            if !a_rc.is_finite() {
                return Err(format!(
                    "matrix contains non-finite value at ({row}, {col})"
                ));
            }
            if !a_cr.is_finite() {
                return Err(format!(
                    "matrix contains non-finite value at ({col}, {row})"
                ));
            }
            let mut v = if row == col {
                a_rc
            } else {
                0.5_f64 * (a_rc + a_cr)
            };
            if row == col && diag_shift != 0.0_f64 {
                v += diag_shift;
            }
            out[row + col * n] = v;
            out[col + row * n] = v;
        }
    }
    Ok((out, n))
}

fn load_square_matrix_from_npy_subset_sym_col_major_f64(
    path: &str,
    subset: &[usize],
    diag_shift: f64,
) -> Result<(Vec<f64>, usize), String> {
    let file = File::open(path).map_err(|e| format!("open NPY failed: {e}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| e.to_string())? };
    let (rows, cols, data_offset, dtype) = parse_npy_header_for_float_matrix(&mmap[..])?;
    if rows == 0 || cols == 0 || rows != cols {
        return Err(format!(
            "GRM/kinship matrix must be non-empty square, got ({rows}, {cols})"
        ));
    }
    let n = rows;
    validate_subset_indices_usize(
        subset,
        n,
        "load_square_matrix_from_npy_subset_sym_col_major_f64",
    )?;
    let k = subset.len();
    let n_elem = n
        .checked_mul(n)
        .ok_or_else(|| "matrix element count overflow".to_string())?;
    let bytes_per = match dtype {
        NpyFloatDtype::F32Le => 4usize,
        NpyFloatDtype::F64Le => 8usize,
    };
    let payload_bytes = n_elem
        .checked_mul(bytes_per)
        .ok_or_else(|| "matrix payload size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(payload_bytes)
        .ok_or_else(|| "matrix payload end overflow".to_string())?;
    if data_end > mmap.len() {
        return Err("NPY data truncated".to_string());
    }
    let payload = &mmap[data_offset..data_end];

    let mut out = vec![0.0_f64; k * k];
    for local_row in 0..k {
        let src_row = subset[local_row];
        for local_col in local_row..k {
            let src_col = subset[local_col];
            let idx_rc = src_row
                .checked_mul(n)
                .and_then(|v| v.checked_add(src_col))
                .ok_or_else(|| "subset NPY index overflow".to_string())?;
            let idx_cr = src_col
                .checked_mul(n)
                .and_then(|v| v.checked_add(src_row))
                .ok_or_else(|| "subset NPY index overflow".to_string())?;
            let a_rc = read_npy_float_elem_as_f64(payload, dtype, idx_rc)?;
            let a_cr = read_npy_float_elem_as_f64(payload, dtype, idx_cr)?;
            if !a_rc.is_finite() {
                return Err(format!(
                    "matrix contains non-finite value at ({src_row}, {src_col})"
                ));
            }
            if !a_cr.is_finite() {
                return Err(format!(
                    "matrix contains non-finite value at ({src_col}, {src_row})"
                ));
            }
            let mut v = if local_row == local_col {
                a_rc
            } else {
                0.5_f64 * (a_rc + a_cr)
            };
            if local_row == local_col && diag_shift != 0.0_f64 {
                v += diag_shift;
            }
            out[local_row + local_col * k] = v;
            out[local_col + local_row * k] = v;
        }
    }
    Ok((out, k))
}

fn subset_square_row_major_to_sym_col_major_f64(
    src_row_major: &[f64],
    n: usize,
    subset: &[usize],
    diag_shift: f64,
    label: &str,
) -> Result<(Vec<f64>, usize), String> {
    validate_eigh_dims(n, src_row_major.len(), label)?;
    validate_subset_indices_usize(subset, n, label)?;
    let k = subset.len();
    let mut out = vec![0.0_f64; k * k];
    for local_row in 0..k {
        let src_row = subset[local_row];
        for local_col in local_row..k {
            let src_col = subset[local_col];
            let a_rc = src_row_major[src_row * n + src_col];
            let a_cr = src_row_major[src_col * n + src_row];
            if !a_rc.is_finite() {
                return Err(format!(
                    "{label}: non-finite value at ({src_row}, {src_col})"
                ));
            }
            if !a_cr.is_finite() {
                return Err(format!(
                    "{label}: non-finite value at ({src_col}, {src_row})"
                ));
            }
            let mut v = if local_row == local_col {
                a_rc
            } else {
                0.5_f64 * (a_rc + a_cr)
            };
            if local_row == local_col && diag_shift != 0.0_f64 {
                v += diag_shift;
            }
            out[local_row + local_col * k] = v;
            out[local_col + local_row * k] = v;
        }
    }
    Ok((out, k))
}

fn load_square_matrix_subset_sym_col_major_f64(
    path: &str,
    subset: &[usize],
    diag_shift: f64,
) -> Result<(Vec<f64>, usize), String> {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "npy" => load_square_matrix_from_npy_subset_sym_col_major_f64(path, subset, diag_shift),
        "txt" | "tsv" | "csv" => {
            let (a_row_major, n) = load_square_matrix_from_text_f64(path, 0.0)?;
            subset_square_row_major_to_sym_col_major_f64(
                &a_row_major,
                n,
                subset,
                diag_shift,
                "load_square_matrix_subset_sym_col_major_f64",
            )
        }
        _ => {
            let (a_row_major, n) = load_square_matrix_from_text_f64(path, 0.0)?;
            subset_square_row_major_to_sym_col_major_f64(
                &a_row_major,
                n,
                subset,
                diag_shift,
                "load_square_matrix_subset_sym_col_major_f64",
            )
        }
    }
}

pub(crate) fn load_square_matrix_subset_row_major_f64(
    path: &str,
    subset: &[usize],
    diag_shift: f64,
) -> Result<(Vec<f64>, usize), String> {
    let (a_col_major, n) = load_square_matrix_subset_sym_col_major_f64(path, subset, diag_shift)?;
    Ok((copy_col_major_to_row_major(&a_col_major, n), n))
}

pub(crate) fn square_matrix_subset_cross_dot_f64(
    path: &str,
    row_indices: &[usize],
    col_indices: &[usize],
    weights: &[f64],
) -> Result<Vec<f64>, String> {
    if col_indices.len() != weights.len() {
        return Err(format!(
            "square_matrix_subset_cross_dot_f64 weight length mismatch: cols={} weights={}",
            col_indices.len(),
            weights.len()
        ));
    }
    if row_indices.is_empty() {
        return Ok(Vec::new());
    }
    if col_indices.is_empty() {
        return Ok(vec![0.0_f64; row_indices.len()]);
    }

    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext != "npy" {
        let (a_row_major, n) = load_square_matrix_from_text_f64(path, 0.0)?;
        validate_subset_indices_usize(
            row_indices,
            n,
            "square_matrix_subset_cross_dot_f64(row_indices)",
        )?;
        validate_subset_indices_usize(
            col_indices,
            n,
            "square_matrix_subset_cross_dot_f64(col_indices)",
        )?;
        let mut out = vec![0.0_f64; row_indices.len()];
        for (dst_row, &src_row) in row_indices.iter().enumerate() {
            let mut acc = 0.0_f64;
            for (&src_col, &w) in col_indices.iter().zip(weights.iter()) {
                let a_rc = a_row_major[src_row * n + src_col];
                let a_cr = a_row_major[src_col * n + src_row];
                if !a_rc.is_finite() {
                    return Err(format!(
                        "square_matrix_subset_cross_dot_f64: non-finite value at ({src_row}, {src_col})"
                    ));
                }
                if !a_cr.is_finite() {
                    return Err(format!(
                        "square_matrix_subset_cross_dot_f64: non-finite value at ({src_col}, {src_row})"
                    ));
                }
                let v = if src_row == src_col {
                    a_rc
                } else {
                    0.5_f64 * (a_rc + a_cr)
                };
                acc += v * w;
            }
            out[dst_row] = acc;
        }
        return Ok(out);
    }

    let file = File::open(path).map_err(|e| format!("open NPY failed: {e}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| e.to_string())? };
    let (rows, cols, data_offset, dtype) = parse_npy_header_for_float_matrix(&mmap[..])?;
    if rows == 0 || cols == 0 || rows != cols {
        return Err(format!(
            "GRM/kinship matrix must be non-empty square, got ({rows}, {cols})"
        ));
    }
    let n = rows;
    validate_subset_indices_usize(
        row_indices,
        n,
        "square_matrix_subset_cross_dot_f64(row_indices)",
    )?;
    validate_subset_indices_usize(
        col_indices,
        n,
        "square_matrix_subset_cross_dot_f64(col_indices)",
    )?;
    let n_elem = n
        .checked_mul(n)
        .ok_or_else(|| "matrix element count overflow".to_string())?;
    let bytes_per = match dtype {
        NpyFloatDtype::F32Le => 4usize,
        NpyFloatDtype::F64Le => 8usize,
    };
    let payload_bytes = n_elem
        .checked_mul(bytes_per)
        .ok_or_else(|| "matrix payload size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(payload_bytes)
        .ok_or_else(|| "matrix payload end overflow".to_string())?;
    if data_end > mmap.len() {
        return Err("NPY data truncated".to_string());
    }
    let payload = &mmap[data_offset..data_end];

    let mut out = vec![0.0_f64; row_indices.len()];
    for (dst_row, &src_row) in row_indices.iter().enumerate() {
        let mut acc = 0.0_f64;
        for (&src_col, &w) in col_indices.iter().zip(weights.iter()) {
            let idx_rc = src_row
                .checked_mul(n)
                .and_then(|v| v.checked_add(src_col))
                .ok_or_else(|| "square_matrix_subset_cross_dot_f64 index overflow".to_string())?;
            let idx_cr = src_col
                .checked_mul(n)
                .and_then(|v| v.checked_add(src_row))
                .ok_or_else(|| "square_matrix_subset_cross_dot_f64 index overflow".to_string())?;
            let a_rc = read_npy_float_elem_as_f64(payload, dtype, idx_rc)?;
            let a_cr = read_npy_float_elem_as_f64(payload, dtype, idx_cr)?;
            if !a_rc.is_finite() {
                return Err(format!(
                    "square_matrix_subset_cross_dot_f64: non-finite value at ({src_row}, {src_col})"
                ));
            }
            if !a_cr.is_finite() {
                return Err(format!(
                    "square_matrix_subset_cross_dot_f64: non-finite value at ({src_col}, {src_row})"
                ));
            }
            let v = if src_row == src_col {
                a_rc
            } else {
                0.5_f64 * (a_rc + a_cr)
            };
            acc += v * w;
        }
        out[dst_row] = acc;
    }
    Ok(out)
}

fn parse_text_matrix_row(line: &str) -> Vec<&str> {
    if line.contains('\t') {
        line.split('\t')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    } else if line.contains(',') {
        line.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        line.split_whitespace().collect()
    }
}

fn load_square_matrix_from_text_f64(
    path: &str,
    diag_shift: f64,
) -> Result<(Vec<f64>, usize), String> {
    let file = File::open(path).map_err(|e| format!("open text matrix failed: {e}"))?;
    let reader = BufReader::new(file);

    let mut ncols: Option<usize> = None;
    let mut nrows: usize = 0;
    let mut out = Vec::<f64>::new();

    for (ln_idx, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| format!("read line {} failed: {e}", ln_idx + 1))?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let toks = parse_text_matrix_row(t);
        if toks.is_empty() {
            continue;
        }
        let row_cols = toks.len();
        if let Some(prev) = ncols {
            if row_cols != prev {
                return Err(format!(
                    "text matrix row width mismatch at line {}: got {}, expected {}",
                    ln_idx + 1,
                    row_cols,
                    prev
                ));
            }
        } else {
            ncols = Some(row_cols);
        }
        for tok in toks {
            let v = tok
                .parse::<f64>()
                .map_err(|_| format!("invalid float at line {}: {tok}", ln_idx + 1))?;
            if !v.is_finite() {
                return Err(format!("non-finite value at line {}: {tok}", ln_idx + 1));
            }
            out.push(v);
        }
        nrows = nrows.saturating_add(1);
    }

    let cols = ncols.ok_or_else(|| "text matrix is empty".to_string())?;
    if cols == 0 || nrows == 0 || nrows != cols {
        return Err(format!(
            "GRM/kinship matrix must be non-empty square, got ({nrows}, {cols})"
        ));
    }
    let n = nrows;
    let expect = n
        .checked_mul(n)
        .ok_or_else(|| "text matrix size overflow".to_string())?;
    if out.len() != expect {
        return Err(format!(
            "text matrix payload mismatch: got {}, expected {}",
            out.len(),
            expect
        ));
    }
    apply_diag_shift_row_major(&mut out, n, diag_shift);
    Ok((out, n))
}

fn load_square_matrix_f64_from_file(
    path: &str,
    diag_shift: f64,
) -> Result<(Vec<f64>, usize), String> {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "npy" => load_square_matrix_from_npy_f64(path, diag_shift),
        "txt" | "tsv" | "csv" => load_square_matrix_from_text_f64(path, diag_shift),
        _ => load_square_matrix_from_text_f64(path, diag_shift),
    }
}

#[inline]
fn symmetric_eigh_f64_matrix_file_with_driver(
    path: &str,
    driver: EighDriver,
    jobz: EighJobz,
    diag_shift: f64,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str, usize), String> {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "npy" {
        let (a_col_major, n) = load_square_matrix_from_npy_sym_col_major_f64(path, diag_shift)?;
        if n == 0 {
            let vecs = if matches!(jobz, EighJobz::ValuesAndVectors) {
                Some(Vec::new())
            } else {
                None
            };
            return Ok((Vec::new(), vecs, "empty", 0));
        }
        let primary = resolve_eigh_driver_for_problem(driver, n, jobz);
        let secondary = match primary {
            EighDriver::Dsyevd => EighDriver::Dsyevr,
            EighDriver::Dsyevr | EighDriver::Auto => EighDriver::Dsyevd,
        };

        #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
        {
            match lapack_run_driver_from_col_major_owned(a_col_major, n, jobz, primary) {
                Ok((evals, evecs, backend)) => return Ok((evals, evecs, backend, n)),
                Err(primary_err) => {
                    let a_retry =
                        load_square_matrix_from_npy_sym_col_major_f64(path, diag_shift)?.0;
                    match lapack_run_driver_from_col_major_owned(a_retry, n, jobz, secondary) {
                        Ok((evals, evecs, backend)) => return Ok((evals, evecs, backend, n)),
                        Err(secondary_err) => {
                            let allow_nalgebra_fallback = if cfg!(target_os = "windows") {
                                match std::env::var("JX_EIGH_ALLOW_NALGEBRA_FALLBACK") {
                                    Ok(v) => {
                                        let t = v.trim().to_ascii_lowercase();
                                        matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
                                    }
                                    Err(_) => true,
                                }
                            } else {
                                env_truthy("JX_EIGH_ALLOW_NALGEBRA_FALLBACK")
                            };
                            if !allow_nalgebra_fallback {
                                return Err(format!(
                                    "LAPACK eigh failed (driver={}, jobz={}): primary={}; secondary={}",
                                    match primary {
                                        EighDriver::Dsyevd => "dsyevd",
                                        EighDriver::Dsyevr | EighDriver::Auto => "dsyevr",
                                    },
                                    eigh_jobz_label(jobz),
                                    primary_err,
                                    secondary_err
                                ));
                            }
                        }
                    }
                }
            }
        }

        let a_fallback = load_square_matrix_from_npy_sym_col_major_f64(path, diag_shift)?.0;
        let a_row_major = copy_col_major_to_row_major(&a_fallback, n);
        let dm = DMatrix::from_row_slice(n, n, &a_row_major);
        let se = nalgebra::linalg::SymmetricEigen::new(dm);
        let eval_unsorted = se.eigenvalues.as_slice();
        let evec_unsorted = se.eigenvectors;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            eval_unsorted[a]
                .partial_cmp(&eval_unsorted[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut s = vec![0.0_f64; n];
        let mut u = if matches!(jobz, EighJobz::ValuesAndVectors) {
            Some(vec![0.0_f64; n * n])
        } else {
            None
        };
        for (col_new, &col_old) in order.iter().enumerate() {
            s[col_new] = eval_unsorted[col_old];
            if let Some(ref mut uu) = u {
                for row in 0..n {
                    uu[row * n + col_new] = evec_unsorted[(row, col_old)];
                }
            }
        }
        return Ok((s, u, "nalgebra", n));
    }

    let (a_row_major, n) = load_square_matrix_f64_from_file(path, diag_shift)?;
    if n == 0 {
        let vecs = if matches!(jobz, EighJobz::ValuesAndVectors) {
            Some(Vec::new())
        } else {
            None
        };
        return Ok((Vec::new(), vecs, "empty", 0));
    }
    let (evals, evecs, backend) =
        symmetric_eigh_f64_row_major_with_driver(&a_row_major, n, driver, jobz)?;
    Ok((evals, evecs, backend, n))
}

#[inline]
fn symmetric_eigh_f64_matrix_file_subset_with_driver(
    path: &str,
    subset: &[usize],
    driver: EighDriver,
    jobz: EighJobz,
    diag_shift: f64,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str, usize), String> {
    let (a_col_major, n) = load_square_matrix_subset_sym_col_major_f64(path, subset, diag_shift)?;
    if n == 0 {
        let vecs = if matches!(jobz, EighJobz::ValuesAndVectors) {
            Some(Vec::new())
        } else {
            None
        };
        return Ok((Vec::new(), vecs, "empty", 0));
    }
    let primary = resolve_eigh_driver_for_problem(driver, n, jobz);
    let secondary = match primary {
        EighDriver::Dsyevd => EighDriver::Dsyevr,
        EighDriver::Dsyevr | EighDriver::Auto => EighDriver::Dsyevd,
    };

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        match lapack_run_driver_from_col_major_owned(a_col_major, n, jobz, primary) {
            Ok((evals, evecs, backend)) => return Ok((evals, evecs, backend, n)),
            Err(primary_err) => {
                let a_retry =
                    load_square_matrix_subset_sym_col_major_f64(path, subset, diag_shift)?.0;
                match lapack_run_driver_from_col_major_owned(a_retry, n, jobz, secondary) {
                    Ok((evals, evecs, backend)) => return Ok((evals, evecs, backend, n)),
                    Err(secondary_err) => {
                        let allow_nalgebra_fallback = if cfg!(target_os = "windows") {
                            match std::env::var("JX_EIGH_ALLOW_NALGEBRA_FALLBACK") {
                                Ok(v) => {
                                    let t = v.trim().to_ascii_lowercase();
                                    matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
                                }
                                Err(_) => true,
                            }
                        } else {
                            env_truthy("JX_EIGH_ALLOW_NALGEBRA_FALLBACK")
                        };
                        if !allow_nalgebra_fallback {
                            return Err(format!(
                                "LAPACK subset eigh failed (driver={}, jobz={}): primary={}; secondary={}",
                                match primary {
                                    EighDriver::Dsyevd => "dsyevd",
                                    EighDriver::Dsyevr | EighDriver::Auto => "dsyevr",
                                },
                                eigh_jobz_label(jobz),
                                primary_err,
                                secondary_err
                            ));
                        }
                    }
                }
            }
        }
    }

    let a_fallback = load_square_matrix_subset_sym_col_major_f64(path, subset, diag_shift)?.0;
    let a_row_major = copy_col_major_to_row_major(&a_fallback, n);
    let dm = DMatrix::from_row_slice(n, n, &a_row_major);
    let se = nalgebra::linalg::SymmetricEigen::new(dm);
    let eval_unsorted = se.eigenvalues.as_slice();
    let evec_unsorted = se.eigenvectors;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eval_unsorted[a]
            .partial_cmp(&eval_unsorted[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut s = vec![0.0_f64; n];
    let mut u = if matches!(jobz, EighJobz::ValuesAndVectors) {
        Some(vec![0.0_f64; n * n])
    } else {
        None
    };
    for (col_new, &col_old) in order.iter().enumerate() {
        s[col_new] = eval_unsorted[col_old];
        if let Some(ref mut uu) = u {
            for row in 0..n {
                uu[row * n + col_new] = evec_unsorted[(row, col_old)];
            }
        }
    }
    Ok((s, u, "nalgebra", n))
}

#[inline]
fn lapack_dsyevr_col_major_raw(
    a_col_major: &mut [f64],
    n: usize,
    jobz: EighJobz,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str), String> {
    let n_i = n as CblasInt;
    let lda = n_i;
    let jobz_flag = [
        eigh_jobz_label(jobz).as_bytes()[0] as std::os::raw::c_char,
        0,
    ];
    let uplo = [b'U' as std::os::raw::c_char, 0];
    let need_vec = matches!(jobz, EighJobz::ValuesAndVectors);

    let mut w = vec![0.0_f64; n];
    let mut z = if need_vec {
        vec![0.0_f64; n * n]
    } else {
        vec![0.0_f64; 1]
    };
    let ldz = if need_vec { n_i } else { 1 as CblasInt };
    let range = [b'A' as std::os::raw::c_char, 0];
    let vl = 0.0_f64;
    let vu = 0.0_f64;
    let il: CblasInt = 0;
    let iu: CblasInt = 0;
    let abstol = 0.0_f64;
    let mut m_found: CblasInt = 0;
    let mut isuppz = vec![0 as CblasInt; 2 * n];
    let mut info: CblasInt = 0;
    let mut lwork: CblasInt = -1;
    let mut liwork: CblasInt = -1;
    let mut work_q = [0.0_f64];
    let mut iwork_q = [0 as CblasInt];

    let lapack_ready = unsafe {
        lapack_dsyevr_dispatch(
            jobz_flag.as_ptr(),
            range.as_ptr(),
            uplo.as_ptr(),
            &n_i,
            a_col_major.as_mut_ptr(),
            &lda,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &mut m_found,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            &ldz,
            isuppz.as_mut_ptr(),
            work_q.as_mut_ptr(),
            &lwork,
            iwork_q.as_mut_ptr(),
            &liwork,
            &mut info,
        )
    };
    if lapack_ready.is_err() {
        return Err("dsyevr workspace query dispatch unavailable".to_string());
    }
    if info != 0 {
        return Err(format!("dsyevr workspace query failed: info={info}"));
    }

    let (lwork_use, lwork_len) = lapack_f64_workspace_len(work_q[0], "dsyevr")?;
    let (liwork_use, liwork_len) = lapack_i32_workspace_len(iwork_q[0], "dsyevr")?;
    let mut work = vec![0.0_f64; lwork_len];
    let mut iwork = vec![0 as CblasInt; liwork_len];
    info = 0;
    lwork = lwork_use;
    liwork = liwork_use;
    m_found = 0;
    let run_ok = unsafe {
        lapack_dsyevr_dispatch(
            jobz_flag.as_ptr(),
            range.as_ptr(),
            uplo.as_ptr(),
            &n_i,
            a_col_major.as_mut_ptr(),
            &lda,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &mut m_found,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            &ldz,
            isuppz.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        )
    };
    if run_ok.is_err() {
        return Err("dsyevr run dispatch unavailable".to_string());
    }
    if info != 0 {
        return Err(format!("dsyevr run failed: info={info}"));
    }
    if (m_found as usize) != n {
        return Err(format!("dsyevr returned m={m_found}, expected n={n}"));
    }

    Ok((
        w,
        if need_vec { Some(z) } else { None },
        lapack_dsyevr_backend_label(),
    ))
}

#[inline]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn lapack_dsyevd_col_major_inplace(
    a_col_major: &mut [f64],
    n: usize,
    jobz: EighJobz,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str), String> {
    let n_i = n as CblasInt;
    let lda = n_i;
    let jobz_flag = [
        eigh_jobz_label(jobz).as_bytes()[0] as std::os::raw::c_char,
        0,
    ];
    let uplo = [b'U' as std::os::raw::c_char, 0];
    let need_vec = matches!(jobz, EighJobz::ValuesAndVectors);

    let mut w = vec![0.0_f64; n];
    let mut info: CblasInt = 0;
    let mut lwork: CblasInt = -1;
    let mut liwork: CblasInt = -1;
    let mut work_q = [0.0_f64];
    let mut iwork_q = [0 as CblasInt];

    let lapack_ready = unsafe {
        lapack_dsyevd_dispatch(
            jobz_flag.as_ptr(),
            uplo.as_ptr(),
            &n_i,
            a_col_major.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work_q.as_mut_ptr(),
            &lwork,
            iwork_q.as_mut_ptr(),
            &liwork,
            &mut info,
        )
    };
    if lapack_ready.is_err() {
        return Err("dsyevd workspace query dispatch unavailable".to_string());
    }
    if info != 0 {
        return Err(format!("dsyevd workspace query failed: info={info}"));
    }

    let (lwork_use, lwork_len) = lapack_f64_workspace_len(work_q[0], "dsyevd")?;
    let (liwork_use, liwork_len) = lapack_i32_workspace_len(iwork_q[0], "dsyevd")?;
    let mut work = vec![0.0_f64; lwork_len];
    let mut iwork = vec![0 as CblasInt; liwork_len];
    info = 0;
    lwork = lwork_use;
    liwork = liwork_use;
    let run_ok = unsafe {
        lapack_dsyevd_dispatch(
            jobz_flag.as_ptr(),
            uplo.as_ptr(),
            &n_i,
            a_col_major.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        )
    };
    if run_ok.is_err() {
        return Err("dsyevd run dispatch unavailable".to_string());
    }
    if info != 0 {
        return Err(format!("dsyevd run failed: info={info}"));
    }

    if need_vec {
        let u_row = copy_col_major_to_row_major(a_col_major, n);
        Ok((w, Some(u_row), lapack_dsyevd_backend_label()))
    } else {
        Ok((w, None, lapack_dsyevd_backend_label()))
    }
}

#[inline]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn lapack_run_driver_from_col_major_owned(
    mut a_col_major: Vec<f64>,
    n: usize,
    jobz: EighJobz,
    driver: EighDriver,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str), String> {
    match driver {
        EighDriver::Dsyevd => lapack_dsyevd_col_major_inplace(&mut a_col_major, n, jobz),
        EighDriver::Dsyevr => {
            let (evals, z_col_major, backend) =
                lapack_dsyevr_col_major_raw(&mut a_col_major, n, jobz)?;
            drop(a_col_major);
            let u_row = z_col_major.map(|z| copy_col_major_to_row_major(&z, n));
            Ok((evals, u_row, backend))
        }
        EighDriver::Auto => Err("internal error: unresolved auto LAPACK driver".to_string()),
    }
}

#[inline]
fn symmetric_eigh_f64_row_major_with_driver(
    a_row_major: &[f64],
    n: usize,
    driver: EighDriver,
    jobz: EighJobz,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str), String> {
    if n == 0 {
        let vecs = if matches!(jobz, EighJobz::ValuesAndVectors) {
            Some(Vec::new())
        } else {
            None
        };
        return Ok((Vec::new(), vecs, "empty"));
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        let mut lapack_errors: Vec<String> = Vec::new();
        let primary = resolve_eigh_driver_for_problem(driver, n, jobz);
        let secondary = match primary {
            EighDriver::Dsyevd => EighDriver::Dsyevr,
            EighDriver::Dsyevr | EighDriver::Auto => EighDriver::Dsyevd,
        };
        let a_primary =
            symmetrize_row_major_to_col_major_f64(a_row_major, n, "symmetric_eigh_f64_row_major")?;

        match lapack_run_driver_from_col_major_owned(a_primary, n, jobz, primary) {
            Ok(res) => return Ok(res),
            Err(err) => {
                lapack_errors.push(format!(
                    "{}: {err}",
                    match primary {
                        EighDriver::Dsyevd => "dsyevd",
                        EighDriver::Dsyevr | EighDriver::Auto => "dsyevr",
                    }
                ));
            }
        }

        let a_retry =
            symmetrize_row_major_to_col_major_f64(a_row_major, n, "symmetric_eigh_f64_row_major")?;
        match lapack_run_driver_from_col_major_owned(a_retry, n, jobz, secondary) {
            Ok(res) => return Ok(res),
            Err(err) => {
                lapack_errors.push(format!(
                    "{}: {err}",
                    match secondary {
                        EighDriver::Dsyevd => "dsyevd",
                        EighDriver::Dsyevr | EighDriver::Auto => "dsyevr",
                    }
                ));
            }
        }

        let allow_nalgebra_fallback = if cfg!(target_os = "windows") {
            // Windows default keeps fallback enabled, but allow explicit env
            // override for diagnostics/perf debugging.
            match std::env::var("JX_EIGH_ALLOW_NALGEBRA_FALLBACK") {
                Ok(v) => {
                    let t = v.trim().to_ascii_lowercase();
                    matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
                }
                Err(_) => true,
            }
        } else {
            env_truthy("JX_EIGH_ALLOW_NALGEBRA_FALLBACK")
        };
        if !allow_nalgebra_fallback {
            return Err(format!(
                "LAPACK eigh failed (driver={}, jobz={}): {}",
                match driver {
                    EighDriver::Dsyevd => "dsyevd",
                    EighDriver::Dsyevr | EighDriver::Auto => "dsyevr(auto)",
                },
                eigh_jobz_label(jobz),
                lapack_errors.join("; ")
            ));
        }
    }

    let a_sym_row_major = symmetrize_row_major_f64(a_row_major, n, "symmetric_eigh_f64_row_major")?;
    let dm = DMatrix::from_row_slice(n, n, &a_sym_row_major);
    let se = nalgebra::linalg::SymmetricEigen::new(dm);
    let eval_unsorted = se.eigenvalues.as_slice();
    let evec_unsorted = se.eigenvectors;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eval_unsorted[a]
            .partial_cmp(&eval_unsorted[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut s = vec![0.0_f64; n];
    let mut u = if matches!(jobz, EighJobz::ValuesAndVectors) {
        Some(vec![0.0_f64; n * n])
    } else {
        None
    };
    for (col_new, &col_old) in order.iter().enumerate() {
        s[col_new] = eval_unsorted[col_old];
        if let Some(ref mut uu) = u {
            for row in 0..n {
                uu[row * n + col_new] = evec_unsorted[(row, col_old)];
            }
        }
    }
    Ok((s, u, "nalgebra"))
}

pub fn symmetric_eigh_f64_row_major(
    a_row_major: &[f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>, &'static str), String> {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    let driver = resolve_eigh_driver_from_env();
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let driver = EighDriver::Dsyevd;
    let (evals, evecs, backend) = symmetric_eigh_f64_row_major_with_driver(
        a_row_major,
        n,
        driver,
        EighJobz::ValuesAndVectors,
    )?;
    let vecs = evecs.ok_or_else(|| "eigh internal error: missing eigenvectors".to_string())?;
    Ok((evals, vecs, backend))
}

#[pyfunction]
#[pyo3(signature = (n=2048, threads=0, seed=20260501, driver=None, jobz="V", require_lapack=false))]
pub fn rust_eigh_debug_f64(
    n: usize,
    threads: usize,
    seed: u64,
    driver: Option<String>,
    jobz: &str,
    require_lapack: bool,
) -> PyResult<(
    String,
    String,
    String,
    usize,
    isize,
    isize,
    isize,
    bool,
    f64,
    f64,
    f64,
)> {
    if n == 0 {
        return Err(PyRuntimeError::new_err("n must be > 0"));
    }
    let driver_sel = parse_eigh_driver_token(driver.unwrap_or_else(|| "auto".to_string()).as_str());
    let jobz_sel = parse_eigh_jobz_token(jobz);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = rng.random::<f64>() - 0.5_f64;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }

    let backend = crate::blas::rust_sgemm_backend();
    let (threads_before, threads_in_stage, run, threads_after) =
        crate::blas::with_eigh_thread_stage(threads, || {
            let t0 = Instant::now();
            let run = symmetric_eigh_f64_row_major_with_driver(&a, n, driver_sel, jobz_sel);
            let elapsed = t0.elapsed().as_secs_f64();
            (run, elapsed)
        });
    let (run, elapsed) = run;

    let (evals, _evecs, evd_backend) = run.map_err(PyRuntimeError::new_err)?;
    let lapack_used = is_lapack_backend(evd_backend);
    if require_lapack && !lapack_used {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_debug_f64 expected LAPACK backend, got {evd_backend}"
        )));
    }
    let min_eval = evals.first().copied().unwrap_or(f64::NAN);
    let max_eval = evals.last().copied().unwrap_or(f64::NAN);

    Ok((
        backend,
        evd_backend.to_string(),
        eigh_jobz_label(jobz_sel).to_string(),
        n,
        threads_before,
        threads_in_stage,
        threads_after,
        lapack_used,
        elapsed,
        min_eval,
        max_eval,
    ))
}

#[pyfunction]
#[pyo3(signature = (a, threads=0, driver=None, jobz="V", require_lapack=false))]
pub fn rust_eigh_from_array_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    threads: usize,
    driver: Option<String>,
    jobz: &str,
    require_lapack: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
    String,
    String,
    usize,
    isize,
    isize,
    isize,
    bool,
    f64,
)> {
    let a_view = a.as_array();
    let nrows = a_view.shape().first().copied().unwrap_or(0usize);
    let ncols = a_view.shape().get(1).copied().unwrap_or(0usize);
    if nrows == 0 || ncols == 0 || nrows != ncols {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_array_f64 expects a non-empty square matrix; got shape=({}, {})",
            nrows, ncols
        )));
    }
    let n = nrows;

    let driver_sel = parse_eigh_driver_token(driver.unwrap_or_else(|| "auto".to_string()).as_str());
    let jobz_sel = parse_eigh_jobz_token(jobz);
    let a_row_major = pyreadonly_array2_to_row_major_cow(&a);

    let backend = crate::blas::rust_sgemm_backend();
    let (threads_before, threads_in_stage, run, threads_after) =
        crate::blas::with_eigh_thread_stage(threads, || {
            let t0 = Instant::now();
            let run = symmetric_eigh_f64_row_major_with_driver(
                a_row_major.as_ref(),
                n,
                driver_sel,
                jobz_sel,
            );
            let elapsed = t0.elapsed().as_secs_f64();
            (run, elapsed)
        });
    let (run, elapsed) = run;

    let (evals, evecs_opt, evd_backend) = run.map_err(PyRuntimeError::new_err)?;
    let lapack_used = is_lapack_backend(evd_backend);
    if require_lapack && !lapack_used {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_array_f64 expected LAPACK backend, got {evd_backend}"
        )));
    }

    let evals_arr = PyArray1::from_owned_array(py, Array1::from_vec(evals)).into_bound();
    let evecs_arr = match evecs_opt {
        Some(v) => Some(
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n, n), v)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
            .into_bound(),
        ),
        None => None,
    };

    Ok((
        evals_arr,
        evecs_arr,
        backend,
        evd_backend.to_string(),
        n,
        threads_before,
        threads_in_stage,
        threads_after,
        lapack_used,
        elapsed,
    ))
}

#[pyfunction]
#[pyo3(signature = (path, threads=0, driver=None, jobz="V", require_lapack=false, diag_shift=0.0))]
pub fn rust_eigh_from_matrix_file_f64<'py>(
    py: Python<'py>,
    path: String,
    threads: usize,
    driver: Option<String>,
    jobz: &str,
    require_lapack: bool,
    diag_shift: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
    String,
    String,
    usize,
    isize,
    isize,
    isize,
    bool,
    f64,
)> {
    let path_trimmed = path.trim();
    if path_trimmed.is_empty() {
        return Err(PyRuntimeError::new_err("path must not be empty"));
    }
    let driver_sel = parse_eigh_driver_token(driver.unwrap_or_else(|| "auto".to_string()).as_str());
    let jobz_sel = parse_eigh_jobz_token(jobz);

    let backend = crate::blas::rust_sgemm_backend();
    let (threads_before, threads_in_stage, run, threads_after) =
        crate::blas::with_eigh_thread_stage(threads, || {
            let t0 = Instant::now();
            let run = symmetric_eigh_f64_matrix_file_with_driver(
                path_trimmed,
                driver_sel,
                jobz_sel,
                diag_shift,
            );
            let elapsed = t0.elapsed().as_secs_f64();
            (run, elapsed)
        });
    let (run, elapsed) = run;

    let (evals, evecs_opt, evd_backend, n) = run.map_err(PyRuntimeError::new_err)?;
    let lapack_used = is_lapack_backend(evd_backend);
    if require_lapack && !lapack_used {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_matrix_file_f64 expected LAPACK backend, got {evd_backend}"
        )));
    }

    let evals_arr = PyArray1::from_owned_array(py, Array1::from_vec(evals)).into_bound();
    let evecs_arr = match evecs_opt {
        Some(v) => Some(
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n, n), v)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
            .into_bound(),
        ),
        None => None,
    };

    Ok((
        evals_arr,
        evecs_arr,
        backend,
        evd_backend.to_string(),
        n,
        threads_before,
        threads_in_stage,
        threads_after,
        lapack_used,
        elapsed,
    ))
}

#[pyfunction]
#[pyo3(signature = (path, subset_indices, threads=0, driver=None, jobz="V", require_lapack=false, diag_shift=0.0))]
pub fn rust_eigh_from_matrix_file_subset_f64<'py>(
    py: Python<'py>,
    path: String,
    subset_indices: PyReadonlyArray1<'py, i64>,
    threads: usize,
    driver: Option<String>,
    jobz: &str,
    require_lapack: bool,
    diag_shift: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
    String,
    String,
    usize,
    isize,
    isize,
    isize,
    bool,
    f64,
)> {
    let path_trimmed = path.trim();
    if path_trimmed.is_empty() {
        return Err(PyRuntimeError::new_err("path must not be empty"));
    }
    let subset_raw = subset_indices.as_slice()?;
    if subset_raw.is_empty() {
        return Err(PyRuntimeError::new_err(
            "rust_eigh_from_matrix_file_subset_f64 subset_indices must not be empty",
        ));
    }
    let mut subset_vec = Vec::<usize>::with_capacity(subset_raw.len());
    for (pos, &v) in subset_raw.iter().enumerate() {
        if v < 0 {
            return Err(PyRuntimeError::new_err(format!(
                "rust_eigh_from_matrix_file_subset_f64 subset index at position {} is negative ({})",
                pos, v
            )));
        }
        subset_vec.push(v as usize);
    }
    let driver_sel = parse_eigh_driver_token(driver.unwrap_or_else(|| "auto".to_string()).as_str());
    let jobz_sel = parse_eigh_jobz_token(jobz);

    let backend = crate::blas::rust_sgemm_backend();
    let (threads_before, threads_in_stage, run, threads_after) =
        crate::blas::with_eigh_thread_stage(threads, || {
            let t0 = Instant::now();
            let run = symmetric_eigh_f64_matrix_file_subset_with_driver(
                path_trimmed,
                subset_vec.as_slice(),
                driver_sel,
                jobz_sel,
                diag_shift,
            );
            let elapsed = t0.elapsed().as_secs_f64();
            (run, elapsed)
        });
    let (run, elapsed) = run;

    let (evals, evecs_opt, evd_backend, n) = run.map_err(PyRuntimeError::new_err)?;
    let lapack_used = is_lapack_backend(evd_backend);
    if require_lapack && !lapack_used {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_matrix_file_subset_f64 expected LAPACK backend, got {evd_backend}"
        )));
    }

    let evals_arr = PyArray1::from_owned_array(py, Array1::from_vec(evals)).into_bound();
    let evecs_arr = match evecs_opt {
        Some(v) => Some(
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n, n), v)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
            .into_bound(),
        ),
        None => None,
    };

    Ok((
        evals_arr,
        evecs_arr,
        backend,
        evd_backend.to_string(),
        n,
        threads_before,
        threads_in_stage,
        threads_after,
        lapack_used,
        elapsed,
    ))
}

#[pyfunction]
#[pyo3(signature = (a, threads=0, driver=None, jobz="V", require_lapack=false))]
pub fn rust_eigh_from_array_f64_inplace<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    threads: usize,
    driver: Option<String>,
    jobz: &str,
    require_lapack: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
    String,
    String,
    usize,
    isize,
    isize,
    isize,
    bool,
    f64,
)> {
    let shape = a.shape();
    let nrows = shape.first().copied().unwrap_or(0usize);
    let ncols = shape.get(1).copied().unwrap_or(0usize);
    if nrows == 0 || ncols == 0 || nrows != ncols {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_array_f64_inplace expects a non-empty square matrix; got shape=({}, {})",
            nrows, ncols
        )));
    }
    let n = nrows;
    let a_ro = a.readonly();

    let driver_sel = parse_eigh_driver_token(driver.unwrap_or_else(|| "auto".to_string()).as_str());
    let jobz_sel = parse_eigh_jobz_token(jobz);
    let a_row_major = pyreadonly_array2_to_row_major_cow(&a_ro);

    let backend = crate::blas::rust_sgemm_backend();
    let (threads_before, threads_in_stage, run, threads_after) =
        crate::blas::with_eigh_thread_stage(threads, || {
            let t0 = Instant::now();
            let run = symmetric_eigh_f64_row_major_with_driver(
                a_row_major.as_ref(),
                n,
                driver_sel,
                jobz_sel,
            );
            let elapsed = t0.elapsed().as_secs_f64();
            (run, elapsed)
        });
    let (run, elapsed) = run;

    let (evals, evecs_opt, evd_backend) = run.map_err(PyRuntimeError::new_err)?;
    let lapack_used = is_lapack_backend(evd_backend);
    if require_lapack && !lapack_used {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_array_f64_inplace expected LAPACK backend, got {evd_backend}"
        )));
    }

    let evals_arr = PyArray1::from_owned_array(py, Array1::from_vec(evals)).into_bound();
    let evecs_arr = match evecs_opt {
        Some(v) => Some(
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n, n), v)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
            .into_bound(),
        ),
        None => None,
    };

    Ok((
        evals_arr,
        evecs_arr,
        backend,
        evd_backend.to_string(),
        n,
        threads_before,
        threads_in_stage,
        threads_after,
        lapack_used,
        elapsed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetrize_row_major_f64_averages_offdiag() {
        let a = vec![1.0_f64, 2.0_f64, 4.0_f64, 3.0_f64];
        let out = symmetrize_row_major_f64(&a, 2, "test").unwrap();
        assert_eq!(out, vec![1.0_f64, 3.0_f64, 3.0_f64, 3.0_f64]);
    }

    #[test]
    fn test_eigh_row_major_small_matrix() {
        let a = vec![2.0_f64, 1.0_f64, 1.0_f64, 2.0_f64];
        let (evals, evecs_opt, _backend) = symmetric_eigh_f64_row_major_with_driver(
            &a,
            2,
            EighDriver::Dsyevd,
            EighJobz::ValuesAndVectors,
        )
        .unwrap();
        assert_eq!(evals.len(), 2);
        assert!((evals[0] - 1.0_f64).abs() < 1e-9_f64);
        assert!((evals[1] - 3.0_f64).abs() < 1e-9_f64);
        let evecs = evecs_opt.unwrap();
        assert_eq!(evecs.len(), 4);
        let dot = evecs[0] * evecs[2] + evecs[1] * evecs[3];
        assert!(dot.abs() < 1e-9_f64);
    }

    #[test]
    fn test_lapack_workspace_len_rejects_i32_overflow() {
        let overflow = (CblasInt::MAX as f64) + 1024.0_f64;
        let err = lapack_f64_workspace_len(overflow, "test").unwrap_err();
        assert!(err.contains("exceeds LAPACK integer limit"));
    }
}
