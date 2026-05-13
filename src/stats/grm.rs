use numpy::ndarray::{Array1, Array2};
use numpy::PyArray1;
use numpy::PyUntypedArrayMethods;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::bedmath::{
    adaptive_grm_block_rows, decode_row_centered_full_lut, decode_row_centered_full_lut_f64,
    decode_subset_row_from_full_scratch, decode_subset_row_from_full_scratch_f64,
    is_identity_indices, packed_byte_lut,
};
use crate::blas::{
    cblas_dsyrk_dispatch, cblas_sgemm_dispatch, cblas_ssyrk_dispatch, CblasInt,
    OpenBlasThreadGuard, CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS, CBLAS_UPPER,
};
use crate::gfcore::BedSnpIter;
use crate::stats_common::{
    check_ctrlc, env_truthy, get_cached_pool, map_err_string_to_py, parse_index_vec_i64,
};

#[path = "../math/grm.rs"]
mod grm_math;
use grm_math::{grm_packed_f32_with_stats_impl, grm_packed_f64_with_stats_impl};

#[allow(clippy::too_many_arguments)]
fn decode_grm_block(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip_vec: &[bool],
    row_maf_vec: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    method: usize,
    eps: f32,
    code4_lut: &[[u8; 4]; 256],
    row_start: usize,
    row_end: usize,
    n: usize,
    out_block: &mut [f32],
    out_varsum: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    if cur_rows == 0 {
        return Ok(());
    }
    if out_block.len() < cur_rows.saturating_mul(n) {
        return Err("decode_grm_block: out_block too small".to_string());
    }
    if out_varsum.len() < cur_rows {
        return Err("decode_grm_block: out_varsum too small".to_string());
    }
    let block = &mut out_block[..cur_rows * n];
    let varsum = &mut out_varsum[..cur_rows];

    let mut decode_run = || {
        if full_sample_fast {
            block
                .par_chunks_mut(n)
                .zip(varsum.par_iter_mut())
                .enumerate()
                .for_each(|(off, (out_row, row_varsum_dst))| {
                    let idx = row_start + off;
                    let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                    let flip = row_flip_vec[idx];
                    let p = row_maf_vec[idx].clamp(0.0, 0.5);
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
                    let value_lut: [f32; 4] = if flip {
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
                    decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, out_row);
                    if method == 1 {
                        *row_varsum_dst = var as f64;
                    }
                });
        } else {
            block
                .par_chunks_mut(n)
                .zip(varsum.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || vec![0.0_f32; n_samples],
                    |full_row, (off, (out_row, row_varsum_dst))| {
                        let idx = row_start + off;
                        let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                        let flip = row_flip_vec[idx];
                        let p = row_maf_vec[idx].clamp(0.0, 0.5);
                        let default_mean_g = 2.0_f32 * p;
                        let (var_centered, _row_sum) = decode_subset_row_from_full_scratch(
                            row,
                            n_samples,
                            sample_idx,
                            flip,
                            default_mean_g,
                            method,
                            eps,
                            code4_lut,
                            full_row.as_mut_slice(),
                            out_row,
                        );
                        if method == 1 {
                            *row_varsum_dst = var_centered;
                        }
                    },
                );
        }
    };

    if let Some(tp) = pool {
        tp.install(&mut decode_run);
    } else {
        decode_run();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_grm_block_f64(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip_vec: &[bool],
    row_maf_vec: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    method: usize,
    eps: f64,
    code4_lut: &[[u8; 4]; 256],
    row_start: usize,
    row_end: usize,
    n: usize,
    out_block: &mut [f64],
    out_varsum: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    if cur_rows == 0 {
        return Ok(());
    }
    if out_block.len() < cur_rows.saturating_mul(n) {
        return Err("decode_grm_block_f64: out_block too small".to_string());
    }
    if out_varsum.len() < cur_rows {
        return Err("decode_grm_block_f64: out_varsum too small".to_string());
    }
    let block = &mut out_block[..cur_rows * n];
    let varsum = &mut out_varsum[..cur_rows];

    let mut decode_run = || {
        if full_sample_fast {
            block
                .par_chunks_mut(n)
                .zip(varsum.par_iter_mut())
                .enumerate()
                .for_each(|(off, (out_row, row_varsum_dst))| {
                    let idx = row_start + off;
                    let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                    let flip = row_flip_vec[idx];
                    let p = row_maf_vec[idx].clamp(0.0, 0.5) as f64;
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
                    let value_lut: [f64; 4] = if flip {
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
                        row, n_samples, code4_lut, &value_lut, out_row,
                    );
                    if method == 1 {
                        *row_varsum_dst = var;
                    }
                });
        } else {
            block
                .par_chunks_mut(n)
                .zip(varsum.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || vec![0.0_f64; n_samples],
                    |full_row, (off, (out_row, row_varsum_dst))| {
                        let idx = row_start + off;
                        let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                        let flip = row_flip_vec[idx];
                        let p = row_maf_vec[idx].clamp(0.0, 0.5) as f64;
                        let default_mean_g = 2.0_f64 * p;
                        let (var_centered, _row_sum) = decode_subset_row_from_full_scratch_f64(
                            row,
                            n_samples,
                            sample_idx,
                            flip,
                            default_mean_g,
                            method,
                            eps,
                            code4_lut,
                            full_row.as_mut_slice(),
                            out_row,
                        );
                        if method == 1 {
                            *row_varsum_dst = var_centered;
                        }
                    },
                );
        }
    };

    if let Some(tp) = pool {
        tp.install(&mut decode_run);
    } else {
        decode_run();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn grm_packed_f64_from_stats_rust(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_indices: &[usize],
    method: usize,
    block_cols: usize,
    threads: usize,
) -> Result<(Vec<f64>, f64), String> {
    if n_samples == 0 {
        return Err("n_samples must be > 0".to_string());
    }
    if method != 1 && method != 2 {
        return Err(format!(
            "unsupported method={method}; expected 1 (centered) or 2 (standardized)"
        ));
    }
    if bytes_per_snp == 0 {
        return Err("bytes_per_snp must be > 0".to_string());
    }
    let expected_bps = n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(format!(
            "bytes_per_snp mismatch: got {bytes_per_snp}, expected {expected_bps}"
        ));
    }

    let m = row_flip.len();
    if m == 0 {
        return Err("row_flip must contain at least one SNP row".to_string());
    }
    if row_maf.len() != m {
        return Err(format!(
            "row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        ));
    }
    if packed_flat.len() != m.saturating_mul(bytes_per_snp) {
        return Err(format!(
            "packed size mismatch: got {}, expected {}",
            packed_flat.len(),
            m.saturating_mul(bytes_per_snp)
        ));
    }
    if sample_indices.is_empty() {
        return Err("sample_indices must not be empty".to_string());
    }
    if let Some(&bad) = sample_indices.iter().find(|&&sid| sid >= n_samples) {
        return Err(format!("sample index out of range: {bad} >= {n_samples}"));
    }

    let sample_idx = sample_indices;
    let n = sample_idx.len();
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);

    let varsum_full = if method == 1 && full_sample_fast {
        let mut acc = 0.0_f64;
        for &maf in row_maf {
            let p = maf as f64;
            let v = 2.0_f64 * p * (1.0_f64 - p);
            if v.is_finite() && v > 0.0 {
                acc += v;
            }
        }
        if !(acc.is_finite() && acc > 0.0) {
            return Err("invalid centered GRM denominator: sum(2p(1-p)) <= 0".to_string());
        }
        acc
    } else {
        0.0_f64
    };

    let total_threads = effective_threads(threads);
    let pool = if total_threads > 1 {
        Some(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(total_threads)
                .build()
                .map_err(|e| format!("rayon pool: {e}"))?,
        ))
    } else {
        None
    };
    let row_step = adaptive_grm_block_rows(
        block_cols.max(1),
        m.max(1),
        n,
        if full_sample_fast { 0 } else { n_samples },
        total_threads,
    )
    .max(1);
    let cblas_copy_rhs = std::env::var("JX_GRM_PACKED_CBLAS_COPY_RHS")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("1") || s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let cblas_force_tmp_accum = std::env::var("JX_GRM_PACKED_CBLAS_BETA0")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);

    let mut grm = vec![0.0_f64; n * n];
    let mut block = vec![0.0_f64; row_step * n];
    let mut block_varsum = vec![0.0_f64; row_step];
    let mut varsum_acc = varsum_full;
    let mut is_first_block = true;
    let eps = 1e-12_f64;
    let byte_lut = packed_byte_lut();

    for row_start in (0..m).step_by(row_step) {
        let row_end = (row_start + row_step).min(m);
        let cur_rows = row_end.saturating_sub(row_start);
        if cur_rows == 0 {
            continue;
        }

        decode_grm_block_f64(
            packed_flat,
            bytes_per_snp,
            n_samples,
            row_flip,
            row_maf,
            sample_idx,
            full_sample_fast,
            method,
            eps,
            &byte_lut.code4,
            row_start,
            row_end,
            n,
            &mut block,
            &mut block_varsum,
            pool.as_ref(),
        )?;

        grm_rankk_update_f64(
            &mut grm,
            &block[..cur_rows * n],
            cur_rows,
            n,
            cblas_copy_rhs,
            is_first_block,
            cblas_force_tmp_accum,
        )?;
        is_first_block = false;
        if method == 1 && !full_sample_fast {
            varsum_acc += block_varsum[..cur_rows].iter().sum::<f64>();
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
            let idx_lo = i * n + j;
            let v = grm[idx_lo] * inv_scale;
            grm[idx_lo] = v;
            grm[j * n + i] = v;
        }
    }

    let varsum_ret = if method == 1 { varsum_acc } else { m as f64 };
    Ok((grm, varsum_ret))
}

#[inline]
fn decode_grm_stream_block_into(
    it: &mut BedSnpIter,
    out_block: &mut [f32],
    row_step: usize,
    n_samples: usize,
    method: usize,
    maf_thr: f32,
    miss_thr: f32,
    eps: f64,
) -> (usize, usize, f64, bool) {
    let mut cur_rows = 0usize;
    let mut scanned_add = 0usize;
    let mut varsum_add = 0.0_f64;
    let mut reached_eof = false;
    while cur_rows < row_step {
        let dst = &mut out_block[cur_rows * n_samples..(cur_rows + 1) * n_samples];
        let (scanned_now, maybe_var, eof_now) =
            it.next_snp_grm_row_into(dst, method, maf_thr, miss_thr, eps);
        scanned_add = scanned_add.saturating_add(scanned_now);
        let Some(var) = maybe_var else {
            if eof_now {
                reached_eof = true;
                break;
            }
            continue;
        };
        if method == 1 {
            varsum_add += var;
        }
        cur_rows += 1;
    }
    (cur_rows, scanned_add, varsum_add, reached_eof)
}

#[inline]
fn decode_grm_stream_block_parallel_into(
    it: &mut BedSnpIter,
    out_block: &mut [f32],
    row_step: usize,
    n_samples: usize,
    method: usize,
    maf_thr: f32,
    miss_thr: f32,
    eps: f64,
    decode_batch_snps: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> (usize, usize, f64, bool) {
    let mut cur_rows = 0usize;
    let mut scanned_add = 0usize;
    let mut varsum_add = 0.0_f64;
    let mut reached_eof = false;
    let n_total = it.n_snps();
    let batch_cap = decode_batch_snps.max(1);
    let mut keep_flags = vec![0_u8; batch_cap];
    let mut vars = vec![0.0_f64; batch_cap];

    while cur_rows < row_step {
        let base = it.cursor();
        if base >= n_total {
            reached_eof = true;
            break;
        }
        let need_rows = row_step - cur_rows;
        let scan_rows = (n_total - base).min(batch_cap).min(need_rows);
        if scan_rows == 0 {
            reached_eof = true;
            break;
        }

        keep_flags[..scan_rows].fill(0);
        if method == 1 {
            vars[..scan_rows].fill(0.0_f64);
        }
        let row_base = cur_rows;
        let scan_slice = &mut out_block[row_base * n_samples..(row_base + scan_rows) * n_samples];

        let mut run = || {
            let it_ref: &BedSnpIter = &*it;
            scan_slice
                .par_chunks_mut(n_samples)
                .zip(keep_flags[..scan_rows].par_iter_mut())
                .zip(vars[..scan_rows].par_iter_mut())
                .enumerate()
                .for_each(|(off, ((row_dst, keep_dst), var_dst))| {
                    let snp_idx = base + off;
                    let Some((alt_sum, non_missing)) =
                        it_ref.decode_snp_raw_into_with_stats_at(snp_idx, row_dst)
                    else {
                        return;
                    };
                    let Some((prep, var)) = grm_stream_row_prepare_from_stats(
                        method,
                        maf_thr,
                        miss_thr,
                        eps,
                        n_samples,
                        alt_sum,
                        non_missing,
                    ) else {
                        return;
                    };
                    apply_grm_stream_prepared_row_inplace(row_dst, prep);
                    *keep_dst = 1;
                    if method == 1 {
                        *var_dst = var;
                    }
                });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }

        let mut kept_rows = 0usize;
        for read in 0..scan_rows {
            if keep_flags[read] == 0 {
                continue;
            }
            if kept_rows != read {
                let src0 = read * n_samples;
                let dst0 = kept_rows * n_samples;
                scan_slice.copy_within(src0..src0 + n_samples, dst0);
            }
            if method == 1 {
                varsum_add += vars[read];
            }
            kept_rows += 1;
        }
        cur_rows += kept_rows;

        scanned_add = scanned_add.saturating_add(scan_rows);
        it.set_cursor(base + scan_rows);
        if base + scan_rows >= n_total {
            reached_eof = true;
            break;
        }
    }

    (cur_rows, scanned_add, varsum_add, reached_eof)
}

#[inline]
fn decode_grm_stream_block_dispatch(
    it: &mut BedSnpIter,
    out_block: &mut [f32],
    row_step: usize,
    n_samples: usize,
    method: usize,
    maf_thr: f32,
    miss_thr: f32,
    eps: f64,
    parallel_decode: bool,
    decode_batch_snps: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> (usize, usize, f64, bool) {
    if parallel_decode {
        decode_grm_stream_block_parallel_into(
            it,
            out_block,
            row_step,
            n_samples,
            method,
            maf_thr,
            miss_thr,
            eps,
            decode_batch_snps,
            pool,
        )
    } else {
        decode_grm_stream_block_into(
            it, out_block, row_step, n_samples, method, maf_thr, miss_thr, eps,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrmAccumMode {
    Auto,
    Syrk,
    Gemm,
}

#[derive(Clone, Copy, Debug)]
struct GrmStreamRowPrepared {
    mean_g: f32,
    std_scale: f32,
    flip_major: bool,
}

#[derive(Debug)]
struct GrmStreamPreStats {
    keep_indices: Vec<usize>,
    prepared_rows: Vec<GrmStreamRowPrepared>,
    eff_m: usize,
    varsum: f64,
}

#[inline]
fn parse_grm_accum_mode(raw: &str) -> Result<GrmAccumMode, String> {
    let m = raw.trim().to_ascii_lowercase();
    match m.as_str() {
        "" | "auto" => Ok(GrmAccumMode::Auto),
        "syrk" => Ok(GrmAccumMode::Syrk),
        "gemm" => Ok(GrmAccumMode::Gemm),
        _ => Err(format!(
            "unsupported accumulation mode='{raw}', expected one of: auto, syrk, gemm"
        )),
    }
}

#[inline]
fn grm_accum_mode_from_env(preferred_var: &str) -> Result<GrmAccumMode, String> {
    if let Ok(raw) = std::env::var(preferred_var) {
        return parse_grm_accum_mode(&raw);
    }
    if let Ok(raw) = std::env::var("JX_GRM_ACCUM") {
        return parse_grm_accum_mode(&raw);
    }
    Ok(GrmAccumMode::Auto)
}

#[inline]
fn grm_accum_mode_name(mode: GrmAccumMode) -> &'static str {
    match mode {
        GrmAccumMode::Auto => "auto",
        GrmAccumMode::Syrk => "syrk",
        GrmAccumMode::Gemm => "gemm",
    }
}

#[inline]
fn grm_stream_row_prepare_from_stats(
    method: usize,
    maf_thr: f32,
    miss_thr: f32,
    eps: f64,
    n_samples: usize,
    mut alt_sum: f64,
    non_missing: usize,
) -> Option<(GrmStreamRowPrepared, f64)> {
    if n_samples == 0 {
        return None;
    }
    let missing_rate = 1.0_f64 - (non_missing as f64 / n_samples as f64);
    if missing_rate > miss_thr as f64 {
        return None;
    }
    if non_missing == 0 {
        if maf_thr > 0.0_f32 {
            return None;
        }
        let prep = GrmStreamRowPrepared {
            mean_g: 0.0_f32,
            std_scale: if method == 2 { 0.0_f32 } else { 1.0_f32 },
            flip_major: false,
        };
        return Some((prep, 0.0_f64));
    }

    let mut alt_freq = alt_sum / (2.0_f64 * non_missing as f64);
    let flip_major = alt_freq > 0.5_f64;
    if flip_major {
        alt_sum = 2.0_f64 * non_missing as f64 - alt_sum;
        alt_freq = alt_sum / (2.0_f64 * non_missing as f64);
    }
    let maf = alt_freq.min(1.0_f64 - alt_freq);
    if maf < maf_thr as f64 {
        return None;
    }

    let mean_g = (alt_sum / non_missing as f64) as f32;
    let var = (2.0_f64 * alt_freq * (1.0_f64 - alt_freq)).max(0.0_f64);
    let std_scale = if method == 2 {
        if var > eps {
            (1.0_f64 / var.sqrt()) as f32
        } else {
            0.0_f32
        }
    } else {
        1.0_f32
    };

    let prep = GrmStreamRowPrepared {
        mean_g,
        std_scale,
        flip_major,
    };
    Some((prep, var))
}

#[inline]
fn apply_grm_stream_prepared_row_inplace(row: &mut [f32], prep: GrmStreamRowPrepared) {
    for g in row.iter_mut() {
        if *g < 0.0_f32 {
            *g = 0.0_f32;
            continue;
        }
        let gv = if prep.flip_major { 2.0_f32 - *g } else { *g };
        *g = (gv - prep.mean_g) * prep.std_scale;
    }
}

#[inline]
fn decode_grm_stream_block_prepared_into(
    it: &BedSnpIter,
    out_block: &mut [f32],
    row_step: usize,
    n_samples: usize,
    keep_indices: &[usize],
    prepared_rows: &[GrmStreamRowPrepared],
    prepared_cursor: &mut usize,
    raw_scanned_cursor: &mut usize,
    n_total: usize,
    parallel_decode: bool,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> (usize, usize, bool) {
    let remain = keep_indices.len().saturating_sub(*prepared_cursor);
    if remain == 0 {
        let scanned_add = n_total.saturating_sub(*raw_scanned_cursor);
        *raw_scanned_cursor = n_total;
        return (0, scanned_add, true);
    }

    let rows = remain.min(row_step);
    let start = *prepared_cursor;
    let end = start + rows;
    let idx_slice = &keep_indices[start..end];
    let prep_slice = &prepared_rows[start..end];
    let out_slice = &mut out_block[..rows * n_samples];
    let mut run = || {
        out_slice
            .par_chunks_mut(n_samples)
            .zip(idx_slice.par_iter())
            .zip(prep_slice.par_iter())
            .for_each(|((dst, &snp_idx), &prep)| {
                it.decode_snp_raw_into_with_stats_at(snp_idx, dst)
                    .expect("pre-stat keep index out of BED range");
                apply_grm_stream_prepared_row_inplace(dst, prep);
            });
    };
    if parallel_decode {
        if let Some(tp) = pool {
            tp.install(&mut run);
        } else {
            run();
        }
    } else {
        for off in 0..rows {
            let snp_idx = idx_slice[off];
            let prep = prep_slice[off];
            let dst = &mut out_slice[off * n_samples..(off + 1) * n_samples];
            it.decode_snp_raw_into_with_stats_at(snp_idx, dst)
                .expect("pre-stat keep index out of BED range");
            apply_grm_stream_prepared_row_inplace(dst, prep);
        }
    }
    *prepared_cursor = end;
    let reached_eof = end >= keep_indices.len();
    let mut raw_target = keep_indices[end - 1].saturating_add(1).min(n_total);
    if reached_eof {
        raw_target = n_total;
    }
    let scanned_add = raw_target.saturating_sub(*raw_scanned_cursor);
    *raw_scanned_cursor = raw_target;
    (rows, scanned_add, reached_eof)
}

#[inline]
fn grm_progress_map_ratio(done: usize, total: usize, cap: usize) -> usize {
    if total == 0 || cap == 0 {
        return 0;
    }
    let d = done.min(total) as u128;
    let t = total as u128;
    ((d * cap as u128) / t) as usize
}

#[inline]
fn grm_progress_map_between(done: usize, total: usize, lo: usize, hi: usize) -> usize {
    if hi <= lo {
        return lo;
    }
    lo.saturating_add(grm_progress_map_ratio(done, total, hi - lo))
}

#[inline]
fn grm_progress_notify(
    progress_callback: Option<&Py<PyAny>>,
    done: usize,
    total: usize,
    notify_step: usize,
    last_notified: &mut usize,
    force: bool,
) -> Result<(), String> {
    let done_clamped = done.min(total);
    if !force && done_clamped < last_notified.saturating_add(notify_step) {
        return Ok(());
    }
    *last_notified = done_clamped;
    if let Some(cb) = progress_callback {
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (done_clamped, total))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    } else {
        check_ctrlc()?;
    }
    Ok(())
}

fn write_npy_f32_matrix(path: &str, data: &[f32], rows: usize, cols: usize) -> Result<(), String> {
    let expected = rows.saturating_mul(cols);
    if data.len() != expected {
        return Err(format!(
            "write_npy_f32_matrix: data length mismatch, got {}, expected {}",
            data.len(),
            expected
        ));
    }
    let f = File::create(path).map_err(|e| format!("create npy file failed: {e}"))?;
    let mut w = BufWriter::new(f);

    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    while (10 + header.len() + 1) % 16 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let header_len_u16 = u16::try_from(header.len())
        .map_err(|_| "npy header too long for v1.0 format".to_string())?;

    w.write_all(b"\x93NUMPY")
        .map_err(|e| format!("write npy magic failed: {e}"))?;
    w.write_all(&[1_u8, 0_u8])
        .map_err(|e| format!("write npy version failed: {e}"))?;
    w.write_all(&header_len_u16.to_le_bytes())
        .map_err(|e| format!("write npy header length failed: {e}"))?;
    w.write_all(header.as_bytes())
        .map_err(|e| format!("write npy header failed: {e}"))?;

    #[cfg(target_endian = "little")]
    {
        let byte_len = data.len().saturating_mul(std::mem::size_of::<f32>());
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        w.write_all(bytes)
            .map_err(|e| format!("write npy payload failed: {e}"))?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &v in data.iter() {
            w.write_all(&v.to_le_bytes())
                .map_err(|e| format!("write npy payload failed: {e}"))?;
        }
    }
    w.flush()
        .map_err(|e| format!("flush npy file failed: {e}"))?;
    Ok(())
}

#[inline]
fn grm_stream_scan_prestats(
    it: &mut BedSnpIter,
    n_total: usize,
    n_samples: usize,
    method: usize,
    maf_thr: f32,
    miss_thr: f32,
    eps: f64,
    progress_callback: Option<&Py<PyAny>>,
    notify_step: usize,
    last_notified: &mut usize,
    phase1_cap: usize,
) -> Result<GrmStreamPreStats, String> {
    let mut keep_indices: Vec<usize> = Vec::new();
    let mut prepared_rows: Vec<GrmStreamRowPrepared> = Vec::new();
    let mut eff_m = 0usize;
    let mut varsum = 0.0_f64;
    let mut scanned = 0usize;

    while let Some((snp_idx, alt_sum, non_missing)) = it.next_snp_stats_only() {
        scanned = scanned.saturating_add(1);
        if (scanned & 0x3FFF) == 0 {
            check_ctrlc()?;
        }
        let done_mapped = grm_progress_map_ratio(scanned, n_total, phase1_cap);
        grm_progress_notify(
            progress_callback,
            done_mapped,
            n_total,
            notify_step,
            last_notified,
            false,
        )?;
        if let Some((prep, var)) = grm_stream_row_prepare_from_stats(
            method,
            maf_thr,
            miss_thr,
            eps,
            n_samples,
            alt_sum,
            non_missing,
        ) {
            keep_indices.push(snp_idx);
            prepared_rows.push(prep);
            eff_m = eff_m.saturating_add(1);
            if method == 1 {
                varsum += var;
            }
        }
    }
    // Pre-stat is phase-1; keep headroom for compute/finalize stages.
    grm_progress_notify(
        progress_callback,
        phase1_cap,
        n_total,
        notify_step,
        last_notified,
        true,
    )?;

    Ok(GrmStreamPreStats {
        keep_indices,
        prepared_rows,
        eff_m,
        varsum,
    })
}

#[inline]
fn effective_threads(threads: usize) -> usize {
    if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
    }
}

#[inline]
fn grm_stream_block_rows(requested_block_rows: usize, m: usize, n_samples: usize) -> usize {
    let exact_mode = std::env::var("JX_GRM_STREAM_BLOCK_EXACT")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);
    if exact_mode {
        return requested_block_rows.max(1).min(m.max(1));
    }
    if let Ok(raw) = std::env::var("JX_GRM_STREAM_BLOCK_ROWS") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            if v > 0 {
                return v.min(m.max(1));
            }
        }
    }
    let base = requested_block_rows.max(1).min(m.max(1));
    let target_mb = std::env::var("JX_GRM_STREAM_BLOCK_TARGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1024.0_f64);
    let target_bytes = (target_mb * 1024.0_f64 * 1024.0_f64) as usize;
    if target_bytes == 0 {
        return base;
    }
    let row_bytes = n_samples.saturating_mul(std::mem::size_of::<f32>()).max(1);
    let cap_rows = target_bytes
        .saturating_div(row_bytes.saturating_mul(2).max(1))
        .max(1);
    let max_rows = std::env::var("JX_GRM_STREAM_BLOCK_MAX_ROWS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(65_536usize);
    base.min(cap_rows).min(max_rows).max(1).min(m.max(1))
}

#[inline]
fn grm_stream_decode_batch_rows(row_step: usize, n_samples: usize) -> usize {
    if let Ok(raw) = std::env::var("JX_GRM_STREAM_DECODE_BATCH_ROWS") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            if v > 0 {
                return v.min(row_step.max(1));
            }
        }
    }
    let target_mb = std::env::var("JX_GRM_STREAM_DECODE_BATCH_MB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(32.0_f64);
    let target_bytes = (target_mb * 1024.0_f64 * 1024.0_f64) as usize;
    let row_bytes = n_samples.saturating_mul(std::mem::size_of::<f32>()).max(1);
    let by_mem = target_bytes.saturating_div(row_bytes).max(1);
    by_mem.max(64).min(4096).min(row_step.max(1))
}

#[inline]
fn grm_decode_threads(total_threads: usize, overlap_enabled: bool) -> usize {
    if !overlap_enabled {
        return 1;
    }
    if let Ok(raw) = std::env::var("JX_GRM_DECODE_THREADS") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            return v.max(1).min(total_threads.max(1));
        }
    }
    if total_threads >= 8 {
        (total_threads / 4).max(2).min(4)
    } else if total_threads >= 4 {
        2
    } else {
        1
    }
}

#[inline]
fn grm_blas_threads(total_threads: usize, decode_threads: usize, overlap_enabled: bool) -> usize {
    if !overlap_enabled {
        return total_threads.max(1);
    }
    total_threads.saturating_sub(decode_threads).max(1)
}

#[inline]
fn grm_overlap_enabled_default() -> bool {
    static OV: OnceLock<bool> = OnceLock::new();
    *OV.get_or_init(|| {
        let raw = std::env::var("JX_GRM_STREAM_OVERLAP")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        !matches!(raw.as_str(), "0" | "false" | "no" | "off")
    })
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn grm_probe_accum_mode_f32(block: &[f32], rows: usize, n: usize) -> GrmAccumMode {
    if rows == 0 || n == 0 {
        return GrmAccumMode::Syrk;
    }
    let gemm_min_n = std::env::var("JX_GRM_ACCUM_GEMM_MIN_N")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1024usize);
    if n < gemm_min_n {
        return GrmAccumMode::Syrk;
    }

    let probe_n = std::env::var("JX_GRM_ACCUM_PROBE_N")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(n)
        .min(n);
    let probe_k = std::env::var("JX_GRM_ACCUM_PROBE_K")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(rows.max(1))
        .min(rows.max(1));
    if probe_n == 0 || probe_k == 0 {
        return GrmAccumMode::Syrk;
    }

    let mut probe_a = vec![0.0_f32; probe_k * probe_n];
    for r in 0..probe_k {
        let src = &block[r * n..r * n + probe_n];
        let dst = &mut probe_a[r * probe_n..(r + 1) * probe_n];
        dst.copy_from_slice(src);
    }

    let reps = std::env::var("JX_GRM_ACCUM_PROBE_REPEATS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(2usize);
    let bench = |mode: GrmAccumMode| -> f64 {
        let mut best = f64::INFINITY;
        let mut c = vec![0.0_f32; probe_n * probe_n];
        for _ in 0..reps {
            let t0 = Instant::now();
            unsafe {
                match mode {
                    GrmAccumMode::Syrk => {
                        cblas_ssyrk_dispatch(
                            CBLAS_COL_MAJOR,
                            CBLAS_UPPER,
                            CBLAS_NO_TRANS,
                            probe_n as CblasInt,
                            probe_k as CblasInt,
                            1.0,
                            probe_a.as_ptr(),
                            probe_n as CblasInt,
                            0.0,
                            c.as_mut_ptr(),
                            probe_n as CblasInt,
                        );
                    }
                    GrmAccumMode::Gemm => {
                        cblas_sgemm_dispatch(
                            CBLAS_COL_MAJOR,
                            CBLAS_NO_TRANS,
                            CBLAS_TRANS,
                            probe_n as CblasInt,
                            probe_n as CblasInt,
                            probe_k as CblasInt,
                            1.0,
                            probe_a.as_ptr(),
                            probe_n as CblasInt,
                            probe_a.as_ptr(),
                            probe_n as CblasInt,
                            0.0,
                            c.as_mut_ptr(),
                            probe_n as CblasInt,
                        );
                    }
                    GrmAccumMode::Auto => {}
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            if dt < best {
                best = dt;
            }
        }
        best
    };

    let syrk_secs = bench(GrmAccumMode::Syrk);
    let gemm_secs = bench(GrmAccumMode::Gemm);
    if gemm_secs > 0.0 && gemm_secs < (syrk_secs * 0.90) {
        GrmAccumMode::Gemm
    } else {
        GrmAccumMode::Syrk
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
#[inline]
fn grm_probe_accum_mode_f32(_block: &[f32], _rows: usize, _n: usize) -> GrmAccumMode {
    GrmAccumMode::Syrk
}

#[inline]
fn resolve_grm_accum_mode_f32(
    mode: GrmAccumMode,
    block: &[f32],
    rows: usize,
    n: usize,
) -> GrmAccumMode {
    if mode == GrmAccumMode::Auto {
        grm_probe_accum_mode_f32(block, rows, n)
    } else {
        mode
    }
}

#[inline]
unsafe fn grm_scale_and_symmetrize_raw(grm_ptr: *mut f32, n: usize, inv_scale: f32) {
    for i in 0..n {
        let ii = i * n + i;
        let d = *grm_ptr.add(ii) * inv_scale;
        std::ptr::write(grm_ptr.add(ii), d);
        for j in 0..i {
            let idx_lo = i * n + j;
            let v = *grm_ptr.add(idx_lo) * inv_scale;
            std::ptr::write(grm_ptr.add(idx_lo), v);
            let idx_up = j * n + i;
            std::ptr::write(grm_ptr.add(idx_up), v);
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn grm_rankk_update_cblas(
    grm: &mut [f32],
    block: &[f32],
    cur_rows: usize,
    n: usize,
    copy_rhs: bool,
    is_first_block: bool,
    force_tmp_accum: bool,
) {
    let _ = copy_rhs;
    let beta = if is_first_block { 0.0_f32 } else { 1.0_f32 };
    if force_tmp_accum {
        let mut tmp = vec![0.0_f32; n * n];
        unsafe {
            cblas_ssyrk_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_UPPER,
                CBLAS_NO_TRANS,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                0.0,
                tmp.as_mut_ptr(),
                n as CblasInt,
            );
        }
        for col in 0..n {
            for row in 0..=col {
                let idx = row + col * n;
                grm[idx] += tmp[idx];
            }
        }
    } else {
        unsafe {
            cblas_ssyrk_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_UPPER,
                CBLAS_NO_TRANS,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                beta,
                grm.as_mut_ptr(),
                n as CblasInt,
            );
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn grm_rankk_update_cblas_raw(
    grm_ptr: *mut f32,
    block: &[f32],
    cur_rows: usize,
    n: usize,
    copy_rhs: bool,
    is_first_block: bool,
    force_tmp_accum: bool,
) {
    let _ = copy_rhs;
    let beta = if is_first_block { 0.0_f32 } else { 1.0_f32 };
    if force_tmp_accum {
        let mut tmp = vec![0.0_f32; n * n];
        cblas_ssyrk_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_UPPER,
            CBLAS_NO_TRANS,
            n as CblasInt,
            cur_rows as CblasInt,
            1.0,
            block.as_ptr(),
            n as CblasInt,
            0.0,
            tmp.as_mut_ptr(),
            n as CblasInt,
        );
        for col in 0..n {
            for row in 0..=col {
                let idx = row + col * n;
                let dst = grm_ptr.add(idx);
                if is_first_block {
                    std::ptr::write(dst, tmp[idx]);
                } else {
                    std::ptr::write(dst, *dst + tmp[idx]);
                }
            }
        }
    } else {
        cblas_ssyrk_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_UPPER,
            CBLAS_NO_TRANS,
            n as CblasInt,
            cur_rows as CblasInt,
            1.0,
            block.as_ptr(),
            n as CblasInt,
            beta,
            grm_ptr,
            n as CblasInt,
        );
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn grm_rankk_update_cblas_raw_gemm(
    grm_ptr: *mut f32,
    block: &[f32],
    cur_rows: usize,
    n: usize,
    is_first_block: bool,
    force_tmp_accum: bool,
) {
    let beta = if is_first_block { 0.0_f32 } else { 1.0_f32 };
    if force_tmp_accum {
        let mut tmp = vec![0.0_f32; n * n];
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            n as CblasInt,
            n as CblasInt,
            cur_rows as CblasInt,
            1.0,
            block.as_ptr(),
            n as CblasInt,
            block.as_ptr(),
            n as CblasInt,
            0.0,
            tmp.as_mut_ptr(),
            n as CblasInt,
        );
        for i in 0..n {
            for j in 0..=i {
                let idx = i * n + j;
                let dst = grm_ptr.add(idx);
                if is_first_block {
                    std::ptr::write(dst, tmp[idx]);
                } else {
                    std::ptr::write(dst, *dst + tmp[idx]);
                }
            }
        }
    } else {
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            n as CblasInt,
            n as CblasInt,
            cur_rows as CblasInt,
            1.0,
            block.as_ptr(),
            n as CblasInt,
            block.as_ptr(),
            n as CblasInt,
            beta,
            grm_ptr,
            n as CblasInt,
        );
    }
}

#[inline]
fn grm_rankk_update_raw(
    grm_ptr: *mut f32,
    block: &[f32],
    cur_rows: usize,
    n: usize,
    accum_mode: GrmAccumMode,
    cblas_copy_rhs: bool,
    is_first_block: bool,
    cblas_force_tmp_accum: bool,
) -> Result<(), String> {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        unsafe {
            match accum_mode {
                GrmAccumMode::Auto | GrmAccumMode::Syrk => grm_rankk_update_cblas_raw(
                    grm_ptr,
                    block,
                    cur_rows,
                    n,
                    cblas_copy_rhs,
                    is_first_block,
                    cblas_force_tmp_accum,
                ),
                GrmAccumMode::Gemm => grm_rankk_update_cblas_raw_gemm(
                    grm_ptr,
                    block,
                    cur_rows,
                    n,
                    is_first_block,
                    cblas_force_tmp_accum,
                ),
            }
        }
        Ok(())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = (
            grm_ptr,
            block,
            cur_rows,
            n,
            accum_mode,
            cblas_copy_rhs,
            is_first_block,
            cblas_force_tmp_accum,
        );
        Err(
            "Packed GRM requires CBLAS backend on this platform; fallback to streaming GRM."
                .to_string(),
        )
    }
}

#[inline]
pub(crate) fn grm_rankk_update(
    grm: &mut [f32],
    block: &[f32],
    cur_rows: usize,
    n: usize,
    cblas_copy_rhs: bool,
    is_first_block: bool,
    cblas_force_tmp_accum: bool,
) -> Result<(), String> {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        grm_rankk_update_cblas(
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            is_first_block,
            cblas_force_tmp_accum,
        );
        Ok(())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = (
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            is_first_block,
            cblas_force_tmp_accum,
        );
        Err(
            "Packed GRM requires CBLAS backend on this platform; fallback to streaming GRM."
                .to_string(),
        )
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn grm_rankk_update_cblas_f64(
    grm: &mut [f64],
    block: &[f64],
    cur_rows: usize,
    n: usize,
    copy_rhs: bool,
    is_first_block: bool,
    force_tmp_accum: bool,
) {
    let _ = copy_rhs;
    let beta = if is_first_block { 0.0_f64 } else { 1.0_f64 };
    if force_tmp_accum {
        let mut tmp = vec![0.0_f64; n * n];
        unsafe {
            cblas_dsyrk_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_UPPER,
                CBLAS_NO_TRANS,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                0.0,
                tmp.as_mut_ptr(),
                n as CblasInt,
            );
        }
        for col in 0..n {
            for row in 0..=col {
                let idx = row + col * n;
                grm[idx] += tmp[idx];
            }
        }
    } else {
        unsafe {
            cblas_dsyrk_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_UPPER,
                CBLAS_NO_TRANS,
                n as CblasInt,
                cur_rows as CblasInt,
                1.0,
                block.as_ptr(),
                n as CblasInt,
                beta,
                grm.as_mut_ptr(),
                n as CblasInt,
            );
        }
    }
}

#[inline]
pub(crate) fn grm_rankk_update_f64(
    grm: &mut [f64],
    block: &[f64],
    cur_rows: usize,
    n: usize,
    cblas_copy_rhs: bool,
    is_first_block: bool,
    cblas_force_tmp_accum: bool,
) -> Result<(), String> {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        grm_rankk_update_cblas_f64(
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            is_first_block,
            cblas_force_tmp_accum,
        );
        Ok(())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = (
            grm,
            block,
            cur_rows,
            n,
            cblas_copy_rhs,
            is_first_block,
            cblas_force_tmp_accum,
        );
        Err(
            "Packed GRM requires CBLAS backend on this platform; fallback to streaming GRM."
                .to_string(),
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    method=1,
    block_cols=65536,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_f32<'py>(
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
) -> PyResult<Bound<'py, PyArray2<f32>>> {
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
    let total_threads = effective_threads(threads);
    let overlap_enabled = grm_overlap_enabled_default();
    let decode_threads = grm_decode_threads(total_threads, overlap_enabled);
    let blas_threads = grm_blas_threads(total_threads, decode_threads, overlap_enabled);
    let pool = get_cached_pool(decode_threads)?;
    let row_step = adaptive_grm_block_rows(
        block_cols.max(1),
        m.max(1),
        n,
        if full_sample_fast { 0 } else { n_samples },
        total_threads,
    );
    let block_target_mb = std::env::var("JX_GRM_PACKED_BLOCK_TARGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(64.0_f64);
    let stage_timing = env_truthy("JX_GRM_PACKED_STAGE_TIMING");
    let notify_step = if progress_every == 0 {
        row_step
    } else {
        progress_every.max(1)
    };
    let cblas_copy_rhs = std::env::var("JX_GRM_PACKED_CBLAS_COPY_RHS")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("1") || s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let cblas_force_tmp_accum = std::env::var("JX_GRM_PACKED_CBLAS_BETA0")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);
    let accum_mode =
        grm_accum_mode_from_env("JX_GRM_PACKED_ACCUM").map_err(PyRuntimeError::new_err)?;

    let grm_vec = py
        .detach(move || -> Result<Vec<f32>, String> {
            let _blas_guard = OpenBlasThreadGuard::enter(blas_threads.max(1));
            let total_t0 = Instant::now();
            let mut decode_secs = 0.0_f64;
            let mut gemm_secs = 0.0_f64;
            let mut grm = Vec::<f32>::with_capacity(n * n);
            let grm_ptr = grm.as_mut_ptr();
            let mut varsum_acc = varsum_full;
            let eps = 1e-12_f32;
            let byte_lut = packed_byte_lut();

            let mut block_a = vec![0.0_f32; row_step * n];
            let mut block_b = vec![0.0_f32; row_step * n];
            let mut varsum_a = vec![0.0_f64; row_step];
            let mut varsum_b = vec![0.0_f64; row_step];
            let pipeline_overlap = overlap_enabled && (total_threads > 1) && (m > row_step);

            let mut cur_start = 0usize;
            let mut cur_end = row_step.min(m);
            let mut use_a = true;
            let mut last_notified = 0usize;
            let mut is_first_block = true;
            let mut accum_mode_runtime = accum_mode;

            let decode_t0 = Instant::now();
            decode_grm_block(
                packed_flat.as_ref(),
                bytes_per_snp,
                n_samples,
                row_flip_vec.as_ref(),
                row_maf_vec.as_ref(),
                &sample_idx,
                full_sample_fast,
                method,
                eps,
                &byte_lut.code4,
                cur_start,
                cur_end,
                n,
                &mut block_a,
                &mut varsum_a,
                pool.as_ref(),
            )?;
            decode_secs += decode_t0.elapsed().as_secs_f64();

            loop {
                let cur_rows = cur_end.saturating_sub(cur_start);
                if cur_rows == 0 {
                    break;
                }
                let next_start = cur_end;
                let next_end = (next_start + row_step).min(m);
                let has_next = next_start < m;

                if pipeline_overlap && has_next {
                    if use_a {
                        let (gemm_dt, decode_dt) =
                            std::thread::scope(|scope| -> Result<(f64, f64), String> {
                                let decode_handle = scope.spawn(|| {
                                    let dt0 = Instant::now();
                                    let r = decode_grm_block(
                                        packed_flat.as_ref(),
                                        bytes_per_snp,
                                        n_samples,
                                        row_flip_vec.as_ref(),
                                        row_maf_vec.as_ref(),
                                        &sample_idx,
                                        full_sample_fast,
                                        method,
                                        eps,
                                        &byte_lut.code4,
                                        next_start,
                                        next_end,
                                        n,
                                        &mut block_b,
                                        &mut varsum_b,
                                        pool.as_ref(),
                                    );
                                    (r, dt0.elapsed().as_secs_f64())
                                });

                                let gt0 = Instant::now();
                                if accum_mode_runtime == GrmAccumMode::Auto {
                                    accum_mode_runtime = resolve_grm_accum_mode_f32(
                                        accum_mode_runtime,
                                        &block_a[..cur_rows * n],
                                        cur_rows,
                                        n,
                                    );
                                }
                                let gemm_res = grm_rankk_update_raw(
                                    grm_ptr,
                                    &block_a[..cur_rows * n],
                                    cur_rows,
                                    n,
                                    accum_mode_runtime,
                                    cblas_copy_rhs,
                                    is_first_block,
                                    cblas_force_tmp_accum,
                                );
                                let gemm_dt = gt0.elapsed().as_secs_f64();

                                let (decode_res, decode_dt) = decode_handle
                                    .join()
                                    .map_err(|_| "decode worker thread panicked".to_string())?;
                                gemm_res?;
                                decode_res?;
                                Ok((gemm_dt, decode_dt))
                            })?;
                        gemm_secs += gemm_dt;
                        decode_secs += decode_dt;
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_a[..cur_rows].iter().sum::<f64>();
                        }
                    } else {
                        let (gemm_dt, decode_dt) =
                            std::thread::scope(|scope| -> Result<(f64, f64), String> {
                                let decode_handle = scope.spawn(|| {
                                    let dt0 = Instant::now();
                                    let r = decode_grm_block(
                                        packed_flat.as_ref(),
                                        bytes_per_snp,
                                        n_samples,
                                        row_flip_vec.as_ref(),
                                        row_maf_vec.as_ref(),
                                        &sample_idx,
                                        full_sample_fast,
                                        method,
                                        eps,
                                        &byte_lut.code4,
                                        next_start,
                                        next_end,
                                        n,
                                        &mut block_a,
                                        &mut varsum_a,
                                        pool.as_ref(),
                                    );
                                    (r, dt0.elapsed().as_secs_f64())
                                });

                                let gt0 = Instant::now();
                                if accum_mode_runtime == GrmAccumMode::Auto {
                                    accum_mode_runtime = resolve_grm_accum_mode_f32(
                                        accum_mode_runtime,
                                        &block_b[..cur_rows * n],
                                        cur_rows,
                                        n,
                                    );
                                }
                                let gemm_res = grm_rankk_update_raw(
                                    grm_ptr,
                                    &block_b[..cur_rows * n],
                                    cur_rows,
                                    n,
                                    accum_mode_runtime,
                                    cblas_copy_rhs,
                                    is_first_block,
                                    cblas_force_tmp_accum,
                                );
                                let gemm_dt = gt0.elapsed().as_secs_f64();

                                let (decode_res, decode_dt) = decode_handle
                                    .join()
                                    .map_err(|_| "decode worker thread panicked".to_string())?;
                                gemm_res?;
                                decode_res?;
                                Ok((gemm_dt, decode_dt))
                            })?;
                        gemm_secs += gemm_dt;
                        decode_secs += decode_dt;
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_b[..cur_rows].iter().sum::<f64>();
                        }
                    }
                } else {
                    if use_a {
                        let gemm_t0 = Instant::now();
                        if accum_mode_runtime == GrmAccumMode::Auto {
                            accum_mode_runtime = resolve_grm_accum_mode_f32(
                                accum_mode_runtime,
                                &block_a[..cur_rows * n],
                                cur_rows,
                                n,
                            );
                        }
                        grm_rankk_update_raw(
                            grm_ptr,
                            &block_a[..cur_rows * n],
                            cur_rows,
                            n,
                            accum_mode_runtime,
                            cblas_copy_rhs,
                            is_first_block,
                            cblas_force_tmp_accum,
                        )?;
                        gemm_secs += gemm_t0.elapsed().as_secs_f64();
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_a[..cur_rows].iter().sum::<f64>();
                        }
                        if has_next {
                            let decode_t0 = Instant::now();
                            decode_grm_block(
                                packed_flat.as_ref(),
                                bytes_per_snp,
                                n_samples,
                                row_flip_vec.as_ref(),
                                row_maf_vec.as_ref(),
                                &sample_idx,
                                full_sample_fast,
                                method,
                                eps,
                                &byte_lut.code4,
                                next_start,
                                next_end,
                                n,
                                &mut block_b,
                                &mut varsum_b,
                                pool.as_ref(),
                            )?;
                            decode_secs += decode_t0.elapsed().as_secs_f64();
                        }
                    } else {
                        let gemm_t0 = Instant::now();
                        if accum_mode_runtime == GrmAccumMode::Auto {
                            accum_mode_runtime = resolve_grm_accum_mode_f32(
                                accum_mode_runtime,
                                &block_b[..cur_rows * n],
                                cur_rows,
                                n,
                            );
                        }
                        grm_rankk_update_raw(
                            grm_ptr,
                            &block_b[..cur_rows * n],
                            cur_rows,
                            n,
                            accum_mode_runtime,
                            cblas_copy_rhs,
                            is_first_block,
                            cblas_force_tmp_accum,
                        )?;
                        gemm_secs += gemm_t0.elapsed().as_secs_f64();
                        if method == 1 && !full_sample_fast {
                            varsum_acc += varsum_b[..cur_rows].iter().sum::<f64>();
                        }
                        if has_next {
                            let decode_t0 = Instant::now();
                            decode_grm_block(
                                packed_flat.as_ref(),
                                bytes_per_snp,
                                n_samples,
                                row_flip_vec.as_ref(),
                                row_maf_vec.as_ref(),
                                &sample_idx,
                                full_sample_fast,
                                method,
                                eps,
                                &byte_lut.code4,
                                next_start,
                                next_end,
                                n,
                                &mut block_a,
                                &mut varsum_a,
                                pool.as_ref(),
                            )?;
                            decode_secs += decode_t0.elapsed().as_secs_f64();
                        }
                    }
                }
                is_first_block = false;

                let done = cur_end;
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

                if !has_next {
                    break;
                }
                cur_start = next_start;
                cur_end = next_end;
                use_a = !use_a;
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
            unsafe {
                grm_scale_and_symmetrize_raw(grm_ptr, n, inv_scale);
                grm.set_len(n * n);
            }
            if stage_timing {
                let total_secs = total_t0.elapsed().as_secs_f64();
                let stage_sum = decode_secs + gemm_secs;
                let overlap_secs = (stage_sum - total_secs).max(0.0_f64);
                let other_secs = if stage_sum <= total_secs {
                    total_secs - stage_sum
                } else {
                    0.0_f64
                };
                let to_pct = |x: f64| -> f64 {
                    if total_secs > 0.0 {
                        x * 100.0 / total_secs
                    } else {
                        0.0
                    }
                };
                eprintln!(
                    "GRM stage timing: decode={:.3}s ({:.1}%), gemm={:.3}s ({:.1}%), \
other={:.3}s ({:.1}%), overlap={:.3}s ({:.1}%), total={:.3}s, row_step={}, \
block_target_mb={:.1}, n_samples={}, full_sample={}, threads_total={}, decode_threads={}, \
blas_threads={}, overlap={}, accum={}",
                    decode_secs,
                    to_pct(decode_secs),
                    gemm_secs,
                    to_pct(gemm_secs),
                    other_secs,
                    to_pct(other_secs),
                    overlap_secs,
                    to_pct(overlap_secs),
                    total_secs,
                    row_step,
                    block_target_mb,
                    n,
                    if full_sample_fast { "yes" } else { "no" },
                    total_threads,
                    decode_threads,
                    blas_threads,
                    if pipeline_overlap { "on" } else { "off" },
                    grm_accum_mode_name(accum_mode_runtime),
                );
            }
            Ok(grm)
        })
        .map_err(map_err_string_to_py)?;

    let grm_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n, n), grm_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok(grm_arr)
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    method=1,
    maf_threshold=0.02,
    max_missing_rate=0.05,
    block_cols=65536,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_bed_f32<'py>(
    py: Python<'py>,
    prefix: String,
    method: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, usize, usize)> {
    let maf_thr = maf_threshold.clamp(0.0, 0.5);
    let miss_thr = max_missing_rate.clamp(0.0, 1.0);
    let (
        packed_arr,
        _miss_arr,
        maf_arr,
        _std_arr,
        row_flip_arr,
        _site_keep_arr,
        n_samples,
        _n_total_sites,
    ) = crate::gfreader::prepare_bed_2bit_packed(py, prefix, maf_thr, miss_thr, 0.0_f32, false)?;
    let packed_ro = packed_arr.readonly();
    let packed_view = packed_ro.as_array();
    if packed_view.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed BED payload must be 2D (m, bytes_per_snp).",
        ));
    }
    let eff_m = packed_view.shape()[0];
    if eff_m == 0 {
        return Err(PyRuntimeError::new_err(
            "No SNPs remained after packed BED filtering; GRM is empty.",
        ));
    }
    let maf_ro = maf_arr.readonly();
    let row_flip_ro = row_flip_arr.readonly();
    if maf_ro.len() != eff_m || row_flip_ro.len() != eff_m {
        return Err(PyRuntimeError::new_err(format!(
            "packed BED metadata length mismatch: packed_rows={eff_m}, maf={}, row_flip={}",
            maf_ro.len(),
            row_flip_ro.len()
        )));
    }

    let grm = grm_packed_f32(
        py,
        packed_ro,
        n_samples,
        row_flip_ro,
        maf_ro,
        None,
        method,
        block_cols,
        threads,
        progress_callback,
        progress_every,
    )?;
    Ok((grm, eff_m, n_samples))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    method=1,
    maf_threshold=0.02,
    max_missing_rate=0.05,
    block_cols=65536,
    threads=0,
    progress_callback=None,
    progress_every=0,
    mmap_window_mb=None
))]
pub fn grm_stream_bed_f32<'py>(
    py: Python<'py>,
    prefix: String,
    method: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, usize, usize)> {
    if method != 1 && method != 2 {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method={method}; expected 1 (centered) or 2 (standardized)"
        )));
    }
    let maf_thr = maf_threshold.clamp(0.0, 0.5);
    let miss_thr = max_missing_rate.clamp(0.0, 1.0);
    let two_stage_default_on = std::env::var("JX_GRM_STREAM_TWO_STAGE")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);
    // Two-stage path: BED read-time pre-stats/filtering + pure packed GRM compute loop.
    if two_stage_default_on && mmap_window_mb.is_none() {
        let (
            packed_arr,
            _miss_arr,
            maf_arr,
            _std_arr,
            row_flip_arr,
            _site_keep_arr,
            n_samples,
            _n_total_sites,
        ) = crate::gfreader::prepare_bed_2bit_packed(
            py,
            prefix.clone(),
            maf_thr,
            miss_thr,
            0.0_f32,
            false,
        )?;
        let packed_ro = packed_arr.readonly();
        let packed_view = packed_ro.as_array();
        if packed_view.ndim() != 2 {
            return Err(PyRuntimeError::new_err(
                "packed BED payload must be 2D (m, bytes_per_snp).",
            ));
        }
        let eff_m = packed_view.shape()[0];
        if eff_m == 0 {
            return Err(PyRuntimeError::new_err(
                "No SNPs remained after packed BED filtering; GRM is empty.",
            ));
        }
        let maf_ro = maf_arr.readonly();
        let row_flip_ro = row_flip_arr.readonly();
        if maf_ro.len() != eff_m || row_flip_ro.len() != eff_m {
            return Err(PyRuntimeError::new_err(format!(
                "packed BED metadata length mismatch: packed_rows={eff_m}, maf={}, row_flip={}",
                maf_ro.len(),
                row_flip_ro.len()
            )));
        }
        let grm = grm_packed_f32(
            py,
            packed_ro,
            n_samples,
            row_flip_ro,
            maf_ro,
            None,
            method,
            block_cols,
            threads,
            progress_callback,
            progress_every,
        )?;
        return Ok((grm, eff_m, n_samples));
    }

    let mut it = if let Some(window_mb) = mmap_window_mb {
        BedSnpIter::new_for_grm_window(&prefix, window_mb).map_err(PyRuntimeError::new_err)?
    } else {
        BedSnpIter::new_for_grm(&prefix).map_err(PyRuntimeError::new_err)?
    };
    let n_samples = it.n_samples();
    let n_total = it.n_snps();
    if n_samples == 0 {
        return Err(PyRuntimeError::new_err("BED sample count is zero."));
    }
    if n_total == 0 {
        return Err(PyRuntimeError::new_err("BED SNP count is zero."));
    }

    let total_threads = effective_threads(threads);
    let overlap_enabled = grm_overlap_enabled_default();
    let can_parallel_decode = !it.is_windowed();
    let decode_threads_hint = if can_parallel_decode {
        grm_decode_threads(total_threads, overlap_enabled)
    } else {
        1usize
    };
    let parallel_decode_pref =
        std::env::var("JX_GRM_STREAM_PAR_DECODE").unwrap_or_else(|_| "auto".to_string());
    let parallel_decode_pref_lc = parallel_decode_pref.trim().to_ascii_lowercase();
    let parallel_decode_min_samples = std::env::var("JX_GRM_STREAM_PAR_DECODE_MIN_SAMPLES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(2048usize);
    let parallel_decode = if !can_parallel_decode {
        false
    } else {
        match parallel_decode_pref_lc.as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => n_samples >= parallel_decode_min_samples,
        }
    };
    let decode_threads = if parallel_decode {
        decode_threads_hint.max(1)
    } else {
        1usize
    };
    let blas_threads = grm_blas_threads(total_threads, decode_threads, overlap_enabled);
    let decode_pool = if decode_threads > 1 {
        get_cached_pool(decode_threads)?
    } else {
        None
    };
    let row_step = grm_stream_block_rows(block_cols.max(1), n_total.max(1), n_samples);
    let decode_batch_rows = grm_stream_decode_batch_rows(row_step, n_samples);
    let notify_step = if progress_every == 0 {
        row_step
    } else {
        progress_every.max(1)
    };
    let cblas_copy_rhs = std::env::var("JX_GRM_PACKED_CBLAS_COPY_RHS")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("1") || s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let cblas_force_tmp_accum = std::env::var("JX_GRM_PACKED_CBLAS_BETA0")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);
    let accum_mode =
        grm_accum_mode_from_env("JX_GRM_STREAM_ACCUM").map_err(PyRuntimeError::new_err)?;
    let stream_prestat_core_requested = std::env::var("JX_GRM_STREAM_PRESTAT_CORE")
        .ok()
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            !matches!(t.as_str(), "0" | "false" | "no" | "off")
        })
        // Default to one-stage true streaming unless explicitly enabled.
        .unwrap_or(false);
    let stream_prestat_core = stream_prestat_core_requested && !it.is_windowed();
    let stage_timing = env_truthy("JX_GRM_STREAM_STAGE_TIMING");

    let grm_vec = py
        .detach(move || -> Result<(Vec<f32>, usize), String> {
            let _blas_guard = OpenBlasThreadGuard::enter(blas_threads.max(1));
            let total_t0 = Instant::now();
            let mut decode_secs = 0.0_f64;
            let mut gemm_secs = 0.0_f64;
            let mut grm = Vec::<f32>::with_capacity(n_samples * n_samples);
            let grm_ptr = grm.as_mut_ptr();
            let mut block_a = vec![0.0_f32; row_step * n_samples];
            let mut block_b = vec![0.0_f32; row_step * n_samples];
            let mut eff_m = 0usize;
            let mut scanned = 0usize;
            let mut last_notified = 0usize;
            let mut varsum_acc = 0.0_f64;
            let progress_pre_final_cap = n_total.saturating_sub(1);
            let progress_phase1_cap = if stream_prestat_core {
                progress_pre_final_cap / 2
            } else {
                0usize
            };
            let progress_phase2_cap = progress_pre_final_cap;
            let eps = 1e-12_f64;
            let pipeline_overlap = overlap_enabled && (total_threads > 1) && (n_total > row_step);
            let mut use_a = true;
            let mut is_first_block = true;
            let mut accum_mode_runtime = accum_mode;
            let mut prepared_cursor = 0usize;
            let mut prepared_raw_scanned = 0usize;
            let mut prestats: Option<GrmStreamPreStats> = None;

            let (mut cur_rows, mut cur_scanned, mut cur_varsum, mut cur_eof) = if stream_prestat_core {
                let prep_t0 = Instant::now();
                let ps = grm_stream_scan_prestats(
                    &mut it,
                    n_total,
                    n_samples,
                    method,
                    maf_thr,
                    miss_thr,
                    eps,
                    progress_callback.as_ref(),
                    notify_step,
                    &mut last_notified,
                    progress_phase1_cap,
                )?;
                decode_secs += prep_t0.elapsed().as_secs_f64();
                if ps.eff_m == 0 {
                    return Err("No SNPs remained after pre-stat filtering; GRM is empty.".to_string());
                }
                if method == 1 {
                    varsum_acc = ps.varsum;
                }

                let decode_t0 = Instant::now();
                let (rows, scanned_now, eof_now) = decode_grm_stream_block_prepared_into(
                    &it,
                    &mut block_a,
                    row_step,
                    n_samples,
                    &ps.keep_indices,
                    &ps.prepared_rows,
                    &mut prepared_cursor,
                    &mut prepared_raw_scanned,
                    n_total,
                    parallel_decode,
                    decode_pool.as_ref(),
                );
                decode_secs += decode_t0.elapsed().as_secs_f64();
                prestats = Some(ps);
                (rows, scanned_now, 0.0_f64, eof_now)
            } else {
                let decode_t0 = Instant::now();
                let out = decode_grm_stream_block_dispatch(
                    &mut it,
                    &mut block_a,
                    row_step,
                    n_samples,
                    method,
                    maf_thr,
                    miss_thr,
                    eps,
                    parallel_decode,
                    decode_batch_rows,
                    decode_pool.as_ref(),
                );
                decode_secs += decode_t0.elapsed().as_secs_f64();
                out
            };

            loop {
                check_ctrlc()?;

                if cur_rows == 0 {
                    scanned = scanned.saturating_add(cur_scanned);
                    let done_cb = if stream_prestat_core {
                        let eff_total = prestats
                            .as_ref()
                            .map(|ps| ps.eff_m)
                            .unwrap_or(1usize)
                            .max(1usize);
                        grm_progress_map_between(
                            eff_m.min(eff_total),
                            eff_total,
                            progress_phase1_cap,
                            progress_phase2_cap,
                        )
                    } else {
                        grm_progress_map_ratio(scanned, n_total, progress_phase2_cap)
                    };
                    grm_progress_notify(
                        progress_callback.as_ref(),
                        done_cb,
                        n_total,
                        notify_step,
                        &mut last_notified,
                        cur_eof,
                    )?;
                    if cur_eof {
                        break;
                    }
                    let decode_t0 = Instant::now();
                    let next = if stream_prestat_core {
                        let ps = prestats
                            .as_ref()
                            .ok_or_else(|| "missing pre-stat cache".to_string())?;
                        if use_a {
                            let (rows, scanned_now, eof_now) = decode_grm_stream_block_prepared_into(
                                &it,
                                &mut block_a,
                                row_step,
                                n_samples,
                                &ps.keep_indices,
                                &ps.prepared_rows,
                                &mut prepared_cursor,
                                &mut prepared_raw_scanned,
                                n_total,
                                parallel_decode,
                                decode_pool.as_ref(),
                            );
                            (rows, scanned_now, 0.0_f64, eof_now)
                        } else {
                            let (rows, scanned_now, eof_now) = decode_grm_stream_block_prepared_into(
                                &it,
                                &mut block_b,
                                row_step,
                                n_samples,
                                &ps.keep_indices,
                                &ps.prepared_rows,
                                &mut prepared_cursor,
                                &mut prepared_raw_scanned,
                                n_total,
                                parallel_decode,
                                decode_pool.as_ref(),
                            );
                            (rows, scanned_now, 0.0_f64, eof_now)
                        }
                    } else if use_a {
                        decode_grm_stream_block_dispatch(
                            &mut it,
                            &mut block_a,
                            row_step,
                            n_samples,
                            method,
                            maf_thr,
                            miss_thr,
                            eps,
                            parallel_decode,
                            decode_batch_rows,
                            decode_pool.as_ref(),
                        )
                    } else {
                        decode_grm_stream_block_dispatch(
                            &mut it,
                            &mut block_b,
                            row_step,
                            n_samples,
                            method,
                            maf_thr,
                            miss_thr,
                            eps,
                            parallel_decode,
                            decode_batch_rows,
                            decode_pool.as_ref(),
                        )
                    };
                    decode_secs += decode_t0.elapsed().as_secs_f64();
                    (cur_rows, cur_scanned, cur_varsum, cur_eof) = next;
                    continue;
                }

                let mut next_meta = (0usize, 0usize, 0.0_f64, true);
                if pipeline_overlap && !cur_eof {
                    if use_a {
                        let (gemm_dt, decode_dt, nxt) = std::thread::scope(
                            |scope| -> Result<(f64, f64, (usize, usize, f64, bool)), String> {
                                let decode_handle = scope.spawn(|| {
                                    let dt0 = Instant::now();
                                    let nxt = if stream_prestat_core {
                                        let ps =
                                            prestats.as_ref().expect("missing pre-stat cache");
                                        let (rows, scanned_now, eof_now) =
                                            decode_grm_stream_block_prepared_into(
                                                &it,
                                                &mut block_b,
                                                row_step,
                                                n_samples,
                                                &ps.keep_indices,
                                                &ps.prepared_rows,
                                                &mut prepared_cursor,
                                                &mut prepared_raw_scanned,
                                                n_total,
                                                parallel_decode,
                                                decode_pool.as_ref(),
                                            );
                                        (rows, scanned_now, 0.0_f64, eof_now)
                                    } else {
                                        decode_grm_stream_block_dispatch(
                                            &mut it,
                                            &mut block_b,
                                            row_step,
                                            n_samples,
                                            method,
                                            maf_thr,
                                            miss_thr,
                                            eps,
                                            parallel_decode,
                                            decode_batch_rows,
                                            decode_pool.as_ref(),
                                        )
                                    };
                                    (nxt, dt0.elapsed().as_secs_f64())
                                });
                                let gt0 = Instant::now();
                                if accum_mode_runtime == GrmAccumMode::Auto {
                                    accum_mode_runtime = resolve_grm_accum_mode_f32(
                                        accum_mode_runtime,
                                        &block_a[..cur_rows * n_samples],
                                        cur_rows,
                                        n_samples,
                                    );
                                }
                                grm_rankk_update_raw(
                                    grm_ptr,
                                    &block_a[..cur_rows * n_samples],
                                    cur_rows,
                                    n_samples,
                                    accum_mode_runtime,
                                    cblas_copy_rhs,
                                    is_first_block,
                                    cblas_force_tmp_accum,
                                )?;
                                let gemm_dt = gt0.elapsed().as_secs_f64();
                                let (decode_out, decode_dt) = decode_handle
                                    .join()
                                    .map_err(|_| "stream decode worker thread panicked".to_string())?;
                                Ok((gemm_dt, decode_dt, decode_out))
                            },
                        )?;
                        gemm_secs += gemm_dt;
                        decode_secs += decode_dt;
                        next_meta = nxt;
                    } else {
                        let (gemm_dt, decode_dt, nxt) = std::thread::scope(
                            |scope| -> Result<(f64, f64, (usize, usize, f64, bool)), String> {
                                let decode_handle = scope.spawn(|| {
                                    let dt0 = Instant::now();
                                    let nxt = if stream_prestat_core {
                                        let ps =
                                            prestats.as_ref().expect("missing pre-stat cache");
                                        let (rows, scanned_now, eof_now) =
                                            decode_grm_stream_block_prepared_into(
                                                &it,
                                                &mut block_a,
                                                row_step,
                                                n_samples,
                                                &ps.keep_indices,
                                                &ps.prepared_rows,
                                                &mut prepared_cursor,
                                                &mut prepared_raw_scanned,
                                                n_total,
                                                parallel_decode,
                                                decode_pool.as_ref(),
                                            );
                                        (rows, scanned_now, 0.0_f64, eof_now)
                                    } else {
                                        decode_grm_stream_block_dispatch(
                                            &mut it,
                                            &mut block_a,
                                            row_step,
                                            n_samples,
                                            method,
                                            maf_thr,
                                            miss_thr,
                                            eps,
                                            parallel_decode,
                                            decode_batch_rows,
                                            decode_pool.as_ref(),
                                        )
                                    };
                                    (nxt, dt0.elapsed().as_secs_f64())
                                });
                                let gt0 = Instant::now();
                                if accum_mode_runtime == GrmAccumMode::Auto {
                                    accum_mode_runtime = resolve_grm_accum_mode_f32(
                                        accum_mode_runtime,
                                        &block_b[..cur_rows * n_samples],
                                        cur_rows,
                                        n_samples,
                                    );
                                }
                                grm_rankk_update_raw(
                                    grm_ptr,
                                    &block_b[..cur_rows * n_samples],
                                    cur_rows,
                                    n_samples,
                                    accum_mode_runtime,
                                    cblas_copy_rhs,
                                    is_first_block,
                                    cblas_force_tmp_accum,
                                )?;
                                let gemm_dt = gt0.elapsed().as_secs_f64();
                                let (decode_out, decode_dt) = decode_handle
                                    .join()
                                    .map_err(|_| "stream decode worker thread panicked".to_string())?;
                                Ok((gemm_dt, decode_dt, decode_out))
                            },
                        )?;
                        gemm_secs += gemm_dt;
                        decode_secs += decode_dt;
                        next_meta = nxt;
                    }
                } else {
                    let gemm_t0 = Instant::now();
                    if use_a {
                        if accum_mode_runtime == GrmAccumMode::Auto {
                            accum_mode_runtime = resolve_grm_accum_mode_f32(
                                accum_mode_runtime,
                                &block_a[..cur_rows * n_samples],
                                cur_rows,
                                n_samples,
                            );
                        }
                        grm_rankk_update_raw(
                            grm_ptr,
                            &block_a[..cur_rows * n_samples],
                            cur_rows,
                            n_samples,
                            accum_mode_runtime,
                            cblas_copy_rhs,
                            is_first_block,
                            cblas_force_tmp_accum,
                        )?;
                    } else {
                        if accum_mode_runtime == GrmAccumMode::Auto {
                            accum_mode_runtime = resolve_grm_accum_mode_f32(
                                accum_mode_runtime,
                                &block_b[..cur_rows * n_samples],
                                cur_rows,
                                n_samples,
                            );
                        }
                        grm_rankk_update_raw(
                            grm_ptr,
                            &block_b[..cur_rows * n_samples],
                            cur_rows,
                            n_samples,
                            accum_mode_runtime,
                            cblas_copy_rhs,
                            is_first_block,
                            cblas_force_tmp_accum,
                        )?;
                    }
                    gemm_secs += gemm_t0.elapsed().as_secs_f64();
                    if !cur_eof {
                        let decode_t0 = Instant::now();
                        next_meta = if stream_prestat_core {
                            let ps = prestats
                                .as_ref()
                                .ok_or_else(|| "missing pre-stat cache".to_string())?;
                            if use_a {
                                let (rows, scanned_now, eof_now) = decode_grm_stream_block_prepared_into(
                                    &it,
                                    &mut block_b,
                                    row_step,
                                    n_samples,
                                    &ps.keep_indices,
                                    &ps.prepared_rows,
                                    &mut prepared_cursor,
                                    &mut prepared_raw_scanned,
                                    n_total,
                                    parallel_decode,
                                    decode_pool.as_ref(),
                                );
                                (rows, scanned_now, 0.0_f64, eof_now)
                            } else {
                                let (rows, scanned_now, eof_now) = decode_grm_stream_block_prepared_into(
                                    &it,
                                    &mut block_a,
                                    row_step,
                                    n_samples,
                                    &ps.keep_indices,
                                    &ps.prepared_rows,
                                    &mut prepared_cursor,
                                    &mut prepared_raw_scanned,
                                    n_total,
                                    parallel_decode,
                                    decode_pool.as_ref(),
                                );
                                (rows, scanned_now, 0.0_f64, eof_now)
                            }
                        } else if use_a {
                            decode_grm_stream_block_dispatch(
                                &mut it,
                                &mut block_b,
                                row_step,
                                n_samples,
                                method,
                                maf_thr,
                                miss_thr,
                                eps,
                                parallel_decode,
                                decode_batch_rows,
                                decode_pool.as_ref(),
                            )
                        } else {
                            decode_grm_stream_block_dispatch(
                                &mut it,
                                &mut block_a,
                                row_step,
                                n_samples,
                                method,
                                maf_thr,
                                miss_thr,
                                eps,
                                parallel_decode,
                                decode_batch_rows,
                                decode_pool.as_ref(),
                            )
                        };
                        decode_secs += decode_t0.elapsed().as_secs_f64();
                    }
                }
                is_first_block = false;

                scanned = scanned.saturating_add(cur_scanned);
                eff_m = eff_m.saturating_add(cur_rows);
                if method == 1 {
                    varsum_acc += cur_varsum;
                }

                let done_cb = if stream_prestat_core {
                    let eff_total = prestats
                        .as_ref()
                        .map(|ps| ps.eff_m)
                        .unwrap_or(1usize)
                        .max(1usize);
                    grm_progress_map_between(
                        eff_m.min(eff_total),
                        eff_total,
                        progress_phase1_cap,
                        progress_phase2_cap,
                    )
                } else {
                    grm_progress_map_ratio(scanned, n_total, progress_phase2_cap)
                };
                grm_progress_notify(
                    progress_callback.as_ref(),
                    done_cb,
                    n_total,
                    notify_step,
                    &mut last_notified,
                    cur_eof,
                )?;

                if cur_eof {
                    break;
                }
                (cur_rows, cur_scanned, cur_varsum, cur_eof) = next_meta;
                use_a = !use_a;
            }

            if eff_m == 0 {
                return Err("No SNPs remained after filtering; GRM is empty.".to_string());
            }
            let scale = if method == 1 {
                if !(varsum_acc.is_finite() && varsum_acc > 0.0) {
                    return Err("invalid centered GRM denominator: sum(2p(1-p)) <= 0".to_string());
                }
                varsum_acc as f32
            } else {
                eff_m as f32
            };
            if !(scale.is_finite() && scale > 0.0) {
                return Err("invalid GRM scale factor".to_string());
            }
            let inv_scale = 1.0_f32 / scale;
            unsafe {
                grm_scale_and_symmetrize_raw(grm_ptr, n_samples, inv_scale);
                grm.set_len(n_samples * n_samples);
            }

            if stage_timing {
                let total_secs = total_t0.elapsed().as_secs_f64();
                let other_secs = (total_secs - decode_secs - gemm_secs).max(0.0_f64);
                let to_pct = |x: f64| -> f64 {
                    if total_secs > 0.0 {
                        x * 100.0 / total_secs
                    } else {
                        0.0
                    }
                };
                eprintln!(
                    "GRM stream timing: decode={:.3}s ({:.1}%), gemm={:.3}s ({:.1}%), \
other={:.3}s ({:.1}%), total={:.3}s, row_step={}, decode_batch_rows={}, n_samples={}, eff_m={}, \
threads_total={}, decode_threads={}, blas_threads={}, overlap={}, par_decode={}, accum={}, prestat_core={}",
                    decode_secs,
                    to_pct(decode_secs),
                    gemm_secs,
                    to_pct(gemm_secs),
                    other_secs,
                    to_pct(other_secs),
                    total_secs,
                    row_step,
                    decode_batch_rows,
                    n_samples,
                    eff_m,
                    total_threads,
                    decode_threads,
                    blas_threads,
                    if pipeline_overlap { "on" } else { "off" },
                    if parallel_decode { "on" } else { "off" },
                    grm_accum_mode_name(accum_mode_runtime),
                    if stream_prestat_core { "on" } else { "off" }
                );
            }
            grm_progress_notify(
                progress_callback.as_ref(),
                n_total,
                n_total,
                notify_step,
                &mut last_notified,
                true,
            )?;
            Ok((grm, eff_m))
        })
        .map_err(map_err_string_to_py)?;

    let (grm_owned, eff_m) = grm_vec;
    let grm_arr = PyArray2::from_owned_array(
        py,
        Array2::from_shape_vec((n_samples, n_samples), grm_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    Ok((grm_arr, eff_m, n_samples))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    out_npy_path,
    method=1,
    maf_threshold=0.02,
    max_missing_rate=0.05,
    block_cols=65536,
    threads=0,
    progress_callback=None,
    progress_every=0,
    mmap_window_mb=None
))]
pub fn grm_stream_bed_f32_to_npy(
    py: Python,
    prefix: String,
    out_npy_path: String,
    method: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
    block_cols: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<(usize, usize)> {
    let (grm_arr, eff_m, n_samples) = grm_stream_bed_f32(
        py,
        prefix,
        method,
        maf_threshold,
        max_missing_rate,
        block_cols,
        threads,
        progress_callback,
        progress_every,
        mmap_window_mb,
    )?;
    let grm_ro = grm_arr.readonly();
    let data = grm_ro
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("GRM array is not contiguous"))?;
    write_npy_f32_matrix(&out_npy_path, data, n_samples, n_samples)
        .map_err(PyRuntimeError::new_err)?;
    Ok((eff_m, n_samples))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    method=1,
    block_cols=65536,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_f32_with_stats<'py>(
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
    grm_packed_f32_with_stats_impl(
        py,
        packed,
        n_samples,
        row_flip,
        row_maf,
        sample_indices,
        method,
        block_cols,
        threads,
        progress_callback,
        progress_every,
    )
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    sample_indices=None,
    method=1,
    block_cols=65536,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_packed_f64_with_stats<'py>(
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
    grm_packed_f64_with_stats_impl(
        py,
        packed,
        n_samples,
        row_flip,
        row_maf,
        sample_indices,
        method,
        block_cols,
        threads,
        progress_callback,
        progress_every,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrmSimMode {
    SerialSgemm,
    ThreadsatReduce,
}

#[inline]
fn parse_grm_sim_mode(raw: &str) -> Result<GrmSimMode, String> {
    let m = raw.trim().to_ascii_lowercase();
    match m.as_str() {
        "serial_sgemm" | "serial" => Ok(GrmSimMode::SerialSgemm),
        "threadsat_reduce" | "threadsat" | "reduce" => Ok(GrmSimMode::ThreadsatReduce),
        _ => Err(format!(
            "unsupported mode='{raw}', expected one of: serial_sgemm, threadsat_reduce"
        )),
    }
}

#[inline]
fn fill_synthetic_centered_matrix_f32(out: &mut [f32], m: usize, n: usize, seed: u64) {
    debug_assert_eq!(out.len(), m.saturating_mul(n));
    for i in 0..m {
        let mut row_sum = 0.0_f32;
        let row = &mut out[i * n..(i + 1) * n];
        for (j, v) in row.iter_mut().enumerate() {
            let mut x = seed
                ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            let u = ((x >> 11) as f64) * (1.0_f64 / ((1_u64 << 53) as f64));
            let g = (2.0_f64 * u - 1.0_f64) as f32;
            *v = g;
            row_sum += g;
        }
        let mean = row_sum / (n.max(1) as f32);
        for v in row.iter_mut() {
            *v -= mean;
        }
    }
}

#[inline]
fn sgemm_add_aat_f32(grm: &mut [f32], block: &[f32], rows: usize, n: usize, beta: f32) {
    debug_assert_eq!(block.len(), rows.saturating_mul(n));
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            n as CblasInt,
            n as CblasInt,
            rows as CblasInt,
            1.0,
            block.as_ptr(),
            n as CblasInt,
            block.as_ptr(),
            n as CblasInt,
            beta,
            grm.as_mut_ptr(),
            n as CblasInt,
        );
    }
}

#[inline]
fn grm_sim_serial_sgemm_f32(grm: &mut [f32], x: &[f32], m: usize, n: usize, batch_rows: usize) {
    let mut row0 = 0usize;
    let mut is_first_block = true;
    while row0 < m {
        let rows = (m - row0).min(batch_rows.max(1));
        let block = &x[row0 * n..(row0 + rows) * n];
        let beta = if is_first_block { 0.0_f32 } else { 1.0_f32 };
        sgemm_add_aat_f32(grm, block, rows, n, beta);
        is_first_block = false;
        row0 += rows;
    }
}

#[inline]
fn grm_sim_auto_batch_rows_f32(n_rows: usize, n_samples: usize) -> usize {
    if n_rows == 0 || n_samples == 0 {
        return 1;
    }
    let target_mb = std::env::var("JX_GRM_SIM_BATCH_TARGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .or_else(|| {
            std::env::var("JX_GRM_SIM_L3_HALF_MB")
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
        })
        .unwrap_or(512.0_f64);
    let target_bytes = (target_mb * 1024.0_f64 * 1024.0_f64) as usize;
    let bytes_per_row = n_samples.saturating_mul(std::mem::size_of::<f32>()).max(1);
    target_bytes
        .saturating_div(bytes_per_row)
        .max(64)
        .min(n_rows.max(1))
}

#[inline]
fn grm_sim_auto_tile_cols(n: usize, threads: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if let Ok(raw) = std::env::var("JX_GRM_SIM_TILE_COLS") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            if v > 0 {
                return v.min(n);
            }
        }
    }
    let t = threads.max(1);
    let target_tiles = (t * 2).max(1);
    ((n + target_tiles - 1) / target_tiles)
        .max(64)
        .min(1024)
        .min(n)
}

#[inline]
fn grm_sim_threadsat_tiled_sgemm_f32(
    grm: &mut [f32],
    x: &[f32],
    m: usize,
    n: usize,
    tile_cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(grm.len(), n.saturating_mul(n));
    debug_assert_eq!(x.len(), m.saturating_mul(n));
    let tile = tile_cols.max(1).min(n);
    let chunk_len = n.saturating_mul(tile).max(n);
    let mut run = || {
        grm.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(tile_idx, c_chunk)| {
                let j0 = tile_idx * tile;
                let nj = c_chunk.len() / n;
                let b_ptr = unsafe { x.as_ptr().add(j0) };
                unsafe {
                    cblas_sgemm_dispatch(
                        CBLAS_COL_MAJOR,
                        CBLAS_NO_TRANS,
                        CBLAS_TRANS,
                        n as CblasInt,
                        nj as CblasInt,
                        m as CblasInt,
                        1.0,
                        x.as_ptr(),
                        n as CblasInt,
                        b_ptr,
                        n as CblasInt,
                        0.0,
                        c_chunk.as_mut_ptr(),
                        n as CblasInt,
                    );
                }
            });
    };
    if let Some(tp) = pool {
        tp.install(run);
    } else {
        run();
    }
}

#[pyfunction]
#[pyo3(signature = (
    n_rows,
    n_samples,
    mode="threadsat_reduce".to_string(),
    batch_rows=1024,
    threads=0,
    repeats=3,
    seed=0x1234_5678_9ABC_DEF0
))]
pub fn grm_sim_bench_f32(
    py: Python,
    n_rows: usize,
    n_samples: usize,
    mode: String,
    batch_rows: usize,
    threads: usize,
    repeats: usize,
    seed: u64,
) -> PyResult<(f64, f64, f64, f64, usize, String)> {
    if n_rows == 0 || n_samples == 0 {
        return Err(PyRuntimeError::new_err(
            "n_rows and n_samples must both be > 0",
        ));
    }
    let mode_parsed = parse_grm_sim_mode(&mode).map_err(PyRuntimeError::new_err)?;
    let auto_batch = grm_sim_auto_batch_rows_f32(n_rows, n_samples);
    let user_batch = batch_rows.max(1).min(n_rows);
    let batch = user_batch.max(auto_batch).min(n_rows);
    let reps = repeats.max(1);
    let pool = get_cached_pool(threads)?;
    let used_threads = if let Some(tp) = pool.as_ref() {
        tp.current_num_threads()
    } else if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
    };

    let out = py
        .detach(
            move || -> Result<(f64, f64, f64, f64, usize, String), String> {
                let m = n_rows;
                let n = n_samples;
                let mut x = vec![0.0_f32; m * n];
                fill_synthetic_centered_matrix_f32(&mut x, m, n, seed);
                let mut g = vec![0.0_f32; n * n];
                let tile_cols = grm_sim_auto_tile_cols(n, used_threads);

                let mut best_secs = f64::INFINITY;
                let mut best_diag = 0.0_f64;
                let mut best_sum = 0.0_f64;

                for _ in 0..reps {
                    let t0 = Instant::now();
                    match mode_parsed {
                        GrmSimMode::SerialSgemm => {
                            let blas_threads = if threads > 0 { threads } else { used_threads };
                            let _blas_guard = OpenBlasThreadGuard::enter(blas_threads.max(1));
                            grm_sim_serial_sgemm_f32(&mut g, &x, m, n, batch);
                        }
                        GrmSimMode::ThreadsatReduce => {
                            let _blas_guard = OpenBlasThreadGuard::enter(1);
                            grm_sim_threadsat_tiled_sgemm_f32(
                                &mut g,
                                &x,
                                m,
                                n,
                                tile_cols,
                                pool.as_ref(),
                            );
                        }
                    }
                    let secs = t0.elapsed().as_secs_f64();
                    if secs < best_secs {
                        best_secs = secs;
                        let mut diag = 0.0_f64;
                        let mut sum = 0.0_f64;
                        for i in 0..n {
                            diag += g[i * n + i] as f64;
                        }
                        for &v in g.iter() {
                            sum += v as f64;
                        }
                        best_diag = diag;
                        best_sum = sum;
                    }
                }

                let flops = 2.0_f64 * (m as f64) * (n as f64) * (n as f64);
                let gflops = if best_secs.is_finite() && best_secs > 0.0 {
                    flops / best_secs / 1e9_f64
                } else {
                    0.0_f64
                };
                let mode_name = match mode_parsed {
                    GrmSimMode::SerialSgemm => "serial_sgemm",
                    GrmSimMode::ThreadsatReduce => "threadsat_reduce",
                }
                .to_string();
                Ok((
                    best_secs,
                    gflops,
                    best_diag,
                    best_sum,
                    used_threads,
                    mode_name,
                ))
            },
        )
        .map_err(map_err_string_to_py)?;
    Ok(out)
}
