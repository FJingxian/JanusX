use memmap2::Mmap;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use crate::bedmath::decode_mean_imputed_additive_packed_block_rows_f32;
use crate::bedmath::packed_byte_lut;
use crate::blas::BlasThreadGuard;
use crate::decode::AdditiveDecodePlan;
use crate::gfcore;
use crate::gfcore::{BimChunkReader, SiteInfo};
use crate::glm::{
    lm_stream_bed_additive_prepared_unified, normalize_plink_prefix_local, student_t_p_two_sided,
    LmQrProjection,
};
use crate::gload::WindowedBedMatrix;
use crate::he::row_major_block_mul_mat_f32;
use crate::linalg::{format_chisq_value, sanitize_assoc_pvalue};
use crate::stats_common::{
    arm_interrupt_trap, check_ctrlc, get_cached_pool, map_err_string_to_py,
    parse_index_vec_i64_result, AsyncTsvWriter,
};

#[derive(Clone, Debug)]
struct TraitPreparedMeta {
    row_indices: Vec<i64>,
    row_flip: Vec<bool>,
    row_missing: Vec<f32>,
    row_maf: Vec<f32>,
}

#[derive(Clone, Debug)]
struct TraitJob {
    name: String,
    n_idv: usize,
    y: Vec<f64>,
    x: Vec<f64>,
    x_cols: usize,
    sample_indices: Vec<usize>,
    prepared: TraitPreparedMeta,
}

#[derive(Clone, Debug)]
struct TraitResult {
    idx: usize,
    name: String,
    n_idv: usize,
    kept_rows: usize,
    scanned_rows: usize,
    elapsed_secs: f64,
    rows_text: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RowMissingMode {
    Rate,
    Count,
}

const LM_TRAIT_AGG_HEADER: &str =
    "chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tphenotype\n";
const LM_TRAIT_TEXT_FLUSH_BYTES: usize = 8 * 1024 * 1024;
const LM_TRAIT_FASTPATH_MAX_GENO_BYTES: usize = 512 * 1024 * 1024;

fn req_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "lm_trait_assoc_bed_to_tsv trait spec missing required key '{key}'"
        ))
    })
}

fn opt_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    d.get_item(key)
}

fn array1_f64(obj: Bound<'_, PyAny>, label: &str) -> PyResult<Vec<f64>> {
    let arr = obj
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|_| PyValueError::new_err(format!("{label} must be a 1D float64 numpy array")))?;
    match arr.as_slice() {
        Ok(slc) => Ok(slc.to_vec()),
        Err(_) => Ok(arr.as_array().iter().copied().collect()),
    }
}

fn array2_f64(obj: Bound<'_, PyAny>, label: &str) -> PyResult<(Vec<f64>, usize, usize)> {
    let arr = obj
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|_| PyValueError::new_err(format!("{label} must be a 2D float64 numpy array")))?;
    let view = arr.as_array();
    let rows = view.shape()[0];
    let cols = view.shape()[1];
    let flat = match arr.as_slice() {
        Ok(slc) => slc.to_vec(),
        Err(_) => view.iter().copied().collect(),
    };
    Ok((flat, rows, cols))
}

fn array1_i64(obj: Bound<'_, PyAny>, label: &str) -> PyResult<Vec<i64>> {
    let arr = obj
        .extract::<PyReadonlyArray1<'_, i64>>()
        .map_err(|_| PyValueError::new_err(format!("{label} must be a 1D int64 numpy array")))?;
    match arr.as_slice() {
        Ok(slc) => Ok(slc.to_vec()),
        Err(_) => Ok(arr.as_array().iter().copied().collect()),
    }
}

fn array1_bool(obj: Bound<'_, PyAny>, label: &str) -> PyResult<Vec<bool>> {
    let arr = obj
        .extract::<PyReadonlyArray1<'_, bool>>()
        .map_err(|_| PyValueError::new_err(format!("{label} must be a 1D bool numpy array")))?;
    match arr.as_slice() {
        Ok(slc) => Ok(slc.to_vec()),
        Err(_) => Ok(arr.as_array().iter().copied().collect()),
    }
}

fn array1_f32(obj: Bound<'_, PyAny>, label: &str) -> PyResult<Vec<f32>> {
    let arr = obj
        .extract::<PyReadonlyArray1<'_, f32>>()
        .map_err(|_| PyValueError::new_err(format!("{label} must be a 1D float32 numpy array")))?;
    match arr.as_slice() {
        Ok(slc) => Ok(slc.to_vec()),
        Err(_) => Ok(arr.as_array().iter().copied().collect()),
    }
}

fn temp_worker_path(base_path: &str, worker_idx: usize) -> String {
    format!("{base_path}.worker.{worker_idx}.part.tsv")
}

fn read_trait_rows_with_phenotype(tmp_tsv: &str, trait_name: &str) -> Result<String, String> {
    let file = File::open(tmp_tsv).map_err(|e| format!("open {tmp_tsv}: {e}"))?;
    let mut reader = BufReader::new(file);
    let mut header = String::new();
    let n_read = reader
        .read_line(&mut header)
        .map_err(|e| format!("read header {tmp_tsv}: {e}"))?;
    if n_read == 0 {
        return Ok(String::new());
    }

    let mut out = String::new();
    for line in reader.lines() {
        let row = line.map_err(|e| format!("read {tmp_tsv}: {e}"))?;
        if row.is_empty() {
            continue;
        }
        out.push_str(&row);
        out.push('\t');
        out.push_str(trait_name);
        out.push('\n');
    }
    Ok(out)
}

fn cleanup_trait_tmp_paths(paths: &[String]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

#[inline]
fn vec_f64_bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
}

#[inline]
fn vec_f32_bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
}

fn jobs_support_shared_fast_path(jobs: &[TraitJob]) -> bool {
    if jobs.len() < 2 {
        return false;
    }
    let reference = &jobs[0];
    if reference.y.is_empty() {
        return false;
    }
    let geno_bytes = reference
        .prepared
        .row_indices
        .len()
        .saturating_mul(reference.y.len())
        .saturating_mul(std::mem::size_of::<f32>());
    if geno_bytes == 0 || geno_bytes > LM_TRAIT_FASTPATH_MAX_GENO_BYTES {
        return false;
    }
    jobs.iter().skip(1).all(|job| {
        job.x_cols == reference.x_cols
            && job.sample_indices == reference.sample_indices
            && vec_f64_bits_eq(job.x.as_slice(), reference.x.as_slice())
            && job.prepared.row_indices == reference.prepared.row_indices
            && job.prepared.row_flip == reference.prepared.row_flip
            && vec_f32_bits_eq(
                job.prepared.row_missing.as_slice(),
                reference.prepared.row_missing.as_slice(),
            )
            && vec_f32_bits_eq(
                job.prepared.row_maf.as_slice(),
                reference.prepared.row_maf.as_slice(),
            )
    })
}

fn infer_row_missing_mode(row_missing: &[f32]) -> RowMissingMode {
    if row_missing
        .iter()
        .all(|&v| v.is_finite() && (-1.0e-6_f32..=1.0_f32 + 1.0e-6_f32).contains(&v))
    {
        RowMissingMode::Rate
    } else {
        RowMissingMode::Count
    }
}

fn build_site_prefixes(
    sites: &[SiteInfo],
    maf: &[f32],
    row_missing: &[f32],
    miss_mode: RowMissingMode,
) -> Result<Vec<String>, String> {
    let rows = sites.len();
    if maf.len() != rows || row_missing.len() != rows {
        return Err(format!(
            "site prefix metadata length mismatch: rows={rows}, maf={}, miss={}",
            maf.len(),
            row_missing.len()
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let site = &sites[row_idx];
        let mut text = String::with_capacity(96);
        match miss_mode {
            RowMissingMode::Rate => {
                let _ = write!(
                    &mut text,
                    "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t",
                    site.chrom,
                    site.pos,
                    site.snp,
                    site.ref_allele,
                    site.alt_allele,
                    maf[row_idx],
                    row_missing[row_idx],
                );
            }
            RowMissingMode::Count => {
                let _ = write!(
                    &mut text,
                    "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t",
                    site.chrom,
                    site.pos,
                    site.snp,
                    site.ref_allele,
                    site.alt_allele,
                    maf[row_idx],
                    row_missing[row_idx].round() as i64,
                );
            }
        }
        out.push(text);
    }
    Ok(out)
}

#[inline]
fn resolve_trait_batch_size(n_samples: usize, n_snps: usize, n_traits: usize) -> usize {
    if n_traits <= 1 {
        return 1;
    }
    let per_trait_bytes = n_samples
        .saturating_add(n_snps)
        .saturating_mul(std::mem::size_of::<f32>())
        .saturating_add(256usize);
    let target_bytes = 32usize << 20;
    let by_mem = target_bytes.saturating_div(per_trait_bytes.max(1));
    by_mem.max(64).min(1024).min(n_traits.max(1))
}

#[inline]
fn lm_assoc_from_shared_projection(
    qg: &[f32],
    qty: &[f64],
    gy_raw: f64,
    gg_resid: f64,
    rss0: f64,
    n: usize,
    rank: usize,
) -> (f64, f64, f64, f64) {
    let df = (n as i32) - (rank as i32) - 1;
    if !(gg_resid.is_finite() && gg_resid > 1e-8_f64) || df <= 0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }

    let mut gy_resid = gy_raw;
    for k in 0..rank {
        gy_resid -= (qg[k] as f64) * qty[k];
    }
    let beta = gy_resid / gg_resid;
    let rss1 = (rss0 - gy_resid * beta).max(0.0_f64);
    let ve = rss1 / (df as f64);
    if !(beta.is_finite() && ve.is_finite() && ve > 0.0_f64) {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let se = (ve / gg_resid).sqrt();
    if !(se.is_finite() && se > 0.0_f64) {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let t = beta / se;
    let stat = (n as f64) * (1.0_f64 + (t * t) / (df as f64)).ln();
    let pwald = student_t_p_two_sided(t, df);
    (beta, se, stat, pwald)
}

#[inline]
fn flush_trait_text(writer: &AsyncTsvWriter, text: &mut String) -> Result<(), String> {
    if text.is_empty() {
        return Ok(());
    }
    writer.send(std::mem::take(text).into_bytes())?;
    text.reserve(LM_TRAIT_TEXT_FLUSH_BYTES);
    Ok(())
}

fn decode_prepared_additive_matrix(
    norm_prefix: &str,
    sample_indices: &[usize],
    prepared: &TraitPreparedMeta,
    chunk_size: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
    mmap_window_mb: Option<usize>,
) -> Result<(Vec<f32>, Vec<SiteInfo>), String> {
    let m = prepared.row_indices.len();
    if m == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let full_samples = gfcore::read_fam(norm_prefix)?;
    let n_samples_full = full_samples.len();
    if n_samples_full == 0 {
        return Err("no samples in PLINK FAM".to_string());
    }
    if let Some(max_sid) = sample_indices.iter().copied().max() {
        if max_sid >= n_samples_full {
            return Err(format!(
                "sample index {max_sid} out of bounds for {n_samples_full} BED samples"
            ));
        }
    }

    let bytes_per_snp = n_samples_full.div_ceil(4);
    let bed_path = format!("{norm_prefix}.bed");
    let mut bed_window = if let Some(window_mb) = mmap_window_mb.filter(|&v| v > 0) {
        Some(WindowedBedMatrix::open(norm_prefix, window_mb)?)
    } else {
        None
    };
    let full_mmap = if bed_window.is_none() {
        let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
        if mmap.len() < 3 || mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err("Only SNP-major BED supported".to_string());
        }
        let data_len = mmap.len() - 3;
        if data_len % bytes_per_snp != 0 {
            return Err(format!(
                "BED payload length {data_len} not a multiple of {bytes_per_snp}"
            ));
        }
        Some(mmap)
    } else {
        None
    };
    let n_snps = if let Some(window) = bed_window.as_ref() {
        window.n_source_snps()
    } else {
        let mmap = full_mmap
            .as_ref()
            .ok_or_else(|| "internal error: missing BED source".to_string())?;
        (mmap.len() - 3) / bytes_per_snp
    };
    let source_rows =
        parse_index_vec_i64_result(prepared.row_indices.as_slice(), n_snps, "row_indices")?;
    let mut prev_src = None::<usize>;
    for &src_row in source_rows.iter() {
        if let Some(prev) = prev_src {
            if src_row < prev {
                return Err(
                    "prepared row_indices must be sorted in ascending BED order".to_string()
                );
            }
        }
        prev_src = Some(src_row);
    }

    let decode_plan = AdditiveDecodePlan::from_sample_indices(n_samples_full, sample_indices);
    let sample_identity = decode_plan.sample_identity();
    let n = sample_indices.len();
    let row_block = chunk_size.max(1).min(m.max(1));
    let code4_lut = &packed_byte_lut().code4;
    let mut bim_reader = BimChunkReader::open(norm_prefix)?;
    let mut block = vec![0.0_f32; m.saturating_mul(n)];
    let mut sites = Vec::<SiteInfo>::with_capacity(m);
    let mut rel_row_indices = Vec::<usize>::with_capacity(row_block);
    let mut contig_row_indices = Vec::<usize>::with_capacity(row_block);

    let mut row_start = 0usize;
    while row_start < m {
        check_ctrlc()?;
        let row_end = (row_start + row_block).min(m);
        let source_rows_batch = &source_rows[row_start..row_end];
        let chunk_sites = bim_reader.read_selected_rows(source_rows_batch)?;
        sites.extend(chunk_sites);

        let (chunk_packed, local_row_indices): (&[u8], &[usize]) = match bed_window.as_mut() {
            Some(window) => {
                let slice = window.prepare_source_rows(source_rows_batch, &mut rel_row_indices)?;
                (slice, rel_row_indices.as_slice())
            }
            None => {
                let mmap = full_mmap
                    .as_ref()
                    .ok_or_else(|| "internal error: missing full BED mmap".to_string())?;
                let packed = &mmap[3..];
                let src_start = source_rows_batch[0];
                let src_end = source_rows_batch[source_rows_batch.len() - 1] + 1;
                let start_byte = src_start.saturating_mul(bytes_per_snp);
                let end_byte = src_end.saturating_mul(bytes_per_snp);
                contig_row_indices.clear();
                contig_row_indices.extend(
                    source_rows_batch
                        .iter()
                        .map(|&src_row| src_row.saturating_sub(src_start)),
                );
                (&packed[start_byte..end_byte], contig_row_indices.as_slice())
            }
        };

        decode_mean_imputed_additive_packed_block_rows_f32(
            chunk_packed,
            bytes_per_snp,
            n_samples_full,
            &prepared.row_flip[row_start..row_end],
            &prepared.row_maf[row_start..row_end],
            sample_indices,
            sample_identity,
            Some(local_row_indices),
            0,
            &mut block[row_start * n..row_end * n],
            code4_lut,
            pool,
        )?;
        row_start = row_end;
    }

    Ok((block, sites))
}

fn lm_trait_assoc_bed_to_tsv_shared_fast(
    jobs: &[TraitJob],
    norm_prefix: &str,
    out_tsv: &str,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    mmap_window_mb: Option<usize>,
) -> Result<Vec<(String, usize, usize, usize, f64)>, String> {
    let reference = &jobs[0];
    let n = reference.y.len();
    let m = reference.prepared.row_indices.len();
    if m == 0 {
        return Ok(jobs
            .iter()
            .map(|job| (job.name.clone(), job.n_idv, 0usize, 0usize, 0.0_f64))
            .collect());
    }

    let zero_y = vec![0.0_f64; n];
    let design_ctx = LmQrProjection::from_design(
        reference.x.as_slice(),
        zero_y.as_slice(),
        n,
        reference.x_cols,
    )?;
    let rank = design_ctx.rank();
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let _blas_guard = BlasThreadGuard::enter(threads.max(1));

    let (g_mat, sites) = decode_prepared_additive_matrix(
        norm_prefix,
        reference.sample_indices.as_slice(),
        &reference.prepared,
        chunk_size,
        pool.as_ref(),
        mmap_window_mb,
    )?;
    if sites.len() != m {
        return Err(format!(
            "decoded site metadata mismatch: expected {m}, got {}",
            sites.len()
        ));
    }

    let mut qg_mat = vec![0.0_f32; m.saturating_mul(rank)];
    if rank > 0 {
        row_major_block_mul_mat_f32(
            g_mat.as_slice(),
            m,
            n,
            design_ctx.q_f32(),
            rank,
            qg_mat.as_mut_slice(),
            pool.as_ref(),
        );
    }

    let mut gg_resid = vec![0.0_f64; m];
    for row_idx in 0..m {
        let g_row = &g_mat[row_idx * n..(row_idx + 1) * n];
        let mut ss = 0.0_f64;
        for &value in g_row.iter() {
            let v = value as f64;
            ss += v * v;
        }
        let mut q_quad = 0.0_f64;
        if rank > 0 {
            let qg_row = &qg_mat[row_idx * rank..(row_idx + 1) * rank];
            for &value in qg_row.iter() {
                let v = value as f64;
                q_quad += v * v;
            }
        }
        gg_resid[row_idx] = ss - q_quad;
    }

    let miss_mode = infer_row_missing_mode(reference.prepared.row_missing.as_slice());
    let site_prefixes = build_site_prefixes(
        sites.as_slice(),
        reference.prepared.row_maf.as_slice(),
        reference.prepared.row_missing.as_slice(),
        miss_mode,
    )?;
    let writer = AsyncTsvWriter::with_config(
        out_tsv,
        LM_TRAIT_AGG_HEADER.as_bytes(),
        64 * 1024 * 1024,
        16,
    )?;
    let trait_batch = resolve_trait_batch_size(n, m, jobs.len());
    let q_f64 = design_ctx.q();
    let total_traits = jobs.len();
    let progress_step = total_traits.div_ceil(256).max(1);
    let start_all = Instant::now();
    let mut stats = Vec::<(String, usize, usize, usize, f64)>::with_capacity(total_traits);
    let mut text = String::with_capacity(LM_TRAIT_TEXT_FLUSH_BYTES);

    let mut batch_start = 0usize;
    while batch_start < total_traits {
        check_ctrlc()?;
        let batch_end = (batch_start + trait_batch).min(total_traits);
        let batch_traits = batch_end - batch_start;
        let mut y_batch = vec![0.0_f32; n.saturating_mul(batch_traits)];
        let mut qty_batch = vec![0.0_f64; batch_traits.saturating_mul(rank)];
        let mut rss0_batch = vec![0.0_f64; batch_traits];
        let mut gy_batch = vec![0.0_f32; m.saturating_mul(batch_traits)];

        for trait_local in 0..batch_traits {
            let job = &jobs[batch_start + trait_local];
            let mut yy = 0.0_f64;
            for sample_idx in 0..n {
                let yv = job.y[sample_idx];
                y_batch[sample_idx * batch_traits + trait_local] = yv as f32;
                yy += yv * yv;
            }
            if rank > 0 {
                let qty_row = &mut qty_batch[trait_local * rank..(trait_local + 1) * rank];
                for k in 0..rank {
                    let mut acc = 0.0_f64;
                    for sample_idx in 0..n {
                        acc += q_f64[sample_idx * rank + k] * job.y[sample_idx];
                    }
                    qty_row[k] = acc;
                }
                let q_norm = qty_row.iter().map(|v| v * v).sum::<f64>();
                rss0_batch[trait_local] = yy - q_norm;
            } else {
                rss0_batch[trait_local] = yy;
            }
        }

        row_major_block_mul_mat_f32(
            g_mat.as_slice(),
            m,
            n,
            y_batch.as_slice(),
            batch_traits,
            gy_batch.as_mut_slice(),
            pool.as_ref(),
        );

        for trait_local in 0..batch_traits {
            check_ctrlc()?;
            let job = &jobs[batch_start + trait_local];
            let qty_row = &qty_batch[trait_local * rank..(trait_local + 1) * rank];
            let rss0 = rss0_batch[trait_local];
            for row_idx in 0..m {
                let qg_row = if rank > 0 {
                    &qg_mat[row_idx * rank..(row_idx + 1) * rank]
                } else {
                    &[]
                };
                let gy_raw = gy_batch[row_idx * batch_traits + trait_local] as f64;
                let (beta, se, stat, raw_p) = lm_assoc_from_shared_projection(
                    qg_row,
                    qty_row,
                    gy_raw,
                    gg_resid[row_idx],
                    rss0,
                    n,
                    rank,
                );
                let pwald = sanitize_assoc_pvalue(beta, se, raw_p);
                let _ = writeln!(
                    &mut text,
                    "{}{:.4}\t{:.4}\t{}\t{:.4e}\t{}",
                    site_prefixes[row_idx],
                    beta,
                    se,
                    format_chisq_value(stat),
                    pwald,
                    job.name,
                );
            }
            if text.len() >= LM_TRAIT_TEXT_FLUSH_BYTES {
                flush_trait_text(&writer, &mut text)?;
            }
            stats.push((
                job.name.clone(),
                job.n_idv,
                m,
                m,
                start_all.elapsed().as_secs_f64(),
            ));
            if let Some(cb) = progress_callback {
                let done_traits = stats.len();
                if done_traits % progress_step == 0 || done_traits == total_traits {
                    let trait_name = job.name.clone();
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (done_traits, total_traits, trait_name.as_str()))?;
                        Ok(())
                    })
                    .map_err(|_| "Interrupted by user (Ctrl+C).".to_string())?;
                }
            }
        }

        batch_start = batch_end;
    }

    flush_trait_text(&writer, &mut text)?;
    writer.finish()?;
    Ok(stats)
}

fn lm_trait_assoc_bed_matrix_shared_fast(
    trait_names: &[String],
    y_trait_major: &[f64],
    n_traits: usize,
    n: usize,
    n_idv: usize,
    x: &[f64],
    x_cols: usize,
    sample_indices: &[usize],
    prepared: &TraitPreparedMeta,
    norm_prefix: &str,
    out_tsv: &str,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    mmap_window_mb: Option<usize>,
) -> Result<Vec<(String, usize, usize, usize, f64)>, String> {
    if trait_names.len() != n_traits {
        return Err(format!(
            "trait name count mismatch: expected {n_traits}, got {}",
            trait_names.len()
        ));
    }
    if y_trait_major.len() != n_traits.saturating_mul(n) {
        return Err(format!(
            "trait matrix shape mismatch: expected {} values, got {}",
            n_traits.saturating_mul(n),
            y_trait_major.len()
        ));
    }
    let m = prepared.row_indices.len();
    if m == 0 {
        return Ok(trait_names
            .iter()
            .map(|name| (name.clone(), n_idv, 0usize, 0usize, 0.0_f64))
            .collect());
    }

    let zero_y = vec![0.0_f64; n];
    let design_ctx = LmQrProjection::from_design(x, zero_y.as_slice(), n, x_cols)?;
    let rank = design_ctx.rank();
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let _blas_guard = BlasThreadGuard::enter(threads.max(1));

    let (g_mat, sites) = decode_prepared_additive_matrix(
        norm_prefix,
        sample_indices,
        prepared,
        chunk_size,
        pool.as_ref(),
        mmap_window_mb,
    )?;
    if sites.len() != m {
        return Err(format!(
            "decoded site metadata mismatch: expected {m}, got {}",
            sites.len()
        ));
    }

    let mut qg_mat = vec![0.0_f32; m.saturating_mul(rank)];
    if rank > 0 {
        row_major_block_mul_mat_f32(
            g_mat.as_slice(),
            m,
            n,
            design_ctx.q_f32(),
            rank,
            qg_mat.as_mut_slice(),
            pool.as_ref(),
        );
    }

    let mut gg_resid = vec![0.0_f64; m];
    for row_idx in 0..m {
        let g_row = &g_mat[row_idx * n..(row_idx + 1) * n];
        let mut ss = 0.0_f64;
        for &value in g_row.iter() {
            let v = value as f64;
            ss += v * v;
        }
        let mut q_quad = 0.0_f64;
        if rank > 0 {
            let qg_row = &qg_mat[row_idx * rank..(row_idx + 1) * rank];
            for &value in qg_row.iter() {
                let v = value as f64;
                q_quad += v * v;
            }
        }
        gg_resid[row_idx] = ss - q_quad;
    }

    let miss_mode = infer_row_missing_mode(prepared.row_missing.as_slice());
    let site_prefixes = build_site_prefixes(
        sites.as_slice(),
        prepared.row_maf.as_slice(),
        prepared.row_missing.as_slice(),
        miss_mode,
    )?;
    let writer = AsyncTsvWriter::with_config(
        out_tsv,
        LM_TRAIT_AGG_HEADER.as_bytes(),
        64 * 1024 * 1024,
        16,
    )?;
    let trait_batch = resolve_trait_batch_size(n, m, n_traits);
    let q_f64 = design_ctx.q();
    let progress_step = n_traits.div_ceil(256).max(1);
    let start_all = Instant::now();
    let mut stats = Vec::<(String, usize, usize, usize, f64)>::with_capacity(n_traits);
    let mut text = String::with_capacity(LM_TRAIT_TEXT_FLUSH_BYTES);

    let mut batch_start = 0usize;
    while batch_start < n_traits {
        check_ctrlc()?;
        let batch_end = (batch_start + trait_batch).min(n_traits);
        let batch_traits = batch_end - batch_start;
        let mut y_batch = vec![0.0_f32; n.saturating_mul(batch_traits)];
        let mut qty_batch = vec![0.0_f64; batch_traits.saturating_mul(rank)];
        let mut rss0_batch = vec![0.0_f64; batch_traits];
        let mut gy_batch = vec![0.0_f32; m.saturating_mul(batch_traits)];

        for trait_local in 0..batch_traits {
            let trait_idx = batch_start + trait_local;
            let y_row = &y_trait_major[trait_idx * n..(trait_idx + 1) * n];
            let mut yy = 0.0_f64;
            for sample_idx in 0..n {
                let yv = y_row[sample_idx];
                y_batch[sample_idx * batch_traits + trait_local] = yv as f32;
                yy += yv * yv;
            }
            if rank > 0 {
                let qty_row = &mut qty_batch[trait_local * rank..(trait_local + 1) * rank];
                for k in 0..rank {
                    let mut acc = 0.0_f64;
                    for sample_idx in 0..n {
                        acc += q_f64[sample_idx * rank + k] * y_row[sample_idx];
                    }
                    qty_row[k] = acc;
                }
                let q_norm = qty_row.iter().map(|v| v * v).sum::<f64>();
                rss0_batch[trait_local] = yy - q_norm;
            } else {
                rss0_batch[trait_local] = yy;
            }
        }

        row_major_block_mul_mat_f32(
            g_mat.as_slice(),
            m,
            n,
            y_batch.as_slice(),
            batch_traits,
            gy_batch.as_mut_slice(),
            pool.as_ref(),
        );

        for trait_local in 0..batch_traits {
            check_ctrlc()?;
            let trait_idx = batch_start + trait_local;
            let qty_row = &qty_batch[trait_local * rank..(trait_local + 1) * rank];
            let rss0 = rss0_batch[trait_local];
            for row_idx in 0..m {
                let qg_row = if rank > 0 {
                    &qg_mat[row_idx * rank..(row_idx + 1) * rank]
                } else {
                    &[]
                };
                let gy_raw = gy_batch[row_idx * batch_traits + trait_local] as f64;
                let (beta, se, stat, raw_p) = lm_assoc_from_shared_projection(
                    qg_row,
                    qty_row,
                    gy_raw,
                    gg_resid[row_idx],
                    rss0,
                    n,
                    rank,
                );
                let pwald = sanitize_assoc_pvalue(beta, se, raw_p);
                let _ = writeln!(
                    &mut text,
                    "{}{:.4}\t{:.4}\t{}\t{:.4e}\t{}",
                    site_prefixes[row_idx],
                    beta,
                    se,
                    format_chisq_value(stat),
                    pwald,
                    trait_names[trait_idx],
                );
            }
            if text.len() >= LM_TRAIT_TEXT_FLUSH_BYTES {
                flush_trait_text(&writer, &mut text)?;
            }
            stats.push((
                trait_names[trait_idx].clone(),
                n_idv,
                m,
                m,
                start_all.elapsed().as_secs_f64(),
            ));
            if let Some(cb) = progress_callback {
                let done_traits = stats.len();
                if done_traits % progress_step == 0 || done_traits == n_traits {
                    let trait_name = trait_names[trait_idx].clone();
                    Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(py2, (done_traits, n_traits, trait_name.as_str()))?;
                        Ok(())
                    })
                    .map_err(|_| "Interrupted by user (Ctrl+C).".to_string())?;
                }
            }
        }

        batch_start = batch_end;
    }

    flush_trait_text(&writer, &mut text)?;
    writer.finish()?;
    Ok(stats)
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    trait_names,
    y_trait_major,
    x,
    sample_indices,
    row_indices,
    row_flip,
    row_missing,
    row_maf,
    out_tsv,
    chunk_size=10000,
    threads=0,
    progress_callback=None,
    mmap_window_mb=None,
))]
pub fn lm_trait_assoc_bed_matrix_to_tsv(
    py: Python<'_>,
    prefix: String,
    trait_names: Vec<String>,
    y_trait_major: Bound<'_, PyAny>,
    x: Bound<'_, PyAny>,
    sample_indices: Bound<'_, PyAny>,
    row_indices: Bound<'_, PyAny>,
    row_flip: Bound<'_, PyAny>,
    row_missing: Bound<'_, PyAny>,
    row_maf: Bound<'_, PyAny>,
    out_tsv: String,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    mmap_window_mb: Option<usize>,
) -> PyResult<Vec<(String, usize, usize, usize, f64)>> {
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be > 0"));
    }
    if let Some(window_mb) = mmap_window_mb {
        if window_mb == 0 {
            return Err(PyValueError::new_err("mmap_window_mb must be > 0"));
        }
    }
    if trait_names.is_empty() {
        return Ok(Vec::new());
    }
    if trait_names.iter().any(|name| name.trim().is_empty()) {
        return Err(PyValueError::new_err(
            "trait_names must not contain empty values",
        ));
    }

    let (y_flat, y_rows, y_cols) = array2_f64(y_trait_major, "y_trait_major")?;
    let (x_flat, x_rows, x_cols) = array2_f64(x, "x")?;
    let sample_idx64 = array1_i64(sample_indices, "sample_indices")?;
    let row_idx64 = array1_i64(row_indices, "row_indices")?;
    let row_flip_vec = array1_bool(row_flip, "row_flip")?;
    let row_missing_vec = array1_f32(row_missing, "row_missing")?;
    let row_maf_vec = array1_f32(row_maf, "row_maf")?;

    if y_rows != trait_names.len() {
        return Err(PyValueError::new_err(format!(
            "y_trait_major rows ({y_rows}) must equal len(trait_names) ({})",
            trait_names.len()
        )));
    }
    if x_cols == 0 {
        return Err(PyValueError::new_err("x must have at least one column"));
    }
    if x_rows != y_cols {
        return Err(PyValueError::new_err(format!(
            "x rows ({x_rows}) must equal y_trait_major cols ({y_cols})"
        )));
    }
    if sample_idx64.len() != y_cols {
        return Err(PyValueError::new_err(format!(
            "sample_indices length ({}) must equal y_trait_major cols ({y_cols})",
            sample_idx64.len()
        )));
    }
    if row_flip_vec.len() != row_idx64.len()
        || row_missing_vec.len() != row_idx64.len()
        || row_maf_vec.len() != row_idx64.len()
    {
        return Err(PyValueError::new_err(format!(
            "prepared row metadata length mismatch: row_indices={}, row_flip={}, row_missing={}, row_maf={}",
            row_idx64.len(),
            row_flip_vec.len(),
            row_missing_vec.len(),
            row_maf_vec.len()
        )));
    }

    let norm_prefix = normalize_plink_prefix_local(&prefix);
    let fam_ids = gfcore::read_fam(&norm_prefix).map_err(PyValueError::new_err)?;
    if fam_ids.is_empty() {
        return Err(PyValueError::new_err("no samples in PLINK FAM"));
    }
    let n_samples_full = fam_ids.len();
    let mut sample_idx = Vec::<usize>::with_capacity(sample_idx64.len());
    for raw_idx in sample_idx64.into_iter() {
        if raw_idx < 0 || (raw_idx as usize) >= n_samples_full {
            return Err(PyValueError::new_err(format!(
                "sample index {raw_idx} out of range for n_samples={n_samples_full}"
            )));
        }
        sample_idx.push(raw_idx as usize);
    }

    let prepared = TraitPreparedMeta {
        row_indices: row_idx64,
        row_flip: row_flip_vec,
        row_missing: row_missing_vec,
        row_maf: row_maf_vec,
    };
    arm_interrupt_trap();
    py.detach(
        move || -> PyResult<Vec<(String, usize, usize, usize, f64)>> {
            check_ctrlc().map_err(map_err_string_to_py)?;
            let total_threads = if threads > 0 {
                threads
            } else {
                thread::available_parallelism()
                    .map(|v| v.get())
                    .unwrap_or(1usize)
            };
            lm_trait_assoc_bed_matrix_shared_fast(
                trait_names.as_slice(),
                y_flat.as_slice(),
                y_rows,
                y_cols,
                y_cols,
                x_flat.as_slice(),
                x_cols,
                sample_idx.as_slice(),
                &prepared,
                norm_prefix.as_str(),
                out_tsv.as_str(),
                chunk_size,
                total_threads.max(1),
                progress_callback.as_ref(),
                mmap_window_mb,
            )
            .map_err(map_err_string_to_py)
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    trait_specs,
    out_tsv,
    chunk_size=10000,
    threads=0,
    progress_callback=None,
    mmap_window_mb=None,
))]
pub fn lm_trait_assoc_bed_to_tsv(
    py: Python<'_>,
    prefix: String,
    trait_specs: &Bound<'_, PyAny>,
    out_tsv: String,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    mmap_window_mb: Option<usize>,
) -> PyResult<Vec<(String, usize, usize, usize, f64)>> {
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be > 0"));
    }
    if let Some(window_mb) = mmap_window_mb {
        if window_mb == 0 {
            return Err(PyValueError::new_err("mmap_window_mb must be > 0"));
        }
    }

    let trait_list = trait_specs.cast::<PyList>().map_err(|_| {
        PyValueError::new_err("trait_specs must be a list of per-trait dict objects")
    })?;
    if trait_list.is_empty() {
        return Ok(Vec::new());
    }

    let norm_prefix = normalize_plink_prefix_local(&prefix);
    let fam_ids = gfcore::read_fam(&norm_prefix).map_err(PyValueError::new_err)?;
    if fam_ids.is_empty() {
        return Err(PyValueError::new_err("no samples in PLINK FAM"));
    }
    let n_samples_full = fam_ids.len();
    let mut fam_index: Option<HashMap<String, usize>> = None;

    let mut jobs: Vec<TraitJob> = Vec::with_capacity(trait_list.len());
    for (spec_idx, spec_obj) in trait_list.iter().enumerate() {
        let spec = spec_obj.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!(
                "trait_specs[{spec_idx}] must be a dict with trait_name/y/x/(sample_indices|sample_ids)/row_* keys"
            ))
        })?;
        let name = req_item(&spec, "trait_name")?.extract::<String>()?;
        if name.trim().is_empty() {
            return Err(PyValueError::new_err(format!(
                "trait_specs[{spec_idx}].trait_name must not be empty"
            )));
        }
        let y = array1_f64(req_item(&spec, "y")?, &format!("trait_specs[{spec_idx}].y"))?;
        let (x, x_rows, x_cols) =
            array2_f64(req_item(&spec, "x")?, &format!("trait_specs[{spec_idx}].x"))?;
        if x_cols == 0 {
            return Err(PyValueError::new_err(format!(
                "trait_specs[{spec_idx}].x must have at least one column"
            )));
        }
        if x_rows != y.len() {
            return Err(PyValueError::new_err(format!(
                "trait_specs[{spec_idx}] x rows ({x_rows}) must equal len(y) ({})",
                y.len()
            )));
        }
        let sample_indices = if let Some(sample_indices_obj) = opt_item(&spec, "sample_indices")? {
            let sample_idx64 = array1_i64(
                sample_indices_obj,
                &format!("trait_specs[{spec_idx}].sample_indices"),
            )?;
            if sample_idx64.len() != y.len() {
                return Err(PyValueError::new_err(format!(
                    "trait_specs[{spec_idx}] sample_indices length ({}) must equal len(y) ({})",
                    sample_idx64.len(),
                    y.len()
                )));
            }
            let mut out = Vec::<usize>::with_capacity(sample_idx64.len());
            for raw_idx in sample_idx64.into_iter() {
                if raw_idx < 0 || (raw_idx as usize) >= n_samples_full {
                    return Err(PyValueError::new_err(format!(
                        "trait_specs[{spec_idx}] sample index {raw_idx} out of range for n_samples={n_samples_full}"
                    )));
                }
                out.push(raw_idx as usize);
            }
            out
        } else {
            let sample_ids = req_item(&spec, "sample_ids")?.extract::<Vec<String>>()?;
            if sample_ids.len() != y.len() {
                return Err(PyValueError::new_err(format!(
                    "trait_specs[{spec_idx}] sample_ids length ({}) must equal len(y) ({})",
                    sample_ids.len(),
                    y.len()
                )));
            }
            let index = fam_index.get_or_insert_with(|| {
                fam_ids
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(idx, sid)| (sid, idx))
                    .collect()
            });
            let mut out = Vec::<usize>::with_capacity(sample_ids.len());
            for sid in sample_ids.iter() {
                let idx = index.get(sid).copied().ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "trait_specs[{spec_idx}] sample ID '{sid}' not found in BED/FAM"
                    ))
                })?;
                out.push(idx);
            }
            out
        };

        let row_indices = array1_i64(
            req_item(&spec, "row_indices")?,
            &format!("trait_specs[{spec_idx}].row_indices"),
        )?;
        let row_flip = array1_bool(
            req_item(&spec, "row_flip")?,
            &format!("trait_specs[{spec_idx}].row_flip"),
        )?;
        let row_missing = array1_f32(
            req_item(&spec, "row_missing")?,
            &format!("trait_specs[{spec_idx}].row_missing"),
        )?;
        let row_maf = array1_f32(
            req_item(&spec, "row_maf")?,
            &format!("trait_specs[{spec_idx}].row_maf"),
        )?;
        let row_len = row_indices.len();
        if row_flip.len() != row_len || row_missing.len() != row_len || row_maf.len() != row_len {
            return Err(PyValueError::new_err(format!(
                "trait_specs[{spec_idx}] prepared row metadata length mismatch: row_indices={row_len}, row_flip={}, row_missing={}, row_maf={}",
                row_flip.len(),
                row_missing.len(),
                row_maf.len()
            )));
        }
        let n_idv = if let Some(n_idv_obj) = opt_item(&spec, "n_idv")? {
            n_idv_obj.extract::<usize>()?
        } else {
            y.len()
        };
        jobs.push(TraitJob {
            name,
            n_idv,
            y,
            x,
            x_cols,
            sample_indices,
            prepared: TraitPreparedMeta {
                row_indices,
                row_flip,
                row_missing,
                row_maf,
            },
        });
    }

    arm_interrupt_trap();
    let n_traits = jobs.len();
    let out_path = out_tsv.clone();

    py.detach(
        move || -> PyResult<Vec<(String, usize, usize, usize, f64)>> {
            check_ctrlc().map_err(map_err_string_to_py)?;
            let total_threads = if threads > 0 {
                threads
            } else {
                thread::available_parallelism()
                    .map(|v| v.get())
                    .unwrap_or(1usize)
            };
            if jobs_support_shared_fast_path(jobs.as_slice()) {
                return lm_trait_assoc_bed_to_tsv_shared_fast(
                    jobs.as_slice(),
                    norm_prefix.as_str(),
                    out_path.as_str(),
                    chunk_size,
                    total_threads.max(1),
                    progress_callback.as_ref(),
                    mmap_window_mb,
                )
                .map_err(map_err_string_to_py);
            }

            let worker_count = n_traits.min(total_threads.max(1));
            let inner_threads = std::cmp::max(1usize, total_threads.max(1) / worker_count.max(1));
            let jobs_arc = Arc::new(jobs);
            let next_idx = Arc::new(AtomicUsize::new(0));
            let stop_flag = Arc::new(AtomicBool::new(false));
            let (tx, rx) = mpsc::channel::<Result<TraitResult, String>>();

            let mut handles = Vec::with_capacity(worker_count);
            let mut tmp_paths = Vec::<String>::with_capacity(worker_count);
            for worker_idx in 0..worker_count {
                let jobs_ref = Arc::clone(&jobs_arc);
                let next_ref = Arc::clone(&next_idx);
                let stop_ref = Arc::clone(&stop_flag);
                let tx_ref = tx.clone();
                let prefix_ref = norm_prefix.clone();
                let mmap_window_mb_ref = mmap_window_mb;
                let worker_tmp_tsv = temp_worker_path(out_path.as_str(), worker_idx);
                tmp_paths.push(worker_tmp_tsv.clone());
                let handle = thread::spawn(move || loop {
                    if stop_ref.load(Ordering::SeqCst) {
                        break;
                    }
                    let idx = next_ref.fetch_add(1, Ordering::SeqCst);
                    if idx >= jobs_ref.len() {
                        break;
                    }
                    let job = &jobs_ref[idx];
                    let start = Instant::now();
                    let qr_ctx = match LmQrProjection::from_design(
                        job.x.as_slice(),
                        job.y.as_slice(),
                        job.y.len(),
                        job.x_cols,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            stop_ref.store(true, Ordering::SeqCst);
                            let _ = tx_ref.send(Err(err));
                            break;
                        }
                    };
                    let rhs_add_f32 = match crate::glm::pack_lm_scan_rhs_f32(
                        job.y.as_slice(),
                        if qr_ctx.rank() > 0 {
                            Some(qr_ctx.q_f32())
                        } else {
                            None
                        },
                        qr_ctx.rank(),
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            stop_ref.store(true, Ordering::SeqCst);
                            let _ = tx_ref.send(Err(err));
                            break;
                        }
                    };
                    let run = lm_stream_bed_additive_prepared_unified(
                        prefix_ref.as_str(),
                        &qr_ctx,
                        rhs_add_f32.as_slice(),
                        worker_tmp_tsv.as_str(),
                        job.sample_indices.as_slice(),
                        (
                            job.prepared.row_indices.clone(),
                            job.prepared.row_flip.clone(),
                            job.prepared.row_missing.clone(),
                            job.prepared.row_maf.clone(),
                        ),
                        chunk_size,
                        inner_threads,
                        None,
                        0usize,
                        mmap_window_mb_ref,
                    );
                    match run {
                        Ok((kept_rows, scanned_rows)) => {
                            let rows_text = match read_trait_rows_with_phenotype(
                                worker_tmp_tsv.as_str(),
                                job.name.as_str(),
                            ) {
                                Ok(text) => text,
                                Err(err) => {
                                    let _ = fs::remove_file(&worker_tmp_tsv);
                                    stop_ref.store(true, Ordering::SeqCst);
                                    let _ = tx_ref.send(Err(err));
                                    break;
                                }
                            };
                            let _ = fs::remove_file(&worker_tmp_tsv);
                            let _ = tx_ref.send(Ok(TraitResult {
                                idx,
                                name: job.name.clone(),
                                n_idv: job.n_idv,
                                kept_rows,
                                scanned_rows,
                                elapsed_secs: start.elapsed().as_secs_f64(),
                                rows_text,
                            }));
                        }
                        Err(err) => {
                            let _ = fs::remove_file(&worker_tmp_tsv);
                            stop_ref.store(true, Ordering::SeqCst);
                            let _ = tx_ref.send(Err(err));
                            break;
                        }
                    }
                });
                handles.push(handle);
            }
            drop(tx);

            let mut ordered_results: Vec<Option<TraitResult>> =
                (0..n_traits).map(|_| None).collect();
            let mut final_stats: Vec<Option<(String, usize, usize, usize, f64)>> =
                (0..n_traits).map(|_| None).collect();
            let mut done_traits = 0usize;
            let mut next_write_idx = 0usize;
            let run_result: PyResult<Vec<(String, usize, usize, usize, f64)>> = (|| {
                let mut writer: Option<AsyncTsvWriter> = None;
                let mut wrote_rows = false;
                while done_traits < n_traits {
                    check_ctrlc().map_err(map_err_string_to_py)?;
                    match rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(Ok(result)) => {
                            let result_idx = result.idx;
                            let trait_name_done = result.name.clone();
                            ordered_results[result_idx] = Some(result);
                            done_traits = done_traits.saturating_add(1);
                            if let Some(cb) = progress_callback.as_ref() {
                                Python::attach(|py2| -> PyResult<()> {
                                    py2.check_signals()?;
                                    cb.call1(
                                        py2,
                                        (done_traits, n_traits, trait_name_done.as_str()),
                                    )?;
                                    Ok(())
                                })?;
                            }
                            while next_write_idx < n_traits {
                                let Some(result_ready) = ordered_results[next_write_idx].take()
                                else {
                                    break;
                                };
                                if !result_ready.rows_text.is_empty() {
                                    if writer.is_none() {
                                        writer = Some(
                                            AsyncTsvWriter::with_config(
                                                out_path.as_str(),
                                                LM_TRAIT_AGG_HEADER.as_bytes(),
                                                64 * 1024 * 1024,
                                                16,
                                            )
                                            .map_err(map_err_string_to_py)?,
                                        );
                                    }
                                    if let Some(out) = writer.as_ref() {
                                        out.send(result_ready.rows_text.into_bytes())
                                            .map_err(map_err_string_to_py)?;
                                    }
                                    wrote_rows = true;
                                }
                                final_stats[next_write_idx] = Some((
                                    result_ready.name,
                                    result_ready.n_idv,
                                    result_ready.kept_rows,
                                    result_ready.scanned_rows,
                                    result_ready.elapsed_secs,
                                ));
                                next_write_idx = next_write_idx.saturating_add(1);
                            }
                        }
                        Ok(Err(err)) => {
                            stop_flag.store(true, Ordering::SeqCst);
                            return Err(map_err_string_to_py(err));
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            continue;
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            break;
                        }
                    }
                }

                if let Some(out) = writer {
                    out.finish().map_err(map_err_string_to_py)?;
                }
                if !wrote_rows && Path::new(&out_path).exists() {
                    let _ = fs::remove_file(&out_path);
                }

                final_stats
                    .into_iter()
                    .map(|item| {
                        item.ok_or_else(|| "trait worker exited without a result".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(map_err_string_to_py)
            })();

            if run_result.is_err() {
                stop_flag.store(true, Ordering::SeqCst);
            }
            for handle in handles {
                let _ = handle.join();
            }
            cleanup_trait_tmp_paths(tmp_paths.as_slice());
            run_result
        },
    )
}
