#![allow(dead_code)]

use memmap2::Mmap;
use rayon::prelude::*;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use std::cmp::min;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct MetalAssocConfig {
    pub chunk_snps: usize,
    pub maf_threshold: f32,
    pub max_missing_rate: f32,
    pub fill_missing: bool,
    pub epsilon: f32,
    pub threads_per_threadgroup: usize,
}

impl Default for MetalAssocConfig {
    fn default() -> Self {
        Self {
            chunk_snps: 10_000,
            maf_threshold: 0.0,
            max_missing_rate: 1.0,
            fill_missing: true,
            epsilon: 1e-6,
            threads_per_threadgroup: 256,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SnpAssocRecord {
    pub snp_index: usize,
    pub chrom: String,
    pub pos: i32,
    pub ref_allele: String,
    pub alt_allele: String,
    pub beta: f32,
    pub std_error: f32,
    pub n_non_missing: usize,
    pub maf: f32,
}

#[derive(Clone, Debug)]
pub struct GpuAssocOutput {
    pub records: Vec<SnpAssocRecord>,
    pub n_samples: usize,
    pub n_scanned: usize,
    pub n_kept: usize,
    pub backend: String,
}

#[derive(Clone, Debug)]
pub struct FixedVarLmmSpec<'a> {
    /// Row-major transform L (n x n). If provided, each SNP vector x is rotated as x* = Lx.
    pub l_row_major: Option<&'a [f32]>,
}

#[derive(Clone, Debug)]
pub struct BedSiteInfo {
    pub chrom: String,
    pub pos: i32,
    pub ref_allele: String,
    pub alt_allele: String,
}

#[derive(Clone, Debug)]
pub struct PackedBedSlice<'a> {
    pub packed_rows: &'a [u8],
    pub n_samples: usize,
    pub n_snps: usize,
}

#[derive(Clone, Debug)]
pub enum BedInput<'a> {
    /// Existing PLINK prefix (`prefix.bed/.bim/.fam`) for memmap-BED path.
    Prefix(&'a str),
    /// Existing packed BED payload in SNP-major 2-bit form (no 3-byte header).
    Packed(PackedBedSlice<'a>),
}

struct PrefixBedData {
    mmap: Mmap,
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
}

enum BedRowSource<'a> {
    Prefix(PrefixBedData),
    Packed(PackedBedSlice<'a>),
}

impl<'a> BedRowSource<'a> {
    fn n_samples(&self) -> usize {
        match self {
            BedRowSource::Prefix(v) => v.n_samples,
            BedRowSource::Packed(v) => v.n_samples,
        }
    }

    fn n_snps(&self) -> usize {
        match self {
            BedRowSource::Prefix(v) => v.n_snps,
            BedRowSource::Packed(v) => v.n_snps,
        }
    }

    fn bytes_per_snp(&self) -> usize {
        match self {
            BedRowSource::Prefix(v) => v.bytes_per_snp,
            BedRowSource::Packed(v) => v.n_samples.div_ceil(4),
        }
    }

    fn row_packed(&self, snp_idx: usize) -> Result<&[u8], String> {
        match self {
            BedRowSource::Prefix(v) => {
                if snp_idx >= v.n_snps {
                    return Err(format!(
                        "snp index out of range: {snp_idx} >= n_snps={}",
                        v.n_snps
                    ));
                }
                let start = 3usize + snp_idx * v.bytes_per_snp;
                let end = start + v.bytes_per_snp;
                Ok(&v.mmap[start..end])
            }
            BedRowSource::Packed(v) => {
                if snp_idx >= v.n_snps {
                    return Err(format!(
                        "snp index out of range: {snp_idx} >= n_snps={}",
                        v.n_snps
                    ));
                }
                let bps = v.n_samples.div_ceil(4);
                let start = snp_idx * bps;
                let end = start + bps;
                if end > v.packed_rows.len() {
                    return Err(format!(
                        "packed rows truncated: snp_idx={snp_idx}, end={end}, packed_len={}",
                        v.packed_rows.len()
                    ));
                }
                Ok(&v.packed_rows[start..end])
            }
        }
    }

    fn rows_packed_range(&self, start_snp: usize, count: usize) -> Result<&[u8], String> {
        if count == 0 {
            return Ok(&[]);
        }
        match self {
            BedRowSource::Prefix(v) => {
                let end_snp = start_snp
                    .checked_add(count)
                    .ok_or_else(|| "snp range overflow in Prefix source".to_string())?;
                if end_snp > v.n_snps {
                    return Err(format!(
                        "snp range out of bounds: start={}, count={}, n_snps={}",
                        start_snp, count, v.n_snps
                    ));
                }
                let start = 3usize
                    .checked_add(start_snp.saturating_mul(v.bytes_per_snp))
                    .ok_or_else(|| "byte offset overflow in Prefix source".to_string())?;
                let bytes = count
                    .checked_mul(v.bytes_per_snp)
                    .ok_or_else(|| "byte size overflow in Prefix source".to_string())?;
                let end = start
                    .checked_add(bytes)
                    .ok_or_else(|| "byte end overflow in Prefix source".to_string())?;
                Ok(&v.mmap[start..end])
            }
            BedRowSource::Packed(v) => {
                let bps = v.n_samples.div_ceil(4);
                let end_snp = start_snp
                    .checked_add(count)
                    .ok_or_else(|| "snp range overflow in Packed source".to_string())?;
                if end_snp > v.n_snps {
                    return Err(format!(
                        "snp range out of bounds: start={}, count={}, n_snps={}",
                        start_snp, count, v.n_snps
                    ));
                }
                let start = start_snp
                    .checked_mul(bps)
                    .ok_or_else(|| "byte offset overflow in Packed source".to_string())?;
                let bytes = count
                    .checked_mul(bps)
                    .ok_or_else(|| "byte size overflow in Packed source".to_string())?;
                let end = start
                    .checked_add(bytes)
                    .ok_or_else(|| "byte end overflow in Packed source".to_string())?;
                if end > v.packed_rows.len() {
                    return Err(format!(
                        "packed rows truncated in range read: start_snp={}, count={}, end={}, packed_len={}",
                        start_snp, count, end, v.packed_rows.len()
                    ));
                }
                Ok(&v.packed_rows[start..end])
            }
        }
    }
}

/// LM GWAS scan (all-f32 path) on Apple Metal.
///
/// - `sample_indices`: optional subset in original sample order.
/// - `snp_indices`: optional random SNP subset.
/// - `bed_input`: memmap BED prefix or prepacked rows from existing packed path.
pub fn lm_scan_metal<'a>(
    bed_input: BedInput<'a>,
    y: &[f32],
    sample_indices: Option<&[usize]>,
    snp_indices: Option<&[usize]>,
    cfg: &MetalAssocConfig,
) -> Result<GpuAssocOutput, String> {
    run_assoc_scan_metal(
        bed_input,
        y,
        sample_indices,
        snp_indices,
        cfg,
        None,
        ScanModel::Lm,
    )
}

/// Fixed-variance LMM scan where null covariance is frozen and represented
/// through an optional transform `L`.
///
/// If `lmm_fixed.l_row_major` is Some(L), this function computes:
/// - y* = L y
/// - x* = L x  (for each SNP chunk)
/// and then runs Metal OLS scan on (x*, y*).
pub fn lmm_fixed_scan_metal<'a>(
    bed_input: BedInput<'a>,
    y: &[f32],
    sample_indices: Option<&[usize]>,
    snp_indices: Option<&[usize]>,
    cfg: &MetalAssocConfig,
    lmm_fixed: FixedVarLmmSpec<'_>,
) -> Result<GpuAssocOutput, String> {
    run_assoc_scan_metal(
        bed_input,
        y,
        sample_indices,
        snp_indices,
        cfg,
        lmm_fixed.l_row_major,
        ScanModel::LmmFixed,
    )
}

/// CPU reference LM scan (same input semantics as `lm_scan_metal`).
pub fn lm_scan_cpu<'a>(
    bed_input: BedInput<'a>,
    y: &[f32],
    sample_indices: Option<&[usize]>,
    snp_indices: Option<&[usize]>,
    cfg: &MetalAssocConfig,
) -> Result<GpuAssocOutput, String> {
    run_assoc_scan_cpu(
        bed_input,
        y,
        sample_indices,
        snp_indices,
        cfg,
        None,
        ScanModel::Lm,
    )
}

/// CPU reference for fixed-variance LMM (L-rotation + OLS).
pub fn lmm_fixed_scan_cpu<'a>(
    bed_input: BedInput<'a>,
    y: &[f32],
    sample_indices: Option<&[usize]>,
    snp_indices: Option<&[usize]>,
    cfg: &MetalAssocConfig,
    lmm_fixed: FixedVarLmmSpec<'_>,
) -> Result<GpuAssocOutput, String> {
    run_assoc_scan_cpu(
        bed_input,
        y,
        sample_indices,
        snp_indices,
        cfg,
        lmm_fixed.l_row_major,
        ScanModel::LmmFixed,
    )
}

#[derive(Clone, Debug)]
pub struct LmRunPerf {
    pub backend: String,
    pub elapsed_ms: f64,
    pub peak_rss_raw: Option<i64>,
    pub peak_rss_unit: &'static str,
    pub n_scanned: usize,
    pub n_kept: usize,
}

#[derive(Clone, Debug)]
pub struct LmConsistency {
    pub matched_snps: usize,
    pub max_abs_beta: f32,
    pub max_abs_se: f32,
    pub mean_abs_beta: f32,
    pub mean_abs_se: f32,
}

#[derive(Clone, Debug)]
pub struct LmCompareReport {
    pub cpu: LmRunPerf,
    pub metal: LmRunPerf,
    pub consistency: LmConsistency,
}

/// Compare LM Metal vs CPU reference in Rust.
///
/// Memory uses process peak RSS (`getrusage`) raw unit:
/// - Linux: KiB
/// - macOS: bytes
pub fn compare_lm_metal_vs_cpu<'a>(
    bed_input: BedInput<'a>,
    y: &[f32],
    sample_indices: Option<&[usize]>,
    snp_indices: Option<&[usize]>,
    cfg: &MetalAssocConfig,
) -> Result<LmCompareReport, String> {
    let peak_before_cpu = process_peak_rss_raw();
    let t0 = Instant::now();
    let cpu = lm_scan_cpu(
        clone_bed_input(&bed_input),
        y,
        sample_indices,
        snp_indices,
        cfg,
    )?;
    let cpu_elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let peak_after_cpu = process_peak_rss_raw();

    let peak_before_gpu = process_peak_rss_raw();
    let t1 = Instant::now();
    let gpu = lm_scan_metal(bed_input, y, sample_indices, snp_indices, cfg)?;
    let gpu_elapsed_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let peak_after_gpu = process_peak_rss_raw();

    let consistency = compare_assoc_output(&cpu, &gpu)?;
    let unit = peak_rss_unit();

    let cpu_perf = LmRunPerf {
        backend: cpu.backend.clone(),
        elapsed_ms: cpu_elapsed_ms,
        peak_rss_raw: rss_delta_raw(peak_before_cpu, peak_after_cpu),
        peak_rss_unit: unit,
        n_scanned: cpu.n_scanned,
        n_kept: cpu.n_kept,
    };
    let gpu_perf = LmRunPerf {
        backend: gpu.backend.clone(),
        elapsed_ms: gpu_elapsed_ms,
        peak_rss_raw: rss_delta_raw(peak_before_gpu, peak_after_gpu),
        peak_rss_unit: unit,
        n_scanned: gpu.n_scanned,
        n_kept: gpu.n_kept,
    };

    Ok(LmCompareReport {
        cpu: cpu_perf,
        metal: gpu_perf,
        consistency,
    })
}

/// Compare fixed-variance LMM Metal vs CPU reference.
///
/// `lmm_fixed.l_row_major` must be shape `(n_selected_samples, n_selected_samples)`.
pub fn compare_lmm_fixed_metal_vs_cpu<'a>(
    bed_input: BedInput<'a>,
    y: &[f32],
    sample_indices: Option<&[usize]>,
    snp_indices: Option<&[usize]>,
    cfg: &MetalAssocConfig,
    lmm_fixed: FixedVarLmmSpec<'_>,
) -> Result<LmCompareReport, String> {
    let peak_before_cpu = process_peak_rss_raw();
    let t0 = Instant::now();
    let cpu = lmm_fixed_scan_cpu(
        clone_bed_input(&bed_input),
        y,
        sample_indices,
        snp_indices,
        cfg,
        lmm_fixed.clone(),
    )?;
    let cpu_elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let peak_after_cpu = process_peak_rss_raw();

    let peak_before_gpu = process_peak_rss_raw();
    let t1 = Instant::now();
    let gpu = lmm_fixed_scan_metal(bed_input, y, sample_indices, snp_indices, cfg, lmm_fixed)?;
    let gpu_elapsed_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let peak_after_gpu = process_peak_rss_raw();

    let consistency = compare_assoc_output(&cpu, &gpu)?;
    let unit = peak_rss_unit();

    let cpu_perf = LmRunPerf {
        backend: cpu.backend.clone(),
        elapsed_ms: cpu_elapsed_ms,
        peak_rss_raw: rss_delta_raw(peak_before_cpu, peak_after_cpu),
        peak_rss_unit: unit,
        n_scanned: cpu.n_scanned,
        n_kept: cpu.n_kept,
    };
    let gpu_perf = LmRunPerf {
        backend: gpu.backend.clone(),
        elapsed_ms: gpu_elapsed_ms,
        peak_rss_raw: rss_delta_raw(peak_before_gpu, peak_after_gpu),
        peak_rss_unit: unit,
        n_scanned: gpu.n_scanned,
        n_kept: gpu.n_kept,
    };

    Ok(LmCompareReport {
        cpu: cpu_perf,
        metal: gpu_perf,
        consistency,
    })
}

fn clone_bed_input<'a>(bed_input: &BedInput<'a>) -> BedInput<'a> {
    match bed_input {
        BedInput::Prefix(p) => BedInput::Prefix(p),
        BedInput::Packed(v) => BedInput::Packed(v.clone()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScanModel {
    Lm,
    LmmFixed,
}

fn run_assoc_scan_metal<'a>(
    bed_input: BedInput<'a>,
    y: &[f32],
    sample_indices: Option<&[usize]>,
    snp_indices: Option<&[usize]>,
    cfg: &MetalAssocConfig,
    l_row_major: Option<&[f32]>,
    model: ScanModel,
) -> Result<GpuAssocOutput, String> {
    if cfg.chunk_snps == 0 {
        return Err("chunk_snps must be > 0".to_string());
    }
    if cfg.maf_threshold < 0.0 || cfg.maf_threshold > 0.5 {
        return Err("maf_threshold must be within [0, 0.5]".to_string());
    }
    if cfg.max_missing_rate < 0.0 || cfg.max_missing_rate > 1.0 {
        return Err("max_missing_rate must be within [0, 1]".to_string());
    }

    let (sites, row_source) = open_bed_input(bed_input)?;
    let n_samples_all = row_source.n_samples();
    if n_samples_all == 0 {
        return Err("BED input has zero samples".to_string());
    }

    let sample_idx = resolve_sample_indices(sample_indices, n_samples_all)?;
    let n = sample_idx.len();
    if n == 0 {
        return Err("sample selection is empty".to_string());
    }
    if y.len() != n {
        return Err(format!(
            "y length mismatch: y={}, selected_samples={}",
            y.len(),
            n
        ));
    }

    if let Some(l) = l_row_major {
        let exp = n.saturating_mul(n);
        if l.len() != exp {
            return Err(format!(
                "L shape mismatch: len={} expected n*n={}",
                l.len(),
                exp
            ));
        }
    }

    let snp_order = resolve_snp_indices(snp_indices, row_source.n_snps())?;
    if snp_order.is_empty() {
        return Ok(GpuAssocOutput {
            records: Vec::new(),
            n_samples: n,
            n_scanned: 0,
            n_kept: 0,
            backend: "metal".to_string(),
        });
    }

    let y_star = if let Some(l) = l_row_major {
        mat_vec_f32_row_major(l, n, y)?
    } else {
        y.to_vec()
    };

    let mut scanner = metal_impl::new_scanner(cfg.threads_per_threadgroup)?;

    let mut records: Vec<SnpAssocRecord> = Vec::new();
    let mut n_scanned = 0usize;
    let chunk_cap = cfg.chunk_snps;

    let use_gpu_packed_decode = model == ScanModel::Lm && l_row_major.is_none();
    let use_gpu_packed_lmm_fixed = model == ScanModel::LmmFixed && l_row_major.is_some();
    if use_gpu_packed_decode || use_gpu_packed_lmm_fixed {
        let bps = row_source.bytes_per_snp();
        let mut sample_idx_u32: Vec<u32> = Vec::with_capacity(sample_idx.len());
        for &idx in sample_idx.iter() {
            let v = u32::try_from(idx)
                .map_err(|_| format!("sample index {} exceeds u32 range for GPU path", idx))?;
            sample_idx_u32.push(v);
        }
        if use_gpu_packed_lmm_fixed {
            let l = l_row_major
                .ok_or_else(|| "internal error: fixed-LMM GPU path requires L".to_string())?;
            scanner.prepare_packed_lmm_fixed_pipeline(
                &y_star,
                &sample_idx_u32,
                l,
                bps,
                chunk_cap,
            )?;
        } else {
            scanner.prepare_packed_lm_pipeline(&y_star, &sample_idx_u32, bps, chunk_cap)?;
        }

        let mut slot_sites = [Vec::<usize>::new(), Vec::<usize>::new()];
        slot_sites[0].reserve(chunk_cap);
        slot_sites[1].reserve(chunk_cap);
        let mut slot_counts = [0usize; 2];
        let mut slot_inflight = [false; 2];

        for (chunk_idx, block) in snp_order.chunks(chunk_cap).enumerate() {
            n_scanned = n_scanned.saturating_add(block.len());
            let slot = chunk_idx & 1usize;

            if slot_inflight[slot] {
                let m_prev = slot_counts[slot];
                let (betas, ses, mafs, n_non_missing, keep_flags) =
                    scanner.collect_packed_lm_prepared_slot(slot, m_prev)?;
                append_packed_chunk_records(
                    &slot_sites[slot],
                    &betas,
                    &ses,
                    &mafs,
                    &n_non_missing,
                    &keep_flags,
                    &sites,
                    &mut records,
                )?;
                slot_inflight[slot] = false;
                slot_counts[slot] = 0;
            }

            slot_sites[slot].clear();
            slot_sites[slot].extend_from_slice(block);
            if slot_sites[slot].is_empty() {
                continue;
            }

            let m = slot_sites[slot].len();
            let packed_dst_ptr = scanner.packed_slot_input_ptr(slot)?;
            if let Some(start_snp) = contiguous_block_start(block) {
                let src = row_source.rows_packed_range(start_snp, m)?;
                let expected = m.saturating_mul(bps);
                if src.len() != expected {
                    return Err(format!(
                        "contiguous packed bytes mismatch: start_snp={}, count={}, got={}, expected={}",
                        start_snp,
                        m,
                        src.len(),
                        expected
                    ));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), packed_dst_ptr, src.len());
                }
            } else {
                let mut offset = 0usize;
                for &snp_idx in block.iter() {
                    let row_packed = row_source.row_packed(snp_idx)?;
                    if row_packed.len() != bps {
                        return Err(format!(
                            "packed row bytes mismatch: snp_idx={}, got={}, expected={}",
                            snp_idx,
                            row_packed.len(),
                            bps
                        ));
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            row_packed.as_ptr(),
                            packed_dst_ptr.add(offset),
                            bps,
                        );
                    }
                    offset = offset.saturating_add(bps);
                }
            }

            if use_gpu_packed_lmm_fixed {
                scanner.submit_packed_lmm_fixed_prepared_slot(
                    slot,
                    n,
                    bps,
                    m,
                    cfg.maf_threshold,
                    cfg.max_missing_rate,
                    cfg.fill_missing,
                    cfg.epsilon,
                )?;
            } else {
                scanner.submit_packed_lm_prepared_slot(
                    slot,
                    n,
                    bps,
                    m,
                    cfg.maf_threshold,
                    cfg.max_missing_rate,
                    cfg.fill_missing,
                    cfg.epsilon,
                )?;
            }
            slot_counts[slot] = m;
            slot_inflight[slot] = true;
        }

        for slot in 0..2usize {
            if !slot_inflight[slot] {
                continue;
            }
            let m = slot_counts[slot];
            let (betas, ses, mafs, n_non_missing, keep_flags) =
                scanner.collect_packed_lm_prepared_slot(slot, m)?;
            append_packed_chunk_records(
                &slot_sites[slot],
                &betas,
                &ses,
                &mafs,
                &n_non_missing,
                &keep_flags,
                &sites,
                &mut records,
            )?;
            slot_inflight[slot] = false;
            slot_counts[slot] = 0;
        }
    } else {
        let mut chunk_matrix: Vec<f32> = Vec::new();
        let mut chunk_sites: Vec<usize> = Vec::new();
        let mut chunk_maf: Vec<f32> = Vec::new();
        let mut chunk_nm: Vec<usize> = Vec::new();

        chunk_matrix.reserve(chunk_cap.saturating_mul(n));
        chunk_sites.reserve(chunk_cap);
        chunk_maf.reserve(chunk_cap);
        chunk_nm.reserve(chunk_cap);

        for block in snp_order.chunks(chunk_cap) {
            let prepared: Vec<Result<Option<(usize, f32, usize, Vec<f32>)>, String>> = block
                .par_iter()
                .map(|&snp_idx| {
                    let row_packed = row_source.row_packed(snp_idx)?;
                    let mut row = vec![0.0f32; n];
                    decode_bed_row_subset_f32(row_packed, &sample_idx, &mut row)?;
                    let (keep, maf, non_missing) = qc_filter_and_impute_inplace(
                        &mut row,
                        cfg.maf_threshold,
                        cfg.max_missing_rate,
                        cfg.fill_missing,
                    )?;
                    if !keep {
                        return Ok(None);
                    }
                    if let Some(l) = l_row_major {
                        let mut rot = vec![0.0f32; n];
                        mat_vec_into_f32_row_major(l, n, &row, &mut rot)?;
                        row = rot;
                    }
                    Ok(Some((snp_idx, maf, non_missing, row)))
                })
                .collect();

            n_scanned = n_scanned.saturating_add(block.len());
            chunk_matrix.clear();
            chunk_sites.clear();
            chunk_maf.clear();
            chunk_nm.clear();

            for item in prepared.into_iter() {
                if let Some((snp_idx, maf, non_missing, row)) = item? {
                    chunk_matrix.extend_from_slice(&row);
                    chunk_sites.push(snp_idx);
                    chunk_maf.push(maf);
                    chunk_nm.push(non_missing);
                }
            }

            if !chunk_sites.is_empty() {
                flush_chunk(
                    &mut scanner,
                    &y_star,
                    &chunk_matrix,
                    n,
                    &chunk_sites,
                    &chunk_maf,
                    &chunk_nm,
                    &sites,
                    cfg.epsilon,
                    &mut records,
                )?;
            }
        }
    }

    let backend = match model {
        ScanModel::Lm => "metal-lm-v1".to_string(),
        ScanModel::LmmFixed => "metal-lmm-fixed-v1".to_string(),
    };
    Ok(GpuAssocOutput {
        n_samples: n,
        n_scanned,
        n_kept: records.len(),
        records,
        backend,
    })
}

fn run_assoc_scan_cpu<'a>(
    bed_input: BedInput<'a>,
    y: &[f32],
    sample_indices: Option<&[usize]>,
    snp_indices: Option<&[usize]>,
    cfg: &MetalAssocConfig,
    l_row_major: Option<&[f32]>,
    model: ScanModel,
) -> Result<GpuAssocOutput, String> {
    if cfg.chunk_snps == 0 {
        return Err("chunk_snps must be > 0".to_string());
    }
    if cfg.maf_threshold < 0.0 || cfg.maf_threshold > 0.5 {
        return Err("maf_threshold must be within [0, 0.5]".to_string());
    }
    if cfg.max_missing_rate < 0.0 || cfg.max_missing_rate > 1.0 {
        return Err("max_missing_rate must be within [0, 1]".to_string());
    }

    let (sites, row_source) = open_bed_input(bed_input)?;
    let n_samples_all = row_source.n_samples();
    if n_samples_all == 0 {
        return Err("BED input has zero samples".to_string());
    }

    let sample_idx = resolve_sample_indices(sample_indices, n_samples_all)?;
    let n = sample_idx.len();
    if n == 0 {
        return Err("sample selection is empty".to_string());
    }
    if y.len() != n {
        return Err(format!(
            "y length mismatch: y={}, selected_samples={}",
            y.len(),
            n
        ));
    }

    if let Some(l) = l_row_major {
        let exp = n.saturating_mul(n);
        if l.len() != exp {
            return Err(format!(
                "L shape mismatch: len={} expected n*n={}",
                l.len(),
                exp
            ));
        }
    }

    let snp_order = resolve_snp_indices(snp_indices, row_source.n_snps())?;
    if snp_order.is_empty() {
        return Ok(GpuAssocOutput {
            records: Vec::new(),
            n_samples: n,
            n_scanned: 0,
            n_kept: 0,
            backend: "cpu".to_string(),
        });
    }

    let y_star = if let Some(l) = l_row_major {
        mat_vec_f32_row_major(l, n, y)?
    } else {
        y.to_vec()
    };

    let n_scanned = snp_order.len();
    let records_or_err: Vec<Result<Option<SnpAssocRecord>, String>> = snp_order
        .par_iter()
        .map(|&snp_idx| {
            let row_packed = row_source.row_packed(snp_idx)?;
            let mut row_buf = vec![0.0f32; n];
            decode_bed_row_subset_f32(row_packed, &sample_idx, &mut row_buf)?;
            let (keep, maf, non_missing) = qc_filter_and_impute_inplace(
                &mut row_buf,
                cfg.maf_threshold,
                cfg.max_missing_rate,
                cfg.fill_missing,
            )?;
            if !keep {
                return Ok(None);
            }

            let beta_se = if let Some(l) = l_row_major {
                let mut row_rot = vec![0.0f32; n];
                mat_vec_into_f32_row_major(l, n, &row_buf, &mut row_rot)?;
                lm_beta_se_cpu_row(&y_star, &row_rot, cfg.epsilon)
            } else {
                lm_beta_se_cpu_row(&y_star, &row_buf, cfg.epsilon)
            };
            let (beta, se) = beta_se;

            let s = sites.get(snp_idx).ok_or_else(|| {
                format!(
                    "site metadata out of range: snp_idx={} >= sites={}",
                    snp_idx,
                    sites.len()
                )
            })?;
            Ok(Some(SnpAssocRecord {
                snp_index: snp_idx,
                chrom: s.chrom.clone(),
                pos: s.pos,
                ref_allele: s.ref_allele.clone(),
                alt_allele: s.alt_allele.clone(),
                beta,
                std_error: se,
                n_non_missing: non_missing,
                maf,
            }))
        })
        .collect();

    let mut records: Vec<SnpAssocRecord> = Vec::new();
    records.reserve(records_or_err.len());
    for item in records_or_err.into_iter() {
        if let Some(rec) = item? {
            records.push(rec);
        }
    }

    let backend = match model {
        ScanModel::Lm => "cpu-lm-ref-v1".to_string(),
        ScanModel::LmmFixed => "cpu-lmm-fixed-ref-v1".to_string(),
    };
    Ok(GpuAssocOutput {
        n_samples: n,
        n_scanned,
        n_kept: records.len(),
        records,
        backend,
    })
}

#[inline]
fn lm_beta_se_cpu_row(y: &[f32], x: &[f32], epsilon: f32) -> (f32, f32) {
    let n = y.len();
    if n == 0 || x.len() != n {
        return (0.0, -1.0);
    }
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    let mut syy = 0.0f64;
    for i in 0..n {
        let xv = x[i] as f64;
        let yv = y[i] as f64;
        sx += xv;
        sy += yv;
        sxx += xv * xv;
        sxy += xv * yv;
        syy += yv * yv;
    }
    let fnn = n as f64;
    let sxx_c = sxx - (sx * sx) / fnn;
    let sxy_c = sxy - (sx * sy) / fnn;
    let syy_c = syy - (sy * sy) / fnn;
    if sxx_c <= epsilon as f64 {
        return (0.0, -1.0);
    }
    let beta = sxy_c / sxx_c;
    let mut sse = syy_c - beta * sxy_c;
    if sse < 0.0 {
        sse = 0.0;
    }
    let dof = ((n as f64) - 2.0).max(1.0);
    let sigma2 = sse / dof;
    let se = (sigma2 / sxx_c.max(epsilon as f64)).sqrt();
    (beta as f32, se as f32)
}

fn compare_assoc_output(
    cpu: &GpuAssocOutput,
    gpu: &GpuAssocOutput,
) -> Result<LmConsistency, String> {
    let mut cpu_map: HashMap<usize, (f32, f32)> = HashMap::with_capacity(cpu.records.len());
    for rec in cpu.records.iter() {
        cpu_map.insert(rec.snp_index, (rec.beta, rec.std_error));
    }

    let mut matched = 0usize;
    let mut max_abs_beta = 0.0f32;
    let mut max_abs_se = 0.0f32;
    let mut sum_abs_beta = 0.0f64;
    let mut sum_abs_se = 0.0f64;

    for rec in gpu.records.iter() {
        if let Some(&(b0, s0)) = cpu_map.get(&rec.snp_index) {
            let db = (b0 - rec.beta).abs();
            let ds = (s0 - rec.std_error).abs();
            if db > max_abs_beta {
                max_abs_beta = db;
            }
            if ds > max_abs_se {
                max_abs_se = ds;
            }
            sum_abs_beta += db as f64;
            sum_abs_se += ds as f64;
            matched = matched.saturating_add(1);
        }
    }
    if matched == 0 {
        return Err("no overlapping SNP records between CPU and Metal outputs".to_string());
    }
    Ok(LmConsistency {
        matched_snps: matched,
        max_abs_beta,
        max_abs_se,
        mean_abs_beta: (sum_abs_beta / matched as f64) as f32,
        mean_abs_se: (sum_abs_se / matched as f64) as f32,
    })
}

fn append_packed_chunk_records(
    chunk_sites: &[usize],
    betas: &[f32],
    ses: &[f32],
    mafs: &[f32],
    n_non_missing: &[u32],
    keep_flags: &[u32],
    sites: &[BedSiteInfo],
    out: &mut Vec<SnpAssocRecord>,
) -> Result<(), String> {
    let m = chunk_sites.len();
    if betas.len() != m
        || ses.len() != m
        || mafs.len() != m
        || n_non_missing.len() != m
        || keep_flags.len() != m
    {
        return Err(format!(
            "metal packed output mismatch: betas={}, ses={}, mafs={}, nm={}, keep={}, expected={}",
            betas.len(),
            ses.len(),
            mafs.len(),
            n_non_missing.len(),
            keep_flags.len(),
            m
        ));
    }

    for j in 0..m {
        if keep_flags[j] == 0 {
            continue;
        }
        let snp_idx = chunk_sites[j];
        if snp_idx >= sites.len() {
            return Err(format!(
                "site metadata out of range: snp_idx={} >= sites={}",
                snp_idx,
                sites.len()
            ));
        }
        let s = &sites[snp_idx];
        out.push(SnpAssocRecord {
            snp_index: snp_idx,
            chrom: s.chrom.clone(),
            pos: s.pos,
            ref_allele: s.ref_allele.clone(),
            alt_allele: s.alt_allele.clone(),
            beta: betas[j],
            std_error: ses[j],
            n_non_missing: n_non_missing[j] as usize,
            maf: mafs[j],
        });
    }
    Ok(())
}

fn flush_chunk(
    scanner: &mut MetalLmScanner,
    y_star: &[f32],
    chunk_matrix_snp_major: &[f32],
    n_samples: usize,
    chunk_sites: &[usize],
    chunk_maf: &[f32],
    chunk_nm: &[usize],
    sites: &[BedSiteInfo],
    epsilon: f32,
    out: &mut Vec<SnpAssocRecord>,
) -> Result<(), String> {
    let m = chunk_sites.len();
    if m == 0 {
        return Ok(());
    }
    let expected = m.saturating_mul(n_samples);
    if chunk_matrix_snp_major.len() != expected {
        return Err(format!(
            "chunk matrix size mismatch: got {}, expected {}",
            chunk_matrix_snp_major.len(),
            expected
        ));
    }

    let (betas, ses) = scanner.run(y_star, chunk_matrix_snp_major, n_samples, m, epsilon)?;
    if betas.len() != m || ses.len() != m {
        return Err(format!(
            "metal output mismatch: betas={}, ses={}, expected={}",
            betas.len(),
            ses.len(),
            m
        ));
    }

    for j in 0..m {
        let snp_idx = chunk_sites[j];
        if snp_idx >= sites.len() {
            return Err(format!(
                "site metadata out of range: snp_idx={} >= sites={}",
                snp_idx,
                sites.len()
            ));
        }
        let s = &sites[snp_idx];
        out.push(SnpAssocRecord {
            snp_index: snp_idx,
            chrom: s.chrom.clone(),
            pos: s.pos,
            ref_allele: s.ref_allele.clone(),
            alt_allele: s.alt_allele.clone(),
            beta: betas[j],
            std_error: ses[j],
            n_non_missing: chunk_nm[j],
            maf: chunk_maf[j],
        });
    }
    Ok(())
}

fn open_bed_input<'a>(
    bed_input: BedInput<'a>,
) -> Result<(Vec<BedSiteInfo>, BedRowSource<'a>), String> {
    match bed_input {
        BedInput::Prefix(prefix_raw) => {
            let prefix = normalize_plink_prefix(prefix_raw);
            let fam_ids = read_fam_ids(&prefix)?;
            let n_samples = fam_ids.len();
            if n_samples == 0 {
                return Err("FAM has zero samples".to_string());
            }
            let sites = read_bim_sites(&prefix)?;
            let n_snps = sites.len();
            if n_snps == 0 {
                return Err("BIM has zero SNPs".to_string());
            }

            let bed_path = format!("{prefix}.bed");
            let file = File::open(&bed_path)
                .map_err(|e| format!("failed to open BED file {bed_path}: {e}"))?;
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| format!("failed to mmap BED file {bed_path}: {e}"))?;
            if mmap.len() < 3 {
                return Err(format!("BED file too small: {bed_path}"));
            }
            if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
                return Err(format!(
                    "{bed_path}: only SNP-major BED (0x6C 0x1B 0x01) is supported"
                ));
            }
            let bps = n_samples.div_ceil(4);
            let payload = mmap.len().saturating_sub(3);
            let expected = n_snps.saturating_mul(bps);
            if payload != expected {
                return Err(format!(
                    "{bed_path}: payload size mismatch: got {payload}, expected {expected} (n_snps={n_snps}, bps={bps})"
                ));
            }
            Ok((
                sites,
                BedRowSource::Prefix(PrefixBedData {
                    mmap,
                    n_samples,
                    n_snps,
                    bytes_per_snp: bps,
                }),
            ))
        }
        BedInput::Packed(v) => {
            if v.n_samples == 0 {
                return Err("packed input has zero samples".to_string());
            }
            if v.n_snps == 0 {
                return Err("packed input has zero SNPs".to_string());
            }
            let bps = v.n_samples.div_ceil(4);
            let expected = v.n_snps.saturating_mul(bps);
            if v.packed_rows.len() != expected {
                return Err(format!(
                    "packed bytes mismatch: got {}, expected {}",
                    v.packed_rows.len(),
                    expected
                ));
            }
            let sites = vec![
                BedSiteInfo {
                    chrom: ".".to_string(),
                    pos: -1,
                    ref_allele: "N".to_string(),
                    alt_allele: "N".to_string()
                };
                v.n_snps
            ];
            Ok((sites, BedRowSource::Packed(v)))
        }
    }
}

fn resolve_sample_indices(
    sample_indices: Option<&[usize]>,
    n_samples: usize,
) -> Result<Vec<usize>, String> {
    match sample_indices {
        None => Ok((0..n_samples).collect()),
        Some(v) => {
            if v.is_empty() {
                return Err("sample_indices is empty".to_string());
            }
            let mut seen = vec![false; n_samples];
            let mut out = Vec::with_capacity(v.len());
            for &idx in v.iter() {
                if idx >= n_samples {
                    return Err(format!(
                        "sample index out of range: {} >= {}",
                        idx, n_samples
                    ));
                }
                if seen[idx] {
                    return Err(format!("duplicate sample index: {idx}"));
                }
                seen[idx] = true;
                out.push(idx);
            }
            Ok(out)
        }
    }
}

fn resolve_snp_indices(snp_indices: Option<&[usize]>, n_snps: usize) -> Result<Vec<usize>, String> {
    match snp_indices {
        None => Ok((0..n_snps).collect()),
        Some(v) => {
            if v.is_empty() {
                return Err("snp_indices is empty".to_string());
            }
            let mut out = Vec::with_capacity(v.len());
            for &idx in v.iter() {
                if idx >= n_snps {
                    return Err(format!("snp index out of range: {} >= {}", idx, n_snps));
                }
                out.push(idx);
            }
            Ok(out)
        }
    }
}

#[inline]
fn contiguous_block_start(block: &[usize]) -> Option<usize> {
    let &start = block.first()?;
    for (i, &v) in block.iter().enumerate().skip(1) {
        if v != start.saturating_add(i) {
            return None;
        }
    }
    Some(start)
}

fn normalize_plink_prefix(p: &str) -> String {
    let s = p.trim();
    let low = s.to_ascii_lowercase();
    if low.ends_with(".bed") || low.ends_with(".bim") || low.ends_with(".fam") {
        return s[..s.len() - 4].to_string();
    }
    s.to_string()
}

fn read_fam_ids(prefix: &str) -> Result<Vec<String>, String> {
    let p = format!("{prefix}.fam");
    let fh = File::open(&p).map_err(|e| format!("failed to open FAM {p}: {e}"))?;
    let br = BufReader::new(fh);
    let mut ids = Vec::<String>::new();
    for line in br.lines() {
        let line = line.map_err(|e| format!("failed reading FAM {p}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.is_empty() {
            continue;
        }
        let id = if cols.len() > 1 { cols[1] } else { cols[0] };
        ids.push(id.to_string());
    }
    Ok(ids)
}

fn read_bim_sites(prefix: &str) -> Result<Vec<BedSiteInfo>, String> {
    let p = format!("{prefix}.bim");
    let fh = File::open(&p).map_err(|e| format!("failed to open BIM {p}: {e}"))?;
    let br = BufReader::new(fh);
    let mut out: Vec<BedSiteInfo> = Vec::new();
    for line in br.lines() {
        let line = line.map_err(|e| format!("failed reading BIM {p}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 6 {
            return Err(format!("invalid BIM line (need >= 6 columns): {line}"));
        }
        let chrom = cols[0].to_string();
        let pos = cols[3].parse::<i32>().unwrap_or(-1);
        let ref_allele = cols[4].to_string();
        let alt_allele = cols[5].to_string();
        out.push(BedSiteInfo {
            chrom,
            pos,
            ref_allele,
            alt_allele,
        });
    }
    Ok(out)
}

#[inline]
fn decode_bed_row_subset_f32(
    row_packed: &[u8],
    sample_indices: &[usize],
    out: &mut [f32],
) -> Result<(), String> {
    if out.len() != sample_indices.len() {
        return Err(format!(
            "decode output length mismatch: out={}, sample_indices={}",
            out.len(),
            sample_indices.len()
        ));
    }
    for (k, &si) in sample_indices.iter().enumerate() {
        let byte_idx = si >> 2;
        let pair = si & 3usize;
        if byte_idx >= row_packed.len() {
            return Err(format!(
                "packed row too short: byte_idx={}, row_len={}",
                byte_idx,
                row_packed.len()
            ));
        }
        let b = row_packed[byte_idx];
        let code = (b >> (pair * 2)) & 0b11;
        out[k] = match code {
            0b00 => 0.0,   // hom ref
            0b10 => 1.0,   // het
            0b11 => 2.0,   // hom alt
            _ => f32::NAN, // missing
        };
    }
    Ok(())
}

#[inline]
fn qc_filter_and_impute_inplace(
    row: &mut [f32],
    maf_threshold: f32,
    max_missing_rate: f32,
    fill_missing: bool,
) -> Result<(bool, f32, usize), String> {
    if row.is_empty() {
        return Ok((false, 0.0, 0));
    }
    let mut missing = 0usize;
    let mut alt_sum = 0.0f64;
    for &v in row.iter() {
        if !v.is_finite() || v < 0.0 {
            missing = missing.saturating_add(1);
        } else {
            alt_sum += v as f64;
        }
    }
    let n = row.len();
    let non_missing = n.saturating_sub(missing);
    if non_missing == 0 {
        return Ok((false, 0.0, 0));
    }
    let miss_rate = missing as f32 / n as f32;
    if miss_rate > max_missing_rate {
        return Ok((false, 0.0, non_missing));
    }
    let p = (alt_sum / (2.0f64 * non_missing as f64)) as f32;
    let maf = p.min(1.0 - p);
    if maf < maf_threshold {
        return Ok((false, maf, non_missing));
    }
    if fill_missing && missing > 0 {
        let mean = (2.0f64 * p as f64) as f32;
        for v in row.iter_mut() {
            if !v.is_finite() || *v < 0.0 {
                *v = mean;
            }
        }
    } else if missing > 0 {
        for v in row.iter_mut() {
            if !v.is_finite() || *v < 0.0 {
                *v = 0.0;
            }
        }
    }
    Ok((true, maf, non_missing))
}

#[inline]
fn mat_vec_f32_row_major(mat: &[f32], n: usize, x: &[f32]) -> Result<Vec<f32>, String> {
    if mat.len() != n.saturating_mul(n) {
        return Err(format!(
            "mat_vec shape mismatch: mat_len={}, expected={}",
            mat.len(),
            n.saturating_mul(n)
        ));
    }
    if x.len() != n {
        return Err(format!("mat_vec vector mismatch: x={}, n={}", x.len(), n));
    }
    let mut out = vec![0.0f32; n];
    mat_vec_into_f32_row_major(mat, n, x, &mut out)?;
    Ok(out)
}

#[inline]
fn mat_vec_into_f32_row_major(
    mat: &[f32],
    n: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), String> {
    if mat.len() != n.saturating_mul(n) || x.len() != n || out.len() != n {
        return Err(format!(
            "mat_vec_into shape mismatch: mat={}, x={}, out={}, n={}",
            mat.len(),
            x.len(),
            out.len(),
            n
        ));
    }
    for r in 0..n {
        let row = &mat[r * n..(r + 1) * n];
        let mut acc = 0.0f64;
        for c in 0..n {
            acc += row[c] as f64 * x[c] as f64;
        }
        out[r] = acc as f32;
    }
    Ok(())
}

#[inline]
fn rss_delta_raw(before: Option<i64>, after: Option<i64>) -> Option<i64> {
    match (before, after) {
        (Some(b), Some(a)) => Some(a.saturating_sub(b)),
        _ => None,
    }
}

#[inline]
fn peak_rss_unit() -> &'static str {
    #[cfg(target_os = "linux")]
    {
        "KiB"
    }
    #[cfg(target_os = "macos")]
    {
        "bytes"
    }
    #[cfg(target_os = "windows")]
    {
        "unknown"
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        "unknown"
    }
}

#[allow(clippy::unnecessary_cast)]
fn process_peak_rss_raw() -> Option<i64> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let mut ru = std::mem::MaybeUninit::<libc::rusage>::uninit();
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) };
        if rc != 0 {
            return None;
        }
        let ru = unsafe { ru.assume_init() };
        Some(ru.ru_maxrss as i64)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod metal_impl {
    use super::*;
    use metal::{
        Buffer, CommandBuffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
        MTLSize,
    };
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::ptr;

    const KERNEL_SRC: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    struct AssocParams {
        uint n_samples;
        uint n_snps;
        float epsilon;
    };

    struct PackedAssocParams {
        uint n_samples;
        uint n_snps;
        uint bytes_per_snp;
        float maf_threshold;
        float max_missing_rate;
        float epsilon;
        uint fill_missing;
    };

    kernel void snp_lm_scanner(
        device const float* y      [[ buffer(0) ]],
        device const float* x       [[ buffer(1) ]],   // SNP-major, row stride = n_samples
        device float* betas         [[ buffer(2) ]],
        device float* ses           [[ buffer(3) ]],
        constant AssocParams& prm   [[ buffer(4) ]],
        uint tid                    [[ thread_position_in_grid ]]
    ) {
        if (tid >= prm.n_snps) return;
        uint N = prm.n_samples;
        uint off = tid * N;

        float sx = 0.0f;
        float sy = 0.0f;
        float sxx = 0.0f;
        float sxy = 0.0f;
        float syy = 0.0f;

        for (uint i = 0; i < N; i++) {
            float xv = x[off + i];
            float yv = y[i];
            sx += xv;
            sy += yv;
            sxx += xv * xv;
            sxy += xv * yv;
            syy += yv * yv;
        }

        float fn = float(N);
        float Sxx = sxx - (sx * sx) / fn;
        float Sxy = sxy - (sx * sy) / fn;
        float Syy = syy - (sy * sy) / fn;

        if (Sxx <= prm.epsilon) {
            betas[tid] = 0.0f;
            ses[tid] = -1.0f;
            return;
        }

        float beta = Sxy / Sxx;
        float sse = Syy - beta * Sxy;
        if (sse < 0.0f) sse = 0.0f;
        float dof = max(fn - 2.0f, 1.0f);
        float sigma2 = sse / dof;
        float se = sqrt(sigma2 / max(Sxx, prm.epsilon));

        betas[tid] = beta;
        ses[tid] = se;
    }

    kernel void packed_bed_lm_scanner(
        device const float* y            [[ buffer(0) ]],
        device const uchar* packed_rows  [[ buffer(1) ]],   // SNP-major 2-bit BED rows
        device const uint* sample_idx    [[ buffer(2) ]],   // selected samples in original order
        device float* betas              [[ buffer(3) ]],
        device float* ses                [[ buffer(4) ]],
        device float* mafs               [[ buffer(5) ]],
        device uint* n_non_missing       [[ buffer(6) ]],
        device uint* keep_flags          [[ buffer(7) ]],
        constant PackedAssocParams& prm  [[ buffer(8) ]],
        uint tid                         [[ thread_position_in_grid ]]
    ) {
        if (tid >= prm.n_snps) return;

        uint N = prm.n_samples;
        uint row_off = tid * prm.bytes_per_snp;
        float alt_sum = 0.0f;
        uint missing = 0u;
        float sy = 0.0f;
        float syy = 0.0f;
        float sy_missing = 0.0f;
        float sxx_nonmissing = 0.0f;
        float sxy_nonmissing = 0.0f;

        for (uint i = 0; i < N; i++) {
            uint si = sample_idx[i];
            uint byte_idx = si >> 2;
            uint pair = si & 3u;
            uchar b = packed_rows[row_off + byte_idx];
            uint code = (uint(b) >> (pair * 2u)) & 0x3u;
            float yv = y[i];
            sy += yv;
            syy += yv * yv;
            if (code == 1u) {
                missing += 1u;
                sy_missing += yv;
            } else if (code == 2u) {
                alt_sum += 1.0f;
                sxx_nonmissing += 1.0f;
                sxy_nonmissing += yv;
            } else if (code == 3u) {
                alt_sum += 2.0f;
                sxx_nonmissing += 4.0f;
                sxy_nonmissing += 2.0f * yv;
            }
        }

        uint non_missing = N - missing;
        n_non_missing[tid] = non_missing;
        keep_flags[tid] = 0u;
        betas[tid] = 0.0f;
        ses[tid] = -1.0f;
        mafs[tid] = 0.0f;

        if (non_missing == 0u) {
            return;
        }

        float miss_rate = float(missing) / float(N);
        if (miss_rate > prm.max_missing_rate) {
            return;
        }

        float p = alt_sum / (2.0f * float(non_missing));
        float maf = min(p, 1.0f - p);
        mafs[tid] = maf;
        if (maf < prm.maf_threshold) {
            return;
        }

        float impute = prm.fill_missing != 0u ? (alt_sum / float(non_missing)) : 0.0f;
        float sx = alt_sum + float(missing) * impute;
        float sxx = sxx_nonmissing + float(missing) * impute * impute;
        float sxy = sxy_nonmissing + sy_missing * impute;

        float fn = float(N);
        float Sxx = sxx - (sx * sx) / fn;
        float Sxy = sxy - (sx * sy) / fn;
        float Syy = syy - (sy * sy) / fn;
        if (Sxx <= prm.epsilon) {
            return;
        }

        float beta = Sxy / Sxx;
        float sse = Syy - beta * Sxy;
        if (sse < 0.0f) sse = 0.0f;
        float dof = max(fn - 2.0f, 1.0f);
        float sigma2 = sse / dof;
        float se = sqrt(sigma2 / max(Sxx, prm.epsilon));

        betas[tid] = beta;
        ses[tid] = se;
        keep_flags[tid] = 1u;
    }

    // Decode packed BED rows + QC/impute into dense SNP-major rows.
    kernel void packed_bed_decode_qc(
        device const uchar* packed_rows  [[ buffer(0) ]],
        device const uint* sample_idx    [[ buffer(1) ]],
        device float* x_imputed          [[ buffer(2) ]],   // SNP-major dense rows
        device float* mafs               [[ buffer(3) ]],
        device uint* n_non_missing       [[ buffer(4) ]],
        device uint* keep_flags          [[ buffer(5) ]],
        constant PackedAssocParams& prm  [[ buffer(6) ]],
        uint tid                         [[ thread_position_in_grid ]]
    ) {
        if (tid >= prm.n_snps) return;

        uint N = prm.n_samples;
        uint row_off_packed = tid * prm.bytes_per_snp;
        uint row_off_dense = tid * N;
        float alt_sum = 0.0f;
        uint missing = 0u;

        for (uint i = 0; i < N; i++) {
            uint si = sample_idx[i];
            uint byte_idx = si >> 2;
            uint pair = si & 3u;
            uchar b = packed_rows[row_off_packed + byte_idx];
            uint code = (uint(b) >> (pair * 2u)) & 0x3u;
            if (code == 1u) {
                missing += 1u;
            } else if (code == 2u) {
                alt_sum += 1.0f;
            } else if (code == 3u) {
                alt_sum += 2.0f;
            }
        }

        uint non_missing = N - missing;
        n_non_missing[tid] = non_missing;
        keep_flags[tid] = 0u;
        mafs[tid] = 0.0f;
        if (non_missing == 0u) {
            return;
        }

        float miss_rate = float(missing) / float(N);
        if (miss_rate > prm.max_missing_rate) {
            return;
        }

        float p = alt_sum / (2.0f * float(non_missing));
        float maf = min(p, 1.0f - p);
        mafs[tid] = maf;
        if (maf < prm.maf_threshold) {
            return;
        }

        float impute = prm.fill_missing != 0u ? (alt_sum / float(non_missing)) : 0.0f;
        for (uint i = 0; i < N; i++) {
            uint si = sample_idx[i];
            uint byte_idx = si >> 2;
            uint pair = si & 3u;
            uchar b = packed_rows[row_off_packed + byte_idx];
            uint code = (uint(b) >> (pair * 2u)) & 0x3u;

            float xv = 0.0f;
            if (code == 2u) {
                xv = 1.0f;
            } else if (code == 3u) {
                xv = 2.0f;
            } else if (code == 1u) {
                xv = impute;
            }
            x_imputed[row_off_dense + i] = xv;
        }
        keep_flags[tid] = 1u;
    }

    // Fixed-variance LMM scan: rotate each decoded SNP row with dense L and run OLS on (x*, y*).
    kernel void packed_bed_lmm_fixed_scanner(
        device const float* y_star        [[ buffer(0) ]],
        device const float* l_row_major   [[ buffer(1) ]],   // (n x n)
        device const float* x_imputed     [[ buffer(2) ]],   // SNP-major dense rows
        device const uint* keep_flags     [[ buffer(3) ]],
        device float* betas               [[ buffer(4) ]],
        device float* ses                 [[ buffer(5) ]],
        constant AssocParams& prm         [[ buffer(6) ]],
        uint tid                          [[ thread_position_in_grid ]]
    ) {
        if (tid >= prm.n_snps) return;
        if (keep_flags[tid] == 0u) {
            betas[tid] = 0.0f;
            ses[tid] = -1.0f;
            return;
        }

        uint N = prm.n_samples;
        uint off = tid * N;

        float sx = 0.0f;
        float sy = 0.0f;
        float sxx = 0.0f;
        float sxy = 0.0f;
        float syy = 0.0f;

        for (uint r = 0; r < N; r++) {
            float xv = 0.0f;
            uint l_row_off = r * N;
            for (uint c = 0; c < N; c++) {
                xv += l_row_major[l_row_off + c] * x_imputed[off + c];
            }
            float yv = y_star[r];
            sx += xv;
            sy += yv;
            sxx += xv * xv;
            sxy += xv * yv;
            syy += yv * yv;
        }

        float fn = float(N);
        float Sxx = sxx - (sx * sx) / fn;
        float Sxy = sxy - (sx * sy) / fn;
        float Syy = syy - (sy * sy) / fn;
        if (Sxx <= prm.epsilon) {
            betas[tid] = 0.0f;
            ses[tid] = -1.0f;
            return;
        }

        float beta = Sxy / Sxx;
        float sse = Syy - beta * Sxy;
        if (sse < 0.0f) sse = 0.0f;
        float dof = max(fn - 2.0f, 1.0f);
        float sigma2 = sse / dof;
        float se = sqrt(sigma2 / max(Sxx, prm.epsilon));

        betas[tid] = beta;
        ses[tid] = se;
    }
    "#;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct AssocParams {
        n_samples: u32,
        n_snps: u32,
        epsilon: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PackedAssocParams {
        n_samples: u32,
        n_snps: u32,
        bytes_per_snp: u32,
        maf_threshold: f32,
        max_missing_rate: f32,
        epsilon: f32,
        fill_missing: u32,
    }

    const PACKED_PIPELINE_SLOTS: usize = 2;

    struct PackedSlotBuffers {
        packed_rows_buf: Option<Buffer>,
        decoded_rows_buf: Option<Buffer>,
        beta_buf: Option<Buffer>,
        se_buf: Option<Buffer>,
        maf_buf: Option<Buffer>,
        n_non_missing_buf: Option<Buffer>,
        keep_flag_buf: Option<Buffer>,
        inflight_cmd: Option<CommandBuffer>,
    }

    impl PackedSlotBuffers {
        fn new() -> Self {
            Self {
                packed_rows_buf: None,
                decoded_rows_buf: None,
                beta_buf: None,
                se_buf: None,
                maf_buf: None,
                n_non_missing_buf: None,
                keep_flag_buf: None,
                inflight_cmd: None,
            }
        }
    }

    pub struct MetalLmScanner {
        device: Device,
        pipeline_dense: ComputePipelineState,
        pipeline_packed: ComputePipelineState,
        pipeline_packed_decode_qc: ComputePipelineState,
        pipeline_packed_lmm_fixed: ComputePipelineState,
        queue: metal::CommandQueue,
        threads_per_threadgroup: usize,
        y_buf: Option<Buffer>,
        y_capacity_samples: usize,
        l_buf: Option<Buffer>,
        l_capacity_samples: usize,
        x_buf: Option<Buffer>,
        beta_buf: Option<Buffer>,
        se_buf: Option<Buffer>,
        dense_capacity: (usize, usize), // (n_samples, max_snps)
        sample_idx_buf: Option<Buffer>,
        packed_slots: [PackedSlotBuffers; PACKED_PIPELINE_SLOTS],
        packed_capacity: (usize, usize, usize), // (n_samples, max_snps, bytes_per_snp)
    }

    impl MetalLmScanner {
        pub fn new(threads_per_threadgroup: usize) -> Result<Self, String> {
            let device = Device::system_default()
                .ok_or_else(|| "Metal is not available on this device".to_string())?;
            let queue = device.new_command_queue();
            let options = CompileOptions::new();
            let lib = device
                .new_library_with_source(KERNEL_SRC, &options)
                .map_err(|e| format!("failed to compile Metal shader: {e}"))?;
            let fun_dense = lib
                .get_function("snp_lm_scanner", None)
                .map_err(|e| format!("failed to load Metal kernel snp_lm_scanner: {e}"))?;
            let pipeline_dense = device
                .new_compute_pipeline_state_with_function(&fun_dense)
                .map_err(|e| format!("failed to create Metal pipeline: {e}"))?;
            let fun_packed = lib
                .get_function("packed_bed_lm_scanner", None)
                .map_err(|e| format!("failed to load Metal kernel packed_bed_lm_scanner: {e}"))?;
            let pipeline_packed = device
                .new_compute_pipeline_state_with_function(&fun_packed)
                .map_err(|e| format!("failed to create packed Metal pipeline: {e}"))?;
            let fun_packed_decode_qc = lib
                .get_function("packed_bed_decode_qc", None)
                .map_err(|e| format!("failed to load Metal kernel packed_bed_decode_qc: {e}"))?;
            let pipeline_packed_decode_qc = device
                .new_compute_pipeline_state_with_function(&fun_packed_decode_qc)
                .map_err(|e| format!("failed to create packed decode Metal pipeline: {e}"))?;
            let fun_packed_lmm_fixed = lib
                .get_function("packed_bed_lmm_fixed_scanner", None)
                .map_err(|e| {
                    format!("failed to load Metal kernel packed_bed_lmm_fixed_scanner: {e}")
                })?;
            let pipeline_packed_lmm_fixed = device
                .new_compute_pipeline_state_with_function(&fun_packed_lmm_fixed)
                .map_err(|e| format!("failed to create packed fixed-LMM Metal pipeline: {e}"))?;
            Ok(Self {
                device,
                pipeline_dense,
                pipeline_packed,
                pipeline_packed_decode_qc,
                pipeline_packed_lmm_fixed,
                queue,
                threads_per_threadgroup: threads_per_threadgroup.max(1),
                y_buf: None,
                y_capacity_samples: 0,
                l_buf: None,
                l_capacity_samples: 0,
                x_buf: None,
                beta_buf: None,
                se_buf: None,
                dense_capacity: (0, 0),
                sample_idx_buf: None,
                packed_slots: [PackedSlotBuffers::new(), PackedSlotBuffers::new()],
                packed_capacity: (0, 0, 0),
            })
        }

        fn ensure_y_buffer(&mut self, n_samples: usize) {
            if self.y_capacity_samples >= n_samples {
                return;
            }
            let y_bytes = (n_samples.saturating_mul(size_of::<f32>())) as u64;
            self.y_buf = Some(
                self.device
                    .new_buffer(y_bytes, MTLResourceOptions::StorageModeShared),
            );
            self.y_capacity_samples = n_samples;
        }

        fn ensure_l_buffer(&mut self, n_samples: usize) {
            if self.l_capacity_samples == n_samples {
                return;
            }
            let l_bytes = (n_samples
                .saturating_mul(n_samples)
                .saturating_mul(size_of::<f32>())) as u64;
            self.l_buf = Some(
                self.device
                    .new_buffer(l_bytes, MTLResourceOptions::StorageModeShared),
            );
            self.l_capacity_samples = n_samples;
        }

        fn ensure_dense_buffers(&mut self, n_samples: usize, max_snps: usize) {
            self.ensure_y_buffer(n_samples);
            if self.dense_capacity.0 >= n_samples && self.dense_capacity.1 >= max_snps {
                return;
            }
            let x_bytes = (n_samples
                .saturating_mul(max_snps)
                .saturating_mul(size_of::<f32>())) as u64;
            let out_bytes = (max_snps.saturating_mul(size_of::<f32>())) as u64;

            self.x_buf = Some(
                self.device
                    .new_buffer(x_bytes, MTLResourceOptions::StorageModeShared),
            );
            self.beta_buf = Some(
                self.device
                    .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared),
            );
            self.se_buf = Some(
                self.device
                    .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared),
            );
            self.dense_capacity = (n_samples, max_snps);
        }

        fn ensure_packed_buffers(
            &mut self,
            n_samples: usize,
            max_snps: usize,
            bytes_per_snp: usize,
        ) -> Result<(), String> {
            self.ensure_y_buffer(n_samples);
            if self.packed_capacity.0 >= n_samples
                && self.packed_capacity.1 >= max_snps
                && self.packed_capacity.2 == bytes_per_snp
            {
                return Ok(());
            }

            for (slot_idx, slot) in self.packed_slots.iter().enumerate() {
                if slot.inflight_cmd.is_some() {
                    return Err(format!(
                        "cannot reallocate packed buffers while slot {slot_idx} is in flight"
                    ));
                }
            }

            let packed_bytes = (max_snps.saturating_mul(bytes_per_snp)) as u64;
            let decoded_bytes = (max_snps
                .saturating_mul(n_samples)
                .saturating_mul(size_of::<f32>())) as u64;
            let sample_idx_bytes = (n_samples.saturating_mul(size_of::<u32>())) as u64;
            let out_f32_bytes = (max_snps.saturating_mul(size_of::<f32>())) as u64;
            let out_u32_bytes = (max_snps.saturating_mul(size_of::<u32>())) as u64;

            self.sample_idx_buf = Some(
                self.device
                    .new_buffer(sample_idx_bytes, MTLResourceOptions::StorageModeShared),
            );
            for slot in self.packed_slots.iter_mut() {
                slot.packed_rows_buf = Some(
                    self.device
                        .new_buffer(packed_bytes, MTLResourceOptions::StorageModeShared),
                );
                slot.decoded_rows_buf = Some(
                    self.device
                        .new_buffer(decoded_bytes, MTLResourceOptions::StorageModeShared),
                );
                slot.beta_buf = Some(
                    self.device
                        .new_buffer(out_f32_bytes, MTLResourceOptions::StorageModeShared),
                );
                slot.se_buf = Some(
                    self.device
                        .new_buffer(out_f32_bytes, MTLResourceOptions::StorageModeShared),
                );
                slot.maf_buf = Some(
                    self.device
                        .new_buffer(out_f32_bytes, MTLResourceOptions::StorageModeShared),
                );
                slot.n_non_missing_buf = Some(
                    self.device
                        .new_buffer(out_u32_bytes, MTLResourceOptions::StorageModeShared),
                );
                slot.keep_flag_buf = Some(
                    self.device
                        .new_buffer(out_u32_bytes, MTLResourceOptions::StorageModeShared),
                );
            }
            self.packed_capacity = (n_samples, max_snps, bytes_per_snp);
            Ok(())
        }

        pub fn run(
            &mut self,
            y: &[f32],
            x_snp_major: &[f32],
            n_samples: usize,
            n_snps: usize,
            epsilon: f32,
        ) -> Result<(Vec<f32>, Vec<f32>), String> {
            if n_samples == 0 || n_snps == 0 {
                return Ok((Vec::new(), Vec::new()));
            }
            let expected = n_samples.saturating_mul(n_snps);
            if x_snp_major.len() != expected {
                return Err(format!(
                    "metal input shape mismatch: x_len={}, expected={}",
                    x_snp_major.len(),
                    expected
                ));
            }
            if y.len() != n_samples {
                return Err(format!(
                    "metal input y mismatch: y_len={}, n_samples={}",
                    y.len(),
                    n_samples
                ));
            }

            self.ensure_dense_buffers(n_samples, n_snps);
            let y_buf = self
                .y_buf
                .as_ref()
                .ok_or_else(|| "internal error: y buffer not allocated".to_string())?;
            let x_buf = self
                .x_buf
                .as_ref()
                .ok_or_else(|| "internal error: x buffer not allocated".to_string())?;
            let beta_buf = self
                .beta_buf
                .as_ref()
                .ok_or_else(|| "internal error: beta buffer not allocated".to_string())?;
            let se_buf = self
                .se_buf
                .as_ref()
                .ok_or_else(|| "internal error: se buffer not allocated".to_string())?;

            unsafe {
                ptr::copy_nonoverlapping(y.as_ptr(), y_buf.contents() as *mut f32, n_samples);
                ptr::copy_nonoverlapping(
                    x_snp_major.as_ptr(),
                    x_buf.contents() as *mut f32,
                    x_snp_major.len(),
                );
            }

            let prm = AssocParams {
                n_samples: n_samples as u32,
                n_snps: n_snps as u32,
                epsilon,
            };

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_dense);
            enc.set_buffer(0, Some(y_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(beta_buf), 0);
            enc.set_buffer(3, Some(se_buf), 0);
            enc.set_bytes(
                4,
                size_of::<AssocParams>() as u64,
                &prm as *const AssocParams as *const c_void,
            );

            let tg = min(
                self.threads_per_threadgroup as u64,
                self.pipeline_dense.max_total_threads_per_threadgroup(),
            )
            .max(1);
            let grid = MTLSize::new(n_snps as u64, 1, 1);
            let tgs = MTLSize::new(tg, 1, 1);
            enc.dispatch_threads(grid, tgs);
            enc.end_encoding();

            cmd.commit();
            cmd.wait_until_completed();

            let mut betas = vec![0.0f32; n_snps];
            let mut ses = vec![0.0f32; n_snps];
            unsafe {
                ptr::copy_nonoverlapping(
                    beta_buf.contents() as *const f32,
                    betas.as_mut_ptr(),
                    n_snps,
                );
                ptr::copy_nonoverlapping(se_buf.contents() as *const f32, ses.as_mut_ptr(), n_snps);
            }
            Ok((betas, ses))
        }

        fn packed_slot_index(slot: usize) -> Result<usize, String> {
            if slot >= PACKED_PIPELINE_SLOTS {
                return Err(format!(
                    "packed slot out of range: slot={}, slots={}",
                    slot, PACKED_PIPELINE_SLOTS
                ));
            }
            Ok(slot)
        }

        pub fn prepare_packed_lm_pipeline(
            &mut self,
            y: &[f32],
            sample_indices: &[u32],
            bytes_per_snp: usize,
            max_snps: usize,
        ) -> Result<(), String> {
            let n_samples = sample_indices.len();
            if n_samples == 0 || max_snps == 0 {
                return Err("metal packed prepare requires n_samples>0 and max_snps>0".to_string());
            }
            if y.len() != n_samples {
                return Err(format!(
                    "metal packed y mismatch: y_len={}, n_samples={}",
                    y.len(),
                    n_samples
                ));
            }
            if bytes_per_snp == 0 {
                return Err("metal packed bytes_per_snp must be > 0".to_string());
            }

            self.ensure_packed_buffers(n_samples, max_snps, bytes_per_snp)?;
            let y_buf = self
                .y_buf
                .as_ref()
                .ok_or_else(|| "internal error: packed y buffer not allocated".to_string())?;
            let sample_idx_buf = self.sample_idx_buf.as_ref().ok_or_else(|| {
                "internal error: packed sample index buffer not allocated".to_string()
            })?;
            for (slot_idx, slot_state) in self.packed_slots.iter().enumerate() {
                if slot_state.inflight_cmd.is_some() {
                    return Err(format!("packed slot {} is still in flight", slot_idx));
                }
            }

            unsafe {
                ptr::copy_nonoverlapping(y.as_ptr(), y_buf.contents() as *mut f32, y.len());
                ptr::copy_nonoverlapping(
                    sample_indices.as_ptr(),
                    sample_idx_buf.contents() as *mut u32,
                    sample_indices.len(),
                );
            }
            Ok(())
        }

        pub fn prepare_packed_lmm_fixed_pipeline(
            &mut self,
            y_star: &[f32],
            sample_indices: &[u32],
            l_row_major: &[f32],
            bytes_per_snp: usize,
            max_snps: usize,
        ) -> Result<(), String> {
            self.prepare_packed_lm_pipeline(y_star, sample_indices, bytes_per_snp, max_snps)?;
            let n_samples = sample_indices.len();
            let expected_l = n_samples.saturating_mul(n_samples);
            if l_row_major.len() != expected_l {
                return Err(format!(
                    "fixed-LMM L shape mismatch: len={}, expected={}",
                    l_row_major.len(),
                    expected_l
                ));
            }
            self.ensure_l_buffer(n_samples);
            let l_buf = self
                .l_buf
                .as_ref()
                .ok_or_else(|| "internal error: fixed-LMM L buffer not allocated".to_string())?;
            unsafe {
                ptr::copy_nonoverlapping(
                    l_row_major.as_ptr(),
                    l_buf.contents() as *mut f32,
                    l_row_major.len(),
                );
            }
            Ok(())
        }

        pub fn packed_slot_input_ptr(&self, slot: usize) -> Result<*mut u8, String> {
            let slot_idx = Self::packed_slot_index(slot)?;
            let slot_state = self
                .packed_slots
                .get(slot_idx)
                .ok_or_else(|| format!("invalid packed slot index {}", slot_idx))?;
            if slot_state.inflight_cmd.is_some() {
                return Err(format!(
                    "packed slot {} is still in flight and cannot be reused yet",
                    slot_idx
                ));
            }
            let packed_rows_buf = slot_state
                .packed_rows_buf
                .as_ref()
                .ok_or_else(|| format!("internal error: packed rows buffer {slot_idx} missing"))?;
            Ok(packed_rows_buf.contents() as *mut u8)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn submit_packed_lm_prepared_slot(
            &mut self,
            slot: usize,
            n_samples: usize,
            bytes_per_snp: usize,
            n_snps: usize,
            maf_threshold: f32,
            max_missing_rate: f32,
            fill_missing: bool,
            epsilon: f32,
        ) -> Result<(), String> {
            let slot_idx = Self::packed_slot_index(slot)?;
            if n_samples == 0 || n_snps == 0 {
                return Err("packed submit requires n_samples>0 and n_snps>0".to_string());
            }
            if bytes_per_snp == 0 {
                return Err("metal packed bytes_per_snp must be > 0".to_string());
            }
            if n_samples > self.packed_capacity.0
                || n_snps > self.packed_capacity.1
                || bytes_per_snp != self.packed_capacity.2
            {
                return Err(format!(
                    "packed prepared capacity mismatch: need (n={}, m={}, bps={}), have {:?}",
                    n_samples, n_snps, bytes_per_snp, self.packed_capacity
                ));
            }
            if self.packed_slots[slot_idx].inflight_cmd.is_some() {
                return Err(format!("packed slot {} is still in flight", slot_idx));
            }

            let bps_u32 = u32::try_from(bytes_per_snp)
                .map_err(|_| format!("bytes_per_snp {} exceeds u32 range", bytes_per_snp))?;
            let y_buf = self
                .y_buf
                .as_ref()
                .ok_or_else(|| "internal error: packed y buffer not allocated".to_string())?;
            let sample_idx_buf = self.sample_idx_buf.as_ref().ok_or_else(|| {
                "internal error: packed sample index buffer not allocated".to_string()
            })?;

            let prm = PackedAssocParams {
                n_samples: n_samples as u32,
                n_snps: n_snps as u32,
                bytes_per_snp: bps_u32,
                maf_threshold,
                max_missing_rate,
                epsilon,
                fill_missing: if fill_missing { 1 } else { 0 },
            };

            let cmd = self.queue.new_command_buffer().to_owned();
            {
                let slot_state = self
                    .packed_slots
                    .get(slot_idx)
                    .ok_or_else(|| format!("invalid packed slot index {}", slot_idx))?;
                let packed_rows_buf = slot_state.packed_rows_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed rows buffer {slot_idx} missing")
                })?;
                let beta_buf = slot_state.beta_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed beta buffer {slot_idx} missing")
                })?;
                let se_buf = slot_state.se_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed se buffer {slot_idx} missing")
                })?;
                let maf_buf = slot_state.maf_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed maf buffer {slot_idx} missing")
                })?;
                let n_non_missing_buf = slot_state.n_non_missing_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed non-missing buffer {slot_idx} missing")
                })?;
                let keep_flag_buf = slot_state.keep_flag_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed keep buffer {slot_idx} missing")
                })?;

                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline_packed);
                enc.set_buffer(0, Some(y_buf), 0);
                enc.set_buffer(1, Some(packed_rows_buf), 0);
                enc.set_buffer(2, Some(sample_idx_buf), 0);
                enc.set_buffer(3, Some(beta_buf), 0);
                enc.set_buffer(4, Some(se_buf), 0);
                enc.set_buffer(5, Some(maf_buf), 0);
                enc.set_buffer(6, Some(n_non_missing_buf), 0);
                enc.set_buffer(7, Some(keep_flag_buf), 0);
                enc.set_bytes(
                    8,
                    size_of::<PackedAssocParams>() as u64,
                    &prm as *const PackedAssocParams as *const c_void,
                );

                let tg = min(
                    self.threads_per_threadgroup as u64,
                    self.pipeline_packed.max_total_threads_per_threadgroup(),
                )
                .max(1);
                let grid = MTLSize::new(n_snps as u64, 1, 1);
                let tgs = MTLSize::new(tg, 1, 1);
                enc.dispatch_threads(grid, tgs);
                enc.end_encoding();
            }

            cmd.commit();
            self.packed_slots[slot_idx].inflight_cmd = Some(cmd);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn submit_packed_lmm_fixed_prepared_slot(
            &mut self,
            slot: usize,
            n_samples: usize,
            bytes_per_snp: usize,
            n_snps: usize,
            maf_threshold: f32,
            max_missing_rate: f32,
            fill_missing: bool,
            epsilon: f32,
        ) -> Result<(), String> {
            let slot_idx = Self::packed_slot_index(slot)?;
            if n_samples == 0 || n_snps == 0 {
                return Err("packed fixed-LMM submit requires n_samples>0 and n_snps>0".to_string());
            }
            if bytes_per_snp == 0 {
                return Err("metal packed bytes_per_snp must be > 0".to_string());
            }
            if n_samples > self.packed_capacity.0
                || n_snps > self.packed_capacity.1
                || bytes_per_snp != self.packed_capacity.2
            {
                return Err(format!(
                    "packed fixed-LMM prepared capacity mismatch: need (n={}, m={}, bps={}), have {:?}",
                    n_samples, n_snps, bytes_per_snp, self.packed_capacity
                ));
            }
            if self.packed_slots[slot_idx].inflight_cmd.is_some() {
                return Err(format!("packed slot {} is still in flight", slot_idx));
            }
            if self.l_capacity_samples != n_samples {
                return Err(format!(
                    "fixed-LMM L capacity mismatch: L_n={}, submit_n={}",
                    self.l_capacity_samples, n_samples
                ));
            }

            let bps_u32 = u32::try_from(bytes_per_snp)
                .map_err(|_| format!("bytes_per_snp {} exceeds u32 range", bytes_per_snp))?;
            let y_buf = self
                .y_buf
                .as_ref()
                .ok_or_else(|| "internal error: packed y buffer not allocated".to_string())?;
            let l_buf = self
                .l_buf
                .as_ref()
                .ok_or_else(|| "internal error: fixed-LMM L buffer not allocated".to_string())?;
            let sample_idx_buf = self.sample_idx_buf.as_ref().ok_or_else(|| {
                "internal error: packed sample index buffer not allocated".to_string()
            })?;
            let prm_decode = PackedAssocParams {
                n_samples: n_samples as u32,
                n_snps: n_snps as u32,
                bytes_per_snp: bps_u32,
                maf_threshold,
                max_missing_rate,
                epsilon,
                fill_missing: if fill_missing { 1 } else { 0 },
            };
            let prm_assoc = AssocParams {
                n_samples: n_samples as u32,
                n_snps: n_snps as u32,
                epsilon,
            };

            let cmd = self.queue.new_command_buffer().to_owned();
            {
                let slot_state = self
                    .packed_slots
                    .get(slot_idx)
                    .ok_or_else(|| format!("invalid packed slot index {}", slot_idx))?;
                let packed_rows_buf = slot_state.packed_rows_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed rows buffer {slot_idx} missing")
                })?;
                let decoded_rows_buf = slot_state.decoded_rows_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed decoded buffer {slot_idx} missing")
                })?;
                let beta_buf = slot_state.beta_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed beta buffer {slot_idx} missing")
                })?;
                let se_buf = slot_state.se_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed se buffer {slot_idx} missing")
                })?;
                let maf_buf = slot_state.maf_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed maf buffer {slot_idx} missing")
                })?;
                let n_non_missing_buf = slot_state.n_non_missing_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed non-missing buffer {slot_idx} missing")
                })?;
                let keep_flag_buf = slot_state.keep_flag_buf.as_ref().ok_or_else(|| {
                    format!("internal error: packed keep buffer {slot_idx} missing")
                })?;

                let tg_decode = min(
                    self.threads_per_threadgroup as u64,
                    self.pipeline_packed_decode_qc
                        .max_total_threads_per_threadgroup(),
                )
                .max(1);
                let grid = MTLSize::new(n_snps as u64, 1, 1);
                let tgs_decode = MTLSize::new(tg_decode, 1, 1);
                let enc_decode = cmd.new_compute_command_encoder();
                enc_decode.set_compute_pipeline_state(&self.pipeline_packed_decode_qc);
                enc_decode.set_buffer(0, Some(packed_rows_buf), 0);
                enc_decode.set_buffer(1, Some(sample_idx_buf), 0);
                enc_decode.set_buffer(2, Some(decoded_rows_buf), 0);
                enc_decode.set_buffer(3, Some(maf_buf), 0);
                enc_decode.set_buffer(4, Some(n_non_missing_buf), 0);
                enc_decode.set_buffer(5, Some(keep_flag_buf), 0);
                enc_decode.set_bytes(
                    6,
                    size_of::<PackedAssocParams>() as u64,
                    &prm_decode as *const PackedAssocParams as *const c_void,
                );
                enc_decode.dispatch_threads(grid, tgs_decode);
                enc_decode.end_encoding();

                let tg_assoc = min(
                    self.threads_per_threadgroup as u64,
                    self.pipeline_packed_lmm_fixed
                        .max_total_threads_per_threadgroup(),
                )
                .max(1);
                let tgs_assoc = MTLSize::new(tg_assoc, 1, 1);
                let enc_assoc = cmd.new_compute_command_encoder();
                enc_assoc.set_compute_pipeline_state(&self.pipeline_packed_lmm_fixed);
                enc_assoc.set_buffer(0, Some(y_buf), 0);
                enc_assoc.set_buffer(1, Some(l_buf), 0);
                enc_assoc.set_buffer(2, Some(decoded_rows_buf), 0);
                enc_assoc.set_buffer(3, Some(keep_flag_buf), 0);
                enc_assoc.set_buffer(4, Some(beta_buf), 0);
                enc_assoc.set_buffer(5, Some(se_buf), 0);
                enc_assoc.set_bytes(
                    6,
                    size_of::<AssocParams>() as u64,
                    &prm_assoc as *const AssocParams as *const c_void,
                );
                enc_assoc.dispatch_threads(grid, tgs_assoc);
                enc_assoc.end_encoding();
            }

            cmd.commit();
            self.packed_slots[slot_idx].inflight_cmd = Some(cmd);
            Ok(())
        }

        pub fn collect_packed_lm_prepared_slot(
            &mut self,
            slot: usize,
            n_snps: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>), String> {
            let slot_idx = Self::packed_slot_index(slot)?;
            if n_snps == 0 {
                return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
            }
            if n_snps > self.packed_capacity.1 {
                return Err(format!(
                    "packed collect n_snps out of range: {} > {}",
                    n_snps, self.packed_capacity.1
                ));
            }

            let cmd = {
                let slot_state = self
                    .packed_slots
                    .get_mut(slot_idx)
                    .ok_or_else(|| format!("invalid packed slot index {}", slot_idx))?;
                slot_state
                    .inflight_cmd
                    .take()
                    .ok_or_else(|| format!("packed slot {} has no in-flight command", slot_idx))?
            };
            cmd.wait_until_completed();

            let slot_state = self
                .packed_slots
                .get(slot_idx)
                .ok_or_else(|| format!("invalid packed slot index {}", slot_idx))?;
            let beta_buf = slot_state
                .beta_buf
                .as_ref()
                .ok_or_else(|| format!("internal error: packed beta buffer {slot_idx} missing"))?;
            let se_buf = slot_state
                .se_buf
                .as_ref()
                .ok_or_else(|| format!("internal error: packed se buffer {slot_idx} missing"))?;
            let maf_buf = slot_state
                .maf_buf
                .as_ref()
                .ok_or_else(|| format!("internal error: packed maf buffer {slot_idx} missing"))?;
            let n_non_missing_buf = slot_state.n_non_missing_buf.as_ref().ok_or_else(|| {
                format!("internal error: packed non-missing buffer {slot_idx} missing")
            })?;
            let keep_flag_buf = slot_state
                .keep_flag_buf
                .as_ref()
                .ok_or_else(|| format!("internal error: packed keep buffer {slot_idx} missing"))?;

            let mut betas = vec![0.0f32; n_snps];
            let mut ses = vec![0.0f32; n_snps];
            let mut mafs = vec![0.0f32; n_snps];
            let mut n_non_missing = vec![0u32; n_snps];
            let mut keep_flags = vec![0u32; n_snps];
            unsafe {
                ptr::copy_nonoverlapping(
                    beta_buf.contents() as *const f32,
                    betas.as_mut_ptr(),
                    n_snps,
                );
                ptr::copy_nonoverlapping(se_buf.contents() as *const f32, ses.as_mut_ptr(), n_snps);
                ptr::copy_nonoverlapping(
                    maf_buf.contents() as *const f32,
                    mafs.as_mut_ptr(),
                    n_snps,
                );
                ptr::copy_nonoverlapping(
                    n_non_missing_buf.contents() as *const u32,
                    n_non_missing.as_mut_ptr(),
                    n_snps,
                );
                ptr::copy_nonoverlapping(
                    keep_flag_buf.contents() as *const u32,
                    keep_flags.as_mut_ptr(),
                    n_snps,
                );
            }
            Ok((betas, ses, mafs, n_non_missing, keep_flags))
        }

        pub fn prepare_packed_lm_input(
            &mut self,
            y: &[f32],
            sample_indices: &[u32],
            bytes_per_snp: usize,
            max_snps: usize,
        ) -> Result<*mut u8, String> {
            self.prepare_packed_lm_pipeline(y, sample_indices, bytes_per_snp, max_snps)?;
            self.packed_slot_input_ptr(0)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn run_packed_lm_prepared(
            &mut self,
            n_samples: usize,
            bytes_per_snp: usize,
            n_snps: usize,
            maf_threshold: f32,
            max_missing_rate: f32,
            fill_missing: bool,
            epsilon: f32,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>), String> {
            if n_samples == 0 || n_snps == 0 {
                return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
            }
            self.submit_packed_lm_prepared_slot(
                0,
                n_samples,
                bytes_per_snp,
                n_snps,
                maf_threshold,
                max_missing_rate,
                fill_missing,
                epsilon,
            )?;
            self.collect_packed_lm_prepared_slot(0, n_snps)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn run_packed_lm(
            &mut self,
            y: &[f32],
            packed_rows: &[u8],
            sample_indices: &[u32],
            bytes_per_snp: usize,
            n_snps: usize,
            maf_threshold: f32,
            max_missing_rate: f32,
            fill_missing: bool,
            epsilon: f32,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>), String> {
            let n_samples = sample_indices.len();
            if n_samples == 0 || n_snps == 0 {
                return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
            }
            if y.len() != n_samples {
                return Err(format!(
                    "metal packed y mismatch: y_len={}, n_samples={}",
                    y.len(),
                    n_samples
                ));
            }
            if bytes_per_snp == 0 {
                return Err("metal packed bytes_per_snp must be > 0".to_string());
            }
            let expected = n_snps.saturating_mul(bytes_per_snp);
            if packed_rows.len() != expected {
                return Err(format!(
                    "metal packed shape mismatch: packed_len={}, expected={}",
                    packed_rows.len(),
                    expected
                ));
            }
            let packed_dst =
                self.prepare_packed_lm_input(y, sample_indices, bytes_per_snp, n_snps)?;
            unsafe {
                ptr::copy_nonoverlapping(packed_rows.as_ptr(), packed_dst, packed_rows.len());
            }
            self.run_packed_lm_prepared(
                n_samples,
                bytes_per_snp,
                n_snps,
                maf_threshold,
                max_missing_rate,
                fill_missing,
                epsilon,
            )
        }
    }

    pub fn new_scanner(threads_per_threadgroup: usize) -> Result<MetalLmScanner, String> {
        MetalLmScanner::new(threads_per_threadgroup)
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
mod metal_impl {
    pub struct MetalLmScanner;
    impl MetalLmScanner {
        pub fn run(
            &mut self,
            _y: &[f32],
            _x_snp_major: &[f32],
            _n_samples: usize,
            _n_snps: usize,
            _epsilon: f32,
        ) -> Result<(Vec<f32>, Vec<f32>), String> {
            Err("Metal backend is only available on macOS".to_string())
        }

        pub fn prepare_packed_lm_pipeline(
            &mut self,
            _y: &[f32],
            _sample_indices: &[u32],
            _bytes_per_snp: usize,
            _max_snps: usize,
        ) -> Result<(), String> {
            Err("Metal backend is only available on macOS".to_string())
        }

        pub fn prepare_packed_lmm_fixed_pipeline(
            &mut self,
            _y_star: &[f32],
            _sample_indices: &[u32],
            _l_row_major: &[f32],
            _bytes_per_snp: usize,
            _max_snps: usize,
        ) -> Result<(), String> {
            Err("Metal backend is only available on macOS".to_string())
        }

        pub fn packed_slot_input_ptr(&self, _slot: usize) -> Result<*mut u8, String> {
            Err("Metal backend is only available on macOS".to_string())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn submit_packed_lm_prepared_slot(
            &mut self,
            _slot: usize,
            _n_samples: usize,
            _bytes_per_snp: usize,
            _n_snps: usize,
            _maf_threshold: f32,
            _max_missing_rate: f32,
            _fill_missing: bool,
            _epsilon: f32,
        ) -> Result<(), String> {
            Err("Metal backend is only available on macOS".to_string())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn submit_packed_lmm_fixed_prepared_slot(
            &mut self,
            _slot: usize,
            _n_samples: usize,
            _bytes_per_snp: usize,
            _n_snps: usize,
            _maf_threshold: f32,
            _max_missing_rate: f32,
            _fill_missing: bool,
            _epsilon: f32,
        ) -> Result<(), String> {
            Err("Metal backend is only available on macOS".to_string())
        }

        pub fn collect_packed_lm_prepared_slot(
            &mut self,
            _slot: usize,
            _n_snps: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>), String> {
            Err("Metal backend is only available on macOS".to_string())
        }

        pub fn prepare_packed_lm_input(
            &mut self,
            _y: &[f32],
            _sample_indices: &[u32],
            _bytes_per_snp: usize,
            _max_snps: usize,
        ) -> Result<*mut u8, String> {
            Err("Metal backend is only available on macOS".to_string())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn run_packed_lm_prepared(
            &mut self,
            _n_samples: usize,
            _bytes_per_snp: usize,
            _n_snps: usize,
            _maf_threshold: f32,
            _max_missing_rate: f32,
            _fill_missing: bool,
            _epsilon: f32,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>), String> {
            Err("Metal backend is only available on macOS".to_string())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn run_packed_lm(
            &mut self,
            _y: &[f32],
            _packed_rows: &[u8],
            _sample_indices: &[u32],
            _bytes_per_snp: usize,
            _n_snps: usize,
            _maf_threshold: f32,
            _max_missing_rate: f32,
            _fill_missing: bool,
            _epsilon: f32,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>), String> {
            Err("Metal backend is only available on macOS".to_string())
        }
    }

    pub fn new_scanner(_threads_per_threadgroup: usize) -> Result<MetalLmScanner, String> {
        Err("Metal backend is only available on macOS".to_string())
    }
}

use metal_impl::MetalLmScanner;
