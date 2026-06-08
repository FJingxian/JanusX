use memmap2::Mmap;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::bedmath::adaptive_grm_block_rows;
use crate::blas::{
    cblas_dgemm_dispatch, rust_sgemm_prefers_rayon_rowmajor_f32_kernel, BlasThreadGuard, CblasInt,
    CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
};
use crate::cholesky::{
    sparse_cholesky_analyze_subset_jxgrm_path_cached_with_progress, sparse_jxgrm_header_n_samples,
    subset_sparse_grm_csc, MmapSparseGrmCsc, SparseGrmCscView, SparseJxgrmAnalyzeProgressStage,
    SparseJxgrmCholesky, SparseJxgrmSolveWorkspace,
};
use crate::gfcore::read_fam;
use crate::gfreader::prepare_bed_logic_meta_owned_for_stats_samples;
use crate::gload::{GenotypeMatrix, UnifiedInput};
use crate::he::row_major_block_mul_mat_f32;
use crate::linalg::{
    chi2_sf_df1, chisq_from_beta_se_and_optional_plrt, cholesky_inplace, cholesky_solve_into,
    format_chisq_value, sanitize_assoc_pvalue,
};
use crate::pcg::{PcgJxlmmNullModel, PcgJxlmmNullModelInfo, PcgJxlmmRHatResult};
use crate::stats_common::{env_truthy, get_cached_pool, parse_index_vec_i64, AsyncTsvWriter};

const SPLMM_TINY: f64 = 1e-30_f64;
const SPLMM_EXACT_PG_MIN_REL_SM: f64 = 1e-4_f64;
const SPLMM_DEFAULT_RHAT_MARKERS: usize = 30;
const SPLMM_DEFAULT_RHAT_SEED: u64 = 20260527;
const SPLMM_DEFAULT_SPARSE_CHOLESKY_MAX_L_NNZ: usize = 100_000_000;
const SPLMM_TWO_STAGE_P_THRESHOLD_FLOOR: f64 = 1e-4_f64;
const SPLMM_TWO_STAGE_P_THRESHOLD_NUM: f64 = 10.0_f64;

#[derive(Clone, Copy, Debug, Default)]
struct SplmmPackedScanTiming {
    decode_secs: f64,
    gpy_secs: f64,
    gx_secs: f64,
    row_sumsq_secs: f64,
    repack_secs: f64,
    solve_secs: f64,
    xtz_secs: f64,
    denom_secs: f64,
    sink_secs: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct SplmmTsvTiming {
    format_secs: f64,
    send_secs: f64,
    finish_secs: f64,
    blocks: usize,
    bytes: usize,
}

#[inline]
fn splmm_packed_stage_timing_enabled() -> bool {
    env_truthy("JX_SPLMM_PACKED_STAGE_TIMING")
}

fn emit_splmm_packed_scan_timing(
    mode: &str,
    timing: &SplmmPackedScanTiming,
    total_secs: f64,
    rows: usize,
    n: usize,
    p: usize,
    threads: usize,
) {
    let accounted_secs = timing.decode_secs
        + timing.gpy_secs
        + timing.gx_secs
        + timing.row_sumsq_secs
        + timing.repack_secs
        + timing.solve_secs
        + timing.xtz_secs
        + timing.denom_secs
        + timing.sink_secs;
    let other_secs = (total_secs - accounted_secs).max(0.0_f64);
    let to_pct = |x: f64| -> f64 {
        if total_secs > 0.0 {
            x * 100.0 / total_secs
        } else {
            0.0
        }
    };
    eprintln!(
        "SparseLMM packed timing mode={mode}: decode={:.3}s ({:.1}%), gpy={:.3}s ({:.1}%), gx={:.3}s ({:.1}%), row_sumsq={:.3}s ({:.1}%), repack={:.3}s ({:.1}%), solve={:.3}s ({:.1}%), xtz={:.3}s ({:.1}%), denom={:.3}s ({:.1}%), sink={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, p={}, threads={}",
        timing.decode_secs,
        to_pct(timing.decode_secs),
        timing.gpy_secs,
        to_pct(timing.gpy_secs),
        timing.gx_secs,
        to_pct(timing.gx_secs),
        timing.row_sumsq_secs,
        to_pct(timing.row_sumsq_secs),
        timing.repack_secs,
        to_pct(timing.repack_secs),
        timing.solve_secs,
        to_pct(timing.solve_secs),
        timing.xtz_secs,
        to_pct(timing.xtz_secs),
        timing.denom_secs,
        to_pct(timing.denom_secs),
        timing.sink_secs,
        to_pct(timing.sink_secs),
        other_secs,
        to_pct(other_secs),
        total_secs,
        rows,
        n,
        p,
        if threads > 0 { threads } else { rayon::current_num_threads() },
    );
}

fn emit_splmm_tsv_timing(mode: &str, timing: &SplmmTsvTiming, rows: usize) {
    let total_secs = timing.format_secs + timing.send_secs + timing.finish_secs;
    let to_pct = |x: f64| -> f64 {
        if total_secs > 0.0 {
            x * 100.0 / total_secs
        } else {
            0.0
        }
    };
    eprintln!(
        "SparseLMM TSV timing mode={mode}: format={:.3}s ({:.1}%), send={:.3}s ({:.1}%), finish={:.3}s ({:.1}%), total={:.3}s, rows={}, blocks={}, bytes={}",
        timing.format_secs,
        to_pct(timing.format_secs),
        timing.send_secs,
        to_pct(timing.send_secs),
        timing.finish_secs,
        to_pct(timing.finish_secs),
        total_secs,
        rows,
        timing.blocks,
        timing.bytes,
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplmmScanMode {
    Approx,
    Exact,
    TwoStage,
}

impl SplmmScanMode {
    fn parse(text: &str) -> PyResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "approx" => Ok(Self::Approx),
            "exact" => Ok(Self::Exact),
            "two_stage" | "two-stage" | "twostage" | "hybrid" => Ok(Self::TwoStage),
            _ => Err(PyRuntimeError::new_err(
                "scan_mode must be one of: approx, exact, two_stage",
            )),
        }
    }

    #[inline]
    fn needs_rhat(self) -> bool {
        matches!(self, Self::Approx | Self::TwoStage)
    }

    #[inline]
    fn needs_exact_workspace(self) -> bool {
        matches!(self, Self::Exact | Self::TwoStage)
    }
}

#[derive(Clone, Copy)]
enum PackedGeneticModel {
    Add,
    Dom,
    Rec,
    Het,
}

impl PackedGeneticModel {
    fn parse(text: &str) -> PyResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "add" => Ok(Self::Add),
            "dom" => Ok(Self::Dom),
            "rec" => Ok(Self::Rec),
            "het" => Ok(Self::Het),
            _ => Err(PyRuntimeError::new_err(
                "model must be one of: add, dom, rec, het",
            )),
        }
    }

    #[inline]
    fn apply(self, g: f64) -> f64 {
        match self {
            Self::Add => g,
            Self::Dom => {
                if g > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Rec => {
                if (g - 2.0).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Het => {
                if (g - 1.0).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

enum JxlmmPayload {
    Packed(Arc<[u8]>),
    Mmap(Arc<Mmap>),
}

impl JxlmmPayload {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Packed(bytes) => bytes.as_ref(),
            Self::Mmap(mmap) => &mmap[3..],
        }
    }
}

struct JxlmmPreparedInput {
    bed_prefix: Option<String>,
    payload: JxlmmPayload,
    n_samples_full: usize,
    bytes_per_snp: usize,
    row_flip: Vec<bool>,
    row_maf: Vec<f32>,
    #[allow(dead_code)]
    row_missing: Vec<f32>,
    row_source_indices: Option<Vec<usize>>,
}

impl JxlmmPreparedInput {
    #[inline]
    fn n_rows(&self) -> usize {
        self.row_flip.len()
    }

    #[inline]
    fn row_bytes(&self, row_idx: usize) -> &[u8] {
        let src = self
            .row_source_indices
            .as_ref()
            .map(|v| v[row_idx])
            .unwrap_or(row_idx);
        let packed = self.payload.as_bytes();
        &packed[src * self.bytes_per_snp..(src + 1) * self.bytes_per_snp]
    }

    fn subset_rows(&self, keep_rows: &[usize]) -> Result<Self, String> {
        if keep_rows.is_empty() {
            return Err("SparseLMM subset_rows requires at least one row".to_string());
        }
        let m = self.n_rows();
        let mut row_flip = Vec::with_capacity(keep_rows.len());
        let mut row_maf = Vec::with_capacity(keep_rows.len());
        let mut row_missing = Vec::with_capacity(keep_rows.len());
        let mut row_source_indices = Vec::with_capacity(keep_rows.len());
        for &row_idx in keep_rows {
            if row_idx >= m {
                return Err(format!(
                    "SparseLMM subset row index out of bounds: row_idx={row_idx}, n_rows={m}"
                ));
            }
            row_flip.push(self.row_flip[row_idx]);
            row_maf.push(self.row_maf[row_idx]);
            row_missing.push(self.row_missing[row_idx]);
            let src = self
                .row_source_indices
                .as_ref()
                .map(|v| v[row_idx])
                .unwrap_or(row_idx);
            row_source_indices.push(src);
        }
        Ok(Self {
            bed_prefix: self.bed_prefix.clone(),
            payload: match &self.payload {
                JxlmmPayload::Packed(bytes) => JxlmmPayload::Packed(Arc::clone(bytes)),
                JxlmmPayload::Mmap(mmap) => JxlmmPayload::Mmap(Arc::clone(mmap)),
            },
            n_samples_full: self.n_samples_full,
            bytes_per_snp: self.bytes_per_snp,
            row_flip,
            row_maf,
            row_missing,
            row_source_indices: Some(row_source_indices),
        })
    }
}

impl Clone for JxlmmPreparedInput {
    fn clone(&self) -> Self {
        Self {
            bed_prefix: self.bed_prefix.clone(),
            payload: match &self.payload {
                JxlmmPayload::Packed(bytes) => JxlmmPayload::Packed(Arc::clone(bytes)),
                JxlmmPayload::Mmap(mmap) => JxlmmPayload::Mmap(Arc::clone(mmap)),
            },
            n_samples_full: self.n_samples_full,
            bytes_per_snp: self.bytes_per_snp,
            row_flip: self.row_flip.clone(),
            row_maf: self.row_maf.clone(),
            row_missing: self.row_missing.clone(),
            row_source_indices: self.row_source_indices.clone(),
        }
    }
}

struct PreparedJxlmmAssoc {
    gm: PackedGeneticModel,
    y_vec: Vec<f64>,
    x_design: Vec<f64>,
    scan_prepared: JxlmmPreparedInput,
    operator_prepared: JxlmmPreparedInput,
    scan_sample_idx: Vec<usize>,
    operator_sample_idx: Vec<usize>,
}

#[inline]
fn normalize_plink_prefix_local(prefix: &str) -> String {
    let mut out = prefix.trim().to_string();
    let lower = out.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        out.truncate(out.len().saturating_sub(4));
    }
    out
}

#[inline]
fn transform_alleles_by_model_local<'a>(
    ref_allele: &'a str,
    alt_allele: &'a str,
    gm: PackedGeneticModel,
) -> (std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>) {
    use std::borrow::Cow;
    match gm {
        PackedGeneticModel::Add => (Cow::Borrowed(ref_allele), Cow::Borrowed(alt_allele)),
        PackedGeneticModel::Dom => (
            Cow::Owned(format!("{ref_allele}{ref_allele}")),
            Cow::Owned(format!("{ref_allele}{alt_allele}/{alt_allele}{alt_allele}")),
        ),
        PackedGeneticModel::Rec => (
            Cow::Owned(format!("{ref_allele}{alt_allele}/{ref_allele}{ref_allele}")),
            Cow::Owned(format!("{alt_allele}{alt_allele}")),
        ),
        PackedGeneticModel::Het => (
            Cow::Owned(format!("{ref_allele}{ref_allele}/{alt_allele}{alt_allele}")),
            Cow::Owned(format!("{ref_allele}{alt_allele}")),
        ),
    }
}

#[inline]
fn build_design_with_intercept(x_cov: Option<&[f64]>, n: usize, p_cov: usize) -> Vec<f64> {
    let p = p_cov + 1;
    let mut x = vec![0.0_f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        if let Some(x_cov) = x_cov {
            let src = &x_cov[i * p_cov..(i + 1) * p_cov];
            x[i * p + 1..(i + 1) * p].copy_from_slice(src);
        }
    }
    x
}

fn filter_meta_with_site_keep(
    meta: crate::gfreader::PreparedBedLogicMetaOwned,
    site_keep_full: Option<&[bool]>,
) -> Result<crate::gfreader::PreparedBedLogicMetaOwned, String> {
    let Some(site_keep_full) = site_keep_full else {
        return Ok(meta);
    };
    if site_keep_full.len() != meta.n_snps_total {
        return Err(format!(
            "site_keep length mismatch: got {}, expected {}",
            site_keep_full.len(),
            meta.n_snps_total
        ));
    }
    let mut row_flip = Vec::with_capacity(meta.row_flip.len());
    let mut row_source_indices = Vec::with_capacity(meta.row_source_indices.len());
    let mut missing_rate = Vec::with_capacity(meta.missing_rate.len());
    let mut maf = Vec::with_capacity(meta.maf.len());
    let mut sites = Vec::with_capacity(meta.sites.len());
    let mut site_keep = vec![false; meta.n_snps_total];
    for idx in 0..meta.row_source_indices.len() {
        let src = meta.row_source_indices[idx];
        if site_keep_full[src] {
            row_flip.push(meta.row_flip[idx]);
            row_source_indices.push(src);
            missing_rate.push(meta.missing_rate[idx]);
            maf.push(meta.maf[idx]);
            sites.push(meta.sites[idx].clone());
            site_keep[src] = true;
        }
    }
    if row_flip.is_empty() {
        return Err("No SNPs left after applying site_keep".to_string());
    }
    Ok(crate::gfreader::PreparedBedLogicMetaOwned {
        site_keep,
        row_flip,
        row_source_indices,
        missing_rate,
        maf,
        sites,
        n_samples: meta.n_samples,
        n_snps_total: meta.n_snps_total,
        bytes_per_snp: meta.bytes_per_snp,
    })
}

fn prepare_external_packed_input<'py>(
    packed: Option<PyReadonlyArray2<'py, u8>>,
    packed_n_samples: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<JxlmmPreparedInput> {
    let packed_ro = packed.ok_or_else(|| {
        PyRuntimeError::new_err(
            "splmm_assoc_pcg_bed: packed payload path requires `packed` argument.",
        )
    })?;
    let maf_ro = maf.ok_or_else(|| {
        PyRuntimeError::new_err("splmm_assoc_pcg_bed: packed payload path requires `maf` argument.")
    })?;
    let row_flip_ro = row_flip.ok_or_else(|| {
        PyRuntimeError::new_err(
            "splmm_assoc_pcg_bed: packed payload path requires `row_flip` argument.",
        )
    })?;
    if packed_n_samples == 0 {
        return Err(PyRuntimeError::new_err(
            "splmm_assoc_pcg_bed: packed payload path requires packed_n_samples > 0",
        ));
    }

    let packed_arr = packed_ro.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp).",
        ));
    }
    let m_packed = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = packed_n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }

    let row_source_indices = if let Some(row_indices) = row_indices {
        Some(parse_index_vec_i64(
            row_indices.as_slice()?,
            m_packed,
            "row_indices",
        )?)
    } else {
        None
    };
    let m = row_source_indices
        .as_ref()
        .map(|v| v.len())
        .unwrap_or(m_packed);
    let row_maf = match maf_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => maf_ro.as_array().iter().copied().collect(),
    };
    let row_flip = match row_flip_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip_ro.as_array().iter().copied().collect(),
    };
    let row_missing = if let Some(row_missing_ro) = row_missing {
        match row_missing_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_missing_ro.as_array().iter().copied().collect(),
        }
    } else {
        vec![f32::NAN; m]
    };
    if row_maf.len() != m || row_flip.len() != m || row_missing.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "packed metadata length mismatch: rows={m}, row_maf={}, row_flip={}, row_missing={}",
            row_maf.len(),
            row_flip.len(),
            row_missing.len()
        )));
    }
    let packed_arc: Arc<[u8]> = match packed_ro.as_slice() {
        Ok(s) => Arc::from(s),
        Err(_) => Arc::from(
            packed_arr
                .iter()
                .copied()
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        ),
    };
    Ok(JxlmmPreparedInput {
        bed_prefix: None,
        payload: JxlmmPayload::Packed(packed_arc),
        n_samples_full: packed_n_samples,
        bytes_per_snp,
        row_flip,
        row_maf,
        row_missing,
        row_source_indices,
    })
}

fn prepare_prefix_input_from_external_meta<'py>(
    prefix: &str,
    n_samples_full: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    payload_mmap: Option<Arc<Mmap>>,
) -> PyResult<JxlmmPreparedInput> {
    let bed_prefix = normalize_plink_prefix_local(prefix);
    if bed_prefix.is_empty() {
        return Err(PyRuntimeError::new_err(
            "BED-prefix mode requires a non-empty prefix",
        ));
    }
    if n_samples_full == 0 {
        return Err(PyRuntimeError::new_err("No samples found in BED input."));
    }
    let maf_ro = maf.ok_or_else(|| {
        PyRuntimeError::new_err("splmm_assoc_pcg_bed: mmap metadata path requires `maf` argument.")
    })?;
    let row_flip_ro = row_flip.ok_or_else(|| {
        PyRuntimeError::new_err(
            "splmm_assoc_pcg_bed: mmap metadata path requires `row_flip` argument.",
        )
    })?;
    let row_indices_ro = row_indices.ok_or_else(|| {
        PyRuntimeError::new_err(
            "splmm_assoc_pcg_bed: mmap metadata path requires `row_indices` argument.",
        )
    })?;
    let mmap = if let Some(mmap) = payload_mmap {
        mmap
    } else {
        let bed_path = format!("{bed_prefix}.bed");
        let bed_file = File::open(&bed_path)
            .map_err(|e| PyRuntimeError::new_err(format!("open {bed_path}: {e}")))?;
        Arc::new(
            unsafe { Mmap::map(&bed_file) }
                .map_err(|e| PyRuntimeError::new_err(format!("mmap {bed_path}: {e}")))?,
        )
    };
    if mmap.len() < 3 {
        return Err(PyRuntimeError::new_err("BED too small"));
    }
    if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
        return Err(PyRuntimeError::new_err("Only SNP-major BED supported"));
    }
    let bytes_per_snp = n_samples_full.div_ceil(4);
    let payload_len = mmap.len().saturating_sub(3);
    if bytes_per_snp == 0 || payload_len % bytes_per_snp != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "invalid BED payload length: data_len={payload_len}, bytes_per_snp={bytes_per_snp}"
        )));
    }
    let n_snps_bed = payload_len / bytes_per_snp;
    let row_source_indices =
        parse_index_vec_i64(row_indices_ro.as_slice()?, n_snps_bed, "row_indices")?;
    let m = row_source_indices.len();
    let row_maf = match maf_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => maf_ro.as_array().iter().copied().collect(),
    };
    let row_flip = match row_flip_ro.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip_ro.as_array().iter().copied().collect(),
    };
    let row_missing = if let Some(row_missing_ro) = row_missing {
        match row_missing_ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => row_missing_ro.as_array().iter().copied().collect(),
        }
    } else {
        vec![f32::NAN; m]
    };
    if row_maf.len() != m || row_flip.len() != m || row_missing.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "mmap metadata length mismatch: rows={m}, row_maf={}, row_flip={}, row_missing={}",
            row_maf.len(),
            row_flip.len(),
            row_missing.len()
        )));
    }
    Ok(JxlmmPreparedInput {
        bed_prefix: Some(bed_prefix),
        payload: JxlmmPayload::Mmap(mmap),
        n_samples_full,
        bytes_per_snp,
        row_flip,
        row_maf,
        row_missing,
        row_source_indices: Some(row_source_indices),
    })
}

fn prepare_prefix_input(
    prefix: &str,
    n_samples_full: usize,
    site_keep: Option<&[bool]>,
    sample_idx_probe: Option<&[usize]>,
    payload_mmap: Option<Arc<Mmap>>,
) -> Result<JxlmmPreparedInput, String> {
    let bed_prefix = normalize_plink_prefix_local(prefix);
    if bed_prefix.is_empty() {
        return Err("BED-prefix mode requires a non-empty prefix".to_string());
    }
    if n_samples_full == 0 {
        return Err("No samples found in BED input.".to_string());
    }
    let meta = prepare_bed_logic_meta_owned_for_stats_samples(
        &bed_prefix,
        0.0_f32,
        1.0_f32,
        0.0_f32,
        false,
        sample_idx_probe,
        false,
    )?;
    let meta = filter_meta_with_site_keep(meta, site_keep)?;
    let mmap = if let Some(mmap) = payload_mmap {
        mmap
    } else {
        let bed_path = format!("{bed_prefix}.bed");
        let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        Arc::new(unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?)
    };
    Ok(JxlmmPreparedInput {
        bed_prefix: Some(bed_prefix),
        payload: JxlmmPayload::Mmap(mmap),
        n_samples_full,
        bytes_per_snp: meta.bytes_per_snp,
        row_flip: meta.row_flip,
        row_maf: meta.maf,
        row_missing: meta.missing_rate,
        row_source_indices: Some(meta.row_source_indices),
    })
}

#[inline]
fn parse_optional_index_array<'py>(
    raw: Option<&PyReadonlyArray1<'py, i64>>,
    limit: usize,
    label: &str,
) -> PyResult<Option<Vec<usize>>> {
    raw.map(|arr| parse_index_vec_i64(arr.as_slice()?, limit, label))
        .transpose()
}

#[allow(clippy::too_many_arguments)]
fn prepare_splmm_assoc_inputs<'py>(
    prefix: &str,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    operator_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    packed: Option<PyReadonlyArray2<'py, u8>>,
    packed_n_samples: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    model: &str,
) -> PyResult<PreparedJxlmmAssoc> {
    let gm = PackedGeneticModel::parse(model)?;
    let bed_prefix = normalize_plink_prefix_local(prefix);
    if bed_prefix.is_empty() {
        return Err(PyRuntimeError::new_err(
            "BED-prefix mode requires a non-empty prefix",
        ));
    }
    let n_samples_full = read_fam(&bed_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples_full == 0 {
        return Err(PyRuntimeError::new_err("No samples found in BED input."));
    }

    let y_vec = y.as_slice()?.to_vec();
    let n = y_vec.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y must not be empty"));
    }

    let (x_cov_owned, p_cov) = if let Some(x_cov) = &x_cov {
        let x_arr = x_cov.as_array();
        if x_arr.ndim() != 2 {
            return Err(PyRuntimeError::new_err("x_cov must be 2D (n, p_cov)"));
        }
        let (xn, xp) = (x_arr.shape()[0], x_arr.shape()[1]);
        if xn != n {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov row count mismatch: got {xn}, expected {n}"
            )));
        }
        let owned = match x_cov.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => x_arr.iter().copied().collect(),
        };
        (Some(owned), xp)
    } else {
        (None, 0usize)
    };
    let x_design = build_design_with_intercept(x_cov_owned.as_deref(), n, p_cov);
    let p = p_cov + 1;
    if n <= p {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p for SparseLMM: n={n}, p={p}"
        )));
    }

    let use_external_packed = packed.is_some() || packed_n_samples > 0;
    let use_external_mmap_meta = !use_external_packed
        && (maf.is_some() || row_flip.is_some() || row_missing.is_some() || row_indices.is_some());
    let site_keep_full = if let Some(site_keep) = site_keep {
        Some(match site_keep.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => site_keep.as_array().iter().copied().collect(),
        })
    } else {
        None
    };
    let scan_sample_probe =
        parse_optional_index_array(sample_indices.as_ref(), n_samples_full, "sample_indices")?;
    let operator_sample_probe = parse_optional_index_array(
        operator_sample_indices.as_ref(),
        n_samples_full,
        "operator_sample_indices",
    )?
    .or_else(|| scan_sample_probe.clone());
    let shared_bed_mmap = if use_external_packed {
        None
    } else {
        let bed_path = format!("{bed_prefix}.bed");
        let bed_file = File::open(&bed_path)
            .map_err(|e| PyRuntimeError::new_err(format!("open {bed_path}: {e}")))?;
        Some(Arc::new(unsafe { Mmap::map(&bed_file) }.map_err(|e| {
            PyRuntimeError::new_err(format!("mmap {bed_path}: {e}"))
        })?))
    };

    let scan_prepared = if use_external_packed {
        prepare_external_packed_input(
            packed,
            packed_n_samples,
            maf,
            row_flip,
            row_missing,
            row_indices,
        )?
    } else if use_external_mmap_meta {
        prepare_prefix_input_from_external_meta(
            &bed_prefix,
            n_samples_full,
            maf,
            row_flip,
            row_missing,
            row_indices,
            shared_bed_mmap.as_ref().map(Arc::clone),
        )?
    } else {
        prepare_prefix_input(
            &bed_prefix,
            n_samples_full,
            site_keep_full.as_deref(),
            scan_sample_probe.as_deref(),
            shared_bed_mmap.as_ref().map(Arc::clone),
        )
        .map_err(PyRuntimeError::new_err)?
    };
    let operator_prepared = if use_external_packed || use_external_mmap_meta {
        scan_prepared.clone()
    } else if !use_external_packed
        && operator_sample_probe.as_deref() == scan_sample_probe.as_deref()
    {
        scan_prepared.clone()
    } else {
        prepare_prefix_input(
            &bed_prefix,
            n_samples_full,
            site_keep_full.as_deref(),
            operator_sample_probe.as_deref(),
            shared_bed_mmap.as_ref().map(Arc::clone),
        )
        .map_err(PyRuntimeError::new_err)?
    };

    let scan_sample_idx: Vec<usize> = if let Some(parsed) = scan_sample_probe {
        if parsed.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                parsed.len()
            )));
        }
        parsed
    } else {
        if n != scan_prepared.n_samples_full {
            return Err(PyRuntimeError::new_err(format!(
                "len(y)={} must equal scan n_samples={} when sample_indices is not provided",
                n, scan_prepared.n_samples_full
            )));
        }
        (0..n).collect()
    };

    let operator_sample_idx: Vec<usize> = if let Some(parsed) = operator_sample_probe {
        if parsed.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "operator_sample_indices length mismatch: got {}, expected {n}",
                parsed.len()
            )));
        }
        parsed
    } else {
        if n != operator_prepared.n_samples_full {
            return Err(PyRuntimeError::new_err(format!(
                "len(y)={} must equal operator n_samples={} when operator_sample_indices is not provided",
                n, operator_prepared.n_samples_full
            )));
        }
        (0..n).collect()
    };

    Ok(PreparedJxlmmAssoc {
        gm,
        y_vec,
        x_design,
        scan_prepared,
        operator_prepared,
        scan_sample_idx,
        operator_sample_idx,
    })
}

#[inline]
fn decode_packed_row_model_into_f64(
    row: &[u8],
    flip: bool,
    row_maf: f32,
    n: usize,
    gm: PackedGeneticModel,
    sample_identity: bool,
    sample_byte_idx: Option<&[usize]>,
    sample_bit_shift: Option<&[u8]>,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), n);
    let mean_g = (2.0_f64 * row_maf as f64).max(0.0);
    if sample_identity {
        let mut j = 0usize;
        for &b in row.iter() {
            let mut lane = 0usize;
            while lane < 4 && j < n {
                let code = (b >> (lane * 2)) & 0b11;
                let mut gv = match code {
                    0b00 => 0.0_f64,
                    0b10 => 1.0_f64,
                    0b11 => 2.0_f64,
                    _ => mean_g,
                };
                if flip && code != 0b01 {
                    gv = 2.0_f64 - gv;
                }
                out[j] = gm.apply(gv);
                lane += 1;
                j += 1;
            }
            if j >= n {
                break;
            }
        }
        return;
    }

    let byte_idx = sample_byte_idx.expect("sample_byte_idx must exist for non-identity mapping");
    let bit_shift = sample_bit_shift.expect("sample_bit_shift must exist for non-identity mapping");
    for j in 0..n {
        let b = row[byte_idx[j]];
        let code = (b >> bit_shift[j]) & 0b11;
        let mut gv = match code {
            0b00 => 0.0_f64,
            0b10 => 1.0_f64,
            0b11 => 2.0_f64,
            _ => mean_g,
        };
        if flip && code != 0b01 {
            gv = 2.0_f64 - gv;
        }
        out[j] = gm.apply(gv);
    }
}

fn choose_rhat_rows(m: usize, count: usize, seed: u64) -> Vec<usize> {
    let k = count.min(m);
    if k == m {
        return (0..m).collect();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = sample(&mut rng, m, k).into_vec();
    out.sort_unstable();
    out
}

#[inline]
fn n_rhat_progress_total(m: usize, count: usize) -> usize {
    count.min(m).saturating_add(1).max(1)
}

#[inline]
fn emit_progress_callback(
    cb: Option<&Py<PyAny>>,
    stage: usize,
    done: usize,
    total: usize,
) -> Result<(), String> {
    let total_use = total.max(1);
    let done_use = done.min(total_use);
    if let Some(cb) = cb {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (stage, done_use, total_use))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn xtx_chol_from_design(x_design: &[f64], n: usize, p: usize) -> Result<Vec<f64>, String> {
    if n == 0 || p == 0 {
        return Err("SparseLMM XtX chol requires n > 0 and p > 0".to_string());
    }
    if x_design.len() != n * p {
        return Err(format!(
            "SparseLMM XtX design length mismatch: got {}, expected {}",
            x_design.len(),
            n * p
        ));
    }
    // XtX via dsyrk: X is n×p row-major ≡ p×n column-major with lda=p.
    let mut xtx = vec![0.0_f64; p * p];
    {
        use crate::blas::{
            cblas_dsyrk_dispatch, CblasInt, CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_UPPER,
        };
        unsafe {
            cblas_dsyrk_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_UPPER,
                CBLAS_NO_TRANS,
                p as CblasInt,
                n as CblasInt,
                1.0_f64,
                x_design.as_ptr(),
                p as CblasInt,
                0.0_f64,
                xtx.as_mut_ptr(),
                p as CblasInt,
            );
        }
    }
    for a in 0..p {
        for b in 0..a {
            xtx[b * p + a] = xtx[a * p + b];
        }
    }
    spd_cholesky_with_jitter(&xtx, p, "SparseLMM XtX")
}

#[inline]
fn residualized_sumsq_from_xtx_chol(
    xtx_chol: &[f64],
    p: usize,
    xts: &[f64],
    alpha: &mut [f64],
    s_sq: f64,
) -> f64 {
    cholesky_solve_into(xtx_chol, p, xts, alpha);
    let x_quad = xts
        .iter()
        .zip(alpha.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    (s_sq - x_quad).max(0.0_f64)
}

#[inline]
fn cast_f64_slice_to_f32(input: &[f64]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(input.len());
    for &value in input {
        if !value.is_finite() {
            return Err("SparseLMM scan received non-finite dense operand".to_string());
        }
        out.push(value as f32);
    }
    Ok(out)
}

fn row_major_block_sumsq_f64(
    block: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(out.len(), rows);
    let use_parallel = pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(cols) >= 16_384usize;
    if use_parallel {
        let mut run = || {
            out.par_iter_mut().enumerate().for_each(|(r, dst)| {
                let row = &block[r * cols..(r + 1) * cols];
                let mut ss = 0.0_f64;
                for &value in row {
                    let vf = value as f64;
                    ss += vf * vf;
                }
                *dst = ss;
            });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }
    for r in 0..rows {
        let row = &block[r * cols..(r + 1) * cols];
        let mut ss = 0.0_f64;
        for &value in row {
            let vf = value as f64;
            ss += vf * vf;
        }
        out[r] = ss;
    }
}

#[inline]
fn row_major_to_col_major_f64(
    input: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<f64>, String> {
    if input.len() != n_rows.saturating_mul(n_cols) {
        return Err(format!(
            "row-major matrix length mismatch: got {}, expected {}",
            input.len(),
            n_rows.saturating_mul(n_cols)
        ));
    }
    let mut out = vec![0.0_f64; n_rows.saturating_mul(n_cols)];
    for row in 0..n_rows {
        let src = &input[row * n_cols..(row + 1) * n_cols];
        for col in 0..n_cols {
            out[col * n_rows + row] = src[col];
        }
    }
    Ok(out)
}

#[inline]
fn xt_vec_row_major(
    x_row_major: &[f64],
    n_samples: usize,
    n_covariates: usize,
    vec_in: &[f64],
    out: &mut [f64],
) {
    out.fill(0.0);
    for i in 0..n_samples {
        let yi = vec_in[i];
        let row = &x_row_major[i * n_covariates..(i + 1) * n_covariates];
        for j in 0..n_covariates {
            out[j] += row[j] * yi;
        }
    }
}

#[inline]
fn spd_cholesky_with_jitter(matrix: &[f64], dim: usize, label: &str) -> Result<Vec<f64>, String> {
    if dim == 0 {
        return Err(format!("{label} requires dim > 0"));
    }
    if matrix.len() != dim * dim {
        return Err(format!(
            "{label} matrix length mismatch: got {}, expected {}",
            matrix.len(),
            dim * dim
        ));
    }
    let mut chol = matrix.to_vec();
    if cholesky_inplace(&mut chol, dim).is_some() {
        return Ok(chol);
    }
    let trace = (0..dim).map(|i| matrix[i * dim + i].abs()).sum::<f64>();
    let base = (trace / (dim.max(1) as f64)).max(1.0) * 1e-10_f64;
    for k in 0..8 {
        chol.copy_from_slice(matrix);
        let jitter = base * 10.0_f64.powi(k);
        for i in 0..dim {
            chol[i * dim + i] += jitter;
        }
        if cholesky_inplace(&mut chol, dim).is_some() {
            return Ok(chol);
        }
    }
    Err(format!("{label} is not SPD even after diagonal jitter"))
}

#[inline]
fn sparse_diag_stats<V: crate::cholesky::SparseGrmCscView + ?Sized>(
    csc: &V,
    scale: f64,
    diag_add: f64,
) -> Result<(f64, f64, f64), String> {
    if csc.n_samples() == 0 {
        return Err("SparseLMM diagonal stats require n_samples > 0".to_string());
    }
    let mut sum_abs = 0.0_f64;
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;
    for col in 0..csc.n_samples() {
        let start = csc.col_ptr()[col] as usize;
        let end = csc.col_ptr()[col + 1] as usize;
        let mut found_diag = false;
        for idx in start..end {
            if csc.row_indices()[idx] as usize == col {
                let diag = csc.values()[idx] * scale + diag_add;
                if !diag.is_finite() {
                    return Err(format!(
                        "SparseLMM diagonal contains non-finite value at column {col}"
                    ));
                }
                let abs_diag = diag.abs();
                sum_abs += abs_diag;
                if diag < min_diag {
                    min_diag = diag;
                }
                if diag > max_diag {
                    max_diag = diag;
                }
                found_diag = true;
                break;
            }
        }
        if !found_diag {
            return Err(format!("SparseLMM diagonal is missing at column {col}"));
        }
    }
    Ok((
        (sum_abs / (csc.n_samples() as f64)).max(SPLMM_TINY),
        min_diag,
        max_diag,
    ))
}

#[inline]
fn sparse_cholesky_max_l_nnz() -> usize {
    std::env::var("JX_SPARSE_CHOLESKY_MAX_L_NNZ")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(SPLMM_DEFAULT_SPARSE_CHOLESKY_MAX_L_NNZ)
}

#[inline]
fn normalize_sparse_subset_request<'a>(
    sample_idx: &'a [usize],
    full_n: usize,
) -> Result<Option<&'a [usize]>, String> {
    if sample_idx.iter().any(|&sid| sid >= full_n) {
        return Err(format!(
            "SparseLMM sample index out of bounds for sparse n_samples={full_n}"
        ));
    }
    if sample_idx.len() == full_n && sample_idx.iter().enumerate().all(|(i, &sid)| sid == i) {
        Ok(None)
    } else {
        Ok(Some(sample_idx))
    }
}

#[inline]
fn sparse_splmm_load_factor(
    prefix: &str,
    sparse_jxgrm_path: Option<&str>,
    expected_n: usize,
    sample_idx: &[usize],
    sigma_g2: f64,
    sigma_e2: f64,
    progress_callback: Option<&Py<PyAny>>,
) -> Result<SparseJxgrmCholesky, String> {
    let path = sparse_jxgrm_path.map(|s| s.to_string()).unwrap_or_else(|| {
        let spgrm_path = format!("{prefix}.spgrm");
        let jxgrm_path = format!("{prefix}.jxgrm");
        if std::path::Path::new(&jxgrm_path).exists() && !std::path::Path::new(&spgrm_path).exists()
        {
            jxgrm_path
        } else {
            spgrm_path
        }
    });
    if !Path::new(&path).exists() {
        return Err(format!(
            "SparseLMM requires a sparse kinship file, but none was found at: {path}"
        ));
    }
    let sparse_n = sparse_jxgrm_header_n_samples(&path)?;
    let subset_request = normalize_sparse_subset_request(sample_idx, sparse_n)?;
    let target_n = subset_request.map(|idx| idx.len()).unwrap_or(sparse_n);
    if target_n != expected_n {
        return Err(format!(
            "SparseLMM kinship size mismatch: sparse target n={}, expected {} from {path}",
            target_n, expected_n
        ));
    }
    if !sigma_g2.is_finite() || sigma_g2 < 0.0 || !sigma_e2.is_finite() || sigma_e2 < 0.0 {
        return Err(format!(
            "SparseLMM requires finite non-negative variance components, got sigma_g2={sigma_g2}, sigma_e2={sigma_e2}"
        ));
    }
    let analysis = sparse_cholesky_analyze_subset_jxgrm_path_cached_with_progress(
        &path,
        subset_request,
        |stage, done, total| {
            let stage_idx = match stage {
                SparseJxgrmAnalyzeProgressStage::OpenFile => 1usize,
                SparseJxgrmAnalyzeProgressStage::ValidateCsc => 2usize,
                SparseJxgrmAnalyzeProgressStage::DirectSamples => 3usize,
                SparseJxgrmAnalyzeProgressStage::SymbolicAnalyze => 4usize,
            };
            emit_progress_callback(progress_callback, stage_idx, done, total)
        },
    )?;
    if analysis.dim() != expected_n {
        return Err(format!(
            "SparseLMM analyzed dim mismatch: analyzed {}, expected {} from {path}",
            analysis.dim(),
            expected_n
        ));
    }
    let (diag_mean_abs, diag_min, diag_max) = analysis.diag_stats_scaled(sigma_g2, sigma_e2)?;
    let max_l_nnz = sparse_cholesky_max_l_nnz();
    let estimated_l_nnz = analysis.factor_nnz_estimate();
    if estimated_l_nnz > max_l_nnz {
        return Err(format!(
            "SparseLMM symbolic factor is too large for direct Cholesky: estimated L nnz={} exceeds limit {}. \
Set `JX_SPARSE_CHOLESKY_MAX_L_NNZ` to override.",
            estimated_l_nnz, max_l_nnz
        ));
    }
    let rel_shifts = [
        0.0_f64, 1e-12_f64, 1e-10_f64, 1e-8_f64, 1e-6_f64, 1e-5_f64, 1e-4_f64, 1e-3_f64, 1e-2_f64,
        1e-1_f64,
    ];
    emit_progress_callback(progress_callback, 5, 0, rel_shifts.len().max(1))?;
    let mut last_err = None::<String>;
    for (attempt_idx, &rel) in rel_shifts.iter().enumerate() {
        let diag_shift = diag_mean_abs * rel;
        match analysis
            .factorize_sigma_g2_k_plus_sigma_e2_i_with_diag_shift(sigma_g2, sigma_e2, diag_shift)
        {
            Ok(factor) => {
                emit_progress_callback(
                    progress_callback,
                    5,
                    rel_shifts.len().max(1),
                    rel_shifts.len().max(1),
                )?;
                return Ok(factor);
            }
            Err(err) => {
                last_err = Some(err);
                emit_progress_callback(
                    progress_callback,
                    5,
                    (attempt_idx + 1).min(rel_shifts.len().max(1)),
                    rel_shifts.len().max(1),
                )?;
            }
        }
    }
    let last_msg = last_err.unwrap_or_else(|| "unknown sparse Cholesky failure".to_string());
    let sigma_e2_near_boundary = sigma_e2 <= diag_mean_abs * 1e-8_f64;
    let hint = if sigma_e2_near_boundary {
        "sigma_e2 is near zero, so sparse thresholding can make V indefinite; auto-ridge failed. Try a smaller sparse cutoff such as `-splmm 0.01` or `-splmm 0.001`."
    } else {
        "hard-thresholded sparse GRM is not numerically SPD enough for LLT; try a smaller sparse cutoff such as `-splmm 0.01` or `-splmm 0.001`."
    };
    let msg = format!(
        "SparseLMM factorization failed after adaptive diagonal ridge escalation; \
sigma_g2={sigma_g2:.6e}, sigma_e2={sigma_e2:.6e}, mean_diag={diag_mean_abs:.6e}, \
min_diag={diag_min:.6e}, max_diag={diag_max:.6e}. Last error: {last_msg}. Hint: {hint}"
    );
    Err(msg)
}

#[inline]
fn sparse_solve_rhs_with_workspace(
    factor: &SparseJxgrmCholesky,
    rhs_col_major: &[f64],
    n_rhs: usize,
    workspace: &mut SparseJxgrmSolveWorkspace,
) -> Result<Vec<f64>, String> {
    let mut out = rhs_col_major.to_vec();
    factor.solve_in_place_with_workspace(&mut out, n_rhs, workspace)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn append_splmm_tsv_block(
    text_buf: &mut String,
    row_start: usize,
    rows_here: usize,
    results: &[f64],
    chrom: &[String],
    pos: &[i64],
    snp: &[String],
    allele0: &[String],
    allele1: &[String],
    row_maf: &[f32],
    row_missing: &[f32],
    gm: PackedGeneticModel,
) {
    for local_idx in 0..rows_here {
        let row_idx = row_start + local_idx;
        let base = local_idx * 3;
        let beta = results[base];
        let se = results[base + 1];
        let pwald = sanitize_assoc_pvalue(beta, se, results[base + 2]);
        let chisq_txt = format_chisq_value(chisq_from_beta_se_and_optional_plrt(beta, se, None));
        let (a0, a1) = transform_alleles_by_model_local(&allele0[row_idx], &allele1[row_idx], gm);
        let _ = writeln!(
            text_buf,
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4e}",
            chrom[row_idx],
            pos[row_idx],
            snp[row_idx],
            a0,
            a1,
            row_maf[row_idx],
            row_missing[row_idx],
            beta,
            se,
            chisq_txt,
            pwald,
        );
    }
}

#[inline]
fn row_major_snp_block_f32_to_col_major_f64(
    block: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(block.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    // `block` is laid out as rows SNPs × cols samples, row-major:
    //   block[snp * cols + sample]
    // The sparse solver expects a column-major RHS matrix with cols samples as
    // matrix rows and rows SNPs as RHS columns. In column-major storage, each
    // RHS column is contiguous with length `cols`, so the required layout is:
    //   out[snp * cols + sample]
    // Therefore this conversion is a widening copy, not a mathematical transpose.
    for snp in 0..rows {
        let src = &block[snp * cols..(snp + 1) * cols];
        let dst = &mut out[snp * cols..(snp + 1) * cols];
        for sample in 0..cols {
            dst[sample] = src[sample] as f64;
        }
    }
}

#[inline]
fn adaptive_exact_block_rows(requested: usize, n: usize) -> usize {
    // Exact block-scan peak memory model, per row (one SNP) of the block:
    //   block          f32 [n]     = 4n
    //   spy_block      f32 [1]     = 4
    //   rhs_col_major  f64 [n]     = 8n
    //   z_col_major    f64 [n]     = 8n
    //   c_block        f64 [p]     ≈8p   (p is small, typically 1-4)
    //   out_block      f64 [3]     = 24
    //   --------------------------------------------------
    //   total per row              = 20n + 8p + 28  bytes
    //
    // We approximate p=4 for safety and add 25% margin for the sparse-solve
    // workspace and allocation overhead.
    // Default cap: 128 MiB, overridable via JX_SPLMM_EXACT_MAX_BLOCK_BYTES.
    const DEFAULT_MAX_BYTES: usize = 128 * 1024 * 1024; // 128 MiB
    let max_bytes: usize = std::env::var("JX_SPLMM_EXACT_MAX_BLOCK_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_BYTES);
    // nominal: 20n+60 bytes per row, then ×1.25 margin → 25n+75
    let bytes_per_row = (25 * n.max(1)).saturating_add(75);
    let max_rows = max_bytes / bytes_per_row;
    requested.min(max_rows).max(1)
}

// C = X^T Z_block: X is n×p row-major, Z_col_major is n×rows_here column-major.
// c_block is filled with rows_here × p output, row-major.
//
// Uses cblas_dgemm when BLAS is compiled in and the inner-product count
// is large enough to amortise the X layout conversion (row→col major).
// Falls back to a hand-written loop for small n or rows_here.
#[inline]
fn xt_mat_rhs_block_prefers_blas(n: usize, p: usize, rows_here: usize) -> bool {
    // Threshold: total inner-loop iterations at which BLAS overhead is worth it.
    const BLAS_FLOPS_THRESHOLD: usize = 4096; // n * rows_here >= threshold
    n.saturating_mul(rows_here) >= BLAS_FLOPS_THRESHOLD && p > 0 && rows_here > 0 && n > 0
}

#[inline]
fn xt_mat_rhs_block(
    x_design: &[f64],
    x_design_col_major: Option<&[f64]>,
    z_col_major: &[f64],
    n: usize,
    p: usize,
    rows_here: usize,
    c_block: &mut [f64],
) {
    let use_dgemm = xt_mat_rhs_block_prefers_blas(n, p, rows_here);

    if use_dgemm {
        let x_col_storage;
        let x_col = if let Some(cached) = x_design_col_major {
            debug_assert_eq!(cached.len(), p.saturating_mul(n));
            cached
        } else {
            x_col_storage = row_major_to_col_major_f64(x_design, n, p)
                .expect("validated row-major design dimensions must convert to column-major");
            x_col_storage.as_slice()
        };

        // 2) Treat c_block as C_col(p×rows) in column-major. This is byte-identical
        // to rows_here×p row-major, so we can feed it straight into GEMM and then
        // reuse the same buffer as per-SNP contiguous row slices without copying.
        debug_assert_eq!(c_block.len(), rows_here.saturating_mul(p));
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                p as CblasInt,
                rows_here as CblasInt,
                n as CblasInt,
                1.0_f64,
                x_col.as_ptr(),
                p as CblasInt,
                z_col_major.as_ptr(),
                n as CblasInt,
                0.0_f64,
                c_block.as_mut_ptr(),
                p as CblasInt,
            );
        }
        return;
    }

    // Fallback hand-written loop.
    c_block.fill(0.0);
    for i in 0..n {
        let xi = &x_design[i * p..(i + 1) * p];
        for j in 0..rows_here {
            let z_ji = z_col_major[j * n + i];
            if z_ji != 0.0 {
                let c_j = &mut c_block[j * p..(j + 1) * p];
                for k in 0..p {
                    c_j[k] += xi[k] * z_ji;
                }
            }
        }
    }
}

#[inline]
fn exact_pg_is_numerically_singular(s_p_s: f64, s_m_s: f64) -> bool {
    s_p_s.is_finite()
        && s_m_s.is_finite()
        && s_p_s > 0.0
        && s_m_s > SPLMM_TINY
        && s_p_s <= SPLMM_EXACT_PG_MIN_REL_SM * s_m_s.max(1.0)
}

#[inline]
fn splmm_effective_sigma2(sigma_g2: f64, sigma_e2: f64) -> Result<f64, String> {
    if sigma_g2.is_finite() && sigma_g2 > 0.0 {
        return Ok(sigma_g2);
    }
    if sigma_e2.is_finite() && sigma_e2 > 0.0 {
        return Ok(sigma_e2);
    }
    Err(format!(
        "SparseLMM requires finite positive trait scale from sigma_g2/sigma_e2, got sigma_g2={sigma_g2}, sigma_e2={sigma_e2}"
    ))
}

#[inline]
fn splmm_wald_from_scaled_denom(
    score: f64,
    denom_scaled: f64,
    wald_sigma2: f64,
) -> Option<(f64, f64, f64)> {
    if !(score.is_finite()
        && denom_scaled.is_finite()
        && denom_scaled > SPLMM_TINY
        && wald_sigma2.is_finite()
        && wald_sigma2 > 0.0)
    {
        return None;
    }
    let beta = score / denom_scaled;
    let var_beta = wald_sigma2 / denom_scaled;
    if !(beta.is_finite() && var_beta.is_finite() && var_beta > 0.0) {
        return None;
    }
    let se = var_beta.sqrt();
    let chisq = (score * score) / (wald_sigma2 * denom_scaled);
    if !(se.is_finite() && se > 0.0 && chisq.is_finite() && chisq >= 0.0) {
        return None;
    }
    Some((beta, se, chi2_sf_df1(chisq)))
}

/// Tiled sparse solve using JanusX's cached thread pool (when available).
/// `workspaces` are pre-allocated once and reused across blocks.
#[inline]
fn sparse_solve_rhs_tiled(
    factor: &SparseJxgrmCholesky,
    rhs_col_major: &mut [f64],
    n_rhs: usize,
    workspaces: &mut [SparseJxgrmSolveWorkspace],
    pool: Option<&rayon::ThreadPool>,
) -> Result<(), String> {
    factor.solve_in_place_tiled(rhs_col_major, n_rhs, workspaces, pool)
}

// Core block-scan loop for exact g'Pg denominator.
// Works with any GenotypeMatrix backend via UnifiedInput.
fn exact_scan_blocks_core<G: GenotypeMatrix>(
    factor: &SparseJxgrmCholesky,
    input: &mut UnifiedInput<G>,
    x_design: &[f64],
    x_design_col_major: &[f64],
    x_design_xtx_chol: &[f64],
    score_vec: &[f64],
    score_scale: f64,
    xt_den_x_chol: &[f64],
    denom_scale: f64,
    wald_sigma2: f64,
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    threads: usize,
    block_rows: usize,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    progress_done_offset: usize,
    progress_total_override: usize,
    _solve_workspace: &mut SparseJxgrmSolveWorkspace,
    sink: &mut dyn FnMut(usize, usize, &[f64]) -> Result<(), String>,
) -> Result<(), String> {
    let stage_timing = splmm_packed_stage_timing_enabled();
    let total_t0 = stage_timing.then(Instant::now);
    let mut timing = SplmmPackedScanTiming::default();
    if !(score_scale.is_finite() && score_scale > 0.0) {
        return Err(format!(
            "SparseLMM exact scan requires finite positive score scale, got {score_scale}"
        ));
    }
    if !(denom_scale.is_finite() && denom_scale > 0.0) {
        return Err(format!(
            "SparseLMM exact scan requires finite positive denominator scale, got {denom_scale}"
        ));
    }
    if !(wald_sigma2.is_finite() && wald_sigma2 > 0.0) {
        return Err(format!(
            "SparseLMM exact scan requires finite positive Wald sigma2, got {wald_sigma2}"
        ));
    }
    let n = score_vec.len();
    if n == 0 {
        return Err("SparseLMM exact scan requires non-empty score vector".to_string());
    }
    if x_design.len() % n != 0 {
        return Err(format!(
            "SparseLMM exact scan design length mismatch: len(x_design)={} is not divisible by n={n}",
            x_design.len()
        ));
    }
    let p = x_design.len() / n;
    if p == 0 {
        return Err("SparseLMM exact scan requires at least one design column".to_string());
    }
    if xt_den_x_chol.len() != p * p {
        return Err(format!(
            "SparseLMM exact scan denominator chol length mismatch: got {}, expected {}",
            xt_den_x_chol.len(),
            p * p
        ));
    }
    if x_design_xtx_chol.len() != p * p {
        return Err(format!(
            "SparseLMM exact scan XtX chol length mismatch: got {}, expected {}",
            x_design_xtx_chol.len(),
            p * p
        ));
    }
    if x_design_col_major.len() != p * n {
        return Err(format!(
            "SparseLMM exact scan X_col_major length mismatch: got {}, expected {}",
            x_design_col_major.len(),
            p * n
        ));
    }
    if scan_sample_idx.len() != n {
        return Err(format!(
            "SparseLMM exact scan sample length mismatch: sample_idx={}, expected {n}",
            scan_sample_idx.len()
        ));
    }

    let sample_identity = scan_sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let m = input.n_markers();
    let progress_total = if progress_total_override == 0 {
        m.max(1)
    } else {
        progress_total_override.max(1)
    };
    let progress_block = if progress_every == 0 {
        block_rows.max(512).max(1)
    } else {
        progress_every.max(1)
    };
    if scan_progress_callback.is_some() {
        emit_progress_callback(
            scan_progress_callback,
            progress_stage,
            progress_done_offset.min(progress_total),
            progress_total,
        )?;
    }
    if !matches!(gm, PackedGeneticModel::Add) {
        return Err("SparseLMM exact denominator mode requires additive model".to_string());
    }

    let scan_block_rows = adaptive_exact_block_rows(
        adaptive_grm_block_rows(progress_block, m, n, 0usize, threads).max(1),
        n,
    );
    let score_f32 = cast_f64_slice_to_f32(score_vec)?;
    let x_design_f32 = cast_f64_slice_to_f32(x_design)?;
    let blas_threads = if rust_sgemm_prefers_rayon_rowmajor_f32_kernel() {
        1
    } else {
        threads.max(1)
    };
    let _blas_guard = BlasThreadGuard::enter(blas_threads);

    let mut block = vec![0.0_f32; scan_block_rows * n];
    let mut spy_block = vec![0.0_f32; scan_block_rows];
    let mut xts_block = vec![0.0_f32; scan_block_rows * p];
    let mut row_ss = vec![0.0_f64; scan_block_rows];
    let mut rhs_col_major = vec![0.0_f64; scan_block_rows * n];
    let mut z_col_major = vec![0.0_f64; scan_block_rows * n];
    let mut c_block = vec![0.0_f64; scan_block_rows * p];
    let mut xts = vec![0.0_f64; p];
    let mut alpha = vec![0.0_f64; p];
    let mut c_scaled = vec![0.0_f64; p];
    let mut alpha_xtx = vec![0.0_f64; p];
    let mut out_block = vec![0.0_f64; scan_block_rows * 3];

    const MIN_TILE_COLS: usize = 32;
    let solve_tiles = if threads > 1 && scan_block_rows >= MIN_TILE_COLS {
        threads.min(scan_block_rows.div_ceil(MIN_TILE_COLS).max(1))
    } else {
        1
    };
    let tile_cols_capacity = scan_block_rows.div_ceil(solve_tiles);
    let mut tiled_workspaces: Vec<SparseJxgrmSolveWorkspace> = (0..solve_tiles)
        .map(|_| factor.make_solve_workspace(tile_cols_capacity.max(1)))
        .collect::<Result<Vec<_>, _>>()?;

    let mut row_start = 0usize;
    while row_start < m {
        let row_end = (row_start + scan_block_rows).min(m);
        let rows_here = row_end - row_start;
        let block_slice = &mut block[..rows_here * n];
        let spy_slice = &mut spy_block[..rows_here];
        let xts_slice = &mut xts_block[..rows_here * p];
        let ss_slice = &mut row_ss[..rows_here];

        // Step 1: decode genotype block via UnifiedInput
        let t0 = stage_timing.then(Instant::now);
        input.matrix.decode_additive_block(
            &input.stats,
            row_start,
            block_slice,
            scan_sample_idx,
            sample_identity,
            pool.as_ref(),
        )?;
        if let Some(t0) = t0 {
            timing.decode_secs += t0.elapsed().as_secs_f64();
        }

        // Step 2: numerator = G_block * Py
        let t0 = stage_timing.then(Instant::now);
        row_major_block_mul_mat_f32(
            block_slice,
            rows_here,
            n,
            score_f32.as_slice(),
            1,
            spy_slice,
            pool.as_ref(),
        );
        if let Some(t0) = t0 {
            timing.gpy_secs += t0.elapsed().as_secs_f64();
        }
        let t0 = stage_timing.then(Instant::now);
        row_major_block_mul_mat_f32(
            block_slice,
            rows_here,
            n,
            x_design_f32.as_slice(),
            p,
            xts_slice,
            pool.as_ref(),
        );
        if let Some(t0) = t0 {
            timing.gx_secs += t0.elapsed().as_secs_f64();
        }
        let t0 = stage_timing.then(Instant::now);
        row_major_block_sumsq_f64(block_slice, rows_here, n, ss_slice, pool.as_ref());
        if let Some(t0) = t0 {
            timing.row_sumsq_secs += t0.elapsed().as_secs_f64();
        }

        // Step 3: convert to column-major f64 for sparse solve
        let rhs_slice = &mut rhs_col_major[..rows_here * n];
        let t0 = stage_timing.then(Instant::now);
        row_major_snp_block_f32_to_col_major_f64(block_slice, rows_here, n, rhs_slice);
        if let Some(t0) = t0 {
            timing.repack_secs += t0.elapsed().as_secs_f64();
        }

        // Step 4: solve V * Z_block = G_block (tiled, in-place)
        let z_slice = &mut z_col_major[..rows_here * n];
        z_slice.copy_from_slice(rhs_slice);
        let t0 = stage_timing.then(Instant::now);
        sparse_solve_rhs_tiled(
            factor,
            z_slice,
            rows_here,
            &mut tiled_workspaces,
            pool.as_deref(),
        )?;
        if let Some(t0) = t0 {
            timing.solve_secs += t0.elapsed().as_secs_f64();
        }

        // Step 5: C_block = X^T Z_block
        let c_slice = &mut c_block[..rows_here * p];
        let t0 = stage_timing.then(Instant::now);
        xt_mat_rhs_block(
            x_design,
            Some(x_design_col_major),
            z_slice,
            n,
            p,
            rows_here,
            c_slice,
        );
        if let Some(t0) = t0 {
            timing.xtz_secs += t0.elapsed().as_secs_f64();
        }

        // Steps 6-8: per-SNP denominator and output
        let out_slice = &mut out_block[..rows_here * 3];
        let t0 = stage_timing.then(Instant::now);
        for local_idx in 0..rows_here {
            let g = &rhs_slice[local_idx * n..(local_idx + 1) * n];
            let z = &z_slice[local_idx * n..(local_idx + 1) * n];
            let s_v_s = g.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>();
            let c_j = &c_slice[local_idx * p..(local_idx + 1) * p];
            let xts_row = &xts_slice[local_idx * p..(local_idx + 1) * p];
            for j in 0..p {
                xts[j] = xts_row[j] as f64;
            }
            let s_m_s = residualized_sumsq_from_xtx_chol(
                x_design_xtx_chol,
                p,
                &xts,
                &mut alpha_xtx,
                ss_slice[local_idx],
            );
            for j in 0..p {
                c_scaled[j] = c_j[j] * denom_scale;
            }
            cholesky_solve_into(xt_den_x_chol, p, &c_scaled, &mut alpha);
            let x_quad = c_scaled
                .iter()
                .zip(alpha.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let denom_scaled = (denom_scale * s_v_s - x_quad).max(0.0);
            let out_row = &mut out_slice[local_idx * 3..(local_idx + 1) * 3];
            let denom_full = denom_scaled / wald_sigma2;
            if exact_pg_is_numerically_singular(denom_full, s_m_s) {
                out_row[0] = f64::NAN;
                out_row[1] = f64::NAN;
                out_row[2] = 1.0_f64;
            } else if let Some((beta, se, pwald)) = splmm_wald_from_scaled_denom(
                score_scale * (spy_slice[local_idx] as f64),
                denom_scaled,
                wald_sigma2,
            ) {
                out_row[0] = beta;
                out_row[1] = se;
                out_row[2] = pwald;
            } else {
                out_row[0] = f64::NAN;
                out_row[1] = f64::NAN;
                out_row[2] = 1.0_f64;
            }
        }
        if let Some(t0) = t0 {
            timing.denom_secs += t0.elapsed().as_secs_f64();
        }
        let t0 = stage_timing.then(Instant::now);
        sink(row_start, rows_here, out_slice)?;
        if let Some(t0) = t0 {
            timing.sink_secs += t0.elapsed().as_secs_f64();
        }
        emit_progress_callback(
            scan_progress_callback,
            progress_stage,
            (progress_done_offset + row_end).min(progress_total),
            progress_total,
        )?;
        row_start = row_end;
    }
    if let Some(total_t0) = total_t0 {
        emit_splmm_packed_scan_timing(
            "exact",
            &timing,
            total_t0.elapsed().as_secs_f64(),
            m,
            n,
            p,
            threads,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scan_with_p0y_and_exact_p0_sparse(
    factor: &SparseJxgrmCholesky,
    scan_prepared: &JxlmmPreparedInput,
    x_design: &[f64],
    x_design_col_major: &[f64],
    x_design_xtx_chol: &[f64],
    p0y_vec: &[f64],
    xt_p0_x_chol: &[f64],
    sigma2: f64,
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    threads: usize,
    block_rows: usize,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    progress_done_offset: usize,
    progress_total_override: usize,
    solve_workspace: &mut SparseJxgrmSolveWorkspace,
) -> Result<Vec<f64>, String> {
    let score_vec = if sigma2 == 1.0_f64 {
        p0y_vec.to_vec()
    } else {
        p0y_vec.iter().map(|v| *v / sigma2).collect::<Vec<_>>()
    };
    let m = scan_prepared.n_rows();
    let mut out = vec![0.0_f64; m * 3];
    let mut memory_sink = |row_start: usize, rows_here: usize, block: &[f64]| {
        out[row_start * 3..][..rows_here * 3].copy_from_slice(block);
        Ok(())
    };
    let adapter = JxlmmMatrixAdapter {
        inner: scan_prepared,
    };
    let stats = unified_stats_from_jxlmm(scan_prepared);
    let mut input = UnifiedInput {
        matrix: adapter,
        stats,
    };
    exact_scan_blocks_core(
        factor,
        &mut input,
        x_design,
        x_design_col_major,
        x_design_xtx_chol,
        score_vec.as_slice(),
        sigma2,
        xt_p0_x_chol,
        sigma2,
        sigma2,
        scan_sample_idx,
        gm,
        threads,
        block_rows,
        scan_progress_callback,
        progress_every,
        progress_stage,
        progress_done_offset,
        progress_total_override,
        solve_workspace,
        &mut memory_sink,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn grammar_scan_blocks_core<G: GenotypeMatrix>(
    input: &mut UnifiedInput<G>,
    x_design: &[f64],
    score_vec: &[f64],
    score_scale: f64,
    xtx_chol: &[f64],
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    denom_scale: f64,
    wald_sigma2: f64,
    threads: usize,
    block_rows: usize,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    progress_done_offset: usize,
    progress_total_override: usize,
    sink: &mut dyn FnMut(usize, usize, &[f64]) -> Result<(), String>,
) -> Result<(), String> {
    let stage_timing = splmm_packed_stage_timing_enabled();
    let total_t0 = stage_timing.then(Instant::now);
    let mut timing = SplmmPackedScanTiming::default();
    if !(score_scale.is_finite() && score_scale > 0.0) {
        return Err(format!(
            "SparseLMM scan requires finite positive score scale, got {score_scale}"
        ));
    }
    if !(denom_scale.is_finite() && denom_scale > 0.0) {
        return Err(format!(
            "SparseLMM scan requires finite positive denominator scale, got {denom_scale}"
        ));
    }
    if !(wald_sigma2.is_finite() && wald_sigma2 > 0.0) {
        return Err(format!(
            "SparseLMM scan requires finite positive Wald sigma2, got {wald_sigma2}"
        ));
    }
    let n = score_vec.len();
    if n == 0 {
        return Err("SparseLMM scan requires non-empty score vector".to_string());
    }
    if x_design.len() % n != 0 {
        return Err(format!(
            "SparseLMM scan design length mismatch: len(x_design)={} is not divisible by n={n}",
            x_design.len()
        ));
    }
    let p = x_design.len() / n;
    if p == 0 {
        return Err("SparseLMM scan requires at least one design column".to_string());
    }
    if xtx_chol.len() != p * p {
        return Err(format!(
            "SparseLMM scan XtX chol length mismatch: got {}, expected {}",
            xtx_chol.len(),
            p * p
        ));
    }
    if scan_sample_idx.len() != n {
        return Err(format!(
            "SparseLMM scan sample length mismatch: sample_idx={}, expected {n}",
            scan_sample_idx.len()
        ));
    }

    let sample_identity = scan_sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
    let sample_byte_idx: Option<Vec<usize>> = if sample_identity {
        None
    } else {
        Some(scan_sample_idx.iter().map(|&sid| sid >> 2).collect())
    };
    let sample_bit_shift: Option<Vec<u8>> = if sample_identity {
        None
    } else {
        Some(
            scan_sample_idx
                .iter()
                .map(|&sid| ((sid & 3) << 1) as u8)
                .collect(),
        )
    };

    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let m = input.n_markers();
    let progress_total = if progress_total_override == 0 {
        m.max(1)
    } else {
        progress_total_override.max(1)
    };
    let progress_block = if progress_every == 0 {
        block_rows.max(512).max(1)
    } else {
        progress_every.max(1)
    };
    if scan_progress_callback.is_some() {
        emit_progress_callback(
            scan_progress_callback,
            progress_stage,
            progress_done_offset.min(progress_total),
            progress_total,
        )?;
    }

    if matches!(gm, PackedGeneticModel::Add) {
        let scan_block_rows = adaptive_grm_block_rows(progress_block, m, n, 0usize, threads).max(1);
        let score_f32 = cast_f64_slice_to_f32(score_vec)?;
        let x_design_f32 = cast_f64_slice_to_f32(x_design)?;
        let blas_threads = if rust_sgemm_prefers_rayon_rowmajor_f32_kernel() {
            1usize
        } else {
            threads.max(1)
        };
        let _blas_guard = BlasThreadGuard::enter(blas_threads);
        let mut block = vec![0.0_f32; scan_block_rows * n];
        let mut spy_block = vec![0.0_f32; scan_block_rows];
        let mut xts_block = vec![0.0_f32; scan_block_rows * p];
        let mut row_ss = vec![0.0_f64; scan_block_rows];
        let mut xts = vec![0.0_f64; p];
        let mut alpha = vec![0.0_f64; p];
        let mut out_block = vec![0.0_f64; scan_block_rows * 3];
        let mut row_start = 0usize;
        while row_start < m {
            let row_end = (row_start + scan_block_rows).min(m);
            let rows_here = row_end - row_start;
            let block_slice = &mut block[..rows_here * n];
            let spy_slice = &mut spy_block[..rows_here];
            let xts_slice = &mut xts_block[..rows_here * p];
            let ss_slice = &mut row_ss[..rows_here];
            // Decode via UnifiedInput
            let t0 = stage_timing.then(Instant::now);
            input.matrix.decode_additive_block(
                &input.stats,
                row_start,
                block_slice,
                scan_sample_idx,
                sample_identity,
                pool.as_ref(),
            )?;
            if let Some(t0) = t0 {
                timing.decode_secs += t0.elapsed().as_secs_f64();
            }
            let t0 = stage_timing.then(Instant::now);
            row_major_block_mul_mat_f32(
                block_slice,
                rows_here,
                n,
                score_f32.as_slice(),
                1usize,
                spy_slice,
                pool.as_ref(),
            );
            if let Some(t0) = t0 {
                timing.gpy_secs += t0.elapsed().as_secs_f64();
            }
            let t0 = stage_timing.then(Instant::now);
            row_major_block_mul_mat_f32(
                block_slice,
                rows_here,
                n,
                x_design_f32.as_slice(),
                p,
                xts_slice,
                pool.as_ref(),
            );
            if let Some(t0) = t0 {
                timing.gx_secs += t0.elapsed().as_secs_f64();
            }
            let t0 = stage_timing.then(Instant::now);
            row_major_block_sumsq_f64(block_slice, rows_here, n, ss_slice, pool.as_ref());
            if let Some(t0) = t0 {
                timing.row_sumsq_secs += t0.elapsed().as_secs_f64();
            }
            let out_slice = &mut out_block[..rows_here * 3];
            let t0 = stage_timing.then(Instant::now);
            for local_idx in 0..rows_here {
                let xts_row = &xts_slice[local_idx * p..(local_idx + 1) * p];
                for j in 0..p {
                    xts[j] = xts_row[j] as f64;
                }
                let s_m_s = residualized_sumsq_from_xtx_chol(
                    xtx_chol,
                    p,
                    &xts,
                    &mut alpha,
                    ss_slice[local_idx],
                );
                let out_row = &mut out_slice[local_idx * 3..(local_idx + 1) * 3];
                let score = score_scale * (spy_slice[local_idx] as f64);
                let denom_scaled = denom_scale * s_m_s;
                if let Some((beta, se, pwald)) =
                    splmm_wald_from_scaled_denom(score, denom_scaled, wald_sigma2)
                {
                    out_row[0] = beta;
                    out_row[1] = se;
                    out_row[2] = pwald;
                } else {
                    out_row[0] = f64::NAN;
                    out_row[1] = f64::NAN;
                    out_row[2] = 1.0_f64;
                }
            }
            if let Some(t0) = t0 {
                timing.denom_secs += t0.elapsed().as_secs_f64();
            }
            let t0 = stage_timing.then(Instant::now);
            sink(row_start, rows_here, out_slice)?;
            if let Some(t0) = t0 {
                timing.sink_secs += t0.elapsed().as_secs_f64();
            }
            emit_progress_callback(
                scan_progress_callback,
                progress_stage,
                (progress_done_offset + row_end).min(progress_total),
                progress_total,
            )?;
            row_start = row_end;
        }
    } else {
        let thread_hint = threads.max(1);
        let row_tile = progress_block.div_ceil(thread_hint).clamp(32, 1024);
        let mut out_block = vec![0.0_f64; progress_block * 3];
        let mut row_start = 0usize;
        while row_start < m {
            let row_end = (row_start + progress_block).min(m);
            let rows_here = row_end - row_start;
            let out_slice = &mut out_block[..rows_here * 3];
            let mut run_block = || {
                out_slice
                    .par_chunks_mut(row_tile * 3)
                    .enumerate()
                    .for_each_init(
                        || (vec![0.0_f64; n], vec![0.0_f64; p], vec![0.0_f64; p]),
                        |(snp, xts, alpha), (tile_idx, out_tile)| {
                            let tile_row_start = row_start + tile_idx * row_tile;
                            for (local_idx, out_row) in out_tile.chunks_mut(3).enumerate() {
                                let row_idx = tile_row_start + local_idx;
                                let src_idx = input.stats.row_source_indices[row_idx];
                                let row = input.matrix.source_row_bytes(src_idx);
                                decode_packed_row_model_into_f64(
                                    row,
                                    input.stats.row_flip[row_idx],
                                    input.stats.maf[row_idx],
                                    n,
                                    gm,
                                    sample_identity,
                                    sample_byte_idx.as_deref(),
                                    sample_bit_shift.as_deref(),
                                    snp,
                                );
                                xts.fill(0.0);
                                let mut score = 0.0_f64;
                                let mut s_sq = 0.0_f64;
                                for i in 0..n {
                                    let s_i = snp[i];
                                    score += s_i * score_vec[i];
                                    s_sq += s_i * s_i;
                                    let x_row = &x_design[i * p..(i + 1) * p];
                                    for j in 0..p {
                                        xts[j] += x_row[j] * s_i;
                                    }
                                }
                                let s_m_s =
                                    residualized_sumsq_from_xtx_chol(xtx_chol, p, xts, alpha, s_sq);
                                let denom_scaled = denom_scale * s_m_s;
                                let score = score_scale * score;
                                if let Some((beta, se, pwald)) =
                                    splmm_wald_from_scaled_denom(score, denom_scaled, wald_sigma2)
                                {
                                    out_row[0] = beta;
                                    out_row[1] = se;
                                    out_row[2] = pwald;
                                } else {
                                    out_row[0] = f64::NAN;
                                    out_row[1] = f64::NAN;
                                    out_row[2] = 1.0;
                                }
                            }
                        },
                    );
            };
            if let Some(tp) = &pool {
                tp.install(run_block);
            } else {
                run_block();
            }
            sink(row_start, rows_here, &out_block[..rows_here * 3])?;
            emit_progress_callback(
                scan_progress_callback,
                progress_stage,
                (progress_done_offset + row_end).min(progress_total),
                progress_total,
            )?;
            row_start = row_end;
        }
    }

    if let Some(total_t0) = total_t0 {
        emit_splmm_packed_scan_timing(
            "approx",
            &timing,
            total_t0.elapsed().as_secs_f64(),
            m,
            n,
            p,
            threads,
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
// GRAMMAR-gamma scan — in-memory sink writes into pre-allocated Vec<f64>.
fn scan_with_py_and_rhat(
    scan_prepared: &JxlmmPreparedInput,
    x_design: &[f64],
    py_vec: &[f64],
    xtx_chol: &[f64],
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    r_hat: f64,
    threads: usize,
    block_rows: usize,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    progress_done_offset: usize,
    progress_total_override: usize,
) -> Result<Vec<f64>, String> {
    let adapter = JxlmmMatrixAdapter {
        inner: scan_prepared,
    };
    let stats = unified_stats_from_jxlmm(scan_prepared);
    let mut input = UnifiedInput {
        matrix: adapter,
        stats,
    };
    let m = input.n_markers();
    let mut out = vec![0.0_f64; m * 3];
    let mut memory_sink = |row_start: usize, rows_here: usize, block: &[f64]| {
        out[row_start * 3..][..rows_here * 3].copy_from_slice(block);
        Ok(())
    };
    grammar_scan_blocks_core(
        &mut input,
        x_design,
        py_vec,
        1.0_f64,
        xtx_chol,
        scan_sample_idx,
        gm,
        r_hat,
        1.0_f64,
        threads,
        block_rows,
        scan_progress_callback,
        progress_every,
        progress_stage,
        progress_done_offset,
        progress_total_override,
        &mut memory_sink,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn scan_with_p0y_and_gamma0(
    scan_prepared: &JxlmmPreparedInput,
    x_design: &[f64],
    p0y_vec: &[f64],
    xtx_chol: &[f64],
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    gamma0: f64,
    sigma2: f64,
    threads: usize,
    block_rows: usize,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    progress_done_offset: usize,
    progress_total_override: usize,
) -> Result<Vec<f64>, String> {
    let score_vec = if sigma2 == 1.0_f64 {
        p0y_vec.to_vec()
    } else {
        p0y_vec.iter().map(|v| *v / sigma2).collect::<Vec<_>>()
    };
    let adapter = JxlmmMatrixAdapter {
        inner: scan_prepared,
    };
    let stats = unified_stats_from_jxlmm(scan_prepared);
    let mut input = UnifiedInput {
        matrix: adapter,
        stats,
    };
    let m = input.n_markers();
    let mut out = vec![0.0_f64; m * 3];
    let mut memory_sink = |row_start: usize, rows_here: usize, block: &[f64]| {
        out[row_start * 3..][..rows_here * 3].copy_from_slice(block);
        Ok(())
    };
    grammar_scan_blocks_core(
        &mut input,
        x_design,
        score_vec.as_slice(),
        sigma2,
        xtx_chol,
        scan_sample_idx,
        gm,
        gamma0,
        sigma2,
        threads,
        block_rows,
        scan_progress_callback,
        progress_every,
        progress_stage,
        progress_done_offset,
        progress_total_override,
        &mut memory_sink,
    )?;
    Ok(out)
}

fn scan_to_tsv_with_p0y_and_gamma0(
    scan_prepared: &JxlmmPreparedInput,
    x_design: &[f64],
    p0y_vec: &[f64],
    xtx_chol: &[f64],
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    gamma0: f64,
    sigma2: f64,
    threads: usize,
    block_rows: usize,
    chrom: &[String],
    pos: &[i64],
    snp: &[String],
    allele0: &[String],
    allele1: &[String],
    out_tsv: &str,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
) -> Result<usize, String> {
    let stage_timing = splmm_packed_stage_timing_enabled();
    let mut tsv_timing = SplmmTsvTiming::default();
    let score_vec = if sigma2 == 1.0_f64 {
        p0y_vec.to_vec()
    } else {
        p0y_vec.iter().map(|v| *v / sigma2).collect::<Vec<_>>()
    };
    if chrom.len() != scan_prepared.n_rows()
        || pos.len() != scan_prepared.n_rows()
        || snp.len() != scan_prepared.n_rows()
        || allele0.len() != scan_prepared.n_rows()
        || allele1.len() != scan_prepared.n_rows()
    {
        return Err(format!(
            "SparseLMM TSV metadata length mismatch: rows={}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            scan_prepared.n_rows(),
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        ));
    }
    let adapter = JxlmmMatrixAdapter {
        inner: scan_prepared,
    };
    let stats = unified_stats_from_jxlmm(scan_prepared);
    let mut input = UnifiedInput {
        matrix: adapter,
        stats,
    };
    let row_maf = input.stats.maf.clone();
    let row_miss = input.stats.miss.clone();
    let m = input.n_markers();
    let writer = AsyncTsvWriter::with_config(
        out_tsv,
        b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\n",
        64 * 1024 * 1024,
        4,
    )
    .map_err(|e| e.to_string())?;
    let mut text_buf = String::with_capacity(block_rows.max(512).max(1) * 112);
    let mut tsv_sink = |row_start: usize, rows_here: usize, block: &[f64]| {
        text_buf.clear();
        let t0 = stage_timing.then(Instant::now);
        append_splmm_tsv_block(
            &mut text_buf,
            row_start,
            rows_here,
            block,
            chrom,
            pos,
            snp,
            allele0,
            allele1,
            &row_maf,
            &row_miss,
            gm,
        );
        if let Some(t0) = t0 {
            tsv_timing.format_secs += t0.elapsed().as_secs_f64();
        }
        let payload = std::mem::take(&mut text_buf).into_bytes();
        let payload_len = payload.len();
        let t0 = stage_timing.then(Instant::now);
        writer.send(payload)?;
        if let Some(t0) = t0 {
            tsv_timing.send_secs += t0.elapsed().as_secs_f64();
        }
        tsv_timing.blocks = tsv_timing.blocks.saturating_add(1);
        tsv_timing.bytes = tsv_timing.bytes.saturating_add(payload_len);
        Ok(())
    };
    let run_res = grammar_scan_blocks_core(
        &mut input,
        x_design,
        score_vec.as_slice(),
        sigma2,
        xtx_chol,
        scan_sample_idx,
        gm,
        gamma0,
        sigma2,
        threads,
        block_rows,
        scan_progress_callback,
        progress_every,
        progress_stage,
        0,
        0,
        &mut tsv_sink,
    );
    let t0 = stage_timing.then(Instant::now);
    let writer_result = writer.finish();
    if let Some(t0) = t0 {
        tsv_timing.finish_secs += t0.elapsed().as_secs_f64();
    }
    if stage_timing {
        emit_splmm_tsv_timing("approx", &tsv_timing, m);
    }
    run_res?;
    writer_result?;
    Ok(m)
}

fn scan_to_tsv_with_p0y_and_exact_p0_sparse(
    factor: &SparseJxgrmCholesky,
    scan_prepared: &JxlmmPreparedInput,
    x_design: &[f64],
    x_design_col_major: &[f64],
    x_design_xtx_chol: &[f64],
    p0y_vec: &[f64],
    xt_p0_x_chol: &[f64],
    sigma2: f64,
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    threads: usize,
    block_rows: usize,
    chrom: &[String],
    pos: &[i64],
    snp: &[String],
    allele0: &[String],
    allele1: &[String],
    out_tsv: &str,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    solve_workspace: &mut SparseJxgrmSolveWorkspace,
) -> Result<usize, String> {
    let stage_timing = splmm_packed_stage_timing_enabled();
    let mut tsv_timing = SplmmTsvTiming::default();
    let score_vec = if sigma2 == 1.0_f64 {
        p0y_vec.to_vec()
    } else {
        p0y_vec.iter().map(|v| *v / sigma2).collect::<Vec<_>>()
    };
    if chrom.len() != scan_prepared.n_rows()
        || pos.len() != scan_prepared.n_rows()
        || snp.len() != scan_prepared.n_rows()
        || allele0.len() != scan_prepared.n_rows()
        || allele1.len() != scan_prepared.n_rows()
    {
        return Err(format!(
            "SparseLMM TSV metadata length mismatch: rows={}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            scan_prepared.n_rows(),
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        ));
    }
    let m = scan_prepared.n_rows();
    let writer = AsyncTsvWriter::with_config(
        out_tsv,
        b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\n",
        64 * 1024 * 1024,
        4,
    )
    .map_err(|e| e.to_string())?;

    let mut text_buf = String::with_capacity(block_rows.max(512).max(1) * 112);
    let mut tsv_sink = |row_start: usize, rows_here: usize, block: &[f64]| {
        text_buf.clear();
        let t0 = stage_timing.then(Instant::now);
        append_splmm_tsv_block(
            &mut text_buf,
            row_start,
            rows_here,
            block,
            chrom,
            pos,
            snp,
            allele0,
            allele1,
            &scan_prepared.row_maf,
            &scan_prepared.row_missing,
            gm,
        );
        if let Some(t0) = t0 {
            tsv_timing.format_secs += t0.elapsed().as_secs_f64();
        }
        let payload = std::mem::take(&mut text_buf).into_bytes();
        let payload_len = payload.len();
        let t0 = stage_timing.then(Instant::now);
        writer.send(payload)?;
        if let Some(t0) = t0 {
            tsv_timing.send_secs += t0.elapsed().as_secs_f64();
        }
        tsv_timing.blocks = tsv_timing.blocks.saturating_add(1);
        tsv_timing.bytes = tsv_timing.bytes.saturating_add(payload_len);
        Ok(())
    };
    let adapter = JxlmmMatrixAdapter {
        inner: scan_prepared,
    };
    let stats = unified_stats_from_jxlmm(scan_prepared);
    let mut input = UnifiedInput {
        matrix: adapter,
        stats,
    };
    let run_res = exact_scan_blocks_core(
        factor,
        &mut input,
        x_design,
        x_design_col_major,
        x_design_xtx_chol,
        score_vec.as_slice(),
        sigma2,
        xt_p0_x_chol,
        sigma2,
        sigma2,
        scan_sample_idx,
        gm,
        threads,
        block_rows,
        scan_progress_callback,
        progress_every,
        progress_stage,
        0,
        0,
        solve_workspace,
        &mut tsv_sink,
    );
    let t0 = stage_timing.then(Instant::now);
    let writer_result = writer.finish();
    if let Some(t0) = t0 {
        tsv_timing.finish_secs += t0.elapsed().as_secs_f64();
    }
    if stage_timing {
        emit_splmm_tsv_timing("exact", &tsv_timing, m);
    }
    run_res?;
    writer_result?;
    Ok(m)
}

#[inline]
fn splmm_two_stage_p_threshold(n_snps: usize) -> f64 {
    if n_snps == 0 {
        SPLMM_TWO_STAGE_P_THRESHOLD_FLOOR
    } else {
        SPLMM_TWO_STAGE_P_THRESHOLD_FLOOR.max(SPLMM_TWO_STAGE_P_THRESHOLD_NUM / (n_snps as f64))
    }
}

fn select_two_stage_candidates(results: &[f64], p_threshold: f64) -> Vec<usize> {
    let m = results.len() / 3;
    let mut keep = Vec::new();
    keep.reserve(m.min(4096));
    for row_idx in 0..m {
        let pwald = results[row_idx * 3 + 2];
        if pwald.is_finite() && pwald <= p_threshold {
            keep.push(row_idx);
        }
    }
    keep
}

#[inline]
fn splmm_exact_row_is_valid(row: &[f64]) -> bool {
    row.len() == 3 && row[0].is_finite() && row[1].is_finite() && row[1] > 0.0 && row[2].is_finite()
}

fn merge_exact_results_into_scan(
    full_scan: &mut [f64],
    exact_scan: &[f64],
    candidate_rows: &[usize],
) -> Result<(), String> {
    if exact_scan.len() != candidate_rows.len() * 3 {
        return Err(format!(
            "SparseLMM exact merge length mismatch: exact_scan={}, candidate_rows={}",
            exact_scan.len(),
            candidate_rows.len()
        ));
    }
    for (local_idx, &row_idx) in candidate_rows.iter().enumerate() {
        let dst = row_idx
            .checked_mul(3)
            .ok_or_else(|| "SparseLMM exact merge overflow in destination offset".to_string())?;
        let src = local_idx
            .checked_mul(3)
            .ok_or_else(|| "SparseLMM exact merge overflow in source offset".to_string())?;
        if dst + 3 > full_scan.len() {
            return Err(format!(
                "SparseLMM exact merge row out of bounds: row_idx={row_idx}, total_rows={}",
                full_scan.len() / 3
            ));
        }
        let exact_row = &exact_scan[src..src + 3];
        if splmm_exact_row_is_valid(exact_row) {
            full_scan[dst..dst + 3].copy_from_slice(exact_row);
        }
    }
    Ok(())
}

fn scan_to_tsv_with_p0y_and_gamma0_collect_candidates(
    scan_prepared: &JxlmmPreparedInput,
    x_design: &[f64],
    p0y_vec: &[f64],
    xtx_chol: &[f64],
    scan_sample_idx: &[usize],
    chrom: &[String],
    pos: &[i64],
    snp: &[String],
    allele0: &[String],
    allele1: &[String],
    gm: PackedGeneticModel,
    out_tsv: &str,
    gamma0: f64,
    sigma2: f64,
    threads: usize,
    block_rows: usize,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    candidate_p_threshold: f64,
) -> Result<Vec<usize>, String> {
    let stage_timing = splmm_packed_stage_timing_enabled();
    let mut tsv_timing = SplmmTsvTiming::default();
    let score_vec = if sigma2 == 1.0_f64 {
        p0y_vec.to_vec()
    } else {
        p0y_vec.iter().map(|v| *v / sigma2).collect::<Vec<_>>()
    };
    let m = scan_prepared.n_rows();
    if chrom.len() != m
        || pos.len() != m
        || snp.len() != m
        || allele0.len() != m
        || allele1.len() != m
    {
        return Err(format!(
            "SparseLMM TSV metadata length mismatch: rows={m}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        ));
    }
    let writer = AsyncTsvWriter::with_config(
        out_tsv,
        b"chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\n",
        64 * 1024 * 1024,
        4,
    )
    .map_err(|e| e.to_string())?;
    let adapter = JxlmmMatrixAdapter {
        inner: scan_prepared,
    };
    let stats = unified_stats_from_jxlmm(scan_prepared);
    let mut input = UnifiedInput {
        matrix: adapter,
        stats,
    };
    let row_maf = input.stats.maf.clone();
    let row_miss = input.stats.miss.clone();
    let mut candidate_rows = Vec::<usize>::new();
    candidate_rows.reserve(m.min(4096));
    let mut text_buf = String::with_capacity(block_rows.max(512).max(1) * 112);
    let mut tsv_sink = |row_start: usize, rows_here: usize, block: &[f64]| {
        for local_idx in 0..rows_here {
            let pwald = block[local_idx * 3 + 2];
            if pwald.is_finite() && pwald <= candidate_p_threshold {
                candidate_rows.push(row_start + local_idx);
            }
        }
        text_buf.clear();
        let t0 = stage_timing.then(Instant::now);
        append_splmm_tsv_block(
            &mut text_buf,
            row_start,
            rows_here,
            block,
            chrom,
            pos,
            snp,
            allele0,
            allele1,
            &row_maf,
            &row_miss,
            gm,
        );
        if let Some(t0) = t0 {
            tsv_timing.format_secs += t0.elapsed().as_secs_f64();
        }
        let payload = std::mem::take(&mut text_buf).into_bytes();
        let payload_len = payload.len();
        let t0 = stage_timing.then(Instant::now);
        writer.send(payload)?;
        if let Some(t0) = t0 {
            tsv_timing.send_secs += t0.elapsed().as_secs_f64();
        }
        tsv_timing.blocks = tsv_timing.blocks.saturating_add(1);
        tsv_timing.bytes = tsv_timing.bytes.saturating_add(payload_len);
        Ok(())
    };
    let run_res = grammar_scan_blocks_core(
        &mut input,
        x_design,
        score_vec.as_slice(),
        sigma2,
        xtx_chol,
        scan_sample_idx,
        gm,
        gamma0,
        sigma2,
        threads,
        block_rows,
        scan_progress_callback,
        progress_every,
        progress_stage,
        0,
        0,
        &mut tsv_sink,
    );
    let t0 = stage_timing.then(Instant::now);
    let writer_result = writer.finish();
    if let Some(t0) = t0 {
        tsv_timing.finish_secs += t0.elapsed().as_secs_f64();
    }
    if stage_timing {
        emit_splmm_tsv_timing("two_stage_collect", &tsv_timing, m);
    }
    run_res?;
    writer_result?;
    Ok(candidate_rows)
}

#[allow(clippy::too_many_arguments)]
fn merge_two_stage_tsv_with_exact_rows(
    approx_tsv: &str,
    out_tsv: &str,
    scan_prepared: &JxlmmPreparedInput,
    chrom: &[String],
    pos: &[i64],
    snp: &[String],
    allele0: &[String],
    allele1: &[String],
    exact_results: &[f64],
    candidate_rows: &[usize],
    gm: PackedGeneticModel,
) -> Result<usize, String> {
    let m = scan_prepared.n_rows();
    if exact_results.len() != candidate_rows.len() * 3 {
        return Err(format!(
            "SparseLMM exact TSV merge length mismatch: exact_results={}, candidate_rows={}",
            exact_results.len(),
            candidate_rows.len()
        ));
    }
    let approx_file =
        File::open(approx_tsv).map_err(|e| format!("open approx TSV {approx_tsv}: {e}"))?;
    let mut reader = BufReader::new(approx_file);
    let out_file =
        File::create(out_tsv).map_err(|e| format!("create merged TSV {out_tsv}: {e}"))?;
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, out_file);

    let mut line = String::new();
    if reader
        .read_line(&mut line)
        .map_err(|e| format!("read approx TSV header {approx_tsv}: {e}"))?
        == 0
    {
        return Err(format!("approx TSV is empty: {approx_tsv}"));
    }
    writer
        .write_all(line.as_bytes())
        .map_err(|e| format!("write merged TSV header {out_tsv}: {e}"))?;
    line.clear();

    let mut row_idx = 0usize;
    let mut candidate_ptr = 0usize;
    let mut exact_buf = String::with_capacity(256);
    loop {
        let n_read = reader
            .read_line(&mut line)
            .map_err(|e| format!("read approx TSV body {approx_tsv}: {e}"))?;
        if n_read == 0 {
            break;
        }
        if candidate_ptr < candidate_rows.len() && candidate_rows[candidate_ptr] == row_idx {
            let exact_row = &exact_results[candidate_ptr * 3..candidate_ptr * 3 + 3];
            if splmm_exact_row_is_valid(exact_row) {
                exact_buf.clear();
                append_splmm_tsv_block(
                    &mut exact_buf,
                    row_idx,
                    1,
                    exact_row,
                    chrom,
                    pos,
                    snp,
                    allele0,
                    allele1,
                    &scan_prepared.row_maf,
                    &scan_prepared.row_missing,
                    gm,
                );
                writer
                    .write_all(exact_buf.as_bytes())
                    .map_err(|e| format!("write exact merged row {row_idx} to {out_tsv}: {e}"))?;
            } else {
                writer
                    .write_all(line.as_bytes())
                    .map_err(|e| format!("write approx merged row {row_idx} to {out_tsv}: {e}"))?;
            }
            candidate_ptr += 1;
        } else {
            writer
                .write_all(line.as_bytes())
                .map_err(|e| format!("write approx merged row {row_idx} to {out_tsv}: {e}"))?;
        }
        line.clear();
        row_idx += 1;
    }
    if row_idx != m {
        return Err(format!(
            "SparseLMM merged TSV row count mismatch: merged={}, expected={m}",
            row_idx
        ));
    }
    if candidate_ptr != candidate_rows.len() {
        return Err(format!(
            "SparseLMM merged TSV candidate count mismatch: merged={}, expected={}",
            candidate_ptr,
            candidate_rows.len()
        ));
    }
    writer
        .flush()
        .map_err(|e| format!("flush merged TSV {out_tsv}: {e}"))?;
    Ok(m)
}

struct SplmmPreparedScanState {
    factor: SparseJxgrmCholesky,
    solve_workspace: SparseJxgrmSolveWorkspace,
    x_design_col_major: Vec<f64>,
    x_design_xtx_chol: Option<Vec<f64>>,
    null_model: PcgJxlmmNullModel,
    null_info: PcgJxlmmNullModelInfo,
    gamma0: f64,
    r_hat: f64,
    rhat_info: PcgJxlmmRHatResult,
}

impl SplmmPreparedScanState {
    #[inline]
    fn require_x_design_xtx_chol(&self) -> Result<&[f64], String> {
        self.x_design_xtx_chol
            .as_deref()
            .ok_or_else(|| "SparseLMM internal error: scan state is missing XtX chol".to_string())
    }
}

struct SplmmBuiltNullState {
    x_design_col_major: Vec<f64>,
    py: Vec<f64>,
    beta_hat: Vec<f64>,
    null_model: PcgJxlmmNullModel,
    null_info: PcgJxlmmNullModelInfo,
}

fn build_sparse_jxlmm_null_state(
    factor: &SparseJxgrmCholesky,
    x_design: &[f64],
    y_vec: &[f64],
    sigma2: f64,
    solve_workspace: &mut SparseJxgrmSolveWorkspace,
    stage1_cb: Option<&Py<PyAny>>,
) -> Result<SplmmBuiltNullState, String> {
    let n = y_vec.len();
    if n == 0 {
        return Err("SparseLMM null-state build requires non-empty y".to_string());
    }
    if x_design.len() % n != 0 {
        return Err(format!(
            "SparseLMM null-state design length mismatch: len(x_design)={} is not divisible by n={n}",
            x_design.len()
        ));
    }
    let p = x_design.len() / n;
    if p == 0 {
        return Err("SparseLMM null-state build requires at least one design column".to_string());
    }
    if !(sigma2.is_finite() && sigma2 > 0.0_f64) {
        return Err(format!(
            "SparseLMM null-state build requires finite positive sigma2, got {sigma2}"
        ));
    }

    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 6, 0, 1)?;
    }
    let y_vinv_col = sparse_solve_rhs_with_workspace(factor, y_vec, 1, solve_workspace)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 6, 1, 1)?;
    }

    let x_design_col_major = row_major_to_col_major_f64(&x_design, n, p)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 7, 0, p.max(1))?;
    }
    let x_vinv_col =
        sparse_solve_rhs_with_workspace(factor, &x_design_col_major, p, solve_workspace)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 7, p.max(1), p.max(1))?;
    }

    let mut xt_v_inv_y = vec![0.0_f64; p];
    xt_vec_row_major(&x_design, n, p, &y_vinv_col, &mut xt_v_inv_y);

    let mut xt_v_inv_x = vec![0.0_f64; p * p];
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            p as CblasInt,
            p as CblasInt,
            n as CblasInt,
            1.0_f64,
            x_design_col_major.as_ptr(),
            n as CblasInt,
            x_vinv_col.as_ptr(),
            n as CblasInt,
            0.0_f64,
            xt_v_inv_x.as_mut_ptr(),
            p as CblasInt,
        );
    }
    let xt_v_inv_x_chol = spd_cholesky_with_jitter(&xt_v_inv_x, p, "SparseLMM XtVinvX")?;
    drop(xt_v_inv_x);

    let mut beta_hat = vec![0.0_f64; p];
    cholesky_solve_into(&xt_v_inv_x_chol, p, &xt_v_inv_y, &mut beta_hat);
    drop(xt_v_inv_y);

    let mut py = y_vinv_col;
    for (cov_idx, beta) in beta_hat.iter().copied().enumerate() {
        if beta == 0.0_f64 {
            continue;
        }
        let vinvx_col = &x_vinv_col[cov_idx * n..(cov_idx + 1) * n];
        for i in 0..n {
            py[i] -= vinvx_col[i] * beta;
        }
    }
    drop(x_vinv_col);

    let mut p0y = py.clone();
    for value in p0y.iter_mut() {
        *value *= sigma2;
    }
    let mut xt_p0_x_chol = xt_v_inv_x_chol.clone();
    let sqrt_sigma2 = sigma2.sqrt();
    if sqrt_sigma2 != 1.0_f64 {
        for value in xt_p0_x_chol.iter_mut() {
            *value *= sqrt_sigma2;
        }
    }

    let null_model = PcgJxlmmNullModel {
        n_samples: n,
        n_covariates: p,
        sigma2,
        p0y,
        xt_p0_x_chol,
    };
    let null_info = PcgJxlmmNullModelInfo {
        v_inv_y: crate::pcg::PcgSolveInfo {
            converged: true,
            iters: 1,
            rel_res: 0.0,
        },
        v_inv_x: crate::pcg::PcgMatrixSolveInfo {
            n_rows: n,
            n_cols: p,
            converged_all: true,
            max_iters: 1,
            max_rel_res: 0.0,
            column_info: vec![
                crate::pcg::PcgSolveInfo {
                    converged: true,
                    iters: 1,
                    rel_res: 0.0,
                };
                p
            ],
        },
    };
    Ok(SplmmBuiltNullState {
        x_design_col_major,
        py,
        beta_hat,
        null_model,
        null_info,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_splmm_scan_state(
    factor: SparseJxgrmCholesky,
    scan_prepared: &JxlmmPreparedInput,
    x_design: &[f64],
    y_vec: &[f64],
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    sigma_g2: f64,
    sigma_e2: f64,
    threads: usize,
    block_rows: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    stage1_cb: Option<&Py<PyAny>>,
    progress_every: usize,
    scan_mode: SplmmScanMode,
) -> Result<SplmmPreparedScanState, String> {
    let n = y_vec.len();
    let p = x_design.len() / n;
    let sigma2 = splmm_effective_sigma2(sigma_g2, sigma_e2)?;
    if scan_mode.needs_rhat() && rhat_markers == 0 {
        return Err("SparseLMM approximate scan modes require rhat_markers > 0".to_string());
    }
    let n_rhat_cap = if scan_mode.needs_rhat() {
        rhat_markers.min(scan_prepared.n_rows()).max(1)
    } else {
        1usize
    };
    let rhat_progress_total = if scan_mode.needs_rhat() {
        n_rhat_progress_total(scan_prepared.n_rows(), rhat_markers)
    } else {
        1usize
    };
    let solve_cap = if scan_mode.needs_exact_workspace() {
        let m = scan_prepared.n_rows();
        let progress_block = if progress_every == 0 {
            block_rows.max(512).max(1)
        } else {
            progress_every.max(1)
        };
        let est_block_rows = adaptive_exact_block_rows(
            adaptive_grm_block_rows(progress_block, m, n, 0usize, threads).max(1),
            n,
        );
        est_block_rows.max(p).max(n_rhat_cap).max(1)
    } else {
        p.max(n_rhat_cap).max(1)
    };
    let mut solve_workspace = factor.make_solve_workspace(solve_cap)?;

    let x_design_xtx_chol = if scan_mode.needs_rhat() || scan_mode.needs_exact_workspace() {
        Some(xtx_chol_from_design(x_design, n, p)?)
    } else {
        None
    };
    let null_state = build_sparse_jxlmm_null_state(
        &factor,
        x_design,
        y_vec,
        sigma2,
        &mut solve_workspace,
        stage1_cb,
    )?;
    let x_design_col_major = null_state.x_design_col_major;
    let null_model = null_state.null_model;
    let null_info = null_state.null_info;

    let (gamma0, r_hat, rhat_result) = if !scan_mode.needs_rhat() {
        (
            f64::NAN,
            f64::NAN,
            PcgJxlmmRHatResult {
                n_markers_requested: 0,
                n_markers_used: 0,
                solve_info: crate::pcg::PcgMatrixSolveInfo {
                    n_rows: n,
                    n_cols: 0,
                    converged_all: true,
                    max_iters: 1,
                    max_rel_res: 0.0,
                    column_info: vec![],
                },
            },
        )
    } else {
        let rhat_rows = choose_rhat_rows(scan_prepared.n_rows(), rhat_markers, rhat_seed);
        let n_rhat = rhat_rows.len();
        if stage1_cb.is_some() {
            emit_progress_callback(stage1_cb, 8, 0, rhat_progress_total)?;
        }
        let sample_identity = scan_sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
        let sample_byte_idx: Option<Vec<usize>> = if sample_identity {
            None
        } else {
            Some(scan_sample_idx.iter().map(|&sid| sid >> 2).collect())
        };
        let sample_bit_shift: Option<Vec<u8>> = if sample_identity {
            None
        } else {
            Some(
                scan_sample_idx
                    .iter()
                    .map(|&sid| ((sid & 3) << 1) as u8)
                    .collect(),
            )
        };
        let x_design_xtx_chol = x_design_xtx_chol
            .as_deref()
            .ok_or_else(|| "SparseLMM approximate scan modes require XtX chol".to_string())?;
        let mut sampled_markers = vec![0.0_f64; n * n_rhat];
        let mut tmp_snp = vec![0.0_f64; n];
        let mut xts = vec![0.0_f64; p];
        let mut alpha = vec![0.0_f64; p];
        for (col, &row_idx) in rhat_rows.iter().enumerate() {
            decode_packed_row_model_into_f64(
                scan_prepared.row_bytes(row_idx),
                scan_prepared.row_flip[row_idx],
                scan_prepared.row_maf[row_idx],
                n,
                gm,
                sample_identity,
                sample_byte_idx.as_deref(),
                sample_bit_shift.as_deref(),
                &mut tmp_snp,
            );
            for i in 0..n {
                sampled_markers[col * n + i] = tmp_snp[i];
            }
            if stage1_cb.is_some() {
                emit_progress_callback(stage1_cb, 8, col + 1, rhat_progress_total)?;
            }
        }
        let v_inv_s_col = sparse_solve_rhs_with_workspace(
            &factor,
            &sampled_markers,
            n_rhat,
            &mut solve_workspace,
        )?;
        let mut ratio_sum = 0.0_f64;
        let mut n_used = 0usize;
        let mut v_inv_col = vec![0.0_f64; n];
        for col in 0..n_rhat {
            let snp_col = &sampled_markers[col * n..(col + 1) * n];
            for row in 0..n {
                v_inv_col[row] = v_inv_s_col[col * n + row];
            }
            xt_vec_row_major(x_design, n, p, snp_col, &mut xts);
            let s_sq = snp_col.iter().map(|v| v * v).sum::<f64>();
            if !s_sq.is_finite() || s_sq <= SPLMM_TINY {
                continue;
            }
            let s_m_s =
                residualized_sumsq_from_xtx_chol(x_design_xtx_chol, p, &xts, &mut alpha, s_sq);
            if !s_m_s.is_finite() || s_m_s <= SPLMM_TINY {
                continue;
            }
            let s_p0_s =
                crate::pcg::pcg_jxlmm_s_p0_s_exact(&null_model, &x_design, snp_col, &v_inv_col)?;
            if !s_p0_s.is_finite() || s_p0_s <= SPLMM_TINY {
                continue;
            }
            let ratio = s_p0_s / s_m_s;
            if ratio.is_finite() && ratio > 0.0 {
                ratio_sum += ratio;
                n_used += 1;
            }
        }
        if n_used == 0 {
            return Err("SparseLMM r-hat estimation found no valid sampled markers".to_string());
        }
        if stage1_cb.is_some() {
            emit_progress_callback(stage1_cb, 8, rhat_progress_total, rhat_progress_total)?;
        }
        let gamma0_val = ratio_sum / (n_used as f64);
        let r_hat_val = gamma0_val / sigma2;
        drop(v_inv_col);
        drop(v_inv_s_col);
        drop(sampled_markers);
        drop(tmp_snp);
        (
            gamma0_val,
            r_hat_val,
            PcgJxlmmRHatResult {
                n_markers_requested: n_rhat,
                n_markers_used: n_used,
                solve_info: crate::pcg::PcgMatrixSolveInfo {
                    n_rows: n,
                    n_cols: n_rhat,
                    converged_all: true,
                    max_iters: 1,
                    max_rel_res: 0.0,
                    column_info: vec![
                        crate::pcg::PcgSolveInfo {
                            converged: true,
                            iters: 1,
                            rel_res: 0.0
                        };
                        n_rhat
                    ],
                },
            },
        )
    };
    Ok(SplmmPreparedScanState {
        factor,
        solve_workspace,
        x_design_col_major,
        x_design_xtx_chol,
        null_model,
        null_info,
        gamma0,
        r_hat,
        rhat_info: rhat_result,
    })
}

#[allow(clippy::too_many_arguments)]
fn estimate_rhat_and_scan_sparse(
    factor: SparseJxgrmCholesky,
    scan_prepared: JxlmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    scan_sample_idx: Vec<usize>,
    gm: PackedGeneticModel,
    sigma_g2: f64,
    sigma_e2: f64,
    threads: usize,
    block_rows: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    scan_mode: SplmmScanMode,
) -> Result<(f64, Vec<f64>, PcgJxlmmNullModelInfo, PcgJxlmmRHatResult), String> {
    let stage1_cb = stage1_progress_callback.as_ref();
    let mut state = prepare_splmm_scan_state(
        factor,
        &scan_prepared,
        &x_design,
        &y_vec,
        &scan_sample_idx,
        gm,
        sigma_g2,
        sigma_e2,
        threads,
        block_rows,
        rhat_markers,
        rhat_seed,
        stage1_cb,
        progress_every,
        scan_mode,
    )?;
    let total_rows = scan_prepared.n_rows();
    let x_design_xtx_chol = state.require_x_design_xtx_chol()?.to_vec();
    let out = match scan_mode {
        SplmmScanMode::Approx => scan_with_p0y_and_gamma0(
            &scan_prepared,
            &x_design,
            &state.null_model.p0y,
            x_design_xtx_chol.as_slice(),
            &scan_sample_idx,
            gm,
            state.gamma0,
            state.null_model.sigma2,
            threads,
            block_rows,
            scan_progress_callback.as_ref(),
            progress_every,
            9,
            0,
            0,
        )?,
        SplmmScanMode::Exact => scan_with_p0y_and_exact_p0_sparse(
            &state.factor,
            &scan_prepared,
            &x_design,
            &state.x_design_col_major,
            x_design_xtx_chol.as_slice(),
            &state.null_model.p0y,
            &state.null_model.xt_p0_x_chol,
            state.null_model.sigma2,
            &scan_sample_idx,
            gm,
            threads,
            block_rows,
            scan_progress_callback.as_ref(),
            progress_every,
            9,
            0,
            0,
            &mut state.solve_workspace,
        )?,
        SplmmScanMode::TwoStage => {
            let mut approx_out = scan_with_p0y_and_gamma0(
                &scan_prepared,
                &x_design,
                &state.null_model.p0y,
                x_design_xtx_chol.as_slice(),
                &scan_sample_idx,
                gm,
                state.gamma0,
                state.null_model.sigma2,
                threads,
                block_rows,
                scan_progress_callback.as_ref(),
                progress_every,
                9,
                0,
                total_rows,
            )?;
            let p_threshold = splmm_two_stage_p_threshold(total_rows);
            let candidate_rows = select_two_stage_candidates(&approx_out, p_threshold);
            if !candidate_rows.is_empty() {
                let subset = scan_prepared.subset_rows(&candidate_rows)?;
                let exact_out = scan_with_p0y_and_exact_p0_sparse(
                    &state.factor,
                    &subset,
                    &x_design,
                    &state.x_design_col_major,
                    x_design_xtx_chol.as_slice(),
                    &state.null_model.p0y,
                    &state.null_model.xt_p0_x_chol,
                    state.null_model.sigma2,
                    &scan_sample_idx,
                    gm,
                    threads,
                    block_rows,
                    scan_progress_callback.as_ref(),
                    progress_every,
                    9,
                    total_rows,
                    total_rows + candidate_rows.len(),
                    &mut state.solve_workspace,
                )?;
                merge_exact_results_into_scan(&mut approx_out, &exact_out, &candidate_rows)?;
            }
            approx_out
        }
    };
    Ok((state.r_hat, out, state.null_info, state.rhat_info))
}

fn estimate_rhat_and_scan(
    operator_prepared: JxlmmPreparedInput,
    scan_prepared: JxlmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    operator_sample_idx: Vec<usize>,
    scan_sample_idx: Vec<usize>,
    sparse_factor_sample_idx: Option<Vec<usize>>,
    gm: PackedGeneticModel,
    sigma_g2: f64,
    sigma_e2: f64,
    block_rows: usize,
    threads: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    sparse_jxgrm_path: Option<String>,
    scan_mode: SplmmScanMode,
) -> Result<(f64, Vec<f64>, PcgJxlmmNullModelInfo, PcgJxlmmRHatResult), String> {
    let stage1_cb = stage1_progress_callback.as_ref();
    let prefix = operator_prepared
        .bed_prefix
        .as_deref()
        .ok_or_else(|| "SparseLMM requires operator input with a PLINK BED prefix.".to_string())?;
    let sparse_factor_sample_idx_ref = sparse_factor_sample_idx
        .as_deref()
        .unwrap_or(operator_sample_idx.as_slice());
    let sparse_expected_n = if sparse_jxgrm_path.is_some() {
        sparse_factor_sample_idx_ref.len()
    } else {
        operator_prepared.n_samples_full
    };
    let factor = sparse_splmm_load_factor(
        prefix,
        sparse_jxgrm_path.as_deref(),
        sparse_expected_n,
        sparse_factor_sample_idx_ref,
        sigma_g2,
        sigma_e2,
        stage1_cb,
    )?;
    estimate_rhat_and_scan_sparse(
        factor,
        scan_prepared,
        x_design,
        y_vec,
        scan_sample_idx,
        gm,
        sigma_g2,
        sigma_e2,
        threads,
        block_rows,
        rhat_markers,
        rhat_seed,
        stage1_progress_callback,
        scan_progress_callback,
        progress_every,
        scan_mode,
    )
}

#[allow(clippy::too_many_arguments)]
fn estimate_rhat_and_scan_to_tsv(
    operator_prepared: JxlmmPreparedInput,
    scan_prepared: JxlmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    operator_sample_idx: Vec<usize>,
    scan_sample_idx: Vec<usize>,
    sparse_factor_sample_idx: Option<Vec<usize>>,
    gm: PackedGeneticModel,
    sigma_g2: f64,
    sigma_e2: f64,
    threads: usize,
    block_rows: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: String,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    sparse_jxgrm_path: Option<String>,
    scan_mode: SplmmScanMode,
) -> Result<(f64, usize, PcgJxlmmNullModelInfo, PcgJxlmmRHatResult), String> {
    let stage1_cb = stage1_progress_callback.as_ref();
    let prefix = operator_prepared
        .bed_prefix
        .as_deref()
        .ok_or_else(|| "SparseLMM requires operator input with a PLINK BED prefix.".to_string())?;
    let sparse_factor_sample_idx_ref = sparse_factor_sample_idx
        .as_deref()
        .unwrap_or(operator_sample_idx.as_slice());
    let sparse_expected_n = if sparse_jxgrm_path.is_some() {
        sparse_factor_sample_idx_ref.len()
    } else {
        operator_prepared.n_samples_full
    };
    let factor = sparse_splmm_load_factor(
        prefix,
        sparse_jxgrm_path.as_deref(),
        sparse_expected_n,
        sparse_factor_sample_idx_ref,
        sigma_g2,
        sigma_e2,
        stage1_cb,
    )?;
    let mut state = prepare_splmm_scan_state(
        factor,
        &scan_prepared,
        &x_design,
        &y_vec,
        &scan_sample_idx,
        gm,
        sigma_g2,
        sigma_e2,
        threads,
        block_rows,
        rhat_markers,
        rhat_seed,
        stage1_cb,
        progress_every,
        scan_mode,
    )?;
    let total_rows = scan_prepared.n_rows();
    let x_design_xtx_chol = state.require_x_design_xtx_chol()?.to_vec();
    let written_rows = match scan_mode {
        SplmmScanMode::Approx => scan_to_tsv_with_p0y_and_gamma0(
            &scan_prepared,
            &x_design,
            &state.null_model.p0y,
            x_design_xtx_chol.as_slice(),
            &scan_sample_idx,
            gm,
            state.gamma0,
            state.null_model.sigma2,
            threads,
            block_rows,
            chrom.as_slice(),
            pos.as_slice(),
            snp.as_slice(),
            allele0.as_slice(),
            allele1.as_slice(),
            &out_tsv,
            scan_progress_callback.as_ref(),
            progress_every,
            9,
        )?,
        SplmmScanMode::Exact => scan_to_tsv_with_p0y_and_exact_p0_sparse(
            &state.factor,
            &scan_prepared,
            &x_design,
            &state.x_design_col_major,
            x_design_xtx_chol.as_slice(),
            &state.null_model.p0y,
            &state.null_model.xt_p0_x_chol,
            state.null_model.sigma2,
            &scan_sample_idx,
            gm,
            threads,
            block_rows,
            chrom.as_slice(),
            pos.as_slice(),
            snp.as_slice(),
            allele0.as_slice(),
            allele1.as_slice(),
            &out_tsv,
            scan_progress_callback.as_ref(),
            progress_every,
            9,
            &mut state.solve_workspace,
        )?,
        SplmmScanMode::TwoStage => {
            let approx_tmp_tsv = format!("{out_tsv}.approx.twostage.tmp");
            let run_res = (|| -> Result<usize, String> {
                let p_threshold = splmm_two_stage_p_threshold(total_rows);
                let candidate_rows = scan_to_tsv_with_p0y_and_gamma0_collect_candidates(
                    &scan_prepared,
                    &x_design,
                    &state.null_model.p0y,
                    x_design_xtx_chol.as_slice(),
                    &scan_sample_idx,
                    chrom.as_slice(),
                    pos.as_slice(),
                    snp.as_slice(),
                    allele0.as_slice(),
                    allele1.as_slice(),
                    gm,
                    &approx_tmp_tsv,
                    state.gamma0,
                    state.null_model.sigma2,
                    threads,
                    block_rows,
                    scan_progress_callback.as_ref(),
                    progress_every,
                    9,
                    p_threshold,
                )?;
                if candidate_rows.is_empty() {
                    std::fs::rename(&approx_tmp_tsv, &out_tsv).map_err(|e| {
                        format!(
                            "rename SparseLMM two-stage approx TSV {approx_tmp_tsv} -> {out_tsv}: {e}"
                        )
                    })?;
                    return Ok(total_rows);
                }
                let subset = scan_prepared.subset_rows(&candidate_rows)?;
                let exact_out = scan_with_p0y_and_exact_p0_sparse(
                    &state.factor,
                    &subset,
                    &x_design,
                    &state.x_design_col_major,
                    x_design_xtx_chol.as_slice(),
                    &state.null_model.p0y,
                    &state.null_model.xt_p0_x_chol,
                    state.null_model.sigma2,
                    &scan_sample_idx,
                    gm,
                    threads,
                    block_rows,
                    scan_progress_callback.as_ref(),
                    progress_every,
                    9,
                    total_rows,
                    total_rows + candidate_rows.len(),
                    &mut state.solve_workspace,
                )?;
                let merged_rows = merge_two_stage_tsv_with_exact_rows(
                    &approx_tmp_tsv,
                    &out_tsv,
                    &scan_prepared,
                    chrom.as_slice(),
                    pos.as_slice(),
                    snp.as_slice(),
                    allele0.as_slice(),
                    allele1.as_slice(),
                    &exact_out,
                    &candidate_rows,
                    gm,
                )?;
                std::fs::remove_file(&approx_tmp_tsv).map_err(|e| {
                    format!("remove SparseLMM two-stage temp TSV {approx_tmp_tsv}: {e}")
                })?;
                Ok(merged_rows)
            })();
            if run_res.is_err() {
                let _ = std::fs::remove_file(&approx_tmp_tsv);
            }
            run_res?
        }
    };
    Ok((state.r_hat, written_rows, state.null_info, state.rhat_info))
}

#[pyfunction]
#[pyo3(signature = (
    jxgrm_path,
    sample_indices=None
))]
pub fn splmm_sparse_grm_diag_stats<'py>(
    py: Python<'py>,
    jxgrm_path: String,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(f64, f64, f64, usize, usize)> {
    let sample_idx_raw = if let Some(sidx) = sample_indices {
        Some(sidx.as_slice()?.to_vec())
    } else {
        None
    };

    py.detach(move || {
        let csc = MmapSparseGrmCsc::open(&jxgrm_path)?;
        let sample_idx_vec = if let Some(raw) = sample_idx_raw.as_deref() {
            Some(
                parse_index_vec_i64(raw, csc.n_samples(), "sample_indices")
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };

        if let Some(sample_idx) = sample_idx_vec.as_deref() {
            let is_identity_subset = sample_idx.len() == csc.n_samples()
                && sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
            if is_identity_subset {
                let (mean_diag, min_diag, max_diag) = sparse_diag_stats(&csc, 1.0_f64, 0.0_f64)?;
                Ok((mean_diag, min_diag, max_diag, csc.n_samples(), csc.nnz()))
            } else {
                let subset = subset_sparse_grm_csc(&csc, sample_idx)?;
                let (mean_diag, min_diag, max_diag) = sparse_diag_stats(&subset, 1.0_f64, 0.0_f64)?;
                Ok((mean_diag, min_diag, max_diag, subset.n_samples, subset.nnz))
            }
        } else {
            let (mean_diag, min_diag, max_diag) = sparse_diag_stats(&csc, 1.0_f64, 0.0_f64)?;
            Ok((mean_diag, min_diag, max_diag, csc.n_samples(), csc.nnz()))
        }
    })
    .map_err(|e: String| PyRuntimeError::new_err(e))
}

#[pyfunction]
#[pyo3(signature = (
    jxgrm_path,
    y,
    sigma_g2,
    sigma_e2,
    x_cov=None,
    sample_indices=None
))]
pub fn splmm_sparse_null_model_debug<'py>(
    py: Python<'py>,
    jxgrm_path: String,
    y: PyReadonlyArray1<'py, f64>,
    sigma_g2: f64,
    sigma_e2: f64,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let y_vec = y.as_slice()?.to_vec();
    let n = y_vec.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("y must not be empty"));
    }
    let sparse_n = sparse_jxgrm_header_n_samples(&jxgrm_path).map_err(PyRuntimeError::new_err)?;
    let sample_idx = if let Some(raw) = sample_indices.as_ref() {
        parse_index_vec_i64(raw.as_slice()?, sparse_n, "sample_indices")
            .map_err(PyRuntimeError::new_err)?
    } else {
        if n != sparse_n {
            return Err(PyRuntimeError::new_err(format!(
                "SparseLMM null debug requires y length to match sparse n when sample_indices is omitted: y={n}, sparse_n={sparse_n}"
            )));
        }
        (0..sparse_n).collect::<Vec<_>>()
    };
    if sample_idx.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "SparseLMM null debug sample/y mismatch: sample_indices={}, y={n}",
            sample_idx.len()
        )));
    }

    let (x_cov_owned, p_cov) = if let Some(x_cov) = &x_cov {
        let x_arr = x_cov.as_array();
        if x_arr.ndim() != 2 {
            return Err(PyRuntimeError::new_err("x_cov must be 2D (n, p_cov)"));
        }
        let (xn, xp) = (x_arr.shape()[0], x_arr.shape()[1]);
        if xn != n {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov row count mismatch: got {xn}, expected {n}"
            )));
        }
        let owned = match x_cov.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => x_arr.iter().copied().collect(),
        };
        (Some(owned), xp)
    } else {
        (None, 0usize)
    };
    let x_design = build_design_with_intercept(x_cov_owned.as_deref(), n, p_cov);
    let p = p_cov + 1;
    if n <= p {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p for SparseLMM null debug: n={n}, p={p}"
        )));
    }

    let sigma2 = splmm_effective_sigma2(sigma_g2, sigma_e2).map_err(PyRuntimeError::new_err)?;
    let sample_idx_for_factor = sample_idx.clone();
    let x_design_for_factor = x_design.clone();
    let y_vec_for_factor = y_vec.clone();
    let built = py
        .detach(move || {
            let factor = sparse_splmm_load_factor(
                "",
                Some(&jxgrm_path),
                n,
                sample_idx_for_factor.as_slice(),
                sigma_g2,
                sigma_e2,
                None,
            )?;
            let mut solve_workspace = factor.make_solve_workspace(p.max(1))?;
            build_sparse_jxlmm_null_state(
                &factor,
                x_design_for_factor.as_slice(),
                y_vec_for_factor.as_slice(),
                sigma2,
                &mut solve_workspace,
                None,
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok((
        built.null_model.sigma2,
        PyArray1::from_owned_array(py, numpy::ndarray::Array1::from_vec(built.py)),
        PyArray1::from_owned_array(py, numpy::ndarray::Array1::from_vec(built.null_model.p0y)),
        PyArray1::from_owned_array(py, numpy::ndarray::Array1::from_vec(built.beta_hat)),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    sigma_g2,
    sigma_e2,
    x_cov=None,
    sample_indices=None,
    operator_sample_indices=None,
    site_keep=None,
    tol=1e-5,
    max_iter=200,
    block_rows=0,
    std_eps=1e-12,
    use_train_maf=true,
    threads=0,
    model="add",
    rhat_markers=SPLMM_DEFAULT_RHAT_MARKERS,
    rhat_seed=SPLMM_DEFAULT_RHAT_SEED,
    packed=None,
    packed_n_samples=0,
    maf=None,
    row_flip=None,
    row_missing=None,
    row_indices=None,
    sparse_sample_indices=None,
    sparse_jxgrm_path=None,
    stage1_progress_callback=None,
    scan_progress_callback=None,
    progress_every=0,
    rhat_tol=1e-3,
    scan_mode="two_stage"
))]
pub fn splmm_assoc_pcg_bed<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    sigma_g2: f64,
    sigma_e2: f64,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    operator_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    tol: f64,
    max_iter: usize,
    block_rows: usize,
    std_eps: f64,
    use_train_maf: bool,
    threads: usize,
    model: &str,
    rhat_markers: usize,
    rhat_seed: u64,
    packed: Option<PyReadonlyArray2<'py, u8>>,
    packed_n_samples: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    sparse_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    sparse_jxgrm_path: Option<String>,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    rhat_tol: f64,
    scan_mode: &str,
) -> PyResult<(
    f64,
    bool,
    usize,
    f64,
    bool,
    usize,
    f64,
    usize,
    usize,
    Bound<'py, PyArray2<f64>>,
)> {
    if max_iter == 0 {
        return Err(PyRuntimeError::new_err("max_iter must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be finite and > 0"));
    }
    if !(std_eps.is_finite() && std_eps > 0.0) {
        return Err(PyRuntimeError::new_err("std_eps must be finite and > 0"));
    }
    let scan_mode = SplmmScanMode::parse(scan_mode)?;
    if scan_mode.needs_rhat() && rhat_markers == 0 {
        return Err(PyRuntimeError::new_err("rhat_markers must be > 0"));
    }
    let _ = (tol, max_iter, std_eps, use_train_maf, rhat_tol);
    emit_progress_callback(stage1_progress_callback.as_ref(), 0, 0, 1)
        .map_err(PyRuntimeError::new_err)?;
    let prepared = prepare_splmm_assoc_inputs(
        &prefix,
        y,
        x_cov,
        sample_indices,
        operator_sample_indices,
        site_keep,
        packed,
        packed_n_samples,
        maf,
        row_flip,
        row_missing,
        row_indices,
        model,
    )?;
    let sparse_factor_sample_idx = parse_optional_index_array(
        sparse_sample_indices.as_ref(),
        prepared.operator_prepared.n_samples_full,
        "sparse_sample_indices",
    )?;
    emit_progress_callback(stage1_progress_callback.as_ref(), 0, 1, 1)
        .map_err(PyRuntimeError::new_err)?;

    let (r_hat, out, null_info, rhat_info) = py
        .detach(move || {
            estimate_rhat_and_scan(
                prepared.operator_prepared,
                prepared.scan_prepared,
                prepared.x_design,
                prepared.y_vec,
                prepared.operator_sample_idx,
                prepared.scan_sample_idx,
                sparse_factor_sample_idx,
                prepared.gm,
                sigma_g2,
                sigma_e2,
                block_rows,
                threads,
                rhat_markers,
                rhat_seed,
                stage1_progress_callback,
                scan_progress_callback,
                progress_every,
                sparse_jxgrm_path,
                scan_mode,
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    let m = out.len() / 3;
    let out_arr = numpy::ndarray::Array2::from_shape_vec((m, 3), out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        r_hat,
        null_info.v_inv_y.converged,
        null_info.v_inv_y.iters,
        null_info.v_inv_y.rel_res,
        null_info.v_inv_x.converged_all,
        null_info.v_inv_x.max_iters,
        null_info.v_inv_x.max_rel_res,
        rhat_info.n_markers_requested,
        rhat_info.n_markers_used,
        PyArray2::from_owned_array(py, out_arr),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    sigma_g2,
    sigma_e2,
    chrom,
    pos,
    snp,
    allele0,
    allele1,
    out_tsv,
    x_cov=None,
    sample_indices=None,
    operator_sample_indices=None,
    site_keep=None,
    tol=1e-5,
    max_iter=200,
    block_rows=0,
    std_eps=1e-12,
    use_train_maf=true,
    threads=0,
    model="add",
    rhat_markers=SPLMM_DEFAULT_RHAT_MARKERS,
    rhat_seed=SPLMM_DEFAULT_RHAT_SEED,
    packed=None,
    packed_n_samples=0,
    maf=None,
    row_flip=None,
    row_missing=None,
    row_indices=None,
    sparse_sample_indices=None,
    sparse_jxgrm_path=None,
    stage1_progress_callback=None,
    scan_progress_callback=None,
    progress_every=0,
    rhat_tol=1e-3,
    scan_mode="two_stage"
))]
pub fn splmm_assoc_pcg_bed_to_tsv<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    sigma_g2: f64,
    sigma_e2: f64,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: String,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    operator_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    tol: f64,
    max_iter: usize,
    block_rows: usize,
    std_eps: f64,
    use_train_maf: bool,
    threads: usize,
    model: &str,
    rhat_markers: usize,
    rhat_seed: u64,
    packed: Option<PyReadonlyArray2<'py, u8>>,
    packed_n_samples: usize,
    maf: Option<PyReadonlyArray1<'py, f32>>,
    row_flip: Option<PyReadonlyArray1<'py, bool>>,
    row_missing: Option<PyReadonlyArray1<'py, f32>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    sparse_sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    sparse_jxgrm_path: Option<String>,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    rhat_tol: f64,
    scan_mode: &str,
) -> PyResult<(f64, bool, usize, f64, bool, usize, f64, usize, usize, usize)> {
    if max_iter == 0 {
        return Err(PyRuntimeError::new_err("max_iter must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be finite and > 0"));
    }
    if !(std_eps.is_finite() && std_eps > 0.0) {
        return Err(PyRuntimeError::new_err("std_eps must be finite and > 0"));
    }
    let scan_mode = SplmmScanMode::parse(scan_mode)?;
    if scan_mode.needs_rhat() && rhat_markers == 0 {
        return Err(PyRuntimeError::new_err("rhat_markers must be > 0"));
    }
    let _ = (tol, max_iter, std_eps, use_train_maf, rhat_tol);

    emit_progress_callback(stage1_progress_callback.as_ref(), 0, 0, 1)
        .map_err(PyRuntimeError::new_err)?;
    let prepared = prepare_splmm_assoc_inputs(
        &prefix,
        y,
        x_cov,
        sample_indices,
        operator_sample_indices,
        site_keep,
        packed,
        packed_n_samples,
        maf,
        row_flip,
        row_missing,
        row_indices,
        model,
    )?;
    let sparse_factor_sample_idx = parse_optional_index_array(
        sparse_sample_indices.as_ref(),
        prepared.operator_prepared.n_samples_full,
        "sparse_sample_indices",
    )?;
    emit_progress_callback(stage1_progress_callback.as_ref(), 0, 1, 1)
        .map_err(PyRuntimeError::new_err)?;

    // Auto-populate chrom/pos/snp/allele from BIM when the caller passes
    // empty arrays (saves Python from building five large lists).
    let (chrom, pos, snp, allele0, allele1): (
        Vec<String>,
        Vec<i64>,
        Vec<String>,
        Vec<String>,
        Vec<String>,
    ) = if chrom.is_empty()
        && pos.is_empty()
        && snp.is_empty()
        && allele0.is_empty()
        && allele1.is_empty()
    {
        let bed_prefix = normalize_plink_prefix_local(&prefix);
        let sites = crate::gfcore::read_bim(&bed_prefix)
            .map_err(|e| PyRuntimeError::new_err(format!("read BIM: {e}")))?;
        let ri = prepared
            .scan_prepared
            .row_source_indices
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("auto-populate requires row_source_indices"))?;
        let m = prepared.scan_prepared.n_rows();
        let mut c = Vec::with_capacity(m);
        let mut p = Vec::with_capacity(m);
        let mut s = Vec::with_capacity(m);
        let mut a0 = Vec::with_capacity(m);
        let mut a1 = Vec::with_capacity(m);
        for &src in ri.iter() {
            let site = &sites[src];
            c.push(site.chrom.clone());
            p.push(site.pos as i64);
            s.push(site.snp.clone());
            a0.push(site.ref_allele.clone());
            a1.push(site.alt_allele.clone());
        }
        (c, p, s, a0, a1)
    } else {
        (chrom, pos, snp, allele0, allele1)
    };
    if chrom.len() != prepared.scan_prepared.n_rows()
        || pos.len() != prepared.scan_prepared.n_rows()
        || snp.len() != prepared.scan_prepared.n_rows()
        || allele0.len() != prepared.scan_prepared.n_rows()
        || allele1.len() != prepared.scan_prepared.n_rows()
    {
        return Err(PyRuntimeError::new_err(format!(
            "SparseLMM TSV metadata length mismatch: rows={}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            prepared.scan_prepared.n_rows(),
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        )));
    }

    let (r_hat, written_rows, null_info, rhat_info) = py
        .detach(move || {
            estimate_rhat_and_scan_to_tsv(
                prepared.operator_prepared,
                prepared.scan_prepared,
                prepared.x_design,
                prepared.y_vec,
                prepared.operator_sample_idx,
                prepared.scan_sample_idx,
                sparse_factor_sample_idx,
                prepared.gm,
                sigma_g2,
                sigma_e2,
                threads,
                block_rows,
                rhat_markers,
                rhat_seed,
                chrom,
                pos,
                snp,
                allele0,
                allele1,
                out_tsv,
                stage1_progress_callback,
                scan_progress_callback,
                progress_every,
                sparse_jxgrm_path,
                scan_mode,
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok((
        r_hat,
        null_info.v_inv_y.converged,
        null_info.v_inv_y.iters,
        null_info.v_inv_y.rel_res,
        null_info.v_inv_x.converged_all,
        null_info.v_inv_x.max_iters,
        null_info.v_inv_x.max_rel_res,
        rhat_info.n_markers_requested,
        rhat_info.n_markers_used,
        written_rows,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    py_vec,
    r_hat,
    packed,
    packed_n_samples,
    maf,
    row_flip,
    x_cov=None,
    sample_indices=None,
    row_indices=None,
    threads=0,
    block_rows=0,
    model="add",
    progress_callback=None,
    progress_every=0
))]
pub fn splmm_scan_grammar_packed<'py>(
    py: Python<'py>,
    py_vec: PyReadonlyArray1<'py, f64>,
    r_hat: f64,
    packed: PyReadonlyArray2<'py, u8>,
    packed_n_samples: usize,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    threads: usize,
    block_rows: usize,
    model: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let gm = PackedGeneticModel::parse(model)?;
    let py_vec_owned = py_vec.as_slice()?.to_vec();
    let n = py_vec_owned.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("py_vec must not be empty"));
    }

    let (x_cov_owned, p_cov) = if let Some(x_cov) = &x_cov {
        let x_arr = x_cov.as_array();
        if x_arr.ndim() != 2 {
            return Err(PyRuntimeError::new_err("x_cov must be 2D (n, p_cov)"));
        }
        let (xn, xp) = (x_arr.shape()[0], x_arr.shape()[1]);
        if xn != n {
            return Err(PyRuntimeError::new_err(format!(
                "x_cov row count mismatch: got {xn}, expected {n}"
            )));
        }
        let owned = match x_cov.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => x_arr.iter().copied().collect(),
        };
        (Some(owned), xp)
    } else {
        (None, 0usize)
    };
    let x_design = build_design_with_intercept(x_cov_owned.as_deref(), n, p_cov);
    let p = p_cov + 1;
    if n <= p {
        return Err(PyRuntimeError::new_err(format!(
            "n must be > p for SparseLMM grammar scan: n={n}, p={p}"
        )));
    }
    let xtx_chol = xtx_chol_from_design(&x_design, n, p).map_err(PyRuntimeError::new_err)?;
    let scan_prepared = prepare_external_packed_input(
        Some(packed),
        packed_n_samples,
        Some(maf),
        Some(row_flip),
        None,
        row_indices,
    )?;
    let scan_sample_idx: Vec<usize> = if let Some(sample_indices) = sample_indices {
        let parsed = parse_index_vec_i64(
            sample_indices.as_slice()?,
            scan_prepared.n_samples_full,
            "sample_indices",
        )?;
        if parsed.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "sample_indices length mismatch: got {}, expected {n}",
                parsed.len()
            )));
        }
        parsed
    } else {
        if n != scan_prepared.n_samples_full {
            return Err(PyRuntimeError::new_err(format!(
                "len(py_vec)={} must equal n_samples={} when sample_indices is not provided",
                n, scan_prepared.n_samples_full
            )));
        }
        (0..n).collect()
    };

    let out = py
        .detach(move || {
            scan_with_py_and_rhat(
                &scan_prepared,
                &x_design,
                &py_vec_owned,
                &xtx_chol,
                &scan_sample_idx,
                gm,
                r_hat,
                threads,
                block_rows,
                progress_callback.as_ref(),
                progress_every,
                2,
                0,
                0,
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    let m = out.len() / 3;
    let out_arr = numpy::ndarray::Array2::from_shape_vec((m, 3), out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, out_arr))
}

/// Deprecated alias for `splmm_scan_grammar_packed`. The old name was misleading —
/// this function has always used the GRAMMAR-gamma denominator (r_hat * g'Mg),
/// not the exact g'Pg denominator.
#[pyfunction]
#[pyo3(signature = (
    py_vec,
    r_hat,
    packed,
    packed_n_samples,
    maf,
    row_flip,
    x_cov=None,
    sample_indices=None,
    row_indices=None,
    threads=0,
    block_rows=0,
    model="add",
    progress_callback=None,
    progress_every=0
))]
#[allow(non_snake_case)]
pub fn splmm_scan_exact_packed<'py>(
    py: Python<'py>,
    py_vec: PyReadonlyArray1<'py, f64>,
    r_hat: f64,
    packed: PyReadonlyArray2<'py, u8>,
    packed_n_samples: usize,
    maf: PyReadonlyArray1<'py, f32>,
    row_flip: PyReadonlyArray1<'py, bool>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    threads: usize,
    block_rows: usize,
    model: &str,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    use pyo3::intern;
    let warnings = py.import(intern!(py, "warnings"))?;
    warnings.call_method1(
        intern!(py, "warn"),
        ("splmm_scan_exact_packed is deprecated; use splmm_scan_grammar_packed instead",),
    )?;
    splmm_scan_grammar_packed(
        py,
        py_vec,
        r_hat,
        packed,
        packed_n_samples,
        maf,
        row_flip,
        x_cov,
        sample_indices,
        row_indices,
        threads,
        block_rows,
        model,
        progress_callback,
        progress_every,
    )
}

// ---------------------------------------------------------------------------
// gload::GenotypeMatrix adapter for JxlmmPreparedInput
// ---------------------------------------------------------------------------
// Zero-copy bridge: lets existing code paths call functions that expect a
// UnifiedInput<G> without cloning the metadata vectors.  New modules should
// construct a UnifiedInput<BedMmapMatrix> or UnifiedInput<PackedBedMatrix>
// directly via gload::open_bed_mmap_unified() and pass that to
// block-decode helpers.  Long-term JxlmmPreparedInput should be retired.

/// Build a GlobalStats from a JxlmmPreparedInput (clones vectors once).
fn unified_stats_from_jxlmm(p: &JxlmmPreparedInput) -> crate::gload::GlobalStats {
    let row_source_indices = p
        .row_source_indices
        .clone()
        .unwrap_or_else(|| (0..p.row_maf.len()).collect());
    crate::gload::GlobalStats {
        maf: p.row_maf.clone(),
        miss: p.row_missing.clone(),
        row_flip: p.row_flip.clone(),
        row_source_indices,
        site_keep: Vec::new(),
        n_samples_full: p.n_samples_full,
        n_markers_total: 0,
        bytes_per_snp: p.bytes_per_snp,
    }
}

/// Wraps a `&JxlmmPreparedInput` as a [`GenotypeMatrix`].
struct JxlmmMatrixAdapter<'a> {
    inner: &'a JxlmmPreparedInput,
}

impl GenotypeMatrix for JxlmmMatrixAdapter<'_> {
    fn n_samples_full(&self) -> usize {
        self.inner.n_samples_full
    }
    fn bytes_per_snp(&self) -> usize {
        self.inner.bytes_per_snp
    }
    fn packed_flat(&self) -> &[u8] {
        self.inner.payload.as_bytes()
    }
    fn source_row_bytes(&self, source_idx: usize) -> &[u8] {
        let offset = source_idx.saturating_mul(self.inner.bytes_per_snp);
        &self.inner.payload.as_bytes()[offset..][..self.inner.bytes_per_snp]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cholesky::sparse_cholesky_analyze_jxgrm_csc;
    use crate::spgrm::SparseGrmCsc;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(
                    a.is_nan() && b.is_nan(),
                    "mismatch at {idx}: lhs={a:?}, rhs={b:?}"
                );
            } else {
                assert!(
                    (a - b).abs() <= tol,
                    "mismatch at {idx}: lhs={a:.12e}, rhs={b:.12e}, diff={:.12e}, tol={tol:.12e}",
                    (a - b).abs()
                );
            }
        }
    }

    fn pack_plink_codes(codes: &[u8]) -> Vec<u8> {
        let mut row = vec![0u8; codes.len().div_ceil(4)];
        for (sample_idx, &code) in codes.iter().enumerate() {
            row[sample_idx >> 2] |= (code & 0b11) << ((sample_idx & 3) * 2);
        }
        row
    }

    fn make_test_scan_prepared() -> (JxlmmPreparedInput, Vec<usize>) {
        let source_rows = [
            pack_plink_codes(&[0b00, 0b10, 0b11, 0b01]),
            pack_plink_codes(&[0b10, 0b10, 0b00, 0b11]),
            pack_plink_codes(&[0b11, 0b00, 0b10, 0b00]),
        ];
        let mut packed_flat = Vec::<u8>::new();
        for row in source_rows.iter() {
            packed_flat.extend_from_slice(row);
        }
        let prepared = JxlmmPreparedInput {
            bed_prefix: None,
            payload: JxlmmPayload::Packed(Arc::<[u8]>::from(packed_flat)),
            n_samples_full: 4,
            bytes_per_snp: 1,
            row_flip: vec![false, false, false],
            row_maf: vec![0.375, 0.5, 0.5],
            row_missing: vec![0.0, 0.25, 0.0],
            row_source_indices: Some(vec![2, 0, 1]),
        };
        let scan_sample_idx = vec![3usize, 1usize, 0usize];
        (prepared, scan_sample_idx)
    }

    fn make_test_design() -> Vec<f64> {
        let x_cov = vec![0.0_f64, 1.0_f64, 2.0_f64];
        build_design_with_intercept(Some(&x_cov), 3, 1)
    }

    fn make_diag_factor(diag: &[f64]) -> SparseJxgrmCholesky {
        let n = diag.len();
        let csc = SparseGrmCsc {
            n_samples: n,
            nnz: n,
            col_ptr: (0..=n).map(|v| v as u64).collect(),
            row_indices: (0..n).map(|v| v as u32).collect(),
            values: diag.to_vec(),
        };
        sparse_cholesky_analyze_jxgrm_csc(&csc)
            .unwrap()
            .factorize_diag_shifted(0.0)
            .unwrap()
    }

    fn sample_mapping(scan_sample_idx: &[usize]) -> (bool, Option<Vec<usize>>, Option<Vec<u8>>) {
        let sample_identity = scan_sample_idx.iter().enumerate().all(|(i, &sid)| sid == i);
        if sample_identity {
            (true, None, None)
        } else {
            (
                false,
                Some(scan_sample_idx.iter().map(|&sid| sid >> 2).collect()),
                Some(
                    scan_sample_idx
                        .iter()
                        .map(|&sid| ((sid & 3) << 1) as u8)
                        .collect(),
                ),
            )
        }
    }

    fn decode_test_row(
        prepared: &JxlmmPreparedInput,
        row_idx: usize,
        scan_sample_idx: &[usize],
        gm: PackedGeneticModel,
    ) -> Vec<f64> {
        let n = scan_sample_idx.len();
        let (sample_identity, sample_byte_idx, sample_bit_shift) = sample_mapping(scan_sample_idx);
        let mut out = vec![0.0_f64; n];
        decode_packed_row_model_into_f64(
            prepared.row_bytes(row_idx),
            prepared.row_flip[row_idx],
            prepared.row_maf[row_idx],
            n,
            gm,
            sample_identity,
            sample_byte_idx.as_deref(),
            sample_bit_shift.as_deref(),
            &mut out,
        );
        out
    }

    fn manual_grammar_scan(
        prepared: &JxlmmPreparedInput,
        x_design: &[f64],
        py_vec: &[f64],
        xtx_chol: &[f64],
        scan_sample_idx: &[usize],
        gm: PackedGeneticModel,
        r_hat: f64,
    ) -> Vec<f64> {
        let n = py_vec.len();
        let p = x_design.len() / n;
        let mut out = vec![0.0_f64; prepared.n_rows() * 3];
        let mut xts = vec![0.0_f64; p];
        let mut alpha = vec![0.0_f64; p];
        for row_idx in 0..prepared.n_rows() {
            let g = decode_test_row(prepared, row_idx, scan_sample_idx, gm);
            xts.fill(0.0);
            let mut spy = 0.0_f64;
            let mut s_sq = 0.0_f64;
            for i in 0..n {
                let gi = g[i];
                spy += gi * py_vec[i];
                s_sq += gi * gi;
                let x_row = &x_design[i * p..(i + 1) * p];
                for j in 0..p {
                    xts[j] += x_row[j] * gi;
                }
            }
            let s_m_s = residualized_sumsq_from_xtx_chol(xtx_chol, p, &xts, &mut alpha, s_sq);
            let denom = r_hat * s_m_s;
            let out_row = &mut out[row_idx * 3..(row_idx + 1) * 3];
            if denom.is_finite() && denom > SPLMM_TINY {
                out_row[0] = spy / denom;
                out_row[1] = 1.0_f64 / denom.sqrt();
                out_row[2] = chi2_sf_df1((spy * spy) / denom);
            } else {
                out_row[0] = f64::NAN;
                out_row[1] = f64::NAN;
                out_row[2] = 1.0_f64;
            }
        }
        out
    }

    fn compute_xt_v_inv_x_chol(
        factor: &SparseJxgrmCholesky,
        x_design: &[f64],
        n: usize,
        p: usize,
    ) -> Vec<f64> {
        let x_col = row_major_to_col_major_f64(x_design, n, p).unwrap();
        let mut workspace = factor.make_solve_workspace(p.max(1)).unwrap();
        let x_v_inv_col =
            sparse_solve_rhs_with_workspace(factor, &x_col, p, &mut workspace).unwrap();
        let mut xt_v_inv_x = vec![0.0_f64; p * p];
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                p as CblasInt,
                p as CblasInt,
                n as CblasInt,
                1.0_f64,
                x_col.as_ptr(),
                n as CblasInt,
                x_v_inv_col.as_ptr(),
                n as CblasInt,
                0.0_f64,
                xt_v_inv_x.as_mut_ptr(),
                p as CblasInt,
            );
        }
        spd_cholesky_with_jitter(&xt_v_inv_x, p, "test XtVinvX").unwrap()
    }

    fn unique_temp_path(stem: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "janusx_splmm_{stem}_{}_{}.tsv",
            std::process::id(),
            stamp
        ))
    }

    fn manual_exact_scan(
        factor: &SparseJxgrmCholesky,
        prepared: &JxlmmPreparedInput,
        x_design: &[f64],
        py_vec: &[f64],
        xt_v_inv_x_chol: &[f64],
        scan_sample_idx: &[usize],
        gm: PackedGeneticModel,
    ) -> Vec<f64> {
        let n = py_vec.len();
        let p = x_design.len() / n;
        let mut out = vec![0.0_f64; prepared.n_rows() * 3];
        let mut c_j = vec![0.0_f64; p];
        let mut alpha = vec![0.0_f64; p];
        for row_idx in 0..prepared.n_rows() {
            let g = decode_test_row(prepared, row_idx, scan_sample_idx, gm);
            let z = factor.solve_vec(&g).unwrap();
            let spy = g.iter().zip(py_vec.iter()).map(|(a, b)| a * b).sum::<f64>();
            let s_v_s = g.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>();
            xt_vec_row_major(x_design, n, p, &z, &mut c_j);
            cholesky_solve_into(xt_v_inv_x_chol, p, &c_j, &mut alpha);
            let x_quad = c_j
                .iter()
                .zip(alpha.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let denom = (s_v_s - x_quad).max(0.0_f64);
            let out_row = &mut out[row_idx * 3..(row_idx + 1) * 3];
            if denom.is_finite() && denom > SPLMM_TINY {
                out_row[0] = spy / denom;
                out_row[1] = 1.0_f64 / denom.sqrt();
                out_row[2] = chi2_sf_df1((spy * spy) / denom);
            } else {
                out_row[0] = f64::NAN;
                out_row[1] = f64::NAN;
                out_row[2] = 1.0_f64;
            }
        }
        out
    }

    #[test]
    fn row_major_snp_block_f32_to_col_major_f64_is_widening_copy() {
        let block = vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32];
        let mut out = vec![0.0_f64; block.len()];
        row_major_snp_block_f32_to_col_major_f64(&block, 2, 3, &mut out);
        assert_eq!(
            out,
            vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64]
        );
    }

    #[test]
    fn xt_mat_rhs_block_matches_naive_reference() {
        let n = 3usize;
        let p = 2usize;
        let rows_here = 2usize;
        let x_design = vec![
            1.0_f64, 2.0_f64, //
            3.0_f64, 4.0_f64, //
            5.0_f64, 6.0_f64,
        ];
        let z_col_major = vec![
            10.0_f64, 11.0_f64, 12.0_f64, //
            20.0_f64, 21.0_f64, 22.0_f64,
        ];
        let mut got = vec![0.0_f64; rows_here * p];
        let x_col = row_major_to_col_major_f64(&x_design, n, p).unwrap();
        xt_mat_rhs_block(
            &x_design,
            Some(&x_col),
            &z_col_major,
            n,
            p,
            rows_here,
            &mut got,
        );
        let expected = vec![
            103.0_f64, 136.0_f64, //
            193.0_f64, 256.0_f64,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn grammar_scan_core_matches_wrapper_and_manual_reference() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let py_vec = vec![0.4_f64, -1.0_f64, 0.8_f64];
        let xtx_chol = xtx_chol_from_design(&x_design, py_vec.len(), 2).unwrap();
        let r_hat = 1.7_f64;

        let wrapper = scan_with_py_and_rhat(
            &prepared,
            &x_design,
            &py_vec,
            &xtx_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            r_hat,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
        )
        .unwrap();

        let adapter = JxlmmMatrixAdapter { inner: &prepared };
        let stats = unified_stats_from_jxlmm(&prepared);
        let mut input = UnifiedInput {
            matrix: adapter,
            stats,
        };
        let mut core = vec![0.0_f64; prepared.n_rows() * 3];
        let mut sink = |row_start: usize, rows_here: usize, block: &[f64]| {
            core[row_start * 3..][..rows_here * 3].copy_from_slice(block);
            Ok(())
        };
        grammar_scan_blocks_core(
            &mut input,
            &x_design,
            &py_vec,
            1.0_f64,
            &xtx_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            r_hat,
            1.0_f64,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
            &mut sink,
        )
        .unwrap();

        let manual = manual_grammar_scan(
            &prepared,
            &x_design,
            &py_vec,
            &xtx_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            r_hat,
        );

        assert_close(&core, &wrapper, 1e-8_f64);
        assert_close(&core, &manual, 1e-5_f64);
    }

    #[test]
    fn exact_scan_core_matches_manual_reference() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let py_vec = vec![0.3_f64, -0.8_f64, 0.6_f64];
        let factor = make_diag_factor(&[2.0_f64, 3.0_f64, 4.0_f64]);
        let xt_v_inv_x_chol = compute_xt_v_inv_x_chol(&factor, &x_design, py_vec.len(), 2);
        let xtx_chol = xtx_chol_from_design(&x_design, py_vec.len(), 2).unwrap();
        let x_col = row_major_to_col_major_f64(&x_design, py_vec.len(), 2).unwrap();

        let mut workspace_core = factor.make_solve_workspace(2).unwrap();
        let mut core = vec![0.0_f64; prepared.n_rows() * 3];
        let mut sink = |row_start: usize, rows_here: usize, block: &[f64]| {
            core[row_start * 3..][..rows_here * 3].copy_from_slice(block);
            Ok(())
        };
        let adapter = JxlmmMatrixAdapter { inner: &prepared };
        let stats = unified_stats_from_jxlmm(&prepared);
        let mut input = UnifiedInput {
            matrix: adapter,
            stats,
        };
        exact_scan_blocks_core(
            &factor,
            &mut input,
            &x_design,
            &x_col,
            &xtx_chol,
            &py_vec,
            1.0_f64,
            &xt_v_inv_x_chol,
            1.0_f64,
            1.0_f64,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
            &mut workspace_core,
            &mut sink,
        )
        .unwrap();

        let manual = manual_exact_scan(
            &factor,
            &prepared,
            &x_design,
            &py_vec,
            &xt_v_inv_x_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
        );

        assert_close(&core, &manual, 1e-5_f64);
    }

    #[test]
    fn p0_grammar_scan_matches_full_p_scan() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let py_vec = vec![0.3_f64, -0.8_f64, 0.6_f64];
        let xtx_chol = xtx_chol_from_design(&x_design, py_vec.len(), 2).unwrap();
        let r_hat = 0.75_f64;
        let sigma2 = 3.5_f64;
        let gamma0 = sigma2 * r_hat;
        let p0y_vec: Vec<f64> = py_vec.iter().map(|v| v * sigma2).collect();

        let full = scan_with_py_and_rhat(
            &prepared,
            &x_design,
            &py_vec,
            &xtx_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            r_hat,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
        )
        .unwrap();
        let p0 = scan_with_p0y_and_gamma0(
            &prepared,
            &x_design,
            &p0y_vec,
            &xtx_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            gamma0,
            sigma2,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
        )
        .unwrap();

        assert_close(&full, &p0, 1e-10_f64);
    }

    #[test]
    fn p0_exact_scan_matches_full_p_scan() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let py_vec = vec![0.3_f64, -0.8_f64, 0.6_f64];
        let factor = make_diag_factor(&[2.0_f64, 3.0_f64, 4.0_f64]);
        let xt_v_inv_x_chol = compute_xt_v_inv_x_chol(&factor, &x_design, py_vec.len(), 2);
        let xtx_chol = xtx_chol_from_design(&x_design, py_vec.len(), 2).unwrap();
        let x_col = row_major_to_col_major_f64(&x_design, py_vec.len(), 2).unwrap();
        let sigma2 = 2.75_f64;
        let p0y_vec: Vec<f64> = py_vec.iter().map(|v| v * sigma2).collect();
        let mut xt_p0_x_chol = xt_v_inv_x_chol.clone();
        let sqrt_sigma2 = sigma2.sqrt();
        for value in xt_p0_x_chol.iter_mut() {
            *value *= sqrt_sigma2;
        }

        let mut workspace_full = factor.make_solve_workspace(2).unwrap();
        let mut full = vec![0.0_f64; prepared.n_rows() * 3];
        let mut full_sink = |row_start: usize, rows_here: usize, block: &[f64]| {
            full[row_start * 3..][..rows_here * 3].copy_from_slice(block);
            Ok(())
        };
        let adapter = JxlmmMatrixAdapter { inner: &prepared };
        let stats = unified_stats_from_jxlmm(&prepared);
        let mut input = UnifiedInput {
            matrix: adapter,
            stats,
        };
        exact_scan_blocks_core(
            &factor,
            &mut input,
            &x_design,
            &x_col,
            &xtx_chol,
            &py_vec,
            1.0_f64,
            &xt_v_inv_x_chol,
            1.0_f64,
            1.0_f64,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
            &mut workspace_full,
            &mut full_sink,
        )
        .unwrap();

        let mut workspace_p0 = factor.make_solve_workspace(2).unwrap();
        let p0 = scan_with_p0y_and_exact_p0_sparse(
            &factor,
            &prepared,
            &x_design,
            &x_col,
            &xtx_chol,
            &p0y_vec,
            &xt_p0_x_chol,
            sigma2,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
            &mut workspace_p0,
        )
        .unwrap();

        assert_close(&full, &p0, 1e-10_f64);
    }

    #[test]
    fn build_sparse_jxlmm_null_state_matches_manual_reference() {
        let x_design = make_test_design();
        let y_vec = vec![0.5_f64, -1.0_f64, 1.25_f64];
        let factor = make_diag_factor(&[2.0_f64, 3.0_f64, 4.0_f64]);
        let sigma2 = 2.75_f64;
        let n = y_vec.len();
        let p = x_design.len() / n;

        let mut workspace = factor.make_solve_workspace(p.max(1)).unwrap();
        let built =
            build_sparse_jxlmm_null_state(&factor, &x_design, &y_vec, sigma2, &mut workspace, None)
                .unwrap();

        let x_col = row_major_to_col_major_f64(&x_design, n, p).unwrap();
        let mut workspace_manual = factor.make_solve_workspace(p.max(1)).unwrap();
        let y_vinv =
            sparse_solve_rhs_with_workspace(&factor, &y_vec, 1, &mut workspace_manual).unwrap();
        let x_vinv =
            sparse_solve_rhs_with_workspace(&factor, &x_col, p, &mut workspace_manual).unwrap();
        let mut xt_v_inv_y = vec![0.0_f64; p];
        xt_vec_row_major(&x_design, n, p, &y_vinv, &mut xt_v_inv_y);
        let xt_v_inv_x_chol = compute_xt_v_inv_x_chol(&factor, &x_design, n, p);
        let mut beta_hat = vec![0.0_f64; p];
        cholesky_solve_into(&xt_v_inv_x_chol, p, &xt_v_inv_y, &mut beta_hat);
        let mut py_manual = y_vinv.clone();
        for (cov_idx, beta) in beta_hat.iter().copied().enumerate() {
            if beta == 0.0_f64 {
                continue;
            }
            let vinvx_col = &x_vinv[cov_idx * n..(cov_idx + 1) * n];
            for i in 0..n {
                py_manual[i] -= vinvx_col[i] * beta;
            }
        }
        let p0y_manual: Vec<f64> = py_manual.iter().map(|v| v * sigma2).collect();

        assert_close(&built.x_design_col_major, &x_col, 1e-12_f64);
        assert_close(&built.beta_hat, &beta_hat, 1e-12_f64);
        assert_close(&built.py, &py_manual, 1e-12_f64);
        assert_close(&built.null_model.p0y, &p0y_manual, 1e-12_f64);
        assert_eq!(built.null_model.n_samples, n);
        assert_eq!(built.null_model.n_covariates, p);
        assert!((built.null_model.sigma2 - sigma2).abs() <= 1e-12_f64);
    }

    #[test]
    fn estimate_rhat_and_scan_sparse_exact_mode_supports_block_rows_zero() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let y_vec = vec![0.5_f64, -1.0_f64, 1.25_f64];
        let factor = make_diag_factor(&[2.0_f64, 3.0_f64, 4.0_f64]);

        let (r_hat, out, null_info, rhat_info) = estimate_rhat_and_scan_sparse(
            factor,
            prepared,
            x_design,
            y_vec,
            scan_sample_idx,
            PackedGeneticModel::Add,
            1.0_f64,
            1.0_f64,
            1,
            0,
            2,
            20260604_u64,
            None,
            None,
            0,
            SplmmScanMode::Exact,
        )
        .unwrap();

        assert!(r_hat.is_nan());
        assert_eq!(out.len(), 9);
        assert!(out.iter().all(|v| v.is_finite() || v.is_nan()));
        assert!(null_info.v_inv_y.converged);
        assert!(null_info.v_inv_x.converged_all);
        assert_eq!(rhat_info.n_markers_requested, 0);
        assert_eq!(rhat_info.n_markers_used, 0);
    }

    #[test]
    fn exact_pg_relative_guard_flags_near_singular_rows() {
        let s_m_s = 10.0_f64;
        let s_p_s = SPLMM_EXACT_PG_MIN_REL_SM * s_m_s * 0.5;
        assert!(exact_pg_is_numerically_singular(s_p_s, s_m_s));
        assert!(!exact_pg_is_numerically_singular(
            SPLMM_EXACT_PG_MIN_REL_SM * s_m_s * 2.0,
            s_m_s,
        ));
    }

    #[test]
    fn merge_exact_results_into_scan_skips_invalid_rows() {
        let mut approx = vec![
            0.1_f64, 1.0_f64, 0.9_f64, 0.2_f64, 2.0_f64, 0.8_f64, 0.3_f64, 3.0_f64, 0.7_f64,
        ];
        let exact = vec![f64::NAN, f64::NAN, 1.0_f64, 9.0_f64, 0.5_f64, 1.0e-6_f64];
        merge_exact_results_into_scan(&mut approx, &exact, &[0usize, 2usize]).unwrap();
        assert_eq!(approx[0..3], [0.1_f64, 1.0_f64, 0.9_f64]);
        assert_eq!(approx[3..6], [0.2_f64, 2.0_f64, 0.8_f64]);
        assert_eq!(approx[6..9], [9.0_f64, 0.5_f64, 1.0e-6_f64]);
    }

    #[test]
    fn merge_two_stage_tsv_with_exact_rows_replaces_only_candidates() {
        let (prepared, _) = make_test_scan_prepared();
        let chrom = vec!["1".to_string(), "1".to_string(), "2".to_string()];
        let pos = vec![101_i64, 202_i64, 303_i64];
        let snp = vec!["rs0".to_string(), "rs1".to_string(), "rs2".to_string()];
        let allele0 = vec!["A".to_string(), "C".to_string(), "G".to_string()];
        let allele1 = vec!["T".to_string(), "G".to_string(), "A".to_string()];
        let approx_path = unique_temp_path("approx");
        let merged_path = unique_temp_path("merged");
        let header = "chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\n";
        let approx_rows = [
            "1\t101\trs0\tA\tT\t0.3750\t0.0000\t0.1000\t1.0000\t0.0100\t9.0000e-1\n",
            "1\t202\trs1\tC\tG\t0.5000\t0.2500\t0.2000\t2.0000\t0.0100\t8.0000e-1\n",
            "2\t303\trs2\tG\tA\t0.5000\t0.0000\t0.3000\t3.0000\t0.0100\t7.0000e-1\n",
        ];
        {
            let mut file = File::create(&approx_path).unwrap();
            file.write_all(header.as_bytes()).unwrap();
            for row in approx_rows.iter() {
                file.write_all(row.as_bytes()).unwrap();
            }
        }

        let written = merge_two_stage_tsv_with_exact_rows(
            approx_path.to_str().unwrap(),
            merged_path.to_str().unwrap(),
            &prepared,
            &chrom,
            &pos,
            &snp,
            &allele0,
            &allele1,
            &[9.0_f64, 0.5_f64, 1.0e-6_f64],
            &[1usize],
            PackedGeneticModel::Add,
        )
        .unwrap();
        assert_eq!(written, 3usize);

        let merged = fs::read_to_string(&merged_path).unwrap();
        let lines: Vec<&str> = merged.lines().collect();
        assert_eq!(lines.len(), 4usize);
        assert_eq!(lines[0], header.trim_end());
        assert_eq!(lines[1], approx_rows[0].trim_end());
        assert_eq!(lines[3], approx_rows[2].trim_end());
        assert_ne!(lines[2], approx_rows[1].trim_end());
        assert!(lines[2].contains("\trs1\t"));
        assert!(lines[2].contains("\t9.0000\t0.5000\t"));
        assert!(lines[2].ends_with("\t1.0000e-6"));

        let _ = fs::remove_file(&approx_path);
        let _ = fs::remove_file(&merged_path);
    }

    #[test]
    fn merge_two_stage_tsv_with_exact_rows_keeps_approx_for_invalid_exact() {
        let (prepared, _) = make_test_scan_prepared();
        let chrom = vec!["1".to_string(), "1".to_string(), "2".to_string()];
        let pos = vec![101_i64, 202_i64, 303_i64];
        let snp = vec!["rs0".to_string(), "rs1".to_string(), "rs2".to_string()];
        let allele0 = vec!["A".to_string(), "C".to_string(), "G".to_string()];
        let allele1 = vec!["T".to_string(), "G".to_string(), "A".to_string()];
        let approx_path = unique_temp_path("approx_invalid");
        let merged_path = unique_temp_path("merged_invalid");
        let header = "chrom\tpos\tsnp\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\n";
        let approx_rows = [
            "1\t101\trs0\tA\tT\t0.3750\t0.0000\t0.1000\t1.0000\t0.0100\t9.0000e-1\n",
            "1\t202\trs1\tC\tG\t0.5000\t0.2500\t0.2000\t2.0000\t0.0100\t8.0000e-1\n",
            "2\t303\trs2\tG\tA\t0.5000\t0.0000\t0.3000\t3.0000\t0.0100\t7.0000e-1\n",
        ];
        {
            let mut file = File::create(&approx_path).unwrap();
            file.write_all(header.as_bytes()).unwrap();
            for row in approx_rows.iter() {
                file.write_all(row.as_bytes()).unwrap();
            }
        }

        let written = merge_two_stage_tsv_with_exact_rows(
            approx_path.to_str().unwrap(),
            merged_path.to_str().unwrap(),
            &prepared,
            &chrom,
            &pos,
            &snp,
            &allele0,
            &allele1,
            &[f64::NAN, f64::NAN, 1.0_f64],
            &[1usize],
            PackedGeneticModel::Add,
        )
        .unwrap();
        assert_eq!(written, 3usize);

        let merged = fs::read_to_string(&merged_path).unwrap();
        let lines: Vec<&str> = merged.lines().collect();
        assert_eq!(lines.len(), 4usize);
        assert_eq!(lines[2], approx_rows[1].trim_end());

        let _ = fs::remove_file(&approx_path);
        let _ = fs::remove_file(&merged_path);
    }
}
