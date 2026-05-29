use memmap2::Mmap;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cell::Cell;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use crate::bedmath::{
    adaptive_grm_block_rows, decode_mean_imputed_additive_packed_block_rows_f32, packed_byte_lut,
};
use crate::blas::{rust_sgemm_prefers_rayon_rowmajor_f32_kernel, BlasThreadGuard};
use crate::cholesky::{
    sparse_cholesky_analyze_subset_jxgrm_path_cached_with_progress, sparse_jxgrm_header_n_samples,
    subset_sparse_grm_csc, MmapSparseGrmCsc, SparseGrmCscView, SparseJxgrmAnalyzeProgressStage,
    SparseJxgrmCholesky,
};
use crate::gfcore::read_fam;
use crate::gfreader::prepare_bed_logic_meta_owned_for_stats_samples;
use crate::he::row_major_block_mul_mat_f32;
use crate::linalg::{
    chi2_sf_df1, chisq_from_beta_se_and_optional_plrt, cholesky_inplace, cholesky_solve_into,
    format_chisq_value,
};
use crate::pcg::{
    pcg_build_jxlmm_null_model, pcg_estimate_jxlmm_r_hat, PcgGrmBedConfig, PcgGrmBedOperator,
    PcgJxlmmNullModel, PcgJxlmmNullModelInfo, PcgJxlmmRHatResult,
};
use crate::stats_common::{get_cached_pool, parse_index_vec_i64};

const JXLMM_TINY: f64 = 1e-30_f64;
const JXLMM_DEFAULT_RHAT_MARKERS: usize = 30;
const JXLMM_DEFAULT_RHAT_SEED: u64 = 20260527;
const JXLMM_DEFAULT_SPARSE_CHOLESKY_MAX_L_NNZ: usize = 100_000_000;

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
            "jxlmm_assoc_pcg_bed: packed payload path requires `packed` argument.",
        )
    })?;
    let maf_ro = maf.ok_or_else(|| {
        PyRuntimeError::new_err("jxlmm_assoc_pcg_bed: packed payload path requires `maf` argument.")
    })?;
    let row_flip_ro = row_flip.ok_or_else(|| {
        PyRuntimeError::new_err(
            "jxlmm_assoc_pcg_bed: packed payload path requires `row_flip` argument.",
        )
    })?;
    if packed_n_samples == 0 {
        return Err(PyRuntimeError::new_err(
            "jxlmm_assoc_pcg_bed: packed payload path requires packed_n_samples > 0",
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
        PyRuntimeError::new_err("jxlmm_assoc_pcg_bed: mmap metadata path requires `maf` argument.")
    })?;
    let row_flip_ro = row_flip.ok_or_else(|| {
        PyRuntimeError::new_err(
            "jxlmm_assoc_pcg_bed: mmap metadata path requires `row_flip` argument.",
        )
    })?;
    let row_indices_ro = row_indices.ok_or_else(|| {
        PyRuntimeError::new_err(
            "jxlmm_assoc_pcg_bed: mmap metadata path requires `row_indices` argument.",
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
fn prepare_jxlmm_assoc_inputs<'py>(
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
            "n must be > p for JXLMM: n={n}, p={p}"
        )));
    }

    let use_external_packed = packed.is_some() || packed_n_samples > 0;
    let use_external_mmap_meta =
        !use_external_packed && (maf.is_some() || row_flip.is_some() || row_missing.is_some() || row_indices.is_some());
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
        prepare_external_packed_input(packed, packed_n_samples, maf, row_flip, row_missing, row_indices)?
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

fn make_bed_operator<'a, F>(
    prepared: &'a JxlmmPreparedInput,
    sample_idx: &[usize],
    threads: usize,
    block_rows: usize,
    std_eps: f64,
    use_train_maf: bool,
    progress_callback: Option<&mut F>,
    progress_every_rows: usize,
) -> Result<PcgGrmBedOperator<'a>, String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    let cfg = PcgGrmBedConfig {
        sample_idx: sample_idx.to_vec(),
        row_flip: prepared.row_flip.clone(),
        row_maf: prepared.row_maf.clone(),
        row_indices: prepared.row_source_indices.clone(),
        threads,
        block_rows,
        std_eps: std_eps as f32,
        use_train_maf,
    };
    match &prepared.payload {
        JxlmmPayload::Packed(bytes) => PcgGrmBedOperator::from_packed_rows_with_progress(
            std::borrow::Cow::Borrowed(bytes.as_ref()),
            prepared.n_samples_full,
            cfg,
            progress_callback,
            progress_every_rows,
        ),
        JxlmmPayload::Mmap(_) => PcgGrmBedOperator::from_bed_mmap_prefix_with_progress(
            prepared
                .bed_prefix
                .as_deref()
                .ok_or_else(|| "JXLMM mmap payload missing BED prefix".to_string())?,
            cfg,
            progress_callback,
            progress_every_rows,
        ),
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

#[inline]
fn emit_progress_callback_throttled(
    cb: Option<&Py<PyAny>>,
    stage: usize,
    done: usize,
    total: usize,
    every: usize,
    last_sent: &Cell<usize>,
) -> Result<(), String> {
    if cb.is_none() {
        return Ok(());
    }
    let total_use = total.max(1);
    let done_use = done.min(total_use);
    let step = every.max(1);
    let last = last_sent.get();
    if done_use == total_use || done_use >= last.saturating_add(step) {
        emit_progress_callback(cb, stage, done_use, total_use)?;
        last_sent.set(done_use);
    }
    Ok(())
}

fn xtx_chol_from_design(x_design: &[f64], n: usize, p: usize) -> Result<Vec<f64>, String> {
    if n == 0 || p == 0 {
        return Err("JXLMM XtX chol requires n > 0 and p > 0".to_string());
    }
    if x_design.len() != n * p {
        return Err(format!(
            "JXLMM XtX design length mismatch: got {}, expected {}",
            x_design.len(),
            n * p
        ));
    }
    let mut xtx = vec![0.0_f64; p * p];
    for i in 0..n {
        let row = &x_design[i * p..(i + 1) * p];
        for a in 0..p {
            let xa = row[a];
            for b in 0..=a {
                xtx[a * p + b] += xa * row[b];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            xtx[b * p + a] = xtx[a * p + b];
        }
    }

    let mut chol = xtx.clone();
    if cholesky_inplace(&mut chol, p).is_some() {
        return Ok(chol);
    }
    let mut jitter = 1e-10_f64;
    for _ in 0..8 {
        chol.copy_from_slice(&xtx);
        for d in 0..p {
            chol[d * p + d] += jitter;
        }
        if cholesky_inplace(&mut chol, p).is_some() {
            return Ok(chol);
        }
        jitter *= 10.0;
    }
    Err("JXLMM XtX is not SPD even after diagonal jitter".to_string())
}

#[inline]
fn cast_f64_slice_to_f32(input: &[f64]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(input.len());
    for &value in input {
        if !value.is_finite() {
            return Err("JXLMM scan received non-finite dense operand".to_string());
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
fn col_major_to_row_major_f64(
    input: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<f64>, String> {
    if input.len() != n_rows.saturating_mul(n_cols) {
        return Err(format!(
            "col-major matrix length mismatch: got {}, expected {}",
            input.len(),
            n_rows.saturating_mul(n_cols)
        ));
    }
    let mut out = vec![0.0_f64; n_rows.saturating_mul(n_cols)];
    for col in 0..n_cols {
        let src = &input[col * n_rows..(col + 1) * n_rows];
        for row in 0..n_rows {
            out[row * n_cols + col] = src[row];
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
fn xt_mat_row_major(
    x_row_major: &[f64],
    y_row_major: &[f64],
    n_samples: usize,
    n_covariates: usize,
    n_cols_y: usize,
    out: &mut [f64],
) {
    out.fill(0.0);
    for i in 0..n_samples {
        let x_row = &x_row_major[i * n_covariates..(i + 1) * n_covariates];
        let y_row = &y_row_major[i * n_cols_y..(i + 1) * n_cols_y];
        for r in 0..n_covariates {
            let xr = x_row[r];
            for c in 0..n_cols_y {
                out[r * n_cols_y + c] += xr * y_row[c];
            }
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
        return Err("Sparse JXLMM diagonal stats require n_samples > 0".to_string());
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
                        "Sparse JXLMM diagonal contains non-finite value at column {col}"
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
            return Err(format!("Sparse JXLMM diagonal is missing at column {col}"));
        }
    }
    Ok((
        (sum_abs / (csc.n_samples() as f64)).max(JXLMM_TINY),
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
        .unwrap_or(JXLMM_DEFAULT_SPARSE_CHOLESKY_MAX_L_NNZ)
}

#[inline]
fn normalize_sparse_subset_request<'a>(
    sample_idx: &'a [usize],
    full_n: usize,
) -> Result<Option<&'a [usize]>, String> {
    if sample_idx.iter().any(|&sid| sid >= full_n) {
        return Err(format!(
            "Sparse JXLMM sample index out of bounds for sparse n_samples={full_n}"
        ));
    }
    if sample_idx.len() == full_n && sample_idx.iter().enumerate().all(|(i, &sid)| sid == i) {
        Ok(None)
    } else {
        Ok(Some(sample_idx))
    }
}

#[inline]
fn sparse_jxlmm_load_factor(
    prefix: &str,
    sparse_jxgrm_path: Option<&str>,
    expected_n: usize,
    sample_idx: &[usize],
    sigma_g2: f64,
    sigma_e2: f64,
    force_sparse: bool,
    progress_callback: Option<&Py<PyAny>>,
) -> Result<Option<SparseJxgrmCholesky>, String> {
    let path = sparse_jxgrm_path
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{prefix}.jxgrm"));
    if !Path::new(&path).exists() {
        if force_sparse {
            return Err(format!(
                "Sparse JXLMM requested but sparse kinship file not found: {path}"
            ));
        }
        return Ok(None);
    }
    let sparse_n = sparse_jxgrm_header_n_samples(&path)?;
    let subset_request = normalize_sparse_subset_request(sample_idx, sparse_n)?;
    let target_n = subset_request.map(|idx| idx.len()).unwrap_or(sparse_n);
    if target_n != expected_n {
        return Err(format!(
            "Sparse JXLMM kinship size mismatch: sparse target n={}, expected {} from {path}",
            target_n, expected_n
        ));
    }
    if !sigma_g2.is_finite() || sigma_g2 < 0.0 || !sigma_e2.is_finite() || sigma_e2 < 0.0 {
        return Err(format!(
            "Sparse JXLMM requires finite non-negative variance components, got sigma_g2={sigma_g2}, sigma_e2={sigma_e2}"
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
            "Sparse JXLMM analyzed dim mismatch: analyzed {}, expected {} from {path}",
            analysis.dim(),
            expected_n
        ));
    }
    let (diag_mean_abs, diag_min, diag_max) = analysis.diag_stats_scaled(sigma_g2, sigma_e2)?;
    let max_l_nnz = sparse_cholesky_max_l_nnz();
    let estimated_l_nnz = analysis.factor_nnz_estimate();
    if estimated_l_nnz > max_l_nnz {
        let msg = format!(
            "Sparse JXLMM symbolic factor is too large for direct Cholesky: estimated L nnz={} exceeds limit {}. \
Set `JX_SPARSE_CHOLESKY_MAX_L_NNZ` to override, or fall back to operator PCG.",
            estimated_l_nnz, max_l_nnz
        );
        if force_sparse {
            return Err(msg);
        }
        return Ok(None);
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
                return Ok(Some(factor));
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
        "sigma_e2 is near zero, so sparse thresholding can make V indefinite; auto-ridge failed. Try a smaller sparse cutoff such as `-jxlmm 0.01` or `-jxlmm 0.001`."
    } else {
        "hard-thresholded sparse GRM is not numerically SPD enough for LLT; try a smaller sparse cutoff such as `-jxlmm 0.01` or `-jxlmm 0.001`."
    };
    let msg = format!(
        "Sparse JXLMM factorization failed after adaptive diagonal ridge escalation; \
sigma_g2={sigma_g2:.6e}, sigma_e2={sigma_e2:.6e}, mean_diag={diag_mean_abs:.6e}, \
min_diag={diag_min:.6e}, max_diag={diag_max:.6e}. Last error: {last_msg}. Hint: {hint}"
    );
    if force_sparse {
        Err(msg)
    } else {
        Ok(None)
    }
}

#[inline]
fn sparse_solve_rhs(
    factor: &SparseJxgrmCholesky,
    rhs_col_major: &[f64],
    n_rhs: usize,
) -> Result<Vec<f64>, String> {
    let mut out = rhs_col_major.to_vec();
    factor.solve_in_place(&mut out, n_rhs)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn append_jxlmm_tsv_block(
    text_buf: &mut String,
    row_start: usize,
    rows_here: usize,
    results: &[f64],
    chrom: &[String],
    pos: &[i64],
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
        let pwald = results[base + 2];
        let chisq_txt = format_chisq_value(chisq_from_beta_se_and_optional_plrt(beta, se, None));
        let (a0, a1) = transform_alleles_by_model_local(&allele0[row_idx], &allele1[row_idx], gm);
        let _ = writeln!(
            text_buf,
            "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4e}",
            chrom[row_idx],
            pos[row_idx],
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

#[allow(clippy::too_many_arguments)]
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
) -> Result<Vec<f64>, String> {
    if !(r_hat.is_finite() && r_hat > 0.0) {
        return Err(format!(
            "JXLMM scan requires finite positive r_hat, got {r_hat}"
        ));
    }
    let n = py_vec.len();
    if n == 0 {
        return Err("JXLMM scan requires non-empty Py vector".to_string());
    }
    if x_design.len() % n != 0 {
        return Err(format!(
            "JXLMM scan design length mismatch: len(x_design)={} is not divisible by n={n}",
            x_design.len()
        ));
    }
    let p = x_design.len() / n;
    if p == 0 {
        return Err("JXLMM scan requires at least one design column".to_string());
    }
    if xtx_chol.len() != p * p {
        return Err(format!(
            "JXLMM scan XtX chol length mismatch: got {}, expected {}",
            xtx_chol.len(),
            p * p
        ));
    }
    if scan_sample_idx.len() != n {
        return Err(format!(
            "JXLMM scan sample length mismatch: sample_idx={}, expected {n}",
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
    let m = scan_prepared.n_rows();
    let mut out = vec![0.0_f64; m * 3];
    let progress_block = if progress_every == 0 {
        block_rows.max(512).max(1)
    } else {
        progress_every.max(1)
    };
    if scan_progress_callback.is_some() {
        emit_progress_callback(scan_progress_callback, progress_stage, 0, m.max(1))?;
    }

    if matches!(gm, PackedGeneticModel::Add) {
        let scan_block_rows = adaptive_grm_block_rows(progress_block, m, n, 0usize, threads).max(1);
        let code4_lut = &packed_byte_lut().code4;
        let py_f32 = cast_f64_slice_to_f32(py_vec)?;
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
        let mut row_start = 0usize;
        while row_start < m {
            let row_end = (row_start + scan_block_rows).min(m);
            let rows_here = row_end - row_start;
            let block_slice = &mut block[..rows_here * n];
            let spy_slice = &mut spy_block[..rows_here];
            let xts_slice = &mut xts_block[..rows_here * p];
            let ss_slice = &mut row_ss[..rows_here];
            decode_mean_imputed_additive_packed_block_rows_f32(
                scan_prepared.payload.as_bytes(),
                scan_prepared.bytes_per_snp,
                scan_prepared.n_samples_full,
                &scan_prepared.row_flip,
                &scan_prepared.row_maf,
                scan_sample_idx,
                sample_identity,
                scan_prepared.row_source_indices.as_deref(),
                row_start,
                block_slice,
                code4_lut,
                pool.as_ref(),
            )?;
            row_major_block_mul_mat_f32(
                block_slice,
                rows_here,
                n,
                py_f32.as_slice(),
                1usize,
                spy_slice,
                pool.as_ref(),
            );
            row_major_block_mul_mat_f32(
                block_slice,
                rows_here,
                n,
                x_design_f32.as_slice(),
                p,
                xts_slice,
                pool.as_ref(),
            );
            row_major_block_sumsq_f64(block_slice, rows_here, n, ss_slice, pool.as_ref());
            let out_block = &mut out[row_start * 3..row_end * 3];
            for local_idx in 0..rows_here {
                let xts_row = &xts_slice[local_idx * p..(local_idx + 1) * p];
                for j in 0..p {
                    xts[j] = xts_row[j] as f64;
                }
                cholesky_solve_into(xtx_chol, p, &xts, &mut alpha);
                let x_quad = xts
                    .iter()
                    .zip(alpha.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let s_m_s = (ss_slice[local_idx] - x_quad).max(0.0_f64);
                let denom = r_hat * s_m_s;
                let out_row = &mut out_block[local_idx * 3..(local_idx + 1) * 3];
                if denom.is_finite() && denom > JXLMM_TINY {
                    let spy = spy_slice[local_idx] as f64;
                    let beta = spy / denom;
                    let se = 1.0_f64 / denom.sqrt();
                    let chisq = (spy * spy) / denom;
                    out_row[0] = beta;
                    out_row[1] = se;
                    out_row[2] = chi2_sf_df1(chisq);
                } else {
                    out_row[0] = f64::NAN;
                    out_row[1] = f64::NAN;
                    out_row[2] = 1.0_f64;
                }
            }
            emit_progress_callback(scan_progress_callback, progress_stage, row_end, m.max(1))?;
            row_start = row_end;
        }
    } else {
        let thread_hint = threads.max(1);
        let row_tile = progress_block.div_ceil(thread_hint).clamp(32, 1024);
        let mut row_start = 0usize;
        while row_start < m {
            let row_end = (row_start + progress_block).min(m);
            let out_block = &mut out[row_start * 3..row_end * 3];
            let mut run_block = || {
                out_block
                    .par_chunks_mut(row_tile * 3)
                    .enumerate()
                    .for_each_init(
                        || (vec![0.0_f64; n], vec![0.0_f64; p], vec![0.0_f64; p]),
                        |(snp, xts, alpha), (tile_idx, out_tile)| {
                            let tile_row_start = row_start + tile_idx * row_tile;
                            for (local_idx, out_row) in out_tile.chunks_mut(3).enumerate() {
                                let row_idx = tile_row_start + local_idx;
                                decode_packed_row_model_into_f64(
                                    scan_prepared.row_bytes(row_idx),
                                    scan_prepared.row_flip[row_idx],
                                    scan_prepared.row_maf[row_idx],
                                    n,
                                    gm,
                                    sample_identity,
                                    sample_byte_idx.as_deref(),
                                    sample_bit_shift.as_deref(),
                                    snp,
                                );
                                xts.fill(0.0);
                                let mut spy = 0.0_f64;
                                let mut s_sq = 0.0_f64;
                                for i in 0..n {
                                    let s_i = snp[i];
                                    spy += s_i * py_vec[i];
                                    s_sq += s_i * s_i;
                                    let x_row = &x_design[i * p..(i + 1) * p];
                                    for j in 0..p {
                                        xts[j] += x_row[j] * s_i;
                                    }
                                }
                                cholesky_solve_into(xtx_chol, p, xts, alpha);
                                let x_quad = xts
                                    .iter()
                                    .zip(alpha.iter())
                                    .map(|(a, b)| a * b)
                                    .sum::<f64>();
                                let s_m_s = (s_sq - x_quad).max(0.0);
                                let denom = r_hat * s_m_s;
                                if denom.is_finite() && denom > JXLMM_TINY {
                                    let beta = spy / denom;
                                    let se = 1.0 / denom.sqrt();
                                    let chisq = (spy * spy) / denom;
                                    out_row[0] = beta;
                                    out_row[1] = se;
                                    out_row[2] = chi2_sf_df1(chisq);
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
            emit_progress_callback(scan_progress_callback, progress_stage, row_end, m.max(1))?;
            row_start = row_end;
        }
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn scan_to_tsv_with_py_and_rhat(
    scan_prepared: &JxlmmPreparedInput,
    x_design: &[f64],
    py_vec: &[f64],
    xtx_chol: &[f64],
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    r_hat: f64,
    threads: usize,
    block_rows: usize,
    chrom: &[String],
    pos: &[i64],
    allele0: &[String],
    allele1: &[String],
    out_tsv: &str,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
) -> Result<usize, String> {
    if chrom.len() != scan_prepared.n_rows()
        || pos.len() != scan_prepared.n_rows()
        || allele0.len() != scan_prepared.n_rows()
        || allele1.len() != scan_prepared.n_rows()
    {
        return Err(format!(
            "JXLMM TSV metadata length mismatch: rows={}, chrom={}, pos={}, allele0={}, allele1={}",
            scan_prepared.n_rows(),
            chrom.len(),
            pos.len(),
            allele0.len(),
            allele1.len()
        ));
    }
    if !(r_hat.is_finite() && r_hat > 0.0) {
        return Err(format!(
            "JXLMM scan requires finite positive r_hat, got {r_hat}"
        ));
    }
    let n = py_vec.len();
    if n == 0 {
        return Err("JXLMM scan requires non-empty Py vector".to_string());
    }
    if x_design.len() % n != 0 {
        return Err(format!(
            "JXLMM scan design length mismatch: len(x_design)={} is not divisible by n={n}",
            x_design.len()
        ));
    }
    let p = x_design.len() / n;
    if p == 0 {
        return Err("JXLMM scan requires at least one design column".to_string());
    }
    if xtx_chol.len() != p * p {
        return Err(format!(
            "JXLMM scan XtX chol length mismatch: got {}, expected {}",
            xtx_chol.len(),
            p * p
        ));
    }
    if scan_sample_idx.len() != n {
        return Err(format!(
            "JXLMM scan sample length mismatch: sample_idx={}, expected {n}",
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
    let m = scan_prepared.n_rows();
    let progress_block = if progress_every == 0 {
        block_rows.max(512).max(1)
    } else {
        progress_every.max(1)
    };
    if scan_progress_callback.is_some() {
        emit_progress_callback(scan_progress_callback, progress_stage, 0, m.max(1))?;
    }

    let out_path = out_tsv.to_string();
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(4);
    let writer_handle = std::thread::spawn(move || -> Result<(), String> {
        let out_file = File::create(&out_path).map_err(|e| format!("create {out_path}: {e}"))?;
        let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, out_file);
        writer
            .write_all(b"chrom\tpos\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald\n")
            .map_err(|e| format!("write header {out_path}: {e}"))?;
        for block in rx {
            if !block.is_empty() {
                writer
                    .write_all(&block)
                    .map_err(|e| format!("write {out_path}: {e}"))?;
            }
        }
        writer
            .flush()
            .map_err(|e| format!("flush {out_path}: {e}"))?;
        Ok(())
    });

    let mut text_buf = String::with_capacity(progress_block.max(1) * 112);
    let code4_lut = &packed_byte_lut().code4;

    let run_res: Result<(), String> = (|| {
        if matches!(gm, PackedGeneticModel::Add) {
            let scan_block_rows =
                adaptive_grm_block_rows(progress_block, m, n, 0usize, threads).max(1);
            let py_f32 = cast_f64_slice_to_f32(py_vec)?;
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
            let mut out_block = vec![0.0_f64; scan_block_rows * 3];
            let mut xts = vec![0.0_f64; p];
            let mut alpha = vec![0.0_f64; p];
            let mut row_start = 0usize;
            while row_start < m {
                let row_end = (row_start + scan_block_rows).min(m);
                let rows_here = row_end - row_start;
                let block_slice = &mut block[..rows_here * n];
                let spy_slice = &mut spy_block[..rows_here];
                let xts_slice = &mut xts_block[..rows_here * p];
                let ss_slice = &mut row_ss[..rows_here];
                let out_slice = &mut out_block[..rows_here * 3];
                decode_mean_imputed_additive_packed_block_rows_f32(
                    scan_prepared.payload.as_bytes(),
                    scan_prepared.bytes_per_snp,
                    scan_prepared.n_samples_full,
                    &scan_prepared.row_flip,
                    &scan_prepared.row_maf,
                    scan_sample_idx,
                    sample_identity,
                    scan_prepared.row_source_indices.as_deref(),
                    row_start,
                    block_slice,
                    code4_lut,
                    pool.as_ref(),
                )?;
                row_major_block_mul_mat_f32(
                    block_slice,
                    rows_here,
                    n,
                    py_f32.as_slice(),
                    1usize,
                    spy_slice,
                    pool.as_ref(),
                );
                row_major_block_mul_mat_f32(
                    block_slice,
                    rows_here,
                    n,
                    x_design_f32.as_slice(),
                    p,
                    xts_slice,
                    pool.as_ref(),
                );
                row_major_block_sumsq_f64(block_slice, rows_here, n, ss_slice, pool.as_ref());
                for local_idx in 0..rows_here {
                    let xts_row = &xts_slice[local_idx * p..(local_idx + 1) * p];
                    for j in 0..p {
                        xts[j] = xts_row[j] as f64;
                    }
                    cholesky_solve_into(xtx_chol, p, &xts, &mut alpha);
                    let x_quad = xts
                        .iter()
                        .zip(alpha.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    let s_m_s = (ss_slice[local_idx] - x_quad).max(0.0_f64);
                    let denom = r_hat * s_m_s;
                    let out_row = &mut out_slice[local_idx * 3..(local_idx + 1) * 3];
                    if denom.is_finite() && denom > JXLMM_TINY {
                        let spy = spy_slice[local_idx] as f64;
                        let beta = spy / denom;
                        let se = 1.0_f64 / denom.sqrt();
                        let chisq = (spy * spy) / denom;
                        out_row[0] = beta;
                        out_row[1] = se;
                        out_row[2] = chi2_sf_df1(chisq);
                    } else {
                        out_row[0] = f64::NAN;
                        out_row[1] = f64::NAN;
                        out_row[2] = 1.0_f64;
                    }
                }
                text_buf.clear();
                append_jxlmm_tsv_block(
                    &mut text_buf,
                    row_start,
                    rows_here,
                    out_slice,
                    chrom,
                    pos,
                    allele0,
                    allele1,
                    &scan_prepared.row_maf,
                    &scan_prepared.row_missing,
                    gm,
                );
                let payload = std::mem::take(&mut text_buf).into_bytes();
                tx.send(payload)
                    .map_err(|e| format!("writer queue send failed for {out_tsv}: {e}"))?;
                emit_progress_callback(scan_progress_callback, progress_stage, row_end, m.max(1))?;
                row_start = row_end;
            }
        } else {
            let row_tile = progress_block.div_ceil(threads.max(1)).clamp(32, 1024);
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
                                    decode_packed_row_model_into_f64(
                                        scan_prepared.row_bytes(row_idx),
                                        scan_prepared.row_flip[row_idx],
                                        scan_prepared.row_maf[row_idx],
                                        n,
                                        gm,
                                        sample_identity,
                                        sample_byte_idx.as_deref(),
                                        sample_bit_shift.as_deref(),
                                        snp,
                                    );
                                    xts.fill(0.0);
                                    let mut spy = 0.0_f64;
                                    let mut s_sq = 0.0_f64;
                                    for i in 0..n {
                                        let s_i = snp[i];
                                        spy += s_i * py_vec[i];
                                        s_sq += s_i * s_i;
                                        let x_row = &x_design[i * p..(i + 1) * p];
                                        for j in 0..p {
                                            xts[j] += x_row[j] * s_i;
                                        }
                                    }
                                    cholesky_solve_into(xtx_chol, p, xts, alpha);
                                    let x_quad = xts
                                        .iter()
                                        .zip(alpha.iter())
                                        .map(|(a, b)| a * b)
                                        .sum::<f64>();
                                    let s_m_s = (s_sq - x_quad).max(0.0);
                                    let denom = r_hat * s_m_s;
                                    if denom.is_finite() && denom > JXLMM_TINY {
                                        let beta = spy / denom;
                                        let se = 1.0 / denom.sqrt();
                                        let chisq = (spy * spy) / denom;
                                        out_row[0] = beta;
                                        out_row[1] = se;
                                        out_row[2] = chi2_sf_df1(chisq);
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
                text_buf.clear();
                append_jxlmm_tsv_block(
                    &mut text_buf,
                    row_start,
                    rows_here,
                    out_slice,
                    chrom,
                    pos,
                    allele0,
                    allele1,
                    &scan_prepared.row_maf,
                    &scan_prepared.row_missing,
                    gm,
                );
                let payload = std::mem::take(&mut text_buf).into_bytes();
                tx.send(payload)
                    .map_err(|e| format!("writer queue send failed for {out_tsv}: {e}"))?;
                emit_progress_callback(scan_progress_callback, progress_stage, row_end, m.max(1))?;
                row_start = row_end;
            }
        }
        Ok(())
    })();

    drop(tx);
    let writer_result = writer_handle
        .join()
        .map_err(|_| format!("writer thread panicked for {out_tsv}"))?;
    run_res?;
    writer_result?;
    Ok(m)
}

#[allow(clippy::too_many_arguments)]
fn estimate_rhat_and_scan_sparse(
    factor: SparseJxgrmCholesky,
    scan_prepared: JxlmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    scan_sample_idx: Vec<usize>,
    gm: PackedGeneticModel,
    threads: usize,
    block_rows: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> Result<(f64, Vec<f64>, PcgJxlmmNullModelInfo, PcgJxlmmRHatResult), String> {
    let n = y_vec.len();
    let p = x_design.len() / n;
    let stage1_cb = stage1_progress_callback.as_ref();
    let rhat_progress_total = n_rhat_progress_total(scan_prepared.n_rows(), rhat_markers);

    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 6, 0, 1)?;
    }
    let y_vinv_col = sparse_solve_rhs(&factor, &y_vec, 1)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 6, 1, 1)?;
    }

    let x_col = row_major_to_col_major_f64(&x_design, n, p)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 7, 0, p.max(1))?;
    }
    let x_vinv_col = sparse_solve_rhs(&factor, &x_col, p)?;
    let x_vinv_row = col_major_to_row_major_f64(&x_vinv_col, n, p)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 7, p.max(1), p.max(1))?;
    }

    let mut xt_v_inv_y = vec![0.0_f64; p];
    xt_vec_row_major(&x_design, n, p, &y_vinv_col, &mut xt_v_inv_y);

    let mut xt_v_inv_x = vec![0.0_f64; p * p];
    xt_mat_row_major(&x_design, &x_vinv_row, n, p, p, &mut xt_v_inv_x);
    let xt_v_inv_x_chol = spd_cholesky_with_jitter(&xt_v_inv_x, p, "Sparse JXLMM XtVinvX")?;
    let mut beta_hat = vec![0.0_f64; p];
    cholesky_solve_into(&xt_v_inv_x_chol, p, &xt_v_inv_y, &mut beta_hat);

    let mut py = y_vinv_col.clone();
    for i in 0..n {
        let vinvx_row = &x_vinv_row[i * p..(i + 1) * p];
        let adjust = vinvx_row
            .iter()
            .zip(beta_hat.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        py[i] -= adjust;
    }

    let mut xtx = vec![0.0_f64; p * p];
    xt_mat_row_major(&x_design, &x_design, n, p, p, &mut xtx);
    let xtx_chol = spd_cholesky_with_jitter(&xtx, p, "Sparse JXLMM XtX")?;

    let null_model = PcgJxlmmNullModel {
        n_samples: n,
        n_covariates: p,
        v_inv_x: x_vinv_row,
        beta_hat,
        py,
        xt_v_inv_x_chol,
        xtx_chol,
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
    let mut sampled_markers = vec![0.0_f64; n * n_rhat];
    let mut tmp_snp = vec![0.0_f64; n];
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
    let v_inv_s_col = sparse_solve_rhs(&factor, &sampled_markers, n_rhat)?;
    let mut ratio_sum = 0.0_f64;
    let mut n_used = 0usize;
    let mut v_inv_col = vec![0.0_f64; n];
    for col in 0..n_rhat {
        let snp_col = &sampled_markers[col * n..(col + 1) * n];
        for row in 0..n {
            v_inv_col[row] = v_inv_s_col[col * n + row];
        }
        let s_m_s = crate::pcg::pcg_jxlmm_s_m_s(&null_model, &x_design, snp_col)?;
        if !s_m_s.is_finite() || s_m_s <= JXLMM_TINY {
            continue;
        }
        let s_p_s = crate::pcg::pcg_jxlmm_s_p_s_exact(&null_model, &x_design, snp_col, &v_inv_col)?;
        if !s_p_s.is_finite() || s_p_s <= JXLMM_TINY {
            continue;
        }
        let ratio = s_p_s / s_m_s;
        if ratio.is_finite() && ratio > 0.0 {
            ratio_sum += ratio;
            n_used += 1;
        }
    }
    if n_used == 0 {
        return Err("Sparse JXLMM r-hat estimation found no valid sampled markers".to_string());
    }
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 8, rhat_progress_total, rhat_progress_total)?;
    }
    let r_hat = ratio_sum / (n_used as f64);
    let rhat_result = PcgJxlmmRHatResult {
        r_hat,
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
                    rel_res: 0.0,
                };
                n_rhat
            ],
        },
    };
    let out = scan_with_py_and_rhat(
        &scan_prepared,
        &x_design,
        &null_model.py,
        &null_model.xtx_chol,
        &scan_sample_idx,
        gm,
        r_hat,
        threads,
        block_rows,
        scan_progress_callback.as_ref(),
        progress_every,
        9,
    )?;

    Ok((r_hat, out, null_info, rhat_result))
}

#[allow(clippy::too_many_arguments)]
fn estimate_rhat_and_scan_sparse_to_tsv(
    factor: SparseJxgrmCholesky,
    scan_prepared: JxlmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    scan_sample_idx: Vec<usize>,
    gm: PackedGeneticModel,
    threads: usize,
    block_rows: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    chrom: Vec<String>,
    pos: Vec<i64>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: String,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> Result<(f64, usize, PcgJxlmmNullModelInfo, PcgJxlmmRHatResult), String> {
    let n = y_vec.len();
    let p = x_design.len() / n;
    let stage1_cb = stage1_progress_callback.as_ref();
    let rhat_progress_total = n_rhat_progress_total(scan_prepared.n_rows(), rhat_markers);

    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 6, 0, 1)?;
    }
    let y_vinv_col = sparse_solve_rhs(&factor, &y_vec, 1)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 6, 1, 1)?;
    }

    let x_col = row_major_to_col_major_f64(&x_design, n, p)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 7, 0, p.max(1))?;
    }
    let x_vinv_col = sparse_solve_rhs(&factor, &x_col, p)?;
    let x_vinv_row = col_major_to_row_major_f64(&x_vinv_col, n, p)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 7, p.max(1), p.max(1))?;
    }

    let mut xt_v_inv_y = vec![0.0_f64; p];
    xt_vec_row_major(&x_design, n, p, &y_vinv_col, &mut xt_v_inv_y);

    let mut xt_v_inv_x = vec![0.0_f64; p * p];
    xt_mat_row_major(&x_design, &x_vinv_row, n, p, p, &mut xt_v_inv_x);
    let xt_v_inv_x_chol = spd_cholesky_with_jitter(&xt_v_inv_x, p, "Sparse JXLMM XtVinvX")?;
    let mut beta_hat = vec![0.0_f64; p];
    cholesky_solve_into(&xt_v_inv_x_chol, p, &xt_v_inv_y, &mut beta_hat);

    let mut py = y_vinv_col.clone();
    for i in 0..n {
        let vinvx_row = &x_vinv_row[i * p..(i + 1) * p];
        let adjust = vinvx_row
            .iter()
            .zip(beta_hat.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        py[i] -= adjust;
    }

    let mut xtx = vec![0.0_f64; p * p];
    xt_mat_row_major(&x_design, &x_design, n, p, p, &mut xtx);
    let xtx_chol = spd_cholesky_with_jitter(&xtx, p, "Sparse JXLMM XtX")?;

    let null_model = PcgJxlmmNullModel {
        n_samples: n,
        n_covariates: p,
        v_inv_x: x_vinv_row,
        beta_hat,
        py,
        xt_v_inv_x_chol,
        xtx_chol,
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
    let mut sampled_markers = vec![0.0_f64; n * n_rhat];
    let mut tmp_snp = vec![0.0_f64; n];
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
    let v_inv_s_col = sparse_solve_rhs(&factor, &sampled_markers, n_rhat)?;
    let mut ratio_sum = 0.0_f64;
    let mut n_used = 0usize;
    let mut v_inv_col = vec![0.0_f64; n];
    for col in 0..n_rhat {
        let snp_col = &sampled_markers[col * n..(col + 1) * n];
        for row in 0..n {
            v_inv_col[row] = v_inv_s_col[col * n + row];
        }
        let s_m_s = crate::pcg::pcg_jxlmm_s_m_s(&null_model, &x_design, snp_col)?;
        if !s_m_s.is_finite() || s_m_s <= JXLMM_TINY {
            continue;
        }
        let s_p_s = crate::pcg::pcg_jxlmm_s_p_s_exact(&null_model, &x_design, snp_col, &v_inv_col)?;
        if !s_p_s.is_finite() || s_p_s <= JXLMM_TINY {
            continue;
        }
        let ratio = s_p_s / s_m_s;
        if ratio.is_finite() && ratio > 0.0 {
            ratio_sum += ratio;
            n_used += 1;
        }
    }
    if n_used == 0 {
        return Err("Sparse JXLMM r-hat estimation found no valid sampled markers".to_string());
    }
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 8, rhat_progress_total, rhat_progress_total)?;
    }
    let r_hat = ratio_sum / (n_used as f64);
    let rhat_result = PcgJxlmmRHatResult {
        r_hat,
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
                    rel_res: 0.0,
                };
                n_rhat
            ],
        },
    };
    let written_rows = scan_to_tsv_with_py_and_rhat(
        &scan_prepared,
        &x_design,
        &null_model.py,
        &null_model.xtx_chol,
        &scan_sample_idx,
        gm,
        r_hat,
        threads,
        block_rows,
        chrom.as_slice(),
        pos.as_slice(),
        allele0.as_slice(),
        allele1.as_slice(),
        &out_tsv,
        scan_progress_callback.as_ref(),
        progress_every,
        9,
    )?;

    Ok((r_hat, written_rows, null_info, rhat_result))
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
    tol: f64,
    max_iter: usize,
    block_rows: usize,
    std_eps: f64,
    use_train_maf: bool,
    threads: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    rhat_tol: f64,
    force_sparse_jxlmm: bool,
    sparse_jxgrm_path: Option<String>,
) -> Result<(f64, Vec<f64>, PcgJxlmmNullModelInfo, PcgJxlmmRHatResult), String> {
    let n = y_vec.len();
    let p = x_design.len() / n;

    let stage1_cb = stage1_progress_callback.as_ref();
    if let Some(prefix) = operator_prepared.bed_prefix.as_deref() {
        let sparse_factor_sample_idx_ref =
            sparse_factor_sample_idx.as_deref().unwrap_or(operator_sample_idx.as_slice());
        let sparse_expected_n = if sparse_jxgrm_path.is_some() {
            sparse_factor_sample_idx_ref.len()
        } else {
            operator_prepared.n_samples_full
        };
        if let Some(factor) = sparse_jxlmm_load_factor(
            prefix,
            sparse_jxgrm_path.as_deref(),
            sparse_expected_n,
            sparse_factor_sample_idx_ref,
            sigma_g2,
            sigma_e2,
            force_sparse_jxlmm,
            stage1_cb,
        )? {
            return estimate_rhat_and_scan_sparse(
                factor,
                scan_prepared,
                x_design,
                y_vec,
                scan_sample_idx,
                gm,
                threads,
                block_rows,
                rhat_markers,
                rhat_seed,
                stage1_progress_callback,
                scan_progress_callback,
                progress_every,
            );
        }
    } else if force_sparse_jxlmm {
        return Err("Sparse JXLMM requested but operator input has no BED prefix.".to_string());
    }
    let prepare_every_rows = if progress_every == 0 {
        block_rows.max(256).max(1)
    } else {
        progress_every.max(1)
    };
    let rhat_rows = choose_rhat_rows(scan_prepared.n_rows(), rhat_markers, rhat_seed);
    let n_rhat = rhat_rows.len();
    let build_total = operator_prepared.n_rows().max(1);
    let jacobi_total = 1usize;
    let v_inv_y_total = max_iter.max(1);
    let v_inv_x_total = p.saturating_mul(max_iter.max(1)).max(1);
    let rhat_decode_total = n_rhat.max(1);
    let rhat_solve_total = n_rhat.saturating_mul(max_iter.max(1)).max(1);
    let rhat_total = rhat_decode_total.saturating_add(rhat_solve_total);
    let v_inv_x_last_sent = Cell::new(0usize);
    let rhat_last_sent = Cell::new(rhat_decode_total);
    let mut prepare_progress = |done: usize, total: usize| {
        let done_use = done.min(total.max(1)).min(build_total);
        emit_progress_callback(stage1_cb, 0, done_use, build_total)
    };
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 0, 0, build_total)?;
    }
    let mut k_grm = make_bed_operator(
        &operator_prepared,
        &operator_sample_idx,
        threads,
        block_rows,
        std_eps,
        use_train_maf,
        if stage1_cb.is_some() {
            Some(&mut prepare_progress)
        } else {
            None
        },
        prepare_every_rows,
    )?;
    emit_progress_callback(stage1_cb, 0, build_total, build_total)?;

    emit_progress_callback(stage1_cb, 1, 0, jacobi_total)?;
    let jacobi_inv_diag = k_grm.variance_jacobi_inv_diag(sigma_g2, sigma_e2)?;
    emit_progress_callback(stage1_cb, 1, jacobi_total, jacobi_total)?;

    emit_progress_callback(stage1_cb, 2, 0, v_inv_y_total)?;
    let (null_model, null_info) = pcg_build_jxlmm_null_model(
        &x_design,
        &y_vec,
        n,
        p,
        max_iter,
        tol,
        JXLMM_TINY,
        &mut k_grm,
        sigma_g2,
        sigma_e2,
        Some(jacobi_inv_diag.as_slice()),
        |iter, iter_max, _rel_res| {
            emit_progress_callback(stage1_cb, 2, iter.min(iter_max.max(1)), v_inv_y_total)
        },
        |col, iter, iter_max, _rel_res| {
            let cols_total = p.max(1);
            let iter_use = iter.clamp(1, iter_max.max(1));
            let done = iter_use
                .saturating_sub(1)
                .saturating_mul(cols_total)
                .saturating_add((col + 1).min(cols_total));
            emit_progress_callback_throttled(
                stage1_cb,
                3,
                done,
                v_inv_x_total,
                cols_total,
                &v_inv_x_last_sent,
            )
        },
    )?;

    emit_progress_callback(stage1_cb, 4, 0, rhat_total)?;
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
    let mut sampled_markers = vec![0.0_f64; n * n_rhat];
    let mut tmp_snp = vec![0.0_f64; n];
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
        emit_progress_callback(stage1_cb, 4, (col + 1).min(rhat_total), rhat_total)?;
    }
    let rhat_result = pcg_estimate_jxlmm_r_hat(
        &null_model,
        &x_design,
        &sampled_markers,
        n_rhat,
        max_iter,
        rhat_tol,
        JXLMM_TINY,
        &mut k_grm,
        sigma_g2,
        sigma_e2,
        Some(jacobi_inv_diag.as_slice()),
        |col, iter, iter_max, _rel_res| {
            let cols_total = n_rhat.max(1);
            let iter_use = iter.clamp(1, iter_max.max(1));
            let done = rhat_decode_total.saturating_add(
                iter_use
                    .saturating_sub(1)
                    .saturating_mul(cols_total)
                    .saturating_add((col + 1).min(cols_total)),
            );
            emit_progress_callback_throttled(
                stage1_cb,
                4,
                done,
                rhat_total,
                cols_total,
                &rhat_last_sent,
            )
        },
    )?;
    emit_progress_callback(stage1_cb, 4, rhat_total, rhat_total)?;
    let r_hat = rhat_result.r_hat;
    let out = scan_with_py_and_rhat(
        &scan_prepared,
        &x_design,
        &null_model.py,
        &null_model.xtx_chol,
        &scan_sample_idx,
        gm,
        r_hat,
        threads,
        block_rows,
        scan_progress_callback.as_ref(),
        progress_every,
        5,
    )?;

    Ok((r_hat, out, null_info, rhat_result))
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
    allele0: Vec<String>,
    allele1: Vec<String>,
    out_tsv: String,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    force_sparse_jxlmm: bool,
    sparse_jxgrm_path: Option<String>,
) -> Result<(f64, usize, PcgJxlmmNullModelInfo, PcgJxlmmRHatResult), String> {
    let stage1_cb = stage1_progress_callback.as_ref();
    if let Some(prefix) = operator_prepared.bed_prefix.as_deref() {
        let sparse_factor_sample_idx_ref =
            sparse_factor_sample_idx.as_deref().unwrap_or(operator_sample_idx.as_slice());
        let sparse_expected_n = if sparse_jxgrm_path.is_some() {
            sparse_factor_sample_idx_ref.len()
        } else {
            operator_prepared.n_samples_full
        };
        if let Some(factor) = sparse_jxlmm_load_factor(
            prefix,
            sparse_jxgrm_path.as_deref(),
            sparse_expected_n,
            sparse_factor_sample_idx_ref,
            sigma_g2,
            sigma_e2,
            force_sparse_jxlmm,
            stage1_cb,
        )? {
            return estimate_rhat_and_scan_sparse_to_tsv(
                factor,
                scan_prepared,
                x_design,
                y_vec,
                scan_sample_idx,
                gm,
                threads,
                block_rows,
                rhat_markers,
                rhat_seed,
                chrom,
                pos,
                allele0,
                allele1,
                out_tsv,
                stage1_progress_callback,
                scan_progress_callback,
                progress_every,
            );
        }
    }
    Err("JXLMM direct-to-TSV scan currently requires sparse JXLMM factorization path.".to_string())
}

#[pyfunction]
#[pyo3(signature = (
    jxgrm_path,
    sample_indices=None
))]
pub fn jxlmm_sparse_grm_diag_stats<'py>(
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
    rhat_markers=JXLMM_DEFAULT_RHAT_MARKERS,
    rhat_seed=JXLMM_DEFAULT_RHAT_SEED,
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
    force_sparse_jxlmm=false
))]
pub fn jxlmm_assoc_pcg_bed<'py>(
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
    force_sparse_jxlmm: bool,
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
    if rhat_markers == 0 {
        return Err(PyRuntimeError::new_err("rhat_markers must be > 0"));
    }
    emit_progress_callback(stage1_progress_callback.as_ref(), 0, 0, 1)
        .map_err(PyRuntimeError::new_err)?;
    let prepared = prepare_jxlmm_assoc_inputs(
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
                tol,
                max_iter,
                block_rows,
                std_eps,
                use_train_maf,
                threads,
                rhat_markers,
                rhat_seed,
                stage1_progress_callback,
                scan_progress_callback,
                progress_every,
                rhat_tol,
                force_sparse_jxlmm,
                sparse_jxgrm_path,
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
    rhat_markers=JXLMM_DEFAULT_RHAT_MARKERS,
    rhat_seed=JXLMM_DEFAULT_RHAT_SEED,
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
    force_sparse_jxlmm=false
))]
pub fn jxlmm_assoc_pcg_bed_to_tsv<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    sigma_g2: f64,
    sigma_e2: f64,
    chrom: Vec<String>,
    pos: Vec<i64>,
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
    force_sparse_jxlmm: bool,
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
    if rhat_markers == 0 {
        return Err(PyRuntimeError::new_err("rhat_markers must be > 0"));
    }
    let _ = (tol, max_iter, std_eps, use_train_maf, rhat_tol);

    emit_progress_callback(stage1_progress_callback.as_ref(), 0, 0, 1)
        .map_err(PyRuntimeError::new_err)?;
    let prepared = prepare_jxlmm_assoc_inputs(
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
    if chrom.len() != prepared.scan_prepared.n_rows()
        || pos.len() != prepared.scan_prepared.n_rows()
        || allele0.len() != prepared.scan_prepared.n_rows()
        || allele1.len() != prepared.scan_prepared.n_rows()
    {
        return Err(PyRuntimeError::new_err(format!(
            "JXLMM TSV metadata length mismatch: rows={}, chrom={}, pos={}, allele0={}, allele1={}",
            prepared.scan_prepared.n_rows(),
            chrom.len(),
            pos.len(),
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
                allele0,
                allele1,
                out_tsv,
                stage1_progress_callback,
                scan_progress_callback,
                progress_every,
                force_sparse_jxlmm,
                sparse_jxgrm_path,
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
pub fn jxlmm_scan_exact_packed<'py>(
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
            "n must be > p for JXLMM exact scan: n={n}, p={p}"
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
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    let m = out.len() / 3;
    let out_arr = numpy::ndarray::Array2::from_shape_vec((m, 3), out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, out_arr))
}
