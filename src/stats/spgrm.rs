//! Sparse genomic relationship matrix (sparse GRM) construction.
//!
//! Let `g_j` denote the genotype vector of retained marker `j` after missing values are
//! mean-imputed and the major-allele orientation is harmonized by `row_flip`, and let `p_j`
//! denote the corresponding minor-allele frequency.
//!
//! The dense relationship matrix underlying this module is defined as:
//!
//! - centered GRM (`method = 1`)
//!
//!   `x_j = g_j - 2 p_j`
//!
//!   `K = (sum_j x_j x_j') / (sum_j 2 p_j (1 - p_j))`
//!
//! - standardized GRM (`method = 2`)
//!
//!   `z_j = (g_j - 2 p_j) / sqrt(2 p_j (1 - p_j))`
//!
//!   `K = (sum_j z_j z_j') / m`
//!
//! where `m` is the number of retained markers after marker-level quality control.
//! The sparse GRM is obtained by retaining all diagonal entries and those upper-triangular
//! off-diagonal entries satisfying the user-specified threshold criterion. A negative
//! non-absolute threshold disables off-diagonal thresholding and keeps all off-diagonal
//! entries.
//!
//! Construction proceeds as follows.
//!
//! 1. Marker preprocessing.
//!    Input genotypes are filtered by missing rate and minor-allele frequency. For each retained
//!    marker, the module records allele orientation (`row_flip`) and allele frequency (`row_maf`).
//!
//! 2. Block partition.
//!    The sample axis is partitioned into sample tiles, and retained markers are partitioned into
//!    SNP blocks. Each sparse GRM task corresponds to one sample-tile pair.
//!
//! 3. Block decoding and crossproduct accumulation.
//!    For each SNP block, the corresponding genotype rows are decoded into dense additive blocks.
//!    For a diagonal sample-tile pair, the local contribution is accumulated with symmetric
//!    rank-k updates; for an off-diagonal tile pair, the local contribution is accumulated with
//!    general matrix multiplication. Summing across all SNP blocks yields the unscaled tile-level
//!    crossproduct.
//!
//! 4. Scaling and thresholding.
//!    After all SNP blocks have been processed for a given sample-tile pair, the accumulated
//!    crossproduct is scaled by the denominator of the selected GRM definition. Diagonal entries
//!    are retained unconditionally, whereas off-diagonal entries are kept only if they satisfy the
//!    sparse threshold criterion.
//!
//! 5. Sparse output.
//!    Retained entries are emitted in upper-triangular form and written to CSC-based `.spgrm`
//!    output. When necessary, intermediate sparse chunks are spilled and merged into the final
//!    sparse GRM file.
//!
//! Two data-access routes are implemented.
//!
//! - Packed route:
//!   the filtered packed BED payload and marker metadata are already resident in memory.
//!
//! - Stream/mmap route:
//!   filtered marker metadata are prepared first, after which BED payload blocks are fetched
//!   through full mmap or windowed mmap and reused across sample-tile tasks within a batch.
//!
use memmap2::{Mmap, MmapMut, MmapOptions};
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering as AtomicOrdering},
    Arc, Mutex,
};
use std::time::Instant;

use crate::bedmath::{
    adaptive_grm_block_rows, decode_standardized_packed_block_rows_f32_with_plan,
    is_identity_indices, packed_byte_lut, SubsetDecodePlan,
};
use crate::blas::{
    cblas_sgemm_dispatch, cblas_ssyrk_dispatch, BlasThreadGuard, CblasInt, CBLAS_COL_MAJOR,
    CBLAS_LOWER, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_UPPER,
};
use crate::gfcore::{block_rows_from_memory_target_mb, parse_positive_env_f64, read_fam};
use crate::gfreader::{
    prepare_bed_logic_meta_owned_for_stats_samples_sparse_windowed,
    prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window,
};
use crate::gload::WindowedBedMatrix;
use crate::grm::decode_grm_block;
use crate::pipeline::run_double_buffer;
use crate::stats_common::{
    check_ctrlc, get_cached_pool, map_err_string_to_py, parse_index_vec_i64,
};

const SPGRM_DEFAULT_SAMPLE_BLOCK: usize = 256;
const SPGRM_SMALL_N_FULL_SAMPLE_BLOCK_MAX: usize = 2048;
const SPGRM_SMALL_N_FULL_SAMPLE_ROW_BLOCK_FLOOR: usize = 8192;
const SPGRM_DEFAULT_SNP_BLOCK_ROWS: usize = 4096;
const SPGRM_DEFAULT_WRITE_BUF_BYTES: usize = 16 * 1024 * 1024;
const SPGRM_DEFAULT_SPILL_NNZ: usize = 10_000_000;
const SPGRM_DEFAULT_BATCH_ACCUM_TILE_FACTOR: usize = 4;
const SPGRM_DEFAULT_BATCH_DECODED_STRIPE_FLOOR: usize = 8;
const SPGRM_DEFAULT_BATCH_DECODED_STRIPE_THREADS_FACTOR: usize = 2;
const SPGRM_DEFAULT_BATCH_MIN_ACCUM_BYTES: usize = 8 * 1024 * 1024;
const SPGRM_JXGRM_VALUE_ALIGN_BYTES: usize = std::mem::size_of::<f64>();
const SPGRM_DEFAULT_STREAMING_MERGE_MIN_BYTES: usize = 2 * 1024 * 1024 * 1024;
const SPGRM_DECODE_STRIPE_ESTIMATE: usize = 2;
const SPGRM_GCTA_IMPORT_SUFFIX: &str = ".gcta.spgrm";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpgrmNpyFloatDtype {
    F32Le,
    F64Le,
}

#[derive(Default)]
struct SpgrmStageTiming {
    decode_ns: AtomicU64,
    gemm_ns: AtomicU64,
    threshold_ns: AtomicU64,
    spill_ns: AtomicU64,
}

impl SpgrmStageTiming {
    #[inline]
    fn add_decode_ns(&self, ns: u64) {
        self.decode_ns.fetch_add(ns, AtomicOrdering::Relaxed);
    }

    #[inline]
    fn add_gemm_ns(&self, ns: u64) {
        self.gemm_ns.fetch_add(ns, AtomicOrdering::Relaxed);
    }

    #[inline]
    fn add_threshold_ns(&self, ns: u64) {
        self.threshold_ns.fetch_add(ns, AtomicOrdering::Relaxed);
    }

    #[inline]
    fn add_spill_ns(&self, ns: u64) {
        self.spill_ns.fetch_add(ns, AtomicOrdering::Relaxed);
    }

    #[inline]
    fn decode_ns(&self) -> u64 {
        self.decode_ns.load(AtomicOrdering::Relaxed)
    }

    #[inline]
    fn gemm_ns(&self) -> u64 {
        self.gemm_ns.load(AtomicOrdering::Relaxed)
    }

    #[inline]
    fn threshold_ns(&self) -> u64 {
        self.threshold_ns.load(AtomicOrdering::Relaxed)
    }

    #[inline]
    fn spill_ns(&self) -> u64 {
        self.spill_ns.load(AtomicOrdering::Relaxed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SpgrmEntry {
    col: u32,
    row: u32,
    value: f64,
}

#[derive(Debug)]
struct SpgrmWorkerScratch {
    row_block_f32: Vec<f32>,
    col_block_f32: Vec<f32>,
    accum_f64: Vec<f64>,
    accum_tmp_f32: Vec<f32>,
    dummy_row_varsum: Vec<f64>,
    dummy_col_varsum: Vec<f64>,
}

impl SpgrmWorkerScratch {
    fn new(row_step: usize, sample_step: usize) -> Self {
        Self {
            row_block_f32: vec![0.0_f32; row_step.saturating_mul(sample_step)],
            col_block_f32: vec![0.0_f32; row_step.saturating_mul(sample_step)],
            accum_f64: vec![0.0_f64; sample_step.saturating_mul(sample_step)],
            accum_tmp_f32: vec![0.0_f32; sample_step.saturating_mul(sample_step)],
            dummy_row_varsum: vec![0.0_f64; row_step],
            dummy_col_varsum: vec![0.0_f64; row_step],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpgrmTask {
    col_start: usize,
    col_end: usize,
    row_start: usize,
    row_end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpgrmTaskBatchSpan {
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpgrmTaskBatchLimits {
    max_decoded_bytes: usize,
    max_accum_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpgrmStripe {
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct SpgrmStreamStripeScratch {
    stripe: SpgrmStripe,
    full_sample_fast: bool,
    subset_plan: Option<SubsetDecodePlan>,
    decoded: Vec<f32>,
}

impl Clone for SpgrmStreamStripeScratch {
    fn clone(&self) -> Self {
        Self {
            stripe: self.stripe,
            full_sample_fast: self.full_sample_fast,
            subset_plan: self.subset_plan.clone(),
            decoded: vec![0.0_f32; self.decoded.len()],
        }
    }
}

impl SpgrmStreamStripeScratch {
    fn new(
        stripe: SpgrmStripe,
        row_step: usize,
        sample_idx: &[usize],
        n_samples_full: usize,
    ) -> Self {
        let stripe_idx = &sample_idx[stripe.start..stripe.end];
        let full_sample_fast = is_identity_indices(stripe_idx, n_samples_full);
        let subset_plan = if full_sample_fast {
            None
        } else {
            Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
                stripe_idx,
                n_samples_full,
            ))
        };
        Self {
            stripe,
            full_sample_fast,
            subset_plan,
            decoded: vec![0.0_f32; row_step.saturating_mul(stripe.end - stripe.start)],
        }
    }

    #[inline]
    fn width(&self) -> usize {
        self.stripe.end - self.stripe.start
    }
}

#[derive(Clone, Copy, Debug)]
struct SpgrmMemoryPlan {
    sample_step: usize,
    row_step: usize,
    max_decoded_bytes: usize,
}

#[derive(Clone, Debug)]
struct SpgrmDecodedBatchBuffer {
    stripe_scratch: Vec<SpgrmStreamStripeScratch>,
    row_start: usize,
    cur_rows: usize,
}

#[inline]
fn spgrm_resolve_packed_memory_plan(
    m: usize,
    n_use: usize,
    n_samples_full: usize,
    requested_row_step: usize,
    requested_sample_step: usize,
    threads: usize,
) -> SpgrmMemoryPlan {
    let decode_target_mb = spgrm_decode_target_mb();
    let sample_step = spgrm_auto_sample_block_with_budget(
        n_use,
        m,
        requested_sample_step,
        decode_target_mb,
        1usize,
    );
    let decoded_sample_width = sample_step.saturating_mul(SPGRM_DECODE_STRIPE_ESTIMATE);
    let row_step = spgrm_row_block_rows(
        requested_row_step,
        m,
        decoded_sample_width,
        n_samples_full,
        threads,
        decode_target_mb,
        1usize,
    );
    let max_decoded_bytes = spgrm_batch_max_decoded_bytes(
        row_step,
        decoded_sample_width,
        threads,
        decode_target_mb,
        1usize,
    );
    SpgrmMemoryPlan {
        sample_step,
        row_step,
        max_decoded_bytes,
    }
}

#[inline]
fn spgrm_resolve_stream_memory_plan(
    m: usize,
    n_use: usize,
    n_samples_full: usize,
    requested_row_step: usize,
    requested_sample_step: usize,
    threads: usize,
    decode_buffers: usize,
) -> SpgrmMemoryPlan {
    let decode_target_mb = spgrm_decode_target_mb();
    let sample_step = spgrm_auto_stream_sample_block_with_budget(
        n_use,
        m,
        requested_sample_step,
        decode_target_mb,
        decode_buffers,
    );
    let decoded_sample_width = n_use.max(1);
    let row_step = spgrm_row_block_rows(
        requested_row_step,
        m,
        decoded_sample_width,
        n_samples_full,
        threads,
        decode_target_mb,
        decode_buffers,
    );
    let max_decoded_bytes = spgrm_batch_max_decoded_bytes(
        row_step,
        decoded_sample_width,
        threads,
        decode_target_mb,
        decode_buffers,
    );
    SpgrmMemoryPlan {
        sample_step,
        row_step,
        max_decoded_bytes,
    }
}

#[derive(Debug)]
struct SpgrmBatchTaskAccum {
    task: SpgrmTask,
    row_stripe_idx: usize,
    col_stripe_idx: usize,
    row_offset_in_stripe: usize,
    col_offset_in_stripe: usize,
    accum_row_major: bool,
    accum: Vec<f64>,
    accum_tmp_f32: Vec<f32>,
}

#[derive(Debug)]
struct SpgrmChunkReader {
    reader: BufReader<File>,
    remaining: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SpgrmChunkHead {
    chunk_idx: usize,
    col: u32,
    row: u32,
    value: f64,
}

impl Eq for SpgrmChunkHead {}

impl Ord for SpgrmChunkHead {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .col
            .cmp(&self.col)
            .then_with(|| other.row.cmp(&self.row))
            .then_with(|| other.chunk_idx.cmp(&self.chunk_idx))
    }
}

impl PartialOrd for SpgrmChunkHead {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug, PartialEq)]
pub struct SparseGrmCoo {
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
    pub values: Vec<f64>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug, PartialEq)]
pub struct SparseGrmCsc {
    pub n_samples: usize,
    pub nnz: usize,
    // CSC column pointer, length = n_samples + 1.
    pub col_ptr: Vec<u64>,
    pub row_indices: Vec<u32>,
    pub values: Vec<f64>,
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
fn normalize_spgrm_path(prefix: &str) -> String {
    let trimmed = prefix.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.to_ascii_lowercase().ends_with(".spgrm") {
        return trimmed.to_string();
    }
    // Backward compat: if an old .jxgrm file exists, use it.
    if trimmed.to_ascii_lowercase().ends_with(".jxgrm") {
        return trimmed.to_string();
    }
    let path_spgrm = format!("{trimmed}.spgrm");
    let path_jxgrm = format!("{trimmed}.jxgrm");
    if std::path::Path::new(&path_jxgrm).exists() && !std::path::Path::new(&path_spgrm).exists() {
        return path_jxgrm;
    }
    path_spgrm
}

#[inline]
fn strip_ascii_suffix_case_insensitive(value: &str, suffix: &str) -> Option<String> {
    if value.len() < suffix.len() {
        return None;
    }
    let split = value.len() - suffix.len();
    if value[split..].eq_ignore_ascii_case(suffix) {
        Some(value[..split].to_string())
    } else {
        None
    }
}

#[derive(Clone, Debug)]
struct GctaSparseImportPaths {
    input_sp_path: PathBuf,
    input_id_path: PathBuf,
    output_spgrm_path: PathBuf,
}

#[inline]
fn resolve_gcta_sparse_import_paths(path: &str) -> Option<GctaSparseImportPaths> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    let prefix = strip_ascii_suffix_case_insensitive(trimmed, ".grm.sp")
        .unwrap_or_else(|| trimmed.to_string());
    let input_sp_path = if trimmed.len() >= ".grm.sp".len()
        && trimmed[trimmed.len() - ".grm.sp".len()..].eq_ignore_ascii_case(".grm.sp")
    {
        PathBuf::from(trimmed)
    } else {
        PathBuf::from(format!("{prefix}.grm.sp"))
    };
    let input_id_path = PathBuf::from(format!("{prefix}.grm.id"));
    if !input_sp_path.is_file() || !input_id_path.is_file() {
        return None;
    }
    let output_spgrm_path = PathBuf::from(format!("{prefix}{SPGRM_GCTA_IMPORT_SUFFIX}"));
    Some(GctaSparseImportPaths {
        input_sp_path,
        input_id_path,
        output_spgrm_path,
    })
}

#[inline]
fn sparse_file_modified_ns(path: &Path) -> Option<u128> {
    fs::metadata(path)
        .ok()
        .and_then(|meta| meta.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
}

#[inline]
fn gcta_import_output_is_fresh(paths: &GctaSparseImportPaths) -> bool {
    let out_path = paths.output_spgrm_path.as_path();
    let out_id = PathBuf::from(format!("{}.id", out_path.display()));
    if !out_path.is_file() || !out_id.is_file() {
        return false;
    }
    let out_ns = sparse_file_modified_ns(out_path);
    let out_id_ns = sparse_file_modified_ns(out_id.as_path());
    let input_sp_ns = sparse_file_modified_ns(paths.input_sp_path.as_path());
    let input_id_ns = sparse_file_modified_ns(paths.input_id_path.as_path());
    match (out_ns, out_id_ns, input_sp_ns, input_id_ns) {
        (Some(out_ns), Some(out_id_ns), Some(input_sp_ns), Some(input_id_ns)) => {
            out_ns >= input_sp_ns && out_ns >= input_id_ns && out_id_ns >= input_id_ns
        }
        _ => false,
    }
}

fn gcta_sparse_id_count(id_path: &Path) -> Result<usize, String> {
    let file = File::open(id_path).map_err(|e| {
        format!(
            "failed to open GCTA sparse GRM id file {}: {e}",
            id_path.display()
        )
    })?;
    let reader = BufReader::new(file);
    let mut n = 0usize;
    for line in reader.lines() {
        let line = line.map_err(|e| {
            format!(
                "failed to read GCTA sparse GRM id file {}: {e}",
                id_path.display()
            )
        })?;
        if !line.trim().is_empty() {
            n = n.saturating_add(1);
        }
    }
    if n == 0 {
        return Err(format!(
            "empty GCTA sparse GRM id file: {}",
            id_path.display()
        ));
    }
    Ok(n)
}

#[inline]
fn parse_gcta_sparse_triplet(
    sp_path: &Path,
    line_no: usize,
    line: &str,
    n_samples: usize,
) -> Result<(usize, usize, f64), String> {
    let mut fields = line.split_whitespace();
    let row = fields
        .next()
        .ok_or_else(|| {
            format!(
                "{}:{line_no}: expected 3 columns in GCTA sparse GRM",
                sp_path.display()
            )
        })?
        .parse::<usize>()
        .map_err(|e| format!("{}:{line_no}: invalid row index: {e}", sp_path.display()))?;
    let col = fields
        .next()
        .ok_or_else(|| {
            format!(
                "{}:{line_no}: expected 3 columns in GCTA sparse GRM",
                sp_path.display()
            )
        })?
        .parse::<usize>()
        .map_err(|e| format!("{}:{line_no}: invalid column index: {e}", sp_path.display()))?;
    let value = fields
        .next()
        .ok_or_else(|| {
            format!(
                "{}:{line_no}: expected 3 columns in GCTA sparse GRM",
                sp_path.display()
            )
        })?
        .parse::<f64>()
        .map_err(|e| format!("{}:{line_no}: invalid value: {e}", sp_path.display()))?;
    if fields.next().is_some() {
        return Err(format!(
            "{}:{line_no}: expected 3 columns in GCTA sparse GRM",
            sp_path.display()
        ));
    }
    if !(col <= row && row < n_samples) {
        return Err(format!(
            "{}:{line_no}: invalid lower-triangle index row={row} col={col} for n_samples={n_samples}",
            sp_path.display()
        ));
    }
    if !value.is_finite() {
        return Err(format!(
            "{}:{line_no}: non-finite value in GCTA sparse GRM",
            sp_path.display()
        ));
    }
    Ok((row, col, value))
}

fn convert_gcta_sparse_to_spgrm(paths: &GctaSparseImportPaths) -> Result<String, String> {
    let n_samples = gcta_sparse_id_count(paths.input_id_path.as_path())?;
    let mut col_counts = vec![0u64; n_samples + 1];
    let mut nnz = 0usize;
    {
        let file = File::open(paths.input_sp_path.as_path()).map_err(|e| {
            format!(
                "failed to open GCTA sparse GRM {}: {e}",
                paths.input_sp_path.display()
            )
        })?;
        let reader = BufReader::new(file);
        for (line_no0, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                format!(
                    "failed to read GCTA sparse GRM {}: {e}",
                    paths.input_sp_path.display()
                )
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let (_row, col, _value) = parse_gcta_sparse_triplet(
                paths.input_sp_path.as_path(),
                line_no0 + 1,
                trimmed,
                n_samples,
            )?;
            col_counts[col + 1] = col_counts[col + 1].saturating_add(1);
            nnz = nnz.saturating_add(1);
        }
    }
    if nnz == 0 {
        return Err(format!(
            "empty GCTA sparse GRM file: {}",
            paths.input_sp_path.display()
        ));
    }
    for idx in 0..n_samples {
        col_counts[idx + 1] = col_counts[idx + 1].saturating_add(col_counts[idx]);
    }
    let mut row_indices = vec![0u32; nnz];
    let mut values = vec![0.0_f64; nnz];
    let mut next_ptr = col_counts[..n_samples]
        .iter()
        .map(|&v| v as usize)
        .collect::<Vec<_>>();
    let mut prev_row_by_col = vec![None::<usize>; n_samples];
    {
        let file = File::open(paths.input_sp_path.as_path()).map_err(|e| {
            format!(
                "failed to reopen GCTA sparse GRM {}: {e}",
                paths.input_sp_path.display()
            )
        })?;
        let reader = BufReader::new(file);
        for (line_no0, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                format!(
                    "failed to read GCTA sparse GRM {}: {e}",
                    paths.input_sp_path.display()
                )
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let (row, col, value) = parse_gcta_sparse_triplet(
                paths.input_sp_path.as_path(),
                line_no0 + 1,
                trimmed,
                n_samples,
            )?;
            if let Some(prev_row) = prev_row_by_col[col] {
                if row <= prev_row {
                    return Err(format!(
                        "{}:{}: row indices within column {col} must be strictly increasing; got prev_row={prev_row}, row={row}",
                        paths.input_sp_path.display(),
                        line_no0 + 1
                    ));
                }
            }
            prev_row_by_col[col] = Some(row);
            let dst = next_ptr[col];
            row_indices[dst] = row as u32;
            values[dst] = value;
            next_ptr[col] = dst.saturating_add(1);
        }
    }
    let csc = SparseGrmCsc {
        n_samples,
        nnz,
        col_ptr: col_counts,
        row_indices,
        values,
    };
    let out_path = paths.output_spgrm_path.to_string_lossy().to_string();
    let tmp_path = format!(
        "{out_path}.tmp.{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    write_sparse_grm_csc(&tmp_path, &csc)?;
    let tmp_id = format!("{tmp_path}.id");
    fs::copy(paths.input_id_path.as_path(), &tmp_id).map_err(|e| {
        format!(
            "failed to copy GCTA sparse GRM id file {} -> {}: {e}",
            paths.input_id_path.display(),
            tmp_id
        )
    })?;
    let meta_path = format!("{tmp_path}.meta.json");
    let meta = serde_json::json!({
        "source": "gcta_sparse_import",
        "input_grm_sp": paths.input_sp_path,
        "input_grm_id": paths.input_id_path,
        "n_samples": n_samples,
        "nnz": nnz,
    });
    let meta_file = File::create(&meta_path)
        .map_err(|e| format!("failed to create sparse GRM meta file {meta_path}: {e}"))?;
    serde_json::to_writer(meta_file, &meta)
        .map_err(|e| format!("failed to write sparse GRM meta file {meta_path}: {e}"))?;
    fs::rename(&tmp_path, &out_path)
        .map_err(|e| format!("failed to finalize imported sparse GRM {out_path}: {e}"))?;
    fs::rename(&tmp_id, format!("{out_path}.id")).map_err(|e| {
        format!(
            "failed to finalize imported sparse GRM id {}.id: {e}",
            out_path
        )
    })?;
    fs::rename(&meta_path, format!("{out_path}.meta.json")).map_err(|e| {
        format!(
            "failed to finalize imported sparse GRM meta {}.meta.json: {e}",
            out_path
        )
    })?;
    Ok(out_path)
}

pub(crate) fn resolve_sparse_grm_input_path(path: &str) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("Sparse GRM path must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.ends_with(".spgrm") || lower.ends_with(".jxgrm") {
        return Ok(trimmed.to_string());
    }
    let candidate_spgrm = format!("{trimmed}.spgrm");
    let candidate_jxgrm = format!("{trimmed}.jxgrm");
    if Path::new(&candidate_spgrm).exists() || Path::new(&candidate_jxgrm).exists() {
        return Ok(normalize_spgrm_path(trimmed));
    }
    if let Some(paths) = resolve_gcta_sparse_import_paths(trimmed) {
        if gcta_import_output_is_fresh(&paths) {
            return Ok(paths.output_spgrm_path.to_string_lossy().to_string());
        }
        return convert_gcta_sparse_to_spgrm(&paths);
    }
    Ok(trimmed.to_string())
}

#[inline]
fn spgrm_align_up_to(offset: usize, align: usize) -> Option<usize> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    let mask = align - 1;
    offset.checked_add(mask).map(|x| x & !mask)
}

#[inline]
fn spgrm_jxgrm_values_padding_bytes(nnz: usize) -> Result<usize, String> {
    let row_bytes = nnz
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "Sparse GRM row payload byte size overflow".to_string())?;
    let aligned = spgrm_align_up_to(row_bytes, SPGRM_JXGRM_VALUE_ALIGN_BYTES)
        .ok_or_else(|| "Sparse GRM alignment overflow".to_string())?;
    aligned
        .checked_sub(row_bytes)
        .ok_or_else(|| "Sparse GRM padding underflow".to_string())
}

#[inline]
fn spgrm_sample_block(n: usize, requested: usize) -> usize {
    let base = if requested == 0 {
        if n <= SPGRM_SMALL_N_FULL_SAMPLE_BLOCK_MAX {
            n.max(1)
        } else {
            SPGRM_DEFAULT_SAMPLE_BLOCK
        }
    } else {
        requested
    };
    base.max(1).min(n.max(1))
}

#[inline]
fn spgrm_decode_target_mb() -> Option<f64> {
    parse_positive_env_f64(&[
        "JX_SPGRM_DECODE_TARGET_MB",
        "JANUSX_SPGRM_DECODE_TARGET_MB",
        "JX_BED_BLOCK_TARGET_MB",
        "JANUSX_BED_BLOCK_TARGET_MB",
    ])
}

#[inline]
fn spgrm_stream_double_buffer_requested(threads: usize) -> bool {
    if let Ok(raw) = std::env::var("JX_SPGRM_STREAM_OVERLAP") {
        let t = raw.trim().to_ascii_lowercase();
        return !matches!(t.as_str(), "0" | "false" | "no" | "off");
    }
    if let Ok(raw) = std::env::var("JANUSX_SPGRM_STREAM_OVERLAP") {
        let t = raw.trim().to_ascii_lowercase();
        return !matches!(t.as_str(), "0" | "false" | "no" | "off");
    }
    threads > 1
}

#[inline]
fn spgrm_auto_sample_block_with_budget(
    n_use: usize,
    m: usize,
    requested: usize,
    decode_target_mb: Option<f64>,
    decode_buffers: usize,
) -> usize {
    if requested > 0 {
        return requested.max(1).min(n_use.max(1));
    }
    let heuristic = spgrm_sample_block(n_use, 0usize);
    if let Some(target_mb) = decode_target_mb {
        let min_row_step = SPGRM_DEFAULT_SNP_BLOCK_ROWS.min(m.max(1)).max(1);
        let per_sample_decode_bytes = min_row_step
            .saturating_mul(SPGRM_DECODE_STRIPE_ESTIMATE)
            .saturating_mul(std::mem::size_of::<f32>())
            .max(1);
        let by_budget = block_rows_from_memory_target_mb(
            target_mb,
            per_sample_decode_bytes,
            n_use.max(1),
            1,
            decode_buffers.max(1),
            0,
        );
        return heuristic.min(by_budget.max(1).min(n_use.max(1)));
    }
    heuristic
}

#[inline]
fn spgrm_auto_stream_sample_block_with_budget(
    n_use: usize,
    m: usize,
    requested: usize,
    decode_target_mb: Option<f64>,
    decode_buffers: usize,
) -> usize {
    if requested > 0 {
        return requested.max(1).min(n_use.max(1));
    }
    if let Some(target_mb) = decode_target_mb {
        let min_row_step = SPGRM_DEFAULT_SNP_BLOCK_ROWS.min(m.max(1)).max(1);
        let per_sample_decode_bytes = min_row_step
            .saturating_mul(SPGRM_DECODE_STRIPE_ESTIMATE)
            .saturating_mul(std::mem::size_of::<f32>())
            .max(1);
        let by_budget = block_rows_from_memory_target_mb(
            target_mb,
            per_sample_decode_bytes,
            n_use.max(1),
            1,
            decode_buffers.max(1),
            0,
        );
        // Stream batches keep decoded stripes in a shared batch buffer instead of
        // per-worker scratch, so they can safely spend the full decode budget on
        // wider sample tiles. This avoids over-fragmenting the sample axis into
        // many column groups and repeatedly re-decoding the same SNP blocks.
        return by_budget.max(1).min(n_use.max(1));
    }
    spgrm_sample_block(n_use, 0usize)
}

#[inline]
fn spgrm_stage_timing_enabled() -> bool {
    std::env::var("JANUSX_SPGRM_TIMING")
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no" || s == "off")
        })
        .unwrap_or(false)
}

#[inline]
fn spgrm_elapsed_ns(t0: Instant) -> u64 {
    t0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

#[inline]
fn spgrm_stage_secs(ns: u64) -> f64 {
    (ns as f64) * 1e-9_f64
}

fn spgrm_emit_timing_summary(
    mode: &str,
    n_use: usize,
    m: usize,
    sample_step: usize,
    row_step: usize,
    timing: &SpgrmStageTiming,
) {
    eprintln!(
        "spgrm timing [{mode}] n_use={n_use} m={m} sample_step={sample_step} row_step={row_step} decode_sum={:.3}s gemm_sum={:.3}s threshold_sum={:.3}s spill={:.3}s",
        spgrm_stage_secs(timing.decode_ns()),
        spgrm_stage_secs(timing.gemm_ns()),
        spgrm_stage_secs(timing.threshold_ns()),
        spgrm_stage_secs(timing.spill_ns()),
    );
}

#[inline]
fn spgrm_parse_npy_shape_local(header: &str) -> Result<(usize, usize), String> {
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
fn spgrm_parse_npy_descr_local(header: &str) -> Result<SpgrmNpyFloatDtype, String> {
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
        "<f4" | "|f4" | "=f4" => Ok(SpgrmNpyFloatDtype::F32Le),
        "<f8" | "|f8" | "=f8" => Ok(SpgrmNpyFloatDtype::F64Le),
        d if d.starts_with(">f4") || d.starts_with(">f8") => {
            Err("big-endian NPY float dtype is not supported".to_string())
        }
        _ => Err(format!(
            "NPY dtype is not supported for sparse GRM extraction: {descr}"
        )),
    }
}

#[inline]
fn spgrm_parse_npy_header_for_float_matrix(
    bytes: &[u8],
) -> Result<(usize, usize, usize, SpgrmNpyFloatDtype), String> {
    if bytes.len() < 10 {
        return Err("NPY file too small".to_string());
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("invalid NPY magic".to_string());
    }

    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("NPY file too small for v2/v3 header".to_string());
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            )
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
    let (rows, cols) = spgrm_parse_npy_shape_local(header)?;
    let dtype = spgrm_parse_npy_descr_local(header)?;
    Ok((rows, cols, header_end, dtype))
}

#[inline]
fn spgrm_npy_read_value(
    payload: &[u8],
    dtype: SpgrmNpyFloatDtype,
    elem_idx: usize,
) -> Result<f64, String> {
    match dtype {
        SpgrmNpyFloatDtype::F32Le => {
            let byte_idx = elem_idx
                .checked_mul(4)
                .ok_or_else(|| "NPY element byte offset overflow".to_string())?;
            let end = byte_idx
                .checked_add(4)
                .ok_or_else(|| "NPY element byte end overflow".to_string())?;
            if end > payload.len() {
                return Err("NPY payload truncated while reading f32 element".to_string());
            }
            Ok(f32::from_le_bytes([
                payload[byte_idx],
                payload[byte_idx + 1],
                payload[byte_idx + 2],
                payload[byte_idx + 3],
            ]) as f64)
        }
        SpgrmNpyFloatDtype::F64Le => {
            let byte_idx = elem_idx
                .checked_mul(8)
                .ok_or_else(|| "NPY element byte offset overflow".to_string())?;
            let end = byte_idx
                .checked_add(8)
                .ok_or_else(|| "NPY element byte end overflow".to_string())?;
            if end > payload.len() {
                return Err("NPY payload truncated while reading f64 element".to_string());
            }
            Ok(f64::from_le_bytes([
                payload[byte_idx],
                payload[byte_idx + 1],
                payload[byte_idx + 2],
                payload[byte_idx + 3],
                payload[byte_idx + 4],
                payload[byte_idx + 5],
                payload[byte_idx + 6],
                payload[byte_idx + 7],
            ]))
        }
    }
}

struct SpgrmNpyF32Writer {
    _file: File,
    mmap: MmapMut,
    data_offset: usize,
    len_f32: usize,
}

impl SpgrmNpyF32Writer {
    fn create(path: &str, rows: usize, cols: usize) -> Result<Self, String> {
        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let header_len_u16 = u16::try_from(header.len())
            .map_err(|_| "npy header too long for v1.0 format".to_string())?;
        let data_offset = 10usize
            .checked_add(header.len())
            .ok_or_else(|| "npy payload offset overflow".to_string())?;
        let len_f32 = rows
            .checked_mul(cols)
            .ok_or_else(|| "npy matrix element count overflow".to_string())?;
        let payload_bytes = len_f32
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "npy payload byte size overflow".to_string())?;
        let total_bytes = data_offset
            .checked_add(payload_bytes)
            .ok_or_else(|| "npy total byte size overflow".to_string())?;

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("create npy file failed: {e}"))?;
        file.set_len(total_bytes as u64)
            .map_err(|e| format!("resize npy file failed: {e}"))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| format!("seek npy file failed: {e}"))?;
        file.write_all(b"\x93NUMPY")
            .map_err(|e| format!("write npy magic failed: {e}"))?;
        file.write_all(&[1_u8, 0_u8])
            .map_err(|e| format!("write npy version failed: {e}"))?;
        file.write_all(&header_len_u16.to_le_bytes())
            .map_err(|e| format!("write npy header length failed: {e}"))?;
        file.write_all(header.as_bytes())
            .map_err(|e| format!("write npy header failed: {e}"))?;
        file.flush()
            .map_err(|e| format!("flush npy header failed: {e}"))?;

        let mmap = unsafe { MmapOptions::new().len(total_bytes).map_mut(&file) }
            .map_err(|e| format!("map npy file failed: {e}"))?;
        Ok(Self {
            _file: file,
            mmap,
            data_offset,
            len_f32,
        })
    }

    fn payload_mut(&mut self) -> Result<&mut [f32], String> {
        if self.data_offset % std::mem::align_of::<f32>() != 0 {
            return Err("npy payload offset is not f32-aligned".to_string());
        }
        let payload = &mut self.mmap[self.data_offset..];
        let expected_bytes = self
            .len_f32
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "npy payload byte size overflow".to_string())?;
        if payload.len() != expected_bytes {
            return Err(format!(
                "npy payload size mismatch: got {}, expected {}",
                payload.len(),
                expected_bytes
            ));
        }
        #[cfg(target_endian = "little")]
        unsafe {
            Ok(std::slice::from_raw_parts_mut(
                payload.as_mut_ptr() as *mut f32,
                self.len_f32,
            ))
        }
        #[cfg(not(target_endian = "little"))]
        {
            Err("direct NPY writer requires little-endian host".to_string())
        }
    }

    fn flush(&mut self) -> Result<(), String> {
        self.mmap
            .flush()
            .map_err(|e| format!("flush npy mmap failed: {e}"))
    }
}

#[inline]
fn spgrm_row_block_rows(
    requested: usize,
    m: usize,
    decoded_sample_width: usize,
    n_samples_full: usize,
    threads: usize,
    decode_target_mb: Option<f64>,
    decode_buffers: usize,
) -> usize {
    if requested > 0 {
        return requested.max(1);
    }
    if let Some(target_mb) = decode_target_mb {
        let row_bytes = decoded_sample_width
            .max(1)
            .saturating_mul(std::mem::size_of::<f32>())
            .max(1);
        return block_rows_from_memory_target_mb(
            target_mb,
            row_bytes,
            m.max(1),
            1,
            decode_buffers.max(1),
            0,
        )
        .max(1);
    }
    let mut out = adaptive_grm_block_rows(
        SPGRM_DEFAULT_SNP_BLOCK_ROWS,
        m.max(1),
        decoded_sample_width.max(1),
        n_samples_full,
        threads,
    )
    .max(1);
    if decoded_sample_width >= n_samples_full
        && n_samples_full <= SPGRM_SMALL_N_FULL_SAMPLE_BLOCK_MAX
    {
        out = out.max(SPGRM_SMALL_N_FULL_SAMPLE_ROW_BLOCK_FLOOR.min(m.max(1)));
    }
    out
}

#[inline]
fn spgrm_progress_notify(
    progress_callback: Option<&Py<PyAny>>,
    done: usize,
    total: usize,
    notify_step: usize,
    last_notified: &mut usize,
    force: bool,
) -> Result<(), String> {
    let total_use = total.max(1);
    let done_clamped = done.min(total_use);
    if !force && done_clamped < last_notified.saturating_add(notify_step.max(1)) {
        return Ok(());
    }
    *last_notified = done_clamped;
    if let Some(cb) = progress_callback {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (done_clamped, total_use))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    } else {
        check_ctrlc()?;
    }
    Ok(())
}

#[inline]
fn spgrm_spill_nnz_limit() -> usize {
    std::env::var("JANUSX_SPGRM_SPILL_NNZ")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(SPGRM_DEFAULT_SPILL_NNZ)
}

#[inline]
fn spgrm_streaming_merge_min_bytes() -> usize {
    if let Some(bytes) = std::env::var("JANUSX_SPGRM_STREAMING_MERGE_MIN_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
    {
        return bytes;
    }
    std::env::var("JANUSX_SPGRM_STREAMING_MERGE_MIN_MB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .map(|mb| mb.saturating_mul(1024).saturating_mul(1024))
        .unwrap_or(SPGRM_DEFAULT_STREAMING_MERGE_MIN_BYTES)
}

#[inline]
fn spgrm_batch_max_accum_bytes(sample_step: usize, threads: usize) -> usize {
    let cell_bytes = std::mem::size_of::<f64>() + std::mem::size_of::<f32>();
    if let Some(mb) = std::env::var("JANUSX_SPGRM_BATCH_MAX_ACCUM_MB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
    {
        return mb.saturating_mul(1024).saturating_mul(1024).max(cell_bytes);
    }
    let tile_bytes = sample_step
        .saturating_mul(sample_step)
        .saturating_mul(cell_bytes);
    tile_bytes
        .saturating_mul(threads.max(1))
        .saturating_mul(SPGRM_DEFAULT_BATCH_ACCUM_TILE_FACTOR)
        .max(SPGRM_DEFAULT_BATCH_MIN_ACCUM_BYTES)
        .max(cell_bytes)
}

fn spgrm_batch_max_decoded_bytes(
    row_step: usize,
    decoded_sample_width: usize,
    threads: usize,
    decode_target_mb: Option<f64>,
    decode_buffers: usize,
) -> usize {
    let full_batch_bytes = row_step
        .saturating_mul(decoded_sample_width.max(1))
        .saturating_mul(std::mem::size_of::<f32>())
        .max(std::mem::size_of::<f32>());
    if let Some(mb) = std::env::var("JANUSX_SPGRM_BATCH_MAX_DECODED_MB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
    {
        return mb
            .saturating_mul(1024)
            .saturating_mul(1024)
            .max(full_batch_bytes);
    }
    if let Some(n_stripes) = std::env::var("JANUSX_SPGRM_BATCH_MAX_UNIQUE_STRIPES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
    {
        let single_stripe_bytes = row_step
            .saturating_mul(
                decoded_sample_width
                    .max(1)
                    .div_ceil(SPGRM_DECODE_STRIPE_ESTIMATE.max(1)),
            )
            .saturating_mul(std::mem::size_of::<f32>())
            .max(std::mem::size_of::<f32>());
        return single_stripe_bytes
            .saturating_mul(n_stripes)
            .max(full_batch_bytes);
    }
    if let Some(target_mb) = decode_target_mb {
        let per_buffer_bytes =
            ((target_mb * 1024.0_f64 * 1024.0_f64) as usize).saturating_div(decode_buffers.max(1));
        return per_buffer_bytes.max(full_batch_bytes);
    }
    full_batch_bytes
        .saturating_mul(
            threads
                .max(1)
                .saturating_mul(SPGRM_DEFAULT_BATCH_DECODED_STRIPE_THREADS_FACTOR)
                .max(SPGRM_DEFAULT_BATCH_DECODED_STRIPE_FLOOR),
        )
        .max(full_batch_bytes)
}

#[inline]
fn spgrm_batch_limits(
    _row_step: usize,
    sample_step: usize,
    max_decoded_bytes: usize,
    threads: usize,
) -> SpgrmTaskBatchLimits {
    SpgrmTaskBatchLimits {
        max_decoded_bytes,
        max_accum_bytes: spgrm_batch_max_accum_bytes(sample_step, threads),
    }
}

#[inline]
fn spgrm_entry_cmp(a: &SpgrmEntry, b: &SpgrmEntry) -> Ordering {
    a.col.cmp(&b.col).then_with(|| a.row.cmp(&b.row))
}

fn spgrm_compress_sorted_entries(entries: &mut Vec<SpgrmEntry>) -> usize {
    if entries.is_empty() {
        return 0;
    }
    let mut write = 0usize;
    for read in 1..entries.len() {
        let cur = entries[read];
        if entries[write].col == cur.col && entries[write].row == cur.row {
            entries[write].value += cur.value;
        } else {
            write += 1;
            if write != read {
                entries[write] = cur;
            }
        }
    }
    entries.truncate(write + 1);
    entries.len()
}

#[inline]
fn spgrm_record_bytes() -> usize {
    std::mem::size_of::<u32>() * 2 + std::mem::size_of::<f64>()
}

fn read_chunk_entry(
    reader: &mut SpgrmChunkReader,
    chunk_idx: usize,
) -> Result<Option<SpgrmChunkHead>, String> {
    if reader.remaining == 0 {
        return Ok(None);
    }
    let mut col_buf = [0u8; 4];
    let mut row_buf = [0u8; 4];
    let mut val_buf = [0u8; 8];
    reader
        .reader
        .read_exact(&mut col_buf)
        .map_err(|e| format!("read sparse GRM chunk col failed: {e}"))?;
    reader
        .reader
        .read_exact(&mut row_buf)
        .map_err(|e| format!("read sparse GRM chunk row failed: {e}"))?;
    reader
        .reader
        .read_exact(&mut val_buf)
        .map_err(|e| format!("read sparse GRM chunk value failed: {e}"))?;
    reader.remaining = reader.remaining.saturating_sub(1);
    Ok(Some(SpgrmChunkHead {
        chunk_idx,
        col: u32::from_le_bytes(col_buf),
        row: u32::from_le_bytes(row_buf),
        value: f64::from_le_bytes(val_buf),
    }))
}

struct SpgrmCscStreamWriter {
    n_samples: usize,
    nnz: usize,
    written: usize,
    current_col: usize,
    col_ptr: Vec<u64>,
    prev_entry: Option<SpgrmEntry>,
    meta_file: File,
    row_writer: BufWriter<File>,
    value_writer: BufWriter<File>,
}

impl SpgrmCscStreamWriter {
    fn new(path: &str, n_samples: usize, nnz: usize) -> Result<Self, String> {
        let meta_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("create sparse GRM file {path} failed: {e}"))?;
        {
            let mut header_writer = BufWriter::with_capacity(
                SPGRM_DEFAULT_WRITE_BUF_BYTES,
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .map_err(|e| format!("open sparse GRM header handle for {path} failed: {e}"))?,
            );
            header_writer
                .write_all(&(n_samples as u64).to_le_bytes())
                .map_err(|e| format!("write sparse GRM header(n) failed: {e}"))?;
            header_writer
                .write_all(&(nnz as u64).to_le_bytes())
                .map_err(|e| format!("write sparse GRM header(nnz) failed: {e}"))?;
            let col_ptr_bytes =
                vec![0u8; (n_samples + 1).saturating_mul(std::mem::size_of::<u64>())];
            header_writer
                .write_all(col_ptr_bytes.as_slice())
                .map_err(|e| format!("write sparse GRM placeholder col_ptr failed: {e}"))?;
            header_writer
                .flush()
                .map_err(|e| format!("flush sparse GRM header failed: {e}"))?;
        }
        let row_offset = 2usize.saturating_mul(std::mem::size_of::<u64>())
            + (n_samples + 1).saturating_mul(std::mem::size_of::<u64>());
        let row_bytes = nnz.saturating_mul(std::mem::size_of::<u32>());
        let value_offset = row_offset
            .saturating_add(row_bytes)
            .saturating_add(spgrm_jxgrm_values_padding_bytes(nnz)?);

        let mut row_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("open sparse GRM row handle for {path} failed: {e}"))?;
        row_file
            .seek(SeekFrom::Start(row_offset as u64))
            .map_err(|e| format!("seek sparse GRM row payload failed: {e}"))?;
        let mut value_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("open sparse GRM value handle for {path} failed: {e}"))?;
        value_file
            .seek(SeekFrom::Start(value_offset as u64))
            .map_err(|e| format!("seek sparse GRM value payload failed: {e}"))?;

        Ok(Self {
            n_samples,
            nnz,
            written: 0usize,
            current_col: 0usize,
            col_ptr: vec![0u64; n_samples + 1],
            prev_entry: None,
            meta_file,
            row_writer: BufWriter::with_capacity(SPGRM_DEFAULT_WRITE_BUF_BYTES, row_file),
            value_writer: BufWriter::with_capacity(SPGRM_DEFAULT_WRITE_BUF_BYTES, value_file),
        })
    }

    fn push(&mut self, entry: SpgrmEntry) -> Result<(), String> {
        let col = entry.col as usize;
        let row = entry.row as usize;
        if col >= self.n_samples {
            return Err(format!(
                "Sparse GRM CSC writer column index out of range: {col} >= {}",
                self.n_samples
            ));
        }
        if row >= self.n_samples {
            return Err(format!(
                "Sparse GRM CSC writer row index out of range: {row} >= {}",
                self.n_samples
            ));
        }
        if row < col {
            return Err(format!(
                "Sparse GRM CSC writer expects lower triangle; got row={row} < col={col}"
            ));
        }
        if !entry.value.is_finite() {
            return Err(format!(
                "Sparse GRM CSC writer received non-finite value at pair ({row}, {col})"
            ));
        }
        if let Some(prev) = self.prev_entry {
            match spgrm_entry_cmp(&prev, &entry) {
                Ordering::Less => {}
                Ordering::Equal => {
                    return Err(format!(
                        "Sparse GRM duplicate COO entry detected at pair ({row}, {col})"
                    ));
                }
                Ordering::Greater => {
                    return Err(format!(
                        "Sparse GRM merge order is not sorted at pair ({row}, {col})"
                    ));
                }
            }
        }
        while self.current_col < col {
            self.col_ptr[self.current_col + 1] = self.written as u64;
            self.current_col += 1;
        }
        self.row_writer
            .write_all(&entry.row.to_le_bytes())
            .map_err(|e| format!("write sparse GRM row payload failed: {e}"))?;
        self.value_writer
            .write_all(&entry.value.to_le_bytes())
            .map_err(|e| format!("write sparse GRM value payload failed: {e}"))?;
        self.written = self.written.saturating_add(1);
        self.prev_entry = Some(entry);
        Ok(())
    }

    fn finish(mut self) -> Result<(), String> {
        while self.current_col < self.n_samples {
            self.col_ptr[self.current_col + 1] = self.written as u64;
            self.current_col += 1;
        }
        if self.written != self.nnz {
            return Err(format!(
                "Sparse GRM CSC writer count mismatch: wrote {}, expected {}",
                self.written, self.nnz
            ));
        }
        self.row_writer
            .flush()
            .map_err(|e| format!("flush sparse GRM row payload failed: {e}"))?;
        self.row_writer
            .get_ref()
            .sync_data()
            .map_err(|e| format!("sync sparse GRM row payload failed: {e}"))?;
        let row_end = (2usize * std::mem::size_of::<u64>())
            .saturating_add((self.n_samples + 1).saturating_mul(std::mem::size_of::<u64>()))
            .saturating_add(self.nnz.saturating_mul(std::mem::size_of::<u32>()));
        let pad_bytes = spgrm_jxgrm_values_padding_bytes(self.nnz)?;
        if pad_bytes > 0 {
            self.meta_file
                .seek(SeekFrom::Start(row_end as u64))
                .map_err(|e| format!("seek sparse GRM alignment padding failed: {e}"))?;
            self.meta_file
                .write_all(&[0u8; 8][..pad_bytes])
                .map_err(|e| format!("write sparse GRM alignment padding failed: {e}"))?;
        }
        self.value_writer
            .flush()
            .map_err(|e| format!("flush sparse GRM value payload failed: {e}"))?;
        self.value_writer
            .get_ref()
            .sync_data()
            .map_err(|e| format!("sync sparse GRM value payload failed: {e}"))?;
        self.meta_file
            .seek(SeekFrom::Start(2u64 * std::mem::size_of::<u64>() as u64))
            .map_err(|e| format!("seek sparse GRM col_ptr payload failed: {e}"))?;
        let mut col_writer =
            BufWriter::with_capacity(SPGRM_DEFAULT_WRITE_BUF_BYTES, self.meta_file);
        write_u64_slice(&mut col_writer, self.col_ptr.as_slice())?;
        col_writer
            .flush()
            .map_err(|e| format!("flush sparse GRM col_ptr payload failed: {e}"))?;
        col_writer
            .get_ref()
            .sync_data()
            .map_err(|e| format!("sync sparse GRM file failed: {e}"))?;
        Ok(())
    }
}

struct SpgrmCscTempPayloadWriter {
    out_path: String,
    row_path: PathBuf,
    value_path: PathBuf,
    n_samples: usize,
    written: usize,
    current_col: usize,
    col_ptr: Vec<u64>,
    prev_entry: Option<SpgrmEntry>,
    row_writer: BufWriter<File>,
    value_writer: BufWriter<File>,
}

impl SpgrmCscTempPayloadWriter {
    fn new(out_path: &str, n_samples: usize) -> Result<Self, String> {
        let suffix = format!(
            "merge{}.{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let row_path = spgrm_temp_merge_path(out_path, &suffix, "rows");
        let value_path = spgrm_temp_merge_path(out_path, &suffix, "vals");
        let row_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&row_path)
            .map_err(|e| {
                format!(
                    "create sparse GRM merge temp row file {} failed: {e}",
                    row_path.display()
                )
            })?;
        let value_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&value_path)
            .map_err(|e| {
                format!(
                    "create sparse GRM merge temp value file {} failed: {e}",
                    value_path.display()
                )
            })?;
        Ok(Self {
            out_path: out_path.to_string(),
            row_path,
            value_path,
            n_samples,
            written: 0usize,
            current_col: 0usize,
            col_ptr: vec![0u64; n_samples + 1],
            prev_entry: None,
            row_writer: BufWriter::with_capacity(SPGRM_DEFAULT_WRITE_BUF_BYTES, row_file),
            value_writer: BufWriter::with_capacity(SPGRM_DEFAULT_WRITE_BUF_BYTES, value_file),
        })
    }

    fn push(&mut self, entry: SpgrmEntry) -> Result<(), String> {
        let col = entry.col as usize;
        let row = entry.row as usize;
        if col >= self.n_samples {
            return Err(format!(
                "Sparse GRM temp writer column index out of range: {col} >= {}",
                self.n_samples
            ));
        }
        if row >= self.n_samples {
            return Err(format!(
                "Sparse GRM temp writer row index out of range: {row} >= {}",
                self.n_samples
            ));
        }
        if row < col {
            return Err(format!(
                "Sparse GRM temp writer expects lower triangle; got row={row} < col={col}"
            ));
        }
        if !entry.value.is_finite() {
            return Err(format!(
                "Sparse GRM temp writer received non-finite value at pair ({row}, {col})"
            ));
        }
        if let Some(prev) = self.prev_entry {
            match spgrm_entry_cmp(&prev, &entry) {
                Ordering::Less => {}
                Ordering::Equal => {
                    return Err(format!(
                        "Sparse GRM duplicate temp merge entry detected at pair ({row}, {col})"
                    ));
                }
                Ordering::Greater => {
                    return Err(format!(
                        "Sparse GRM temp merge order is not sorted at pair ({row}, {col})"
                    ));
                }
            }
        }
        while self.current_col < col {
            self.col_ptr[self.current_col + 1] = self.written as u64;
            self.current_col += 1;
        }
        self.row_writer
            .write_all(&entry.row.to_le_bytes())
            .map_err(|e| format!("write sparse GRM temp row payload failed: {e}"))?;
        self.value_writer
            .write_all(&entry.value.to_le_bytes())
            .map_err(|e| format!("write sparse GRM temp value payload failed: {e}"))?;
        self.written = self.written.saturating_add(1);
        self.prev_entry = Some(entry);
        Ok(())
    }

    fn finish(mut self) -> Result<(usize, usize), String> {
        while self.current_col < self.n_samples {
            self.col_ptr[self.current_col + 1] = self.written as u64;
            self.current_col += 1;
        }
        let row_path = self.row_path.clone();
        let value_path = self.value_path.clone();
        let result = (|| -> Result<(usize, usize), String> {
            self.row_writer
                .flush()
                .map_err(|e| format!("flush sparse GRM temp row payload failed: {e}"))?;
            self.value_writer
                .flush()
                .map_err(|e| format!("flush sparse GRM temp value payload failed: {e}"))?;
            let row_file = self
                .row_writer
                .into_inner()
                .map_err(|e| format!("close sparse GRM temp row payload failed: {}", e.error()))?;
            row_file
                .sync_data()
                .map_err(|e| format!("sync sparse GRM temp row payload failed: {e}"))?;
            drop(row_file);
            let value_file = self.value_writer.into_inner().map_err(|e| {
                format!("close sparse GRM temp value payload failed: {}", e.error())
            })?;
            value_file
                .sync_data()
                .map_err(|e| format!("sync sparse GRM temp value payload failed: {e}"))?;
            drop(value_file);

            let mut out_file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(self.out_path.as_str())
                .map_err(|e| format!("create sparse GRM file {} failed: {e}", self.out_path))?;
            {
                let mut header_writer =
                    BufWriter::with_capacity(SPGRM_DEFAULT_WRITE_BUF_BYTES, &mut out_file);
                header_writer
                    .write_all(&(self.n_samples as u64).to_le_bytes())
                    .map_err(|e| format!("write sparse GRM header(n) failed: {e}"))?;
                header_writer
                    .write_all(&(self.written as u64).to_le_bytes())
                    .map_err(|e| format!("write sparse GRM header(nnz) failed: {e}"))?;
                write_u64_slice(&mut header_writer, self.col_ptr.as_slice())?;
                header_writer
                    .flush()
                    .map_err(|e| format!("flush sparse GRM header failed: {e}"))?;
            }

            out_file
                .seek(SeekFrom::End(0))
                .map_err(|e| format!("seek sparse GRM row payload append failed: {e}"))?;
            let mut row_reader = File::open(&row_path).map_err(|e| {
                format!(
                    "open sparse GRM merge temp row file {} failed: {e}",
                    row_path.display()
                )
            })?;
            std::io::copy(&mut row_reader, &mut out_file)
                .map_err(|e| format!("append sparse GRM row payload failed: {e}"))?;

            let pad_bytes = spgrm_jxgrm_values_padding_bytes(self.written)?;
            if pad_bytes > 0 {
                out_file
                    .write_all(&[0u8; 8][..pad_bytes])
                    .map_err(|e| format!("write sparse GRM alignment padding failed: {e}"))?;
            }

            let mut value_reader = File::open(&value_path).map_err(|e| {
                format!(
                    "open sparse GRM merge temp value file {} failed: {e}",
                    value_path.display()
                )
            })?;
            std::io::copy(&mut value_reader, &mut out_file)
                .map_err(|e| format!("append sparse GRM value payload failed: {e}"))?;
            out_file
                .sync_data()
                .map_err(|e| format!("sync sparse GRM file failed: {e}"))?;
            Ok((self.n_samples, self.written))
        })();
        let _ = std::fs::remove_file(&row_path);
        let _ = std::fs::remove_file(&value_path);
        result
    }
}

#[inline]
fn spgrm_temp_merge_path(out_path: &str, suffix: &str, kind: &str) -> PathBuf {
    let mut path = PathBuf::from(out_path);
    let new_ext = match path.extension().and_then(|s| s.to_str()) {
        Some(ext) if !ext.is_empty() => format!("{ext}.{suffix}.{kind}.tmp"),
        _ => format!("{suffix}.{kind}.tmp"),
    };
    path.set_extension(new_ext);
    path
}

fn for_each_merged_chunk_entry<F>(chunk_paths: &[PathBuf], mut callback: F) -> Result<(), String>
where
    F: FnMut(SpgrmEntry) -> Result<(), String>,
{
    let mut readers = Vec::<SpgrmChunkReader>::with_capacity(chunk_paths.len());
    let mut heap = BinaryHeap::<SpgrmChunkHead>::new();
    for (chunk_idx, chunk_path) in chunk_paths.iter().enumerate() {
        let mut reader = open_spgrm_chunk_reader(chunk_path)?;
        if let Some(head) = read_chunk_entry(&mut reader, chunk_idx)? {
            heap.push(head);
        }
        readers.push(reader);
    }

    let mut pending: Option<SpgrmEntry> = None;
    while let Some(head) = heap.pop() {
        let entry = SpgrmEntry {
            col: head.col,
            row: head.row,
            value: head.value,
        };
        if let Some(prev) = pending.as_mut() {
            if prev.col == entry.col && prev.row == entry.row {
                prev.value += entry.value;
            } else {
                callback(*prev)?;
                *prev = entry;
            }
        } else {
            pending = Some(entry);
        }
        if let Some(next) = read_chunk_entry(&mut readers[head.chunk_idx], head.chunk_idx)? {
            heap.push(next);
        }
    }
    if let Some(entry) = pending {
        callback(entry)?;
    }
    Ok(())
}

#[inline]
fn spgrm_chunk_merge_estimated_bytes(nnz: usize) -> usize {
    nnz.saturating_mul(spgrm_record_bytes())
}

#[inline]
fn spgrm_should_use_streaming_merge(chunk_paths: &[PathBuf], nnz: usize) -> bool {
    chunk_paths.len() > 1
        && spgrm_chunk_merge_estimated_bytes(nnz) >= spgrm_streaming_merge_min_bytes()
}

struct SpgrmPreparedBuild {
    m: usize,
    bytes_per_snp: usize,
    n_use: usize,
    global_full_fast: bool,
    row_step: usize,
    inv_scale: f64,
    threshold: f64,
    abs_threshold: bool,
    sample_step: usize,
    total_pairs: usize,
    pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    tasks: Vec<SpgrmTask>,
    task_batches: Vec<SpgrmTaskBatchSpan>,
}

#[inline]
fn spgrm_blas_threads_for_task_count(task_count: usize, threads: usize) -> Option<usize> {
    if threads <= 1 {
        None
    } else if task_count > 1 {
        Some(1usize)
    } else {
        Some(threads.max(1))
    }
}

#[inline]
fn spgrm_blas_threads_for_build(build: &SpgrmPreparedBuild, threads: usize) -> Option<usize> {
    spgrm_blas_threads_for_task_count(build.total_pairs, threads)
}

#[inline]
fn spgrm_keep_value(value: f64, threshold: f64, abs_threshold: bool) -> bool {
    if abs_threshold {
        value.abs() > threshold
    } else if threshold < 0.0 {
        true
    } else {
        value > threshold
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_spgrm_build(
    packed_flat: &[u8],
    n_samples_full: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    method: usize,
    threshold: f64,
    abs_threshold: bool,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
) -> Result<SpgrmPreparedBuild, String> {
    let (m, bytes_per_snp, n_use, global_full_fast) = validate_spgrm_inputs(
        packed_flat,
        n_samples_full,
        row_flip,
        row_maf,
        sample_idx,
        method,
        threshold,
    )?;
    let memory_plan = spgrm_resolve_packed_memory_plan(
        m,
        n_use,
        n_samples_full,
        block_rows,
        sample_block,
        threads,
    );
    let sample_step = memory_plan.sample_step;
    let row_step = memory_plan.row_step;
    let denom = if method == 1 {
        centered_varsum_from_packed(
            packed_flat,
            n_samples_full,
            row_flip,
            row_maf,
            sample_idx,
            row_step,
            threads,
        )?
    } else {
        m as f64
    };
    if !(denom.is_finite() && denom > 0.0) {
        return Err("Sparse GRM denominator is not positive".to_string());
    }
    let n_blocks = n_use.div_ceil(sample_step);
    let total_pairs = n_blocks.saturating_mul(n_blocks + 1) / 2;
    let tasks = build_spgrm_tasks(n_use, sample_step);
    let task_batches = build_spgrm_task_batches(
        tasks.as_slice(),
        row_step,
        spgrm_batch_limits(
            row_step,
            sample_step,
            memory_plan.max_decoded_bytes,
            threads,
        ),
    );
    Ok(SpgrmPreparedBuild {
        m,
        bytes_per_snp,
        n_use,
        global_full_fast,
        row_step,
        inv_scale: 1.0_f64 / denom,
        threshold,
        abs_threshold,
        sample_step,
        total_pairs,
        pool: get_cached_pool(threads).map_err(|e| e.to_string())?,
        tasks,
        task_batches,
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_spgrm_task_batch_entries(
    build: &SpgrmPreparedBuild,
    task_batch: &[SpgrmTask],
    packed_flat: &[u8],
    n_samples_full: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    method: usize,
    threads: usize,
    timing: Option<Arc<SpgrmStageTiming>>,
) -> Result<Vec<SpgrmEntry>, String> {
    let batch_results: Vec<Result<Vec<SpgrmEntry>, String>> = if threads > 1 && task_batch.len() > 1
    {
        let build_batch = || {
            task_batch
                .par_iter()
                .copied()
                .map_init(
                    || SpgrmWorkerScratch::new(build.row_step, build.sample_step),
                    |scratch, task| {
                        compute_spgrm_task_entries(
                            scratch,
                            task,
                            packed_flat,
                            build.bytes_per_snp,
                            n_samples_full,
                            row_flip,
                            row_maf,
                            sample_idx,
                            method,
                            build.row_step,
                            build.global_full_fast,
                            build.inv_scale,
                            build.threshold,
                            build.abs_threshold,
                            build.m,
                            build.pool.as_ref(),
                            timing.as_deref(),
                        )
                    },
                )
                .collect()
        };
        if let Some(tp) = build.pool.as_ref() {
            tp.install(build_batch)
        } else {
            build_batch()
        }
    } else {
        let mut scratch = SpgrmWorkerScratch::new(build.row_step, build.sample_step);
        let mut out = Vec::<Result<Vec<SpgrmEntry>, String>>::with_capacity(task_batch.len());
        for &task in task_batch {
            out.push(compute_spgrm_task_entries(
                &mut scratch,
                task,
                packed_flat,
                build.bytes_per_snp,
                n_samples_full,
                row_flip,
                row_maf,
                sample_idx,
                method,
                build.row_step,
                build.global_full_fast,
                build.inv_scale,
                build.threshold,
                build.abs_threshold,
                build.m,
                build.pool.as_ref(),
                timing.as_deref(),
            ));
        }
        out
    };
    let mut out = Vec::<SpgrmEntry>::new();
    for entries_res in batch_results {
        let mut entries = entries_res?;
        out.append(&mut entries);
    }
    Ok(out)
}

fn spgrm_chunk_path(out_path: &str, chunk_idx: usize) -> PathBuf {
    let mut path = PathBuf::from(out_path);
    let new_ext = match path.extension().and_then(|s| s.to_str()) {
        Some(ext) if !ext.is_empty() => format!("{ext}.chunk{chunk_idx:06}.coo"),
        _ => format!("chunk{chunk_idx:06}.coo"),
    };
    path.set_extension(new_ext);
    path
}

fn spill_sorted_entries_to_chunk(
    chunk_path: &PathBuf,
    entries: &mut Vec<SpgrmEntry>,
) -> Result<usize, String> {
    entries.sort_unstable_by(spgrm_entry_cmp);
    let file = File::create(chunk_path).map_err(|e| {
        format!(
            "create sparse GRM chunk {} failed: {e}",
            chunk_path.display()
        )
    })?;
    let est_bytes = entries
        .len()
        .saturating_mul(spgrm_record_bytes())
        .max(SPGRM_DEFAULT_WRITE_BUF_BYTES);
    let mut writer = BufWriter::with_capacity(est_bytes.min(64 * 1024 * 1024), file);
    writer
        .write_all(&(entries.len() as u64).to_le_bytes())
        .map_err(|e| format!("write sparse GRM chunk header failed: {e}"))?;
    for entry in entries.iter() {
        writer
            .write_all(&entry.col.to_le_bytes())
            .and_then(|_| writer.write_all(&entry.row.to_le_bytes()))
            .and_then(|_| writer.write_all(&entry.value.to_le_bytes()))
            .map_err(|e| format!("write sparse GRM chunk payload failed: {e}"))?;
    }
    writer.flush().map_err(|e| {
        format!(
            "flush sparse GRM chunk {} failed: {e}",
            chunk_path.display()
        )
    })?;
    let count = entries.len();
    entries.clear();
    Ok(count)
}

fn open_spgrm_chunk_reader(chunk_path: &PathBuf) -> Result<SpgrmChunkReader, String> {
    let file = File::open(chunk_path)
        .map_err(|e| format!("open sparse GRM chunk {} failed: {e}", chunk_path.display()))?;
    let mut reader = BufReader::with_capacity(SPGRM_DEFAULT_WRITE_BUF_BYTES, file);
    let mut count_buf = [0u8; 8];
    reader.read_exact(&mut count_buf).map_err(|e| {
        format!(
            "read sparse GRM chunk header {} failed: {e}",
            chunk_path.display()
        )
    })?;
    let remaining_u64 = u64::from_le_bytes(count_buf);
    let remaining = usize::try_from(remaining_u64).map_err(|_| {
        format!(
            "sparse GRM chunk too large for usize: {}",
            chunk_path.display()
        )
    })?;
    Ok(SpgrmChunkReader { reader, remaining })
}

#[inline]
fn build_spgrm_tasks(n_use: usize, sample_step: usize) -> Vec<SpgrmTask> {
    let n_blocks = n_use.div_ceil(sample_step);
    let total_pairs = n_blocks.saturating_mul(n_blocks + 1) / 2;
    let mut tasks = Vec::<SpgrmTask>::with_capacity(total_pairs);
    for col_start in (0..n_use).step_by(sample_step) {
        let col_end = (col_start + sample_step).min(n_use);
        for row_start in (col_start..n_use).step_by(sample_step) {
            let row_end = (row_start + sample_step).min(n_use);
            tasks.push(SpgrmTask {
                col_start,
                col_end,
                row_start,
                row_end,
            });
        }
    }
    tasks
}

#[inline]
fn build_spgrm_row_band_tasks(
    n_use: usize,
    sample_step: usize,
    row_band_start: usize,
    row_band_end: usize,
) -> Vec<SpgrmTask> {
    build_spgrm_tasks(n_use, sample_step)
        .into_iter()
        .filter(|task| task.row_end > row_band_start && task.row_start < row_band_end)
        .collect()
}

#[inline]
fn spgrm_task_accum_bytes(task: SpgrmTask) -> usize {
    (task.row_end - task.row_start)
        .saturating_mul(task.col_end - task.col_start)
        .saturating_mul(std::mem::size_of::<f64>() + std::mem::size_of::<f32>())
}

#[inline]
fn spgrm_stripe_decoded_bytes(stripe: SpgrmStripe, row_step: usize) -> usize {
    row_step
        .saturating_mul(stripe.end.saturating_sub(stripe.start))
        .saturating_mul(std::mem::size_of::<f32>())
}

#[inline]
fn spgrm_find_stripe_index(stripes: &[SpgrmStripe], stripe: SpgrmStripe) -> Option<usize> {
    stripes.iter().position(|&x| x == stripe)
}

#[inline]
fn spgrm_collect_group_stripes(tasks: &[SpgrmTask]) -> (Vec<SpgrmStripe>, usize) {
    let mut out = Vec::<SpgrmStripe>::new();
    let mut group_accum_bytes = 0usize;
    for &task in tasks {
        group_accum_bytes = group_accum_bytes.saturating_add(spgrm_task_accum_bytes(task));
        let col_stripe = SpgrmStripe {
            start: task.col_start,
            end: task.col_end,
        };
        if spgrm_find_stripe_index(out.as_slice(), col_stripe).is_none() {
            out.push(col_stripe);
        }
        let row_stripe = SpgrmStripe {
            start: task.row_start,
            end: task.row_end,
        };
        if spgrm_find_stripe_index(out.as_slice(), row_stripe).is_none() {
            out.push(row_stripe);
        }
    }
    (out, group_accum_bytes)
}

#[inline]
fn build_spgrm_task_batches(
    tasks: &[SpgrmTask],
    row_step: usize,
    limits: SpgrmTaskBatchLimits,
) -> Vec<SpgrmTaskBatchSpan> {
    if tasks.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::<SpgrmTaskBatchSpan>::new();
    let mut batch_start = 0usize;
    let mut batch_decoded_bytes = 0usize;
    let mut batch_accum_bytes = 0usize;
    let mut batch_stripes = Vec::<SpgrmStripe>::new();
    let mut i = 0usize;
    while i < tasks.len() {
        let col_start = tasks[i].col_start;
        let col_end = tasks[i].col_end;
        let col_group_start = i;
        let mut j = i + 1;
        while j < tasks.len() && tasks[j].col_start == col_start && tasks[j].col_end == col_end {
            j += 1;
        }
        let group_tasks = &tasks[col_group_start..j];
        let (group_stripes, group_accum_bytes) = spgrm_collect_group_stripes(group_tasks);
        let mut candidate_decoded_bytes = batch_decoded_bytes;
        for stripe in group_stripes.iter().copied() {
            if spgrm_find_stripe_index(batch_stripes.as_slice(), stripe).is_none() {
                candidate_decoded_bytes = candidate_decoded_bytes
                    .saturating_add(spgrm_stripe_decoded_bytes(stripe, row_step));
            }
        }
        let candidate_accum_bytes = batch_accum_bytes.saturating_add(group_accum_bytes);
        let exceeds_limits = candidate_decoded_bytes > limits.max_decoded_bytes
            || candidate_accum_bytes > limits.max_accum_bytes;
        if col_group_start > batch_start && exceeds_limits {
            out.push(SpgrmTaskBatchSpan {
                start: batch_start,
                end: col_group_start,
            });
            batch_start = col_group_start;
            batch_decoded_bytes = 0usize;
            batch_accum_bytes = 0usize;
            batch_stripes.clear();
        }
        for stripe in group_stripes {
            if spgrm_find_stripe_index(batch_stripes.as_slice(), stripe).is_none() {
                batch_decoded_bytes = batch_decoded_bytes
                    .saturating_add(spgrm_stripe_decoded_bytes(stripe, row_step));
                batch_stripes.push(stripe);
            }
        }
        batch_accum_bytes = batch_accum_bytes.saturating_add(group_accum_bytes);
        i = j;
    }
    if batch_start < tasks.len() {
        out.push(SpgrmTaskBatchSpan {
            start: batch_start,
            end: tasks.len(),
        });
    }
    out
}

#[inline]
fn spgrm_get_or_push_stripe(stripes: &mut Vec<SpgrmStripe>, start: usize, end: usize) -> usize {
    if let Some((idx, _)) = stripes
        .iter()
        .enumerate()
        .find(|(_, stripe)| stripe.start == start && stripe.end == end)
    {
        idx
    } else {
        stripes.push(SpgrmStripe { start, end });
        stripes.len() - 1
    }
}

#[inline]
fn spgrm_intra_tile_sample_block(width: usize, threads: usize) -> usize {
    if threads <= 1 || width <= 512 {
        return width.max(1);
    }
    if let Some(v) = std::env::var("JANUSX_SPGRM_INTRA_TILE_BLOCK")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
    {
        return v.min(width).max(1);
    }
    width.max(1)
}

#[inline]
fn spgrm_single_diag_task_subtile_step(task_batch: &[SpgrmTask], threads: usize) -> Option<usize> {
    if task_batch.len() != 1 || threads <= 1 {
        return None;
    }
    let task = task_batch[0];
    if task.row_start != task.col_start || task.row_end != task.col_end {
        return None;
    }
    let width = task.row_end.saturating_sub(task.row_start);
    let step = spgrm_intra_tile_sample_block(width, threads);
    if step >= width {
        None
    } else {
        Some(step)
    }
}

fn spgrm_build_stream_batch(
    task_batch: &[SpgrmTask],
    row_step: usize,
    sample_idx: &[usize],
    n_samples_full: usize,
    threads: usize,
) -> (Vec<SpgrmStreamStripeScratch>, Vec<SpgrmBatchTaskAccum>) {
    let mut stripes = Vec::<SpgrmStripe>::new();
    let mut task_specs =
        Vec::<(SpgrmTask, usize, usize, usize, usize, bool)>::with_capacity(task_batch.len());
    if let Some(subtile_step) = spgrm_single_diag_task_subtile_step(task_batch, threads) {
        let task = task_batch[0];
        let stripe_idx = spgrm_get_or_push_stripe(&mut stripes, task.col_start, task.col_end);
        for col_start in (task.col_start..task.col_end).step_by(subtile_step) {
            let col_end = (col_start + subtile_step).min(task.col_end);
            for row_start in (col_start..task.row_end).step_by(subtile_step) {
                let row_end = (row_start + subtile_step).min(task.row_end);
                task_specs.push((
                    SpgrmTask {
                        col_start,
                        col_end,
                        row_start,
                        row_end,
                    },
                    stripe_idx,
                    stripe_idx,
                    row_start - task.row_start,
                    col_start - task.col_start,
                    true,
                ));
            }
        }
    } else {
        for &task in task_batch {
            let col_idx = spgrm_get_or_push_stripe(&mut stripes, task.col_start, task.col_end);
            let row_idx = if task.row_start == task.col_start && task.row_end == task.col_end {
                col_idx
            } else {
                spgrm_get_or_push_stripe(&mut stripes, task.row_start, task.row_end)
            };
            task_specs.push((task, row_idx, col_idx, 0usize, 0usize, false));
        }
    }
    let stripe_scratch = stripes
        .into_iter()
        .map(|stripe| SpgrmStreamStripeScratch::new(stripe, row_step, sample_idx, n_samples_full))
        .collect::<Vec<_>>();
    let task_accum = task_specs
        .into_iter()
        .map(
            |(
                task,
                row_stripe_idx,
                col_stripe_idx,
                row_offset_in_stripe,
                col_offset_in_stripe,
                accum_row_major,
            )| {
                let n_row = task.row_end - task.row_start;
                let n_col = task.col_end - task.col_start;
                SpgrmBatchTaskAccum {
                    task,
                    row_stripe_idx,
                    col_stripe_idx,
                    row_offset_in_stripe,
                    col_offset_in_stripe,
                    accum_row_major,
                    accum: vec![0.0_f64; n_row.saturating_mul(n_col)],
                    accum_tmp_f32: vec![0.0_f32; n_row.saturating_mul(n_col)],
                }
            },
        )
        .collect::<Vec<_>>();
    (stripe_scratch, task_accum)
}

#[allow(clippy::too_many_arguments)]
fn spgrm_decode_stream_batch_buffer(
    buf: &mut SpgrmDecodedBatchBuffer,
    row_start: usize,
    row_step: usize,
    m: usize,
    sample_idx: &[usize],
    meta: &crate::gfreader::PreparedBedLogicMetaOwned,
    row_mean: &[f32],
    row_inv_sd: &[f32],
    bed_window: Option<&mut WindowedBedMatrix>,
    full_mmap: Option<&Mmap>,
    rel_row_indices: &mut Vec<usize>,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&std::sync::Arc<rayon::ThreadPool>>,
    timing: Option<&SpgrmStageTiming>,
) -> Result<(), String> {
    let cur_rows = (row_start + row_step).min(m).saturating_sub(row_start);
    if cur_rows == 0 {
        buf.row_start = row_start;
        buf.cur_rows = 0;
        return Ok(());
    }
    let packed_block: &[u8];
    let packed_row_indices: Option<&[usize]>;
    let decode_row_start: usize;
    let row_flip_use: &[bool];
    let row_mean_use: &[f32];
    let row_inv_sd_use: &[f32];

    if let Some(window) = bed_window {
        let row_end = row_start + cur_rows;
        packed_block = window.prepare_source_rows(
            &meta.row_source_indices[row_start..row_end],
            rel_row_indices,
        )?;
        packed_row_indices = Some(rel_row_indices.as_slice());
        decode_row_start = 0usize;
        row_flip_use = &meta.row_flip[row_start..row_end];
        row_mean_use = &row_mean[row_start..row_end];
        row_inv_sd_use = &row_inv_sd[row_start..row_end];
    } else {
        let mmap = full_mmap.ok_or_else(|| "internal error: missing full BED mmap".to_string())?;
        packed_block = &mmap[3..];
        packed_row_indices = Some(meta.row_source_indices.as_slice());
        decode_row_start = row_start;
        row_flip_use = meta.row_flip.as_slice();
        row_mean_use = row_mean;
        row_inv_sd_use = row_inv_sd;
    }

    for stripe in buf.stripe_scratch.iter_mut() {
        let stripe_idx = &sample_idx[stripe.stripe.start..stripe.stripe.end];
        let stripe_width = stripe.width();
        let t_decode = Instant::now();
        decode_standardized_packed_block_rows_f32_with_plan(
            packed_block,
            meta.bytes_per_snp,
            meta.n_samples,
            row_flip_use,
            row_mean_use,
            row_inv_sd_use,
            stripe_idx,
            stripe.full_sample_fast,
            stripe.subset_plan.as_ref(),
            packed_row_indices,
            decode_row_start,
            &mut stripe.decoded[..cur_rows * stripe_width],
            code4_lut,
            pool,
        )?;
        if let Some(timing) = timing {
            timing.add_decode_ns(spgrm_elapsed_ns(t_decode));
        }
    }

    buf.row_start = row_start;
    buf.cur_rows = cur_rows;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn spgrm_consume_stream_batch_buffer(
    buf: &SpgrmDecodedBatchBuffer,
    task_accum: &mut [SpgrmBatchTaskAccum],
    batch_idx: usize,
    m: usize,
    progress_done_offset: usize,
    total_progress: usize,
    notify_step: usize,
    last_notified: &mut usize,
    progress_callback: Option<&Py<PyAny>>,
    threads: usize,
    pool: Option<&std::sync::Arc<rayon::ThreadPool>>,
    timing: Option<&SpgrmStageTiming>,
) -> Result<(), String> {
    if buf.cur_rows == 0 {
        return Ok(());
    }
    let is_first_block = buf.row_start == 0;
    if threads > 1 && task_accum.len() > 1 {
        let timing_ref = timing;
        let stripe_scratch = buf.stripe_scratch.as_slice();
        let mut gemm_work = || {
            task_accum.par_iter_mut().for_each(|task_acc| {
                spgrm_accumulate_task_gemm(
                    task_acc,
                    stripe_scratch,
                    buf.cur_rows,
                    is_first_block,
                    timing_ref,
                );
            });
        };
        if let Some(tp) = pool {
            tp.install(gemm_work);
        } else {
            gemm_work();
        }
    } else {
        for task_acc in task_accum.iter_mut() {
            spgrm_accumulate_task_gemm(
                task_acc,
                buf.stripe_scratch.as_slice(),
                buf.cur_rows,
                is_first_block,
                timing,
            );
        }
    }

    let batch_progress_offset = batch_idx.saturating_mul(m);
    let done_rows = batch_progress_offset.saturating_add(buf.row_start + buf.cur_rows);
    spgrm_progress_notify(
        progress_callback,
        progress_done_offset.saturating_add(done_rows),
        total_progress,
        notify_step,
        last_notified,
        false,
    )
}

fn spgrm_open_bed_payload_mmap(
    prefix: &str,
    n_samples: usize,
    n_snps_total: usize,
    bytes_per_snp: usize,
) -> Result<Mmap, String> {
    let bed_path = format!("{prefix}.bed");
    let bed_file = File::open(&bed_path).map_err(|e| format!("failed to open {bed_path}: {e}"))?;
    let mmap =
        unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
    if mmap.len() < 3 {
        return Err("BED too small".to_string());
    }
    if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
        return Err("Only SNP-major BED supported".to_string());
    }
    let expected_bps = n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(format!(
            "Sparse GRM bytes_per_snp mismatch: meta={bytes_per_snp}, expected={expected_bps}"
        ));
    }
    let data_len = mmap.len() - 3;
    if data_len % bytes_per_snp != 0 {
        return Err(format!(
            "invalid BED payload length: data_len={data_len}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    let n_snps_bed = data_len / bytes_per_snp;
    if n_snps_bed != n_snps_total {
        return Err(format!(
            "BED/BIM SNP count mismatch: bed={n_snps_bed}, expected={n_snps_total}"
        ));
    }
    Ok(mmap)
}

fn spgrm_row_mean_and_inv_sd(row_maf: &[f32], method: usize) -> (Vec<f32>, Vec<f32>) {
    let row_mean = row_maf
        .iter()
        .map(|&maf| (2.0_f32 * maf).clamp(0.0_f32, 2.0_f32))
        .collect::<Vec<_>>();
    let row_inv_sd = if method == 2 {
        row_maf
            .iter()
            .map(|&maf| {
                let p = maf.clamp(0.0_f32, 1.0_f32);
                let var = 2.0_f32 * p * (1.0_f32 - p);
                if var > 1e-12_f32 {
                    1.0_f32 / var.sqrt()
                } else {
                    0.0_f32
                }
            })
            .collect::<Vec<_>>()
    } else {
        vec![1.0_f32; row_maf.len()]
    };
    (row_mean, row_inv_sd)
}

fn spgrm_collect_batch_entries(
    task_accum: &[SpgrmBatchTaskAccum],
    inv_scale: f64,
    threshold: f64,
    abs_threshold: bool,
) -> Result<Vec<SpgrmEntry>, String> {
    let mut out = Vec::<SpgrmEntry>::new();
    for task_acc in task_accum {
        let task = task_acc.task;
        let n_col = task.col_end - task.col_start;
        let n_row = task.row_end - task.row_start;
        let diagonal_tile = task.row_start == task.col_start;
        out.reserve(task_acc.accum.len() / 4 + n_col.max(1));
        for local_col in 0..n_col {
            let global_col = task.col_start + local_col;
            let row_begin = if diagonal_tile { local_col } else { 0usize };
            for local_row in row_begin..n_row {
                let global_row = task.row_start + local_row;
                let accum_idx = if task_acc.accum_row_major {
                    local_row * n_col + local_col
                } else {
                    local_row + local_col * n_row
                };
                let scaled = task_acc.accum[accum_idx] * inv_scale;
                if !scaled.is_finite() {
                    return Err(format!(
                        "Sparse GRM produced non-finite value at pair ({global_row}, {global_col})"
                    ));
                }
                if global_row == global_col || spgrm_keep_value(scaled, threshold, abs_threshold) {
                    out.push(SpgrmEntry {
                        col: global_col as u32,
                        row: global_row as u32,
                        value: scaled,
                    });
                }
            }
        }
    }
    Ok(out)
}

fn spgrm_collect_batch_dense_rows(
    task_accum: &[SpgrmBatchTaskAccum],
    row_band_start: usize,
    row_band_end: usize,
    inv_scale: f64,
    n_use: usize,
    out: &mut [f32],
) -> Result<(), String> {
    let n_rows_band = row_band_end.saturating_sub(row_band_start);
    let expected_len = n_rows_band.saturating_mul(n_use);
    if out.len() != expected_len {
        return Err(format!(
            "dense GRM part output length mismatch: got {}, expected {}",
            out.len(),
            expected_len
        ));
    }
    for task_acc in task_accum {
        let task = task_acc.task;
        let n_col = task.col_end - task.col_start;
        let n_row = task.row_end - task.row_start;
        let diagonal_tile = task.row_start == task.col_start;
        for local_col in 0..n_col {
            let global_col = task.col_start + local_col;
            let row_begin = if diagonal_tile { local_col } else { 0usize };
            for local_row in row_begin..n_row {
                let global_row = task.row_start + local_row;
                if global_row < row_band_start || global_row >= row_band_end {
                    continue;
                }
                let accum_idx = if task_acc.accum_row_major {
                    local_row * n_col + local_col
                } else {
                    local_row + local_col * n_row
                };
                let scaled = task_acc.accum[accum_idx] * inv_scale;
                if !scaled.is_finite() {
                    return Err(format!(
                        "Dense GRM part produced non-finite value at pair ({global_row}, {global_col})"
                    ));
                }
                let out_row = global_row - row_band_start;
                out[out_row * n_use + global_col] = scaled as f32;
            }
        }
    }
    Ok(())
}

fn spgrm_collect_batch_dense_full_matrix(
    task_accum: &[SpgrmBatchTaskAccum],
    inv_scale: f64,
    n_use: usize,
    out: &mut [f32],
) -> Result<(), String> {
    let expected_len = n_use.saturating_mul(n_use);
    if out.len() != expected_len {
        return Err(format!(
            "dense GRM full output length mismatch: got {}, expected {}",
            out.len(),
            expected_len
        ));
    }
    for task_acc in task_accum {
        let task = task_acc.task;
        let n_col = task.col_end - task.col_start;
        let n_row = task.row_end - task.row_start;
        let diagonal_tile = task.row_start == task.col_start;
        for local_col in 0..n_col {
            let global_col = task.col_start + local_col;
            let row_begin = if diagonal_tile { local_col } else { 0usize };
            for local_row in row_begin..n_row {
                let global_row = task.row_start + local_row;
                let accum_idx = if task_acc.accum_row_major {
                    local_row * n_col + local_col
                } else {
                    local_row + local_col * n_row
                };
                let scaled = task_acc.accum[accum_idx] * inv_scale;
                if !scaled.is_finite() {
                    return Err(format!(
                        "Dense GRM full writer produced non-finite value at pair ({global_row}, {global_col})"
                    ));
                }
                let scaled_f32 = scaled as f32;
                out[global_row * n_use + global_col] = scaled_f32;
                out[global_col * n_use + global_row] = scaled_f32;
            }
        }
    }
    Ok(())
}

fn write_sorted_entries_to_jxgrm(
    out_path: &str,
    n_samples: usize,
    entries: &mut Vec<SpgrmEntry>,
) -> Result<(usize, usize), String> {
    entries.sort_unstable_by(spgrm_entry_cmp);
    let nnz = spgrm_compress_sorted_entries(entries);
    let mut writer = SpgrmCscStreamWriter::new(out_path, n_samples, nnz)?;
    for &entry in entries.iter() {
        writer.push(entry)?;
    }
    writer.finish()?;
    Ok((n_samples, nnz))
}

fn merge_chunked_entries_to_jxgrm_in_memory(
    out_path: &str,
    n_samples: usize,
    chunk_paths: &[PathBuf],
    nnz: usize,
) -> Result<(usize, usize), String> {
    let mut merged = Vec::<SpgrmEntry>::with_capacity(nnz);
    for_each_merged_chunk_entry(chunk_paths, |entry| {
        merged.push(entry);
        Ok(())
    })?;
    let merged_nnz = merged.len();
    let mut writer = SpgrmCscStreamWriter::new(out_path, n_samples, merged_nnz)?;
    for entry in merged {
        writer.push(entry)?;
    }
    writer.finish()?;
    Ok((n_samples, merged_nnz))
}

fn merge_chunked_entries_to_jxgrm_streaming(
    out_path: &str,
    n_samples: usize,
    chunk_paths: &[PathBuf],
) -> Result<(usize, usize), String> {
    let mut writer = SpgrmCscTempPayloadWriter::new(out_path, n_samples)?;
    for_each_merged_chunk_entry(chunk_paths, |entry| writer.push(entry))?;
    writer.finish()
}

fn merge_chunked_entries_to_jxgrm(
    out_path: &str,
    n_samples: usize,
    chunk_paths: &[PathBuf],
    nnz: usize,
) -> Result<(usize, usize), String> {
    if spgrm_should_use_streaming_merge(chunk_paths, nnz) {
        merge_chunked_entries_to_jxgrm_streaming(out_path, n_samples, chunk_paths)
    } else {
        merge_chunked_entries_to_jxgrm_in_memory(out_path, n_samples, chunk_paths, nnz)
    }
}

#[inline]
fn validate_spgrm_inputs(
    packed_flat: &[u8],
    n_samples_full: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    method: usize,
    threshold: f64,
) -> Result<(usize, usize, usize, bool), String> {
    if n_samples_full == 0 {
        return Err("Sparse GRM requires n_samples > 0".to_string());
    }
    if n_samples_full > (u32::MAX as usize) {
        return Err(format!(
            "Sparse GRM row indices are stored as u32; got n_samples={n_samples_full} > {}",
            u32::MAX
        ));
    }
    if method != 1 && method != 2 {
        return Err(format!(
            "Sparse GRM method must be 1 (centered) or 2 (standardized); got {method}"
        ));
    }
    if !threshold.is_finite() {
        return Err("Sparse GRM threshold must be finite".to_string());
    }
    if sample_idx.is_empty() {
        return Err("Sparse GRM sample_indices must not be empty".to_string());
    }
    if let Some(&bad) = sample_idx.iter().find(|&&sid| sid >= n_samples_full) {
        return Err(format!(
            "Sparse GRM sample index out of range: {bad} >= {n_samples_full}"
        ));
    }
    let m = row_flip.len();
    if m == 0 {
        return Err("Sparse GRM requires at least one SNP row".to_string());
    }
    if row_maf.len() != m {
        return Err(format!(
            "Sparse GRM row_maf length mismatch: got {}, expected {m}",
            row_maf.len()
        ));
    }
    let bytes_per_snp = n_samples_full.div_ceil(4);
    let expected_len = m
        .checked_mul(bytes_per_snp)
        .ok_or_else(|| "Sparse GRM packed length overflow".to_string())?;
    if packed_flat.len() != expected_len {
        return Err(format!(
            "Sparse GRM packed length mismatch: got {}, expected {expected_len}",
            packed_flat.len()
        ));
    }
    let n_use = sample_idx.len();
    let full_sample_fast = is_identity_indices(sample_idx, n_samples_full);
    Ok((m, bytes_per_snp, n_use, full_sample_fast))
}

fn centered_varsum_from_packed(
    packed_flat: &[u8],
    n_samples_full: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    block_rows: usize,
    threads: usize,
) -> Result<f64, String> {
    let m = row_flip.len();
    let n_use = sample_idx.len();
    let full_sample_fast = is_identity_indices(sample_idx, n_samples_full);
    if full_sample_fast {
        let mut acc = 0.0_f64;
        for &maf in row_maf {
            let p = maf as f64;
            let v = 2.0_f64 * p * (1.0_f64 - p);
            if v.is_finite() && v > 0.0 {
                acc += v;
            }
        }
        if acc.is_finite() && acc > 0.0 {
            return Ok(acc);
        }
        return Err("Sparse GRM centered denominator is not positive".to_string());
    }

    let bytes_per_snp = n_samples_full.div_ceil(4);
    let row_step = block_rows.max(1);
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let mut block = vec![0.0_f32; row_step * n_use];
    let mut block_varsum = vec![0.0_f64; row_step];
    let mut acc = 0.0_f64;
    for row_start in (0..m).step_by(row_step) {
        let row_end = (row_start + row_step).min(m);
        let cur_rows = row_end - row_start;
        decode_grm_block(
            packed_flat,
            bytes_per_snp,
            n_samples_full,
            row_flip,
            row_maf,
            sample_idx,
            false,
            1usize,
            1e-12_f32,
            &crate::bedmath::packed_byte_lut().code4,
            row_start,
            row_end,
            n_use,
            &mut block[..cur_rows * n_use],
            &mut block_varsum[..cur_rows],
            pool.as_ref(),
        )?;
        acc += block_varsum[..cur_rows].iter().sum::<f64>();
    }
    if acc.is_finite() && acc > 0.0 {
        Ok(acc)
    } else {
        Err("Sparse GRM centered denominator is not positive".to_string())
    }
}

#[inline]
fn pair_block_accumulate_sgemm(
    left_f32: &[f32],
    right_f32: &[f32],
    rows: usize,
    n_left: usize,
    n_right: usize,
    accum: &mut [f32],
    beta: f32,
) {
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            n_left as CblasInt,
            n_right as CblasInt,
            rows as CblasInt,
            1.0_f32,
            left_f32.as_ptr(),
            n_left as CblasInt,
            right_f32.as_ptr(),
            n_right as CblasInt,
            beta,
            accum.as_mut_ptr(),
            n_left as CblasInt,
        );
    }
}

#[inline]
fn spgrm_merge_col_major_tmp_f32_into_f64(
    accum_f64: &mut [f64],
    tmp_f32: &[f32],
    n_row: usize,
    n_col: usize,
    lower_only: bool,
    is_first_block: bool,
) {
    if is_first_block {
        for col in 0..n_col {
            let row_start = if lower_only { col } else { 0usize };
            for row in row_start..n_row {
                let idx = row + col * n_row;
                accum_f64[idx] = tmp_f32[idx] as f64;
            }
        }
    } else {
        for col in 0..n_col {
            let row_start = if lower_only { col } else { 0usize };
            for row in row_start..n_row {
                let idx = row + col * n_row;
                accum_f64[idx] += tmp_f32[idx] as f64;
            }
        }
    }
}

#[inline]
fn spgrm_merge_row_major_tmp_f32_into_f64(
    accum_f64: &mut [f64],
    tmp_f32: &[f32],
    n_row: usize,
    n_col: usize,
    lower_only: bool,
    is_first_block: bool,
) {
    if is_first_block {
        for row in 0..n_row {
            let col_end = if lower_only {
                (row + 1).min(n_col)
            } else {
                n_col
            };
            for col in 0..col_end {
                let idx = row * n_col + col;
                accum_f64[idx] = tmp_f32[idx] as f64;
            }
        }
    } else {
        for row in 0..n_row {
            let col_end = if lower_only {
                (row + 1).min(n_col)
            } else {
                n_col
            };
            for col in 0..col_end {
                let idx = row * n_col + col;
                accum_f64[idx] += tmp_f32[idx] as f64;
            }
        }
    }
}

#[inline]
fn pair_block_accumulate_sgemm_tmp_into_f64(
    left_f32: &[f32],
    right_f32: &[f32],
    rows: usize,
    n_left: usize,
    n_right: usize,
    accum_f64: &mut [f64],
    tmp_f32: &mut [f32],
    is_first_block: bool,
) {
    let tmp = &mut tmp_f32[..n_left.saturating_mul(n_right)];
    tmp.fill(0.0_f32);
    pair_block_accumulate_sgemm(left_f32, right_f32, rows, n_left, n_right, tmp, 0.0_f32);
    spgrm_merge_col_major_tmp_f32_into_f64(accum_f64, tmp, n_left, n_right, false, is_first_block);
}

#[inline]
fn pair_block_accumulate_sgemm_rowmajor_strided(
    left_f32: &[f32],
    left_stride: usize,
    left_col_start: usize,
    right_f32: &[f32],
    right_stride: usize,
    right_col_start: usize,
    rows: usize,
    n_left: usize,
    n_right: usize,
    accum: &mut [f32],
    beta: f32,
) {
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            n_left as CblasInt,
            n_right as CblasInt,
            rows as CblasInt,
            1.0_f32,
            left_f32.as_ptr().add(left_col_start),
            left_stride as CblasInt,
            right_f32.as_ptr().add(right_col_start),
            right_stride as CblasInt,
            beta,
            accum.as_mut_ptr(),
            n_right as CblasInt,
        );
    }
}

#[inline]
fn pair_block_accumulate_sgemm_rowmajor_strided_tmp_into_f64(
    left_f32: &[f32],
    left_stride: usize,
    left_col_start: usize,
    right_f32: &[f32],
    right_stride: usize,
    right_col_start: usize,
    rows: usize,
    n_left: usize,
    n_right: usize,
    accum_f64: &mut [f64],
    tmp_f32: &mut [f32],
    lower_only: bool,
    is_first_block: bool,
) {
    let tmp = &mut tmp_f32[..n_left.saturating_mul(n_right)];
    tmp.fill(0.0_f32);
    pair_block_accumulate_sgemm_rowmajor_strided(
        left_f32,
        left_stride,
        left_col_start,
        right_f32,
        right_stride,
        right_col_start,
        rows,
        n_left,
        n_right,
        tmp,
        0.0_f32,
    );
    spgrm_merge_row_major_tmp_f32_into_f64(
        accum_f64,
        tmp,
        n_left,
        n_right,
        lower_only,
        is_first_block,
    );
}

/// Runtime probe: benchmark ssyrk vs sgemm for self-block (C = A * A^T).
/// Cache result per (rows, n_cols) pair.  Falls back to sgemm on non-BLAS platforms.
fn spgrm_probe_self_accum_syrk_faster(n_cols: usize, rows: usize) -> bool {
    if rows == 0 || n_cols == 0 {
        return false;
    }
    // Small matrices: ssyrk overhead may dominate.
    if n_cols < 64 || rows < 64 {
        return false;
    }

    // Thread-safe probe cache.
    use std::collections::HashMap;
    use std::sync::Mutex;
    static PROBE_CACHE: Mutex<Option<HashMap<(usize, usize), bool>>> = Mutex::new(None);
    {
        let cache = PROBE_CACHE.lock().unwrap();
        if let Some(ref map) = *cache {
            if let Some(&v) = map.get(&(n_cols, rows)) {
                return v;
            }
        }
    }

    // Benchmark: run syrk and gemm each a few times, pick the faster.
    let probe_rows = rows.min(256); // small representative block
    let probe_n = n_cols.min(256);
    let a = vec![0.0_f32; probe_rows * probe_n];
    let mut c = vec![0.0_f32; probe_n * probe_n];

    let mut bench = |use_syrk: bool| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            c.fill(0.0);
            let t0 = std::time::Instant::now();
            unsafe {
                if use_syrk {
                    cblas_ssyrk_dispatch(
                        CBLAS_COL_MAJOR,
                        CBLAS_UPPER,
                        CBLAS_NO_TRANS,
                        probe_n as CblasInt,
                        probe_rows as CblasInt,
                        1.0_f32,
                        a.as_ptr(),
                        probe_n as CblasInt,
                        0.0_f32,
                        c.as_mut_ptr(),
                        probe_n as CblasInt,
                    );
                } else {
                    cblas_sgemm_dispatch(
                        CBLAS_COL_MAJOR,
                        CBLAS_NO_TRANS,
                        CBLAS_TRANS,
                        probe_n as CblasInt,
                        probe_n as CblasInt,
                        probe_rows as CblasInt,
                        1.0_f32,
                        a.as_ptr(),
                        probe_n as CblasInt,
                        a.as_ptr(),
                        probe_n as CblasInt,
                        0.0_f32,
                        c.as_mut_ptr(),
                        probe_n as CblasInt,
                    );
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            if dt < best {
                best = dt;
            }
        }
        best
    };

    let syrk_s = bench(true);
    let gemm_s = bench(false);
    let faster = syrk_s < gemm_s;

    let mut cache = PROBE_CACHE.lock().unwrap();
    cache
        .get_or_insert_with(HashMap::new)
        .insert((n_cols, rows), faster);
    faster
}

#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn self_block_accumulate_ssyrk(
    block_f32: &[f32],
    rows: usize,
    n_cols: usize,
    accum: &mut [f32],
    beta: f32,
) {
    unsafe {
        cblas_ssyrk_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_LOWER,
            CBLAS_NO_TRANS,
            n_cols as CblasInt,
            rows as CblasInt,
            1.0_f32,
            block_f32.as_ptr(),
            n_cols as CblasInt,
            beta,
            accum.as_mut_ptr(),
            n_cols as CblasInt,
        );
    }
}

#[inline]
fn self_block_accumulate_mixed_f32_to_f64(
    block_f32: &[f32],
    rows: usize,
    n_cols: usize,
    accum_f64: &mut [f64],
    tmp_f32: &mut [f32],
    is_first_block: bool,
) {
    let tmp = &mut tmp_f32[..n_cols.saturating_mul(n_cols)];
    tmp.fill(0.0_f32);
    if spgrm_probe_self_accum_syrk_faster(n_cols, rows) {
        unsafe {
            cblas_ssyrk_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_LOWER,
                CBLAS_NO_TRANS,
                n_cols as CblasInt,
                rows as CblasInt,
                1.0_f32,
                block_f32.as_ptr(),
                n_cols as CblasInt,
                0.0_f32,
                tmp.as_mut_ptr(),
                n_cols as CblasInt,
            );
        }
    } else {
        unsafe {
            cblas_sgemm_dispatch(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n_cols as CblasInt,
                n_cols as CblasInt,
                rows as CblasInt,
                1.0_f32,
                block_f32.as_ptr(),
                n_cols as CblasInt,
                block_f32.as_ptr(),
                n_cols as CblasInt,
                0.0_f32,
                tmp.as_mut_ptr(),
                n_cols as CblasInt,
            );
        }
    }
    spgrm_merge_col_major_tmp_f32_into_f64(accum_f64, tmp, n_cols, n_cols, true, is_first_block);
}

#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn self_block_accumulate_sgemm(
    block_f32: &[f32],
    rows: usize,
    n_cols: usize,
    accum: &mut [f32],
    beta: f32,
) {
    // Probe once per (n_cols, rows) pair — use ssyrk when faster.
    if spgrm_probe_self_accum_syrk_faster(n_cols, rows) {
        return self_block_accumulate_ssyrk(block_f32, rows, n_cols, accum, beta);
    }
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            n_cols as CblasInt,
            n_cols as CblasInt,
            rows as CblasInt,
            1.0_f32,
            block_f32.as_ptr(),
            n_cols as CblasInt,
            block_f32.as_ptr(),
            n_cols as CblasInt,
            beta,
            accum.as_mut_ptr(),
            n_cols as CblasInt,
        );
    }
}

#[inline]
fn spgrm_accumulate_task_gemm(
    task_acc: &mut SpgrmBatchTaskAccum,
    stripe_scratch: &[SpgrmStreamStripeScratch],
    cur_rows: usize,
    is_first_block: bool,
    timing: Option<&SpgrmStageTiming>,
) {
    let row_block = &stripe_scratch[task_acc.row_stripe_idx];
    let col_block = &stripe_scratch[task_acc.col_stripe_idx];
    let row_stride = row_block.width();
    let col_stride = col_block.width();
    let n_row = task_acc.task.row_end - task_acc.task.row_start;
    let n_col = task_acc.task.col_end - task_acc.task.col_start;
    let diagonal_tile = task_acc.task.row_start == task_acc.task.col_start;
    let t_gemm = Instant::now();
    if task_acc.accum_row_major {
        pair_block_accumulate_sgemm_rowmajor_strided_tmp_into_f64(
            &row_block.decoded[..cur_rows * row_stride],
            row_stride,
            task_acc.row_offset_in_stripe,
            &col_block.decoded[..cur_rows * col_stride],
            col_stride,
            task_acc.col_offset_in_stripe,
            cur_rows,
            n_row,
            n_col,
            task_acc.accum.as_mut_slice(),
            task_acc.accum_tmp_f32.as_mut_slice(),
            diagonal_tile,
            is_first_block,
        );
    } else if task_acc.row_stripe_idx == task_acc.col_stripe_idx {
        self_block_accumulate_mixed_f32_to_f64(
            &row_block.decoded[..cur_rows * n_row],
            cur_rows,
            n_row,
            task_acc.accum.as_mut_slice(),
            task_acc.accum_tmp_f32.as_mut_slice(),
            is_first_block,
        );
    } else {
        pair_block_accumulate_sgemm_tmp_into_f64(
            &row_block.decoded[..cur_rows * n_row],
            &col_block.decoded[..cur_rows * n_col],
            cur_rows,
            n_row,
            n_col,
            task_acc.accum.as_mut_slice(),
            task_acc.accum_tmp_f32.as_mut_slice(),
            is_first_block,
        );
    }
    if let Some(timing) = timing {
        timing.add_gemm_ns(spgrm_elapsed_ns(t_gemm));
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_spgrm_task_entries(
    scratch: &mut SpgrmWorkerScratch,
    task: SpgrmTask,
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples_full: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    method: usize,
    row_step: usize,
    global_full_fast: bool,
    inv_scale: f64,
    threshold: f64,
    abs_threshold: bool,
    m: usize,
    pool: Option<&std::sync::Arc<rayon::ThreadPool>>,
    timing: Option<&SpgrmStageTiming>,
) -> Result<Vec<SpgrmEntry>, String> {
    let col_idx = &sample_idx[task.col_start..task.col_end];
    let row_idx = &sample_idx[task.row_start..task.row_end];
    let n_col = col_idx.len();
    let n_row = row_idx.len();
    let diagonal_tile = task.row_start == task.col_start;
    let row_full_fast = global_full_fast && task.row_start == 0 && task.row_end == n_samples_full;
    let col_full_fast = global_full_fast && task.col_start == 0 && task.col_end == n_samples_full;

    let accum_len = n_row.saturating_mul(n_col);
    let mut decode_ns = 0u64;
    let mut gemm_ns = 0u64;
    let mut first_block = true;
    for snp_start in (0..m).step_by(row_step) {
        let snp_end = (snp_start + row_step).min(m);
        let cur_rows = snp_end - snp_start;
        let t_decode = Instant::now();
        decode_grm_block(
            packed_flat,
            bytes_per_snp,
            n_samples_full,
            row_flip,
            row_maf,
            row_idx,
            row_full_fast,
            method,
            1e-12_f32,
            &crate::bedmath::packed_byte_lut().code4,
            snp_start,
            snp_end,
            n_row,
            &mut scratch.row_block_f32[..cur_rows * n_row],
            &mut scratch.dummy_row_varsum[..cur_rows],
            pool,
        )?;
        decode_ns = decode_ns.saturating_add(spgrm_elapsed_ns(t_decode));
        if diagonal_tile {
            let t_gemm = Instant::now();
            self_block_accumulate_mixed_f32_to_f64(
                &scratch.row_block_f32[..cur_rows * n_row],
                cur_rows,
                n_row,
                &mut scratch.accum_f64[..accum_len],
                &mut scratch.accum_tmp_f32[..accum_len],
                first_block,
            );
            gemm_ns = gemm_ns.saturating_add(spgrm_elapsed_ns(t_gemm));
        } else {
            let t_decode = Instant::now();
            decode_grm_block(
                packed_flat,
                bytes_per_snp,
                n_samples_full,
                row_flip,
                row_maf,
                col_idx,
                col_full_fast,
                method,
                1e-12_f32,
                &crate::bedmath::packed_byte_lut().code4,
                snp_start,
                snp_end,
                n_col,
                &mut scratch.col_block_f32[..cur_rows * n_col],
                &mut scratch.dummy_col_varsum[..cur_rows],
                pool,
            )?;
            decode_ns = decode_ns.saturating_add(spgrm_elapsed_ns(t_decode));
            let t_gemm = Instant::now();
            pair_block_accumulate_sgemm_tmp_into_f64(
                &scratch.row_block_f32[..cur_rows * n_row],
                &scratch.col_block_f32[..cur_rows * n_col],
                cur_rows,
                n_row,
                n_col,
                &mut scratch.accum_f64[..accum_len],
                &mut scratch.accum_tmp_f32[..accum_len],
                first_block,
            );
            gemm_ns = gemm_ns.saturating_add(spgrm_elapsed_ns(t_gemm));
        }
        first_block = false;
    }

    let t_threshold = Instant::now();
    let mut out = Vec::<SpgrmEntry>::with_capacity(accum_len / 4 + n_col.max(1));
    for local_col in 0..n_col {
        let global_col = task.col_start + local_col;
        let row_begin = if diagonal_tile { local_col } else { 0usize };
        for local_row in row_begin..n_row {
            let global_row = task.row_start + local_row;
            let scaled = scratch.accum_f64[local_row + local_col * n_row] * inv_scale;
            if !scaled.is_finite() {
                return Err(format!(
                    "Sparse GRM produced non-finite value at pair ({global_row}, {global_col})"
                ));
            }
            if global_row == global_col || spgrm_keep_value(scaled, threshold, abs_threshold) {
                out.push(SpgrmEntry {
                    col: global_col as u32,
                    row: global_row as u32,
                    value: scaled,
                });
            }
        }
    }
    let threshold_ns = spgrm_elapsed_ns(t_threshold);
    if let Some(timing) = timing {
        timing.add_decode_ns(decode_ns);
        timing.add_gemm_ns(gemm_ns);
        timing.add_threshold_ns(threshold_ns);
    }
    Ok(out)
}

#[cfg_attr(not(test), allow(dead_code))]
fn sparse_grm_coo_from_packed(
    packed_flat: &[u8],
    n_samples_full: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    method: usize,
    threshold: f64,
    abs_threshold: bool,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<SparseGrmCoo, String> {
    let build = prepare_spgrm_build(
        packed_flat,
        n_samples_full,
        row_flip,
        row_maf,
        sample_idx,
        method,
        threshold,
        abs_threshold,
        block_rows,
        sample_block,
        threads,
    )?;
    let _blas_guard = spgrm_blas_threads_for_build(&build, threads).map(BlasThreadGuard::enter);
    let notify_step = if progress_every == 0 {
        1usize
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    let mut done_pairs = 0usize;

    let mut coo_rows = Vec::<u32>::with_capacity(build.n_use.saturating_mul(4));
    let mut coo_cols = Vec::<u32>::with_capacity(build.n_use.saturating_mul(4));
    let mut coo_vals = Vec::<f64>::with_capacity(build.n_use.saturating_mul(4));
    for batch in build.task_batches.iter() {
        let task_batch = &build.tasks[batch.start..batch.end];
        let entries = compute_spgrm_task_batch_entries(
            &build,
            task_batch,
            packed_flat,
            n_samples_full,
            row_flip,
            row_maf,
            sample_idx,
            method,
            threads,
            None,
        )?;
        coo_rows.reserve(entries.len());
        coo_cols.reserve(entries.len());
        coo_vals.reserve(entries.len());
        for entry in entries {
            coo_rows.push(entry.row);
            coo_cols.push(entry.col);
            coo_vals.push(entry.value);
        }
        done_pairs = done_pairs.saturating_add(task_batch.len());
        spgrm_progress_notify(
            progress_callback,
            done_pairs,
            build.total_pairs,
            notify_step,
            &mut last_notified,
            done_pairs == build.total_pairs,
        )?;
    }

    Ok(SparseGrmCoo {
        rows: coo_rows,
        cols: coo_cols,
        values: coo_vals,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn coo_lower_to_csc(n_samples: usize, coo: SparseGrmCoo) -> Result<SparseGrmCsc, String> {
    if coo.rows.len() != coo.cols.len() || coo.rows.len() != coo.values.len() {
        return Err("Sparse GRM COO arrays have mismatched lengths".to_string());
    }
    let nnz = coo.values.len();
    let mut col_ptr = vec![0u64; n_samples + 1];
    for &col in &coo.cols {
        let c = col as usize;
        if c >= n_samples {
            return Err(format!(
                "Sparse GRM COO column index out of range: {c} >= {n_samples}"
            ));
        }
        col_ptr[c + 1] = col_ptr[c + 1].saturating_add(1);
    }
    for c in 0..n_samples {
        col_ptr[c + 1] = col_ptr[c + 1].saturating_add(col_ptr[c]);
    }
    let mut next = col_ptr.clone();
    let mut row_indices = vec![0u32; nnz];
    let mut values = vec![0.0_f64; nnz];
    for idx in 0..nnz {
        let row = coo.rows[idx] as usize;
        let col = coo.cols[idx] as usize;
        if row >= n_samples {
            return Err(format!(
                "Sparse GRM COO row index out of range: {row} >= {n_samples}"
            ));
        }
        if row < col {
            return Err(format!(
                "Sparse GRM expects lower-triangle COO ordering; got row={row} < col={col}"
            ));
        }
        let dst = next[col] as usize;
        row_indices[dst] = row as u32;
        values[dst] = coo.values[idx];
        next[col] = next[col].saturating_add(1);
    }
    Ok(SparseGrmCsc {
        n_samples,
        nnz,
        col_ptr,
        row_indices,
        values,
    })
}

#[cfg(target_endian = "little")]
fn write_u64_slice<W: Write>(writer: &mut W, data: &[u64]) -> Result<(), String> {
    let byte_len = data.len().saturating_mul(std::mem::size_of::<u64>());
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
    writer
        .write_all(bytes)
        .map_err(|e| format!("write u64 payload failed: {e}"))
}

#[cfg(not(target_endian = "little"))]
fn write_u64_slice<W: Write>(writer: &mut W, data: &[u64]) -> Result<(), String> {
    for &v in data {
        writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| format!("write u64 payload failed: {e}"))?;
    }
    Ok(())
}

#[cfg(target_endian = "little")]
#[cfg_attr(not(test), allow(dead_code))]
fn write_u32_slice<W: Write>(writer: &mut W, data: &[u32]) -> Result<(), String> {
    let byte_len = data.len().saturating_mul(std::mem::size_of::<u32>());
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
    writer
        .write_all(bytes)
        .map_err(|e| format!("write u32 payload failed: {e}"))
}

#[cfg(not(target_endian = "little"))]
fn write_u32_slice<W: Write>(writer: &mut W, data: &[u32]) -> Result<(), String> {
    for &v in data {
        writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| format!("write u32 payload failed: {e}"))?;
    }
    Ok(())
}

#[cfg(target_endian = "little")]
#[cfg_attr(not(test), allow(dead_code))]
fn write_f64_slice<W: Write>(writer: &mut W, data: &[f64]) -> Result<(), String> {
    let byte_len = data.len().saturating_mul(std::mem::size_of::<f64>());
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
    writer
        .write_all(bytes)
        .map_err(|e| format!("write f64 payload failed: {e}"))
}

#[cfg(not(target_endian = "little"))]
fn write_f64_slice<W: Write>(writer: &mut W, data: &[f64]) -> Result<(), String> {
    for &v in data {
        writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| format!("write f64 payload failed: {e}"))?;
    }
    Ok(())
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn write_sparse_grm_csc(path: &str, csc: &SparseGrmCsc) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("create {path} failed: {e}"))?;
    let mut writer = BufWriter::with_capacity(SPGRM_DEFAULT_WRITE_BUF_BYTES, file);
    writer
        .write_all(&(csc.n_samples as u64).to_le_bytes())
        .map_err(|e| format!("write sparse GRM header(n) failed: {e}"))?;
    writer
        .write_all(&(csc.nnz as u64).to_le_bytes())
        .map_err(|e| format!("write sparse GRM header(nnz) failed: {e}"))?;
    write_u64_slice(&mut writer, csc.col_ptr.as_slice())?;
    write_u32_slice(&mut writer, csc.row_indices.as_slice())?;
    let pad_bytes = spgrm_jxgrm_values_padding_bytes(csc.nnz)?;
    if pad_bytes > 0 {
        writer
            .write_all(&[0u8; 8][..pad_bytes])
            .map_err(|e| format!("write sparse GRM alignment padding failed: {e}"))?;
    }
    write_f64_slice(&mut writer, csc.values.as_slice())?;
    writer
        .flush()
        .map_err(|e| format!("flush sparse GRM file failed: {e}"))?;
    Ok(())
}

pub fn spgrm_packed_to_jxgrm_core(
    packed_flat: &[u8],
    n_samples_full: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    out_prefix: &str,
    method: usize,
    threshold: f64,
    abs_threshold: bool,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<(String, usize, usize), String> {
    let build = prepare_spgrm_build(
        packed_flat,
        n_samples_full,
        row_flip,
        row_maf,
        sample_idx,
        method,
        threshold,
        abs_threshold,
        block_rows,
        sample_block,
        threads,
    )?;
    let out_path = normalize_spgrm_path(out_prefix);
    if out_path.is_empty() {
        return Err("Sparse GRM output prefix must not be empty".to_string());
    }
    let spill_nnz_limit = spgrm_spill_nnz_limit().max(build.n_use.max(1));
    let timing = if spgrm_stage_timing_enabled() {
        Some(Arc::new(SpgrmStageTiming::default()))
    } else {
        None
    };
    let notify_step = if progress_every == 0 {
        1usize
    } else {
        progress_every.max(1)
    };
    let _blas_guard = spgrm_blas_threads_for_build(&build, threads).map(BlasThreadGuard::enter);
    let mut chunk_paths = Vec::<PathBuf>::new();
    let write_result = (|| -> Result<(usize, usize), String> {
        let mut last_notified = 0usize;
        let mut done_pairs = 0usize;
        let mut chunk_idx = 0usize;
        let mut nnz_total = 0usize;
        let mut buffer = Vec::<SpgrmEntry>::new();

        for batch in build.task_batches.iter() {
            let task_batch = &build.tasks[batch.start..batch.end];
            let mut entries = compute_spgrm_task_batch_entries(
                &build,
                task_batch,
                packed_flat,
                n_samples_full,
                row_flip,
                row_maf,
                sample_idx,
                method,
                threads,
                timing.clone(),
            )?;
            nnz_total = nnz_total.saturating_add(entries.len());
            buffer.append(&mut entries);
            if buffer.len() >= spill_nnz_limit {
                let chunk_path = spgrm_chunk_path(&out_path, chunk_idx);
                let t_spill = Instant::now();
                let _ = spill_sorted_entries_to_chunk(&chunk_path, &mut buffer)?;
                if let Some(timing) = timing.as_deref() {
                    timing.add_spill_ns(spgrm_elapsed_ns(t_spill));
                }
                chunk_paths.push(chunk_path);
                chunk_idx = chunk_idx.saturating_add(1);
            }
            done_pairs = done_pairs.saturating_add(task_batch.len());
            spgrm_progress_notify(
                progress_callback,
                done_pairs,
                build.total_pairs,
                notify_step,
                &mut last_notified,
                done_pairs == build.total_pairs,
            )?;
        }

        if chunk_paths.is_empty() {
            let t_spill = Instant::now();
            let out = write_sorted_entries_to_jxgrm(&out_path, build.n_use, &mut buffer);
            if let Some(timing) = timing.as_deref() {
                timing.add_spill_ns(spgrm_elapsed_ns(t_spill));
            }
            out
        } else {
            if !buffer.is_empty() {
                let chunk_path = spgrm_chunk_path(&out_path, chunk_idx);
                let t_spill = Instant::now();
                let _ = spill_sorted_entries_to_chunk(&chunk_path, &mut buffer)?;
                if let Some(timing) = timing.as_deref() {
                    timing.add_spill_ns(spgrm_elapsed_ns(t_spill));
                }
                chunk_paths.push(chunk_path);
            }
            let t_spill = Instant::now();
            let out = merge_chunked_entries_to_jxgrm(
                &out_path,
                build.n_use,
                chunk_paths.as_slice(),
                nnz_total,
            );
            if let Some(timing) = timing.as_deref() {
                timing.add_spill_ns(spgrm_elapsed_ns(t_spill));
            }
            out
        }
    })();
    for chunk_path in chunk_paths {
        let _ = std::fs::remove_file(&chunk_path);
    }
    if write_result.is_err() {
        let _ = std::fs::remove_file(&out_path);
    }
    let (n_samples, nnz) = write_result?;
    if let Some(timing) = timing.as_deref() {
        spgrm_emit_timing_summary(
            "packed",
            build.n_use,
            build.m,
            build.sample_step,
            build.row_step,
            timing,
        );
    }
    Ok((out_path, n_samples, nnz))
}

#[allow(clippy::too_many_arguments)]
fn spgrm_stream_bed_to_jxgrm_core(
    bed_prefix: &str,
    sample_idx: &[usize],
    meta: crate::gfreader::PreparedBedLogicMetaOwned,
    out_prefix: &str,
    method: usize,
    threshold: f64,
    abs_threshold: bool,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_done_offset: usize,
) -> Result<(String, usize, usize), String> {
    let n_samples_full = meta.n_samples;
    if n_samples_full == 0 {
        return Err("Sparse GRM requires n_samples > 0".to_string());
    }
    if n_samples_full > (u32::MAX as usize) {
        return Err(format!(
            "Sparse GRM row indices are stored as u32; got n_samples={n_samples_full} > {}",
            u32::MAX
        ));
    }
    if method != 1 && method != 2 {
        return Err(format!(
            "Sparse GRM method must be 1 (centered) or 2 (standardized); got {method}"
        ));
    }
    if !threshold.is_finite() {
        return Err("Sparse GRM threshold must be finite".to_string());
    }
    if sample_idx.is_empty() {
        return Err("Sparse GRM sample_indices must not be empty".to_string());
    }
    if let Some(&bad) = sample_idx.iter().find(|&&sid| sid >= n_samples_full) {
        return Err(format!(
            "Sparse GRM sample index out of range: {bad} >= {n_samples_full}"
        ));
    }
    let m = meta.row_source_indices.len();
    if m == 0 {
        return Err("Sparse GRM requires at least one SNP row".to_string());
    }
    if meta.row_flip.len() != m || meta.maf.len() != m {
        return Err("Sparse GRM metadata length mismatch".to_string());
    }
    let n_use = sample_idx.len();
    let stream_double_buffer = spgrm_stream_double_buffer_requested(threads);
    let decode_buffers = if stream_double_buffer { 2usize } else { 1usize };
    let memory_plan = spgrm_resolve_stream_memory_plan(
        m,
        n_use,
        n_samples_full,
        block_rows,
        sample_block,
        threads,
        decode_buffers,
    );
    let sample_step = memory_plan.sample_step;
    let row_step = memory_plan.row_step;
    let denom = if method == 1 {
        let acc = meta
            .maf
            .iter()
            .map(|&maf| {
                let p = maf as f64;
                2.0_f64 * p * (1.0_f64 - p)
            })
            .filter(|v| v.is_finite() && *v > 0.0)
            .sum::<f64>();
        if !(acc.is_finite() && acc > 0.0) {
            return Err("Sparse GRM centered denominator is not positive".to_string());
        }
        acc
    } else {
        m as f64
    };
    let inv_scale = 1.0_f64 / denom;
    let tasks = build_spgrm_tasks(n_use, sample_step);
    let task_batches = build_spgrm_task_batches(
        tasks.as_slice(),
        row_step,
        spgrm_batch_limits(
            row_step,
            sample_step,
            memory_plan.max_decoded_bytes,
            threads,
        ),
    );
    let total_batches = task_batches.len().max(1);
    let total_progress = progress_done_offset
        .saturating_add(total_batches.saturating_mul(m))
        .max(1);
    let notify_step = if progress_every == 0 {
        row_step.max(1)
    } else {
        progress_every.max(1)
    };
    let spill_nnz_limit = spgrm_spill_nnz_limit().max(n_use.max(1));
    let timing = if spgrm_stage_timing_enabled() {
        Some(Arc::new(SpgrmStageTiming::default()))
    } else {
        None
    };
    let out_path = normalize_spgrm_path(out_prefix);
    if out_path.is_empty() {
        return Err("Sparse GRM output prefix must not be empty".to_string());
    }
    let (row_mean, row_inv_sd) = spgrm_row_mean_and_inv_sd(meta.maf.as_slice(), method);
    let mut bed_window = if let Some(window_mb) = mmap_window_mb {
        if window_mb == 0 {
            return Err("Sparse GRM mmap_window_mb must be > 0".to_string());
        }
        Some(WindowedBedMatrix::open(bed_prefix, window_mb)?)
    } else {
        None
    };
    let full_mmap = if bed_window.is_none() {
        Some(spgrm_open_bed_payload_mmap(
            bed_prefix,
            meta.n_samples,
            meta.n_snps_total,
            meta.bytes_per_snp,
        )?)
    } else {
        None
    };
    let code4_lut = &packed_byte_lut().code4;
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let use_double_buffer = stream_double_buffer && m > row_step;
    let stream_split_single_task =
        spgrm_single_diag_task_subtile_step(tasks.as_slice(), threads).is_some();
    let blas_threads = if threads <= 1 {
        None
    } else if tasks.len() > 1 || stream_split_single_task {
        Some(1usize)
    } else {
        Some(threads.max(1))
    };
    let _blas_guard = blas_threads.map(BlasThreadGuard::enter);
    let mut chunk_paths = Vec::<PathBuf>::new();
    let write_result = (|| -> Result<(usize, usize), String> {
        let mut last_notified = 0usize;
        let mut chunk_idx = 0usize;
        let mut nnz_total = 0usize;
        let mut buffer = Vec::<SpgrmEntry>::new();

        spgrm_progress_notify(
            progress_callback,
            progress_done_offset,
            total_progress,
            notify_step,
            &mut last_notified,
            true,
        )?;

        for (batch_idx, batch) in task_batches.iter().enumerate() {
            let task_batch = &tasks[batch.start..batch.end];
            let (stripe_template, mut task_accum) =
                spgrm_build_stream_batch(task_batch, row_step, sample_idx, n_samples_full, threads);

            if use_double_buffer {
                let decode_error = Arc::new(Mutex::new(None::<String>));
                let producer_error = decode_error.clone();
                let mut next_row_start = 0usize;
                let mut rel_row_indices = Vec::<usize>::new();
                let make_buffer = || SpgrmDecodedBatchBuffer {
                    stripe_scratch: stripe_template.clone(),
                    row_start: 0usize,
                    cur_rows: 0usize,
                };
                let producer = |buf: &mut SpgrmDecodedBatchBuffer| -> bool {
                    if next_row_start >= m {
                        buf.row_start = 0usize;
                        buf.cur_rows = 0usize;
                        return false;
                    }
                    let row_start = next_row_start;
                    match spgrm_decode_stream_batch_buffer(
                        buf,
                        row_start,
                        row_step,
                        m,
                        sample_idx,
                        &meta,
                        row_mean.as_slice(),
                        row_inv_sd.as_slice(),
                        bed_window.as_mut(),
                        full_mmap.as_ref(),
                        &mut rel_row_indices,
                        code4_lut,
                        pool.as_ref(),
                        timing.as_deref(),
                    ) {
                        Ok(()) => {
                            next_row_start = row_start + buf.cur_rows;
                            next_row_start < m
                        }
                        Err(err) => {
                            *producer_error.lock().unwrap() = Some(err);
                            buf.row_start = usize::MAX;
                            buf.cur_rows = 0usize;
                            false
                        }
                    }
                };
                let consumer_error = decode_error.clone();
                let consumer = |buf: &mut SpgrmDecodedBatchBuffer| -> Result<(), String> {
                    if buf.cur_rows == 0 {
                        if buf.row_start == usize::MAX {
                            return Err(consumer_error
                                .lock()
                                .unwrap()
                                .take()
                                .unwrap_or_else(|| "Sparse GRM decode failed".to_string()));
                        }
                        return Ok(());
                    }
                    spgrm_consume_stream_batch_buffer(
                        buf,
                        task_accum.as_mut_slice(),
                        batch_idx,
                        m,
                        progress_done_offset,
                        total_progress,
                        notify_step,
                        &mut last_notified,
                        progress_callback,
                        threads,
                        pool.as_ref(),
                        timing.as_deref(),
                    )
                };
                run_double_buffer(2usize, make_buffer, producer, consumer)?;
                let decode_err = decode_error.lock().unwrap().take();
                if let Some(err) = decode_err {
                    return Err(err);
                }
            } else {
                let mut decoded = SpgrmDecodedBatchBuffer {
                    stripe_scratch: stripe_template,
                    row_start: 0usize,
                    cur_rows: 0usize,
                };
                let mut rel_row_indices = Vec::<usize>::new();
                for row_start in (0..m).step_by(row_step) {
                    spgrm_decode_stream_batch_buffer(
                        &mut decoded,
                        row_start,
                        row_step,
                        m,
                        sample_idx,
                        &meta,
                        row_mean.as_slice(),
                        row_inv_sd.as_slice(),
                        bed_window.as_mut(),
                        full_mmap.as_ref(),
                        &mut rel_row_indices,
                        code4_lut,
                        pool.as_ref(),
                        timing.as_deref(),
                    )?;
                    spgrm_consume_stream_batch_buffer(
                        &decoded,
                        task_accum.as_mut_slice(),
                        batch_idx,
                        m,
                        progress_done_offset,
                        total_progress,
                        notify_step,
                        &mut last_notified,
                        progress_callback,
                        threads,
                        pool.as_ref(),
                        timing.as_deref(),
                    )?;
                }
            }

            let t_threshold = Instant::now();
            let mut entries = spgrm_collect_batch_entries(
                task_accum.as_slice(),
                inv_scale,
                threshold,
                abs_threshold,
            )?;
            if let Some(timing) = timing.as_deref() {
                timing.add_threshold_ns(spgrm_elapsed_ns(t_threshold));
            }
            nnz_total = nnz_total.saturating_add(entries.len());
            buffer.append(&mut entries);
            if buffer.len() >= spill_nnz_limit {
                let chunk_path = spgrm_chunk_path(&out_path, chunk_idx);
                let t_spill = Instant::now();
                let _ = spill_sorted_entries_to_chunk(&chunk_path, &mut buffer)?;
                if let Some(timing) = timing.as_deref() {
                    timing.add_spill_ns(spgrm_elapsed_ns(t_spill));
                }
                chunk_paths.push(chunk_path);
                chunk_idx = chunk_idx.saturating_add(1);
            }
        }

        let write_result = if chunk_paths.is_empty() {
            let t_spill = Instant::now();
            let out = write_sorted_entries_to_jxgrm(&out_path, n_use, &mut buffer);
            if let Some(timing) = timing.as_deref() {
                timing.add_spill_ns(spgrm_elapsed_ns(t_spill));
            }
            out
        } else {
            if !buffer.is_empty() {
                let chunk_path = spgrm_chunk_path(&out_path, chunk_idx);
                let t_spill = Instant::now();
                let _ = spill_sorted_entries_to_chunk(&chunk_path, &mut buffer)?;
                if let Some(timing) = timing.as_deref() {
                    timing.add_spill_ns(spgrm_elapsed_ns(t_spill));
                }
                chunk_paths.push(chunk_path);
            }
            let t_spill = Instant::now();
            let out =
                merge_chunked_entries_to_jxgrm(&out_path, n_use, chunk_paths.as_slice(), nnz_total);
            if let Some(timing) = timing.as_deref() {
                timing.add_spill_ns(spgrm_elapsed_ns(t_spill));
            }
            out
        }?;

        spgrm_progress_notify(
            progress_callback,
            total_progress,
            total_progress,
            notify_step,
            &mut last_notified,
            true,
        )?;
        Ok(write_result)
    })();
    for chunk_path in chunk_paths {
        let _ = std::fs::remove_file(&chunk_path);
    }
    if write_result.is_err() {
        let _ = std::fs::remove_file(&out_path);
    }
    let (n_samples, nnz) = write_result?;
    if let Some(timing) = timing.as_deref() {
        spgrm_emit_timing_summary("stream", n_use, m, sample_step, row_step, timing);
    }
    Ok((out_path, n_samples, nnz))
}

#[allow(clippy::too_many_arguments)]
fn grm_stream_bed_row_band_f32_fill_core(
    bed_prefix: &str,
    sample_idx: &[usize],
    meta: &crate::gfreader::PreparedBedLogicMetaOwned,
    row_band_start: usize,
    row_band_end: usize,
    method: usize,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    out: &mut [f32],
) -> Result<(usize, usize), String> {
    let n_samples_full = meta.n_samples;
    if n_samples_full == 0 {
        return Err("Dense GRM part requires n_samples > 0".to_string());
    }
    if method != 1 && method != 2 {
        return Err(format!(
            "Dense GRM part method must be 1 (centered) or 2 (standardized); got {method}"
        ));
    }
    if sample_idx.is_empty() {
        return Err("Dense GRM part sample_indices must not be empty".to_string());
    }
    if let Some(&bad) = sample_idx.iter().find(|&&sid| sid >= n_samples_full) {
        return Err(format!(
            "Dense GRM part sample index out of range: {bad} >= {n_samples_full}"
        ));
    }
    let m = meta.row_source_indices.len();
    if m == 0 {
        return Err("Dense GRM part requires at least one SNP row".to_string());
    }
    if meta.row_flip.len() != m || meta.maf.len() != m {
        return Err("Dense GRM part metadata length mismatch".to_string());
    }
    let n_use = sample_idx.len();
    if row_band_start >= row_band_end || row_band_end > n_use {
        return Err(format!(
            "Dense GRM part row band is invalid: start={}, end={}, n_samples={n_use}",
            row_band_start, row_band_end
        ));
    }

    let stream_double_buffer = spgrm_stream_double_buffer_requested(threads);
    let decode_buffers = if stream_double_buffer { 2usize } else { 1usize };
    let memory_plan = spgrm_resolve_stream_memory_plan(
        m,
        n_use,
        n_samples_full,
        block_rows,
        sample_block,
        threads,
        decode_buffers,
    );
    let row_band_len = row_band_end - row_band_start;
    let sample_step = memory_plan.sample_step.max(1);
    let row_step = memory_plan.row_step;
    let denom = if method == 1 {
        let acc = meta
            .maf
            .iter()
            .map(|&maf| {
                let p = maf as f64;
                2.0_f64 * p * (1.0_f64 - p)
            })
            .filter(|v| v.is_finite() && *v > 0.0)
            .sum::<f64>();
        if !(acc.is_finite() && acc > 0.0) {
            return Err("Dense GRM part centered denominator is not positive".to_string());
        }
        acc
    } else {
        m as f64
    };
    let inv_scale = 1.0_f64 / denom;
    let tasks = build_spgrm_row_band_tasks(n_use, sample_step, row_band_start, row_band_end);
    if tasks.is_empty() {
        return Err("Dense GRM part row band resolved no tasks".to_string());
    }
    let task_batches = build_spgrm_task_batches(
        tasks.as_slice(),
        row_step,
        spgrm_batch_limits(
            row_step,
            sample_step,
            memory_plan.max_decoded_bytes,
            threads,
        ),
    );
    let total_batches = task_batches.len().max(1);
    let total_progress = total_batches.saturating_mul(m).max(1);
    let notify_step = if progress_every == 0 {
        row_step.max(1)
    } else {
        progress_every.max(1)
    };
    let timing = if spgrm_stage_timing_enabled() {
        Some(Arc::new(SpgrmStageTiming::default()))
    } else {
        None
    };
    let (row_mean, row_inv_sd) = spgrm_row_mean_and_inv_sd(meta.maf.as_slice(), method);
    let mut bed_window = if let Some(window_mb) = mmap_window_mb {
        if window_mb == 0 {
            return Err("Dense GRM part mmap_window_mb must be > 0".to_string());
        }
        Some(WindowedBedMatrix::open(bed_prefix, window_mb)?)
    } else {
        None
    };
    let full_mmap = if bed_window.is_none() {
        Some(spgrm_open_bed_payload_mmap(
            bed_prefix,
            meta.n_samples,
            meta.n_snps_total,
            meta.bytes_per_snp,
        )?)
    } else {
        None
    };
    let code4_lut = &packed_byte_lut().code4;
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let use_double_buffer = stream_double_buffer && m > row_step;
    let stream_split_single_task =
        spgrm_single_diag_task_subtile_step(tasks.as_slice(), threads).is_some();
    let blas_threads = if threads <= 1 {
        None
    } else if tasks.len() > 1 || stream_split_single_task {
        Some(1usize)
    } else {
        Some(threads.max(1))
    };
    let _blas_guard = blas_threads.map(BlasThreadGuard::enter);
    let expected_len = row_band_len.saturating_mul(n_use);
    if out.len() != expected_len {
        return Err(format!(
            "dense GRM part output length mismatch: got {}, expected {}",
            out.len(),
            expected_len
        ));
    }
    out.fill(0.0_f32);
    let mut last_notified = 0usize;

    spgrm_progress_notify(
        progress_callback,
        0usize,
        total_progress,
        notify_step,
        &mut last_notified,
        true,
    )?;

    for (batch_idx, batch) in task_batches.iter().enumerate() {
        let task_batch = &tasks[batch.start..batch.end];
        let (stripe_template, mut task_accum) =
            spgrm_build_stream_batch(task_batch, row_step, sample_idx, n_samples_full, threads);

        if use_double_buffer {
            let decode_error = Arc::new(Mutex::new(None::<String>));
            let producer_error = decode_error.clone();
            let mut next_row_start = 0usize;
            let mut rel_row_indices = Vec::<usize>::new();
            let make_buffer = || SpgrmDecodedBatchBuffer {
                stripe_scratch: stripe_template.clone(),
                row_start: 0usize,
                cur_rows: 0usize,
            };
            let producer = |buf: &mut SpgrmDecodedBatchBuffer| -> bool {
                if next_row_start >= m {
                    buf.row_start = 0usize;
                    buf.cur_rows = 0usize;
                    return false;
                }
                let row_start = next_row_start;
                match spgrm_decode_stream_batch_buffer(
                    buf,
                    row_start,
                    row_step,
                    m,
                    sample_idx,
                    &meta,
                    row_mean.as_slice(),
                    row_inv_sd.as_slice(),
                    bed_window.as_mut(),
                    full_mmap.as_ref(),
                    &mut rel_row_indices,
                    code4_lut,
                    pool.as_ref(),
                    timing.as_deref(),
                ) {
                    Ok(()) => {
                        next_row_start = row_start + buf.cur_rows;
                        next_row_start < m
                    }
                    Err(err) => {
                        *producer_error.lock().unwrap() = Some(err);
                        buf.row_start = usize::MAX;
                        buf.cur_rows = 0usize;
                        false
                    }
                }
            };
            let consumer_error = decode_error.clone();
            let consumer = |buf: &mut SpgrmDecodedBatchBuffer| -> Result<(), String> {
                if buf.cur_rows == 0 {
                    if buf.row_start == usize::MAX {
                        return Err(consumer_error
                            .lock()
                            .unwrap()
                            .take()
                            .unwrap_or_else(|| "Dense GRM part decode failed".to_string()));
                    }
                    return Ok(());
                }
                spgrm_consume_stream_batch_buffer(
                    buf,
                    task_accum.as_mut_slice(),
                    batch_idx,
                    m,
                    0usize,
                    total_progress,
                    notify_step,
                    &mut last_notified,
                    progress_callback,
                    threads,
                    pool.as_ref(),
                    timing.as_deref(),
                )
            };
            run_double_buffer(2usize, make_buffer, producer, consumer)?;
            let decode_err = decode_error.lock().unwrap().take();
            if let Some(err) = decode_err {
                return Err(err);
            }
        } else {
            let mut decoded = SpgrmDecodedBatchBuffer {
                stripe_scratch: stripe_template,
                row_start: 0usize,
                cur_rows: 0usize,
            };
            let mut rel_row_indices = Vec::<usize>::new();
            for row_start in (0..m).step_by(row_step) {
                spgrm_decode_stream_batch_buffer(
                    &mut decoded,
                    row_start,
                    row_step,
                    m,
                    sample_idx,
                    &meta,
                    row_mean.as_slice(),
                    row_inv_sd.as_slice(),
                    bed_window.as_mut(),
                    full_mmap.as_ref(),
                    &mut rel_row_indices,
                    code4_lut,
                    pool.as_ref(),
                    timing.as_deref(),
                )?;
                spgrm_consume_stream_batch_buffer(
                    &decoded,
                    task_accum.as_mut_slice(),
                    batch_idx,
                    m,
                    0usize,
                    total_progress,
                    notify_step,
                    &mut last_notified,
                    progress_callback,
                    threads,
                    pool.as_ref(),
                    timing.as_deref(),
                )?;
            }
        }

        spgrm_collect_batch_dense_rows(
            task_accum.as_slice(),
            row_band_start,
            row_band_end,
            inv_scale,
            n_use,
            out,
        )?;
    }

    spgrm_progress_notify(
        progress_callback,
        total_progress,
        total_progress,
        notify_step,
        &mut last_notified,
        true,
    )?;
    if let Some(timing) = timing.as_deref() {
        spgrm_emit_timing_summary("dense-row-band", n_use, m, sample_step, row_step, timing);
    }
    Ok((m, n_use))
}

#[allow(clippy::too_many_arguments)]
fn grm_stream_bed_row_band_f32_core(
    bed_prefix: &str,
    sample_idx: &[usize],
    meta: crate::gfreader::PreparedBedLogicMetaOwned,
    row_band_start: usize,
    row_band_end: usize,
    method: usize,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<(Vec<f32>, usize, usize), String> {
    let n_use = sample_idx.len();
    let row_band_len = row_band_end.saturating_sub(row_band_start);
    let mut out = vec![0.0_f32; row_band_len.saturating_mul(n_use)];
    let (eff_m, n_use_out) = grm_stream_bed_row_band_f32_fill_core(
        bed_prefix,
        sample_idx,
        &meta,
        row_band_start,
        row_band_end,
        method,
        block_rows,
        sample_block,
        threads,
        mmap_window_mb,
        progress_callback,
        progress_every,
        out.as_mut_slice(),
    )?;
    Ok((out, eff_m, n_use_out))
}

#[allow(clippy::too_many_arguments)]
fn grm_stream_bed_row_band_f32_to_npy_core(
    bed_prefix: &str,
    sample_idx: &[usize],
    meta: crate::gfreader::PreparedBedLogicMetaOwned,
    row_band_start: usize,
    row_band_end: usize,
    out_npy_path: &str,
    method: usize,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<(usize, usize), String> {
    let n_use = sample_idx.len();
    let row_band_len = row_band_end.saturating_sub(row_band_start);
    let mut writer = SpgrmNpyF32Writer::create(out_npy_path, row_band_len, n_use)?;
    {
        let out = writer.payload_mut()?;
        let expected_len = row_band_len.saturating_mul(n_use);
        if out.len() != expected_len {
            return Err(format!(
                "dense GRM part npy payload length mismatch: got {}, expected {}",
                out.len(),
                expected_len
            ));
        }
        out.fill(0.0_f32);
        let _ = grm_stream_bed_row_band_f32_fill_core(
            bed_prefix,
            sample_idx,
            &meta,
            row_band_start,
            row_band_end,
            method,
            block_rows,
            sample_block,
            threads,
            mmap_window_mb,
            progress_callback,
            progress_every,
            out,
        )?;
    }
    writer.flush()?;
    Ok((meta.row_source_indices.len(), n_use))
}

#[allow(clippy::too_many_arguments)]
fn grm_stream_bed_full_f32_to_npy_core(
    bed_prefix: &str,
    sample_idx: &[usize],
    meta: crate::gfreader::PreparedBedLogicMetaOwned,
    out_npy_path: &str,
    method: usize,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<(usize, usize), String> {
    let n_samples_full = meta.n_samples;
    if n_samples_full == 0 {
        return Err("Dense GRM full writer requires n_samples > 0".to_string());
    }
    if method != 1 && method != 2 {
        return Err(format!(
            "Dense GRM full writer method must be 1 (centered) or 2 (standardized); got {method}"
        ));
    }
    if sample_idx.is_empty() {
        return Err("Dense GRM full writer sample_indices must not be empty".to_string());
    }
    if let Some(&bad) = sample_idx.iter().find(|&&sid| sid >= n_samples_full) {
        return Err(format!(
            "Dense GRM full writer sample index out of range: {bad} >= {n_samples_full}"
        ));
    }
    let m = meta.row_source_indices.len();
    if m == 0 {
        return Err("Dense GRM full writer requires at least one SNP row".to_string());
    }
    if meta.row_flip.len() != m || meta.maf.len() != m {
        return Err("Dense GRM full writer metadata length mismatch".to_string());
    }
    let n_use = sample_idx.len();

    let stream_double_buffer = spgrm_stream_double_buffer_requested(threads);
    let decode_buffers = if stream_double_buffer { 2usize } else { 1usize };
    let memory_plan = spgrm_resolve_stream_memory_plan(
        m,
        n_use,
        n_samples_full,
        block_rows,
        sample_block,
        threads,
        decode_buffers,
    );
    let sample_step = memory_plan.sample_step.max(1);
    let row_step = memory_plan.row_step;
    let denom = if method == 1 {
        let acc = meta
            .maf
            .iter()
            .map(|&maf| {
                let p = maf as f64;
                2.0_f64 * p * (1.0_f64 - p)
            })
            .filter(|v| v.is_finite() && *v > 0.0)
            .sum::<f64>();
        if !(acc.is_finite() && acc > 0.0) {
            return Err("Dense GRM full writer centered denominator is not positive".to_string());
        }
        acc
    } else {
        m as f64
    };
    let inv_scale = 1.0_f64 / denom;
    let tasks = build_spgrm_tasks(n_use, sample_step);
    let task_batches = build_spgrm_task_batches(
        tasks.as_slice(),
        row_step,
        spgrm_batch_limits(
            row_step,
            sample_step,
            memory_plan.max_decoded_bytes,
            threads,
        ),
    );
    let total_batches = task_batches.len().max(1);
    let total_progress = total_batches.saturating_mul(m).max(1);
    let notify_step = if progress_every == 0 {
        row_step.max(1)
    } else {
        progress_every.max(1)
    };
    let timing = if spgrm_stage_timing_enabled() {
        Some(Arc::new(SpgrmStageTiming::default()))
    } else {
        None
    };
    let (row_mean, row_inv_sd) = spgrm_row_mean_and_inv_sd(meta.maf.as_slice(), method);
    let mut bed_window = if let Some(window_mb) = mmap_window_mb {
        if window_mb == 0 {
            return Err("Dense GRM full writer mmap_window_mb must be > 0".to_string());
        }
        Some(WindowedBedMatrix::open(bed_prefix, window_mb)?)
    } else {
        None
    };
    let full_mmap = if bed_window.is_none() {
        Some(spgrm_open_bed_payload_mmap(
            bed_prefix,
            meta.n_samples,
            meta.n_snps_total,
            meta.bytes_per_snp,
        )?)
    } else {
        None
    };
    let code4_lut = &packed_byte_lut().code4;
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let use_double_buffer = stream_double_buffer && m > row_step;
    let stream_split_single_task =
        spgrm_single_diag_task_subtile_step(tasks.as_slice(), threads).is_some();
    let blas_threads = if threads <= 1 {
        None
    } else if tasks.len() > 1 || stream_split_single_task {
        Some(1usize)
    } else {
        Some(threads.max(1))
    };
    let _blas_guard = blas_threads.map(BlasThreadGuard::enter);
    let mut writer = SpgrmNpyF32Writer::create(out_npy_path, n_use, n_use)?;
    {
        let out = writer.payload_mut()?;
        out.fill(0.0_f32);
        let mut last_notified = 0usize;

        spgrm_progress_notify(
            progress_callback,
            0usize,
            total_progress,
            notify_step,
            &mut last_notified,
            true,
        )?;

        for (batch_idx, batch) in task_batches.iter().enumerate() {
            let task_batch = &tasks[batch.start..batch.end];
            let (stripe_template, mut task_accum) =
                spgrm_build_stream_batch(task_batch, row_step, sample_idx, n_samples_full, threads);

            if use_double_buffer {
                let decode_error = Arc::new(Mutex::new(None::<String>));
                let producer_error = decode_error.clone();
                let mut next_row_start = 0usize;
                let mut rel_row_indices = Vec::<usize>::new();
                let make_buffer = || SpgrmDecodedBatchBuffer {
                    stripe_scratch: stripe_template.clone(),
                    row_start: 0usize,
                    cur_rows: 0usize,
                };
                let producer = |buf: &mut SpgrmDecodedBatchBuffer| -> bool {
                    if next_row_start >= m {
                        buf.row_start = 0usize;
                        buf.cur_rows = 0usize;
                        return false;
                    }
                    let row_start = next_row_start;
                    match spgrm_decode_stream_batch_buffer(
                        buf,
                        row_start,
                        row_step,
                        m,
                        sample_idx,
                        &meta,
                        row_mean.as_slice(),
                        row_inv_sd.as_slice(),
                        bed_window.as_mut(),
                        full_mmap.as_ref(),
                        &mut rel_row_indices,
                        code4_lut,
                        pool.as_ref(),
                        timing.as_deref(),
                    ) {
                        Ok(()) => {
                            next_row_start = row_start + buf.cur_rows;
                            next_row_start < m
                        }
                        Err(err) => {
                            *producer_error.lock().unwrap() = Some(err);
                            buf.row_start = usize::MAX;
                            buf.cur_rows = 0usize;
                            false
                        }
                    }
                };
                let consumer_error = decode_error.clone();
                let consumer = |buf: &mut SpgrmDecodedBatchBuffer| -> Result<(), String> {
                    if buf.cur_rows == 0 {
                        if buf.row_start == usize::MAX {
                            return Err(consumer_error.lock().unwrap().take().unwrap_or_else(
                                || "Dense GRM full writer decode failed".to_string(),
                            ));
                        }
                        return Ok(());
                    }
                    spgrm_consume_stream_batch_buffer(
                        buf,
                        task_accum.as_mut_slice(),
                        batch_idx,
                        m,
                        0usize,
                        total_progress,
                        notify_step,
                        &mut last_notified,
                        progress_callback,
                        threads,
                        pool.as_ref(),
                        timing.as_deref(),
                    )
                };
                run_double_buffer(2usize, make_buffer, producer, consumer)?;
                let decode_err = decode_error.lock().unwrap().take();
                if let Some(err) = decode_err {
                    return Err(err);
                }
            } else {
                let mut decoded = SpgrmDecodedBatchBuffer {
                    stripe_scratch: stripe_template,
                    row_start: 0usize,
                    cur_rows: 0usize,
                };
                let mut rel_row_indices = Vec::<usize>::new();
                for row_start in (0..m).step_by(row_step) {
                    spgrm_decode_stream_batch_buffer(
                        &mut decoded,
                        row_start,
                        row_step,
                        m,
                        sample_idx,
                        &meta,
                        row_mean.as_slice(),
                        row_inv_sd.as_slice(),
                        bed_window.as_mut(),
                        full_mmap.as_ref(),
                        &mut rel_row_indices,
                        code4_lut,
                        pool.as_ref(),
                        timing.as_deref(),
                    )?;
                    spgrm_consume_stream_batch_buffer(
                        &decoded,
                        task_accum.as_mut_slice(),
                        batch_idx,
                        m,
                        0usize,
                        total_progress,
                        notify_step,
                        &mut last_notified,
                        progress_callback,
                        threads,
                        pool.as_ref(),
                        timing.as_deref(),
                    )?;
                }
            }

            spgrm_collect_batch_dense_full_matrix(task_accum.as_slice(), inv_scale, n_use, out)?;
        }

        spgrm_progress_notify(
            progress_callback,
            total_progress,
            total_progress,
            notify_step,
            &mut last_notified,
            true,
        )?;
        if let Some(timing) = timing.as_deref() {
            spgrm_emit_timing_summary("dense-full", n_use, m, sample_step, row_step, timing);
        }
    }
    writer.flush()?;
    Ok((m, n_use))
}

pub fn spgrm_bed_to_jxgrm_core(
    prefix: &str,
    sample_idx: Option<&[usize]>,
    out_prefix: &str,
    method: usize,
    threshold: f64,
    abs_threshold: bool,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<(String, usize, usize), String> {
    let bed_prefix = normalize_plink_prefix_local(prefix);
    if bed_prefix.is_empty() {
        return Err("Sparse GRM BED prefix must not be empty".to_string());
    }
    let prepared = if let Some(window_mb) = mmap_window_mb {
        if !snps_only {
            prepare_bed_logic_meta_owned_for_stats_samples_sparse_windowed(
                &bed_prefix,
                maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                sample_idx,
                window_mb,
                threads,
                progress_callback,
                0usize,
                progress_every,
            )?
        } else {
            prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window(
                &bed_prefix,
                maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                sample_idx,
                true,
                mmap_window_mb,
                threads.max(1),
            )?
        }
    } else {
        prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window(
            &bed_prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            sample_idx,
            true,
            mmap_window_mb,
            threads.max(1),
        )?
    };
    let selected_idx: Cow<'_, [usize]> = match sample_idx {
        Some(idx) => Cow::Borrowed(idx),
        None => Cow::Owned((0..prepared.n_samples).collect()),
    };
    let progress_done_offset = prepared.n_snps_total;
    spgrm_stream_bed_to_jxgrm_core(
        &bed_prefix,
        selected_idx.as_ref(),
        prepared,
        out_prefix,
        method,
        threshold,
        abs_threshold,
        block_rows,
        sample_block,
        threads,
        mmap_window_mb,
        progress_callback,
        progress_every,
        progress_done_offset,
    )
}

fn spgrm_dense_f32_to_jxgrm_core(
    grm: &[f32],
    n_samples: usize,
    out_prefix: &str,
    threshold: f64,
    abs_threshold: bool,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<(String, usize, usize), String> {
    if n_samples == 0 {
        return Err("Sparse GRM dense writer requires n_samples > 0".to_string());
    }
    let expected_len = n_samples
        .checked_mul(n_samples)
        .ok_or_else(|| "Sparse GRM dense writer shape overflow".to_string())?;
    if grm.len() != expected_len {
        return Err(format!(
            "Sparse GRM dense writer length mismatch: got {}, expected {}",
            grm.len(),
            expected_len
        ));
    }
    if !threshold.is_finite() {
        return Err("Sparse GRM threshold must be finite".to_string());
    }
    let out_path = normalize_spgrm_path(out_prefix);
    if out_path.is_empty() {
        return Err("Sparse GRM output prefix must not be empty".to_string());
    }

    let mut nnz = 0usize;
    for col in 0..n_samples {
        for row in col..n_samples {
            let value = grm[row * n_samples + col] as f64;
            if !value.is_finite() {
                return Err(format!(
                    "Sparse GRM dense writer found non-finite value at pair ({row}, {col})"
                ));
            }
            if row == col || spgrm_keep_value(value, threshold, abs_threshold) {
                nnz = nnz.saturating_add(1);
            }
        }
    }

    let notify_step = if progress_every == 0 {
        (n_samples / 128).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    let mut writer = SpgrmCscStreamWriter::new(&out_path, n_samples, nnz)?;
    for col in 0..n_samples {
        for row in col..n_samples {
            let value = grm[row * n_samples + col] as f64;
            if row == col || spgrm_keep_value(value, threshold, abs_threshold) {
                writer.push(SpgrmEntry {
                    col: col as u32,
                    row: row as u32,
                    value,
                })?;
            }
        }
        spgrm_progress_notify(
            progress_callback,
            col + 1,
            n_samples,
            notify_step,
            &mut last_notified,
            col + 1 == n_samples,
        )?;
    }
    writer.finish()?;
    Ok((out_path, n_samples, nnz))
}

fn spgrm_dense_npy_to_jxgrm_core(
    npy_path: &str,
    out_prefix: &str,
    threshold: f64,
    abs_threshold: bool,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<(String, usize, usize), String> {
    if !threshold.is_finite() {
        return Err("Sparse GRM threshold must be finite".to_string());
    }
    let out_path = normalize_spgrm_path(out_prefix);
    if out_path.is_empty() {
        return Err("Sparse GRM output prefix must not be empty".to_string());
    }

    let file = File::open(npy_path).map_err(|e| format!("open dense GRM NPY failed: {e}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| e.to_string())? };
    let (rows, cols, data_offset, dtype) = spgrm_parse_npy_header_for_float_matrix(&mmap[..])?;
    if rows == 0 || cols == 0 || rows != cols {
        return Err(format!(
            "Sparse GRM dense NPY must be non-empty square, got ({rows}, {cols})"
        ));
    }
    let n_samples = rows;
    let n_elem = n_samples
        .checked_mul(n_samples)
        .ok_or_else(|| "Sparse GRM dense NPY element count overflow".to_string())?;
    let bytes_per = match dtype {
        SpgrmNpyFloatDtype::F32Le => 4usize,
        SpgrmNpyFloatDtype::F64Le => 8usize,
    };
    let payload_bytes = n_elem
        .checked_mul(bytes_per)
        .ok_or_else(|| "Sparse GRM dense NPY payload size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(payload_bytes)
        .ok_or_else(|| "Sparse GRM dense NPY payload end overflow".to_string())?;
    if data_end > mmap.len() {
        return Err("Sparse GRM dense NPY payload is truncated".to_string());
    }
    let payload = &mmap[data_offset..data_end];

    let notify_step = if progress_every == 0 {
        (n_samples / 128).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;

    let mut nnz = 0usize;
    for col in 0..n_samples {
        for row in col..n_samples {
            let elem_idx = row
                .checked_mul(n_samples)
                .and_then(|base| base.checked_add(col))
                .ok_or_else(|| "Sparse GRM dense NPY index overflow".to_string())?;
            let value = spgrm_npy_read_value(payload, dtype, elem_idx)?;
            if !value.is_finite() {
                return Err(format!(
                    "Sparse GRM dense NPY found non-finite value at pair ({row}, {col})"
                ));
            }
            if row == col || spgrm_keep_value(value, threshold, abs_threshold) {
                nnz = nnz.saturating_add(1);
            }
        }
    }

    let mut writer = SpgrmCscStreamWriter::new(&out_path, n_samples, nnz)?;
    for col in 0..n_samples {
        for row in col..n_samples {
            let elem_idx = row
                .checked_mul(n_samples)
                .and_then(|base| base.checked_add(col))
                .ok_or_else(|| "Sparse GRM dense NPY index overflow".to_string())?;
            let value = spgrm_npy_read_value(payload, dtype, elem_idx)?;
            if row == col || spgrm_keep_value(value, threshold, abs_threshold) {
                writer.push(SpgrmEntry {
                    col: col as u32,
                    row: row as u32,
                    value,
                })?;
            }
        }
        spgrm_progress_notify(
            progress_callback,
            col + 1,
            n_samples,
            notify_step,
            &mut last_notified,
            col + 1 == n_samples,
        )?;
    }
    writer.finish()?;
    Ok((out_path, n_samples, nnz))
}

#[pyfunction]
#[pyo3(signature = (
    packed,
    n_samples,
    row_flip,
    row_maf,
    out_prefix,
    sample_indices=None,
    method=1,
    threshold=0.05_f64,
    abs_threshold=false,
    block_rows=0,
    sample_block=0,
    threads=0,
    progress_callback=None,
    progress_every=0
))]
pub fn spgrm_packed_to_jxgrm<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    out_prefix: String,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    threshold: f64,
    abs_threshold: bool,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(String, usize, usize)> {
    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = n_samples.div_ceil(4);
    if bytes_per_snp != expected_bps {
        return Err(PyRuntimeError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps}"
        )));
    }
    let packed_flat: Vec<u8> = match packed.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => packed_arr.iter().copied().collect(),
    };
    let row_flip_vec = row_flip.as_slice()?.to_vec();
    let row_maf_vec = row_maf.as_slice()?.to_vec();
    let sample_idx: Vec<usize> = if let Some(sample_indices) = sample_indices {
        parse_index_vec_i64(sample_indices.as_slice()?, n_samples, "sample_indices")?
    } else {
        (0..n_samples).collect()
    };
    py.detach(move || {
        spgrm_packed_to_jxgrm_core(
            packed_flat.as_slice(),
            n_samples,
            row_flip_vec.as_slice(),
            row_maf_vec.as_slice(),
            sample_idx.as_slice(),
            &out_prefix,
            method,
            threshold,
            abs_threshold,
            block_rows,
            sample_block,
            threads,
            progress_callback.as_ref(),
            progress_every,
        )
    })
    .map_err(map_err_string_to_py)
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    out_prefix=None,
    sample_indices=None,
    method=1,
    threshold=0.05_f64,
    abs_threshold=false,
    maf_threshold=0.02_f32,
    max_missing_rate=0.05_f32,
    het_threshold=0.0_f32,
    snps_only=false,
    block_rows=0,
    sample_block=0,
    threads=0,
    mmap_window_mb=None,
    progress_callback=None,
    progress_every=0
))]
pub fn spgrm_bed_to_jxgrm<'py>(
    py: Python<'py>,
    prefix: String,
    out_prefix: Option<String>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    threshold: f64,
    abs_threshold: bool,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(String, usize, usize)> {
    let bed_prefix = normalize_plink_prefix_local(&prefix);
    let n_samples_full = read_fam(&bed_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples_full == 0 {
        return Err(PyRuntimeError::new_err("No samples found in BED input."));
    }
    let sample_idx_vec: Option<Vec<usize>> = if let Some(sample_indices) = sample_indices {
        Some(parse_index_vec_i64(
            sample_indices.as_slice()?,
            n_samples_full,
            "sample_indices",
        )?)
    } else {
        None
    };
    let out_use = out_prefix.unwrap_or_else(|| bed_prefix.clone());
    py.detach(move || {
        spgrm_bed_to_jxgrm_core(
            &bed_prefix,
            sample_idx_vec.as_deref(),
            &out_use,
            method,
            threshold,
            abs_threshold,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            block_rows,
            sample_block,
            threads,
            mmap_window_mb,
            progress_callback.as_ref(),
            progress_every,
        )
    })
    .map_err(map_err_string_to_py)
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    row_source_indices,
    row_flip,
    row_maf,
    n_total_sites,
    out_prefix=None,
    sample_indices=None,
    method=1,
    threshold=0.05_f64,
    abs_threshold=false,
    block_rows=0,
    sample_block=0,
    threads=0,
    mmap_window_mb=None,
    progress_callback=None,
    progress_every=0
))]
pub fn spgrm_bed_to_jxgrm_from_meta<'py>(
    py: Python<'py>,
    prefix: String,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    n_total_sites: usize,
    out_prefix: Option<String>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    threshold: f64,
    abs_threshold: bool,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(String, usize, usize)> {
    let bed_prefix = normalize_plink_prefix_local(&prefix);
    let n_samples_full = read_fam(&bed_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples_full == 0 {
        return Err(PyRuntimeError::new_err("No samples found in BED input."));
    }
    if n_total_sites == 0 {
        return Err(PyRuntimeError::new_err(
            "n_total_sites must be positive for sparse GRM meta route.",
        ));
    }

    let sample_idx_vec: Option<Vec<usize>> = if let Some(sample_indices) = sample_indices {
        Some(parse_index_vec_i64(
            sample_indices.as_slice()?,
            n_samples_full,
            "sample_indices",
        )?)
    } else {
        None
    };
    let row_source_vec: Vec<usize> = row_source_indices
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("row_source_indices must be contiguous int64"))?
        .iter()
        .map(|&rid| {
            if rid < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "row_source_indices must be non-negative, got {rid}"
                )))
            } else {
                Ok(rid as usize)
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let row_flip_vec: Vec<bool> = match row_flip.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip.as_array().iter().copied().collect(),
    };
    let row_maf_vec: Vec<f32> = match row_maf.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_maf.as_array().iter().copied().collect(),
    };
    if row_source_vec.is_empty() {
        return Err(PyRuntimeError::new_err(
            "row_source_indices must not be empty",
        ));
    }
    if row_flip_vec.len() != row_source_vec.len() || row_maf_vec.len() != row_source_vec.len() {
        return Err(PyRuntimeError::new_err(format!(
            "row meta length mismatch: row_source_indices={}, row_flip={}, row_maf={}",
            row_source_vec.len(),
            row_flip_vec.len(),
            row_maf_vec.len(),
        )));
    }
    if let Some(&bad) = row_source_vec.iter().find(|&&rid| rid >= n_total_sites) {
        return Err(PyRuntimeError::new_err(format!(
            "row_source index out of range: {bad} >= n_total_sites={n_total_sites}"
        )));
    }

    let out_use = out_prefix.unwrap_or_else(|| bed_prefix.clone());
    py.detach(move || {
        let selected_idx: Cow<'_, [usize]> = match sample_idx_vec.as_deref() {
            Some(idx) => Cow::Borrowed(idx),
            None => Cow::Owned((0..n_samples_full).collect()),
        };
        let meta = crate::gfreader::PreparedBedLogicMetaOwned {
            site_keep: Vec::new(),
            row_flip: row_flip_vec,
            row_source_indices: row_source_vec,
            missing_rate: Vec::new(),
            maf: row_maf_vec,
            sites: Vec::new(),
            n_samples: n_samples_full,
            n_snps_total: n_total_sites,
            bytes_per_snp: n_samples_full.div_ceil(4),
        };
        spgrm_stream_bed_to_jxgrm_core(
            &bed_prefix,
            selected_idx.as_ref(),
            meta,
            &out_use,
            method,
            threshold,
            abs_threshold,
            block_rows,
            sample_block,
            threads,
            mmap_window_mb,
            progress_callback.as_ref(),
            progress_every,
            0usize,
        )
    })
    .map_err(map_err_string_to_py)
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    row_source_indices,
    row_flip,
    row_maf,
    n_total_sites,
    row_part_start,
    row_part_end,
    sample_indices=None,
    method=1,
    block_rows=0,
    sample_block=0,
    threads=0,
    mmap_window_mb=None,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_bed_f32_row_band_from_meta<'py>(
    py: Python<'py>,
    prefix: String,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    n_total_sites: usize,
    row_part_start: usize,
    row_part_end: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, usize, usize)> {
    let bed_prefix = normalize_plink_prefix_local(&prefix);
    let n_samples_full = read_fam(&bed_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples_full == 0 {
        return Err(PyRuntimeError::new_err("No samples found in BED input."));
    }
    if n_total_sites == 0 {
        return Err(PyRuntimeError::new_err(
            "n_total_sites must be positive for dense GRM part meta route.",
        ));
    }

    let sample_idx_vec: Option<Vec<usize>> = if let Some(sample_indices) = sample_indices {
        Some(parse_index_vec_i64(
            sample_indices.as_slice()?,
            n_samples_full,
            "sample_indices",
        )?)
    } else {
        None
    };
    let row_source_vec: Vec<usize> = row_source_indices
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("row_source_indices must be contiguous int64"))?
        .iter()
        .map(|&rid| {
            if rid < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "row_source_indices must be non-negative, got {rid}"
                )))
            } else {
                Ok(rid as usize)
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let row_flip_vec: Vec<bool> = match row_flip.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip.as_array().iter().copied().collect(),
    };
    let row_maf_vec: Vec<f32> = match row_maf.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_maf.as_array().iter().copied().collect(),
    };
    if row_source_vec.is_empty() {
        return Err(PyRuntimeError::new_err(
            "row_source_indices must not be empty",
        ));
    }
    if row_flip_vec.len() != row_source_vec.len() || row_maf_vec.len() != row_source_vec.len() {
        return Err(PyRuntimeError::new_err(format!(
            "row meta length mismatch: row_source_indices={}, row_flip={}, row_maf={}",
            row_source_vec.len(),
            row_flip_vec.len(),
            row_maf_vec.len(),
        )));
    }
    if let Some(&bad) = row_source_vec.iter().find(|&&rid| rid >= n_total_sites) {
        return Err(PyRuntimeError::new_err(format!(
            "row_source index out of range: {bad} >= n_total_sites={n_total_sites}"
        )));
    }

    let selected_idx: Vec<usize> = match sample_idx_vec {
        Some(idx) => idx,
        None => (0..n_samples_full).collect(),
    };
    if row_part_start >= row_part_end || row_part_end > selected_idx.len() {
        return Err(PyRuntimeError::new_err(format!(
            "row band is invalid: start={}, end={}, n_samples={}",
            row_part_start,
            row_part_end,
            selected_idx.len()
        )));
    }
    let part_rows = row_part_end - row_part_start;
    let meta = crate::gfreader::PreparedBedLogicMetaOwned {
        site_keep: Vec::new(),
        row_flip: row_flip_vec,
        row_source_indices: row_source_vec,
        missing_rate: Vec::new(),
        maf: row_maf_vec,
        sites: Vec::new(),
        n_samples: n_samples_full,
        n_snps_total: n_total_sites,
        bytes_per_snp: n_samples_full.div_ceil(4),
    };

    let (part_vec, eff_m, n_use) = py
        .detach(move || {
            grm_stream_bed_row_band_f32_core(
                &bed_prefix,
                selected_idx.as_slice(),
                meta,
                row_part_start,
                row_part_end,
                method,
                block_rows,
                sample_block,
                threads,
                mmap_window_mb,
                progress_callback.as_ref(),
                progress_every,
            )
        })
        .map_err(map_err_string_to_py)?;
    let out_arr = Array2::from_shape_vec((part_rows, n_use), part_vec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((PyArray2::from_owned_array(py, out_arr), eff_m, n_use))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    out_npy_path,
    row_source_indices,
    row_flip,
    row_maf,
    n_total_sites,
    row_part_start,
    row_part_end,
    sample_indices=None,
    method=1,
    block_rows=0,
    sample_block=0,
    threads=0,
    mmap_window_mb=None,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_bed_f32_row_band_from_meta_to_npy<'py>(
    py: Python<'py>,
    prefix: String,
    out_npy_path: String,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    n_total_sites: usize,
    row_part_start: usize,
    row_part_end: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(usize, usize)> {
    let bed_prefix = normalize_plink_prefix_local(&prefix);
    let n_samples_full = read_fam(&bed_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples_full == 0 {
        return Err(PyRuntimeError::new_err("No samples found in BED input."));
    }
    if n_total_sites == 0 {
        return Err(PyRuntimeError::new_err(
            "n_total_sites must be positive for dense GRM part meta route.",
        ));
    }

    let sample_idx_vec: Option<Vec<usize>> = if let Some(sample_indices) = sample_indices {
        Some(parse_index_vec_i64(
            sample_indices.as_slice()?,
            n_samples_full,
            "sample_indices",
        )?)
    } else {
        None
    };
    let row_source_vec: Vec<usize> = row_source_indices
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("row_source_indices must be contiguous int64"))?
        .iter()
        .map(|&rid| {
            if rid < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "row_source_indices must be non-negative, got {rid}"
                )))
            } else {
                Ok(rid as usize)
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let row_flip_vec: Vec<bool> = match row_flip.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip.as_array().iter().copied().collect(),
    };
    let row_maf_vec: Vec<f32> = match row_maf.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_maf.as_array().iter().copied().collect(),
    };
    if row_source_vec.is_empty() {
        return Err(PyRuntimeError::new_err(
            "row_source_indices must not be empty",
        ));
    }
    if row_flip_vec.len() != row_source_vec.len() || row_maf_vec.len() != row_source_vec.len() {
        return Err(PyRuntimeError::new_err(format!(
            "row meta length mismatch: row_source_indices={}, row_flip={}, row_maf={}",
            row_source_vec.len(),
            row_flip_vec.len(),
            row_maf_vec.len(),
        )));
    }
    if let Some(&bad) = row_source_vec.iter().find(|&&rid| rid >= n_total_sites) {
        return Err(PyRuntimeError::new_err(format!(
            "row_source index out of range: {bad} >= n_total_sites={n_total_sites}"
        )));
    }

    let selected_idx: Vec<usize> = match sample_idx_vec {
        Some(idx) => idx,
        None => (0..n_samples_full).collect(),
    };
    if row_part_start >= row_part_end || row_part_end > selected_idx.len() {
        return Err(PyRuntimeError::new_err(format!(
            "row band is invalid: start={}, end={}, n_samples={}",
            row_part_start,
            row_part_end,
            selected_idx.len()
        )));
    }
    let meta = crate::gfreader::PreparedBedLogicMetaOwned {
        site_keep: Vec::new(),
        row_flip: row_flip_vec,
        row_source_indices: row_source_vec,
        missing_rate: Vec::new(),
        maf: row_maf_vec,
        sites: Vec::new(),
        n_samples: n_samples_full,
        n_snps_total: n_total_sites,
        bytes_per_snp: n_samples_full.div_ceil(4),
    };

    py.detach(move || {
        grm_stream_bed_row_band_f32_to_npy_core(
            &bed_prefix,
            selected_idx.as_slice(),
            meta,
            row_part_start,
            row_part_end,
            &out_npy_path,
            method,
            block_rows,
            sample_block,
            threads,
            mmap_window_mb,
            progress_callback.as_ref(),
            progress_every,
        )
    })
    .map_err(map_err_string_to_py)
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    out_npy_path,
    row_source_indices,
    row_flip,
    row_maf,
    n_total_sites,
    sample_indices=None,
    method=1,
    block_rows=0,
    sample_block=0,
    threads=0,
    mmap_window_mb=None,
    progress_callback=None,
    progress_every=0
))]
pub fn grm_bed_f32_tiled_from_meta_to_npy<'py>(
    py: Python<'py>,
    prefix: String,
    out_npy_path: String,
    row_source_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    n_total_sites: usize,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: usize,
    block_rows: usize,
    sample_block: usize,
    threads: usize,
    mmap_window_mb: Option<usize>,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(usize, usize)> {
    let bed_prefix = normalize_plink_prefix_local(&prefix);
    let n_samples_full = read_fam(&bed_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples_full == 0 {
        return Err(PyRuntimeError::new_err("No samples found in BED input."));
    }
    if n_total_sites == 0 {
        return Err(PyRuntimeError::new_err(
            "n_total_sites must be positive for dense GRM tiled meta route.",
        ));
    }

    let sample_idx_vec: Option<Vec<usize>> = if let Some(sample_indices) = sample_indices {
        Some(parse_index_vec_i64(
            sample_indices.as_slice()?,
            n_samples_full,
            "sample_indices",
        )?)
    } else {
        None
    };
    let row_source_vec: Vec<usize> = row_source_indices
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("row_source_indices must be contiguous int64"))?
        .iter()
        .map(|&rid| {
            if rid < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "row_source_indices must be non-negative, got {rid}"
                )))
            } else {
                Ok(rid as usize)
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let row_flip_vec: Vec<bool> = match row_flip.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_flip.as_array().iter().copied().collect(),
    };
    let row_maf_vec: Vec<f32> = match row_maf.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => row_maf.as_array().iter().copied().collect(),
    };
    if row_source_vec.is_empty() {
        return Err(PyRuntimeError::new_err(
            "row_source_indices must not be empty",
        ));
    }
    if row_flip_vec.len() != row_source_vec.len() || row_maf_vec.len() != row_source_vec.len() {
        return Err(PyRuntimeError::new_err(format!(
            "row meta length mismatch: row_source_indices={}, row_flip={}, row_maf={}",
            row_source_vec.len(),
            row_flip_vec.len(),
            row_maf_vec.len(),
        )));
    }
    if let Some(&bad) = row_source_vec.iter().find(|&&rid| rid >= n_total_sites) {
        return Err(PyRuntimeError::new_err(format!(
            "row_source index out of range: {bad} >= n_total_sites={n_total_sites}"
        )));
    }

    let selected_idx: Vec<usize> = match sample_idx_vec {
        Some(idx) => idx,
        None => (0..n_samples_full).collect(),
    };
    if selected_idx.is_empty() {
        return Err(PyRuntimeError::new_err(
            "sample_indices must not be empty for dense GRM tiled meta route.",
        ));
    }
    let meta = crate::gfreader::PreparedBedLogicMetaOwned {
        site_keep: Vec::new(),
        row_flip: row_flip_vec,
        row_source_indices: row_source_vec,
        missing_rate: Vec::new(),
        maf: row_maf_vec,
        sites: Vec::new(),
        n_samples: n_samples_full,
        n_snps_total: n_total_sites,
        bytes_per_snp: n_samples_full.div_ceil(4),
    };

    py.detach(move || {
        grm_stream_bed_full_f32_to_npy_core(
            &bed_prefix,
            selected_idx.as_slice(),
            meta,
            &out_npy_path,
            method,
            block_rows,
            sample_block,
            threads,
            mmap_window_mb,
            progress_callback.as_ref(),
            progress_every,
        )
    })
    .map_err(map_err_string_to_py)
}

#[pyfunction]
#[pyo3(signature = (
    grm,
    out_prefix,
    threshold=0.05_f64,
    abs_threshold=false,
    progress_callback=None,
    progress_every=0
))]
pub fn spgrm_dense_f32_to_jxgrm<'py>(
    py: Python<'py>,
    grm: PyReadonlyArray2<'py, f32>,
    out_prefix: String,
    threshold: f64,
    abs_threshold: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(String, usize, usize)> {
    let grm_arr = grm.as_array();
    if grm_arr.ndim() != 2 {
        return Err(PyRuntimeError::new_err("grm must be 2D (n, n)."));
    }
    let shape = grm_arr.shape();
    if shape[0] != shape[1] {
        return Err(PyRuntimeError::new_err(format!(
            "grm must be square, got shape=({}, {})",
            shape[0], shape[1]
        )));
    }
    let n_samples = shape[0];
    let grm_flat: Vec<f32> = match grm.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => grm_arr.iter().copied().collect(),
    };
    py.detach(move || {
        spgrm_dense_f32_to_jxgrm_core(
            grm_flat.as_slice(),
            n_samples,
            &out_prefix,
            threshold,
            abs_threshold,
            progress_callback.as_ref(),
            progress_every,
        )
    })
    .map_err(map_err_string_to_py)
}

#[pyfunction]
#[pyo3(signature = (
    npy_path,
    out_prefix,
    threshold=0.05_f64,
    abs_threshold=false,
    progress_callback=None,
    progress_every=0
))]
pub fn spgrm_dense_npy_to_jxgrm<'py>(
    py: Python<'py>,
    npy_path: String,
    out_prefix: String,
    threshold: f64,
    abs_threshold: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(String, usize, usize)> {
    py.detach(move || {
        spgrm_dense_npy_to_jxgrm_core(
            &npy_path,
            &out_prefix,
            threshold,
            abs_threshold,
            progress_callback.as_ref(),
            progress_every,
        )
    })
    .map_err(map_err_string_to_py)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_site_major_dosages(sample_major: &[Vec<Option<u8>>]) -> Vec<u8> {
        let n_samples = sample_major.len();
        let n_sites = sample_major.first().map(|v| v.len()).unwrap_or(0);
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut packed = vec![0u8; n_sites * bytes_per_snp];
        for site_idx in 0..n_sites {
            for (sample_idx, sample) in sample_major.iter().enumerate() {
                let code = match sample[site_idx] {
                    Some(0) => 0b00u8,
                    Some(1) => 0b10u8,
                    Some(2) => 0b11u8,
                    None => 0b01u8,
                    Some(v) => panic!("invalid dosage {v}"),
                };
                let byte_off = site_idx * bytes_per_snp + (sample_idx >> 2);
                let lane_shift = (sample_idx & 3) * 2;
                packed[byte_off] |= code << lane_shift;
            }
        }
        packed
    }

    fn write_toy_plink(prefix: &str, sample_major: &[Vec<Option<u8>>]) {
        let packed = pack_site_major_dosages(sample_major);
        let n_samples = sample_major.len();
        let n_sites = sample_major.first().map(|v| v.len()).unwrap_or(0);

        let mut bed_bytes = vec![0x6C_u8, 0x1B_u8, 0x01_u8];
        bed_bytes.extend_from_slice(packed.as_slice());
        std::fs::write(format!("{prefix}.bed"), bed_bytes).unwrap();

        let fam = (0..n_samples)
            .map(|i| format!("F{i}\tI{i}\t0\t0\t0\t-9\n"))
            .collect::<String>();
        std::fs::write(format!("{prefix}.fam"), fam).unwrap();

        let bim = (0..n_sites)
            .map(|i| format!("1\tsnp{}\t0\t{}\tA\tC\n", i + 1, i + 1))
            .collect::<String>();
        std::fs::write(format!("{prefix}.bim"), bim).unwrap();
    }

    fn dense_from_csc_lower(csc: &SparseGrmCsc) -> Vec<f64> {
        let n = csc.n_samples;
        let mut out = vec![0.0_f64; n * n];
        for col in 0..n {
            let start = csc.col_ptr[col] as usize;
            let end = csc.col_ptr[col + 1] as usize;
            for idx in start..end {
                let row = csc.row_indices[idx] as usize;
                let v = csc.values[idx];
                out[row * n + col] = v;
                out[col * n + row] = v;
            }
        }
        out
    }

    #[test]
    fn sparse_grm_coo_to_csc_matches_expected_dense() {
        let geno = vec![
            vec![Some(0), Some(0), Some(1)],
            vec![Some(0), Some(1), Some(1)],
            vec![Some(2), Some(1), Some(2)],
            vec![Some(2), Some(2), Some(2)],
        ];
        let packed = pack_site_major_dosages(&geno);
        let row_flip = vec![false, false, false];
        let row_maf = vec![0.5_f32, 0.5_f32, 0.25_f32];
        let sample_idx = vec![0usize, 1usize, 2usize, 3usize];
        let coo = sparse_grm_coo_from_packed(
            packed.as_slice(),
            geno.len(),
            row_flip.as_slice(),
            row_maf.as_slice(),
            sample_idx.as_slice(),
            1usize,
            0.20_f64,
            false,
            2usize,
            2usize,
            1usize,
            None,
            0usize,
        )
        .unwrap();
        let csc = coo_lower_to_csc(sample_idx.len(), coo).unwrap();
        let dense = dense_from_csc_lower(&csc);

        let x = [
            [-1.0_f64, -1.0_f64, -0.5_f64],
            [-1.0_f64, 0.0_f64, -0.5_f64],
            [1.0_f64, 0.0_f64, 0.5_f64],
            [1.0_f64, 1.0_f64, 0.5_f64],
        ];
        let denom = 0.5_f64 + 0.5_f64 + 0.1875_f64;
        let mut expected = vec![0.0_f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0_f64;
                for k in 0..3 {
                    s += x[i][k] * x[j][k];
                }
                expected[i * 4 + j] = s / denom;
            }
        }

        for i in 0..4 {
            assert!((dense[i * 4 + i] - expected[i * 4 + i]).abs() < 1e-10);
        }
        assert!((dense[2 * 4] - expected[2 * 4]).abs() < 1e-10);
        assert_eq!(dense[1 * 4], 0.0_f64);
        assert_eq!(dense[3], 0.0_f64);
        assert_eq!(csc.nnz, 7usize);
    }

    #[test]
    fn sparse_grm_writer_emits_header_and_payload() {
        let csc = SparseGrmCsc {
            n_samples: 3usize,
            nnz: 4usize,
            col_ptr: vec![0u64, 2u64, 3u64, 4u64],
            row_indices: vec![0u32, 2u32, 1u32, 2u32],
            values: vec![1.0_f64, 0.25_f64, 1.1_f64, 0.9_f64],
        };
        let mut out_path = std::env::temp_dir();
        out_path.push(format!(
            "janusx_spgrm_test_{}_{}.spgrm",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let out_str = out_path.to_string_lossy().to_string();
        write_sparse_grm_csc(&out_str, &csc).unwrap();
        let bytes = std::fs::read(&out_str).unwrap();
        let expected_len = 16usize
            + csc.col_ptr.len() * std::mem::size_of::<u64>()
            + csc.row_indices.len() * std::mem::size_of::<u32>()
            + spgrm_jxgrm_values_padding_bytes(csc.nnz).unwrap()
            + csc.values.len() * std::mem::size_of::<f64>();
        assert_eq!(bytes.len(), expected_len);
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 3u64);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 4u64);
        let _ = std::fs::remove_file(out_str);
    }

    #[test]
    fn sparse_grm_writer_pads_values_to_f64_alignment_for_odd_nnz() {
        let csc = SparseGrmCsc {
            n_samples: 2usize,
            nnz: 3usize,
            col_ptr: vec![0u64, 2u64, 3u64],
            row_indices: vec![0u32, 1u32, 1u32],
            values: vec![1.0_f64, 0.25_f64, 0.9_f64],
        };
        let mut out_path = std::env::temp_dir();
        out_path.push(format!(
            "janusx_spgrm_pad_test_{}_{}.spgrm",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let out_str = out_path.to_string_lossy().to_string();
        write_sparse_grm_csc(&out_str, &csc).unwrap();
        let bytes = std::fs::read(&out_str).unwrap();
        let row_end = 16usize
            + csc.col_ptr.len() * std::mem::size_of::<u64>()
            + csc.row_indices.len() * std::mem::size_of::<u32>();
        assert_eq!(row_end % 8, 4);
        assert_eq!(&bytes[row_end..row_end + 4], &[0u8; 4]);
        let _ = std::fs::remove_file(out_str);
    }

    #[test]
    fn sparse_grm_chunked_merge_streaming_matches_in_memory() {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "janusx_spgrm_merge_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let chunk_paths = (0..3usize)
            .map(|idx| dir.join(format!("chunk{idx:03}.coo")))
            .collect::<Vec<_>>();
        let mut chunk0 = vec![
            SpgrmEntry {
                col: 1,
                row: 2,
                value: 0.2_f64,
            },
            SpgrmEntry {
                col: 0,
                row: 0,
                value: 1.0_f64,
            },
            SpgrmEntry {
                col: 1,
                row: 1,
                value: 1.1_f64,
            },
        ];
        let mut chunk1 = vec![
            SpgrmEntry {
                col: 1,
                row: 2,
                value: 0.3_f64,
            },
            SpgrmEntry {
                col: 2,
                row: 2,
                value: 0.9_f64,
            },
            SpgrmEntry {
                col: 0,
                row: 3,
                value: 0.4_f64,
            },
        ];
        let mut chunk2 = vec![
            SpgrmEntry {
                col: 2,
                row: 3,
                value: 0.5_f64,
            },
            SpgrmEntry {
                col: 3,
                row: 3,
                value: 1.2_f64,
            },
            SpgrmEntry {
                col: 0,
                row: 3,
                value: 0.1_f64,
            },
        ];
        spill_sorted_entries_to_chunk(&chunk_paths[0], &mut chunk0).unwrap();
        spill_sorted_entries_to_chunk(&chunk_paths[1], &mut chunk1).unwrap();
        spill_sorted_entries_to_chunk(&chunk_paths[2], &mut chunk2).unwrap();

        let out_mem = dir.join("merge_in_memory.spgrm");
        let out_stream = dir.join("merge_streaming.spgrm");
        merge_chunked_entries_to_jxgrm_in_memory(
            out_mem.to_str().unwrap(),
            4usize,
            chunk_paths.as_slice(),
            9usize,
        )
        .unwrap();
        merge_chunked_entries_to_jxgrm_streaming(
            out_stream.to_str().unwrap(),
            4usize,
            chunk_paths.as_slice(),
        )
        .unwrap();

        let mem = crate::cholesky::read_sparse_grm_csc(out_mem.to_str().unwrap()).unwrap();
        let stream = crate::cholesky::read_sparse_grm_csc(out_stream.to_str().unwrap()).unwrap();
        assert_eq!(mem.n_samples, stream.n_samples);
        assert_eq!(mem.nnz, stream.nnz);
        assert_eq!(mem.col_ptr, stream.col_ptr);
        assert_eq!(mem.row_indices, stream.row_indices);
        assert_eq!(mem.values.len(), stream.values.len());
        for (lhs, rhs) in mem.values.iter().zip(stream.values.iter()) {
            assert!((lhs - rhs).abs() < 1e-12);
        }
        assert_eq!(mem.nnz, 6usize);

        let _ = std::fs::remove_file(out_mem);
        let _ = std::fs::remove_file(out_stream);
        for chunk_path in chunk_paths {
            let _ = std::fs::remove_file(chunk_path);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn spgrm_bed_windowed_matches_full_mmap() {
        let geno = vec![
            vec![Some(0), Some(0), Some(1), Some(2)],
            vec![Some(0), Some(1), Some(1), Some(2)],
            vec![Some(2), Some(1), Some(2), Some(1)],
            vec![Some(2), Some(2), Some(2), Some(0)],
        ];

        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "janusx_spgrm_bed_window_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let prefix = dir.join("toy");
        let prefix_str = prefix.to_string_lossy().to_string();
        write_toy_plink(&prefix_str, &geno);

        let out_full = dir.join("full");
        let out_window = dir.join("window");
        let out_full_str = out_full.to_string_lossy().to_string();
        let out_window_str = out_window.to_string_lossy().to_string();

        let (path_full, n_full, nnz_full) = spgrm_bed_to_jxgrm_core(
            &prefix_str,
            None,
            &out_full_str,
            1usize,
            0.0_f64,
            true,
            0.0_f32,
            1.0_f32,
            0.0_f32,
            false,
            0usize,
            0usize,
            1usize,
            None,
            None,
            0usize,
        )
        .unwrap();
        let (path_window, n_window, nnz_window) = spgrm_bed_to_jxgrm_core(
            &prefix_str,
            None,
            &out_window_str,
            1usize,
            0.0_f64,
            true,
            0.0_f32,
            1.0_f32,
            0.0_f32,
            false,
            0usize,
            0usize,
            1usize,
            Some(1usize),
            None,
            0usize,
        )
        .unwrap();

        let full = crate::cholesky::read_sparse_grm_csc(&path_full).unwrap();
        let window = crate::cholesky::read_sparse_grm_csc(&path_window).unwrap();
        assert_eq!(n_full, geno.len());
        assert_eq!(n_window, geno.len());
        assert_eq!(nnz_full, nnz_window);
        assert_eq!(full.n_samples, window.n_samples);
        assert_eq!(full.nnz, window.nnz);
        assert_eq!(full.col_ptr, window.col_ptr);
        assert_eq!(full.row_indices, window.row_indices);
        assert_eq!(full.values.len(), window.values.len());
        for (lhs, rhs) in full.values.iter().zip(window.values.iter()) {
            assert!((lhs - rhs).abs() < 1e-12);
        }

        let _ = std::fs::remove_file(format!("{prefix_str}.bed"));
        let _ = std::fs::remove_file(format!("{prefix_str}.bim"));
        let _ = std::fs::remove_file(format!("{prefix_str}.fam"));
        let _ = std::fs::remove_file(path_full);
        let _ = std::fs::remove_file(path_window);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn spgrm_stream_auto_sample_block_uses_full_decode_budget() {
        let sample_step = spgrm_auto_stream_sample_block_with_budget(
            5_000usize,
            997_897usize,
            0usize,
            Some(153.0),
            2usize,
        );
        assert_eq!(sample_step, 2_448usize);
        assert!(sample_step > SPGRM_DEFAULT_SAMPLE_BLOCK);
    }

    #[test]
    fn spgrm_packed_auto_sample_block_remains_capped_by_worker_scratch_budget() {
        let sample_step = spgrm_auto_sample_block_with_budget(
            5_000usize,
            997_897usize,
            0usize,
            Some(153.0),
            2usize,
        );
        assert_eq!(sample_step, SPGRM_DEFAULT_SAMPLE_BLOCK);
    }

    #[test]
    fn spgrm_batch_planner_keeps_column_groups_intact_under_tight_limits() {
        let tasks = build_spgrm_tasks(8, 2);
        let batches = build_spgrm_task_batches(
            tasks.as_slice(),
            4,
            SpgrmTaskBatchLimits {
                max_decoded_bytes: 1,
                max_accum_bytes: 1,
            },
        );
        assert_eq!(
            batches,
            vec![
                SpgrmTaskBatchSpan { start: 0, end: 4 },
                SpgrmTaskBatchSpan { start: 4, end: 7 },
                SpgrmTaskBatchSpan { start: 7, end: 9 },
                SpgrmTaskBatchSpan { start: 9, end: 10 },
            ]
        );
    }

    #[test]
    fn spgrm_batch_planner_splits_on_accum_footprint() {
        let tasks = build_spgrm_tasks(8, 2);
        let tile_bytes = 2usize * 2usize * std::mem::size_of::<f32>();
        let batches = build_spgrm_task_batches(
            tasks.as_slice(),
            4,
            SpgrmTaskBatchLimits {
                max_decoded_bytes: usize::MAX,
                max_accum_bytes: tile_bytes * 5,
            },
        );
        assert_eq!(
            batches,
            vec![
                SpgrmTaskBatchSpan { start: 0, end: 4 },
                SpgrmTaskBatchSpan { start: 4, end: 9 },
                SpgrmTaskBatchSpan { start: 9, end: 10 },
            ]
        );
    }

    #[test]
    fn spgrm_batch_planner_uses_true_decoded_stripe_bytes_for_narrow_tail_blocks() {
        let tasks = build_spgrm_tasks(5, 2);
        let row_step = 4usize;
        let decoded_limit = row_step * (2usize + 2usize + 1usize) * std::mem::size_of::<f32>();
        let batches = build_spgrm_task_batches(
            tasks.as_slice(),
            row_step,
            SpgrmTaskBatchLimits {
                max_decoded_bytes: decoded_limit,
                max_accum_bytes: usize::MAX,
            },
        );
        assert_eq!(batches, vec![SpgrmTaskBatchSpan { start: 0, end: 6 }]);
    }

    #[test]
    fn spgrm_stream_batch_default_keeps_single_large_diag_tile_unsplit() {
        let n = 640usize;
        let task_batch = vec![SpgrmTask {
            col_start: 0,
            col_end: n,
            row_start: 0,
            row_end: n,
        }];
        let sample_idx = (0..n).collect::<Vec<_>>();
        let (stripe_scratch, task_accum) = spgrm_build_stream_batch(
            task_batch.as_slice(),
            4usize,
            sample_idx.as_slice(),
            n,
            8usize,
        );
        assert_eq!(stripe_scratch.len(), 1usize);
        assert_eq!(task_accum.len(), 1usize);
        assert!(!task_accum[0].accum_row_major);
    }

    #[test]
    fn self_block_accumulate_ssyrk_matches_sgemm_on_lower_triangle() {
        let rows = 3usize;
        let n_cols = 4usize;
        let block_col_major = vec![
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32, 9.0_f32,
            10.0_f32, 11.0_f32, 12.0_f32,
        ];
        let mut accum_syrk = vec![0.0_f32; n_cols * n_cols];
        let mut accum_gemm = vec![0.0_f32; n_cols * n_cols];
        self_block_accumulate_ssyrk(
            block_col_major.as_slice(),
            rows,
            n_cols,
            accum_syrk.as_mut_slice(),
            0.0_f32,
        );
        self_block_accumulate_sgemm(
            block_col_major.as_slice(),
            rows,
            n_cols,
            accum_gemm.as_mut_slice(),
            0.0_f32,
        );
        for col in 0..n_cols {
            for row in col..n_cols {
                let idx = row + col * n_cols;
                assert!(
                    (accum_syrk[idx] - accum_gemm[idx]).abs() < 1e-4_f32,
                    "lower-triangle mismatch at ({row},{col}): syrk={} gemm={}",
                    accum_syrk[idx],
                    accum_gemm[idx],
                );
            }
        }
    }

    #[test]
    fn spgrm_negative_nonabs_threshold_keeps_all_offdiagonals() {
        assert!(spgrm_keep_value(-1.5, -0.1, false));
        assert!(spgrm_keep_value(0.0, -0.1, false));
        assert!(spgrm_keep_value(2.0, -0.1, false));
    }

    #[test]
    fn spgrm_zero_threshold_still_filters_nonpositive_offdiagonals() {
        assert!(!spgrm_keep_value(-1e-6, 0.0, false));
        assert!(!spgrm_keep_value(0.0, 0.0, false));
        assert!(spgrm_keep_value(1e-6, 0.0, false));
    }
}
