// SparseLMM exact scan on the lambda-only sparse-Cholesky scale:
// V_lambda = K_sparse + lambda I
// P_lambda = V_lambda^{-1} - V_lambda^{-1}X(X'V_lambda^{-1}X)^{-1}X'V_lambda^{-1}
// beta_hat = (g' P_lambda y) / (g' P_lambda g)
// sigma2_hat = (y' P_lambda y) / (n - rank(X) - 1)
// se(beta_hat) = sqrt(sigma2_hat / (g' P_lambda g))
//
// Exact SparseLMM keeps lambda, P_lambda, and sigma2 on one internal scale from null fit
// through the per-SNP sparse solve, without a separate external variance rescaling chain.

use memmap2::Mmap;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::time::Instant;

use crate::assoc2tsv::{
    append_assoc_block_from_arrays, append_assoc_block_from_core_sites, AssocArraysBlock,
    AssocCoreSitesBlock, AssocMissBlock, AssocResultCols, AssocResultLayout,
};
use crate::bedmath::adaptive_grm_block_rows;
use crate::bedmath::{decode_mean_imputed_additive_packed_block_rows_f32, packed_byte_lut};
use crate::blas::{cblas_dgemm_dispatch, CblasInt, CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS};
use crate::cholesky::{
    sparse_cholesky_analyze_subset_jxgrm_path_cached_with_progress, sparse_jxgrm_header_n_samples,
    subset_sparse_grm_csc, MmapSparseGrmCsc, SparseGrmCscView, SparseJxgrmAnalyzeProgressStage,
    SparseJxgrmCholesky, SparseJxgrmCholeskyAnalysis, SparseJxgrmSolveWorkspace,
};
use crate::decode::{
    decode_indexed_row_model_into_f64, decode_packed_row_model_enum_into_f64, AdditiveDecodePlan,
    PackedGeneticModel, PackedRowDecodeSource,
};
use crate::gfcore::{read_fam, BimChunkReader};
use crate::gfreader::{
    count_packed_row_counts, count_packed_row_counts_selected_with_excluded,
    prepare_bed_logic_meta_owned_for_stats_samples,
};
use crate::gload::{GenotypeMatrix, UnifiedInput, WindowedBedMatrix};
use crate::he::row_major_block_mul_mat_f32;
use crate::linalg::{chi2_sf_df1, cholesky_inplace, cholesky_solve_into};
use crate::pcg::{
    PcgMatrixSolveInfo, PcgSolveInfo, PcgSplmmNullModel, PcgSplmmNullModelInfo, PcgSplmmRHatResult,
};
use crate::splmm_approx::{
    estimate_residualized_approx_scan_sparse, estimate_residualized_approx_scan_to_tsv_sparse,
    fit_sparse_reml_on_residualized_response,
};
use crate::stats_common::{env_truthy, get_cached_pool, parse_index_vec_i64};

const SPLMM_TINY: f64 = 1e-30_f64;
const SPLMM_DEFAULT_RHAT_MARKERS: usize = 30;
const SPLMM_DEFAULT_RHAT_SEED: u64 = 20260527;
const SPLMM_DEFAULT_SPARSE_CHOLESKY_MAX_L_NNZ: usize = 100_000_000;
#[derive(Clone, Copy, Debug, Default)]
struct SplmmPackedScanTiming {
    decode_secs: f64,
    gpy_secs: f64,
    gx_secs: f64,
    assoc_secs: f64,
    row_sumsq_secs: f64,
    repack_secs: f64,
    solve_secs: f64,
    xtz_secs: f64,
    denom_secs: f64,
    sink_secs: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SplmmTsvTiming {
    pub(crate) format_secs: f64,
    pub(crate) write_secs: f64,
    pub(crate) send_secs: f64,
    pub(crate) finish_secs: f64,
    pub(crate) blocks: usize,
    pub(crate) bytes: usize,
    pub(crate) format_threads: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct SplmmAssocTopLevelTiming {
    prepare_inputs_secs: f64,
    bim_meta_secs: f64,
    null_scan_core_secs: f64,
    writer_wait_secs: f64,
    detach_wall_secs: f64,
    other_secs: f64,
    total_secs: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct SplmmNullStateTiming {
    solve_y_secs: f64,
    solve_x_secs: f64,
    xt_v_inv_y_secs: f64,
    xt_v_inv_x_secs: f64,
    chol_secs: f64,
    beta_py_secs: f64,
    scale_secs: f64,
    total_secs: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct SplmmPrepareTiming {
    workspace_secs: f64,
    xtx_chol_secs: f64,
    null_state_secs: f64,
    rhat_decode_secs: f64,
    rhat_solve_secs: f64,
    rhat_mx_secs: f64,
    rhat_p0_secs: f64,
    rhat_reduce_secs: f64,
    total_secs: f64,
    n_rhat_requested: usize,
    n_rhat_used: usize,
    null_state: SplmmNullStateTiming,
}

struct SplmmTsvBlockPayload {
    row_start: usize,
    rows_here: usize,
    results: Vec<f64>,
}

pub(crate) enum SplmmTsvMetaInput {
    External {
        chrom: Vec<String>,
        pos: Vec<i64>,
        snp: Vec<String>,
        allele0: Vec<String>,
        allele1: Vec<String>,
    },
    BimSelected {
        bed_prefix: String,
        row_source_indices: Arc<[usize]>,
    },
}

impl SplmmTsvMetaInput {
    #[inline]
    fn n_rows(&self) -> usize {
        match self {
            Self::External { chrom, .. } => chrom.len(),
            Self::BimSelected {
                row_source_indices, ..
            } => row_source_indices.len(),
        }
    }
}

pub(crate) struct SplmmScanToTsvResult {
    pub(crate) r_hat: f64,
    pub(crate) written_rows: usize,
    pub(crate) factor_nnz: usize,
    pub(crate) null_info: PcgSplmmNullModelInfo,
    pub(crate) rhat_info: PcgSplmmRHatResult,
    pub(crate) factor_load_secs: f64,
    pub(crate) scan_prepare_secs: f64,
    pub(crate) scan_exec_secs: f64,
    pub(crate) writer_wait_secs: f64,
}

pub(crate) type SplmmScanResult = (
    f64,
    Vec<f64>,
    PcgSplmmNullModelInfo,
    PcgSplmmRHatResult,
    usize,
);

#[inline]
fn splmm_packed_stage_timing_enabled() -> bool {
    env_truthy("JX_SPLMM_PACKED_STAGE_TIMING")
}

#[inline]
pub(crate) fn splmm_top_level_timing_enabled() -> bool {
    env_truthy("JX_SPLMM_TOP_LEVEL_TIMING")
        || env_truthy("JX_SPLMM_TOPLEVEL_TIMING")
        || splmm_packed_stage_timing_enabled()
}

#[inline]
fn splmm_prepare_stage_timing_enabled() -> bool {
    env_truthy("JX_SPLMM_PREPARE_STAGE_TIMING") || splmm_packed_stage_timing_enabled()
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
        + timing.assoc_secs
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
        "SparseLMM packed timing mode={mode}: decode={:.3}s ({:.1}%), gpy={:.3}s ({:.1}%), gx={:.3}s ({:.1}%), assoc={:.3}s ({:.1}%), row_sumsq={:.3}s ({:.1}%), repack={:.3}s ({:.1}%), solve={:.3}s ({:.1}%), xtz={:.3}s ({:.1}%), denom={:.3}s ({:.1}%), sink={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, p={}, threads={}",
        timing.decode_secs,
        to_pct(timing.decode_secs),
        timing.gpy_secs,
        to_pct(timing.gpy_secs),
        timing.gx_secs,
        to_pct(timing.gx_secs),
        timing.assoc_secs,
        to_pct(timing.assoc_secs),
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

fn emit_splmm_prepare_timing(
    mode: &str,
    timing: &SplmmPrepareTiming,
    rows: usize,
    n: usize,
    p: usize,
) {
    let accounted_secs = timing.workspace_secs
        + timing.xtx_chol_secs
        + timing.null_state_secs
        + timing.rhat_decode_secs
        + timing.rhat_solve_secs
        + timing.rhat_mx_secs
        + timing.rhat_p0_secs
        + timing.rhat_reduce_secs;
    let other_secs = (timing.total_secs - accounted_secs).max(0.0_f64);
    let to_pct = |x: f64| -> f64 {
        if timing.total_secs > 0.0 {
            x * 100.0 / timing.total_secs
        } else {
            0.0
        }
    };
    eprintln!(
        "SparseLMM prepare timing mode={mode}: workspace={:.3}s ({:.1}%), xtx_chol={:.3}s ({:.1}%), null_state={:.3}s ({:.1}%), rhat_decode={:.3}s ({:.1}%), rhat_solve={:.3}s ({:.1}%), rhat_mx={:.3}s ({:.1}%), rhat_p0={:.3}s ({:.1}%), rhat_reduce={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, p={}, rhat_requested={}, rhat_used={}",
        timing.workspace_secs,
        to_pct(timing.workspace_secs),
        timing.xtx_chol_secs,
        to_pct(timing.xtx_chol_secs),
        timing.null_state_secs,
        to_pct(timing.null_state_secs),
        timing.rhat_decode_secs,
        to_pct(timing.rhat_decode_secs),
        timing.rhat_solve_secs,
        to_pct(timing.rhat_solve_secs),
        timing.rhat_mx_secs,
        to_pct(timing.rhat_mx_secs),
        timing.rhat_p0_secs,
        to_pct(timing.rhat_p0_secs),
        timing.rhat_reduce_secs,
        to_pct(timing.rhat_reduce_secs),
        other_secs,
        to_pct(other_secs),
        timing.total_secs,
        rows,
        n,
        p,
        timing.n_rhat_requested,
        timing.n_rhat_used,
    );

    let null_total = timing.null_state.total_secs;
    let null_to_pct = |x: f64| -> f64 {
        if null_total > 0.0 {
            x * 100.0 / null_total
        } else {
            0.0
        }
    };
    eprintln!(
        "SparseLMM null-state timing: solve_y={:.3}s ({:.1}%), solve_x={:.3}s ({:.1}%), xt_v_inv_y={:.3}s ({:.1}%), xt_v_inv_x={:.3}s ({:.1}%), chol={:.3}s ({:.1}%), beta_py={:.3}s ({:.1}%), scale={:.3}s ({:.1}%), total={:.3}s, n={}, p={}",
        timing.null_state.solve_y_secs,
        null_to_pct(timing.null_state.solve_y_secs),
        timing.null_state.solve_x_secs,
        null_to_pct(timing.null_state.solve_x_secs),
        timing.null_state.xt_v_inv_y_secs,
        null_to_pct(timing.null_state.xt_v_inv_y_secs),
        timing.null_state.xt_v_inv_x_secs,
        null_to_pct(timing.null_state.xt_v_inv_x_secs),
        timing.null_state.chol_secs,
        null_to_pct(timing.null_state.chol_secs),
        timing.null_state.beta_py_secs,
        null_to_pct(timing.null_state.beta_py_secs),
        timing.null_state.scale_secs,
        null_to_pct(timing.null_state.scale_secs),
        null_total,
        n,
        p,
    );
}

fn emit_splmm_tsv_timing(mode: &str, timing: &SplmmTsvTiming, rows: usize) {
    let worker_total_secs = timing.format_secs + timing.write_secs;
    let accounted_secs = worker_total_secs + timing.send_secs + timing.finish_secs;
    let to_pct = |x: f64| -> f64 {
        if accounted_secs > 0.0 {
            x * 100.0 / accounted_secs
        } else {
            0.0
        }
    };
    eprintln!(
        "SparseLMM TSV timing mode={mode}: format={:.3}s ({:.1}%), write={:.3}s ({:.1}%), send={:.3}s ({:.1}%), finish_wait={:.3}s ({:.1}%), worker_total={:.3}s, accounted={:.3}s, rows={}, blocks={}, bytes={}, format_threads={}",
        timing.format_secs,
        to_pct(timing.format_secs),
        timing.write_secs,
        to_pct(timing.write_secs),
        timing.send_secs,
        to_pct(timing.send_secs),
        timing.finish_secs,
        to_pct(timing.finish_secs),
        worker_total_secs,
        accounted_secs,
        rows,
        timing.blocks,
        timing.bytes,
        timing.format_threads,
    );
}

fn emit_splmm_assoc_top_level_timing(
    mode: &str,
    timing: &SplmmAssocTopLevelTiming,
    rows: usize,
    n: usize,
    p: usize,
    threads: usize,
) {
    let to_pct = |x: f64| -> f64 {
        if timing.total_secs > 0.0 {
            x * 100.0 / timing.total_secs
        } else {
            0.0
        }
    };
    eprintln!(
        "SparseLMM top-level timing mode={mode}: prepare_inputs={:.3}s ({:.1}%), bim_meta={:.3}s ({:.1}%), null_scan_core={:.3}s ({:.1}%), writer_wait={:.3}s ({:.1}%), detach_wall={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, p={}, threads={}",
        timing.prepare_inputs_secs,
        to_pct(timing.prepare_inputs_secs),
        timing.bim_meta_secs,
        to_pct(timing.bim_meta_secs),
        timing.null_scan_core_secs,
        to_pct(timing.null_scan_core_secs),
        timing.writer_wait_secs,
        to_pct(timing.writer_wait_secs),
        timing.detach_wall_secs,
        to_pct(timing.detach_wall_secs),
        timing.other_secs,
        to_pct(timing.other_secs),
        timing.total_secs,
        rows,
        n,
        p,
        threads,
    );
}

pub(crate) fn emit_splmm_null_scan_core_timing(
    mode: &str,
    factor_load_secs: f64,
    scan_prepare_secs: f64,
    scan_exec_secs: f64,
    writer_wait_secs: f64,
    rows: usize,
    n: usize,
    p: usize,
    threads: usize,
) {
    let total_secs = factor_load_secs + scan_prepare_secs + scan_exec_secs + writer_wait_secs;
    let to_pct = |x: f64| -> f64 {
        if total_secs > 0.0 {
            x * 100.0 / total_secs
        } else {
            0.0
        }
    };
    eprintln!(
        "SparseLMM null-scan-core timing mode={mode}: factor_load={:.3}s ({:.1}%), scan_prepare={:.3}s ({:.1}%), scan_exec={:.3}s ({:.1}%), writer_wait={:.3}s ({:.1}%), total={:.3}s, rows={}, n={}, p={}, threads={}",
        factor_load_secs,
        to_pct(factor_load_secs),
        scan_prepare_secs,
        to_pct(scan_prepare_secs),
        scan_exec_secs,
        to_pct(scan_exec_secs),
        writer_wait_secs,
        to_pct(writer_wait_secs),
        total_secs,
        rows,
        n,
        p,
        threads,
    );
}

#[inline]
fn splmm_tsv_format_threads(scan_threads: usize) -> usize {
    if let Ok(raw) = std::env::var("JX_SPLMM_TSV_THREADS") {
        if let Ok(val) = raw.trim().parse::<usize>() {
            if val > 0 {
                return val;
            }
        }
    }
    if scan_threads <= 2 {
        1
    } else {
        scan_threads.min(4)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_splmm_async_tsv_writer<R, F>(
    out_tsv: &str,
    _row_capacity_hint: usize,
    scan_threads: usize,
    meta_input: SplmmTsvMetaInput,
    row_maf: &[f32],
    row_missing: &[f32],
    gm: PackedGeneticModel,
    mode: &str,
    stage_timing: bool,
    run_scan: F,
) -> Result<(R, SplmmTsvTiming), String>
where
    F: FnOnce(&mut dyn FnMut(usize, usize, &[f64]) -> Result<(), String>) -> Result<R, String>,
{
    const HEADER: &[u8] = b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n";
    const WRITER_CAPACITY: usize = 64 * 1024 * 1024;
    const QUEUE_DEPTH_HINT: usize = 4;
    const MIN_FLUSH_BYTES: usize = 8 * 1024 * 1024;
    const MAX_FLUSH_BYTES: usize = 64 * 1024 * 1024;
    const MIN_QUEUE_DEPTH: usize = 16;
    const MAX_QUEUE_DEPTH: usize = 64;

    let queue_depth_eff = QUEUE_DEPTH_HINT
        .max((WRITER_CAPACITY / (4 * 1024 * 1024)).max(1))
        .clamp(MIN_QUEUE_DEPTH, MAX_QUEUE_DEPTH);
    let rows_total = meta_input.n_rows();
    let format_threads = splmm_tsv_format_threads(scan_threads.max(1));

    let (scan_out, tsv_timing) = std::thread::scope(
        |scope| -> Result<(R, SplmmTsvTiming), String> {
            let (tx, rx) = sync_channel::<SplmmTsvBlockPayload>(queue_depth_eff);
            let handle = scope.spawn(move || -> Result<SplmmTsvTiming, String> {
                let out_file =
                    File::create(out_tsv).map_err(|e| format!("create {out_tsv}: {e}"))?;
                let mut writer = BufWriter::with_capacity(WRITER_CAPACITY.max(1), out_file);
                let mut bim_reader = match &meta_input {
                    SplmmTsvMetaInput::BimSelected { bed_prefix, .. } => {
                        Some(BimChunkReader::open(bed_prefix)?)
                    }
                    SplmmTsvMetaInput::External { .. } => None,
                };
                let mut timing = SplmmTsvTiming::default();
                timing.format_threads = format_threads;
                let t0 = stage_timing.then(Instant::now);
                writer
                    .write_all(HEADER)
                    .map_err(|e| format!("write header {out_tsv}: {e}"))?;
                writer
                    .flush()
                    .map_err(|e| format!("flush header {out_tsv}: {e}"))?;
                if let Some(t0) = t0 {
                    timing.write_secs += t0.elapsed().as_secs_f64();
                }
                let flush_threshold = WRITER_CAPACITY.clamp(MIN_FLUSH_BYTES, MAX_FLUSH_BYTES);
                let mut pending_bytes = 0usize;
                let format_pool = if format_threads > 1 {
                    Some(
                        rayon::ThreadPoolBuilder::new()
                            .num_threads(format_threads)
                            .build()
                            .map_err(|e| format!("SparseLMM TSV format pool: {e}"))?,
                    )
                } else {
                    None
                };
                let model_key = splmm_model_key(gm);
                for payload in rx {
                    let t0 = stage_timing.then(Instant::now);
                    let row_end = payload.row_start + payload.rows_here;
                    let row_maf_block = &row_maf[payload.row_start..row_end];
                    let row_missing_block = &row_missing[payload.row_start..row_end];
                    let formatted_blocks: Vec<String> = match &meta_input {
                        SplmmTsvMetaInput::External {
                            chrom,
                            pos,
                            snp,
                            allele0,
                            allele1,
                        } => {
                            let chrom_block = &chrom[payload.row_start..row_end];
                            let pos_block = &pos[payload.row_start..row_end];
                            let snp_block = &snp[payload.row_start..row_end];
                            let allele0_block = &allele0[payload.row_start..row_end];
                            let allele1_block = &allele1[payload.row_start..row_end];
                            if payload.rows_here >= 1024 && format_threads > 1 {
                                let rows_per_chunk = payload
                                    .rows_here
                                    .div_ceil(format_threads.saturating_mul(2))
                                    .clamp(256, 4096);
                                let n_chunks = payload.rows_here.div_ceil(rows_per_chunk);
                                format_pool
                                    .as_ref()
                                    .expect("format pool must exist when format_threads > 1")
                                    .install(|| {
                                        (0..n_chunks)
                                            .into_par_iter()
                                            .map(|chunk_idx| {
                                                let local_row_start = chunk_idx * rows_per_chunk;
                                                let rows_here = (payload.rows_here
                                                    - local_row_start)
                                                    .min(rows_per_chunk);
                                                let local_row_end = local_row_start + rows_here;
                                                let mut out = String::with_capacity(
                                                    rows_here.saturating_mul(96),
                                                );
                                                append_assoc_block_from_arrays(
                                                    &mut out,
                                                    AssocResultLayout::ResultCols {
                                                        schema: AssocResultCols::Basic3,
                                                        row_stride: 3,
                                                    },
                                                    model_key,
                                                    AssocArraysBlock {
                                                        chrom: &chrom_block
                                                            [local_row_start..local_row_end],
                                                        pos: &pos_block
                                                            [local_row_start..local_row_end],
                                                        snp: &snp_block
                                                            [local_row_start..local_row_end],
                                                        allele0: &allele0_block
                                                            [local_row_start..local_row_end],
                                                        allele1: &allele1_block
                                                            [local_row_start..local_row_end],
                                                        maf: &row_maf_block
                                                            [local_row_start..local_row_end],
                                                        miss: AssocMissBlock::Rate(
                                                            &row_missing_block
                                                                [local_row_start..local_row_end],
                                                        ),
                                                    },
                                                    &payload.results
                                                        [local_row_start * 3..local_row_end * 3],
                                                )
                                                .expect(
                                                    "SparseLMM external TSV formatting should be length-consistent",
                                                );
                                                out
                                            })
                                            .collect()
                                    })
                            } else {
                                let mut out =
                                    String::with_capacity(payload.rows_here.saturating_mul(96));
                                append_assoc_block_from_arrays(
                                    &mut out,
                                    AssocResultLayout::ResultCols {
                                        schema: AssocResultCols::Basic3,
                                        row_stride: 3,
                                    },
                                    model_key,
                                    AssocArraysBlock {
                                        chrom: chrom_block,
                                        pos: pos_block,
                                        snp: snp_block,
                                        allele0: allele0_block,
                                        allele1: allele1_block,
                                        maf: row_maf_block,
                                        miss: AssocMissBlock::Rate(row_missing_block),
                                    },
                                    payload.results.as_slice(),
                                )
                                .expect(
                                    "SparseLMM external TSV formatting should be length-consistent",
                                );
                                vec![out]
                            }
                        }
                        SplmmTsvMetaInput::BimSelected {
                            row_source_indices, ..
                        } => {
                            let row_source_block = &row_source_indices[payload.row_start..row_end];
                            let block_sites = bim_reader
                                .as_mut()
                                .ok_or_else(|| {
                                    "internal error: missing BIM reader for SparseLMM TSV writer"
                                        .to_string()
                                })?
                                .read_selected_rows(row_source_block)?;
                            if payload.rows_here >= 1024 && format_threads > 1 {
                                let rows_per_chunk = payload
                                    .rows_here
                                    .div_ceil(format_threads.saturating_mul(2))
                                    .clamp(256, 4096);
                                let n_chunks = payload.rows_here.div_ceil(rows_per_chunk);
                                format_pool
                                    .as_ref()
                                    .expect("format pool must exist when format_threads > 1")
                                    .install(|| {
                                        (0..n_chunks)
                                            .into_par_iter()
                                            .map(|chunk_idx| {
                                                let local_row_start = chunk_idx * rows_per_chunk;
                                                let rows_here = (payload.rows_here
                                                    - local_row_start)
                                                    .min(rows_per_chunk);
                                                let local_row_end = local_row_start + rows_here;
                                                let mut out = String::with_capacity(
                                                    rows_here.saturating_mul(96),
                                                );
                                                append_assoc_block_from_core_sites(
                                                    &mut out,
                                                    AssocResultLayout::ResultCols {
                                                        schema: AssocResultCols::Basic3,
                                                        row_stride: 3,
                                                    },
                                                    model_key,
                                                    AssocCoreSitesBlock {
                                                        sites: &block_sites
                                                            [local_row_start..local_row_end],
                                                        maf: &row_maf_block
                                                            [local_row_start..local_row_end],
                                                        miss: AssocMissBlock::Rate(
                                                            &row_missing_block
                                                                [local_row_start..local_row_end],
                                                        ),
                                                    },
                                                    &payload.results
                                                        [local_row_start * 3..local_row_end * 3],
                                                )
                                                .expect(
                                                    "SparseLMM BIM TSV formatting should be length-consistent",
                                                );
                                                out
                                            })
                                            .collect()
                                    })
                            } else {
                                let mut out =
                                    String::with_capacity(payload.rows_here.saturating_mul(96));
                                append_assoc_block_from_core_sites(
                                    &mut out,
                                    AssocResultLayout::ResultCols {
                                        schema: AssocResultCols::Basic3,
                                        row_stride: 3,
                                    },
                                    model_key,
                                    AssocCoreSitesBlock {
                                        sites: block_sites.as_slice(),
                                        maf: row_maf_block,
                                        miss: AssocMissBlock::Rate(row_missing_block),
                                    },
                                    payload.results.as_slice(),
                                )
                                .expect(
                                    "SparseLMM BIM TSV formatting should be length-consistent",
                                );
                                vec![out]
                            }
                        }
                    };
                    if let Some(t0) = t0 {
                        timing.format_secs += t0.elapsed().as_secs_f64();
                    }
                    let mut bytes_now = 0usize;
                    for block in formatted_blocks {
                        bytes_now = bytes_now.saturating_add(block.len());
                        let t0 = stage_timing.then(Instant::now);
                        writer
                            .write_all(block.as_bytes())
                            .map_err(|e| format!("write {out_tsv}: {e}"))?;
                        if let Some(t0) = t0 {
                            timing.write_secs += t0.elapsed().as_secs_f64();
                        }
                    }
                    pending_bytes = pending_bytes.saturating_add(bytes_now);
                    if pending_bytes >= flush_threshold {
                        let t0 = stage_timing.then(Instant::now);
                        writer
                            .flush()
                            .map_err(|e| format!("flush {out_tsv}: {e}"))?;
                        if let Some(t0) = t0 {
                            timing.write_secs += t0.elapsed().as_secs_f64();
                        }
                        pending_bytes = 0usize;
                    }
                    timing.blocks = timing.blocks.saturating_add(1);
                    timing.bytes = timing.bytes.saturating_add(bytes_now);
                }
                let t0 = stage_timing.then(Instant::now);
                writer
                    .flush()
                    .map_err(|e| format!("flush {out_tsv}: {e}"))?;
                if let Some(t0) = t0 {
                    timing.write_secs += t0.elapsed().as_secs_f64();
                }
                Ok(timing)
            });

            let mut send_secs = 0.0_f64;
            let mut sink = |row_start: usize, rows_here: usize, block: &[f64]| {
                let t0 = stage_timing.then(Instant::now);
                tx.send(SplmmTsvBlockPayload {
                    row_start,
                    rows_here,
                    results: block.to_vec(),
                })
                .map_err(|e| format!("send {out_tsv}: {e}"))?;
                if let Some(t0) = t0 {
                    send_secs += t0.elapsed().as_secs_f64();
                }
                Ok(())
            };

            let run_res = run_scan(&mut sink);
            drop(tx);
            let t0 = stage_timing.then(Instant::now);
            let mut timing = handle
                .join()
                .map_err(|_| format!("writer thread panicked for {out_tsv}"))??;
            if let Some(t0) = t0 {
                timing.finish_secs += t0.elapsed().as_secs_f64();
            }
            timing.send_secs = send_secs;
            let out = run_res?;
            Ok((out, timing))
        },
    )?;

    if stage_timing {
        emit_splmm_tsv_timing(mode, &tsv_timing, rows_total);
    }
    Ok((scan_out, tsv_timing))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplmmScanMode {
    Approx,
    Exact,
}

impl SplmmScanMode {
    fn parse(text: &str) -> PyResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "approx" => Ok(Self::Approx),
            "exact" => Ok(Self::Exact),
            _ => Err(PyRuntimeError::new_err(
                "scan_mode must be one of: approx, exact",
            )),
        }
    }

    #[inline]
    fn needs_rhat(self) -> bool {
        matches!(self, Self::Approx)
    }

    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::Approx => "approx",
            Self::Exact => "exact",
        }
    }
}

#[derive(Clone)]
enum SplmmPreparedPayload {
    Packed(Arc<[u8]>),
    Mmap(Arc<Mmap>),
}

impl SplmmPreparedPayload {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Packed(bytes) => bytes.as_ref(),
            Self::Mmap(mmap) => &mmap[3..],
        }
    }
}

#[derive(Clone)]
pub(crate) struct SplmmPreparedInput {
    bed_prefix: Option<String>,
    payload: Option<SplmmPreparedPayload>,
    n_samples_full: usize,
    bytes_per_snp: usize,
    row_flip: Arc<[bool]>,
    row_maf: Arc<[f32]>,
    #[allow(dead_code)]
    row_missing: Arc<[f32]>,
    row_source_indices: Option<Arc<[usize]>>,
}

impl SplmmPreparedInput {
    #[inline]
    pub(crate) fn n_rows(&self) -> usize {
        self.row_flip.len()
    }

    #[inline]
    fn payload_bytes(&self) -> Result<&[u8], String> {
        self.payload
            .as_ref()
            .map(SplmmPreparedPayload::as_bytes)
            .ok_or_else(|| {
                "SparseLMM prepared input has no resident BED payload; windowed access is required."
                    .to_string()
            })
    }

    #[cfg(test)]
    #[inline]
    fn row_bytes(&self, row_idx: usize) -> Result<&[u8], String> {
        let src =
            crate::decode::resolve_source_row_index(self.row_source_indices.as_deref(), row_idx);
        let packed = self.payload_bytes()?;
        Ok(&packed[src * self.bytes_per_snp..(src + 1) * self.bytes_per_snp])
    }
}

struct PreparedSplmmAssoc {
    gm: PackedGeneticModel,
    y_vec: Vec<f64>,
    x_design: Vec<f64>,
    scan_prepared: SplmmPreparedInput,
    operator_prepared: SplmmPreparedInput,
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

fn inspect_plink_bed_shape_local(
    bed_prefix: &str,
    n_samples_full: usize,
) -> Result<(usize, usize), String> {
    if n_samples_full == 0 {
        return Err("No samples found in BED input.".to_string());
    }
    let bytes_per_snp = n_samples_full.div_ceil(4);
    let bed_path = format!("{bed_prefix}.bed");
    let mut bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
    let mut header = [0u8; 3];
    bed_file
        .read_exact(&mut header)
        .map_err(|e| format!("read {bed_path} header: {e}"))?;
    if header != [0x6C, 0x1B, 0x01] {
        return Err("Only SNP-major BED supported".to_string());
    }
    let bed_len = bed_file
        .metadata()
        .map_err(|e| format!("metadata {bed_path}: {e}"))?
        .len() as usize;
    if bed_len < 3 {
        return Err("BED too small".to_string());
    }
    let payload_len = bed_len - 3;
    if bytes_per_snp == 0 || payload_len % bytes_per_snp != 0 {
        return Err(format!(
            "invalid BED payload length: data_len={payload_len}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    Ok((bytes_per_snp, payload_len / bytes_per_snp))
}

#[inline]
fn splmm_model_key(gm: PackedGeneticModel) -> &'static str {
    match gm {
        PackedGeneticModel::Add => "add",
        PackedGeneticModel::Dom => "dom",
        PackedGeneticModel::Rec => "rec",
        PackedGeneticModel::Het => "het",
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

#[allow(clippy::too_many_arguments)]
fn build_splmm_tsv_meta_input(
    prefix: &str,
    scan_prepared: &SplmmPreparedInput,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
) -> Result<SplmmTsvMetaInput, String> {
    if chrom.is_empty()
        && pos.is_empty()
        && snp.is_empty()
        && allele0.is_empty()
        && allele1.is_empty()
    {
        let bed_prefix = if let Some(bed_prefix) = scan_prepared.bed_prefix.as_deref() {
            bed_prefix.to_string()
        } else {
            normalize_plink_prefix_local(prefix)
        };
        let row_source_indices = scan_prepared
            .row_source_indices
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| "auto-populate requires row_source_indices".to_string())?;
        return Ok(SplmmTsvMetaInput::BimSelected {
            bed_prefix,
            row_source_indices,
        });
    }
    let rows = scan_prepared.n_rows();
    if chrom.len() != rows
        || pos.len() != rows
        || snp.len() != rows
        || allele0.len() != rows
        || allele1.len() != rows
    {
        return Err(format!(
            "SparseLMM TSV metadata length mismatch: rows={rows}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        ));
    }
    Ok(SplmmTsvMetaInput::External {
        chrom,
        pos,
        snp,
        allele0,
        allele1,
    })
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
) -> PyResult<SplmmPreparedInput> {
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
    Ok(SplmmPreparedInput {
        bed_prefix: None,
        payload: Some(SplmmPreparedPayload::Packed(packed_arc)),
        n_samples_full: packed_n_samples,
        bytes_per_snp,
        row_flip: Arc::from(row_flip),
        row_maf: Arc::from(row_maf),
        row_missing: Arc::from(row_missing),
        row_source_indices: row_source_indices.map(Arc::from),
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
    retain_payload: bool,
) -> PyResult<SplmmPreparedInput> {
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
    let (bytes_per_snp, n_snps_bed) = inspect_plink_bed_shape_local(&bed_prefix, n_samples_full)
        .map_err(PyRuntimeError::new_err)?;
    let payload = if retain_payload {
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
        Some(SplmmPreparedPayload::Mmap(mmap))
    } else {
        None
    };
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
    Ok(SplmmPreparedInput {
        bed_prefix: Some(bed_prefix),
        payload,
        n_samples_full,
        bytes_per_snp,
        row_flip: Arc::from(row_flip),
        row_maf: Arc::from(row_maf),
        row_missing: Arc::from(row_missing),
        row_source_indices: Some(Arc::from(row_source_indices)),
    })
}

fn prepare_prefix_input(
    prefix: &str,
    n_samples_full: usize,
    site_keep: Option<&[bool]>,
    sample_idx_probe: Option<&[usize]>,
    payload_mmap: Option<Arc<Mmap>>,
) -> Result<SplmmPreparedInput, String> {
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
    Ok(SplmmPreparedInput {
        bed_prefix: Some(bed_prefix),
        payload: Some(SplmmPreparedPayload::Mmap(mmap)),
        n_samples_full,
        bytes_per_snp: meta.bytes_per_snp,
        row_flip: Arc::from(meta.row_flip),
        row_maf: Arc::from(meta.maf),
        row_missing: Arc::from(meta.missing_rate),
        row_source_indices: Some(Arc::from(meta.row_source_indices)),
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
    mmap_window_mb: Option<usize>,
) -> PyResult<PreparedSplmmAssoc> {
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
    let windowed_requested = mmap_window_mb.filter(|&v| v > 0).is_some();
    let skip_resident_bed_payload = use_external_mmap_meta && windowed_requested;
    let shared_bed_mmap = if use_external_packed {
        None
    } else if skip_resident_bed_payload {
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
            !skip_resident_bed_payload,
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

    Ok(PreparedSplmmAssoc {
        gm,
        y_vec,
        x_design,
        scan_prepared,
        operator_prepared,
        scan_sample_idx,
        operator_sample_idx,
    })
}

pub(crate) fn choose_rhat_rows(m: usize, count: usize, seed: u64) -> Vec<usize> {
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
pub(crate) fn n_rhat_progress_total(m: usize, count: usize) -> usize {
    count.min(m).saturating_add(1).max(1)
}

pub(crate) fn decode_rhat_markers_col_major(
    scan_prepared: &SplmmPreparedInput,
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    rhat_rows: &[usize],
    progress_callback: Option<&Py<PyAny>>,
    progress_stage: usize,
    progress_total: usize,
    mmap_window_mb: Option<usize>,
) -> Result<Vec<f64>, String> {
    let n = scan_sample_idx.len();
    let n_rhat = rhat_rows.len();
    let decode_plan =
        AdditiveDecodePlan::from_sample_indices(scan_prepared.n_samples_full, scan_sample_idx);
    let mut row_source = if let Some(payload) = scan_prepared.payload.as_ref() {
        PackedRowDecodeSource::Resident {
            packed_flat: payload.as_bytes(),
            bytes_per_snp: scan_prepared.bytes_per_snp,
        }
    } else {
        let prefix = scan_prepared.bed_prefix.as_deref().ok_or_else(|| {
            "SparseLMM requires a PLINK BED prefix for windowed row decode.".to_string()
        })?;
        let window_mb = mmap_window_mb.filter(|&v| v > 0).ok_or_else(|| {
            "SparseLMM windowed row decode requires mmap_window_mb when no resident BED payload is retained."
                .to_string()
        })?;
        PackedRowDecodeSource::Windowed(WindowedBedMatrix::open(prefix, window_mb)?)
    };
    let mut sampled_markers = vec![0.0_f64; n * n_rhat];
    let mut tmp_snp = vec![0.0_f64; n];
    for (col, &row_idx) in rhat_rows.iter().enumerate() {
        decode_indexed_row_model_into_f64(
            &mut row_source,
            scan_prepared.row_flip.as_ref(),
            scan_prepared.row_maf.as_ref(),
            scan_prepared.row_source_indices.as_deref(),
            row_idx,
            n,
            gm,
            decode_plan.sample_identity(),
            decode_plan.sample_byte_idx(),
            decode_plan.sample_bit_shift(),
            &mut tmp_snp,
        )?;
        for i in 0..n {
            sampled_markers[col * n + i] = tmp_snp[i];
        }
        if progress_callback.is_some() {
            emit_progress_callback(progress_callback, progress_stage, col + 1, progress_total)?;
        }
    }
    Ok(sampled_markers)
}

#[inline]
fn trivial_pcg_solve_info() -> PcgSolveInfo {
    PcgSolveInfo {
        converged: true,
        iters: 1,
        rel_res: 0.0,
    }
}

#[inline]
pub(crate) fn trivial_pcg_null_info(n: usize, p: usize) -> PcgSplmmNullModelInfo {
    PcgSplmmNullModelInfo {
        v_inv_y: trivial_pcg_solve_info(),
        v_inv_x: PcgMatrixSolveInfo {
            n_rows: n,
            n_cols: p,
            converged_all: true,
            max_iters: 1,
            max_rel_res: 0.0,
            column_info: vec![trivial_pcg_solve_info(); p],
        },
    }
}

#[inline]
pub(crate) fn trivial_pcg_rhat_info(
    n: usize,
    n_markers_requested: usize,
    n_markers_used: usize,
) -> PcgSplmmRHatResult {
    PcgSplmmRHatResult {
        n_markers_requested,
        n_markers_used,
        solve_info: PcgMatrixSolveInfo {
            n_rows: n,
            n_cols: n_markers_requested,
            converged_all: true,
            max_iters: 1,
            max_rel_res: 0.0,
            column_info: vec![trivial_pcg_solve_info(); n_markers_requested],
        },
    }
}

#[inline]
pub(crate) fn emit_progress_callback(
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
fn scalar_spd_inv_from_chol(chol: &[f64], p: usize) -> Option<f64> {
    if p != 1 || chol.len() != 1 {
        return None;
    }
    let x00 = chol[0] * chol[0];
    if x00.is_finite() && x00 > SPLMM_TINY {
        Some(1.0_f64 / x00)
    } else {
        None
    }
}

#[inline]
fn residualized_sumsq_scalar(xtx00_inv: f64, xts0: f64, s_sq: f64) -> f64 {
    (s_sq - xts0 * xts0 * xtx00_inv).max(0.0_f64)
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

#[inline]
fn pack_score_design_rhs_f32(
    score: &[f64],
    x_design: &[f64],
    p: usize,
) -> Result<Vec<f32>, String> {
    let n = score.len();
    if x_design.len() != n.saturating_mul(p) {
        return Err(format!(
            "SparseLMM scan RHS length mismatch: len(x_design)={}, expected {}",
            x_design.len(),
            n.saturating_mul(p)
        ));
    }
    let score_f32 = cast_f64_slice_to_f32(score)?;
    let x_design_f32 = cast_f64_slice_to_f32(x_design)?;
    let rhs_cols = p
        .checked_add(1)
        .ok_or_else(|| "SparseLMM scan RHS column overflow".to_string())?;
    let mut rhs = vec![0.0_f32; n.saturating_mul(rhs_cols)];
    for i in 0..n {
        let dst = &mut rhs[i * rhs_cols..(i + 1) * rhs_cols];
        dst[0] = score_f32[i];
        dst[1..].copy_from_slice(&x_design_f32[i * p..(i + 1) * p]);
    }
    Ok(rhs)
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
fn additive_row_sumsq_from_counts(
    missing: usize,
    het: usize,
    hom_alt: usize,
    n_selected: usize,
    row_flip: bool,
    row_maf: f32,
) -> f64 {
    let non_missing = n_selected.saturating_sub(missing);
    let hom_two = if row_flip {
        non_missing.saturating_sub(het).saturating_sub(hom_alt)
    } else {
        hom_alt
    };
    let mean_g = (2.0_f64 * row_maf as f64).clamp(0.0_f64, 2.0_f64);
    4.0_f64.mul_add(
        hom_two as f64,
        (het as f64) + (missing as f64) * mean_g * mean_g,
    )
}

fn packed_block_additive_sumsq_from_counts<G: GenotypeMatrix>(
    input: &UnifiedInput<G>,
    row_start: usize,
    rows_here: usize,
    scan_sample_idx: &[usize],
    sample_identity: bool,
    selected_excluded_sample_idx: Option<&[usize]>,
    out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> bool {
    if rows_here == 0 || out.len() != rows_here {
        return rows_here == 0;
    }
    if !sample_identity && selected_excluded_sample_idx.is_none() {
        return false;
    }
    let first_src_idx = input.stats.row_source_indices[row_start];
    if input.matrix.source_row_bytes(first_src_idx).len() != input.stats.bytes_per_snp {
        return false;
    }
    let n_selected = scan_sample_idx.len();
    let n_samples_full = input.stats.n_samples_full;
    let use_parallel = pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows_here.saturating_mul(input.stats.bytes_per_snp) >= 16_384usize;
    let mut run = || {
        out.par_iter_mut().enumerate().for_each(|(local_idx, dst)| {
            let row_idx = row_start + local_idx;
            let src_idx = input.stats.row_source_indices[row_idx];
            let row = input.matrix.source_row_bytes(src_idx);
            let (missing, het, hom_alt) = if sample_identity {
                count_packed_row_counts(row, n_selected)
            } else {
                count_packed_row_counts_selected_with_excluded(
                    row,
                    n_samples_full,
                    scan_sample_idx,
                    selected_excluded_sample_idx,
                )
            };
            *dst = additive_row_sumsq_from_counts(
                missing,
                het,
                hom_alt,
                n_selected,
                input.stats.row_flip[row_idx],
                input.stats.maf[row_idx],
            );
        });
    };
    if use_parallel {
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
    } else {
        for (local_idx, dst) in out.iter_mut().enumerate() {
            let row_idx = row_start + local_idx;
            let src_idx = input.stats.row_source_indices[row_idx];
            let row = input.matrix.source_row_bytes(src_idx);
            let (missing, het, hom_alt) = if sample_identity {
                count_packed_row_counts(row, n_selected)
            } else {
                count_packed_row_counts_selected_with_excluded(
                    row,
                    n_samples_full,
                    scan_sample_idx,
                    selected_excluded_sample_idx,
                )
            };
            *dst = additive_row_sumsq_from_counts(
                missing,
                het,
                hom_alt,
                n_selected,
                input.stats.row_flip[row_idx],
                input.stats.maf[row_idx],
            );
        }
    }
    true
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

fn sparse_grm_to_dense<V: crate::cholesky::SparseGrmCscView + ?Sized>(csc: &V) -> Vec<f64> {
    let n = csc.n_samples();
    let mut dense = vec![0.0_f64; n.saturating_mul(n)];
    for col in 0..n {
        let start = csc.col_ptr()[col] as usize;
        let end = csc.col_ptr()[col + 1] as usize;
        for idx in start..end {
            let row = csc.row_indices()[idx] as usize;
            let value = csc.values()[idx];
            dense[row * n + col] = value;
            dense[col * n + row] = value;
        }
    }
    dense
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
fn sparse_splmm_resolve_path(
    prefix: &str,
    sparse_jxgrm_path: Option<&str>,
) -> Result<String, String> {
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
    Ok(path)
}

#[inline]
fn sparse_splmm_load_analysis(
    path: &str,
    expected_n: usize,
    sample_idx: &[usize],
    progress_callback: Option<&Py<PyAny>>,
) -> Result<Arc<SparseJxgrmCholeskyAnalysis>, String> {
    let sparse_n = sparse_jxgrm_header_n_samples(&path)?;
    let subset_request = normalize_sparse_subset_request(sample_idx, sparse_n)?;
    let target_n = subset_request.map(|idx| idx.len()).unwrap_or(sparse_n);
    if target_n != expected_n {
        return Err(format!(
            "SparseLMM kinship size mismatch: sparse target n={}, expected {} from {path}",
            target_n, expected_n
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
    Ok(analysis)
}

#[inline]
fn sparse_splmm_factorize_analysis(
    analysis: &SparseJxgrmCholeskyAnalysis,
    lbd: f64,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
) -> Result<SparseJxgrmCholesky, String> {
    if !lbd.is_finite() || lbd < 0.0 {
        return Err(format!(
            "SparseLMM requires finite non-negative lambda, got lambda={lbd}"
        ));
    }
    let (diag_mean_abs, diag_min, diag_max) = analysis.diag_stats_scaled(1.0_f64, lbd)?;
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
        match analysis.factorize_sigma_g2_k_plus_sigma_e2_i_with_diag_shift_parallel(
            1.0_f64,
            lbd,
            diag_shift,
            threads.max(1),
        ) {
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
    let lambda_near_boundary = lbd <= diag_mean_abs * 1e-8_f64;
    let hint = if lambda_near_boundary {
        "lambda is near zero, so sparse thresholding can make K + lambda I indefinite; auto-ridge failed. Try a smaller sparse cutoff such as `-splmm 0.01` or `-splmm 0.001`."
    } else {
        "hard-thresholded sparse GRM is not numerically SPD enough for LLT; try a smaller sparse cutoff such as `-splmm 0.01` or `-splmm 0.001`."
    };
    let msg = format!(
        "SparseLMM factorization failed after adaptive diagonal ridge escalation; \
lambda={lbd:.6e}, mean_diag={diag_mean_abs:.6e}, \
min_diag={diag_min:.6e}, max_diag={diag_max:.6e}. Last error: {last_msg}. Hint: {hint}"
    );
    Err(msg)
}

#[inline]
fn sparse_splmm_load_factor(
    prefix: &str,
    sparse_jxgrm_path: Option<&str>,
    expected_n: usize,
    sample_idx: &[usize],
    lbd: f64,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
) -> Result<SparseJxgrmCholesky, String> {
    let path = sparse_splmm_resolve_path(prefix, sparse_jxgrm_path)?;
    let analysis = sparse_splmm_load_analysis(&path, expected_n, sample_idx, progress_callback)?;
    sparse_splmm_factorize_analysis(&analysis, lbd, threads, progress_callback)
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

#[inline]
fn row_major_snp_block_f32_to_col_major_f64(
    block: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(block.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    let use_parallel = pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(cols) >= 16_384usize;
    if use_parallel {
        let mut run = || {
            out.par_chunks_mut(cols).enumerate().for_each(|(snp, dst)| {
                let src = &block[snp * cols..(snp + 1) * cols];
                for sample in 0..cols {
                    dst[sample] = src[sample] as f64;
                }
            });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }
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
fn pairwise_block_dot_f32_f64(
    lhs_block: &[f32],
    rhs_block: &[f64],
    rows: usize,
    cols: usize,
    out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(lhs_block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(rhs_block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(out.len(), rows);

    #[inline]
    fn dot_row(lhs: &[f32], rhs: &[f64]) -> f64 {
        let mut i = 0usize;
        let mut acc0 = 0.0_f64;
        let mut acc1 = 0.0_f64;
        let mut acc2 = 0.0_f64;
        let mut acc3 = 0.0_f64;
        while i + 4 <= lhs.len() {
            acc0 += (lhs[i] as f64) * rhs[i];
            acc1 += (lhs[i + 1] as f64) * rhs[i + 1];
            acc2 += (lhs[i + 2] as f64) * rhs[i + 2];
            acc3 += (lhs[i + 3] as f64) * rhs[i + 3];
            i += 4;
        }
        let mut dot = (acc0 + acc1) + (acc2 + acc3);
        while i < lhs.len() {
            dot += (lhs[i] as f64) * rhs[i];
            i += 1;
        }
        dot
    }

    let use_parallel = pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(cols) >= 16_384usize;
    if use_parallel {
        let mut run = || {
            out.par_iter_mut().enumerate().for_each(|(row_idx, dst)| {
                let lhs = &lhs_block[row_idx * cols..(row_idx + 1) * cols];
                let rhs = &rhs_block[row_idx * cols..(row_idx + 1) * cols];
                *dst = dot_row(lhs, rhs);
            });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }
    for row_idx in 0..rows {
        let lhs = &lhs_block[row_idx * cols..(row_idx + 1) * cols];
        let rhs = &rhs_block[row_idx * cols..(row_idx + 1) * cols];
        out[row_idx] = dot_row(lhs, rhs);
    }
}

#[inline]
fn fused_block_sv_xtz_scalar_p1(
    g_block: &[f32],
    z_block: &[f64],
    x0: &[f64],
    rows: usize,
    cols: usize,
    sv_out: &mut [f64],
    xtz_out: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    debug_assert_eq!(g_block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(z_block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(x0.len(), cols);
    debug_assert_eq!(sv_out.len(), rows);
    debug_assert_eq!(xtz_out.len(), rows);

    #[inline]
    fn fused_row(g: &[f32], z: &[f64], x0: &[f64]) -> (f64, f64) {
        let mut i = 0usize;
        let mut sv0 = 0.0_f64;
        let mut sv1 = 0.0_f64;
        let mut sv2 = 0.0_f64;
        let mut sv3 = 0.0_f64;
        let mut xtz0 = 0.0_f64;
        let mut xtz1 = 0.0_f64;
        let mut xtz2 = 0.0_f64;
        let mut xtz3 = 0.0_f64;
        while i + 4 <= g.len() {
            let z0 = z[i];
            let z1 = z[i + 1];
            let z2 = z[i + 2];
            let z3 = z[i + 3];
            sv0 += (g[i] as f64) * z0;
            sv1 += (g[i + 1] as f64) * z1;
            sv2 += (g[i + 2] as f64) * z2;
            sv3 += (g[i + 3] as f64) * z3;
            xtz0 += x0[i] * z0;
            xtz1 += x0[i + 1] * z1;
            xtz2 += x0[i + 2] * z2;
            xtz3 += x0[i + 3] * z3;
            i += 4;
        }
        let mut sv = (sv0 + sv1) + (sv2 + sv3);
        let mut xtz = (xtz0 + xtz1) + (xtz2 + xtz3);
        while i < g.len() {
            let z_i = z[i];
            sv += (g[i] as f64) * z_i;
            xtz += x0[i] * z_i;
            i += 1;
        }
        (sv, xtz)
    }

    let use_parallel = pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows.saturating_mul(cols) >= 16_384usize;
    if use_parallel {
        let mut run = || {
            sv_out
                .par_iter_mut()
                .zip(xtz_out.par_iter_mut())
                .enumerate()
                .for_each(|(row_idx, (sv_dst, xtz_dst))| {
                    let g = &g_block[row_idx * cols..(row_idx + 1) * cols];
                    let z = &z_block[row_idx * cols..(row_idx + 1) * cols];
                    let (sv, xtz) = fused_row(g, z, x0);
                    *sv_dst = sv;
                    *xtz_dst = xtz;
                });
        };
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }

    for row_idx in 0..rows {
        let g = &g_block[row_idx * cols..(row_idx + 1) * cols];
        let z = &z_block[row_idx * cols..(row_idx + 1) * cols];
        let (sv, xtz) = fused_row(g, z, x0);
        sv_out[row_idx] = sv;
        xtz_out[row_idx] = xtz;
    }
}

#[inline]
fn adaptive_exact_block_rows(requested: usize, n: usize) -> usize {
    // Exact block-scan peak memory model, per row (one SNP) of the block:
    //   block          f32 [n]     = 4n
    //   spy_block      f32 [1]     = 4
    //   z_col_major    f64 [n]     = 8n
    //   xts_block      f32 [p]     ≈4p
    //   c_block        f64 [p]     ≈8p
    //   row_ss         f64 [1]     = 8
    //   sv_block       f64 [1]     = 8
    //   out_block      f64 [3]     = 24
    //   --------------------------------------------------
    //   total per row              ≈12n + 12p + 52 bytes
    //
    // We approximate p=4 for safety and add slack for the sparse-solve
    // workspace and allocation overhead.
    // Default cap: 128 MiB, overridable via JX_SPLMM_EXACT_MAX_BLOCK_BYTES.
    const DEFAULT_MAX_BYTES: usize = 128 * 1024 * 1024; // 128 MiB
    let max_bytes: usize = std::env::var("JX_SPLMM_EXACT_MAX_BLOCK_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_BYTES);
    // Conservative cap after removing the extra RHS buffer:
    //   16n + 160 bytes per row
    let bytes_per_row = (16 * n.max(1)).saturating_add(160);
    let max_rows = max_bytes / bytes_per_row;
    requested.min(max_rows).max(1)
}

#[inline]
fn splmm_scan_block_hint_and_progress_step(
    block_rows: usize,
    progress_every: usize,
) -> (usize, usize) {
    let scan_block_hint = block_rows.max(512).max(1);
    let progress_step = if progress_every == 0 {
        scan_block_hint
    } else {
        progress_every.max(1)
    };
    (scan_block_hint, progress_step)
}

#[inline]
fn xt_mat_rhs_block(
    x_design: &[f64],
    _x_design_col_major: Option<&[f64]>,
    z_col_major: &[f64],
    n: usize,
    p: usize,
    rows_here: usize,
    c_block: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    let use_parallel = pool.map(|tp| tp.current_num_threads() > 1).unwrap_or(false)
        && rows_here.saturating_mul(n).saturating_mul(p.max(1)) >= 262_144usize;
    let mut run = || {
        c_block.par_chunks_mut(p).enumerate().for_each(|(j, c_j)| {
            c_j.fill(0.0_f64);
            let z_col = &z_col_major[j * n..(j + 1) * n];
            for i in 0..n {
                let z_ji = z_col[i];
                if z_ji == 0.0 {
                    continue;
                }
                let xi = &x_design[i * p..(i + 1) * p];
                for k in 0..p {
                    c_j[k] += xi[k] * z_ji;
                }
            }
        });
    };
    if use_parallel {
        if let Some(tp) = pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }

    for j in 0..rows_here {
        let c_j = &mut c_block[j * p..(j + 1) * p];
        c_j.fill(0.0_f64);
        let z_col = &z_col_major[j * n..(j + 1) * n];
        for i in 0..n {
            let z_ji = z_col[i];
            if z_ji == 0.0 {
                continue;
            }
            let xi = &x_design[i * p..(i + 1) * p];
            for k in 0..p {
                c_j[k] += xi[k] * z_ji;
            }
        }
    }
}

#[inline]
fn splmm_wald_from_score_denom(score: f64, denom: f64, sigma2: f64) -> Option<(f64, f64, f64)> {
    if !(score.is_finite()
        && denom.is_finite()
        && denom > SPLMM_TINY
        && sigma2.is_finite()
        && sigma2 > 0.0)
    {
        return None;
    }
    let beta = score / denom;
    let var_beta = sigma2 / denom;
    if !(beta.is_finite() && var_beta.is_finite() && var_beta > 0.0) {
        return None;
    }
    let se = var_beta.sqrt();
    let chisq = (score * score) / (sigma2 * denom);
    if !(se.is_finite() && se > 0.0 && chisq.is_finite() && chisq >= 0.0) {
        return None;
    }
    Some((beta, se, chi2_sf_df1(chisq)))
}

/// Tiled sparse solve using JanusX's cached thread pool (when available).
/// `workspaces` are pre-allocated once and reused across blocks.
///
/// We intentionally do not use faer's internal Rayon parallelism here.
/// On SparseLMM exact benchmarks this caused severe regressions
/// (solve time jumping from tens of seconds to several minutes), so the
/// stable default remains outer tiled parallelism only.
#[inline]
fn sparse_solve_rhs_tiled(
    factor: &SparseJxgrmCholesky,
    rhs_col_major: &mut [f64],
    n_rhs: usize,
    _full_workspace: &mut SparseJxgrmSolveWorkspace,
    workspaces: &mut [SparseJxgrmSolveWorkspace],
    pool: Option<&rayon::ThreadPool>,
    _threads: usize,
) -> Result<(), String> {
    factor.solve_in_place_tiled(rhs_col_major, n_rhs, workspaces, pool)
}

// Core block-scan loop for exact g'Pg denominator.
// Works with any GenotypeMatrix backend via UnifiedInput.
//
// The scan uses the null-model sigma2 = y'P0y / df on the same K + lambda I
// scale used to construct P0. This matches the full-P Wald form
//   Var(beta_hat) = sigma2 / (g'P0g)
// and avoids SNP-specific sigma2 shrinkage that would make strong signals
// systematically over-significant.
fn exact_scan_blocks_core<G: GenotypeMatrix>(
    factor: &SparseJxgrmCholesky,
    input: &mut UnifiedInput<G>,
    x_design: &[f64],
    x_design_col_major: &[f64],
    score_vec: &[f64],
    ypy: f64,
    df: f64,
    xt_w_x_chol: &[f64],
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
    if !(ypy.is_finite() && ypy > 0.0) {
        return Err(format!(
            "SparseLMM exact scan requires finite positive yPy on K + lambda I scale, got {ypy}"
        ));
    }
    if !(df.is_finite() && df > 0.0) {
        return Err(format!(
            "SparseLMM exact scan requires finite positive df, got {df}"
        ));
    }
    let sigma2 = ypy / df;
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return Err(format!(
            "SparseLMM exact scan requires finite positive null sigma2 on K + lambda I scale, got {sigma2}"
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
    if xt_w_x_chol.len() != p * p {
        return Err(format!(
            "SparseLMM exact scan XtWX chol length mismatch: got {}, expected {}",
            xt_w_x_chol.len(),
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

    let decode_plan =
        AdditiveDecodePlan::from_sample_indices(input.stats.n_samples_full, scan_sample_idx);
    let sample_identity = decode_plan.sample_identity();
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let m = input.n_markers();
    let progress_total = if progress_total_override == 0 {
        m.max(1)
    } else {
        progress_total_override.max(1)
    };
    let (scan_block_hint, progress_step) =
        splmm_scan_block_hint_and_progress_step(block_rows, progress_every);
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
        adaptive_grm_block_rows(scan_block_hint, m, n, 0usize, threads).max(1),
        n,
    );
    let score_f32 = cast_f64_slice_to_f32(score_vec)?;

    let mut block = vec![0.0_f32; scan_block_rows * n];
    let mut spy_block = vec![0.0_f32; scan_block_rows];
    let mut z_col_major = vec![0.0_f64; scan_block_rows * n];
    let mut sv_block = vec![0.0_f64; scan_block_rows];
    let exact_scalar_p1 = scalar_spd_inv_from_chol(xt_w_x_chol, p);
    let mut c_block = vec![0.0_f64; scan_block_rows * p];
    let mut alpha = if exact_scalar_p1.is_some() {
        Vec::new()
    } else {
        vec![0.0_f64; p]
    };
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
    let mut next_progress_done = progress_done_offset.saturating_add(progress_step);
    while row_start < m {
        let row_end = (row_start + scan_block_rows).min(m);
        let rows_here = row_end - row_start;
        let block_slice = &mut block[..rows_here * n];
        let spy_slice = &mut spy_block[..rows_here];
        let sv_slice = &mut sv_block[..rows_here];

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
        // Step 3: widen decode block directly into the sparse-solve buffer
        let z_slice = &mut z_col_major[..rows_here * n];
        let t0 = stage_timing.then(Instant::now);
        row_major_snp_block_f32_to_col_major_f64(block_slice, rows_here, n, z_slice, pool.as_ref());
        if let Some(t0) = t0 {
            timing.repack_secs += t0.elapsed().as_secs_f64();
        }

        // Step 4: solve V * Z_block = G_block (tiled, in-place)
        let t0 = stage_timing.then(Instant::now);
        sparse_solve_rhs_tiled(
            factor,
            z_slice,
            rows_here,
            _solve_workspace,
            &mut tiled_workspaces,
            pool.as_deref(),
            threads.max(1),
        )?;
        if let Some(t0) = t0 {
            timing.solve_secs += t0.elapsed().as_secs_f64();
        }

        // Step 5: C_block = X^T Z_block; for p=1 also fuse g'Z here.
        let c_slice = &mut c_block[..rows_here * p];
        let t0 = stage_timing.then(Instant::now);
        if exact_scalar_p1.is_some() {
            let x0 = &x_design[..n];
            fused_block_sv_xtz_scalar_p1(
                block_slice,
                z_slice,
                x0,
                rows_here,
                n,
                sv_slice,
                &mut c_slice[..rows_here],
                pool.as_ref(),
            );
        } else {
            xt_mat_rhs_block(
                x_design,
                Some(x_design_col_major),
                z_slice,
                n,
                p,
                rows_here,
                c_slice,
                pool.as_ref(),
            );
        }
        if let Some(t0) = t0 {
            timing.xtz_secs += t0.elapsed().as_secs_f64();
        }

        // Step 6: batch g'Z for the generic p>1 path.
        if exact_scalar_p1.is_none() {
            let t0 = stage_timing.then(Instant::now);
            pairwise_block_dot_f32_f64(block_slice, z_slice, rows_here, n, sv_slice, pool.as_ref());
            if let Some(t0) = t0 {
                timing.denom_secs += t0.elapsed().as_secs_f64();
            }
        }

        // Steps 7-8: per-SNP denominator and output
        let out_slice = &mut out_block[..rows_here * 3];
        let t0 = stage_timing.then(Instant::now);
        if let Some(xtden00_inv) = exact_scalar_p1 {
            for local_idx in 0..rows_here {
                let s_v_s = sv_slice[local_idx];
                let xtz0 = c_slice[local_idx];
                let x_quad = xtz0 * xtz0 * xtden00_inv;
                let schur = (s_v_s - x_quad).max(0.0);
                let score = spy_slice[local_idx] as f64;
                let out_row = &mut out_slice[local_idx * 3..(local_idx + 1) * 3];
                if let Some((beta, se, pwald)) = splmm_wald_from_score_denom(score, schur, sigma2) {
                    out_row[0] = beta;
                    out_row[1] = se;
                    out_row[2] = pwald;
                } else {
                    out_row[0] = f64::NAN;
                    out_row[1] = f64::NAN;
                    out_row[2] = 1.0_f64;
                }
            }
        } else {
            for local_idx in 0..rows_here {
                let s_v_s = sv_slice[local_idx];
                let c_j = &c_slice[local_idx * p..(local_idx + 1) * p];
                cholesky_solve_into(xt_w_x_chol, p, c_j, &mut alpha);
                let x_quad = c_j
                    .iter()
                    .zip(alpha.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let schur = (s_v_s - x_quad).max(0.0);
                let score = spy_slice[local_idx] as f64;
                let out_row = &mut out_slice[local_idx * 3..(local_idx + 1) * 3];
                if let Some((beta, se, pwald)) = splmm_wald_from_score_denom(score, schur, sigma2) {
                    out_row[0] = beta;
                    out_row[1] = se;
                    out_row[2] = pwald;
                } else {
                    out_row[0] = f64::NAN;
                    out_row[1] = f64::NAN;
                    out_row[2] = 1.0_f64;
                }
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
        let done_abs = progress_done_offset.saturating_add(row_end);
        if scan_progress_callback.is_some() && (row_end == m || done_abs >= next_progress_done) {
            emit_progress_callback(
                scan_progress_callback,
                progress_stage,
                done_abs.min(progress_total),
                progress_total,
            )?;
            while next_progress_done <= done_abs {
                let advanced = next_progress_done.saturating_add(progress_step);
                if advanced == next_progress_done {
                    break;
                }
                next_progress_done = advanced;
            }
        }
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
fn scan_with_py_and_exact_p_sparse(
    factor: &SparseJxgrmCholesky,
    scan_prepared: &SplmmPreparedInput,
    x_design: &[f64],
    x_design_col_major: &[f64],
    py_vec: &[f64],
    ypy: f64,
    df: f64,
    xt_w_x_chol: &[f64],
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    threads: usize,
    block_rows: usize,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    progress_done_offset: usize,
    progress_total_override: usize,
    mmap_window_mb: Option<usize>,
    solve_workspace: &mut SparseJxgrmSolveWorkspace,
) -> Result<Vec<f64>, String> {
    let m = scan_prepared.n_rows();
    let mut out = vec![0.0_f64; m * 3];
    let mut memory_sink = |row_start: usize, rows_here: usize, block: &[f64]| {
        out[row_start * 3..][..rows_here * 3].copy_from_slice(block);
        Ok(())
    };
    let mut input = unified_input_from_splmm_prepared(scan_prepared, mmap_window_mb)?;
    exact_scan_blocks_core(
        factor,
        &mut input,
        x_design,
        x_design_col_major,
        py_vec,
        ypy,
        df,
        xt_w_x_chol,
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

    let decode_plan =
        AdditiveDecodePlan::from_sample_indices(input.stats.n_samples_full, scan_sample_idx);
    let sample_identity = decode_plan.sample_identity();
    let sample_byte_idx = decode_plan.sample_byte_idx();
    let sample_bit_shift = decode_plan.sample_bit_shift();
    let selected_excluded_sample_idx = decode_plan.selected_excluded_sample_indices();

    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let m = input.n_markers();
    let progress_total = if progress_total_override == 0 {
        m.max(1)
    } else {
        progress_total_override.max(1)
    };
    let (scan_block_hint, progress_step) =
        splmm_scan_block_hint_and_progress_step(block_rows, progress_every);
    if scan_progress_callback.is_some() {
        emit_progress_callback(
            scan_progress_callback,
            progress_stage,
            progress_done_offset.min(progress_total),
            progress_total,
        )?;
    }

    let scan_block_rows = adaptive_grm_block_rows(scan_block_hint, m, n, 0usize, threads).max(1);
    let mut next_progress_done = progress_done_offset.saturating_add(progress_step);

    if matches!(gm, PackedGeneticModel::Add) {
        let rhs_f32 = pack_score_design_rhs_f32(score_vec, x_design, p)?;
        let rhs_cols = p + 1;
        let mut block = vec![0.0_f32; scan_block_rows * n];
        let mut row_ss = vec![0.0_f64; scan_block_rows];
        let mut dot_block = vec![0.0_f32; scan_block_rows * rhs_cols];
        let mut out_block = vec![0.0_f64; scan_block_rows * 3];
        let thread_hint = threads.max(1);
        let row_tile = scan_block_rows.div_ceil(thread_hint).clamp(32, 1024);
        let scalar_xtx00_inv = scalar_spd_inv_from_chol(xtx_chol, p);
        let mut row_start = 0usize;
        while row_start < m {
            let row_end = (row_start + scan_block_rows).min(m);
            let rows_here = row_end - row_start;
            let block_slice = &mut block[..rows_here * n];
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
            let packed_ss_ok = packed_block_additive_sumsq_from_counts(
                input,
                row_start,
                rows_here,
                scan_sample_idx,
                sample_identity,
                selected_excluded_sample_idx,
                ss_slice,
                pool.as_ref(),
            );
            if !packed_ss_ok {
                row_major_block_sumsq_f64(block_slice, rows_here, n, ss_slice, pool.as_ref());
            }
            if let Some(t0) = t0 {
                timing.row_sumsq_secs += t0.elapsed().as_secs_f64();
            }
            let dot_slice = &mut dot_block[..rows_here * rhs_cols];
            let t0 = stage_timing.then(Instant::now);
            // Compute g'Py and X'g for the whole decoded block in one pass.
            row_major_block_mul_mat_f32(
                block_slice,
                rows_here,
                n,
                rhs_f32.as_slice(),
                rhs_cols,
                dot_slice,
                pool.as_ref(),
            );
            if let Some(t0) = t0 {
                timing.gx_secs += t0.elapsed().as_secs_f64();
            }
            let out_slice = &mut out_block[..rows_here * 3];
            let t0 = stage_timing.then(Instant::now);
            let mut run_assoc = || {
                if let Some(xtx00_inv) = scalar_xtx00_inv {
                    out_slice.par_chunks_mut(row_tile * 3).enumerate().for_each(
                        |(tile_idx, out_tile)| {
                            let tile_row_start = tile_idx * row_tile;
                            for (local_off, out_row) in out_tile.chunks_mut(3).enumerate() {
                                let local_idx = tile_row_start + local_off;
                                let dot_row =
                                    &dot_slice[local_idx * rhs_cols..(local_idx + 1) * rhs_cols];
                                let score = score_scale * (dot_row[0] as f64);
                                let xts0 = dot_row[1] as f64;
                                let s_m_s =
                                    residualized_sumsq_scalar(xtx00_inv, xts0, ss_slice[local_idx]);
                                let denom_scaled = denom_scale * s_m_s;
                                if let Some((beta, se, pwald)) =
                                    splmm_wald_from_score_denom(score, denom_scaled, wald_sigma2)
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
                        },
                    );
                } else {
                    out_slice
                        .par_chunks_mut(row_tile * 3)
                        .enumerate()
                        .for_each_init(
                            || (vec![0.0_f64; p], vec![0.0_f64; p]),
                            |(xts, alpha), (tile_idx, out_tile)| {
                                let tile_row_start = tile_idx * row_tile;
                                for (local_off, out_row) in out_tile.chunks_mut(3).enumerate() {
                                    let local_idx = tile_row_start + local_off;
                                    let dot_row = &dot_slice
                                        [local_idx * rhs_cols..(local_idx + 1) * rhs_cols];
                                    for j in 0..p {
                                        xts[j] = dot_row[j + 1] as f64;
                                    }
                                    let s_m_s = residualized_sumsq_from_xtx_chol(
                                        xtx_chol,
                                        p,
                                        xts,
                                        alpha,
                                        ss_slice[local_idx],
                                    );
                                    let score = score_scale * (dot_row[0] as f64);
                                    let denom_scaled = denom_scale * s_m_s;
                                    if let Some((beta, se, pwald)) = splmm_wald_from_score_denom(
                                        score,
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
                            },
                        );
                }
            };
            if let Some(tp) = &pool {
                tp.install(run_assoc);
            } else {
                run_assoc();
            }
            if let Some(t0) = t0 {
                timing.assoc_secs += t0.elapsed().as_secs_f64();
            }
            let t0 = stage_timing.then(Instant::now);
            sink(row_start, rows_here, out_slice)?;
            if let Some(t0) = t0 {
                timing.sink_secs += t0.elapsed().as_secs_f64();
            }
            let done_abs = progress_done_offset.saturating_add(row_end);
            if scan_progress_callback.is_some() && (row_end == m || done_abs >= next_progress_done)
            {
                emit_progress_callback(
                    scan_progress_callback,
                    progress_stage,
                    done_abs.min(progress_total),
                    progress_total,
                )?;
                while next_progress_done <= done_abs {
                    let advanced = next_progress_done.saturating_add(progress_step);
                    if advanced == next_progress_done {
                        break;
                    }
                    next_progress_done = advanced;
                }
            }
            row_start = row_end;
        }
    } else {
        let thread_hint = threads.max(1);
        let row_tile = scan_block_rows.div_ceil(thread_hint).clamp(32, 1024);
        let mut out_block = vec![0.0_f64; scan_block_rows * 3];
        let mut row_start = 0usize;
        while row_start < m {
            let row_end = (row_start + scan_block_rows).min(m);
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
                                decode_packed_row_model_enum_into_f64(
                                    row,
                                    input.stats.row_flip[row_idx],
                                    input.stats.maf[row_idx],
                                    n,
                                    gm,
                                    sample_identity,
                                    sample_byte_idx,
                                    sample_bit_shift,
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
                                    splmm_wald_from_score_denom(score, denom_scaled, wald_sigma2)
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
            let done_abs = progress_done_offset.saturating_add(row_end);
            if scan_progress_callback.is_some() && (row_end == m || done_abs >= next_progress_done)
            {
                emit_progress_callback(
                    scan_progress_callback,
                    progress_stage,
                    done_abs.min(progress_total),
                    progress_total,
                )?;
                while next_progress_done <= done_abs {
                    let advanced = next_progress_done.saturating_add(progress_step);
                    if advanced == next_progress_done {
                        break;
                    }
                    next_progress_done = advanced;
                }
            }
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
pub(crate) fn scan_with_py_and_rhat(
    scan_prepared: &SplmmPreparedInput,
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
    mmap_window_mb: Option<usize>,
) -> Result<Vec<f64>, String> {
    let mut input = unified_input_from_splmm_prepared(scan_prepared, mmap_window_mb)?;
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
pub(crate) fn scan_to_tsv_with_py_and_rhat(
    scan_prepared: &SplmmPreparedInput,
    x_design: &[f64],
    py_vec: &[f64],
    xtx_chol: &[f64],
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    r_hat: f64,
    threads: usize,
    block_rows: usize,
    meta_input: SplmmTsvMetaInput,
    out_tsv: &str,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    mmap_window_mb: Option<usize>,
) -> Result<(usize, SplmmTsvTiming), String> {
    let stage_timing = splmm_packed_stage_timing_enabled();
    let mut input = unified_input_from_splmm_prepared(scan_prepared, mmap_window_mb)?;
    let m = input.n_markers();
    run_splmm_async_tsv_writer(
        out_tsv,
        block_rows,
        threads,
        meta_input,
        scan_prepared.row_maf.as_ref(),
        scan_prepared.row_missing.as_ref(),
        gm,
        "approx",
        stage_timing,
        |tsv_sink| {
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
                0,
                0,
                tsv_sink,
            )?;
            Ok(m)
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn scan_to_tsv_with_py_and_exact_p_sparse(
    factor: &SparseJxgrmCholesky,
    scan_prepared: &SplmmPreparedInput,
    x_design: &[f64],
    x_design_col_major: &[f64],
    py_vec: &[f64],
    ypy: f64,
    df: f64,
    xt_w_x_chol: &[f64],
    scan_sample_idx: &[usize],
    gm: PackedGeneticModel,
    threads: usize,
    block_rows: usize,
    meta_input: SplmmTsvMetaInput,
    out_tsv: &str,
    scan_progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    progress_stage: usize,
    mmap_window_mb: Option<usize>,
    solve_workspace: &mut SparseJxgrmSolveWorkspace,
) -> Result<(usize, SplmmTsvTiming), String> {
    let stage_timing = splmm_packed_stage_timing_enabled();
    let m = scan_prepared.n_rows();
    let mut input = unified_input_from_splmm_prepared(scan_prepared, mmap_window_mb)?;
    run_splmm_async_tsv_writer(
        out_tsv,
        block_rows,
        threads,
        meta_input,
        scan_prepared.row_maf.as_ref(),
        scan_prepared.row_missing.as_ref(),
        gm,
        "exact",
        stage_timing,
        |tsv_sink| {
            exact_scan_blocks_core(
                factor,
                &mut input,
                x_design,
                x_design_col_major,
                py_vec,
                ypy,
                df,
                xt_w_x_chol,
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
                tsv_sink,
            )?;
            Ok(m)
        },
    )
}

struct SplmmPreparedScanState {
    factor: SparseJxgrmCholesky,
    solve_workspace: SparseJxgrmSolveWorkspace,
    x_design_col_major: Vec<f64>,
    null_model: PcgSplmmNullModel,
    null_info: PcgSplmmNullModelInfo,
    prepare_timing: SplmmPrepareTiming,
}

struct SplmmBuiltNullState {
    x_design_col_major: Vec<f64>,
    py: Vec<f64>,
    beta_hat: Vec<f64>,
    null_model: PcgSplmmNullModel,
    null_info: PcgSplmmNullModelInfo,
    timing: SplmmNullStateTiming,
}

fn build_sparse_splmm_null_state(
    factor: &SparseJxgrmCholesky,
    x_design: &[f64],
    y_vec: &[f64],
    solve_workspace: &mut SparseJxgrmSolveWorkspace,
    stage1_cb: Option<&Py<PyAny>>,
) -> Result<SplmmBuiltNullState, String> {
    let total_t0 = Instant::now();
    let mut timing = SplmmNullStateTiming::default();
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
    if n <= p {
        return Err(format!(
            "SparseLMM null-state build requires n > p, got n={n}, p={p}"
        ));
    }
    let df = (n - p) as f64;

    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 6, 0, 1)?;
    }
    let solve_y_t0 = Instant::now();
    let y_vinv_col = sparse_solve_rhs_with_workspace(factor, y_vec, 1, solve_workspace)?;
    timing.solve_y_secs = solve_y_t0.elapsed().as_secs_f64();
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 6, 1, 1)?;
    }

    let x_design_col_major = row_major_to_col_major_f64(&x_design, n, p)?;
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 7, 0, p.max(1))?;
    }
    let solve_x_t0 = Instant::now();
    let x_vinv_col =
        sparse_solve_rhs_with_workspace(factor, &x_design_col_major, p, solve_workspace)?;
    timing.solve_x_secs = solve_x_t0.elapsed().as_secs_f64();
    if stage1_cb.is_some() {
        emit_progress_callback(stage1_cb, 7, p.max(1), p.max(1))?;
    }

    let mut xt_v_inv_y = vec![0.0_f64; p];
    let xt_v_inv_y_t0 = Instant::now();
    xt_vec_row_major(&x_design, n, p, &y_vinv_col, &mut xt_v_inv_y);
    timing.xt_v_inv_y_secs = xt_v_inv_y_t0.elapsed().as_secs_f64();

    let mut xt_v_inv_x = vec![0.0_f64; p * p];
    let xt_v_inv_x_t0 = Instant::now();
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
    timing.xt_v_inv_x_secs = xt_v_inv_x_t0.elapsed().as_secs_f64();
    let chol_t0 = Instant::now();
    let xt_v_inv_x_chol = spd_cholesky_with_jitter(&xt_v_inv_x, p, "SparseLMM XtVinvX")?;
    timing.chol_secs = chol_t0.elapsed().as_secs_f64();
    drop(xt_v_inv_x);

    let mut beta_hat = vec![0.0_f64; p];
    let beta_py_t0 = Instant::now();
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
    timing.beta_py_secs = beta_py_t0.elapsed().as_secs_f64();

    let scale_t0 = Instant::now();
    let ypy = y_vec
        .iter()
        .zip(py.iter())
        .map(|(y, py_i)| y * py_i)
        .sum::<f64>()
        .max(0.0_f64);
    if !(ypy.is_finite() && ypy > 0.0_f64) {
        return Err(format!(
            "SparseLMM null-state build produced invalid yPy on K + lambda I scale: {ypy}"
        ));
    }
    let sigma2 = ypy / df;
    if !(sigma2.is_finite() && sigma2 > 0.0_f64) {
        return Err(format!(
            "SparseLMM null-state build produced invalid sigma2 on K + lambda I scale: {sigma2}"
        ));
    }
    timing.scale_secs = scale_t0.elapsed().as_secs_f64();
    timing.total_secs = total_t0.elapsed().as_secs_f64();

    let null_model = PcgSplmmNullModel {
        df,
        ypy,
        sigma2,
        py: py.clone(),
        xt_w_x_chol: xt_v_inv_x_chol,
    };
    let null_info = PcgSplmmNullModelInfo {
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
        timing,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_splmm_scan_state(
    factor: SparseJxgrmCholesky,
    scan_prepared: &SplmmPreparedInput,
    x_design: &[f64],
    y_vec: &[f64],
    threads: usize,
    block_rows: usize,
    stage1_cb: Option<&Py<PyAny>>,
    scan_mode: SplmmScanMode,
) -> Result<SplmmPreparedScanState, String> {
    let total_t0 = Instant::now();
    let mut prepare_timing = SplmmPrepareTiming::default();
    if scan_mode != SplmmScanMode::Exact {
        return Err(format!(
            "SparseLMM internal error: prepare_splmm_scan_state only supports exact mode, got {}",
            scan_mode.as_str()
        ));
    }
    let n = y_vec.len();
    let p = x_design.len() / n;
    let m = scan_prepared.n_rows();
    let scan_block_hint = block_rows.max(512).max(1);
    let est_block_rows = adaptive_exact_block_rows(
        adaptive_grm_block_rows(scan_block_hint, m, n, 0usize, threads).max(1),
        n,
    );
    let solve_cap = est_block_rows.max(p).max(1);
    let workspace_t0 = Instant::now();
    let mut solve_workspace = factor.make_solve_workspace(solve_cap)?;
    prepare_timing.workspace_secs = workspace_t0.elapsed().as_secs_f64();
    let null_state =
        build_sparse_splmm_null_state(&factor, x_design, y_vec, &mut solve_workspace, stage1_cb)?;
    prepare_timing.null_state_secs = null_state.timing.total_secs;
    let x_design_col_major = null_state.x_design_col_major;
    let null_model = null_state.null_model;
    let null_info = null_state.null_info;
    prepare_timing.null_state = null_state.timing;
    prepare_timing.total_secs = total_t0.elapsed().as_secs_f64();
    Ok(SplmmPreparedScanState {
        factor,
        solve_workspace,
        x_design_col_major,
        null_model,
        null_info,
        prepare_timing,
    })
}

#[allow(clippy::too_many_arguments)]
fn estimate_rhat_and_scan_sparse(
    factor: SparseJxgrmCholesky,
    scan_prepared: SplmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    scan_sample_idx: Vec<usize>,
    gm: PackedGeneticModel,
    _lbd: f64,
    threads: usize,
    block_rows: usize,
    _rhat_markers: usize,
    _rhat_seed: u64,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    scan_mode: SplmmScanMode,
    mmap_window_mb: Option<usize>,
) -> Result<SplmmScanResult, String> {
    if scan_mode != SplmmScanMode::Exact {
        return Err(format!(
            "SparseLMM internal error: estimate_rhat_and_scan_sparse only supports exact mode, got {}",
            scan_mode.as_str()
        ));
    }
    let stage1_cb = stage1_progress_callback.as_ref();
    let factor_nnz = factor.factor_nnz();
    let mut state = prepare_splmm_scan_state(
        factor,
        &scan_prepared,
        &x_design,
        &y_vec,
        threads,
        block_rows,
        stage1_cb,
        scan_mode,
    )?;
    if splmm_prepare_stage_timing_enabled() {
        emit_splmm_prepare_timing(
            scan_mode.as_str(),
            &state.prepare_timing,
            scan_prepared.n_rows(),
            y_vec.len(),
            x_design.len() / y_vec.len(),
        );
    }
    let out = scan_with_py_and_exact_p_sparse(
        &state.factor,
        &scan_prepared,
        &x_design,
        &state.x_design_col_major,
        &state.null_model.py,
        state.null_model.ypy,
        state.null_model.df,
        &state.null_model.xt_w_x_chol,
        &scan_sample_idx,
        gm,
        threads,
        block_rows,
        scan_progress_callback.as_ref(),
        progress_every,
        9,
        0,
        0,
        mmap_window_mb,
        &mut state.solve_workspace,
    )?;
    Ok((
        f64::NAN,
        out,
        state.null_info,
        trivial_pcg_rhat_info(y_vec.len(), 0, 0),
        factor_nnz,
    ))
}

fn estimate_rhat_and_scan(
    operator_prepared: SplmmPreparedInput,
    scan_prepared: SplmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    operator_sample_idx: Vec<usize>,
    scan_sample_idx: Vec<usize>,
    sparse_factor_sample_idx: Option<Vec<usize>>,
    gm: PackedGeneticModel,
    lbd: f64,
    block_rows: usize,
    threads: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    sparse_jxgrm_path: Option<String>,
    scan_mode: SplmmScanMode,
    approx_fit_tol: f64,
    approx_fit_max_iter: usize,
    mmap_window_mb: Option<usize>,
) -> Result<SplmmScanResult, String> {
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
        lbd,
        threads,
        stage1_cb,
    )?;
    if scan_mode == SplmmScanMode::Approx {
        return estimate_residualized_approx_scan_sparse(
            factor,
            scan_prepared,
            x_design,
            y_vec,
            scan_sample_idx,
            gm,
            lbd,
            threads,
            block_rows,
            rhat_markers,
            rhat_seed,
            stage1_progress_callback,
            scan_progress_callback,
            progress_every,
            approx_fit_tol,
            approx_fit_max_iter,
            mmap_window_mb,
        );
    }
    estimate_rhat_and_scan_sparse(
        factor,
        scan_prepared,
        x_design,
        y_vec,
        scan_sample_idx,
        gm,
        lbd,
        threads,
        block_rows,
        rhat_markers,
        rhat_seed,
        stage1_progress_callback,
        scan_progress_callback,
        progress_every,
        scan_mode,
        mmap_window_mb,
    )
}

#[allow(clippy::too_many_arguments)]
fn estimate_rhat_and_scan_to_tsv(
    operator_prepared: SplmmPreparedInput,
    scan_prepared: SplmmPreparedInput,
    x_design: Vec<f64>,
    y_vec: Vec<f64>,
    operator_sample_idx: Vec<usize>,
    scan_sample_idx: Vec<usize>,
    sparse_factor_sample_idx: Option<Vec<usize>>,
    gm: PackedGeneticModel,
    lbd: f64,
    threads: usize,
    block_rows: usize,
    rhat_markers: usize,
    rhat_seed: u64,
    meta_input: SplmmTsvMetaInput,
    out_tsv: String,
    stage1_progress_callback: Option<Py<PyAny>>,
    scan_progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    sparse_jxgrm_path: Option<String>,
    scan_mode: SplmmScanMode,
    approx_fit_tol: f64,
    approx_fit_max_iter: usize,
    mmap_window_mb: Option<usize>,
) -> Result<SplmmScanToTsvResult, String> {
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
    let factor_load_t0 = Instant::now();
    let factor = sparse_splmm_load_factor(
        prefix,
        sparse_jxgrm_path.as_deref(),
        sparse_expected_n,
        sparse_factor_sample_idx_ref,
        lbd,
        threads,
        stage1_cb,
    )?;
    let factor_load_secs = factor_load_t0.elapsed().as_secs_f64();
    if scan_mode == SplmmScanMode::Approx {
        return estimate_residualized_approx_scan_to_tsv_sparse(
            factor,
            scan_prepared,
            x_design,
            y_vec,
            scan_sample_idx,
            gm,
            lbd,
            threads,
            block_rows,
            rhat_markers,
            rhat_seed,
            meta_input,
            out_tsv,
            stage1_progress_callback,
            scan_progress_callback,
            progress_every,
            approx_fit_tol,
            approx_fit_max_iter,
            factor_load_secs,
            mmap_window_mb,
        );
    }
    if scan_mode != SplmmScanMode::Exact {
        return Err(format!(
            "SparseLMM internal error: estimate_rhat_and_scan_to_tsv only supports exact mode here, got {}",
            scan_mode.as_str()
        ));
    }
    let prepare_t0 = Instant::now();
    let mut state = prepare_splmm_scan_state(
        factor,
        &scan_prepared,
        &x_design,
        &y_vec,
        threads,
        block_rows,
        stage1_cb,
        scan_mode,
    )?;
    let scan_prepare_secs = prepare_t0.elapsed().as_secs_f64();
    if splmm_prepare_stage_timing_enabled() {
        emit_splmm_prepare_timing(
            scan_mode.as_str(),
            &state.prepare_timing,
            scan_prepared.n_rows(),
            y_vec.len(),
            x_design.len() / y_vec.len(),
        );
    }
    let total_rows = scan_prepared.n_rows();
    let scan_exec_t0 = Instant::now();
    let (written_rows, tsv_timing) = scan_to_tsv_with_py_and_exact_p_sparse(
        &state.factor,
        &scan_prepared,
        &x_design,
        &state.x_design_col_major,
        &state.null_model.py,
        state.null_model.ypy,
        state.null_model.df,
        &state.null_model.xt_w_x_chol,
        &scan_sample_idx,
        gm,
        threads,
        block_rows,
        meta_input,
        &out_tsv,
        scan_progress_callback.as_ref(),
        progress_every,
        9,
        mmap_window_mb,
        &mut state.solve_workspace,
    )?;
    let scan_exec_secs = (scan_exec_t0.elapsed().as_secs_f64() - tsv_timing.finish_secs).max(0.0);
    if splmm_top_level_timing_enabled() {
        emit_splmm_null_scan_core_timing(
            scan_mode.as_str(),
            factor_load_secs,
            scan_prepare_secs,
            scan_exec_secs,
            tsv_timing.finish_secs,
            total_rows,
            y_vec.len(),
            x_design.len() / y_vec.len(),
            threads.max(1),
        );
    }
    Ok(SplmmScanToTsvResult {
        r_hat: f64::NAN,
        written_rows,
        factor_nnz: state.factor.factor_nnz(),
        null_info: state.null_info,
        rhat_info: trivial_pcg_rhat_info(y_vec.len(), 0, 0),
        factor_load_secs,
        scan_prepare_secs,
        scan_exec_secs,
        writer_wait_secs: tsv_timing.finish_secs,
    })
}

#[pyfunction]
#[pyo3(signature = (
    jxgrm_path,
    sample_indices=None
))]
pub fn splmm_load_sparse_grm_subset_dense<'py>(
    py: Python<'py>,
    jxgrm_path: String,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let sample_idx_raw = if let Some(sidx) = sample_indices {
        Some(sidx.as_slice()?.to_vec())
    } else {
        None
    };

    let (n, dense_flat) = py
        .detach(move || -> Result<(usize, Vec<f64>), String> {
            let csc = MmapSparseGrmCsc::open(&jxgrm_path)?;
            if let Some(raw) = sample_idx_raw.as_deref() {
                let sample_idx = parse_index_vec_i64(raw, csc.n_samples(), "sample_indices")
                    .map_err(|e| e.to_string())?;
                let subset = subset_sparse_grm_csc(&csc, sample_idx.as_slice())?;
                Ok((subset.n_samples, sparse_grm_to_dense(&subset)))
            } else {
                Ok((csc.n_samples(), sparse_grm_to_dense(&csc)))
            }
        })
        .map_err(PyRuntimeError::new_err)?;

    let arr = numpy::ndarray::Array2::from_shape_vec((n, n), dense_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
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

    if !(sigma_g2.is_finite() && sigma_g2 > 0.0 && sigma_e2.is_finite() && sigma_e2 >= 0.0) {
        return Err(PyRuntimeError::new_err(format!(
            "SparseLMM null debug requires finite sigma_g2 > 0 and sigma_e2 >= 0, got sigma_g2={sigma_g2}, sigma_e2={sigma_e2}"
        )));
    }
    let lbd = sigma_e2 / sigma_g2;
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
                lbd,
                1,
                None,
            )?;
            let mut solve_workspace = factor.make_solve_workspace(p.max(1))?;
            build_sparse_splmm_null_state(
                &factor,
                x_design_for_factor.as_slice(),
                y_vec_for_factor.as_slice(),
                &mut solve_workspace,
                None,
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    let p0y = built
        .null_model
        .py
        .iter()
        .map(|v| *v * built.null_model.sigma2)
        .collect::<Vec<_>>();
    Ok((
        built.null_model.sigma2,
        PyArray1::from_owned_array(py, numpy::ndarray::Array1::from_vec(built.py)),
        PyArray1::from_owned_array(py, numpy::ndarray::Array1::from_vec(p0y)),
        PyArray1::from_owned_array(py, numpy::ndarray::Array1::from_vec(built.beta_hat)),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    jxgrm_path,
    y,
    x_cov=None,
    sample_indices=None,
    low=-5.0,
    high=5.0,
    grid_size=7,
    tol=1e-3,
    max_iter=200,
    threads=1
))]
pub fn splmm_residualized_approx_null_fit_from_jxgrm<'py>(
    py: Python<'py>,
    jxgrm_path: String,
    y: PyReadonlyArray1<'py, f64>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    low: f64,
    high: f64,
    grid_size: usize,
    tol: f64,
    max_iter: usize,
    threads: usize,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64)> {
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
                "SparseLMM residualized approx null fit requires y length to match sparse n when sample_indices is omitted: y={n}, sparse_n={sparse_n}"
            )));
        }
        (0..sparse_n).collect::<Vec<_>>()
    };
    if sample_idx.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "SparseLMM residualized approx null fit sample/y mismatch: sample_indices={}, y={n}",
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
            "n must be > p for SparseLMM residualized approx null fit: n={n}, p={p}"
        )));
    }
    let df_reml = (n - p) as f64;

    let sample_idx_for_fit = sample_idx.clone();
    let x_design_for_fit = x_design.clone();
    let y_vec_for_fit = y_vec.clone();
    let jxgrm_path_for_fit = jxgrm_path.clone();
    py.detach(move || {
        let analysis = sparse_splmm_load_analysis(
            &jxgrm_path_for_fit,
            n,
            sample_idx_for_fit.as_slice(),
            None,
        )?;
        let fit = fit_sparse_reml_on_residualized_response(
            analysis.as_ref(),
            x_design_for_fit.as_slice(),
            y_vec_for_fit.as_slice(),
            low,
            high,
            grid_size,
            tol,
            max_iter,
            threads.max(1),
        )?;
        let log10_lambda = fit.lambda.log10();
        Ok((
            fit.lambda,
            fit.sigma_g2,
            fit.sigma_e2,
            fit.ml,
            fit.reml,
            log10_lambda,
            df_reml,
        ))
    })
    .map_err(|e: String| PyRuntimeError::new_err(e))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    lbd,
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
    scan_mode="exact",
    mmap_window_mb=None
))]
pub fn splmm_assoc_pcg_bed<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    lbd: f64,
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
    mmap_window_mb: Option<usize>,
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
    usize,
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
    if !(lbd.is_finite() && lbd >= 0.0) {
        return Err(PyRuntimeError::new_err("lbd must be finite and >= 0"));
    }
    let scan_mode = SplmmScanMode::parse(scan_mode)?;
    if scan_mode.needs_rhat() && rhat_markers == 0 {
        return Err(PyRuntimeError::new_err("rhat_markers must be > 0"));
    }
    let _ = (std_eps, use_train_maf, rhat_tol);
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
        mmap_window_mb,
    )?;
    let sparse_factor_sample_idx = parse_optional_index_array(
        sparse_sample_indices.as_ref(),
        prepared.operator_prepared.n_samples_full,
        "sparse_sample_indices",
    )?;
    emit_progress_callback(stage1_progress_callback.as_ref(), 0, 1, 1)
        .map_err(PyRuntimeError::new_err)?;

    let (r_hat, out, null_info, rhat_info, factor_nnz) = py
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
                lbd,
                block_rows,
                threads,
                rhat_markers,
                rhat_seed,
                stage1_progress_callback,
                scan_progress_callback,
                progress_every,
                sparse_jxgrm_path,
                scan_mode,
                tol,
                max_iter,
                mmap_window_mb,
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
        factor_nnz,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    lbd,
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
    scan_mode="exact",
    mmap_window_mb=None
))]
pub fn splmm_assoc_pcg_bed_to_tsv<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    lbd: f64,
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
    mmap_window_mb: Option<usize>,
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
    usize,
    (f64, f64, f64, f64, f64, f64, f64),
    usize,
)> {
    let top_level_timing = splmm_top_level_timing_enabled();
    let total_t0 = top_level_timing.then(Instant::now);
    let mut top_timing = SplmmAssocTopLevelTiming::default();
    if max_iter == 0 {
        return Err(PyRuntimeError::new_err("max_iter must be > 0"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be finite and > 0"));
    }
    if !(std_eps.is_finite() && std_eps > 0.0) {
        return Err(PyRuntimeError::new_err("std_eps must be finite and > 0"));
    }
    if !(lbd.is_finite() && lbd >= 0.0) {
        return Err(PyRuntimeError::new_err("lbd must be finite and >= 0"));
    }
    let scan_mode = SplmmScanMode::parse(scan_mode)?;
    if scan_mode.needs_rhat() && rhat_markers == 0 {
        return Err(PyRuntimeError::new_err("rhat_markers must be > 0"));
    }
    let _ = (std_eps, use_train_maf, rhat_tol);

    emit_progress_callback(stage1_progress_callback.as_ref(), 0, 0, 1)
        .map_err(PyRuntimeError::new_err)?;
    let t0 = top_level_timing.then(Instant::now);
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
        mmap_window_mb,
    )?;
    let sparse_factor_sample_idx = parse_optional_index_array(
        sparse_sample_indices.as_ref(),
        prepared.operator_prepared.n_samples_full,
        "sparse_sample_indices",
    )?;
    if let Some(t0) = t0 {
        top_timing.prepare_inputs_secs += t0.elapsed().as_secs_f64();
    }
    emit_progress_callback(stage1_progress_callback.as_ref(), 0, 1, 1)
        .map_err(PyRuntimeError::new_err)?;
    let scan_rows = prepared.scan_prepared.n_rows();
    let n = prepared.y_vec.len();
    let p = prepared.x_design.len() / n;

    // Auto-populate chrom/pos/snp/allele from BIM sequentially when the caller
    // passes empty arrays, so we do not allocate full per-row String vectors.
    let t0 = top_level_timing.then(Instant::now);
    let meta_input = build_splmm_tsv_meta_input(
        &prefix,
        &prepared.scan_prepared,
        chrom,
        pos,
        snp,
        allele0,
        allele1,
    )
    .map_err(PyRuntimeError::new_err)?;
    if let Some(t0) = t0 {
        top_timing.bim_meta_secs += t0.elapsed().as_secs_f64();
    }

    let t0 = top_level_timing.then(Instant::now);
    let scan_out = py
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
                lbd,
                threads,
                block_rows,
                rhat_markers,
                rhat_seed,
                meta_input,
                out_tsv,
                stage1_progress_callback,
                scan_progress_callback,
                progress_every,
                sparse_jxgrm_path,
                scan_mode,
                tol,
                max_iter,
                mmap_window_mb,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    let detach_secs = t0.map(|t0| t0.elapsed().as_secs_f64()).unwrap_or(0.0_f64);
    top_timing.writer_wait_secs = scan_out.writer_wait_secs;
    top_timing.null_scan_core_secs =
        scan_out.factor_load_secs + scan_out.scan_prepare_secs + scan_out.scan_exec_secs;
    top_timing.detach_wall_secs = detach_secs;
    top_timing.total_secs = total_t0
        .map(|t0| t0.elapsed().as_secs_f64())
        .unwrap_or_else(|| {
            top_timing.prepare_inputs_secs
                + top_timing.bim_meta_secs
                + top_timing.null_scan_core_secs
                + top_timing.writer_wait_secs
        });
    let accounted_secs = top_timing.prepare_inputs_secs
        + top_timing.bim_meta_secs
        + top_timing.null_scan_core_secs
        + top_timing.writer_wait_secs;
    top_timing.other_secs = (top_timing.total_secs - accounted_secs).max(0.0_f64);
    if top_level_timing {
        emit_splmm_assoc_top_level_timing(
            scan_mode.as_str(),
            &top_timing,
            scan_rows,
            n,
            p,
            threads.max(1),
        );
    }
    let SplmmScanToTsvResult {
        r_hat,
        written_rows,
        factor_nnz,
        null_info,
        rhat_info,
        ..
    } = scan_out;

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
        (
            top_timing.prepare_inputs_secs,
            top_timing.bim_meta_secs,
            top_timing.null_scan_core_secs,
            top_timing.writer_wait_secs,
            top_timing.detach_wall_secs,
            top_timing.other_secs,
            top_timing.total_secs,
        ),
        factor_nnz,
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
                None,
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
// gload::GenotypeMatrix adapter for SplmmPreparedInput
// ---------------------------------------------------------------------------
// Zero-copy bridge: lets existing code paths call functions that expect a
// UnifiedInput<G> without cloning the metadata vectors.  New modules should
// construct a UnifiedInput<BedMmapMatrix> or UnifiedInput<PackedBedMatrix>
// directly via gload::open_bed_mmap_unified() and pass that to
// block-decode helpers.  Long-term SplmmPreparedInput should be retired.

/// Build a GlobalStats from a SplmmPreparedInput (clones vectors once).
fn unified_stats_from_splmm_prepared(p: &SplmmPreparedInput) -> crate::gload::GlobalStats {
    let row_source_indices = p
        .row_source_indices
        .as_ref()
        .map(|idx| idx.as_ref().to_vec())
        .unwrap_or_else(|| (0..p.row_maf.len()).collect());
    crate::gload::GlobalStats {
        maf: p.row_maf.as_ref().to_vec(),
        miss: p.row_missing.as_ref().to_vec(),
        row_flip: p.row_flip.as_ref().to_vec(),
        row_source_indices,
        site_keep: Vec::new(),
        n_samples_full: p.n_samples_full,
        n_markers_total: 0,
        bytes_per_snp: p.bytes_per_snp,
    }
}

#[inline]
fn unified_input_from_splmm_prepared(
    prepared: &SplmmPreparedInput,
    mmap_window_mb: Option<usize>,
) -> Result<UnifiedInput<SplmmPreparedMatrixAdapter<'_>>, String> {
    let matrix = if let Some(window_mb) = mmap_window_mb.filter(|&v| v > 0) {
        if let Some(prefix) = prepared.bed_prefix.as_deref() {
            SplmmPreparedMatrixAdapter::Windowed {
                matrix: WindowedBedMatrix::open(prefix, window_mb)?,
            }
        } else {
            SplmmPreparedMatrixAdapter::Direct { inner: prepared }
        }
    } else {
        if prepared.payload.is_none() {
            return Err(
                "SparseLMM direct BED scan requires a resident BED payload; provide mmap_window_mb to use windowed scanning."
                    .to_string(),
            );
        }
        SplmmPreparedMatrixAdapter::Direct { inner: prepared }
    };
    Ok(UnifiedInput {
        matrix,
        stats: unified_stats_from_splmm_prepared(prepared),
    })
}

/// Wraps a SparseLMM prepared input as a [`GenotypeMatrix`].
enum SplmmPreparedMatrixAdapter<'a> {
    Direct { inner: &'a SplmmPreparedInput },
    Windowed { matrix: WindowedBedMatrix },
}

impl GenotypeMatrix for SplmmPreparedMatrixAdapter<'_> {
    fn n_samples_full(&self) -> usize {
        match self {
            Self::Direct { inner } => inner.n_samples_full,
            Self::Windowed { matrix } => matrix.n_samples_full(),
        }
    }
    fn bytes_per_snp(&self) -> usize {
        match self {
            Self::Direct { inner } => inner.bytes_per_snp,
            Self::Windowed { matrix } => matrix.bytes_per_snp(),
        }
    }
    fn packed_flat(&self) -> &[u8] {
        match self {
            Self::Direct { inner } => inner
                .payload_bytes()
                .expect("direct SparseLMM matrix adapter requires resident BED payload"),
            Self::Windowed { matrix } => matrix.packed_flat(),
        }
    }
    fn source_row_bytes(&self, source_idx: usize) -> &[u8] {
        match self {
            Self::Direct { inner } => {
                let offset = source_idx.saturating_mul(inner.bytes_per_snp);
                &inner
                    .payload_bytes()
                    .expect("direct SparseLMM matrix adapter requires resident BED payload")
                    [offset..][..inner.bytes_per_snp]
            }
            Self::Windowed { matrix } => matrix.source_row_bytes(source_idx),
        }
    }

    fn decode_additive_block(
        &mut self,
        stats: &crate::gload::GlobalStats,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        match self {
            Self::Direct { inner } => {
                let payload = inner
                    .payload_bytes()
                    .expect("direct SparseLMM matrix adapter requires resident BED payload");
                let cols = if sample_identity {
                    stats.n_samples_full
                } else {
                    sample_idx.len()
                };
                let rows_here = out.len().saturating_div(cols.max(1));
                if rows_here == 0 {
                    return Ok(());
                }
                let code4_lut = &packed_byte_lut().code4;
                decode_mean_imputed_additive_packed_block_rows_f32(
                    payload,
                    stats.bytes_per_snp,
                    stats.n_samples_full,
                    stats.row_flip_slice(row_start, rows_here),
                    stats.row_maf_slice(row_start, rows_here),
                    sample_idx,
                    sample_identity,
                    Some(stats.row_source_slice(row_start, rows_here)),
                    0,
                    out,
                    code4_lut,
                    pool,
                )
            }
            Self::Windowed { matrix } => matrix.decode_additive_block(
                stats,
                row_start,
                out,
                sample_idx,
                sample_identity,
                pool,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cholesky::sparse_cholesky_analyze_jxgrm_csc;
    use crate::spgrm::SparseGrmCsc;
    use crate::splmm_approx::ResidualizedApproxScanModel;

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

    fn make_test_scan_prepared() -> (SplmmPreparedInput, Vec<usize>) {
        let source_rows = [
            pack_plink_codes(&[0b00, 0b10, 0b11, 0b01]),
            pack_plink_codes(&[0b10, 0b10, 0b00, 0b11]),
            pack_plink_codes(&[0b11, 0b00, 0b10, 0b00]),
        ];
        let mut packed_flat = Vec::<u8>::new();
        for row in source_rows.iter() {
            packed_flat.extend_from_slice(row);
        }
        let prepared = SplmmPreparedInput {
            bed_prefix: None,
            payload: Some(SplmmPreparedPayload::Packed(Arc::<[u8]>::from(packed_flat))),
            n_samples_full: 4,
            bytes_per_snp: 1,
            row_flip: Arc::from(vec![false, false, false]),
            row_maf: Arc::from(vec![0.375, 0.5, 0.5]),
            row_missing: Arc::from(vec![0.0, 0.25, 0.0]),
            row_source_indices: Some(Arc::from(vec![2, 0, 1])),
        };
        let scan_sample_idx = vec![3usize, 1usize, 0usize];
        (prepared, scan_sample_idx)
    }

    fn make_test_design() -> Vec<f64> {
        let x_cov = vec![0.0_f64, 1.0_f64, 2.0_f64];
        build_design_with_intercept(Some(&x_cov), 3, 1)
    }

    fn make_diag_analysis(diag: &[f64]) -> SparseJxgrmCholeskyAnalysis {
        let n = diag.len();
        let csc = SparseGrmCsc {
            n_samples: n,
            nnz: n,
            col_ptr: (0..=n).map(|v| v as u64).collect(),
            row_indices: (0..n).map(|v| v as u32).collect(),
            values: diag.to_vec(),
        };
        sparse_cholesky_analyze_jxgrm_csc(&csc).unwrap()
    }

    fn make_diag_factor(diag: &[f64]) -> SparseJxgrmCholesky {
        make_diag_analysis(diag)
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
        prepared: &SplmmPreparedInput,
        row_idx: usize,
        scan_sample_idx: &[usize],
        gm: PackedGeneticModel,
    ) -> Vec<f64> {
        let n = scan_sample_idx.len();
        let (sample_identity, sample_byte_idx, sample_bit_shift) = sample_mapping(scan_sample_idx);
        let mut out = vec![0.0_f64; n];
        decode_packed_row_model_enum_into_f64(
            prepared
                .row_bytes(row_idx)
                .expect("test prepared input should retain resident payload"),
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
        prepared: &SplmmPreparedInput,
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

    fn manual_exact_scan(
        factor: &SparseJxgrmCholesky,
        prepared: &SplmmPreparedInput,
        x_design: &[f64],
        py_vec: &[f64],
        ypy: f64,
        df: f64,
        xt_v_inv_x_chol: &[f64],
        scan_sample_idx: &[usize],
        gm: PackedGeneticModel,
    ) -> Vec<f64> {
        let n = py_vec.len();
        let p = x_design.len() / n;
        let sigma2 = ypy / df;
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
                out_row[1] = (sigma2 / denom).sqrt();
                out_row[2] = chi2_sf_df1((spy * spy) / (sigma2 * denom));
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
        row_major_snp_block_f32_to_col_major_f64(&block, 2, 3, &mut out, None);
        assert_eq!(
            out,
            vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64]
        );
    }

    #[test]
    fn pairwise_block_dot_f32_f64_matches_reference() {
        let lhs = vec![
            1.0_f32, 2.0_f32, 3.0_f32, //
            4.0_f32, 5.0_f32, 6.0_f32,
        ];
        let rhs = vec![
            0.5_f64, 1.0_f64, 1.5_f64, //
            2.0_f64, 2.5_f64, 3.0_f64,
        ];
        let mut out = vec![0.0_f64; 2];
        pairwise_block_dot_f32_f64(&lhs, &rhs, 2, 3, &mut out, None);
        assert_eq!(out, vec![7.0_f64, 39.5_f64]);
    }

    #[test]
    fn fused_block_sv_xtz_scalar_p1_matches_reference() {
        let g = vec![
            1.0_f32, 2.0_f32, 3.0_f32, //
            4.0_f32, 5.0_f32, 6.0_f32,
        ];
        let z = vec![
            0.5_f64, 1.0_f64, 1.5_f64, //
            2.0_f64, 2.5_f64, 3.0_f64,
        ];
        let x0 = vec![10.0_f64, 20.0_f64, 30.0_f64];
        let mut sv = vec![0.0_f64; 2];
        let mut xtz = vec![0.0_f64; 2];
        fused_block_sv_xtz_scalar_p1(&g, &z, &x0, 2, 3, &mut sv, &mut xtz, None);
        assert_eq!(sv, vec![7.0_f64, 39.5_f64]);
        assert_eq!(xtz, vec![70.0_f64, 160.0_f64]);
    }

    #[test]
    fn xt_mat_rhs_block_large_matches_naive_reference() {
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
            None,
        );
        let expected = vec![
            103.0_f64, 136.0_f64, //
            193.0_f64, 256.0_f64,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn xt_mat_rhs_block_matches_naive_reference() {
        let n = 128usize;
        let p = 3usize;
        let rows_here = 40usize;
        let mut x_design = vec![0.0_f64; n * p];
        for i in 0..n {
            for j in 0..p {
                x_design[i * p + j] = ((i + 1 + 3 * j) as f64) / ((j + 2) as f64);
            }
        }
        let mut z_col_major = vec![0.0_f64; n * rows_here];
        for col in 0..rows_here {
            for row in 0..n {
                z_col_major[col * n + row] = ((row + 2 * col + 1) as f64) / 17.0_f64;
            }
        }
        let mut expected = vec![0.0_f64; rows_here * p];
        for j in 0..rows_here {
            let c_j = &mut expected[j * p..(j + 1) * p];
            for i in 0..n {
                let xi = &x_design[i * p..(i + 1) * p];
                let z_ji = z_col_major[j * n + i];
                for k in 0..p {
                    c_j[k] += xi[k] * z_ji;
                }
            }
        }
        let x_col = row_major_to_col_major_f64(&x_design, n, p).unwrap();
        let mut got = vec![0.0_f64; rows_here * p];
        xt_mat_rhs_block(
            &x_design,
            Some(&x_col),
            &z_col_major,
            n,
            p,
            rows_here,
            &mut got,
            None,
        );
        for (lhs, rhs) in got.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() <= 1.0e-10_f64);
        }
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
            None,
        )
        .unwrap();

        let mut input = unified_input_from_splmm_prepared(&prepared, None).unwrap();
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
    fn grammar_scan_core_p1_matches_manual_reference() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let py_vec = vec![0.4_f64, -1.0_f64, 0.8_f64];
        let xtx_chol = xtx_chol_from_design(&x_design, py_vec.len(), 1).unwrap();
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
            None,
        )
        .unwrap();

        let mut input = unified_input_from_splmm_prepared(&prepared, None).unwrap();
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
        let x_col = row_major_to_col_major_f64(&x_design, py_vec.len(), 2).unwrap();
        let ypy = 1.0_f64;
        let df = 1.0_f64;

        let mut workspace_core = factor.make_solve_workspace(2).unwrap();
        let mut core = vec![0.0_f64; prepared.n_rows() * 3];
        let mut sink = |row_start: usize, rows_here: usize, block: &[f64]| {
            core[row_start * 3..][..rows_here * 3].copy_from_slice(block);
            Ok(())
        };
        let mut input = unified_input_from_splmm_prepared(&prepared, None).unwrap();
        exact_scan_blocks_core(
            &factor,
            &mut input,
            &x_design,
            &x_col,
            &py_vec,
            ypy,
            df,
            &xt_v_inv_x_chol,
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
            ypy,
            df,
            &xt_v_inv_x_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
        );

        assert_close(&core, &manual, 1e-5_f64);
    }

    #[test]
    fn residualized_approx_scan_matches_legacy_p0_scan_when_a_is_mx_residualized() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let n = scan_sample_idx.len();
        let p = x_design.len() / n;
        let xtx_chol = xtx_chol_from_design(&x_design, n, p).unwrap();
        let a_vec = vec![0.25_f64, -0.5_f64, 0.25_f64];
        let gamma = 0.4_f64;
        let legacy = scan_with_py_and_rhat(
            &prepared,
            &x_design,
            &a_vec,
            &xtx_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            gamma,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
            None,
        )
        .unwrap();

        let model = ResidualizedApproxScanModel::from_a_vec(&x_design, &a_vec, gamma).unwrap();
        let mut got = vec![0.0_f64; prepared.n_rows() * 3];
        for row_idx in 0..prepared.n_rows() {
            let g = decode_test_row(
                &prepared,
                row_idx,
                &scan_sample_idx,
                PackedGeneticModel::Add,
            );
            let (beta, se, pwald) = model.assoc_from_marker(&g).unwrap();
            got[row_idx * 3] = beta;
            got[row_idx * 3 + 1] = se;
            got[row_idx * 3 + 2] = pwald;
        }

        assert_close(&got, &legacy, 1e-10_f64);
    }

    #[test]
    fn p0_exact_scan_matches_full_p_scan() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let py_vec = vec![0.3_f64, -0.8_f64, 0.6_f64];
        let factor = make_diag_factor(&[2.0_f64, 3.0_f64, 4.0_f64]);
        let xt_v_inv_x_chol = compute_xt_v_inv_x_chol(&factor, &x_design, py_vec.len(), 2);
        let x_col = row_major_to_col_major_f64(&x_design, py_vec.len(), 2).unwrap();
        let sigma2 = 2.75_f64;
        let df = 1.0_f64;
        let ypy = sigma2 * df;

        let mut workspace_full = factor.make_solve_workspace(2).unwrap();
        let mut full = vec![0.0_f64; prepared.n_rows() * 3];
        let mut full_sink = |row_start: usize, rows_here: usize, block: &[f64]| {
            full[row_start * 3..][..rows_here * 3].copy_from_slice(block);
            Ok(())
        };
        let mut input = unified_input_from_splmm_prepared(&prepared, None).unwrap();
        exact_scan_blocks_core(
            &factor,
            &mut input,
            &x_design,
            &x_col,
            &py_vec,
            ypy,
            df,
            &xt_v_inv_x_chol,
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
        let p0 = scan_with_py_and_exact_p_sparse(
            &factor,
            &prepared,
            &x_design,
            &x_col,
            &py_vec,
            ypy,
            df,
            &xt_v_inv_x_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
            None,
            &mut workspace_p0,
        )
        .unwrap();

        assert_close(&full, &p0, 1e-10_f64);
    }

    #[test]
    fn exact_scan_uses_null_sigma2_instead_of_per_snp_sigma2() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let py_vec = vec![0.3_f64, -0.8_f64, 0.6_f64];
        let factor = make_diag_factor(&[2.0_f64, 3.0_f64, 4.0_f64]);
        let xt_v_inv_x_chol = compute_xt_v_inv_x_chol(&factor, &x_design, py_vec.len(), 2);
        let x_col = row_major_to_col_major_f64(&x_design, py_vec.len(), 2).unwrap();
        let sigma2_null = 2.75_f64;
        let df = 1.0_f64;
        let ypy = sigma2_null * df;

        let mut workspace = factor.make_solve_workspace(2).unwrap();
        let out = scan_with_py_and_exact_p_sparse(
            &factor,
            &prepared,
            &x_design,
            &x_col,
            &py_vec,
            ypy,
            df,
            &xt_v_inv_x_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
            None,
            &mut workspace,
        )
        .unwrap();

        let g = decode_test_row(&prepared, 0, &scan_sample_idx, PackedGeneticModel::Add);
        let z = factor.solve_vec(&g).unwrap();
        let spy = g.iter().zip(py_vec.iter()).map(|(a, b)| a * b).sum::<f64>();
        let s_v_s = g.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>();
        let mut c_j = vec![0.0_f64; 2];
        let mut alpha = vec![0.0_f64; 2];
        xt_vec_row_major(&x_design, py_vec.len(), 2, &z, &mut c_j);
        cholesky_solve_into(&xt_v_inv_x_chol, 2, &c_j, &mut alpha);
        let x_quad = c_j
            .iter()
            .zip(alpha.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let denom = (s_v_s - x_quad).max(0.0_f64);
        let sigma2_alt = (ypy - (spy * spy) / denom).max(0.0_f64) / df;
        let se_null = (sigma2_null / denom).sqrt();
        let se_alt = (sigma2_alt / denom).sqrt();

        assert!((out[1] - se_null).abs() <= 1e-10_f64);
        assert!((out[1] - se_alt).abs() > 1e-6_f64);
    }

    #[test]
    fn build_sparse_splmm_null_state_matches_manual_reference() {
        let x_design = make_test_design();
        let y_vec = vec![0.5_f64, -1.0_f64, 1.25_f64];
        let factor = make_diag_factor(&[2.0_f64, 3.0_f64, 4.0_f64]);
        let n = y_vec.len();
        let p = x_design.len() / n;

        let mut workspace = factor.make_solve_workspace(p.max(1)).unwrap();
        let built = build_sparse_splmm_null_state(&factor, &x_design, &y_vec, &mut workspace, None)
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
        let ypy_manual = y_vec
            .iter()
            .zip(py_manual.iter())
            .map(|(y, py_i)| y * py_i)
            .sum::<f64>();
        let sigma2_manual = ypy_manual / ((n - p) as f64);

        assert_close(&built.x_design_col_major, &x_col, 1e-12_f64);
        assert_close(&built.beta_hat, &beta_hat, 1e-12_f64);
        assert_close(&built.py, &py_manual, 1e-12_f64);
        assert_close(&built.null_model.py, &py_manual, 1e-12_f64);
        assert!((built.null_model.ypy - ypy_manual).abs() <= 1e-12_f64);
        assert!((built.null_model.sigma2 - sigma2_manual).abs() <= 1e-12_f64);
    }

    #[test]
    fn exact_scan_is_invariant_to_global_v_scale_when_lambda_matches() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let y_vec = vec![0.5_f64, -1.0_f64, 1.25_f64];
        let analysis = make_diag_analysis(&[1.0_f64, 2.0_f64, 4.0_f64]);
        let lambda = 0.75_f64;
        let sigma2_scale = 1.8_f64;
        let factor_lambda = analysis.factorize_k_plus_lambda_i(lambda).unwrap();
        let factor_scaled = analysis
            .factorize_sigma_g2_k_plus_sigma_e2_i(sigma2_scale, lambda * sigma2_scale)
            .unwrap();
        let n = y_vec.len();
        let p = x_design.len() / n;

        let mut workspace_lambda = factor_lambda.make_solve_workspace(p.max(1)).unwrap();
        let built_lambda = build_sparse_splmm_null_state(
            &factor_lambda,
            &x_design,
            &y_vec,
            &mut workspace_lambda,
            None,
        )
        .unwrap();
        let mut workspace_scaled = factor_scaled.make_solve_workspace(p.max(1)).unwrap();
        let built_scaled = build_sparse_splmm_null_state(
            &factor_scaled,
            &x_design,
            &y_vec,
            &mut workspace_scaled,
            None,
        )
        .unwrap();

        let mut scan_workspace_lambda = factor_lambda.make_solve_workspace(2).unwrap();
        let out_lambda = scan_with_py_and_exact_p_sparse(
            &factor_lambda,
            &prepared,
            &x_design,
            &built_lambda.x_design_col_major,
            &built_lambda.null_model.py,
            built_lambda.null_model.ypy,
            built_lambda.null_model.df,
            &built_lambda.null_model.xt_w_x_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
            None,
            &mut scan_workspace_lambda,
        )
        .unwrap();

        let mut scan_workspace_scaled = factor_scaled.make_solve_workspace(2).unwrap();
        let out_scaled = scan_with_py_and_exact_p_sparse(
            &factor_scaled,
            &prepared,
            &x_design,
            &built_scaled.x_design_col_major,
            &built_scaled.null_model.py,
            built_scaled.null_model.ypy,
            built_scaled.null_model.df,
            &built_scaled.null_model.xt_w_x_chol,
            &scan_sample_idx,
            PackedGeneticModel::Add,
            1,
            0,
            None,
            2,
            9,
            0,
            0,
            None,
            &mut scan_workspace_scaled,
        )
        .unwrap();

        assert_close(&out_lambda, &out_scaled, 1e-10_f64);
    }

    #[test]
    fn estimate_rhat_and_scan_sparse_exact_mode_supports_block_rows_zero() {
        let (prepared, scan_sample_idx) = make_test_scan_prepared();
        let x_design = make_test_design();
        let y_vec = vec![0.5_f64, -1.0_f64, 1.25_f64];
        let factor = make_diag_factor(&[2.0_f64, 3.0_f64, 4.0_f64]);

        let (r_hat, out, null_info, rhat_info, factor_nnz) = estimate_rhat_and_scan_sparse(
            factor,
            prepared,
            x_design,
            y_vec,
            scan_sample_idx,
            PackedGeneticModel::Add,
            1.0_f64,
            1,
            0,
            2,
            20260604_u64,
            None,
            None,
            0,
            SplmmScanMode::Exact,
            None,
        )
        .unwrap();
        assert!(factor_nnz > 0);

        assert!(r_hat.is_nan());
        assert_eq!(out.len(), 9);
        assert!(out.iter().all(|v| v.is_finite() || v.is_nan()));
        assert!(null_info.v_inv_y.converged);
        assert!(null_info.v_inv_x.converged_all);
        assert_eq!(rhat_info.n_markers_requested, 0);
        assert_eq!(rhat_info.n_markers_used, 0);
    }

    #[test]
    fn splmm_scan_hint_keeps_decode_block_size_independent_of_progress_step() {
        let (scan_block_hint, progress_step) =
            splmm_scan_block_hint_and_progress_step(764_773, 4_096);
        assert_eq!(scan_block_hint, 764_773);
        assert_eq!(progress_step, 4_096);

        let (scan_block_hint_zero, progress_step_zero) =
            splmm_scan_block_hint_and_progress_step(0, 0);
        assert_eq!(scan_block_hint_zero, 512);
        assert_eq!(progress_step_zero, 512);
    }
}
